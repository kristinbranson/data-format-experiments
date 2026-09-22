#!/usr/bin/env python3
"""Convert the IBL Brain-Wide Map release to the decoder pickle format.

The implementation follows the preprocessing in Zhang et al. (2025): all
Kilosort 2.5 clusters from all probes in a session are combined, spikes are
counted in non-overlapping 20 ms bins, wheel velocity is computed from the
1 kHz interpolated/low-pass-filtered wheel trace, and left whisker-pad motion
energy is preferred with the right camera used as a fallback.  The requested
common window is [-0.5, 1.5) s around stimulus onset.

The full output is large.  Each converted session is therefore checkpointed
before the final dictionary is assembled.  Re-running the script resumes from
valid checkpoints.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pickle
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from brainbox.behavior.wheel import interpolate_position, velocity_filtered
from iblatlas.regions import BrainRegions
from one.api import ONE, One
from scipy.interpolate import interp1d


ROOT = Path("/app")
CACHE_DIR = ROOT / "data" / "one_cache"
BWM_TABLE = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"
SUBSET_TABLE = ROOT / "data" / "DATALIMIT_SUBSET.csv"
DEFAULT_OUTPUT = ROOT / "converted_data.pkl"
CHECKPOINT_DIR = ROOT / ".converted_sessions"

OFF_START = -0.5
OFF_END = 1.5
BIN_SIZE = 0.02
N_BINS = 100
CONVERTER_VERSION = 1


@dataclass(frozen=True)
class SessionSpec:
    eid: str
    subject: str
    session_path: Path
    probes: tuple[str, ...]


def _latest(paths) -> Path | None:
    """Return the newest dated revision, with an un-revised file as fallback."""
    candidates = [Path(p) for p in paths if Path(p).is_file()]
    if not candidates:
        return None

    def key(path: Path):
        revisions = re.findall(r"#(\d{4}-\d{2}-\d{2})#", str(path))
        return (revisions[-1] if revisions else "0000-00-00", str(path))

    return max(candidates, key=key)


def _required_file(paths, description: str) -> Path:
    path = _latest(paths)
    if path is None:
        raise FileNotFoundError(description)
    return path


def _load_release_specs(max_sessions: int | None = None) -> list[SessionSpec]:
    release = pd.read_csv(BWM_TABLE, index_col=0)

    if SUBSET_TABLE.exists():
        subset = pd.read_csv(SUBSET_TABLE)
        if "eid" in subset.columns:
            selected = set(subset["eid"].astype(str))
        else:
            uuid_pattern = re.compile(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                re.I,
            )
            selected = {
                str(value)
                for value in subset.to_numpy().ravel()
                if uuid_pattern.match(str(value))
            }
        if not selected:
            raise ValueError(f"Could not find session eids in {SUBSET_TABLE}")
        release = release[release["eid"].astype(str).isin(selected)]

    # Load the local release table without querying Alyx.  The cache deliberately
    # contains no top-level tables because multiple tagged releases are present.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        one = ONE(mode="local", cache_dir=CACHE_DIR, silent=True)
    One.load_cache(one, CACHE_DIR / "Brainwidemap")

    specs: list[SessionSpec] = []
    for eid, rows in release.groupby("eid", sort=False):
        session_path = Path(one.eid2path(str(eid)))
        specs.append(
            SessionSpec(
                eid=str(eid),
                subject=str(rows.iloc[0]["subject"]),
                session_path=session_path,
                probes=tuple(rows["probe_name"].astype(str)),
            )
        )
    if max_sessions is not None:
        specs = specs[:max_sessions]
    return specs


def _load_trials(session_path: Path) -> pd.DataFrame:
    table = _required_file(
        session_path.glob("alf/**/_ibl_trials.table.pqt"), "trials table"
    )
    trials = pd.read_parquet(table)
    required = [
        "stimOn_times",
        "choice",
        "feedback_times",
        "probabilityLeft",
        "firstMovement_times",
        "feedbackType",
        "goCue_times",
    ]
    missing = sorted(set(required) - set(trials.columns))
    if missing:
        raise ValueError(f"trials table lacks columns: {missing}")
    return trials


def _paper_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    """Trial exclusions from the BWM paper and the supplied repository."""
    needed = [
        "stimOn_times",
        "choice",
        "feedback_times",
        "probabilityLeft",
        "firstMovement_times",
        "feedbackType",
    ]
    reaction_time = trials["firstMovement_times"] - trials["stimOn_times"]
    trial_duration = trials["feedback_times"] - trials["goCue_times"]
    return (
        trials[needed].notna().all(axis=1).to_numpy()
        & (trials["choice"].to_numpy() != 0)
        & (reaction_time.to_numpy() >= 0.08)
        & (reaction_time.to_numpy() <= 2.0)
        & (trial_duration.to_numpy() <= 10.0)
    )


def _interpolate_trials(
    sample_times: np.ndarray, sample_values: np.ndarray, stimulus_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Match repository endpoint interpolation and interval validity checks."""
    sample_times = np.asarray(sample_times)
    sample_values = np.asarray(sample_values)
    begins = stimulus_times + OFF_START
    ends = stimulus_times + OFF_END
    idx_beg = np.searchsorted(sample_times, begins, side="right")
    idx_end = np.searchsorted(sample_times, ends, side="left")
    values = np.full((len(stimulus_times), N_BINS), np.nan, dtype=np.float32)
    good = np.zeros(len(stimulus_times), dtype=bool)

    for trial, (beg_idx, end_idx) in enumerate(zip(idx_beg, idx_end)):
        if end_idx <= beg_idx:
            continue
        times = sample_times[beg_idx:end_idx]
        vals = sample_values[beg_idx:end_idx]
        if abs(begins[trial] - times[0]) > BIN_SIZE:
            continue
        if abs(ends[trial] - times[-1]) > BIN_SIZE:
            continue
        target_times = np.linspace(
            begins[trial] + BIN_SIZE, ends[trial], N_BINS, dtype=np.float64
        )
        values[trial] = interp1d(
            times, vals, kind="linear", fill_value="extrapolate"
        )(target_times)
        good[trial] = np.isfinite(values[trial]).all()
    return values, good


def _load_wheel_speed(
    session_path: Path, stimulus_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    position_path = _required_file(
        session_path.glob("alf/**/_ibl_wheel.position.npy"), "wheel position"
    )
    timestamps_path = _required_file(
        session_path.glob("alf/**/_ibl_wheel.timestamps.npy"), "wheel timestamps"
    )
    position = np.load(position_path)
    timestamps = np.load(timestamps_path)
    if len(position) != len(timestamps):
        raise ValueError("wheel position/timestamp length mismatch")
    position_1khz, times_1khz = interpolate_position(timestamps, position, freq=1000)
    velocity, _ = velocity_filtered(
        position_1khz, fs=1000, corner_frequency=20, order=8
    )
    return _interpolate_trials(times_1khz, np.abs(velocity), stimulus_times)


def _load_whisker_energy(
    session_path: Path, stimulus_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str]:
    errors: list[str] = []
    for view in ("left", "right"):
        try:
            energy_path = _required_file(
                session_path.glob(f"alf/**/{view}Camera.ROIMotionEnergy.npy"),
                f"{view} whisker motion energy",
            )
            times_path = _required_file(
                session_path.glob(f"alf/**/_ibl_{view}Camera.times.npy"),
                f"{view} camera timestamps",
            )
            energy = np.load(energy_path)
            times = np.load(times_path)
            # This is SessionLoader._check_video_timestamps: older camera sessions
            # may contain timestamps for a few initial frames that were not saved.
            if len(times) < len(energy):
                raise ValueError("camera timestamps are shorter than motion energy")
            if len(times) > len(energy):
                times = times[-len(energy) :]
            values, good = _interpolate_trials(times, energy, stimulus_times)
            return values, good, view
        except Exception as exc:  # left-to-right fallback in the reference code
            errors.append(f"{view}: {exc}")
    raise RuntimeError("; ".join(errors))


def _trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    """Zero-based trial position in each uninterrupted probability block."""
    out = np.zeros(len(probability_left), dtype=np.float32)
    for trial in range(1, len(probability_left)):
        out[trial] = (
            out[trial - 1] + 1
            if probability_left[trial] == probability_left[trial - 1]
            else 0
        )
    return out


def _three_bins(values: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Discretize a session's samples into low/middle/high tertiles."""
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    labels = np.digitize(values, [q1, q2], right=False).astype(np.int64)
    return labels, (float(q1), float(q2))


def _spike_files(session_path: Path, probe: str) -> tuple[Path, Path, Path, Path]:
    base = session_path / "alf" / probe / "pykilosort"
    return (
        _required_file(base.glob("**/spikes.times.npy"), f"{probe} spike times"),
        _required_file(base.glob("**/spikes.clusters.npy"), f"{probe} spike clusters"),
        _required_file(base.glob("**/clusters.channels.npy"), f"{probe} cluster channels"),
        _required_file(
            base.glob("**/channels.brainLocationIds_ccf_2017.npy"),
            f"{probe} channel atlas ids",
        ),
    )


def _bin_probe(
    times_path: Path,
    clusters_path: Path,
    interval_begins: np.ndarray,
    n_cluster_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin one probe exactly as bincount2D, without rescanning the full spike train."""
    times = np.load(times_path, mmap_mode="r")
    clusters = np.load(clusters_path, mmap_mode="r")
    if len(times) != len(clusters):
        raise ValueError("spike time/cluster length mismatch")

    # The reference utility only returns cluster IDs represented in the spike
    # train.  Ordinarily every cluster row is represented, but preserve its
    # behavior for the rare missing ID.
    used = np.unique(clusters)
    if used.size == n_cluster_rows and np.array_equal(
        used, np.arange(n_cluster_rows, dtype=used.dtype)
    ):
        remap = None
    else:
        remap = np.full(max(n_cluster_rows, int(used.max()) + 1), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))

    binned = np.zeros((len(interval_begins), len(used), N_BINS), dtype=np.float32)
    interval_ends = interval_begins + (OFF_END - OFF_START)
    start_idx = np.searchsorted(times, interval_begins, side="left")
    end_idx = np.searchsorted(times, interval_ends, side="left")
    for trial, (lo, hi) in enumerate(zip(start_idx, end_idx)):
        if hi <= lo:
            continue
        spike_bins = np.floor(
            (np.asarray(times[lo:hi]) - interval_begins[trial]) / BIN_SIZE
        ).astype(np.int64)
        cluster_ids = np.asarray(clusters[lo:hi], dtype=np.int64)
        if remap is not None:
            cluster_ids = remap[cluster_ids]
        valid = (
            (cluster_ids >= 0)
            & (cluster_ids < len(used))
            & (spike_bins >= 0)
            & (spike_bins < N_BINS)
        )
        flat = cluster_ids[valid] * N_BINS + spike_bins[valid]
        binned[trial].flat[:] = np.bincount(
            flat, minlength=len(used) * N_BINS
        ).astype(np.float32, copy=False)
    return binned, used.astype(np.int64, copy=False)


def _convert_session(spec: SessionSpec, checkpoint_path: Path) -> dict:
    started = time.time()
    trials = _load_trials(spec.session_path)
    stimulus_times = trials["stimOn_times"].to_numpy(dtype=np.float64)
    paper_mask = _paper_trial_mask(trials)

    wheel, wheel_good = _load_wheel_speed(spec.session_path, stimulus_times)
    motion, motion_good, camera_view = _load_whisker_energy(
        spec.session_path, stimulus_times
    )
    keep = paper_mask & wheel_good & motion_good
    keep_idx = np.flatnonzero(keep)
    if len(keep_idx) < 2:
        raise ValueError(f"only {len(keep_idx)} aligned valid trials")

    wheel = wheel[keep]
    motion = motion[keep]
    wheel_labels, wheel_edges = _three_bins(wheel)
    motion_labels, motion_edges = _three_bins(motion)

    probability_left = trials["probabilityLeft"].to_numpy(dtype=np.float64)
    prior_lookup = {0.2: 0, 0.5: 1, 0.8: 2}
    prior = np.full(len(trials), -1, dtype=np.int64)
    for value, label in prior_lookup.items():
        prior[np.isclose(probability_left, value)] = label
    if np.any(prior[keep] < 0):
        raise ValueError("probabilityLeft contains a value outside {0.2, 0.5, 0.8}")

    raw_choice = trials["choice"].to_numpy()
    # In IBL choice coding, -1 is the rightward response and +1 is leftward.
    choice = (raw_choice == -1).astype(np.int64)
    trial_in_block = _trial_number_in_block(probability_left)
    relative_time = (
        OFF_START + np.arange(N_BINS, dtype=np.float32) * BIN_SIZE
    ).astype(np.float32)

    inputs = np.empty((len(keep_idx), 2, N_BINS), dtype=np.float32)
    inputs[:, 0, :] = relative_time
    inputs[:, 1, :] = trial_in_block[keep, None]

    outputs = np.empty((len(keep_idx), 4, N_BINS), dtype=np.int64)
    outputs[:, 0, :] = choice[keep, None]
    outputs[:, 1, :] = prior[keep, None]
    outputs[:, 2, :] = wheel_labels
    outputs[:, 3, :] = motion_labels

    probe_bins: list[np.ndarray] = []
    probe_regions: list[np.ndarray] = []
    atlas = BrainRegions()
    interval_begins = stimulus_times[keep] + OFF_START
    for probe in spec.probes:
        times_path, clusters_path, cluster_channels_path, atlas_ids_path = _spike_files(
            spec.session_path, probe
        )
        cluster_channels = np.load(cluster_channels_path).astype(np.int64, copy=False)
        channel_atlas_ids = np.load(atlas_ids_path)
        counts, used = _bin_probe(
            times_path, clusters_path, interval_begins, len(cluster_channels)
        )
        cluster_atlas_ids = channel_atlas_ids[cluster_channels[used]]
        regions = atlas.id2acronym(cluster_atlas_ids, mapping="Beryl")
        probe_bins.append(counts)
        probe_regions.append(np.asarray(regions, dtype=str))

    neural = np.concatenate(probe_bins, axis=1)
    regions = np.concatenate(probe_regions)
    if neural.shape[1] != len(regions):
        raise AssertionError("neural/region neuron count mismatch")

    payload = {
        "version": CONVERTER_VERSION,
        "eid": spec.eid,
        "subject": spec.subject,
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "regions": regions,
        "session_info": {
            "eid": spec.eid,
            "subject": spec.subject,
            "probe_names": list(spec.probes),
            "camera_view": camera_view,
            "source_trial_count": int(len(trials)),
            "paper_eligible_trial_count": int(paper_mask.sum()),
            "retained_trial_count": int(len(keep_idx)),
            "excluded_for_stream_alignment_or_nonfinite": int(
                (paper_mask & ~(wheel_good & motion_good)).sum()
            ),
            "wheel_speed_tertile_edges": list(wheel_edges),
            "whisker_motion_energy_tertile_edges": list(motion_edges),
            "n_neurons": int(neural.shape[1]),
            "elapsed_seconds": float(time.time() - started),
        },
    }

    tmp_path = checkpoint_path.with_suffix(".tmp")
    with open(tmp_path, "wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, checkpoint_path)
    return payload["session_info"]


def _valid_checkpoint(path: Path, eid: str) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, "rb") as stream:
            payload = pickle.load(stream)
        return payload.get("version") == CONVERTER_VERSION and payload.get("eid") == eid
    except Exception:
        return False


def _assemble(
    specs: list[SessionSpec], checkpoint_paths: dict[str, Path], output_path: Path
) -> dict:
    payloads = []
    for index, spec in enumerate(specs, 1):
        path = checkpoint_paths.get(spec.eid)
        if path is None or not path.exists():
            continue
        with open(path, "rb") as stream:
            payloads.append(pickle.load(stream))
        if index % 25 == 0 or index == len(specs):
            print(f"Assembled {index}/{len(specs)} candidate sessions", flush=True)

    if not payloads:
        raise RuntimeError("No sessions were converted")

    brain_regions = sorted(
        {str(region) for payload in payloads for region in payload["regions"]}
    )
    region_to_idx = {region: i for i, region in enumerate(brain_regions)}
    subjects = sorted({payload["subject"] for payload in payloads})
    subject_to_idx = {subject: i for i, subject in enumerate(subjects)}

    neural = [list(payload["neural"]) for payload in payloads]
    inputs = [list(payload["input"]) for payload in payloads]
    outputs = [list(payload["output"]) for payload in payloads]
    brain_region_idx = [
        np.fromiter(
            (region_to_idx[str(region)] for region in payload["regions"]),
            dtype=np.int32,
            count=len(payload["regions"]),
        )
        for payload in payloads
    ]

    data = {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": np.asarray(
            [subject_to_idx[payload["subject"]] for payload in payloads],
            dtype=np.int32,
        ),
        "brain_regions": brain_regions,
        "brain_region_idx": brain_region_idx,
        "input_names": ["time_since_stimulus_onset", "trial_number_in_block"],
        "output_names": [
            "choice",
            "prior_probability_of_left",
            "wheel_speed",
            "whisker_motion_energy",
        ],
        "output_values": [
            ["left", "right"],
            ["0.2", "0.5", "0.8"],
            ["low", "medium", "high"],
            ["low", "medium", "high"],
        ],
        "metadata": {
            "task_description": (
                "IBL visual two-alternative forced-choice task; decode left/right "
                "choice, block prior, wheel-speed tertile, and whisker-motion-energy "
                "tertile from stimulus-aligned population spike counts."
            ),
            "time_bin_size": 20.0,
            "temporal_alignment_event": "visual stimulus onset (trials.stimOn_times)",
            "off_start": OFF_START,
            "off_end": OFF_END,
            "neural_measurement": "Kilosort 2.5 spike counts in non-overlapping bins",
            "neuron_filter": "all Kilosort clusters, matching the methods-paper decoder pipeline",
            "trial_filter": (
                "Required task events present; non-zero choice; first movement 0.08-2.0 s "
                "after stimulus; go-cue-to-feedback duration <=10 s; wheel and camera "
                "windows aligned and finite."
            ),
            "continuous_output_discretization": (
                "Within-session tertiles pooled over all retained trials and time bins"
            ),
            "trial_number_in_block_convention": (
                "zero-based run-length within the full unfiltered probabilityLeft sequence"
            ),
            "time_coordinate_convention": "left edge of each 20 ms neural bin",
            "source_release_session_count": int(len(specs)),
            "retained_session_count": int(len(payloads)),
            "session_info": [payload["session_info"] for payload in payloads],
        },
    }

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, output_path)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="rebuild session checkpoints")
    parser.add_argument(
        "--keep-session-cache",
        action="store_true",
        help="keep large resumable session checkpoints after final assembly",
    )
    args = parser.parse_args()

    specs = _load_release_specs(args.max_sessions)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = {
        spec.eid: CHECKPOINT_DIR / f"{spec.eid}.pkl" for spec in specs
    }
    print(
        f"Converting {len(specs)} release sessions with {args.workers} workers",
        flush=True,
    )

    failures: dict[str, str] = {}
    pending = []
    for spec in specs:
        checkpoint = checkpoint_paths[spec.eid]
        if args.force or not _valid_checkpoint(checkpoint, spec.eid):
            pending.append(spec)

    completed = len(specs) - len(pending)
    if completed:
        print(f"Resuming from {completed} valid session checkpoints", flush=True)

    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_spec = {
            pool.submit(_convert_session, spec, checkpoint_paths[spec.eid]): spec
            for spec in pending
        }
        for future in concurrent.futures.as_completed(future_to_spec):
            spec = future_to_spec[future]
            completed += 1
            try:
                info = future.result()
                print(
                    f"[{completed}/{len(specs)}] {spec.eid}: "
                    f"{info['retained_trial_count']} trials, {info['n_neurons']} neurons",
                    flush=True,
                )
            except Exception as exc:
                failures[spec.eid] = repr(exc)
                print(
                    f"[{completed}/{len(specs)}] SKIP {spec.eid}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    data = _assemble(specs, checkpoint_paths, args.output)
    failure_path = args.output.with_suffix(".failures.json")
    with open(failure_path, "w") as stream:
        json.dump(failures, stream, indent=2, sort_keys=True)

    size_gib = args.output.stat().st_size / (1024**3)
    print(
        f"Wrote {args.output} ({size_gib:.2f} GiB): "
        f"{len(data['neural'])} sessions, "
        f"{sum(map(len, data['neural']))} trials in "
        f"{(time.time() - started) / 60:.1f} min",
        flush=True,
    )
    if failures:
        print(f"Skipped {len(failures)} sessions; details: {failure_path}")

    if not args.keep_session_cache:
        for path in checkpoint_paths.values():
            path.unlink(missing_ok=True)
        try:
            CHECKPOINT_DIR.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
