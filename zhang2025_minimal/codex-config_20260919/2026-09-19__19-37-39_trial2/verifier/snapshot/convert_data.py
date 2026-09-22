#!/usr/bin/env python3
"""Convert the IBL Brain-Wide Map release to the decoder pickle format.

The conversion follows Zhang et al.'s released preprocessing code where the
four requested targets can share one representation: trials are aligned to
stimulus onset, span [-0.5, 1.5) s, and use non-overlapping 20 ms spike-count
bins.  Static targets are repeated over time so that they can coexist with the
two dynamic targets in one output array.

The script is deliberately usable with both the full mounted release and the
small grading cache.  If /app/data/DATALIMIT_SUBSET.csv exists, only EIDs in
that file are processed.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from iblatlas.regions import BrainRegions
from one.api import ONE
from scipy.interpolate import interp1d


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data" / "one_cache"
RELEASE_CSV = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"
SUBSET_CSV = ROOT / "data" / "DATALIMIT_SUBSET.csv"
DEFAULT_OUTPUT = ROOT / "converted_data.pkl"

BIN_SIZE_S = 0.020
OFF_START_S = -0.5
OFF_END_S = 1.5
N_TIME = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))
TIME_FROM_STIMULUS = (
    OFF_START_S + np.arange(N_TIME, dtype=np.float32) * BIN_SIZE_S
)


def _ordered_unique(values):
    """Return unique values without sorting or relying on set order."""
    return list(dict.fromkeys(values))


def load_release_table(max_sessions: int | None = None) -> pd.DataFrame:
    """Load the paper's frozen insertion list and optionally restrict EIDs."""
    release = pd.read_csv(RELEASE_CSV)
    if "eid" not in release or "pid" not in release:
        raise ValueError(f"Unexpected release table columns: {release.columns.tolist()}")

    if SUBSET_CSV.exists():
        subset = pd.read_csv(SUBSET_CSV)
        if "eid" in subset:
            allowed = set(subset["eid"].dropna().astype(str))
        else:
            # Datalimit files may be a one-column, headerless EID list.
            allowed = set(subset.iloc[:, 0].dropna().astype(str))
            first_header = str(subset.columns[0])
            if len(first_header) == 36 and first_header.count("-") == 4:
                allowed.add(first_header)
        release = release[release["eid"].astype(str).isin(allowed)]

    eids = _ordered_unique(release["eid"].astype(str))
    if max_sessions is not None:
        eids = eids[:max_sessions]
    return release[release["eid"].astype(str).isin(eids)].copy()


def trial_numbers_in_block(probability_left: np.ndarray) -> np.ndarray:
    """One-indexed trial ordinal within each contiguous probability block."""
    out = np.ones(len(probability_left), dtype=np.float32)
    for i in range(1, len(out)):
        if probability_left[i] == probability_left[i - 1]:
            out[i] = out[i - 1] + 1.0
    return out


def valid_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    """Implement the repository's load_trials_and_mask defaults.

    In addition to the default required fields and 0.08--2.00 s reaction-time
    interval, prepare_data passes max_trial_len=10 s.  No-choice trials are
    excluded.  The initial unbiased block is retained.
    """
    required = [
        "stimOn_times",
        "choice",
        "feedback_times",
        "probabilityLeft",
        "firstMovement_times",
        "feedbackType",
        "goCue_times",
    ]
    missing = [name for name in required if name not in trials]
    if missing:
        raise ValueError(f"trial table is missing columns {missing}")

    vals = trials[required].to_numpy(dtype=float)
    mask = np.all(np.isfinite(vals), axis=1)
    reaction_time = (
        trials["firstMovement_times"].to_numpy(dtype=float)
        - trials["stimOn_times"].to_numpy(dtype=float)
    )
    duration = (
        trials["feedback_times"].to_numpy(dtype=float)
        - trials["goCue_times"].to_numpy(dtype=float)
    )
    mask &= reaction_time >= 0.08
    mask &= reaction_time <= 2.0
    mask &= duration <= 10.0
    mask &= trials["choice"].to_numpy(dtype=float) != 0
    return mask


def interpolate_trials(
    sample_times: np.ndarray,
    sample_values: np.ndarray,
    stimulus_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match get_behavior_per_interval's endpoint sampling and coverage test."""
    sample_times = np.asarray(sample_times, dtype=float)
    sample_values = np.asarray(sample_values, dtype=float).squeeze()
    if sample_times.ndim != 1 or sample_values.ndim != 1:
        raise ValueError("behavior times and values must be one-dimensional")
    if len(sample_times) != len(sample_values) or len(sample_times) < 2:
        raise ValueError("behavior times/values have incompatible lengths")

    # Repository behavior bins are sampled at each bin's right edge.
    relative_endpoints = OFF_START_S + BIN_SIZE_S * np.arange(1, N_TIME + 1)
    result = np.full((len(stimulus_times), N_TIME), np.nan, dtype=np.float32)
    good = np.zeros(len(stimulus_times), dtype=bool)
    for i, stimulus_time in enumerate(stimulus_times):
        begin = stimulus_time + OFF_START_S
        end = stimulus_time + OFF_END_S
        ib = np.searchsorted(sample_times, begin, side="right")
        ie = np.searchsorted(sample_times, end, side="left")
        if ie - ib < 2:
            continue
        local_t = sample_times[ib:ie]
        local_v = sample_values[ib:ie]
        if abs(begin - local_t[0]) > BIN_SIZE_S:
            continue
        if abs(end - local_t[-1]) > BIN_SIZE_S:
            continue
        query = stimulus_time + relative_endpoints
        # interp1d with extrapolation is the exact operation used by the
        # repository.  The final query is the interval end, typically just
        # beyond the final camera/wheel sample selected with side="left".
        interp = interp1d(
            local_t, local_v, kind="linear", fill_value="extrapolate"
        )(query)
        if not np.all(np.isfinite(interp)):
            continue
        result[i] = interp.astype(np.float32)
        good[i] = True
    return result, good


def load_behaviors(
    one: ONE, eid: str, stimulus_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load repository-defined wheel speed and whisker-pad motion energy."""
    loader = SessionLoader(one=one, eid=eid)
    loader.load_wheel()
    wheel_times = loader.wheel["times"].to_numpy(dtype=float)
    wheel_speed = np.abs(loader.wheel["velocity"].to_numpy(dtype=float))
    wheel, wheel_good = interpolate_trials(wheel_times, wheel_speed, stimulus_times)

    last_error = None
    for view in ("left", "right"):
        try:
            loader.load_motion_energy(views=[view])
            key = f"{view}Camera"
            motion_df = loader.motion_energy[key]
            motion, motion_good = interpolate_trials(
                motion_df["times"].to_numpy(dtype=float),
                motion_df["whiskerMotionEnergy"].to_numpy(dtype=float),
                stimulus_times,
            )
            # The reference uses left whenever it can be loaded, with right as
            # a fallback.  A wholly unusable left trace is treated as failure.
            if np.any(motion_good):
                return wheel, motion, wheel_good & motion_good, view
        except Exception as exc:  # fallback is part of the reference loader
            last_error = exc
    raise ValueError(f"no usable whisker motion-energy trace ({last_error})")


def discretize_tertiles(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Convert a session's continuous behavior into low/middle/high tertiles."""
    thresholds = np.quantile(values.reshape(-1), [1.0 / 3.0, 2.0 / 3.0])
    if not np.all(np.isfinite(thresholds)):
        raise ValueError("non-finite tertile thresholds")
    # np.digitize creates exactly the required ordered labels 0, 1, 2.
    labels = np.digitize(values, thresholds, right=False).astype(np.int8)
    return labels, [float(thresholds[0]), float(thresholds[1])]


def bin_probe_spikes(
    one: ONE,
    eid: str,
    pid: str,
    probe_name: str,
    stimulus_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one Kilosort probe and count every sorted cluster in 20 ms bins."""
    loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=probe_name)
    spikes, clusters, channels = loader.load_spike_sorting()
    cluster_table = SpikeSortingLoader.merge_clusters(
        spikes, clusters, channels, compute_metrics=False
    ).to_df()

    spike_times = np.asarray(spikes["times"], dtype=float)
    spike_clusters = np.asarray(spikes["clusters"], dtype=np.int64)
    cluster_ids = np.unique(spike_clusters)
    if not len(cluster_ids):
        raise ValueError(f"probe {probe_name} has no spikes")
    if cluster_ids.min() < 0 or cluster_ids.max() >= len(cluster_table):
        raise ValueError(f"probe {probe_name} cluster indices do not match metadata")

    # Searchsorted maps arbitrary (though usually consecutive) Kilosort IDs to
    # rows in the output.  All sorted clusters are retained; no QC label filter.
    counts = np.zeros(
        (len(stimulus_times), len(cluster_ids), N_TIME), dtype=np.uint16
    )
    for trial_i, stimulus_time in enumerate(stimulus_times):
        begin = stimulus_time + OFF_START_S
        end = stimulus_time + OFF_END_S
        ib = np.searchsorted(spike_times, begin, side="left")
        ie = np.searchsorted(spike_times, end, side="left")
        if ie <= ib:
            continue
        local_time = spike_times[ib:ie]
        bin_idx = np.floor((local_time - begin) / BIN_SIZE_S).astype(np.int64)
        local_cluster = np.searchsorted(cluster_ids, spike_clusters[ib:ie])
        in_range = (bin_idx >= 0) & (bin_idx < N_TIME)
        flat = local_cluster[in_range] * N_TIME + bin_idx[in_range]
        hist = np.bincount(flat, minlength=len(cluster_ids) * N_TIME)
        if hist.max(initial=0) > np.iinfo(np.uint16).max:
            raise OverflowError("20 ms spike count exceeds uint16 capacity")
        counts[trial_i] = hist.reshape(len(cluster_ids), N_TIME)

    regions = cluster_table.iloc[cluster_ids]["acronym"].fillna("void").astype(str)
    beryl = BrainRegions().acronym2acronym(regions.to_numpy(), mapping="Beryl")
    return counts, np.asarray(beryl, dtype=str)


def make_session(
    one: ONE, eid_rows: pd.DataFrame
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, dict]:
    """Convert one behavioral session and return trial arrays plus metadata."""
    eid = str(eid_rows.iloc[0]["eid"])
    session_loader = SessionLoader(one=one, eid=eid)
    session_loader.load_trials()
    trials = session_loader.trials.copy()
    if len(trials) < 2:
        raise ValueError("fewer than two source trials")

    block_number = trial_numbers_in_block(
        trials["probabilityLeft"].to_numpy(dtype=float)
    )
    source_mask = valid_trial_mask(trials)
    source_indices = np.flatnonzero(source_mask)
    if len(source_indices) < 2:
        raise ValueError("fewer than two trials pass the paper's trial mask")

    stimulus_times = trials["stimOn_times"].to_numpy(dtype=float)[source_indices]
    wheel, motion, behavior_good, motion_view = load_behaviors(
        one, eid, stimulus_times
    )
    source_indices = source_indices[behavior_good]
    stimulus_times = stimulus_times[behavior_good]
    wheel = wheel[behavior_good]
    motion = motion[behavior_good]
    if len(source_indices) < 2:
        raise ValueError("fewer than two trials have complete behavior coverage")

    wheel_class, wheel_thresholds = discretize_tertiles(wheel)
    motion_class, motion_thresholds = discretize_tertiles(motion)

    probe_counts = []
    probe_regions = []
    for row in eid_rows.itertuples(index=False):
        counts, regions = bin_probe_spikes(
            one, eid, str(row.pid), str(row.probe_name), stimulus_times
        )
        probe_counts.append(counts)
        probe_regions.append(regions)
    neural_3d = np.concatenate(probe_counts, axis=1).astype(np.float32)
    region_labels = np.concatenate(probe_regions)
    if neural_3d.shape[1] == 0:
        raise ValueError("session has no sorted clusters")

    choice_raw = trials["choice"].to_numpy(dtype=float)[source_indices]
    if not np.all(np.isin(choice_raw, [-1.0, 1.0])):
        raise ValueError("choice contains values other than -1 and +1")
    choice = (choice_raw == 1.0).astype(np.int8)

    prior_raw = trials["probabilityLeft"].to_numpy(dtype=float)[source_indices]
    prior = np.full(len(prior_raw), -1, dtype=np.int8)
    for value, label in ((0.2, 0), (0.5, 1), (0.8, 2)):
        prior[np.isclose(prior_raw, value)] = label
    if np.any(prior < 0):
        raise ValueError(f"unexpected probabilityLeft values {np.unique(prior_raw)}")

    trial_in_block = block_number[source_indices]
    neural = []
    decoder_input = []
    decoder_output = []
    for i in range(len(source_indices)):
        neural.append(np.ascontiguousarray(neural_3d[i]))
        decoder_input.append(
            np.vstack(
                [TIME_FROM_STIMULUS, np.full(N_TIME, trial_in_block[i], np.float32)]
            ).astype(np.float32, copy=False)
        )
        decoder_output.append(
            np.vstack(
                [
                    np.full(N_TIME, choice[i], np.int8),
                    np.full(N_TIME, prior[i], np.int8),
                    wheel_class[i],
                    motion_class[i],
                ]
            )
        )

    info = {
        "eid": eid,
        "subject": str(eid_rows.iloc[0]["subject"]),
        "date": str(eid_rows.iloc[0]["date"]),
        "session_number": int(eid_rows.iloc[0]["session_number"]),
        "probe_names": [str(x) for x in eid_rows["probe_name"]],
        "source_trial_count": int(len(trials)),
        "retained_trial_count": int(len(source_indices)),
        "retained_trial_indices": source_indices.astype(int).tolist(),
        "motion_energy_camera": motion_view,
        "wheel_speed_tertile_thresholds": wheel_thresholds,
        "whisker_motion_energy_tertile_thresholds": motion_thresholds,
        "n_neurons": int(neural_3d.shape[1]),
    }
    return neural, decoder_input, decoder_output, region_labels, info


def convert(output_path: Path, max_sessions: int | None = None) -> dict:
    release = load_release_table(max_sessions=max_sessions)
    eids = _ordered_unique(release["eid"].astype(str))
    if not eids:
        raise RuntimeError("No sessions selected from the release table")

    # Omitting the password uses the staged Alyx token and cached REST replies.
    # This is needed because revised ALF objects are not all represented in the
    # old static Brainwidemap parquet table.
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        silent=True,
        cache_dir=str(CACHE_DIR),
    )

    neural_all = []
    input_all = []
    output_all = []
    region_labels_all = []
    session_info = []
    skipped_sessions = []
    start = time.time()

    for number, eid in enumerate(eids, start=1):
        tic = time.time()
        try:
            rows = release[release["eid"].astype(str) == eid]
            neural, decoder_input, decoder_output, regions, info = make_session(
                one, rows
            )
            neural_all.append(neural)
            input_all.append(decoder_input)
            output_all.append(decoder_output)
            region_labels_all.append(regions)
            session_info.append(info)
            print(
                f"[{number}/{len(eids)}] {eid}: {len(neural)} trials, "
                f"{len(regions)} neurons ({time.time() - tic:.1f}s)",
                flush=True,
            )
        except Exception as exc:
            skipped_sessions.append({"eid": eid, "reason": repr(exc)})
            print(
                f"[{number}/{len(eids)}] SKIP {eid}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    if not neural_all:
        reasons = json.dumps(skipped_sessions[:5], indent=2)
        raise RuntimeError(f"No sessions were converted. First failures:\n{reasons}")

    subjects = sorted({info["subject"] for info in session_info})
    subject_lookup = {name: i for i, name in enumerate(subjects)}
    subject_idx = np.asarray(
        [subject_lookup[info["subject"]] for info in session_info], dtype=np.int64
    )

    brain_regions = sorted(
        {str(region) for labels in region_labels_all for region in labels}
    )
    region_lookup = {name: i for i, name in enumerate(brain_regions)}
    brain_region_idx = [
        np.asarray([region_lookup[str(x)] for x in labels], dtype=np.int64)
        for labels in region_labels_all
    ]

    data = {
        "neural": neural_all,
        "input": input_all,
        "output": output_all,
        "subjects": subjects,
        "subject_idx": subject_idx,
        "brain_regions": brain_regions,
        "brain_region_idx": brain_region_idx,
        "input_names": ["time since stimulus onset", "trial number in block"],
        "output_names": [
            "choice",
            "prior probability of left",
            "wheel speed",
            "whisker motion energy",
        ],
        "output_values": [
            ["left", "right"],
            ["0.2", "0.5", "0.8"],
            ["low", "medium", "high"],
            ["low", "medium", "high"],
        ],
        "metadata": {
            "task_description": (
                "IBL visual two-alternative forced-choice task; decode choice, "
                "block probability, wheel speed, and whisker-pad motion energy."
            ),
            "time_bin_size": BIN_SIZE_S * 1000.0,
            "time_bin_size_units": "ms",
            "temporal_alignment_event": "visual stimulus onset (stimOn_times)",
            "off_start": OFF_START_S,
            "off_end": OFF_END_S,
            "neural_measurement": "Kilosort 2.5 spike counts in non-overlapping bins",
            "neuron_filter": "all sorted clusters (no cluster-QC threshold)",
            "trial_filter": (
                "finite required events, nonzero choice, first movement latency "
                "0.08-2.00 s, trial duration <=10 s, and complete dynamic behavior"
            ),
            "dynamic_output_discretization": "within-session tertiles over retained trial-time bins",
            "trial_number_indexing": "one-indexed within contiguous probabilityLeft blocks",
            "release_file": str(RELEASE_CSV),
            "selected_session_count": len(eids),
            "converted_session_count": len(neural_all),
            "skipped_sessions": skipped_sessions,
            "session_info": session_info,
            "conversion_seconds": float(time.time() - start),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output_path)
    print(
        f"Wrote {output_path} with {len(neural_all)} sessions and "
        f"{sum(map(len, neural_all))} trials.",
        flush=True,
    )
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Process only the first N sessions (for smoke testing).",
    )
    args = parser.parse_args()
    with warnings.catch_warnings():
        # Multiple-revision ALF warnings are expected in this frozen cache; ONE
        # still selects the newest revision, which is the desired behavior.
        warnings.filterwarnings("ignore", message="Multiple revisions:.*")
        convert(args.output.resolve(), max_sessions=args.max_sessions)


if __name__ == "__main__":
    main()
