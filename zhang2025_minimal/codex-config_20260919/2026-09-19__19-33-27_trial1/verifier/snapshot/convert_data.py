#!/usr/bin/env python3
"""Convert the IBL brain-wide map release to the decoder interchange format.

The preprocessing follows ``code/code_zhang2025/src/0_data_caching.py`` and its
helpers: probes from one recording session are combined, spikes are counted in
20 ms bins from 0.5 s before through 1.5 s after stimulus onset, behavior is
linearly interpolated at bin right edges, and the BWM trial exclusions are
applied.  The two continuous decoding targets are converted to session-wise
tertiles as required by this task.

The source release is large.  Each worker therefore writes one restartable
session shard.  The parent process only loads those shards once, when producing
the required final pickle.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "data" / "one_cache"
DEFAULT_RELEASE = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"
DEFAULT_OUTPUT = ROOT / "converted_data.pkl"
DEFAULT_SHARDS = ROOT / ".converted_sessions"

BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = 100
# This is exactly the grid used by the reference get_behavior_per_interval:
# behavior values correspond to the right edge of each neural count bin.
RELATIVE_TIMES = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS).astype(np.float32)
CONVERSION_VERSION = 1


def _one(cache_dir: str):
    """Build a ONE client against the staged, locally cached release."""
    from one.api import ONE, One

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        silent=True,
        cache_dir=cache_dir,
    )
    # Load the local tables through the base class.  OneAlyx.load_cache also
    # checks/downloads remote tables and rewrites these files; the base method is
    # read-only and therefore safe in concurrent workers.
    One.load_cache(one, tables_dir=Path(cache_dir) / "Brainwidemap", clobber=True)
    return one


def _trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    """Return the task's one-based trial number within each probability block."""
    out = np.empty(len(probability_left), dtype=np.float32)
    number = 0
    previous = None
    for i, value in enumerate(probability_left):
        if i == 0 or not np.isfinite(value) or value != previous:
            number = 1
        else:
            number += 1
        out[i] = number
        previous = value
    return out


def _interpolate_trials(
    times: np.ndarray, values: np.ndarray, onsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate one continuous stream using the reference coverage checks."""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).squeeze()
    good_source = np.isfinite(times)
    times, values = times[good_source], values[good_source]
    if times.ndim != 1 or values.ndim != 1 or len(times) != len(values) or len(times) < 2:
        raise ValueError("continuous behavior stream is empty or malformed")

    # np.interp requires increasing sample times.  Repeated timestamps are rare;
    # retain their last sample, matching interpolation on the ordered ALF stream.
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    if np.any(np.diff(times) <= 0):
        _, reverse_idx = np.unique(times[::-1], return_index=True)
        keep = np.sort(len(times) - 1 - reverse_idx)
        times, values = times[keep], values[keep]

    result = np.empty((len(onsets), N_BINS), dtype=np.float32)
    good = np.ones(len(onsets), dtype=bool)
    for i, onset in enumerate(onsets):
        begin, end = onset + OFF_START, onset + OFF_END
        ib = np.searchsorted(times, begin, side="right")
        ie = np.searchsorted(times, end, side="left")
        # Same tests as get_behavior_per_interval in the supplied repository.
        if ib >= ie or ib >= len(times) or ie <= 0:
            good[i] = False
            continue
        segment_t, segment_v = times[ib:ie], values[ib:ie]
        if (
            len(segment_t) < 2
            or abs(begin - segment_t[0]) > BIN_SIZE
            or abs(end - segment_t[-1]) > BIN_SIZE
        ):
            good[i] = False
            continue
        grid = onset + RELATIVE_TIMES.astype(np.float64)
        row = np.interp(grid, segment_t, segment_v)
        if not np.all(np.isfinite(row)):
            good[i] = False
            continue
        result[i] = row.astype(np.float32)
    return result, good


def _load_motion_energy(session_loader) -> tuple[np.ndarray, np.ndarray, str]:
    """Load left whisker motion energy, falling back to the right camera."""
    errors = []
    for view in ("left", "right"):
        try:
            session_loader.load_motion_energy(views=[view])
            key = f"{view}Camera"
            frame = session_loader.motion_energy[key]
            times = frame["times"].to_numpy()
            values = frame["whiskerMotionEnergy"].to_numpy()
            if len(times) >= 2 and len(times) == len(values):
                return times, values, key
        except Exception as exc:  # fallback is part of the reference pipeline
            errors.append(f"{view}: {exc}")
    raise RuntimeError("no usable whisker motion energy (" + "; ".join(errors) + ")")


def _session_tertiles(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Discretize a stream into low/middle/high within-session tertiles."""
    cuts = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    labels = np.digitize(values, cuts, right=False).astype(np.int8)
    return labels, [float(cuts[0]), float(cuts[1])]


def _bin_spikes(
    spike_parts: list[tuple[np.ndarray, np.ndarray, int]],
    onsets: np.ndarray,
    n_neurons: int,
) -> np.ndarray:
    """Count all sorted units in the requested stimulus-aligned bins."""
    flat_parts = []
    starts = onsets + OFF_START
    ends = onsets + OFF_END
    for times, clusters, offset in spike_parts:
        times = np.asarray(times)
        clusters = np.asarray(clusters, dtype=np.int64)
        # SpikeSortingLoader returns time-sorted arrays.  Work trial-by-trial so
        # overlapping windows, if any, count a spike in both trials as in the
        # reference implementation.
        for trial, (begin, end) in enumerate(zip(starts, ends)):
            lo = np.searchsorted(times, begin, side="left")
            hi = np.searchsorted(times, end, side="left")
            if hi <= lo:
                continue
            bins = np.floor((times[lo:hi] - begin) / BIN_SIZE).astype(np.int64)
            clu = clusters[lo:hi] + offset
            valid = (bins >= 0) & (bins < N_BINS) & (clu >= 0) & (clu < n_neurons)
            if np.any(valid):
                flat_parts.append(((trial * n_neurons + clu[valid]) * N_BINS + bins[valid]))

    size = len(onsets) * n_neurons * N_BINS
    if flat_parts:
        flat = np.concatenate(flat_parts)
        counts = np.bincount(flat, minlength=size)
        del flat, flat_parts
        return counts.reshape(len(onsets), n_neurons, N_BINS).astype(np.float32)
    return np.zeros((len(onsets), n_neurons, N_BINS), dtype=np.float32)


def _process_session(job: dict) -> dict:
    """Process one session and atomically save its shard."""
    # Keep the worker logs useful: ONE emits a revision warning for nearly every
    # load in this frozen cache even though the selected revisions are valid.
    warnings.filterwarnings("ignore", message="Multiple revisions")
    warnings.filterwarnings("ignore", message="No default revision")

    eid = job["eid"]
    shard = Path(job["shard"])
    if shard.exists():
        try:
            with shard.open("rb") as stream:
                cached = pickle.load(stream)
            if cached.get("conversion_version") == CONVERSION_VERSION:
                return {"eid": eid, "status": "cached", "shard": str(shard)}
        except Exception:
            pass

    started = time.time()
    try:
        from brainbox.io.one import SessionLoader, SpikeSortingLoader
        from iblatlas.regions import BrainRegions

        one = _one(job["cache_dir"])
        loader = SessionLoader(one=one, eid=eid)
        loader.load_trials()
        trials = loader.trials.copy()

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
            raise ValueError(f"missing trial columns: {missing}")

        probability = trials["probabilityLeft"].to_numpy(dtype=float)
        trial_in_block = _trial_number_in_block(probability)
        finite = np.all(np.isfinite(trials[required].to_numpy(dtype=float)), axis=1)
        reaction_time = (
            trials["firstMovement_times"].to_numpy(dtype=float)
            - trials["stimOn_times"].to_numpy(dtype=float)
        )
        trial_length = (
            trials["feedback_times"].to_numpy(dtype=float)
            - trials["goCue_times"].to_numpy(dtype=float)
        )
        choice_raw = trials["choice"].to_numpy(dtype=float)
        mask = (
            finite
            & (reaction_time >= 0.08)
            & (reaction_time <= 2.0)
            & (trial_length <= 10.0)
            & (choice_raw != 0)
            & np.isin(probability, [0.2, 0.5, 0.8])
        )
        candidate = np.flatnonzero(mask)
        if len(candidate) < 2:
            raise ValueError("fewer than two trials pass the BWM trial criteria")

        loader.load_wheel()
        wheel_times = loader.wheel["times"].to_numpy()
        wheel_speed = np.abs(loader.wheel["velocity"].to_numpy())
        motion_times, motion_values, motion_source = _load_motion_energy(loader)

        candidate_onsets = trials["stimOn_times"].to_numpy(dtype=float)[candidate]
        wheel, wheel_good = _interpolate_trials(wheel_times, wheel_speed, candidate_onsets)
        motion, motion_good = _interpolate_trials(motion_times, motion_values, candidate_onsets)
        behavior_good = wheel_good & motion_good
        candidate = candidate[behavior_good]
        wheel, motion = wheel[behavior_good], motion[behavior_good]
        if len(candidate) < 2:
            raise ValueError("fewer than two trials have complete wheel and whisker streams")

        onsets = trials["stimOn_times"].to_numpy(dtype=float)[candidate]

        # Load and retain every Kilosort cluster, exactly as prepare_data(qc=None)
        # in the methods repository.  Probe-local IDs are shifted on merge.
        spike_parts = []
        region_parts = []
        offset = 0
        brain_regions = BrainRegions()
        for pid, probe_name in zip(job["pids"], job["probe_names"]):
            spike_loader = SpikeSortingLoader(
                pid=pid, one=one, eid=eid, pname=probe_name
            )
            spikes, clusters, channels = spike_loader.load_spike_sorting()
            merged = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
            n_clusters = len(merged)
            local_ids = np.asarray(spikes["clusters"], dtype=np.int64)
            if len(local_ids) and (local_ids.min() < 0 or local_ids.max() >= n_clusters):
                raise ValueError(f"non-contiguous cluster IDs for {probe_name}")
            mapped = brain_regions.acronym2acronym(
                merged["acronym"].to_numpy(), mapping="Beryl"
            ).astype(str)
            spike_parts.append((np.asarray(spikes["times"]), local_ids, offset))
            region_parts.append(mapped)
            offset += n_clusters

        if offset == 0:
            raise ValueError("session has no neural units")
        neural_array = _bin_spikes(spike_parts, onsets, offset)
        regions = np.concatenate(region_parts)

        wheel_cat, wheel_cuts = _session_tertiles(wheel)
        motion_cat, motion_cuts = _session_tertiles(motion)
        choice = (choice_raw[candidate] == 1).astype(np.int8)  # left=-1, right=1
        prior_lookup = {0.2: 0, 0.5: 1, 0.8: 2}
        prior = np.array([prior_lookup[float(x)] for x in probability[candidate]], dtype=np.int8)

        neural = [neural_array[i] for i in range(len(candidate))]
        input_data = []
        output_data = []
        for row, trial_index in enumerate(candidate):
            inp = np.empty((2, N_BINS), dtype=np.float32)
            inp[0] = RELATIVE_TIMES
            inp[1] = trial_in_block[trial_index]
            out = np.empty((4, N_BINS), dtype=np.int8)
            out[0] = choice[row]
            out[1] = prior[row]
            out[2] = wheel_cat[row]
            out[3] = motion_cat[row]
            input_data.append(inp)
            output_data.append(out)

        payload = {
            "conversion_version": CONVERSION_VERSION,
            "eid": eid,
            "subject": job["subject"],
            "date": job["date"],
            "lab": job["lab"],
            "neural": neural,
            "input": input_data,
            "output": output_data,
            "regions": regions,
            "info": {
                "eid": eid,
                "subject": job["subject"],
                "date": job["date"],
                "lab": job["lab"],
                "probe_names": list(job["probe_names"]),
                "n_source_trials": int(len(trials)),
                "n_trials": int(len(candidate)),
                "n_neurons": int(offset),
                "motion_energy_source": motion_source,
                "wheel_speed_tertile_edges": wheel_cuts,
                "whisker_motion_energy_tertile_edges": motion_cuts,
            },
        }
        shard.parent.mkdir(parents=True, exist_ok=True)
        temporary = shard.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, shard)
        return {
            "eid": eid,
            "status": "ok",
            "shard": str(shard),
            "trials": len(candidate),
            "neurons": offset,
            "seconds": round(time.time() - started, 1),
        }
    except Exception as exc:
        return {
            "eid": eid,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "seconds": round(time.time() - started, 1),
        }


def _read_subset(path: Path) -> set[str] | None:
    """Read the optional data-limit manifest installed by stage_cache.sh."""
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    for name in ("eid", "session", "session_id"):
        if name in frame:
            return set(frame[name].astype(str))
    return set(frame.iloc[:, 0].astype(str))


def _jobs(release_file: Path, cache_dir: Path, shard_dir: Path) -> list[dict]:
    release = pd.read_csv(release_file, index_col=0)
    subset = _read_subset(ROOT / "data" / "DATALIMIT_SUBSET.csv")
    if subset is not None:
        release = release[release["eid"].astype(str).isin(subset)]

    jobs = []
    for eid, rows in release.groupby("eid", sort=False):
        first = rows.iloc[0]
        jobs.append(
            {
                "eid": str(eid),
                "subject": str(first["subject"]),
                "date": str(first["date"]),
                "lab": str(first["lab"]),
                "pids": rows["pid"].astype(str).tolist(),
                "probe_names": rows["probe_name"].astype(str).tolist(),
                "cache_dir": str(cache_dir),
                "shard": str(shard_dir / f"{eid}.pkl"),
            }
        )
    return jobs


def _assemble(jobs: list[dict], output_file: Path) -> dict:
    sessions = []
    for job in jobs:
        shard = Path(job["shard"])
        if not shard.exists():
            continue
        with shard.open("rb") as stream:
            payload = pickle.load(stream)
        if len(payload["neural"]) >= 2:
            sessions.append(payload)
    if not sessions:
        raise RuntimeError("no sessions were converted successfully")

    subjects = sorted({session["subject"] for session in sessions})
    subject_map = {name: i for i, name in enumerate(subjects)}
    brain_regions = sorted({region for session in sessions for region in session["regions"]})
    region_map = {name: i for i, name in enumerate(brain_regions)}

    data = {
        "neural": [session["neural"] for session in sessions],
        "input": [session["input"] for session in sessions],
        "output": [session["output"] for session in sessions],
        "subjects": subjects,
        "subject_idx": np.array(
            [subject_map[session["subject"]] for session in sessions], dtype=np.int64
        ),
        "brain_regions": brain_regions,
        "brain_region_idx": [
            np.array([region_map[x] for x in session["regions"]], dtype=np.int64)
            for session in sessions
        ],
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
                "IBL visual two-alternative forced-choice task; decode choice, block "
                "prior, wheel speed tertile, and whisker motion-energy tertile."
            ),
            "time_bin_size": BIN_SIZE * 1000.0,
            "temporal_alignment_event": "visual stimulus onset (stimOn_times)",
            "off_start": OFF_START,
            "off_end": OFF_END,
            "neural_measure": "Kilosort spike counts in non-overlapping 20 ms bins",
            "neuron_filter": "all sorted clusters (qc=None), matching the methods code",
            "trial_filter": (
                "finite required BWM events, choice!=0, first movement latency "
                "0.08-2.00 s, go-cue-to-feedback <=10 s, and complete behavior streams"
            ),
            "continuous_output_discretization": (
                "within-session tertiles computed over all retained trial-time samples"
            ),
            "behavior_sampling": "linear interpolation at neural-bin right edges",
            "trial_number_in_block": "one-based count, computed before trial exclusions",
            "source_release": "IBL brain-wide map release listed in bwm_release.csv",
            "session_info": [session["info"] for session in sessions],
        },
    }

    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, output_file)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--release-file", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=None, help="development-only session limit")
    parser.add_argument("--keep-shards", action="store_true")
    args = parser.parse_args()

    jobs = _jobs(args.release_file, args.cache_dir, args.shard_dir)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    args.shard_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting {len(jobs)} sessions with {args.workers} workers", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_session, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if result["status"] == "failed":
                print(
                    f"[{done}/{len(jobs)}] FAILED {result['eid']}: {result['error']}",
                    flush=True,
                )
            else:
                detail = (
                    f"{result.get('trials', '-')} trials, {result.get('neurons', '-')} neurons"
                )
                print(
                    f"[{done}/{len(jobs)}] {result['status']} {result['eid']} "
                    f"({detail}, {result.get('seconds', 0)} s)",
                    flush=True,
                )

    failures = [result for result in results if result["status"] == "failed"]
    failure_file = args.shard_dir / "failures.json"
    failure_file.write_text(json.dumps(failures, indent=2))
    print(f"Assembling successful sessions; {len(failures)} sessions skipped", flush=True)
    data = _assemble(jobs, args.output)
    total_trials = sum(map(len, data["neural"]))
    total_neurons = sum(len(x) for x in data["brain_region_idx"])
    print(
        f"Wrote {args.output}: {len(data['neural'])} sessions, "
        f"{total_trials} trials, {total_neurons} neurons",
        flush=True,
    )

    if not args.keep_shards:
        # This directory is created exclusively by this script and every target is
        # an explicit file under it; the final dataset has already been fsynced and
        # atomically renamed above.
        shutil.rmtree(args.shard_dir)


if __name__ == "__main__":
    main()
