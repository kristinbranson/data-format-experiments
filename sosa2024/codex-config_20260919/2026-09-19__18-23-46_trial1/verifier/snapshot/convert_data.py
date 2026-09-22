#!/usr/bin/env python3
"""Convert the supplied Sosa et al. NWB files to decoder-ready trial data."""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
import time
from collections import Counter
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = Path("/app/data")
BEHAVIOR = "processing/behavior/BehavioralTimeSeries"
OPHYS = "processing/ophys"
ZONE_BOUNDS = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
ZONE_TO_INT = {"A": 0, "B": 1, "C": 2}
EXPECTED_DT_S = 1.0 / 15.5078125


def natural_key(path: str | Path) -> tuple[int, int]:
    """Sort files by numeric mouse and session identifiers."""
    match = re.search(r"sub-m(\d+)_ses-(\d+)", str(path))
    if match is None:
        raise ValueError(f"Cannot parse subject/session from {path}")
    return int(match.group(1)), int(match.group(2))


def discover_files(sample: bool) -> list[Path]:
    files = [Path(p) for p in glob.glob(str(DATA_ROOT / "sub-m*" / "*.nwb"))]
    files.sort(key=natural_key)
    if not files:
        raise FileNotFoundError(f"No NWB files found below {DATA_ROOT}")
    if sample:
        # Two switch sessions spanning all A/B/C labels.
        wanted = {(11, 3), (12, 3)}
        files = [p for p in files if natural_key(p) in wanted]
        if len(files) != 2:
            raise RuntimeError(f"Expected sample sessions {sorted(wanted)}, found {files}")
    return files


def read_series(nwb: h5py.File, name: str) -> tuple[np.ndarray, np.ndarray]:
    group = nwb[f"{BEHAVIOR}/{name}"]
    return group["data"][()], group["timestamps"][()]


def nearest_timestamp_indices(reference: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Map sparse event times to nearest synchronized behavior frame."""
    if events.size == 0:
        return np.empty(0, dtype=np.int64)
    right = np.searchsorted(reference, events)
    right = np.clip(right, 0, reference.size - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(reference[left] - events) < np.abs(reference[right] - events)
    out = np.where(choose_left, left, right).astype(np.int64)
    tolerance = np.median(np.diff(reference)) / 2.0 + 1e-9
    if np.any(np.abs(reference[out] - events) > tolerance):
        raise ValueError("A sparse reward timestamp cannot be aligned to behavior frames")
    return out


def classify_observed_zone(values: np.ndarray) -> str | None:
    """Label a trial from positions carrying the zone-entry signal."""
    if values.size == 0:
        return None
    median_position = float(np.median(values))
    centers = {label: np.mean(bounds) for label, bounds in ZONE_BOUNDS.items()}
    return min(centers, key=lambda label: abs(median_position - centers[label]))


def infer_zone_labels(
    position: np.ndarray,
    reward_zone: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
) -> tuple[list[str], list[str | None], int]:
    """Recover per-trial A/B/C labels using the known trial-30 switch structure."""
    observed = [
        classify_observed_zone(position[s:e][reward_zone[s:e] > 0])
        for s, e in zip(starts, stops)
    ]
    labels: list[str | None] = [None] * len(starts)
    for lo, hi in ((0, min(30, len(starts))), (min(30, len(starts)), len(starts))):
        segment_observed = [observed[i] for i in range(lo, hi) if observed[i] is not None]
        if not segment_observed:
            raise ValueError(f"No observed reward-zone entries in trial segment [{lo}, {hi})")
        label, _ = Counter(segment_observed).most_common(1)[0]
        labels[lo:hi] = [label] * (hi - lo)
    contradictions = sum(obs is not None and obs != labels[i] for i, obs in enumerate(observed))
    if contradictions:
        raise ValueError(f"Found {contradictions} zone observations inconsistent with inferred schedule")
    return [str(label) for label in labels], observed, contradictions


def distance_to_interval(position: np.ndarray, start: float, end: float) -> np.ndarray:
    distance = np.zeros(position.shape, dtype=np.float32)
    before = position < start
    after = position > end
    distance[before] = position[before] - start
    distance[after] = position[after] - end
    return distance


def discretize_distance(distance: np.ndarray) -> np.ndarray:
    out = np.full(distance.shape, 3, dtype=np.int64)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance <= -10)] = 1
    out[(distance > -10) & (distance < 0)] = 2
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
    return out


def discretize_position(position: np.ndarray) -> np.ndarray:
    out = np.zeros(position.shape, dtype=np.int64)
    out[(position >= 90) & (position < 180)] = 1
    out[(position >= 180) & (position < 270)] = 2
    out[(position >= 270) & (position <= 360)] = 3
    out[position > 360] = 4
    return out


def discretize_speed(speed: np.ndarray) -> np.ndarray:
    out = np.zeros(speed.shape, dtype=np.int64)
    out[(speed >= 2) & (speed <= 10)] = 1
    out[(speed > 10) & (speed <= 20)] = 2
    out[(speed > 20) & (speed <= 40)] = 3
    out[speed > 40] = 4
    return out


def load_curated_events(
    nwb: h5py.File, n_behavior_frames: int
) -> tuple[np.ndarray, int, list[dict[str, int]]]:
    """Pool manually curated cells from every imaging plane on the time axis."""
    segmentation = nwb[f"{OPHYS}/ImageSegmentation/PlaneSegmentation"]
    iscell = segmentation["iscell"][:, 0] > 0.5
    plane_idx = segmentation["planeIdx"][()].astype(np.int64)
    plane_groups = nwb[f"{OPHYS}/Deconvolved"]
    plane_names = sorted(plane_groups, key=lambda x: int(x.replace("plane", "")))

    curated_total = int(np.sum(iscell))
    pooled = np.empty((n_behavior_frames, curated_total), dtype=np.float32)
    cursor = 0
    plane_info: list[dict[str, int]] = []
    for plane_name in plane_names:
        plane = int(plane_name.replace("plane", ""))
        local_mask = iscell[plane_idx == plane]
        dataset = plane_groups[f"{plane_name}/data"]
        if dataset.shape[1] != local_mask.size:
            raise ValueError(
                f"{plane_name}: {dataset.shape[1]} event ROIs != {local_mask.size} segmentation ROIs"
            )
        if dataset.shape[0] < n_behavior_frames:
            raise ValueError(f"{plane_name}: neural stream is shorter than behavior")
        # Reading a dense plane once is substantially faster than thousands of HDF5
        # point selections; only curated columns survive beyond this loop.
        dense = dataset[:n_behavior_frames, :]
        selected = np.asarray(dense[:, local_mask], dtype=np.float32)
        n_selected = selected.shape[1]
        pooled[:, cursor : cursor + n_selected] = selected
        plane_info.append(
            {"plane": plane, "raw_rois": int(local_mask.size), "curated_cells": n_selected}
        )
        cursor += n_selected
        del dense, selected
    if cursor != curated_total or not np.isfinite(pooled).all():
        raise ValueError("Curated event matrix failed shape/finite validation")
    return pooled, int(iscell.size), plane_info


def plot_processing(
    session_id: str,
    out_path: Path,
    neural_by_time: np.ndarray,
    timestamps: np.ndarray,
    position: np.ndarray,
    speed: np.ndarray,
    lick: np.ndarray,
    environment: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    zone_labels: list[str],
    outcomes: np.ndarray,
    keep: np.ndarray,
    n_raw_rois: int,
) -> None:
    """Show each transformation and alignment for one representative kept trial."""
    trial = int(np.flatnonzero(keep)[min(5, np.sum(keep) - 1)])
    s, e = int(starts[trial]), int(stops[trial])
    t = timestamps[s:e] - timestamps[s]
    pos = position[s:e]
    spd = speed[s:e]
    binary_lick = (lick[s:e] > 0).astype(int)
    label = zone_labels[trial]
    z0, z1 = ZONE_BOUNDS[label]
    distance = distance_to_interval(pos, z0, z1)

    fig, axes = plt.subplots(8, 1, figsize=(16, 22), constrained_layout=True)
    axes[0].plot(timestamps, position, lw=0.5, color="0.35")
    axes[0].scatter(timestamps[starts], position[starts], s=8, label="trial starts")
    axes[0].scatter(timestamps[stops], position[stops], s=8, label="teleports")
    axes[0].set(title=f"{session_id}: paired trial boundaries", ylabel="position (cm)")
    axes[0].legend(ncol=2)

    axes[1].bar(["raw ROIs", "curated cells"], [n_raw_rois, neural_by_time.shape[1]])
    axes[1].set(title="Manual Suite2p iscell curation", ylabel="count")

    axes[2].imshow(
        neural_by_time[s:e, : min(100, neural_by_time.shape[1])].T,
        aspect="auto",
        interpolation="none",
        extent=[t[0], t[-1], min(100, neural_by_time.shape[1]), 0],
    )
    axes[2].set(title="Curated deconvolved activity (native time bins)", ylabel="cell")

    axes[3].plot(t, pos, label="continuous position")
    axes[3].step(t, discretize_position(pos) * 90, where="mid", label="position class ×90")
    axes[3].axhspan(z0, z1, alpha=0.2, color="tab:green", label=f"zone {label}")
    axes[3].set(title="Absolute-position discretization and active reward zone", ylabel="cm")
    axes[3].legend(ncol=3)

    axes[4].plot(t, distance, label="signed distance to interval")
    axes[4].step(t, discretize_distance(distance) * 10, where="mid", label="distance class ×10")
    for boundary in (-50, -10, 0, 10, 50):
        axes[4].axhline(boundary, color="0.7", lw=0.6)
    axes[4].set(title="Reward-zone distance transform and classes", ylabel="cm / class")
    axes[4].legend()

    axes[5].plot(t, spd, label="continuous speed")
    axes[5].step(t, discretize_speed(spd) * 10, where="mid", label="speed class ×10")
    for boundary in (2, 10, 20, 40):
        axes[5].axhline(boundary, color="0.7", lw=0.6)
    axes[5].set(title="Speed discretization", ylabel="cm/s")
    axes[5].legend()

    axes[6].plot(t, lick[s:e], label="cumulative within-frame count")
    axes[6].step(t, binary_lick, where="mid", label="binary lick")
    axes[6].set(title="Lick binarization after artifact-trial filtering", ylabel="lick")
    axes[6].legend()

    axes[7].plot(t, t, label="time from trial start")
    axes[7].plot(t, np.full_like(t, environment[s]), label=f"environment={int(environment[s])}")
    axes[7].plot(t, np.full_like(t, trial), label=f"trial={trial}")
    previous = int(outcomes[trial - 1]) if trial else 0
    axes[7].plot(t, np.full_like(t, previous), label=f"previous outcome={previous}")
    axes[7].set(title="Decoder inputs on the aligned trial", xlabel="time from start (s)")
    axes[7].legend(ncol=4)

    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def convert_session(path: Path, show_processing: bool) -> tuple[list, list, list, dict, dict]:
    started = time.perf_counter()
    subject_number, session_number = natural_key(path)
    session_id = f"m{subject_number}_ses-{session_number:02d}"
    with h5py.File(path, "r") as nwb:
        position, timestamps = read_series(nwb, "position")
        speed, speed_time = read_series(nwb, "speed")
        lick, lick_time = read_series(nwb, "lick")
        environment, env_time = read_series(nwb, "environment")
        reward_zone, rz_time = read_series(nwb, "reward_zone")
        trial_start, start_time = read_series(nwb, "trial_start")
        teleport, stop_time = read_series(nwb, "teleport")
        aligned_times = (speed_time, lick_time, env_time, rz_time, start_time, stop_time)
        if any(not np.allclose(timestamps, other) for other in aligned_times):
            raise ValueError(f"{session_id}: behavior timestamps are not aligned")
        dt = np.diff(timestamps)
        if not np.allclose(dt, EXPECTED_DT_S, rtol=1e-6, atol=1e-9):
            raise ValueError(f"{session_id}: nonuniform or unexpected sample interval")

        starts = np.flatnonzero(trial_start > 0).astype(np.int64)
        stops = np.flatnonzero(teleport > 0).astype(np.int64)
        if starts.size != stops.size or np.any(stops <= starts):
            raise ValueError(f"{session_id}: invalid trial start/teleport pairs")

        reward_times = nwb[f"{BEHAVIOR}/Reward/timestamps"][()]
        reward_indices = nearest_timestamp_indices(timestamps, reward_times)
        outcomes = np.array(
            [np.any((reward_indices >= s) & (reward_indices < e)) for s, e in zip(starts, stops)],
            dtype=np.int64,
        )
        zone_labels, observed_zones, contradictions = infer_zone_labels(
            position, reward_zone, starts, stops
        )
        lick_artifact_fraction = np.array(
            [np.mean(lick[s:e] > 2) for s, e in zip(starts, stops)], dtype=np.float64
        )
        keep = lick_artifact_fraction <= 0.30

        # Every trial must have exactly one valid environment value.
        trial_environments = []
        for s, e in zip(starts, stops):
            valid = np.unique(environment[s:e][environment[s:e] >= 0])
            if valid.size != 1 or valid[0] not in (0, 1):
                raise ValueError(f"{session_id}: trial lacks one binary environment label")
            trial_environments.append(int(valid[0]))

        neural_by_time, n_raw_rois, plane_info = load_curated_events(nwb, len(timestamps))

        if show_processing:
            plot_processing(
                session_id,
                Path(f"/app/processing_{session_id}.png"),
                neural_by_time,
                timestamps,
                position,
                speed,
                lick,
                environment,
                starts,
                stops,
                zone_labels,
                outcomes,
                keep,
                n_raw_rois,
            )

        neural_trials: list[np.ndarray] = []
        input_trials: list[np.ndarray] = []
        output_trials: list[np.ndarray] = []
        for trial, (s, e) in enumerate(zip(starts, stops)):
            if not keep[trial]:
                continue
            n_time = int(e - s)
            label = zone_labels[trial]
            z0, z1 = ZONE_BOUNDS[label]
            trial_position = np.asarray(position[s:e], dtype=np.float32)
            trial_speed = np.asarray(speed[s:e], dtype=np.float32)
            distance = distance_to_interval(trial_position, z0, z1)
            previous_outcome = int(outcomes[trial - 1]) if trial > 0 else 0

            neural = np.ascontiguousarray(neural_by_time[s:e, :].T, dtype=np.float32)
            inputs = np.vstack(
                (
                    np.asarray(timestamps[s:e] - timestamps[s], dtype=np.float32),
                    np.full(n_time, trial_environments[trial], dtype=np.float32),
                    np.full(n_time, trial, dtype=np.float32),
                    np.full(n_time, previous_outcome, dtype=np.float32),
                )
            )
            outputs = np.vstack(
                (
                    discretize_distance(distance),
                    discretize_position(trial_position),
                    discretize_speed(trial_speed),
                    (lick[s:e] > 0).astype(np.int64),
                    np.full(n_time, ZONE_TO_INT[label], dtype=np.int64),
                    np.full(n_time, outcomes[trial], dtype=np.int64),
                )
            )
            if neural.shape[1] != n_time or inputs.shape != (4, n_time) or outputs.shape != (6, n_time):
                raise ValueError(f"{session_id} trial {trial}: inconsistent converted shapes")
            if not (np.isfinite(neural).all() and np.isfinite(inputs).all()):
                raise ValueError(f"{session_id} trial {trial}: nonfinite neural/input values")
            neural_trials.append(neural)
            input_trials.append(inputs)
            output_trials.append(outputs)

        if len(neural_trials) < 2:
            raise ValueError(f"{session_id}: fewer than two valid trials")

        info = {
            "session_id": session_id,
            "source_file": str(path.relative_to(DATA_ROOT)),
            "subject": f"m{subject_number}",
            "session_number": session_number,
            "native_trials": int(starts.size),
            "retained_trials": len(neural_trials),
            "lick_artifact_trials_removed": int(np.sum(~keep)),
            "raw_rois": n_raw_rois,
            "curated_cells": int(neural_by_time.shape[1]),
            "planes": plane_info,
            "native_behavior_frames": int(len(timestamps)),
            "reward_events": int(len(reward_times)),
            "rewarded_trials": int(np.sum(outcomes)),
            "zone_schedule": [zone_labels[0], zone_labels[min(30, len(zone_labels) - 1)]],
            "zone_observation_contradictions": contradictions,
            "environment": int(trial_environments[0]),
            "time_bin_ms": float(np.median(dt) * 1000.0),
        }
        stats = {
            "native_trials": int(starts.size),
            "retained_trials": len(neural_trials),
            "removed_trials": int(np.sum(~keep)),
            "curated_cells": int(neural_by_time.shape[1]),
            "raw_rois": n_raw_rois,
            "rewarded_native_trials": int(np.sum(outcomes)),
            "retained_timepoints": int(sum(x.shape[1] for x in neural_trials)),
            "zone_counts": Counter(zone_labels[i] for i in range(len(zone_labels)) if keep[i]),
            "environment_counts": Counter(trial_environments[i] for i in range(len(starts)) if keep[i]),
        }

    elapsed = time.perf_counter() - started
    print(
        f"[{session_id}] {len(neural_trials)}/{len(starts)} trials, "
        f"{neural_trials[0].shape[0]}/{n_raw_rois} curated cells, {elapsed:.2f} s",
        flush=True,
    )
    return neural_trials, input_trials, output_trials, info, stats


def build_dataset(files: list[Path], show_processing: bool) -> dict:
    subjects = sorted({f"m{natural_key(p)[0]}" for p in files}, key=lambda x: int(x[1:]))
    subject_lookup = {subject: i for i, subject in enumerate(subjects)}
    neural: list[list[np.ndarray]] = []
    inputs: list[list[np.ndarray]] = []
    outputs: list[list[np.ndarray]] = []
    subject_idx: list[int] = []
    brain_region_idx: list[np.ndarray] = []
    session_info: list[dict] = []
    all_stats: list[dict] = []

    started = time.perf_counter()
    for index, path in enumerate(files):
        converted = convert_session(path, show_processing and index < 2)
        session_neural, session_input, session_output, info, stats = converted
        neural.append(session_neural)
        inputs.append(session_input)
        outputs.append(session_output)
        subject_idx.append(subject_lookup[info["subject"]])
        brain_region_idx.append(np.zeros(session_neural[0].shape[0], dtype=np.int64))
        session_info.append(info)
        all_stats.append(stats)

    total_elapsed = time.perf_counter() - started
    total_native = sum(x["native_trials"] for x in all_stats)
    total_retained = sum(x["retained_trials"] for x in all_stats)
    total_removed = sum(x["removed_trials"] for x in all_stats)
    total_cells = sum(x["curated_cells"] for x in all_stats)
    total_raw = sum(x["raw_rois"] for x in all_stats)
    total_time = sum(x["retained_timepoints"] for x in all_stats)
    total_rewarded = sum(x["rewarded_native_trials"] for x in all_stats)
    zones = sum((x["zone_counts"] for x in all_stats), Counter())
    envs = sum((x["environment_counts"] for x in all_stats), Counter())
    print(
        f"Converted {len(files)} sessions in {total_elapsed:.2f} s: "
        f"{total_retained}/{total_native} trials, {total_cells}/{total_raw} cells, "
        f"{total_time} trial timepoints",
        flush=True,
    )
    print(
        f"QC: removed lick-artifact trials={total_removed}; native reward rate="
        f"{total_rewarded / total_native:.6f}; zones={dict(zones)}; environments={dict(envs)}",
        flush=True,
    )

    return {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": ["CA1"],
        "brain_region_idx": brain_region_idx,
        "input_names": [
            "time from trial start (s)",
            "environment type",
            "trial number",
            "previous trial outcome",
        ],
        "output_names": [
            "distance to reward zone",
            "absolute position",
            "speed",
            "lick",
            "reward zone location",
            "reward outcome",
        ],
        "output_values": [
            ["< -50 cm", "-50 to -10 cm", "-10 to <0 cm", "inside zone (0 cm)", ">0 to +10 cm", "+10 to +50 cm", "> +50 cm"],
            ["<90 cm", "90 to <180 cm", "180 to <270 cm", "270 to 360 cm", ">360 cm"],
            ["<2 cm/s", "2 to 10 cm/s", ">10 to 20 cm/s", ">20 to 40 cm/s", ">40 cm/s"],
            ["no lick", "lick"],
            ["A", "B", "C"],
            ["omitted", "rewarded"],
        ],
        "metadata": {
            "task_description": "Decode position relative to the hidden reward zone, absolute position, speed, licking, active reward-zone identity, and reward outcome from CA1 calcium events during a 450-cm virtual-navigation task.",
            "time_bin_size": EXPECTED_DT_S * 1000.0,
            "temporal_alignment_event": "start of trial (entry onto the 450-cm virtual track)",
            "off_start": 0.0,
            "off_end": None,
            "source_dataset": "DANDI 001361, Sosa, Plitt & Giocomo (2025)",
            "neural_measure": "OASIS-deconvolved calcium events from manually curated Suite2p cells; all planes pooled",
            "trial_interval": "zero-based NWB frames [trial_start, teleport); teleport/ITI excluded",
            "trial_filter": "Removed trials where >30% of on-track frames had cumulative lick count >2 (paper lick-sensor artifact criterion)",
            "reward_zone_bounds_cm": ZONE_BOUNDS,
            "distance_definition": "signed distance to the nearest point in the active zone interval; zero everywhere inside the interval",
            "first_trial_previous_outcome": 0,
            "session_info": session_info,
            "conversion_summary": {
                "sessions": len(files),
                "native_trials": total_native,
                "retained_trials": total_retained,
                "lick_artifact_trials_removed": total_removed,
                "curated_cell_session_entries": total_cells,
                "raw_roi_session_entries": total_raw,
                "retained_timepoints": total_time,
                "native_rewarded_trials": total_rewarded,
                "native_reward_rate": total_rewarded / total_native,
                "retained_zone_trial_counts": dict(zones),
                "retained_environment_trial_counts": {str(k): v for k, v in envs.items()},
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="process all sessions (default)")
    mode.add_argument("--sample", action="store_true", help="process two representative sessions")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="save processing_<session_id>.png for up to two sessions",
    )
    args = parser.parse_args()

    files = discover_files(sample=args.sample)
    print(f"Mode={'sample' if args.sample else 'full'}; sessions={len(files)}", flush=True)
    data = build_dataset(files, show_processing=args.show_processing)
    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    write_started = time.perf_counter()
    with args.outpicklefile.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"Wrote {args.outpicklefile} ({args.outpicklefile.stat().st_size / 2**30:.3f} GiB) "
        f"in {time.perf_counter() - write_started:.2f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
