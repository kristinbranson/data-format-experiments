#!/usr/bin/env python3
"""Convert the Lee et al. CA1 geometry dataset to decoder format.

Usage:
    python -u /app/convert_data.py OUT.pkl [--full | --sample]
                                      [--show-processing]

Each source animal-day is a session. Continuous recordings are divided from
frame zero into complete one-minute trials. The reference position decoder's
three-frame Gaussian smoothing / average pooling is retained, producing 100-ms
decoder bins from the synchronized 30-Hz source streams.
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


DATA_DIR = Path("/app/data")
ANIMALS = [
    "QLAK-CA1-08",
    "QLAK-CA1-30",
    "QLAK-CA1-50",
    "QLAK-CA1-51",
    "QLAK-CA1-56",
    "QLAK-CA1-74",
    "QLAK-CA1-75",
]
SOURCE_FPS = 30
TRIAL_SECONDS = 60
SOURCE_FRAMES_PER_TRIAL = SOURCE_FPS * TRIAL_SECONDS
POOL_FRAMES = 3
POOLED_BINS_PER_TRIAL = SOURCE_FRAMES_PER_TRIAL // POOL_FRAMES
NEURAL_SMOOTH_SIGMA_FRAMES = 3.0
SPATIAL_BIN_CM = 25.0
N_SPATIAL_BINS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="process all sessions (default)")
    mode.add_argument("--sample", action="store_true", help="process only two sessions")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="save processing_<session_id>.png for up to two sessions",
    )
    return parser.parse_args()


def blocked_vector(blocked_entry) -> tuple[np.ndarray, list[int]]:
    """Return a 9-vector (1=blocked) from the native nested blocked field."""
    values = np.asarray(blocked_entry[0], dtype=float).reshape(-1)
    if values.size == 1 and values[0] == -1:
        indices: list[int] = []
    else:
        if not np.allclose(values, np.round(values)):
            raise ValueError(f"Non-integer blocked indices: {values}")
        indices = [int(v) for v in values]
    if any(v < 0 or v > 8 for v in indices):
        raise ValueError(f"Blocked index outside 0..8: {indices}")
    vector = np.zeros(9, dtype=np.float32)
    vector[indices] = 1.0
    return vector, indices


def pool_position(position_trial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean-pool (2, 1800) position and return coordinates plus 9-class labels."""
    pooled = position_trial.reshape(2, POOLED_BINS_PER_TRIAL, POOL_FRAMES).mean(axis=2)
    xy_bin = np.floor(pooled / SPATIAL_BIN_CM).astype(np.int16)
    np.clip(xy_bin, 0, N_SPATIAL_BINS - 1, out=xy_bin)
    # Native position is (x, y), whereas the source blocked-partition field is
    # row-major as [[0,1,2],[3,4,5],[6,7,8]]. Empirically and geometrically,
    # rows correspond to y and columns to x, so transpose into that canonical
    # geometry order before flattening.
    labels = (xy_bin[1] * N_SPATIAL_BINS + xy_bin[0]).astype(np.uint8)[None, :]
    return pooled, labels


def convert_session(
    trace_day: np.ndarray,
    position_day: np.ndarray,
    blocked_entry,
    animal: str,
    subject_index: int,
    day: int,
    environment: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, dict]:
    """Convert one synchronized animal-day recording."""
    if trace_day.ndim != 2 or position_day.ndim != 2 or position_day.shape[0] != 2:
        raise ValueError(
            f"{animal} day {day}: unexpected trace/position shapes "
            f"{trace_day.shape}, {position_day.shape}"
        )
    if trace_day.shape[1] != position_day.shape[1]:
        raise ValueError(f"{animal} day {day}: trace and position frame counts differ")
    if not np.isfinite(position_day).all():
        raise ValueError(f"{animal} day {day}: position contains NaN/Inf")

    # Native data represent an absent CellReg cell as NaN for its entire day.
    valid_cells = np.flatnonzero(np.isfinite(trace_day[:, 0]))
    if valid_cells.size == 0:
        raise ValueError(f"{animal} day {day}: no registered cells")

    n_source_frames = trace_day.shape[1]
    n_trials = n_source_frames // SOURCE_FRAMES_PER_TRIAL
    if n_trials < 2:
        raise ValueError(f"{animal} day {day}: fewer than two complete trials")
    n_used_frames = n_trials * SOURCE_FRAMES_PER_TRIAL

    # Shape (cell, trial, source-frame). A single C-level filter call handles all
    # trials while axis=2 keeps smoothing strictly within each trial.
    selected = np.asarray(trace_day[valid_cells, :n_used_frames], dtype=np.float32)
    selected = selected.reshape(valid_cells.size, n_trials, SOURCE_FRAMES_PER_TRIAL)
    smoothed = gaussian_filter1d(
        selected,
        sigma=NEURAL_SMOOTH_SIGMA_FRAMES,
        axis=2,
        mode="reflect",
    )
    pooled_neural = smoothed.reshape(
        valid_cells.size, n_trials, POOLED_BINS_PER_TRIAL, POOL_FRAMES
    ).mean(axis=3, dtype=np.float32)
    del selected, smoothed

    geometry, blocked_indices = blocked_vector(blocked_entry)
    neural_trials: list[np.ndarray] = []
    input_trials: list[np.ndarray] = []
    output_trials: list[np.ndarray] = []
    for trial in range(n_trials):
        start = trial * SOURCE_FRAMES_PER_TRIAL
        stop = start + SOURCE_FRAMES_PER_TRIAL
        _, labels = pool_position(position_day[:, start:stop])
        neural_trial = np.ascontiguousarray(pooled_neural[:, trial, :], dtype=np.float32)
        if not np.isfinite(neural_trial).all():
            raise ValueError(f"{animal} day {day}: retained neural data contain NaN/Inf")
        if labels.min() < 0 or labels.max() > 8:
            raise ValueError(f"{animal} day {day}: output outside 0..8")
        neural_trials.append(neural_trial)
        input_trials.append(geometry.copy())
        output_trials.append(labels)
    del pooled_neural

    session_info = {
        "session_id": f"{animal}_day{day:02d}",
        "subject": animal,
        "subject_idx": subject_index,
        "source_day_index": day,
        "environment": environment,
        "blocked_partition_indices": blocked_indices,
        "source_n_cells": int(trace_day.shape[0]),
        "retained_n_neurons": int(valid_cells.size),
        "source_neuron_indices": valid_cells.astype(int).tolist(),
        "source_n_frames": int(n_source_frames),
        "used_source_frames": int(n_used_frames),
        "discarded_tail_frames": int(n_source_frames - n_used_frames),
        "n_trials": int(n_trials),
        "trial_source_start_frames": [
            int(i * SOURCE_FRAMES_PER_TRIAL) for i in range(n_trials)
        ],
    }
    return neural_trials, input_trials, output_trials, valid_cells, session_info


def plot_processing(
    trace_day: np.ndarray,
    position_day: np.ndarray,
    valid_cells: np.ndarray,
    neural_trials: list[np.ndarray],
    output_trials: list[np.ndarray],
    geometry: np.ndarray,
    session_info: dict,
) -> Path:
    """Visualize cell selection, segmentation, smoothing/pooling, and binning."""
    session_id = session_info["session_id"]
    raw = np.asarray(trace_day[valid_cells[:20], :SOURCE_FRAMES_PER_TRIAL], dtype=np.float32)
    smooth = gaussian_filter1d(raw, sigma=NEURAL_SMOOTH_SIGMA_FRAMES, axis=1)
    pooled_xy, labels = pool_position(position_day[:, :SOURCE_FRAMES_PER_TRIAL])
    raw_t = np.arange(SOURCE_FRAMES_PER_TRIAL) / SOURCE_FPS
    pooled_t = (np.arange(POOLED_BINS_PER_TRIAL) * POOL_FRAMES + 1) / SOURCE_FPS

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    ax = axes.ravel()

    geom = geometry.reshape(3, 3)
    ax[0].imshow(geom, vmin=0, vmax=1, cmap="gray_r")
    for i in range(3):
        for j in range(3):
            ax[0].text(j, i, f"{3*i+j}\n{int(geom[i,j])}", ha="center", va="center")
    ax[0].set_title("Static geometry input (1 = blocked)")

    ax[1].plot(position_day[0, :SOURCE_FRAMES_PER_TRIAL],
               position_day[1, :SOURCE_FRAMES_PER_TRIAL], lw=0.5)
    for edge in (25, 50):
        ax[1].axvline(edge, color="k", ls=":")
        ax[1].axhline(edge, color="k", ls=":")
    ax[1].set(xlim=(0, 75), ylim=(0, 75), xlabel="source x (cm)", ylabel="source y (cm)",
              title="Raw position, trial 0, with 3x3 boundaries")

    ax[2].eventplot([np.flatnonzero(row) / SOURCE_FPS for row in raw],
                    lineoffsets=np.arange(len(raw)), linelengths=0.8)
    ax[2].set(xlim=(0, 60), xlabel="time (s)", ylabel="sample retained cells",
              title="Raw binary rising-phase events")

    for i, row in enumerate(smooth[:10]):
        ax[3].plot(raw_t, row + i * 0.25, lw=0.7)
    ax[3].set(xlim=(0, 10), xlabel="time (s)", ylabel="offset cells",
              title="Gaussian smoothing (sigma = 3 source frames)")

    ax[4].imshow(neural_trials[0][:20], aspect="auto", interpolation="nearest",
                 extent=(0, 60, min(20, valid_cells.size), 0))
    ax[4].set(xlabel="time (s)", ylabel="sample retained cells",
              title="3-frame mean-pooled neural activity (10 Hz)")

    ax[5].plot(raw_t, position_day[0, :SOURCE_FRAMES_PER_TRIAL], alpha=0.35, label="raw x")
    ax[5].plot(raw_t, position_day[1, :SOURCE_FRAMES_PER_TRIAL], alpha=0.35, label="raw y")
    ax[5].plot(pooled_t, pooled_xy[0], lw=1, label="pooled x")
    ax[5].plot(pooled_t, pooled_xy[1], lw=1, label="pooled y")
    ax[5].set(xlim=(0, 20), xlabel="time (s)", ylabel="cm",
              title="Raw and aligned 3-frame pooled position")
    ax[5].legend(ncol=2, fontsize=8)

    ax[6].step(pooled_t, labels[0], where="mid")
    ax[6].set(xlim=(0, 20), ylim=(-0.5, 8.5), yticks=range(9), xlabel="time (s)",
              ylabel="class", title="Time-aligned 3x3 output class")

    nframes = session_info["source_n_frames"]
    used = session_info["used_source_frames"]
    ax[7].barh([0], [used / SOURCE_FPS], color="tab:blue", label="complete trials")
    ax[7].barh([0], [(nframes - used) / SOURCE_FPS], left=[used / SOURCE_FPS],
               color="tab:orange", label="discarded tail")
    for frame in session_info["trial_source_start_frames"]:
        ax[7].axvline(frame / SOURCE_FPS, color="white", lw=0.25)
    ax[7].set(xlabel="source recording time (s)", yticks=[],
              title=f"{session_info['n_trials']} non-overlapping 60-s trials")
    ax[7].legend(fontsize=8)

    ax[8].axis("off")
    ax[8].text(
        0,
        1,
        "\n".join(
            [
                session_id,
                f"environment: {session_info['environment']}",
                f"source cells: {session_info['source_n_cells']}",
                f"retained valid cells: {session_info['retained_n_neurons']}",
                f"source frames: {nframes}",
                f"used / tail frames: {used} / {nframes-used}",
                "neural/output bins: 600 x 100 ms per trial",
                f"shape check: neural {neural_trials[0].shape}, output {output_trials[0].shape}",
            ]
        ),
        va="top",
        family="monospace",
    )

    fig.suptitle(f"Conversion processing: {session_id}", fontsize=16)
    fig.tight_layout()
    path = Path("/app") / f"processing_{session_id}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    sample = args.sample
    start_all = time.perf_counter()
    print(f"Mode: {'sample (2 sessions)' if sample else 'full (all sessions)'}", flush=True)
    print(
        f"Source: {SOURCE_FPS} Hz; trial: {TRIAL_SECONDS} s; "
        f"pool: {POOL_FRAMES} frames -> {1000*POOL_FRAMES/SOURCE_FPS:.1f} ms",
        flush=True,
    )

    data = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": ANIMALS.copy(),
        "subject_idx": [],
        "brain_regions": ["CA1"],
        "brain_region_idx": [],
        "input_names": [f"blocked_partition_{i}" for i in range(9)],
        "output_names": ["position_3x3"],
        "output_values": [[f"y{y}_x{x}" for y in range(3) for x in range(3)]],
        "metadata": {
            "task_description": (
                "Decode mouse location in one of nine 25x25-cm spatial bins from "
                "CA1 calcium-event activity, conditioned on blocked arena partitions."
            ),
            "time_bin_size": 100.0,
            "temporal_alignment_event": "start of each non-overlapping 1-minute session segment",
            "off_start": 0.0,
            "off_end": 60.0,
            "source_sampling_rate_hz": SOURCE_FPS,
            "source_trial_frames": SOURCE_FRAMES_PER_TRIAL,
            "temporal_pool_frames": POOL_FRAMES,
            "neural_processing": (
                "Native binary rising-phase events; per-trial Gaussian smoothing "
                "sigma=3 source frames (reflect mode), then non-overlapping "
                "3-frame mean pooling, matching the reference decoder."
            ),
            "position_processing": (
                "Synchronized x-y position mean-pooled over identical 3-frame windows; "
                "floor-divided by 25 cm, clipped to 0..2, and flattened in the "
                "source geometry's row-major (y-row, x-column) partition order."
            ),
            "trial_processing": (
                "Consecutive complete 1800-frame trials from source frame 0; final "
                "incomplete tail excluded without padding."
            ),
            "cell_filter": (
                "All manually curated cells registered (finite) on the source day; "
                "no place-cell or activity threshold."
            ),
            "spatial_class_order": [f"y{y}_x{x}" for y in range(3) for x in range(3)],
            "geometry_input_definition": "Nine binary source partition flags; 1=blocked, 0=accessible.",
            "session_info": [],
        },
    }

    n_plotted = 0
    stop_after = 2 if sample else None
    for subject_index, animal in enumerate(ANIMALS):
        if stop_after is not None and len(data["neural"]) >= stop_after:
            break
        load_start = time.perf_counter()
        print(f"Loading {animal} ...", flush=True)
        bundle = joblib.load(DATA_DIR / animal)
        if list(bundle.keys()) != [animal]:
            raise ValueError(f"Unexpected top-level keys in {animal}: {list(bundle.keys())}")
        record = bundle[animal]
        trace = record["trace"]
        position = record["position"]
        blocked = record["blocked"]
        envs = np.asarray(record["envs"]).reshape(-1)
        del record, bundle
        gc.collect()
        print(
            f"Loaded {animal} in {time.perf_counter()-load_start:.2f}s: "
            f"trace {trace.shape}, position {position.shape}",
            flush=True,
        )

        if not (trace.shape[0] == position.shape[0] == len(blocked) == len(envs)):
            raise ValueError(f"{animal}: day counts disagree across source fields")
        for day in range(trace.shape[0]):
            if stop_after is not None and len(data["neural"]) >= stop_after:
                break
            session_start = time.perf_counter()
            result = convert_session(
                trace[day],
                position[day],
                blocked[day],
                animal,
                subject_index,
                day,
                str(envs[day]),
            )
            neural_trials, input_trials, output_trials, valid_cells, session_info = result
            data["neural"].append(neural_trials)
            data["input"].append(input_trials)
            data["output"].append(output_trials)
            data["subject_idx"].append(subject_index)
            data["brain_region_idx"].append(np.zeros(valid_cells.size, dtype=np.int16))
            data["metadata"]["session_info"].append(session_info)

            if args.show_processing and n_plotted < 2:
                geometry, _ = blocked_vector(blocked[day])
                plot_path = plot_processing(
                    trace[day],
                    position[day],
                    valid_cells,
                    neural_trials,
                    output_trials,
                    geometry,
                    session_info,
                )
                print(f"Saved {plot_path}", flush=True)
                n_plotted += 1

            print(
                f"  {session_info['session_id']}: {valid_cells.size} neurons, "
                f"{len(neural_trials)} trials in {time.perf_counter()-session_start:.2f}s",
                flush=True,
            )
        del trace, position, blocked, envs
        gc.collect()

    data["subject_idx"] = np.asarray(data["subject_idx"], dtype=np.int16)
    nsessions = len(data["neural"])
    ntrials = sum(len(x) for x in data["neural"])
    nneurons = sum(x[0].shape[0] for x in data["neural"])
    if sample:
        if nsessions != 2:
            raise AssertionError(f"Sample mode produced {nsessions}, expected 2 sessions")
    else:
        if nsessions != 207 or ntrials != 8187 or nneurons != 69744:
            raise AssertionError(
                "Full native-count check failed: "
                f"sessions={nsessions}, trials={ntrials}, session-neurons={nneurons}"
            )

    for s in range(nsessions):
        if len(data["neural"][s]) < 2:
            raise AssertionError(f"Session {s} has fewer than two trials")
        n = data["neural"][s][0].shape[0]
        if len(data["brain_region_idx"][s]) != n:
            raise AssertionError(f"Session {s}: brain region length mismatch")
        for neural, inp, out in zip(data["neural"][s], data["input"][s], data["output"][s]):
            if neural.shape != (n, POOLED_BINS_PER_TRIAL):
                raise AssertionError(f"Session {s}: bad neural shape {neural.shape}")
            if inp.shape != (9,) or out.shape != (1, POOLED_BINS_PER_TRIAL):
                raise AssertionError(f"Session {s}: bad input/output shapes {inp.shape}, {out.shape}")

    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    write_start = time.perf_counter()
    print(
        f"Writing {args.outpicklefile}: {nsessions} sessions, {ntrials} trials, "
        f"{nneurons} session-neurons ...",
        flush=True,
    )
    with args.outpicklefile.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed = time.perf_counter() - start_all
    size_gib = args.outpicklefile.stat().st_size / 1024**3
    print(f"Write time: {time.perf_counter()-write_start:.2f}s", flush=True)
    print(f"Output size: {size_gib:.3f} GiB", flush=True)
    print(f"Total conversion time: {elapsed:.2f}s", flush=True)
    print("Conversion completed successfully.", flush=True)


if __name__ == "__main__":
    main()
