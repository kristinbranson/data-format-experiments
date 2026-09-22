#!/usr/bin/env python3
"""Convert the Lee et al. CA1 geometry dataset to decoder format.

Usage:
    python -u /app/convert_data.py OUTPUT [--full | --sample] [--show-processing]
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


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
SOURCE_HZ = 30
POOL_FRAMES = 3
TARGET_HZ = SOURCE_HZ / POOL_FRAMES
SMOOTH_SIGMA_FRAMES = 3.0
TRIAL_SECONDS = 60
RAW_TRIAL_FRAMES = SOURCE_HZ * TRIAL_SECONDS
TRIAL_TIMEPOINTS = int(TARGET_HZ * TRIAL_SECONDS)
ARENA_SIZE_CM = 75.0
N_POSITION_BINS = 3
POSITION_BIN_CM = ARENA_SIZE_CM / N_POSITION_BINS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process the first two sessions only.")
    parser.add_argument(
        "--show-processing", action="store_true",
        help="Save processing_<session_id>.png for up to two sessions.",
    )
    return parser.parse_args()


def blocked_mask_x_y(blocked_entry: object) -> np.ndarray:
    """Return 9 features in position-compatible x-major order; 1 means blocked.

    Source blocked indices are row-major in (y, x). Position is stored as (x, y),
    hence the transpose before flattening. The square sentinel -1 means no blocks.
    """
    source_indices = np.asarray(blocked_entry[0], dtype=float).reshape(-1)
    source_y_x = np.zeros((N_POSITION_BINS, N_POSITION_BINS), dtype=np.float32)
    valid = source_indices[source_indices >= 0].astype(np.int64)
    if valid.size:
        source_y_x.reshape(-1)[valid] = 1.0
    return source_y_x.T.reshape(-1).copy()


def pool_position(position: np.ndarray, n_raw_keep: int) -> tuple[np.ndarray, np.ndarray]:
    """Average aligned raw x/y samples in non-overlapping three-frame bins."""
    trimmed = np.asarray(position[:, :n_raw_keep], dtype=np.float64)
    pooled_xy = trimmed.reshape(2, -1, POOL_FRAMES).mean(axis=2)
    xy_bins = np.floor(pooled_xy / POSITION_BIN_CM).astype(np.int8)
    np.clip(xy_bins, 0, N_POSITION_BINS - 1, out=xy_bins)
    classes = (N_POSITION_BINS * xy_bins[0] + xy_bins[1]).astype(np.int8)
    return pooled_xy, classes


def pool_neural(trace: np.ndarray, registered: np.ndarray, n_raw_keep: int) -> np.ndarray:
    """Apply the reference decoder's sigma-3 smoothing and three-frame pooling."""
    source = np.asarray(trace[registered], dtype=np.float32)
    smoothed = np.empty_like(source, dtype=np.float32)
    gaussian_filter1d(source, sigma=SMOOTH_SIGMA_FRAMES, axis=1, output=smoothed)
    del source
    # Smooth over the full acquired session so trial boundaries do not create artifacts,
    # then omit the terminal incomplete minute and pool exactly as AvgPool1d does.
    pooled = smoothed[:, :n_raw_keep].reshape(
        smoothed.shape[0], -1, POOL_FRAMES
    ).mean(axis=2, dtype=np.float32)
    return np.ascontiguousarray(pooled, dtype=np.float32)


def plot_processing(
    session_id: str,
    raw_trace: np.ndarray,
    registered: np.ndarray,
    raw_position: np.ndarray,
    pooled_neural: np.ndarray,
    pooled_xy: np.ndarray,
    classes: np.ndarray,
    blocked: np.ndarray,
) -> Path:
    """Visualize loading, smoothing/pooling, alignment, geometry, and classes."""
    nneurons_show = min(25, int(registered.sum()))
    seconds_show = min(120, raw_position.shape[1] / SOURCE_HZ)
    raw_n = int(seconds_show * SOURCE_HZ)
    pooled_n = int(seconds_show * TARGET_HZ)
    selected_raw = raw_trace[registered][:nneurons_show, :raw_n]

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
    axes[0, 0].imshow(selected_raw, aspect="auto", interpolation="nearest", cmap="binary")
    axes[0, 0].set_title("1. Loaded binary rise events (30 Hz)")
    axes[0, 0].set_ylabel("registered cell")
    axes[0, 0].set_xlabel("raw frame")

    axes[0, 1].imshow(
        pooled_neural[:nneurons_show, :pooled_n], aspect="auto",
        interpolation="nearest", cmap="viridis",
    )
    axes[0, 1].set_title("2. Gaussian sigma=3 frames + 3-frame mean (10 Hz)")
    axes[0, 1].set_ylabel("same cell")
    axes[0, 1].set_xlabel("pooled time bin")

    raw_t = np.arange(raw_n) / SOURCE_HZ
    pooled_t = (np.arange(pooled_n) + 0.5) / TARGET_HZ
    for dim, color in enumerate(("tab:blue", "tab:orange")):
        axes[1, 0].plot(raw_t, raw_position[dim, :raw_n], color=color, alpha=.35, lw=.7)
        axes[1, 0].plot(pooled_t, pooled_xy[dim, :pooled_n], color=color, lw=1.0,
                        label=("x" if dim == 0 else "y"))
    axes[1, 0].axvline(60, color="black", ls="--", lw=.8, label="trial boundary")
    axes[1, 0].set_title("3. Raw and identically pooled position (aligned)")
    axes[1, 0].set_xlabel("seconds")
    axes[1, 0].set_ylabel("cm")
    axes[1, 0].legend(ncol=3, fontsize=8)

    occupancy = np.bincount(classes, minlength=9).reshape(3, 3).astype(float)
    occupancy /= max(1.0, occupancy.sum())
    image = axes[1, 1].imshow(occupancy, origin="lower", cmap="magma")
    blocked_grid = blocked.reshape(3, 3)
    for x in range(3):
        for y in range(3):
            axes[1, 1].text(y, x, f"{occupancy[x,y]:.3f}\n"
                                      f"{'BLOCKED' if blocked_grid[x,y] else 'open'}",
                            ha="center", va="center", color="white", fontsize=8)
    axes[1, 1].set_title("4. Position occupancy and transposed blocked mask")
    axes[1, 1].set_xlabel("y bin")
    axes[1, 1].set_ylabel("x bin")
    fig.colorbar(image, ax=axes[1, 1], shrink=.7, label="occupancy fraction")

    axes[2, 0].step(pooled_t, classes[:pooled_n], where="mid", lw=.8)
    axes[2, 0].axvline(60, color="black", ls="--", lw=.8)
    axes[2, 0].set_yticks(range(9))
    axes[2, 0].set_title("5. Categorical output: class = 3*x_bin + y_bin")
    axes[2, 0].set_xlabel("seconds")
    axes[2, 0].set_ylabel("position class")

    axes[2, 1].axis("off")
    axes[2, 1].text(
        0, 1,
        "6. Trial construction and checks\n\n"
        f"session: {session_id}\n"
        f"registered cells: {registered.sum()}\n"
        f"raw frames: {raw_position.shape[1]} at {SOURCE_HZ} Hz\n"
        f"complete 60 s trials: {classes.size // TRIAL_TIMEPOINTS}\n"
        f"time bins/trial: {TRIAL_TIMEPOINTS} at {TARGET_HZ:g} Hz\n"
        "Neural, input, and output are sliced on identical boundaries.\n"
        "The terminal partial minute is excluded.",
        va="top", family="monospace",
    )
    path = APP_DIR / f"processing_{session_id}.png"
    fig.suptitle(f"Conversion processing audit: {session_id}", fontsize=15)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def validate_built_data(data: dict, expected_sessions: int) -> dict:
    assert len(data["neural"]) == expected_sessions
    assert len(data["input"]) == expected_sessions == len(data["output"])
    total_trials = 0
    total_neurons = 0
    output_counts = np.zeros(9, dtype=np.int64)
    for s, (neural_trials, input_trials, output_trials) in enumerate(
        zip(data["neural"], data["input"], data["output"])
    ):
        assert len(neural_trials) >= 2
        assert len(neural_trials) == len(input_trials) == len(output_trials)
        nneurons = neural_trials[0].shape[0]
        assert data["brain_region_idx"][s].shape == (nneurons,)
        total_neurons += nneurons
        total_trials += len(neural_trials)
        for neural, inputs, outputs in zip(neural_trials, input_trials, output_trials):
            assert neural.shape == (nneurons, TRIAL_TIMEPOINTS)
            assert neural.dtype == np.float32 and np.isfinite(neural).all()
            assert (neural >= 0).all()
            assert inputs.shape == (9,) and inputs.dtype == np.float32
            assert np.isin(inputs, (0, 1)).all()
            assert outputs.shape == (1, TRIAL_TIMEPOINTS) and outputs.dtype == np.int8
            assert outputs.min() >= 0 and outputs.max() <= 8
            output_counts += np.bincount(outputs[0], minlength=9)
    assert data["subject_idx"].shape == (expected_sessions,)
    assert len(data["metadata"]["session_info"]) == expected_sessions
    return {
        "sessions": expected_sessions,
        "trials": total_trials,
        "session_neuron_instances": total_neurons,
        "output_counts": output_counts,
        "output_fractions": output_counts / output_counts.sum(),
    }


def convert(sample: bool, show_processing: bool) -> tuple[dict, dict]:
    neural_all: list[list[np.ndarray]] = []
    input_all: list[list[np.ndarray]] = []
    output_all: list[list[np.ndarray]] = []
    brain_region_idx: list[np.ndarray] = []
    subject_idx: list[int] = []
    session_info: list[dict] = []
    processing_plots = 0
    start_total = time.perf_counter()

    stop = False
    for animal_index, animal in enumerate(ANIMALS):
        if stop:
            break
        load_start = time.perf_counter()
        source = joblib.load(DATA_DIR / animal)[animal]
        print(f"Loaded {animal} in {time.perf_counter()-load_start:.2f} s", flush=True)
        ndays = source["trace"].shape[0]
        for day in range(ndays):
            if sample and len(neural_all) >= 2:
                stop = True
                break
            session_start = time.perf_counter()
            trace = source["trace"][day]
            position = source["position"][day]
            registered = np.isfinite(trace[:, 0])
            assert registered.any()
            assert np.isfinite(trace[registered]).all()
            assert np.isin(trace[registered], (0.0, 1.0)).all()
            assert np.isfinite(position).all()
            assert trace.shape[1] == position.shape[1]

            ntrials = trace.shape[1] // RAW_TRIAL_FRAMES
            n_raw_keep = ntrials * RAW_TRIAL_FRAMES
            pooled_neural = pool_neural(trace, registered, n_raw_keep)
            pooled_xy, position_classes = pool_position(position, n_raw_keep)
            blocked = blocked_mask_x_y(source["blocked"][day])
            assert pooled_neural.shape[1] == position_classes.size == ntrials * TRIAL_TIMEPOINTS

            neural_trials = []
            input_trials = []
            output_trials = []
            for trial in range(ntrials):
                start = trial * TRIAL_TIMEPOINTS
                end = start + TRIAL_TIMEPOINTS
                neural_trials.append(np.ascontiguousarray(pooled_neural[:, start:end]))
                input_trials.append(blocked.copy())
                output_trials.append(position_classes[None, start:end].copy())

            session_id = f"{animal}_day{day:02d}"
            if show_processing and processing_plots < 2:
                plot_path = plot_processing(
                    session_id, trace, registered, position, pooled_neural,
                    pooled_xy, position_classes, blocked,
                )
                processing_plots += 1
                print(f"Saved {plot_path}", flush=True)

            neural_all.append(neural_trials)
            input_all.append(input_trials)
            output_all.append(output_trials)
            brain_region_idx.append(np.zeros(int(registered.sum()), dtype=np.int64))
            subject_idx.append(animal_index)
            registered_ids = np.flatnonzero(registered).astype(np.int32)
            environment = str(np.asarray(source["envs"])[day].squeeze())
            session_info.append({
                "session_id": session_id,
                "subject": animal,
                "source_day_index": day,
                "environment": environment,
                "source_frames": int(trace.shape[1]),
                "retained_source_frames": int(n_raw_keep),
                "dropped_terminal_frames": int(trace.shape[1] - n_raw_keep),
                "n_trials": ntrials,
                "registered_source_cell_indices": registered_ids,
            })
            print(
                f"Processed session {len(neural_all):3d}: {session_id}, "
                f"cells={registered.sum()}, trials={ntrials}, "
                f"time={time.perf_counter()-session_start:.2f} s",
                flush=True,
            )
            del pooled_neural, pooled_xy, position_classes
        del source
        gc.collect()

    data = {
        "neural": neural_all,
        "input": input_all,
        "output": output_all,
        "subjects": ANIMALS.copy(),
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": ["CA1"],
        "brain_region_idx": brain_region_idx,
        "input_names": [f"blocked_x{x}_y{y}" for x in range(3) for y in range(3)],
        "output_names": ["position_3x3_bin"],
        "output_values": [[f"x{x}_y{y}" for x in range(3) for y in range(3)]],
        "metadata": {
            "task_description": (
                "Decode the mouse's 3x3 spatial position bin from CA1 rise-event "
                "activity, with the arena's blocked-grid geometry as static trial input."
            ),
            "time_bin_size": 100.0,
            "temporal_alignment_event": "start of each consecutive 60-second recording segment",
            "off_start": 0.0,
            "off_end": 60.0,
            "source_sampling_rate_hz": SOURCE_HZ,
            "target_sampling_rate_hz": TARGET_HZ,
            "trial_duration_seconds": TRIAL_SECONDS,
            "source_frames_per_trial": RAW_TRIAL_FRAMES,
            "timepoints_per_trial": TRIAL_TIMEPOINTS,
            "neural_signal": "binarized significant calcium-transient rising phases",
            "neural_preprocessing": (
                "Gaussian filter sigma=3 source frames along time, then non-overlapping "
                "3-frame mean pooling, matching the reference within-session decoder."
            ),
            "position_preprocessing": (
                "Non-overlapping 3-frame mean pooling aligned to neural bins; x and y "
                "each discretized into three 25-cm bins; class=3*x_bin+y_bin."
            ),
            "arena_size_cm": [75.0, 75.0],
            "position_bin_edges_cm": [0.0, 25.0, 50.0, 75.0],
            "input_encoding": "blocked_x_y is 1 for blocked and 0 for accessible",
            "brain_region": "dorsal hippocampal CA1",
            "cell_filter": "all and only cells registered (finite trace) in each session",
            "trial_filter": "all complete consecutive 60-s segments; terminal partial minute omitted",
            "session_info": session_info,
        },
    }
    stats = validate_built_data(data, len(neural_all))
    stats["conversion_seconds_before_pickle"] = time.perf_counter() - start_total
    return data, stats


def main() -> None:
    args = parse_args()
    sample = bool(args.sample)
    print(f"Mode: {'sample (first 2 sessions)' if sample else 'full (all sessions)'}", flush=True)
    data, stats = convert(sample=sample, show_processing=args.show_processing)
    print("Internal validation passed", flush=True)
    print(
        f"Summary: sessions={stats['sessions']}, trials={stats['trials']}, "
        f"session-neuron instances={stats['session_neuron_instances']}", flush=True,
    )
    print(f"Output fractions: {np.array2string(stats['output_fractions'], precision=6)}", flush=True)
    print(f"Conversion before pickle: {stats['conversion_seconds_before_pickle']:.2f} s", flush=True)

    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.outpicklefile.with_name(args.outpicklefile.name + ".tmp")
    pickle_start = time.perf_counter()
    with temporary.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, args.outpicklefile)
    print(
        f"Wrote {args.outpicklefile} ({args.outpicklefile.stat().st_size / 2**30:.3f} GiB) "
        f"in {time.perf_counter()-pickle_start:.2f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
