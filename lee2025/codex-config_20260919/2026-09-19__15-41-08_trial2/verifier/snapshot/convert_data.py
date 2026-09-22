#!/usr/bin/env python3
"""Convert the Lee et al. CA1 geometry dataset to decoder format.

Usage:
    python -u /app/convert_data.py OUTPUT.pkl [--full | --sample]
                                      [--show-processing]

The conversion follows the temporal processing in the authors' position decoder:
binary rising-phase calcium events are Gaussian-smoothed at sigma=3 native
frames, then both neural activity and position are mean-pooled over disjoint
3-frame (100 ms) bins. Recording days are sessions and complete contiguous
60-second windows are trials.
"""

from __future__ import annotations

import argparse
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
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
FPS = 30
POOL_FRAMES = 3
TIME_BIN_MS = 100.0
TRIAL_SECONDS = 60
RAW_TRIAL_FRAMES = FPS * TRIAL_SECONDS
TIMEPOINTS_PER_TRIAL = RAW_TRIAL_FRAMES // POOL_FRAMES
ARENA_CM = 75.0
SPATIAL_BINS = 3
SPATIAL_BIN_CM = ARENA_CM / SPATIAL_BINS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", help="process all sessions (default)")
    group.add_argument("--sample", action="store_true", help="process only the first two sessions")
    parser.add_argument(
        "--show-processing", action="store_true",
        help="save processing_<session_id>.png for up to two sessions",
    )
    return parser.parse_args()


def blocked_vector(raw_blocked: object) -> np.ndarray:
    """Return x-first 3x3 geometry with 1 for blocked, 0 for accessible.

    Native partition indices flatten a (y, x) matrix, whereas position and the
    output class flatten (x, y). Transposition puts both streams in one frame.
    """
    values = np.asarray(raw_blocked).reshape(-1)
    native_yx = np.zeros(9, dtype=np.float32)
    if values.size == 1 and float(values[0]) == -1.0:
        return native_yx
    indices = values.astype(np.int64)
    if np.any(indices < 0) or np.any(indices > 8):
        raise ValueError(f"Invalid blocked partition index: {values}")
    native_yx[indices] = 1.0
    return np.ascontiguousarray(native_yx.reshape(3, 3).T.ravel())


def pool_position(raw_position: np.ndarray, n_used_frames: int) -> np.ndarray:
    """Mean-pool an aligned 2 x time position stream into 100 ms bins."""
    selected = np.asarray(raw_position[:, :n_used_frames], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise ValueError("Position contains NaN/Inf in a retained complete trial")
    return selected.reshape(2, -1, POOL_FRAMES).mean(axis=2)


def position_classes(position_binned: np.ndarray) -> np.ndarray:
    """Convert 2-D centimeters to x-first, row-major 3x3 categorical labels."""
    upper = np.nextafter(np.float32(ARENA_CM), np.float32(0.0))
    xy = np.floor(np.clip(position_binned, 0.0, upper) / SPATIAL_BIN_CM).astype(np.int64)
    labels = xy[0] * SPATIAL_BINS + xy[1]
    if labels.min() < 0 or labels.max() > 8:
        raise AssertionError("Position discretization produced a class outside 0..8")
    return labels


def process_neural(raw_trace: np.ndarray, present: np.ndarray, n_used_frames: int) -> np.ndarray:
    """Select registered cells, Gaussian-smooth, and mean-pool aligned frames."""
    # Smooth the complete physical session before discarding the incomplete-trial
    # tail. This avoids an artificial filter boundary at the last retained minute.
    selected = np.asarray(raw_trace[present, :], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise ValueError("Registered neural traces contain intermittent NaN/Inf")
    # scipy's output argument avoids a second session-sized smoothed array.
    gaussian_filter1d(selected, sigma=POOL_FRAMES, axis=1, output=selected)
    selected = selected[:, :n_used_frames]
    return selected.reshape(selected.shape[0], -1, POOL_FRAMES).mean(axis=2)


def save_processing_plot(
    session_id: str,
    raw_position: np.ndarray,
    raw_trace: np.ndarray,
    present: np.ndarray,
    binned_position: np.ndarray,
    binned_neural: np.ndarray,
    labels: np.ndarray,
    geometry: np.ndarray,
) -> None:
    """Visualize source, smoothing/pooling, geometry, and discretization."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    ax = axes[0, 0]
    ax.plot(raw_position[0], raw_position[1], lw=0.25, alpha=0.6)
    for edge in (25, 50):
        ax.axvline(edge, color="k", lw=0.6); ax.axhline(edge, color="k", lw=0.6)
    for idx in np.flatnonzero(geometry):
        x, y = divmod(int(idx), 3)
        ax.add_patch(plt.Rectangle((25*x, 25*y), 25, 25, color="gray", alpha=0.3))
    ax.set(xlim=(0, 75), ylim=(0, 75), aspect="equal", title="Raw 30 Hz position + blocked bins",
           xlabel="x (cm)", ylabel="y (cm)")

    nshow = min(50, int(present.sum()))
    axes[0, 1].imshow(raw_trace[present, :RAW_TRIAL_FRAMES][:nshow], aspect="auto",
                      interpolation="nearest", cmap="binary")
    axes[0, 1].set(title="Raw binary rising-phase events (first minute)",
                   xlabel="30 Hz frame", ylabel="neuron")
    axes[0, 2].imshow(binned_neural[:nshow, :TIMEPOINTS_PER_TRIAL], aspect="auto",
                      interpolation="nearest", cmap="viridis")
    axes[0, 2].set(title="Gaussian-smoothed + 3-frame mean", xlabel="100 ms bin", ylabel="neuron")

    raw_t = np.arange(FPS * 10) / FPS
    bin_t = np.arange((FPS * 10) // POOL_FRAMES) * (POOL_FRAMES / FPS)
    axes[1, 0].plot(raw_t, raw_position[0, :FPS*10], alpha=0.5, label="raw x")
    axes[1, 0].plot(raw_t, raw_position[1, :FPS*10], alpha=0.5, label="raw y")
    axes[1, 0].plot(bin_t, binned_position[0, :len(bin_t)], lw=1.2, label="pooled x")
    axes[1, 0].plot(bin_t, binned_position[1, :len(bin_t)], lw=1.2, label="pooled y")
    axes[1, 0].set(title="Aligned position pooling (first 10 s)", xlabel="seconds", ylabel="cm")
    axes[1, 0].legend(ncol=2, fontsize=8)

    axes[1, 1].step(np.arange(TIMEPOINTS_PER_TRIAL) / 10,
                    labels[:TIMEPOINTS_PER_TRIAL], where="post")
    axes[1, 1].set(title="Final 3x3 position class (first trial)", xlabel="seconds",
                   ylabel="class", yticks=np.arange(9))
    axes[1, 2].imshow(geometry.reshape(3, 3), vmin=0, vmax=1, cmap="Greys")
    axes[1, 2].set(title="Static decoder input (1 = blocked)", xlabel="column", ylabel="row",
                   xticks=range(3), yticks=range(3))

    fig.suptitle(session_id)
    fig.tight_layout()
    out = Path(f"/app/processing_{session_id}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  saved {out}", flush=True)


def session_targets(sample: bool) -> dict[str, set[int] | None]:
    """Return animal/day selection. Sample mode means exactly two global sessions."""
    if sample:
        return {ANIMALS[0]: {0, 1}}
    return {animal: None for animal in ANIMALS}


def convert(sample: bool, show_processing: bool) -> dict:
    started = time.perf_counter()
    neural: list[list[np.ndarray]] = []
    decoder_input: list[list[np.ndarray]] = []
    output: list[list[np.ndarray]] = []
    subject_idx: list[int] = []
    brain_region_idx: list[np.ndarray] = []
    session_info: list[dict] = []
    plot_count = 0
    class_counts = np.zeros(9, dtype=np.int64)
    blocked_position_count = 0
    total_position_count = 0
    targets = session_targets(sample)

    for animal_idx, animal in enumerate(ANIMALS):
        if animal not in targets:
            continue
        load_start = time.perf_counter()
        payload = joblib.load(DATA_DIR / animal)[animal]
        print(f"Loaded {animal} in {time.perf_counter()-load_start:.2f} s", flush=True)
        trace = np.asarray(payload["trace"])
        position = np.asarray(payload["position"])
        envs = np.asarray(payload["envs"]).squeeze()
        selected_days = range(trace.shape[0]) if targets[animal] is None else sorted(targets[animal])

        for day in selected_days:
            session_start = time.perf_counter()
            n_raw_frames = int(position.shape[2])
            n_trials = n_raw_frames // RAW_TRIAL_FRAMES
            n_used_frames = n_trials * RAW_TRIAL_FRAMES
            if n_trials < 2:
                raise ValueError(f"{animal} day {day} has fewer than two complete trials")

            present = np.isfinite(trace[day, :, 0])
            if not present.any():
                raise ValueError(f"{animal} day {day} has no registered neurons")
            geometry = blocked_vector(payload["blocked"][day][0])
            binned_position = pool_position(position[day], n_used_frames)
            labels = position_classes(binned_position)
            binned_neural = process_neural(trace[day], present, n_used_frames)

            if binned_neural.shape[1] != labels.size:
                raise AssertionError("Neural and position bins are temporally misaligned")
            if binned_neural.shape[1] != n_trials * TIMEPOINTS_PER_TRIAL:
                raise AssertionError("Unexpected number of pooled bins")

            session_neural = []
            session_input = []
            session_output = []
            for trial in range(n_trials):
                sl = slice(trial * TIMEPOINTS_PER_TRIAL, (trial + 1) * TIMEPOINTS_PER_TRIAL)
                session_neural.append(np.ascontiguousarray(binned_neural[:, sl], dtype=np.float32))
                session_input.append(geometry.copy())
                session_output.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))

            # A diagnostic rather than a rejection rule: tracking at walls can place
            # occasional samples on a blocked-bin boundary.
            occupied_geometry = geometry[labels]
            blocked_position_count += int(occupied_geometry.sum())
            total_position_count += int(labels.size)
            class_counts += np.bincount(labels, minlength=9)

            neural.append(session_neural)
            decoder_input.append(session_input)
            output.append(session_output)
            subject_idx.append(animal_idx)
            brain_region_idx.append(np.zeros(int(present.sum()), dtype=np.int64))
            session_id = f"{animal}_day{day:02d}"
            session_info.append({
                "session_id": session_id,
                "subject": animal,
                "day_index": int(day),
                "environment": str(envs[day]),
                "original_frames": n_raw_frames,
                "used_frames": n_used_frames,
                "discarded_tail_frames": n_raw_frames - n_used_frames,
                "n_trials": n_trials,
                "n_neurons": int(present.sum()),
                "blocked_indices": np.flatnonzero(geometry).astype(int).tolist(),
            })

            if show_processing and plot_count < 2:
                save_processing_plot(session_id, position[day], trace[day], present,
                                     binned_position, binned_neural, labels, geometry)
                plot_count += 1
            print(
                f"  {session_id}: {int(present.sum())} neurons, {n_trials} trials, "
                f"{n_raw_frames-n_used_frames} tail frames; {time.perf_counter()-session_start:.2f} s",
                flush=True,
            )

        del payload, trace, position

    data = {
        "neural": neural,
        "input": decoder_input,
        "output": output,
        "subjects": ANIMALS.copy(),
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": ["CA1"],
        "brain_region_idx": brain_region_idx,
        "input_names": [f"blocked_x{x}_y{y}" for x in range(3) for y in range(3)],
        "output_names": ["position_3x3"],
        "output_values": [[f"x{x}_y{y}" for x in range(3) for y in range(3)]],
        "metadata": {
            "task_description": "Decode mouse position in nine 25x25 cm bins from CA1 activity, with blocked arena geometry as context.",
            "time_bin_size": TIME_BIN_MS,
            "temporal_alignment_event": "start of each contiguous one-minute window within the recording session",
            "off_start": 0.0,
            "off_end": 60.0,
            "source_sampling_rate_hz": FPS,
            "trial_duration_seconds": TRIAL_SECONDS,
            "neural_processing": "binary rising-phase events; Gaussian sigma=3 source frames; non-overlapping 3-frame mean pooling",
            "position_processing": "non-overlapping 3-frame mean pooling; clipped to arena; floor-divided into 25 cm bins; class=x_bin*3+y_bin",
            "geometry_encoding": "nine static x-first values (class=x*3+y); 1=blocked, 0=accessible; native (y,x) blocked matrix transposed into position frame",
            "neuron_curation": "all manually curated cells registered (finite) in each session; no place-cell or activity threshold",
            "tail_policy": "discard incomplete final 60-second window",
            "session_info": session_info,
        },
    }

    fractions = class_counts / class_counts.sum()
    print(f"Converted {len(neural)} sessions and {sum(map(len, neural))} trials", flush=True)
    print(f"Session-neuron total: {sum(len(x) for x in brain_region_idx)}", flush=True)
    print(f"Position class counts: {class_counts.tolist()}", flush=True)
    print(f"Position class fractions: {np.round(fractions, 6).tolist()}", flush=True)
    print(
        f"Samples assigned to a blocked partition: {blocked_position_count}/{total_position_count} "
        f"({blocked_position_count/max(1,total_position_count):.6%})",
        flush=True,
    )
    print(f"Conversion computation time: {time.perf_counter()-started:.2f} s", flush=True)
    return data


def validate_before_save(data: dict, sample: bool) -> None:
    nsessions = len(data["neural"])
    expected_sessions = 2 if sample else 207
    if nsessions != expected_sessions:
        raise AssertionError(f"Expected {expected_sessions} sessions, got {nsessions}")
    for s in range(nsessions):
        if not (len(data["neural"][s]) == len(data["input"][s]) == len(data["output"][s])):
            raise AssertionError(f"Trial-count mismatch in session {s}")
        nneurons = len(data["brain_region_idx"][s])
        for n, i, o in zip(data["neural"][s], data["input"][s], data["output"][s]):
            if n.shape != (nneurons, TIMEPOINTS_PER_TRIAL):
                raise AssertionError(f"Bad neural shape {n.shape} in session {s}")
            if i.shape != (9,) or o.shape != (1, TIMEPOINTS_PER_TRIAL):
                raise AssertionError(f"Bad input/output shape in session {s}: {i.shape}, {o.shape}")
            if not np.isfinite(n).all() or not np.isfinite(i).all() or not np.isfinite(o).all():
                raise AssertionError(f"NaN/Inf in session {s}")
    if not sample:
        if sum(map(len, data["neural"])) != 8187:
            raise AssertionError("Full dataset trial count is not 8,187")
        if sum(len(x) for x in data["brain_region_idx"]) != 69744:
            raise AssertionError("Full dataset session-neuron count is not 69,744")


def main() -> None:
    args = parse_args()
    sample = bool(args.sample)
    data = convert(sample=sample, show_processing=args.show_processing)
    validate_before_save(data, sample=sample)
    save_start = time.perf_counter()
    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    with args.outpicklefile.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    size_gib = args.outpicklefile.stat().st_size / 1024**3
    print(f"Saved {args.outpicklefile} ({size_gib:.3f} GiB) in {time.perf_counter()-save_start:.2f} s", flush=True)


if __name__ == "__main__":
    main()
