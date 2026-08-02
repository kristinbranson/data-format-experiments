#!/usr/bin/env python3
import argparse
import math
import os
import pickle
import re
import time
from dataclasses import dataclass

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FPS = 30
TRIAL_SECONDS = 60
TRIAL_FRAMES = FPS * TRIAL_SECONDS
TEMPORAL_BIN_FRAMES = 3
TIME_BIN_MS = 100.0
NOMINAL_SESSION_SECONDS = 40 * 60
NOMINAL_SESSION_FRAMES = NOMINAL_SESSION_SECONDS * FPS
POSITION_BINS = 3
BUFFER = 1e-5


@dataclass(frozen=True)
class SessionRecord:
    animal: str
    day_idx: int

    @property
    def session_id(self) -> str:
        return f"{self.animal}_day{self.day_idx:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the CA1 geometry-remapping dataset to the decoder format."
    )
    parser.add_argument("outpicklefile", help="Output pickle filename.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only 2 sessions for testing.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Plot processing summaries for up to 2 sessions as processing_<session_id>.png.",
    )
    return parser.parse_args()


def get_animals(data_dir: str) -> list[str]:
    animals = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path) and re.fullmatch(r"QLAK-CA1-\d+", name):
            animals.append(name)
    return animals


def blocked_to_open_vector(blocked_entry) -> np.ndarray:
    open_mask = np.ones(9, dtype=np.float32)
    blocked_values = np.array(blocked_entry[0], dtype=np.float32).reshape(-1)
    if not (blocked_values.size == 1 and blocked_values[0] == -1):
        open_mask[blocked_values.astype(int)] = 0.0
    # Raw blocked indices are stored in a y-major 3x3 layout, while output classes use x_bin * 3 + y_bin.
    return open_mask.reshape(3, 3).T.reshape(-1)


def temporal_bin_mean(arr: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (arr.shape[-1] // bin_size) * bin_size
    if usable <= 0:
        raise ValueError("Segment is too short to create at least one temporal bin.")
    trimmed = arr[..., :usable]
    new_shape = arr.shape[:-1] + (usable // bin_size, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)


def discretize_position_3x3(position_xy_by_time: np.ndarray, session_max_xy: np.ndarray) -> np.ndarray:
    if position_xy_by_time.shape[0] != 2:
        raise ValueError(f"Expected position shape (2, T), got {position_xy_by_time.shape}")
    denom = (session_max_xy + BUFFER) / POSITION_BINS
    denom = np.where(denom <= 0, 1.0, denom)
    binned = np.floor(position_xy_by_time / denom[:, np.newaxis]).astype(np.int64)
    binned = np.clip(binned, 0, POSITION_BINS - 1)
    classes = binned[0] * POSITION_BINS + binned[1]
    return classes[np.newaxis, :]


def get_valid_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])


def get_trial_slices(n_frames_session: int) -> list[tuple[int, int]]:
    usable_frames = min(n_frames_session, NOMINAL_SESSION_FRAMES)
    slices = []
    for trial_idx in range(NOMINAL_SESSION_SECONDS // TRIAL_SECONDS):
        start = trial_idx * TRIAL_FRAMES
        end = min(start + TRIAL_FRAMES, usable_frames)
        if end - start >= TEMPORAL_BIN_FRAMES:
            slices.append((start, end))
    return slices


def load_animal(path: str, animal: str) -> dict:
    return joblib.load(os.path.join(path, animal))[animal]


def maybe_plot_processing(
    session: SessionRecord,
    env_label: str,
    open_mask: np.ndarray,
    raw_position: np.ndarray,
    raw_neural: np.ndarray,
    trial_slices: list[tuple[int, int]],
    first_trial_neural_binned: np.ndarray,
    first_trial_output: np.ndarray,
    session_max_xy: np.ndarray,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"Processing Summary: {session.session_id} ({env_label})")

    ax = axes[0, 0]
    open_grid = open_mask.reshape(3, 3)
    ax.imshow(open_grid, cmap="Greys", vmin=0, vmax=1)
    for x in range(3):
        for y in range(3):
            ax.text(y, x, str(x * 3 + y), ha="center", va="center", color="tab:red", fontsize=10)
    ax.set_title("Raw Geometry Input (1=open, 0=blocked)")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))

    ax = axes[0, 1]
    ax.plot(raw_position[0], raw_position[1], color="tab:blue", linewidth=0.5, alpha=0.8)
    ax.set_title("Raw Trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax = axes[1, 0]
    trial_lengths = np.array([end - start for start, end in trial_slices], dtype=np.int64)
    ax.bar(np.arange(len(trial_lengths)), trial_lengths / FPS, color="tab:green")
    ax.axhline(TRIAL_SECONDS, linestyle="--", color="tab:gray", linewidth=1)
    ax.set_title("Trial Durations (seconds)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Seconds")

    ax = axes[1, 1]
    neural_preview = raw_neural[: min(20, raw_neural.shape[0]), : min(600, raw_neural.shape[1])]
    ax.imshow(neural_preview, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title("Raw Neural Preview (first 20 neurons)")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Neuron")

    ax = axes[2, 0]
    neural_binned_preview = first_trial_neural_binned[: min(20, first_trial_neural_binned.shape[0])]
    ax.imshow(neural_binned_preview, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title("Binned Neural Preview (trial 0)")
    ax.set_xlabel("100 ms bins")
    ax.set_ylabel("Neuron")

    ax = axes[2, 1]
    ax.plot(first_trial_output.squeeze(), color="tab:orange", linewidth=1)
    ax.set_ylim(-0.5, 8.5)
    ax.set_yticks(range(9))
    ax.set_title(
        "Binned Position Classes (trial 0)\n"
        f"session max x/y = ({session_max_xy[0]:.2f}, {session_max_xy[1]:.2f})"
    )
    ax.set_xlabel("100 ms bins")
    ax.set_ylabel("3x3 bin")

    fig.tight_layout()
    fig.savefig(f"processing_{session.session_id}.png", dpi=150)
    plt.close(fig)


def convert_session(
    session: SessionRecord,
    animal_data: dict,
    show_processing: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, dict]:
    day = session.day_idx
    trace_day = animal_data["trace"][day]
    position_day = animal_data["position"][day]
    env_label = animal_data["envs"].reshape(-1)[day]
    open_mask = blocked_to_open_vector(animal_data["blocked"][day])
    valid_cells = get_valid_cell_mask(trace_day)

    trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)
    position_valid = position_day.astype(np.float32, copy=False)
    usable_frames = min(trace_valid.shape[1], NOMINAL_SESSION_FRAMES)
    session_max_xy = np.nanmax(position_valid[:, :usable_frames], axis=1)
    trial_slices = get_trial_slices(trace_valid.shape[1])

    neural_trials: list[np.ndarray] = []
    input_trials: list[np.ndarray] = []
    output_trials: list[np.ndarray] = []

    first_trial_neural_binned = None
    first_trial_output = None

    for start, end in trial_slices:
        neural_raw = trace_valid[:, start:end]
        position_raw = position_valid[:, start:end]
        neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
        position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
        output_binned = discretize_position_3x3(position_binned, session_max_xy).astype(np.int64, copy=False)

        neural_trials.append(neural_binned)
        input_trials.append(open_mask.copy())
        output_trials.append(output_binned)

        if first_trial_neural_binned is None:
            first_trial_neural_binned = neural_binned
            first_trial_output = output_binned

    debug_info = {
        "env_label": str(env_label),
        "open_mask": open_mask,
        "usable_frames": usable_frames,
        "original_frames": int(trace_valid.shape[1]),
        "trial_slices": trial_slices,
        "session_max_xy": session_max_xy,
    }

    if show_processing and first_trial_neural_binned is not None and first_trial_output is not None:
        maybe_plot_processing(
            session=session,
            env_label=str(env_label),
            open_mask=open_mask,
            raw_position=position_valid[:, :usable_frames],
            raw_neural=trace_valid[:, :usable_frames],
            trial_slices=trial_slices,
            first_trial_neural_binned=first_trial_neural_binned,
            first_trial_output=first_trial_output,
            session_max_xy=session_max_xy,
        )

    brain_region_idx = np.zeros(valid_cells.sum(), dtype=np.int64)
    return neural_trials, input_trials, output_trials, brain_region_idx, debug_info


def build_dataset(out_path: str, sample_only: bool, show_processing: bool) -> None:
    data_dir = os.path.join(os.getcwd(), "data")
    animals = get_animals(data_dir)

    subjects = animals
    subject_to_idx = {animal: idx for idx, animal in enumerate(subjects)}

    converted = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": subjects,
        "subject_idx": [],
        "brain_regions": ["CA1"],
        "brain_region_idx": [],
        "input_names": [f"open_partition_{idx}" for idx in range(9)],
        "output_names": ["position_bin_3x3"],
        "output_values": [[f"bin_{idx}" for idx in range(9)]],
        "metadata": {
            "task_description": (
                "Freely exploring mice in 10 geometric environments; decode 3x3 position bin "
                "from CA1 calcium-event activity with static geometry input."
            ),
            "time_bin_size": TIME_BIN_MS,
            "temporal_alignment_event": "start of each consecutive 1-minute chunk within a session",
            "off_start": 0.0,
            "off_end": float(TRIAL_SECONDS),
            "nominal_session_duration_s": float(NOMINAL_SESSION_SECONDS),
            "trial_duration_s": float(TRIAL_SECONDS),
            "source_paper": "Lee, Keinath, Cianfarano and Brandon (2025), Neuron 113(2):307-320",
            "source_signal": "binarized rising-phase calcium event trace",
            "spatial_binning_rule": (
                "session-wise max-based discretization matching reference code, adapted to 3x3 bins"
            ),
            "geometry_input_source": "raw blocked partition indices from source files",
            "session_ids": [],
            "session_geometry_labels": [],
            "session_original_frames": [],
            "session_usable_frames": [],
            "session_trial_counts": [],
            "session_valid_neuron_counts": [],
        },
    }

    total_start = time.perf_counter()
    plotted = 0
    sessions_done = 0

    for animal in animals:
        animal_start = time.perf_counter()
        animal_data = load_animal(data_dir, animal)
        ndays = animal_data["trace"].shape[0]
        print(f"Loaded {animal}: {ndays} sessions")

        for day_idx in range(ndays):
            if sample_only and sessions_done >= 2:
                break

            session = SessionRecord(animal=animal, day_idx=day_idx)
            session_start = time.perf_counter()
            do_plot = show_processing and plotted < 2
            neural_trials, input_trials, output_trials, brain_region_idx, debug_info = convert_session(
                session=session,
                animal_data=animal_data,
                show_processing=do_plot,
            )

            converted["neural"].append(neural_trials)
            converted["input"].append(input_trials)
            converted["output"].append(output_trials)
            converted["subject_idx"].append(subject_to_idx[animal])
            converted["brain_region_idx"].append(brain_region_idx)
            converted["metadata"]["session_ids"].append(session.session_id)
            converted["metadata"]["session_geometry_labels"].append(debug_info["env_label"])
            converted["metadata"]["session_original_frames"].append(debug_info["original_frames"])
            converted["metadata"]["session_usable_frames"].append(debug_info["usable_frames"])
            converted["metadata"]["session_trial_counts"].append(len(neural_trials))
            converted["metadata"]["session_valid_neuron_counts"].append(int(brain_region_idx.shape[0]))

            if do_plot:
                plotted += 1

            sessions_done += 1
            elapsed = time.perf_counter() - session_start
            print(
                f"  Converted {session.session_id}: {len(neural_trials)} trials, "
                f"{brain_region_idx.shape[0]} neurons, {elapsed:.2f}s"
            )

        animal_elapsed = time.perf_counter() - animal_start
        print(f"Finished {animal} in {animal_elapsed:.2f}s")
        if sample_only and sessions_done >= 2:
            break

    converted["subject_idx"] = np.asarray(converted["subject_idx"], dtype=np.int64)

    with open(out_path, "wb") as f:
        pickle.dump(converted, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_elapsed = time.perf_counter() - total_start
    total_trials = sum(len(session_trials) for session_trials in converted["neural"])
    print(f"Saved {out_path}")
    print(f"Sessions converted: {sessions_done}")
    print(f"Total trials: {total_trials}")
    print(f"Elapsed time: {total_elapsed:.2f}s")


def main() -> None:
    args = parse_args()
    sample_only = args.sample
    if not args.full and not args.sample:
        sample_only = False
    build_dataset(args.outpicklefile, sample_only=sample_only, show_processing=args.show_processing)


if __name__ == "__main__":
    main()
