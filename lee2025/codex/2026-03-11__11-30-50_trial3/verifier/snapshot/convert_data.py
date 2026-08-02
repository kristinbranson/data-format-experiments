#!/usr/bin/env python3
import argparse
import gc
import os
import pickle
import time
from dataclasses import dataclass

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FPS = 30.0
TRIAL_SECONDS = 60
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)
POSITION_BINS = 3
GEOMETRY_BINS = 3
POSITION_BUFFER = 1e-5
PLOT_MAX_SESSIONS = 2
REGION_NAME = "CA1"


@dataclass(frozen=True)
class SessionRef:
    animal: str
    day_index: int

    @property
    def session_id(self) -> str:
        return f"{self.animal}_day{self.day_index:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the CA1 geometry dataset into the decoder format."
    )
    parser.add_argument("outpicklefile", type=str, help="Output pickle file path.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only 2 sessions for testing.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing plots for up to 2 sessions as processing_<session_id>.png.",
    )
    return parser.parse_args()


def get_animal_ids(data_dir: str) -> list[str]:
    return sorted(
        filename
        for filename in os.listdir(data_dir)
        if filename.startswith("QLAK-CA1-") and "." not in filename
    )


def load_animal_dataset(data_dir: str, animal: str) -> dict:
    return joblib.load(os.path.join(data_dir, animal))[animal]


def iter_session_refs(data_dir: str, sample: bool) -> tuple[list[str], list[SessionRef]]:
    animals = get_animal_ids(data_dir)
    session_refs: list[SessionRef] = []
    for animal in animals:
        dat = load_animal_dataset(data_dir, animal)
        for day_index in range(dat["envs"].shape[0]):
            session_refs.append(SessionRef(animal=animal, day_index=day_index))
            if sample and len(session_refs) >= 2:
                del dat
                gc.collect()
                return animals, session_refs
        del dat
        gc.collect()
    return animals, session_refs


def extract_day_blocked_entry(blocked_list: list, day_index: int) -> np.ndarray:
    entry = blocked_list[day_index]
    if isinstance(entry, list) and len(entry) == 1:
        entry = entry[0]
    return np.atleast_1d(np.asarray(entry)).astype(int)


def blocked_to_geometry_vector(blocked_list: list, day_index: int) -> tuple[np.ndarray, np.ndarray]:
    blocked = extract_day_blocked_entry(blocked_list, day_index)
    geometry = np.ones(GEOMETRY_BINS * GEOMETRY_BINS, dtype=np.float32)
    if not (blocked.size == 1 and blocked[0] == -1):
        geometry[blocked] = 0.0
    # Raw blocked indices use the README 3x3 numbering. Transpose to match
    # the coordinate frame used by position and the 15x15 spatial maps.
    geometry_grid = geometry.reshape(GEOMETRY_BINS, GEOMETRY_BINS).T.astype(np.float32)
    return geometry_grid, geometry_grid.reshape(-1).astype(np.float32)


def session_present_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])


def compute_position_bins(position_day: np.ndarray, n_bins: int = POSITION_BINS) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(position_day, dtype=np.float64).T
    scale = (np.nanmax(coords, axis=0) + POSITION_BUFFER) / float(n_bins)
    binned = np.floor(coords / scale).astype(np.int64)
    binned = np.clip(binned, 0, n_bins - 1)
    class_idx = (binned[:, 0] * n_bins + binned[:, 1]).astype(np.int64)
    return binned, class_idx


def aggregate_valid_map(smoothed_maps_day: np.ndarray) -> np.ndarray:
    valid_mask = np.any(~np.isnan(smoothed_maps_day), axis=2).astype(np.float32)
    return valid_mask.reshape(3, 5, 3, 5).max(axis=(1, 3))


def trial_slices(n_frames: int, trial_frames: int = TRIAL_FRAMES) -> list[slice]:
    n_trials = n_frames // trial_frames
    return [slice(i * trial_frames, (i + 1) * trial_frames) for i in range(n_trials)]


def plot_processing_figure(
    session_ref: SessionRef,
    env_name: str,
    geometry_grid: np.ndarray,
    valid_grid: np.ndarray,
    position_day: np.ndarray,
    position_bins: np.ndarray,
    neural_trial0: np.ndarray,
    output_trial0: np.ndarray,
    n_trials: int,
) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ax = ax.ravel()

    coords = position_day.T
    ax[0].scatter(coords[:, 0], coords[:, 1], c=np.arange(coords.shape[0]), s=1, cmap="viridis")
    ax[0].set_title("Full-session trajectory")
    ax[0].set_xlabel("Position dim 0")
    ax[0].set_ylabel("Position dim 1")

    im1 = ax[1].imshow(valid_grid, cmap="gray_r", vmin=0, vmax=1)
    ax[1].set_title("3x3 valid map from 15x15 masks")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    im2 = ax[2].imshow(geometry_grid, cmap="gray_r", vmin=0, vmax=1)
    ax[2].set_title("3x3 geometry from blocked field")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    occupancy = np.zeros((POSITION_BINS, POSITION_BINS), dtype=np.float32)
    for xbin, ybin in position_bins:
        occupancy[xbin, ybin] += 1.0
    occupancy /= occupancy.max()
    im3 = ax[3].imshow(occupancy, cmap="magma", vmin=0, vmax=1)
    ax[3].set_title("3x3 occupancy from raw position")
    fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

    first_minutes = min(3 * TRIAL_FRAMES, output_trial0.shape[1] * min(3, n_trials))
    session_output = (position_bins[:, 0] * POSITION_BINS + position_bins[:, 1]).astype(int)
    ax[4].plot(session_output[:first_minutes], lw=0.8)
    for boundary in range(TRIAL_FRAMES, first_minutes, TRIAL_FRAMES):
        ax[4].axvline(boundary, color="k", ls=":", lw=1)
    ax[4].set_title("Position-bin labels over time")
    ax[4].set_xlabel("Frame")
    ax[4].set_ylabel("3x3 bin")

    nneurons_plot = min(50, neural_trial0.shape[0])
    event_sample = neural_trial0[:nneurons_plot]
    ax[5].imshow(event_sample, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
    ax[5].set_title("First trial neural events (sample)")
    ax[5].set_xlabel("Frame in first 1-minute trial")
    ax[5].set_ylabel("Neuron")

    fig.suptitle(
        f"{session_ref.session_id} | env={env_name} | trials={n_trials} | geometry_match={bool(np.array_equal(geometry_grid, valid_grid))}"
    )
    fig.tight_layout()
    fig.savefig(f"processing_{session_ref.session_id}.png", dpi=150)
    plt.close(fig)


def process_session(
    dat: dict,
    session_ref: SessionRef,
    subject_lookup: dict[str, int],
    show_processing: bool,
    plot_session_ids: set[str],
) -> tuple[dict, dict]:
    day = session_ref.day_index
    env_name = str(dat["envs"][day, 0])
    position_day = np.asarray(dat["position"][day], dtype=np.float64)
    trace_day = np.asarray(dat["trace"][day], dtype=np.float64)
    smoothed_day = np.asarray(dat["maps"]["smoothed"][:, :, :, day], dtype=np.float64)

    present_mask = session_present_cell_mask(trace_day)
    if not np.any(present_mask):
        raise ValueError(f"{session_ref.session_id}: no present cells after NaN filtering")

    geometry_grid, geometry_vector = blocked_to_geometry_vector(dat["blocked"], day)
    valid_grid = aggregate_valid_map(smoothed_day)
    if not np.array_equal(geometry_grid, valid_grid):
        raise ValueError(
            f"{session_ref.session_id}: geometry derived from blocked field does not match maps valid mask"
        )

    position_bins, output_class = compute_position_bins(position_day)
    n_frames = position_day.shape[1]
    slices = trial_slices(n_frames)
    if len(slices) < 2:
        raise ValueError(f"{session_ref.session_id}: fewer than 2 full 1-minute trials available")

    neural_trials: list[np.ndarray] = []
    input_trials: list[np.ndarray] = []
    output_trials: list[np.ndarray] = []

    session_trace = trace_day[present_mask].astype(np.float16, copy=False)
    for trial_slice in slices:
        neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
        output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
        neural_trials.append(neural_trial)
        input_trials.append(geometry_vector.copy())
        output_trials.append(output_trial)

    if show_processing and session_ref.session_id in plot_session_ids:
        plot_processing_figure(
            session_ref=session_ref,
            env_name=env_name,
            geometry_grid=geometry_grid,
            valid_grid=valid_grid,
            position_day=position_day,
            position_bins=position_bins,
            neural_trial0=neural_trials[0],
            output_trial0=output_trials[0],
            n_trials=len(neural_trials),
        )

    session_data = {
        "neural": neural_trials,
        "input": input_trials,
        "output": output_trials,
        "subject_idx": subject_lookup[session_ref.animal],
        "brain_region_idx": np.zeros(session_trace.shape[0], dtype=np.int64),
    }
    session_meta = {
        "session_id": session_ref.session_id,
        "animal": session_ref.animal,
        "day_index": day,
        "environment": env_name,
        "n_frames_raw": int(n_frames),
        "n_trials": len(neural_trials),
        "n_neurons": int(session_trace.shape[0]),
        "discarded_tail_frames": int(n_frames - len(slices) * TRIAL_FRAMES),
    }
    return session_data, session_meta


def convert_dataset(outpicklefile: str, sample: bool, show_processing: bool) -> None:
    data_dir = "data"
    animals, session_refs = iter_session_refs(data_dir, sample=sample)
    subject_lookup = {animal: idx for idx, animal in enumerate(animals)}
    selected_plot_ids = {ref.session_id for ref in session_refs[:PLOT_MAX_SESSIONS]} if show_processing else set()

    converted = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": animals,
        "subject_idx": [],
        "brain_regions": [REGION_NAME],
        "brain_region_idx": [],
        "input_names": [f"geometry_bin_{i}" for i in range(GEOMETRY_BINS * GEOMETRY_BINS)],
        "output_names": ["position_bin_3x3"],
        "output_values": [[f"x{x}_y{y}" for x in range(POSITION_BINS) for y in range(POSITION_BINS)]],
        "metadata": {
            "task_description": "Decode 3x3-binned mouse position from CA1 calcium-event activity with static 3x3 environment geometry input.",
            "time_bin_size": 1000.0 / FPS,
            "temporal_alignment_event": "start of each non-overlapping 1-minute within-session segment",
            "off_start": 0.0,
            "off_end": float(TRIAL_SECONDS),
            "raw_fps": FPS,
            "trial_length_frames": TRIAL_FRAMES,
            "trial_length_seconds": TRIAL_SECONDS,
            "source_format": "joblib animal files in data/",
            "source_processing_reference": "Lee et al. (2025) georepca1 code and methods",
            "position_binning_rule": "session-wide floor(position / ((session_max + buffer) / 3)) with clipping to [0,2]",
            "geometry_alignment": "blocked partitions reshaped to 3x3 then transposed to match position/map axes",
            "brain_region": REGION_NAME,
            "session_info": [],
        },
    }

    animal_cache: dict[str, dict] = {}
    total_start = time.time()
    for session_index, session_ref in enumerate(session_refs):
        session_start = time.time()
        if session_ref.animal not in animal_cache:
            animal_cache.clear()
            gc.collect()
            animal_cache[session_ref.animal] = load_animal_dataset(data_dir, session_ref.animal)

        session_data, session_meta = process_session(
            dat=animal_cache[session_ref.animal],
            session_ref=session_ref,
            subject_lookup=subject_lookup,
            show_processing=show_processing,
            plot_session_ids=selected_plot_ids,
        )

        converted["neural"].append(session_data["neural"])
        converted["input"].append(session_data["input"])
        converted["output"].append(session_data["output"])
        converted["subject_idx"].append(session_data["subject_idx"])
        converted["brain_region_idx"].append(session_data["brain_region_idx"])
        converted["metadata"]["session_info"].append(session_meta)

        elapsed = time.time() - session_start
        print(
            f"Processed {session_ref.session_id}: "
            f"{session_meta['n_neurons']} neurons, {session_meta['n_trials']} trials, "
            f"{session_meta['n_frames_raw']} raw frames in {elapsed:.2f}s"
        )

    converted["subject_idx"] = np.asarray(converted["subject_idx"], dtype=np.int64)

    with open(outpicklefile, "wb") as f:
        pickle.dump(converted, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_elapsed = time.time() - total_start
    total_trials = sum(len(session_trials) for session_trials in converted["neural"])
    print(f"Saved converted dataset to {outpicklefile}")
    print(f"Sessions: {len(converted['neural'])}")
    print(f"Trials: {total_trials}")
    print(f"Subjects: {len(converted['subjects'])}")
    print(f"Elapsed: {total_elapsed:.2f}s ({total_elapsed / max(len(session_refs), 1):.2f}s/session)")


def main() -> None:
    args = parse_args()
    sample = args.sample
    if not args.full and not args.sample:
        sample = False
    convert_dataset(
        outpicklefile=args.outpicklefile,
        sample=sample,
        show_processing=args.show_processing,
    )


if __name__ == "__main__":
    main()
