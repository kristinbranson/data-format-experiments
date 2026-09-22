#!/usr/bin/env python3
"""Convert the Lee et al. CA1 geometry data for position decoding.

The source joblib files contain the final, rise-extracted calcium-event trains
and frame-aligned head position used by the paper.  This converter follows the
paper's within-session position decoder where it is compatible with the target
format: cells need more than five events during movement, event trains are
Gaussian-smoothed with sigma=3 frames, and consecutive groups of three 30-Hz
frames are averaged.  Unlike the paper's decoder, slow samples remain in the
dataset because removing them would destroy the requested contiguous 1-minute
trials.
"""

from __future__ import annotations

import argparse
import gc
import pickle
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter1d


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
POOL_FRAMES = 3
TIME_BIN_MS = 100.0
TRIAL_SECONDS = 60
TRIAL_BINS = int(TRIAL_SECONDS * SOURCE_FPS / POOL_FRAMES)
ARENA_SIZE_CM = 75.0
N_SPATIAL_BINS = 3
PAPER_DECODER_SPATIAL_BINS = 15
VELOCITY_THRESHOLD_CM_S = 5.0
VELOCITY_FILTER_SIGMA_FRAMES = 5
MIN_MOVING_EVENTS = 5
NEURAL_FILTER_SIGMA_FRAMES = 3
MIN_FINAL_TRIAL_SECONDS = 30.0


def blocked_mask(blocked: object) -> np.ndarray:
    """Return nine indicators, with 1 meaning that the bin is blocked."""
    mask = np.zeros(N_SPATIAL_BINS**2, dtype=np.float32)
    values = np.asarray(blocked, dtype=float).reshape(-1)
    for value in values[np.isfinite(values)]:
        index = int(value)
        if index >= 0:
            if index >= mask.size:
                raise ValueError(f"Invalid blocked-bin index {index}")
            mask[index] = 1.0
    return mask


def moving_samples(position: np.ndarray) -> np.ndarray:
    """Reproduce the movement mask used by decode_position_within()."""
    # The reference code first expresses position in its 15-bin coordinate
    # system. Its >1 bin/s criterion is therefore the documented >5 cm/s.
    bin_cm = ARENA_SIZE_CM / PAPER_DECODER_SPATIAL_BINS
    position_15 = position / bin_cm
    speed_bins_s = np.linalg.norm(np.diff(position_15, axis=1), axis=0) * SOURCE_FPS
    speed_bins_s = gaussian_filter1d(
        speed_bins_s, sigma=VELOCITY_FILTER_SIGMA_FRAMES
    )
    moving = np.zeros(position.shape[1], dtype=bool)
    moving[1:] = speed_bins_s > (VELOCITY_THRESHOLD_CM_S / bin_cm)
    return moving


def pool_position(position: np.ndarray, usable_frames: int) -> np.ndarray:
    """Average each three-frame block, as in the paper's decoder."""
    return position[:, :usable_frames].reshape(2, -1, POOL_FRAMES).mean(axis=2)


def discretize_position(position: np.ndarray) -> np.ndarray:
    """Map x-y position to the paper dataset's documented 0..8 grid order."""
    # The epsilon gives an exact 75-cm boundary point to bin 2, matching the
    # reference rate-map code's treatment of the upper boundary.
    xy = np.floor(
        position / ((ARENA_SIZE_CM + np.finfo(np.float64).eps * 64) / N_SPATIAL_BINS)
    ).astype(np.int64)
    xy = np.clip(xy, 0, N_SPATIAL_BINS - 1)
    return xy[0] * N_SPATIAL_BINS + xy[1]


def convert(data_dir: Path, output_path: Path) -> dict:
    neural: list[list[np.ndarray]] = []
    decoder_input: list[list[np.ndarray]] = []
    decoder_output: list[list[np.ndarray]] = []
    subject_idx: list[int] = []
    brain_region_idx: list[np.ndarray] = []
    session_info: list[dict] = []

    for animal_index, animal in enumerate(ANIMALS):
        source_path = data_dir / animal
        print(f"Loading {source_path}", flush=True)
        source = joblib.load(source_path)[animal]
        traces = source["trace"]
        positions = source["position"]
        environments = np.asarray(source["envs"]).squeeze()
        blocked = source["blocked"]

        if traces.shape[0] != positions.shape[0] or traces.shape[0] != len(blocked):
            raise ValueError(f"Inconsistent session count for {animal}")

        for day in range(traces.shape[0]):
            trace = traces[day]
            position = positions[day]
            if trace.shape[1] != position.shape[1]:
                raise ValueError(f"Unaligned trace/position for {animal}, day {day}")

            moving = moving_samples(position)
            registered = np.isfinite(trace[:, 0])
            # nansum makes non-registered cells look inactive; the explicit
            # registration mask documents and checks the intended selection.
            moving_event_count = np.nansum(trace[:, moving], axis=1)
            keep_cells = registered & (moving_event_count > MIN_MOVING_EVENTS)
            if not np.any(keep_cells):
                raise ValueError(f"No eligible cells for {animal}, day {day}")

            # AvgPool1d in the reference decoder drops an incomplete 3-frame
            # group. Smooth before splitting so trial edges do not create
            # artificial filter boundaries.
            usable_frames = (trace.shape[1] // POOL_FRAMES) * POOL_FRAMES
            selected = np.asarray(trace[keep_cells, :usable_frames], dtype=np.float32)
            smoothed = gaussian_filter1d(
                selected, sigma=NEURAL_FILTER_SIGMA_FRAMES, axis=1
            )
            pooled_neural = smoothed.reshape(
                selected.shape[0], -1, POOL_FRAMES
            ).mean(axis=2, dtype=np.float32)
            pooled_position = pool_position(position, usable_frames)
            labels = discretize_position(pooled_position)

            geometry = blocked_mask(blocked[day])
            session_neural: list[np.ndarray] = []
            session_input: list[np.ndarray] = []
            session_output: list[np.ndarray] = []
            for start in range(0, pooled_neural.shape[1], TRIAL_BINS):
                stop = min(start + TRIAL_BINS, pooled_neural.shape[1])
                duration_s = (stop - start) * TIME_BIN_MS / 1000.0
                if duration_s < MIN_FINAL_TRIAL_SECONDS:
                    break
                session_neural.append(
                    np.ascontiguousarray(pooled_neural[:, start:stop], dtype=np.float32)
                )
                # Geometry is static during a recording day, so a 1-D input is
                # the non-redundant representation supported by the target API.
                session_input.append(geometry.copy())
                session_output.append(labels[np.newaxis, start:stop].copy())

            if len(session_neural) < 2:
                raise ValueError(f"Too few trials for {animal}, day {day}")

            neural.append(session_neural)
            decoder_input.append(session_input)
            decoder_output.append(session_output)
            subject_idx.append(animal_index)
            brain_region_idx.append(np.zeros(int(keep_cells.sum()), dtype=np.int64))
            session_info.append(
                {
                    "subject": animal,
                    "source_day_index": day,
                    "environment": str(environments[day]),
                    "blocked_bins": np.flatnonzero(geometry).astype(int).tolist(),
                    "source_frames": int(trace.shape[1]),
                    "registered_neurons": int(registered.sum()),
                    "retained_neurons": int(keep_cells.sum()),
                    "n_trials": len(session_neural),
                    "trial_timepoints": [int(x.shape[1]) for x in session_neural],
                }
            )

        del source, traces, positions
        gc.collect()

    output_values = [
        f"bin_{x * N_SPATIAL_BINS + y}_(x{x}_y{y})"
        for x in range(N_SPATIAL_BINS)
        for y in range(N_SPATIAL_BINS)
    ]
    data = {
        "neural": neural,
        "input": decoder_input,
        "output": decoder_output,
        "subjects": ANIMALS.copy(),
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": ["CA1"],
        "brain_region_idx": brain_region_idx,
        "input_names": [f"blocked_spatial_bin_{i}" for i in range(9)],
        "output_names": ["mouse_position_3x3_bin"],
        "output_values": [output_values],
        "metadata": {
            "task_description": (
                "Decode the mouse's 3x3 arena position from CA1 rise-extracted "
                "calcium activity, conditional on the environment's blocked bins."
            ),
            "time_bin_size": TIME_BIN_MS,
            "temporal_alignment_event": "start of each contiguous 1-minute recording segment",
            "off_start": 0.0,
            "off_end": 60.0,
            "source": "Lee et al. (2025), Neuron 113:307-320",
            "source_sampling_rate_hz": SOURCE_FPS,
            "neural_signal": "rise-extracted binary calcium events",
            "neural_processing": (
                "Gaussian smoothing (sigma=3 source frames) followed by mean pooling "
                "non-overlapping groups of 3 frames, matching the paper position decoder"
            ),
            "neuron_filter": (
                "registered on the source day and >5 calcium events during periods "
                "moving >5 cm/s, matching the paper position decoder"
            ),
            "timepoint_filter": (
                "none; low-speed samples retained to preserve contiguous 1-minute trials"
            ),
            "position_binning": (
                "mean x-y position per 3-frame block, then floor into a 3x3 grid "
                "over the documented 75x75 cm arena; label = 3*x_bin + y_bin"
            ),
            "trial_policy": (
                "fixed 600-bin (60 s) segments; final segment retained when at least "
                "30 s, and shorter post-40-minute acquisition overhangs discarded"
            ),
            "input_encoding": "nine binary indicators (1=blocked), ordered 0 through 8",
            "session_definition": "one recording day in one fixed environment geometry",
            "session_info": session_info,
        },
    }

    print(f"Writing {output_path}", flush=True)
    with output_path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--output", type=Path, default=Path("/app/converted_data.pkl"))
    args = parser.parse_args()
    data = convert(args.data_dir, args.output)
    print(
        f"Converted {len(data['neural'])} sessions and "
        f"{sum(map(len, data['neural']))} trials.",
        flush=True,
    )


if __name__ == "__main__":
    main()
