#!/usr/bin/env python3
"""Convert the Lee et al. CA1 geometry data for position decoding.

The published joblib files already contain the authors' manually curated,
registered cells, timestamp-aligned position, and thresholded rising-phase
calcium events.  This converter therefore starts from those files rather than
repeating upstream image processing that cannot be reproduced from this data.

Conversion decisions
--------------------
* A source recording day is one session.  Cells absent on that day are NaN in
  the registered-cell tensor and are removed session-by-session.
* As in ``decode_position_within`` in the paper repository, cells must have
  more than five events while the animal is moving faster than 5 cm/s.  The
  speed estimate uses the repository's 5-frame Gaussian filter.
* The repository smooths event traces with a 3-frame Gaussian before temporal
  average pooling for position decoding.  We retain that smoothing and pool
  to non-overlapping 1 s bins.  The coarser downstream bin is appropriate for
  these 40-minute recordings and still preserves the paper's event-rate
  representation while making the common neural decoder tractable.
* Only complete, consecutive 60 s trials are kept.  A short recording tail is
  discarded rather than padding it or changing the time-bin duration.
* Position is averaged within each temporal bin and discretized into the
  arena's 3 x 3, 25-cm sectors.  The class is ``3*x_bin + y_bin``.  The source
  ``blocked`` indices are y-major, so the geometry matrix is transposed before
  flattening to use this same class coordinate convention.
"""

from __future__ import annotations

import argparse
import gc
import pickle
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter1d


SUBJECTS = [
    "QLAK-CA1-08",
    "QLAK-CA1-30",
    "QLAK-CA1-50",
    "QLAK-CA1-51",
    "QLAK-CA1-56",
    "QLAK-CA1-74",
    "QLAK-CA1-75",
]

SOURCE_FPS = 30
TIME_BIN_FRAMES = 30
TIME_BIN_MS = 1000.0
TRIAL_SECONDS = 60
TIME_BINS_PER_TRIAL = int(TRIAL_SECONDS * SOURCE_FPS / TIME_BIN_FRAMES)
RAW_FRAMES_PER_TRIAL = TRIAL_SECONDS * SOURCE_FPS
ARENA_SIZE_CM = 75.0
N_SPATIAL_BINS = 3
SPATIAL_BIN_CM = ARENA_SIZE_CM / N_SPATIAL_BINS


def blocked_geometry(blocked_for_day: object) -> tuple[np.ndarray, list[int]]:
    """Return a 9-vector (1=blocked) in x-major position-class order."""
    # MATLAB cells become a one-element Python list containing either scalar
    # -1 (nothing blocked) or an array of y-major flat indices.
    raw = np.asarray(blocked_for_day[0]).reshape(-1)
    blocked_yx = np.zeros((N_SPATIAL_BINS, N_SPATIAL_BINS), dtype=np.float32)
    for value in raw:
        idx = int(value)
        if idx >= 0:
            blocked_yx.flat[idx] = 1.0

    blocked_xy = blocked_yx.T
    vector = blocked_xy.reshape(-1)
    return vector, np.flatnonzero(vector).astype(int).tolist()


def moving_mask(position: np.ndarray) -> np.ndarray:
    """Reproduce the paper code's >5 cm/s locomotion criterion."""
    speed = np.zeros(position.shape[1], dtype=np.float64)
    instantaneous = np.linalg.norm(np.diff(position, axis=1) * SOURCE_FPS, axis=0)
    speed[1:] = gaussian_filter1d(instantaneous, sigma=5)
    return speed > 5.0


def position_classes(position: np.ndarray, used_frames: int) -> np.ndarray:
    """Average position in each 1 s bin, then return x-major 3x3 classes."""
    mean_position = position[:, :used_frames].reshape(
        2, -1, TIME_BIN_FRAMES
    ).mean(axis=2)
    # The data include exact 75-cm boundary values; clipping assigns those to
    # the outermost sector rather than producing an invalid bin 3.
    xy_bin = np.floor(mean_position / SPATIAL_BIN_CM).astype(np.int64)
    xy_bin = np.clip(xy_bin, 0, N_SPATIAL_BINS - 1)
    return (N_SPATIAL_BINS * xy_bin[0] + xy_bin[1]).astype(np.int64)


def convert(data_dir: Path, output_path: Path) -> dict:
    neural: list[list[np.ndarray]] = []
    decoder_input: list[list[np.ndarray]] = []
    decoder_output: list[list[np.ndarray]] = []
    subject_idx: list[int] = []
    brain_region_idx: list[np.ndarray] = []
    session_info: list[dict] = []

    for subject_number, subject in enumerate(SUBJECTS):
        source_path = data_dir / subject
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source joblib file: {source_path}")

        print(f"Loading {subject}", flush=True)
        wrapped = joblib.load(source_path)
        source = wrapped[subject]
        traces = np.asarray(source["trace"])
        positions = np.asarray(source["position"])
        environments = np.asarray(source["envs"]).reshape(-1)

        if traces.ndim != 3 or positions.ndim != 3:
            raise ValueError(f"Unexpected source shapes for {subject}")
        if traces.shape[0] != positions.shape[0] or traces.shape[0] != len(environments):
            raise ValueError(f"Session count mismatch for {subject}")
        if traces.shape[2] != positions.shape[2]:
            raise ValueError(f"Trace/position time mismatch for {subject}")

        for day in range(traces.shape[0]):
            position = positions[day]
            moving = moving_mask(position)

            # np.sum deliberately matches the reference implementation: a NaN
            # from an unregistered cell makes the comparison false.
            # Index the day first so NumPy keeps the array in (cell, time)
            # order; combining the integer and boolean indices in one
            # expression would move the advanced-indexed time axis first.
            events_while_moving = np.sum(traces[day][:, moving], axis=1)
            cell_mask = events_while_moving > 5
            n_cells = int(cell_mask.sum())
            if n_cells == 0:
                raise ValueError(f"No qualifying cells for {subject}, day {day}")

            total_frames = traces.shape[2]
            n_trials = total_frames // RAW_FRAMES_PER_TRIAL
            used_frames = n_trials * RAW_FRAMES_PER_TRIAL
            if n_trials < 2:
                raise ValueError(f"Fewer than two complete trials for {subject}, day {day}")

            selected = traces[day, cell_mask, :].astype(np.float32, copy=True)
            # Same event-trace smoothing used by the repository's within-day
            # position decoder.  Explicit output avoids a float64 temporary.
            smoothed = np.empty_like(selected)
            gaussian_filter1d(selected, sigma=3, axis=1, output=smoothed)
            del selected
            binned_neural = smoothed[:, :used_frames].reshape(
                n_cells, n_trials, TIME_BINS_PER_TRIAL, TIME_BIN_FRAMES
            ).mean(axis=3, dtype=np.float32)
            del smoothed

            classes = position_classes(position, used_frames).reshape(
                n_trials, TIME_BINS_PER_TRIAL
            )
            geometry, blocked_bins = blocked_geometry(source["blocked"][day])

            neural.append(
                [np.ascontiguousarray(binned_neural[:, trial, :]) for trial in range(n_trials)]
            )
            decoder_input.append([geometry.copy() for _ in range(n_trials)])
            decoder_output.append(
                [np.ascontiguousarray(classes[trial][None, :]) for trial in range(n_trials)]
            )
            subject_idx.append(subject_number)
            brain_region_idx.append(np.zeros(n_cells, dtype=np.int64))
            session_info.append(
                {
                    "subject": subject,
                    "source_day_index": int(day),
                    "environment": str(environments[day]),
                    "blocked_bins_x_major": blocked_bins,
                    "source_frame_count": int(total_frames),
                    "used_frame_count": int(used_frames),
                    "discarded_tail_frames": int(total_frames - used_frames),
                    "n_complete_trials": int(n_trials),
                    "n_registered_cells": int(np.all(np.isfinite(traces[day]), axis=1).sum()),
                    "n_cells_after_activity_filter": n_cells,
                }
            )

        del wrapped, source, traces, positions
        gc.collect()

    data = {
        "neural": neural,
        "input": decoder_input,
        "output": decoder_output,
        "subjects": SUBJECTS.copy(),
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": ["CA1"],
        "brain_region_idx": brain_region_idx,
        "input_names": [
            f"blocked_x{x}_y{y}"
            for x in range(N_SPATIAL_BINS)
            for y in range(N_SPATIAL_BINS)
        ],
        "output_names": ["position_3x3_bin"],
        "output_values": [[
            f"x{x}_y{y}"
            for x in range(N_SPATIAL_BINS)
            for y in range(N_SPATIAL_BINS)
        ]],
        "metadata": {
            "task_description": (
                "Decode the mouse's position among nine 25-cm sectors of a 75 x 75 cm "
                "arena from CA1 rising-phase calcium-event activity, with blocked-sector "
                "geometry supplied as static context."
            ),
            "time_bin_size": TIME_BIN_MS,
            "temporal_alignment_event": "start of each consecutive complete 60-second segment",
            "off_start": 0.0,
            "off_end": 60.0,
            "source_sampling_rate_hz": SOURCE_FPS,
            "trial_duration_seconds": TRIAL_SECONDS,
            "spatial_bin_size_cm": SPATIAL_BIN_CM,
            "spatial_class_formula": "3 * x_bin + y_bin",
            "neural_representation": (
                "Mean of paper-provided binary rising-phase events after a 3-frame "
                "Gaussian temporal filter, pooled in non-overlapping 1-second bins."
            ),
            "cell_filter": (
                "Registered on the source day and >5 events while speed exceeded 5 cm/s; "
                "speed Gaussian-filtered with sigma=5 source frames, matching the paper code."
            ),
            "incomplete_trial_policy": "discard recording tail shorter than 60 seconds",
            "geometry_encoding": (
                "Nine static float32 indicators (1=blocked), transposed from source y-major "
                "indices into the output's x-major class order."
            ),
            "session_info": session_info,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
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
