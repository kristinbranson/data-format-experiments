#!/usr/bin/env python3
"""Convert the Lee et al. CA1 geometry data to the neural-decoder format.

The downloaded joblib files already contain the paper's final, manually curated
rise-extracted calcium events and frame-aligned DeepLabCut positions.  Thus this
script deliberately does not redo calcium extraction or interpolate either
stream.  Each recording day is one decoder session and is split into 40
contiguous, nearly equal windows (the nominal 40 one-minute trials).  ``array_split``
keeps every frame despite small recording-length differences around 40 minutes.

Decisions tied to the reference analysis/data:

* A neuron is included in a day only when its trace is finite for the whole day.
  NaN traces denote a registered cell absent on that day.  Across the supplied
  data this recovers all 69,744 curated session-specific rate maps reported in
  the paper.  No additional place-cell or activity screen is imposed: the
  >5-event screen in ``decode_position_within`` is a decoder-model feature
  screen, not part of the source neural preprocessing.
* Neural values are the supplied binary rising-phase event vectors, stored as
  float32 as required by the downstream trainer.  Frames are retained at the
  simultaneous acquisition rate of 30 Hz.
* Position uses the physical 75 cm arena grid: x and y are clipped to bins 0--2
  after division by 25 cm.  The categorical ID is row-major ``y * 3 + x``.
  This is the convention of the supplied ``blocked`` IDs and makes positions
  fall only in accessible cells for all ten geometries.
* Geometry is a static nine-element float vector with 1 for blocked and 0 for
  accessible.  The square's sentinel blocked ID (-1) produces all zeros.

Run from anywhere with ``python /app/convert_data.py``.  Optional paths are
available mainly to make the conversion reproducible in another checkout.
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
from pathlib import Path

import joblib
import numpy as np


ANIMALS = [
    "QLAK-CA1-08",
    "QLAK-CA1-30",
    "QLAK-CA1-50",
    "QLAK-CA1-51",
    "QLAK-CA1-56",
    "QLAK-CA1-74",
    "QLAK-CA1-75",
]
FPS = 30.0
N_TRIALS = 40
ARENA_SIZE_CM = 75.0
GRID_SIZE = 3


def geometry_vector(blocked_entry: object) -> np.ndarray:
    """Return the source geometry as 1=blocked, 0=accessible."""
    # MATLAB cells become e.g. [array(-1.)] or [array([3., 5., 6., 8.])].
    raw = blocked_entry[0] if isinstance(blocked_entry, list) else blocked_entry
    blocked = np.asarray(raw).reshape(-1).astype(np.int64)
    geometry = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32)
    blocked = blocked[blocked >= 0]  # -1 is the source sentinel for the square.
    if blocked.size:
        if blocked.max() >= geometry.size:
            raise ValueError(f"Invalid blocked-cell ID(s): {blocked.tolist()}")
        geometry[blocked] = 1.0
    return geometry


def discretize_position(position: np.ndarray) -> np.ndarray:
    """Map source (x, y) positions to one row-major 3x3 class time series."""
    if position.ndim != 2 or position.shape[0] != 2:
        raise ValueError(f"Expected position shape (2, time), got {position.shape}")
    bin_width = ARENA_SIZE_CM / GRID_SIZE
    xy = np.floor(position / bin_width).astype(np.int64)
    # Source values can be exactly 75 cm; those belong to the last, not a fourth,
    # spatial bin.  Clipping also protects against tiny tracking overshoots.
    xy = np.clip(xy, 0, GRID_SIZE - 1)
    location = xy[1] * GRID_SIZE + xy[0]
    return location[np.newaxis, :]


def split_trials(array: np.ndarray, axis: int) -> list[np.ndarray]:
    """Split a full recording into the nominal 40 contiguous minute windows."""
    return [np.ascontiguousarray(x) for x in np.array_split(array, N_TRIALS, axis=axis)]


def convert(data_dir: Path) -> dict:
    neural: list[list[np.ndarray]] = []
    decoder_input: list[list[np.ndarray]] = []
    output: list[list[np.ndarray]] = []
    subject_idx: list[int] = []
    brain_region_idx: list[np.ndarray] = []
    session_info: list[dict] = []

    for animal_index, animal in enumerate(ANIMALS):
        source_path = data_dir / animal
        print(f"Loading {source_path} ...", flush=True)
        wrapped = joblib.load(source_path)
        source = wrapped[animal]

        traces = np.asarray(source["trace"])
        positions = np.asarray(source["position"])
        envs = np.asarray(source["envs"]).reshape(-1)
        blocked = source["blocked"]

        if traces.ndim != 3 or positions.ndim != 3:
            raise ValueError(f"Unexpected arrays for {animal}: {traces.shape}, {positions.shape}")
        if traces.shape[0] != positions.shape[0] or traces.shape[2] != positions.shape[2]:
            raise ValueError(f"Neural/position alignment mismatch for {animal}")
        if traces.shape[0] != len(envs) or traces.shape[0] != len(blocked):
            raise ValueError(f"Session metadata length mismatch for {animal}")

        for day in range(traces.shape[0]):
            day_trace = traces[day]
            finite_any = np.any(np.isfinite(day_trace), axis=1)
            finite_all = np.all(np.isfinite(day_trace), axis=1)
            if not np.array_equal(finite_any, finite_all):
                raise ValueError(f"Partially finite neuron trace in {animal}, day {day}")
            present = finite_all
            if not np.any(present):
                raise ValueError(f"No registered neurons in {animal}, day {day}")

            # The source is binary, but float32 avoids thousands of downstream
            # dtype warnings and is the trainer's native representation.
            session_neural_full = np.asarray(day_trace[present], dtype=np.float32)
            if not np.all((session_neural_full == 0) | (session_neural_full == 1)):
                raise ValueError(f"Non-binary rise-extracted trace in {animal}, day {day}")

            session_output_full = discretize_position(positions[day])
            geometry = geometry_vector(blocked[day])

            # Count (but do not silently delete or relocate) source tracking
            # samples in an omitted partition.  A handful lie just across a 25 cm
            # boundary.  Retaining them preserves exact neural/behavioral frame
            # alignment; the count makes this source-data edge case auditable.
            blocked_position_samples = int(
                np.count_nonzero(geometry[session_output_full[0]] != 0)
            )
            # Gross convention mistakes place tens of percent of samples behind
            # walls.  A 5% guard catches those while allowing the source track's
            # boundary excursions (the largest observed case is below 1%).
            if blocked_position_samples / session_output_full.shape[1] > 0.05:
                raise ValueError(
                    f"More than 5% of positions enter blocked bins in {animal}, day {day}; "
                    "check the x/y convention"
                )

            neural_trials = split_trials(session_neural_full, axis=1)
            output_trials = split_trials(session_output_full, axis=1)
            input_trials = [geometry.copy() for _ in range(N_TRIALS)]

            neural.append(neural_trials)
            decoder_input.append(input_trials)
            output.append(output_trials)
            subject_idx.append(animal_index)
            brain_region_idx.append(np.zeros(int(present.sum()), dtype=np.int64))
            session_info.append(
                {
                    "subject": animal,
                    "source_day_index": day,
                    "environment": str(envs[day]),
                    "n_source_frames": int(day_trace.shape[1]),
                    "n_neurons": int(present.sum()),
                    "blocked_position_samples": blocked_position_samples,
                    "trial_lengths": [int(x.shape[1]) for x in neural_trials],
                }
            )

        del wrapped, source, traces, positions, envs, blocked
        gc.collect()

    total_session_neurons = sum(len(x) for x in brain_region_idx)
    if total_session_neurons != 69_744:
        raise ValueError(
            "Registered-neuron total differs from the paper's 69,744 rate maps: "
            f"got {total_session_neurons}"
        )

    cell_names = [f"row_{row}_col_{col}" for row in range(3) for col in range(3)]
    result = {
        "neural": neural,
        "input": decoder_input,
        "output": output,
        "subjects": ANIMALS.copy(),
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": ["CA1"],
        "brain_region_idx": brain_region_idx,
        "input_names": [f"blocked_{name}" for name in cell_names],
        "output_names": ["mouse_position_3x3"],
        "output_values": [cell_names],
        "metadata": {
            "task_description": (
                "Decode the mouse's row-major 3x3 spatial bin from CA1 rise-extracted "
                "calcium events, with the arena's blocked-cell geometry as context."
            ),
            "time_bin_size": 1000.0 / FPS,
            "temporal_alignment_event": "start of each contiguous nominal one-minute window",
            "off_start": 0.0,
            "off_end": 60.0,
            "source_paper": (
                "Lee et al. (2025), Identifying representational structure in CA1 "
                "to benchmark theoretical models of cognitive mapping"
            ),
            "source_sampling_rate_hz": FPS,
            "neural_representation": (
                "binary rising phase of calcium transients supplied by the authors; "
                "thresholded at 2.5 noise-standard-deviations in the source pipeline"
            ),
            "session_definition": "one recording day in one environment",
            "trialization": (
                "each approximately 40-minute session split into 40 contiguous, "
                "nearly equal windows; all source frames retained"
            ),
            "position_binning": (
                "fixed 25 cm bins over the 75x75 cm arena; class = y_bin*3+x_bin"
            ),
            "geometry_encoding": (
                "nine static row-major values per trial; 1=blocked, 0=accessible"
            ),
            "neuron_selection": (
                "all manually curated neurons registered in that day (finite trace); "
                "absent cross-day registrations represented by NaN in the source are removed"
            ),
            "session_info": session_info,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--output", type=Path, default=Path("/app/converted_data.pkl"))
    args = parser.parse_args()

    converted = convert(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    print(f"Writing {args.output} ...", flush=True)
    with temporary.open("wb") as handle:
        pickle.dump(converted, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, args.output)
    size_gib = args.output.stat().st_size / 1024**3
    print(
        f"Wrote {len(converted['neural'])} sessions, "
        f"{sum(map(len, converted['neural']))} trials, "
        f"{size_gib:.2f} GiB to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
