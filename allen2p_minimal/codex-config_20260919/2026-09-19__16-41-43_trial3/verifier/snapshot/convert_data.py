#!/usr/bin/env python3
"""Convert Allen Visual Behavior 2P NWB files to the decoder format.

The conversion intentionally uses the AllenSDK's public NWB reader.  Important
choices follow the technical whitepaper and the analysis paper:

* NWB files and cells have already passed Allen ophys/ROI QC.
* The neural signal is the inferred calcium-event trace, not dF/F.  This avoids
  carrying the slow GCaMP decay into later stimulus epochs, as in the paper.
* Only active Visual Behavior experiments are used.  Passive replay sessions
  have synthetic go/catch labels but no genuine choice/outcome.
* Each imaging plane (``ophys_experiment_id``) is one decoder session, matching
  the plane-wise decoding analysis and avoiding artificial interpolation among
  interleaved Multiscope planes.
* Experiment-defined go and catch trials are retained; aborted and
  auto-rewarded trials are excluded.
* All streams are placed on a common 100 ms grid anchored at each trial start.
  Neural samples are selected by nearest synchronized ophys timestamp.  Running
  speed and blink-filtered pupil diameter are linearly interpolated to that
  same grid.

Run with no arguments to create ``/app/converted_data.pkl``.
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
    BehaviorOphysExperiment,
)


APP_DIR = Path(__file__).resolve().parent
DATASET_DIR = APP_DIR / "data" / "visual-behavior-ophys-1.1.0"
NWB_DIR = DATASET_DIR / "behavior_ophys_experiments"
EXPERIMENT_TABLE = DATASET_DIR / "project_metadata" / "ophys_experiment_table.csv"
DEFAULT_OUTPUT = APP_DIR / "converted_data.pkl"

# A common grid is required because Scientifica and Multiscope planes were
# acquired at about 31 Hz and 11 Hz, respectively.  Ten Hz is still finer than
# the roughly 200 ms effective resolution of the inferred events.
BIN_SEC = 0.100

# Global labels make a category mean the same thing in every session.  The two
# counterbalanced image sets are disjoint.  Class zero is the gray/no-image
# state, including omitted flashes (an omission is not an image).
IMAGE_NAMES = [
    "gray",
    "im000", "im031", "im035", "im045", "im054", "im073", "im075", "im106",
    "im061", "im062", "im063", "im065", "im066", "im069", "im077", "im085",
]
IMAGE_TO_CLASS = {name: idx for idx, name in enumerate(IMAGE_NAMES)}

OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]


def _local_experiment_ids() -> set[int]:
    """Return experiment ids for NWBs actually present in the supplied data."""
    ids: set[int] = set()
    for path in NWB_DIR.glob("behavior_ophys_experiment_*.nwb"):
        match = re.search(r"_(\d+)\.nwb$", path.name)
        if match:
            ids.add(int(match.group(1)))
    return ids


def select_experiments(max_sessions: int | None = None) -> pd.DataFrame:
    """Select local, QC-passed, active Visual Behavior imaging experiments."""
    table = pd.read_csv(EXPERIMENT_TABLE, index_col="ophys_experiment_id")
    local_ids = _local_experiment_ids()
    table = table.loc[table.index.intersection(local_ids)].copy()

    # Passive sessions replay an earlier stimulus sequence after water delivery;
    # the spout is retracted.  Their SDK trial table mechanically calls every go
    # a miss and every catch a correct rejection, which is not an animal outcome.
    active = table[table["behavior_type"].eq("active_behavior") & ~table["passive"]]
    active = active.sort_index()
    if max_sessions is not None:
        active = active.iloc[:max_sessions]
    if active.empty:
        raise RuntimeError("No local active Visual Behavior ophys experiments found")
    return active


def _nearest_indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Indices of source samples nearest to each target time."""
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 0, len(source_times) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(target_times - source_times[left]) <= np.abs(
        source_times[right] - target_times
    )
    return np.where(choose_left, left, right)


def _interp_valid(times: np.ndarray, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Linear interpolation after omitting NaN/inf samples."""
    valid = np.isfinite(times) & np.isfinite(values)
    if valid.sum() < 2:
        raise ValueError("fewer than two finite samples available for interpolation")
    x = times[valid]
    y = values[valid]
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    # np.interp uses the nearest endpoint outside the measured range.  Trial
    # samples normally lie inside the behavior/eye recordings; endpoint use is
    # retained as a safe treatment of a few synchronization-edge samples.
    return np.interp(targets, x, y).astype(np.float32, copy=False)


def _percentile_classes(values: np.ndarray) -> np.ndarray:
    """Discretize finite values with the session's 20/40/60/80 percentiles."""
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite value remained before percentile binning")
    cuts = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
    # side='right' gives classes 0..4 and a deterministic treatment of ties.
    return np.searchsorted(cuts, values, side="right").astype(np.uint8)


def _trial_outcome(row: pd.Series) -> int:
    flags = np.asarray([bool(row[name]) for name in OUTCOME_NAMES])
    if flags.sum() != 1:
        raise ValueError(
            f"retained trial {row.name} does not have exactly one outcome: {flags.tolist()}"
        )
    return int(np.flatnonzero(flags)[0])


def _stimulus_labels(
    centers: np.ndarray,
    presentations: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return image identity and one-bin image-change labels at bin centers."""
    image = np.zeros(len(centers), dtype=np.uint8)
    changed = np.zeros(len(centers), dtype=np.uint8)
    if len(centers) == 0:
        return image, changed

    # Keep actual image flashes only.  NaN is gray and "omitted" is an expected
    # flash deliberately replaced by continued gray.
    overlap = presentations[
        (presentations["start_time"] < centers[-1] + BIN_SEC / 2)
        & (presentations["end_time"] > centers[0] - BIN_SEC / 2)
    ]
    for _, stim in overlap.iterrows():
        name = stim["image_name"]
        if pd.notna(name) and name != "omitted":
            if name not in IMAGE_TO_CLASS:
                raise ValueError(f"unknown Visual Behavior image name {name!r}")
            on = (centers >= float(stim["start_time"])) & (
                centers < float(stim["end_time"])
            )
            image[on] = IMAGE_TO_CLASS[name]

        # A change is an event, not a 250 ms state: label the first common time
        # bin whose center is on or after its display-lag-corrected onset.
        if bool(stim.get("is_change", False)):
            idx = int(np.searchsorted(centers, float(stim["start_time"]), side="left"))
            if idx < len(changed):
                changed[idx] = 1
    return image, changed


def convert_experiment(experiment_id: int) -> dict:
    """Load and convert one imaging plane/session."""
    path = NWB_DIR / f"behavior_ophys_experiment_{experiment_id}.nwb"
    dataset = BehaviorOphysExperiment.from_nwb_path(str(path))

    events = np.vstack(dataset.events["events"].to_numpy()).astype(np.float32, copy=False)
    ophys_times = np.asarray(dataset.ophys_timestamps, dtype=np.float64)
    if events.shape[1] != len(ophys_times):
        raise ValueError(
            f"event/timestamp length mismatch: {events.shape[1]} != {len(ophys_times)}"
        )
    if events.shape[0] == 0:
        raise ValueError("no QC-passed cells")

    trials = dataset.trials
    keep = (
        (trials["go"].astype(bool) | trials["catch"].astype(bool))
        & ~trials["aborted"].astype(bool)
        & ~trials["auto_rewarded"].astype(bool)
    )
    trials = trials.loc[keep].copy()
    if len(trials) < 2:
        raise ValueError(f"only {len(trials)} retained trials")

    running = dataset.running_speed
    eye = dataset.eye_tracking
    if eye is None:
        raise ValueError("eye tracking is unavailable")

    # SDK pupil_area is pi * max(ellipse radius)^2 after blink/outlier removal;
    # converting it to a diameter is monotonic (and therefore preserves the
    # requested percentile classes).
    pupil_area = eye["pupil_area"].to_numpy(dtype=np.float64)
    pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)

    trial_centers: list[np.ndarray] = []
    outcomes: list[int] = []
    for _, row in trials.iterrows():
        duration = float(row["stop_time"] - row["start_time"])
        n_bins = int(np.floor(duration / BIN_SEC))
        if n_bins < 1:
            continue
        centers = float(row["start_time"]) + (np.arange(n_bins) + 0.5) * BIN_SEC
        trial_centers.append(centers)
        outcomes.append(_trial_outcome(row))

    if len(trial_centers) < 2:
        raise ValueError(f"only {len(trial_centers)} nonempty retained trials")
    all_centers = np.concatenate(trial_centers)

    run_values = _interp_valid(
        running["timestamps"].to_numpy(dtype=np.float64),
        running["speed"].to_numpy(dtype=np.float64),
        all_centers,
    )
    pupil_values = _interp_valid(
        eye["timestamps"].to_numpy(dtype=np.float64),
        pupil_diameter,
        all_centers,
    )
    run_class = _percentile_classes(run_values)
    pupil_class = _percentile_classes(pupil_values)

    neural_trials: list[np.ndarray] = []
    input_trials: list[np.ndarray] = []
    output_trials: list[np.ndarray] = []
    cursor = 0
    presentations = dataset.stimulus_presentations
    for centers, outcome in zip(trial_centers, outcomes):
        n_bins = len(centers)
        nearest = _nearest_indices(ophys_times, centers)
        neural_trials.append(events[:, nearest].astype(np.float32, copy=False))
        input_trials.append(np.empty((0, n_bins), dtype=np.float32))

        image, change = _stimulus_labels(centers, presentations)
        outcome_row = np.full(n_bins, outcome, dtype=np.uint8)
        output_trials.append(
            np.vstack(
                [
                    image,
                    change,
                    run_class[cursor : cursor + n_bins],
                    pupil_class[cursor : cursor + n_bins],
                    outcome_row,
                ]
            ).astype(np.uint8, copy=False)
        )
        cursor += n_bins

    meta = dataset.metadata
    result = {
        "neural": neural_trials,
        "input": input_trials,
        "output": output_trials,
        "mouse_id": str(meta["mouse_id"]),
        "brain_region": str(meta["targeted_structure"]),
        "n_neurons": int(events.shape[0]),
        "session_info": {
            "ophys_experiment_id": int(experiment_id),
            "ophys_session_id": int(meta["ophys_session_id"]),
            "behavior_session_id": int(meta["behavior_session_id"]),
            "ophys_container_id": int(meta["ophys_container_id"]),
            "mouse_id": str(meta["mouse_id"]),
            "session_type": str(meta["session_type"]),
            "targeted_structure": str(meta["targeted_structure"]),
            "imaging_depth_um": int(meta["imaging_depth"]),
            "cre_line": str(meta["cre_line"]),
            "indicator": str(meta["indicator"]),
            "equipment_name": str(meta["equipment_name"]),
            "source_ophys_frame_rate_hz": float(meta["ophys_frame_rate"]),
            "n_neurons": int(events.shape[0]),
            "n_trials": len(neural_trials),
        },
    }
    del dataset, events
    gc.collect()
    return result


def build_dataset(max_sessions: int | None = None) -> dict:
    experiments = select_experiments(max_sessions=max_sessions)
    converted: list[dict] = []
    skipped: list[dict] = []
    total = len(experiments)
    for number, experiment_id in enumerate(experiments.index, start=1):
        print(f"[{number}/{total}] converting experiment {experiment_id}", flush=True)
        try:
            converted.append(convert_experiment(int(experiment_id)))
        except Exception as exc:
            # Missing pupil data prevents constructing a required decoder output;
            # such a session is unusable rather than safely imputable.  Preserve
            # the exact exclusion and reason in metadata.
            print(f"  skipped: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            skipped.append(
                {
                    "ophys_experiment_id": int(experiment_id),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    if not converted:
        raise RuntimeError("No experiments could be converted")

    subjects = sorted({session["mouse_id"] for session in converted})
    subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
    # Preserve conventional V1-before-LM order when both are present.
    present_regions = {session["brain_region"] for session in converted}
    brain_regions = [r for r in ["VISp", "VISl"] if r in present_regions]
    brain_regions.extend(sorted(present_regions - set(brain_regions)))
    region_lookup = {region: idx for idx, region in enumerate(brain_regions)}

    return {
        "neural": [session["neural"] for session in converted],
        "input": [session["input"] for session in converted],
        "output": [session["output"] for session in converted],
        "subjects": subjects,
        "subject_idx": np.asarray(
            [subject_lookup[session["mouse_id"]] for session in converted], dtype=np.int64
        ),
        "brain_regions": brain_regions,
        "brain_region_idx": [
            np.full(
                session["n_neurons"],
                region_lookup[session["brain_region"]],
                dtype=np.int64,
            )
            for session in converted
        ],
        "input_names": [],
        "output_names": [
            "image_identity",
            "image_change",
            "running_speed_percentile",
            "pupil_diameter_percentile",
            "trial_outcome",
        ],
        "output_values": [
            IMAGE_NAMES,
            ["no_change", "change"],
            ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
            ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
            OUTCOME_NAMES,
        ],
        "metadata": {
            "task_description": (
                "Allen Visual Behavior change-detection task; decode flashed image "
                "identity, image-change events, running and pupil quintiles, and "
                "hit/miss/false-alarm/correct-rejection trial outcome from inferred "
                "calcium events."
            ),
            "time_bin_size": BIN_SEC * 1000.0,
            "temporal_alignment_event": (
                "experiment-defined trial start on the synchronized ophys clock; "
                "100 ms bin centers begin 50 ms after trial start"
            ),
            "off_start": 0.0,
            "off_end": None,
            "neural_signal": "AllenSDK inferred calcium events",
            "resampling": (
                "nearest ophys event sample at each 100 ms center; linear interpolation "
                "of running speed and blink-filtered pupil diameter"
            ),
            "trial_filter": (
                "go or catch; exclude aborted and auto_rewarded; active_behavior only"
            ),
            "percentile_scope": "separately within each session over retained trial bins",
            "image_background_class": (
                "gray includes inter-stimulus gray and omitted flashes"
            ),
            "session_unit": "one QC-passed ophys experiment (imaging plane)",
            "source_dataset": "Allen Visual Behavior Ophys manifest 1.1.0",
            "session_info": [session["session_info"] for session in converted],
            "skipped_sessions": skipped,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="convert only the first N selected experiments (for a smoke test)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    data = build_dataset(max_sessions=args.max_sessions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = args.output.with_name(args.output.name + ".tmp")
    with temp_output.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp_output, args.output)
    size_gib = args.output.stat().st_size / (1024**3)
    print(
        f"wrote {args.output} ({size_gib:.2f} GiB, "
        f"{len(data['neural'])} sessions)",
        flush=True,
    )


if __name__ == "__main__":
    main()
