#!/usr/bin/env python3
"""Convert Sosa et al. NWB files to the decoder-compatible trial format.

All NWB access is through pynwb. The neural processing follows the paper's
per-trial maximin dF/F and OASIS procedure rather than using the NWB file's
non-reference ``Deconvolved`` series.
"""

from __future__ import annotations

import argparse
import pickle
import re
import time
from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO
from scipy import ndimage
from suite2p.extraction import dcnv


DATA_ROOT = Path("/app/data")
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
ZONE_BOUNDS = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
SWITCH_DAYS = {3, 5, 7, 8, 10, 12, 14}

INPUT_NAMES = [
    "time_from_trial_start_s",
    "environment",
    "trial_number",
    "previous_trial_outcome",
]
OUTPUT_NAMES = [
    "distance_to_reward_zone",
    "absolute_position",
    "speed",
    "lick",
    "reward_zone_location",
    "reward_outcome",
]
OUTPUT_VALUES = [
    ["< -50 cm", "-50 to -10 cm", "-10 to < 0 cm", "0 cm (inside reward zone)",
     "> 0 to +10 cm", "> +10 to +50 cm", "> +50 cm"],
    ["< 90 cm", "90 to < 180 cm", "180 to < 270 cm", "270 to 360 cm", "> 360 cm"],
    ["< 2 cm/s", "2 to < 10 cm/s", "10 to < 20 cm/s", "20 to 40 cm/s", "> 40 cm/s"],
    ["no lick", "lick"],
    ["A (80-130 cm)", "B (200-250 cm)", "C (320-370 cm)"],
    ["omitted", "rewarded"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process two small switch sessions.")
    parser.add_argument("--show-processing", action="store_true",
                        help="Save processing plots for up to two sessions.")
    return parser.parse_args()


def discover_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.glob("sub-*/*.nwb"))
    if not files:
        raise FileNotFoundError(f"No NWB files found below {DATA_ROOT}")
    if sample:
        wanted = {("m11", 3), ("m11", 5)}
        files = [p for p in files if session_key(p) in wanted]
        if len(files) != 2:
            raise RuntimeError(f"Expected two sample sessions, found {len(files)}")
    return files


def session_key(path: Path) -> tuple[str, int]:
    match = re.search(r"sub-(m\d+)_ses-(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot parse session path: {path}")
    return match.group(1), int(match.group(2))


def scene_from_identifier(identifier: str) -> str:
    return identifier.rstrip("/").split("/")[-1]


def scene_zone_labels(scene: str) -> list[str]:
    labels: list[str] = []
    for token in scene.split("_"):
        if token in ZONE_BOUNDS:
            labels.append(token)
        elif token.startswith("Location") and token[-1:] in ZONE_BOUNDS:
            labels.append(token[-1])
    # Preserve order while eliminating accidental duplicates.
    labels = list(dict.fromkeys(labels))
    if len(labels) not in (1, 2):
        raise ValueError(f"Could not derive one or two reward zones from scene {scene!r}: {labels}")
    return labels


def zones_by_trial(scene: str, n_trials: int) -> list[str]:
    labels = scene_zone_labels(scene)
    if len(labels) == 1:
        return [labels[0]] * n_trials
    if n_trials <= 30:
        raise ValueError(f"Switch scene {scene!r} has only {n_trials} trials")
    return [labels[0]] * 30 + [labels[1]] * (n_trials - 30)


def mode_value(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot compute mode of empty/non-finite values")
    unique, counts = np.unique(values, return_counts=True)
    return float(unique[np.argmax(counts)])


def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Reference nansmooth behavior without propagating missing values."""
    nan_mask = np.isnan(values)
    clean = values.copy()
    clean[nan_mask] = 0.0
    weights = np.ones(values.shape, dtype=np.float64)
    weights[nan_mask] = 0.001
    smooth = ndimage.gaussian_filter1d(clean, sigma, axis=-1)
    smooth_weights = ndimage.gaussian_filter1d(weights, sigma, axis=-1)
    return smooth / smooth_weights


def calculate_reference_dff(
    fluorescence: np.ndarray,
    neuropil: np.ndarray,
    trial_starts: np.ndarray,
    trial_ends: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Reproduce preprocessing.dff for trial periods, before deconvolution."""
    if fluorescence.shape != neuropil.shape:
        raise ValueError(f"F/Fneu shape mismatch: {fluorescence.shape} vs {neuropil.shape}")
    n_cells, n_time = fluorescence.shape
    dff = np.full((n_cells, n_time), np.nan, dtype=np.float64)
    example: dict[str, np.ndarray] = {}

    for trial_idx, (start, stop) in enumerate(zip(trial_starts, trial_ends)):
        # Reference operation: subtract 0.7*Fneu, then add its per-trial mean
        # before baseline division so the denominator is not artificially small.
        f = fluorescence[:, start:stop].astype(np.float64, copy=True)
        fn = neuropil[:, start:stop].astype(np.float64, copy=False)
        corrected = f - 0.7 * fn + 0.7 * np.nanmean(fn, axis=1, keepdims=True)
        baseline = gaussian_smooth(corrected, 15)
        baseline = ndimage.minimum_filter1d(baseline, 300, axis=-1)
        baseline = ndimage.maximum_filter1d(baseline, 300, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            trial_dff = (corrected - baseline) / np.abs(baseline)
        trial_dff = gaussian_smooth(trial_dff, 2)
        dff[:, start:stop] = trial_dff

        if trial_idx == 0:
            example = {
                "raw_f": f[:3].copy(),
                "raw_fneu": fn[:3].copy(),
                "corrected_f": corrected[:3].copy(),
                "baseline": baseline[:3].copy(),
                "dff": trial_dff[:3].copy(),
            }

    return dff, example


def speed_correlations(dff: np.ndarray, speed: np.ndarray) -> np.ndarray:
    valid = np.isfinite(dff[0]) & np.isfinite(speed)
    if np.count_nonzero(valid) < 2:
        raise ValueError("Insufficient finite trial samples for speed correlation")
    x = dff[:, valid]
    y = speed[valid].astype(np.float64, copy=False)
    x_centered = x - np.mean(x, axis=1, keepdims=True)
    y_centered = y - np.mean(y)
    numerator = x_centered @ y_centered
    denominator = np.sqrt(np.sum(x_centered * x_centered, axis=1) * np.sum(y_centered * y_centered))
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator / denominator


def discretize_reward_distance(distance: np.ndarray) -> np.ndarray:
    out = np.empty(distance.shape, dtype=np.int16)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance <= -10)] = 1
    out[(distance > -10) & (distance < 0)] = 2
    out[distance == 0] = 3
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
    return out


def discretize_position(position: np.ndarray) -> np.ndarray:
    out = np.zeros(position.shape, dtype=np.int16)
    out[(position >= 90) & (position < 180)] = 1
    out[(position >= 180) & (position < 270)] = 2
    out[(position >= 270) & (position <= 360)] = 3
    out[position > 360] = 4
    return out


def discretize_speed(speed: np.ndarray) -> np.ndarray:
    out = np.zeros(speed.shape, dtype=np.int16)
    out[(speed >= 2) & (speed < 10)] = 1
    out[(speed >= 10) & (speed < 20)] = 2
    out[(speed >= 20) & (speed <= 40)] = 3
    out[speed > 40] = 4
    return out


def reward_distance(position: np.ndarray, zone: str) -> np.ndarray:
    start, stop = ZONE_BOUNDS[zone]
    return np.where(position < start, position - start,
                    np.where(position > stop, position - stop, 0.0))


def check_discretizer_boundaries() -> None:
    dist = np.array([-51, -50, -10, -9.999, -0.1, 0, 0.1, 10, 10.001, 50, 50.001])
    expected_dist = np.array([0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6])
    if not np.array_equal(discretize_reward_distance(dist), expected_dist):
        raise AssertionError("Reward-distance boundary test failed")
    pos = np.array([89.9, 90, 179.9, 180, 269.9, 270, 360, 360.1])
    if not np.array_equal(discretize_position(pos), [0, 1, 1, 2, 2, 3, 3, 4]):
        raise AssertionError("Position boundary test failed")
    speed = np.array([1.9, 2, 9.9, 10, 19.9, 20, 40, 40.1])
    if not np.array_equal(discretize_speed(speed), [0, 1, 1, 2, 2, 3, 3, 4]):
        raise AssertionError("Speed boundary test failed")


def load_behavior(behavior, common_length: int) -> dict[str, np.ndarray]:
    names = ["position", "speed", "lick", "environment", "trial number", "scanning",
             "reward_zone", "autoreward", "trial_start", "teleport"]
    arrays = {name: np.asarray(behavior[name].data[:common_length]) for name in names}
    arrays["timestamps"] = np.asarray(behavior["position"].timestamps[:common_length])
    return arrays


def load_selected_roi_series(container, roi_ids: np.ndarray, common_length: int,
                             n_segmentation_rows: int) -> np.ndarray:
    """Load selected global segmentation rows from one or more plane series."""
    out = np.empty((len(roi_ids), common_length), dtype=np.float32)
    filled = np.zeros(len(roi_ids), dtype=bool)
    for name in sorted(container.roi_response_series):
        series = container[name]
        region_ids = np.asarray(series.rois.data[:], dtype=int)
        if len(region_ids) != series.data.shape[1]:
            raise ValueError(f"ROI region/data mismatch for {name}")
        if np.any(region_ids < 0) or np.any(region_ids >= n_segmentation_rows):
            raise ValueError(f"Invalid global segmentation index in {name}")
        local = np.flatnonzero(np.isin(region_ids, roi_ids))
        if local.size == 0:
            continue
        global_selected = region_ids[local]
        destination = np.searchsorted(roi_ids, global_selected)
        if not np.array_equal(roi_ids[destination], global_selected):
            raise AssertionError(f"Could not map ROI regions for {name}")
        out[destination] = np.asarray(series.data[:common_length, local]).T
        filled[destination] = True
    if not np.all(filled):
        missing = roi_ids[~filled]
        raise ValueError(f"Response series omit {len(missing)} selected segmentation rows")
    return out


def build_processing_plot(
    session_id: str,
    example: dict[str, np.ndarray],
    example_events: np.ndarray,
    behavior_trial: dict[str, np.ndarray | str | int | float],
    inputs: np.ndarray,
    outputs: np.ndarray,
    n_manual: int,
    n_kept: int,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(8, 1, figsize=(16, 24), sharex=False)
    frame = np.arange(example["raw_f"].shape[1])
    axes[0].plot(frame, example["raw_f"][0], label="raw F")
    axes[0].plot(frame, 0.7 * example["raw_fneu"][0], label="0.7 Fneu", alpha=0.8)
    axes[0].set_ylabel("fluorescence")
    axes[0].set_title("1. NWB fluorescence and neuropil")
    axes[0].legend()

    axes[1].plot(frame, example["corrected_f"][0], label="neuropil corrected")
    axes[1].plot(frame, example["baseline"][0], label="maximin baseline")
    axes[1].set_ylabel("fluorescence")
    axes[1].set_title("2. Per-trial neuropil correction and maximin baseline")
    axes[1].legend()

    axes[2].plot(frame, example["dff"][0], label="smoothed dF/F")
    axes[2].plot(frame, example_events[0], label="OASIS events")
    axes[2].set_ylabel("activity")
    axes[2].set_title(f"3. dF/F and deconvolution; curated {n_manual}, after interneuron filter {n_kept}")
    axes[2].legend()

    t = inputs[0]
    pos = np.asarray(behavior_trial["position"])
    zone = str(behavior_trial["zone"])
    z0, z1 = ZONE_BOUNDS[zone]
    axes[3].plot(t, pos, color="black", label="position")
    axes[3].axhspan(z0, z1, color="gold", alpha=0.3, label=f"zone {zone}")
    axes[3].set_ylabel("cm")
    axes[3].set_title("4. Trial-start alignment and active reward zone")
    axes[3].legend()

    for i, name in enumerate(INPUT_NAMES):
        axes[4].plot(t, inputs[i], label=name)
    axes[4].set_title("5. Decoder inputs (time-varying and repeated trial context)")
    axes[4].legend(ncol=2)

    speed = np.asarray(behavior_trial["speed"])
    lick = np.asarray(behavior_trial["lick"])
    axes[5].plot(t, speed, label="continuous speed (cm/s)")
    axes[5].plot(t, (lick > 0) * 10, label="binary lick ×10", alpha=0.7)
    axes[5].set_title("6. Source behavior")
    axes[5].legend()

    axes[6].step(t, outputs[0], where="mid", label="distance bin")
    axes[6].step(t, outputs[1] + 8, where="mid", label="position bin +8")
    axes[6].step(t, outputs[2] + 14, where="mid", label="speed bin +14")
    axes[6].set_title("7. Continuous-output discretization")
    axes[6].legend(ncol=3)

    axes[7].step(t, outputs[3], where="mid", label="lick")
    axes[7].step(t, outputs[4] + 2, where="mid", label="zone +2")
    axes[7].step(t, outputs[5] + 6, where="mid", label="reward +6")
    axes[7].set_title("8. Binary/per-trial categorical outputs")
    axes[7].set_xlabel("seconds from trial start")
    axes[7].legend(ncol=3)

    fig.suptitle(session_id)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id)
    path = Path("/app") / f"processing_{safe}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def convert_session(path: Path, show_processing: bool = False) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    subject_from_path, day = session_key(path)
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        subject = nwb.subject.subject_id
        if subject != subject_from_path:
            raise ValueError(f"Subject mismatch in {path}: {subject} vs {subject_from_path}")
        behavior = nwb.processing["behavior"]["BehavioralTimeSeries"]
        ophys = nwb.processing["ophys"]
        fluorescence_container = ophys["Fluorescence"]
        neuropil_container = ophys["Neuropil"]
        segmentation = ophys["ImageSegmentation"]["PlaneSegmentation"]

        behavior_len = len(behavior["position"].data)
        fluorescence_series = list(fluorescence_container.roi_response_series.values())
        neuropil_series = list(neuropil_container.roi_response_series.values())
        if not fluorescence_series or not neuropil_series:
            raise ValueError(f"Missing fluorescence or neuropil response series in {path}")
        common_length = min([behavior_len] + [x.data.shape[0] for x in fluorescence_series] +
                            [x.data.shape[0] for x in neuropil_series])
        b = load_behavior(behavior, common_length)
        starts = np.flatnonzero(b["trial_start"] > 0)
        stops = np.flatnonzero(b["teleport"] > 0)
        if len(starts) != len(stops) or len(starts) < 2:
            raise ValueError(f"Invalid trial event counts in {path}: {len(starts)}/{len(stops)}")
        if np.any(stops <= starts):
            raise ValueError(f"Non-positive trial interval in {path}")

        iscell = np.asarray(segmentation["iscell"].data[:])
        if iscell.ndim != 2 or iscell.shape[1] < 1:
            raise ValueError(f"Unexpected iscell shape in {path}: {iscell.shape}")
        manual_ids = np.flatnonzero(iscell[:, 0] == 1)
        if manual_ids.size == 0:
            raise ValueError(f"No manually curated cells in {path}")

        # pynwb-backed DynamicTableRegion and RoiResponseSeries access only.
        fluorescence = load_selected_roi_series(
            fluorescence_container, manual_ids, common_length, len(segmentation))
        neuropil = load_selected_roi_series(
            neuropil_container, manual_ids, common_length, len(segmentation))
        dff, plot_example = calculate_reference_dff(fluorescence, neuropil, starts, stops)
        correlations = speed_correlations(dff, b["speed"])
        keep = ~(correlations > 0.5)
        kept_ids = manual_ids[keep]
        dff = dff[keep]
        if dff.shape[0] == 0:
            raise ValueError(f"Interneuron filter removed every cell in {path}")

        reward_times = np.asarray(behavior["Reward"].timestamps[:])
        timestamps = b["timestamps"]
        outcomes = np.array([
            np.any((reward_times >= timestamps[start]) & (reward_times < timestamps[stop]))
            for start, stop in zip(starts, stops)
        ], dtype=np.int16)
        scene = scene_from_identifier(nwb.identifier)
        zones = zones_by_trial(scene, len(starts))

        neural_trials: list[np.ndarray] = []
        input_trials: list[np.ndarray] = []
        output_trials: list[np.ndarray] = []
        excluded_trials: list[dict] = []
        plot_payload = None

        for trial_idx, (start, stop, zone) in enumerate(zip(starts, stops, zones)):
            reason = None
            if stop - start < 2:
                reason = "fewer than two samples"
            elif not np.all(b["scanning"][start:stop] == 1):
                reason = "outside valid scanning period"
            required = np.vstack((b["position"][start:stop], b["speed"][start:stop],
                                  b["lick"][start:stop], timestamps[start:stop]))
            if not np.all(np.isfinite(required)):
                reason = "non-finite required behavior"
            lick_bad_fraction = float(np.mean(b["lick"][start:stop] > 2))
            if lick_bad_fraction > 0.30:
                reason = "corrupt lick sensor (>30% frames with count >2)"
            if reason is not None:
                excluded_trials.append({"original_trial_index": trial_idx, "reason": reason})
                continue

            time_from_start = timestamps[start:stop] - timestamps[start]
            environment = mode_value(b["environment"][start:stop])
            trial_number = mode_value(b["trial number"][start:stop])
            previous_outcome = int(outcomes[trial_idx - 1]) if trial_idx > 0 else 0
            inputs = np.vstack((
                time_from_start,
                np.full(stop - start, environment),
                np.full(stop - start, trial_number),
                np.full(stop - start, previous_outcome),
            )).astype(np.float32)

            position = b["position"][start:stop]
            speed = b["speed"][start:stop]
            lick = b["lick"][start:stop]
            distance = reward_distance(position, zone)
            outputs = np.vstack((
                discretize_reward_distance(distance),
                discretize_position(position),
                discretize_speed(speed),
                (lick > 0).astype(np.int16),
                np.full(stop - start, "ABC".index(zone), dtype=np.int16),
                np.full(stop - start, outcomes[trial_idx], dtype=np.int16),
            )).astype(np.int16, copy=False)

            trial_dff = dff[:, start:stop]
            if not np.all(np.isfinite(trial_dff)):
                raise ValueError(f"Non-finite dF/F in {path}, trial {trial_idx}")
            events = dcnv.oasis(trial_dff, 2000, 0.7, FRAME_RATE_HZ).astype(np.float32)
            events[events < 0] = 0
            if not np.all(np.isfinite(events)):
                raise ValueError(f"Non-finite OASIS events in {path}, trial {trial_idx}")
            if events.shape[1] != inputs.shape[1] or inputs.shape[1] != outputs.shape[1]:
                raise AssertionError(f"Alignment shape mismatch in {path}, trial {trial_idx}")

            neural_trials.append(events)
            input_trials.append(inputs)
            output_trials.append(outputs)
            if plot_payload is None:
                plot_payload = (events[:3].copy(), {
                    "position": position.copy(), "speed": speed.copy(), "lick": lick.copy(),
                    "zone": zone, "original_trial_index": trial_idx,
                }, inputs.copy(), outputs.copy())

        if len(neural_trials) < 2:
            raise ValueError(f"Fewer than two retained trials in {path}")
        if len(kept_ids) != neural_trials[0].shape[0]:
            raise AssertionError("Retained cell count mismatch")

        session_id = f"sub-{subject}_ses-{day:02d}"
        plot_path = None
        if show_processing and plot_payload is not None:
            plot_events, plot_behavior, plot_inputs, plot_outputs = plot_payload
            # The example traces precede cell filtering. Select rows that survive when possible.
            example_rows = np.flatnonzero(keep[:min(3, len(keep))])
            if len(example_rows) == 0:
                example_rows = np.arange(min(3, plot_example["dff"].shape[0]))
            for key in plot_example:
                plot_example[key] = plot_example[key][example_rows]
            plot_events = plot_events[:len(example_rows)]
            plot_path = build_processing_plot(
                session_id, plot_example, plot_events, plot_behavior, plot_inputs, plot_outputs,
                len(manual_ids), len(kept_ids),
            )

        session_meta = {
            "session_id": session_id,
            "source_file": str(path),
            "nwb_identifier": nwb.identifier,
            "scene": scene,
            "task_day": day,
            "is_switch_day": day in SWITCH_DAYS,
            "source_timepoints_behavior": behavior_len,
            "source_timepoints_neural": int(min(x.data.shape[0] for x in fluorescence_series)),
            "common_timepoints": common_length,
            "source_trials": len(starts),
            "retained_trials": len(neural_trials),
            "excluded_trials": excluded_trials,
            "candidate_rois": len(segmentation),
            "manual_cells": len(manual_ids),
            "excluded_speed_correlated_cells": int(np.sum(~keep)),
            "retained_cells": len(kept_ids),
            "imaging_planes": sorted(np.unique(np.asarray(segmentation["planeIdx"].data[:])).astype(int).tolist()),
            "reward_events_total": len(reward_times),
            "rewarded_trials": int(np.sum(outcomes)),
            "unmapped_reward_events": int(len(reward_times) - np.sum(outcomes)),
            "processing_plot": str(plot_path) if plot_path else None,
        }

    elapsed = time.perf_counter() - t0
    session_meta["conversion_seconds"] = elapsed
    converted = {
        "subject": subject,
        "neural": neural_trials,
        "input": input_trials,
        "output": output_trials,
        "brain_region_idx": np.zeros(len(kept_ids), dtype=np.int16),
    }
    return converted, session_meta


def validate_converted(data: dict) -> None:
    n_sessions = len(data["neural"])
    if not (n_sessions == len(data["input"]) == len(data["output"])):
        raise AssertionError("Top-level session lengths differ")
    if len(data["subject_idx"]) != n_sessions or len(data["brain_region_idx"]) != n_sessions:
        raise AssertionError("Session metadata lengths differ")
    for s in range(n_sessions):
        if len(data["neural"][s]) < 2:
            raise AssertionError(f"Session {s} has fewer than two trials")
        if not (len(data["neural"][s]) == len(data["input"][s]) == len(data["output"][s])):
            raise AssertionError(f"Session {s} trial-list lengths differ")
        n_cells = len(data["brain_region_idx"][s])
        for tr, (neural, inputs, outputs) in enumerate(zip(
                data["neural"][s], data["input"][s], data["output"][s])):
            if neural.shape[0] != n_cells or inputs.shape[0] != 4 or outputs.shape[0] != 6:
                raise AssertionError(f"Session {s}, trial {tr}: dimension mismatch")
            if not (neural.shape[1] == inputs.shape[1] == outputs.shape[1]):
                raise AssertionError(f"Session {s}, trial {tr}: time mismatch")
            if not (np.all(np.isfinite(neural)) and np.all(np.isfinite(inputs)) and
                    np.all(np.isfinite(outputs))):
                raise AssertionError(f"Session {s}, trial {tr}: non-finite values")
            for out_idx, names in enumerate(OUTPUT_VALUES):
                values = outputs[out_idx]
                if values.min() < 0 or values.max() >= len(names):
                    raise AssertionError(f"Session {s}, trial {tr}, output {out_idx}: invalid class")


def main() -> None:
    args = parse_args()
    check_discretizer_boundaries()
    files = discover_files(args.sample)
    print(f"Mode: {'sample' if args.sample else 'full'}; sessions: {len(files)}", flush=True)
    print(f"Reference time bin: {TIME_BIN_MS:.9f} ms", flush=True)

    converted_sessions: list[dict] = []
    session_info: list[dict] = []
    total_start = time.perf_counter()
    for idx, path in enumerate(files):
        print(f"[{idx + 1}/{len(files)}] {path}", flush=True)
        converted, metadata = convert_session(path, show_processing=args.show_processing and idx < 2)
        converted_sessions.append(converted)
        session_info.append(metadata)
        elapsed = time.perf_counter() - total_start
        projected = elapsed / (idx + 1) * len(files)
        print(
            f"  kept {metadata['retained_trials']}/{metadata['source_trials']} trials, "
            f"{metadata['retained_cells']}/{metadata['manual_cells']} curated cells; "
            f"session {metadata['conversion_seconds']:.2f}s; projected total {projected / 60:.2f} min",
            flush=True,
        )

    subjects = sorted({x["subject"] for x in converted_sessions}, key=lambda x: int(x[1:]))
    subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
    data = {
        "neural": [x["neural"] for x in converted_sessions],
        "input": [x["input"] for x in converted_sessions],
        "output": [x["output"] for x in converted_sessions],
        "subjects": subjects,
        "subject_idx": np.asarray([subject_lookup[x["subject"]] for x in converted_sessions], dtype=np.int16),
        "brain_regions": ["CA1"],
        "brain_region_idx": [x["brain_region_idx"] for x in converted_sessions],
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "output_values": OUTPUT_VALUES,
        "metadata": {
            "task_description": (
                "Head-fixed mice navigate a 450-cm virtual corridor with hidden reward zones A/B/C, "
                "random reward omissions, seven reward-location switches, and one environment switch. "
                "Outputs classify reward-zone distance, corridor position, speed, licking, zone identity, "
                "and reward outcome from CA1 calcium events plus trial context."
            ),
            "time_bin_size": TIME_BIN_MS,
            "time_bin_size_units": "ms",
            "temporal_alignment_event": "trial_start (entry into the 0-cm virtual corridor)",
            "off_start": 0.0,
            "off_end": None,
            "neural_signal": "Per-trial maximin dF/F, Gaussian-smoothed and OASIS-deconvolved calcium events",
            "neural_processing": (
                "Manual Suite2p cells; 0.7 neuropil subtraction; per-trial 300-sample maximin baseline; "
                "dF/F Gaussian sigma=2 samples; OASIS tau=0.7 at 15.5078125 Hz; exclude dF/F-speed r>0.5"
            ),
            "trial_interval": "NWB trial_start frame inclusive to teleport frame exclusive",
            "trial_filter": "Exclude trials where >30% of frames have cumulative lick count >2 or invalid synchronized data",
            "reward_zone_bounds_cm": {key: list(value) for key, value in ZONE_BOUNDS.items()},
            "source": "DANDI:001361/0.251124.0550; Sosa, Plitt & Giocomo (2025)",
            "session_info": session_info,
        },
    }
    validate_converted(data)

    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    write_start = time.perf_counter()
    with args.outpicklefile.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    total_elapsed = time.perf_counter() - total_start
    print(f"Wrote {args.outpicklefile} ({args.outpicklefile.stat().st_size / 1e9:.3f} GB) "
          f"in {time.perf_counter() - write_start:.2f}s", flush=True)
    print(f"Conversion complete in {total_elapsed / 60:.2f} min", flush=True)
    print(f"Retained sessions={len(data['neural'])}, trials={sum(map(len, data['neural']))}, "
          f"neurons={sum(map(len, data['brain_region_idx']))}", flush=True)


if __name__ == "__main__":
    main()
