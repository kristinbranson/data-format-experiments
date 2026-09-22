#!/usr/bin/env python3
"""Convert Sosa et al. CA1 NWB data to the supplied decoder format.

All NWB access is through PyNWB.  Neural preprocessing reproduces the paper's
trial-wise neuropil correction, maximin dF/F, smoothing, and OASIS event
extraction rather than using the distinct Suite2p ``Deconvolved`` NWB export.
"""

from __future__ import annotations

import argparse
import pickle
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d
from suite2p.extraction import dcnv


DATA_ROOT = Path("/app/data")
EFFECTIVE_FS_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / EFFECTIVE_FS_HZ
NEUROPIL_COEF = np.float32(0.7)
OASIS_TAU_S = 0.7
LICK_ERROR_FRACTION = 0.30
ZONE_BOUNDS = {
    "A": (80.0, 130.0),
    "B": (200.0, 250.0),
    "C": (320.0, 370.0),
}
ZONE_TO_CLASS = {"A": 0, "B": 1, "C": 2}

INPUT_NAMES = [
    "time_from_trial_start_s",
    "environment_type",
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
    ["<-50 cm", "-50 to <-10 cm", "-10 to <0 cm", "0 cm (inside zone)",
     ">0 to 10 cm", ">10 to 50 cm", ">50 cm"],
    ["<90 cm", "90 to <180 cm", "180 to <270 cm", "270 to 360 cm", ">360 cm"],
    ["<2 cm/s", "2 to <10 cm/s", "10 to <20 cm/s", "20 to 40 cm/s", ">40 cm/s"],
    ["no lick", "lick"],
    ["A", "B", "C"],
    ["omitted", "rewarded"],
]


def natural_session_key(path: Path) -> tuple[int, int]:
    """Sort paths numerically by mouse and session rather than lexically."""
    match = re.search(r"sub-m(\d+)_ses-(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot parse mouse/session from {path}")
    return int(match.group(1)), int(match.group(2))


def discover_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.glob("sub-*/*.nwb"), key=natural_session_key)
    if not files:
        raise FileNotFoundError(f"No NWB files found below {DATA_ROOT}")
    if not sample:
        return files

    # Two day-8 environment switches jointly cover ENV1/ENV2 and zones A/B/C.
    wanted = {
        "sub-m3_ses-08_behavior+ophys.nwb",
        "sub-m12_ses-08_behavior+ophys.nwb",
    }
    chosen = [path for path in files if path.name in wanted]
    if len(chosen) != 2:
        raise RuntimeError(f"Expected two representative sample sessions, found {chosen}")
    return chosen


def parse_zone_schedule(scene: str, n_trials: int) -> np.ndarray:
    """Return A/B/C label per raw trial from the scene identifier."""
    if "_to_" in scene:
        left, right = scene.split("_to_", 1)
        left_match = re.search(r"([ABC])$", left)
        right_match = re.search(r"([ABC])$", right)
        if left_match is None or right_match is None:
            raise ValueError(f"Cannot parse switching reward zones from {scene!r}")
        labels = [left_match.group(1)] * min(30, n_trials)
        labels.extend([right_match.group(1)] * max(0, n_trials - 30))
        return np.asarray(labels)

    match = re.search(r"Location([ABC])$", scene)
    if match is None:
        raise ValueError(f"Cannot parse fixed reward zone from {scene!r}")
    return np.full(n_trials, match.group(1))


def distance_classes(distance: np.ndarray) -> np.ndarray:
    """Discretize signed distance to the nearest point in the reward zone."""
    out = np.full(distance.shape, 6, dtype=np.int8)
    out[distance < -50.0] = 0
    out[(distance >= -50.0) & (distance < -10.0)] = 1
    out[(distance >= -10.0) & (distance < 0.0)] = 2
    out[distance == 0.0] = 3
    out[(distance > 0.0) & (distance <= 10.0)] = 4
    out[(distance > 10.0) & (distance <= 50.0)] = 5
    return out


def position_classes(position: np.ndarray) -> np.ndarray:
    out = np.full(position.shape, 4, dtype=np.int8)
    out[position < 90.0] = 0
    out[(position >= 90.0) & (position < 180.0)] = 1
    out[(position >= 180.0) & (position < 270.0)] = 2
    out[(position >= 270.0) & (position <= 360.0)] = 3
    return out


def speed_classes(speed: np.ndarray) -> np.ndarray:
    out = np.full(speed.shape, 4, dtype=np.int8)
    out[speed < 2.0] = 0
    out[(speed >= 2.0) & (speed < 10.0)] = 1
    out[(speed >= 10.0) & (speed < 20.0)] = 2
    out[(speed >= 20.0) & (speed <= 40.0)] = 3
    return out


def check_discretization_boundaries() -> None:
    d = np.asarray([-51, -50, -10, -0.1, 0, 0.1, 10, 10.1, 50, 50.1])
    assert np.array_equal(distance_classes(d), [0, 1, 2, 2, 3, 4, 4, 5, 5, 6])
    p = np.asarray([89.9, 90, 179.9, 180, 269.9, 270, 360, 360.1])
    assert np.array_equal(position_classes(p), [0, 1, 1, 2, 2, 3, 3, 4])
    v = np.asarray([1.9, 2, 9.9, 10, 19.9, 20, 40, 40.1])
    assert np.array_equal(speed_classes(v), [0, 1, 1, 2, 2, 3, 3, 4])


def reward_outcomes(reward_times: np.ndarray, timestamps: np.ndarray,
                    starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Map sparse delivery events into binary track-trial outcomes."""
    reward_times = np.sort(np.asarray(reward_times, dtype=np.float64))
    left = np.searchsorted(reward_times, timestamps[starts], side="left")
    right = np.searchsorted(reward_times, timestamps[stops], side="left")
    return (right > left).astype(np.int8)


def trial_dff_and_events(fluorescence: np.ndarray, neuropil: np.ndarray,
                         starts: np.ndarray, stops: np.ndarray,
                         capture_trace: bool = False) -> tuple[np.ndarray, np.ndarray, dict | None]:
    """Reproduce reference trial-wise dF/F and OASIS event processing.

    Arrays are time x neuron.  All operations use only track samples, matching
    ``preprocessing.dff`` after translating its legacy one-based boundaries to
    the NWB event-flag indices.
    """
    if fluorescence.shape != neuropil.shape:
        raise ValueError(f"F/Fneu shape mismatch: {fluorescence.shape} vs {neuropil.shape}")
    dff = np.full(fluorescence.shape, np.nan, dtype=np.float32)
    trace = None
    trace_trial = min(5, len(starts) - 1)

    for trial_idx, (start, stop) in enumerate(zip(starts, stops)):
        f_trial = fluorescence[start:stop]
        n_trial = neuropil[start:stop]
        corrected = f_trial - NEUROPIL_COEF * n_trial
        corrected += NEUROPIL_COEF * np.mean(n_trial, axis=0, keepdims=True)

        baseline = gaussian_filter1d(corrected, 15, axis=0)
        baseline = minimum_filter1d(baseline, 300, axis=0)
        baseline = maximum_filter1d(baseline, 300, axis=0)
        denom = np.abs(baseline)
        if np.any(denom == 0):
            raise ValueError(f"Zero fluorescence baseline in trial {trial_idx}")
        trial_dff = gaussian_filter1d((corrected - baseline) / denom, 2, axis=0)
        dff[start:stop] = trial_dff.astype(np.float32, copy=False)

        if capture_trace and trial_idx == trace_trial and fluorescence.shape[1] > 0:
            trace = {
                "raw_trial_index": int(trial_idx),
                "fluorescence": f_trial[:, 0].copy(),
                "neuropil": n_trial[:, 0].copy(),
                "corrected": corrected[:, 0].copy(),
                "baseline": baseline[:, 0].copy(),
                "dff": trial_dff[:, 0].copy(),
            }

    # Free the two large raw inputs before allocating a full event array in caller.
    events = np.full(dff.shape, np.nan, dtype=np.float32)
    for start, stop in zip(starts, stops):
        # Reference call: dcnv.oasis(cells x time, batch_size=2000, tau, fs).
        events[start:stop] = dcnv.oasis(
            dff[start:stop].T, 2000, OASIS_TAU_S, EFFECTIVE_FS_HZ
        ).T.astype(np.float32, copy=False)

    if trace is not None:
        start, stop = starts[trace_trial], stops[trace_trial]
        trace["events"] = events[start:stop, 0].copy()
    return dff, events, trace


def speed_correlations(dff: np.ndarray, speed: np.ndarray,
                       starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Vectorized Pearson correlation over all valid trial samples."""
    n_neurons = dff.shape[1]
    n = 0
    sum_x = np.zeros(n_neurons, dtype=np.float64)
    sum_xx = np.zeros(n_neurons, dtype=np.float64)
    sum_xy = np.zeros(n_neurons, dtype=np.float64)
    sum_y = 0.0
    sum_yy = 0.0
    for start, stop in zip(starts, stops):
        x = dff[start:stop].astype(np.float64, copy=False)
        y = speed[start:stop].astype(np.float64, copy=False)
        n += len(y)
        sum_x += np.sum(x, axis=0)
        sum_xx += np.sum(x * x, axis=0)
        sum_xy += np.sum(x * y[:, None], axis=0)
        sum_y += float(np.sum(y))
        sum_yy += float(np.sum(y * y))
    numerator = sum_xy - sum_x * sum_y / n
    denominator = np.sqrt((sum_xx - sum_x * sum_x / n) * (sum_yy - sum_y * sum_y / n))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def load_processed_neural(nwb, starts: np.ndarray, stops: np.ndarray,
                          speed: np.ndarray, capture_trace: bool) -> tuple[list[np.ndarray], int, int, dict | None, int]:
    """Load F/Fneu through PyNWB, process each plane, and filter neurons."""
    ophys = nwb.processing["ophys"]
    segmentation = ophys["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
    iscell = np.asarray(segmentation["iscell"].data[:])[:, 0] > 0
    fluorescence_series = ophys["Fluorescence"].roi_response_series
    neuropil_series = ophys["Neuropil"].roi_response_series

    plane_events: list[np.ndarray] = []
    manual_count = 0
    interneuron_count = 0
    plot_trace = None
    discarded_tail_frames = 0

    if set(fluorescence_series.keys()) != set(neuropil_series.keys()):
        raise ValueError("Fluorescence/Neuropil plane names do not match")

    for plane_idx, key in enumerate(fluorescence_series.keys()):
        f_series = fluorescence_series[key]
        n_series = neuropil_series[key]
        roi_ids = np.asarray(f_series.rois.data[:], dtype=np.int64)
        if not np.array_equal(roi_ids, np.asarray(n_series.rois.data[:], dtype=np.int64)):
            raise ValueError(f"F/Fneu ROI mappings differ for {key}")
        local_keep = np.flatnonzero(iscell[roi_ids])
        manual_count += len(local_keep)
        if len(local_keep) == 0:
            continue

        n_ophys_frames = int(f_series.data.shape[0])
        if int(n_series.data.shape[0]) != n_ophys_frames:
            raise ValueError(f"F/Fneu frame mismatch for {key}")
        if n_ophys_frames < len(speed):
            raise ValueError(
                f"Ophys ends before behavior for {key}: {n_ophys_frames} < {len(speed)}"
            )
        # Ten dual-plane files contain exactly one additional terminal ophys
        # row after the behavior stream and after the final teleport.  Both
        # streams start at time zero; discard only this unmatched tail row.
        extra_tail = n_ophys_frames - len(speed)
        if extra_tail > 1:
            raise ValueError(
                f"Unexpected ophys/behavior mismatch for {key}: +{extra_tail} frames"
            )
        discarded_tail_frames = max(discarded_tail_frames, extra_tail)
        fluorescence = np.asarray(f_series.data[:len(speed), local_keep], dtype=np.float32)
        neuropil = np.asarray(n_series.data[:len(speed), local_keep], dtype=np.float32)

        dff, events, trace = trial_dff_and_events(
            fluorescence, neuropil, starts, stops,
            capture_trace=capture_trace and plot_trace is None,
        )
        correlations = speed_correlations(dff, speed, starts, stops)
        keep = np.isfinite(correlations) & (correlations <= 0.5)
        interneuron_count += int(np.sum(~keep))
        plane_events.append(events[:, keep])
        if trace is not None:
            trace["speed_correlation"] = float(correlations[0])
            plot_trace = trace

        del fluorescence, neuropil, dff, events

    retained_count = manual_count - interneuron_count
    if retained_count <= 0:
        raise ValueError("No neurons remain after curation")
    return plane_events, manual_count, interneuron_count, plot_trace, discarded_tail_frames


def plot_processing(session_id: str, trace: dict, inputs: list[np.ndarray],
                    outputs: list[np.ndarray], positions: list[np.ndarray],
                    speeds: list[np.ndarray], zone_labels: list[str],
                    manual_count: int, interneuron_count: int,
                    raw_trials: int, excluded_trials: int) -> Path:
    """Plot each major transformation for one representative trial/session."""
    trial_to_plot = min(5, len(inputs) - 1)
    x = np.arange(len(trace["dff"])) * TIME_BIN_MS / 1000.0
    fig, axes = plt.subplots(4, 2, figsize=(18, 16))

    ax = axes[0, 0]
    ax.plot(x, trace["fluorescence"], label="raw F", alpha=0.8)
    ax.plot(x, trace["neuropil"], label="Fneu", alpha=0.7)
    ax.plot(x, trace["corrected"], label="neuropil-corrected F", alpha=0.8)
    ax.plot(x, trace["baseline"], label="maximin baseline", linewidth=2)
    ax.set(title="Fluorescence preprocessing", xlabel="Time from trial start (s)", ylabel="AU")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, trace["dff"], label="smoothed dF/F")
    ax.plot(x, trace["events"], label="OASIS events")
    ax.set(title=f"Neural transform (dF/F-speed r={trace['speed_correlation']:.3f})",
           xlabel="Time (s)", ylabel="Activity")
    ax.legend(fontsize=8)

    t = inputs[trial_to_plot][0]
    pos = positions[trial_to_plot]
    speed = speeds[trial_to_plot]
    zone = zone_labels[trial_to_plot]
    z0, z1 = ZONE_BOUNDS[zone]
    ax = axes[1, 0]
    ax.plot(t, pos, label="position (cm)")
    ax.plot(t, speed, label="speed (cm/s)", alpha=0.75)
    ax.axhspan(z0, z1, color="gold", alpha=0.2, label=f"zone {zone}")
    ax.set(title="Trial alignment and continuous behavior", xlabel="Time from start (s)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for idx, name in enumerate(INPUT_NAMES):
        values = inputs[trial_to_plot][idx]
        scale = np.ptp(values)
        normalized = (values - np.min(values)) / scale if scale else np.zeros_like(values)
        ax.plot(t, normalized + idx, label=name)
    ax.set(title="Decoder inputs", xlabel="Time (s)", yticks=np.arange(len(INPUT_NAMES)))
    ax.set_yticklabels(INPUT_NAMES, fontsize=8)

    ax = axes[2, 0]
    for idx, name in enumerate(OUTPUT_NAMES[:4]):
        values = outputs[trial_to_plot][idx]
        denom = max(1, int(values.max()))
        ax.step(t, values / denom + idx, where="post", label=name)
    ax.set(title="Time-varying categorical outputs", xlabel="Time (s)",
           yticks=np.arange(4))
    ax.set_yticklabels(OUTPUT_NAMES[:4], fontsize=8)

    ax = axes[2, 1]
    for idx, name in enumerate(OUTPUT_NAMES[4:], start=4):
        ax.step(t, outputs[trial_to_plot][idx] + (idx - 4) * 3, where="post", label=name)
    ax.set(title="Repeated per-trial outputs", xlabel="Time (s)")
    ax.legend(fontsize=8)

    ax = axes[3, 0]
    retained = manual_count - interneuron_count
    ax.bar(["manual cells", "speed-correlated\nexcluded", "retained"],
           [manual_count, interneuron_count, retained], color=["steelblue", "firebrick", "seagreen"])
    ax.set(title="Neuron curation", ylabel="ROI count")

    ax = axes[3, 1]
    concatenated = np.concatenate(outputs, axis=1)
    class_counts = [len(np.unique(concatenated[i])) for i in range(concatenated.shape[0])]
    ax.bar(OUTPUT_NAMES, class_counts)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.set(title=f"Observed output classes; trials {raw_trials-excluded_trials}/{raw_trials}",
           ylabel="Number of classes")

    fig.suptitle(f"Conversion processing: {session_id}", fontsize=15)
    fig.tight_layout()
    out = Path("/app") / f"processing_{session_id}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def process_session(path: Path, show_processing: bool) -> tuple[dict, dict]:
    """Convert one NWB session and return arrays plus metadata/statistics."""
    started = time.perf_counter()
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        behavior = nwb.processing["behavior"]["BehavioralTimeSeries"].time_series

        timestamps = np.asarray(behavior["position"].timestamps[:], dtype=np.float64)
        position = np.asarray(behavior["position"].data[:], dtype=np.float32)
        speed = np.asarray(behavior["speed"].data[:], dtype=np.float32)
        lick = np.asarray(behavior["lick"].data[:], dtype=np.float32)
        environment = np.asarray(behavior["environment"].data[:], dtype=np.float32)
        trial_number = np.asarray(behavior["trial number"].data[:], dtype=np.float32)
        starts = np.flatnonzero(np.asarray(behavior["trial_start"].data[:]) > 0)
        stops = np.flatnonzero(np.asarray(behavior["teleport"].data[:]) > 0)
        if len(starts) != len(stops) or np.any(stops <= starts):
            raise ValueError(f"Invalid trial bounds in {path.name}")
        if not all(len(x) == len(timestamps) for x in (position, speed, lick, environment, trial_number)):
            raise ValueError(f"Behavior series length mismatch in {path.name}")

        outcomes = reward_outcomes(
            np.asarray(behavior["Reward"].timestamps[:]), timestamps, starts, stops
        )
        scene = nwb.identifier.rsplit("/", 1)[-1]
        zone_labels_raw = parse_zone_schedule(scene, len(starts))
        corrupt_lick = np.asarray([
            np.mean(lick[start:stop] > 2) > LICK_ERROR_FRACTION
            for start, stop in zip(starts, stops)
        ])

        plane_events, manual_count, interneuron_count, trace, discarded_tail_frames = load_processed_neural(
            nwb, starts, stops, speed, capture_trace=show_processing
        )

        neural_trials: list[np.ndarray] = []
        input_trials: list[np.ndarray] = []
        output_trials: list[np.ndarray] = []
        kept_positions: list[np.ndarray] = []
        kept_speeds: list[np.ndarray] = []
        kept_zone_labels: list[str] = []
        kept_raw_indices: list[int] = []

        for raw_idx, (start, stop) in enumerate(zip(starts, stops)):
            if corrupt_lick[raw_idx]:
                continue
            if stop - start < 2:
                raise ValueError(f"Trial {raw_idx} in {path.name} has fewer than two frames")

            env_values = np.unique(environment[start:stop])
            env_values = env_values[env_values >= 0]
            if len(env_values) != 1 or env_values[0] not in (0, 1):
                raise ValueError(f"Trial {raw_idx} has invalid environment values {env_values}")
            source_trial_number = trial_number[start]
            if not np.isclose(source_trial_number, raw_idx):
                raise ValueError(
                    f"Trial-number mismatch in {path.name}: row {raw_idx}, value {source_trial_number}"
                )

            trial_position = position[start:stop]
            trial_speed = speed[start:stop]
            trial_lick = lick[start:stop]
            time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
            previous_outcome = outcomes[raw_idx - 1] if raw_idx > 0 else 0
            inputs = np.vstack([
                time_from_start,
                np.full(stop - start, env_values[0], dtype=np.float32),
                np.full(stop - start, source_trial_number, dtype=np.float32),
                np.full(stop - start, previous_outcome, dtype=np.float32),
            ]).astype(np.float32, copy=False)

            zone_label = str(zone_labels_raw[raw_idx])
            zone_start, zone_stop = ZONE_BOUNDS[zone_label]
            signed_distance = np.where(
                trial_position < zone_start,
                trial_position - zone_start,
                np.where(trial_position > zone_stop, trial_position - zone_stop, 0.0),
            )
            outputs = np.vstack([
                distance_classes(signed_distance),
                position_classes(trial_position),
                speed_classes(trial_speed),
                (trial_lick > 0).astype(np.int8),
                np.full(stop - start, ZONE_TO_CLASS[zone_label], dtype=np.int8),
                np.full(stop - start, outcomes[raw_idx], dtype=np.int8),
            ]).astype(np.int8, copy=False)

            neural = np.concatenate([events[start:stop].T for events in plane_events], axis=0)
            if neural.shape[1] != inputs.shape[1] or inputs.shape[1] != outputs.shape[1]:
                raise AssertionError("Neural/input/output time axes differ")
            if not (np.all(np.isfinite(neural)) and np.all(np.isfinite(inputs))):
                raise ValueError(f"Nonfinite converted data in {path.name}, trial {raw_idx}")

            neural_trials.append(neural.astype(np.float32, copy=False))
            input_trials.append(inputs)
            output_trials.append(outputs)
            kept_positions.append(trial_position.copy())
            kept_speeds.append(trial_speed.copy())
            kept_zone_labels.append(zone_label)
            kept_raw_indices.append(raw_idx)

        retained_neurons = manual_count - interneuron_count
        if len(neural_trials) < 2:
            raise ValueError(f"Only {len(neural_trials)} retained trials in {path.name}")
        if any(trial.shape[0] != retained_neurons for trial in neural_trials):
            raise AssertionError("Retained neuron count differs across trials")

        subject_id = str(nwb.subject.subject_id)
        _, session_day = natural_session_key(path)
        session_id = f"{subject_id}_ses-{session_day:02d}"
        median_dt = float(np.median(np.diff(timestamps)))
        session_metadata = {
            "session_id": session_id,
            "subject": subject_id,
            "session_day": int(session_day),
            "scene": scene,
            "source_file": str(path),
            "raw_trials": int(len(starts)),
            "retained_trials": int(len(neural_trials)),
            "excluded_corrupt_lick_trials": int(np.sum(corrupt_lick)),
            "kept_raw_trial_indices": kept_raw_indices,
            "segmented_rois": int(len(nwb.processing["ophys"]["ImageSegmentation"].plane_segmentations["PlaneSegmentation"])),
            "manual_cells": int(manual_count),
            "excluded_speed_correlated_cells": int(interneuron_count),
            "retained_neurons": int(retained_neurons),
            "behavior_median_dt_s": median_dt,
            "discarded_unmatched_terminal_ophys_frames": int(discarded_tail_frames),
        }

        if show_processing:
            if trace is None:
                raise RuntimeError(f"No processing trace captured for {path.name}")
            plot_path = plot_processing(
                session_id, trace, input_trials, output_trials, kept_positions,
                kept_speeds, kept_zone_labels, manual_count, interneuron_count,
                len(starts), int(np.sum(corrupt_lick)),
            )
            session_metadata["processing_plot"] = str(plot_path)

    elapsed = time.perf_counter() - started
    session_metadata["conversion_seconds"] = elapsed
    converted = {
        "neural": neural_trials,
        "input": input_trials,
        "output": output_trials,
        "subject": subject_id,
        "n_neurons": retained_neurons,
    }
    return converted, session_metadata


def build_metadata(session_info: list[dict], sample: bool) -> dict:
    return {
        "task_description": (
            "Head-fixed mice navigate a 450-cm virtual corridor in ENV1 or ENV2 "
            "with hidden 50-cm reward zones A/B/C and approximately 15% reward omissions. "
            "Neural CA1 population activity predicts reward-relative distance, absolute "
            "position, speed, licking, zone identity, and reward outcome."
        ),
        "time_bin_size": float(TIME_BIN_MS),
        "temporal_alignment_event": "trial_start flag: entry into the 0-cm start of the virtual track",
        "off_start": 0.0,
        "off_end": None,
        "variable_trial_duration": True,
        "source_dataset": "DANDI:001361/0.251124.0550",
        "source_format": "NWB loaded exclusively with PyNWB",
        "brain_region": "dorsal hippocampal CA1",
        "neural_signal": (
            "Trial-wise maximin dF/F from ROI and neuropil fluorescence, smoothed and "
            "OASIS-deconvolved according to Sosa et al.; manual cells retained and "
            "dF/F-speed Pearson r>0.5 putative interneurons excluded."
        ),
        "trial_filter": (
            "Track interval [trial_start, teleport); trials excluded only when >30% of "
            "samples have cumulative lick count >2."
        ),
        "reward_zone_bounds_cm": {key: list(value) for key, value in ZONE_BOUNDS.items()},
        "distance_definition": (
            "Signed distance to nearest point of active closed reward-zone interval: "
            "negative before, zero inside, positive after."
        ),
        "sample_conversion": bool(sample),
        "session_info": session_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default)")
    mode.add_argument("--sample", action="store_true", help="Process two representative sessions")
    parser.add_argument(
        "--show-processing", action="store_true",
        help="Save processing_<session_id>.png for up to two sessions",
    )
    args = parser.parse_args()

    check_discretization_boundaries()
    files = discover_files(sample=args.sample)
    print(f"Mode: {'sample' if args.sample else 'full'}; sessions: {len(files)}", flush=True)
    print(f"Output: {args.outpicklefile}", flush=True)

    neural: list[list[np.ndarray]] = []
    inputs: list[list[np.ndarray]] = []
    outputs: list[list[np.ndarray]] = []
    subject_ids: list[str] = []
    brain_region_idx: list[np.ndarray] = []
    session_info: list[dict] = []
    total_started = time.perf_counter()

    for session_idx, path in enumerate(files):
        plot_this = args.show_processing and session_idx < 2
        converted, info = process_session(path, show_processing=plot_this)
        neural.append(converted["neural"])
        inputs.append(converted["input"])
        outputs.append(converted["output"])
        subject_ids.append(converted["subject"])
        brain_region_idx.append(np.zeros(converted["n_neurons"], dtype=np.int64))
        session_info.append(info)
        print(
            f"[{session_idx + 1:3d}/{len(files)}] {path.name}: "
            f"trials {info['retained_trials']}/{info['raw_trials']}, "
            f"neurons {info['retained_neurons']}/{info['manual_cells']}, "
            f"{info['conversion_seconds']:.2f} s",
            flush=True,
        )

    subjects = sorted(set(subject_ids), key=lambda value: int(re.search(r"\d+", value).group()))
    subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
    subject_idx = np.asarray([subject_lookup[subject] for subject in subject_ids], dtype=np.int64)

    data = {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": subject_idx,
        "brain_regions": ["CA1"],
        "brain_region_idx": brain_region_idx,
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "output_values": OUTPUT_VALUES,
        "metadata": build_metadata(session_info, sample=args.sample),
    }

    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    write_started = time.perf_counter()
    with args.outpicklefile.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    write_seconds = time.perf_counter() - write_started
    total_seconds = time.perf_counter() - total_started

    n_trials = sum(len(session) for session in neural)
    n_neurons = sum(session[0].shape[0] for session in neural)
    n_timepoints = sum(trial.shape[1] for session in neural for trial in session)
    size_gb = args.outpicklefile.stat().st_size / 1e9
    print(
        f"Wrote {args.outpicklefile} ({size_gb:.3f} GB): {len(neural)} sessions, "
        f"{n_trials} trials, {n_neurons} session-neurons, {n_timepoints} trial timepoints",
        flush=True,
    )
    print(f"Pickle write: {write_seconds:.2f} s; total conversion: {total_seconds:.2f} s", flush=True)


if __name__ == "__main__":
    main()
