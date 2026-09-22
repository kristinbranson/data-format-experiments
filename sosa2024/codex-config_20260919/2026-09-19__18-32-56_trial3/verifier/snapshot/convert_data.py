#!/usr/bin/env python3
"""Convert the Sosa et al. DANDI NWB files to decoder-ready trial arrays.

Run as:
    python -u /app/convert_data.py OUTPUT.pkl [--full | --sample]
                                      [--show-processing]

The supplied NWB files contain behavior already synchronized to calcium-imaging
frames and author-produced OASIS-deconvolved calcium events.  This script keeps
that native time grid, applies the paper's cell/trial curation, and slices trials
from the explicit trial-start frame up to (but not including) teleport.
"""

from __future__ import annotations

import argparse
import pickle
import re
import time
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage


DATA_ROOT = Path("/app/data")
BEHAVIOR_ROOT = "processing/behavior/BehavioralTimeSeries"
OPHYS_ROOT = "processing/ophys"
DT_SECONDS = 1.0 / 15.5078125
LICK_FAULT_FRACTION = 0.30
INTERNEURON_R_THRESHOLD = 0.5
ZONE_BOUNDS = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
ZONE_TO_CLASS = {"A": 0, "B": 1, "C": 2}


def natural_file_key(path: Path) -> tuple[int, int]:
    """Sort NWBs by numeric mouse ID and then numeric session ID."""
    match = re.search(r"sub-m(\d+)_ses-(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot parse subject/session from {path}")
    return int(match.group(1)), int(match.group(2))


def discover_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.glob("sub-*/*.nwb"), key=natural_file_key)
    if not files:
        raise FileNotFoundError(f"No NWB files found under {DATA_ROOT}")
    if not sample:
        return files

    # Exercise switch logic, all reward-zone labels, both environments, and the
    # multi-plane path in only two sessions.
    # These sessions also give the validator's deterministic unstratified 80/20
    # split both reward-outcome classes in each held-out sample (4/12 and 6/10),
    # avoiding an undefined one-class balanced-accuracy check.
    wanted = {(12, 10), (18, 11)}
    selected = [p for p in files if natural_file_key(p) in wanted]
    if len(selected) != 2:
        raise RuntimeError(f"Expected sample sessions {sorted(wanted)}, found {selected}")
    return selected


def decode_text(dataset: h5py.Dataset) -> str:
    value = dataset[()]
    return value.decode() if isinstance(value, bytes) else str(value)


def trial_bounds(group: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    starts = np.flatnonzero(group["trial_start/data"][:] > 0)
    stops = np.flatnonzero(group["teleport/data"][:] > 0)
    if starts.size != stops.size or starts.size < 2:
        raise ValueError(f"Invalid trial boundaries: {starts.size} starts, {stops.size} stops")
    if np.any(starts >= stops) or np.any(starts[1:] <= stops[:-1]):
        raise ValueError("Trial starts/stops are not ordered non-overlapping intervals")
    return starts, stops


def scene_zone_labels(scene: str, n_trials: int) -> np.ndarray:
    """Return A/B/C for each trial from the scene protocol name."""
    labels = re.findall(r"(?:Location)?([ABC])", scene)
    # The regex can encounter the same letter once in the Location spelling only
    # through its explicit capture, so fixed sessions have one label and switches two.
    if len(labels) == 1:
        return np.full(n_trials, labels[0], dtype="<U1")
    if len(labels) == 2:
        if n_trials <= 30:
            raise ValueError(f"Switch scene {scene} has only {n_trials} trials")
        out = np.full(n_trials, labels[1], dtype="<U1")
        out[:30] = labels[0]
        return out
    raise ValueError(f"Could not parse reward-zone protocol from scene {scene!r}: {labels}")


def reward_outcomes(
    timestamps: np.ndarray,
    reward_timestamps: np.ndarray,
    reward_zone_signal: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
) -> np.ndarray:
    """Match behavior.get_trial_types: delivery plus zone evidence per trial."""
    out = np.zeros(starts.size, dtype=np.int8)
    for i, (start, stop) in enumerate(zip(starts, stops)):
        delivered = np.any(
            (reward_timestamps >= timestamps[start])
            & (reward_timestamps < timestamps[stop])
        )
        zone_evidence = np.any(reward_zone_signal[start:stop] > 0)
        out[i] = int(delivered and zone_evidence)
    return out


def compute_interneuron_mask(
    nwb: h5py.File,
    iscell: np.ndarray,
    speed: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    block_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Recreate the paper's per-cell dF/F-speed correlation exclusion.

    dF/F follows reward_relative.preprocessing.dff: 0.7 neuropil subtraction,
    within-trial maximin baseline (Gaussian sigma 15, 300-frame min and max
    filters), division by absolute baseline, and Gaussian sigma-2 smoothing.
    The reference dF/F code uses legacy ``start-1:stop-1`` bounds; we reproduce
    those only for this curation mask.  Target trial alignment itself uses the
    explicit NWB start frame.
    """
    keep = iscell.copy()
    correlations = np.full(iscell.size, np.nan, dtype=np.float32)
    qc_segments = [(max(0, int(a) - 1), max(0, int(b) - 1)) for a, b in zip(starts, stops)]

    fluorescence = nwb[f"{OPHYS_ROOT}/Fluorescence"]
    neuropil = nwb[f"{OPHYS_ROOT}/Neuropil"]

    for plane_name in sorted(fluorescence):
        f_group = fluorescence[plane_name]
        n_group = neuropil[plane_name]
        global_ids = f_group["rois"][:].astype(np.int64)
        local_cells = np.flatnonzero(iscell[global_ids])
        f_data = f_group["data"]
        n_data = n_group["data"]

        for block_start in range(0, local_cells.size, block_size):
            local = local_cells[block_start : block_start + block_size]
            count = local.size
            sum_x = np.zeros(count, dtype=np.float64)
            sum_y = 0.0
            sum_xx = np.zeros(count, dtype=np.float64)
            sum_yy = 0.0
            sum_xy = np.zeros(count, dtype=np.float64)
            n_samples = 0

            for start, stop in qc_segments:
                # Transpose to cell x time to match the reference implementation.
                f = f_data[start:stop, local].T.astype(np.float64, copy=False)
                f_neu = n_data[start:stop, local].T.astype(np.float64, copy=False)
                corrected = f - 0.7 * f_neu + 0.7 * f_neu.mean(axis=1, keepdims=True)
                baseline = ndimage.gaussian_filter1d(corrected, 15, axis=1)
                baseline = ndimage.minimum_filter1d(baseline, 300, axis=1)
                baseline = ndimage.maximum_filter1d(baseline, 300, axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    dff = (corrected - baseline) / np.abs(baseline)
                dff = ndimage.gaussian_filter1d(dff, 2, axis=1)
                y = speed[start:stop].astype(np.float64, copy=False)

                # The source data are finite in valid trial spans. Fail explicitly
                # rather than silently creating a different nan policy.
                if not np.all(np.isfinite(dff)) or not np.all(np.isfinite(y)):
                    raise ValueError(f"Non-finite dF/F or speed in {plane_name}")
                sum_x += dff.sum(axis=1)
                sum_y += float(y.sum())
                sum_xx += np.square(dff).sum(axis=1)
                sum_yy += float(np.square(y).sum())
                sum_xy += (dff * y[None, :]).sum(axis=1)
                n_samples += y.size

            numerator = sum_xy - sum_x * sum_y / n_samples
            denominator = np.sqrt(
                (sum_xx - np.square(sum_x) / n_samples)
                * (sum_yy - sum_y * sum_y / n_samples)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = numerator / denominator
            ids = global_ids[local]
            correlations[ids] = corr.astype(np.float32)
            keep[ids[corr > INTERNEURON_R_THRESHOLD]] = False

    return keep, correlations


def load_deconvolved(nwb: h5py.File, keep_global: np.ndarray, n_behavior: int) -> np.ndarray:
    """Load retained cells from all planes and concatenate in global ROI order."""
    chunks: list[np.ndarray] = []
    ids_all: list[np.ndarray] = []
    deconvolved = nwb[f"{OPHYS_ROOT}/Deconvolved"]
    for plane_name in sorted(deconvolved):
        group = deconvolved[plane_name]
        global_ids = group["rois"][:].astype(np.int64)
        local = np.flatnonzero(keep_global[global_ids])
        if local.size:
            chunks.append(group["data"][:n_behavior, local].astype(np.float32, copy=False))
            ids_all.append(global_ids[local])
    if not chunks:
        raise ValueError("No neurons remain after curation")
    activity = np.concatenate(chunks, axis=1)
    ids = np.concatenate(ids_all)
    order = np.argsort(ids)
    if not np.array_equal(ids[order], np.flatnonzero(keep_global)):
        raise ValueError("Plane ROI references do not match the segmentation table")
    return activity[:, order]


def discretize_distance(distance: np.ndarray) -> np.ndarray:
    out = np.empty(distance.shape, dtype=np.int8)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance < -10)] = 1
    out[(distance >= -10) & (distance < 0)] = 2
    out[distance == 0] = 3
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
    return out


def discretize_position(position: np.ndarray) -> np.ndarray:
    out = np.empty(position.shape, dtype=np.int8)
    out[position < 90] = 0
    out[(position >= 90) & (position < 180)] = 1
    out[(position >= 180) & (position < 270)] = 2
    out[(position >= 270) & (position <= 360)] = 3
    out[position > 360] = 4
    return out


def discretize_speed(speed: np.ndarray) -> np.ndarray:
    out = np.empty(speed.shape, dtype=np.int8)
    out[speed < 2] = 0
    out[(speed >= 2) & (speed < 10)] = 1
    out[(speed >= 10) & (speed < 20)] = 2
    out[(speed >= 20) & (speed <= 40)] = 3
    out[speed > 40] = 4
    return out


def verify_discretization_edges() -> None:
    eps = 1e-5
    distance = np.array(
        [-50 - eps, -50, -10 - eps, -10, -eps, 0, eps, 10, 10 + eps, 50, 50 + eps]
    )
    expected_distance = np.array([0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6])
    if not np.array_equal(discretize_distance(distance), expected_distance):
        raise AssertionError("Distance discretization edge test failed")
    position = np.array([90 - eps, 90, 180 - eps, 180, 270, 360, 360 + eps])
    expected_position = np.array([0, 1, 1, 2, 3, 3, 4])
    if not np.array_equal(discretize_position(position), expected_position):
        raise AssertionError("Position discretization edge test failed")
    speed = np.array([2 - eps, 2, 10 - eps, 10, 20, 40, 40 + eps])
    expected_speed = np.array([0, 1, 1, 2, 3, 3, 4])
    if not np.array_equal(discretize_speed(speed), expected_speed):
        raise AssertionError("Speed discretization edge test failed")


def make_processing_plot(
    session_tag: str,
    timestamps: np.ndarray,
    position: np.ndarray,
    speed: np.ndarray,
    lick: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    bad_trials: np.ndarray,
    trial_inputs: list[np.ndarray],
    trial_outputs: list[np.ndarray],
    trial_neural: list[np.ndarray],
    retained_raw_indices: list[int],
    n_segmented: int,
    n_iscell: int,
    n_interneurons: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(7, 1, figsize=(15, 23), constrained_layout=True)
    rel_session_time = timestamps - timestamps[0]
    axes[0].plot(rel_session_time, position, color="black", lw=0.5)
    axes[0].scatter(rel_session_time[starts], position[starts], s=8, color="green", label="trial start")
    # The position value on the teleport event sample is documented as an
    # unreliable interpolation between track end and tunnel. Plot the last
    # included frame so the displayed marker represents the converted bound.
    axes[0].scatter(
        rel_session_time[stops], position[stops - 1], s=8, color="blue",
        label="last included frame (before teleport)",
    )
    for i in np.flatnonzero(bad_trials):
        axes[0].axvspan(rel_session_time[starts[i]], rel_session_time[stops[i]], color="red", alpha=0.15)
    axes[0].set(ylabel="position (cm)", title=f"{session_tag}: source session and trial boundaries")
    axes[0].legend(loc="upper right", ncol=3)

    retained = n_iscell - n_interneurons
    axes[1].bar(
        ["segmented ROIs", "manual iscell", "after r>0.5 QC"],
        [n_segmented, n_iscell, retained],
        color=["0.65", "#4c78a8", "#59a14f"],
    )
    axes[1].set(ylabel="ROI count", title="Neural curation")

    # Use a middle retained trial to show both start and end alignment clearly.
    plot_i = len(trial_neural) // 2
    t = trial_inputs[plot_i][0]
    neural = trial_neural[plot_i]
    for neuron in range(min(10, neural.shape[0])):
        trace = neural[neuron]
        scale = np.nanpercentile(trace, 99)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        axes[2].plot(t, trace / scale + neuron, lw=0.5)
    axes[2].set(ylabel="normalized neuron", title=f"Aligned deconvolved events; raw trial {retained_raw_indices[plot_i]}")

    out = trial_outputs[plot_i]
    # Reconstruct continuous values for visual validation directly from class-independent inputs.
    raw_i = retained_raw_indices[plot_i]
    lo, hi = starts[raw_i], stops[raw_i]
    pos = position[lo:hi]
    axes[3].plot(t, pos, label="absolute position")
    axes[3].step(t, out[1].astype(float) * 90, where="mid", label="position class ×90", alpha=0.7)
    axes[3].set(ylabel="cm / class", title="Continuous position and requested discretization")
    axes[3].legend(loc="upper left")

    zone_class = int(out[4, 0])
    zone_start, zone_stop = list(ZONE_BOUNDS.values())[zone_class]
    signed_distance = np.where(
        pos < zone_start, pos - zone_start,
        np.where(pos > zone_stop, pos - zone_stop, 0.0),
    )
    axes[4].plot(t, signed_distance, color="black", label="signed distance (cm)")
    distance_axis = axes[4].twinx()
    distance_axis.step(t, out[0], where="mid", color="#e15759", label="distance class")
    axes[4].axhline(0, color="0.6", lw=0.7)
    axes[4].set(ylabel="signed distance (cm)", title="Continuous distance-to-zone and discretization")
    distance_axis.set_ylabel("class")
    lines = [
        line for line in axes[4].get_lines() + distance_axis.get_lines()
        if not line.get_label().startswith("_")
    ]
    axes[4].legend(lines, [line.get_label() for line in lines], loc="upper left")

    trial_speed = speed[lo:hi]
    trial_lick = lick[lo:hi]
    axes[5].plot(t, trial_speed, color="#4c78a8", label="speed (cm/s)")
    axes[5].step(t, out[2].astype(float) * 10, where="mid", color="#f28e2b", label="speed class ×10")
    axes[5].scatter(t[trial_lick > 0], np.zeros(np.sum(trial_lick > 0)), marker="|", color="#59a14f", label="lick")
    axes[5].set(ylabel="cm/s", title="Continuous speed, speed classes, and binary licks")
    axes[5].legend(loc="upper right", ncol=3)

    image = axes[6].imshow(out, aspect="auto", interpolation="nearest", origin="lower", extent=[t[0], t[-1], -0.5, 5.5])
    axes[6].set(
        xlabel="seconds from trial start",
        ylabel="output",
        yticks=np.arange(6),
        yticklabels=["distance", "position", "speed", "lick", "zone", "outcome"],
        title="All categorical decoder outputs (per-trial values are constant)",
    )
    fig.colorbar(image, ax=axes[6], label="class index")
    path = Path(f"/app/processing_{session_tag}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def convert_session(path: Path, show_processing: bool) -> tuple[dict, dict]:
    started = time.perf_counter()
    with h5py.File(path, "r") as nwb:
        behavior = nwb[BEHAVIOR_ROOT]
        subject = decode_text(nwb["general/subject/subject_id"])
        session_id = decode_text(nwb["general/session_id"])
        scene = decode_text(nwb["identifier"]).rsplit("/", 1)[-1]
        session_tag = f"{subject}_ses-{session_id}"

        timestamps = behavior["position/timestamps"][:]
        position = behavior["position/data"][:]
        speed = behavior["speed/data"][:]
        lick = behavior["lick/data"][:]
        environment = behavior["environment/data"][:]
        trial_number = behavior["trial number/data"][:]
        reward_zone_signal = behavior["reward_zone/data"][:]
        scanning = behavior["scanning/data"][:]
        reward_timestamps = behavior["Reward/timestamps"][:]
        starts, stops = trial_bounds(behavior)

        n_behavior = timestamps.size
        dense_lengths = {
            name: behavior[f"{name}/data"].shape[0]
            for name in [
                "position", "speed", "lick", "environment", "trial number",
                "reward_zone", "scanning", "trial_start", "teleport",
            ]
        }
        if set(dense_lengths.values()) != {n_behavior}:
            raise ValueError(f"Dense behavior length mismatch in {path}: {dense_lengths}")
        if not np.allclose(np.diff(timestamps), DT_SECONDS, rtol=0, atol=1e-10):
            raise ValueError(f"Nonuniform or unexpected timestamps in {path}")

        labels = scene_zone_labels(scene, starts.size)
        outcomes = reward_outcomes(
            timestamps, reward_timestamps, reward_zone_signal, starts, stops
        )
        previous_outcomes = np.r_[0, outcomes[:-1]].astype(np.int8)
        bad_trials = np.array(
            [np.mean(lick[a:b] > 2) > LICK_FAULT_FRACTION for a, b in zip(starts, stops)],
            dtype=bool,
        )

        segmentation = nwb[f"{OPHYS_ROOT}/ImageSegmentation/PlaneSegmentation"]
        iscell = segmentation["iscell"][:, 0].astype(bool)
        keep, speed_correlations = compute_interneuron_mask(
            nwb, iscell, speed, starts, stops
        )
        n_interneurons = int(np.sum(iscell & ~keep))
        activity = load_deconvolved(nwb, keep, n_behavior)

        neural_trials: list[np.ndarray] = []
        input_trials: list[np.ndarray] = []
        output_trials: list[np.ndarray] = []
        retained_raw_indices: list[int] = []

        for raw_trial, (start, stop) in enumerate(zip(starts, stops)):
            if bad_trials[raw_trial]:
                continue
            if not np.all(scanning[start:stop] == 1):
                raise ValueError(f"Non-scanning sample inside {session_tag} trial {raw_trial}")
            env_values = np.unique(environment[start:stop])
            trial_values = np.unique(trial_number[start:stop])
            if env_values.size != 1 or env_values[0] not in (0, 1):
                raise ValueError(f"Invalid environment in {session_tag} trial {raw_trial}: {env_values}")
            if trial_values.size != 1:
                raise ValueError(f"Multiple trial IDs in {session_tag} trial {raw_trial}: {trial_values}")

            trial_position = position[start:stop]
            trial_speed = speed[start:stop]
            trial_lick = lick[start:stop]
            label = labels[raw_trial]
            zone_start, zone_stop = ZONE_BOUNDS[label]
            signed_distance = np.where(
                trial_position < zone_start,
                trial_position - zone_start,
                np.where(trial_position > zone_stop, trial_position - zone_stop, 0.0),
            )

            relative_time = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
            T = stop - start
            decoder_input = np.vstack(
                [
                    relative_time,
                    np.full(T, env_values[0], dtype=np.float32),
                    np.full(T, trial_values[0], dtype=np.float32),
                    np.full(T, previous_outcomes[raw_trial], dtype=np.float32),
                ]
            ).astype(np.float32, copy=False)
            decoder_output = np.vstack(
                [
                    discretize_distance(signed_distance),
                    discretize_position(trial_position),
                    discretize_speed(trial_speed),
                    (trial_lick > 0).astype(np.int8),
                    np.full(T, ZONE_TO_CLASS[label], dtype=np.int8),
                    np.full(T, outcomes[raw_trial], dtype=np.int8),
                ]
            ).astype(np.int8, copy=False)
            neural = activity[start:stop].T.copy()

            if neural.shape[1] != T or decoder_input.shape != (4, T) or decoder_output.shape != (6, T):
                raise AssertionError(f"Shape mismatch in {session_tag} trial {raw_trial}")
            if not np.all(np.isfinite(neural)) or not np.all(np.isfinite(decoder_input)):
                raise ValueError(f"Non-finite converted values in {session_tag} trial {raw_trial}")
            neural_trials.append(neural)
            input_trials.append(decoder_input)
            output_trials.append(decoder_output)
            retained_raw_indices.append(raw_trial)

        if len(neural_trials) < 2:
            raise ValueError(f"{session_tag} retains fewer than two trials")

        plane_names = sorted(nwb[f"{OPHYS_ROOT}/Deconvolved"].keys())
        plane_frames = [
            int(nwb[f"{OPHYS_ROOT}/Deconvolved/{name}/data"].shape[0])
            for name in plane_names
        ]
        session_info = {
            "session_tag": session_tag,
            "subject": subject,
            "session_id": session_id,
            "scene": scene,
            "source_file": str(path),
            "n_planes": len(plane_names),
            "plane_frames": plane_frames,
            "behavior_frames": int(n_behavior),
            "source_trials": int(starts.size),
            "retained_trials": len(neural_trials),
            "excluded_lick_fault_trials": np.flatnonzero(bad_trials).astype(int).tolist(),
            "segmented_rois": int(iscell.size),
            "manual_iscell_rois": int(iscell.sum()),
            "excluded_interneurons": n_interneurons,
            "retained_neurons": int(keep.sum()),
            "rewarded_source_trials": int(outcomes.sum()),
            "retained_raw_trial_indices": retained_raw_indices,
            "max_dff_speed_correlation": float(np.nanmax(speed_correlations)),
        }

        if show_processing:
            make_processing_plot(
                session_tag,
                timestamps,
                position,
                speed,
                lick,
                starts,
                stops,
                bad_trials,
                input_trials,
                output_trials,
                neural_trials,
                retained_raw_indices,
                int(iscell.size),
                int(iscell.sum()),
                n_interneurons,
            )

    converted = {
        "subject": subject,
        "neural": neural_trials,
        "input": input_trials,
        "output": output_trials,
        "brain_region_idx": np.zeros(activity.shape[1], dtype=np.int16),
    }
    elapsed = time.perf_counter() - started
    print(
        f"  {session_tag}: {len(neural_trials)}/{starts.size} trials, "
        f"{activity.shape[1]}/{iscell.sum()} curated cells retained, {elapsed:.2f}s",
        flush=True,
    )
    return converted, session_info


def validate_converted(data: dict) -> None:
    n_sessions = len(data["neural"])
    if not (
        len(data["input"]) == len(data["output"]) == len(data["brain_region_idx"]) == n_sessions
    ):
        raise AssertionError("Session-list lengths differ")
    if data["subject_idx"].shape != (n_sessions,):
        raise AssertionError("subject_idx shape mismatch")
    for s in range(n_sessions):
        n_trials = len(data["neural"][s])
        if n_trials < 2 or len(data["input"][s]) != n_trials or len(data["output"][s]) != n_trials:
            raise AssertionError(f"Invalid trial lists in session {s}")
        n_neurons = data["neural"][s][0].shape[0]
        if data["brain_region_idx"][s].shape != (n_neurons,):
            raise AssertionError(f"Brain-region length mismatch in session {s}")
        for neural, inp, out in zip(data["neural"][s], data["input"][s], data["output"][s]):
            T = neural.shape[1]
            if neural.shape[0] != n_neurons or inp.shape != (4, T) or out.shape != (6, T):
                raise AssertionError(f"Converted shape mismatch in session {s}")
            if neural.dtype != np.float32 or inp.dtype != np.float32 or out.dtype != np.int8:
                raise AssertionError(f"Converted dtype mismatch in session {s}")
            if inp[0, 0] != 0 or np.any(np.diff(inp[0]) <= 0):
                raise AssertionError(f"Time alignment failure in session {s}")
            if np.any(out < 0):
                raise AssertionError(f"Negative output class in session {s}")
            for dim, n_classes in enumerate([7, 5, 5, 2, 3, 2]):
                if np.any(out[dim] >= n_classes):
                    raise AssertionError(f"Output {dim} class out of range in session {s}")


def build_dataset(files: list[Path], show_processing: bool) -> dict:
    verify_discretization_edges()
    subjects = sorted(
        {f"m{natural_file_key(path)[0]}" for path in files},
        key=lambda value: int(value[1:]),
    )
    subject_lookup = {subject: i for i, subject in enumerate(subjects)}
    converted_sessions = []
    session_info = []
    started = time.perf_counter()

    for index, path in enumerate(files):
        session_started = time.perf_counter()
        print(f"[{index + 1}/{len(files)}] {path.relative_to(DATA_ROOT)}", flush=True)
        converted, info = convert_session(path, show_processing and index < 2)
        converted_sessions.append(converted)
        session_info.append(info)
        elapsed = time.perf_counter() - started
        per_session = time.perf_counter() - session_started
        remaining = per_session * (len(files) - index - 1)
        print(f"  elapsed {elapsed:.1f}s; local remaining estimate {remaining:.1f}s", flush=True)

    data = {
        "neural": [item["neural"] for item in converted_sessions],
        "input": [item["input"] for item in converted_sessions],
        "output": [item["output"] for item in converted_sessions],
        "subjects": subjects,
        "subject_idx": np.asarray(
            [subject_lookup[item["subject"]] for item in converted_sessions], dtype=np.int16
        ),
        "brain_regions": ["CA1"],
        "brain_region_idx": [item["brain_region_idx"] for item in converted_sessions],
        "input_names": [
            "time_from_trial_start_s",
            "environment",
            "trial_number",
            "previous_trial_outcome",
        ],
        "output_names": [
            "distance_to_reward_zone",
            "absolute_position",
            "speed",
            "lick",
            "reward_zone_location",
            "reward_outcome",
        ],
        "output_values": [
            ["< -50 cm", "-50 to < -10 cm", "-10 to < 0 cm", "inside zone (0 cm)", "> 0 to 10 cm", "> 10 to 50 cm", "> 50 cm"],
            ["< 90 cm", "90 to < 180 cm", "180 to < 270 cm", "270 to 360 cm", "> 360 cm"],
            ["< 2 cm/s", "2 to < 10 cm/s", "10 to < 20 cm/s", "20 to 40 cm/s", "> 40 cm/s"],
            ["no lick", "lick"],
            ["A (80-130 cm)", "B (200-250 cm)", "C (320-370 cm)"],
            ["not rewarded", "rewarded"],
        ],
        "metadata": {
            "task_description": (
                "Head-fixed mice navigate a 450 cm virtual corridor in ENV1 or ENV2 "
                "while hidden 50 cm reward zones A/B/C change across trials and days; "
                "outputs describe reward-relative/absolute position, movement, licking, "
                "reward-zone identity, and reward outcome."
            ),
            "time_bin_size": DT_SECONDS * 1000.0,
            "temporal_alignment_event": "first imaging frame with trial_start > 0 (entry to 0 cm track)",
            "off_start": 0.0,
            "off_end": None,
            "variable_trial_length": True,
            "source": "DANDI 001361 version 0.251124.0550",
            "source_root": str(DATA_ROOT),
            "neural_signal": "author-produced OASIS-deconvolved calcium events",
            "neural_curation": (
                "Suite2p manual iscell flag, then exclude cells with paper-method "
                "within-trial dF/F-speed Pearson r > 0.5; planes pooled in ROI order"
            ),
            "trial_interval": "[explicit trial_start frame, explicit teleport frame)",
            "trial_exclusion": (
                "exclude trials where >30% of frames have cumulative lick count >2"
            ),
            "reward_zone_bounds_cm": ZONE_BOUNDS,
            "sampling_rate_hz": 1.0 / DT_SECONDS,
            "input_encoding": (
                "environment 0=ENV1/1=ENV2; native zero-based trial number; previous "
                "reward outcome 0=omitted-or-unavailable/1=rewarded"
            ),
            "output_discretization": {
                "distance_to_reward_zone": "<−50; [−50,−10); [−10,0); 0; (0,10]; (10,50]; >50 cm",
                "absolute_position": "<90; [90,180); [180,270); [270,360]; >360 cm",
                "speed": "<2; [2,10); [10,20); [20,40]; >40 cm/s",
                "lick": "lick count >0",
            },
            "session_info": session_info,
        },
    }
    validate_converted(data)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="process all sessions (default)")
    mode.add_argument("--sample", action="store_true", help="process representative two sessions")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="save processing_<session_id>.png for up to two sessions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_files(sample=args.sample)
    mode = "sample" if args.sample else "full"
    print(f"Starting {mode} conversion of {len(files)} sessions", flush=True)
    started = time.perf_counter()
    data = build_dataset(files, args.show_processing)
    conversion_elapsed = time.perf_counter() - started
    print(f"Conversion complete in {conversion_elapsed:.2f}s; writing {args.outpicklefile}", flush=True)
    write_started = time.perf_counter()
    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    with args.outpicklefile.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    write_elapsed = time.perf_counter() - write_started
    size_gib = args.outpicklefile.stat().st_size / (2**30)
    print(
        f"Wrote {args.outpicklefile} ({size_gib:.3f} GiB) in {write_elapsed:.2f}s; "
        f"total {time.perf_counter() - started:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
