#!/usr/bin/env python3
"""Convert Hasnain/Birnbaum et al. context-task data for neural decoding.

Usage:
    python -u /app/convert_data.py OUTPUT.pkl [--full|--sample]
                                           [--show-processing]

The implementation intentionally mirrors the supplied MATLAB loading pipeline:
author-selected ALM probes, go-cue alignment, 5 ms spike bins, a 15-sample
causal Gaussian, reference video synchronization/interpolation, and >1 Hz unit
curation. Decoder-specific median discretization is applied only after alignment.
"""

from __future__ import annotations

import argparse
import gc
import pickle
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter
from scipy.signal.windows import gaussian


DATA_DIR = Path("/app/data/Ephys_Behavior")

# This is the exact two-context session/probe list used by the paper's Figure 8
# scripts and their load<animal>_ALMVideo helpers. Probe numbers are MATLAB 1-based.
CONTEXT_SESSIONS = [
    ("JEB6", "2021-04-18", 2),
    ("JEB7", "2021-04-29", 1),
    ("JEB7", "2021-04-30", 1),
    ("EKH1", "2021-08-07", 2),
    ("EKH3", "2021-08-11", 2),
    ("JGR2", "2021-11-16", 1),
    ("JGR2", "2021-11-17", 1),
    ("JGR3", "2021-11-18", 1),
    # The JEB19 author loader orders these sessions in reverse chronological order.
    ("JEB19", "2023-04-21", 1),
    ("JEB19", "2023-04-20", 1),
    ("JEB19", "2023-04-19", 1),
    ("JEB19", "2023-04-18", 1),
]

DT = 0.005
FILTER_TMIN = -3.0  # Figure 8 processing start; output is cropped after smoothing.
OUTPUT_TMIN = -2.5
TMAX = 2.5
SMOOTH_SAMPLES = 15
LOW_FR_HZ = 1.0

FILTER_EDGES = np.arange(FILTER_TMIN, TMAX + DT / 2, DT, dtype=np.float64)
FILTER_TIME = (FILTER_EDGES[:-1] + DT / 2).astype(np.float32)
OUTPUT_MASK = (FILTER_TIME >= OUTPUT_TMIN) & (FILTER_TIME < TMAX)
OUTPUT_TIME = FILTER_TIME[OUTPUT_MASK]
N_TIME = int(OUTPUT_TIME.size)

if N_TIME != 1000:
    raise RuntimeError(f"Expected 1000 output bins, constructed {N_TIME}")


@dataclass
class SessionResult:
    neural: list[np.ndarray]
    inputs: list[np.ndarray]
    outputs: list[np.ndarray]
    brain_region_idx: np.ndarray
    metadata: dict[str, Any]
    diagnostics: dict[str, Any]


def matlab_char(dataset: h5py.Dataset) -> str:
    """Decode a MATLAB v7.3 UTF-16 char dataset."""
    values = np.asarray(dataset[()]).ravel(order="F")
    return "".join(chr(int(value)) for value in values if int(value))


def referenced_objects(
    handle: h5py.File, dataset: h5py.Dataset, *, preserve_empty: bool = False
) -> list[h5py.Group | h5py.Dataset | None]:
    """Dereference a MATLAB cell/struct-field dataset in column-major order."""
    output: list[h5py.Group | h5py.Dataset | None] = []
    for reference in np.asarray(dataset[()]).ravel(order="F"):
        if isinstance(reference, h5py.Reference) and reference:
            output.append(handle[reference])
        elif preserve_empty:
            output.append(None)
    return output


def referenced_at(
    handle: h5py.File, dataset: h5py.Dataset, index: int
) -> h5py.Group | h5py.Dataset | None:
    values = np.asarray(dataset[()]).ravel(order="F")
    reference = values[index]
    if isinstance(reference, h5py.Reference) and reference:
        return handle[reference]
    return None


def matlab_mode(values: np.ndarray) -> float:
    """MATLAB-compatible mode for finite numeric vectors (smallest on ties)."""
    finite = np.asarray(values, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    unique, counts = np.unique(finite, return_counts=True)
    return float(unique[np.flatnonzero(counts == counts.max())[0]])


def causal_gaussian_kernel(length: int = SMOOTH_SAMPLES) -> np.ndarray:
    """Reproduce MATLAB gausswin(N), zeroing its leading half as mySmooth.m."""
    # MATLAB gausswin's default alpha is 2.5.
    kernel = gaussian(length, std=(length - 1) / (2 * 2.5)).astype(np.float32)
    kernel[: length // 2] = 0
    kernel /= kernel.sum()
    return kernel


CAUSAL_KERNEL = causal_gaussian_kernel()
CAUSAL_TAPS = CAUSAL_KERNEL[SMOOTH_SAMPLES // 2 :]


def reference_smooth(rates: np.ndarray) -> np.ndarray:
    """Vectorized exact equivalent of mySmooth(..., 15, 'reflect').

    The supplied MATLAB function's "reflect" branch prepends (without reversing)
    the first N samples. For an odd causal kernel, conv(...,'same') is equivalent
    to an FIR filter using the nonzero second half of the kernel.
    """
    prefix = rates[..., :SMOOTH_SAMPLES]
    padded = np.concatenate((prefix, rates), axis=-1)
    filtered = lfilter(CAUSAL_TAPS, np.array([1.0], dtype=np.float32), padded, axis=-1)
    return filtered[..., SMOOTH_SAMPLES:].astype(np.float32, copy=False)


def trial_flags(handle: h5py.File) -> dict[str, np.ndarray]:
    flags = {}
    for name in ("L", "R", "hit", "miss", "no", "early", "autowater"):
        flags[name] = np.asarray(handle[f"obj/bp/{name}"][()]).ravel().astype(bool)
    flags["stim"] = np.asarray(handle["obj/bp/stim/enable"][()]).ravel().astype(bool)
    flags["have_ephys"] = np.asarray(
        handle["obj/trials/bp/haveEphys"][()]
    ).ravel().astype(bool)
    flags["have_video"] = np.asarray(
        handle["obj/trials/bp/haveVid"][()]
    ).ravel().astype(bool)
    return flags


def reference_conditions(flags: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Conditions used by the paper's context analysis before low-FR filtering."""
    hit, miss, no = flags["hit"], flags["miss"], flags["no"]
    aw, stim, early = flags["autowater"], flags["stim"], flags["early"]
    return [
        hit | miss | no,
        hit & ~stim & ~aw,
        hit & ~stim & aw,
        miss & ~stim & ~aw,
        miss & ~stim & aw,
        hit & ~stim & ~aw & ~early,
        hit & ~stim & aw & ~early,
    ]


def select_quality_indices(handle: h5py.File, cluster_group: h5py.Group) -> tuple[list[int], list[str]]:
    """Port findClusters(..., {'all'}), including its case-sensitive exclusions."""
    quality_objects = referenced_objects(handle, cluster_group["quality"])
    selected: list[int] = []
    selected_labels: list[str] = []
    for index, quality_object in enumerate(quality_objects):
        label = matlab_char(quality_object).strip() if isinstance(quality_object, h5py.Dataset) else ""
        if label in {"garbage", "gabrga", "noisy", "real?"}:
            continue
        selected.append(index)
        selected_labels.append(label)
    return selected, selected_labels


def load_and_process_neural(
    handle: h5py.File, probe_number: int, go_cue: np.ndarray, flags: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load, align, bin, smooth, and curate one session's selected ALM probe."""
    n_trials = go_cue.size
    probe_cells = referenced_objects(handle, handle["obj/clu"], preserve_empty=True)
    if probe_number - 1 >= len(probe_cells):
        raise ValueError(f"Probe {probe_number} absent from obj.clu")
    cluster_group = probe_cells[probe_number - 1]
    if not isinstance(cluster_group, h5py.Group):
        raise ValueError(f"Probe {probe_number} does not contain cluster structures")

    selected_indices, quality_labels = select_quality_indices(handle, cluster_group)
    trial_time_objects = referenced_objects(handle, cluster_group["trialtm"])
    trial_number_objects = referenced_objects(handle, cluster_group["trial"])

    n_quality = len(selected_indices)
    n_filter_time = FILTER_TIME.size
    counts = np.zeros((n_quality, n_trials, n_filter_time), dtype=np.float32)

    for output_index, cluster_index in enumerate(selected_indices):
        trial_time = np.asarray(trial_time_objects[cluster_index][()]).ravel().astype(np.float64)
        trial_number = np.asarray(trial_number_objects[cluster_index][()]).ravel().astype(np.int64) - 1
        valid_trial = (trial_number >= 0) & (trial_number < n_trials)
        trial_time = trial_time[valid_trial]
        trial_number = trial_number[valid_trial]
        aligned = trial_time - go_cue[trial_number]
        bin_index = np.floor((aligned - FILTER_TMIN) / DT).astype(np.int64)
        valid_bin = (bin_index >= 0) & (bin_index < n_filter_time)
        np.add.at(
            counts[output_index],
            (trial_number[valid_bin], bin_index[valid_bin]),
            1.0,
        )

    rates = reference_smooth(counts / np.float32(DT))
    del counts

    condition_means = np.zeros((n_quality, len(reference_conditions(flags))), dtype=np.float64)
    for condition_index, mask in enumerate(reference_conditions(flags)):
        trial_indices = np.flatnonzero(mask)
        if trial_indices.size:
            condition_means[:, condition_index] = rates[:, trial_indices, :].mean(
                axis=(1, 2), dtype=np.float64
            )
    mean_fr = condition_means.mean(axis=1)
    keep = mean_fr > LOW_FR_HZ
    rates = rates[keep][:, :, OUTPUT_MASK]

    normalized_quality = np.array([label.strip().lower() for label in quality_labels], dtype=object)
    well_isolated = np.isin(normalized_quality, ["excellent", "great", "good", "fair", "ood"])

    diagnostics = {
        "n_clusters_on_probe": int(cluster_group["quality"].shape[0]),
        "n_quality_selected": int(n_quality),
        "n_low_fr_removed": int((~keep).sum()),
        "n_neurons": int(keep.sum()),
        "n_well_isolated": int(np.sum(well_isolated & keep)),
        "mean_fr_all_quality_selected": mean_fr.astype(np.float32),
        "keep_mask": keep,
        "quality_labels": quality_labels,
    }
    return rates, diagnostics


def trajectory_group(handle: h5py.File, view_index: int) -> h5py.Group:
    views = referenced_objects(handle, handle["obj/traj"])
    group = views[view_index]
    if not isinstance(group, h5py.Group):
        raise ValueError(f"Trajectory view {view_index + 1} missing")
    return group


def feature_names(handle: h5py.File, group: h5py.Group, trial_index: int = 0) -> list[str]:
    names_object = referenced_at(handle, group["featNames"], trial_index)
    if not isinstance(names_object, h5py.Dataset):
        raise ValueError("DLC feature names missing")
    return [
        matlab_char(item)
        for item in referenced_objects(handle, names_object)
        if isinstance(item, h5py.Dataset)
    ]


def trajectory_array(dataset: h5py.Dataset, n_features: int) -> np.ndarray:
    """Return DLC data in MATLAB logical order: frames x 3 x features."""
    array = np.asarray(dataset[()], dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"Unexpected DLC array shape {array.shape}")
    if array.shape[0] == n_features and array.shape[1] == 3:
        return np.transpose(array, (2, 1, 0))
    if array.shape[2] == n_features and array.shape[1] == 3:
        return array
    raise ValueError(f"Cannot orient DLC array {array.shape} with {n_features} features")


def frame_times_for_trial(
    handle: h5py.File, group: h5py.Group, trial_index: int, n_frames: int
) -> np.ndarray:
    frame_object = referenced_at(handle, group["frameTimes"], trial_index)
    if isinstance(frame_object, h5py.Dataset):
        times = np.asarray(frame_object[()]).ravel().astype(np.float64)
        if times.size == n_frames and np.any(np.isfinite(times)):
            return times
    # This is the exact synthetic time convention used in findPosition.m.
    return np.arange(1, n_frames + 1, dtype=np.float64) / 400.0


def trial_video_is_valid(handle: h5py.File, group: h5py.Group, trial_index: int) -> bool:
    if "NdroppedFrames" not in group:
        return True
    dropped_object = referenced_at(handle, group["NdroppedFrames"], trial_index)
    if not isinstance(dropped_object, h5py.Dataset):
        return False
    dropped = np.asarray(dropped_object[()], dtype=np.float64).ravel()
    return dropped.size > 0 and not np.all(np.isnan(dropped))


def interpolate_with_nan(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> np.ndarray:
    valid_x = np.isfinite(x)
    if valid_x.sum() < 2:
        return np.full(target.shape, np.nan, dtype=np.float64)
    # MATLAB interp1 does not extrapolate and propagates NaN-valued coordinate gaps.
    return np.interp(target, x[valid_x], y[valid_x], left=np.nan, right=np.nan)


def aligned_feature_position(
    handle: h5py.File,
    group: h5py.Group,
    trial_index: int,
    feature: str,
    go_cue: float,
    video_offset: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    names = feature_names(handle, group, trial_index)
    if feature not in names or not trial_video_is_valid(handle, group, trial_index):
        missing = np.full(N_TIME, np.nan, dtype=np.float64)
        return missing.copy(), missing.copy(), {}
    ts_object = referenced_at(handle, group["ts"], trial_index)
    if not isinstance(ts_object, h5py.Dataset):
        missing = np.full(N_TIME, np.nan, dtype=np.float64)
        return missing.copy(), missing.copy(), {}
    ts = trajectory_array(ts_object, len(names))
    feature_index = names.index(feature)
    frames = frame_times_for_trial(handle, group, trial_index, ts.shape[0])
    aligned_frames = frames - video_offset - go_cue
    x_raw = ts[:, 0, feature_index]
    y_raw = ts[:, 1, feature_index]
    x = interpolate_with_nan(aligned_frames, x_raw, OUTPUT_TIME)
    y = interpolate_with_nan(aligned_frames, y_raw, OUTPUT_TIME)
    diagnostic = {
        "aligned_frames": aligned_frames.astype(np.float32),
        "x_raw": x_raw.astype(np.float32),
        "y_raw": y_raw.astype(np.float32),
    }
    return x, y, diagnostic


def fill_nearest(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return values.copy()
    positions = np.arange(values.size)
    insertion = np.searchsorted(finite_indices, positions)
    left_position = np.clip(insertion - 1, 0, finite_indices.size - 1)
    right_position = np.clip(insertion, 0, finite_indices.size - 1)
    left = finite_indices[left_position]
    right = finite_indices[right_position]
    nearest = np.where(positions - left <= right - positions, left, right)
    output = values.copy()
    output[~np.isfinite(output)] = values[nearest[~np.isfinite(output)]]
    return output


def feature_speed(x: np.ndarray, y: np.ndarray, *, tongue: bool) -> tuple[np.ndarray, np.ndarray]:
    """Port findVelocity.m and return speed plus pre-fill visibility."""
    visibility = np.isfinite(x) | np.isfinite(y)
    if tongue:
        x_velocity = np.gradient(x)
        y_velocity = np.gradient(y)
        x_velocity[~np.isfinite(x_velocity)] = 0.0
        y_velocity[~np.isfinite(y_velocity)] = 0.0
    else:
        x_filled = fill_nearest(x)
        y_filled = fill_nearest(y)
        if not np.any(np.isfinite(x_filled)) or not np.any(np.isfinite(y_filled)):
            return np.full(x.shape, np.nan), visibility
        base_derivative = np.nanmedian(np.diff(np.column_stack((x_filled, y_filled)), axis=0), axis=0)
        x_velocity = np.gradient(x_filled) - base_derivative[0]
        # Preserve the supplied findVelocity.m behavior (it subtracts x baseline here).
        y_velocity = np.gradient(y_filled) - base_derivative[0]
        x_velocity = fill_nearest(x_velocity)
        y_velocity = fill_nearest(y_velocity)
    return np.hypot(x_velocity, y_velocity), visibility


def video_offset_seconds(handle: h5py.File) -> float:
    bit_start_neural = np.asarray(handle["obj/sglx/bitcode/bitstart"][()]).ravel()
    sampling_rate = float(np.asarray(handle["obj/sglx/fs"][()]).squeeze())
    bit_start_behavior = np.asarray(handle["obj/bp/ev/bitStart"][()]).ravel()
    return matlab_mode(bit_start_neural) / sampling_rate - matlab_mode(bit_start_behavior)


def load_motion_energy(path: Path) -> list[np.ndarray]:
    motion = loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
    data = np.atleast_1d(motion.data).ravel()
    return [np.asarray(item, dtype=np.float64).ravel() for item in data]


def aligned_motion_energy(
    handle: h5py.File,
    motion_data: list[np.ndarray],
    trial_index: int,
    go_cue: float,
    video_offset: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    group = trajectory_group(handle, 0)
    if trial_index >= len(motion_data) or not trial_video_is_valid(handle, group, trial_index):
        return np.full(N_TIME, np.nan), {}
    values = motion_data[trial_index]
    if values.size < 2:
        return np.full(N_TIME, np.nan), {}
    frame_object = referenced_at(handle, group["frameTimes"], trial_index)
    if isinstance(frame_object, h5py.Dataset):
        frames = np.asarray(frame_object[()]).ravel().astype(np.float64)
    else:
        frames = np.arange(1, values.size + 1, dtype=np.float64) / 400.0
    if frames.size != values.size:
        frames = np.arange(1, values.size + 1, dtype=np.float64) / 400.0
    aligned_frames = frames - video_offset - go_cue
    interpolated = interpolate_with_nan(aligned_frames, values, OUTPUT_TIME)
    interpolated = fill_nearest(interpolated)
    return interpolated, {
        "aligned_frames": aligned_frames.astype(np.float32),
        "raw": values.astype(np.float32),
    }


def discretize_session(
    values: np.ndarray, visibility: np.ndarray
) -> tuple[np.ndarray, float]:
    valid = visibility & np.isfinite(values)
    if not np.any(valid):
        return np.full(values.shape, 2, dtype=np.int8), float("nan")
    threshold = float(np.percentile(values[valid], 50))
    output = np.full(values.shape, 2, dtype=np.int8)
    output[valid & (values < threshold)] = 0
    output[valid & (values >= threshold)] = 1
    return output, threshold


def process_session(animal: str, date: str, probe_number: int) -> SessionResult:
    session_id = f"{animal}_{date}"
    data_path = DATA_DIR / f"data_structure_{session_id}.mat"
    motion_path = DATA_DIR / f"motionEnergy_{session_id}.mat"
    if not data_path.exists() or not motion_path.exists():
        raise FileNotFoundError(f"Missing paired data for {session_id}")

    started = time.perf_counter()
    with h5py.File(data_path, "r") as handle:
        n_trials_original = int(np.asarray(handle["obj/bp/Ntrials"][()]).squeeze())
        go_cue = np.asarray(handle["obj/bp/ev/goCue"][()]).ravel().astype(np.float64)
        if go_cue.size != n_trials_original:
            raise ValueError(f"{session_id}: goCue length mismatch")
        flags = trial_flags(handle)
        if not all(array.size == n_trials_original for array in flags.values()):
            raise ValueError(f"{session_id}: trial flag length mismatch")

        rates, neural_info = load_and_process_neural(handle, probe_number, go_cue, flags)
        if rates.shape[0] < 10:
            raise ValueError(f"{session_id}: only {rates.shape[0]} retained neurons")

        use = (
            np.isfinite(go_cue)
            & flags["have_ephys"]
            & ~flags["early"]
            & ~flags["stim"]
        )
        retained_trials = np.flatnonzero(use)
        if retained_trials.size < 2:
            raise ValueError(f"{session_id}: fewer than two retained trials")

        outcome_count = (
            flags["hit"].astype(np.int8)
            + flags["miss"].astype(np.int8)
            + flags["no"].astype(np.int8)
        )
        if not np.all(outcome_count[retained_trials] == 1):
            raise ValueError(f"{session_id}: hit/miss/no are not mutually exhaustive")

        offset = video_offset_seconds(handle)
        side_group = trajectory_group(handle, 0)
        bottom_group = trajectory_group(handle, 1)
        motion_data = load_motion_energy(motion_path)

        n_retained = retained_trials.size
        tongue_speed = np.full((n_retained, N_TIME), np.nan, dtype=np.float32)
        tongue_visible = np.zeros((n_retained, N_TIME), dtype=bool)
        paw_speed = np.full((n_retained, N_TIME), np.nan, dtype=np.float32)
        paw_visible = np.zeros((n_retained, N_TIME), dtype=bool)
        motion_energy = np.full((n_retained, N_TIME), np.nan, dtype=np.float32)
        motion_visible = np.zeros((n_retained, N_TIME), dtype=bool)
        diagnostic_trial: dict[str, Any] = {}

        for output_trial_index, source_trial_index in enumerate(retained_trials):
            tx, ty, tongue_raw = aligned_feature_position(
                handle,
                side_group,
                int(source_trial_index),
                "tongue",
                go_cue[source_trial_index],
                offset,
            )
            tspeed, tvisible = feature_speed(tx, ty, tongue=True)
            tongue_speed[output_trial_index] = tspeed
            tongue_visible[output_trial_index] = tvisible

            px, py, paw_raw = aligned_feature_position(
                handle,
                bottom_group,
                int(source_trial_index),
                "top_paw",
                go_cue[source_trial_index],
                offset,
            )
            pspeed, pvisible = feature_speed(px, py, tongue=False)
            paw_speed[output_trial_index] = pspeed
            paw_visible[output_trial_index] = pvisible

            me, motion_raw = aligned_motion_energy(
                handle,
                motion_data,
                int(source_trial_index),
                go_cue[source_trial_index],
                offset,
            )
            motion_energy[output_trial_index] = me
            motion_visible[output_trial_index] = np.isfinite(me)

            if output_trial_index == 0:
                diagnostic_trial = {
                    "source_trial_index": int(source_trial_index),
                    "tongue_raw": tongue_raw,
                    "paw_raw": paw_raw,
                    "motion_raw": motion_raw,
                    "tongue_speed": tspeed.astype(np.float32),
                    "tongue_visible": tvisible,
                    "paw_speed": pspeed.astype(np.float32),
                    "paw_visible": pvisible,
                    "motion_energy": me.astype(np.float32),
                }

        tongue_class, tongue_threshold = discretize_session(tongue_speed, tongue_visible)
        paw_class, paw_threshold = discretize_session(paw_speed, paw_visible)
        motion_class, motion_threshold = discretize_session(motion_energy, motion_visible)

        input_template = OUTPUT_TIME.reshape(1, -1).astype(np.float32)
        neural_trials: list[np.ndarray] = []
        input_trials: list[np.ndarray] = []
        output_trials: list[np.ndarray] = []

        static_counts = Counter()
        for output_trial_index, source_trial_index in enumerate(retained_trials):
            if flags["no"][source_trial_index]:
                lick_direction = 2
            elif flags["L"][source_trial_index]:
                lick_direction = 0
            elif flags["R"][source_trial_index]:
                lick_direction = 1
            else:
                raise ValueError(f"{session_id}: trial {source_trial_index + 1} has no direction")

            context = 0 if flags["autowater"][source_trial_index] else 1
            if flags["miss"][source_trial_index]:
                outcome = 0
            elif flags["hit"][source_trial_index]:
                outcome = 1
            else:
                outcome = 2

            output = np.empty((6, N_TIME), dtype=np.int8)
            output[0].fill(lick_direction)
            output[1].fill(context)
            output[2].fill(outcome)
            output[3] = tongue_class[output_trial_index]
            output[4] = paw_class[output_trial_index]
            output[5] = motion_class[output_trial_index]

            neural_trials.append(
                np.ascontiguousarray(rates[:, source_trial_index, :], dtype=np.float32)
            )
            input_trials.append(input_template)
            output_trials.append(output)
            static_counts.update(
                {f"lick_{lick_direction}": 1, f"context_{context}": 1, f"outcome_{outcome}": 1}
            )

        elapsed = time.perf_counter() - started
        metadata = {
            "session_id": session_id,
            "subject": animal,
            "date": date,
            "source_file": str(data_path),
            "motion_energy_file": str(motion_path),
            "alm_probe_matlab_index": int(probe_number),
            "n_trials_original": int(n_trials_original),
            "n_trials_retained": int(n_retained),
            "excluded_early": int(flags["early"].sum()),
            "excluded_stimulation": int((flags["stim"] & ~flags["early"]).sum()),
            "excluded_invalid_go_or_ephys": int(
                np.sum(~np.isfinite(go_cue) | ~flags["have_ephys"])
            ),
            "source_trial_indices_0based": retained_trials.astype(np.int32),
            "video_offset_seconds": float(offset),
            "tongue_velocity_median": tongue_threshold,
            "paw_velocity_median": paw_threshold,
            "motion_energy_median": motion_threshold,
            "n_neurons": int(rates.shape[0]),
            "n_well_isolated_neurons": neural_info["n_well_isolated"],
            "n_quality_selected_before_low_fr": neural_info["n_quality_selected"],
            "n_low_fr_removed": neural_info["n_low_fr_removed"],
            "static_class_counts": dict(static_counts),
            "processing_seconds": float(elapsed),
        }
        diagnostics = {
            **neural_info,
            "session_id": session_id,
            "sample_rates": rates[: min(20, rates.shape[0]), retained_trials[0], :].copy(),
            "retained_trials": retained_trials,
            "flags": {name: value[retained_trials].copy() for name, value in flags.items()},
            "tongue_class": tongue_class,
            "paw_class": paw_class,
            "motion_class": motion_class,
            "thresholds": (tongue_threshold, paw_threshold, motion_threshold),
            "diagnostic_trial": diagnostic_trial,
        }

    del rates, tongue_speed, paw_speed, motion_energy
    gc.collect()
    return SessionResult(
        neural=neural_trials,
        inputs=input_trials,
        outputs=output_trials,
        brain_region_idx=np.zeros(neural_trials[0].shape[0], dtype=np.int32),
        metadata=metadata,
        diagnostics=diagnostics,
    )


def plot_processing(result: SessionResult) -> None:
    diagnostic = result.diagnostics
    session_id = diagnostic["session_id"]
    trial = diagnostic["diagnostic_trial"]
    thresholds = diagnostic["thresholds"]
    figure, axes = plt.subplots(4, 2, figsize=(16, 14), constrained_layout=True)

    image = axes[0, 0].imshow(
        diagnostic["sample_rates"],
        aspect="auto",
        extent=(OUTPUT_TIME[0], OUTPUT_TIME[-1], diagnostic["sample_rates"].shape[0], 0),
        cmap="viridis",
    )
    axes[0, 0].axvline(0, color="white", linestyle="--")
    axes[0, 0].set(title="Aligned, binned, causally smoothed ALM rates", ylabel="Neuron")
    figure.colorbar(image, ax=axes[0, 0], label="Hz")

    mean_fr = diagnostic["mean_fr_all_quality_selected"]
    keep = diagnostic["keep_mask"]
    axes[0, 1].hist(mean_fr[~keep], bins=20, alpha=0.7, label="removed")
    axes[0, 1].hist(mean_fr[keep], bins=20, alpha=0.7, label="retained")
    axes[0, 1].axvline(LOW_FR_HZ, color="black", linestyle="--", label=">1 Hz")
    axes[0, 1].set(title="Unit curation after manual quality labels", xlabel="Reference mean firing rate (Hz)")
    axes[0, 1].legend()

    def plot_stream(ax: plt.Axes, speed: np.ndarray, visible: np.ndarray, classes: np.ndarray, threshold: float, title: str) -> None:
        ax.plot(OUTPUT_TIME, speed, color="tab:blue", linewidth=0.8, label="aligned value")
        if np.isfinite(threshold):
            ax.axhline(threshold, color="black", linestyle="--", label="session median")
        invisible = ~visible
        if np.any(invisible):
            ax.fill_between(OUTPUT_TIME, 0, 1, where=invisible, transform=ax.get_xaxis_transform(), color="gray", alpha=0.25, label="missing")
        ax2 = ax.twinx()
        ax2.step(OUTPUT_TIME, classes, where="mid", color="tab:orange", alpha=0.55)
        ax2.set_ylim(-0.2, 2.2)
        ax2.set_ylabel("class")
        ax.set(title=title, ylabel="value")
        ax.legend(loc="upper left")

    plot_stream(
        axes[1, 0],
        trial["tongue_speed"],
        trial["tongue_visible"],
        diagnostic["tongue_class"][0],
        thresholds[0],
        "Tongue: align → speed → visibility/median classes",
    )
    plot_stream(
        axes[1, 1],
        trial["paw_speed"],
        trial["paw_visible"],
        diagnostic["paw_class"][0],
        thresholds[1],
        "Top paw: align/fill → speed → visibility/median classes",
    )
    plot_stream(
        axes[2, 0],
        trial["motion_energy"],
        np.isfinite(trial["motion_energy"]),
        diagnostic["motion_class"][0],
        thresholds[2],
        "Motion energy: align/fill → median classes",
    )

    axes[2, 1].plot(OUTPUT_TIME, OUTPUT_TIME, color="black")
    axes[2, 1].axvline(0, color="tab:red", linestyle="--", label="go cue / WC water drop")
    axes[2, 1].set(
        title="Decoder input and temporal alignment",
        xlabel="Aligned time (s)",
        ylabel="Input: time from event (s)",
    )
    axes[2, 1].legend()

    output = result.outputs[0]
    for row, name in enumerate(
        ["lick direction", "context", "outcome", "tongue", "paw", "motion energy"]
    ):
        axes[3, 0].step(OUTPUT_TIME, output[row] + row * 3, where="mid", label=name)
    axes[3, 0].set(title="All six converted output streams (sample trial)", xlabel="Time (s)")
    axes[3, 0].legend(ncol=2, fontsize=8)

    flattened = np.concatenate(result.outputs, axis=1)
    distributions = []
    for row in range(flattened.shape[0]):
        values, counts = np.unique(flattened[row], return_counts=True)
        distributions.append({int(value): int(count) for value, count in zip(values, counts)})
    axes[3, 1].axis("off")
    axes[3, 1].text(
        0,
        1,
        "Session audit\n"
        + "\n".join(
            [
                f"source trial (1-based): {trial['source_trial_index'] + 1}",
                f"trials: {result.metadata['n_trials_retained']} / {result.metadata['n_trials_original']}",
                f"neurons: {result.metadata['n_neurons']}",
                f"video offset: {result.metadata['video_offset_seconds']:.6f} s",
                "output sample counts:",
                *[
                    f"  {name}: {distribution}"
                    for name, distribution in zip(
                        ["lick", "context", "outcome", "tongue", "paw", "motion"],
                        distributions,
                    )
                ],
            ]
        ),
        va="top",
        family="monospace",
        fontsize=9,
    )

    figure.suptitle(f"Conversion processing audit: {session_id}", fontsize=16)
    output_path = Path(f"/app/processing_{session_id}.png")
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    print(f"Saved processing plot: {output_path}", flush=True)


def validate_converted(data: dict[str, Any]) -> None:
    n_sessions = len(data["neural"])
    if not (n_sessions == len(data["input"]) == len(data["output"])):
        raise ValueError("Session list lengths differ")
    for session_index in range(n_sessions):
        n_trials = len(data["neural"][session_index])
        if n_trials < 2:
            raise ValueError(f"Session {session_index} has fewer than two trials")
        if not (
            n_trials == len(data["input"][session_index]) == len(data["output"][session_index])
        ):
            raise ValueError(f"Session {session_index} trial list lengths differ")
        n_neurons = data["neural"][session_index][0].shape[0]
        if len(data["brain_region_idx"][session_index]) != n_neurons:
            raise ValueError(f"Session {session_index} brain-region length mismatch")
        for neural, inputs, outputs in zip(
            data["neural"][session_index],
            data["input"][session_index],
            data["output"][session_index],
        ):
            if neural.shape != (n_neurons, N_TIME):
                raise ValueError(f"Unexpected neural shape {neural.shape}")
            if inputs.shape != (1, N_TIME) or outputs.shape != (6, N_TIME):
                raise ValueError(f"Unexpected input/output shapes {inputs.shape}/{outputs.shape}")
            if not np.all(np.isfinite(neural)) or not np.all(np.isfinite(inputs)):
                raise ValueError("Neural/input contains nonfinite data")
            if not np.all(np.isin(outputs, [0, 1, 2])):
                raise ValueError("Output contains invalid categorical values")
            if np.any(outputs[1] > 1):
                raise ValueError("Context output contains invalid class")


def build_dataset(session_specs: list[tuple[str, str, int]], show_processing: bool) -> dict[str, Any]:
    neural: list[list[np.ndarray]] = []
    inputs: list[list[np.ndarray]] = []
    outputs: list[list[np.ndarray]] = []
    brain_region_idx: list[np.ndarray] = []
    session_info: list[dict[str, Any]] = []
    subjects: list[str] = []
    subject_lookup: dict[str, int] = {}
    subject_idx: list[int] = []

    conversion_started = time.perf_counter()
    for session_index, (animal, date, probe_number) in enumerate(session_specs):
        print(
            f"[{session_index + 1}/{len(session_specs)}] Processing {animal}_{date}, ALM probe {probe_number}",
            flush=True,
        )
        result = process_session(animal, date, probe_number)
        neural.append(result.neural)
        inputs.append(result.inputs)
        outputs.append(result.outputs)
        brain_region_idx.append(result.brain_region_idx)
        session_info.append(result.metadata)
        if animal not in subject_lookup:
            subject_lookup[animal] = len(subjects)
            subjects.append(animal)
        subject_idx.append(subject_lookup[animal])

        print(
            f"  {result.metadata['n_trials_retained']} trials, "
            f"{result.metadata['n_neurons']} neurons, "
            f"{result.metadata['processing_seconds']:.2f} s",
            flush=True,
        )
        if show_processing and session_index < 2:
            plot_processing(result)
        del result
        gc.collect()

    total_seconds = time.perf_counter() - conversion_started
    data: dict[str, Any] = {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int32),
        "brain_regions": ["ALM"],
        "brain_region_idx": brain_region_idx,
        "input_names": ["time_from_go_cue_seconds"],
        "output_names": [
            "lick_direction",
            "behavioral_context",
            "outcome",
            "tongue_velocity",
            "paw_velocity",
            "motion_energy",
        ],
        "output_values": [
            ["left", "right", "none"],
            ["WC", "DR"],
            ["incorrect", "correct", "ignore"],
            ["below_session_median", "at_or_above_session_median", "not_visible"],
            ["below_session_median", "at_or_above_session_median", "not_visible"],
            ["below_session_median", "at_or_above_session_median", "no_video"],
        ],
        "metadata": {
            "task_description": (
                "Two-context directional licking task: delayed response (DR) and "
                "water-cued (WC); decoder targets trial direction/context/outcome and "
                "time-varying tongue, paw, and motion-energy classes."
            ),
            "time_bin_size": 5.0,
            "temporal_alignment_event": (
                "obj.bp.ev.goCue onset (auditory go cue in DR; water-drop onset in WC)"
            ),
            "off_start": -2.5,
            "off_end": 2.5,
            "time_bin_centers_seconds": OUTPUT_TIME.copy(),
            "neural_representation": (
                "ALM firing rate (Hz), 5 ms bins, 15-sample causal Gaussian smoothing"
            ),
            "neuron_curation": (
                "author-selected ALM probe; findClusters quality='all' exclusions; "
                "reference condition-averaged firing rate strictly >1 Hz"
            ),
            "trial_curation": (
                "finite go cue and ephys; excludes early-lick and stimulation trials; "
                "retains ignore trials for requested decoder class"
            ),
            "velocity_definition": (
                "Euclidean speed from reference x/y first derivatives; side-view central "
                "tongue and bottom-view top_paw; pre-fill DLC missingness defines class 2"
            ),
            "discretization": (
                "Per session and stream, class 0 < finite retained-sample 50th percentile, "
                "class 1 >= percentile, class 2 missing/not visible/no video"
            ),
            "source_paper": "Separating cognitive and motor processes in the behaving mouse",
            "paper_reported_context_sessions": 12,
            "paper_reported_context_units": 522,
            "paper_reported_context_single_units": 214,
            "paper_reported_context_mice": 6,
            "native_subject_id_count": len(subjects),
            "paper_subject_count_discrepancy": (
                "Author Figure 8 loaders and native filenames identify seven IDs; no "
                "undocumented merge was applied to force the paper's six-mouse statement."
            ),
            "session_info": session_info,
            "conversion_seconds": float(total_seconds),
        },
    }
    validate_converted(data)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all 12 context sessions (default)")
    mode.add_argument("--sample", action="store_true", help="Process the first two context sessions")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing_<session_id>.png for up to two sessions",
    )
    arguments = parser.parse_args()

    sessions = CONTEXT_SESSIONS[:2] if arguments.sample else CONTEXT_SESSIONS
    print(
        f"Mode: {'sample' if arguments.sample else 'full'}; sessions={len(sessions)}; "
        f"bins={N_TIME}; dt={DT * 1000:.1f} ms",
        flush=True,
    )
    data = build_dataset(sessions, arguments.show_processing)

    arguments.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    save_started = time.perf_counter()
    with arguments.outpicklefile.open("wb") as output_file:
        pickle.dump(data, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    save_seconds = time.perf_counter() - save_started
    size_mb = arguments.outpicklefile.stat().st_size / (1024**2)
    print(
        f"Saved {arguments.outpicklefile} ({size_mb:.1f} MiB) in {save_seconds:.2f} s",
        flush=True,
    )
    print(
        f"Summary: {len(data['neural'])} sessions, "
        f"{sum(len(session) for session in data['neural'])} trials, "
        f"{sum(len(index) for index in data['brain_region_idx'])} session-summed neurons",
        flush=True,
    )


if __name__ == "__main__":
    main()
