#!/usr/bin/env python3
"""Convert the Hasnain/Birnbaum ALM dataset for neural decoding.

The implementation follows the processing in ``code/DataLoadingScripts``:
go-cue alignment, 5 ms spike-rate bins, the repository's 15-bin causal
Gaussian smoother, published probe choices, manual cluster curation, and a
1 Hz firing-rate cutoff.  Early-lick and photostimulation trials are removed;
ignore trials are retained because ignore is a requested decoder outcome.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.signal import convolve


DATA_ROOT = Path("/app/data")
OUTPUT_PATH = Path("/app/converted_data.pkl")

TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 200.0
SMOOTH_BINS = 15
LOW_FR_HZ = 1.0

# These are the ALM probes selected by the repository's
# DataLoadingScripts/Recording and video/load*ALMVideo.m files.
ALM_PROBES = {
    "EKH1_2021-08-07": (2,),
    "EKH3_2021-08-11": (2,),
    "JEB13_2022-09-13": (2,),
    "JEB13_2022-09-14": (2,),
    "JEB13_2022-09-21": (1,),
    "JEB13_2022-09-24": (1,),
    "JEB13_2022-09-25": (1,),
    "JEB14_2022-08-22": (1,),
    "JEB14_2022-08-23": (1,),
    "JEB14_2022-08-24": (1,),
    "JEB14_2022-08-25": (1,),
    "JEB15_2022-07-26": (1, 2),
    "JEB15_2022-07-27": (1, 2),
    "JEB15_2022-07-28": (1, 2),
    "JEB15_2022-07-29": (2,),
    "JEB19_2023-04-18": (1,),
    "JEB19_2023-04-19": (1,),
    "JEB19_2023-04-20": (1,),
    "JEB19_2023-04-21": (1,),
    "JEB6_2021-04-18": (2,),
    "JEB7_2021-04-29": (1,),
    "JEB7_2021-04-30": (1,),
    "JGR2_2021-11-16": (1,),
    "JGR2_2021-11-17": (1,),
    "JGR3_2021-11-18": (1,),
}

# The same repository loaders select these 19 sessions for the randomized-delay
# experiment. Three additional files in that directory are deliberately absent:
# JEB23 2023-10-20 and JEB24 2023-10-03/04 are commented out by the authors and
# do not have released motion-energy companions.
RANDOMIZED_DELAY_PROBES = {
    "JEB11_2022-05-10": (1,),
    "JEB11_2022-05-11": (1,),
    "JEB12_2022-05-12": (1,),
    "JEB12_2022-05-13": (1,),
    "JEB23_2023-10-10": (1,),
    "JEB23_2023-10-11": (1,),
    "JEB23_2023-10-12": (1,),
    "JEB23_2023-10-13": (1,),
    "JEB23_2023-10-18": (1,),
    "JEB23_2023-10-19": (1,),
    "JEB23_2023-10-21": (1,),
    "JEB24_2023-10-23": (1,),
    "JEB24_2023-10-24": (1,),
    "JEB24_2023-10-25": (1,),
    "JEB24_2023-10-26": (1,),
    "JEB24_2023-10-27": (1,),
    "JEB24_2023-10-31": (1,),
    "JEB24_2023-11-02": (1,),
    "JEB24_2023-11-03": (1,),
}

SESSION_GROUPS = (
    ("Ephys_Behavior", "fixed_delay", ALM_PROBES),
    ("RandomizedDelay_Ephys_Behavior", "randomized_delay", RANDOMIZED_DELAY_PROBES),
)

REJECTED_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}


def _vector(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dataset).reshape(-1, order="F")


def _mode(values: np.ndarray) -> float:
    values = np.asarray(values).ravel()
    values = values[np.isfinite(values)]
    unique, counts = np.unique(values, return_counts=True)
    return float(unique[np.argmax(counts)])


def _fit_trial_mask(values: np.ndarray, ntrials: int) -> np.ndarray:
    """Align bookkeeping masks to the Bpod trial count.

    A few sessions have one trailing acquisition bookkeeping entry after the
    Bpod object was trimmed. Existing entries map one-to-one from trial 1, as
    confirmed by trials.bp.sglxFileNum/BPnum, so trim only the trailing excess.
    """
    values = np.asarray(values).reshape(-1).astype(bool)
    if values.size >= ntrials:
        return values[:ntrials]
    return np.pad(values, (0, ntrials - values.size), constant_values=False)


def _decode_char(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).reshape(-1, order="F")
    return "".join(chr(int(value)) for value in values if int(value) != 0)


def _decode_cellstr(matfile: h5py.File, dataset: h5py.Dataset) -> list[str]:
    result = []
    for reference in np.asarray(dataset).reshape(-1, order="F"):
        item = matfile[reference]
        matlab_class = item.attrs.get("MATLAB_class", b"")
        if matlab_class == b"char":
            result.append(_decode_char(item))
        else:
            result.append("")
    return result


def _deref_vector(matfile: h5py.File, refs: h5py.Dataset, index: int) -> np.ndarray:
    ref = np.asarray(refs)[index, 0]
    return np.asarray(matfile[ref]).reshape(-1, order="F")


def _session_key(path: Path) -> str:
    match = re.fullmatch(r"data_structure_(.+)\.mat", path.name)
    if match is None:
        raise ValueError(f"Unexpected data filename: {path.name}")
    return match.group(1)


def _gaussian_kernel(length: int) -> np.ndarray:
    # MATLAB gausswin(length) uses alpha=2.5. This is its defining formula.
    n = np.arange(length, dtype=np.float64) - (length - 1.0) / 2.0
    kernel = np.exp(-0.5 * (2.5 * n / (length / 2.0)) ** 2)
    # mySmooth.m zeros the first floor(N/2) weights to make the filter causal.
    kernel[: length // 2] = 0.0
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


SMOOTH_KERNEL = _gaussian_kernel(SMOOTH_BINS)
TIME = (np.arange(int(round((TMAX - TMIN) / DT)), dtype=np.float32) * DT
        + TMIN + DT / 2.0)
EDGES = TMIN + np.arange(TIME.size + 1, dtype=np.float64) * DT


def _smooth_counts(counts: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of mySmooth(counts, 15, 'reflect')."""
    # Despite its name, repository code prepends the first N samples rather
    # than reversing them. Preserve that exact behavior.
    padded = np.concatenate((counts[:, :SMOOTH_BINS], counts), axis=1)
    smoothed = convolve(padded, SMOOTH_KERNEL[None, :], mode="same", method="direct")
    return smoothed[:, SMOOTH_BINS:] / np.float32(DT)


def _load_neural(matfile: h5py.File, probes: tuple[int, ...]) -> tuple[np.ndarray, dict]:
    ntrials = int(_vector(matfile["obj/bp/Ntrials"])[0])
    units = []
    quality_counts: dict[str, int] = {}

    for probe in probes:
        cluster_group = matfile[np.asarray(matfile["obj/clu"])[probe - 1, 0]]
        for cluster_idx in range(cluster_group["trial"].shape[0]):
            quality_obj = matfile[np.asarray(cluster_group["quality"])[cluster_idx, 0]]
            if quality_obj.attrs.get("MATLAB_class", b"") == b"char":
                quality = _decode_char(quality_obj).strip()
            else:
                quality = ""
            quality_counts[quality or "unlabelled"] = quality_counts.get(quality or "unlabelled", 0) + 1
            if quality.lower() in REJECTED_QUALITIES:
                continue

            spike_trials = _deref_vector(matfile, cluster_group["trial"], cluster_idx).astype(np.int64) - 1
            spike_times = _deref_vector(matfile, cluster_group["trialtm"], cluster_idx).astype(np.float64)
            go_cue = _vector(matfile["obj/bp/ev/goCue"]).astype(np.float64)
            valid_trial = (spike_trials >= 0) & (spike_trials < ntrials)
            spike_trials = spike_trials[valid_trial]
            aligned = spike_times[valid_trial] - go_cue[spike_trials]
            valid_time = (aligned >= TMIN) & (aligned < TMAX)
            spike_trials = spike_trials[valid_time]
            aligned = aligned[valid_time]

            # floor is the histc bin assignment for half-open bins.
            time_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
            counts = np.zeros((ntrials, TIME.size), dtype=np.float32)
            np.add.at(counts, (spike_trials, time_idx), 1.0)
            units.append(_smooth_counts(counts))

    if not units:
        raise ValueError("No manually curated units found")
    neural = np.stack(units, axis=1).astype(np.float32, copy=False)
    mean_fr = neural.mean(axis=(0, 2), dtype=np.float64)
    keep = mean_fr > LOW_FR_HZ
    return neural[:, keep, :], {
        "n_curated_before_fr_filter": int(len(units)),
        "n_units_after_fr_filter": int(keep.sum()),
        "mean_fr_hz_min_retained": float(mean_fr[keep].min()) if keep.any() else None,
        "quality_counts_before_curation": quality_counts,
    }


def _load_neural_v5(obj, probes: tuple[int, ...]) -> tuple[np.ndarray, dict]:
    """Equivalent neural loader for conventional (pre-v7.3) MAT files."""
    ntrials = int(obj.bp.Ntrials)
    units = []
    quality_counts: dict[str, int] = {}

    # All selected legacy sessions contain one probe. With squeeze_me=True,
    # MATLAB's 1-element outer cell disappears and obj.clu is the unit array.
    if probes != (1,):
        raise ValueError(f"Legacy MAT loader expected probe 1, received {probes}")
    clusters = np.asarray(obj.clu, dtype=object).reshape(-1)
    go_cue = np.asarray(obj.bp.ev.goCue, dtype=np.float64).reshape(-1)
    for cluster in clusters:
        raw_quality = getattr(cluster, "quality", "")
        quality = raw_quality.strip() if isinstance(raw_quality, str) else ""
        quality_counts[quality or "unlabelled"] = quality_counts.get(quality or "unlabelled", 0) + 1
        if quality.lower() in REJECTED_QUALITIES:
            continue
        spike_trials = np.asarray(cluster.trial, dtype=np.int64).reshape(-1) - 1
        spike_times = np.asarray(cluster.trialtm, dtype=np.float64).reshape(-1)
        valid_trial = (spike_trials >= 0) & (spike_trials < ntrials)
        spike_trials = spike_trials[valid_trial]
        aligned = spike_times[valid_trial] - go_cue[spike_trials]
        valid_time = (aligned >= TMIN) & (aligned < TMAX)
        spike_trials = spike_trials[valid_time]
        aligned = aligned[valid_time]
        time_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
        counts = np.zeros((ntrials, TIME.size), dtype=np.float32)
        np.add.at(counts, (spike_trials, time_idx), 1.0)
        units.append(_smooth_counts(counts))

    if not units:
        raise ValueError("No manually curated units found")
    neural = np.stack(units, axis=1).astype(np.float32, copy=False)
    mean_fr = neural.mean(axis=(0, 2), dtype=np.float64)
    keep = mean_fr > LOW_FR_HZ
    return neural[:, keep, :], {
        "n_curated_before_fr_filter": int(len(units)),
        "n_units_after_fr_filter": int(keep.sum()),
        "mean_fr_hz_min_retained": float(mean_fr[keep].min()) if keep.any() else None,
        "quality_counts_before_curation": quality_counts,
    }


def _nearest_fill(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return values
    missing = np.flatnonzero(~np.isfinite(values))
    if missing.size:
        pos = np.searchsorted(finite, missing)
        left_pos = np.maximum(pos - 1, 0)
        right_pos = np.minimum(pos, finite.size - 1)
        left = finite[left_pos]
        right = finite[right_pos]
        choose_right = np.abs(right - missing) < np.abs(missing - left)
        nearest = np.where(choose_right, right, left)
        values[missing] = values[nearest]
    return values


def _interp(source_time: np.ndarray, values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    """Linear interpolation with MATLAB interp1-style NaNs out of bounds."""
    if source_time.size < 2 or values.size != source_time.size:
        return np.full(target_time.shape, np.nan, dtype=np.float64)
    order = np.argsort(source_time)
    source_time = source_time[order]
    values = values[order]
    return np.interp(target_time, source_time, values, left=np.nan, right=np.nan)


def _feature_index(matfile: h5py.File, trajectory_group: h5py.Group,
                   trial: int, name: str) -> int:
    names_obj = matfile[np.asarray(trajectory_group["featNames"])[trial, 0]]
    names = _decode_cellstr(matfile, names_obj)
    try:
        return names.index(name)
    except ValueError as exc:
        raise ValueError(f"DLC feature {name!r} absent; available: {names}") from exc


def _trajectory_xy(matfile: h5py.File, trajectory_group: h5py.Group,
                   trial: int, feature: str, aligned_time: np.ndarray,
                   video_shift: float, go_cue: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dropped_obj = matfile[np.asarray(trajectory_group["NdroppedFrames"])[trial, 0]]
    dropped = np.asarray(dropped_obj).reshape(-1, order="F")
    if dropped.size == 0 or not np.isfinite(dropped[0]):
        nan = np.full(aligned_time.shape, np.nan, dtype=np.float64)
        return nan, nan.copy(), np.zeros(aligned_time.shape, dtype=bool)

    ts_obj = matfile[np.asarray(trajectory_group["ts"])[trial, 0]]
    frame_obj = matfile[np.asarray(trajectory_group["frameTimes"])[trial, 0]]
    ts = np.asarray(ts_obj, dtype=np.float64)
    frame_times = np.asarray(frame_obj).reshape(-1, order="F")
    if ts.ndim != 3 or frame_times.size == 0:
        nan = np.full(aligned_time.shape, np.nan, dtype=np.float64)
        return nan, nan.copy(), np.zeros(aligned_time.shape, dtype=bool)

    feature_idx = _feature_index(matfile, trajectory_group, trial, feature)
    # HDF5 dimensions are reversed from MATLAB: feature x coordinate x frame.
    x_raw = ts[feature_idx, 0, :]
    y_raw = ts[feature_idx, 1, :]
    source_time = frame_times - video_shift - go_cue
    x = _interp(source_time, x_raw, aligned_time)
    y = _interp(source_time, y_raw, aligned_time)
    visible = np.isfinite(x) & np.isfinite(y)
    return x, y, visible


def _trajectory_xy_v5(trajectory: np.ndarray, trial: int, feature: str,
                      aligned_time: np.ndarray, video_shift: float,
                      go_cue: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    item = np.asarray(trajectory, dtype=object).reshape(-1)[trial]
    dropped = np.asarray(item.NdroppedFrames).reshape(-1)
    if dropped.size == 0 or not np.isfinite(float(dropped[0])):
        nan = np.full(aligned_time.shape, np.nan, dtype=np.float64)
        return nan, nan.copy(), np.zeros(aligned_time.shape, dtype=bool)
    names = [str(x) for x in np.asarray(item.featNames, dtype=object).reshape(-1)]
    try:
        feature_idx = names.index(feature)
    except ValueError as exc:
        raise ValueError(f"DLC feature {feature!r} absent; available: {names}") from exc
    ts = np.asarray(item.ts, dtype=np.float64)
    frame_times = np.asarray(item.frameTimes, dtype=np.float64).reshape(-1)
    if ts.ndim != 3 or frame_times.size == 0:
        nan = np.full(aligned_time.shape, np.nan, dtype=np.float64)
        return nan, nan.copy(), np.zeros(aligned_time.shape, dtype=bool)
    source_time = frame_times - video_shift - go_cue
    x = _interp(source_time, ts[:, 0, feature_idx], aligned_time)
    y = _interp(source_time, ts[:, 1, feature_idx], aligned_time)
    visible = np.isfinite(x) & np.isfinite(y)
    return x, y, visible


def _speed(x: np.ndarray, y: np.ndarray, tongue: bool) -> np.ndarray:
    """Match findVelocity.m (including its shared x/y baseline subtraction)."""
    xy = np.column_stack((x, y))
    differences = np.diff(xy, axis=0)
    baseline = np.asarray([
        np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else np.nan
        for column in differences.T
    ])
    if tongue:
        xvel = np.gradient(x)
        yvel = np.gradient(y)
        xvel[~np.isfinite(xvel)] = 0.0
        yvel[~np.isfinite(yvel)] = 0.0
    else:
        x_filled = _nearest_fill(x)
        y_filled = _nearest_fill(y)
        xvel = np.gradient(x_filled) - baseline[0]
        # The repository subtracts basederiv(1) from both components.
        yvel = np.gradient(y_filled) - baseline[0]
    return np.hypot(xvel, yvel)


def _motion_trials(motion_path: Path, ntrials: int) -> np.ndarray:
    me_obj = loadmat(motion_path, squeeze_me=True, struct_as_record=False,
                     variable_names=["me"])["me"]
    motion_data = me_obj.data
    # Some released files wrap the actual data in a second ``me`` struct.
    # This is the same ``if isstruct(me.data); me.data=me.data.data`` case
    # handled by DataLoadingScripts/loadMotionEnergy.m.
    if hasattr(motion_data, "data"):
        motion_data = motion_data.data
    trials = np.asarray(motion_data, dtype=object).reshape(-1)
    if trials.size != ntrials:
        raise ValueError(f"Motion-energy trials ({trials.size}) != Bpod trials ({ntrials})")
    return trials


def _finalize_behavior_outputs(
    tongue_speed: np.ndarray, paw_speed: np.ndarray, motion: np.ndarray,
    tongue_visible: np.ndarray, paw_visible: np.ndarray, motion_video: np.ndarray,
    selected: np.ndarray, hit: np.ndarray, miss: np.ndarray, ignore: np.ndarray,
    right_target: np.ndarray, water_cued: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    tongue_values = tongue_speed[tongue_visible & np.isfinite(tongue_speed)]
    paw_values = paw_speed[paw_visible & np.isfinite(paw_speed)]
    motion_values = motion[motion_video & np.isfinite(motion)]
    if not (tongue_values.size and paw_values.size and motion_values.size):
        raise ValueError("A movement stream has no valid samples")
    thresholds = {
        "tongue_velocity_median": float(np.percentile(tongue_values, 50)),
        "paw_velocity_median": float(np.percentile(paw_values, 50)),
        "motion_energy_median": float(np.percentile(motion_values, 50)),
    }

    nsel = selected.size
    output = np.empty((nsel, 6, TIME.size), dtype=np.int8)
    for out_trial, source_trial in enumerate(selected):
        output[out_trial, 3:6, :] = 2
        valid = tongue_visible[out_trial] & np.isfinite(tongue_speed[out_trial])
        output[out_trial, 3, valid] = (
            tongue_speed[out_trial, valid] >= thresholds["tongue_velocity_median"])
        valid = paw_visible[out_trial] & np.isfinite(paw_speed[out_trial])
        output[out_trial, 4, valid] = (
            paw_speed[out_trial, valid] >= thresholds["paw_velocity_median"])
        valid = motion_video[out_trial] & np.isfinite(motion[out_trial])
        output[out_trial, 5, valid] = (
            motion[out_trial, valid] >= thresholds["motion_energy_median"])

        if ignore[source_trial]:
            lick_direction, outcome = 2, 2  # none, ignore
        elif hit[source_trial]:
            lick_direction = 1 if right_target[source_trial] else 0
            outcome = 1  # correct
        elif miss[source_trial]:
            lick_direction = 0 if right_target[source_trial] else 1
            outcome = 0  # incorrect
        else:
            raise ValueError(f"Trial {source_trial + 1} has no hit/miss/ignore outcome")
        context = 0 if water_cued[source_trial] else 1  # WC, DR
        output[out_trial, 0:3, :] = np.asarray(
            (lick_direction, context, outcome), dtype=np.int8)[:, None]
    return output, water_cued[selected], thresholds


def _load_behavior(matfile: h5py.File, motion_path: Path,
                   selected: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    ntrials = int(_vector(matfile["obj/bp/Ntrials"])[0])
    go_cue = _vector(matfile["obj/bp/ev/goCue"]).astype(np.float64)
    hit = _vector(matfile["obj/bp/hit"]).astype(bool)
    miss = _vector(matfile["obj/bp/miss"]).astype(bool)
    ignore = _vector(matfile["obj/bp/no"]).astype(bool)
    right_target = _vector(matfile["obj/bp/R"]).astype(bool)
    water_cued = _vector(matfile["obj/bp/autowater"]).astype(bool)
    have_video = _vector(matfile["obj/trials/bp/haveVid"]).astype(bool)

    fs = float(_vector(matfile["obj/sglx/fs"])[0])
    video_shift = (_mode(matfile["obj/sglx/bitcode/bitstart"]) / fs
                   - _mode(matfile["obj/bp/ev/bitStart"]))
    trajectory_side = matfile[np.asarray(matfile["obj/traj"])[0, 0]]
    trajectory_bottom = matfile[np.asarray(matfile["obj/traj"])[1, 0]]

    motion_trials = _motion_trials(motion_path, ntrials)

    nsel = selected.size
    tongue_speed = np.full((nsel, TIME.size), np.nan, dtype=np.float32)
    paw_speed = np.full((nsel, TIME.size), np.nan, dtype=np.float32)
    motion = np.full((nsel, TIME.size), np.nan, dtype=np.float32)
    tongue_visible = np.zeros((nsel, TIME.size), dtype=bool)
    paw_visible = np.zeros((nsel, TIME.size), dtype=bool)
    motion_video = np.zeros((nsel, TIME.size), dtype=bool)

    for out_trial, source_trial in enumerate(selected):
        if not have_video[source_trial]:
            continue

        tx, ty, tongue_vis = _trajectory_xy(
            matfile, trajectory_side, source_trial, "tongue", TIME,
            video_shift, go_cue[source_trial])
        tongue_speed[out_trial] = _speed(tx, ty, tongue=True)
        tongue_visible[out_trial] = tongue_vis

        # Both paws were tracked in the bottom view. Average their speed where
        # available; this gives one requested paw-velocity stream without
        # privileging either paw.
        paw_speeds = []
        paw_masks = []
        for feature in ("top_paw", "bottom_paw"):
            px, py, paw_vis = _trajectory_xy(
                matfile, trajectory_bottom, source_trial, feature, TIME,
                video_shift, go_cue[source_trial])
            paw_speeds.append(_speed(px, py, tongue=False))
            paw_masks.append(paw_vis)
        paw_stack = np.stack(paw_speeds)
        mask_stack = np.stack(paw_masks)
        count = mask_stack.sum(axis=0)
        summed = np.where(mask_stack, paw_stack, 0.0).sum(axis=0)
        paw_speed[out_trial] = np.divide(summed, count, out=np.full(TIME.shape, np.nan), where=count > 0)
        paw_visible[out_trial] = count > 0

        frame_obj = matfile[np.asarray(trajectory_side["frameTimes"])[source_trial, 0]]
        frame_times = np.asarray(frame_obj).reshape(-1, order="F")
        raw_motion = np.asarray(motion_trials[source_trial], dtype=np.float64).reshape(-1)
        if frame_times.size >= 2 and raw_motion.size == frame_times.size:
            source_time = frame_times - video_shift - go_cue[source_trial]
            interp_motion = _interp(source_time, raw_motion, TIME)
            if np.isfinite(interp_motion).any():
                # loadMotionEnergy.m fills edge NaNs with the nearest value.
                interp_motion = _nearest_fill(interp_motion)
                motion[out_trial] = interp_motion
                motion_video[out_trial] = True

    return _finalize_behavior_outputs(
        tongue_speed, paw_speed, motion, tongue_visible, paw_visible, motion_video,
        selected, hit, miss, ignore, right_target, water_cued)


def _load_behavior_v5(obj, motion_path: Path,
                      selected: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Behavior/video loader for conventional (pre-v7.3) MAT files."""
    ntrials = int(obj.bp.Ntrials)
    go_cue = np.asarray(obj.bp.ev.goCue, dtype=np.float64).reshape(-1)
    hit = np.asarray(obj.bp.hit).reshape(-1).astype(bool)
    miss = np.asarray(obj.bp.miss).reshape(-1).astype(bool)
    ignore = np.asarray(obj.bp.no).reshape(-1).astype(bool)
    right_target = np.asarray(obj.bp.R).reshape(-1).astype(bool)
    water_cued = np.asarray(obj.bp.autowater).reshape(-1).astype(bool)
    have_video = np.asarray(obj.trials.bp.haveVid).reshape(-1).astype(bool)
    fs = float(np.asarray(obj.sglx.fs).reshape(-1)[0])
    video_shift = (_mode(np.asarray(obj.sglx.bitcode.bitstart)) / fs
                   - _mode(np.asarray(obj.bp.ev.bitStart)))
    trajectory_side = np.asarray(obj.traj, dtype=object).reshape(-1)[0]
    trajectory_bottom = np.asarray(obj.traj, dtype=object).reshape(-1)[1]
    motion_trials = _motion_trials(motion_path, ntrials)

    nsel = selected.size
    tongue_speed = np.full((nsel, TIME.size), np.nan, dtype=np.float32)
    paw_speed = np.full((nsel, TIME.size), np.nan, dtype=np.float32)
    motion = np.full((nsel, TIME.size), np.nan, dtype=np.float32)
    tongue_visible = np.zeros((nsel, TIME.size), dtype=bool)
    paw_visible = np.zeros((nsel, TIME.size), dtype=bool)
    motion_video = np.zeros((nsel, TIME.size), dtype=bool)
    side_trials = np.asarray(trajectory_side, dtype=object).reshape(-1)

    for out_trial, source_trial in enumerate(selected):
        if not have_video[source_trial]:
            continue
        tx, ty, tongue_vis = _trajectory_xy_v5(
            trajectory_side, source_trial, "tongue", TIME,
            video_shift, go_cue[source_trial])
        tongue_speed[out_trial] = _speed(tx, ty, tongue=True)
        tongue_visible[out_trial] = tongue_vis

        paw_speeds = []
        paw_masks = []
        for feature in ("top_paw", "bottom_paw"):
            px, py, paw_vis = _trajectory_xy_v5(
                trajectory_bottom, source_trial, feature, TIME,
                video_shift, go_cue[source_trial])
            paw_speeds.append(_speed(px, py, tongue=False))
            paw_masks.append(paw_vis)
        paw_stack = np.stack(paw_speeds)
        mask_stack = np.stack(paw_masks)
        count = mask_stack.sum(axis=0)
        summed = np.where(mask_stack, paw_stack, 0.0).sum(axis=0)
        paw_speed[out_trial] = np.divide(
            summed, count, out=np.full(TIME.shape, np.nan), where=count > 0)
        paw_visible[out_trial] = count > 0

        frame_times = np.asarray(
            side_trials[source_trial].frameTimes, dtype=np.float64).reshape(-1)
        raw_motion = np.asarray(motion_trials[source_trial], dtype=np.float64).reshape(-1)
        if frame_times.size >= 2 and raw_motion.size == frame_times.size:
            source_time = frame_times - video_shift - go_cue[source_trial]
            interp_motion = _interp(source_time, raw_motion, TIME)
            if np.isfinite(interp_motion).any():
                motion[out_trial] = _nearest_fill(interp_motion)
                motion_video[out_trial] = True

    return _finalize_behavior_outputs(
        tongue_speed, paw_speed, motion, tongue_visible, paw_visible, motion_video,
        selected, hit, miss, ignore, right_target, water_cued)


def convert(data_root: Path = DATA_ROOT, output_path: Path = OUTPUT_PATH,
            limit: int | None = None) -> dict:
    sessions = []
    for directory_name, task_variant, probe_map in SESSION_GROUPS:
        directory = data_root / directory_name
        available = {_session_key(path): path for path in directory.glob("data_structure_*.mat")}
        missing = sorted(set(probe_map) - set(available))
        if missing:
            raise ValueError(f"Missing selected sessions in {directory}: {missing}")
        for key in sorted(probe_map):
            sessions.append((available[key], directory / f"motionEnergy_{key}.mat",
                             probe_map[key], task_variant))
    if limit is not None:
        sessions = sessions[:limit]

    neural_sessions: list[list[np.ndarray]] = []
    input_sessions: list[list[np.ndarray]] = []
    output_sessions: list[list[np.ndarray]] = []
    session_subjects: list[str] = []
    region_indices: list[np.ndarray] = []
    session_info = []

    for session_idx, (data_path, motion_path, probes, task_variant) in enumerate(sessions, start=1):
        key = _session_key(data_path)
        subject, date = key.split("_", 1)
        if not motion_path.exists():
            raise FileNotFoundError(motion_path)
        print(f"[{session_idx}/{len(sessions)}] {key}", flush=True)

        if h5py.is_hdf5(data_path):
            with h5py.File(data_path, "r") as matfile:
                early = _vector(matfile["obj/bp/early"]).astype(bool)
                stimulated = _vector(matfile["obj/bp/stim/enable"]).astype(bool)
                have_ephys = _fit_trial_mask(
                    _vector(matfile["obj/trials/bp/haveEphys"]), early.size)
                candidate_trials = np.flatnonzero(~early & ~stimulated & have_ephys)
                if candidate_trials.size < 2:
                    raise ValueError(f"{key}: fewer than two usable trials")
                neural_all, neural_info = _load_neural(matfile, probes)
                neural_present = np.any(neural_all != 0, axis=(1, 2))
                selected = candidate_trials[neural_present[candidate_trials]]
                output_all, _, thresholds = _load_behavior(matfile, motion_path, selected)
                neural_selected = neural_all[selected]
        else:
            obj = loadmat(data_path, squeeze_me=True, struct_as_record=False,
                          variable_names=["obj"])["obj"]
            early = np.asarray(obj.bp.early).reshape(-1).astype(bool)
            stimulated = np.asarray(obj.bp.stim.enable).reshape(-1).astype(bool)
            have_ephys = _fit_trial_mask(obj.trials.bp.haveEphys, early.size)
            candidate_trials = np.flatnonzero(~early & ~stimulated & have_ephys)
            if candidate_trials.size < 2:
                raise ValueError(f"{key}: fewer than two usable trials")
            neural_all, neural_info = _load_neural_v5(obj, probes)
            neural_present = np.any(neural_all != 0, axis=(1, 2))
            selected = candidate_trials[neural_present[candidate_trials]]
            output_all, _, thresholds = _load_behavior_v5(obj, motion_path, selected)
            neural_selected = neural_all[selected]
            del obj

        neural_trials = [np.ascontiguousarray(neural_selected[t])
                         for t in range(selected.size)]
        # Time from the go cue is a continuous, time-varying decoder input.
        input_trials = [TIME[None, :].copy() for _ in range(selected.size)]
        output_trials = [np.ascontiguousarray(output_all[t])
                         for t in range(selected.size)]
        neural_sessions.append(neural_trials)
        input_sessions.append(input_trials)
        output_sessions.append(output_trials)
        session_subjects.append(subject)
        region_indices.append(np.zeros(neural_selected.shape[1], dtype=np.int64))
        session_info.append({
            "session_id": key,
            "subject": subject,
            "date": date,
            "task_variant": task_variant,
            "source_file": data_path.name,
            "source_directory": data_path.parent.name,
            "alm_probes": list(probes),
            "n_source_trials": int(early.size),
            "n_trials_retained": int(selected.size),
            "n_early_lick_excluded": int(early.sum()),
            "n_photostim_excluded": int((stimulated & ~early).sum()),
            "n_no_ephys_excluded": int((~have_ephys & ~early & ~stimulated).sum()),
            "n_all_zero_neural_excluded": int(candidate_trials.size - selected.size),
            **neural_info,
            **thresholds,
        })
        del neural_all, neural_selected, output_all

    subjects = list(dict.fromkeys(session_subjects))
    subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
    data = {
        "neural": neural_sessions,
        "input": input_sessions,
        "output": output_sessions,
        "subjects": subjects,
        "subject_idx": np.asarray([subject_lookup[x] for x in session_subjects], dtype=np.int64),
        "brain_regions": ["ALM"],
        "brain_region_idx": region_indices,
        "input_names": ["time_from_go_cue_s"],
        "output_names": [
            "lick_direction", "behavioral_context", "outcome",
            "tongue_velocity", "paw_velocity", "motion_energy",
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
                "Fixed- and randomized-delay response (DR) and water-cued (WC) licking "
                "tasks; decode actual lick direction, context, outcome, and discretized "
                "movement from ALM activity."
            ),
            "time_bin_size": DT * 1000.0,
            "temporal_alignment_event": "auditory go cue onset (water delivery onset in WC trials)",
            "off_start": TMIN,
            "off_end": TMAX,
            "neural_measurement": "single-trial firing rate (spikes/s), causal Gaussian smoothed",
            "neural_smoothing_bins": SMOOTH_BINS,
            "neuron_inclusion": (
                "Repository-curated units excluding garbage/noisy/real? labels, with mean "
                "go-cue-window firing rate > 1 Hz; published ALM probe selections."
            ),
            "trial_inclusion": (
                "All hit, miss, and ignore trials except early-lick and photostimulation trials. "
                "Trials outside electrophysiology coverage are excluded; ignore trials are "
                "retained to satisfy the requested outcome/none classes. Trials with no "
                "activity in any retained unit across the full window are also excluded."
            ),
            "movement_discretization": (
                "Per-session 50th percentile over visible retained samples; class 1 is >= median."
            ),
            "paw_velocity_definition": (
                "Mean Euclidean image-plane speed of the two bottom-view paw keypoints when visible."
            ),
            "session_info": session_info,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(sessions)} sessions to {output_path}", flush=True)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None,
                        help="Development aid: convert only the first N sessions")
    args = parser.parse_args()
    convert(args.data_root, args.output, args.limit)


if __name__ == "__main__":
    main()
