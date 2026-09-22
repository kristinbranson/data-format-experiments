#!/usr/bin/env python3
"""Convert Hasnain/Birnbaum two-context ALM recordings for decoding.

Usage:
    python -u /app/convert_data.py OUTPUT.pkl [--full | --sample]
                                         [--show-processing]

The implementation follows the paper's released MATLAB pipeline: author-curated
sessions/probes, go-cue alignment, 5-ms firing-rate bins, a 15-bin causal
Gaussian, >1-Hz unit filtering, and bitcode-derived video synchronization.
"""

from __future__ import annotations

import argparse
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter
from scipy.signal.windows import gaussian


DATA_ROOT = Path("/app/data/Ephys_Behavior")
TMIN = -2.5
TMAX = 2.5
DT = 0.005
TIME = (np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2).astype(np.float32)
N_TIME = len(TIME)
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
LOW_FR_HZ = 1.0


@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    probes: tuple[int, ...]

    @property
    def session_id(self) -> str:
        return f"{self.animal}_{self.date}"

    @property
    def data_path(self) -> Path:
        return DATA_ROOT / f"data_structure_{self.session_id}.mat"

    @property
    def motion_path(self) -> Path:
        return DATA_ROOT / f"motionEnergy_{self.session_id}.mat"


# Exact Figure 8 loader order. JEB19's loader lists dates in reverse order.
SESSION_SPECS = (
    SessionSpec("JEB6", "2021-04-18", (2,)),
    SessionSpec("JEB7", "2021-04-29", (1,)),
    SessionSpec("JEB7", "2021-04-30", (1,)),
    SessionSpec("EKH1", "2021-08-07", (2,)),
    SessionSpec("EKH3", "2021-08-11", (2,)),
    SessionSpec("JGR2", "2021-11-16", (1,)),
    SessionSpec("JGR2", "2021-11-17", (1,)),
    SessionSpec("JGR3", "2021-11-18", (1,)),
    SessionSpec("JEB19", "2023-04-21", (1,)),
    SessionSpec("JEB19", "2023-04-20", (1,)),
    SessionSpec("JEB19", "2023-04-19", (1,)),
    SessionSpec("JEB19", "2023-04-18", (1,)),
)


def matlab_char(array: np.ndarray) -> str:
    """Decode a MATLAB UTF-16 char dataset."""
    return "".join(chr(int(x)) for x in np.asarray(array).ravel(order="F") if x)


def deref_array(handle: h5py.File, ref: h5py.Reference) -> np.ndarray:
    return np.asarray(handle[ref])


def deref_vector(handle: h5py.File, ref: h5py.Reference, dtype=None) -> np.ndarray:
    out = deref_array(handle, ref).ravel(order="F")
    return out.astype(dtype, copy=False) if dtype is not None else out


def deref_string(handle: h5py.File, ref: h5py.Reference) -> str:
    obj = handle[ref]
    if not isinstance(obj, h5py.Dataset) or obj.size == 0:
        return ""
    return matlab_char(obj[()])


def matlab_cell_refs(dataset: h5py.Dataset) -> np.ndarray:
    return dataset[()].ravel(order="F")


def mode_value(values: np.ndarray) -> float:
    values = np.asarray(values).ravel()
    values = values[np.isfinite(values)]
    unique, counts = np.unique(values, return_counts=True)
    return float(unique[np.argmax(counts)])


def causal_gaussian_kernel(n: int = 15) -> np.ndarray:
    """Match MATLAB gausswin(n, 2.5), then the released causalization."""
    kernel = gaussian(n, std=(n - 1) / (2 * 2.5), sym=True).astype(np.float32)
    kernel[: n // 2] = 0
    kernel /= kernel.sum()
    return kernel


KERNEL = causal_gaussian_kernel(15)


def reference_smooth(values: np.ndarray, n: int = 15) -> np.ndarray:
    """Match mySmooth(..., 15, 'reflect') on the last (time) dimension.

    Despite its comment, the MATLAB code prepends the first N samples in their
    original order. With its causal kernel, lfilter gives the same retained
    samples as conv(..., 'same') followed by trimming the prefix.
    """
    values = np.asarray(values, dtype=np.float32)
    padded = np.concatenate((values[..., :n], values), axis=-1)
    # Only coefficients 7:15 survive. For MATLAB conv(...,'same'), these map
    # to lags 0:7 in the order below.
    causal_coefficients = KERNEL[n // 2 :]
    filtered = lfilter(causal_coefficients, [1.0], padded, axis=-1)
    return filtered[..., n:].astype(np.float32, copy=False)


def direct_field(handle: h5py.File, path: str, dtype=None) -> np.ndarray:
    array = np.asarray(handle[path]).ravel(order="F")
    return array.astype(dtype, copy=False) if dtype is not None else array


def load_behavior(handle: h5py.File) -> dict[str, np.ndarray]:
    fields = ("R", "L", "hit", "miss", "no", "early", "autowater")
    bp = {field: direct_field(handle, f"obj/bp/{field}", bool) for field in fields}
    bp["stim"] = direct_field(handle, "obj/bp/stim/enable", bool)
    bp["goCue"] = direct_field(handle, "obj/bp/ev/goCue", np.float64)
    bp["bitStart"] = direct_field(handle, "obj/bp/ev/bitStart", np.float64)
    ntrials = int(np.asarray(handle["obj/bp/Ntrials"]).squeeze())
    if any(len(value) != ntrials for value in bp.values()):
        raise ValueError("Behavior fields do not all match Ntrials")
    outcome_sum = bp["hit"].astype(int) + bp["miss"].astype(int) + bp["no"].astype(int)
    if not np.all(outcome_sum == 1):
        raise ValueError("hit/miss/no are not mutually exhaustive")
    side_sum = bp["R"].astype(int) + bp["L"].astype(int)
    if not np.all(side_sum == 1):
        raise ValueError("R/L are not mutually exhaustive")
    if not np.all(np.isfinite(bp["goCue"])):
        raise ValueError("Non-finite go cue in source behavior")
    return bp


def video_offset_seconds(handle: h5py.File, bp: dict[str, np.ndarray]) -> float:
    neural_bit_start = direct_field(handle, "obj/sglx/bitcode/bitstart", np.float64)
    fs = float(np.asarray(handle["obj/sglx/fs"]).squeeze())
    return mode_value(neural_bit_start) / fs - mode_value(bp["bitStart"])


def selected_clusters(handle: h5py.File, probes: tuple[int, ...]) -> list[dict]:
    probe_refs = matlab_cell_refs(handle["obj/clu"])
    clusters = []
    for probe in probes:
        if probe < 1 or probe > len(probe_refs):
            raise IndexError(f"Requested probe {probe}, only {len(probe_refs)} present")
        group = handle[probe_refs[probe - 1]]
        if not isinstance(group, h5py.Group):
            raise ValueError(f"Probe {probe} has no cluster struct")
        quality_refs = matlab_cell_refs(group["quality"])
        trial_refs = matlab_cell_refs(group["trial"])
        trialtm_refs = matlab_cell_refs(group["trialtm"])
        for cluster_index, quality_ref in enumerate(quality_refs):
            quality = deref_string(handle, quality_ref).strip().lower()
            if quality in QUALITY_EXCLUDE:
                continue
            trials = deref_vector(handle, trial_refs[cluster_index], np.int64) - 1
            trial_times = deref_vector(handle, trialtm_refs[cluster_index], np.float64)
            if len(trials) != len(trial_times):
                raise ValueError("Cluster trial and trial-time vectors differ")
            clusters.append({
                "probe": probe,
                "cluster_index": cluster_index,
                "quality": quality,
                "trials": trials,
                "trial_times": trial_times,
            })
    return clusters


def low_fr_filter(clusters: list[dict], bp: dict[str, np.ndarray]) -> tuple[list[dict], np.ndarray]:
    """Recreate the Figure 8 context-workflow low-FR calculation."""
    hit, miss = bp["hit"], bp["miss"]
    stim, aw, early = bp["stim"], bp["autowater"], bp["early"]
    ntrials = len(hit)
    conditions = (
        np.ones(ntrials, dtype=bool),
        hit & ~stim & ~aw,
        hit & ~stim & aw,
        miss & ~stim & ~aw,
        miss & ~stim & aw,
        hit & ~stim & ~aw & ~early,
        hit & ~stim & aw & ~early,
    )
    edges = np.arange(-3.0, 2.5 + 0.005, 0.01, dtype=np.float64)
    mean_rates = np.empty(len(clusters), dtype=np.float64)
    for unit, cluster in enumerate(clusters):
        spike_trials = cluster["trials"]
        aligned = cluster["trial_times"] - bp["goCue"][spike_trials]
        condition_psths = []
        for condition in conditions:
            if not np.any(condition):
                condition_psths.append(np.full(len(edges) - 1, np.nan, dtype=np.float32))
                continue
            use = condition[spike_trials]
            counts = np.histogram(aligned[use], bins=edges)[0].astype(np.float32)
            rate = counts / (float(np.sum(condition)) * 0.01)
            condition_psths.append(reference_smooth(rate))
        mean_rates[unit] = np.nanmean(np.stack(condition_psths))
    keep = mean_rates > LOW_FR_HZ
    return [cluster for cluster, use in zip(clusters, keep) if use], mean_rates


def build_neural(clusters: list[dict], bp: dict[str, np.ndarray], raw_trials: np.ndarray) -> np.ndarray:
    n_units, n_trials = len(clusters), len(raw_trials)
    counts = np.zeros((n_units, n_trials, N_TIME), dtype=np.float32)
    raw_to_kept = np.full(len(bp["hit"]), -1, dtype=np.int64)
    raw_to_kept[raw_trials] = np.arange(n_trials)
    for unit, cluster in enumerate(clusters):
        spike_trials = cluster["trials"]
        mapped_trials = raw_to_kept[spike_trials]
        use = mapped_trials >= 0
        if not np.any(use):
            continue
        spike_trials = spike_trials[use]
        mapped_trials = mapped_trials[use]
        aligned = cluster["trial_times"][use] - bp["goCue"][spike_trials]
        bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
        in_window = (bins >= 0) & (bins < N_TIME)
        np.add.at(counts[unit], (mapped_trials[in_window], bins[in_window]), 1)
    return reference_smooth(counts / DT)


def decode_cellstr(handle: h5py.File, dataset: h5py.Dataset) -> list[str]:
    return [deref_string(handle, ref) for ref in matlab_cell_refs(dataset)]


def trajectory_group(handle: h5py.File, view: int) -> h5py.Group:
    refs = matlab_cell_refs(handle["obj/traj"])
    group = handle[refs[view - 1]]
    if not isinstance(group, h5py.Group):
        raise ValueError(f"Camera view {view} missing")
    return group


def feature_names_for_trial(handle: h5py.File, group: h5py.Group, trial: int) -> list[str]:
    feature_cell = handle[matlab_cell_refs(group["featNames"])[trial]]
    return decode_cellstr(handle, feature_cell)


def trajectory_for_trial(handle: h5py.File, group: h5py.Group, trial: int) -> tuple[np.ndarray, np.ndarray]:
    ts_ref = matlab_cell_refs(group["ts"])[trial]
    ft_ref = matlab_cell_refs(group["frameTimes"])[trial]
    stored = deref_array(handle, ts_ref)
    # HDF5 dimension order is reversed relative to MATLAB: feature, coord, frame.
    if stored.ndim != 3:
        raise ValueError(f"Unexpected DLC trajectory shape {stored.shape}")
    trajectory = np.transpose(stored, (2, 1, 0)).astype(np.float64, copy=False)
    frame_times = deref_vector(handle, ft_ref, np.float64)
    if len(trajectory) != len(frame_times):
        raise ValueError("DLC frames and frameTimes lengths differ")
    return trajectory, frame_times


def interpolate_with_nans(times: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    finite_time = np.isfinite(times)
    if np.sum(finite_time) < 2:
        return np.full(len(target), np.nan, dtype=np.float64)
    x = times[finite_time]
    y = values[finite_time]
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique = np.r_[True, np.diff(x) > 0]
    x, y = x[unique], y[unique]
    if len(x) < 2:
        return np.full(len(target), np.nan, dtype=np.float64)
    return np.interp(target, x, y, left=np.nan, right=np.nan)


def fill_nearest(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return values.copy()
    indices = np.arange(len(values))
    # np.interp linearly fills; nearest requires choosing the closer bracketing index.
    valid_indices = indices[finite]
    insertion = np.searchsorted(valid_indices, indices)
    left_pos = np.clip(insertion - 1, 0, len(valid_indices) - 1)
    right_pos = np.clip(insertion, 0, len(valid_indices) - 1)
    left = valid_indices[left_pos]
    right = valid_indices[right_pos]
    nearest = np.where(indices - left <= right - indices, left, right)
    out = values.copy()
    out[~finite] = values[nearest[~finite]]
    return out


def feature_speed(
    handle: h5py.File,
    group: h5py.Group,
    trial: int,
    feature: str,
    aligned_time: np.ndarray,
    is_tongue: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trajectory, frame_times = trajectory_for_trial(handle, group, trial)
    names = feature_names_for_trial(handle, group, trial)
    if feature not in names:
        nan = np.full(N_TIME, np.nan, dtype=np.float64)
        return nan, np.zeros(N_TIME, dtype=bool), nan, nan
    feature_index = names.index(feature)
    x = interpolate_with_nans(aligned_time, trajectory[:, 0, feature_index], TIME)
    y = interpolate_with_nans(aligned_time, trajectory[:, 1, feature_index], TIME)
    visible = np.isfinite(x) & np.isfinite(y)
    if is_tongue:
        x_for_velocity, y_for_velocity = x, y
    else:
        x_for_velocity, y_for_velocity = fill_nearest(x), fill_nearest(y)
    if np.any(np.isfinite(x_for_velocity)) and np.any(np.isfinite(y_for_velocity)):
        x_velocity = np.gradient(x_for_velocity)
        y_velocity = np.gradient(y_for_velocity)
        if not is_tongue:
            stacked = np.column_stack((x_for_velocity, y_for_velocity))
            differences = np.diff(stacked, axis=0)
            baseline_derivative = np.array([
                np.median(column[np.isfinite(column)]) if np.any(np.isfinite(column)) else 0.0
                for column in differences.T
            ])
            # Match the released function, including its use of x baseline for y.
            x_velocity = x_velocity - baseline_derivative[0]
            y_velocity = y_velocity - baseline_derivative[0]
        speed = np.hypot(x_velocity, y_velocity)
    else:
        speed = np.full(N_TIME, np.nan, dtype=np.float64)
    speed[~visible] = np.nan
    # A derivative adjacent to a visibility gap may itself be undefined even
    # when the position at the center sample is finite.
    visible = visible & np.isfinite(speed)
    return speed, visible, x, y


def load_motion_energy(path: Path) -> np.ndarray:
    motion = loadmat(path, simplify_cells=True)["me"]
    data = motion["data"]
    if isinstance(data, dict):
        data = data["data"]
    return np.atleast_1d(data)


def align_kinematics_and_motion(
    handle: h5py.File,
    bp: dict[str, np.ndarray],
    raw_trials: np.ndarray,
    motion_path: Path,
    offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    side = trajectory_group(handle, 1)
    bottom = trajectory_group(handle, 2)
    motion_raw = load_motion_energy(motion_path)
    if len(motion_raw) != len(bp["hit"]):
        raise ValueError("Motion-energy trial count differs from behavior")

    n_trials = len(raw_trials)
    tongue = np.full((n_trials, N_TIME), np.nan, dtype=np.float64)
    paw = np.full((n_trials, N_TIME), np.nan, dtype=np.float64)
    motion = np.full((n_trials, N_TIME), np.nan, dtype=np.float64)
    tongue_visible = np.zeros((n_trials, N_TIME), dtype=bool)
    paw_visible = np.zeros((n_trials, N_TIME), dtype=bool)
    sample = {}

    for kept_trial, raw_trial in enumerate(raw_trials):
        side_trajectory, side_frames = trajectory_for_trial(handle, side, int(raw_trial))
        del side_trajectory  # Loaded again feature-wise; retained here for timestamps.
        aligned_side_time = side_frames - offset - bp["goCue"][raw_trial]
        _, bottom_frames = trajectory_for_trial(handle, bottom, int(raw_trial))
        aligned_bottom_time = bottom_frames - offset - bp["goCue"][raw_trial]

        tongue[kept_trial], tongue_visible[kept_trial], tongue_x, tongue_y = feature_speed(
            handle, side, int(raw_trial), "tongue", aligned_side_time, True
        )
        paw[kept_trial], paw_visible[kept_trial], paw_x, paw_y = feature_speed(
            handle, bottom, int(raw_trial), "top_paw", aligned_bottom_time, False
        )

        trial_motion = np.asarray(motion_raw[raw_trial], dtype=np.float64).squeeze()
        if trial_motion.ndim != 1:
            trial_motion = trial_motion.ravel()
        if trial_motion.size:
            if len(trial_motion) == len(side_frames) and np.sum(np.isfinite(side_frames)) >= 2:
                motion_times = aligned_side_time
            else:
                # Same catch-path as loadMotionEnergy.m when frameTimes are
                # absent/all-NaN: nominal 400 Hz and its explicit 0.5-s shift.
                motion_times = np.arange(1, len(trial_motion) + 1, dtype=np.float64) / 400
                motion_times = motion_times - 0.5 - bp["goCue"][raw_trial]
            interpolated = interpolate_with_nans(motion_times, trial_motion, TIME)
            motion[kept_trial] = fill_nearest(interpolated)

        if kept_trial == 0:
            sample = {
                "raw_trial": int(raw_trial),
                "tongue_x": tongue_x,
                "tongue_y": tongue_y,
                "paw_x": paw_x,
                "paw_y": paw_y,
            }

    tongue_threshold = float(np.nanmedian(tongue[tongue_visible]))
    paw_threshold = float(np.nanmedian(paw[paw_visible]))
    valid_motion = np.isfinite(motion)
    motion_threshold = float(np.nanmedian(motion[valid_motion]))
    thresholds = np.array([tongue_threshold, paw_threshold, motion_threshold], dtype=np.float64)
    if not np.all(np.isfinite(thresholds)):
        raise ValueError(f"Could not compute finite session thresholds: {thresholds}")
    sample.update({
        "tongue_speed": tongue[0].copy(),
        "paw_speed": paw[0].copy(),
        "motion": motion[0].copy(),
    })
    return tongue, paw, motion, np.stack((tongue_visible, paw_visible, valid_motion)), {
        "thresholds": thresholds,
        "sample": sample,
    }


def static_trial_outputs(bp: dict[str, np.ndarray], raw_trials: np.ndarray) -> np.ndarray:
    static = np.empty((len(raw_trials), 3), dtype=np.int8)
    for kept, raw in enumerate(raw_trials):
        if bp["no"][raw]:
            lick = 2
        elif bp["hit"][raw]:
            lick = 1 if bp["R"][raw] else 0
        elif bp["miss"][raw]:
            lick = 0 if bp["R"][raw] else 1
        else:
            raise ValueError("Unrecognized trial outcome")
        context = 0 if bp["autowater"][raw] else 1
        outcome = 0 if bp["miss"][raw] else (1 if bp["hit"][raw] else 2)
        static[kept] = (lick, context, outcome)
    return static


def discretize_outputs(
    static: np.ndarray,
    tongue: np.ndarray,
    paw: np.ndarray,
    motion: np.ndarray,
    visibility: np.ndarray,
    thresholds: np.ndarray,
) -> list[np.ndarray]:
    outputs = []
    continuous = (tongue, paw, motion)
    for trial in range(len(static)):
        out = np.empty((6, N_TIME), dtype=np.int8)
        out[:3] = static[trial, :, None]
        for output_index, values in enumerate(continuous, start=3):
            valid = visibility[output_index - 3, trial]
            classes = np.full(N_TIME, 2, dtype=np.int8)
            classes[valid] = (values[trial, valid] >= thresholds[output_index - 3]).astype(np.int8)
            out[output_index] = classes
        outputs.append(out)
    return outputs


def convert_session(spec: SessionSpec) -> tuple[dict, dict]:
    start = time.perf_counter()
    with h5py.File(spec.data_path, "r") as handle:
        bp = load_behavior(handle)
        raw_trial_count = len(bp["hit"])
        keep_trials = ~bp["early"] & ~bp["stim"]
        raw_trials = np.flatnonzero(keep_trials)
        if len(raw_trials) < 2:
            raise ValueError(f"{spec.session_id}: fewer than two retained trials")

        quality_clusters = selected_clusters(handle, spec.probes)
        clusters, mean_rates = low_fr_filter(quality_clusters, bp)
        if len(clusters) < 10:
            raise ValueError(f"{spec.session_id}: only {len(clusters)} retained units")
        neural = build_neural(clusters, bp, raw_trials)
        if neural.shape != (len(clusters), len(raw_trials), N_TIME):
            raise AssertionError(f"Unexpected neural shape {neural.shape}")

        offset = video_offset_seconds(handle, bp)
        tongue, paw, motion, visibility, kin_info = align_kinematics_and_motion(
            handle, bp, raw_trials, spec.motion_path, offset
        )
        static = static_trial_outputs(bp, raw_trials)
        outputs = discretize_outputs(
            static, tongue, paw, motion, visibility, kin_info["thresholds"]
        )

        neural_trials = [neural[:, trial, :].copy() for trial in range(len(raw_trials))]
        input_trials = [TIME[None, :].copy() for _ in raw_trials]
        unit_info = [(c["probe"], c["cluster_index"], c["quality"]) for c in clusters]
        session_info = {
            "session_id": spec.session_id,
            "source_file": str(spec.data_path),
            "motion_energy_file": str(spec.motion_path),
            "probes": list(spec.probes),
            "raw_trial_count": raw_trial_count,
            "retained_trial_count": int(len(raw_trials)),
            "retained_raw_trial_indices_0based": raw_trials.tolist(),
            "quality_eligible_unit_count": len(quality_clusters),
            "retained_unit_count": len(clusters),
            "retained_units_probe_cluster0_quality": unit_info,
            "video_offset_seconds": offset,
            "tongue_velocity_median": float(kin_info["thresholds"][0]),
            "paw_velocity_median": float(kin_info["thresholds"][1]),
            "motion_energy_median": float(kin_info["thresholds"][2]),
            "outcome_counts_hit_miss_ignore": [
                int(np.sum(static[:, 2] == 1)),
                int(np.sum(static[:, 2] == 0)),
                int(np.sum(static[:, 2] == 2)),
            ],
            "context_counts_WC_DR": [int(np.sum(static[:, 1] == 0)), int(np.sum(static[:, 1] == 1))],
        }

        diagnostics = {
            "session_id": spec.session_id,
            "mean_rates_all_quality_units": mean_rates,
            "neural_sample": neural[: min(12, len(clusters)), 0].copy(),
            "kinematic_sample": kin_info["sample"],
            "thresholds": kin_info["thresholds"],
            "static": static,
            "output_sample": outputs[0],
            "visibility_fractions": visibility.mean(axis=(1, 2)),
        }

    elapsed = time.perf_counter() - start
    print(
        f"[{spec.session_id}] {raw_trial_count}->{len(raw_trials)} trials, "
        f"{len(quality_clusters)}->{len(clusters)} units, {elapsed:.2f} s",
        flush=True,
    )
    return {
        "neural": neural_trials,
        "input": input_trials,
        "output": outputs,
        "brain_region_idx": np.zeros(len(clusters), dtype=np.int64),
        "session_info": session_info,
        "elapsed_seconds": elapsed,
    }, diagnostics


def plot_processing(diagnostics: dict) -> Path:
    sid = diagnostics["session_id"]
    sample = diagnostics["kinematic_sample"]
    thresholds = diagnostics["thresholds"]
    output = diagnostics["output_sample"]
    figure, axes = plt.subplots(4, 2, figsize=(16, 16), constrained_layout=True)

    axes[0, 0].plot(TIME, diagnostics["neural_sample"].T, alpha=0.7)
    axes[0, 0].axvline(0, color="k", linestyle="--")
    axes[0, 0].set(title="Binned + causal-smoothed firing rates", ylabel="spikes/s")

    rates = diagnostics["mean_rates_all_quality_units"]
    axes[0, 1].hist(rates[np.isfinite(rates)], bins=30)
    axes[0, 1].axvline(LOW_FR_HZ, color="r", linestyle="--", label=">1 Hz retained")
    axes[0, 1].set(title="Reference mean-FR unit curation", xlabel="mean firing rate (Hz)")
    axes[0, 1].legend()

    axes[1, 0].plot(TIME, sample["tongue_x"], label="tongue x")
    axes[1, 0].plot(TIME, sample["tongue_y"], label="tongue y")
    axes[1, 0].plot(TIME, sample["paw_x"], label="paw x", alpha=0.7)
    axes[1, 0].plot(TIME, sample["paw_y"], label="paw y", alpha=0.7)
    axes[1, 0].axvline(0, color="k", linestyle="--")
    axes[1, 0].set(title="Synchronized/interpolated DLC positions", ylabel="pixels")
    axes[1, 0].legend(ncol=2)

    axes[1, 1].plot(TIME, sample["tongue_speed"], label="tongue speed")
    axes[1, 1].axhline(thresholds[0], color="C0", linestyle="--")
    axes[1, 1].plot(TIME, sample["paw_speed"], label="paw speed")
    axes[1, 1].axhline(thresholds[1], color="C1", linestyle="--")
    axes[1, 1].set(title="Derivative speed and session medians", ylabel="pixels/bin")
    axes[1, 1].legend()

    axes[2, 0].plot(TIME, sample["motion"], color="C2")
    axes[2, 0].axhline(thresholds[2], color="k", linestyle="--")
    axes[2, 0].axvline(0, color="k", linestyle=":")
    axes[2, 0].set(title="Aligned motion energy + median", ylabel="a.u.")

    for index, name in enumerate(("tongue", "paw", "motion"), start=3):
        axes[2, 1].step(TIME, output[index] + (index - 3) * 3, where="mid", label=name)
    axes[2, 1].set(title="Discretized time-varying outputs", yticks=[])
    axes[2, 1].legend()

    labels = ("lick", "context", "outcome")
    static = diagnostics["static"]
    for index, name in enumerate(labels):
        values, counts = np.unique(static[:, index], return_counts=True)
        axes[3, 0].bar(values + index * 0.22, counts, width=0.2, label=name)
    axes[3, 0].set(title="Retained per-trial label counts", xlabel="category")
    axes[3, 0].legend()

    axes[3, 1].axis("off")
    axes[3, 1].text(
        0,
        1,
        f"session: {sid}\nraw trial: {sample['raw_trial']}\n"
        f"video-aligned bins: {N_TIME}\n"
        f"visible fractions (tongue, paw, motion):\n"
        f"{np.array2string(diagnostics['visibility_fractions'], precision=4)}\n"
        f"thresholds:\n{np.array2string(thresholds, precision=5)}",
        va="top",
        family="monospace",
        fontsize=12,
    )
    for axis in axes.flat:
        if axis.axison and axis is not axes[3, 1]:
            axis.set_xlim(TIME[0], TIME[-1]) if axis in (axes[0, 0], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]) else None
    figure.suptitle(f"Conversion processing: {sid}", fontsize=16)
    path = Path(f"/app/processing_{sid}.png")
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def validate_converted(data: dict) -> None:
    if len(data["neural"]) != len(data["input"]) or len(data["neural"]) != len(data["output"]):
        raise AssertionError("Session counts differ")
    for session, (neural, inputs, outputs, regions) in enumerate(zip(
        data["neural"], data["input"], data["output"], data["brain_region_idx"]
    )):
        if not (len(neural) == len(inputs) == len(outputs)) or len(neural) < 2:
            raise AssertionError(f"Session {session}: invalid trial counts")
        n_units = neural[0].shape[0]
        if len(regions) != n_units:
            raise AssertionError(f"Session {session}: region/unit mismatch")
        for trial, (n, x, y) in enumerate(zip(neural, inputs, outputs)):
            if n.shape != (n_units, N_TIME) or x.shape != (1, N_TIME) or y.shape != (6, N_TIME):
                raise AssertionError(f"Session {session} trial {trial}: shape mismatch")
            if n.dtype != np.float32 or x.dtype != np.float32 or y.dtype != np.int8:
                raise AssertionError(f"Session {session} trial {trial}: dtype mismatch")
            if not (np.all(np.isfinite(n)) and np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
                raise AssertionError(f"Session {session} trial {trial}: non-finite values")
            if not np.allclose(x[0], TIME):
                raise AssertionError(f"Session {session} trial {trial}: time input mismatch")
            for output_index, nclasses in enumerate((3, 2, 3, 3, 3, 3)):
                if np.min(y[output_index]) < 0 or np.max(y[output_index]) >= nclasses:
                    raise AssertionError(f"Session {session} trial {trial}: invalid output class")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default)")
    mode.add_argument("--sample", action="store_true", help="Process the first two sessions")
    parser.add_argument("--show-processing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = SESSION_SPECS[:2] if args.sample else SESSION_SPECS
    print(f"Converting {len(specs)} {'sample' if args.sample else 'full'} sessions", flush=True)
    overall_start = time.perf_counter()
    converted = []
    diagnostics = []
    for spec in specs:
        session, diag = convert_session(spec)
        converted.append(session)
        diagnostics.append(diag)

    subjects = list(dict.fromkeys(spec.animal for spec in specs))
    subject_idx = np.array([subjects.index(spec.animal) for spec in specs], dtype=np.int64)
    session_info = [session["session_info"] for session in converted]
    data = {
        "neural": [session["neural"] for session in converted],
        "input": [session["input"] for session in converted],
        "output": [session["output"] for session in converted],
        "subjects": subjects,
        "subject_idx": subject_idx,
        "brain_regions": ["ALM"],
        "brain_region_idx": [session["brain_region_idx"] for session in converted],
        "input_names": ["time_from_go_cue_s"],
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
            ["below_50th_percentile", "at_or_above_50th_percentile", "not_visible"],
            ["below_50th_percentile", "at_or_above_50th_percentile", "not_visible"],
            ["below_50th_percentile", "at_or_above_50th_percentile", "no_video"],
        ],
        "metadata": {
            "task_description": (
                "Head-fixed mice alternate between delayed-response (DR) and water-cued (WC) "
                "directional licking contexts; decode actual lick, context, outcome, and movement."
            ),
            "time_bin_size": 5.0,
            "temporal_alignment_event": "go cue onset (DR) or matched water-drop response onset (WC)",
            "off_start": TMIN,
            "off_end": TMAX,
            "time_bin_centers_seconds": TIME.copy(),
            "neural_representation": "spikes/s in 5-ms bins, 15-bin causal Gaussian smoothed",
            "unit_quality_exclusions": sorted(QUALITY_EXCLUDE),
            "low_firing_rate_rule": "reference context-workflow mean PSTH strictly > 1 Hz",
            "trial_filter": "exclude early-lick or stimulation-enabled trials; retain hit/miss/no",
            "velocity_definition": "Euclidean magnitude of x/y first derivatives in pixels per 5-ms bin",
            "discretization": "per-session median over retained visible/valid aligned samples",
            "session_info": session_info,
            "source_paper": "Separating cognitive and motor processes in the behaving mouse",
            "conversion_script": "/app/convert_data.py",
        },
    }
    validate_converted(data)

    if args.show_processing:
        for diag in diagnostics[:2]:
            path = plot_processing(diag)
            print(f"Saved processing plot: {path}", flush=True)

    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    with args.outpicklefile.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed = time.perf_counter() - overall_start
    total_trials = sum(len(session) for session in data["neural"])
    total_units = sum(len(index) for index in data["brain_region_idx"])
    size_mb = args.outpicklefile.stat().st_size / (1024 ** 2)
    print(
        f"Saved {args.outpicklefile} ({size_mb:.1f} MiB): {len(specs)} sessions, "
        f"{total_trials} trials, {total_units} session-units in {elapsed:.2f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
