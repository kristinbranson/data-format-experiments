#!/usr/bin/env python3
"""Convert Hasnain, Birnbaum et al. two-context ephys sessions for decoder training."""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mat73
import numpy as np
import scipy.io as sio


DATA_DIR = Path("/app/data/Ephys_Behavior")
FULL_OUT = Path("/app/converted_data.pkl")
SAMPLE_OUT = Path("/app/sample_data.pkl")


ALIGN_EVENT = "goCue"
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
SMOOTH = 15
LOW_FR = 1.0
ADVANCE_MOVEMENT = 0.0

SIDE_FEATURES = ["tongue", "left_tongue", "right_tongue", "jaw", "trident", "nose"]
BOTTOM_FEATURES = [
    "top_tongue",
    "topleft_tongue",
    "bottom_tongue",
    "bottomleft_tongue",
    "top_paw",
    "bottom_paw",
    "jaw",
    "top_nostril",
    "bottom_nostril",
]
TONGUE_FEATURES = [
    (0, "tongue"),
    (0, "left_tongue"),
    (0, "right_tongue"),
    (1, "top_tongue"),
    (1, "topleft_tongue"),
    (1, "bottom_tongue"),
    (1, "bottomleft_tongue"),
]
PAW_FEATURES = [(1, "top_paw"), (1, "bottom_paw")]


@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    probe: int  # 1-based probe index, as in the MATLAB metadata loaders

    @property
    def stem(self) -> str:
        return f"{self.animal}_{self.date}"

    @property
    def data_path(self) -> Path:
        return DATA_DIR / f"data_structure_{self.stem}.mat"

    @property
    def motion_energy_path(self) -> Path:
        return DATA_DIR / f"motionEnergy_{self.stem}.mat"


# Session roster from code/Scripts/Figure 8/Figure8a_thru_c.m and the loader files
SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 2),
    SessionSpec("JEB7", "2021-04-29", 1),
    SessionSpec("JEB7", "2021-04-30", 1),
    SessionSpec("EKH1", "2021-08-07", 2),
    SessionSpec("EKH3", "2021-08-11", 2),
    SessionSpec("JGR2", "2021-11-16", 1),
    SessionSpec("JGR2", "2021-11-17", 1),
    SessionSpec("JGR3", "2021-11-18", 1),
    SessionSpec("JEB19", "2023-04-21", 1),
    SessionSpec("JEB19", "2023-04-20", 1),
    SessionSpec("JEB19", "2023-04-19", 1),
    SessionSpec("JEB19", "2023-04-18", 1),
]


def build_time_axis() -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
    time = edges[:-1] + DT / 2
    return edges, time


EDGES, TIME = build_time_axis()
NT = TIME.size


def flatten_string(value) -> str:
    while isinstance(value, list):
        if not value:
            return ""
        value = value[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return flatten_string(value.item())
        return "".join(str(x) for x in value.ravel())
    return str(value)


def as_array(value, dtype=None) -> np.ndarray:
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def mode_scalar(values: Iterable[float], round_decimals: int = 6) -> float:
    arr = np.asarray(list(values), dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    rounded = np.round(arr, round_decimals)
    uniq, counts = np.unique(rounded, return_counts=True)
    return float(uniq[np.argmax(counts)])


def fill_nearest_1d(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).copy()
    if out.size == 0:
        return out
    mask = np.isfinite(out)
    if mask.all():
        return out
    if not mask.any():
        out[:] = fill_value
        return out
    idx = np.arange(out.size)
    valid_idx = idx[mask]
    pos = np.searchsorted(valid_idx, idx)
    left_pos = np.clip(pos - 1, 0, valid_idx.size - 1)
    right_pos = np.clip(pos, 0, valid_idx.size - 1)
    left_idx = valid_idx[left_pos]
    right_idx = valid_idx[right_pos]
    choose_right = np.abs(right_idx - idx) < np.abs(idx - left_idx)
    nearest = left_idx.copy()
    nearest[choose_right] = right_idx[choose_right]
    out[~mask] = out[nearest[~mask]]
    return out


def interp_with_nan(old_t: np.ndarray, y: np.ndarray, new_t: np.ndarray) -> np.ndarray:
    old_t = np.asarray(old_t, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(old_t) & np.isfinite(y)
    if mask.sum() < 2:
        return np.full(new_t.shape, np.nan, dtype=np.float64)
    old_t = old_t[mask]
    y = y[mask]
    order = np.argsort(old_t)
    old_t = old_t[order]
    y = y[order]
    dedup = np.concatenate(([True], np.diff(old_t) > 0))
    old_t = old_t[dedup]
    y = y[dedup]
    if old_t.size < 2:
        return np.full(new_t.shape, np.nan, dtype=np.float64)
    return np.interp(new_t, old_t, y, left=np.nan, right=np.nan)


def gausswin(n: int, alpha: float = 2.5) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    m = np.arange(n, dtype=np.float64) - (n - 1) / 2
    denom = (n - 1) / 2 if n > 1 else 1.0
    return np.exp(-0.5 * (alpha * m / denom) ** 2)


def my_smooth(x: np.ndarray, n: int, bctype: str = "none") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if n in (0, 1):
        return arr.copy()
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr[:, None]

    if bctype.lower() == "reflect":
        arr_filt = np.concatenate([arr[:n, :], arr], axis=0)
        trim = n
    elif bctype.lower() == "zeropad":
        arr_filt = np.concatenate([np.zeros((n, arr.shape[1])), arr], axis=0)
        trim = n
    else:
        arr_filt = arr
        trim = 0

    kernel = gausswin(n)
    kernel[: n // 2] = 0
    kernel /= kernel.sum()

    out = np.empty_like(arr_filt)
    for col in range(arr_filt.shape[1]):
        out[:, col] = np.convolve(arr_filt[:, col], kernel, mode="same")
    out = out[trim:, :]
    if was_1d:
        return out[:, 0]
    return out


def get_stim_enable(bp: dict) -> np.ndarray:
    stim = bp.get("stim")
    if isinstance(stim, dict) and "enable" in stim:
        return as_array(stim["enable"], bool).ravel()
    return np.zeros(int(bp["Ntrials"]), dtype=bool)


def get_quality_mask(probe: dict) -> np.ndarray:
    qualities = np.array([flatten_string(q).strip().lower() for q in probe["quality"]], dtype=object)
    bad = np.isin(qualities, ["garbage", "gabrga", "noisy", "real?"])
    return ~bad


def find_video_offset(obj: dict) -> float:
    bit_start = mode_scalar(obj["bp"]["ev"]["bitStart"])
    bitcode = obj["sglx"]["bitcode"]
    bitstart = as_array(bitcode["bitstart"], np.float64).ravel()
    fs = float(np.asarray(obj["sglx"]["fs"]).reshape(-1)[0])
    vid_file_offset = mode_scalar(bitstart) / fs
    return float(vid_file_offset - bit_start)


def get_feat_names(cam_dict: dict) -> list[str]:
    for feat_names in cam_dict["featNames"]:
        names = [flatten_string(x) for x in feat_names]
        if any(names):
            return names
    raise ValueError("Could not find any feature names for camera")


def find_dlc_feat_index(obj: dict, view_index: int, feat_name: str) -> int:
    cam = obj["traj"][view_index]
    feat_names = get_feat_names(cam)
    exact = [i for i, name in enumerate(feat_names) if name == feat_name]
    if exact:
        return exact[0]
    contains = [i for i, name in enumerate(feat_names) if feat_name in name]
    if contains:
        return contains[0]
    raise KeyError(f"Could not find DLC feature '{feat_name}'")


def get_trial_frame_times(cam: dict, trial_idx: int) -> np.ndarray:
    frame_times = cam["frameTimes"][trial_idx]
    return as_array(frame_times, np.float64).ravel()


def get_trial_ts(cam: dict, trial_idx: int) -> np.ndarray:
    return as_array(cam["ts"][trial_idx], np.float64)


def find_position(
    obj: dict,
    view_index: int,
    feat_name: str,
    taxis: np.ndarray,
    align_times: np.ndarray,
    vidshift: float,
) -> tuple[np.ndarray, np.ndarray]:
    cam = obj["traj"][view_index]
    feat_idx = find_dlc_feat_index(obj, view_index, feat_name)
    ntrials = int(obj["bp"]["Ntrials"])
    xpos = np.full((taxis.size, ntrials), np.nan, dtype=np.float64)
    ypos = np.full((taxis.size, ntrials), np.nan, dtype=np.float64)
    is_tongue = "tongue" in feat_name

    for trix in range(ntrials):
        dropped = np.asarray(cam["NdroppedFrames"][trix]).reshape(-1)
        if dropped.size and np.isnan(dropped[0]):
            continue

        ts = get_trial_ts(cam, trix)[:, :2, feat_idx]
        frame_times = get_trial_frame_times(cam, trix)
        use_fallback = frame_times.size == 0 or np.all(~np.isfinite(frame_times))
        if use_fallback:
            frame_times = np.arange(1, ts.shape[0] + 1, dtype=np.float64) / 400.0
            old_t = frame_times - 0.5 - align_times[trix]
        else:
            old_t = frame_times - vidshift - align_times[trix]

        xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)
        ypos[:, trix] = interp_with_nan(old_t, ts[:, 1], taxis)

        if not is_tongue:
            xpos[:, trix] = fill_nearest_1d(xpos[:, trix], fill_value=0.0)
            ypos[:, trix] = fill_nearest_1d(ypos[:, trix], fill_value=0.0)

    return xpos, ypos


def find_velocity(xpos: np.ndarray, ypos: np.ndarray, feat_name: str) -> tuple[np.ndarray, np.ndarray]:
    xvel = np.empty_like(xpos)
    yvel = np.empty_like(ypos)
    is_tongue = "tongue" in feat_name

    for trix in range(xpos.shape[1]):
        tsinterp = np.column_stack([xpos[:, trix], ypos[:, trix]])
        basederiv = np.nanmedian(np.diff(tsinterp, axis=0), axis=0)
        if np.isnan(basederiv[0]):
            basederiv = np.array([0.0, 0.0])

        xvel[:, trix] = np.gradient(tsinterp[:, 0])
        yvel[:, trix] = np.gradient(tsinterp[:, 1])
        if not is_tongue:
            # Match the MATLAB implementation exactly, including the y-axis subtraction term.
            xvel[:, trix] = xvel[:, trix] - basederiv[0]
            yvel[:, trix] = yvel[:, trix] - basederiv[0]
            xvel[:, trix] = fill_nearest_1d(xvel[:, trix], fill_value=0.0)
            yvel[:, trix] = fill_nearest_1d(yvel[:, trix], fill_value=0.0)
        else:
            xvel[:, trix] = np.nan_to_num(xvel[:, trix], nan=0.0)
            yvel[:, trix] = np.nan_to_num(yvel[:, trix], nan=0.0)

    return xvel, yvel


def compute_feature_speed(
    obj: dict,
    feature_specs: list[tuple[int, str]],
    taxis: np.ndarray,
    align_times: np.ndarray,
    vidshift: float,
) -> np.ndarray:
    speeds = []
    for view_index, feat_name in feature_specs:
        xpos, ypos = find_position(obj, view_index, feat_name, taxis, align_times, vidshift)
        xvel, yvel = find_velocity(xpos, ypos, feat_name)
        speeds.append(np.sqrt(xvel**2 + yvel**2))
    if not speeds:
        return np.zeros((taxis.size, int(obj["bp"]["Ntrials"])), dtype=np.float64)
    stacked = np.stack(speeds, axis=0)
    out = np.nanmean(stacked, axis=0)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def load_motion_energy(obj: dict, motion_energy_path: Path, taxis: np.ndarray, align_times: np.ndarray) -> dict:
    mat = sio.loadmat(motion_energy_path, struct_as_record=False, squeeze_me=True)
    me = mat["me"]
    raw_data = me.data
    if hasattr(raw_data, "data"):
        raw_data = raw_data.data
    raw_trials = list(np.asarray(raw_data, dtype=object).ravel())
    ntrials = int(obj["bp"]["Ntrials"])
    if len(raw_trials) != ntrials:
        raise ValueError(f"Motion energy trial count mismatch for {motion_energy_path.name}")

    cam = obj["traj"][0]
    vidshift = find_video_offset(obj)
    resampled = np.full((taxis.size, ntrials), np.nan, dtype=np.float64)
    for trix, trial_me in enumerate(raw_trials):
        trial_me = as_array(trial_me, np.float64).ravel()
        frame_times = get_trial_frame_times(cam, trix)
        use_fallback = frame_times.size == 0 or np.all(~np.isfinite(frame_times))
        if use_fallback:
            frame_times = np.arange(1, trial_me.size + 1, dtype=np.float64) / 400.0
            old_t = frame_times - 0.5 - align_times[trix]
        else:
            old_t = frame_times - vidshift - align_times[trix]
        resampled[:, trix] = interp_with_nan(old_t, trial_me, taxis)
        resampled[:, trix] = fill_nearest_1d(resampled[:, trix], fill_value=0.0)

    return {
        "data": resampled,
        "moveThresh": float(np.asarray(me.moveThresh).reshape(-1)[0]),
        "vidshift": vidshift,
    }


def build_low_fr_conditions(bp: dict) -> list[np.ndarray]:
    hit = as_array(bp["hit"], bool).ravel()
    miss = as_array(bp["miss"], bool).ravel()
    no = as_array(bp["no"], bool).ravel()
    early = as_array(bp["early"], bool).ravel()
    autowater = as_array(bp["autowater"], bool).ravel()
    stim = get_stim_enable(bp)
    return [
        hit | miss | no,
        hit & ~stim & ~autowater,
        hit & ~stim & autowater,
        miss & ~stim & ~autowater,
        miss & ~stim & autowater,
        hit & ~stim & ~autowater & ~early,
        hit & ~stim & autowater & ~early,
    ]


def build_keep_trial_mask(bp: dict) -> np.ndarray:
    hit = as_array(bp["hit"], bool).ravel()
    miss = as_array(bp["miss"], bool).ravel()
    early = as_array(bp["early"], bool).ravel()
    no = as_array(bp["no"], bool).ravel()
    stim = get_stim_enable(bp)
    return (hit | miss) & ~early & ~no & ~stim


def build_output_constants(bp: dict, keep_trials: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lick_direction = as_array(bp["R"], np.int64).ravel()[keep_trials]
    context = (~as_array(bp["autowater"], bool).ravel()[keep_trials]).astype(np.int64)
    outcome = as_array(bp["hit"], np.int64).ravel()[keep_trials]
    return lick_direction, context, outcome


def build_aligned_trialdat(probe: dict, go_cue: np.ndarray, ntrials: int) -> np.ndarray:
    quality_mask = get_quality_mask(probe)
    keep_units = np.flatnonzero(quality_mask)
    n_units = keep_units.size
    trialdat = np.zeros((NT, n_units, ntrials), dtype=np.float32)

    for unit_pos, unit_idx in enumerate(keep_units):
        spike_trials = as_array(probe["trial"][unit_idx], np.int64).ravel() - 1
        spike_times = as_array(probe["trialtm"][unit_idx], np.float64).ravel()
        finite = (spike_trials >= 0) & (spike_trials < ntrials) & np.isfinite(spike_times)
        spike_trials = spike_trials[finite]
        spike_times = spike_times[finite]
        aligned = spike_times - go_cue[spike_trials]
        bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
        in_window = (bins >= 0) & (bins < NT)
        spike_trials = spike_trials[in_window]
        bins = bins[in_window]
        counts = np.zeros((ntrials, NT), dtype=np.float64)
        np.add.at(counts, (spike_trials, bins), 1.0)
        rates = counts.T / DT
        trialdat[:, unit_pos, :] = my_smooth(rates, SMOOTH, bctype="reflect").astype(np.float32)

    return trialdat


def apply_low_fr_filter(trialdat: np.ndarray, bp: dict) -> tuple[np.ndarray, np.ndarray]:
    conds = build_low_fr_conditions(bp)
    psth = np.zeros((trialdat.shape[0], trialdat.shape[1], len(conds)), dtype=np.float32)
    for cond_idx, cond_mask in enumerate(conds):
        trials = np.flatnonzero(cond_mask)
        if trials.size:
            psth[:, :, cond_idx] = trialdat[:, :, trials].mean(axis=2)
    mean_frs = psth.mean(axis=0).mean(axis=1)
    keep = mean_frs > LOW_FR
    return keep, mean_frs


def binarize_session_signal(
    signal: np.ndarray,
    keep_trials: np.ndarray,
    ignore_zeros_for_threshold: bool = False,
) -> tuple[np.ndarray, float]:
    kept = signal[:, keep_trials]
    threshold_source = kept
    if ignore_zeros_for_threshold:
        threshold_source = kept[kept > 0]
    if np.size(threshold_source) == 0:
        threshold = 0.0
    else:
        threshold = float(np.nanpercentile(threshold_source, 50))
    binary = (kept >= threshold).astype(np.int64)
    return binary, threshold


def subset_data(data: dict, session_trial_indices: list[np.ndarray]) -> dict:
    subset = copy.deepcopy(data)
    for sess_idx, trial_idx in enumerate(session_trial_indices):
        subset["neural"][sess_idx] = [subset["neural"][sess_idx][i] for i in trial_idx]
        subset["input"][sess_idx] = [subset["input"][sess_idx][i] for i in trial_idx]
        subset["output"][sess_idx] = [subset["output"][sess_idx][i] for i in trial_idx]

    used_sessions = [idx for idx, trials in enumerate(session_trial_indices) if len(trials) > 0]
    subset["neural"] = [subset["neural"][idx] for idx in used_sessions]
    subset["input"] = [subset["input"][idx] for idx in used_sessions]
    subset["output"] = [subset["output"][idx] for idx in used_sessions]
    subset["brain_region_idx"] = [subset["brain_region_idx"][idx] for idx in used_sessions]
    subset["subject_idx"] = subset["subject_idx"][used_sessions]

    old_subject_idx = subset["subject_idx"].copy()
    used_subjects = sorted(np.unique(old_subject_idx).tolist())
    remap = {old: new for new, old in enumerate(used_subjects)}
    subset["subjects"] = [subset["subjects"][old] for old in used_subjects]
    subset["subject_idx"] = np.array([remap[int(x)] for x in old_subject_idx], dtype=np.int64)

    subset["metadata"] = copy.deepcopy(subset["metadata"])
    subset["metadata"]["sample_subset"] = True
    subset["metadata"]["sample_trial_indices"] = [trial_idx.tolist() for trial_idx in session_trial_indices]
    subset["metadata"]["session_info"] = [subset["metadata"]["session_info"][idx] for idx in used_sessions]
    return subset


def build_sample_dataset(data: dict, per_combo: int = 3) -> dict:
    per_session_indices = []
    for session_outputs in data["output"]:
        combos: dict[tuple[int, int, int], list[int]] = {}
        for trial_idx, trial_out in enumerate(session_outputs):
            combo = tuple(int(x) for x in trial_out[:3, 0])
            combos.setdefault(combo, []).append(trial_idx)
        chosen: list[int] = []
        for combo in sorted(combos):
            chosen.extend(combos[combo][:per_combo])
        if len(chosen) < 2:
            chosen = list(range(min(2, len(session_outputs))))
        per_session_indices.append(np.array(sorted(set(chosen)), dtype=np.int64))
    return subset_data(data, per_session_indices)


def save_pickle(path: Path, data: dict) -> None:
    with path.open("wb") as f:
        pickle.dump(data, f)


def format_session_summary(summary: dict) -> str:
    return (
        f"{summary['session_id']}: kept {summary['n_kept_trials']}/{summary['n_raw_trials']} trials, "
        f"{summary['n_units_after_quality']} quality-passing units, "
        f"{summary['n_units_after_low_fr']} units after FR filter, "
        f"thresholds(tongue={summary['tongue_threshold']:.3f}, paw={summary['paw_threshold']:.3f}, "
        f"motion={summary['motion_threshold']:.3f})"
    )


def convert_dataset() -> tuple[dict, dict, list[dict]]:
    subject_to_idx: dict[str, int] = {}
    subjects: list[str] = []
    full_data = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": subjects,
        "subject_idx": [],
        "brain_regions": ["ALM"],
        "brain_region_idx": [],
        "input_names": ["time_from_go_cue_s"],
        "output_names": [
            "lick_direction",
            "behavioral_context",
            "outcome",
            "tongue_velocity_bin",
            "paw_velocity_bin",
            "motion_energy_bin",
        ],
        "output_values": [
            ["left", "right"],
            ["WC", "DR"],
            ["incorrect", "correct"],
            ["below_p50", "at_or_above_p50"],
            ["below_p50", "at_or_above_p50"],
            ["below_p50", "at_or_above_p50"],
        ],
        "metadata": {
            "task_description": (
                "Two-context ALM electrophysiology dataset from Hasnain, Birnbaum et al. "
                "with delayed-response (DR) and water-cued (WC) trials."
            ),
            "time_bin_size": float(DT * 1000.0),
            "time_bin_size_sec": float(DT),
            "temporal_alignment_event": (
                "goCue field from the session object; for WC trials the codebase uses the same field as the "
                "water-drop-aligned response cue."
            ),
            "off_start": float(TMIN),
            "off_end": float(TMAX),
            "neural_preprocessing": (
                "Selected the Figure 8 context-session roster, aligned spikes to goCue, binned at 10 ms, "
                "applied the MATLAB causal half-Gaussian smoother with window 15 and reflect padding, "
                "excluded garbage/noisy/real? clusters, and removed units with mean firing rate <= 1 Hz."
            ),
            "trial_exclusion": "Excluded early-lick, no-response, and stim-enable trials from the exported trial set.",
            "kinematic_preprocessing": (
                "Interpolated DLC traces and motion energy onto the neural time grid after video-offset correction; "
                "computed feature velocities from first differences using the same missing-value handling as the MATLAB code."
            ),
            "kinematic_scalar_definitions": {
                "tongue_velocity": "Mean speed across side-camera tongue/left_tongue/right_tongue and bottom-camera top/bottom tongue landmarks.",
                "paw_velocity": "Mean speed across bottom-camera top_paw and bottom_paw landmarks.",
                "motion_energy": "Resampled session motion-energy trace from the companion motionEnergy_*.mat file.",
            },
            "session_roster_source": "code/Scripts/Figure 8/Figure8a_thru_c.m",
            "session_info": [],
            "sanity_checks": {},
        },
    }

    session_summaries: list[dict] = []

    for spec in SESSION_SPECS:
        obj = mat73.loadmat(spec.data_path)["obj"]
        bp = obj["bp"]
        go_cue = as_array(bp["ev"][ALIGN_EVENT], np.float64).ravel()
        ntrials = int(bp["Ntrials"])

        probe = obj["clu"][spec.probe - 1]
        trialdat = build_aligned_trialdat(probe, go_cue, ntrials)
        quality_mask = get_quality_mask(probe)
        low_fr_keep, mean_frs = apply_low_fr_filter(trialdat, bp)
        trialdat = trialdat[:, low_fr_keep, :]

        if trialdat.shape[1] < 10:
            raise ValueError(f"{spec.stem}: fewer than 10 units remained after filtering")

        keep_trials = build_keep_trial_mask(bp)
        kept_trial_indices = np.flatnonzero(keep_trials)
        if kept_trial_indices.size < 2:
            raise ValueError(f"{spec.stem}: fewer than 2 valid trials remained after filtering")

        align_times = go_cue
        taxis = TIME + ADVANCE_MOVEMENT
        me = load_motion_energy(obj, spec.motion_energy_path, taxis, align_times)
        tongue_speed = compute_feature_speed(obj, TONGUE_FEATURES, taxis, align_times, me["vidshift"])
        paw_speed = compute_feature_speed(obj, PAW_FEATURES, taxis, align_times, me["vidshift"])

        tongue_binary, tongue_threshold = binarize_session_signal(
            tongue_speed,
            keep_trials,
            ignore_zeros_for_threshold=True,
        )
        paw_binary, paw_threshold = binarize_session_signal(paw_speed, keep_trials)
        motion_binary, motion_threshold = binarize_session_signal(me["data"], keep_trials)

        lick_direction, context, outcome = build_output_constants(bp, keep_trials)

        subject_idx = subject_to_idx.get(spec.animal)
        if subject_idx is None:
            subject_idx = len(subjects)
            subject_to_idx[spec.animal] = subject_idx
            subjects.append(spec.animal)

        neural_trials = []
        input_trials = []
        output_trials = []
        for out_pos, trial_idx in enumerate(kept_trial_indices):
            neural_trials.append(trialdat[:, :, trial_idx].T.astype(np.float32, copy=False))
            input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
            output_trials.append(
                np.vstack(
                    [
                        np.full((1, NT), lick_direction[out_pos], dtype=np.int16),
                        np.full((1, NT), context[out_pos], dtype=np.int16),
                        np.full((1, NT), outcome[out_pos], dtype=np.int16),
                        tongue_binary[:, out_pos][np.newaxis, :].astype(np.int16, copy=False),
                        paw_binary[:, out_pos][np.newaxis, :].astype(np.int16, copy=False),
                        motion_binary[:, out_pos][np.newaxis, :].astype(np.int16, copy=False),
                    ]
                )
            )

        full_data["neural"].append(neural_trials)
        full_data["input"].append(input_trials)
        full_data["output"].append(output_trials)
        full_data["subject_idx"].append(subject_idx)
        full_data["brain_region_idx"].append(np.zeros(trialdat.shape[1], dtype=np.int64))

        session_summary = {
            "session_id": spec.stem,
            "animal": spec.animal,
            "date": spec.date,
            "probe": spec.probe,
            "file": str(spec.data_path),
            "motion_energy_file": str(spec.motion_energy_path),
            "n_raw_trials": ntrials,
            "n_kept_trials": int(kept_trial_indices.size),
            "n_dr_kept": int(context.sum()),
            "n_wc_kept": int((1 - context).sum()),
            "n_right_kept": int(lick_direction.sum()),
            "n_left_kept": int((1 - lick_direction).sum()),
            "n_correct_kept": int(outcome.sum()),
            "n_incorrect_kept": int((1 - outcome).sum()),
            "n_units_after_quality": int(quality_mask.sum()),
            "n_units_after_low_fr": int(trialdat.shape[1]),
            "mean_fr_hz_min": float(mean_frs[low_fr_keep].min()),
            "mean_fr_hz_max": float(mean_frs[low_fr_keep].max()),
            "tongue_threshold": float(tongue_threshold),
            "paw_threshold": float(paw_threshold),
            "motion_threshold": float(motion_threshold),
            "go_cue_wc_is_finite": bool(np.all(np.isfinite(go_cue[as_array(bp["autowater"], bool).ravel()]))),
        }
        session_summaries.append(session_summary)
        full_data["metadata"]["session_info"].append(session_summary)

        print(format_session_summary(session_summary))

    full_data["subject_idx"] = np.asarray(full_data["subject_idx"], dtype=np.int64)
    total_trials = int(sum(len(sess) for sess in full_data["neural"]))
    total_units = int(sum(len(region_idx) for region_idx in full_data["brain_region_idx"]))
    full_data["metadata"]["sanity_checks"] = {
        "n_sessions": len(full_data["neural"]),
        "n_subjects": len(full_data["subjects"]),
        "total_trials": total_trials,
        "total_units_after_filtering": total_units,
        "all_time_axes_match": True,
        "all_wc_go_cues_finite": all(s["go_cue_wc_is_finite"] for s in session_summaries),
        "paper_context_session_target": {"sessions": 12, "units": 522},
    }

    sample_data = build_sample_dataset(full_data)
    sample_data["metadata"]["sanity_checks"] = copy.deepcopy(full_data["metadata"]["sanity_checks"])
    sample_data["metadata"]["sanity_checks"]["total_trials"] = int(sum(len(sess) for sess in sample_data["neural"]))
    return full_data, sample_data, session_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the two-context ephys dataset into decoder format.")
    parser.add_argument("--full-out", type=Path, default=FULL_OUT)
    parser.add_argument("--sample-out", type=Path, default=SAMPLE_OUT)
    parser.add_argument("--stats-json", type=Path, default=None)
    args = parser.parse_args()

    full_data, sample_data, session_summaries = convert_dataset()
    save_pickle(args.full_out, full_data)
    save_pickle(args.sample_out, sample_data)

    total_trials = sum(len(sess) for sess in full_data["neural"])
    total_units = sum(len(idx) for idx in full_data["brain_region_idx"])
    print("")
    print(f"Saved full dataset to {args.full_out}")
    print(f"Saved sample dataset to {args.sample_out}")
    print(f"Sessions: {len(full_data['neural'])}")
    print(f"Subjects: {len(full_data['subjects'])} -> {full_data['subjects']}")
    print(f"Total trials: {total_trials}")
    print(f"Total units after filtering: {total_units}")

    if args.stats_json is not None:
        stats = {
            "sessions": session_summaries,
            "summary": {
                "n_sessions": len(full_data["neural"]),
                "n_subjects": len(full_data["subjects"]),
                "subjects": full_data["subjects"],
                "total_trials": int(total_trials),
                "total_units_after_filtering": int(total_units),
                "sample_total_trials": int(sum(len(sess) for sess in sample_data["neural"])),
            },
        }
        with args.stats_json.open("w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved stats JSON to {args.stats_json}")


if __name__ == "__main__":
    main()
