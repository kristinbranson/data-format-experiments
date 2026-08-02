#!/usr/bin/env python3
"""Convert Hasnain, Birnbaum et al. ALM context-task data for decoder training.

This script follows the session roster and core preprocessing used by the
paper's Figure 8 context-analysis pipeline:
  - ALM ephys sessions only
  - `goCue` alignment
  - 10 ms neural/video bins
  - time window [-3.0, 2.5] s relative to go cue
  - low firing-rate filtering at 1 Hz after trial-averaged PSTH smoothing

The output format is the dictionary structure expected by `train_decoder.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import scipy.io
from scipy.signal.windows import gaussian


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "Ephys_Behavior"

ALIGN_EVENT = "goCue"
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
SMOOTH = 15
LOW_FR = 1.0
ADVANCE_MOVEMENT = 0.0

BAD_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}

SIDE_FEATURES = [
    "tongue",
    "left_tongue",
    "right_tongue",
    "jaw",
    "trident",
    "nose",
]
BOTTOM_FEATURES = [
    "top_tongue",
    "topleft_tongue",
    "bottom_tongue",
    "bottomleft_tongue",
    "jaw",
    "top_paw",
    "bottom_paw",
    "top_nostril",
    "bottom_nostril",
]


@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    probe: int

    @property
    def stem(self) -> str:
        return f"{self.animal}_{self.date}"

    @property
    def data_path(self) -> Path:
        return DATA_DIR / f"data_structure_{self.stem}.mat"

    @property
    def motion_energy_path(self) -> Path:
        return DATA_DIR / f"motionEnergy_{self.stem}.mat"


# Figure 8 context-task roster from the reference MATLAB code.
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


def matlab_mode(x: np.ndarray) -> float:
    vals = np.asarray(x).reshape(-1)
    vals = vals[~np.isnan(vals)]
    uniq, counts = np.unique(vals, return_counts=True)
    return float(uniq[np.argmax(counts)])


def gausswin(length: int, alpha: float = 2.5) -> np.ndarray:
    std = (length - 1) / (2 * alpha)
    return gaussian(length, std=std, sym=True)


def my_smooth(x: np.ndarray, n: int, bctype: str = "none") -> np.ndarray:
    """Port of `mySmooth.m` operating on axis 0."""
    x = np.asarray(x, dtype=np.float64)
    if n in (0, 1):
        return x.copy()
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    else:
        squeeze = False

    if bctype == "reflect":
        x_filt = np.concatenate([x[:n, :], x], axis=0)
        trim = n
    elif bctype == "zeropad":
        x_filt = np.concatenate([np.zeros((n, x.shape[1])), x], axis=0)
        trim = n
    else:
        x_filt = x
        trim = 0

    kern = gausswin(n)
    kern[: math.floor(kern.size / 2)] = 0.0
    kern /= kern.sum()

    out = np.zeros_like(x_filt)
    for j in range(x_filt.shape[1]):
        out[:, j] = np.convolve(x_filt[:, j], kern, mode="same")
    out = out[trim:, :]
    if squeeze:
        out = out[:, 0]
    return out


def fill_nearest_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    if x.size == 0:
        return x
    idx = np.flatnonzero(~np.isnan(x))
    if idx.size == 0:
        return np.zeros_like(x)
    xp = idx.astype(np.float64)
    fp = x[idx]
    out = np.interp(np.arange(x.size, dtype=np.float64), xp, fp)
    return out


def interp_with_nan(x_old: np.ndarray, y_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    x_old = np.asarray(x_old, dtype=np.float64).reshape(-1)
    y_old = np.asarray(y_old, dtype=np.float64)
    x_new = np.asarray(x_new, dtype=np.float64).reshape(-1)

    valid = ~np.isnan(x_old)
    if y_old.ndim == 1:
        valid &= ~np.isnan(y_old)
        if valid.sum() < 2:
            return np.full_like(x_new, np.nan, dtype=np.float64)
        return np.interp(x_new, x_old[valid], y_old[valid], left=np.nan, right=np.nan)

    out = np.full((x_new.size, y_old.shape[1]), np.nan, dtype=np.float64)
    for col in range(y_old.shape[1]):
        valid_col = valid & ~np.isnan(y_old[:, col])
        if valid_col.sum() < 2:
            continue
        out[:, col] = np.interp(
            x_new,
            x_old[valid_col],
            y_old[valid_col, col],
            left=np.nan,
            right=np.nan,
        )
    return out


def h5_ref_at(dataset: h5py.Dataset, index: int):
    refs = np.asarray(dataset[()], dtype=object).reshape(-1)
    return refs[index]


def read_h5_string(f: h5py.File, node_or_ref) -> str:
    node = f[node_or_ref] if not isinstance(node_or_ref, (h5py.Dataset, h5py.Group)) else node_or_ref
    if isinstance(node, h5py.Group):
        return ""
    arr = np.asarray(node[()])
    if arr.dtype == object:
        if arr.size == 0:
            return ""
        return read_h5_string(f, arr.reshape(-1)[0])
    if arr.dtype == np.uint16:
        chars = [chr(int(v)) for v in arr.reshape(-1) if int(v) != 0]
        return "".join(chars).strip()
    if arr.dtype == np.uint64 and np.all(arr == 0):
        return ""
    if np.issubdtype(arr.dtype, np.number):
        return ""
    return ""


def read_h5_numeric(node: h5py.Dataset) -> np.ndarray:
    return np.asarray(node[()], dtype=np.float64)


def read_h5_vector(node: h5py.Dataset) -> np.ndarray:
    return read_h5_numeric(node).reshape(-1)


def read_h5_scalar(node: h5py.Dataset) -> float:
    return float(read_h5_vector(node)[0])


def read_feat_names(f: h5py.File, view_group: h5py.Group, trial_idx: int) -> list[str]:
    feat_cell = f[h5_ref_at(view_group["featNames"], trial_idx)]
    refs = np.asarray(feat_cell[()], dtype=object).reshape(-1)
    return [read_h5_string(f, ref) for ref in refs]


def find_feat_index(f: h5py.File, view_group: h5py.Group, feat_name: str, n_trials: int) -> int:
    for trial_idx in range(n_trials):
        names = read_feat_names(f, view_group, trial_idx)
        if feat_name in names:
            return names.index(feat_name)
    raise KeyError(f"Could not find feature '{feat_name}'")


def read_frame_times(f: h5py.File, view_group: h5py.Group, trial_idx: int) -> np.ndarray:
    frame_ds = f[h5_ref_at(view_group["frameTimes"], trial_idx)]
    return read_h5_vector(frame_ds)


def read_ndropped_frames(f: h5py.File, view_group: h5py.Group, trial_idx: int) -> float:
    ndropped_ds = f[h5_ref_at(view_group["NdroppedFrames"], trial_idx)]
    vals = read_h5_vector(ndropped_ds)
    return float(vals[0]) if vals.size else np.nan


def read_feature_xy(
    f: h5py.File,
    view_group: h5py.Group,
    trial_idx: int,
    feat_idx: int,
) -> np.ndarray:
    ts_ds = f[h5_ref_at(view_group["ts"], trial_idx)]
    arr = np.asarray(ts_ds[()], dtype=np.float64)
    # HDF5 stores the MATLAB array with axes reversed here: (feature, xyz, frame).
    xy = arr[feat_idx, 0:2, :].T
    return xy


def find_video_offset(bit_start: np.ndarray, bitcode_bitstart: np.ndarray, fs: float) -> float:
    return matlab_mode(bitcode_bitstart) / fs - matlab_mode(bit_start)


def load_motion_energy(path: Path) -> tuple[np.ndarray, float]:
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    me = mat["me"]
    trial_arrays = np.asarray(me.data, dtype=object).reshape(-1)
    return trial_arrays, float(me.moveThresh)


def aligned_motion_energy(
    frame_times_by_trial: list[np.ndarray],
    raw_motion_energy: np.ndarray,
    align_times: np.ndarray,
    vidshift: float,
    taxis: np.ndarray,
) -> np.ndarray:
    n_trials = align_times.size
    out = np.full((taxis.size, n_trials), np.nan, dtype=np.float64)
    for tr in range(n_trials):
        frame_times = frame_times_by_trial[tr]
        me_trial = np.asarray(raw_motion_energy[tr], dtype=np.float64).reshape(-1)
        out[:, tr] = interp_with_nan(frame_times - vidshift - align_times[tr], me_trial, taxis)
        out[:, tr] = fill_nearest_1d(out[:, tr])
    return out


def aligned_feature_position(
    f: h5py.File,
    view_group: h5py.Group,
    feat_idx: int,
    feat_name: str,
    n_trials: int,
    align_times: np.ndarray,
    vidshift: float,
    taxis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xpos = np.full((taxis.size, n_trials), np.nan, dtype=np.float64)
    ypos = np.full((taxis.size, n_trials), np.nan, dtype=np.float64)

    for tr in range(n_trials):
        ndropped = read_ndropped_frames(f, view_group, tr)
        if np.isnan(ndropped):
            continue
        frame_times = read_frame_times(f, view_group, tr)
        xy = read_feature_xy(f, view_group, tr, feat_idx)
        interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
        xpos[:, tr] = interp_xy[:, 0]
        ypos[:, tr] = interp_xy[:, 1]
        if "tongue" not in feat_name:
            xpos[:, tr] = fill_nearest_1d(xpos[:, tr])
            ypos[:, tr] = fill_nearest_1d(ypos[:, tr])

    return xpos, ypos


def aligned_feature_velocity(xpos: np.ndarray, ypos: np.ndarray, feat_name: str) -> tuple[np.ndarray, np.ndarray]:
    xvel = np.zeros_like(xpos)
    yvel = np.zeros_like(ypos)
    for tr in range(xpos.shape[1]):
        tsinterp = np.column_stack([xpos[:, tr], ypos[:, tr]])
        diff_xy = np.diff(tsinterp, axis=0)
        if np.all(np.isnan(diff_xy)):
            basederiv = np.zeros(2, dtype=np.float64)
        else:
            basederiv = np.nanmedian(diff_xy, axis=0)
        xvel[:, tr] = np.gradient(tsinterp[:, 0])
        yvel[:, tr] = np.gradient(tsinterp[:, 1])
        if "tongue" not in feat_name:
            xvel[:, tr] = xvel[:, tr] - basederiv[0]
            yvel[:, tr] = yvel[:, tr] - basederiv[0]
            xvel[:, tr] = fill_nearest_1d(xvel[:, tr])
            yvel[:, tr] = fill_nearest_1d(yvel[:, tr])
        else:
            xvel[:, tr][np.isnan(xvel[:, tr])] = 0.0
            yvel[:, tr][np.isnan(yvel[:, tr])] = 0.0
    return xvel, yvel


def build_edges_and_time() -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
    if not np.isclose(edges[-1], TMAX):
        edges = np.append(edges, TMAX)
    time = edges[:-1] + DT / 2
    return edges, time


def compute_condition_masks(bp: dict[str, np.ndarray]) -> list[np.ndarray]:
    hit = bp["hit"].astype(bool)
    miss = bp["miss"].astype(bool)
    no = bp["no"].astype(bool)
    early = bp["early"].astype(bool)
    autowater = bp["autowater"].astype(bool)
    stim_enable = bp["stim_enable"].astype(bool)

    return [
        hit | miss | no,
        hit & ~stim_enable & ~autowater,
        hit & ~stim_enable & autowater,
        miss & ~stim_enable & ~autowater,
        miss & ~stim_enable & autowater,
        hit & ~stim_enable & ~autowater & ~early,
        hit & ~stim_enable & autowater & ~early,
    ]


def trial_selector(bp: dict[str, np.ndarray]) -> np.ndarray:
    hit_or_miss = bp["hit"].astype(bool) | bp["miss"].astype(bool)
    return hit_or_miss & ~bp["early"].astype(bool) & ~bp["stim_enable"].astype(bool)


def load_trial_spikes(
    f: h5py.File,
    clu_group: h5py.Group,
    clu_idx: int,
    align_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trial_ds = f[h5_ref_at(clu_group["trial"], clu_idx)]
    trialtm_ds = f[h5_ref_at(clu_group["trialtm"], clu_idx)]
    session_trial = np.asarray(trial_ds[()], dtype=np.float64).reshape(-1).astype(np.int64) - 1
    trialtm = np.asarray(trialtm_ds[()], dtype=np.float64).reshape(-1)
    aligned = trialtm - align_times[session_trial]
    return session_trial, trialtm, aligned


def compute_psth_mean_fr(
    session_trial: np.ndarray,
    aligned_spikes: np.ndarray,
    condition_masks: list[np.ndarray],
    edges: np.ndarray,
) -> float:
    psths = []
    for cond_mask in condition_masks:
        trix = np.flatnonzero(cond_mask)
        if trix.size == 0:
            psths.append(np.zeros(edges.size - 1, dtype=np.float64))
            continue
        spike_mask = np.isin(session_trial, trix)
        if not np.any(spike_mask):
            psths.append(np.zeros(edges.size - 1, dtype=np.float64))
            continue
        counts, _ = np.histogram(aligned_spikes[spike_mask], bins=edges)
        psth = my_smooth(counts / trix.size / DT, SMOOTH, "reflect")
        psths.append(psth)
    psth_stack = np.column_stack(psths)
    return float(np.nanmean(psth_stack))


def build_neural_trials(
    session_trial: np.ndarray,
    aligned_spikes: np.ndarray,
    included_trials: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    n_time = edges.size - 1
    out = np.zeros((n_time, included_trials.size), dtype=np.float64)
    for col, tr in enumerate(included_trials):
        spike_mask = session_trial == tr
        if not np.any(spike_mask):
            continue
        counts, _ = np.histogram(aligned_spikes[spike_mask], bins=edges)
        out[:, col] = my_smooth(counts / DT, SMOOTH, "reflect")
    return out


def lick_direction_from_trial(bp: dict[str, np.ndarray], trial_idx: int) -> int:
    hit = bool(bp["hit"][trial_idx])
    is_right_trial = bool(bp["R"][trial_idx])
    if hit:
        return int(is_right_trial)
    return int(not is_right_trial)


def derive_session_outputs(
    bp: dict[str, np.ndarray],
    included_trials: np.ndarray,
    tongue_speed: np.ndarray,
    paw_speed: np.ndarray,
    motion_energy: np.ndarray,
) -> tuple[list[np.ndarray], dict[str, float]]:
    tongue_use = tongue_speed[:, included_trials]
    paw_use = paw_speed[:, included_trials]
    me_use = motion_energy[:, included_trials]

    positive_tongue = tongue_use[tongue_use > 0]
    if positive_tongue.size:
        tongue_thresh = float(np.nanpercentile(positive_tongue, 50))
    else:
        tongue_thresh = 0.0
    paw_thresh = float(np.nanpercentile(paw_use, 50))
    me_thresh = float(np.nanpercentile(me_use, 50))

    out_trials = []
    for tr in included_trials:
        lick_dir = lick_direction_from_trial(bp, tr)
        context = 0 if bp["autowater"][tr] else 1
        outcome = 1 if bp["hit"][tr] else 0

        tr_out = np.vstack(
            [
                np.full(tongue_speed.shape[0], lick_dir, dtype=np.int16),
                np.full(tongue_speed.shape[0], context, dtype=np.int16),
                np.full(tongue_speed.shape[0], outcome, dtype=np.int16),
                (tongue_speed[:, tr] >= tongue_thresh).astype(np.int16),
                (paw_speed[:, tr] >= paw_thresh).astype(np.int16),
                (motion_energy[:, tr] >= me_thresh).astype(np.int16),
            ]
        )
        out_trials.append(tr_out)

    return out_trials, {
        "tongue_velocity_median_threshold": tongue_thresh,
        "paw_velocity_median_threshold": paw_thresh,
        "motion_energy_median_threshold": me_thresh,
    }


def load_session(spec: SessionSpec) -> dict:
    edges, time = build_edges_and_time()
    taxis = time + ADVANCE_MOVEMENT

    with h5py.File(spec.data_path, "r") as f:
        bp_group = f["obj/bp"]
        ev_group = bp_group["ev"]
        stim_group = bp_group["stim"]

        bp = {
            "Ntrials": int(round(read_h5_scalar(bp_group["Ntrials"]))),
            "hit": read_h5_vector(bp_group["hit"]).astype(np.int16),
            "miss": read_h5_vector(bp_group["miss"]).astype(np.int16),
            "no": read_h5_vector(bp_group["no"]).astype(np.int16),
            "early": read_h5_vector(bp_group["early"]).astype(np.int16),
            "autowater": read_h5_vector(bp_group["autowater"]).astype(np.int16),
            "R": read_h5_vector(bp_group["R"]).astype(np.int16),
            "L": read_h5_vector(bp_group["L"]).astype(np.int16),
            "stim_enable": read_h5_vector(stim_group["enable"]).astype(np.int16),
            "bitStart": read_h5_vector(ev_group["bitStart"]),
            "sample": read_h5_vector(ev_group["sample"]),
            "delay": read_h5_vector(ev_group["delay"]),
            "goCue": read_h5_vector(ev_group["goCue"]),
        }

        included_mask = trial_selector(bp)
        included_trials = np.flatnonzero(included_mask)
        if included_trials.size < 2:
            raise RuntimeError(f"{spec.stem}: fewer than 2 usable trials after filtering")

        condition_masks = compute_condition_masks(bp)

        fs = read_h5_scalar(f["obj/sglx/fs"])
        bitcode_bitstart = read_h5_vector(f["obj/sglx/bitcode/bitstart"])
        vidshift = find_video_offset(bp["bitStart"], bitcode_bitstart, fs)

        traj_root = f["obj/traj"]
        side_group = f[h5_ref_at(traj_root, 0)]
        bottom_group = f[h5_ref_at(traj_root, 1)]

        frame_times_by_trial = [read_frame_times(f, side_group, tr) for tr in range(bp["Ntrials"])]
        raw_motion_energy, raw_move_thresh = load_motion_energy(spec.motion_energy_path)
        motion_energy = aligned_motion_energy(
            frame_times_by_trial=frame_times_by_trial,
            raw_motion_energy=raw_motion_energy,
            align_times=bp[ALIGN_EVENT],
            vidshift=vidshift,
            taxis=taxis,
        )

        feat_indices = {
            "top_tongue": find_feat_index(f, bottom_group, "top_tongue", bp["Ntrials"]),
            "bottom_tongue": find_feat_index(f, bottom_group, "bottom_tongue", bp["Ntrials"]),
            "top_paw": find_feat_index(f, bottom_group, "top_paw", bp["Ntrials"]),
            "bottom_paw": find_feat_index(f, bottom_group, "bottom_paw", bp["Ntrials"]),
        }

        velocities = {}
        for feat_name, feat_idx in feat_indices.items():
            xpos, ypos = aligned_feature_position(
                f=f,
                view_group=bottom_group,
                feat_idx=feat_idx,
                feat_name=feat_name,
                n_trials=bp["Ntrials"],
                align_times=bp[ALIGN_EVENT],
                vidshift=vidshift,
                taxis=taxis,
            )
            velocities[feat_name] = aligned_feature_velocity(xpos, ypos, feat_name)

        top_tongue_xvel, top_tongue_yvel = velocities["top_tongue"]
        bottom_tongue_xvel, bottom_tongue_yvel = velocities["bottom_tongue"]
        tongue_tip_xvel = 0.5 * (top_tongue_xvel + bottom_tongue_xvel)
        tongue_tip_yvel = 0.5 * (top_tongue_yvel + bottom_tongue_yvel)
        tongue_speed = np.sqrt(tongue_tip_xvel ** 2 + tongue_tip_yvel ** 2)
        tongue_speed = np.nan_to_num(tongue_speed, nan=0.0)

        top_paw_xvel, top_paw_yvel = velocities["top_paw"]
        bottom_paw_xvel, bottom_paw_yvel = velocities["bottom_paw"]
        top_paw_speed = np.sqrt(top_paw_xvel ** 2 + top_paw_yvel ** 2)
        bottom_paw_speed = np.sqrt(bottom_paw_xvel ** 2 + bottom_paw_yvel ** 2)
        paw_speed = 0.5 * (top_paw_speed + bottom_paw_speed)
        paw_speed = np.nan_to_num(paw_speed, nan=0.0)

        clu_group = f[h5_ref_at(f["obj/clu"], spec.probe - 1)]
        qds = clu_group["quality"]

        kept_cluster_indices = []
        prefilter_unit_count = 0
        raw_quality_kept = 0
        neural_by_trial = []
        cluster_ids = []

        for clu_idx in range(qds.shape[0]):
            quality = read_h5_string(f, h5_ref_at(qds, clu_idx)).strip().lower()
            if quality in BAD_QUALITIES:
                continue
            raw_quality_kept += 1

            session_trial, _trialtm, aligned = load_trial_spikes(f, clu_group, clu_idx, bp[ALIGN_EVENT])
            mean_fr = compute_psth_mean_fr(
                session_trial=session_trial,
                aligned_spikes=aligned,
                condition_masks=condition_masks,
                edges=edges,
            )
            if mean_fr <= LOW_FR:
                continue

            kept_cluster_indices.append(clu_idx)
            prefilter_unit_count += 1
            cluster_ids.append(clu_idx)
            neural_by_trial.append(
                build_neural_trials(
                    session_trial=session_trial,
                    aligned_spikes=aligned,
                    included_trials=included_trials,
                    edges=edges,
                )
            )

        if not neural_by_trial:
            raise RuntimeError(f"{spec.stem}: no units survived quality and low-FR filtering")

        neural_arr = np.stack(neural_by_trial, axis=1)  # (time, neurons, trials)

    session_neural = [
        np.asarray(neural_arr[:, :, tr].T, dtype=np.float32)
        for tr in range(neural_arr.shape[2])
    ]
    session_input = [
        np.asarray(time[None, :], dtype=np.float32)
        for _ in range(neural_arr.shape[2])
    ]
    session_output, thresholds = derive_session_outputs(
        bp=bp,
        included_trials=included_trials,
        tongue_speed=tongue_speed,
        paw_speed=paw_speed,
        motion_energy=motion_energy,
    )

    sanity = {
        "session": spec.stem,
        "probe": spec.probe,
        "n_trials_total": int(bp["Ntrials"]),
        "n_trials_included": int(included_trials.size),
        "n_units_after_quality_only": int(raw_quality_kept),
        "n_units_after_low_fr": int(len(session_neural[0])),
        "raw_motion_energy_threshold": float(raw_move_thresh),
        **thresholds,
    }

    return {
        "neural": session_neural,
        "input": session_input,
        "output": session_output,
        "brain_region_idx": np.zeros(session_neural[0].shape[0], dtype=np.int64),
        "subject": spec.animal,
        "sanity": sanity,
    }


def subset_data(data: dict, session_indices: Iterable[int]) -> dict:
    session_indices = list(session_indices)
    subject_names = [data["subjects"][data["subject_idx"][i]] for i in session_indices]
    unique_subjects = list(OrderedDict.fromkeys(subject_names))
    subject_to_idx = {name: i for i, name in enumerate(unique_subjects)}

    subset = {
        "neural": [data["neural"][i] for i in session_indices],
        "input": [data["input"][i] for i in session_indices],
        "output": [data["output"][i] for i in session_indices],
        "subjects": unique_subjects,
        "subject_idx": np.asarray([subject_to_idx[name] for name in subject_names], dtype=np.int64),
        "brain_regions": list(data["brain_regions"]),
        "brain_region_idx": [data["brain_region_idx"][i].copy() for i in session_indices],
        "input_names": list(data["input_names"]),
        "output_names": list(data["output_names"]),
        "output_values": [list(vals) for vals in data["output_values"]],
        "metadata": dict(data["metadata"]),
    }
    subset["metadata"]["session_info"] = [data["metadata"]["session_info"][i] for i in session_indices]
    subset["metadata"]["source_data_file"] = data["metadata"]["source_data_file"]
    subset["metadata"]["is_sample_subset"] = True
    return subset


def build_dataset() -> dict:
    sessions = [load_session(spec) for spec in SESSION_SPECS]

    subjects = list(OrderedDict.fromkeys(sess["subject"] for sess in sessions))
    subject_to_idx = {subj: i for i, subj in enumerate(subjects)}

    data = {
        "neural": [sess["neural"] for sess in sessions],
        "input": [sess["input"] for sess in sessions],
        "output": [sess["output"] for sess in sessions],
        "subjects": subjects,
        "subject_idx": np.asarray([subject_to_idx[sess["subject"]] for sess in sessions], dtype=np.int64),
        "brain_regions": ["ALM"],
        "brain_region_idx": [sess["brain_region_idx"] for sess in sessions],
        "input_names": ["time_from_go_cue_seconds"],
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
            ["low", "high"],
            ["low", "high"],
            ["low", "high"],
        ],
        "metadata": {
            "task_description": (
                "ALM neural activity aligned to the go cue during the two-context task; "
                "decoder predicts choice, context, outcome, and binarized movement variables."
            ),
            "time_bin_size": float(DT * 1000),
            "temporal_alignment_event": "Go cue onset",
            "off_start": float(TMIN),
            "off_end": float(TMAX),
            "source_data_file": "Separating cognitive and motor processes in the behaving mouse",
            "alignment_event_key": ALIGN_EVENT,
            "selected_session_criterion": (
                "Figure 8 context-task ALM ephys session roster from the reference MATLAB code"
            ),
            "neural_processing": (
                "10 ms spike-rate bins with causal Gaussian smoothing (window 15) and 1 Hz low-FR filtering"
            ),
            "trial_filtering": (
                "Retained hit and miss trials with stim.enable == 0 and early == 0; ignore/no trials excluded"
            ),
            "movement_processing": (
                "Motion energy aligned with video-offset correction; tongue and paw speeds derived from "
                "aligned DLC trajectories and binarized with per-session medians"
            ),
            "session_info": [sess["sanity"] for sess in sessions],
            "paper_reported_context_sessions": 12,
            "paper_reported_context_units": 522,
        },
    }
    return data


def summarize_dataset(data: dict) -> dict:
    n_sessions = len(data["neural"])
    n_trials = [len(sess) for sess in data["neural"]]
    n_neurons = [sess[0].shape[0] for sess in data["neural"]]
    total_units = int(sum(n_neurons))
    total_trials = int(sum(n_trials))
    return {
        "n_sessions": n_sessions,
        "n_subjects": len(data["subjects"]),
        "trials_per_session": n_trials,
        "neurons_per_session": n_neurons,
        "total_trials": total_trials,
        "total_units": total_units,
        "timepoints_per_trial": int(data["neural"][0][0].shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ALM context-task data for decoder training.")
    parser.add_argument("--output", default="converted_data.pkl", help="Output pickle for the full dataset.")
    parser.add_argument("--sample-output", default="sample_data.pkl", help="Output pickle for the sample subset.")
    args = parser.parse_args()

    data = build_dataset()
    summary = summarize_dataset(data)

    with open(ROOT / args.output, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    sample_data = subset_data(data, session_indices=range(4))
    with open(ROOT / args.sample_output, "wb") as f:
        pickle.dump(sample_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Conversion complete.")
    print(json.dumps(summary, indent=2))
    print("Session sanity checks:")
    for session_info in data["metadata"]["session_info"]:
        print(json.dumps(session_info, sort_keys=True))

    total_units = summary["total_units"]
    delta_units = total_units - data["metadata"]["paper_reported_context_units"]
    print(
        "Paper/code sanity check: "
        f"{summary['n_sessions']} context sessions, {total_units} retained units "
        f"(paper reports {data['metadata']['paper_reported_context_units']}, delta {delta_units:+d})."
    )


if __name__ == "__main__":
    main()
