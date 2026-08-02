#!/usr/bin/env python3

import argparse
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mat73
import numpy as np
from scipy.io import loadmat


DT = 0.005
TMIN = -2.5
TMAX = 2.5
SMOOTH_N = 15
BCTYPE = "reflect"
LOW_FR_HZ = 1.0
ALIGN_EVENT = "goCue"
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)


@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    probe_index: int  # 0-based

    @property
    def session_id(self) -> str:
        return f"{self.animal}_{self.date}"

    @property
    def data_path(self) -> Path:
        return Path("data/Ephys_Behavior") / f"data_structure_{self.session_id}.mat"

    @property
    def motion_energy_path(self) -> Path:
        return Path("data/Ephys_Behavior") / f"motionEnergy_{self.session_id}.mat"


CONTEXT_SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 1),
    SessionSpec("JEB7", "2021-04-29", 0),
    SessionSpec("JEB7", "2021-04-30", 0),
    SessionSpec("EKH1", "2021-08-07", 1),
    SessionSpec("EKH3", "2021-08-11", 1),
    SessionSpec("JGR2", "2021-11-16", 0),
    SessionSpec("JGR2", "2021-11-17", 0),
    SessionSpec("JGR3", "2021-11-18", 0),
    SessionSpec("JEB19", "2023-04-21", 0),
    SessionSpec("JEB19", "2023-04-20", 0),
    SessionSpec("JEB19", "2023-04-19", 0),
    SessionSpec("JEB19", "2023-04-18", 0),
]


def gaussian_window(n: int, alpha: float = 2.5) -> np.ndarray:
    if n <= 1:
        return np.ones((n,), dtype=np.float64)
    half = (n - 1) / 2.0
    k = np.arange(0, n, dtype=np.float64) - half
    return np.exp(-0.5 * (alpha * k / half) ** 2)


def my_smooth(x: np.ndarray, n: int, bctype: str = "none") -> np.ndarray:
    if n <= 1:
        return x.copy()

    x = np.asarray(x, dtype=np.float64)
    was_1d = x.ndim == 1
    if was_1d:
        x = x[:, None]

    if bctype.lower() == "reflect":
        x_filt = np.concatenate([x[:n, :], x], axis=0)
        trim = n
    elif bctype.lower() == "zeropad":
        x_filt = np.concatenate([np.zeros((n, x.shape[1]), dtype=x.dtype), x], axis=0)
        trim = n
    else:
        x_filt = x
        trim = 0

    kern = gaussian_window(n)
    kern[: len(kern) // 2] = 0.0
    kern /= np.sum(kern)

    out = np.empty_like(x_filt, dtype=np.float64)
    for col in range(x_filt.shape[1]):
        out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")
    out = out[trim:, :]

    if was_1d:
        return out[:, 0]
    return out


def mode_value(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    vals, counts = np.unique(x, return_counts=True)
    return float(vals[np.argmax(counts)])


def ensure_1d_numeric(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return arr


def event_list_to_array(x) -> np.ndarray:
    if x is None:
        return np.empty((0,), dtype=np.float64)
    if isinstance(x, list):
        vals = []
        for v in x:
            if v is None:
                continue
            arr = np.asarray(v, dtype=np.float64).reshape(-1)
            if arr.size:
                vals.append(arr)
        if not vals:
            return np.empty((0,), dtype=np.float64)
        return np.concatenate(vals).astype(np.float64, copy=False)
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return arr


def fill_nearest_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    if x.size == 0:
        return x
    good = np.isfinite(x)
    if not np.any(good):
        return x
    idx = np.arange(x.size)
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x


def fill_nearest_2d(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    for col in range(out.shape[1]):
        out[:, col] = fill_nearest_1d(out[:, col])
    return out


def normalize_trial_struct(view_dict: dict, trial_index: int) -> dict:
    return {key: view_dict[key][trial_index] for key in view_dict.keys()}


def flatten_feat_names(feats) -> list[str]:
    out = []
    for feat in feats:
        if feat is None:
            continue
        if isinstance(feat, list):
            if len(feat) == 0:
                continue
            if isinstance(feat[0], list):
                out.extend(flatten_feat_names(feat))
            else:
                out.append(str(feat[0]))
        else:
            out.append(str(feat))
    return out


def find_feature_index(view_dict: dict, feat_name: str) -> int:
    feat_lists = view_dict["featNames"]
    for feats in feat_lists:
        feats = flatten_feat_names(feats)
        if feat_name in feats:
            return feats.index(feat_name)
    raise KeyError(f"Could not find feature '{feat_name}'")


def compute_vidshift(obj: dict) -> float:
    bit_start = mode_value(obj["bp"]["ev"]["bitStart"])
    vid_file_offset = mode_value(obj["sglx"]["bitcode"]["bitstart"]) / float(obj["sglx"]["fs"])
    return float(vid_file_offset - bit_start)


def align_feature_positions(
    obj: dict,
    view_index: int,
    feat_name: str,
    align_times: np.ndarray,
    time_axis: np.ndarray,
    vidshift: float,
) -> tuple[np.ndarray, np.ndarray]:
    view_dict = obj["traj"][view_index]
    feat_index = find_feature_index(view_dict, feat_name)
    n_trials = int(obj["bp"]["Ntrials"])
    xpos = np.full((time_axis.size, n_trials), np.nan, dtype=np.float64)
    ypos = np.full((time_axis.size, n_trials), np.nan, dtype=np.float64)

    for trial_idx in range(n_trials):
        trial = normalize_trial_struct(view_dict, trial_idx)
        dropped = trial.get("NdroppedFrames", np.nan)
        dropped_arr = ensure_1d_numeric([] if dropped is None else dropped)
        if dropped_arr.size and np.all(np.isnan(dropped_arr)):
            continue

        frame_times = trial.get("frameTimes")
        ts = trial.get("ts")
        if ts is None:
            continue
        ts = np.asarray(ts, dtype=np.float64)
        if ts.ndim != 3 or feat_index >= ts.shape[2]:
            continue

        if frame_times is None:
            frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0
        else:
            frame_times = ensure_1d_numeric(frame_times)

        if frame_times.size == 0 or np.all(np.isnan(frame_times)):
            frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0

        xy = ts[:, 0:2, feat_index]
        if "tongue" not in feat_name:
            xy = my_smooth(xy, 1, "reflect")

        shifted_time = frame_times - vidshift - align_times[trial_idx]
        valid = np.isfinite(shifted_time) & np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        if np.sum(valid) < 2:
            continue

        xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], left=np.nan, right=np.nan)
        ypos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 1], left=np.nan, right=np.nan)

        if "tongue" not in feat_name:
            xpos[:, trial_idx] = fill_nearest_1d(xpos[:, trial_idx])
            ypos[:, trial_idx] = fill_nearest_1d(ypos[:, trial_idx])

    return xpos, ypos


def compute_velocity(xpos: np.ndarray, ypos: np.ndarray, feat_name: str) -> tuple[np.ndarray, np.ndarray]:
    xvel = np.full_like(xpos, np.nan, dtype=np.float64)
    yvel = np.full_like(ypos, np.nan, dtype=np.float64)

    for trial_idx in range(xpos.shape[1]):
        tsinterp = np.column_stack([xpos[:, trial_idx], ypos[:, trial_idx]])
        diffs = np.diff(tsinterp, axis=0)
        if np.any(np.isfinite(diffs)):
            basederiv = np.nanmedian(diffs, axis=0)
        else:
            basederiv = np.zeros((2,), dtype=np.float64)
        xv = np.gradient(tsinterp[:, 0])
        yv = np.gradient(tsinterp[:, 1])

        if "tongue" not in feat_name:
            xv = xv - basederiv[0]
            yv = yv - basederiv[0]
            xv = fill_nearest_1d(xv)
            yv = fill_nearest_1d(yv)
        else:
            xv = np.where(np.isfinite(xv), xv, 0.0)
            yv = np.where(np.isfinite(yv), yv, 0.0)

        xvel[:, trial_idx] = xv
        yvel[:, trial_idx] = yv

    return xvel, yvel


def load_motion_energy(spec: SessionSpec) -> dict:
    me_mat = loadmat(spec.motion_energy_path, squeeze_me=True, struct_as_record=False)
    me = me_mat["me"]
    return {
        "data": np.atleast_1d(me.data),
        "moveThresh": float(me.moveThresh),
    }


def align_motion_energy(obj: dict, me: dict, align_times: np.ndarray, time_axis: np.ndarray) -> np.ndarray:
    vidshift = compute_vidshift(obj)
    view_dict = obj["traj"][0]
    n_trials = int(obj["bp"]["Ntrials"])
    aligned = np.full((time_axis.size, n_trials), np.nan, dtype=np.float64)

    for trial_idx in range(n_trials):
        frame_times = view_dict["frameTimes"][trial_idx]
        ts = view_dict["ts"][trial_idx]
        me_trial = np.asarray(me["data"][trial_idx], dtype=np.float64).reshape(-1)
        if frame_times is None:
            if ts is None:
                continue
            frame_times = (np.arange(np.asarray(ts).shape[0], dtype=np.float64) + 1.0) / 400.0
        else:
            frame_times = ensure_1d_numeric(frame_times)
        if frame_times.size == 0 or np.all(np.isnan(frame_times)):
            if ts is None:
                continue
            frame_times = (np.arange(np.asarray(ts).shape[0], dtype=np.float64) + 1.0) / 400.0

        n = min(frame_times.size, me_trial.size)
        frame_times = frame_times[:n]
        me_trial = me_trial[:n]
        valid = np.isfinite(frame_times) & np.isfinite(me_trial)
        if np.sum(valid) < 2:
            continue

        shifted_time = frame_times[valid] - vidshift - align_times[trial_idx]
        aligned[:, trial_idx] = np.interp(time_axis, shifted_time, me_trial[valid], left=np.nan, right=np.nan)
        aligned[:, trial_idx] = fill_nearest_1d(aligned[:, trial_idx])

    return aligned


def select_valid_trials(obj: dict) -> np.ndarray:
    bp = obj["bp"]
    early = ensure_1d_numeric(bp["early"]) != 0
    no = ensure_1d_numeric(bp["no"]) != 0
    hit = ensure_1d_numeric(bp["hit"]) != 0
    miss = ensure_1d_numeric(bp["miss"]) != 0
    stim_enable = ensure_1d_numeric(bp["stim"]["enable"]) != 0
    valid = (~early) & (~no) & (~stim_enable) & (hit | miss)
    return np.flatnonzero(valid)


def get_first_lick_direction(obj: dict, trial_idx: int, go_time: float) -> int | None:
    lick_l = event_list_to_array(obj["bp"]["ev"]["lickL"][trial_idx])
    lick_r = event_list_to_array(obj["bp"]["ev"]["lickR"][trial_idx])
    lick_l = lick_l[lick_l >= go_time]
    lick_r = lick_r[lick_r >= go_time]
    first_l = lick_l[0] if lick_l.size else math.inf
    first_r = lick_r[0] if lick_r.size else math.inf
    if math.isinf(first_l) and math.isinf(first_r):
        return None
    return 0 if first_l < first_r else 1


def keep_quality(quality: str) -> bool:
    if quality is None:
        quality = ""
    quality = str(quality).strip()
    return quality not in QUALITY_EXCLUDE


def bin_unit_spikes(
    trialtm: np.ndarray,
    trial_numbers_1based: np.ndarray,
    align_times: np.ndarray,
    kept_trials_0based: np.ndarray,
) -> np.ndarray:
    trialtm = ensure_1d_numeric(trialtm)
    trial_numbers_1based = np.asarray(trial_numbers_1based, dtype=np.int64).reshape(-1)

    local_map = np.full(align_times.size + 1, -1, dtype=np.int64)
    local_map[kept_trials_0based + 1] = np.arange(kept_trials_0based.size, dtype=np.int64)

    spike_local_trial = local_map[trial_numbers_1based]
    valid_spikes = spike_local_trial >= 0
    aligned = trialtm[valid_spikes] - align_times[trial_numbers_1based[valid_spikes] - 1]
    spike_local_trial = spike_local_trial[valid_spikes]
    bin_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
    valid_bins = (bin_idx >= 0) & (bin_idx < TIME_AXIS.size)
    aligned_counts = np.zeros((kept_trials_0based.size, TIME_AXIS.size), dtype=np.float64)
    np.add.at(aligned_counts, (spike_local_trial[valid_bins], bin_idx[valid_bins]), 1.0)
    rates = aligned_counts / DT
    rates = my_smooth(rates.T, SMOOTH_N, BCTYPE).T
    return rates


def mean_firing_rate_window(
    trialtm: np.ndarray,
    trial_numbers_1based: np.ndarray,
    align_times: np.ndarray,
) -> float:
    trialtm = ensure_1d_numeric(trialtm)
    trial_numbers_1based = np.asarray(trial_numbers_1based, dtype=np.int64).reshape(-1)
    aligned = trialtm - align_times[trial_numbers_1based - 1]
    in_window = (aligned >= TMIN) & (aligned < TMAX)
    return float(np.sum(in_window) / (align_times.size * (TMAX - TMIN)))


def convert_session(spec: SessionSpec, make_plot: bool = False) -> dict:
    session_start = time.perf_counter()
    print(f"[session] loading {spec.session_id}")
    obj = mat73.loadmat(spec.data_path)["obj"]
    me = load_motion_energy(spec)

    bp = obj["bp"]
    align_times = ensure_1d_numeric(bp["ev"][ALIGN_EVENT])
    valid_trials = select_valid_trials(obj)
    if valid_trials.size < 2:
        raise RuntimeError(f"{spec.session_id}: fewer than 2 valid trials after filtering")

    autowater = ensure_1d_numeric(bp["autowater"])
    hit = ensure_1d_numeric(bp["hit"])

    trial_labels = []
    kept_trials = []
    for trial_idx in valid_trials:
        lick_dir = get_first_lick_direction(obj, int(trial_idx), align_times[trial_idx])
        if lick_dir is None:
            continue
        kept_trials.append(int(trial_idx))
        trial_labels.append(
            (
                lick_dir,
                0 if autowater[trial_idx] != 0 else 1,
                1 if hit[trial_idx] != 0 else 0,
            )
        )

    kept_trials = np.asarray(kept_trials, dtype=np.int64)
    if kept_trials.size < 2:
        raise RuntimeError(f"{spec.session_id}: fewer than 2 trials with valid lick direction")

    print(f"[session] {spec.session_id} valid trials after all filters: {kept_trials.size}")

    clu = obj["clu"][spec.probe_index]
    if clu is None:
        raise RuntimeError(f"{spec.session_id}: selected probe {spec.probe_index + 1} has no neural data")

    unit_quality = [str(q).strip() if q is not None else "" for q in clu["quality"]]
    quality_keep = np.asarray([keep_quality(q) for q in unit_quality], dtype=bool)
    initial_units = int(quality_keep.sum())
    print(f"[session] {spec.session_id} units after quality filter: {initial_units}")

    neural_trials_by_unit = []
    kept_unit_indices = []
    for unit_idx, use_unit in enumerate(quality_keep):
        if not use_unit:
            continue
        mean_fr = mean_firing_rate_window(
            clu["trialtm"][unit_idx],
            np.asarray(clu["trial"][unit_idx], dtype=np.int64),
            align_times,
        )
        if mean_fr <= LOW_FR_HZ:
            continue
        trialdat = bin_unit_spikes(
            clu["trialtm"][unit_idx],
            np.asarray(clu["trial"][unit_idx], dtype=np.int64),
            align_times,
            kept_trials,
        )
        kept_unit_indices.append(unit_idx)
        neural_trials_by_unit.append(trialdat)

    if len(neural_trials_by_unit) == 0:
        raise RuntimeError(f"{spec.session_id}: no units survived filtering")

    neural_by_trial = np.stack(neural_trials_by_unit, axis=0).astype(np.float32)
    print(f"[session] {spec.session_id} units after FR filter: {neural_by_trial.shape[0]}")

    vidshift = compute_vidshift(obj)
    tongue_x, tongue_y = align_feature_positions(obj, 0, "tongue", align_times, TIME_AXIS, vidshift)
    tongue_visible = np.isfinite(tongue_x) & np.isfinite(tongue_y)
    tongue_vx, tongue_vy = compute_velocity(tongue_x, tongue_y, "tongue")
    tongue_speed = np.sqrt(tongue_vx**2 + tongue_vy**2)

    paw_speeds = []
    for paw_feat in ("top_paw", "bottom_paw"):
        paw_x, paw_y = align_feature_positions(obj, 1, paw_feat, align_times, TIME_AXIS, vidshift)
        paw_vx, paw_vy = compute_velocity(paw_x, paw_y, paw_feat)
        paw_speeds.append(np.sqrt(paw_vx**2 + paw_vy**2))
    paw_stack = np.stack(paw_speeds, axis=0)
    paw_counts = np.sum(np.isfinite(paw_stack), axis=0)
    paw_speed = np.full(paw_counts.shape, np.nan, dtype=np.float64)
    np.divide(np.nansum(paw_stack, axis=0), paw_counts, out=paw_speed, where=paw_counts > 0)

    motion_energy = align_motion_energy(obj, me, align_times, TIME_AXIS)

    tongue_visible_kept = tongue_visible[:, kept_trials]
    if np.any(tongue_visible_kept):
        tongue_threshold = float(np.nanpercentile(tongue_speed[:, kept_trials][tongue_visible_kept], 50))
    else:
        tongue_threshold = 0.0
    paw_threshold = float(np.nanpercentile(paw_speed[:, kept_trials], 50))
    me_threshold = float(np.nanpercentile(motion_energy[:, kept_trials], 50))

    session_neural = []
    session_input = []
    session_output = []

    for local_trial_idx, trial_idx in enumerate(kept_trials):
        lick_dir, context_label, outcome_label = trial_labels[local_trial_idx]
        neural_trial = neural_by_trial[:, local_trial_idx, :]
        time_input = TIME_AXIS[None, :].astype(np.float32)
        tongue_bin = ((tongue_speed[:, trial_idx] >= tongue_threshold) & tongue_visible[:, trial_idx]).astype(np.int64)
        output_trial = np.vstack(
            [
                np.full(TIME_AXIS.size, lick_dir, dtype=np.int64),
                np.full(TIME_AXIS.size, context_label, dtype=np.int64),
                np.full(TIME_AXIS.size, outcome_label, dtype=np.int64),
                tongue_bin,
                (paw_speed[:, trial_idx] >= paw_threshold).astype(np.int64),
                (motion_energy[:, trial_idx] >= me_threshold).astype(np.int64),
            ]
        )
        session_neural.append(neural_trial.astype(np.float32))
        session_input.append(time_input)
        session_output.append(output_trial)

    elapsed = time.perf_counter() - session_start
    print(
        f"[session] {spec.session_id} done in {elapsed:.2f}s | "
        f"trials={len(session_neural)} units={neural_by_trial.shape[0]}"
    )

    result = {
        "session_id": spec.session_id,
        "subject": spec.animal,
        "neural": session_neural,
        "input": session_input,
        "output": session_output,
        "brain_region_idx": np.zeros((neural_by_trial.shape[0],), dtype=np.int64),
        "thresholds": {
            "tongue_velocity": tongue_threshold,
            "paw_velocity": paw_threshold,
            "motion_energy": me_threshold,
        },
        "summary": {
            "n_trials": len(session_neural),
            "n_units": int(neural_by_trial.shape[0]),
            "n_trials_dr": int(sum(1 for _, c, _ in trial_labels if c == 1)),
            "n_trials_wc": int(sum(1 for _, c, _ in trial_labels if c == 0)),
            "n_correct": int(sum(1 for _, _, o in trial_labels if o == 1)),
            "n_incorrect": int(sum(1 for _, _, o in trial_labels if o == 0)),
            "n_left": int(sum(1 for l, _, _ in trial_labels if l == 0)),
            "n_right": int(sum(1 for l, _, _ in trial_labels if l == 1)),
        },
    }

    if make_plot:
        create_processing_plot(
            spec=spec,
            result=result,
            tongue_speed=tongue_speed[:, kept_trials],
            paw_speed=paw_speed[:, kept_trials],
            motion_energy=motion_energy[:, kept_trials],
        )

    return result


def create_processing_plot(
    spec: SessionSpec,
    result: dict,
    tongue_speed: np.ndarray,
    paw_speed: np.ndarray,
    motion_energy: np.ndarray,
) -> None:
    summary = result["summary"]
    thresholds = result["thresholds"]
    example_trial = 0

    fig, axes = plt.subplots(4, 2, figsize=(16, 14), constrained_layout=True)
    fig.suptitle(f"Processing Summary: {spec.session_id}", fontsize=16)

    neural = result["neural"][example_trial]
    ax = axes[0, 0]
    unit_sample = min(50, neural.shape[0])
    ax.imshow(neural[:unit_sample], aspect="auto", origin="lower", extent=[TMIN, TMAX, 0, unit_sample], cmap="viridis")
    ax.axvline(0.0, color="white", linestyle="--", linewidth=1)
    ax.set_title("Neural Activity")
    ax.set_xlabel("Time from go cue (s)")
    ax.set_ylabel("Sampled units")

    ax = axes[0, 1]
    ax.plot(TIME_AXIS, motion_energy[:, example_trial], label="motion energy", color="black")
    ax.plot(TIME_AXIS, tongue_speed[:, example_trial], label="tongue speed", color="tab:red")
    ax.plot(TIME_AXIS, paw_speed[:, example_trial], label="paw speed", color="tab:blue")
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Aligned Continuous Traces")
    ax.set_xlabel("Time from go cue (s)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.hist(tongue_speed.reshape(-1), bins=80, color="tab:red", alpha=0.8)
    ax.axvline(thresholds["tongue_velocity"], color="black", linestyle="--")
    ax.set_title("Tongue Speed Distribution")

    ax = axes[1, 1]
    ax.hist(paw_speed.reshape(-1), bins=80, color="tab:blue", alpha=0.8)
    ax.axvline(thresholds["paw_velocity"], color="black", linestyle="--")
    ax.set_title("Paw Speed Distribution")

    ax = axes[2, 0]
    ax.hist(motion_energy.reshape(-1), bins=80, color="gray", alpha=0.9)
    ax.axvline(thresholds["motion_energy"], color="tab:red", linestyle="--")
    ax.set_title("Motion Energy Distribution")

    ax = axes[2, 1]
    outputs = result["output"][example_trial]
    ax.step(TIME_AXIS, outputs[3] + 0.0, where="mid", label="tongue_bin")
    ax.step(TIME_AXIS, outputs[4] + 1.2, where="mid", label="paw_bin")
    ax.step(TIME_AXIS, outputs[5] + 2.4, where="mid", label="me_bin")
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(-0.3, 3.8)
    ax.set_title("Discretized Time-varying Outputs")
    ax.set_xlabel("Time from go cue (s)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[3, 0]
    cats = ["left", "right", "WC", "DR", "incorrect", "correct"]
    vals = [
        summary["n_left"],
        summary["n_right"],
        summary["n_trials_wc"],
        summary["n_trials_dr"],
        summary["n_incorrect"],
        summary["n_correct"],
    ]
    ax.bar(cats, vals, color=["tab:purple", "tab:green", "tab:orange", "tab:cyan", "tab:red", "tab:green"])
    ax.set_title("Session Trial Counts")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[3, 1]
    ax.text(
        0.0,
        1.0,
        "\n".join(
            [
                f"session_id: {spec.session_id}",
                f"units kept: {summary['n_units']}",
                f"trials kept: {summary['n_trials']}",
                f"tongue median: {thresholds['tongue_velocity']:.3f}",
                f"paw median: {thresholds['paw_velocity']:.3f}",
                f"motion-energy median: {thresholds['motion_energy']:.3f}",
                "alignment: bp.ev.goCue (WC uses stored goCue-equivalent event)",
                "trial filter: hit/miss, !early, !ignore, !stim",
                "unit filter: quality!='garbage/noisy/real?' and mean FR > 1 Hz",
            ]
        ),
        va="top",
        ha="left",
        family="monospace",
    )
    ax.axis("off")
    ax.set_title("Processing Notes")

    out_path = Path(f"processing_{spec.session_id}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def build_dataset(converted_sessions: list[dict]) -> dict:
    subjects = sorted({sess["subject"] for sess in converted_sessions})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}

    dataset = {
        "neural": [sess["neural"] for sess in converted_sessions],
        "input": [sess["input"] for sess in converted_sessions],
        "output": [sess["output"] for sess in converted_sessions],
        "subjects": subjects,
        "subject_idx": np.asarray([subject_to_idx[sess["subject"]] for sess in converted_sessions], dtype=np.int64),
        "brain_regions": ["ALM"],
        "brain_region_idx": [sess["brain_region_idx"] for sess in converted_sessions],
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
            ["left", "right"],
            ["WC", "DR"],
            ["incorrect", "correct"],
            ["below_session_median", "at_or_above_session_median"],
            ["below_session_median", "at_or_above_session_median"],
            ["below_session_median", "at_or_above_session_median"],
        ],
        "metadata": {
            "task_description": (
                "Two-context ALM electrophysiology dataset aligned to the stored goCue event "
                "(go cue in DR, goCue-equivalent water-presentation event in WC). "
                "Decoder predicts lick direction, behavioral context, outcome, and discretized "
                "tongue/paw/motion-energy movement variables from single-trial neural activity."
            ),
            "time_bin_size": 5.0,
            "temporal_alignment_event": "bp.ev.goCue",
            "off_start": TMIN,
            "off_end": TMAX,
            "n_sessions": len(converted_sessions),
            "source_session_ids": [sess["session_id"] for sess in converted_sessions],
            "neural_processing": {
                "representation": "smoothed single-trial firing rates",
                "smoothing_window": SMOOTH_N,
                "smoothing_boundary_condition": BCTYPE,
                "low_fr_threshold_hz": LOW_FR_HZ,
            },
            "trial_filter": {
                "exclude_early": True,
                "exclude_ignore": True,
                "exclude_stim": True,
                "include_hit": True,
                "include_miss": True,
            },
            "session_subset": "Figure 8 two-context ALM session list from reference code",
        },
    }
    return dataset


def print_dataset_summary(dataset: dict, converted_sessions: list[dict], elapsed: float) -> None:
    total_trials = sum(len(sess["neural"]) for sess in converted_sessions)
    total_units = sum(sess["brain_region_idx"].size for sess in converted_sessions)
    print("[summary] sessions:", len(converted_sessions))
    print("[summary] subjects:", len(dataset["subjects"]))
    print("[summary] total trials:", total_trials)
    print("[summary] total units:", total_units)
    print(f"[summary] elapsed seconds: {elapsed:.2f}")
    for sess in converted_sessions:
        s = sess["summary"]
        print(
            "[summary] "
            f"{sess['session_id']} | units={s['n_units']} trials={s['n_trials']} "
            f"DR={s['n_trials_dr']} WC={s['n_trials_wc']} "
            f"L={s['n_left']} R={s['n_right']} correct={s['n_correct']} incorrect={s['n_incorrect']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the Hasnain/Birnbaum et al. dataset into decoder format.")
    parser.add_argument("outpicklefile", type=Path, help="Output pickle file path.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all reference two-context sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only the first 2 sessions.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing summary plots for up to 2 sessions as processing_<session_id>.png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_sample = args.sample
    session_specs = CONTEXT_SESSION_SPECS[:2] if use_sample else CONTEXT_SESSION_SPECS

    total_start = time.perf_counter()
    converted_sessions = []
    for sess_idx, spec in enumerate(session_specs):
        make_plot = args.show_processing and sess_idx < 2
        converted_sessions.append(convert_session(spec, make_plot=make_plot))

    dataset = build_dataset(converted_sessions)
    with args.outpicklefile.open("wb") as f:
        pickle.dump(dataset, f)

    elapsed = time.perf_counter() - total_start
    print_dataset_summary(dataset, converted_sessions, elapsed)
    print(f"[write] saved {args.outpicklefile}")


if __name__ == "__main__":
    main()
