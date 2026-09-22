#!/usr/bin/env python3
"""Convert Li et al. ALM ephys + behavior sessions into decoder format.

This script follows the reference MATLAB pipeline where applicable:
- use the curated ALM session/probe lists from the reference loaders
- align neural activity to go cue
- bin spikes at 5 ms from -2.5 s to 2.5 s
- smooth binned rates with the same causal Gaussian kernel used by `mySmooth`
- exclude bad cluster qualities and low firing-rate units
- exclude early-lick and photostim trials

The target decoder additionally requires explicit categorical outputs for
movement and motion observables, so this script keeps missingness visible
instead of silently filling it into a numeric class.
"""

from __future__ import annotations

import argparse
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mat73
import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.io.matlab import mat_struct


DATA_ROOT = Path("/app/data")
EPHYS_DIR = DATA_ROOT / "Ephys_Behavior"
RAND_DIR = DATA_ROOT / "RandomizedDelay_Ephys_Behavior"

TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 200.0
SMOOTH = 15
LOW_FR_HZ = 1.0
MIN_UNITS_PER_SESSION = 10
RESPONSE_WINDOW_S = 3.0
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}

EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_CENTERS = (EDGES[:-1] + EDGES[1:]) / 2.0


@dataclass(frozen=True)
class SessionSpec:
    subject: str
    date: str
    cohort: str
    probes: tuple[int, ...]

    @property
    def session_id(self) -> str:
        return f"{self.subject}_{self.date}"

    @property
    def data_dir(self) -> Path:
        return EPHYS_DIR if self.cohort == "fixed" else RAND_DIR

    @property
    def data_path(self) -> Path:
        return self.data_dir / f"data_structure_{self.subject}_{self.date}.mat"

    @property
    def motion_path(self) -> Path:
        return self.data_dir / f"motionEnergy_{self.subject}_{self.date}.mat"


FIXED_SPECS = [
    SessionSpec("EKH1", "2021-08-07", "fixed", (2,)),
    SessionSpec("EKH3", "2021-08-11", "fixed", (2,)),
    SessionSpec("JEB13", "2022-09-13", "fixed", (2,)),
    SessionSpec("JEB13", "2022-09-14", "fixed", (2,)),
    SessionSpec("JEB13", "2022-09-21", "fixed", (1,)),
    SessionSpec("JEB13", "2022-09-24", "fixed", (1,)),
    SessionSpec("JEB13", "2022-09-25", "fixed", (1,)),
    SessionSpec("JEB14", "2022-08-22", "fixed", (1,)),
    SessionSpec("JEB14", "2022-08-23", "fixed", (1,)),
    SessionSpec("JEB14", "2022-08-24", "fixed", (1,)),
    SessionSpec("JEB14", "2022-08-25", "fixed", (1,)),
    SessionSpec("JEB15", "2022-07-26", "fixed", (1, 2)),
    SessionSpec("JEB15", "2022-07-27", "fixed", (1, 2)),
    SessionSpec("JEB15", "2022-07-28", "fixed", (1, 2)),
    SessionSpec("JEB15", "2022-07-29", "fixed", (2,)),
    SessionSpec("JEB19", "2023-04-21", "fixed", (1,)),
    SessionSpec("JEB19", "2023-04-20", "fixed", (1,)),
    SessionSpec("JEB19", "2023-04-19", "fixed", (1,)),
    SessionSpec("JEB19", "2023-04-18", "fixed", (1,)),
    SessionSpec("JEB6", "2021-04-18", "fixed", (2,)),
    SessionSpec("JEB7", "2021-04-29", "fixed", (1,)),
    SessionSpec("JEB7", "2021-04-30", "fixed", (1,)),
    SessionSpec("JGR2", "2021-11-16", "fixed", (1,)),
    SessionSpec("JGR2", "2021-11-17", "fixed", (1,)),
    SessionSpec("JGR3", "2021-11-18", "fixed", (1,)),
]

RANDOMIZED_SPECS = [
    SessionSpec("JEB11", "2022-05-10", "randomized", (1,)),
    SessionSpec("JEB11", "2022-05-11", "randomized", (1,)),
    SessionSpec("JEB12", "2022-05-12", "randomized", (1,)),
    SessionSpec("JEB12", "2022-05-13", "randomized", (1,)),
    SessionSpec("JEB23", "2023-10-10", "randomized", (1,)),
    SessionSpec("JEB23", "2023-10-11", "randomized", (1,)),
    SessionSpec("JEB23", "2023-10-12", "randomized", (1,)),
    SessionSpec("JEB23", "2023-10-13", "randomized", (1,)),
    SessionSpec("JEB23", "2023-10-18", "randomized", (1,)),
    SessionSpec("JEB23", "2023-10-19", "randomized", (1,)),
    SessionSpec("JEB23", "2023-10-21", "randomized", (1,)),
    SessionSpec("JEB24", "2023-10-23", "randomized", (1,)),
    SessionSpec("JEB24", "2023-10-24", "randomized", (1,)),
    SessionSpec("JEB24", "2023-10-25", "randomized", (1,)),
    SessionSpec("JEB24", "2023-10-26", "randomized", (1,)),
    SessionSpec("JEB24", "2023-10-27", "randomized", (1,)),
    SessionSpec("JEB24", "2023-10-31", "randomized", (1,)),
    SessionSpec("JEB24", "2023-11-02", "randomized", (1,)),
    SessionSpec("JEB24", "2023-11-03", "randomized", (1,)),
]

ALL_SPECS = FIXED_SPECS + RANDOMIZED_SPECS
SAMPLE_IDS = {"JEB13_2022-09-13", "JEB23_2023-10-18"}


def matlab_gausswin(n: int, alpha: float = 2.5) -> np.ndarray:
    if n <= 0:
        raise ValueError("Kernel length must be positive.")
    if n == 1:
        return np.ones(1, dtype=np.float64)
    idx = np.arange(n, dtype=np.float64) - (n - 1.0) / 2.0
    sigma = (n - 1.0) / (2.0 * alpha)
    return np.exp(-0.5 * (idx / sigma) ** 2)


def my_smooth(x: np.ndarray, n: int, bctype: str = "none") -> np.ndarray:
    """Python port of the reference MATLAB `mySmooth`.

    Important details preserved:
    - operate on the first dimension only
    - prepend the first N samples for "reflect" boundary handling
    - use a causalized Gaussian kernel
    """
    x = np.asarray(x, dtype=np.float64)
    one_dim = x.ndim == 1
    if one_dim:
        x = x[:, None]

    if n in (0, 1):
        out = x.copy()
        return out[:, 0] if one_dim else out

    if bctype.lower() == "reflect":
        x_filt = np.concatenate([x[:n, :], x], axis=0)
        trim = n
    elif bctype.lower() == "zeropad":
        x_filt = np.concatenate([np.zeros((n, x.shape[1]), dtype=x.dtype), x], axis=0)
        trim = n
    else:
        x_filt = x
        trim = 0

    kern = matlab_gausswin(n)
    kern[: math.floor(kern.size / 2)] = 0.0
    kern /= kern.sum()

    out = np.empty_like(x_filt, dtype=np.float64)
    for col in range(x_filt.shape[1]):
        out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")
    out = out[trim:, :]
    return out[:, 0] if one_dim else out


def convert_mat_struct(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: convert_mat_struct(val) for key, val in obj.items()}
    if isinstance(obj, mat_struct):
        return {field: convert_mat_struct(getattr(obj, field)) for field in obj._fieldnames}
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return convert_mat_struct(obj.item())
        if obj.dtype == object:
            return [convert_mat_struct(item) for item in obj.tolist()]
        return np.asarray(obj)
    if isinstance(obj, list):
        return [convert_mat_struct(item) for item in obj]
    return obj


def load_any_mat(path: Path) -> dict[str, Any]:
    if h5py.is_hdf5(path):
        return mat73.loadmat(str(path))
    return convert_mat_struct(loadmat(path, squeeze_me=True, struct_as_record=False))


def normalize_struct_list(items: list[dict[str, Any]]) -> dict[str, list[Any]]:
    if not items:
        return {}
    keys = list(items[0].keys())
    out = {key: [] for key in keys}
    for item in items:
        for key in keys:
            out[key].append(item.get(key, np.nan))
    return out


def normalize_feat_names(names: Any) -> list[str]:
    out: list[str] = []
    if isinstance(names, np.ndarray):
        names = names.tolist()
    for item in names:
        if isinstance(item, np.ndarray):
            if item.ndim == 0:
                out.append(str(item.item()))
            else:
                flat = item.tolist()
                if isinstance(flat, list) and flat:
                    out.append(str(flat[0]))
                else:
                    out.append(str(flat))
        elif isinstance(item, list):
            out.append(str(item[0]))
        else:
            out.append(str(item))
    return out


def normalize_view(view: Any) -> dict[str, list[Any]]:
    if isinstance(view, dict):
        out = {}
        for key, val in view.items():
            if isinstance(val, np.ndarray):
                val = val.tolist()
            if not isinstance(val, list):
                val = [val]
            out[key] = val
        if "featNames" in out:
            out["featNames"] = [normalize_feat_names(names) for names in out["featNames"]]
        return out
    if isinstance(view, list):
        out = normalize_struct_list(view)
        if "featNames" in out:
            out["featNames"] = [normalize_feat_names(names) for names in out["featNames"]]
        return out
    raise TypeError(f"Unsupported trajectory view type: {type(view)}")


def normalize_probe(probe: Any) -> dict[str, list[Any]]:
    if probe is None:
        return {}
    if isinstance(probe, dict):
        out = {}
        for key, val in probe.items():
            if isinstance(val, np.ndarray):
                val = val.tolist()
            if not isinstance(val, list):
                val = [val]
            out[key] = val
        return out
    if isinstance(probe, list):
        return normalize_struct_list(probe)
    raise TypeError(f"Unsupported probe type: {type(probe)}")


def normalize_obj(payload: dict[str, Any]) -> dict[str, Any]:
    obj = payload["obj"] if "obj" in payload else payload
    obj["bp"] = obj.get("bp", {})
    obj["traj"] = [normalize_view(view) for view in obj.get("traj", [])]

    clu = obj.get("clu", [])
    if isinstance(clu, dict):
        clu = [clu]
    elif isinstance(clu, np.ndarray):
        clu = clu.tolist()
    if clu and isinstance(clu[0], dict) and isinstance(clu[0].get("tm"), np.ndarray):
        clu = [normalize_probe(clu)]
    else:
        clu = [normalize_probe(probe) if probe is not None else {} for probe in clu]
    obj["clu"] = clu
    return obj


def as_1d_numeric(x: Any, dtype=np.float64) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim == 0:
        arr = arr[None]
    return arr.reshape(-1)


def to_bool_array(x: Any, ntrials: int) -> np.ndarray:
    if x is None:
        return np.zeros(ntrials, dtype=bool)
    arr = as_1d_numeric(x, dtype=np.float64)
    if arr.size != ntrials:
        arr = np.broadcast_to(arr, (ntrials,))
    arr = np.nan_to_num(arr, nan=0.0)
    return arr.astype(bool)


def get_bp_array(bp: dict[str, Any], key: str, ntrials: int, dtype=np.float64) -> np.ndarray:
    if key not in bp:
        if np.issubdtype(np.dtype(dtype), np.bool_):
            return np.zeros(ntrials, dtype=bool)
        return np.full(ntrials, np.nan, dtype=dtype)
    arr = as_1d_numeric(bp[key], dtype=np.float64)
    if arr.size != ntrials:
        arr = np.broadcast_to(arr, (ntrials,))
    if np.issubdtype(np.dtype(dtype), np.bool_):
        return np.nan_to_num(arr, nan=0.0).astype(bool)
    return arr.astype(dtype)


def get_stim_enable(bp: dict[str, Any], ntrials: int) -> np.ndarray:
    stim = bp.get("stim")
    if not isinstance(stim, dict):
        return np.zeros(ntrials, dtype=bool)
    return to_bool_array(stim.get("enable"), ntrials)


def ensure_event_list(events: Any, ntrials: int) -> list[np.ndarray]:
    if isinstance(events, np.ndarray) and events.dtype != object:
        events = events.tolist()
    if not isinstance(events, list):
        events = [events] * ntrials
    if len(events) != ntrials:
        raise ValueError(f"Event list length mismatch: expected {ntrials}, got {len(events)}")
    out: list[np.ndarray] = []
    for item in events:
        if item is None:
            out.append(np.empty(0, dtype=np.float64))
            continue
        arr = np.asarray(item, dtype=np.float64)
        if arr.ndim == 0:
            if np.isnan(arr):
                out.append(np.empty(0, dtype=np.float64))
            else:
                out.append(arr.reshape(1))
        else:
            out.append(arr.reshape(-1))
    return out


def get_vidshift(obj: dict[str, Any]) -> float:
    bp = obj["bp"]
    bit_start = get_bp_array(bp["ev"], "bitStart", int(bp["Ntrials"]), dtype=np.float64)
    if "sglx" not in obj or not isinstance(obj["sglx"], dict):
        return 0.0
    sglx = obj["sglx"]
    bitcode = sglx.get("bitcode")
    fs = sglx.get("fs")
    if not isinstance(bitcode, dict) or fs is None:
        return 0.0
    bit_file_offset = as_1d_numeric(bitcode.get("bitstart"), dtype=np.float64)
    fs_val = float(np.asarray(fs).reshape(-1)[0])
    return float(np.nanmedian(bit_file_offset) / fs_val - np.nanmedian(bit_start))


def find_feat_index(view: dict[str, list[Any]], feat_name: str) -> int:
    feat_names_all = view.get("featNames", [])
    for names in feat_names_all:
        if not names:
            continue
        lowered = [str(name).lower() for name in names]
        if feat_name.lower() in lowered:
            return lowered.index(feat_name.lower())
    raise KeyError(f"Could not find feature {feat_name}")


def nearest_fill(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(x))
    if valid.size == 0:
        return x.copy()
    fn = interp1d(
        valid,
        x[valid],
        kind="nearest",
        bounds_error=False,
        fill_value=(x[valid[0]], x[valid[-1]]),
        assume_sorted=True,
    )
    return fn(np.arange(x.size))


def interp_feature(frame_times: np.ndarray, values: np.ndarray, taxis: np.ndarray) -> np.ndarray:
    out = np.full((taxis.size, values.shape[1]), np.nan, dtype=np.float64)
    for col in range(values.shape[1]):
        valid = np.isfinite(frame_times) & np.isfinite(values[:, col])
        if valid.sum() < 2:
            continue
        out[:, col] = np.interp(
            taxis,
            frame_times[valid],
            values[valid, col],
            left=np.nan,
            right=np.nan,
        )
    return out


def extract_feature_traces(
    obj: dict[str, Any],
    feat_name: str,
    view_idx: int,
    go_cue: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    taxis = TIME_CENTERS
    ntrials = int(obj["bp"]["Ntrials"])
    view = obj["traj"][view_idx]
    feat_idx = find_feat_index(view, feat_name)
    vidshift = get_vidshift(obj)

    xpos = np.full((taxis.size, ntrials), np.nan, dtype=np.float64)
    ypos = np.full_like(xpos, np.nan)
    visible = np.zeros((taxis.size, ntrials), dtype=bool)

    for trix in range(ntrials):
        ndropped = view.get("NdroppedFrames", [np.nan] * ntrials)[trix]
        ndropped_arr = np.asarray(ndropped, dtype=np.float64)
        if ndropped_arr.ndim == 0 and np.isnan(ndropped_arr):
            continue

        ts = np.asarray(view["ts"][trix], dtype=np.float64)
        if ts.ndim != 3 or feat_idx >= ts.shape[2]:
            continue
        feat_xy = ts[:, :2, feat_idx]

        frame_times = view.get("frameTimes", [None] * ntrials)[trix]
        if frame_times is None:
            frame_times = np.arange(1, feat_xy.shape[0] + 1, dtype=np.float64) / 400.0
        else:
            frame_times = np.asarray(frame_times, dtype=np.float64).reshape(-1)
            if frame_times.size == 1 and np.isnan(frame_times[0]):
                frame_times = np.arange(1, feat_xy.shape[0] + 1, dtype=np.float64) / 400.0

        aligned_times = frame_times - vidshift - go_cue[trix]
        interp_xy = interp_feature(aligned_times, feat_xy, taxis)
        visible[:, trix] = np.isfinite(interp_xy[:, 0]) & np.isfinite(interp_xy[:, 1])
        xpos[:, trix] = interp_xy[:, 0]
        ypos[:, trix] = interp_xy[:, 1]

        if "tongue" not in feat_name.lower():
            xpos[:, trix] = nearest_fill(xpos[:, trix])
            ypos[:, trix] = nearest_fill(ypos[:, trix])

    return xpos, ypos, visible


def extract_velocity(
    xpos: np.ndarray,
    ypos: np.ndarray,
    feat_name: str,
    visible: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ntime, ntrials = xpos.shape
    xvel = np.full((ntime, ntrials), np.nan, dtype=np.float64)
    yvel = np.full_like(xvel, np.nan)
    if visible is None:
        visible = np.isfinite(xpos) & np.isfinite(ypos)

    for trix in range(ntrials):
        trial_xy = np.column_stack([xpos[:, trix], ypos[:, trix]])
        diffs = np.diff(trial_xy, axis=0)
        finite_diff = np.isfinite(diffs)
        if not finite_diff.any():
            basederiv = np.zeros(2, dtype=np.float64)
        else:
            basederiv = np.array(
                [
                    np.nanmedian(diffs[:, 0]) if np.any(np.isfinite(diffs[:, 0])) else 0.0,
                    np.nanmedian(diffs[:, 1]) if np.any(np.isfinite(diffs[:, 1])) else 0.0,
                ],
                dtype=np.float64,
            )

        xv = np.gradient(trial_xy[:, 0])
        yv = np.gradient(trial_xy[:, 1])
        if "tongue" not in feat_name.lower():
            xv = xv - basederiv[0]
            yv = yv - basederiv[0]
            xv = nearest_fill(xv)
            yv = nearest_fill(yv)
        else:
            xv = xv.copy()
            yv = yv.copy()
            xv[~np.isfinite(xv)] = 0.0
            yv[~np.isfinite(yv)] = 0.0
        xvel[:, trix] = xv
        yvel[:, trix] = yv

    return xvel, yvel, visible


def compute_speed_categories(
    speed: np.ndarray,
    visible: np.ndarray,
) -> tuple[np.ndarray, float]:
    out = np.full(speed.shape, 2, dtype=np.int8)
    valid_speed = speed[visible & np.isfinite(speed)]
    if valid_speed.size == 0:
        return out, float("nan")
    threshold = float(np.nanpercentile(valid_speed, 50.0))
    out[visible] = (speed[visible] >= threshold).astype(np.int8)
    return out, threshold


def load_motion_energy(spec: SessionSpec) -> dict[str, Any] | None:
    if not spec.motion_path.exists():
        return None
    payload = load_any_mat(spec.motion_path)
    me = payload.get("me")
    if me is None:
        return None
    if isinstance(me, dict):
        return me
    if isinstance(me, list):
        return {"data": me, "moveThresh": np.nan}
    if isinstance(me, np.ndarray):
        return {"data": me.tolist(), "moveThresh": np.nan}
    raise TypeError(f"Unsupported motion energy structure in {spec.motion_path}: {type(me)}")


def align_motion_energy(
    obj: dict[str, Any],
    spec: SessionSpec,
    go_cue: np.ndarray,
) -> np.ndarray:
    ntrials = int(obj["bp"]["Ntrials"])
    me = load_motion_energy(spec)
    aligned = np.full((TIME_CENTERS.size, ntrials), np.nan, dtype=np.float64)
    if me is None or "data" not in me:
        return aligned

    me_data = me["data"]
    if isinstance(me_data, dict) and "data" in me_data:
        me_data = me_data["data"]
    if isinstance(me_data, np.ndarray):
        me_data = me_data.tolist()

    vidshift = get_vidshift(obj)
    view0 = obj["traj"][0] if obj["traj"] else {}

    for trix in range(ntrials):
        if trix >= len(me_data):
            continue
        trace = np.asarray(me_data[trix], dtype=np.float64).reshape(-1)
        if trace.size == 0:
            continue
        frame_times = view0.get("frameTimes", [None] * ntrials)[trix] if view0 else None
        if frame_times is None:
            frame_times = np.arange(1, trace.size + 1, dtype=np.float64) / 400.0
        else:
            frame_times = np.asarray(frame_times, dtype=np.float64).reshape(-1)
            if frame_times.size == 1 and np.isnan(frame_times[0]):
                frame_times = np.arange(1, trace.size + 1, dtype=np.float64) / 400.0
        valid = np.isfinite(frame_times) & np.isfinite(trace)
        if valid.sum() < 2:
            continue
        aligned[:, trix] = np.interp(
            TIME_CENTERS,
            frame_times[valid] - vidshift - go_cue[trix],
            trace[valid],
            left=np.nan,
            right=np.nan,
        )
    return aligned


def first_post_go_lick_direction(
    lick_l: list[np.ndarray],
    lick_r: list[np.ndarray],
    go_cue: np.ndarray,
) -> np.ndarray:
    direction = np.full(go_cue.shape, 2, dtype=np.int8)
    for trix in range(go_cue.size):
        left = lick_l[trix]
        right = lick_r[trix]
        left = left[(left >= go_cue[trix]) & (left < go_cue[trix] + RESPONSE_WINDOW_S)]
        right = right[(right >= go_cue[trix]) & (right < go_cue[trix] + RESPONSE_WINDOW_S)]
        left_first = left[0] if left.size else np.inf
        right_first = right[0] if right.size else np.inf
        if not np.isfinite(left_first) and not np.isfinite(right_first):
            direction[trix] = 2
        elif left_first <= right_first:
            direction[trix] = 0
        else:
            direction[trix] = 1
    return direction


def build_valid_mask(bp: dict[str, Any], ntrials: int, go_cue: np.ndarray) -> np.ndarray:
    early = get_bp_array(bp, "early", ntrials, dtype=bool)
    stim = get_stim_enable(bp, ntrials)
    hit = get_bp_array(bp, "hit", ntrials, dtype=bool)
    miss = get_bp_array(bp, "miss", ntrials, dtype=bool)
    no = get_bp_array(bp, "no", ntrials, dtype=bool)
    outcome = hit | miss | no
    return (~early) & (~stim) & outcome & np.isfinite(go_cue)


def max_recorded_trial(obj: dict[str, Any], probes: tuple[int, ...]) -> int:
    max_trial = 0
    for probe_num in probes:
        probe_idx = probe_num - 1
        if probe_idx < 0 or probe_idx >= len(obj["clu"]):
            continue
        probe = obj["clu"][probe_idx]
        if not probe or "trial" not in probe:
            continue
        for trial_arr in probe["trial"]:
            arr = as_1d_numeric(trial_arr, dtype=np.float64)
            if arr.size:
                max_trial = max(max_trial, int(np.nanmax(arr)))
    return max_trial


def build_condition_positions(bp: dict[str, Any], valid_mask: np.ndarray) -> list[np.ndarray]:
    ntrials = valid_mask.size
    hit = get_bp_array(bp, "hit", ntrials, dtype=bool)
    right = get_bp_array(bp, "R", ntrials, dtype=bool)
    left = get_bp_array(bp, "L", ntrials, dtype=bool)
    autowater = get_bp_array(bp, "autowater", ntrials, dtype=bool)
    conds = [
        right & hit & (~autowater),
        left & hit & (~autowater),
        right & hit & autowater,
        left & hit & autowater,
    ]
    lookup = np.full(ntrials, -1, dtype=np.int32)
    lookup[np.flatnonzero(valid_mask)] = np.arange(valid_mask.sum(), dtype=np.int32)
    positions = []
    for cond in conds:
        pos = lookup[np.flatnonzero(valid_mask & cond)]
        positions.append(pos[pos >= 0])
    return positions


def cluster_good_mask(qualities: list[Any]) -> np.ndarray:
    labels = [str(q).strip().lower() if q is not None else "" for q in qualities]
    return np.array([label not in QUALITY_EXCLUDE for label in labels], dtype=bool)


def bin_cluster_rates(
    trial_ids: np.ndarray,
    trial_times: np.ndarray,
    go_cue: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    n_valid = int(valid_mask.sum())
    counts = np.zeros((TIME_CENTERS.size, n_valid), dtype=np.float64)
    if n_valid == 0 or trial_ids.size == 0:
        return counts.astype(np.float32)

    original_idx = trial_ids.astype(np.int64) - 1
    in_range = (original_idx >= 0) & (original_idx < go_cue.size)
    original_idx = original_idx[in_range]
    trial_times = trial_times[in_range]

    trial_lookup = np.full(valid_mask.size, -1, dtype=np.int32)
    trial_lookup[np.flatnonzero(valid_mask)] = np.arange(n_valid, dtype=np.int32)
    kept_pos = trial_lookup[original_idx]

    aligned = trial_times - go_cue[original_idx]
    keep = (
        (kept_pos >= 0)
        & np.isfinite(aligned)
        & (aligned >= TMIN)
        & (aligned < TMAX)
    )
    if not np.any(keep):
        return counts.astype(np.float32)

    kept_pos = kept_pos[keep]
    aligned = aligned[keep]
    bin_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
    flat_idx = bin_idx * n_valid + kept_pos
    counts_flat = np.bincount(flat_idx, minlength=TIME_CENTERS.size * n_valid)
    counts = counts_flat.reshape(TIME_CENTERS.size, n_valid).astype(np.float64)
    rates = my_smooth(counts / DT, SMOOTH, "reflect")
    return rates.astype(np.float32)


def process_probe(
    probe: dict[str, list[Any]],
    go_cue: np.ndarray,
    valid_mask: np.ndarray,
    condition_positions: list[np.ndarray],
) -> list[np.ndarray]:
    if not probe or "trial" not in probe:
        return []
    qualities = probe.get("quality", [""] * len(probe["trial"]))
    good_clusters = cluster_good_mask(qualities)
    kept_rates: list[np.ndarray] = []

    for clu_idx in np.flatnonzero(good_clusters):
        trial_ids = as_1d_numeric(probe["trial"][clu_idx], dtype=np.float64)
        trial_times = as_1d_numeric(probe["trialtm"][clu_idx], dtype=np.float64)
        rates = bin_cluster_rates(trial_ids, trial_times, go_cue, valid_mask)

        psth_stack = []
        for cond_pos in condition_positions:
            if cond_pos.size == 0:
                psth_stack.append(np.zeros(TIME_CENTERS.size, dtype=np.float32))
            else:
                psth_stack.append(rates[:, cond_pos].mean(axis=1))
        mean_fr = float(np.mean(np.stack(psth_stack, axis=1)))
        if mean_fr > LOW_FR_HZ:
            kept_rates.append(rates)
    return kept_rates


def build_session_neural(
    kept_rates: list[np.ndarray],
    ntrials: int,
) -> list[np.ndarray]:
    nneurons = len(kept_rates)
    if nneurons == 0:
        return []
    session_trials: list[np.ndarray] = []
    for trix in range(ntrials):
        trial_mat = np.empty((nneurons, TIME_CENTERS.size), dtype=np.float32)
        for neuron_idx, rates in enumerate(kept_rates):
            trial_mat[neuron_idx, :] = rates[:, trix]
        session_trials.append(trial_mat)
    return session_trials


def repeat_per_trial(values: np.ndarray, ntime: int) -> np.ndarray:
    return np.repeat(values[:, None], ntime, axis=1)


def choose_trials(matrix: np.ndarray, valid_idx: np.ndarray) -> np.ndarray:
    return matrix[:, valid_idx]


def process_session(spec: SessionSpec, show_processing: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    start = time.time()
    obj = normalize_obj(load_any_mat(spec.data_path))
    bp = obj["bp"]
    ntrials = int(float(bp["Ntrials"]))

    go_cue = get_bp_array(bp["ev"], "goCue", ntrials, dtype=np.float64)
    valid_mask = build_valid_mask(bp, ntrials, go_cue)
    last_neural_trial = max_recorded_trial(obj, spec.probes)
    if last_neural_trial > 0 and last_neural_trial < ntrials:
        valid_mask &= (np.arange(1, ntrials + 1) <= last_neural_trial)
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size < 2:
        raise RuntimeError(f"{spec.session_id}: fewer than 2 valid trials after filtering.")

    condition_positions = build_condition_positions(bp, valid_mask)

    kept_rates: list[np.ndarray] = []
    for probe_num in spec.probes:
        probe_idx = probe_num - 1
        if probe_idx < 0 or probe_idx >= len(obj["clu"]):
            continue
        kept_rates.extend(process_probe(obj["clu"][probe_idx], go_cue, valid_mask, condition_positions))

    if len(kept_rates) < MIN_UNITS_PER_SESSION:
        raise RuntimeError(
            f"{spec.session_id}: only {len(kept_rates)} units after filtering; expected at least {MIN_UNITS_PER_SESSION}."
        )

    session_neural = build_session_neural(kept_rates, valid_idx.size)

    lick_l = ensure_event_list(bp["ev"].get("lickL", []), ntrials)
    lick_r = ensure_event_list(bp["ev"].get("lickR", []), ntrials)
    lick_direction = first_post_go_lick_direction(lick_l, lick_r, go_cue)[valid_idx]

    autowater = get_bp_array(bp, "autowater", ntrials, dtype=bool)[valid_idx]
    context = np.where(autowater, 0, 1).astype(np.int8)

    hit = get_bp_array(bp, "hit", ntrials, dtype=bool)[valid_idx]
    miss = get_bp_array(bp, "miss", ntrials, dtype=bool)[valid_idx]
    no = get_bp_array(bp, "no", ntrials, dtype=bool)[valid_idx]
    outcome = np.full(valid_idx.size, 2, dtype=np.int8)
    outcome[miss] = 0
    outcome[hit] = 1
    outcome[no] = 2

    motion = align_motion_energy(obj, spec, go_cue)
    motion = choose_trials(motion, valid_idx)
    motion_visible = np.isfinite(motion)
    motion_cat, motion_thresh = compute_speed_categories(motion, motion_visible)

    tongue_x, tongue_y, tongue_visible = extract_feature_traces(obj, "tongue", 0, go_cue)
    tongue_x = choose_trials(tongue_x, valid_idx)
    tongue_y = choose_trials(tongue_y, valid_idx)
    tongue_visible = choose_trials(tongue_visible, valid_idx).astype(bool)
    tongue_vx, tongue_vy, tongue_visible = extract_velocity(tongue_x, tongue_y, "tongue", tongue_visible)
    tongue_speed = np.sqrt(tongue_vx**2 + tongue_vy**2)
    tongue_cat, tongue_thresh = compute_speed_categories(tongue_speed, tongue_visible)

    paw_features = []
    paw_visible_masks = []
    for feat_name in ("top_paw", "bottom_paw"):
        try:
            paw_x, paw_y, paw_visible = extract_feature_traces(obj, feat_name, 1, go_cue)
        except KeyError:
            continue
        paw_x = choose_trials(paw_x, valid_idx)
        paw_y = choose_trials(paw_y, valid_idx)
        paw_visible = choose_trials(paw_visible, valid_idx).astype(bool)
        paw_vx, paw_vy, paw_visible = extract_velocity(paw_x, paw_y, feat_name, paw_visible)
        paw_features.append(np.sqrt(paw_vx**2 + paw_vy**2))
        paw_visible_masks.append(paw_visible)
    if paw_features:
        paw_stack = np.stack(paw_features, axis=2)
        paw_vis_stack = np.stack(paw_visible_masks, axis=2)
        paw_visible = paw_vis_stack.any(axis=2)
        paw_weighted = np.where(paw_vis_stack, paw_stack, 0.0)
        paw_counts = paw_vis_stack.sum(axis=2)
        paw_speed = np.divide(
            paw_weighted.sum(axis=2),
            paw_counts,
            out=np.full(paw_visible.shape, np.nan, dtype=np.float64),
            where=paw_counts > 0,
        )
        paw_cat, paw_thresh = compute_speed_categories(paw_speed, paw_visible)
    else:
        paw_speed = np.full((TIME_CENTERS.size, valid_idx.size), np.nan, dtype=np.float64)
        paw_visible = np.zeros_like(paw_speed, dtype=bool)
        paw_cat = np.full_like(paw_speed, 2, dtype=np.int8)
        paw_thresh = float("nan")

    session_input = [TIME_CENTERS[None, :].astype(np.float32).copy() for _ in range(valid_idx.size)]

    session_output: list[np.ndarray] = []
    for trix in range(valid_idx.size):
        per_trial = np.vstack(
            [
                np.full(TIME_CENTERS.size, lick_direction[trix], dtype=np.int8),
                np.full(TIME_CENTERS.size, context[trix], dtype=np.int8),
                np.full(TIME_CENTERS.size, outcome[trix], dtype=np.int8),
                tongue_cat[:, trix],
                paw_cat[:, trix],
                motion_cat[:, trix],
            ]
        )
        session_output.append(per_trial)

    session_info = {
        "session_id": spec.session_id,
        "subject": spec.subject,
        "date": spec.date,
        "cohort": spec.cohort,
        "n_trials_raw": ntrials,
        "n_trials_kept": int(valid_idx.size),
        "n_neurons_kept": int(len(kept_rates)),
        "tongue_threshold": tongue_thresh,
        "paw_threshold": paw_thresh,
        "motion_threshold": motion_thresh,
        "runtime_sec": time.time() - start,
        "last_neural_trial_1based": int(last_neural_trial),
        "valid_trial_indices_1based": (valid_idx + 1).tolist(),
    }

    plot_payload = {
        "session_id": spec.session_id,
        "context": context,
        "outcome": outcome,
        "lick_direction": lick_direction,
        "time": TIME_CENTERS,
        "neural_mean": np.stack([trial.mean(axis=0) for trial in session_neural], axis=0),
        "tongue_speed": tongue_speed.T,
        "paw_speed": paw_speed.T,
        "motion": motion.T,
        "tongue_cat": tongue_cat.T,
        "paw_cat": paw_cat.T,
        "motion_cat": motion_cat.T,
    }
    if show_processing:
        plot_processing(plot_payload)

    session_data = {
        "neural": session_neural,
        "input": session_input,
        "output": session_output,
        "brain_region_idx": np.zeros(len(kept_rates), dtype=np.int64),
        "session_info": session_info,
    }
    return session_data, plot_payload


def plot_processing(payload: dict[str, Any]) -> None:
    session_id = payload["session_id"]
    time_axis = payload["time"]
    ntrials = payload["neural_mean"].shape[0]
    trial_pick = min(max(ntrials // 2, 0), ntrials - 1)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(f"Processing Summary: {session_id}")

    axes[0, 0].imshow(payload["neural_mean"], aspect="auto", cmap="magma", extent=[time_axis[0], time_axis[-1], ntrials, 1])
    axes[0, 0].set_title("Mean Neural Rate per Trial")
    axes[0, 0].set_xlabel("Time from go cue (s)")
    axes[0, 0].set_ylabel("Kept trials")

    axes[0, 1].hist(payload["context"], bins=np.arange(-0.5, 2.5, 1.0), alpha=0.7, label="context")
    axes[0, 1].hist(payload["outcome"], bins=np.arange(-0.5, 3.5, 1.0), alpha=0.7, label="outcome")
    axes[0, 1].hist(payload["lick_direction"], bins=np.arange(-0.5, 3.5, 1.0), alpha=0.7, label="lick")
    axes[0, 1].set_title("Per-trial Output Distribution")
    axes[0, 1].legend()

    axes[1, 0].plot(time_axis, payload["tongue_speed"][trial_pick], label="tongue speed")
    axes[1, 0].step(time_axis, payload["tongue_cat"][trial_pick], where="mid", label="tongue class")
    axes[1, 0].set_title(f"Tongue Speed / Class, trial {trial_pick + 1}")
    axes[1, 0].set_xlabel("Time from go cue (s)")
    axes[1, 0].legend()

    axes[1, 1].plot(time_axis, payload["paw_speed"][trial_pick], label="paw speed")
    axes[1, 1].step(time_axis, payload["paw_cat"][trial_pick], where="mid", label="paw class")
    axes[1, 1].set_title(f"Paw Speed / Class, trial {trial_pick + 1}")
    axes[1, 1].set_xlabel("Time from go cue (s)")
    axes[1, 1].legend()

    axes[2, 0].plot(time_axis, payload["motion"][trial_pick], label="motion energy")
    axes[2, 0].step(time_axis, payload["motion_cat"][trial_pick], where="mid", label="motion class")
    axes[2, 0].set_title(f"Motion Energy / Class, trial {trial_pick + 1}")
    axes[2, 0].set_xlabel("Time from go cue (s)")
    axes[2, 0].legend()

    mean_trace = payload["neural_mean"][trial_pick]
    axes[2, 1].plot(time_axis, mean_trace, color="black")
    axes[2, 1].axvline(0.0, color="red", linestyle="--", linewidth=1)
    axes[2, 1].set_title(f"Example Trial Mean Neural Activity, trial {trial_pick + 1}")
    axes[2, 1].set_xlabel("Time from go cue (s)")

    out_path = Path(f"/app/processing_{session_id}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_dataset(session_specs: list[SessionSpec], show_processing: bool = False) -> dict[str, Any]:
    neural: list[list[np.ndarray]] = []
    decoder_input: list[list[np.ndarray]] = []
    decoder_output: list[list[np.ndarray]] = []
    brain_region_idx: list[np.ndarray] = []
    session_infos: list[dict[str, Any]] = []
    subjects: list[str] = []
    subject_lookup: dict[str, int] = {}
    subject_idx: list[int] = []

    for sess_idx, spec in enumerate(session_specs, start=1):
        print(f"[{sess_idx:02d}/{len(session_specs):02d}] Processing {spec.session_id}")
        session_start = time.time()
        session_data, _ = process_session(spec, show_processing=show_processing and sess_idx <= 2)
        neural.append(session_data["neural"])
        decoder_input.append(session_data["input"])
        decoder_output.append(session_data["output"])
        brain_region_idx.append(session_data["brain_region_idx"])
        session_infos.append(session_data["session_info"])
        if spec.subject not in subject_lookup:
            subject_lookup[spec.subject] = len(subjects)
            subjects.append(spec.subject)
        subject_idx.append(subject_lookup[spec.subject])
        print(
            f"    kept {session_data['session_info']['n_neurons_kept']} units, "
            f"{session_data['session_info']['n_trials_kept']} trials "
            f"in {time.time() - session_start:.1f}s"
        )

    data = {
        "neural": neural,
        "input": decoder_input,
        "output": decoder_output,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": ["ALM"],
        "brain_region_idx": brain_region_idx,
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
            ["low", "high", "not_visible"],
            ["low", "high", "not_visible"],
            ["low", "high", "no_video"],
        ],
        "metadata": {
            "task_description": (
                "ALM electrophysiology during fixed-delay and randomized-delay "
                "direction-report tasks with water-cued context trials."
            ),
            "time_bin_size": 1000.0 * DT,
            "temporal_alignment_event": "Go cue onset",
            "off_start": TMIN,
            "off_end": TMAX,
            "session_info": session_infos,
            "reference_notes": {
                "neural_binning": "5 ms bins, -2.5 s to +2.5 s around go cue",
                "neural_smoothing": "causal gaussian, window length 15, reflect boundary",
                "quality_filter": "exclude garbage/gabrga/noisy/real?",
                "low_fr_threshold_hz": LOW_FR_HZ,
                "trial_filter": "exclude early and photostim trials",
            },
        },
    }
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all reference sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only 2 sessions for testing.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing summary plots for up to 2 sessions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.full and not args.sample:
        args.full = True

    if args.sample:
        session_specs = [spec for spec in ALL_SPECS if spec.session_id in SAMPLE_IDS]
    else:
        session_specs = ALL_SPECS

    t0 = time.time()
    data = build_dataset(session_specs, show_processing=args.show_processing)
    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    with args.outpicklefile.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_trials = sum(len(sess) for sess in data["neural"])
    total_neurons = int(sum(len(idx) for idx in data["brain_region_idx"]))
    print(f"Saved {args.outpicklefile}")
    print(f"Sessions: {len(data['neural'])}")
    print(f"Subjects: {len(data['subjects'])}")
    print(f"Trials: {total_trials}")
    print(f"Neurons: {total_neurons}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
