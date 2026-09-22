#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
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
import scipy.io as sio
from scipy.io.matlab import mat_struct
from scipy.signal.windows import gaussian


APP_ROOT = Path("/app")
CODE_ROOT = APP_ROOT / "code"
DATA_ROOT = APP_ROOT / "data"
MANIFEST_ROOT = CODE_ROOT / "DataLoadingScripts" / "Recording and video"

TMIN = -2.5
TMAX = 2.5
DT = 1 / 100  # 10 ms; matches most figure/decoder scripts
SMOOTH_N = 15
BCTYPE = "reflect"
ADVANCE_MOVEMENT = 0.0
LOW_FR_HZ = 1.0
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}

INPUT_NAMES = ["time_from_go_cue"]
OUTPUT_NAMES = [
    "lick_direction",
    "behavioral_context",
    "outcome",
    "tongue_velocity",
    "paw_velocity",
    "motion_energy",
]
OUTPUT_VALUES = [
    ["left", "right", "none"],
    ["DR", "WC"],
    ["incorrect", "correct", "ignore"],
    ["low", "high", "not_visible"],
    ["low", "high", "not_visible"],
    ["low", "high", "no_video"],
]


@dataclass(frozen=True)
class SessionSpec:
    subject: str
    date: str
    probes: tuple[int, ...]
    data_path: Path
    cohort: str

    @property
    def session_id(self) -> str:
        return f"{self.subject}_{self.date}"


def matlab_quality_ok(quality: str) -> bool:
    return quality.strip().lower() not in QUALITY_EXCLUDE


def mode_float(values: np.ndarray, decimals: int = 6) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    rounded = np.round(arr, decimals=decimals)
    uniq, counts = np.unique(rounded, return_counts=True)
    return float(uniq[np.argmax(counts)])


def fill_nearest_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    if x.size == 0:
        return x
    idx = np.flatnonzero(np.isfinite(x))
    if idx.size == 0:
        return x
    missing = np.flatnonzero(~np.isfinite(x))
    if missing.size:
        x[missing] = np.interp(missing, idx, x[idx])
    return x


def smooth_causal_reflect(x: np.ndarray, n: int, bctype: str = "reflect") -> np.ndarray:
    if n in (0, 1):
        return np.asarray(x, dtype=float).copy()

    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]

    if bctype.lower() == "reflect":
        padded = np.concatenate([arr[:n, :], arr], axis=0)
        trim = n
    elif bctype.lower() == "zeropad":
        padded = np.concatenate([np.zeros((n, arr.shape[1])), arr], axis=0)
        trim = n
    else:
        padded = arr
        trim = 0

    std = ((n - 1) / 2) / 2.5
    kern = gaussian(n, std=std)
    kern[: len(kern) // 2] = 0.0
    kern = kern / kern.sum()

    out = np.zeros_like(padded, dtype=float)
    for col in range(padded.shape[1]):
        out[:, col] = np.convolve(padded[:, col], kern, mode="same")
    out = out[trim:, :]
    return out[:, 0] if x.ndim == 1 else out


def interp_preserve_nan(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    t_src = np.asarray(t_src, dtype=float).reshape(-1)
    y_src = np.asarray(y_src, dtype=float).reshape(-1)
    out = np.full(t_dst.shape, np.nan, dtype=float)
    finite = np.isfinite(t_src) & np.isfinite(y_src)
    if finite.sum() < 2:
        return out

    idx = np.flatnonzero(finite)
    splits = np.where(np.diff(idx) > 1)[0] + 1
    for block in np.split(idx, splits):
        if block.size < 2:
            continue
        tt = t_src[block]
        yy = y_src[block]
        mask = (t_dst >= tt[0]) & (t_dst <= tt[-1])
        if np.any(mask):
            out[mask] = np.interp(t_dst[mask], tt, yy, left=np.nan, right=np.nan)
    return out


def interp_numeric(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    t_src = np.asarray(t_src, dtype=float).reshape(-1)
    y_src = np.asarray(y_src, dtype=float).reshape(-1)
    finite = np.isfinite(t_src) & np.isfinite(y_src)
    out = np.full(t_dst.shape, np.nan, dtype=float)
    if finite.sum() < 2:
        return out
    tt = t_src[finite]
    yy = y_src[finite]
    mask = (t_dst >= tt[0]) & (t_dst <= tt[-1])
    if np.any(mask):
        out[mask] = np.interp(t_dst[mask], tt, yy, left=np.nan, right=np.nan)
    return out


def to_numeric_1d(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def to_stripped_string(x) -> str:
    if isinstance(x, str):
        return x.strip()
    if x is None:
        return ""
    arr = np.asarray(x)
    if arr.size == 0:
        return ""
    if arr.dtype.kind in {"U", "S"}:
        flat = arr.reshape(-1)
        if flat.size == 1:
            return str(flat[0]).strip()
        return "".join(str(v) for v in flat).strip()
    if arr.dtype == object:
        return to_stripped_string(arr.reshape(-1)[0])
    return str(arr.reshape(-1)[0]).strip()


def h5_deref(f: h5py.File, ref):
    if ref is None:
        return None
    try:
        if not ref:
            return None
    except TypeError:
        pass
    return f[ref]


def h5_decode_string_dataset(ds) -> str:
    if ds is None:
        return ""
    arr = ds[()]
    arr = np.asarray(arr)
    if arr.dtype.kind in {"U", "S"}:
        return "".join(str(v) for v in arr.reshape(-1)).strip()
    if arr.dtype == np.uint16:
        return "".join(chr(int(v)) for v in arr.reshape(-1) if int(v) != 0).strip()
    if arr.dtype == object and arr.size:
        return h5_decode_string_dataset(ds.file[arr.reshape(-1)[0]])
    return str(arr.reshape(-1)[0]).strip() if arr.size else ""


def h5_cell_size(cell_ds) -> int:
    return int(np.asarray(cell_ds[()]).size)


def h5_cell_ref(cell_ds, idx: int):
    refs = np.asarray(cell_ds[()]).reshape(-1, order="F")
    if idx >= refs.size:
        raise IndexError(f"Cell index {idx} out of range for {refs.size} elements")
    return refs[idx]


def h5_read_cell_numeric_1d(f: h5py.File, cell_ds, idx: int) -> np.ndarray:
    ref = h5_cell_ref(cell_ds, idx)
    target = h5_deref(f, ref)
    if target is None:
        return np.array([], dtype=float)
    return np.asarray(target[()], dtype=float).reshape(-1)


def load_session_v73(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        obj = f["obj"]
        bp = obj["bp"]
        ev = bp["ev"]
        sglx = obj["sglx"]
        bitcode = sglx["bitcode"]

        ntrials = int(np.asarray(bp["Ntrials"][()], dtype=float).reshape(-1)[0])
        bp_out = {
            "Ntrials": ntrials,
            "hit": bp["hit"][()].reshape(-1).astype(bool),
            "miss": bp["miss"][()].reshape(-1).astype(bool),
            "no": bp["no"][()].reshape(-1).astype(bool),
            "early": bp["early"][()].reshape(-1).astype(bool),
            "autowater": bp["autowater"][()].reshape(-1).astype(bool),
            "R": bp["R"][()].reshape(-1).astype(bool),
            "L": bp["L"][()].reshape(-1).astype(bool),
            "stim_enable": bp["stim"]["enable"][()].reshape(-1).astype(bool),
            "ev": {
                "bitStart": ev["bitStart"][()].reshape(-1),
                "sample": ev["sample"][()].reshape(-1),
                "delay": ev["delay"][()].reshape(-1),
                "goCue": ev["goCue"][()].reshape(-1),
                "reward": ev["reward"][()].reshape(-1),
                "lickL": [h5_read_cell_numeric_1d(f, ev["lickL"], i) for i in range(ntrials)],
                "lickR": [h5_read_cell_numeric_1d(f, ev["lickR"], i) for i in range(ntrials)],
            },
        }

        probe_locs = []
        if "ex" in obj and "probe" in obj["ex"] and "loc" in obj["ex"]["probe"]:
            loc_ds = obj["ex"]["probe"]["loc"]
            if loc_ds.dtype == object:
                for p in range(h5_cell_size(loc_ds)):
                    target = h5_deref(f, h5_cell_ref(loc_ds, p))
                    probe_locs.append(h5_decode_string_dataset(target) if target is not None else "")
            else:
                probe_locs.append(h5_decode_string_dataset(loc_ds))

        traj_views = []
        traj_ds = obj["traj"]
        for view_idx in range(h5_cell_size(traj_ds)):
            view = h5_deref(f, h5_cell_ref(traj_ds, view_idx))
            trials = []
            for tr_idx in range(ntrials):
                feat_outer = h5_deref(f, h5_cell_ref(view["featNames"], tr_idx))
                feat_names = []
                for ref in feat_outer[()].reshape(-1):
                    feat_names.append(h5_decode_string_dataset(h5_deref(f, ref)))
                ts = np.asarray(h5_deref(f, h5_cell_ref(view["ts"], tr_idx))[()], dtype=float)
                if ts.ndim == 3:
                    ts = np.transpose(ts, (2, 1, 0))
                frame_times = np.asarray(h5_deref(f, h5_cell_ref(view["frameTimes"], tr_idx))[()], dtype=float).reshape(-1)
                ndropped = np.asarray(h5_deref(f, h5_cell_ref(view["NdroppedFrames"], tr_idx))[()], dtype=float).reshape(-1)
                ndropped_val = float(ndropped[0]) if ndropped.size else np.nan
                trials.append(
                    {
                        "featNames": feat_names,
                        "ts": ts,
                        "frameTimes": frame_times,
                        "NdroppedFrames": ndropped_val,
                    }
                )
            traj_views.append(trials)

        clu_probes = []
        if "clu" in obj:
            clu_ds = obj["clu"]
            for probe_idx in range(h5_cell_size(clu_ds)):
                probe_target = h5_deref(f, h5_cell_ref(clu_ds, probe_idx))
                if probe_target is None or not isinstance(probe_target, h5py.Group):
                    clu_probes.append([])
                    continue
                nunits = h5_cell_size(probe_target["quality"])
                site_field = "site" if "site" in probe_target else ("channel" if "channel" in probe_target else None)
                units = []
                for unit_idx in range(nunits):
                    quality = h5_decode_string_dataset(h5_deref(f, h5_cell_ref(probe_target["quality"], unit_idx)))
                    tm = np.asarray(h5_deref(f, h5_cell_ref(probe_target["tm"], unit_idx))[()], dtype=float).reshape(-1)
                    trialtm = np.asarray(h5_deref(f, h5_cell_ref(probe_target["trialtm"], unit_idx))[()], dtype=float).reshape(-1)
                    trial = np.asarray(h5_deref(f, h5_cell_ref(probe_target["trial"], unit_idx))[()], dtype=float).reshape(-1).astype(int)
                    site = (
                        np.asarray(h5_deref(f, h5_cell_ref(probe_target[site_field], unit_idx))[()], dtype=float).reshape(-1)
                        if site_field is not None
                        else np.array([], dtype=float)
                    )
                    units.append(
                        {
                            "quality": quality,
                            "tm": tm,
                            "trialtm": trialtm,
                            "trial": trial,
                            "site": int(site[0]) if site.size else -1,
                        }
                    )
                clu_probes.append(units)

        if len(probe_locs) < len(clu_probes):
            probe_locs.extend(["ALM"] * (len(clu_probes) - len(probe_locs)))

        return {
            "format": "v73",
            "bp": bp_out,
            "sglx_fs": float(np.asarray(sglx["fs"][()], dtype=float).reshape(-1)[0]),
            "sglx_bitstart": bitcode["bitstart"][()].reshape(-1),
            "traj": traj_views,
            "clu_probes": clu_probes,
            "probe_locs": probe_locs,
        }


def flatten_v5_cell(arr) -> list:
    if isinstance(arr, np.ndarray):
        return list(np.ravel(arr))
    return [arr]


def load_session_v5(path: Path) -> dict:
    obj = sio.loadmat(path, struct_as_record=False, squeeze_me=True)["obj"]
    ntrials = int(obj.bp.Ntrials)

    traj_views = []
    for view in flatten_v5_cell(obj.traj):
        trials = []
        for tr in np.ravel(view):
            frame_times = np.asarray(getattr(tr, "frameTimes", np.array([])), dtype=float).reshape(-1)
            trials.append(
                {
                    "featNames": [to_stripped_string(x) for x in np.ravel(tr.featNames)],
                    "ts": np.asarray(tr.ts, dtype=float),
                    "frameTimes": frame_times,
                    "NdroppedFrames": float(np.asarray(getattr(tr, "NdroppedFrames", np.nan)).reshape(-1)[0]),
                }
            )
        traj_views.append(trials)

    clu_probes = []
    if hasattr(obj, "clu") and obj.clu is not None:
        units = []
        clu_arr = np.ravel(obj.clu)
        for unit in clu_arr:
            units.append(
                {
                    "quality": to_stripped_string(unit.quality),
                    "tm": np.asarray(unit.tm, dtype=float).reshape(-1),
                    "trialtm": np.asarray(unit.trialtm, dtype=float).reshape(-1),
                    "trial": np.asarray(unit.trial, dtype=float).reshape(-1).astype(int),
                    "site": int(np.asarray(unit.channel).reshape(-1)[0]) if hasattr(unit, "channel") else -1,
                }
            )
        clu_probes.append(units)

    return {
        "format": "v5",
        "bp": {
            "Ntrials": ntrials,
            "hit": to_numeric_1d(obj.bp.hit).astype(bool),
            "miss": to_numeric_1d(obj.bp.miss).astype(bool),
            "no": to_numeric_1d(obj.bp.no).astype(bool),
            "early": to_numeric_1d(obj.bp.early).astype(bool),
            "autowater": to_numeric_1d(obj.bp.autowater).astype(bool),
            "R": to_numeric_1d(obj.bp.R).astype(bool),
            "L": to_numeric_1d(obj.bp.L).astype(bool),
            "stim_enable": to_numeric_1d(obj.bp.stim.enable).astype(bool),
            "ev": {
                "bitStart": to_numeric_1d(obj.bp.ev.bitStart),
                "sample": to_numeric_1d(obj.bp.ev.sample),
                "delay": to_numeric_1d(obj.bp.ev.delay),
                "goCue": to_numeric_1d(obj.bp.ev.goCue),
                "reward": to_numeric_1d(obj.bp.ev.reward),
                "lickL": [np.asarray(x, dtype=float).reshape(-1) for x in np.ravel(obj.bp.ev.lickL)],
                "lickR": [np.asarray(x, dtype=float).reshape(-1) for x in np.ravel(obj.bp.ev.lickR)],
            },
        },
        "sglx_fs": float(np.asarray(obj.sglx.fs).reshape(-1)[0]),
        "sglx_bitstart": to_numeric_1d(obj.sglx.bitcode.bitstart),
        "traj": traj_views,
        "clu_probes": clu_probes,
        "probe_locs": [to_stripped_string(obj.ex.probe.loc)],
    }


def load_raw_session(path: Path) -> dict:
    if h5py.is_hdf5(path):
        return load_session_v73(path)
    return load_session_v5(path)


def load_motion_energy(spec: SessionSpec) -> tuple[list[np.ndarray] | None, float | None]:
    me_path = spec.data_path.with_name(f"motionEnergy_{spec.subject}_{spec.date}.mat")
    if not me_path.exists():
        return None, None
    me = sio.loadmat(me_path, struct_as_record=False, squeeze_me=True)["me"]
    move_thresh = None
    payload = me
    while hasattr(payload, "_fieldnames") and "data" in payload._fieldnames:
        if hasattr(payload, "moveThresh"):
            move_thresh = float(np.asarray(payload.moveThresh).reshape(-1)[0])
        next_payload = payload.data
        if next_payload is payload:
            break
        payload = next_payload
    trial_source = payload
    data = []
    for trial_arr in np.ravel(trial_source):
        data.append(np.asarray(trial_arr, dtype=float).reshape(-1))
    return data, move_thresh


def parse_probe_list(text: str) -> tuple[int, ...]:
    return tuple(int(v) for v in re.findall(r"\d+", text))


def parse_manifest_sessions() -> list[SessionSpec]:
    sessions: list[SessionSpec] = []
    for manifest in sorted(MANIFEST_ROOT.glob("load*_ALMVideo.m")):
        subject = manifest.stem.replace("load", "").replace("_ALMVideo", "")
        lines = [line for line in manifest.read_text().splitlines() if not line.lstrip().startswith("%")]
        text = "\n".join(lines)
        dates = re.findall(r"meta\(end\)\.date = '([^']+)';", text)
        probes = re.findall(r"meta\(end\)\.probe = ([^;]+);", text)
        if len(dates) != len(probes):
            raise RuntimeError(f"Could not parse {manifest}")
        for date, probe_text in zip(dates, probes):
            probe_list = parse_probe_list(probe_text)
            for cohort in ("Ephys_Behavior", "RandomizedDelay_Ephys_Behavior"):
                data_path = DATA_ROOT / cohort / f"data_structure_{subject}_{date}.mat"
                if data_path.exists():
                    sessions.append(
                        SessionSpec(
                            subject=subject,
                            date=date,
                            probes=probe_list,
                            data_path=data_path,
                            cohort=cohort,
                        )
                    )
                    break
    return sessions


def pick_sample_sessions(sessions: list[SessionSpec]) -> list[SessionSpec]:
    preferred = ["JEB6_2021-04-18", "JEB23_2023-10-18"]
    by_id = {s.session_id: s for s in sessions}
    picked = [by_id[sid] for sid in preferred if sid in by_id]
    if len(picked) < 2:
        for session in sessions:
            if session not in picked:
                picked.append(session)
            if len(picked) == 2:
                break
    return picked


def find_video_offset(raw: dict) -> float:
    bit_start = mode_float(raw["bp"]["ev"]["bitStart"])
    vid_file_offset = mode_float(raw["sglx_bitstart"]) / raw["sglx_fs"]
    return vid_file_offset - bit_start


def build_time_axis() -> np.ndarray:
    edges = np.arange(TMIN, TMAX + DT, DT)
    return edges[:-1] + DT / 2


def get_feat_index(trial: dict, feat_name: str) -> int | None:
    for idx, name in enumerate(trial["featNames"]):
        if name == feat_name:
            return idx
    return None


def align_feature(raw: dict, time_axis: np.ndarray, align_times: np.ndarray, view_idx: int, feat_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ntrials = raw["bp"]["Ntrials"]
    vidshift = find_video_offset(raw)
    x_all = np.full((time_axis.size, ntrials), np.nan, dtype=float)
    y_all = np.full((time_axis.size, ntrials), np.nan, dtype=float)
    vis_all = np.zeros((time_axis.size, ntrials), dtype=bool)
    is_tongue = "tongue" in feat_name

    for tr_idx in range(ntrials):
        trial = raw["traj"][view_idx][tr_idx]
        feat_idx = get_feat_index(trial, feat_name)
        if feat_idx is None:
            continue
        ndropped = trial["NdroppedFrames"]
        if np.isnan(ndropped):
            continue
        ts = np.asarray(trial["ts"][:, :2, feat_idx], dtype=float)
        frame_times = np.asarray(trial["frameTimes"], dtype=float).reshape(-1)
        if frame_times.size == 0:
            frame_times = (np.arange(ts.shape[0]) + 1) / 400.0
        x = interp_preserve_nan(frame_times - vidshift - align_times[tr_idx], ts[:, 0], time_axis + ADVANCE_MOVEMENT)
        y = interp_preserve_nan(frame_times - vidshift - align_times[tr_idx], ts[:, 1], time_axis + ADVANCE_MOVEMENT)
        vis = np.isfinite(x) & np.isfinite(y)
        if not is_tongue:
            x = fill_nearest_1d(x)
            y = fill_nearest_1d(y)
        x_all[:, tr_idx] = x
        y_all[:, tr_idx] = y
        vis_all[:, tr_idx] = vis
    return x_all, y_all, vis_all


def compute_speed(x: np.ndarray, y: np.ndarray, feat_name: str) -> np.ndarray:
    speed = np.full_like(x, np.nan, dtype=float)
    is_tongue = "tongue" in feat_name
    for tr_idx in range(x.shape[1]):
        xx = x[:, tr_idx]
        yy = y[:, tr_idx]
        if not (np.isfinite(xx).any() and np.isfinite(yy).any()):
            continue
        xvel = np.gradient(xx)
        yvel = np.gradient(yy)
        if not is_tongue:
            diffs = np.column_stack([np.diff(xx), np.diff(yy)])
            if np.isfinite(diffs).any():
                basederiv = np.nanmedian(diffs, axis=0)
            else:
                basederiv = np.array([0.0, 0.0], dtype=float)
            baseline = basederiv[0] if np.isfinite(basederiv[0]) else 0.0
            xvel = xvel - baseline
            yvel = yvel - baseline
            xvel = fill_nearest_1d(xvel)
            yvel = fill_nearest_1d(yvel)
        speed[:, tr_idx] = np.sqrt(xvel ** 2 + yvel ** 2)
    return speed


def align_motion_energy(raw: dict, spec: SessionSpec, time_axis: np.ndarray, align_times: np.ndarray) -> np.ndarray | None:
    me_data, _ = load_motion_energy(spec)
    if me_data is None:
        return None
    vidshift = find_video_offset(raw)
    ntrials = raw["bp"]["Ntrials"]
    aligned = np.full((time_axis.size, ntrials), np.nan, dtype=float)
    for tr_idx in range(ntrials):
        if tr_idx >= len(me_data):
            continue
        y = me_data[tr_idx]
        if y.size == 0:
            continue
        trial = raw["traj"][0][tr_idx]
        frame_times = np.asarray(trial["frameTimes"], dtype=float).reshape(-1)
        if frame_times.size == 0 or not np.all(np.isfinite(frame_times)):
            frame_times = (np.arange(y.size) + 1) / 400.0
            tt = frame_times - 0.5 - align_times[tr_idx]
        else:
            tt = frame_times - vidshift - align_times[tr_idx]
        sig = interp_numeric(tt, y, time_axis + ADVANCE_MOVEMENT)
        aligned[:, tr_idx] = fill_nearest_1d(sig)
    return aligned


def lick_direction_value(bp: dict, trial_idx: int) -> int:
    if bp["no"][trial_idx]:
        return 2
    right_choice = (bp["R"][trial_idx] and bp["hit"][trial_idx]) or (bp["L"][trial_idx] and bp["miss"][trial_idx])
    left_choice = (bp["L"][trial_idx] and bp["hit"][trial_idx]) or (bp["R"][trial_idx] and bp["miss"][trial_idx])
    if left_choice:
        return 0
    if right_choice:
        return 1
    return 2


def context_value(bp: dict, trial_idx: int) -> int:
    return 1 if bp["autowater"][trial_idx] else 0


def outcome_value(bp: dict, trial_idx: int) -> int:
    if bp["miss"][trial_idx]:
        return 0
    if bp["hit"][trial_idx]:
        return 1
    return 2


def discretize_visible_signal(values: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, float]:
    out = np.full(values.shape, 2, dtype=np.int64)
    valid_vals = values[visible & np.isfinite(values)]
    if valid_vals.size == 0:
        return out, float("nan")
    thresh = float(np.nanpercentile(valid_vals, 50))
    low_mask = visible & np.isfinite(values) & (values < thresh)
    high_mask = visible & np.isfinite(values) & ~low_mask
    out[low_mask] = 0
    out[high_mask] = 1
    return out, thresh


def discretize_motion_energy(me_aligned: np.ndarray | None) -> tuple[np.ndarray, float]:
    if me_aligned is None:
        return None, float("nan")
    out = np.full(me_aligned.shape, 2, dtype=np.int64)
    valid = np.isfinite(me_aligned)
    vals = me_aligned[valid]
    if vals.size == 0:
        return out, float("nan")
    thresh = float(np.nanpercentile(vals, 50))
    out[valid & (me_aligned < thresh)] = 0
    out[valid & (me_aligned >= thresh)] = 1
    return out, thresh


def normalize_region(loc: str) -> str:
    return "ALM" if "ALM" in loc.upper() else (loc.strip() or "ALM")


def selected_units(raw: dict, probes: tuple[int, ...]) -> tuple[list[dict], list[str]]:
    units = []
    regions = []
    for probe in probes:
        probe_idx = probe - 1
        probe_units = raw["clu_probes"][probe_idx] if probe_idx < len(raw["clu_probes"]) else []
        probe_loc = raw["probe_locs"][probe_idx] if probe_idx < len(raw["probe_locs"]) else "ALM"
        for unit in probe_units:
            if matlab_quality_ok(unit["quality"]):
                units.append(unit)
                regions.append(normalize_region(probe_loc))
    return units, regions


def bin_session_neural(raw: dict, units: list[dict], valid_trials: np.ndarray, time_axis: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    ntrials = valid_trials.size
    if ntrials == 0 or len(units) == 0:
        return [], np.array([], dtype=int)

    trial_map = {trial_idx: i for i, trial_idx in enumerate(valid_trials.tolist())}
    nbins = time_axis.size
    edges = np.arange(TMIN, TMAX + DT, DT)
    keep_mats = []
    keep_idx = []
    align_times = raw["bp"]["ev"]["goCue"]

    for unit_idx, unit in enumerate(units):
        counts = np.zeros((ntrials, nbins), dtype=float)
        trial_idx = unit["trial"] - 1
        aligned = unit["trialtm"] - align_times[trial_idx]
        valid_mask = np.array([t in trial_map for t in trial_idx], dtype=bool)
        in_win = (aligned >= TMIN) & (aligned < TMAX)
        mask = valid_mask & in_win
        if np.any(mask):
            mapped_trials = np.array([trial_map[t] for t in trial_idx[mask]], dtype=int)
            bin_idx = np.floor((aligned[mask] - TMIN) / DT).astype(int)
            np.add.at(counts, (mapped_trials, bin_idx), 1.0)
        fr = smooth_causal_reflect((counts / DT).T, SMOOTH_N, BCTYPE).T
        mean_fr = float(np.nanmean(fr))
        if mean_fr > LOW_FR_HZ:
            keep_idx.append(unit_idx)
            keep_mats.append(fr.astype(np.float32))

    if not keep_mats:
        return [], np.array([], dtype=int)

    neural_trials = []
    stacked = np.stack(keep_mats, axis=0)  # neurons x trials x time
    for tr in range(ntrials):
        neural_trials.append(stacked[:, tr, :])
    return neural_trials, np.array(keep_idx, dtype=int)


def build_session_outputs(
    raw: dict,
    spec: SessionSpec,
    valid_trials: np.ndarray,
    time_axis: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    align_times = raw["bp"]["ev"]["goCue"]
    nraw = raw["bp"]["Ntrials"]

    tongue_x, tongue_y, tongue_vis = align_feature(raw, time_axis, align_times, 0, "tongue")
    tongue_speed = compute_speed(tongue_x, tongue_y, "tongue")
    tongue_cat, tongue_thr = discretize_visible_signal(tongue_speed, tongue_vis)

    paw_feats = ["top_paw", "bottom_paw"]
    paw_speeds = []
    paw_vis = []
    for feat in paw_feats:
        x, y, vis = align_feature(raw, time_axis, align_times, 1, feat)
        paw_speeds.append(compute_speed(x, y, feat))
        paw_vis.append(vis)
    paw_stack = np.stack(paw_speeds, axis=0)
    paw_finite = np.isfinite(paw_stack)
    paw_speed = np.max(np.where(paw_finite, paw_stack, -np.inf), axis=0)
    paw_speed[~paw_finite.any(axis=0)] = np.nan
    paw_visible = np.any(np.stack(paw_vis, axis=0), axis=0)
    paw_cat, paw_thr = discretize_visible_signal(paw_speed, paw_visible)

    me_aligned = align_motion_energy(raw, spec, time_axis, align_times)
    me_cat, me_thr = discretize_motion_energy(me_aligned)
    if me_cat is None:
        me_cat = np.full((time_axis.size, nraw), 2, dtype=np.int64)

    inputs = []
    outputs = []
    T = time_axis.size
    for raw_trial_idx in valid_trials.tolist():
        inp = time_axis[np.newaxis, :].astype(np.float32)
        out = np.empty((len(OUTPUT_NAMES), T), dtype=np.int64)
        out[0, :] = lick_direction_value(raw["bp"], raw_trial_idx)
        out[1, :] = context_value(raw["bp"], raw_trial_idx)
        out[2, :] = outcome_value(raw["bp"], raw_trial_idx)
        out[3, :] = tongue_cat[:, raw_trial_idx]
        out[4, :] = paw_cat[:, raw_trial_idx]
        out[5, :] = me_cat[:, raw_trial_idx]
        inputs.append(inp)
        outputs.append(out)

    thresholds = {
        "tongue_velocity_median": tongue_thr,
        "paw_velocity_median": paw_thr,
        "motion_energy_median": me_thr,
    }
    return inputs, outputs, thresholds


def plot_processing_summary(
    spec: SessionSpec,
    time_axis: np.ndarray,
    neural_trials: list[np.ndarray],
    output_trials: list[np.ndarray],
    thresholds: dict,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(spec.session_id)

    trial_idx = 0
    neural = neural_trials[trial_idx]
    show_n = min(40, neural.shape[0])
    axes[0, 0].imshow(neural[:show_n], aspect="auto", interpolation="nearest", extent=[time_axis[0], time_axis[-1], show_n, 0])
    axes[0, 0].axvline(0.0, color="white", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Neural (first trial)")
    axes[0, 0].set_xlabel("Time from go cue (s)")
    axes[0, 0].set_ylabel("Neuron")

    out = output_trials[trial_idx]
    for idx, name in enumerate(OUTPUT_NAMES):
        axes[0, 1].plot(time_axis, out[idx] + idx * 3, label=name)
    axes[0, 1].axvline(0.0, color="k", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Categorical outputs (first trial)")
    axes[0, 1].set_xlabel("Time from go cue (s)")
    axes[0, 1].legend(fontsize=8, loc="upper right")

    axes[1, 0].hist(out[3], bins=np.arange(-0.5, 3.5, 1.0), rwidth=0.8)
    axes[1, 0].set_title(f"Tongue classes; thr={thresholds['tongue_velocity_median']:.3g}")
    axes[1, 1].hist(out[4], bins=np.arange(-0.5, 3.5, 1.0), rwidth=0.8)
    axes[1, 1].set_title(f"Paw classes; thr={thresholds['paw_velocity_median']:.3g}")

    axes[2, 0].hist(out[5], bins=np.arange(-0.5, 3.5, 1.0), rwidth=0.8)
    axes[2, 0].set_title(f"Motion energy classes; thr={thresholds['motion_energy_median']:.3g}")

    lick_vals = np.array([trial[0, 0] for trial in output_trials], dtype=int)
    counts = np.bincount(lick_vals, minlength=3)
    axes[2, 1].bar(np.arange(3), counts)
    axes[2, 1].set_xticks(np.arange(3), OUTPUT_VALUES[0])
    axes[2, 1].set_title("Lick direction counts")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def convert_one_session(spec: SessionSpec, show_processing: bool = False) -> tuple[dict | None, dict]:
    t0 = time.perf_counter()
    raw = load_raw_session(spec.data_path)
    candidate_trials = np.flatnonzero(~raw["bp"]["early"] & np.isfinite(raw["bp"]["ev"]["goCue"]))
    initial_valid_trial_count = int(candidate_trials.size)
    units, unit_regions = selected_units(raw, spec.probes)
    time_axis = build_time_axis()
    neural_trials, keep_unit_idx = bin_session_neural(raw, units, candidate_trials, time_axis)

    nonzero_trial_mask = np.array([np.any(trial != 0) for trial in neural_trials], dtype=bool)
    if nonzero_trial_mask.size and not np.all(nonzero_trial_mask):
        candidate_trials = candidate_trials[nonzero_trial_mask]
        neural_trials = [trial for trial, keep in zip(neural_trials, nonzero_trial_mask.tolist()) if keep]
    valid_trials = candidate_trials

    info = {
        "session_id": spec.session_id,
        "subject": spec.subject,
        "date": spec.date,
        "cohort": spec.cohort,
        "raw_trials": int(raw["bp"]["Ntrials"]),
        "candidate_valid_trials": initial_valid_trial_count,
        "valid_trials": int(valid_trials.size),
        "units_pre_filter": int(len(units)),
        "units_post_filter": int(len(keep_unit_idx)),
        "zero_neural_trials_removed": int(nonzero_trial_mask.size - nonzero_trial_mask.sum()) if nonzero_trial_mask.size else 0,
    }

    if len(neural_trials) < 2 or len(keep_unit_idx) < 10:
        info["skipped"] = True
        info["elapsed_sec"] = time.perf_counter() - t0
        return None, info

    inputs, outputs, thresholds = build_session_outputs(raw, spec, valid_trials, time_axis)
    unit_regions = [unit_regions[i] for i in keep_unit_idx.tolist()]
    info["skipped"] = False
    info["elapsed_sec"] = time.perf_counter() - t0
    info["thresholds"] = thresholds

    if show_processing:
        out_path = APP_ROOT / f"processing_{spec.session_id}.png"
        plot_processing_summary(spec, time_axis, neural_trials, outputs, thresholds, out_path)

    session_data = {
        "neural": neural_trials,
        "input": inputs,
        "output": outputs,
        "brain_regions": unit_regions,
        "subject": spec.subject,
        "info": info,
    }
    return session_data, info


def build_dataset(processed_sessions: list[dict]) -> dict:
    subjects = sorted({sess["subject"] for sess in processed_sessions})
    subject_to_idx = {subj: i for i, subj in enumerate(subjects)}

    brain_regions = ["ALM"]

    data = {
        "neural": [sess["neural"] for sess in processed_sessions],
        "input": [sess["input"] for sess in processed_sessions],
        "output": [sess["output"] for sess in processed_sessions],
        "subjects": subjects,
        "subject_idx": np.array([subject_to_idx[sess["subject"]] for sess in processed_sessions], dtype=np.int64),
        "brain_regions": brain_regions,
        "brain_region_idx": [np.zeros(sess["neural"][0].shape[0], dtype=np.int64) for sess in processed_sessions],
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "output_values": OUTPUT_VALUES,
        "metadata": {
            "task_description": (
                "Go-cue-aligned ALM electrophysiology from fixed-delay, two-context, and randomized-delay directional licking sessions. "
                "Decoder predicts lick direction, context, outcome, and discretized tongue/paw/motion-energy variables."
            ),
            "time_bin_size": 10.0,
            "temporal_alignment_event": "goCue",
            "off_start": TMIN,
            "off_end": TMAX,
            "neural_representation": "smoothed_firing_rate",
            "smoothing_window_samples": SMOOTH_N,
            "video_frame_rate_hz": 400.0,
            "unit_quality_exclusion": sorted(QUALITY_EXCLUDE),
            "unit_fr_threshold_hz": LOW_FR_HZ,
            "trial_exclusion": "early_lick_only",
            "session_ids": [sess["info"]["session_id"] for sess in processed_sessions],
            "session_info": [sess["info"] for sess in processed_sessions],
            "reference_manifest_dir": str(MANIFEST_ROOT),
        },
    }
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ALM dataset into decoder format.")
    parser.add_argument("outpicklefile", type=str, help="Output pickle path")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", help="Process all curated sessions (default)")
    group.add_argument("--sample", action="store_true", help="Process only two representative sessions")
    parser.add_argument("--show-processing", action="store_true", help="Save processing plots for up to 2 sessions")
    args = parser.parse_args()

    session_specs = parse_manifest_sessions()
    if args.sample:
        session_specs = pick_sample_sessions(session_specs)
    mode = "sample" if args.sample else "full"
    print(f"Processing mode: {mode}")
    print(f"Curated sessions available: {len(parse_manifest_sessions())}")
    print(f"Sessions selected: {len(session_specs)}")

    processed_sessions = []
    timings = []
    plot_budget = 2 if args.show_processing else 0
    for idx, spec in enumerate(session_specs, start=1):
        do_plot = plot_budget > 0
        session_data, info = convert_one_session(spec, show_processing=do_plot)
        if do_plot:
            plot_budget -= 1
        timings.append(info["elapsed_sec"])
        print(
            f"[{idx:02d}/{len(session_specs):02d}] {spec.session_id}: "
            f"raw_trials={info['raw_trials']} valid_trials={info['valid_trials']} "
            f"units={info['units_pre_filter']}->{info['units_post_filter']} "
            f"elapsed={info['elapsed_sec']:.2f}s"
        )
        if session_data is not None:
            processed_sessions.append(session_data)
        else:
            print(f"  skipped: insufficient trials or units after filtering")

    if not processed_sessions:
        raise RuntimeError("No sessions survived filtering.")

    data = build_dataset(processed_sessions)
    out_path = Path(args.outpicklefile)
    with out_path.open("wb") as f:
        pickle.dump(data, f)

    print(f"Saved {len(processed_sessions)} sessions to {out_path}")
    print(f"Mean session processing time: {np.mean(timings):.2f}s")
    print(f"Estimated total time at this rate for full curated set: {np.mean(timings) * len(parse_manifest_sessions()):.2f}s")


if __name__ == "__main__":
    main()
