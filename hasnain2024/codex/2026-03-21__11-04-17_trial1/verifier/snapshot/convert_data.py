#!/usr/bin/env python3
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


TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 200.0
SMOOTH = 15
LOW_FR_HZ = 1.0
ALIGN_EVENT = "goCue"
BCTYPE = "reflect"


@dataclass(frozen=True)
class SessionSpec:
    subject: str
    date: str
    probes: tuple[int, ...]
    session_path: Path
    folder: str

    @property
    def session_id(self) -> str:
        return f"{self.subject}_{self.date}"


def log(msg: str) -> None:
    print(msg, flush=True)


def is_hdf5_mat(path: Path) -> bool:
    try:
        with h5py.File(path, "r"):
            return True
    except OSError:
        return False


def matlab_gausswin(n: int, alpha: float = 2.5) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    if n == 1:
        return np.ones((1,), dtype=np.float64)
    k = np.arange(0, n, dtype=np.float64) - (n - 1.0) / 2.0
    return np.exp(-0.5 * (alpha * k / ((n - 1.0) / 2.0)) ** 2)


def my_smooth(x: np.ndarray, n: int, bctype: str = "none") -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if n in (0, 1):
        return x.copy()
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    else:
        squeeze = False
    if x.shape[0] == 0:
        out = x.copy()
        return out[:, 0] if squeeze else out

    if bctype.lower() == "reflect":
        pad = x[:n, :]
        x_filt = np.concatenate([pad, x], axis=0)
        trim = n
    elif bctype.lower() == "zeropad":
        pad = np.zeros((n, x.shape[1]), dtype=x.dtype)
        x_filt = np.concatenate([pad, x], axis=0)
        trim = n
    else:
        x_filt = x
        trim = 0

    kern = matlab_gausswin(n)
    kern[: n // 2] = 0.0
    kern /= kern.sum()

    out = np.empty_like(x_filt, dtype=np.float64)
    for j in range(x_filt.shape[1]):
        out[:, j] = np.convolve(x_filt[:, j], kern, mode="same")
    out = out[trim:, :]
    if squeeze:
        return out[:, 0]
    return out


def nearest_fill_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    if x.size == 0:
        return x
    mask = np.isfinite(x)
    if mask.all():
        return x
    if not mask.any():
        return np.zeros_like(x)
    idx = np.arange(x.size)
    x[~mask] = np.interp(idx[~mask], idx[mask], x[mask])
    return x


def nearest_fill_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    for j in range(x.shape[1]):
        x[:, j] = nearest_fill_1d(x[:, j])
    return x


def robust_mode(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    vals, counts = np.unique(np.round(x, 6), return_counts=True)
    return float(vals[np.argmax(counts)])


def mat_to_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore").strip()
    if isinstance(x, np.ndarray):
        if x.dtype.kind in {"U", "S"}:
            return "".join(np.asarray(x).reshape(-1).tolist()).strip()
        if x.dtype == object:
            flat = x.reshape(-1)
            parts = [mat_to_str(v) for v in flat]
            return "".join(parts).strip()
        if x.dtype.kind in {"i", "u"}:
            return "".join(chr(int(v)) for v in x.reshape(-1) if int(v) != 0).strip()
    return str(x).strip()


def normalize_feature_names(x) -> list[str]:
    if isinstance(x, (list, tuple)):
        return [mat_to_str(v) for v in x]
    arr = np.asarray(x)
    if arr.dtype == object:
        return [mat_to_str(v) for v in arr.reshape(-1)]
    return [mat_to_str(v) for v in arr.reshape(-1)]


def find_data_files(data_dir: Path) -> dict[tuple[str, str], Path]:
    out = {}
    for path in sorted(data_dir.glob("*/*.mat")):
        if not path.name.startswith("data_structure_"):
            continue
        parts = path.stem.split("_")
        out[(parts[2], parts[3])] = path
    return out


def parse_reference_session_specs(code_dir: Path, data_dir: Path) -> list[SessionSpec]:
    data_files = find_data_files(data_dir)
    specs: list[SessionSpec] = []
    loader_dir = code_dir / "DataLoadingScripts" / "Recording and video"
    for loader in sorted(loader_dir.glob("load*_ALMVideo.m")):
        loader_subject = loader.name.split("_")[0].replace("load", "")
        current = {"subject": loader_subject}
        for line in loader.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            m = re.search(r"anm = '([^']+)'", s)
            if m:
                current["subject"] = m.group(1)
            m = re.search(r"date = '([^']+)'", s)
            if m:
                current["date"] = m.group(1)
            m = re.search(r"probe = \[([0-9 ]+)\]", s)
            if m:
                current["probes"] = tuple(int(v) for v in m.group(1).split())
            m = re.search(r"probe = (\d+)", s)
            if m:
                current["probes"] = (int(m.group(1)),)
            if {"subject", "date", "probes"} <= current.keys():
                key = (current["subject"], current["date"])
                if key in data_files:
                    path = data_files[key]
                    specs.append(
                        SessionSpec(
                            subject=current["subject"],
                            date=current["date"],
                            probes=current["probes"],
                            session_path=path,
                            folder=path.parent.name,
                        )
                    )
                current = {"subject": loader_subject}
    return specs


def choose_sample_specs(specs: list[SessionSpec]) -> list[SessionSpec]:
    if len(specs) <= 2:
        return specs
    hdf5_spec = next((s for s in specs if is_hdf5_mat(s.session_path)), None)
    mat_spec = next((s for s in specs if not is_hdf5_mat(s.session_path)), None)
    chosen = []
    if hdf5_spec is not None:
        chosen.append(hdf5_spec)
    if mat_spec is not None and mat_spec not in chosen:
        chosen.append(mat_spec)
    for spec in specs:
        if spec not in chosen:
            chosen.append(spec)
        if len(chosen) == 2:
            break
    return chosen[:2]


def h5_resolve(f: h5py.File, obj):
    if isinstance(obj, h5py.Reference):
        return f[obj]
    return obj


def h5_is_empty(obj) -> bool:
    obj = h5_resolve(obj.file if hasattr(obj, "file") else None, obj)
    return isinstance(obj, h5py.Dataset) and bool(obj.attrs.get("MATLAB_empty", 0))


def h5_read_numeric(f: h5py.File, obj) -> np.ndarray:
    obj = h5_resolve(f, obj)
    if isinstance(obj, h5py.Dataset) and bool(obj.attrs.get("MATLAB_empty", 0)):
        return np.array([], dtype=np.float64)
    arr = obj[()]
    out = np.asarray(arr)
    if out.dtype == object:
        raise TypeError("Expected numeric dataset, got object refs")
    return np.asarray(out).squeeze()


def h5_read_string(f: h5py.File, obj) -> str:
    obj = h5_resolve(f, obj)
    if isinstance(obj, h5py.Dataset) and bool(obj.attrs.get("MATLAB_empty", 0)):
        return ""
    arr = np.asarray(obj[()])
    if arr.dtype.kind in {"i", "u"}:
        return "".join(chr(int(v)) for v in arr.reshape(-1) if int(v) != 0).strip()
    return mat_to_str(arr)


def h5_read_string_list(f: h5py.File, obj) -> list[str]:
    obj = h5_resolve(f, obj)
    arr = np.asarray(obj[()])
    if arr.dtype != object:
        return [h5_read_string(f, obj)]
    return [h5_read_string(f, ref) for ref in arr.reshape(-1)]


def h5_nonempty_probe_groups(f: h5py.File, clu_dataset) -> list[h5py.Group]:
    refs = np.asarray(clu_dataset[()])
    groups = []
    for ref in refs.reshape(-1):
        item = f[ref]
        if isinstance(item, h5py.Group):
            groups.append(item)
    return groups


def unwrap_motion_energy_container(x):
    current = x
    while hasattr(current, "data") and not isinstance(current, np.ndarray):
        current = current.data
    while isinstance(current, np.ndarray) and current.dtype == object and current.size == 1:
        item = current.reshape(-1)[0]
        if hasattr(item, "data"):
            current = item.data
        else:
            break
    return current


def load_motion_energy(path: Path) -> tuple[list[np.ndarray], float]:
    me = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
    me_root = me
    if isinstance(me_root, np.ndarray) and me_root.dtype == object and me_root.size == 1:
        item = me_root.reshape(-1)[0]
        if hasattr(item, "data") or hasattr(item, "moveThresh"):
            me_root = item

    if hasattr(me_root, "data") and not isinstance(me_root, np.ndarray):
        raw_data = unwrap_motion_energy_container(me_root.data)
        thresh_obj = getattr(me_root, "moveThresh", float("nan"))
    else:
        raw_data = unwrap_motion_energy_container(me_root)
        thresh_obj = float("nan")

    flat = np.asarray(raw_data, dtype=object).reshape(-1)
    data = [np.asarray(unwrap_motion_energy_container(v), dtype=np.float64).reshape(-1) for v in flat]
    if hasattr(thresh_obj, "data"):
        thresh_obj = thresh_obj.data
    thresh_arr = np.asarray(thresh_obj).reshape(-1)
    thresh = float(thresh_arr[0]) if thresh_arr.size else float("nan")
    return data, thresh


def load_session_mat(spec: SessionSpec) -> dict:
    obj = sio.loadmat(spec.session_path, squeeze_me=True, struct_as_record=False)["obj"]
    bp = obj.bp

    clu = obj.clu
    if isinstance(clu, np.ndarray) and clu.dtype == object and clu.ndim == 2:
        probes = [clu[i, 0] for i in range(clu.shape[0])]
    elif isinstance(clu, np.ndarray) and clu.dtype == object and clu.ndim == 1 and len(clu) and hasattr(clu[0], "trialtm"):
        probes = [clu]
    else:
        probes = [clu]
    probes = [p for p in probes if p is not None]
    units = []
    for probe_idx in spec.probes:
        probe = probes[probe_idx - 1]
        for unit in np.asarray(probe).reshape(-1):
            units.append(
                {
                    "quality": mat_to_str(getattr(unit, "quality", "")),
                    "trialtm": np.asarray(getattr(unit, "trialtm"), dtype=np.float64).reshape(-1),
                    "trial": np.asarray(getattr(unit, "trial"), dtype=np.int64).reshape(-1),
                }
            )

    traj_views = []
    for view in range(2):
        trials = []
        for tr in np.asarray(obj.traj[view]).reshape(-1):
            trials.append(
                {
                    "feat_names": normalize_feature_names(tr.featNames),
                    "ts": np.asarray(tr.ts, dtype=np.float64),
                    "frame_times": np.asarray(tr.frameTimes, dtype=np.float64).reshape(-1) if hasattr(tr, "frameTimes") else None,
                    "n_dropped": float(tr.NdroppedFrames) if hasattr(tr, "NdroppedFrames") else float("nan"),
                }
            )
        traj_views.append(trials)

    me_path = spec.session_path.parent / f"motionEnergy_{spec.subject}_{spec.date}.mat"
    motion_energy, motion_thresh = load_motion_energy(me_path)

    bitcode = getattr(obj.sglx, "bitcode", None)
    bitstart = np.asarray(getattr(bitcode, "bitstart", np.array([])), dtype=np.float64).reshape(-1)
    fs = float(getattr(obj.sglx, "fs", np.nan))

    return {
        "format": "mat",
        "session_id": spec.session_id,
        "subject": spec.subject,
        "R": np.asarray(bp.R, dtype=np.float64).reshape(-1),
        "L": np.asarray(bp.L, dtype=np.float64).reshape(-1),
        "hit": np.asarray(bp.hit, dtype=np.float64).reshape(-1),
        "miss": np.asarray(bp.miss, dtype=np.float64).reshape(-1),
        "no": np.asarray(bp.no, dtype=np.float64).reshape(-1),
        "early": np.asarray(bp.early, dtype=np.float64).reshape(-1),
        "autowater": np.asarray(bp.autowater, dtype=np.float64).reshape(-1),
        "stim_enable": np.asarray(bp.stim.enable, dtype=np.float64).reshape(-1),
        "events": {
            "bitStart": np.asarray(bp.ev.bitStart, dtype=np.float64).reshape(-1),
            "sample": np.asarray(bp.ev.sample, dtype=np.float64).reshape(-1),
            "delay": np.asarray(bp.ev.delay, dtype=np.float64).reshape(-1),
            "goCue": np.asarray(bp.ev.goCue, dtype=np.float64).reshape(-1),
            "reward": np.asarray(bp.ev.reward, dtype=np.float64).reshape(-1),
        },
        "units": units,
        "traj": traj_views,
        "bitcode_bitstart": bitstart,
        "fs": fs,
        "motion_energy": motion_energy,
        "motion_thresh": motion_thresh,
    }


def load_session_hdf5(spec: SessionSpec) -> dict:
    with h5py.File(spec.session_path, "r") as f:
        bp = f["obj/bp"]
        probe_refs = np.asarray(f["obj/clu"][()]).reshape(-1)
        units = []
        for probe_idx in spec.probes:
            if probe_idx < 1 or probe_idx > probe_refs.size:
                raise IndexError(f"{spec.session_id}: probe index {probe_idx} out of range for {probe_refs.size} raw probe slots")
            probe = f[probe_refs[probe_idx - 1]]
            if not isinstance(probe, h5py.Group):
                raise IndexError(f"{spec.session_id}: probe slot {probe_idx} is empty in HDF5 file")
            for i in range(probe["trialtm"].shape[0]):
                units.append(
                    {
                        "quality": h5_read_string(f, probe["quality"][i, 0]),
                        "trialtm": np.asarray(h5_read_numeric(f, probe["trialtm"][i, 0]), dtype=np.float64).reshape(-1),
                        "trial": np.asarray(h5_read_numeric(f, probe["trial"][i, 0]), dtype=np.int64).reshape(-1),
                    }
                )

        traj_views = []
        needed_by_view = [["tongue"], ["top_paw", "bottom_paw"]]
        traj_refs = np.asarray(f["obj/traj"][()])
        for view_idx in range(2):
            view_group = f[traj_refs[view_idx, 0]]
            feat_indices = {}
            for trial_idx in range(view_group["featNames"].shape[0]):
                feat_names = h5_read_string_list(f, view_group["featNames"][trial_idx, 0])
                for name in needed_by_view[view_idx]:
                    if name not in feat_indices and name in feat_names:
                        feat_indices[name] = feat_names.index(name)
                if len(feat_indices) == len(needed_by_view[view_idx]):
                    break

            trials = []
            for trial_idx in range(view_group["ts"].shape[0]):
                ts = np.asarray(h5_read_numeric(f, view_group["ts"][trial_idx, 0]), dtype=np.float64)
                if ts.ndim == 3:
                    ts = np.transpose(ts, (2, 1, 0))
                if ts.ndim == 2:
                    ts = ts[:, :, None]
                frame_times = None
                try:
                    frame_times = np.asarray(h5_read_numeric(f, view_group["frameTimes"][trial_idx, 0]), dtype=np.float64).reshape(-1)
                    if frame_times.size == 0:
                        frame_times = None
                except Exception:
                    frame_times = None
                try:
                    ndropped = float(np.asarray(h5_read_numeric(f, view_group["NdroppedFrames"][trial_idx, 0])).reshape(-1)[0])
                except Exception:
                    ndropped = float("nan")

                selected_names = []
                selected_ts = []
                for name in needed_by_view[view_idx]:
                    if name in feat_indices and ts.size:
                        selected_names.append(name)
                        selected_ts.append(ts[:, :, feat_indices[name]])
                if selected_ts:
                    selected_ts = np.stack(selected_ts, axis=2)
                else:
                    selected_ts = np.empty((ts.shape[0], 3, 0), dtype=np.float64)

                trials.append(
                    {
                        "feat_names": selected_names,
                        "ts": selected_ts,
                        "frame_times": frame_times,
                        "n_dropped": ndropped,
                    }
                )
            traj_views.append(trials)

        me_path = spec.session_path.parent / f"motionEnergy_{spec.subject}_{spec.date}.mat"
        motion_energy, motion_thresh = load_motion_energy(me_path)

        bitstart = np.asarray(h5_read_numeric(f, f["obj/sglx/bitcode/bitstart"]), dtype=np.float64).reshape(-1)
        fs = float(np.asarray(h5_read_numeric(f, f["obj/sglx/fs"])).reshape(-1)[0])

        return {
            "format": "hdf5",
            "session_id": spec.session_id,
            "subject": spec.subject,
            "R": np.asarray(bp["R"][()], dtype=np.float64).reshape(-1),
            "L": np.asarray(bp["L"][()], dtype=np.float64).reshape(-1),
            "hit": np.asarray(bp["hit"][()], dtype=np.float64).reshape(-1),
            "miss": np.asarray(bp["miss"][()], dtype=np.float64).reshape(-1),
            "no": np.asarray(bp["no"][()], dtype=np.float64).reshape(-1),
            "early": np.asarray(bp["early"][()], dtype=np.float64).reshape(-1),
            "autowater": np.asarray(bp["autowater"][()], dtype=np.float64).reshape(-1),
            "stim_enable": np.asarray(f["obj/bp/stim/enable"][()], dtype=np.float64).reshape(-1),
            "events": {
                "bitStart": np.asarray(f["obj/bp/ev/bitStart"][()], dtype=np.float64).reshape(-1),
                "sample": np.asarray(f["obj/bp/ev/sample"][()], dtype=np.float64).reshape(-1),
                "delay": np.asarray(f["obj/bp/ev/delay"][()], dtype=np.float64).reshape(-1),
                "goCue": np.asarray(f["obj/bp/ev/goCue"][()], dtype=np.float64).reshape(-1),
                "reward": np.asarray(f["obj/bp/ev/reward"][()], dtype=np.float64).reshape(-1),
            },
            "units": units,
            "traj": traj_views,
            "bitcode_bitstart": bitstart,
            "fs": fs,
            "motion_energy": motion_energy,
            "motion_thresh": motion_thresh,
        }


def load_session(spec: SessionSpec) -> dict:
    return load_session_hdf5(spec) if is_hdf5_mat(spec.session_path) else load_session_mat(spec)


def good_quality(label: str) -> bool:
    label = label.strip().lower()
    return label not in {"garbage", "gabrga", "noisy", "real?"}


def interp_to_taxis(old_t: np.ndarray, values: np.ndarray, new_t: np.ndarray) -> np.ndarray:
    old_t = np.asarray(old_t, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.isfinite(old_t) & np.isfinite(values)
    if mask.sum() < 2:
        return np.full_like(new_t, np.nan, dtype=np.float64)
    return np.interp(new_t, old_t[mask], values[mask], left=np.nan, right=np.nan)


def find_video_offset(raw: dict) -> float:
    bitstart = np.asarray(raw["bitcode_bitstart"], dtype=np.float64).reshape(-1)
    fs = float(raw["fs"])
    if bitstart.size == 0 or not np.isfinite(fs) or fs <= 0:
        return 0.5
    return robust_mode(bitstart) / fs - robust_mode(raw["events"]["bitStart"])


def feature_xy(raw: dict, view_idx: int, feature_name: str, align_times: np.ndarray, time_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trials = raw["traj"][view_idx]
    taxis = time_vec.copy()
    vidshift = find_video_offset(raw)
    xpos = np.full((time_vec.size, len(trials)), np.nan, dtype=np.float64)
    ypos = np.full((time_vec.size, len(trials)), np.nan, dtype=np.float64)

    for trix, trial in enumerate(trials):
        if not math.isfinite(trial["n_dropped"]):
            continue
        if feature_name not in trial["feat_names"]:
            continue
        feat_idx = trial["feat_names"].index(feature_name)
        ts = np.asarray(trial["ts"][:, :2, feat_idx], dtype=np.float64)
        if ts.ndim != 2 or ts.shape[0] == 0:
            continue
        frame_times = trial["frame_times"]
        if frame_times is None or frame_times.size == 0 or np.all(~np.isfinite(frame_times)):
            frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0
        old_t = frame_times - vidshift - align_times[trix]
        xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
        ypos[:, trix] = interp_to_taxis(old_t, ts[:, 1], taxis)

        if "tongue" not in feature_name:
            xpos[:, trix] = nearest_fill_1d(xpos[:, trix])
            ypos[:, trix] = nearest_fill_1d(ypos[:, trix])

    return xpos, ypos


def feature_speed(xpos: np.ndarray, ypos: np.ndarray, feature_name: str) -> np.ndarray:
    n_time, n_trials = xpos.shape
    xvel = np.full((n_time, n_trials), np.nan, dtype=np.float64)
    yvel = np.full((n_time, n_trials), np.nan, dtype=np.float64)

    for i in range(n_trials):
        tsinterp = np.column_stack([xpos[:, i], ypos[:, i]])
        if not np.isfinite(tsinterp).any():
            xv = np.zeros((n_time,), dtype=np.float64)
            yv = np.zeros((n_time,), dtype=np.float64)
            basederiv = np.array([0.0, 0.0], dtype=np.float64)
        else:
            if n_time > 1:
                deriv = np.diff(tsinterp, axis=0)
                if np.isfinite(deriv).any():
                    basederiv = np.nanmedian(deriv, axis=0)
                else:
                    basederiv = np.array([0.0, 0.0], dtype=np.float64)
            else:
                basederiv = np.array([0.0, 0.0], dtype=np.float64)
            basederiv = np.nan_to_num(basederiv, nan=0.0)
            xv = np.gradient(tsinterp[:, 0])
            yv = np.gradient(tsinterp[:, 1])
        if "tongue" not in feature_name:
            xv = xv - basederiv[0]
            yv = yv - basederiv[1]
            xv = nearest_fill_1d(xv)
            yv = nearest_fill_1d(yv)
        else:
            xv = np.nan_to_num(xv, nan=0.0)
            yv = np.nan_to_num(yv, nan=0.0)
        xvel[:, i] = xv
        yvel[:, i] = yv
    return np.sqrt(xvel**2 + yvel**2)


def aligned_motion_energy(raw: dict, align_times: np.ndarray, time_vec: np.ndarray) -> np.ndarray:
    me_trials = raw["motion_energy"]
    side_trials = raw["traj"][0]
    vidshift = find_video_offset(raw)
    out = np.full((time_vec.size, len(me_trials)), np.nan, dtype=np.float64)
    for trix, me in enumerate(me_trials):
        me = np.asarray(me, dtype=np.float64).reshape(-1)
        if me.size == 0:
            continue
        frame_times = side_trials[trix]["frame_times"] if trix < len(side_trials) else None
        if frame_times is None or frame_times.size == 0 or np.all(~np.isfinite(frame_times)):
            old_t = (np.arange(me.size, dtype=np.float64) + 1.0) / 400.0 - 0.5 - align_times[trix]
        else:
            old_t = frame_times - vidshift - align_times[trix]
        out[:, trix] = interp_to_taxis(old_t, me, time_vec)
        out[:, trix] = nearest_fill_1d(out[:, trix])
    return out


def compute_unit_trial_matrix(unit: dict, go_cue: np.ndarray, trial_to_pos: np.ndarray, n_sel: int, time_edges: np.ndarray) -> np.ndarray:
    aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
    trial_pos = trial_to_pos[unit["trial"] - 1]
    keep = (
        (trial_pos >= 0)
        & np.isfinite(aligned)
        & (aligned >= time_edges[0])
        & (aligned < time_edges[-1])
    )
    mat = np.zeros((n_sel, time_edges.size - 1), dtype=np.float64)
    if np.any(keep):
        bins = np.floor((aligned[keep] - time_edges[0]) / DT).astype(np.int64)
        np.add.at(mat, (trial_pos[keep], bins), 1.0 / DT)
    mat = my_smooth(mat.T, SMOOTH, BCTYPE).T
    return mat


def summarize_threshold(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.nanpercentile(values, 50.0))


def session_valid_trial_mask(raw: dict) -> np.ndarray:
    return (
        (raw["stim_enable"] == 0)
        & (raw["early"] == 0)
        & ((raw["hit"] == 1) | (raw["miss"] == 1))
        & ((raw["R"] == 1) | (raw["L"] == 1))
    )


def plot_processing(session_id: str, time_vec: np.ndarray, session_out: dict, outdir: Path) -> None:
    if not session_out["neural"]:
        return
    trial_idx = 0
    neural = session_out["neural"][trial_idx]
    cont = session_out["continuous"]
    binary = session_out["output"][trial_idx]

    fig, axes = plt.subplots(3, 2, figsize=(16, 11))
    ax = axes[0, 0]
    nshow = min(40, neural.shape[0])
    ax.imshow(neural[:nshow], aspect="auto", cmap="viridis", extent=[time_vec[0], time_vec[-1], nshow, 0])
    ax.set_title("Neural activity (sample trial)")
    ax.set_xlabel("Time from go cue (s)")
    ax.set_ylabel("Neuron")

    ax = axes[0, 1]
    ax.plot(time_vec, session_out["input"][trial_idx][0], color="black", lw=1.5)
    ax.set_title("Decoder input")
    ax.set_xlabel("Time from go cue (s)")
    ax.set_ylabel("Seconds")

    ax = axes[1, 0]
    ax.hist(cont["tongue"].reshape(-1), bins=60, alpha=0.5, label=f"tongue @ {cont['tongue_thr']:.3f}")
    ax.hist(cont["paw"].reshape(-1), bins=60, alpha=0.5, label=f"paw @ {cont['paw_thr']:.3f}")
    ax.hist(cont["motion"].reshape(-1), bins=60, alpha=0.5, label=f"motion @ {cont['motion_thr']:.3f}")
    ax.set_title("Session thresholds")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(time_vec, cont["tongue"][:, trial_idx], label="tongue")
    ax.plot(time_vec, cont["paw"][:, trial_idx], label="paw")
    ax.plot(time_vec, cont["motion"][:, trial_idx], label="motion")
    ax.set_title("Continuous outputs (sample trial)")
    ax.set_xlabel("Time from go cue (s)")
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    ax.step(time_vec, binary[3], where="mid", label="tongue_bin")
    ax.step(time_vec, binary[4], where="mid", label="paw_bin")
    ax.step(time_vec, binary[5], where="mid", label="motion_bin")
    ax.set_ylim(-0.2, 1.2)
    ax.set_title("Discretized continuous outputs")
    ax.set_xlabel("Time from go cue (s)")
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    ax.step(time_vec, binary[0], where="mid", label="lick_direction")
    ax.step(time_vec, binary[1], where="mid", label="context")
    ax.step(time_vec, binary[2], where="mid", label="outcome")
    ax.set_ylim(-0.2, 1.2)
    ax.set_title("Trial labels as time series")
    ax.set_xlabel("Time from go cue (s)")
    ax.legend(fontsize=8)

    fig.suptitle(session_id)
    fig.tight_layout()
    fig.savefig(outdir / f"processing_{session_id}.png", dpi=150)
    plt.close(fig)


def convert_one_session(spec: SessionSpec, show_processing: bool, outdir: Path) -> dict | None:
    t0 = time.time()
    raw = load_session(spec)
    valid = session_valid_trial_mask(raw)
    selected_trials = np.flatnonzero(valid)

    covered_trial_max = [
        int(np.nanmax(unit["trial"]))
        for unit in raw["units"]
        if good_quality(unit["quality"]) and np.asarray(unit["trial"]).size
    ]
    if covered_trial_max:
        max_neural_trial = min(raw["R"].size, max(covered_trial_max))
        selected_trials = selected_trials[selected_trials + 1 <= max_neural_trial]

    if selected_trials.size < 2:
        log(f"SKIP {spec.session_id}: only {selected_trials.size} valid hit/miss non-stim non-early trials")
        return None

    time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
    time_vec = time_edges[:-1] + DT / 2.0

    trial_to_pos = np.full(raw["R"].size, -1, dtype=np.int64)
    trial_to_pos[selected_trials] = np.arange(selected_trials.size, dtype=np.int64)

    kept_units = 0
    neural_trials = [[] for _ in range(selected_trials.size)]
    for unit in raw["units"]:
        if not good_quality(unit["quality"]):
            continue
        unit_mat = compute_unit_trial_matrix(unit, raw["events"]["goCue"], trial_to_pos, selected_trials.size, time_edges)
        if float(unit_mat.mean()) <= LOW_FR_HZ:
            continue
        kept_units += 1
        unit_mat = unit_mat.astype(np.float32)
        for trix in range(selected_trials.size):
            neural_trials[trix].append(unit_mat[trix])

    if kept_units < 10:
        log(f"SKIP {spec.session_id}: only {kept_units} units after quality/FR filtering")
        return None

    tongue_pos = feature_xy(raw, 0, "tongue", raw["events"]["goCue"], time_vec)
    tongue_speed = feature_speed(*tongue_pos, "tongue")
    tongue_valid = np.isfinite(tongue_pos[0]) & np.isfinite(tongue_pos[1])
    tongue_speed = tongue_speed.astype(np.float64, copy=False)
    tongue_speed[~tongue_valid] = np.nan

    paw_speeds = []
    for paw_name in ("top_paw", "bottom_paw"):
        paw_pos = feature_xy(raw, 1, paw_name, raw["events"]["goCue"], time_vec)
        paw_speeds.append(feature_speed(*paw_pos, paw_name))
    paw_stack = np.stack(paw_speeds, axis=0)
    paw_count = np.sum(np.isfinite(paw_stack), axis=0)
    paw_sum = np.nansum(paw_stack, axis=0)
    paw_speed = np.divide(paw_sum, np.maximum(paw_count, 1), where=np.maximum(paw_count, 1) > 0)
    paw_speed = np.nan_to_num(paw_speed, nan=0.0)

    motion = aligned_motion_energy(raw, raw["events"]["goCue"], time_vec)
    motion = np.nan_to_num(motion, nan=0.0)

    tongue_sel = tongue_speed[:, selected_trials]
    paw_sel = paw_speed[:, selected_trials]
    motion_sel = motion[:, selected_trials]

    tongue_thr = summarize_threshold(tongue_sel)
    paw_thr = summarize_threshold(paw_sel)
    motion_thr = summarize_threshold(motion_sel)

    input_trials = []
    output_trials = []
    final_neural = []
    for local_idx, trial_idx in enumerate(selected_trials):
        neural_arr = np.stack(neural_trials[local_idx], axis=0).astype(np.float32)
        final_neural.append(neural_arr)
        input_trials.append(time_vec[None, :].astype(np.float32))

        lick_direction = np.int64(1 if raw["R"][trial_idx] == 1 else 0)
        context = np.int64(1 if raw["autowater"][trial_idx] == 0 else 0)
        outcome = np.int64(1 if raw["hit"][trial_idx] == 1 else 0)
        output_arr = np.vstack(
            [
                np.full(time_vec.size, lick_direction, dtype=np.int64),
                np.full(time_vec.size, context, dtype=np.int64),
                np.full(time_vec.size, outcome, dtype=np.int64),
                np.where(
                    np.isfinite(tongue_sel[:, local_idx]),
                    tongue_sel[:, local_idx] >= tongue_thr,
                    0,
                ).astype(np.int64),
                (paw_sel[:, local_idx] >= paw_thr).astype(np.int64),
                (motion_sel[:, local_idx] >= motion_thr).astype(np.int64),
            ]
        )
        output_trials.append(output_arr)

    session_out = {
        "session_id": spec.session_id,
        "subject": spec.subject,
        "neural": final_neural,
        "input": input_trials,
        "output": output_trials,
        "brain_region_idx": np.zeros((kept_units,), dtype=np.int64),
        "n_trials": len(final_neural),
        "n_units": kept_units,
        "format": raw["format"],
        "source_file": str(spec.session_path),
        "continuous": {
            "tongue": tongue_sel.astype(np.float32),
            "paw": paw_sel.astype(np.float32),
            "motion": motion_sel.astype(np.float32),
            "tongue_thr": tongue_thr,
            "paw_thr": paw_thr,
            "motion_thr": motion_thr,
        },
    }
    if show_processing:
        plot_processing(spec.session_id, time_vec, session_out, outdir)

    log(
        f"SESSION {spec.session_id}: kept {session_out['n_trials']} trials, "
        f"{session_out['n_units']} units, format={raw['format']}, "
        f"time={time.time() - t0:.2f}s"
    )
    return session_out


def build_dataset(session_specs: list[SessionSpec], show_processing: bool, outdir: Path) -> dict:
    neural = []
    inputs = []
    outputs = []
    subject_names: list[str] = []
    subject_index = []
    brain_region_idx = []
    session_info = []

    for spec in session_specs:
        session = convert_one_session(spec, show_processing=show_processing, outdir=outdir)
        if session is None:
            continue

        neural.append(session["neural"])
        inputs.append(session["input"])
        outputs.append(session["output"])
        if session["subject"] not in subject_names:
            subject_names.append(session["subject"])
        subject_index.append(subject_names.index(session["subject"]))
        brain_region_idx.append(session["brain_region_idx"])
        session_info.append(
            {
                "session_id": session["session_id"],
                "subject": session["subject"],
                "n_trials": session["n_trials"],
                "n_units": session["n_units"],
                "format": session["format"],
                "source_file": session["source_file"],
            }
        )

    data = {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subject_names,
        "subject_idx": np.asarray(subject_index, dtype=np.int64),
        "brain_regions": ["ALM"],
        "brain_region_idx": brain_region_idx,
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
            ["below_session_median", "at_or_above_session_median"],
            ["below_session_median", "at_or_above_session_median"],
            ["below_session_median", "at_or_above_session_median"],
        ],
        "metadata": {
            "task_description": (
                "ALM spike-rate decoding aligned to go cue, predicting lick direction, "
                "behavioral context, outcome, and binarized tongue/paw/motion variables."
            ),
            "time_bin_size": float(DT * 1000.0),
            "temporal_alignment_event": "go cue onset (stored bp.ev.goCue for every trial)",
            "off_start": float(TMIN),
            "off_end": float(TMAX),
            "neural_processing": "Selected ALM probe, bad-label exclusion, goCue alignment, 5 ms binning, causal Gaussian smoothing, FR > 1 Hz",
            "trial_filter": "~stim.enable & ~early & (hit | miss)",
            "session_selection": "Reference code ALM session list intersected with available data files",
            "video_processing": "Reference-style interpolation to neural time base with video offset correction",
            "session_info": session_info,
        },
    }
    return data


def print_summary(data: dict) -> None:
    n_sessions = len(data["neural"])
    n_trials = sum(len(s) for s in data["neural"])
    n_units = sum(s[0].shape[0] for s in data["neural"]) if n_sessions else 0
    log(f"SUMMARY sessions={n_sessions} trials={n_trials} total_units={n_units}")
    if n_sessions:
        trial_counts = [len(s) for s in data["neural"]]
        unit_counts = [s[0].shape[0] for s in data["neural"]]
        log(
            "SUMMARY trials/session "
            f"min={min(trial_counts)} mean={np.mean(trial_counts):.2f} max={max(trial_counts)}"
        )
        log(
            "SUMMARY units/session "
            f"min={min(unit_counts)} mean={np.mean(unit_counts):.2f} max={max(unit_counts)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert neural decoder dataset.")
    parser.add_argument("outpicklefile", type=str, help="Output pickle path")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all reference-selected sessions (default)")
    mode.add_argument("--sample", action="store_true", help="Process only 2 sessions")
    parser.add_argument("--show-processing", action="store_true", help="Save processing plots for converted sessions")
    args = parser.parse_args()

    root = Path(".").resolve()
    code_dir = root / "code"
    data_dir = root / "data"
    out_path = Path(args.outpicklefile).resolve()

    t0 = time.time()
    specs = parse_reference_session_specs(code_dir, data_dir)
    log(f"Found {len(specs)} reference-selected sessions available in data/")

    if args.sample:
        specs = choose_sample_specs(specs)
        log("Sample mode: " + ", ".join(s.session_id for s in specs))
    else:
        log("Full mode")

    data = build_dataset(specs, show_processing=args.show_processing, outdir=root)
    print_summary(data)

    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    log(f"Wrote {out_path}")
    log(f"TOTAL TIME {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
