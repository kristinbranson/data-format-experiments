#!/usr/bin/env python3
"""Convert the Birnbaum et al. ALM ephys/video data into decoder format."""

from __future__ import annotations

import argparse
import math
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.interpolate import interp1d
from scipy.signal.windows import gaussian

import mat73


APP = Path("/app")
CODE = APP / "code"
DATA = APP / "data"

FIXED_SCRIPT = CODE / "Scripts" / "Figure 3" / "Figure3h.m"
RANDOMIZED_SCRIPT = CODE / "Scripts" / "Figure 3" / "Figure3h.m"
LOADER_DIR = CODE / "DataLoadingScripts" / "Recording and video"

ALIGN_EVENT = "goCue"
TMIN = -2.5
TMAX = 2.5
DT = 0.01
SMOOTH_BINS = 15
LOW_FR = 1.0
MIN_UNITS_PER_SESSION = 10
MIN_TRIALS_PER_SESSION = 2
BAD_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}

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
    ["WC", "DR"],
    ["incorrect", "correct", "ignore"],
    ["low", "high", "not_visible"],
    ["low", "high", "not_visible"],
    ["low", "high", "no_video"],
]


@dataclass(frozen=True)
class SessionSpec:
    cohort: str
    folder: str
    animal: str
    date: str
    probes: tuple[int, ...]

    @property
    def session_id(self) -> str:
        return f"{self.animal}_{self.date}"

    @property
    def data_path(self) -> Path:
        return DATA / self.folder / f"data_structure_{self.animal}_{self.date}.mat"

    @property
    def motion_path(self) -> Path:
        return DATA / self.folder / f"motionEnergy_{self.animal}_{self.date}.mat"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert neural dataset to decoder format.")
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process a small 2-session sample.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Plot processing diagnostics for up to two sessions.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    specs = get_session_specs(sample=args.sample)
    print(f"Selected {len(specs)} sessions for conversion.")

    converted = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": [],
        "subject_idx": [],
        "brain_regions": ["ALM"],
        "brain_region_idx": [],
        "input_names": ["time_from_go_cue_s"],
        "output_names": OUTPUT_NAMES,
        "output_values": OUTPUT_VALUES,
        "metadata": {
            "task_description": (
                "ALM electrophysiology aligned to go cue; decode lick direction, "
                "behavioral context, outcome, and discretized tongue/paw/motion signals."
            ),
            "time_bin_size": DT * 1000.0,
            "temporal_alignment_event": "Go cue onset",
            "off_start": TMIN,
            "off_end": TMAX,
            "alignment_event_field": f"bp.ev.{ALIGN_EVENT}",
            "neural_smoothing": {
                "method": "causal_gaussian",
                "window_bins": SMOOTH_BINS,
                "boundary_condition": "reflect",
            },
            "trial_filter": "~stim.enable & ~early & (hit|miss|no)",
            "unit_filter": {
                "quality_excluded": sorted(BAD_QUALITIES),
                "min_mean_firing_rate_hz": LOW_FR,
                "min_units_per_session": MIN_UNITS_PER_SESSION,
            },
            "session_info": [],
        },
    }

    subject_to_idx: dict[str, int] = {}

    for sess_num, spec in enumerate(specs, start=1):
        session_t0 = time.perf_counter()
        print(f"[{sess_num:02d}/{len(specs):02d}] Processing {spec.session_id} ({spec.cohort})")

        obj = load_mat_file(spec.data_path)["obj"]
        processed = process_session(spec, obj, show_processing=args.show_processing and sess_num <= 2)
        if processed is None:
            print(f"  skipped {spec.session_id}")
            continue

        subject_idx = subject_to_idx.setdefault(spec.animal, len(subject_to_idx))
        converted["neural"].append(processed["neural_trials"])
        converted["input"].append(processed["input_trials"])
        converted["output"].append(processed["output_trials"])
        converted["subject_idx"].append(subject_idx)
        converted["brain_region_idx"].append(processed["brain_region_idx"])
        converted["metadata"]["session_info"].append(processed["session_info"])

        print(
            f"  kept {len(processed['neural_trials'])} trials, "
            f"{processed['brain_region_idx'].shape[0]} units, "
            f"{processed['neural_trials'][0].shape[1]} bins "
            f"in {time.perf_counter() - session_t0:.1f}s"
        )

    converted["subjects"] = [subject for subject, _ in sorted(subject_to_idx.items(), key=lambda kv: kv[1])]
    converted["subject_idx"] = np.asarray(converted["subject_idx"], dtype=np.int64)

    with args.outpicklefile.open("wb") as f:
        pickle.dump(converted, f)

    print(
        f"Saved {len(converted['neural'])} sessions to {args.outpicklefile} "
        f"in {time.perf_counter() - t0:.1f}s"
    )


def get_session_specs(sample: bool) -> list[SessionSpec]:
    fixed_animals = parse_loader_animals(FIXED_SCRIPT, marker="fixmeta")
    randomized_animals = parse_loader_animals(RANDOMIZED_SCRIPT, marker="randmeta")

    specs: list[SessionSpec] = []
    for animal in fixed_animals:
        specs.extend(parse_loader_sessions(animal, cohort="fixed", folder="Ephys_Behavior"))
    for animal in randomized_animals:
        specs.extend(
            parse_loader_sessions(animal, cohort="randomized", folder="RandomizedDelay_Ephys_Behavior")
        )

    if not sample:
        return specs

    first_fixed = next(spec for spec in specs if spec.cohort == "fixed")
    first_randomized = next(spec for spec in specs if spec.cohort == "randomized")
    return [first_fixed, first_randomized]


def parse_loader_animals(script_path: Path, marker: str) -> list[str]:
    text = script_path.read_text()
    animals: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("%"):
            continue
        match = re.search(rf"{re.escape(marker)}\s*=\s*load([A-Za-z0-9]+)_ALMVideo\(", line)
        if match:
            animals.append(match.group(1))
    return animals


def parse_loader_sessions(animal: str, cohort: str, folder: str) -> list[SessionSpec]:
    text = (LOADER_DIR / f"load{animal}_ALMVideo.m").read_text()
    sessions: list[SessionSpec] = []
    current_date: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("%"):
            continue
        m_date = re.search(r"date = '([^']+)'", line)
        if m_date:
            current_date = m_date.group(1)
        m_probe = re.search(r"probe = (\[[^\]]+\]|\d+);", line)
        if m_probe and current_date:
            probes = tuple(int(x) for x in re.findall(r"\d+", m_probe.group(1)))
            sessions.append(SessionSpec(cohort=cohort, folder=folder, animal=animal, date=current_date, probes=probes))
            current_date = None
    return sessions


def load_mat_file(path: Path) -> dict[str, Any]:
    try:
        return mat73.loadmat(str(path))
    except Exception:
        return scipy.io.loadmat(str(path), simplify_cells=True)


def gget(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return list(value.ravel())
        return list(value)
    return [value]


def normalize_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def flatten_dicts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        out: list[dict[str, Any]] = []
        for item in value:
            out.extend(flatten_dicts(item))
        return out
    if isinstance(value, np.ndarray) and value.dtype == object:
        out: list[dict[str, Any]] = []
        for item in value.ravel():
            out.extend(flatten_dicts(item))
        return out
    return []


def is_flat_unit_list(clu: Any) -> bool:
    return (
        isinstance(clu, list)
        and len(clu) > 0
        and isinstance(clu[0], dict)
        and "tm" in clu[0]
        and "quality" in clu[0]
        and not isinstance(gget(clu[0], "quality"), (list, np.ndarray))
    )


def split_probe_struct(probe: dict[str, Any]) -> list[dict[str, Any]]:
    qualities = as_list(gget(probe, "quality"))
    tm_list = as_list(gget(probe, "tm"))
    trial_list = as_list(gget(probe, "trial"))
    trialtm_list = as_list(gget(probe, "trialtm"))
    site_list = as_list(gget(probe, "site", [None] * len(qualities)))
    n_units = len(qualities)

    units: list[dict[str, Any]] = []
    for i in range(n_units):
        units.append(
            {
                "quality": qualities[i] if i < len(qualities) else "",
                "tm": np.asarray(tm_list[i]).reshape(-1) if i < len(tm_list) else np.array([], dtype=float),
                "trial": np.asarray(trial_list[i]).reshape(-1) if i < len(trial_list) else np.array([], dtype=float),
                "trialtm": (
                    np.asarray(trialtm_list[i]).reshape(-1) if i < len(trialtm_list) else np.array([], dtype=float)
                ),
                "site": site_list[i] if i < len(site_list) else None,
            }
        )
    return units


def extract_selected_units(obj: Any, probes: tuple[int, ...]) -> list[dict[str, Any]]:
    clu = gget(obj, "clu")
    if clu is None:
        return []

    if is_flat_unit_list(clu):
        return [unit for unit in clu if normalize_string(gget(unit, "quality")).lower() not in BAD_QUALITIES]

    if isinstance(clu, dict):
        units = split_probe_struct(clu)
        return [unit for unit in units if normalize_string(unit["quality"]).lower() not in BAD_QUALITIES]

    clu_list = as_list(clu)
    units: list[dict[str, Any]] = []
    for probe_num in probes:
        probe = clu_list[probe_num - 1]
        for unit in split_probe_struct(probe):
            if normalize_string(unit["quality"]).lower() in BAD_QUALITIES:
                continue
            units.append(unit)
    return units


def bool_array(bp: Any, field: str, n_trials: int) -> np.ndarray:
    values = gget(bp, field, None)
    if values is None:
        return np.zeros(n_trials, dtype=bool)
    values = np.asarray(values).reshape(-1)
    return values.astype(bool)


def stim_enable_array(bp: Any, n_trials: int) -> np.ndarray:
    stim = gget(bp, "stim", None)
    if stim is None:
        return np.zeros(n_trials, dtype=bool)
    enable = gget(stim, "enable", None)
    if enable is None:
        return np.zeros(n_trials, dtype=bool)
    return np.asarray(enable).reshape(-1).astype(bool)


def event_array(ev: Any, field: str) -> np.ndarray:
    arr = np.asarray(gget(ev, field)).reshape(-1)
    return arr.astype(float)


def get_trials_view(obj: Any, view_index: int) -> list[dict[str, Any]]:
    traj = gget(obj, "traj")
    if traj is None:
        return []
    view = as_list(traj)[view_index]
    if isinstance(view, dict):
        lengths = []
        for value in view.values():
            if isinstance(value, (list, tuple)):
                lengths.append(len(value))
            elif isinstance(value, np.ndarray) and value.dtype == object:
                lengths.append(value.size)
        n_trials = max(lengths) if lengths else 0
        trials: list[dict[str, Any]] = []
        for trial_idx in range(n_trials):
            trial_dict: dict[str, Any] = {}
            for key, value in view.items():
                if isinstance(value, list):
                    trial_dict[key] = value[trial_idx]
                elif isinstance(value, tuple):
                    trial_dict[key] = value[trial_idx]
                elif isinstance(value, np.ndarray) and value.dtype == object:
                    flat = list(value.ravel())
                    trial_dict[key] = flat[trial_idx]
                else:
                    trial_dict[key] = value
            trials.append(trial_dict)
        return trials
    return flatten_dicts(view)


def flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(flatten_strings(item))
        return out
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            out: list[str] = []
            for item in value.ravel():
                out.extend(flatten_strings(item))
            return out
        return [str(x) for x in value.ravel()]
    return [str(value)]


def feature_index(trial_view: dict[str, Any], feature_name: str) -> int | None:
    feat_names = flatten_strings(gget(trial_view, "featNames"))
    for idx, name in enumerate(feat_names):
        if name == feature_name:
            return idx
    return None


def session_time_axis() -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(TMIN, TMAX + DT * 0.5, DT)
    time_axis = edges[:-1] + DT / 2.0
    return edges, time_axis


def matlab_mode(values: np.ndarray) -> float:
    values = np.asarray(values).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    rounded = np.round(values, 9)
    unique, counts = np.unique(rounded, return_counts=True)
    return float(unique[np.argmax(counts)])


def find_video_offset(obj: Any) -> float:
    bp = gget(obj, "bp")
    ev = gget(bp, "ev")
    sglx = gget(obj, "sglx")
    bitcode = gget(sglx, "bitcode", None)
    fs = gget(sglx, "fs", None)
    if ev is None or bitcode is None or fs in (None, 0):
        return 0.0
    bit_start = event_array(ev, "bitStart")
    bitcode_start = np.asarray(gget(bitcode, "bitstart")).reshape(-1).astype(float)
    return matlab_mode(bitcode_start) / float(fs) - matlab_mode(bit_start)


def make_causal_kernel(window_bins: int) -> np.ndarray:
    if window_bins <= 1:
        return np.array([1.0], dtype=np.float64)
    std = (window_bins - 1) / (2.0 * 2.5)
    kernel = gaussian(window_bins, std=std).astype(np.float64)
    kernel[: window_bins // 2] = 0.0
    kernel /= kernel.sum()
    return kernel


GAUSSIAN_KERNEL = make_causal_kernel(SMOOTH_BINS)


def my_smooth_1d(values: np.ndarray, boundary_condition: str = "reflect") -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or SMOOTH_BINS <= 1:
        return values.astype(np.float32)
    if boundary_condition == "reflect":
        padded = np.concatenate([values[:SMOOTH_BINS], values])
        trim = SMOOTH_BINS
    elif boundary_condition == "zeropad":
        padded = np.concatenate([np.zeros(SMOOTH_BINS, dtype=np.float64), values])
        trim = SMOOTH_BINS
    else:
        padded = values
        trim = 0
    smoothed = np.convolve(padded, GAUSSIAN_KERNEL, mode="same")
    return smoothed[trim:].astype(np.float32)


def fill_nearest_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    if values.size == 0:
        return values
    valid = np.isfinite(values)
    if not valid.any():
        return values
    idx = np.arange(values.size)
    values[~valid] = np.interp(idx[~valid], idx[valid], values[valid])
    return values


def linear_interp(times: np.ndarray, values: np.ndarray, new_times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(times) & np.isfinite(values)
    if valid.sum() < 2:
        if valid.sum() == 1:
            return np.full(new_times.shape, values[valid][0], dtype=np.float64)
        return np.full(new_times.shape, np.nan, dtype=np.float64)
    interp = interp1d(
        times[valid],
        values[valid],
        kind="linear",
        bounds_error=False,
        fill_value=(values[valid][0], values[valid][-1]),
        assume_sorted=False,
    )
    return np.asarray(interp(new_times), dtype=np.float64)


def nearest_interp(times: np.ndarray, values: np.ndarray, new_times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(times) & np.isfinite(values)
    if valid.sum() == 0:
        return np.full(new_times.shape, np.nan, dtype=np.float64)
    if valid.sum() == 1:
        return np.full(new_times.shape, values[valid][0], dtype=np.float64)
    interp = interp1d(
        times[valid],
        values[valid],
        kind="nearest",
        bounds_error=False,
        fill_value=(values[valid][0], values[valid][-1]),
        assume_sorted=False,
    )
    return np.asarray(interp(new_times), dtype=np.float64)


def post_go_choice(lick_l: Any, lick_r: Any, go_time: float) -> int:
    left = event_list_after(lick_l, go_time)
    right = event_list_after(lick_r, go_time)
    t_left = left.min() if left.size else np.inf
    t_right = right.min() if right.size else np.inf
    if np.isinf(t_left) and np.isinf(t_right):
        return 2
    return 0 if t_left < t_right else 1


def event_list_after(value: Any, threshold: float) -> np.ndarray:
    if value is None:
        return np.array([], dtype=np.float64)
    arr = np.array(value, dtype=object).reshape(-1)
    cleaned = []
    for item in arr:
        if item is None:
            continue
        try:
            cleaned.append(float(item))
        except Exception:
            continue
    if not cleaned:
        return np.array([], dtype=np.float64)
    cleaned_arr = np.asarray(cleaned, dtype=np.float64)
    return cleaned_arr[cleaned_arr > threshold]


def load_motion_energy_source(obj: Any, spec: SessionSpec) -> tuple[list[Any] | None, bool]:
    if spec.motion_path.exists():
        me_struct = load_mat_file(spec.motion_path).get("me")
        return motion_energy_data_list(me_struct), True
    me_struct = gget(obj, "me", None)
    if me_struct is None:
        return None, False
    return motion_energy_data_list(me_struct), True


def motion_energy_data_list(me_struct: Any) -> list[Any] | None:
    if me_struct is None:
        return None
    if isinstance(me_struct, np.ndarray):
        return [item for item in me_struct.ravel()]
    if isinstance(me_struct, list):
        return me_struct
    data = gget(me_struct, "data", me_struct)
    if isinstance(data, dict):
        data = gget(data, "data", data)
    return as_list(data)


def trial_invalid_video(trial_view: dict[str, Any]) -> bool:
    dropped = gget(trial_view, "NdroppedFrames", None)
    if dropped is None:
        return False
    arr = np.asarray(dropped).reshape(-1)
    if arr.size == 0:
        return False
    return np.isnan(arr.astype(float)).all()


def trial_frame_times(trial_view: dict[str, Any], n_frames: int) -> np.ndarray:
    frame_times = gget(trial_view, "frameTimes", None)
    if frame_times is None:
        return np.arange(1, n_frames + 1, dtype=np.float64) / 400.0
    frame_times = np.asarray(frame_times, dtype=np.float64).reshape(-1)
    if frame_times.size == 0 or np.isnan(frame_times).all():
        return np.arange(1, n_frames + 1, dtype=np.float64) / 400.0
    return frame_times


def build_speed_trace(
    obj: Any,
    view_index: int,
    feature_names: list[str],
    go_times: np.ndarray,
    taxis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    trials_view = get_trials_view(obj, view_index)
    n_trials = len(go_times)
    speed = np.full((n_trials, taxis.size), np.nan, dtype=np.float32)
    visible = np.zeros((n_trials, taxis.size), dtype=bool)
    chosen_feature: str | None = None

    if len(trials_view) < n_trials:
        return speed, visible, None

    for candidate in feature_names:
        first_idx = feature_index(trials_view[0], candidate)
        if first_idx is not None:
            chosen_feature = candidate
            break
    if chosen_feature is None:
        return speed, visible, None

    vidshift = find_video_offset(obj)
    feat_is_tongue = "tongue" in chosen_feature

    for trial_idx in range(n_trials):
        trial_view = trials_view[trial_idx]
        if trial_invalid_video(trial_view):
            continue
        feat_idx = feature_index(trial_view, chosen_feature)
        ts = np.asarray(gget(trial_view, "ts", None), dtype=np.float64)
        if feat_idx is None or ts.ndim != 3 or ts.shape[2] <= feat_idx:
            continue
        coords = ts[:, :2, feat_idx]
        n_frames = coords.shape[0]
        frame_times = trial_frame_times(trial_view, n_frames) - vidshift - go_times[trial_idx]

        x = coords[:, 0]
        y = coords[:, 1]
        frame_visible = np.isfinite(x) & np.isfinite(y)
        if not frame_visible.any():
            continue

        x_interp = linear_interp(frame_times, x, taxis)
        y_interp = linear_interp(frame_times, y, taxis)
        vis_interp = nearest_interp(frame_times, frame_visible.astype(float), taxis) >= 0.5

        x_interp[~vis_interp] = np.nan
        y_interp[~vis_interp] = np.nan

        if feat_is_tongue:
            x_vel = np.gradient(x_interp)
            y_vel = np.gradient(y_interp)
            x_vel[~np.isfinite(x_vel)] = 0.0
            y_vel[~np.isfinite(y_vel)] = 0.0
        else:
            x_filled = fill_nearest_1d(x_interp)
            y_filled = fill_nearest_1d(y_interp)
            if not np.isfinite(x_filled).any() or not np.isfinite(y_filled).any():
                continue
            stacked = np.column_stack([x_filled, y_filled])
            basederiv = np.nanmedian(np.diff(stacked, axis=0), axis=0)
            baseline = basederiv[0] if np.isfinite(basederiv[0]) else 0.0
            x_vel = np.gradient(x_filled) - baseline
            y_vel = np.gradient(y_filled) - baseline
            x_vel = fill_nearest_1d(x_vel)
            y_vel = fill_nearest_1d(y_vel)

        speed[trial_idx] = np.sqrt(x_vel ** 2 + y_vel ** 2).astype(np.float32)
        visible[trial_idx] = vis_interp

    return speed, visible, chosen_feature


def build_motion_energy_trace(
    obj: Any,
    spec: SessionSpec,
    go_times: np.ndarray,
    taxis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_trials = len(go_times)
    aligned = np.full((n_trials, taxis.size), np.nan, dtype=np.float32)
    valid = np.zeros((n_trials, taxis.size), dtype=bool)
    motion_trials, available = load_motion_energy_source(obj, spec)
    if not available or motion_trials is None:
        return aligned, valid

    side_trials = get_trials_view(obj, 0)
    vidshift = find_video_offset(obj)

    for trial_idx in range(min(n_trials, len(motion_trials))):
        if trial_idx >= len(side_trials):
            continue
        if trial_invalid_video(side_trials[trial_idx]):
            continue
        raw = motion_trials[trial_idx]
        if raw is None:
            continue
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)
        if raw.size < 2 or np.isnan(raw).all():
            continue
        frame_times = trial_frame_times(side_trials[trial_idx], raw.size) - vidshift - go_times[trial_idx]
        interp = linear_interp(frame_times, raw, taxis)
        interp = fill_nearest_1d(interp)
        aligned[trial_idx] = interp.astype(np.float32)
        valid[trial_idx] = np.isfinite(interp)
    return aligned, valid


def discretize_session_signal(
    signal: np.ndarray,
    visible: np.ndarray,
    missing_code: int,
    kept_trials: np.ndarray,
) -> tuple[np.ndarray, float]:
    out = np.full(signal.shape, missing_code, dtype=np.int64)
    keep_signal = signal[kept_trials]
    keep_visible = visible[kept_trials]
    values = keep_signal[keep_visible]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return out, float("nan")
    threshold = float(np.nanpercentile(values, 50))
    visible_and_low = visible & np.isfinite(signal) & (signal < threshold)
    visible_and_high = visible & np.isfinite(signal) & ~visible_and_low
    out[visible_and_low] = 0
    out[visible_and_high] = 1
    return out, threshold


def low_fr_condition_indices(bp: Any, n_trials: int) -> list[np.ndarray]:
    hit = bool_array(bp, "hit", n_trials)
    miss = bool_array(bp, "miss", n_trials)
    no = bool_array(bp, "no", n_trials)
    right = bool_array(bp, "R", n_trials)
    left = bool_array(bp, "L", n_trials)
    autowater = bool_array(bp, "autowater", n_trials)
    early = bool_array(bp, "early", n_trials)
    stim = stim_enable_array(bp, n_trials)
    masks = [
        hit | miss | no,
        right & hit & ~stim & ~autowater & ~early,
        left & hit & ~stim & ~autowater & ~early,
        right & miss & ~stim & ~autowater & ~early,
        left & miss & ~stim & ~autowater & ~early,
        right & no & ~stim & ~autowater & ~early,
        left & no & ~stim & ~autowater & ~early,
        hit & ~stim & ~autowater & ~early,
    ]
    return [np.flatnonzero(mask) for mask in masks]


def process_session(spec: SessionSpec, obj: Any, show_processing: bool) -> dict[str, Any] | None:
    bp = gget(obj, "bp")
    ev = gget(bp, "ev")
    n_trials = int(gget(bp, "Ntrials", 0) or 0)
    if n_trials == 0:
        return None

    edges, taxis = session_time_axis()
    go_times = event_array(ev, ALIGN_EVENT)
    units = extract_selected_units(obj, spec.probes)
    if not units:
        return None

    trialdat = np.zeros((len(units), n_trials, taxis.size), dtype=np.float32)
    neural_covered = np.zeros(n_trials, dtype=bool)
    for unit_idx, unit in enumerate(units):
        trial_ids = np.asarray(unit["trial"], dtype=np.int64).reshape(-1) - 1
        trial_times = np.asarray(unit["trialtm"], dtype=np.float64).reshape(-1)
        if trial_ids.size == 0:
            continue
        valid = (trial_ids >= 0) & (trial_ids < n_trials) & np.isfinite(trial_times)
        trial_ids = trial_ids[valid]
        trial_times = trial_times[valid]
        neural_covered[trial_ids] = True
        aligned_times = trial_times - go_times[trial_ids]
        for tr in np.unique(trial_ids):
            tr_mask = trial_ids == tr
            counts, _ = np.histogram(aligned_times[tr_mask], bins=edges)
            rates = counts.astype(np.float64) / DT
            trialdat[unit_idx, tr] = my_smooth_1d(rates, boundary_condition="reflect")

    condition_indices = low_fr_condition_indices(bp, n_trials)
    psth = np.zeros((trialdat.shape[0], taxis.size, len(condition_indices)), dtype=np.float32)
    for cond_idx, trial_idx in enumerate(condition_indices):
        if trial_idx.size == 0:
            continue
        psth[:, :, cond_idx] = np.nanmean(trialdat[:, trial_idx, :], axis=1)
    mean_fr = np.nanmean(psth, axis=(1, 2))
    keep_units = mean_fr > LOW_FR
    trialdat = trialdat[keep_units]
    if trialdat.shape[0] < MIN_UNITS_PER_SESSION:
        return None

    hit = bool_array(bp, "hit", n_trials)
    miss = bool_array(bp, "miss", n_trials)
    no = bool_array(bp, "no", n_trials)
    early = bool_array(bp, "early", n_trials)
    stim = stim_enable_array(bp, n_trials)
    autowater = bool_array(bp, "autowater", n_trials)
    keep_trials = np.flatnonzero((hit | miss | no) & ~early & ~stim & neural_covered)
    if keep_trials.size < MIN_TRIALS_PER_SESSION:
        return None

    lick_l = as_list(gget(ev, "lickL"))
    lick_r = as_list(gget(ev, "lickR"))
    lick_direction = np.array(
        [post_go_choice(lick_l[tr], lick_r[tr], go_times[tr]) for tr in range(n_trials)],
        dtype=np.int64,
    )
    context = np.where(autowater, 0, 1).astype(np.int64)
    outcome = np.full(n_trials, 2, dtype=np.int64)
    outcome[miss] = 0
    outcome[hit] = 1
    outcome[no] = 2

    tongue_speed, tongue_visible, tongue_feature = build_speed_trace(
        obj, view_index=0, feature_names=["tongue", "left_tongue", "right_tongue"], go_times=go_times, taxis=taxis
    )
    paw_speed, paw_visible, paw_feature = build_speed_trace(
        obj, view_index=1, feature_names=["top_paw", "bottom_paw"], go_times=go_times, taxis=taxis
    )
    motion_energy, motion_valid = build_motion_energy_trace(obj, spec, go_times, taxis)

    tongue_disc, tongue_thresh = discretize_session_signal(tongue_speed, tongue_visible, 2, keep_trials)
    paw_disc, paw_thresh = discretize_session_signal(paw_speed, paw_visible, 2, keep_trials)
    me_disc, me_thresh = discretize_session_signal(motion_energy, motion_valid, 2, keep_trials)

    neural_trials: list[np.ndarray] = []
    input_trials: list[np.ndarray] = []
    output_trials: list[np.ndarray] = []
    for tr in keep_trials:
        neural_trials.append(trialdat[:, tr, :].astype(np.float32, copy=False))
        input_trials.append(taxis[np.newaxis, :].astype(np.float32, copy=False))
        trial_output = np.vstack(
            [
                np.full(taxis.size, lick_direction[tr], dtype=np.int64),
                np.full(taxis.size, context[tr], dtype=np.int64),
                np.full(taxis.size, outcome[tr], dtype=np.int64),
                tongue_disc[tr],
                paw_disc[tr],
                me_disc[tr],
            ]
        )
        output_trials.append(trial_output)

    if show_processing:
        plot_processing(
            spec=spec,
            taxis=taxis,
            neural_trial=neural_trials[0],
            tongue_speed=tongue_speed[keep_trials[0]],
            tongue_disc=tongue_disc[keep_trials[0]],
            paw_speed=paw_speed[keep_trials[0]],
            paw_disc=paw_disc[keep_trials[0]],
            motion_energy=motion_energy[keep_trials[0]],
            motion_disc=me_disc[keep_trials[0]],
            thresholds={"tongue": tongue_thresh, "paw": paw_thresh, "motion": me_thresh},
        )

    session_info = {
        "session_id": spec.session_id,
        "cohort": spec.cohort,
        "folder": spec.folder,
        "animal": spec.animal,
        "date": spec.date,
        "selected_probes": list(spec.probes),
        "n_trials_raw": n_trials,
        "n_trials_kept": int(keep_trials.size),
        "n_units_after_quality": int(len(units)),
        "n_units_kept": int(trialdat.shape[0]),
        "time_bins": int(taxis.size),
        "tongue_feature": tongue_feature,
        "paw_feature": paw_feature,
        "thresholds": {"tongue": tongue_thresh, "paw": paw_thresh, "motion_energy": me_thresh},
    }

    return {
        "neural_trials": neural_trials,
        "input_trials": input_trials,
        "output_trials": output_trials,
        "brain_region_idx": np.zeros(trialdat.shape[0], dtype=np.int64),
        "session_info": session_info,
    }


def plot_processing(
    spec: SessionSpec,
    taxis: np.ndarray,
    neural_trial: np.ndarray,
    tongue_speed: np.ndarray,
    tongue_disc: np.ndarray,
    paw_speed: np.ndarray,
    paw_disc: np.ndarray,
    motion_energy: np.ndarray,
    motion_disc: np.ndarray,
    thresholds: dict[str, float],
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    im = axes[0].imshow(neural_trial, aspect="auto", origin="lower", extent=[taxis[0], taxis[-1], 0, neural_trial.shape[0]])
    axes[0].set_title(f"{spec.session_id} neural sample trial")
    axes[0].set_ylabel("Neuron")
    fig.colorbar(im, ax=axes[0], label="Hz")

    axes[1].plot(taxis, tongue_speed, label="tongue speed", color="tab:red")
    if np.isfinite(thresholds["tongue"]):
        axes[1].axhline(thresholds["tongue"], linestyle="--", color="tab:red", alpha=0.6)
    axes[1].plot(taxis, paw_speed, label="paw speed", color="tab:green")
    if np.isfinite(thresholds["paw"]):
        axes[1].axhline(thresholds["paw"], linestyle="--", color="tab:green", alpha=0.6)
    axes[1].plot(taxis, motion_energy, label="motion energy", color="tab:blue")
    if np.isfinite(thresholds["motion"]):
        axes[1].axhline(thresholds["motion"], linestyle="--", color="tab:blue", alpha=0.6)
    axes[1].legend(loc="upper right")
    axes[1].set_ylabel("Continuous")

    axes[2].imshow(np.vstack([tongue_disc, paw_disc, motion_disc]), aspect="auto", origin="lower", extent=[taxis[0], taxis[-1], 0, 3], vmin=0, vmax=2)
    axes[2].set_yticks([0.5, 1.5, 2.5])
    axes[2].set_yticklabels(["tongue", "paw", "motion"])
    axes[2].set_ylabel("Discrete")

    axes[3].hist(tongue_speed[np.isfinite(tongue_speed)], bins=50, alpha=0.5, color="tab:red", label="tongue")
    axes[3].hist(paw_speed[np.isfinite(paw_speed)], bins=50, alpha=0.5, color="tab:green", label="paw")
    axes[3].hist(motion_energy[np.isfinite(motion_energy)], bins=50, alpha=0.5, color="tab:blue", label="motion")
    axes[3].legend(loc="upper right")
    axes[3].set_xlabel("Time from go cue (s)")
    axes[3].set_ylabel("Histogram")

    for ax in axes:
        ax.axvline(0.0, linestyle="--", color="k", alpha=0.5)
    fig.tight_layout()
    fig.savefig(APP / f"processing_{spec.session_id}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
