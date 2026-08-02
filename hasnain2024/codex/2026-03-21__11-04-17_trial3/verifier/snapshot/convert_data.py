#!/usr/bin/env python3
import argparse
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pymatreader import read_mat
from scipy.interpolate import interp1d

from decoder import verify_data_format


ROOT = Path("/app")
DATA_ROOT = ROOT / "data"
LOADER_ROOT = ROOT / "code" / "DataLoadingScripts" / "Recording and video"

DT = 0.005
TMIN = -2.5
TMAX = 2.5
SMOOTH = 15
BCTYPE = "reflect"
LOW_FR = 1.0
MIN_UNITS_PER_SESSION = 10
ADVANCE_MOVEMENT = 0.0

TONGUE_FEATURES = {
    1: ["tongue", "left_tongue", "right_tongue"],
    2: ["top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue"],
}
PAW_FEATURES = {
    2: ["top_paw", "bottom_paw"],
}

FIXED_DELAY_LOADERS = [
    "loadJEB6_ALMVideo.m",
    "loadJEB7_ALMVideo.m",
    "loadEKH1_ALMVideo.m",
    "loadEKH3_ALMVideo.m",
    "loadJGR2_ALMVideo.m",
    "loadJGR3_ALMVideo.m",
    "loadJEB13_ALMVideo.m",
    "loadJEB14_ALMVideo.m",
    "loadJEB15_ALMVideo.m",
    "loadJEB19_ALMVideo.m",
]
RANDOMIZED_DELAY_LOADERS = [
    "loadJEB11_ALMVideo.m",
    "loadJEB12_ALMVideo.m",
    "loadJEB23_ALMVideo.m",
    "loadJEB24_ALMVideo.m",
]


@dataclass(frozen=True)
class SessionSpec:
    subject: str
    date: str
    probe: Tuple[int, ...]
    folder: str
    task: str

    @property
    def session_id(self) -> str:
        return f"{self.subject}_{self.date}"

    @property
    def data_path(self) -> Path:
        return DATA_ROOT / self.folder / f"data_structure_{self.subject}_{self.date}.mat"

    @property
    def motion_energy_path(self) -> Path:
        return DATA_ROOT / self.folder / f"motionEnergy_{self.subject}_{self.date}.mat"


@dataclass(frozen=True)
class SelectedNeuron:
    probe_num: int
    neuron_index: int
    region_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Hasnain/Birnbaum mouse neural data to decoder format.")
    parser.add_argument("outpicklefile", type=Path, help="Output pickle path.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all reference-selected sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only two sessions for testing.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing diagnostic plots for up to two sessions as processing_<session_id>.png.",
    )
    return parser.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


def strip_comment(line: str) -> str:
    if "%" in line:
        return line.split("%", 1)[0]
    return line


def parse_probe_spec(raw_probe: str) -> Tuple[int, ...]:
    probe_nums = tuple(int(x) for x in re.findall(r"\d+", raw_probe))
    if not probe_nums:
        raise ValueError(f"Could not parse probe spec from {raw_probe!r}")
    return probe_nums


def parse_loader_file(loader_name: str, folder: str, task: str) -> List[SessionSpec]:
    path = LOADER_ROOT / loader_name
    subject_match = re.match(r"load([A-Z0-9]+)_ALMVideo\.m", loader_name)
    if not subject_match:
        raise ValueError(f"Could not parse subject from loader file {loader_name}")
    subject = subject_match.group(1)

    date_re = re.compile(r"meta\(end\)\.date = '([^']+)'")
    probe_re = re.compile(r"meta\(end\)\.probe = (\[[^\]]+\]|[0-9]+)")

    sessions: List[SessionSpec] = []
    current_date: Optional[str] = None
    current_probe: Optional[Tuple[int, ...]] = None

    for raw_line in path.read_text().splitlines():
        line = strip_comment(raw_line).strip()
        if not line:
            continue

        date_match = date_re.search(line)
        if date_match:
            current_date = date_match.group(1)
            continue

        probe_match = probe_re.search(line)
        if probe_match:
            current_probe = parse_probe_spec(probe_match.group(1))
            continue

        if "datapth = fullfile" in line and current_date is not None and current_probe is not None:
            sessions.append(
                SessionSpec(
                    subject=subject,
                    date=current_date,
                    probe=current_probe,
                    folder=folder,
                    task=task,
                )
            )
            current_date = None
            current_probe = None

    return sessions


def get_reference_sessions() -> List[SessionSpec]:
    sessions: List[SessionSpec] = []
    for loader in FIXED_DELAY_LOADERS:
        sessions.extend(parse_loader_file(loader, folder="Ephys_Behavior", task="fixed_delay"))
    for loader in RANDOMIZED_DELAY_LOADERS:
        sessions.extend(parse_loader_file(loader, folder="RandomizedDelay_Ephys_Behavior", task="randomized_delay"))
    return sessions


def normalize_string(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.dtype.kind in {"U", "S"}:
            return "".join(str(x) for x in value.reshape(-1)).strip()
        if value.dtype.kind in {"i", "u", "f"} and np.nanmax(value) <= 255:
            chars = [chr(int(x)) for x in value.reshape(-1) if int(x) != 0]
            return "".join(chars).strip()
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return normalize_string(value[0])
        return "".join(normalize_string(v) for v in value).strip()
    return str(value).strip()


def to_array(x, dtype=None) -> np.ndarray:
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def to_vector(x, dtype=float) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype).reshape(-1)
    return arr


def robust_mode(x: Sequence[float]) -> float:
    arr = to_vector(x, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    rounded = np.round(arr, 6)
    vals, counts = np.unique(rounded, return_counts=True)
    return float(vals[np.argmax(counts)])


def make_time_edges() -> np.ndarray:
    return np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)


def make_time_centers(edges: np.ndarray) -> np.ndarray:
    return edges[:-1] + DT / 2.0


def gaussian_window(n: int, alpha: float = 2.5) -> np.ndarray:
    if n <= 1:
        return np.ones(1, dtype=np.float64)
    m = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    sigma = (n - 1) / (2.0 * alpha)
    return np.exp(-0.5 * (m / sigma) ** 2)


def causal_gaussian_smooth(x: np.ndarray, n: int, bctype: str = BCTYPE) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if n <= 1:
        return x.astype(np.float32, copy=True)

    squeeze = False
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True

    if bctype == "reflect":
        x_filt = np.concatenate([x[:n, :], x], axis=0)
        trim = n
    elif bctype == "zeropad":
        x_filt = np.concatenate([np.zeros((n, x.shape[1]), dtype=x.dtype), x], axis=0)
        trim = n
    else:
        x_filt = x
        trim = 0

    kern = gaussian_window(n)
    kern[: n // 2] = 0.0
    kern /= kern.sum()

    out = np.empty_like(x_filt, dtype=np.float64)
    for col in range(x_filt.shape[1]):
        out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")

    out = out[trim:, :]
    if squeeze:
        return out[:, 0].astype(np.float32)
    return out.astype(np.float32)


def fill_nearest_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).copy()
    mask = np.isnan(x)
    if not mask.any():
        return x
    valid = np.flatnonzero(~mask)
    if valid.size == 0:
        return x
    idx = np.arange(x.size)
    pos = np.searchsorted(valid, idx)
    left = valid[np.clip(pos - 1, 0, valid.size - 1)]
    right = valid[np.clip(pos, 0, valid.size - 1)]
    choose_left = np.abs(idx - left) <= np.abs(right - idx)
    nearest = np.where(choose_left, left, right)
    x[mask] = x[nearest[mask]]
    return x


def normalize_feat_names(raw_feat_names) -> List[str]:
    return [normalize_string(x) for x in raw_feat_names]


def quality_keep_mask(qualities: Sequence[str]) -> np.ndarray:
    cleaned = [normalize_string(q).lower() for q in qualities]
    bad = {"garbage", "gabrga", "noisy", "real?"}
    return np.array([q not in bad for q in cleaned], dtype=bool)


def normalize_probe_container(probes):
    if probes is None:
        return []
    if isinstance(probes, list):
        return probes
    return [probes]


def compute_video_offset(obj: dict) -> float:
    try:
        bit_start = robust_mode(obj["bp"]["ev"]["bitStart"])
        vid_file_offset = robust_mode(obj["sglx"]["bitcode"]["bitstart"]) / float(obj["sglx"]["fs"])
        if np.isfinite(bit_start) and np.isfinite(vid_file_offset):
            return float(vid_file_offset - bit_start)
    except Exception:
        pass
    return 0.5


def get_probe_loc(obj: dict, probe_index_zero_based: int) -> str:
    ex_probe = obj.get("ex", {}).get("probe")
    if ex_probe is None:
        return "ALM"
    if isinstance(ex_probe, dict):
        loc = ex_probe.get("loc")
        if isinstance(loc, list) and probe_index_zero_based < len(loc):
            parsed = normalize_string(loc[probe_index_zero_based])
            return parsed if parsed else "ALM"
        parsed = normalize_string(loc)
        return parsed if parsed else "ALM"
    if isinstance(ex_probe, list):
        if probe_index_zero_based >= len(ex_probe):
            return "ALM"
        probe_info = ex_probe[probe_index_zero_based]
        if not isinstance(probe_info, dict):
            return "ALM"
        loc = normalize_string(probe_info.get("loc", ""))
        return loc if loc else "ALM"
    return "ALM"


def get_view_trial(view: dict, trial_index: int) -> dict:
    return {key: view[key][trial_index] for key in view.keys()}


def get_frame_times(trial_view: dict, n_frames: int) -> np.ndarray:
    frame_times = trial_view.get("frameTimes")
    if frame_times is None:
        return (np.arange(n_frames, dtype=np.float32) + 1.0) / 400.0
    arr = to_vector(frame_times, float)
    if arr.size != n_frames or not np.isfinite(arr).any():
        return (np.arange(n_frames, dtype=np.float32) + 1.0) / 400.0
    return arr.astype(np.float32)


def find_feature_index(view: dict, feat_name: str) -> Optional[int]:
    feat_names = view.get("featNames", [])
    for trial_feat_names in feat_names:
        names = normalize_feat_names(trial_feat_names)
        for idx, name in enumerate(names):
            if name == feat_name:
                return idx
    return None


def aligned_position(
    obj: dict,
    view_index_one_based: int,
    feat_name: str,
    time_centers: np.ndarray,
    align_times: np.ndarray,
    vidshift: float,
) -> Tuple[np.ndarray, np.ndarray]:
    view = obj["traj"][view_index_one_based - 1]
    n_trials = int(round(float(obj["bp"]["Ntrials"])))
    feat_index = find_feature_index(view, feat_name)
    xpos = np.full((time_centers.size, n_trials), np.nan, dtype=np.float32)
    ypos = np.full((time_centers.size, n_trials), np.nan, dtype=np.float32)
    if feat_index is None:
        return xpos, ypos

    taxis = time_centers + ADVANCE_MOVEMENT

    for trial in range(n_trials):
        trial_view = get_view_trial(view, trial)
        dropped = trial_view.get("NdroppedFrames")
        if dropped is not None:
            dropped_arr = np.asarray(dropped)
            if dropped_arr.size and np.isnan(np.asarray(dropped_arr, dtype=float)).all():
                continue

        ts = np.asarray(trial_view["ts"], dtype=np.float32)
        if ts.ndim != 3 or feat_index >= ts.shape[2]:
            continue
        frame_times = get_frame_times(trial_view, ts.shape[0])
        if not np.isfinite(frame_times).any():
            continue

        xy = ts[:, :2, feat_index]
        old_time = frame_times - vidshift - float(align_times[trial])

        interp = interp1d(
            old_time,
            xy,
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
            assume_sorted=True,
        )
        xy_aligned = interp(taxis)
        xpos[:, trial] = xy_aligned[:, 0]
        ypos[:, trial] = xy_aligned[:, 1]

        if "tongue" not in feat_name:
            xpos[:, trial] = fill_nearest_1d(xpos[:, trial])
            ypos[:, trial] = fill_nearest_1d(ypos[:, trial])

    return xpos, ypos


def feature_velocity(xpos: np.ndarray, ypos: np.ndarray, feat_name: str) -> Tuple[np.ndarray, np.ndarray]:
    xvel = np.full_like(xpos, np.nan, dtype=np.float32)
    yvel = np.full_like(ypos, np.nan, dtype=np.float32)
    for trial in range(xpos.shape[1]):
        tsinterp = np.column_stack([xpos[:, trial], ypos[:, trial]]).astype(np.float32)
        diffs = np.diff(tsinterp, axis=0)
        if diffs.size == 0 or np.isnan(diffs).all():
            base = np.array([0.0, 0.0], dtype=np.float32)
        else:
            base = np.nanmedian(diffs, axis=0)
        xv = np.gradient(tsinterp[:, 0]).astype(np.float32)
        yv = np.gradient(tsinterp[:, 1]).astype(np.float32)
        if "tongue" not in feat_name:
            xv = xv - base[0]
            yv = yv - base[1]
            xv = fill_nearest_1d(xv)
            yv = fill_nearest_1d(yv)
        else:
            xv = np.nan_to_num(xv, nan=0.0)
            yv = np.nan_to_num(yv, nan=0.0)
        xvel[:, trial] = xv
        yvel[:, trial] = yv
    return xvel, yvel


def aggregate_speed(
    obj: dict,
    feature_map: Dict[int, List[str]],
    time_centers: np.ndarray,
    align_times: np.ndarray,
    vidshift: float,
) -> Tuple[np.ndarray, List[str]]:
    speed_components = []
    used_features: List[str] = []
    for view_index, feat_names in feature_map.items():
        for feat_name in feat_names:
            xpos, ypos = aligned_position(obj, view_index, feat_name, time_centers, align_times, vidshift)
            if not np.isfinite(xpos).any() and not np.isfinite(ypos).any():
                continue
            xvel, yvel = feature_velocity(xpos, ypos, feat_name)
            speed = np.sqrt(np.square(xvel) + np.square(yvel))
            speed_components.append(speed)
            used_features.append(f"view{view_index}:{feat_name}")
    if not speed_components:
        n_trials = int(round(float(obj["bp"]["Ntrials"])))
        return np.zeros((time_centers.size, n_trials), dtype=np.float32), used_features
    stacked = np.stack(speed_components, axis=2)
    valid = np.isfinite(stacked)
    count = np.sum(valid, axis=2)
    summed = np.sum(np.where(valid, stacked, 0.0), axis=2)
    agg = np.divide(summed, np.maximum(count, 1), dtype=np.float32)
    agg[count == 0] = np.nan
    agg = np.nan_to_num(agg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return agg, used_features


def unwrap_embedded_motion_energy(raw_me) -> List[np.ndarray]:
    data = raw_me
    seen = set()
    while isinstance(data, dict) and "data" in data and id(data) not in seen:
        seen.add(id(data))
        next_data = data["data"]
        if next_data is data:
            break
        data = next_data
    out: List[np.ndarray] = []
    for trial in data:
        arr = np.asarray(trial)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        out.append(arr.astype(np.float32))
    return out


def load_motion_energy_raw(obj: dict, spec: SessionSpec) -> Optional[dict]:
    if spec.motion_energy_path.exists():
        raw_me = read_mat(spec.motion_energy_path).get("me")
        if raw_me is None:
            return None
        move_thresh = np.nan
        cursor = raw_me
        seen = set()
        while isinstance(cursor, dict) and id(cursor) not in seen:
            seen.add(id(cursor))
            if "moveThresh" in cursor:
                move_thresh = cursor["moveThresh"]
            if "data" not in cursor or cursor["data"] is cursor:
                break
            cursor = cursor["data"]
        return {"data": unwrap_embedded_motion_energy(raw_me), "moveThresh": move_thresh}
    if "me" in obj:
        raw_me = obj["me"]
        if isinstance(raw_me, dict):
            return {"data": unwrap_embedded_motion_energy(raw_me), "moveThresh": raw_me.get("moveThresh", np.nan)}
        return {"data": unwrap_embedded_motion_energy(raw_me), "moveThresh": np.nan}
    return None


def aligned_motion_energy(
    obj: dict,
    raw_me: Optional[dict],
    time_centers: np.ndarray,
    align_times: np.ndarray,
    vidshift: float,
) -> np.ndarray:
    n_trials = int(round(float(obj["bp"]["Ntrials"])))
    aligned = np.full((time_centers.size, n_trials), np.nan, dtype=np.float32)
    if raw_me is None:
        return np.zeros((time_centers.size, n_trials), dtype=np.float32)

    me_data = raw_me.get("data", [])
    view = obj["traj"][0]
    taxis = time_centers + ADVANCE_MOVEMENT

    for trial in range(min(len(me_data), n_trials)):
        me_trial = np.asarray(me_data[trial], dtype=np.float32).reshape(-1)
        trial_view = get_view_trial(view, trial)
        ts = np.asarray(trial_view["ts"])
        frame_times = get_frame_times(trial_view, ts.shape[0])
        if frame_times.size != me_trial.size:
            frame_times = (np.arange(me_trial.size, dtype=np.float32) + 1.0) / 400.0
            old_time = frame_times - 0.5 - float(align_times[trial])
        else:
            old_time = frame_times - vidshift - float(align_times[trial])

        interp = interp1d(
            old_time,
            me_trial,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
            assume_sorted=True,
        )
        aligned[:, trial] = interp(taxis).astype(np.float32)
        aligned[:, trial] = fill_nearest_1d(aligned[:, trial])

    return np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def percentile_threshold(traces: np.ndarray, percentile: float = 50.0) -> float:
    flat = traces.reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0
    return float(np.percentile(flat, percentile))


def discretize_trace(traces: np.ndarray, threshold: float) -> np.ndarray:
    flat = traces.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    out = np.zeros(flat.size, dtype=np.int64)
    out[order[flat.size // 2 :]] = 1
    return out.reshape(traces.shape)


def build_trial_mask(obj: dict) -> np.ndarray:
    bp = obj["bp"]
    stim_enable = to_vector(bp["stim"]["enable"], float).astype(bool)
    early = to_vector(bp["early"], float).astype(bool)
    no = to_vector(bp["no"], float).astype(bool)
    valid = (~stim_enable) & (~early) & (~no)
    return valid


def count_window_spikes(trials_1based: np.ndarray, trialtm: np.ndarray, align_times: np.ndarray, trial_mask: np.ndarray) -> int:
    keep_trial_index = np.full(trial_mask.size + 1, -1, dtype=np.int32)
    keep_trial_index[1:] = np.where(trial_mask, 1, -1)
    spike_trial_mask = keep_trial_index[trials_1based] > 0
    if not spike_trial_mask.any():
        return 0
    aligned = trialtm[spike_trial_mask] - align_times[trials_1based[spike_trial_mask] - 1]
    return int(np.sum((aligned >= TMIN) & (aligned < TMAX)))


def neuron_mean_fr(
    probe: dict,
    neuron_index: int,
    align_times: np.ndarray,
    trial_mask: np.ndarray,
) -> float:
    trials = to_vector(probe["trial"][neuron_index], int)
    trialtm = to_vector(probe["trialtm"][neuron_index], float)
    n_spikes = count_window_spikes(trials, trialtm, align_times, trial_mask)
    n_trials = int(trial_mask.sum())
    if n_trials == 0:
        return 0.0
    return n_spikes / (n_trials * (TMAX - TMIN))


def binned_neuron_trials(
    probe: dict,
    neuron_index: int,
    align_times: np.ndarray,
    keep_trials_0based: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    trials = to_vector(probe["trial"][neuron_index], int)
    trialtm = to_vector(probe["trialtm"][neuron_index], float)
    trial_to_keep = np.full(align_times.size + 1, -1, dtype=np.int32)
    trial_to_keep[keep_trials_0based + 1] = np.arange(keep_trials_0based.size, dtype=np.int32)

    keep_index = trial_to_keep[trials]
    mask = keep_index >= 0
    keep_index = keep_index[mask]
    aligned = trialtm[mask] - align_times[trials[mask] - 1]
    bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
    valid = (bin_index >= 0) & (bin_index < edges.size - 1)

    counts = np.zeros((edges.size - 1, keep_trials_0based.size), dtype=np.float32)
    np.add.at(counts, (bin_index[valid], keep_index[valid]), 1.0)
    rates = counts / DT
    return causal_gaussian_smooth(rates, SMOOTH, BCTYPE)


def select_neurons(
    obj: dict,
    spec: SessionSpec,
    trial_mask: np.ndarray,
) -> Tuple[List[SelectedNeuron], List[dict]]:
    probes = normalize_probe_container(obj.get("clu"))
    align_times = to_vector(obj["bp"]["ev"]["goCue"], float)
    selected: List[SelectedNeuron] = []
    probe_summaries: List[dict] = []

    for probe_num in spec.probe:
        probe_zero = probe_num - 1
        if probes is None or probe_zero >= len(probes):
            raise ValueError(f"{spec.session_id}: selected probe {probe_num} missing from raw file")
        probe = probes[probe_zero]
        if probe is None:
            raise ValueError(f"{spec.session_id}: selected probe {probe_num} has no cluster data")

        qualities = [normalize_string(q) for q in probe["quality"]]
        quality_mask = quality_keep_mask(qualities)
        loc = get_probe_loc(obj, probe_zero)
        region_label = session_brain_region_label(loc)
        kept_indices = []
        for neuron_index in np.flatnonzero(quality_mask):
            mean_fr = neuron_mean_fr(probe, int(neuron_index), align_times, trial_mask)
            if mean_fr > LOW_FR:
                kept_indices.append(int(neuron_index))
                selected.append(
                    SelectedNeuron(
                        probe_num=probe_num,
                        neuron_index=int(neuron_index),
                        region_label=region_label,
                    )
                )

        probe_summaries.append(
            {
                "probe_num": probe_num,
                "probe_loc": loc,
                "region_label": region_label,
                "n_neurons_raw_quality_filtered": int(np.sum(quality_mask)),
                "n_neurons_kept": int(len(kept_indices)),
            }
        )

    return selected, probe_summaries


def max_supported_trial(selected_neurons: Sequence[SelectedNeuron], probes) -> int:
    max_trial = 0
    for selected in selected_neurons:
        probe = probes[selected.probe_num - 1]
        trials = to_vector(probe["trial"][selected.neuron_index], int)
        if trials.size:
            max_trial = max(max_trial, int(np.max(trials)))
    return max_trial


def session_brain_region_label(loc: str) -> str:
    loc = normalize_string(loc)
    return loc if loc else "ALM"


def plot_processing(
    spec: SessionSpec,
    session_info: dict,
    neural_trials: List[np.ndarray],
    time_centers: np.ndarray,
    lick_direction: np.ndarray,
    context: np.ndarray,
    outcome: np.ndarray,
    tongue_speed: np.ndarray,
    paw_speed: np.ndarray,
    motion_energy: np.ndarray,
    tongue_bin: np.ndarray,
    paw_bin: np.ndarray,
    me_bin: np.ndarray,
    tongue_thr: float,
    paw_thr: float,
    me_thr: float,
) -> None:
    if not neural_trials:
        return

    trial_ix = 0
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex="col")
    fig.suptitle(f"Processing summary: {spec.session_id}")

    axes[0, 0].imshow(neural_trials[trial_ix], aspect="auto", interpolation="nearest", cmap="viridis")
    axes[0, 0].set_title("Neural trial (neurons x time)")
    axes[0, 0].set_ylabel("Neuron")

    axes[0, 1].plot(time_centers, tongue_speed[:, trial_ix], label="tongue speed")
    axes[0, 1].axhline(tongue_thr, color="k", linestyle="--", linewidth=1, label="session median")
    axes[0, 1].step(time_centers, tongue_bin[:, trial_ix] * tongue_thr, where="mid", color="tab:red", alpha=0.7, label="bin")
    axes[0, 1].set_title("Tongue speed -> bin")
    axes[0, 1].legend(loc="upper right", fontsize=8)

    axes[1, 0].plot(time_centers, paw_speed[:, trial_ix], color="tab:green", label="paw speed")
    axes[1, 0].axhline(paw_thr, color="k", linestyle="--", linewidth=1, label="session median")
    axes[1, 0].step(time_centers, paw_bin[:, trial_ix] * paw_thr, where="mid", color="tab:orange", alpha=0.7, label="bin")
    axes[1, 0].set_title("Paw speed -> bin")
    axes[1, 0].legend(loc="upper right", fontsize=8)

    axes[1, 1].plot(time_centers, motion_energy[:, trial_ix], color="tab:purple", label="motion energy")
    axes[1, 1].axhline(me_thr, color="k", linestyle="--", linewidth=1, label="session median")
    axes[1, 1].step(time_centers, me_bin[:, trial_ix] * me_thr, where="mid", color="tab:brown", alpha=0.7, label="bin")
    axes[1, 1].set_title("Motion energy -> bin")
    axes[1, 1].legend(loc="upper right", fontsize=8)

    out_mat = np.vstack(
        [
            np.full(time_centers.size, lick_direction[trial_ix]),
            np.full(time_centers.size, context[trial_ix]),
            np.full(time_centers.size, outcome[trial_ix]),
            tongue_bin[:, trial_ix],
            paw_bin[:, trial_ix],
            me_bin[:, trial_ix],
        ]
    )
    axes[2, 0].imshow(out_mat, aspect="auto", interpolation="nearest", cmap="tab20")
    axes[2, 0].set_yticks(range(6))
    axes[2, 0].set_yticklabels(
        ["lick_dir", "context", "outcome", "tongue_bin", "paw_bin", "motion_bin"],
        fontsize=8,
    )
    axes[2, 0].set_title("Outputs for one trial")

    axes[2, 1].hist(tongue_speed.reshape(-1), bins=80, alpha=0.5, label="tongue")
    axes[2, 1].hist(paw_speed.reshape(-1), bins=80, alpha=0.5, label="paw")
    axes[2, 1].hist(motion_energy.reshape(-1), bins=80, alpha=0.5, label="motion")
    axes[2, 1].axvline(tongue_thr, color="tab:blue", linestyle="--")
    axes[2, 1].axvline(paw_thr, color="tab:green", linestyle="--")
    axes[2, 1].axvline(me_thr, color="tab:purple", linestyle="--")
    axes[2, 1].set_title("Session distributions and medians")
    axes[2, 1].legend(fontsize=8)

    axes[3, 0].plot(time_centers, np.mean(neural_trials[trial_ix], axis=0), color="black")
    axes[3, 0].axvline(0.0, color="tab:red", linestyle="--")
    axes[3, 0].set_title("Mean neural activity (trial average across neurons)")
    axes[3, 0].set_xlabel("Time from go cue (s)")

    axes[3, 1].axis("off")
    axes[3, 1].text(
        0.0,
        1.0,
        "\n".join(
            [
                f"task={spec.task}",
                f"probe={list(spec.probe)}",
                f"raw_trials={session_info['n_trials_raw']}",
                f"kept_trials={session_info['n_trials_kept']}",
                f"raw_neurons_probes={session_info['n_neurons_raw_selected_probes']}",
                f"kept_neurons={session_info['n_neurons_kept']}",
                f"regions={','.join(session_info['region_labels'])}",
            ]
        ),
        va="top",
        ha="left",
        family="monospace",
    )

    for ax in axes.flat:
        if ax is not axes[2, 0] and ax is not axes[3, 1]:
            ax.axvline(0.0, color="tab:red", linestyle="--", linewidth=1)

    fig.tight_layout()
    fig.savefig(ROOT / f"processing_{spec.session_id}.png", dpi=150)
    plt.close(fig)


def process_session(
    spec: SessionSpec,
    show_processing: bool,
) -> Optional[Tuple[dict, List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray, int, np.ndarray]]:
    t0 = time.perf_counter()
    obj = read_mat(spec.data_path)["obj"]
    probes = normalize_probe_container(obj.get("clu"))
    align_times = to_vector(obj["bp"]["ev"]["goCue"], float)
    time_edges = make_time_edges()
    time_centers = make_time_centers(time_edges)

    trial_mask = build_trial_mask(obj)
    keep_trials = np.flatnonzero(trial_mask)
    if keep_trials.size < 2:
        log(f"Skipping {spec.session_id}: fewer than 2 valid trials after trial filtering")
        return None

    selected_neurons, probe_summaries = select_neurons(obj, spec, trial_mask)
    max_trial_with_spikes = max_supported_trial(selected_neurons, probes)
    if max_trial_with_spikes > 0:
        neural_coverage_mask = (np.arange(trial_mask.size, dtype=np.int32) + 1) <= max_trial_with_spikes
        refined_trial_mask = trial_mask & neural_coverage_mask
        if not np.array_equal(refined_trial_mask, trial_mask):
            dropped = int(np.sum(trial_mask & ~refined_trial_mask))
            log(
                f"{spec.session_id}: dropping {dropped} behavior-valid trials after raw trial "
                f"{max_trial_with_spikes} due to missing neural coverage"
            )
            trial_mask = refined_trial_mask
            keep_trials = np.flatnonzero(trial_mask)
            if keep_trials.size < 2:
                log(f"Skipping {spec.session_id}: fewer than 2 valid trials after neural-coverage filtering")
                return None
            selected_neurons, probe_summaries = select_neurons(obj, spec, trial_mask)
            max_trial_with_spikes = max_supported_trial(selected_neurons, probes)
    if len(selected_neurons) < MIN_UNITS_PER_SESSION:
        log(f"Skipping {spec.session_id}: only {len(selected_neurons)} units after filtering")
        return None

    vidshift = compute_video_offset(obj)
    tongue_speed_all, tongue_feats = aggregate_speed(obj, TONGUE_FEATURES, time_centers, align_times, vidshift)
    paw_speed_all, paw_feats = aggregate_speed(obj, PAW_FEATURES, time_centers, align_times, vidshift)
    motion_energy_all = aligned_motion_energy(obj, load_motion_energy_raw(obj, spec), time_centers, align_times, vidshift)

    tongue_speed = tongue_speed_all[:, keep_trials]
    paw_speed = paw_speed_all[:, keep_trials]
    motion_energy = motion_energy_all[:, keep_trials]

    tongue_thr = percentile_threshold(tongue_speed, 50.0)
    paw_thr = percentile_threshold(paw_speed, 50.0)
    me_thr = percentile_threshold(motion_energy, 50.0)

    tongue_bin = discretize_trace(tongue_speed, tongue_thr)
    paw_bin = discretize_trace(paw_speed, paw_thr)
    me_bin = discretize_trace(motion_energy, me_thr)

    n_trials = keep_trials.size
    n_neurons = len(selected_neurons)
    neural_trials = [np.zeros((n_neurons, time_centers.size), dtype=np.float32) for _ in range(n_trials)]
    region_labels_per_neuron = []
    for out_idx, selected in enumerate(selected_neurons):
        probe = probes[selected.probe_num - 1]
        rates = binned_neuron_trials(probe, selected.neuron_index, align_times, keep_trials, time_edges)
        region_labels_per_neuron.append(selected.region_label)
        for tr in range(n_trials):
            neural_trials[tr][out_idx, :] = rates[:, tr]

    bp = obj["bp"]
    R = to_vector(bp["R"], float).astype(int)
    autowater = to_vector(bp["autowater"], float).astype(int)
    hit = to_vector(bp["hit"], float).astype(int)
    miss = to_vector(bp["miss"], float).astype(int)

    lick_direction = R[keep_trials]
    context = 1 - autowater[keep_trials]
    outcome = hit[keep_trials]
    if not np.all((outcome == 0) | (outcome == 1)):
        raise ValueError(f"{spec.session_id}: outcome contains values outside hit/miss after filtering")
    if not np.all((hit[keep_trials] + miss[keep_trials]) == 1):
        raise ValueError(f"{spec.session_id}: hit/miss are not mutually exclusive on kept trials")

    input_trials: List[np.ndarray] = []
    output_trials: List[np.ndarray] = []
    for tr in range(n_trials):
        input_trials.append(time_centers[None, :].astype(np.float32))
        output_trials.append(
            np.vstack(
                [
                    np.full(time_centers.size, lick_direction[tr], dtype=np.int64),
                    np.full(time_centers.size, context[tr], dtype=np.int64),
                    np.full(time_centers.size, outcome[tr], dtype=np.int64),
                    tongue_bin[:, tr].astype(np.int64),
                    paw_bin[:, tr].astype(np.int64),
                    me_bin[:, tr].astype(np.int64),
                ]
            )
        )

    session_info = {
        "session_id": spec.session_id,
        "subject": spec.subject,
        "date": spec.date,
        "task": spec.task,
        "probe": list(spec.probe),
        "region_labels": stable_unique(region_labels_per_neuron),
        "probe_summaries": probe_summaries,
        "n_trials_raw": int(round(float(obj["bp"]["Ntrials"]))),
        "n_trials_kept": int(n_trials),
        "max_trial_with_spikes": int(max_trial_with_spikes),
        "n_neurons_raw_selected_probes": int(sum(x["n_neurons_raw_quality_filtered"] for x in probe_summaries)),
        "n_neurons_kept": int(n_neurons),
        "tongue_threshold": float(tongue_thr),
        "paw_threshold": float(paw_thr),
        "motion_energy_threshold": float(me_thr),
        "tongue_features_used": tongue_feats,
        "paw_features_used": paw_feats,
    }

    if show_processing:
        plot_processing(
            spec,
            session_info,
            neural_trials,
            time_centers,
            lick_direction,
            context,
            outcome,
            tongue_speed,
            paw_speed,
            motion_energy,
            tongue_bin,
            paw_bin,
            me_bin,
            tongue_thr,
            paw_thr,
            me_thr,
        )

    dt_s = time.perf_counter() - t0
    log(
        f"Processed {spec.session_id}: {n_trials} trials kept, {n_neurons} neurons kept "
        f"in {dt_s:.2f}s"
    )
    return (
        session_info,
        neural_trials,
        input_trials,
        output_trials,
        time_centers,
        n_neurons,
        np.asarray(region_labels_per_neuron, dtype=object),
    )


def stable_unique(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def build_dataset(session_specs: Sequence[SessionSpec], show_processing: bool) -> dict:
    all_session_infos = []
    neural_sessions: List[List[np.ndarray]] = []
    input_sessions: List[List[np.ndarray]] = []
    output_sessions: List[List[np.ndarray]] = []
    subject_order: List[str] = []
    subject_idx: List[int] = []
    brain_region_order: List[str] = []
    brain_region_idx: List[np.ndarray] = []

    plot_budget = 2
    for spec in session_specs:
        do_plot = show_processing and plot_budget > 0
        result = process_session(spec, show_processing=do_plot)
        if result is None:
            continue
        if do_plot:
            plot_budget -= 1

        session_info, neural_trials, input_trials, output_trials, _, n_neurons, neuron_regions = result
        all_session_infos.append(session_info)
        neural_sessions.append(neural_trials)
        input_sessions.append(input_trials)
        output_sessions.append(output_trials)

        if spec.subject not in subject_order:
            subject_order.append(spec.subject)
        subject_idx.append(subject_order.index(spec.subject))

        session_region_idx = np.zeros(n_neurons, dtype=np.int64)
        for i, region_label in enumerate(neuron_regions.tolist()):
            if region_label not in brain_region_order:
                brain_region_order.append(region_label)
            session_region_idx[i] = brain_region_order.index(region_label)
        brain_region_idx.append(session_region_idx)

    data = {
        "neural": neural_sessions,
        "input": input_sessions,
        "output": output_sessions,
        "subjects": subject_order,
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": brain_region_order,
        "brain_region_idx": brain_region_idx,
        "input_names": ["time_from_go_cue"],
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
                "Go-cue-aligned ALM neural activity from fixed-delay and randomized-delay "
                "directional licking tasks, with decoder outputs for lick direction, context, "
                "outcome, and session-median-binarized movement variables."
            ),
            "time_bin_size": 5.0,
            "time_bin_size_sec": DT,
            "temporal_alignment_event": "Auditory go cue onset",
            "off_start": TMIN,
            "off_end": TMAX,
            "smoothing_window_bins": SMOOTH,
            "smoothing_boundary_condition": BCTYPE,
            "neuron_quality_filter": "all_except_garbage_gabrga_noisy_realq",
            "low_firing_rate_threshold_hz": LOW_FR,
            "minimum_units_per_session": MIN_UNITS_PER_SESSION,
            "trial_filter": "exclude stim.enable, early, and no-response trials",
            "session_specs": [session_info["session_id"] for session_info in all_session_infos],
            "session_info": all_session_infos,
        },
    }
    return data


def main() -> None:
    warnings.filterwarnings("ignore", message="Complex objects .* are not supported.*")
    args = parse_args()
    full = not args.sample

    session_specs = get_reference_sessions()
    if args.sample:
        fixed = [s for s in session_specs if s.task == "fixed_delay"]
        rand = [s for s in session_specs if s.task == "randomized_delay"]
        session_specs = [fixed[0], rand[0]]
    log(f"Selected {len(session_specs)} sessions ({'full' if full else 'sample'} mode)")

    t0 = time.perf_counter()
    data = build_dataset(session_specs, show_processing=args.show_processing)
    valid, errors, warnings_out = verify_data_format(data)
    if not valid:
        for err in errors:
            log(f"ERROR: {err}")
        raise SystemExit(1)
    for warning_msg in warnings_out:
        log(f"WARNING: {warning_msg}")

    import pickle

    with args.outpicklefile.open("wb") as f:
        pickle.dump(data, f)

    elapsed = time.perf_counter() - t0
    nsessions = len(data["neural"])
    ntrials = sum(len(x) for x in data["neural"])
    nneurons = int(sum(x.size for x in data["brain_region_idx"]))
    log(
        f"Saved {args.outpicklefile} with {nsessions} sessions, {ntrials} trials, "
        f"{nneurons} neurons in {elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
