from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat


DATA_ROOT = Path("/app/data")
OUTPUT_PATH = Path("/app/converted_data.pkl")

DT = 0.01
TMIN = -3.0
TMAX = 2.5
SMOOTH_WINDOW = 15
BCTYPE = "reflect"
LOW_FR_HZ = 1.0

EDGES = np.arange(TMIN, TMAX + 1e-9, DT)
TIME = EDGES[:-1] + (DT / 2.0)

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
    ["WC", "DR"],
    ["incorrect", "correct", "ignore"],
    ["<50th percentile", ">=50th percentile", "not visible"],
    ["<50th percentile", ">=50th percentile", "not visible"],
    ["<50th percentile", ">=50th percentile", "no video"],
]


@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    folder: str
    probe: int

    @property
    def session_id(self) -> str:
        return f"{self.animal}_{self.date}"

    @property
    def data_path(self) -> Path:
        return DATA_ROOT / self.folder / f"data_structure_{self.session_id}.mat"

    @property
    def motion_energy_path(self) -> Path:
        return DATA_ROOT / self.folder / f"motionEnergy_{self.session_id}.mat"


SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", "Ephys_Behavior", 2),
    SessionSpec("JEB7", "2021-04-29", "Ephys_Behavior", 1),
    SessionSpec("JEB7", "2021-04-30", "Ephys_Behavior", 1),
    SessionSpec("EKH1", "2021-08-07", "Ephys_Behavior", 2),
    SessionSpec("EKH3", "2021-08-11", "Ephys_Behavior", 2),
    SessionSpec("JGR2", "2021-11-16", "Ephys_Behavior", 1),
    SessionSpec("JGR2", "2021-11-17", "Ephys_Behavior", 1),
    SessionSpec("JGR3", "2021-11-18", "Ephys_Behavior", 1),
    SessionSpec("JEB19", "2023-04-18", "Ephys_Behavior", 1),
    SessionSpec("JEB19", "2023-04-19", "Ephys_Behavior", 1),
    SessionSpec("JEB19", "2023-04-20", "Ephys_Behavior", 1),
    SessionSpec("JEB19", "2023-04-21", "Ephys_Behavior", 1),
]


def decode_char(array: np.ndarray) -> str:
    array = np.asarray(array)
    return "".join(chr(int(x)) for x in array.reshape(-1) if int(x) != 0)


def read_ref_array(h5: h5py.File, ref) -> np.ndarray:
    return np.asarray(h5[ref][()]).squeeze()


def read_ref_string(h5: h5py.File, ref) -> str:
    return decode_char(h5[ref][()])


def matlab_mode(values: np.ndarray) -> float:
    values = np.asarray(values).reshape(-1)
    unique, counts = np.unique(values, return_counts=True)
    return float(unique[np.argmax(counts)])


def gausswin(n: int, alpha: float = 2.5) -> np.ndarray:
    idx = np.arange(n) - (n - 1) / 2.0
    denom = (n - 1) / 2.0
    return np.exp(-0.5 * (alpha * idx / denom) ** 2)


def my_smooth(x: np.ndarray, n: int, bctype: str = "none") -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if n in (0, 1):
        return x.copy()

    squeeze = False
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True

    if bctype == "reflect":
        x_filt = np.concatenate([x[:n], x], axis=0)
        trim = n
    elif bctype == "zeropad":
        x_filt = np.concatenate([np.zeros((n, x.shape[1])), x], axis=0)
        trim = n
    else:
        x_filt = x
        trim = 0

    kernel = gausswin(n)
    kernel[: len(kernel) // 2] = 0.0
    kernel /= kernel.sum()

    out = np.stack(
        [np.convolve(x_filt[:, j], kernel, mode="same") for j in range(x_filt.shape[1])],
        axis=1,
    )
    out = out[trim:]
    return out[:, 0] if squeeze else out


def fill_nearest(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    squeeze = False
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True

    idx = np.arange(x.shape[0])
    out = x.copy()
    for col in range(x.shape[1]):
        valid = np.isfinite(x[:, col])
        if not np.any(valid):
            continue
        interp = interp1d(
            idx[valid],
            x[valid, col],
            kind="nearest",
            bounds_error=False,
            fill_value=(x[valid, col][0], x[valid, col][-1]),
        )
        out[:, col] = interp(idx)
    return out[:, 0] if squeeze else out


def read_cell_string_list(h5: h5py.File, dataset: h5py.Dataset, nrows: int) -> list[str]:
    for row in range(nrows):
        ref = dataset[row, 0]
        cell = h5[ref][()]
        if cell.size == 0:
            continue
        return [read_ref_string(h5, item) for item in cell.reshape(-1)]
    return []


def read_motion_energy(path: Path):
    loaded = loadmat(path, squeeze_me=True, struct_as_record=False)
    me = loaded["me"]
    data = me.data
    if not isinstance(data, np.ndarray) and hasattr(data, "data"):
        data = data.data
    return data, float(me.moveThresh)


def align_and_histogram(trial_ids: np.ndarray, aligned_times: np.ndarray, n_trials: int) -> np.ndarray:
    counts = np.zeros((n_trials, TIME.size), dtype=np.float64)
    if aligned_times.size == 0:
        return counts

    bins = np.searchsorted(EDGES, aligned_times, side="right") - 1
    valid = (bins >= 0) & (bins < TIME.size) & (trial_ids >= 1) & (trial_ids <= n_trials)
    if not np.any(valid):
        return counts

    np.add.at(counts, (trial_ids[valid] - 1, bins[valid]), 1.0)
    return counts


def make_condition_masks(bp: dict[str, np.ndarray]) -> list[np.ndarray]:
    hit = bp["hit"]
    miss = bp["miss"]
    no = bp["no"]
    early = bp["early"]
    stim = bp["stim_enable"]
    autowater = bp["autowater"]
    return [
        hit | miss | no,
        hit & ~stim & ~autowater,
        hit & ~stim & autowater,
        miss & ~stim & ~autowater,
        miss & ~stim & autowater,
        hit & ~stim & ~autowater & ~early,
        hit & ~stim & autowater & ~early,
    ]


def trial_choice_code(bp: dict[str, np.ndarray], trial_idx: int) -> int:
    if bp["no"][trial_idx]:
        return 2
    if (bp["R"][trial_idx] and bp["hit"][trial_idx]) or (bp["L"][trial_idx] and bp["miss"][trial_idx]):
        return 1
    if (bp["L"][trial_idx] and bp["hit"][trial_idx]) or (bp["R"][trial_idx] and bp["miss"][trial_idx]):
        return 0
    raise ValueError(f"Unable to infer lick direction for trial {trial_idx}")


def trial_outcome_code(bp: dict[str, np.ndarray], trial_idx: int) -> int:
    if bp["miss"][trial_idx]:
        return 0
    if bp["hit"][trial_idx]:
        return 1
    if bp["no"][trial_idx]:
        return 2
    raise ValueError(f"Unable to infer outcome for trial {trial_idx}")


def load_bp_fields(h5: h5py.File) -> dict[str, np.ndarray]:
    bp = h5["obj"]["bp"]
    ev = bp["ev"]
    return {
        "Ntrials": int(np.asarray(bp["Ntrials"])[0, 0]),
        "hit": np.asarray(bp["hit"]).reshape(-1).astype(bool),
        "miss": np.asarray(bp["miss"]).reshape(-1).astype(bool),
        "no": np.asarray(bp["no"]).reshape(-1).astype(bool),
        "early": np.asarray(bp["early"]).reshape(-1).astype(bool),
        "R": np.asarray(bp["R"]).reshape(-1).astype(bool),
        "L": np.asarray(bp["L"]).reshape(-1).astype(bool),
        "autowater": np.asarray(bp["autowater"]).reshape(-1).astype(bool),
        "stim_enable": np.asarray(bp["stim"]["enable"]).reshape(-1).astype(bool),
        "goCue": np.asarray(ev["goCue"]).reshape(-1),
        "sample": np.asarray(ev["sample"]).reshape(-1),
        "delay": np.asarray(ev["delay"]).reshape(-1),
        "bitStart": np.asarray(ev["bitStart"]).reshape(-1),
    }


def load_probe_clusters(h5: h5py.File, probe_index: int) -> list[dict[str, np.ndarray | str]]:
    clu_refs = h5["obj"]["clu"][()]
    probe_group = h5[clu_refs[probe_index - 1, 0]]
    n_clusters = probe_group["quality"].shape[0]

    clusters = []
    for clu_idx in range(n_clusters):
        quality = read_ref_string(h5, probe_group["quality"][clu_idx, 0]).strip()
        clusters.append(
            {
                "quality": quality,
                "trial": read_ref_array(h5, probe_group["trial"][clu_idx, 0]).astype(np.int64),
                "trialtm": read_ref_array(h5, probe_group["trialtm"][clu_idx, 0]).astype(np.float64),
            }
        )
    return clusters


def compute_neural_trials(bp: dict[str, np.ndarray], clusters: list[dict[str, np.ndarray | str]], valid_mask: np.ndarray):
    condition_masks = make_condition_masks(bp)
    n_trials = bp["Ntrials"]
    good_trial_mats = []

    for cluster in clusters:
        quality = str(cluster["quality"]).strip().lower()
        if quality in QUALITY_EXCLUDE:
            continue

        trial_ids = np.asarray(cluster["trial"], dtype=np.int64)
        trialtm = np.asarray(cluster["trialtm"], dtype=np.float64)
        aligned_times = trialtm - bp["goCue"][trial_ids - 1]
        counts = align_and_histogram(trial_ids, aligned_times, n_trials)

        psths = []
        for cond_mask in condition_masks:
            n_cond = int(cond_mask.sum())
            if n_cond == 0:
                psth = np.zeros(TIME.size, dtype=np.float64)
            else:
                psth = counts[cond_mask].sum(axis=0) / n_cond / DT
                psth = my_smooth(psth, SMOOTH_WINDOW, BCTYPE)
            psths.append(psth)
        mean_fr = float(np.mean(np.stack(psths, axis=1)))
        if mean_fr <= LOW_FR_HZ:
            continue

        cluster_trials = counts[valid_mask] / DT
        cluster_trials = my_smooth(cluster_trials.T, SMOOTH_WINDOW, BCTYPE).T
        good_trial_mats.append(cluster_trials.astype(np.float32))

    if not good_trial_mats:
        return [], 0

    trial_array = np.stack(good_trial_mats, axis=1)
    neural_trials = [trial_array[trial_idx] for trial_idx in range(trial_array.shape[0])]
    return neural_trials, trial_array.shape[1]


def get_video_shift(h5: h5py.File, bp: dict[str, np.ndarray]) -> float:
    fs = float(np.asarray(h5["obj"]["sglx"]["fs"]).reshape(-1)[0])
    bitstart = np.asarray(h5["obj"]["sglx"]["bitcode"]["bitstart"]).reshape(-1)
    return matlab_mode(bitstart) / fs - matlab_mode(bp["bitStart"])


def load_view_info(h5: h5py.File):
    traj_refs = h5["obj"]["traj"][()]
    views = [h5[traj_refs[0, 0]], h5[traj_refs[1, 0]]]
    feat_names = [read_cell_string_list(h5, view["featNames"], view["featNames"].shape[0]) for view in views]
    return views, feat_names


def get_trial_video(h5: h5py.File, view_group: h5py.Group, trial_idx: int):
    ndropped = read_ref_array(h5, view_group["NdroppedFrames"][trial_idx, 0])
    if np.asarray(ndropped).size == 0 or not np.isfinite(np.asarray(ndropped).reshape(-1)[0]):
        return None, None

    ts = np.asarray(h5[view_group["ts"][trial_idx, 0]][()])
    ts = np.transpose(ts, (2, 1, 0))

    frame_times = read_ref_array(h5, view_group["frameTimes"][trial_idx, 0])
    frame_times = np.asarray(frame_times).reshape(-1)
    if frame_times.size == 0 or not np.any(np.isfinite(frame_times)):
        frame_times = np.arange(1, ts.shape[0] + 1, dtype=np.float64) / 400.0

    return ts, frame_times


def interpolate_coords(coords: np.ndarray, frame_times: np.ndarray, align_time: float, vidshift: float) -> np.ndarray:
    if frame_times.size < 2:
        return np.full((TIME.size, 2), np.nan, dtype=np.float64)
    interp = interp1d(
        frame_times - vidshift - align_time,
        coords,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )
    return np.asarray(interp(TIME), dtype=np.float64)


def feature_speed(coords_interp: np.ndarray, tongue_feature: bool):
    visible = np.isfinite(coords_interp).all(axis=1)
    if not np.any(visible):
        return np.full(TIME.size, np.nan, dtype=np.float64), visible

    if tongue_feature:
        vel = np.gradient(coords_interp, axis=0)
        vel[~np.isfinite(vel)] = 0.0
    else:
        coords_filled = fill_nearest(coords_interp)
        if not np.isfinite(coords_filled).any():
            return np.full(TIME.size, np.nan, dtype=np.float64), visible
        vel = np.gradient(coords_filled, axis=0)
        base_deriv = np.nanmedian(np.diff(coords_filled, axis=0), axis=0)
        vel[:, 0] = vel[:, 0] - base_deriv[0]
        vel[:, 1] = vel[:, 1] - base_deriv[0]
        vel = fill_nearest(vel)

    speed = np.sqrt((vel ** 2).sum(axis=1))
    return speed, visible


def aggregate_speeds(speeds: list[np.ndarray], visibles: list[np.ndarray]):
    if not speeds:
        return np.full(TIME.size, np.nan, dtype=np.float64), np.zeros(TIME.size, dtype=bool)

    speed_stack = np.stack(speeds, axis=0)
    vis_stack = np.stack(visibles, axis=0)
    visible_any = vis_stack.any(axis=0)

    out = np.full(TIME.size, np.nan, dtype=np.float64)
    for t in range(TIME.size):
        mask = vis_stack[:, t]
        if np.any(mask):
            out[t] = float(np.mean(speed_stack[mask, t]))
    return out, visible_any


def interpolate_motion_energy(me_trace: np.ndarray, frame_times: np.ndarray, align_time: float, vidshift: float) -> np.ndarray:
    if frame_times.size != me_trace.size:
        frame_times = np.arange(1, me_trace.size + 1, dtype=np.float64) / 400.0
    interp = interp1d(
        frame_times - vidshift - align_time,
        me_trace,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )
    return fill_nearest(np.asarray(interp(TIME), dtype=np.float64))


def compute_behavioral_outputs(spec: SessionSpec, bp: dict[str, np.ndarray], valid_trials: np.ndarray):
    motion_energy_data, _ = read_motion_energy(spec.motion_energy_path)

    with h5py.File(spec.data_path, "r") as h5:
        vidshift = get_video_shift(h5, bp)
        view_groups, feat_names = load_view_info(h5)

        tongue_indices = {
            0: [feat_names[0].index(name) for name in ["tongue", "left_tongue", "right_tongue"] if name in feat_names[0]],
            1: [
                feat_names[1].index(name)
                for name in ["top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue"]
                if name in feat_names[1]
            ],
        }
        paw_indices = {
            1: [feat_names[1].index(name) for name in ["top_paw", "bottom_paw"] if name in feat_names[1]],
        }

        tongue_series = []
        tongue_visible = []
        paw_series = []
        paw_visible = []
        me_series = []
        me_visible = []

        for trial_idx in valid_trials:
            align_time = bp["goCue"][trial_idx]

            trial_tongue_speeds = []
            trial_tongue_visible = []
            trial_paw_speeds = []
            trial_paw_visible = []

            side_ts, side_ft = get_trial_video(h5, view_groups[0], trial_idx)
            bottom_ts, bottom_ft = get_trial_video(h5, view_groups[1], trial_idx)

            for feat_idx in tongue_indices[0]:
                if side_ts is None:
                    continue
                coords = side_ts[:, :2, feat_idx]
                coords_interp = interpolate_coords(coords, side_ft, align_time, vidshift)
                speed, visible = feature_speed(coords_interp, tongue_feature=True)
                trial_tongue_speeds.append(speed)
                trial_tongue_visible.append(visible)

            for feat_idx in tongue_indices[1]:
                if bottom_ts is None:
                    continue
                coords = bottom_ts[:, :2, feat_idx]
                coords_interp = interpolate_coords(coords, bottom_ft, align_time, vidshift)
                speed, visible = feature_speed(coords_interp, tongue_feature=True)
                trial_tongue_speeds.append(speed)
                trial_tongue_visible.append(visible)

            for feat_idx in paw_indices[1]:
                if bottom_ts is None:
                    continue
                coords = bottom_ts[:, :2, feat_idx]
                coords_interp = interpolate_coords(coords, bottom_ft, align_time, vidshift)
                speed, visible = feature_speed(coords_interp, tongue_feature=False)
                trial_paw_speeds.append(speed)
                trial_paw_visible.append(visible)

            agg_tongue, agg_tongue_visible = aggregate_speeds(trial_tongue_speeds, trial_tongue_visible)
            if trial_paw_speeds:
                agg_paw = np.mean(np.stack(trial_paw_speeds, axis=0), axis=0)
                agg_paw_visible = np.ones(TIME.size, dtype=bool)
            else:
                agg_paw = np.full(TIME.size, np.nan, dtype=np.float64)
                agg_paw_visible = np.zeros(TIME.size, dtype=bool)

            me_trace = np.asarray(motion_energy_data[trial_idx]).reshape(-1)
            if side_ts is None:
                motion = np.full(TIME.size, np.nan, dtype=np.float64)
                motion_visible = np.zeros(TIME.size, dtype=bool)
            else:
                motion = interpolate_motion_energy(me_trace, side_ft, align_time, vidshift)
                motion_visible = np.isfinite(motion)

            tongue_series.append(agg_tongue)
            tongue_visible.append(agg_tongue_visible)
            paw_series.append(agg_paw)
            paw_visible.append(agg_paw_visible)
            me_series.append(motion)
            me_visible.append(motion_visible)

    tongue_series = np.asarray(tongue_series, dtype=np.float64)
    tongue_visible = np.asarray(tongue_visible, dtype=bool)
    paw_series = np.asarray(paw_series, dtype=np.float64)
    paw_visible = np.asarray(paw_visible, dtype=bool)
    me_series = np.asarray(me_series, dtype=np.float64)
    me_visible = np.asarray(me_visible, dtype=bool)

    tongue_thresh = float(np.nanpercentile(tongue_series[tongue_visible], 50))
    paw_thresh = float(np.nanpercentile(paw_series[paw_visible], 50))
    me_thresh = float(np.nanpercentile(me_series[me_visible], 50))

    outputs = []
    for out_idx, trial_idx in enumerate(valid_trials):
        lick_code = trial_choice_code(bp, trial_idx)
        context_code = 0 if bp["autowater"][trial_idx] else 1
        outcome_code = trial_outcome_code(bp, trial_idx)

        trial_output = np.empty((len(OUTPUT_NAMES), TIME.size), dtype=np.int64)
        trial_output[0] = lick_code
        trial_output[1] = context_code
        trial_output[2] = outcome_code

        trial_output[3] = np.where(~tongue_visible[out_idx], 2, (tongue_series[out_idx] >= tongue_thresh).astype(np.int64))
        trial_output[4] = np.where(~paw_visible[out_idx], 2, (paw_series[out_idx] >= paw_thresh).astype(np.int64))
        trial_output[5] = np.where(~me_visible[out_idx], 2, (me_series[out_idx] >= me_thresh).astype(np.int64))
        outputs.append(trial_output)

    return outputs, {
        "tongue_velocity": tongue_thresh,
        "paw_velocity": paw_thresh,
        "motion_energy": me_thresh,
    }


def process_session(spec: SessionSpec):
    with h5py.File(spec.data_path, "r") as h5:
        bp = load_bp_fields(h5)
        valid_mask = (~bp["early"]) & (~bp["stim_enable"])
        valid_trials = np.flatnonzero(valid_mask)

        clusters = load_probe_clusters(h5, spec.probe)
        neural_trials, n_neurons = compute_neural_trials(bp, clusters, valid_mask)

    if n_neurons < 10:
        raise ValueError(f"{spec.session_id}: only {n_neurons} neurons after filtering")
    if len(valid_trials) < 2:
        raise ValueError(f"{spec.session_id}: fewer than 2 valid trials after filtering")

    outputs, thresholds = compute_behavioral_outputs(spec, bp, valid_trials)
    inputs = [TIME[np.newaxis, :].astype(np.float32).copy() for _ in valid_trials]

    session_info = {
        "session_id": spec.session_id,
        "animal": spec.animal,
        "date": spec.date,
        "probe": spec.probe,
        "n_trials_total": int(bp["Ntrials"]),
        "n_trials_kept": int(len(valid_trials)),
        "n_neurons_kept": int(n_neurons),
        "wc_trials_kept": int(bp["autowater"][valid_trials].sum()),
        "dr_trials_kept": int((~bp["autowater"][valid_trials]).sum()),
        "thresholds": thresholds,
    }

    return {
        "neural": neural_trials,
        "input": inputs,
        "output": outputs,
        "session_info": session_info,
    }


def build_dataset():
    sessions = [process_session(spec) for spec in SESSION_SPECS]

    subjects = []
    subject_lookup = {}
    subject_idx = []
    for spec in SESSION_SPECS:
        if spec.animal not in subject_lookup:
            subject_lookup[spec.animal] = len(subjects)
            subjects.append(spec.animal)
        subject_idx.append(subject_lookup[spec.animal])

    data = {
        "neural": [session["neural"] for session in sessions],
        "input": [session["input"] for session in sessions],
        "output": [session["output"] for session in sessions],
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": ["ALM"],
        "brain_region_idx": [
            np.zeros(session["neural"][0].shape[0], dtype=np.int64) for session in sessions
        ],
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "output_values": OUTPUT_VALUES,
        "metadata": {
            "task_description": (
                "Two-context directional licking task with delayed-response (DR) and water-cued (WC) "
                "trials. Neural activity is used to decode lick direction, context, outcome, and "
                "time-varying movement variables."
            ),
            "source_cohort": "Figure 8 two-context ALM ephys sessions from Hasnain, Birnbaum et al. 2024",
            "time_bin_size": DT * 1000.0,
            "temporal_alignment_event": "Go cue / water delivery onset (obj.bp.ev.goCue)",
            "off_start": TMIN,
            "off_end": TMAX,
            "trial_filter": "~early & ~stim.enable",
            "neural_processing": {
                "bin_size_s": DT,
                "smoothing": "causal Gaussian, window 15 bins, reflect boundary",
                "unit_filter": "all non-garbage/non-noisy units with mean firing rate > 1 Hz",
            },
            "session_info": [session["session_info"] for session in sessions],
        },
    }
    return data


def main():
    data = build_dataset()
    with OUTPUT_PATH.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved converted dataset to {OUTPUT_PATH}")
    print(f"Sessions: {len(data['neural'])}")
    print(f"Trials per session: {[len(session) for session in data['neural']]}")


if __name__ == "__main__":
    main()
