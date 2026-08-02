import argparse
import copy
import json
import os
import pickle
from collections import OrderedDict

import h5py
import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d


SESSION_SPECS = [
    {"animal": "JEB6", "date": "2021-04-18", "probes": [2]},
    {"animal": "JEB7", "date": "2021-04-29", "probes": [1]},
    {"animal": "JEB7", "date": "2021-04-30", "probes": [1]},
    {"animal": "EKH1", "date": "2021-08-07", "probes": [2]},
    {"animal": "EKH3", "date": "2021-08-11", "probes": [2]},
    {"animal": "JGR2", "date": "2021-11-16", "probes": [1]},
    {"animal": "JGR2", "date": "2021-11-17", "probes": [1]},
    {"animal": "JGR3", "date": "2021-11-18", "probes": [1]},
    {"animal": "JEB19", "date": "2023-04-21", "probes": [1]},
    {"animal": "JEB19", "date": "2023-04-20", "probes": [1]},
    {"animal": "JEB19", "date": "2023-04-19", "probes": [1]},
    {"animal": "JEB19", "date": "2023-04-18", "probes": [1]},
]

DATA_DIR = "/app/data"
EPHYS_DIR = os.path.join(DATA_DIR, "Ephys_Behavior")
ALIGN_EVENT = "goCue"
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
SMOOTH = 15
BCTYPE = "reflect"
LOW_FR_HZ = 1.0

TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2

INPUT_NAMES = ["time_from_go_cue_sec"]
OUTPUT_NAMES = [
    "lick_direction",
    "behavioral_context",
    "outcome",
    "tongue_velocity",
    "paw_velocity",
    "motion_energy",
]
OUTPUT_VALUES = [
    ["left", "right"],
    ["WC", "DR"],
    ["incorrect", "correct"],
    ["lt_p50", "ge_p50"],
    ["lt_p50", "ge_p50"],
    ["lt_p50", "ge_p50"],
]


def matlab_mode(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    vals, counts = np.unique(x, return_counts=True)
    return float(vals[np.argmax(counts)])


def gausswin(n: int, alpha: float = 2.5) -> np.ndarray:
    idx = np.arange(n, dtype=np.float64) - (n - 1) / 2
    half = (n - 1) / 2
    if half == 0:
        return np.ones(1, dtype=np.float64)
    return np.exp(-0.5 * (alpha * idx / half) ** 2)


def my_smooth(x: np.ndarray, n: int, bctype: str = "none") -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    squeeze = False
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    if n in (0, 1):
        return x[:, 0] if squeeze else x
    if bctype.lower() == "reflect":
        x_filt = np.concatenate([x[:n], x], axis=0)
        trim = n
    elif bctype.lower() == "zeropad":
        x_filt = np.concatenate([np.zeros((n, x.shape[1]), dtype=x.dtype), x], axis=0)
        trim = n
    else:
        x_filt = x
        trim = 0
    kern = gausswin(n)
    kern[: n // 2] = 0
    kern /= kern.sum()
    out = np.zeros_like(x_filt, dtype=np.float64)
    for col in range(x.shape[1]):
        out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")
    out = out[trim:]
    return out[:, 0] if squeeze else out


def decode_char_dataset(ds: h5py.Dataset) -> str:
    arr = np.array(ds)
    return "".join(chr(int(x)) for x in arr.reshape(-1, order="F") if int(x) != 0)


def read_numeric_dataset(ds: h5py.Dataset):
    arr = np.array(ds)
    if arr.ndim == 0:
        return arr.item()
    if arr.ndim == 2 and 1 in arr.shape:
        return arr.reshape(-1, order="F")
    return arr


def deref_cell_dataset(f: h5py.File, ds: h5py.Dataset):
    refs = np.array(ds).reshape(-1, order="F")
    return [f[ref] for ref in refs]


def deref_numeric_list(f: h5py.File, ds: h5py.Dataset):
    return [np.asarray(read_numeric_dataset(f[ref]), dtype=np.float64) for ref in np.array(ds).reshape(-1, order="F")]


def deref_string_list(f: h5py.File, ds: h5py.Dataset):
    return [decode_char_dataset(f[ref]) for ref in np.array(ds).reshape(-1, order="F")]


def load_behavior(f: h5py.File) -> dict:
    bp = f["obj"]["bp"]
    out = {
        "Ntrials": int(np.asarray(read_numeric_dataset(bp["Ntrials"])).item()),
        "R": np.asarray(read_numeric_dataset(bp["R"]), dtype=bool),
        "L": np.asarray(read_numeric_dataset(bp["L"]), dtype=bool),
        "hit": np.asarray(read_numeric_dataset(bp["hit"]), dtype=bool),
        "miss": np.asarray(read_numeric_dataset(bp["miss"]), dtype=bool),
        "no": np.asarray(read_numeric_dataset(bp["no"]), dtype=bool),
        "early": np.asarray(read_numeric_dataset(bp["early"]), dtype=bool),
        "autowater": np.asarray(read_numeric_dataset(bp["autowater"]), dtype=bool),
    }
    if "stim" in bp and "enable" in bp["stim"]:
        out["stim_enable"] = np.asarray(read_numeric_dataset(bp["stim"]["enable"]), dtype=bool)
    else:
        out["stim_enable"] = np.zeros(out["Ntrials"], dtype=bool)
    ev = bp["ev"]
    out["ev"] = {
        "bitStart": np.asarray(read_numeric_dataset(ev["bitStart"]), dtype=np.float64),
        "sample": np.asarray(read_numeric_dataset(ev["sample"]), dtype=np.float64),
        "delay": np.asarray(read_numeric_dataset(ev["delay"]), dtype=np.float64),
        "goCue": np.asarray(read_numeric_dataset(ev["goCue"]), dtype=np.float64),
        "reward": np.asarray(read_numeric_dataset(ev["reward"]), dtype=np.float64),
    }
    return out


def cluster_quality_is_kept(quality: str) -> bool:
    quality = str(quality).strip()
    return quality not in {"garbage", "gabrga", "noisy", "real?"}


def cluster_quality_is_single(quality: str) -> bool:
    q = str(quality).strip().lower()
    return any(tag in q for tag in ("fair", "good", "great", "excellent"))


def load_probe_clusters(f: h5py.File, probe_number: int):
    clu_cells = np.array(f["obj"]["clu"]).reshape(-1, order="F")
    probe_group = f[clu_cells[probe_number - 1]]
    qualities = deref_string_list(f, probe_group["quality"])
    trials = deref_numeric_list(f, probe_group["trial"])
    trialtm = deref_numeric_list(f, probe_group["trialtm"])
    return [
        {
            "quality": qualities[i],
            "trial": np.asarray(trials[i], dtype=np.int64) - 1,
            "trialtm": np.asarray(trialtm[i], dtype=np.float64),
        }
        for i in range(len(qualities))
    ]


def get_video_offset(f: h5py.File, behavior: dict) -> float:
    if "sglx" not in f["obj"] or "bitcode" not in f["obj"]["sglx"]:
        return 0.5
    sglx = f["obj"]["sglx"]
    bitstart = np.asarray(read_numeric_dataset(sglx["bitcode"]["bitstart"]), dtype=np.float64)
    fs = float(np.asarray(read_numeric_dataset(sglx["fs"])).item())
    return matlab_mode(bitstart) / fs - matlab_mode(behavior["ev"]["bitStart"])


def load_traj_feature_series(
    f: h5py.File,
    behavior: dict,
    time_axis: np.ndarray,
    feature_name: str,
    view_index: int,
):
    traj_refs = np.array(f["obj"]["traj"]).reshape(-1, order="F")
    view_group = f[traj_refs[view_index - 1]]
    feature_refs = np.array(view_group["featNames"]).reshape(-1, order="F")
    feature_names = None
    feature_indices = {}
    feature_names_by_trial = {}
    for trial_idx, feat_ref in enumerate(feature_refs):
        feats = deref_string_list(f, f[feat_ref])
        if feature_name in feats:
            feature_names = feats
            feature_indices[trial_idx] = feats.index(feature_name)
        feature_names_by_trial[trial_idx] = feats
    if feature_names is None:
        raise KeyError(f"Feature {feature_name} not found in view {view_index}")

    vidshift = get_video_offset(f, behavior)
    ntrials = behavior["Ntrials"]
    xpos = np.full((time_axis.size, ntrials), np.nan, dtype=np.float64)
    ypos = np.full((time_axis.size, ntrials), np.nan, dtype=np.float64)

    ndropped_refs = np.array(view_group["NdroppedFrames"]).reshape(-1, order="F")
    frame_refs = np.array(view_group["frameTimes"]).reshape(-1, order="F")
    ts_refs = np.array(view_group["ts"]).reshape(-1, order="F")

    for trial_idx in range(ntrials):
        ndropped = np.asarray(read_numeric_dataset(f[ndropped_refs[trial_idx]]), dtype=np.float64)
        if ndropped.size and np.isnan(ndropped).all():
            continue
        if trial_idx not in feature_indices:
            continue
        feat_idx = feature_indices[trial_idx]
        frame_times = np.asarray(read_numeric_dataset(f[frame_refs[trial_idx]]), dtype=np.float64)
        ts = np.asarray(read_numeric_dataset(f[ts_refs[trial_idx]]), dtype=np.float64)
        if ts.ndim != 3:
            continue
        ts = normalize_traj_array(ts, len(feature_names_by_trial[trial_idx]))
        coords = ts[:, 0:2, feat_idx]
        if "tongue" not in feature_name:
            coords = my_smooth(coords, 1, "reflect")
        shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
        if shifted_t.ndim != 1 or shifted_t.size == 0:
            continue
        for dim in range(2):
            valid = np.isfinite(shifted_t) & np.isfinite(coords[:, dim])
            if valid.sum() < 2:
                continue
            interp = interp1d(
                shifted_t[valid],
                coords[valid, dim],
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True,
            )
            vals = interp(time_axis)
            if dim == 0:
                xpos[:, trial_idx] = vals
            else:
                ypos[:, trial_idx] = vals
        if "tongue" not in feature_name:
            xpos[:, trial_idx] = fill_nearest(xpos[:, trial_idx])
            ypos[:, trial_idx] = fill_nearest(ypos[:, trial_idx])
    return xpos, ypos


def normalize_traj_array(ts: np.ndarray, nfeatures: int) -> np.ndarray:
    if ts.shape[2] == nfeatures:
        return ts
    if ts.shape[0] == nfeatures:
        return np.transpose(ts, (2, 1, 0))
    raise ValueError(f"Unrecognized trajectory shape {ts.shape} for {nfeatures} features")


def fill_nearest(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mask = np.isfinite(x)
    if mask.all():
        return x
    if not mask.any():
        return np.zeros_like(x)
    idx = np.arange(x.size)
    interp = interp1d(
        idx[mask],
        x[mask],
        kind="nearest",
        bounds_error=False,
        fill_value=(x[mask][0], x[mask][-1]),
        assume_sorted=True,
    )
    return interp(idx)


def compute_velocity_from_position(xpos: np.ndarray, ypos: np.ndarray, feature_name: str):
    xvel = np.full_like(xpos, np.nan, dtype=np.float64)
    yvel = np.full_like(ypos, np.nan, dtype=np.float64)
    for trial_idx in range(xpos.shape[1]):
        tsinterp = np.column_stack([xpos[:, trial_idx], ypos[:, trial_idx]])
        if not np.isfinite(tsinterp).any():
            if "tongue" in feature_name:
                xvel[:, trial_idx] = 0.0
                yvel[:, trial_idx] = 0.0
            else:
                xvel[:, trial_idx] = fill_nearest(xvel[:, trial_idx])
                yvel[:, trial_idx] = fill_nearest(yvel[:, trial_idx])
            continue
        basederiv = np.nanmedian(np.diff(tsinterp, axis=0), axis=0)
        if not np.isfinite(basederiv[0]):
            basederiv[0] = 0.0
        xvel[:, trial_idx] = np.gradient(tsinterp[:, 0])
        yvel[:, trial_idx] = np.gradient(tsinterp[:, 1])
        if "tongue" not in feature_name:
            xvel[:, trial_idx] = xvel[:, trial_idx] - basederiv[0]
            yvel[:, trial_idx] = yvel[:, trial_idx] - basederiv[0]
            xvel[:, trial_idx] = fill_nearest(xvel[:, trial_idx])
            yvel[:, trial_idx] = fill_nearest(yvel[:, trial_idx])
        else:
            xvel[:, trial_idx][~np.isfinite(xvel[:, trial_idx])] = 0.0
            yvel[:, trial_idx][~np.isfinite(yvel[:, trial_idx])] = 0.0
    return xvel, yvel


def choose_paw_feature(f: h5py.File) -> str:
    traj_refs = np.array(f["obj"]["traj"]).reshape(-1, order="F")
    view_group = f[traj_refs[1]]
    first_feats = deref_string_list(f, f[np.array(view_group["featNames"]).reshape(-1, order="F")[0]])
    for name in ("top_paw", "bottom_paw"):
        if name in first_feats:
            return name
    raise KeyError("No paw feature found in bottom-view trajectory data")


def load_motion_energy(animal: str, date: str, behavior: dict) -> tuple[np.ndarray, float]:
    path = os.path.join(EPHYS_DIR, f"motionEnergy_{animal}_{date}.mat")
    dat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
    raw = dat.data
    if not isinstance(raw, np.ndarray) and hasattr(raw, "data"):
        raw = raw.data
    move_thresh = float(dat.moveThresh)
    return raw, move_thresh


def align_motion_energy(
    f: h5py.File,
    behavior: dict,
    raw_motion_energy,
) -> np.ndarray:
    traj_refs = np.array(f["obj"]["traj"]).reshape(-1, order="F")
    view_group = f[traj_refs[0]]
    frame_refs = np.array(view_group["frameTimes"]).reshape(-1, order="F")
    ts_refs = np.array(view_group["ts"]).reshape(-1, order="F")
    vidshift = get_video_offset(f, behavior)
    ntrials = behavior["Ntrials"]
    aligned = np.full((TIME_AXIS.size, ntrials), np.nan, dtype=np.float64)
    for trial_idx in range(ntrials):
        me_trial = np.asarray(raw_motion_energy[trial_idx], dtype=np.float64).reshape(-1)
        try:
            frame_times = np.asarray(read_numeric_dataset(f[frame_refs[trial_idx]]), dtype=np.float64)
        except Exception:
            ts = np.asarray(read_numeric_dataset(f[ts_refs[trial_idx]]), dtype=np.float64)
            frame_times = np.arange(1, ts.shape[0] + 1, dtype=np.float64) / 400.0
        shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
        valid = np.isfinite(shifted_t) & np.isfinite(me_trial)
        if valid.sum() < 2:
            continue
        interp = interp1d(
            shifted_t[valid],
            me_trial[valid],
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
            assume_sorted=True,
        )
        aligned[:, trial_idx] = interp(TIME_AXIS)
        aligned[:, trial_idx] = fill_nearest(aligned[:, trial_idx])
    return aligned


def get_context_conditions(behavior: dict):
    hit = behavior["hit"]
    miss = behavior["miss"]
    no = behavior["no"]
    early = behavior["early"]
    stim = behavior["stim_enable"]
    autowater = behavior["autowater"]
    return [
        hit | miss | no,
        hit & ~stim & ~autowater,
        hit & ~stim & autowater,
        miss & ~stim & ~autowater,
        miss & ~stim & autowater,
        hit & ~stim & ~autowater & ~early,
        hit & ~stim & autowater & ~early,
    ]


def bin_spikes_for_session(behavior: dict, probes: list[dict]):
    ntrials = behavior["Ntrials"]
    conditions = get_context_conditions(behavior)
    kept_neural = []
    kept_quality = []
    single_unit_flags = []
    for probe in probes:
        for clu in probe:
            quality = clu["quality"]
            if not cluster_quality_is_kept(quality):
                continue
            trial_idx = clu["trial"]
            valid_trial_spikes = (trial_idx >= 0) & (trial_idx < ntrials)
            trial_idx = trial_idx[valid_trial_spikes]
            aligned = clu["trialtm"][valid_trial_spikes] - behavior["ev"][ALIGN_EVENT][trial_idx]
            trial_counts = np.zeros((TIME_AXIS.size, ntrials), dtype=np.float64)
            if aligned.size:
                unique_trials = np.unique(trial_idx)
                for t in unique_trials:
                    mask = trial_idx == t
                    counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
                    trial_counts[:, t] = my_smooth(counts / DT, SMOOTH, BCTYPE)
            psth = np.zeros((TIME_AXIS.size, len(conditions)), dtype=np.float64)
            for ci, mask in enumerate(conditions):
                trix = np.flatnonzero(mask)
                if trix.size:
                    psth[:, ci] = np.mean(trial_counts[:, trix], axis=1)
            mean_fr = float(np.mean(psth))
            if mean_fr > LOW_FR_HZ:
                kept_neural.append(trial_counts)
                kept_quality.append(quality)
                single_unit_flags.append(cluster_quality_is_single(quality))
    if not kept_neural:
        raise RuntimeError("No neurons survived quality and firing-rate filtering")
    neural = np.stack(kept_neural, axis=0)
    return neural, kept_quality, np.asarray(single_unit_flags, dtype=bool)


def actual_lick_direction(behavior: dict) -> np.ndarray:
    return ((behavior["R"] & behavior["hit"]) | (behavior["L"] & behavior["miss"])).astype(np.int64)


def analysis_trial_mask(behavior: dict) -> np.ndarray:
    return (behavior["hit"] | behavior["miss"]) & (~behavior["early"]) & (~behavior["no"]) & (~behavior["stim_enable"])


def repeat_labels(values: np.ndarray, n_time: int) -> np.ndarray:
    return np.repeat(values[:, None], n_time, axis=1)


def percentile_threshold(values: np.ndarray, drop_zeros_if_needed: bool = False) -> float:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    thresh = float(np.nanpercentile(vals, 50))
    if drop_zeros_if_needed and thresh <= 0:
        pos = vals[vals > 0]
        if pos.size:
            thresh = float(np.nanpercentile(pos, 50))
    return thresh


def build_session_payload(spec: dict) -> dict:
    data_path = os.path.join(EPHYS_DIR, f"data_structure_{spec['animal']}_{spec['date']}.mat")
    print(f"Loading {os.path.basename(data_path)}")
    with h5py.File(data_path, "r") as f:
        behavior = load_behavior(f)
        probes = [load_probe_clusters(f, probe_number) for probe_number in spec["probes"]]
        neural_all, kept_quality, single_flags = bin_spikes_for_session(behavior, probes)
        raw_motion_energy, manual_motion_thresh = load_motion_energy(spec["animal"], spec["date"], behavior)
        motion_energy = align_motion_energy(f, behavior, raw_motion_energy)
        tongue_xpos, tongue_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, "tongue", 1)
        paw_feature = choose_paw_feature(f)
        paw_xpos, paw_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, paw_feature, 2)

    tongue_xvel, tongue_yvel = compute_velocity_from_position(tongue_xpos, tongue_ypos, "tongue")
    paw_xvel, paw_yvel = compute_velocity_from_position(paw_xpos, paw_ypos, paw_feature)
    tongue_speed = np.sqrt(tongue_xvel**2 + tongue_yvel**2)
    paw_speed = np.sqrt(paw_xvel**2 + paw_yvel**2)

    use_trials = np.flatnonzero(analysis_trial_mask(behavior))
    if use_trials.size < 2:
        raise RuntimeError(f"{spec['animal']} {spec['date']} has fewer than 2 usable trials")

    neural_selected = neural_all[:, :, use_trials]
    tongue_selected = tongue_speed[:, use_trials]
    paw_selected = paw_speed[:, use_trials]
    me_selected = motion_energy[:, use_trials]

    tongue_thresh = percentile_threshold(tongue_selected, drop_zeros_if_needed=True)
    paw_thresh = percentile_threshold(paw_selected)
    me_thresh = percentile_threshold(me_selected)

    lick_dir = actual_lick_direction(behavior)[use_trials]
    context = (~behavior["autowater"][use_trials]).astype(np.int64)
    outcome = behavior["hit"][use_trials].astype(np.int64)

    input_trials = []
    output_trials = []
    neural_trials = []
    for local_idx, trial_idx in enumerate(use_trials):
        neural_trials.append(neural_selected[:, :, local_idx].astype(np.float32))
        input_trials.append(TIME_AXIS[None, :].astype(np.float32))
        output = np.vstack(
            [
                repeat_labels(np.asarray([lick_dir[local_idx]], dtype=np.int64), TIME_AXIS.size),
                repeat_labels(np.asarray([context[local_idx]], dtype=np.int64), TIME_AXIS.size),
                repeat_labels(np.asarray([outcome[local_idx]], dtype=np.int64), TIME_AXIS.size),
                (tongue_selected[:, local_idx] >= tongue_thresh).astype(np.int64)[None, :],
                (paw_selected[:, local_idx] >= paw_thresh).astype(np.int64)[None, :],
                (me_selected[:, local_idx] >= me_thresh).astype(np.int64)[None, :],
            ]
        )
        output_trials.append(output.astype(np.int64))

    session_stats = {
        "session_id": f"{spec['animal']}_{spec['date']}",
        "animal": spec["animal"],
        "date": spec["date"],
        "n_trials_total": int(behavior["Ntrials"]),
        "n_trials_kept": int(use_trials.size),
        "n_neurons_kept": int(neural_all.shape[0]),
        "n_single_unit_quality_kept": int(single_flags.sum()),
        "n_dr_trials_kept": int(np.sum(context == 1)),
        "n_wc_trials_kept": int(np.sum(context == 0)),
        "n_correct_trials_kept": int(np.sum(outcome == 1)),
        "n_incorrect_trials_kept": int(np.sum(outcome == 0)),
        "n_right_trials_kept": int(np.sum(lick_dir == 1)),
        "n_left_trials_kept": int(np.sum(lick_dir == 0)),
        "tongue_speed_p50": tongue_thresh,
        "paw_speed_p50": paw_thresh,
        "motion_energy_p50": me_thresh,
        "manual_motion_energy_move_thresh": manual_motion_thresh,
        "quality_labels_kept": kept_quality,
        "usable_trial_indices_0based": use_trials.tolist(),
        "event_times_sec_relative_to_go_cue": {
            "bitStart_mode": float(matlab_mode(behavior["ev"]["bitStart"]) - matlab_mode(behavior["ev"][ALIGN_EVENT])),
            "sample_mode": float(matlab_mode(behavior["ev"]["sample"]) - matlab_mode(behavior["ev"][ALIGN_EVENT])),
            "delay_mode": float(matlab_mode(behavior["ev"]["delay"]) - matlab_mode(behavior["ev"][ALIGN_EVENT])),
            "goCue_mode": float(matlab_mode(behavior["ev"]["goCue"]) - matlab_mode(behavior["ev"][ALIGN_EVENT])),
        },
    }
    return {
        "spec": spec,
        "neural_trials": neural_trials,
        "input_trials": input_trials,
        "output_trials": output_trials,
        "brain_region_idx": np.zeros(neural_all.shape[0], dtype=np.int64),
        "session_stats": session_stats,
    }


def build_dataset() -> tuple[dict, dict]:
    subjects = []
    subject_lookup = OrderedDict()
    data = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": subjects,
        "subject_idx": [],
        "brain_regions": ["ALM"],
        "brain_region_idx": [],
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "output_values": OUTPUT_VALUES,
        "metadata": {},
    }
    session_stats = []
    total_single = 0

    for spec in SESSION_SPECS:
        payload = build_session_payload(spec)
        sid = spec["animal"]
        if sid not in subject_lookup:
            subject_lookup[sid] = len(subject_lookup)
            subjects.append(sid)
        data["neural"].append(payload["neural_trials"])
        data["input"].append(payload["input_trials"])
        data["output"].append(payload["output_trials"])
        data["subject_idx"].append(subject_lookup[sid])
        data["brain_region_idx"].append(payload["brain_region_idx"])
        session_stats.append(payload["session_stats"])
        total_single += payload["session_stats"]["n_single_unit_quality_kept"]

    data["subject_idx"] = np.asarray(data["subject_idx"], dtype=np.int64)
    data["metadata"] = {
        "task_description": (
            "Alternating delayed-response (DR) and water-cued (WC) directional licking task. "
            "Neural activity is aligned to go cue onset, using the two-context ALM ephys cohort "
            "from the Figure 8 context analyses in the reference code."
        ),
        "time_bin_size": float(DT * 1000.0),
        "temporal_alignment_event": "Go cue onset (reference code alignEvent = goCue)",
        "off_start": float(TMIN),
        "off_end": float(TMAX),
        "time_axis_sec": TIME_AXIS.astype(np.float32),
        "session_ids": [s["session_id"] for s in session_stats],
        "session_stats": session_stats,
        "reference_processing": {
            "tmin_sec": TMIN,
            "tmax_sec": TMAX,
            "dt_sec": DT,
            "smooth_window_bins": SMOOTH,
            "smooth_boundary_condition": BCTYPE,
            "low_firing_rate_hz": LOW_FR_HZ,
            "cluster_qualities": "all_except_garbage_noisy_realq",
            "video_alignment": "frameTimes - video_offset - goCue",
            "motion_energy_source": "motionEnergy_Animal_Date.mat, resampled to neural time axis",
            "velocity_definition": (
                "Speed magnitude sqrt(xvel^2 + yvel^2) from reference x/y velocity components; "
                "tongue invisibility periods are set to zero as in the MATLAB code."
            ),
        },
        "sanity_checks": {
            "expected_context_sessions_from_reference_code": 12,
            "expected_context_units_from_paper": 522,
            "expected_context_single_units_from_paper": 214,
            "converted_total_neurons": int(sum(len(s) and s[0].shape[0] for s in data["neural"])),
            "converted_total_single_unit_quality_neurons": int(total_single),
            "converted_total_trials": int(sum(len(s) for s in data["neural"])),
        },
        "notes": [
            "This conversion follows the session subset used in the Figure 8 context analyses in the repository.",
            "Trials with early licks, no responses, or stimulation enabled are excluded from the decoder dataset.",
            "Behavioral outputs are tiled across time so the decoder can be trained in the provided timepoint-wise format.",
        ],
    }
    return data, {"session_stats": session_stats}


def make_sample_dataset(full_data: dict, session_count: int) -> dict:
    sample = copy.deepcopy(full_data)
    sample["neural"] = sample["neural"][:session_count]
    sample["input"] = sample["input"][:session_count]
    sample["output"] = sample["output"][:session_count]
    sample["subject_idx"] = sample["subject_idx"][:session_count]
    sample["brain_region_idx"] = sample["brain_region_idx"][:session_count]
    sample_ids = sample["metadata"]["session_ids"][:session_count]
    sample["metadata"]["session_ids"] = sample_ids
    sample["metadata"]["session_stats"] = sample["metadata"]["session_stats"][:session_count]
    sample["metadata"]["notes"] = list(sample["metadata"]["notes"]) + [
        f"Sample dataset contains the first {session_count} sessions from the full converted cohort."
    ]
    present_subjects = sorted(set(sample["subject_idx"].tolist()))
    remap = {old: new for new, old in enumerate(present_subjects)}
    sample["subjects"] = [sample["subjects"][old] for old in present_subjects]
    sample["subject_idx"] = np.asarray([remap[int(x)] for x in sample["subject_idx"]], dtype=np.int64)
    return sample


def summarize_dataset(data: dict):
    nsessions = len(data["neural"])
    ntrials = [len(x) for x in data["neural"]]
    nneurons = [session[0].shape[0] for session in data["neural"]]
    total_trials = int(sum(ntrials))
    total_neurons = int(sum(nneurons))
    print(f"Sessions: {nsessions}")
    print(f"Subjects: {len(data['subjects'])} -> {data['subjects']}")
    print(f"Trials per session: {ntrials}")
    print(f"Total trials: {total_trials}")
    print(f"Neurons per session: {nneurons}")
    print(f"Total neurons across sessions: {total_neurons}")
    print("Session-level kept-trial / kept-neuron summary:")
    for stat in data["metadata"]["session_stats"]:
        print(
            f"  {stat['session_id']}: trials={stat['n_trials_kept']} "
            f"(DR={stat['n_dr_trials_kept']}, WC={stat['n_wc_trials_kept']}), "
            f"neurons={stat['n_neurons_kept']}, single_quality={stat['n_single_unit_quality_kept']}"
        )
    print("Reference sanity check:")
    sc = data["metadata"]["sanity_checks"]
    print(
        f"  Converted neurons {sc['converted_total_neurons']} vs paper {sc['expected_context_units_from_paper']}; "
        f"converted single-quality {sc['converted_total_single_unit_quality_neurons']} "
        f"vs paper {sc['expected_context_single_units_from_paper']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Convert Hasnain/Birnbaum alternating-context ALM sessions")
    parser.add_argument(
        "--dataset",
        choices=["full", "sample", "both"],
        default="both",
        help="Which output dataset(s) to write",
    )
    parser.add_argument("--full-output", default="/app/converted_data.pkl")
    parser.add_argument("--sample-output", default="/app/sample_data.pkl")
    parser.add_argument("--sample-sessions", type=int, default=4)
    parser.add_argument("--stats-json", default=None)
    args = parser.parse_args()

    full_data, stats = build_dataset()
    summarize_dataset(full_data)

    if args.dataset in {"full", "both"}:
        with open(args.full_output, "wb") as f:
            pickle.dump(full_data, f)
        print(f"Wrote full dataset to {args.full_output}")

    if args.dataset in {"sample", "both"}:
        sample = make_sample_dataset(full_data, args.sample_sessions)
        with open(args.sample_output, "wb") as f:
            pickle.dump(sample, f)
        print(f"Wrote sample dataset to {args.sample_output}")

    if args.stats_json:
        with open(args.stats_json, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Wrote stats to {args.stats_json}")


if __name__ == "__main__":
    main()
