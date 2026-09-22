import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d


DATA_SUBDIR = "Ephys_Behavior"
SESSION_SPECS = [
    {"subject": "JEB6", "date": "2021-04-18", "probe": 2},
    {"subject": "JEB7", "date": "2021-04-29", "probe": 1},
    {"subject": "JEB7", "date": "2021-04-30", "probe": 1},
    {"subject": "EKH1", "date": "2021-08-07", "probe": 2},
    {"subject": "EKH3", "date": "2021-08-11", "probe": 2},
    {"subject": "JGR2", "date": "2021-11-16", "probe": 1},
    {"subject": "JGR2", "date": "2021-11-17", "probe": 1},
    {"subject": "JGR3", "date": "2021-11-18", "probe": 1},
    {"subject": "JEB19", "date": "2023-04-18", "probe": 1},
    {"subject": "JEB19", "date": "2023-04-19", "probe": 1},
    {"subject": "JEB19", "date": "2023-04-20", "probe": 1},
    {"subject": "JEB19", "date": "2023-04-21", "probe": 1},
]

TIME_MIN = -3.0
TIME_MAX = 2.5
DT = 1.0 / 100.0
SMOOTH = 15
BOUNDARY = "reflect"
ADVANCE_MOVEMENT = 0.0

TONGUE_FEATURES = [
    (1, "tongue"),
    (1, "left_tongue"),
    (1, "right_tongue"),
    (2, "top_tongue"),
    (2, "topleft_tongue"),
    (2, "bottom_tongue"),
    (2, "bottomleft_tongue"),
]
PAW_FEATURES = [
    (2, "top_paw"),
    (2, "bottom_paw"),
]


def matlab_mode(values: np.ndarray):
    values = np.asarray(values).reshape(-1)
    values = values[np.isfinite(values)]
    uniq, counts = np.unique(values, return_counts=True)
    return uniq[np.argmax(counts)]


def gausswin(n: int, alpha: float = 2.5) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    if n == 1:
        return np.ones((1,), dtype=np.float64)
    idx = np.arange(n, dtype=np.float64) - (n - 1.0) / 2.0
    sigma = (n - 1.0) / (2.0 * alpha)
    return np.exp(-0.5 * (idx / sigma) ** 2)


def my_smooth(x: np.ndarray, n: int, bctype: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    squeeze = False
    if arr.ndim == 1:
        arr = arr[:, None]
        squeeze = True

    if n in (0, 1):
        out = arr
        return out[:, 0] if squeeze else out

    if bctype.lower() == "reflect":
        arr_filt = np.concatenate([arr[:n, :], arr], axis=0)
        trim = n
    elif bctype.lower() == "zeropad":
        arr_filt = np.concatenate([np.zeros((n, arr.shape[1])), arr], axis=0)
        trim = n
    else:
        arr_filt = arr
        trim = 0

    kern = gausswin(n)
    kern[: len(kern) // 2] = 0.0
    kern = kern / np.sum(kern)

    out = np.zeros_like(arr_filt)
    for col in range(arr_filt.shape[1]):
        out[:, col] = np.convolve(arr_filt[:, col], kern, mode="same")
    out = out[trim:, :]
    return out[:, 0] if squeeze else out


def fill_nearest_1d(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    if out.ndim != 1:
        raise ValueError("fill_nearest_1d expects a 1D array")
    valid = np.isfinite(out)
    if valid.all() or not valid.any():
        return out

    idx = np.arange(out.size)
    prev_idx = np.where(valid, idx, -1)
    prev_idx = np.maximum.accumulate(prev_idx)
    next_idx = np.where(valid, idx, out.size)
    next_idx = np.minimum.accumulate(next_idx[::-1])[::-1]

    missing = ~valid
    use_prev = (prev_idx >= 0) & (
        (next_idx == out.size) | ((idx - prev_idx) <= (next_idx - idx))
    )
    nearest = np.where(use_prev, prev_idx, next_idx)
    nearest = nearest[missing]
    out[missing] = out[nearest]
    return out


def interp_with_nans(old_t: np.ndarray, y: np.ndarray, new_t: np.ndarray) -> np.ndarray:
    old_t = np.asarray(old_t, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if old_t.size < 2 or y.size < 2:
        return np.full(new_t.shape, np.nan, dtype=np.float64)

    finite_t = np.isfinite(old_t)
    old_t = old_t[finite_t]
    y = y[finite_t]
    if old_t.size < 2:
        return np.full(new_t.shape, np.nan, dtype=np.float64)

    # Keep MATLAB-like NaN propagation in the values rather than bridging gaps.
    interp = interp1d(
        old_t,
        y,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )
    return np.asarray(interp(new_t), dtype=np.float64)


def decode_mat_string(dataset: h5py.Dataset) -> str:
    arr = np.asarray(dataset[()])
    if arr.dtype == np.uint16:
        chars = [chr(int(x)) for x in arr.reshape(-1) if int(x) != 0]
        return "".join(chars)
    if arr.dtype.kind in {"S", "U"}:
        return "".join(arr.astype(str).reshape(-1))
    return str(arr.squeeze())


def read_h5_numeric(obj) -> np.ndarray:
    return np.asarray(obj[()]).squeeze()


def read_trial_ref_numeric(mat: h5py.File, ref) -> np.ndarray:
    if not ref:
        return np.array([], dtype=np.float64)
    return np.asarray(mat[ref][()]).squeeze()


def read_trial_ref_string_list(mat: h5py.File, ref) -> list[str]:
    cell = mat[ref]
    refs = np.asarray(cell[()]).squeeze()
    return [decode_mat_string(mat[item]) for item in refs]


def load_trial_ts(mat: h5py.File, ref) -> np.ndarray:
    ts = np.asarray(mat[ref][()])
    if ts.ndim != 3:
        raise ValueError(f"Unexpected ts shape: {ts.shape}")
    return np.transpose(ts, (2, 1, 0))


def quality_is_usable(label: str) -> bool:
    label = label.strip().lower()
    return label not in {"garbage", "gabrga", "noisy", "real?"}


def compute_velocity(xpos: np.ndarray, ypos: np.ndarray, is_tongue: bool) -> tuple[np.ndarray, np.ndarray]:
    xvel = np.gradient(xpos)
    yvel = np.gradient(ypos)
    if not is_tongue:
        if np.any(np.isfinite(xpos)):
            basederiv_x = np.nanmedian(np.diff(np.column_stack([xpos, ypos]), axis=0)[:, 0])
            if not np.isfinite(basederiv_x):
                basederiv_x = 0.0
        else:
            basederiv_x = 0.0
        xvel = xvel - basederiv_x
        yvel = yvel - basederiv_x
        xvel = fill_nearest_1d(xvel)
        yvel = fill_nearest_1d(yvel)
    else:
        xvel[np.isnan(xvel)] = 0.0
        yvel[np.isnan(yvel)] = 0.0
    return xvel, yvel


def get_time_axis() -> np.ndarray:
    edges = np.arange(TIME_MIN, TIME_MAX + DT, DT, dtype=np.float64)
    return edges[:-1] + DT / 2.0


def get_edges() -> np.ndarray:
    return np.arange(TIME_MIN, TIME_MAX + DT, DT, dtype=np.float64)


def load_basic_behavior(mat: h5py.File) -> dict[str, np.ndarray]:
    bp = mat["obj"]["bp"]

    data = {
        "ntrials": int(read_h5_numeric(bp["Ntrials"])),
        "R": read_h5_numeric(bp["R"]).astype(bool),
        "L": read_h5_numeric(bp["L"]).astype(bool),
        "hit": read_h5_numeric(bp["hit"]).astype(bool),
        "miss": read_h5_numeric(bp["miss"]).astype(bool),
        "no": read_h5_numeric(bp["no"]).astype(bool),
        "early": read_h5_numeric(bp["early"]).astype(bool),
        "autowater": read_h5_numeric(bp["autowater"]).astype(bool),
        "bitStart": read_h5_numeric(bp["ev"]["bitStart"]).astype(np.float64),
        "sample": read_h5_numeric(bp["ev"]["sample"]).astype(np.float64),
        "delay": read_h5_numeric(bp["ev"]["delay"]).astype(np.float64),
        "goCue": read_h5_numeric(bp["ev"]["goCue"]).astype(np.float64),
    }

    if "stim" in bp and isinstance(bp["stim"], h5py.Group) and "enable" in bp["stim"]:
        data["stim_enable"] = read_h5_numeric(bp["stim"]["enable"]).astype(bool)
    else:
        data["stim_enable"] = np.zeros((data["ntrials"],), dtype=bool)

    data["lickL_refs"] = np.asarray(bp["ev"]["lickL"][()]).squeeze()
    data["lickR_refs"] = np.asarray(bp["ev"]["lickR"][()]).squeeze()
    return data


def get_condition_masks(beh: dict[str, np.ndarray]) -> list[np.ndarray]:
    hit = beh["hit"]
    miss = beh["miss"]
    no = beh["no"]
    early = beh["early"]
    autowater = beh["autowater"]
    stim_enable = beh["stim_enable"]

    return [
        hit | miss | no,
        hit & ~stim_enable & ~autowater,
        hit & ~stim_enable & autowater,
        miss & ~stim_enable & ~autowater,
        miss & ~stim_enable & autowater,
        hit & ~stim_enable & ~autowater & ~early,
        hit & ~stim_enable & autowater & ~early,
    ]


def load_neural_session(
    mat: h5py.File,
    probe: int,
    go_cue: np.ndarray,
    condition_masks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    edges = get_edges()
    ntrials = go_cue.size

    clu_ref = mat["obj"]["clu"][()][probe - 1, 0]
    clu = mat[clu_ref]
    if isinstance(clu, h5py.Dataset):
        raise ValueError(f"Probe {probe} is empty")

    quality_refs = np.asarray(clu["quality"][()]).squeeze()
    keep_unit_indices = []
    for unit_idx, ref in enumerate(quality_refs):
        label = decode_mat_string(mat[ref]) if ref else ""
        if quality_is_usable(label):
            keep_unit_indices.append(unit_idx)

    n_units_pre = len(keep_unit_indices)
    trialdat = np.zeros((n_units_pre, ntrials, edges.size - 1), dtype=np.float32)
    sites = np.zeros((n_units_pre,), dtype=np.int64)

    trial_refs = np.asarray(clu["trial"][()]).squeeze()
    trialtm_refs = np.asarray(clu["trialtm"][()]).squeeze()
    if "site" in clu:
        site_refs = np.asarray(clu["site"][()]).squeeze()
    elif "channel" in clu:
        site_refs = np.asarray(clu["channel"][()]).squeeze()
    else:
        site_refs = None

    for out_idx, unit_idx in enumerate(keep_unit_indices):
        spike_trials = read_trial_ref_numeric(mat, trial_refs[unit_idx]).astype(np.int64)
        spike_trialtm = read_trial_ref_numeric(mat, trialtm_refs[unit_idx]).astype(np.float64)
        if site_refs is not None:
            sites[out_idx] = int(read_trial_ref_numeric(mat, site_refs[unit_idx]))

        if spike_trials.size == 0:
            continue

        spike_trials = spike_trials - 1
        aligned_times = spike_trialtm - go_cue[spike_trials]
        for tr in np.unique(spike_trials):
            spk = aligned_times[spike_trials == tr]
            counts, _ = np.histogram(spk, bins=edges)
            rates = counts.astype(np.float64) / DT
            trialdat[out_idx, tr, :] = my_smooth(rates, SMOOTH, BOUNDARY).astype(np.float32)

    psth_by_cond = []
    for mask in condition_masks:
        if np.any(mask):
            psth = trialdat[:, mask, :].mean(axis=1)
        else:
            psth = np.zeros((trialdat.shape[0], trialdat.shape[2]), dtype=np.float32)
        psth_by_cond.append(psth)
    psth_by_cond = np.stack(psth_by_cond, axis=2)

    mean_frs = psth_by_cond.mean(axis=(1, 2))
    use = mean_frs > 1.0
    return trialdat[use], sites[use]


def load_traj_groups(mat: h5py.File) -> list[h5py.Group]:
    traj_refs = np.asarray(mat["obj"]["traj"][()]).squeeze()
    return [mat[ref] for ref in traj_refs]


def get_feature_maps(mat: h5py.File, traj_groups: list[h5py.Group], ntrials: int) -> list[dict[str, int]]:
    feature_maps: list[dict[str, int]] = []
    for group in traj_groups:
        feat_map = {}
        for trial_idx in range(ntrials):
            ref = group["featNames"][trial_idx, 0]
            if not ref:
                continue
            names = read_trial_ref_string_list(mat, ref)
            if names:
                feat_map = {name: idx for idx, name in enumerate(names)}
                break
        feature_maps.append(feat_map)
    return feature_maps


def get_video_offset(mat: h5py.File, bit_start: np.ndarray) -> float:
    fs = float(read_h5_numeric(mat["obj"]["sglx"]["fs"]))
    bitcode_starts = read_h5_numeric(mat["obj"]["sglx"]["bitcode"]["bitstart"]).astype(np.float64)
    return matlab_mode(bitcode_starts) / fs - matlab_mode(bit_start)


def load_trial_video_bundle(
    mat: h5py.File,
    traj_group: h5py.Group,
    trial_idx: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    nd_ref = traj_group["NdroppedFrames"][trial_idx, 0]
    nd = read_trial_ref_numeric(mat, nd_ref)
    if np.size(nd) == 1 and np.isnan(float(nd)):
        return None, None

    ts_ref = traj_group["ts"][trial_idx, 0]
    ts = load_trial_ts(mat, ts_ref)

    frame_ref = traj_group["frameTimes"][trial_idx, 0]
    frame_times = read_trial_ref_numeric(mat, frame_ref).astype(np.float64)
    if frame_times.size == 0 or not np.any(np.isfinite(frame_times)):
        frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0
    return ts, frame_times


def compute_composite_speed(
    mat: h5py.File,
    beh: dict[str, np.ndarray],
    feature_specs: list[tuple[int, str]],
) -> tuple[np.ndarray, np.ndarray]:
    ntrials = beh["ntrials"]
    taxis = get_time_axis() + ADVANCE_MOVEMENT
    vidshift = get_video_offset(mat, beh["bitStart"])
    traj_groups = load_traj_groups(mat)
    feature_maps = get_feature_maps(mat, traj_groups, ntrials)

    per_feature_speed = np.full(
        (len(feature_specs), ntrials, taxis.size), np.nan, dtype=np.float32
    )
    per_feature_visible = np.zeros(
        (len(feature_specs), ntrials, taxis.size), dtype=bool
    )

    views_needed = sorted({view for view, _ in feature_specs})
    for trial_idx in range(ntrials):
        bundles: dict[int, tuple[np.ndarray | None, np.ndarray | None]] = {}
        for view in views_needed:
            bundles[view] = load_trial_video_bundle(mat, traj_groups[view - 1], trial_idx)

        align_time = beh["goCue"][trial_idx]
        for feat_out_idx, (view, feat_name) in enumerate(feature_specs):
            ts, frame_times = bundles[view]
            feat_map = feature_maps[view - 1]
            feat_idx = feat_map.get(feat_name)
            if ts is None or frame_times is None or feat_idx is None:
                continue

            old_t = frame_times - vidshift - align_time
            x = ts[:, 0, feat_idx]
            y = ts[:, 1, feat_idx]
            x_interp = interp_with_nans(old_t, x, taxis)
            y_interp = interp_with_nans(old_t, y, taxis)
            visible = np.isfinite(x_interp) & np.isfinite(y_interp)

            is_tongue = "tongue" in feat_name
            x_proc = x_interp.copy()
            y_proc = y_interp.copy()
            if not is_tongue:
                x_proc = fill_nearest_1d(x_proc)
                y_proc = fill_nearest_1d(y_proc)

            xvel, yvel = compute_velocity(x_proc, y_proc, is_tongue=is_tongue)
            speed = np.sqrt(xvel**2 + yvel**2)

            per_feature_speed[feat_out_idx, trial_idx, :] = speed.astype(np.float32)
            per_feature_visible[feat_out_idx, trial_idx, :] = visible

    visible_counts = per_feature_visible.sum(axis=0)
    speed_sum = np.where(per_feature_visible, np.nan_to_num(per_feature_speed), 0.0).sum(axis=0)
    composite_speed = np.divide(
        speed_sum,
        np.maximum(visible_counts, 1),
        out=np.zeros_like(speed_sum, dtype=np.float32),
        where=np.maximum(visible_counts, 1) > 0,
    )
    composite_visible = visible_counts > 0
    return composite_speed, composite_visible


def load_motion_energy_aligned(
    mat: h5py.File,
    motion_path: Path,
    beh: dict[str, np.ndarray],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not motion_path.exists():
        return None, None

    motion_file = sio.loadmat(motion_path, squeeze_me=True, struct_as_record=False)
    me = motion_file["me"]
    motion_trials = np.asarray(me.data, dtype=object).reshape(-1)

    ntrials = beh["ntrials"]
    taxis = get_time_axis() + ADVANCE_MOVEMENT
    vidshift = get_video_offset(mat, beh["bitStart"])
    traj_groups = load_traj_groups(mat)

    aligned = np.full((ntrials, taxis.size), np.nan, dtype=np.float32)
    visible = np.zeros((ntrials, taxis.size), dtype=bool)

    for trial_idx in range(ntrials):
        bundle = load_trial_video_bundle(mat, traj_groups[0], trial_idx)
        ts, frame_times = bundle
        if ts is None or frame_times is None:
            continue

        trial_motion = np.asarray(motion_trials[trial_idx]).squeeze().astype(np.float64)
        if trial_motion.size == 0:
            continue

        old_t = frame_times - vidshift - beh["goCue"][trial_idx]
        trial_aligned = interp_with_nans(old_t, trial_motion, taxis)
        visible[trial_idx, :] = np.isfinite(trial_aligned)
        if np.any(visible[trial_idx, :]):
            aligned[trial_idx, :] = fill_nearest_1d(trial_aligned).astype(np.float32)

    return aligned, visible


def first_post_go_lick_direction(
    mat: h5py.File,
    lick_l_refs,
    lick_r_refs,
    go_cue: np.ndarray,
) -> np.ndarray:
    lick_dir = np.full((go_cue.size,), 2, dtype=np.int64)
    for trial_idx in range(go_cue.size):
        lick_l = read_trial_ref_numeric(mat, lick_l_refs[trial_idx]).astype(np.float64)
        lick_r = read_trial_ref_numeric(mat, lick_r_refs[trial_idx]).astype(np.float64)

        post_l = lick_l[lick_l >= go_cue[trial_idx] - 1e-9]
        post_r = lick_r[lick_r >= go_cue[trial_idx] - 1e-9]

        first_l = post_l[0] if post_l.size else np.inf
        first_r = post_r[0] if post_r.size else np.inf
        if first_l < first_r:
            lick_dir[trial_idx] = 0
        elif first_r < first_l:
            lick_dir[trial_idx] = 1
    return lick_dir


def discretize_with_visibility(
    values: np.ndarray,
    visible: np.ndarray,
    keep_mask: np.ndarray,
    absent_code: int,
) -> tuple[np.ndarray, float | None]:
    if values is None or visible is None:
        classes = np.full((int(np.sum(keep_mask)), get_time_axis().size), absent_code, dtype=np.int64)
        return classes, None

    keep_values = values[keep_mask]
    keep_visible = visible[keep_mask]
    if np.any(keep_visible):
        threshold = float(np.nanpercentile(keep_values[keep_visible], 50))
    else:
        threshold = None

    classes = np.full(keep_values.shape, absent_code, dtype=np.int64)
    if threshold is not None:
        present = keep_visible
        classes[present] = (keep_values[present] >= threshold).astype(np.int64)
    return classes, threshold


def process_session(data_root: Path, spec: dict) -> dict:
    session_tag = f"{spec['subject']}_{spec['date']}"
    session_path = data_root / DATA_SUBDIR / f"data_structure_{session_tag}.mat"
    motion_path = data_root / DATA_SUBDIR / f"motionEnergy_{session_tag}.mat"

    with h5py.File(session_path, "r") as mat:
        beh = load_basic_behavior(mat)
        valid_trials = ~beh["early"] & ~beh["stim_enable"]
        valid_idx = np.flatnonzero(valid_trials)
        if valid_idx.size < 2:
            raise ValueError(f"{session_tag} has fewer than two valid trials after filtering")

        condition_masks = get_condition_masks(beh)
        neural_all_trials, sites = load_neural_session(
            mat=mat,
            probe=spec["probe"],
            go_cue=beh["goCue"],
            condition_masks=condition_masks,
        )

        tongue_speed, tongue_visible = compute_composite_speed(mat, beh, TONGUE_FEATURES)
        paw_speed, paw_visible = compute_composite_speed(mat, beh, PAW_FEATURES)
        motion_energy, motion_visible = load_motion_energy_aligned(mat, motion_path, beh)
        lick_dir = first_post_go_lick_direction(mat, beh["lickL_refs"], beh["lickR_refs"], beh["goCue"])

    context = np.where(beh["autowater"], 0, 1).astype(np.int64)
    outcome = np.full((beh["ntrials"],), 2, dtype=np.int64)
    outcome[beh["miss"]] = 0
    outcome[beh["hit"]] = 1

    tongue_classes, tongue_threshold = discretize_with_visibility(
        tongue_speed, tongue_visible, valid_trials, absent_code=2
    )
    paw_classes, paw_threshold = discretize_with_visibility(
        paw_speed, paw_visible, valid_trials, absent_code=2
    )
    motion_classes, motion_threshold = discretize_with_visibility(
        motion_energy, motion_visible, valid_trials, absent_code=2
    )

    taxis = get_time_axis().astype(np.float32)
    input_template = taxis[None, :]
    neural_trials = []
    input_trials = []
    output_trials = []

    valid_neural = neural_all_trials[:, valid_idx, :]
    for kept_trial_pos, trial_idx in enumerate(valid_idx):
        neural_trials.append(valid_neural[:, kept_trial_pos, :].astype(np.float32))
        input_trials.append(input_template.copy())

        out = np.zeros((6, taxis.size), dtype=np.int64)
        out[0, :] = lick_dir[trial_idx]
        out[1, :] = context[trial_idx]
        out[2, :] = outcome[trial_idx]
        out[3, :] = tongue_classes[kept_trial_pos]
        out[4, :] = paw_classes[kept_trial_pos]
        out[5, :] = motion_classes[kept_trial_pos]
        output_trials.append(out)

    return {
        "subject": spec["subject"],
        "date": spec["date"],
        "probe": spec["probe"],
        "neural": neural_trials,
        "input": input_trials,
        "output": output_trials,
        "brain_region_idx": np.zeros((valid_neural.shape[0],), dtype=np.int64),
        "n_neurons": int(valid_neural.shape[0]),
        "n_trials": int(valid_idx.size),
        "n_trials_total": int(beh["ntrials"]),
        "n_trials_dropped_early_or_stim": int(beh["ntrials"] - valid_idx.size),
        "motion_threshold": motion_threshold,
        "tongue_threshold": tongue_threshold,
        "paw_threshold": paw_threshold,
        "bit_start_offset_s": float(np.median(beh["bitStart"] - beh["goCue"])),
    }


def build_dataset(data_root: Path) -> dict:
    sessions = [process_session(data_root, spec) for spec in SESSION_SPECS]

    subjects = []
    subject_to_idx = {}
    subject_idx = []
    for sess in sessions:
        if sess["subject"] not in subject_to_idx:
            subject_to_idx[sess["subject"]] = len(subjects)
            subjects.append(sess["subject"])
        subject_idx.append(subject_to_idx[sess["subject"]])

    data = {
        "neural": [sess["neural"] for sess in sessions],
        "input": [sess["input"] for sess in sessions],
        "output": [sess["output"] for sess in sessions],
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": ["ALM"],
        "brain_region_idx": [sess["brain_region_idx"] for sess in sessions],
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
            ["<50th_percentile", ">=50th_percentile", "not_visible"],
            ["<50th_percentile", ">=50th_percentile", "not_visible"],
            ["<50th_percentile", ">=50th_percentile", "no_video"],
        ],
        "metadata": {
            "task_description": (
                "Go-cue aligned ALM context-task sessions from Hasnain, Birnbaum et al. "
                "Neural firing rates predict lick direction, context, outcome, and discretized video/motion variables."
            ),
            "time_bin_size": float(DT * 1000.0),
            "temporal_alignment_event": "goCue onset",
            "off_start": float(TIME_MIN),
            "off_end": float(TIME_MAX),
            "neural_preprocessing": (
                "Spike times aligned to goCue, binned at 10 ms, converted to firing rates, "
                "and smoothed with the paper's causal Gaussian filter (window 15, reflect boundary)."
            ),
            "trial_filter": "Kept trials with ~early and ~stim.enable; retained hit, miss, and ignore trials.",
            "session_selection": (
                "Used the 12 hand-picked two-context ALM electrophysiology sessions from the paper's context scripts."
            ),
            "source_session_info": [
                {
                    "subject": sess["subject"],
                    "date": sess["date"],
                    "probe": sess["probe"],
                    "n_trials_exported": sess["n_trials"],
                    "n_trials_total": sess["n_trials_total"],
                    "n_trials_dropped_early_or_stim": sess["n_trials_dropped_early_or_stim"],
                    "n_neurons": sess["n_neurons"],
                    "bit_start_offset_s": sess["bit_start_offset_s"],
                    "tongue_threshold": sess["tongue_threshold"],
                    "paw_threshold": sess["paw_threshold"],
                    "motion_threshold": sess["motion_threshold"],
                }
                for sess in sessions
            ],
        },
    }
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/app/data", type=Path)
    parser.add_argument("--output", default="/app/converted_data.pkl", type=Path)
    args = parser.parse_args()

    data = build_dataset(args.data_root)
    with open(args.output, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    nsessions = len(data["neural"])
    ntrials = sum(len(sess) for sess in data["neural"])
    nneurons = [sess[0].shape[0] for sess in data["neural"]]
    print(f"Wrote {args.output}")
    print(f"Sessions: {nsessions}")
    print(f"Trials: {ntrials}")
    print(f"Neurons/session: {nneurons}")


if __name__ == "__main__":
    main()
