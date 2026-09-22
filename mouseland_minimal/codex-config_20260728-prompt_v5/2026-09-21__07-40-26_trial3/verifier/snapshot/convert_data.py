import argparse
import gc
import os
import pickle
import re
from collections import defaultdict
from datetime import datetime

import numpy as np


DATA_ROOT = "/app/data"
SPK_DIR = os.path.join(DATA_ROOT, "spk")
BEH_DIR = os.path.join(DATA_ROOT, "beh")
RETINO_DIR = os.path.join(DATA_ROOT, "retinotopy")
EXP_INFO_PATH = os.path.join(BEH_DIR, "Imaging_Exp_info.npy")

OUTPUT_PATH = "/app/converted_data.pkl"

TIME_BIN_MS = 1000.0 * 24.0 * 3600.0 * 0.31469352543354034
MAX_NEURONS_PER_REGION = 128
VISUAL_CORTEX_REGIONS = ["V1", "mHV", "lHV", "aHV"]

STIM_CATEGORY_VALUES = ["circle", "leaf", "rock", "wood"]
STIM_CATEGORY_TO_IDX = {name: idx for idx, name in enumerate(STIM_CATEGORY_VALUES)}

BASE_KEY_RE = re.compile(r"^(?P<mouse>.+?)_(?P<date>\d{4}_\d{2}_\d{2})_(?P<blk>\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Zhong et al. imaging data for decoder training.")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Output pickle path.")
    parser.add_argument(
        "--max-neurons-per-region",
        type=int,
        default=MAX_NEURONS_PER_REGION,
        help="Deterministic cap per visual region after curation.",
    )
    parser.add_argument(
        "--limit-sessions",
        type=int,
        default=None,
        help="Optional debug limit on the number of sessions to convert.",
    )
    return parser.parse_args()


def full_key_from_record(record):
    base = f"{record['mname']}_{record['datexp']}_{record['blk']}"
    if "stimtype" in record:
        return f"{base}_{record['stimtype']}"
    return base


def base_key_from_full_key(full_key):
    parts = full_key.split("_")
    if len(parts) >= 5 and parts[-1].startswith("swap"):
        return "_".join(parts[:-1])
    return full_key


def parse_base_key(base_key):
    match = BASE_KEY_RE.match(base_key)
    if match is None:
        raise ValueError(f"Could not parse base key: {base_key}")
    return match.group("mouse"), match.group("date"), match.group("blk")


def date_from_base_key(base_key):
    _, date_str, _ = parse_base_key(base_key)
    return datetime.strptime(date_str, "%Y_%m_%d").date()


def canonical_stimulus_category(wall_name):
    category = re.sub(r"_swap[12]$", "", str(wall_name))
    category = re.sub(r"\d+$", "", category)
    if category not in STIM_CATEGORY_TO_IDX:
        raise ValueError(f"Unexpected wall name/category: {wall_name} -> {category}")
    return category


def array_equal_safe(a, b):
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.dtype.kind in {"U", "S", "O"} or b_arr.dtype.kind in {"U", "S", "O"}:
        return np.array_equal(a_arr, b_arr)
    return np.array_equal(a_arr, b_arr, equal_nan=True)


def behavior_sessions_match(a, b):
    fields = [
        "ntrials",
        "UniqWalls",
        "WallName",
        "isRew",
        "SoundPos",
        "SoundDelPos",
        "RewPos",
        "LickPos",
        "LickTrind",
        "ft_trInd",
        "ft_Pos",
        "ft_move",
        "ft_CorrSpc",
        "ft_GraySpc",
        "ft_RunSpeed",
    ]
    for field in fields:
        if not array_equal_safe(a[field], b[field]):
            return False, field
    return True, None


def load_canonical_behavior_sessions():
    exp_info = np.load(EXP_INFO_PATH, allow_pickle=True).item()
    canonical = {}
    behavior_files = {}

    for exp_type, records in exp_info.items():
        behavior_path = os.path.join(BEH_DIR, f"Beh_{exp_type}.npy")
        beh = np.load(behavior_path, allow_pickle=True).item()
        behavior_files[exp_type] = beh

        for record in records:
            full_key = full_key_from_record(record)
            base_key = f"{record['mname']}_{record['datexp']}_{record['blk']}"
            session = beh[full_key]

            if base_key not in canonical:
                canonical[base_key] = {
                    "behavior": session,
                    "exp_type": exp_type,
                    "full_key": full_key,
                    "record": dict(record),
                }
                continue

            same, field = behavior_sessions_match(canonical[base_key]["behavior"], session)
            if not same:
                raise ValueError(
                    f"Behavior mismatch for duplicated recording {base_key} between "
                    f"{canonical[base_key]['full_key']} and {full_key} at field {field}"
                )
    return canonical


def get_spk_session_keys():
    session_keys = []
    for filename in sorted(os.listdir(SPK_DIR)):
        if not filename.endswith("_neural_data.npy"):
            continue
        session_keys.append(filename[: -len("_neural_data.npy")])
    return session_keys


def compute_training_day_by_session(session_keys):
    by_mouse = defaultdict(list)
    for session_key in session_keys:
        mouse, _, _ = parse_base_key(session_key)
        by_mouse[mouse].append(session_key)

    first_day = {}
    for mouse, keys in by_mouse.items():
        first_day[mouse] = min(date_from_base_key(key) for key in keys)

    training_day = {}
    for session_key in session_keys:
        mouse, _, _ = parse_base_key(session_key)
        training_day[session_key] = float((date_from_base_key(session_key) - first_day[mouse]).days)
    return training_day


def build_region_index(iarea):
    region_idx = np.full(len(iarea), -1, dtype=np.int16)
    region_idx[iarea == 8] = 0
    region_idx[np.isin(iarea, [0, 1, 2, 9])] = 1
    region_idx[np.isin(iarea, [5, 6])] = 2
    region_idx[np.isin(iarea, [3, 4])] = 3
    return region_idx


def sort_take_desc(scores, indices, n_take):
    if len(indices) <= n_take:
        return indices
    order = np.argsort(scores, kind="mergesort")[::-1]
    return indices[order[:n_take]]


def select_visual_neurons(spk_planes, region_idx_full, corridor_mask, gray_mask, max_per_region):
    responsive_by_region = defaultdict(list)
    fallback_by_region = defaultdict(list)

    offset = 0
    for plane_idx, plane in enumerate(spk_planes):
        n_neurons = plane.shape[0]
        plane_regions = region_idx_full[offset : offset + n_neurons]
        valid_local = np.flatnonzero(plane_regions >= 0)

        if len(valid_local):
            plane_valid = plane[valid_local]
            corridor_data = plane_valid[:, corridor_mask]
            corridor_mean = corridor_data.mean(axis=1)
            corridor_var = corridor_data.var(axis=1)
            del corridor_data

            if gray_mask.any():
                gray_mean = plane_valid[:, gray_mask].mean(axis=1)
                responsive = corridor_mean > gray_mean
            else:
                responsive = np.ones_like(corridor_mean, dtype=bool)

            responsive &= np.isfinite(corridor_var)

            for region in range(len(VISUAL_CORTEX_REGIONS)):
                region_mask = plane_regions[valid_local] == region
                if not np.any(region_mask):
                    continue

                region_local = valid_local[region_mask]
                region_scores = corridor_var[region_mask]
                region_responsive = responsive[region_mask]

                global_rows = region_local + offset
                responsive_by_region[region].append(
                    (global_rows[region_responsive], region_scores[region_responsive])
                )
                fallback_by_region[region].append((global_rows, region_scores))

        offset += n_neurons

    chosen = []
    for region in range(len(VISUAL_CORTEX_REGIONS)):
        resp_indices = np.concatenate(
            [chunk[0] for chunk in responsive_by_region[region]] or [np.empty(0, dtype=np.int64)]
        )
        resp_scores = np.concatenate(
            [chunk[1] for chunk in responsive_by_region[region]] or [np.empty(0, dtype=np.float32)]
        )
        region_selected = sort_take_desc(resp_scores, resp_indices, max_per_region)

        if len(region_selected) < max_per_region:
            all_indices = np.concatenate(
                [chunk[0] for chunk in fallback_by_region[region]] or [np.empty(0, dtype=np.int64)]
            )
            all_scores = np.concatenate(
                [chunk[1] for chunk in fallback_by_region[region]] or [np.empty(0, dtype=np.float32)]
            )
            if len(all_indices):
                already = set(region_selected.tolist())
                fill_mask = np.array([idx not in already for idx in all_indices], dtype=bool)
                fill_indices = all_indices[fill_mask]
                fill_scores = all_scores[fill_mask]
                n_fill = max_per_region - len(region_selected)
                fill_selected = sort_take_desc(fill_scores, fill_indices, n_fill)
                region_selected = np.concatenate([region_selected, fill_selected])

        chosen.append(region_selected)

    if not chosen:
        return np.empty(0, dtype=np.int64)

    selected_global = np.concatenate(chosen)
    selected_global = np.unique(selected_global)
    selected_global.sort()
    return selected_global.astype(np.int64)


def split_global_indices_by_plane(global_indices, plane_sizes):
    by_plane = []
    start = 0
    for size in plane_sizes:
        end = start + size
        mask = (global_indices >= start) & (global_indices < end)
        by_plane.append((global_indices[mask] - start).astype(np.int64))
        start = end
    return by_plane


def nearest_retained_licks(retained_frame_idx, lick_frame_idx, lick_pos, texture_length):
    lick_binary = np.zeros(len(retained_frame_idx), dtype=np.int8)
    if len(retained_frame_idx) == 0:
        return lick_binary

    lick_frame_idx = np.asarray(lick_frame_idx, dtype=np.float64)
    lick_pos = np.asarray(lick_pos, dtype=np.float64)
    valid = np.isfinite(lick_frame_idx) & np.isfinite(lick_pos) & (lick_pos >= 0.0) & (lick_pos < texture_length)
    lick_frame_idx = lick_frame_idx[valid]
    if len(lick_frame_idx) == 0:
        return lick_binary

    retained = retained_frame_idx.astype(np.float64)
    insert = np.searchsorted(retained, lick_frame_idx)
    candidate_left = np.clip(insert - 1, 0, len(retained) - 1)
    candidate_right = np.clip(insert, 0, len(retained) - 1)

    left_dist = np.abs(retained[candidate_left] - lick_frame_idx)
    right_dist = np.abs(retained[candidate_right] - lick_frame_idx)
    use_right = right_dist < left_dist
    nearest = candidate_left.copy()
    nearest[use_right] = candidate_right[use_right]
    nearest_dist = np.minimum(left_dist, right_dist)

    lick_binary[np.unique(nearest[nearest_dist <= 0.5])] = 1
    return lick_binary


def make_position_bins(position, texture_length):
    pos = np.clip(np.asarray(position, dtype=np.float32), 0.0, float(texture_length) - 1e-6)
    bin_edges = np.linspace(0.0, float(texture_length), 5)
    return np.clip(np.digitize(pos, bin_edges[1:-1], right=False), 0, 3).astype(np.int8)


def make_speed_bins(speed, speed_edges):
    return np.clip(np.digitize(np.asarray(speed, dtype=np.float32), speed_edges, right=False), 0, 3).astype(np.int8)


def compute_speed_edges(session_keys, canonical_behavior):
    all_speeds = []
    for session_key in session_keys:
        sess = canonical_behavior[session_key]["behavior"]
        mask = sess["ft_CorrSpc"] & (sess["ft_move"] > 0)
        if np.any(mask):
            all_speeds.append(np.asarray(sess["ft_RunSpeed"][mask], dtype=np.float32))
    speed_values = np.concatenate(all_speeds)
    q25, q50, q75 = np.quantile(speed_values, [0.25, 0.5, 0.75])
    return np.array([q25, q50, q75], dtype=np.float32)


def build_subject_metadata(session_keys):
    subjects = sorted({parse_base_key(key)[0] for key in session_keys})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
    subject_idx = np.array([subject_to_idx[parse_base_key(key)[0]] for key in session_keys], dtype=np.int64)
    return subjects, subject_idx


def load_retinotopy_region_idx(session_key):
    mouse, date_str, _ = parse_base_key(session_key)
    retino_path = os.path.join(RETINO_DIR, f"{mouse}_{date_str}_trans.npz")
    retino = np.load(retino_path, allow_pickle=True)
    return build_region_index(retino["iarea"])


def build_selected_session_matrix(spk_planes, selected_global):
    plane_sizes = [plane.shape[0] for plane in spk_planes]
    per_plane = split_global_indices_by_plane(selected_global, plane_sizes)
    selected_parts = []
    for plane, local_idx in zip(spk_planes, per_plane):
        if len(local_idx):
            selected_parts.append(plane[local_idx])
    if not selected_parts:
        raise ValueError("No neurons selected for session.")
    return np.concatenate(selected_parts, axis=0)


def convert_session(
    session_key,
    canonical_behavior_entry,
    day_of_training,
    speed_edges,
    max_neurons_per_region,
):
    behavior = canonical_behavior_entry["behavior"]
    spk_path = os.path.join(SPK_DIR, f"{session_key}_neural_data.npy")
    spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]

    n_frames = spk_planes[0].shape[1]
    region_idx_full = load_retinotopy_region_idx(session_key)
    if len(region_idx_full) != sum(plane.shape[0] for plane in spk_planes):
        raise ValueError(f"Retinotopy length mismatch for {session_key}")

    ft_trind = np.asarray(behavior["ft_trInd"][:n_frames])
    ft_pos = np.asarray(behavior["ft_Pos"][:n_frames], dtype=np.float32)
    ft_move = np.asarray(behavior["ft_move"][:n_frames], dtype=np.float32)
    ft_corr = np.asarray(behavior["ft_CorrSpc"][:n_frames], dtype=bool)
    ft_gray = np.asarray(behavior["ft_GraySpc"][:n_frames], dtype=bool)
    ft_speed = np.asarray(behavior["ft_RunSpeed"][:n_frames], dtype=np.float32)
    ft_time = np.asarray(behavior["ft"][:n_frames], dtype=np.float64)

    retained_mask = ft_corr & (ft_move > 0)
    gray_running_mask = ft_gray & (ft_move > 0)

    selected_global = select_visual_neurons(
        spk_planes=spk_planes,
        region_idx_full=region_idx_full,
        corridor_mask=retained_mask,
        gray_mask=gray_running_mask,
        max_per_region=max_neurons_per_region,
    )
    if len(selected_global) == 0:
        raise ValueError(f"No neurons selected for session {session_key}")

    selected_brain_regions = region_idx_full[selected_global].astype(np.int64)
    session_matrix = build_selected_session_matrix(spk_planes, selected_global)
    del spk_planes
    gc.collect()

    texture_length = float(behavior.get("Texture_Length", 40.0))
    n_trials = int(behavior["ntrials"])

    neural_trials = []
    input_trials = []
    output_trials = []

    trial_start_time = np.asarray(behavior["Trial_start_time"], dtype=np.float64)
    sound_time = np.asarray(behavior["SoundTime"], dtype=np.float64)
    lick_trial_index = np.asarray(behavior["LickTrind"], dtype=np.float64)
    lick_frame_idx = np.asarray(behavior["LickFr"], dtype=np.float64)
    lick_pos = np.asarray(behavior["LickPos"], dtype=np.float64)
    wall_names = np.asarray(behavior["WallName"])
    is_rew = np.asarray(behavior["isRew"], dtype=bool)

    for trial in range(n_trials):
        frame_idx = np.flatnonzero((ft_trind == trial) & retained_mask)
        if len(frame_idx) == 0:
            continue

        neural = session_matrix[:, frame_idx].astype(np.float16, copy=False)

        times_since_start = ((ft_time[frame_idx] - trial_start_time[trial]) * 24.0 * 3600.0).astype(np.float32)
        times_to_sound = ((sound_time[trial] - ft_time[frame_idx]) * 24.0 * 3600.0).astype(np.float32)
        reward_available = np.full(len(frame_idx), float(is_rew[trial]), dtype=np.float32)
        training_day = np.full(len(frame_idx), day_of_training, dtype=np.float32)

        input_arr = np.vstack(
            [
                times_to_sound,
                training_day,
                times_since_start,
                reward_available,
            ]
        ).astype(np.float32, copy=False)

        trial_lick_mask = lick_trial_index == trial
        licking = nearest_retained_licks(
            retained_frame_idx=frame_idx,
            lick_frame_idx=lick_frame_idx[trial_lick_mask],
            lick_pos=lick_pos[trial_lick_mask],
            texture_length=texture_length,
        )

        stim_category = STIM_CATEGORY_TO_IDX[canonical_stimulus_category(wall_names[trial])]
        stim_out = np.full(len(frame_idx), stim_category, dtype=np.int8)
        position_out = make_position_bins(ft_pos[frame_idx], texture_length)
        speed_out = make_speed_bins(ft_speed[frame_idx], speed_edges)

        output_arr = np.vstack(
            [
                stim_out,
                licking,
                position_out,
                speed_out,
            ]
        ).astype(np.int8, copy=False)

        neural_trials.append(neural)
        input_trials.append(input_arr)
        output_trials.append(output_arr)

    del session_matrix
    gc.collect()

    if len(neural_trials) < 2:
        return None

    return {
        "neural": neural_trials,
        "input": input_trials,
        "output": output_trials,
        "brain_region_idx": selected_brain_regions,
        "session_info": {
            "session_key": session_key,
            "behavior_group": canonical_behavior_entry["exp_type"],
            "behavior_key": canonical_behavior_entry["full_key"],
            "day_of_training": day_of_training,
            "n_trials_retained": len(neural_trials),
            "n_neurons_retained": int(len(selected_brain_regions)),
        },
    }


def convert_dataset(output_path, max_neurons_per_region, limit_sessions=None):
    canonical_behavior = load_canonical_behavior_sessions()
    session_keys = get_spk_session_keys()
    if limit_sessions is not None:
        session_keys = session_keys[:limit_sessions]

    missing_behavior = [key for key in session_keys if key not in canonical_behavior]
    if missing_behavior:
        raise ValueError(f"Missing behavior for sessions: {missing_behavior[:5]}")

    day_by_session = compute_training_day_by_session(session_keys)
    speed_edges = compute_speed_edges(session_keys, canonical_behavior)
    subjects, subject_idx_full = build_subject_metadata(session_keys)

    converted = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": subjects,
        "subject_idx": [],
        "brain_regions": VISUAL_CORTEX_REGIONS,
        "brain_region_idx": [],
        "input_names": [
            "time_to_sound_cue_s",
            "day_of_training",
            "time_since_trial_start_s",
            "reward_available",
        ],
        "output_names": [
            "visual_stimulus_category",
            "licking",
            "position_bin",
            "running_speed_bin",
        ],
        "output_values": [
            STIM_CATEGORY_VALUES,
            ["no_lick", "lick"],
            ["0-1m", "1-2m", "2-3m", "3-4m"],
            ["speed_q1", "speed_q2", "speed_q3", "speed_q4"],
        ],
        "metadata": {
            "task_description": (
                "Head-fixed mice ran through 4 m virtual-reality corridors with natural texture stimuli; "
                "a sound cue occurred at a random corridor position and rewards were available only on rewarded supervised trials."
            ),
            "time_bin_size": float(TIME_BIN_MS),
            "temporal_alignment_event": "Trial start / corridor entry, with paper-matched retention of running frames inside the corridor only.",
            "off_start": 0.0,
            "off_end": None,
            "neural_signal": "Suite2p deconvolved fluorescence traces",
            "frame_selection": "ft_CorrSpc & (ft_move > 0)",
            "neuron_selection": (
                "Restricted to retinotopically assigned visual cortex neurons (V1, mHV, lHV, aHV). "
                f"Within each region, retained up to {max_neurons_per_region} deterministic high-variance corridor-responsive neurons."
            ),
            "stimulus_category_mapping": {
                "circle1/circle2/circle3": "circle",
                "leaf1/leaf2/leaf3/leaf1_swap1/leaf1_swap2": "leaf",
                "rock1/rock2": "rock",
                "wood1/wood2/wood5/wood1_swap1/wood1_swap2": "wood",
            },
            "running_speed_bin_edges_cm_s": speed_edges.astype(float).tolist(),
            "day_of_training_definition": "Calendar days since the first imaging session for that mouse.",
            "source_sessions": session_keys,
            "source_behavior_keys": {
                key: canonical_behavior[key]["full_key"] for key in session_keys
            },
        },
    }

    kept_subject_idx = []
    session_metadata = []

    session_to_subject = {key: int(subject_idx_full[i]) for i, key in enumerate(session_keys)}

    for i, session_key in enumerate(session_keys, start=1):
        print(f"[{i}/{len(session_keys)}] Converting {session_key}", flush=True)
        converted_session = convert_session(
            session_key=session_key,
            canonical_behavior_entry=canonical_behavior[session_key],
            day_of_training=day_by_session[session_key],
            speed_edges=speed_edges,
            max_neurons_per_region=max_neurons_per_region,
        )
        if converted_session is None:
            print(f"  Skipping {session_key}: fewer than 2 valid trials after filtering", flush=True)
            continue

        converted["neural"].append(converted_session["neural"])
        converted["input"].append(converted_session["input"])
        converted["output"].append(converted_session["output"])
        converted["brain_region_idx"].append(converted_session["brain_region_idx"])
        kept_subject_idx.append(session_to_subject[session_key])
        session_metadata.append(converted_session["session_info"])

    converted["subject_idx"] = np.asarray(kept_subject_idx, dtype=np.int64)
    converted["metadata"]["session_info"] = session_metadata

    with open(output_path, "wb") as handle:
        pickle.dump(converted, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"Saved {len(converted['neural'])} sessions and "
        f"{sum(len(x) for x in converted['neural'])} trials to {output_path}",
        flush=True,
    )


def main():
    args = parse_args()
    convert_dataset(
        output_path=args.output,
        max_neurons_per_region=args.max_neurons_per_region,
        limit_sessions=args.limit_sessions,
    )


if __name__ == "__main__":
    main()
