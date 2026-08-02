import argparse
import copy
import gc
import math
import os
import pickle
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


DATA_ROOT = Path("/app/data")
BEH_ROOT = DATA_ROOT / "beh"
SPK_ROOT = DATA_ROOT / "spk"
RET_ROOT = DATA_ROOT / "retinotopy"

VISUAL_AREA_CODES = {
    "mHV": {0, 1, 2, 9},
    "aHV": {3, 4},
    "lHV": {5, 6},
    "V1": {8},
}
BRAIN_REGIONS = ["V1", "mHV", "lHV", "aHV"]
POSITION_BIN_EDGES = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
POSITION_BIN_CENTERS = np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float32)
SECONDS_PER_DAY = 24.0 * 3600.0
DEFAULT_MAX_NEURONS = 512
DEFAULT_SAMPLE_SESSIONS = 8


def natural_key(text: str):
    parts = re.split(r"(\d+)", text)
    return [int(p) if p.isdigit() else p for p in parts]


def strip_swap_suffix(session_key: str) -> str:
    return re.sub(r"_swap[12]$", "", session_key)


def parse_session_key(session_key: str):
    subject, year, month, day, block = session_key.split("_")
    date_str = f"{year}_{month}_{day}"
    date_obj = datetime.strptime(date_str, "%Y_%m_%d").date()
    return {
        "subject": subject,
        "date": date_obj,
        "date_str": date_str,
        "block": block,
    }


def load_behavior_views():
    views = defaultdict(list)
    for beh_path in sorted(BEH_ROOT.glob("Beh_*.npy")):
        beh_dict = np.load(beh_path, allow_pickle=True).item()
        for view_key, view in beh_dict.items():
            base_key = strip_swap_suffix(view_key)
            views[base_key].append(
                {
                    "source_file": beh_path.name,
                    "view_key": view_key,
                    "data": view,
                }
            )
    return views


def _arrays_match(a, b):
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.dtype.kind in {"U", "S", "O"} or b_arr.dtype.kind in {"U", "S", "O"}:
        return a_arr.shape == b_arr.shape and np.array_equal(a_arr, b_arr)
    return a_arr.shape == b_arr.shape and np.allclose(a_arr, b_arr, equal_nan=True)


def merge_behavior_views(base_key: str, session_views: list[dict]):
    if not session_views:
        raise KeyError(f"No behavior view found for session {base_key}")

    sort_key = lambda item: (
        len(np.asarray(item["data"]["UniqWalls"])),
        np.sum(~np.isnan(np.asarray(item["data"].get("stim_id", []), dtype=float))),
        item["source_file"],
        item["view_key"],
    )
    canonical = max(session_views, key=sort_key)
    template = canonical["data"]

    check_fields = [
        "ntrials",
        "WallName",
        "Trial_start_time",
        "Trial_end_time",
        "SoundPos",
        "SoundTime",
        "SoundTimeDelay",
        "RewPos",
        "RewTime",
        "isRew",
        "StartFr",
        "EndFr",
        "SoundFr",
        "LickTrind",
        "LickPos",
        "LickTime",
        "ft",
        "ft_trInd",
        "ft_Pos",
        "ft_PosCum",
        "ft_move",
        "ft_CorrSpc",
        "ft_RunSpeed",
    ]

    for other in session_views:
        other_data = other["data"]
        for field in check_fields:
            if field not in template or field not in other_data:
                continue
            if not _arrays_match(template[field], other_data[field]):
                raise ValueError(
                    f"Behavior views disagree for session {base_key} on field {field}: "
                    f"{canonical['source_file']}:{canonical['view_key']} vs "
                    f"{other['source_file']}:{other['view_key']}"
                )

    merged = dict(template)
    merged["source_behavior_views"] = [
        f"{item['source_file']}::{item['view_key']}" for item in sorted(
            session_views,
            key=lambda item: (item["source_file"], item["view_key"]),
        )
    ]
    return merged


def compute_subject_day_index(session_keys: list[str]):
    parsed = {key: parse_session_key(key) for key in session_keys}
    first_day = {}
    for key, info in parsed.items():
        subj = info["subject"]
        if subj not in first_day or info["date"] < first_day[subj]:
            first_day[subj] = info["date"]

    day_index = {}
    for key, info in parsed.items():
        day_index[key] = float((info["date"] - first_day[info["subject"]]).days)
    return day_index, parsed


def area_code_to_region_name(area_code: float):
    if np.isnan(area_code):
        return None
    area_code = int(area_code)
    for region_name, codes in VISUAL_AREA_CODES.items():
        if area_code in codes:
            return region_name
    return None


def pick_neurons_by_variance(spk, area_codes, running_corridor_mask, max_neurons):
    region_to_indices = {region: [] for region in BRAIN_REGIONS}
    for neuron_idx, area_code in enumerate(area_codes):
        region_name = area_code_to_region_name(area_code)
        if region_name is not None:
            region_to_indices[region_name].append(neuron_idx)

    valid_indices = np.array(
        sorted(i for indices in region_to_indices.values() for i in indices),
        dtype=np.int64,
    )
    if len(valid_indices) == 0:
        raise ValueError("No neurons mapped to visual cortex")

    if len(valid_indices) <= max_neurons:
        region_idx = np.array(
            [BRAIN_REGIONS.index(area_code_to_region_name(area_codes[i])) for i in valid_indices],
            dtype=np.int16,
        )
        return valid_indices, region_idx

    if running_corridor_mask.sum() == 0:
        raise ValueError("No running corridor frames available for variance ranking")

    spk_valid = spk[valid_indices][:, running_corridor_mask]
    variances = np.var(spk_valid, axis=1)
    variance_map = dict(zip(valid_indices.tolist(), variances.tolist()))

    per_region_quota = max_neurons // len(BRAIN_REGIONS)
    selected = []
    for region_name in BRAIN_REGIONS:
        candidates = np.array(region_to_indices[region_name], dtype=np.int64)
        if len(candidates) == 0:
            continue
        order = sorted(
            candidates.tolist(),
            key=lambda idx: (-variance_map[idx], idx),
        )
        selected.extend(order[: min(per_region_quota, len(order))])

    if len(selected) < max_neurons:
        already_selected = set(selected)
        remainder = []
        for region_name in BRAIN_REGIONS:
            remainder.extend(idx for idx in region_to_indices[region_name] if idx not in already_selected)
        remainder = sorted(remainder, key=lambda idx: (-variance_map[idx], idx))
        selected.extend(remainder[: max_neurons - len(selected)])

    selected = np.array(sorted(selected), dtype=np.int64)
    region_idx = np.array(
        [BRAIN_REGIONS.index(area_code_to_region_name(area_codes[i])) for i in selected],
        dtype=np.int16,
    )
    return selected, region_idx


def interpolate_features(positions, values, target_positions):
    positions = np.asarray(positions, dtype=np.float32)
    order = np.argsort(positions)
    positions = positions[order]
    if values.ndim == 1:
        values = np.asarray(values, dtype=np.float32)[order][np.newaxis, :]
    else:
        values = np.asarray(values, dtype=np.float32)[:, order]

    unique_pos, inverse = np.unique(positions, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float32)

    if len(unique_pos) == 1:
        means = values.mean(axis=1, keepdims=True)
        return np.repeat(means, len(target_positions), axis=1)

    collapsed = np.empty((values.shape[0], len(unique_pos)), dtype=np.float32)
    for row in range(values.shape[0]):
        collapsed[row] = np.bincount(inverse, weights=values[row], minlength=len(unique_pos)) / counts

    interp = np.empty((collapsed.shape[0], len(target_positions)), dtype=np.float32)
    for row in range(collapsed.shape[0]):
        interp[row] = np.interp(target_positions, unique_pos, collapsed[row]).astype(np.float32)
    return interp


def trial_frame_indices(beh, n_frames, trial_idx):
    ft_trial_idx_raw = np.asarray(beh["ft_trInd"][:n_frames], dtype=np.float64)
    valid = ~np.isnan(ft_trial_idx_raw)
    ft_trial_idx = np.full(n_frames, -1, dtype=np.int64)
    ft_trial_idx[valid] = np.rint(ft_trial_idx_raw[valid]).astype(np.int64)
    move = np.asarray(beh["ft_move"][:n_frames]) > 0
    in_corridor = np.asarray(beh["ft_CorrSpc"][:n_frames]).astype(bool)
    return np.where((ft_trial_idx == trial_idx) & move & in_corridor)[0]


def stimulus_catalog(behavior_by_session):
    names = set()
    for beh in behavior_by_session.values():
        names.update(map(str, np.asarray(beh["WallName"]).tolist()))
    return sorted(names, key=natural_key)


def build_single_session(
    session_key,
    beh,
    subject_day_value,
    stimulus_to_id,
    max_neurons,
):
    spk_path = SPK_ROOT / f"{session_key}_neural_data.npy"
    ret_path = RET_ROOT / f"{session_key.rsplit('_', 1)[0]}_trans.npz"

    spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
    spk = np.concatenate([plane.astype(np.float32, copy=False) for plane in spk_planes], axis=0)
    ret = np.load(ret_path, allow_pickle=True)
    area_codes = ret["iarea"]

    n_frames = spk.shape[1]
    running_corridor_mask = (np.asarray(beh["ft_move"][:n_frames]) > 0) & np.asarray(beh["ft_CorrSpc"][:n_frames]).astype(bool)
    selected_neurons, region_idx = pick_neurons_by_variance(
        spk=spk,
        area_codes=area_codes,
        running_corridor_mask=running_corridor_mask,
        max_neurons=max_neurons,
    )
    spk_selected = spk[selected_neurons]

    ft = np.asarray(beh["ft"][:n_frames], dtype=np.float64)
    ft_pos = np.asarray(beh["ft_Pos"][:n_frames], dtype=np.float32)
    ft_speed = np.asarray(beh["ft_RunSpeed"][:n_frames], dtype=np.float32)
    lick_trials = np.rint(np.asarray(beh["LickTrind"])).astype(np.int64)
    lick_positions = np.asarray(beh["LickPos"], dtype=np.float32)

    session_neural = []
    session_input = []
    session_output_partial = []
    speed_values = []
    trial_stats = {
        "ntrials_raw": int(beh["ntrials"]),
        "ntrials_kept": 0,
        "empty_running_trials": 0,
    }

    for trial_idx in range(int(beh["ntrials"])):
        frame_idx = trial_frame_indices(beh, n_frames, trial_idx)
        if len(frame_idx) == 0:
            trial_stats["empty_running_trials"] += 1
            continue

        trial_positions = ft_pos[frame_idx]
        if np.nanmax(trial_positions) < POSITION_BIN_CENTERS[-1]:
            trial_stats["empty_running_trials"] += 1
            continue

        neural_interp = interpolate_features(
            positions=trial_positions,
            values=spk_selected[:, frame_idx],
            target_positions=POSITION_BIN_CENTERS,
        ).astype(np.float16)

        trial_start_time = float(np.asarray(beh["Trial_start_time"])[trial_idx])
        sound_time = float(np.asarray(beh["SoundTime"])[trial_idx])
        time_since_start = ((ft[frame_idx] - trial_start_time) * SECONDS_PER_DAY).astype(np.float32)
        time_to_sound = ((sound_time - ft[frame_idx]) * SECONDS_PER_DAY).astype(np.float32)
        speed_interp = interpolate_features(
            positions=trial_positions,
            values=ft_speed[frame_idx],
            target_positions=POSITION_BIN_CENTERS,
        )[0].astype(np.float32)

        input_trial = np.vstack(
            [
                interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0],
                np.full(4, subject_day_value, dtype=np.float32),
                interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0],
                np.full(4, float(np.asarray(beh["isRew"])[trial_idx]), dtype=np.float32),
            ]
        ).astype(np.float32)

        lick_pos_trial = lick_positions[lick_trials == trial_idx]
        lick_trial = np.array(
            [
                int(np.any((lick_pos_trial >= POSITION_BIN_EDGES[bin_idx]) & (lick_pos_trial < POSITION_BIN_EDGES[bin_idx + 1])))
                for bin_idx in range(4)
            ],
            dtype=np.int16,
        )
        if len(lick_pos_trial) and np.any(lick_pos_trial >= POSITION_BIN_EDGES[-1]):
            lick_trial[-1] = 1

        stim_name = str(np.asarray(beh["WallName"])[trial_idx])
        output_trial_partial = {
            "visual_stimulus_category": np.full(4, stimulus_to_id[stim_name], dtype=np.int16),
            "licking": lick_trial,
            "position_bin": np.arange(4, dtype=np.int16),
            "running_speed_continuous": speed_interp,
        }

        session_neural.append(neural_interp)
        session_input.append(input_trial)
        session_output_partial.append(output_trial_partial)
        speed_values.append(speed_interp)

    trial_stats["ntrials_kept"] = len(session_neural)
    if len(session_neural) < 2:
        raise ValueError(f"Session {session_key} has fewer than 2 usable trials after running-frame filtering")

    raw_counts = {
        "n_neurons_raw": int(len(area_codes)),
        "n_neurons_visual_cortex": int(sum(area_code_to_region_name(code) is not None for code in area_codes)),
        "n_neurons_selected": int(len(selected_neurons)),
    }

    del spk
    del spk_selected
    del spk_planes
    gc.collect()

    return {
        "neural": session_neural,
        "input": session_input,
        "output_partial": session_output_partial,
        "brain_region_idx": region_idx,
        "speed_values": np.concatenate(speed_values).astype(np.float32),
        "trial_stats": trial_stats,
        "raw_counts": raw_counts,
    }


def finalize_outputs(session_output_partial, speed_edges):
    outputs = []
    for trial in session_output_partial:
        speed_bins = np.digitize(trial["running_speed_continuous"], speed_edges[1:-1], right=False).astype(np.int16)
        outputs.append(
            np.vstack(
                [
                    trial["visual_stimulus_category"],
                    trial["licking"],
                    trial["position_bin"],
                    speed_bins,
                ]
            ).astype(np.int16)
        )
    return outputs


def make_sample_dataset(full_data, sample_session_count):
    nsessions = len(full_data["neural"])
    if sample_session_count >= nsessions:
        return copy.deepcopy(full_data)

    sample_indices = np.linspace(0, nsessions - 1, sample_session_count, dtype=int)
    sample_indices = np.unique(sample_indices)
    subjects_old = full_data["subjects"]
    subject_idx_old = full_data["subject_idx"][sample_indices]
    kept_subjects = sorted(set(subject_idx_old.tolist()))
    remap = {old: new for new, old in enumerate(kept_subjects)}

    sample = {
        "neural": [full_data["neural"][i] for i in sample_indices],
        "input": [full_data["input"][i] for i in sample_indices],
        "output": [full_data["output"][i] for i in sample_indices],
        "subjects": [subjects_old[i] for i in kept_subjects],
        "subject_idx": np.array([remap[i] for i in subject_idx_old], dtype=np.int16),
        "brain_regions": list(full_data["brain_regions"]),
        "brain_region_idx": [full_data["brain_region_idx"][i] for i in sample_indices],
        "input_names": list(full_data["input_names"]),
        "output_names": list(full_data["output_names"]),
        "output_values": copy.deepcopy(full_data["output_values"]),
        "metadata": copy.deepcopy(full_data["metadata"]),
    }
    sample["metadata"]["sample_session_indices"] = sample_indices.tolist()
    sample["metadata"]["sample_session_keys"] = [full_data["metadata"]["session_keys"][i] for i in sample_indices]
    sample["metadata"]["is_sample"] = True
    return sample


def convert_dataset(max_neurons_per_session):
    spk_session_keys = sorted(
        [path.name.replace("_neural_data.npy", "") for path in SPK_ROOT.glob("*_neural_data.npy")],
        key=natural_key,
    )

    behavior_views = load_behavior_views()
    missing_behavior = [key for key in spk_session_keys if key not in behavior_views]
    if missing_behavior:
        raise KeyError(f"Missing behavior entries for sessions: {missing_behavior}")

    behavior_by_session = {
        key: merge_behavior_views(key, behavior_views[key]) for key in spk_session_keys
    }
    stimulus_values = stimulus_catalog(behavior_by_session)
    stimulus_to_id = {name: idx for idx, name in enumerate(stimulus_values)}
    day_index, parsed = compute_subject_day_index(spk_session_keys)

    subjects = sorted({info["subject"] for info in parsed.values()})
    subject_to_id = {subject: idx for idx, subject in enumerate(subjects)}

    full_neural = []
    full_input = []
    output_partials = []
    brain_region_idx = []
    subject_idx = []

    raw_neuron_counts = []
    kept_neuron_counts = []
    raw_trial_counts = []
    kept_trial_counts = []
    empty_trial_counts = []
    session_source_views = []
    all_speed_values = []

    frame_periods = []
    for key in spk_session_keys:
        ft = np.asarray(behavior_by_session[key]["ft"], dtype=np.float64)
        if len(ft) > 1:
            frame_periods.append(float(np.median(np.diff(ft) * SECONDS_PER_DAY)))

    for session_idx, session_key in enumerate(spk_session_keys):
        print(f"[{session_idx + 1}/{len(spk_session_keys)}] processing {session_key}")
        beh = behavior_by_session[session_key]
        session_data = build_single_session(
            session_key=session_key,
            beh=beh,
            subject_day_value=day_index[session_key],
            stimulus_to_id=stimulus_to_id,
            max_neurons=max_neurons_per_session,
        )

        full_neural.append(session_data["neural"])
        full_input.append(session_data["input"])
        output_partials.append(session_data["output_partial"])
        brain_region_idx.append(session_data["brain_region_idx"])
        subject_idx.append(subject_to_id[parsed[session_key]["subject"]])
        session_source_views.append(beh["source_behavior_views"])
        all_speed_values.append(session_data["speed_values"])

        raw_neuron_counts.append(session_data["raw_counts"]["n_neurons_raw"])
        kept_neuron_counts.append(session_data["raw_counts"]["n_neurons_selected"])
        raw_trial_counts.append(session_data["trial_stats"]["ntrials_raw"])
        kept_trial_counts.append(session_data["trial_stats"]["ntrials_kept"])
        empty_trial_counts.append(session_data["trial_stats"]["empty_running_trials"])

    speed_values = np.concatenate(all_speed_values).astype(np.float32)
    q25, q50, q75 = np.quantile(speed_values, [0.25, 0.50, 0.75]).astype(np.float32)
    speed_edges = np.array([-np.inf, q25, q50, q75, np.inf], dtype=np.float32)

    full_output = [finalize_outputs(partials, speed_edges) for partials in output_partials]

    metadata = {
        "task_description": (
            "Head-fixed mice ran through 4 m virtual corridors with naturalistic textures. "
            "Neural activity was restricted to moving periods within the corridor and summarized "
            "over four 1 m bins aligned to corridor entry."
        ),
        "time_bin_size": float((1.0 / 0.60) * 1000.0),
        "temporal_alignment_event": "corridor entry (trial start)",
        "off_start": 0.0,
        "off_end": float(4.0 / 0.60),
        "binning_scheme": "4 spatial bins of 1 m each inside the 4 m corridor; moving frames only",
        "nominal_vr_speed_cm_per_s": 60.0,
        "frame_period_s_median": float(np.median(frame_periods)),
        "speed_bin_edges_cm_per_s": [float(q25), float(q50), float(q75)],
        "neuron_selection": (
            f"Visual-cortex neurons only (V1, mHV, lHV, aHV), then deterministic "
            f"variance-ranked cap at {max_neurons_per_session} neurons per session "
            f"with per-region balancing."
        ),
        "session_keys": spk_session_keys,
        "session_source_behavior_views": session_source_views,
        "n_sessions_raw": len(spk_session_keys),
        "n_subjects_raw": len(subjects),
        "raw_neuron_count_range": [int(np.min(raw_neuron_counts)), int(np.max(raw_neuron_counts))],
        "selected_neuron_count_range": [int(np.min(kept_neuron_counts)), int(np.max(kept_neuron_counts))],
        "raw_trial_count_range": [int(np.min(raw_trial_counts)), int(np.max(raw_trial_counts))],
        "kept_trial_count_range": [int(np.min(kept_trial_counts)), int(np.max(kept_trial_counts))],
        "empty_running_trial_total": int(np.sum(empty_trial_counts)),
    }

    data = {
        "neural": full_neural,
        "input": full_input,
        "output": full_output,
        "subjects": subjects,
        "subject_idx": np.array(subject_idx, dtype=np.int16),
        "brain_regions": BRAIN_REGIONS,
        "brain_region_idx": brain_region_idx,
        "input_names": [
            "time_to_sound_cue_s",
            "day_of_training",
            "time_since_trial_start_s",
            "reward_availability",
        ],
        "output_names": [
            "visual_stimulus_category",
            "licking",
            "position_bin",
            "running_speed_bin",
        ],
        "output_values": [
            stimulus_values,
            ["no_lick", "lick"],
            ["0_to_1m", "1_to_2m", "2_to_3m", "3_to_4m"],
            ["speed_q1", "speed_q2", "speed_q3", "speed_q4"],
        ],
        "metadata": metadata,
    }

    print()
    print("Sanity checks")
    print(f"  Physical sessions (behavior index): {len(behavior_by_session)}")
    print(f"  Subjects: {len(subjects)}")
    print(f"  Raw neuron count range: {min(raw_neuron_counts)} to {max(raw_neuron_counts)}")
    print(f"  Selected neuron count range: {min(kept_neuron_counts)} to {max(kept_neuron_counts)}")
    print(f"  Raw trial count range: {min(raw_trial_counts)} to {max(raw_trial_counts)}")
    print(f"  Kept trial count range: {min(kept_trial_counts)} to {max(kept_trial_counts)}")
    print(f"  Empty running trials dropped: {int(np.sum(empty_trial_counts))}")
    print(f"  Median frame period (s): {metadata['frame_period_s_median']:.6f}")
    print(
        "  Speed quantile edges (cm/s): "
        f"{metadata['speed_bin_edges_cm_per_s'][0]:.3f}, "
        f"{metadata['speed_bin_edges_cm_per_s'][1]:.3f}, "
        f"{metadata['speed_bin_edges_cm_per_s'][2]:.3f}"
    )
    print(f"  Stimulus categories ({len(stimulus_values)}): {', '.join(stimulus_values)}")

    return data


def write_pickle(path: Path, payload):
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(description="Convert Zhong et al. 2025 imaging data into decoder format.")
    parser.add_argument("--full-output", default="/app/converted_data.pkl", help="Path to the full output pickle.")
    parser.add_argument("--sample-output", default="/app/sample_data.pkl", help="Path to the sample output pickle.")
    parser.add_argument("--max-neurons-per-session", type=int, default=DEFAULT_MAX_NEURONS)
    parser.add_argument("--sample-session-count", type=int, default=DEFAULT_SAMPLE_SESSIONS)
    parser.add_argument("--sample-only", action="store_true", help="Write only the sample pickle.")
    parser.add_argument("--full-only", action="store_true", help="Write only the full pickle.")
    args = parser.parse_args()

    if args.sample_only and args.full_only:
        raise ValueError("--sample-only and --full-only cannot both be set")

    data = convert_dataset(max_neurons_per_session=args.max_neurons_per_session)
    sample = make_sample_dataset(data, sample_session_count=args.sample_session_count)

    full_output = Path(args.full_output)
    sample_output = Path(args.sample_output)

    if not args.sample_only:
        write_pickle(full_output, data)
        print(f"\nSaved full dataset to {full_output}")
    if not args.full_only:
        write_pickle(sample_output, sample)
        print(f"Saved sample dataset to {sample_output}")


if __name__ == "__main__":
    main()
