import argparse
import copy
import gc
import os
import pickle
import re
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))
import utils  # noqa: E402


ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")
BEH_DIR = os.path.join(DATA_DIR, "beh")
SPK_DIR = os.path.join(DATA_DIR, "spk")
RETINO_DIR = os.path.join(DATA_DIR, "retinotopy")
EXP_INFO_PATH = os.path.join(BEH_DIR, "Imaging_Exp_info.npy")

SUP_GROUP_PRIORITY = [
    "sup_train1_before_learning",
    "sup_train1_after_learning",
    "sup_test1",
    "sup_train2_before_learning",
    "sup_train2_after_learning",
    "sup_test2",
    "sup_test3",
]
GROUPED_REGIONS = ["V1", "mHV", "lHV", "aHV", "outside_visual_cortex"]
AREA_SELECTION_ORDER = ["V1", "mHV", "lHV", "aHV"]
REGION_TO_INDEX = {name: idx for idx, name in enumerate(GROUPED_REGIONS)}
DP_THRESHOLD = 0.3
MAX_NEURONS = 512
N_POSITION_BINS_TOTAL = 60
N_POSITION_BINS_CORRIDOR = 40
POSITION_STEP_M = 0.1
VR_SPEED_M_PER_S = 0.6
TIME_BIN_SIZE_MS = POSITION_STEP_M / VR_SPEED_M_PER_S * 1000.0
MIN_AREA_NEURONS = MAX_NEURONS // len(AREA_SELECTION_ORDER)
DEFAULT_SAMPLE_SESSIONS = 5
DEFAULT_SAMPLE_TRIALS = 60


def natural_sort_key(value):
    parts = re.split(r"(\d+)", str(value))
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return key


def parse_date(date_str):
    return datetime.strptime(date_str, "%Y_%m_%d")


def load_exp_info():
    return np.load(EXP_INFO_PATH, allow_pickle=True).item()


def load_behavior(group_name):
    path = os.path.join(BEH_DIR, f"Beh_{group_name}.npy")
    return np.load(path, allow_pickle=True).item()


def grouped_region_indices(iarea):
    region_idx = np.full(len(iarea), REGION_TO_INDEX["outside_visual_cortex"], dtype=np.int16)
    region_idx[np.asarray(iarea == 8)] = REGION_TO_INDEX["V1"]
    region_idx[np.asarray(np.isin(iarea, [0, 1, 2, 9]))] = REGION_TO_INDEX["mHV"]
    region_idx[np.asarray(np.isin(iarea, [5, 6]))] = REGION_TO_INDEX["lHV"]
    region_idx[np.asarray(np.isin(iarea, [3, 4]))] = REGION_TO_INDEX["aHV"]
    return region_idx


def build_session_catalog(exp_info):
    session_candidates = defaultdict(list)
    for group_name in SUP_GROUP_PRIORITY:
        for entry in exp_info[group_name]:
            base_session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
            behavior_key = base_session_id
            if "stimtype" in entry:
                behavior_key = f"{behavior_key}_{entry['stimtype']}"
            session_candidates[base_session_id].append(
                {
                    "group_name": group_name,
                    "behavior_key": behavior_key,
                    "entry": entry,
                }
            )

    catalog = []
    for base_session_id in sorted(session_candidates):
        candidates = sorted(
            session_candidates[base_session_id],
            key=lambda item: SUP_GROUP_PRIORITY.index(item["group_name"]),
        )
        canonical = candidates[0]
        entry = canonical["entry"]
        catalog.append(
            {
                "base_session_id": base_session_id,
                "behavior_group": canonical["group_name"],
                "behavior_key": canonical["behavior_key"],
                "subject": entry["mname"],
                "date": entry["datexp"],
                "block": entry["blk"],
                "source_groups": [item["group_name"] for item in candidates],
            }
        )
    return catalog


def attach_session_days(catalog):
    by_subject = defaultdict(list)
    for session in catalog:
        by_subject[session["subject"]].append(session)
    for subject_sessions in by_subject.values():
        subject_sessions.sort(key=lambda item: parse_date(item["date"]))
        first_date = parse_date(subject_sessions[0]["date"])
        for index, session in enumerate(subject_sessions):
            session_date = parse_date(session["date"])
            session["training_day_index"] = float((session_date - first_date).days)
            session["training_day_rank"] = int(index)
    catalog.sort(key=lambda item: (item["subject"], parse_date(item["date"]), item["block"]))
    return catalog


def collect_stimulus_names(catalog):
    stimulus_names = set()
    for session in catalog:
        beh = load_behavior(session["behavior_group"])[session["behavior_key"]]
        stimulus_names.update(map(str, beh["UniqWalls"]))
    return sorted(stimulus_names, key=natural_sort_key)


def select_neurons(dp, region_idx, max_neurons):
    candidate_mask = (region_idx != REGION_TO_INDEX["outside_visual_cortex"]) & (np.abs(dp) >= DP_THRESHOLD)
    selected = []
    selected_set = set()
    abs_dp = np.abs(dp)

    for region_name in AREA_SELECTION_ORDER:
        region_code = REGION_TO_INDEX[region_name]
        region_candidates = np.flatnonzero(candidate_mask & (region_idx == region_code))
        if region_candidates.size == 0:
            continue
        order = region_candidates[np.argsort(abs_dp[region_candidates])[::-1]]
        for neuron_idx in order[:MIN_AREA_NEURONS]:
            selected.append(int(neuron_idx))
            selected_set.add(int(neuron_idx))

    if len(selected) < max_neurons:
        remaining_candidates = np.flatnonzero(candidate_mask)
        remaining_candidates = remaining_candidates[np.argsort(abs_dp[remaining_candidates])[::-1]]
        for neuron_idx in remaining_candidates:
            neuron_idx = int(neuron_idx)
            if neuron_idx in selected_set:
                continue
            selected.append(neuron_idx)
            selected_set.add(neuron_idx)
            if len(selected) >= max_neurons:
                break

    if len(selected) < max_neurons:
        fallback_candidates = np.flatnonzero(region_idx != REGION_TO_INDEX["outside_visual_cortex"])
        fallback_candidates = fallback_candidates[np.argsort(abs_dp[fallback_candidates])[::-1]]
        for neuron_idx in fallback_candidates:
            neuron_idx = int(neuron_idx)
            if neuron_idx in selected_set:
                continue
            selected.append(neuron_idx)
            selected_set.add(neuron_idx)
            if len(selected) >= max_neurons:
                break

    selected = np.asarray(selected[:max_neurons], dtype=np.int32)
    selected = selected[np.argsort(abs_dp[selected])[::-1]]
    return np.sort(selected)


def interpolate_running_signal(signal_2d, accumulated_position, corridor_length, ntrials):
    return utils.spk_pos_interp(
        raw_spk=signal_2d,
        accum_pos=accumulated_position,
        corridorLen=corridor_length,
        new_shape=[ntrials, 0],
    )


def bin_licks(lick_positions, nbins):
    lick_bins = np.zeros(nbins, dtype=np.int16)
    if lick_positions.size == 0:
        return lick_bins
    valid_positions = lick_positions[(lick_positions >= 0) & (lick_positions < nbins)]
    if valid_positions.size == 0:
        return lick_bins
    indices = np.floor(valid_positions).astype(int)
    lick_bins[np.clip(indices, 0, nbins - 1)] = 1
    return lick_bins


def make_sample_dataset(data, max_sessions, max_trials):
    max_sessions = min(max_sessions, len(data["neural"]))
    keep_sessions = []
    seen_subjects = set()
    for session_idx, subject_code in enumerate(data["subject_idx"]):
        subject_code = int(subject_code)
        if subject_code in seen_subjects:
            continue
        keep_sessions.append(session_idx)
        seen_subjects.add(subject_code)
        if len(keep_sessions) >= max_sessions:
            break
    if len(keep_sessions) < max_sessions:
        for session_idx in range(len(data["neural"])):
            if session_idx in keep_sessions:
                continue
            keep_sessions.append(session_idx)
            if len(keep_sessions) >= max_sessions:
                break

    used_subject_codes = sorted({int(data["subject_idx"][idx]) for idx in keep_sessions})
    subject_code_remap = {old: new for new, old in enumerate(used_subject_codes)}
    sample = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": [data["subjects"][idx] for idx in used_subject_codes],
        "subject_idx": np.asarray(
            [subject_code_remap[int(data["subject_idx"][idx])] for idx in keep_sessions],
            dtype=np.int16,
        ),
        "brain_regions": list(data["brain_regions"]),
        "brain_region_idx": [],
        "input_names": list(data["input_names"]),
        "output_names": list(data["output_names"]),
        "output_values": copy.deepcopy(data["output_values"]),
        "metadata": copy.deepcopy(data["metadata"]),
    }
    for session_idx in keep_sessions:
        trial_slice = slice(0, min(max_trials, len(data["neural"][session_idx])))
        sample["neural"].append(data["neural"][session_idx][trial_slice])
        sample["input"].append(data["input"][session_idx][trial_slice])
        sample["output"].append(data["output"][session_idx][trial_slice])
        sample["brain_region_idx"].append(data["brain_region_idx"][session_idx].copy())
    sample["metadata"]["session_info"] = sample["metadata"]["session_info"][: len(keep_sessions)]
    sample["metadata"]["sample_from_full_dataset"] = True
    return sample


def convert_dataset(max_neurons, limit_sessions=None):
    exp_info = load_exp_info()
    catalog = attach_session_days(build_session_catalog(exp_info))
    if limit_sessions is not None:
        catalog = catalog[:limit_sessions]
    stimulus_names = collect_stimulus_names(catalog)
    stimulus_to_index = {name: idx for idx, name in enumerate(stimulus_names)}
    subjects = sorted({session["subject"] for session in catalog})
    subject_to_index = {subject: idx for idx, subject in enumerate(subjects)}

    position_values = [f"{start}-{start + 1}m" for start in range(4)]
    output_values = [
        stimulus_names,
        ["no_lick", "lick"],
        position_values,
        ["speed_q1", "speed_q2", "speed_q3", "speed_q4"],
    ]

    position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
    position_units = np.arange(N_POSITION_BINS_CORRIDOR, dtype=np.float32)
    time_since_start = position_units / 6.0

    neural_sessions = []
    input_sessions = []
    output_sessions = []
    speed_sessions = []
    brain_region_idx_sessions = []
    subject_idx = []
    session_info = []
    all_speed_values = []
    selected_neuron_counts = []
    total_neuron_counts = []
    stimulus_counts = defaultdict(int)

    for session_idx, session in enumerate(catalog, start=1):
        print(
            f"[{session_idx:02d}/{len(catalog)}] {session['base_session_id']} "
            f"from {session['behavior_group']}",
            flush=True,
        )

        beh = load_behavior(session["behavior_group"])[session["behavior_key"]]
        ntrials = int(beh["ntrials"])
        corridor_length = int(round(float(beh["Corridor_Length"])))
        if corridor_length != N_POSITION_BINS_TOTAL:
            raise ValueError(
                f"{session['base_session_id']}: expected corridor length {N_POSITION_BINS_TOTAL}, "
                f"found {corridor_length}"
            )

        spk = utils.load_spk(
            {"mname": session["subject"], "datexp": session["date"], "blk": session["block"]},
            root=SPK_DIR,
        )
        iarea = np.load(
            os.path.join(RETINO_DIR, f"{session['subject']}_{session['date']}_trans.npz"),
            allow_pickle=True,
        )["iarea"]
        region_idx_all = grouped_region_indices(iarea)

        nframes = spk.shape[1]
        vr_move = np.asarray(beh["ft_move"][:nframes] > 0)
        corridor_frames = np.asarray(beh["ft_CorrSpc"][:nframes], dtype=bool)
        valid_corridor_frames = vr_move & corridor_frames
        wall_ids = np.asarray(beh["ft_WallID"][:nframes])
        stim_names = np.asarray(beh["UniqWalls"]).astype(str)
        stim_id = np.asarray(beh["stim_id"], dtype=float)
        familiar_rewarded = stim_names[np.where(stim_id == 2)[0][0]]
        familiar_unrewarded = stim_names[np.where(stim_id == 0)[0][0]]

        stim1_frames = (wall_ids == familiar_rewarded) & valid_corridor_frames
        stim2_frames = (wall_ids == familiar_unrewarded) & valid_corridor_frames
        if stim1_frames.sum() == 0 or stim2_frames.sum() == 0:
            raise ValueError(f"{session['base_session_id']}: missing familiar corridor frames for d' selection")

        stim1_mean = np.nanmean(spk[:, stim1_frames], axis=1)
        stim2_mean = np.nanmean(spk[:, stim2_frames], axis=1)
        stim1_std = np.nanstd(spk[:, stim1_frames], axis=1)
        stim2_std = np.nanstd(spk[:, stim2_frames], axis=1)
        dp = 2.0 * (stim1_mean - stim2_mean) / (stim1_std + stim2_std + 1e-12)

        selected_neurons = select_neurons(dp, region_idx_all, max_neurons=max_neurons)
        selected_regions = region_idx_all[selected_neurons].astype(np.int16)
        total_neuron_counts.append(int(spk.shape[0]))
        selected_neuron_counts.append(int(selected_neurons.size))
        print(
            f"  neurons: total={spk.shape[0]} selected={selected_neurons.size} "
            f"(threshold |d'|>={DP_THRESHOLD})",
            flush=True,
        )

        move_position = np.asarray(beh["ft_PosCum"][:nframes][vr_move], dtype=np.float64)
        move_spk = np.asarray(spk[selected_neurons][:, vr_move], dtype=np.float32)
        interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
        interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR].astype(np.float16, copy=False)

        move_speed = np.asarray(beh["ft_RunSpeed"][:nframes][vr_move], dtype=np.float32)[None, :]
        interp_speed = interpolate_running_signal(move_speed, move_position, corridor_length, ntrials)[0]
        interp_speed = interp_speed[:, :N_POSITION_BINS_CORRIDOR].astype(np.float32, copy=False)
        all_speed_values.append(interp_speed.reshape(-1))

        lick_positions = np.asarray(beh["LickPos"], dtype=np.float32)
        lick_trial_index = np.asarray(beh["LickTrind"], dtype=np.int64)
        sound_positions = np.asarray(beh["SoundPos"], dtype=np.float32)
        trial_wall_names = np.asarray(beh["WallName"]).astype(str)
        trial_is_rewarded = np.asarray(beh["isRew"], dtype=bool)

        session_neural = []
        session_input = []
        session_output = []
        session_speed = []
        session_stimulus_counter = defaultdict(int)

        for trial_idx in range(ntrials):
            trial_stimulus = str(trial_wall_names[trial_idx])
            session_stimulus_counter[trial_stimulus] += 1
            stimulus_counts[trial_stimulus] += 1

            cue_position = float(sound_positions[trial_idx])
            if not np.isfinite(cue_position):
                raise ValueError(f"{session['base_session_id']}: non-finite SoundPos on trial {trial_idx}")

            lick_bins = bin_licks(lick_positions[lick_trial_index == trial_idx], N_POSITION_BINS_CORRIDOR)
            input_trial = np.vstack(
                [
                    cue_position / 6.0 - time_since_start,
                    np.full(N_POSITION_BINS_CORRIDOR, session["training_day_index"], dtype=np.float32),
                    time_since_start,
                    np.full(
                        N_POSITION_BINS_CORRIDOR,
                        float(trial_is_rewarded[trial_idx]),
                        dtype=np.float32,
                    ),
                ]
            ).astype(np.float32, copy=False)
            output_trial = np.vstack(
                [
                    np.full(
                        N_POSITION_BINS_CORRIDOR,
                        stimulus_to_index[trial_stimulus],
                        dtype=np.int16,
                    ),
                    lick_bins,
                    position_indices,
                    np.zeros(N_POSITION_BINS_CORRIDOR, dtype=np.int16),
                ]
            )

            session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
            session_input.append(np.ascontiguousarray(input_trial))
            session_output.append(np.ascontiguousarray(output_trial))
            session_speed.append(np.ascontiguousarray(interp_speed[trial_idx]))

        neural_sessions.append(session_neural)
        input_sessions.append(session_input)
        output_sessions.append(session_output)
        speed_sessions.append(session_speed)
        brain_region_idx_sessions.append(np.ascontiguousarray(selected_regions))
        subject_idx.append(subject_to_index[session["subject"]])
        session_info.append(
            {
                "base_session_id": session["base_session_id"],
                "behavior_group": session["behavior_group"],
                "behavior_key": session["behavior_key"],
                "source_groups": session["source_groups"],
                "subject": session["subject"],
                "date": session["date"],
                "block": session["block"],
                "training_day_index": session["training_day_index"],
                "training_day_rank": session["training_day_rank"],
                "n_trials": ntrials,
                "n_neurons_total": int(spk.shape[0]),
                "n_neurons_selected": int(selected_neurons.size),
                "stimulus_counts": dict(sorted(session_stimulus_counter.items(), key=lambda item: natural_sort_key(item[0]))),
            }
        )

        del spk, move_spk, interp_spk, interp_speed, move_speed
        gc.collect()

    speed_values = np.concatenate(all_speed_values)
    speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
    print(f"Global running-speed quartile edges: {speed_edges.tolist()}", flush=True)

    for session_outputs, session_speeds in zip(output_sessions, speed_sessions):
        for output_trial, speed_trial in zip(session_outputs, session_speeds):
            output_trial[3, :] = np.digitize(speed_trial, speed_edges, right=False).astype(np.int16)

    data = {
        "neural": neural_sessions,
        "input": input_sessions,
        "output": output_sessions,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int16),
        "brain_regions": GROUPED_REGIONS,
        "brain_region_idx": brain_region_idx_sessions,
        "input_names": [
            "time_to_sound_cue_s",
            "day_of_training",
            "time_since_trial_start_s",
            "reward_availability",
        ],
        "output_names": [
            "visual_stimulus_category",
            "licking",
            "corridor_position_bin",
            "running_speed_bin",
        ],
        "output_values": output_values,
        "metadata": {
            "task_description": (
                "Task-mouse virtual-corridor recordings from Zhong et al. 2025. "
                "Trials are aligned to corridor entry, use running-only frames, and neural "
                "activity is interpolated onto 10 cm corridor bins across the 4 m corridor."
            ),
            "time_bin_size": TIME_BIN_SIZE_MS,
            "temporal_alignment_event": "corridor entry / trial start",
            "off_start": 0.0,
            "off_end": float(N_POSITION_BINS_CORRIDOR / 6.0),
            "corridor_length_m": 4.0,
            "bin_size_m": POSITION_STEP_M,
            "vr_speed_m_per_s": VR_SPEED_M_PER_S,
            "n_sessions": len(catalog),
            "n_subjects": len(subjects),
            "session_groups_included": SUP_GROUP_PRIORITY,
            "rewarded_task_only": True,
            "rewarded_task_only_reason": (
                "The decoder input requires reward availability per trial. "
                "Unsupervised and naive cohorts do not have rewarded corridors."
            ),
            "neuron_selection": {
                "source": "visual cortex grouped using retinotopy",
                "dprime_reference": "familiar rewarded vs familiar unrewarded corridor during running in corridor",
                "dprime_threshold_abs": DP_THRESHOLD,
                "max_neurons_per_session": max_neurons,
                "selection_strategy": (
                    "Balanced top-|d'| sampling across V1, mHV, lHV, and aHV, "
                    "with remaining slots filled by highest-|d'| visual-cortex neurons."
                ),
            },
            "running_speed_bin_edges": speed_edges.tolist(),
            "stimulus_counts": dict(sorted(stimulus_counts.items(), key=lambda item: natural_sort_key(item[0]))),
            "selected_neuron_count_range": [
                int(np.min(selected_neuron_counts)),
                int(np.max(selected_neuron_counts)),
            ],
            "original_neuron_count_range": [
                int(np.min(total_neuron_counts)),
                int(np.max(total_neuron_counts)),
            ],
            "session_info": session_info,
        },
    }

    return data


def main():
    parser = argparse.ArgumentParser(description="Convert Zhong et al. 2025 task-mouse imaging data.")
    parser.add_argument(
        "--full-out",
        default=os.path.join(ROOT, "converted_data.pkl"),
        help="Path for the full converted dataset pickle.",
    )
    parser.add_argument(
        "--sample-out",
        default=os.path.join(ROOT, "sample_data.pkl"),
        help="Path for the sample converted dataset pickle.",
    )
    parser.add_argument(
        "--max-neurons",
        type=int,
        default=MAX_NEURONS,
        help="Maximum number of selected neurons per session.",
    )
    parser.add_argument(
        "--sample-sessions",
        type=int,
        default=DEFAULT_SAMPLE_SESSIONS,
        help="Number of sessions to keep in sample_data.pkl.",
    )
    parser.add_argument(
        "--sample-trials",
        type=int,
        default=DEFAULT_SAMPLE_TRIALS,
        help="Maximum number of trials per session in sample_data.pkl.",
    )
    parser.add_argument(
        "--limit-sessions",
        type=int,
        default=None,
        help="Optional debugging limit on the number of sessions to convert.",
    )
    args = parser.parse_args()

    print("Building converted dataset...", flush=True)
    data = convert_dataset(
        max_neurons=args.max_neurons,
        limit_sessions=args.limit_sessions,
    )
    sample_data = make_sample_dataset(
        data,
        max_sessions=args.sample_sessions,
        max_trials=args.sample_trials,
    )

    with open(args.full_out, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved full dataset to {args.full_out}", flush=True)

    with open(args.sample_out, "wb") as handle:
        pickle.dump(sample_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved sample dataset to {args.sample_out}", flush=True)

    total_trials = sum(len(session_trials) for session_trials in data["neural"])
    print(
        f"Summary: sessions={len(data['neural'])}, trials={total_trials}, "
        f"subjects={len(data['subjects'])}",
        flush=True,
    )
    print(
        f"Neuron count range (selected): "
        f"{data['metadata']['selected_neuron_count_range']}",
        flush=True,
    )
    print(
        f"Neuron count range (original): "
        f"{data['metadata']['original_neuron_count_range']}",
        flush=True,
    )
    print(
        f"Stimuli: {', '.join(data['output_values'][0])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
