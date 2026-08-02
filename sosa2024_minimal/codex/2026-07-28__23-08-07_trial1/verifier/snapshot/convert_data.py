import argparse
import pickle
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d


ZONE_BOUNDS_CM = {
    "A": (80.0, 130.0),
    "B": (200.0, 250.0),
    "C": (320.0, 370.0),
}
ZONE_TO_IDX = {"A": 0, "B": 1, "C": 2}
ENV_TO_IDX = {"Env1": 0, "Env2": 1}
SAMPLE_SUBJECT = "m3"
SAMPLE_EXP_DAYS = {1, 3, 5, 7, 8, 10, 12, 14}
CHANGE_TRIAL = 30


def decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode_scalar(value[()])
    return str(value)


def subject_session_key(path: Path):
    subject = int(path.parent.name.split("-m")[-1])
    session = int(path.stem.split("_ses-")[-1].split("_")[0])
    return subject, session


def scene_schedule(scene: str, ntrials: int, change_trial: int = CHANGE_TRIAL):
    env_by_trial = np.zeros(ntrials, dtype=np.int64)
    zone_labels = np.empty(ntrials, dtype=object)

    fixed_match = re.fullmatch(r"(Env[12])_Location([ABC])", scene)
    same_env_switch_match = re.fullmatch(r"(Env[12])_Location([ABC])_to_([ABC])", scene)
    env_switch_match = re.fullmatch(r"(Env[12])_([ABC])_to_(Env[12])_([ABC])", scene)

    if fixed_match:
        env = ENV_TO_IDX[fixed_match.group(1)]
        zone = fixed_match.group(2)
        env_by_trial[:] = env
        zone_labels[:] = zone
    elif same_env_switch_match:
        env = ENV_TO_IDX[same_env_switch_match.group(1)]
        zone0 = same_env_switch_match.group(2)
        zone1 = same_env_switch_match.group(3)
        split = min(change_trial, ntrials)
        env_by_trial[:] = env
        zone_labels[:split] = zone0
        zone_labels[split:] = zone1
    elif env_switch_match:
        env0 = ENV_TO_IDX[env_switch_match.group(1)]
        zone0 = env_switch_match.group(2)
        env1 = ENV_TO_IDX[env_switch_match.group(3)]
        zone1 = env_switch_match.group(4)
        split = min(change_trial, ntrials)
        env_by_trial[:split] = env0
        env_by_trial[split:] = env1
        zone_labels[:split] = zone0
        zone_labels[split:] = zone1
    else:
        raise ValueError(f"Unrecognized scene format: {scene}")

    zone_idx = np.array([ZONE_TO_IDX[z] for z in zone_labels], dtype=np.int64)
    zone_bounds = np.array([ZONE_BOUNDS_CM[z] for z in zone_labels], dtype=np.float32)
    return env_by_trial, zone_labels, zone_idx, zone_bounds


def find_trial_segments(position: np.ndarray, trial_start: np.ndarray):
    starts = np.flatnonzero(trial_start > 0)
    next_starts = np.concatenate([starts[1:], [len(position)]])
    ends = []

    for start, next_start in zip(starts, next_starts):
        trial_pos = position[start:next_start]
        valid = np.flatnonzero((trial_pos >= 0.0) & (trial_pos <= 450.5))
        if len(valid) == 0:
            continue
        end = start + valid[-1] + 1
        if end <= start:
            continue
        ends.append(end)

    starts = starts[:len(ends)]
    ends = np.array(ends, dtype=np.int64)
    return starts.astype(np.int64), ends


def get_plane_names(group: h5py.Group):
    return sorted(
        [name for name in group.keys() if name.startswith("plane")],
        key=lambda name: int(name.replace("plane", "")),
    )


def get_curated_plane_masks(f: h5py.File):
    seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]
    iscell = np.asarray(seg["iscell"])[:, 0].astype(bool)
    if "planeIdx" in seg:
        plane_idx = np.asarray(seg["planeIdx"]).astype(int)
    else:
        plane_idx = np.zeros(len(iscell), dtype=int)

    plane_names = get_plane_names(f["processing/ophys/Deconvolved"])
    masks = {}
    for plane_name in plane_names:
        plane_num = int(plane_name.replace("plane", ""))
        mask = iscell[plane_idx == plane_num]
        ncols = f["processing/ophys/Deconvolved"][plane_name]["data"].shape[1]
        if mask.shape[0] != ncols:
            raise ValueError(
                f"{plane_name}: iscell mask length {mask.shape[0]} does not match data columns {ncols}"
            )
        masks[plane_name] = mask
    return masks


def compute_interneuron_mask(
    f: h5py.File,
    plane_masks: dict[str, np.ndarray],
    starts: np.ndarray,
    ends: np.ndarray,
    speed: np.ndarray,
    common_length: int,
):
    fluorescence = f["processing/ophys/Fluorescence"]
    neuropil = f["processing/ophys/Neuropil"]

    all_is_int = []
    valid_mask = np.zeros(len(speed), dtype=bool)
    for start, end in zip(starts, ends):
        valid_mask[start:end] = True
    speed_valid = speed[valid_mask].astype(np.float32)
    speed_centered = speed_valid - speed_valid.mean()
    speed_ss = np.sum(speed_centered ** 2)

    for plane_name in get_plane_names(fluorescence):
        curated_mask = plane_masks[plane_name]
        if not np.any(curated_mask):
            continue

        f_roi = np.asarray(fluorescence[plane_name]["data"][:common_length, curated_mask], dtype=np.float32).T
        f_neu = np.asarray(neuropil[plane_name]["data"][:common_length, curated_mask], dtype=np.float32).T
        dff = np.full_like(f_roi, np.nan, dtype=np.float32)

        for start, end in zip(starts, ends):
            trial_roi = f_roi[:, start:end] - 0.7 * f_neu[:, start:end]
            trial_roi = trial_roi + 0.7 * np.mean(f_neu[:, start:end], axis=1, keepdims=True)
            baseline = gaussian_filter1d(trial_roi, sigma=15, axis=1, mode="nearest")
            baseline = minimum_filter1d(baseline, size=300, axis=1, mode="nearest")
            baseline = maximum_filter1d(baseline, size=300, axis=1, mode="nearest")
            trial_dff = (trial_roi - baseline) / np.abs(baseline)
            trial_dff = gaussian_filter1d(trial_dff, sigma=2, axis=1, mode="nearest")
            dff[:, start:end] = trial_dff

        dff_valid = dff[:, valid_mask]
        dff_centered = dff_valid - np.nanmean(dff_valid, axis=1, keepdims=True)
        numerator = np.nansum(dff_centered * speed_centered[None, :], axis=1)
        denom = np.sqrt(np.nansum(dff_centered ** 2, axis=1) * speed_ss)
        corr = np.divide(
            numerator,
            denom,
            out=np.zeros_like(numerator, dtype=np.float32),
            where=denom > 0,
        )
        all_is_int.append(corr > 0.5)

        del f_roi, f_neu, dff, dff_valid, dff_centered

    if not all_is_int:
        return np.zeros(0, dtype=bool)
    return np.concatenate(all_is_int, axis=0)


def load_curated_events(
    f: h5py.File,
    plane_masks: dict[str, np.ndarray],
    keep_mask: np.ndarray,
    common_length: int,
):
    events = f["processing/ophys/Deconvolved"]
    plane_names = get_plane_names(events)
    keep_splits = []
    cursor = 0
    for plane_name in plane_names:
        n_curated = int(np.sum(plane_masks[plane_name]))
        keep_splits.append(keep_mask[cursor:cursor + n_curated])
        cursor += n_curated
    if cursor != keep_mask.shape[0]:
        raise ValueError("Interneuron keep mask does not match curated cell count")

    arrays = []
    for plane_name, plane_keep in zip(plane_names, keep_splits):
        if not np.any(plane_keep):
            continue
        curated_mask = plane_masks[plane_name]
        plane_data = np.asarray(events[plane_name]["data"][:common_length, curated_mask], dtype=np.float32)
        arrays.append(plane_data[:, plane_keep])
    if not arrays:
        return np.zeros((common_length, 0), dtype=np.float32)
    return np.concatenate(arrays, axis=1)


def discretize_distance_to_zone(position_cm: np.ndarray, zone_start: float, zone_end: float):
    dist = np.zeros_like(position_cm, dtype=np.float32)
    before = position_cm < zone_start
    after = position_cm > zone_end
    dist[before] = position_cm[before] - zone_start
    dist[after] = position_cm[after] - zone_end

    bins = np.empty(position_cm.shape[0], dtype=np.int64)
    bins[dist < -50.0] = 0
    bins[(dist >= -50.0) & (dist < -10.0)] = 1
    bins[(dist >= -10.0) & (dist < 0.0)] = 2
    bins[dist == 0.0] = 3
    bins[(dist > 0.0) & (dist <= 10.0)] = 4
    bins[(dist > 10.0) & (dist <= 50.0)] = 5
    bins[dist > 50.0] = 6
    return bins


def discretize_position(position_cm: np.ndarray):
    clipped = np.clip(position_cm, 0.0, 450.0)
    bins = np.floor(clipped / 90.0).astype(np.int64)
    bins[bins > 4] = 4
    return bins


def discretize_speed(speed_cm_s: np.ndarray):
    bins = np.empty(speed_cm_s.shape[0], dtype=np.int64)
    bins[speed_cm_s < 2.0] = 0
    bins[(speed_cm_s >= 2.0) & (speed_cm_s < 10.0)] = 1
    bins[(speed_cm_s >= 10.0) & (speed_cm_s < 20.0)] = 2
    bins[(speed_cm_s >= 20.0) & (speed_cm_s <= 40.0)] = 3
    bins[speed_cm_s > 40.0] = 4
    return bins


def build_session(path: Path):
    with h5py.File(path, "r") as f:
        subject = decode_scalar(f["general/subject/subject_id"][()])
        identifier = decode_scalar(f["identifier"][()])
        session_id = decode_scalar(f["general/session_id"][()])
        scene = identifier.split("/")[-1]
        date = identifier.split("/")[-2]

        beh = f["processing/behavior/BehavioralTimeSeries"]
        position = np.asarray(beh["position/data"], dtype=np.float32)
        position_ts = np.asarray(beh["position/timestamps"], dtype=np.float64)
        speed = np.asarray(beh["speed/data"], dtype=np.float32)
        lick = np.asarray(beh["lick/data"], dtype=np.float32)
        environment = np.asarray(beh["environment/data"], dtype=np.float32)
        reward_zone = np.asarray(beh["reward_zone/data"], dtype=np.float32)
        trial_start = np.asarray(beh["trial_start/data"], dtype=np.float32)
        reward_ts = np.asarray(beh["Reward/timestamps"], dtype=np.float64)
        dense_lengths = [
            len(position),
            len(position_ts),
            len(speed),
            len(lick),
            len(environment),
            len(reward_zone),
            len(trial_start),
        ]
        for grp_name in ("Deconvolved", "Fluorescence", "Neuropil"):
            grp = f[f"processing/ophys/{grp_name}"]
            for plane_name in get_plane_names(grp):
                dense_lengths.append(grp[plane_name]["data"].shape[0])
        common_length = min(dense_lengths)
        clipped_samples = max(dense_lengths) - common_length
        position = position[:common_length]
        position_ts = position_ts[:common_length]
        speed = speed[:common_length]
        lick = lick[:common_length]
        environment = environment[:common_length]
        reward_zone = reward_zone[:common_length]
        trial_start = trial_start[:common_length]

        starts, ends = find_trial_segments(position, trial_start)
        ntrials = len(starts)
        if ntrials < 2:
            raise ValueError(f"{path.name}: expected at least 2 trials, found {ntrials}")

        env_by_trial, zone_labels, zone_idx, zone_bounds = scene_schedule(scene, ntrials)

        behavior_env_by_trial = []
        parsed_zone_matches = 0
        parsed_zone_checks = 0
        reward_outcome = []
        trial_lengths = []
        next_starts = np.concatenate([starts[1:], [len(position)]])
        for trial_idx, (start, end, next_start) in enumerate(zip(starts, ends, next_starts)):
            env_vals = environment[start:end]
            env_vals = env_vals[env_vals >= 0]
            behavior_env_by_trial.append(int(round(float(np.median(env_vals)))))

            zone_vals = reward_zone[start:end] > 0
            if np.any(zone_vals):
                zone_pos = np.clip(position[start:end][zone_vals], 0.0, 450.0)
                center = float(np.mean(zone_pos))
                inferred = min(
                    ZONE_BOUNDS_CM,
                    key=lambda label: abs(center - np.mean(ZONE_BOUNDS_CM[label])),
                )
                parsed_zone_checks += 1
                parsed_zone_matches += int(inferred == zone_labels[trial_idx])

            reward_outcome.append(
                int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9)))
            )
            trial_lengths.append(end - start)

        env_mismatch = int(np.sum(env_by_trial != np.asarray(behavior_env_by_trial, dtype=np.int64)))

        plane_masks = get_curated_plane_masks(f)
        n_curated = int(sum(np.sum(mask) for mask in plane_masks.values()))
        is_int = compute_interneuron_mask(f, plane_masks, starts, ends, speed, common_length)
        keep_mask = ~is_int
        events = load_curated_events(f, plane_masks, keep_mask, common_length)
        n_neurons = events.shape[1]

        if n_neurons == 0:
            raise ValueError(f"{path.name}: no neurons remain after interneuron exclusion")

        neural_trials = []
        input_trials = []
        output_trials = []
        zone_counter = Counter()
        env_counter = Counter()
        for trial_idx, (start, end) in enumerate(zip(starts, ends)):
            pos_trial = np.clip(position[start:end], 0.0, 450.0)
            speed_trial = speed[start:end]
            lick_trial = (lick[start:end] > 0).astype(np.int64)
            zone_start, zone_end = zone_bounds[trial_idx]
            reward_trial = reward_outcome[trial_idx]
            prev_reward = reward_outcome[trial_idx - 1] if trial_idx > 0 else 0
            time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)

            neural = events[start:end].T.astype(np.float32)
            inputs = np.vstack([
                time_trial,
                np.full(time_trial.shape[0], env_by_trial[trial_idx], dtype=np.float32),
                np.full(time_trial.shape[0], trial_idx, dtype=np.float32),
                np.full(time_trial.shape[0], prev_reward, dtype=np.float32),
            ])
            outputs = np.vstack([
                discretize_distance_to_zone(pos_trial, zone_start, zone_end),
                discretize_position(pos_trial),
                discretize_speed(speed_trial),
                lick_trial,
                np.full(time_trial.shape[0], zone_idx[trial_idx], dtype=np.int64),
                np.full(time_trial.shape[0], reward_trial, dtype=np.int64),
            ])

            neural_trials.append(neural)
            input_trials.append(inputs)
            output_trials.append(outputs)
            zone_counter[zone_labels[trial_idx]] += 1
            env_counter[int(env_by_trial[trial_idx])] += 1

        dt_seconds = float(np.median(np.diff(position_ts)))
        summary = {
            "path": str(path),
            "subject": subject,
            "exp_day": int(session_id),
            "date": date,
            "scene": scene,
            "n_trials": int(ntrials),
            "n_curated_cells": int(n_curated),
            "n_interneurons_removed": int(np.sum(is_int)),
            "n_neurons_kept": int(n_neurons),
            "trial_length_frames_mean": float(np.mean(trial_lengths)),
            "trial_length_frames_min": int(np.min(trial_lengths)),
            "trial_length_frames_max": int(np.max(trial_lengths)),
            "rewarded_fraction": float(np.mean(reward_outcome)),
            "env_counts": {str(k): int(v) for k, v in sorted(env_counter.items())},
            "zone_counts": dict(zone_counter),
            "env_mismatch_trials": env_mismatch,
            "zone_match_fraction_when_observed": (
                float(parsed_zone_matches / parsed_zone_checks) if parsed_zone_checks else None
            ),
            "dt_seconds": dt_seconds,
            "samples_clipped_to_align_streams": int(clipped_samples),
        }

    return {
        "subject": subject,
        "session_summary": summary,
        "neural_trials": neural_trials,
        "input_trials": input_trials,
        "output_trials": output_trials,
        "brain_region_idx": np.zeros(n_neurons, dtype=np.int64),
    }


def summarize_dataset(data: dict):
    session_info = data["metadata"]["session_info"]
    ntrials = [len(session) for session in data["neural"]]
    nneurons = [session[0].shape[0] for session in data["neural"]]
    trial_lengths = [
        trial.shape[1]
        for session in data["neural"]
        for trial in session
    ]
    rewarded = []
    zones = []
    envs = []
    for session in data["output"]:
        for trial in session:
            rewarded.append(int(trial[5, 0]))
            zones.append(int(trial[4, 0]))
    for session in data["input"]:
        for trial in session:
            envs.append(int(trial[1, 0]))

    removed = [info["n_interneurons_removed"] for info in session_info]
    curated = [info["n_curated_cells"] for info in session_info]

    print(f"Sessions: {len(data['neural'])}")
    print(f"Subjects: {len(data['subjects'])} -> {data['subjects']}")
    print(f"Total trials: {sum(ntrials)}")
    print(
        "Trials/session mean±sd/min/max: "
        f"{np.mean(ntrials):.3f} ± {np.std(ntrials):.3f} / {np.min(ntrials)} / {np.max(ntrials)}"
    )
    print(
        "Neurons/session after filtering mean±sd/min/max: "
        f"{np.mean(nneurons):.3f} ± {np.std(nneurons):.3f} / {np.min(nneurons)} / {np.max(nneurons)}"
    )
    print(
        "Curated cells/session before interneuron filter mean±sd/min/max: "
        f"{np.mean(curated):.3f} ± {np.std(curated):.3f} / {np.min(curated)} / {np.max(curated)}"
    )
    print(
        "Interneuron removal/session mean±sd/max: "
        f"{np.mean(removed):.3f} ± {np.std(removed):.3f} / {np.max(removed)}"
    )
    print(
        "Interneuron fraction/session mean±sd: "
        f"{np.mean(np.array(removed) / np.array(curated)):.4f} ± "
        f"{np.std(np.array(removed) / np.array(curated)):.4f}"
    )
    print(
        "Trial length in frames mean±sd/min/max: "
        f"{np.mean(trial_lengths):.3f} ± {np.std(trial_lengths):.3f} / {np.min(trial_lengths)} / {np.max(trial_lengths)}"
    )
    print(
        "Trial length in seconds mean±sd: "
        f"{np.mean(trial_lengths) * data['metadata']['time_bin_size_ms'] / 1000.0:.3f} ± "
        f"{np.std(trial_lengths) * data['metadata']['time_bin_size_ms'] / 1000.0:.3f}"
    )
    print(f"Rewarded fraction: {np.mean(rewarded):.4f} (omission {1.0 - np.mean(rewarded):.4f})")
    print(f"Reward-zone trial counts: {dict(Counter(zones))}")
    print(f"Environment trial counts: {dict(Counter(envs))}")
    print(
        "Scene sanity: "
        f"{sum(info['env_mismatch_trials'] for info in session_info)} env-mismatched trials, "
        f"mean reward-zone agreement "
        f"{np.nanmean([info['zone_match_fraction_when_observed'] for info in session_info if info['zone_match_fraction_when_observed'] is not None]):.4f}"
    )
    print("Paper sanity checks:")
    print("  Switch-task mice expected: 11. Converted:", len(data["subjects"]))
    if "m11" in data["subjects"]:
        m11_sessions = int(np.sum(data["subject_idx"] == data["subjects"].index("m11")))
        print("  Imaging started on day 3 for m11 only. Converted sessions for m11:", m11_sessions)
    else:
        print("  Imaging started on day 3 for m11 only. Converted subset does not include m11.")
    print("  Switch sessions expected from days 3,5,7,8,10,12,14 across 11 mice: 77 total.")


def build_full_dataset(data_dir: Path):
    session_records = []
    for path in sorted(data_dir.glob("sub-*/sub-*_behavior+ophys.nwb"), key=subject_session_key):
        print(f"Converting {path.relative_to(data_dir.parent)}")
        record = build_session(path)
        info = record["session_summary"]
        print(
            f"  subject={info['subject']} day={info['exp_day']:02d} scene={info['scene']} "
            f"trials={info['n_trials']} curated={info['n_curated_cells']} "
            f"removed_int={info['n_interneurons_removed']} kept={info['n_neurons_kept']}"
        )
        session_records.append(record)

    subjects = []
    subject_to_idx = {}
    for record in session_records:
        subject = record["subject"]
        if subject not in subject_to_idx:
            subject_to_idx[subject] = len(subjects)
            subjects.append(subject)

    dt_all = [record["session_summary"]["dt_seconds"] for record in session_records]
    data = {
        "neural": [record["neural_trials"] for record in session_records],
        "input": [record["input_trials"] for record in session_records],
        "output": [record["output_trials"] for record in session_records],
        "subjects": subjects,
        "subject_idx": np.array([subject_to_idx[record["subject"]] for record in session_records], dtype=np.int64),
        "brain_regions": ["CA1"],
        "brain_region_idx": [record["brain_region_idx"] for record in session_records],
        "input_names": [
            "time_from_trial_start_s",
            "environment_type",
            "trial_number",
            "previous_trial_outcome",
        ],
        "output_names": [
            "distance_to_reward_zone",
            "absolute_position_bin",
            "speed_bin",
            "lick",
            "reward_zone_location",
            "reward_outcome",
        ],
        "output_values": [
            ["lt_-50cm", "-50_to_-10cm", "-10_to_lt_0cm", "0cm", "gt_0_to_10cm", "10_to_50cm", "gt_50cm"],
            ["bin0", "bin1", "bin2", "bin3", "bin4"],
            ["lt_2cm_s", "2_to_10cm_s", "10_to_20cm_s", "20_to_40cm_s", "gt_40cm_s"],
            ["no", "yes"],
            ["A", "B", "C"],
            ["omitted", "rewarded"],
        ],
        "metadata": {
            "task_description": (
                "Head-fixed mouse CA1 calcium imaging during a 450 cm virtual linear-track task with "
                "hidden reward zones that switch across trials, days, and environments."
            ),
            "time_bin_size_ms": float(np.median(dt_all) * 1000.0),
            "temporal_alignment_event": "trial start / entry onto the 0 cm corridor position",
            "off_start": 0.0,
            "off_end": None,
            "source_paper": "A flexible hippocampal population code for experience relative to reward",
            "source_code_dir": "/app/code",
            "source_data_dir": str(data_dir),
            "source_data_format": "NWB",
            "neural_signal": "suite2p OASIS deconvolved calcium events exported in the NWB files",
            "neuron_filter": "manually curated iscell == 1, then exclude putative interneurons with corr(dff, speed) > 0.5",
            "trial_window": "frames from trial_start through the final corridor frame before teleport; positions clipped to [0, 450] cm",
            "reward_zone_bounds_cm": {key: list(val) for key, val in ZONE_BOUNDS_CM.items()},
            "trial_numbering": "0-indexed within session",
            "previous_trial_outcome_first_trial": 0,
            "session_info": [record["session_summary"] for record in session_records],
        },
    }
    return data


def subset_dataset(data: dict, keep_indices: list[int]):
    keep_indices = list(keep_indices)
    keep_subject_old = sorted(set(int(data["subject_idx"][idx]) for idx in keep_indices))
    subject_remap = {old: new for new, old in enumerate(keep_subject_old)}
    subset = {
        "neural": [data["neural"][idx] for idx in keep_indices],
        "input": [data["input"][idx] for idx in keep_indices],
        "output": [data["output"][idx] for idx in keep_indices],
        "subjects": [data["subjects"][old] for old in keep_subject_old],
        "subject_idx": np.array([subject_remap[int(data["subject_idx"][idx])] for idx in keep_indices], dtype=np.int64),
        "brain_regions": list(data["brain_regions"]),
        "brain_region_idx": [data["brain_region_idx"][idx] for idx in keep_indices],
        "input_names": list(data["input_names"]),
        "output_names": list(data["output_names"]),
        "output_values": [list(values) for values in data["output_values"]],
        "metadata": dict(data["metadata"]),
    }
    subset["metadata"]["session_info"] = [data["metadata"]["session_info"][idx] for idx in keep_indices]
    subset["metadata"]["sample_definition"] = {
        "subject": SAMPLE_SUBJECT,
        "exp_days": sorted(SAMPLE_EXP_DAYS),
    }
    return subset


def build_sample_dataset(full_data: dict):
    keep_indices = [
        idx
        for idx, info in enumerate(full_data["metadata"]["session_info"])
        if info["subject"] == SAMPLE_SUBJECT and int(info["exp_day"]) in SAMPLE_EXP_DAYS
    ]
    if not keep_indices:
        keep_indices = list(range(min(8, len(full_data["neural"]))))
    return subset_dataset(full_data, keep_indices)


def save_pickle(path: Path, data: dict):
    with path.open("wb") as f:
        pickle.dump(data, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--mode", choices=["full", "sample", "both"], default="both")
    parser.add_argument("--full-output", type=Path, default=Path("/app/converted_data.pkl"))
    parser.add_argument("--sample-output", type=Path, default=Path("/app/sample_data.pkl"))
    args = parser.parse_args()

    full_data = None

    if args.mode in {"full", "both"}:
        print("Building full dataset")
        full_data = build_full_dataset(args.data_dir)
        summarize_dataset(full_data)
        save_pickle(args.full_output, full_data)
        print(f"Saved full dataset to {args.full_output}")

    if args.mode in {"sample", "both"}:
        if full_data is None:
            if args.full_output.exists():
                print(f"Loading existing full dataset from {args.full_output}")
                with args.full_output.open("rb") as f:
                    full_data = pickle.load(f)
            else:
                print("Full dataset not found on disk; rebuilding before making sample dataset")
                full_data = build_full_dataset(args.data_dir)
        sample_data = build_sample_dataset(full_data)
        print("Sample dataset summary")
        summarize_dataset(sample_data)
        save_pickle(args.sample_output, sample_data)
        print(f"Saved sample dataset to {args.sample_output}")


if __name__ == "__main__":
    main()
