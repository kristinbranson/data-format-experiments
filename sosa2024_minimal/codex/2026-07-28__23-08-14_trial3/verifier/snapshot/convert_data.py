#!/usr/bin/env python3

import argparse
import collections
import json
import os
import pickle
import re
from pathlib import Path

import h5py
import numpy as np


ZONE_BOUNDS_CM = {
    "A": (80.0, 130.0),
    "B": (200.0, 250.0),
    "C": (320.0, 370.0),
}
ZONE_TO_INDEX = {"A": 0, "B": 1, "C": 2}
ZONE_CENTERS_CM = {
    label: 0.5 * (bounds[0] + bounds[1]) for label, bounds in ZONE_BOUNDS_CM.items()
}

EXPECTED_SAMPLE_BASENAMES = [
    "sub-m3_ses-01_behavior+ophys.nwb",
    "sub-m3_ses-03_behavior+ophys.nwb",
    "sub-m3_ses-08_behavior+ophys.nwb",
    "sub-m11_ses-03_behavior+ophys.nwb",
    "sub-m12_ses-01_behavior+ophys.nwb",
    "sub-m17_ses-01_behavior+ophys.nwb",
    "sub-m17_ses-08_behavior+ophys.nwb",
    "sub-m18_ses-08_behavior+ophys.nwb",
]

INPUT_NAMES = [
    "time_from_trial_start_sec",
    "environment_type",
    "trial_number",
    "previous_trial_outcome",
]

OUTPUT_NAMES = [
    "distance_to_reward_zone",
    "absolute_position",
    "speed",
    "lick",
    "reward_zone_location",
    "reward_outcome",
]

OUTPUT_VALUES = [
    [
        "< -50 cm",
        "-50 to -10 cm",
        "-10 cm to < 0 cm",
        "0 cm",
        "> 0 cm to +10 cm",
        "+10 to +50 cm",
        "> +50 cm",
    ],
    [
        "0 to <90 cm",
        "90 to <180 cm",
        "180 to <270 cm",
        "270 to <360 cm",
        "360 to 450 cm",
    ],
    [
        "< 2 cm/s",
        "2 to <10 cm/s",
        "10 to <20 cm/s",
        "20 to <40 cm/s",
        ">= 40 cm/s",
    ],
    ["no", "yes"],
    ["A", "B", "C"],
    ["no", "yes"],
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Sosa, Plitt, Giocomo 2024 NWB sessions into decoder format."
    )
    parser.add_argument(
        "--data-dir",
        default="/app/data",
        help="Directory containing sub-*/sub-*_behavior+ophys.nwb files.",
    )
    parser.add_argument(
        "--full-out",
        default="/app/converted_data.pkl",
        help="Output pickle for the full converted dataset.",
    )
    parser.add_argument(
        "--sample-out",
        default="/app/sample_data.pkl",
        help="Output pickle for a smaller representative sample dataset.",
    )
    parser.add_argument(
        "--lick-error-threshold",
        type=float,
        default=0.35,
        help="Drop trials where more than this fraction of frames have lick count > 2.",
    )
    return parser.parse_args()


def nwb_files(data_dir: str):
    paths = sorted(Path(data_dir).glob("sub-*/sub-*_behavior+ophys.nwb"))
    if not paths:
        raise FileNotFoundError(f"No NWB files found under {data_dir}")
    return paths


def parse_subject_and_day(path: Path):
    subject = path.parent.name.replace("sub-", "")
    match = re.search(r"ses-(\d+)", path.name)
    if match is None:
        raise ValueError(f"Could not parse session/day from {path.name}")
    return subject, int(match.group(1))


def mode_or_none(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    counts = collections.Counter(values)
    return counts.most_common(1)[0][0]


def infer_zone_from_position(zone_position_cm):
    if not np.isfinite(zone_position_cm):
        return None
    return min(
        ZONE_CENTERS_CM,
        key=lambda label: abs(zone_position_cm - ZONE_CENTERS_CM[label]),
    )


def infer_trial_zone_positions(pos, reward_zone_signal, times, reward_times, trial_starts, teleports):
    observed_labels = []
    observed_positions = []
    for start, stop in zip(trial_starts, teleports):
        pos_trial = pos[start:stop]
        zone_signal_trial = reward_zone_signal[start:stop]
        zone_frames = np.flatnonzero(zone_signal_trial > 0)
        zone_position_cm = np.nan
        if zone_frames.size > 0:
            zone_position_cm = float(np.nanmedian(pos_trial[zone_frames]))
        else:
            in_trial_reward = reward_times[(reward_times >= times[start]) & (reward_times < times[stop])]
            if in_trial_reward.size > 0:
                reward_idx = np.searchsorted(times, in_trial_reward[0], side="left")
                reward_idx = min(max(reward_idx, start), stop - 1)
                zone_position_cm = float(pos[reward_idx])
        observed_positions.append(zone_position_cm)
        observed_labels.append(infer_zone_from_position(zone_position_cm))
    return observed_labels, observed_positions


def infer_constant_or_switch_schedule(observed_values, switch_trial=30):
    ntrials = len(observed_values)
    pre = mode_or_none(observed_values[: min(switch_trial, ntrials)])
    post = mode_or_none(observed_values[switch_trial:]) if ntrials > switch_trial else None
    overall = mode_or_none(observed_values)

    is_switch = pre is not None and post is not None and pre != post
    filled = []
    if is_switch:
        for trial_idx in range(ntrials):
            filled.append(pre if trial_idx < switch_trial else post)
    else:
        filled = [overall for _ in range(ntrials)]

    return {
        "observed": list(observed_values),
        "filled": filled,
        "mode_pre": pre,
        "mode_post": post,
        "mode_all": overall,
        "switch_trial": switch_trial if is_switch else None,
        "is_switch": bool(is_switch),
    }


def reward_outcomes_per_trial(reward_times, times, trial_starts, teleports):
    outcomes = []
    for start, stop in zip(trial_starts, teleports):
        has_reward = np.any((reward_times >= times[start]) & (reward_times < times[stop]))
        outcomes.append(int(has_reward))
    return outcomes


def drop_lick_error_trials(lick, trial_starts, teleports, threshold):
    drop_mask = np.zeros(len(trial_starts), dtype=bool)
    error_fraction = np.zeros(len(trial_starts), dtype=np.float32)
    for idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
        lick_trial = lick[start:stop]
        frac = float(np.mean(lick_trial > 2))
        error_fraction[idx] = frac
        if frac > threshold:
            drop_mask[idx] = True
    return drop_mask, error_fraction


def signed_distance_to_zone(pos_cm, zone_bounds_cm):
    zone_start, zone_end = zone_bounds_cm
    dist = np.zeros_like(pos_cm, dtype=np.float32)
    before = pos_cm < zone_start
    after = pos_cm > zone_end
    dist[before] = pos_cm[before] - zone_start
    dist[after] = pos_cm[after] - zone_end
    return dist


def bin_distance_to_zone(dist_cm):
    out = np.zeros(dist_cm.shape, dtype=np.int64)
    out[dist_cm < -50.0] = 0
    out[(dist_cm >= -50.0) & (dist_cm < -10.0)] = 1
    out[(dist_cm >= -10.0) & (dist_cm < 0.0)] = 2
    out[dist_cm == 0.0] = 3
    out[(dist_cm > 0.0) & (dist_cm <= 10.0)] = 4
    out[(dist_cm > 10.0) & (dist_cm <= 50.0)] = 5
    out[dist_cm > 50.0] = 6
    return out


def bin_absolute_position(pos_cm):
    pos_clipped = np.clip(pos_cm, 0.0, 450.0 - 1e-6)
    return np.minimum((pos_clipped / 90.0).astype(np.int64), 4)


def bin_speed(speed_cm_s):
    out = np.zeros(speed_cm_s.shape, dtype=np.int64)
    out[(speed_cm_s >= 2.0) & (speed_cm_s < 10.0)] = 1
    out[(speed_cm_s >= 10.0) & (speed_cm_s < 20.0)] = 2
    out[(speed_cm_s >= 20.0) & (speed_cm_s < 40.0)] = 3
    out[speed_cm_s >= 40.0] = 4
    return out


def session_summary_record(
    path,
    subject,
    day,
    ntrials_raw,
    ntrials_kept,
    ntrials_dropped_lick_error,
    nneurons,
    dt_sec,
    zone_schedule,
    env_schedule,
):
    return {
        "file": path.name,
        "subject": subject,
        "day": day,
        "ntrials_raw": ntrials_raw,
        "ntrials_kept": ntrials_kept,
        "ntrials_dropped_lick_error": ntrials_dropped_lick_error,
        "nneurons": nneurons,
        "time_bin_size_sec": dt_sec,
        "reward_zone_mode_pre": zone_schedule["mode_pre"],
        "reward_zone_mode_post": zone_schedule["mode_post"],
        "reward_zone_mode_all": zone_schedule["mode_all"],
        "reward_zone_switch_trial": zone_schedule["switch_trial"],
        "environment_mode_pre": env_schedule["mode_pre"],
        "environment_mode_post": env_schedule["mode_post"],
        "environment_mode_all": env_schedule["mode_all"],
        "environment_switch_trial": env_schedule["switch_trial"],
    }


def convert_session(path: Path, lick_error_threshold: float):
    subject, day = parse_subject_and_day(path)
    with h5py.File(path, "r") as f:
        beh = f["processing/behavior/BehavioralTimeSeries"]

        times = beh["position"]["timestamps"][()].astype(np.float64)
        pos = beh["position"]["data"][()].astype(np.float32)
        speed = beh["speed"]["data"][()].astype(np.float32)
        lick = beh["lick"]["data"][()].astype(np.float32)
        reward_zone_signal = beh["reward_zone"]["data"][()].astype(np.float32)
        environment = beh["environment"]["data"][()].astype(np.float32)
        trial_number = beh["trial number"]["data"][()].astype(np.float32)
        trial_start_signal = beh["trial_start"]["data"][()]
        teleport_signal = beh["teleport"]["data"][()]
        reward_times = beh["Reward"]["timestamps"][()].astype(np.float64)

        trial_starts = np.flatnonzero(trial_start_signal > 0)
        teleports = np.flatnonzero(teleport_signal > 0)
        if trial_starts.size != teleports.size:
            raise ValueError(
                f"{path.name}: trial starts ({trial_starts.size}) and teleports ({teleports.size}) do not match"
            )

        dt_sec = float(np.median(np.diff(times)))
        observed_zone_labels, observed_zone_positions = infer_trial_zone_positions(
            pos=pos,
            reward_zone_signal=reward_zone_signal,
            times=times,
            reward_times=reward_times,
            trial_starts=trial_starts,
            teleports=teleports,
        )
        observed_env = []
        for start, stop in zip(trial_starts, teleports):
            env_trial = environment[start:stop]
            env_trial = env_trial[env_trial >= 0]
            observed_env.append(int(np.rint(np.nanmedian(env_trial))) if env_trial.size else None)

        zone_schedule = infer_constant_or_switch_schedule(observed_zone_labels)
        env_schedule = infer_constant_or_switch_schedule(observed_env)
        reward_outcomes = reward_outcomes_per_trial(reward_times, times, trial_starts, teleports)
        previous_reward_outcomes = [0] + reward_outcomes[:-1]

        drop_mask, lick_error_fraction = drop_lick_error_trials(
            lick=lick,
            trial_starts=trial_starts,
            teleports=teleports,
            threshold=lick_error_threshold,
        )

        deconv = f["processing/ophys/Deconvolved/plane0/data"]
        rois = f["processing/ophys/Fluorescence/plane0/rois"][()]
        iscell = f["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][()][:, 0].astype(bool)
        selected_mask = iscell[rois]
        selected_indices = np.flatnonzero(selected_mask)
        selected_rois = rois[selected_mask]
        plane_idx = f["processing/ophys/ImageSegmentation/PlaneSegmentation/planeIdx"][()][selected_rois]

        neural_trials = []
        input_trials = []
        output_trials = []

        kept_trial_indices = []
        dropped_trials = []
        for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
            if drop_mask[trial_idx]:
                dropped_trials.append(trial_idx)
                continue

            T = int(stop - start)
            if T <= 0:
                dropped_trials.append(trial_idx)
                continue

            zone_label = zone_schedule["filled"][trial_idx]
            env_label = env_schedule["filled"][trial_idx]
            if zone_label is None or env_label is None:
                dropped_trials.append(trial_idx)
                continue

            pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
            speed_trial = speed[start:stop].astype(np.float32)
            lick_trial = (lick[start:stop] > 0).astype(np.int64)
            time_trial = (times[start:stop] - times[start]).astype(np.float32)

            neural = deconv[start:stop, selected_indices].T.astype(np.float32)
            trial_number_values = trial_number[start:stop]
            trial_number_values = trial_number_values[trial_number_values >= 0]
            trial_number_scalar = (
                float(np.nanmedian(trial_number_values))
                if trial_number_values.size
                else float(trial_idx)
            )

            zone_bounds_cm = ZONE_BOUNDS_CM[zone_label]
            dist_trial = signed_distance_to_zone(pos_trial, zone_bounds_cm)

            input_trial = np.vstack(
                [
                    time_trial,
                    np.full(T, env_label, dtype=np.float32),
                    np.full(T, trial_number_scalar, dtype=np.float32),
                    np.full(T, float(previous_reward_outcomes[trial_idx]), dtype=np.float32),
                ]
            )
            output_trial = np.vstack(
                [
                    bin_distance_to_zone(dist_trial),
                    bin_absolute_position(pos_trial),
                    bin_speed(speed_trial),
                    lick_trial,
                    np.full(T, ZONE_TO_INDEX[zone_label], dtype=np.int64),
                    np.full(T, reward_outcomes[trial_idx], dtype=np.int64),
                ]
            )

            neural_trials.append(neural)
            input_trials.append(input_trial.astype(np.float32))
            output_trials.append(output_trial.astype(np.int64))
            kept_trial_indices.append(trial_idx)

    if not neural_trials:
        raise ValueError(f"{path.name}: no valid trials remained after filtering")

    session_info = session_summary_record(
        path=path,
        subject=subject,
        day=day,
        ntrials_raw=len(trial_starts),
        ntrials_kept=len(neural_trials),
        ntrials_dropped_lick_error=int(drop_mask.sum()),
        nneurons=int(selected_mask.sum()),
        dt_sec=dt_sec,
        zone_schedule=zone_schedule,
        env_schedule=env_schedule,
    )
    session_info["trial_indices_kept"] = kept_trial_indices
    session_info["trial_indices_dropped"] = dropped_trials
    session_info["observed_zone_positions_cm"] = observed_zone_positions
    session_info["observed_zone_labels"] = observed_zone_labels
    session_info["observed_environment"] = observed_env
    session_info["reward_outcomes_all_trials"] = reward_outcomes
    session_info["previous_reward_outcomes_all_trials"] = previous_reward_outcomes
    session_info["lick_error_fraction_per_trial"] = lick_error_fraction.tolist()
    session_info["available_roi_series"] = "processing/ophys/Deconvolved/plane0"
    session_info["available_response_series_roi_count"] = int(len(rois))
    session_info["selected_roi_count_after_iscell"] = int(selected_mask.sum())
    session_info["selected_plane_indices"] = sorted(np.unique(plane_idx).astype(int).tolist())
    session_info["lick_error_threshold_fraction"] = float(lick_error_threshold)

    return {
        "subject": subject,
        "day": day,
        "neural": neural_trials,
        "input": input_trials,
        "output": output_trials,
        "brain_region_idx": np.zeros(int(selected_mask.sum()), dtype=np.int64),
        "session_info": session_info,
    }


def build_dataset(session_records):
    subjects = sorted({record["subject"] for record in session_records})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}

    session_info = [record["session_info"] for record in session_records]
    time_bin_sizes_ms = [1000.0 * info["time_bin_size_sec"] for info in session_info]

    data = {
        "neural": [record["neural"] for record in session_records],
        "input": [record["input"] for record in session_records],
        "output": [record["output"] for record in session_records],
        "subjects": subjects,
        "subject_idx": np.asarray(
            [subject_to_idx[record["subject"]] for record in session_records],
            dtype=np.int64,
        ),
        "brain_regions": ["CA1"],
        "brain_region_idx": [record["brain_region_idx"] for record in session_records],
        "input_names": list(INPUT_NAMES),
        "output_names": list(OUTPUT_NAMES),
        "output_values": [list(values) for values in OUTPUT_VALUES],
        "metadata": {
            "task_description": (
                "Head-fixed mouse virtual-reality linear track task with hidden reward zones "
                "that switch across trials and environments; neural activity is deconvolved "
                "calcium events aligned to trial start."
            ),
            "time_bin_size": float(np.median(time_bin_sizes_ms)),
            "temporal_alignment_event": "trial start (entry to the start of the 450 cm linear track)",
            "off_start": 0.0,
            "off_end": None,
            "position_range_cm": [0.0, 450.0],
            "reward_zone_definitions_cm": {k: list(v) for k, v in ZONE_BOUNDS_CM.items()},
            "environment_definitions": {"0": "ENV1", "1": "ENV2"},
            "trial_window_definition": "Frames from trial_start inclusive to teleport exclusive.",
            "lick_sensor_error_threshold_fraction": float(
                session_records[0]["session_info"].get("lick_error_threshold_fraction", 0.35)
            ),
            "neural_signal": "Deconvolved calcium events from processing/ophys/Deconvolved/plane0",
            "roi_selection": (
                "ROIs referenced by processing/ophys/Fluorescence/plane0/rois and retained "
                "only if the corresponding PlaneSegmentation iscell flag is true."
            ),
            "notes": [
                "The NWB files do not expose NWB trials/units tables, so trials are reconstructed from framewise trial_start and teleport channels.",
                "Reward-zone labels are inferred from framewise reward_zone observations and filled with the dominant pre/post-switch label when omission trials lack direct observations.",
                "Multi-plane mice m17 and m18 include only the published plane0 response series in NWB; the converter uses the available response series rather than unlinked segmentation rows from the second plane.",
            ],
            "session_info": session_info,
        },
    }
    return data


def summarize_dataset(data):
    nsessions = len(data["neural"])
    ntrials = np.asarray([len(session) for session in data["neural"]], dtype=np.int64)
    nneurons = np.asarray(
        [session[0].shape[0] if session else 0 for session in data["neural"]],
        dtype=np.int64,
    )
    trial_lengths = np.asarray(
        [
            trial.shape[1]
            for session in data["neural"]
            for trial in session
        ],
        dtype=np.int64,
    )
    lick_drops = sum(
        info["ntrials_dropped_lick_error"] for info in data["metadata"]["session_info"]
    )
    zone_switch_sessions = sum(
        info["reward_zone_switch_trial"] is not None
        for info in data["metadata"]["session_info"]
    )
    env_switch_sessions = sum(
        info["environment_switch_trial"] is not None
        for info in data["metadata"]["session_info"]
    )

    summary = {
        "n_subjects": len(data["subjects"]),
        "n_sessions": int(nsessions),
        "n_trials": int(ntrials.sum()),
        "mean_trials_per_session": float(np.mean(ntrials)),
        "min_trials_per_session": int(np.min(ntrials)),
        "max_trials_per_session": int(np.max(ntrials)),
        "mean_neurons_per_session": float(np.mean(nneurons)),
        "min_neurons_per_session": int(np.min(nneurons)),
        "max_neurons_per_session": int(np.max(nneurons)),
        "mean_frames_per_trial": float(np.mean(trial_lengths)),
        "min_frames_per_trial": int(np.min(trial_lengths)),
        "max_frames_per_trial": int(np.max(trial_lengths)),
        "time_bin_size_ms": float(data["metadata"]["time_bin_size"]),
        "dropped_trials_lick_error": int(lick_drops),
        "reward_zone_switch_sessions": int(zone_switch_sessions),
        "environment_switch_sessions": int(env_switch_sessions),
    }
    return summary


def print_subject_schedules(data):
    per_subject = collections.defaultdict(list)
    for info in data["metadata"]["session_info"]:
        per_subject[info["subject"]].append(info)

    print("\nRecovered subject schedules:")
    for subject in sorted(per_subject):
        infos = sorted(per_subject[subject], key=lambda item: item["day"])
        parts = []
        for info in infos:
            zone = info["reward_zone_mode_all"]
            if info["reward_zone_switch_trial"] is not None:
                zone = f"{info['reward_zone_mode_pre']}->{info['reward_zone_mode_post']}"
            env = f"ENV{info['environment_mode_all'] + 1}"
            if info["environment_switch_trial"] is not None:
                env = f"ENV{info['environment_mode_pre'] + 1}->ENV{info['environment_mode_post'] + 1}"
            parts.append(f"d{info['day']:02d}:{env}:{zone}")
        print(f"  {subject}: {' | '.join(parts)}")


def select_sample_records(session_records):
    basename_to_record = {record["session_info"]["file"]: record for record in session_records}
    sample = [
        basename_to_record[basename]
        for basename in EXPECTED_SAMPLE_BASENAMES
        if basename in basename_to_record
    ]
    if len(sample) < 4:
        raise ValueError("Sample session selection returned too few sessions")
    return sample


def save_pickle(path, payload):
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def main():
    args = parse_args()
    files = nwb_files(args.data_dir)

    print(f"Found {len(files)} NWB sessions under {args.data_dir}")
    session_records = []
    for idx, path in enumerate(files, start=1):
        record = convert_session(path, lick_error_threshold=args.lick_error_threshold)
        session_records.append(record)
        info = record["session_info"]
        print(
            f"[{idx:03d}/{len(files):03d}] {path.name}: "
            f"{info['ntrials_kept']}/{info['ntrials_raw']} trials kept, "
            f"{info['nneurons']} neurons, "
            f"zone={info['reward_zone_mode_all'] if info['reward_zone_switch_trial'] is None else f'{info['reward_zone_mode_pre']}->{info['reward_zone_mode_post']}'}, "
            f"env={info['environment_mode_all'] if info['environment_switch_trial'] is None else f'{info['environment_mode_pre']}->{info['environment_mode_post']}'}"
        )

    full_data = build_dataset(session_records)
    full_summary = summarize_dataset(full_data)
    print("\nFull dataset summary:")
    print(json.dumps(full_summary, indent=2))
    print_subject_schedules(full_data)

    sample_records = select_sample_records(session_records)
    sample_data = build_dataset(sample_records)
    sample_summary = summarize_dataset(sample_data)
    print("\nSample dataset summary:")
    print(json.dumps(sample_summary, indent=2))

    save_pickle(args.full_out, full_data)
    save_pickle(args.sample_out, sample_data)
    print(f"\nSaved full dataset to {args.full_out}")
    print(f"Saved sample dataset to {args.sample_out}")


if __name__ == "__main__":
    main()
