#!/usr/bin/env python3

import argparse
import pickle
import re
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TRACK_START_CM = 0.0
TRACK_END_CM = 450.0
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
SWITCH_TRIAL = 30
LICK_ERROR_FRAC = 0.35  # Reference code threshold in glmUtils.get_timeseries_data.

ZONE_COORDS_CM = {
    "A": (80.0, 130.0),
    "B": (200.0, 250.0),
    "C": (320.0, 370.0),
}
ZONE_TO_INT = {"A": 0, "B": 1, "C": 2}
ENV_TO_INT = {"Env1": 0, "Env2": 1}

INPUT_NAMES = [
    "time_from_trial_start_s",
    "environment",
    "trial_number",
    "previous_trial_outcome",
]

OUTPUT_NAMES = [
    "distance_to_reward_zone_bin",
    "absolute_position_bin",
    "speed_bin",
    "lick",
    "reward_zone_location",
    "reward_outcome",
]

OUTPUT_VALUES = [
    ["lt_-50", "minus50_to_minus10", "minus10_to_lt0", "in_zone", "gt0_to_10", "gt10_to_50", "gt50"],
    ["bin0", "bin1", "bin2", "bin3", "bin4"],
    ["lt_2", "2_to_10", "10_to_20", "20_to_40", "gt_40"],
    ["no", "yes"],
    ["A", "B", "C"],
    ["no", "yes"],
]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert NWB sessions into decoder-ready trial data.")
    parser.add_argument("outpicklefile", type=str, help="Output pickle path.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only 2 sessions for testing.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing plots for up to 2 sessions as processing_<session_id>.png.",
    )
    return parser.parse_args()


def get_session_files(sample: bool) -> list[Path]:
    files = sorted(Path("data").glob("sub-*/sub-*_behavior+ophys.nwb"))
    if sample:
        return files[:2]
    return files


def decode_h5_scalar(x):
    if isinstance(x, bytes):
        return x.decode()
    if hasattr(x, "shape") and x.shape == ():
        value = x[()]
        if isinstance(value, bytes):
            return value.decode()
        return value
    return x


def parse_scene(identifier: str) -> dict:
    scene = identifier.rstrip("/").split("/")[-1]

    match = re.fullmatch(r"(Env[12])_Location([ABC])", scene)
    if match:
        env, zone = match.groups()
        return {
            "scene": scene,
            "pre_env": env,
            "post_env": env,
            "pre_zone": zone,
            "post_zone": zone,
            "switch": False,
        }

    match = re.fullmatch(r"(Env[12])_Location([ABC])_to_([ABC])", scene)
    if match:
        env, pre_zone, post_zone = match.groups()
        return {
            "scene": scene,
            "pre_env": env,
            "post_env": env,
            "pre_zone": pre_zone,
            "post_zone": post_zone,
            "switch": True,
        }

    match = re.fullmatch(r"(Env[12])_([ABC])_to_(Env[12])_([ABC])", scene)
    if match:
        pre_env, pre_zone, post_env, post_zone = match.groups()
        return {
            "scene": scene,
            "pre_env": pre_env,
            "post_env": post_env,
            "pre_zone": pre_zone,
            "post_zone": post_zone,
            "switch": True,
        }

    raise ValueError(f"Unrecognized scene format: {scene}")


def zone_for_trial(scene_info: dict, trial_idx: int) -> tuple[str, str]:
    if scene_info["switch"] and trial_idx >= SWITCH_TRIAL:
        return scene_info["post_env"], scene_info["post_zone"]
    return scene_info["pre_env"], scene_info["pre_zone"]


def reconstruct_trials(trial_start: np.ndarray, teleport: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start > 0.5)
    teleports = np.flatnonzero(teleport > 0.5)
    trials = []
    tp_ptr = 0

    for start in starts:
        while tp_ptr < len(teleports) and teleports[tp_ptr] <= start:
            tp_ptr += 1
        if tp_ptr >= len(teleports):
            break
        stop = teleports[tp_ptr]
        if stop > start:
            trials.append((int(start), int(stop)))
        tp_ptr += 1

    return trials


def is_bad_lick_trial(lick_trial: np.ndarray) -> bool:
    if lick_trial.size == 0:
        return True
    return float(np.mean(lick_trial > 2)) > LICK_ERROR_FRAC


def reward_outcomes_from_timestamps(reward_times: np.ndarray, trial_times: np.ndarray, trials: list[tuple[int, int]]) -> np.ndarray:
    outcomes = np.zeros(len(trials), dtype=np.int8)
    reward_ptr = 0
    for trial_idx, (start, stop) in enumerate(trials):
        start_t = trial_times[start]
        stop_t = trial_times[stop]
        while reward_ptr < len(reward_times) and reward_times[reward_ptr] < start_t:
            reward_ptr += 1
        outcomes[trial_idx] = int(reward_ptr < len(reward_times) and reward_times[reward_ptr] < stop_t)
    return outcomes


def discretize_distance_to_zone(position_cm: np.ndarray, zone_start: float, zone_end: float) -> np.ndarray:
    distance = np.where(
        position_cm < zone_start,
        position_cm - zone_start,
        np.where(position_cm > zone_end, position_cm - zone_end, 0.0),
    )
    bins = np.full(distance.shape, 6, dtype=np.int16)
    bins[distance < -50.0] = 0
    bins[(distance >= -50.0) & (distance < -10.0)] = 1
    bins[(distance >= -10.0) & (distance < 0.0)] = 2
    bins[distance == 0.0] = 3
    bins[(distance > 0.0) & (distance <= 10.0)] = 4
    bins[(distance > 10.0) & (distance <= 50.0)] = 5
    return bins


def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, TRACK_START_CM, np.nextafter(TRACK_END_CM, TRACK_START_CM))
    edges = np.linspace(TRACK_START_CM, TRACK_END_CM, 6)
    return np.digitize(clipped, edges[1:-1], right=False).astype(np.int16)


def discretize_speed(speed_cm_s: np.ndarray) -> np.ndarray:
    bins = np.full(speed_cm_s.shape, 4, dtype=np.int16)
    bins[speed_cm_s < 2.0] = 0
    bins[(speed_cm_s >= 2.0) & (speed_cm_s < 10.0)] = 1
    bins[(speed_cm_s >= 10.0) & (speed_cm_s < 20.0)] = 2
    bins[(speed_cm_s >= 20.0) & (speed_cm_s <= 40.0)] = 3
    return bins


def binarize_licks(lick_trial: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(lick_trial), 0, 1).astype(np.int16)


def make_processing_plot(
    session_label: str,
    savedir: Path,
    neural_trials: list[np.ndarray],
    input_trials: list[np.ndarray],
    output_trials: list[np.ndarray],
    raw_examples: list[dict],
):
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), constrained_layout=True)

    trial0 = 0
    ex0 = raw_examples[0]
    t0 = input_trials[trial0][0]
    axes[0, 0].plot(t0, ex0["position"], color="black", lw=1.5)
    axes[0, 0].axhspan(ex0["zone_start"], ex0["zone_end"], color="gold", alpha=0.25)
    axes[0, 0].set_title(f"{session_label} trial {ex0['trial_index']} position")
    axes[0, 0].set_ylabel("Position (cm)")

    axes[1, 0].plot(t0, ex0["speed"], color="tab:blue", lw=1.2)
    axes[1, 0].set_ylabel("Speed (cm/s)")

    axes[2, 0].step(t0, ex0["lick"], where="post", color="tab:red")
    axes[2, 0].set_ylabel("Lick")

    sample_neurons = min(60, neural_trials[trial0].shape[0])
    axes[3, 0].imshow(
        neural_trials[trial0][:sample_neurons],
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
    )
    axes[3, 0].set_ylabel("Neuron")
    axes[3, 0].set_xlabel("Frame")

    trial1 = min(1, len(raw_examples) - 1)
    ex1 = raw_examples[trial1]
    t1 = input_trials[trial1][0]
    axes[0, 1].plot(t1, ex1["position"], color="black", lw=1.5, label="position")
    axes[0, 1].axhspan(ex1["zone_start"], ex1["zone_end"], color="gold", alpha=0.25, label="reward zone")
    axes[0, 1].set_title(f"{session_label} trial {ex1['trial_index']} outputs")
    axes[0, 1].set_ylabel("Position (cm)")

    axes[1, 1].step(t1, output_trials[trial1][0], where="post", label="dist bin")
    axes[1, 1].step(t1, output_trials[trial1][1], where="post", label="pos bin")
    axes[1, 1].step(t1, output_trials[trial1][2], where="post", label="speed bin")
    axes[1, 1].legend(loc="upper right", fontsize=8)
    axes[1, 1].set_ylabel("Binned outputs")

    axes[2, 1].step(t1, output_trials[trial1][3], where="post", label="lick")
    axes[2, 1].step(t1, output_trials[trial1][4], where="post", label="zone")
    axes[2, 1].step(t1, output_trials[trial1][5], where="post", label="reward")
    axes[2, 1].legend(loc="upper right", fontsize=8)
    axes[2, 1].set_ylabel("Discrete outputs")

    durations = np.array([trial.shape[1] for trial in neural_trials], dtype=float) / FRAME_RATE_HZ
    axes[3, 1].hist(durations, bins=20, color="0.3")
    axes[3, 1].set_xlabel("Trial duration (s)")
    axes[3, 1].set_ylabel("Count")

    plot_path = savedir / f"processing_{session_label}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def load_session(path: Path, show_processing: bool) -> tuple[dict, dict]:
    with h5py.File(path, "r") as handle:
        identifier = decode_h5_scalar(handle["identifier"])
        subject = decode_h5_scalar(handle["general/subject/subject_id"])
        session_id = decode_h5_scalar(handle["general/session_id"])
        region = decode_h5_scalar(handle["general/optophysiology/ImagingPlane/location"])
        scene_info = parse_scene(identifier)

        behavior = handle["processing/behavior/BehavioralTimeSeries"]
        position = behavior["position/data"][:].astype(np.float32)
        position_t = behavior["position/timestamps"][:].astype(np.float64)
        environment = behavior["environment/data"][:].astype(np.float32)
        lick = behavior["lick/data"][:].astype(np.float32)
        speed = behavior["speed/data"][:].astype(np.float32)
        trial_start = behavior["trial_start/data"][:].astype(np.float32)
        teleport = behavior["teleport/data"][:].astype(np.float32)
        reward_times = behavior["Reward/timestamps"][:].astype(np.float64)

        segmentation = handle["processing/ophys/ImageSegmentation/PlaneSegmentation"]
        iscell = segmentation["iscell"][:]
        plane_idx = segmentation["planeIdx"][:].astype(np.int16)
        curated_idx = np.flatnonzero(iscell[:, 0] > 0.5)
        brain_region_idx = np.zeros(curated_idx.size, dtype=np.int16)

        deconv_group = handle["processing/ophys/Deconvolved"]
        plane_keys = sorted(deconv_group.keys(), key=lambda key: int(key.replace("plane", "")))
        if len(plane_keys) == 1:
            deconvolved = np.asarray(deconv_group[plane_keys[0]]["data"][:, curated_idx], dtype=np.float16)
        else:
            n_frames = deconv_group[plane_keys[0]]["data"].shape[0]
            n_rois = plane_idx.shape[0]
            all_deconvolved = np.empty((n_frames, n_rois), dtype=np.float16)
            for plane_key in plane_keys:
                plane_number = int(plane_key.replace("plane", ""))
                cols = np.flatnonzero(plane_idx == plane_number)
                plane_data = np.asarray(deconv_group[plane_key]["data"], dtype=np.float16)
                if plane_data.shape[1] != cols.size:
                    raise ValueError(
                        f"{path.name}: {plane_key} has {plane_data.shape[1]} ROIs but planeIdx maps {cols.size}"
                    )
                all_deconvolved[:, cols] = plane_data
            deconvolved = all_deconvolved[:, curated_idx]

    trials = reconstruct_trials(trial_start, teleport)
    reward_outcomes = reward_outcomes_from_timestamps(reward_times, position_t, trials)

    neural_trials = []
    input_trials = []
    output_trials = []
    raw_examples = []
    dropped_bad_lick = 0

    for trial_idx, (start, stop) in enumerate(trials):
        if stop <= start:
            continue

        pos_trial = position[start:stop]
        speed_trial = speed[start:stop]
        lick_trial = lick[start:stop]
        time_trial = position_t[start:stop] - position_t[start]

        if pos_trial.size < 2:
            continue
        if is_bad_lick_trial(lick_trial):
            dropped_bad_lick += 1
            continue

        env_name, zone_name = zone_for_trial(scene_info, trial_idx)
        zone_start, zone_end = ZONE_COORDS_CM[zone_name]
        env_code = float(ENV_TO_INT[env_name])
        zone_code = int(ZONE_TO_INT[zone_name])
        reward_code = int(reward_outcomes[trial_idx])
        prev_reward = int(reward_outcomes[trial_idx - 1]) if trial_idx > 0 else 0

        neural_trial = deconvolved[start:stop].T

        dist_bin = discretize_distance_to_zone(pos_trial, zone_start, zone_end)
        pos_bin = discretize_absolute_position(pos_trial)
        speed_bin = discretize_speed(speed_trial)
        lick_bin = binarize_licks(lick_trial)

        n_time = neural_trial.shape[1]
        input_trial = np.vstack(
            [
                time_trial.astype(np.float32),
                np.full(n_time, env_code, dtype=np.float32),
                np.full(n_time, float(trial_idx), dtype=np.float32),
                np.full(n_time, float(prev_reward), dtype=np.float32),
            ]
        )
        output_trial = np.vstack(
            [
                dist_bin,
                pos_bin,
                speed_bin,
                lick_bin,
                np.full(n_time, zone_code, dtype=np.int16),
                np.full(n_time, reward_code, dtype=np.int16),
            ]
        )

        neural_trials.append(neural_trial)
        input_trials.append(input_trial)
        output_trials.append(output_trial)

        if show_processing and len(raw_examples) < 2:
            raw_examples.append(
                {
                    "trial_index": trial_idx,
                    "position": pos_trial,
                    "speed": speed_trial,
                    "lick": lick_bin,
                    "zone_start": zone_start,
                    "zone_end": zone_end,
                }
            )

    session_label = f"sub-{subject}_ses-{session_id}"
    if show_processing and neural_trials:
        make_processing_plot(
            session_label=session_label,
            savedir=Path("."),
            neural_trials=neural_trials,
            input_trials=input_trials,
            output_trials=output_trials,
            raw_examples=raw_examples or [
                {
                    "trial_index": 0,
                    "position": position[trials[0][0]:trials[0][1]],
                    "speed": speed[trials[0][0]:trials[0][1]],
                    "lick": binarize_licks(lick[trials[0][0]:trials[0][1]]),
                    "zone_start": ZONE_COORDS_CM[zone_for_trial(scene_info, 0)[1]][0],
                    "zone_end": ZONE_COORDS_CM[zone_for_trial(scene_info, 0)[1]][1],
                }
            ],
        )

    stats = {
        "path": str(path),
        "subject": subject,
        "session_id": session_id,
        "scene": scene_info["scene"],
        "n_trials_raw": len(trials),
        "n_trials_kept": len(neural_trials),
        "n_neurons_kept": int(curated_idx.size),
        "n_dropped_bad_lick": dropped_bad_lick,
        "brain_region": region,
    }

    session_data = {
        "subject": subject,
        "brain_region": region,
        "neural_trials": neural_trials,
        "input_trials": input_trials,
        "output_trials": output_trials,
        "stats": stats,
    }
    return session_data, stats


def build_dataset(session_files: list[Path], show_processing: bool) -> tuple[dict, list[dict]]:
    data = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": [],
        "subject_idx": None,
        "brain_regions": [],
        "brain_region_idx": [],
        "input_names": list(INPUT_NAMES),
        "output_names": list(OUTPUT_NAMES),
        "output_values": list(OUTPUT_VALUES),
        "metadata": {
            "task_description": (
                "CA1 two-photon imaging during virtual linear-track navigation with hidden reward zones; "
                "decoder predicts position, reward-zone-relative distance, movement, lick, reward-zone identity, "
                "and reward outcome from trial-aligned neural activity."
            ),
            "time_bin_size": TIME_BIN_MS,
            "temporal_alignment_event": "trial start",
            "off_start": 0.0,
            "off_end": None,
            "frame_rate_hz": FRAME_RATE_HZ,
            "track_length_cm": TRACK_END_CM,
            "reward_zone_coordinates_cm": ZONE_COORDS_CM,
            "switch_trial_index": SWITCH_TRIAL,
            "lick_sensor_error_fraction_threshold": LICK_ERROR_FRAC,
            "neural_signal": "NWB exported deconvolved calcium activity",
            "neuron_filter": "Suite2p iscell[:,0] == 1",
            "notes": [
                "Trials span trial_start to teleport onset, excluding teleport frames.",
                "Per-trial constants are repeated across timepoints.",
                "Reward-zone identity is parsed from the NWB identifier scene string.",
            ],
            "session_info": [],
        },
    }

    subject_to_idx = {}
    region_to_idx = {}
    subject_idx = []
    session_stats = []

    t0 = time.perf_counter()
    for session_number, path in enumerate(session_files):
        session_t0 = time.perf_counter()
        session_data, stats = load_session(
            path=path,
            show_processing=show_processing and session_number < 2,
        )
        elapsed = time.perf_counter() - session_t0
        print(
            f"[{session_number + 1:03d}/{len(session_files):03d}] "
            f"{stats['subject']} ses-{stats['session_id']} "
            f"scene={stats['scene']} trials={stats['n_trials_kept']}/{stats['n_trials_raw']} "
            f"neurons={stats['n_neurons_kept']} dropped_bad_lick={stats['n_dropped_bad_lick']} "
            f"time={elapsed:.2f}s"
        )

        if len(session_data["neural_trials"]) < 2:
            print(f"  skipping {path.name}: fewer than 2 usable trials after QC")
            continue

        subject = session_data["subject"]
        region = session_data["brain_region"]
        if subject not in subject_to_idx:
            subject_to_idx[subject] = len(data["subjects"])
            data["subjects"].append(subject)
        if region not in region_to_idx:
            region_to_idx[region] = len(data["brain_regions"])
            data["brain_regions"].append(region)

        data["neural"].append(session_data["neural_trials"])
        data["input"].append(session_data["input_trials"])
        data["output"].append(session_data["output_trials"])
        data["brain_region_idx"].append(
            np.full(session_data["stats"]["n_neurons_kept"], region_to_idx[region], dtype=np.int16)
        )
        subject_idx.append(subject_to_idx[subject])
        data["metadata"]["session_info"].append(stats)
        session_stats.append(stats)

    data["subject_idx"] = np.asarray(subject_idx, dtype=np.int16)
    total_elapsed = time.perf_counter() - t0
    print(f"Processed {len(data['neural'])} sessions in {total_elapsed:.2f}s")
    return data, session_stats


def main():
    args = parse_args()
    session_files = get_session_files(sample=args.sample)
    if not session_files:
        raise FileNotFoundError("No NWB session files found under data/")

    data, session_stats = build_dataset(session_files, show_processing=args.show_processing)

    out_path = Path(args.outpicklefile)
    with out_path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    total_trials = sum(len(session_trials) for session_trials in data["neural"])
    total_neurons = int(sum(brain_idx.shape[0] for brain_idx in data["brain_region_idx"]))
    print(f"Saved {out_path}")
    print(
        f"Summary: sessions={len(data['neural'])} subjects={len(data['subjects'])} "
        f"trials={total_trials} neurons={total_neurons}"
    )
    if session_stats:
        mean_trials = np.mean([s["n_trials_kept"] for s in session_stats])
        mean_neurons = np.mean([s["n_neurons_kept"] for s in session_stats])
        print(f"Mean trials/session={mean_trials:.2f} mean neurons/session={mean_neurons:.2f}")


if __name__ == "__main__":
    main()
