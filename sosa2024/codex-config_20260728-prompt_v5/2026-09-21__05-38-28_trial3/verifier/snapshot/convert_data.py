#!/usr/bin/env python3

import argparse
import os
import pickle
import re
import time
from collections import Counter
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TRACK_LENGTH_CM = 450.0
TRACK_BIN_SIZE_CM = 10.0
SWITCH_TRIAL_INDEX = 30
LICK_QC_FRACTION = 0.35
DEFAULT_DT_SEC = 1.0 / 15.5078125
BEHAVIOR_TS_PATH = "processing/behavior/BehavioralTimeSeries"
OPHYS_TS_PATH = "processing/ophys"
BRAIN_REGION_NAME = "hippocampus, CA1"
REWARD_ZONE_COORDS = {
    "A": (80.0, 130.0),
    "B": (200.0, 250.0),
    "C": (320.0, 370.0),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert NWB sessions into decoder-ready pickle format."
    )
    parser.add_argument("outpicklefile", type=str, help="Output pickle path.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only 2 sessions.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing visualizations for up to 2 sessions.",
    )
    return parser.parse_args()


def parse_scene_reward_sequence(scene):
    match = re.search(r"Location([ABC])_to_([ABC])$", scene)
    if match:
        return [match.group(1), match.group(2)]

    match = re.search(r"Env[12]_([ABC])_to_Env[12]_([ABC])$", scene)
    if match:
        return [match.group(1), match.group(2)]

    match = re.search(r"Location([ABC])$", scene)
    if match:
        return [match.group(1)]

    raise ValueError(f"Unrecognized scene format: {scene}")


def reward_labels_for_trials(scene, n_trials):
    seq = parse_scene_reward_sequence(scene)
    if len(seq) == 1:
        return [seq[0]] * n_trials
    split = min(SWITCH_TRIAL_INDEX, n_trials)
    return [seq[0]] * split + [seq[1]] * max(0, n_trials - split)


def pick_processing_files(all_files):
    singles = [p for p in all_files if p.parent.name not in {"sub-m17", "sub-m18"}]
    multis = [p for p in all_files if p.parent.name in {"sub-m17", "sub-m18"}]
    picked = []
    if singles:
        picked.append(singles[0])
    if multis:
        picked.append(multis[0])
    if len(picked) < 2:
        for path in all_files:
            if path not in picked:
                picked.append(path)
            if len(picked) == 2:
                break
    return picked[:2]


def pick_sample_files(all_files):
    return pick_processing_files(all_files)


def read_str(dataset):
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode()
    if np.isscalar(value):
        return str(value)
    return str(value.item())


def load_session_raw(path):
    with h5py.File(path, "r") as f:
        scene = read_str(f["identifier"]).split("/")[-1]
        subject = read_str(f["general/subject/subject_id"])
        session_id = read_str(f["general/session_id"])

        behavior = {}
        for name in [
            "environment",
            "position",
            "speed",
            "lick",
            "reward_zone",
            "scanning",
            "trial number",
            "trial_start",
            "teleport",
            "autoreward",
        ]:
            behavior[name] = f[f"{BEHAVIOR_TS_PATH}/{name}/data"][:]
        behavior["timestamps"] = f[f"{BEHAVIOR_TS_PATH}/position/timestamps"][:]
        reward_timestamps = f[f"{BEHAVIOR_TS_PATH}/Reward/timestamps"][:]
        reward_amounts = f[f"{BEHAVIOR_TS_PATH}/Reward/data"][:]

        iscell = f[f"{OPHYS_TS_PATH}/ImageSegmentation/PlaneSegmentation/iscell"][:]
        plane_idx_all = f[f"{OPHYS_TS_PATH}/ImageSegmentation/PlaneSegmentation/planeIdx"][:]
        plane_series = sorted(f[f"{OPHYS_TS_PATH}/Deconvolved"].keys())
        neural_lens = [
            f[f"{OPHYS_TS_PATH}/Deconvolved/{series_name}/data"].shape[0]
            for series_name in plane_series
        ]
        neural_len = min(neural_lens)
        behavior_len = min(len(v) for v in behavior.values())
        common_len = min(neural_len, behavior_len)

        neural_planes = []
        plane_idx_curated = []
        for series_name in plane_series:
            response_group = f[f"{OPHYS_TS_PATH}/Deconvolved/{series_name}"]
            rois = response_group["rois"][:]
            response_plane_idx = plane_idx_all[rois]
            response_iscell = iscell[rois, 0].astype(np.int64) == 1
            response_data = response_group["data"][:common_len, :]
            neural_planes.append(response_data[:, response_iscell].astype(np.float32, copy=False))
            plane_idx_curated.append(response_plane_idx[response_iscell].astype(np.int64, copy=False))
        neural = np.concatenate(neural_planes, axis=1)
        plane_idx = np.concatenate(plane_idx_curated, axis=0)

        for key in behavior:
            behavior[key] = behavior[key][:common_len]

        reward_valid = reward_timestamps <= behavior["timestamps"][-1]
        reward_timestamps = reward_timestamps[reward_valid]
        reward_amounts = reward_amounts[reward_valid]
        reward_frame_idx = np.searchsorted(behavior["timestamps"], reward_timestamps, side="left")
        reward_frame_idx = np.clip(reward_frame_idx, 0, common_len - 1)

        dt_behavior = float(np.median(np.diff(behavior["timestamps"][: min(common_len, 1000)])))
        ophys_rate_attr = float(
            f[f"{OPHYS_TS_PATH}/Deconvolved/{plane_series[0]}/starting_time"].attrs["rate"]
        )

    return {
        "path": path,
        "subject": subject,
        "session_id": session_id,
        "scene": scene,
        "behavior": behavior,
        "reward_frame_idx": reward_frame_idx,
        "reward_amounts": reward_amounts,
        "neural": neural,
        "plane_idx": plane_idx,
        "n_rois_total": int(iscell.shape[0]),
        "n_cells_curated": int(neural.shape[1]),
        "dt_behavior_sec": dt_behavior,
        "ophys_rate_attr_hz": ophys_rate_attr,
        "trimmed_timepoints": int(neural_len - common_len),
        "is_multiplane": int(np.unique(plane_idx).size) > 1,
    }


def build_trial_bounds(behavior):
    starts = np.flatnonzero(behavior["trial_start"] > 0)
    teleports = np.flatnonzero(behavior["teleport"] > 0)
    bounds = []
    tele_ptr = 0
    for start in starts:
        while tele_ptr < len(teleports) and teleports[tele_ptr] <= start:
            tele_ptr += 1
        if tele_ptr >= len(teleports):
            break
        stop = teleports[tele_ptr]
        tele_ptr += 1
        if stop <= start:
            continue
        bounds.append((int(start), int(stop)))
    return bounds, starts, teleports


def trial_reward_outcomes(bounds, reward_frame_idx):
    outcomes = np.zeros(len(bounds), dtype=np.int64)
    reward_trials = []
    for i, (start, stop) in enumerate(bounds):
        has_reward = np.any((reward_frame_idx >= start) & (reward_frame_idx < stop))
        outcomes[i] = int(has_reward)
        reward_trials.append(np.flatnonzero((reward_frame_idx >= start) & (reward_frame_idx < stop)))
    return outcomes, reward_trials


def trial_environment_value(environment_slice):
    valid = environment_slice[environment_slice >= 0]
    if valid.size == 0:
        raise ValueError("No valid environment values within trial slice.")
    values, counts = np.unique(valid, return_counts=True)
    return float(values[np.argmax(counts)])


def signed_distance_to_zone(position_cm, zone_start, zone_end):
    dist = np.zeros_like(position_cm, dtype=np.float32)
    before = position_cm < zone_start
    after = position_cm > zone_end
    dist[before] = position_cm[before] - zone_start
    dist[after] = position_cm[after] - zone_end
    return dist


def discretize_distance(distance_cm):
    out = np.full(distance_cm.shape, 6, dtype=np.int64)
    out[distance_cm < -50.0] = 0
    out[(distance_cm >= -50.0) & (distance_cm < -10.0)] = 1
    out[(distance_cm >= -10.0) & (distance_cm < 0.0)] = 2
    out[distance_cm == 0.0] = 3
    out[(distance_cm > 0.0) & (distance_cm < 10.0)] = 4
    out[(distance_cm >= 10.0) & (distance_cm <= 50.0)] = 5
    out[distance_cm > 50.0] = 6
    return out


def discretize_absolute_position(position_cm):
    out = np.zeros(position_cm.shape, dtype=np.int64)
    out[(position_cm >= 90.0) & (position_cm < 180.0)] = 1
    out[(position_cm >= 180.0) & (position_cm < 270.0)] = 2
    out[(position_cm >= 270.0) & (position_cm <= 360.0)] = 3
    out[position_cm > 360.0] = 4
    return out


def discretize_speed(speed_cm_s):
    out = np.full(speed_cm_s.shape, 4, dtype=np.int64)
    out[speed_cm_s < 2.0] = 0
    out[(speed_cm_s >= 2.0) & (speed_cm_s < 10.0)] = 1
    out[(speed_cm_s >= 10.0) & (speed_cm_s < 20.0)] = 2
    out[(speed_cm_s >= 20.0) & (speed_cm_s <= 40.0)] = 3
    out[speed_cm_s > 40.0] = 4
    return out


def make_trial_arrays(session_raw, show_processing=False):
    behavior = session_raw["behavior"]
    neural = session_raw["neural"]
    bounds, starts, teleports = build_trial_bounds(behavior)
    reward_outcomes_raw, reward_event_groups = trial_reward_outcomes(
        bounds, session_raw["reward_frame_idx"]
    )
    reward_labels = reward_labels_for_trials(session_raw["scene"], len(bounds))

    neural_trials = []
    input_trials = []
    output_trials = []
    trial_debug = []
    dropped = Counter()

    previous_outcomes = np.zeros(len(bounds), dtype=np.int64)
    if len(bounds) > 1:
        previous_outcomes[1:] = reward_outcomes_raw[:-1]

    for raw_trial_idx, (start, stop) in enumerate(bounds):
        pos = behavior["position"][start:stop].astype(np.float32, copy=False)
        speed = behavior["speed"][start:stop].astype(np.float32, copy=False)
        lick_raw = behavior["lick"][start:stop].astype(np.float32, copy=False)
        env_raw = behavior["environment"][start:stop]
        time_raw = behavior["timestamps"][start:stop].astype(np.float32, copy=False)

        if pos.size == 0:
            dropped["empty"] += 1
            continue

        bad_lick = np.mean(lick_raw > 2.0) > LICK_QC_FRACTION
        if bad_lick:
            dropped["lick_qc"] += 1
            continue

        zone_label = reward_labels[raw_trial_idx]
        zone_start, zone_end = REWARD_ZONE_COORDS[zone_label]
        distance = signed_distance_to_zone(pos, zone_start, zone_end)
        env_value = trial_environment_value(env_raw)
        trial_label = int(behavior["trial number"][start])
        reward_outcome = int(reward_outcomes_raw[raw_trial_idx])
        previous_outcome = int(previous_outcomes[raw_trial_idx])
        lick_binary = (lick_raw > 0.0).astype(np.int64, copy=False)
        time_from_start = (time_raw - time_raw[0]).astype(np.float32, copy=False)

        neural_trial = neural[start:stop].T.astype(np.float32, copy=False)
        input_trial = np.vstack(
            [
                time_from_start,
                np.full(time_from_start.shape, env_value, dtype=np.float32),
                np.full(time_from_start.shape, float(trial_label), dtype=np.float32),
                np.full(time_from_start.shape, float(previous_outcome), dtype=np.float32),
            ]
        )
        output_trial = np.vstack(
            [
                discretize_distance(distance),
                discretize_absolute_position(pos),
                discretize_speed(speed),
                lick_binary,
                np.full(pos.shape, {"A": 0, "B": 1, "C": 2}[zone_label], dtype=np.int64),
                np.full(pos.shape, reward_outcome, dtype=np.int64),
            ]
        )

        neural_trials.append(neural_trial)
        input_trials.append(input_trial.astype(np.float32, copy=False))
        output_trials.append(output_trial.astype(np.int64, copy=False))
        trial_debug.append(
            {
                "raw_trial_idx": raw_trial_idx,
                "trial_label": trial_label,
                "start": start,
                "stop": stop,
                "zone_label": zone_label,
                "zone_start": zone_start,
                "zone_end": zone_end,
                "reward_outcome": reward_outcome,
                "previous_outcome": previous_outcome,
                "env_value": env_value,
                "distance_cm": distance,
                "position_cm": pos,
                "speed_cm_s": speed,
                "lick_raw": lick_raw,
                "time_from_start_sec": time_from_start,
                "reward_event_count": int(len(reward_event_groups[raw_trial_idx])),
            }
        )

    session_info = {
        "subject": session_raw["subject"],
        "session_id": session_raw["session_id"],
        "scene": session_raw["scene"],
        "source_file": session_raw["path"].name,
        "n_rois_total": session_raw["n_rois_total"],
        "n_cells_curated": session_raw["n_cells_curated"],
        "n_trials_bounds": len(bounds),
        "n_trials_kept": len(neural_trials),
        "n_trials_dropped_lick_qc": int(dropped["lick_qc"]),
        "n_trials_dropped_empty": int(dropped["empty"]),
        "n_trial_start_markers": int(len(starts)),
        "n_teleport_markers": int(len(teleports)),
        "n_unique_trial_labels_nonnegative": int(
            len(np.unique(behavior["trial number"][behavior["trial number"] >= 0]))
        ),
        "dt_behavior_sec": session_raw["dt_behavior_sec"],
        "ophys_rate_attr_hz": session_raw["ophys_rate_attr_hz"],
        "is_multiplane": bool(session_raw["is_multiplane"]),
        "trimmed_timepoints": session_raw["trimmed_timepoints"],
        "plane_counts_curated": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(session_raw["plane_idx"], return_counts=True))
        },
    }

    plot_payload = None
    if show_processing:
        plot_payload = {
            "session_key": session_raw["path"].stem.replace("_behavior+ophys", ""),
            "trial_debug": trial_debug,
            "neural_trials": neural_trials,
            "input_trials": input_trials,
            "output_trials": output_trials,
        }

    return neural_trials, input_trials, output_trials, session_info, plot_payload


def plot_processing_summary(plot_payload, input_names, output_names):
    session_key = plot_payload["session_key"]
    trial_debug = plot_payload["trial_debug"]
    neural_trials = plot_payload["neural_trials"]
    output_trials = plot_payload["output_trials"]
    if not trial_debug:
        return

    n_trials_plot = min(3, len(trial_debug))
    fig, axes = plt.subplots(4, n_trials_plot, figsize=(6 * n_trials_plot, 16), sharex=False)
    if n_trials_plot == 1:
        axes = np.asarray(axes)[:, np.newaxis]

    for col in range(n_trials_plot):
        debug = trial_debug[col]
        t = debug["time_from_start_sec"]
        pos = debug["position_cm"]
        speed = debug["speed_cm_s"]
        dist = debug["distance_cm"]
        zone_start = debug["zone_start"]
        zone_end = debug["zone_end"]

        ax = axes[0, col]
        ax.plot(t, pos, color="tab:blue", lw=2, label="position")
        ax.axhspan(zone_start, zone_end, color="tab:green", alpha=0.2, label="reward zone")
        ax2 = ax.twinx()
        ax2.plot(t, speed, color="tab:orange", alpha=0.8, lw=1.5, label="speed")
        ax.set_title(
            f"Trial {debug['trial_label']} | zone {debug['zone_label']} | "
            f"reward {debug['reward_outcome']}"
        )
        ax.set_ylabel("Position (cm)")
        ax2.set_ylabel("Speed (cm/s)")

        ax = axes[1, col]
        ax.plot(t, dist, color="tab:red", lw=2)
        ax.axhline(0.0, color="black", ls="--", lw=1)
        ax.set_ylabel("Signed distance\nto reward zone (cm)")
        ax.set_xlabel("Time from trial start (s)")

        ax = axes[2, col]
        ax.step(t, output_trials[col][0], where="post", label=output_names[0], lw=1.5)
        ax.step(t, output_trials[col][1], where="post", label=output_names[1], lw=1.2)
        ax.step(t, output_trials[col][2], where="post", label=output_names[2], lw=1.2)
        ax.step(t, output_trials[col][3], where="post", label=output_names[3], lw=1.2)
        ax.set_ylabel("Discrete outputs")
        ax.set_xlabel("Time from trial start (s)")
        if col == 0:
            ax.legend(loc="upper right", fontsize=8)

        ax = axes[3, col]
        trial_neural = neural_trials[col]
        nneurons_plot = min(80, trial_neural.shape[0])
        ax.imshow(
            trial_neural[:nneurons_plot, :],
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
        )
        ax.set_ylabel("Neurons")
        ax.set_xlabel("Frame")

    fig.suptitle(f"Processing summary: {session_key}")
    fig.tight_layout()
    fig.savefig(f"processing_{session_key}.png", dpi=150)
    plt.close(fig)


def build_metadata(session_infos, subjects):
    dt_values = [info["dt_behavior_sec"] for info in session_infos if info["n_trials_kept"] > 0]
    dt_sec = float(np.median(dt_values)) if dt_values else DEFAULT_DT_SEC
    mismatch_sessions = [
        info["source_file"] for info in session_infos if info["trimmed_timepoints"] > 0
    ]
    return {
        "task_description": (
            "CA1 deconvolved calcium activity during virtual linear-track navigation; "
            "decoder predicts position-, reward-, and behavior-related variables."
        ),
        "time_bin_size": dt_sec * 1000.0,
        "temporal_alignment_event": "trial start (entry onto the 450 cm track)",
        "off_start": 0.0,
        "off_end": None,
        "source_data_format": "NWB processed behavior+ophys sessions",
        "neural_signal": "Deconvolved calcium events from NWB ophys/Deconvolved/plane0",
        "trial_window": "Samples from trial_start inclusive to teleport exclusive",
        "reward_zone_coordinates_cm": {
            key: [float(val[0]), float(val[1])] for key, val in REWARD_ZONE_COORDS.items()
        },
        "lick_qc_fraction_threshold": LICK_QC_FRACTION,
        "subjects": subjects,
        "session_info": session_infos,
        "sessions_trimmed_to_behavior_length": mismatch_sessions,
    }


def convert_dataset(files, show_processing=False):
    data = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": [],
        "subject_idx": None,
        "brain_regions": [BRAIN_REGION_NAME],
        "brain_region_idx": [],
        "input_names": [
            "time_from_trial_start_sec",
            "environment_type",
            "trial_number",
            "previous_trial_outcome",
        ],
        "output_names": [
            "distance_to_reward_zone",
            "absolute_position",
            "speed",
            "lick",
            "reward_zone_location",
            "reward_outcome",
        ],
        "output_values": [
            [
                "< -50 cm",
                "-50 to -10 cm",
                "-10 cm to < 0 cm",
                "0 cm",
                "> 0 cm to +10 cm",
                "+10 to +50 cm",
                "> +50 cm",
            ],
            ["< 90 cm", "90 to 180 cm", "180 to 270 cm", "270 to 360 cm", "> 360 cm"],
            ["< 2 cm/s", "2-10 cm/s", "10-20 cm/s", "20-40 cm/s", "> 40 cm/s"],
            ["no", "yes"],
            ["A", "B", "C"],
            ["no", "yes"],
        ],
        "metadata": {},
    }

    subject_lookup = {}
    subject_idx = []
    session_infos = []

    processing_files = {p.resolve() for p in pick_processing_files(files)} if show_processing else set()

    total_start = time.perf_counter()
    for session_i, path in enumerate(files):
        session_start = time.perf_counter()
        session_raw = load_session_raw(path)
        make_plots = path.resolve() in processing_files
        neural_trials, input_trials, output_trials, session_info, plot_payload = make_trial_arrays(
            session_raw,
            show_processing=make_plots,
        )

        if len(neural_trials) < 2:
            print(
                f"Skipping {path.name}: only {len(neural_trials)} valid trials after filtering.",
                flush=True,
            )
            continue

        subject = session_raw["subject"]
        if subject not in subject_lookup:
            subject_lookup[subject] = len(subject_lookup)
            data["subjects"].append(subject)
        subject_idx.append(subject_lookup[subject])

        data["neural"].append(neural_trials)
        data["input"].append(input_trials)
        data["output"].append(output_trials)
        data["brain_region_idx"].append(
            np.zeros(session_raw["n_cells_curated"], dtype=np.int64)
        )
        session_infos.append(session_info)

        if plot_payload is not None:
            plot_processing_summary(plot_payload, data["input_names"], data["output_names"])

        elapsed = time.perf_counter() - session_start
        print(
            f"[{session_i + 1:03d}/{len(files):03d}] {path.name}: "
            f"{session_info['n_cells_curated']} cells, "
            f"{session_info['n_trials_kept']} kept trials, "
            f"dropped {session_info['n_trials_dropped_lick_qc']} lick-QC trials, "
            f"{elapsed:.2f}s",
            flush=True,
        )

    total_elapsed = time.perf_counter() - total_start
    data["subject_idx"] = np.asarray(subject_idx, dtype=np.int64)
    data["metadata"] = build_metadata(session_infos, data["subjects"])
    data["metadata"]["conversion_runtime_sec"] = total_elapsed

    print(
        f"Converted {len(data['neural'])} sessions from {len(data['subjects'])} subjects "
        f"in {total_elapsed:.2f}s",
        flush=True,
    )
    return data


def main():
    args = parse_args()

    files = sorted(Path("/app/data").glob("sub-*/sub-*_behavior+ophys.nwb"))
    if not args.sample:
        args.full = True

    if args.sample:
        files = pick_sample_files(files)
        print("Running in sample mode on:", ", ".join(path.name for path in files), flush=True)
    else:
        print(f"Running in full mode on {len(files)} sessions", flush=True)

    data = convert_dataset(files, show_processing=args.show_processing)

    outpath = Path(args.outpicklefile)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {outpath}", flush=True)


if __name__ == "__main__":
    main()
