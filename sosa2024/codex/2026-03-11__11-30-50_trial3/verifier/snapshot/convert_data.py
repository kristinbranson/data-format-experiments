#!/usr/bin/env python3

import argparse
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = Path("data")
SWITCH_TRIAL = 30
LICK_ERROR_FRACTION = 0.35
MIN_TRIAL_FRAMES = 5

ZONE_TO_CODE = {"A": 0, "B": 1, "C": 2}
ZONE_TO_COORDS_CM = {
    "A": (80.0, 130.0),
    "B": (200.0, 250.0),
    "C": (320.0, 370.0),
}

SCENE_SINGLE_RE = re.compile(r"^(Env[12])_Location([ABC])$")
SCENE_SWITCH_RE = re.compile(r"^(Env[12])_Location([ABC])_to_([ABC])$")
SCENE_CROSS_ENV_RE = re.compile(r"^(Env[12])_([ABC])_to_(Env[12])_([ABC])$")


@dataclass(frozen=True)
class SceneInfo:
    before_env: str
    before_zone: str
    after_env: str | None
    after_zone: str | None

    @property
    def has_switch(self) -> bool:
        return self.after_zone is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NWB sessions into decoder-ready pickle format."
    )
    parser.add_argument("outpicklefile", type=Path, help="Output pickle path.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", help="Process all sessions (default).")
    group.add_argument("--sample", action="store_true", help="Process only 2 sessions.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save diagnostic plots for up to 2 sessions as processing_<session_id>.png.",
    )
    return parser.parse_args()


def list_nwb_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
    if sample:
        return files[:2]
    return files


def decode_bytes(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def parse_scene(scene: str) -> SceneInfo:
    match = SCENE_SINGLE_RE.match(scene)
    if match:
        env, zone = match.groups()
        return SceneInfo(before_env=env, before_zone=zone, after_env=None, after_zone=None)

    match = SCENE_SWITCH_RE.match(scene)
    if match:
        env, before_zone, after_zone = match.groups()
        return SceneInfo(
            before_env=env,
            before_zone=before_zone,
            after_env=env,
            after_zone=after_zone,
        )

    match = SCENE_CROSS_ENV_RE.match(scene)
    if match:
        before_env, before_zone, after_env, after_zone = match.groups()
        return SceneInfo(
            before_env=before_env,
            before_zone=before_zone,
            after_env=after_env,
            after_zone=after_zone,
        )

    raise ValueError(f"Unsupported scene format: {scene}")


def zone_for_trial(scene_info: SceneInfo, trial_number: int) -> str:
    if scene_info.has_switch and trial_number >= SWITCH_TRIAL:
        assert scene_info.after_zone is not None
        return scene_info.after_zone
    return scene_info.before_zone


def env_to_binary(values: np.ndarray) -> int:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    values = values[values >= 0]
    if values.size == 0:
        raise ValueError("No valid environment values in trial.")
    return int(np.round(np.median(values)) > 0.5)


def modal_trial_number(values: np.ndarray) -> int:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    values = np.rint(values).astype(np.int64)
    values = values[values >= 0]
    if values.size == 0:
        raise ValueError("No valid trial numbers in trial.")
    counts = np.bincount(values)
    return int(np.argmax(counts))


def find_complete_trial_bounds(trial_start: np.ndarray, teleport: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start > 0)
    teleports = np.flatnonzero(teleport > 0)
    bounds: list[tuple[int, int]] = []
    teleport_idx = 0

    for start in starts:
        while teleport_idx < len(teleports) and teleports[teleport_idx] <= start:
            teleport_idx += 1
        if teleport_idx >= len(teleports):
            break
        stop = teleports[teleport_idx]
        if stop > start:
            bounds.append((int(start), int(stop)))
        teleport_idx += 1

    return bounds


def reward_outcome_for_trial(
    reward_timestamps: np.ndarray,
    t_start: float,
    t_stop: float,
    reward_zone_segment: np.ndarray,
) -> int:
    left = np.searchsorted(reward_timestamps, t_start, side="left")
    right = np.searchsorted(reward_timestamps, t_stop, side="left")
    has_reward = right > left
    has_rzone_entry = np.any(reward_zone_segment > 0)
    return int(has_reward and has_rzone_entry)


def has_lick_sensor_error(lick_segment: np.ndarray) -> bool:
    if lick_segment.size == 0:
        return True
    return bool(np.mean(lick_segment > 2) > LICK_ERROR_FRACTION)


def signed_distance_to_zone(position_cm: np.ndarray, zone_start: float, zone_end: float) -> np.ndarray:
    distance = np.zeros_like(position_cm, dtype=np.float32)
    before = position_cm < zone_start
    after = position_cm > zone_end
    distance[before] = position_cm[before] - zone_start
    distance[after] = position_cm[after] - zone_end
    return distance


def discretize_distance(distance_cm: np.ndarray) -> np.ndarray:
    out = np.full(distance_cm.shape, -1, dtype=np.int16)
    out[distance_cm < -50] = 0
    out[(distance_cm >= -50) & (distance_cm < -10)] = 1
    out[(distance_cm >= -10) & (distance_cm < 0)] = 2
    out[distance_cm == 0] = 3
    out[(distance_cm > 0) & (distance_cm <= 10)] = 4
    out[(distance_cm > 10) & (distance_cm <= 50)] = 5
    out[distance_cm > 50] = 6
    if np.any(out < 0):
        raise ValueError("Failed to discretize reward-zone distance.")
    return out


def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, np.nextafter(450.0, 0.0))
    bins = np.floor(clipped / 90.0).astype(np.int16)
    bins[bins > 4] = 4
    return bins


def discretize_speed(speed_cm_s: np.ndarray) -> np.ndarray:
    out = np.full(speed_cm_s.shape, -1, dtype=np.int16)
    out[speed_cm_s < 2] = 0
    out[(speed_cm_s >= 2) & (speed_cm_s < 10)] = 1
    out[(speed_cm_s >= 10) & (speed_cm_s < 20)] = 2
    out[(speed_cm_s >= 20) & (speed_cm_s < 40)] = 3
    out[speed_cm_s >= 40] = 4
    if np.any(out < 0):
        raise ValueError("Failed to discretize speed.")
    return out


def build_trial_plot(
    session_id: str,
    trial_examples: list[dict],
    output_names: list[str],
    output_values: list[list[str]],
) -> None:
    if not trial_examples:
        return

    ncols = len(trial_examples)
    fig, axes = plt.subplots(4, ncols, figsize=(7 * ncols, 14), squeeze=False)

    for col, ex in enumerate(trial_examples):
        t = ex["time"]
        neural = ex["neural"]
        outputs = ex["output"]
        position = ex["position"]
        speed = ex["speed"]
        lick = ex["lick"]
        zone_start, zone_end = ex["zone_coords"]
        reward_times = ex["reward_times"]

        sample_neurons = neural[: min(60, neural.shape[0])]
        axes[0, col].imshow(sample_neurons, aspect="auto", interpolation="nearest", cmap="magma")
        axes[0, col].set_title(
            f"{session_id} trial {ex['trial_number']}  env={ex['environment']}  "
            f"zone={ex['zone_label']}  rewarded={ex['reward_outcome']}"
        )
        axes[0, col].set_ylabel("Neuron")

        axes[1, col].plot(t, position, color="black", linewidth=1.5)
        axes[1, col].fill_between(
            t,
            zone_start,
            zone_end,
            color="tab:green",
            alpha=0.2,
            linewidth=0,
        )
        if reward_times.size:
            for rt in reward_times:
                axes[1, col].axvline(rt, color="tab:orange", linestyle="--", linewidth=1)
        axes[1, col].set_ylabel("Pos (cm)")

        axes[2, col].plot(t, speed, color="tab:blue", linewidth=1.5, label="speed")
        axes[2, col].step(t, lick * max(1.0, np.nanmax(speed)), where="mid", color="tab:red", label="lick")
        axes[2, col].legend(loc="upper right", fontsize=8)
        axes[2, col].set_ylabel("Speed / lick")

        axes[3, col].imshow(outputs, aspect="auto", interpolation="nearest", cmap="tab20")
        axes[3, col].set_yticks(np.arange(len(output_names)))
        axes[3, col].set_yticklabels(output_names)
        axes[3, col].set_xlabel("Frame")
        axes[3, col].set_ylabel("Discrete outputs")

    fig.tight_layout()
    fig.savefig(f"processing_{session_id}.png", dpi=150)
    plt.close(fig)


def process_session(path: Path) -> tuple[dict, list[dict]]:
    session_id = path.stem.replace("_behavior+ophys", "")
    with h5py.File(path, "r") as f:
        identifier = decode_bytes(f["identifier"][()])
        scene = identifier.split("/")[-1]
        scene_info = parse_scene(scene)

        subject = path.parent.name.replace("sub-", "")
        session_name = path.stem

        behavior_group = f["processing/behavior/BehavioralTimeSeries"]
        ophys_group = f["processing/ophys"]

        position = behavior_group["position/data"][()].astype(np.float32)
        speed = behavior_group["speed/data"][()].astype(np.float32)
        lick = behavior_group["lick/data"][()].astype(np.float32)
        environment = behavior_group["environment/data"][()].astype(np.float32)
        reward_zone = behavior_group["reward_zone/data"][()].astype(np.float32)
        trial_number = behavior_group["trial number/data"][()].astype(np.float32)
        trial_start = behavior_group["trial_start/data"][()].astype(np.float32)
        teleport = behavior_group["teleport/data"][()].astype(np.float32)
        timestamps = behavior_group["position/timestamps"][()].astype(np.float64)
        reward_timestamps = behavior_group["Reward/timestamps"][()].astype(np.float64)

        iscell = ophys_group["ImageSegmentation/PlaneSegmentation/iscell"][()]
        accepted_mask = np.asarray(iscell[:, 0]) == 1
        accepted_idx = np.flatnonzero(accepted_mask)
        plane_idx_all = ophys_group["ImageSegmentation/PlaneSegmentation/planeIdx"][()].astype(np.int16)
        plane_idx = plane_idx_all[accepted_idx]

        deconv_shape_t = None
        deconv = None
        planes = sorted(int(x) for x in np.unique(plane_idx_all))
        for plane in planes:
            plane_roi_idx = np.flatnonzero(plane_idx_all == plane)
            accepted_total_idx = plane_roi_idx[accepted_mask[plane_roi_idx]]
            accepted_local_idx = np.flatnonzero(accepted_mask[plane_roi_idx])
            plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
            plane_data = plane_data.astype(np.float32, copy=False)

            if deconv_shape_t is None:
                deconv_shape_t = plane_data.shape[0]
                deconv = np.empty((deconv_shape_t, accepted_idx.size), dtype=np.float32)

            dest_cols = np.searchsorted(accepted_idx, accepted_total_idx)
            deconv[:, dest_cols] = plane_data

        if deconv is None:
            raise ValueError(f"No accepted deconvolved traces found in {path}")
        brain_region_idx = np.zeros(accepted_idx.size, dtype=np.int64)

    trial_bounds = find_complete_trial_bounds(trial_start, teleport)
    valid_behavior_frames = (
        np.isfinite(position)
        & np.isfinite(speed)
        & np.isfinite(lick)
        & np.isfinite(environment)
        & np.isfinite(timestamps)
        & (position > -100.0)
    )
    valid_neural_frames = np.all(np.isfinite(deconv), axis=1)

    trial_meta: list[dict] = []
    reward_by_trial_number: dict[int, int] = {}
    dropped_missing = 0
    dropped_lick = 0
    kept_examples: list[dict] = []

    for start, stop in trial_bounds:
        if stop - start < MIN_TRIAL_FRAMES:
            dropped_missing += 1
            continue

        trial_num = modal_trial_number(trial_number[start:stop])
        env_bin = env_to_binary(environment[start:stop])
        zone_label = zone_for_trial(scene_info, trial_num)
        zone_coords = ZONE_TO_COORDS_CM[zone_label]
        reward_outcome = reward_outcome_for_trial(
            reward_timestamps=reward_timestamps,
            t_start=float(timestamps[start]),
            t_stop=float(timestamps[stop]),
            reward_zone_segment=reward_zone[start:stop],
        )
        reward_by_trial_number[trial_num] = reward_outcome

        trial_meta.append(
            {
                "start": start,
                "stop": stop,
                "trial_number": trial_num,
                "environment": env_bin,
                "zone_label": zone_label,
                "zone_coords": zone_coords,
                "reward_outcome": reward_outcome,
                "lick_error": has_lick_sensor_error(lick[start:stop]),
            }
        )

    neural_trials: list[np.ndarray] = []
    input_trials: list[np.ndarray] = []
    output_trials: list[np.ndarray] = []

    for meta in trial_meta:
        if meta["lick_error"]:
            dropped_lick += 1
            continue

        start = meta["start"]
        stop = meta["stop"]
        trial_num = meta["trial_number"]
        prev_outcome = reward_by_trial_number.get(trial_num - 1, 0)

        frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
        if np.count_nonzero(frame_mask) < MIN_TRIAL_FRAMES:
            dropped_missing += 1
            continue

        trial_slice = slice(start, stop)
        position_trial = position[trial_slice][frame_mask]
        speed_trial = speed[trial_slice][frame_mask]
        lick_trial = lick[trial_slice][frame_mask]
        time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
        neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)

        zone_start, zone_end = meta["zone_coords"]
        distance_trial = signed_distance_to_zone(position_trial, zone_start, zone_end)

        input_trial = np.vstack(
            [
                time_trial.astype(np.float32),
                np.full(time_trial.shape, meta["environment"], dtype=np.float32),
                np.full(time_trial.shape, trial_num, dtype=np.float32),
                np.full(time_trial.shape, prev_outcome, dtype=np.float32),
            ]
        )

        output_trial = np.vstack(
            [
                discretize_distance(distance_trial),
                discretize_absolute_position(position_trial),
                discretize_speed(speed_trial),
                (lick_trial > 0).astype(np.int16),
                np.full(time_trial.shape, ZONE_TO_CODE[meta["zone_label"]], dtype=np.int16),
                np.full(time_trial.shape, meta["reward_outcome"], dtype=np.int16),
            ]
        )

        neural_trials.append(neural_trial)
        input_trials.append(input_trial.astype(np.float32, copy=False))
        output_trials.append(output_trial)

        if len(kept_examples) < 3:
            trial_reward_times = reward_timestamps[
                (reward_timestamps >= timestamps[start]) & (reward_timestamps < timestamps[stop])
            ] - timestamps[start]
            kept_examples.append(
                {
                    "trial_number": trial_num,
                    "environment": meta["environment"],
                    "zone_label": meta["zone_label"],
                    "zone_coords": meta["zone_coords"],
                    "reward_outcome": meta["reward_outcome"],
                    "time": time_trial,
                    "position": position_trial,
                    "speed": speed_trial,
                    "lick": (lick_trial > 0).astype(np.int16),
                    "neural": neural_trial,
                    "output": output_trial,
                    "reward_times": trial_reward_times,
                }
            )

    session_summary = {
        "subject": subject,
        "session_name": session_name,
        "scene": scene,
        "file": str(path),
        "n_neurons": int(accepted_idx.size),
        "n_planes": int(np.unique(plane_idx).size),
        "n_trials_complete": int(len(trial_meta)),
        "n_trials_kept": int(len(neural_trials)),
        "n_trials_dropped_lick": int(dropped_lick),
        "n_trials_dropped_missing": int(dropped_missing),
        "time_bin_size_ms": float(np.median(np.diff(timestamps)) * 1000.0),
        "trial_numbers_kept": [int(np.rint(x[0, 0])) for x in input_trials] if input_trials else [],
    }

    return (
        {
            "neural": neural_trials,
            "input": input_trials,
            "output": output_trials,
            "brain_region_idx": brain_region_idx,
            "subject": subject,
            "summary": session_summary,
            "examples": kept_examples,
        },
        kept_examples,
    )


def build_dataset(processed_sessions: list[dict]) -> dict:
    subjects = sorted({sess["subject"] for sess in processed_sessions})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}

    median_bin_ms = float(
        np.median([sess["summary"]["time_bin_size_ms"] for sess in processed_sessions])
    )

    dataset = {
        "neural": [sess["neural"] for sess in processed_sessions],
        "input": [sess["input"] for sess in processed_sessions],
        "output": [sess["output"] for sess in processed_sessions],
        "subjects": subjects,
        "subject_idx": np.asarray(
            [subject_to_idx[sess["subject"]] for sess in processed_sessions],
            dtype=np.int64,
        ),
        "brain_regions": ["CA1"],
        "brain_region_idx": [sess["brain_region_idx"] for sess in processed_sessions],
        "input_names": [
            "time_from_trial_start_sec",
            "environment",
            "trial_number",
            "previous_trial_rewarded",
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
            ["lt_-50", "-50_to_-10", "-10_to_0", "0", "0_to_10", "10_to_50", "gt_50"],
            ["bin0", "bin1", "bin2", "bin3", "bin4"],
            ["lt_2", "2_to_10", "10_to_20", "20_to_40", "gt_40"],
            ["no", "yes"],
            ["A", "B", "C"],
            ["no", "yes"],
        ],
        "metadata": {
            "task_description": (
                "CA1 deconvolved calcium activity during a hidden reward-zone VR task; "
                "decoder predicts trial-aligned behavioral and task variables."
            ),
            "time_bin_size": median_bin_ms,
            "temporal_alignment_event": "trial_start",
            "off_start": 0.0,
            "off_end": None,
            "source_format": "NWB",
            "source_dataset": "DANDI 001361 switch-task cohort",
            "neural_signal": "suite2p/OASIS deconvolved activity",
            "trial_definition": "frames from trial_start inclusive to teleport exclusive",
            "lick_error_rule": (
                "drop trials with >35% of in-trial frames having cumulative lick count >2"
            ),
            "reward_zone_coords_cm": {k: list(v) for k, v in ZONE_TO_COORDS_CM.items()},
            "session_info": [sess["summary"] for sess in processed_sessions],
        },
    }
    return dataset


def main() -> None:
    args = parse_args()
    sample_mode = args.sample
    nwb_files = list_nwb_files(sample=sample_mode)
    if not nwb_files:
        raise FileNotFoundError("No NWB files found under data/.")

    start_time = time.time()
    processed_sessions: list[dict] = []
    plots_made = 0

    print(f"Found {len(nwb_files)} NWB files to process.")

    for idx, path in enumerate(nwb_files, start=1):
        session_start = time.time()
        session, examples = process_session(path)
        if len(session["neural"]) < 2:
            print(
                f"[{idx}/{len(nwb_files)}] Skipping {path.name}: "
                f"only {len(session['neural'])} valid trials after filtering."
            )
            continue

        processed_sessions.append(session)

        elapsed = time.time() - session_start
        total_elapsed = time.time() - start_time
        mean_per_session = total_elapsed / len(processed_sessions)
        eta = mean_per_session * (len(nwb_files) - idx)
        print(
            f"[{idx}/{len(nwb_files)}] {path.name}: "
            f"{session['summary']['n_trials_kept']} trials, "
            f"{session['summary']['n_neurons']} neurons, "
            f"{elapsed:.2f}s (ETA {eta / 60:.1f} min)"
        )

        if args.show_processing and plots_made < 2:
            build_trial_plot(
                session["summary"]["session_name"],
                examples,
                output_names=[
                    "dist_to_zone",
                    "abs_pos",
                    "speed",
                    "lick",
                    "zone",
                    "reward",
                ],
                output_values=[],
            )
            plots_made += 1

    if not processed_sessions:
        raise RuntimeError("No sessions remained after filtering.")

    dataset = build_dataset(processed_sessions)

    with args.outpicklefile.open("wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_time = time.time() - start_time
    print(
        f"Wrote {args.outpicklefile} with {len(processed_sessions)} sessions "
        f"in {total_time / 60:.2f} min."
    )


if __name__ == "__main__":
    main()
