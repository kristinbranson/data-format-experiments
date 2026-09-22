#!/usr/bin/env python3
import argparse
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pynwb import NWBHDF5IO


REWARD_ZONE_COORDS_CM = {
    "A": (80.0, 130.0),
    "B": (200.0, 250.0),
    "C": (320.0, 370.0),
}
OUTPUT_VALUES = [
    ["lt_-50", "-50_to_-10", "-10_to_0", "in_zone", "0_to_10", "10_to_50", "gt_50"],
    ["lt_90", "90_to_180", "180_to_270", "270_to_360", "gt_360"],
    ["lt_2", "2_to_10", "10_to_20", "20_to_40", "gt_40"],
    ["no", "yes"],
    ["A", "B", "C"],
    ["no", "yes"],
]
INPUT_NAMES = [
    "time_from_trial_start_s",
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
EXPECTED_FRAME_DT_S = 1.0 / 15.5078125
LICK_SENSOR_COUNT_THRESHOLD = 2.0
LICK_SENSOR_BAD_FRAC = 0.35
SCENE_SWITCH_TRIAL = 30


@dataclass
class SessionMeta:
    subject: str
    subject_original: str
    session_name: str
    session_id: str
    date: str
    scene: str
    file_path: str
    n_planes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NWB sessions into decoder-ready trial-aligned pickle data."
    )
    parser.add_argument("outpicklefile", type=str, help="Output pickle path.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Process all sessions (default behavior).",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Process only two sessions for testing.",
    )
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing figures for up to two sessions.",
    )
    return parser.parse_args()


def scene_to_reward_labels(scene: str, ntrials: int, change_trial: int = SCENE_SWITCH_TRIAL) -> list[str]:
    if scene in {"Env1_LocationA", "Env2_LocationA", "Env3_LocationA"}:
        return ["A"] * ntrials
    if scene in {"Env1_LocationB", "Env2_LocationB", "Env3_LocationB"}:
        return ["B"] * ntrials
    if scene in {"Env1_LocationC", "Env2_LocationC", "Env3_LocationC"}:
        return ["C"] * ntrials

    first_n = min(change_trial, ntrials)
    second_n = max(0, ntrials - change_trial)

    if "A_to" in scene and scene.endswith("B"):
        return ["A"] * first_n + ["B"] * second_n
    if "B_to" in scene and scene.endswith("A"):
        return ["B"] * first_n + ["A"] * second_n
    if "A_to" in scene and scene.endswith("C"):
        return ["A"] * first_n + ["C"] * second_n
    if "C_to" in scene and scene.endswith("A"):
        return ["C"] * first_n + ["A"] * second_n
    if "B_to" in scene and scene.endswith("C"):
        return ["B"] * first_n + ["C"] * second_n
    if "C_to" in scene and scene.endswith("B"):
        return ["C"] * first_n + ["B"] * second_n
    if "Training" in scene:
        return ["A"] * ntrials

    raise ValueError(f"Unsupported scene string for reward-zone inference: {scene}")


def reward_label_to_code(label: str) -> int:
    return {"A": 0, "B": 1, "C": 2}[label]


def parse_identifier(identifier: str, file_path: Path) -> SessionMeta:
    parts = identifier.strip("/").split("/")
    subject_original = parts[-3] if len(parts) >= 3 else file_path.parent.name.replace("sub-", "GCAMP")
    date = parts[-2] if len(parts) >= 2 else "unknown_date"
    scene = parts[-1] if len(parts) >= 1 else file_path.stem
    subject = file_path.parent.name.replace("sub-", "")
    session_match = re.search(r"_ses-(\d+)_", file_path.name)
    session_id = session_match.group(1) if session_match else "unknown"
    n_planes = 1
    return SessionMeta(
        subject=subject,
        subject_original=subject_original,
        session_name=file_path.stem.replace("_behavior+ophys", ""),
        session_id=session_id,
        date=date,
        scene=scene,
        file_path=str(file_path),
        n_planes=n_planes,
    )


def pair_trial_bounds(trial_start: np.ndarray, teleport: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start > 0)
    stops = np.flatnonzero(teleport > 0)
    bounds: list[tuple[int, int]] = []
    stop_idx = 0
    for start in starts:
        while stop_idx < len(stops) and stops[stop_idx] <= start:
            stop_idx += 1
        if stop_idx >= len(stops):
            break
        stop = int(stops[stop_idx])
        bounds.append((int(start), stop))
        stop_idx += 1
    return bounds


def rasterize_reward_events(frame_times: np.ndarray, reward_times: np.ndarray) -> np.ndarray:
    reward_frames = np.zeros(frame_times.shape[0], dtype=np.int8)
    if reward_times.size == 0:
        return reward_frames
    reward_indices = np.searchsorted(frame_times, reward_times)
    reward_indices = np.clip(reward_indices, 0, frame_times.shape[0] - 1)
    nearest = frame_times[reward_indices]
    if not np.allclose(nearest, reward_times, atol=1e-9, rtol=0):
        diffs = np.abs(nearest - reward_times)
        raise ValueError(f"Reward timestamps do not align to frame timestamps. Max diff={diffs.max()}")
    reward_frames[reward_indices] = 1
    return reward_frames


def get_plane_number(name: str) -> int:
    match = re.search(r"plane(\d+)", name)
    if match is None:
        raise ValueError(f"Unexpected plane name: {name}")
    return int(match.group(1))


def load_curated_deconvolved(nwb) -> tuple[np.ndarray, np.ndarray, int]:
    seg = nwb.processing["ophys"]["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
    iscell = np.asarray(seg["iscell"].data[:] if hasattr(seg["iscell"], "data") else seg["iscell"][:])
    plane_idx = np.asarray(seg["planeIdx"].data[:] if hasattr(seg["planeIdx"], "data") else seg["planeIdx"][:]).astype(int)
    keep = iscell[:, 0] == 1

    deconv_mod = nwb.processing["ophys"]["Deconvolved"]
    plane_names = sorted(deconv_mod.roi_response_series.keys(), key=get_plane_number)
    arrays = []
    plane_per_cell = []
    for plane_name in plane_names:
        plane_num = get_plane_number(plane_name)
        rs = deconv_mod.roi_response_series[plane_name]
        plane_keep_mask = keep & (plane_idx == plane_num)
        plane_keep_local = plane_keep_mask[plane_idx == plane_num]
        if not np.any(plane_keep_local):
            continue
        plane_data = np.asarray(rs.data[:, plane_keep_local], dtype=np.float32)
        arrays.append(plane_data)
        plane_per_cell.append(np.full(plane_data.shape[1], plane_num, dtype=np.int16))

    if not arrays:
        raise ValueError("No curated cells found in session.")

    deconv = np.concatenate(arrays, axis=1)
    planes = np.concatenate(plane_per_cell, axis=0)
    return deconv, planes, len(plane_names)


def signed_distance_to_interval(position_cm: np.ndarray, start_cm: float, stop_cm: float) -> np.ndarray:
    out = np.zeros_like(position_cm, dtype=np.float32)
    below = position_cm < start_cm
    above = position_cm > stop_cm
    out[below] = position_cm[below] - start_cm
    out[above] = position_cm[above] - stop_cm
    return out


def discretize_reward_distance(distance_cm: np.ndarray) -> np.ndarray:
    out = np.full(distance_cm.shape, 6, dtype=np.int8)
    out[distance_cm < -50.0] = 0
    out[(distance_cm >= -50.0) & (distance_cm < -10.0)] = 1
    out[(distance_cm >= -10.0) & (distance_cm < 0.0)] = 2
    out[distance_cm == 0.0] = 3
    out[(distance_cm > 0.0) & (distance_cm <= 10.0)] = 4
    out[(distance_cm > 10.0) & (distance_cm <= 50.0)] = 5
    out[distance_cm > 50.0] = 6
    return out


def discretize_position(position_cm: np.ndarray) -> np.ndarray:
    out = np.full(position_cm.shape, 4, dtype=np.int8)
    out[position_cm < 90.0] = 0
    out[(position_cm >= 90.0) & (position_cm < 180.0)] = 1
    out[(position_cm >= 180.0) & (position_cm < 270.0)] = 2
    out[(position_cm >= 270.0) & (position_cm < 360.0)] = 3
    out[position_cm >= 360.0] = 4
    return out


def discretize_speed(speed_cm_s: np.ndarray) -> np.ndarray:
    out = np.full(speed_cm_s.shape, 4, dtype=np.int8)
    out[speed_cm_s < 2.0] = 0
    out[(speed_cm_s >= 2.0) & (speed_cm_s < 10.0)] = 1
    out[(speed_cm_s >= 10.0) & (speed_cm_s < 20.0)] = 2
    out[(speed_cm_s >= 20.0) & (speed_cm_s <= 40.0)] = 3
    out[speed_cm_s > 40.0] = 4
    return out


def correct_lick_trial(lick_counts: np.ndarray) -> tuple[np.ndarray, bool]:
    corrected = lick_counts.copy()
    bad_trial = float(np.mean(corrected > LICK_SENSOR_COUNT_THRESHOLD)) > LICK_SENSOR_BAD_FRAC
    if bad_trial:
        corrected[:] = 0.0
    corrected = (corrected > 0).astype(np.int8)
    return corrected, bad_trial


def trial_reward_outcomes(bounds: list[tuple[int, int]], reward_frames: np.ndarray) -> np.ndarray:
    outcomes = np.zeros(len(bounds), dtype=np.int8)
    for idx, (start, stop) in enumerate(bounds):
        outcomes[idx] = int(np.any(reward_frames[start:stop] > 0))
    return outcomes


def mode_int(values: np.ndarray) -> int:
    uniq, counts = np.unique(values.astype(int), return_counts=True)
    return int(uniq[np.argmax(counts)])


def session_brain_region_idx(nneurons: int) -> np.ndarray:
    return np.zeros(nneurons, dtype=np.int16)


def make_processing_plot(
    meta: SessionMeta,
    frame_times: np.ndarray,
    position: np.ndarray,
    speed: np.ndarray,
    lick_binary: np.ndarray,
    reward_frames: np.ndarray,
    trial_bounds: list[tuple[int, int]],
    reward_zone_labels: list[str],
    reward_zone_codes: np.ndarray,
    neural_trials: list[np.ndarray],
    input_trials: list[np.ndarray],
    output_trials: list[np.ndarray],
) -> None:
    if not trial_bounds:
        return

    example_idx = min(5, len(trial_bounds) - 1)
    start, stop = trial_bounds[example_idx]
    zone_label = reward_zone_labels[example_idx]
    zone_start, zone_stop = REWARD_ZONE_COORDS_CM[zone_label]

    fig, axes = plt.subplots(4, 2, figsize=(18, 14))

    axes[0, 0].plot(frame_times, position, color="black", linewidth=0.8)
    axes[0, 0].scatter(frame_times[reward_frames > 0], position[reward_frames > 0], color="gold", s=8, label="reward")
    for s, e in trial_bounds[: min(20, len(trial_bounds))]:
        axes[0, 0].axvline(frame_times[s], color="green", alpha=0.08)
        axes[0, 0].axvline(frame_times[e], color="red", alpha=0.08)
    axes[0, 0].set_title("Full Session Position / Reward")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Position (cm)")

    axes[0, 1].plot(frame_times, speed, color="tab:blue", linewidth=0.8, label="speed")
    axes[0, 1].plot(frame_times, lick_binary * np.nanmax(speed), color="tab:orange", alpha=0.6, label="lick")
    axes[0, 1].set_title("Full Session Speed / Lick")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].legend(loc="upper right")

    trial_t = frame_times[start:stop] - frame_times[start]
    axes[1, 0].plot(trial_t, position[start:stop], color="black", linewidth=1.0)
    axes[1, 0].axhspan(zone_start, zone_stop, color="gold", alpha=0.25, label=f"zone {zone_label}")
    axes[1, 0].scatter(trial_t[reward_frames[start:stop] > 0], position[start:stop][reward_frames[start:stop] > 0], color="red", s=16)
    axes[1, 0].set_title(f"Example Trial Position ({meta.session_name}, trial {example_idx})")
    axes[1, 0].set_xlabel("Time from trial start (s)")
    axes[1, 0].set_ylabel("Position (cm)")
    axes[1, 0].legend(loc="upper left")

    axes[1, 1].plot(trial_t, speed[start:stop], color="tab:blue", label="speed")
    axes[1, 1].step(trial_t, lick_binary[start:stop] * np.nanmax(speed[start:stop]), where="mid", color="tab:orange", label="lick")
    axes[1, 1].set_title("Example Trial Speed / Lick")
    axes[1, 1].set_xlabel("Time from trial start (s)")
    axes[1, 1].legend(loc="upper right")

    sample_neurons = min(60, neural_trials[example_idx].shape[0])
    neural_img = neural_trials[example_idx][:sample_neurons, :]
    vmax = float(np.percentile(neural_img, 99)) if neural_img.size else 1.0
    axes[2, 0].imshow(neural_img, aspect="auto", cmap="viridis", interpolation="nearest", vmax=vmax)
    axes[2, 0].set_title("Example Trial Neural Activity")
    axes[2, 0].set_xlabel("Frame")
    axes[2, 0].set_ylabel("Neuron")

    output_img = output_trials[example_idx]
    axes[2, 1].imshow(output_img, aspect="auto", cmap="tab20", interpolation="nearest", vmin=0)
    axes[2, 1].set_title("Example Trial Discrete Outputs")
    axes[2, 1].set_xlabel("Frame")
    axes[2, 1].set_yticks(np.arange(len(OUTPUT_NAMES)))
    axes[2, 1].set_yticklabels(OUTPUT_NAMES)

    trial_lengths = np.array([n.shape[1] for n in neural_trials], dtype=int)
    axes[3, 0].hist(trial_lengths, bins=min(20, len(trial_lengths)), color="gray", edgecolor="black")
    axes[3, 0].set_title("Trial Lengths")
    axes[3, 0].set_xlabel("Frames / trial")
    axes[3, 0].set_ylabel("Count")

    rz_codes = reward_zone_codes.astype(int)
    trial_reward = output_trials[0][5, 0:1]
    reward_outcomes = np.array([trial[5, 0] for trial in output_trials], dtype=int)
    axes[3, 1].scatter(np.arange(len(rz_codes)), rz_codes, c=reward_outcomes, cmap="coolwarm", s=18)
    axes[3, 1].set_title("Trial Reward Zone / Outcome")
    axes[3, 1].set_xlabel("Trial index")
    axes[3, 1].set_ylabel("Reward zone code")

    fig.suptitle(f"Processing check: {meta.session_name} ({meta.scene})")
    fig.tight_layout()
    fig.savefig(f"processing_{meta.session_name}.png", dpi=150)
    plt.close(fig)


def convert_session(file_path: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, SessionMeta, dict]:
    with NWBHDF5IO(str(file_path), "r", load_namespaces=True) as io:
        nwb = io.read()
        meta = parse_identifier(nwb.identifier, file_path)

        beh = nwb.processing["behavior"]["BehavioralTimeSeries"]
        frame_times = np.asarray(beh.time_series["position"].timestamps[:], dtype=np.float64)
        position = np.asarray(beh.time_series["position"].data[:], dtype=np.float32)
        speed = np.asarray(beh.time_series["speed"].data[:], dtype=np.float32)
        environment = np.asarray(beh.time_series["environment"].data[:], dtype=np.float32)
        trial_start = np.asarray(beh.time_series["trial_start"].data[:], dtype=np.float32)
        teleport = np.asarray(beh.time_series["teleport"].data[:], dtype=np.float32)
        lick_counts = np.asarray(beh.time_series["lick"].data[:], dtype=np.float32)
        trial_number = np.asarray(beh.time_series["trial number"].data[:], dtype=np.float32)
        reward_times = np.asarray(beh.time_series["Reward"].timestamps[:], dtype=np.float64)
        reward_frames = rasterize_reward_events(frame_times, reward_times)

        deconv, plane_per_cell, n_planes = load_curated_deconvolved(nwb)
        meta.n_planes = n_planes
        frame_delta = deconv.shape[0] - frame_times.shape[0]
        if frame_delta == 1:
            deconv = deconv[: frame_times.shape[0], :]
        elif frame_delta != 0:
            raise ValueError(
                f"{meta.session_name}: neural frames ({deconv.shape[0]}) do not match behavior frames ({frame_times.shape[0]})"
            )
        frame_dt_s = float(np.median(np.diff(frame_times)))
        if not np.isclose(frame_dt_s, EXPECTED_FRAME_DT_S, atol=1e-9, rtol=0):
            raise ValueError(
                f"{meta.session_name}: unexpected frame dt {frame_dt_s} s; expected {EXPECTED_FRAME_DT_S} s"
            )

    trial_bounds = pair_trial_bounds(trial_start, teleport)
    if len(trial_bounds) < 2:
        raise ValueError(f"{meta.session_name}: expected at least 2 complete trials, found {len(trial_bounds)}")

    reward_labels = scene_to_reward_labels(meta.scene, len(trial_bounds))
    reward_codes = np.array([reward_label_to_code(label) for label in reward_labels], dtype=np.int8)
    reward_outcomes = trial_reward_outcomes(trial_bounds, reward_frames)
    prev_reward_outcomes = np.zeros_like(reward_outcomes)
    if reward_outcomes.size > 1:
        prev_reward_outcomes[1:] = reward_outcomes[:-1]

    neural_trials: list[np.ndarray] = []
    input_trials: list[np.ndarray] = []
    output_trials: list[np.ndarray] = []
    bad_lick_trials = 0
    trial_numbers_kept = []

    for trial_idx, (start, stop) in enumerate(trial_bounds):
        if stop <= start:
            continue
        if stop > deconv.shape[0]:
            raise ValueError(f"{meta.session_name}: trial stop exceeds neural data length")

        env_trial = environment[start:stop]
        env_valid = env_trial[env_trial >= 0]
        if env_valid.size == 0:
            continue
        env_code = float(mode_int(env_valid))

        trial_num_native = int(round(float(trial_number[start])))
        if trial_num_native < 0:
            trial_num_native = trial_idx

        lick_binary, bad_lick = correct_lick_trial(lick_counts[start:stop])
        bad_lick_trials += int(bad_lick)

        neural_trial = np.nan_to_num(deconv[start:stop, :].T, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16, copy=False)
        pos_trial = np.nan_to_num(position[start:stop], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        speed_trial = np.nan_to_num(speed[start:stop], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        trial_time_s = (frame_times[start:stop] - frame_times[start]).astype(np.float32, copy=False)

        rz_label = reward_labels[trial_idx]
        rz_start, rz_stop = REWARD_ZONE_COORDS_CM[rz_label]
        rz_dist = signed_distance_to_interval(pos_trial, rz_start, rz_stop)

        input_trial = np.vstack([
            trial_time_s,
            np.full(trial_time_s.shape, env_code, dtype=np.float32),
            np.full(trial_time_s.shape, float(trial_num_native), dtype=np.float32),
            np.full(trial_time_s.shape, float(prev_reward_outcomes[trial_idx]), dtype=np.float32),
        ]).astype(np.float32, copy=False)

        output_trial = np.vstack([
            discretize_reward_distance(rz_dist),
            discretize_position(pos_trial),
            discretize_speed(speed_trial),
            lick_binary,
            np.full(trial_time_s.shape, reward_codes[trial_idx], dtype=np.int8),
            np.full(trial_time_s.shape, reward_outcomes[trial_idx], dtype=np.int8),
        ]).astype(np.int16, copy=False)

        neural_trials.append(neural_trial)
        input_trials.append(input_trial)
        output_trials.append(output_trial)
        trial_numbers_kept.append(trial_num_native)

    if len(neural_trials) < 2:
        raise ValueError(f"{meta.session_name}: fewer than 2 usable trials after conversion")

    stats = {
        "n_trials": len(neural_trials),
        "n_neurons": int(neural_trials[0].shape[0]),
        "mean_trial_len": float(np.mean([trial.shape[1] for trial in neural_trials])),
        "bad_lick_trials": int(bad_lick_trials),
        "reward_fraction": float(np.mean(reward_outcomes[: len(neural_trials)])),
        "reward_zone_codes": reward_codes.tolist(),
        "trial_numbers_kept": trial_numbers_kept,
        "frame_dt_s": frame_dt_s,
    }
    return neural_trials, input_trials, output_trials, plane_per_cell, meta, stats


def build_dataset(files: list[Path], show_processing: bool) -> dict:
    neural_all = []
    input_all = []
    output_all = []
    plane_all = []
    session_info = []
    subjects = []
    subject_to_idx = {}
    subject_idx = []
    time_bin_ms = None

    t0 = time.perf_counter()
    for session_idx, file_path in enumerate(files):
        session_t0 = time.perf_counter()
        neural_trials, input_trials, output_trials, plane_per_cell, meta, stats = convert_session(file_path)

        if time_bin_ms is None:
            time_bin_ms = float(stats["frame_dt_s"]) * 1000.0

        if meta.subject not in subject_to_idx:
            subject_to_idx[meta.subject] = len(subjects)
            subjects.append(meta.subject)
        subject_idx.append(subject_to_idx[meta.subject])

        neural_all.append(neural_trials)
        input_all.append(input_trials)
        output_all.append(output_trials)
        plane_all.append(session_brain_region_idx(neural_trials[0].shape[0]))
        session_info.append({
            "subject": meta.subject,
            "subject_original": meta.subject_original,
            "session_name": meta.session_name,
            "session_id": meta.session_id,
            "scene": meta.scene,
            "date": meta.date,
            "file_path": meta.file_path,
            "n_planes": meta.n_planes,
            **stats,
        })

        if show_processing and session_idx < 2:
            full_lick = np.concatenate([trial[3] for trial in output_trials]).astype(np.int8)
            with NWBHDF5IO(str(file_path), "r", load_namespaces=True) as io:
                nwb = io.read()
                beh = nwb.processing["behavior"]["BehavioralTimeSeries"]
                frame_times = np.asarray(beh.time_series["position"].timestamps[:], dtype=np.float64)
                position = np.asarray(beh.time_series["position"].data[:], dtype=np.float32)
                speed = np.asarray(beh.time_series["speed"].data[:], dtype=np.float32)
                reward_times = np.asarray(beh.time_series["Reward"].timestamps[:], dtype=np.float64)
                reward_frames = rasterize_reward_events(frame_times, reward_times)
                lick_counts = np.asarray(beh.time_series["lick"].data[:], dtype=np.float32)
                trial_start = np.asarray(beh.time_series["trial_start"].data[:], dtype=np.float32)
                teleport = np.asarray(beh.time_series["teleport"].data[:], dtype=np.float32)
            corrected_full_lick = np.zeros_like(lick_counts, dtype=np.int8)
            for start, stop in pair_trial_bounds(trial_start, teleport):
                corrected_full_lick[start:stop], _ = correct_lick_trial(lick_counts[start:stop])
            make_processing_plot(
                meta=meta,
                frame_times=frame_times,
                position=position,
                speed=speed,
                lick_binary=corrected_full_lick,
                reward_frames=reward_frames,
                trial_bounds=pair_trial_bounds(trial_start, teleport),
                reward_zone_labels=scene_to_reward_labels(meta.scene, len(pair_trial_bounds(trial_start, teleport))),
                reward_zone_codes=np.array([reward_label_to_code(label) for label in scene_to_reward_labels(meta.scene, len(pair_trial_bounds(trial_start, teleport)))], dtype=np.int8),
                neural_trials=neural_trials,
                input_trials=input_trials,
                output_trials=output_trials,
            )

        session_elapsed = time.perf_counter() - session_t0
        print(
            f"[{session_idx + 1:03d}/{len(files):03d}] {meta.session_name}: "
            f"{len(neural_trials)} trials, {neural_trials[0].shape[0]} neurons, "
            f"{session_elapsed:.2f}s"
        )

    dataset = {
        "neural": neural_all,
        "input": input_all,
        "output": output_all,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int16),
        "brain_regions": ["CA1"],
        "brain_region_idx": plane_all,
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "output_values": OUTPUT_VALUES,
        "metadata": {
            "task_description": (
                "Trial-aligned CA1 deconvolved activity in a 450 cm VR hidden-reward task; "
                "decoder inputs are trial time, environment, trial number, and previous-trial outcome; "
                "decoder outputs are reward-zone distance, absolute position, speed, lick, reward-zone identity, and reward outcome."
            ),
            "time_bin_size": time_bin_ms,
            "temporal_alignment_event": "trial start",
            "off_start": 0.0,
            "off_end": None,
            "brain_region_note": "All neurons are from hippocampal CA1; multi-plane identity is stored in session_info.",
            "neural_signal": "Deconvolved calcium events from NWB processing/ophys/Deconvolved.",
            "cell_curation": "Suite2p/manual curated cells only (iscell[:,0] == 1).",
            "reward_zone_locations_cm": {key: list(val) for key, val in REWARD_ZONE_COORDS_CM.items()},
            "reward_zone_switch_trial": SCENE_SWITCH_TRIAL,
            "lick_sensor_rule": {
                "count_threshold": LICK_SENSOR_COUNT_THRESHOLD,
                "bad_fraction_threshold": LICK_SENSOR_BAD_FRAC,
                "bad_trial_replacement": "binary lick set to 0 for the full trial",
            },
            "position_bin_rule": "[0,90), [90,180), [180,270), [270,360), [360,inf)",
            "speed_bin_rule": "<2, [2,10), [10,20), [20,40], >40",
            "reward_distance_bin_rule": (
                "signed distance to nearest point in active reward zone interval; "
                "bins: <-50, [-50,-10), [-10,0), 0, (0,10], (10,50], >50"
            ),
            "session_info": session_info,
            "source_data_root": "/app/data",
            "conversion_runtime_s": time.perf_counter() - t0,
        },
    }
    return dataset


def main() -> None:
    args = parse_args()
    data_root = Path("/app/data")
    files = sorted(data_root.glob("sub-*/sub-*_behavior+ophys.nwb"))
    if not files:
        raise FileNotFoundError("No NWB session files found under /app/data")

    if args.sample:
        files = files[:2]
        mode = "sample"
    else:
        mode = "full"

    print(f"Mode: {mode}")
    print(f"Sessions selected: {len(files)}")

    dataset = build_dataset(files, show_processing=args.show_processing)

    out_path = Path(args.outpicklefile)
    with out_path.open("wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_trials = sum(len(session) for session in dataset["neural"])
    total_neurons = sum(session[0].shape[0] for session in dataset["neural"])
    print(f"Wrote {out_path}")
    print(f"Subjects: {len(dataset['subjects'])}")
    print(f"Sessions: {len(dataset['neural'])}")
    print(f"Trials: {total_trials}")
    print(f"Session-neuron sum: {total_neurons}")
    print(f"time_bin_size_ms: {dataset['metadata']['time_bin_size']:.6f}")


if __name__ == "__main__":
    main()
