#!/usr/bin/env python3

import argparse
import glob
import math
import os
import pickle
import time
from dataclasses import dataclass

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TARGET_RATE_HZ = 15.5078125
TARGET_DT_S = 1.0 / TARGET_RATE_HZ
TRACK_LENGTH_CM = 450.0
SWITCH_TRIAL_INDEX = 30
LICK_ARTIFACT_FRACTION = 0.35  # Matches glmUtils.get_timeseries_data in the reference code.

ZONE_BOUNDS = {
    0: (80.0, 130.0),   # A
    1: (200.0, 250.0),  # B
    2: (320.0, 370.0),  # C
}
ZONE_NAMES = ["A", "B", "C"]
ZONE_CENTERS = np.array([(lo + hi) / 2.0 for lo, hi in ZONE_BOUNDS.values()], dtype=np.float32)

INPUT_NAMES = [
    "time_from_trial_start_s",
    "environment",
    "trial_number",
    "previous_trial_rewarded",
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
    ["lt_-50", "-50_to_-10", "-10_to_lt_0", "0", "gt_0_to_10", "10_to_50", "gt_50"],
    ["bin0", "bin1", "bin2", "bin3", "bin4"],
    ["lt_2", "2_to_10", "10_to_20", "20_to_40", "gt_40"],
    ["no", "yes"],
    ["A", "B", "C"],
    ["omitted", "rewarded"],
]


@dataclass
class SessionArrays:
    path: str
    subject: str
    session_label: str
    rate_hz: float
    timestamps: np.ndarray
    position: np.ndarray
    speed: np.ndarray
    lick: np.ndarray
    environment: np.ndarray
    reward_zone_signal: np.ndarray
    scanning: np.ndarray
    trial_number: np.ndarray
    trial_start: np.ndarray
    teleport: np.ndarray
    reward_times: np.ndarray
    neural: np.ndarray  # (time, neurons)
    roi_ids: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the Sosa et al. NWB archive into the decoder pickle format."
    )
    parser.add_argument("outpicklefile", type=str, help="Output pickle file path.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only 2 representative sessions.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing visualizations for up to 2 sessions.",
    )
    return parser.parse_args()


def list_session_files() -> list[str]:
    files = sorted(glob.glob(os.path.join("data", "sub-*", "sub-*_behavior+ophys.nwb")))
    if not files:
        raise FileNotFoundError("No NWB files found under data/sub-*/")
    return files


def decode_if_bytes(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode_if_bytes(value[()])
    return str(value)


def mode_int(values: np.ndarray, default: int = 0) -> int:
    if values.size == 0:
        return default
    values = values.astype(np.int64, copy=False)
    values = values[values >= 0]
    if values.size == 0:
        return default
    return int(np.bincount(values).argmax())


def infer_sample_files(files: list[str]) -> list[str]:
    metadata = []
    for path in files:
        with h5py.File(path, "r") as f:
            env = f["processing/behavior/BehavioralTimeSeries/environment/data"][()]
            rate = float(f["processing/ophys/Deconvolved/plane0/starting_time"].attrs["rate"])
            env_valid = tuple(int(v) for v in np.unique(env) if v >= 0)
            metadata.append((path, rate, env_valid))

    chosen = []
    for want_rate in (TARGET_RATE_HZ, 2.0 * TARGET_RATE_HZ):
        candidates = [m[0] for m in metadata if math.isclose(m[1], want_rate) and len(m[2]) > 1]
        if not candidates:
            candidates = [m[0] for m in metadata if math.isclose(m[1], want_rate)]
        if candidates:
            chosen.append(candidates[0])

    if len(chosen) < 2:
        for path, _, _ in metadata:
            if path not in chosen:
                chosen.append(path)
            if len(chosen) == 2:
                break

    return chosen[:2]


def select_session_files(files: list[str], sample_mode: bool) -> list[str]:
    if not sample_mode:
        return files
    chosen = infer_sample_files(files)
    print("Sample mode selected files:")
    for path in chosen:
        print(f"  - {path}")
    return chosen


def rebin_1d_mean(x: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return x.astype(np.float32, copy=False)
    n_full = x.shape[0] // factor
    out = []
    if n_full:
        out.append(x[: n_full * factor].reshape(n_full, factor).mean(axis=1))
    if x.shape[0] % factor:
        out.append(np.array([x[n_full * factor :].mean()], dtype=np.float32))
    return np.concatenate(out).astype(np.float32, copy=False)


def rebin_1d_any(x: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return x.astype(np.int64, copy=False)
    n_full = x.shape[0] // factor
    out = []
    if n_full:
        out.append(np.any(x[: n_full * factor].reshape(n_full, factor) > 0, axis=1))
    if x.shape[0] % factor:
        out.append(np.array([np.any(x[n_full * factor :] > 0)], dtype=bool))
    return np.concatenate(out).astype(np.int64, copy=False)


def rebin_2d_sum_time_last(x: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return x.astype(np.float32, copy=False)
    t = x.shape[1]
    n_full = t // factor
    parts = []
    if n_full:
        full = x[:, : n_full * factor].reshape(x.shape[0], n_full, factor).sum(axis=2)
        parts.append(full)
    if t % factor:
        parts.append(x[:, n_full * factor :].sum(axis=1, keepdims=True))
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def discretize_distance(distance_cm: np.ndarray) -> np.ndarray:
    out = np.zeros(distance_cm.shape, dtype=np.int64)
    out[distance_cm < -50.0] = 0
    out[(distance_cm >= -50.0) & (distance_cm < -10.0)] = 1
    out[(distance_cm >= -10.0) & (distance_cm < 0.0)] = 2
    out[distance_cm == 0.0] = 3
    out[(distance_cm > 0.0) & (distance_cm <= 10.0)] = 4
    out[(distance_cm > 10.0) & (distance_cm <= 50.0)] = 5
    out[distance_cm > 50.0] = 6
    return out


def discretize_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, TRACK_LENGTH_CM - 1e-6)
    return np.minimum((clipped / (TRACK_LENGTH_CM / 5.0)).astype(np.int64), 4)


def discretize_speed(speed_cm_s: np.ndarray) -> np.ndarray:
    out = np.zeros(speed_cm_s.shape, dtype=np.int64)
    out[(speed_cm_s >= 2.0) & (speed_cm_s < 10.0)] = 1
    out[(speed_cm_s >= 10.0) & (speed_cm_s < 20.0)] = 2
    out[(speed_cm_s >= 20.0) & (speed_cm_s < 40.0)] = 3
    out[speed_cm_s >= 40.0] = 4
    return out


def compute_distance_to_zone(position_cm: np.ndarray, zone_idx: int) -> np.ndarray:
    zone_start, zone_end = ZONE_BOUNDS[zone_idx]
    distance = np.zeros(position_cm.shape, dtype=np.float32)
    before = position_cm < zone_start
    after = position_cm > zone_end
    distance[before] = position_cm[before] - zone_start
    distance[after] = position_cm[after] - zone_end
    return distance


def trial_majority_zone(observed_zone: np.ndarray, start: int, stop: int, fallback: int) -> int:
    subset = observed_zone[start:stop]
    subset = subset[subset >= 0]
    if subset.size == 0:
        return fallback
    return int(np.bincount(subset).argmax())


def infer_trial_zones(
    positions: np.ndarray,
    reward_zone_signal: np.ndarray,
    trial_slices: list[tuple[int, int]],
    trial_rewarded: np.ndarray,
    reward_frame_idx: np.ndarray,
    trial_by_frame: np.ndarray,
    trial_env: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_trials = len(trial_slices)
    observed = np.full(n_trials, -1, dtype=np.int64)
    observed_position = np.full(n_trials, np.nan, dtype=np.float32)

    reward_pos_by_trial: dict[int, float] = {}
    for frame_idx in reward_frame_idx:
        if frame_idx < 0 or frame_idx >= trial_by_frame.shape[0]:
            continue
        trial_id = int(trial_by_frame[frame_idx])
        if trial_id < 0:
            continue
        reward_pos_by_trial.setdefault(trial_id, float(positions[frame_idx]))

    for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
        mask = reward_zone_signal[start:stop_exclusive] > 0
        if np.any(mask):
            zone_pos = float(np.nanmedian(positions[start:stop_exclusive][mask]))
        elif trial_id in reward_pos_by_trial:
            zone_pos = reward_pos_by_trial[trial_id]
        else:
            zone_pos = np.nan

        if np.isfinite(zone_pos):
            observed_position[trial_id] = zone_pos
            observed[trial_id] = int(np.argmin(np.abs(ZONE_CENTERS - zone_pos)))

    filled = observed.copy()
    unique_observed = np.unique(observed[observed >= 0])
    unique_env = np.unique(trial_env)

    if unique_observed.size == 0:
        raise RuntimeError("Could not infer any reward-zone labels from the session.")

    if unique_observed.size == 1:
        filled[filled < 0] = int(unique_observed[0])
        return filled, observed_position

    if unique_observed.size > 2:
        raise RuntimeError(f"Observed more than two reward zones within one session: {unique_observed.tolist()}")

    split = min(SWITCH_TRIAL_INDEX, n_trials)
    pre_majority = trial_majority_zone(observed, 0, split, fallback=int(unique_observed[0]))
    post_fallback = int(unique_observed[1] if unique_observed.size > 1 else pre_majority)
    post_majority = trial_majority_zone(observed, split, n_trials, fallback=post_fallback)
    filled[:split][filled[:split] < 0] = pre_majority
    filled[split:][filled[split:] < 0] = post_majority

    if unique_env.size > 1 and split < n_trials:
        filled[:split] = pre_majority
        filled[split:] = post_majority

    return filled, observed_position


def build_trial_slices(trial_start_signal: np.ndarray, teleport_signal: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start_signal > 0.5)
    teleports = np.flatnonzero(teleport_signal > 0.5)
    if starts.size == 0 or teleports.size == 0:
        raise RuntimeError("Missing trial_start or teleport markers.")
    n = min(starts.size, teleports.size)
    slices: list[tuple[int, int]] = []
    for start, stop in zip(starts[:n], teleports[:n]):
        if stop < start:
            continue
        slices.append((int(start), int(stop) + 1))
    if len(slices) < 2:
        raise RuntimeError("Need at least two trials in a session.")
    return slices


def load_session_arrays(path: str) -> SessionArrays:
    with h5py.File(path, "r") as f:
        subject = decode_if_bytes(f["general/subject/subject_id"][()])
        session_id = decode_if_bytes(f["general/session_id"][()])
        session_label = f"{subject}_ses-{session_id}"

        beh = f["processing/behavior/BehavioralTimeSeries"]
        neural_group = f["processing/ophys/Deconvolved/plane0"]
        seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]

        rate_hz = float(neural_group["starting_time"].attrs["rate"])
        timestamps = beh["position/timestamps"][()].astype(np.float64, copy=False)

        neural = neural_group["data"][()].astype(np.float32, copy=False)
        roi_ids = neural_group["rois"][()].astype(np.int64, copy=False)
        iscell = seg["iscell"][()][roi_ids, 0] > 0.5
        neural = neural[:, iscell]
        roi_ids = roi_ids[iscell]

        t_neural = neural.shape[0]
        t_behavior = beh["position/data"].shape[0]
        t_common = min(t_neural, t_behavior)

        return SessionArrays(
            path=path,
            subject=subject,
            session_label=session_label,
            rate_hz=rate_hz,
            timestamps=timestamps[:t_common],
            position=beh["position/data"][()][:t_common].astype(np.float32, copy=False),
            speed=beh["speed/data"][()][:t_common].astype(np.float32, copy=False),
            lick=beh["lick/data"][()][:t_common].astype(np.float32, copy=False),
            environment=beh["environment/data"][()][:t_common].astype(np.float32, copy=False),
            reward_zone_signal=beh["reward_zone/data"][()][:t_common].astype(np.float32, copy=False),
            scanning=beh["scanning/data"][()][:t_common].astype(np.float32, copy=False),
            trial_number=beh["trial number/data"][()][:t_common].astype(np.float32, copy=False),
            trial_start=beh["trial_start/data"][()][:t_common].astype(np.float32, copy=False),
            teleport=beh["teleport/data"][()][:t_common].astype(np.float32, copy=False),
            reward_times=beh["Reward/timestamps"][()].astype(np.float64, copy=False),
            neural=neural[:t_common],
            roi_ids=roi_ids,
        )


def reward_frames_for_session(arrays: SessionArrays) -> np.ndarray:
    if arrays.reward_times.size == 0:
        return np.empty((0,), dtype=np.int64)
    idx = np.searchsorted(arrays.timestamps, arrays.reward_times, side="left")
    return np.clip(idx, 0, arrays.timestamps.shape[0] - 1).astype(np.int64, copy=False)


def plot_processing_summary(
    session_label: str,
    trial_env: np.ndarray,
    trial_zone: np.ndarray,
    trial_rewarded: np.ndarray,
    example_position: np.ndarray,
    example_speed: np.ndarray,
    example_lick: np.ndarray,
    example_output: np.ndarray,
    example_neural: np.ndarray,
    original_rate_hz: float,
    output_path: str,
) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(16, 18), constrained_layout=True)

    trials = np.arange(trial_env.shape[0])
    axes[0].step(trials, trial_env, where="mid", label="environment")
    axes[0].step(trials, trial_zone + 0.05, where="mid", label="reward zone")
    axes[0].step(trials, trial_rewarded + 0.1, where="mid", label="rewarded")
    axes[0].axvline(SWITCH_TRIAL_INDEX - 0.5, color="k", linestyle="--", alpha=0.4)
    axes[0].set_title(f"{session_label}: per-trial state (original rate {original_rate_hz:.6f} Hz)")
    axes[0].set_ylabel("state")
    axes[0].legend(loc="upper right")

    axes[1].plot(example_position, label="position (cm)")
    axes[1].plot(example_speed, label="speed (cm/s)")
    axes[1].plot(example_lick * 20.0, label="lick x20")
    axes[1].set_title("Example trial continuous variables after alignment / rebinning")
    axes[1].legend(loc="upper right")

    im = axes[2].imshow(example_output, aspect="auto", interpolation="nearest")
    axes[2].set_yticks(np.arange(len(OUTPUT_NAMES)))
    axes[2].set_yticklabels(OUTPUT_NAMES)
    axes[2].set_title("Example trial discretized outputs")
    fig.colorbar(im, ax=axes[2], fraction=0.02, pad=0.01)

    nplot = min(40, example_neural.shape[0])
    axes[3].imshow(example_neural[:nplot], aspect="auto", interpolation="nearest")
    axes[3].set_ylabel("neuron")
    axes[3].set_title("Example trial neural activity")

    axes[4].hist(trial_zone, bins=np.arange(5) - 0.5, rwidth=0.8)
    axes[4].set_xticks(np.arange(3))
    axes[4].set_xticklabels(ZONE_NAMES)
    axes[4].set_title("Reward-zone distribution across trials")
    axes[4].set_xlabel("zone")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def convert_session(
    arrays: SessionArrays,
    show_processing: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, dict]:
    t0 = time.perf_counter()

    trial_slices = build_trial_slices(arrays.trial_start, arrays.teleport)
    reward_frame_idx = reward_frames_for_session(arrays)
    rewarded_trial_ids = set(int(t) for t in arrays.trial_number[reward_frame_idx] if t >= 0)

    trial_rewarded = np.array(
        [1 if trial_id in rewarded_trial_ids else 0 for trial_id in range(len(trial_slices))],
        dtype=np.int64,
    )
    trial_prev_rewarded = np.concatenate([[0], trial_rewarded[:-1]]).astype(np.float32, copy=False)

    trial_env = np.zeros((len(trial_slices),), dtype=np.int64)
    lick_artifact = np.zeros((len(trial_slices),), dtype=bool)
    for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
        env = arrays.environment[start:stop_exclusive]
        env = env[env >= 0]
        trial_env[trial_id] = mode_int(env, default=0)

        lick_trial = arrays.lick[start:stop_exclusive]
        if lick_trial.size and np.mean(lick_trial > 2) > LICK_ARTIFACT_FRACTION:
            lick_artifact[trial_id] = True

    trial_zone, observed_zone_position = infer_trial_zones(
        positions=np.clip(arrays.position, 0.0, TRACK_LENGTH_CM),
        reward_zone_signal=arrays.reward_zone_signal,
        trial_slices=trial_slices,
        trial_rewarded=trial_rewarded,
        reward_frame_idx=reward_frame_idx,
        trial_by_frame=arrays.trial_number.astype(np.int64, copy=False),
        trial_env=trial_env,
    )

    if math.isclose(arrays.rate_hz, TARGET_RATE_HZ):
        factor = 1
    elif math.isclose(arrays.rate_hz, 2.0 * TARGET_RATE_HZ):
        factor = 2
    else:
        raise RuntimeError(f"Unexpected sampling rate {arrays.rate_hz} in {arrays.session_label}")

    neural_trials: list[np.ndarray] = []
    input_trials: list[np.ndarray] = []
    output_trials: list[np.ndarray] = []

    for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
        if lick_artifact[trial_id]:
            continue

        neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
        position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
        speed = np.clip(arrays.speed[start:stop_exclusive], 0.0, None)
        lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)

        if factor == 2:
            neural_trial = rebin_2d_sum_time_last(neural_trial, factor)
            position = rebin_1d_mean(position, factor)
            speed = rebin_1d_mean(speed, factor)
            lick_binary = rebin_1d_any(lick_binary, factor)
        else:
            position = position.astype(np.float32, copy=False)
            speed = speed.astype(np.float32, copy=False)
            lick_binary = lick_binary.astype(np.int64, copy=False)

        t_bins = neural_trial.shape[1]
        time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
        env_series = np.full((t_bins,), float(trial_env[trial_id]), dtype=np.float32)
        trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
        prev_reward_series = np.full((t_bins,), float(trial_prev_rewarded[trial_id]), dtype=np.float32)

        zone_idx = int(trial_zone[trial_id])
        dist = compute_distance_to_zone(position, zone_idx)
        dist_bin = discretize_distance(dist)
        pos_bin = discretize_position(position)
        speed_bin = discretize_speed(speed)
        zone_bin = np.full((t_bins,), zone_idx, dtype=np.int64)
        reward_bin = np.full((t_bins,), int(trial_rewarded[trial_id]), dtype=np.int64)

        input_trial = np.stack(
            [time_from_start, env_series, trial_series, prev_reward_series],
            axis=0,
        ).astype(np.float32, copy=False)
        output_trial = np.stack(
            [dist_bin, pos_bin, speed_bin, lick_binary, zone_bin, reward_bin],
            axis=0,
        ).astype(np.int64, copy=False)

        neural_trials.append(neural_trial.astype(np.float32, copy=False))
        input_trials.append(input_trial)
        output_trials.append(output_trial)

    if len(neural_trials) < 2:
        raise RuntimeError(f"{arrays.session_label}: fewer than 2 valid trials after filtering.")

    session_info = {
        "session_label": arrays.session_label,
        "source_file": arrays.path,
        "subject": arrays.subject,
        "original_rate_hz": arrays.rate_hz,
        "rebin_factor": factor,
        "n_trials_raw": len(trial_slices),
        "n_trials_kept": len(neural_trials),
        "n_trials_lick_artifact_removed": int(lick_artifact.sum()),
        "n_neurons_kept": int(arrays.neural.shape[1]),
        "observed_zone_positions_cm": observed_zone_position.tolist(),
        "conversion_seconds": time.perf_counter() - t0,
    }

    if show_processing:
        example_idx = min(5, len(neural_trials) - 1)
        plot_processing_summary(
            session_label=arrays.session_label,
            trial_env=trial_env,
            trial_zone=trial_zone,
            trial_rewarded=trial_rewarded,
            example_position=np.clip(arrays.position[trial_slices[example_idx][0] : trial_slices[example_idx][1]], 0.0, TRACK_LENGTH_CM)
            if factor == 1
            else rebin_1d_mean(
                np.clip(arrays.position[trial_slices[example_idx][0] : trial_slices[example_idx][1]], 0.0, TRACK_LENGTH_CM),
                factor,
            ),
            example_speed=np.clip(arrays.speed[trial_slices[example_idx][0] : trial_slices[example_idx][1]], 0.0, None)
            if factor == 1
            else rebin_1d_mean(
                np.clip(arrays.speed[trial_slices[example_idx][0] : trial_slices[example_idx][1]], 0.0, None),
                factor,
            ),
            example_lick=(arrays.lick[trial_slices[example_idx][0] : trial_slices[example_idx][1]] > 0).astype(np.int64)
            if factor == 1
            else rebin_1d_any(
                (arrays.lick[trial_slices[example_idx][0] : trial_slices[example_idx][1]] > 0).astype(np.int64),
                factor,
            ),
            example_output=output_trials[example_idx],
            example_neural=neural_trials[example_idx],
            original_rate_hz=arrays.rate_hz,
            output_path=f"processing_{arrays.session_label}.png",
        )

    brain_region_idx = np.zeros((arrays.neural.shape[1],), dtype=np.int64)
    return neural_trials, input_trials, output_trials, brain_region_idx, session_info


def build_metadata(session_infos: list[dict]) -> dict:
    return {
        "task_description": (
            "Trial-aligned hippocampal CA1 calcium-event decoding during VR navigation with "
            "hidden reward zones that change across sessions and within switch sessions."
        ),
        "time_bin_size": 1000.0 * TARGET_DT_S,
        "temporal_alignment_event": "trial_start",
        "off_start": 0.0,
        "off_end": None,
        "source_dataset": "DANDI 001361 - A flexible hippocampal population code for experience relative to reward",
        "common_sampling_rate_hz": TARGET_RATE_HZ,
        "reward_zone_bounds_cm": {ZONE_NAMES[i]: list(bounds) for i, bounds in ZONE_BOUNDS.items()},
        "switch_trial_index": SWITCH_TRIAL_INDEX,
        "lick_artifact_fraction_threshold": LICK_ARTIFACT_FRACTION,
        "session_info": session_infos,
    }


def main() -> None:
    args = parse_args()
    t_all = time.perf_counter()

    files = list_session_files()
    if not args.sample and not args.full:
        args.full = True
    files = select_session_files(files, sample_mode=args.sample)

    all_subjects = []
    subject_to_idx = {}

    data = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": [],
        "subject_idx": [],
        "brain_regions": ["CA1"],
        "brain_region_idx": [],
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "output_values": OUTPUT_VALUES,
        "metadata": {},
    }

    session_infos = []

    for session_idx, path in enumerate(files):
        session_t0 = time.perf_counter()
        arrays = load_session_arrays(path)
        print(
            f"[{session_idx + 1}/{len(files)}] {arrays.session_label}: "
            f"{arrays.neural.shape[0]} frames, {arrays.neural.shape[1]} curated neurons, rate={arrays.rate_hz:.6f} Hz"
        )

        neural_trials, input_trials, output_trials, brain_region_idx, session_info = convert_session(
            arrays=arrays,
            show_processing=args.show_processing and session_idx < 2,
        )

        if arrays.subject not in subject_to_idx:
            subject_to_idx[arrays.subject] = len(all_subjects)
            all_subjects.append(arrays.subject)

        data["neural"].append(neural_trials)
        data["input"].append(input_trials)
        data["output"].append(output_trials)
        data["subject_idx"].append(subject_to_idx[arrays.subject])
        data["brain_region_idx"].append(brain_region_idx)
        session_infos.append(session_info)

        elapsed = time.perf_counter() - session_t0
        mean_t = float(np.mean([trial.shape[1] for trial in neural_trials]))
        print(
            f"    kept {len(neural_trials)} trials, mean trial bins={mean_t:.2f}, "
            f"session conversion time={elapsed:.2f}s"
        )

    data["subjects"] = all_subjects
    data["subject_idx"] = np.asarray(data["subject_idx"], dtype=np.int64)
    data["metadata"] = build_metadata(session_infos)

    with open(args.outpicklefile, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_elapsed = time.perf_counter() - t_all
    print(f"Saved converted dataset to {args.outpicklefile}")
    print(f"Processed {len(files)} sessions in {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
