import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO


DATA_ROOT = Path("/app/data")
OUTPUT_PATH = Path("/app/converted_data.pkl")

FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
ZONE_STARTS_CM = np.array([80.0, 200.0, 320.0], dtype=np.float32)
ZONE_ENDS_CM = np.array([130.0, 250.0, 370.0], dtype=np.float32)
ZONE_CENTERS_CM = (ZONE_STARTS_CM + ZONE_ENDS_CM) / 2.0
ZONE_LABELS = ["A", "B", "C"]
TRACK_LENGTH_CM = 450.0
SWITCH_SPLIT_TRIAL = 30
INTERNEURON_SPEED_CORR_THR = 0.5


def _safe_mode(values: np.ndarray, fallback: int = 0) -> int:
    if values.size == 0:
        return fallback
    values = values.astype(int)
    counts = np.bincount(values)
    return int(np.argmax(counts))


def _prepare_trial_bounds(trial_start: np.ndarray, teleport: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start > 0)
    teleports = np.flatnonzero(teleport > 0)

    if teleports.size == starts.size + 1 and teleports[0] < starts[0]:
        teleports = teleports[1:]
    if starts.size == teleports.size + 1 and starts[-1] > teleports[-1]:
        starts = starts[:-1]

    n_pairs = min(starts.size, teleports.size)
    starts = starts[:n_pairs]
    teleports = teleports[:n_pairs]

    bounds = []
    for start, stop in zip(starts, teleports):
        if stop > start:
            bounds.append((int(start), int(stop)))
    return bounds


def _infer_reward_zone_by_trial(position: np.ndarray, reward_zone: np.ndarray, bounds: list[tuple[int, int]]) -> np.ndarray:
    raw_zone = np.full(len(bounds), -1, dtype=np.int64)
    for trial_idx, (start, stop) in enumerate(bounds):
        rz_pos = position[start:stop][reward_zone[start:stop] > 0]
        if rz_pos.size:
            raw_zone[trial_idx] = int(np.argmin(np.abs(ZONE_CENTERS_CM - np.median(rz_pos))))

    valid = raw_zone >= 0
    if not np.any(valid):
        raise RuntimeError("Could not infer reward zone from reward_zone events in any trial.")

    unique_valid = np.unique(raw_zone[valid])
    if unique_valid.size == 1:
        return np.full(len(bounds), int(unique_valid[0]), dtype=np.int64)

    pre_valid = raw_zone[:SWITCH_SPLIT_TRIAL][raw_zone[:SWITCH_SPLIT_TRIAL] >= 0]
    post_valid = raw_zone[SWITCH_SPLIT_TRIAL:][raw_zone[SWITCH_SPLIT_TRIAL:] >= 0]

    if pre_valid.size and post_valid.size:
        inferred = np.empty(len(bounds), dtype=np.int64)
        inferred[:SWITCH_SPLIT_TRIAL] = _safe_mode(pre_valid)
        inferred[SWITCH_SPLIT_TRIAL:] = _safe_mode(post_valid, fallback=int(inferred[SWITCH_SPLIT_TRIAL - 1]))
        return inferred

    # Fallback: nearest-neighbor fill when one side has no detectable zone-entry events.
    inferred = raw_zone.copy()
    valid_idx = np.flatnonzero(inferred >= 0)
    first_valid = valid_idx[0]
    inferred[:first_valid] = inferred[first_valid]
    for idx in range(first_valid + 1, len(inferred)):
        if inferred[idx] < 0:
            inferred[idx] = inferred[idx - 1]
    last_valid = valid_idx[-1]
    inferred[last_valid + 1:] = inferred[last_valid]
    return inferred


def _speed_corr_filter(deconv: np.ndarray, speed: np.ndarray, bounds: list[tuple[int, int]]) -> np.ndarray:
    track_mask = np.zeros(deconv.shape[0], dtype=bool)
    for start, stop in bounds:
        track_mask[start:stop] = True

    x = deconv[track_mask]
    y = speed[track_mask].astype(np.float32)
    y = np.nan_to_num(y, nan=0.0)
    y_centered = y - y.mean()
    y_ss = np.sum(y_centered * y_centered)
    if y_ss <= 0:
        return np.ones(deconv.shape[1], dtype=bool)

    x = np.nan_to_num(x.astype(np.float32), nan=0.0)
    x_centered = x - x.mean(axis=0, keepdims=True)
    x_ss = np.sum(x_centered * x_centered, axis=0)
    denom = np.sqrt(x_ss * y_ss)
    corr = np.zeros(deconv.shape[1], dtype=np.float32)
    nonzero = denom > 0
    corr[nonzero] = np.sum(x_centered[:, nonzero] * y_centered[:, None], axis=0) / denom[nonzero]
    return corr <= INTERNEURON_SPEED_CORR_THR


def _bin_distance_to_reward_zone(position_cm: np.ndarray, zone_idx: int) -> np.ndarray:
    start = ZONE_STARTS_CM[zone_idx]
    end = ZONE_ENDS_CM[zone_idx]
    dist = np.zeros_like(position_cm, dtype=np.float32)
    before = position_cm < start
    after = position_cm > end
    dist[before] = position_cm[before] - start
    dist[after] = position_cm[after] - end

    bins = np.empty(position_cm.shape[0], dtype=np.uint8)
    bins[dist < -50.0] = 0
    bins[(dist >= -50.0) & (dist < -10.0)] = 1
    bins[(dist >= -10.0) & (dist < 0.0)] = 2
    bins[dist == 0.0] = 3
    bins[(dist > 0.0) & (dist <= 10.0)] = 4
    bins[(dist > 10.0) & (dist <= 50.0)] = 5
    bins[dist > 50.0] = 6
    return bins


def _bin_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    bins = np.empty(position_cm.shape[0], dtype=np.uint8)
    bins[position_cm < 90.0] = 0
    bins[(position_cm >= 90.0) & (position_cm < 180.0)] = 1
    bins[(position_cm >= 180.0) & (position_cm < 270.0)] = 2
    bins[(position_cm >= 270.0) & (position_cm < 360.0)] = 3
    bins[position_cm >= 360.0] = 4
    return bins


def _bin_speed(speed_cm_s: np.ndarray) -> np.ndarray:
    bins = np.empty(speed_cm_s.shape[0], dtype=np.uint8)
    bins[speed_cm_s < 2.0] = 0
    bins[(speed_cm_s >= 2.0) & (speed_cm_s < 10.0)] = 1
    bins[(speed_cm_s >= 10.0) & (speed_cm_s < 20.0)] = 2
    bins[(speed_cm_s >= 20.0) & (speed_cm_s < 40.0)] = 3
    bins[speed_cm_s >= 40.0] = 4
    return bins


def _load_session(session_path: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, dict]:
    with NWBHDF5IO(str(session_path), "r", load_namespaces=True) as io:
        nwb = io.read()
        beh = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series
        seg = nwb.processing["ophys"].data_interfaces["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
        deconv_iface = nwb.processing["ophys"].data_interfaces["Deconvolved"]

        timestamps = np.asarray(beh["position"].timestamps[:], dtype=np.float64)
        position = np.asarray(beh["position"].data[:], dtype=np.float32)
        speed = np.asarray(beh["speed"].data[:], dtype=np.float32)
        lick = np.asarray(beh["lick"].data[:], dtype=np.float32)
        environment = np.asarray(beh["environment"].data[:], dtype=np.float32)
        reward_zone = np.asarray(beh["reward_zone"].data[:], dtype=np.float32)
        trial_number = np.asarray(beh["trial number"].data[:], dtype=np.float32)
        trial_start = np.asarray(beh["trial_start"].data[:], dtype=np.float32)
        teleport = np.asarray(beh["teleport"].data[:], dtype=np.float32)
        reward_timestamps = np.asarray(beh["Reward"].timestamps[:], dtype=np.float64)

        iscell_all = np.asarray(seg["iscell"].data[:])[:, 0].astype(bool)
        plane_idx_all = np.asarray(seg["planeIdx"].data[:], dtype=np.int64)
        deconv_parts = []
        plane_parts = []
        for series_name, rr in sorted(
            deconv_iface.roi_response_series.items(),
            key=lambda item: int(item[0].replace("plane", "")),
        ):
            plane_num = int(series_name.replace("plane", ""))
            plane_mask = plane_idx_all == plane_num
            if rr.data.shape[1] != int(np.sum(plane_mask)):
                raise RuntimeError(
                    f"{session_path.name}: plane {plane_num} has {rr.data.shape[1]} ROIs in Deconvolved "
                    f"but {int(np.sum(plane_mask))} entries in PlaneSegmentation."
                )
            keep_plane = iscell_all[plane_mask]
            deconv_parts.append(np.asarray(rr.data[:, :], dtype=np.float32)[:, keep_plane])
            plane_parts.append(np.full(int(np.sum(keep_plane)), plane_num, dtype=np.int64))

        deconv = np.concatenate(deconv_parts, axis=1)
        plane_idx = np.concatenate(plane_parts)
        n_frames = min(
            timestamps.size,
            position.size,
            speed.size,
            lick.size,
            environment.size,
            reward_zone.size,
            trial_number.size,
            trial_start.size,
            teleport.size,
            deconv.shape[0],
        )
        timestamps = timestamps[:n_frames]
        position = position[:n_frames]
        speed = speed[:n_frames]
        lick = lick[:n_frames]
        environment = environment[:n_frames]
        reward_zone = reward_zone[:n_frames]
        trial_number = trial_number[:n_frames]
        trial_start = trial_start[:n_frames]
        teleport = teleport[:n_frames]
        deconv = deconv[:n_frames]

        bounds = _prepare_trial_bounds(trial_start, teleport)
        if len(bounds) < 2:
            raise RuntimeError(f"{session_path.name}: fewer than two valid trials after trial_start/teleport pairing.")

        zone_by_trial = _infer_reward_zone_by_trial(position, reward_zone, bounds)

        non_interneuron = _speed_corr_filter(deconv, speed, bounds)
        deconv = deconv[:, non_interneuron]
        plane_idx = plane_idx[non_interneuron]
        if deconv.shape[1] == 0:
            raise RuntimeError(f"{session_path.name}: no neurons remain after curation filters.")

        neural_trials = []
        input_trials = []
        output_trials = []

        reward_outcome_by_trial = np.zeros(len(bounds), dtype=np.uint8)
        for trial_idx, (start, stop) in enumerate(bounds):
            trial_start_t = timestamps[start]
            trial_end_t = timestamps[stop]
            reward_outcome_by_trial[trial_idx] = np.uint8(np.any((reward_timestamps >= trial_start_t) & (reward_timestamps < trial_end_t)))

        for trial_idx, (start, stop) in enumerate(bounds):
            pos = np.clip(position[start:stop], 0.0, TRACK_LENGTH_CM).astype(np.float32, copy=False)
            spd = np.maximum(speed[start:stop], 0.0).astype(np.float32, copy=False)
            lick_bin = (lick[start:stop] > 0).astype(np.uint8, copy=False)
            env_vals = environment[start:stop]
            env_vals = env_vals[env_vals >= 0]
            env = float(_safe_mode(env_vals.astype(int), fallback=0))
            tnum = float(trial_number[start]) if trial_number[start] >= 0 else float(trial_idx)
            prev_reward = float(reward_outcome_by_trial[trial_idx - 1]) if trial_idx > 0 else 0.0
            zone_idx = int(zone_by_trial[trial_idx])
            reward_outcome = int(reward_outcome_by_trial[trial_idx])
            time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)

            neural_trial = np.ascontiguousarray(deconv[start:stop].T, dtype=np.float16)
            T = neural_trial.shape[1]

            input_trial = np.vstack([
                time_from_start,
                np.full(T, env, dtype=np.float32),
                np.full(T, tnum, dtype=np.float32),
                np.full(T, prev_reward, dtype=np.float32),
            ]).astype(np.float32, copy=False)

            output_trial = np.vstack([
                _bin_distance_to_reward_zone(pos, zone_idx),
                _bin_absolute_position(pos),
                _bin_speed(spd),
                lick_bin,
                np.full(T, zone_idx, dtype=np.uint8),
                np.full(T, reward_outcome, dtype=np.uint8),
            ]).astype(np.uint8, copy=False)

            neural_trials.append(neural_trial)
            input_trials.append(input_trial)
            output_trials.append(output_trial)

        session_info = {
            "file": str(session_path),
            "identifier": str(nwb.identifier),
            "subject": str(nwb.subject.subject_id),
            "n_trials": len(bounds),
            "n_neurons": int(deconv.shape[1]),
            "n_curated_iscell": int(np.sum(iscell_all)),
            "n_excluded_speed_correlated": int(np.sum(~non_interneuron)),
            "reward_zone_labels": [ZONE_LABELS[idx] for idx in zone_by_trial.tolist()],
            "environments": sorted({int(_safe_mode(environment[start:stop][environment[start:stop] >= 0].astype(int), fallback=0)) for start, stop in bounds}),
        }

    return neural_trials, input_trials, output_trials, plane_idx, session_info


def build_dataset() -> dict:
    session_paths = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
    if not session_paths:
        raise RuntimeError(f"No NWB files found under {DATA_ROOT}")

    subjects = []
    subject_to_idx = {}

    data = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": subjects,
        "subject_idx": [],
        "brain_regions": ["CA1"],
        "brain_region_idx": [],
        "input_names": [
            "time_from_trial_start_s",
            "environment",
            "trial_number",
            "previous_trial_reward_outcome",
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
            ["lt_-50cm", "-50_to_-10cm", "-10_to_0cm", "0cm", "0_to_10cm", "10_to_50cm", "gt_50cm"],
            ["0_90cm", "90_180cm", "180_270cm", "270_360cm", "360_450cm"],
            ["lt_2cm_s", "2_to_10cm_s", "10_to_20cm_s", "20_to_40cm_s", "gt_40cm_s"],
            ["no", "yes"],
            ["A", "B", "C"],
            ["no", "yes"],
        ],
        "metadata": {
            "task_description": (
                "Two-photon CA1 imaging during a 450 cm virtual linear-track task with hidden reward zones "
                "that switch across blocks and environments; decoder predicts behavioral and task variables "
                "from deconvolved neural activity."
            ),
            "time_bin_size": TIME_BIN_MS,
            "frame_rate_hz": FRAME_RATE_HZ,
            "temporal_alignment_event": "trial start (entry to the 0 cm start of the virtual track)",
            "off_start": 0.0,
            "off_end": None,
            "track_length_cm": TRACK_LENGTH_CM,
            "reward_zone_starts_cm": ZONE_STARTS_CM.tolist(),
            "reward_zone_ends_cm": ZONE_ENDS_CM.tolist(),
            "reward_zone_labels": ZONE_LABELS,
            "trial_slicing": "Frames from trial_start inclusive to teleport exclusive, matching keep_teleports=False preprocessing.",
            "neural_signal": "Suite2p deconvolved activity from NWB, filtered to iscell ROIs.",
            "interneuron_filter": (
                "Excluded curated ROIs with deconvolved activity-speed correlation > 0.5 on on-track frames "
                "as a practical approximation to the paper's speed-correlated interneuron screen."
            ),
            "session_info": [],
            "source_paper": "Sosa, Plitt, Giocomo. A flexible hippocampal population code for experience relative to reward.",
        },
    }

    for session_idx, session_path in enumerate(session_paths, start=1):
        subject = session_path.parent.name.replace("sub-", "")
        if subject not in subject_to_idx:
            subject_to_idx[subject] = len(subjects)
            subjects.append(subject)

        neural_trials, input_trials, output_trials, plane_idx, session_info = _load_session(session_path)

        data["neural"].append(neural_trials)
        data["input"].append(input_trials)
        data["output"].append(output_trials)
        data["subject_idx"].append(subject_to_idx[subject])
        data["brain_region_idx"].append(np.zeros(plane_idx.shape[0], dtype=np.int64))
        data["metadata"]["session_info"].append(session_info)
        if session_idx % 10 == 0 or session_idx == len(session_paths):
            print(f"processed {session_idx}/{len(session_paths)} sessions", flush=True)

    data["subject_idx"] = np.asarray(data["subject_idx"], dtype=np.int64)
    return data


def print_summary(data: dict) -> None:
    n_sessions = len(data["neural"])
    n_trials = sum(len(sess) for sess in data["neural"])
    n_neurons = [sess[0].shape[0] for sess in data["neural"]]
    trial_lengths = [trial.shape[1] for sess in data["neural"] for trial in sess]
    excluded = [info["n_excluded_speed_correlated"] for info in data["metadata"]["session_info"]]
    env_counter = Counter()
    zone_counter = Counter()
    reward_counter = Counter()

    for sess_out in data["output"]:
        for trial_out in sess_out:
            env_counter.update()
            zone_counter[int(trial_out[4, 0])] += 1
            reward_counter[int(trial_out[5, 0])] += 1

    print(f"sessions: {n_sessions}")
    print(f"trials: {n_trials}")
    print(f"subjects: {len(data['subjects'])} -> {data['subjects']}")
    print(f"neurons/session min-median-max: {min(n_neurons)} / {int(np.median(n_neurons))} / {max(n_neurons)}")
    print(f"trial length min-median-max: {min(trial_lengths)} / {int(np.median(trial_lengths))} / {max(trial_lengths)} frames")
    print(f"speed-correlated exclusions total: {sum(excluded)}")
    print(f"reward-zone trial counts: {dict(sorted(zone_counter.items()))}")
    print(f"reward-outcome trial counts: {dict(sorted(reward_counter.items()))}")


def main() -> None:
    data = build_dataset()
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print_summary(data)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
