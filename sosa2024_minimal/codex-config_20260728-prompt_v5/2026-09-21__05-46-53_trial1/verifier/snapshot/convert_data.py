import math
import pickle
import re
from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO


DATA_ROOT = Path("/app/data")
OUTPUT_PATH = Path("/app/converted_data.pkl")

TRACK_LENGTH_CM = 450.0
RZ_BOUNDS = {
    "A": (80.0, 130.0),
    "B": (200.0, 250.0),
    "C": (320.0, 370.0),
}
RZ_CENTERS = {k: 0.5 * (v[0] + v[1]) for k, v in RZ_BOUNDS.items()}
SWITCH_TRIAL = 30
LICK_ERROR_THRESHOLD = 0.35

# The NWB streams are aligned at ~15.5 Hz. Aggregating 16 imaging frames per bin
# keeps all modalities aligned while producing a dataset the shared decoder can
# train on end-to-end.
BIN_FRAMES = 16


def parse_scene(scene):
    match = re.fullmatch(r"(Env[12])_Location([ABC])", scene)
    if match:
        env, zone = match.groups()
        return (env,), (zone,)

    match = re.fullmatch(r"(Env[12])_Location([ABC])_to_([ABC])", scene)
    if match:
        env, zone0, zone1 = match.groups()
        return (env, env), (zone0, zone1)

    match = re.fullmatch(r"(Env[12])_([ABC])_to_(Env[12])_([ABC])", scene)
    if match:
        env0, zone0, env1, zone1 = match.groups()
        return (env0, env1), (zone0, zone1)

    raise ValueError(f"Unrecognized scene format: {scene}")


def expected_trial_labels(scene, ntrials):
    envs, zones = parse_scene(scene)
    env_by_trial = []
    zone_by_trial = []
    if len(envs) == 1:
        env_by_trial = [envs[0]] * ntrials
        zone_by_trial = [zones[0]] * ntrials
    else:
        split = min(SWITCH_TRIAL, ntrials)
        env_by_trial = [envs[0]] * split + [envs[1]] * (ntrials - split)
        zone_by_trial = [zones[0]] * split + [zones[1]] * (ntrials - split)
    return env_by_trial, zone_by_trial


def zone_from_position(position_cm):
    best_zone = min(
        RZ_CENTERS,
        key=lambda zone: abs(position_cm - RZ_CENTERS[zone]),
    )
    return best_zone


def get_iscell_mask(plane_segmentation):
    iscell = np.asarray(plane_segmentation.to_dataframe()["iscell"].tolist())
    if iscell.ndim == 1:
        keep = iscell.astype(float) > 0
    else:
        keep = iscell[:, 0].astype(float) > 0
    return keep


def load_curated_deconvolved_activity(ophys_interfaces):
    deconv_name = next(name for name in ophys_interfaces.keys() if "Deconvolved" in name)
    deconv_iface = ophys_interfaces[deconv_name]
    plane_seg = ophys_interfaces["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
    keep_cells = get_iscell_mask(plane_seg)

    parts = []
    for plane_name in sorted(deconv_iface.roi_response_series.keys()):
        rrs = deconv_iface.roi_response_series[plane_name]
        roi_indices = np.asarray(rrs.rois.data[:], dtype=int)
        plane_data = np.asarray(rrs.data[:], dtype=np.float32)
        if plane_data.shape[1] != roi_indices.shape[0]:
            raise ValueError(
                f"ROI response series {plane_name} has {plane_data.shape[1]} columns "
                f"but {roi_indices.shape[0]} ROI indices"
            )
        parts.append(plane_data[:, keep_cells[roi_indices]])

    if not parts:
        raise ValueError("No deconvolved ROI response series found")

    return np.concatenate(parts, axis=1), keep_cells


def discretize_distance(distance_cm):
    out = np.full(distance_cm.shape, 6, dtype=np.int16)
    out[distance_cm < -50.0] = 0
    out[(distance_cm >= -50.0) & (distance_cm < -10.0)] = 1
    out[(distance_cm >= -10.0) & (distance_cm < 0.0)] = 2
    out[distance_cm == 0.0] = 3
    out[(distance_cm > 0.0) & (distance_cm <= 10.0)] = 4
    out[(distance_cm > 10.0) & (distance_cm <= 50.0)] = 5
    out[distance_cm > 50.0] = 6
    return out


def discretize_position(position_cm):
    out = np.zeros(position_cm.shape, dtype=np.int16)
    out[(position_cm >= 90.0) & (position_cm < 180.0)] = 1
    out[(position_cm >= 180.0) & (position_cm < 270.0)] = 2
    out[(position_cm >= 270.0) & (position_cm < 360.0)] = 3
    out[position_cm >= 360.0] = 4
    return out


def discretize_speed(speed_cm_s):
    out = np.zeros(speed_cm_s.shape, dtype=np.int16)
    out[(speed_cm_s >= 2.0) & (speed_cm_s < 10.0)] = 1
    out[(speed_cm_s >= 10.0) & (speed_cm_s < 20.0)] = 2
    out[(speed_cm_s >= 20.0) & (speed_cm_s <= 40.0)] = 3
    out[speed_cm_s > 40.0] = 4
    return out


def reward_zone_distance(position_cm, zone_label):
    start_cm, end_cm = RZ_BOUNDS[zone_label]
    dist = np.zeros(position_cm.shape, dtype=np.float32)
    before = position_cm < start_cm
    after = position_cm > end_cm
    dist[before] = position_cm[before] - start_cm
    dist[after] = position_cm[after] - end_cm
    return dist


def bin_trial(neural_trial, time_trial_s, pos_trial_cm, speed_trial_cm_s, lick_trial, zone_label):
    nframes = neural_trial.shape[1]
    nbins = math.ceil(nframes / BIN_FRAMES)

    neural_binned = np.zeros((neural_trial.shape[0], nbins), dtype=np.float32)
    time_binned = np.zeros(nbins, dtype=np.float32)
    pos_binned = np.zeros(nbins, dtype=np.float32)
    speed_binned = np.zeros(nbins, dtype=np.float32)
    lick_binned = np.zeros(nbins, dtype=np.int16)

    for bin_idx in range(nbins):
        start = bin_idx * BIN_FRAMES
        stop = min((bin_idx + 1) * BIN_FRAMES, nframes)
        sl = slice(start, stop)

        neural_binned[:, bin_idx] = neural_trial[:, sl].sum(axis=1, dtype=np.float32)
        time_binned[bin_idx] = time_trial_s[start]
        pos_binned[bin_idx] = np.mean(pos_trial_cm[sl], dtype=np.float64)
        speed_binned[bin_idx] = np.mean(speed_trial_cm_s[sl], dtype=np.float64)
        lick_binned[bin_idx] = int(np.any(lick_trial[sl] > 0))

    distance_cm = reward_zone_distance(pos_binned, zone_label)

    return {
        "neural": neural_binned,
        "time": time_binned,
        "distance_bin": discretize_distance(distance_cm),
        "position_bin": discretize_position(pos_binned),
        "speed_bin": discretize_speed(speed_binned),
        "lick_bin": lick_binned,
    }


def convert_session(path, subject_to_idx):
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()

        subject = nwb.subject.subject_id
        scene = nwb.identifier.split("/")[-1]
        behavior = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series
        ophys = nwb.processing["ophys"].data_interfaces

        deconv, keep_cells = load_curated_deconvolved_activity(ophys)

        timestamps = np.asarray(behavior["position"].timestamps[:], dtype=np.float64)
        position = np.asarray(behavior["position"].data[:], dtype=np.float32)
        speed = np.asarray(behavior["speed"].data[:], dtype=np.float32)
        lick = np.asarray(behavior["lick"].data[:], dtype=np.float32)
        env = np.asarray(behavior["environment"].data[:], dtype=np.float32)
        reward_zone = np.asarray(behavior["reward_zone"].data[:], dtype=np.float32)
        trial_number = np.asarray(behavior["trial number"].data[:], dtype=np.float32)
        trial_start = np.flatnonzero(np.asarray(behavior["trial_start"].data[:]) > 0)
        teleport = np.flatnonzero(np.asarray(behavior["teleport"].data[:]) > 0)
        reward_timestamps = np.asarray(behavior["Reward"].timestamps[:], dtype=np.float64)

        if len(trial_start) != len(teleport):
            raise ValueError(f"{path.name}: trial starts and teleports do not match")

        ntrials = len(trial_start)
        _, expected_zone_labels = expected_trial_labels(scene, ntrials)

        reward_outcomes = np.zeros(ntrials, dtype=np.int16)
        zone_labels = []
        drop_trial = np.zeros(ntrials, dtype=bool)

        for trial_idx, (start, stop) in enumerate(zip(trial_start, teleport)):
            trial_reward_ts = reward_timestamps[
                (reward_timestamps >= timestamps[start]) & (reward_timestamps < timestamps[stop])
            ]
            reward_outcomes[trial_idx] = int(len(trial_reward_ts) > 0)

            trial_lick = lick[start:stop]
            if np.mean(trial_lick > 2) > LICK_ERROR_THRESHOLD:
                drop_trial[trial_idx] = True

            rz_mask = reward_zone[start:stop] > 0
            if np.any(rz_mask):
                zone_labels.append(zone_from_position(float(np.nanmedian(position[start:stop][rz_mask]))))
            else:
                zone_labels.append(expected_zone_labels[trial_idx])

        prev_reward_outcomes = np.zeros(ntrials, dtype=np.int16)
        prev_reward_outcomes[1:] = reward_outcomes[:-1]

        neural_trials = []
        input_trials = []
        output_trials = []

        for trial_idx, (start, stop) in enumerate(zip(trial_start, teleport)):
            if drop_trial[trial_idx]:
                continue

            neural_trial = deconv[start:stop].T
            time_trial_s = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
            pos_trial_cm = position[start:stop]
            speed_trial_cm_s = speed[start:stop]
            lick_trial = lick[start:stop]

            binned = bin_trial(
                neural_trial=neural_trial,
                time_trial_s=time_trial_s,
                pos_trial_cm=pos_trial_cm,
                speed_trial_cm_s=speed_trial_cm_s,
                lick_trial=lick_trial,
                zone_label=zone_labels[trial_idx],
            )

            ntime = binned["neural"].shape[1]
            env_trial = env[start:stop]
            env_trial = env_trial[env_trial >= 0]
            env_value = float(np.round(np.nanmedian(env_trial))) if len(env_trial) else float(parse_scene(scene)[0][-1] == "2")

            trialnum_trial = trial_number[start:stop]
            trialnum_trial = trialnum_trial[trialnum_trial >= 0]
            trialnum_value = float(np.round(np.nanmedian(trialnum_trial))) if len(trialnum_trial) else float(trial_idx)

            reward_zone_value = {"A": 0, "B": 1, "C": 2}[zone_labels[trial_idx]]
            reward_outcome_value = int(reward_outcomes[trial_idx])

            input_trial = np.vstack(
                [
                    binned["time"],
                    np.full(ntime, env_value, dtype=np.float32),
                    np.full(ntime, trialnum_value, dtype=np.float32),
                    np.full(ntime, float(prev_reward_outcomes[trial_idx]), dtype=np.float32),
                ]
            ).astype(np.float32)

            output_trial = np.vstack(
                [
                    binned["distance_bin"],
                    binned["position_bin"],
                    binned["speed_bin"],
                    binned["lick_bin"],
                    np.full(ntime, reward_zone_value, dtype=np.int16),
                    np.full(ntime, reward_outcome_value, dtype=np.int16),
                ]
            ).astype(np.int16)

            neural_trials.append(binned["neural"].astype(np.float32))
            input_trials.append(input_trial)
            output_trials.append(output_trial)

        return {
            "subject": subject,
            "subject_idx": subject_to_idx[subject],
            "scene": scene,
            "session_id": nwb.session_id,
            "source_file": str(path),
            "raw_roi_count": int(keep_cells.shape[0]),
            "kept_roi_count": int(keep_cells.sum()),
            "brain_region_idx": np.zeros(int(keep_cells.sum()), dtype=np.int16),
            "neural": neural_trials,
            "input": input_trials,
            "output": output_trials,
            "n_trials_raw": int(ntrials),
            "n_trials_kept": int(len(neural_trials)),
            "n_trials_dropped_lick_sensor": int(drop_trial.sum()),
        }


def main():
    session_paths = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
    if not session_paths:
        raise FileNotFoundError(f"No NWB files found under {DATA_ROOT}")

    subjects = []
    for path in session_paths:
        with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
            subject = io.read().subject.subject_id
        if subject not in subjects:
            subjects.append(subject)
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}

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
            ["< -50 cm", "-50 to -10 cm", "-10 to <0 cm", "0 cm", ">0 to 10 cm", ">10 to 50 cm", ">50 cm"],
            ["<90 cm", "90 to 180 cm", "180 to 270 cm", "270 to 360 cm", ">360 cm"],
            ["<2 cm/s", "2-10 cm/s", "10-20 cm/s", "20-40 cm/s", ">40 cm/s"],
            ["no", "yes"],
            ["A", "B", "C"],
            ["no", "yes"],
        ],
        "metadata": {
            "task_description": (
                "Head-fixed virtual-navigation task with hidden reward zones in two environments; "
                "decode trial-relative time/context and behavior from CA1 calcium activity."
            ),
            "time_bin_size": float(BIN_FRAMES * 1000.0 * 0.06448362720402656),
            "time_bin_size_frames": BIN_FRAMES,
            "temporal_alignment_event": "trial start",
            "off_start": 0.0,
            "off_end": None,
            "source_data": "Sosa et al. NWB sessions",
            "neural_signal": "Suite2p deconvolved events",
            "roi_filter": "iscell[:,0] == 1",
            "trial_window": "trial_start inclusive to teleport exclusive",
            "trial_exclusion": {
                "lick_sensor_error_threshold": LICK_ERROR_THRESHOLD,
                "rule": "drop trials where more than 35% of frames have lick values greater than 2",
            },
            "reward_zone_bounds_cm": {k: list(v) for k, v in RZ_BOUNDS.items()},
            "session_info": [],
        },
    }

    for path in session_paths:
        session = convert_session(path, subject_to_idx)
        if session["n_trials_kept"] < 2:
            continue

        data["neural"].append(session["neural"])
        data["input"].append(session["input"])
        data["output"].append(session["output"])
        data["subject_idx"].append(session["subject_idx"])
        data["brain_region_idx"].append(session["brain_region_idx"])
        data["metadata"]["session_info"].append(
            {
                "subject": session["subject"],
                "session_id": session["session_id"],
                "scene": session["scene"],
                "source_file": session["source_file"],
                "n_trials_raw": session["n_trials_raw"],
                "n_trials_kept": session["n_trials_kept"],
                "n_trials_dropped_lick_sensor": session["n_trials_dropped_lick_sensor"],
                "raw_roi_count": session["raw_roi_count"],
                "kept_roi_count": session["kept_roi_count"],
            }
        )

    data["subject_idx"] = np.asarray(data["subject_idx"], dtype=np.int16)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Sessions: {len(data['neural'])}")
    print(f"Subjects: {len(data['subjects'])}")
    print(f"Total trials: {sum(len(session) for session in data['neural'])}")


if __name__ == "__main__":
    main()
