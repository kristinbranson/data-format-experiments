import argparse
import json
import pickle
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from iblatlas.regions import BrainRegions
from scipy import signal


BWM_RELEASE_CSV = Path("/app/code/code_zhang2025/data/bwm_release.csv")
ONE_CACHE_DIR = Path("/app/data/one_cache")

BIN_SIZE_S = 0.02
OFF_START_S = -0.5
OFF_END_S = 1.5
N_BINS = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))

MIN_RT_S = 0.08
MAX_RT_S = 2.0
MAX_TRIAL_LEN_S = 10.0

WHEEL_FS = 1000
WHEEL_FILTER_CORNER_HZ = 20
WHEEL_FILTER_ORDER = 8


def latest_revision_file(base: Path, patterns):
    if isinstance(patterns, str):
        patterns = [patterns]
    matches = []
    for pattern in patterns:
        matches.extend(base.glob(pattern))
    if not matches:
        return None

    def sort_key(path: Path):
        revisions = [part for part in path.parts if part.startswith("#") and part.endswith("#")]
        revision = revisions[-1] if revisions else ""
        return revision, str(path)

    matches.sort(key=sort_key)
    return matches[-1]


def ordered_unique(values):
    seen = OrderedDict()
    for value in values:
        seen.setdefault(value, None)
    return list(seen.keys())


def interpolate_position(re_ts, re_pos, freq=WHEEL_FS):
    t = np.arange(re_ts[0], re_ts[-1], 1.0 / freq, dtype=np.float64)
    if t.size and t[-1] > re_ts[-1]:
        t = t[:-1]
    yinterp = np.interp(t, re_ts, re_pos)
    return yinterp, t


def velocity_filtered(pos, fs=WHEEL_FS, corner_frequency=WHEEL_FILTER_CORNER_HZ, order=WHEEL_FILTER_ORDER):
    sos = signal.butter(
        N=order,
        Wn=corner_frequency / fs * 2.0,
        btype="lowpass",
        output="sos",
    )
    vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos)), 0, 0.0) * fs
    acc = np.insert(np.diff(vel), 0, 0.0) * fs
    return vel, acc


def load_trials_table(alf_path: Path):
    trials_path = latest_revision_file(alf_path, "#*/_ibl_trials.table.pqt")
    if trials_path is None:
        raise FileNotFoundError(f"Missing trials table in {alf_path}")
    return pd.read_parquet(trials_path)


def compute_trial_mask(trials_df: pd.DataFrame):
    required = [
        "stimOn_times",
        "choice",
        "feedback_times",
        "probabilityLeft",
        "firstMovement_times",
        "feedbackType",
        "goCue_times",
    ]
    mask = np.ones(len(trials_df), dtype=bool)
    for column in required:
        mask &= np.isfinite(trials_df[column].to_numpy())

    rt = trials_df["firstMovement_times"].to_numpy() - trials_df["stimOn_times"].to_numpy()
    trial_len = trials_df["feedback_times"].to_numpy() - trials_df["goCue_times"].to_numpy()
    mask &= rt >= MIN_RT_S
    mask &= rt <= MAX_RT_S
    mask &= trial_len <= MAX_TRIAL_LEN_S
    mask &= trials_df["choice"].to_numpy() != 0
    return mask


def compute_trial_number_in_block(probability_left):
    values = np.asarray(probability_left, dtype=np.float64)
    numbers = np.zeros(values.shape[0], dtype=np.float32)
    count = 0
    previous = np.nan
    for idx, value in enumerate(values):
        same_block = np.isfinite(value) and np.isfinite(previous) and np.isclose(value, previous)
        count = count + 1 if same_block else 1
        numbers[idx] = float(count)
        previous = value
    return numbers


def load_wheel_speed(alf_path: Path):
    timestamps_path = latest_revision_file(alf_path, ["_ibl_wheel.timestamps.npy", "#*/_ibl_wheel.timestamps.npy"])
    position_path = latest_revision_file(alf_path, ["_ibl_wheel.position.npy", "#*/_ibl_wheel.position.npy"])
    if timestamps_path is None or position_path is None:
        raise FileNotFoundError("Missing wheel timestamps or position")

    timestamps = np.load(timestamps_path).astype(np.float64)
    position = np.load(position_path).astype(np.float64)
    if timestamps.shape[0] != position.shape[0]:
        raise ValueError("Wheel timestamps/position length mismatch")

    position_interp, time_interp = interpolate_position(timestamps, position, freq=WHEEL_FS)
    velocity, _ = velocity_filtered(position_interp, fs=WHEEL_FS)
    speed = np.abs(velocity).astype(np.float32)
    return time_interp.astype(np.float64), speed


def load_motion_energy(alf_path: Path):
    for view in ("left", "right"):
        me_path = latest_revision_file(alf_path, f"#*/{view}Camera.ROIMotionEnergy.npy")
        if me_path is None:
            continue
        times_path = latest_revision_file(
            alf_path,
            [f"_ibl_{view}Camera.times.npy", f"#*/_ibl_{view}Camera.times.npy"],
        )
        if times_path is None:
            continue

        motion_energy = np.load(me_path).astype(np.float32)
        timestamps = np.load(times_path).astype(np.float64)

        if timestamps.shape[0] < motion_energy.shape[0]:
            raise ValueError(f"{view} camera timestamps shorter than motion energy")
        if timestamps.shape[0] > motion_energy.shape[0]:
            timestamps = timestamps[-motion_energy.shape[0]:]

        return timestamps, motion_energy, view

    raise FileNotFoundError("Missing left/right whisker motion energy")


def region_ids_to_acronyms(region_ids, brain_regions: BrainRegions):
    region_ids = np.asarray(region_ids, dtype=np.int64)
    acronyms = np.asarray(brain_regions.id2acronym(region_ids), dtype=object)
    for idx, acronym in enumerate(acronyms):
        if acronym is None or acronym == "":
            acronyms[idx] = f"id_{int(region_ids[idx])}"
    return acronyms.astype(str)


def load_probe_good_units(probe_alf_path: Path, brain_regions: BrainRegions):
    sorter_base = probe_alf_path / "pykilosort"
    metrics_path = latest_revision_file(sorter_base, "#*/clusters.metrics.pqt")
    spike_times_path = latest_revision_file(sorter_base, "#*/spikes.times.npy")
    spike_clusters_path = latest_revision_file(sorter_base, "#*/spikes.clusters.npy")
    cluster_channels_path = latest_revision_file(sorter_base, "#*/clusters.channels.npy")
    channel_regions_path = latest_revision_file(
        sorter_base,
        "#*/channels.brainLocationIds_ccf_2017.npy",
    )

    if any(path is None for path in [
        metrics_path,
        spike_times_path,
        spike_clusters_path,
        cluster_channels_path,
        channel_regions_path,
    ]):
        raise FileNotFoundError(f"Missing spike sorting files in {probe_alf_path}")

    metrics = pd.read_parquet(metrics_path, columns=["label"])
    spike_times = np.load(spike_times_path).astype(np.float64)
    spike_clusters = np.load(spike_clusters_path).astype(np.int64)
    cluster_channels = np.load(cluster_channels_path).astype(np.int64)
    channel_region_ids = np.load(channel_regions_path).astype(np.int64)

    n_clusters = cluster_channels.shape[0]
    keep_mask = metrics["label"].to_numpy() >= 1.0
    if keep_mask.shape[0] != n_clusters:
        raise ValueError(f"Cluster metrics/channel count mismatch in {probe_alf_path}")

    cluster_region_ids = np.zeros(n_clusters, dtype=np.int64)
    valid_channel = (cluster_channels >= 0) & (cluster_channels < channel_region_ids.shape[0])
    cluster_region_ids[valid_channel] = channel_region_ids[cluster_channels[valid_channel]]
    cluster_acronyms = region_ids_to_acronyms(cluster_region_ids, brain_regions)

    keep_mask &= ~np.isin(cluster_acronyms, ["void", "root"])
    kept_clusters = np.flatnonzero(keep_mask)
    if kept_clusters.size == 0:
        return None

    remap = np.full(n_clusters, -1, dtype=np.int64)
    remap[kept_clusters] = np.arange(kept_clusters.size, dtype=np.int64)
    spike_keep = remap[spike_clusters] >= 0

    return {
        "spike_times": spike_times[spike_keep],
        "spike_clusters": remap[spike_clusters[spike_keep]],
        "cluster_acronyms": cluster_acronyms[kept_clusters],
    }


def merge_session_probes(probe_infos):
    merged_times = []
    merged_clusters = []
    merged_regions = []
    cluster_offset = 0

    for info in probe_infos:
        if info is None:
            continue
        merged_times.append(info["spike_times"])
        merged_clusters.append(info["spike_clusters"] + cluster_offset)
        merged_regions.extend(info["cluster_acronyms"].tolist())
        cluster_offset += info["cluster_acronyms"].shape[0]

    if not merged_times:
        return None

    spike_times = np.concatenate(merged_times)
    spike_clusters = np.concatenate(merged_clusters)
    order = np.argsort(spike_times, kind="stable")
    return {
        "spike_times": spike_times[order],
        "spike_clusters": spike_clusters[order],
        "cluster_acronyms": np.asarray(merged_regions, dtype=object),
    }


def bin_spikes_for_trial(spike_times, spike_clusters, n_clusters, trial_start, binsize=BIN_SIZE_S, n_bins=N_BINS):
    trial_end = trial_start + binsize * n_bins
    start_idx = np.searchsorted(spike_times, trial_start, side="left")
    end_idx = np.searchsorted(spike_times, trial_end, side="left")
    if end_idx <= start_idx:
        return np.zeros((n_clusters, n_bins), dtype=np.float16)

    times = spike_times[start_idx:end_idx]
    clusters = spike_clusters[start_idx:end_idx]
    bin_idx = np.floor((times - trial_start) / binsize).astype(np.int64)
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    if not np.any(valid):
        return np.zeros((n_clusters, n_bins), dtype=np.float16)

    flat_idx = clusters[valid] * n_bins + bin_idx[valid]
    counts = np.bincount(flat_idx, minlength=n_clusters * n_bins).reshape(n_clusters, n_bins)
    return counts.astype(np.float16)


def interpolate_behavior_trial(sample_times, sample_values, interval_start, interval_end, binsize=BIN_SIZE_S, n_bins=N_BINS):
    if not (np.isfinite(interval_start) and np.isfinite(interval_end)):
        return None

    start_idx = np.searchsorted(sample_times, interval_start, side="right")
    end_idx = np.searchsorted(sample_times, interval_end, side="left")
    trial_times = sample_times[start_idx:end_idx]
    trial_values = sample_values[start_idx:end_idx]

    if trial_values.size == 0:
        return None
    if abs(interval_start - trial_times[0]) > binsize:
        return None
    if abs(interval_end - trial_times[-1]) > binsize:
        return None

    x_interp = np.linspace(interval_start + binsize, interval_end, n_bins, dtype=np.float64)
    y_interp = np.interp(x_interp, trial_times, trial_values).astype(np.float32)
    if not np.all(np.isfinite(y_interp)):
        return None
    return y_interp


def probability_left_to_class(value):
    mapping = {0.2: 0, 0.5: 1, 0.8: 2}
    for key, encoded in mapping.items():
        if np.isclose(value, key):
            return encoded
    raise ValueError(f"Unexpected probabilityLeft value: {value}")


def choice_to_class(value):
    if np.isclose(value, 1.0):
        return 0
    if np.isclose(value, -1.0):
        return 1
    raise ValueError(f"Unexpected choice value: {value}")


def discretize_three_bins(values):
    values = np.asarray(values, dtype=np.float64)
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    if q2 <= q1:
        eps = np.finfo(np.float32).eps
        q2 = q1 + eps
    bins = np.digitize(values, [q1, q2], right=False).astype(np.int8)
    return bins, (float(q1), float(q2))


def summarize_counts(label, counts):
    counts = Counter(counts)
    ordered = ", ".join(f"{k}: {counts[k]}" for k in sorted(counts))
    return f"{label}: {ordered}"


def convert_dataset(session_limit=None):
    release_df = pd.read_csv(BWM_RELEASE_CSV)
    session_rows = []
    for eid, group in release_df.groupby("eid", sort=False):
        first = group.iloc[0]
        session_rows.append({
            "eid": eid,
            "subject": first["subject"],
            "lab": first["lab"],
            "date": first["date"],
            "session_number": int(first["session_number"]),
            "probe_names": list(group["probe_name"]),
        })

    if session_limit is not None:
        session_rows = session_rows[:session_limit]

    brain_regions = BrainRegions()

    kept_sessions = []
    dropped_sessions = []
    wheel_continuous_all = []
    whisker_continuous_all = []
    trials_per_session = []
    neurons_per_session = []
    motion_views = []

    for session_idx, session in enumerate(session_rows, start=1):
        session_path = (
            ONE_CACHE_DIR
            / session["lab"]
            / "Subjects"
            / session["subject"]
            / session["date"]
            / f"{session['session_number']:03d}"
        )
        alf_path = session_path / "alf"

        try:
            trials_df = load_trials_table(alf_path)
            trial_mask = compute_trial_mask(trials_df)
            trial_number_in_block = compute_trial_number_in_block(trials_df["probabilityLeft"].to_numpy())
            wheel_times, wheel_speed = load_wheel_speed(alf_path)
            whisker_times, whisker_me, motion_view = load_motion_energy(alf_path)

            probe_infos = []
            for probe_name in session["probe_names"]:
                probe_infos.append(load_probe_good_units(alf_path / probe_name, brain_regions))
            merged = merge_session_probes(probe_infos)
            if merged is None:
                dropped_sessions.append((session["eid"], "no_good_units"))
                continue

            n_clusters = merged["cluster_acronyms"].shape[0]
            session_neural = []
            session_input = []
            session_static = []
            session_wheel = []
            session_whisker = []

            for trial_idx, trial_row in trials_df.iterrows():
                if not trial_mask[trial_idx]:
                    continue
                trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
                trial_end = float(trial_row["stimOn_times"] + OFF_END_S)

                wheel_trial = interpolate_behavior_trial(wheel_times, wheel_speed, trial_start, trial_end)
                whisker_trial = interpolate_behavior_trial(whisker_times, whisker_me, trial_start, trial_end)
                if wheel_trial is None or whisker_trial is None:
                    continue

                neural_trial = bin_spikes_for_trial(
                    merged["spike_times"],
                    merged["spike_clusters"],
                    n_clusters,
                    trial_start,
                )
                if neural_trial.shape != (n_clusters, N_BINS):
                    raise ValueError("Unexpected neural trial shape")
                if not np.any(neural_trial):
                    continue

                time_since_stim = np.linspace(
                    OFF_START_S + BIN_SIZE_S,
                    OFF_END_S,
                    N_BINS,
                    dtype=np.float32,
                )
                input_trial = np.vstack([
                    time_since_stim,
                    np.full(N_BINS, trial_number_in_block[trial_idx], dtype=np.float32),
                ]).astype(np.float32)

                choice_class = choice_to_class(float(trial_row["choice"]))
                prior_class = probability_left_to_class(float(trial_row["probabilityLeft"]))

                session_neural.append(neural_trial)
                session_input.append(input_trial)
                session_static.append((choice_class, prior_class))
                session_wheel.append(wheel_trial)
                session_whisker.append(whisker_trial)
                wheel_continuous_all.append(wheel_trial)
                whisker_continuous_all.append(whisker_trial)

            if len(session_neural) < 2:
                dropped_sessions.append((session["eid"], "fewer_than_two_trials"))
                continue

            kept_sessions.append({
                "eid": session["eid"],
                "subject": session["subject"],
                "cluster_acronyms": merged["cluster_acronyms"].copy(),
                "neural": session_neural,
                "input": session_input,
                "static_outputs": session_static,
                "wheel_continuous": session_wheel,
                "whisker_continuous": session_whisker,
            })
            trials_per_session.append(len(session_neural))
            neurons_per_session.append(n_clusters)
            motion_views.append(motion_view)

            if session_idx % 25 == 0 or session_idx == len(session_rows):
                print(
                    f"[{session_idx}/{len(session_rows)}] kept={len(kept_sessions)} "
                    f"dropped={len(dropped_sessions)}"
                )

        except Exception as exc:
            dropped_sessions.append((session["eid"], f"error:{type(exc).__name__}:{exc}"))

    if not kept_sessions:
        raise RuntimeError("No sessions were converted successfully")

    wheel_flat = np.concatenate(wheel_continuous_all)
    whisker_flat = np.concatenate(whisker_continuous_all)
    _, wheel_edges = discretize_three_bins(wheel_flat)
    _, whisker_edges = discretize_three_bins(whisker_flat)

    subjects = ordered_unique(session["subject"] for session in kept_sessions)
    subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
    all_region_names = ordered_unique(
        region
        for session in kept_sessions
        for region in session["cluster_acronyms"]
    )
    region_lookup = {region: idx for idx, region in enumerate(all_region_names)}

    data = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": subjects,
        "subject_idx": np.asarray(
            [subject_lookup[session["subject"]] for session in kept_sessions],
            dtype=np.int64,
        ),
        "brain_regions": all_region_names,
        "brain_region_idx": [],
        "input_names": ["time_since_stimulus_onset_s", "trial_number_in_block"],
        "output_names": [
            "choice",
            "prior_probability_left",
            "wheel_speed_bin",
            "whisker_motion_energy_bin",
        ],
        "output_values": [
            ["left", "right"],
            ["0.2", "0.5", "0.8"],
            ["low", "medium", "high"],
            ["low", "medium", "high"],
        ],
        "metadata": {
            "task_description": (
                "IBL decision-making task; decode choice, prior probability of left, "
                "wheel speed, and whisker motion energy from stimulus-onset-aligned neural activity."
            ),
            "time_bin_size": BIN_SIZE_S * 1000.0,
            "temporal_alignment_event": "stimulus onset (stimOn_times)",
            "off_start": OFF_START_S,
            "off_end": OFF_END_S,
            "n_time_bins": N_BINS,
            "trial_mask_rules": {
                "required_events": [
                    "stimOn_times",
                    "choice",
                    "feedback_times",
                    "probabilityLeft",
                    "firstMovement_times",
                    "feedbackType",
                    "goCue_times",
                ],
                "exclude_no_choice": True,
                "reaction_time_range_s": [MIN_RT_S, MAX_RT_S],
                "max_trial_length_s": MAX_TRIAL_LEN_S,
            },
            "neuron_qc": "clusters.metrics.label >= 1, excluding root/void regions",
            "wheel_processing": {
                "interpolation_frequency_hz": WHEEL_FS,
                "velocity_lowpass_corner_hz": WHEEL_FILTER_CORNER_HZ,
                "velocity_lowpass_order": WHEEL_FILTER_ORDER,
                "decoded_quantity": "absolute wheel velocity",
            },
            "whisker_motion_energy_camera_rule": "left camera if available, otherwise right camera",
            "dynamic_output_binning": {
                "wheel_speed_edges": list(wheel_edges),
                "whisker_motion_energy_edges": list(whisker_edges),
                "binning_rule": "global tertiles over all retained session/trial/time bins",
            },
            "session_ids": [session["eid"] for session in kept_sessions],
        },
    }

    for session in kept_sessions:
        data["neural"].append(session["neural"])
        data["input"].append(session["input"])
        data["brain_region_idx"].append(
            np.asarray([region_lookup[region] for region in session["cluster_acronyms"]], dtype=np.int64)
        )

        session_output = []
        for static_values, wheel_trial, whisker_trial in zip(
            session["static_outputs"],
            session["wheel_continuous"],
            session["whisker_continuous"],
        ):
            choice_class, prior_class = static_values
            wheel_bins = np.digitize(wheel_trial, [wheel_edges[0], wheel_edges[1]], right=False).astype(np.int8)
            whisker_bins = np.digitize(
                whisker_trial,
                [whisker_edges[0], whisker_edges[1]],
                right=False,
            ).astype(np.int8)
            output_trial = np.vstack([
                np.full(N_BINS, choice_class, dtype=np.int8),
                np.full(N_BINS, prior_class, dtype=np.int8),
                wheel_bins,
                whisker_bins,
            ])
            session_output.append(output_trial)
        data["output"].append(session_output)

    stats = {
        "release_sessions": len(session_rows),
        "kept_sessions": len(kept_sessions),
        "dropped_sessions": len(dropped_sessions),
        "kept_subjects": len(subjects),
        "kept_brain_regions": len(all_region_names),
        "total_trials": int(sum(trials_per_session)),
        "trials_per_session_mean": float(np.mean(trials_per_session)),
        "trials_per_session_median": float(np.median(trials_per_session)),
        "neurons_per_session_mean": float(np.mean(neurons_per_session)),
        "neurons_per_session_median": float(np.median(neurons_per_session)),
        "motion_views": dict(Counter(motion_views)),
        "drop_reasons_top20": Counter(reason for _, reason in dropped_sessions).most_common(20),
    }

    print(f"Converted sessions: {stats['kept_sessions']} / {stats['release_sessions']}")
    print(f"Unique subjects kept: {stats['kept_subjects']}")
    print(f"Unique brain regions kept: {stats['kept_brain_regions']}")
    print(f"Total retained trials: {stats['total_trials']}")
    print(
        "Trials/session mean median: "
        f"{stats['trials_per_session_mean']:.1f} / {stats['trials_per_session_median']:.1f}"
    )
    print(
        "Neurons/session mean median: "
        f"{stats['neurons_per_session_mean']:.1f} / {stats['neurons_per_session_median']:.1f}"
    )
    print(summarize_counts("Motion camera usage", motion_views))
    print(f"Top drop reasons: {stats['drop_reasons_top20']}")

    return data, stats


def build_sample_subset(data, n_sessions):
    n_sessions = min(n_sessions, len(data["neural"]))
    sample_region_idx = data["brain_region_idx"][:n_sessions]
    used_region_ids = ordered_unique(
        int(region_id)
        for session_regions in sample_region_idx
        for region_id in session_regions.tolist()
    )
    region_lookup = {old_idx: new_idx for new_idx, old_idx in enumerate(used_region_ids)}

    sample = {
        "neural": data["neural"][:n_sessions],
        "input": data["input"][:n_sessions],
        "output": data["output"][:n_sessions],
        "subjects": [],
        "subject_idx": data["subject_idx"][:n_sessions].copy(),
        "brain_regions": [data["brain_regions"][idx] for idx in used_region_ids],
        "brain_region_idx": [
            np.asarray([region_lookup[int(region_id)] for region_id in session_regions], dtype=np.int64)
            for session_regions in sample_region_idx
        ],
        "input_names": data["input_names"],
        "output_names": data["output_names"],
        "output_values": data["output_values"],
        "metadata": dict(data["metadata"]),
    }

    used_subjects = ordered_unique(data["subjects"][idx] for idx in sample["subject_idx"])
    subject_lookup = {subject: idx for idx, subject in enumerate(used_subjects)}
    sample["subjects"] = used_subjects
    sample["subject_idx"] = np.asarray(
        [subject_lookup[data["subjects"][idx]] for idx in sample["subject_idx"]],
        dtype=np.int64,
    )
    sample["metadata"]["sample_session_count"] = n_sessions
    sample["metadata"]["session_ids"] = data["metadata"]["session_ids"][:n_sessions]
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/app/converted_data.pkl"))
    parser.add_argument("--sample-output", type=Path, default=None)
    parser.add_argument("--sample-sessions", type=int, default=8)
    parser.add_argument("--session-limit", type=int, default=None)
    parser.add_argument("--stats-json", type=Path, default=None)
    args = parser.parse_args()

    data, stats = convert_dataset(session_limit=args.session_limit)
    with args.output.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved converted dataset to {args.output}")

    if args.sample_output is not None:
        sample = build_sample_subset(data, args.sample_sessions)
        with args.sample_output.open("wb") as stream:
            pickle.dump(sample, stream, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved sample dataset to {args.sample_output}")

    if args.stats_json is not None:
        args.stats_json.write_text(json.dumps(stats, indent=2))
        print(f"Saved conversion stats to {args.stats_json}")


if __name__ == "__main__":
    main()
