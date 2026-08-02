import argparse
import json
import math
import os
import pickle
from collections import Counter
from typing import Any

import h5py
import numpy as np


WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_SIZE_S = 0.05
BIN_EDGES_S = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_SIZE_S, BIN_SIZE_S, dtype=np.float64)
BIN_CENTERS_S = BIN_EDGES_S[:-1] + (BIN_SIZE_S / 2.0)

INPUT_NAMES = ["time_from_tone_onset_s", "photostim_on"]
OUTPUT_NAMES = ["choice", "outcome", "early_lick", "tongue_y_position"]
OUTPUT_VALUES = [
    ["left", "right"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["lt_40pct", "p40_to_p60", "gt_60pct"],
]


def decode_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return value


def decode_array(values: np.ndarray) -> np.ndarray:
    return np.array([decode_scalar(v) for v in values], dtype=object)


def as_float_or_none(value: Any) -> float | None:
    value = decode_scalar(value)
    if value in ("N/A", "", None):
        return None
    return float(value)


def get_subject_from_path(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


def get_session_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def load_sorted_nwb_paths(data_root: str) -> list[str]:
    paths = []
    for subject in sorted(os.listdir(data_root)):
        subject_path = os.path.join(data_root, subject)
        if not os.path.isdir(subject_path):
            continue
        for filename in sorted(os.listdir(subject_path)):
            if filename.endswith(".nwb"):
                paths.append(os.path.join(subject_path, filename))
    return paths


def assert_one_event_per_trial(event_times: np.ndarray, trial_start: np.ndarray, trial_stop: np.ndarray, name: str) -> np.ndarray:
    left = np.searchsorted(event_times, trial_start, side="left")
    right = np.searchsorted(event_times, trial_stop, side="right")
    counts = right - left
    if not np.all(counts == 1):
        bad = np.where(counts != 1)[0][:10]
        raise ValueError(f"{name}: expected exactly one event per trial, bad trial indices={bad.tolist()}, counts={counts[bad].tolist()}")
    return event_times[right - 1]


def last_event_before_per_trial(event_times: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray, name: str) -> np.ndarray:
    idx = np.searchsorted(event_times, upper_bound, side="left") - 1
    if np.any(idx < 0):
        bad = np.where(idx < 0)[0][:10]
        raise ValueError(f"{name}: missing prior event for trial indices={bad.tolist()}")
    values = event_times[idx]
    valid = values >= lower_bound
    if not np.all(valid):
        bad = np.where(~valid)[0][:10]
        raise ValueError(f"{name}: event before lower bound for trial indices={bad.tolist()}")
    return values


def compute_choice_labels(
    trial_start: np.ndarray,
    trial_stop: np.ndarray,
    go_times: np.ndarray,
    instructions: np.ndarray,
    left_lick_times: np.ndarray,
    right_lick_times: np.ndarray,
) -> tuple[np.ndarray, Counter]:
    choice = np.zeros(len(trial_start), dtype=np.int8)
    source_counter = Counter()

    for trial_idx in range(len(trial_start)):
        start = trial_start[trial_idx]
        stop = trial_stop[trial_idx]
        go_time = go_times[trial_idx]

        left_all = left_lick_times[(left_lick_times >= start) & (left_lick_times <= stop)]
        right_all = right_lick_times[(right_lick_times >= start) & (right_lick_times <= stop)]
        left_post = left_all[left_all >= go_time]
        right_post = right_all[right_all >= go_time]

        if left_post.size or right_post.size:
            left_first = left_post[0] if left_post.size else np.inf
            right_first = right_post[0] if right_post.size else np.inf
            choice[trial_idx] = 0 if left_first < right_first else 1
            source_counter["post_go_lick"] += 1
            continue

        if left_all.size or right_all.size:
            left_first = left_all[0] if left_all.size else np.inf
            right_first = right_all[0] if right_all.size else np.inf
            choice[trial_idx] = 0 if left_first < right_first else 1
            source_counter["any_trial_lick"] += 1
            continue

        choice[trial_idx] = 0 if instructions[trial_idx] == "left" else 1
        source_counter["instruction_fallback"] += 1

    return choice, source_counter


def bin_photostim_series(
    trial_start: np.ndarray,
    go_times: np.ndarray,
    onset_values: np.ndarray,
    duration_values: np.ndarray,
) -> tuple[np.ndarray, int]:
    n_trials = len(trial_start)
    n_bins = len(BIN_CENTERS_S)
    photostim = np.zeros((n_trials, n_bins), dtype=np.float32)
    stim_trial_count = 0

    bin_left = BIN_EDGES_S[:-1][None, :]
    bin_right = BIN_EDGES_S[1:][None, :]

    for trial_idx in range(n_trials):
        onset = as_float_or_none(onset_values[trial_idx])
        duration = as_float_or_none(duration_values[trial_idx])
        if onset is None or duration is None:
            continue
        stim_trial_count += 1
        rel_on = (trial_start[trial_idx] + onset) - go_times[trial_idx]
        rel_off = rel_on + duration
        overlap = (bin_left < rel_off) & (bin_right > rel_on)
        photostim[trial_idx] = overlap.astype(np.float32)[0]

    return photostim, stim_trial_count


def bin_tongue_y(
    timestamps: np.ndarray,
    tongue_y: np.ndarray,
    go_times: np.ndarray,
    q40: float,
    q60: float,
) -> tuple[np.ndarray, int]:
    trial_edges = go_times[:, None] + BIN_EDGES_S[None, :]
    left_idx = np.searchsorted(timestamps, trial_edges[:, :-1], side="left")
    right_idx = np.searchsorted(timestamps, trial_edges[:, 1:], side="left")

    # Match the marker alignment style in the reference repo by using the last sample
    # available inside each bin. If a bin has no sample, fall back to the latest sample
    # before the bin end.
    fallback_missing = int(np.sum(right_idx <= left_idx))
    sample_idx = np.clip(right_idx - 1, 0, len(timestamps) - 1)
    y_binned = tongue_y[sample_idx]

    tongue_disc = np.ones_like(y_binned, dtype=np.int8)
    tongue_disc[y_binned < q40] = 0
    tongue_disc[y_binned > q60] = 2
    return tongue_disc, fallback_missing


def bin_spike_counts_for_good_units(
    spike_times_flat: np.ndarray,
    spike_times_index: np.ndarray,
    good_unit_mask: np.ndarray,
    trial_start: np.ndarray,
    trial_stop: np.ndarray,
    go_times: np.ndarray,
) -> np.ndarray:
    n_trials = len(trial_start)
    n_bins = len(BIN_CENTERS_S)
    n_good_units = int(np.sum(good_unit_mask))
    counts = np.zeros((n_trials, n_bins, n_good_units), dtype=np.uint8)

    unit_start = 0
    good_unit_col = 0
    for unit_idx, unit_end in enumerate(spike_times_index):
        unit_end = int(unit_end)
        spikes = spike_times_flat[unit_start:unit_end]
        unit_start = unit_end

        if not good_unit_mask[unit_idx]:
            continue

        trial_idx = np.searchsorted(trial_start, spikes, side="right") - 1
        valid = (trial_idx >= 0) & (trial_idx < n_trials)
        if np.any(valid):
            valid_trial_idx = trial_idx[valid]
            valid &= spikes <= trial_stop[valid_trial_idx]

        if np.any(valid):
            valid_trial_idx = trial_idx[valid]
            rel_spikes = spikes[valid] - go_times[valid_trial_idx]
            in_window = (rel_spikes >= WINDOW_START_S) & (rel_spikes < WINDOW_END_S)
            valid_trial_idx = valid_trial_idx[in_window]
            rel_spikes = rel_spikes[in_window]

            if rel_spikes.size:
                bin_idx = np.floor((rel_spikes - WINDOW_START_S) / BIN_SIZE_S).astype(np.int64)
                np.add.at(counts[:, :, good_unit_col], (valid_trial_idx, bin_idx), 1)

        good_unit_col += 1

    return counts


def convert_session(path: str) -> dict[str, Any] | None:
    with h5py.File(path, "r") as f:
        classifications = decode_array(f["units/classification"][:]).astype(str)
        good_unit_mask = classifications == "good"
        n_good_units = int(np.sum(good_unit_mask))
        if n_good_units == 0:
            return None

        n_ephys_trials = int(f["units/is_good_trials"].shape[1])
        trial_group = f["intervals/trials"]
        n_behavior_trials = int(len(trial_group["start_time"]))
        behavior_trial_start = np.asarray(trial_group["start_time"][:], dtype=np.float64)
        behavior_trial_stop = np.asarray(trial_group["stop_time"][:], dtype=np.float64)

        behavioral_events = f["acquisition/BehavioralEvents"]
        go_events = np.asarray(behavioral_events["go_start_times"]["timestamps"][:], dtype=np.float64)
        sample_events = np.asarray(behavioral_events["sample_start_times"]["timestamps"][:], dtype=np.float64)
        left_lick_times = np.asarray(behavioral_events["left_lick_times"]["timestamps"][:], dtype=np.float64)
        right_lick_times = np.asarray(behavioral_events["right_lick_times"]["timestamps"][:], dtype=np.float64)

        if "BehavioralTimeSeries" not in f["acquisition"]:
            raise ValueError(f"Missing BehavioralTimeSeries in {path}")
        tongue_group = f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
        tongue_data = np.asarray(tongue_group["data"][:], dtype=np.float64)
        tongue_timestamps = np.asarray(tongue_group["timestamps"][:], dtype=np.float64)
        tongue_y_raw = tongue_data[:, 1]
        q40 = float(np.percentile(tongue_y_raw, 40))
        q60 = float(np.percentile(tongue_y_raw, 60))

        good_unit_indices = np.flatnonzero(good_unit_mask)
        obs_intervals = np.asarray(f["units/obs_intervals"][:], dtype=np.float64)
        obs_index = np.asarray(f["units/obs_intervals_index"][:], dtype=np.int64)
        first_good_unit = int(good_unit_indices[0])
        obs_start = 0 if first_good_unit == 0 else int(obs_index[first_good_unit - 1])
        obs_stop = int(obs_index[first_good_unit])
        first_unit_obs = obs_intervals[obs_start:obs_stop]
        if first_unit_obs.shape[0] != n_ephys_trials:
            raise ValueError(f"Unexpected obs_intervals length in {path}")

        # Match the ephys-covered trials back to the behavioral trial table by exact interval.
        trial_lookup = {
            (round(float(start), 4), round(float(stop), 4)): idx
            for idx, (start, stop) in enumerate(zip(behavior_trial_start, behavior_trial_stop))
        }
        trial_indices = []
        for start, stop in first_unit_obs:
            key = (round(float(start), 4), round(float(stop), 4))
            if key not in trial_lookup:
                raise ValueError(f"Could not match obs_interval {key} to a behavioral trial in {path}")
            trial_indices.append(trial_lookup[key])
        trial_indices = np.asarray(trial_indices, dtype=np.int64)
        if np.unique(trial_indices).size != trial_indices.size:
            raise ValueError(f"obs_intervals map to duplicate behavioral trials in {path}")
        if np.any(np.diff(trial_indices) <= 0):
            raise ValueError(f"obs_intervals are not strictly ordered in {path}")

        trial_start = behavior_trial_start[trial_indices]
        trial_stop = behavior_trial_stop[trial_indices]
        instructions = decode_array(trial_group["trial_instruction"][:])[trial_indices].astype(str)
        outcomes_raw = decode_array(trial_group["outcome"][:])[trial_indices].astype(str)
        early_raw = decode_array(trial_group["early_lick"][:])[trial_indices].astype(str)
        photostim_onset = decode_array(trial_group["photostim_onset"][:])[trial_indices]
        photostim_duration = decode_array(trial_group["photostim_duration"][:])[trial_indices]

        go_times = assert_one_event_per_trial(go_events, trial_start, trial_stop, "go_start_times")
        sample_start_times = last_event_before_per_trial(sample_events, trial_start, go_times, "sample_start_times")
        sample_rel = sample_start_times - go_times

        choice, choice_sources = compute_choice_labels(
            trial_start=trial_start,
            trial_stop=trial_stop,
            go_times=go_times,
            instructions=instructions,
            left_lick_times=left_lick_times,
            right_lick_times=right_lick_times,
        )

        outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
        early_map = {"no early": 0, "early": 1}
        outcome = np.array([outcome_map[x] for x in outcomes_raw], dtype=np.int8)
        early = np.array([early_map[x] for x in early_raw], dtype=np.int8)

        time_from_tone = (BIN_CENTERS_S[None, :] - sample_rel[:, None]).astype(np.float32)
        photostim, stim_trial_count = bin_photostim_series(
            trial_start=trial_start,
            go_times=go_times,
            onset_values=photostim_onset,
            duration_values=photostim_duration,
        )
        tongue_disc, tongue_fallback_missing = bin_tongue_y(
            timestamps=tongue_timestamps,
            tongue_y=tongue_y_raw,
            go_times=go_times,
            q40=q40,
            q60=q60,
        )

        spike_counts = bin_spike_counts_for_good_units(
            spike_times_flat=np.asarray(f["units/spike_times"][:], dtype=np.float64),
            spike_times_index=np.asarray(f["units/spike_times_index"][:], dtype=np.int64),
            good_unit_mask=good_unit_mask,
            trial_start=trial_start,
            trial_stop=trial_stop,
            go_times=go_times,
        )

        anno_names = decode_array(f["units/anno_name"][:]).astype(str)[good_unit_mask]
        brain_region_labels = anno_names.tolist()

    n_trials = len(trial_start)
    neural_trials = []
    input_trials = []
    output_trials = []
    for trial_idx in range(n_trials):
        neural_trials.append((spike_counts[trial_idx].T.astype(np.float16) * (1.0 / BIN_SIZE_S)))

        input_trials.append(
            np.vstack(
                [
                    time_from_tone[trial_idx],
                    photostim[trial_idx],
                ]
            ).astype(np.float32)
        )

        output_trials.append(
            np.vstack(
                [
                    np.full(len(BIN_CENTERS_S), choice[trial_idx], dtype=np.int8),
                    np.full(len(BIN_CENTERS_S), outcome[trial_idx], dtype=np.int8),
                    np.full(len(BIN_CENTERS_S), early[trial_idx], dtype=np.int8),
                    tongue_disc[trial_idx],
                ]
            )
        )

    outcome_counter = Counter(int(x) for x in outcome.tolist())
    early_counter = Counter(int(x) for x in early.tolist())
    choice_counter = Counter(int(x) for x in choice.tolist())

    return {
        "session_id": get_session_id_from_path(path),
        "subject": get_subject_from_path(path),
        "neural": neural_trials,
        "input": input_trials,
        "output": output_trials,
        "brain_region_labels": brain_region_labels,
        "stats": {
            "n_trials": int(n_trials),
            "n_good_units": int(len(brain_region_labels)),
            "n_behavior_trials": int(n_behavior_trials),
            "n_ephys_trials": int(n_ephys_trials),
            "stim_trial_count": int(stim_trial_count),
            "choice_counts": dict(choice_counter),
            "choice_sources": dict(choice_sources),
            "outcome_counts": dict(outcome_counter),
            "early_counts": dict(early_counter),
            "tongue_q40": q40,
            "tongue_q60": q60,
            "tongue_bin_fallback_count": int(tongue_fallback_missing),
        },
    }


def build_dataset(session_results: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    subjects = sorted({session["subject"] for session in session_results})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}

    brain_regions = []
    brain_region_to_idx = {}

    neural = []
    inputs = []
    outputs = []
    subject_idx = []
    brain_region_idx = []

    total_choice_sources = Counter()
    total_outcomes = Counter()
    total_early = Counter()
    total_choice = Counter()
    total_stim_trials = 0
    total_tongue_fallback = 0

    session_summaries = []

    for session in session_results:
        neural.append(session["neural"])
        inputs.append(session["input"])
        outputs.append(session["output"])
        subject_idx.append(subject_to_idx[session["subject"]])

        region_indices = []
        for label in session["brain_region_labels"]:
            if label not in brain_region_to_idx:
                brain_region_to_idx[label] = len(brain_regions)
                brain_regions.append(label)
            region_indices.append(brain_region_to_idx[label])
        brain_region_idx.append(np.asarray(region_indices, dtype=np.int32))

        stats = session["stats"]
        total_choice_sources.update(stats["choice_sources"])
        total_outcomes.update(stats["outcome_counts"])
        total_early.update(stats["early_counts"])
        total_choice.update(stats["choice_counts"])
        total_stim_trials += stats["stim_trial_count"]
        total_tongue_fallback += stats["tongue_bin_fallback_count"]
        session_summaries.append(
            {
                "session_id": session["session_id"],
                "subject": session["subject"],
                "n_trials": stats["n_trials"],
                "n_good_units": stats["n_good_units"],
                "stim_trial_count": stats["stim_trial_count"],
                "tongue_q40": stats["tongue_q40"],
                "tongue_q60": stats["tongue_q60"],
            }
        )

    metadata = {
        "task_description": (
            "Auditory delayed-response task aligned to go cue; neural decoder predicts "
            "lick choice, trial outcome, early lick, and discretized tongue y-position."
        ),
        "time_bin_size": BIN_SIZE_S * 1000.0,
        "temporal_alignment_event": "Go cue onset",
        "off_start": WINDOW_START_S,
        "off_end": WINDOW_END_S,
        "bin_edges_s": BIN_EDGES_S.tolist(),
        "bin_centers_s": BIN_CENTERS_S.tolist(),
        "n_timepoints": int(len(BIN_CENTERS_S)),
        "session_ids": [session["session_id"] for session in session_results],
        "inclusion_rules": [
            "Include NWB sessions with at least one unit whose classification is 'good'.",
            "Keep all trials after session inclusion so outcome=ignore, early-lick, and photostim conditions remain available for decoding.",
            "Use go-cue alignment and 50 ms non-overlapping bins.",
        ],
        "choice_definition": (
            "First post-go lick if available; otherwise first lick anywhere in the trial; "
            "otherwise instructed side fallback for no-lick trials."
        ),
        "choice_source_counts": dict(total_choice_sources),
        "outcome_value_counts": dict(total_outcomes),
        "early_lick_value_counts": dict(total_early),
        "choice_value_counts": dict(total_choice),
        "stim_trial_count_total": int(total_stim_trials),
        "tongue_bin_fallback_count_total": int(total_tongue_fallback),
        "session_summary": session_summaries,
    }

    data = {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int32),
        "brain_regions": brain_regions,
        "brain_region_idx": brain_region_idx,
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "output_values": OUTPUT_VALUES,
        "metadata": metadata,
    }

    summary = {
        "n_sessions": len(session_results),
        "n_subjects": len(subjects),
        "n_brain_regions": len(brain_regions),
        "n_trials_total": int(sum(len(session["neural"]) for session in session_results)),
        "n_good_units_total": int(sum(len(session["brain_region_labels"]) for session in session_results)),
        "sessions_per_subject": dict(Counter(session["subject"] for session in session_results)),
        "choice_source_counts": dict(total_choice_sources),
        "stim_trial_count_total": int(total_stim_trials),
        "tongue_bin_fallback_count_total": int(total_tongue_fallback),
    }
    return data, summary


def write_pickle(path: str, obj: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MAP NWB sessions into the decoder pickle format.")
    parser.add_argument("--data-root", default="/app/data")
    parser.add_argument("--full-out", default="/app/converted_data.pkl")
    parser.add_argument("--sample-out", default="/app/sample_data.pkl")
    parser.add_argument("--sample-sessions", type=int, default=5)
    parser.add_argument("--limit-sessions", type=int, default=None)
    args = parser.parse_args()

    all_paths = load_sorted_nwb_paths(args.data_root)
    print(f"Found {len(all_paths)} NWB files under {args.data_root}")

    session_results = []
    skipped_zero_good = []
    for idx, path in enumerate(all_paths, start=1):
        session_id = get_session_id_from_path(path)
        print(f"[{idx}/{len(all_paths)}] Converting {session_id}")
        result = convert_session(path)
        if result is None:
            skipped_zero_good.append(session_id)
            print(f"  skipped: no good units")
            continue
        session_results.append(result)
        print(
            "  kept: "
            f"{result['stats']['n_trials']} trials, "
            f"{result['stats']['n_good_units']} good units, "
            f"{result['stats']['stim_trial_count']} stim trials"
        )
        if args.limit_sessions is not None and len(session_results) >= args.limit_sessions:
            break

    full_data, full_summary = build_dataset(session_results)
    full_data["metadata"]["skipped_zero_good_sessions"] = skipped_zero_good
    full_data["metadata"]["source_nwb_count"] = len(all_paths)
    full_data["metadata"]["kept_session_count"] = len(session_results)
    write_pickle(args.full_out, full_data)

    sample_count = min(args.sample_sessions, len(session_results))
    sample_results = session_results[:sample_count]
    sample_data, sample_summary = build_dataset(sample_results)
    sample_data["metadata"]["sample_session_count"] = sample_count
    sample_data["metadata"]["sample_source"] = "first kept sessions in sorted file order"
    write_pickle(args.sample_out, sample_data)

    print("")
    print("Conversion complete.")
    print(json.dumps(
        {
            "full_summary": full_summary,
            "sample_summary": sample_summary,
            "skipped_zero_good_sessions": skipped_zero_good,
            "full_out": args.full_out,
            "sample_out": args.sample_out,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
