import glob
import json
import math
import os
import pickle

import h5py
import numpy as np


DATA_GLOB = "/app/data/sub-*/*.nwb"
OUTPUT_PATH = "/app/converted_data.pkl"

ALIGN_EVENT = "Go cue onset"
BEGIN_TIME = -2.5
END_TIME = 1.5
BIN_WIDTH = 0.05
BIN_STRIDE = 0.05
VISIBILITY_THRESHOLD = 0.9
REGIONS_TO_KEEP = {"left ALM", "right ALM"}
MIN_CONTROL_PERFORMANCE = 0.65
MIN_CORRECT_PER_SIDE = 50


def decode_array(dataset):
    arr = np.asarray(dataset)
    if arr.dtype.kind == "S":
        return arr.astype(str)
    if arr.dtype == object:
        flat = arr.reshape(-1)
        decoded = [
            item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
            for item in flat
        ]
        return np.asarray(decoded, dtype=object).reshape(arr.shape)
    return arr


def decode_scalar(dataset):
    value = np.asarray(dataset)[()]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def compute_bin_centers(begin_time, end_time, stride):
    span = (end_time - begin_time) / stride
    if np.allclose(span, math.floor(span) + 1):
        n_bins = math.floor(span) + 2
    else:
        n_bins = math.floor(span) + 1
    return begin_time + np.arange(n_bins) * stride


BIN_CENTERS = compute_bin_centers(BEGIN_TIME, END_TIME, BIN_STRIDE).astype(np.float32)
BIN_EDGES = np.concatenate(
    [BIN_CENTERS - BIN_WIDTH / 2.0, [BIN_CENTERS[-1] + BIN_WIDTH / 2.0]]
).astype(np.float64)
N_BINS = len(BIN_CENTERS)


def assign_events_to_windows(event_times, window_starts, window_ends, bin_width, n_bins):
    counts = np.zeros((len(window_starts), n_bins), dtype=np.uint16)
    if event_times.size == 0:
        return counts

    trial_idx = np.searchsorted(window_starts, event_times, side="right") - 1
    valid = trial_idx >= 0
    if not np.any(valid):
        return counts

    trial_idx = trial_idx[valid]
    event_times = event_times[valid]
    valid = event_times < window_ends[trial_idx]
    if not np.any(valid):
        return counts

    trial_idx = trial_idx[valid]
    event_times = event_times[valid]
    bin_idx = np.floor((event_times - window_starts[trial_idx]) / bin_width).astype(np.int64)
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    if not np.any(valid):
        return counts

    np.add.at(counts, (trial_idx[valid], bin_idx[valid]), 1)
    return counts


def build_tongue_bins(track_times, track_y, track_prob, go_times):
    tongue_bins = np.full((len(go_times), N_BINS), 3, dtype=np.int64)
    visible = track_prob > VISIBILITY_THRESHOLD
    if not np.any(visible):
        return tongue_bins, np.nan, np.nan

    y_visible = track_y[visible]
    p40, p60 = np.percentile(y_visible, [40, 60])

    window_starts = go_times + BIN_EDGES[0]
    window_ends = go_times + BIN_EDGES[-1]

    trial_idx = np.searchsorted(window_starts, track_times, side="right") - 1
    valid = trial_idx >= 0
    trial_idx = trial_idx[valid]
    times = track_times[valid]
    ys = track_y[valid]
    probs = track_prob[valid]

    valid = times < window_ends[trial_idx]
    trial_idx = trial_idx[valid]
    times = times[valid]
    ys = ys[valid]
    probs = probs[valid]

    valid = probs > VISIBILITY_THRESHOLD
    trial_idx = trial_idx[valid]
    times = times[valid]
    ys = ys[valid]

    if times.size == 0:
        return tongue_bins, float(p40), float(p60)

    bin_idx = np.floor((times - window_starts[trial_idx]) / BIN_WIDTH).astype(np.int64)
    valid = (bin_idx >= 0) & (bin_idx < N_BINS)
    trial_idx = trial_idx[valid]
    bin_idx = bin_idx[valid]
    ys = ys[valid]

    classes = np.full(len(ys), 1, dtype=np.int64)
    classes[ys < p40] = 0
    classes[ys > p60] = 2

    flat = tongue_bins.reshape(-1)
    flat[trial_idx * N_BINS + bin_idx] = classes
    return tongue_bins, float(p40), float(p60)


def first_choice_after_go(left_licks, right_licks, go_time, response_window=1.5):
    left_start = np.searchsorted(left_licks, go_time, side="left")
    left_end = np.searchsorted(left_licks, go_time + response_window, side="left")
    right_start = np.searchsorted(right_licks, go_time, side="left")
    right_end = np.searchsorted(right_licks, go_time + response_window, side="left")

    first_left = left_licks[left_start] if left_start < left_end else np.inf
    first_right = right_licks[right_start] if right_start < right_end else np.inf

    if np.isfinite(first_left) and first_left < first_right:
        return 0
    if np.isfinite(first_right):
        return 1
    return 2


def parse_photostim_series(trial_start, photostim_onset, photostim_duration, go_time):
    stim = np.zeros(N_BINS, dtype=np.float32)
    if photostim_onset == "N/A":
        return stim

    onset_abs = trial_start + float(photostim_onset)
    offset_abs = onset_abs + float(photostim_duration)
    rel_on = onset_abs - go_time
    rel_off = offset_abs - go_time
    stim[(BIN_CENTERS >= rel_on) & (BIN_CENTERS < rel_off)] = 1.0
    return stim


def session_behavior_metrics(outcome, trial_instruction, early_lick, photostim_onset):
    is_control = photostim_onset == "N/A"
    no_early = early_lick == "no early"
    instructed = np.isin(trial_instruction, ["left", "right"])
    behavior_trials = no_early & is_control & instructed & np.isin(outcome, ["hit", "miss", "ignore"])

    n_control_trials = int(np.sum(behavior_trials))
    left_correct = int(np.sum(behavior_trials & (trial_instruction == "left") & (outcome == "hit")))
    right_correct = int(np.sum(behavior_trials & (trial_instruction == "right") & (outcome == "hit")))
    performance = float((left_correct + right_correct) / n_control_trials) if n_control_trials else float("nan")

    return {
        "n_control_trials": n_control_trials,
        "left_correct_control_trials": left_correct,
        "right_correct_control_trials": right_correct,
        "control_trial_performance": performance,
        "passes_filter": (
            n_control_trials > 0
            and performance > MIN_CONTROL_PERFORMANCE
            and left_correct >= MIN_CORRECT_PER_SIDE
            and right_correct >= MIN_CORRECT_PER_SIDE
        ),
    }


def session_region_labels(h5_file, units_to_keep):
    electrodes_flat = h5_file["units/electrodes"][()]
    electrodes_index = h5_file["units/electrodes_index"][()]
    electrode_starts = np.r_[0, electrodes_index[:-1]]
    electrode_locs = decode_array(h5_file["general/extracellular_ephys/electrodes/location"])
    electrode_regions = np.asarray(
        [json.loads(loc)["brain_regions"] for loc in electrode_locs], dtype=object
    )

    unit_regions = np.asarray(
        [electrode_regions[electrodes_flat[electrode_starts[idx]]] for idx in units_to_keep],
        dtype=object,
    )
    return unit_regions


def process_session(path, region_to_index):
    with h5py.File(path, "r") as f:
        subject = decode_scalar(f["general/subject/subject_id"])

        classification = decode_array(f["units/classification"])
        spike_times_flat = f["units/spike_times"][()]
        spike_times_index = f["units/spike_times_index"][()]
        spike_starts = np.r_[0, spike_times_index[:-1]]

        unit_regions = session_region_labels(f, np.arange(len(classification)))
        keep_mask = (classification == "good") & np.isin(unit_regions, list(REGIONS_TO_KEEP))
        keep_units = np.flatnonzero(keep_mask)
        if keep_units.size == 0:
            return None

        kept_regions = unit_regions[keep_units]
        brain_region_idx = np.asarray([region_to_index[reg] for reg in kept_regions], dtype=np.int64)

        trials = f["intervals/trials"]
        n_trials = len(trials["id"])
        trial_start = trials["start_time"][()]
        outcome = decode_array(trials["outcome"])
        trial_instruction = decode_array(trials["trial_instruction"])
        early = decode_array(trials["early_lick"])
        photostim_onset = decode_array(trials["photostim_onset"])
        photostim_duration = decode_array(trials["photostim_duration"])

        behavior_metrics = session_behavior_metrics(
            outcome, trial_instruction, early, photostim_onset
        )
        if not behavior_metrics["passes_filter"]:
            return None

        go_times = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()]
        left_licks = f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()]
        right_licks = f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()]

        track = np.asarray(f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data"])
        track_times = np.asarray(
            f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps"]
        )
        tongue_bins, p40, p60 = build_tongue_bins(track_times, track[:, 1], track[:, 2], go_times)

        window_starts = go_times + BIN_EDGES[0]
        window_ends = go_times + BIN_EDGES[-1]

        rates = np.empty((keep_units.size, n_trials, N_BINS), dtype=np.float16)
        for out_idx, unit_idx in enumerate(keep_units):
            spikes = spike_times_flat[spike_starts[unit_idx] : spike_times_index[unit_idx]]
            counts = assign_events_to_windows(spikes, window_starts, window_ends, BIN_WIDTH, N_BINS)
            rates[out_idx] = (counts / BIN_WIDTH).astype(np.float16)

        keep_trials = np.any(rates != 0, axis=(0, 2))
        if not np.any(keep_trials):
            return None

        kept_trial_indices = np.flatnonzero(keep_trials)
        neural_trials = [rates[:, trial_idx, :].copy() for trial_idx in kept_trial_indices]
        input_trials = []
        output_trials = []

        for trial_idx in kept_trial_indices:
            stim_series = parse_photostim_series(
                trial_start[trial_idx],
                photostim_onset[trial_idx],
                photostim_duration[trial_idx],
                go_times[trial_idx],
            )
            input_trials.append(
                np.vstack([BIN_CENTERS, stim_series]).astype(np.float32, copy=False)
            )

            choice = first_choice_after_go(left_licks, right_licks, go_times[trial_idx])
            outcome_label = {"ignore": 0, "miss": 1, "hit": 2}[outcome[trial_idx]]
            early_label = {"no early": 0, "early": 1}[early[trial_idx]]
            output_trials.append(
                np.vstack(
                    [
                        np.full(N_BINS, choice, dtype=np.int64),
                        np.full(N_BINS, outcome_label, dtype=np.int64),
                        np.full(N_BINS, early_label, dtype=np.int64),
                        tongue_bins[trial_idx].astype(np.int64, copy=False),
                    ]
                )
            )

        session_metadata = {
            "file": os.path.basename(path),
            "subject": subject,
            "n_trials_original": n_trials,
            "n_trials_kept": int(len(kept_trial_indices)),
            "n_neurons": int(keep_units.size),
            "tongue_visible_y_40": p40,
            "tongue_visible_y_60": p60,
            **behavior_metrics,
        }

        return {
            "subject": subject,
            "neural": neural_trials,
            "input": input_trials,
            "output": output_trials,
            "brain_region_idx": brain_region_idx,
            "session_metadata": session_metadata,
        }


def main():
    region_names = sorted(REGIONS_TO_KEEP)
    region_to_index = {name: idx for idx, name in enumerate(region_names)}

    data = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": [],
        "subject_idx": None,
        "brain_regions": region_names,
        "brain_region_idx": [],
        "input_names": ["time_from_tone_onset_s", "photostimulation_on"],
        "output_names": ["choice", "outcome", "early_lick", "tongue_y_position"],
        "output_values": [
            ["left", "right", "no lick"],
            ["ignore", "miss", "hit"],
            ["no", "yes"],
            ["<40th percentile", "40th-60th percentile", ">60th percentile", "not visible"],
        ],
        "metadata": {
            "task_description": (
                "Auditory delayed-response task decoded from ALM neural activity. "
                "Trials are aligned to Go-cue onset and include behavioral labels plus "
                "time-varying tongue position and photostimulation state."
            ),
            "time_bin_size": BIN_WIDTH * 1000.0,
            "temporal_alignment_event": ALIGN_EVENT,
            "off_start": BEGIN_TIME,
            "off_end": END_TIME,
            "unit_qc": "classification == good",
            "session_qc": (
                "control-trial performance > 0.65, excluding early-lick trials, and at least "
                "50 correct left plus 50 correct right control trials per session"
            ),
            "region_subset": sorted(REGIONS_TO_KEEP),
            "tongue_visibility_threshold": VISIBILITY_THRESHOLD,
            "tongue_percentiles_computed_over": "all visible side-view tongue frames within session",
            "notes": (
                "The NWB release contains multi-area recordings. The reference preprocessing "
                "code analyzes one region at a time, so this export keeps ALM units only to "
                "match the behaviorally central, photoinhibited region. Sessions are filtered "
                "using the behavioral criteria stated in the methods paper, but all trials "
                "within retained sessions are kept because outcome and early-lick labels are "
                "decoder targets in this task."
            ),
            "session_info": [],
        },
    }

    subject_to_index = {}
    subject_idx = []

    for path in sorted(glob.glob(DATA_GLOB)):
        session = process_session(path, region_to_index)
        if session is None:
            continue

        subject = session["subject"]
        if subject not in subject_to_index:
            subject_to_index[subject] = len(data["subjects"])
            data["subjects"].append(subject)

        subject_idx.append(subject_to_index[subject])
        data["neural"].append(session["neural"])
        data["input"].append(session["input"])
        data["output"].append(session["output"])
        data["brain_region_idx"].append(session["brain_region_idx"])
        data["metadata"]["session_info"].append(session["session_metadata"])

    data["subject_idx"] = np.asarray(subject_idx, dtype=np.int64)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_trials = sum(len(session_trials) for session_trials in data["neural"])
    total_neurons = int(sum(len(idx) for idx in data["brain_region_idx"]))
    print(f"Saved {OUTPUT_PATH}")
    print(f"Sessions: {len(data['neural'])}")
    print(f"Subjects: {len(data['subjects'])}")
    print(f"Trials: {total_trials}")
    print(f"Neurons: {total_neurons}")
    print(f"Brain regions: {data['brain_regions']}")


if __name__ == "__main__":
    main()
