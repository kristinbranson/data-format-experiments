#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO


BIN_SIZE_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
NWB_GLOB = "sub-*/*.nwb"
TONGUE_VISIBILITY_THRESHOLD = 0.9


def make_time_grid():
    bin_edges = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_SIZE_S / 2, BIN_SIZE_S)
    bin_centers = bin_edges[:-1] + BIN_SIZE_S / 2
    return bin_edges, bin_centers


BIN_EDGES_REL_S, BIN_CENTERS_REL_S = make_time_grid()


def last_event_per_trial(event_times, trial_starts, trial_stops):
    left = np.searchsorted(event_times, trial_starts, side="left")
    right = np.searchsorted(event_times, trial_stops, side="right")
    out = np.full(trial_starts.shape, np.nan, dtype=np.float64)
    valid = right > left
    out[valid] = event_times[right[valid] - 1]
    return out


def choice_from_instruction_and_outcome(instruction, outcome):
    if outcome == "ignore":
        return 2
    if instruction == "left":
        return 0 if outcome == "hit" else 1
    if instruction == "right":
        return 1 if outcome == "hit" else 0
    raise ValueError(f"Unexpected trial instruction: {instruction!r}")


def outcome_to_int(outcome):
    mapping = {"ignore": 0, "miss": 1, "hit": 2}
    if outcome not in mapping:
        raise ValueError(f"Unexpected outcome: {outcome!r}")
    return mapping[outcome]


def early_lick_to_int(value):
    mapping = {"no early": 0, "early": 1}
    if value not in mapping:
        raise ValueError(f"Unexpected early_lick value: {value!r}")
    return mapping[value]


def build_tongue_states(tongue_ts, go_times_kept):
    tracking = np.asarray(tongue_ts.data[:], dtype=np.float32)
    timestamps = np.asarray(tongue_ts.timestamps[:], dtype=np.float64)
    y_all = tracking[:, 1]
    likelihood_all = tracking[:, 2]

    visible_all = likelihood_all >= TONGUE_VISIBILITY_THRESHOLD
    visible_y = y_all[visible_all]
    if visible_y.size == 0:
        p40 = np.nan
        p60 = np.nan
    else:
        p40, p60 = np.percentile(visible_y, [40, 60])

    abs_centers = go_times_kept[:, None] + BIN_CENTERS_REL_S[None, :]
    frame_idx = np.searchsorted(timestamps, abs_centers.ravel(), side="right") - 1
    frame_idx = frame_idx.reshape(abs_centers.shape)

    tongue_state = np.full(abs_centers.shape, 3, dtype=np.int8)
    valid = (frame_idx >= 0) & (frame_idx < len(timestamps))
    if np.any(valid):
        y = y_all[frame_idx[valid]]
        likelihood = likelihood_all[frame_idx[valid]]
        visible = likelihood >= TONGUE_VISIBILITY_THRESHOLD
        state = np.full(y.shape, 3, dtype=np.int8)
        if visible_y.size > 0:
            state[visible & (y < p40)] = 0
            state[visible & (y >= p40) & (y <= p60)] = 1
            state[visible & (y > p60)] = 2
        tongue_state[valid] = state

    return tongue_state, float(p40) if visible_y.size else None, float(p60) if visible_y.size else None


def build_photostim_input(trials_kept, go_times_kept):
    photostim = np.zeros((len(trials_kept), len(BIN_CENTERS_REL_S)), dtype=np.float32)
    for i, (_, trial) in enumerate(trials_kept.iterrows()):
        onset = trial["photostim_onset"]
        duration = trial["photostim_duration"]
        if onset == "N/A" or duration == "N/A":
            continue
        onset_abs = float(trial["start_time"]) + float(onset)
        offset_abs = onset_abs + float(duration)
        onset_rel = onset_abs - go_times_kept[i]
        offset_rel = offset_abs - go_times_kept[i]
        photostim[i] = ((BIN_CENTERS_REL_S >= onset_rel) & (BIN_CENTERS_REL_S < offset_rel)).astype(np.float32)
    return photostim


def build_neural_trials(units_df, keep_mask, go_times_kept):
    good_units = units_df.loc[keep_mask]
    nneurons = len(good_units)
    ntrials = len(go_times_kept)
    ntime = len(BIN_CENTERS_REL_S)

    neural = np.empty((ntrials, nneurons, ntime), dtype=np.float16)
    abs_edges = go_times_kept[:, None] + BIN_EDGES_REL_S[None, :]

    for unit_idx, spike_times in enumerate(good_units["spike_times"].to_list()):
        spikes = np.asarray(spike_times, dtype=np.float64)
        edge_idx = np.searchsorted(spikes, abs_edges.ravel(), side="left").reshape(ntrials, -1)
        counts = np.diff(edge_idx, axis=1)
        neural[:, unit_idx, :] = (counts / BIN_SIZE_S).astype(np.float16, copy=False)

    return [neural[trial_idx] for trial_idx in range(ntrials)]


def recorded_trial_mask_from_obs_intervals(trials, obs_intervals, atol=1e-4):
    obs_intervals = np.asarray(obs_intervals, dtype=np.float64)
    if obs_intervals.ndim != 2 or obs_intervals.shape[1] != 2:
        return np.ones(len(trials), dtype=bool)

    trial_starts = trials["start_time"].to_numpy(dtype=np.float64)
    trial_stops = trials["stop_time"].to_numpy(dtype=np.float64)
    keep = np.zeros(len(trials), dtype=bool)
    for obs_start, obs_stop in obs_intervals:
        keep |= np.isclose(trial_starts, obs_start, atol=atol) & np.isclose(trial_stops, obs_stop, atol=atol)
    return keep


def convert_session(path, brain_region_to_idx):
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()

        trials = nwb.trials.to_dataframe()

        units_df = nwb.units.to_dataframe()
        good_unit_mask = units_df["classification"].astype(str).to_numpy() == "good"
        if good_unit_mask.sum() == 0:
            return None

        first_good_unit = units_df.loc[good_unit_mask].iloc[0]
        recorded_trials = recorded_trial_mask_from_obs_intervals(trials, first_good_unit["obs_intervals"])
        keep_trials = (
            recorded_trials
            & (trials["auto_water"].to_numpy() == 0)
            & (trials["free_water"].to_numpy() == 0)
        )
        trials_kept = trials.loc[keep_trials].copy()
        if len(trials_kept) < 2:
            return None

        anno_names = units_df.loc[good_unit_mask, "anno_name"].astype(str).str.strip().replace("", "Unknown")
        brain_region_idx = np.empty(len(anno_names), dtype=np.int64)
        for i, region in enumerate(anno_names):
            if region not in brain_region_to_idx:
                brain_region_to_idx[region] = len(brain_region_to_idx)
            brain_region_idx[i] = brain_region_to_idx[region]

        go_times = np.asarray(
            nwb.acquisition["BehavioralEvents"].time_series["go_start_times"].timestamps[:],
            dtype=np.float64,
        )
        go_times_kept = go_times[keep_trials]

        delay_times = np.asarray(
            nwb.acquisition["BehavioralEvents"].time_series["delay_start_times"].timestamps[:],
            dtype=np.float64,
        )
        delay_last = last_event_per_trial(
            delay_times,
            trials_kept["start_time"].to_numpy(dtype=np.float64),
            trials_kept["stop_time"].to_numpy(dtype=np.float64),
        )
        tone_onset = np.where(np.isnan(delay_last), go_times_kept - 1.85, delay_last - 0.65)
        time_from_tone = (go_times_kept[:, None] + BIN_CENTERS_REL_S[None, :]) - tone_onset[:, None]

        photostim = build_photostim_input(trials_kept, go_times_kept)

        tongue_ts = nwb.acquisition["BehavioralTimeSeries"].time_series["Camera0_side_TongueTracking"]
        tongue_state, p40, p60 = build_tongue_states(tongue_ts, go_times_kept)

        neural_trials = build_neural_trials(units_df, good_unit_mask, go_times_kept)
        nonzero_trial_mask = np.array([np.any(trial != 0) for trial in neural_trials], dtype=bool)
        if nonzero_trial_mask.sum() < 2:
            return None

        trials_kept = trials_kept.iloc[nonzero_trial_mask].copy()
        go_times_kept = go_times_kept[nonzero_trial_mask]
        time_from_tone = time_from_tone[nonzero_trial_mask]
        photostim = photostim[nonzero_trial_mask]
        tongue_state = tongue_state[nonzero_trial_mask]
        neural_trials = [trial for trial, keep in zip(neural_trials, nonzero_trial_mask) if keep]

        inputs = []
        outputs = []
        for i, (_, trial) in enumerate(trials_kept.iterrows()):
            input_trial = np.vstack([time_from_tone[i], photostim[i]]).astype(np.float32, copy=False)

            choice = choice_from_instruction_and_outcome(
                str(trial["trial_instruction"]),
                str(trial["outcome"]),
            )
            outcome = outcome_to_int(str(trial["outcome"]))
            early = early_lick_to_int(str(trial["early_lick"]))

            output_trial = np.vstack(
                [
                    np.full(len(BIN_CENTERS_REL_S), choice, dtype=np.int8),
                    np.full(len(BIN_CENTERS_REL_S), outcome, dtype=np.int8),
                    np.full(len(BIN_CENTERS_REL_S), early, dtype=np.int8),
                    tongue_state[i],
                ]
            )
            inputs.append(input_trial)
            outputs.append(output_trial)

        session_info = {
            "session_id": path.stem,
            "source_file": str(path),
            "subject": str(nwb.subject.subject_id),
            "n_trials_total": int(len(trials)),
            "n_trials_recorded": int(recorded_trials.sum()),
            "n_trials_kept": int(len(trials_kept)),
            "n_trials_nonzero_neural": int(nonzero_trial_mask.sum()),
            "n_good_units": int(good_unit_mask.sum()),
            "tongue_visible_p40": p40,
            "tongue_visible_p60": p60,
        }

        return {
            "subject": str(nwb.subject.subject_id),
            "neural": neural_trials,
            "input": inputs,
            "output": outputs,
            "brain_region_idx": brain_region_idx,
            "session_info": session_info,
        }


def convert_dataset(data_dir):
    subjects = []
    subject_to_idx = {}
    brain_region_to_idx = {}

    data = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": subjects,
        "subject_idx": [],
        "brain_regions": None,
        "brain_region_idx": [],
        "input_names": ["time_from_tone_onset_s", "photostimulation_on"],
        "output_names": ["choice", "outcome", "early_lick", "tongue_y_position"],
        "output_values": [
            ["left", "right", "no lick"],
            ["ignore", "miss", "hit"],
            ["no", "yes"],
            ["<40th percentile", "40th to 60th percentile", ">60th percentile", "not visible"],
        ],
        "metadata": {
            "task_description": (
                "Auditory delayed response task; decode choice, outcome, early lick, "
                "and discretized tongue position from go-cue-aligned neural activity."
            ),
            "time_bin_size": 50.0,
            "temporal_alignment_event": "Go cue onset",
            "off_start": WINDOW_START_S,
            "off_end": WINDOW_END_S,
            "time_bin_centers_s": BIN_CENTERS_REL_S.tolist(),
            "tone_visibility_threshold": TONGUE_VISIBILITY_THRESHOLD,
            "trial_inclusion": (
                "Included all non-auto-water, non-free-water trials; retained early-lick "
                "and ignore trials because they are explicit decoder targets."
            ),
            "unit_inclusion": 'Included units with NWB units.classification == "good".',
            "source_dataset": "MAP NWB sessions in /app/data",
            "session_info": [],
        },
    }

    session_paths = sorted(Path(data_dir).glob(NWB_GLOB))
    for session_idx, path in enumerate(session_paths, start=1):
        print(f"[{session_idx}/{len(session_paths)}] Converting {path.name}", flush=True)
        session_data = convert_session(path, brain_region_to_idx)
        if session_data is None:
            print(f"  skipped {path.name} (insufficient trials or good units)", flush=True)
            continue

        subject = session_data["subject"]
        if subject not in subject_to_idx:
            subject_to_idx[subject] = len(subjects)
            subjects.append(subject)

        data["neural"].append(session_data["neural"])
        data["input"].append(session_data["input"])
        data["output"].append(session_data["output"])
        data["subject_idx"].append(subject_to_idx[subject])
        data["brain_region_idx"].append(session_data["brain_region_idx"])
        data["metadata"]["session_info"].append(session_data["session_info"])

    data["subject_idx"] = np.asarray(data["subject_idx"], dtype=np.int64)
    brain_regions = [None] * len(brain_region_to_idx)
    for region, idx in brain_region_to_idx.items():
        brain_regions[idx] = region
    data["brain_regions"] = brain_regions
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/app/data")
    parser.add_argument("--output", default="/app/converted_data.pkl")
    args = parser.parse_args()

    data = convert_dataset(args.data_dir)
    with open(args.output, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"Saved converted dataset with {len(data['neural'])} sessions to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
