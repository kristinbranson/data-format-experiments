#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np


WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_SIZE_S = 0.05
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_SIZE_S, dtype=np.float64)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + (BIN_SIZE_S / 2.0)
N_BINS = len(BIN_CENTERS_REL)


CHOICE_MAP = {"left": 0, "right": 1}
OUTCOME_MAP = {"ignore": 0, "miss": 1, "hit": 2}
EARLY_LICK_MAP = {"no early": 0, "early": 1}


def _decode_scalar(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode_str_array(dataset):
    arr = np.asarray(dataset[()])
    flat = arr.reshape(-1)
    decoded = [_decode_scalar(x) for x in flat]
    return np.asarray(decoded, dtype=object).reshape(arr.shape)


def _parse_optional_float_array(dataset):
    raw = _decode_str_array(dataset).reshape(-1)
    out = np.full(raw.shape[0], np.nan, dtype=np.float64)
    for i, value in enumerate(raw):
        if value in ("N/A", "", None):
            continue
        out[i] = float(value)
    return out


def _extract_unit_spikes(units_group, unit_idx):
    spike_times = units_group["spike_times"]
    spike_index = units_group["spike_times_index"]
    end = int(spike_index[unit_idx])
    start = 0 if unit_idx == 0 else int(spike_index[unit_idx - 1])
    return np.asarray(spike_times[start:end], dtype=np.float64)


def _extract_obs_intervals(units_group, unit_idx):
    obs = units_group["obs_intervals"]
    obs_index = units_group["obs_intervals_index"]
    end = int(obs_index[unit_idx])
    start = 0 if unit_idx == 0 else int(obs_index[unit_idx - 1])
    return np.asarray(obs[start:end], dtype=np.float64)


def _resolve_sample_onsets(trial_starts, go_starts, sample_starts):
    sample_idx = np.searchsorted(sample_starts, go_starts, side="right") - 1
    valid = sample_idx >= 0
    sample_onsets = np.full(go_starts.shape, np.nan, dtype=np.float64)
    sample_onsets[valid] = sample_starts[sample_idx[valid]]

    invalid = np.isnan(sample_onsets) | (sample_onsets < (trial_starts - 1e-9))
    if np.any(invalid):
        for i in np.where(invalid)[0]:
            mask = (sample_starts >= (trial_starts[i] - 1e-9)) & (sample_starts <= (go_starts[i] + 1e-9))
            candidates = sample_starts[mask]
            if len(candidates) == 0:
                raise RuntimeError(f"Could not resolve sample onset for trial {i}.")
            sample_onsets[i] = candidates[-1]
    return sample_onsets


def _bin_tongue_y(timestamps, y_values, go_time):
    bin_starts = go_time + BIN_EDGES_REL[:-1]
    bin_ends = go_time + BIN_EDGES_REL[1:]
    idx = np.searchsorted(timestamps, bin_ends, side="left") - 1
    valid = idx >= 0
    if np.any(valid):
        valid_idx = idx[valid]
        valid[valid] &= timestamps[valid_idx] >= bin_starts[valid]
    sampled = np.zeros(N_BINS, dtype=np.float32)
    if np.any(valid):
        sampled[valid] = y_values[idx[valid]].astype(np.float32)
    return sampled


def _subset_data(data, session_indices, trials_per_session):
    session_indices = list(session_indices)
    used_subjects = []
    subject_remap = {}
    used_regions = []
    region_remap = {}

    subset = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": [],
        "subject_idx": [],
        "brain_regions": [],
        "brain_region_idx": [],
        "input_names": list(data["input_names"]),
        "output_names": list(data["output_names"]),
        "output_values": [list(v) for v in data["output_values"]],
        "metadata": dict(data["metadata"]),
    }

    for sess_idx in session_indices:
        ntrials = min(trials_per_session, len(data["neural"][sess_idx]))
        subset["neural"].append([data["neural"][sess_idx][i] for i in range(ntrials)])
        subset["input"].append([data["input"][sess_idx][i] for i in range(ntrials)])
        subset["output"].append([data["output"][sess_idx][i] for i in range(ntrials)])

        subj_name = data["subjects"][int(data["subject_idx"][sess_idx])]
        if subj_name not in subject_remap:
            subject_remap[subj_name] = len(used_subjects)
            used_subjects.append(subj_name)
        subset["subject_idx"].append(subject_remap[subj_name])

        region_names = []
        remapped_regions = np.empty_like(data["brain_region_idx"][sess_idx], dtype=np.int64)
        for i, region_idx in enumerate(data["brain_region_idx"][sess_idx]):
            region_name = data["brain_regions"][int(region_idx)]
            if region_name not in region_remap:
                region_remap[region_name] = len(used_regions)
                used_regions.append(region_name)
            remapped_regions[i] = region_remap[region_name]
        subset["brain_region_idx"].append(remapped_regions)

    subset["subjects"] = used_subjects
    subset["subject_idx"] = np.asarray(subset["subject_idx"], dtype=np.int64)
    subset["brain_regions"] = used_regions
    subset["metadata"]["n_sessions"] = len(subset["neural"])
    subset["metadata"]["dataset_variant"] = "sample"
    return subset


def convert_dataset(data_dir, exclude_auto_free=True):
    data_dir = Path(data_dir)
    nwb_paths = sorted(data_dir.glob("sub-*/*.nwb"))
    if not nwb_paths:
        raise FileNotFoundError(f"No NWB files found under {data_dir}")

    subjects = []
    subject_to_idx = {}
    brain_regions = []
    brain_region_to_idx = {}

    neural_sessions = []
    input_sessions = []
    output_sessions = []
    subject_idx = []
    brain_region_idx_sessions = []

    summary = {
        "n_files": len(nwb_paths),
        "n_sessions": 0,
        "n_subjects": 0,
        "total_trials_raw": 0,
        "total_trials_kept": 0,
        "total_auto_water_excluded": 0,
        "total_free_water_excluded": 0,
        "total_good_units": 0,
        "per_session": [],
    }

    for session_number, path in enumerate(nwb_paths, start=1):
        with h5py.File(path, "r") as f:
            trials = f["intervals/trials"]
            n_trials = len(trials["id"])

            trial_starts = np.asarray(trials["start_time"], dtype=np.float64)
            trial_stops = np.asarray(trials["stop_time"], dtype=np.float64)
            trial_instruction = _decode_str_array(trials["trial_instruction"]).reshape(-1)
            outcomes = _decode_str_array(trials["outcome"]).reshape(-1)
            early_lick = _decode_str_array(trials["early_lick"]).reshape(-1)
            auto_water = np.asarray(trials["auto_water"], dtype=np.int64)
            free_water = np.asarray(trials["free_water"], dtype=np.int64)
            photostim_onset = _parse_optional_float_array(trials["photostim_onset"])
            photostim_duration = _parse_optional_float_array(trials["photostim_duration"])
            photostim_power = _parse_optional_float_array(trials["photostim_power"])

            trial_keep = np.ones(n_trials, dtype=bool)
            if exclude_auto_free:
                trial_keep &= auto_water == 0
                trial_keep &= free_water == 0

            be = f["acquisition/BehavioralEvents"]
            go_starts = np.asarray(be["go_start_times"]["timestamps"], dtype=np.float64)
            if len(go_starts) != n_trials:
                raise RuntimeError(f"{path.name}: expected {n_trials} go cues, found {len(go_starts)}")
            sample_starts = np.asarray(be["sample_start_times"]["timestamps"], dtype=np.float64)
            tone_onsets = _resolve_sample_onsets(trial_starts, go_starts, sample_starts)

            tongue_group = f["acquisition/BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
            tongue_timestamps = np.asarray(tongue_group["timestamps"], dtype=np.float64)
            tongue_y_all = np.asarray(tongue_group["data"][:, 1], dtype=np.float32)
            tongue_q40, tongue_q60 = np.percentile(tongue_y_all, [40, 60])

            units = f["units"]
            classification = _decode_str_array(units["classification"]).reshape(-1)
            anno_name = _decode_str_array(units["anno_name"]).reshape(-1)
            good_unit_indices = np.flatnonzero(classification == "good")

            if len(good_unit_indices) == 0:
                continue

            is_good_trials = np.asarray(units["is_good_trials"][good_unit_indices]).astype(bool)
            if is_good_trials.ndim == 1:
                is_good_trials = is_good_trials[np.newaxis, :]
            recorded_trial_mask = np.zeros(n_trials, dtype=bool)
            first_obs_intervals = _extract_obs_intervals(units, int(good_unit_indices[0]))
            if first_obs_intervals.ndim == 2 and len(first_obs_intervals) > 0:
                n_recorded_trials = min(len(first_obs_intervals), is_good_trials.shape[1])
                common_obs_mask = np.all(is_good_trials[:, :n_recorded_trials], axis=0)
                interval_starts = first_obs_intervals[:n_recorded_trials, 0]
                interval_stops = first_obs_intervals[:n_recorded_trials, 1]
                mapped_idx = np.searchsorted(trial_starts, interval_starts, side="left")
                valid = mapped_idx < n_trials
                valid &= np.isclose(trial_starts[mapped_idx], interval_starts, atol=1e-4)
                valid &= np.isclose(trial_stops[mapped_idx], interval_stops, atol=1e-4)
                if np.sum(valid) == n_recorded_trials:
                    recorded_trial_mask[mapped_idx[common_obs_mask]] = True
                else:
                    recorded_trial_mask[:n_recorded_trials] = common_obs_mask
            else:
                n_recorded_trials = min(n_trials, is_good_trials.shape[1])
                recorded_trial_mask[:n_recorded_trials] = np.all(is_good_trials[:, :n_recorded_trials], axis=0)
            trial_keep &= recorded_trial_mask

            good_region_idx = np.empty(len(good_unit_indices), dtype=np.int64)
            for i, unit_idx in enumerate(good_unit_indices):
                region_name = str(anno_name[unit_idx])
                if region_name not in brain_region_to_idx:
                    brain_region_to_idx[region_name] = len(brain_regions)
                    brain_regions.append(region_name)
                good_region_idx[i] = brain_region_to_idx[region_name]

            if len(good_region_idx) == 0:
                continue

            keep_trial_indices = np.flatnonzero(trial_keep)
            n_keep_trials = len(keep_trial_indices)
            session_neural = np.zeros((n_keep_trials, len(good_unit_indices), N_BINS), dtype=np.float32)
            session_input = []
            session_output = []

            abs_edges = go_starts[trial_keep, None] + BIN_EDGES_REL[None, :]
            abs_centers = go_starts[trial_keep, None] + BIN_CENTERS_REL[None, :]

            for pos, unit_idx in enumerate(good_unit_indices):
                unit_spikes = _extract_unit_spikes(units, int(unit_idx))
                spike_bins = np.searchsorted(unit_spikes, abs_edges, side="left")
                spike_counts = np.diff(spike_bins, axis=1).astype(np.float32) / np.float32(BIN_SIZE_S)
                session_neural[:, pos, :] = spike_counts

            kept_tone_onsets = tone_onsets[trial_keep]
            kept_trial_starts = trial_starts[trial_keep]
            kept_photostim_onset = photostim_onset[trial_keep]
            kept_photostim_duration = photostim_duration[trial_keep]
            kept_trial_instruction = trial_instruction[trial_keep]
            kept_outcomes = outcomes[trial_keep]
            kept_early = early_lick[trial_keep]

            for local_idx, trial_idx in enumerate(keep_trial_indices):
                tone_rel = kept_tone_onsets[local_idx] - go_starts[trial_idx]
                time_from_tone = (BIN_CENTERS_REL - tone_rel).astype(np.float32)

                stim_on = np.zeros(N_BINS, dtype=np.float32)
                if np.isfinite(kept_photostim_onset[local_idx]) and np.isfinite(kept_photostim_duration[local_idx]):
                    stim_start_abs = kept_trial_starts[local_idx] + kept_photostim_onset[local_idx]
                    stim_stop_abs = stim_start_abs + kept_photostim_duration[local_idx]
                    stim_on = ((abs_centers[local_idx] >= stim_start_abs) & (abs_centers[local_idx] < stim_stop_abs)).astype(np.float32)

                tongue_y_trial = _bin_tongue_y(tongue_timestamps, tongue_y_all, go_starts[trial_idx])
                tongue_cat = np.zeros(N_BINS, dtype=np.int16)
                tongue_cat[tongue_y_trial >= tongue_q40] = 1
                tongue_cat[tongue_y_trial > tongue_q60] = 2

                choice_value = CHOICE_MAP[str(kept_trial_instruction[local_idx])]
                outcome_value = OUTCOME_MAP[str(kept_outcomes[local_idx])]
                early_value = EARLY_LICK_MAP[str(kept_early[local_idx])]

                output_trial = np.vstack(
                    [
                        np.full(N_BINS, choice_value, dtype=np.int16),
                        np.full(N_BINS, outcome_value, dtype=np.int16),
                        np.full(N_BINS, early_value, dtype=np.int16),
                        tongue_cat,
                    ]
                )
                input_trial = np.vstack([time_from_tone, stim_on]).astype(np.float32)

                session_input.append(input_trial)
                session_output.append(output_trial)

            nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
            zero_only_trials = int((~nonzero_trial_mask).sum())
            session_neural = session_neural[nonzero_trial_mask]
            session_input = [session_input[i] for i in np.flatnonzero(nonzero_trial_mask)]
            session_output = [session_output[i] for i in np.flatnonzero(nonzero_trial_mask)]
            n_keep_trials = int(session_neural.shape[0])
            if n_keep_trials < 2:
                continue

            neural_sessions.append([np.ascontiguousarray(session_neural[i]) for i in range(n_keep_trials)])
            input_sessions.append(session_input)
            output_sessions.append(session_output)
            brain_region_idx_sessions.append(good_region_idx)

            subject = _decode_scalar(f["general"]["subject"]["subject_id"][()])
            subject = str(subject)
            if subject not in subject_to_idx:
                subject_to_idx[subject] = len(subjects)
                subjects.append(subject)
            subject_idx.append(subject_to_idx[subject])

            summary["n_sessions"] += 1
            summary["total_trials_raw"] += n_trials
            summary["total_trials_kept"] += n_keep_trials
            summary["total_auto_water_excluded"] += int(auto_water.sum())
            summary["total_free_water_excluded"] += int(free_water.sum())
            summary["total_good_units"] += len(good_unit_indices)
            summary["per_session"].append(
                {
                    "session_file": path.name,
                    "subject": subject,
                    "trials_raw": int(n_trials),
                    "trials_kept": int(n_keep_trials),
                    "trials_with_ephys": int(recorded_trial_mask.sum()),
                    "trials_all_zero_excluded": int(zero_only_trials),
                    "good_units": int(len(good_unit_indices)),
                    "tongue_q40": float(tongue_q40),
                    "tongue_q60": float(tongue_q60),
                    "photostim_trials": int(np.isfinite(kept_photostim_onset).sum()),
                    "mean_trial_duration_s": float(np.mean(trial_stops[trial_keep] - trial_starts[trial_keep])),
                    "mean_tone_onset_rel_go_s": float(np.mean(kept_tone_onsets - go_starts[trial_keep])),
                }
            )

        if (session_number % 10 == 0) or (session_number == len(nwb_paths)):
            print(
                f"[{session_number:03d}/{len(nwb_paths):03d}] "
                f"sessions={summary['n_sessions']} "
                f"trials_kept={summary['total_trials_kept']} "
                f"good_units={summary['total_good_units']}"
            )

    summary["n_subjects"] = len(subjects)

    data = {
        "neural": neural_sessions,
        "input": input_sessions,
        "output": output_sessions,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": brain_regions,
        "brain_region_idx": brain_region_idx_sessions,
        "input_names": ["time_from_tone_onset_s", "photostimulation_on"],
        "output_names": ["choice", "outcome", "early_lick", "tongue_y_position"],
        "output_values": [
            ["left", "right"],
            ["ignore", "miss", "hit"],
            ["no", "yes"],
            ["lt_40th_pct", "40th_to_60th_pct", "gt_60th_pct"],
        ],
        "metadata": {
            "task_description": (
                "Auditory delayed-response task. Neural activity is aligned to go cue onset; "
                "decoder predicts left/right trial type, outcome, early lick, and discretized tongue y-position."
            ),
            "time_bin_size": float(BIN_SIZE_S * 1000.0),
            "temporal_alignment_event": "go cue onset",
            "off_start": float(WINDOW_START_S),
            "off_end": float(WINDOW_END_S),
            "n_timepoints": int(N_BINS),
            "bin_edges_s": BIN_EDGES_REL.tolist(),
            "bin_centers_s": BIN_CENTERS_REL.tolist(),
            "trial_filtering": {
                "exclude_auto_water": bool(exclude_auto_free),
                "exclude_free_water": bool(exclude_auto_free),
                "exclude_trials_without_common_good_ephys": True,
                "keep_early_lick_trials": True,
                "keep_ignore_trials": True,
                "keep_photostim_trials": True,
            },
            "neural_processing": {
                "unit_inclusion": "NWB units/classification == 'good'",
                "binning": "non-overlapping 50 ms spike-count bins converted to firing rates (Hz)",
            },
            "input_processing": {
                "time_from_tone_onset_s": "bin center minus final sample-start time for that trial",
                "photostimulation_on": "1 if bin center falls within trial photostim interval, else 0",
            },
            "output_processing": {
                "choice": "mapped from trial_instruction to match the paper code's left/right trial label",
                "outcome": "mapped from NWB trial outcome",
                "early_lick": "mapped from NWB early_lick field",
                "tongue_y_position": "per-session 40/60 percentile bins from the full tongue_y session trace",
            },
            "reference_summary_discrepancy_note": (
                "The NWB release in /app/data contains 174 files and 69,453 classifier-good units, "
                "which differs slightly from the 173 sessions and 69,943 good units quoted in methods.txt."
            ),
        },
    }
    return data, summary


def main():
    parser = argparse.ArgumentParser(description="Convert the MAP NWB dataset into the decoder pickle format.")
    parser.add_argument("--data-dir", default="/app/data")
    parser.add_argument("--full-out", default="/app/converted_data.pkl")
    parser.add_argument("--sample-out", default="/app/sample_data.pkl")
    parser.add_argument("--summary-json", default="/app/conversion_summary.json")
    parser.add_argument("--sample-sessions", type=int, default=3)
    parser.add_argument("--sample-trials-per-session", type=int, default=32)
    args = parser.parse_args()

    print("Starting conversion")
    data, summary = convert_dataset(args.data_dir)

    summary["input_names"] = data["input_names"]
    summary["output_names"] = data["output_names"]
    summary["brain_region_count"] = len(data["brain_regions"])
    summary["timepoints_per_trial"] = N_BINS

    with open(args.full_out, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote full dataset to {args.full_out}")

    sample_sessions = list(range(min(args.sample_sessions, len(data["neural"]))))
    sample_data = _subset_data(data, sample_sessions, args.sample_trials_per_session)
    with open(args.sample_out, "wb") as f:
        pickle.dump(sample_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote sample dataset to {args.sample_out}")

    with open(args.summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote conversion summary to {args.summary_json}")

    print("Conversion summary:")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_session"}, indent=2))


if __name__ == "__main__":
    main()
