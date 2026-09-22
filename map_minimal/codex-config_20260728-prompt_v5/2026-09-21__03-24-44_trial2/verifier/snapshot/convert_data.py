from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO


DATA_DIR = Path("/app/data")
DEFAULT_OUTPUT = Path("/app/converted_data.pkl")

WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_SIZE_S = 0.05
N_BINS = int(round((WINDOW_END_S - WINDOW_START_S) / BIN_SIZE_S))
BIN_EDGES_S = np.linspace(WINDOW_START_S, WINDOW_END_S, N_BINS + 1, dtype=np.float64)
BIN_CENTERS_S = (BIN_EDGES_S[:-1] + BIN_EDGES_S[1:]) / 2.0

TONGUE_VISIBLE_THRESHOLD = 0.9

CHOICE_TO_INT = {"left": 0, "right": 1, "no lick": 2}
OUTCOME_TO_INT = {"ignore": 0, "miss": 1, "hit": 2}
EARLY_TO_INT = {"no early": 0, "early": 1}


def iter_session_paths(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("sub-*/*.nwb"))


def maybe_float(value) -> float | None:
    text = str(value)
    if text == "N/A":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def get_vector_strings(table, column: str) -> np.ndarray:
    raw = table[column].data[:]
    return np.asarray(["" if x is None else str(x) for x in raw], dtype=object)


def get_go_times_and_response_ends(nwb, n_trials: int) -> tuple[np.ndarray, np.ndarray]:
    events = nwb.acquisition["BehavioralEvents"].time_series
    go_times_all = np.asarray(events["go_start_times"].timestamps[:], dtype=np.float64)
    response_ends_all = np.asarray(events["go_stop_times"].timestamps[:], dtype=np.float64)
    if len(go_times_all) < n_trials or len(response_ends_all) < n_trials:
        raise ValueError(
            f"go event count mismatch: n_trials={n_trials}, go={len(go_times_all)}, go_stop={len(response_ends_all)}"
        )
    return go_times_all[:n_trials], response_ends_all[:n_trials]


def get_sample_onsets(nwb, trial_starts: np.ndarray, go_times: np.ndarray, trial_stops: np.ndarray) -> np.ndarray:
    events = nwb.acquisition["BehavioralEvents"].time_series
    sample_starts = np.asarray(events["sample_start_times"].timestamps[:], dtype=np.float64)
    per_trial = np.full(len(go_times), np.nan, dtype=np.float64)

    for i, (trial_start, go_time) in enumerate(zip(trial_starts, go_times)):
        lo = np.searchsorted(sample_starts, trial_start, side="left")
        hi = np.searchsorted(sample_starts, go_time, side="right")
        if hi > lo:
            per_trial[i] = sample_starts[hi - 1]

    valid = ~np.isnan(per_trial)
    if np.any(valid):
        median_go_minus_sample = float(np.median(go_times[valid] - per_trial[valid]))
    else:
        median_go_minus_sample = 1.85

    for i in np.where(~valid)[0]:
        trial_start = trial_starts[i]
        trial_stop = trial_stops[i]
        lo = np.searchsorted(sample_starts, trial_start, side="left")
        hi = np.searchsorted(sample_starts, trial_stop, side="right")
        if hi > lo:
            per_trial[i] = sample_starts[hi - 1]
        else:
            per_trial[i] = go_times[i] - median_go_minus_sample

    return per_trial


def get_first_response_choices(
    nwb, go_times: np.ndarray, response_ends: np.ndarray
) -> np.ndarray:
    events = nwb.acquisition["BehavioralEvents"].time_series
    left_licks = np.asarray(events["left_lick_times"].timestamps[:], dtype=np.float64)
    right_licks = np.asarray(events["right_lick_times"].timestamps[:], dtype=np.float64)

    choices = np.empty(len(go_times), dtype=np.int8)
    for i, (go_time, response_end) in enumerate(zip(go_times, response_ends)):
        left_idx = np.searchsorted(left_licks, go_time, side="left")
        right_idx = np.searchsorted(right_licks, go_time, side="left")

        left_time = left_licks[left_idx] if left_idx < len(left_licks) else np.inf
        right_time = right_licks[right_idx] if right_idx < len(right_licks) else np.inf

        if left_time >= response_end:
            left_time = np.inf
        if right_time >= response_end:
            right_time = np.inf

        if left_time < right_time:
            choices[i] = CHOICE_TO_INT["left"]
        elif right_time < left_time:
            choices[i] = CHOICE_TO_INT["right"]
        else:
            choices[i] = CHOICE_TO_INT["no lick"]

    return choices


def get_photostim_relative_intervals(trials_df, go_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stim_starts = np.full(len(trials_df), np.nan, dtype=np.float64)
    stim_ends = np.full(len(trials_df), np.nan, dtype=np.float64)

    for i, trial in enumerate(trials_df.itertuples()):
        onset = maybe_float(trial.photostim_onset)
        duration = maybe_float(trial.photostim_duration)
        if onset is None or duration is None:
            continue
        start_abs = float(trial.start_time) + onset
        end_abs = start_abs + duration
        stim_starts[i] = start_abs - go_times[i]
        stim_ends[i] = end_abs - go_times[i]

    return stim_starts, stim_ends


def build_tongue_binned_labels(
    nwb, go_times: np.ndarray, likelihood_threshold: float
) -> tuple[list[np.ndarray], tuple[float, float]]:
    ts = nwb.acquisition["BehavioralTimeSeries"].time_series["Camera0_side_TongueTracking"]
    timestamps = np.asarray(ts.timestamps[:], dtype=np.float64)
    data = np.asarray(ts.data[:], dtype=np.float32)
    y = data[:, 1]
    likelihood = data[:, 2]

    visible = likelihood >= likelihood_threshold
    if np.any(visible):
        q40, q60 = np.quantile(y[visible], [0.4, 0.6]).astype(np.float32)
    else:
        q40, q60 = np.quantile(y, [0.4, 0.6]).astype(np.float32)

    per_trial = []
    for go_time in go_times:
        bin_starts = go_time + BIN_EDGES_S[:-1]
        bin_ends = go_time + BIN_EDGES_S[1:]
        frame_idx = np.searchsorted(timestamps, bin_ends, side="left") - 1
        labels = np.full(N_BINS, 3, dtype=np.int8)

        valid = (frame_idx >= 0) & (timestamps[np.clip(frame_idx, 0, len(timestamps) - 1)] >= bin_starts)
        if np.any(valid):
            idx = frame_idx[valid]
            y_valid = y[idx]
            like_valid = likelihood[idx] >= likelihood_threshold
            valid_bins = np.where(valid)[0]
            visible_bins = valid_bins[like_valid]
            if len(visible_bins):
                y_vis = y_valid[like_valid]
                labels[visible_bins[y_vis < q40]] = 0
                mid = (y_vis >= q40) & (y_vis <= q60)
                labels[visible_bins[mid]] = 1
                labels[visible_bins[y_vis > q60]] = 2

        per_trial.append(labels)

    return per_trial, (float(q40), float(q60))


def build_neural_trials(nwb, good_unit_indices: np.ndarray, go_times: np.ndarray) -> list[np.ndarray]:
    units = nwb.units
    spike_ends = np.asarray(units["spike_times"].data[:], dtype=np.int64)
    spike_values = np.asarray(units["spike_times"].target.data[:], dtype=np.float64)

    n_trials = len(go_times)
    n_units = len(good_unit_indices)
    counts = np.zeros((n_units, n_trials, N_BINS), dtype=np.uint16)

    window_starts = go_times + WINDOW_START_S
    window_ends = go_times + WINDOW_END_S

    for unit_pos, unit_idx in enumerate(good_unit_indices):
        start = 0 if unit_idx == 0 else spike_ends[unit_idx - 1]
        end = spike_ends[unit_idx]
        spikes = spike_values[start:end]
        if len(spikes) == 0:
            continue

        trial_idx = np.searchsorted(window_ends, spikes, side="right")
        valid = trial_idx < n_trials
        if not np.any(valid):
            continue

        spikes = spikes[valid]
        trial_idx = trial_idx[valid]

        in_window = spikes >= window_starts[trial_idx]
        if not np.any(in_window):
            continue

        spikes = spikes[in_window]
        trial_idx = trial_idx[in_window]
        rel_spikes = spikes - go_times[trial_idx]
        bin_idx = np.floor((rel_spikes - WINDOW_START_S) / BIN_SIZE_S).astype(np.int64)
        good = (bin_idx >= 0) & (bin_idx < N_BINS)
        if np.any(good):
            np.add.at(counts[unit_pos], (trial_idx[good], bin_idx[good]), 1)

    neural_trials = []
    for trial_idx in range(n_trials):
        trial_rates = counts[:, trial_idx, :].astype(np.float16) / np.float16(BIN_SIZE_S)
        neural_trials.append(trial_rates)

    return neural_trials


def compute_control_performance(trials_df) -> tuple[float, int, int, int]:
    early = trials_df["early_lick"].astype(str).to_numpy() == "early"
    auto = trials_df["auto_water"].to_numpy().astype(int) == 1
    free = trials_df["free_water"].to_numpy().astype(int) == 1
    control = trials_df["photostim_duration"].astype(str).to_numpy() == "N/A"
    outcome = trials_df["outcome"].astype(str).to_numpy()
    instruction = trials_df["trial_instruction"].astype(str).to_numpy()

    responded = outcome == "hit"
    errors = outcome == "miss"
    regular = (~early) & (~auto) & (~free) & control & (responded | errors)
    n_regular = int(np.sum(regular))
    if n_regular == 0:
        return float("nan"), 0, 0, 0

    left_hits = int(np.sum(regular & responded & (instruction == "left")))
    right_hits = int(np.sum(regular & responded & (instruction == "right")))
    performance = float(np.sum(regular & responded) / n_regular)
    return performance, left_hits, right_hits, n_regular


def convert_session(path: Path, brain_region_to_idx: dict[str, int]) -> tuple[dict | None, dict[str, int]]:
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        trials_df = nwb.trials.to_dataframe()
        n_trials = len(trials_df)
        if n_trials < 2:
            return None, brain_region_to_idx

        units = nwb.units
        classifications = get_vector_strings(units, "classification")
        anno_names = get_vector_strings(units, "anno_name")
        good_mask = (classifications == "good") & (anno_names != "")
        good_unit_indices = np.flatnonzero(good_mask)
        if len(good_unit_indices) == 0:
            return None, brain_region_to_idx

        recorded_trial_counts = [len(np.asarray(units["is_good_trials"][u])) for u in good_unit_indices]
        n_recorded_trials = int(min(recorded_trial_counts)) if recorded_trial_counts else n_trials
        n_recorded_trials = min(n_recorded_trials, n_trials)
        if n_recorded_trials < 2:
            return None, brain_region_to_idx
        if n_recorded_trials < n_trials:
            trials_df = trials_df.iloc[:n_recorded_trials].copy()
            n_trials = n_recorded_trials

        go_times, response_ends = get_go_times_and_response_ends(nwb, n_trials)
        trial_starts = trials_df["start_time"].to_numpy(dtype=np.float64)
        trial_stops = trials_df["stop_time"].to_numpy(dtype=np.float64)
        sample_onsets = get_sample_onsets(nwb, trial_starts, go_times, trial_stops)
        sample_onsets_rel = sample_onsets - go_times
        stim_starts_rel, stim_ends_rel = get_photostim_relative_intervals(trials_df, go_times)
        choices = get_first_response_choices(nwb, go_times, response_ends)
        tongue_labels, tongue_quantiles = build_tongue_binned_labels(
            nwb, go_times, likelihood_threshold=TONGUE_VISIBLE_THRESHOLD
        )
        neural_trials = build_neural_trials(nwb, good_unit_indices, go_times)

        outcome_strings = trials_df["outcome"].astype(str).to_numpy()
        early_strings = trials_df["early_lick"].astype(str).to_numpy()
        outcome_labels = np.asarray([OUTCOME_TO_INT[x] for x in outcome_strings], dtype=np.int8)
        early_labels = np.asarray([EARLY_TO_INT[x] for x in early_strings], dtype=np.int8)

        input_trials = []
        output_trials = []
        for trial_idx in range(n_trials):
            trial_input = np.empty((2, N_BINS), dtype=np.float32)
            trial_input[0] = (BIN_CENTERS_S - sample_onsets_rel[trial_idx]).astype(np.float32)
            if np.isnan(stim_starts_rel[trial_idx]) or np.isnan(stim_ends_rel[trial_idx]):
                trial_input[1] = 0.0
            else:
                trial_input[1] = (
                    (BIN_CENTERS_S >= stim_starts_rel[trial_idx]) & (BIN_CENTERS_S < stim_ends_rel[trial_idx])
                ).astype(np.float32)
            input_trials.append(trial_input)

            trial_output = np.empty((4, N_BINS), dtype=np.int8)
            trial_output[0] = choices[trial_idx]
            trial_output[1] = outcome_labels[trial_idx]
            trial_output[2] = early_labels[trial_idx]
            trial_output[3] = tongue_labels[trial_idx]
            output_trials.append(trial_output)

        valid_trial_indices = [i for i, trial in enumerate(neural_trials) if np.any(trial)]
        if len(valid_trial_indices) < len(neural_trials):
            neural_trials = [neural_trials[i] for i in valid_trial_indices]
            input_trials = [input_trials[i] for i in valid_trial_indices]
            output_trials = [output_trials[i] for i in valid_trial_indices]
            trials_df = trials_df.iloc[valid_trial_indices].reset_index(drop=True)
            n_trials = len(valid_trial_indices)
        else:
            n_trials = len(neural_trials)

        if n_trials < 2:
            return None, brain_region_to_idx

        unit_regions = anno_names[good_unit_indices]
        region_indices = np.empty(len(unit_regions), dtype=np.int32)
        for i, region in enumerate(unit_regions):
            if region not in brain_region_to_idx:
                brain_region_to_idx[region] = len(brain_region_to_idx)
            region_indices[i] = brain_region_to_idx[region]

        subject_id = str(getattr(nwb.subject, "subject_id", path.parent.name.replace("sub-", "")))
        performance, left_hits, right_hits, n_regular = compute_control_performance(trials_df)

        session_record = {
            "session_path": str(path),
            "subject_id": subject_id,
            "neural": neural_trials,
            "input": input_trials,
            "output": output_trials,
            "brain_region_idx": region_indices,
            "session_info": {
                "session_path": str(path),
                "n_trials": n_trials,
                "n_good_units": int(len(good_unit_indices)),
                "control_performance_non_early_no_auto_free": performance,
                "control_hit_left": left_hits,
                "control_hit_right": right_hits,
                "control_trials_responded": n_regular,
                "tongue_visible_q40": tongue_quantiles[0],
                "tongue_visible_q60": tongue_quantiles[1],
            },
        }
        return session_record, brain_region_to_idx


def build_dataset(data_dir: Path, max_sessions: int | None = None) -> dict:
    session_paths = iter_session_paths(data_dir)
    if max_sessions is not None:
        session_paths = session_paths[:max_sessions]

    subjects: list[str] = []
    subject_to_idx: dict[str, int] = {}
    brain_region_to_idx: dict[str, int] = {}

    neural: list[list[np.ndarray]] = []
    inputs: list[list[np.ndarray]] = []
    outputs: list[list[np.ndarray]] = []
    subject_idx: list[int] = []
    brain_region_idx: list[np.ndarray] = []
    session_info: list[dict] = []
    skipped_sessions: list[dict] = []

    for i, path in enumerate(session_paths, 1):
        print(f"[{i}/{len(session_paths)}] converting {path.name}", flush=True)
        try:
            session_record, brain_region_to_idx = convert_session(path, brain_region_to_idx)
        except Exception as exc:  # pragma: no cover - debug path
            skipped_sessions.append({"session_path": str(path), "reason": repr(exc)})
            print(f"  skipped: {exc}", flush=True)
            continue

        if session_record is None:
            skipped_sessions.append({"session_path": str(path), "reason": "fewer than 2 trials or 0 good units"})
            print("  skipped: fewer than 2 trials or 0 good units", flush=True)
            continue

        subj = session_record["subject_id"]
        if subj not in subject_to_idx:
            subject_to_idx[subj] = len(subjects)
            subjects.append(subj)

        neural.append(session_record["neural"])
        inputs.append(session_record["input"])
        outputs.append(session_record["output"])
        subject_idx.append(subject_to_idx[subj])
        brain_region_idx.append(session_record["brain_region_idx"])
        session_info.append(session_record["session_info"])

    brain_regions = [None] * len(brain_region_to_idx)
    for name, idx in brain_region_to_idx.items():
        brain_regions[idx] = name

    data = {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int32),
        "brain_regions": brain_regions,
        "brain_region_idx": brain_region_idx,
        "input_names": ["time_from_tone_onset_s", "photostimulation_on"],
        "output_names": ["choice", "outcome", "early_lick", "tongue_y_position"],
        "output_values": [
            ["left", "right", "no lick"],
            ["ignore", "miss", "hit"],
            ["no", "yes"],
            ["lt_40th_percentile", "p40_to_p60", "gt_60th_percentile", "not_visible"],
        ],
        "metadata": {
            "task_description": (
                "Delayed auditory licking task; decode choice, outcome, early lick, and discretized tongue y-position "
                "from neural activity aligned to the go cue."
            ),
            "time_bin_size": BIN_SIZE_S * 1000.0,
            "temporal_alignment_event": "go cue onset",
            "off_start": WINDOW_START_S,
            "off_end": WINDOW_END_S,
            "time_bin_centers_s": BIN_CENTERS_S.astype(float).tolist(),
            "n_source_sessions": len(iter_session_paths(data_dir)),
            "n_kept_sessions": len(neural),
            "tongue_visibility_likelihood_threshold": TONGUE_VISIBLE_THRESHOLD,
            "session_info": session_info,
            "skipped_sessions": skipped_sessions,
        },
    }
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()

    data = build_dataset(args.data_dir, max_sessions=args.max_sessions)
    with args.output.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"saved {args.output}", flush=True)
    print(f"kept sessions: {len(data['neural'])}", flush=True)
    print(f"subjects: {len(data['subjects'])}", flush=True)
    print(f"brain regions: {len(data['brain_regions'])}", flush=True)


if __name__ == "__main__":
    main()
