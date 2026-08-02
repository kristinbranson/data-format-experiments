import argparse
import pickle
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


WINDOW_START = -2.5
WINDOW_END = 1.5
BIN_WIDTH = 0.05
BIN_STRIDE = 0.05


def fixed_bin_edges(start_time: float, end_time: float, width: float) -> np.ndarray:
    """Create exact-width bins covering [start_time, end_time)."""
    n_bins = int(round((end_time - start_time) / width))
    return start_time + np.arange(n_bins + 1, dtype=np.float64) * width


BIN_EDGES = fixed_bin_edges(WINDOW_START, WINDOW_END, BIN_WIDTH)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2.0


def decode_bytes_array(arr) -> np.ndarray:
    out = []
    for value in arr:
        if isinstance(value, (bytes, bytearray)):
            out.append(value.decode())
        elif hasattr(value, "decode"):
            out.append(value.decode())
        else:
            out.append(str(value))
    return np.asarray(out)


def parse_optional_float_array(arr) -> np.ndarray:
    out = np.empty(len(arr), dtype=np.float64)
    for i, value in enumerate(arr):
        if isinstance(value, (bytes, bytearray)):
            value = value.decode()
        elif hasattr(value, "decode"):
            value = value.decode()
        else:
            value = str(value)
        if value in {"N/A", "nan", "NaN", ""}:
            out[i] = np.nan
        else:
            out[i] = float(value)
    return out


def ragged_row(data_ds, index_ds, row_idx: int) -> np.ndarray:
    stop = int(index_ds[row_idx])
    start = 0 if row_idx == 0 else int(index_ds[row_idx - 1])
    return np.asarray(data_ds[start:stop], dtype=np.float64)


def first_post_go_choice(left_licks: np.ndarray, right_licks: np.ndarray, go_time: float, stop_time: float, instruction: str):
    left_idx = np.searchsorted(left_licks, go_time, side="left")
    right_idx = np.searchsorted(right_licks, go_time, side="left")

    left_time = left_licks[left_idx] if left_idx < len(left_licks) and left_licks[left_idx] <= stop_time else np.nan
    right_time = right_licks[right_idx] if right_idx < len(right_licks) and right_licks[right_idx] <= stop_time else np.nan

    if np.isfinite(left_time) and np.isfinite(right_time):
        return (0, False) if left_time <= right_time else (1, False)
    if np.isfinite(left_time):
        return 0, False
    if np.isfinite(right_time):
        return 1, False
    return (0 if instruction == "left" else 1), True


def last_sample_before_go(sample_starts: np.ndarray, trial_start: float, go_time: float):
    lo = np.searchsorted(sample_starts, trial_start, side="left")
    hi = np.searchsorted(sample_starts, go_time, side="right")
    if hi > lo:
        return float(sample_starts[hi - 1]), False
    return float(go_time - 1.85), True


def trial_tongue_categories(
    video_timestamps: np.ndarray,
    tongue_y: np.ndarray,
    abs_edges: np.ndarray,
    p40: float,
    p60: float,
) -> np.ndarray:
    starts = np.searchsorted(video_timestamps, abs_edges[:-1], side="left")
    ends = np.searchsorted(video_timestamps, abs_edges[1:], side="left")
    values = np.empty(len(starts), dtype=np.float64)

    for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
        if end_idx > start_idx:
            values[i] = tongue_y[end_idx - 1]
        else:
            fallback_idx = max(0, min(len(tongue_y) - 1, end_idx - 1))
            values[i] = tongue_y[fallback_idx]

    cats = np.zeros(values.shape[0], dtype=np.int8)
    cats[values > p60] = 2
    mid = (values >= p40) & (values <= p60)
    cats[mid] = 1
    return cats


def build_session(
    path: Path,
    trial_limit: int | None = None,
) -> tuple[dict | None, dict]:
    stats = {
        "session_id": path.stem,
        "subject": path.parent.name.replace("sub-", ""),
        "n_units_good": 0,
        "n_trials_raw": 0,
        "n_trials_kept": 0,
        "n_trials_dropped_auto_free": 0,
        "n_trials_dropped_window": 0,
        "n_trials_dropped_all_zero_neural": 0,
        "n_trials_dropped_tone": 0,
        "choice_fallback_trials": 0,
        "tone_fallback_trials": 0,
        "stim_trials_kept": 0,
        "early_trials_kept": 0,
        "ignore_trials_kept": 0,
        "hit_trials_kept": 0,
        "miss_trials_kept": 0,
    }

    with h5py.File(path, "r") as f:
        units = f["units"]
        classification = decode_bytes_array(units["classification"][()])
        anno_name = decode_bytes_array(units["anno_name"][()])
        good_unit_idx = np.flatnonzero(classification == "good")
        stats["n_units_good"] = int(good_unit_idx.size)
        if good_unit_idx.size == 0:
            return None, stats
        ephys_trial_count = int(units["is_good_trials"].shape[1])

        trials = f["intervals/trials"]
        trial_start = np.asarray(trials["start_time"][()], dtype=np.float64)[:ephys_trial_count]
        trial_stop = np.asarray(trials["stop_time"][()], dtype=np.float64)[:ephys_trial_count]
        trial_instruction = decode_bytes_array(trials["trial_instruction"][()])[:ephys_trial_count]
        early_lick = decode_bytes_array(trials["early_lick"][()])[:ephys_trial_count]
        outcome = decode_bytes_array(trials["outcome"][()])[:ephys_trial_count]
        auto_water = np.asarray(trials["auto_water"][()], dtype=np.int8)[:ephys_trial_count]
        free_water = np.asarray(trials["free_water"][()], dtype=np.int8)[:ephys_trial_count]
        stim_onset = parse_optional_float_array(trials["photostim_onset"][()])[:ephys_trial_count]
        stim_duration = parse_optional_float_array(trials["photostim_duration"][()])[:ephys_trial_count]
        stim_power = parse_optional_float_array(trials["photostim_power"][()])[:ephys_trial_count]

        stats["n_trials_raw"] = int(trial_start.size)

        go_times = np.asarray(f["acquisition/BehavioralEvents/go_start_times/timestamps"][()], dtype=np.float64)[:ephys_trial_count]
        sample_starts = np.asarray(f["acquisition/BehavioralEvents/sample_start_times/timestamps"][()], dtype=np.float64)
        left_licks = np.asarray(f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()], dtype=np.float64)
        right_licks = np.asarray(f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()], dtype=np.float64)
        video_timestamps = np.asarray(
            f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps"][()],
            dtype=np.float64,
        )
        tongue_data = np.asarray(
            f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data"][()],
            dtype=np.float64,
        )
        tongue_y = tongue_data[:, 1]
        tongue_p40, tongue_p60 = np.percentile(tongue_y, [40.0, 60.0])

        keep_mask = (auto_water == 0) & (free_water == 0)
        stats["n_trials_dropped_auto_free"] = int(np.sum(~keep_mask))
        keep_idx = np.flatnonzero(keep_mask)
        if trial_limit is not None:
            keep_idx = keep_idx[:trial_limit]

        valid_trials = []
        tone_abs = []
        choice_codes = []
        outcome_codes = []
        early_codes = []
        stim_on_rel = []
        stim_off_rel = []
        session_output_counts = Counter()

        for trial_idx in keep_idx:
            go_abs = float(go_times[trial_idx])
            abs_edges = go_abs + BIN_EDGES

            if abs_edges[0] < video_timestamps[0] or abs_edges[-1] > video_timestamps[-1]:
                stats["n_trials_dropped_window"] += 1
                continue

            tone_time_abs, tone_used_fallback = last_sample_before_go(sample_starts, float(trial_start[trial_idx]), go_abs)
            if tone_used_fallback:
                stats["tone_fallback_trials"] += 1

            choice_code, choice_used_fallback = first_post_go_choice(
                left_licks,
                right_licks,
                go_abs,
                float(trial_stop[trial_idx]),
                trial_instruction[trial_idx],
            )
            if choice_used_fallback:
                stats["choice_fallback_trials"] += 1

            tone_abs.append(tone_time_abs)
            valid_trials.append(trial_idx)
            choice_codes.append(choice_code)

            if outcome[trial_idx] == "ignore":
                outcome_code = 0
                stats["ignore_trials_kept"] += 1
            elif outcome[trial_idx] == "miss":
                outcome_code = 1
                stats["miss_trials_kept"] += 1
            elif outcome[trial_idx] == "hit":
                outcome_code = 2
                stats["hit_trials_kept"] += 1
            else:
                raise ValueError(f"Unexpected outcome {outcome[trial_idx]!r} in {path.name}")
            outcome_codes.append(outcome_code)
            session_output_counts[f"outcome_{outcome[trial_idx]}"] += 1

            early_code = 1 if early_lick[trial_idx] == "early" else 0
            early_codes.append(early_code)
            stats["early_trials_kept"] += early_code

            if np.isfinite(stim_power[trial_idx]) and stim_power[trial_idx] > 0 and np.isfinite(stim_onset[trial_idx]) and np.isfinite(stim_duration[trial_idx]):
                go_rel = go_abs - float(trial_start[trial_idx])
                on_rel = float(stim_onset[trial_idx] - go_rel)
                off_rel = on_rel + float(stim_duration[trial_idx])
                stats["stim_trials_kept"] += 1
            else:
                on_rel = np.nan
                off_rel = np.nan
            stim_on_rel.append(on_rel)
            stim_off_rel.append(off_rel)

        if len(valid_trials) < 2:
            return None, stats

        valid_trials = np.asarray(valid_trials, dtype=np.int64)
        tone_abs = np.asarray(tone_abs, dtype=np.float64)
        choice_codes = np.asarray(choice_codes, dtype=np.int8)
        outcome_codes = np.asarray(outcome_codes, dtype=np.int8)
        early_codes = np.asarray(early_codes, dtype=np.int8)
        stim_on_rel = np.asarray(stim_on_rel, dtype=np.float64)
        stim_off_rel = np.asarray(stim_off_rel, dtype=np.float64)
        go_abs = go_times[valid_trials]

        region_names = anno_name[good_unit_idx].tolist()

        # Build trial-aligned input/output tensors.
        input_trials = []
        output_trials = []
        abs_edge_matrix = go_abs[:, None] + BIN_EDGES[None, :]

        for row_idx, trial_idx in enumerate(valid_trials):
            tone_on_rel = tone_abs[row_idx] - go_abs[row_idx]
            time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)
            if np.isfinite(stim_on_rel[row_idx]):
                stim_on = ((BIN_CENTERS >= stim_on_rel[row_idx]) & (BIN_CENTERS < stim_off_rel[row_idx])).astype(np.float32)
            else:
                stim_on = np.zeros(BIN_CENTERS.shape[0], dtype=np.float32)

            input_trials.append(np.vstack([time_from_tone, stim_on]).astype(np.float32, copy=False))

            tongue_cat = trial_tongue_categories(
                video_timestamps=video_timestamps,
                tongue_y=tongue_y,
                abs_edges=abs_edge_matrix[row_idx],
                p40=tongue_p40,
                p60=tongue_p60,
            )
            output_trials.append(
                np.vstack(
                    [
                        np.full(BIN_CENTERS.shape[0], choice_codes[row_idx], dtype=np.int8),
                        np.full(BIN_CENTERS.shape[0], outcome_codes[row_idx], dtype=np.int8),
                        np.full(BIN_CENTERS.shape[0], early_codes[row_idx], dtype=np.int8),
                        tongue_cat,
                    ]
                )
            )

        # Build firing rates from global spike trains using absolute bin edges.
        spike_times_ds = units["spike_times"]
        spike_times_index_ds = units["spike_times_index"]
        flat_abs_edges = abs_edge_matrix.reshape(-1)
        n_trials = valid_trials.size
        n_bins = BIN_CENTERS.size
        neural_trials = [np.empty((good_unit_idx.size, n_bins), dtype=np.float16) for _ in range(n_trials)]

        for unit_row, unit_idx in enumerate(good_unit_idx):
            spike_times = ragged_row(spike_times_ds, spike_times_index_ds, int(unit_idx))
            edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
            counts = np.diff(edge_idx, axis=1)
            rates = (counts.astype(np.float32) / BIN_WIDTH).astype(np.float16)
            for trial_row in range(n_trials):
                neural_trials[trial_row][unit_row, :] = rates[trial_row]

        nonzero_mask = np.asarray([np.any(trial != 0) for trial in neural_trials], dtype=bool)
        if not np.all(nonzero_mask):
            stats["n_trials_dropped_all_zero_neural"] = int(np.sum(~nonzero_mask))
            neural_trials = [trial for trial, keep in zip(neural_trials, nonzero_mask) if keep]
            input_trials = [trial for trial, keep in zip(input_trials, nonzero_mask) if keep]
            output_trials = [trial for trial, keep in zip(output_trials, nonzero_mask) if keep]
            valid_trials = valid_trials[nonzero_mask]
            n_trials = len(neural_trials)

        if n_trials < 2:
            return None, stats

        stats["n_trials_kept"] = int(n_trials)

        session = {
            "session_id": path.stem,
            "subject": stats["subject"],
            "region_names": region_names,
            "neural": neural_trials,
            "input": input_trials,
            "output": output_trials,
            "n_units_good": int(good_unit_idx.size),
            "n_trials_kept": int(n_trials),
            "trial_indices_source": valid_trials,
            "tongue_percentiles": (float(tongue_p40), float(tongue_p60)),
        }
        return session, stats


def convert_dataset(data_dir: Path, session_limit: int | None = None, trial_limit: int | None = None):
    sessions = []
    session_stats = []

    for path in sorted(data_dir.glob("sub-*/*.nwb")):
        session, stats = build_session(path, trial_limit=trial_limit)
        session_stats.append(stats)
        if session is not None:
            sessions.append(session)
            if session_limit is not None and len(sessions) >= session_limit:
                break

    if not sessions:
        raise RuntimeError("No sessions were converted.")

    subjects = sorted({s["subject"] for s in sessions})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}

    brain_regions = sorted({region for session in sessions for region in session["region_names"]})
    region_to_idx = {region: idx for idx, region in enumerate(brain_regions)}

    data = {
        "neural": [session["neural"] for session in sessions],
        "input": [session["input"] for session in sessions],
        "output": [session["output"] for session in sessions],
        "subjects": subjects,
        "subject_idx": np.asarray([subject_to_idx[session["subject"]] for session in sessions], dtype=np.int16),
        "brain_regions": brain_regions,
        "brain_region_idx": [
            np.asarray([region_to_idx[region] for region in session["region_names"]], dtype=np.int32)
            for session in sessions
        ],
        "input_names": ["time_from_tone_onset_s", "photostim_on"],
        "output_names": ["choice", "outcome", "early_lick", "tongue_y"],
        "output_values": [
            ["left", "right"],
            ["ignore", "miss", "hit"],
            ["no", "yes"],
            ["lt_40pct", "p40_to_p60", "gt_60pct"],
        ],
        "metadata": {
            "task_description": (
                "Auditory delayed-response task; decoder predicts choice, outcome, early lick, and "
                "discretized tongue y-position from go-cue-aligned firing rates plus tone/stimulation inputs."
            ),
            "time_bin_size": 50.0,
            "time_bin_size_sec": BIN_WIDTH,
            "temporal_alignment_event": "go cue onset",
            "off_start": WINDOW_START,
            "off_end": WINDOW_END,
            "bin_centers_sec": BIN_CENTERS.astype(np.float32),
            "binning_note": "Exact 50-ms bins spanning [-2.5, 1.5) s relative to the go cue.",
            "neural_representation": "firing rate in spikes/s from 50-ms bins",
            "trial_filter": (
                "Sessions with zero good units were excluded. Trials with auto_water or free_water were excluded. "
                "Early-lick, ignore, and photostimulation trials were retained because they are decoder targets/inputs."
            ),
            "choice_definition": (
                "Choice is the first post-go lick side within the trial; ignore trials fall back to the instructed side "
                "because the target format requires a binary choice label."
            ),
            "tongue_y_definition": (
                "Side-camera tongue y-coordinate sampled as the last frame in each neural bin, discretized by the "
                "40th and 60th percentiles computed over the full session."
            ),
            "source_data_dir": str(data_dir),
            "n_sessions": len(sessions),
        },
    }

    summary = summarize_conversion(data, sessions, session_stats)
    return data, summary


def summarize_conversion(data: dict, sessions: list[dict], session_stats: list[dict]) -> dict:
    included_ids = {session["session_id"] for session in sessions}
    included_stats = [stats for stats in session_stats if stats["session_id"] in included_ids]
    skipped_stats = [stats for stats in session_stats if stats["session_id"] not in included_ids]

    n_trials_total = int(sum(len(session_trials) for session_trials in data["neural"]))
    n_units_total = int(sum(len(region_idx) for region_idx in data["brain_region_idx"]))
    n_timepoints = int(BIN_CENTERS.size)

    outcome_hist = Counter()
    early_hist = Counter()
    choice_hist = Counter()
    stim_trials = 0
    for session_outputs in data["output"]:
        for trial_output in session_outputs:
            choice_hist[int(trial_output[0, 0])] += 1
            outcome_hist[int(trial_output[1, 0])] += 1
            early_hist[int(trial_output[2, 0])] += 1
    for stats in included_stats:
        stim_trials += int(stats["stim_trials_kept"])

    summary = {
        "sessions_included": len(sessions),
        "sessions_skipped_zero_good_or_too_few_trials": len(skipped_stats),
        "included_session_ids": [session["session_id"] for session in sessions],
        "skipped_session_ids": [stats["session_id"] for stats in skipped_stats],
        "subjects": data["subjects"],
        "n_trials_total": n_trials_total,
        "n_units_total": n_units_total,
        "n_brain_regions": len(data["brain_regions"]),
        "n_timepoints": n_timepoints,
        "time_window_sec": [WINDOW_START, WINDOW_END],
        "choice_hist": dict(choice_hist),
        "outcome_hist": dict(outcome_hist),
        "early_hist": dict(early_hist),
        "stim_trials": stim_trials,
        "choice_fallback_trials": int(sum(stats["choice_fallback_trials"] for stats in included_stats)),
        "tone_fallback_trials": int(sum(stats["tone_fallback_trials"] for stats in included_stats)),
        "dropped_auto_free_trials": int(sum(stats["n_trials_dropped_auto_free"] for stats in session_stats)),
        "dropped_window_trials": int(sum(stats["n_trials_dropped_window"] for stats in session_stats)),
        "dropped_all_zero_neural_trials": int(sum(stats["n_trials_dropped_all_zero_neural"] for stats in session_stats)),
        "good_units_per_session_mean": float(np.mean([stats["n_units_good"] for stats in included_stats])),
        "good_units_per_session_min": int(np.min([stats["n_units_good"] for stats in included_stats])),
        "good_units_per_session_max": int(np.max([stats["n_units_good"] for stats in included_stats])),
        "trials_per_session_mean": float(np.mean([stats["n_trials_kept"] for stats in included_stats])),
        "trials_per_session_min": int(np.min([stats["n_trials_kept"] for stats in included_stats])),
        "trials_per_session_max": int(np.max([stats["n_trials_kept"] for stats in included_stats])),
    }
    return summary


def print_summary(summary: dict):
    print(f"Sessions included: {summary['sessions_included']}")
    print(f"Sessions skipped: {summary['sessions_skipped_zero_good_or_too_few_trials']}")
    print(f"Total trials: {summary['n_trials_total']}")
    print(f"Total good units: {summary['n_units_total']}")
    print(f"Brain regions: {summary['n_brain_regions']}")
    print(f"Timepoints per trial: {summary['n_timepoints']}")
    print(
        "Trials/session mean-min-max: "
        f"{summary['trials_per_session_mean']:.2f} / {summary['trials_per_session_min']} / {summary['trials_per_session_max']}"
    )
    print(
        "Good units/session mean-min-max: "
        f"{summary['good_units_per_session_mean']:.2f} / {summary['good_units_per_session_min']} / {summary['good_units_per_session_max']}"
    )
    print(f"Stim trials kept: {summary['stim_trials']}")
    print(f"Choice fallbacks on ignore trials: {summary['choice_fallback_trials']}")
    print(f"Tone onset fallbacks: {summary['tone_fallback_trials']}")
    print(f"Dropped auto/free-water trials: {summary['dropped_auto_free_trials']}")
    print(f"Dropped short-window trials: {summary['dropped_window_trials']}")
    print(f"Dropped all-zero-neural trials: {summary['dropped_all_zero_neural_trials']}")
    print(f"Choice histogram: {summary['choice_hist']}")
    print(f"Outcome histogram: {summary['outcome_hist']}")
    print(f"Early histogram: {summary['early_hist']}")
    print(f"Skipped session IDs: {summary['skipped_session_ids']}")


def main():
    parser = argparse.ArgumentParser(description="Convert the MAP NWB sessions into decoder-ready pickle format.")
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-limit", type=int, default=None)
    parser.add_argument("--trial-limit-per-session", type=int, default=None)
    args = parser.parse_args()

    data, summary = convert_dataset(
        data_dir=args.data_dir,
        session_limit=args.session_limit,
        trial_limit=args.trial_limit_per_session,
    )

    with args.output.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print_summary(summary)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
