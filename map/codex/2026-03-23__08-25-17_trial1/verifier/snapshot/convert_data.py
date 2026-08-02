import argparse
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BIN_WIDTH_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
EXPECTED_SAMPLE_ONSET_REL_GO = -1.85
@dataclass
class SessionResult:
    session_id: str
    subject_id: str
    source_file: str
    neural: list
    inputs: list
    outputs: list
    brain_region_names: np.ndarray
    stats: dict
    plot_payload: dict | None = None


def decode_str_array(values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind in {"S", "O"}:
        return arr.astype("U")
    if arr.dtype.kind == "U":
        return arr
    return arr.astype(str)


def bin_edges_and_centers():
    edges = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_WIDTH_S * 0.5, BIN_WIDTH_S, dtype=np.float64)
    centers = edges[:-1] + BIN_WIDTH_S / 2.0
    return edges, centers


def get_session_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("sub-*/*.nwb"))


def get_session_identity(file_path: Path) -> tuple[str, str]:
    subject_id = file_path.parent.name.replace("sub-", "")
    session_id = file_path.stem
    return subject_id, session_id


def get_ragged_row_bounds(index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = index.astype(np.int64)
    starts = np.empty_like(ends)
    starts[0] = 0
    starts[1:] = ends[:-1]
    return starts, ends


def event_slices_for_trials(event_times: np.ndarray, trial_starts: np.ndarray, trial_stops: np.ndarray):
    start_idx = np.searchsorted(event_times, trial_starts, side="left")
    stop_idx = np.searchsorted(event_times, trial_stops, side="right")
    return start_idx, stop_idx


def select_trial_indices(
    go_times_all: np.ndarray,
    good_unit_obs_intervals: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    session_obs_start = float(np.min(good_unit_obs_intervals[:, 0]))
    session_obs_stop = float(np.max(good_unit_obs_intervals[:, 1]))
    full_window_mask = (
        (go_times_all + WINDOW_START_S >= session_obs_start)
        & (go_times_all + WINDOW_END_S <= session_obs_stop)
    )
    trial_idx = np.flatnonzero(full_window_mask)
    if len(trial_idx) == 0:
        raise ValueError("No trials contain the full neural extraction window inside the session observation interval")
    return trial_idx, session_obs_start, session_obs_stop


def infer_choice_for_trial(
    trial_start: float,
    trial_stop: float,
    go_time: float,
    instruction: str,
    left_lick_times: np.ndarray,
    right_lick_times: np.ndarray,
) -> tuple[int, str]:
    left_start = np.searchsorted(left_lick_times, trial_start, side="left")
    left_go = np.searchsorted(left_lick_times, go_time, side="left")
    left_stop = np.searchsorted(left_lick_times, trial_stop, side="right")
    right_start = np.searchsorted(right_lick_times, trial_start, side="left")
    right_go = np.searchsorted(right_lick_times, go_time, side="left")
    right_stop = np.searchsorted(right_lick_times, trial_stop, side="right")

    left_post = left_lick_times[left_go:left_stop]
    right_post = right_lick_times[right_go:right_stop]
    if len(left_post) or len(right_post):
        first_left = left_post[0] if len(left_post) else np.inf
        first_right = right_post[0] if len(right_post) else np.inf
        return (0, "post_go_lick") if first_left < first_right else (1, "post_go_lick")

    left_any = left_lick_times[left_start:left_stop]
    right_any = right_lick_times[right_start:right_stop]
    if len(left_any) or len(right_any):
        first_left = left_any[0] if len(left_any) else np.inf
        first_right = right_any[0] if len(right_any) else np.inf
        return (0, "any_lick") if first_left < first_right else (1, "any_lick")

    return (0 if instruction == "left" else 1, "instruction_fallback")


def bin_tongue_y(
    tongue_timestamps: np.ndarray,
    tongue_y: np.ndarray,
    go_times: np.ndarray,
    bin_edges_rel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    abs_edges = go_times[:, None] + bin_edges_rel[None, :]
    start_idx = np.searchsorted(tongue_timestamps, abs_edges[:, :-1], side="left")
    end_idx = np.searchsorted(tongue_timestamps, abs_edges[:, 1:], side="left") - 1

    clipped_end = np.clip(end_idx, 0, len(tongue_y) - 1)
    valid = end_idx >= start_idx
    binned = np.full(end_idx.shape, np.nan, dtype=np.float32)
    binned[valid] = tongue_y[clipped_end[valid]].astype(np.float32)
    return binned, valid


def build_processing_plot(plot_payload: dict, output_path: Path):
    session_id = plot_payload["session_id"]
    centers = plot_payload["bin_centers"]
    example_trial = plot_payload["example_trial"]
    raw_tongue_t = plot_payload["raw_tongue_t"]
    raw_tongue_y = plot_payload["raw_tongue_y"]
    binned_tongue_y = plot_payload["binned_tongue_y"]
    tongue_thresholds = plot_payload["tongue_thresholds"]
    tone_onset = plot_payload["tone_onset_rel_go"]
    stim_window = plot_payload["stim_window_rel_go"]
    input_trial = plot_payload["input_trial"]
    output_trial = plot_payload["output_trial"]
    neural_trial = plot_payload["neural_trial"]
    sample_events = plot_payload["sample_events_rel_go"]
    delay_events = plot_payload["delay_events_rel_go"]
    lick_events_left = plot_payload["left_licks_rel_go"]
    lick_events_right = plot_payload["right_licks_rel_go"]

    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    fig.suptitle(f"Processing Summary: {session_id}")

    ax = axes[0, 0]
    nneurons_to_plot = min(60, neural_trial.shape[0])
    if nneurons_to_plot > 0:
        im = ax.imshow(
            neural_trial[:nneurons_to_plot],
            aspect="auto",
            origin="lower",
            extent=[centers[0], centers[-1], 0, nneurons_to_plot],
            cmap="viridis",
        )
        fig.colorbar(im, ax=ax, label="Hz")
    ax.axvline(0.0, color="white", linestyle="--", linewidth=1)
    ax.set_title(f"Binned Neural Activity (trial {example_trial + 1})")
    ax.set_xlabel("Time from go cue (s)")
    ax.set_ylabel("Neuron")

    ax = axes[0, 1]
    ax.plot(raw_tongue_t, raw_tongue_y, color="0.7", linewidth=1, label="raw tongue y")
    ax.step(centers, binned_tongue_y, where="mid", color="tab:red", linewidth=2, label="binned tongue y")
    for thresh, label in zip(tongue_thresholds, ["40th", "60th"]):
        ax.axhline(thresh, color="tab:blue", linestyle="--", linewidth=1, label=label)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Tongue Y Alignment and Binning")
    ax.set_xlabel("Time from go cue (s)")
    ax.set_ylabel("Tongue y (pixels)")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 0]
    ax.plot(centers, input_trial[0], label="time_from_tone_onset_s")
    ax.plot(centers, input_trial[1], label="photostim_on")
    ax.axvline(tone_onset, color="tab:green", linestyle="--", linewidth=1, label="tone onset")
    if stim_window is not None:
        ax.axvspan(stim_window[0], stim_window[1], color="tab:orange", alpha=0.2, label="photostim")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Constructed Inputs")
    ax.set_xlabel("Time from go cue (s)")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 1]
    names = ["choice", "outcome", "early_lick", "tongue_y_bin"]
    for idx, name in enumerate(names):
        ax.step(centers, output_trial[idx] + idx * 3.5, where="mid", label=name)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Constructed Outputs")
    ax.set_xlabel("Time from go cue (s)")
    ax.legend(loc="best", fontsize=8)

    ax = axes[2, 0]
    ax.hist(plot_payload["all_binned_tongue_y"], bins=60, color="0.5")
    ax.axvline(tongue_thresholds[0], color="tab:blue", linestyle="--", linewidth=1)
    ax.axvline(tongue_thresholds[1], color="tab:blue", linestyle="--", linewidth=1)
    ax.set_title("Session Tongue Y Distribution")
    ax.set_xlabel("Tongue y (pixels)")
    ax.set_ylabel("Count")

    ax = axes[2, 1]
    for sample_t in sample_events:
        ax.axvline(sample_t, color="tab:green", linewidth=1)
    for delay_t in delay_events:
        ax.axvline(delay_t, color="tab:purple", linewidth=1)
    for lick_t in lick_events_left:
        ax.axvline(lick_t, color="tab:blue", linewidth=1)
    for lick_t in lick_events_right:
        ax.axvline(lick_t, color="tab:red", linewidth=1)
    if stim_window is not None:
        ax.axvspan(stim_window[0], stim_window[1], color="tab:orange", alpha=0.25)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(WINDOW_START_S, WINDOW_END_S)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title("Raw Task/Event Timing for Example Trial")
    ax.set_xlabel("Time from go cue (s)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def process_session(file_path: Path, show_processing: bool = False) -> SessionResult | None:
    bin_edges_rel, bin_centers_rel = bin_edges_and_centers()
    bin_width = float(BIN_WIDTH_S)
    trial_plot_index = 0

    with h5py.File(file_path, "r") as h5:
        subject_id, session_id = get_session_identity(file_path)

        units = h5["units"]
        n_recorded_trials = int(units["is_good_trials"].shape[1])
        if n_recorded_trials == 0:
            print(f"Skipping {session_id}: zero recorded trials in units/is_good_trials")
            return None

        trials = h5["intervals"]["trials"]
        n_behavior_trials = int(len(trials["id"]))
        trial_start_all = trials["start_time"][:].astype(np.float64)
        trial_stop_all = trials["stop_time"][:].astype(np.float64)
        trial_instruction_all = np.char.lower(decode_str_array(trials["trial_instruction"][:]))
        early_lick_all = np.char.lower(decode_str_array(trials["early_lick"][:]))
        outcome_all = np.char.lower(decode_str_array(trials["outcome"][:]))
        photostim_onset_all = decode_str_array(trials["photostim_onset"][:])
        photostim_duration_all = decode_str_array(trials["photostim_duration"][:])

        go_times_all = h5["acquisition"]["BehavioralEvents"]["go_start_times"]["timestamps"][:n_behavior_trials].astype(np.float64)
        sample_start_times = h5["acquisition"]["BehavioralEvents"]["sample_start_times"]["timestamps"][:].astype(np.float64)
        delay_start_times = h5["acquisition"]["BehavioralEvents"]["delay_start_times"]["timestamps"][:].astype(np.float64)
        left_lick_times = h5["acquisition"]["BehavioralEvents"]["left_lick_times"]["timestamps"][:].astype(np.float64)
        right_lick_times = h5["acquisition"]["BehavioralEvents"]["right_lick_times"]["timestamps"][:].astype(np.float64)

        tongue_group = h5["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
        tongue_values = tongue_group["data"][:].astype(np.float64)
        tongue_y = tongue_values[:, 1]
        tongue_likelihood = tongue_values[:, 2]
        tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)

        classification = np.char.lower(decode_str_array(units["classification"][:]))
        good_unit_idx = np.flatnonzero(classification == "good")
        if len(good_unit_idx) == 0:
            print(f"Skipping {session_id}: zero good units")
            return None

        brain_region_names = decode_str_array(units["anno_name"][good_unit_idx])
        if np.any(brain_region_names == ""):
            raise ValueError(f"{session_id}: found kept good units with empty anno_name")

        unit_obs_intervals = units["obs_intervals"][good_unit_idx].astype(np.float64)
        selected_trial_idx, session_obs_start, session_obs_stop = select_trial_indices(
            go_times_all=go_times_all,
            good_unit_obs_intervals=unit_obs_intervals,
        )
        trial_start = trial_start_all[selected_trial_idx]
        trial_stop = trial_stop_all[selected_trial_idx]
        trial_instruction = trial_instruction_all[selected_trial_idx]
        early_lick = early_lick_all[selected_trial_idx]
        outcome = outcome_all[selected_trial_idx]
        photostim_onset = photostim_onset_all[selected_trial_idx]
        photostim_duration = photostim_duration_all[selected_trial_idx]
        go_times = go_times_all[selected_trial_idx]

        is_good_trials_raw = units["is_good_trials"][good_unit_idx, :n_recorded_trials].astype(bool)
        uses_direct_is_good_trials = is_good_trials_raw.shape[1] == len(selected_trial_idx)
        if uses_direct_is_good_trials:
            is_good_trials = is_good_trials_raw.copy()
        else:
            is_good_trials = np.ones((len(good_unit_idx), len(selected_trial_idx)), dtype=bool)
        spike_times = units["spike_times"][:].astype(np.float64)
        spike_times_index = units["spike_times_index"][:]
        spike_starts, spike_ends = get_ragged_row_bounds(spike_times_index)

    n_trials = len(trial_start)
    n_units = len(good_unit_idx)
    n_bins = len(bin_centers_rel)

    # Per-trial task timing
    sample_slice_starts, sample_slice_ends = event_slices_for_trials(sample_start_times, trial_start, trial_stop)
    delay_slice_starts, delay_slice_ends = event_slices_for_trials(delay_start_times, trial_start, trial_stop)

    sample_onset_abs = np.empty(n_trials, dtype=np.float64)
    sample_onset_fallbacks = 0
    for trial in range(n_trials):
        events = sample_start_times[sample_slice_starts[trial]:sample_slice_ends[trial]]
        if len(events):
            sample_onset_abs[trial] = events[0]
        else:
            sample_onset_abs[trial] = go_times[trial] + EXPECTED_SAMPLE_ONSET_REL_GO
            sample_onset_fallbacks += 1

    sample_onset_rel_go = sample_onset_abs - go_times

    # Inputs
    input_trials = []
    photostim_trial_count = 0
    for trial in range(n_trials):
        inp = np.zeros((2, n_bins), dtype=np.float32)
        inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)

        if str(photostim_onset[trial]) != "N/A":
            stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
            stim_rel_off = stim_rel_on + float(photostim_duration[trial])
            inp[1] = ((bin_centers_rel >= stim_rel_on) & (bin_centers_rel < stim_rel_off)).astype(np.float32)
            photostim_trial_count += 1
        input_trials.append(inp)

    # Choice / outcome / early lick
    choice_trials = np.empty(n_trials, dtype=np.int8)
    choice_source_counts = {"post_go_lick": 0, "any_lick": 0, "instruction_fallback": 0}
    for trial in range(n_trials):
        choice_val, source = infer_choice_for_trial(
            trial_start=trial_start[trial],
            trial_stop=trial_stop[trial],
            go_time=go_times[trial],
            instruction=str(trial_instruction[trial]),
            left_lick_times=left_lick_times,
            right_lick_times=right_lick_times,
        )
        choice_trials[trial] = choice_val
        choice_source_counts[source] += 1

    outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
    early_map = {"no early": 0, "early": 1}
    outcome_trials = np.array([outcome_map[str(x)] for x in outcome], dtype=np.int8)
    early_trials = np.array([early_map[str(x)] for x in early_lick], dtype=np.int8)

    # Tongue alignment and discretization
    tongue_y_binned, tongue_valid = bin_tongue_y(
        tongue_timestamps=tongue_timestamps,
        tongue_y=tongue_y,
        go_times=go_times,
        bin_edges_rel=bin_edges_rel,
    )
    valid_values = tongue_y_binned[np.isfinite(tongue_y_binned)]
    if len(valid_values) == 0:
        raise ValueError(f"{session_id}: no valid tongue_y values after alignment")
    tongue_p40 = float(np.percentile(valid_values, 40))
    tongue_p60 = float(np.percentile(valid_values, 60))
    tongue_discrete = np.zeros_like(tongue_y_binned, dtype=np.int8)
    tongue_discrete[(tongue_y_binned >= tongue_p40) & (tongue_y_binned <= tongue_p60)] = 1
    tongue_discrete[tongue_y_binned > tongue_p60] = 2

    output_trials = []
    for trial in range(n_trials):
        out = np.empty((4, n_bins), dtype=np.int8)
        out[0] = choice_trials[trial]
        out[1] = outcome_trials[trial]
        out[2] = early_trials[trial]
        out[3] = tongue_discrete[trial]
        output_trials.append(out)

    # Neural binning
    neural_session = np.zeros((n_trials, n_units, n_bins), dtype=np.float16)
    abs_edges = go_times[:, None] + bin_edges_rel[None, :]
    flat_edges = abs_edges.reshape(-1)

    for unit_pos, unit_idx in enumerate(good_unit_idx):
        spikes = spike_times[spike_starts[unit_idx]:spike_ends[unit_idx]]
        edge_indices = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_bins + 1)
        counts = np.diff(edge_indices, axis=1).astype(np.float32)
        fr = counts / bin_width
        if uses_direct_is_good_trials:
            invalid_trials = ~is_good_trials[unit_pos]
        else:
            obs_start = unit_obs_intervals[unit_pos, 0]
            obs_stop = unit_obs_intervals[unit_pos, 1]
            valid_obs = (go_times + WINDOW_START_S >= obs_start) & (go_times + WINDOW_END_S <= obs_stop)
            invalid_trials = ~valid_obs
        if np.any(invalid_trials):
            fr[invalid_trials] = 0.0
        neural_session[:, unit_pos, :] = fr.astype(np.float16)

    nonzero_trial_mask = np.any(neural_session > 0, axis=(1, 2))
    dropped_zero_trials = int(np.sum(~nonzero_trial_mask))
    if dropped_zero_trials:
        neural_session = neural_session[nonzero_trial_mask]
        sample_onset_rel_go = sample_onset_rel_go[nonzero_trial_mask]
        tongue_y_binned = tongue_y_binned[nonzero_trial_mask]
        tongue_discrete = tongue_discrete[nonzero_trial_mask]
        choice_trials = choice_trials[nonzero_trial_mask]
        outcome_trials = outcome_trials[nonzero_trial_mask]
        early_trials = early_trials[nonzero_trial_mask]
        input_trials = [trial for keep, trial in zip(nonzero_trial_mask, input_trials) if keep]
        output_trials = [trial for keep, trial in zip(nonzero_trial_mask, output_trials) if keep]
        trial_start = trial_start[nonzero_trial_mask]
        trial_stop = trial_stop[nonzero_trial_mask]
        go_times = go_times[nonzero_trial_mask]
        sample_slice_starts = sample_slice_starts[nonzero_trial_mask]
        sample_slice_ends = sample_slice_ends[nonzero_trial_mask]
        delay_slice_starts = delay_slice_starts[nonzero_trial_mask]
        delay_slice_ends = delay_slice_ends[nonzero_trial_mask]
        photostim_onset = photostim_onset[nonzero_trial_mask]
        photostim_duration = photostim_duration[nonzero_trial_mask]
        trial_instruction = trial_instruction[nonzero_trial_mask]
        early_lick = early_lick[nonzero_trial_mask]
        outcome = outcome[nonzero_trial_mask]
        is_good_trials = is_good_trials[:, nonzero_trial_mask]
        n_trials = int(neural_session.shape[0])

    neural_trials = [neural_session[trial] for trial in range(n_trials)]

    # Plot payload
    plot_payload = None
    if show_processing:
        trial_plot_index = int(np.argmax(outcome_trials == 2)) if np.any(outcome_trials == 2) else 0
        go = go_times[trial_plot_index]
        t0 = go + WINDOW_START_S
        t1 = go + WINDOW_END_S
        tongue_window = (tongue_timestamps >= t0) & (tongue_timestamps < t1)
        sample_events = sample_start_times[sample_slice_starts[trial_plot_index]:sample_slice_ends[trial_plot_index]] - go
        delay_events = delay_start_times[delay_slice_starts[trial_plot_index]:delay_slice_ends[trial_plot_index]] - go
        left_start = np.searchsorted(left_lick_times, t0, side="left")
        left_stop = np.searchsorted(left_lick_times, t1, side="right")
        right_start = np.searchsorted(right_lick_times, t0, side="left")
        right_stop = np.searchsorted(right_lick_times, t1, side="right")
        stim_window = None
        if str(photostim_onset[trial_plot_index]) != "N/A":
            stim_rel_on = trial_start[trial_plot_index] + float(photostim_onset[trial_plot_index]) - go
            stim_window = (stim_rel_on, stim_rel_on + float(photostim_duration[trial_plot_index]))
        plot_payload = {
            "session_id": session_id,
            "bin_centers": bin_centers_rel,
            "example_trial": trial_plot_index,
            "raw_tongue_t": tongue_timestamps[tongue_window] - go,
            "raw_tongue_y": tongue_y[tongue_window],
            "binned_tongue_y": tongue_y_binned[trial_plot_index],
            "tongue_thresholds": (tongue_p40, tongue_p60),
            "tone_onset_rel_go": sample_onset_rel_go[trial_plot_index],
            "stim_window_rel_go": stim_window,
            "input_trial": input_trials[trial_plot_index],
            "output_trial": output_trials[trial_plot_index],
            "neural_trial": neural_trials[trial_plot_index].astype(np.float32),
            "sample_events_rel_go": sample_events,
            "delay_events_rel_go": delay_events,
            "left_licks_rel_go": left_lick_times[left_start:left_stop] - go,
            "right_licks_rel_go": right_lick_times[right_start:right_stop] - go,
            "all_binned_tongue_y": valid_values,
            "tongue_likelihood_window": tongue_likelihood[tongue_window],
        }

    stats = {
        "session_id": session_id,
        "subject_id": subject_id,
        "selected_trial_start_index": int(selected_trial_idx[0]),
        "selected_trial_stop_index": int(selected_trial_idx[-1]),
        "n_behavior_trials": n_behavior_trials,
        "n_recorded_trials_from_nwb": n_recorded_trials,
        "n_trials": n_trials,
        "n_good_units": n_units,
        "photostim_trials": photostim_trial_count,
        "outcome_counts": {
            "ignore": int(np.sum(outcome_trials == 0)),
            "miss": int(np.sum(outcome_trials == 1)),
            "hit": int(np.sum(outcome_trials == 2)),
        },
        "early_lick_count": int(np.sum(early_trials == 1)),
        "choice_source_counts": choice_source_counts,
        "sample_onset_fallbacks": sample_onset_fallbacks,
        "dropped_all_zero_neural_trials": dropped_zero_trials,
        "session_obs_start_s": session_obs_start,
        "session_obs_stop_s": session_obs_stop,
        "used_direct_is_good_trials": uses_direct_is_good_trials,
        "tongue_thresholds": [tongue_p40, tongue_p60],
        "fraction_invalid_unit_trials": float(np.mean(~is_good_trials)),
        "mean_good_trial_fraction_per_good_unit": float(np.mean(is_good_trials.mean(axis=1))),
    }

    return SessionResult(
        session_id=session_id,
        subject_id=subject_id,
        source_file=str(file_path),
        neural=neural_trials,
        inputs=input_trials,
        outputs=output_trials,
        brain_region_names=brain_region_names,
        stats=stats,
        plot_payload=plot_payload,
    )


def build_dataset(results: list[SessionResult], mode: str) -> dict:
    subjects = sorted({r.subject_id for r in results})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}

    brain_regions = sorted({region for r in results for region in r.brain_region_names.tolist()})
    region_to_idx = {region: idx for idx, region in enumerate(brain_regions)}

    brain_region_idx = [
        np.array([region_to_idx[name] for name in r.brain_region_names], dtype=np.int64)
        for r in results
    ]

    metadata = {
        "task_description": (
            "Auditory delayed-response task. Neural activity is aligned to go cue onset and used to decode "
            "choice, outcome, early lick, and discretized tongue y-position, with time-from-tone and "
            "photostimulation as decoder inputs."
        ),
        "time_bin_size": 50.0,
        "time_bin_size_s": BIN_WIDTH_S,
        "temporal_alignment_event": "Go cue onset",
        "off_start": WINDOW_START_S,
        "off_end": WINDOW_END_S,
        "bin_centers_s": (np.arange(WINDOW_START_S + BIN_WIDTH_S / 2.0, WINDOW_END_S, BIN_WIDTH_S)).astype(np.float32),
        "source_dataset": "Mesoscale Activity Map Dataset (DANDI 000363)",
        "source_format": "NWB",
        "session_ids": [r.session_id for r in results],
        "source_files": [r.source_file for r in results],
        "conversion_mode": mode,
        "neuron_qc_rule": "units/classification == good; sessions with zero good units excluded",
        "trial_selection_rule": (
            "Keep trials whose full [-2.5 s, +1.5 s] go-aligned neural window lies inside the union of good-unit "
            "obs_intervals; combine this with units/is_good_trials when its column count matches the selected trials."
        ),
        "brain_region_label_rule": "Exact units/anno_name CCF annotations for kept units",
        "choice_fallback_rule": (
            "first post-go lick; else first lick anywhere in trial; else trial instruction for no-lick ignore trials"
        ),
        "tongue_alignment_rule": "last tongue_y frame in each 50 ms bin",
        "tongue_discretization_rule": "per-session 40th and 60th percentiles over aligned binned tongue_y",
        "sample_onset_rule": "earliest sample_start_times event within each trial window",
        "n_sessions": len(results),
        "n_subjects": len(subjects),
    }

    return {
        "neural": [r.neural for r in results],
        "input": [r.inputs for r in results],
        "output": [r.outputs for r in results],
        "subjects": subjects,
        "subject_idx": np.array([subject_to_idx[r.subject_id] for r in results], dtype=np.int64),
        "brain_regions": brain_regions,
        "brain_region_idx": brain_region_idx,
        "input_names": ["time_from_tone_onset_s", "photostim_on"],
        "output_names": ["choice", "outcome", "early_lick", "tongue_y_bin"],
        "output_values": [
            ["left", "right"],
            ["ignore", "miss", "hit"],
            ["no", "yes"],
            ["lt_p40", "p40_to_p60", "gt_p60"],
        ],
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert MAP NWB sessions into decoder-ready pickle format.")
    parser.add_argument("outpicklefile", type=str, help="Output pickle file path.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only 2 sessions for testing.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing visualizations for up to 2 sessions as processing_<session_id>.png.",
    )
    args = parser.parse_args()

    run_mode = "sample" if args.sample else "full"
    data_dir = Path("/app/data")
    out_path = Path(args.outpicklefile)

    all_files = get_session_files(data_dir)
    if run_mode == "sample":
        target_files = all_files[:12]
    else:
        target_files = all_files

    print(f"Found {len(all_files)} NWB files")
    print(f"Mode: {run_mode}")
    print(f"Output: {out_path}")

    start_total = time.time()
    results: list[SessionResult] = []
    plotted = 0

    for idx, file_path in enumerate(target_files, start=1):
        start_session = time.time()
        result = process_session(file_path, show_processing=args.show_processing and plotted < 2)
        if result is None:
            continue
        results.append(result)

        if args.show_processing and result.plot_payload is not None and plotted < 2:
            plot_path = Path(f"/app/processing_{result.session_id}.png")
            build_processing_plot(result.plot_payload, plot_path)
            print(f"Saved processing plot: {plot_path.name}")
            plotted += 1

        elapsed = time.time() - start_session
        running = time.time() - start_total
        print(
            f"[{idx}/{len(target_files)}] {result.session_id}: "
            f"{result.stats['n_good_units']} good units, {result.stats['n_trials']} trials, "
            f"{elapsed:.2f}s (running total {running:.2f}s)"
        )
        if run_mode == "sample" and len(results) >= 2:
            break

    dataset = build_dataset(results, mode=run_mode)

    with open(out_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_time = time.time() - start_total
    total_trials = sum(r.stats["n_trials"] for r in results)
    total_units = sum(r.stats["n_good_units"] for r in results)
    print(f"Processed sessions: {len(results)}")
    print(f"Total trials: {total_trials}")
    print(f"Total good units: {total_units}")
    print(f"Total conversion time: {total_time:.2f}s")
    if results:
        print(f"Mean time per kept session: {total_time / len(results):.2f}s")
    print(f"Saved converted dataset to {out_path}")


if __name__ == "__main__":
    main()
