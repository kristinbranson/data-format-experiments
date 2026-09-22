#!/usr/bin/env python3
"""Convert the MAP NWB dataset into the decoder training format."""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import time
from collections import Counter

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_SIZE_S = 0.05
N_BINS = int(round((WINDOW_END_S - WINDOW_START_S) / BIN_SIZE_S))
BIN_EDGES_REL = WINDOW_START_S + np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE_S
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE_S / 2.0
RESPONSE_WINDOW_S = 1.5
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
EXCLUDED_SESSION = "sub-440958_ses-20190216T162508_behavior+ecephys+ogen"

INPUT_NAMES = ["time_from_tone_onset_s", "photostim_on"]
OUTPUT_NAMES = ["choice", "outcome", "early_lick", "tongue_y_position_discrete"]
OUTPUT_VALUES = [
    ["left", "right", "no lick"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["lt_p40", "p40_to_p60", "gt_p60", "not_visible"],
]
CHOICE_TO_INT = {"left": 0, "right": 1, "no lick": 2}
OUTCOME_TO_INT = {"ignore": 0, "miss": 1, "hit": 2}
EARLY_TO_INT = {"no early": 0, "early": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MAP NWB data into decoder-compatible pickle format."
    )
    parser.add_argument("outpicklefile", type=str, help="Output pickle file path.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all valid sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only the first 2 valid sessions.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing visualizations for up to 2 sessions as processing_<session_id>.png.",
    )
    return parser.parse_args()


def session_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    for suffix in (".nwb",):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


def decode_string_array(dataset) -> np.ndarray:
    values = dataset[()]
    out = []
    for item in values:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return np.asarray(out, dtype=object)


def decode_optional_float_array(dataset) -> np.ndarray:
    values = decode_string_array(dataset)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for idx, item in enumerate(values):
        if item != "N/A":
            out[idx] = float(item)
    return out


def get_subject_id(h5file: h5py.File, path: str) -> str:
    try:
        subject = h5file["general"]["subject"]["subject_id"][()]
        if isinstance(subject, bytes):
            return subject.decode("utf-8")
        return str(subject)
    except Exception:
        return os.path.basename(os.path.dirname(path)).replace("sub-", "")


def build_tone_onsets_abs(
    trial_start_times: np.ndarray,
    go_times_abs: np.ndarray,
    sample_start_times_abs: np.ndarray,
) -> np.ndarray:
    """Use the latest sample-start event within each trial before the go cue."""
    start_idx = np.searchsorted(sample_start_times_abs, trial_start_times, side="left")
    end_idx = np.searchsorted(sample_start_times_abs, go_times_abs, side="left")
    tone_onsets_abs = np.full(go_times_abs.shape, np.nan, dtype=np.float64)
    for trial_idx in range(len(go_times_abs)):
        if end_idx[trial_idx] > start_idx[trial_idx]:
            tone_onsets_abs[trial_idx] = sample_start_times_abs[end_idx[trial_idx] - 1]
    return tone_onsets_abs


def reconstruct_choice_labels(
    go_times_abs: np.ndarray,
    left_lick_times_abs: np.ndarray,
    right_lick_times_abs: np.ndarray,
) -> np.ndarray:
    """Use the first lick in the 1.5 s response window after go cue."""
    labels = np.empty(len(go_times_abs), dtype=object)
    for trial_idx, go_time in enumerate(go_times_abs):
        window_end = go_time + RESPONSE_WINDOW_S
        left_idx = np.searchsorted(left_lick_times_abs, go_time, side="left")
        right_idx = np.searchsorted(right_lick_times_abs, go_time, side="left")

        left_time = np.inf
        right_time = np.inf
        if left_idx < len(left_lick_times_abs) and left_lick_times_abs[left_idx] < window_end:
            left_time = left_lick_times_abs[left_idx]
        if right_idx < len(right_lick_times_abs) and right_lick_times_abs[right_idx] < window_end:
            right_time = right_lick_times_abs[right_idx]

        if left_time == np.inf and right_time == np.inf:
            labels[trial_idx] = "no lick"
        elif left_time <= right_time:
            labels[trial_idx] = "left"
        else:
            labels[trial_idx] = "right"
    return labels


def build_photostim_binary(
    stim_on_rel_s: np.ndarray,
    stim_off_rel_s: np.ndarray,
) -> np.ndarray:
    bin_starts = BIN_EDGES_REL[:-1][None, :]
    bin_ends = BIN_EDGES_REL[1:][None, :]
    photostim = np.zeros((len(stim_on_rel_s), N_BINS), dtype=np.float32)
    valid = np.isfinite(stim_on_rel_s) & np.isfinite(stim_off_rel_s)
    if np.any(valid):
        on = stim_on_rel_s[valid][:, None]
        off = stim_off_rel_s[valid][:, None]
        photostim[valid] = ((bin_starts < off) & (bin_ends > on)).astype(np.float32)
    return photostim


def build_common_neural_trial_mask(
    is_good_trials: np.ndarray,
    good_unit_indices: np.ndarray,
    ntrials: int,
) -> np.ndarray:
    """Keep only trials with valid neural coverage for every retained good unit."""
    coverage_mask = np.zeros(ntrials, dtype=bool)
    if len(good_unit_indices) == 0:
        return coverage_mask

    coverage_trials = min(ntrials, is_good_trials.shape[1])
    if coverage_trials > 0:
        coverage_mask[:coverage_trials] = np.all(
            is_good_trials[good_unit_indices, :coverage_trials],
            axis=0,
        )
    return coverage_mask


def get_good_unit_spike_span(
    spike_times_all: np.ndarray,
    spike_times_index: np.ndarray,
    good_unit_indices: np.ndarray,
) -> tuple[float, float]:
    starts = np.empty(len(good_unit_indices), dtype=np.int64)
    stops = spike_times_index[good_unit_indices]
    starts[0] = 0 if good_unit_indices[0] == 0 else spike_times_index[good_unit_indices[0] - 1]
    for out_idx in range(1, len(good_unit_indices)):
        unit_idx = good_unit_indices[out_idx]
        starts[out_idx] = 0 if unit_idx == 0 else spike_times_index[unit_idx - 1]

    nonempty = stops > starts
    if not np.any(nonempty):
        return np.nan, np.nan

    starts = starts[nonempty]
    stops = stops[nonempty]
    global_min = float(np.min(spike_times_all[starts]))
    global_max = float(np.max(spike_times_all[stops - 1]))
    return global_min, global_max


def align_tongue_to_bins(
    tongue_timestamps_abs: np.ndarray,
    tongue_y: np.ndarray,
    tongue_likelihood: np.ndarray,
    go_times_abs: np.ndarray,
    visible_q40: float,
    visible_q60: float,
) -> tuple[np.ndarray, np.ndarray]:
    bin_starts_abs = go_times_abs[:, None] + BIN_EDGES_REL[:-1][None, :]
    bin_ends_abs = go_times_abs[:, None] + BIN_EDGES_REL[1:][None, :]

    last_idx = np.searchsorted(tongue_timestamps_abs, bin_ends_abs, side="left") - 1
    valid = last_idx >= 0

    sampled_y = np.full(bin_starts_abs.shape, np.nan, dtype=np.float32)
    sampled_likelihood = np.full(bin_starts_abs.shape, np.nan, dtype=np.float32)

    if np.any(valid):
        flat_valid = valid.ravel()
        flat_idx = last_idx.ravel()
        chosen = flat_idx[flat_valid]
        last_times = tongue_timestamps_abs[chosen]
        flat_starts = bin_starts_abs.ravel()[flat_valid]
        keep = last_times >= flat_starts
        flat_positions = np.flatnonzero(flat_valid)[keep]
        chosen = chosen[keep]
        sampled_y.ravel()[flat_positions] = tongue_y[chosen]
        sampled_likelihood.ravel()[flat_positions] = tongue_likelihood[chosen]

    classes = np.full(bin_starts_abs.shape, 3, dtype=np.int8)
    visible = (
        np.isfinite(sampled_y)
        & np.isfinite(sampled_likelihood)
        & (sampled_likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
    )
    classes[visible & (sampled_y < visible_q40)] = 0
    classes[visible & (sampled_y >= visible_q40) & (sampled_y <= visible_q60)] = 1
    classes[visible & (sampled_y > visible_q60)] = 2
    return classes, sampled_y


def bin_good_unit_spikes(
    spike_times_all: np.ndarray,
    spike_times_index: np.ndarray,
    good_unit_indices: np.ndarray,
    go_times_abs: np.ndarray,
) -> list[np.ndarray]:
    """Return a list of (n_neurons, n_timepoints) float16 firing-rate matrices."""
    n_units = len(good_unit_indices)
    n_trials = len(go_times_abs)
    neural_stack = np.empty((n_units, n_trials, N_BINS), dtype=np.float16)
    bin_edges_abs = go_times_abs[:, None] + BIN_EDGES_REL[None, :]

    for out_idx, unit_idx in enumerate(good_unit_indices):
        start = 0 if unit_idx == 0 else int(spike_times_index[unit_idx - 1])
        stop = int(spike_times_index[unit_idx])
        spikes = spike_times_all[start:stop]
        insertion_idx = np.searchsorted(spikes, bin_edges_abs, side="left")
        counts = np.diff(insertion_idx, axis=1)
        neural_stack[out_idx] = (counts.astype(np.float32) / BIN_SIZE_S).astype(np.float16)

    return [neural_stack[:, trial_idx, :].copy() for trial_idx in range(n_trials)]


def summarize_distributions(values: list[str]) -> dict[str, int]:
    return dict(Counter(values))


def pick_example_trial(session_result: dict) -> int:
    photostim = session_result["inputs_2d"][:, 1, :].sum(axis=1) > 0
    early = session_result["trial_early_labels"] == "early"
    with_visible = np.any(session_result["tongue_classes"] != 3, axis=1)
    for mask in (photostim & with_visible, early & with_visible, with_visible):
        idx = np.flatnonzero(mask)
        if len(idx):
            return int(idx[0])
    return 0


def plot_processing_steps(session_result: dict, output_dir: str) -> None:
    session_name = session_result["session_name"]
    example_trial = pick_example_trial(session_result)

    fig, axes = plt.subplots(4, 2, figsize=(18, 18))
    axes = axes.ravel()

    # 1. Unit QC counts
    axes[0].bar(
        ["good", "unlabelled", "nan"],
        [
            session_result["classification_counts"].get("good", 0),
            session_result["classification_counts"].get("unlabelled", 0),
            session_result["classification_counts"].get("nan", 0),
        ],
        color=["tab:green", "tab:gray", "tab:orange"],
    )
    axes[0].set_title("Unit Classification Counts")
    axes[0].set_ylabel("Units")

    # 2. Trial filtering summary
    trial_counts = [
        session_result["n_trials_total"],
        session_result["n_trials_excluded_auto_free"],
        session_result["n_trials_excluded_missing_tone"],
        session_result["n_trials_excluded_neural_coverage"],
        session_result["n_trials_excluded_spike_span"],
        session_result["n_trials_excluded_zero_neural"],
        session_result["n_trials_kept"],
    ]
    axes[1].bar(
        [
            "raw",
            "excluded\nauto/free",
            "excluded\nno tone",
            "excluded\nno neural",
            "excluded\nspike span",
            "excluded\nzero FR",
            "kept",
        ],
        trial_counts,
        color=["tab:blue", "tab:red", "tab:orange", "tab:purple", "tab:gray", "tab:brown", "tab:green"],
    )
    axes[1].set_title("Trial Filtering")
    axes[1].set_ylabel("Trials")

    # 3. Tongue likelihood histogram
    axes[2].hist(session_result["tongue_likelihood_raw"], bins=100, color="tab:purple")
    axes[2].axvline(TONGUE_LIKELIHOOD_THRESHOLD, color="black", linestyle="--", linewidth=1.5)
    axes[2].set_title("Tongue Likelihood Distribution")
    axes[2].set_xlabel("Likelihood")
    axes[2].set_ylabel("Frames")

    # 4. Visible tongue y histogram with percentile boundaries
    visible_y = session_result["tongue_y_visible_raw"]
    axes[3].hist(visible_y, bins=100, color="tab:brown")
    axes[3].axvline(session_result["tongue_q40"], color="black", linestyle="--", linewidth=1.5)
    axes[3].axvline(session_result["tongue_q60"], color="black", linestyle="--", linewidth=1.5)
    axes[3].set_title("Visible Tongue Y Distribution")
    axes[3].set_xlabel("Tongue y")
    axes[3].set_ylabel("Frames")

    # 5. Raw aligned tongue y and binned tongue classes for one trial
    trial_mask = (
        (session_result["tongue_timestamps_abs"] >= session_result["go_times_abs"][example_trial] + WINDOW_START_S)
        & (session_result["tongue_timestamps_abs"] < session_result["go_times_abs"][example_trial] + WINDOW_END_S)
    )
    rel_t = session_result["tongue_timestamps_abs"][trial_mask] - session_result["go_times_abs"][example_trial]
    raw_y = session_result["tongue_y_raw"][trial_mask]
    raw_like = session_result["tongue_likelihood_raw"][trial_mask]
    visible = raw_like >= TONGUE_LIKELIHOOD_THRESHOLD
    axes[4].scatter(rel_t[visible], raw_y[visible], s=5, alpha=0.4, label="visible")
    axes[4].scatter(rel_t[~visible], raw_y[~visible], s=5, alpha=0.2, label="low conf")
    axes[4].axvline(session_result["tone_rel_s"][example_trial], color="tab:orange", linestyle="--", label="tone")
    if np.isfinite(session_result["stim_on_rel_s"][example_trial]):
        axes[4].axvspan(
            session_result["stim_on_rel_s"][example_trial],
            session_result["stim_off_rel_s"][example_trial],
            color="tab:red",
            alpha=0.2,
            label="photostim",
        )
    axes[4].set_title(f"Raw Tongue Track Around Go: Trial {example_trial}")
    axes[4].set_xlabel("Time from go cue (s)")
    axes[4].set_ylabel("Tongue y")
    axes[4].legend(loc="best", fontsize=8)

    # 6. Binned neural/input/output example
    sample_neurons = min(40, session_result["neural_trials"][example_trial].shape[0])
    neural_img = session_result["neural_trials"][example_trial][:sample_neurons]
    axes[5].imshow(neural_img, aspect="auto", interpolation="nearest", origin="lower")
    axes[5].set_title(f"Neural Firing Rates: Trial {example_trial}")
    axes[5].set_xlabel("50 ms bin")
    axes[5].set_ylabel("Neuron")

    # 7. Inputs
    axes[6].plot(BIN_CENTERS_REL, session_result["inputs_2d"][example_trial, 0], label="time_from_tone")
    axes[6].step(BIN_CENTERS_REL, session_result["inputs_2d"][example_trial, 1], where="mid", label="photostim")
    axes[6].set_title(f"Aligned Inputs: Trial {example_trial}")
    axes[6].set_xlabel("Time from go cue (s)")
    axes[6].legend(loc="best", fontsize=8)

    # 8. Outputs
    axes[7].step(BIN_CENTERS_REL, session_result["outputs_2d"][example_trial, 0], where="mid", label="choice")
    axes[7].step(BIN_CENTERS_REL, session_result["outputs_2d"][example_trial, 1], where="mid", label="outcome")
    axes[7].step(BIN_CENTERS_REL, session_result["outputs_2d"][example_trial, 2], where="mid", label="early")
    axes[7].step(BIN_CENTERS_REL, session_result["outputs_2d"][example_trial, 3], where="mid", label="tongue_y")
    axes[7].set_title(f"Converted Outputs: Trial {example_trial}")
    axes[7].set_xlabel("Time from go cue (s)")
    axes[7].legend(loc="best", fontsize=8)

    fig.suptitle(
        f"{session_name}\n"
        f"good units={session_result['n_good_units']}, kept trials={session_result['n_trials_kept']}, "
        f"choice={session_result['choice_counts']}, outcome={session_result['outcome_counts']}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(os.path.join(output_dir, f"processing_{session_name}.png"), dpi=150)
    plt.close(fig)


def process_session(path: str) -> dict | None:
    session_t0 = time.time()
    session_name = session_name_from_path(path)
    print(f"[session] {session_name}: loading", flush=True)

    with h5py.File(path, "r") as h5file:
        subject_id = get_subject_id(h5file, path)

        classification = decode_string_array(h5file["units"]["classification"])
        classification_counts = Counter(classification.tolist())
        good_unit_indices = np.flatnonzero(classification == "good")
        if len(good_unit_indices) == 0:
            print(f"[session] {session_name}: skipping (0 classifier-good units)", flush=True)
            return None
        spike_times_all = np.asarray(h5file["units"]["spike_times"][()], dtype=np.float64)
        spike_times_index = np.asarray(h5file["units"]["spike_times_index"][()], dtype=np.int64)
        spike_span_min, spike_span_max = get_good_unit_spike_span(
            spike_times_all=spike_times_all,
            spike_times_index=spike_times_index,
            good_unit_indices=good_unit_indices,
        )

        trial_start_times = np.asarray(h5file["intervals"]["trials"]["start_time"][()], dtype=np.float64)
        trial_stop_times = np.asarray(h5file["intervals"]["trials"]["stop_time"][()], dtype=np.float64)
        auto_water = np.asarray(h5file["intervals"]["trials"]["auto_water"][()], dtype=bool)
        free_water = np.asarray(h5file["intervals"]["trials"]["free_water"][()], dtype=bool)
        early_lick = decode_string_array(h5file["intervals"]["trials"]["early_lick"])
        outcome = decode_string_array(h5file["intervals"]["trials"]["outcome"])
        trial_instruction = decode_string_array(h5file["intervals"]["trials"]["trial_instruction"])
        photostim_onset_trial_rel = decode_optional_float_array(h5file["intervals"]["trials"]["photostim_onset"])
        photostim_duration = decode_optional_float_array(h5file["intervals"]["trials"]["photostim_duration"])

        go_times_abs = np.asarray(
            h5file["acquisition"]["BehavioralEvents"]["go_start_times"]["timestamps"][()],
            dtype=np.float64,
        )
        sample_start_times_abs = np.asarray(
            h5file["acquisition"]["BehavioralEvents"]["sample_start_times"]["timestamps"][()],
            dtype=np.float64,
        )
        left_lick_times_abs = np.asarray(
            h5file["acquisition"]["BehavioralEvents"]["left_lick_times"]["timestamps"][()],
            dtype=np.float64,
        )
        right_lick_times_abs = np.asarray(
            h5file["acquisition"]["BehavioralEvents"]["right_lick_times"]["timestamps"][()],
            dtype=np.float64,
        )

        ntrials = min(
            len(trial_start_times),
            len(trial_stop_times),
            len(auto_water),
            len(free_water),
            len(early_lick),
            len(outcome),
            len(trial_instruction),
            len(photostim_onset_trial_rel),
            len(photostim_duration),
            len(go_times_abs),
        )
        trial_start_times = trial_start_times[:ntrials]
        trial_stop_times = trial_stop_times[:ntrials]
        auto_water = auto_water[:ntrials]
        free_water = free_water[:ntrials]
        early_lick = early_lick[:ntrials]
        outcome = outcome[:ntrials]
        trial_instruction = trial_instruction[:ntrials]
        photostim_onset_trial_rel = photostim_onset_trial_rel[:ntrials]
        photostim_duration = photostim_duration[:ntrials]
        go_times_abs = go_times_abs[:ntrials]

        tone_onsets_abs = build_tone_onsets_abs(trial_start_times, go_times_abs, sample_start_times_abs)
        is_good_trials = np.asarray(h5file["units"]["is_good_trials"][()], dtype=bool)
        neural_coverage_mask = build_common_neural_trial_mask(
            is_good_trials=is_good_trials,
            good_unit_indices=good_unit_indices,
            ntrials=ntrials,
        )
        spike_support_mask = (
            np.isfinite(spike_span_min)
            & np.isfinite(spike_span_max)
            & ((go_times_abs + WINDOW_END_S) > spike_span_min)
            & ((go_times_abs + WINDOW_START_S) < spike_span_max)
        )
        keep_trials = (
            (~auto_water)
            & (~free_water)
            & np.isfinite(tone_onsets_abs)
            & neural_coverage_mask
            & spike_support_mask
        )
        kept_trial_indices = np.flatnonzero(keep_trials)

        if len(kept_trial_indices) < 2:
            print(f"[session] {session_name}: skipping (<2 kept trials after filtering)", flush=True)
            return None

        go_keep = go_times_abs[kept_trial_indices]
        tone_rel_s = tone_onsets_abs[kept_trial_indices] - go_keep
        trial_start_keep = trial_start_times[kept_trial_indices]
        trial_instruction_keep = trial_instruction[kept_trial_indices]
        outcome_keep = outcome[kept_trial_indices]
        early_lick_keep = early_lick[kept_trial_indices]
        photostim_onset_keep = photostim_onset_trial_rel[kept_trial_indices]
        photostim_duration_keep = photostim_duration[kept_trial_indices]

        choice_labels = reconstruct_choice_labels(go_keep, left_lick_times_abs, right_lick_times_abs)
        trial_choice_int = np.asarray([CHOICE_TO_INT[x] for x in choice_labels], dtype=np.int8)
        trial_outcome_int = np.asarray([OUTCOME_TO_INT[x] for x in outcome_keep], dtype=np.int8)
        trial_early_int = np.asarray([EARLY_TO_INT[x] for x in early_lick_keep], dtype=np.int8)

        stim_on_rel_s = np.full(len(kept_trial_indices), np.nan, dtype=np.float64)
        valid_stim = np.isfinite(photostim_onset_keep) & np.isfinite(photostim_duration_keep)
        stim_on_rel_s[valid_stim] = (
            trial_start_keep[valid_stim] + photostim_onset_keep[valid_stim] - go_keep[valid_stim]
        )
        stim_off_rel_s = stim_on_rel_s + photostim_duration_keep

        time_from_tone = BIN_CENTERS_REL[None, :] - tone_rel_s[:, None]
        photostim_binary = build_photostim_binary(stim_on_rel_s, stim_off_rel_s)
        inputs_2d = np.stack([time_from_tone, photostim_binary], axis=1).astype(np.float32)

        tongue_data = np.asarray(
            h5file["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]["data"][()],
            dtype=np.float32,
        )
        tongue_timestamps_abs = np.asarray(
            h5file["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]["timestamps"][()],
            dtype=np.float64,
        )
        tongue_y_raw = tongue_data[:, 1]
        tongue_likelihood_raw = tongue_data[:, 2]
        visible_global = (
            np.isfinite(tongue_y_raw)
            & np.isfinite(tongue_likelihood_raw)
            & (tongue_likelihood_raw >= TONGUE_LIKELIHOOD_THRESHOLD)
        )
        if np.count_nonzero(visible_global) == 0:
            tongue_q40 = 0.0
            tongue_q60 = 0.0
        else:
            tongue_q40, tongue_q60 = np.quantile(tongue_y_raw[visible_global], [0.4, 0.6])
        tongue_classes, _ = align_tongue_to_bins(
            tongue_timestamps_abs=tongue_timestamps_abs,
            tongue_y=tongue_y_raw,
            tongue_likelihood=tongue_likelihood_raw,
            go_times_abs=go_keep,
            visible_q40=float(tongue_q40),
            visible_q60=float(tongue_q60),
        )

        outputs_2d = np.empty((len(kept_trial_indices), len(OUTPUT_NAMES), N_BINS), dtype=np.int8)
        outputs_2d[:, 0, :] = trial_choice_int[:, None]
        outputs_2d[:, 1, :] = trial_outcome_int[:, None]
        outputs_2d[:, 2, :] = trial_early_int[:, None]
        outputs_2d[:, 3, :] = tongue_classes

        neural_trials = bin_good_unit_spikes(
            spike_times_all=spike_times_all,
            spike_times_index=spike_times_index,
            good_unit_indices=good_unit_indices,
            go_times_abs=go_keep,
        )
        nonzero_neural_mask = np.asarray([np.any(trial) for trial in neural_trials], dtype=bool)
        n_zero_neural = int(np.count_nonzero(~nonzero_neural_mask))
        if n_zero_neural:
            neural_trials = [trial for trial, keep in zip(neural_trials, nonzero_neural_mask) if keep]
            inputs_2d = inputs_2d[nonzero_neural_mask]
            outputs_2d = outputs_2d[nonzero_neural_mask]
            go_keep = go_keep[nonzero_neural_mask]
            tone_rel_s = tone_rel_s[nonzero_neural_mask]
            stim_on_rel_s = stim_on_rel_s[nonzero_neural_mask]
            stim_off_rel_s = stim_off_rel_s[nonzero_neural_mask]
            choice_labels = choice_labels[nonzero_neural_mask]
            outcome_keep = outcome_keep[nonzero_neural_mask]
            early_lick_keep = early_lick_keep[nonzero_neural_mask]
            trial_choice_int = trial_choice_int[nonzero_neural_mask]
            trial_outcome_int = trial_outcome_int[nonzero_neural_mask]
            trial_early_int = trial_early_int[nonzero_neural_mask]
            tongue_classes = tongue_classes[nonzero_neural_mask]
        if len(neural_trials) < 2:
            print(f"[session] {session_name}: skipping (<2 kept trials after zero-neural filtering)", flush=True)
            return None

        anno_name = decode_string_array(h5file["units"]["anno_name"])
        region_labels = anno_name[good_unit_indices].astype(object)
        region_labels = np.asarray(
            [label if str(label).strip() else "Unknown" for label in region_labels],
            dtype=object,
        )

    session_elapsed = time.time() - session_t0
    print(
        f"[session] {session_name}: good_units={len(good_unit_indices)}, "
        f"kept_trials={len(neural_trials)}, time={session_elapsed:.2f}s",
        flush=True,
    )

    return {
        "session_name": session_name,
        "subject_id": subject_id,
        "neural_trials": neural_trials,
        "inputs_2d": inputs_2d,
        "outputs_2d": outputs_2d,
        "region_labels": region_labels,
        "classification_counts": classification_counts,
        "n_good_units": int(len(good_unit_indices)),
        "n_trials_total": int(ntrials),
        "n_trials_excluded_auto_free": int(np.count_nonzero(auto_water | free_water)),
        "n_trials_excluded_neural_coverage": int(np.count_nonzero(~neural_coverage_mask)),
        "n_trials_excluded_missing_tone": int(np.count_nonzero(~np.isfinite(tone_onsets_abs))),
        "n_trials_excluded_spike_span": int(np.count_nonzero(~spike_support_mask)),
        "n_trials_excluded_zero_neural": n_zero_neural,
        "n_trials_kept": int(len(neural_trials)),
        "trial_choice_labels": choice_labels,
        "trial_outcome_labels": outcome_keep,
        "trial_early_labels": early_lick_keep,
        "choice_counts": summarize_distributions(choice_labels.tolist()),
        "outcome_counts": summarize_distributions(outcome_keep.tolist()),
        "early_counts": summarize_distributions(early_lick_keep.tolist()),
        "go_times_abs": go_keep,
        "tone_rel_s": tone_rel_s,
        "stim_on_rel_s": stim_on_rel_s,
        "stim_off_rel_s": stim_off_rel_s,
        "tongue_q40": float(tongue_q40),
        "tongue_q60": float(tongue_q60),
        "tongue_timestamps_abs": tongue_timestamps_abs,
        "tongue_y_raw": tongue_y_raw,
        "tongue_likelihood_raw": tongue_likelihood_raw,
        "tongue_y_visible_raw": tongue_y_raw[visible_global],
        "tongue_classes": tongue_classes,
    }


def build_data_dict(processed_sessions: list[dict]) -> dict:
    subjects: list[str] = []
    subject_to_idx: dict[str, int] = {}
    all_regions = sorted({str(label) for session in processed_sessions for label in session["region_labels"]})
    region_to_idx = {region: idx for idx, region in enumerate(all_regions)}

    neural = []
    input_data = []
    output_data = []
    subject_idx = np.empty(len(processed_sessions), dtype=np.int32)
    brain_region_idx = []
    session_names = []

    for session_idx, session in enumerate(processed_sessions):
        session_names.append(session["session_name"])
        subject = session["subject_id"]
        if subject not in subject_to_idx:
            subject_to_idx[subject] = len(subjects)
            subjects.append(subject)
        subject_idx[session_idx] = subject_to_idx[subject]

        neural.append(session["neural_trials"])
        input_data.append([trial.astype(np.float32, copy=False) for trial in session["inputs_2d"]])
        output_data.append([trial.astype(np.int8, copy=False) for trial in session["outputs_2d"]])
        brain_region_idx.append(
            np.asarray([region_to_idx[str(label)] for label in session["region_labels"]], dtype=np.int32)
        )

    metadata = {
        "task_description": (
            "Auditory delayed-response task. Decoder predicts trial choice, outcome, early lick, "
            "and time-varying discretized tongue y-position from go-cue-aligned neural activity."
        ),
        "time_bin_size": 50.0,
        "temporal_alignment_event": "Go cue onset",
        "off_start": WINDOW_START_S,
        "off_end": WINDOW_END_S,
        "input_description": {
            "time_from_tone_onset_s": "Continuous time relative to per-trial tone/sample onset.",
            "photostim_on": "Binary indicator of photostimulation overlap in each 50 ms bin.",
        },
        "output_description": {
            "choice": "First lick direction in the 1.5 s response window after go cue; no lick if absent.",
            "outcome": "Direct mapping of NWB trial outcome string.",
            "early_lick": "Direct mapping of NWB early-lick label.",
            "tongue_y_position_discrete": (
                "Session-wise discretization of side-view tongue y using visible-frame percentiles; "
                "class 3 indicates not visible."
            ),
        },
        "neuron_filtering": {
            "session_inclusion": "Keep sessions with at least one units/classification == 'good' unit.",
            "unit_inclusion": "Keep units where units/classification == 'good'.",
            "excluded_session": EXCLUDED_SESSION,
        },
        "trial_filtering": {
            "excluded_trials": [
                "auto_water",
                "free_water",
                "missing_tone_onset",
                "missing_common_neural_coverage",
                "outside_good_unit_spike_span",
                "all_zero_neural_window",
            ],
            "retained_trials_even_if": ["early_lick", "ignore", "photostim"],
        },
        "tongue_visibility_threshold": TONGUE_LIKELIHOOD_THRESHOLD,
        "session_names": session_names,
        "source_dataset": "Mesoscale Activity Map Dataset (DANDI 000363)",
    }

    return {
        "neural": neural,
        "input": input_data,
        "output": output_data,
        "subjects": subjects,
        "subject_idx": subject_idx,
        "brain_regions": all_regions,
        "brain_region_idx": brain_region_idx,
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "output_values": OUTPUT_VALUES,
        "metadata": metadata,
    }


def list_valid_session_paths() -> list[str]:
    all_paths = sorted(glob.glob("/app/data/sub-*/*.nwb"))
    valid_paths = []
    for path in all_paths:
        session_name = session_name_from_path(path)
        if session_name == EXCLUDED_SESSION:
            continue
        valid_paths.append(path)
    return valid_paths


def main() -> None:
    args = parse_args()
    out_path = args.outpicklefile
    sample_mode = args.sample
    valid_paths = list_valid_session_paths()
    if sample_mode:
        valid_paths = valid_paths[:2]

    overall_t0 = time.time()
    processed_sessions: list[dict] = []
    session_times = []

    print(
        f"[convert] mode={'sample' if sample_mode else 'full'} "
        f"sessions_to_scan={len(valid_paths)} out={out_path}",
        flush=True,
    )

    for session_idx, path in enumerate(valid_paths, start=1):
        t0 = time.time()
        result = process_session(path)
        elapsed = time.time() - t0
        session_times.append(elapsed)
        if result is not None:
            processed_sessions.append(result)
        mean_so_far = float(np.mean(session_times))
        remaining = len(valid_paths) - session_idx
        eta = remaining * mean_so_far
        print(
            f"[progress] {session_idx}/{len(valid_paths)} scanned; kept={len(processed_sessions)}; "
            f"mean_per_session={mean_so_far:.2f}s; eta~{eta/60.0:.1f} min",
            flush=True,
        )

    if not processed_sessions:
        raise RuntimeError("No valid sessions were processed.")

    if args.show_processing:
        for session in processed_sessions[:2]:
            plot_processing_steps(session, output_dir=os.path.dirname(out_path) or ".")

    data = build_data_dict(processed_sessions)
    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - overall_t0
    total_trials = sum(len(session["neural_trials"]) for session in processed_sessions)
    total_units = int(sum(session["n_good_units"] for session in processed_sessions))
    print(
        f"[done] wrote {out_path} with {len(processed_sessions)} sessions, "
        f"{total_trials} trials, {total_units} units in {elapsed/60.0:.2f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
