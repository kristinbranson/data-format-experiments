#!/usr/bin/env python3
"""Convert the MAP NWB archive into the decoder training format."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import OrderedDict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path("/app/data")
BIN_SIZE_S = 0.05
OFF_START_S = -2.5
OFF_END_S = 1.5
VISIBILITY_THRESHOLD = 0.9
N_BINS = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))
BIN_EDGES_REL = OFF_START_S + np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE_S
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE_S / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MAP NWB data into decoder-compatible pickle format."
    )
    parser.add_argument(
        "outpicklefile",
        type=Path,
        help="Output pickle file path.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--full",
        action="store_true",
        help="Process all valid sessions (default).",
    )
    mode.add_argument(
        "--sample",
        action="store_true",
        help="Process only the first two valid sessions.",
    )
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing visualizations for up to two sessions.",
    )
    return parser.parse_args()


def now() -> float:
    return time.perf_counter()


def as_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def read_str_array(dataset) -> np.ndarray:
    return np.array([as_str(x) for x in dataset[:]], dtype=object)


def normalize_region_name(name: str) -> str:
    name = str(name).strip()
    if not name or name.lower() == "nan":
        return ""
    return " ".join(name.split())


def normalize_fallback_region(name: str) -> str:
    name = normalize_region_name(name)
    lower = name.lower()
    if lower.startswith("left "):
        return name[5:]
    if lower.startswith("right "):
        return name[6:]
    return name


def load_insertion_regions(nwb_file: h5py.File) -> list[str]:
    raw = read_str_array(nwb_file["general/extracellular_ephys/electrodes/location"])
    regions: list[str] = []
    for item in raw:
        region = ""
        try:
            parsed = json.loads(item)
            region = normalize_fallback_region(parsed.get("brain_regions", ""))
        except json.JSONDecodeError:
            region = normalize_fallback_region(item)
        regions.append(region or "unknown")
    return regions


def unit_start_indices(index_array: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], index_array[:-1])).astype(np.int64, copy=False)


def get_last_events_within(
    event_times: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the last event in [lower_bounds, upper_bounds] for each interval."""
    out = np.full(lower_bounds.shape, np.nan, dtype=np.float64)
    if event_times.size == 0:
        return out, np.zeros(lower_bounds.shape, dtype=bool)
    idx = np.searchsorted(event_times, upper_bounds, side="right") - 1
    valid = idx >= 0
    if np.any(valid):
        candidate = np.clip(idx, 0, event_times.size - 1)
        candidate_times = event_times[candidate]
        valid &= candidate_times >= lower_bounds
        out[valid] = candidate_times[valid]
    return out, valid


def get_first_events_within(
    event_times: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the first event in [lower_bounds, upper_bounds] for each interval."""
    out = np.full(lower_bounds.shape, np.nan, dtype=np.float64)
    if event_times.size == 0:
        return out, np.zeros(lower_bounds.shape, dtype=bool)
    idx = np.searchsorted(event_times, lower_bounds, side="left")
    valid = idx < event_times.size
    if np.any(valid):
        candidate = np.clip(idx, 0, event_times.size - 1)
        candidate_times = event_times[candidate]
        valid &= candidate_times <= upper_bounds
        out[valid] = candidate_times[valid]
    return out, valid


def choose_session_paths(all_paths: list[Path], sample_only: bool) -> list[Path]:
    if not sample_only:
        return all_paths
    return all_paths[:2]


def get_common_observation_window(
    obs_intervals: np.ndarray,
    obs_intervals_index: np.ndarray,
    good_unit_idx: np.ndarray,
) -> tuple[float, float]:
    """Return the shared observed time support across retained units."""
    obs_starts = unit_start_indices(obs_intervals_index)
    good_starts = np.empty(good_unit_idx.shape[0], dtype=np.float64)
    good_ends = np.empty(good_unit_idx.shape[0], dtype=np.float64)

    for out_i, unit_i in enumerate(good_unit_idx):
        start = obs_starts[unit_i]
        stop = obs_intervals_index[unit_i]
        unit_obs = np.asarray(obs_intervals[start:stop], dtype=np.float64)
        if unit_obs.size == 0:
            good_starts[out_i] = np.inf
            good_ends[out_i] = -np.inf
            continue
        good_starts[out_i] = float(np.min(unit_obs[:, 0]))
        good_ends[out_i] = float(np.max(unit_obs[:, 1]))

    return float(np.max(good_starts)), float(np.min(good_ends))


def derive_choice_labels(
    left_lick_times: np.ndarray,
    right_lick_times: np.ndarray,
    go_start: np.ndarray,
    go_stop: np.ndarray,
) -> np.ndarray:
    left_first, left_valid = get_first_events_within(left_lick_times, go_start, go_stop)
    right_first, right_valid = get_first_events_within(right_lick_times, go_start, go_stop)

    choice = np.full(go_start.shape, 2, dtype=np.int64)  # 2 = no lick
    left_only = left_valid & ~right_valid
    right_only = right_valid & ~left_valid
    both = left_valid & right_valid
    choice[left_only] = 0
    choice[right_only] = 1
    choice[both] = (right_first[both] < left_first[both]).astype(np.int64)
    return choice


def make_photostim_matrix(
    stim_start_rel: np.ndarray,
    stim_stop_rel: np.ndarray,
) -> np.ndarray:
    mat = np.zeros((stim_start_rel.shape[0], N_BINS), dtype=np.float32)
    valid = np.isfinite(stim_start_rel) & np.isfinite(stim_stop_rel)
    if np.any(valid):
        start = stim_start_rel[valid][:, None]
        stop = stim_stop_rel[valid][:, None]
        active = (BIN_CENTERS_REL[None, :] >= start) & (BIN_CENTERS_REL[None, :] < stop)
        mat[valid] = active.astype(np.float32)
    return mat


def make_tone_time_matrix(sample_start_rel: np.ndarray) -> np.ndarray:
    return (BIN_CENTERS_REL[None, :] - sample_start_rel[:, None]).astype(np.float32)


def bin_spikes_for_session(
    spike_times_dataset,
    spike_index: np.ndarray,
    good_unit_idx: np.ndarray,
    go_times: np.ndarray,
) -> np.ndarray:
    abs_edges = go_times[:, None] + BIN_EDGES_REL[None, :]
    unit_starts = unit_start_indices(spike_index)
    fr = np.empty((go_times.shape[0], good_unit_idx.shape[0], N_BINS), dtype=np.float16)
    for out_i, unit_i in enumerate(good_unit_idx):
        spikes = np.asarray(
            spike_times_dataset[unit_starts[unit_i] : spike_index[unit_i]],
            dtype=np.float64,
        )
        if spikes.size == 0:
            fr[:, out_i, :] = 0.0
            continue
        counts = np.searchsorted(spikes, abs_edges.ravel(), side="left").reshape(
            go_times.shape[0], N_BINS + 1
        )
        fr[:, out_i, :] = (np.diff(counts, axis=1) / BIN_SIZE_S).astype(np.float16)
    return fr


def align_tongue_bins(
    frame_times: np.ndarray,
    tongue_y: np.ndarray,
    tongue_likelihood: np.ndarray,
    go_times: np.ndarray,
    q40: float,
    q60: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    abs_starts = go_times[:, None] + BIN_EDGES_REL[:-1][None, :]
    abs_ends = go_times[:, None] + BIN_EDGES_REL[1:][None, :]

    idx_last = np.searchsorted(frame_times, abs_ends.ravel(), side="left").reshape(
        go_times.shape[0], N_BINS
    ) - 1
    clipped = np.clip(idx_last, 0, frame_times.size - 1)
    last_times = frame_times[clipped]
    last_y = tongue_y[clipped]
    last_lik = tongue_likelihood[clipped]

    in_bin = (idx_last >= 0) & (last_times >= abs_starts)
    visible = (
        in_bin
        & np.isfinite(last_y)
        & np.isfinite(last_lik)
        & (last_lik >= VISIBILITY_THRESHOLD)
    )

    out = np.full((go_times.shape[0], N_BINS), 3, dtype=np.int64)
    if np.any(visible):
        out[visible & (last_y < q40)] = 0
        out[visible & (last_y >= q40) & (last_y <= q60)] = 1
        out[visible & (last_y > q60)] = 2

    return out, last_y, visible


def choose_example_trial(visible_mask: np.ndarray, photostim_on: np.ndarray) -> int:
    visible_counts = visible_mask.sum(axis=1)
    stim_counts = photostim_on.sum(axis=1)
    order = np.lexsort((visible_counts, stim_counts))
    return int(order[-1])


def make_processing_plot(
    session_id: str,
    session_stats: dict,
    neural_trials: list[np.ndarray],
    input_trials: list[np.ndarray],
    output_trials: list[np.ndarray],
    sample_start_rel: np.ndarray,
    stim_start_rel: np.ndarray,
    stim_stop_rel: np.ndarray,
    tongue_y_last: np.ndarray,
    tongue_visible: np.ndarray,
    tongue_q40: float,
    tongue_q60: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    ax = axes[0, 0]
    labels = ["raw trials", "obs-valid", "event-valid", "kept"]
    counts = [
        session_stats["n_trials_raw"],
        session_stats["n_trials_after_good_trial_mask"],
        session_stats["n_trials_after_event_checks"],
        session_stats["n_trials_kept"],
    ]
    ax.bar(range(len(labels)), counts, color=["0.7", "0.5", "0.4", "0.2"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Trials")
    ax.set_title("Trial Filtering")

    ax = axes[0, 1]
    ax.hist(sample_start_rel, bins=40, color="tab:blue", alpha=0.8)
    ax.axvline(-1.85, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Sample start relative to go (s)")
    ax.set_ylabel("Trials")
    ax.set_title("Tone-Onset Alignment")

    ax = axes[1, 0]
    trial_idx = choose_example_trial(tongue_visible, input_trials[0][1][None, :] if False else np.stack([x[1] for x in input_trials]))
    example_neural = neural_trials[trial_idx]
    show_neurons = min(60, example_neural.shape[0])
    ax.imshow(
        example_neural[:show_neurons],
        aspect="auto",
        interpolation="nearest",
        extent=[OFF_START_S, OFF_END_S, show_neurons, 0],
        cmap="viridis",
    )
    ax.axvline(0.0, color="white", linestyle="--", linewidth=1)
    ax.set_xlabel("Time from go cue (s)")
    ax.set_ylabel("Neuron")
    ax.set_title("Example Neural Firing Rates")

    ax = axes[1, 1]
    example_input = input_trials[trial_idx]
    ax.plot(BIN_CENTERS_REL, example_input[0], label="time_from_tone_onset_s")
    ax.step(BIN_CENTERS_REL, example_input[1], where="mid", label="photostim_on")
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Time from go cue (s)")
    ax.set_title("Example Decoder Inputs")
    ax.legend(loc="upper left")

    ax = axes[2, 0]
    y = tongue_y_last[trial_idx]
    vis = tongue_visible[trial_idx]
    ax.plot(BIN_CENTERS_REL[vis], y[vis], "o-", label="visible binned tongue y", color="tab:red")
    ax.plot(BIN_CENTERS_REL[~vis], np.full(np.sum(~vis), tongue_q40), "x", label="not visible", color="0.5")
    ax.axhline(tongue_q40, color="tab:green", linestyle="--", linewidth=1, label="40th pct")
    ax.axhline(tongue_q60, color="tab:purple", linestyle="--", linewidth=1, label="60th pct")
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Time from go cue (s)")
    ax.set_ylabel("Tongue y")
    ax.set_title("Example Tongue Processing")
    ax.legend(loc="best", fontsize=8)

    ax = axes[2, 1]
    choice_vals = np.concatenate([trial[0] for trial in output_trials])
    outcome_vals = np.concatenate([trial[1] for trial in output_trials])
    tongue_vals = np.concatenate([trial[3] for trial in output_trials])
    ax.bar(
        np.arange(3) - 0.25,
        [np.mean(choice_vals == i) for i in range(3)],
        width=0.25,
        label="choice",
    )
    ax.bar(
        np.arange(3),
        [np.mean(outcome_vals == i) for i in range(3)],
        width=0.25,
        label="outcome",
    )
    ax.bar(
        np.arange(4) + 0.25,
        [np.mean(tongue_vals == i) for i in range(4)],
        width=0.25,
        label="tongue_y_bin",
    )
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["0", "1", "2", "3"])
    ax.set_ylim(0, 1)
    ax.set_title("Output Distributions")
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"Processing Summary: {session_id}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def process_session(session_path: Path, make_plot: bool = False) -> dict | None:
    session_id = session_path.stem
    session_t0 = now()
    print(f"\nProcessing {session_id}")
    with h5py.File(session_path, "r") as nwb:
        subject_id = session_path.parent.name
        classification = read_str_array(nwb["units/classification"])
        good_unit_idx = np.flatnonzero(classification == "good")
        n_units_raw = classification.shape[0]
        if good_unit_idx.size == 0:
            print(f"  Skipping {session_id}: no good units in NWB classification.")
            return None

        trials = nwb["intervals/trials"]
        trial_start = np.asarray(trials["start_time"][:], dtype=np.float64)
        trial_stop = np.asarray(trials["stop_time"][:], dtype=np.float64)
        n_trials_raw = trial_start.shape[0]

        is_good_trials_shape = tuple(nwb["units/is_good_trials"].shape)
        common_obs_start, common_obs_end = get_common_observation_window(
            np.asarray(nwb["units/obs_intervals"][:], dtype=np.float64),
            np.asarray(nwb["units/obs_intervals_index"][:], dtype=np.int64),
            good_unit_idx.astype(np.int64),
        )
        if not np.isfinite(common_obs_start) or not np.isfinite(common_obs_end):
            print(f"  Skipping {session_id}: invalid good-unit observation interval.")
            return None
        if common_obs_end <= common_obs_start:
            print(f"  Skipping {session_id}: empty shared good-unit observation interval.")
            return None

        go_start = np.asarray(
            nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:],
            dtype=np.float64,
        )
        go_stop = np.asarray(
            nwb["acquisition/BehavioralEvents/go_stop_times/timestamps"][:],
            dtype=np.float64,
        )
        if go_start.shape[0] != n_trials_raw:
            raise ValueError(f"{session_id}: expected one go cue per trial.")
        if go_stop.shape[0] != n_trials_raw:
            go_stop = np.minimum(go_start + 1.5, trial_stop)
        response_stop = np.minimum(go_start + 1.5, trial_stop)
        common_good_trials = (
            np.isfinite(go_start)
            & ((go_start + OFF_START_S) >= common_obs_start)
            & ((go_start + OFF_END_S) <= common_obs_end)
        )
        n_trials_after_good_trial_mask = int(common_good_trials.sum())

        sample_start_times = np.asarray(
            nwb["acquisition/BehavioralEvents/sample_start_times/timestamps"][:],
            dtype=np.float64,
        )
        sample_start, sample_valid = get_last_events_within(sample_start_times, trial_start, go_start)

        photostim_start_times = np.asarray(
            nwb["acquisition/BehavioralEvents/photostim_start_times/timestamps"][:],
            dtype=np.float64,
        )
        photostim_stop_times = np.asarray(
            nwb["acquisition/BehavioralEvents/photostim_stop_times/timestamps"][:],
            dtype=np.float64,
        )
        stim_start, _ = get_first_events_within(photostim_start_times, trial_start, trial_stop)
        stim_stop, _ = get_last_events_within(photostim_stop_times, trial_start, trial_stop)

        event_valid = sample_valid & np.isfinite(go_start) & np.isfinite(response_stop)
        keep_trials = common_good_trials & event_valid
        n_trials_after_event_checks = int((common_good_trials & event_valid).sum())

        if int(keep_trials.sum()) < 2:
            print(f"  Skipping {session_id}: fewer than 2 valid trials after filtering.")
            return None

        keep_idx = np.flatnonzero(keep_trials)
        keep_go_start = go_start[keep_idx]
        keep_response_stop = response_stop[keep_idx]
        sample_start_rel = sample_start[keep_idx] - keep_go_start
        stim_start_rel = stim_start[keep_idx] - keep_go_start
        stim_stop_rel = stim_stop[keep_idx] - keep_go_start

        t_spike = now()
        neural = bin_spikes_for_session(
            nwb["units/spike_times"],
            np.asarray(nwb["units/spike_times_index"][:], dtype=np.int64),
            good_unit_idx.astype(np.int64),
            keep_go_start,
        )
        print(f"  Neural binning: {now() - t_spike:.2f}s")
        neural_supported = np.any(neural != 0, axis=(1, 2))
        n_zero_neural_trials = int((~neural_supported).sum())
        if n_zero_neural_trials:
            keep_idx = keep_idx[neural_supported]
            keep_go_start = keep_go_start[neural_supported]
            keep_response_stop = keep_response_stop[neural_supported]
            sample_start_rel = sample_start_rel[neural_supported]
            stim_start_rel = stim_start_rel[neural_supported]
            stim_stop_rel = stim_stop_rel[neural_supported]
            neural = neural[neural_supported]
        if keep_idx.shape[0] < 2:
            print(f"  Skipping {session_id}: fewer than 2 neural-supported trials after filtering.")
            return None

        t_labels = now()
        outcome_raw = read_str_array(trials["outcome"])[keep_idx]
        early_raw = read_str_array(trials["early_lick"])[keep_idx]
        outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
        early_map = {"no early": 0, "early": 1}
        outcome = np.array([outcome_map[x] for x in outcome_raw], dtype=np.int64)
        early = np.array([early_map[x] for x in early_raw], dtype=np.int64)

        left_lick_times = np.asarray(
            nwb["acquisition/BehavioralEvents/left_lick_times/timestamps"][:],
            dtype=np.float64,
        )
        right_lick_times = np.asarray(
            nwb["acquisition/BehavioralEvents/right_lick_times/timestamps"][:],
            dtype=np.float64,
        )
        choice = derive_choice_labels(
            left_lick_times,
            right_lick_times,
            keep_go_start,
            keep_response_stop,
        )
        print(f"  Behavioral labels: {now() - t_labels:.2f}s")

        t_tongue = now()
        tongue_data = np.asarray(
            nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data"][:],
            dtype=np.float64,
        )
        tongue_times = np.asarray(
            nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps"][:],
            dtype=np.float64,
        )
        tongue_y = tongue_data[:, 1]
        tongue_lik = tongue_data[:, 2]
        session_visible = (
            np.isfinite(tongue_y)
            & np.isfinite(tongue_lik)
            & (tongue_lik >= VISIBILITY_THRESHOLD)
        )
        if np.any(session_visible):
            q40, q60 = np.quantile(tongue_y[session_visible], [0.4, 0.6])
        else:
            q40 = np.nan
            q60 = np.nan
        tongue_bins, tongue_y_last, tongue_visible = align_tongue_bins(
            tongue_times,
            tongue_y,
            tongue_lik,
            keep_go_start,
            q40,
            q60,
        )
        print(f"  Tongue alignment: {now() - t_tongue:.2f}s")

        tone_time = make_tone_time_matrix(sample_start_rel)
        photostim_on = make_photostim_matrix(stim_start_rel, stim_stop_rel)

        choice_2d = np.broadcast_to(choice[:, None], (choice.shape[0], N_BINS))
        outcome_2d = np.broadcast_to(outcome[:, None], (outcome.shape[0], N_BINS))
        early_2d = np.broadcast_to(early[:, None], (early.shape[0], N_BINS))

        input_trials = [
            np.vstack((tone_time[i], photostim_on[i])).astype(np.float32, copy=False)
            for i in range(keep_idx.shape[0])
        ]
        output_trials = [
            np.vstack((choice_2d[i], outcome_2d[i], early_2d[i], tongue_bins[i])).astype(
                np.int8, copy=False
            )
            for i in range(keep_idx.shape[0])
        ]
        neural_trials = [neural[i].astype(np.float16, copy=False) for i in range(keep_idx.shape[0])]

        insertion_regions = load_insertion_regions(nwb)
        unit_electrodes = np.asarray(nwb["units/electrodes"][:], dtype=np.int64)
        unit_electrodes_index = np.asarray(nwb["units/electrodes_index"][:], dtype=np.int64)
        unit_electrode_start = unit_start_indices(unit_electrodes_index)
        anno_name = read_str_array(nwb["units/anno_name"])
        unit_region_names: list[str] = []
        for unit_i in good_unit_idx:
            region = normalize_region_name(anno_name[unit_i])
            if not region:
                start_idx = unit_electrode_start[unit_i]
                if start_idx < unit_electrodes.shape[0]:
                    region = insertion_regions[unit_electrodes[start_idx]]
            unit_region_names.append(region or "unknown")

        session_stats = {
            "session_id": session_id,
            "subject_id": subject_id,
            "n_units_raw": int(n_units_raw),
            "n_units_kept": int(good_unit_idx.size),
            "n_trials_raw": int(n_trials_raw),
            "n_trials_after_good_trial_mask": int(n_trials_after_good_trial_mask),
            "n_trials_after_event_checks": int(n_trials_after_event_checks),
            "n_trials_kept": int(keep_idx.shape[0]),
            "is_good_trials_shape": is_good_trials_shape,
            "common_obs_start": float(common_obs_start),
            "common_obs_end": float(common_obs_end),
            "n_zero_neural_trials_dropped": int(n_zero_neural_trials),
            "sample_start_rel_min": float(np.min(sample_start_rel)),
            "sample_start_rel_max": float(np.max(sample_start_rel)),
            "n_photostim_trials": int(np.sum(np.isfinite(stim_start_rel))),
            "tongue_q40": None if np.isnan(q40) else float(q40),
            "tongue_q60": None if np.isnan(q60) else float(q60),
            "processing_seconds": float(now() - session_t0),
        }
        print(
            "  Kept "
            f"{session_stats['n_units_kept']} good units and {session_stats['n_trials_kept']} trials "
            f"in {session_stats['processing_seconds']:.2f}s"
        )

        if make_plot:
            plot_name = Path(f"/app/processing_{session_id}.png")
            make_processing_plot(
                session_id=session_id,
                session_stats=session_stats,
                neural_trials=neural_trials,
                input_trials=input_trials,
                output_trials=output_trials,
                sample_start_rel=sample_start_rel,
                stim_start_rel=stim_start_rel,
                stim_stop_rel=stim_stop_rel,
                tongue_y_last=tongue_y_last,
                tongue_visible=tongue_visible,
                tongue_q40=q40,
                tongue_q60=q60,
                out_path=plot_name,
            )
            print(f"  Saved processing plot: {plot_name.name}")

        return {
            "session_id": session_id,
            "subject_id": subject_id,
            "neural_trials": neural_trials,
            "input_trials": input_trials,
            "output_trials": output_trials,
            "unit_region_names": unit_region_names,
            "stats": session_stats,
        }


def build_dataset(processed_sessions: list[dict]) -> dict:
    subjects_order = list(OrderedDict((sess["subject_id"], None) for sess in processed_sessions).keys())
    subject_to_idx = {subject: i for i, subject in enumerate(subjects_order)}

    brain_region_order = list(
        OrderedDict(
            (region, None)
            for sess in processed_sessions
            for region in sess["unit_region_names"]
        ).keys()
    )
    brain_region_to_idx = {region: i for i, region in enumerate(brain_region_order)}

    data = {
        "neural": [sess["neural_trials"] for sess in processed_sessions],
        "input": [sess["input_trials"] for sess in processed_sessions],
        "output": [sess["output_trials"] for sess in processed_sessions],
        "subjects": subjects_order,
        "subject_idx": np.array(
            [subject_to_idx[sess["subject_id"]] for sess in processed_sessions], dtype=np.int64
        ),
        "brain_regions": brain_region_order,
        "brain_region_idx": [
            np.array([brain_region_to_idx[r] for r in sess["unit_region_names"]], dtype=np.int64)
            for sess in processed_sessions
        ],
        "input_names": ["time_from_tone_onset_s", "photostim_on"],
        "output_names": ["choice", "outcome", "early_lick", "tongue_y_bin"],
        "output_values": [
            ["left", "right", "no lick"],
            ["ignore", "miss", "hit"],
            ["no", "yes"],
            ["lt40", "40to60", "gt60", "not visible"],
        ],
        "metadata": {
            "task_description": (
                "Auditory delayed-response licking task. Decoder predicts choice, outcome, "
                "early lick, and discretized tongue y from go-cue-aligned neural firing rates."
            ),
            "time_bin_size": 50.0,
            "temporal_alignment_event": "go cue onset",
            "off_start": OFF_START_S,
            "off_end": OFF_END_S,
            "visibility_threshold": VISIBILITY_THRESHOLD,
            "neural_representation": "firing_rate_hz",
            "sample_event_rule": "last sample_start within trial before go cue",
            "choice_rule": (
                "first lick in the answer window [go_start, min(go_start + 1.5 s, trial_stop)], "
                "else no lick"
            ),
            "trial_qc_rule": (
                "require the full decoding window [go-2.5 s, go+1.5 s) to lie within the "
                "shared units/obs_intervals support of retained good units, plus valid go/sample "
                "events, and drop any trial whose binned neural data are all zero"
            ),
            "session_ids": [sess["session_id"] for sess in processed_sessions],
            "session_info": [sess["stats"] for sess in processed_sessions],
            "source_archive": "DANDI:000363/0.230822.0128",
        },
    }
    return data


def main() -> None:
    args = parse_args()
    sample_only = bool(args.sample)
    show_processing = bool(args.show_processing)

    all_paths = sorted(DATA_DIR.glob("sub-*/*.nwb"))
    chosen_paths = choose_session_paths(all_paths, sample_only=False)
    if sample_only:
        # Sample mode should keep the first two valid sessions after QC, not merely the
        # first two files in the raw archive.
        chosen_paths = all_paths

    print(f"Found {len(all_paths)} NWB files under {DATA_DIR}")
    print(
        f"Conversion mode: {'sample (first 2 valid sessions)' if sample_only else 'full'}"
    )
    print(
        f"Neural bins: width={BIN_SIZE_S:.3f}s, window=[{OFF_START_S:.1f}, {OFF_END_S:.1f})s, "
        f"T={N_BINS}"
    )

    processed_sessions: list[dict] = []
    total_t0 = now()
    plots_remaining = 2 if show_processing else 0
    for session_path in chosen_paths:
        result = process_session(session_path, make_plot=plots_remaining > 0)
        if result is None:
            continue
        processed_sessions.append(result)
        if plots_remaining > 0:
            plots_remaining -= 1
        if sample_only and len(processed_sessions) >= 2:
            break

    if len(processed_sessions) < 2:
        raise RuntimeError("Need at least 2 valid sessions for decoder evaluation.")

    data = build_dataset(processed_sessions)
    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    with args.outpicklefile.open("wb") as f:
        pickle.dump(data, f)

    total_seconds = now() - total_t0
    total_trials = sum(len(sess["neural_trials"]) for sess in processed_sessions)
    total_neurons = sum(len(sess["unit_region_names"]) for sess in processed_sessions)
    print(
        f"\nSaved converted dataset to {args.outpicklefile} "
        f"({len(processed_sessions)} sessions, {total_trials} trials, {total_neurons} neurons)"
    )
    print(f"Total conversion time: {total_seconds:.2f}s")
    print(
        f"Average time per kept session: {total_seconds / max(1, len(processed_sessions)):.2f}s"
    )


if __name__ == "__main__":
    main()
