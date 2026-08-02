#!/usr/bin/env python3
import argparse
import gc
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path("data")
BIN_SIZE_S = 0.05
REL_START_S = -2.5
REL_END_S = 1.5
TONE_ONSET_REL_GO_S = -1.85
N_BINS = int(round((REL_END_S - REL_START_S) / BIN_SIZE_S))
REL_EDGES = np.linspace(REL_START_S, REL_END_S, N_BINS + 1, dtype=np.float64)
REL_CENTERS = (REL_EDGES[:-1] + REL_EDGES[1:]) / 2.0
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
SAMPLE_SESSION_COUNT = 2
ZERO_GOOD_SESSION = "sub-440958_ses-20190216T162508_behavior+ecephys+ogen.nwb"


@dataclass
class SessionResult:
    session_id: str
    subject_id: str
    neural_trials: list
    input_trials: list
    output_trials: list
    region_names: list
    region_labels_per_unit: np.ndarray
    n_raw_units: int
    n_good_units: int
    n_trials: int
    diagnostics: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MAP NWB data into the decoder-compatible pickle format."
    )
    parser.add_argument("outpicklefile", type=str, help="Output pickle path.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all valid sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only 2 valid sessions.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing diagnostic plots for up to 2 processed sessions.",
    )
    return parser.parse_args()


def decode_str_array(ds: h5py.Dataset) -> np.ndarray:
    arr = ds[()]
    if np.isscalar(arr):
        arr = np.array([arr], dtype=object)
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        elif isinstance(x, np.bytes_):
            out.append(x.astype(str))
        else:
            out.append(str(x))
    return np.array(out, dtype=object)


def parse_optional_float_array(strings: np.ndarray) -> np.ndarray:
    out = np.full(strings.shape, np.nan, dtype=np.float64)
    for i, value in enumerate(strings):
        if value == "N/A":
            continue
        out[i] = float(value)
    return out


def get_nwb_files(sample_only: bool) -> list[Path]:
    files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
    valid = []
    for path in files:
        with h5py.File(path, "r") as f:
            good = decode_str_array(f["units/classification"]) == "good"
            if np.any(good):
                valid.append(path)
    if sample_only:
        return valid[:SAMPLE_SESSION_COUNT]
    return valid


def get_ragged_row(flat: np.ndarray, index: np.ndarray, row: int) -> np.ndarray:
    start = 0 if row == 0 else int(index[row - 1])
    end = int(index[row])
    return flat[start:end]


def first_side(left_times: np.ndarray, right_times: np.ndarray) -> int | None:
    if left_times.size == 0 and right_times.size == 0:
        return None
    left_first = left_times[0] if left_times.size else np.inf
    right_first = right_times[0] if right_times.size else np.inf
    return 0 if left_first <= right_first else 1


def lick_choice_with_fallback(
    left_times: np.ndarray,
    right_times: np.ndarray,
    start_time: float,
    go_time: float,
    stop_time: float,
    instructed_choice: int,
) -> int:
    left_post = left_times[(left_times >= go_time) & (left_times <= stop_time)]
    right_post = right_times[(right_times >= go_time) & (right_times <= stop_time)]
    post_choice = first_side(left_post, right_post)
    if post_choice is not None:
        return post_choice

    left_any = left_times[(left_times >= start_time) & (left_times <= stop_time)]
    right_any = right_times[(right_times >= start_time) & (right_times <= stop_time)]
    any_choice = first_side(left_any, right_any)
    if any_choice is not None:
        return any_choice
    return instructed_choice


def clean_tongue_tracking(
    x: np.ndarray,
    y: np.ndarray,
    likelihood: np.ndarray,
) -> tuple[np.ndarray, dict]:
    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64).copy()
    likelihood = np.asarray(likelihood, dtype=np.float64)

    speed = np.zeros_like(x)
    if len(x) > 1:
        speed[1:] = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    speed_threshold = float(np.nanmean(speed) + 5.0 * np.nanstd(speed))
    outlier_mask = ~np.isfinite(x) | ~np.isfinite(y) | (speed > speed_threshold)

    frame_idx = np.arange(len(x), dtype=np.float64)
    keep_mask = ~outlier_mask
    if np.sum(keep_mask) >= 2:
        x[outlier_mask] = np.interp(frame_idx[outlier_mask], frame_idx[keep_mask], x[keep_mask])
        y[outlier_mask] = np.interp(frame_idx[outlier_mask], frame_idx[keep_mask], y[keep_mask])
    else:
        x[outlier_mask] = np.nanmean(x)
        y[outlier_mask] = np.nanmean(y)

    visible_mask = np.isfinite(likelihood) & (likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
    mean_y = float(np.nanmean(y[visible_mask])) if np.any(visible_mask) else float(np.nanmean(y))
    occluded_mask = ~visible_mask
    y[occluded_mask] = mean_y

    diagnostics = {
        "speed_threshold": speed_threshold,
        "n_velocity_outliers": int(np.sum(outlier_mask)),
        "n_low_likelihood_frames": int(np.sum(occluded_mask)),
        "mean_visible_y": mean_y,
    }
    return y, diagnostics


def align_tongue_y(
    timestamps: np.ndarray,
    cleaned_y: np.ndarray,
    go_times: np.ndarray,
) -> np.ndarray:
    abs_centers = go_times[:, None] + REL_CENTERS[None, :]
    idx = np.searchsorted(timestamps, abs_centers, side="right") - 1
    idx = np.clip(idx, 0, len(timestamps) - 1)
    return cleaned_y[idx]


def build_choice_array(
    trial_instruction: np.ndarray,
    outcome_code: np.ndarray,
    start_times: np.ndarray,
    go_times: np.ndarray,
    stop_times: np.ndarray,
    left_lick_times: np.ndarray,
    right_lick_times: np.ndarray,
) -> np.ndarray:
    choice = np.zeros(len(trial_instruction), dtype=np.int16)
    instructed = np.where(trial_instruction == "left", 0, 1).astype(np.int16)

    hit_mask = outcome_code == 2
    miss_mask = outcome_code == 1
    ignore_mask = outcome_code == 0

    choice[hit_mask] = instructed[hit_mask]
    choice[miss_mask] = 1 - instructed[miss_mask]

    ignore_trials = np.where(ignore_mask)[0]
    for trial_idx in ignore_trials:
        choice[trial_idx] = lick_choice_with_fallback(
            left_times=left_lick_times,
            right_times=right_lick_times,
            start_time=float(start_times[trial_idx]),
            go_time=float(go_times[trial_idx]),
            stop_time=float(stop_times[trial_idx]),
            instructed_choice=int(instructed[trial_idx]),
        )
    return choice


def build_photostim_matrix(
    photostim_onset_str: np.ndarray,
    photostim_duration_str: np.ndarray,
    start_times: np.ndarray,
    go_times: np.ndarray,
) -> np.ndarray:
    onset_trial = parse_optional_float_array(photostim_onset_str)
    duration = parse_optional_float_array(photostim_duration_str)
    go_minus_start = go_times - start_times
    onset_rel_go = onset_trial - go_minus_start
    offset_rel_go = onset_rel_go + duration

    stim = np.zeros((len(go_times), N_BINS), dtype=np.float16)
    valid = np.isfinite(onset_rel_go) & np.isfinite(offset_rel_go)
    for trial_idx in np.where(valid)[0]:
        mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & (REL_CENTERS < offset_rel_go[trial_idx])
        stim[trial_idx, mask] = 1.0
    return stim


def bin_spikes_to_firing_rates(
    spike_times_flat: np.ndarray,
    spike_times_index: np.ndarray,
    good_unit_indices: np.ndarray,
    trial_edges_abs: np.ndarray,
) -> np.ndarray:
    n_trials = trial_edges_abs.shape[0]
    n_edges = trial_edges_abs.shape[1]
    flat_edges = trial_edges_abs.reshape(-1)
    start_index = np.concatenate(([0], spike_times_index[:-1]))
    firing_rates = np.empty((len(good_unit_indices), n_trials, n_edges - 1), dtype=np.float16)

    for i, unit_idx in enumerate(good_unit_indices):
        spikes = spike_times_flat[start_index[unit_idx]: spike_times_index[unit_idx]]
        edge_idx = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_edges)
        counts = np.diff(edge_idx, axis=1)
        firing_rates[i] = (counts / BIN_SIZE_S).astype(np.float16)
    return firing_rates


def compute_valid_trial_mask(
    obs_intervals: np.ndarray,
    go_times: np.ndarray,
) -> np.ndarray:
    trial_start = go_times + REL_START_S
    trial_end = go_times + REL_END_S
    starts = obs_intervals[:, 0][None, :]
    ends = obs_intervals[:, 1][None, :]
    return np.any((trial_start[:, None] >= starts) & (trial_end[:, None] <= ends), axis=1)


def make_session_plot(
    result: SessionResult,
    out_path: Path,
) -> None:
    diag = result.diagnostics
    fig, ax = plt.subplots(3, 2, figsize=(16, 12))

    trial_idx = int(diag["example_trial_idx"])
    neural = result.neural_trials[trial_idx]
    show_neurons = min(40, neural.shape[0])
    ax[0, 0].imshow(neural[:show_neurons], aspect="auto", cmap="viridis", interpolation="nearest")
    ax[0, 0].set_title(f"Neural FR Heatmap ({show_neurons} neurons, trial {trial_idx})")
    ax[0, 0].set_xlabel("50 ms bins")
    ax[0, 0].set_ylabel("Neuron")

    ax[0, 1].plot(REL_CENTERS, result.input_trials[trial_idx][0], label="time_from_tone_onset_s")
    ax[0, 1].step(REL_CENTERS, result.input_trials[trial_idx][1], where="mid", label="photostim_on")
    ax[0, 1].set_title("Inputs")
    ax[0, 1].legend(loc="upper left")

    ax[1, 0].plot(diag["tracking_preview_time"], diag["tracking_preview_raw"], alpha=0.6, label="raw tongue y")
    ax[1, 0].plot(diag["tracking_preview_time"], diag["tracking_preview_clean"], alpha=0.8, label="clean tongue y")
    ax[1, 0].set_title("Tongue Tracking Cleaning")
    ax[1, 0].legend(loc="upper right")

    ax[1, 1].plot(REL_CENTERS, diag["aligned_tongue_y_trial"], label="aligned y")
    ax[1, 1].step(REL_CENTERS, result.output_trials[trial_idx][3], where="mid", label="discrete class")
    ax[1, 1].set_title("Aligned Tongue Y")
    ax[1, 1].legend(loc="upper left")

    ax[2, 0].hist(diag["photostim_onsets_rel_go"], bins=30)
    ax[2, 0].set_title("Photostim Onset Relative to Go")
    ax[2, 0].set_xlabel("Seconds")

    outcome_vals = np.array([trial[1, 0] for trial in result.output_trials], dtype=np.int16)
    choice_vals = np.array([trial[0, 0] for trial in result.output_trials], dtype=np.int16)
    early_vals = np.array([trial[2, 0] for trial in result.output_trials], dtype=np.int16)
    ax[2, 1].hist(
        [choice_vals, outcome_vals, early_vals],
        bins=np.arange(-0.5, 3.5, 1.0),
        label=["choice", "outcome", "early"],
    )
    ax[2, 1].set_title("Output Value Distributions")
    ax[2, 1].legend(loc="upper right")

    fig.suptitle(f"Processing Diagnostics: {result.session_id}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def process_session(path: Path) -> SessionResult:
    t0 = time.perf_counter()
    session_id = path.stem
    subject_id = path.parent.name
    print(f"[session] {session_id} - loading")

    with h5py.File(path, "r") as f:
        trials = f["intervals/trials"]
        n_trials_raw = len(trials["id"])

        classification = decode_str_array(f["units/classification"])
        good_mask = classification == "good"
        good_unit_indices = np.flatnonzero(good_mask)
        if len(good_unit_indices) == 0:
            raise ValueError(f"Session {session_id} has no good units")

        anno_name = decode_str_array(f["units/anno_name"])
        region_labels = anno_name[good_unit_indices]

        start_times_all = trials["start_time"][()].astype(np.float64)
        stop_times_all = trials["stop_time"][()].astype(np.float64)
        go_times_all = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()].astype(np.float64)
        if len(go_times_all) != n_trials_raw:
            raise ValueError(f"{session_id}: go cue count {len(go_times_all)} does not match trial count {n_trials_raw}")

        obs_intervals_index = f["units/obs_intervals_index"][()]
        obs_intervals = get_ragged_row(
            f["units/obs_intervals"],
            obs_intervals_index,
            int(good_unit_indices[0]),
        )
        valid_trial_mask = compute_valid_trial_mask(obs_intervals=obs_intervals, go_times=go_times_all)
        if not np.any(valid_trial_mask):
            raise ValueError(f"{session_id}: no valid trials found from obs_intervals")

        start_times = start_times_all[valid_trial_mask]
        stop_times = stop_times_all[valid_trial_mask]
        go_times = go_times_all[valid_trial_mask]
        n_trials = len(go_times)

        trial_instruction = decode_str_array(trials["trial_instruction"])[valid_trial_mask]
        outcome_str = decode_str_array(trials["outcome"])[valid_trial_mask]
        early_lick_str = decode_str_array(trials["early_lick"])[valid_trial_mask]
        photostim_onset_str = decode_str_array(trials["photostim_onset"])[valid_trial_mask]
        photostim_duration_str = decode_str_array(trials["photostim_duration"])[valid_trial_mask]

        outcome_code = np.array([{"ignore": 0, "miss": 1, "hit": 2}[x] for x in outcome_str], dtype=np.int16)
        early_code = np.array([{"no early": 0, "early": 1}[x] for x in early_lick_str], dtype=np.int16)

        left_lick_times = f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()].astype(np.float64)
        right_lick_times = f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()].astype(np.float64)
        choice_code = build_choice_array(
            trial_instruction=trial_instruction,
            outcome_code=outcome_code,
            start_times=start_times,
            go_times=go_times,
            stop_times=stop_times,
            left_lick_times=left_lick_times,
            right_lick_times=right_lick_times,
        )

        input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
        input_stim = build_photostim_matrix(
            photostim_onset_str=photostim_onset_str,
            photostim_duration_str=photostim_duration_str,
            start_times=start_times,
            go_times=go_times,
        )

        tongue_ts = f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
        tongue_data = tongue_ts["data"][()]
        tongue_x = tongue_data[:, 0]
        tongue_y = tongue_data[:, 1]
        tongue_likelihood = tongue_data[:, 2]
        tongue_timestamps = tongue_ts["timestamps"][()].astype(np.float64)
        cleaned_tongue_y, tongue_clean_diag = clean_tongue_tracking(
            x=tongue_x,
            y=tongue_y,
            likelihood=tongue_likelihood,
        )
        aligned_tongue_y = align_tongue_y(
            timestamps=tongue_timestamps,
            cleaned_y=cleaned_tongue_y,
            go_times=go_times,
        )

        spike_times_flat = f["units/spike_times"][()]
        spike_times_index = f["units/spike_times_index"][()]
        trial_edges_abs = go_times[:, None] + REL_EDGES[None, :]
        t_neural = time.perf_counter()
        firing_rates = bin_spikes_to_firing_rates(
            spike_times_flat=spike_times_flat,
            spike_times_index=spike_times_index,
            good_unit_indices=good_unit_indices,
            trial_edges_abs=trial_edges_abs,
        )
        nonzero_trial_mask = np.any(firing_rates != 0, axis=(0, 2))
        if not np.all(nonzero_trial_mask):
            firing_rates = firing_rates[:, nonzero_trial_mask, :]
            start_times = start_times[nonzero_trial_mask]
            stop_times = stop_times[nonzero_trial_mask]
            go_times = go_times[nonzero_trial_mask]
            trial_instruction = trial_instruction[nonzero_trial_mask]
            outcome_code = outcome_code[nonzero_trial_mask]
            early_code = early_code[nonzero_trial_mask]
            choice_code = choice_code[nonzero_trial_mask]
            photostim_onset_str = photostim_onset_str[nonzero_trial_mask]
            input_time = input_time[nonzero_trial_mask]
            input_stim = input_stim[nonzero_trial_mask]
            aligned_tongue_y = aligned_tongue_y[nonzero_trial_mask]
            n_trials = len(go_times)
        neural_time_s = time.perf_counter() - t_neural

        q40, q60 = np.percentile(aligned_tongue_y.reshape(-1), [40, 60])
        tongue_disc = np.ones(aligned_tongue_y.shape, dtype=np.int16)
        tongue_disc[aligned_tongue_y < q40] = 0
        tongue_disc[aligned_tongue_y > q60] = 2

    neural_trials = [firing_rates[:, i, :].copy() for i in range(n_trials)]
    del firing_rates
    gc.collect()

    input_trials = []
    output_trials = []
    for trial_idx in range(n_trials):
        input_trial = np.vstack([input_time[trial_idx], input_stim[trial_idx]]).astype(np.float16)
        output_trial = np.vstack(
            [
                np.full(N_BINS, choice_code[trial_idx], dtype=np.int16),
                np.full(N_BINS, outcome_code[trial_idx], dtype=np.int16),
                np.full(N_BINS, early_code[trial_idx], dtype=np.int16),
                tongue_disc[trial_idx].astype(np.int16),
            ]
        )
        input_trials.append(input_trial)
        output_trials.append(output_trial)

    photostim_onsets_rel_go = []
    onset_trial = parse_optional_float_array(photostim_onset_str)
    go_minus_start = go_times - start_times
    onset_rel_go = onset_trial - go_minus_start
    photostim_onsets_rel_go = onset_rel_go[np.isfinite(onset_rel_go)]

    preview_n = min(2000, len(tongue_timestamps))
    diagnostics = {
        "tracking_preview_time": tongue_timestamps[:preview_n] - go_times[0],
        "tracking_preview_raw": tongue_y[:preview_n],
        "tracking_preview_clean": cleaned_tongue_y[:preview_n],
        "aligned_tongue_y_trial": aligned_tongue_y[0],
        "photostim_onsets_rel_go": photostim_onsets_rel_go,
        "example_trial_idx": 0,
        "q40": float(q40),
        "q60": float(q60),
        "n_trials": int(n_trials),
        "n_trials_raw": int(n_trials_raw),
        "n_trials_obs_valid": int(np.sum(valid_trial_mask)),
        "n_good_units": int(len(good_unit_indices)),
        "n_raw_units": int(len(classification)),
        "binning_time_s": float(neural_time_s),
        "session_time_s": float(time.perf_counter() - t0),
        **tongue_clean_diag,
    }

    print(
        f"[session] {session_id} - done "
        f"({n_trials} trials, {len(good_unit_indices)} good units, "
        f"{diagnostics['session_time_s']:.1f}s total, {neural_time_s:.1f}s binning)"
    )

    return SessionResult(
        session_id=session_id,
        subject_id=subject_id,
        neural_trials=neural_trials,
        input_trials=input_trials,
        output_trials=output_trials,
        region_names=sorted(set(region_labels.tolist())),
        region_labels_per_unit=region_labels.copy(),
        n_raw_units=int(len(classification)),
        n_good_units=int(len(good_unit_indices)),
        n_trials=int(n_trials),
        diagnostics=diagnostics,
    )


def build_dataset(results: list[SessionResult]) -> dict:
    subjects = []
    subject_to_idx = {}
    brain_regions = []
    region_to_idx = {}

    data = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": subjects,
        "subject_idx": np.empty(len(results), dtype=np.int16),
        "brain_regions": brain_regions,
        "brain_region_idx": [],
        "input_names": ["time_from_tone_onset_s", "photostim_on"],
        "output_names": ["choice", "outcome", "early_lick", "tongue_y_position"],
        "output_values": [
            ["left", "right"],
            ["ignore", "miss", "hit"],
            ["no", "yes"],
            ["lt_40pct", "40_to_60pct", "gt_60pct"],
        ],
        "metadata": {
            "task_description": (
                "Auditory delayed-response task aligned to go cue. Decoder predicts "
                "choice, outcome, early lick, and discretized tongue y-position from neural activity."
            ),
            "time_bin_size": 50.0,
            "temporal_alignment_event": "Go cue onset",
            "off_start": REL_START_S,
            "off_end": REL_END_S,
            "bin_centers_s": REL_CENTERS.astype(np.float32),
            "n_timepoints": N_BINS,
            "source_dataset": "Mesoscale Activity Map Dataset (local NWB copy)",
            "session_filter": (
                "Included sessions with at least one good unit; excluded the single raw session with zero good units."
            ),
            "excluded_sessions": [ZERO_GOOD_SESSION],
            "unit_filter": "units/classification == 'good'",
            "tone_onset_definition": "Canonical sample onset at -1.85 s relative to go cue from task structure",
            "photostim_definition": "Binary per-bin input from trial-table photostim onset/duration converted to go coordinates",
            "choice_definition": (
                "Hit=instruction side, miss=opposite side, ignore=lick-derived when available else instruction side fallback"
            ),
            "tongue_preprocessing": (
                "5-sigma velocity outlier interpolation, low-likelihood frames set to mean visible y, "
                "last-frame-carried-forward alignment to bin centers, per-session 40th/60th percentile discretization"
            ),
            "reference_processing_note": (
                "Reference session/unit curation and go-centered alignment preserved where possible; "
                "neural bins intentionally changed from reference 40 ms/3.4 ms to requested 50 ms bins"
            ),
            "session_ids": [r.session_id for r in results],
        },
    }

    for session_idx, result in enumerate(results):
        if result.subject_id not in subject_to_idx:
            subject_to_idx[result.subject_id] = len(subjects)
            subjects.append(result.subject_id)
        data["subject_idx"][session_idx] = subject_to_idx[result.subject_id]

        session_region_idx = np.empty(len(result.region_labels_per_unit), dtype=np.int16)
        for unit_idx, label in enumerate(result.region_labels_per_unit):
            if label not in region_to_idx:
                region_to_idx[label] = len(brain_regions)
                brain_regions.append(label)
            session_region_idx[unit_idx] = region_to_idx[label]

        data["neural"].append(result.neural_trials)
        data["input"].append(result.input_trials)
        data["output"].append(result.output_trials)
        data["brain_region_idx"].append(session_region_idx)

    return data


def main() -> None:
    args = parse_args()
    sample_only = args.sample and not args.full
    files = get_nwb_files(sample_only=sample_only)
    mode_name = "sample" if sample_only else "full"
    print(f"[setup] mode={mode_name}, sessions_to_process={len(files)}")

    results = []
    t_start = time.perf_counter()
    for i, path in enumerate(files, start=1):
        print(f"[progress] {i}/{len(files)} {path.name}")
        result = process_session(path)
        results.append(result)

        if args.show_processing and len(results) <= 2:
            plot_name = f"processing_{result.session_id}.png"
            make_session_plot(result, Path(plot_name))
            print(f"[plot] saved {plot_name}")

    data = build_dataset(results)
    with open(args.outpicklefile, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.perf_counter() - t_start
    out_size_mb = Path(args.outpicklefile).stat().st_size / (1024 * 1024)
    total_trials = sum(r.n_trials for r in results)
    total_good_units = sum(r.n_good_units for r in results)
    print(
        f"[done] wrote {args.outpicklefile} "
        f"({out_size_mb:.1f} MiB) in {elapsed / 60.0:.2f} min | "
        f"sessions={len(results)} trials={total_trials} good_units_total={total_good_units}"
    )


if __name__ == "__main__":
    main()
