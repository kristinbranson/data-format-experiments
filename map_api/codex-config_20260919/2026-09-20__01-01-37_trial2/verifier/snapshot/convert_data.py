#!/usr/bin/env python3
"""Convert the MAP NWB dataset to the neural-decoder pickle schema.

All NWB access is through PyNWB. Spike bins are half-open and aligned to the
go-cue onset. See CONVERSION_NOTES.md for the scientific rationale.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pynwb import NWBHDF5IO


DATA_DIR = Path("/app/data")
BIN_SIZE_S = 0.050
OFF_START = -2.5
OFF_END = 1.5
EDGES_REL = np.arange(OFF_START, OFF_END + BIN_SIZE_S / 2, BIN_SIZE_S)
CENTERS_REL = EDGES_REL[:-1] + BIN_SIZE_S / 2
N_TIME = len(CENTERS_REL)
TONGUE_LIKELIHOOD_CUTOFF = 0.9


def nwb_files() -> list[Path]:
    """Return source sessions in deterministic subject/session order."""
    return sorted(DATA_DIR.glob("sub-*/*.nwb"))


def as_string_array(column) -> np.ndarray:
    return np.asarray(column[:]).astype(str)


def first_event_per_trial(
    events: np.ndarray, starts: np.ndarray, upper_bounds: np.ndarray, name: str
) -> np.ndarray:
    """Select the first event in each [trial start, upper bound] interval."""
    indices = np.searchsorted(events, starts, side="left")
    if np.any(indices >= len(events)):
        raise ValueError(f"Missing {name} event after a trial start")
    selected = events[indices]
    bad = selected > upper_bounds + 1e-8
    if np.any(bad):
        raise ValueError(f"Missing {name} event in {int(bad.sum())} trial intervals")
    return selected


def bin_spikes(
    units, good_indices: np.ndarray, go_times: np.ndarray
) -> np.ndarray:
    """Return trial x neuron x time firing rates in Hz."""
    absolute_edges = go_times[:, None] + EDGES_REL[None, :]
    rates = np.empty((len(go_times), len(good_indices), N_TIME), dtype=np.float32)
    spike_column = units["spike_times"]
    for out_idx, unit_idx in enumerate(good_indices):
        spikes = np.asarray(spike_column[int(unit_idx)], dtype=np.float64)
        edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
        rates[:, out_idx, :] = np.diff(edge_positions, axis=1) / BIN_SIZE_S
    return rates


def trial_photostimulation(trials_df, centers_abs: np.ndarray) -> np.ndarray:
    """Binary photostimulation state evaluated at each bin center."""
    onset_raw = np.asarray(trials_df.photostim_onset).astype(str)
    duration_raw = np.asarray(trials_df.photostim_duration).astype(str)
    trial_start = np.asarray(trials_df.start_time, dtype=np.float64)
    onset = np.full(len(trials_df), np.nan)
    duration = np.full(len(trials_df), np.nan)
    active = onset_raw != "N/A"
    onset[active] = np.asarray(onset_raw[active], dtype=float) + trial_start[active]
    duration[active] = np.asarray(duration_raw[active], dtype=float)
    return (
        active[:, None]
        & (centers_abs >= onset[:, None])
        & (centers_abs < onset[:, None] + duration[:, None])
    ).astype(np.float32)


def derive_choice(instruction: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Map task result to actual lick choice: left=0, right=1, no lick=2."""
    instructed = np.where(instruction == "left", 0, 1)
    choice = instructed.copy()
    choice[outcome == "miss"] = 1 - choice[outcome == "miss"]
    choice[outcome == "ignore"] = 2
    return choice.astype(np.uint8)


def tongue_categories(nwb, centers_abs: np.ndarray) -> tuple[np.ndarray, tuple[float, float], dict]:
    """Align tongue y by last observation and discretize within session."""
    ts = nwb.acquisition["BehavioralTimeSeries"].time_series[
        "Camera0_side_TongueTracking"
    ]
    timestamps = np.asarray(ts.timestamps[:], dtype=np.float64)
    tracking = np.asarray(ts.data[:], dtype=np.float64)
    y_all = tracking[:, 1]
    likelihood_all = tracking[:, 2]
    visible_all = (
        np.isfinite(y_all)
        & np.isfinite(likelihood_all)
        & (likelihood_all >= TONGUE_LIKELIHOOD_CUTOFF)
    )
    if visible_all.sum() < 2:
        raise ValueError("Fewer than two visible tongue samples in session")
    q40, q60 = np.percentile(y_all[visible_all], [40, 60])

    frame_idx = np.searchsorted(timestamps, centers_abs, side="right") - 1
    valid_idx = (frame_idx >= 0) & (frame_idx < len(timestamps))
    safe_idx = np.clip(frame_idx, 0, len(timestamps) - 1)
    y = y_all[safe_idx]
    likelihood = likelihood_all[safe_idx]
    visible = (
        valid_idx
        & np.isfinite(y)
        & np.isfinite(likelihood)
        & (likelihood >= TONGUE_LIKELIHOOD_CUTOFF)
    )
    category = np.full(centers_abs.shape, 3, dtype=np.uint8)
    category[visible & (y < q40)] = 0
    category[visible & (y >= q40) & (y <= q60)] = 1
    category[visible & (y > q60)] = 2
    diagnostics = {
        "visible_native_fraction": float(visible_all.mean()),
        "visible_aligned_fraction": float(visible.mean()),
        "class_counts": np.bincount(category.ravel(), minlength=4).tolist(),
    }
    return category, (float(q40), float(q60)), diagnostics


def event_choice_check(nwb, trials_df, go_times: np.ndarray, choice: np.ndarray) -> dict:
    """Compare derived choice against the first left/right response lick."""
    events = nwb.acquisition["BehavioralEvents"].time_series
    left = np.asarray(events["left_lick_times"].timestamps[:], dtype=float)
    right = np.asarray(events["right_lick_times"].timestamps[:], dtype=float)
    event_choice = np.full(len(go_times), 2, dtype=np.uint8)
    for i, go in enumerate(go_times):
        stop = min(float(trials_df.stop_time.iloc[i]), go + 1.5)
        li = np.searchsorted(left, go, side="left")
        ri = np.searchsorted(right, go, side="left")
        lt = left[li] if li < len(left) and left[li] < stop else np.inf
        rt = right[ri] if ri < len(right) and right[ri] < stop else np.inf
        if lt < rt:
            event_choice[i] = 0
        elif rt < lt:
            event_choice[i] = 1
    return {
        "compared": int(len(choice)),
        "matches": int(np.sum(choice == event_choice)),
        "mismatches": int(np.sum(choice != event_choice)),
    }


def process_session(path: Path, region_to_idx: dict[str, int]) -> tuple[dict | None, dict]:
    started = time.perf_counter()
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        classifications = as_string_array(nwb.units["classification"])
        good = np.flatnonzero(classifications == "good")
        if len(good) == 0:
            return None, {"file": path.name, "skip_reason": "zero curated units"}

        trials_df_all = nwb.trials[:]
        # `is_good_trials` preserves the source ephys trial count. Eight NWBs
        # contain trailing behavioral-table rows beyond the recording; these
        # periods have no spikes for any unit and must not become zero trials.
        recorded_lengths = np.asarray(
            [len(nwb.units["is_good_trials"][int(i)]) for i in good], dtype=int
        )
        if np.any(recorded_lengths != recorded_lengths[0]):
            raise ValueError(f"{path.name}: inconsistent recorded-trial counts across units")
        n_recorded_trials = int(recorded_lengths[0])
        if n_recorded_trials > len(trials_df_all):
            raise ValueError(f"{path.name}: unit trial count exceeds behavioral table")
        trials_df = trials_df_all.iloc[:n_recorded_trials].copy()
        events = nwb.acquisition["BehavioralEvents"].time_series
        go_times_all = np.asarray(events["go_start_times"].timestamps[:], dtype=np.float64)
        if len(go_times_all) < n_recorded_trials:
            raise ValueError(f"{path.name}: {len(go_times_all)} go cues for {n_recorded_trials} recorded trials")
        go_times = go_times_all[:n_recorded_trials]

        trial_starts = np.asarray(trials_df.start_time, dtype=np.float64)
        sample_events = np.asarray(events["sample_start_times"].timestamps[:], dtype=np.float64)
        tone_onsets = first_event_per_trial(sample_events, trial_starts, go_times, "tone")
        centers_abs = go_times[:, None] + CENTERS_REL[None, :]

        rates = bin_spikes(nwb.units, good, go_times)
        # A rare trailing interval can be present in `obs_intervals` and the
        # validity-vector length despite containing no spikes from any unit.
        # Treat population-all-zero windows as absent neural data, not trials.
        neural_valid = np.any(rates != 0, axis=(1, 2))
        excluded_all_zero = int((~neural_valid).sum())
        if excluded_all_zero:
            rates = rates[neural_valid]
            trials_df = trials_df.iloc[np.flatnonzero(neural_valid)].copy()
            go_times = go_times[neural_valid]
            tone_onsets = tone_onsets[neural_valid]
            centers_abs = centers_abs[neural_valid]
        n_trials = len(trials_df)
        if n_trials < 2:
            raise ValueError(f"{path.name}: fewer than two trials with neural data")
        tone_time = (centers_abs - tone_onsets[:, None]).astype(np.float32)
        photostim = trial_photostimulation(trials_df, centers_abs)

        instruction = np.asarray(trials_df.trial_instruction).astype(str)
        outcome_text = np.asarray(trials_df.outcome).astype(str)
        early_text = np.asarray(trials_df.early_lick).astype(str)
        choice = derive_choice(instruction, outcome_text)
        outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
        outcome = np.asarray([outcome_map[x] for x in outcome_text], dtype=np.uint8)
        early = (early_text == "early").astype(np.uint8)
        tongue, tongue_thresholds, tongue_diag = tongue_categories(nwb, centers_abs)

        input_cube = np.stack([tone_time, photostim], axis=1)
        output_cube = np.empty((n_trials, 4, N_TIME), dtype=np.uint8)
        output_cube[:, 0, :] = choice[:, None]
        output_cube[:, 1, :] = outcome[:, None]
        output_cube[:, 2, :] = early[:, None]
        output_cube[:, 3, :] = tongue

        annotations = as_string_array(nwb.units["anno_name"])[good]
        if np.any(annotations == ""):
            raise ValueError(f"{path.name}: curated unit without CCF annotation")
        region_idx = np.empty(len(annotations), dtype=np.int32)
        for i, label in enumerate(annotations):
            if label not in region_to_idx:
                region_to_idx[label] = len(region_to_idx)
            region_idx[i] = region_to_idx[label]

        choice_check = event_choice_check(nwb, trials_df, go_times, choice)
        session = {
            "neural": [rates[i] for i in range(n_trials)],
            "input": [input_cube[i] for i in range(n_trials)],
            "output": [output_cube[i] for i in range(n_trials)],
            "subject": str(nwb.subject.subject_id),
            "brain_region_idx": region_idx,
            "info": {
                "session_id": str(nwb.identifier),
                "source_file": str(path),
                "n_trials": n_trials,
                "n_behavioral_table_trials": len(trials_df_all),
                "n_recorded_trial_entries": n_recorded_trials,
                "excluded_population_all_zero_trials": excluded_all_zero,
                "n_neurons": len(good),
                "tongue_percentiles_y": tongue_thresholds,
                "tongue_diagnostics": tongue_diag,
                "choice_event_check": choice_check,
            },
        }
    session["info"]["processing_seconds"] = time.perf_counter() - started
    return session, session["info"]


def plot_processing(session: dict, output_path: Path) -> None:
    """Plot every material processing layer for representative trials."""
    output_array = np.stack(session["output"])
    input_array = np.stack(session["input"])
    candidates = [
        0,
        int(np.argmax(np.sum(output_array[:, 3, :] != 3, axis=1))),
        int(np.argmax(np.sum(input_array[:, 1, :], axis=1))),
        int(np.argmax(output_array[:, 2, 0])),
    ]
    trials = np.asarray(list(dict.fromkeys(candidates)))
    fig, axes = plt.subplots(5, len(trials), figsize=(5 * len(trials), 15), sharex=True)
    if len(trials) == 1:
        axes = axes[:, None]
    extent = [OFF_START, OFF_END, 0, min(50, session["neural"][0].shape[0])]
    for col, trial in enumerate(trials):
        neural = session["neural"][trial]
        inputs = session["input"][trial]
        outputs = session["output"][trial]
        axes[0, col].imshow(neural[:50], aspect="auto", origin="lower", extent=extent)
        axes[0, col].set_title(f"trial {trial}: firing rates")
        axes[1, col].plot(CENTERS_REL, neural.mean(axis=0))
        axes[1, col].set_ylabel("mean Hz")
        axes[2, col].plot(CENTERS_REL, inputs[0], label="time from tone")
        axes[2, col].step(CENTERS_REL, inputs[1], where="mid", label="photostim")
        axes[2, col].legend(fontsize=8)
        axes[3, col].step(CENTERS_REL, outputs[0], label="choice")
        axes[3, col].step(CENTERS_REL, outputs[1], label="outcome")
        axes[3, col].step(CENTERS_REL, outputs[2], label="early")
        axes[3, col].legend(fontsize=8)
        axes[4, col].step(CENTERS_REL, outputs[3], where="mid")
        q40, q60 = session["info"]["tongue_percentiles_y"]
        axes[4, col].set_ylabel("tongue y class")
        axes[4, col].set_title(f"visible y thresholds: {q40:.1f}, {q60:.1f}")
        axes[4, col].set_xlabel("seconds from go cue")
        for row in range(5):
            axes[row, col].axvline(0, color="k", ls="--", lw=0.7)
    fig.suptitle(session["info"]["session_id"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def convert(output_path: Path, sample: bool, show_processing: bool) -> None:
    files = nwb_files()
    max_sessions = 2 if sample else None
    region_to_idx: dict[str, int] = {}
    subjects: list[str] = []
    subject_to_idx: dict[str, int] = {}
    neural, inputs, outputs = [], [], []
    subject_idx, brain_region_idx, session_info = [], [], []
    skipped = []
    total_started = time.perf_counter()

    for path in files:
        if max_sessions is not None and len(neural) >= max_sessions:
            break
        print(f"Processing {path.name}", flush=True)
        session, diagnostics = process_session(path, region_to_idx)
        if session is None:
            skipped.append(diagnostics)
            print(f"  skipped: {diagnostics['skip_reason']}", flush=True)
            continue
        subject = session["subject"]
        if subject not in subject_to_idx:
            subject_to_idx[subject] = len(subjects)
            subjects.append(subject)
        neural.append(session["neural"])
        inputs.append(session["input"])
        outputs.append(session["output"])
        subject_idx.append(subject_to_idx[subject])
        brain_region_idx.append(session["brain_region_idx"])
        session_info.append(session["info"])
        elapsed = session["info"]["processing_seconds"]
        print(
            f"  {session['info']['n_trials']} trials, {session['info']['n_neurons']} neurons, "
            f"{elapsed:.2f} s; tongue={session['info']['tongue_diagnostics']['class_counts']}",
            flush=True,
        )
        if show_processing and len(neural) <= 2:
            plot_path = Path(f"/app/processing_{session['info']['session_id']}.png")
            plot_processing(session, plot_path)
            print(f"  wrote {plot_path}", flush=True)

    brain_regions = [None] * len(region_to_idx)
    for label, index in region_to_idx.items():
        brain_regions[index] = label
    dataset = {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int32),
        "brain_regions": brain_regions,
        "brain_region_idx": brain_region_idx,
        "input_names": ["time from tone onset (s)", "photostimulation on"],
        "output_names": ["lick direction choice", "outcome", "early lick", "tongue y-position"],
        "output_values": [
            ["left", "right", "no lick"],
            ["ignore", "miss", "hit"],
            ["no", "yes"],
            ["below 40th percentile", "40th to 60th percentile", "above 60th percentile", "not visible"],
        ],
        "metadata": {
            "task_description": (
                "Auditory delayed-response directional-licking task; decode choice, outcome, "
                "early licking, and session-discretized tongue y-position from classifier-QC neural activity."
            ),
            "time_bin_size": 50.0,
            "temporal_alignment_event": "go cue onset",
            "off_start": OFF_START,
            "off_end": OFF_END,
            "neural_measure": "firing rate (Hz), half-open nonoverlapping 50-ms spike bins",
            "n_timepoints": N_TIME,
            "time_bin_centers_seconds_from_go": CENTERS_REL.astype(np.float32),
            "unit_filter": "NWB units.classification == 'good' (released region-specific classifier QC)",
            "trial_filter": "all trials retained; sessions with zero curated units excluded",
            "tongue_visibility_likelihood_cutoff": TONGUE_LIKELIHOOD_CUTOFF,
            "tongue_alignment": "last video frame at or before each neural-bin center",
            "tone_onset_definition": "first sample_start event within the native trial",
            "session_info": session_info,
            "skipped_sessions": skipped,
            "source": "DANDI:000363/0.230822.0128",
            "conversion_mode": "sample" if sample else "full",
        },
    }

    total_trials = sum(len(x) for x in neural)
    total_neurons = sum(x[0].shape[0] for x in neural)
    print(
        f"Writing {output_path}: {len(neural)} sessions, {total_trials} trials, "
        f"{total_neurons} neurons, {len(subjects)} subjects, {len(brain_regions)} regions",
        flush=True,
    )
    with output_path.open("wb") as stream:
        pickle.dump(dataset, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"Finished in {time.perf_counter() - total_started:.2f} s; "
        f"pickle size {output_path.stat().st_size / 1e9:.3f} GB",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="process all sessions (default)")
    mode.add_argument("--sample", action="store_true", help="process first two usable sessions")
    parser.add_argument("--show-processing", action="store_true")
    args = parser.parse_args()
    convert(args.outpicklefile, sample=args.sample, show_processing=args.show_processing)


if __name__ == "__main__":
    main()
