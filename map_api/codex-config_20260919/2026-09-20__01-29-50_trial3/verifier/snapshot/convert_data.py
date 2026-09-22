#!/usr/bin/env python3
"""Convert the MAP NWB release to the decoder-compatible pickle format.

All NWB access is through pynwb. Spike rates use non-overlapping 50-ms bins
aligned to go cue onset over [-2.5, 1.5) seconds.
"""

from __future__ import annotations

import argparse
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
EDGES = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
TONGUE_VISIBLE_LIKELIHOOD = 0.9


def nearest_indices(timestamps: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Indices of nearest sorted timestamps for each query time."""
    right = np.searchsorted(timestamps, query, side="left")
    right = np.clip(right, 0, len(timestamps) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(query - timestamps[left]) <= np.abs(timestamps[right] - query)
    return np.where(choose_left, left, right)


def last_sample_before_go(sample_times: np.ndarray, trial_starts: np.ndarray,
                          go_times: np.ndarray) -> np.ndarray:
    """Last sample/tone onset in each trial at or before its go cue."""
    inds = np.searchsorted(sample_times, go_times, side="right") - 1
    if np.any(inds < 0):
        raise ValueError("A trial has no sample onset before go cue")
    tone = sample_times[inds]
    if np.any(tone < trial_starts):
        bad = np.flatnonzero(tone < trial_starts)[:5]
        raise ValueError(f"Sample onset falls before trial start for trials {bad.tolist()}")
    return tone


def photostim_bins(starts: np.ndarray, stops: np.ndarray,
                   absolute_edges: np.ndarray) -> np.ndarray:
    """Mark bins with positive-duration overlap with any photostimulation."""
    n_trials = absolute_edges.shape[0]
    result = np.zeros((n_trials, len(CENTERS)), dtype=np.float32)
    left, right = absolute_edges[:, :-1], absolute_edges[:, 1:]
    # There are comparatively few laser intervals; each operation is vectorized
    # over every trial and time bin.
    for onset, offset in zip(starts, stops):
        result[np.logical_and(left < offset, right > onset)] = 1.0
    return result


def derive_choice(instruction: str, outcome: str) -> int:
    """0 left, 1 right, 2 no lick from task instruction and outcome."""
    if outcome == "ignore":
        return 2
    instructed = 0 if instruction == "left" else 1
    return instructed if outcome == "hit" else 1 - instructed


def full_unit_trial_mask(units, unit_i: int, trial_starts: np.ndarray,
                         trial_stops: np.ndarray) -> np.ndarray:
    """Expand a unit's insertion-local good-trial vector to session trials.

    NWB `is_good_trials` is indexed by the unit's `obs_intervals`, not always by
    the full session trial table. Each observation interval corresponds to one
    trial during that probe insertion.
    """
    local_good = np.asarray(units["is_good_trials"][unit_i], dtype=bool)
    intervals = np.asarray(units["obs_intervals"][unit_i], dtype=np.float64)
    if intervals.ndim == 1:
        intervals = intervals.reshape(1, 2)
    if len(local_good) != len(intervals):
        raise ValueError(f"unit {unit_i}: is_good_trials/obs_intervals mismatch")
    trial_i = np.searchsorted(trial_starts, intervals[:, 0], side="left")
    if np.any(trial_i >= len(trial_starts)):
        raise ValueError(f"unit {unit_i}: observation interval outside trial table")
    if not (np.allclose(trial_starts[trial_i], intervals[:, 0], atol=1e-5) and
            np.allclose(trial_stops[trial_i], intervals[:, 1], atol=1e-5)):
        raise ValueError(f"unit {unit_i}: observation intervals do not map to trials")
    full = np.zeros(len(trial_starts), dtype=bool)
    full[trial_i] = local_good
    return full


def first_response_lick_choice(left_licks: np.ndarray, right_licks: np.ndarray,
                               go: float) -> int:
    """Choice from the first lick in the 1.5-s response window, or 2."""
    li = np.searchsorted(left_licks, go, side="left")
    ri = np.searchsorted(right_licks, go, side="left")
    lt = left_licks[li] if li < len(left_licks) else np.inf
    rt = right_licks[ri] if ri < len(right_licks) else np.inf
    if min(lt, rt) >= go + 1.5:
        return 2
    return 0 if lt <= rt else 1


def plot_processing(session_id: str, plot_payload: dict) -> None:
    """Plot alignment, binning, inputs, outputs, and tongue discretization."""
    fr = plot_payload["fr"]
    inp = plot_payload["input"]
    out = plot_payload["output"]
    raw_y = plot_payload["raw_y"]
    raw_likelihood = plot_payload["raw_likelihood"]
    q40, q60 = plot_payload["quantiles"]

    fig, axes = plt.subplots(5, 1, figsize=(13, 16), constrained_layout=True)
    im = axes[0].imshow(fr[: min(80, len(fr))], aspect="auto", origin="lower",
                        extent=[OFF_START, OFF_END, 0, min(80, len(fr))], cmap="viridis")
    axes[0].axvline(0, color="w", ls="--", lw=1)
    axes[0].set(title="Trial 1: 50-ms firing rates (subset of neurons)", ylabel="Neuron")
    fig.colorbar(im, ax=axes[0], label="Hz")
    axes[1].plot(CENTERS, inp[0], label="seconds from tone onset")
    axes[1].step(CENTERS, inp[1], where="mid", label="photostimulation on")
    axes[1].axvline(0, color="k", ls="--", lw=1, label="go")
    axes[1].legend(); axes[1].set(title="Decoder inputs", ylabel="Value")
    for row, name in enumerate(["choice", "outcome", "early lick", "tongue-y class"]):
        axes[2].step(CENTERS, out[row] + row * 4, where="mid", label=name)
    axes[2].legend(ncol=2); axes[2].set(title="Decoder outputs (offset for display)", ylabel="Class + offset")
    vis = raw_likelihood >= TONGUE_VISIBLE_LIKELIHOOD
    axes[3].scatter(np.flatnonzero(vis)[::20], raw_y[vis][::20], s=1, alpha=.35, label="visible frames")
    axes[3].axhline(q40, color="C1", label="40th percentile")
    axes[3].axhline(q60, color="C2", label="60th percentile")
    axes[3].legend(); axes[3].set(title="Session tongue-y discretization", ylabel="y pixel")
    axes[4].hist(raw_likelihood, bins=100, log=True)
    axes[4].axvline(TONGUE_VISIBLE_LIKELIHOOD, color="r", label="visibility threshold")
    axes[4].legend(); axes[4].set(title="DeepLabCut tongue likelihood", xlabel="likelihood", ylabel="frames (log)")
    fig.suptitle(session_id)
    fig.savefig(f"/app/processing_{session_id}.png", dpi=130)
    plt.close(fig)


def convert_session(path: Path, show_processing: bool = False) -> tuple[dict | None, Counter]:
    """Load and convert one NWB session."""
    audit = Counter()
    with NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
        nwb = io.read()
        trials = nwb.trials.to_dataframe()
        classifications = np.asarray(nwb.units["classification"][:]).astype(str)
        unit_inds = np.flatnonzero(classifications == "good")
        if len(unit_inds) == 0:
            audit["sessions_zero_good_units"] += 1
            return None, audit

        trial_starts = trials.start_time.to_numpy(dtype=np.float64)
        trial_stops = trials.stop_time.to_numpy(dtype=np.float64)
        unit_valid = np.stack([
            full_unit_trial_mask(nwb.units, i, trial_starts, trial_stops) for i in unit_inds
        ])
        trial_mask = np.all(unit_valid, axis=0)
        trial_inds = np.flatnonzero(trial_mask)
        if len(trial_inds) < 2:
            raise ValueError(f"{path.name}: fewer than two jointly valid trials")
        audit["trials_native"] += len(trials)
        audit["trials_invalid_period"] += int(np.sum(~trial_mask))

        events = nwb.acquisition["BehavioralEvents"].time_series
        go_all = np.asarray(events["go_start_times"].timestamps[:], dtype=np.float64)
        if len(go_all) != len(trials):
            raise ValueError(f"{path.name}: go cue/trial count mismatch")
        sample_times = np.asarray(events["sample_start_times"].timestamps[:], dtype=np.float64)
        tone_all = last_sample_before_go(sample_times, trial_starts, go_all)
        go = go_all[trial_inds]
        tone = tone_all[trial_inds]
        absolute_edges = go[:, None] + EDGES[None, :]

        laser_starts = np.asarray(events["photostim_start_times"].timestamps[:], dtype=np.float64)
        laser_stops = np.asarray(events["photostim_stop_times"].timestamps[:], dtype=np.float64)
        if len(laser_starts) != len(laser_stops):
            raise ValueError(f"{path.name}: photostim start/stop mismatch")
        laser = photostim_bins(laser_starts, laser_stops, absolute_edges)
        tone_time = go[:, None] + CENTERS[None, :] - tone[:, None]

        # Vectorized binning: one searchsorted call per unit returns all trial edges.
        rates = np.empty((len(unit_inds), len(trial_inds), len(CENTERS)), dtype=np.float32)
        for out_i, unit_i in enumerate(unit_inds):
            spikes = np.asarray(nwb.units["spike_times"][unit_i], dtype=np.float64)
            edge_indices = np.searchsorted(spikes, absolute_edges, side="left")
            rates[out_i] = np.diff(edge_indices, axis=1) / BIN_SIZE_S

        # A small number of NWB observation intervals extend past the actual
        # final spike timestamp (an export edge case). A completely silent
        # hundreds-neuron population over four seconds marks an unrecorded trial,
        # not physiological silence, so exclude it as an invalid data period.
        recorded_trial = np.any(rates != 0, axis=(0, 2))
        if np.any(~recorded_trial):
            audit["trials_all_neural_zero_excluded"] += int(np.sum(~recorded_trial))
            trial_inds = trial_inds[recorded_trial]
            go = go[recorded_trial]
            tone = tone[recorded_trial]
            absolute_edges = absolute_edges[recorded_trial]
            laser = laser[recorded_trial]
            tone_time = tone_time[recorded_trial]
            rates = rates[:, recorded_trial, :]
        if len(trial_inds) < 2:
            raise ValueError(f"{path.name}: fewer than two recorded trials")

        tracking = nwb.acquisition["BehavioralTimeSeries"].time_series["Camera0_side_TongueTracking"]
        track_t = np.asarray(tracking.timestamps[:], dtype=np.float64)
        track_data = np.asarray(tracking.data[:], dtype=np.float64)
        raw_y, likelihood = track_data[:, 1], track_data[:, 2]
        visible = np.isfinite(raw_y) & np.isfinite(likelihood) & (likelihood >= TONGUE_VISIBLE_LIKELIHOOD)
        if not np.any(visible):
            raise ValueError(f"{path.name}: no visible tongue frames")
        q40, q60 = np.quantile(raw_y[visible], [0.4, 0.6])
        query = go[:, None] + CENTERS[None, :]
        track_idx = nearest_indices(track_t, query.ravel()).reshape(query.shape)
        y = raw_y[track_idx]
        tongue_visible = likelihood[track_idx] >= TONGUE_VISIBLE_LIKELIHOOD
        tongue_class = np.full(query.shape, 3, dtype=np.int8)
        tongue_class[tongue_visible & (y < q40)] = 0
        tongue_class[tongue_visible & (y >= q40) & (y <= q60)] = 1
        tongue_class[tongue_visible & (y > q60)] = 2

        tr = trials.iloc[trial_inds]
        instructions = tr.trial_instruction.astype(str).to_numpy()
        outcomes_str = tr.outcome.astype(str).to_numpy()
        early_str = tr.early_lick.astype(str).to_numpy()
        choices = np.asarray([derive_choice(a, b) for a, b in zip(instructions, outcomes_str)], dtype=np.int8)
        outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
        early_map = {"no early": 0, "early": 1}
        outcomes = np.asarray([outcome_map[x] for x in outcomes_str], dtype=np.int8)
        early = np.asarray([early_map[x] for x in early_str], dtype=np.int8)

        left_licks = np.asarray(events["left_lick_times"].timestamps[:], dtype=np.float64)
        right_licks = np.asarray(events["right_lick_times"].timestamps[:], dtype=np.float64)
        lick_choices = np.asarray([
            first_response_lick_choice(left_licks, right_licks, g) for g in go
        ], dtype=np.int8)
        audit["choice_lick_matches"] += int(np.sum(choices == lick_choices))
        audit["choice_lick_mismatches"] += int(np.sum(choices != lick_choices))

        neural, inputs, outputs = [], [], []
        for j in range(len(trial_inds)):
            neural.append(np.ascontiguousarray(rates[:, j, :], dtype=np.float32))
            inputs.append(np.ascontiguousarray(np.vstack((tone_time[j], laser[j])), dtype=np.float32))
            out = np.empty((4, len(CENTERS)), dtype=np.int8)
            out[0] = choices[j]
            out[1] = outcomes[j]
            out[2] = early[j]
            out[3] = tongue_class[j]
            outputs.append(out)

        regions = np.asarray(nwb.units["anno_name"][:]).astype(str)[unit_inds]
        session_id = str(nwb.identifier)
        subject = str(nwb.subject.subject_id)
        session = {
            "neural": neural,
            "input": inputs,
            "output": outputs,
            "subject": subject,
            "regions": regions.tolist(),
            "info": {
                "session_id": session_id,
                "source_file": str(path),
                "native_trials": int(len(trials)),
                "included_trials": int(len(trial_inds)),
                "good_units": int(len(unit_inds)),
                "tongue_y_q40": float(q40),
                "tongue_y_q60": float(q60),
                "auto_water_trials_included": int(tr.auto_water.sum()),
                "free_water_trials_included": int(tr.free_water.sum()),
            },
        }
        audit["sessions_included"] += 1
        audit["trials_included"] += len(trial_inds)
        audit["good_units"] += len(unit_inds)

        if show_processing:
            plot_processing(session_id, {
                "fr": neural[0], "input": inputs[0], "output": outputs[0],
                "raw_y": raw_y, "raw_likelihood": likelihood,
                "quantiles": (q40, q60),
            })
        return session, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="process all sessions (default)")
    mode.add_argument("--sample", action="store_true", help="process only two sessions")
    parser.add_argument("--show-processing", action="store_true")
    args = parser.parse_args()

    files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
    if args.sample:
        files = files[:2]
    print(f"Found {len(files)} NWB sessions; processing with pynwb", flush=True)
    started = time.perf_counter()
    converted = []
    audit = Counter()
    for i, path in enumerate(files):
        t0 = time.perf_counter()
        session, session_audit = convert_session(
            path, show_processing=args.show_processing and i < 2
        )
        audit.update(session_audit)
        if session is not None:
            converted.append(session)
            info = session["info"]
            label = f"{info['included_trials']} trials, {info['good_units']} neurons"
        else:
            label = "excluded (zero good neurons)"
        elapsed = time.perf_counter() - t0
        total_elapsed = time.perf_counter() - started
        eta = total_elapsed / (i + 1) * (len(files) - i - 1)
        print(f"[{i+1}/{len(files)}] {path.name}: {label}; {elapsed:.2f}s; ETA {eta/60:.1f} min", flush=True)

    subjects = sorted({x["subject"] for x in converted})
    subject_lookup = {x: i for i, x in enumerate(subjects)}
    brain_regions = sorted({r for x in converted for r in x["regions"]})
    region_lookup = {x: i for i, x in enumerate(brain_regions)}

    data = {
        "neural": [x["neural"] for x in converted],
        "input": [x["input"] for x in converted],
        "output": [x["output"] for x in converted],
        "subjects": subjects,
        "subject_idx": np.asarray([subject_lookup[x["subject"]] for x in converted], dtype=np.int32),
        "brain_regions": brain_regions,
        "brain_region_idx": [np.asarray([region_lookup[r] for r in x["regions"]], dtype=np.int32) for x in converted],
        "input_names": ["time from tone onset (s)", "photostimulation on"],
        "output_names": ["lick direction choice", "outcome", "early lick", "tongue y-position"],
        "output_values": [
            ["left", "right", "no lick"],
            ["ignore", "miss", "hit"],
            ["no", "yes"],
            ["below 40th percentile", "40th to 60th percentile", "above 60th percentile", "not visible"],
        ],
        "metadata": {
            "task_description": "Auditory delayed-response task; decode choice, outcome, early licking, and discretized tongue y-position from classifier-QC neural firing rates.",
            "time_bin_size": 50.0,
            "temporal_alignment_event": "go cue onset",
            "off_start": OFF_START,
            "off_end": OFF_END,
            "n_timepoints": len(CENTERS),
            "time_bin_units": "ms",
            "neural_units": "spikes/s",
            "bin_interval_convention": "left-closed, right-open; [-2.5, 1.5) s relative to go cue",
            "tone_onset_definition": "last sample/tone onset within trial at or before go cue (completed epoch after any early-lick replay)",
            "unit_filter": "NWB units classification == 'good' (paper's region-specific 15-metric classifier)",
            "trial_filter": "all retained units have is_good_trials == True; requested early/miss/ignore/stimulation classes retained",
            "tongue_visibility_likelihood_threshold": TONGUE_VISIBLE_LIKELIHOOD,
            "tongue_percentile_scope": "all visible side-camera tongue frames within each session",
            "session_info": [x["info"] for x in converted],
            "conversion_audit": dict(audit),
        },
    }

    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    print(f"Serializing {len(converted)} sessions to {args.outpicklefile}", flush=True)
    with args.outpicklefile.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_gb = args.outpicklefile.stat().st_size / 1e9
    elapsed = time.perf_counter() - started
    print(f"Finished in {elapsed/60:.2f} min; output {size_gb:.3f} GB", flush=True)
    print(f"Audit: {dict(audit)}", flush=True)


if __name__ == "__main__":
    main()
