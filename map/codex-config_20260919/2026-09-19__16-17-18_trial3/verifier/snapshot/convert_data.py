#!/usr/bin/env python3
"""Convert the MAP NWB release to the neural-decoder pickle schema.

Usage: python -u /app/convert_data.py OUT [--full|--sample] [--show-processing]
"""

from __future__ import annotations

import argparse
import glob
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path("/app/data")
OFF_START, OFF_END, BIN_SIZE_S = -2.5, 1.5, 0.050
BIN_EDGES = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
N_TIME = BIN_CENTERS.size
DLC_VISIBLE_THRESHOLD = 0.9


def decode_text(value) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def subject_from_path(path: str) -> str:
    return Path(path).name.split("_")[0].removeprefix("sub-")


def session_id_from_path(path: str) -> str:
    return Path(path).name.split("_behavior")[0]


def discover_usable_files() -> tuple[list[str], list[dict]]:
    """Return sorted sessions with at least one classifier-good unit."""
    files = sorted(glob.glob(str(DATA_DIR / "sub-*" / "*.nwb")))
    usable, excluded = [], []
    for path in files:
        with h5py.File(path, "r") as nwb:
            classification = nwb["units/classification"][()]
            n_good = int(np.count_nonzero(classification == b"good"))
            if n_good:
                usable.append(path)
            else:
                excluded.append({
                    "session_id": session_id_from_path(path),
                    "reason": "no classifier-good units / missing classifier labels",
                    "n_raw_units": int(classification.size),
                })
    return usable, excluded


def clean_tongue_tracking(timestamps: np.ndarray, data: np.ndarray):
    """Apply confidence masking and the paper's five-sigma velocity cleaning."""
    xy = np.asarray(data[:, :2], dtype=np.float64).copy()
    likelihood = np.asarray(data[:, 2], dtype=np.float64)
    finite = np.isfinite(timestamps) & np.all(np.isfinite(xy), axis=1)
    visible = finite & (likelihood >= DLC_VISIBLE_THRESHOLD)
    dt = np.diff(timestamps)
    dxy = np.diff(xy, axis=0)
    valid_velocity = (dt > 0) & (dt <= 0.010) & visible[:-1] & visible[1:]
    speed = np.full(timestamps.shape, np.nan, dtype=np.float64)
    speed[1:][valid_velocity] = np.linalg.norm(dxy[valid_velocity], axis=1) / dt[valid_velocity]
    velocity_values = speed[np.isfinite(speed)]
    if velocity_values.size:
        velocity_threshold = float(velocity_values.mean() + 5.0 * velocity_values.std())
        outlier = visible & (speed > velocity_threshold)
    else:
        velocity_threshold = float("inf")
        outlier = np.zeros_like(visible)
    base_valid = visible & ~outlier
    if np.count_nonzero(base_valid) < 2:
        raise ValueError("fewer than two clean visible tongue frames")
    # Only high-confidence velocity outliers are imputed. Low-confidence frames
    # remain invisible and therefore become output class 3.
    for dim in range(2):
        xy[outlier, dim] = np.interp(
            timestamps[outlier], timestamps[base_valid], xy[base_valid, dim]
        )
    clean_y = xy[:, 1]
    q40, q60 = np.percentile(clean_y[visible], [40.0, 60.0])
    summary = {
        "n_frames": int(timestamps.size),
        "visible_fraction": float(visible.mean()),
        "velocity_outliers": int(outlier.sum()),
        "velocity_threshold_px_per_s": velocity_threshold,
    }
    return clean_y, visible, outlier, float(q40), float(q60), summary


def sample_tongue_categories(centers_global, timestamps, clean_y, visible, q40, q60):
    """Use the final raw frame at/before each center, as in reference alignment."""
    flat = centers_global.ravel()
    idx = np.searchsorted(timestamps, flat, side="right") - 1
    safe_idx = np.clip(idx, 0, timestamps.size - 1)
    age = flat - timestamps[safe_idx]
    near = (idx >= 0) & (age >= -1e-9) & (age <= 0.0051)
    is_visible = near & visible[safe_idx]
    categories = np.full(flat.shape, 3, dtype=np.int8)
    y = clean_y[safe_idx]
    categories[is_visible & (y < q40)] = 0
    categories[is_visible & (y >= q40) & (y <= q60)] = 1
    categories[is_visible & (y > q60)] = 2
    return categories.reshape(centers_global.shape), safe_idx.reshape(centers_global.shape)


def photostim_state(centers_global, starts, stops):
    if starts.size == 0:
        return np.zeros(centers_global.shape, dtype=np.float32)
    if starts.size != stops.size:
        raise ValueError(f"photostim starts/stops mismatch: {starts.size} vs {stops.size}")
    flat = centers_global.ravel()
    interval_idx = np.searchsorted(starts, flat, side="right") - 1
    safe_idx = np.clip(interval_idx, 0, starts.size - 1)
    on = (interval_idx >= 0) & (flat >= starts[safe_idx]) & (flat < stops[safe_idx])
    return on.reshape(centers_global.shape).astype(np.float32)


def bin_selected_units(spike_times_ds, spike_index, unit_indices, go_times):
    """Assign each selected-unit spike to its non-overlapping trial/bin."""
    window_starts = go_times + OFF_START
    window_ends = go_times + OFF_END
    if np.any(window_starts[1:] < window_ends[:-1]):
        raise ValueError("requested trial windows overlap")
    n_trials = go_times.size
    rates = np.zeros((n_trials, unit_indices.size, N_TIME), dtype=np.float32)
    for out_unit, raw_unit in enumerate(unit_indices):
        lo = 0 if raw_unit == 0 else int(spike_index[raw_unit - 1])
        hi = int(spike_index[raw_unit])
        spikes = np.asarray(spike_times_ds[lo:hi], dtype=np.float64)
        if not spikes.size:
            continue
        trial_idx = np.searchsorted(window_starts, spikes, side="right") - 1
        possible = (trial_idx >= 0) & (trial_idx < n_trials)
        spikes, trial_idx = spikes[possible], trial_idx[possible]
        rel = spikes - go_times[trial_idx]
        inside = (rel >= OFF_START) & (rel < OFF_END)
        rel, trial_idx = rel[inside], trial_idx[inside]
        bin_idx = np.floor((rel - OFF_START) / BIN_SIZE_S).astype(np.int64)
        good_bin = (bin_idx >= 0) & (bin_idx < N_TIME)
        flat_idx = trial_idx[good_bin] * N_TIME + bin_idx[good_bin]
        counts = np.bincount(flat_idx, minlength=n_trials * N_TIME)
        rates[:, out_unit, :] = counts.reshape(n_trials, N_TIME) / BIN_SIZE_S
    return rates


def map_choice(instructions, outcomes):
    result = np.empty(outcomes.size, dtype=np.int8)
    for i, (instruction_raw, outcome_raw) in enumerate(zip(instructions, outcomes)):
        instruction, outcome = decode_text(instruction_raw), decode_text(outcome_raw)
        if outcome == "ignore":
            result[i] = 2
        elif outcome == "hit":
            result[i] = 0 if instruction == "left" else 1
        elif outcome == "miss":
            result[i] = 1 if instruction == "left" else 0
        else:
            raise ValueError(f"unexpected outcome {outcome!r}")
    return result


def map_outcome(outcomes):
    code = {"ignore": 0, "miss": 1, "hit": 2}
    return np.asarray([code[decode_text(x)] for x in outcomes], dtype=np.int8)


def map_early(early):
    code = {"no early": 0, "early": 1}
    return np.asarray([code[decode_text(x)] for x in early], dtype=np.int8)


@dataclass
class SessionResult:
    neural: list[np.ndarray]
    decoder_input: list[np.ndarray]
    output: list[np.ndarray]
    brain_region_names: list[str]
    subject: str
    info: dict
    plot_payload: dict


def convert_session(path: str) -> SessionResult:
    start_clock = time.perf_counter()
    session_id = session_id_from_path(path)
    with h5py.File(path, "r") as nwb:
        classification = nwb["units/classification"][()]
        unit_indices = np.flatnonzero(classification == b"good").astype(np.int64)
        region_names = [decode_text(x).strip() for x in nwb["units/anno_name"][unit_indices]]
        if any((not x) or x.lower() == "nan" for x in region_names):
            raise ValueError(f"{session_id}: curated unit lacks a CCF annotation")

        good_trial_matrix = np.asarray(nwb["units/is_good_trials"][unit_indices, :], dtype=bool)
        n_ephys_trials = good_trial_matrix.shape[1]
        valid_prefix = np.all(good_trial_matrix, axis=0)

        trials = nwb["intervals/trials"]
        n_trial_table = int(trials["id"].shape[0])
        if n_ephys_trials > n_trial_table:
            raise ValueError(f"{session_id}: neural trial mask longer than trial table")

        # Observation intervals identify which contiguous block of the trial
        # table was recorded.  Most sessions start at trial 0, but some ephys
        # recordings start after behavior or stop before behavior ends.
        obs_index = np.asarray(nwb["units/obs_intervals_index"], dtype=np.int64)
        first_unit = int(unit_indices[0])
        obs_lo = 0 if first_unit == 0 else int(obs_index[first_unit - 1])
        obs_hi = int(obs_index[first_unit])
        obs_intervals = np.asarray(nwb["units/obs_intervals"][obs_lo:obs_hi], dtype=np.float64)
        if obs_intervals.shape[0] != n_ephys_trials:
            raise ValueError(f"{session_id}: observation intervals do not match trial-mask width")
        trial_starts_table = np.asarray(trials["start_time"], dtype=np.float64)
        ephys_trial_idx = np.searchsorted(trial_starts_table, obs_intervals[:, 0])
        if (np.any(ephys_trial_idx >= n_trial_table)
                or not np.allclose(trial_starts_table[ephys_trial_idx], obs_intervals[:, 0])):
            raise ValueError(f"{session_id}: cannot map observation intervals to trial table")
        if np.any(np.diff(ephys_trial_idx) != 1):
            raise ValueError(f"{session_id}: ephys-backed trial rows are not contiguous")

        candidate_trial_idx = ephys_trial_idx[np.flatnonzero(valid_prefix)]
        auto_water = np.asarray(trials["auto_water"])[candidate_trial_idx] != 0
        free_water = np.asarray(trials["free_water"])[candidate_trial_idx] != 0
        water_trial = auto_water | free_water
        selected_trial_idx = candidate_trial_idx[~water_trial]
        if selected_trial_idx.size < 2:
            raise ValueError(f"{session_id}: fewer than two valid non-water trials")

        go_all = np.asarray(nwb["acquisition/BehavioralEvents/go_start_times/timestamps"])
        trial_starts_all = trial_starts_table
        go_times = go_all[selected_trial_idx]
        trial_starts = trial_starts_all[selected_trial_idx]
        centers_global = go_times[:, None] + BIN_CENTERS[None, :]

        sample_starts = np.asarray(nwb["acquisition/BehavioralEvents/sample_start_times/timestamps"])
        first_sample_idx = np.searchsorted(sample_starts, trial_starts, side="left")
        if np.any(first_sample_idx >= sample_starts.size):
            raise ValueError(f"{session_id}: missing sample-start event")
        tone_onsets = sample_starts[first_sample_idx]
        if np.any(tone_onsets >= go_times):
            raise ValueError(f"{session_id}: first sample-start not before go")
        elapsed_from_tone = (centers_global - tone_onsets[:, None]).astype(np.float32)

        stim_starts = np.asarray(nwb["acquisition/BehavioralEvents/photostim_start_times/timestamps"])
        stim_stops = np.asarray(nwb["acquisition/BehavioralEvents/photostim_stop_times/timestamps"])
        stim_on = photostim_state(centers_global, stim_starts, stim_stops)
        inputs = np.stack((elapsed_from_tone, stim_on), axis=1).astype(np.float32)

        tracking = nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
        tracking_ts = np.asarray(tracking["timestamps"], dtype=np.float64)
        tracking_data = np.asarray(tracking["data"], dtype=np.float64)
        clean_y, visible, outlier, q40, q60, tongue_summary = clean_tongue_tracking(tracking_ts, tracking_data)
        tongue_category, tongue_frame_idx = sample_tongue_categories(
            centers_global, tracking_ts, clean_y, visible, q40, q60
        )

        outcomes = np.asarray(trials["outcome"])[selected_trial_idx]
        instructions = np.asarray(trials["trial_instruction"])[selected_trial_idx]
        early = np.asarray(trials["early_lick"])[selected_trial_idx]
        outputs = np.empty((selected_trial_idx.size, 4, N_TIME), dtype=np.int8)
        outputs[:, 0, :] = map_choice(instructions, outcomes)[:, None]
        outputs[:, 1, :] = map_outcome(outcomes)[:, None]
        outputs[:, 2, :] = map_early(early)[:, None]
        outputs[:, 3, :] = tongue_category

        neural_rates = bin_selected_units(
            nwb["units/spike_times"], np.asarray(nwb["units/spike_times_index"]), unit_indices, go_times
        )

        # A small number of NWB trial-mask prefixes extend one trial beyond the
        # actual spike recording.  With hundreds of curated units, a completely
        # zero population over four seconds is a definitive no-coverage marker,
        # not a plausible silent behavioral trial.
        zero_spike_trials = ~np.any(neural_rates > 0, axis=(1, 2))
        n_zero_spike_trials = int(zero_spike_trials.sum())
        if n_zero_spike_trials:
            keep = ~zero_spike_trials
            neural_rates = neural_rates[keep]
            inputs = inputs[keep]
            outputs = outputs[keep]
            stim_on = stim_on[keep]
            tongue_frame_idx = tongue_frame_idx[keep]
            go_times = go_times[keep]
            tone_onsets = tone_onsets[keep]
            selected_trial_idx = selected_trial_idx[keep]
        if selected_trial_idx.size < 2:
            raise ValueError(f"{session_id}: fewer than two trials with neural coverage")

        elapsed = time.perf_counter() - start_clock
        info = {
            "session_id": session_id,
            "source_file": str(Path(path).relative_to(DATA_DIR)),
            "subject": subject_from_path(path),
            "n_raw_units": int(classification.size),
            "n_classifier_good_units": int(unit_indices.size),
            "n_trial_table_rows": n_trial_table,
            "n_ephys_backed_trials": int(n_ephys_trials),
            "ephys_first_source_trial_index_zero_based": int(ephys_trial_idx[0]),
            "ephys_last_source_trial_index_zero_based": int(ephys_trial_idx[-1]),
            "n_invalid_neural_trials_dropped": int((~valid_prefix).sum()),
            "n_auto_free_water_trials_dropped": int(water_trial.sum()),
            "n_all_zero_neural_trials_dropped": n_zero_spike_trials,
            "n_trials_retained": int(selected_trial_idx.size),
            "retained_source_trial_indices_zero_based": selected_trial_idx.astype(int).tolist(),
            "tongue_q40": q40,
            "tongue_q60": q60,
            **tongue_summary,
            "conversion_seconds": elapsed,
        }
        example = min(4, selected_trial_idx.size - 1)
        plot_payload = {
            "session_id": session_id,
            "n_trial_table_rows": n_trial_table,
            "n_ephys_trials": n_ephys_trials,
            "n_retained": int(selected_trial_idx.size),
            "tone_rel_go": tone_onsets - go_times,
            "stim_on": stim_on,
            "inputs": inputs,
            "outputs": outputs,
            "neural": neural_rates,
            "example": example,
            "tracking_ts": tracking_ts,
            "tracking_y_raw": tracking_data[:, 1],
            "tracking_likelihood": tracking_data[:, 2],
            "tracking_y_clean": clean_y,
            "tracking_frame_idx": tongue_frame_idx,
            "go_times": go_times,
            "q40": q40,
            "q60": q60,
        }

    assert neural_rates.shape == (selected_trial_idx.size, unit_indices.size, N_TIME)
    assert inputs.shape == (selected_trial_idx.size, 2, N_TIME)
    assert outputs.shape == (selected_trial_idx.size, 4, N_TIME)
    if not np.all(np.isfinite(neural_rates)) or not np.all(np.isfinite(inputs)):
        raise AssertionError(f"{session_id}: non-finite converted values")
    return SessionResult(
        neural=[neural_rates[i] for i in range(neural_rates.shape[0])],
        decoder_input=[inputs[i] for i in range(inputs.shape[0])],
        output=[outputs[i] for i in range(outputs.shape[0])],
        brain_region_names=region_names,
        subject=subject_from_path(path), info=info, plot_payload=plot_payload,
    )


def plot_processing(payload: dict, output_path: Path) -> None:
    """Plot curation, alignment, cleaning, binning, inputs, and outputs."""
    ex, go = payload["example"], payload["go_times"][payload["example"]]
    ts = payload["tracking_ts"]
    local = (ts >= go + OFF_START) & (ts < go + OFF_END)
    t_rel = ts[local] - go
    fig, axes = plt.subplots(4, 2, figsize=(16, 15), constrained_layout=True)
    ax = axes.ravel()
    ax[0].bar(["trial table", "ephys-backed", "all-unit valid"],
              [payload["n_trial_table_rows"], payload["n_ephys_trials"], payload["n_retained"]],
              color=["0.7", "tab:blue", "tab:green"])
    ax[0].set_title("Trial curation"); ax[0].set_ylabel("trials")
    ax[1].hist(payload["tone_rel_go"], bins=40, color="tab:purple")
    ax[1].axvline(-1.85, color="k", ls="--", label="canonical")
    ax[1].set_title("First tone onset relative to go"); ax[1].set_xlabel("seconds"); ax[1].legend()
    ax[2].plot(t_rel, payload["tracking_y_raw"][local], color="0.7", lw=.7, label="raw y")
    ax[2].plot(t_rel, payload["tracking_y_clean"][local], color="tab:blue", lw=.8, label="clean y")
    ax2 = ax[2].twinx(); ax2.plot(t_rel, payload["tracking_likelihood"][local], color="tab:orange", alpha=.5, lw=.6)
    ax2.axhline(DLC_VISIBLE_THRESHOLD, color="tab:red", ls=":")
    ax[2].axhline(payload["q40"], color="k", ls="--"); ax[2].axhline(payload["q60"], color="k", ls=":")
    ax[2].set_title(f"Tongue cleaning, example trial {ex}"); ax[2].set_xlabel("time from go (s)")
    ax[2].set_ylabel("tongue y (pixels)"); ax2.set_ylabel("DLC likelihood"); ax[2].legend(loc="upper left")
    visible_y = payload["tracking_y_clean"][payload["tracking_likelihood"] >= DLC_VISIBLE_THRESHOLD]
    ax[3].hist(visible_y, bins=100, color="tab:blue", alpha=.75)
    ax[3].axvline(payload["q40"], color="k", ls="--", label="40th")
    ax[3].axvline(payload["q60"], color="k", ls=":", label="60th")
    ax[3].set_title("Session visible-y discretization"); ax[3].legend()
    im = ax[4].imshow(payload["neural"][ex, :min(80, payload["neural"].shape[1]), :], aspect="auto",
                      extent=[OFF_START, OFF_END, min(80, payload["neural"].shape[1]), 0], cmap="viridis")
    ax[4].axvline(0, color="w"); ax[4].set_title("Final 50-ms firing rates"); ax[4].set_xlabel("time from go (s)")
    fig.colorbar(im, ax=ax[4], label="Hz")
    ax[5].plot(BIN_CENTERS, payload["inputs"][ex, 0], label="time from tone")
    ax[5].step(BIN_CENTERS, payload["inputs"][ex, 1], where="mid", label="photostim on")
    ax[5].axvline(0, color="k", ls="--"); ax[5].set_title("Final decoder inputs"); ax[5].legend()
    im2 = ax[6].imshow(payload["outputs"][ex], aspect="auto", interpolation="nearest",
                       extent=[OFF_START, OFF_END, 3.5, -.5], cmap="tab10", vmin=0, vmax=3)
    ax[6].set_yticks(range(4), labels=["choice", "outcome", "early", "tongue"])
    ax[6].axvline(0, color="w"); ax[6].set_title("Final categorical outputs")
    fig.colorbar(im2, ax=ax[6], ticks=[0, 1, 2, 3])
    ax[7].plot(BIN_CENTERS, payload["neural"].mean(axis=(0, 1)), color="tab:green")
    ax7 = ax[7].twinx(); ax7.fill_between(BIN_CENTERS, 0, payload["stim_on"].mean(axis=0), color="tab:orange", alpha=.3)
    ax[7].axvline(0, color="k", ls="--"); ax[7].set_title("Session alignment sanity check")
    ax[7].set_xlabel("time from go (s)"); ax[7].set_ylabel("mean firing rate (Hz)"); ax7.set_ylabel("stim fraction")
    fig.suptitle(payload["session_id"], fontsize=15)
    fig.savefig(output_path, dpi=150); plt.close(fig)


def validate_in_memory(data: dict) -> None:
    ns = len(data["neural"])
    assert ns == len(data["input"]) == len(data["output"])
    assert data["subject_idx"].shape == (ns,) and len(data["brain_region_idx"]) == ns
    for s in range(ns):
        nt = len(data["neural"][s]); assert nt >= 2
        assert nt == len(data["input"][s]) == len(data["output"][s])
        nn = data["neural"][s][0].shape[0]; assert data["brain_region_idx"][s].shape == (nn,)
        for n, x, y in zip(data["neural"][s], data["input"][s], data["output"][s]):
            assert n.shape == (nn, N_TIME) and n.dtype == np.float32
            assert x.shape == (2, N_TIME) and x.dtype == np.float32
            assert y.shape == (4, N_TIME) and np.issubdtype(y.dtype, np.integer)
            assert np.all(np.isfinite(n)) and np.all(np.isfinite(x))
            assert np.all((y[0] >= 0) & (y[0] <= 2)) and np.all((y[1] >= 0) & (y[1] <= 2))
            assert np.all((y[2] >= 0) & (y[2] <= 1)) and np.all((y[3] >= 0) & (y[3] <= 3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="process all usable sessions (default)")
    mode.add_argument("--sample", action="store_true", help="process two sessions")
    parser.add_argument("--show-processing", action="store_true")
    args = parser.parse_args()
    all_files, excluded = discover_usable_files()
    files = all_files[:2] if args.sample else all_files
    print(f"Mode: {'sample' if args.sample else 'full'}; processing {len(files)} of {len(all_files)} usable sessions", flush=True)
    print(f"Excluded during discovery: {excluded}", flush=True)

    subjects = sorted({subject_from_path(path) for path in files})
    region_set = set()
    for path in files:
        with h5py.File(path, "r") as nwb:
            good = nwb["units/classification"][()] == b"good"
            region_set.update(decode_text(x).strip() for x in nwb["units/anno_name"][good])
    brain_regions = sorted(region_set)
    subject_lookup = {name: i for i, name in enumerate(subjects)}
    region_lookup = {name: i for i, name in enumerate(brain_regions)}
    neural, decoder_input, output = [], [], []
    subject_idx, brain_region_idx, session_info = [], [], []
    total_clock = time.perf_counter()
    for number, path in enumerate(files, 1):
        result = convert_session(path)
        neural.append(result.neural); decoder_input.append(result.decoder_input); output.append(result.output)
        subject_idx.append(subject_lookup[result.subject])
        brain_region_idx.append(np.asarray([region_lookup[x] for x in result.brain_region_names], dtype=np.int32))
        session_info.append(result.info)
        print(f"[{number:3d}/{len(files)}] {result.info['session_id']}: "
              f"{result.info['n_classifier_good_units']} neurons, "
              f"{result.info['n_trials_retained']}/{result.info['n_ephys_backed_trials']} trials, "
              f"{result.info['conversion_seconds']:.2f} s", flush=True)
        if args.show_processing and number <= 2:
            plot_path = Path("/app") / f"processing_{result.info['session_id']}.png"
            plot_processing(result.plot_payload, plot_path); print(f"  saved {plot_path}", flush=True)

    data = {
        "neural": neural, "input": decoder_input, "output": output,
        "subjects": subjects, "subject_idx": np.asarray(subject_idx, dtype=np.int32),
        "brain_regions": brain_regions, "brain_region_idx": brain_region_idx,
        "input_names": ["time from tone onset (s)", "photostimulation on"],
        "output_names": ["lick direction choice", "outcome", "early lick", "tongue y-position"],
        "output_values": [
            ["left", "right", "no lick"], ["ignore", "miss", "hit"], ["no", "yes"],
            ["below 40th percentile", "40th to 60th percentile", "above 60th percentile", "not visible"],
        ],
        "metadata": {
            "task_description": "Auditory delayed-response task: mice report a low/high tone by licking right/left after a go cue; early lick, no-response, and photoinhibition trials are retained.",
            "time_bin_size": 50.0, "time_bin_unit": "ms",
            "temporal_alignment_event": "auditory go cue onset", "off_start": OFF_START, "off_end": OFF_END,
            "n_timepoints": int(N_TIME), "time_bin_centers_seconds": BIN_CENTERS.tolist(),
            "neural_measurement": "classifier-good Kilosort2 unit firing rate (Hz)",
            "spike_bin_convention": "left-closed, right-open non-overlapping bins",
            "tone_onset_definition": "first sample_start event within the behavioral trial",
            "photostimulation_definition": "binary state at bin center from NWB start/stop events",
            "tongue_visibility_likelihood_threshold": DLC_VISIBLE_THRESHOLD,
            "tongue_discretization": "per-session 40th/60th percentiles over cleaned visible side-camera tongue-y frames; low-confidence or absent frames are class 3",
            "neuron_curation": "NWB units/classification == 'good' with nonempty Allen CCF annotation",
            "trial_curation": "ephys-backed trial block mapped from unit observation intervals; trials invalid for any retained unit, auto/free-water trials, or all-zero/no-spike-coverage trials removed; early/ignore/photostimulation trials retained",
            "source_dandiset": "DANDI:000363/0.230822.0128",
            "excluded_sessions": excluded if not args.sample else [], "session_info": session_info,
        },
    }
    validate_in_memory(data)
    total_trials = sum(len(x) for x in neural); total_neurons = sum(x[0].shape[0] for x in neural)
    print(f"Validated in memory: {len(neural)} sessions, {total_trials} trials, {total_neurons} session-neurons, {len(brain_regions)} regions", flush=True)
    output_path = Path(args.outpicklefile).resolve(); output_path.parent.mkdir(parents=True, exist_ok=True)
    write_clock = time.perf_counter()
    with output_path.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    write_seconds = time.perf_counter() - write_clock; total_seconds = time.perf_counter() - total_clock
    print(f"Saved {output_path} ({output_path.stat().st_size / 2**30:.3f} GiB) in {write_seconds:.2f} s; total conversion {total_seconds:.2f} s", flush=True)


if __name__ == "__main__":
    main()
