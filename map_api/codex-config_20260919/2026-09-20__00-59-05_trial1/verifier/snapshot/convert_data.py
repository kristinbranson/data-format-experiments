#!/usr/bin/env python3
"""Convert the MAP NWB release to the neural-decoder pickle schema.

All NWB access is through PyNWB. Trials are aligned to go onset, spikes are
converted to rates in 50-ms non-overlapping bins over [-2.5, 1.5), and the
requested task/behavior variables are assembled at the same 80 bin centers.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pynwb import NWBHDF5IO


DATA_DIR = Path("/app/data")
BIN_WIDTH = 0.050
OFF_START = -2.5
OFF_END = 1.5
BIN_EDGES = np.arange(OFF_START, OFF_END + BIN_WIDTH / 2, BIN_WIDTH)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
N_TIME = len(BIN_CENTERS)
DLC_VISIBLE_THRESHOLD = 0.9

INPUT_NAMES = ["time from tone onset (s)", "photostimulation on"]
OUTPUT_NAMES = [
    "lick direction choice", "outcome", "early lick", "tongue y-position"
]
OUTPUT_VALUES = [
    ["left", "right", "no lick"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    [
        "below 40th percentile",
        "40th to 60th percentile",
        "above 60th percentile",
        "not visible",
    ],
]


def _events(nwb, name: str) -> np.ndarray:
    """Return an event timestamp vector without bypassing PyNWB."""
    series = nwb.acquisition["BehavioralEvents"].time_series[name]
    return np.asarray(series.timestamps[:], dtype=np.float64)


def _events_by_trial(
    timestamps: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
) -> list[np.ndarray]:
    """Associate absolute event timestamps with half-open trial intervals."""
    left = np.searchsorted(timestamps, starts, side="left")
    right = np.searchsorted(timestamps, stops, side="right")
    return [timestamps[a:b] for a, b in zip(left, right)]


def _classify_choice(instruction: str, outcome: str) -> int:
    """Map task-defined response to left=0, right=1, no lick=2."""
    if outcome == "ignore":
        return 2
    if outcome == "hit":
        return 0 if instruction == "left" else 1
    if outcome == "miss":
        return 1 if instruction == "left" else 0
    raise ValueError(f"Unexpected outcome {outcome!r}")


def _nearest_indices(source_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """Indices of temporally nearest source samples for sorted query times."""
    idx = np.searchsorted(source_times, query_times, side="left")
    idx = np.clip(idx, 1, len(source_times) - 1)
    prev = idx - 1
    use_prev = query_times - source_times[prev] <= source_times[idx] - query_times
    return np.where(use_prev, prev, idx)


def _unit_spike_times(units, unit_index: int) -> np.ndarray:
    """Read one ragged unit spike vector via the PyNWB DynamicTable API."""
    return np.asarray(units["spike_times"][unit_index], dtype=np.float64)


def _any_observation_mask(units, good_indices: np.ndarray,
                          starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Trials whose requested interval is observed for at least one kept unit.

    The reference concatenates classifier-good units across probes even when their
    manually accepted observation periods differ, so requiring every unit would
    discard valid neural population data. We reject only periods outside all kept
    units, which otherwise become entirely zero trials.
    """
    valid = np.zeros(len(starts), dtype=bool)
    for unit_idx in good_indices:
        intervals = np.asarray(units["obs_intervals"][int(unit_idx)], dtype=np.float64)
        intervals = np.atleast_2d(intervals)
        covered = np.zeros(len(starts), dtype=bool)
        for onset, offset in intervals:
            # The final alignment check below uses go-based extraction boundaries;
            # this first pass cheaply rejects trials wholly outside recording time.
            covered |= (starts >= onset) & (stops <= offset)
        valid |= covered
    return valid


def _bin_spikes(
    units,
    good_indices: np.ndarray,
    go_times: np.ndarray,
) -> np.ndarray:
    """Vectorized per-unit spike binning; result is trials x neurons x time."""
    absolute_edges = go_times[:, None] + BIN_EDGES[None, :]
    # Flattened edges are sorted for this task (successive go cues are >4 s apart),
    # allowing one searchsorted call per neuron rather than one per trial.
    flattened = absolute_edges.ravel()
    monotonic = np.all(flattened[1:] >= flattened[:-1])
    rates = np.empty((len(go_times), len(good_indices), N_TIME), dtype=np.float32)
    for out_idx, unit_idx in enumerate(good_indices):
        spikes = _unit_spike_times(units, int(unit_idx))
        if monotonic:
            positions = np.searchsorted(spikes, flattened, side="left")
            counts = np.diff(positions.reshape(len(go_times), N_TIME + 1), axis=1)
        else:
            counts = np.stack([
                np.diff(np.searchsorted(spikes, row, side="left"))
                for row in absolute_edges
            ])
        rates[:, out_idx, :] = counts.astype(np.float32) / BIN_WIDTH
    return rates


def _safe_session_name(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def _plot_processing(
    path: Path,
    go_times: np.ndarray,
    tone_times: np.ndarray,
    inputs: np.ndarray,
    outputs: np.ndarray,
    rates: np.ndarray,
    tongue_times: np.ndarray,
    tongue_y: np.ndarray,
    tongue_likelihood: np.ndarray,
    tongue_thresholds: tuple[float, float],
) -> None:
    """Plot the alignment, binning, inputs, and tongue discretization."""
    trial = min(2, len(go_times) - 1)
    rel_video = tongue_times - go_times[trial]
    video_mask = (rel_video >= OFF_START) & (rel_video <= OFF_END)
    mean_rate = rates[trial].mean(axis=0)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=False)
    ax = axes.ravel()
    ax[0].imshow(
        rates[trial, : min(80, rates.shape[1])], aspect="auto", origin="lower",
        extent=[OFF_START, OFF_END, 0, min(80, rates.shape[1])], cmap="magma"
    )
    ax[0].axvline(0, color="cyan", lw=1)
    ax[0].set(title="50-ms go-aligned firing rates", ylabel="good unit")

    ax[1].plot(BIN_CENTERS, mean_rate, color="black")
    ax[1].axvline(0, color="tab:red", label="go")
    ax[1].axvline(tone_times[trial] - go_times[trial], color="tab:blue", label="tone")
    ax[1].set(title="Population mean and event alignment", ylabel="rate (Hz)")
    ax[1].legend()

    ax[2].plot(BIN_CENTERS, inputs[trial, 0], label="time since tone")
    ax[2].step(BIN_CENTERS, inputs[trial, 1], where="mid", label="laser on")
    ax[2].set(title="Decoder inputs", ylabel="value")
    ax[2].legend()

    ax[3].plot(rel_video[video_mask], tongue_y[video_mask], lw=.5, color="0.5")
    visible = video_mask & (tongue_likelihood >= DLC_VISIBLE_THRESHOLD)
    ax[3].scatter(rel_video[visible], tongue_y[visible], s=2, color="tab:blue")
    for threshold in tongue_thresholds:
        ax[3].axhline(threshold, color="tab:orange", ls="--")
    ax[3].set(title="Raw tongue y, visible frames, 40/60% thresholds", ylabel="pixels")

    ax[4].plot(rel_video[video_mask], tongue_likelihood[video_mask], lw=.5)
    ax[4].axhline(DLC_VISIBLE_THRESHOLD, color="tab:red", ls="--")
    ax[4].set(title="DLC tongue visibility", ylabel="likelihood", ylim=(-.05, 1.05))

    ax[5].step(BIN_CENTERS, outputs[trial, 3], where="mid", label="tongue class")
    ax[5].set(
        title=(f"Output classes; choice={outputs[trial,0,0]}, "
               f"outcome={outputs[trial,1,0]}, early={outputs[trial,2,0]}"),
        ylabel="class", yticks=[0, 1, 2, 3]
    )
    for a in ax:
        a.set_xlabel("time from go cue (s)")
        a.grid(alpha=.15)
    fig.suptitle(path.name)
    fig.tight_layout()
    fig.savefig(Path("/app") / f"processing_{_safe_session_name(path)}.png", dpi=150)
    plt.close(fig)


def convert_session(path: Path, show_processing: bool = False) -> dict | None:
    """Convert one NWB file, returning None when it has no classifier-good units."""
    started = time.perf_counter()
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        units = nwb.units
        classifications = np.asarray(units["classification"][:]).astype(str)
        good_indices = np.flatnonzero(classifications == "good")
        if len(good_indices) == 0:
            print(f"SKIP {path.name}: no classifier-good units", flush=True)
            return None
        annotations_all = np.asarray(units["anno_name"][:]).astype(str)
        annotations = annotations_all[good_indices]
        if np.any(annotations == ""):
            raise ValueError(f"{path.name}: classifier-good unit lacks CCF annotation")

        trials_all = nwb.trials.to_dataframe()
        starts_all = trials_all["start_time"].to_numpy(np.float64)
        stops_all = trials_all["stop_time"].to_numpy(np.float64)
        trials = trials_all.copy()
        starts = trials["start_time"].to_numpy(np.float64)
        stops = trials["stop_time"].to_numpy(np.float64)
        if len(trials) < 2:
            print(f"SKIP {path.name}: fewer than two commonly observed trials", flush=True)
            return None
        go_by_trial = _events_by_trial(_events(nwb, "go_start_times"), starts, stops)
        sample_by_trial = _events_by_trial(_events(nwb, "sample_start_times"), starts, stops)
        if not all(len(x) == 1 for x in go_by_trial):
            bad = [i for i, x in enumerate(go_by_trial) if len(x) != 1]
            raise ValueError(f"{path.name}: trials without exactly one go event: {bad[:10]}")
        if not all(len(x) >= 1 for x in sample_by_trial):
            bad = [i for i, x in enumerate(sample_by_trial) if len(x) < 1]
            raise ValueError(f"{path.name}: trials without a sample/tone onset: {bad[:10]}")
        go_times = np.asarray([x[0] for x in go_by_trial])
        # Last onset is the successfully completed replay on early-lick trials.
        tone_times = np.asarray([x[x <= g][-1] for x, g in zip(sample_by_trial, go_times)])

        stim_starts_by_trial = _events_by_trial(
            _events(nwb, "photostim_start_times"), starts, stops
        )
        stim_stops_by_trial = _events_by_trial(
            _events(nwb, "photostim_stop_times"), starts, stops
        )
        for i, (on, off) in enumerate(zip(stim_starts_by_trial, stim_stops_by_trial)):
            if len(on) != len(off):
                raise ValueError(f"{path.name}: unpaired laser events in trial {i}")

        absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
        inputs = np.empty((len(trials), 2, N_TIME), dtype=np.float32)
        inputs[:, 0, :] = (absolute_centers - tone_times[:, None]).astype(np.float32)
        inputs[:, 1, :] = 0
        for i, (onsets, offsets) in enumerate(zip(stim_starts_by_trial, stim_stops_by_trial)):
            for onset, offset in zip(onsets, offsets):
                active = ((absolute_centers[i] >= onset) &
                          (absolute_centers[i] < offset))
                inputs[i, 1, active] = 1.0

        behavior = nwb.acquisition["BehavioralTimeSeries"]
        tongue = behavior.time_series["Camera0_side_TongueTracking"]
        tongue_times = np.asarray(tongue.timestamps[:], dtype=np.float64)
        tongue_data = np.asarray(tongue.data[:], dtype=np.float64)
        tongue_y = tongue_data[:, 1]
        tongue_likelihood = tongue_data[:, 2]
        session_visible = np.isfinite(tongue_y) & np.isfinite(tongue_likelihood) & (
            tongue_likelihood >= DLC_VISIBLE_THRESHOLD
        )
        if session_visible.sum() < 2:
            raise ValueError(f"{path.name}: insufficient visible tongue samples")
        q40, q60 = np.quantile(tongue_y[session_visible], [0.4, 0.6])
        nearest = _nearest_indices(tongue_times, absolute_centers.ravel()).reshape(
            len(trials), N_TIME
        )
        matched_y = tongue_y[nearest]
        matched_likelihood = tongue_likelihood[nearest]
        visible = np.isfinite(matched_y) & np.isfinite(matched_likelihood) & (
            matched_likelihood >= DLC_VISIBLE_THRESHOLD
        )
        tongue_class = np.full((len(trials), N_TIME), 3, dtype=np.int8)
        tongue_class[visible & (matched_y < q40)] = 0
        tongue_class[visible & (matched_y >= q40) & (matched_y <= q60)] = 1
        tongue_class[visible & (matched_y > q60)] = 2

        outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
        early_map = {"no early": 0, "early": 1}
        outcomes_text = trials["outcome"].astype(str).to_numpy()
        instructions = trials["trial_instruction"].astype(str).to_numpy()
        early_text = trials["early_lick"].astype(str).to_numpy()
        choice = np.asarray([
            _classify_choice(inst, outcome)
            for inst, outcome in zip(instructions, outcomes_text)
        ], dtype=np.int8)
        outcome = np.asarray([outcome_map[x] for x in outcomes_text], dtype=np.int8)
        early = np.asarray([early_map[x] for x in early_text], dtype=np.int8)
        outputs = np.empty((len(trials), 4, N_TIME), dtype=np.int8)
        outputs[:, 0, :] = choice[:, None]
        outputs[:, 1, :] = outcome[:, None]
        outputs[:, 2, :] = early[:, None]
        outputs[:, 3, :] = tongue_class

        rates = _bin_spikes(units, good_indices, go_times)
        if not np.isfinite(rates).all() or np.any(rates < 0):
            raise ValueError(f"{path.name}: invalid firing rates")
        if not np.allclose(rates / (1 / BIN_WIDTH), np.round(rates / (1 / BIN_WIDTH))):
            raise ValueError(f"{path.name}: rates are not integer-count multiples")

        # NWBs can include behavioral trials after every ephys probe stopped. The
        # reference bins all classifier-good units without applying per-unit manual
        # observation masks; preserve that logic, but discard windows with no spike
        # from any retained unit because they contain no neural recording at all.
        neural_present = np.any(rates != 0, axis=(1, 2))
        n_excluded_no_neural = int((~neural_present).sum())
        if n_excluded_no_neural:
            trials = trials.loc[neural_present].copy()
            go_times = go_times[neural_present]
            tone_times = tone_times[neural_present]
            inputs = inputs[neural_present]
            outputs = outputs[neural_present]
            rates = rates[neural_present]
        if len(trials) < 2:
            print(f"SKIP {path.name}: fewer than two trials with neural data", flush=True)
            return None

        if show_processing:
            _plot_processing(
                path, go_times, tone_times, inputs, outputs, rates, tongue_times,
                tongue_y, tongue_likelihood, (float(q40), float(q60))
            )

        result = {
            "path": str(path),
            "session_id": nwb.identifier,
            "subject": str(nwb.subject.subject_id),
            "neural": [rates[i] for i in range(len(trials))],
            "input": [inputs[i] for i in range(len(trials))],
            "output": [outputs[i] for i in range(len(trials))],
            "annotations": annotations.tolist(),
            "info": {
                "n_trials_nwb": len(trials_all),
                "n_trials": len(trials),
                "trials_excluded_no_population_spikes": n_excluded_no_neural,
                "n_neurons": len(good_indices),
                "tone_to_go_s": [float(np.min(go_times - tone_times)),
                                  float(np.max(go_times - tone_times))],
                "tongue_y_q40": float(q40),
                "tongue_y_q60": float(q60),
                "tongue_visible_fraction_at_bins": float(visible.mean()),
                "photostim_trial_count": int(sum(len(x) > 0 for x in stim_starts_by_trial)),
                "auto_water_trial_count": int(trials["auto_water"].sum()),
                "free_water_trial_count": int(trials["free_water"].sum()),
                "choice_counts": np.bincount(choice, minlength=3).tolist(),
                "outcome_counts": np.bincount(outcome, minlength=3).tolist(),
                "early_lick_counts": np.bincount(early, minlength=2).tolist(),
                "tongue_class_counts": np.bincount(tongue_class.ravel(), minlength=4).tolist(),
            },
        }
    elapsed = time.perf_counter() - started
    print(
        f"DONE {path.name}: {result['info']['n_trials']} trials, "
        f"{result['info']['n_neurons']} neurons, {elapsed:.2f} s", flush=True
    )
    return result


def validate_converted(data: dict) -> None:
    """Fast structural and semantic checks before writing the pickle."""
    ns = len(data["neural"])
    assert ns == len(data["input"]) == len(data["output"])
    assert ns == len(data["subject_idx"]) == len(data["brain_region_idx"])
    assert len(data["metadata"]["session_info"]) == ns
    for s in range(ns):
        nt = len(data["neural"][s])
        assert nt >= 2 and nt == len(data["input"][s]) == len(data["output"][s])
        nn = len(data["brain_region_idx"][s])
        for neural, inp, out in zip(data["neural"][s], data["input"][s], data["output"][s]):
            assert neural.shape == (nn, N_TIME)
            assert inp.shape == (2, N_TIME)
            assert out.shape == (4, N_TIME)
            assert neural.dtype == np.float32 and inp.dtype == np.float32
            assert np.issubdtype(out.dtype, np.integer)
            assert np.all((inp[1] == 0) | (inp[1] == 1))
            assert np.all(np.diff(inp[0]) > 0)
            assert np.all((out[0] >= 0) & (out[0] <= 2))
            assert np.all((out[1] >= 0) & (out[1] <= 2))
            assert np.all((out[2] >= 0) & (out[2] <= 1))
            assert np.all((out[3] >= 0) & (out[3] <= 3))


def build_dataset(files: list[Path], show_processing: bool) -> dict:
    converted = []
    for path in files:
        session = convert_session(path, show_processing=show_processing and len(converted) < 2)
        if session is not None:
            converted.append(session)

    subjects = sorted({s["subject"] for s in converted})
    subject_lookup = {name: i for i, name in enumerate(subjects)}
    brain_regions = sorted({region for s in converted for region in s["annotations"]})
    region_lookup = {name: i for i, name in enumerate(brain_regions)}
    session_info = []
    for s in converted:
        info = dict(s["info"])
        info.update(path=s["path"], session_id=s["session_id"], subject=s["subject"])
        session_info.append(info)

    data = {
        "neural": [s["neural"] for s in converted],
        "input": [s["input"] for s in converted],
        "output": [s["output"] for s in converted],
        "subjects": subjects,
        "subject_idx": np.asarray([subject_lookup[s["subject"]] for s in converted], dtype=np.int16),
        "brain_regions": brain_regions,
        "brain_region_idx": [
            np.asarray([region_lookup[x] for x in s["annotations"]], dtype=np.int16)
            for s in converted
        ],
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "output_values": OUTPUT_VALUES,
        "metadata": {
            "task_description": (
                "Auditory delayed-response task; decode lick choice, outcome, early lick, "
                "and discretized side-camera tongue y-position from go-aligned neural activity."
            ),
            "time_bin_size": 50.0,
            "time_bin_units": "ms",
            "temporal_alignment_event": "auditory Go cue onset",
            "off_start": OFF_START,
            "off_end": OFF_END,
            "n_timepoints": N_TIME,
            "bin_edges_relative_to_go_s": BIN_EDGES.tolist(),
            "bin_centers_relative_to_go_s": BIN_CENTERS.tolist(),
            "neural_measure": "firing rate (Hz), spike counts / 0.05 s",
            "bin_interval_convention": "half-open [left, right)",
            "unit_filter": "NWB units.classification == 'good' (regional classifier QC)",
            "trial_filter": (
                "all trials with one go cue and required streams, excluding only windows "
                "with zero spikes across the entire classifier-good population"
            ),
            "choice_definition": "ignore=no lick; hit=instructed side; miss=opposite side",
            "tone_onset_definition": "last sample_start event before go within trial",
            "photostim_definition": "bin center in [photostim_start, photostim_stop)",
            "tongue_tracking_series": "Camera0_side_TongueTracking",
            "tongue_visibility_likelihood_threshold": DLC_VISIBLE_THRESHOLD,
            "tongue_percentile_population": "all finite session frames with likelihood >= 0.9",
            "tongue_frame_matching": "nearest timestamp to each neural bin center",
            "source": "DANDI:000363/0.230822.0128",
            "session_info": session_info,
        },
    }
    validate_converted(data)
    return data


def print_summary(data: dict) -> None:
    trials = sum(len(x) for x in data["neural"])
    neurons = sum(x[0].shape[0] for x in data["neural"])
    print(f"SUMMARY sessions={len(data['neural'])} subjects={len(data['subjects'])} "
          f"session-neurons={neurons} trials={trials} regions={len(data['brain_regions'])}")
    for i, name in enumerate(OUTPUT_NAMES):
        counts = Counter()
        for session in data["output"]:
            for trial in session:
                if i < 3:
                    counts[int(trial[i, 0])] += 1
                else:
                    counts.update(map(int, trial[i]))
        print(f"OUTPUT {name}: {dict(sorted(counts.items()))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="process all sessions (default)")
    mode.add_argument("--sample", action="store_true", help="process first two usable sessions")
    parser.add_argument("--show-processing", action="store_true")
    args = parser.parse_args()

    files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
    if not files:
        raise FileNotFoundError(f"No NWB files under {DATA_DIR}")
    if args.sample:
        # First two files are known to contain classifier-good units; conversion still
        # handles and skips zero-good sessions in general.
        files = files[:2]
    print(f"MODE {'sample' if args.sample else 'full'}: {len(files)} NWB files", flush=True)
    started = time.perf_counter()
    data = build_dataset(files, args.show_processing)
    print_summary(data)
    with args.outpicklefile.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed = time.perf_counter() - started
    size_gib = args.outpicklefile.stat().st_size / 1024**3
    print(f"WROTE {args.outpicklefile} ({size_gib:.3f} GiB) in {elapsed:.2f} s", flush=True)


if __name__ == "__main__":
    main()
