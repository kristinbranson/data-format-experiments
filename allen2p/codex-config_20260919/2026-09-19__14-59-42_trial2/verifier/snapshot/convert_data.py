#!/usr/bin/env python3
"""Convert Allen VisualBehavior ophys NWB files to decoder-ready trials.

Usage
-----
python -u /app/convert_data.py OUTPUT.pkl [--full | --sample] [--show-processing]

The released L0 calcium-event trace is the neural signal. All behavior and
stimulus variables are evaluated on its ophys timestamp grid. Trials use the
SDK's half-open [start_time, stop_time) bounds and retain go/catch trials only.
"""

from __future__ import annotations

import argparse
import pickle
import time
from collections import Counter
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


APP_DIR = Path("/app")
DATA_ROOT = APP_DIR / "data" / "visual-behavior-ophys-1.1.0"
EXPERIMENT_DIR = DATA_ROOT / "behavior_ophys_experiments"
EXPERIMENT_TABLE = DATA_ROOT / "project_metadata" / "ophys_experiment_table.csv"

PROJECT_CODE = "VisualBehavior"
SAMPLE_EXPERIMENT_IDS = (844395446, 845037476)  # same mouse; image sets A/B; all outcomes

EVENT_ROOT = "processing/ophys/event_detection"
CELL_ROOT = "processing/ophys/image_segmentation/cell_specimen_table"
RUN_ROOT = "processing/running/speed"
EYE_ROOT = "acquisition/EyeTracking"
TRIAL_ROOT = "intervals/trials"

OUTCOME_COLUMNS = ("hit", "miss", "false_alarm", "correct_reject")
OUTPUT_NAMES = (
    "image_identity",
    "image_change",
    "running_speed_bin",
    "pupil_diameter_bin",
    "trial_outcome",
)
QUINTILE_NAMES = (
    "0-20th percentile",
    "20-40th percentile",
    "40-60th percentile",
    "60-80th percentile",
    "80-100th percentile",
)


def _decode_strings(values: np.ndarray) -> list[str]:
    """Decode an HDF5 byte/object string vector."""
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def _file_map() -> dict[int, Path]:
    return {
        int(path.stem.rsplit("_", 1)[1]): path
        for path in EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb")
    }


def select_experiments(sample: bool) -> tuple[pd.DataFrame, list[dict]]:
    """Select active, published VisualBehavior experiments with eye data."""
    table = pd.read_csv(EXPERIMENT_TABLE)
    files = _file_map()
    selected = table[
        (table["project_code"] == PROJECT_CODE)
        & (table["behavior_type"] == "active_behavior")
        & (~table["passive"].astype(bool))
        & (table["ophys_experiment_id"].isin(files))
    ].copy()
    selected = selected.sort_values("ophys_experiment_id")

    keep_rows = []
    excluded = []
    for row in selected.itertuples(index=False):
        experiment_id = int(row.ophys_experiment_id)
        path = files[experiment_id]
        with h5py.File(path, "r") as nwb:
            has_eye = f"{EYE_ROOT}/pupil_tracking/area" in nwb
        if has_eye:
            keep_rows.append(experiment_id)
        else:
            excluded.append(
                {
                    "ophys_experiment_id": experiment_id,
                    "mouse_id": str(row.mouse_id),
                    "reason": "missing eye-tracking stream required for pupil output",
                }
            )

    selected = selected[selected["ophys_experiment_id"].isin(keep_rows)].copy()
    selected["nwb_path"] = selected["ophys_experiment_id"].map(files)
    if sample:
        selected = selected[
            selected["ophys_experiment_id"].isin(SAMPLE_EXPERIMENT_IDS)
        ].copy()
        found = set(selected["ophys_experiment_id"].astype(int))
        missing = set(SAMPLE_EXPERIMENT_IDS) - found
        if missing:
            raise RuntimeError(f"Sample experiment(s) unavailable after curation: {sorted(missing)}")
        selected = selected.set_index("ophys_experiment_id").loc[list(SAMPLE_EXPERIMENT_IDS)].reset_index()

    if selected.empty:
        raise RuntimeError("No experiments passed selection")
    return selected.reset_index(drop=True), excluded


def _image_presentation_groups(nwb: h5py.File) -> list[h5py.Group]:
    """Return task image interval tables (excluding movie/spontaneous blocks)."""
    groups = []
    for name, group in nwb["intervals"].items():
        if not isinstance(group, h5py.Group) or not name.endswith("_presentations"):
            continue
        if "image_name" not in group or "active" not in group:
            continue
        active = np.asarray(group["active"][:], dtype=bool)
        if active.any():
            groups.append(group)
    if not groups:
        raise RuntimeError(f"No active image-presentation group in {nwb.filename}")
    return groups


def collect_image_names(selected: pd.DataFrame) -> list[str]:
    """Collect the global image vocabulary from selected source files."""
    names: set[str] = set()
    for path in selected["nwb_path"]:
        with h5py.File(path, "r") as nwb:
            for group in _image_presentation_groups(nwb):
                active = np.asarray(group["active"][:], dtype=bool)
                for name in np.asarray(_decode_strings(group["image_name"][:]))[active]:
                    if name and name != "omitted":
                        names.add(str(name))
    if len(names) != 16:
        raise RuntimeError(f"Expected 16 natural-image identities across selection, found {len(names)}: {sorted(names)}")
    return sorted(names)


def _interp_finite(target_t: np.ndarray, source_t: np.ndarray, source_x: np.ndarray) -> np.ndarray:
    """Linearly interpolate finite source samples, rejecting unusable streams."""
    valid = np.isfinite(source_t) & np.isfinite(source_x)
    if valid.sum() < 2:
        raise RuntimeError("Fewer than two finite samples in behavioral stream")
    source_t = np.asarray(source_t[valid], dtype=np.float64)
    source_x = np.asarray(source_x[valid], dtype=np.float64)
    order = np.argsort(source_t, kind="stable")
    source_t, source_x = source_t[order], source_x[order]
    unique = np.r_[True, np.diff(source_t) > 0]
    return np.interp(target_t, source_t[unique], source_x[unique]).astype(np.float32)


def _nan_gap_stats(timestamps: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Return missing fraction and longest consecutive missing duration."""
    bad = ~np.isfinite(values)
    missing_fraction = float(bad.mean())
    padded = np.r_[False, bad, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    if not len(edges):
        return missing_fraction, 0.0
    lengths = edges[1::2] - edges[::2]
    dt = float(np.median(np.diff(timestamps)))
    return missing_fraction, float(lengths.max() * dt)


def _quantile_bins(values: np.ndarray, used_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Discretize a continuous session stream using retained-sample quintiles."""
    retained = np.asarray(values[used_indices], dtype=np.float64)
    if not np.isfinite(retained).all():
        raise RuntimeError("Non-finite retained continuous values before discretization")
    thresholds = np.percentile(retained, [20, 40, 60, 80])
    if np.any(np.diff(thresholds) <= 0):
        raise RuntimeError(f"Non-distinct quintile thresholds: {thresholds}")
    bins = np.digitize(values, thresholds, right=False).astype(np.int16)
    return bins, thresholds.astype(np.float64)


def _trial_table(nwb: h5py.File) -> dict[str, np.ndarray]:
    group = nwb[TRIAL_ROOT]
    needed = (
        "id",
        "start_time",
        "stop_time",
        "change_time",
        "go",
        "catch",
        "aborted",
        "auto_rewarded",
        *OUTCOME_COLUMNS,
    )
    return {name: np.asarray(group[name][:]) for name in needed}


def _trial_specs(trials: dict[str, np.ndarray], ophys_t: np.ndarray) -> list[dict]:
    keep = (
        (trials["go"].astype(bool) | trials["catch"].astype(bool))
        & ~trials["aborted"].astype(bool)
        & ~trials["auto_rewarded"].astype(bool)
    )
    specs = []
    for raw_idx in np.flatnonzero(keep):
        outcome_flags = np.array(
            [bool(trials[name][raw_idx]) for name in OUTCOME_COLUMNS], dtype=bool
        )
        if outcome_flags.sum() != 1:
            raise RuntimeError(
                f"Trial {int(trials['id'][raw_idx])} has {outcome_flags.sum()} outcome flags"
            )
        start = float(trials["start_time"][raw_idx])
        stop = float(trials["stop_time"][raw_idx])
        lo = int(np.searchsorted(ophys_t, start, side="left"))
        hi = int(np.searchsorted(ophys_t, stop, side="left"))
        if hi - lo < 2:
            raise RuntimeError(f"Trial {int(trials['id'][raw_idx])} has only {hi-lo} ophys frames")
        if not (ophys_t[lo] >= start and ophys_t[hi - 1] < stop):
            raise RuntimeError("Half-open trial boundary invariant failed")
        specs.append(
            {
                "raw_index": int(raw_idx),
                "trial_id": int(trials["id"][raw_idx]),
                "lo": lo,
                "hi": hi,
                "start_time": start,
                "stop_time": stop,
                "change_time": float(trials["change_time"][raw_idx]),
                "go": bool(trials["go"][raw_idx]),
                "catch": bool(trials["catch"][raw_idx]),
                "outcome": int(np.flatnonzero(outcome_flags)[0]),
            }
        )
    return specs


def _stimulus_on_ophys_grid(
    nwb: h5py.File, ophys_t: np.ndarray, image_to_id: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, int]:
    identity = np.zeros(len(ophys_t), dtype=np.int16)  # 0 is gray
    change = np.zeros(len(ophys_t), dtype=np.int16)
    n_presentations = 0
    for group in _image_presentation_groups(nwb):
        names = _decode_strings(group["image_name"][:])
        starts = np.asarray(group["start_time"][:], dtype=float)
        stops = np.asarray(group["stop_time"][:], dtype=float)
        active = np.asarray(group["active"][:], dtype=bool)
        is_change = np.asarray(group["is_change"][:], dtype=float)
        for name, start, stop, is_active, changed in zip(
            names, starts, stops, active, is_change
        ):
            if not is_active:
                continue
            n_presentations += 1
            lo = int(np.searchsorted(ophys_t, start, side="left"))
            hi = int(np.searchsorted(ophys_t, stop, side="left"))
            lo, hi = max(0, lo), min(len(ophys_t), hi)
            if hi <= lo or name == "omitted":
                continue
            if name not in image_to_id:
                raise RuntimeError(f"Unknown image name {name!r}")
            identity[lo:hi] = image_to_id[name]
            if np.isfinite(changed) and changed > 0.5:
                change[lo:hi] = 1
    return identity, change, n_presentations


def _plot_processing(
    experiment_id: int,
    neural_trial: np.ndarray,
    output_trial: np.ndarray,
    trial_t: np.ndarray,
    trial_spec: dict,
    run_source_t: np.ndarray,
    run_source_x: np.ndarray,
    run_interp: np.ndarray,
    run_thresholds: np.ndarray,
    eye_source_t: np.ndarray,
    pupil_source: np.ndarray,
    pupil_interp: np.ndarray,
    pupil_thresholds: np.ndarray,
    outcome_counts: Counter,
    ophys_dt: np.ndarray,
) -> None:
    """Save an eight-panel visual audit of the conversion stages."""
    rel_t = trial_t - trial_spec["start_time"]
    fig, axes = plt.subplots(4, 2, figsize=(18, 16))
    axes = axes.ravel()

    show_n = min(60, neural_trial.shape[0])
    vmax = float(np.percentile(neural_trial[:show_n], 99.5))
    axes[0].imshow(
        neural_trial[:show_n], aspect="auto", interpolation="nearest",
        extent=[rel_t[0], rel_t[-1], show_n, 0], vmin=0,
        vmax=max(vmax, np.finfo(float).eps), cmap="magma"
    )
    axes[0].axvline(trial_spec["change_time"] - trial_spec["start_time"], color="cyan")
    axes[0].set(title="Raw released L0 events, trial slice", ylabel="Neuron", xlabel="s from trial start")

    axes[1].plot(rel_t, neural_trial.mean(axis=0), color="black", label="mean event")
    axes[1].step(rel_t, output_trial[0] / max(1, output_trial[0].max()), where="post", label="image ID (scaled)")
    axes[1].step(rel_t, output_trial[1], where="post", label="change")
    axes[1].axvline(trial_spec["change_time"] - trial_spec["start_time"], color="tab:red", ls="--")
    axes[1].set(title="Neural/stimulus/change alignment", xlabel="s from trial start")
    axes[1].legend(fontsize=8)

    rmask = (run_source_t >= trial_spec["start_time"]) & (run_source_t < trial_spec["stop_time"])
    axes[2].plot(run_source_t[rmask] - trial_spec["start_time"], run_source_x[rmask], ".", ms=2, alpha=.4, label="released samples")
    axes[2].plot(rel_t, run_interp[trial_spec["lo"]:trial_spec["hi"]], lw=1, label="on ophys grid")
    for q in run_thresholds:
        axes[2].axhline(q, color="gray", lw=.5, ls=":")
    axes[2].set(title="Running interpolation + quintile thresholds", xlabel="s from trial start", ylabel="cm/s")
    axes[2].legend(fontsize=8)

    emask = (eye_source_t >= trial_spec["start_time"]) & (eye_source_t < trial_spec["stop_time"])
    axes[3].plot(eye_source_t[emask] - trial_spec["start_time"], pupil_source[emask], ".", ms=2, alpha=.5, label="processed diameter")
    axes[3].plot(rel_t, pupil_interp[trial_spec["lo"]:trial_spec["hi"]], lw=1, label="finite interpolation")
    for q in pupil_thresholds:
        axes[3].axhline(q, color="gray", lw=.5, ls=":")
    axes[3].set(title="Blink-masked pupil + interpolation", xlabel="s from trial start", ylabel="diameter (pixels)")
    axes[3].legend(fontsize=8)

    axes[4].step(rel_t, output_trial[0], where="post", label="image identity")
    axes[4].step(rel_t, output_trial[1] * 17, where="post", label="change x17")
    axes[4].step(rel_t, output_trial[2] + 18, where="post", label="running bin +18")
    axes[4].step(rel_t, output_trial[3] + 24, where="post", label="pupil bin +24")
    axes[4].step(rel_t, output_trial[4] + 30, where="post", label="outcome +30")
    axes[4].set(title="Final categorical output rows", xlabel="s from trial start")
    axes[4].legend(fontsize=7, ncol=2)

    labels = [name.replace("_", "\n") for name in OUTCOME_COLUMNS]
    axes[5].bar(labels, [outcome_counts[i] for i in range(4)])
    axes[5].set(title="Retained source trial outcomes", ylabel="Trials")

    axes[6].hist(run_interp, bins=100, color="tab:blue", alpha=.7)
    for q in run_thresholds:
        axes[6].axvline(q, color="black", lw=.7)
    axes[6].set(title="Session running distribution/quintiles", xlabel="cm/s")

    axes[7].hist(ophys_dt * 1000, bins=80, color="tab:green", alpha=.7)
    axes[7].axvline(np.median(ophys_dt) * 1000, color="black", lw=1)
    axes[7].set(title="Ophys frame-interval audit", xlabel="Frame interval (ms)")

    fig.suptitle(f"VisualBehavior processing audit — experiment {experiment_id}")
    fig.tight_layout()
    out = APP_DIR / f"processing_{experiment_id}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  saved {out}")


def convert_session(
    row: pd.Series,
    image_to_id: dict[str, int],
    show_processing: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, dict]:
    """Convert one NWB experiment/session."""
    experiment_id = int(row["ophys_experiment_id"])
    path = Path(row["nwb_path"])
    started = time.perf_counter()
    with h5py.File(path, "r") as nwb:
        event_ds = nwb[f"{EVENT_ROOT}/data"]
        ophys_t = np.asarray(nwb[f"{EVENT_ROOT}/timestamps"][:], dtype=np.float64)
        if event_ds.shape[0] != len(ophys_t):
            raise RuntimeError("Event/timestamp length mismatch")
        n_neurons = int(event_ds.shape[1])

        cell_ids = np.asarray(nwb[f"{CELL_ROOT}/cell_specimen_id"][:], dtype=np.int64)
        valid_rois = np.asarray(nwb[f"{CELL_ROOT}/valid_roi"][:], dtype=bool)
        event_rois = np.asarray(nwb[f"{EVENT_ROOT}/rois"][:], dtype=np.int64)
        if not valid_rois.all() or len(event_rois) != n_neurons:
            raise RuntimeError("Unexpected invalid/missing event ROI in published NWB")
        event_cell_ids = cell_ids[event_rois]

        trials = _trial_table(nwb)
        specs = _trial_specs(trials, ophys_t)
        if len(specs) < 2:
            raise RuntimeError(f"Experiment {experiment_id} has fewer than two retained trials")

        identity, image_change, n_presentations = _stimulus_on_ophys_grid(
            nwb, ophys_t, image_to_id
        )

        run_t = np.asarray(nwb[f"{RUN_ROOT}/timestamps"][:], dtype=np.float64)
        run_source = np.asarray(nwb[f"{RUN_ROOT}/data"][:], dtype=np.float64)
        running = _interp_finite(ophys_t, run_t, run_source)

        eye_t = np.asarray(nwb[f"{EYE_ROOT}/eye_tracking/timestamps"][:], dtype=np.float64)
        pupil_area = np.asarray(nwb[f"{EYE_ROOT}/pupil_tracking/area"][:], dtype=np.float64)
        with np.errstate(invalid="ignore"):
            pupil_diameter_source = 2.0 * np.sqrt(pupil_area / np.pi)
        pupil = _interp_finite(ophys_t, eye_t, pupil_diameter_source)
        pupil_missing, pupil_max_gap = _nan_gap_stats(eye_t, pupil_diameter_source)

        used_indices = np.concatenate(
            [np.arange(spec["lo"], spec["hi"], dtype=np.int64) for spec in specs]
        )
        if len(np.unique(used_indices)) != len(used_indices):
            raise RuntimeError("Retained trial ophys slices overlap")
        running_bin, running_thresholds = _quantile_bins(running, used_indices)
        pupil_bin, pupil_thresholds = _quantile_bins(pupil, used_indices)

        # Read only the scientifically selected neural dataset; HDF5 casts while
        # reading, halving peak memory relative to a float64 ndarray conversion.
        events = event_ds.astype(np.float32)[:]
        if not np.isfinite(events).all():
            raise RuntimeError("Neural event data contain NaN/Inf")

        neural_trials: list[np.ndarray] = []
        input_trials: list[np.ndarray] = []
        output_trials: list[np.ndarray] = []
        outcome_counts: Counter = Counter()
        trial_ids = []
        for spec in specs:
            lo, hi = spec["lo"], spec["hi"]
            neural = np.ascontiguousarray(events[lo:hi, :].T, dtype=np.float32)
            decoder_input = np.empty((0, hi - lo), dtype=np.float32)
            outcome = np.full(hi - lo, spec["outcome"], dtype=np.int16)
            decoder_output = np.vstack(
                (
                    identity[lo:hi],
                    image_change[lo:hi],
                    running_bin[lo:hi],
                    pupil_bin[lo:hi],
                    outcome,
                )
            ).astype(np.int16, copy=False)
            if neural.shape[1] != decoder_output.shape[1]:
                raise RuntimeError("Per-trial temporal dimension mismatch")
            neural_trials.append(neural)
            input_trials.append(decoder_input)
            output_trials.append(np.ascontiguousarray(decoder_output))
            outcome_counts[spec["outcome"]] += 1
            trial_ids.append(spec["trial_id"])

        if show_processing:
            spec = specs[len(specs) // 2]
            idx = len(specs) // 2
            _plot_processing(
                experiment_id=experiment_id,
                neural_trial=neural_trials[idx],
                output_trial=output_trials[idx],
                trial_t=ophys_t[spec["lo"]:spec["hi"]],
                trial_spec=spec,
                run_source_t=run_t,
                run_source_x=run_source,
                run_interp=running,
                run_thresholds=running_thresholds,
                eye_source_t=eye_t,
                pupil_source=pupil_diameter_source,
                pupil_interp=pupil,
                pupil_thresholds=pupil_thresholds,
                outcome_counts=outcome_counts,
                ophys_dt=np.diff(ophys_t),
            )

    elapsed = time.perf_counter() - started
    info = {
        "ophys_experiment_id": experiment_id,
        "ophys_session_id": int(row["ophys_session_id"]),
        "behavior_session_id": int(row["behavior_session_id"]),
        "mouse_id": str(row["mouse_id"]),
        "session_type": str(row["session_type"]),
        "experience_level": str(row["experience_level"]),
        "image_set": str(row["image_set"]),
        "targeted_structure": str(row["targeted_structure"]),
        "n_neurons": n_neurons,
        "cell_specimen_ids": event_cell_ids,
        "n_trials": len(specs),
        "trial_ids": np.asarray(trial_ids, dtype=np.int64),
        "outcome_trial_counts": {OUTCOME_COLUMNS[i]: int(outcome_counts[i]) for i in range(4)},
        "n_active_image_presentations": int(n_presentations),
        "median_ophys_interval_ms": float(np.median(np.diff(ophys_t)) * 1000.0),
        "running_quintile_thresholds_cm_per_s": running_thresholds.tolist(),
        "pupil_quintile_thresholds_pixels": pupil_thresholds.tolist(),
        "pupil_source_missing_fraction": pupil_missing,
        "pupil_source_longest_missing_gap_s": pupil_max_gap,
        "conversion_seconds": elapsed,
    }
    print(
        f"  experiment {experiment_id}: {n_neurons} neurons, {len(specs)} trials, "
        f"{sum(x.shape[1] for x in neural_trials):,} samples in {elapsed:.2f}s",
        flush=True,
    )
    return neural_trials, input_trials, output_trials, np.zeros(n_neurons, dtype=np.int64), info


def validate_converted(data: dict) -> None:
    """Fast internal structural and scientific invariants before serialization."""
    nsessions = len(data["neural"])
    if not (nsessions == len(data["input"]) == len(data["output"])):
        raise RuntimeError("Session-list length mismatch")
    if len(data["subject_idx"]) != nsessions or len(data["brain_region_idx"]) != nsessions:
        raise RuntimeError("Metadata session-list length mismatch")
    for session in range(nsessions):
        ntrials = len(data["neural"][session])
        if ntrials < 2 or len(data["input"][session]) != ntrials or len(data["output"][session]) != ntrials:
            raise RuntimeError(f"Session {session} trial-list mismatch")
        nneurons = data["neural"][session][0].shape[0]
        if len(data["brain_region_idx"][session]) != nneurons:
            raise RuntimeError(f"Session {session} brain-region length mismatch")
        for neural, decoder_input, decoder_output in zip(
            data["neural"][session], data["input"][session], data["output"][session]
        ):
            if neural.dtype != np.float32 or decoder_input.dtype != np.float32:
                raise RuntimeError("Unexpected floating dtype")
            if decoder_output.dtype != np.int16:
                raise RuntimeError("Unexpected output dtype")
            if decoder_input.shape != (0, neural.shape[1]) or decoder_output.shape != (5, neural.shape[1]):
                raise RuntimeError("Unexpected trial shape")
            if neural.shape[0] != nneurons or not np.isfinite(neural).all():
                raise RuntimeError("Invalid neural trial")
            expected_ranges = ((0, 16), (0, 1), (0, 4), (0, 4), (0, 3))
            for row, (low, high) in zip(decoder_output, expected_ranges):
                if row.min() < low or row.max() > high:
                    raise RuntimeError("Output value outside declared categories")
            if np.unique(decoder_output[4]).size != 1:
                raise RuntimeError("Trial outcome is not static")


def build_dataset(sample: bool, show_processing: bool) -> dict:
    selected, excluded = select_experiments(sample=sample)
    image_names = collect_image_names(selected)
    image_to_id = {name: idx + 1 for idx, name in enumerate(image_names)}
    print(
        f"Selected {len(selected)} {'sample' if sample else 'full'} sessions; "
        f"image identities: {image_names}",
        flush=True,
    )

    subjects = sorted(selected["mouse_id"].astype(str).unique(), key=int)
    subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}

    neural_sessions = []
    input_sessions = []
    output_sessions = []
    brain_region_idx = []
    session_info = []
    started = time.perf_counter()
    for session, (_, row) in enumerate(selected.iterrows()):
        converted = convert_session(
            row=row,
            image_to_id=image_to_id,
            show_processing=show_processing and session < 2,
        )
        neural, decoder_input, decoder_output, region_idx, info = converted
        neural_sessions.append(neural)
        input_sessions.append(decoder_input)
        output_sessions.append(decoder_output)
        brain_region_idx.append(region_idx)
        session_info.append(info)

    median_bin_ms = float(np.median([x["median_ophys_interval_ms"] for x in session_info]))
    data = {
        "neural": neural_sessions,
        "input": input_sessions,
        "output": output_sessions,
        "subjects": subjects,
        "subject_idx": np.asarray(
            [subject_lookup[str(mouse)] for mouse in selected["mouse_id"]], dtype=np.int64
        ),
        "brain_regions": ["VISp"],
        "brain_region_idx": brain_region_idx,
        "input_names": [],
        "output_names": list(OUTPUT_NAMES),
        "output_values": [
            ["gray", *image_names],
            ["no_change", "change"],
            list(QUINTILE_NAMES),
            list(QUINTILE_NAMES),
            list(OUTCOME_COLUMNS),
        ],
        "metadata": {
            "task_description": (
                "Active go/no-go natural-image change detection. Decode image identity, "
                "changed-image interval, running-speed quintile, pupil-diameter quintile, "
                "and static trial outcome from released L0 calcium events."
            ),
            "time_bin_size": median_bin_ms,
            "time_bin_size_units": "ms",
            "temporal_alignment_event": (
                "SDK trial start on the synchronized ophys timestamp grid "
                "(first ophys frame at or after start_time)"
            ),
            "off_start": 0.0,
            "off_end": None,
            "trial_window": "Variable SDK-defined half-open [start_time, stop_time) interval",
            "neural_signal": "Released raw FastLZero L0 calcium-event magnitude",
            "neural_sampling": "Native single-plane ophys frames, nominally 31 Hz",
            "project_code": PROJECT_CODE,
            "session_curation": (
                "Published-QC, active-behavior VisualBehavior experiments with eye tracking; "
                "passive sessions and three missing-eye sessions excluded"
            ),
            "trial_curation": "Go and catch only; aborted and auto-rewarded excluded",
            "image_identity_encoding": (
                "gray during ISI/omission; one of 16 image names during non-gray presentation"
            ),
            "image_change_encoding": (
                "1 throughout the 250-ms changed-image presentation, 0 otherwise; "
                "catch sham changes remain 0"
            ),
            "behavior_discretization": (
                "Per-session 20/40/60/80 percentiles over retained trial ophys samples"
            ),
            "pupil_processing": (
                "Processed blink-masked pupil area -> circularized diameter; finite linear "
                "interpolation to ophys timestamps before quintile discretization"
            ),
            "outcome_static_per_trial": True,
            "outcome_storage": "Static value repeated across time to share a (5,T) output array",
            "source_release": "visual-behavior-ophys-1.1.0",
            "source_data_root": str(DATA_ROOT),
            "sample_mode": bool(sample),
            "excluded_sessions": excluded,
            "session_info": session_info,
            "conversion_seconds_before_pickle": time.perf_counter() - started,
        },
    }
    validate_converted(data)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all eligible sessions (default)")
    mode.add_argument("--sample", action="store_true", help="Process two representative sessions")
    parser.add_argument(
        "--show-processing", action="store_true",
        help="Save processing_<session_id>.png audits for up to two sessions",
    )
    args = parser.parse_args()

    overall = time.perf_counter()
    data = build_dataset(sample=args.sample, show_processing=args.show_processing)
    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    write_start = time.perf_counter()
    with args.outpicklefile.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    write_seconds = time.perf_counter() - write_start
    total_seconds = time.perf_counter() - overall

    ntrials = sum(map(len, data["neural"]))
    nneurons = sum(session[0].shape[0] for session in data["neural"])
    nsamples = sum(trial.shape[1] for session in data["neural"] for trial in session)
    size_gib = args.outpicklefile.stat().st_size / 1024**3
    print(
        f"Wrote {args.outpicklefile} ({size_gib:.3f} GiB): "
        f"{len(data['neural'])} sessions, {len(data['subjects'])} subjects, "
        f"{ntrials:,} trials, {nneurons:,} session-neuron channels, "
        f"{nsamples:,} trial timepoints. Pickle write {write_seconds:.2f}s; "
        f"total {total_seconds:.2f}s.",
        flush=True,
    )


if __name__ == "__main__":
    main()
