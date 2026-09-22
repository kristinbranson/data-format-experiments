#!/usr/bin/env python3
"""Convert the local Allen VisualBehavior cache to decoder format.

All experiment access is through VisualBehaviorOphysProjectCache. NWB files are
never opened directly.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parent
SDK = APP / "code"
if str(SDK) not in sys.path:
    sys.path.insert(0, str(SDK))

from allensdk.brain_observatory.behavior.behavior_project_cache import (  # noqa: E402
    VisualBehaviorOphysProjectCache,
)

DATA_DIR = APP / "data"
OUTCOMES = ["hit", "miss", "false_alarm", "correct_reject"]


def make_cache():
    return VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=DATA_DIR)


def local_experiment_ids() -> set[int]:
    folder = DATA_DIR / "visual-behavior-ophys-1.1.0" / "behavior_ophys_experiments"
    return {int(p.stem.rsplit("_", 1)[1]) for p in folder.glob("*.nwb")}


def selected_table(cache):
    table = cache.get_ophys_experiment_table()
    mask = (
        (table["project_code"] == "VisualBehavior")
        & (table["behavior_type"] == "active_behavior")
        & table.index.isin(local_experiment_ids())
    )
    return table.loc[mask].sort_index()


def task_stimuli(dataset):
    stim = dataset.stimulus_presentations
    names = stim["stimulus_block_name"].fillna("").astype(str)
    return stim.loc[names.str.contains("change_detection", case=False)].sort_values("start_time")


def discover_images(cache, table) -> list[str]:
    """Load one representative of each image set through AllenSDK."""
    found: set[str] = set()
    for _, group in table.groupby("image_set", dropna=False):
        ds = cache.get_behavior_ophys_experiment(int(group.index[0]))
        stim = task_stimuli(ds)
        vals = stim.loc[~stim["omitted"].fillna(False), "image_name"].dropna().astype(str)
        found.update(v for v in vals.unique() if v.lower() not in {"omitted", "nan"})
    return ["gray"] + sorted(found)


def percentile_bins(values: np.ndarray) -> np.ndarray:
    """Five approximately equal-count percentile bins, deterministically."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("nonfinite or empty values supplied to percentile_bins")
    edges = np.percentile(values, [20, 40, 60, 80])
    return np.searchsorted(edges, values, side="right").astype(np.int16)


def interpolate_valid(t_src, value_src, t_dst):
    t_src = np.asarray(t_src, dtype=np.float64)
    value_src = np.asarray(value_src, dtype=np.float64)
    valid = np.isfinite(t_src) & np.isfinite(value_src)
    if valid.sum() < 2:
        raise ValueError("fewer than two finite source samples")
    order = np.argsort(t_src[valid])
    return np.interp(t_dst, t_src[valid][order], value_src[valid][order])


def build_continuous_outputs(dataset, timestamps, image_to_idx):
    n = len(timestamps)
    image = np.full(n, image_to_idx["gray"], dtype=np.int16)
    change = np.zeros(n, dtype=np.int16)
    stim = task_stimuli(dataset)
    for row in stim.itertuples():
        start = int(np.searchsorted(timestamps, float(row.start_time), side="left"))
        end = int(np.searchsorted(timestamps, float(row.end_time), side="left"))
        omitted = bool(row.omitted) if not np.isnan(row.omitted) else False
        name = str(row.image_name)
        if not omitted and name in image_to_idx and end > start:
            image[start:end] = image_to_idx[name]
        if bool(row.is_change) and start < n:
            change[start] = 1

    run = dataset.running_speed
    speed = interpolate_valid(run["timestamps"], run["speed"], timestamps)
    speed_bin = percentile_bins(speed)

    eye = dataset.eye_tracking
    if eye is None:
        raise ValueError("eye tracking unavailable")
    area = eye["pupil_area"].to_numpy(dtype=np.float64)
    diameter = 2.0 * np.sqrt(np.maximum(area, 0.0) / np.pi)
    pupil = interpolate_valid(eye["timestamps"], diameter, timestamps)
    pupil_bin = percentile_bins(pupil)
    return image, change, speed_bin, pupil_bin, speed, pupil


def convert_experiment(cache, experiment_id, image_to_idx, show_processing=False):
    started = time.perf_counter()
    ds = cache.get_behavior_ophys_experiment(int(experiment_id))
    ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
    events = np.stack(ds.events["events"].to_numpy()).astype(np.float32, copy=False)
    n = min(ts.size, events.shape[1])
    ts, events = ts[:n], events[:, :n]
    if len(ts) < 2 or np.any(np.diff(ts) <= 0):
        raise ValueError("invalid ophys timestamps")
    if not np.all(np.isfinite(events)):
        raise ValueError("nonfinite calcium events")
    image, change, speed_bin, pupil_bin, speed, pupil = build_continuous_outputs(
        ds, ts, image_to_idx
    )

    trials = ds.trials
    keep = (trials["go"] | trials["catch"]) & ~trials["aborted"] & ~trials["auto_rewarded"]
    trials = trials.loc[keep]
    neural_trials, input_trials, output_trials = [], [], []
    plot_rows = []
    outcome_counts = np.zeros(4, dtype=np.int64)
    for trial_id, row in trials.iterrows():
        flags = np.asarray([bool(row[x]) for x in OUTCOMES])
        if flags.sum() != 1:
            continue
        lo = int(np.searchsorted(ts, float(row.start_time), side="left"))
        hi = int(np.searchsorted(ts, float(row.stop_time), side="left"))
        if hi - lo < 2:
            continue
        outcome = int(np.flatnonzero(flags)[0])
        y = np.empty((5, hi - lo), dtype=np.int16)
        y[0] = image[lo:hi]
        y[1] = change[lo:hi]
        y[2] = speed_bin[lo:hi]
        y[3] = pupil_bin[lo:hi]
        y[4] = outcome
        x = events[:, lo:hi]
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
            continue
        neural_trials.append(x)
        input_trials.append(np.empty((0, hi - lo), dtype=np.float32))
        output_trials.append(y)
        outcome_counts[outcome] += 1
        if len(plot_rows) < 6:
            plot_rows.append((lo, hi, outcome, int(trial_id)))
    if len(neural_trials) < 2:
        raise ValueError(f"only {len(neural_trials)} valid trials")

    if show_processing:
        plot_processing(experiment_id, ts, events, image, change, speed, speed_bin,
                        pupil, pupil_bin, plot_rows)
    elapsed = time.perf_counter() - started
    info = {
        "ophys_experiment_id": int(experiment_id),
        "ophys_session_id": int(ds.metadata["ophys_session_id"]),
        "mouse_id": str(ds.metadata["mouse_id"]),
        "targeted_structure": str(ds.metadata["targeted_structure"]),
        "session_type": str(ds.metadata["session_type"]),
        "n_neurons": int(events.shape[0]),
        "n_trials": len(neural_trials),
        "n_native_trials": int(len(ds.trials)),
        "outcome_counts": outcome_counts.tolist(),
        "median_frame_interval_ms": float(np.median(np.diff(ts)) * 1000),
        "seconds": elapsed,
    }
    return neural_trials, input_trials, output_trials, info


def plot_processing(experiment_id, ts, events, image, change, speed, speed_bin,
                    pupil, pupil_bin, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(6, 1, figsize=(18, 16), sharex=False)
    if rows:
        lo = rows[0][0]
        hi = min(rows[-1][1], lo + 31 * 45)
    else:
        lo, hi = 0, min(len(ts), 31 * 45)
    t = ts[lo:hi] - ts[lo]
    z = events[:min(80, len(events)), lo:hi]
    ax[0].imshow(z, aspect="auto", interpolation="none", extent=[t[0], t[-1], z.shape[0], 0])
    ax[0].set_ylabel("cell"); ax[0].set_title(f"{experiment_id}: SDK L0 calcium events")
    ax[1].step(t, image[lo:hi], where="post", label="image class")
    ax[1].plot(t, change[lo:hi] * max(1, image.max()), "r", label="change impulse")
    ax[1].legend(); ax[1].set_ylabel("stimulus")
    ax[2].plot(t, speed[lo:hi], alpha=.7, label="cm/s")
    ax2 = ax[2].twinx(); ax2.step(t, speed_bin[lo:hi], where="mid", color="tab:orange", label="quintile")
    ax[2].set_ylabel("running cm/s"); ax2.set_ylabel("bin 0–4")
    ax[3].plot(t, pupil[lo:hi], alpha=.7, label="diameter px")
    ax3 = ax[3].twinx(); ax3.step(t, pupil_bin[lo:hi], where="mid", color="tab:orange", label="quintile")
    ax[3].set_ylabel("pupil diameter"); ax3.set_ylabel("bin 0–4")
    for a in ax[:4]:
        a.set_xlim(t[0], t[-1])
    ax[3].set_xlabel("seconds from displayed window start")
    ax[4].hist([r[1] - r[0] for r in rows], bins=max(1, len(rows)))
    ax[4].set_ylabel("trial count"); ax[4].set_xlabel("trial length (ophys frames)")
    counts = np.bincount([r[2] for r in rows], minlength=4)
    ax[5].bar(OUTCOMES, counts)
    ax[5].set_ylabel("shown trial count"); ax[5].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(APP / f"processing_{experiment_id}.png", dpi=130)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("outpicklefile")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="process all sessions (default)")
    mode.add_argument("--sample", action="store_true", help="process two sessions")
    p.add_argument("--show-processing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    total_start = time.perf_counter()
    warnings.filterwarnings("ignore", message="Ignoring cached namespace")
    cache = make_cache()
    table = selected_table(cache)
    images = discover_images(cache, table)
    image_to_idx = {name: i for i, name in enumerate(images)}
    if args.sample:
        # Cover both image sets and use cell-rich sessions so the two-session
        # decoder smoke test has enough neural signal and class examples.
        cells = cache.get_ophys_cells_table()
        cell_counts = cells.groupby("ophys_experiment_id").size()
        chosen = []
        for _, group in table.groupby("image_set", sort=True):
            counts = cell_counts.reindex(group.index).fillna(0)
            chosen.append(int(counts.idxmax()))
        ids = chosen[:2]
    else:
        ids = [int(x) for x in table.index]
    print(f"Mode={'sample' if args.sample else 'full'} sessions={len(ids)} images={images}", flush=True)

    neural, inputs, outputs, infos, kept_ids = [], [], [], [], []
    for k, experiment_id in enumerate(ids, 1):
        try:
            result = convert_experiment(
                cache, experiment_id, image_to_idx,
                show_processing=args.show_processing and k <= 2,
            )
        except Exception as exc:
            print(f"SKIP {experiment_id}: {type(exc).__name__}: {exc}", flush=True)
            continue
        n, i, o, info = result
        neural.append(n); inputs.append(i); outputs.append(o); infos.append(info); kept_ids.append(experiment_id)
        print(f"[{k}/{len(ids)}] id={experiment_id} neurons={info['n_neurons']} "
              f"trials={info['n_trials']} time={info['seconds']:.2f}s", flush=True)

    if not neural:
        raise RuntimeError("no sessions converted")
    kept = table.loc[kept_ids]
    subjects = sorted(kept["mouse_id"].astype(str).unique())
    subject_map = {x: i for i, x in enumerate(subjects)}
    regions = sorted(kept["targeted_structure"].astype(str).unique())
    region_map = {x: i for i, x in enumerate(regions)}
    subject_idx = np.asarray([subject_map[str(x)] for x in kept["mouse_id"]], dtype=np.int64)
    brain_region_idx = [
        np.full(info["n_neurons"], region_map[info["targeted_structure"]], dtype=np.int64)
        for info in infos
    ]
    median_bin = float(np.median([x["median_frame_interval_ms"] for x in infos]))
    data = {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": subject_idx,
        "brain_regions": regions,
        "brain_region_idx": brain_region_idx,
        "input_names": [],
        "output_names": ["image_identity", "image_change", "running_speed_quintile",
                         "pupil_diameter_quintile", "trial_outcome"],
        "output_values": [images, ["no_change", "change"],
                          ["q1_lowest", "q2", "q3", "q4", "q5_highest"],
                          ["q1_lowest", "q2", "q3", "q4", "q5_highest"], OUTCOMES],
        "metadata": {
            "task_description": "Decode flashed image identity/change, running and pupil quintiles, and go/catch trial outcome from VISp L0 calcium events during active visual change detection.",
            "time_bin_size": median_bin,
            "temporal_alignment_event": "Native synchronized ophys frame timestamps; each trial is the half-open SDK interval [start_time, stop_time).",
            "off_start": None,
            "off_end": None,
            "neural_signal": "AllenSDK release-QC-valid unfiltered L0 calcium event magnitudes",
            "trial_filter": "(go OR catch) AND NOT aborted AND NOT auto_rewarded; exactly one valid outcome",
            "project_code": "VisualBehavior",
            "session_info": infos,
            "image_gray_definition": "gray class includes 500-ms ISIs and omitted image intervals",
            "pupil_definition": "2*sqrt(blink-clean pupil_area/pi), interpolated at ophys timestamps",
            "quintile_definition": "per-session 20/40/60/80 percentiles on aligned continuous samples",
        },
    }
    out = Path(args.outpicklefile)
    with out.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    total_trials = sum(map(len, neural))
    total_frames = sum(x.shape[1] for s in neural for x in s)
    print(f"WROTE {out} sessions={len(neural)} trials={total_trials} frames={total_frames} "
          f"size={out.stat().st_size} elapsed={time.perf_counter()-total_start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
