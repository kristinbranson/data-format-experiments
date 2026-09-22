#!/usr/bin/env python3
"""Convert locally cached Allen Visual Behavior Ophys data for neural decoding.

Scientific data are accessed only through VisualBehaviorOphysProjectCache.
One target session is one ophys experiment (imaging plane). All streams are
sampled on a uniform 100 ms grid expressed in the experiment's ophys clock.
"""
from __future__ import annotations

import argparse
import gc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import pickle
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/app/code")
from allensdk.brain_observatory.behavior.behavior_project_cache import (  # noqa: E402
    VisualBehaviorOphysProjectCache,
)

DATA_DIR = Path("/app/data")
BIN_SEC = 0.100
MAX_PUPIL_GAP_SEC = 0.500
OUTCOMES = ["hit", "miss", "false_alarm", "correct_reject"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("outpicklefile", type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="process all active sessions (default)")
    mode.add_argument("--sample", action="store_true", help="process two active sessions")
    p.add_argument("--show-processing", action="store_true", help="save processing plots for up to two sessions")
    return p.parse_args()


def local_experiment_ids(table: pd.DataFrame) -> list[int]:
    """Identify cached experiment resources by filename; do not open NWB files."""
    valid = set(map(int, table.index))
    found = set()
    for path in DATA_DIR.rglob("*.nwb"):
        for token in re.findall(r"\d{8,}", path.name):
            value = int(token)
            if value in valid:
                found.add(value)
    if not found:
        raise RuntimeError("No locally cached ophys experiment resources found")
    return sorted(found)


def bool_col(df, name):
    return df[name].fillna(False).astype(bool).to_numpy()


def linear_sample_matrix(source_t, matrix, target_t):
    """Vectorized linear interpolation of rows in matrix onto target_t."""
    source_t = np.asarray(source_t, dtype=np.float64)
    target_t = np.asarray(target_t, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != source_t.size:
        raise ValueError(f"trace/timestamp mismatch: {matrix.shape}, {source_t.shape}")
    if target_t.size == 0 or target_t[0] < source_t[0] or target_t[-1] > source_t[-1]:
        raise ValueError("target grid outside ophys timestamp support")
    hi = np.searchsorted(source_t, target_t, side="left")
    hi = np.clip(hi, 1, source_t.size - 1)
    lo = hi - 1
    den = source_t[hi] - source_t[lo]
    alpha = ((target_t - source_t[lo]) / den).astype(np.float32)
    return matrix[:, lo] * (1.0 - alpha)[None, :] + matrix[:, hi] * alpha[None, :]


def interp_vector(source_t, values, target_t, max_gap=None):
    """Interpolate finite values without extrapolation; optionally reject long gaps."""
    source_t = np.asarray(source_t, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    target_t = np.asarray(target_t, dtype=np.float64)
    good = np.isfinite(source_t) & np.isfinite(values)
    st = source_t[good]
    sv = values[good]
    out = np.full(target_t.shape, np.nan, dtype=np.float32)
    if st.size < 2:
        return out
    inside = (target_t >= st[0]) & (target_t <= st[-1])
    out[inside] = np.interp(target_t[inside], st, sv).astype(np.float32)
    if max_gap is not None and inside.any():
        pos = np.searchsorted(st, target_t[inside], side="left")
        pos = np.clip(pos, 1, st.size - 1)
        gap_ok = (st[pos] - st[pos - 1]) <= max_gap
        ii = np.flatnonzero(inside)
        out[ii[~gap_ok]] = np.nan
    return out


def trial_grid(start, stop):
    """100 ms bin centers anchored to the experimental trial start."""
    n = int(np.floor((float(stop) - float(start)) / BIN_SEC + 1e-9))
    if n < 1:
        return np.empty(0, dtype=np.float64)
    return float(start) + BIN_SEC * (np.arange(n, dtype=np.float64) + 0.5)


def stack_dff(experiment):
    cells = experiment.cell_specimen_table
    dff = experiment.dff_traces
    if not cells.index.equals(dff.index):
        dff = dff.loc[cells.index]
    traces = np.stack(dff["dff"].to_numpy()).astype(np.float32, copy=False)
    if traces.shape[0] != len(cells):
        raise ValueError("cell table and dF/F row count differ")
    if not np.isfinite(traces).all():
        raise ValueError("nonfinite values in released dF/F traces")
    return cells.index.to_numpy(), traces


def visible_images_and_changes(stim, grid):
    image = np.empty(grid.size, dtype=object)
    image[:] = None
    change = np.zeros(grid.size, dtype=np.int8)
    if grid.size == 0:
        return image, change
    relevant = stim[(stim["start_time"] <= grid[-1]) & (stim["end_time"] > grid[0])]
    for row in relevant.itertuples():
        name = str(row.image_name)
        omitted = bool(row.omitted) if pd.notna(row.omitted) else False
        if not omitted and name not in {"omitted", "nan", "None"}:
            mask = (grid >= float(row.start_time)) & (grid < float(row.end_time))
            image[mask] = name
        is_change = bool(row.is_change) if pd.notna(row.is_change) else False
        if is_change and not omitted:
            k = int(np.floor((float(row.start_time) - (grid[0] - BIN_SEC / 2)) / BIN_SEC))
            if 0 <= k < grid.size:
                change[k] = 1
    return image, change


def prepare_experiment(cache, eid, row, verbose=True):
    st = time.time()
    exp = cache.get_behavior_ophys_experiment(int(eid))
    ophys_t = np.asarray(exp.ophys_timestamps, dtype=np.float64)
    cell_ids, dff = stack_dff(exp)
    if dff.shape[1] != ophys_t.size:
        raise ValueError("dF/F trace length differs from ophys timestamps")

    trials = exp.trials.copy()
    eligible = (
        (trials["go"].fillna(False) | trials["catch"].fillna(False))
        & ~trials["aborted"].fillna(False)
        & ~trials["auto_rewarded"].fillna(False)
    )
    trials = trials.loc[eligible]

    stim = exp.stimulus_presentations
    stim = stim[stim["stimulus_block_name"].str.contains("change_detection", na=False)].copy()
    stim = stim.sort_values("start_time")
    running = exp.running_speed.sort_values("timestamps")
    eye = exp.eye_tracking
    exclusion = Counter()
    records = []

    if eye is None:
        exclusion["missing_eye_table"] += len(trials)
    else:
        eye = eye.sort_values("timestamps")
        pupil = np.sqrt(
            eye["pupil_width"].to_numpy(float) * eye["pupil_height"].to_numpy(float)
        )
        blink = eye["likely_blink"].fillna(True).to_numpy(bool)
        pupil[blink] = np.nan

    for trial_id, tr in trials.iterrows():
        outcome_flags = [bool(tr[x]) if pd.notna(tr[x]) else False for x in OUTCOMES]
        if sum(outcome_flags) != 1:
            exclusion["ambiguous_outcome"] += 1
            continue
        grid = trial_grid(tr.start_time, tr.stop_time)
        if grid.size == 0:
            exclusion["empty_grid"] += 1
            continue
        if grid[0] < ophys_t[0] or grid[-1] > ophys_t[-1]:
            exclusion["outside_ophys_support"] += 1
            continue
        run = interp_vector(running["timestamps"], running["speed"], grid)
        if not np.isfinite(run).all():
            exclusion["missing_running"] += 1
            continue
        if eye is None:
            continue
        pup = interp_vector(
            eye["timestamps"], pupil, grid, max_gap=MAX_PUPIL_GAP_SEC
        )
        if not np.isfinite(pup).all():
            exclusion["missing_pupil"] += 1
            continue
        neural = linear_sample_matrix(ophys_t, dff, grid).astype(np.float32, copy=False)
        image_names, changes = visible_images_and_changes(stim, grid)
        records.append(
            {
                "trial_id": int(trial_id),
                "grid": grid,
                "neural": neural,
                "running": run,
                "pupil": pup,
                "image_names": image_names,
                "changes": changes,
                "outcome": int(np.flatnonzero(outcome_flags)[0]),
                "start_time": float(tr.start_time),
                "stop_time": float(tr.stop_time),
            }
        )

    native_rate = float(1.0 / np.median(np.diff(ophys_t)))
    result = {
        "eid": int(eid),
        "mouse_id": str(row.mouse_id),
        "region": str(row.targeted_structure),
        "cell_ids": cell_ids,
        "records": records,
        "exclusion": dict(exclusion),
        "info": {
            "ophys_experiment_id": int(eid),
            "ophys_session_id": int(row.ophys_session_id),
            "behavior_session_id": int(row.behavior_session_id),
            "mouse_id": str(row.mouse_id),
            "targeted_structure": str(row.targeted_structure),
            "imaging_depth": int(row.imaging_depth),
            "cre_line": str(row.cre_line),
            "experience_level": str(row.experience_level),
            "session_type": str(row.session_type),
            "native_ophys_rate_hz": native_rate,
            "n_cells": int(len(cell_ids)),
            "eligible_trials_before_stream_qc": int(len(trials)),
            "retained_trials": int(len(records)),
            "trial_exclusions": dict(exclusion),
        },
    }
    if verbose:
        print(
            f"experiment {eid}: cells={len(cell_ids)} eligible={len(trials)} "
            f"retained={len(records)} rate={native_rate:.2f}Hz "
            f"time={time.time()-st:.2f}s exclusions={dict(exclusion)}",
            flush=True,
        )
    del exp, dff
    gc.collect()
    return result


def prepare_experiment_worker(eid):
    """Process-safe wrapper; each worker accesses data through its own SDK cache."""
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=DATA_DIR)
    table = cache.get_ophys_experiment_table()
    return prepare_experiment(cache, int(eid), table.loc[int(eid)], verbose=False)


def percentile_edges(values):
    x = np.concatenate([np.asarray(v, dtype=np.float64) for v in values])
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("no finite values for percentile discretization")
    edges = np.percentile(x, [0, 20, 40, 60, 80, 100]).astype(np.float64)
    if np.any(np.diff(edges) <= 0):
        raise ValueError(f"duplicate/non-increasing percentile edges: {edges}")
    return edges


def discretize(x, edges):
    # Interior edges divide values into categories 0..4; max remains category 4.
    return np.searchsorted(edges[1:-1], x, side="right").astype(np.int16)


def make_plot(session, run_edges, pupil_edges, image_to_code):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rec = session["records"][0]
    t = rec["grid"] - rec["grid"][0]
    image = np.array([image_to_code.get(x, 0) for x in rec["image_names"]])
    fig, ax = plt.subplots(6, 1, figsize=(13, 11), sharex=True)
    show_n = min(20, rec["neural"].shape[0])
    ax[0].imshow(rec["neural"][:show_n], aspect="auto", interpolation="nearest",
                 extent=[t[0], t[-1] + BIN_SEC, show_n, 0])
    ax[0].set_ylabel("dF/F cells")
    ax[1].step(t, image, where="mid"); ax[1].set_ylabel("image code")
    ax[2].step(t, rec["changes"], where="mid"); ax[2].set_ylabel("change")
    ax[3].plot(t, rec["running"], label="speed")
    for edge in run_edges[1:-1]: ax[3].axhline(edge, color="k", alpha=.2)
    ax[3].set_ylabel("running")
    ax[4].plot(t, rec["pupil"], label="diameter")
    for edge in pupil_edges[1:-1]: ax[4].axhline(edge, color="k", alpha=.2)
    ax[4].set_ylabel("pupil")
    ax[5].step(t, discretize(rec["running"], run_edges), where="mid", label="run bin")
    ax[5].step(t, discretize(rec["pupil"], pupil_edges), where="mid", label="pupil bin", alpha=.7)
    ax[5].set_ylabel("quintile"); ax[5].set_xlabel("seconds from trial start"); ax[5].legend()
    fig.suptitle(f"Experiment {session['eid']}, trial {rec['trial_id']}: native-to-100ms processing")
    fig.tight_layout()
    out = Path(f"/app/processing_{session['eid']}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main():
    args = parse_args()
    total_start = time.time()
    warnings.filterwarnings("ignore", message=".*UpdatedStimulusPresentationTableWarning.*")
    warnings.filterwarnings("ignore", message="Ignoring cached namespace.*")
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("Creating VisualBehaviorOphysProjectCache for /app/data", flush=True)
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=DATA_DIR)
    table = cache.get_ophys_experiment_table()
    ids = local_experiment_ids(table)
    local = table.loc[ids]
    active = local[(local["behavior_type"] == "active_behavior") & (~local["passive"].astype(bool))]
    ids = list(map(int, active.index))
    if args.sample:
        ids = ids[:2]
    print(f"Mode={'sample' if args.sample else 'full'}: {len(ids)} active experiments", flush=True)

    prepared = []
    failed = []
    if args.sample or len(ids) < 4:
        for i, eid in enumerate(ids, 1):
            try:
                sess = prepare_experiment(cache, eid, active.loc[eid])
                if len(sess["records"]) >= 2:
                    prepared.append(sess)
                else:
                    failed.append((eid, "fewer_than_two_retained_trials"))
            except Exception as exc:
                failed.append((eid, f"{type(exc).__name__}: {exc}"))
                print(f"FAILED experiment {eid}: {type(exc).__name__}: {exc}", flush=True)
            print(f"progress {i}/{len(ids)} elapsed={time.time()-total_start:.1f}s", flush=True)
    else:
        workers = min(4, os.cpu_count() or 1)
        print(f"Parallel experiment processing with {workers} workers", flush=True)
        by_id = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(prepare_experiment_worker, eid): eid for eid in ids}
            for i, future in enumerate(as_completed(futures), 1):
                eid = futures[future]
                try:
                    sess = future.result()
                    print(
                        f"experiment {eid}: cells={len(sess['cell_ids'])} "
                        f"retained={len(sess['records'])} exclusions={sess['exclusion']}",
                        flush=True,
                    )
                    if len(sess["records"]) >= 2:
                        by_id[eid] = sess
                    else:
                        failed.append((eid, "fewer_than_two_retained_trials"))
                except Exception as exc:
                    failed.append((eid, f"{type(exc).__name__}: {exc}"))
                    print(f"FAILED experiment {eid}: {type(exc).__name__}: {exc}", flush=True)
                if i % 5 == 0 or i == len(ids):
                    print(f"progress {i}/{len(ids)} elapsed={time.time()-total_start:.1f}s", flush=True)
        prepared = [by_id[eid] for eid in ids if eid in by_id]
    if not prepared:
        raise RuntimeError(f"No sessions retained; failures={failed}")

    run_edges = percentile_edges([r["running"] for s in prepared for r in s["records"]])
    pupil_edges = percentile_edges([r["pupil"] for s in prepared for r in s["records"]])
    real_images = sorted({str(x) for s in prepared for r in s["records"] for x in r["image_names"] if x is not None})
    image_values = ["none/gray"] + real_images
    image_to_code = {name: i for i, name in enumerate(image_values)}
    print("running percentile edges", run_edges.tolist())
    print("pupil percentile edges", pupil_edges.tolist())
    print("image categories", image_values)

    subjects = sorted({s["mouse_id"] for s in prepared})
    subject_map = {x: i for i, x in enumerate(subjects)}
    brain_regions = sorted({s["region"] for s in prepared})
    region_map = {x: i for i, x in enumerate(brain_regions)}
    neural, inputs, outputs, region_indices, session_info = [], [], [], [], []

    for sess in prepared:
        sn, si, so = [], [], []
        for rec in sess["records"]:
            T = rec["neural"].shape[1]
            image = np.array([image_to_code.get(x, 0) for x in rec["image_names"]], dtype=np.int16)
            out = np.vstack([
                image,
                rec["changes"].astype(np.int16),
                discretize(rec["running"], run_edges),
                discretize(rec["pupil"], pupil_edges),
                np.full(T, rec["outcome"], dtype=np.int16),
            ])
            inp = np.empty((0, T), dtype=np.float32)
            if rec["neural"].ndim != 2 or inp.shape[1] != T or out.shape != (5, T):
                raise AssertionError("per-trial shape inconsistency")
            if not np.isfinite(rec["neural"]).all():
                raise AssertionError("nonfinite converted neural data")
            sn.append(rec["neural"].astype(np.float32, copy=False))
            si.append(inp)
            so.append(out)
        neural.append(sn); inputs.append(si); outputs.append(so)
        region_indices.append(np.full(len(sess["cell_ids"]), region_map[sess["region"]], dtype=np.int16))
        session_info.append(sess["info"])

    data = {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": np.array([subject_map[s["mouse_id"]] for s in prepared], dtype=np.int16),
        "brain_regions": brain_regions,
        "brain_region_idx": region_indices,
        "input_names": [],
        "output_names": ["image_identity", "image_change", "running_speed_bin", "pupil_diameter_bin", "trial_outcome"],
        "output_values": [
            image_values,
            ["no_change", "change"],
            ["q1_slowest", "q2", "q3", "q4", "q5_fastest"],
            ["q1_smallest", "q2", "q3", "q4", "q5_largest"],
            OUTCOMES,
        ],
        "metadata": {
            "task_description": "Decode visible image identity, image-change events, running-speed quintile, pupil-diameter quintile, and trial outcome from visual-cortical dF/F during active go/catch change-detection trials.",
            "time_bin_size": 100.0,
            "temporal_alignment_event": "trial start; 100 ms bin centers represented in absolute ophys timestamp coordinates",
            "off_start": None,
            "off_end": None,
            "neural_signal": "AllenSDK released detrended dF/F",
            "session_unit": "ophys experiment (one imaging plane)",
            "trial_filter": "(go or catch) and not aborted and not auto_rewarded; complete ophys/running/pupil support",
            "stimulus_filter": "stimulus_block_name contains change_detection",
            "pupil_definition": "sqrt(pupil_width * pupil_height), with likely_blink invalid and interpolation gaps <=0.5 s",
            "percentile_scope": "global across all retained aligned time bins",
            "running_percentile_edges": run_edges.tolist(),
            "pupil_percentile_edges": pupil_edges.tolist(),
            "image_encoding": image_to_code,
            "session_info": session_info,
            "failed_or_excluded_sessions": failed,
            "source_cache": "/app/data loaded with VisualBehaviorOphysProjectCache",
        },
    }

    # Whole-dataset assertions and summaries.
    assert len(data["neural"]) == len(data["input"]) == len(data["output"]) == len(data["subject_idx"])
    assert all(len(x) >= 2 for x in data["neural"])
    class_counts = [np.zeros(len(v), dtype=np.int64) for v in data["output_values"]]
    total_trials = 0
    for session in data["output"]:
        total_trials += len(session)
        for trial in session:
            for j, values in enumerate(data["output_values"]):
                if trial[j].min() < 0 or trial[j].max() >= len(values):
                    raise AssertionError(f"output {j} code outside declared values")
                class_counts[j] += np.bincount(trial[j], minlength=len(values))
    print(f"retained sessions={len(neural)} trials={total_trials} subjects={len(subjects)}")
    for name, count in zip(data["output_names"], class_counts):
        print(name, "counts", count.tolist(), "fractions", (count / count.sum()).round(6).tolist())

    if args.show_processing:
        for sess in prepared[:2]:
            make_plot(sess, run_edges, pupil_edges, image_to_code)

    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    with args.outpicklefile.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = args.outpicklefile.stat().st_size / 1024**2
    print(f"saved {args.outpicklefile} ({size_mb:.1f} MiB) in {time.time()-total_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
