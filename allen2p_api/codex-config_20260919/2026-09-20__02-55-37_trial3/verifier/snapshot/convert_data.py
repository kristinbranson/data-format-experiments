#!/usr/bin/env python3
"""Convert cached Allen Visual Behavior Ophys experiments for neural decoding.

All detailed data access goes through VisualBehaviorOphysProjectCache. This
script intentionally never opens an NWB file directly.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import pickle
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

APP = Path(__file__).resolve().parent
SDK_SOURCE = APP / "code"
if str(SDK_SOURCE) not in sys.path:
    sys.path.insert(0, str(SDK_SOURCE))

from allensdk.brain_observatory.behavior.behavior_project_cache import (  # noqa: E402
    VisualBehaviorOphysProjectCache,
)


CACHE_DIR = APP / "data"
RELEASE_DIR = CACHE_DIR / "visual-behavior-ophys-1.1.0"
EXPERIMENT_DIR = RELEASE_DIR / "behavior_ophys_experiments"
DT = 0.100  # seconds
OUTCOME_COLUMNS = ["hit", "miss", "false_alarm", "correct_reject"]
OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]


def cached_experiment_ids() -> list[int]:
    """Enumerate cache assets without reading their contents."""
    prefix = "behavior_ophys_experiment_"
    return sorted(int(p.stem.removeprefix(prefix)) for p in EXPERIMENT_DIR.glob(f"{prefix}*.nwb"))


def active_experiment_table(cache) -> pd.DataFrame:
    table = cache.get_ophys_experiment_table()
    ids = [x for x in cached_experiment_ids() if x in table.index]
    selected = table.loc[ids]
    selected = selected[selected["behavior_type"].eq("active_behavior")]
    return selected.sort_index()


def valid_trials(trials: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    flags = trials[OUTCOME_COLUMNS].fillna(False).astype(bool)
    contingent = trials["go"].fillna(False).astype(bool) | trials["catch"].fillna(False).astype(bool)
    non_aborted = ~trials["aborted"].fillna(False).astype(bool)
    non_auto = ~trials["auto_rewarded"].fillna(False).astype(bool)
    one_outcome = flags.sum(axis=1).eq(1)
    finite_bounds = np.isfinite(trials["start_time"]) & np.isfinite(trials["stop_time"])
    positive = trials["stop_time"].to_numpy() > trials["start_time"].to_numpy()
    mask = contingent & non_aborted & non_auto & one_outcome & finite_bounds & positive
    counts = {
        "raw": int(len(trials)),
        "noncontingent_or_aborted": int((~(contingent & non_aborted)).sum()),
        "auto_rewarded": int((contingent & non_aborted & ~non_auto).sum()),
        "invalid_outcome_or_bounds": int((contingent & non_aborted & non_auto & ~(one_outcome & finite_bounds & positive)).sum()),
        "eligible": int(mask.sum()),
    }
    return trials.loc[mask].copy(), counts


def active_stimulus_table(stim: pd.DataFrame) -> pd.DataFrame:
    mask = stim["active"].fillna(False).astype(bool) if "active" in stim else np.ones(len(stim), bool)
    if "stimulus_block_name" in stim:
        names = stim["stimulus_block_name"].fillna("").astype(str)
        mask &= names.str.contains("change_detection", case=False, regex=False)
    ans = stim.loc[mask].copy()
    return ans.sort_values("start_time")


def stack_event_traces(events: pd.DataFrame, nframes: int) -> tuple[np.ndarray, np.ndarray]:
    cell_ids = events.index.to_numpy(dtype=np.int64)
    if len(cell_ids) == 0:
        return np.empty((0, nframes), np.float32), cell_ids
    traces = np.vstack(events["events"].to_numpy()).astype(np.float32, copy=False)
    if traces.shape != (len(cell_ids), nframes):
        raise ValueError(f"event shape {traces.shape} != ({len(cell_ids)}, {nframes})")
    if not np.isfinite(traces).all():
        raise ValueError("non-finite inferred event value")
    return traces, cell_ids


def bin_events(events: np.ndarray, timestamps: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Sum frames in half-open bins using a small trial-local cumulative sum."""
    indices = np.searchsorted(timestamps, edges, side="left")
    lo, hi = int(indices[0]), int(indices[-1])
    local = events[:, lo:hi]
    cumulative = np.empty((events.shape[0], local.shape[1] + 1), dtype=np.float32)
    cumulative[:, 0] = 0.0
    np.cumsum(local, axis=1, dtype=np.float32, out=cumulative[:, 1:])
    rel = indices - lo
    return (cumulative[:, rel[1:]] - cumulative[:, rel[:-1]]).astype(np.float32, copy=False)


def interpolate_finite(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray | None:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    good = np.isfinite(times) & np.isfinite(values)
    if good.sum() < 2:
        return None
    times, values = times[good], values[good]
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    unique = np.r_[True, np.diff(times) > 0]
    times, values = times[unique], values[unique]
    if len(times) < 2 or query[0] < times[0] or query[-1] > times[-1]:
        return None
    return np.interp(query, times, values).astype(np.float32)


def label_stimuli(stim: pd.DataFrame, centers: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    labels = np.full(len(centers), "gray", dtype=object)
    changes = np.zeros(len(centers), dtype=np.int16)
    if len(stim) == 0:
        return labels, changes, 0
    starts = stim["start_time"].to_numpy(float)
    ends = stim["end_time"].to_numpy(float)
    idx = np.searchsorted(starts, centers, side="right") - 1
    valid_idx = idx >= 0
    rows = np.maximum(idx, 0)
    image_name = stim["image_name"].fillna("").astype(str).to_numpy()
    omitted = stim["omitted"].fillna(False).astype(bool).to_numpy() if "omitted" in stim else np.zeros(len(stim), bool)
    shown = valid_idx & (centers < ends[rows]) & (~omitted[rows]) & (image_name[rows] != "") & (image_name[rows] != "omitted")
    labels[shown] = image_name[rows[shown]]
    is_change = stim["is_change"].fillna(False).astype(bool).to_numpy()
    change_rows = np.flatnonzero(is_change & (starts < edges[-1]) & (ends > edges[0]))
    for row in change_rows:
        # `is_change` belongs to the newly shown image presentation. Mark its
        # complete on-screen interval (normally 250 ms), i.e. the period
        # immediately after identity changed, rather than an undersampled
        # mathematical impulse at onset.
        changes[(centers >= starts[row]) & (centers < ends[row])] = 1
    return labels, changes, len(change_rows)


def outcome_code(row: pd.Series) -> int:
    flags = np.asarray([bool(row[x]) for x in OUTCOME_COLUMNS])
    if flags.sum() != 1:
        raise ValueError("trial does not have exactly one outcome")
    return int(np.flatnonzero(flags)[0])


def convert_experiment(cache, experiment_id: int, meta: pd.Series, keep_diagnostic: bool = False) -> tuple[dict | None, dict]:
    started = time.perf_counter()
    obj = cache.get_behavior_ophys_experiment(int(experiment_id))
    timestamps = np.asarray(obj.ophys_timestamps, dtype=float)
    events, cell_ids = stack_event_traces(obj.events, len(timestamps))
    eligible, exclusions = valid_trials(obj.trials)
    stim = active_stimulus_table(obj.stimulus_presentations)
    running = obj.running_speed
    eye = obj.eye_tracking
    if eye is None or len(eye) == 0 or "pupil_area" not in eye:
        exclusions["no_eye_session"] = len(eligible)
        print(f"  experiment {experiment_id}: excluded (no processed pupil data)", flush=True)
        return None, exclusions

    pupil_area = eye["pupil_area"].to_numpy(float)
    pupil_diameter = 2.0 * np.sqrt(np.maximum(pupil_area, 0.0) / np.pi)
    records = []
    dropped_coverage = 0
    dropped_short = 0
    dropped_stim_consistency = 0
    trial_ids = []
    for trial_id, row in eligible.iterrows():
        start, stop = float(row.start_time), float(row.stop_time)
        nbins = int(np.floor((stop - start) / DT + 1e-9))
        if nbins < 2:
            dropped_short += 1
            continue
        edges = start + np.arange(nbins + 1, dtype=float) * DT
        centers = edges[:-1] + DT / 2.0
        run = interpolate_finite(running["timestamps"].to_numpy(), running["speed"].to_numpy(), centers)
        pupil = interpolate_finite(eye["timestamps"].to_numpy(), pupil_diameter, centers)
        if run is None or pupil is None:
            dropped_coverage += 1
            continue
        trial_stim = stim[(stim["start_time"] < edges[-1]) & (stim["end_time"] > edges[0])]
        image_labels, change, nchanges = label_stimuli(trial_stim, centers, edges)
        is_go, is_catch = bool(row.go), bool(row.catch)
        if (is_go and nchanges != 1) or (is_catch and nchanges != 0):
            dropped_stim_consistency += 1
            continue
        neural = bin_events(events, timestamps, edges)
        if neural.shape[1] != nbins:
            raise AssertionError("neural bin length mismatch")
        code = outcome_code(row)
        records.append({
            "neural": neural,
            "image_labels": image_labels,
            "change": change,
            "running": run,
            "pupil": pupil,
            "outcome": code,
            "trial_id": int(trial_id),
            "start_time": start,
            "stop_time_binned": float(edges[-1]),
        })
        trial_ids.append(int(trial_id))

    exclusions.update({
        "dropped_coverage": dropped_coverage,
        "dropped_short": dropped_short,
        "dropped_stim_consistency": dropped_stim_consistency,
        "retained": len(records),
    })
    if len(records) < 2 or len(cell_ids) == 0:
        print(f"  experiment {experiment_id}: excluded ({len(records)} trials, {len(cell_ids)} cells)", flush=True)
        return None, exclusions

    result = {
        "experiment_id": int(experiment_id),
        "ophys_session_id": int(meta.ophys_session_id),
        "behavior_session_id": int(meta.behavior_session_id),
        "mouse_id": str(meta.mouse_id),
        "region": str(meta.targeted_structure),
        "cre_line": str(meta.cre_line),
        "experience_level": str(meta.experience_level),
        "session_type": str(meta.session_type),
        "project_code": str(meta.project_code),
        "native_frame_rate_hz": float(obj.metadata.get("ophys_frame_rate", np.nan)),
        "cell_ids": cell_ids,
        "records": records,
        "exclusions": exclusions,
        "diagnostic": {
            "ophys_timestamps": timestamps,
            "event_mean": events.mean(axis=0, dtype=np.float64).astype(np.float32),
            "running_times": running["timestamps"].to_numpy(float),
            "running_speed": running["speed"].to_numpy(float),
            "eye_times": eye["timestamps"].to_numpy(float),
            "pupil_diameter": pupil_diameter,
        } if keep_diagnostic else None,
    }
    elapsed = time.perf_counter() - started
    print(f"  experiment {experiment_id}: {len(cell_ids)} cells, {len(records)} trials, {elapsed:.2f}s", flush=True)
    del obj, events
    gc.collect()
    return result, exclusions


def convert_experiment_worker(item: tuple[int, dict]) -> tuple[int, dict | None, dict]:
    """Independent SDK cache/session construction for safe process parallelism."""
    experiment_id, meta_dict = item
    try:
        cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=str(CACHE_DIR))
        session, exclusions = convert_experiment(cache, experiment_id, pd.Series(meta_dict), keep_diagnostic=False)
        return experiment_id, session, exclusions
    except Exception as exc:
        print(f"  experiment {experiment_id}: ERROR {exc!r}", flush=True)
        return experiment_id, None, {"fatal_error": repr(exc)}


def percentile_edges(values: list[np.ndarray]) -> np.ndarray:
    joined = np.concatenate(values).astype(np.float64, copy=False)
    edges = np.quantile(joined[np.isfinite(joined)], [0.2, 0.4, 0.6, 0.8])
    if not np.all(np.diff(edges) > 0):
        warnings.warn(f"Repeated percentile edges: {edges}")
    return edges


def discretize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, values, side="right").astype(np.int16)


def plot_processing(session: dict, run_edges: np.ndarray, pupil_edges: np.ndarray) -> None:
    rec = session["records"][0]
    diag = session["diagnostic"]
    if diag is None:
        raise ValueError("diagnostic data were not retained")
    start, stop = rec["start_time"], rec["stop_time_binned"]
    centers = start + DT / 2 + np.arange(len(rec["running"])) * DT
    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
    native = (diag["ophys_timestamps"] >= start) & (diag["ophys_timestamps"] < stop)
    axes[0].plot(diag["ophys_timestamps"][native], diag["event_mean"][native], color="0.65", label="native mean event")
    axes[0].step(centers, rec["neural"].mean(axis=0), where="mid", label="100 ms summed mean")
    axes[0].set_ylabel("Neural events")
    axes[0].legend(loc="upper right")
    image_names = sorted(set(rec["image_labels"]))
    image_map = {x: i for i, x in enumerate(image_names)}
    axes[1].step(centers, [image_map[x] for x in rec["image_labels"]], where="mid", label="image identity")
    axes[1].step(centers, rec["change"] * max(1, len(image_names) - 1), where="mid", label="change")
    axes[1].set_yticks(range(len(image_names)), image_names)
    axes[1].legend(loc="upper right")
    rm = (diag["running_times"] >= start) & (diag["running_times"] < stop)
    axes[2].plot(diag["running_times"][rm], diag["running_speed"][rm], color="0.75", label="SDK samples")
    axes[2].plot(centers, rec["running"], label="aligned")
    axes[2].set_ylabel("Running cm/s")
    axes[2].legend(loc="upper right")
    axes[3].step(centers, discretize(rec["running"], run_edges), where="mid")
    axes[3].set_ylabel("Running quintile")
    pm = (diag["eye_times"] >= start) & (diag["eye_times"] < stop)
    axes[4].plot(diag["eye_times"][pm], diag["pupil_diameter"][pm], color="0.75", label="processed diameter")
    axes[4].plot(centers, rec["pupil"], label="aligned diameter")
    ax2 = axes[4].twinx()
    ax2.step(centers, discretize(rec["pupil"], pupil_edges), where="mid", color="tab:red", alpha=.5, label="quintile")
    axes[4].set_ylabel("Pupil diameter px")
    ax2.set_ylabel("Pupil quintile")
    axes[4].set_xlabel("Synchronized time (s)")
    axes[4].legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.suptitle(f"Experiment {session['experiment_id']}: every conversion stage, trial {rec['trial_id']}")
    fig.tight_layout()
    fig.savefig(APP / f"processing_{session['experiment_id']}.png", dpi=140)
    plt.close(fig)


def assemble(sessions: list[dict], show_processing: bool) -> dict:
    run_edges = percentile_edges([r["running"] for s in sessions for r in s["records"]])
    pupil_edges = percentile_edges([r["pupil"] for s in sessions for r in s["records"]])
    image_values = ["gray"] + sorted({str(x) for s in sessions for r in s["records"] for x in r["image_labels"] if x != "gray"})
    image_map = {x: i for i, x in enumerate(image_values)}
    subjects = sorted({s["mouse_id"] for s in sessions})
    subject_map = {x: i for i, x in enumerate(subjects)}
    regions = sorted({s["region"] for s in sessions})
    region_map = {x: i for i, x in enumerate(regions)}

    neural, inputs, outputs, region_indices, session_info = [], [], [], [], []
    for s in sessions:
        sn, si, so = [], [], []
        for r in s["records"]:
            T = r["neural"].shape[1]
            image = np.fromiter((image_map[str(x)] for x in r["image_labels"]), dtype=np.int16, count=T)
            outcome = np.full(T, r["outcome"], dtype=np.int16)
            output = np.vstack([
                image,
                r["change"].astype(np.int16, copy=False),
                discretize(r["running"], run_edges),
                discretize(r["pupil"], pupil_edges),
                outcome,
            ]).astype(np.int16, copy=False)
            if output.shape != (5, T) or not np.isfinite(r["neural"]).all():
                raise AssertionError("invalid final trial")
            sn.append(r["neural"].astype(np.float32, copy=False))
            si.append(np.empty((0, T), dtype=np.float32))
            so.append(output)
        neural.append(sn); inputs.append(si); outputs.append(so)
        region_indices.append(np.full(len(s["cell_ids"]), region_map[s["region"]], dtype=np.int64))
        session_info.append({
            k: s[k] for k in ["experiment_id", "ophys_session_id", "behavior_session_id", "mouse_id", "region", "cre_line", "experience_level", "session_type", "project_code", "native_frame_rate_hz"]
        } | {
            "cell_specimen_ids": s["cell_ids"].tolist(),
            "trial_ids": [r["trial_id"] for r in s["records"]],
            "trial_start_times": [r["start_time"] for r in s["records"]],
            "trial_stop_times_binned": [r["stop_time_binned"] for r in s["records"]],
            "exclusions": s["exclusions"],
        })
    if show_processing:
        for s in sessions[:2]:
            plot_processing(s, run_edges, pupil_edges)

    return {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": np.asarray([subject_map[s["mouse_id"]] for s in sessions], dtype=np.int64),
        "brain_regions": regions,
        "brain_region_idx": region_indices,
        "input_names": [],
        "output_names": ["image_identity", "image_change", "running_speed_quintile", "pupil_diameter_quintile", "trial_outcome"],
        "output_values": [
            image_values,
            ["no_change", "change"],
            ["Q1_slowest", "Q2", "Q3", "Q4", "Q5_fastest"],
            ["Q1_smallest", "Q2", "Q3", "Q4", "Q5_largest"],
            OUTCOME_NAMES,
        ],
        "metadata": {
            "task_description": "Decode active visual change-detection stimulus identity/change, running and pupil quintiles, and contingent trial outcome from inferred calcium events.",
            "time_bin_size": DT * 1000.0,
            "temporal_alignment_event": "Variable-length trial start on the common synchronized clock; bins follow ophys timestamps.",
            "off_start": None,
            "off_end": None,
            "neural_signal": "AllenSDK unfiltered FastLZero inferred event magnitudes summed in 100 ms bins",
            "trial_selection": "active_behavior; go or catch; not aborted; not auto_rewarded; exactly one contingent outcome; complete synchronized coverage",
            "running_quintile_edges_cm_per_s": run_edges.tolist(),
            "pupil_quintile_edges_equivalent_diameter_pixels": pupil_edges.tolist(),
            "pupil_definition": "2*sqrt(processed pupil_area/pi); blink/outlier samples masked upstream by SDK, internal gaps linearly interpolated",
            "image_identity_values": image_values,
            "gray_definition": "inter-stimulus gray periods and omitted presentations",
            "region_aliases": {"VISp": "V1/primary visual cortex", "VISl": "LM/lateral visual area"},
            "session_info": session_info,
            "source": "Allen Visual Behavior Ophys release 1.1.0 loaded exclusively with VisualBehaviorOphysProjectCache",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="process all active cached experiments (default)")
    mode.add_argument("--sample", action="store_true", help="process two active cached experiments")
    parser.add_argument("--show-processing", action="store_true")
    args = parser.parse_args()
    total_start = time.perf_counter()
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=str(CACHE_DIR))
    table = active_experiment_table(cache)
    if args.sample:
        table = table.iloc[:2]
    print(f"Selected {len(table)} active cached experiments", flush=True)
    sessions, excluded = [], {}
    use_parallel = not args.sample and not args.show_processing and len(table) > 2
    if use_parallel:
        workers = min(4, len(table))
        print(f"Using {workers} worker processes (one independent AllenSDK cache per worker)", flush=True)
        items = [(int(eid), meta.to_dict()) for eid, meta in table.iterrows()]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, (experiment_id, session, exclusions) in enumerate(pool.map(convert_experiment_worker, items), start=1):
                print(f"[{i}/{len(table)}] collected experiment {experiment_id}", flush=True)
                excluded[experiment_id] = exclusions
                if session is not None:
                    sessions.append(session)
    else:
        for i, (experiment_id, meta) in enumerate(table.iterrows(), start=1):
            print(f"[{i}/{len(table)}] loading via AllenSDK", flush=True)
            try:
                session, exclusions = convert_experiment(
                    cache, int(experiment_id), meta,
                    keep_diagnostic=args.show_processing and i <= 2,
                )
                excluded[int(experiment_id)] = exclusions
                if session is not None:
                    sessions.append(session)
            except Exception as exc:
                excluded[int(experiment_id)] = {"fatal_error": repr(exc)}
                print(f"  experiment {experiment_id}: ERROR {exc!r}", flush=True)
    if not sessions:
        raise RuntimeError("No sessions survived conversion")
    print(f"Assembling {len(sessions)} sessions and pooled percentile bins", flush=True)
    data = assemble(sessions, args.show_processing)
    data["metadata"]["all_experiment_exclusions"] = excluded
    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    with args.outpicklefile.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    ntrials = sum(map(len, data["neural"]))
    ncells = sum(x[0].shape[0] for x in data["neural"])
    elapsed = time.perf_counter() - total_start
    print(f"Wrote {args.outpicklefile}: {len(sessions)} sessions, {ntrials} trials, {ncells} cell entries", flush=True)
    print(f"Total conversion time: {elapsed:.2f}s ({elapsed/len(table):.2f}s/selected experiment)", flush=True)


if __name__ == "__main__":
    main()
