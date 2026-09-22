#!/usr/bin/env python3
"""Convert cached Allen Visual Behavior 2P data for neural decoding.

All detailed data access is through VisualBehaviorOphysProjectCache.  This
script deliberately never opens an NWB file with h5py/pynwb.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


APP_DIR = Path("/app")
DATA_DIR = APP_DIR / "data"
SDK_DIR = APP_DIR / "code"
FS = 30.0
DT = 1.0 / FS
IMAGE_VALUES = [
    "gray", "im000", "im031", "im035", "im045", "im054", "im061",
    "im062", "im063", "im065", "im066", "im069", "im073", "im075",
    "im077", "im085", "im106",
]
IMAGE_TO_INT = {name: idx for idx, name in enumerate(IMAGE_VALUES)}
OUTCOME_COLUMNS = ["hit", "miss", "false_alarm", "correct_reject"]


class MissingRequiredData(RuntimeError):
    """A published session cannot supply one of the required decoder outputs."""


def _import_cache_class():
    """Import the supplied SDK checkout, preferring it over site packages."""
    sdk = str(SDK_DIR)
    if sdk not in sys.path:
        sys.path.insert(0, sdk)
    from allensdk.brain_observatory.behavior.behavior_project_cache.behavior_project_cache import (  # noqa: E501
        VisualBehaviorOphysProjectCache,
    )
    return VisualBehaviorOphysProjectCache


def make_cache():
    cls = _import_cache_class()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls.from_s3_cache(cache_dir=DATA_DIR)


def local_experiment_ids(cache) -> list[int]:
    """Return active exact-VisualBehavior experiments present on local disk."""
    table = cache.get_ophys_experiment_table()
    file_ids = {
        int(path.stem.rsplit("_", 1)[1])
        for path in (DATA_DIR / "visual-behavior-ophys-1.1.0" /
                     "behavior_ophys_experiments").glob("*.nwb")
    }
    keep = (
        table.index.isin(file_ids)
        & table["project_code"].eq("VisualBehavior")
        & ~table["passive"].astype(bool)
    )
    return sorted(table.index[keep].astype(int).tolist())


def finite_interp(source_t, source_x, target_t, label: str):
    """Linearly interpolate finite samples, requiring useful coverage."""
    source_t = np.asarray(source_t, dtype=np.float64)
    source_x = np.asarray(source_x, dtype=np.float64)
    good = np.isfinite(source_t) & np.isfinite(source_x)
    if good.sum() < 2:
        raise ValueError(f"fewer than two finite {label} samples")
    t = source_t[good]
    x = source_x[good]
    order = np.argsort(t, kind="stable")
    t, x = t[order], x[order]
    unique = np.r_[True, np.diff(t) > 0]
    t, x = t[unique], x[unique]
    if len(t) < 2:
        raise ValueError(f"fewer than two unique-time {label} samples")
    return np.interp(target_t, t, x).astype(np.float32)


def interp_rows_shared_time(source_t: np.ndarray, source_x: np.ndarray,
                            target_t: np.ndarray) -> np.ndarray:
    """Linear interpolation for many rows with one shared timestamp lookup."""
    source_t = np.asarray(source_t, dtype=np.float64)
    source_x = np.asarray(source_x, dtype=np.float32)
    if len(source_t) < 2 or np.any(np.diff(source_t) <= 0):
        raise ValueError("ophys timestamps must be strictly increasing")
    right = np.searchsorted(source_t, target_t, side="right")
    right = np.clip(right, 1, len(source_t) - 1)
    left = right - 1
    denom = source_t[right] - source_t[left]
    weight = ((target_t - source_t[left]) / denom).astype(np.float32)
    # One advanced-index lookup per side; unlike looping np.interp, timestamp
    # bracketing is performed once rather than once for every neuron.
    lo = source_x[:, left]
    hi = source_x[:, right]
    out = lo + (hi - lo) * weight[None, :]
    before = target_t <= source_t[0]
    after = target_t >= source_t[-1]
    if before.any():
        out[:, before] = source_x[:, [0]]
    if after.any():
        out[:, after] = source_x[:, [-1]]
    return out.astype(np.float32, copy=False)


def _trial_grid(start: float, stop: float) -> np.ndarray:
    """30 Hz relative grid on the half-open SDK trial interval."""
    n = int(np.ceil((stop - start) * FS - 1e-10))
    grid = start + np.arange(max(0, n), dtype=np.float64) * DT
    return grid[grid < stop]


def extract_session(experiment_id: int) -> dict:
    """Load and resample one experiment, returning output-ready intermediates."""
    t0 = time.perf_counter()
    warnings.filterwarnings("ignore")
    cache = make_cache()
    exp = cache.get_behavior_ophys_experiment(int(experiment_id))
    meta = exp.metadata
    if meta["project_code"] != "VisualBehavior":
        raise ValueError(f"{experiment_id}: unexpected project {meta['project_code']}")

    trials = exp.trials.sort_values("start_time")
    eligible = (
        (trials["go"] | trials["catch"])
        & ~trials["aborted"]
        & ~trials["auto_rewarded"]
    )
    trials = trials.loc[eligible].copy()
    if len(trials) < 2:
        raise ValueError(f"{experiment_id}: fewer than two eligible trials")

    outcome_bool = trials[OUTCOME_COLUMNS].to_numpy(dtype=bool)
    if not np.all(outcome_bool.sum(axis=1) == 1):
        bad = trials.index[outcome_bool.sum(axis=1) != 1].tolist()
        raise ValueError(f"{experiment_id}: nonexclusive outcomes in {bad[:5]}")
    outcomes = outcome_bool.argmax(axis=1).astype(np.int16)

    grids = [_trial_grid(float(r.start_time), float(r.stop_time))
             for r in trials.itertuples()]
    if any(len(g) == 0 for g in grids):
        raise ValueError(f"{experiment_id}: empty eligible trial grid")
    lengths = np.asarray([len(g) for g in grids], dtype=np.int32)
    offsets = np.r_[0, np.cumsum(lengths)]
    all_t = np.concatenate(grids)

    event_table = exp.events
    cell_ids = event_table.index.to_numpy(copy=True)
    events = np.vstack(event_table["events"].to_numpy()).astype(np.float32)
    ophys_t = np.asarray(exp.ophys_timestamps, dtype=np.float64)
    if events.shape[1] != len(ophys_t):
        raise ValueError(f"{experiment_id}: event/timestamp length mismatch")
    # np.interp is fast and memory-bounded; interpolate every cell once onto all
    # requested trial timestamps (which are monotonic because trials are sorted).
    neural_all = interp_rows_shared_time(ophys_t, events, all_t)
    neural = [neural_all[:, offsets[i]:offsets[i + 1]]
              for i in range(len(lengths))]

    running = exp.running_speed
    try:
        run_all = finite_interp(running["timestamps"], running["speed"],
                                all_t, "running speed")
    except ValueError as exc:
        raise MissingRequiredData(str(exc)) from exc

    eye = exp.eye_tracking
    if eye is None or eye.empty:
        raise MissingRequiredData("eye tracking table is missing or empty")
    # SDK/whitepaper define width and height as ellipse half-axes and diameter
    # as the major full axis. Cleaned fields are NaN for likely blinks.
    pupil_diameter = 2.0 * np.maximum(
        eye["pupil_width"].to_numpy(dtype=np.float64),
        eye["pupil_height"].to_numpy(dtype=np.float64),
    )
    try:
        pupil_all = finite_interp(eye["timestamps"], pupil_diameter,
                                  all_t, "pupil diameter")
    except ValueError as exc:
        raise MissingRequiredData(str(exc)) from exc

    image_all = np.zeros(len(all_t), dtype=np.int16)
    change_all = np.zeros(len(all_t), dtype=np.int16)
    stim = exp.stimulus_presentations
    task = stim[
        stim["stimulus_block_name"].str.contains("change_detection", na=False)
    ].sort_values("start_time")
    for row in task.itertuples():
        start = float(row.start_time)
        end = float(row.end_time)
        lo = int(np.searchsorted(all_t, start, side="left"))
        hi = int(np.searchsorted(all_t, end, side="left"))
        # Concatenated trial grids have gaps; interval condition prevents a
        # stimulus in an excluded gap from contaminating the next kept trial.
        if hi > lo and not bool(row.omitted):
            idx = np.arange(lo, hi)
            valid = (all_t[idx] >= start) & (all_t[idx] < end)
            name = str(row.image_name)
            if name not in IMAGE_TO_INT:
                raise ValueError(f"{experiment_id}: unknown image {name}")
            image_all[idx[valid]] = IMAGE_TO_INT[name]
            # "Right after" an identity change is the changed-image flash,
            # not only an arbitrarily narrow single sample at its leading edge.
            if bool(row.is_change):
                change_all[idx[valid]] = 1

    result = {
        "experiment_id": int(experiment_id),
        "ophys_session_id": int(meta["ophys_session_id"]),
        "mouse_id": str(meta["mouse_id"]),
        "region": str(meta["targeted_structure"]),
        "session_type": str(meta["session_type"]),
        "image_set": str(meta["session_type"]).split("images_")[-1][:1],
        "experience_level": "Familiar" if "images_A" in str(meta["session_type"])
                            else "Novel",
        "cell_ids": cell_ids,
        "trial_ids": trials.index.to_numpy(copy=True),
        "trial_starts": trials["start_time"].to_numpy(dtype=np.float64),
        "trial_stops": trials["stop_time"].to_numpy(dtype=np.float64),
        "lengths": lengths,
        "neural": neural,
        "running": run_all,
        "pupil": pupil_all,
        "image": image_all,
        "change": change_all,
        "outcome": outcomes,
        "source_trial_count": int(len(exp.trials)),
        "eligible_trial_count": int(len(trials)),
        "go_count": int(trials["go"].sum()),
        "catch_count": int(trials["catch"].sum()),
        "outcome_counts": outcome_bool.sum(axis=0).astype(int).tolist(),
        "ophys_frame_rate": float(meta["ophys_frame_rate"]),
        "seconds": time.perf_counter() - t0,
    }
    return result


def percentile_edges(values: list[np.ndarray], label: str) -> np.ndarray:
    joined = np.concatenate(values).astype(np.float32, copy=False)
    if not np.all(np.isfinite(joined)):
        raise ValueError(f"nonfinite {label} values after interpolation")
    edges = np.quantile(joined, [0.2, 0.4, 0.6, 0.8]).astype(np.float64)
    if np.any(np.diff(edges) <= 0):
        raise ValueError(f"non-unique {label} quintile edges: {edges}")
    return edges


def split_vector(x: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    offsets = np.r_[0, np.cumsum(lengths)]
    return [x[offsets[i]:offsets[i + 1]] for i in range(len(lengths))]


def finalize(raw_sessions: list[dict], run_edges, pupil_edges,
             exclusions: list[dict] | None = None) -> dict:
    subjects = sorted({s["mouse_id"] for s in raw_sessions})
    subject_map = {name: i for i, name in enumerate(subjects)}
    regions = sorted({s["region"] for s in raw_sessions})
    region_map = {name: i for i, name in enumerate(regions)}
    neural, inputs, outputs, region_idx, session_info = [], [], [], [], []

    for s in raw_sessions:
        lengths = s["lengths"]
        run_bin = np.searchsorted(run_edges, s["running"], side="right").astype(np.int16)
        pupil_bin = np.searchsorted(pupil_edges, s["pupil"], side="right").astype(np.int16)
        run_trials = split_vector(run_bin, lengths)
        pupil_trials = split_vector(pupil_bin, lengths)
        image_trials = split_vector(s["image"], lengths)
        change_trials = split_vector(s["change"], lengths)
        out_trials = []
        in_trials = []
        for i, length in enumerate(lengths):
            out_trials.append(np.vstack([
                image_trials[i], change_trials[i], run_trials[i], pupil_trials[i],
                np.full(int(length), s["outcome"][i], dtype=np.int16),
            ]).astype(np.int16, copy=False))
            in_trials.append(np.empty((0, int(length)), dtype=np.float32))
        neural.append(s["neural"])
        inputs.append(in_trials)
        outputs.append(out_trials)
        region_idx.append(np.full(len(s["cell_ids"]), region_map[s["region"]],
                                  dtype=np.int32))
        session_info.append({k: s[k] for k in [
            "experiment_id", "ophys_session_id", "mouse_id", "region",
            "session_type", "image_set", "experience_level", "source_trial_count",
            "eligible_trial_count", "go_count", "catch_count", "outcome_counts",
            "ophys_frame_rate",
        ]})

    return {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": np.asarray([subject_map[s["mouse_id"]] for s in raw_sessions],
                                  dtype=np.int32),
        "brain_regions": regions,
        "brain_region_idx": region_idx,
        "input_names": [],
        "output_names": [
            "image_identity", "image_change", "running_speed_quintile",
            "pupil_diameter_quintile", "trial_outcome",
        ],
        "output_values": [
            IMAGE_VALUES,
            ["no_change", "change"],
            ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
            ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
            ["hit", "miss", "false_alarm", "correct_reject"],
        ],
        "metadata": {
            "task_description": (
                "Decode flashed natural-image identity, image-change events, running "
                "and pupil quintiles, and go/catch trial outcome from VISp calcium events."
            ),
            "time_bin_size": 1000.0 / FS,
            "temporal_alignment_event": "trial start (AllenSDK trials.start_time)",
            "off_start": 0.0,
            "off_end": None,
            "sampling_rate_hz": FS,
            "neural_signal": "AllenSDK raw L0 calcium event magnitude",
            "neural_resampling": "linear interpolation from synchronized ophys timestamps",
            "trial_interval": "half-open [start_time, stop_time)",
            "trial_filter": "active go/catch; aborted and auto_rewarded excluded",
            "project_filter": "project_code == VisualBehavior; passive == False",
            "running_speed_quintile_edges_cm_per_s": run_edges.tolist(),
            "pupil_diameter_quintile_edges_pixels": pupil_edges.tolist(),
            "pupil_definition": "2 * max(cleaned pupil_width, cleaned pupil_height)",
            "image_change_definition": "1 throughout the 250 ms changed-image presentation",
            "session_info": session_info,
            "excluded_sessions": exclusions or [],
        },
    }


def validate_internal(data: dict):
    ns = len(data["neural"])
    assert ns == len(data["input"]) == len(data["output"])
    assert len(data["subject_idx"]) == ns == len(data["brain_region_idx"])
    for si in range(ns):
        assert len(data["neural"][si]) >= 2
        assert len(data["neural"][si]) == len(data["input"][si]) == len(data["output"][si])
        ncell = data["neural"][si][0].shape[0]
        assert len(data["brain_region_idx"][si]) == ncell and ncell > 0
        for n, x, y in zip(data["neural"][si], data["input"][si], data["output"][si]):
            assert n.dtype == np.float32 and n.ndim == 2
            assert x.dtype == np.float32 and x.shape == (0, n.shape[1])
            assert y.dtype == np.int16 and y.shape == (5, n.shape[1])
            assert np.all(np.isfinite(n)) and np.all(np.isfinite(y))
            assert np.all(y[4] == y[4, 0])
            assert y[0].min() >= 0 and y[0].max() < len(IMAGE_VALUES)
            assert y[1].min() >= 0 and y[1].max() <= 1
            assert y[2].min() >= 0 and y[2].max() <= 4
            assert y[3].min() >= 0 and y[3].max() <= 4
            assert y[4].min() >= 0 and y[4].max() <= 3


def plot_processing(raw_sessions: list[dict], data: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for si, raw in enumerate(raw_sessions[:2]):
        ti = min(2, len(raw["neural"]) - 1)
        neural = raw["neural"][ti]
        output = data["output"][si][ti]
        t = np.arange(neural.shape[1]) / FS
        fig, axes = plt.subplots(5, 1, figsize=(14, 14), constrained_layout=True)
        axes[0].plot(t, neural[:min(20, len(neural))].T, alpha=0.35)
        axes[0].plot(t, neural.mean(axis=0), color="black", lw=2, label="cell mean")
        axes[0].set_ylabel("raw L0 events")
        axes[0].legend()
        axes[0].set_title(
            f"Experiment {raw['experiment_id']}, trial {raw['trial_ids'][ti]}: "
            "ophys-timestamp interpolation to 30 Hz"
        )
        axes[1].step(t, output[0], where="post", label="image class")
        axes[1].stem(t[output[1] == 1], np.full((output[1] == 1).sum(), 17),
                     linefmt="r-", markerfmt="ro", basefmt=" ", label="change")
        axes[1].set_ylabel("stimulus class")
        axes[1].legend()
        run = split_vector(raw["running"], raw["lengths"])[ti]
        pupil = split_vector(raw["pupil"], raw["lengths"])[ti]
        axes[2].plot(t, run, label="interpolated cm/s")
        axes[2].step(t, output[2], where="post", label="quintile")
        axes[2].set_ylabel("running")
        axes[2].legend()
        axes[3].plot(t, pupil, label="cleaned/interpolated diameter")
        axes[3].step(t, output[3], where="post", label="quintile")
        axes[3].set_ylabel("pupil pixels")
        axes[3].legend()
        axes[4].step(t, output[4], where="post", label="static outcome")
        axes[4].set_ylabel("outcome class")
        axes[4].set_xlabel("seconds from trial start")
        axes[4].legend()
        fig.savefig(APP_DIR / f"processing_{raw['experiment_id']}.png", dpi=140)
        plt.close(fig)


def summarize(data: dict, elapsed: float):
    trials = [len(s) for s in data["neural"]]
    cells = [s[0].shape[0] for s in data["neural"]]
    frames = sum(x.shape[1] for s in data["neural"] for x in s)
    print(f"Converted sessions: {len(trials)}")
    print(f"Subjects: {len(data['subjects'])}; regions: {data['brain_regions']}")
    print(f"Trials: {sum(trials)} (per session {min(trials)}..{max(trials)}, mean {np.mean(trials):.2f})")
    print(f"Cell-session observations: {sum(cells)} (per session {min(cells)}..{max(cells)}, mean {np.mean(cells):.2f})")
    print(f"Total trial timepoints: {frames}; 30 Hz duration: {frames / FS / 3600:.2f} h")
    print(f"Running quintile edges: {data['metadata']['running_speed_quintile_edges_cm_per_s']}")
    print(f"Pupil quintile edges: {data['metadata']['pupil_diameter_quintile_edges_pixels']}")
    print(f"Total conversion time: {elapsed:.2f} s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("outpicklefile", type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="process all sessions (default)")
    mode.add_argument("--sample", action="store_true", help="process two sessions")
    p.add_argument("--show-processing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    started = time.perf_counter()
    cache = make_cache()
    ids = local_experiment_ids(cache)
    if args.sample:
        # Deterministic sessions from different mice/image sets when possible.
        table = cache.get_ophys_experiment_table().loc[ids]
        first = int(table.index[0])
        b = table[table["image_set"].eq("B")]
        second = int(b.index[0]) if len(b) else int(table.index[1])
        ids = [first, second]
    print(f"Selected {len(ids)} active exact-VisualBehavior experiments", flush=True)

    raw_sessions = []
    exclusions = []
    # Detailed SDK object construction includes compressed trace reads.  It is
    # I/O/decompression-bound on the full cache; moderate process parallelism
    # is substantially faster while remaining modest on the provided host.
    workers = min(16, len(ids), max(1, (os.cpu_count() or 2) // 2))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_session, eid): eid for eid in ids}
        for done, future in enumerate(as_completed(futures), 1):
            eid = futures[future]
            try:
                result = future.result()
            except MissingRequiredData as exc:
                exclusions.append({"experiment_id": int(eid), "reason": str(exc)})
                print(f"[{done}/{len(ids)}] {eid}: EXCLUDED ({exc})", flush=True)
                continue
            except Exception as exc:
                raise RuntimeError(f"experiment {eid} failed") from exc
            raw_sessions.append(result)
            print(
                f"[{done}/{len(ids)}] {eid}: {result['eligible_trial_count']} trials, "
                f"{len(result['cell_ids'])} cells, {result['seconds']:.2f}s",
                flush=True,
            )
    raw_sessions.sort(key=lambda x: x["experiment_id"])

    run_edges = percentile_edges([s["running"] for s in raw_sessions], "running")
    pupil_edges = percentile_edges([s["pupil"] for s in raw_sessions], "pupil")
    data = finalize(raw_sessions, run_edges, pupil_edges, exclusions=exclusions)
    validate_internal(data)
    if args.show_processing:
        plot_processing(raw_sessions, data)

    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    with args.outpicklefile.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    summarize(data, time.perf_counter() - started)
    print(f"Saved {args.outpicklefile} ({args.outpicklefile.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
