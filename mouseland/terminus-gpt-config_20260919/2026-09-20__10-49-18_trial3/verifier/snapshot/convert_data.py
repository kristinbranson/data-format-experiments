#!/usr/bin/env python3
"""Convert Zhong et al. imaging data to decoder-compatible trial dictionaries.

Usage: python -u /app/convert_data.py OUT.pkl [--full|--sample] [--show-processing]
"""
import argparse
import gc
import glob
import os
import pickle
import re
import sys
import time
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np

ROOT = "/app/data"
STIM_VALUES = [
    "circle1", "circle2", "circle3", "leaf1", "leaf2", "leaf3",
    "leaf1_swap1", "leaf1_swap2",
]
STIM_TO_ID = {v: i for i, v in enumerate(STIM_VALUES)}
REGIONS = ["V1", "mHV", "lHV", "aHV", "unassigned"]


def physical_id(key):
    """Remove analysis-view suffixes, leaving mouse_date_block."""
    return re.sub(r"_swap[12]$", "", key)


def parse_session_id(sid):
    """Parse <mouse>_<YYYY>_<MM>_<DD>_<block> without splitting date underscores."""
    m = re.fullmatch(r"(.+?)_(\d{4}_\d{2}_\d{2})_([^_]+)", sid)
    if not m:
        raise ValueError(f"Malformed physical session ID: {sid!r}")
    return m.group(1), m.group(2), m.group(3)


def neural_files():
    return {
        os.path.basename(p).removesuffix("_neural_data.npy"): p
        for p in glob.glob(os.path.join(ROOT, "spk", "*_neural_data.npy"))
    }


def load_behavior_views(valid_ids):
    """Load all analysis views and group them by unique physical recording."""
    views = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(ROOT, "beh", "Beh_*.npy"))):
        group = os.path.basename(path)[4:-4]
        obj = np.load(path, allow_pickle=True).item()
        for key, beh in obj.items():
            sid = physical_id(key)
            if sid in valid_ids:
                views[sid].append((group, key, beh))
        del obj
        gc.collect()
    missing = sorted(set(valid_ids) - set(views))
    if missing:
        raise RuntimeError(f"No behavior found for sessions: {missing}")
    return views


def merged_stimuli(session_views):
    """Merge semantic TrialStim labels across duplicate analysis views."""
    n = int(session_views[0][2]["ntrials"])
    for _, _, b in session_views:
        if int(b["ntrials"]) != n:
            raise ValueError("Duplicate behavior views have different trial counts")
    labels = []
    for i in range(n):
        concrete = {
            str(b["TrialStim"][i]) for _, _, b in session_views
            if str(b["TrialStim"][i]) != "stimulus_of_trial"
        }
        if len(concrete) > 1:
            raise ValueError(f"Conflicting stimulus labels at trial {i}: {concrete}")
        if concrete:
            label = next(iter(concrete))
        else:
            wall = {str(b["WallName"][i]) for _, _, b in session_views}
            if len(wall) != 1:
                raise ValueError(f"Conflicting WallName fallback at trial {i}: {wall}")
            label = next(iter(wall))
        if label not in STIM_TO_ID:
            raise ValueError(f"Unknown visual stimulus category {label!r}")
        labels.append(label)
    return np.asarray(labels)


def choose_behavior(session_views):
    """Choose one physical stream; duplicate views differ only in semantic labels."""
    b0 = session_views[0][2]
    n = int(b0["ntrials"])
    nf = len(b0["ft"])
    for _, _, b in session_views[1:]:
        if int(b["ntrials"]) != n or len(b["ft"]) != nf:
            raise ValueError("Duplicate views differ in physical stream dimensions")
        # Core physical streams must be equivalent.
        for key in ("StartFr", "GrayFr", "SoundFr", "isRew", "ft_Pos", "ft_RunSpeed"):
            if not np.allclose(np.asarray(b0[key]), np.asarray(b[key]), equal_nan=True):
                raise ValueError(f"Duplicate views differ in {key}")
    return b0


def session_dt_seconds(beh):
    """Convert median MATLAB-datenum frame interval from days to seconds."""
    d = np.diff(np.asarray(beh["ft"], dtype=np.float64))
    d = d[np.isfinite(d) & (d > 0)]
    if not len(d):
        raise ValueError("No positive frame intervals")
    dt = float(np.median(d) * 86400.0)
    if not (0.2 < dt < 0.5):
        raise ValueError(f"Unexpected imaging frame interval {dt} s")
    return dt


def trial_bounds(beh, nfr):
    starts = np.ceil(np.asarray(beh["StartFr"], dtype=np.float64)).astype(np.int64)
    ends = np.ceil(np.asarray(beh["GrayFr"], dtype=np.float64)).astype(np.int64)
    starts = np.clip(starts, 0, nfr)
    ends = np.clip(ends, 0, nfr)
    if np.any(ends <= starts):
        bad = np.flatnonzero(ends <= starts)
        raise ValueError(f"Invalid trial windows: {bad[:20].tolist()}")
    return starts, ends


def compute_speed_thresholds(selected_ids, views, nfr_by_session):
    """Global quartiles over exactly the retained corridor imaging frames."""
    chunks = []
    for sid in selected_ids:
        b = choose_behavior(views[sid])
        nfr = min(int(nfr_by_session[sid]), len(b["ft_RunSpeed"]))
        starts, ends = trial_bounds(b, nfr)
        speed = np.asarray(b["ft_RunSpeed"][:nfr], dtype=np.float64)
        chunks.extend(speed[a:z] for a, z in zip(starts, ends))
    values = np.concatenate(chunks)
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite retained running speeds")
    q = np.quantile(values, [0.25, 0.5, 0.75]).astype(np.float64)
    return q, int(values.size), (float(values.min()), float(values.max()))


def map_regions(iarea):
    """Exact reference neu_area_ID mapping plus explicit unassigned class."""
    a = np.asarray(iarea)
    out = np.full(a.shape, 4, dtype=np.int16)
    out[a == 8] = 0
    out[np.isin(a, [0, 1, 2, 9])] = 1
    out[np.isin(a, [5, 6])] = 2
    out[np.isin(a, [3, 4])] = 3
    return out


def retinotopy_path(sid):
    mouse, date, _block = parse_session_id(sid)
    return os.path.join(ROOT, "retinotopy", f"{mouse}_{date}_trans.npz")


def metadata_by_session():
    info = np.load(os.path.join(ROOT, "beh", "Imaging_Exp_info.npy"), allow_pickle=True).item()
    out = defaultdict(list)
    for group, entries in info.items():
        for d in entries:
            sid = f"{d['mname']}_{d['datexp']}_{d['blk']}"
            out[sid].append({
                "experiment_group": group,
                "session_number": d.get("sess#"),
                "experiment_type": d.get("exptype"),
                "reward_type": d.get("rewType"),
                "gender": d.get("Gender"),
            })
    return out


def subject_day_values(session_ids):
    first = {}
    dates = {}
    for sid in session_ids:
        mouse, date, _ = parse_session_id(sid)
        d = datetime.strptime(date, "%Y_%m_%d").date()
        dates[sid] = d
        first[mouse] = min(first.get(mouse, d), d)
    return {sid: float((dates[sid] - first[sid.split("_")[0]]).days) for sid in session_ids}


def inspect_neural_shapes(paths):
    """Load one session at a time to obtain reference-concatenated dimensions."""
    shapes = {}
    dtypes = {}
    t0 = time.time()
    for j, sid in enumerate(paths, 1):
        obj = np.load(paths[sid], allow_pickle=True).item()
        parts = obj["spks"]
        if len(parts) != 3 or any(x.ndim != 2 for x in parts):
            raise ValueError(f"Unexpected spks structure for {sid}")
        nfrs = {x.shape[1] for x in parts}
        if len(nfrs) != 1:
            raise ValueError(f"Neural components have unequal frame counts for {sid}")
        shapes[sid] = (sum(x.shape[0] for x in parts), parts[0].shape[1])
        dtypes[sid] = [str(x.dtype) for x in parts]
        del obj, parts
        gc.collect()
        print(f"Shape scan {j}/{len(paths)} {sid}: {shapes[sid]}", flush=True)
    print(f"Shape scan finished in {time.time()-t0:.1f} s", flush=True)
    return shapes, dtypes


def make_lick_raster(lick_frames, start, end):
    out = np.zeros(end - start, dtype=np.int16)
    idx = np.floor(np.asarray(lick_frames, dtype=np.float64)).astype(np.int64)
    idx = idx[(idx >= start) & (idx < end)] - start
    if len(idx):
        out[np.unique(idx)] = 1
    return out


def convert_session(sid, path, session_views, shape, speed_q, day_value):
    t0 = time.time()
    beh = choose_behavior(session_views)
    stimuli = merged_stimuli(session_views)
    obj = np.load(path, allow_pickle=True).item()
    parts = obj["spks"]
    # Preserve exact reference load_spk neuron order, but concatenate only retained
    # trial slices. This avoids an unnecessary multi-GB full-session copy.
    if len(parts) != 3 or len({x.shape[1] for x in parts}) != 1:
        raise ValueError(f"Unexpected neural components for {sid}")
    nneu, nfr = sum(x.shape[0] for x in parts), parts[0].shape[1]
    if shape is not None and (nneu, nfr) != tuple(shape):
        raise ValueError(f"Neural shape changed for {sid}: {(nneu,nfr)} != {shape}")
    starts, ends = trial_bounds(beh, nfr)
    dt = session_dt_seconds(beh)
    pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
    speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
    sound = np.asarray(beh["SoundFr"], dtype=np.float64)
    reward = np.asarray(beh["isRew"], dtype=np.int16)
    lick_frames = np.asarray(beh["LickFr"], dtype=np.float64)
    lick_session = np.zeros(nfr, dtype=np.int16)
    lick_idx = np.floor(lick_frames).astype(np.int64)
    lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]
    if len(lick_idx):
        lick_session[np.unique(lick_idx)] = 1

    neural_trials, input_trials, output_trials = [], [], []
    for i, (a, z) in enumerate(zip(starts, ends)):
        frames = np.arange(a, z, dtype=np.float64)
        T = z - a
        # Equivalent to reference concatenate(parts, axis=0)[:, a:z], without
        # materializing excluded inter-trial and gray-space frames.
        nt = np.concatenate([x[:, a:z] for x in parts], axis=0).astype(np.float32, copy=False)
        inp = np.empty((4, T), dtype=np.float32)
        inp[0] = (sound[i] - frames) * dt
        inp[1] = day_value
        inp[2] = (frames - float(beh["StartFr"][i])) * dt
        inp[3] = reward[i]
        out = np.empty((4, T), dtype=np.int16)
        out[0] = STIM_TO_ID[str(stimuli[i])]
        out[1] = lick_session[a:z]
        # Source visual corridor is 0..40 (= four physical metres).
        out[2] = np.clip(np.digitize(pos[a:z], [10.0, 20.0, 30.0]), 0, 3)
        out[3] = np.clip(np.digitize(speed[a:z], speed_q), 0, 3)
        if not (nt.shape[1] == inp.shape[1] == out.shape[1]):
            raise AssertionError("Trial stream length mismatch")
        neural_trials.append(nt)
        input_trials.append(inp)
        output_trials.append(out)
    del obj, parts
    gc.collect()

    rp = retinotopy_path(sid)
    with np.load(rp) as r:
        region_idx = map_regions(r["iarea"])
    if len(region_idx) != nneu:
        raise ValueError(f"Retinotopy/neural mismatch for {sid}: {len(region_idx)} != {nneu}")
    elapsed = time.time() - t0
    retained = sum(x.shape[1] for x in neural_trials)
    print(f"Converted {sid}: {nneu} neurons, {len(neural_trials)} trials, "
          f"{retained} trial frames, dt={dt*1000:.2f} ms, {elapsed:.1f} s", flush=True)
    stats = {
        "session_id": sid, "n_neurons": nneu, "n_neural_frames": nfr,
        "n_trials": len(neural_trials), "retained_trial_frames": retained,
        "frame_bin_ms": dt * 1000.0,
        "stimulus_counts": dict(Counter(map(str, stimuli))),
        "rewarded_trials": int(reward.sum()),
    }
    return neural_trials, input_trials, output_trials, region_idx, stats


def plot_processing(sid, neural_trials, input_trials, output_trials, speed_q):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nshow = min(5, len(neural_trials))
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), constrained_layout=True)
    trial = min(2, nshow - 1)
    n = neural_trials[trial]
    # Plot a deterministic spread of up to 100 neurons.
    ix = np.linspace(0, n.shape[0] - 1, min(100, n.shape[0])).astype(int)
    axes[0].imshow(n[ix], aspect="auto", interpolation="nearest", cmap="viridis")
    axes[0].set_title(f"{sid}: raw deconvolved activity (100 sampled neurons), trial {trial}")
    axes[1].plot(input_trials[trial][0], label="time to cue (s)")
    axes[1].plot(input_trials[trial][2], label="time since entry (s)")
    axes[1].axhline(0, color="k", lw=.5); axes[1].legend(); axes[1].set_title("Temporal alignment")
    axes[2].step(np.arange(n.shape[1]), output_trials[trial][1], where="mid", label="lick")
    axes[2].step(np.arange(n.shape[1]), output_trials[trial][2], where="mid", label="position bin")
    axes[2].step(np.arange(n.shape[1]), output_trials[trial][3], where="mid", label="speed bin")
    axes[2].legend(); axes[2].set_title("Categorical time-varying outputs")
    axes[3].hist(np.concatenate([o[2] for o in output_trials]), bins=np.arange(5)-.5)
    axes[3].set_xticks(range(4)); axes[3].set_title("Position-bin distribution")
    axes[4].hist(np.concatenate([o[3] for o in output_trials]), bins=np.arange(5)-.5)
    axes[4].set_xticks(range(4)); axes[4].set_title(f"Speed bins; thresholds={speed_q.tolist()}")
    out = f"/app/processing_{sid}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved processing plot {out}", flush=True)


def validate_local(data):
    ns = len(data["neural"])
    assert ns == len(data["input"]) == len(data["output"])
    assert len(data["subject_idx"]) == ns == len(data["brain_region_idx"])
    total_trials = 0
    for s in range(ns):
        assert len(data["neural"][s]) >= 2
        assert len(data["neural"][s]) == len(data["input"][s]) == len(data["output"][s])
        assert len(data["brain_region_idx"][s]) == data["neural"][s][0].shape[0]
        for n, i, o in zip(data["neural"][s], data["input"][s], data["output"][s]):
            assert n.ndim == i.ndim == o.ndim == 2
            assert n.shape[1] == i.shape[1] == o.shape[1]
            assert i.shape[0] == 4 and o.shape[0] == 4
            assert np.all(np.isfinite(n)) and np.all(np.isfinite(i))
            assert o.min() >= 0
            assert o[0].max() < len(STIM_VALUES) and o[1].max() < 2
            assert o[2].max() < 4 and o[3].max() < 4
            total_trials += 1
    print(f"Local validation passed: {ns} sessions, {total_trials} trials", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outpicklefile")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="process all sessions (default)")
    mode.add_argument("--sample", action="store_true", help="process two smallest sessions")
    ap.add_argument("--show-processing", action="store_true")
    args = ap.parse_args()
    t_all = time.time()
    paths_all = neural_files()
    if len(paths_all) != 89:
        raise RuntimeError(f"Expected 89 unique neural sessions, found {len(paths_all)}")
    if args.sample:
        # Smallest estimated converted sessions that each contain rewarded and
        # unrewarded trials plus corridor licks, so every decoder output varies.
        selected = ["TX60_2021_04_10_1", "TX109_2023_03_27_1"]
        if any(s not in paths_all for s in selected):
            raise RuntimeError("Representative sample sessions are missing")
        mode_name = "sample"
    else:
        selected = sorted(paths_all)
        mode_name = "full"
    paths = {sid: paths_all[sid] for sid in selected}
    print(f"Mode={mode_name}; selected {len(selected)} sessions", flush=True)
    print("Sessions:", selected, flush=True)

    all_views = load_behavior_views(paths_all.keys())
    if args.sample:
        # Explicit pre-scan is useful for sample diagnostics. Full mode avoids
        # reading all 434 GB twice; shapes are inferred during the conversion load.
        shapes, source_dtypes = inspect_neural_shapes(paths)
        nfr_for_speed = {sid: shapes[sid][1] for sid in selected}
    else:
        shapes = {sid: None for sid in selected}
        source_dtypes = {sid: ["float32", "float32", "float32"] for sid in selected}
        # All GrayFr trial ends precede both neural and behavior stream ends;
        # the behavior stream is only 0-2 trailing frames longer.
        nfr_for_speed = {sid: len(choose_behavior(all_views[sid])["ft"]) for sid in selected}
    speed_q, n_speed, speed_range = compute_speed_thresholds(selected, all_views, nfr_for_speed)
    print(f"Speed quartiles from {n_speed} retained frames: {speed_q}; range={speed_range}", flush=True)
    day = subject_day_values(sorted(paths_all))
    meta_views = metadata_by_session()
    subjects = sorted({sid.split("_")[0] for sid in selected})
    subject_map = {v: i for i, v in enumerate(subjects)}

    data = {
        "neural": [], "input": [], "output": [],
        "subjects": subjects,
        "subject_idx": np.asarray([subject_map[s.split("_")[0]] for s in selected], dtype=np.int64),
        "brain_regions": REGIONS,
        "brain_region_idx": [],
        "input_names": ["time_to_sound_cue_s", "day_of_training", "time_since_trial_start_s", "reward_availability"],
        "output_names": ["visual_stimulus_category", "licking", "corridor_position_bin", "running_speed_quartile"],
        "output_values": [
            STIM_VALUES,
            ["not_licking", "licking"],
            ["0-1 m", "1-2 m", "2-3 m", "3-4 m"],
            ["0-25%", "25-50%", "50-75%", "75-100%"],
        ],
        "metadata": {},
    }
    session_stats = []
    for idx, sid in enumerate(selected):
        n, i, o, r, st = convert_session(
            sid, paths[sid], all_views[sid], shapes[sid], speed_q, day[sid]
        )
        data["neural"].append(n); data["input"].append(i); data["output"].append(o)
        data["brain_region_idx"].append(r)
        st["subject"] = sid.split("_")[0]
        st["day_from_subject_first_recording"] = day[sid]
        st["experiment_views"] = meta_views.get(sid, [])
        st["source_neural_dtypes"] = source_dtypes[sid]
        session_stats.append(st)
        if args.show_processing and idx < 2:
            plot_processing(sid, n, i, o, speed_q)

    dt_values = [x["frame_bin_ms"] for x in session_stats]
    data["metadata"] = {
        "task_description": (
            "Decode visual stimulus category, binary licking, four 1-m corridor position bins, "
            "and global running-speed quartiles from Suite2p deconvolved visual-cortex activity."
        ),
        "time_bin_size": float(np.median(dt_values)),
        "temporal_alignment_event": "visual corridor entry (StartFr; first included frame is ceil(StartFr))",
        "off_start": 0.0,
        "off_end": None,
        "trial_window": "[ceil(StartFr), ceil(GrayFr)); visual 4-m corridor only; variable duration",
        "source_position_scale": "15 source units per metre; visual corridor 0-40 source units",
        "speed_quartile_thresholds": speed_q.tolist(),
        "speed_threshold_population_frames": n_speed,
        "speed_source_range": list(speed_range),
        "day_definition": "calendar days since each subject's earliest recording among all 89 recordings",
        "neural_processing": "reference load_spk concatenation of all three released Suite2p deconvolved trace arrays; no additional normalization",
        "lick_rasterization": "binary presence; fractional LickFr assigned to floor(LickFr)",
        "session_info": session_stats,
        "conversion_mode": mode_name,
    }
    validate_local(data)
    os.makedirs(os.path.dirname(os.path.abspath(args.outpicklefile)), exist_ok=True)
    print(f"Writing {args.outpicklefile} with pickle protocol 4...", flush=True)
    tw = time.time()
    with open(args.outpicklefile, "wb") as f:
        pickle.dump(data, f, protocol=4)
    size = os.path.getsize(args.outpicklefile)
    print(f"Wrote {size/1e9:.3f} GB in {time.time()-tw:.1f} s", flush=True)
    print(f"Total conversion time: {time.time()-t_all:.1f} s", flush=True)


if __name__ == "__main__":
    main()
