#!/usr/bin/env python3
"""Convert Zhong et al. imaging data to decoder-compatible trial data."""

from __future__ import annotations

import argparse
import gc
import pickle
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BEH = DATA / "beh"
SPK = DATA / "spk"
RET = DATA / "retinotopy"

STIMULI = [
    "circle1", "circle2", "leaf1", "leaf2", "leaf3",
    "leaf1_swap1", "leaf1_swap2",
]
REGIONS = ["V1", "mHV", "lHV", "aHV", "unmapped/non-visual"]
INPUT_NAMES = [
    "time_to_sound_cue_s", "day_of_training",
    "time_since_trial_start_s", "reward_available",
]
OUTPUT_NAMES = [
    "visual_stimulus_category", "licking",
    "corridor_position_bin", "running_speed_quartile",
]
OUTPUT_VALUES = [
    STIMULI,
    ["not_licking", "licking"],
    ["0-1_m", "1-2_m", "2-3_m", "3-4_m"],
    ["Q1_slowest", "Q2", "Q3", "Q4_fastest"],
]

# Validated over all 89 scalar object-NPY neural files. It permits an exact
# behavior-only prepass without loading 412 GB of neural matrices.
OBJECT_NPY_OVERHEAD = 473


def sid_from_key(key: str) -> str:
    parts = key.split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected behavior key: {key}")
    return "_".join(parts[:5])


def sid_date(sid: str) -> datetime:
    return datetime.strptime("_".join(sid.split("_")[1:4]), "%Y_%m_%d")


def spk_path(sid: str) -> Path:
    return SPK / f"{sid}_neural_data.npy"


def ret_path(sid: str) -> Path:
    return RET / f"{'_'.join(sid.split('_')[:4])}_trans.npz"


def plain(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    return x


def build_catalog() -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Deduplicate paper views and merge their complementary stimulus labels."""
    exp_info = np.load(BEH / "Imaging_Exp_info.npy", allow_pickle=True).item()
    record_meta: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for exp_type, records in exp_info.items():
        for rec in records:
            sid = f"{rec['mname']}_{rec['datexp']}_{rec['blk']}"
            record_meta[sid].append({"experiment_type": exp_type, **plain(rec)})

    entries: dict[str, dict[str, Any]] = {}
    mappings: dict[str, dict[str, int]] = defaultdict(dict)
    for exp_type in exp_info:
        behavior = np.load(BEH / f"Beh_{exp_type}.npy", allow_pickle=True).item()
        for key, beh in behavior.items():
            sid = sid_from_key(key)
            if sid not in entries:
                entries[sid] = {
                    "session_id": sid,
                    "subject": sid.split("_")[0],
                    "date": "_".join(sid.split("_")[1:4]),
                    "block": sid.split("_")[4],
                    "source_experiment": exp_type,
                    "source_behavior_key": key,
                    "experiment_labels": [],
                    "records": record_meta[sid],
                    "has_reward": bool(np.any(beh["isRew"])),
                }
            if exp_type not in entries[sid]["experiment_labels"]:
                entries[sid]["experiment_labels"].append(exp_type)
            for wall, stim in zip(beh["UniqWalls"], beh["stim_id"]):
                if not np.isfinite(stim):
                    continue
                wall, stim = str(wall), int(stim)
                old = mappings[sid].get(wall)
                if old is not None and old != stim:
                    raise ValueError(f"Conflicting stimulus ID: {sid} {wall}")
                mappings[sid][wall] = stim
        del behavior
        gc.collect()
    catalog = list(entries.values())
    if len(catalog) != 89 or len({e["session_id"] for e in catalog}) != 89:
        raise AssertionError(f"Expected 89 unique sessions, got {len(catalog)}")
    return catalog, dict(mappings)


def infer_shape(sid: str) -> tuple[int, int]:
    """Infer exact neuron/frame counts without loading a multi-GB neural object."""
    with np.load(ret_path(sid), allow_pickle=True) as ret:
        nneurons = len(ret["iarea"])
    payload = spk_path(sid).stat().st_size - OBJECT_NPY_OVERHEAD
    divisor = nneurons * np.dtype(np.float32).itemsize
    if payload <= 0 or payload % divisor:
        raise ValueError(f"Unexpected neural NPY layout for {sid}")
    return int(nneurons), int(payload // divisor)


def grouped(catalog: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in catalog:
        out[entry["source_experiment"]].append(entry)
    return dict(out)


def compute_global_stats(
    catalog: list[dict[str, Any]], mappings: dict[str, dict[str, int]]
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Compute speed quartiles on exactly the samples eligible for conversion."""
    speeds, intervals = [], []
    stats = Counter()
    for exp_type, entries in grouped(catalog).items():
        behavior = np.load(BEH / f"Beh_{exp_type}.npy", allow_pickle=True).item()
        for entry in entries:
            sid = entry["session_id"]
            beh = behavior[entry["source_behavior_key"]]
            nneurons, nfr = infer_shape(sid)
            if nfr > len(beh["ft"]):
                raise AssertionError(f"Neural frames exceed behavior frames: {sid}")
            trial_cat = np.array(
                [mappings[sid].get(str(w), -1) for w in beh["WallName"]],
                dtype=np.int16,
            )
            ft_trial = np.asarray(beh["ft_trInd"][:nfr])
            finite = np.isfinite(ft_trial)
            ti = np.zeros(nfr, dtype=np.int64)
            ti[finite] = ft_trial[finite].astype(np.int64)
            in_range = finite & (ti >= 0) & (ti < len(trial_cat))
            labeled = np.zeros(nfr, dtype=bool)
            labeled[in_range] = trial_cat[ti[in_range]] >= 0
            valid = (
                labeled
                & np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool)
                & (np.asarray(beh["ft_move"][:nfr]) > 0)
            )
            speeds.append(np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float64)[valid])
            dt = np.diff(np.asarray(beh["ft"][:nfr], dtype=np.float64)) * 86_400_000
            intervals.append(dt[np.isfinite(dt) & (dt > 0)])
            stats["sessions"] += 1
            stats["neurons"] += nneurons
            stats["valid_frames"] += int(valid.sum())
            stats[f"terminal_frame_offset_{len(beh['ft']) - nfr}"] += 1
        del behavior
        gc.collect()
    speed = np.concatenate(speeds)
    cuts = np.quantile(speed, [0.25, 0.5, 0.75])
    classes = np.searchsorted(cuts, speed, side="right")
    stats_out = {
        **dict(stats),
        "speed_min": float(speed.min()),
        "speed_max": float(speed.max()),
        "speed_class_counts": np.bincount(classes, minlength=4).tolist(),
    }
    nominal_ms = float(np.median(np.concatenate(intervals)))
    return cuts, nominal_ms, stats_out


def region_idx(iarea: np.ndarray) -> np.ndarray:
    out = np.full(len(iarea), 4, dtype=np.int8)
    out[iarea == 8] = 0
    out[np.isin(iarea, [0, 1, 2, 9])] = 1
    out[np.isin(iarea, [5, 6])] = 2
    out[np.isin(iarea, [3, 4])] = 3
    return out


def select_entries(catalog: list[dict[str, Any]], sample: bool) -> list[dict[str, Any]]:
    if not sample:
        return catalog
    # Exercise both reward/lick classes in the required two-session test.
    return [
        next(e for e in catalog if not e["has_reward"]),
        next(e for e in catalog if e["has_reward"]),
    ]


def processing_plot(
    outdir: Path,
    sid: str,
    beh: dict[str, Any],
    planes: list[np.ndarray],
    nfr: int,
    neural: list[np.ndarray],
    inputs: list[np.ndarray],
    outputs: list[np.ndarray],
    source_trials: list[int],
    regions: np.ndarray,
    speed_cuts: np.ndarray,
) -> None:
    """Visual audit of raw alignment and every conversion transform."""
    slot = min(len(source_trials) // 2, len(source_trials) - 1)
    tr = source_trials[slot]
    n, x, y = neural[slot], inputs[slot], outputs[slot]
    raw_trial = np.asarray(beh["ft_trInd"][:nfr]) == tr
    keep = np.flatnonzero(
        raw_trial
        & np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool)
        & (np.asarray(beh["ft_move"][:nfr]) > 0)
    )
    t = x[2]
    fig, ax = plt.subplots(5, 2, figsize=(18, 20), constrained_layout=True)
    ax = ax.ravel()

    neighborhood = np.flatnonzero(raw_trial)
    if len(neighborhood):
        lo, hi = max(0, neighborhood[0] - 5), min(nfr, neighborhood[-1] + 6)
        frame = np.arange(lo, hi)
        ax[0].plot(frame, np.asarray(beh["ft_Pos"])[frame] / 10, label="position m")
        ax[0].step(frame, np.asarray(beh["ft_CorrSpc"], int)[frame] * 4,
                   where="mid", label="corridor mask x4")
        ax[0].step(frame, (np.asarray(beh["ft_move"])[frame] > 0) * 3.5,
                   where="mid", label="moving mask x3.5")
        ax[0].scatter(keep, np.asarray(beh["ft_Pos"])[keep] / 10,
                      s=15, color="black", label="retained")
    ax[0].set_title("Raw frame alignment and validity filtering")
    ax[0].set_xlabel("global imaging frame")
    ax[0].legend(fontsize=8)

    shown = min(50, n.shape[0])
    im = ax[1].imshow(n[:shown], aspect="auto", interpolation="nearest")
    ax[1].set_title(f"Converted deconvolved traces ({shown}/{n.shape[0]} neurons)")
    ax[1].set_xlabel("retained native sample")
    fig.colorbar(im, ax=ax[1], shrink=0.8)

    ax[2].plot(t, x[0], ".-", label="time to cue")
    ax[2].axhline(0, color="purple", linestyle="--", label="cue")
    ax[2].set_title("Temporal alignment inputs")
    ax[2].set_xlabel("time since corridor entry (s)")
    ax[2].set_ylabel("seconds")
    ax[2].legend()

    ax[3].step(t, x[3], where="mid", label="reward available")
    ax[3].step(t, y[1], where="mid", label="lick binary")
    ax[3].set_title("Reward context and frame-binned licks")
    ax[3].set_ylim(-0.1, 1.2)
    ax[3].legend()

    pos_m = np.asarray(beh["ft_Pos"])[keep] / 10
    ax[4].plot(t, pos_m, ".-", label="raw position")
    ax[4].step(t, y[2] + 0.5, where="mid", label="1-m bin center")
    ax[4].set_title("Position discretization")
    ax[4].legend()

    speed = np.asarray(beh["ft_RunSpeed"])[keep]
    ax[5].plot(t, speed, ".-", label="raw speed")
    for q, cut in enumerate(speed_cuts, 1):
        ax[5].axhline(cut, linestyle="--", label=f"Q{q}: {cut:.2f}")
    ax[5].scatter(t, y[3] * 8 - 15, c=y[3], cmap="viridis",
                  marker="s", label="quartile (offset)")
    ax[5].set_title("Global-quartile speed discretization")
    ax[5].set_ylabel("cm/s")
    ax[5].legend(fontsize=8)

    ax[6].bar(np.arange(7), np.bincount(
        np.concatenate([trial[0] for trial in outputs]), minlength=7
    ))
    ax[6].set_xticks(np.arange(7), STIMULI, rotation=45, ha="right")
    ax[6].set_title("Session stimulus-frame distribution")

    lengths = [trial.shape[1] for trial in neural]
    ax[7].hist(lengths, bins=min(30, max(5, len(set(lengths)))))
    ax[7].axvline(len(keep), color="red", label="plotted trial")
    ax[7].set_title("Retained native samples per trial")
    ax[7].legend()

    ax[8].bar(np.arange(5), np.bincount(regions, minlength=5))
    ax[8].set_xticks(np.arange(5), REGIONS, rotation=30, ha="right")
    ax[8].set_title("Reference region grouping (all neurons retained)")

    raw = np.concatenate([plane[:, keep] for plane in planes], axis=0)
    diff = np.abs(raw - n)
    ax[9].hist(diff.ravel(), bins=10)
    ax[9].set_yscale("symlog")
    ax[9].set_title(
        f"Raw→converted neural check: max={diff.max(initial=0):.3g}, "
        f"allclose={np.allclose(raw, n)}"
    )
    fig.suptitle(f"Processing audit: {sid}, source trial {tr}")
    path = outdir / f"processing_{sid}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  saved processing plot: {path}", flush=True)


def convert_session(
    entry: dict[str, Any],
    beh: dict[str, Any],
    wall_map: dict[str, int],
    speed_cuts: np.ndarray,
    day: float,
    outdir: Path,
    show_plot: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, dict[str, Any]]:
    """Load one source session and construct aligned per-trial matrices."""
    sid = entry["session_id"]
    started = time.perf_counter()
    obj = np.load(spk_path(sid), allow_pickle=True).item()
    if set(obj) != {"spks"} or not obj["spks"]:
        raise ValueError(f"Unexpected neural object in {sid}")
    planes = obj["spks"]
    if any(a.ndim != 2 or a.dtype != np.float32 for a in planes):
        raise ValueError(f"Expected a list of float32 matrices in {sid}")
    if len({a.shape[1] for a in planes}) != 1:
        raise ValueError(f"Neural plane frame mismatch in {sid}")
    neural_nfr = planes[0].shape[1]
    nfr = min(neural_nfr, len(beh["ft"]))
    nneurons = sum(a.shape[0] for a in planes)

    with np.load(ret_path(sid), allow_pickle=True) as ret:
        iarea = np.asarray(ret["iarea"])
    if len(iarea) != nneurons:
        raise AssertionError(f"Retinotopy/neural mismatch in {sid}")
    regions = region_idx(iarea)

    ft_trial = np.asarray(beh["ft_trInd"][:nfr])
    valid = (
        np.isfinite(ft_trial)
        & np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool)
        & (np.asarray(beh["ft_move"][:nfr]) > 0)
    )
    ft = np.asarray(beh["ft"][:nfr], dtype=np.float64)
    pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float64)
    speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float64)
    lick = np.zeros(nfr, dtype=np.int8)
    lick_frames = np.asarray(beh["LickFr"], dtype=np.float64)
    lick_frames = lick_frames[np.isfinite(lick_frames)].astype(np.int64)
    lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < nfr)]
    lick[np.unique(lick_frames)] = 1

    neural, inputs, outputs, included = [], [], [], []
    dropped = Counter()
    for tr in range(int(beh["ntrials"])):
        category = wall_map.get(str(beh["WallName"][tr]))
        if category is None:
            dropped["unmapped_stimulus"] += 1
            continue
        frames = np.flatnonzero(valid & (ft_trial == tr))
        if not len(frames):
            dropped["no_valid_running_corridor_frames"] += 1
            continue
        T = len(frames)
        n = np.empty((nneurons, T), dtype=np.float32)
        row = 0
        for plane in planes:
            nr = plane.shape[0]
            n[row:row + nr] = plane[:, frames]
            row += nr
        if not np.isfinite(n).all():
            raise ValueError(f"Non-finite neural values: {sid} trial {tr}")

        sample_time = ft[frames]
        x = np.empty((4, T), dtype=np.float32)
        x[0] = (float(beh["SoundTime"][tr]) - sample_time) * 86_400
        x[1] = day
        x[2] = (sample_time - float(beh["Trial_start_time"][tr])) * 86_400
        x[3] = float(bool(beh["isRew"][tr]))
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite input values: {sid} trial {tr}")

        y = np.empty((4, T), dtype=np.int8)
        y[0] = category
        y[1] = lick[frames]
        y[2] = np.floor(pos[frames] / 10).astype(np.int8)
        if np.any((y[2] < 0) | (y[2] > 3)):
            raise ValueError(f"Position outside 0-4 m: {sid} trial {tr}")
        y[3] = np.searchsorted(speed_cuts, speed[frames], side="right").astype(np.int8)

        neural.append(n)
        inputs.append(x)
        outputs.append(y)
        included.append(tr)

    if len(neural) < 2:
        raise ValueError(f"Fewer than two converted trials: {sid}")
    for n, x, y in zip(neural, inputs, outputs):
        if n.shape[1] != x.shape[1] or n.shape[1] != y.shape[1]:
            raise AssertionError(f"Time dimension mismatch: {sid}")
    if show_plot:
        processing_plot(
            outdir, sid, beh, list(planes), nfr, neural, inputs, outputs,
            included, regions, speed_cuts,
        )

    info = {
        "session_id": sid,
        "subject": entry["subject"],
        "date": entry["date"],
        "block": entry["block"],
        "day_of_training": day,
        "experiment_labels": entry["experiment_labels"],
        "experiment_records": entry["records"],
        "source_behavior_file": f"Beh_{entry['source_experiment']}.npy",
        "source_behavior_key": entry["source_behavior_key"],
        "source_neural_file": spk_path(sid).name,
        "reward_mode": str(beh["Reward_Mode"]),
        "corridor_length_dm": float(beh["Corridor_Length"]),
        "texture_length_dm": float(beh["Texture_Length"]),
        "gray_space_length_dm": float(beh["Gray_Space_length"]),
        "wall_to_stimulus_id": dict(wall_map),
        "n_source_trials": int(beh["ntrials"]),
        "n_converted_trials": len(neural),
        "included_trial_indices": included,
        "dropped_trials": dict(dropped),
        "n_neurons": nneurons,
        "n_neural_frames": neural_nfr,
        "n_behavior_frames": len(beh["ft"]),
        "n_retained_samples": int(sum(a.shape[1] for a in neural)),
        "load_and_convert_seconds": float(time.perf_counter() - started),
    }
    return neural, inputs, outputs, regions, info


def convert(outpath: Path, sample: bool, show_processing: bool) -> None:
    total_start = time.perf_counter()
    print("Building deduplicated session catalog...", flush=True)
    catalog, mappings = build_catalog()
    subjects = sorted({entry["subject"] for entry in catalog})
    subject_lookup = {name: i for i, name in enumerate(subjects)}
    first_date = {
        subject: min(
            sid_date(entry["session_id"])
            for entry in catalog if entry["subject"] == subject
        )
        for subject in subjects
    }

    print("Computing full-dataset speed quartiles and sampling interval...", flush=True)
    cuts, nominal_ms, raw_stats = compute_global_stats(catalog, mappings)
    print(
        f"  speed quartiles (cm/s): {cuts.tolist()}\n"
        f"  speed class counts: {raw_stats['speed_class_counts']}\n"
        f"  nominal imaging interval: {nominal_ms:.6f} ms\n"
        f"  terminal frame offsets: "
        f"1={raw_stats.get('terminal_frame_offset_1', 0)}, "
        f"2={raw_stats.get('terminal_frame_offset_2', 0)}, "
        f"3={raw_stats.get('terminal_frame_offset_3', 0)}",
        flush=True,
    )

    selected = select_entries(catalog, sample)
    mode = "sample" if sample else "full"
    print(f"Converting {len(selected)} sessions in {mode} mode...", flush=True)
    if sample:
        print("  " + ", ".join(e["session_id"] for e in selected), flush=True)

    neural: list[list[np.ndarray]] = []
    inputs: list[list[np.ndarray]] = []
    outputs: list[list[np.ndarray]] = []
    brain_idx: list[np.ndarray] = []
    subject_idx: list[int] = []
    session_info: list[dict[str, Any]] = []
    current_exp, current_behavior = None, None
    plots_left = 2 if show_processing else 0

    for number, entry in enumerate(selected, 1):
        session_start = time.perf_counter()
        if current_exp != entry["source_experiment"]:
            del current_behavior
            gc.collect()
            current_exp = entry["source_experiment"]
            print(f"Loading behavior: Beh_{current_exp}.npy", flush=True)
            current_behavior = np.load(
                BEH / f"Beh_{current_exp}.npy", allow_pickle=True
            ).item()
        beh = current_behavior[entry["source_behavior_key"]]
        day = float((sid_date(entry["session_id"]) - first_date[entry["subject"]]).days)
        print(
            f"[{number}/{len(selected)}] {entry['session_id']} "
            f"({entry['source_experiment']})",
            flush=True,
        )
        n, x, y, r, info = convert_session(
            entry, beh, mappings[entry["session_id"]], cuts, day,
            outpath.parent, plots_left > 0,
        )
        plots_left = max(0, plots_left - 1)
        neural.append(n)
        inputs.append(x)
        outputs.append(y)
        brain_idx.append(r)
        subject_idx.append(subject_lookup[entry["subject"]])
        session_info.append(info)

        elapsed = time.perf_counter() - session_start
        so_far = time.perf_counter() - total_start
        eta = so_far / number * (len(selected) - number)
        print(
            f"  trials={len(n)}, neurons={n[0].shape[0]}, "
            f"samples={sum(a.shape[1] for a in n)}, "
            f"payload={sum(a.nbytes for a in n) / 1e9:.3f} GB, "
            f"session={elapsed:.2f}s, elapsed={so_far:.2f}s, ETA={eta:.2f}s",
            flush=True,
        )
    del current_behavior
    gc.collect()

    data = {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int16),
        "brain_regions": REGIONS,
        "brain_region_idx": brain_idx,
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "output_values": OUTPUT_VALUES,
        "metadata": {
            "task_description": (
                "Decode visual stimulus category, frame-binned licking, 1-m "
                "corridor position, and running-speed quartile from deconvolved "
                "neural activity and cue/training/trial/reward context."
            ),
            "time_bin_size": nominal_ms,
            "time_bin_size_units": "ms (nominal median native imaging interval)",
            "temporal_sampling": (
                "Native active-running imaging samples in the textured 0-4 m "
                "corridor; explicit time inputs preserve acquisition jitter and "
                "gaps caused by excluding stopped frames."
            ),
            "temporal_alignment_event": (
                "trial start / entry into the 4-m textured corridor"
            ),
            "off_start": 0.0,
            "off_end": None,
            "cue_time_sign_convention": (
                "positive before cue, zero at cue, negative after cue"
            ),
            "position_units": "m",
            "running_speed_units": "cm/s",
            "running_speed_quartile_thresholds": cuts.tolist(),
            "stimulus_id_definition": dict(enumerate(STIMULI)),
            "neural_signal": (
                "Supplied Suite2p non-negative deconvolved fluorescence traces "
                "(0.75-s decay); no additional normalization"
            ),
            "trial_filter": (
                "finite trial ID AND ft_CorrSpc AND ft_move>0 "
                "AND mapped canonical stimulus ID"
            ),
            "day_of_training_definition": (
                "elapsed calendar days from each subject's earliest indexed "
                "imaging session"
            ),
            "source_recordings_total": 89,
            "conversion_mode": mode,
            "full_dataset_prepass": raw_stats,
            "session_info": session_info,
        },
    }

    if not (len(neural) == len(inputs) == len(outputs) == len(brain_idx)):
        raise AssertionError("Session dimension mismatch")
    for s in range(len(neural)):
        if len(neural[s]) < 2:
            raise AssertionError(f"Session {s} has fewer than two trials")
        if len(brain_idx[s]) != neural[s][0].shape[0]:
            raise AssertionError(f"Session {s} region/neuron mismatch")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing pickle: {outpath}", flush=True)
    write_start = time.perf_counter()
    with outpath.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_seconds = time.perf_counter() - write_start
    total_seconds = time.perf_counter() - total_start
    ntrials = sum(len(sess) for sess in neural)
    nsamples = sum(a.shape[1] for sess in neural for a in sess)
    payload = sum(a.nbytes for sess in neural for a in sess)
    print(
        f"Conversion complete: sessions={len(neural)}, trials={ntrials}, "
        f"samples={nsamples}, neural_payload={payload / 1e9:.3f} GB, "
        f"file_size={outpath.stat().st_size / 1e9:.3f} GB, "
        f"pickle_write={write_seconds:.2f}s, total={total_seconds:.2f}s",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="process all sessions (default)")
    mode.add_argument("--sample", action="store_true", help="process two representative sessions")
    parser.add_argument(
        "--show-processing", action="store_true",
        help="save processing audit plots for up to two sessions",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(
        args.outpicklefile.resolve(),
        sample=bool(args.sample),
        show_processing=bool(args.show_processing),
    )
