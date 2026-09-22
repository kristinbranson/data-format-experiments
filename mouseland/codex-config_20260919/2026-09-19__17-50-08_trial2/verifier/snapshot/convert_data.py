#!/usr/bin/env python3
"""Convert Zhong et al. (2025) imaging data to decoder-compatible trials.

Usage:
    python -u /app/convert_data.py OUTPUT [--full | --sample] [--show-processing]

The paper's general neural curation is followed: supplied Suite2p deconvolved
activity, mapped visual-cortex cells, frames in the 0--4 m textured corridor,
and frames for which the virtual reality moved (ft_move > 0). Temporal inputs
use the original timestamps, so elapsed-time gaps are not hidden by curation.
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
FS_HZ = 3.17
MS_PER_FRAME = 1000.0 / FS_HZ
DAY_TO_SECONDS = 86_400.0

BRAIN_REGIONS = ["V1", "mHV", "lHV", "aHV"]
CATEGORY_VALUES = ["circle", "leaf", "rock", "wood"]
CATEGORY_TO_CODE = {name: i for i, name in enumerate(CATEGORY_VALUES)}

# Two post-learning task recordings with rewards and licking. They exercise all
# time-varying fields while keeping the sample deterministic and reasonably small.
SAMPLE_BASES = ["TX60_2021_05_04_1", "TX108_2023_04_01_1"]


def physical_base(session_key: str) -> str:
    """Remove the swap-analysis suffix, which is not a separate recording."""
    return re.sub(r"_swap[12]$", "", session_key)


def behavior_catalog() -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, str]]:
    """Load each physical behavior session once and validate aliases.

    Returns dictionaries keyed by physical session base: behavior data, source
    filename, and source key. Direct WallName labels are invariant to the
    alternate swap1/swap2 `stim_id` views.
    """
    sessions: dict[str, dict[str, Any]] = {}
    source_files: dict[str, str] = {}
    source_keys: dict[str, str] = {}
    for path in sorted((DATA_ROOT / "beh").glob("Beh_*.npy")):
        loaded = np.load(path, allow_pickle=True).item()
        for key, beh in loaded.items():
            base = physical_base(key)
            if base in sessions:
                old = sessions[base]
                checks = (
                    int(old["ntrials"]) == int(beh["ntrials"]),
                    len(old["ft"]) == len(beh["ft"]),
                    np.array_equal(old["WallName"], beh["WallName"]),
                    np.array_equal(old["isRew"], beh["isRew"]),
                )
                if not all(checks):
                    raise ValueError(f"Conflicting behavior aliases for {base}")
                continue
            sessions[base] = beh
            source_files[base] = path.name
            source_keys[base] = key
        del loaded
        gc.collect()
    return sessions, source_files, source_keys


def jsonable(value: Any) -> Any:
    """Convert experiment-registry values to plain serializable Python data."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def registry_catalog() -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    """Collect aliases and resolve the requested per-session training day."""
    registry = np.load(DATA_ROOT / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()
    aliases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for experiment_type, entries in registry.items():
        for entry in entries:
            base = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
            plain = {str(k): jsonable(v) for k, v in entry.items()}
            plain["experiment_type"] = experiment_type
            aliases[base].append(plain)

    training_day: dict[str, float] = {}
    for base, entries in aliases.items():
        explicit_days = [e["days"] for e in entries if "days" in e]
        session_indices = [e["sess#"] for e in entries if "sess#" in e]
        if explicit_days:
            training_day[base] = float(min(explicit_days))
        elif session_indices:
            # Shared recordings may have alternate statistical labels (notably
            # swap1/swap2). The minimum is the physical stage/day index.
            training_day[base] = float(min(session_indices))
        else:
            raise ValueError(f"No day/session field for {base}")
    return dict(aliases), training_day


def area_codes(iarea: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the mapped-cell mask and zero-based grouped-region indices."""
    iarea = np.asarray(iarea)
    mapped = np.isin(iarea, [0, 1, 2, 3, 4, 5, 6, 8, 9])
    codes = np.full(iarea.shape, -1, dtype=np.int8)
    codes[iarea == 8] = 0
    codes[np.isin(iarea, [0, 1, 2, 9])] = 1
    codes[np.isin(iarea, [5, 6])] = 2
    codes[np.isin(iarea, [3, 4])] = 3
    if np.any(codes[mapped] < 0):
        raise AssertionError("Mapped area without grouped brain-region code")
    return mapped, codes[mapped]


def stimulus_category(name: str) -> int:
    """Pool exemplar numbers and spatial swaps into their physical category."""
    match = re.match(r"([A-Za-z]+)", str(name))
    if match is None or match.group(1).lower() not in CATEGORY_TO_CODE:
        raise ValueError(f"Unknown stimulus category in WallName={name!r}")
    return CATEGORY_TO_CODE[match.group(1).lower()]


def load_selected_spikes(base: str, mapped: np.ndarray) -> tuple[np.ndarray, int, list[tuple[int, int]]]:
    """Load plane-wise spks into one mapped-neuron matrix without a full concat copy."""
    path = DATA_ROOT / "spk" / f"{base}_neural_data.npy"
    neural_dict = np.load(path, allow_pickle=True).item()
    planes = neural_dict["spks"]
    if not planes:
        raise ValueError(f"No spks planes in {path}")
    frame_counts = {int(p.shape[1]) for p in planes}
    if len(frame_counts) != 1:
        raise ValueError(f"Plane frame-count mismatch in {base}: {frame_counts}")
    nframes = frame_counts.pop()
    plane_sizes = [int(p.shape[0]) for p in planes]
    if sum(plane_sizes) != len(mapped):
        raise ValueError(
            f"Retinotopy/neural neuron mismatch in {base}: {len(mapped)} vs {sum(plane_sizes)}"
        )

    selected = np.empty((int(mapped.sum()), nframes), dtype=np.float32)
    source_ranges: list[tuple[int, int]] = []
    src0 = 0
    dst0 = 0
    for plane in planes:
        src1 = src0 + plane.shape[0]
        local_mask = mapped[src0:src1]
        count = int(local_mask.sum())
        selected[dst0 : dst0 + count] = plane[local_mask]
        source_ranges.append((src0, src1))
        src0 = src1
        dst0 += count
    del planes, neural_dict
    return selected, nframes, source_ranges


def retinotopy_path(base: str) -> Path:
    parts = base.split("_")
    # base is mouse_YYYY_MM_DD_block; retinotopy omits block.
    stem = "_".join(parts[:-1])
    return DATA_ROOT / "retinotopy" / f"{stem}_trans.npz"


def convert_session(
    base: str,
    beh: dict[str, Any],
    day: float,
    plot_requested: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, dict[str, Any], dict[str, Any] | None]:
    """Convert one physical recording and return trials plus session metadata."""
    start = time.perf_counter()
    retino = np.load(retinotopy_path(base), allow_pickle=True)
    iarea = np.asarray(retino["iarea"])
    mapped, region_codes = area_codes(iarea)
    del retino

    load_start = time.perf_counter()
    spk, neural_nframes, _ = load_selected_spikes(base, mapped)
    load_seconds = time.perf_counter() - load_start

    frame_fields = [
        "ft",
        "ft_trInd",
        "ft_Pos",
        "ft_move",
        "ft_CorrSpc",
        "ft_RunSpeed",
    ]
    common_nframes = min([neural_nframes] + [len(beh[k]) for k in frame_fields])
    ft = np.asarray(beh["ft"][:common_nframes], dtype=np.float64)
    trial_stamp = np.asarray(beh["ft_trInd"][:common_nframes])
    corridor = np.asarray(beh["ft_CorrSpc"][:common_nframes], dtype=bool)
    movement = np.asarray(beh["ft_move"][:common_nframes]) > 0
    valid = np.isfinite(trial_stamp) & corridor & movement

    lick_global = np.zeros(common_nframes, dtype=np.int8)
    lick_float = np.asarray(beh["LickFr"], dtype=np.float64)
    lick_frames = lick_float[np.isfinite(lick_float)].astype(np.int64)
    lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < common_nframes)]
    lick_global[lick_frames] = 1

    neural_trials: list[np.ndarray] = []
    input_trials: list[np.ndarray] = []
    output_trials: list[np.ndarray] = []
    speed_trials: list[np.ndarray] = []
    included_trial_indices: list[int] = []
    excluded_trials: list[int] = []

    ntrials = int(beh["ntrials"])
    for trial in range(ntrials):
        frames = np.flatnonzero(valid & (trial_stamp == trial))
        if frames.size == 0:
            excluded_trials.append(trial)
            continue

        neural = spk[:, frames].copy()
        if not np.all(np.isfinite(neural)):
            raise ValueError(f"Non-finite neural values in {base} trial {trial}")

        time_to_sound = (float(beh["SoundTime"][trial]) - ft[frames]) * DAY_TO_SECONDS
        time_since_start = (ft[frames] - float(beh["Trial_start_time"][trial])) * DAY_TO_SECONDS
        decoder_input = np.vstack(
            [
                time_to_sound,
                np.full(frames.size, day, dtype=np.float64),
                time_since_start,
                np.full(frames.size, float(bool(beh["isRew"][trial])), dtype=np.float64),
            ]
        ).astype(np.float32)

        position = np.asarray(beh["ft_Pos"][:common_nframes])[frames]
        position_class = np.clip(np.floor(position / 10.0), 0, 3).astype(np.int8)
        speed = np.asarray(beh["ft_RunSpeed"][:common_nframes], dtype=np.float32)[frames]
        category = stimulus_category(str(beh["WallName"][trial]))
        decoder_output = np.empty((4, frames.size), dtype=np.int8)
        decoder_output[0] = category
        decoder_output[1] = lick_global[frames]
        decoder_output[2] = position_class
        decoder_output[3] = 0  # Filled after global speed thresholds are known.

        if not np.all(np.isfinite(decoder_input)) or not np.all(np.isfinite(speed)):
            raise ValueError(f"Non-finite input/speed values in {base} trial {trial}")

        neural_trials.append(neural)
        input_trials.append(decoder_input)
        output_trials.append(decoder_output)
        speed_trials.append(speed)
        included_trial_indices.append(trial)

    if len(neural_trials) < 2:
        raise ValueError(f"{base} has fewer than two usable trials after curation")

    plot_context = None
    if plot_requested:
        # Construct directly here to avoid retaining the large full-session matrix.
        raw_n = min(common_nframes, 700)
        first_trial = included_trial_indices[0]
        plot_context = {
            "base": base,
            "raw_region_ids": iarea.copy(),
            "mapped": mapped.copy(),
            "region_codes": region_codes.copy(),
            "raw_neural": spk[: min(spk.shape[0], 24), :raw_n].copy(),
            "valid_mask": valid[:raw_n].copy(),
            "ft_pos": np.asarray(beh["ft_Pos"][:raw_n]).copy(),
            "ft_move": np.asarray(beh["ft_move"][:raw_n]).copy(),
            "ft_corr": np.asarray(beh["ft_CorrSpc"][:raw_n]).copy(),
            "trial_frames": np.flatnonzero(valid & (trial_stamp == first_trial)),
            "first_trial": first_trial,
        }

    session_info = {
        "session_id": base,
        "training_day": day,
        "n_trials_source": ntrials,
        "n_trials_included": len(neural_trials),
        "excluded_trial_indices": excluded_trials,
        "n_neurons_source": int(len(iarea)),
        "n_neurons_mapped": int(mapped.sum()),
        "neural_nframes": int(neural_nframes),
        "behavior_nframes": int(len(beh["ft"])),
        "common_nframes": int(common_nframes),
        "curated_timepoints": int(sum(x.shape[1] for x in neural_trials)),
        "rewarded_trials_source": int(np.sum(beh["isRew"])),
        "load_neural_seconds": load_seconds,
        "conversion_seconds": time.perf_counter() - start,
    }
    print(
        f"  {base}: {len(neural_trials)}/{ntrials} trials, "
        f"{mapped.sum():,}/{len(mapped):,} neurons, "
        f"{session_info['curated_timepoints']:,} timepoints, "
        f"load {load_seconds:.2f}s, total {session_info['conversion_seconds']:.2f}s",
        flush=True,
    )
    del spk
    gc.collect()
    return (
        neural_trials,
        input_trials,
        output_trials,
        speed_trials,
        region_codes.astype(np.int16),
        session_info,
        plot_context,
    )


def fill_speed_classes(
    outputs: list[list[np.ndarray]], speeds: list[list[np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one global quartile definition and fill output row 3."""
    all_speed = np.concatenate([trial for session in speeds for trial in session]).astype(np.float64)
    thresholds = np.quantile(all_speed, [0.25, 0.50, 0.75])
    counts = np.zeros(4, dtype=np.int64)
    for out_session, speed_session in zip(outputs, speeds, strict=True):
        for out_trial, speed_trial in zip(out_session, speed_session, strict=True):
            labels = np.searchsorted(thresholds, speed_trial, side="right").astype(np.int8)
            out_trial[3] = labels
            counts += np.bincount(labels, minlength=4)
    return thresholds, counts


def validate_converted(data: dict[str, Any]) -> None:
    """Fast strict validation before writing a potentially large pickle."""
    nsessions = len(data["neural"])
    assert nsessions == len(data["input"]) == len(data["output"])
    assert nsessions == len(data["subject_idx"]) == len(data["brain_region_idx"])
    for s in range(nsessions):
        assert len(data["neural"][s]) >= 2
        assert len(data["neural"][s]) == len(data["input"][s]) == len(data["output"][s])
        nneurons = len(data["brain_region_idx"][s])
        assert nneurons > 0
        for neural, decoder_input, decoder_output in zip(
            data["neural"][s], data["input"][s], data["output"][s], strict=True
        ):
            assert neural.dtype == np.float32 and neural.ndim == 2
            assert decoder_input.dtype == np.float32 and decoder_input.shape == (4, neural.shape[1])
            assert decoder_output.dtype == np.int8 and decoder_output.shape == (4, neural.shape[1])
            assert neural.shape[0] == nneurons and neural.shape[1] > 0
            assert np.all(np.isfinite(neural)) and np.all(np.isfinite(decoder_input))
            assert np.all((decoder_output >= 0) & (decoder_output <= np.array([3, 1, 3, 3])[:, None]))


def processing_plot(
    context: dict[str, Any],
    neural_trials: list[np.ndarray],
    input_trials: list[np.ndarray],
    output_trials: list[np.ndarray],
    speed_trials: list[np.ndarray],
    thresholds: np.ndarray,
) -> None:
    """Plot each major transformation for visual alignment inspection."""
    base = context["base"]
    fig, axes = plt.subplots(4, 3, figsize=(18, 15), constrained_layout=True)

    raw_ids = context["raw_region_ids"]
    labels, counts = np.unique(raw_ids, return_counts=True)
    axes[0, 0].bar(labels.astype(str), counts)
    axes[0, 0].set_title("Raw retinotopy IDs (before curation)")
    axes[0, 0].set_xlabel("iarea")
    axes[0, 0].set_ylabel("neurons")

    region_counts = np.bincount(context["region_codes"], minlength=4)
    axes[0, 1].bar(BRAIN_REGIONS, region_counts)
    axes[0, 1].set_title(f"Mapped neurons retained ({region_counts.sum():,})")

    raw_neural = context["raw_neural"]
    vmax = np.percentile(raw_neural, 99.5) if raw_neural.size else 1
    axes[0, 2].imshow(raw_neural, aspect="auto", interpolation="none", vmin=0, vmax=max(vmax, 1e-6))
    axes[0, 2].set_title("Supplied deconvolved spks (raw frames)")
    axes[0, 2].set_ylabel("sample mapped neurons")

    x = np.arange(len(context["valid_mask"]))
    axes[1, 0].plot(x, context["ft_pos"] / 10.0, label="position (m)")
    axes[1, 0].fill_between(x, 0, 1, where=context["ft_corr"], transform=axes[1, 0].get_xaxis_transform(), alpha=0.12, label="texture")
    axes[1, 0].scatter(x[context["valid_mask"]], context["ft_pos"][context["valid_mask"]] / 10.0, s=5, label="retained")
    axes[1, 0].set_title("Trial/frame curation: texture & ft_move>0")
    axes[1, 0].legend(fontsize=8)

    trial_neural = neural_trials[0][: min(24, neural_trials[0].shape[0])]
    vmax = np.percentile(trial_neural, 99.5) if trial_neural.size else 1
    axes[1, 1].imshow(trial_neural, aspect="auto", interpolation="none", vmin=0, vmax=max(vmax, 1e-6))
    axes[1, 1].set_title(f"Converted trial {context['first_trial']} neural")
    axes[1, 1].set_ylabel("sample neurons")

    inp = input_trials[0]
    axes[1, 2].plot(inp[2], label="time since entry")
    axes[1, 2].plot(inp[0], label="time to cue")
    axes[1, 2].axhline(0, color="k", linewidth=0.7)
    axes[1, 2].set_title("Timestamp-derived temporal inputs")
    axes[1, 2].set_ylabel("seconds")
    axes[1, 2].legend(fontsize=8)

    out = output_trials[0]
    axes[2, 0].step(np.arange(out.shape[1]), out[2], where="mid")
    axes[2, 0].set_yticks(range(4), ["0–1", "1–2", "2–3", "3–4"])
    axes[2, 0].set_title("Four 1-m position classes")

    speed = speed_trials[0]
    axes[2, 1].plot(speed, color="tab:green", label="speed")
    for threshold in thresholds:
        axes[2, 1].axhline(threshold, color="k", linestyle="--", linewidth=0.7)
    axes[2, 1].step(np.arange(len(speed)), out[3] * max(float(np.max(speed)), 1) / 3, where="mid", color="tab:red", alpha=0.6, label="quartile class (scaled)")
    axes[2, 1].set_title("Running speed and global quartile bins")
    axes[2, 1].legend(fontsize=8)

    axes[2, 2].step(np.arange(out.shape[1]), out[1], where="mid")
    axes[2, 2].set_yticks([0, 1], ["no lick", "lick"])
    axes[2, 2].set_title("LickFr events on retained frames")

    category_counts = Counter(int(o[0, 0]) for o in output_trials)
    axes[3, 0].bar(CATEGORY_VALUES, [category_counts.get(i, 0) for i in range(4)])
    axes[3, 0].set_title("Per-trial visual categories")
    axes[3, 0].tick_params(axis="x", rotation=30)

    reward_counts = Counter(int(i[3, 0]) for i in input_trials)
    axes[3, 1].bar(["non-reward", "reward"], [reward_counts.get(0, 0), reward_counts.get(1, 0)])
    axes[3, 1].set_title("Per-trial reward availability")

    shapes = [n.shape[1] for n in neural_trials]
    axes[3, 2].hist(shapes, bins=min(30, max(5, len(set(shapes)))))
    axes[3, 2].set_title("Retained native frames per trial")
    axes[3, 2].set_xlabel("timepoints")

    fig.suptitle(f"Processing diagnostics: {base}", fontsize=16)
    outpath = ROOT / f"processing_{base}.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  wrote {outpath}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default)")
    mode.add_argument("--sample", action="store_true", help="Process two deterministic sessions")
    parser.add_argument("--show-processing", action="store_true", help="Save processing plots for up to two sessions")
    args = parser.parse_args()

    overall_start = time.perf_counter()
    print("Loading behavior and experiment catalogs...", flush=True)
    behaviors, behavior_files, behavior_keys = behavior_catalog()
    aliases, training_days = registry_catalog()

    spk_bases = {p.name.removesuffix("_neural_data.npy") for p in (DATA_ROOT / "spk").glob("*_neural_data.npy")}
    # Retinotopy lacks block suffix, so compare through each neural base's derived path.
    if spk_bases != set(behaviors) or spk_bases != set(aliases):
        raise ValueError(
            f"Catalog mismatch: spk={len(spk_bases)}, behavior={len(behaviors)}, registry={len(aliases)}"
        )
    missing_retino = [base for base in spk_bases if not retinotopy_path(base).exists()]
    if missing_retino:
        raise FileNotFoundError(f"Missing retinotopy files: {missing_retino}")
    print(
        f"Inventory: {len(spk_bases)} sessions, "
        f"{len({base.split('_')[0] for base in spk_bases})} subjects, "
        f"{sum(int(behaviors[b]['ntrials']) for b in spk_bases):,} source trials",
        flush=True,
    )

    if args.sample:
        bases = SAMPLE_BASES.copy()
        missing = [base for base in bases if base not in spk_bases]
        if missing:
            raise ValueError(f"Configured sample sessions missing: {missing}")
        mode_name = "sample"
    else:
        bases = sorted(spk_bases)
        mode_name = "full"

    neural_all: list[list[np.ndarray]] = []
    input_all: list[list[np.ndarray]] = []
    output_all: list[list[np.ndarray]] = []
    speed_all: list[list[np.ndarray]] = []
    brain_region_all: list[np.ndarray] = []
    session_info: list[dict[str, Any]] = []
    plot_contexts: list[dict[str, Any]] = []

    print(f"Converting {len(bases)} {mode_name} sessions...", flush=True)
    for index, base in enumerate(bases):
        result = convert_session(
            base,
            behaviors[base],
            training_days[base],
            plot_requested=args.show_processing and len(plot_contexts) < 2,
        )
        neural, decoder_input, decoder_output, speeds, region_idx, info, context = result
        info["behavior_file"] = behavior_files[base]
        info["behavior_key"] = behavior_keys[base]
        info["registry_aliases"] = aliases[base]
        neural_all.append(neural)
        input_all.append(decoder_input)
        output_all.append(decoder_output)
        speed_all.append(speeds)
        brain_region_all.append(region_idx)
        session_info.append(info)
        if context is not None:
            context["session_index"] = index
            plot_contexts.append(context)

    thresholds, speed_counts = fill_speed_classes(output_all, speed_all)
    print(
        "Global running-speed quartiles: "
        + ", ".join(f"{x:.8g}" for x in thresholds)
        + f"; counts={speed_counts.tolist()}",
        flush=True,
    )

    subjects = sorted({base.split("_")[0] for base in bases})
    subject_lookup = {subject: i for i, subject in enumerate(subjects)}
    subject_idx = np.asarray([subject_lookup[base.split("_")[0]] for base in bases], dtype=np.int16)

    total_source_trials = sum(info["n_trials_source"] for info in session_info)
    total_included_trials = sum(info["n_trials_included"] for info in session_info)
    data: dict[str, Any] = {
        "neural": neural_all,
        "input": input_all,
        "output": output_all,
        "subjects": subjects,
        "subject_idx": subject_idx,
        "brain_regions": BRAIN_REGIONS.copy(),
        "brain_region_idx": brain_region_all,
        "input_names": [
            "time_to_sound_cue_s",
            "day_of_training",
            "time_since_trial_start_s",
            "reward_available",
        ],
        "output_names": [
            "visual_stimulus_category",
            "licking",
            "corridor_position_bin",
            "running_speed_quartile",
        ],
        "output_values": [
            CATEGORY_VALUES.copy(),
            ["not_licking", "licking"],
            ["0-1 m", "1-2 m", "2-3 m", "3-4 m"],
            ["Q1", "Q2", "Q3", "Q4"],
        ],
        "metadata": {
            "task_description": (
                "Decode visual texture category, lick events, 1-m corridor position, and "
                "global running-speed quartile from deconvolved visual-cortex activity plus "
                "cue timing, training day/stage, elapsed trial time, and reward availability."
            ),
            "time_bin_size": MS_PER_FRAME,
            "native_frame_rate_hz": FS_HZ,
            "temporal_alignment_event": "entry into the 0-4 m textured visual corridor",
            "off_start": 0.0,
            "off_end": None,
            "curation": "ft_CorrSpc & (ft_move > 0), after common neural/behavior truncation",
            "time_sampling_note": (
                "Samples are native 3.17-Hz imaging frames; stationary frames are omitted per "
                "the reference analysis, and timestamp inputs preserve elapsed-time gaps."
            ),
            "neural_signal": "Suite2p non-negative deconvolved fluorescence (spks), float32",
            "neuron_filter": "mapped V1/mHV/lHV/aHV cells; source iarea -1 and 7 excluded",
            "training_day_rule": "registry `days` if present, otherwise minimum alias `sess#`",
            "time_to_sound_sign": "positive before cue, zero at cue, negative after cue",
            "position_units_source": "decimeters; converted to four 1-m classes over 0-4 m",
            "speed_units": "native ft_RunSpeed units",
            "speed_quartile_thresholds": thresholds.astype(float).tolist(),
            "speed_quartile_counts": speed_counts.astype(int).tolist(),
            "session_ids": bases,
            "session_info": session_info,
            "source_trial_count": int(total_source_trials),
            "included_trial_count": int(total_included_trials),
            "excluded_trial_count": int(total_source_trials - total_included_trials),
            "conversion_mode": mode_name,
        },
    }

    validate_converted(data)
    print("Internal structure validation passed.", flush=True)

    if args.show_processing:
        for context in plot_contexts:
            s = context["session_index"]
            processing_plot(
                context,
                neural_all[s],
                input_all[s],
                output_all[s],
                speed_all[s],
                thresholds,
            )

    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {args.outpicklefile} ...", flush=True)
    write_start = time.perf_counter()
    with args.outpicklefile.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_seconds = time.perf_counter() - write_start
    size_gib = args.outpicklefile.stat().st_size / (1024**3)
    elapsed = time.perf_counter() - overall_start
    print(
        f"Finished {mode_name} conversion: {len(bases)} sessions, "
        f"{total_included_trials:,}/{total_source_trials:,} trials, "
        f"{size_gib:.3f} GiB, write {write_seconds:.2f}s, total {elapsed:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
