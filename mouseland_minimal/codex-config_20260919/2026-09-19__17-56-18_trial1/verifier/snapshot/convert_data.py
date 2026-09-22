#!/usr/bin/env python3
"""Convert the Zhong et al. imaging data to the neural-decoder schema.

The conversion follows the analysis choices in ``code/utils.py``:

* a session is a unique (mouse, date, imaging block) recording (the experiment
  table repeats recordings when they are used in several figures);
* deconvolved Suite2p ``spks`` are concatenated in plane order;
* only retinotopically assigned visual-cortex neurons and imaging frames for
  which the animal is advancing through the 4 m textured corridor are used;
* behavioral events are aligned to the original imaging-frame clock.

The source uses decimetres for VR position and MATLAB datenums for timestamps.
Outputs here use seconds, centimetres/second, and zero-based class labels.
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"

# This is the exact grouping used by neu_area_ID() in the paper repository.
AREA_ID_TO_REGION = {
    8: 0,                         # V1
    0: 1, 1: 1, 2: 1, 9: 1,     # medial higher visual areas
    5: 2, 6: 2,                  # lateral higher visual areas
    3: 3, 4: 3,                  # anterior higher visual areas
}
REGION_NAMES = ["V1", "mHV", "lHV", "aHV"]


def _recording_id(entry: dict) -> str:
    return f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"


def _behavior_key(entry: dict, behavior: dict) -> str:
    """Resolve the optional swap-stimulus suffix used in test-3 tables."""
    base = _recording_id(entry)
    suffix = entry.get("stimtype")
    if suffix and f"{base}_{suffix}" in behavior:
        return f"{base}_{suffix}"
    if base not in behavior:
        raise KeyError(f"No behavior record for {base}")
    return base


def discover_sessions(data_root: Path) -> list[dict]:
    """Return one descriptor per physical recording, in repository order."""
    exp_info = np.load(
        data_root / "beh" / "Imaging_Exp_info.npy", allow_pickle=True
    ).item()
    sessions: OrderedDict[str, dict] = OrderedDict()
    for experiment_type, entries in exp_info.items():
        behavior_file = data_root / "beh" / f"Beh_{experiment_type}.npy"
        for entry in entries:
            rid = _recording_id(entry)
            if rid not in sessions:
                sessions[rid] = {
                    "recording_id": rid,
                    "experiment_type": experiment_type,
                    "entry": entry,
                    "behavior_file": behavior_file,
                }
    return list(sessions.values())


def load_behavior(desc: dict) -> dict:
    records = np.load(desc["behavior_file"], allow_pickle=True).item()
    return records[_behavior_key(desc["entry"], records)]


def retained_frame_mask(beh: dict, nframes: int) -> np.ndarray:
    """Paper's ``fr_valid = VRmove & isCorridor`` frame selection."""
    return (
        np.asarray(beh["ft_move"][:nframes]) > 0
    ) & np.asarray(beh["ft_CorrSpc"][:nframes], dtype=bool)


def compute_speed_edges(sessions: list[dict]) -> tuple[np.ndarray, dict]:
    """Global quartiles over the exact samples that enter the decoder."""
    chunks = []
    frame_dts_ms = []
    for i, desc in enumerate(sessions):
        beh = load_behavior(desc)
        # The processed behavior clock has one terminal interpolation sample beyond
        # the Suite2p arrays (as seen throughout the repository). Avoid opening the
        # hundreds of GB of neural files merely to discard that one sample here.
        nframes = len(beh["ft"]) - 1
        valid = retained_frame_mask(beh, nframes)
        chunks.append(np.asarray(beh["ft_RunSpeed"][:nframes][valid], dtype=np.float32))
        dt = np.diff(np.asarray(beh["ft"][:nframes], dtype=np.float64)) * 86_400_000.0
        frame_dts_ms.append(dt[(dt > 100) & (dt < 1000)])
        del beh
        if (i + 1) % 10 == 0 or i + 1 == len(sessions):
            print(f"Speed pass: {i + 1}/{len(sessions)} sessions", flush=True)
    speeds = np.concatenate(chunks)
    edges = np.quantile(speeds, [0.25, 0.50, 0.75])
    dt_all = np.concatenate(frame_dts_ms)
    stats = {
        "speed_quartile_edges_cm_s": edges.tolist(),
        "observed_median_frame_interval_ms": float(np.median(dt_all)),
        "observed_frame_interval_iqr_ms": np.quantile(dt_all, [0.25, 0.75]).tolist(),
    }
    return edges, stats


def visual_category(name: str) -> str:
    """Collapse crop/version suffixes while retaining the source category."""
    match = re.match(r"[A-Za-z]+", str(name))
    if match is None:
        raise ValueError(f"Cannot derive visual category from {name!r}")
    return match.group(0).lower()


def session_day(desc: dict) -> float:
    """Use the repository's session/day number; document the missing-value rule."""
    value = desc["entry"].get("sess#")
    if value is not None and np.isfinite(value):
        return float(value)
    # Several train2-after records have no sess# even though their stage is known.
    # Treating the two paper stages as 0/1 is consistent with train1 and avoids an
    # invented calendar-day estimate (recording dates are not training start dates).
    typ = desc["experiment_type"]
    return 1.0 if "after" in typ else 0.0


def neuron_selection(desc: dict, plane_sizes: list[int]) -> tuple[list[np.ndarray], np.ndarray]:
    ret = np.load(
        DATA_ROOT / "retinotopy" /
        f"{desc['entry']['mname']}_{desc['entry']['datexp']}_trans.npz"
    )
    area_ids = np.asarray(ret["iarea"]).astype(np.int16, copy=False)
    expected = int(sum(plane_sizes))
    if len(area_ids) != expected:
        raise ValueError(
            f"{desc['recording_id']}: {len(area_ids)} retinotopy labels for "
            f"{expected} neurons"
        )

    plane_local_indices = []
    region_chunks = []
    offset = 0
    for size in plane_sizes:
        ids = area_ids[offset:offset + size]
        keep = np.isin(ids, np.fromiter(AREA_ID_TO_REGION, dtype=np.int16))
        local = np.flatnonzero(keep)
        plane_local_indices.append(local)
        region_chunks.append(
            np.fromiter((AREA_ID_TO_REGION[int(x)] for x in ids[local]), dtype=np.int16)
        )
        offset += size
    return plane_local_indices, np.concatenate(region_chunks)


def convert_session(
    desc: dict,
    speed_edges: np.ndarray,
    category_to_id: dict[str, int],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, dict]:
    beh = load_behavior(desc)
    spk_path = DATA_ROOT / "spk" / f"{desc['recording_id']}_neural_data.npy"
    spk_obj = np.load(spk_path, allow_pickle=True).item()
    planes = spk_obj["spks"]
    nframes = min(a.shape[1] for a in planes)
    planes = [a[:, :nframes] for a in planes]

    plane_keep, region_idx = neuron_selection(desc, [a.shape[0] for a in planes])
    n_neurons = len(region_idx)
    valid = retained_frame_mask(beh, nframes)
    trial_stamp = np.asarray(beh["ft_trInd"][:nframes])
    frame_times = np.asarray(beh["ft"][:nframes], dtype=np.float64)
    frame_pos = np.asarray(beh["ft_Pos"][:nframes], dtype=np.float64)
    frame_speed = np.asarray(beh["ft_RunSpeed"][:nframes], dtype=np.float64)
    ntrials = int(beh["ntrials"])

    lick_frame = np.rint(np.asarray(beh["LickFr"], dtype=np.float64)).astype(np.int64)
    lick_trial = np.asarray(beh["LickTrind"], dtype=np.float64)
    day = session_day(desc)

    neural_trials: list[np.ndarray] = []
    input_trials: list[np.ndarray] = []
    output_trials: list[np.ndarray] = []
    dropped_empty = 0

    for trial in range(ntrials):
        frames = np.flatnonzero(valid & (trial_stamp == trial))
        if len(frames) == 0:
            dropped_empty += 1
            continue
        T = len(frames)

        # Allocate once, then copy each imaging plane directly into final row order.
        neural = np.empty((n_neurons, T), dtype=np.float32)
        row = 0
        for plane, keep in zip(planes, plane_keep):
            n = len(keep)
            neural[row:row + n] = plane[np.ix_(keep, frames)]
            row += n

        elapsed_s = (frame_times[frames] - float(beh["Trial_start_time"][trial])) * 86_400.0
        to_cue_s = (float(beh["SoundTime"][trial]) - frame_times[frames]) * 86_400.0
        inputs = np.empty((4, T), dtype=np.float32)
        inputs[0] = to_cue_s
        inputs[1] = day
        inputs[2] = elapsed_s
        inputs[3] = float(bool(beh["isRew"][trial]))

        outputs = np.empty((4, T), dtype=np.int16)
        category = visual_category(beh["WallName"][trial])
        outputs[0] = category_to_id[category]

        # Licks are assigned to their nearest original imaging frame. If that frame
        # was removed by the paper's running mask, the lick is removed as well.
        trial_licks = lick_frame[lick_trial == trial]
        outputs[1] = np.isin(frames, trial_licks).astype(np.int16)

        # Source positions are decimetres; the requested bins are four 1 m bins.
        outputs[2] = np.clip(np.floor(frame_pos[frames] / 10.0), 0, 3).astype(np.int16)
        outputs[3] = np.digitize(frame_speed[frames], speed_edges, right=False).astype(np.int16)

        neural_trials.append(neural)
        input_trials.append(inputs)
        output_trials.append(outputs)

    info = {
        "recording_id": desc["recording_id"],
        "experiment_type": desc["experiment_type"],
        "source_session_number": desc["entry"].get("sess#"),
        "decoder_day_value": day,
        "n_source_trials": ntrials,
        "n_retained_trials": len(neural_trials),
        "n_empty_trials_dropped": dropped_empty,
        "n_source_neurons": int(sum(a.shape[0] for a in planes)),
        "n_visual_cortex_neurons": n_neurons,
    }
    del spk_obj, planes, beh
    gc.collect()
    return neural_trials, input_trials, output_trials, region_idx, info


def build_dataset(data_root: Path) -> dict:
    global DATA_ROOT
    DATA_ROOT = data_root
    sessions = discover_sessions(data_root)
    print(f"Discovered {len(sessions)} unique imaging recordings", flush=True)

    categories = set()
    for desc in sessions:
        beh = load_behavior(desc)
        categories.update(visual_category(x) for x in beh["WallName"])
    category_names = sorted(categories)
    category_to_id = {name: i for i, name in enumerate(category_names)}

    speed_edges, timing_stats = compute_speed_edges(sessions)
    print(f"Global running-speed quartiles: {speed_edges}", flush=True)

    subjects = sorted({str(x["entry"]["mname"]) for x in sessions})
    subject_to_id = {name: i for i, name in enumerate(subjects)}
    neural_all, input_all, output_all, regions_all = [], [], [], []
    subject_idx = []
    session_info = []

    for i, desc in enumerate(sessions):
        neural, inputs, outputs, regions, info = convert_session(
            desc, speed_edges, category_to_id
        )
        if len(neural) < 2:
            raise ValueError(f"{desc['recording_id']} retained fewer than two trials")
        neural_all.append(neural)
        input_all.append(inputs)
        output_all.append(outputs)
        regions_all.append(regions)
        subject_idx.append(subject_to_id[str(desc["entry"]["mname"])])
        session_info.append(info)
        print(
            f"Converted {i + 1}/{len(sessions)} {desc['recording_id']}: "
            f"{len(neural)} trials, {len(regions)} neurons",
            flush=True,
        )

    return {
        "neural": neural_all,
        "input": input_all,
        "output": output_all,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int16),
        "brain_regions": REGION_NAMES,
        "brain_region_idx": regions_all,
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
            category_names,
            ["not_licking", "licking"],
            ["0-1 m", "1-2 m", "2-3 m", "3-4 m"],
            ["Q1 (slowest)", "Q2", "Q3", "Q4 (fastest)"],
        ],
        "metadata": {
            "task_description": (
                "Head-fixed mice ran through 4 m natural-texture VR corridors; "
                "neural activity predicts stimulus category, licking, position, and speed."
            ),
            # Imaging was nominally approximately 3 Hz in the paper. Exact frame
            # timestamps drive event variables; observed timing is also recorded.
            "time_bin_size": 1000.0 / 3.0,
            "temporal_alignment_event": "entry into the 4 m textured corridor",
            "off_start": 0.0,
            "off_end": None,
            "source_neural_signal": "Suite2p non-negative deconvolved fluorescence (spks)",
            "source_position_unit": "decimetres",
            "source_speed_unit": "centimetres per second",
            "frame_filter": "ft_move > 0 and ft_CorrSpc, matching paper analysis code",
            "neuron_filter": (
                "Retinotopically assigned V1/mHV/lHV/aHV neurons; iarea -1 and 7 excluded"
            ),
            "visual_category_rule": (
                "WallName crop/version suffixes collapsed to circle, leaf, rock, or wood"
            ),
            "day_rule": (
                "Imaging_Exp_info sess#; missing train2-after values inferred as stage 1"
            ),
            "lick_alignment_rule": "nearest original imaging frame (rounded LickFr)",
            "speed_binning_rule": "global quartiles over retained decoder samples",
            **timing_stats,
            "session_info": session_info,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "converted_data.pkl")
    args = parser.parse_args()

    data = build_dataset(args.data_root.resolve())
    print(f"Writing {args.output} ...", flush=True)
    with args.output.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_gib = args.output.stat().st_size / (1024 ** 3)
    print(f"Wrote {args.output} ({size_gib:.2f} GiB)", flush=True)


if __name__ == "__main__":
    main()
