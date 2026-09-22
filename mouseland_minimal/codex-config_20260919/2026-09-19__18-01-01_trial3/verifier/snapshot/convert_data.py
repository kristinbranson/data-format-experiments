#!/usr/bin/env python3
"""Convert the Zhong et al. imaging release to the decoder data format.

The paper's common preprocessing is retained: Suite2p deconvolved activity is
linearly interpolated over cumulative VR position using only frames on which
the VR moved.  The decoder dataset keeps the 4 m textured corridor (40 x
0.1 m samples) and excludes the following 2 m gray interval.

This is intentionally a large conversion (about 274 GB).  No arbitrary neuron
or trial subsampling is performed.
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np


FRAME_RATE_HZ = 3.17
VR_SPEED_DM_S = 6.0  # 60 cm/s, expressed in the behavior file's decimeters.
CORRIDOR_BINS = 40
FULL_TRIAL_BINS = 60
AREA_NAMES = ["V1", "mHV", "lHV", "aHV"]
CATEGORY_NAMES = ["circle", "leaf", "rock", "brick"]


def physical_id(row: dict) -> tuple[str, str, str]:
    return row["mname"], row["datexp"], str(row["blk"])


def behavior_key(row: dict) -> str:
    key = "_".join(physical_id(row))
    if "stimtype" in row:
        key += "_" + str(row["stimtype"])
    return key


def visual_category(name: str) -> int:
    """Collapse image exemplars/swaps to the four paper stimulus categories."""
    name = str(name).lower()
    if name.startswith("circle"):
        return 0
    if name.startswith("leaf"):
        return 1
    if name.startswith("rock"):
        return 2
    # The raw files call the brick-texture family "wood".
    if name.startswith("wood") or name.startswith("brick"):
        return 3
    raise ValueError(f"Unrecognized visual stimulus name: {name!r}")


def day_value(references: list[tuple[str, dict]]) -> float:
    """Use the release's explicit training-day/session annotations.

    Some later training recordings have a ``days`` field; the remaining
    before/after and test recordings use ``sess#``.  A physical recording can
    occur in several paper-analysis groups, so the minimum session annotation
    avoids relabeling a pre-exposure recording as a later analysis session.
    """
    explicit = [row["days"] for _, row in references if "days" in row]
    if explicit:
        return float(explicit[0])
    sessions = [row["sess#"] for _, row in references if "sess#" in row]
    return float(min(sessions)) if sessions else 0.0


def extract_behavior(d: dict) -> dict:
    """Retain only fields used by conversion, releasing multi-GB raw dicts."""
    return {
        "ntrials": int(d["ntrials"]),
        "ft_move": np.asarray(d["ft_move"]),
        "ft_PosCum": np.asarray(d["ft_PosCum"]),
        "run_pos": np.asarray(d["run_pos"], dtype=np.float32)[:, :CORRIDOR_BINS],
        "SoundDelPos": np.asarray(d["SoundDelPos"]),
        "isRew": np.asarray(d["isRew"], dtype=bool),
        "WallName": np.asarray(d["WallName"]),
        "LickTrind": np.asarray(d["LickTrind"]).astype(np.int64),
        "LickPos": np.asarray(d["LickPos"]),
        "Corridor_Length": float(d["Corridor_Length"]),
        "Texture_Length": float(d["Texture_Length"]),
    }


def load_inventory(data_root: Path):
    exp_info = np.load(
        data_root / "beh" / "Imaging_Exp_info.npy", allow_pickle=True
    ).item()

    sessions: list[tuple[str, str, str]] = []
    references: OrderedDict[tuple[str, str, str], list[tuple[str, dict]]] = OrderedDict()
    for exp_type, rows in exp_info.items():
        for row in rows:
            pid = physical_id(row)
            if pid not in references:
                references[pid] = []
                sessions.append(pid)
            references[pid].append((exp_type, row))

    # Load every behavior file at most once. For duplicated analysis references,
    # the underlying full behavior record is identical; swap1/swap2 differ only
    # in the paper's stimulus-ID lookup and share all trial/frame fields.
    behaviors: dict[tuple[str, str, str], dict] = {}
    for exp_type, rows in exp_info.items():
        unresolved = [row for row in rows if physical_id(row) not in behaviors]
        if not unresolved:
            continue
        path = data_root / "beh" / f"Beh_{exp_type}.npy"
        raw = np.load(path, allow_pickle=True).item()
        for row in rows:
            pid = physical_id(row)
            if pid in behaviors:
                continue
            key = behavior_key(row)
            if key in raw:
                behaviors[pid] = extract_behavior(raw[key])
        del raw
        gc.collect()

    missing = [pid for pid in sessions if pid not in behaviors]
    if missing:
        raise KeyError(f"No behavior record found for {missing}")
    return sessions, references, behaviors


def area_indices(iarea: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the visual-area grouping used by code/utils.py::neu_area_ID."""
    region = np.full(len(iarea), -1, dtype=np.int8)
    region[iarea == 8] = 0
    region[np.isin(iarea, [0, 1, 2, 9])] = 1
    region[np.isin(iarea, [5, 6])] = 2
    region[np.isin(iarea, [3, 4])] = 3
    keep = region >= 0  # equivalently excludes iarea -1 and 7 (outside visual cortex)
    return keep, region[keep]


def interpolation_lookup(x: np.ndarray, targets: np.ndarray):
    """Indices/weights for scipy interp1d(..., fill_value='extrapolate')."""
    if len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("Running-only cumulative VR position is not strictly increasing")
    hi = np.searchsorted(x, targets, side="left")
    hi = np.clip(hi, 1, len(x) - 1)
    lo = hi - 1
    # Keep the interpolation arithmetic in float64, as scipy.interp1d does for
    # these float64 positions, then cast once into the decoder's float32 cube.
    weight = (targets - x[lo]) / (x[hi] - x[lo])
    return lo, hi, weight


def interpolate_session(
    planes: list[np.ndarray],
    keep: np.ndarray,
    behavior: dict,
    chunk_neurons: int,
) -> np.ndarray:
    """Return contiguous (trial, retained neuron, 40-bin) float32 activity."""
    nframes = int(planes[0].shape[1])
    if any(p.shape[1] != nframes for p in planes):
        raise ValueError("Neural planes have inconsistent frame counts")
    if len(behavior["ft_move"]) < nframes:
        raise ValueError("Behavior frame stream is shorter than neural activity")

    moving = behavior["ft_move"][:nframes] > 0
    x = np.asarray(behavior["ft_PosCum"][:nframes][moving], dtype=np.float64)
    targets = (
        np.arange(behavior["ntrials"], dtype=np.float64)[:, None] * FULL_TRIAL_BINS
        + np.arange(CORRIDOR_BINS, dtype=np.float64)[None, :]
    ).ravel()
    lo, hi, weight = interpolation_lookup(x, targets)

    nneurons = int(keep.sum())
    cube = np.empty(
        (behavior["ntrials"], nneurons, CORRIDOR_BINS), dtype=np.float32
    )
    source_offset = 0
    dest_offset = 0
    for plane in planes:
        plane_keep = np.flatnonzero(keep[source_offset : source_offset + len(plane)])
        source_offset += len(plane)
        for start in range(0, len(plane_keep), chunk_neurons):
            ids = plane_keep[start : start + chunk_neurons]
            y = plane[ids][:, moving]
            # Linear interpolation/extrapolation, identical to the paper helper's
            # scipy interp1d call but vectorized across neurons.
            values = y[:, lo] * (1.0 - weight) + y[:, hi] * weight
            count = len(ids)
            cube[:, dest_offset : dest_offset + count, :] = values.reshape(
                count, behavior["ntrials"], CORRIDOR_BINS
            ).transpose(1, 0, 2)
            dest_offset += count
            del y, values
    if source_offset != len(keep) or dest_offset != nneurons:
        raise RuntimeError("Neuron accounting mismatch while interpolating")
    return cube


def trial_covariates(behavior: dict, day: float, speed_edges: np.ndarray):
    positions = np.arange(CORRIDOR_BINS, dtype=np.float32)
    elapsed_s = positions / VR_SPEED_DM_S
    position_class = (np.arange(CORRIDOR_BINS) // 10).astype(np.int16)

    lick = np.zeros((behavior["ntrials"], CORRIDOR_BINS), dtype=np.int16)
    lick_bin = np.floor(behavior["LickPos"]).astype(np.int64)
    valid = (
        (behavior["LickTrind"] >= 0)
        & (behavior["LickTrind"] < behavior["ntrials"])
        & (lick_bin >= 0)
        & (lick_bin < CORRIDOR_BINS)
    )
    lick[behavior["LickTrind"][valid], lick_bin[valid]] = 1

    session_inputs = []
    session_outputs = []
    for trial in range(behavior["ntrials"]):
        cue_dm = float(np.mod(behavior["SoundDelPos"][trial], FULL_TRIAL_BINS))
        inp = np.empty((4, CORRIDOR_BINS), dtype=np.float32)
        inp[0] = (cue_dm - positions) / VR_SPEED_DM_S
        inp[1] = day
        inp[2] = elapsed_s
        inp[3] = float(behavior["isRew"][trial])

        out = np.empty((4, CORRIDOR_BINS), dtype=np.int16)
        out[0] = visual_category(behavior["WallName"][trial])
        out[1] = lick[trial]
        out[2] = position_class
        out[3] = np.digitize(behavior["run_pos"][trial], speed_edges).astype(np.int16)
        session_inputs.append(inp)
        session_outputs.append(out)
    return session_inputs, session_outputs


def convert(data_root: Path, output_path: Path, chunk_neurons: int = 256):
    sessions, references, behaviors = load_inventory(data_root)
    if len(sessions) != 89:
        raise ValueError(f"Expected 89 unique imaging recordings, found {len(sessions)}")

    all_speeds = np.concatenate([behaviors[pid]["run_pos"].ravel() for pid in sessions])
    speed_edges = np.quantile(all_speeds, [0.25, 0.50, 0.75]).astype(np.float32)
    if not np.all(np.isfinite(speed_edges)):
        raise ValueError("Non-finite running-speed quartiles")

    subjects: list[str] = []
    neural, inputs, outputs = [], [], []
    subject_idx, brain_region_idx, session_info = [], [], []

    for session_number, pid in enumerate(sessions):
        mouse, date, block = pid
        if mouse not in subjects:
            subjects.append(mouse)
        subject_idx.append(subjects.index(mouse))

        retino_path = data_root / "retinotopy" / f"{mouse}_{date}_trans.npz"
        iarea = np.load(retino_path, allow_pickle=True)["iarea"]
        keep, regions = area_indices(iarea)

        spk_path = data_root / "spk" / f"{mouse}_{date}_{block}_neural_data.npy"
        raw_neural = np.load(spk_path, allow_pickle=True).item()
        planes = raw_neural["spks"]
        if sum(len(p) for p in planes) != len(iarea):
            raise ValueError(f"Neuron/retinotopy mismatch for {pid}")

        behavior = behaviors[pid]
        cube = interpolate_session(planes, keep, behavior, chunk_neurons)
        session_neural = [cube[trial] for trial in range(behavior["ntrials"])]
        day = day_value(references[pid])
        session_input, session_output = trial_covariates(behavior, day, speed_edges)

        neural.append(session_neural)
        inputs.append(session_input)
        outputs.append(session_output)
        brain_region_idx.append(regions.astype(np.int64))
        session_info.append(
            {
                "subject": mouse,
                "date": date,
                "block": block,
                "training_day_or_session": day,
                "analysis_groups": sorted({typ for typ, _ in references[pid]}),
                "n_trials": behavior["ntrials"],
                "n_neurons_released": int(len(iarea)),
                "n_neurons_visual_cortex": int(keep.sum()),
            }
        )
        size_gb = cube.nbytes / 1e9
        print(
            f"[{session_number + 1:02d}/{len(sessions)}] {mouse} {date} block {block}: "
            f"{behavior['ntrials']} trials, {keep.sum()} neurons, {size_gb:.2f} GB",
            flush=True,
        )
        del raw_neural, planes, cube, iarea, keep, regions
        gc.collect()

    data = {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": AREA_NAMES,
        "brain_region_idx": brain_region_idx,
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
            CATEGORY_NAMES,
            ["not licking", "licking"],
            ["0-1 m", "1-2 m", "2-3 m", "3-4 m"],
            ["0-25%", "25-50%", "50-75%", "75-100%"],
        ],
        "metadata": {
            "task_description": (
                "Head-fixed mice ran through 4 m virtual corridors containing "
                "natural-texture stimuli; some cohorts learned which corridor "
                "made water reward available after a randomized sound cue."
            ),
            "time_bin_size": 1000.0 / VR_SPEED_DM_S,
            "temporal_alignment_event": "entry into the 4 m textured corridor (trial start)",
            "off_start": 0.0,
            "off_end": CORRIDOR_BINS / VR_SPEED_DM_S,
            "source_neural_signal": "Suite2p non-negative deconvolved fluorescence (tau=0.75 s)",
            "source_imaging_frame_rate_hz": FRAME_RATE_HZ,
            "resampling": (
                "Linear interpolation over cumulative VR position using only ft_move>0 "
                "frames, matching code/utils.py spk_pos_interp; first 40 of the "
                "paper's 60 0.1-m bins retained (4-m corridor, gray space excluded)."
            ),
            "virtual_corridor_speed_cm_s": 60.0,
            "spatial_bin_size_m": 0.1,
            "running_speed_quartile_edges": speed_edges.tolist(),
            "neuron_filter": (
                "Retain released Suite2p cells assigned to visual cortex; iarea -1 "
                "and 7 excluded, and remaining iarea labels grouped exactly as "
                "code/utils.py neu_area_ID."
            ),
            "lick_binning": "A spatial sample is 1 if one or more licks occurred in its 0.1-m bin.",
            "visual_category_mapping": (
                "Exemplar suffixes and spatial swaps collapsed to circle, leaf, rock, "
                "or brick; raw 'wood' names are the paper's brick-texture family."
            ),
            "training_day_definition": (
                "Explicit Imaging_Exp_info 'days' when supplied, otherwise the "
                "release's sess# annotation; constant within a trial."
            ),
            "session_info": session_info,
        },
    }

    tmp_path = output_path.with_name(output_path.name + ".tmp")
    print(f"Writing {output_path} via {tmp_path} ...", flush=True)
    with open(tmp_path, "wb", buffering=16 * 1024 * 1024) as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, output_path)
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1e9:.2f} GB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/app/data"))
    parser.add_argument("--output", type=Path, default=Path("/app/converted_data.pkl"))
    parser.add_argument(
        "--chunk-neurons",
        type=int,
        default=256,
        help="Interpolation work chunk; affects memory/speed, not results.",
    )
    args = parser.parse_args()
    if args.chunk_neurons < 1:
        parser.error("--chunk-neurons must be positive")
    convert(args.data_root, args.output, args.chunk_neurons)


if __name__ == "__main__":
    main()
