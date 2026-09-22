#!/usr/bin/env python3
"""Convert the Zhong et al. imaging data to the neural-decoder format.

The paper contains 89 unique imaging recordings, but several recordings occur in
more than one analysis-specific ``Beh_*.npy`` file.  This converter includes each
physical recording once.  It uses the same valid-frame definition as
``code/utils.py::Get_dprime_selective_neuron``: the animal must be in the 4 m
textured corridor and the VR must be moving (``ft_move > 0``).

The paper's neural arrays are Suite2p non-negative deconvolved fluorescence
traces.  No additional smoothing, normalization, or spatial interpolation is
applied here: spatial interpolation in the paper was for position-aligned
analyses, whereas the requested decoder alignment is temporal (corridor entry).
"""

from __future__ import annotations

import argparse
import gc
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


FRAME_RATE_HZ = 3.17
SECONDS_PER_DAY = 86_400.0

# Exact grouping used by code/utils.py::neu_area_ID.  Area -1 (outside the
# retinotopic atlas) and area 7 (not assigned to one of the four visual-area
# groups used in the paper) are excluded.
AREA_CODES = {
    "V1": (8,),
    "medial": (0, 1, 2, 9),
    "anterior": (3, 4),
    "lateral": (5, 6),
}


def session_id(record: dict) -> str:
    return f"{record['mname']}_{record['datexp']}_{record['blk']}"


def behavior_key(record: dict) -> str:
    key = session_id(record)
    if "stimtype" in record:
        key += f"_{record['stimtype']}"
    return key


def make_source_map(exp_info: dict, spk_root: Path) -> dict:
    """Map every unique neural recording to one behavior copy.

    Repeated behavior copies differ only in analysis labels such as ``stim_id``;
    trial timing, actual ``WallName``, and behavioral streams are the same.  The
    first occurrence in the repository's experiment table is therefore a
    deterministic canonical copy.
    """
    sources = {}
    for experiment_type, records in exp_info.items():
        for record in records:
            sid = session_id(record)
            sources.setdefault(
                sid,
                {
                    "experiment_type": experiment_type,
                    "behavior_key": behavior_key(record),
                    "record": record,
                },
            )

    spk_sessions = {
        path.name.removesuffix("_neural_data.npy")
        for path in spk_root.glob("*_neural_data.npy")
    }
    if set(sources) != spk_sessions:
        missing_behavior = sorted(spk_sessions - set(sources))
        missing_neural = sorted(set(sources) - spk_sessions)
        raise RuntimeError(
            "Experiment table/neural file mismatch: "
            f"missing behavior={missing_behavior}, missing neural={missing_neural}"
        )
    return sources


def compact_behavior(beh: dict) -> dict:
    """Keep only fields needed by this conversion, detached from the large dict."""
    array_fields = (
        "WallName",
        "isRew",
        "ft",
        "ft_trInd",
        "ft_CorrSpc",
        "ft_move",
        "ft_RunSpeed",
        "ft_Pos",
        "Trial_start_time",
        "SoundTime",
        "LickFr",
    )
    out = {field: np.array(beh[field], copy=True) for field in array_fields}
    out["ntrials"] = int(beh["ntrials"])
    out["texture_length_dm"] = float(beh["Texture_Length"])
    out["reward_mode"] = str(beh["Reward_Mode"])
    return out


def load_behaviors(data_root: Path, sources: dict) -> dict:
    """Load each required behavior file once and retain compact session streams."""
    grouped = defaultdict(list)
    for sid, source in sources.items():
        grouped[source["experiment_type"]].append((sid, source["behavior_key"]))

    behaviors = {}
    for experiment_type, requested in grouped.items():
        path = data_root / "beh" / f"Beh_{experiment_type}.npy"
        behavior_file = np.load(path, allow_pickle=True).item()
        for sid, key in requested:
            if key not in behavior_file:
                raise KeyError(f"{key!r} is absent from {path}")
            behaviors[sid] = compact_behavior(behavior_file[key])
        del behavior_file
        gc.collect()
    return behaviors


def valid_frame_mask(beh: dict, nframes: int) -> np.ndarray:
    """Reproduce the paper's running-and-textured-corridor frame mask."""
    nframes = min(
        nframes,
        len(beh["ft"]),
        len(beh["ft_trInd"]),
        len(beh["ft_CorrSpc"]),
        len(beh["ft_move"]),
    )
    trial = beh["ft_trInd"][:nframes]
    valid_trial = (
        np.isfinite(trial)
        & (trial >= 0)
        & (trial < beh["ntrials"])
    )
    return (
        valid_trial
        & beh["ft_CorrSpc"][:nframes].astype(bool)
        & (beh["ft_move"][:nframes] > 0)
    )


def global_speed_edges(behaviors: dict) -> np.ndarray:
    """Compute dataset-wide quartiles over exactly the retained samples."""
    speeds = []
    for beh in behaviors.values():
        mask = valid_frame_mask(beh, len(beh["ft"]))
        speeds.append(beh["ft_RunSpeed"][: len(mask)][mask].astype(np.float32))
    all_speeds = np.concatenate(speeds)
    edges = np.quantile(all_speeds, (0.25, 0.50, 0.75))
    if not np.all(np.diff(edges) > 0):
        raise RuntimeError(f"Running-speed quartiles are not distinct: {edges}")
    return edges.astype(np.float64)


def area_labels(iarea: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the retained-neuron mask and its four exclusive region indices."""
    region = np.full(len(iarea), -1, dtype=np.int8)
    names = list(AREA_CODES)
    for idx, name in enumerate(names):
        region[np.isin(iarea, AREA_CODES[name])] = idx
    keep = region >= 0
    return keep, region[keep].astype(np.int64)


def visual_vocabulary(behaviors: dict) -> list[str]:
    return sorted({str(wall) for beh in behaviors.values() for wall in beh["WallName"]})


def elapsed_days_by_session(session_ids: list[str], sources: dict) -> dict[str, float]:
    """Elapsed calendar days from each subject's earliest imaging session.

    The experiment table does not provide a complete training-day counter (only
    selected later recordings have a ``days`` field).  Dates are available for
    every recording, so elapsed calendar day is the only uniform, non-imputed
    continuous measure across supervised, unsupervised, grating, and naive mice.
    """
    parsed = {}
    first_date = {}
    for sid in session_ids:
        record = sources[sid]["record"]
        date = datetime.strptime(record["datexp"], "%Y_%m_%d").date()
        subject = str(record["mname"])
        parsed[sid] = (subject, date)
        first_date[subject] = min(date, first_date.get(subject, date))
    return {
        sid: float((date - first_date[subject]).days)
        for sid, (subject, date) in parsed.items()
    }


def convert(data_root: Path, output_path: Path) -> dict:
    exp_info = np.load(
        data_root / "beh" / "Imaging_Exp_info.npy", allow_pickle=True
    ).item()
    sources = make_source_map(exp_info, data_root / "spk")
    session_ids = sorted(sources)
    behaviors = load_behaviors(data_root, sources)

    speed_edges = global_speed_edges(behaviors)
    stimulus_values = visual_vocabulary(behaviors)
    stimulus_to_id = {name: idx for idx, name in enumerate(stimulus_values)}
    day_by_session = elapsed_days_by_session(session_ids, sources)
    subjects = sorted({sources[sid]["record"]["mname"] for sid in session_ids})
    subject_to_id = {name: idx for idx, name in enumerate(subjects)}

    neural_sessions = []
    input_sessions = []
    output_sessions = []
    brain_region_idx = []
    subject_idx = []
    session_info = []

    for session_number, sid in enumerate(session_ids, start=1):
        source = sources[sid]
        record = source["record"]
        beh = behaviors[sid]
        spk_path = data_root / "spk" / f"{sid}_neural_data.npy"
        raw_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
        if not raw_planes:
            raise RuntimeError(f"No neural planes in {spk_path}")
        nframes = min(plane.shape[1] for plane in raw_planes)
        if any(plane.dtype != np.float32 for plane in raw_planes):
            raise TypeError(f"Expected float32 deconvolved traces in {spk_path}")

        retino_name = f"{record['mname']}_{record['datexp']}_trans.npz"
        iarea = np.load(data_root / "retinotopy" / retino_name)["iarea"]
        raw_neuron_count = sum(plane.shape[0] for plane in raw_planes)
        if len(iarea) != raw_neuron_count:
            raise RuntimeError(
                f"Retinotopy/neural neuron mismatch in {sid}: "
                f"{len(iarea)} != {raw_neuron_count}"
            )
        keep_neuron, regions = area_labels(iarea)

        frame_mask = valid_frame_mask(beh, nframes)
        frame_indices = np.flatnonzero(frame_mask)
        trial_ids = beh["ft_trInd"][: len(frame_mask)][frame_mask].astype(np.int64)
        if len(frame_indices) == 0:
            raise RuntimeError(f"No valid running corridor frames in {sid}")
        if np.any(np.diff(trial_ids) < 0):
            raise RuntimeError(f"Trial indices are not chronological in {sid}")

        # Select the valid rows and columns plane by plane.  This avoids first
        # concatenating the full (often multi-gigabyte) unfiltered recording.
        selected_planes = []
        offset = 0
        for plane in raw_planes:
            local_keep = np.flatnonzero(keep_neuron[offset : offset + plane.shape[0]])
            selected_planes.append(plane[np.ix_(local_keep, frame_indices)])
            offset += plane.shape[0]
        selected_neural = np.concatenate(selected_planes, axis=0)
        del selected_planes, raw_planes

        lick_frame = beh["LickFr"]
        lick_frame = lick_frame[np.isfinite(lick_frame)].astype(np.int64)
        session_neural = []
        session_input = []
        session_output = []

        kept_trials = np.unique(trial_ids)
        for trial in kept_trials:
            columns = np.flatnonzero(trial_ids == trial)
            frames = frame_indices[columns]
            times = beh["ft"][frames]
            ntime = len(frames)

            # Each trial owns a contiguous array so the pickle is self-contained
            # and does not retain a whole-session backing allocation per view.
            trial_neural = np.ascontiguousarray(selected_neural[:, columns])

            time_to_sound = (
                (beh["SoundTime"][trial] - times) * SECONDS_PER_DAY
            ).astype(np.float32)
            time_from_start = (
                (times - beh["Trial_start_time"][trial]) * SECONDS_PER_DAY
            ).astype(np.float32)
            trial_input = np.empty((4, ntime), dtype=np.float32)
            trial_input[0] = time_to_sound
            trial_input[1] = day_by_session[sid]
            trial_input[2] = time_from_start
            trial_input[3] = float(beh["isRew"][trial])

            # Texture_Length is 40 decimeters.  Using the recorded setting keeps
            # the four bins exactly 1 m even if a future source file differs.
            position_bin = np.floor(
                beh["ft_Pos"][frames] / (beh["texture_length_dm"] / 4.0)
            ).astype(np.int64)
            position_bin = np.clip(position_bin, 0, 3)
            speed_bin = np.digitize(
                beh["ft_RunSpeed"][frames], speed_edges, right=False
            ).astype(np.int64)

            trial_output = np.empty((4, ntime), dtype=np.int64)
            trial_output[0] = stimulus_to_id[str(beh["WallName"][trial])]
            trial_output[1] = np.isin(frames, lick_frame).astype(np.int64)
            trial_output[2] = position_bin
            trial_output[3] = speed_bin

            session_neural.append(trial_neural)
            session_input.append(trial_input)
            session_output.append(trial_output)

        if len(session_neural) < 2:
            raise RuntimeError(f"Fewer than two usable trials in {sid}")

        neural_sessions.append(session_neural)
        input_sessions.append(session_input)
        output_sessions.append(session_output)
        brain_region_idx.append(regions)
        subject_idx.append(subject_to_id[str(record["mname"])])
        session_info.append(
            {
                "session_id": sid,
                "subject": str(record["mname"]),
                "date": str(record["datexp"]),
                "block": str(record["blk"]),
                "source_experiment_type": source["experiment_type"],
                "reward_mode": beh["reward_mode"],
                "training_day": day_by_session[sid],
                "n_trials": len(session_neural),
                "n_valid_frames": len(frame_indices),
                "n_neurons_raw": raw_neuron_count,
                "n_neurons_retained": int(keep_neuron.sum()),
            }
        )

        del selected_neural
        gc.collect()
        print(
            f"[{session_number:02d}/{len(session_ids)}] {sid}: "
            f"{len(session_neural)} trials, {len(frame_indices)} frames, "
            f"{keep_neuron.sum()}/{raw_neuron_count} neurons",
            flush=True,
        )

    data = {
        "neural": neural_sessions,
        "input": input_sessions,
        "output": output_sessions,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": list(AREA_CODES),
        "brain_region_idx": brain_region_idx,
        "input_names": [
            "time_to_sound_cue_s",
            "training_day",
            "time_since_trial_start_s",
            "reward_available",
        ],
        "output_names": [
            "visual_stimulus",
            "licking",
            "corridor_position_bin",
            "running_speed_quartile",
        ],
        "output_values": [
            stimulus_values,
            ["not licking", "licking"],
            ["0-1 m", "1-2 m", "2-3 m", "3-4 m"],
            ["lowest 25%", "25-50%", "50-75%", "highest 25%"],
        ],
        "metadata": {
            "task_description": (
                "Decode corridor texture, licking, 1 m corridor position, and "
                "running-speed quartile from visual-cortex deconvolved activity."
            ),
            "time_bin_size": 1000.0 / FRAME_RATE_HZ,
            "temporal_alignment_event": "entry into the 4 m textured corridor",
            "off_start": 0.0,
            "off_end": None,
            "neural_signal": "Suite2p non-negative deconvolved fluorescence",
            "nominal_frame_rate_hz": FRAME_RATE_HZ,
            "frame_filter": "ft_CorrSpc & (ft_move > 0) with a valid trial index",
            "time_to_sound_sign": "positive before the cue; negative after the cue",
            "training_day_definition": (
                "elapsed calendar days since the subject's earliest imaging session"
            ),
            "running_speed_quartile_edges": speed_edges.tolist(),
            "position_source_units": "decimeters",
            "neuron_filter": (
                "retained only V1, medial, anterior, and lateral visual-area "
                "groups defined by code/utils.py::neu_area_ID"
            ),
            "session_info": session_info,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {output_path}", flush=True)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/app/data"))
    parser.add_argument("--output", type=Path, default=Path("/app/converted_data.pkl"))
    args = parser.parse_args()
    convert(args.data_root, args.output)


if __name__ == "__main__":
    main()
