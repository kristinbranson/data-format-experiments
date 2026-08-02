import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, maximum_filter1d, minimum_filter1d

try:
    from suite2p.extraction.dcnv import preprocess as suite2p_preprocess
except Exception:
    suite2p_preprocess = None


REFERENCE = {
    "paper": "Majnik et al. (2025) Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p",
    "doi": "10.7554/eLife.107540.1",
}

FULL_SUBJECTS = ["jm031", "jm032", "jm038", "jm039", "jm040", "jm046"]
SAMPLE_SUBJECTS = FULL_SUBJECTS


@dataclass
class SessionRecord:
    subject: str
    session_name: str
    session_date: str
    neural_trials: list[np.ndarray]
    input_trials: list[np.ndarray]
    output_continuous_trials: list[np.ndarray]
    brain_region_idx: np.ndarray
    info: dict[str, Any]


def baseline_correct_fluorescence(
    F: np.ndarray,
    Fneu: np.ndarray,
    ops: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    neucoeff = float(ops.get("neucoeff", 0.7))
    baseline = ops.get("baseline", "maximin")
    win_baseline = float(ops.get("win_baseline", 60.0))
    sig_baseline = float(ops.get("sig_baseline", 10.0))
    prctile_baseline = float(ops.get("prctile_baseline", 8.0))
    fs = float(ops["fs"])

    Fc = F.astype(np.float32, copy=False) - neucoeff * Fneu.astype(np.float32, copy=False)
    Fc = np.asarray(Fc, dtype=np.float32)

    if suite2p_preprocess is not None:
        return suite2p_preprocess(
            Fc.copy(),
            baseline=baseline,
            win_baseline=win_baseline,
            sig_baseline=sig_baseline,
            fs=fs,
            prctile_baseline=prctile_baseline,
            batch_size=128,
            device=device,
        ).astype(np.float32, copy=False)

    win = int(win_baseline * fs)
    if baseline == "maximin":
        Flow = gaussian_filter(Fc, [0.0, sig_baseline])
        Flow = minimum_filter1d(Flow, win, axis=1)
        Flow = maximum_filter1d(Flow, win, axis=1)
    elif baseline == "constant":
        Flow = gaussian_filter(Fc, [0.0, sig_baseline])
        Flow = np.amin(Flow)
    elif baseline in {"prctile", "constant_prctile"}:
        Flow = np.percentile(Fc, prctile_baseline, axis=1, keepdims=True)
    else:
        Flow = 0.0
    return (Fc - Flow).astype(np.float32, copy=False)


def detect_gap_indices(interframe_int: np.ndarray) -> np.ndarray:
    interframe_int = np.asarray(interframe_int, dtype=np.float64)
    if interframe_int.size == 0:
        return np.array([], dtype=np.int64)
    median_dt = float(np.median(interframe_int))
    return np.flatnonzero(interframe_int > 1.5 * median_dt).astype(np.int64)


def interpolate_nans_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if not np.isnan(x).any():
        return x
    valid = ~np.isnan(x)
    if not np.any(valid):
        return np.zeros_like(x, dtype=np.float32)
    idx = np.arange(x.size, dtype=np.float32)
    x[~valid] = np.interp(idx[~valid], idx[valid], x[valid]).astype(np.float32)
    return x


def align_motion_to_imaging(
    motion_energy: np.ndarray,
    tstamps: np.ndarray,
    interframe_int: np.ndarray,
    target_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    motion_energy = np.asarray(motion_energy, dtype=np.float32)
    tstamps = np.asarray(tstamps)
    interframe_int = np.asarray(interframe_int)

    info = {
        "motion_frames_raw": int(motion_energy.shape[0]),
        "imaging_frames": int(target_frames),
        "gap_indices_detected": [],
        "inserted_missing_values": 0,
        "length_fix_strategy": "none",
    }

    if motion_energy.shape[0] == target_frames:
        return motion_energy, info

    diff = int(target_frames - motion_energy.shape[0])
    gap_idx = detect_gap_indices(interframe_int)
    info["gap_indices_detected"] = gap_idx.tolist()

    if diff < 0:
        info["length_fix_strategy"] = "trim_excess_motion_frames"
        return motion_energy[:target_frames], info

    repaired = motion_energy.astype(np.float32, copy=True)
    if diff > 0:
        info["length_fix_strategy"] = "insert_nan_at_detected_camera_gaps_then_interpolate"
        if gap_idx.size >= diff:
            gap_positions = gap_idx[:diff]
        else:
            gap_positions = gap_idx
        offset = 0
        for pos in gap_positions:
            insert_at = int(pos + 1 + offset)
            repaired = np.insert(repaired, insert_at, np.nan).astype(np.float32, copy=False)
            offset += 1
        if repaired.shape[0] < target_frames:
            pad = np.full(target_frames - repaired.shape[0], np.nan, dtype=np.float32)
            repaired = np.concatenate([repaired, pad], axis=0)
        repaired = repaired[:target_frames]
        repaired = interpolate_nans_1d(repaired)
        info["inserted_missing_values"] = int(diff)
    else:
        repaired = repaired[:target_frames]

    return repaired.astype(np.float32, copy=False), info


def average_nonoverlapping(arr: np.ndarray, bin_size: int) -> np.ndarray:
    n_complete = arr.shape[-1] // bin_size
    trimmed = arr[..., : n_complete * bin_size]
    new_shape = (*trimmed.shape[:-1], n_complete, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def split_into_trials(
    neural_binned: np.ndarray,
    motion_binned: np.ndarray,
    fs: float,
    frame_bin: int,
    trial_duration_s: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    time_bin_s = frame_bin / fs
    bins_per_trial = int(round(trial_duration_s / time_bin_s))
    n_complete_trials = neural_binned.shape[1] // bins_per_trial

    neural_binned = neural_binned[:, : n_complete_trials * bins_per_trial]
    motion_binned = motion_binned[: n_complete_trials * bins_per_trial]

    neural_trials = []
    input_trials = []
    output_trials = []

    time_axis = (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * time_bin_s
    for trial_idx in range(n_complete_trials):
        start = trial_idx * bins_per_trial
        end = start + bins_per_trial
        neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
        input_trials.append(time_axis[start:end][np.newaxis, :].astype(np.float32, copy=False))
        output_trials.append(motion_binned[start:end][np.newaxis, :].astype(np.float32, copy=False))

    dropped_bins = int((neural_binned.shape[1] // 1) - n_complete_trials * bins_per_trial)
    info = {
        "time_bin_s": float(time_bin_s),
        "bins_per_trial": int(bins_per_trial),
        "n_trials": int(n_complete_trials),
        "dropped_binned_timepoints": int(dropped_bins),
    }
    return neural_trials, input_trials, output_trials, info


def load_session(
    session_dir: Path,
    device: torch.device,
    frame_bin: int,
    trial_duration_s: float,
) -> SessionRecord:
    suite2p_dir = session_dir / "suite2p" / "plane0"
    move_dir = session_dir / "move_deve"

    F = np.load(suite2p_dir / "F.npy")
    Fneu = np.load(suite2p_dir / "Fneu.npy")
    ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
    iscell = np.load(suite2p_dir / "iscell.npy")
    motion_energy = np.load(move_dir / "motion_energy_glob.npy")
    tstamps = np.load(move_dir / "tstamps.npy")
    interframe_int = np.load(move_dir / "interframe_int.npy")

    iscell_prob = iscell[:, 1]
    if np.any(iscell_prob < 0.5):
        raise ValueError(f"{session_dir}: expected Track2p output already filtered at iscell >= 0.5")

    neural = baseline_correct_fluorescence(F=F, Fneu=Fneu, ops=ops, device=device)
    motion_aligned, motion_info = align_motion_to_imaging(
        motion_energy=motion_energy,
        tstamps=tstamps,
        interframe_int=interframe_int,
        target_frames=neural.shape[1],
    )

    neural_binned = average_nonoverlapping(neural, frame_bin).astype(np.float32, copy=False)
    motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], frame_bin)[0].astype(np.float32, copy=False)

    neural_trials, input_trials, output_trials, trial_info = split_into_trials(
        neural_binned=neural_binned,
        motion_binned=motion_binned,
        fs=float(ops["fs"]),
        frame_bin=frame_bin,
        trial_duration_s=trial_duration_s,
    )

    session_date = session_dir.name.split("_")[0]
    session_info = {
        "subject": session_dir.parent.name,
        "session_name": session_dir.name,
        "session_date": session_date,
        "tracked_neurons": int(neural.shape[0]),
        "raw_imaging_frames": int(neural.shape[1]),
        "raw_motion_frames": int(motion_energy.shape[0]),
        "raw_duration_s": float(neural.shape[1] / float(ops["fs"])),
        "suite2p_ops_subset": {
            "fs": float(ops["fs"]),
            "neucoeff": float(ops.get("neucoeff", 0.7)),
            "baseline": str(ops.get("baseline", "maximin")),
            "sig_baseline": float(ops.get("sig_baseline", 10.0)),
            "win_baseline": float(ops.get("win_baseline", 60.0)),
            "prctile_baseline": float(ops.get("prctile_baseline", 8.0)),
        },
        "motion_alignment": motion_info,
        "trialization": trial_info,
    }

    return SessionRecord(
        subject=session_dir.parent.name,
        session_name=session_dir.name,
        session_date=session_date,
        neural_trials=neural_trials,
        input_trials=input_trials,
        output_continuous_trials=output_trials,
        brain_region_idx=np.zeros(neural.shape[0], dtype=np.int64),
        info=session_info,
    )


def select_session_dirs(data_dir: Path, mode: str) -> list[Path]:
    if mode == "full":
        subjects = FULL_SUBJECTS
        per_subject_limit = None
    elif mode == "sample":
        subjects = SAMPLE_SUBJECTS
        per_subject_limit = 1
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    session_dirs: list[Path] = []
    for subject in subjects:
        subject_dir = data_dir / subject
        sessions = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
        if per_subject_limit is not None:
            sessions = sessions[:per_subject_limit]
        session_dirs.extend(sessions)
    return session_dirs


def build_dataset(session_records: list[SessionRecord], mode: str, frame_bin: int, trial_duration_s: float) -> dict[str, Any]:
    all_motion = []
    for record in session_records:
        for trial in record.output_continuous_trials:
            all_motion.append(trial.reshape(-1))
    all_motion_concat = np.concatenate(all_motion, axis=0).astype(np.float32)
    global_edges = np.quantile(all_motion_concat, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)

    subjects = sorted({record.subject for record in session_records})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}

    neural: list[list[np.ndarray]] = []
    input_data: list[list[np.ndarray]] = []
    output_data: list[list[np.ndarray]] = []
    brain_region_idx: list[np.ndarray] = []
    subject_idx = []

    session_info = []
    for record in session_records:
        neural.append(record.neural_trials)
        input_data.append(record.input_trials)
        discretized_trials = []
        for trial in record.output_continuous_trials:
            bins = np.digitize(trial, global_edges, right=False).astype(np.int64)
            discretized_trials.append(bins)
        output_data.append(discretized_trials)
        brain_region_idx.append(record.brain_region_idx)
        subject_idx.append(subject_to_idx[record.subject])
        info = dict(record.info)
        normalized_trials = []
        for trial in record.output_continuous_trials:
            normalized_trials.append({
                "min": float(np.min(trial)),
                "max": float(np.max(trial)),
                "mean": float(np.mean(trial)),
            })
        info["motion_trial_summary_before_discretization"] = normalized_trials[:2]
        session_info.append(info)

    data = {
        "neural": neural,
        "input": input_data,
        "output": output_data,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": ["barrel cortex L2/3"],
        "brain_region_idx": brain_region_idx,
        "input_names": ["time_from_session_start_s"],
        "output_names": ["motion_energy_bin"],
        "output_values": [[
            "0-20 percentile",
            "20-40 percentile",
            "40-60 percentile",
            "60-80 percentile",
            "80-100 percentile",
        ]],
        "metadata": {
            "task_description": "Decode discretized spontaneous motion-energy state from longitudinal barrel-cortex calcium activity.",
            "time_bin_size": float((frame_bin / 30.0) * 1000.0),
            "temporal_alignment_event": "start of each consecutive 2-minute block cut from a continuous session",
            "off_start": 0.0,
            "off_end": float(trial_duration_s),
            "source_reference": REFERENCE,
            "conversion_mode": mode,
            "input_description": "Absolute time from session start, sampled at the center of each 10-frame bin.",
            "neural_processing": "Suite2p neuropil subtraction and maximin baseline correction using per-session ops.npy parameters, then 10-frame averaging.",
            "output_processing": "Motion energy aligned to imaging frames, repaired only when shorter than imaging trace, 10-frame averaged, normalized per session, then discretized into global quintiles within this export.",
            "motion_quintile_edges": [float(x) for x in global_edges],
            "frame_bin_size": int(frame_bin),
            "trial_duration_s": float(trial_duration_s),
            "session_info": session_info,
        },
    }
    return data


def summarize_dataset(data: dict[str, Any]) -> dict[str, Any]:
    n_sessions = len(data["neural"])
    n_trials = sum(len(session_trials) for session_trials in data["neural"])
    neuron_counts = [session_trials[0].shape[0] for session_trials in data["neural"]]
    trial_lengths = [trial.shape[1] for session_trials in data["neural"] for trial in session_trials]

    output_counts = np.zeros(5, dtype=np.int64)
    for session_trials in data["output"]:
        for trial in session_trials:
            values, counts = np.unique(trial, return_counts=True)
            output_counts[values.astype(int)] += counts.astype(np.int64)

    session_durations = [
        info["raw_duration_s"]
        for info in data["metadata"]["session_info"]
    ]
    motion_repairs = [
        info["motion_alignment"]["inserted_missing_values"]
        for info in data["metadata"]["session_info"]
    ]

    summary = {
        "n_subjects": len(data["subjects"]),
        "n_sessions": n_sessions,
        "n_trials": n_trials,
        "sessions_per_subject": {
            subject: int(np.sum(data["subject_idx"] == idx))
            for idx, subject in enumerate(data["subjects"])
        },
        "neuron_count_mean": float(np.mean(neuron_counts)),
        "neuron_count_std": float(np.std(neuron_counts, ddof=0)),
        "neuron_count_by_session": neuron_counts,
        "trial_length_unique": sorted(set(int(x) for x in trial_lengths)),
        "session_duration_s_unique": sorted(set(float(x) for x in session_durations)),
        "motion_repairs_total": int(np.sum(motion_repairs)),
        "motion_repairs_by_session": motion_repairs,
        "output_class_fractions": (output_counts / output_counts.sum()).tolist(),
        "paper_tracked_neuron_mean": 526.0,
        "paper_tracked_neuron_std": 190.0,
    }
    return summary


def normalize_session_outputs_in_place(session_records: list[SessionRecord]) -> None:
    for record in session_records:
        concatenated = np.concatenate([trial.reshape(-1) for trial in record.output_continuous_trials], axis=0)
        normalized = minmax_normalize(concatenated)
        cursor = 0
        new_trials = []
        for trial in record.output_continuous_trials:
            n = trial.size
            new_trials.append(normalized[cursor:cursor + n].reshape(trial.shape).astype(np.float32, copy=False))
            cursor += n
        record.output_continuous_trials = new_trials


def convert_dataset(
    data_dir: Path,
    mode: str,
    device: torch.device,
    frame_bin: int = 10,
    trial_duration_s: float = 120.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    session_dirs = select_session_dirs(data_dir=data_dir, mode=mode)
    session_records = [
        load_session(
            session_dir=session_dir,
            device=device,
            frame_bin=frame_bin,
            trial_duration_s=trial_duration_s,
        )
        for session_dir in session_dirs
    ]
    normalize_session_outputs_in_place(session_records)
    data = build_dataset(
        session_records=session_records,
        mode=mode,
        frame_bin=frame_bin,
        trial_duration_s=trial_duration_s,
    )
    summary = summarize_dataset(data)
    return data, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Track2p developmental barrel-cortex data to decoder format.")
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--mode", choices=["full", "sample"], default="full")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    data, summary = convert_dataset(
        data_dir=args.data_dir,
        mode=args.mode,
        device=device,
    )

    with args.output.open("wb") as f:
        pickle.dump(data, f)

    print(f"Saved converted dataset to {args.output}")
    print(json.dumps(summary, indent=2))

    if args.summary_json is not None:
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Saved summary JSON to {args.summary_json}")


if __name__ == "__main__":
    main()
