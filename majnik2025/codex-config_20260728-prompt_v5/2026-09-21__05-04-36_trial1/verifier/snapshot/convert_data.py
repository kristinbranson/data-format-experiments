#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from suite2p.extraction.dcnv import preprocess as suite2p_preprocess

from decoder import verify_data_format


DATA_ROOT = Path("/app/data")
BIN_FRAMES = 10
TRIAL_SECONDS = 60
FRAME_RATE_HZ = 30.0
TRIAL_FRAMES = int(TRIAL_SECONDS * FRAME_RATE_HZ)
BINS_PER_TRIAL = TRIAL_FRAMES // BIN_FRAMES
NEUROPIL_COEFF = 0.7
BASELINE_MODE = "maximin"
WIN_BASELINE_SECONDS = 60.0
SIG_BASELINE_FRAMES = 10.0
PRCTILE_BASELINE = 8.0
SUITE2P_BATCH_SIZE = 128
OUTPUT_LABELS = ["q1_lowest", "q2_low", "q3_mid", "q4_high", "q5_highest"]
BRAIN_REGION = "barrel cortex L2/3"


@dataclass(frozen=True)
class SessionInfo:
    subject: str
    session: str
    session_dir: Path
    suite2p_dir: Path
    move_dir: Path
    nframes: int
    nneurons: int
    motion_len: int

    @property
    def session_id(self) -> str:
        return f"{self.subject}_{self.session}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Track2p longitudinal calcium imaging data into decoder format."
    )
    parser.add_argument("outpicklefile", type=Path, help="Output pickle file path.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only 2 representative sessions.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing visualizations for up to 2 sessions as processing_<session_id>.png.",
    )
    return parser.parse_args()


def discover_sessions(data_root: Path) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
            suite2p_dir = session_dir / "suite2p" / "plane0"
            move_dir = session_dir / "move_deve"
            ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
            f = np.load(suite2p_dir / "F.npy", mmap_mode="r")
            motion = np.load(move_dir / "motion_energy_glob.npy", mmap_mode="r")
            sessions.append(
                SessionInfo(
                    subject=subject_dir.name,
                    session=session_dir.name,
                    session_dir=session_dir,
                    suite2p_dir=suite2p_dir,
                    move_dir=move_dir,
                    nframes=int(ops["nframes"]),
                    nneurons=int(f.shape[0]),
                    motion_len=int(motion.shape[0]),
                )
            )
    return sessions


def choose_sample_sessions(sessions: list[SessionInfo]) -> list[SessionInfo]:
    if len(sessions) <= 2:
        return sessions

    selected: list[SessionInfo] = []

    missing_sessions = [session for session in sessions if session.motion_len < session.nframes]
    if missing_sessions:
        selected.append(max(missing_sessions, key=lambda s: (s.nframes - s.motion_len, s.nframes, s.nneurons)))
    else:
        selected.append(sessions[0])

    remaining = [session for session in sessions if session.session_id != selected[0].session_id]
    longest = max(remaining, key=lambda s: (s.nframes, s.nneurons))
    selected.append(longest)

    return sorted(selected, key=lambda s: (s.subject, s.session))


def detect_timestamp_scale_to_seconds(tstamps: np.ndarray, fs: float) -> float:
    if tstamps.size < 2:
        return 1.0
    dt = float(np.median(np.diff(tstamps)))
    target_dt = 1.0 / fs
    candidates = [1e-3, 1.0, 1e3, 86400.0]
    best_scale = 1.0
    best_err = float("inf")
    for scale in candidates:
        err = abs(dt * scale - target_dt)
        if err < best_err:
            best_err = err
            best_scale = scale
    return best_scale


def non_overlapping_mean_last_axis(arr: np.ndarray, factor: int) -> np.ndarray:
    usable = arr.shape[-1] - (arr.shape[-1] % factor)
    if usable <= 0:
        raise ValueError(f"Cannot bin array with shape {arr.shape} by factor {factor}.")
    trimmed = arr[..., :usable]
    new_shape = trimmed.shape[:-1] + (usable // factor, factor)
    return trimmed.reshape(new_shape).mean(axis=-1)


def align_motion_to_imaging(
    motion: np.ndarray,
    tstamps: np.ndarray,
    nframes: int,
    fs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    tstamps = np.asarray(tstamps, dtype=np.float64)
    motion = np.asarray(motion, dtype=np.float64)
    scale_to_seconds = detect_timestamp_scale_to_seconds(tstamps, fs)

    if motion.size == nframes:
        return (
            motion.astype(np.float32, copy=False),
            np.zeros(nframes, dtype=bool),
            np.arange(nframes, dtype=np.int64),
            scale_to_seconds,
        )

    span = float(tstamps[-1] - tstamps[0])
    if span <= 0:
        raise ValueError("Motion timestamps are not strictly increasing.")
    nominal_step = span / float(nframes - 1)
    frame_idx = np.rint((tstamps - tstamps[0]) / nominal_step).astype(np.int64)
    frame_idx = np.clip(frame_idx, 0, nframes - 1)

    sums = np.zeros(nframes, dtype=np.float64)
    counts = np.zeros(nframes, dtype=np.int64)
    np.add.at(sums, frame_idx, motion)
    np.add.at(counts, frame_idx, 1)

    aligned = np.full(nframes, np.nan, dtype=np.float64)
    valid = counts > 0
    aligned[valid] = sums[valid] / counts[valid]
    missing = ~valid

    if missing.all():
        raise ValueError("All motion samples are missing after timestamp alignment.")

    if missing.any():
        valid_idx = np.flatnonzero(valid)
        missing_idx = np.flatnonzero(missing)
        aligned[missing] = np.interp(missing_idx, valid_idx, aligned[valid])

    return aligned.astype(np.float32), missing, frame_idx, scale_to_seconds


def compute_motion_quintiles(values: np.ndarray, nclasses: int = 5) -> tuple[np.ndarray, np.ndarray]:
    quantiles = np.linspace(0.0, 1.0, nclasses + 1)[1:-1]
    thresholds = np.quantile(values, quantiles)
    categories = np.digitize(values, thresholds, right=False).astype(np.int64)

    if np.unique(categories).size < nclasses:
        order = np.argsort(values, kind="mergesort")
        categories = np.empty(values.shape[0], dtype=np.int64)
        boundaries = np.linspace(0, values.shape[0], nclasses + 1, dtype=int)
        for cls in range(nclasses):
            categories[order[boundaries[cls]:boundaries[cls + 1]]] = cls
    return categories, thresholds.astype(np.float32)


def split_time_series_into_trials(arr: np.ndarray, bins_per_trial: int) -> list[np.ndarray]:
    usable = arr.shape[-1] - (arr.shape[-1] % bins_per_trial)
    if usable <= 0:
        raise ValueError(f"Array with shape {arr.shape} does not contain a full trial.")
    arr = arr[..., :usable]
    ntrials = usable // bins_per_trial
    return [arr[:, i * bins_per_trial:(i + 1) * bins_per_trial] for i in range(ntrials)]


def zscore_rows(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (x - mean) / std


def plot_processing_summary(
    session: SessionInfo,
    corrected_dff: np.ndarray,
    binned_neural: np.ndarray,
    motion_raw: np.ndarray,
    motion_aligned: np.ndarray,
    missing_mask: np.ndarray,
    raw_frame_idx: np.ndarray,
    binned_motion: np.ndarray,
    motion_bins: np.ndarray,
    time_input: np.ndarray,
    thresholds: np.ndarray,
) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(18, 22), constrained_layout=True)
    fig.suptitle(f"Processing summary: {session.session_id}", fontsize=16)

    frame_times = np.arange(corrected_dff.shape[1]) / FRAME_RATE_HZ
    plot_frames = min(3000, corrected_dff.shape[1])
    neurons_to_show = min(3, corrected_dff.shape[0])
    offset = 0.0
    for neuron in range(neurons_to_show):
        trace = corrected_dff[neuron, :plot_frames]
        axes[0].plot(frame_times[:plot_frames], trace + offset, lw=0.9, label=f"neuron {neuron}")
        offset += np.nanstd(trace) * 6 + 1.0
    axes[0].set_title("Baseline-corrected neural traces (first 100 s)")
    axes[0].set_ylabel("Corrected fluorescence + offset")
    axes[0].legend(loc="upper right")

    observed_frame_times = raw_frame_idx / FRAME_RATE_HZ
    axes[1].plot(np.arange(motion_aligned.size) / FRAME_RATE_HZ, motion_aligned, lw=0.75, color="black", label="aligned/interpolated")
    axes[1].scatter(observed_frame_times, motion_raw, s=4, alpha=0.35, color="tab:blue", label="observed camera samples")
    if missing_mask.any():
        y0 = np.nanmin(motion_aligned)
        axes[1].scatter(
            np.flatnonzero(missing_mask) / FRAME_RATE_HZ,
            np.full(int(missing_mask.sum()), y0),
            s=6,
            color="tab:red",
            label="missing camera frame",
        )
    axes[1].set_title("Motion aligned to imaging frames")
    axes[1].set_ylabel("Motion energy")
    axes[1].legend(loc="upper right")

    axes[2].plot(time_input[0], binned_motion, lw=0.8, color="tab:green", label="10-frame mean motion")
    for thr in thresholds:
        axes[2].axhline(float(thr), ls="--", lw=0.8, color="gray")
    for trial_start in range(0, binned_motion.size + 1, BINS_PER_TRIAL):
        if trial_start < binned_motion.size:
            axes[2].axvline(float(time_input[0, trial_start]), color="tab:orange", alpha=0.2)
    axes[2].set_title("Binned motion with session-specific quintile thresholds")
    axes[2].set_ylabel("Motion energy")
    axes[2].legend(loc="upper right")

    nneurons_show = min(60, binned_neural.shape[0])
    im = axes[3].imshow(
        zscore_rows(binned_neural[:nneurons_show]),
        aspect="auto",
        cmap="gray_r",
        vmin=-1.0,
        vmax=3.0,
    )
    for trial_start in range(0, binned_neural.shape[1] + 1, BINS_PER_TRIAL):
        axes[3].axvline(trial_start - 0.5, color="tab:orange", alpha=0.25)
    axes[3].set_title("Binned neural activity (z-scored, sample of neurons)")
    axes[3].set_ylabel("Neuron")
    fig.colorbar(im, ax=axes[3], fraction=0.02, pad=0.01)

    preview_bins = min(2 * BINS_PER_TRIAL, motion_bins.size)
    axes[4].step(time_input[0, :preview_bins], motion_bins[:preview_bins], where="mid", color="tab:purple", label="motion quintile")
    axes[4].plot(time_input[0, :preview_bins], time_input[0, :preview_bins], color="tab:brown", alpha=0.7, label="time input (s)")
    axes[4].set_title("Final decoder inputs/outputs for the first two 60 s trials")
    axes[4].set_ylabel("Value")
    axes[4].set_xlabel("Elapsed session time (s)")
    axes[4].legend(loc="upper left")

    fig.savefig(f"/app/processing_{session.session_id}.png", dpi=150)
    plt.close(fig)


def process_session(session: SessionInfo, show_processing: bool = False) -> tuple[dict, dict]:
    stage_times: dict[str, float] = {}

    t0 = time.perf_counter()
    f = np.load(session.suite2p_dir / "F.npy").astype(np.float32)
    fneu = np.load(session.suite2p_dir / "Fneu.npy").astype(np.float32)
    motion_raw = np.load(session.move_dir / "motion_energy_glob.npy")
    tstamps = np.load(session.move_dir / "tstamps.npy")
    stage_times["load_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    corrected = f - NEUROPIL_COEFF * fneu
    corrected = suite2p_preprocess(
        corrected.copy(),
        baseline=BASELINE_MODE,
        win_baseline=WIN_BASELINE_SECONDS,
        sig_baseline=SIG_BASELINE_FRAMES,
        fs=FRAME_RATE_HZ,
        prctile_baseline=PRCTILE_BASELINE,
        batch_size=SUITE2P_BATCH_SIZE,
        device=torch.device("cpu"),
    ).astype(np.float32)
    stage_times["neural_preprocess_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    motion_aligned, missing_mask, raw_frame_idx, timestamp_scale = align_motion_to_imaging(
        motion=motion_raw,
        tstamps=tstamps,
        nframes=session.nframes,
        fs=FRAME_RATE_HZ,
    )
    stage_times["motion_align_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    binned_neural = non_overlapping_mean_last_axis(corrected, BIN_FRAMES).astype(np.float32)
    binned_motion = non_overlapping_mean_last_axis(motion_aligned, BIN_FRAMES).astype(np.float32)
    frame_times = np.arange(session.nframes, dtype=np.float32) / np.float32(FRAME_RATE_HZ)
    binned_time = non_overlapping_mean_last_axis(frame_times, BIN_FRAMES).astype(np.float32)
    motion_bins, thresholds = compute_motion_quintiles(binned_motion.astype(np.float64))
    stage_times["binning_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    neural_trials = split_time_series_into_trials(binned_neural, BINS_PER_TRIAL)
    input_trials = split_time_series_into_trials(binned_time[np.newaxis, :], BINS_PER_TRIAL)
    output_trials = split_time_series_into_trials(motion_bins[np.newaxis, :], BINS_PER_TRIAL)
    neural_trials = [trial.astype(np.float32, copy=False) for trial in neural_trials]
    input_trials = [trial.astype(np.float32, copy=False) for trial in input_trials]
    output_trials = [trial.astype(np.int64, copy=False) for trial in output_trials]
    stage_times["trial_split_s"] = time.perf_counter() - t0

    if show_processing:
        t0 = time.perf_counter()
        plot_processing_summary(
            session=session,
            corrected_dff=corrected,
            binned_neural=binned_neural,
            motion_raw=np.asarray(motion_raw, dtype=np.float32),
            motion_aligned=motion_aligned,
            missing_mask=missing_mask,
            raw_frame_idx=raw_frame_idx,
            binned_motion=binned_motion,
            motion_bins=motion_bins,
            time_input=binned_time[np.newaxis, :],
            thresholds=thresholds,
        )
        stage_times["plot_s"] = time.perf_counter() - t0

    session_data = {
        "neural_trials": neural_trials,
        "input_trials": input_trials,
        "output_trials": output_trials,
        "brain_region_idx": np.zeros(session.nneurons, dtype=np.int64),
    }
    session_meta = {
        "subject": session.subject,
        "session": session.session,
        "session_id": session.session_id,
        "nframes_raw": session.nframes,
        "nneurons": session.nneurons,
        "ntrials": len(neural_trials),
        "timestamp_scale_to_seconds": float(timestamp_scale),
        "motion_missing_frames": int(missing_mask.sum()),
        "motion_missing_fraction": float(missing_mask.mean()),
        "motion_quantile_thresholds": thresholds.tolist(),
        "timing": {k: float(v) for k, v in stage_times.items()},
    }
    return session_data, session_meta


def build_dataset(sessions: list[SessionInfo], show_processing: bool) -> dict:
    dataset = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": [],
        "subject_idx": None,
        "brain_regions": [BRAIN_REGION],
        "brain_region_idx": [],
        "input_names": ["time_from_session_start_sec"],
        "output_names": ["motion_energy_quintile"],
        "output_values": [OUTPUT_LABELS],
        "metadata": {},
    }

    subject_to_idx: dict[str, int] = {}
    subject_idx: list[int] = []
    session_metadata: list[dict] = []

    for idx, session in enumerate(sessions):
        session_start = time.perf_counter()
        session_data, session_meta = process_session(
            session=session,
            show_processing=show_processing and idx < 2,
        )
        elapsed = time.perf_counter() - session_start
        session_meta["timing"]["session_total_s"] = float(elapsed)

        if session.subject not in subject_to_idx:
            subject_to_idx[session.subject] = len(dataset["subjects"])
            dataset["subjects"].append(session.subject)
        subject_idx.append(subject_to_idx[session.subject])

        dataset["neural"].append(session_data["neural_trials"])
        dataset["input"].append(session_data["input_trials"])
        dataset["output"].append(session_data["output_trials"])
        dataset["brain_region_idx"].append(session_data["brain_region_idx"])
        session_metadata.append(session_meta)

        print(
            f"[{idx + 1}/{len(sessions)}] {session.session_id}: "
            f"{session_meta['nneurons']} neurons, {session_meta['ntrials']} trials, "
            f"missing motion frames={session_meta['motion_missing_frames']}, "
            f"time={elapsed:.2f}s",
            flush=True,
        )
        for key, value in session_meta["timing"].items():
            if key != "session_total_s":
                print(f"    {key}: {value:.3f}s", flush=True)

    dataset["subject_idx"] = np.asarray(subject_idx, dtype=np.int64)
    dataset["metadata"] = {
        "task_description": (
            "Decode session time-varying motion energy quintile from longitudinal "
            "barrel cortex calcium activity in spontaneous-behavior recordings."
        ),
        "time_bin_size": float(1000.0 * BIN_FRAMES / FRAME_RATE_HZ),
        "temporal_alignment_event": "trial start of fixed 60-second windows tiled across each session",
        "off_start": 0.0,
        "off_end": float(TRIAL_SECONDS),
        "source_dataset": "Majnik et al. 2025 Track2p developmental barrel cortex dataset",
        "native_imaging_rate_hz": float(FRAME_RATE_HZ),
        "native_behavior_rate_hz": float(FRAME_RATE_HZ),
        "binning_frames": BIN_FRAMES,
        "binning_description": "Non-overlapping means of 10 consecutive 30 Hz samples",
        "neural_signal_description": (
            "Suite2p neuropil-subtracted fluorescence with default baseline preprocessing "
            "(paper-described dF/F proxy)"
        ),
        "motion_alignment_description": (
            "Motion energy aligned to imaging-frame timeline using timestamps and "
            "linearly interpolated only at dropped camera frames"
        ),
        "output_description": "Per-session motion energy discretized into five equal-percentile bins",
        "session_ids": [meta["session_id"] for meta in session_metadata],
        "session_info": session_metadata,
    }
    return dataset


def main() -> None:
    args = parse_args()
    mode_full = args.full or not args.sample

    overall_start = time.perf_counter()
    sessions = discover_sessions(DATA_ROOT)
    sessions_to_process = sessions if mode_full else choose_sample_sessions(sessions)

    print(f"Discovered {len(sessions)} sessions total.", flush=True)
    print(
        f"Processing {len(sessions_to_process)} sessions in "
        f"{'full' if mode_full else 'sample'} mode.",
        flush=True,
    )

    dataset = build_dataset(sessions_to_process, show_processing=args.show_processing)
    valid, errors, warnings = verify_data_format(dataset)
    if not valid:
        raise RuntimeError("Converted dataset failed format verification:\n" + "\n".join(errors))
    if warnings:
        print("Format warnings:", flush=True)
        for warning in warnings:
            print(f"  - {warning}", flush=True)

    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    with args.outpicklefile.open("wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_trials = sum(len(session_trials) for session_trials in dataset["neural"])
    total_neuron_entries = sum(
        trial.shape[0] for session_trials in dataset["neural"] for trial in session_trials[:1]
    )
    total_timepoints = sum(
        trial.shape[1] for session_trials in dataset["neural"] for trial in session_trials
    )
    elapsed = time.perf_counter() - overall_start

    print(f"Saved converted dataset to {args.outpicklefile}", flush=True)
    print(f"Sessions: {len(dataset['neural'])}", flush=True)
    print(f"Subjects: {len(dataset['subjects'])}", flush=True)
    print(f"Trials: {total_trials}", flush=True)
    print(f"Session-neuron counts (sum over sessions): {total_neuron_entries}", flush=True)
    print(f"Total trial time bins: {total_timepoints}", flush=True)
    print(f"Elapsed time: {elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()
