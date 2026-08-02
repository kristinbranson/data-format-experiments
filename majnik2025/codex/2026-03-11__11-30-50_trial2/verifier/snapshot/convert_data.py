#!/usr/bin/env python3
"""Convert Track2p developmental barrel-cortex data into decoder format."""

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


DATA_ROOT = Path("data")
BIN_FRAMES = 10
TRIAL_SECONDS = 120.0
BRAIN_REGION = "barrel cortex"
N_OUTPUT_BINS = 5


@dataclass(frozen=True)
class SessionRef:
    subject: str
    session_id: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Track2p developmental imaging data into decoder format."
    )
    parser.add_argument("outpicklefile", type=Path, help="Output pickle file path.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only 2 sessions.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing visualizations for up to 2 sessions.",
    )
    args = parser.parse_args()
    if not args.full and not args.sample:
        args.full = True
    return args


def discover_sessions(data_root: Path) -> list[SessionRef]:
    sessions: list[SessionRef] = []
    for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
        for session_dir in sorted(
            p for p in subject_dir.iterdir() if p.is_dir() and len(p.name) >= 10 and p.name[:4].isdigit()
        ):
            sessions.append(
                SessionRef(subject=subject_dir.name, session_id=session_dir.name, path=session_dir)
            )
    return sessions


def average_nonoverlapping(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n_frames = x.shape[-1]
    usable = (n_frames // bin_frames) * bin_frames
    if usable == 0:
        raise ValueError(f"Not enough frames to bin: got {n_frames}, need at least {bin_frames}")
    x = x[..., :usable]
    new_shape = x.shape[:-1] + (usable // bin_frames, bin_frames)
    return x.reshape(new_shape).mean(axis=-1)


def compute_fluorescence_signal(F: np.ndarray, Fneu: np.ndarray, ops: dict) -> np.ndarray:
    """Approximate the paper's Suite2p-based dF/F signal."""
    Fc = F.astype(np.float32, copy=False) - float(ops["neucoeff"]) * Fneu.astype(np.float32, copy=False)
    processed = suite2p_preprocess(
        Fc.copy(),
        baseline=ops["baseline"],
        win_baseline=float(ops["win_baseline"]),
        sig_baseline=float(ops["sig_baseline"]),
        fs=float(ops["fs"]),
        prctile_baseline=float(ops["prctile_baseline"]),
        batch_size=min(512, max(32, Fc.shape[0])),
        device=torch.device("cpu"),
    )
    return processed.astype(np.float32, copy=False)


def reconstruct_motion_trace(
    motion_energy: np.ndarray, tstamps: np.ndarray, n_imaging_frames: int
) -> tuple[np.ndarray, dict]:
    """Map motion-energy samples onto the imaging frame grid and interpolate missing frames."""
    motion_energy = motion_energy.astype(np.float32, copy=False)
    tstamps = tstamps.astype(np.float64, copy=False)

    if motion_energy.ndim != 1 or tstamps.ndim != 1:
        raise ValueError("Motion-energy inputs must be 1D arrays.")
    if len(motion_energy) != len(tstamps):
        raise ValueError("motion_energy and tstamps must have the same length.")
    if n_imaging_frames < 2:
        raise ValueError("Expected at least 2 imaging frames.")

    frame_dt = (tstamps[-1] - tstamps[0]) / (n_imaging_frames - 1)
    if not np.isfinite(frame_dt) or frame_dt <= 0:
        raise ValueError(f"Invalid motion timestamp scale: frame_dt={frame_dt}")

    frame_idx = np.round((tstamps - tstamps[0]) / frame_dt).astype(np.int64)
    frame_idx = np.clip(frame_idx, 0, n_imaging_frames - 1)

    full = np.full(n_imaging_frames, np.nan, dtype=np.float32)
    counts = np.zeros(n_imaging_frames, dtype=np.int64)
    sums = np.zeros(n_imaging_frames, dtype=np.float64)
    np.add.at(sums, frame_idx, motion_energy.astype(np.float64, copy=False))
    np.add.at(counts, frame_idx, 1)
    valid = counts > 0
    full[valid] = (sums[valid] / counts[valid]).astype(np.float32)

    if not np.any(valid):
        raise ValueError("No valid motion frames after alignment.")

    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) == 1:
        full[:] = full[valid_idx[0]]
    else:
        missing = np.flatnonzero(~valid)
        if len(missing):
            full[missing] = np.interp(missing, valid_idx, full[valid_idx]).astype(np.float32)

    info = {
        "n_motion_frames_raw": int(len(motion_energy)),
        "n_imaging_frames": int(n_imaging_frames),
        "n_missing_motion_frames": int((~valid).sum()),
        "motion_frame_dt": float(frame_dt),
    }
    return full, info


def normalize_motion(x: np.ndarray) -> np.ndarray:
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax <= xmin:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - xmin) / (xmax - xmin)).astype(np.float32)


def quintile_one_hot(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    classes = np.searchsorted(edges, x, side="right").astype(np.int64)
    one_hot = np.eye(N_OUTPUT_BINS, dtype=np.int64)[classes].T
    return one_hot, classes, edges


def make_time_input(n_bins: int, fs: float, bin_frames: int) -> np.ndarray:
    return (np.arange(n_bins, dtype=np.float32) * (bin_frames / fs)).astype(np.float32)


def split_trials(
    neural_binned: np.ndarray,
    time_binned_s: np.ndarray,
    output_one_hot: np.ndarray,
    fs: float,
    bin_frames: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], dict]:
    bins_per_trial = int(round(TRIAL_SECONDS * fs / bin_frames))
    usable_bins = (neural_binned.shape[1] // bins_per_trial) * bins_per_trial
    if usable_bins < bins_per_trial:
        raise ValueError("Session is too short to form even one 2-minute trial.")

    neural_binned = neural_binned[:, :usable_bins]
    time_binned_s = time_binned_s[:usable_bins]
    output_one_hot = output_one_hot[:, :usable_bins]

    neural_trials: list[np.ndarray] = []
    input_trials: list[np.ndarray] = []
    output_trials: list[np.ndarray] = []
    for start in range(0, usable_bins, bins_per_trial):
        stop = start + bins_per_trial
        neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))
        input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
        output_trials.append(output_one_hot[:, start:stop].astype(np.int64, copy=False))

    info = {
        "bins_per_trial": bins_per_trial,
        "n_trials": len(neural_trials),
        "usable_bins": usable_bins,
        "usable_seconds": float(usable_bins * bin_frames / fs),
    }
    return neural_trials, input_trials, output_trials, info


def plot_processing_summary(
    session: SessionRef,
    plot_path: Path,
    F: np.ndarray,
    Fneu: np.ndarray,
    neural_processed: np.ndarray,
    neural_binned: np.ndarray,
    motion_raw: np.ndarray,
    motion_aligned: np.ndarray,
    motion_binned_norm: np.ndarray,
    motion_classes: np.ndarray,
    time_binned_s: np.ndarray,
    trial_neural: np.ndarray,
    trial_output: np.ndarray,
    motion_edges: np.ndarray,
    motion_info: dict,
) -> None:
    sample_neuron = 0
    preview_frames = min(3000, F.shape[1])
    preview_bins = min(600, len(time_binned_s))

    fig, ax = plt.subplots(4, 2, figsize=(18, 16))

    ax[0, 0].plot(F[sample_neuron, :preview_frames], label="F", lw=1)
    ax[0, 0].plot(Fneu[sample_neuron, :preview_frames], label="Fneu", lw=1, alpha=0.8)
    ax[0, 0].set_title("Raw Suite2p Fluorescence")
    ax[0, 0].set_xlabel("Frame")
    ax[0, 0].legend(loc="upper right")

    ax[0, 1].plot(neural_processed[sample_neuron, :preview_frames], lw=1, label="processed")
    ax[0, 1].plot(
        np.repeat(neural_binned[sample_neuron, :preview_bins], BIN_FRAMES)[:preview_frames],
        lw=1.2,
        label="10-frame mean",
    )
    ax[0, 1].set_title("Neural Processing")
    ax[0, 1].set_xlabel("Frame")
    ax[0, 1].legend(loc="upper right")

    ax[1, 0].plot(motion_raw[:preview_frames], lw=1, label="raw video metric")
    ax[1, 0].plot(motion_aligned[:preview_frames], lw=1, label="aligned/interpolated")
    ax[1, 0].set_title(
        f"Motion Alignment (missing frames filled: {motion_info['n_missing_motion_frames']})"
    )
    ax[1, 0].set_xlabel("Imaging frame")
    ax[1, 0].legend(loc="upper right")

    ax[1, 1].plot(time_binned_s[:preview_bins], motion_binned_norm[:preview_bins], lw=1)
    for edge in motion_edges:
        ax[1, 1].axhline(edge, color="tab:red", ls="--", lw=0.8)
    ax[1, 1].set_title("Normalized Motion After 10-frame Binning")
    ax[1, 1].set_xlabel("Time from session start (s)")

    vmax = np.percentile(np.abs(trial_neural), 99)
    vmax = 1.0 if not np.isfinite(vmax) or vmax <= 0 else vmax
    ax[2, 0].imshow(trial_neural[: min(80, trial_neural.shape[0])], aspect="auto", cmap="viridis")
    ax[2, 0].set_title("First Trial Neural Activity")
    ax[2, 0].set_xlabel("Binned time")
    ax[2, 0].set_ylabel("Neuron")

    ax[2, 1].imshow(trial_output, aspect="auto", cmap="Greys")
    ax[2, 1].set_title("First Trial One-hot Motion Quintiles")
    ax[2, 1].set_xlabel("Binned time")
    ax[2, 1].set_ylabel("Quintile")

    counts = np.bincount(motion_classes, minlength=N_OUTPUT_BINS)
    ax[3, 0].bar(np.arange(N_OUTPUT_BINS), counts)
    ax[3, 0].set_title("Session Motion Quintile Counts")
    ax[3, 0].set_xlabel("Quintile")
    ax[3, 0].set_ylabel("Count")

    ax[3, 1].hist(motion_binned_norm, bins=60, color="tab:green", alpha=0.8)
    for edge in motion_edges:
        ax[3, 1].axvline(edge, color="tab:red", ls="--", lw=0.8)
    ax[3, 1].set_title("Normalized Motion Histogram")
    ax[3, 1].set_xlabel("Normalized motion")

    fig.suptitle(f"Processing Summary: {session.subject}/{session.session_id}")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def process_session(session: SessionRef, show_processing: bool = False) -> tuple[dict, dict]:
    start_time = time.time()
    suite2p_dir = session.path / "suite2p" / "plane0"
    move_dir = session.path / "move_deve"

    F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
    Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
    ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
    motion_raw = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
    tstamps = np.load(move_dir / "tstamps.npy", allow_pickle=True)

    fs = float(ops["fs"])
    neural_processed = compute_fluorescence_signal(F, Fneu, ops)
    neural_binned = average_nonoverlapping(neural_processed, BIN_FRAMES).astype(np.float32, copy=False)

    motion_aligned, motion_info = reconstruct_motion_trace(motion_raw, tstamps, F.shape[1])
    motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], BIN_FRAMES)[0].astype(
        np.float32, copy=False
    )
    motion_binned_norm = normalize_motion(motion_binned)
    output_one_hot, motion_classes, motion_edges = quintile_one_hot(motion_binned_norm)

    time_binned_s = make_time_input(neural_binned.shape[1], fs=fs, bin_frames=BIN_FRAMES)
    neural_trials, input_trials, output_trials, trial_info = split_trials(
        neural_binned=neural_binned,
        time_binned_s=time_binned_s,
        output_one_hot=output_one_hot,
        fs=fs,
        bin_frames=BIN_FRAMES,
    )

    if show_processing:
        plot_processing_summary(
            session=session,
            plot_path=Path(f"processing_{session.subject}_{session.session_id}.png"),
            F=F,
            Fneu=Fneu,
            neural_processed=neural_processed,
            neural_binned=neural_binned,
            motion_raw=motion_raw.astype(np.float32, copy=False),
            motion_aligned=motion_aligned,
            motion_binned_norm=motion_binned_norm,
            motion_classes=motion_classes,
            time_binned_s=time_binned_s,
            trial_neural=neural_trials[0],
            trial_output=output_trials[0],
            motion_edges=motion_edges,
            motion_info=motion_info,
        )

    elapsed = time.time() - start_time
    summary = {
        "subject": session.subject,
        "session_id": session.session_id,
        "n_neurons": int(F.shape[0]),
        "n_frames_raw": int(F.shape[1]),
        "n_bins": int(neural_binned.shape[1]),
        "n_trials": int(trial_info["n_trials"]),
        "fs": fs,
        "duration_s_raw": float(F.shape[1] / fs),
        "duration_s_used": float(trial_info["usable_seconds"]),
        "missing_motion_frames": int(motion_info["n_missing_motion_frames"]),
        "motion_quintile_edges": motion_edges.tolist(),
        "processing_seconds": float(elapsed),
    }
    converted = {
        "neural": neural_trials,
        "input": input_trials,
        "output": output_trials,
        "brain_region_idx": np.zeros(F.shape[0], dtype=np.int64),
        "summary": summary,
    }
    return converted, summary


def validate_converted_session(converted: dict) -> None:
    neural = converted["neural"]
    input_ = converted["input"]
    output = converted["output"]
    n_trials = len(neural)
    if n_trials < 2:
        raise ValueError("Each converted session must contain at least 2 trials.")
    if not (len(input_) == len(output) == n_trials):
        raise ValueError("Neural/input/output trial counts do not match.")
    n_neurons = neural[0].shape[0]
    for trial_idx in range(n_trials):
        if neural[trial_idx].shape[0] != n_neurons:
            raise ValueError("Neuron count changed across trials within a session.")
        if neural[trial_idx].shape[1] != input_[trial_idx].shape[1]:
            raise ValueError("Input time dimension does not match neural time dimension.")
        if neural[trial_idx].shape[1] != output[trial_idx].shape[1]:
            raise ValueError("Output time dimension does not match neural time dimension.")
        if np.isnan(neural[trial_idx]).any() or np.isnan(input_[trial_idx]).any() or np.isnan(output[trial_idx]).any():
            raise ValueError("Converted arrays must not contain NaN values.")
        if output[trial_idx].shape[0] != N_OUTPUT_BINS:
            raise ValueError("Expected one-hot output with five rows.")
        if not np.all(output[trial_idx].sum(axis=0) == 1):
            raise ValueError("Each output timepoint must belong to exactly one motion quintile.")


def build_dataset(converted_sessions: list[dict], session_refs: list[SessionRef]) -> dict:
    subjects = sorted({session.subject for session in session_refs})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}

    dataset = {
        "neural": [session["neural"] for session in converted_sessions],
        "input": [session["input"] for session in converted_sessions],
        "output": [session["output"] for session in converted_sessions],
        "subjects": subjects,
        "subject_idx": np.array([subject_to_idx[session.subject] for session in session_refs], dtype=np.int64),
        "brain_regions": [BRAIN_REGION],
        "brain_region_idx": [session["brain_region_idx"] for session in converted_sessions],
        "input_names": ["time_from_session_start_s"],
        "output_names": [f"motion_energy_q{i}" for i in range(N_OUTPUT_BINS)],
        "output_values": [[f"not_q{i}", f"q{i}"] for i in range(N_OUTPUT_BINS)],
        "metadata": {
            "task_description": (
                "Decode spontaneous motion state from longitudinal barrel-cortex calcium activity. "
                "Motion energy is reconstructed from synchronized videography, averaged in 10-frame bins, "
                "normalized within session, and represented as one-hot session-wise quintiles."
            ),
            "time_bin_size": float(1000.0 * BIN_FRAMES / 30.0),
            "temporal_alignment_event": "start of each consecutive 2-minute recording block",
            "off_start": 0.0,
            "off_end": float(TRIAL_SECONDS),
            "raw_frame_rate_hz": 30.0,
            "bin_size_frames": BIN_FRAMES,
            "trial_block_seconds": TRIAL_SECONDS,
            "neural_signal": (
                "Suite2p-style neuropil-subtracted and baseline-corrected fluorescence derived from F/Fneu/ops"
            ),
            "behavior_signal": "global motion energy from videography aligned to imaging frames",
            "output_representation": "five binary one-hot channels corresponding to motion-energy quintiles",
            "session_ids": [f"{session.subject}/{session.session_id}" for session in session_refs],
            "session_info": [session["summary"] for session in converted_sessions],
        },
    }
    return dataset


def main() -> None:
    args = parse_args()
    total_start = time.time()

    sessions = discover_sessions(DATA_ROOT)
    if args.sample:
        sessions = sessions[-2:]
    if not sessions:
        raise SystemExit("No sessions found under data/.")

    print(f"Found {len(sessions)} sessions to convert.")
    converted_sessions: list[dict] = []
    session_summaries: list[dict] = []
    plot_budget = 2 if args.show_processing else 0

    for idx, session in enumerate(sessions, start=1):
        do_plot = plot_budget > 0
        print(f"[{idx}/{len(sessions)}] Processing {session.subject}/{session.session_id} ...")
        converted, summary = process_session(session, show_processing=do_plot)
        validate_converted_session(converted)
        converted_sessions.append(converted)
        session_summaries.append(summary)
        plot_budget -= int(do_plot)
        print(
            f"  neurons={summary['n_neurons']} raw_frames={summary['n_frames_raw']} "
            f"trials={summary['n_trials']} missing_motion_frames={summary['missing_motion_frames']} "
            f"time={summary['processing_seconds']:.2f}s"
        )

    dataset = build_dataset(converted_sessions, sessions)
    with args.outpicklefile.open("wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - total_start
    total_trials = sum(len(session["neural"]) for session in converted_sessions)
    print(f"Saved converted dataset to {args.outpicklefile}")
    print(f"Sessions: {len(sessions)}")
    print(f"Trials: {total_trials}")
    print(f"Total conversion time: {elapsed:.2f}s")
    if session_summaries:
        mean_time = np.mean([summary["processing_seconds"] for summary in session_summaries])
        est_full = mean_time * 41
        print(f"Mean processing time per session: {mean_time:.2f}s")
        print(f"Estimated full-dataset time at this rate: {est_full / 60:.2f} minutes")


if __name__ == "__main__":
    main()
