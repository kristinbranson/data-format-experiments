#!/usr/bin/env python3

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
from suite2p.extraction import dcnv


RAW_FS_HZ = 30.0
BIN_FRAMES = 10
TRIAL_DURATION_SEC = 120.0
TIME_BIN_SIZE_MS = 1000.0 * BIN_FRAMES / RAW_FS_HZ
BRAIN_REGION = "barrel cortex L2/3"
OUTPUT_CLASS_NAMES = ["Q1", "Q2", "Q3", "Q4", "Q5"]


@dataclass(frozen=True)
class SessionInfo:
    subject: str
    session: str
    path: Path

    @property
    def session_id(self) -> str:
        return f"{self.subject}_{self.session}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Track2p dataset to decoder format.")
    parser.add_argument("outpicklefile", type=Path, help="Output pickle file path.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only 2 representative sessions.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing plots for up to 2 sessions as processing_<session_id>.png.",
    )
    return parser.parse_args()


def discover_sessions(data_root: Path) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
            sessions.append(SessionInfo(subject=subject_dir.name, session=session_dir.name, path=session_dir))
    if not sessions:
        raise RuntimeError("No sessions found under data/.")
    return sessions


def select_sample_sessions(all_sessions: list[SessionInfo]) -> list[SessionInfo]:
    def session_nframes(session: SessionInfo) -> int:
        return int(np.load(session.path / "suite2p" / "plane0" / "ops.npy", allow_pickle=True).item()["nframes"])

    def has_missing_behavior_frames(session: SessionInfo) -> bool:
        move_dir = session.path / "move_deve"
        motion_len = int(np.load(move_dir / "motion_energy_glob.npy", mmap_mode="r").shape[0])
        nframes = session_nframes(session)
        return motion_len != nframes

    first = next((session for session in all_sessions if has_missing_behavior_frames(session)), all_sessions[0])
    first_nframes = session_nframes(first)

    second = None
    for session in all_sessions:
        if session == first:
            continue
        if session_nframes(session) != first_nframes and has_missing_behavior_frames(session):
            second = session
            break
    if second is None:
        for session in all_sessions:
            if session != first and session_nframes(session) != first_nframes:
                second = session
                break
    if second is None:
        second = all_sessions[min(1, len(all_sessions) - 1)]
    return [first, second]


def interpolate_nans(x: np.ndarray) -> np.ndarray:
    if not np.isnan(x).any():
        return x
    idx = np.arange(x.shape[0])
    valid = ~np.isnan(x)
    if not valid.any():
        raise ValueError("Cannot interpolate an array containing only NaNs.")
    out = x.copy()
    out[~valid] = np.interp(idx[~valid], idx[valid], x[valid]).astype(np.float32)
    return out


def reconstruct_motion_trace(motion: np.ndarray, interframe_int: np.ndarray, target_len: int) -> tuple[np.ndarray, np.ndarray]:
    motion = np.asarray(motion, dtype=np.float32)
    interframe_int = np.asarray(interframe_int, dtype=np.float64)
    if motion.shape[0] == target_len:
        return motion, np.empty(0, dtype=np.int64)

    median_interval = float(np.median(interframe_int))
    steps = np.rint(interframe_int / median_interval).astype(np.int64)
    steps[steps < 1] = 1

    observed_idx = np.empty(motion.shape[0], dtype=np.int64)
    observed_idx[0] = 0
    observed_idx[1:] = np.cumsum(steps)

    if observed_idx[-1] != target_len - 1:
        raise ValueError(
            f"Timing reconstruction failed: last observed index {observed_idx[-1]} does not match target {target_len - 1}"
        )

    full = np.full(target_len, np.nan, dtype=np.float32)
    full[observed_idx] = motion
    missing_idx = np.flatnonzero(np.isnan(full))
    full = interpolate_nans(full)
    return full, missing_idx


def compute_baseline_corrected_fluorescence(s2p_dir: Path) -> tuple[np.ndarray, dict, dict]:
    ops = np.load(s2p_dir / "ops.npy", allow_pickle=True).item()
    F = np.load(s2p_dir / "F.npy", mmap_mode="r")
    Fneu = np.load(s2p_dir / "Fneu.npy", mmap_mode="r")

    corrected = np.array(F, dtype=np.float32, copy=True)
    corrected -= np.float32(ops["neucoeff"]) * np.asarray(Fneu, dtype=np.float32)
    processed = dcnv.preprocess(
        corrected,
        baseline=ops["baseline"],
        win_baseline=float(ops["win_baseline"]),
        sig_baseline=float(ops["sig_baseline"]),
        fs=float(ops["fs"]),
        prctile_baseline=float(ops["prctile_baseline"]),
        batch_size=int(ops.get("batch_size", 2000)),
        device=torch.device("cpu"),
    ).astype(np.float32, copy=False)

    sample_neuron = min(2, processed.shape[0] - 1)
    preview_len = min(3000, processed.shape[1])
    preview = {
        "sample_neuron": sample_neuron,
        "F": np.asarray(F[sample_neuron, :preview_len], dtype=np.float32),
        "Fneu": np.asarray(Fneu[sample_neuron, :preview_len], dtype=np.float32),
        "corrected": corrected[sample_neuron, :preview_len].copy(),
        "processed": processed[sample_neuron, :preview_len].copy(),
    }
    return processed, ops, preview


def bin_array_mean(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n_bins = x.shape[-1] // bin_frames
    trimmed = x[..., : n_bins * bin_frames]
    new_shape = trimmed.shape[:-1] + (n_bins, bin_frames)
    return trimmed.reshape(new_shape).mean(axis=-1, dtype=np.float32).astype(np.float32, copy=False)


def segment_trials(
    neural_binned: np.ndarray,
    time_binned: np.ndarray,
    motion_norm_binned: np.ndarray,
    motion_edges: np.ndarray,
    trial_bins: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    n_total_bins = neural_binned.shape[1]
    n_trials = n_total_bins // trial_bins
    neural_trials: list[np.ndarray] = []
    input_trials: list[np.ndarray] = []
    output_trials: list[np.ndarray] = []

    for trial_idx in range(n_trials):
        start = trial_idx * trial_bins
        stop = start + trial_bins
        neural_trial = neural_binned[:, start:stop].astype(np.float32, copy=False)
        input_trial = time_binned[np.newaxis, start:stop].astype(np.float32, copy=False)
        output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False).astype(np.int64)
        output_trial = output_trial[np.newaxis, :]
        neural_trials.append(neural_trial)
        input_trials.append(input_trial)
        output_trials.append(output_trial)

    return neural_trials, input_trials, output_trials


def process_session(session: SessionInfo) -> dict:
    session_start = time.perf_counter()
    s2p_dir = session.path / "suite2p" / "plane0"
    move_dir = session.path / "move_deve"

    neural_processed, ops, neural_preview = compute_baseline_corrected_fluorescence(s2p_dir)
    n_frames = int(ops["nframes"])

    motion_raw = np.load(move_dir / "motion_energy_glob.npy")
    timestamps = np.load(move_dir / "tstamps.npy")
    interframe_int = np.load(move_dir / "interframe_int.npy")
    motion_full, missing_idx = reconstruct_motion_trace(motion_raw, interframe_int, n_frames)

    time_full_sec = (np.arange(n_frames, dtype=np.float32) + 0.5) / np.float32(ops["fs"])
    neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
    motion_binned = bin_array_mean(motion_full[np.newaxis, :], BIN_FRAMES)[0]
    time_binned = bin_array_mean(time_full_sec[np.newaxis, :], BIN_FRAMES)[0]

    preview_len = min(3000, n_frames)
    motion_preview = {
        "raw": np.asarray(motion_raw[:preview_len], dtype=np.float32),
        "reconstructed": motion_full[:preview_len].copy(),
        "missing_idx": missing_idx[missing_idx < preview_len].copy(),
        "timestamps": np.asarray(timestamps[: min(preview_len, timestamps.shape[0])], dtype=np.float64),
        "interframe_int": np.asarray(interframe_int[: min(preview_len - 1, interframe_int.shape[0])], dtype=np.float64),
    }

    elapsed = time.perf_counter() - session_start
    print(
        f"[session] {session.session_id}: neurons={neural_binned.shape[0]} "
        f"frames={n_frames} bins={neural_binned.shape[1]} missing_behavior_frames={missing_idx.size} "
        f"time={elapsed:.2f}s"
    )

    return {
        "info": session,
        "ops": ops,
        "brain_region_idx": np.zeros(neural_binned.shape[0], dtype=np.int64),
        "neural_binned": neural_binned,
        "time_binned": time_binned,
        "motion_binned": motion_binned,
        "missing_idx": missing_idx,
        "neural_preview": neural_preview,
        "motion_preview": motion_preview,
    }


def plot_processing(
    session_data: dict,
    motion_min: float,
    motion_max: float,
    motion_edges: np.ndarray,
    out_path: Path,
) -> None:
    info = session_data["info"]
    neural_preview = session_data["neural_preview"]
    motion_preview = session_data["motion_preview"]
    neural_binned = session_data["neural_binned"]
    time_binned = session_data["time_binned"]
    motion_binned = session_data["motion_binned"]

    motion_norm_binned = (motion_binned - motion_min) / (motion_max - motion_min)
    motion_classes = np.digitize(motion_norm_binned, motion_edges[1:-1], right=False)

    fig, ax = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle(f"Processing overview: {info.session_id}")

    preview_time = np.arange(neural_preview["F"].shape[0]) / RAW_FS_HZ
    ax[0, 0].plot(preview_time, neural_preview["F"], label="F", lw=1.0)
    ax[0, 0].plot(preview_time, neural_preview["Fneu"], label="Fneu", lw=1.0, alpha=0.7)
    ax[0, 0].plot(preview_time, neural_preview["corrected"], label="F - neucoeff*Fneu", lw=1.0)
    ax[0, 0].plot(preview_time, neural_preview["processed"], label="baseline-corrected", lw=1.0)
    ax[0, 0].set_title(f"Neural preprocessing (neuron {neural_preview['sample_neuron']})")
    ax[0, 0].set_xlabel("Time (s)")
    ax[0, 0].legend(fontsize=8)

    motion_time = np.arange(motion_preview["reconstructed"].shape[0]) / RAW_FS_HZ
    ax[0, 1].plot(motion_time, motion_preview["reconstructed"], color="black", lw=1.0, label="reconstructed/interpolated")
    ax[0, 1].scatter(
        np.arange(motion_preview["raw"].shape[0]) / RAW_FS_HZ,
        motion_preview["raw"],
        s=6,
        color="tab:blue",
        alpha=0.6,
        label="observed",
    )
    if motion_preview["missing_idx"].size:
        ax[0, 1].scatter(
            motion_preview["missing_idx"] / RAW_FS_HZ,
            motion_preview["reconstructed"][motion_preview["missing_idx"]],
            s=14,
            color="tab:red",
            label="inserted missing frames",
        )
    ax[0, 1].set_title("Behavior reconstruction")
    ax[0, 1].set_xlabel("Time (s)")
    ax[0, 1].legend(fontsize=8)

    show_bins = min(360, time_binned.shape[0])
    ax[1, 0].plot(time_binned[:show_bins], motion_binned[:show_bins], lw=1.0)
    ax[1, 0].set_title("Binned motion energy (first 2 minutes)")
    ax[1, 0].set_xlabel("Elapsed time (s)")

    sample_neurons = min(50, neural_binned.shape[0])
    sample_bins = min(720, neural_binned.shape[1])
    ax[1, 1].imshow(neural_binned[:sample_neurons, :sample_bins], aspect="auto", cmap="viridis")
    ax[1, 1].set_title("Binned neural activity")
    ax[1, 1].set_xlabel("Binned time")
    ax[1, 1].set_ylabel("Neuron")

    ax[2, 0].hist(motion_norm_binned, bins=100, color="0.7")
    for edge in motion_edges[1:-1]:
        ax[2, 0].axvline(edge, color="tab:red", ls="--", lw=1.0)
    ax[2, 0].set_title("Normalized motion histogram with quintile edges")
    ax[2, 0].set_xlabel("Normalized motion energy")

    ax[2, 1].step(time_binned[:show_bins], motion_classes[:show_bins], where="mid")
    ax[2, 1].set_title("Discretized motion classes (first 2 minutes)")
    ax[2, 1].set_xlabel("Elapsed time (s)")
    ax[2, 1].set_ylabel("Class")
    ax[2, 1].set_yticks(range(5), OUTPUT_CLASS_NAMES)

    trial_bins = int(TRIAL_DURATION_SEC * RAW_FS_HZ / BIN_FRAMES)
    trial_starts = np.arange(0, time_binned.shape[0] + 1, trial_bins)
    ax[3, 0].plot(time_binned, motion_norm_binned, lw=1.0)
    for start in trial_starts:
        if start < time_binned.shape[0]:
            ax[3, 0].axvline(time_binned[start], color="tab:orange", alpha=0.5)
    ax[3, 0].set_title("Trial segmentation boundaries")
    ax[3, 0].set_xlabel("Elapsed time (s)")

    first_trial = neural_binned[:sample_neurons, :trial_bins]
    ax[3, 1].imshow(first_trial, aspect="auto", cmap="viridis")
    ax[3, 1].set_title("First 2-minute trial neural matrix")
    ax[3, 1].set_xlabel("Binned time within trial")
    ax[3, 1].set_ylabel("Neuron")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_dataset(processed_sessions: list[dict]) -> tuple[dict, float, float, np.ndarray]:
    motion_all = np.concatenate([session["motion_binned"] for session in processed_sessions]).astype(np.float32)
    motion_min = float(np.min(motion_all))
    motion_max = float(np.max(motion_all))
    if not np.isfinite(motion_min) or not np.isfinite(motion_max) or motion_max <= motion_min:
        raise ValueError("Motion energy range is invalid after preprocessing.")

    motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)
    motion_edges = np.quantile(motion_norm_all, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).astype(np.float32)
    motion_edges = np.maximum.accumulate(motion_edges)

    trial_bins = int(TRIAL_DURATION_SEC * RAW_FS_HZ / BIN_FRAMES)

    subjects = sorted({session["info"].subject for session in processed_sessions})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}

    data = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": subjects,
        "subject_idx": np.array([subject_to_idx[session["info"].subject] for session in processed_sessions], dtype=np.int64),
        "brain_regions": [BRAIN_REGION],
        "brain_region_idx": [],
        "input_names": ["elapsed_time_sec"],
        "output_names": ["motion_energy_quintile"],
        "output_values": [OUTPUT_CLASS_NAMES],
        "metadata": {
            "task_description": "Decode spontaneous-motion state from longitudinal barrel-cortex calcium activity.",
            "time_bin_size": TIME_BIN_SIZE_MS,
            "temporal_alignment_event": "Start of each consecutive 2-minute block cut from a continuous recording session",
            "off_start": 0.0,
            "off_end": TRIAL_DURATION_SEC,
            "raw_frame_rate_hz": RAW_FS_HZ,
            "bin_frames": BIN_FRAMES,
            "trial_duration_sec": TRIAL_DURATION_SEC,
            "neural_trace_type": "Suite2p baseline-corrected fluorescence from F and Fneu using saved ops parameters",
            "behavior_signal": "Global motion energy from squared pixel-wise frame differences",
            "behavior_missing_frame_strategy": "Reinsert missing camera frames from timing gaps, then linearly interpolate them before binning",
            "motion_normalization": "Global min-max over all 10-frame-averaged motion-energy samples",
            "motion_discretization": "Global equal-percentile quintiles over normalized motion-energy samples",
            "motion_normalization_min": motion_min,
            "motion_normalization_max": motion_max,
            "motion_quintile_edges_normalized": motion_edges.tolist(),
            "session_ids": [session["info"].session_id for session in processed_sessions],
        },
    }

    for session in processed_sessions:
        motion_norm = (session["motion_binned"] - motion_min) / (motion_max - motion_min)
        neural_trials, input_trials, output_trials = segment_trials(
            session["neural_binned"],
            session["time_binned"],
            motion_norm,
            motion_edges,
            trial_bins=trial_bins,
        )
        if len(neural_trials) < 2:
            raise ValueError(f"Session {session['info'].session_id} has fewer than 2 trials after segmentation.")
        data["neural"].append(neural_trials)
        data["input"].append(input_trials)
        data["output"].append(output_trials)
        data["brain_region_idx"].append(session["brain_region_idx"])

    return data, motion_min, motion_max, motion_edges


def print_summary(data: dict) -> None:
    n_sessions = len(data["neural"])
    n_trials = sum(len(session_trials) for session_trials in data["neural"])
    total_neurons_across_sessions = sum(session_region_idx.shape[0] for session_region_idx in data["brain_region_idx"])
    trial_lengths = [trial.shape[1] for session in data["neural"] for trial in session]
    print(f"[summary] sessions={n_sessions} trials={n_trials} subjects={len(data['subjects'])}")
    print(f"[summary] total_neurons_across_sessions={total_neurons_across_sessions}")
    print(f"[summary] trial_bins={sorted(set(trial_lengths))}")

    outputs = np.concatenate([trial[0] for session in data["output"] for trial in session])
    counts = np.bincount(outputs, minlength=len(OUTPUT_CLASS_NAMES))
    freqs = counts / counts.sum()
    print(f"[summary] output_class_counts={counts.tolist()}")
    print(f"[summary] output_class_freqs={[round(float(x), 4) for x in freqs]}")


def main() -> None:
    args = parse_args()
    data_root = Path("data")
    if not data_root.exists():
        raise FileNotFoundError("Expected data/ directory not found.")

    all_sessions = discover_sessions(data_root)
    sessions = select_sample_sessions(all_sessions) if args.sample else all_sessions
    print(f"[setup] mode={'sample' if args.sample else 'full'} nsessions={len(sessions)}")

    total_start = time.perf_counter()
    processed_sessions = []
    for session in sessions:
        processed_sessions.append(process_session(session))

    data, motion_min, motion_max, motion_edges = build_dataset(processed_sessions)
    print_summary(data)

    with args.outpicklefile.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[output] wrote {args.outpicklefile}")

    if args.show_processing:
        for session in processed_sessions[:2]:
            plot_path = Path(f"processing_{session['info'].session_id}.png")
            plot_processing(session, motion_min, motion_max, motion_edges, plot_path)
            print(f"[plot] wrote {plot_path}")

    total_elapsed = time.perf_counter() - total_start
    mean_time = total_elapsed / max(len(sessions), 1)
    print(f"[timing] total={total_elapsed:.2f}s mean_per_session={mean_time:.2f}s")


if __name__ == "__main__":
    main()
