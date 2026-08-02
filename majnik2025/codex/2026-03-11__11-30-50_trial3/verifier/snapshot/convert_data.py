#!/usr/bin/env python3
import argparse
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from suite2p.extraction import dcnv

from decoder import verify_data_format


DATA_ROOT = Path("data")
MOTION_BIN_SIZE_FRAMES = 10
IMAGING_FS = 30.0
BLOCK_DURATION_SEC = 120.0
BLOCK_BINS = int(BLOCK_DURATION_SEC * IMAGING_FS / MOTION_BIN_SIZE_FRAMES)


@dataclass(frozen=True)
class SessionInfo:
    subject: str
    session_id: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the Track2p developmental barrel cortex dataset."
    )
    parser.add_argument("outpicklefile", help="Path to output pickle file.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only 2 sessions.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing visualizations for up to 2 sessions.",
    )
    return parser.parse_args()


def discover_sessions(sample: bool) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    for subject_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("jm")):
        session_dirs = sorted(
            p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()
        )
        for session_dir in session_dirs:
            sessions.append(
                SessionInfo(
                    subject=subject_dir.name,
                    session_id=f"{subject_dir.name}_{session_dir.name}",
                    path=session_dir,
                )
            )
    if sample:
        selected: list[SessionInfo] = []
        seen_nframes: set[int] = set()
        for session in sessions:
            nframes = int(load_ops(session)["nframes"])
            if nframes not in seen_nframes:
                selected.append(session)
                seen_nframes.add(nframes)
            if len(selected) == 2:
                break
        if len(selected) < 2:
            for session in sessions:
                if session not in selected:
                    selected.append(session)
                if len(selected) == 2:
                    break
        return selected
    return sessions


def load_ops(session: SessionInfo) -> dict:
    return np.load(
        session.path / "suite2p" / "plane0" / "ops.npy", allow_pickle=True
    ).item()


def align_motion_to_imaging(
    motion: np.ndarray, tstamps: np.ndarray, nframes: int
) -> tuple[np.ndarray, dict]:
    if len(motion) == nframes:
        aligned = motion.astype(np.float32, copy=False)
        stats = {
            "raw_motion_len": int(len(motion)),
            "aligned_motion_len": int(len(aligned)),
            "missing_frames": 0,
            "duplicate_timestamp_bins": 0,
        }
        return aligned, stats

    if len(motion) != len(tstamps):
        raise ValueError(f"motion/tstamps length mismatch: {len(motion)} vs {len(tstamps)}")

    denom = tstamps[-1] - tstamps[0]
    if denom <= 0:
        raise ValueError("Non-increasing behavior timestamps")

    frame_idx = np.round((tstamps - tstamps[0]) / denom * (nframes - 1)).astype(np.int64)
    frame_idx = np.clip(frame_idx, 0, nframes - 1)

    summed = np.zeros(nframes, dtype=np.float64)
    counts = np.zeros(nframes, dtype=np.int64)
    np.add.at(summed, frame_idx, motion.astype(np.float64))
    np.add.at(counts, frame_idx, 1)

    aligned = np.full(nframes, np.nan, dtype=np.float64)
    good = counts > 0
    aligned[good] = summed[good] / counts[good]

    known_idx = np.flatnonzero(good)
    if known_idx.size == 0:
        raise ValueError("No valid behavior samples after timestamp mapping")
    if known_idx.size == 1:
        aligned[:] = aligned[known_idx[0]]
    else:
        missing = np.flatnonzero(~good)
        if missing.size:
            aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])

    stats = {
        "raw_motion_len": int(len(motion)),
        "aligned_motion_len": int(len(aligned)),
        "missing_frames": int(nframes - known_idx.size),
        "duplicate_timestamp_bins": int(len(frame_idx) - len(np.unique(frame_idx))),
    }
    return aligned.astype(np.float32), stats


def bin_average_1d(x: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (len(x) // bin_size) * bin_size
    x = x[:usable]
    return x.reshape(-1, bin_size).mean(axis=1, dtype=np.float64).astype(np.float32)


def bin_average_2d(x: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (x.shape[1] // bin_size) * bin_size
    x = x[:, :usable]
    nbins = usable // bin_size
    return x.reshape(x.shape[0], nbins, bin_size).mean(axis=2, dtype=np.float64).astype(np.float32)


def compute_suite2p_baseline_corrected(F: np.ndarray, Fneu: np.ndarray, ops: dict) -> np.ndarray:
    fc = F.astype(np.float32, copy=False) - float(ops.get("neucoeff", 0.7)) * Fneu.astype(
        np.float32, copy=False
    )
    return dcnv.preprocess(
        fc.copy(),
        baseline=ops.get("baseline", "maximin"),
        win_baseline=float(ops.get("win_baseline", 60.0)),
        sig_baseline=float(ops.get("sig_baseline", 10.0)),
        fs=float(ops.get("fs", IMAGING_FS)),
        prctile_baseline=float(ops.get("prctile_baseline", 8.0)),
        batch_size=int(ops.get("batch_size", 100)),
        device=torch.device("cpu"),
    ).astype(np.float32, copy=False)


def split_into_blocks_2d(x: np.ndarray, block_bins: int) -> list[np.ndarray]:
    nblocks = x.shape[1] // block_bins
    usable = nblocks * block_bins
    x = x[:, :usable]
    return [x[:, i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]


def split_into_blocks_1d(x: np.ndarray, block_bins: int) -> list[np.ndarray]:
    nblocks = len(x) // block_bins
    usable = nblocks * block_bins
    x = x[:usable]
    return [x[i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]


def make_time_input(nbins: int, fs: float, bin_size_frames: int) -> np.ndarray:
    step = bin_size_frames / fs
    return (np.arange(nbins, dtype=np.float32) * step)[None, :]


def motion_to_bins(x: np.ndarray, quantile_edges: np.ndarray) -> np.ndarray:
    return np.digitize(x, quantile_edges, right=False).astype(np.int64)


def summarize_timing(start_time: float, label: str) -> None:
    print(f"{label}: {time.perf_counter() - start_time:.2f}s")


def plot_processing(
    session: SessionInfo,
    raw_f: np.ndarray,
    raw_fneu: np.ndarray,
    dff: np.ndarray,
    neural_binned: np.ndarray,
    motion_raw: np.ndarray,
    motion_aligned: np.ndarray,
    motion_binned_norm: np.ndarray,
    motion_bins: np.ndarray,
    time_binned: np.ndarray,
    quantile_edges: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    neuron = 0

    axes[0, 0].plot(raw_f[neuron], label="F", alpha=0.9)
    axes[0, 0].plot(raw_fneu[neuron], label="Fneu", alpha=0.7)
    axes[0, 0].set_title("Raw fluorescence and neuropil")
    axes[0, 0].legend(loc="upper right")

    axes[0, 1].plot(dff[neuron], color="tab:green")
    axes[0, 1].set_title("Suite2p baseline-corrected fluorescence")

    nneurons_show = min(40, neural_binned.shape[0])
    vmax = np.percentile(neural_binned[:nneurons_show], 99)
    axes[1, 0].imshow(
        neural_binned[:nneurons_show],
        aspect="auto",
        cmap="gray_r",
        vmin=0,
        vmax=vmax if vmax > 0 else 1.0,
    )
    axes[1, 0].set_title("Binned neural activity")
    axes[1, 0].set_ylabel("Neuron")

    axes[1, 1].plot(motion_raw, alpha=0.5, label="raw motion")
    axes[1, 1].plot(motion_aligned, alpha=0.8, label="aligned/interpolated")
    axes[1, 1].set_title("Motion alignment to imaging frames")
    axes[1, 1].legend(loc="upper right")

    axes[2, 0].plot(time_binned, motion_binned_norm, color="tab:blue", label="normalized motion")
    for edge in quantile_edges:
        axes[2, 0].axhline(edge, color="tab:red", linestyle="--", alpha=0.5)
    axes[2, 0].set_title("10-frame averaged normalized motion")
    axes[2, 0].set_xlabel("Elapsed time (s)")

    axes[2, 1].step(time_binned, motion_bins, where="mid", color="tab:purple")
    axes[2, 1].set_title("Discretized motion bins")
    axes[2, 1].set_xlabel("Elapsed time (s)")
    axes[2, 1].set_ylabel("Bin")

    fig.suptitle(session.session_id)
    fig.tight_layout()
    fig.savefig(f"processing_{session.session_id}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.full and not args.sample:
        args.full = True

    all_start = time.perf_counter()
    sessions = discover_sessions(sample=args.sample)
    if not sessions:
        raise RuntimeError("No sessions found in data/")
    print(f"Discovered {len(sessions)} sessions")

    pass1_start = time.perf_counter()
    motion_binned_by_session: dict[str, np.ndarray] = {}
    session_meta: list[dict] = []
    all_motion_binned: list[np.ndarray] = []

    for session in sessions:
        ops = load_ops(session)
        nframes = int(ops["nframes"])
        motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
        tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
        motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)
        motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)
        motion_binned_by_session[session.session_id] = motion_binned
        all_motion_binned.append(motion_binned)
        session_meta.append(
            {
                "session_id": session.session_id,
                "subject": session.subject,
                "session_path": str(session.path),
                "nframes_raw": nframes,
                "duration_sec": nframes / float(ops["fs"]),
                "nneurons": int(np.load(session.path / "suite2p" / "plane0" / "F.npy", mmap_mode="r").shape[0]),
                "behavior_missing_frames": align_stats["missing_frames"],
                "behavior_duplicate_timestamp_bins": align_stats["duplicate_timestamp_bins"],
                "binned_timepoints": int(len(motion_binned)),
                "n_trials_expected": int(len(motion_binned) // BLOCK_BINS),
            }
        )

    all_motion = np.concatenate(all_motion_binned)
    motion_min = float(all_motion.min())
    motion_max = float(all_motion.max())
    motion_norm_all = (all_motion - motion_min) / max(motion_max - motion_min, 1e-12)
    quantile_edges = np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    summarize_timing(pass1_start, "Pass 1 (behavior scan)")

    pass2_start = time.perf_counter()
    subjects = sorted({session.subject for session in sessions})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}

    converted = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": subjects,
        "subject_idx": np.array([subject_to_idx[session.subject] for session in sessions], dtype=np.int64),
        "brain_regions": ["barrel cortex L2/3"],
        "brain_region_idx": [],
        "input_names": ["elapsed_time_sec"],
        "output_names": ["motion_energy_bin"],
        "output_values": [[
            "0-20pct",
            "20-40pct",
            "40-60pct",
            "60-80pct",
            "80-100pct",
        ]],
        "metadata": {
            "task_description": "Decode spontaneous animal motion energy quintile from barrel cortex population activity during continuous longitudinal imaging sessions.",
            "time_bin_size": 1000.0 * MOTION_BIN_SIZE_FRAMES / IMAGING_FS,
            "temporal_alignment_event": "start of each consecutive 2-minute block; input stores absolute elapsed time from session start",
            "off_start": 0.0,
            "off_end": BLOCK_DURATION_SEC,
            "preprocessing": {
                "neural_trace": "Suite2p baseline-corrected fluorescence reconstructed from F and Fneu with per-session ops parameters",
                "neural_bin_size_frames": MOTION_BIN_SIZE_FRAMES,
                "behavior_bin_size_frames": MOTION_BIN_SIZE_FRAMES,
                "trial_definition": "consecutive non-overlapping 2-minute blocks",
                "motion_alignment": "behavior timestamps mapped to imaging frame grid with linear interpolation over missing camera frames",
                "motion_normalization": "global min-max across included sessions after 10-frame averaging",
                "motion_quantile_edges": quantile_edges.tolist(),
            },
            "session_info": session_meta,
        },
    }

    show_processing_ids = {session.session_id for session in sessions[:2]} if args.show_processing else set()

    for idx, session in enumerate(sessions):
        session_start = time.perf_counter()
        ops = load_ops(session)
        F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
        Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
        motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
        tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)

        motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])
        motion_binned = motion_binned_by_session[session.session_id]
        motion_binned_norm = ((motion_binned - motion_min) / max(motion_max - motion_min, 1e-12)).astype(
            np.float32
        )
        motion_disc = motion_to_bins(motion_binned_norm, quantile_edges)

        neural = compute_suite2p_baseline_corrected(F, Fneu, ops)
        neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
        time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]

        if neural_binned.shape[1] != len(motion_disc):
            raise ValueError(
                f"{session.session_id}: neural/motion binned length mismatch "
                f"{neural_binned.shape[1]} vs {len(motion_disc)}"
            )
        if neural_binned.shape[1] % BLOCK_BINS != 0:
            raise ValueError(f"{session.session_id}: binned timepoints not divisible by {BLOCK_BINS}")

        neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
        input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
        output_trials = [x[None, :].astype(np.int64) for x in split_into_blocks_1d(motion_disc, BLOCK_BINS)]

        if not (len(neural_trials) == len(input_trials) == len(output_trials)):
            raise ValueError(f"{session.session_id}: trial count mismatch after block splitting")
        if len(neural_trials) < 2:
            raise ValueError(f"{session.session_id}: needs at least two trials, got {len(neural_trials)}")

        converted["neural"].append([x.astype(np.float32, copy=False) for x in neural_trials])
        converted["input"].append(input_trials)
        converted["output"].append(output_trials)
        converted["brain_region_idx"].append(np.zeros(neural_binned.shape[0], dtype=np.int64))

        if session.session_id in show_processing_ids:
            plot_processing(
                session=session,
                raw_f=F,
                raw_fneu=Fneu,
                dff=neural,
                neural_binned=neural_binned,
                motion_raw=motion.astype(np.float32, copy=False),
                motion_aligned=motion_aligned,
                motion_binned_norm=motion_binned_norm,
                motion_bins=motion_disc,
                time_binned=time_binned,
                quantile_edges=quantile_edges,
            )

        print(
            f"[{idx + 1}/{len(sessions)}] {session.session_id}: "
            f"neurons={neural_binned.shape[0]}, trials={len(neural_trials)}, "
            f"binned_T={neural_binned.shape[1]} in {time.perf_counter() - session_start:.2f}s"
        )

    summarize_timing(pass2_start, "Pass 2 (neural conversion)")

    valid, errors, warnings = verify_data_format(converted)
    if not valid:
        raise RuntimeError("Converted data failed validation:\n" + "\n".join(errors))
    if warnings:
        print("Validation warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("Validation passed with no warnings")

    with open(args.outpicklefile, "wb") as f:
        pickle.dump(converted, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved converted dataset to {args.outpicklefile}")
    summarize_timing(all_start, "Total conversion time")


if __name__ == "__main__":
    main()
