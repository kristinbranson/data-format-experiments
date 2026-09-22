#!/usr/bin/env python3

import argparse
import math
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


DATA_ROOT = Path("/app/data")
FRAME_RATE_HZ = 30.0
BIN_FRAMES = 10
BIN_SIZE_SEC = BIN_FRAMES / FRAME_RATE_HZ
BIN_SIZE_MS = BIN_SIZE_SEC * 1000.0
TRIAL_SECONDS = 60.0
TRIAL_FRAMES = int(round(TRIAL_SECONDS * FRAME_RATE_HZ))
TRIAL_BINS = TRIAL_FRAMES // BIN_FRAMES
BRAIN_REGION_NAME = "barrel cortex L2/3"
OUTPUT_NAMES = ["motion_energy_bin"]
OUTPUT_VALUES = [[
    "lowest_quintile",
    "second_quintile",
    "middle_quintile",
    "fourth_quintile",
    "highest_quintile",
]]


@dataclass(frozen=True)
class SessionInfo:
    subject: str
    session: str
    session_dir: Path
    n_neurons: int
    n_frames: int
    motion_len: int

    @property
    def session_id(self) -> str:
        return f"{self.subject}_{self.session}"

    @property
    def missing_motion_frames(self) -> int:
        return self.n_frames - self.motion_len


def sorted_subjects(data_root: Path) -> list[str]:
    return sorted(
        path.name for path in data_root.iterdir()
        if path.is_dir() and path.name.startswith("jm")
    )


def discover_sessions(data_root: Path) -> tuple[list[str], list[SessionInfo]]:
    subjects = sorted_subjects(data_root)
    session_infos: list[SessionInfo] = []
    for subject in subjects:
        subject_dir = data_root / subject
        for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            f_path = session_dir / "suite2p" / "plane0" / "F.npy"
            motion_path = session_dir / "move_deve" / "motion_energy_glob.npy"
            if not f_path.exists() or not motion_path.exists():
                continue
            f_mmap = np.load(f_path, mmap_mode="r")
            motion_mmap = np.load(motion_path, mmap_mode="r")
            session_infos.append(
                SessionInfo(
                    subject=subject,
                    session=session_dir.name,
                    session_dir=session_dir,
                    n_neurons=int(f_mmap.shape[0]),
                    n_frames=int(f_mmap.shape[1]),
                    motion_len=int(motion_mmap.shape[0]),
                )
            )
    return subjects, session_infos


def select_sample_sessions(session_infos: list[SessionInfo]) -> list[SessionInfo]:
    ordered = sorted(session_infos, key=lambda info: (info.subject, info.session))
    with_gaps = sorted(
        [info for info in ordered if info.missing_motion_frames > 0],
        key=lambda info: (-info.missing_motion_frames, info.subject, info.session),
    )
    without_gaps = [info for info in ordered if info.missing_motion_frames == 0]

    chosen: list[SessionInfo] = []
    if with_gaps:
        chosen.append(with_gaps[0])

    if without_gaps:
        preferred = None
        if chosen:
            preferred = next(
                (info for info in without_gaps if info.n_frames != chosen[0].n_frames),
                None,
            )
        if preferred is None:
            preferred = without_gaps[0]
        if preferred not in chosen:
            chosen.append(preferred)

    for info in ordered:
        if info not in chosen:
            chosen.append(info)
        if len(chosen) == 2:
            break

    return sorted(chosen[:2], key=lambda info: (info.subject, info.session))


def load_ops(session_dir: Path) -> dict:
    return np.load(
        session_dir / "suite2p" / "plane0" / "ops.npy",
        allow_pickle=True,
    ).item()


def infer_behavior_frame_indices(tstamps: np.ndarray, n_frames: int) -> np.ndarray:
    if tstamps.ndim != 1 or tstamps.size == 0:
        raise ValueError("Behavior timestamps must be a non-empty 1D array.")
    if tstamps.size == n_frames:
        return np.arange(n_frames, dtype=np.int64)
    if tstamps.size == 1:
        return np.array([0], dtype=np.int64)

    dt = float((tstamps[-1] - tstamps[0]) / (n_frames - 1))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Invalid timestamp step inferred from tstamps: {dt}")

    frame_idx = np.rint((tstamps - tstamps[0]) / dt).astype(np.int64)
    frame_idx -= frame_idx[0]

    if np.unique(frame_idx).size != frame_idx.size:
        raise ValueError("Behavior timestamps map multiple samples to the same imaging frame.")
    if np.any(np.diff(frame_idx) < 1):
        raise ValueError("Behavior frame indices are not strictly increasing.")
    if frame_idx[-1] >= n_frames:
        raise ValueError(
            f"Inferred behavior frame indices exceed imaging length: "
            f"last={frame_idx[-1]}, n_frames={n_frames}"
        )

    return frame_idx


def reconstruct_motion_to_imaging_grid(
    motion: np.ndarray,
    tstamps: np.ndarray,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_idx = infer_behavior_frame_indices(tstamps, n_frames)
    full_motion = np.full(n_frames, np.nan, dtype=np.float32)
    full_motion[frame_idx] = motion.astype(np.float32, copy=False)
    missing_mask = np.isnan(full_motion)
    valid_idx = np.flatnonzero(~missing_mask)
    full_motion = np.interp(
        np.arange(n_frames, dtype=np.float64),
        valid_idx.astype(np.float64),
        full_motion[valid_idx].astype(np.float64),
    ).astype(np.float32)
    return full_motion, frame_idx, missing_mask


def compute_suite2p_dff(session_dir: Path) -> tuple[np.ndarray, dict]:
    f_path = session_dir / "suite2p" / "plane0" / "F.npy"
    fneu_path = session_dir / "suite2p" / "plane0" / "Fneu.npy"
    ops = load_ops(session_dir)

    f = np.load(f_path).astype(np.float32, copy=False)
    fneu = np.load(fneu_path).astype(np.float32, copy=False)
    dff = f.copy()
    dff -= np.float32(ops["neucoeff"]) * fneu
    del f
    del fneu

    dff = dcnv.preprocess(
        dff,
        baseline=ops["baseline"],
        win_baseline=float(ops["win_baseline"]),
        sig_baseline=float(ops["sig_baseline"]),
        fs=float(ops["fs"]),
        prctile_baseline=float(ops["prctile_baseline"]),
        batch_size=int(ops.get("batch_size", 100)),
        device=torch.device("cpu"),
    ).astype(np.float32, copy=False)
    return dff, ops


def bin_array(values: np.ndarray, bin_frames: int) -> np.ndarray:
    n_time = values.shape[-1]
    usable = (n_time // bin_frames) * bin_frames
    if usable == 0:
        raise ValueError("Array is too short to bin.")
    if values.ndim == 1:
        trimmed = values[:usable]
        return trimmed.reshape(-1, bin_frames).mean(axis=1).astype(np.float32, copy=False)
    if values.ndim == 2:
        trimmed = values[:, :usable]
        return trimmed.reshape(trimmed.shape[0], -1, bin_frames).mean(axis=2).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported array ndim for binning: {values.ndim}")


def discretize_motion_quintiles(motion_binned: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    discrete = np.searchsorted(edges, motion_binned, side="right").astype(np.int64)
    if np.unique(discrete).size < 5:
        order = np.argsort(motion_binned, kind="stable")
        discrete = np.empty_like(order, dtype=np.int64)
        discrete[order] = np.minimum(4, (5 * np.arange(motion_binned.size)) // motion_binned.size)
        method = "rank_fallback"
    else:
        method = "quantile_edges"
    return discrete, edges, method


def split_session_into_trials(
    neural_binned: np.ndarray,
    time_binned: np.ndarray,
    motion_bins: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    n_bins = neural_binned.shape[1]
    usable = (n_bins // TRIAL_BINS) * TRIAL_BINS
    if usable < TRIAL_BINS * 2:
        raise ValueError("Session does not contain at least two 60-second trials after binning.")

    neural_binned = neural_binned[:, :usable]
    time_binned = time_binned[:usable]
    motion_bins = motion_bins[:usable]
    n_trials = usable // TRIAL_BINS

    neural_trials = []
    input_trials = []
    output_trials = []
    for trial_idx in range(n_trials):
        sl = slice(trial_idx * TRIAL_BINS, (trial_idx + 1) * TRIAL_BINS)
        neural_trials.append(neural_binned[:, sl].astype(np.float32, copy=False))
        input_trials.append(time_binned[np.newaxis, sl].astype(np.float32, copy=False))
        output_trials.append(motion_bins[np.newaxis, sl].astype(np.int64, copy=False))

    recon_neural = np.concatenate(neural_trials, axis=1)
    recon_time = np.concatenate([trial[0] for trial in input_trials])
    recon_output = np.concatenate([trial[0] for trial in output_trials])
    if not np.allclose(recon_neural, neural_binned):
        raise ValueError("Neural trial splitting failed reconstruction check.")
    if not np.allclose(recon_time, time_binned):
        raise ValueError("Input trial splitting failed reconstruction check.")
    if not np.array_equal(recon_output, motion_bins):
        raise ValueError("Output trial splitting failed reconstruction check.")

    return neural_trials, input_trials, output_trials


def save_processing_plot(
    session_info: SessionInfo,
    raw_f: np.ndarray,
    raw_fneu: np.ndarray,
    dff: np.ndarray,
    frame_idx: np.ndarray,
    raw_motion: np.ndarray,
    full_motion: np.ndarray,
    missing_mask: np.ndarray,
    motion_binned: np.ndarray,
    motion_edges: np.ndarray,
    motion_bins: np.ndarray,
    time_binned: np.ndarray,
    neural_trials: list[np.ndarray],
    output_trials: list[np.ndarray],
    out_dir: Path,
    discretization_method: str,
) -> None:
    neuron_idx = min(5, raw_f.shape[0] - 1)
    preview_frames = min(int(5 * 60 * FRAME_RATE_HZ), raw_f.shape[1])
    preview_bins = min(TRIAL_BINS, motion_binned.shape[0])
    preview_neurons = min(40, dff.shape[0])

    fig, axes = plt.subplots(3, 2, figsize=(18, 12), dpi=150)
    fig.suptitle(session_info.session_id)

    t_frames = np.arange(preview_frames) / FRAME_RATE_HZ
    neuropil_sub = raw_f[neuron_idx, :preview_frames] - load_ops(session_info.session_dir)["neucoeff"] * raw_fneu[neuron_idx, :preview_frames]
    axes[0, 0].plot(t_frames, raw_f[neuron_idx, :preview_frames], label="F", lw=0.8)
    axes[0, 0].plot(t_frames, raw_fneu[neuron_idx, :preview_frames], label="Fneu", lw=0.8)
    axes[0, 0].plot(t_frames, neuropil_sub, label="F-0.7*Fneu", lw=0.8)
    axes[0, 0].plot(t_frames, dff[neuron_idx, :preview_frames], label="baseline-corrected", lw=1.0)
    axes[0, 0].set_title("Neural preprocessing")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].legend(fontsize=8, ncol=2)

    observed_idx = frame_idx[frame_idx < preview_frames]
    observed_motion = raw_motion[:observed_idx.size]
    axes[0, 1].plot(t_frames, full_motion[:preview_frames], color="black", lw=1.0, label="reconstructed full motion")
    axes[0, 1].scatter(
        observed_idx / FRAME_RATE_HZ,
        observed_motion,
        s=6,
        color="tab:orange",
        alpha=0.7,
        label="observed motion samples",
    )
    if missing_mask[:preview_frames].any():
        missing_times = np.flatnonzero(missing_mask[:preview_frames]) / FRAME_RATE_HZ
        axes[0, 1].vlines(
            missing_times,
            ymin=float(np.nanmin(full_motion[:preview_frames])),
            ymax=float(np.nanmax(full_motion[:preview_frames])),
            color="tab:red",
            alpha=0.1,
            linewidth=0.5,
        )
    axes[0, 1].set_title("Behavior alignment to imaging grid")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].legend(fontsize=8)

    im = axes[1, 0].imshow(
        neural_trials[0][:preview_neurons, :],
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    axes[1, 0].set_title("First 60 s trial neural data (binned)")
    axes[1, 0].set_xlabel("Binned time")
    axes[1, 0].set_ylabel("Neuron")

    axes[1, 1].hist(motion_binned, bins=50, color="0.7", edgecolor="0.3")
    for edge in motion_edges:
        axes[1, 1].axvline(float(edge), color="tab:red", linestyle="--", lw=1.0)
    axes[1, 1].set_title(f"Motion histogram and quintile edges ({discretization_method})")
    axes[1, 1].set_xlabel("Binned motion energy")

    axes[2, 0].plot(time_binned[:preview_bins], motion_binned[:preview_bins], color="tab:blue", lw=1.0, label="binned motion")
    axes[2, 0].step(
        time_binned[:preview_bins],
        motion_bins[:preview_bins],
        where="mid",
        color="tab:green",
        lw=1.0,
        label="discrete quintile bin",
    )
    axes[2, 0].set_title("Continuous motion to categorical output")
    axes[2, 0].set_xlabel("Session time (s)")
    axes[2, 0].legend(fontsize=8)

    class_frac = np.bincount(motion_bins, minlength=5) / motion_bins.size
    summary_lines = [
        f"subject={session_info.subject}",
        f"session={session_info.session}",
        f"neurons={session_info.n_neurons}",
        f"frames={session_info.n_frames}",
        f"motion samples={session_info.motion_len}",
        f"missing behavior frames={int(missing_mask.sum())}",
        f"trials={len(neural_trials)}",
        f"trial bins={TRIAL_BINS}",
        f"class fractions={np.array2string(class_frac, precision=3)}",
        f"first trial output range={int(output_trials[0].min())}-{int(output_trials[0].max())}",
    ]
    axes[2, 1].axis("off")
    axes[2, 1].text(
        0.01,
        0.99,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plot_path = out_dir / f"processing_{session_info.session_id}.png"
    fig.savefig(plot_path)
    plt.close(fig)


def convert_one_session(session_info: SessionInfo, make_plot: bool, out_dir: Path) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    raw_f = np.load(session_info.session_dir / "suite2p" / "plane0" / "F.npy").astype(np.float32, copy=False)
    raw_fneu = np.load(session_info.session_dir / "suite2p" / "plane0" / "Fneu.npy").astype(np.float32, copy=False)
    raw_motion = np.load(session_info.session_dir / "move_deve" / "motion_energy_glob.npy")
    tstamps = np.load(session_info.session_dir / "move_deve" / "tstamps.npy")
    interframe_int = np.load(session_info.session_dir / "move_deve" / "interframe_int.npy")
    load_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    dff, ops = compute_suite2p_dff(session_info.session_dir)
    neural_sec = time.perf_counter() - t1

    t2 = time.perf_counter()
    full_motion, frame_idx, missing_mask = reconstruct_motion_to_imaging_grid(
        raw_motion,
        tstamps,
        session_info.n_frames,
    )
    motion_sec = time.perf_counter() - t2

    t3 = time.perf_counter()
    time_values = np.arange(session_info.n_frames, dtype=np.float32) / np.float32(FRAME_RATE_HZ)
    neural_binned = bin_array(dff, BIN_FRAMES)
    motion_binned = bin_array(full_motion, BIN_FRAMES)
    time_binned = bin_array(time_values, BIN_FRAMES)
    motion_bins, motion_edges, discretization_method = discretize_motion_quintiles(motion_binned)
    neural_trials, input_trials, output_trials = split_session_into_trials(
        neural_binned,
        time_binned,
        motion_bins,
    )
    post_sec = time.perf_counter() - t3

    inferred_missing = int(missing_mask.sum())
    expected_missing = int(session_info.missing_motion_frames)
    if inferred_missing != expected_missing:
        raise ValueError(
            f"Missing-frame mismatch for {session_info.session_id}: "
            f"expected {expected_missing}, inferred {inferred_missing}"
        )

    if raw_motion.shape[0] == session_info.n_frames:
        if not np.allclose(full_motion[frame_idx], raw_motion.astype(np.float32)):
            raise ValueError(f"Observed motion mismatch in no-drop session {session_info.session_id}")
    else:
        if not np.allclose(full_motion[frame_idx], raw_motion.astype(np.float32)):
            raise ValueError(f"Observed motion mismatch at valid timestamps in {session_info.session_id}")

    if not np.isfinite(neural_binned).all():
        raise ValueError(f"Non-finite neural values after preprocessing for {session_info.session_id}")
    if not np.isfinite(time_binned).all():
        raise ValueError(f"Non-finite input values for {session_info.session_id}")
    if not np.isfinite(motion_binned).all():
        raise ValueError(f"Non-finite motion values after reconstruction for {session_info.session_id}")

    if make_plot:
        save_processing_plot(
            session_info=session_info,
            raw_f=raw_f,
            raw_fneu=raw_fneu,
            dff=dff,
            frame_idx=frame_idx,
            raw_motion=raw_motion,
            full_motion=full_motion,
            missing_mask=missing_mask,
            motion_binned=motion_binned,
            motion_edges=motion_edges,
            motion_bins=motion_bins,
            time_binned=time_binned,
            neural_trials=neural_trials,
            output_trials=output_trials,
            out_dir=out_dir,
            discretization_method=discretization_method,
        )

    session_payload = {
        "neural_trials": neural_trials,
        "input_trials": input_trials,
        "output_trials": output_trials,
        "brain_region_idx": np.zeros(session_info.n_neurons, dtype=np.int64),
        "motion_edges": motion_edges.astype(np.float32),
        "n_trials": len(neural_trials),
    }
    session_stats = {
        "session_id": session_info.session_id,
        "subject": session_info.subject,
        "session": session_info.session,
        "n_neurons": session_info.n_neurons,
        "n_frames": session_info.n_frames,
        "n_trials": len(neural_trials),
        "motion_samples": int(raw_motion.shape[0]),
        "missing_motion_frames": inferred_missing,
        "n_binned_timepoints": int(neural_binned.shape[1]),
        "class_fraction": (np.bincount(motion_bins, minlength=5) / motion_bins.size).tolist(),
        "motion_edges": motion_edges.astype(float).tolist(),
        "tstamps_start": float(tstamps[0]),
        "tstamps_end": float(tstamps[-1]),
        "interframe_median": float(np.median(interframe_int)),
        "load_sec": load_sec,
        "neural_sec": neural_sec,
        "motion_sec": motion_sec,
        "post_sec": post_sec,
        "total_sec": time.perf_counter() - t0,
        "discretization_method": discretization_method,
        "ops_fs": float(ops["fs"]),
        "ops_nframes": int(ops["nframes"]),
        "ops_baseline": str(ops["baseline"]),
    }
    return session_payload, session_stats


def build_dataset(
    session_infos: list[SessionInfo],
    all_subjects_sorted: list[str],
    show_processing: bool,
    out_path: Path,
) -> tuple[dict, list[dict]]:
    selected_subjects = sorted({info.subject for info in session_infos})
    subject_to_idx = {subject: idx for idx, subject in enumerate(selected_subjects)}
    plot_session_ids = {
        info.session_id for info in session_infos[: min(2, len(session_infos))]
    } if show_processing else set()

    data = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": selected_subjects,
        "subject_idx": [],
        "brain_regions": [BRAIN_REGION_NAME],
        "brain_region_idx": [],
        "input_names": ["time_from_session_start_sec"],
        "output_names": OUTPUT_NAMES,
        "output_values": OUTPUT_VALUES,
        "metadata": {
            "task_description": (
                "Decode session-relative spontaneous motion energy from barrel cortex "
                "population activity. Sessions are split into contiguous non-overlapping "
                "60 s windows."
            ),
            "time_bin_size": BIN_SIZE_MS,
            "temporal_alignment_event": "start of each contiguous 60-second session window",
            "off_start": 0.0,
            "off_end": TRIAL_SECONDS,
            "raw_frame_rate_hz": FRAME_RATE_HZ,
            "bin_size_frames": BIN_FRAMES,
            "trial_length_seconds": TRIAL_SECONDS,
            "neural_signal": (
                "Suite2p neuropil-subtracted baseline-corrected fluorescence "
                "(F - neucoeff*Fneu, then suite2p.extraction.dcnv.preprocess)"
            ),
            "behavior_signal": (
                "Global motion energy reconstructed on the imaging frame grid from "
                "move_deve/motion_energy_glob.npy and timestamps"
            ),
            "motion_binning": "Five session-specific equal-percentile bins on 10-frame-averaged motion energy",
            "source_data_root": str(DATA_ROOT),
            "all_subjects_in_release": all_subjects_sorted,
            "session_ids": [],
            "session_motion_bin_edges": [],
            "session_missing_motion_frames": [],
            "session_info": [],
        },
    }

    session_stats: list[dict] = []
    total_frames = sum(info.n_frames for info in session_infos)
    frames_done = 0
    start = time.perf_counter()

    for idx, session_info in enumerate(session_infos, start=1):
        make_plot = session_info.session_id in plot_session_ids
        payload, stats = convert_one_session(session_info, make_plot=make_plot, out_dir=out_path.parent)
        data["neural"].append(payload["neural_trials"])
        data["input"].append(payload["input_trials"])
        data["output"].append(payload["output_trials"])
        data["subject_idx"].append(subject_to_idx[session_info.subject])
        data["brain_region_idx"].append(payload["brain_region_idx"])
        data["metadata"]["session_ids"].append(session_info.session_id)
        data["metadata"]["session_motion_bin_edges"].append(stats["motion_edges"])
        data["metadata"]["session_missing_motion_frames"].append(stats["missing_motion_frames"])
        data["metadata"]["session_info"].append({
            "session_id": session_info.session_id,
            "subject": session_info.subject,
            "session": session_info.session,
            "n_neurons": stats["n_neurons"],
            "n_frames": stats["n_frames"],
            "n_trials": stats["n_trials"],
            "missing_motion_frames": stats["missing_motion_frames"],
            "motion_discretization": stats["discretization_method"],
        })
        session_stats.append(stats)

        frames_done += session_info.n_frames
        elapsed = time.perf_counter() - start
        sec_per_frame = elapsed / frames_done
        eta = sec_per_frame * (total_frames - frames_done)
        print(
            f"[{idx:02d}/{len(session_infos):02d}] {session_info.session_id}: "
            f"neurons={stats['n_neurons']}, frames={stats['n_frames']}, "
            f"trials={stats['n_trials']}, missing_motion={stats['missing_motion_frames']}, "
            f"time={stats['total_sec']:.2f}s, eta={eta:.1f}s",
            flush=True,
        )

    data["subject_idx"] = np.asarray(data["subject_idx"], dtype=np.int64)
    return data, session_stats


def print_summary(session_stats: list[dict], all_session_infos: list[SessionInfo]) -> None:
    total_trials = sum(stats["n_trials"] for stats in session_stats)
    total_neuron_session = sum(stats["n_neurons"] for stats in session_stats)
    mean_neurons = np.mean([stats["n_neurons"] for stats in session_stats])
    mean_trials = np.mean([stats["n_trials"] for stats in session_stats])
    total_sec = sum(stats["total_sec"] for stats in session_stats)
    processed_frames = sum(stats["n_frames"] for stats in session_stats)
    sec_per_frame = total_sec / processed_frames
    full_frames = sum(info.n_frames for info in all_session_infos)
    est_full_sec = sec_per_frame * full_frames

    all_class_fracs = np.array([stats["class_fraction"] for stats in session_stats], dtype=np.float64)
    print("\nConversion summary", flush=True)
    print(f"  Sessions processed: {len(session_stats)}", flush=True)
    print(f"  Total trials: {total_trials}", flush=True)
    print(f"  Total neuron-session entries: {total_neuron_session}", flush=True)
    print(f"  Mean neurons/session: {mean_neurons:.2f}", flush=True)
    print(f"  Mean trials/session: {mean_trials:.2f}", flush=True)
    print(f"  Mean class fractions: {np.mean(all_class_fracs, axis=0)}", flush=True)
    print(f"  Processing time/session: {np.mean([stats['total_sec'] for stats in session_stats]):.2f}s", flush=True)
    print(f"  Processing time/frame: {sec_per_frame:.6f}s", flush=True)
    print(f"  Estimated full-dataset conversion time from this run: {est_full_sec:.1f}s", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Track2p dataset into decoder format.")
    parser.add_argument("outpicklefile", type=Path, help="Output pickle file.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only two representative sessions.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save detailed processing figures for up to two sessions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.outpicklefile.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_subjects_sorted, all_session_infos = discover_sessions(DATA_ROOT)
    if not all_session_infos:
        raise RuntimeError(f"No sessions found in {DATA_ROOT}")

    mode = "sample" if args.sample else "full"
    if mode == "sample":
        session_infos = select_sample_sessions(all_session_infos)
    else:
        session_infos = sorted(all_session_infos, key=lambda info: (info.subject, info.session))

    print(f"Mode: {mode}", flush=True)
    print(f"Output: {out_path}", flush=True)
    print("Sessions selected:", flush=True)
    for info in session_infos:
        print(
            f"  {info.session_id}: neurons={info.n_neurons}, frames={info.n_frames}, "
            f"motion_len={info.motion_len}, missing_motion={info.missing_motion_frames}",
            flush=True,
        )

    data, session_stats = build_dataset(
        session_infos=session_infos,
        all_subjects_sorted=all_subjects_sorted,
        show_processing=args.show_processing,
        out_path=out_path,
    )
    print_summary(session_stats, all_session_infos)

    with open(out_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved converted dataset to {out_path}", flush=True)


if __name__ == "__main__":
    main()
