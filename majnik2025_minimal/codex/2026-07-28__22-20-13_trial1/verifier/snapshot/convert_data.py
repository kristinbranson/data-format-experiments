import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from suite2p.extraction import dcnv

from decoder import print_data_summary, verify_data_format


DATA_ROOT = Path("/app/data")
BRAIN_REGION = "barrel cortex"
FRAME_RATE_HZ = 30.0
TRIAL_SECONDS = 120.0
TRIAL_FRAMES = int(TRIAL_SECONDS * FRAME_RATE_HZ)
BIN_FRAMES = 10
BINS_PER_TRIAL = TRIAL_FRAMES // BIN_FRAMES
TIME_BIN_SIZE_MS = 1000.0 * BIN_FRAMES / FRAME_RATE_HZ


def sorted_subject_dirs(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("jm")]
    )


def sorted_session_dirs(subject_dir: Path) -> list[Path]:
    return sorted(
        [path for path in subject_dir.iterdir() if path.is_dir() and path.name[:4].isdigit()]
    )


def collect_session_dirs(root: Path, mode: str) -> list[Path]:
    session_dirs = []
    for subject_dir in sorted_subject_dirs(root):
        sessions = sorted_session_dirs(subject_dir)
        if mode == "sample":
            if sessions:
                session_dirs.append(sessions[0])
        else:
            session_dirs.extend(sessions)
    return session_dirs


def load_suite2p_dff(session_dir: Path) -> tuple[np.ndarray, dict]:
    plane_dir = session_dir / "suite2p" / "plane0"
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
    Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)

    Fcorr = F
    Fcorr -= float(ops["neucoeff"]) * Fneu
    del Fneu

    dff_num = Fcorr.copy()
    dff_num = dcnv.preprocess(
        F=dff_num,
        baseline=ops["baseline"],
        win_baseline=ops["win_baseline"],
        sig_baseline=ops["sig_baseline"],
        fs=ops["fs"],
        prctile_baseline=ops["prctile_baseline"],
        batch_size=ops.get("batch_size", 100),
        device=torch.device("cpu"),
    )

    baseline = Fcorr
    baseline -= dff_num
    np.maximum(baseline, 1e-3, out=baseline)
    dff_num /= baseline
    return dff_num.astype(np.float32, copy=False), ops


def repair_motion_energy(session_dir: Path, n_frames: int) -> tuple[np.ndarray, int]:
    move_dir = session_dir / "move_deve"
    motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float32, copy=False)
    if len(motion) == n_frames:
        return motion, 0

    interframe = np.load(move_dir / "interframe_int.npy").astype(np.float64, copy=False) * 1000.0
    nominal = float(np.median(interframe))
    jumps = np.maximum(0, np.rint(interframe / nominal).astype(np.int64) - 1)
    missing_count = int(jumps.sum())
    expected_missing = n_frames - len(motion)
    if missing_count != expected_missing:
        raise ValueError(
            f"{session_dir}: inferred {missing_count} missing motion frames but expected {expected_missing}"
        )

    repaired = np.empty(n_frames, dtype=np.float32)
    src = 0
    dst = 0
    for gap in jumps:
        repaired[dst] = motion[src]
        src += 1
        dst += 1
        if gap:
            repaired[dst:dst + gap] = np.nan
            dst += gap
    repaired[dst] = motion[src]
    dst += 1
    src += 1
    if dst != n_frames or src != len(motion):
        raise ValueError(
            f"{session_dir}: motion repair finished at dst={dst}, src={src}, expected ({n_frames}, {len(motion)})"
        )

    nan_mask = np.isnan(repaired)
    if nan_mask.any():
        idx = np.arange(n_frames)
        repaired[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], repaired[~nan_mask]).astype(np.float32)
    return repaired, missing_count


def session_to_trials(
    neural_full: np.ndarray, motion_full: np.ndarray, session_duration_frames: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    usable_frames = (session_duration_frames // TRIAL_FRAMES) * TRIAL_FRAMES
    if usable_frames < TRIAL_FRAMES:
        raise ValueError(f"Session has only {session_duration_frames} frames, not enough for one 2-minute block")
    if usable_frames != session_duration_frames:
        print(
            f"  Dropping trailing frames: kept {usable_frames} of {session_duration_frames} for consecutive 2-minute blocks"
        )

    n_trials = usable_frames // TRIAL_FRAMES
    if n_trials < 2:
        raise ValueError(f"Session only has {n_trials} trials after blocking; at least 2 are required")

    neural_binned = neural_full[:, :usable_frames].reshape(
        neural_full.shape[0], n_trials, BINS_PER_TRIAL, BIN_FRAMES
    ).mean(axis=3)
    motion_binned = motion_full[:usable_frames].reshape(
        n_trials, BINS_PER_TRIAL, BIN_FRAMES
    ).mean(axis=2)

    base_time = (np.arange(BINS_PER_TRIAL, dtype=np.float32) * BIN_FRAMES + (BIN_FRAMES / 2.0)) / FRAME_RATE_HZ
    time_trials = [
        (base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :].astype(np.float32, copy=False)
        for trial_index in range(n_trials)
    ]
    neural_trials = [
        neural_binned[:, trial_index, :].astype(np.float32, copy=True) for trial_index in range(n_trials)
    ]
    motion_trials = [
        motion_binned[trial_index, :].astype(np.float32, copy=True) for trial_index in range(n_trials)
    ]
    return neural_trials, time_trials, motion_trials


def build_dataset(mode: str) -> tuple[dict, dict]:
    session_dirs = collect_session_dirs(DATA_ROOT, mode)
    if not session_dirs:
        raise ValueError(f"No sessions found under {DATA_ROOT}")

    subjects = sorted({session_dir.parent.name for session_dir in session_dirs})
    subject_to_index = {subject: index for index, subject in enumerate(subjects)}

    session_records = []
    missing_behavior_by_session = {}
    duration_minutes = []
    neurons_per_mouse = {}

    print(f"Building {mode} dataset from {len(session_dirs)} recording sessions")
    for session_index, session_dir in enumerate(session_dirs):
        subject = session_dir.parent.name
        session_id = session_dir.name
        session_key = f"{subject}/{session_id}"
        print(f"[{session_index + 1:02d}/{len(session_dirs):02d}] {session_key}")

        neural_full, ops = load_suite2p_dff(session_dir)
        if float(ops["fs"]) != FRAME_RATE_HZ:
            raise ValueError(f"{session_key}: expected {FRAME_RATE_HZ} Hz but found {ops['fs']}")

        n_frames = neural_full.shape[1]
        motion_full, missing_count = repair_motion_energy(session_dir, n_frames)
        neural_trials, time_trials, motion_trials = session_to_trials(neural_full, motion_full, n_frames)

        duration_minutes.append(n_frames / FRAME_RATE_HZ / 60.0)
        neurons_per_mouse.setdefault(subject, neural_full.shape[0])
        missing_behavior_by_session[session_key] = missing_count

        session_records.append(
            {
                "subject": subject,
                "session_id": session_id,
                "session_key": session_key,
                "n_frames": int(n_frames),
                "n_neurons": int(neural_full.shape[0]),
                "n_trials": len(neural_trials),
                "neural_trials": neural_trials,
                "input_trials": time_trials,
                "motion_trials": motion_trials,
            }
        )

    all_motion = np.concatenate(
        [trial for record in session_records for trial in record["motion_trials"]]
    ).astype(np.float32, copy=False)
    motion_min = float(all_motion.min())
    motion_max = float(all_motion.max())
    motion_scale = motion_max - motion_min
    if motion_scale <= 0:
        raise ValueError("Motion energy has zero range after processing")

    motion_quantiles_raw = np.percentile(all_motion, [20, 40, 60, 80]).astype(np.float32)
    motion_quantiles_norm = ((motion_quantiles_raw - motion_min) / motion_scale).astype(np.float32)

    data = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": subjects,
        "subject_idx": np.array(
            [subject_to_index[record["subject"]] for record in session_records], dtype=np.int64
        ),
        "brain_regions": [BRAIN_REGION],
        "brain_region_idx": [],
        "input_names": ["time_from_session_start_s"],
        "output_names": ["motion_energy_quantile"],
        "output_values": [[
            "lowest_20pct",
            "20_40pct",
            "40_60pct",
            "60_80pct",
            "highest_20pct",
        ]],
        "metadata": {
            "task_description": (
                "Decode spontaneous-behavior motion-energy quintile from longitudinal barrel-cortex calcium activity."
            ),
            "time_bin_size": TIME_BIN_SIZE_MS,
            "temporal_alignment_event": "start of each consecutive 2-minute block from a continuous recording",
            "off_start": 0.0,
            "off_end": TRIAL_SECONDS,
            "raw_frame_rate_hz": FRAME_RATE_HZ,
            "raw_trial_definition_seconds": TRIAL_SECONDS,
            "raw_bin_frames": BIN_FRAMES,
            "neural_signal": (
                "Suite2p neuropil-subtracted fluorescence with Suite2p default baseline estimation, converted to dF/F, then averaged in 10-frame bins."
            ),
            "behavior_signal": (
                "Global motion energy from videography; missing camera frames repaired from interframe intervals and linearly interpolated before 10-frame averaging."
            ),
            "motion_energy_normalization": {
                "type": "global_min_max_after_10_frame_averaging",
                "min": motion_min,
                "max": motion_max,
            },
            "motion_energy_percentiles_raw": motion_quantiles_raw.tolist(),
            "motion_energy_percentiles_normalized": motion_quantiles_norm.tolist(),
            "source_paper": (
                "Majnik et al. 2025, Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p."
            ),
            "session_ids": [record["session_key"] for record in session_records],
            "session_frame_counts": [record["n_frames"] for record in session_records],
            "session_trial_counts": [record["n_trials"] for record in session_records],
            "missing_behavior_frames_by_session": missing_behavior_by_session,
            "conversion_mode": mode,
        },
    }

    for record in session_records:
        data["neural"].append(record["neural_trials"])
        data["input"].append(record["input_trials"])
        output_trials = []
        for motion_trial in record["motion_trials"]:
            motion_norm = (motion_trial - motion_min) / motion_scale
            motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False).astype(np.int64)
            output_trials.append(motion_bins[np.newaxis, :])
        data["output"].append(output_trials)
        data["brain_region_idx"].append(np.zeros(record["n_neurons"], dtype=np.int64))

    summary = {
        "mode": mode,
        "nsessions": len(session_records),
        "subjects": subjects,
        "neurons_per_mouse": neurons_per_mouse,
        "duration_minutes_unique": sorted({round(value, 3) for value in duration_minutes}),
        "total_missing_behavior_frames": int(sum(missing_behavior_by_session.values())),
        "sessions_with_missing_behavior": {
            key: value for key, value in missing_behavior_by_session.items() if value
        },
        "paper_check_tracked_neurons_mean": float(np.mean(list(neurons_per_mouse.values()))),
        "paper_check_tracked_neurons_std": float(np.std(list(neurons_per_mouse.values()), ddof=1))
        if len(neurons_per_mouse) > 1
        else 0.0,
    }
    return data, summary


def print_custom_summary(data: dict, conversion_summary: dict) -> None:
    print("\nConversion summary:")
    print(f"  Mode: {conversion_summary['mode']}")
    print(f"  Sessions: {conversion_summary['nsessions']}")
    print(f"  Subjects: {', '.join(conversion_summary['subjects'])}")
    print(f"  Unique session durations (minutes, nominal from frame count): {conversion_summary['duration_minutes_unique']}")
    print(f"  Neurons per mouse: {conversion_summary['neurons_per_mouse']}")
    print(
        "  Paper sanity check, tracked neurons per mouse: "
        f"mean={conversion_summary['paper_check_tracked_neurons_mean']:.2f}, "
        f"std={conversion_summary['paper_check_tracked_neurons_std']:.2f}"
    )
    print(f"  Total repaired missing behavior frames: {conversion_summary['total_missing_behavior_frames']}")
    if conversion_summary["sessions_with_missing_behavior"]:
        print("  Sessions with repaired missing behavior frames:")
        for session_key, missing_count in conversion_summary["sessions_with_missing_behavior"].items():
            print(f"    {session_key}: {missing_count}")

    output_concat = np.concatenate(
        [trial[0] for session_trials in data["output"] for trial in session_trials]
    )
    unique, counts = np.unique(output_concat, return_counts=True)
    print("  Global output bin fractions:")
    for value, count in zip(unique, counts):
        print(f"    {data['output_values'][0][int(value)]}: {count / output_concat.size:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Track2p developmental imaging data for decoder training.")
    parser.add_argument(
        "--mode",
        choices=["full", "sample"],
        default="full",
        help="Convert all sessions or a smaller representative sample.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output pickle path. Defaults to converted_data.pkl for full and sample_data.pkl for sample.",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = Path("/app/converted_data.pkl" if args.mode == "full" else "/app/sample_data.pkl")

    data, conversion_summary = build_dataset(args.mode)

    print_custom_summary(data, conversion_summary)
    valid, errors, warnings = verify_data_format(data)
    if not valid:
        print("\nverify_data_format reported errors:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    if warnings:
        print("\nverify_data_format warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\nverify_data_format: no warnings")

    print("\nDecoder summary:")
    print_data_summary(data)

    with output_path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved dataset to {output_path}")


if __name__ == "__main__":
    main()
