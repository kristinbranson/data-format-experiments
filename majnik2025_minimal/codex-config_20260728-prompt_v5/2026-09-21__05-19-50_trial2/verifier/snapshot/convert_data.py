import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter1d, minimum_filter1d


DATA_ROOT = Path("/app/data")
OUTPUT_PATH = Path("/app/converted_data.pkl")

FS = 30.0
BIN_FRAMES = 10
TRIAL_SECONDS = 60
NEUCOEFF = 0.0
BASELINE = "maximin"
SIG_BASELINE = 10.0
WIN_BASELINE = 60.0


def suite2p_style_baseline_correct(F: np.ndarray, Fneu: np.ndarray, fs: float) -> np.ndarray:
    """Match the paper/repo preprocessing: neuropil coefficient 0 and maximin baseline subtraction."""
    Fc = F.astype(np.float32, copy=False) - NEUCOEFF * Fneu.astype(np.float32, copy=False)
    if BASELINE != "maximin":
        raise ValueError(f"Unsupported baseline mode: {BASELINE}")

    win = int(WIN_BASELINE * fs)
    Flow = gaussian_filter(Fc, [0.0, SIG_BASELINE])
    Flow = minimum_filter1d(Flow, win, axis=1)
    Flow = maximum_filter1d(Flow, win, axis=1)
    return (Fc - Flow).astype(np.float32, copy=False)


def repair_motion_trace(motion: np.ndarray, interframe_int: np.ndarray, target_len: int) -> np.ndarray:
    """Interpolate only the missing camera frames indicated by doubled inter-frame intervals."""
    motion = motion.astype(np.float32, copy=False)
    if len(motion) == target_len:
        return motion

    median_ifi = float(np.median(interframe_int))
    gap_sizes = np.rint(interframe_int / median_ifi).astype(int)
    if np.any(gap_sizes < 1):
        raise ValueError("Found invalid inter-frame interval ratio while repairing motion trace.")

    repaired = [float(motion[0])]
    for i, gap in enumerate(gap_sizes):
        if gap > 1:
            repaired.extend([np.nan] * (gap - 1))
        repaired.append(float(motion[i + 1]))

    repaired = np.asarray(repaired, dtype=np.float32)
    if repaired.size < target_len:
        repaired = np.pad(repaired, (0, target_len - repaired.size), constant_values=np.nan)
    elif repaired.size > target_len:
        repaired = repaired[:target_len]

    nan_mask = np.isnan(repaired)
    if nan_mask.any():
        valid_idx = np.flatnonzero(~nan_mask)
        repaired[nan_mask] = np.interp(np.flatnonzero(nan_mask), valid_idx, repaired[valid_idx]).astype(np.float32)

    if repaired.size != target_len:
        raise ValueError(f"Repaired motion length {repaired.size} does not match target length {target_len}.")
    return repaired.astype(np.float32, copy=False)


def mean_bin_time_series(x: np.ndarray, bin_frames: int) -> np.ndarray:
    if x.shape[-1] % bin_frames != 0:
        raise ValueError(f"Time axis length {x.shape[-1]} is not divisible by bin size {bin_frames}.")
    new_shape = x.shape[:-1] + (x.shape[-1] // bin_frames, bin_frames)
    return x.reshape(new_shape).mean(axis=-1)


def split_trials(x: np.ndarray, trial_bins: int) -> list[np.ndarray]:
    if x.shape[-1] % trial_bins != 0:
        raise ValueError(f"Time axis length {x.shape[-1]} is not divisible by trial length {trial_bins}.")
    ntrials = x.shape[-1] // trial_bins
    return [x[..., i * trial_bins:(i + 1) * trial_bins] for i in range(ntrials)]


def motion_to_quintiles(motion_binned: np.ndarray) -> np.ndarray:
    edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
    labels = np.digitize(motion_binned, edges, right=False).astype(np.int64)
    return labels[np.newaxis, :]


def main() -> None:
    subjects = sorted([p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("jm")])
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}

    neural_sessions = []
    input_sessions = []
    output_sessions = []
    subject_idx = []
    brain_region_idx = []
    session_info = []

    trial_bins = int(TRIAL_SECONDS * FS / BIN_FRAMES)
    time_bin_size_ms = 1000.0 * BIN_FRAMES / FS

    for subject in subjects:
        subject_dir = DATA_ROOT / subject
        session_dirs = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
        for session_dir in session_dirs:
            plane_dir = session_dir / "suite2p" / "plane0"
            move_dir = session_dir / "move_deve"

            F = np.load(plane_dir / "F.npy", allow_pickle=True)
            Fneu = np.load(plane_dir / "Fneu.npy", allow_pickle=True)
            iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
            ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
            motion = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
            interframe_int = np.load(move_dir / "interframe_int.npy", allow_pickle=True)

            nframes = int(ops["nframes"])
            fs = float(ops["fs"])
            if fs != FS:
                raise ValueError(f"Unexpected sampling rate {fs} in {session_dir}")
            if F.shape != Fneu.shape:
                raise ValueError(f"F/Fneu shape mismatch in {session_dir}")
            if F.shape[1] != nframes:
                raise ValueError(f"Neural frame count mismatch in {session_dir}")
            if not np.all(iscell[:, 0] == 1):
                raise ValueError(f"Found non-cell ROIs in tracked output for {session_dir}")
            if not np.all(iscell[:, 1] > 0.5):
                raise ValueError(f"Found tracked ROIs below the paper's iscell threshold in {session_dir}")

            neural = suite2p_style_baseline_correct(F, Fneu, fs)
            motion_aligned = repair_motion_trace(motion, interframe_int, nframes)

            neural_binned = mean_bin_time_series(neural, BIN_FRAMES).astype(np.float32, copy=False)
            motion_binned = mean_bin_time_series(motion_aligned[np.newaxis, :], BIN_FRAMES)[0].astype(np.float32, copy=False)

            frame_times = np.arange(nframes, dtype=np.float32) / fs
            time_binned = mean_bin_time_series(frame_times[np.newaxis, :], BIN_FRAMES)[0].astype(np.float32, copy=False)

            motion_labels = motion_to_quintiles(motion_binned)

            neural_trials = [trial.astype(np.float32, copy=False) for trial in split_trials(neural_binned, trial_bins)]
            input_trials = [trial.astype(np.float32, copy=False) for trial in split_trials(time_binned[np.newaxis, :], trial_bins)]
            output_trials = [trial.astype(np.int64, copy=False) for trial in split_trials(motion_labels, trial_bins)]

            neural_sessions.append(neural_trials)
            input_sessions.append(input_trials)
            output_sessions.append(output_trials)
            subject_idx.append(subject_to_idx[subject])
            brain_region_idx.append(np.zeros(neural.shape[0], dtype=np.int64))
            session_info.append(
                {
                    "subject": subject,
                    "session": session_dir.name,
                    "date": session_dir.name.split("_")[0],
                    "n_neurons": int(neural.shape[0]),
                    "n_frames_imaging": nframes,
                    "n_frames_motion_raw": int(len(motion)),
                    "n_frames_motion_missing": int(nframes - len(motion)),
                    "duration_sec": float(nframes / fs),
                    "n_trials": len(neural_trials),
                }
            )

    data = {
        "neural": neural_sessions,
        "input": input_sessions,
        "output": output_sessions,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": ["barrel cortex"],
        "brain_region_idx": brain_region_idx,
        "input_names": ["time_from_session_start_sec"],
        "output_names": ["motion_energy_quintile"],
        "output_values": [["lowest", "low", "medium", "high", "highest"]],
        "metadata": {
            "task_description": "Decode five-level motion-energy state from longitudinal barrel-cortex calcium imaging during spontaneous behavior.",
            "time_bin_size": float(time_bin_size_ms),
            "temporal_alignment_event": "session start",
            "off_start": None,
            "off_end": None,
            "neural_signal": "Suite2p-style baseline-corrected fluorescence computed from F and Fneu with neucoeff=0 and maximin baseline subtraction.",
            "behavior_signal": "Global motion energy from videography, repaired for missing camera frames by interpolation and averaged in 10-frame bins.",
            "binning_frames": BIN_FRAMES,
            "trial_duration_sec": TRIAL_SECONDS,
            "quantile_binning": "Per-session quintiles computed on the 10-frame-averaged motion energy trace.",
            "session_order": [info["session"] for info in session_info],
            "session_info": session_info,
        },
    }

    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(data, f)

    print(f"Saved converted dataset to {OUTPUT_PATH}")
    print(f"Sessions: {len(neural_sessions)}")
    print(f"Trials: {sum(len(session) for session in neural_sessions)}")


if __name__ == "__main__":
    main()
