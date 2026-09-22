from __future__ import annotations

import pickle
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


DATA_DIR = Path("/app/data/Ephys_Behavior")
OUTPUT_PATH = Path("/app/converted_data.pkl")

# Published DR+WC ALM-video cohort from the paper code:
# JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19 (12 sessions total).
SESSIONS = [
    {"subject": "JEB6", "date": "2021-04-18", "probe": 2},
    {"subject": "JEB7", "date": "2021-04-29", "probe": 1},
    {"subject": "JEB7", "date": "2021-04-30", "probe": 1},
    {"subject": "EKH1", "date": "2021-08-07", "probe": 2},
    {"subject": "EKH3", "date": "2021-08-11", "probe": 2},
    {"subject": "JGR2", "date": "2021-11-16", "probe": 1},
    {"subject": "JGR2", "date": "2021-11-17", "probe": 1},
    {"subject": "JGR3", "date": "2021-11-18", "probe": 1},
    {"subject": "JEB19", "date": "2023-04-21", "probe": 1},
    {"subject": "JEB19", "date": "2023-04-20", "probe": 1},
    {"subject": "JEB19", "date": "2023-04-19", "probe": 1},
    {"subject": "JEB19", "date": "2023-04-18", "probe": 1},
]

TMIN = -3.0
TMAX = 2.5
DT = 1.0 / 200.0
SMOOTH_N = 15
LOW_FR_HZ = 1.0

OUTPUT_NAMES = [
    "lick_direction",
    "behavioral_context",
    "outcome",
    "tongue_velocity",
    "paw_velocity",
    "motion_energy",
]

OUTPUT_VALUES = [
    ["left", "right", "none"],
    ["WC", "DR"],
    ["incorrect", "correct", "ignore"],
    ["low", "high", "not_visible"],
    ["low", "high", "not_visible"],
    ["low", "high", "no_video"],
]

INPUT_NAMES = ["time_from_go_cue"]


def matlab_class(obj: h5py.Group | h5py.Dataset) -> str:
    cls = obj.attrs.get("MATLAB_class", b"")
    if isinstance(cls, np.bytes_):
        cls = bytes(cls)
    if isinstance(cls, bytes):
        return cls.decode()
    return str(cls)


def decode_char(ds: h5py.Dataset) -> str:
    arr = np.array(ds[()])
    return "".join(chr(int(x)) for x in arr.squeeze())


def make_object_array(shape: tuple[int, ...], items: list[object]) -> np.ndarray:
    out = np.empty(shape, dtype=object)
    for idx, item in zip(np.ndindex(shape), items):
        out[idx] = item
    return out


def read_matlab_any(
    h5file: h5py.File, obj: h5py.Reference | h5py.Group | h5py.Dataset
) -> object:
    if isinstance(obj, h5py.Reference):
        if not obj:
            return None
        obj = h5file[obj]

    cls = matlab_class(obj)

    if isinstance(obj, h5py.Group):
        return {key: read_matlab_any(h5file, obj[key]) for key in obj.keys()}

    arr = obj[()]
    if cls == "char":
        return decode_char(obj)
    if cls == "canonical empty":
        return np.array([])
    if cls == "cell" or getattr(arr, "dtype", None) == object:
        return make_object_array(arr.shape, [read_matlab_any(h5file, ref) for ref in arr.flat])
    return np.array(arr)


def as_1d_float(x: object) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def as_bool_1d(x: object) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1).astype(bool)


def build_smoothing_kernel(n: int = SMOOTH_N) -> np.ndarray:
    x = np.arange(1, n + 1, dtype=np.float64)
    sigma = np.std(x)
    kernel = np.exp(-0.5 * ((x - (n + 1) / 2.0) / sigma) ** 2)
    kernel[: n // 2] = 0.0
    kernel /= kernel.sum()
    return kernel


SMOOTH_KERNEL = build_smoothing_kernel()
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_BINS = TIME_EDGES[:-1] + DT / 2.0
NT = TIME_BINS.size


def smooth_causal_reflect(x: np.ndarray, kernel: np.ndarray = SMOOTH_KERNEL) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
        squeeze = True
    else:
        squeeze = False
    n = kernel.size
    padded = np.concatenate([x[:, :n], x], axis=1)
    out = np.empty_like(padded)
    for i in range(padded.shape[0]):
        out[i] = np.convolve(padded[i], kernel, mode="same")
    out = out[:, n:]
    return out[0] if squeeze else out


def fill_nearest_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).copy()
    good = np.flatnonzero(np.isfinite(y))
    if good.size == 0:
        return y
    y[: good[0]] = y[good[0]]
    y[good[-1] + 1 :] = y[good[-1]]
    return y


def interpolate_visible_segments(
    src_t: np.ndarray, src_xy: np.ndarray, target_t: np.ndarray
) -> np.ndarray:
    out = np.full((target_t.size, src_xy.shape[1]), np.nan, dtype=np.float64)
    valid = np.all(np.isfinite(src_xy), axis=1) & np.isfinite(src_t)
    if not np.any(valid):
        return out

    valid_idx = np.flatnonzero(valid)
    split_idx = np.where(np.diff(valid_idx) > 1)[0] + 1
    groups = np.split(valid_idx, split_idx)

    for grp in groups:
        if grp.size == 1:
            nearest = int(np.argmin(np.abs(target_t - src_t[grp[0]])))
            out[nearest] = src_xy[grp[0]]
            continue

        seg_t = src_t[grp]
        seg_mask = (target_t >= seg_t[0]) & (target_t <= seg_t[-1])
        if not np.any(seg_mask):
            continue

        for dim in range(src_xy.shape[1]):
            out[seg_mask, dim] = np.interp(target_t[seg_mask], seg_t, src_xy[grp, dim])

    return out


def interp_motion(src_t: np.ndarray, src_y: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    valid = np.isfinite(src_t) & np.isfinite(src_y)
    if valid.sum() < 2:
        return np.full(target_t.shape, np.nan, dtype=np.float64)
    out = np.interp(target_t, src_t[valid], src_y[valid], left=np.nan, right=np.nan)
    return fill_nearest_1d(out)


def average_over_visible(values: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    any_visible = visible.any(axis=0)
    mean_values = np.full(values.shape[1], np.nan, dtype=np.float64)
    if np.any(any_visible):
        counts = visible[:, any_visible].sum(axis=0)
        summed = np.where(visible[:, any_visible], values[:, any_visible], 0.0).sum(axis=0)
        mean_values[any_visible] = summed / counts
    return mean_values, any_visible


def get_feature_positions(
    h5file: h5py.File,
    view_group: h5py.Group,
    trial_idx: int,
    feature_names: list[str],
    align_time: float,
    video_shift: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feat_names = np.asarray(read_matlab_any(h5file, view_group["featNames"][trial_idx, 0])).reshape(-1)
    names = [str(x) for x in feat_names.tolist()]
    ts = np.asarray(read_matlab_any(h5file, view_group["ts"][trial_idx, 0]), dtype=np.float64)
    # MATLAB arrays come through as feat x coord x time in h5py.
    ts = np.transpose(ts, (2, 1, 0))
    frame_times = as_1d_float(read_matlab_any(h5file, view_group["frameTimes"][trial_idx, 0]))
    frame_times = frame_times - video_shift - align_time

    all_speed = []
    all_visible = []
    for feature_name in feature_names:
        if feature_name not in names:
            continue
        feature_idx = names.index(feature_name)
        xy = ts[:, :2, feature_idx]
        interp_xy = interpolate_visible_segments(frame_times, xy, TIME_BINS)
        visible = np.all(np.isfinite(interp_xy), axis=1)

        filled_xy = interp_xy.copy()
        filled_xy[:, 0] = fill_nearest_1d(filled_xy[:, 0])
        filled_xy[:, 1] = fill_nearest_1d(filled_xy[:, 1])

        if np.all(~np.isfinite(filled_xy)):
            speed = np.full(TIME_BINS.shape, np.nan, dtype=np.float64)
        else:
            vel = np.gradient(filled_xy, axis=0)
            baseline_deriv = np.nanmedian(np.diff(filled_xy, axis=0), axis=0)
            if "tongue" not in feature_name:
                vel[:, 0] = vel[:, 0] - baseline_deriv[0]
                vel[:, 1] = vel[:, 1] - baseline_deriv[0]
            speed = np.sqrt((vel**2).sum(axis=1))
        all_speed.append(speed)
        all_visible.append(visible)

    if not all_speed:
        return (
            np.full(TIME_BINS.shape, np.nan, dtype=np.float64),
            np.zeros(TIME_BINS.shape, dtype=bool),
            np.zeros(TIME_BINS.shape, dtype=bool),
        )

    speed_stack = np.stack(all_speed, axis=0)
    visible_stack = np.stack(all_visible, axis=0)
    mean_speed, any_visible = average_over_visible(speed_stack, visible_stack)
    return mean_speed, any_visible, np.ones(TIME_BINS.shape, dtype=bool)


def get_video_shift(h5file: h5py.File, bit_start: np.ndarray) -> float:
    bitcode_start = as_1d_float(h5file["obj/sglx/bitcode/bitstart"])
    fs = float(np.asarray(h5file["obj/sglx/fs"]).squeeze())
    return float(np.nanmedian(bitcode_start / fs) - np.nanmedian(bit_start))


def compute_neural_session(
    h5file: h5py.File,
    probe_num: int,
    keep_trials: np.ndarray,
    go_cue: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    trial_lookup = np.full(go_cue.size, -1, dtype=np.int32)
    trial_lookup[keep_trials] = np.arange(keep_trials.size, dtype=np.int32)

    clu_group = h5file[h5file["obj/clu"][probe_num - 1, 0]]
    qualities = clu_group["quality"]
    trials = clu_group["trial"]
    trial_times = clu_group["trialtm"]

    kept_unit_data: list[np.ndarray] = []

    for clu_idx in range(qualities.shape[0]):
        quality = str(read_matlab_any(h5file, qualities[clu_idx, 0]) or "").strip()
        if quality in {"garbage", "gabrga", "noisy", "real?"}:
            continue

        clu_trials = as_1d_float(read_matlab_any(h5file, trials[clu_idx, 0])).astype(np.int64) - 1
        clu_trial_times = as_1d_float(read_matlab_any(h5file, trial_times[clu_idx, 0]))
        aligned_times = clu_trial_times - go_cue[clu_trials]

        mapped_trials = trial_lookup[clu_trials]
        valid = (mapped_trials >= 0) & (aligned_times >= TMIN) & (aligned_times < TMAX)
        mapped_trials = mapped_trials[valid]
        aligned_times = aligned_times[valid]

        bin_idx = np.floor((aligned_times - TMIN) / DT).astype(np.int32)
        counts = np.zeros((keep_trials.size, NT), dtype=np.float64)
        if mapped_trials.size:
            np.add.at(counts, (mapped_trials, bin_idx), 1.0)

        rates = smooth_causal_reflect(counts / DT).astype(np.float32)
        mean_fr = float(rates.mean())
        if mean_fr > LOW_FR_HZ:
            kept_unit_data.append(rates)

    if len(kept_unit_data) < 10:
        raise RuntimeError(f"Session failed unit inclusion: only {len(kept_unit_data)} units after filtering.")

    stacked = np.stack(kept_unit_data, axis=1)  # trials x units x time
    trials_out = [stacked[trial_idx].astype(np.float32) for trial_idx in range(stacked.shape[0])]
    brain_region_idx = np.zeros(stacked.shape[1], dtype=np.int64)
    return trials_out, brain_region_idx


def compute_behavior_labels(
    bp: h5py.Group, keep_trials: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = as_bool_1d(bp["R"])
    l = as_bool_1d(bp["L"])
    hit = as_bool_1d(bp["hit"])
    miss = as_bool_1d(bp["miss"])
    no = as_bool_1d(bp["no"])
    autowater = as_bool_1d(bp["autowater"])

    lick_direction = np.full(keep_trials.size, 2, dtype=np.int64)
    right_choice = (r & hit) | (l & miss)
    left_choice = (l & hit) | (r & miss)
    lick_direction[left_choice[keep_trials]] = 0
    lick_direction[right_choice[keep_trials]] = 1

    context = np.where(autowater[keep_trials], 0, 1).astype(np.int64)  # WC, DR

    outcome = np.full(keep_trials.size, 2, dtype=np.int64)
    outcome[miss[keep_trials]] = 0
    outcome[hit[keep_trials]] = 1

    return lick_direction, context, outcome


def compute_video_outputs(
    session_info: dict[str, object],
    keep_trials: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    subject = str(session_info["subject"])
    date = str(session_info["date"])
    data_path = DATA_DIR / f"data_structure_{subject}_{date}.mat"
    me_path = DATA_DIR / f"motionEnergy_{subject}_{date}.mat"

    with h5py.File(data_path, "r") as h5file:
        bp = h5file["obj/bp"]
        go_cue = as_1d_float(bp["ev/goCue"])
        bit_start = as_1d_float(bp["ev/bitStart"])
        video_shift = get_video_shift(h5file, bit_start)

        side_view = h5file[h5file["obj/traj"][0, 0]]
        bottom_view = h5file[h5file["obj/traj"][1, 0]]

        tongue_speed = np.full((keep_trials.size, NT), np.nan, dtype=np.float64)
        tongue_visible = np.zeros((keep_trials.size, NT), dtype=bool)
        paw_speed = np.full((keep_trials.size, NT), np.nan, dtype=np.float64)
        paw_visible = np.zeros((keep_trials.size, NT), dtype=bool)

        me_raw = loadmat(me_path, simplify_cells=True)["me"]["data"]
        motion_energy = np.full((keep_trials.size, NT), np.nan, dtype=np.float64)
        motion_available = np.zeros((keep_trials.size, NT), dtype=bool)

        for out_trial_idx, trial_idx in enumerate(keep_trials):
            align_time = float(go_cue[trial_idx])

            side_tongue_speed, side_tongue_visible, _ = get_feature_positions(
                h5file,
                side_view,
                int(trial_idx),
                ["tongue", "left_tongue", "right_tongue"],
                align_time,
                video_shift,
            )
            bottom_tongue_speed, bottom_tongue_visible, _ = get_feature_positions(
                h5file,
                bottom_view,
                int(trial_idx),
                ["top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue"],
                align_time,
                video_shift,
            )

            tongue_speeds = np.stack([side_tongue_speed, bottom_tongue_speed], axis=0)
            tongue_vis = np.stack([side_tongue_visible, bottom_tongue_visible], axis=0)
            tongue_speed[out_trial_idx], tongue_visible[out_trial_idx] = average_over_visible(
                tongue_speeds, tongue_vis
            )

            paw_speed_trial, paw_visible_trial, _ = get_feature_positions(
                h5file,
                bottom_view,
                int(trial_idx),
                ["top_paw", "bottom_paw"],
                align_time,
                video_shift,
            )
            paw_speed[out_trial_idx] = paw_speed_trial
            paw_visible[out_trial_idx] = paw_visible_trial

            frame_times = as_1d_float(read_matlab_any(h5file, side_view["frameTimes"][trial_idx, 0]))
            frame_times = frame_times - video_shift - align_time
            me_trial = np.asarray(me_raw[trial_idx], dtype=np.float64).reshape(-1)
            if (
                frame_times.size != me_trial.size
                or frame_times.size < 2
                or np.isfinite(frame_times).sum() < 2
            ):
                side_ts = np.asarray(read_matlab_any(h5file, side_view["ts"][trial_idx, 0]), dtype=np.float64)
                side_ts = np.transpose(side_ts, (2, 1, 0))
                fallback_times = np.arange(1, side_ts.shape[0] + 1, dtype=np.float64) / 400.0
                fallback_times = fallback_times - 0.5 - align_time
                if fallback_times.size == me_trial.size and fallback_times.size >= 2:
                    frame_times = fallback_times

            if frame_times.size == me_trial.size and frame_times.size >= 2:
                aligned_me = interp_motion(frame_times, me_trial, TIME_BINS)
                available = np.isfinite(aligned_me)
                motion_energy[out_trial_idx] = aligned_me
                motion_available[out_trial_idx] = available

    tongue_threshold = float(np.nanpercentile(tongue_speed[tongue_visible], 50))
    paw_threshold = float(np.nanpercentile(paw_speed[paw_visible], 50))
    me_threshold = float(np.nanpercentile(motion_energy[motion_available], 50))

    tongue_disc = np.full((keep_trials.size, NT), 2, dtype=np.int64)
    tongue_disc[tongue_visible & (tongue_speed < tongue_threshold)] = 0
    tongue_disc[tongue_visible & (tongue_speed >= tongue_threshold)] = 1

    paw_disc = np.full((keep_trials.size, NT), 2, dtype=np.int64)
    paw_disc[paw_visible & (paw_speed < paw_threshold)] = 0
    paw_disc[paw_visible & (paw_speed >= paw_threshold)] = 1

    me_disc = np.full((keep_trials.size, NT), 2, dtype=np.int64)
    me_disc[motion_available & (motion_energy < me_threshold)] = 0
    me_disc[motion_available & (motion_energy >= me_threshold)] = 1

    thresholds = {
        "tongue_velocity_p50": tongue_threshold,
        "paw_velocity_p50": paw_threshold,
        "motion_energy_p50": me_threshold,
    }
    return tongue_disc, paw_disc, me_disc, thresholds


def build_trial_outputs(
    lick_direction: np.ndarray,
    context: np.ndarray,
    outcome: np.ndarray,
    tongue_disc: np.ndarray,
    paw_disc: np.ndarray,
    me_disc: np.ndarray,
) -> list[np.ndarray]:
    outputs = []
    for trial_idx in range(lick_direction.size):
        outputs.append(
            np.vstack(
                [
                    np.full(NT, lick_direction[trial_idx], dtype=np.int64),
                    np.full(NT, context[trial_idx], dtype=np.int64),
                    np.full(NT, outcome[trial_idx], dtype=np.int64),
                    tongue_disc[trial_idx],
                    paw_disc[trial_idx],
                    me_disc[trial_idx],
                ]
            )
        )
    return outputs


def convert_session(session_info: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    subject = str(session_info["subject"])
    date = str(session_info["date"])
    probe_num = int(session_info["probe"])
    data_path = DATA_DIR / f"data_structure_{subject}_{date}.mat"

    with h5py.File(data_path, "r") as h5file:
        bp = h5file["obj/bp"]
        stim_enable = as_bool_1d(bp["stim/enable"])
        early = as_bool_1d(bp["early"])
        autolearn = (
            as_bool_1d(bp["autolearn"]) if "autolearn" in bp.keys() else np.zeros_like(early, dtype=bool)
        )
        keep_mask = (~stim_enable) & (~early) & (~autolearn)
        keep_trials = np.flatnonzero(keep_mask)
        if keep_trials.size < 2:
            raise RuntimeError(f"{subject} {date} has fewer than two usable trials.")

        go_cue = as_1d_float(bp["ev/goCue"])
        neural_trials, brain_region_idx = compute_neural_session(h5file, probe_num, keep_trials, go_cue)
        lick_direction, context, outcome = compute_behavior_labels(bp, keep_trials)

    tongue_disc, paw_disc, me_disc, thresholds = compute_video_outputs(session_info, keep_trials)
    output_trials = build_trial_outputs(lick_direction, context, outcome, tongue_disc, paw_disc, me_disc)
    input_trials = [TIME_BINS[np.newaxis, :].astype(np.float32) for _ in range(keep_trials.size)]

    session_meta = {
        "subject": subject,
        "date": date,
        "probe": probe_num,
        "n_trials_total": int(keep_mask.size),
        "n_trials_used": int(keep_trials.size),
        "trial_filter": "~stim.enable & ~early & ~autolearn",
        "thresholds": thresholds,
        "n_neurons": int(brain_region_idx.size),
    }

    session_data = {
        "neural": neural_trials,
        "input": input_trials,
        "output": output_trials,
        "brain_region_idx": brain_region_idx,
    }
    return session_data, session_meta


def main() -> None:
    subjects: list[str] = []
    subject_to_idx: dict[str, int] = {}

    all_neural: list[list[np.ndarray]] = []
    all_input: list[list[np.ndarray]] = []
    all_output: list[list[np.ndarray]] = []
    all_subject_idx: list[int] = []
    all_brain_region_idx: list[np.ndarray] = []
    session_metadata: list[dict[str, object]] = []

    for session_info in SESSIONS:
        session_data, session_meta = convert_session(session_info)
        subject = str(session_meta["subject"])
        if subject not in subject_to_idx:
            subject_to_idx[subject] = len(subjects)
            subjects.append(subject)

        all_neural.append(session_data["neural"])
        all_input.append(session_data["input"])
        all_output.append(session_data["output"])
        all_subject_idx.append(subject_to_idx[subject])
        all_brain_region_idx.append(session_data["brain_region_idx"])
        session_metadata.append(session_meta)

    data = {
        "neural": all_neural,
        "input": all_input,
        "output": all_output,
        "subjects": subjects,
        "subject_idx": np.asarray(all_subject_idx, dtype=np.int64),
        "brain_regions": ["ALM"],
        "brain_region_idx": all_brain_region_idx,
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "output_values": OUTPUT_VALUES,
        "metadata": {
            "task_description": (
                "Alternating delayed-response (DR) and water-cued (WC) directional licking "
                "task with ALM recordings, aligned to go cue / water-drop time."
            ),
            "time_bin_size": float(DT),
            "time_bin_size_ms": float(DT * 1000.0),
            "temporal_alignment_event": "go cue onset",
            "off_start": float(TMIN),
            "off_end": float(TMAX),
            "neural_preprocessing": (
                "Published DR+WC ALM-video cohort, garbage/noisy clusters removed, "
                "goCue alignment, 5 ms bins, causal Gaussian smoothing (N=15) with reflected "
                "leading padding, units retained if mean firing rate > 1 Hz."
            ),
            "video_preprocessing": (
                "Video trajectories aligned with the paper's video offset, feature positions "
                "linearly interpolated onto the neural grid, visibility preserved for tongue/paw "
                "categorization, and motion energy aligned as in the reference code."
            ),
            "trial_filtering": "~stim.enable & ~early & ~autolearn",
            "session_info": session_metadata,
        },
    }

    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(data, f)

    print(f"Saved converted data to {OUTPUT_PATH}")
    print(f"Sessions: {len(all_neural)}")
    print(f"Trials: {sum(len(sess) for sess in all_neural)}")
    print(f"Neurons: {sum(len(idx) for idx in all_brain_region_idx)}")


if __name__ == "__main__":
    main()
