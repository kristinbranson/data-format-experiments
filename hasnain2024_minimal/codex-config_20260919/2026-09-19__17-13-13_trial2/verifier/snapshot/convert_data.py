#!/usr/bin/env python3
"""Convert Hasnain & Birnbaum et al. data for the supplied decoder.

The conversion follows the two-context (DR/WC) Figure 8 analysis pipeline:
12 electrophysiology sessions, go-cue alignment, -3 to 2.5 s at 100 Hz,
causal 15-bin Gaussian smoothing, the repository's cluster-quality rules,
and its >1 Hz inclusion threshold.  Control, non-early trials are retained;
unlike the paper's analyses, ignore trials are retained because they are an
explicit decoder target in this task.

MATLAB v7.3 files are read directly with h5py so this script does not require
MATLAB or a third-party MAT-file compatibility package.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np
from scipy import interpolate, io, signal
from scipy.signal.windows import gaussian


TMIN = -3.0
TMAX = 2.5
DT = 0.01
SMOOTH_BINS = 15
LOW_FR_HZ = 1.0

# Session order and ALM probe numbers are taken from the Figure 8 scripts and
# the corresponding load<animal>_ALMVideo.m files.
SESSIONS = (
    ("JEB6", "2021-04-18", 2),
    ("JEB7", "2021-04-29", 1),
    ("JEB7", "2021-04-30", 1),
    ("EKH1", "2021-08-07", 2),
    ("EKH3", "2021-08-11", 2),
    ("JGR2", "2021-11-16", 1),
    ("JGR2", "2021-11-17", 1),
    ("JGR3", "2021-11-18", 1),
    ("JEB19", "2023-04-21", 1),
    ("JEB19", "2023-04-20", 1),
    ("JEB19", "2023-04-19", 1),
    ("JEB19", "2023-04-18", 1),
)


def _array(dataset: h5py.Dataset) -> np.ndarray:
    """Read a MATLAB numeric array (HDF5 dimensions are reversed)."""
    value = np.asarray(dataset)
    if value.ndim > 1:
        value = value.transpose(tuple(range(value.ndim - 1, -1, -1)))
    return np.squeeze(value)


def _deref(handle: h5py.File, refs: h5py.Dataset, index: int):
    return handle[refs[...].flat[index]]


def _string(dataset: h5py.Dataset) -> str:
    chars = np.asarray(dataset).ravel(order="F")
    return "".join(chr(int(char)) for char in chars).rstrip("\x00").strip()


def _cell_strings(handle: h5py.File, cell: h5py.Dataset) -> list[str]:
    return [_string(handle[ref]) for ref in cell[...].flat]


def _causal_smooth(rates: np.ndarray) -> np.ndarray:
    """Match utils/mySmooth.m with N=15 and boundary type 'reflect'."""
    # MATLAB gausswin(N) uses alpha=2.5, equivalent to std=N/(2*alpha).
    kernel = gaussian(SMOOTH_BINS, std=SMOOTH_BINS / 5.0)
    kernel[: SMOOTH_BINS // 2] = 0.0  # causal operation in mySmooth.m
    kernel /= kernel.sum()

    # Despite its name, the repository's 'reflect' option prepends a copy of
    # the first N bins. Convolution is along time (axis 1) for all trials.
    padded = np.concatenate((rates[:, :SMOOTH_BINS], rates), axis=1)
    smoothed = signal.convolve(padded, kernel[None, :], mode="same")
    return smoothed[:, SMOOTH_BINS:]


def _nearest_fill(values: np.ndarray) -> np.ndarray:
    """Match MATLAB fillmissing(..., 'nearest') for a one-dimensional trace."""
    values = np.asarray(values, dtype=np.float64).copy()
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return values
    missing = np.flatnonzero(~np.isfinite(values))
    if len(missing):
        nearest = np.searchsorted(valid, missing)
        left = valid[np.maximum(nearest - 1, 0)]
        right = valid[np.minimum(nearest, len(valid) - 1)]
        choose_right = np.abs(right - missing) < np.abs(missing - left)
        source = np.where(choose_right, right, left)
        values[missing] = values[source]
    return values


def _interp_trace(source_time: np.ndarray, values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    """Linear interpolation with NaN outside video coverage, like interp1."""
    source_time = np.asarray(source_time, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = min(len(source_time), len(values))
    if n < 2:
        return np.full(target_time.shape, np.nan, dtype=np.float64)
    source_time, values = source_time[:n], values[:n]
    # interp1 propagates missing tracking through intervals touching a NaN.
    fun = interpolate.interp1d(
        source_time,
        values,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )
    return np.asarray(fun(target_time), dtype=np.float64)


def _load_behavior(handle: h5py.File) -> dict[str, np.ndarray]:
    bp = handle["obj/bp"]
    result = {
        name: _array(bp[name]).astype(bool)
        for name in ("R", "L", "hit", "miss", "no", "early", "autowater")
    }
    result["stim"] = _array(bp["stim/enable"]).astype(bool)
    result["go_cue"] = _array(bp["ev/goCue"]).astype(np.float64)
    result["ntrials"] = np.asarray([len(result["go_cue"])], dtype=np.int64)
    return result


def _figure8_conditions(behavior: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Conditions used before Figure 8's low-firing-rate unit filter."""
    hit, miss, no = behavior["hit"], behavior["miss"], behavior["no"]
    stim, wc, early = behavior["stim"], behavior["autowater"], behavior["early"]
    return [
        hit | miss | no,
        hit & ~stim & ~wc,
        hit & ~stim & wc,
        miss & ~stim & ~wc,
        miss & ~stim & wc,
        hit & ~stim & ~wc & ~early,
        hit & ~stim & wc & ~early,
    ]


def _load_neural(
    handle: h5py.File,
    probe: int,
    behavior: dict[str, np.ndarray],
    edges: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Return all paper-curated neurons as (trial, neuron, time)."""
    ntrials = int(behavior["ntrials"][0])
    go_cue = behavior["go_cue"]
    probe_group = _deref(handle, handle["obj/clu"], probe - 1)
    exclusions = {"garbage", "gabrga", "noisy", "real?"}
    conditions = _figure8_conditions(behavior)
    neurons: list[np.ndarray] = []
    qualities: list[str] = []

    for unit in range(probe_group["quality"].shape[0]):
        quality = _string(_deref(handle, probe_group["quality"], unit))
        # Match findClusters.m exactly: labels are stripped but case-sensitive.
        if quality in exclusions:
            continue

        spike_trial = _array(_deref(handle, probe_group["trial"], unit)).astype(np.int64)
        spike_time = _array(_deref(handle, probe_group["trialtm"], unit)).astype(np.float64)
        aligned = spike_time - go_cue[spike_trial - 1]
        bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
        valid = (bin_index >= 0) & (bin_index < len(edges) - 1)

        counts = np.zeros((ntrials, len(edges) - 1), dtype=np.float64)
        np.add.at(counts, (spike_trial[valid] - 1, bin_index[valid]), 1.0)
        rates = _causal_smooth(counts / DT)

        # removeLowFRClusters.m averages condition PSTHs equally (rather than
        # weighting conditions by their trial counts) and ignores empty cells.
        condition_means = [rates[mask].mean(axis=0) for mask in conditions if np.any(mask)]
        mean_rate = float(np.mean(np.stack(condition_means)))
        if mean_rate > LOW_FR_HZ:
            neurons.append(rates.astype(np.float32))
            qualities.append(quality)

    if not neurons:
        raise RuntimeError("No neurons survived curation")
    return np.stack(neurons, axis=1), qualities


def _video_group(handle: h5py.File, camera: int) -> h5py.Group:
    return _deref(handle, handle["obj/traj"], camera)


def _feature_index(handle: h5py.File, group: h5py.Group, feature: str) -> int:
    names_cell = _deref(handle, group["featNames"], 0)
    names = _cell_strings(handle, names_cell)
    try:
        return names.index(feature)
    except ValueError as exc:
        raise RuntimeError(f"DLC feature {feature!r} not found; available: {names}") from exc


def _video_offset(handle: h5py.File, behavior: dict[str, np.ndarray]) -> float:
    bit_start = _array(handle["obj/bp/ev/bitStart"]).astype(float)
    video_bit_start = _array(handle["obj/sglx/bitcode/bitstart"]).astype(float)
    sampling_rate = float(_array(handle["obj/sglx/fs"]))
    # All values are effectively constant; median is the robust equivalent of
    # MATLAB mode here and avoids floating-point uniqueness issues.
    return float(np.median(video_bit_start) / sampling_rate - np.median(bit_start))


def _load_velocity(
    handle: h5py.File,
    behavior: dict[str, np.ndarray],
    target_time: np.ndarray,
    camera: int,
    feature: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load speed and the pre-fill DLC visibility mask for every trial/bin."""
    group = _video_group(handle, camera)
    feature_index = _feature_index(handle, group, feature)
    ntrials, ntime = len(behavior["go_cue"]), len(target_time)
    speed = np.full((ntrials, ntime), np.nan, dtype=np.float64)
    visible = np.zeros((ntrials, ntime), dtype=bool)
    offset = _video_offset(handle, behavior)

    for trial in range(ntrials):
        dropped = _array(_deref(handle, group["NdroppedFrames"], trial))
        if np.size(dropped) and not np.all(np.isfinite(dropped)):
            continue
        frames = _array(_deref(handle, group["frameTimes"], trial)).astype(np.float64).reshape(-1)
        tracking = _array(_deref(handle, group["ts"], trial)).astype(np.float64)
        if tracking.ndim != 3 or feature_index >= tracking.shape[2]:
            continue
        aligned_frames = frames - offset - behavior["go_cue"][trial]
        xpos = _interp_trace(aligned_frames, tracking[:, 0, feature_index], target_time)
        ypos = _interp_trace(aligned_frames, tracking[:, 1, feature_index], target_time)
        visible[trial] = np.isfinite(xpos) & np.isfinite(ypos)

        # The paper fills non-tongue coordinates with the nearest observation
        # before differentiating. Tongue gaps remain gaps; invalid velocity is
        # later represented by the explicit not-visible class.
        if "tongue" not in feature:
            xpos, ypos = _nearest_fill(xpos), _nearest_fill(ypos)
        xvel, yvel = np.gradient(xpos), np.gradient(ypos)
        if "tongue" not in feature:
            # Remove slow tracking drift, following findVelocity.m. Subtracting
            # each coordinate's own median derivative fixes an apparent y/x typo
            # in that helper without changing visibility or threshold semantics.
            if np.any(np.isfinite(xvel)):
                xvel -= np.nanmedian(np.diff(xpos))
            if np.any(np.isfinite(yvel)):
                yvel -= np.nanmedian(np.diff(ypos))
        speed[trial] = np.hypot(xvel, yvel)

    return speed, visible


def _load_motion_energy(
    path: Path,
    handle: h5py.File,
    behavior: dict[str, np.ndarray],
    target_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mat = io.loadmat(path, simplify_cells=True, variable_names=["me"])
    raw = mat["me"]["data"]
    raw_trials = list(raw) if isinstance(raw, np.ndarray) and raw.dtype == object else [raw]
    group = _video_group(handle, 0)
    offset = _video_offset(handle, behavior)
    ntrials, ntime = len(behavior["go_cue"]), len(target_time)
    values = np.full((ntrials, ntime), np.nan, dtype=np.float64)
    available = np.zeros((ntrials, ntime), dtype=bool)

    for trial in range(min(ntrials, len(raw_trials))):
        dropped = _array(_deref(handle, group["NdroppedFrames"], trial))
        if np.size(dropped) and not np.all(np.isfinite(dropped)):
            continue
        frames = _array(_deref(handle, group["frameTimes"], trial)).astype(np.float64).reshape(-1)
        aligned_frames = frames - offset - behavior["go_cue"][trial]
        trace = np.asarray(raw_trials[trial], dtype=np.float64).reshape(-1)
        values[trial] = _interp_trace(aligned_frames, trace, target_time)
        available[trial] = np.isfinite(values[trial])
    return values, available


def _median_discretize(values: np.ndarray, available: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, float]:
    """Map below/equal-above session median to 0/1 and unavailable to 2."""
    usable = keep[:, None] & available & np.isfinite(values)
    if not np.any(usable):
        return np.full(values.shape, 2, dtype=np.int8), float("nan")
    threshold = float(np.percentile(values[usable], 50))
    output = np.full(values.shape, 2, dtype=np.int8)
    output[usable & (values < threshold)] = 0
    output[usable & (values >= threshold)] = 1
    return output, threshold


def convert(data_dir: Path, output_path: Path) -> dict:
    edges = np.arange(TMIN, TMAX + DT / 2, DT, dtype=np.float64)
    time = (edges[:-1] + DT / 2).astype(np.float32)
    subjects = list(dict.fromkeys(session[0] for session in SESSIONS))

    neural_sessions: list[list[np.ndarray]] = []
    input_sessions: list[list[np.ndarray]] = []
    output_sessions: list[list[np.ndarray]] = []
    region_indices: list[np.ndarray] = []
    session_info: list[dict] = []

    for session_index, (animal, date, probe) in enumerate(SESSIONS, start=1):
        stem = f"{animal}_{date}"
        data_path = data_dir / f"data_structure_{stem}.mat"
        motion_path = data_dir / f"motionEnergy_{stem}.mat"
        print(f"[{session_index:02d}/{len(SESSIONS)}] {stem} (ALM probe {probe})", flush=True)
        if not data_path.exists() or not motion_path.exists():
            raise FileNotFoundError(f"Missing source data for {stem}")

        with h5py.File(data_path, "r") as handle:
            behavior = _load_behavior(handle)
            neural_all, qualities = _load_neural(handle, probe, behavior, edges)

            # Paper analyses omit early licks and use no-photostimulation trials.
            # Ignore trials are deliberately retained for the required output.
            keep = ~behavior["early"] & ~behavior["stim"]

            # Primary side-view tongue and top paw are repository-standard
            # features (Figure 1 uses top_paw_yvel_view2).
            tongue_speed, tongue_visible = _load_velocity(
                handle, behavior, time, camera=0, feature="tongue"
            )
            paw_speed, paw_visible = _load_velocity(
                handle, behavior, time, camera=1, feature="top_paw"
            )
            motion_energy, motion_available = _load_motion_energy(
                motion_path, handle, behavior, time
            )

        tongue_class, tongue_threshold = _median_discretize(tongue_speed, tongue_visible, keep)
        paw_class, paw_threshold = _median_discretize(paw_speed, paw_visible, keep)
        motion_class, motion_threshold = _median_discretize(motion_energy, motion_available, keep)

        # getPrevChoice.m defines right choice as R-hit or L-miss. An ignore
        # trial has no choice; all remaining response trials are left choice.
        lick = np.zeros(len(keep), dtype=np.int8)  # left
        lick[(behavior["R"] & behavior["hit"]) | (behavior["L"] & behavior["miss"])] = 1
        lick[behavior["no"]] = 2
        context = behavior["autowater"].astype(np.int8)  # 0 DR, 1 WC
        outcome = np.zeros(len(keep), dtype=np.int8)  # incorrect
        outcome[behavior["hit"]] = 1
        outcome[behavior["no"]] = 2

        trial_ids = np.flatnonzero(keep)
        neural_trials: list[np.ndarray] = []
        input_trials: list[np.ndarray] = []
        output_trials: list[np.ndarray] = []
        for trial in trial_ids:
            neural_trials.append(neural_all[trial].copy())
            # Time is the sole decoder input and is continuous/time-varying.
            input_trials.append(time[None, :].copy())
            trial_output = np.empty((6, len(time)), dtype=np.int8)
            trial_output[0] = lick[trial]
            trial_output[1] = context[trial]
            trial_output[2] = outcome[trial]
            trial_output[3] = tongue_class[trial]
            trial_output[4] = paw_class[trial]
            trial_output[5] = motion_class[trial]
            output_trials.append(trial_output)

        neural_sessions.append(neural_trials)
        input_sessions.append(input_trials)
        output_sessions.append(output_trials)
        region_indices.append(np.zeros(neural_all.shape[1], dtype=np.int64))
        session_info.append(
            {
                "subject": animal,
                "date": date,
                "source_file": data_path.name,
                "alm_probe": probe,
                "n_source_trials": int(len(keep)),
                "n_included_trials": int(len(trial_ids)),
                "n_neurons": int(neural_all.shape[1]),
                "unit_quality_counts": {
                    quality: qualities.count(quality) for quality in sorted(set(qualities))
                },
                "tongue_velocity_median": tongue_threshold,
                "paw_velocity_median": paw_threshold,
                "motion_energy_median": motion_threshold,
            }
        )
        print(
            f"    kept {len(trial_ids)}/{len(keep)} trials, {neural_all.shape[1]} units",
            flush=True,
        )

    data = {
        "neural": neural_sessions,
        "input": input_sessions,
        "output": output_sessions,
        "subjects": subjects,
        "subject_idx": np.asarray(
            [subjects.index(animal) for animal, _, _ in SESSIONS], dtype=np.int64
        ),
        "brain_regions": ["ALM"],
        "brain_region_idx": region_indices,
        "input_names": ["time from go cue onset (s)"],
        "output_names": [
            "lick direction",
            "behavioral context",
            "outcome",
            "tongue velocity",
            "paw velocity",
            "motion energy",
        ],
        "output_values": [
            ["left", "right", "none"],
            ["DR", "WC"],
            ["incorrect", "correct", "ignore"],
            ["< session median", ">= session median", "not visible"],
            ["< session median", ">= session median", "not visible"],
            ["< session median", ">= session median", "no video"],
        ],
        "metadata": {
            "task_description": (
                "Two-context directional licking: auditory delayed-response (DR) and "
                "water-cued (WC) trials; decode choice, context, outcome, and movement."
            ),
            "time_bin_size": DT * 1000.0,
            "temporal_alignment_event": "go cue onset (DR) or matched water-delivery/go-cue event (WC)",
            "off_start": TMIN,
            "off_end": TMAX,
            # A plain list keeps the optional --stats-json path in the supplied
            # validator JSON-serializable.
            "time_bin_centers_s": time.tolist(),
            "neural_representation": (
                "Spike rate (Hz), 10 ms bins, causal 15-bin Gaussian smoothing with "
                "the repository's reflect boundary handling"
            ),
            "trial_filter": "no photostimulation and no early lick; ignore trials retained",
            "neuron_filter": (
                "Figure 8 ALM probe; repository quality exclusions and condition-averaged "
                "firing rate > 1 Hz"
            ),
            "velocity_definition": (
                "Euclidean speed from x/y DLC coordinates; side-view tongue and "
                "bottom-view top paw; categorical thresholds pool visible samples per session"
            ),
            "session_info": session_info,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {output_path} ({output_path.stat().st_size / 2**20:.1f} MiB)")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "Ephys_Behavior",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "converted_data.pkl",
    )
    args = parser.parse_args()
    convert(args.data_dir, args.output)


if __name__ == "__main__":
    main()
