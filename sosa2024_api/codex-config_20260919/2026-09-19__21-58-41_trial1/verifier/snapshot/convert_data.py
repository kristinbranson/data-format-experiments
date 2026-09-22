#!/usr/bin/env python3
"""Convert Sosa et al. NWB sessions to the neural-decoder pickle format.

All NWB access uses pynwb. Neural events are recomputed from fluorescence and
neuropil using the processing in the paper's reference repository.
"""

from __future__ import annotations

import argparse
import gc
import pickle
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pynwb import NWBHDF5IO
from scipy import ndimage
from suite2p.extraction import dcnv


DATA_ROOT = Path("/app/data")
FRAME_RATE = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE
NEUROPIL_COEF = 0.7
OASIS_TAU_S = 0.7
LICK_ERROR_FRACTION = 0.30
INTERNEURON_R_THRESHOLD = 0.5
REWARD_ZONES = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
ZONE_CODES = {"A": 0, "B": 1, "C": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="process all sessions (default)")
    mode.add_argument("--sample", action="store_true", help="process two representative sessions")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="save processing_<session_id>.png for up to two sessions",
    )
    return parser.parse_args()


def discover_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.glob("sub-*/*.nwb"))
    if not files:
        raise FileNotFoundError(f"No NWB files found under {DATA_ROOT}")
    if not sample:
        return files

    # Exercise single-plane, dual-plane, reward-switch, and environment-switch paths.
    preferred = [
        DATA_ROOT / "sub-m11/sub-m11_ses-03_behavior+ophys.nwb",
        DATA_ROOT / "sub-m17/sub-m17_ses-08_behavior+ophys.nwb",
    ]
    return preferred if all(p.exists() for p in preferred) else files[:2]


def parse_scene_zones(scene: str) -> tuple[str, str | None]:
    match = re.search(r"Location([ABC])(?:_to_([ABC]))?$", scene)
    if match is None:
        match = re.search(r"Env\d_([ABC])_to_Env\d_([ABC])$", scene)
    if match is None:
        raise ValueError(f"Cannot parse reward zone(s) from scene {scene!r}")
    return match.group(1), match.group(2)


def trial_events(behavior) -> tuple[np.ndarray, np.ndarray]:
    starts = np.flatnonzero(np.asarray(behavior["trial_start"].data[:]) > 0)
    stops = np.flatnonzero(np.asarray(behavior["teleport"].data[:]) > 0)
    if starts.size != stops.size:
        raise ValueError(f"trial starts ({starts.size}) != teleports ({stops.size})")
    if not (np.all(starts < stops) and (starts.size < 2 or np.all(stops[:-1] < starts[1:]))):
        raise ValueError("Trial start/teleport events do not strictly alternate")
    return starts, stops


def reference_dff(
    fluorescence: np.ndarray,
    neuropil: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
) -> np.ndarray:
    """Reference per-trial maximin dF/F, adapted from preprocessing.dff."""
    if fluorescence.shape != neuropil.shape:
        raise ValueError("Fluorescence and neuropil shapes differ")
    out = np.full(fluorescence.shape, np.nan, dtype=np.float32)
    for start, stop in zip(starts, stops):
        fneu = neuropil[:, start:stop]
        corrected = fluorescence[:, start:stop] - NEUROPIL_COEF * fneu
        # Restore the within-trial mean neuropil after subtracting its fluctuations.
        corrected += NEUROPIL_COEF * np.mean(fneu, axis=1, keepdims=True)
        smooth = ndimage.gaussian_filter(corrected, sigma=(0, 15))
        baseline = ndimage.minimum_filter1d(smooth, 300, axis=-1)
        baseline = ndimage.maximum_filter1d(baseline, 300, axis=-1)
        dff = (corrected - baseline) / np.abs(baseline)
        out[:, start:stop] = ndimage.gaussian_filter1d(dff, 2, axis=-1)
    return out


def correlations_with_speed(dff: np.ndarray, speed: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    x = dff[:, valid_mask].astype(np.float64, copy=False)
    y = speed[valid_mask].astype(np.float64, copy=False)
    x -= np.mean(x, axis=1, keepdims=True)
    y = y - np.mean(y)
    denominator = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (x @ y) / denominator


def reference_events(dff: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    events = np.zeros(dff.shape, dtype=np.float32)
    for start, stop in zip(starts, stops):
        events[:, start:stop] = dcnv.oasis(
            dff[:, start:stop], batch_size=2000, tau=OASIS_TAU_S, fs=FRAME_RATE
        )
    return events


def trace_processing(
    fluorescence: np.ndarray,
    neuropil: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
) -> dict[str, np.ndarray]:
    corrected_all = np.full(fluorescence.shape, np.nan, dtype=np.float32)
    baseline_all = np.full(fluorescence.shape, np.nan, dtype=np.float32)
    for start, stop in zip(starts, stops):
        fn = neuropil[start:stop]
        corrected = fluorescence[start:stop] - NEUROPIL_COEF * fn
        corrected += NEUROPIL_COEF * np.mean(fn)
        smooth = ndimage.gaussian_filter1d(corrected, 15)
        baseline = ndimage.maximum_filter1d(ndimage.minimum_filter1d(smooth, 300), 300)
        corrected_all[start:stop] = corrected
        baseline_all[start:stop] = baseline
    return {"corrected": corrected_all, "baseline": baseline_all}


def process_plane(
    ophys,
    plane: str,
    iscell: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    speed: np.ndarray,
    n_behavior: int,
    collect_diagnostics: bool,
) -> tuple[np.ndarray, dict]:
    f_series = ophys["Fluorescence"].roi_response_series[plane]
    fn_series = ophys["Neuropil"].roi_response_series[plane]
    roi_indices = np.asarray(f_series.rois.data[:], dtype=np.int64)
    local_cells = np.flatnonzero(iscell[roi_indices])
    if local_cells.size == 0:
        raise ValueError(f"No curated cells in {plane}")
    if f_series.data.shape[0] < n_behavior or fn_series.data.shape[0] < n_behavior:
        raise ValueError(f"Neural series in {plane} is shorter than behavior")

    fluorescence = np.asarray(f_series.data[:n_behavior, local_cells], dtype=np.float32).T
    neuropil = np.asarray(fn_series.data[:n_behavior, local_cells], dtype=np.float32).T
    if not (np.all(np.isfinite(fluorescence)) and np.all(np.isfinite(neuropil))):
        raise ValueError(f"Non-finite raw fluorescence in {plane}")

    dff = reference_dff(fluorescence, neuropil, starts, stops)
    valid_mask = np.zeros(n_behavior, dtype=bool)
    for start, stop in zip(starts, stops):
        valid_mask[start:stop] = True
    speed_corr = correlations_with_speed(dff, speed, valid_mask)
    keep = np.isfinite(speed_corr) & (speed_corr <= INTERNEURON_R_THRESHOLD)
    if not np.any(keep):
        raise ValueError(f"All cells rejected in {plane}")

    kept_dff = np.ascontiguousarray(dff[keep])
    events = reference_events(kept_dff, starts, stops)
    diagnostics: dict = {
        "plane": plane,
        "n_rois": int(len(roi_indices)),
        "n_iscell": int(local_cells.size),
        "n_interneurons": int(np.count_nonzero(~keep)),
        "speed_corr": speed_corr,
    }
    if collect_diagnostics:
        chosen = int(np.flatnonzero(keep)[0])
        diagnostics.update(
            {
                "raw_f": fluorescence[chosen].copy(),
                "raw_fneu": neuropil[chosen].copy(),
                "dff": dff[chosen].copy(),
                "events": events[0].copy(),
                **trace_processing(fluorescence[chosen], neuropil[chosen], starts, stops),
            }
        )
    del fluorescence, neuropil, dff, kept_dff
    return events, diagnostics


def distance_classes(position: np.ndarray, zone: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    start, stop = zone
    distance = np.where(position < start, position - start, np.where(position > stop, position - stop, 0.0))
    classes = np.full(position.shape, -1, dtype=np.int8)
    classes[distance < -50] = 0
    classes[(distance >= -50) & (distance < -10)] = 1
    classes[(distance >= -10) & (distance < 0)] = 2
    classes[distance == 0] = 3
    classes[(distance > 0) & (distance <= 10)] = 4
    classes[(distance > 10) & (distance <= 50)] = 5
    classes[distance > 50] = 6
    if np.any(classes < 0):
        raise ValueError("Unclassified reward-zone distance")
    return distance, classes


def position_classes(position: np.ndarray) -> np.ndarray:
    classes = np.full(position.shape, -1, dtype=np.int8)
    classes[position < 90] = 0
    classes[(position >= 90) & (position <= 180)] = 1
    classes[(position > 180) & (position <= 270)] = 2
    classes[(position > 270) & (position <= 360)] = 3
    classes[position > 360] = 4
    return classes


def speed_classes(speed: np.ndarray) -> np.ndarray:
    classes = np.full(speed.shape, -1, dtype=np.int8)
    classes[speed < 2] = 0
    classes[(speed >= 2) & (speed <= 10)] = 1
    classes[(speed > 10) & (speed <= 20)] = 2
    classes[(speed > 20) & (speed <= 40)] = 3
    classes[speed > 40] = 4
    return classes


def validate_session(neural: list[np.ndarray], inputs: list[np.ndarray], outputs: list[np.ndarray]) -> None:
    if not (len(neural) == len(inputs) == len(outputs) and len(neural) >= 2):
        raise ValueError("Session trial lists are inconsistent or contain fewer than two trials")
    nneurons = neural[0].shape[0]
    for trial, (n, x, y) in enumerate(zip(neural, inputs, outputs)):
        if n.ndim != 2 or x.ndim != 2 or y.ndim != 2:
            raise ValueError(f"Trial {trial}: arrays must all be 2D")
        if n.shape[0] != nneurons or not (n.shape[1] == x.shape[1] == y.shape[1]):
            raise ValueError(f"Trial {trial}: inconsistent dimensions")
        if x.shape[0] != 4 or y.shape[0] != 6:
            raise ValueError(f"Trial {trial}: incorrect input/output dimensions")
        if not (np.all(np.isfinite(n)) and np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError(f"Trial {trial}: non-finite converted data")
    domains = [range(7), range(5), range(5), range(2), range(3), range(2)]
    for dim, allowed in enumerate(domains):
        values = np.unique(np.concatenate([y[dim] for y in outputs]))
        if not set(values.tolist()).issubset(set(allowed)):
            raise ValueError(f"Output {dim} has invalid values {values}")


def process_session(path: Path, collect_diagnostics: bool = False) -> tuple[dict, dict]:
    started = time.perf_counter()
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        subject = nwb.subject.subject_id
        session_id = str(nwb.session_id)
        scene = nwb.identifier.rstrip("/").split("/")[-1]
        initial_zone, switched_zone = parse_scene_zones(scene)
        behavior = nwb.processing["behavior"]["BehavioralTimeSeries"].time_series
        ophys = nwb.processing["ophys"]
        starts, stops = trial_events(behavior)

        timestamps = np.asarray(behavior["position"].timestamps[:], dtype=np.float64)
        position = np.asarray(behavior["position"].data[:], dtype=np.float64)
        speed = np.asarray(behavior["speed"].data[:], dtype=np.float64)
        lick = np.asarray(behavior["lick"].data[:], dtype=np.float64)
        environment = np.asarray(behavior["environment"].data[:], dtype=np.float64)
        reward_times = np.asarray(behavior["Reward"].timestamps[:], dtype=np.float64)
        n_behavior = len(position)
        if not all(len(x) == n_behavior for x in (timestamps, speed, lick, environment)):
            raise ValueError("Behavior series lengths differ")
        dt = np.diff(timestamps)
        if not np.allclose(dt, 1.0 / FRAME_RATE, rtol=0, atol=1e-9):
            raise ValueError(f"Unexpected behavior sample interval in {path.name}")

        ps = ophys["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
        iscell_table = np.asarray(ps["iscell"][:])
        iscell = iscell_table[:, 0] > 0
        plane_names = list(ophys["Fluorescence"].roi_response_series)
        plane_events = []
        plane_diagnostics = []
        for plane_i, plane in enumerate(plane_names):
            events, diag = process_plane(
                ophys,
                plane,
                iscell,
                starts,
                stops,
                speed,
                n_behavior,
                collect_diagnostics and plane_i == 0,
            )
            plane_events.append(events)
            plane_diagnostics.append(diag)
        events = np.concatenate(plane_events, axis=0) if len(plane_events) > 1 else plane_events[0]

        outcomes = np.asarray(
            [np.any((reward_times >= timestamps[a]) & (reward_times < timestamps[z])) for a, z in zip(starts, stops)],
            dtype=np.int8,
        )
        bad_lick = np.asarray([np.mean(lick[a:z] > 2) > LICK_ERROR_FRACTION for a, z in zip(starts, stops)])

        neural_trials: list[np.ndarray] = []
        input_trials: list[np.ndarray] = []
        output_trials: list[np.ndarray] = []
        kept_ordinals = []
        signed_distances = []
        for trial_i, (start, stop) in enumerate(zip(starts, stops)):
            if bad_lick[trial_i]:
                continue
            pos = position[start:stop]
            spd = speed[start:stop]
            env_values = np.unique(environment[start:stop])
            env_values = env_values[env_values >= 0]
            if env_values.size != 1:
                raise ValueError(f"Trial {trial_i} does not have one valid environment")
            env = int(round(float(env_values[0])))
            if env not in (0, 1):
                raise ValueError(f"Trial {trial_i}: invalid ENV value {env}")
            zone_label = initial_zone if switched_zone is None or trial_i < 30 else switched_zone
            signed_distance, dist_class = distance_classes(pos, REWARD_ZONES[zone_label])
            T = stop - start
            previous_outcome = int(outcomes[trial_i - 1]) if trial_i > 0 else 0
            input_trial = np.vstack(
                (
                    timestamps[start:stop] - timestamps[start],
                    np.full(T, env),
                    np.full(T, trial_i),
                    np.full(T, previous_outcome),
                )
            ).astype(np.float32)
            output_trial = np.vstack(
                (
                    dist_class,
                    position_classes(pos),
                    speed_classes(spd),
                    (lick[start:stop] > 0).astype(np.int8),
                    np.full(T, ZONE_CODES[zone_label], dtype=np.int8),
                    np.full(T, outcomes[trial_i], dtype=np.int8),
                )
            ).astype(np.int8)
            neural_trials.append(np.ascontiguousarray(events[:, start:stop], dtype=np.float32))
            input_trials.append(input_trial)
            output_trials.append(output_trial)
            kept_ordinals.append(trial_i)
            if collect_diagnostics:
                signed_distances.append(signed_distance)

        validate_session(neural_trials, input_trials, output_trials)
        info = {
            "session_id": f"{subject}_ses-{session_id}",
            "subject": subject,
            "nwb_session_id": session_id,
            "scene": scene,
            "source_file": str(path.relative_to(DATA_ROOT)),
            "n_planes": len(plane_names),
            "n_rois": int(len(iscell)),
            "n_iscell": int(np.count_nonzero(iscell)),
            "n_interneurons_excluded": int(sum(d["n_interneurons"] for d in plane_diagnostics)),
            "n_neurons": int(events.shape[0]),
            "n_trials_raw": int(len(starts)),
            "n_bad_lick_trials_excluded": int(np.count_nonzero(bad_lick)),
            "n_trials": int(len(neural_trials)),
            "n_timepoints": int(sum(x.shape[1] for x in neural_trials)),
            "frame_rate_hz": FRAME_RATE,
        }
        diagnostics = {
            "info": info,
            "plane": plane_diagnostics[0] if collect_diagnostics else {},
            "starts": starts,
            "stops": stops,
            "position": position,
            "speed": speed,
            "lick": lick,
            "timestamps": timestamps,
            "outcomes": outcomes,
            "bad_lick": bad_lick,
            "kept_ordinals": np.asarray(kept_ordinals),
            "signed_distances": signed_distances,
        }
    info["processing_seconds"] = time.perf_counter() - started
    return {"neural": neural_trials, "input": input_trials, "output": output_trials, "info": info}, diagnostics


def plot_processing(session: dict, diagnostics: dict) -> Path:
    info = session["info"]
    plane = diagnostics["plane"]
    kept = diagnostics["kept_ordinals"]
    representative = 30 if 30 in kept else int(kept[0])
    converted_i = int(np.flatnonzero(kept == representative)[0])
    start = int(diagnostics["starts"][representative])
    stop = int(diagnostics["stops"][representative])
    t = diagnostics["timestamps"][start:stop] - diagnostics["timestamps"][start]
    x = session["input"][converted_i]
    y = session["output"][converted_i]
    neural = session["neural"][converted_i]

    fig, axes = plt.subplots(4, 2, figsize=(17, 15), constrained_layout=True)
    ax = axes.ravel()
    trial_lengths = diagnostics["stops"] - diagnostics["starts"]
    ax[0].plot(trial_lengths / FRAME_RATE, color="black", lw=1)
    ax[0].scatter(np.arange(len(trial_lengths))[diagnostics["bad_lick"]], trial_lengths[diagnostics["bad_lick"]] / FRAME_RATE, color="red", label="bad lick QC")
    ax[0].axvline(29.5, color="purple", ls="--", label="switch after trial 30")
    ax[0].set(title="Trial segmentation and QC", xlabel="zero-based trial", ylabel="duration (s)")
    ax[0].legend(fontsize=8)

    raw_f = plane["raw_f"][start:stop]
    raw_fn = plane["raw_fneu"][start:stop]
    ax[1].plot(t, raw_f, label="F", lw=0.8)
    ax[1].plot(t, raw_fn, label="Fneu", lw=0.8, alpha=0.8)
    ax[1].plot(t, plane["corrected"][start:stop], label="F - 0.7 Fneu + mean", lw=0.8)
    ax[1].plot(t, plane["baseline"][start:stop], label="maximin baseline", lw=1.2)
    ax[1].set(title="Fluorescence correction and baseline", xlabel="time from trial start (s)")
    ax[1].legend(fontsize=8)

    ax[2].plot(t, plane["dff"][start:stop], label="smoothed dF/F")
    ax[2].plot(t, plane["events"][start:stop], label="OASIS events")
    ax[2].set(title="Reference neural processing", xlabel="time (s)")
    ax[2].legend(fontsize=8)

    pos = diagnostics["position"][start:stop]
    dist = diagnostics["signed_distances"][converted_i]
    ax[3].plot(t, pos, label="absolute position (cm)")
    ax[3].plot(t, dist, label="signed distance to zone")
    ax[3].step(t, 20 * y[0], where="mid", label="20 × distance class", alpha=0.7)
    ax[3].set(title="Position, reward-zone distance, discretization", xlabel="time (s)")
    ax[3].legend(fontsize=8)

    ax[4].plot(t, diagnostics["speed"][start:stop], color="black", lw=0.8, label="speed")
    for threshold in (2, 10, 20, 40):
        ax[4].axhline(threshold, color="gray", ls="--", lw=0.7)
    ax[4].step(t, 10 * y[2], where="mid", label="10 × speed class")
    ax[4].set(title="Speed thresholds", xlabel="time (s)", ylabel="cm/s")
    ax[4].legend(fontsize=8)

    ax[5].plot(t, diagnostics["lick"][start:stop], label="cumulative lick/frame")
    ax[5].step(t, y[3], where="mid", label="binary lick")
    ax[5].set(title="Lick binarization", xlabel="time (s)")
    ax[5].legend(fontsize=8)

    ax[6].imshow(y, aspect="auto", interpolation="nearest", extent=[t[0], t[-1], 5.5, -0.5])
    ax[6].set(title="All six categorical outputs", xlabel="time (s)", ylabel="output dimension", yticks=range(6))

    display = neural[: min(80, neural.shape[0])]
    ax[7].imshow(display, aspect="auto", interpolation="nearest", extent=[t[0], t[-1], display.shape[0], 0], cmap="magma")
    ax[7].set(title=f"OASIS neural events ({display.shape[0]}/{neural.shape[0]} neurons)", xlabel="time (s)", ylabel="neuron")
    fig.suptitle(
        f"{info['session_id']} | {info['scene']} | iscell {info['n_iscell']} → {info['n_neurons']} neurons; "
        f"trials {info['n_trials_raw']} → {info['n_trials']}",
        fontsize=13,
    )
    out = Path(f"processing_{info['session_id']}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def dataset_statistics(data: dict) -> dict:
    ntrials = sum(len(s) for s in data["neural"])
    nsamples = sum(x.shape[1] for s in data["neural"] for x in s)
    neuron_counts = np.asarray([len(x) for x in data["brain_region_idx"]])
    distributions = []
    for output_i, values in enumerate(data["output_values"]):
        counts = np.zeros(len(values), dtype=np.int64)
        for session in data["output"]:
            for trial in session:
                counts += np.bincount(trial[output_i].astype(int), minlength=len(values))
        distributions.append((counts / counts.sum()).tolist())
    return {
        "sessions": len(data["neural"]),
        "subjects": len(data["subjects"]),
        "trials": ntrials,
        "samples": nsamples,
        "cell_session_instances": int(neuron_counts.sum()),
        "neurons_per_session_min": int(neuron_counts.min()),
        "neurons_per_session_mean": float(neuron_counts.mean()),
        "neurons_per_session_max": int(neuron_counts.max()),
        "output_distributions": distributions,
    }


def main() -> None:
    args = parse_args()
    files = discover_files(args.sample)
    overall_start = time.perf_counter()
    print(f"Mode: {'sample' if args.sample else 'full'}; sessions: {len(files)}", flush=True)
    print(f"Paper-matched neural processing: maximin dF/F + OASIS at {FRAME_RATE:.7f} Hz", flush=True)

    neural, inputs, outputs, infos, brain_region_idx = [], [], [], [], []
    subjects = sorted({p.parent.name.removeprefix("sub-") for p in files})
    subject_lookup = {subject: i for i, subject in enumerate(subjects)}
    subject_idx = []
    for session_i, path in enumerate(files):
        t0 = time.perf_counter()
        result, diagnostics = process_session(path, args.show_processing and session_i < 2)
        neural.append(result["neural"])
        inputs.append(result["input"])
        outputs.append(result["output"])
        infos.append(result["info"])
        brain_region_idx.append(np.zeros(result["info"]["n_neurons"], dtype=np.int64))
        subject_idx.append(subject_lookup[result["info"]["subject"]])
        if args.show_processing and session_i < 2:
            plot_path = plot_processing(result, diagnostics)
            print(f"  saved {plot_path}", flush=True)
        elapsed = time.perf_counter() - t0
        print(
            f"[{session_i + 1:3d}/{len(files)}] {result['info']['session_id']}: "
            f"{result['info']['n_neurons']} neurons, {result['info']['n_trials']} trials, "
            f"{result['info']['n_timepoints']} samples, {elapsed:.2f}s",
            flush=True,
        )
        del diagnostics
        gc.collect()

    data = {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": ["CA1"],
        "brain_region_idx": brain_region_idx,
        "input_names": [
            "time from trial start (s)",
            "environment type (ENV1=0, ENV2=1)",
            "trial number (zero-based)",
            "previous trial outcome (omitted=0, rewarded=1)",
        ],
        "output_names": [
            "distance to reward zone",
            "absolute position in corridor",
            "speed",
            "lick",
            "reward zone location",
            "reward outcome",
        ],
        "output_values": [
            ["< -50 cm", "-50 to < -10 cm", "-10 to < 0 cm", "in reward zone (0 cm)", "> 0 to 10 cm", "> 10 to 50 cm", "> 50 cm"],
            ["< 90 cm", "90 to 180 cm", "> 180 to 270 cm", "> 270 to 360 cm", "> 360 cm"],
            ["< 2 cm/s", "2 to 10 cm/s", "> 10 to 20 cm/s", "> 20 to 40 cm/s", "> 40 cm/s"],
            ["no lick", "lick"],
            ["A (80-130 cm)", "B (200-250 cm)", "C (320-370 cm)"],
            ["omitted", "rewarded"],
        ],
        "metadata": {
            "task_description": "Head-fixed mice traverse a 450-cm virtual corridor in ENV1/ENV2 with hidden 50-cm reward zones A/B/C; neural activity predicts position, movement, licking, reward location, and outcome.",
            "time_bin_size": TIME_BIN_MS,
            "temporal_alignment_event": "trial_start: entry onto the 0-cm start of the virtual linear track",
            "off_start": 0.0,
            "off_end": None,
            "variable_trial_duration": True,
            "neural_activity": "OASIS-deconvolved, per-trial maximin dF/F from manually curated CA1 ROIs; putative interneurons with dF/F-speed r > 0.5 excluded",
            "trial_interval": "[trial_start sample, teleport sample); teleport sample excluded",
            "lick_qc": "Trials with >30% of frames having cumulative lick count >2 are excluded; remaining lick counts are binarized >0",
            "reward_zone_coordinates_cm": {"A": [80.0, 130.0], "B": [200.0, 250.0], "C": [320.0, 370.0]},
            "distance_definition": "Signed distance to nearest point in active reward-zone interval: negative before, zero anywhere inside, positive after",
            "sample_rate_hz": FRAME_RATE,
            "source_format": "NWB read exclusively with pynwb",
            "session_info": infos,
        },
    }
    stats = dataset_statistics(data)
    data["metadata"]["conversion_statistics"] = stats
    print("Conversion statistics:", stats, flush=True)

    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    write_start = time.perf_counter()
    with args.outpicklefile.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    write_seconds = time.perf_counter() - write_start
    total_seconds = time.perf_counter() - overall_start
    print(
        f"Wrote {args.outpicklefile} ({args.outpicklefile.stat().st_size / 2**30:.3f} GiB) "
        f"in {write_seconds:.2f}s; total {total_seconds:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
