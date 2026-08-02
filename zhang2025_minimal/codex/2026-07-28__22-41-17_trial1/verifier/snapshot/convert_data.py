#!/usr/bin/env python3
"""Convert the cached IBL brain-wide-map release into the decoder format."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from iblatlas.regions import BrainRegions

sys.path.insert(0, "/app/code/ibllib")
from brainbox.behavior.wheel import interpolate_position, velocity_filtered  # noqa: E402


ROOT = Path("/app")
DATA_ROOT = ROOT / "data" / "one_cache"
RELEASE_CSV = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"

ALIGN_EVENT = "stimOn_times"
WINDOW = (-0.5, 1.5)
BIN_SIZE_S = 0.02
NBINS = int(round((WINDOW[1] - WINDOW[0]) / BIN_SIZE_S))
WHEEL_FS = 1000


def pick_latest(session_path: Path, pattern: str) -> Path:
    matches = sorted(session_path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern} in {session_path}")
    return matches[-1]


def find_session_path(row: pd.Series) -> Path:
    return DATA_ROOT / row["lab"] / "Subjects" / row["subject"] / row["date"] / f"{int(row['session_number']):03d}"


def load_trials_table(session_path: Path) -> pd.DataFrame:
    return pd.read_parquet(pick_latest(session_path, "alf/**/_ibl_trials.table.pqt"))


def make_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    required = [
        "stimOn_times",
        "choice",
        "feedback_times",
        "probabilityLeft",
        "firstMovement_times",
        "feedbackType",
    ]
    mask = np.ones(len(trials), dtype=bool)
    rt = trials["firstMovement_times"].to_numpy() - trials["stimOn_times"].to_numpy()
    trial_len = trials["feedback_times"].to_numpy() - trials["goCue_times"].to_numpy()
    mask &= rt >= 0.08
    mask &= rt <= 2.0
    mask &= trial_len <= 10.0
    mask &= trials["choice"].to_numpy() != 0
    for col in required:
        mask &= ~trials[col].isna().to_numpy()
    return mask


def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    trial_num = np.ones(prob_left.shape[0], dtype=np.float32)
    curr = 1.0
    for i in range(1, prob_left.shape[0]):
        if np.isclose(prob_left[i], prob_left[i - 1]):
            curr += 1.0
        else:
            curr = 1.0
        trial_num[i] = curr
    return trial_num


def load_wheel_speed(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.timestamps.npy"))
    position = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.position.npy"))
    if timestamps.shape[0] != position.shape[0]:
        raise ValueError("Wheel timestamps/position length mismatch")
    interp_pos, interp_t = interpolate_position(timestamps, position, freq=WHEEL_FS)
    velocity, _ = velocity_filtered(interp_pos, fs=WHEEL_FS, corner_frequency=20, order=8)
    return interp_t.astype(np.float32), np.abs(velocity).astype(np.float32)


def _load_camera_stream(session_path: Path, view: str) -> tuple[np.ndarray, np.ndarray]:
    times = np.load(pick_latest(session_path, f"alf/**/*_ibl_{view}Camera.times.npy"))
    values = np.load(pick_latest(session_path, f"alf/**/{view}Camera.ROIMotionEnergy.npy"))
    if times.shape[0] < values.shape[0]:
        raise ValueError(f"{view} camera timestamps shorter than motion-energy array")
    if times.shape[0] > values.shape[0]:
        times = times[-values.shape[0]:]
    return times.astype(np.float32), values.astype(np.float32)


def load_whisker_motion_energy(session_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        times, values = _load_camera_stream(session_path, "left")
        return times, values, "left"
    except Exception:
        times, values = _load_camera_stream(session_path, "right")
        return times, values, "right"


def load_good_units(session_path: Path, probe_names: tuple[str, ...], br: BrainRegions) -> tuple[np.ndarray, np.ndarray, list[str]]:
    all_times = []
    all_clusters = []
    all_regions: list[str] = []
    offset = 0

    for probe_name in probe_names:
        probe_base = max(session_path.glob(f"alf/{probe_name}/pykilosort/*"))
        metrics = pd.read_parquet(probe_base / "clusters.metrics.pqt", columns=["cluster_id", "label"])
        cluster_ids = metrics["cluster_id"].to_numpy(dtype=np.int64)
        good_mask = metrics["label"].to_numpy(dtype=float) >= 1.0
        good_cluster_ids = cluster_ids[good_mask]
        if good_cluster_ids.size == 0:
            continue

        cluster_channels = np.load(probe_base / "clusters.channels.npy")[good_mask].astype(np.int64)
        channel_region_ids = np.load(probe_base / "channels.brainLocationIds_ccf_2017.npy")
        cluster_region_ids = channel_region_ids[cluster_channels]
        allen_regions = br.id2acronym(cluster_region_ids)
        beryl_regions = br.acronym2acronym(allen_regions, mapping="Beryl")
        all_regions.extend(str(x) for x in beryl_regions.tolist())

        spikes_times = np.load(probe_base / "spikes.times.npy").astype(np.float32)
        spikes_clusters = np.load(probe_base / "spikes.clusters.npy").astype(np.int64)
        max_cluster_id = int(max(spikes_clusters.max(initial=-1), good_cluster_ids.max(initial=-1))) + 1
        cluster_map = np.full(max_cluster_id, -1, dtype=np.int32)
        cluster_map[good_cluster_ids] = np.arange(offset, offset + good_cluster_ids.shape[0], dtype=np.int32)
        valid = (spikes_clusters >= 0) & (spikes_clusters < max_cluster_id)
        mapped = cluster_map[spikes_clusters[valid]]
        keep = mapped >= 0
        all_times.append(spikes_times[valid][keep])
        all_clusters.append(mapped[keep])
        offset += good_cluster_ids.shape[0]

    if offset == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32), []

    spike_times = np.concatenate(all_times)
    spike_clusters = np.concatenate(all_clusters)
    order = np.argsort(spike_times, kind="stable")
    return spike_times[order], spike_clusters[order], all_regions


def bin_spikes(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    intervals: np.ndarray,
    n_neurons: int,
) -> list[np.ndarray]:
    starts = intervals[:, 0]
    ends = intervals[:, 1]
    start_idx = np.searchsorted(spike_times, starts, side="left")
    end_idx = np.searchsorted(spike_times, ends, side="left")
    trials = []

    for lo, hi, start in zip(start_idx, end_idx, starts, strict=True):
        if hi <= lo:
            trials.append(np.zeros((n_neurons, NBINS), dtype=np.float16))
            continue
        times = spike_times[lo:hi] - start
        bins = np.floor(times / BIN_SIZE_S).astype(np.int64)
        valid = (bins >= 0) & (bins < NBINS)
        flat = spike_clusters[lo:hi][valid] * NBINS + bins[valid]
        counts = np.bincount(flat, minlength=n_neurons * NBINS).reshape(n_neurons, NBINS)
        trials.append(counts.astype(np.float16))

    return trials


def interpolate_behavior_into_trials(
    times: np.ndarray,
    values: np.ndarray,
    intervals: np.ndarray,
) -> tuple[list[np.ndarray | None], np.ndarray]:
    starts = intervals[:, 0]
    ends = intervals[:, 1]
    idx_beg = np.searchsorted(times, starts, side="right")
    idx_end = np.searchsorted(times, ends, side="left")
    x_interp = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
    outputs: list[np.ndarray | None] = []
    valid = np.zeros(intervals.shape[0], dtype=bool)

    for i, (ib, ie, beg, end) in enumerate(zip(idx_beg, idx_end, starts, ends, strict=True)):
        t = times[ib:ie]
        y = values[ib:ie]
        if t.shape[0] == 0:
            outputs.append(None)
            continue
        if np.isnan(y).any():
            outputs.append(None)
            continue
        if np.abs(beg - t[0]) > BIN_SIZE_S or np.abs(end - t[-1]) > BIN_SIZE_S:
            outputs.append(None)
            continue
        rel_t = t - beg
        interp = np.interp(x_interp, rel_t, y).astype(np.float32)
        if np.isnan(interp).any():
            outputs.append(None)
            continue
        outputs.append(interp)
        valid[i] = True

    return outputs, valid


def map_choice(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], -1, dtype=np.int8)
    out[np.isclose(values, -1.0)] = 0
    out[np.isclose(values, 1.0)] = 1
    if np.any(out < 0):
        raise ValueError("Unexpected choice values encountered")
    return out


def map_probability_left(values: np.ndarray) -> np.ndarray:
    rounded = np.round(values.astype(np.float64), 1)
    out = np.full(values.shape[0], -1, dtype=np.int8)
    out[np.isclose(rounded, 0.2)] = 0
    out[np.isclose(rounded, 0.5)] = 1
    out[np.isclose(rounded, 0.8)] = 2
    if np.any(out < 0):
        raise ValueError("Unexpected probabilityLeft values encountered")
    return out


def digitize_tertiles(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(values, thresholds, right=False).astype(np.int8)


def build_output_trials(record: dict, wheel_thresholds: np.ndarray, whisker_thresholds: np.ndarray) -> list[np.ndarray]:
    trials = []
    for choice, prior, wheel, whisker in zip(
        record["choice"],
        record["prior"],
        record["wheel_cont"],
        record["whisker_cont"],
        strict=True,
    ):
        wheel_disc = digitize_tertiles(wheel, wheel_thresholds)
        whisker_disc = digitize_tertiles(whisker, whisker_thresholds)
        out = np.vstack(
            [
                np.full((1, NBINS), choice, dtype=np.int8),
                np.full((1, NBINS), prior, dtype=np.int8),
                wheel_disc[None, :],
                whisker_disc[None, :],
            ]
        )
        trials.append(out)
    return trials


def summarize_records(records: list[dict], skip_reasons: Counter) -> dict:
    ntrials = [len(r["neural"]) for r in records]
    nneurons = [len(r["region_labels"]) for r in records]
    summary = {
        "included_sessions": len(records),
        "included_trials": int(sum(ntrials)),
        "mean_trials_per_session": float(np.mean(ntrials)) if ntrials else 0.0,
        "median_trials_per_session": float(np.median(ntrials)) if ntrials else 0.0,
        "mean_neurons_per_session": float(np.mean(nneurons)) if nneurons else 0.0,
        "median_neurons_per_session": float(np.median(nneurons)) if nneurons else 0.0,
        "skip_reasons": dict(skip_reasons),
    }
    return summary


def convert_dataset(max_sessions: int | None = None) -> tuple[dict, dict]:
    release = pd.read_csv(RELEASE_CSV, index_col=0)
    sessions = (
        release.groupby("eid", sort=False)
        .agg(
            {
                "subject": "first",
                "date": "first",
                "session_number": "first",
                "lab": "first",
                "probe_name": lambda x: tuple(sorted(x)),
            }
        )
        .reset_index()
    )

    br = BrainRegions()
    records: list[dict] = []
    skip_reasons: Counter = Counter()
    wheel_pool = []
    whisker_pool = []
    start_time = time.time()

    for idx, row in enumerate(sessions.to_dict("records"), start=1):
        session_path = find_session_path(row)
        try:
            trials = load_trials_table(session_path)
            trial_mask = make_trial_mask(trials)
            if trial_mask.sum() < 2:
                skip_reasons["too_few_trials_after_reference_mask"] += 1
                continue

            wheel_times, wheel_speed = load_wheel_speed(session_path)
            whisker_times, whisker_me, whisker_view = load_whisker_motion_energy(session_path)
            spike_times, spike_clusters, region_labels = load_good_units(session_path, row["probe_name"], br)
            if len(region_labels) == 0:
                skip_reasons["no_good_units"] += 1
                continue

            stim_on = trials[ALIGN_EVENT].to_numpy(dtype=np.float32)
            intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]].astype(np.float32)
            neural_trials = bin_spikes(spike_times, spike_clusters, intervals, len(region_labels))
            neural_valid = np.array([np.any(trial > 0) for trial in neural_trials], dtype=bool)
            wheel_trials, wheel_valid = interpolate_behavior_into_trials(wheel_times, wheel_speed, intervals)
            whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
            final_mask = trial_mask & wheel_valid & whisker_valid & neural_valid
            if final_mask.sum() < 2:
                skip_reasons["too_few_trials_after_stream_alignment"] += 1
                continue

            trial_number = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy(dtype=np.float32))
            time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
            selected_idx = np.flatnonzero(final_mask)
            neural_selected = [neural_trials[i] for i in selected_idx]
            wheel_selected = [wheel_trials[i] for i in selected_idx]
            whisker_selected = [whisker_trials[i] for i in selected_idx]
            assert all(x is not None for x in wheel_selected)
            assert all(x is not None for x in whisker_selected)

            input_trials = [
                np.vstack(
                    [
                        time_input,
                        np.full(NBINS, trial_number[i], dtype=np.float32),
                    ]
                ).astype(np.float32)
                for i in selected_idx
            ]

            choice = map_choice(trials["choice"].to_numpy(dtype=np.float32)[selected_idx])
            prior = map_probability_left(trials["probabilityLeft"].to_numpy(dtype=np.float32)[selected_idx])
            wheel_cont = [np.asarray(x, dtype=np.float32) for x in wheel_selected]
            whisker_cont = [np.asarray(x, dtype=np.float32) for x in whisker_selected]
            wheel_pool.append(np.concatenate(wheel_cont))
            whisker_pool.append(np.concatenate(whisker_cont))

            records.append(
                {
                    "eid": row["eid"],
                    "subject": row["subject"],
                    "date": row["date"],
                    "lab": row["lab"],
                    "probe_names": list(row["probe_name"]),
                    "session_path": str(session_path),
                    "whisker_view": whisker_view,
                    "n_trials_total": int(len(trials)),
                    "n_trials_reference_mask": int(trial_mask.sum()),
                    "n_trials_final": int(final_mask.sum()),
                    "neural": neural_selected,
                    "input": input_trials,
                    "choice": choice,
                    "prior": prior,
                    "wheel_cont": wheel_cont,
                    "whisker_cont": whisker_cont,
                    "region_labels": region_labels,
                }
            )

            if idx % 25 == 0 or idx == len(sessions):
                elapsed = time.time() - start_time
                print(
                    f"[{idx:03d}/{len(sessions)}] included={len(records)} "
                    f"last_eid={row['eid']} trials={int(final_mask.sum())} neurons={len(region_labels)} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

            if max_sessions is not None and len(records) >= max_sessions:
                break

        except FileNotFoundError:
            skip_reasons["missing_required_file"] += 1
        except Exception as exc:
            skip_reasons[type(exc).__name__] += 1
            print(f"Skipped session {row['eid']} due to {type(exc).__name__}: {exc}", flush=True)

    if not records:
        raise RuntimeError("No sessions were converted")

    wheel_thresholds = np.quantile(np.concatenate(wheel_pool), [1 / 3, 2 / 3]).astype(np.float32)
    whisker_thresholds = np.quantile(np.concatenate(whisker_pool), [1 / 3, 2 / 3]).astype(np.float32)

    subjects = []
    subject_to_idx = {}
    for rec in records:
        if rec["subject"] not in subject_to_idx:
            subject_to_idx[rec["subject"]] = len(subjects)
            subjects.append(rec["subject"])

    brain_regions = sorted({region for rec in records for region in rec["region_labels"]})
    brain_region_to_idx = {region: i for i, region in enumerate(brain_regions)}

    data = {
        "neural": [rec["neural"] for rec in records],
        "input": [rec["input"] for rec in records],
        "output": [build_output_trials(rec, wheel_thresholds, whisker_thresholds) for rec in records],
        "subjects": subjects,
        "subject_idx": np.array([subject_to_idx[rec["subject"]] for rec in records], dtype=np.int16),
        "brain_regions": brain_regions,
        "brain_region_idx": [
            np.array([brain_region_to_idx[region] for region in rec["region_labels"]], dtype=np.int16)
            for rec in records
        ],
        "input_names": ["time_since_stimulus_onset_s", "trial_number_in_block"],
        "output_names": [
            "choice",
            "prior_probability_left",
            "wheel_speed_tertile",
            "whisker_motion_energy_tertile",
        ],
        "output_values": [
            ["left", "right"],
            ["0.2", "0.5", "0.8"],
            ["low", "medium", "high"],
            ["low", "medium", "high"],
        ],
        "metadata": {
            "task_description": (
                "IBL biased visual decision task; decode stimulus-aligned choice, block prior, "
                "wheel speed tertiles, and whisker motion-energy tertiles from population spiking."
            ),
            "time_bin_size": BIN_SIZE_S * 1000.0,
            "temporal_alignment_event": "stimulus onset (trials.stimOn_times)",
            "off_start": WINDOW[0],
            "off_end": WINDOW[1],
            "bin_time_reference": "bin end times relative to stimulus onset",
            "trial_exclusion": {
                "required_non_nan_events": [
                    "stimOn_times",
                    "choice",
                    "feedback_times",
                    "probabilityLeft",
                    "firstMovement_times",
                    "feedbackType",
                ],
                "reaction_time_range_s": [0.08, 2.0],
                "max_trial_length_s": 10.0,
                "exclude_no_choice": True,
                "exclude_unbiased_block": False,
                "exclude_all_zero_spike_trials_after_binning": True,
            },
            "neuron_qc": "clusters.metrics label >= 1.0 (well-isolated units)",
            "brain_region_mapping": "clusters.channels -> channels.brainLocationIds_ccf_2017 -> Allen acronym -> Beryl acronym",
            "wheel_processing": "interpolate wheel position to 1000 Hz, then Butterworth-filtered velocity (20 Hz corner, order 8); output uses absolute velocity",
            "whisker_motion_energy_view_policy": "prefer left camera; fall back to right camera if left stream is unavailable",
            "continuous_output_discretization": {
                "method": "global tertiles over all included time bins in the converted dataset",
                "wheel_speed_thresholds": wheel_thresholds.tolist(),
                "whisker_motion_energy_thresholds": whisker_thresholds.tolist(),
            },
            "source_release_rows": int(len(release)),
            "source_release_sessions": int(sessions["eid"].nunique()),
            "reference_good_units_release": 75708,
            "included_session_ids": [rec["eid"] for rec in records],
            "session_info": [
                {
                    "eid": rec["eid"],
                    "subject": rec["subject"],
                    "date": rec["date"],
                    "lab": rec["lab"],
                    "probe_names": rec["probe_names"],
                    "session_path": rec["session_path"],
                    "whisker_view": rec["whisker_view"],
                    "n_trials_total": rec["n_trials_total"],
                    "n_trials_reference_mask": rec["n_trials_reference_mask"],
                    "n_trials_final": rec["n_trials_final"],
                    "n_neurons": len(rec["region_labels"]),
                }
                for rec in records
            ],
            "conversion_summary": summarize_records(records, skip_reasons),
        },
    }

    sanity = {
        "included_sessions": len(records),
        "included_trials": int(sum(len(rec["neural"]) for rec in records)),
        "included_subjects": len(subjects),
        "total_good_neurons_included_sessions": int(sum(len(rec["region_labels"]) for rec in records)),
        "wheel_speed_tertile_thresholds": wheel_thresholds.tolist(),
        "whisker_motion_energy_tertile_thresholds": whisker_thresholds.tolist(),
        "skip_reasons": dict(skip_reasons),
    }
    return data, sanity


def save_pickle(obj: object, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert cached IBL BWM data into decoder format.")
    parser.add_argument("--output", type=Path, default=ROOT / "converted_data.pkl")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    data, sanity = convert_dataset(max_sessions=args.max_sessions)
    save_pickle(data, args.output)

    print(json.dumps(sanity, indent=2))
    print(f"Saved {args.output}")
    if args.summary_json is not None:
        with args.summary_json.open("w") as f:
            json.dump(sanity, f, indent=2)
        print(f"Saved {args.summary_json}")


if __name__ == "__main__":
    main()
