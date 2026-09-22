#!/usr/bin/env python3
"""Convert the Mesoscale Activity Map NWB files for neural decoding.

Decisions carried over from the source papers/code
--------------------------------------------------
* Keep only units labeled ``good`` by the published region-specific QC
  classifiers.  These are also the units with anatomical ``anno_name`` values.
* Use the NWB go-cue timestamps as the common clock.  Spike timestamps, video
  timestamps, task events, and trial intervals are all expressed in this clock.
* Convert non-overlapping 50-ms spike-count bins to Hz by dividing by 0.05 s.

Task-specific decisions
-----------------------
* Keep early-lick, no-response, and photostimulation trials.  The papers omit
  these for many analyses, but they are required classes/inputs in this task.
  Trials must nevertheless be inside every retained unit's NWB
  ``is_good_trials`` coverage, so an all-zero recording gap is never treated as
  biological silence.
* Exclude auto-water and free-water trials, as does the repository's
  ``get_regular_trial_mask``. They are not requested decoder classes; free-water
  trials also have no spikes in this NWB release.
* Define choice from the authoritative instruction/outcome trial labels: a hit
  has the instructed choice, a miss the opposite choice, and an ignored trial
  has no lick.  This avoids rare missing/mis-timestamped lick events in the NWB.
* Sample the DLC tongue trace at each neural-bin center, using the preceding
  camera frame as in the repository's marker-alignment code.  DLC likelihood
  >= 0.9 denotes a visible tongue.  Session percentiles use all visible frames.
* Preserve the fine Allen CCF annotation in ``brain_regions`` rather than
  replacing it with a coarser insertion target.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np


OFF_START = -2.5
OFF_END = 1.5
BIN_SIZE_S = 0.05
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE_S))
BIN_EDGES = OFF_START + np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE_S
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
TONGUE_LIKELIHOOD_THRESHOLD = 0.9


def _decode(value) -> str:
    """Decode an NWB variable-length string, treating NaN as missing."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ""
    return str(value)


def _decode_array(dataset) -> np.ndarray:
    return np.asarray([_decode(x) for x in dataset[:]])


def _optional_float(value) -> float:
    text = _decode(value)
    return np.nan if text in {"", "N/A", "nan"} else float(text)


def _bin_good_units(
    all_spikes: np.ndarray,
    spike_ends: np.ndarray,
    good_rows: np.ndarray,
    go_times: np.ndarray,
) -> np.ndarray:
    """Return (trial, good_unit, time) non-overlapping firing-rate bins."""
    n_trials = len(go_times)
    n_units = len(good_rows)
    rates = np.zeros((n_trials, n_units, N_BINS), dtype=np.float32)
    window_starts = go_times + OFF_START
    window_stops = go_times + OFF_END

    # Go-cue separations in this dataset exceed the four-second analysis window,
    # so the most recent window start identifies a spike's unique trial.
    for out_unit, unit_row in enumerate(good_rows):
        start = 0 if unit_row == 0 else int(spike_ends[unit_row - 1])
        stop = int(spike_ends[unit_row])
        spikes = all_spikes[start:stop]
        if spikes.size == 0:
            continue

        trial = np.searchsorted(window_starts, spikes, side="right") - 1
        in_range = trial >= 0
        trial = trial[in_range]
        spikes = spikes[in_range]
        in_range = (trial < n_trials) & (spikes < window_stops[trial])
        trial = trial[in_range]
        spikes = spikes[in_range]
        if spikes.size == 0:
            continue

        # The tiny epsilon only stabilizes decimal timestamps that are a few ulps
        # below an exact edge. Bins remain left-closed and right-open.
        time_bin = np.floor(
            (spikes - window_starts[trial]) / BIN_SIZE_S + 1e-10
        ).astype(np.int64)
        valid = (time_bin >= 0) & (time_bin < N_BINS)
        flat_bin = trial[valid] * N_BINS + time_bin[valid]
        counts = np.bincount(flat_bin, minlength=n_trials * N_BINS)
        rates[:, out_unit, :] = counts.reshape(n_trials, N_BINS) / BIN_SIZE_S

    return rates


def _tone_onsets(go_times: np.ndarray, sample_starts: np.ndarray) -> np.ndarray:
    """Find the final tone onset before each go cue (replays produce extras)."""
    idx = np.searchsorted(sample_starts, go_times, side="right") - 1
    if np.any(idx < 0):
        raise ValueError("A go cue has no preceding sample/tone onset")
    return sample_starts[idx]


def _tongue_categories(
    tongue_data: np.ndarray,
    camera_times: np.ndarray,
    go_times: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Return (trial,time) tongue-y categories and the session cut points."""
    y = tongue_data[:, 1]
    likelihood = tongue_data[:, 2]
    visible_session = (
        np.isfinite(y)
        & np.isfinite(likelihood)
        & (likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
    )
    if not np.any(visible_session):
        # All requested samples will be category 3. NaN cut points make this
        # exceptional condition explicit in session metadata.
        q40 = q60 = np.nan
    else:
        q40, q60 = np.percentile(y[visible_session], [40, 60])

    target_times = go_times[:, None] + BIN_CENTERS[None, :]
    frame = np.searchsorted(camera_times, target_times, side="right") - 1
    valid_frame = frame >= 0
    frame = np.clip(frame, 0, len(camera_times) - 1)
    # Mark gaps longer than 20 ms as unavailable rather than carrying a stale
    # video sample forward. Nominal frame spacing is 3.4 ms.
    valid_frame &= (target_times - camera_times[frame]) <= 0.020
    sampled_y = y[frame]
    sampled_likelihood = likelihood[frame]
    visible = (
        valid_frame
        & np.isfinite(sampled_y)
        & np.isfinite(sampled_likelihood)
        & (sampled_likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
    )

    categories = np.full(target_times.shape, 3, dtype=np.uint8)
    if np.isfinite(q40):
        categories[visible & (sampled_y < q40)] = 0
        categories[visible & (sampled_y >= q40) & (sampled_y <= q60)] = 1
        categories[visible & (sampled_y > q60)] = 2
    return categories, float(q40), float(q60)


def _trial_choice(instruction: str, outcome: str) -> int:
    # Output order is left, right, no lick.
    if outcome == "ignore":
        return 2
    if outcome == "hit":
        return 0 if instruction == "left" else 1
    if outcome == "miss":
        return 1 if instruction == "left" else 0
    raise ValueError(f"Unexpected outcome {outcome!r}")


def _convert_session(path: Path) -> dict | None:
    with h5py.File(path, "r") as nwb:
        classification = _decode_array(nwb["units/classification"])
        good_rows = np.flatnonzero(classification == "good")
        if good_rows.size == 0:
            return None

        all_go_times = nwb[
            "acquisition/BehavioralEvents/go_start_times/timestamps"
        ][:].astype(np.float64)
        n_source_trials = len(nwb["intervals/trials/id"])
        if len(all_go_times) != n_source_trials:
            raise ValueError(
                f"{path.name}: {len(all_go_times)} go cues for "
                f"{n_source_trials} trials"
            )

        # In seven retained NWBs the behavioral table extends beyond ephys
        # acquisition (usually after it, but once before it). Map the compact
        # unit mask back to the source trials using their common observation
        # interval. In four more files, probe-valid trial masks differ slightly.
        good_trial_matrix = nwb["units/is_good_trials"][good_rows, :]
        n_ephys_trials = good_trial_matrix.shape[1]
        if n_ephys_trials > n_source_trials:
            raise ValueError(f"{path.name}: unit trial mask exceeds trial table")
        if n_ephys_trials == n_source_trials:
            ephys_source_idx = np.arange(n_source_trials)
        else:
            observation_intervals = nwb["units/obs_intervals"][:]
            observation_ends = nwb["units/obs_intervals_index"][:]
            unit_starts = []
            unit_stops = []
            for unit_row in good_rows:
                obs_start = (
                    0 if unit_row == 0 else int(observation_ends[unit_row - 1])
                )
                obs_stop = int(observation_ends[unit_row])
                if obs_stop <= obs_start:
                    raise ValueError(f"{path.name}: good unit has no observations")
                unit_starts.append(observation_intervals[obs_start, 0])
                unit_stops.append(observation_intervals[obs_stop - 1, 1])
            common_start = max(unit_starts)
            common_stop = min(unit_stops)
            ephys_source_idx = np.flatnonzero(
                (all_go_times + OFF_START >= common_start - 1e-6)
                & (all_go_times + OFF_END <= common_stop + 1e-6)
            )
            if len(ephys_source_idx) != n_ephys_trials:
                raise ValueError(
                    f"{path.name}: mapped {len(ephys_source_idx)} source trials "
                    f"to a {n_ephys_trials}-column unit mask"
                )

        common_good_mask = np.all(good_trial_matrix, axis=0)
        common_good_idx = ephys_source_idx[common_good_mask]
        auto_water_all = nwb["intervals/trials/auto_water"][:].astype(bool)
        free_water_all = nwb["intervals/trials/free_water"][:].astype(bool)
        water_mask = auto_water_all[common_good_idx] | free_water_all[common_good_idx]
        valid_trial_idx = common_good_idx[~water_mask]
        if valid_trial_idx.size < 2:
            raise ValueError(f"{path.name}: fewer than two common valid trials")
        go_times = all_go_times[valid_trial_idx]
        n_trials = len(go_times)

        all_spikes = nwb["units/spike_times"][:]
        spike_ends = nwb["units/spike_times_index"][:]
        rates = _bin_good_units(all_spikes, spike_ends, good_rows, go_times)
        del all_spikes
        # One final recorded trial in the release is marked valid in NWB but has
        # no spikes from any cluster. Treat this population-wide dropout as
        # missing acquisition, not simultaneous biological silence.
        population_recorded = np.any(rates != 0, axis=(1, 2))
        n_population_dropouts = int(np.count_nonzero(~population_recorded))
        if n_population_dropouts:
            rates = rates[population_recorded]
            valid_trial_idx = valid_trial_idx[population_recorded]
            go_times = go_times[population_recorded]
            n_trials = len(go_times)

        annotations = _decode_array(nwb["units/anno_name"])[good_rows]
        if np.any(annotations == ""):
            raise ValueError(f"{path.name}: a good unit lacks an Allen annotation")

        sample_starts = nwb[
            "acquisition/BehavioralEvents/sample_start_times/timestamps"
        ][:]
        tone_onset = _tone_onsets(go_times, sample_starts)

        trial_start = nwb["intervals/trials/start_time"][:][valid_trial_idx]
        stim_onset_raw = nwb["intervals/trials/photostim_onset"][:][valid_trial_idx]
        stim_duration_raw = nwb["intervals/trials/photostim_duration"][:][valid_trial_idx]
        stim_onset = np.asarray([_optional_float(x) for x in stim_onset_raw])
        stim_duration = np.asarray([_optional_float(x) for x in stim_duration_raw])

        tongue_path = (
            "acquisition/BehavioralTimeSeries/"
            "Camera0_side_TongueTracking"
        )
        tongue_data = nwb[f"{tongue_path}/data"][:]
        camera_times = nwb[f"{tongue_path}/timestamps"][:]
        tongue_cat, tongue_q40, tongue_q60 = _tongue_categories(
            tongue_data, camera_times, go_times
        )
        del tongue_data, camera_times

        instruction = _decode_array(
            nwb["intervals/trials/trial_instruction"]
        )[valid_trial_idx]
        outcome_text = _decode_array(
            nwb["intervals/trials/outcome"]
        )[valid_trial_idx]
        early_text = _decode_array(
            nwb["intervals/trials/early_lick"]
        )[valid_trial_idx]
        outcome_map = {"ignore": 0, "miss": 1, "hit": 2}

        inputs: list[np.ndarray] = []
        outputs: list[np.ndarray] = []
        for trial in range(n_trials):
            time_from_tone = (
                go_times[trial] + BIN_CENTERS - tone_onset[trial]
            ).astype(np.float32)
            photo_on = np.zeros(N_BINS, dtype=np.float32)
            if np.isfinite(stim_onset[trial]) and np.isfinite(stim_duration[trial]):
                photo_start = trial_start[trial] + stim_onset[trial]
                photo_stop = photo_start + stim_duration[trial]
                absolute_centers = go_times[trial] + BIN_CENTERS
                photo_on[(absolute_centers >= photo_start) & (absolute_centers < photo_stop)] = 1.0
            inputs.append(np.stack((time_from_tone, photo_on), axis=0))

            trial_output = np.empty((4, N_BINS), dtype=np.uint8)
            trial_output[0, :] = _trial_choice(
                instruction[trial], outcome_text[trial]
            )
            trial_output[1, :] = outcome_map[outcome_text[trial]]
            trial_output[2, :] = 1 if early_text[trial] == "early" else 0
            trial_output[3, :] = tongue_cat[trial]
            outputs.append(trial_output)

        subject_id = path.parent.name.removeprefix("sub-")
        return {
            # Each first-axis slice is a C-contiguous (unit,time) ndarray. Keeping
            # these as views avoids an unnecessary second in-memory copy here.
            "neural": [rates[t] for t in range(n_trials)],
            "input": inputs,
            "output": outputs,
            "subject": subject_id,
            "annotations": annotations.tolist(),
            "session_info": {
                "source_file": str(path.relative_to(path.parents[1])),
                "n_source_trials": int(n_source_trials),
                "n_ephys_mask_trials": int(n_ephys_trials),
                "n_common_good_trials": int(len(common_good_idx)),
                "n_excluded_water_trials": int(np.count_nonzero(water_mask)),
                "n_population_recording_dropouts": n_population_dropouts,
                "n_trials": int(n_trials),
                "n_good_units": int(len(good_rows)),
                "tongue_y_40th_percentile": tongue_q40,
                "tongue_y_60th_percentile": tongue_q60,
            },
        }


def convert(data_dir: Path, output_path: Path) -> dict:
    paths = sorted(data_dir.glob("sub-*/*.nwb"))
    if not paths:
        raise FileNotFoundError(f"No NWB files found under {data_dir}")

    converted_sessions = []
    skipped_sessions = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index:3d}/{len(paths)}] {path.name}", flush=True)
        session = _convert_session(path)
        if session is None:
            skipped_sessions.append(path.name)
            print("    skipped: no classifier-labeled good units", flush=True)
        else:
            converted_sessions.append(session)

    subjects = sorted({session["subject"] for session in converted_sessions})
    subject_lookup = {name: i for i, name in enumerate(subjects)}
    brain_regions = sorted(
        {
            annotation
            for session in converted_sessions
            for annotation in session["annotations"]
        }
    )
    region_lookup = {name: i for i, name in enumerate(brain_regions)}

    data = {
        "neural": [session["neural"] for session in converted_sessions],
        "input": [session["input"] for session in converted_sessions],
        "output": [session["output"] for session in converted_sessions],
        "subjects": subjects,
        "subject_idx": np.asarray(
            [subject_lookup[session["subject"]] for session in converted_sessions],
            dtype=np.int32,
        ),
        "brain_regions": brain_regions,
        "brain_region_idx": [
            np.asarray(
                [region_lookup[name] for name in session["annotations"]],
                dtype=np.int32,
            )
            for session in converted_sessions
        ],
        "input_names": ["time from tone onset (s)", "photostimulation on"],
        "output_names": [
            "lick direction choice",
            "outcome",
            "early lick",
            "tongue y-position",
        ],
        "output_values": [
            ["left", "right", "no lick"],
            ["ignore", "miss", "hit"],
            ["no", "yes"],
            [
                "below session 40th percentile",
                "session 40th to 60th percentile",
                "above session 60th percentile",
                "not visible",
            ],
        ],
        "metadata": {
            "task_description": (
                "Auditory delayed-response task: decode lick choice, trial "
                "outcome, early licking, and discretized tongue height from "
                "brain-wide population firing rates."
            ),
            "time_bin_size": 50.0,
            "temporal_alignment_event": "auditory go cue onset",
            "off_start": OFF_START,
            "off_end": OFF_END,
            "neural_measure": "firing rate (spikes/s)",
            "binning": "80 non-overlapping, left-closed 50-ms bins",
            "bin_centers_relative_to_go_s": BIN_CENTERS.tolist(),
            "unit_filter": "NWB units/classification == 'good'",
            "trial_filter": (
                "All trials retained, including photostimulation, early-lick, "
                "and ignored/no-response trials required by decoder variables, "
                "provided every retained unit marks the trial as observed in "
                "NWB units/is_good_trials. Auto-water and free-water trials are "
                "excluded following the source repository; population-wide "
                "zero-spike acquisition dropouts are also excluded."
            ),
            "tongue_visibility_rule": (
                "DLC likelihood >= 0.9 at preceding camera frame; camera gaps "
                ">20 ms are not visible"
            ),
            "tongue_percentile_population": (
                "all DLC-visible side-camera frames within each session"
            ),
            "source_dataset": "Mesoscale Activity Map NWB release",
            "session_info": [
                session["session_info"] for session in converted_sessions
            ],
            "skipped_sessions": skipped_sessions,
        },
    }

    print(f"Writing {output_path} ...", flush=True)
    with output_path.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"Wrote {len(data['neural'])} sessions, "
        f"{sum(len(x) for x in data['neural'])} trials, and "
        f"{sum(len(x) for x in data['brain_region_idx'])} session-units.",
        flush=True,
    )
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument(
        "--output", type=Path, default=Path("/app/converted_data.pkl")
    )
    args = parser.parse_args()
    convert(args.data_dir, args.output)


if __name__ == "__main__":
    main()
