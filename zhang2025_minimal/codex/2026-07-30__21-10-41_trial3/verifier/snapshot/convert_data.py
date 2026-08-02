import argparse
import json
import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from one.api import ONE
from iblatlas.regions import BrainRegions

sys.path.insert(0, str(Path(__file__).resolve().parent / "code" / "ibllib"))
from brainbox.behavior.wheel import interpolate_position, velocity_filtered  # noqa: E402


WINDOW_START = -0.5
WINDOW_END = 1.5
BINSIZE = 0.02
NBINS = int(round((WINDOW_END - WINDOW_START) / BINSIZE))
SPIKE_SORTING_REVISION = "2024-05-06"
TRIAL_NAN_EXCLUDE = (
    "stimOn_times",
    "choice",
    "feedback_times",
    "probabilityLeft",
    "firstMovement_times",
    "feedbackType",
)
GOOD_LABEL_THRESHOLD = 1.0
GREY_REGION_ID = 8
INVALID_REGION_ACRONYMS = {"void", "root", "grey"}
REGION_MAPPING = "Beryl"
MIN_NEURONS_PER_SESSION_REGION = 5
MIN_SESSIONS_PER_REGION = 2


@dataclass
class ProbeInfo:
    probe_name: str
    collection: str
    n_clusters_total: int
    good_cluster_ids: np.ndarray
    cluster_regions: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert IBL BWM data into the decoder training format.")
    parser.add_argument("--cache-dir", type=str, default="/app/data", help="ONE cache directory.")
    parser.add_argument(
        "--release-csv",
        type=str,
        default="/app/code/code_zhang2025/data/bwm_release.csv",
        help="Path to the BWM release CSV with 699 probe insertions.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/app/converted_data.pkl",
        help="Output pickle path.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Optional path to save conversion summary as JSON.",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Optional limit on the number of eligible sessions to convert.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used only for deterministic ordering in any sampled outputs.",
    )
    parser.add_argument(
        "--reprocess-existing-pkl",
        type=str,
        default=None,
        help=(
            "Optional path to an existing converted pickle. When provided, skip raw data loading and "
            "re-apply brain-region mapping/filtering before saving to --output."
        ),
    )
    return parser


def build_region_maps(one: ONE) -> tuple[dict[int, str], set[int]]:
    records = one.alyx.rest("brain-regions", "list", no_cache=True)
    parent = {int(rec["id"]): (None if rec["parent"] is None else int(rec["parent"])) for rec in records}
    id_to_acronym = {int(rec["id"]): str(rec["acronym"]) for rec in records}
    grey_cache: dict[int, bool] = {}

    def is_grey(region_id: int) -> bool:
        if region_id in grey_cache:
            return grey_cache[region_id]
        curr = region_id
        seen = set()
        while curr is not None and curr not in seen:
            if curr == GREY_REGION_ID:
                grey_cache[region_id] = True
                return True
            seen.add(curr)
            curr = parent.get(curr)
        grey_cache[region_id] = False
        return False

    grey_ids = {rid for rid in id_to_acronym if is_grey(rid)}
    return id_to_acronym, grey_ids


def load_trials(one: ONE, eid: str) -> pd.DataFrame:
    trials = one.load_object(eid, "trials", collection="alf").to_df()
    if "intervals_0" not in trials.columns and "intervals" in trials.columns:
        intervals = np.asarray(trials["intervals"].to_list())
        if intervals.ndim == 2 and intervals.shape[1] == 2:
            trials["intervals_0"] = intervals[:, 0]
            trials["intervals_1"] = intervals[:, 1]
    return trials


def make_base_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    mask = np.ones(len(trials), dtype=bool)
    for col in TRIAL_NAN_EXCLUDE:
        mask &= trials[col].notna().to_numpy()
    reaction_time = (trials["firstMovement_times"] - trials["stimOn_times"]).to_numpy()
    mask &= reaction_time >= 0.08
    mask &= reaction_time <= 2.0
    mask &= trials["choice"].to_numpy() != 0
    if "goCue_times" in trials.columns:
        go_cue = trials["goCue_times"].to_numpy()
        feedback = trials["feedback_times"].to_numpy()
        trial_len_ok = np.ones(len(trials), dtype=bool)
        good_len = np.isfinite(go_cue) & np.isfinite(feedback)
        trial_len_ok[good_len] = (feedback[good_len] - go_cue[good_len]) <= 10.0
        mask &= trial_len_ok
    return mask


def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    trial_num = np.zeros(len(prob_left), dtype=np.float32)
    if len(prob_left) == 0:
        return trial_num
    current = 1
    trial_num[0] = current
    for idx in range(1, len(prob_left)):
        prev = prob_left[idx - 1]
        curr = prob_left[idx]
        if np.isfinite(prev) and np.isfinite(curr) and curr == prev:
            current += 1
        else:
            current = 1
        trial_num[idx] = current
    return trial_num


def load_wheel_speed(one: ONE, eid: str) -> tuple[np.ndarray, np.ndarray]:
    wheel = one.load_object(eid, "wheel", collection="alf")
    timestamps = np.asarray(wheel["timestamps"], dtype=np.float64)
    position = np.asarray(wheel["position"], dtype=np.float64)
    interp_pos, interp_time = interpolate_position(timestamps, position, freq=1000)
    velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
    speed = np.abs(np.asarray(velocity, dtype=np.float32))
    return np.asarray(interp_time, dtype=np.float64), speed


def load_whisker_motion_energy(one: ONE, eid: str) -> tuple[str | None, np.ndarray | None, np.ndarray | None]:
    try:
        cam = one.load_object(eid, "leftCamera", attribute=["times", "ROIMotionEnergy"], collection="alf")
        times = np.asarray(cam["times"], dtype=np.float64)
        values = np.asarray(cam["ROIMotionEnergy"], dtype=np.float32)
        if len(times) and len(times) == len(values):
            return "left", times, values
    except Exception:
        pass
    return None, None, None


def interpolate_trial_signal(
    signal_times: np.ndarray,
    signal_values: np.ndarray,
    start_time: float,
    end_time: float,
    binsize: float = BINSIZE,
) -> np.ndarray | None:
    start_idx = np.searchsorted(signal_times, start_time, side="right")
    end_idx = np.searchsorted(signal_times, end_time, side="left")
    times = signal_times[start_idx:end_idx]
    values = signal_values[start_idx:end_idx]
    if len(times) == 0:
        return None
    finite = np.isfinite(times) & np.isfinite(values)
    times = times[finite]
    values = values[finite]
    if len(times) < 2:
        return None
    if abs(start_time - times[0]) > binsize:
        return None
    if abs(end_time - times[-1]) > binsize:
        return None
    sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
    interp_vals = np.interp(sample_times, times, values)
    if not np.all(np.isfinite(interp_vals)):
        return None
    return np.asarray(interp_vals, dtype=np.float32)


def load_probe_info(
    one: ONE,
    eid: str,
    probe_name: str,
    id_to_acronym: dict[int, str],
    grey_ids: set[int],
) -> ProbeInfo | None:
    collection = f"alf/{probe_name}/pykilosort"
    metrics_path = one.load_dataset(
        eid,
        "clusters.metrics.pqt",
        collection=collection,
        revision=SPIKE_SORTING_REVISION,
        download_only=True,
    )
    metrics = pd.read_parquet(metrics_path)
    cluster_channels = np.asarray(
        one.load_dataset(
            eid,
            "clusters.channels.npy",
            collection=collection,
            revision=SPIKE_SORTING_REVISION,
        ),
        dtype=np.int64,
    )
    channel_region_ids = np.asarray(
        one.load_dataset(
            eid,
            "channels.brainLocationIds_ccf_2017.npy",
            collection=collection,
            revision=SPIKE_SORTING_REVISION,
        ),
        dtype=np.int64,
    )
    if len(cluster_channels) != len(metrics):
        raise ValueError(f"{eid} {probe_name}: mismatch between cluster channels and metrics rows")

    valid_channel_idx = (cluster_channels >= 0) & (cluster_channels < len(channel_region_ids))
    region_ids = np.full(len(cluster_channels), -1, dtype=np.int64)
    region_ids[valid_channel_idx] = channel_region_ids[cluster_channels[valid_channel_idx]]
    region_acronyms = np.array([id_to_acronym.get(int(rid), "void") for rid in region_ids], dtype=object)

    good = metrics["label"].to_numpy(dtype=float) >= GOOD_LABEL_THRESHOLD
    good &= np.isin(region_ids, list(grey_ids))
    good &= ~np.isin(region_acronyms, list(INVALID_REGION_ACRONYMS))
    good_cluster_ids = np.flatnonzero(good).astype(np.int32)
    if len(good_cluster_ids) == 0:
        return None

    return ProbeInfo(
        probe_name=probe_name,
        collection=collection,
        n_clusters_total=len(metrics),
        good_cluster_ids=good_cluster_ids,
        cluster_regions=region_acronyms[good_cluster_ids].astype(object),
    )


def bin_probe_spikes(
    one: ONE,
    eid: str,
    probe: ProbeInfo,
    trial_starts: np.ndarray,
) -> np.ndarray:
    spikes_times_path = one.load_dataset(
        eid,
        "spikes.times.npy",
        collection=probe.collection,
        revision=SPIKE_SORTING_REVISION,
        download_only=True,
    )
    spikes_clusters_path = one.load_dataset(
        eid,
        "spikes.clusters.npy",
        collection=probe.collection,
        revision=SPIKE_SORTING_REVISION,
        download_only=True,
    )
    spike_times = np.load(spikes_times_path, mmap_mode="r")
    spike_clusters = np.load(spikes_clusters_path, mmap_mode="r")

    keep_bool = np.zeros(probe.n_clusters_total, dtype=bool)
    keep_bool[probe.good_cluster_ids] = True
    spike_keep = keep_bool[spike_clusters]
    kept_times = np.asarray(spike_times[spike_keep], dtype=np.float32)
    kept_clusters_raw = np.asarray(spike_clusters[spike_keep], dtype=np.int32)

    local_map = np.full(keep_bool.shape[0], -1, dtype=np.int32)
    local_map[probe.good_cluster_ids] = np.arange(len(probe.good_cluster_ids), dtype=np.int32)
    kept_clusters = local_map[kept_clusters_raw]

    probe_counts = np.zeros((len(trial_starts), len(probe.good_cluster_ids), NBINS), dtype=np.uint16)

    for trial_idx, start_time in enumerate(trial_starts):
        end_time = start_time + (WINDOW_END - WINDOW_START)
        start_idx = np.searchsorted(kept_times, start_time, side="left")
        end_idx = np.searchsorted(kept_times, end_time, side="left")
        if end_idx <= start_idx:
            continue
        trial_times = kept_times[start_idx:end_idx]
        trial_clusters = kept_clusters[start_idx:end_idx]
        bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)
        in_bounds = (bins >= 0) & (bins < NBINS) & (trial_clusters >= 0)
        if not np.any(in_bounds):
            continue
        np.add.at(
            probe_counts[trial_idx],
            (trial_clusters[in_bounds], bins[in_bounds]),
            1,
        )

    return probe_counts


def process_session(
    one: ONE,
    session_rows: pd.DataFrame,
    id_to_acronym: dict[int, str],
    grey_ids: set[int],
) -> dict | None:
    eid = str(session_rows["eid"].iloc[0])
    subject = str(session_rows["subject"].iloc[0])
    lab = str(session_rows["lab"].iloc[0])
    date = str(session_rows["date"].iloc[0])

    trials = load_trials(one, eid)
    base_mask = make_base_trial_mask(trials)
    block_trial_num = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy(dtype=float))

    try:
        wheel_times, wheel_speed = load_wheel_speed(one, eid)
    except Exception as exc:
        print(f"Skip session {eid}: wheel load failed: {exc}")
        return None

    camera_view, whisker_times, whisker_me = load_whisker_motion_energy(one, eid)
    if whisker_times is None or whisker_me is None:
        print(f"Skip session {eid}: no whisker motion energy trace available.")
        return None

    kept_trial_indices = []
    input_trials = []
    choice_trials = []
    prior_trials = []
    wheel_trials = []
    whisker_trials = []

    relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)

    for trial_idx in np.flatnonzero(base_mask):
        stim_on = float(trials.iloc[trial_idx]["stimOn_times"])
        start_time = stim_on + WINDOW_START
        end_time = stim_on + WINDOW_END
        wheel_interp = interpolate_trial_signal(wheel_times, wheel_speed, start_time, end_time)
        whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
        if wheel_interp is None or whisker_interp is None:
            continue

        choice_val = float(trials.iloc[trial_idx]["choice"])
        prob_left = float(trials.iloc[trial_idx]["probabilityLeft"])
        if choice_val == 1:
            choice_out = 0
        elif choice_val == -1:
            choice_out = 1
        else:
            continue

        if prob_left == 0.2:
            prior_out = 0
        elif prob_left == 0.5:
            prior_out = 1
        elif prob_left == 0.8:
            prior_out = 2
        else:
            continue

        trial_num = float(block_trial_num[trial_idx])
        input_trial = np.vstack(
            [
                relative_time,
                np.full(NBINS, trial_num, dtype=np.float32),
            ]
        ).astype(np.float32, copy=False)

        kept_trial_indices.append(trial_idx)
        input_trials.append(input_trial)
        choice_trials.append(choice_out)
        prior_trials.append(prior_out)
        wheel_trials.append(wheel_interp)
        whisker_trials.append(whisker_interp)

    if len(kept_trial_indices) < 2:
        print(f"Skip session {eid}: only {len(kept_trial_indices)} trials after alignment and behavior checks.")
        return None

    probes = []
    for _, row in session_rows.sort_values("probe_name").iterrows():
        probe = load_probe_info(one, eid, str(row["probe_name"]), id_to_acronym, grey_ids)
        if probe is not None:
            probes.append(probe)
    if not probes:
        print(f"Skip session {eid}: no good grey-matter units after QC.")
        return None

    total_neurons = sum(len(probe.good_cluster_ids) for probe in probes)
    trial_starts = trials.iloc[kept_trial_indices]["stimOn_times"].to_numpy(dtype=np.float64) + WINDOW_START
    session_counts = np.zeros((len(kept_trial_indices), total_neurons, NBINS), dtype=np.uint16)
    session_regions = []

    offset = 0
    for probe in probes:
        probe_counts = bin_probe_spikes(one, eid, probe, trial_starts)
        end_offset = offset + probe_counts.shape[1]
        session_counts[:, offset:end_offset, :] = probe_counts
        session_regions.extend(probe.cluster_regions.tolist())
        offset = end_offset

    nonzero_trial_mask = session_counts.reshape(session_counts.shape[0], -1).sum(axis=1) > 0
    if nonzero_trial_mask.sum() < 2:
        print(f"Skip session {eid}: fewer than 2 nonzero neural trials after binning.")
        return None
    if not np.all(nonzero_trial_mask):
        session_counts = session_counts[nonzero_trial_mask]
        input_trials = [x for x, keep in zip(input_trials, nonzero_trial_mask) if keep]
        choice_trials = [x for x, keep in zip(choice_trials, nonzero_trial_mask) if keep]
        prior_trials = [x for x, keep in zip(prior_trials, nonzero_trial_mask) if keep]
        wheel_trials = [x for x, keep in zip(wheel_trials, nonzero_trial_mask) if keep]
        whisker_trials = [x for x, keep in zip(whisker_trials, nonzero_trial_mask) if keep]

    neural_trials = [session_counts[idx].astype(np.float16, copy=True) for idx in range(session_counts.shape[0])]

    return {
        "eid": eid,
        "subject": subject,
        "lab": lab,
        "date": date,
        "camera_view": camera_view,
        "probe_names": [probe.probe_name for probe in probes],
        "n_trials_total": int(len(trials)),
        "n_trials_base_valid": int(base_mask.sum()),
        "n_trials_kept": int(len(kept_trial_indices)),
        "n_neurons": int(total_neurons),
        "brain_regions": session_regions,
        "neural_trials": neural_trials,
        "input_trials": input_trials,
        "choice_trials": choice_trials,
        "prior_trials": prior_trials,
        "wheel_trials": wheel_trials,
        "whisker_trials": whisker_trials,
    }


def discretize(values: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    low_edge, high_edge = edges
    return np.digitize(values, bins=np.array([low_edge, high_edge], dtype=np.float32), right=False).astype(np.int8)


def build_output_trials(
    session_result: dict,
    wheel_edges: tuple[float, float],
    whisker_edges: tuple[float, float],
) -> list[np.ndarray]:
    outputs = []
    for choice_out, prior_out, wheel_vals, whisker_vals in zip(
        session_result["choice_trials"],
        session_result["prior_trials"],
        session_result["wheel_trials"],
        session_result["whisker_trials"],
    ):
        output_trial = np.vstack(
            [
                np.full(NBINS, choice_out, dtype=np.int8),
                np.full(NBINS, prior_out, dtype=np.int8),
                discretize(wheel_vals, wheel_edges),
                discretize(whisker_vals, whisker_edges),
            ]
        )
        outputs.append(output_trial)
    return outputs


def build_data_from_session_results(session_results: list[dict], release_df: pd.DataFrame, args: argparse.Namespace) -> dict:
    wheel_all = np.concatenate([np.concatenate(sess["wheel_trials"]) for sess in session_results]).astype(np.float32)
    whisker_all = np.concatenate([np.concatenate(sess["whisker_trials"]) for sess in session_results]).astype(np.float32)
    wheel_edges = tuple(np.quantile(wheel_all, [1 / 3, 2 / 3]).astype(np.float32).tolist())
    whisker_edges = tuple(np.quantile(whisker_all, [1 / 3, 2 / 3]).astype(np.float32).tolist())

    subjects = []
    subject_to_idx = {}
    brain_regions = []
    region_to_idx = {}

    neural = []
    input_data = []
    output_data = []
    subject_idx = []
    brain_region_idx = []
    session_info = []

    for sess in session_results:
        if sess["subject"] not in subject_to_idx:
            subject_to_idx[sess["subject"]] = len(subjects)
            subjects.append(sess["subject"])

        session_region_idx = np.empty(len(sess["brain_regions"]), dtype=np.int32)
        for neuron_idx, region_name in enumerate(sess["brain_regions"]):
            if region_name not in region_to_idx:
                region_to_idx[region_name] = len(brain_regions)
                brain_regions.append(region_name)
            session_region_idx[neuron_idx] = region_to_idx[region_name]

        neural.append(sess["neural_trials"])
        input_data.append(sess["input_trials"])
        output_data.append(build_output_trials(sess, wheel_edges, whisker_edges))
        subject_idx.append(subject_to_idx[sess["subject"]])
        brain_region_idx.append(session_region_idx)
        session_info.append(
            {
                "eid": sess["eid"],
                "subject": sess["subject"],
                "lab": sess["lab"],
                "date": sess["date"],
                "camera_view": sess["camera_view"],
                "probe_names": sess["probe_names"],
                "n_trials_total": sess["n_trials_total"],
                "n_trials_base_valid": sess["n_trials_base_valid"],
                "n_trials_kept": sess["n_trials_kept"],
                "n_neurons": sess["n_neurons"],
            }
        )

    data = {
        "neural": neural,
        "input": input_data,
        "output": output_data,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int32),
        "brain_regions": brain_regions,
        "brain_region_idx": brain_region_idx,
        "input_names": ["time_since_stimulus_onset_s", "trial_number_in_block"],
        "output_names": [
            "choice",
            "prior_probability_left",
            "wheel_speed_bin",
            "whisker_motion_energy_bin",
        ],
        "output_values": [
            ["left", "right"],
            ["0.2", "0.5", "0.8"],
            ["low", "medium", "high"],
            ["low", "medium", "high"],
        ],
        "metadata": {
            "task_description": (
                "IBL biased visual decision task; decode choice, block prior, wheel speed, "
                "and whisker motion energy from stimulus-aligned population activity."
            ),
            "time_bin_size": float(BINSIZE * 1000.0),
            "temporal_alignment_event": "stimulus onset",
            "off_start": float(WINDOW_START),
            "off_end": float(WINDOW_END),
            "source_release_csv": str(args.release_csv),
            "source_cache_dir": str(args.cache_dir),
            "source_papers": [
                "A brain-wide map of neural activity during complex behaviour",
                "Exploiting correlations across trials and behavioral sessions to improve neural decoding",
            ],
            "session_info": session_info,
            "trial_filters": {
                "required_non_nan_columns": list(TRIAL_NAN_EXCLUDE),
                "reaction_time_range_s": [0.08, 2.0],
                "exclude_no_choice": True,
                "max_trial_length_s": 10.0,
            },
            "neuron_filters": {
                "good_unit_label_threshold": GOOD_LABEL_THRESHOLD,
                "grey_matter_only": True,
                "excluded_region_acronyms": sorted(INVALID_REGION_ACRONYMS),
                "brain_region_mapping": REGION_MAPPING,
                "min_neurons_per_session_region": MIN_NEURONS_PER_SESSION_REGION,
                "min_sessions_per_region": MIN_SESSIONS_PER_REGION,
            },
            "behavior_processing": {
                "wheel_preprocessing": "1000 Hz interpolation followed by Butterworth-filtered velocity; speed is abs(velocity)",
                "whisker_view_selection": "left camera only",
                "dynamic_signal_interpolation": "linear interpolation to stimulus-aligned 20 ms bin right edges",
                "wheel_speed_bin_edges": [float(wheel_edges[0]), float(wheel_edges[1])],
                "whisker_motion_energy_bin_edges": [float(whisker_edges[0]), float(whisker_edges[1])],
            },
        },
    }

    return data


def apply_region_filters(data: dict) -> tuple[dict, dict]:
    brainreg = BrainRegions()
    old_region_names = np.asarray(data["brain_regions"], dtype=object)
    old_idx_per_session = data["brain_region_idx"]
    session_info = list(data["metadata"]["session_info"])

    beryl_names_per_session = []
    session_region_keep = []
    sessions_with_region = {}

    for session_idx, region_idx in enumerate(old_idx_per_session):
        session_old_names = old_region_names[region_idx]
        session_beryl = np.asarray(brainreg.acronym2acronym(session_old_names, mapping=REGION_MAPPING), dtype=object)
        invalid = np.isin(session_beryl, ["root", "void"])
        valid_names = session_beryl[~invalid]
        keep_mask = np.zeros(len(session_beryl), dtype=bool)
        if len(valid_names):
            unique_names, counts = np.unique(valid_names, return_counts=True)
            valid_regions = set(unique_names[counts >= MIN_NEURONS_PER_SESSION_REGION].tolist())
            keep_mask = np.isin(session_beryl, list(valid_regions)) & ~invalid
            for region_name in np.unique(session_beryl[keep_mask]):
                sessions_with_region.setdefault(str(region_name), set()).add(session_idx)
        beryl_names_per_session.append(session_beryl)
        session_region_keep.append(keep_mask)

    globally_valid_regions = {
        region_name
        for region_name, session_ids in sessions_with_region.items()
        if len(session_ids) >= MIN_SESSIONS_PER_REGION
    }

    new_neural = []
    new_input = []
    new_output = []
    new_subject_idx = []
    new_session_info = []
    new_region_names_per_session = []

    for session_idx, keep_mask in enumerate(session_region_keep):
        final_keep = keep_mask & np.isin(beryl_names_per_session[session_idx], list(globally_valid_regions))
        if not np.any(final_keep):
            continue

        filtered_trials = [
            trial[final_keep].astype(np.float16, copy=False)
            for trial in data["neural"][session_idx]
        ]
        if filtered_trials[0].shape[0] == 0:
            continue

        new_neural.append(filtered_trials)
        new_input.append(data["input"][session_idx])
        new_output.append(data["output"][session_idx])
        new_subject_idx.append(int(data["subject_idx"][session_idx]))
        sess_info = dict(session_info[session_idx])
        sess_info["n_neurons"] = int(final_keep.sum())
        new_session_info.append(sess_info)
        new_region_names_per_session.append(beryl_names_per_session[session_idx][final_keep].astype(object))

    if not new_neural:
        raise RuntimeError("Region filtering removed all sessions.")

    used_subject_idx = np.asarray(new_subject_idx, dtype=np.int32)
    used_subjects = sorted(set(int(x) for x in used_subject_idx.tolist()))
    subject_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(used_subjects)}
    new_subject_names = [data["subjects"][old_idx] for old_idx in used_subjects]
    remapped_subject_idx = np.asarray([subject_remap[int(x)] for x in used_subject_idx], dtype=np.int32)

    new_brain_regions = []
    region_to_idx = {}
    new_brain_region_idx = []
    for region_names in new_region_names_per_session:
        session_idx_arr = np.empty(len(region_names), dtype=np.int32)
        for neuron_idx, region_name in enumerate(region_names.tolist()):
            if region_name not in region_to_idx:
                region_to_idx[region_name] = len(new_brain_regions)
                new_brain_regions.append(region_name)
            session_idx_arr[neuron_idx] = region_to_idx[region_name]
        new_brain_region_idx.append(session_idx_arr)

    data["neural"] = new_neural
    data["input"] = new_input
    data["output"] = new_output
    data["subjects"] = new_subject_names
    data["subject_idx"] = remapped_subject_idx
    data["brain_regions"] = new_brain_regions
    data["brain_region_idx"] = new_brain_region_idx
    data["metadata"]["session_info"] = new_session_info
    data["metadata"]["neuron_filters"]["brain_region_mapping"] = REGION_MAPPING
    data["metadata"]["neuron_filters"]["min_neurons_per_session_region"] = MIN_NEURONS_PER_SESSION_REGION
    data["metadata"]["neuron_filters"]["min_sessions_per_region"] = MIN_SESSIONS_PER_REGION
    data["metadata"]["region_filter_summary"] = {
        "mapping": REGION_MAPPING,
        "min_neurons_per_session_region": MIN_NEURONS_PER_SESSION_REGION,
        "min_sessions_per_region": MIN_SESSIONS_PER_REGION,
        "session_count_before_region_filter": int(len(old_idx_per_session)),
        "session_count_after_region_filter": int(len(new_neural)),
        "brain_region_count_before_region_filter": int(len(old_region_names)),
        "brain_region_count_after_region_filter": int(len(new_brain_regions)),
    }

    filter_stats = {
        "session_count_before_region_filter": int(len(old_idx_per_session)),
        "session_count_after_region_filter": int(len(new_neural)),
        "brain_region_count_before_region_filter": int(len(old_region_names)),
        "brain_region_count_after_region_filter": int(len(new_brain_regions)),
    }
    return data, filter_stats


def build_summary(data: dict, release_df: pd.DataFrame) -> dict:
    session_info = data["metadata"]["session_info"]
    summary = {
        "release_probe_count": int(len(release_df)),
        "release_session_count": int(release_df["eid"].nunique()),
        "converted_session_count": int(len(data["neural"])),
        "converted_subject_count": int(len(data["subjects"])),
        "converted_brain_region_count": int(len(data["brain_regions"])),
        "total_trial_count": int(sum(len(x) for x in data["neural"])),
        "total_neuron_count": int(sum(len(x) for x in data["brain_region_idx"])),
        "wheel_speed_bin_edges": list(data["metadata"]["behavior_processing"]["wheel_speed_bin_edges"]),
        "whisker_motion_energy_bin_edges": list(data["metadata"]["behavior_processing"]["whisker_motion_energy_bin_edges"]),
        "camera_view_counts": {
            str(k): int(v)
            for k, v in pd.Series([sess["camera_view"] for sess in session_info]).value_counts().sort_index().items()
        },
        "n_trials_per_session": [int(len(x)) for x in data["neural"]],
        "n_neurons_per_session": [int(len(x)) for x in data["brain_region_idx"]],
    }
    if "region_filter_summary" in data["metadata"]:
        summary["region_filter_summary"] = dict(data["metadata"]["region_filter_summary"])
    return summary


def convert(args: argparse.Namespace) -> tuple[dict, dict]:
    np.random.seed(args.seed)

    release_df = pd.read_csv(args.release_csv)
    release_df["eid"] = release_df["eid"].astype(str)
    release_df["probe_name"] = release_df["probe_name"].astype(str)
    release_df["subject"] = release_df["subject"].astype(str)
    release_df["lab"] = release_df["lab"].astype(str)
    release_df["date"] = release_df["date"].astype(str)

    if args.reprocess_existing_pkl:
        with Path(args.reprocess_existing_pkl).open("rb") as f:
            data = pickle.load(f)
        data, filter_stats = apply_region_filters(data)
        print(
            "Reprocessed existing pickle with region filters: "
            f"{filter_stats['session_count_before_region_filter']} -> {filter_stats['session_count_after_region_filter']} sessions, "
            f"{filter_stats['brain_region_count_before_region_filter']} -> {filter_stats['brain_region_count_after_region_filter']} regions"
        )
        summary = build_summary(data, release_df)
        return data, summary

    warnings.filterwarnings("ignore", message="Multiple revisions:")
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        cache_dir=args.cache_dir,
        silent=True,
    )
    id_to_acronym, grey_ids = build_region_maps(one)

    grouped = list(release_df.groupby("eid", sort=False))
    print(f"Release probes: {len(release_df)}")
    print(f"Release sessions: {len(grouped)}")

    session_results = []
    total_seen = 0
    for session_idx, (eid, session_rows) in enumerate(grouped, start=1):
        print("=" * 80)
        print(f"[{session_idx}/{len(grouped)}] Processing session {eid}")
        try:
            result = process_session(one, session_rows, id_to_acronym, grey_ids)
        except Exception as exc:
            print(f"Skip session {eid}: unexpected error: {exc}")
            result = None
        if result is None:
            continue
        session_results.append(result)
        total_seen += 1
        print(
            f"Kept {result['n_trials_kept']} trials, {result['n_neurons']} neurons, "
            f"camera={result['camera_view']}, subject={result['subject']}"
        )
        if args.max_sessions is not None and total_seen >= args.max_sessions:
            break

    if not session_results:
        raise RuntimeError("No sessions were converted successfully.")

    data = build_data_from_session_results(session_results, release_df, args)
    data, filter_stats = apply_region_filters(data)
    print(
        "Applied region filters after conversion: "
        f"{filter_stats['session_count_before_region_filter']} -> {filter_stats['session_count_after_region_filter']} sessions, "
        f"{filter_stats['brain_region_count_before_region_filter']} -> {filter_stats['brain_region_count_after_region_filter']} regions"
    )
    summary = build_summary(data, release_df)

    return data, summary


def main() -> None:
    args = build_parser().parse_args()
    data, summary = convert(args)

    output_path = Path(args.output)
    with output_path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("=" * 80)
    print(f"Saved converted dataset to {output_path}")
    print(json.dumps(summary, indent=2))

    if args.summary_json:
        summary_path = Path(args.summary_json)
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved conversion summary to {summary_path}")


if __name__ == "__main__":
    main()
