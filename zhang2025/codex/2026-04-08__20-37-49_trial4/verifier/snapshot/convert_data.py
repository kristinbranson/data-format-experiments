#!/usr/bin/env python3
import argparse
import os
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from one.api import ONE
from iblatlas.regions import BrainRegions
from iblutil.numerical import bincount2D


ROOT = Path(__file__).resolve().parent
IBLLIB_ROOT = ROOT / "code" / "ibllib"
if str(IBLLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(IBLLIB_ROOT))

from brainbox.behavior.wheel import interpolate_position, velocity_filtered


READONLY_CACHE = ROOT / "data" / "one_cache"
WRITABLE_CACHE = ROOT / "cache" / "one_cache"
RELEASE_CSV = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"
SESSION_RECORD_CACHE = ROOT / "cache" / "session_records"

ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.5, 1.5)
BIN_SIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BIN_SIZE))
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)


@dataclass
class SessionRecord:
    eid: str
    subject: str
    lab: str
    neural: list
    input_trials: list
    choice_codes: list
    prior_codes: list
    wheel_values: list
    whisker_values: list
    cluster_regions_beryl: np.ndarray
    trial_indices: np.ndarray
    trial_numbers_in_block: np.ndarray
    diagnostic: dict | None


def parse_args():
    parser = argparse.ArgumentParser(description="Convert IBL brain-wide map data to decoder format.")
    parser.add_argument("outpicklefile", type=str, help="Output pickle path.")
    parser.add_argument("--full", action="store_true", help="Process the full release. Default behavior.")
    parser.add_argument("--sample", action="store_true", help="Process exactly 2 successful sessions.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum worker processes to use in full mode (default: 4).",
    )
    parser.add_argument(
        "--from-session-cache",
        action="store_true",
        help="Build the final dataset from cached per-session records in cache/session_records.",
    )
    parser.add_argument(
        "--resume-session-cache",
        action="store_true",
        help="In full mode, keep existing cache/session_records/*.pkl files and only process missing sessions.",
    )
    parser.add_argument(
        "--fill-session-cache-only",
        action="store_true",
        help="Process sessions into cache/session_records and exit without building the final dataset.",
    )
    parser.add_argument(
        "--repack-session-cache",
        action="store_true",
        help="Rewrite cached session records with compact dtypes and exit without building the final dataset.",
    )
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing plots for up to 2 processed sessions as processing_<session_id>.png.",
    )
    return parser.parse_args()


def build_one():
    WRITABLE_CACHE.mkdir(parents=True, exist_ok=True)
    return ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        cache_dir=WRITABLE_CACHE,
    )


def load_release_sessions():
    bwm = pd.read_csv(RELEASE_CSV, index_col=0)
    session_rows = (
        bwm.drop_duplicates("eid")[["eid", "subject", "lab", "date", "session_number"]]
        .reset_index(drop=True)
    )
    probe_rows = {
        eid: grp[["pid", "probe_name"]].reset_index(drop=True)
        for eid, grp in bwm.groupby("eid", sort=False)
    }
    return session_rows, probe_rows


def session_rel_path(row) -> Path:
    return Path(row.lab) / "Subjects" / row.subject / str(row.date) / f"{int(row.session_number):03d}"


def locate_dataset(row, relative_glob: str):
    rel = session_rel_path(row)
    for root in (READONLY_CACHE, WRITABLE_CACHE):
        base = root / rel
        matches = sorted(base.glob(relative_glob))
        if matches:
            return matches[-1]
    return None


def ensure_dataset_path_any(one: ONE, row, relative_globs: list[str], dataset_name: str, collection: str):
    for relative_glob in relative_globs:
        path = locate_dataset(row, relative_glob)
        if path is not None:
            return path
    one.load_dataset(row.eid, dataset_name, collection=collection, download_only=True)
    for relative_glob in relative_globs:
        path = locate_dataset(row, relative_glob)
        if path is not None:
            return path
    raise FileNotFoundError(
        f"Missing dataset after download attempt: eid={row.eid} dataset={dataset_name} collection={collection}"
    )


def ensure_dataset_path(one: ONE, row, relative_glob: str, dataset_name: str, collection: str):
    return ensure_dataset_path_any(one, row, [relative_glob], dataset_name, collection)


def build_trial_mask(
    trials_df: pd.DataFrame,
    min_rt=0.08,
    max_rt=2.0,
    nan_exclude="default",
    min_trial_len=None,
    max_trial_len=10.0,
    exclude_unbiased=False,
    exclude_nochoice=True,
):
    if nan_exclude == "default":
        nan_exclude = [
            "stimOn_times",
            "choice",
            "feedback_times",
            "probabilityLeft",
            "firstMovement_times",
            "feedbackType",
        ]

    if min_rt is not None:
        query = f"(firstMovement_times - stimOn_times < {min_rt})"
    else:
        query = ""
    if max_rt is not None:
        query += f" | (firstMovement_times - stimOn_times > {max_rt})"
    if min_trial_len is not None:
        query += f" | (feedback_times - goCue_times < {min_trial_len})"
    if max_trial_len is not None:
        query += f" | (feedback_times - goCue_times > {max_trial_len})"
    for event in nan_exclude:
        query += f" | {event}.isnull()"
    if exclude_unbiased:
        query += " | (probabilityLeft == 0.5)"
    if exclude_nochoice:
        query += " | (choice == 0)"
    if min_rt is None:
        query = query[3:]
    return ~trials_df.eval(query)


def compute_trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    out = np.zeros(len(probability_left), dtype=np.float32)
    prev = None
    count = 0
    for idx, value in enumerate(probability_left):
        if np.isnan(value):
            out[idx] = np.nan
            prev = np.nan
            count = 0
            continue
        if prev is None or np.isnan(prev) or not np.isclose(prev, value):
            count = 1
        else:
            count += 1
        out[idx] = count
        prev = value
    return out


def map_choice(choice_value: float) -> int:
    if np.isclose(choice_value, 1.0):
        return 0
    if np.isclose(choice_value, -1.0):
        return 1
    raise ValueError(f"Unexpected choice value {choice_value}")


def map_prior(prob_left: float) -> int:
    if np.isclose(prob_left, 0.2):
        return 0
    if np.isclose(prob_left, 0.5):
        return 1
    if np.isclose(prob_left, 0.8):
        return 2
    raise ValueError(f"Unexpected probabilityLeft value {prob_left}")


def load_trials_table(one: ONE, row):
    path = ensure_dataset_path(one, row, "alf/#*/_ibl_trials.table.pqt", "_ibl_trials.table.pqt", "alf")
    trials_df = pd.read_parquet(path)
    expected_cols = [
        "goCue_times",
        "response_times",
        "choice",
        "stimOn_times",
        "contrastLeft",
        "contrastRight",
        "probabilityLeft",
        "feedback_times",
        "feedbackType",
        "rewardVolume",
        "firstMovement_times",
    ]
    missing = [col for col in expected_cols if col not in trials_df.columns]
    if missing:
        raise ValueError(f"Missing expected trial columns: {missing}")
    return trials_df


def load_wheel_speed(one: ONE, row):
    ts_path = ensure_dataset_path_any(
        one,
        row,
        ["alf/_ibl_wheel.timestamps.npy", "alf/#*/_ibl_wheel.timestamps.npy"],
        "_ibl_wheel.timestamps.npy",
        "alf",
    )
    pos_path = ensure_dataset_path_any(
        one,
        row,
        ["alf/_ibl_wheel.position.npy", "alf/#*/_ibl_wheel.position.npy"],
        "_ibl_wheel.position.npy",
        "alf",
    )
    timestamps = np.load(ts_path)
    position = np.load(pos_path)
    pos_interp, t_interp = interpolate_position(timestamps, position, freq=1000)
    vel, _ = velocity_filtered(pos_interp, 1000)
    return {"times": t_interp.astype(np.float32), "values": np.abs(vel).astype(np.float32)}


def load_whisker_motion_energy(one: ONE, row):
    sides = [
        ("left", "_ibl_leftCamera.times.npy", "leftCamera.ROIMotionEnergy.npy"),
        ("right", "_ibl_rightCamera.times.npy", "rightCamera.ROIMotionEnergy.npy"),
    ]
    errors = []
    for side, times_name, energy_name in sides:
        try:
            times_path = ensure_dataset_path_any(
                one,
                row,
                [f"alf/{times_name}", f"alf/#*/{times_name}"],
                times_name,
                "alf",
            )
            energy_path = ensure_dataset_path(one, row, f"alf/#*/{energy_name}", energy_name, "alf")
            times = np.load(times_path).astype(np.float32)
            values = np.load(energy_path).astype(np.float32)
            return {"times": times, "values": values}, side
        except Exception as exc:
            errors.append(f"{side}: {exc}")
            continue
    raise FileNotFoundError(f"No whisker motion energy trace available for session {row.eid} ({'; '.join(errors)})")


def load_probe_data(one: ONE, row, probe_name: str, brain_regions: BrainRegions):
    collection = f"alf/{probe_name}/pykilosort"
    times_path = ensure_dataset_path(one, row, f"{collection}/#*/spikes.times.npy", "spikes.times.npy", collection)
    clu_path = ensure_dataset_path(
        one, row, f"{collection}/#*/spikes.clusters.npy", "spikes.clusters.npy", collection
    )
    cluster_channels_path = ensure_dataset_path(
        one, row, f"{collection}/#*/clusters.channels.npy", "clusters.channels.npy", collection
    )
    cluster_depths_path = ensure_dataset_path(
        one, row, f"{collection}/#*/clusters.depths.npy", "clusters.depths.npy", collection
    )
    cluster_metrics_path = ensure_dataset_path(
        one, row, f"{collection}/#*/clusters.metrics.pqt", "clusters.metrics.pqt", collection
    )
    channel_regions_path = ensure_dataset_path(
        one,
        row,
        f"{collection}/#*/channels.brainLocationIds_ccf_2017.npy",
        "channels.brainLocationIds_ccf_2017.npy",
        collection,
    )

    spike_times = np.load(times_path).astype(np.float32)
    spike_clusters = np.load(clu_path).astype(np.int32)
    cluster_channels = np.load(cluster_channels_path).astype(np.int64)
    cluster_depths = np.load(cluster_depths_path).astype(np.float32)
    cluster_metrics = pd.read_parquet(cluster_metrics_path)
    if "cluster_id" in cluster_metrics.columns:
        cluster_metrics = cluster_metrics.sort_values("cluster_id").reset_index(drop=True)
    channel_region_ids = np.load(channel_regions_path).astype(np.int64)

    nclusters = len(cluster_channels)
    if len(cluster_metrics) != nclusters:
        raise ValueError(
            f"Cluster metadata length mismatch for {row.eid} {probe_name}: "
            f"{len(cluster_metrics)} metrics rows vs {nclusters} channels"
        )

    cluster_region_ids = channel_region_ids[cluster_channels]
    cluster_acronyms = brain_regions.id2acronym(cluster_region_ids)
    clusters_df = cluster_metrics.copy()
    clusters_df["channels"] = cluster_channels
    clusters_df["depths"] = cluster_depths
    clusters_df["acronym"] = cluster_acronyms

    if "label" not in clusters_df.columns:
        raise ValueError(f"Missing cluster quality label for {row.eid} {probe_name}")

    good_mask = clusters_df["label"].fillna(0).to_numpy() >= 1
    if "cluster_id" in clusters_df.columns:
        good_cluster_ids = clusters_df.loc[good_mask, "cluster_id"].to_numpy(dtype=np.int64)
    else:
        good_cluster_ids = np.flatnonzero(good_mask).astype(np.int64)
    if good_cluster_ids.size == 0:
        return None, None

    spike_keep = np.isin(spike_clusters, good_cluster_ids)
    spike_times = spike_times[spike_keep]
    spike_clusters = spike_clusters[spike_keep]
    remap = np.full(int(np.max(good_cluster_ids)) + 1, -1, dtype=np.int32)
    remap[good_cluster_ids] = np.arange(good_cluster_ids.size, dtype=np.int32)
    spike_clusters = remap[spike_clusters]
    clusters_df = clusters_df.loc[good_mask].reset_index(drop=True)

    return {"times": spike_times, "clusters": spike_clusters}, clusters_df


def merge_probes(spikes_list, clusters_list):
    merged_spikes = []
    merged_clusters = []
    cluster_max = 0
    for clusters, spikes in zip(clusters_list, spikes_list):
        spikes_local = {k: np.array(v, copy=True) for k, v in spikes.items()}
        spikes_local["clusters"] = spikes_local["clusters"] + cluster_max
        cluster_max = len(clusters) + cluster_max
        merged_spikes.append(spikes_local)
        merged_clusters.append(clusters)

    merged_clusters = pd.concat(merged_clusters, ignore_index=True)
    merged_spikes = {k: np.concatenate([s[k] for s in merged_spikes]) for k in merged_spikes[0]}
    sort_idx = np.argsort(merged_spikes["times"], kind="stable")
    merged_spikes = {k: v[sort_idx] for k, v in merged_spikes.items()}
    return merged_spikes, merged_clusters


def get_spike_data_per_interval(times, clusters, interval_begs, interval_ends, interval_len, binsize):
    n_bins = int(np.ceil(interval_len / binsize))
    cluster_ids = np.unique(clusters)
    n_clusters = len(cluster_ids)
    binned_spikes = np.zeros((len(interval_begs), n_clusters, n_bins), dtype=np.float32)
    idxs_beg = np.searchsorted(times, interval_begs, side="left")
    idxs_end = np.searchsorted(times, interval_ends, side="left")

    for interval_idx, (ib, ie, t_beg, t_end) in enumerate(zip(idxs_beg, idxs_end, interval_begs, interval_ends)):
        times_curr = times[ib:ie]
        clust_curr = clusters[ib:ie]
        if times_curr.size == 0:
            continue
        binned_tmp, _, cluster_idxs = bincount2D(times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
        _, idxs_tmp, _ = np.intersect1d(cluster_ids, cluster_idxs, return_indices=True)
        binned_spikes[interval_idx, idxs_tmp, :] = binned_tmp[:, :n_bins]

    return binned_spikes, cluster_ids


def bin_spiking_data(spikes, trials_df):
    intervals = np.vstack(
        [
            trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0],
            trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1],
        ]
    ).T
    binned_array, cluster_ids = get_spike_data_per_interval(
        spikes["times"],
        spikes["clusters"],
        interval_begs=intervals[:, 0],
        interval_ends=intervals[:, 1],
        interval_len=TIME_WINDOW[1] - TIME_WINDOW[0],
        binsize=BIN_SIZE,
    )
    binned_trials = np.asarray([x.T for x in binned_array], dtype=np.float32)
    return binned_trials, cluster_ids


def get_behavior_per_interval(target_times, target_vals, trials_df, allow_nans=False):
    interval_begs = trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0]
    interval_ends = trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1]
    n_bins = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BIN_SIZE))

    idxs_beg = np.searchsorted(target_times, interval_begs, side="right")
    idxs_end = np.searchsorted(target_times, interval_ends, side="left")

    vals_list = []
    mask = []
    for interval_idx, (ib, ie) in enumerate(zip(idxs_beg, idxs_end)):
        seg_t = target_times[ib:ie]
        seg_v = target_vals[ib:ie]
        good = True
        if len(seg_v) == 0:
            good = False
        elif np.isnan(interval_begs[interval_idx]) or np.isnan(interval_ends[interval_idx]):
            good = False
        elif np.sum(np.isnan(seg_v)) > 0 and not allow_nans:
            good = False
        elif np.abs(interval_begs[interval_idx] - seg_t[0]) > BIN_SIZE:
            good = False
        elif np.abs(interval_ends[interval_idx] - seg_t[-1]) > BIN_SIZE:
            good = False

        if not good:
            vals_list.append(None)
            mask.append(False)
            continue

        x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
        y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
        y_interp = np.asarray(y_interp, dtype=np.float32).reshape(-1)
        if y_interp.shape[0] != N_BINS or np.any(~np.isfinite(y_interp)):
            vals_list.append(None)
            mask.append(False)
            continue
        vals_list.append(y_interp)
        mask.append(True)

    return vals_list, np.asarray(mask, dtype=bool)


def safe_quantile_edges(values: np.ndarray):
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    if q1 >= q2:
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if np.isclose(vmin, vmax):
            q1 = vmin + 1e-6
            q2 = vmax + 2e-6
        else:
            q1 = vmin + (vmax - vmin) / 3.0
            q2 = vmin + 2.0 * (vmax - vmin) / 3.0
    return np.asarray([q1, q2], dtype=np.float32)


def discretize(values: np.ndarray, edges: np.ndarray):
    return np.digitize(values, bins=edges, right=False).astype(np.uint8)


def compact_neural_trial(neural_trial: np.ndarray) -> np.ndarray:
    max_count = int(neural_trial.max(initial=0))
    if max_count > np.iinfo(np.uint8).max:
        raise ValueError(f"Spike count {max_count} exceeds uint8 range for compact storage")
    return neural_trial.astype(np.uint8, copy=False)


def compact_session_record(record: SessionRecord) -> SessionRecord:
    diagnostic = record.diagnostic
    if diagnostic is not None:
        diagnostic = dict(diagnostic)
        if "neural" in diagnostic:
            diagnostic["neural"] = compact_neural_trial(np.asarray(diagnostic["neural"]))
        if "time_input" in diagnostic:
            diagnostic["time_input"] = np.asarray(diagnostic["time_input"], dtype=np.float16)
        for key in ("wheel_raw_times", "wheel_raw_values", "wheel_interp", "whisker_raw_times", "whisker_raw_values", "whisker_interp"):
            if key in diagnostic:
                diagnostic[key] = np.asarray(diagnostic[key], dtype=np.float16)
        for key in ("trial_indices", "choice_codes", "prior_codes"):
            if key in diagnostic:
                diagnostic[key] = np.asarray(diagnostic[key], dtype=np.int16)
        if "trial_numbers_in_block" in diagnostic:
            diagnostic["trial_numbers_in_block"] = np.asarray(diagnostic["trial_numbers_in_block"], dtype=np.float16)

    return SessionRecord(
        eid=record.eid,
        subject=record.subject,
        lab=record.lab,
        neural=[compact_neural_trial(np.asarray(trial)) for trial in record.neural],
        input_trials=[np.asarray(trial, dtype=np.float16) for trial in record.input_trials],
        choice_codes=[int(code) for code in record.choice_codes],
        prior_codes=[int(code) for code in record.prior_codes],
        wheel_values=[np.asarray(vals, dtype=np.float16) for vals in record.wheel_values],
        whisker_values=[np.asarray(vals, dtype=np.float16) for vals in record.whisker_values],
        cluster_regions_beryl=np.asarray(record.cluster_regions_beryl),
        trial_indices=np.asarray(record.trial_indices, dtype=np.int32),
        trial_numbers_in_block=np.asarray(record.trial_numbers_in_block, dtype=np.float16),
        diagnostic=diagnostic,
    )


def repack_session_cache():
    SESSION_RECORD_CACHE.mkdir(parents=True, exist_ok=True)
    cache_files = sorted(SESSION_RECORD_CACHE.glob("*.pkl"))
    print(f"Repacking {len(cache_files)} cached session records from {SESSION_RECORD_CACHE}", flush=True)
    for idx, path in enumerate(cache_files, start=1):
        with path.open("rb") as f:
            record = pickle.load(f)
        compact = compact_session_record(record)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("wb") as f:
            pickle.dump(compact, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)
        if (idx % 25 == 0) or (idx == len(cache_files)):
            print(f"  repacked {idx}/{len(cache_files)}", flush=True)
    print("Repacked cached session records.", flush=True)


def build_processing_plot(record: SessionRecord, wheel_edges, whisker_edges, outpath: Path):
    if record.diagnostic is None:
        return

    diag = record.diagnostic
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f"Processing diagnostics for {record.eid}")

    neural = diag["neural"]
    axes[0, 0].imshow(neural[: min(60, neural.shape[0])], aspect="auto", interpolation="nearest", cmap="viridis")
    axes[0, 0].set_title("Binned spike counts")
    axes[0, 0].set_ylabel("Neuron")

    axes[0, 1].plot(TIME_GRID, diag["time_input"], color="black")
    axes[0, 1].set_title("Time since stimulus onset input")
    axes[0, 1].set_ylabel("Seconds")

    panels = [
        (
            axes[1, 0],
            diag["wheel_raw_times"],
            diag["wheel_raw_values"],
            diag["wheel_interp"],
            diag["wheel_disc"],
            wheel_edges,
            "Wheel speed",
        ),
        (
            axes[1, 1],
            diag["whisker_raw_times"],
            diag["whisker_raw_values"],
            diag["whisker_interp"],
            diag["whisker_disc"],
            whisker_edges,
            f"Whisker motion energy ({diag['whisker_source']})",
        ),
    ]

    for ax, raw_t, raw_v, interp_v, disc_v, edges, title in panels:
        stim_on = diag["stim_on"]
        raw_mask = (raw_t >= stim_on + TIME_WINDOW[0] - 0.1) & (raw_t <= stim_on + TIME_WINDOW[1] + 0.1)
        ax.plot(raw_t[raw_mask] - stim_on, raw_v[raw_mask], color="0.7", linewidth=1.0, label="raw")
        ax.plot(TIME_GRID, interp_v, color="tab:blue", linewidth=1.2, marker="o", markersize=2, label="interpolated")
        for edge in edges:
            ax.axhline(edge, color="tab:red", linestyle=":", linewidth=1.0)
        ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Time from stimulus onset (s)")
        ax.legend(loc="upper right", fontsize=8)

        ax2 = ax.twinx()
        ax2.step(TIME_GRID, disc_v, where="mid", color="tab:green", linewidth=1.2)
        ax2.set_ylim(-0.25, 2.25)
        ax2.set_yticks([0, 1, 2])
        ax2.set_ylabel("bin")

    trial_ids = diag["trial_indices"]
    axes[2, 0].plot(trial_ids, diag["trial_numbers_in_block"], color="tab:purple")
    axes[2, 0].set_title("Trial number within block")
    axes[2, 0].set_xlabel("Original trial index")
    axes[2, 0].set_ylabel("Block trial #")

    axes[2, 1].plot(trial_ids, diag["choice_codes"], label="choice", color="tab:orange")
    axes[2, 1].plot(trial_ids, diag["prior_codes"], label="prior", color="tab:brown")
    axes[2, 1].set_title("Per-trial categorical outputs")
    axes[2, 1].set_xlabel("Original trial index")
    axes[2, 1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def process_session(one: ONE, row, probe_df: pd.DataFrame, brain_regions: BrainRegions, with_diagnostic: bool):
    session_start = time.time()
    print(f"\n=== Processing session {row.eid} ({row.subject}, {row.lab}) ===", flush=True)

    trials_df = load_trials_table(one, row)
    trial_mask = build_trial_mask(trials_df)
    block_trial_number = compute_trial_number_in_block(trials_df["probabilityLeft"].to_numpy())
    print(f"  total trials: {len(trials_df)}", flush=True)
    print(f"  trials passing base mask: {int(trial_mask.sum())}", flush=True)

    spikes_list = []
    clusters_list = []
    for probe_row in probe_df.itertuples(index=False):
        spikes, clusters = load_probe_data(one, row, probe_row.probe_name, brain_regions)
        if spikes is None or clusters is None:
            print(f"  skipping probe {probe_row.probe_name}: no well-isolated clusters", flush=True)
            continue
        spikes_list.append(spikes)
        clusters_list.append(clusters)
    if not spikes_list:
        raise RuntimeError(f"No probes with well-isolated clusters for session {row.eid}")
    spikes, clusters = merge_probes(spikes_list, clusters_list)

    binned_spikes, cluster_ids = bin_spiking_data(spikes, trials_df)
    cluster_regions_beryl = brain_regions.acronym2acronym(
        clusters.loc[cluster_ids, "acronym"].to_numpy(),
        mapping="Beryl",
    )

    wheel = load_wheel_speed(one, row)
    whisker, whisker_source = load_whisker_motion_energy(one, row)
    wheel_values, wheel_mask = get_behavior_per_interval(wheel["times"], wheel["values"], trials_df, allow_nans=False)
    whisker_values, whisker_mask = get_behavior_per_interval(
        whisker["times"], whisker["values"], trials_df, allow_nans=False
    )

    combined_mask = trial_mask.to_numpy(dtype=bool) & wheel_mask & whisker_mask
    valid_idx = np.flatnonzero(combined_mask)
    print(f"  trials with wheel+whisker coverage: {len(valid_idx)}", flush=True)
    if len(valid_idx) < 2:
        raise RuntimeError(f"Need at least 2 valid trials after filtering, found {len(valid_idx)}")

    neural_trials = []
    input_trials = []
    choice_codes = []
    prior_codes = []
    wheel_cont = []
    whisker_cont = []

    for idx in valid_idx:
        neural_trial = compact_neural_trial(binned_spikes[idx].T)
        if neural_trial.shape != (len(cluster_ids), N_BINS):
            raise ValueError(f"Unexpected neural shape {neural_trial.shape}")

        choice_code = map_choice(trials_df.iloc[idx]["choice"])
        prior_code = map_prior(trials_df.iloc[idx]["probabilityLeft"])
        trial_num = float(block_trial_number[idx])
        if not np.isfinite(trial_num):
            raise ValueError("trial_number_in_block is not finite")

        input_trial = np.vstack(
            [
                TIME_GRID,
                np.full(N_BINS, trial_num, dtype=np.float32),
            ]
        ).astype(np.float16)

        neural_trials.append(neural_trial)
        input_trials.append(input_trial)
        choice_codes.append(int(choice_code))
        prior_codes.append(int(prior_code))
        wheel_cont.append(np.asarray(wheel_values[idx], dtype=np.float16))
        whisker_cont.append(np.asarray(whisker_values[idx], dtype=np.float16))

    diagnostic = None
    if with_diagnostic:
        diag_idx = valid_idx[0]
        diagnostic = {
            "neural": neural_trials[0],
            "time_input": input_trials[0][0],
            "wheel_raw_times": wheel["times"],
            "wheel_raw_values": wheel["values"],
            "wheel_interp": wheel_cont[0],
            "whisker_raw_times": whisker["times"],
            "whisker_raw_values": whisker["values"],
            "whisker_interp": whisker_cont[0],
            "whisker_source": whisker_source,
            "stim_on": float(trials_df.iloc[diag_idx][ALIGN_EVENT]),
            "trial_indices": valid_idx[: min(50, len(valid_idx))],
            "trial_numbers_in_block": block_trial_number[valid_idx[: min(50, len(valid_idx))]],
            "choice_codes": np.asarray(choice_codes[: min(50, len(choice_codes))], dtype=np.int16),
            "prior_codes": np.asarray(prior_codes[: min(50, len(prior_codes))], dtype=np.int16),
        }

    print(f"  retained trials: {len(neural_trials)}", flush=True)
    print(f"  neurons: {len(cluster_ids)}", flush=True)
    print(f"  session time: {time.time() - session_start:.2f}s", flush=True)

    return SessionRecord(
        eid=row.eid,
        subject=row.subject,
        lab=row.lab,
        neural=neural_trials,
        input_trials=input_trials,
        choice_codes=choice_codes,
        prior_codes=prior_codes,
        wheel_values=wheel_cont,
        whisker_values=whisker_cont,
        cluster_regions_beryl=np.asarray(cluster_regions_beryl),
        trial_indices=valid_idx.astype(np.int32),
        trial_numbers_in_block=block_trial_number[valid_idx].astype(np.float16),
        diagnostic=diagnostic,
    )


def build_dataset(session_records: list[SessionRecord]):
    wheel_all = np.concatenate([np.concatenate(rec.wheel_values) for rec in session_records]).astype(np.float32)
    whisker_all = np.concatenate([np.concatenate(rec.whisker_values) for rec in session_records]).astype(np.float32)
    wheel_edges = safe_quantile_edges(wheel_all)
    whisker_edges = safe_quantile_edges(whisker_all)
    print(f"\nGlobal wheel edges: {wheel_edges.tolist()}", flush=True)
    print(f"Global whisker edges: {whisker_edges.tolist()}", flush=True)

    subjects = []
    subject_to_idx = {}
    brain_regions = []
    region_to_idx = {}

    neural = []
    input_data = []
    output = []
    subject_idx = []
    brain_region_idx = []

    for rec in session_records:
        if rec.subject not in subject_to_idx:
            subject_to_idx[rec.subject] = len(subjects)
            subjects.append(rec.subject)
        subject_idx.append(subject_to_idx[rec.subject])

        sess_region_idx = []
        for region in rec.cluster_regions_beryl.tolist():
            if region not in region_to_idx:
                region_to_idx[region] = len(brain_regions)
                brain_regions.append(region)
            sess_region_idx.append(region_to_idx[region])
        brain_region_idx.append(np.asarray(sess_region_idx, dtype=np.int64))

        neural.append(rec.neural)
        input_data.append(rec.input_trials)
        session_output = []
        for choice_code, prior_code, wheel_vals, whisker_vals in zip(
            rec.choice_codes,
            rec.prior_codes,
            rec.wheel_values,
            rec.whisker_values,
        ):
            output_trial = np.vstack(
                [
                    np.full(N_BINS, choice_code, dtype=np.uint8),
                    np.full(N_BINS, prior_code, dtype=np.uint8),
                    discretize(wheel_vals, wheel_edges),
                    discretize(whisker_vals, whisker_edges),
                ]
            )
            session_output.append(output_trial)
        output.append(session_output)

    metadata = {
        "task_description": (
            "IBL visual decision-making task; decode choice, prior probability of left, "
            "wheel-speed tertile, and whisker-motion-energy tertile from stimulus-aligned spike counts."
        ),
        "time_bin_size": 20.0,
        "temporal_alignment_event": "stimulus onset",
        "off_start": TIME_WINDOW[0],
        "off_end": TIME_WINDOW[1],
        "input_time_grid_s": TIME_GRID.tolist(),
        "session_eids": [rec.eid for rec in session_records],
        "session_labs": [rec.lab for rec in session_records],
        "wheel_speed_quantile_edges": wheel_edges.tolist(),
        "whisker_motion_energy_quantile_edges": whisker_edges.tolist(),
        "release_source": str(RELEASE_CSV.relative_to(ROOT)),
        "cache_sources": [str(READONLY_CACHE.relative_to(ROOT)), str(WRITABLE_CACHE.relative_to(ROOT))],
        "trial_filters": {
            "exclude_missing_events": [
                "stimOn_times",
                "choice",
                "feedback_times",
                "probabilityLeft",
                "firstMovement_times",
                "feedbackType",
            ],
            "reaction_time_range_s": [0.08, 2.0],
            "exclude_nochoice": True,
            "max_trial_len_s": 10.0,
        },
        "neural_unit_policy": "well-isolated clusters only (label >= 1), matching released good-unit statistics; Beryl region labels stored",
    }

    data = {
        "neural": neural,
        "input": input_data,
        "output": output,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int32),
        "brain_regions": brain_regions,
        "brain_region_idx": brain_region_idx,
        "input_names": ["time_since_stimulus_onset_s", "trial_number_in_block"],
        "output_names": ["choice", "prior_left_probability", "wheel_speed_bin", "whisker_motion_energy_bin"],
        "output_values": [
            ["left", "right"],
            ["0.2", "0.5", "0.8"],
            ["low", "mid", "high"],
            ["low", "mid", "high"],
        ],
        "metadata": metadata,
    }
    return data, wheel_edges, whisker_edges


def process_session_worker(row_dict, probe_records, with_diagnostic, record_cache_dir: str | None = None):
    row = SimpleNamespace(**row_dict)
    probe_df = pd.DataFrame(probe_records)
    one = build_one()
    brain_regions = BrainRegions()
    try:
        record = process_session(one, row, probe_df, brain_regions, with_diagnostic)
        record_path = None
        if record_cache_dir is not None:
            cache_dir = Path(record_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            record_path = cache_dir / f"{row.eid}.pkl"
            tmp_path = record_path.with_suffix(".tmp")
            with tmp_path.open("wb") as f:
                pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(record_path)
        return {
            "ok": True,
            "eid": row.eid,
            "record": record if record_cache_dir is None else None,
            "record_path": str(record_path) if record_path is not None else None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "eid": row.eid,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def main():
    args = parse_args()
    if args.full and args.sample:
        raise SystemExit("Use either --full or --sample, not both.")
    if args.fill_session_cache_only and args.sample:
        raise SystemExit("--fill-session-cache-only is only supported in full mode.")
    if args.repack_session_cache and args.sample:
        raise SystemExit("--repack-session-cache is only supported in full mode.")

    total_start = time.time()
    sample_limit = 2 if args.sample else None

    print(f"Output file: {args.outpicklefile}", flush=True)
    print(f"Mode: {'sample' if args.sample else 'full'}", flush=True)
    print(f"Readonly cache: {READONLY_CACHE}", flush=True)
    print(f"Writable cache: {WRITABLE_CACHE}", flush=True)

    if args.repack_session_cache:
        repack_session_cache()
        return

    one = build_one()
    brain_regions = BrainRegions()
    session_rows, probe_rows = load_release_sessions()
    print(f"Release sessions listed: {len(session_rows)}", flush=True)

    session_records = []
    skipped = []
    if sample_limit is not None:
        diag_budget = 2 if args.show_processing else 0
        for row in session_rows.head(sample_limit).itertuples(index=False):
            try:
                rec = process_session(
                    one=one,
                    row=row,
                    probe_df=probe_rows[row.eid],
                    brain_regions=brain_regions,
                    with_diagnostic=diag_budget > 0,
                )
                session_records.append(rec)
                if diag_budget > 0:
                    diag_budget -= 1
            except Exception as exc:
                skipped.append((row.eid, str(exc)))
                print(f"  skipped {row.eid}: {exc}", flush=True)
                traceback.print_exc()
    elif args.from_session_cache:
        print(f"Loading cached session records from {SESSION_RECORD_CACHE}", flush=True)
        indexed_records = {}
        for idx, row in enumerate(session_rows.itertuples(index=False)):
            record_path = SESSION_RECORD_CACHE / f"{row.eid}.pkl"
            if record_path.exists():
                with record_path.open("rb") as f:
                    indexed_records[idx] = pickle.load(f)
            else:
                skipped.append((row.eid, "missing cached session record"))
        session_records = [indexed_records[idx] for idx in sorted(indexed_records)]
    else:
        max_workers = min(max(args.max_workers, 1), os.cpu_count() or 1, len(session_rows))
        print(f"Full-mode parallel workers: {max_workers}", flush=True)
        indexed_records = {}
        SESSION_RECORD_CACHE.mkdir(parents=True, exist_ok=True)
        if not args.resume_session_cache:
            for stale_file in SESSION_RECORD_CACHE.glob("*.pkl"):
                stale_file.unlink()
            for stale_file in SESSION_RECORD_CACHE.glob("*.tmp"):
                stale_file.unlink()
        else:
            print(f"Resuming from existing cache in {SESSION_RECORD_CACHE}", flush=True)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for idx, row in enumerate(session_rows.itertuples(index=False)):
                existing_record = SESSION_RECORD_CACHE / f"{row.eid}.pkl"
                if args.resume_session_cache and existing_record.exists():
                    if not args.fill_session_cache_only:
                        with existing_record.open("rb") as f:
                            indexed_records[idx] = pickle.load(f)
                    continue
                futures[
                    pool.submit(
                        process_session_worker,
                        row._asdict(),
                        probe_rows[row.eid].to_dict("records"),
                        args.show_processing and idx < 2,
                        str(SESSION_RECORD_CACHE),
                    )
                ] = idx
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                if result["ok"]:
                    if not args.fill_session_cache_only:
                        with Path(result["record_path"]).open("rb") as f:
                            indexed_records[idx] = pickle.load(f)
                else:
                    skipped.append((result["eid"], result["error"]))
                    print(f"  skipped {result['eid']}: {result['error']}", flush=True)
                    print(result["traceback"], flush=True)
        if not args.fill_session_cache_only:
            session_records = [indexed_records[idx] for idx in sorted(indexed_records)]

    if args.fill_session_cache_only:
        cached_count = len(list(SESSION_RECORD_CACHE.glob("*.pkl")))
        elapsed = time.time() - total_start
        print("\nSession-cache fill complete.", flush=True)
        print(f"  cache files present: {cached_count}", flush=True)
        print(f"  skipped sessions this run: {len(skipped)}", flush=True)
        print(f"  total time: {elapsed:.2f}s", flush=True)
        if skipped:
            print("\nSkipped sessions:", flush=True)
            for eid, reason in skipped[:20]:
                print(f"  {eid}: {reason}", flush=True)
            if len(skipped) > 20:
                print(f"  ... and {len(skipped) - 20} more", flush=True)
        return

    if not session_records:
        raise RuntimeError("No sessions were successfully processed.")
    if sample_limit is not None and len(session_records) < sample_limit:
        raise RuntimeError(f"Sample mode processed only {len(session_records)} sessions out of requested {sample_limit}.")

    data, wheel_edges, whisker_edges = build_dataset(session_records)

    for rec in session_records:
        if rec.diagnostic is not None:
            rec.diagnostic["wheel_disc"] = discretize(rec.diagnostic["wheel_interp"], wheel_edges)
            rec.diagnostic["whisker_disc"] = discretize(rec.diagnostic["whisker_interp"], whisker_edges)
            plot_path = ROOT / f"processing_{rec.eid}.png"
            build_processing_plot(rec, wheel_edges, whisker_edges, plot_path)
            print(f"Saved {plot_path.name}", flush=True)

    outpath = Path(args.outpicklefile)
    tmppath = outpath.with_suffix(outpath.suffix + ".tmp")
    print(f"Writing dataset to {tmppath.name}", flush=True)
    with tmppath.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    tmppath.replace(outpath)
    print(f"Finished writing {outpath.name}", flush=True)

    total_trials = sum(len(s) for s in data["neural"])
    total_neurons = int(sum(len(x) for x in data["brain_region_idx"]))
    elapsed = time.time() - total_start

    print("\nConversion complete.", flush=True)
    print(f"  sessions retained: {len(data['neural'])}", flush=True)
    print(f"  subjects retained: {len(data['subjects'])}", flush=True)
    print(f"  total trials retained: {total_trials}", flush=True)
    print(f"  total neurons retained: {total_neurons}", flush=True)
    print(f"  skipped sessions: {len(skipped)}", flush=True)
    print(f"  output path: {outpath}", flush=True)
    print(f"  total time: {elapsed:.2f}s", flush=True)

    if skipped:
        print("\nSkipped sessions:", flush=True)
        for eid, reason in skipped[:20]:
            print(f"  {eid}: {reason}", flush=True)
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more", flush=True)


if __name__ == "__main__":
    main()
