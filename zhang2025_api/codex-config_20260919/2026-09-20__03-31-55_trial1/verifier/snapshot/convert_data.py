#!/usr/bin/env python3
"""Convert the staged IBL brain-wide-map ONE cache to decoder format.

The script intentionally reads scientific data only through ONE and brainbox loaders.
It follows the supplied Zhang et al. cache preprocessing: merge probes within session,
retain all Kilosort clusters, align to stimulus onset, and use 20-ms spike-count bins
over [-0.5, 1.5) seconds. Continuous behavior is sampled at bin ends.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import logging
import os
import pickle
import threading
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from one.api import One
from one.alf.exceptions import ALFWarning
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from iblatlas.regions import BrainRegions


CACHE_ROOT = Path("/app/data/one_cache")
SPIKE_RELEASE = CACHE_ROOT / "Brainwidemap"
BEHAVIOR_RELEASE = CACHE_ROOT / "2025_Q3_IBL_et_al_BWM"
TRIAL_REVISION = "2025-03-03"
BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = 100
BIN_END_TIMES = np.arange(1, N_BINS + 1, dtype=np.float64) * BIN_SIZE + OFF_START
REQUIRED_TRIAL_COLUMNS = (
    "stimOn_times", "choice", "feedback_times", "probabilityLeft",
    "firstMovement_times", "feedbackType", "goCue_times",
)


def build_one(verbose=True) -> tuple[One, list[str]]:
    """Build one in-memory ONE index from complementary staged releases."""
    one = One(cache_dir=CACHE_ROOT)
    one.load_cache(SPIKE_RELEASE)
    behavior_one = One(cache_dir=CACHE_ROOT)
    behavior_one.load_cache(BEHAVIOR_RELEASE)

    # The 2025 table contains revised video products.  Merge its registered rows into
    # the broader BWM index, keeping newer duplicate dataset records.
    datasets = pd.concat([one._cache["datasets"], behavior_one._cache["datasets"]])
    datasets = datasets[~datasets.index.duplicated(keep="last")].sort_index()

    # The staged trial tables are the 2025-03-03 revision, while the frozen BWM row
    # has the old unrevisioned relative path.  Correct the in-memory ONE index only.
    trial_rows = datasets["rel_path"].eq("alf/_ibl_trials.table.pqt")
    datasets.loc[trial_rows, "rel_path"] = (
        f"alf/#{TRIAL_REVISION}#/_ibl_trials.table.pqt"
    )
    # Old hashes/sizes describe the superseded base file; filesystem existence and
    # scientific shape/value checks below validate the revised file.
    datasets.loc[trial_rows, "hash"] = ""
    datasets.loc[trial_rows, "file_size"] = 0
    datasets.loc[trial_rows, "default_revision"] = True
    one._cache["datasets"] = datasets

    spike_eids = set(map(str, one.search()))
    behavior_eids = set(map(str, behavior_one.search()))
    eids = sorted(spike_eids & behavior_eids)

    subset_file = Path("/app/data/DATALIMIT_SUBSET.csv")
    if subset_file.exists():
        subset = pd.read_csv(subset_file)
        col = "eid" if "eid" in subset.columns else subset.columns[0]
        allowed = set(subset[col].astype(str))
        eids = [eid for eid in eids if eid in allowed]
        if verbose:
            print(f"Datalimit subset active: {len(eids)} release sessions", flush=True)
    else:
        if verbose:
            print(f"Full release scope: {len(eids)} task sessions", flush=True)
    return one, eids


def probe_names(one: One, eid: str) -> list[str]:
    """Return probes having pykilosort collections, without an Alyx query."""
    names = set()
    for collection in one.list_collections(eid):
        parts = collection.split("/")
        if len(parts) >= 3 and parts[0] == "alf" and parts[1].startswith("probe"):
            if "pykilosort" in parts:
                names.add(parts[1])
    return sorted(names)


def load_and_merge_spikes(one: One, eid: str):
    """Load all probes with SpikeSortingLoader and merge cluster indices."""
    all_times, all_clusters, all_regions, all_good = [], [], [], []
    offset = 0
    used_probes = probe_names(one, eid)
    if not used_probes:
        raise RuntimeError("no pykilosort probe collections")

    for pname in used_probes:
        loader = SpikeSortingLoader(eid=eid, pname=pname, one=one)
        spikes, clusters, channels = loader.load_spike_sorting()
        clusters = loader.merge_clusters(spikes, clusters, channels)
        n_clusters = len(clusters["channels"])
        if n_clusters == 0:
            continue
        spike_cluster = np.asarray(spikes["clusters"], dtype=np.int64)
        valid = (spike_cluster >= 0) & (spike_cluster < n_clusters)
        all_times.append(np.asarray(spikes["times"], dtype=np.float64)[valid])
        all_clusters.append(spike_cluster[valid] + offset)
        all_regions.append(np.asarray(clusters["acronym"], dtype=object).astype(str))
        label = np.asarray(clusters.get("label", np.zeros(n_clusters)), dtype=float)
        all_good.append(label >= 1)
        offset += n_clusters

    if not all_times or offset == 0:
        raise RuntimeError("no clusters loaded")
    times = np.concatenate(all_times)
    cluster_ids = np.concatenate(all_clusters)
    order = np.argsort(times, kind="stable")
    return (
        times[order], cluster_ids[order], np.concatenate(all_regions),
        np.concatenate(all_good), used_probes,
    )


def block_trial_numbers(probability_left: np.ndarray) -> np.ndarray:
    """Zero-based ordinal within runs of constant block probability."""
    p = np.asarray(probability_left, dtype=float)
    out = np.zeros(len(p), dtype=np.int32)
    for i in range(1, len(p)):
        out[i] = out[i - 1] + 1 if p[i] == p[i - 1] else 0
    return out


def load_behavior(one: One, eid: str):
    """Load trials, smoothed wheel velocity, and whisker motion energy."""
    sess = SessionLoader(one=one, eid=eid)
    # This explicitly selects the staged corrected trials revision.
    sess.load_trials(revision=TRIAL_REVISION)
    sess.load_wheel()
    whisker_view = None
    for view in ("left", "right"):
        try:
            sess.load_motion_energy(views=[view])
            key = f"{view}Camera"
            frame = sess.motion_energy[key]
            if "whiskerMotionEnergy" in frame and len(frame) > 1:
                whisker_view = view
                motion = frame
                break
        except Exception:
            continue
    if whisker_view is None:
        raise RuntimeError("no left or right whisker motion-energy stream")
    return sess.trials.copy(), sess.wheel.copy(), motion.copy(), whisker_view


def interpolate_trials(times, values, stim_times, *, allow_nan=False):
    """Reference-style linear interpolation at the ends of 20-ms bins."""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    finite_t = np.isfinite(times)
    times, values = times[finite_t], values[finite_t]
    if len(times) < 2 or np.any(np.diff(times) < 0):
        raise RuntimeError("behavior timestamps absent or unsorted")
    targets = np.asarray(stim_times)[:, None] + BIN_END_TIMES[None, :]
    result = np.empty(targets.shape, dtype=np.float64)
    good = np.ones(len(targets), dtype=bool)
    for i, target in enumerate(targets):
        interval_beg = target[0] - BIN_SIZE
        interval_end = target[-1]
        ib = np.searchsorted(times, interval_beg, side="right")
        ie = np.searchsorted(times, interval_end, side="left")
        # Match get_behavior_per_interval's boundary coverage requirement.
        if ib >= len(times) or ie <= 0:
            good[i] = False
            result[i] = np.nan
            continue
        lo = max(0, ib)
        hi = min(len(times), ie + 1)
        local_t, local_v = times[lo:hi], values[lo:hi]
        if (local_t[0] - interval_beg > BIN_SIZE + 1e-9 or
                interval_end - local_t[-1] > BIN_SIZE + 1e-9 or len(local_t) < 2):
            good[i] = False
            result[i] = np.nan
            continue
        result[i] = np.interp(target, local_t, local_v)
        if not allow_nan and not np.all(np.isfinite(result[i])):
            good[i] = False
    return result, good


def trial_mask(trials: pd.DataFrame) -> np.ndarray:
    """Apply the supplied code/paper event, RT, length, and response filters."""
    missing = [col for col in REQUIRED_TRIAL_COLUMNS if col not in trials]
    if missing:
        raise RuntimeError(f"missing trial columns: {missing}")
    mask = np.ones(len(trials), dtype=bool)
    for col in REQUIRED_TRIAL_COLUMNS:
        mask &= np.isfinite(trials[col].to_numpy(dtype=float))
    rt = trials["firstMovement_times"].to_numpy() - trials["stimOn_times"].to_numpy()
    duration = trials["feedback_times"].to_numpy() - trials["goCue_times"].to_numpy()
    choice = trials["choice"].to_numpy()
    prior = trials["probabilityLeft"].to_numpy()
    mask &= (rt >= 0.08) & (rt <= 2.0)
    mask &= duration <= 10.0
    mask &= np.isin(choice, (-1, 1))
    mask &= np.isin(np.round(prior, 6), (0.2, 0.5, 0.8))
    return mask


def discretize_tertiles(values: np.ndarray):
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise RuntimeError("no finite behavior values")
    thresholds = np.quantile(finite, [1 / 3, 2 / 3])
    if not thresholds[0] < thresholds[1]:
        raise RuntimeError(f"collapsed tertile thresholds {thresholds.tolist()}")
    labels = np.digitize(values, thresholds, right=False).astype(np.int8)
    return labels, thresholds


def bin_spikes(times, clusters, stim_times, n_clusters):
    """Count spikes per selected trial using reference half-open bins."""
    edges_rel = np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE + OFF_START
    trials = []
    for stim in np.asarray(stim_times, dtype=np.float64):
        edges = stim + edges_rel
        lo = np.searchsorted(times, edges[0], side="left")
        hi = np.searchsorted(times, edges[-1], side="left")
        st = times[lo:hi]
        sc = clusters[lo:hi]
        tb = np.searchsorted(edges, st, side="right") - 1
        valid = (tb >= 0) & (tb < N_BINS)
        flat = sc[valid] * N_BINS + tb[valid]
        count = np.bincount(flat, minlength=n_clusters * N_BINS)
        trials.append(count.reshape(n_clusters, N_BINS).astype(np.float32))
    return trials


def plot_processing(eid, pop_counts, wheel_raw, whisker_raw, wheel_interp,
                    whisker_interp, wheel_labels, whisker_labels, thresholds):
    """Plot all processing stages for representative retained trials."""
    nshow = min(3, len(pop_counts))
    fig, axes = plt.subplots(4, nshow, figsize=(5 * nshow, 12), sharex="col")
    if nshow == 1:
        axes = axes[:, None]
    rel = BIN_END_TIMES
    for j in range(nshow):
        axes[0, j].plot(rel, pop_counts[j].sum(0), color="black")
        axes[0, j].axvline(0, color="red", ls="--")
        axes[0, j].set_title(f"{eid[:8]} trial {j}: population spikes")
        axes[1, j].plot(rel, wheel_interp[j], color="tab:blue", label="interpolated")
        axes[1, j].step(rel, wheel_labels[j], where="mid", color="tab:orange", label="bin")
        axes[2, j].plot(rel, whisker_interp[j], color="tab:green", label="interpolated")
        axes[2, j].step(rel, whisker_labels[j], where="mid", color="tab:orange", label="bin")
        axes[3, j].imshow(pop_counts[j], aspect="auto", interpolation="nearest",
                          extent=[OFF_START, OFF_END, len(pop_counts[j]), 0])
        axes[3, j].set_xlabel("time from stimulus onset (s)")
    axes[0, 0].set_ylabel("spikes/bin")
    axes[1, 0].set_ylabel("wheel speed / class")
    axes[2, 0].set_ylabel("whisker energy / class")
    axes[3, 0].set_ylabel("neuron")
    axes[1, 0].legend(fontsize=8)
    axes[2, 0].legend(fontsize=8)
    fig.suptitle(
        f"Processing: wheel q={thresholds['wheel']}; whisker q={thresholds['whisker']}"
    )
    fig.tight_layout()
    fig.savefig(f"/app/processing_{eid}.png", dpi=140)
    plt.close(fig)


def convert_session(one: One, eid: str, make_plot=False):
    start = time.perf_counter()
    details = one.get_details(eid, full=False)
    trials, wheel, motion, whisker_view = load_behavior(one, eid)
    block_num = block_trial_numbers(trials["probabilityLeft"].to_numpy())
    base_mask = trial_mask(trials)

    stim = trials["stimOn_times"].to_numpy(dtype=float)
    wheel_speed = np.abs(wheel["velocity"].to_numpy(dtype=float))
    wheel_interp, wheel_good = interpolate_trials(
        wheel["times"].to_numpy(), wheel_speed, stim
    )
    whisker_interp, whisker_good = interpolate_trials(
        motion["times"].to_numpy(), motion["whiskerMotionEnergy"].to_numpy(), stim
    )
    keep = base_mask & wheel_good & whisker_good
    selected = np.flatnonzero(keep)
    if len(selected) < 2:
        raise RuntimeError(f"only {len(selected)} valid trials")

    spike_times, spike_clusters, regions, good_units, probes = load_and_merge_spikes(one, eid)
    neural = bin_spikes(spike_times, spike_clusters, stim[selected], len(regions))

    # A population-wide zero over two seconds indicates a trial outside a valid
    # electrophysiology interval, not physiological silence. Exclude such periods
    # before fitting behavior tertiles so all modalities use the identical trials.
    neural_valid = np.asarray([np.any(x) for x in neural], dtype=bool)
    if not np.all(neural_valid):
        selected = selected[neural_valid]
        neural = [x for x, valid in zip(neural, neural_valid) if valid]
    if len(selected) < 2:
        raise RuntimeError(f"only {len(selected)} trials with valid neural periods")

    # Thresholds are calculated only from final retained, finite aligned samples.
    wheel_labels, wheel_q = discretize_tertiles(wheel_interp[selected])
    whisker_labels, whisker_q = discretize_tertiles(whisker_interp[selected])

    session_input, session_output = [], []
    prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
    for row, trial_idx in enumerate(selected):
        inp = np.vstack([
            BIN_END_TIMES.astype(np.float32),
            np.full(N_BINS, block_num[trial_idx], dtype=np.float32),
        ])
        choice = 0 if trials.iloc[trial_idx]["choice"] == -1 else 1
        p = float(np.round(trials.iloc[trial_idx]["probabilityLeft"], 6))
        out = np.vstack([
            np.full(N_BINS, choice, dtype=np.int8),
            np.full(N_BINS, prior_map[p], dtype=np.int8),
            wheel_labels[row],
            whisker_labels[row],
        ])
        session_input.append(inp)
        session_output.append(out)

    if make_plot:
        plot_processing(
            eid, neural, wheel, motion, wheel_interp[selected], whisker_interp[selected],
            wheel_labels, whisker_labels,
            {"wheel": wheel_q.tolist(), "whisker": whisker_q.tolist()},
        )

    elapsed = time.perf_counter() - start
    info = {
        "eid": eid,
        "subject": str(details["subject"]),
        "lab": str(details["lab"]),
        "date": str(details["date"]),
        "n_trials_raw": int(len(trials)),
        "n_trials_retained": int(len(selected)),
        "retained_trial_indices": selected.astype(int).tolist(),
        "n_neurons": int(len(regions)),
        "n_good_units": int(np.sum(good_units)),
        "probe_names": probes,
        "whisker_camera": whisker_view,
        "wheel_tertiles": wheel_q.tolist(),
        "whisker_tertiles": whisker_q.tolist(),
        "processing_seconds": elapsed,
    }
    print(
        f"OK {eid}: {len(selected)}/{len(trials)} trials, {len(regions)} neurons, "
        f"{len(probes)} probes, {elapsed:.1f}s", flush=True
    )
    return neural, session_input, session_output, regions, info


def validate_session(neural, inputs, outputs, regions):
    assert len(neural) == len(inputs) == len(outputs) >= 2
    n = len(regions)
    for x, i, o in zip(neural, inputs, outputs):
        assert x.shape == (n, N_BINS) and x.dtype == np.float32
        assert i.shape == (2, N_BINS) and i.dtype == np.float32
        assert o.shape == (4, N_BINS) and np.issubdtype(o.dtype, np.integer)
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(i)) and np.all(np.isfinite(o))
        assert set(np.unique(o[0])).issubset({0, 1})
        for j in (1, 2, 3):
            assert set(np.unique(o[j])).issubset({0, 1, 2})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("outpicklefile")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="process all sessions (default)")
    mode.add_argument("--sample", action="store_true", help="process first 2 successful sessions")
    parser.add_argument("--show-processing", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message="No cache tables found")
    warnings.filterwarnings("ignore", category=ALFWarning)
    # Frozen table rows intentionally point at newer staged revisions; suppress ONE's
    # expected revision/hash chatter while retaining our explicit per-session errors.
    for logger_name in ("one.api", "one.util", "one.alf.io", "brainbox.io.one"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    one, eids = build_one()
    target_sessions = 2 if args.sample else None
    atlas = BrainRegions()

    neural_all, input_all, output_all = [], [], []
    region_names_all, session_infos, failures = [], [], []
    subjects = []
    t0 = time.perf_counter()

    worker_state = threading.local()

    def attempt(item):
        idx, eid = item
        try:
            do_plot = args.show_processing and idx < 2
            # ONE contains mutable loader/cache state and is not thread-safe. Keep a
            # distinct composite ONE instance in each full-conversion worker.
            if args.sample or args.show_processing:
                worker_one = one
            else:
                if not hasattr(worker_state, "one"):
                    worker_state.one, _ = build_one(verbose=False)
                worker_one = worker_state.one
            return eid, convert_session(worker_one, eid, do_plot), None
        except Exception as exc:
            return eid, None, f"{type(exc).__name__}: {exc}"
        finally:
            gc.collect()

    indexed_eids = list(enumerate(eids))
    # Sample/plot runs stay sequential for deterministic diagnostics. Full conversion
    # overlaps independent session I/O and NumPy work to keep the estimate <15 min.
    if args.sample or args.show_processing:
        results = map(attempt, indexed_eids)
        executor = None
    else:
        workers = min(16, max(2, os.cpu_count() or 2))
        print(f"Parallel full conversion with {workers} session workers", flush=True)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        results = executor.map(attempt, indexed_eids)

    try:
        for eid, converted, error in results:
            if error is not None:
                failures.append({"eid": eid, "error": error})
                print(f"SKIP {eid}: {error}", flush=True)
                continue
            neural, inp, out, acronyms, info = converted
            validate_session(neural, inp, out, acronyms)
            # Reference code maps native acronyms to the Beryl hierarchy.
            beryl = atlas.acronym2acronym(np.asarray(acronyms), mapping="Beryl")
            beryl = np.asarray(beryl, dtype=str)
            neural_all.append(neural)
            input_all.append(inp)
            output_all.append(out)
            region_names_all.append(beryl)
            session_infos.append(info)
            subjects.append(info["subject"])
            if target_sessions and len(neural_all) >= target_sessions:
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    if not neural_all:
        raise RuntimeError("no sessions converted successfully")

    subject_names = list(dict.fromkeys(subjects))
    subject_lookup = {name: i for i, name in enumerate(subject_names)}
    all_regions = sorted(set(np.concatenate(region_names_all).tolist()))
    region_lookup = {name: i for i, name in enumerate(all_regions)}
    region_indices = [
        np.asarray([region_lookup[x] for x in names], dtype=np.int32)
        for names in region_names_all
    ]

    data = {
        "neural": neural_all,
        "input": input_all,
        "output": output_all,
        "subjects": subject_names,
        "subject_idx": np.asarray([subject_lookup[x] for x in subjects], dtype=np.int32),
        "brain_regions": all_regions,
        "brain_region_idx": region_indices,
        "input_names": ["time since stimulus onset", "trial number in block"],
        "output_names": [
            "choice", "prior probability of left", "wheel speed", "whisker motion energy"
        ],
        "output_values": [
            ["left", "right"],
            ["0.2", "0.5", "0.8"],
            ["slow", "medium", "fast"],
            ["low", "medium", "high"],
        ],
        "metadata": {
            "task_description": (
                "IBL visual decision task; decode choice, block prior, wheel speed, "
                "and whisker-pad motion energy from session-wide population spike counts."
            ),
            "time_bin_size": 20.0,
            "temporal_alignment_event": "visual stimulus onset (trials.stimOn_times)",
            "off_start": OFF_START,
            "off_end": OFF_END,
            "neural_representation": "unsmoothed spike counts in half-open 20-ms bins",
            "behavior_sampling": "linear interpolation at bin-end timestamps",
            "neuron_filter": "all Kilosort clusters (reference decoder cache qc=None)",
            "trial_filter": (
                "required events finite; 0.08<=firstMovement-stimOn<=2 s; "
                "feedback-goCue<=10 s; nonzero choice; valid block; complete behaviors"
            ),
            "discretization": "session-wise tertiles over retained aligned samples",
            "session_info": session_infos,
            "failed_sessions": failures,
            "source_release_sessions": len(eids),
            "conversion_seconds": time.perf_counter() - t0,
        },
    }

    out = Path(args.outpicklefile)
    with out.open("wb") as stream:
        pickle.dump(data, stream, protocol=5)
    total_trials = sum(map(len, neural_all))
    print(
        f"Saved {out}: {len(neural_all)} sessions, {len(subject_names)} subjects, "
        f"{total_trials} trials, {sum(len(x) for x in region_names_all)} session-neurons, "
        f"{len(failures)} skipped, {time.perf_counter() - t0:.1f}s total",
        flush=True,
    )


if __name__ == "__main__":
    main()
