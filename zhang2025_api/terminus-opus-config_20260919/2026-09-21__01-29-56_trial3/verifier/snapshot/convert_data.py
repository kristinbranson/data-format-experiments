#!/usr/bin/env python
"""Convert the IBL brain-wide-map (BWM) public dataset into the decoder format.

Data are read exclusively through the ONE API (``one.api.ONE``) and the ``brainbox``
loaders (``SessionLoader``, ``SpikeSortingLoader``), re-using the reference code of
Zhang et al. (``/app/code/code_zhang2025/src/utils/ibl_data_utils.py``) wherever possible.

Trial geometry (identical to the reference caching script ``src/0_data_caching.py``):
    align_time  = 'stimOn_times'
    time_window = (-0.5, +1.5) s      ->  2 s trials
    binsize     = 0.02 s              ->  T = 100 bins

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""
import argparse
import os
import pickle
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '/app/code/code_zhang2025/src')

from one.api import ONE                                    # noqa: E402
from brainbox.io.one import SessionLoader, SpikeSortingLoader  # noqa: E402
from iblatlas.regions import BrainRegions                  # noqa: E402
from iblutil.numerical import ismember                     # noqa: E402
from utils.ibl_data_utils import merge_probes, load_trials_and_mask  # noqa: E402

# ----------------------------------------------------------------------------- config
CACHE_DIR = '/app/data/one_cache'
BASE_URL = 'https://openalyx.internationalbrainlab.org'
BWM_RELEASE = '/app/code/code_zhang2025/data/bwm_release.csv'

ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
QC_LABEL = 1.0          # 'well-isolated' units: all three RIGOR single-unit metrics pass
NON_GREY = ('root', 'void')
MIN_TRIALS = 2          # decoder needs >= 2 trials per session
MIN_NEURONS = 5         # data paper: >= 5 well-isolated neurons per session
NCLASSES = 3            # wheel speed / whisker motion energy discretisation

INPUT_NAMES = ['time_from_stim_onset', 'trial_number_in_block']
OUTPUT_NAMES = ['choice', 'prior_prob_left', 'wheel_speed', 'whisker_motion_energy']
OUTPUT_VALUES = [
    ['left', 'right'],
    ['p(left)=0.2', 'p(left)=0.5', 'p(left)=0.8'],
    ['low', 'medium', 'high'],
    ['low', 'medium', 'high'],
]

_ONE = None


def get_one():
    """ONE client, built exactly as in the reference caching script.

    The staged cache contains Alyx REST responses that do not expire until 2076, so
    this works offline while still resolving the revised datasets that are the ones
    actually present on disk.
    """
    global _ONE
    if _ONE is None:
        _ONE = ONE(base_url=BASE_URL, silent=True, cache_dir=CACHE_DIR)
    return _ONE


# ------------------------------------------------------------------------ loading
def load_spiking_data(one, pid, eid, pname, qc=QC_LABEL):
    """Reference ``ibl_data_utils.load_spiking_data`` without the network-only ``fs`` read.

    Returns spikes dict and the cluster table restricted to clusters with
    ``label >= qc`` (``qc=1`` -> the data paper's well-isolated neurons).
    """
    loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = loader.load_spike_sorting()
    if not clusters or not spikes:
        # No spike sorting available for this insertion (e.g. it was not resolved);
        # the probe is skipped and the remaining probes of the session are used.
        return None, None
    clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
    if qc is None:
        return spikes, clusters_labeled
    iok = clusters_labeled['label'] >= qc
    load_spiking_data.n_all = len(clusters_labeled)
    selected_clusters = clusters_labeled[iok]
    spike_idx, ib = ismember(spikes['clusters'], selected_clusters.index)
    selected_clusters = selected_clusters.reset_index(drop=True)
    selected_spikes = {k: v[spike_idx] for k, v in spikes.items()}
    selected_spikes['clusters'] = selected_clusters.index[ib].astype(np.int32)
    return selected_spikes, selected_clusters


def load_session_neural(one, eid):
    """Load, merge and QC-filter the spike sorting of every probe of a session.

    Returns (spike_times sorted, spike_cluster_index, beryl_acronym per kept cluster,
             n_clusters_before_qc).
    """
    pids, pnames = one.eid2pid(eid)
    spikes_list, clusters_list, skipped_probes = [], [], []
    n_clusters_all = 0
    for pid, pname in zip(pids, pnames):
        sp, cl = load_spiking_data(one, str(pid), eid, pname)
        if cl is None:
            skipped_probes.append(pname)
            continue
        n_clusters_all += getattr(load_spiking_data, 'n_all', 0)
        clusters_list.append(cl)
        spikes_list.append(sp)
    if not clusters_list:
        return None
    spikes, clusters = merge_probes(spikes_list, clusters_list)

    # Beryl mapping (reference ``list_brain_regions``) and grey-matter restriction
    beryl = BrainRegions().acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl')
    keep = ~np.isin(beryl, NON_GREY)
    keep_ids = np.flatnonzero(keep)
    if keep_ids.size == 0:
        return None
    smask = np.isin(spikes['clusters'], keep_ids)
    st = spikes['times'][smask]
    sc = spikes['clusters'][smask]
    # remap cluster ids to 0..n_keep-1 preserving order
    remap = np.full(len(clusters), -1, dtype=np.int64)
    remap[keep_ids] = np.arange(keep_ids.size)
    sc = remap[sc]
    order = np.argsort(st, kind='stable')
    return (st[order], sc[order], np.asarray(beryl)[keep_ids], keep_ids.size,
            skipped_probes, n_clusters_all, len(clusters), len(pids))


# ------------------------------------------------------------------------ binning
def bin_spikes(spike_times, spike_clusters, t0s, n_neurons, nbins=NBINS, binsize=BINSIZE):
    """Spike counts in (trial, neuron, bin).

    Vectorised equivalent of the reference ``get_spike_data_per_interval`` /
    ``bincount2D`` loop: bin k of a trial covers [t0 + k*binsize, t0 + (k+1)*binsize).
    """
    ntrials = len(t0s)
    out = np.zeros((ntrials, n_neurons, nbins), dtype=np.float32)
    t1s = t0s + nbins * binsize
    i0 = np.searchsorted(spike_times, t0s, side='left')
    i1 = np.searchsorted(spike_times, t1s, side='left')
    for k in range(ntrials):
        a, b = i0[k], i1[k]
        if b <= a:
            continue
        bins = ((spike_times[a:b] - t0s[k]) / binsize).astype(np.int64)
        np.clip(bins, 0, nbins - 1, out=bins)
        idx = spike_clusters[a:b] * nbins + bins
        counts = np.bincount(idx, minlength=n_neurons * nbins)
        out[k] = counts.reshape(n_neurons, nbins).astype(np.float32)
    return out


def bin_behavior(times, values, t0s, nbins=NBINS, binsize=BINSIZE):
    """Interpolate a continuous behavioural signal onto the bin right edges.

    Mirrors the reference ``get_behavior_per_interval``: the target grid is
    ``linspace(t_beg + binsize, t_end, nbins)`` and a trial is rejected when the
    behavioural samples do not cover the interval to within one bin (or contain NaNs).
    """
    ntrials = len(t0s)
    t1s = t0s + nbins * binsize
    grid = t0s[:, None] + (np.arange(1, nbins + 1)[None, :]) * binsize
    vals = np.full((ntrials, nbins), np.nan, dtype=np.float32)
    valid = np.zeros(ntrials, dtype=bool)
    if times is None or values is None or len(times) == 0:
        return vals, valid
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    # ignore NaN samples for interpolation but use them for the validity test
    finite = np.isfinite(times) & np.isfinite(values)
    i_beg = np.searchsorted(times, t0s, side='right')
    i_end = np.searchsorted(times, t1s, side='left')
    interp = np.interp(grid.ravel(), times[finite], values[finite]).reshape(grid.shape)
    for k in range(ntrials):
        a, b = i_beg[k], i_end[k]
        if b <= a:                                   # 'target data not present'
            continue
        if not np.all(finite[a:b]):                  # NaNs inside the interval
            continue
        if abs(t0s[k] - times[a]) > binsize:         # 'target data starts too late'
            continue
        if abs(t1s[k] - times[b - 1]) > binsize:     # 'target data ends too early'
            continue
        valid[k] = True
    vals = interp.astype(np.float32)
    vals[~valid] = 0.0
    return vals, valid


def discretize_tertiles(values):
    """Discretise a (ntrials, nbins) continuous signal into 3 per-session tertile bins."""
    edges = np.percentile(values, [100.0 / 3.0, 200.0 / 3.0])
    classes = np.digitize(values, edges, right=False).astype(np.int32)
    return classes, edges


def trial_number_in_block(prob_left):
    """0-based index of each trial within its block of constant ``probabilityLeft``."""
    pl = np.asarray(prob_left, dtype=np.float64)
    changed = np.ones(len(pl), dtype=bool)
    changed[1:] = ~(pl[1:] == pl[:-1])       # NaN comparisons are False -> new block
    block_id = np.cumsum(changed) - 1
    idx = np.zeros(len(pl), dtype=np.int64)
    for b in np.unique(block_id):
        m = block_id == b
        idx[m] = np.arange(m.sum())
    return idx, block_id


# ------------------------------------------------------------------- session pipeline
def process_session(args):
    """Convert a single session. Returns a dict (or a dict with 'skip' set)."""
    eid, subject, lab, show_processing = args
    t_start = time.time()
    timings = {}
    try:
        one = get_one()

        # --- trials + reference trial mask
        t = time.time()
        sess_loader = SessionLoader(one=one, eid=eid)
        trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0,
                                            sess_loader=sess_loader)
        timings['trials'] = time.time() - t
        mask = mask.to_numpy()
        n_trials_raw = len(trials)

        # --- behaviour streams
        t = time.time()
        sess_loader.load_wheel()
        wheel_times = sess_loader.wheel['times'].to_numpy()
        wheel_speed = np.abs(sess_loader.wheel['velocity'].to_numpy())
        # Whisker motion energy: the reference code uses the left camera and falls back
        # to the right one when the left is unavailable. Some sessions have a camera whose
        # motion energy covers only part of the session, so when both cameras exist the one
        # that covers more trials is used (left preferred on a tie, as in the reference).
        cameras = {}
        for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
            try:
                sess_loader.load_motion_energy(views=[view])
                df = sess_loader.motion_energy[cam]
                cameras[cam] = (df['times'].to_numpy(), df['whiskerMotionEnergy'].to_numpy())
            except Exception:
                continue
        timings['behavior'] = time.time() - t
        if not cameras:
            return {'eid': eid, 'skip': 'no whisker motion energy'}

        # --- neural data
        t = time.time()
        neural = load_session_neural(one, eid)
        timings['spikes'] = time.time() - t
        if neural is None:
            return {'eid': eid, 'skip': 'no units pass QC'}
        (spike_times, spike_clusters, regions, n_units, skipped_probes,
         n_clusters_all, n_good_units, n_probes) = neural
        if n_units < MIN_NEURONS:
            return {'eid': eid,
                    'skip': f'only {n_units} well-isolated grey-matter units '
                            f'(< {MIN_NEURONS})'}

        # --- trial intervals
        align = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
        finite_align = np.isfinite(align)
        t0s_all = align + TIME_WINDOW[0]

        # --- behaviour binning + validity (on all trials, then combine masks)
        t = time.time()
        safe_t0 = np.where(finite_align, t0s_all, np.nanmedian(t0s_all[finite_align]))
        ws, ws_valid = bin_behavior(wheel_times, wheel_speed, safe_t0)
        best = None
        for cam in ('leftCamera', 'rightCamera'):
            if cam not in cameras:
                continue
            me_c, valid_c = bin_behavior(cameras[cam][0], cameras[cam][1], safe_t0)
            if best is None or valid_c.sum() > best[2].sum():
                best = (cam, me_c, valid_c)
        camera_used, me, me_valid = best
        timings['bin_behavior'] = time.time() - t

        keep = mask & finite_align & ws_valid & me_valid
        n_keep = int(keep.sum())
        if n_keep < MIN_TRIALS:
            return {'eid': eid, 'skip': f'only {n_keep} usable trials'}

        # --- spike binning (kept trials only)
        t = time.time()
        t0s = t0s_all[keep]
        counts = bin_spikes(spike_times, spike_clusters, t0s, n_units)
        timings['bin_spikes'] = time.time() - t

        # Drop trials with no spikes at all: these fall in gaps of the ephys recording
        # (verified: e.g. a 5.1 s gap in b182b754) or after the recording ended while
        # the behavioural session continued (8c2f7f4d). They carry no neural
        # information, so they cannot contribute to decoding.
        nonempty = counts.sum(axis=(1, 2)) > 0
        n_empty = int((~nonempty).sum())
        if n_empty:
            kept_idx = np.flatnonzero(keep)
            keep[kept_idx[~nonempty]] = False
            counts = counts[nonempty]
            t0s = t0s[nonempty]
            n_keep = int(keep.sum())
            if n_keep < MIN_TRIALS:
                return {'eid': eid, 'skip': f'only {n_keep} usable trials'}

        # --- outputs
        choice = trials['choice'].to_numpy()[keep]
        # trials.choice: +1 = left, -1 = right  ->  left = 0, right = 1
        choice_cls = (choice < 0).astype(np.int32)
        pleft = trials['probabilityLeft'].to_numpy()[keep]
        prior_cls = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                               np.isclose(pleft, 0.8)], [0, 1, 2], default=-1).astype(np.int32)
        if np.any(prior_cls < 0):
            return {'eid': eid, 'skip': 'unexpected probabilityLeft values'}

        ws_k, me_k = ws[keep], me[keep]
        ws_cls, ws_edges = discretize_tertiles(ws_k)
        me_cls, me_edges = discretize_tertiles(me_k)

        # --- inputs
        bin_centers = (TIME_WINDOW[0] + (np.arange(NBINS) + 0.5) * BINSIZE).astype(np.float32)
        tib_all, _ = trial_number_in_block(trials['probabilityLeft'].to_numpy())
        tib = tib_all[keep].astype(np.float32)

        inputs, outputs, neural_trials = [], [], []
        for i in range(n_keep):
            inputs.append(np.stack([bin_centers,
                                    np.full(NBINS, tib[i], dtype=np.float32)]))
            outputs.append(np.stack([
                np.full(NBINS, choice_cls[i], dtype=np.int32),
                np.full(NBINS, prior_cls[i], dtype=np.int32),
                ws_cls[i].astype(np.int32),
                me_cls[i].astype(np.int32)]))
            neural_trials.append(np.ascontiguousarray(counts[i]))

        out = {
            'eid': eid, 'subject': subject, 'lab': lab,
            'neural': neural_trials, 'input': inputs, 'output': outputs,
            'regions': list(regions), 'n_units': n_units,
            'n_trials_raw': n_trials_raw, 'n_trials_mask': int(mask.sum()),
            'n_trials_kept': n_keep, 'n_trials_empty_neural': n_empty,
            'camera': camera_used, 'skipped_probes': skipped_probes,
            'n_clusters_all': n_clusters_all, 'n_good_units': n_good_units,
            'n_probes': n_probes,
            'wheel_edges': ws_edges.tolist(), 'whisker_edges': me_edges.tolist(),
            'mean_rate': float(counts.mean() / BINSIZE),
            'timings': timings, 'total_time': time.time() - t_start,
        }
        if show_processing:
            plot_processing(out, trials, keep, ws_k, me_k, wheel_times, wheel_speed,
                           cameras[camera_used][0], cameras[camera_used][1], align)
        return out
    except Exception as e:  # noqa: BLE001
        import traceback
        return {'eid': eid, 'skip': f'error: {e}', 'traceback': traceback.format_exc()[-800:]}


# --------------------------------------------------------------------------- plotting
def plot_processing(sess, trials, keep, ws_k, me_k, wheel_times, wheel_speed,
                   me_times, me_vals, align):
    """Diagnostic figure showing every processing step for one session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    eid = sess['eid']
    nb = NBINS
    edges_t = TIME_WINDOW[0] + np.arange(nb + 1) * BINSIZE
    centers = TIME_WINDOW[0] + (np.arange(nb) + 0.5) * BINSIZE
    right_edges = TIME_WINDOW[0] + (np.arange(1, nb + 1)) * BINSIZE
    kept_idx = np.flatnonzero(keep)
    fig, axs = plt.subplots(5, 3, figsize=(20, 18))

    for col, ti in enumerate(kept_idx[:3]):
        j = int(np.searchsorted(kept_idx, ti))
        a0 = align[ti]
        # 1. spike raster/counts
        ax = axs[0, col]
        ax.imshow(sess['neural'][j], aspect='auto', origin='lower', cmap='Greys',
                  extent=[edges_t[0], edges_t[-1], 0, sess['neural'][j].shape[0]])
        ax.axvline(0, color='r', ls='--')
        ax.set_title(f'{eid[:8]} trial {ti}: spike counts (20 ms bins)')
        ax.set_xlabel('time from stimOn (s)'); ax.set_ylabel('neuron')

        # 2. wheel speed: raw vs binned
        ax = axs[1, col]
        m = (wheel_times >= a0 + TIME_WINDOW[0] - 0.05) & (wheel_times <= a0 + TIME_WINDOW[1] + 0.05)
        ax.plot(wheel_times[m] - a0, wheel_speed[m], 'k-', lw=0.8, label='raw |velocity|')
        ax.plot(right_edges, ws_k[j], 'o-', ms=3, color='tab:blue', label='binned (bin right edge)')
        ax.axvline(0, color='r', ls='--')
        fm = trials['firstMovement_times'].to_numpy()[ti] - a0
        ax.axvline(fm, color='g', ls=':', label='first movement')
        ax.set_ylabel('wheel speed (rad/s)'); ax.legend(fontsize=7)

        # 3. wheel discretisation
        ax = axs[2, col]
        ax.plot(right_edges, ws_k[j], 'o-', ms=3, color='tab:blue')
        for e in sess['wheel_edges']:
            ax.axhline(e, color='gray', ls='--')
        ax2 = ax.twinx()
        ax2.step(right_edges, sess['output'][j][2], where='mid', color='tab:orange')
        ax2.set_ylim(-0.2, 2.2); ax2.set_ylabel('wheel class', color='tab:orange')
        ax.set_ylabel('wheel speed')

        # 4. whisker ME raw vs binned
        ax = axs[3, col]
        m = (me_times >= a0 + TIME_WINDOW[0] - 0.05) & (me_times <= a0 + TIME_WINDOW[1] + 0.05)
        ax.plot(me_times[m] - a0, me_vals[m], 'k-', lw=0.8, label='raw whisker ME')
        ax.plot(right_edges, me_k[j], 'o-', ms=3, color='tab:purple', label='binned')
        ax.axvline(0, color='r', ls='--'); ax.legend(fontsize=7)
        ax.set_ylabel('whisker ME (a.u.)')

        # 5. whisker discretisation + per-trial variables
        ax = axs[4, col]
        ax.plot(right_edges, me_k[j], 'o-', ms=3, color='tab:purple')
        for e in sess['whisker_edges']:
            ax.axhline(e, color='gray', ls='--')
        ax2 = ax.twinx()
        ax2.step(right_edges, sess['output'][j][3], where='mid', color='tab:orange')
        ax2.set_ylim(-0.2, 2.2); ax2.set_ylabel('whisker class', color='tab:orange')
        ax.set_xlabel('time from stimOn (s)')
        ax.set_title(f"choice={sess['output'][j][0,0]} prior={sess['output'][j][1,0]} "
                     f"tinblock={sess['input'][j][1,0]:.0f} input t=[{centers[0]:.2f},{centers[-1]:.2f}]",
                     fontsize=8)
    fig.suptitle(f'Processing steps, session {eid} ({sess["n_trials_kept"]} trials, '
                 f'{sess["n_units"]} units, camera={sess["camera"]})')
    fig.tight_layout()
    fig.savefig(f'processing_{eid}.png', dpi=110)
    plt.close(fig)

    # session-level summary figure: distributions
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    axs[0].hist(ws_k.ravel(), bins=100)
    for e in sess['wheel_edges']:
        axs[0].axvline(e, color='r')
    axs[0].set_yscale('log'); axs[0].set_title('wheel speed + tertile edges')
    axs[1].hist(me_k.ravel(), bins=100)
    for e in sess['whisker_edges']:
        axs[1].axvline(e, color='r')
    axs[1].set_yscale('log'); axs[1].set_title('whisker ME + tertile edges')
    allout = np.stack(sess['output'])
    axs[2].bar(range(2), [np.mean(allout[:, 0, 0] == c) for c in range(2)])
    axs[2].set_title('choice fractions (0=left,1=right)')
    axs[3].bar(range(3), [np.mean(allout[:, 1, 0] == c) for c in range(3)])
    axs[3].set_title('prior fractions (0.2,0.5,0.8)')
    fig.tight_layout(); fig.savefig(f'processing_{eid}_dist.png', dpi=110); plt.close(fig)


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--n-workers', type=int, default=24)
    args = ap.parse_args()

    t_start = time.time()
    bwm = pd.read_csv(BWM_RELEASE, index_col=0)
    sess_df = bwm.drop_duplicates('eid')[['eid', 'subject', 'lab']].sort_values('eid')
    if args.sample:
        sess_df = sess_df.iloc[:2]
    print(f'Converting {len(sess_df)} sessions (sample={args.sample}); '
          f'align={ALIGN_TIME}, window={TIME_WINDOW}, binsize={BINSIZE}, T={NBINS}')

    jobs = [(r.eid, r.subject, r.lab, args.show_processing) for r in sess_df.itertuples()]
    results = []
    if len(jobs) == 1 or args.n_workers <= 1 or args.sample:
        for j in jobs:
            results.append(process_session(j))
            r = results[-1]
            print(f"  {r['eid'][:8]}: " + (r['skip'] if 'skip' in r else
                  f"{r['n_trials_kept']} trials, {r['n_units']} units, "
                  f"{r['total_time']:.1f}s, timings={ {k: round(v,2) for k,v in r['timings'].items()} }"), flush=True)
    else:
        with Pool(args.n_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(process_session, jobs)):
                results.append(r)
                msg = r['skip'] if 'skip' in r else (
                    f"{r['n_trials_kept']} trials, {r['n_units']} units, {r['total_time']:.1f}s")
                print(f'  [{i+1}/{len(jobs)}] {r["eid"][:8]}: {msg}', flush=True)

    skipped = [r for r in results if 'skip' in r]
    good = [r for r in results if 'skip' not in r]
    good.sort(key=lambda r: r['eid'])
    print(f'\nProcessed {len(good)} sessions, skipped {len(skipped)}')
    for r in skipped:
        print(f"  SKIP {r['eid']}: {r['skip']}")
        if 'traceback' in r:
            print(r['traceback'])

    # ------------------------------------------------------------------ assemble
    subjects = sorted({r['subject'] for r in good})
    sub_idx = np.array([subjects.index(r['subject']) for r in good], dtype=np.int64)
    brain_regions = sorted({reg for r in good for reg in r['regions']})
    reg_lookup = {reg: i for i, reg in enumerate(brain_regions)}
    brain_region_idx = [np.array([reg_lookup[reg] for reg in r['regions']], dtype=np.int64)
                        for r in good]

    data = {
        'neural': [r['neural'] for r in good],
        'input': [r['input'] for r in good],
        'output': [r['output'] for r in good],
        'subjects': subjects,
        'subject_idx': sub_idx,
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description': (
                'IBL decision-making task (brain-wide map). A Gabor stimulus of one of five '
                'contrasts appears left or right; the mouse turns a wheel to bring it to the '
                'centre. After 90 unbiased trials the prior probability of a left stimulus '
                'alternates in blocks between 0.2 and 0.8. Decoder inputs: time from stimulus '
                'onset and trial number within the current block. Decoder outputs: the '
                'mouse choice (left/right), the block prior probability of left (0.2/0.5/0.8), '
                'and wheel speed and whisker motion energy each discretised into 3 '
                'per-session tertile bins.'),
            'time_bin_size': BINSIZE * 1000.0,
            'temporal_alignment_event': 'visual stimulus onset (trials.stimOn_times)',
            'off_start': TIME_WINDOW[0],
            'off_end': TIME_WINDOW[1],
            'n_time_bins': NBINS,
            'neural_units': 'spike counts per 20 ms bin',
            'alignment_note': ('bin k spans [stimOn - 0.5 + 0.02k, stimOn - 0.5 + 0.02(k+1)); '
                               'the time input is the bin centre; continuous behaviours are '
                               'interpolated at the bin right edge, as in the reference code'),
            'session_info': [
                {'eid': r['eid'], 'subject': r['subject'], 'lab': r['lab'],
                 'n_trials_raw': r['n_trials_raw'], 'n_trials_pass_mask': r['n_trials_mask'],
                 'n_trials_kept': r['n_trials_kept'], 'n_neurons': r['n_units'],
                 'n_trials_dropped_empty_neural': r['n_trials_empty_neural'],
                 'camera': r['camera'], 'mean_firing_rate_hz': r['mean_rate'],
                 'probes_without_spike_sorting': r['skipped_probes'],
                 'n_probes': r['n_probes'],
                 'n_clusters_before_qc': r['n_clusters_all'],
                 'n_units_pass_qc': r['n_good_units'],
                 'wheel_speed_tertile_edges': r['wheel_edges'],
                 'whisker_me_tertile_edges': r['whisker_edges']}
                for r in good],
            'neuron_curation': ('clusters.label >= 1 (all three IBL RIGOR single-unit metrics '
                                'passed: amplitude > 50 uV, noise cutoff < 20 uV, refractory '
                                'period violation) and Beryl region not in {root, void} '
                                '(grey matter only); probes of a session merged; '
                                'sessions with fewer than 5 such units dropped'),
            'trial_curation': ('reference load_trials_and_mask(max_trial_len=10.0): no NaN in '
                               'stimOn_times/choice/feedback_times/probabilityLeft/'
                               'firstMovement_times/feedbackType, first movement 0.08-2.0 s '
                               'after stimulus onset, choice != 0, feedback-goCue <= 10 s; '
                               'plus complete wheel and whisker-motion-energy coverage of the '
                               '2 s window, plus trials with zero spikes in the whole '
                               'window (ephys recording gaps) dropped'),
            'discretization': ('wheel speed and whisker motion energy: 3 classes at the '
                               'per-session 33.3/66.7 percentiles over all kept trials x bins'),
            'source': 'IBL public brain-wide map release (bwm_release.csv), loaded with ONE',
            'skipped_sessions': [{'eid': r['eid'], 'reason': r['skip']} for r in skipped],
        },
    }

    ntrials = sum(len(x) for x in data['neural'])
    nneurons = sum(len(x) for x in brain_region_idx)
    print(f'\nSummary: {len(good)} sessions, {len(subjects)} subjects, {ntrials} trials, '
          f'{nneurons} neurons, {len(brain_regions)} Beryl regions')
    print(f'  trials/session: mean {ntrials/max(1,len(good)):.1f}')
    print(f'  neurons/session: mean {nneurons/max(1,len(good)):.1f}')
    n_pre = sum(r['n_clusters_all'] for r in good)
    n_qc = sum(r['n_good_units'] for r in good)
    n_probes = sum(r['n_probes'] for r in good)
    print(f'  clusters before QC: {n_pre} ({n_pre/max(1,n_probes):.1f}/probe); '
          f'label>=1: {n_qc} ({n_qc/max(1,n_probes):.1f}/probe); '
          f'after grey-matter filter: {nneurons}')

    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f'Wrote {args.outfile} ({os.path.getsize(args.outfile)/1e9:.2f} GB) in '
          f'{time.time()-t_start:.1f}s')


if __name__ == '__main__':
    main()
