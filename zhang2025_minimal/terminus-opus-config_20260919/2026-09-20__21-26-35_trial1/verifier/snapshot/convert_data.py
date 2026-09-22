#!/usr/bin/env python
"""
Convert the IBL brain-wide map (BWM) dataset into the decoder dictionary format.

Processing follows the reference pipeline of Zhang et al. 2025
(`/app/code/code_zhang2025/src/0_data_caching.py` and `src/utils/ibl_data_utils.py`)
and the inclusion criteria of the IBL brain-wide-map data paper.

Decisions taken in this conversion:

* Sessions  : the 459 eids of the public BWM release freeze
              (`code_zhang2025/data/bwm_release.csv`), i.e. exactly the sessions the
              reference code iterates over.
* Probes    : all probes of a session are merged into one population
              (`merge_probes`), as in the reference code and in the data paper
              ("neurons in the same session ... were combined across probes").
* Neurons   : well-isolated neurons only, i.e. clusters with IBL label == 1, which is
              the conjunction of the three RIGOR single-unit metrics used by the data
              paper (amplitude > 50 uV, noise cut-off < 20 uV, refractory-period
              violation).  Additionally restricted to grey matter (Beryl acronym not
              'root'/'void'), again as in the data paper.  The reference caching script
              keeps every Kilosort unit; that is impossible here because 622k units x
              197k trials x 100 bins is ~164 GB, while the well-isolated set (75.7k
              neurons, the number quoted by the data paper) is ~11 GB.
* Trials    : the standard BWM trial mask (`load_trials_and_mask`) with
              max_trial_len=10 s, exactly as `prepare_data` calls it: no NaN in
              stimOn_times / choice / feedback_times / probabilityLeft /
              firstMovement_times / feedbackType, 0.08 s <= reaction time <= 2 s,
              choice != 0, feedback_times - goCue_times <= 10 s.  Trials for which the
              behavioural traces do not cover the decoding window are dropped as well
              (`get_behavior_per_interval`).
* Alignment : stimulus onset (`stimOn_times`), window (-0.5, +1.5) s, 20 ms
              non-overlapping bins -> T = 100 - the exact `params` dict of
              `0_data_caching.py`.
"""

import os
import sys
import glob
import time
import pickle
import argparse
import warnings
import traceback
import multiprocessing as mp

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------------------
# parameters (identical to code_zhang2025/src/0_data_caching.py)
# --------------------------------------------------------------------------------------
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)          # seconds relative to the alignment event
BINSIZE = 0.02                     # seconds
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
MAX_TRIAL_LEN = 10.0               # prepare_data(..., max_trial_len=10.0)
MIN_RT, MAX_RT = 0.08, 2.0
MIN_TRIALS_PER_SESSION = 2         # required by the decoder
MIN_NEURONS_PER_SESSION = 1

BWM_FREEZE = '/app/code/code_zhang2025/data/bwm_release.csv'
DATALIMIT = '/app/data/DATALIMIT_SUBSET.csv'
OUT_FILE = '/app/converted_data.pkl'
TMP_DIR = '/app/work/sessions'

# time stamp of each bin, relative to the alignment event.  These are the right edges of
# the bins, i.e. the sample times used by `get_behavior_per_interval` in the reference
# code (x_interp = linspace(beg + binsize, end, n_bins)).
BIN_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)


def get_one():
    from one.api import ONE
    # No password / no explicit mode: the staged Alyx token plus the cached REST
    # responses under <cache>/.rest let ONE answer every query from disk.
    return ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True)


# --------------------------------------------------------------------------------------
# trial table + mask  (ibl_data_utils.load_trials_and_mask)
# --------------------------------------------------------------------------------------
def load_trials_and_mask(sess_loader, min_rt=MIN_RT, max_rt=MAX_RT,
                         max_trial_len=MAX_TRIAL_LEN):
    nan_exclude = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
                   'firstMovement_times', 'feedbackType']
    if sess_loader.trials.empty:
        sess_loader.load_trials()
    trials = sess_loader.trials

    query = f'(firstMovement_times - stimOn_times < {min_rt})'
    query += f' | (firstMovement_times - stimOn_times > {max_rt})'
    if max_trial_len is not None:
        query += f' | (feedback_times - goCue_times > {max_trial_len})'
    for event in nan_exclude:
        query += f' | {event}.isnull()'
    query += ' | (choice == 0)'
    mask = ~trials.eval(query)
    return trials, mask.to_numpy()


# --------------------------------------------------------------------------------------
# spike binning
# --------------------------------------------------------------------------------------
def bin_spikes(spike_times, spike_clusters, n_clusters, interval_begs):
    """Bin spikes into (n_trials, n_clusters, N_BINS) spike-count arrays.

    Equivalent to ibl_data_utils.get_spike_data_per_interval (bincount2D with
    xlim=[t_beg, t_end], keeping the first N_BINS bins) but vectorised.
    """
    n_trials = len(interval_begs)
    out = np.zeros((n_trials, n_clusters, N_BINS), dtype=np.float32)
    order = np.argsort(spike_times, kind='stable')
    st = spike_times[order]
    sc = spike_clusters[order]
    beg = np.searchsorted(st, interval_begs, side='left')
    end = np.searchsorted(st, interval_begs + N_BINS * BINSIZE, side='left')
    for k in range(n_trials):
        b, e = beg[k], end[k]
        if e <= b:
            continue
        tb = np.floor((st[b:e] - interval_begs[k]) / BINSIZE).astype(np.int64)
        np.clip(tb, 0, N_BINS - 1, out=tb)
        flat = sc[b:e] * N_BINS + tb
        counts = np.bincount(flat, minlength=n_clusters * N_BINS)
        out[k] = counts.reshape(n_clusters, N_BINS)
    return out


# --------------------------------------------------------------------------------------
# behaviour binning  (ibl_data_utils.get_behavior_per_interval)
# --------------------------------------------------------------------------------------
def bin_behavior(target_times, target_vals, interval_begs):
    """Interpolate a continuous behavioural trace onto the trial bins.

    Returns (values (n_trials, N_BINS), good (n_trials,) bool).
    A trial is flagged bad when the trace does not cover the window (the
    'starts too late' / 'ends too early' checks of the reference code) or when the
    window contains NaNs.
    """
    from scipy.interpolate import interp1d
    n_trials = len(interval_begs)
    vals = np.zeros((n_trials, N_BINS), dtype=np.float64)
    good = np.zeros(n_trials, dtype=bool)
    if target_times is None or target_vals is None:
        return vals, good
    interval_ends = interval_begs + N_BINS * BINSIZE
    ib = np.searchsorted(target_times, interval_begs, side='right')
    ie = np.searchsorted(target_times, interval_ends, side='left')
    for k in range(n_trials):
        tt = target_times[ib[k]:ie[k]]
        tv = target_vals[ib[k]:ie[k]]
        if len(tv) == 0:
            continue
        if np.any(np.isnan(tv)):
            continue
        if abs(interval_begs[k] - tt[0]) > BINSIZE:      # data starts too late
            continue
        if abs(interval_ends[k] - tt[-1]) > BINSIZE:     # data ends too early
            continue
        x = np.linspace(interval_begs[k] + BINSIZE, interval_ends[k], N_BINS)
        vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
        good[k] = True
    return vals, good


def discretize3(x):
    """Discretize a continuous signal into 3 bins by its tertiles.

    The tertiles are computed per session: whisker motion energy is in camera-specific
    arbitrary units (left and right cameras differ in resolution and gain) and wheel
    speed also varies in scale across rigs, so a session-wise split is the only one that
    means the same thing (low / medium / high for this mouse in this session) everywhere.
    """
    q = np.quantile(x, [1. / 3., 2. / 3.])
    edges = np.unique(q)
    return np.searchsorted(edges, x, side='right').astype(np.int64)


# --------------------------------------------------------------------------------------
# one session
# --------------------------------------------------------------------------------------
def process_session(arg):
    eid, rows = arg
    try:
        return _process_session(eid, rows)
    except Exception:
        return {'eid': eid, 'skip': 'exception: ' + traceback.format_exc().splitlines()[-1]}


def _process_session(eid, rows):
    from brainbox.io.one import SpikeSortingLoader, SessionLoader
    from iblatlas.regions import BrainRegions

    one = get_one()
    br = BrainRegions()

    # ---------------- spikes: load every probe and merge -------------------------------
    spikes_list, clusters_list = [], []
    for _, r in rows.iterrows():
        ssl = SpikeSortingLoader(pid=r.pid, one=one, eid=eid, pname=r.probe_name)
        sp, cl, ch = ssl.load_spike_sorting()
        if sp is None or len(sp) == 0:
            continue
        cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
        spikes_list.append(sp)
        clusters_list.append(cld)
    if len(spikes_list) == 0:
        return {'eid': eid, 'skip': 'no spike sorting'}

    cmax = 0
    ms, mc = [], []
    for cl, sp in zip(clusters_list, spikes_list):
        sp = dict(sp)
        sp['clusters'] = sp['clusters'] + cmax
        cmax = cl.index.max() + 1
        ms.append(sp)
        mc.append(cl)
    clusters = pd.concat(mc, ignore_index=True)
    spike_times = np.concatenate([s['times'] for s in ms])
    spike_clusters = np.concatenate([s['clusters'] for s in ms])

    # ---------------- neuron selection -------------------------------------------------
    beryl = np.asarray(br.acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl'))
    label = clusters['label'].to_numpy()
    keep = (label >= 1) & ~np.isin(beryl, ['root', 'void'])
    keep_ids = np.nonzero(keep)[0]
    n_neurons = len(keep_ids)
    if n_neurons < MIN_NEURONS_PER_SESSION:
        return {'eid': eid, 'skip': f'only {n_neurons} well-isolated neurons'}

    sel = np.isin(spike_clusters, keep_ids)
    spike_times = spike_times[sel]
    # remap cluster ids to 0..n_neurons-1
    remap = np.full(int(clusters.index.max()) + 1, -1, dtype=np.int64)
    remap[keep_ids] = np.arange(n_neurons)
    spike_clusters = remap[spike_clusters[sel]]
    regions = beryl[keep_ids]

    # ---------------- trials -----------------------------------------------------------
    sl = SessionLoader(one=one, eid=eid)
    trials, trials_mask = load_trials_and_mask(sl)
    align = trials[ALIGN_TIME].to_numpy(dtype=float)
    valid = trials_mask & np.isfinite(align)
    if valid.sum() < MIN_TRIALS_PER_SESSION:
        return {'eid': eid, 'skip': f'only {int(valid.sum())} trials pass the mask'}

    interval_begs = align + TIME_WINDOW[0]

    # ---------------- behaviour --------------------------------------------------------
    sl.load_wheel()
    wheel_speed_t = sl.wheel['times'].to_numpy()
    wheel_speed_v = np.abs(sl.wheel['velocity'].to_numpy())

    whisker_t = whisker_v = None
    for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
        try:
            sl.load_motion_energy(views=[view])
            me = sl.motion_energy[cam]
            whisker_t = me['times'].to_numpy()
            whisker_v = me['whiskerMotionEnergy'].to_numpy()
            break
        except Exception:
            continue
    if whisker_t is None:
        return {'eid': eid, 'skip': 'no whisker motion energy'}

    ws, ws_good = bin_behavior(wheel_speed_t, wheel_speed_v, interval_begs)
    wm, wm_good = bin_behavior(whisker_t, whisker_v, interval_begs)

    valid = valid & ws_good & wm_good
    if valid.sum() < MIN_TRIALS_PER_SESSION:
        return {'eid': eid, 'skip': f'only {int(valid.sum())} trials with complete behaviour'}

    # ---------------- trial number within block ----------------------------------------
    pleft = trials['probabilityLeft'].to_numpy(dtype=float)
    new_block = np.ones(len(pleft), dtype=bool)
    new_block[1:] = pleft[1:] != pleft[:-1]
    block_id = np.cumsum(new_block) - 1
    trial_in_block = np.zeros(len(pleft), dtype=np.int64)
    for b in np.unique(block_id):
        idx = np.nonzero(block_id == b)[0]
        trial_in_block[idx] = np.arange(len(idx))

    # ---------------- restrict everything to the kept trials ---------------------------
    tidx = np.nonzero(valid)[0]
    binned = bin_spikes(spike_times, spike_clusters, n_neurons, interval_begs[tidx])

    ws = ws[tidx]
    wm = wm[tidx]
    choice = trials['choice'].to_numpy()[tidx]          # +1 = left, -1 = right
    pleft_t = pleft[tidx]
    tinb = trial_in_block[tidx]

    # outputs
    out_choice = (choice < 0).astype(np.int64)          # left = 0, right = 1
    prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
    out_prior = np.array([prior_map[round(float(p), 1)] for p in pleft_t], dtype=np.int64)
    out_ws = discretize3(ws)
    out_wm = discretize3(wm)

    n_trials = len(tidx)
    outputs = np.empty((n_trials, 4, N_BINS), dtype=np.int64)
    outputs[:, 0, :] = out_choice[:, None]
    outputs[:, 1, :] = out_prior[:, None]
    outputs[:, 2, :] = out_ws
    outputs[:, 3, :] = out_wm

    inputs = np.empty((n_trials, 2, N_BINS), dtype=np.float32)
    inputs[:, 0, :] = BIN_TIMES[None, :]
    inputs[:, 1, :] = tinb[:, None].astype(np.float32)

    return {
        'eid': eid,
        'skip': None,
        'neural': binned,                       # (n_trials, n_neurons, N_BINS) float32
        'input': inputs,
        'output': outputs,
        'regions': regions,
        'subject': str(rows.subject.iloc[0]),
        'lab': str(rows.lab.iloc[0]),
        'date': str(rows.date.iloc[0]),
        'n_probes': int(len(rows)),
        'n_trials_total': int(len(trials)),
    }


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-sessions', type=int, default=None)
    ap.add_argument('--n-workers', type=int, default=16)
    ap.add_argument('--out', type=str, default=OUT_FILE)
    ap.add_argument('--tmp', type=str, default=TMP_DIR)
    ap.add_argument('--assemble-only', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.tmp, exist_ok=True)

    bwm = pd.read_csv(BWM_FREEZE, index_col=0)
    if os.path.exists(DATALIMIT):
        sub = pd.read_csv(DATALIMIT)
        col = 'eid' if 'eid' in sub.columns else sub.columns[0]
        bwm = bwm[bwm.eid.isin(sub[col].astype(str))]
        print(f'DATALIMIT_SUBSET.csv found: restricting to {bwm.eid.nunique()} sessions')

    eids = list(dict.fromkeys(bwm.eid.tolist()))
    if args.n_sessions is not None:
        eids = eids[:args.n_sessions]
    print(f'{len(eids)} sessions to process')

    jobs = [(eid, bwm[bwm.eid == eid]) for eid in eids
            if not os.path.exists(os.path.join(args.tmp, eid + '.pkl'))]

    if not args.assemble_only and jobs:
        t0 = time.time()
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=args.n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(process_session, jobs)):
                with open(os.path.join(args.tmp, res['eid'] + '.pkl'), 'wb') as f:
                    pickle.dump(res, f, protocol=4)
                msg = res['skip'] if res['skip'] else \
                    f"{res['neural'].shape[0]} trials x {res['neural'].shape[1]} neurons"
                print(f'[{i+1}/{len(jobs)}] {res["eid"]}: {msg}  ({time.time()-t0:.0f}s)',
                      flush=True)

    # ---------------- assemble ---------------------------------------------------------
    data = {'neural': [], 'input': [], 'output': [],
            'subjects': [], 'subject_idx': [],
            'brain_regions': [], 'brain_region_idx': []}
    subjects, regions_all = [], []
    session_info = []
    n_skipped = 0
    for eid in eids:
        p = os.path.join(args.tmp, eid + '.pkl')
        if not os.path.exists(p):
            continue
        with open(p, 'rb') as f:
            res = pickle.load(f)
        if res['skip']:
            n_skipped += 1
            continue
        nt = res['neural'].shape[0]
        data['neural'].append([np.ascontiguousarray(res['neural'][k]) for k in range(nt)])
        data['input'].append([np.ascontiguousarray(res['input'][k]) for k in range(nt)])
        data['output'].append([np.ascontiguousarray(res['output'][k]) for k in range(nt)])
        if res['subject'] not in subjects:
            subjects.append(res['subject'])
        data['subject_idx'].append(subjects.index(res['subject']))
        for r in res['regions']:
            if r not in regions_all:
                regions_all.append(r)
        data['brain_region_idx'].append(
            np.array([regions_all.index(r) for r in res['regions']], dtype=np.int64))
        session_info.append({'eid': eid, 'subject': res['subject'], 'lab': res['lab'],
                             'date': res['date'], 'n_probes': res['n_probes'],
                             'n_neurons': int(res['neural'].shape[1]),
                             'n_trials': int(nt),
                             'n_trials_total': res['n_trials_total']})

    data['subjects'] = subjects
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
    data['brain_regions'] = regions_all
    data['input_names'] = ['time_from_stimulus_onset', 'trial_number_in_block']
    data['output_names'] = ['choice', 'prior', 'wheel_speed', 'whisker_motion_energy']
    data['output_values'] = [
        ['left', 'right'],
        ['p(left)=0.2', 'p(left)=0.5', 'p(left)=0.8'],
        ['low', 'medium', 'high'],
        ['low', 'medium', 'high'],
    ]
    data['metadata'] = {
        'task_description': (
            'IBL decision-making task: a Gabor patch appears left or right of centre '
            'with one of 5 contrasts and the mouse reports its side by turning a wheel. '
            'After 90 unbiased (50:50) trials, the stimulus side is drawn from blocks of '
            '20:80 or 80:20 prior probability, of 20-100 trials, uncued. Decoded from '
            'the population spike counts are the mouse choice (left/right), the prior '
            'probability of the stimulus appearing on the left (0.2/0.5/0.8), and the '
            'time-varying wheel speed and whisker-pad motion energy, each discretized '
            'into 3 (session-wise tertile) bins.'),
        'time_bin_size': BINSIZE * 1000.0,
        'temporal_alignment_event': 'stimulus onset (stimOn_times)',
        'off_start': TIME_WINDOW[0],
        'off_end': TIME_WINDOW[1],
        'n_timepoints': N_BINS,
        'bin_times_s': [float(t) for t in BIN_TIMES],
        'neural_units': 'spike counts per 20 ms bin',
        'spike_sorting': 'IBL pykilosort (Kilosort 2.5 with IBL additions)',
        'neuron_selection': ("well-isolated neurons only (IBL cluster label == 1, i.e. "
                             "passing all three RIGOR single-unit metrics: amplitude > "
                             "50 uV, noise cut-off < 20 uV, refractory-period violation), "
                             "restricted to grey-matter Beryl regions (root/void "
                             "excluded); all probes of a session merged"),
        'trial_selection': ("IBL brain-wide-map trial mask: no NaN in stimOn_times, "
                            "choice, feedback_times, probabilityLeft, "
                            "firstMovement_times or feedbackType; 0.08 s <= "
                            "firstMovement_times - stimOn_times <= 2 s; choice != 0; "
                            "feedback_times - goCue_times <= 10 s; and complete wheel "
                            "and whisker traces over the decoding window"),
        'brain_region_mapping': 'Allen CCF acronyms mapped to the IBL Beryl atlas',
        'dataset': 'IBL brain-wide map public release freeze (bwm_release.csv)',
        'n_sessions_skipped': n_skipped,
        'session_info': session_info,
    }

    with open(args.out, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f'wrote {args.out}: {len(data["neural"])} sessions, '
          f'{sum(len(s) for s in data["neural"])} trials, '
          f'{sum(s[0].shape[0] for s in data["neural"])} neurons, '
          f'{len(subjects)} subjects, {len(regions_all)} brain regions')


if __name__ == '__main__':
    main()
