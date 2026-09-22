#!/usr/bin/env python3
"""
Convert the Zhong et al. 2025 ("Unsupervised pretraining in biological neural networks")
two-photon mesoscope dataset into the decoder-compatible pickle format.

Usage
-----
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Options
-------
    --full             process all 89 recordings (default)
    --sample           process only 2 recordings (for testing)
    --show-processing  save per-step diagnostic plots for up to 2 sessions as
                       processing_<session_id>.png

Pipeline (see CONVERSION_NOTES.md for the full rationale)
---------------------------------------------------------
1.  `Imaging_Exp_info.npy` -> 89 unique recordings (mouse, date, block).  The
    canonical stimulus id of every wall is merged over all experiment types in
    which the recording appears.
2.  Behaviour (`Beh_<exp_type>.npy`) -> per-frame masks and per-trial variables.
    Retained timepoints follow the reference recipe
    `fr_valid = (ft_move > 0) & ft_CorrSpc` (utils.Get_dprime_selective_neuron),
    i.e. inside the 0-4 m texture corridor while the VR (= the mouse) is moving.
3.  Neural data (`<rec>_neural_data.npy`) -> concatenated deconvolved traces,
    restricted to visual-cortex neurons (`iarea not in {-1, 7}`), subsampled to
    at most NNEURONS_KEEP per session (stratified over V1/mHV/lHV/aHV),
    z-scored per neuron, then split into trials.
"""

import argparse
import datetime
import os
import pickle
import sys
import time
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
ROOT = '/app/data'
SPK_DIR = os.path.join(ROOT, 'spk')
BEH_DIR = os.path.join(ROOT, 'beh')
RET_DIR = os.path.join(ROOT, 'retinotopy')

# Neurons kept per session.  2000 == decoder.train_decoder's `svd_max_neurons` default,
# so the SVD initialisation uses the real neurons instead of a random Gaussian projection.
NNEURONS_KEEP = 2000
NEURON_SEED = 0

# Worker processes used to read the 405 GB of spk files (disk-bandwidth limited).
N_WORKERS = int(os.environ.get('CONVERT_WORKERS', '6'))

# Corridor geometry, in decimetres (beh['Texture_Length'] == 40, i.e. 4 m of texture).
CORRIDOR_DM = 40.0
N_POS_BINS = 4                       # 4 equal-length 1-m bins
POS_BIN_DM = CORRIDOR_DM / N_POS_BINS

N_SPEED_BINS = 4                     # quartiles of the pooled running-speed distribution

# Wall-clock time inputs are clipped to this many seconds (see CONVERSION_NOTES Step 5).
TIME_CLIP_S = 30.0

SEC_PER_DAY = 86400.0

BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV']

# Canonical stimulus roles used throughout the reference code
# (utils.Get_coding_direction docstring / utils.get_cat_id).  Id 7 is added here for
# `circle3`, the only stimulus that never receives an id in Imaging_Exp_info.
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2', 'circle3']
STIM_EXTRA = {'circle3': 7}

INPUT_NAMES = ['time_to_sound_cue_s', 'day_of_training', 'time_since_trial_start_s',
               'reward_available']
OUTPUT_NAMES = ['stimulus', 'licking', 'position_bin', 'running_speed_bin']


# --------------------------------------------------------------------------------------
# Reference helpers (copied from /app/code/utils.py)
# --------------------------------------------------------------------------------------
def neu_area_ID(iarea):
    """utils.neu_area_ID -- map retinotopy area codes onto the four visual areas."""
    idx = {}
    for ar in BRAIN_REGIONS:
        if ar == 'V1':
            idx[ar] = iarea == 8
        elif ar == 'mHV':
            idx[ar] = (iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)
        elif ar == 'lHV':
            idx[ar] = (iarea == 5) | (iarea == 6)
        elif ar == 'aHV':
            idx[ar] = (iarea == 3) | (iarea == 4)
    return idx


def load_spk_planes(mname, datexp, blk):
    """utils.load_spk, but returning the individual plane arrays (avoids one full copy)."""
    fn = '%s_%s_%s_neural_data.npy' % (mname, datexp, blk)
    return np.load(os.path.join(SPK_DIR, fn), allow_pickle=True).item()['spks']


def load_iarea(mname, datexp):
    """utils.load_retino, but only the area code of each neuron."""
    d = np.load(os.path.join(RET_DIR, '%s_%s_trans.npz' % (mname, datexp)), allow_pickle=True)
    return d['iarea']


# --------------------------------------------------------------------------------------
# Step 1: recording table
# --------------------------------------------------------------------------------------
def build_recording_table():
    """Return an ordered list of the 89 unique recordings with their merged stimulus map."""
    exp_info = np.load(os.path.join(BEH_DIR, 'Imaging_Exp_info.npy'), allow_pickle=True).item()

    recs = {}
    for exp_type, dbs in exp_info.items():
        for db in dbs:
            key = '%s_%s_%s' % (db['mname'], db['datexp'], db['blk'])
            behkey = key + '_' + db['stimtype'] if 'stimtype' in db else key
            if key not in recs:
                recs[key] = {
                    'key': key, 'mname': db['mname'], 'datexp': db['datexp'],
                    'blk': db['blk'], 'exp_types': [], 'beh_source': (exp_type, behkey),
                    'stim_map': {}, 'exptype': db.get('exptype', None),
                }
            r = recs[key]
            r['exp_types'].append(exp_type)
            if r['exptype'] is None and db.get('exptype') is not None:
                r['exptype'] = db['exptype']
            # the (exp_type, beh key, db) triples supply the stimulus-id mappings that are
            # merged in load_behaviour(), where the Beh files (and their UniqWalls) are read
            r['_db_list'] = r.get('_db_list', []) + [(exp_type, behkey, db)]

    # "cohort" label: mice that were never given water are naive/unsupervised
    for r in recs.values():
        if r['exptype'] is None:
            r['exptype'] = 'naive'

    # day of training = days since that mouse's first recording
    by_mouse = defaultdict(list)
    for r in recs.values():
        by_mouse[r['mname']].append(r)
    for mname, rs in by_mouse.items():
        d0 = min(datetime.date(*map(int, r['datexp'].split('_'))) for r in rs)
        for r in rs:
            r['day'] = (datetime.date(*map(int, r['datexp'].split('_'))) - d0).days

    order = sorted(recs.values(), key=lambda r: (r['mname'], r['datexp'], r['blk']))
    return order, exp_info


# --------------------------------------------------------------------------------------
# Step 2: behaviour
# --------------------------------------------------------------------------------------
def load_behaviour(recs, verbose=True):
    """Fill each recording dict with the per-frame / per-trial behaviour it needs.

    Every Beh_<exp_type>.npy is read exactly once.  The stimulus map is merged over all
    experiment types; the behaviour arrays themselves are taken from the first experiment
    type containing the recording (they are identical across experiment types -- asserted).
    """
    by_key = {r['key']: r for r in recs}
    exp_types = sorted({et for r in recs for et in r['exp_types']})

    for exp_type in exp_types:
        t0 = time.time()
        B = np.load(os.path.join(BEH_DIR, 'Beh_' + exp_type + '.npy'), allow_pickle=True).item()
        for r in recs:
            for (et, behkey, db) in r['_db_list']:
                if et != exp_type:
                    continue
                beh = B[behkey]
                # merge canonical stimulus ids
                for wall, sid in zip(beh['UniqWalls'], db['stim_id']):
                    if not np.isnan(sid):
                        wall = str(wall)
                        prev = r['stim_map'].get(wall)
                        assert prev is None or prev == int(sid), \
                            'stim_id conflict %s %s: %s vs %s' % (r['key'], wall, prev, sid)
                        r['stim_map'][wall] = int(sid)
                if 'beh' not in r:
                    r['beh'] = _extract_beh(beh)
                else:
                    assert r['beh']['ntrials'] == int(beh['ntrials']), \
                        'behaviour mismatch across experiment types for %s' % r['key']
        del B
        if verbose:
            print('  loaded Beh_%s.npy (%.1fs)' % (exp_type, time.time() - t0), flush=True)
    return recs


def _extract_beh(beh):
    """Keep only the behaviour fields required by the conversion (small arrays)."""
    return {
        'ntrials': int(beh['ntrials']),
        'ft': np.asarray(beh['ft'], dtype=np.float64),
        'ft_trInd': np.asarray(beh['ft_trInd'], dtype=np.float64),
        'ft_move': np.asarray(beh['ft_move'], dtype=np.float64),
        'ft_CorrSpc': np.asarray(beh['ft_CorrSpc'], dtype=bool),
        'ft_Pos': np.asarray(beh['ft_Pos'], dtype=np.float64),
        'ft_RunSpeed': np.asarray(beh['ft_RunSpeed'], dtype=np.float64),
        'Trial_start_time': np.asarray(beh['Trial_start_time'], dtype=np.float64),
        'SoundTime': np.asarray(beh['SoundTime'], dtype=np.float64),
        'SoundFr': np.asarray(beh['SoundFr'], dtype=np.float64),
        'LickFr': np.asarray(beh['LickFr'], dtype=np.float64),
        'WallName': np.array([str(w) for w in beh['WallName']]),
        'isRew': np.asarray(beh['isRew'], dtype=bool),
        'Corridor_Length': float(beh['Corridor_Length']),
        'Texture_Length': float(beh['Texture_Length']),
    }


def frame_mask(ft_move, ft_CorrSpc, ft_trInd, ft_Pos, nfr):
    """Reference recipe: inside the texture corridor AND the VR (mouse) is moving.

    utils.Get_dprime_selective_neuron:
        VRmove = beh['ft_move'][:nfr] > 0
        isCorridor = beh['ft_CorrSpc'][:nfr]
        fr_valid = VRmove & isCorridor
    plus a valid trial index (frames outside the behaviour have ft_trInd == NaN).

    One extra edge case is handled: at a corridor boundary the VR position can already
    have wrapped back to ~0 while ft_trInd (and ft_WallID) still report the previous
    trial, so a single frame is labelled with the old corridor but the new position.
    Such frames straddle two corridors -- both their stimulus label and their position
    label are ambiguous -- so they are dropped (21 frames out of 821,600 in the whole
    dataset).
    """
    tr = ft_trInd[:nfr]
    keep = (ft_move[:nfr] > 0) & ft_CorrSpc[:nfr] & ~np.isnan(tr)
    idx = np.flatnonzero(keep)
    if len(idx) > 1:
        tri = tr[idx]
        pos = ft_Pos[:nfr][idx]
        wrap = np.flatnonzero((np.diff(pos) < 0) & (np.diff(tri) == 0)) + 1
        if len(wrap):
            keep[idx[wrap]] = False
    return keep


def compute_frame_selection(rec, nfr):
    b = rec['beh']
    return frame_mask(b['ft_move'], b['ft_CorrSpc'], b['ft_trInd'], b['ft_Pos'], nfr)


def build_trial_variables(rec, nfr, keep, speed_edges):
    """Build per-trial input/output arrays for the retained frames of one recording."""
    b = rec['beh']
    idx = np.flatnonzero(keep)                      # frame indices, ascending
    tri = b['ft_trInd'][idx].astype(np.int64)       # trial index of each retained frame
    ft = b['ft'][idx]

    # ---- per-trial scalars -----------------------------------------------------------
    ntr = b['ntrials']
    wall = b['WallName']
    stim_id = np.array([rec['stim_map'].get(w, STIM_EXTRA.get(w, -1)) for w in wall],
                       dtype=np.int64)
    assert (stim_id >= 0).all(), 'unmapped stimulus in %s: %s' % (
        rec['key'], sorted(set(wall[stim_id < 0])))

    if b['isRew'].any():
        rew_stim = wall[b['isRew']][0]              # utils.get_cat_id
        rew_avail = (wall == rew_stim).astype(np.float32)
    else:
        rew_avail = np.zeros(ntr, dtype=np.float32)

    # ---- time-varying variables ------------------------------------------------------
    t_since_start = (ft - b['Trial_start_time'][tri]) * SEC_PER_DAY
    t_to_cue = (b['SoundTime'][tri] - ft) * SEC_PER_DAY
    t_since_start = np.clip(t_since_start, 0.0, TIME_CLIP_S)
    t_to_cue = np.clip(t_to_cue, -TIME_CLIP_S, TIME_CLIP_S)

    pos = b['ft_Pos'][idx]
    pos_bin = np.clip((pos / POS_BIN_DM).astype(np.int64), 0, N_POS_BINS - 1)

    speed = b['ft_RunSpeed'][idx]
    speed_bin = np.searchsorted(speed_edges, speed, side='right').astype(np.int64)
    speed_bin = np.clip(speed_bin, 0, N_SPEED_BINS - 1)

    # licking: the reference indexes event frames with `.astype(int)` (floor)
    lick_fr = b['LickFr']
    lick_fr = lick_fr[np.isfinite(lick_fr)]
    lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nfr)].astype(np.int64)
    lick_frame = np.zeros(nfr, dtype=bool)
    lick_frame[lick_fr] = True
    licking = lick_frame[idx].astype(np.int64)

    # ---- split into trials -----------------------------------------------------------
    # trial boundaries: retained frames are sorted, and ft_trInd is non-decreasing on them
    assert np.all(np.diff(tri) >= 0), 'trial indices not monotonic for %s' % rec['key']
    _same = np.diff(tri) == 0
    assert not np.any((np.diff(pos) < 0) & _same), \
        'position decreases within a trial for %s' % rec['key']
    bounds = np.flatnonzero(np.diff(tri)) + 1
    starts = np.concatenate(([0], bounds))
    stops = np.concatenate((bounds, [len(tri)]))
    trial_ids = tri[starts]

    inputs, outputs, slices = [], [], []
    for s, e, t in zip(starts, stops, trial_ids):
        T = e - s
        inp = np.empty((4, T), dtype=np.float32)
        inp[0] = t_to_cue[s:e]
        inp[1] = rec['day']
        inp[2] = t_since_start[s:e]
        inp[3] = rew_avail[t]
        out = np.empty((4, T), dtype=np.int64)
        out[0] = stim_id[t]
        out[1] = licking[s:e]
        out[2] = pos_bin[s:e]
        out[3] = speed_bin[s:e]
        inputs.append(inp)
        outputs.append(out)
        slices.append((s, e))

    diag = {'idx': idx, 'tri': tri, 'pos': pos, 'speed': speed, 'licking': licking,
            't_since_start': t_since_start, 't_to_cue': t_to_cue,
            'pos_bin': pos_bin, 'speed_bin': speed_bin, 'trial_ids': trial_ids,
            'starts': starts, 'stops': stops, 'stim_id': stim_id, 'rew_avail': rew_avail}
    return inputs, outputs, trial_ids, slices, diag


# --------------------------------------------------------------------------------------
# Step 3: neural data (runs in worker processes)
# --------------------------------------------------------------------------------------
def select_neurons(iarea, n_keep, seed):
    """Visual-cortex neurons only, stratified-random subsample over the four areas.

    Returns (global neuron indices, brain_region_idx) sorted by neuron index.
    """
    area_idx = neu_area_ID(iarea)
    rng = np.random.default_rng(seed)

    per_area = [np.flatnonzero(area_idx[ar]) for ar in BRAIN_REGIONS]
    n_vis = sum(len(a) for a in per_area)
    if n_vis <= n_keep:
        chosen = [a for a in per_area]
    else:
        # proportional allocation with largest-remainder rounding
        exact = np.array([len(a) for a in per_area], dtype=np.float64) / n_vis * n_keep
        take = np.floor(exact).astype(int)
        rem = n_keep - take.sum()
        if rem > 0:
            order = np.argsort(-(exact - take))
            take[order[:rem]] += 1
        take = np.minimum(take, [len(a) for a in per_area])
        chosen = [rng.choice(a, size=k, replace=False) if k < len(a) else a
                  for a, k in zip(per_area, take)]

    idx = np.concatenate(chosen) if len(chosen) else np.zeros(0, dtype=np.int64)
    region = np.concatenate([np.full(len(c), i, dtype=np.int64)
                             for i, c in enumerate(chosen)]) if len(chosen) else \
        np.zeros(0, dtype=np.int64)
    order = np.argsort(idx)
    return idx[order].astype(np.int64), region[order].astype(np.int64)


def process_neural(task):
    """Worker: load one recording's spk file and return the per-trial neural matrices."""
    t0 = time.time()
    key = task['key']
    planes = load_spk_planes(task['mname'], task['datexp'], task['blk'])
    nfr = planes[0].shape[1]
    n_total = sum(p.shape[0] for p in planes)
    t_load = time.time() - t0

    iarea = task['iarea']
    assert len(iarea) == n_total, \
        '%s: retinotopy has %d neurons, spk has %d' % (key, len(iarea), n_total)

    keep = task['keep_fn'](nfr)
    keep_idx = np.flatnonzero(keep)

    neu_idx, region = select_neurons(iarea, task['n_keep'], task['seed'])

    # gather the selected rows / retained columns plane by plane
    offs = np.cumsum([0] + [p.shape[0] for p in planes])
    blocks = []
    for pi, p in enumerate(planes):
        lo, hi = offs[pi], offs[pi + 1]
        sel = neu_idx[(neu_idx >= lo) & (neu_idx < hi)] - lo
        if len(sel) == 0:
            continue
        blocks.append(p[sel][:, keep_idx])
    X = np.concatenate(blocks, axis=0).astype(np.float32) if blocks else \
        np.zeros((0, len(keep_idx)), np.float32)
    del planes, blocks

    raw_stats = {'mean': float(X.mean()) if X.size else 0.0,
                 'max': float(X.max()) if X.size else 0.0}

    # z-score each neuron over the retained timepoints of this session.
    # mean/std are accumulated in float64: in float32 the ~10^4-term sums lose enough
    # precision to shift individual z-scores by ~1e-3.
    mu = X.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    sd = X.std(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    n_dead = int((sd[:, 0] == 0).sum())
    sd[sd == 0] = 1.0
    X -= mu
    X /= sd

    return {
        'key': key, 'X': X, 'brain_region_idx': region,
        'n_total_neurons': int(n_total), 'n_visual_neurons': int((iarea != -1).sum() -
                                                                 (iarea == 7).sum()),
        'n_kept_neurons': int(len(neu_idx)), 'n_dead': n_dead, 'nfr': int(nfr),
        'raw_stats': raw_stats, 't_load': t_load, 't_total': time.time() - t0,
    }


class _KeepFn:
    """Picklable closure computing the retained-frame mask once nfr is known."""

    def __init__(self, ft_move, ft_CorrSpc, ft_trInd, ft_Pos):
        self.a, self.b, self.c, self.d = ft_move, ft_CorrSpc, ft_trInd, ft_Pos

    def __call__(self, nfr):
        return frame_mask(self.a, self.b, self.c, self.d, nfr)


# --------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------
def plot_processing(rec, diag, neural_trials, nfr, speed_edges, outdir='/app'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    b = rec['beh']
    fig, axes = plt.subplots(4, 2, figsize=(20, 16))
    key = rec['key']

    # (0,0) raw position trace with retained frames highlighted
    ax = axes[0, 0]
    n_show = min(nfr, 900)
    ax.plot(np.arange(n_show), b['ft_Pos'][:n_show], 'k-', lw=0.7, label='ft_Pos (dm)')
    m = diag['idx'][diag['idx'] < n_show]
    ax.plot(m, b['ft_Pos'][m], 'r.', ms=3, label='retained (corridor & VR moving)')
    ax.axhline(CORRIDOR_DM, color='b', ls='--', lw=0.8, label='4 m = end of texture')
    sf = b['SoundFr'][(b['SoundFr'] < n_show)]
    ax.plot(sf, np.interp(sf, np.arange(nfr), b['ft_Pos'][:nfr]), 'gv', ms=5,
            label='sound cue')
    ax.set_xlabel('imaging frame'); ax.set_ylabel('VR position (dm)')
    ax.set_title('%s: frame selection' % key); ax.legend(fontsize=7)

    # (0,1) running speed + retained
    ax = axes[0, 1]
    ax.plot(np.arange(n_show), b['ft_RunSpeed'][:n_show], 'k-', lw=0.7)
    ax.plot(m, b['ft_RunSpeed'][m], 'r.', ms=3)
    for ed in speed_edges:
        ax.axhline(ed, color='b', ls=':', lw=0.8)
    ax.set_xlabel('imaging frame'); ax.set_ylabel('running speed (cm/s)')
    ax.set_title('running speed, dotted = global quartile edges')

    # (1,0) position -> position bin
    ax = axes[1, 0]
    ax.plot(diag['pos'], diag['pos_bin'], 'k.', ms=1)
    for k in range(1, N_POS_BINS):
        ax.axvline(k * POS_BIN_DM, color='r', ls='--', lw=0.8)
    ax.set_xlabel('ft_Pos (dm)'); ax.set_ylabel('position_bin')
    ax.set_title('position discretisation (4 x 1 m)')

    # (1,1) speed -> speed bin
    ax = axes[1, 1]
    ax.plot(diag['speed'], diag['speed_bin'], 'k.', ms=1)
    for ed in speed_edges:
        ax.axvline(ed, color='r', ls='--', lw=0.8)
    ax.set_xlabel('ft_RunSpeed (cm/s)'); ax.set_ylabel('running_speed_bin')
    frac = [np.mean(diag['speed_bin'] == k) for k in range(N_SPEED_BINS)]
    ax.set_title('speed discretisation, session fractions = %s' %
                 np.round(frac, 3).tolist())

    # (2,0) alignment check: time-to-cue vs position for 5 example trials
    ax = axes[2, 0]
    for t in range(min(5, len(diag['trial_ids']))):
        s, e = diag['starts'][t], diag['stops'][t]
        ax.plot(diag['t_since_start'][s:e], diag['pos'][s:e], '.-', ms=3, lw=0.8,
                label='trial %d' % diag['trial_ids'][t])
        icue = np.argmin(np.abs(diag['t_to_cue'][s:e]))
        ax.plot(diag['t_since_start'][s + icue], diag['pos'][s + icue], 'kv', ms=7)
    ax.set_xlabel('time since trial start (s)'); ax.set_ylabel('position (dm)')
    ax.set_title('alignment: trials start at pos 0, marker = time_to_cue crossing 0')
    ax.legend(fontsize=7)

    # (2,1) sound-cue check: position at the zero crossing vs SoundPos
    ax = axes[2, 1]
    got, want = [], []
    for t in range(len(diag['trial_ids'])):
        s, e = diag['starts'][t], diag['stops'][t]
        tt = diag['t_to_cue'][s:e]
        if tt.min() > 0 or tt.max() < 0:
            continue
        got.append(np.interp(0.0, -tt, diag['pos'][s:e]))
        want.append(np.interp(b['SoundFr'][diag['trial_ids'][t]], np.arange(nfr),
                              b['ft_Pos'][:nfr]))
    ax.plot(want, got, 'k.', ms=2)
    lim = [0, CORRIDOR_DM]
    ax.plot(lim, lim, 'r--', lw=0.8)
    ax.set_xlabel('position at SoundFr (dm)'); ax.set_ylabel('position at time_to_cue=0 (dm)')
    ax.set_title('sound cue temporal alignment (n=%d trials)' % len(got))

    # (3,0) licking raster vs binary series for one trial block
    ax = axes[3, 0]
    ntr_show = min(40, len(diag['trial_ids']))
    for t in range(ntr_show):
        s, e = diag['starts'][t], diag['stops'][t]
        lk = diag['licking'][s:e].astype(bool)
        ax.plot(diag['pos'][s:e][lk], np.full(lk.sum(), t), 'k|', ms=4)
        ax.plot(np.interp(b['SoundFr'][diag['trial_ids'][t]], np.arange(nfr),
                          b['ft_Pos'][:nfr]), t, 'g.', ms=4)
    ax.set_xlabel('position (dm)'); ax.set_ylabel('trial'); ax.set_xlim(0, CORRIDOR_DM)
    ax.set_title('licking output (| = lick frame, green = sound cue)')

    # (3,1) neural raster of one trial (z-scored)
    ax = axes[3, 1]
    t = min(3, len(neural_trials) - 1)
    Z = neural_trials[t]
    nsh = min(200, Z.shape[0])
    im = ax.imshow(Z[:nsh], aspect='auto', vmin=-1, vmax=4, cmap='magma',
                   interpolation='nearest')
    ax.set_xlabel('time bin within trial'); ax.set_ylabel('neuron (first %d)' % nsh)
    ax.set_title('z-scored deconvolved activity, trial %d (%d neurons x %d bins)'
                 % (diag['trial_ids'][t], Z.shape[0], Z.shape[1]))
    fig.colorbar(im, ax=ax)

    fig.suptitle('Processing steps for session %s (%s, day %d)'
                 % (key, rec['exptype'], rec['day']))
    fig.tight_layout()
    fn = os.path.join(outdir, 'processing_%s.png' % key)
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    print('  wrote %s' % fn, flush=True)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outfile', type=str)
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true', help='process all sessions (default)')
    g.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='save diagnostic plots for up to 2 sessions')
    ap.add_argument('--nneurons', type=int, default=NNEURONS_KEEP)
    args = ap.parse_args()

    t_start = time.time()
    print('=== Step 1: building recording table ===', flush=True)
    recs, exp_info = build_recording_table()
    print('  %d unique recordings, %d mice' % (len(recs), len({r['mname'] for r in recs})),
          flush=True)

    print('=== Step 2: loading behaviour (all %d Beh files) ===' %
          len({et for r in recs for et in r['exp_types']}), flush=True)
    t0 = time.time()
    load_behaviour(recs)
    print('  behaviour loaded in %.1fs' % (time.time() - t0), flush=True)

    if args.sample:
        # one task (rewarded, large) session and one naive (small) session
        want = ['TX124_2023_12_24_1', 'TX109_2023_04_18_1']
        recs = [r for r in recs if r['key'] in want] or recs[:2]
        print('  SAMPLE MODE: %s' % [r['key'] for r in recs], flush=True)

    # ---- neural data -------------------------------------------------------------------
    # Run this first: only the spk file knows the true number of imaging frames (it can be
    # one fewer than len(ft), and that offset is not always the same), and every downstream
    # frame mask must use that number.
    print('=== Step 3: neural data (%d workers) ===' % N_WORKERS, flush=True)
    tasks = [{
        'key': r['key'], 'mname': r['mname'], 'datexp': r['datexp'], 'blk': r['blk'],
        'iarea': load_iarea(r['mname'], r['datexp']),
        'keep_fn': _KeepFn(r['beh']['ft_move'], r['beh']['ft_CorrSpc'],
                           r['beh']['ft_trInd'], r['beh']['ft_Pos']),
        'n_keep': args.nneurons, 'seed': NEURON_SEED + zlib.crc32(r['key'].encode()),
    } for r in recs]

    results = {}
    t0 = time.time()
    nbytes = 0
    with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(tasks))) as ex:
        for i, res in enumerate(ex.map(process_neural, tasks)):
            results[res['key']] = res
            nbytes += res['X'].nbytes
            el = time.time() - t0
            print('  [%3d/%3d] %-22s neurons %6d->%4d  dead %2d  frames %6d->%6d'
                  '  load %4.1fs tot %4.1fs | elapsed %6.1fs eta %6.1fs (%.1f GB)'
                  % (i + 1, len(tasks), res['key'], res['n_total_neurons'],
                     res['n_kept_neurons'], res['n_dead'], res['nfr'], res['X'].shape[1],
                     res['t_load'], res['t_total'],
                     el, el / (i + 1) * (len(tasks) - i - 1), nbytes / 1e9), flush=True)
    print('  neural processing took %.1fs' % (time.time() - t0), flush=True)

    # ---- global running-speed quartiles over every retained timepoint ------------------
    print('=== Step 4: global running-speed quartiles ===', flush=True)
    for r in recs:
        r['nfr'] = results[r['key']]['nfr']
        r['keep'] = compute_frame_selection(r, r['nfr'])
        assert r['keep'].sum() == results[r['key']]['X'].shape[1], \
            '%s: frame mask disagrees with worker' % r['key']
    all_speed = np.concatenate([r['beh']['ft_RunSpeed'][:r['nfr']][r['keep']] for r in recs])
    speed_edges = np.percentile(all_speed, [25, 50, 75])
    print('  %d retained timepoints over %d sessions' % (len(all_speed), len(recs)))
    print('  running speed (cm/s): min %.2f  q25 %.2f  q50 %.2f  q75 %.2f  max %.2f'
          % (all_speed.min(), speed_edges[0], speed_edges[1], speed_edges[2],
             all_speed.max()), flush=True)
    del all_speed

    # ---- per-trial input/output construction and trial splitting -----------------------
    print('=== Step 5: per-trial input/output construction ===', flush=True)
    for r in recs:
        inputs, outputs, trial_ids, slices, diag = build_trial_variables(
            r, r['nfr'], r['keep'], speed_edges)
        r['inputs'], r['outputs'], r['trial_ids'] = inputs, outputs, trial_ids
        r['slices'], r['diag'] = slices, diag
        X = results[r['key']]['X']
        results[r['key']]['trials'] = [np.ascontiguousarray(X[:, a:b]) for (a, b) in slices]
        results[r['key']]['X'] = None
    print('  %d sessions, %d trials, %d timepoints'
          % (len(recs), sum(len(r['inputs']) for r in recs),
             sum(i.shape[1] for r in recs for i in r['inputs'])), flush=True)

    # ---- consistency checks and assembly ----------------------------------------------
    print('=== Step 6: assembling and checking ===', flush=True)
    subjects = sorted({r['mname'] for r in recs})
    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': subjects,
        'subject_idx': np.array([subjects.index(r['mname']) for r in recs], dtype=np.int64),
        'brain_regions': list(BRAIN_REGIONS),
        'brain_region_idx': [],
        'input_names': list(INPUT_NAMES),
        'output_names': list(OUTPUT_NAMES),
        'output_values': [
            list(STIM_NAMES),
            ['no_lick', 'lick'],
            ['0-1m', '1-2m', '2-3m', '3-4m'],
            ['speed_q1', 'speed_q2', 'speed_q3', 'speed_q4'],
        ],
        'metadata': {},
    }

    session_info = []
    n_total_tp = 0
    for r in recs:
        res = results[r['key']]
        trials = res['trials']
        assert len(trials) == len(r['inputs']), '%s trial count mismatch' % r['key']
        for k, (nz, ip, op) in enumerate(zip(trials, r['inputs'], r['outputs'])):
            assert nz.shape[1] == ip.shape[1] == op.shape[1], \
                '%s trial %d length mismatch' % (r['key'], k)
        data['neural'].append(trials)
        data['input'].append(r['inputs'])
        data['output'].append(r['outputs'])
        data['brain_region_idx'].append(res['brain_region_idx'].astype(np.int64))
        n_total_tp += sum(t.shape[1] for t in trials)
        session_info.append({
            'session_id': r['key'], 'mouse': r['mname'], 'date': r['datexp'],
            'block': r['blk'], 'cohort': r['exptype'], 'day_of_training': r['day'],
            'experiment_types': sorted(set(r['exp_types'])),
            'n_neurons_recorded': res['n_total_neurons'],
            'n_neurons_visual': res['n_visual_neurons'],
            'n_neurons_kept': res['n_kept_neurons'],
            'n_trials': len(trials),
            'n_timepoints': int(sum(t.shape[1] for t in trials)),
            'n_trials_rewarded_corridor': int(r['diag']['rew_avail'].sum()),
            'frame_rate_hz': float(1.0 / (np.median(np.diff(r['beh']['ft'])) * SEC_PER_DAY)),
            'stimuli': {w: int(s) for w, s in sorted(r['stim_map'].items())},
        })

    fs_all = np.array([s['frame_rate_hz'] for s in session_info])
    bin_ms = float(1000.0 / np.median(fs_all))

    data['metadata'] = {
        'task_description':
            'Head-fixed mice ran through 4-m virtual-reality corridors whose walls were '
            'covered with one of two frozen naturalistic texture categories (denoted leaf '
            'and circle; rock and brick in some mice), separated by 2 m of grey space. A '
            'sound cue occurred at a random position (0.5-3.5 m) in every trial; for task '
            '(rewarded) mice it marked the start of the reward zone in the rewarded '
            'corridor only. Decoder outputs are the visual stimulus category of the '
            'corridor (per trial), whether the mouse licked (per time bin), the mouse\'s '
            'position in the corridor in 4 x 1 m bins, and its running speed in 4 '
            'quartile bins. Decoder inputs are the time to the sound cue, the time since '
            'corridor entry, the day of training and whether reward was available in this '
            'corridor.',
        'time_bin_size': bin_ms,
        'temporal_alignment_event':
            'trial start = entry into the virtual-reality corridor (beh["Trial_start_time"], '
            'the frame at which VR position resets to 0)',
        'off_start': 0.0,
        'off_end': None,
        'trial_end_event':
            'exit from the 4-m texture corridor into the grey space (position = 4 m)',
        'trial_length_note':
            'Trials have variable numbers of time bins (median 22) because only timepoints '
            'inside the 0-4 m corridor during which the virtual reality was moving (i.e. the '
            'mouse was running above the 6 cm/s threshold) are retained, exactly as in the '
            'reference analyses.',
        'neural_signal':
            'Suite2p non-negative deconvolved calcium traces (decay timescale 0.75 s), '
            'z-scored per neuron over the retained timepoints of each session.',
        'neural_subsampling':
            'Neurons assigned to a visual area (retinotopy iarea not in {-1, 7}) were '
            'subsampled to at most %d per session, stratified proportionally over '
            'V1/mHV/lHV/aHV with a fixed seed.' % args.nneurons,
        'frame_rate_hz': float(np.median(fs_all)),
        'frame_rate_hz_range': [float(fs_all.min()), float(fs_all.max())],
        'timepoint_selection':
            '(ft_move > 0) & ft_CorrSpc & ~isnan(ft_trInd) -- inside the texture corridor '
            'while the VR is moving (utils.Get_dprime_selective_neuron fr_valid)',
        'position_bin_edges_m': [0.0, 1.0, 2.0, 3.0, 4.0],
        'running_speed_bin_edges_cm_s': [float(x) for x in speed_edges],
        'time_input_clip_s': TIME_CLIP_S,
        'input_descriptions': {
            'time_to_sound_cue_s': 'seconds until the sound cue (positive before the cue, '
                                   'negative after), clipped to +/-%g s' % TIME_CLIP_S,
            'day_of_training': 'days elapsed since this mouse\'s first recording session',
            'time_since_trial_start_s': 'seconds since corridor entry, clipped to %g s'
                                        % TIME_CLIP_S,
            'reward_available': '1 if this corridor was the rewarded corridor for this '
                                'mouse, 0 otherwise (0 for all unsupervised/naive mice)',
        },
        'output_descriptions': {
            'stimulus': 'canonical stimulus role of the corridor, following the reference '
                        'code (0 circle1, 1 circle2, 2 leaf1, 3 leaf2, 4 leaf3, '
                        '5 leaf1_swap1, 6 leaf1_swap2; 7 circle3 added here). Role 2 is the '
                        'trained/rewarded stimulus; rock/brick mice map onto the same roles.',
            'licking': '1 if at least one lick was detected in this imaging frame',
            'position_bin': 'position in the corridor, 4 equal 1-m bins over 0-4 m',
            'running_speed_bin': 'running speed quartile, bin edges from the pooled '
                                 'distribution over every retained timepoint',
        },
        'source': 'Zhong et al. 2025, "Unsupervised pretraining in biological neural '
                  'networks"; data from https://doi.org/10.25378/janelia.28811129.v1',
        'session_info': session_info,
        'n_sessions': len(recs), 'n_subjects': len(subjects),
        'n_trials': int(sum(len(x) for x in data['neural'])),
        'n_timepoints': int(n_total_tp),
    }

    print('  sessions %d | subjects %d | trials %d | timepoints %d | bin %.2f ms'
          % (len(recs), len(subjects), data['metadata']['n_trials'], n_total_tp, bin_ms),
          flush=True)

    # ---- diagnostics -------------------------------------------------------------------
    if args.show_processing:
        print('=== Step 7: processing plots ===', flush=True)
        for r in recs[:2]:
            plot_processing(r, r['diag'], results[r['key']]['trials'],
                            results[r['key']]['nfr'], speed_edges)

    # ---- summary statistics ------------------------------------------------------------
    print('=== Step 8: summary statistics ===', flush=True)
    allout = np.concatenate([o for s in data['output'] for o in s], axis=1)
    allin = np.concatenate([i for s in data['input'] for i in s], axis=1)
    for d, name in enumerate(INPUT_NAMES):
        print('  input  %-26s min %8.3f  max %8.3f  mean %8.3f'
              % (name, allin[d].min(), allin[d].max(), allin[d].mean()))
    for d, name in enumerate(OUTPUT_NAMES):
        nc = len(data['output_values'][d])
        frac = np.bincount(allout[d], minlength=nc) / allout.shape[1]
        print('  output %-26s %s' % (name, np.round(frac, 4).tolist()))
    # per-trial stimulus distribution (not weighted by trial length)
    stim_tr = np.array([o[0, 0] for s in data['output'] for o in s])
    print('  stimulus per-trial counts: %s'
          % np.bincount(stim_tr, minlength=len(STIM_NAMES)).tolist())
    rew_tr = np.array([i[3, 0] for s in data['input'] for i in s])
    print('  rewarded-corridor trials: %d / %d (%.4f)'
          % (rew_tr.sum(), len(rew_tr), rew_tr.mean()))

    # ---- write --------------------------------------------------------------------------
    print('=== Step 9: writing %s ===' % args.outfile, flush=True)
    t0 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print('  wrote %.2f GB in %.1fs' % (os.path.getsize(args.outfile) / 1e9,
                                        time.time() - t0), flush=True)
    print('TOTAL TIME %.1fs' % (time.time() - t_start), flush=True)


if __name__ == '__main__':
    main()
