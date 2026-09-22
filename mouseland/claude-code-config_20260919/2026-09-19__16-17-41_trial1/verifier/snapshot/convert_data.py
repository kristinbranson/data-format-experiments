#!/usr/bin/env python3
"""
Convert the Zhong et al. 2025 ("Unsupervised pretraining in biological neural networks")
two-photon mesoscope dataset into the decoder-ready pickle format.

Usage
-----
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Options
-------
    --full             process all 89 recordings (default)
    --sample           process only 2 recordings (one naive, one task mouse)
    --show-processing  save diagnostic plots `processing_<session_id>.png` for up to
                       2 sessions, showing every step of the conversion

Processing summary (see CONVERSION_NOTES.md for the full justification)
-----------------------------------------------------------------------
* Trial            = one traversal of a 4 m virtual-reality texture corridor.
                     Aligned to corridor entry (`beh['StartFr']` / `Trial_start_time`).
* Frames kept      = `ft_CorrSpc & (ft_move > 0)` truncated to `spk.shape[1]`,
                     i.e. inside the 0-4 m texture area while the VR is moving
                     (== the mouse is running).  This is exactly the reference's
                     `fr_valid` in `utils.Get_dprime_selective_neuron`.
* Neurons kept     = `iarea not in {-1, 7}` (V1 / mHV / lHV / aHV), the reference's
                     "exclude neurons from outside of visual cortex" rule.
* Neural signal    = Suite2p non-negative-deconvolved traces, per-neuron z-scored over
                     all frames of the session (`stats.zscore(spk, axis=1)`, as in
                     `utils.get_kfold_reward_response`).
"""

import argparse
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np

ROOT = '/app/data'
N_STIM = 7
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3', 'leaf1_swap1', 'leaf1_swap2']
REGION_NAMES = ['V1', 'mHV', 'lHV', 'aHV']
CORRIDOR_DM = 40.0            # texture area length in decimetres (4 m)
POS_BIN_DM = 10.0             # 1 m position bins
N_POS_BINS = 4
N_SPEED_BINS = 4
SEC_PER_DAY = 86400.0

# two small recordings used by --sample: one naive mouse (no reward / no licking) and
# one task mouse (rewarded corridor + licking), so the sample exercises every code path.
SAMPLE_SESSIONS = ['TX124_2023_12_24_1', 'TX109_2023_03_27_1']

# fields of the behaviour dict we actually need (everything else is dropped immediately
# so that the 23 Beh_*.npy files, 5 GB in total, do not stay in memory)
BEH_FRAME_FIELDS = ['ft', 'ft_trInd', 'ft_CorrSpc', 'ft_move', 'ft_Pos', 'ft_RunSpeed']
BEH_TRIAL_FIELDS = ['Trial_start_time', 'SoundTime', 'isRew', 'WallName', 'StartFr', 'GrayFr']


# ----------------------------------------------------------------------------------
# loading helpers (mirroring code/utils.py)
# ----------------------------------------------------------------------------------
def neu_area_idx(iarea):
    """Reference `utils.neu_area_ID`, returned as an integer index into REGION_NAMES.

    Returns (keep, region_idx) where keep is the boolean mask of neurons inside
    V1/mHV/lHV/aHV and region_idx is the region index of the kept neurons.
    """
    reg = np.full(len(iarea), -1, dtype=np.int64)
    reg[iarea == 8] = 0                                              # V1
    reg[np.isin(iarea, [0, 1, 2, 9])] = 1                            # medial HVAs
    reg[np.isin(iarea, [5, 6])] = 2                                  # lateral HVAs
    reg[np.isin(iarea, [3, 4])] = 3                                  # anterior HVAs
    keep = reg >= 0                                                  # == (iarea!=-1)&(iarea!=7)
    return keep, reg[keep]


def load_spk_blocks(mname, datexp, blk):
    """Reference `utils.load_spk`, but without concatenating (saves a full copy).

    After reading, the file's pages are dropped from the page cache: the 89 spk files
    total 404 GB and letting them accumulate there forces the kernel into reclaim for
    every subsequent allocation, which slowed the extraction step down ~5x.
    """
    fn = os.path.join(ROOT, 'spk', '%s_%s_%s_neural_data.npy' % (mname, datexp, blk))
    spks = np.load(fn, allow_pickle=True).item()['spks']
    try:
        fd = os.open(fn, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except OSError:
        pass
    return spks


def load_iarea(mname, datexp):
    fn = os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz' % (mname, datexp))
    with np.load(fn, allow_pickle=True) as d:
        return d['iarea']


def build_session_index():
    """Deduplicate `Imaging_Exp_info.npy` down to the 89 physical recordings.

    Returns an ordered dict session_key -> dict(mname, datexp, blk, exp_types, beh_keys,
    rewtype, stim_map) where stim_map maps a UniqWalls name to its canonical stimulus id
    (merged over every exp_type in which the recording appears).
    """
    exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=1).item()
    sessions = {}
    for exp_type, db in exp_info.items():
        for ndb in db:
            key = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
            beh_key = key + ('_' + ndb['stimtype'] if 'stimtype' in ndb else '')
            rec = sessions.setdefault(key, dict(
                key=key, mname=ndb['mname'], datexp=ndb['datexp'], blk=ndb['blk'],
                exp_types=[], beh_keys=[], rewtype=set(), exptype=set(), stim_id_entries=[]))
            rec['exp_types'].append(exp_type)
            rec['beh_keys'].append(beh_key)
            rec['rewtype'].add(str(ndb.get('rewType', '')))
            rec['exptype'].add(str(ndb.get('exptype', '')))
            rec['stim_id_entries'].append((beh_key, np.asarray(ndb['stim_id'], dtype=float)))
    return sessions


def cohort_of(rec):
    ets = rec['exp_types']
    if any(e.startswith('sup_') for e in ets):
        return 'task'
    if any(e.startswith('unsup') for e in ets):
        return 'unsupervised'
    if any(e.endswith('_grating') for e in ets):
        return 'unsupervised_grating'
    return 'naive'


def load_behaviour(sessions, verbose=True):
    """One pass over the 23 Beh_*.npy files, keeping only the fields we need.

    Also merges `stim_id` over all exp_type entries of each recording (a wall that is
    NaN in one experiment type is labelled in another).
    """
    beh = {}
    stim_map = {k: {} for k in sessions}
    for exp_type in sorted(set(sum([r['exp_types'] for r in sessions.values()], []))):
        t0 = time.time()
        B = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type), allow_pickle=1).item()
        for bk, b in B.items():
            # beh keys are '<mname>_<yyyy>_<mm>_<dd>_<blk>[_<stimtype>]'; the physical
            # recording is identified by the first five tokens (mname has no underscore)
            key = '_'.join(bk.split('_')[:5])
            if key not in sessions:
                continue
            walls = list(b['UniqWalls'])
            sid = np.asarray(b['stim_id'], dtype=float)
            for w, s in zip(walls, sid):
                if not np.isnan(s):
                    prev = stim_map[key].get(str(w))
                    assert prev is None or prev == int(s), \
                        'conflicting stim_id for %s / %s' % (key, w)
                    stim_map[key][str(w)] = int(s)
            if key in beh:
                continue
            rec = {f: np.asarray(b[f]) for f in BEH_FRAME_FIELDS}
            rec.update({f: np.asarray(b[f]) for f in BEH_TRIAL_FIELDS})
            rec['LickFr'] = np.asarray(b['LickFr'], dtype=float)
            rec['ntrials'] = int(b['ntrials'])
            rec['Corridor_Length'] = float(b['Corridor_Length'])
            rec['Texture_Length'] = float(b['Texture_Length'])
            rec['Reward_Mode'] = str(b.get('Reward_Mode', ''))
            beh[key] = rec
        del B
        if verbose:
            print('  loaded Beh_%s.npy (%.1fs)' % (exp_type, time.time() - t0), flush=True)
    for k in sessions:
        sessions[k]['stim_map'] = stim_map[k]
    return beh


# ----------------------------------------------------------------------------------
# per-session trial construction (behaviour only -- no neural data needed)
# ----------------------------------------------------------------------------------
def build_trials(rec, b, nfr, day_of_training):
    """Compute the frame indices and the input/output variables for every kept trial.

    Inputs
        rec  : session index record (needs 'stim_map')
        b    : behaviour dict for this recording
        nfr  : number of frames actually present in the neural recording
        day_of_training : days since this mouse's first imaging session

    Returns dict with
        frame_idx   : (n_valid_total,) int32, sorted frame indices used, all trials
        trial_slice : list of (start, stop) into frame_idx, one per kept trial
        inputs      : (n_kept_trials,) list of (4, T) float32
        speed       : (n_valid_total,) float32, running speed of every kept frame
        pos_bin, lick, stim : per-kept-frame / per-trial output pieces
    """
    ft = b['ft'][:nfr]
    tr = b['ft_trInd'][:nfr]
    corr = b['ft_CorrSpc'][:nfr].astype(bool)
    move = b['ft_move'][:nfr] > 0
    pos = b['ft_Pos'][:nfr]
    speed = b['ft_RunSpeed'][:nfr]

    # reference `fr_valid`: inside the texture area AND the VR is moving (mouse running)
    ntrials = b['ntrials']
    valid = corr & move & np.isfinite(tr)
    tr_int = np.where(valid, np.nan_to_num(tr, nan=-1), -1).astype(np.int64)
    tr_int[(tr_int < 0) | (tr_int >= ntrials)] = -1
    # per-frame lick indicator: a frame is "licking" if >=1 lick falls in it.
    lick_frames = np.floor(b['LickFr']).astype(np.int64)
    lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < nfr)]
    lick = np.zeros(nfr, dtype=bool)
    lick[lick_frames] = True

    stim_of_trial = np.full(ntrials, -1, dtype=np.int64)
    smap = rec['stim_map']
    for t in range(ntrials):
        stim_of_trial[t] = smap.get(str(b['WallName'][t]), -1)

    tstart = np.asarray(b['Trial_start_time'], dtype=float)
    tcue = np.asarray(b['SoundTime'], dtype=float)
    isrew = np.asarray(b['isRew']).astype(np.float32)

    order = np.argsort(tr_int, kind='stable')          # group frames by trial, time-ordered
    counts = np.bincount(tr_int[tr_int >= 0], minlength=ntrials)
    ordered = order[int(np.sum(tr_int < 0)):]          # drop the -1 group (invalid frames)

    frame_idx, trial_slices, inputs, outputs, trial_ids = [], [], [], [], []
    ptr = 0
    for t in range(ntrials):
        n = counts[t]
        if n == 0:
            continue
        fi = ordered[ptr:ptr + n]
        ptr += n
        if stim_of_trial[t] < 0:                       # wall without a canonical stim id
            continue
        fi = np.sort(fi)
        T = len(fi)
        ftt = ft[fi]
        inp = np.empty((4, T), dtype=np.float32)
        inp[0] = (tcue[t] - ftt) * SEC_PER_DAY         # + before the sound cue, - after
        inp[1] = day_of_training
        inp[2] = (ftt - tstart[t]) * SEC_PER_DAY       # time since corridor entry
        inp[3] = isrew[t]

        out = np.empty((4, T), dtype=np.int64)
        out[0] = stim_of_trial[t]
        out[1] = lick[fi]
        out[2] = np.clip(np.floor(pos[fi] / POS_BIN_DM), 0, N_POS_BINS - 1)
        out[3] = 0                                      # speed bin filled in later
        frame_idx.append(fi)
        trial_slices.append(T)
        inputs.append(inp)
        outputs.append(out)
        trial_ids.append(t)

    if len(frame_idx) == 0:
        return None
    frame_idx = np.concatenate(frame_idx).astype(np.int64)
    bounds = np.concatenate([[0], np.cumsum(trial_slices)])
    return dict(frame_idx=frame_idx, bounds=bounds, inputs=inputs, outputs=outputs,
                trial_ids=np.array(trial_ids), speed=speed[frame_idx].astype(np.float32),
                pos=pos[frame_idx].astype(np.float32), ft=ft[frame_idx],
                ntrials_raw=ntrials, nvalid_frames=int(valid.sum()))


# ----------------------------------------------------------------------------------
# neural extraction
# ----------------------------------------------------------------------------------
def extract_neural(blocks, keep_mask, frame_idx, zscore=True, nthreads=16, chunk=4096):
    """Select neurons + frames and (optionally) z-score each neuron over the session.

    blocks    : list of (n_i, nfr) float32 arrays as stored on disk
    keep_mask : (n_neurons,) bool over the concatenation of the blocks
    frame_idx : (T,) int64 frame indices to keep
    zscore    : subtract the mean and divide by the s.d. computed over ALL frames of the
                session (reference: `stats.zscore(spk, axis=1)`), not just the kept ones.

    The work is split into row chunks and run on a thread pool; every numpy call below
    releases the GIL, and the job is memory-bandwidth bound, so this scales well.

    Returns (n_kept_neurons, T) float32.
    """
    T = len(frame_idx)
    out = np.empty((int(keep_mask.sum()), T), dtype=np.float32)
    tasks = []
    off, oo = 0, 0
    for blk in blocks:
        n = blk.shape[0]
        km = keep_mask[off:off + n]
        cs = np.concatenate([[0], np.cumsum(km)]).astype(np.int64)
        for a in range(0, n, chunk):
            bnd = min(a + chunk, n)
            if km[a:bnd].any():
                tasks.append((blk, a, bnd, km[a:bnd], oo + int(cs[a])))
        off += n
        oo += int(km.sum())

    def work(task):
        blk, a, bnd, km, o = task
        sub = blk[a:bnd]
        g = sub[:, frame_idx][km]                 # column gather first: small temporary
        if zscore:
            nf = sub.shape[1]
            s1 = sub.sum(axis=1, dtype=np.float64)
            s2 = np.einsum('ij,ij->i', sub, sub, dtype=np.float64)
            mu = s1 / nf
            sd = np.sqrt(np.maximum(s2 / nf - mu * mu, 0.0))
            sd[sd == 0] = 1.0
            g -= mu[km, None].astype(np.float32)
            g /= sd[km, None].astype(np.float32)
        out[o:o + g.shape[0]] = g

    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        list(ex.map(work, tasks))
    return out


# ----------------------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------------------
def plot_processing(session_key, b, nfr, tinfo, neural, speed_edges, outdir='.'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ft = b['ft'][:nfr]
    corr = b['ft_CorrSpc'][:nfr].astype(bool)
    move = b['ft_move'][:nfr] > 0
    pos = b['ft_Pos'][:nfr]
    valid = corr & move & np.isfinite(b['ft_trInd'][:nfr])

    fig = plt.figure(figsize=(20, 26))
    gs = fig.add_gridspec(8, 4, hspace=0.55, wspace=0.28)

    # ---- 1. raw session trace with the frame mask ---------------------------------
    ax = fig.add_subplot(gs[0, :])
    w = slice(0, 400)
    fr = np.arange(nfr)[w]
    ax.plot(fr, pos[w], 'k-', lw=.8, label='ft_Pos (dm)')
    ax.axhline(CORRIDOR_DM, color='gray', ls=':', label='texture/grey boundary (4 m)')
    ax.fill_between(fr, 0, 60, where=corr[w], color='tab:orange', alpha=.15, step='mid',
                    label='ft_CorrSpc (texture area)')
    ax.fill_between(fr, 0, 60, where=(~move)[w], color='tab:red', alpha=.15, step='mid',
                    label='VR not moving (excluded)')
    ax.plot(fr[valid[w]], pos[w][valid[w]], 'g.', ms=4, label='kept frames')
    st = b['StartFr'][(b['StartFr'] >= 0) & (b['StartFr'] < 400)]
    for s in st:
        ax.axvline(s, color='b', lw=.6, alpha=.7)
    ax.set_xlabel('imaging frame'); ax.set_ylabel('position (dm)')
    ax.set_title('%s  step 1: frame curation (blue = StartFr = corridor entry = t0)' % session_key)
    ax.legend(fontsize=7, ncol=5)

    # ---- 2. example trials: inputs -------------------------------------------------
    ntr = min(4, len(tinfo['inputs']))
    for i in range(ntr):
        a, bnd = tinfo['bounds'][i], tinfo['bounds'][i + 1]
        inp, out = tinfo['inputs'][i], tinfo['outputs'][i]
        tt = inp[2]
        ax = fig.add_subplot(gs[1, i])
        ax.plot(tt, inp[0], 'o-', ms=3, label='time to sound cue (s)')
        ax.axhline(0, color='k', lw=.5)
        ax.set_title('trial %d  inputs' % tinfo['trial_ids'][i], fontsize=9)
        ax.set_xlabel('time since trial start (s)')
        ax.legend(fontsize=7)
        ax2 = fig.add_subplot(gs[2, i])
        ax2.plot(tt, tinfo['pos'][a:bnd], 'k.-', ms=3, label='position (dm)')
        ax2.step(tt, out[2] * 10 + 5, where='mid', color='tab:orange',
                 label='position bin (x10+5)')
        for e in [10, 20, 30]:
            ax2.axhline(e, color='gray', ls=':', lw=.5)
        ax2.set_ylim(0, 45); ax2.set_xlabel('time since trial start (s)')
        ax2.legend(fontsize=7); ax2.set_title('position discretisation', fontsize=9)
        ax3 = fig.add_subplot(gs[3, i])
        ax3.plot(tt, tinfo['speed'][a:bnd], 'k.-', ms=3, label='run speed')
        for e in speed_edges:
            ax3.axhline(e, color='tab:red', ls=':', lw=.7)
        ax3b = ax3.twinx()
        ax3b.step(tt, out[3], where='mid', color='tab:orange', label='speed bin')
        ax3b.set_ylim(-0.2, 3.2)
        ax3.set_xlabel('time since trial start (s)')
        ax3.legend(fontsize=7, loc='upper left'); ax3b.legend(fontsize=7, loc='upper right')
        ax3.set_title('speed discretisation (red = global quartiles)', fontsize=9)
        ax4 = fig.add_subplot(gs[4, i])
        ax4.step(tt, out[1], where='mid', color='tab:blue', label='licking')
        ax4.set_ylim(-0.2, 1.2); ax4.set_xlabel('time since trial start (s)')
        ax4.set_title('lick, stim=%s, reward_avail=%d' % (STIM_NAMES[out[0, 0]], inp[3, 0]),
                      fontsize=9)
        ax4.legend(fontsize=7)

    # ---- 3. neural raster for the same trials -------------------------------------
    for i in range(ntr):
        a, bnd = tinfo['bounds'][i], tinfo['bounds'][i + 1]
        ax = fig.add_subplot(gs[5, i])
        sub = neural[:200, a:bnd]
        ax.imshow(sub, aspect='auto', cmap='gray_r', vmin=0, vmax=3,
                  extent=[tinfo['inputs'][i][2][0], tinfo['inputs'][i][2][-1], 200, 0])
        ax.set_xlabel('time since trial start (s)'); ax.set_ylabel('neuron')
        ax.set_title('neural (z-scored), first 200 neurons', fontsize=9)

    # ---- 4. distributions ----------------------------------------------------------
    allout = np.concatenate(tinfo['outputs'], axis=1)
    allinp = np.concatenate(tinfo['inputs'], axis=1)
    ax = fig.add_subplot(gs[6, 0])
    ax.bar(np.arange(N_STIM), [np.mean(allout[0] == c) for c in range(N_STIM)])
    ax.set_xticks(range(N_STIM)); ax.set_xticklabels(STIM_NAMES, rotation=60, fontsize=6)
    ax.set_title('stimulus distribution', fontsize=9)
    ax = fig.add_subplot(gs[6, 1])
    ax.bar([0, 1], [np.mean(allout[1] == 0), np.mean(allout[1] == 1)])
    ax.set_xticks([0, 1]); ax.set_xticklabels(['no lick', 'lick'])
    ax.set_title('licking distribution', fontsize=9)
    ax = fig.add_subplot(gs[6, 2])
    ax.bar(range(N_POS_BINS), [np.mean(allout[2] == c) for c in range(N_POS_BINS)])
    ax.set_title('position-bin distribution', fontsize=9)
    ax = fig.add_subplot(gs[6, 3])
    ax.bar(range(N_SPEED_BINS), [np.mean(allout[3] == c) for c in range(N_SPEED_BINS)])
    ax.set_title('speed-bin distribution (session)', fontsize=9)

    ax = fig.add_subplot(gs[7, 0])
    ax.hist(allinp[0], bins=60); ax.set_title('time to sound cue (s)', fontsize=9)
    ax = fig.add_subplot(gs[7, 1])
    ax.hist(allinp[2], bins=60); ax.set_title('time since trial start (s)', fontsize=9)
    ax = fig.add_subplot(gs[7, 2])
    # alignment check: mean population activity as a function of position bin
    m = [neural[:, allout[2] == c].mean() for c in range(N_POS_BINS)]
    ax.plot(range(N_POS_BINS), m, 'o-'); ax.set_title('mean activity vs position bin', fontsize=9)
    ax = fig.add_subplot(gs[7, 3])
    ax.hist(tinfo['speed'], bins=80)
    for e in speed_edges:
        ax.axvline(e, color='tab:red', ls=':')
    ax.set_title('running speed + global quartile edges', fontsize=9)

    fn = os.path.join(outdir, 'processing_%s.png' % session_key)
    fig.savefig(fn, dpi=90, bbox_inches='tight')
    plt.close(fig)
    print('  wrote %s' % fn, flush=True)


# ----------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--no-zscore', action='store_true',
                    help='keep raw deconvolved traces instead of per-neuron z-scores')
    args = ap.parse_args()

    t_start = time.time()
    print('=' * 78)
    print('Building session index from Imaging_Exp_info.npy ...', flush=True)
    sessions = build_session_index()
    print('  %d unique physical recordings' % len(sessions))

    print('Loading behaviour (23 files) ...', flush=True)
    t0 = time.time()
    beh = load_behaviour(sessions, verbose=False)
    print('  done in %.1fs' % (time.time() - t0), flush=True)

    keys_all = sorted(sessions.keys())
    # day of training = days since the mouse's first imaging session
    def datenum(datexp):
        y, m, d = [int(x) for x in datexp.split('_')]
        import datetime
        return datetime.date(y, m, d).toordinal()
    first_day = {}
    for k in keys_all:
        mn = sessions[k]['mname']
        first_day[mn] = min(first_day.get(mn, 1 << 30), datenum(sessions[k]['datexp']))

    # ------------------------------------------------------------------ pass 1
    # behaviour-only pass over *all* 89 recordings: builds the trial structure and the
    # pooled running-speed quartile edges.  Always over all sessions so that --sample
    # and --full use identical discretisation.
    print('Pass 1: building trial structure for all %d recordings ...' % len(keys_all), flush=True)
    t0 = time.time()
    day = {k: datenum(sessions[k]['datexp']) - first_day[sessions[k]['mname']] for k in keys_all}
    speeds, ntr1 = [], 0
    for k in keys_all:
        # behaviour arrays are 1-2 frames longer than the neural recording; the exact
        # length is only known once the spk file is opened in pass 2, so pass 1 (which
        # only fixes the running-speed quartiles) uses the full behaviour length.
        ti = build_trials(sessions[k], beh[k], len(beh[k]['ft']), day[k])
        speeds.append(ti['speed'])
        ntr1 += len(ti['inputs'])
    speed_all = np.concatenate(speeds)
    speed_edges = np.percentile(speed_all, [25, 50, 75])
    npts1 = len(speed_all)
    del speeds, speed_all
    print('  %d trials, %d timepoints, speed quartile edges = %s  (%.1fs)'
          % (ntr1, npts1, np.round(speed_edges, 4), time.time() - t0), flush=True)

    keys = SAMPLE_SESSIONS if args.sample else keys_all
    print('Pass 2: %d recordings to convert' % len(keys), flush=True)

    # ------------------------------------------------------------------ pass 2
    data = {'neural': [], 'input': [], 'output': [], 'brain_region_idx': []}
    subjects, subject_idx, session_info = [], [], []
    plots_done = 0
    total_bytes = 0

    # background prefetch of the next session's spk file overlaps disk I/O with compute
    q = Queue(maxsize=1)

    def prefetch():
        for k in keys:
            r = sessions[k]
            q.put((k, load_spk_blocks(r['mname'], r['datexp'], r['blk'])))
        q.put(None)

    th = threading.Thread(target=prefetch, daemon=True)
    th.start()

    for si in range(len(keys)):
        t0 = time.time()
        item = q.get()
        k, blocks = item
        r = sessions[k]
        b = beh[k]
        t_load = time.time() - t0

        nfr_true = blocks[0].shape[1]
        assert all(x.shape[1] == nfr_true for x in blocks), k
        assert nfr_true <= len(b['ft']), (k, nfr_true, len(b['ft']))
        iarea = load_iarea(r['mname'], r['datexp'])
        nneu_all = sum(x.shape[0] for x in blocks)
        assert len(iarea) == nneu_all, (k, len(iarea), nneu_all)
        keep, region = neu_area_idx(iarea)

        # rebuild the trial structure with the exact neural frame count
        ti = build_trials(r, b, nfr_true, day[k])
        ti['outputs_speedbin_edges'] = speed_edges
        sb = np.searchsorted(speed_edges, ti['speed'], side='right').astype(np.int64)
        for i in range(len(ti['inputs'])):
            a, bnd = ti['bounds'][i], ti['bounds'][i + 1]
            ti['outputs'][i][3] = sb[a:bnd]

        t1 = time.time()
        neural = extract_neural(blocks, keep, ti['frame_idx'], zscore=not args.no_zscore)
        del blocks, item
        t_extract = time.time() - t1

        trials = []
        for i in range(len(ti['inputs'])):
            a, bnd = ti['bounds'][i], ti['bounds'][i + 1]
            trials.append(np.ascontiguousarray(neural[:, a:bnd]))
        nbytes = sum(x.nbytes for x in trials)
        total_bytes += nbytes

        if args.show_processing and plots_done < 2:
            plot_processing(k, b, nfr_true, ti, neural, speed_edges)
            plots_done += 1
        del neural

        data['neural'].append(trials)
        data['input'].append(ti['inputs'])
        data['output'].append(ti['outputs'])
        data['brain_region_idx'].append(region.astype(np.int64))
        if r['mname'] not in subjects:
            subjects.append(r['mname'])
        subject_idx.append(subjects.index(r['mname']))
        session_info.append(dict(
            session_key=k, mouse=r['mname'], date=r['datexp'], block=r['blk'],
            cohort=cohort_of(r), exp_types=sorted(set(r['exp_types'])),
            reward_mode=b['Reward_Mode'],
            day_of_training=float(ti['inputs'][0][1, 0]),
            n_neurons_all_rois=int(nneu_all), n_neurons_kept=int(keep.sum()),
            n_frames_neural=int(nfr_true), n_trials_raw=int(ti['ntrials_raw']),
            n_trials_kept=len(trials),
            n_rewarded_trials=int(np.sum([inp[3, 0] > 0 for inp in ti['inputs']])),
            frame_interval_s=float(np.median(np.diff(b['ft'])) * SEC_PER_DAY),
            stimuli=sorted(r['stim_map'].items(), key=lambda x: x[1])))
        print('[%3d/%3d] %-22s nneu %6d/%6d  trials %4d  T %6d  %6.2f GB  '
              '(load %4.1fs extract %4.1fs total %4.1fs)  cum %.1f GB, elapsed %.0fs'
              % (si + 1, len(keys), k, keep.sum(), nneu_all, len(trials),
                 sum(x.shape[1] for x in trials), nbytes / 1e9,
                 t_load, t_extract, time.time() - t0, total_bytes / 1e9,
                 time.time() - t_start), flush=True)

    # ------------------------------------------------------------------ assemble
    data['subjects'] = subjects
    data['subject_idx'] = np.array(subject_idx, dtype=np.int64)
    data['brain_regions'] = REGION_NAMES
    data['input_names'] = ['time_to_sound_cue_s', 'day_of_training',
                           'time_since_trial_start_s', 'reward_available']
    data['output_names'] = ['stimulus', 'licking', 'position_bin', 'running_speed_bin']
    data['output_values'] = [
        STIM_NAMES,
        ['no_lick', 'lick'],
        ['0-1m', '1-2m', '2-3m', '3-4m'],
        ['Q1_slowest', 'Q2', 'Q3', 'Q4_fastest'],
    ]
    dt = float(np.mean([s['frame_interval_s'] for s in session_info]))
    data['metadata'] = {
        'task_description':
            'Head-fixed mice run on an air-floating ball through 4 m linear virtual-reality '
            'corridors whose walls are frozen crops of naturalistic textures (leaf/circle, or '
            'rock/wood in some mice), separated by 2 m of grey space. A sound cue is played at '
            'a random position (uniform 0.5-3.5 m) in every corridor; in task ("supervised") '
            'mice water is available after the cue in one of the two corridors, and mice learn '
            'to lick in anticipation of reward. Unsupervised and naive cohorts see the same '
            'corridors without reward. Decoder outputs: the visual stimulus category of the '
            'corridor (7 canonical stimuli), whether the mouse is licking, which 1 m bin of the '
            'corridor it is in, and which quartile of the running-speed distribution it is in. '
            'Decoder inputs: signed time to the sound cue, day of training, time since corridor '
            'entry, and whether the corridor is the rewarded one.',
        'time_bin_size': dt * 1000.0,
        'temporal_alignment_event':
            'trial start = corridor entry (beh["Trial_start_time"] / beh["StartFr"]); '
            'timepoints are the native two-photon imaging frames',
        'off_start': 0.0,
        'off_end': None,
        'off_end_note':
            'Trials end when the mouse leaves the 4 m texture area. Duration is variable '
            'because non-running frames are excluded and mice pause: median 3.8 s, p99 61 s.',
        'sampling_rate_hz': 1.0 / dt,
        'neural_signal': ('Suite2p non-negative deconvolved calcium traces (decay 0.75 s), '
                          + ('raw' if args.no_zscore else
                             'z-scored per neuron over all frames of the session')),
        'frames_included': ('only frames inside the 0-4 m texture area with the virtual '
                            'reality moving (mouse running faster than 6 cm/s), matching '
                            'utils.Get_dprime_selective_neuron: ft_CorrSpc & (ft_move>0)'),
        'neurons_included': ('neurons assigned to a visual cortical area, iarea not in {-1,7} '
                             '(utils.Get_density_map / utils.neu_area_ID)'),
        'speed_bin_edges_cm_s': [float(x) for x in speed_edges],
        'position_bin_edges_m': [0.0, 1.0, 2.0, 3.0, 4.0],
        'stimulus_id_scheme': ('canonical beh["stim_id"], merged over all experiment types of '
                               'a recording; per-mouse texture names are mapped onto it '
                               '(rock1->circle1, rock2->circle2, wood1->leaf1, wood2->leaf2, '
                               'wood5->leaf3)'),
        'input_descriptions': {
            'time_to_sound_cue_s': 'seconds until the sound cue (positive before, negative after)',
            'day_of_training': 'days since this mouse\'s first imaging session',
            'time_since_trial_start_s': 'seconds since corridor entry (real elapsed time)',
            'reward_available': '1 if this corridor is the rewarded one for this mouse, else 0',
        },
        'source': ('Zhong et al. 2025, Nature 644:741, "Unsupervised pretraining in biological '
                   'neural networks"; data doi:10.25378/janelia.28811129'),
        'session_info': session_info,
    }

    print('Writing %s (%.1f GB of neural data) ...' % (args.outfile, total_bytes / 1e9), flush=True)
    t0 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=5)
    print('  wrote %.1f GB in %.1fs' % (os.path.getsize(args.outfile) / 1e9, time.time() - t0))

    ntr = sum(len(s) for s in data['neural'])
    nT = sum(sum(x.shape[1] for x in s) for s in data['neural'])
    print('-' * 78)
    print('sessions        : %d' % len(data['neural']))
    print('subjects        : %d' % len(subjects))
    print('trials          : %d' % ntr)
    print('timepoints      : %d' % nT)
    print('neurons (total) : %d' % sum(len(x) for x in data['brain_region_idx']))
    print('time bin        : %.3f ms' % data['metadata']['time_bin_size'])
    print('total time      : %.1f s' % (time.time() - t_start))


if __name__ == '__main__':
    main()
