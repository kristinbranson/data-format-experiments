#!/usr/bin/env python3
"""Convert the Zhong et al. 2025 virtual-reality dataset to the decoder format.

Usage: python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Processing follows the reference code in /app/code (utils.py, data_process_script.ipynb):
  neural   = suite2p deconvolved traces, np.concatenate(d['spks'], 0)  (utils.load_spk)
  frames   = ft_CorrSpc & (ft_move > 0), truncated to spk.shape[1]     (utils.Get_dprime_selective_neuron)
  areas    = iarea -> V1 / mHV / lHV / aHV                             (utils.neu_area_ID)
  trials   = corridor traversals, aligned to corridor entry (StartFr)
"""
import argparse
import collections
import datetime
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

DATA_ROOT = '/app/data'
SPK_DIR = os.path.join(DATA_ROOT, 'spk')
BEH_DIR = os.path.join(DATA_ROOT, 'beh')
RET_DIR = os.path.join(DATA_ROOT, 'retinotopy')

# area definitions, identical to utils.neu_area_ID
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV']
AREA_CODES = {'V1': [8], 'mHV': [0, 1, 2, 9], 'lHV': [5, 6], 'aHV': [3, 4]}

# canonical stimulus categories (beh['stim_id'] convention)
STIM_VALUES = ['nonrew_crop1', 'nonrew_crop2', 'rew_crop1', 'rew_crop2',
               'rew_crop3', 'rew_crop1_swap1', 'rew_crop1_swap2', 'nonrew_crop3']
UNRESOLVED_STIM_ID = 7

MAX_NEURONS = 2000
RNG_SEED = 0
N_POS_BINS = 4
N_SPEED_BINS = 4
POS_BIN_DM = 10.0   # 1 m = 10 dm; behaviour positions are in decimetres

BEH_FIELDS = ['ft', 'ft_trInd', 'ft_Pos', 'ft_CorrSpc', 'ft_move', 'ft_RunSpeed',
              'StartFr', 'GrayFr', 'EndFr', 'SoundFr', 'SoundPos', 'RewardFr', 'isRew',
              'LickFr', 'LickTrind', 'WallName', 'UniqWalls', 'stim_id', 'ntrials',
              'Corridor_Length', 'Texture_Length', 'Reward_Mode']


def build_recording_table():
    """Unique recordings (sessions) from Imaging_Exp_info.npy.

    A recording (mname_datexp_blk) can appear under several exp_types (and
    stimtypes); its behaviour arrays are identical in that case (verified), only
    UniqWalls/stim_id differ, so wall->stim_id maps are merged over appearances.
    """
    exp_info = np.load(os.path.join(BEH_DIR, 'Imaging_Exp_info.npy'), allow_pickle=True).item()
    recs = collections.OrderedDict()
    for exp_type, db in exp_info.items():
        for ndb in db:
            key = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
            behkey = key + ('_' + ndb['stimtype'] if 'stimtype' in ndb else '')
            rec = recs.setdefault(key, {'key': key, 'mname': ndb['mname'],
                                        'datexp': ndb['datexp'], 'blk': ndb['blk'],
                                        'exp_types': [], 'beh_keys': [],
                                        'exptypes': set(), 'rew_types': set()})
            rec['exp_types'].append(exp_type)
            rec['beh_keys'].append(behkey)
            rec['exptypes'].add(str(ndb.get('exptype')))
            rec['rew_types'].add(str(ndb.get('rewType')))
    by_mouse = collections.defaultdict(list)
    for key, rec in recs.items():
        y, m, d = [int(v) for v in rec['datexp'].split('_')]
        rec['date'] = datetime.date(y, m, d)
        by_mouse[rec['mname']].append(rec)
    for mname, rs in by_mouse.items():
        d0 = min(r['date'] for r in rs)
        for r in rs:
            r['day_of_training'] = float((r['date'] - d0).days)
    return recs


def _load_beh_file(args):
    exp_type, wanted = args
    B = np.load(os.path.join(BEH_DIR, 'Beh_%s.npy' % exp_type), allow_pickle=True).item()
    out = []
    for behkey, reckey in wanted:
        beh = B[behkey]
        out.append((reckey, {f: beh[f] for f in BEH_FIELDS if f in beh}))
    return out


def load_behaviour(recs, nproc=12):
    """Load one behaviour record per recording, merging the wall->stim_id maps."""
    jobs = collections.defaultdict(list)
    for key, rec in recs.items():
        for exp_type, behkey in zip(rec['exp_types'], rec['beh_keys']):
            jobs[exp_type].append((behkey, key))
    with Pool(min(nproc, len(jobs))) as pool:
        results = pool.map(_load_beh_file, list(jobs.items()))
    beh_by_rec = {}
    for res in results:
        for reckey, beh in res:
            rec = recs[reckey]
            w2id = rec.setdefault('wall2id', {})
            for w, s in zip(beh['UniqWalls'], np.atleast_1d(beh['stim_id']).astype(float)):
                if not np.isnan(s):
                    w2id[str(w)] = int(s)
            if reckey not in beh_by_rec:
                beh_by_rec[reckey] = beh
    for key, rec in recs.items():
        beh = beh_by_rec[key]
        # task session = water rewards were actually delivered (isRew == ~isnan(RewardFr))
        rec['is_task'] = bool(np.any(~np.isnan(beh['RewardFr'])))
        rec['ntrials'] = int(beh['ntrials'])
    return beh_by_rec


def trial_frames(beh, nfr):
    """Frames kept for the decoder, grouped by trial.

    Kept frames = inside the 0-4 m texture corridor (ft_CorrSpc) AND the mouse is
    running so that the VR moves (ft_move > 0); behaviour is truncated to the number
    of imaging frames -- exactly utils.Get_dprime_selective_neuron.
    """
    tr = beh['ft_trInd'][:nfr]
    valid = (beh['ft_CorrSpc'][:nfr].astype(bool) & (beh['ft_move'][:nfr] > 0)
             & ~np.isnan(tr))
    frames = np.flatnonzero(valid)
    labels = tr[frames].astype(int)
    order = np.lexsort((frames, labels))
    frames, labels = frames[order], labels[order]
    bounds = np.flatnonzero(np.diff(labels)) + 1
    groups = np.split(np.arange(len(frames)), bounds)
    trial_ids = [int(labels[g[0]]) for g in groups]
    return frames, labels, groups, trial_ids


def frame_times_s(beh, nfr):
    """Imaging frame times in seconds, relative to the first frame of the session."""
    ft = beh['ft'][:nfr]
    return (ft - ft[0]) * 86400.0


def select_neurons(iarea, max_neurons, seed=RNG_SEED):
    """Neurons assigned to a named visual area, with a random subsample."""
    region_idx = np.full(len(iarea), -1, dtype=np.int64)
    for r, name in enumerate(BRAIN_REGIONS):
        region_idx[np.isin(iarea, AREA_CODES[name])] = r
    in_area = np.flatnonzero(region_idx >= 0)
    if len(in_area) > max_neurons:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(in_area, size=max_neurons, replace=False))
    else:
        sel = in_area
    return sel, region_idx[sel], len(iarea), len(in_area)


def load_spk_selected(key, sel, frames):
    """Deconvolved traces of the selected neurons at the selected frames.

    Equivalent to np.concatenate(d['spks'], 0)[sel][:, frames] (utils.load_spk) but
    without materialising the full (n_neurons x n_frames) matrix.
    """
    d = np.load(os.path.join(SPK_DIR, '%s_neural_data.npy' % key), allow_pickle=True).item()
    planes = d['spks']
    nfr = min(p.shape[1] for p in planes)
    nneu = sum(p.shape[0] for p in planes)
    if frames is None:
        return None, nneu, nfr
    out = []
    off = 0
    for p in planes:
        n = p.shape[0]
        rows = sel[(sel >= off) & (sel < off + n)] - off
        if len(rows):
            out.append(np.asarray(p[np.ix_(rows, frames)], dtype=np.float32))
        off += n
    del d, planes
    return np.concatenate(out, 0), nneu, nfr


def stim_categories(beh, wall2id):
    """Canonical stimulus category of every trial."""
    wn = np.array([str(w) for w in beh['WallName']])
    ids = np.array([wall2id.get(w, UNRESOLVED_STIM_ID) for w in wn], dtype=np.int64)
    return ids, wn


def session_speed_values(beh, nfr):
    """Running speed of all kept frames (input to the global speed quartiles)."""
    frames, _, _, _ = trial_frames(beh, nfr)
    return beh['ft_RunSpeed'][:nfr][frames]


def convert_session(job):
    """Convert one recording into per-trial neural / input / output arrays."""
    rec, beh, speed_thr, show_processing, max_neurons = job
    key = rec['key']
    t0 = time.time()

    ret = np.load(os.path.join(RET_DIR, '%s_%s_trans.npz' % (rec['mname'], rec['datexp'])),
                  allow_pickle=True)
    iarea = ret['iarea']
    sel, region_idx, n_neurons_file, n_in_area = select_neurons(iarea, max_neurons)

    # number of imaging frames comes from the neural file (behaviour is 1-2 longer)
    _, n_neu_spk, nfr = load_spk_selected(key, None, None)
    assert n_neu_spk == n_neurons_file, 'neuron count mismatch for %s' % key
    frames, labels, groups, trial_ids = trial_frames(beh, nfr)
    t_frames = frame_times_s(beh, nfr)
    t_load0 = time.time()
    spk, _, _ = load_spk_selected(key, sel, frames)
    t_load = time.time() - t_load0

    # trial-wise behaviour
    fr_axis = np.arange(nfr)
    t_start = np.interp(beh['StartFr'], fr_axis, t_frames)   # corridor entry
    t_cue = np.interp(beh['SoundFr'], fr_axis, t_frames)     # sound cue
    stim_ids, wall_names = stim_categories(beh, rec['wall2id'])
    reward_available = (stim_ids == 2) & rec['is_task']
    day = rec['day_of_training']

    # frame-wise behaviour
    t_sel = t_frames[frames]
    pos = beh['ft_Pos'][:nfr][frames]
    speed = beh['ft_RunSpeed'][:nfr][frames]
    pos_bin = np.clip((pos / POS_BIN_DM).astype(np.int64), 0, N_POS_BINS - 1)
    speed_bin = np.digitize(speed, speed_thr).astype(np.int64)
    lick_any = np.zeros(nfr, dtype=bool)
    if len(beh['LickFr']):
        lf = np.round(beh['LickFr']).astype(int)
        lf = lf[(lf >= 0) & (lf < nfr)]
        lick_any[lf] = True
    licking = lick_any[frames].astype(np.int64)

    time_to_cue = t_cue[labels] - t_sel            # > 0 before the cue, < 0 after
    time_since_start = t_sel - t_start[labels]     # >= 0

    neural_out, input_out, output_out, kept_trials = [], [], [], []
    for g, tid in zip(groups, trial_ids):
        T = len(g)
        neural_out.append(np.ascontiguousarray(spk[:, g]))
        inp = np.empty((4, T), dtype=np.float32)
        inp[0] = time_to_cue[g]
        inp[1] = day
        inp[2] = time_since_start[g]
        inp[3] = float(reward_available[tid])
        input_out.append(inp)
        out = np.empty((4, T), dtype=np.int16)
        out[0] = stim_ids[tid]
        out[1] = licking[g]
        out[2] = pos_bin[g]
        out[3] = speed_bin[g]
        output_out.append(out)
        kept_trials.append(tid)

    for arr, nm in ((neural_out, 'neural'), (input_out, 'input'), (output_out, 'output')):
        for a in arr:
            assert np.all(np.isfinite(a)), '%s of %s has non-finite values' % (nm, key)

    res = {
        'key': key, 'mname': rec['mname'], 'datexp': rec['datexp'], 'blk': rec['blk'],
        'neural': neural_out, 'input': input_out, 'output': output_out,
        'brain_region_idx': region_idx.astype(np.int64), 'trial_ids': kept_trials,
        'n_neurons_file': int(n_neurons_file), 'n_neurons_in_area': int(n_in_area),
        'n_neurons_kept': int(len(sel)), 'n_frames': int(nfr),
        'n_frames_kept': int(len(frames)), 'ntrials_beh': int(beh['ntrials']),
        'is_task': bool(rec['is_task']), 'exptypes': sorted(rec['exptypes']),
        'day_of_training': day, 'dt_s': float(np.median(np.diff(t_frames))),
        'time_s': time.time() - t0, 'load_s': t_load,
        'n_reward_trials': int(np.sum(~np.isnan(beh['RewardFr']))),
        'n_rewcorridor_trials': int(reward_available.sum()),
        'n_licks': int(len(beh['LickFr'])),
    }
    if show_processing:
        plot_processing(rec, beh, nfr, frames, labels, groups, trial_ids, t_frames,
                        stim_ids, spk, pos, speed, pos_bin, speed_bin, licking,
                        time_to_cue, time_since_start, speed_thr)
    del spk
    return res


def plot_processing(rec, beh, nfr, frames, labels, groups, trial_ids, t_frames,
                    stim_ids, spk, pos, speed, pos_bin, speed_bin, licking,
                    time_to_cue, time_since_start, speed_thr):
    """Visualise every processing step for one session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    key = rec['key']
    fig, axes = plt.subplots(6, 1, figsize=(18, 24))

    # panel 1: raw behaviour with the frame mask and the event markers
    ax = axes[0]
    f0 = max(int(beh['StartFr'][3]) - 5, 0)
    f1 = min(int(beh['StartFr'][13]) + 5, nfr)
    fr = np.arange(f0, f1)
    ax.plot(fr, beh['ft_Pos'][f0:f1], 'k-', lw=1, label='position (dm)')
    kept = np.isin(fr, frames)
    ax.plot(fr[kept], beh['ft_Pos'][f0:f1][kept], 'g.', ms=8,
            label='kept frames (in 0-4 m corridor AND running)')
    ax.axhline(40, color='gray', ls=':', label='corridor / grey-space boundary (4 m)')
    for t in range(3, 13):
        ax.axvline(beh['StartFr'][t], color='b', lw=1)
        ax.axvline(beh['SoundFr'][t], color='r', lw=1, ls='--')
    if len(beh['LickFr']):
        lf = beh['LickFr'][(beh['LickFr'] >= f0) & (beh['LickFr'] < f1)]
        if len(lf):
            ax.plot(lf, np.full(len(lf), 55.0), 'm|', ms=10, label='licks (LickFr)')
    ax.set_xlabel('imaging frame')
    ax.set_ylabel('position (dm)')
    ax.set_title('%s  step 1: raw behaviour. blue = trial start (StartFr), red dashed = sound cue (SoundFr)' % key)
    ax.legend(loc='upper right', fontsize=8)

    # panel 2: speed and its quartile discretisation
    ax = axes[1]
    ax.plot(fr, beh['ft_RunSpeed'][f0:f1], 'k-', lw=1, label='run speed (cm/s)')
    fsel = fr[kept]
    pos_in_sel = np.searchsorted(frames, fsel)
    sc = ax.scatter(fsel, beh['ft_RunSpeed'][f0:f1][kept], c=speed_bin[pos_in_sel],
                    cmap='viridis', vmin=0, vmax=3, s=30,
                    label='kept frames, coloured by speed bin')
    for thr in speed_thr:
        ax.axhline(thr, color='r', ls=':')
    plt.colorbar(sc, ax=ax, label='speed bin')
    ax.set_xlabel('imaging frame')
    ax.set_ylabel('cm/s')
    ax.set_title('step 2: output speed_bin. red dotted = global quartile thresholds %s cm/s'
                 % np.round(speed_thr, 2))
    ax.legend(loc='upper right', fontsize=8)

    # panel 3: position and its 1 m bins, per trial, vs time since trial start
    ax = axes[2]
    for g, tid in zip(groups[3:8], trial_ids[3:8]):
        ax.plot(time_since_start[g], pos[g], '-', lw=1,
                label='trial %d (stim %d)' % (tid, stim_ids[tid]))
        ax.scatter(time_since_start[g], pos[g], c=pos_bin[g], cmap='coolwarm',
                   vmin=0, vmax=3, s=60, zorder=3)
    for b in range(1, N_POS_BINS):
        ax.axhline(b * POS_BIN_DM, color='k', ls=':')
    ax.set_xlabel('time since trial start (s)  [decoder input 2]')
    ax.set_ylabel('position (dm)')
    ax.set_title('step 3: output position_bin (colour) for 5 example trials; dotted = 1 m bin edges')
    ax.legend(fontsize=8)

    # panel 4: time-to-cue input, checking the alignment of the cue
    ax = axes[3]
    for g, tid in zip(groups[3:8], trial_ids[3:8]):
        ax.plot(time_since_start[g], time_to_cue[g], '-o', ms=4, label='trial %d' % tid)
        ax.plot(time_since_start[g], (pos[g] - beh['SoundPos'][tid]) / 6.0, ':',
                lw=1, color='gray')
    ax.axhline(0, color='r', ls='--', label='sound cue (time_to_sound_cue = 0)')
    ax.set_xlabel('time since trial start (s)')
    ax.set_ylabel('time to sound cue (s)')
    ax.set_title('step 4: decoder input time_to_sound_cue; grey dotted = (position - SoundPos)/VR speed, '
                 'an independent estimate of the same quantity')
    ax.legend(fontsize=8)

    # panel 5: neural activity of the longest trial, with licking marked
    ax = axes[4]
    gi = int(np.argmax([len(g) for g in groups]))
    g = groups[gi]
    tid = trial_ids[gi]
    sub = spk[:60, g]
    ax.imshow(sub, aspect='auto', cmap='gray_r', interpolation='nearest',
              extent=[time_since_start[g][0], time_since_start[g][-1], 60, 0],
              vmax=np.percentile(sub, 99) + 1e-6)
    ax.set_xlabel('time since trial start (s)')
    ax.set_ylabel('neuron')
    ax.set_title('step 5: trial %d deconvolved activity (60 of the stored neurons), kept frames only; '
                 'magenta = licking frames' % tid)
    lk = np.flatnonzero(licking[g])
    if len(lk):
        ax.plot(time_since_start[g][lk], np.full(len(lk), 2.0), 'm|', ms=14)

    # panel 6: output distributions of this session
    ax = axes[5]
    w = 0.2
    ax.bar(np.arange(N_POS_BINS) - w, np.bincount(pos_bin, minlength=N_POS_BINS) / len(pos_bin),
           width=w, label='position bin (frames)')
    ax.bar(np.arange(N_SPEED_BINS), np.bincount(speed_bin, minlength=N_SPEED_BINS) / len(speed_bin),
           width=w, label='speed bin (frames)')
    ax.bar(np.arange(len(STIM_VALUES)) + w,
           np.bincount(stim_ids[np.array(trial_ids)], minlength=len(STIM_VALUES)) / len(trial_ids),
           width=w, label='stimulus category (trials)')
    ax.bar([len(STIM_VALUES) + 1], [licking.mean()], width=w, label='fraction of licking frames')
    ax.set_xlabel('bin')
    ax.set_ylabel('fraction')
    ax.set_title('step 6: output distributions for this session')
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig('processing_%s.png' % key, dpi=100)
    plt.close(fig)
    print('  wrote processing_%s.png' % key, flush=True)


def main():
    ap = argparse.ArgumentParser(description='convert Zhong et al. 2025 data for the decoder')
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='save processing plots for up to 2 sessions')
    ap.add_argument('--nproc', type=int, default=12)
    ap.add_argument('--max-neurons', type=int, default=MAX_NEURONS)
    args = ap.parse_args()

    t_all = time.time()
    print('building recording table ...', flush=True)
    recs = build_recording_table()
    print('  %d unique recordings, %d mice'
          % (len(recs), len({r['mname'] for r in recs.values()})), flush=True)

    t0 = time.time()
    beh_by_rec = load_behaviour(recs)
    print('loaded behaviour for %d recordings in %.1f s' % (len(beh_by_rec), time.time() - t0), flush=True)

    # global speed quartiles over the kept frames of ALL sessions
    t0 = time.time()
    speeds = []
    for key, rec in recs.items():
        beh = beh_by_rec[key]
        speeds.append(session_speed_values(beh, len(beh['ft']) - 2))
    speeds = np.concatenate(speeds)
    speed_thr = np.percentile(speeds, [25, 50, 75])
    print('global speed quartile thresholds (cm/s): %s from %d kept frames (%.1f s)'
          % (np.round(speed_thr, 4), len(speeds), time.time() - t0), flush=True)

    keys = list(recs.keys())
    if args.sample:
        keys = [k for k in keys if recs[k]['is_task']][:2]
        print('SAMPLE mode: %s' % keys, flush=True)
    show_keys = set(keys[:2]) if args.show_processing else set()
    jobs = [(recs[k], beh_by_rec[k], speed_thr, k in show_keys, args.max_neurons) for k in keys]

    print('converting %d sessions with %d workers ...' % (len(jobs), args.nproc), flush=True)
    t0 = time.time()
    results = []
    if args.nproc > 1 and len(jobs) > 1:
        with Pool(min(args.nproc, len(jobs))) as pool:
            for i, res in enumerate(pool.imap_unordered(convert_session, jobs)):
                results.append(res)
                el = time.time() - t0
                print('[%2d/%2d] %s: %d neurons stored (%d in area / %d recorded), %d trials, '
                      '%d kept frames, %.1f s (spk load %.1f s) | elapsed %.1f s, est total %.1f s'
                      % (i + 1, len(jobs), res['key'], res['n_neurons_kept'],
                         res['n_neurons_in_area'], res['n_neurons_file'], len(res['neural']),
                         res['n_frames_kept'], res['time_s'], res['load_s'], el,
                         el / (i + 1) * len(jobs)), flush=True)
    else:
        for i, job in enumerate(jobs):
            res = convert_session(job)
            results.append(res)
            print('[%2d/%2d] %s: %.1f s (spk load %.1f s)'
                  % (i + 1, len(jobs), res['key'], res['time_s'], res['load_s']), flush=True)
    print('conversion of %d sessions took %.1f s' % (len(jobs), time.time() - t0), flush=True)

    order = {k: i for i, k in enumerate(keys)}
    results.sort(key=lambda r: order[r['key']])

    subjects = sorted({r['mname'] for r in results})
    data = {
        'neural': [r['neural'] for r in results],
        'input': [r['input'] for r in results],
        'output': [r['output'] for r in results],
        'subjects': subjects,
        'subject_idx': np.array([subjects.index(r['mname']) for r in results], dtype=np.int64),
        'brain_regions': BRAIN_REGIONS,
        'brain_region_idx': [r['brain_region_idx'] for r in results],
        'input_names': ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start',
                        'reward_availability'],
        'output_names': ['stimulus_category', 'licking', 'position_bin', 'speed_bin'],
        'output_values': [STIM_VALUES,
                          ['no_lick', 'lick'],
                          ['0-1m', '1-2m', '2-3m', '3-4m'],
                          ['speed_q1', 'speed_q2', 'speed_q3', 'speed_q4']],
    }
    dt = float(np.median([r['dt_s'] for r in results]))
    ntrials = [len(r['neural']) for r in results]
    data['metadata'] = {
        'task_description':
            'Head-fixed mice run on a ball through 4 m linear virtual-reality corridors whose walls '
            'show frozen crops of naturalistic textures (two categories per mouse, e.g. leaf vs '
            'circle); corridors alternate pseudo-randomly with 2 m of grey space. A sound cue occurs '
            'at a random position (0.5-3.5 m) in every trial and, for task mice, water becomes '
            'available after the cue in the rewarded corridor. Decoder inputs: time to the sound cue, '
            'day of training, time since trial start (corridor entry), and whether the trial is in '
            'the rewarded corridor. Decoder outputs, predicted from two-photon deconvolved activity '
            'of visual-cortex neurons: visual stimulus category, licking (binary per time bin), '
            'position in the corridor in 4 x 1 m bins, and running speed in 4 quartile bins.',
        'time_bin_size': dt * 1000.0,
        'temporal_alignment_event':
            'trial start = entry into the virtual-reality corridor (beh["StartFr"])',
        'off_start': 0.0,
        'off_end': None,
        'off_end_note':
            'trials have variable length: a trial ends when the mouse leaves the 4 m texture '
            'corridor, and only frames inside the corridor while the mouse was running are kept, so '
            'the number of kept bins varies (median 21 bins, about 6.6 s)',
        'neural_signal':
            'suite2p non-negative deconvolved fluorescence (0.75 s decay timescale), one value per '
            'imaging frame; no dF/F, as in the reference analyses',
        'frame_rate_hz': 1.0 / dt,
        'frame_selection':
            'ft_CorrSpc (inside the 0-4 m texture corridor) & ft_move>0 (virtual reality moving, i.e. '
            'mouse running above the 6 cm/s threshold), behaviour truncated to spk.shape[1]; '
            'identical to fr_valid in utils.Get_dprime_selective_neuron',
        'neuron_selection':
            'neurons assigned to a named visual area by the retinotopic atlas (V1: iarea 8; mHV: '
            '0,1,2,9; lHV: 5,6; aHV: 3,4, as in utils.neu_area_ID); random subsample of at most %d '
            'neurons per session (seed %d) because storing all in-area neurons would need 152 GB'
            % (args.max_neurons, RNG_SEED),
        'speed_bin_edges_cm_per_s': [float(x) for x in speed_thr],
        'position_bin_edges_m': [0.0, 1.0, 2.0, 3.0, 4.0],
        'stimulus_category_note':
            'canonical beh["stim_id"] code, which is task-relative: crops 1/2/3 of the non-rewarded '
            'texture are 0/1/7 and crops 1/2/3 of the rewarded texture are 2/3/4, while 5/6 are the '
            'two spatially swapped versions of rewarded crop 1. The physical textures differ between '
            'mice (circle/leaf or rock/wood), the code does not.',
        'cohorts':
            'task (sup) mice received water rewards; unsupervised, naive and grating-exposed mice ran '
            'the same corridors without rewards and were not water restricted, so their lick and '
            'reward variables are 0',
        'n_sessions': len(results),
        'n_subjects': len(subjects),
        'n_trials_total': int(np.sum(ntrials)),
        'source': 'Zhong et al. 2025, "Unsupervised pretraining in biological neural networks"; '
                  'raw data in /app/data (beh, spk, retinotopy)',
        'session_info': [
            {'session': i, 'key': r['key'], 'mouse': r['mname'], 'date': r['datexp'],
             'block': r['blk'], 'exptypes': r['exptypes'], 'is_task_session': r['is_task'],
             'day_of_training': r['day_of_training'], 'n_trials': len(r['neural']),
             'n_neurons_stored': r['n_neurons_kept'], 'n_neurons_in_area': r['n_neurons_in_area'],
             'n_neurons_recorded': r['n_neurons_file'], 'n_frames_recorded': r['n_frames'],
             'n_frames_kept': r['n_frames_kept'], 'n_trials_behaviour': r['ntrials_beh'],
             'n_reward_trials': r['n_reward_trials'],
             'n_rewarded_corridor_trials': r['n_rewcorridor_trials'], 'n_licks': r['n_licks']}
            for i, r in enumerate(results)],
    }

    print('\n==== summary ====', flush=True)
    print('sessions %d, subjects %d, trials %d (mean %.1f per session)'
          % (len(results), len(subjects), np.sum(ntrials), np.mean(ntrials)))
    print('trials per session: min %d max %d' % (min(ntrials), max(ntrials)))
    print('neurons recorded/session: min %d max %d; stored/session: min %d max %d'
          % (min(r['n_neurons_file'] for r in results), max(r['n_neurons_file'] for r in results),
             min(r['n_neurons_kept'] for r in results), max(r['n_neurons_kept'] for r in results)))
    print('all trials kept? %s (behaviour trials %d, converted trials %d)'
          % (all(len(r['neural']) == r['ntrials_beh'] for r in results),
             sum(r['ntrials_beh'] for r in results), int(np.sum(ntrials))))
    print('time bin %.2f ms (%.4f Hz)' % (dt * 1000, 1 / dt))
    tl = np.concatenate([[a.shape[1] for a in r['neural']] for r in results])
    print('bins per trial: median %d min %d max %d mean %.1f'
          % (np.median(tl), tl.min(), tl.max(), tl.mean()))
    allout = np.concatenate([np.concatenate(r['output'], 1) for r in results], 1)
    for d_i, name in enumerate(data['output_names']):
        cnt = np.bincount(allout[d_i], minlength=len(data['output_values'][d_i]))
        print('output %d %-18s time-bin fractions %s' % (d_i, name, np.round(cnt / cnt.sum(), 4)))
    allin = np.concatenate([np.concatenate(r['input'], 1) for r in results], 1)
    for d_i, name in enumerate(data['input_names']):
        print('input  %d %-22s range [%.3f, %.3f] mean %.3f'
              % (d_i, name, allin[d_i].min(), allin[d_i].max(), allin[d_i].mean()))
    trial_stim = np.concatenate([[o[0, 0] for o in r['output']] for r in results])
    print('per-trial stimulus category counts %s' % np.bincount(trial_stim, minlength=len(STIM_VALUES)))
    rew = np.concatenate([[i[3, 0] for i in r['input']] for r in results])
    print('fraction of trials in the rewarded corridor: %.4f (%d trials)' % (rew.mean(), int(rew.sum())))
    print('task sessions %d / %d' % (sum(r['is_task'] for r in results), len(results)))

    t0 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print('wrote %s (%.2f GB) in %.1f s; total runtime %.1f s'
          % (args.outfile, os.path.getsize(args.outfile) / 1e9, time.time() - t0,
             time.time() - t_all), flush=True)


if __name__ == '__main__':
    main()
