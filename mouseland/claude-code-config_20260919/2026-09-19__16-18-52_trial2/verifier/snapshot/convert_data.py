#!/usr/bin/env python3
"""
Convert the Zhong et al. 2025 ("Unsupervised pretraining in biological neural
networks") two-photon mesoscope dataset into the decoder-compatible pickle format.

Usage
-----
    python -u /app/convert_data.py <outpicklefile> [--full | --sample]
                                   [--show-processing] [--max-neurons N]
                                   [--workers K]

Design decisions and their justification are documented in /app/CONVERSION_NOTES.md.
Short summary:

  * one session per unique recording (mname, datexp, blk)  -> 89 sessions, 19 mice
  * a *trial* is one corridor traversal, restricted to samples inside the 0-4 m
    texture region while the virtual reality was moving
    (`ft_CorrSpc & (ft_move > 0)`), exactly the mask used by the reference code
    (utils.Get_dprime_selective_neuron) and stated in the paper's Methods
  * temporal alignment: trial start = corridor entry; time bin = 1 imaging frame
    (~314.9 ms, fs = 3.176 Hz); no positional interpolation
  * neurons: Suite2p deconvolved traces, visual-cortex only (iarea not in {-1, 7}),
    z-scored per neuron over the retained samples, then subsampled (stratified by
    visual area) to at most --max-neurons per session
"""

import argparse
import os
import pickle
import time
import zlib
from collections import OrderedDict
from datetime import date

import numpy as np

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

DATA_ROOT = '/app/data'
SEC_PER_DAY = 86400.0

# utils.neu_area_ID: mapping from the retinotopy `iarea` code to visual region.
# iarea == -1 and iarea == 7 are "outside of visual cortex" and are dropped
# (utils.Get_density_map).
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV']
IAREA_TO_REGION = {8: 0,                      # V1
                   0: 1, 1: 1, 2: 1, 9: 1,    # mHV  (medial HVAs)
                   5: 2, 6: 2,                # lHV  (lateral HVAs)
                   3: 3, 4: 3}                # aHV  (anterior HVAs)

# beh['stim_id'] taxonomy, documented in code/data_process_script.ipynb
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']

INPUT_NAMES = ['time_to_sound_cue_s', 'day_of_training',
               'time_since_trial_start_s', 'reward_available']
OUTPUT_NAMES = ['stimulus', 'licking', 'position_bin', 'speed_bin']
OUTPUT_VALUES = [STIM_NAMES,
                 ['no_lick', 'lick'],
                 ['0-1m', '1-2m', '2-3m', '3-4m'],
                 ['speed_q1', 'speed_q2', 'speed_q3', 'speed_q4']]

TEXTURE_LENGTH_DM = 40.0   # 4 m texture region; positions are in decimetres
N_POSITION_BINS = 4        # 4 equal-length 1 m bins
N_SPEED_BINS = 4           # quartiles of the running-speed distribution
MIN_SAMPLES_PER_TRIAL = 2  # guard; the observed minimum is 11
RNG_SEED = 2025            # the seed the reference code uses (utils.get_kfold_reward_response)


# --------------------------------------------------------------------------- #
# stage 1 -- behaviour (cheap; loads only /app/data/beh, ~1.5 s for all 89)
# --------------------------------------------------------------------------- #

BEH_FIELDS = ['ntrials', 'ft', 'ft_trInd', 'ft_CorrSpc', 'ft_move', 'ft_Pos',
              'ft_RunSpeed', 'LickFr', 'UniqWalls', 'WallName', 'isRew',
              'SoundFr', 'StartFr', 'GrayFr', 'Corridor_Length',
              'Texture_Length', 'Reward_Mode']


def load_session_registry(root=DATA_ROOT, verbose=True):
    """Collapse the 142 experiment-type records into 89 unique recordings.

    ``Imaging_Exp_info.npy`` lists each recording once per experiment type it takes
    part in.  The behaviour payload is identical between those views; only
    ``stim_id`` differs, because each view only labels the stimuli relevant to its
    own comparison and NaNs the rest.  We therefore take the union of the
    ``stim_id`` assignments (asserting that they never conflict).
    """
    exp_info = np.load(os.path.join(root, 'beh', 'Imaging_Exp_info.npy'),
                       allow_pickle=True).item()
    reg = OrderedDict()
    n_records = 0
    for exp_type, db in exp_info.items():
        beh_all = np.load(os.path.join(root, 'beh', 'Beh_%s.npy' % exp_type),
                          allow_pickle=True).item()
        for d in db:
            n_records += 1
            key = '%s_%s_%s' % (d['mname'], d['datexp'], d['blk'])
            beh_key = key + ('_' + d['stimtype'] if 'stimtype' in d else '')
            beh = beh_all[beh_key]
            sid = np.atleast_1d(np.array(beh['stim_id'], dtype=float))
            uw = np.asarray(beh['UniqWalls'])
            if key not in reg:
                reg[key] = dict(
                    beh={f: beh[f] for f in BEH_FIELDS if f in beh},
                    uniq_walls=uw, stim_id=sid.copy(),
                    mname=d['mname'], datexp=d['datexp'], blk=d['blk'],
                    exptypes=[], reward_type=d.get('rewType', 'unknown'),
                    cohort=d.get('exptype', 'naive'))
            else:
                e = reg[key]
                assert list(e['uniq_walls']) == list(uw), 'wall mismatch %s' % key
                assert e['beh']['ntrials'] == beh['ntrials'], 'ntrials mismatch %s' % key
                m = ~np.isnan(sid)
                conflict = m & ~np.isnan(e['stim_id']) & (e['stim_id'] != sid)
                assert not conflict.any(), 'stim_id conflict in %s' % key
                e['stim_id'][m] = sid[m]
                if 'exptype' in d:
                    e['cohort'] = d['exptype']
            reg[key]['exptypes'].append(exp_type)
        del beh_all
    if verbose:
        print('registry: %d unique recordings from %d experiment-type records'
              % (len(reg), n_records))
    # deterministic order: by mouse, then date, then block
    return OrderedDict(sorted(reg.items(),
                              key=lambda kv: (kv[1]['mname'], kv[1]['datexp'], kv[1]['blk'])))


def training_day_lookup(reg):
    """Days since each mouse's first imaging session, per session."""
    first = {}
    for entry in reg.values():
        d = date(*map(int, entry['datexp'].split('_')))
        first[entry['mname']] = min(first.get(entry['mname'], d), d)
    return {key: float((date(*map(int, e['datexp'].split('_'))) - first[e['mname']]).days)
            for key, e in reg.items()}


def build_session_behavior(key, entry, day, nframes_neural=None):
    """All behaviour-derived per-sample quantities for one session.

    ``nframes_neural`` is the number of imaging frames in the spk file.  The
    reference code always truncates the behaviour arrays with ``[:nfr]``; when the
    value is not yet known (stage 1) we use the full behaviour length and rebuild
    once the spk file has been opened.
    """
    beh = entry['beh']
    ft_full = np.asarray(beh['ft'], dtype=float)
    n = len(ft_full) if nframes_neural is None else min(len(ft_full), int(nframes_neural))

    ft = ft_full[:n]
    ft_tr = np.asarray(beh['ft_trInd'], dtype=float)[:n]
    ft_corr = np.asarray(beh['ft_CorrSpc'], dtype=bool)[:n]
    ft_move = np.asarray(beh['ft_move'], dtype=float)[:n]
    ft_pos = np.asarray(beh['ft_Pos'], dtype=float)[:n]
    ft_spd = np.asarray(beh['ft_RunSpeed'], dtype=float)[:n]

    dt = float(np.median(np.diff(ft)) * SEC_PER_DAY)       # ~0.3149 s

    running = ft_move > 0                                   # VR moved -> mouse running
    # "running clock": elapsed running time strictly before each frame
    tau = np.zeros(n)
    if n > 1:
        np.cumsum(running[:-1] * dt, out=tau[1:])

    # canonical stimulus role id per trial; NaN -> the trial is dropped (see notes)
    ntrials = int(beh['ntrials'])
    wall2id = {w: s for w, s in zip(entry['uniq_walls'], entry['stim_id'])}
    wall_name = np.asarray(beh['WallName'])[:ntrials]
    trial_stim = np.array([wall2id[w] for w in wall_name], dtype=float)

    # rewarded corridor (utils.get_cat_id: rewarded stimulus = WallName[isRew][0])
    is_rew = np.asarray(beh['isRew'], dtype=bool)[:ntrials]
    if is_rew.any():
        rew_wall = str(wall_name[is_rew][0])
        trial_rew = (wall_name == rew_wall).astype(np.float32)
    else:
        rew_wall = None
        trial_rew = np.zeros(ntrials, dtype=np.float32)

    # ---- retained samples: inside the texture region AND the VR moving ---- #
    tr_of_frame = np.where(np.isfinite(ft_tr), ft_tr, -1).astype(np.int64)
    keep = ft_corr & running & (tr_of_frame >= 0) & (tr_of_frame < ntrials)
    kept = np.flatnonzero(keep)
    kept = kept[np.isfinite(trial_stim[tr_of_frame[kept]])]

    # Boundary guard.  `ft_trInd` labels the corridor *and* the following grey
    # space, and in 21 of 37,801 trials the frame that sits within a hundredth of
    # a frame of the next corridor entry still carries the previous trial index
    # while `ft_Pos` has already wrapped to ~0 -- `ft_CorrSpc` is then True and the
    # sample would be mislabelled (position bin 0, wrong texture, ~11 s after
    # trial start).  Requiring the sample to fall between this trial's corridor
    # entry (`StartFr`) and grey-space entry (`GrayFr`), with one frame of slack
    # for the sub-frame interpolation of those two event times, removes exactly
    # those 21 samples and keeps the 1,301 legitimate last-corridor samples whose
    # frame index exceeds `GrayFr` by <0.1 frame.
    start_fr_all = np.asarray(beh['StartFr'], dtype=float)[:ntrials]
    gray_fr_all = np.asarray(beh['GrayFr'], dtype=float)[:ntrials]
    tr_kept = tr_of_frame[kept]
    inside = (kept >= start_fr_all[tr_kept] - 1.0) & (kept <= gray_fr_all[tr_kept] + 1.0)
    kept = kept[inside]
    trial = tr_of_frame[kept]

    # licks -> binary per frame. Frame k covers [t_k, t_{k+1}), so floor() is the
    # correct bin for a fractional lick frame index.
    lick_bin = np.zeros(n, dtype=bool)
    lf = np.floor(np.asarray(beh['LickFr'], dtype=float)).astype(np.int64)
    lf = lf[(lf >= 0) & (lf < n)]
    lick_bin[lf] = True

    # running-clock time of the alignment event and of the sound cue
    grid = np.arange(n, dtype=float)
    tau_start = np.interp(start_fr_all, grid, tau)
    tau_cue = np.interp(np.asarray(beh['SoundFr'], dtype=float)[:ntrials], grid, tau)

    return dict(
        key=key, n=n, dt=dt, ntrials=ntrials,
        frame_idx=kept, trial=trial,
        time_since_start=(tau[kept] - tau_start[trial]).astype(np.float32),
        time_to_cue=(tau_cue[trial] - tau[kept]).astype(np.float32),
        position=ft_pos[kept], speed=ft_spd[kept],
        lick=lick_bin[kept].astype(np.int64),
        trial_stim=trial_stim, trial_rew=trial_rew, day=np.float32(day),
        rew_wall=rew_wall, wall_name=wall_name,
        mname=entry['mname'], datexp=entry['datexp'], blk=entry['blk'],
        cohort=entry['cohort'], reward_mode=str(beh.get('Reward_Mode', '')),
    )


def finalize_trials(sb, speed_edges):
    """Split a session's retained samples into per-trial input/output arrays.

    Returns ``(inputs, outputs, slices, trial_ids)`` where ``slices[i] = (a, b)``
    indexes the session's retained-sample axis.
    """
    trial = sb['trial']
    # ft_trInd is non-decreasing in time, so samples of a trial are contiguous
    assert np.all(np.diff(trial) >= 0), 'trial indices not sorted for %s' % sb['key']
    uniq, starts = np.unique(trial, return_index=True)
    stops = np.append(starts[1:], len(trial))

    pos_bin = np.clip((sb['position'] / (TEXTURE_LENGTH_DM / N_POSITION_BINS)).astype(np.int64),
                      0, N_POSITION_BINS - 1)
    spd_bin = np.digitize(sb['speed'], speed_edges).astype(np.int64)

    inputs, outputs, slices, trial_ids = [], [], [], []
    for t, a, b in zip(uniq, starts, stops):
        T = b - a
        if T < MIN_SAMPLES_PER_TRIAL:
            continue
        inp = np.empty((4, T), dtype=np.float32)
        inp[0] = sb['time_to_cue'][a:b]
        inp[1] = sb['day']
        inp[2] = sb['time_since_start'][a:b]
        inp[3] = sb['trial_rew'][t]
        out = np.empty((4, T), dtype=np.int64)
        out[0] = int(sb['trial_stim'][t])
        out[1] = sb['lick'][a:b]
        out[2] = pos_bin[a:b]
        out[3] = spd_bin[a:b]
        inputs.append(inp)
        outputs.append(out)
        slices.append((int(a), int(b)))
        trial_ids.append(int(t))
    return inputs, outputs, slices, trial_ids


# --------------------------------------------------------------------------- #
# stage 2 -- neural data (expensive; loads /app/data/spk)
# --------------------------------------------------------------------------- #

def load_region_index(mname, datexp, root=DATA_ROOT):
    """Per-neuron visual-region index; -1 for neurons outside visual cortex."""
    d = np.load(os.path.join(root, 'retinotopy', '%s_%s_trans.npz' % (mname, datexp)),
                allow_pickle=True)
    iarea = np.asarray(d['iarea'])
    region = np.full(len(iarea), -1, dtype=np.int64)
    for code, ridx in IAREA_TO_REGION.items():
        region[iarea == code] = ridx
    return region


def stratified_subsample(region, max_neurons, rng):
    """<=max_neurons neuron indices, stratified across the four visual regions."""
    pool = np.flatnonzero(region >= 0)
    if len(pool) <= max_neurons:
        return np.sort(pool)
    counts = np.array([int(np.sum(region[pool] == r)) for r in range(len(BRAIN_REGIONS))])
    target = max_neurons * counts / counts.sum()
    quota = np.minimum(np.floor(target).astype(int), counts)
    while quota.sum() < max_neurons:
        deficit = np.where(quota < counts, target - quota, -np.inf)
        quota[int(np.argmax(deficit))] += 1
    picked = []
    for r in range(len(BRAIN_REGIONS)):
        p = pool[region[pool] == r]
        picked.append(p if quota[r] >= len(p)
                      else rng.choice(p, size=int(quota[r]), replace=False))
    return np.sort(np.concatenate(picked))


def process_session(entry, key, day, speed_edges, max_neurons, root=DATA_ROOT,
                    return_raw=False, zscore=True):
    """Full per-session pipeline: neural + behaviour, consistently truncated.

    The spk file is opened once; its frame count is used to re-derive the
    behaviour masks (the reference's ``[:nfr]`` truncation) before the trials are
    cut, so the neural and behavioural time axes can never drift apart.
    """
    path = os.path.join(root, 'spk', '%s_%s_%s_neural_data.npy'
                        % (entry['mname'], entry['datexp'], entry['blk']))
    d = np.load(path, allow_pickle=True).item()
    planes = d['spks']
    nframes = planes[0].shape[1]
    n_total = sum(p.shape[0] for p in planes)

    sb = build_session_behavior(key, entry, day, nframes_neural=nframes)
    inputs, outputs, slices, trial_ids = finalize_trials(sb, speed_edges)
    frame_idx = sb['frame_idx']

    region_all = load_region_index(entry['mname'], entry['datexp'], root=root)
    assert len(region_all) == n_total, \
        '%s: retinotopy has %d neurons, spks has %d' % (key, len(region_all), n_total)
    n_visual = int(np.sum(region_all >= 0))

    chunks = []
    while planes:
        p = planes.pop(0)
        chunks.append(p[:, frame_idx])
        del p
    del d
    X = np.concatenate(chunks, axis=0)
    del chunks

    sd_all = X.std(axis=1)
    region_all = region_all.copy()
    n_dead = int(np.sum((sd_all == 0) & (region_all >= 0)))
    region_all[sd_all == 0] = -1

    rng = np.random.default_rng(zlib.crc32(key.encode()) ^ RNG_SEED)
    sel = stratified_subsample(region_all, max_neurons, rng)

    Xs = np.ascontiguousarray(X[sel])
    del X
    raw = Xs.copy() if return_raw else None
    if zscore:
        Xs = (Xs - Xs.mean(axis=1, keepdims=True)) / Xs.std(axis=1, keepdims=True)
    Xs = Xs.astype(np.float32, copy=False)

    trials = [np.ascontiguousarray(Xs[:, a:b]) for (a, b) in slices]
    info = dict(n_total=n_total, n_visual=n_visual, n_dead=n_dead,
                n_kept=int(len(sel)), nframes=int(nframes),
                n_beh_frames=int(sb['n']))
    out = dict(sb=sb, inputs=inputs, outputs=outputs, slices=slices,
               trial_ids=trial_ids, trials=trials,
               region=region_all[sel], info=info)
    if return_raw:
        out['X'] = Xs
        out['raw'] = raw
    return out


def _worker(job):
    entry, key, day, speed_edges, max_neurons, zscore = job
    t0 = time.time()
    res = process_session(entry, key, day, speed_edges, max_neurons, zscore=zscore)
    res['info']['seconds'] = time.time() - t0
    # the behaviour dict is not needed in the parent; drop the bulky part
    res['sb'] = {k: v for k, v in res['sb'].items()
                 if k in ('key', 'n', 'dt', 'ntrials', 'rew_wall', 'day',
                          'mname', 'datexp', 'blk', 'cohort', 'reward_mode')}
    del res['slices']
    return key, res


# --------------------------------------------------------------------------- #
# plotting for --show-processing
# --------------------------------------------------------------------------- #

def show_processing(key, entry, res, speed_edges, outdir='/app'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sb, inputs, outputs = res['sb'], res['inputs'], res['outputs']
    slices, trial_ids = res['slices'], res['trial_ids']
    X, raw = res['X'], res['raw']

    beh = entry['beh']
    n = sb['n']
    ft_pos = np.asarray(beh['ft_Pos'], dtype=float)[:n]
    ft_move = np.asarray(beh['ft_move'], dtype=float)[:n]
    ft_spd = np.asarray(beh['ft_RunSpeed'], dtype=float)[:n]
    start_fr = np.asarray(beh['StartFr'], dtype=float)
    gray_fr = np.asarray(beh['GrayFr'], dtype=float)
    sound_fr = np.asarray(beh['SoundFr'], dtype=float)
    lick_fr = np.asarray(beh['LickFr'], dtype=float)

    show_tr = trial_ids[:6]
    k0 = max(0, int(np.floor(start_fr[show_tr[0]])) - 3)
    k1 = min(n, int(np.ceil(gray_fr[show_tr[-1]])) + 3)
    fr = np.arange(k0, k1)

    kept_mask = np.zeros(n, dtype=bool)
    kept_mask[sb['frame_idx']] = True

    # converted-column bookkeeping
    conv_rs = np.concatenate([np.arange(a, b) for (a, b) in slices])   # retained-sample idx
    col_of = -np.ones(len(sb['frame_idx']), dtype=np.int64)
    col_of[conv_rs] = np.arange(len(conv_rs))
    in_win = np.flatnonzero((sb['frame_idx'] >= k0) & (sb['frame_idx'] < k1))
    in_win = in_win[col_of[in_win] >= 0]
    fkept = sb['frame_idx'][in_win]
    cols = col_of[in_win]

    cat_in = np.concatenate(inputs, axis=1)
    cat_out = np.concatenate(outputs, axis=1)

    fig, axs = plt.subplots(8, 1, figsize=(16, 23), sharex=True)

    def mark(ax):
        for t in show_tr:
            ax.axvline(start_fr[t], color='g', lw=1.0)
            ax.axvline(gray_fr[t], color='k', lw=0.8, ls=':')
            ax.axvline(sound_fr[t], color='r', lw=1.0, ls='--')
        for k in fr[kept_mask[k0:k1]]:
            ax.axvspan(k - .5, k + .5, color='0.88', zorder=-10, lw=0)

    axs[0].plot(fr, ft_pos[k0:k1], 'k.-', ms=3, lw=.7)
    axs[0].axhline(TEXTURE_LENGTH_DM, color='b', lw=.8)
    axs[0].set_ylabel('ft_Pos (dm)')
    axs[0].set_title('%s -- step 1: raw VR position. green = corridor entry (trial start / '
                     'alignment event), dotted = grey-space entry, red dashed = sound cue, '
                     'grey shading = retained samples (corridor & VR moving)' % key, fontsize=9)
    mark(axs[0])

    axs[1].plot(fr, ft_move[k0:k1], 'k.-', ms=3, lw=.7)
    axs[1].axhline(0, color='r', lw=.6)
    axs[1].set_ylabel('ft_move (dm/frame)')
    axs[1].set_title('step 2: running filter, keep ft_move > 0', fontsize=9)
    mark(axs[1])

    axs[2].plot(fr, ft_spd[k0:k1], 'k.-', ms=3, lw=.7)
    for e in speed_edges:
        axs[2].axhline(e, color='m', lw=.7, ls='--')
    axs[2].set_ylabel('ft_RunSpeed')
    axs[2].set_title('step 3: running speed, with the global quartile edges (magenta)', fontsize=9)
    mark(axs[2])

    axs[3].plot(fkept, cat_in[0, cols], 'o-', ms=3, lw=.8, color='r', label='input 0: time to cue (s)')
    axs[3].plot(fkept, cat_in[2, cols], 'o-', ms=3, lw=.8, color='g',
                label='input 2: time since trial start (s)')
    axs[3].axhline(0, color='k', lw=.5)
    axs[3].set_ylabel('seconds')
    axs[3].legend(fontsize=7, loc='upper right')
    axs[3].set_title('step 4: time inputs on the running clock. time-to-cue crosses zero at the '
                     'red dashed cue line; time-since-start is 0 at the green line', fontsize=9)
    mark(axs[3])

    axs[4].step(fkept, cat_out[0, cols], where='mid', color='k', lw=1.5,
                label='output 0: stimulus id')
    axs[4].set_ylabel('stimulus id')
    ax4b = axs[4].twinx()
    ax4b.step(fkept, cat_in[3, cols], where='mid', color='b', lw=1.5,
              label='input 3: reward available')
    ax4b.fill_between(fkept, 0, cat_in[3, cols], step='mid', color='b', alpha=.15)
    ax4b.set_ylim(-0.1, 1.6)
    ax4b.set_ylabel('reward available', color='b')
    h1, l1 = axs[4].get_legend_handles_labels()
    h2, l2 = ax4b.get_legend_handles_labels()
    axs[4].legend(h1 + h2, l1 + l2, fontsize=7, loc='upper left')
    axs[4].set_title('step 5: per-trial input 3 / output 0 (day of training = %.0f; '
                     'rewarded wall = %s)' % (cat_in[1, 0], sb['rew_wall']), fontsize=9)
    mark(axs[4])

    axs[5].step(fkept, cat_out[1, cols], where='mid', color='b', lw=1.2, label='output 1: lick bin')
    lf = lick_fr[(lick_fr >= k0) & (lick_fr < k1)]
    axs[5].plot(lf, np.full(len(lf), 1.12), 'r|', ms=12, label='raw LickFr')
    axs[5].set_ylim(-0.2, 1.4)
    axs[5].set_ylabel('lick')
    axs[5].legend(fontsize=7, loc='upper right')
    axs[5].set_title('step 6: licking output vs raw lick times', fontsize=9)
    mark(axs[5])

    axs[6].plot(fr, ft_pos[k0:k1] / (TEXTURE_LENGTH_DM / N_POSITION_BINS), 'k.-', ms=3, lw=.6,
                label='ft_Pos / 10 dm')
    axs[6].step(fkept, cat_out[2, cols], where='mid', color='b', label='output 2: position bin')
    axs[6].step(fkept, cat_out[3, cols], where='mid', color='m', label='output 3: speed bin')
    axs[6].set_ylim(-0.5, 5.5)
    axs[6].set_ylabel('bin')
    axs[6].legend(fontsize=7, loc='upper right')
    axs[6].set_title('step 7: discretisation -- position into 4 x 1 m bins, speed into global '
                     'quartiles', fontsize=9)
    mark(axs[6])

    nshow = min(40, X.shape[0])
    axs[7].imshow(X[:nshow][:, in_win], aspect='auto', interpolation='nearest',
                  extent=[fkept[0], fkept[-1], nshow, 0], vmin=-1, vmax=4, cmap='gray_r')
    axs[7].set_ylabel('neuron')
    axs[7].set_xlabel('imaging frame index')
    axs[7].set_title('step 8: z-scored deconvolved activity of %d example neurons at the '
                     'retained samples (columns are exactly the shaded frames above)' % nshow,
                     fontsize=9)

    fig.tight_layout()
    fn = os.path.join(outdir, 'processing_%s.png' % key)
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    print('  wrote %s' % fn)

    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    axs[0, 0].hist(raw.ravel()[::37], bins=100, log=True, color='k')
    axs[0, 0].set_title('raw deconvolved values')
    axs[0, 1].hist(X.ravel()[::37], bins=100, log=True, color='b')
    axs[0, 1].set_title('after per-neuron z-score')
    axs[0, 2].hist([b - a for a, b in slices], bins=40, color='g')
    axs[0, 2].set_title('samples per trial (%d trials, mean %.1f)'
                        % (len(slices), np.mean([b - a for a, b in slices])))
    axs[1, 0].hist(cat_in[0], bins=80, color='r')
    axs[1, 0].set_title('input 0: time to sound cue (s)')
    axs[1, 1].hist(cat_in[2], bins=80, color='g')
    axs[1, 1].set_title('input 2: time since trial start (s)')
    axs[1, 2].bar(np.arange(4), np.bincount(cat_out[2], minlength=4) / cat_out.shape[1],
                  width=.35, label='position')
    axs[1, 2].bar(np.arange(4) + .35, np.bincount(cat_out[3], minlength=4) / cat_out.shape[1],
                  width=.35, label='speed')
    axs[1, 2].axhline(0.25, color='k', ls='--', lw=.8)
    axs[1, 2].legend(fontsize=8)
    axs[1, 2].set_title('output 2/3 class fractions (expect ~0.25 each)')
    fig.suptitle('%s -- distributions after conversion' % key)
    fig.tight_layout()
    fn = os.path.join(outdir, 'processing_%s_dist.png' % key)
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    print('  wrote %s' % fn)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outfile', type=str)
    ap.add_argument('--full', action='store_true', default=True,
                    help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true',
                    help='process only 2 sessions, for testing')
    ap.add_argument('--show-processing', action='store_true',
                    help='plot every processing step for up to 2 sessions')
    ap.add_argument('--max-neurons', type=int, default=2000,
                    help='max neurons kept per session (default 2000)')
    ap.add_argument('--no-zscore', action='store_true',
                    help='ablation: keep raw deconvolved amplitudes instead of z-scoring')
    ap.add_argument('--workers', type=int, default=6,
                    help='parallel workers for loading spk files (default 6)')
    args = ap.parse_args()

    t_start = time.time()
    np.random.seed(RNG_SEED)

    # ---------------- stage 1: behaviour ---------------- #
    t0 = time.time()
    reg = load_session_registry()
    day_lut = training_day_lookup(reg)
    sess_beh = OrderedDict((key, build_session_behavior(key, entry, day_lut[key]))
                           for key, entry in reg.items())
    t_beh = time.time() - t0
    print('stage 1 (behaviour, all %d sessions): %.1f s' % (len(reg), t_beh))

    # global running-speed quartiles over *all* sessions, so that the sample run and
    # the full run use identical class boundaries
    all_speed = np.concatenate([sb['speed'] for sb in sess_beh.values()])
    speed_edges = np.percentile(all_speed, [25, 50, 75])
    print('global running-speed quartile edges: %s   (n = %d samples, range [%.2f, %.2f])'
          % (np.round(speed_edges, 4), len(all_speed), all_speed.min(), all_speed.max()))
    del all_speed

    keys = list(sess_beh.keys())
    if args.sample:
        # two of the smallest recordings, one task session and one non-task session
        by_size = sorted(keys, key=lambda k: os.path.getsize(os.path.join(
            DATA_ROOT, 'spk', '%s_%s_%s_neural_data.npy'
            % (sess_beh[k]['mname'], sess_beh[k]['datexp'], sess_beh[k]['blk']))))
        pick = []
        for want_rew in (True, False):
            for k in by_size:
                if (sess_beh[k]['rew_wall'] is not None) == want_rew and k not in pick:
                    pick.append(k)
                    break
        keys = pick
        print('sample mode: %s' % keys)

    # ---------------- stage 2: neural ---------------- #
    results = OrderedDict()
    t0 = time.time()

    if args.show_processing:
        for key in keys[:2]:
            print('show-processing: %s' % key)
            res = process_session(reg[key], key, day_lut[key], speed_edges,
                                  args.max_neurons, return_raw=True,
                                  zscore=not args.no_zscore)
            show_processing(key, reg[key], res, speed_edges)
            del res['X'], res['raw']
            res['sb'] = {k: v for k, v in res['sb'].items()
                         if k in ('key', 'n', 'dt', 'ntrials', 'rew_wall', 'day',
                                  'mname', 'datexp', 'blk', 'cohort', 'reward_mode')}
            del res['slices']
            res['info']['seconds'] = float('nan')
            results[key] = res

    todo = [k for k in keys if k not in results]
    jobs = [(reg[k], k, day_lut[k], speed_edges, args.max_neurons, not args.no_zscore)
            for k in todo]
    if args.workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(processes=min(args.workers, len(jobs))) as pool:
            for i, (key, res) in enumerate(pool.imap_unordered(_worker, jobs)):
                results[key] = res
                el, done, tot = time.time() - t0, i + 1, len(jobs)
                info = res['info']
                print('  [%2d/%2d] %-22s %5d neurons (of %6d visual / %6d total), %4d trials, '
                      '%5.1f s | elapsed %5.0f s, eta %5.0f s'
                      % (done, tot, key, info['n_kept'], info['n_visual'], info['n_total'],
                         len(res['trials']), info['seconds'], el, el / done * (tot - done)),
                      flush=True)
    else:
        for i, job in enumerate(jobs):
            key, res = _worker(job)
            results[key] = res
            info = res['info']
            print('  [%2d/%2d] %-22s %5d neurons, %4d trials, %5.1f s | elapsed %5.0f s'
                  % (i + 1, len(jobs), key, info['n_kept'], len(res['trials']),
                     info['seconds'], time.time() - t0), flush=True)
    t_neural = time.time() - t0
    print('stage 2 (neural): %.1f s for %d sessions (%.2f s/session)'
          % (t_neural, len(keys), t_neural / max(1, len(jobs))))

    # ---------------- assemble ---------------- #
    subjects = sorted({sess_beh[k]['mname'] for k in keys})
    data = dict(neural=[], input=[], output=[], subjects=subjects, subject_idx=[],
                brain_regions=list(BRAIN_REGIONS), brain_region_idx=[],
                input_names=list(INPUT_NAMES), output_names=list(OUTPUT_NAMES),
                output_values=[list(v) for v in OUTPUT_VALUES])
    session_info = []
    n_trials_dropped = 0
    for key in keys:
        res = results[key]
        sb, info = res['sb'], res['info']
        assert len(res['trials']) == len(res['inputs']) == len(res['outputs']), key
        data['neural'].append(res['trials'])
        data['input'].append(res['inputs'])
        data['output'].append(res['outputs'])
        data['subject_idx'].append(subjects.index(sb['mname']))
        data['brain_region_idx'].append(np.asarray(res['region'], dtype=np.int64))
        n_trials_dropped += sb['ntrials'] - len(res['trials'])
        session_info.append(dict(
            session=key, mouse=sb['mname'], date=sb['datexp'], block=sb['blk'],
            cohort=sb['cohort'], reward_mode=sb['reward_mode'],
            rewarded_wall=sb['rew_wall'], day_of_training=float(sb['day']),
            n_trials=len(res['trials']), n_trials_recorded=int(sb['ntrials']),
            n_neurons_kept=int(info['n_kept']), n_neurons_visual=int(info['n_visual']),
            n_neurons_recorded=int(info['n_total']),
            n_zero_variance_neurons=int(info['n_dead']),
            n_timepoints=int(sum(t.shape[1] for t in res['trials'])),
            n_imaging_frames=int(info['nframes']), frame_interval_s=float(sb['dt']),
            experiment_types=sorted(set(reg[key]['exptypes']))))
    data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)

    dts = [s['frame_interval_s'] for s in session_info]
    data['metadata'] = dict(
        task_description=(
            'Head-fixed mice run through 4 m linear virtual-reality corridors whose walls '
            'carry one of several "frozen" naturalistic textures (circle/leaf, or rock/wood '
            'in some mice), separated by 2 m of grey space. A sound cue is played at a '
            'position drawn uniformly from 0.5-3.5 m on every trial; in task ("sup") mice the '
            'cue marks the start of the reward zone in one of the two corridors and licking '
            'after the cue there delivers water. Unsupervised and naive mice see the same '
            'corridors and hear the same cue but are never rewarded. Neural activity is '
            'Suite2p-deconvolved two-photon mesoscope data from V1 and higher visual areas. '
            'Decoder outputs: (0) stimulus = which of the 7 canonical corridor textures is '
            'shown (per trial; the reference "stim_id" role code, so labels are comparable '
            'across mice that saw different image sets); (1) licking = at least one lick in '
            'this time bin; (2) position_bin = which of the four 1 m segments of the corridor '
            'the animal is in; (3) speed_bin = quartile of the running-speed distribution '
            '(edges computed once over the whole dataset). Decoder inputs: signed time to the '
            'sound cue, day of training, time since corridor entry, and whether this corridor '
            'is the rewarded one.'),
        time_bin_size=float(np.mean(dts) * 1000.0),
        temporal_alignment_event='trial start = entry into the visual corridor (position 0 m)',
        off_start=0.0,
        off_end=None,
        off_end_note=('trials have variable length because they end when the animal reaches '
                      '4 m; median %.2f s of running, 99th pct %.2f s'),
        paper=('Zhong, Baptista, Gattoni, Arnold, Flickinger, Stringer & Pachitariu, '
               '"Unsupervised pretraining in biological neural networks", Nature 644:741 (2025)'),
        data_source='figshare item 28811129 (spk/, beh/, retinotopy/)',
        neural_signal=('Suite2p non-negative deconvolved calcium traces (decay tau = 0.75 s)'
                       + (', z-scored per neuron over the retained samples of the session'
                          if not args.no_zscore else ', raw amplitudes (z-scoring disabled)')),
        sampling_rate_hz=float(1.0 / np.mean(dts)),
        frame_interval_s_range=[float(np.min(dts)), float(np.max(dts))],
        sample_selection=('samples inside the 0-4 m texture region (beh["ft_CorrSpc"]) while the '
                          'virtual reality was moving (beh["ft_move"] > 0), matching the reference '
                          'code (utils.Get_dprime_selective_neuron) and the paper ("we only '
                          'selected data points inside the 0-4-m region ... we excluded the data '
                          'points in which the animal was not running")'),
        neuron_selection=('neurons inside visual cortex (retinotopy iarea not in {-1, 7}, i.e. '
                          'utils.neu_area_ID V1/mHV/lHV/aHV), excluding zero-variance neurons, '
                          'then subsampled stratified by visual region (seed %d) to at most %d '
                          'neurons per session' % (RNG_SEED, args.max_neurons)),
        max_neurons_per_session=int(args.max_neurons),
        trial_selection=('every corridor traversal whose wall texture has a canonical stim_id; '
                         'trials showing "circle3" (309 of 38110 trials, 4 sessions) are dropped '
                         'because the reference assigns them no stimulus role and excludes them '
                         'from all analyses'),
        time_base=('times are measured on the "running clock": elapsed time accumulated over '
                   'running frames only, because non-running samples are excluded from the '
                   'analysis and a few percent of trials contain multi-minute stops'),
        speed_bin_edges=[float(e) for e in speed_edges],
        speed_units='cm/s (beh["ft_RunSpeed"])',
        position_bin_edges_m=[0.0, 1.0, 2.0, 3.0, 4.0],
        corridor_length_m=4.0, grey_space_length_m=2.0,
        n_sessions=len(keys), n_subjects=len(subjects),
        session_info=session_info)

    Ts = [t.shape[1] for s in data['neural'] for t in s]
    data['metadata']['off_end_note'] = data['metadata']['off_end_note'] % (
        np.median(Ts) * np.mean(dts), np.percentile(Ts, 99) * np.mean(dts))

    # ---------------- sanity checks ---------------- #
    ntr = [len(x) for x in data['neural']]
    print('\nsanity checks')
    print('  sessions            : %d' % len(data['neural']))
    print('  subjects            : %d' % len(subjects))
    print('  trials              : %d  (per session min/median/max %d/%d/%d)'
          % (sum(ntr), min(ntr), int(np.median(ntr)), max(ntr)))
    print('  timepoints          : %d  (per trial mean %.2f, min %d, max %d)'
          % (sum(Ts), np.mean(Ts), min(Ts), max(Ts)))
    print('  trials dropped      : %d' % n_trials_dropped)
    print('  neurons             : %d total, %.0f mean/session, min %d, max %d'
          % (sum(len(r) for r in data['brain_region_idx']),
             np.mean([len(r) for r in data['brain_region_idx']]),
             min(len(r) for r in data['brain_region_idx']),
             max(len(r) for r in data['brain_region_idx'])))
    for i, r in enumerate(BRAIN_REGIONS):
        print('    %-4s: %7d' % (r, sum(int(np.sum(x == i)) for x in data['brain_region_idx'])))

    cat_out = np.concatenate([o for s in data['output'] for o in s], axis=1)
    cat_in = np.concatenate([o for s in data['input'] for o in s], axis=1)
    for i, nm in enumerate(INPUT_NAMES):
        print('  input  %-26s range [%.3f, %.3f], mean %.3f'
              % (nm, cat_in[i].min(), cat_in[i].max(), cat_in[i].mean()))
    for i, nm in enumerate(OUTPUT_NAMES):
        fr = np.bincount(cat_out[i], minlength=len(OUTPUT_VALUES[i])) / cat_out.shape[1]
        print('  output %-26s fractions %s' % (nm, np.round(fr, 4)))

    for s in range(len(data['neural'])):
        assert len(data['input'][s]) == len(data['neural'][s])
        assert len(data['output'][s]) == len(data['neural'][s])
        assert len(data['brain_region_idx'][s]) == data['neural'][s][0].shape[0]
        for t in range(len(data['neural'][s])):
            nt, it, ot = data['neural'][s][t], data['input'][s][t], data['output'][s][t]
            assert nt.shape[1] == it.shape[1] == ot.shape[1], (s, t)
            assert np.isfinite(nt).all() and np.isfinite(it).all(), (s, t)
            assert (it[2] >= -1e-4).all(), 'negative time since trial start'
            assert np.all(np.diff(it[2]) >= -1e-4), 'time since trial start not monotone'
            for k in range(4):
                assert ot[k].min() >= 0 and ot[k].max() < len(OUTPUT_VALUES[k]), (s, t, k)
            # the VR corridor only ever moves forward within a trial
            assert np.all(np.diff(ot[2]) >= 0), 'position bin decreases in (%d, %d)' % (s, t)
            assert ot[2][0] == 0, 'trial (%d, %d) does not start at 0-1 m' % (s, t)
    print('  all per-trial assertions passed')

    # ---------------- save ---------------- #
    t0 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('\nwrote %s (%.2f GB) in %.1f s'
          % (args.outfile, os.path.getsize(args.outfile) / 1e9, time.time() - t0))
    print('TOTAL %.1f s  (stage 1 %.1f s, stage 2 %.1f s)'
          % (time.time() - t_start, t_beh, t_neural))


if __name__ == '__main__':
    main()
