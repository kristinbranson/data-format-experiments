"""Convert Zhong et al. 2025 virtual-reality visual-discrimination dataset
(two-photon mesoscope imaging, /app/data) into the decoder format described in
the task specification.

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Processing summary (see /app/CONVERSION_NOTES.md for full justification):
  * sessions      : the 89 unique imaging recordings listed in beh/Imaging_Exp_info.npy
  * neurons       : suite2p deconvolved traces (`spks`, concatenated over planes, as in
                    utils.load_spk); only neurons with a retinotopic area label
                    (utils.neu_area_ID -> V1/mHV/lHV/aHV); subsampled (stratified by area)
                    to at most MAX_NEURONS per session
  * timepoints    : frames inside the 4 m texture corridor while the VR was moving
                    (running), i.e. the reference mask (ft_move>0) & ft_CorrSpc, per trial
  * alignment     : trial start = corridor entry (beh['StartFr'])
  * time bin      : the imaging frame interval (~315 ms, fs = 3.17 Hz); no re-binning
"""

import argparse
import os
import pickle
import sys
import time
import datetime
import collections
import multiprocessing as mp

import numpy as np

DATA_ROOT = '/app/data'
MAX_NEURONS = 2000          # per-session cap on the number of neurons kept
MIN_FRAMES_PER_TRIAL = 5    # trials with fewer retained running frames are dropped
SEED = 2025                 # same seed as the reference code (utils.get_kfold_reward_response)
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV']
STIM_CATEGORIES = ['circle', 'leaf', 'rock', 'brick']
POSITION_VALUES = ['0-1m', '1-2m', '2-3m', '3-4m']
SPEED_VALUES = ['Q1 (slowest 25%)', 'Q2', 'Q3', 'Q4 (fastest 25%)']
N_POS_BINS = 4
TEXTURE_LENGTH_DM = 40.0     # 4 m texture corridor, positions are in decimetres


# ----------------------------------------------------------------------------- helpers
def neu_area_ID(iarea):
    """Copied from /app/code/utils.py (reference code) -- area masks from `iarea`."""
    area_name = ['V1', 'mHV', 'lHV', 'aHV']
    idx = {}
    for ar in area_name:
        if ar == 'V1':
            idx[ar] = iarea == 8
        elif ar == 'mHV':
            idx[ar] = (iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)
        elif ar == 'lHV':
            idx[ar] = (iarea == 5) | (iarea == 6)
        elif ar == 'aHV':
            idx[ar] = (iarea == 3) | (iarea == 4)
    return idx


def texture_family(wall_name):
    """Map a wall name (circle1, leaf1_swap2, wood5, rock2, ...) to its texture family.

    The paper uses four texture photographs: circle, leaf, rock and brick; the data files
    call the brick texture `wood`. Frozen-crop variants (leaf1/leaf2/leaf3/leaf1_swap*)
    all come from the same photograph and are pooled, exactly as the paper pools them.
    """
    base = wall_name.split('_')[0]
    base = ''.join(ch for ch in base if not ch.isdigit())
    if base == 'wood':
        base = 'brick'
    return base


def session_key(db):
    return (db['mname'], db['datexp'], str(db['blk']))


def beh_key(db):
    kn = '%s_%s_%s' % (db['mname'], db['datexp'], db['blk'])
    if 'stimtype' in db:
        kn = kn + '_' + db['stimtype']
    return kn


def load_exp_info():
    return np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()


def unique_sessions(exp_info):
    """One entry per unique recording, keeping the first exp_type it appears in.

    Sessions appear in up to 5 exp_types; the behaviour dicts were verified identical
    across those copies (see CONVERSION_NOTES Step 4).
    """
    reps = {}
    exp_types = {}
    for exp_type, dbs in exp_info.items():
        for db in dbs:
            key = session_key(db)
            exp_types.setdefault(key, []).append(exp_type)
            if key not in reps:
                reps[key] = (exp_type, db)
    ordered = sorted(reps.keys())
    return [(k, reps[k][0], reps[k][1], exp_types[k]) for k in ordered]


def cohort_of(db, exp_types):
    """Experimental cohort label for the session."""
    ex = db.get('exptype', None)
    if isinstance(ex, str) and ex in ('sup', 'unsup', 'naive'):
        cohort = ex
    else:
        cohort = 'naive'
    if any('grating' in e for e in exp_types):
        cohort = cohort + '_grating'
    return cohort


# --------------------------------------------------------------- behaviour (pass 1)
def process_behaviour(beh, day_of_training):
    """Extract per-trial frame indices, inputs and outputs from one behaviour dict.

    Only frames inside the texture corridor with the VR moving (mouse running) are kept,
    reproducing the reference mask `fr_valid = (ft_move>0) & ft_CorrSpc`
    (utils.Get_dprime_selective_neuron).

    Returns a dict with, for every kept trial: absolute neural frame indices, the
    four decoder inputs and the (not-yet-binned-for-speed) output values.
    """
    ntrials = int(beh['ntrials'])
    nfr_beh = len(beh['ft'])
    frame_idx = np.arange(nfr_beh)

    t_frame = beh['ft'] * 86400.0                       # datenum (days) -> seconds
    move = beh['ft_move'][:nfr_beh] > 0                  # VR moved => mouse ran > 6 cm/s
    corr = beh['ft_CorrSpc'][:nfr_beh].astype(bool)      # inside the 4 m texture corridor
    ftr = beh['ft_trInd'][:nfr_beh].astype(float)        # trial index of each frame
    pos = beh['ft_Pos'][:nfr_beh].astype(float)          # position within trial, dm
    speed = beh['ft_RunSpeed'][:nfr_beh].astype(float)   # cm/s

    # times of the alignment event (corridor entry) and of the sound cue, in seconds;
    # StartFr / SoundFr are fractional frame indices -> interpolate the frame clock
    t_start = np.interp(beh['StartFr'], frame_idx, t_frame)
    t_cue = np.interp(beh['SoundFr'], frame_idx, t_frame)

    # licking: a frame is "licking" if a lick timestamp falls in it (licks are binned by
    # integer neural frame, as in utils.spk_2_cue / spk_2_firstLick)
    lick_bin = np.zeros(nfr_beh, dtype=bool)
    lf = np.floor(np.asarray(beh['LickFr'], dtype=float))
    lf = lf[np.isfinite(lf)].astype(int)
    lf = lf[(lf >= 0) & (lf < nfr_beh)]
    lick_bin[lf] = True

    valid = move & corr & np.isfinite(ftr)
    vidx = np.where(valid)[0]
    vtr = ftr[vidx].astype(int)
    # group valid frames by trial (frames are in temporal order, trials are sequential)
    order = np.argsort(vtr, kind='stable')
    vidx, vtr = vidx[order], vtr[order]
    lo = np.searchsorted(vtr, np.arange(ntrials), side='left')
    hi = np.searchsorted(vtr, np.arange(ntrials), side='right')

    wall = np.asarray(beh['WallName'])
    stim_cat = np.array([STIM_CATEGORIES.index(texture_family(w)) for w in wall])
    is_rew = np.asarray(beh['isRew']).astype(bool)

    trials = []
    for t in range(ntrials):
        frames = vidx[lo[t]:hi[t]]
        if len(frames) == 0:
            continue
        trials.append(dict(
            trial=t,
            frames=frames,
            # --- inputs
            time_to_cue=(t_cue[t] - t_frame[frames]).astype(np.float32),
            day_of_training=np.float32(day_of_training),
            time_since_start=(t_frame[frames] - t_start[t]).astype(np.float32),
            reward_available=np.float32(1.0 if is_rew[t] else 0.0),
            # --- outputs
            stim_cat=np.int64(stim_cat[t]),
            lick=lick_bin[frames].astype(np.int64),
            pos=pos[frames],
            speed=speed[frames],
        ))
    return dict(trials=trials, nfr_beh=nfr_beh, ntrials=ntrials,
                n_rew_trials=int(is_rew.sum()),
                dt=float(np.median(np.diff(t_frame))))


def first_session_dates(all_sessions):
    """Date of each mouse's first imaging session (over the whole dataset).

    Computed from all 89 sessions so that the `day_of_training` input is identical in
    --sample and --full runs.
    """
    first_date = {}
    for key, _, _, _ in all_sessions:
        m, date = key[0], key[1]
        d = datetime.date(*map(int, date.split('_')))
        if m not in first_date or d < first_date[m]:
            first_date[m] = d
    return first_date


def behaviour_pass(sessions, first_date, verbose=True):
    """Run the behaviour extraction for all requested sessions.

    Each Beh_<exp_type>.npy file is loaded exactly once (they are 100-430 MB each).
    """

    by_exp = collections.defaultdict(list)
    for i, (key, exp_type, db, ets) in enumerate(sessions):
        by_exp[exp_type].append(i)

    results = [None] * len(sessions)
    for exp_type in sorted(by_exp):
        t0 = time.time()
        B = np.load(os.path.join(DATA_ROOT, 'beh', 'Beh_%s.npy' % exp_type),
                    allow_pickle=True).item()
        for i in by_exp[exp_type]:
            key, _, db, ets = sessions[i]
            beh = B[beh_key(db)]
            day = (datetime.date(*map(int, key[1].split('_'))) - first_date[key[0]]).days
            results[i] = process_behaviour(beh, day)
            results[i]['day_of_training'] = day
        del B
        if verbose:
            print('  behaviour: %-32s %2d session(s)  %.1fs'
                  % (exp_type, len(by_exp[exp_type]), time.time() - t0), flush=True)
    return results


# ------------------------------------------------------------------- neural (pass 2)
def select_neurons(iarea, rng):
    """Indices (sorted) of the neurons to keep, and their brain-region index.

    Neurons without an area label (`iarea` in {-1, 7}) are dropped: they are excluded from
    every area analysis of the reference code. The remaining neurons are subsampled to
    MAX_NEURONS, stratified proportionally across the four areas.
    """
    masks = neu_area_ID(iarea)
    per_region = [np.where(masks[r])[0] for r in BRAIN_REGIONS]
    counts = np.array([len(x) for x in per_region])
    total = counts.sum()
    if total > MAX_NEURONS:
        # proportional allocation, guaranteeing at least 1 neuron per non-empty region
        alloc = np.floor(counts / total * MAX_NEURONS).astype(int)
        alloc = np.minimum(alloc, counts)
        alloc[(counts > 0) & (alloc == 0)] = 1
        # distribute the remainder to the largest regions
        while alloc.sum() < MAX_NEURONS:
            room = counts - alloc
            if room.max() <= 0:
                break
            alloc[np.argmax(room)] += 1
        while alloc.sum() > MAX_NEURONS:
            alloc[np.argmax(alloc)] -= 1
    else:
        alloc = counts
    keep, region = [], []
    for r in range(len(BRAIN_REGIONS)):
        if alloc[r] <= 0:
            continue
        pick = rng.choice(per_region[r], size=alloc[r], replace=False)
        keep.append(pick)
        region.append(np.full(alloc[r], r, dtype=np.int64))
    keep = np.concatenate(keep)
    region = np.concatenate(region)
    order = np.argsort(keep)                      # keep neurons in recording order
    return keep[order], region[order], counts, total


def load_spk_rows(mname, datexp, blk, rows, frames):
    """Load the deconvolved traces of the selected neurons at the selected frames.

    Mirrors utils.load_spk (concatenation of the per-plane `spks` arrays along axis 0)
    but only materialises the requested rows/columns, which keeps memory bounded.
    Returns (activity (len(rows), len(frames)) float32, n_frames_in_recording).
    """
    fn = os.path.join(DATA_ROOT, 'spk', '%s_%s_%s_neural_data.npy' % (mname, datexp, blk))
    spks = np.load(fn, allow_pickle=True).item()['spks']
    nfr = spks[0].shape[1]
    frames = frames[frames < nfr]
    out = np.empty((len(rows), len(frames)), dtype=np.float32)
    offset = 0
    filled = 0
    for plane in spks:
        n = plane.shape[0]
        sel = rows[(rows >= offset) & (rows < offset + n)] - offset
        if len(sel):
            out[filled:filled + len(sel)] = plane[sel][:, frames]
            filled += len(sel)
        offset += n
    assert filled == len(rows), (filled, len(rows))
    del spks
    return out, nfr


def convert_session(args):
    """Worker: build the neural matrices (and filtered behaviour) for one session."""
    isess, key, db, beh_res = args
    t0 = time.time()
    mname, datexp, blk = key
    ret = np.load(os.path.join(DATA_ROOT, 'retinotopy', '%s_%s_trans.npz' % (mname, datexp)),
                  allow_pickle=True)
    iarea = ret['iarea']
    rng = np.random.default_rng(SEED + isess)
    rows, region, region_counts, n_labelled = select_neurons(iarea, rng)

    trials = beh_res['trials']
    all_frames = np.concatenate([tr['frames'] for tr in trials]) if trials else np.zeros(0, int)
    all_frames = np.unique(all_frames)
    act, nfr_spk = load_spk_rows(mname, datexp, blk, rows, all_frames)
    kept_frames = all_frames[all_frames < nfr_spk]
    frame_pos = {f: i for i, f in enumerate(kept_frames)}

    neural, keep_trials = [], []
    for tr in trials:
        frames = tr['frames'][tr['frames'] < nfr_spk]
        if len(frames) < MIN_FRAMES_PER_TRIAL:
            continue
        cols = np.array([frame_pos[f] for f in frames], dtype=np.int64)
        neural.append(np.ascontiguousarray(act[:, cols], dtype=np.float32))
        tr = dict(tr)
        tr['frames'] = frames
        keep_trials.append(tr)
    del act

    info = dict(
        n_neurons_recorded=int(len(iarea)),
        n_neurons_labelled=int(n_labelled),
        n_neurons_kept=int(len(rows)),
        region_counts_recorded=region_counts.tolist(),
        n_frames_spk=int(nfr_spk),
        n_frames_beh=int(beh_res['nfr_beh']),
        ntrials_raw=int(beh_res['ntrials']),
        ntrials_kept=len(keep_trials),
        n_rew_trials=int(beh_res['n_rew_trials']),
        dt=beh_res['dt'],
        seconds=time.time() - t0,
    )
    return isess, neural, keep_trials, region, info


# ------------------------------------------------------------------------- plotting
def plot_processing(session_id, neural, trials, speed_edges, info, outfile):
    """Visualise every processing step for one session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ntr = min(6, len(trials))
    fig, ax = plt.subplots(6, 1, figsize=(16, 20))
    # --- 1. neural raster of the first trials
    cat = np.concatenate([neural[t] for t in range(ntr)], axis=1)
    bounds = np.cumsum([neural[t].shape[1] for t in range(ntr)])
    nshow = min(80, cat.shape[0])
    ax[0].imshow(cat[:nshow], aspect='auto', vmin=0,
                 vmax=np.percentile(cat[:nshow], 99) + 1e-6, cmap='gray_r', interpolation='nearest')
    for b in bounds[:-1]:
        ax[0].axvline(b - 0.5, color='r', lw=1)
    ax[0].set_title('%s: deconvolved activity, first %d trials (red = trial boundary), %d neurons shown'
                    % (session_id, ntr, nshow))
    ax[0].set_ylabel('neuron')

    # --- 2. position + discretisation
    pos = np.concatenate([trials[t]['pos'] for t in range(ntr)])
    posbin = np.concatenate([np.clip(np.floor(trials[t]['pos'] / (TEXTURE_LENGTH_DM / N_POS_BINS)),
                                     0, N_POS_BINS - 1) for t in range(ntr)])
    ax[1].plot(pos / 10.0, '-o', ms=3, label='position (m)')
    ax[1].step(np.arange(len(posbin)), posbin, color='r', where='mid', label='position bin')
    for b in bounds[:-1]:
        ax[1].axvline(b - 0.5, color='k', lw=0.5)
    for y in [1, 2, 3]:
        ax[1].axhline(y, color='0.8', lw=0.5)
    ax[1].legend(loc='upper right'); ax[1].set_title('position within corridor and 1-m bins')

    # --- 3. speed + quartile bins
    sp = np.concatenate([trials[t]['speed'] for t in range(ntr)])
    spb = np.digitize(sp, speed_edges)
    ax[2].plot(sp, '-o', ms=3, label='running speed (cm/s)')
    for e in speed_edges:
        ax[2].axhline(e, color='0.7', lw=0.8)
    ax2b = ax[2].twinx(); ax2b.step(np.arange(len(spb)), spb, color='r', where='mid')
    ax2b.set_ylabel('speed bin', color='r')
    ax[2].legend(loc='upper right'); ax[2].set_title('running speed and global quartile bins (grey lines = edges)')

    # --- 4. inputs
    ttc = np.concatenate([trials[t]['time_to_cue'] for t in range(ntr)])
    tss = np.concatenate([trials[t]['time_since_start'] for t in range(ntr)])
    ax[3].plot(tss, '-o', ms=3, label='time since trial start (s)')
    ax[3].plot(ttc, '-o', ms=3, label='time to sound cue (s)')
    ax[3].axhline(0, color='k', lw=0.5)
    for b in bounds[:-1]:
        ax[3].axvline(b - 0.5, color='k', lw=0.5)
    ax[3].legend(loc='upper right'); ax[3].set_title('time-varying inputs (cue crossing zero = cue frame)')

    # --- 5. licking + per-trial variables
    lick = np.concatenate([trials[t]['lick'] for t in range(ntr)])
    ax[4].step(np.arange(len(lick)), lick, where='mid', label='licking')
    ax[4].step(np.arange(len(lick)),
               np.concatenate([np.full(len(trials[t]['lick']), trials[t]['reward_available'])
                               for t in range(ntr)]), where='mid', label='reward available')
    ax[4].step(np.arange(len(lick)),
               np.concatenate([np.full(len(trials[t]['lick']), trials[t]['stim_cat'])
                               for t in range(ntr)]), where='mid', label='stimulus category')
    for b in bounds[:-1]:
        ax[4].axvline(b - 0.5, color='k', lw=0.5)
    ax[4].legend(loc='upper right'); ax[4].set_title('licking output, reward-availability input, stimulus category output')

    # --- 6. session-level distributions
    nfrm = np.array([t['neural'].shape[1] if 'neural' in t else len(t['lick']) for t in trials])
    ax[5].hist(nfrm, bins=np.arange(nfrm.min(), nfrm.max() + 2) - 0.5)
    ax[5].set_title('retained running frames per trial (median %.0f, n=%d trials); '
                    'kept %d of %d recorded neurons'
                    % (np.median(nfrm), len(nfrm), info['n_neurons_kept'], info['n_neurons_recorded']))
    ax[5].set_xlabel('frames')
    fig.tight_layout()
    fig.savefig(outfile, dpi=110)
    plt.close(fig)
    print('  wrote %s' % outfile, flush=True)


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='save processing plots for up to 2 sessions')
    ap.add_argument('--nproc', type=int, default=6, help='parallel workers for the neural pass')
    ap.add_argument('--max-neurons', type=int, default=MAX_NEURONS)
    args = ap.parse_args()

    globals()['MAX_NEURONS'] = args.max_neurons

    t_start_all = time.time()
    exp_info = load_exp_info()
    sessions = unique_sessions(exp_info)
    first_date = first_session_dates(sessions)   # over ALL sessions, not just the sample
    print('found %d unique sessions in Imaging_Exp_info.npy' % len(sessions))
    if args.sample:
        # two sessions from different mice/cohorts: one task (rewarded) and one unsupervised
        pick = [i for i, (k, et, db, ets) in enumerate(sessions)
                if k == ('TX108', '2023_03_25', '1')]
        pick += [i for i, (k, et, db, ets) in enumerate(sessions)
                 if k == ('TX83', '2022_08_29', '1')]
        sessions = [sessions[i] for i in pick]
        print('--sample: processing %d sessions: %s' % (len(sessions), [s[0] for s in sessions]))

    # ---------------- pass 1: behaviour
    print('\n=== pass 1: behaviour ===', flush=True)
    t0 = time.time()
    beh_results = behaviour_pass(sessions, first_date)
    t_beh = time.time() - t0
    print('behaviour pass: %.1fs (%.2fs/session)' % (t_beh, t_beh / len(sessions)), flush=True)

    # global speed quartile edges over all retained frames (25% of the data per bin)
    speeds = np.concatenate([tr['speed'] for res in beh_results for tr in res['trials']])
    speed_edges = np.percentile(speeds, [25, 50, 75])
    print('speed quartile edges (cm/s): %s  [n=%d retained frames]'
          % (np.round(speed_edges, 3).tolist(), len(speeds)), flush=True)

    # ---------------- pass 2: neural
    print('\n=== pass 2: neural ===', flush=True)
    jobs = [(i, sessions[i][0], sessions[i][2], beh_results[i]) for i in range(len(sessions))]
    results = [None] * len(sessions)
    t0 = time.time()
    nproc = max(1, min(args.nproc, len(jobs)))
    if nproc == 1:
        it = map(convert_session, jobs)
    else:
        pool = mp.Pool(nproc, maxtasksperchild=1)
        it = pool.imap_unordered(convert_session, jobs)
    done = 0
    for isess, neural, trials, region, info in it:
        results[isess] = (neural, trials, region, info)
        done += 1
        elapsed = time.time() - t0
        print('  [%3d/%3d] %-22s neurons %5d/%6d  trials %4d/%4d  frames %6d  %5.1fs '
              '(elapsed %.1f min, eta %.1f min)'
              % (done, len(jobs), '_'.join(sessions[isess][0]), info['n_neurons_kept'],
                 info['n_neurons_recorded'], info['ntrials_kept'], info['ntrials_raw'],
                 sum(x.shape[1] for x in neural), info['seconds'],
                 elapsed / 60, elapsed / 60 / done * (len(jobs) - done)), flush=True)
    if nproc > 1:
        pool.close(); pool.join()
    t_neural = time.time() - t0
    print('neural pass: %.1f min (%.1fs/session wall clock)'
          % (t_neural / 60, t_neural / len(sessions)), flush=True)

    # ---------------- assemble
    print('\n=== assembling output ===', flush=True)
    subjects = sorted({k[0] for k, _, _, _ in sessions})
    data = dict(neural=[], input=[], output=[], subjects=subjects, subject_idx=[],
                brain_regions=list(BRAIN_REGIONS), brain_region_idx=[],
                input_names=['time_to_sound_cue', 'day_of_training',
                             'time_since_trial_start', 'reward_available'],
                output_names=['stimulus_category', 'licking', 'position_bin', 'speed_bin'],
                output_values=[list(STIM_CATEGORIES), ['no lick', 'lick'],
                               list(POSITION_VALUES), list(SPEED_VALUES)])
    session_info = []
    for isess, (key, exp_type, db, ets) in enumerate(sessions):
        neural, trials, region, info = results[isess]
        if len(trials) < 2:
            print('  skipping %s: only %d usable trials' % ('_'.join(key), len(trials)))
            continue
        inputs, outputs = [], []
        for tr in trials:
            T = len(tr['frames'])
            inp = np.empty((4, T), dtype=np.float32)
            inp[0] = tr['time_to_cue']
            inp[1] = tr['day_of_training']
            inp[2] = tr['time_since_start']
            inp[3] = tr['reward_available']
            out = np.empty((4, T), dtype=np.int64)
            out[0] = tr['stim_cat']
            out[1] = tr['lick']
            out[2] = np.clip(np.floor(tr['pos'] / (TEXTURE_LENGTH_DM / N_POS_BINS)),
                             0, N_POS_BINS - 1).astype(np.int64)
            out[3] = np.digitize(tr['speed'], speed_edges).astype(np.int64)
            inputs.append(inp)
            outputs.append(out)
        data['neural'].append(neural)
        data['input'].append(inputs)
        data['output'].append(outputs)
        data['brain_region_idx'].append(np.asarray(region, dtype=np.int64))
        data['subject_idx'].append(subjects.index(key[0]))
        session_info.append(dict(session_id='_'.join(key), mouse=key[0], date=key[1], blk=key[2],
                                 exp_type=exp_type, exp_types=ets, cohort=cohort_of(db, ets),
                                 day_of_training=beh_results[isess]['day_of_training'],
                                 reward_session=bool(info['n_rew_trials'] > 0),
                                 reward_mode=str(db.get('rewType', '')),
                                 **{k: info[k] for k in ('n_neurons_recorded', 'n_neurons_labelled',
                                                         'n_neurons_kept', 'n_frames_spk',
                                                         'ntrials_raw', 'ntrials_kept',
                                                         'n_rew_trials', 'dt')}))
    data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)

    dts = np.array([s['dt'] for s in session_info])
    data['metadata'] = dict(
        task_description=(
            'Head-fixed mice run through 4 m virtual-reality corridors whose walls show one of '
            'four naturalistic textures (circle, leaf, rock, brick). In the task (supervised) '
            'cohort one texture is rewarded: a sound cue at a random position (0.5-3.5 m) '
            'signals reward availability, and licking after the cue in the rewarded corridor '
            'delivers water. Unsupervised and naive cohorts run the same corridors without '
            'reward. Neural data are suite2p deconvolved two-photon mesoscope traces from '
            'visual cortical areas (V1, medial, lateral and anterior higher visual areas). '
            'The decoder predicts, from neural activity plus task context (time to the sound '
            'cue, training day, time since corridor entry, reward availability): the visual '
            'stimulus category, whether the mouse is licking, its position in the corridor '
            '(4 x 1 m bins) and its running speed (global quartile bins).'),
        time_bin_size=float(np.median(dts) * 1000.0),
        temporal_alignment_event='trial start = entry into the virtual-reality corridor (beh["StartFr"])',
        off_start=0.0,
        off_end=None,
        trial_definition=('frames from corridor entry to corridor exit (4 m texture area) at which '
                          'the virtual reality was moving, i.e. the mouse was running faster than '
                          'the 6 cm/s threshold ((ft_move>0) & ft_CorrSpc, the reference mask); '
                          'trials have variable length (median ~21 frames ~ 6.6 s)'),
        frame_rate_hz=float(1.0 / np.median(dts)),
        neural_signal='suite2p deconvolved fluorescence (non-negative deconvolution, tau=0.75 s)',
        neuron_selection=('neurons with a retinotopic area label (V1/mHV/lHV/aHV via '
                          'utils.neu_area_ID); subsampled to at most %d per session, stratified '
                          'proportionally by area (seed %d)' % (MAX_NEURONS, SEED)),
        speed_bin_edges_cm_s=[float(x) for x in speed_edges],
        position_bin_edges_m=[0.0, 1.0, 2.0, 3.0, 4.0],
        input_units=['s (positive before the cue)', 'days since the mouse\'s first imaging session',
                     's since corridor entry', 'binary (1 = rewarded corridor)'],
        source_paper='Zhong, Li et al., Unsupervised pretraining in biological neural networks (Nature, 2025)',
        source_data='/app/data (Figshare doi:10.25378/janelia.28811129.v1)',
        session_info=session_info,
    )

    # ---------------- sanity checks
    print('\n=== sanity checks ===')
    nsess = len(data['neural'])
    ntrials = sum(len(s) for s in data['neural'])
    nframes = sum(x.shape[1] for s in data['neural'] for x in s)
    print('sessions %d, subjects %d, trials %d, timepoints %d'
          % (nsess, len(subjects), ntrials, nframes))
    assert len(data['input']) == nsess and len(data['output']) == nsess
    for i in range(nsess):
        assert len(data['input'][i]) == len(data['neural'][i]) == len(data['output'][i])
        nn = data['neural'][i][0].shape[0]
        assert len(data['brain_region_idx'][i]) == nn
        for tr in range(len(data['neural'][i])):
            T = data['neural'][i][tr].shape[1]
            assert data['neural'][i][tr].shape[0] == nn
            assert data['input'][i][tr].shape == (4, T)
            assert data['output'][i][tr].shape == (4, T)
    raw_trials = sum(s['ntrials_raw'] for s in session_info)
    print('trials: %d kept of %d recorded (%.2f%% dropped)'
          % (ntrials, raw_trials, 100 * (1 - ntrials / raw_trials)))
    nn_rec = np.array([s['n_neurons_recorded'] for s in session_info])
    print('neurons recorded/session: min %d median %d max %d (paper: 20,547-89,577)'
          % (nn_rec.min(), np.median(nn_rec), nn_rec.max()))
    fr = np.array([x.shape[1] for s in data['neural'] for x in s])
    print('frames/trial: median %.0f, 1st-99th pct %.0f-%.0f (expect ~21, 15-37)'
          % (np.median(fr), np.percentile(fr, 1), np.percentile(fr, 99)))
    allout = np.concatenate([x for s in data['output'] for x in s], axis=1)
    for d_ in range(4):
        vals, cnt = np.unique(allout[d_], return_counts=True)
        print('output %d (%-18s): %s'
              % (d_, data['output_names'][d_],
                 {data['output_values'][d_][int(v)]: round(c / cnt.sum(), 4)
                  for v, c in zip(vals, cnt)}))
    allin = np.concatenate([x for s in data['input'] for x in s], axis=1)
    for d_ in range(4):
        print('input  %d (%-22s): min %.3f median %.3f max %.3f'
              % (d_, data['input_names'][d_], allin[d_].min(),
                 np.median(allin[d_]), allin[d_].max()))
    assert allin[2].min() >= 0, 'time since trial start must be >= 0'

    # ---------------- plots
    if args.show_processing:
        print('\n=== processing plots ===')
        for i in range(min(2, nsess)):
            trials = results[i][1]
            for t, tr in enumerate(trials):
                tr['neural'] = data['neural'][i][t]
            plot_processing(session_info[i]['session_id'], data['neural'][i], trials,
                            speed_edges, results[i][3],
                            'processing_%s.png' % session_info[i]['session_id'])

    # ---------------- save
    print('\n=== saving ===', flush=True)
    t0 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print('wrote %s (%.2f GB) in %.1fs'
          % (args.outfile, os.path.getsize(args.outfile) / 1e9, time.time() - t0))
    print('total time %.1f min' % ((time.time() - t_start_all) / 60))


if __name__ == '__main__':
    main()
