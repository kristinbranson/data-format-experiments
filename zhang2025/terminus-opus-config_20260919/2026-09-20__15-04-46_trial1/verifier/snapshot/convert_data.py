#!/usr/bin/env python
"""Convert the IBL Brain-Wide Map release into the decoder pickle format.

Processing follows /app/code/code_zhang2025/src/0_data_caching.py and
/app/code/code_zhang2025/src/utils/ibl_data_utils.py:

  * align to stimOn_times, window (-0.5, +1.5) s, 20 ms bins -> T = 100
  * merge all probes of a session
  * trial mask: reference load_trials_and_mask(..., max_trial_len=10.0)
  * wheel speed  = abs(SessionLoader.wheel['velocity'])
    whisker ME   = leftCamera (fallback rightCamera) whiskerMotionEnergy
    both linearly interpolated onto the right edge of every 20 ms bin

Deviations (documented in CONVERSION_NOTES.md):
  * only well-isolated units (clusters['label'] >= 1) in grey-matter Beryl regions
    with >= 5 units in the session and present in >= 2 sessions, matching the
    curation the BWM data paper applies to all of its analyses.
  * wheel speed / whisker ME discretised into 3 per-session terciles, because the
    decoder requires categorical outputs.

Usage: python -u /app/convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""
import argparse
import os
import pickle
import time
import traceback
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

CACHE_DIR = '/app/data/one_cache'
FREEZE_FILE = '/app/code/code_zhang2025/data/bwm_release.csv'

# ---- parameters, copied from 0_data_caching.py -----------------------------
PARAMS = {'interval_len': 2.0, 'binsize': 0.02,
          'align_time': 'stimOn_times', 'time_window': (-0.5, 1.5)}
BINSIZE = PARAMS['binsize']
WIN = PARAMS['time_window']
NBINS = int(np.ceil((WIN[1] - WIN[0]) / BINSIZE))   # 100
# right edge of each bin relative to the alignment event; matches the reference
# get_behavior_per_interval grid np.linspace(t_beg + binsize, t_end, n_bins)
BIN_TIMES = np.linspace(WIN[0] + BINSIZE, WIN[1], NBINS)

MIN_RT, MAX_RT, MAX_TRIAL_LEN = 0.08, 2.0, 10.0
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']
MIN_UNITS_PER_REGION = 5      # data paper: >= 5 well-isolated neurons per session
MIN_SESSIONS_PER_REGION = 2   # data paper: recorded in >= 2 such sessions
MIN_TRIALS = 2                # format requirement
EXCLUDE_REGIONS = ('root', 'void')

INPUT_NAMES = ['time_from_stimulus_onset', 'trial_number_in_block']
OUTPUT_NAMES = ['choice', 'prior_probability_left', 'wheel_speed',
                'whisker_motion_energy']
OUTPUT_VALUES = [['left', 'right'],
                 ['0.2', '0.5', '0.8'],
                 ['low', 'medium', 'high'],
                 ['low', 'medium', 'high']]

_ONE = None
_BR = None


def _get_one():
    global _ONE, _BR
    if _ONE is None:
        from one.api import ONE
        from iblatlas.regions import BrainRegions
        # no password: the staged auth token is used and ONE stays offline
        _ONE = ONE(base_url='https://openalyx.internationalbrainlab.org',
                   silent=True, cache_dir=CACHE_DIR)
        _BR = BrainRegions()
    return _ONE, _BR


# ---------------------------------------------------------------- trials ----
def load_trials_and_mask(sess_loader):
    """Reference load_trials_and_mask with max_trial_len=10.0."""
    if sess_loader.trials.empty:
        sess_loader.load_trials()
    query = '(firstMovement_times - stimOn_times < %s)' % MIN_RT
    query += ' | (firstMovement_times - stimOn_times > %s)' % MAX_RT
    query += ' | (feedback_times - goCue_times > %s)' % MAX_TRIAL_LEN
    for event in NAN_EXCLUDE:
        query += ' | %s.isnull()' % event
    query += ' | (choice == 0)'
    mask = ~sess_loader.trials.eval(query)
    return sess_loader.trials, mask.to_numpy()


def trial_number_in_block(prob_left):
    """0-based index of each trial within its block of constant probabilityLeft.

    Computed on the full trials table so the value reflects the true position in
    the block even when intervening trials are excluded by the trial mask.
    """
    p = np.asarray(prob_left, dtype=float)
    new_block = np.ones(len(p), dtype=bool)
    new_block[1:] = p[1:] != p[:-1]
    starts = np.maximum.accumulate(np.where(new_block, np.arange(len(p)), 0))
    return np.arange(len(p)) - starts


# ---------------------------------------------------------------- spikes ----
def load_session_spikes(one, br, eid, pids, pnames):
    """Load and merge all probes, keeping well-isolated grey-matter units."""
    from brainbox.io.one import SpikeSortingLoader

    times_l, clu_l, reg_l = [], [], []
    offset = 0
    n_all = 0
    n_good = 0
    for pid, pname in zip(pids, pnames):
        ssl = SpikeSortingLoader(pid=str(pid), one=one, eid=eid, pname=pname)
        spikes, clusters, channels = ssl.load_spike_sorting()
        if spikes is None or len(spikes) == 0:
            continue
        cl = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
        n_all += len(cl)
        good = cl['label'].to_numpy(dtype=float) >= 1.0
        n_good += int(good.sum())
        beryl = np.asarray(br.acronym2acronym(cl['acronym'].to_numpy(),
                                              mapping='Beryl'))
        keep = good & ~np.isin(beryl, EXCLUDE_REGIONS)
        if keep.sum() == 0:
            continue
        cluster_ids = cl.index.to_numpy()
        kept_ids = cluster_ids[keep]
        lut = np.full(int(cluster_ids.max()) + 2, -1, dtype=np.int64)
        lut[kept_ids] = np.arange(len(kept_ids)) + offset
        st = np.asarray(spikes['times'])
        sc = np.asarray(spikes['clusters'])
        ok = np.isfinite(st) & (sc >= 0) & (sc < len(lut))
        st, sc = st[ok], sc[ok]
        new = lut[sc]
        sel = new >= 0
        times_l.append(st[sel])
        clu_l.append(new[sel])
        reg_l.append(beryl[keep])
        offset += len(kept_ids)

    if offset == 0:
        return None, None, None, n_all, n_good
    st = np.concatenate(times_l)
    sc = np.concatenate(clu_l)
    reg = np.concatenate(reg_l)
    order = np.argsort(st, kind='stable')
    return st[order], sc[order], reg, n_all, n_good


def bin_spikes(spike_times, spike_clusters, n_units, begs, ends):
    """Spike counts, shape (n_trials, n_units, NBINS).

    Identical to the reference bincount2D(xbin=binsize, xlim=[t_beg, t_end])
    followed by [:, :n_bins]: bin i covers [t_beg + i*bs, t_beg + (i+1)*bs).
    """
    n_trials = len(begs)
    out = np.zeros((n_trials, n_units, NBINS), dtype=np.float32)
    i0 = np.searchsorted(spike_times, begs, side='left')
    i1 = np.searchsorted(spike_times, ends, side='left')
    for k in range(n_trials):
        a, b = i0[k], i1[k]
        if b <= a:
            continue
        t = spike_times[a:b] - begs[k]
        bi = (t / BINSIZE).astype(np.int64)
        np.clip(bi, 0, NBINS - 1, out=bi)
        flat = spike_clusters[a:b] * NBINS + bi
        counts = np.bincount(flat, minlength=n_units * NBINS)
        out[k] = counts.reshape(n_units, NBINS)
    return out


# -------------------------------------------------------------- behaviour ----
def interpolate_behavior(target_times, target_vals, begs, ends):
    """Interpolate a behaviour trace onto the right edge of every bin of every trial.

    Mirrors get_behavior_per_interval: a trial is rejected if the trace has no
    samples inside the interval, starts more than one bin late, or ends more than
    one bin early.  Returns (values (n_trials, NBINS), good (n_trials,)).
    """
    from scipy.interpolate import interp1d

    n_trials = len(begs)
    good = np.ones(n_trials, dtype=bool)
    vals = np.full((n_trials, NBINS), np.nan)
    if target_times is None or target_vals is None or len(target_times) == 0:
        return vals, np.zeros(n_trials, dtype=bool)

    order = np.argsort(target_times)
    tt = np.asarray(target_times)[order]
    tv = np.asarray(target_vals, dtype=float)[order]
    finite = np.isfinite(tt) & np.isfinite(tv)
    tt, tv = tt[finite], tv[finite]
    if len(tt) < 2:
        return vals, np.zeros(n_trials, dtype=bool)

    good &= np.isfinite(begs) & np.isfinite(ends)
    ib = np.searchsorted(tt, np.nan_to_num(begs), side='right')
    ie = np.searchsorted(tt, np.nan_to_num(ends), side='left')
    good &= ie > ib
    idx = np.where(good)[0]
    if len(idx) == 0:
        return vals, good
    first_t = tt[np.clip(ib[idx], 0, len(tt) - 1)]
    last_t = tt[np.clip(ie[idx] - 1, 0, len(tt) - 1)]
    ok = ((np.abs(begs[idx] - first_t) <= BINSIZE) &
          (np.abs(ends[idx] - last_t) <= BINSIZE))
    good[idx[~ok]] = False

    f = interp1d(tt, tv, kind='linear', fill_value='extrapolate',
                 bounds_error=False, assume_sorted=True)
    grid = np.nan_to_num(begs)[:, None] + (BIN_TIMES - WIN[0])[None, :]
    vals = f(grid)
    good &= np.all(np.isfinite(vals), axis=1)
    return vals, good


def discretize_terciles(values):
    """Split a (n_trials, NBINS) array into 3 equally-populated per-session bins."""
    q1, q2 = np.nanquantile(values, [1.0 / 3.0, 2.0 / 3.0])
    if not np.isfinite(q1) or not np.isfinite(q2) or q1 == q2:
        flat = values.ravel()
        ranks = np.argsort(np.argsort(flat))
        d = (ranks * 3 // max(len(flat), 1)).reshape(values.shape)
        return np.clip(d, 0, 2).astype(np.int64), (float('nan'), float('nan'))
    return np.digitize(values, [q1, q2]).astype(np.int64), (float(q1), float(q2))


# ----------------------------------------------------------- one session ----
def process_session(job):
    eid, pids, pnames, subject = job
    t_start = time.time()
    info = {'eid': eid, 'subject': subject, 'n_probes': len(pids)}
    try:
        one, br = _get_one()
        from brainbox.io.one import SessionLoader

        # --- trials -------------------------------------------------------
        sl = SessionLoader(one=one, eid=eid)
        sl.load_trials()
        trials, mask = load_trials_and_mask(sl)
        info['n_trials_raw'] = int(len(trials))
        info['n_trials_mask'] = int(mask.sum())
        if mask.sum() < MIN_TRIALS:
            return None, dict(info, skip='too few trials after mask')

        tib_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
        stim_on = trials[PARAMS['align_time']].to_numpy(dtype=float)
        begs_all = stim_on + WIN[0]
        ends_all = stim_on + WIN[1]

        # --- behaviour ----------------------------------------------------
        sl.load_wheel()
        wheel_t = sl.wheel['times'].to_numpy()
        wheel_v = np.abs(sl.wheel['velocity'].to_numpy())

        me_t = me_v = me_view = None
        for view in ('left', 'right'):
            try:
                sl.load_motion_energy(views=[view])
                key = view + 'Camera'
                if key in sl.motion_energy and \
                        'whiskerMotionEnergy' in sl.motion_energy[key]:
                    me_t = sl.motion_energy[key]['times'].to_numpy()
                    me_v = sl.motion_energy[key]['whiskerMotionEnergy'].to_numpy()
                    me_view = view
                    break
            except Exception:
                continue
        if me_t is None:
            return None, dict(info, skip='no whisker motion energy')
        info['motion_energy_view'] = me_view

        wheel_vals, wheel_ok = interpolate_behavior(wheel_t, wheel_v,
                                                    begs_all, ends_all)
        me_vals, me_ok = interpolate_behavior(me_t, me_v, begs_all, ends_all)
        valid = mask & wheel_ok & me_ok
        info['n_trials_behavior'] = int(valid.sum())
        if valid.sum() < MIN_TRIALS:
            return None, dict(info, skip='too few trials after behaviour mask')

        # prior must be one of the three block values
        pl = trials['probabilityLeft'].to_numpy()
        valid &= (np.isclose(pl, 0.2) | np.isclose(pl, 0.5) | np.isclose(pl, 0.8))
        if valid.sum() < MIN_TRIALS:
            return None, dict(info, skip='too few trials after prior check')

        # --- spikes -------------------------------------------------------
        st, sc, reg, n_all, n_good = load_session_spikes(one, br, eid, pids, pnames)
        info['n_clusters_all'] = n_all
        info['n_clusters_good'] = n_good
        if st is None:
            return None, dict(info, skip='no well-isolated grey-matter units')

        # >= MIN_UNITS_PER_REGION well-isolated units per region in this session
        counts = Counter(reg.tolist())
        keep_unit = np.array([counts[r] >= MIN_UNITS_PER_REGION for r in reg])
        if keep_unit.sum() == 0:
            return None, dict(info, skip='no region with >=5 units')
        remap = np.full(len(reg), -1, dtype=np.int64)
        remap[keep_unit] = np.arange(int(keep_unit.sum()))
        sel = remap[sc] >= 0
        st, sc = st[sel], remap[sc[sel]]
        reg = reg[keep_unit]
        n_units = len(reg)
        info['n_units'] = n_units

        # Neural coverage: the analogue of the reference behaviour-coverage check.
        # A trial is only usable if its window lies inside the span of the spike
        # sorting and the population actually fires in it.  Two real failure modes
        # in the release: (a) the ephys recording stops before the behaviour does,
        # so the last trials have no spikes at all (e.g. eid 8c2f7f4d..., recording
        # ends at 1779 s while 3 kept trials start at 1789-1799 s); (b) a multi-second
        # dropout inside the recording (e.g. eid b182b754..., a 2.07 s gap at 185.6 s
        # swallowing one trial).  Both give an all-zero neural matrix, which carries
        # no information and triggers decoder warnings.
        in_span = (begs_all >= st[0]) & (ends_all <= st[-1])
        valid &= in_span
        idx = np.where(valid)[0]
        if len(idx) < MIN_TRIALS:
            return None, dict(info, skip='too few trials inside the ephys recording')
        binned = bin_spikes(st, sc, n_units, begs_all[idx], ends_all[idx])
        has_spikes = binned.sum(axis=(1, 2)) > 0
        if not has_spikes.all():
            info['n_trials_no_spikes'] = int((~has_spikes).sum())
            idx = idx[has_spikes]
            binned = binned[has_spikes]
            valid[:] = False
            valid[idx] = True
        if len(idx) < MIN_TRIALS:
            return None, dict(info, skip='too few trials with spikes')

        # --- outputs ------------------------------------------------------
        choice_raw = trials['choice'].to_numpy()[idx]
        # IBL convention (verified empirically): +1 = reported LEFT, -1 = RIGHT
        choice = (choice_raw < 0).astype(np.int64)
        pleft = pl[idx]
        prior = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                           np.isclose(pleft, 0.8)], [0, 1, 2]).astype(np.int64)

        wheel_k = wheel_vals[idx]
        me_k = me_vals[idx]
        wheel_d, wheel_thr = discretize_terciles(wheel_k)
        me_d, me_thr = discretize_terciles(me_k)
        info['wheel_thresholds'] = wheel_thr
        info['whisker_thresholds'] = me_thr

        tib = tib_all[idx].astype(np.float32)
        n_trials = len(idx)
        time_row = BIN_TIMES.astype(np.float32)

        neural, inputs, outputs = [], [], []
        for k in range(n_trials):
            neural.append(np.ascontiguousarray(binned[k]))
            inputs.append(np.stack([time_row,
                                    np.full(NBINS, tib[k], dtype=np.float32)]))
            outputs.append(np.stack([
                np.full(NBINS, choice[k], dtype=np.int64),
                np.full(NBINS, prior[k], dtype=np.int64),
                wheel_d[k], me_d[k]]))

        session = {'eid': eid, 'subject': subject, 'neural': neural,
                   'input': inputs, 'output': outputs, 'regions': reg,
                   'trial_idx': idx,
                   'raw': {'wheel_vals': wheel_k, 'me_vals': me_k,
                           'stim_on': stim_on[idx]}}
        info['n_trials_final'] = n_trials
        info['time_s'] = time.time() - t_start
        return session, info
    except Exception as exc:
        info['skip'] = 'error: %s' % exc
        info['traceback'] = traceback.format_exc()
        return None, info


# ------------------------------------------------------------- plotting ----
def plot_processing(session, info, wheel_raw, me_raw):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    eid = session['eid']
    fig, axes = plt.subplots(4, 2, figsize=(16, 15))
    ntr = len(session['neural'])
    k = min(3, ntr - 1)
    t0 = session['raw']['stim_on'][k]

    ax = axes[0, 0]
    nm = session['neural'][k]
    ax.imshow(nm, aspect='auto', interpolation='nearest',
              extent=[BIN_TIMES[0], BIN_TIMES[-1], nm.shape[0], 0], cmap='Greys')
    ax.axvline(0, color='r')
    ax.set_title('%s trial %d binned spike counts (%d units x %d bins)'
                 % (eid[:8], k, nm.shape[0], nm.shape[1]))
    ax.set_xlabel('time from stimulus onset (s)')
    ax.set_ylabel('unit')

    ax = axes[0, 1]
    psth = np.mean(np.stack(session['neural']), axis=0).mean(axis=0) / BINSIZE
    ax.plot(BIN_TIMES, psth)
    ax.axvline(0, color='r', label='stimulus onset')
    ax.set_title('population PSTH (mean firing rate, Hz) -- should step up at t=0')
    ax.set_xlabel('time from stimulus onset (s)')
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    m = (wheel_raw[0] >= t0 + WIN[0] - 0.1) & (wheel_raw[0] <= t0 + WIN[1] + 0.1)
    ax.plot(wheel_raw[0][m] - t0, wheel_raw[1][m], 'k-', lw=0.8,
            label='raw abs(wheel velocity)')
    ax.plot(BIN_TIMES, session['raw']['wheel_vals'][k], 'o-', ms=3, color='C0',
            label='interpolated onto bin right edges')
    ax.axvline(0, color='r')
    ax.set_title('trial %d: wheel speed raw vs interpolated' % k)
    ax.set_xlabel('time from stimulus onset (s)')
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.plot(BIN_TIMES, session['raw']['wheel_vals'][k], 'o-', ms=3, color='C0')
    for q in info.get('wheel_thresholds', ()):
        ax.axhline(q, color='g', ls='--')
    ax2 = ax.twinx()
    ax2.step(BIN_TIMES, session['output'][k][2], color='m', where='mid')
    ax2.set_ylabel('discretized bin', color='m')
    ax.set_title('wheel speed discretization, terciles %s'
                 % (info.get('wheel_thresholds'),))

    ax = axes[2, 0]
    m = (me_raw[0] >= t0 + WIN[0] - 0.1) & (me_raw[0] <= t0 + WIN[1] + 0.1)
    ax.plot(me_raw[0][m] - t0, me_raw[1][m], 'k.-', lw=0.8, ms=3,
            label='raw whisker motion energy')
    ax.plot(BIN_TIMES, session['raw']['me_vals'][k], 'o-', ms=3, color='C1',
            label='interpolated onto bin right edges')
    ax.axvline(0, color='r')
    ax.set_title('trial %d: whisker ME raw vs interpolated' % k)
    ax.set_xlabel('time from stimulus onset (s)')
    ax.legend(fontsize=7)

    ax = axes[2, 1]
    ax.plot(BIN_TIMES, session['raw']['me_vals'][k], 'o-', ms=3, color='C1')
    for q in info.get('whisker_thresholds', ()):
        ax.axhline(q, color='g', ls='--')
    ax2 = ax.twinx()
    ax2.step(BIN_TIMES, session['output'][k][3], color='m', where='mid')
    ax2.set_ylabel('discretized bin', color='m')
    ax.set_title('whisker ME discretization, terciles %s'
                 % (info.get('whisker_thresholds'),))

    ax = axes[3, 0]
    ch = np.array([o[0, 0] for o in session['output']])
    pr = np.array([o[1, 0] for o in session['output']])
    ax.plot(pr, '.-', label='prior class (0=0.2, 1=0.5, 2=0.8)')
    ax.plot(ch, '.', alpha=0.5, label='choice (0=left, 1=right)')
    ax.set_xlabel('kept trial')
    ax.legend(fontsize=7)
    ax.set_title('per-trial outputs')

    ax = axes[3, 1]
    tib = np.array([i[1, 0] for i in session['input']])
    ax.plot(tib, '.-')
    ax.set_xlabel('kept trial')
    ax.set_ylabel('trial number in block')
    ax.set_title('input 1: trial number in block (resets at every block change)')

    fig.suptitle('Processing checks for session %s' % eid)
    fig.tight_layout()
    out = '/app/processing_%s.png' % eid
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print('  wrote %s' % out)


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--n-workers', type=int, default=24)
    args = ap.parse_args()

    t_all = time.time()
    bwm = pd.read_csv(FREEZE_FILE, index_col=0)
    print('freeze file: %d insertions, %d eids, %d subjects, %d labs'
          % (len(bwm), bwm.eid.nunique(), bwm.subject.nunique(), bwm.lab.nunique()))

    limit_csv = '/app/data/DATALIMIT_SUBSET.csv'
    if os.path.exists(limit_csv):
        sub = pd.read_csv(limit_csv)
        col = 'eid' if 'eid' in sub.columns else sub.columns[0]
        keep = set(sub[col].astype(str))
        bwm = bwm[bwm.eid.astype(str).isin(keep)]
        print('DATALIMIT_SUBSET.csv present -> restricted to %d eids'
              % bwm.eid.nunique())

    jobs = []
    for eid, g in bwm.groupby('eid', sort=False):
        jobs.append((eid, list(g.pid), list(g.probe_name), g.subject.iloc[0]))
    jobs.sort(key=lambda j: j[0])
    if args.sample:
        jobs = jobs[:2]
    print('processing %d sessions with %d workers' % (len(jobs), args.n_workers))

    results = []
    serial = args.show_processing or len(jobs) <= 2
    if serial:
        for job in jobs:
            print('-- %s' % job[0], flush=True)
            res = process_session(job)
            results.append(res)
            if res[0] is None:
                print('   SKIP: %s' % res[1].get('skip'))
            else:
                print('   units=%d trials=%d (%.1f s)'
                      % (res[1]['n_units'], res[1]['n_trials_final'],
                         res[1]['time_s']))
                if args.show_processing:
                    one, _ = _get_one()
                    from brainbox.io.one import SessionLoader
                    sl = SessionLoader(one=one, eid=job[0])
                    sl.load_wheel()
                    wraw = (sl.wheel['times'].to_numpy(),
                            np.abs(sl.wheel['velocity'].to_numpy()))
                    view = res[1].get('motion_energy_view', 'left')
                    sl.load_motion_energy(views=[view])
                    cam = sl.motion_energy[view + 'Camera']
                    mraw = (cam['times'].to_numpy(),
                            cam['whiskerMotionEnergy'].to_numpy())
                    plot_processing(res[0], res[1], wraw, mraw)
    else:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=args.n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(process_session, jobs)):
                results.append(res)
                info = res[1]
                el = time.time() - t_all
                eta = el / (i + 1) * (len(jobs) - i - 1) / 60.0
                if res[0] is None:
                    status = 'SKIP: %s' % info.get('skip')
                else:
                    status = 'units=%d trials=%d' % (info['n_units'],
                                                     info['n_trials_final'])
                print('[%d/%d] %s %s | elapsed %.1f min, eta %.1f min'
                      % (i + 1, len(jobs), info['eid'], status, el / 60.0, eta),
                      flush=True)

    sessions = [r[0] for r in results if r[0] is not None]
    infos = [r[1] for r in results]
    skipped = [i for i in infos if 'skip' in i]
    print('')
    print('%d sessions converted, %d skipped' % (len(sessions), len(skipped)))
    for s in skipped:
        print('  SKIPPED %s: %s' % (s['eid'], s.get('skip')))

    # order sessions deterministically
    sessions.sort(key=lambda s: s['eid'])

    # ---- region filter: keep only regions recorded in >= 2 sessions ------
    region_session_count = Counter()
    for s in sessions:
        for r in set(s['regions'].tolist()):
            region_session_count[r] += 1
    min_sess = MIN_SESSIONS_PER_REGION if not args.sample else 1
    keep_regions = set(r for r, c in region_session_count.items()
                       if c >= min_sess)
    print('regions before >=%d-session filter: %d, after: %d'
          % (min_sess, len(region_session_count), len(keep_regions)))

    final = []
    for s in sessions:
        km = np.isin(s['regions'], list(keep_regions))
        if km.sum() == 0 or len(s['neural']) < MIN_TRIALS:
            print('  DROP %s: no units left after region filter' % s['eid'])
            continue
        if not km.all():
            s['neural'] = [np.ascontiguousarray(n[km]) for n in s['neural']]
            s['regions'] = s['regions'][km]
        final.append(s)
    sessions = final

    brain_regions = sorted(set(r for s in sessions for r in s['regions'].tolist()))
    reg_lut = dict((r, i) for i, r in enumerate(brain_regions))
    subjects = sorted(set(s['subject'] for s in sessions))
    subj_lut = dict((s, i) for i, s in enumerate(subjects))

    data = {
        'neural': [s['neural'] for s in sessions],
        'input': [s['input'] for s in sessions],
        'output': [s['output'] for s in sessions],
        'subjects': subjects,
        'subject_idx': np.array([subj_lut[s['subject']] for s in sessions],
                                dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': [np.array([reg_lut[r] for r in s['regions']],
                                      dtype=np.int64) for s in sessions],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description': (
                'IBL decision-making task. Head-fixed mice turn a wheel to move a '
                'Gabor stimulus (contrast 0, 6.25, 12.5, 25 or 100 percent) '
                'appearing on the left or right of a screen to the centre. The prior '
                'probability that the stimulus appears on the left is 0.5 for the '
                'first 90 trials and then alternates between 0.2 and 0.8 in blocks of '
                '20-100 trials. Decoder outputs: the mouse choice (left=0, right=1), '
                'the block prior probability of left (0.2->0, 0.5->1, 0.8->2), and '
                'wheel speed and whisker motion energy each discretised into 3 '
                'per-session terciles (low/medium/high). Decoder inputs: time since '
                'stimulus onset and the trial number within the current block.'),
            'time_bin_size': BINSIZE * 1000.0,
            'temporal_alignment_event': 'visual stimulus onset (trials.stimOn_times)',
            'off_start': WIN[0],
            'off_end': WIN[1],
            'n_timepoints': NBINS,
            'bin_times_s': BIN_TIMES.astype(np.float32),
            'bin_time_convention': ('input[0] and bin_times_s give the RIGHT edge of '
                                    'each 20 ms bin relative to stimulus onset'),
            'neural_units': 'spike counts per 20 ms bin',
            'atlas_mapping': 'Beryl (iblatlas.regions.BrainRegions.acronym2acronym)',
            'neuron_curation': ('well-isolated units only (clusters.label >= 1, i.e. '
                                'all three RIGOR single-unit metrics passed: amplitude '
                                '> 50 uV, noise cut-off < 20 uV, refractory-period '
                                'violation); grey-matter Beryl regions only (root and '
                                'void dropped); >= 5 units per region per session; '
                                'region present in >= 2 sessions'),
            'trial_curation': ('reference load_trials_and_mask: no NaN in stimOn_times, '
                               'choice, feedback_times, probabilityLeft, '
                               'firstMovement_times, feedbackType; 0.08 s <= '
                               'firstMovement_times - stimOn_times <= 2.0 s; '
                               'feedback_times - goCue_times <= 10 s; choice != 0; plus '
                               'wheel and whisker traces must cover the trial window; '
                               'and the trial window must lie inside the span of the '
                               'spike sorting and contain at least one spike'),
            'behaviour_sources': {
                'wheel_speed': ('abs(SessionLoader.wheel.velocity), 1 kHz, linearly '
                                'interpolated onto bin right edges'),
                'whisker_motion_energy': ('SessionLoader.motion_energy leftCamera '
                                          '(fallback rightCamera) whiskerMotionEnergy, '
                                          'linearly interpolated onto bin right edges'),
            },
            'discretization': ('wheel speed and whisker motion energy: per-session '
                               '33.3/66.7 percentiles over all (trial x timepoint) '
                               'samples'),
            'source': ('IBL Brain-Wide Map public release (bwm_release.csv freeze, '
                       '699 insertions / 459 eids / 139 subjects)'),
            'reference_code': '/app/code/code_zhang2025/src/0_data_caching.py',
            'session_info': [
                {'eid': s['eid'], 'subject': s['subject'],
                 'n_neurons': int(len(s['regions'])),
                 'n_trials': int(len(s['neural']))} for s in sessions],
        },
    }

    # ---- summary ---------------------------------------------------------
    n_neurons = [len(s['regions']) for s in sessions]
    n_trials = [len(s['neural']) for s in sessions]
    print('')
    print('SUMMARY')
    print('  sessions      : %d' % len(sessions))
    print('  subjects      : %d' % len(subjects))
    print('  brain regions : %d' % len(brain_regions))
    print('  neurons total : %d (mean %.1f/session)'
          % (sum(n_neurons), np.mean(n_neurons)))
    print('  trials total  : %d (mean %.1f/session)'
          % (sum(n_trials), np.mean(n_trials)))
    tot_all = sum(i.get('n_clusters_all', 0) for i in infos)
    tot_good = sum(i.get('n_clusters_good', 0) for i in infos)
    tot_probes = sum(i.get('n_probes', 0) for i in infos
                     if i.get('n_clusters_all', 0) > 0)
    print('  kilosort clusters loaded : %d (%.1f/probe over %d probes)'
          % (tot_all, tot_all / max(tot_probes, 1), tot_probes))
    print('  well-isolated (label>=1) : %d (%.1f/probe)'
          % (tot_good, tot_good / max(tot_probes, 1)))

    allout = np.concatenate([np.stack(s['output']) for s in sessions], axis=0)
    for d, nm in enumerate(OUTPUT_NAMES):
        v, c = np.unique(allout[:, d, :], return_counts=True)
        tot = c.sum()
        print('  output %-24s %s' % (nm, ', '.join(
            '%d:%.3f' % (int(a), b / tot) for a, b in zip(v, c))))
    allin = np.concatenate([np.stack(s['input']) for s in sessions], axis=0)
    for d, nm in enumerate(INPUT_NAMES):
        print('  input  %-24s [%.3f, %.3f]'
              % (nm, allin[:, d, :].min(), allin[:, d, :].max()))

    for s in sessions:
        s.pop('raw', None)

    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print('')
    print('wrote %s (%.2f GB) in %.1f min'
          % (args.outfile, os.path.getsize(args.outfile) / 1e9,
             (time.time() - t_all) / 60.0))


if __name__ == '__main__':
    main()
