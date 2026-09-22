#!/usr/bin/env python
"""Convert the IBL brain-wide map dataset into the decoder pickle format.

Usage
-----
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Processing follows Zhang et al.'s caching pipeline
(`/app/code/code_zhang2025/src/0_data_caching.py` and `src/utils/ibl_data_utils.py`):
probes merged per session, trials aligned to stimulus onset over (-0.5, +1.5) s, spikes
binned at 20 ms (T = 100), behaviour interpolated onto the same grid, and the reference
trial mask applied. Deviations required by the task specification are marked DEVIATION.
"""
import argparse
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

CACHE_DIR = '/app/data/one_cache'
TABLES_DIR = '/app/data/one_cache/LocalIndex'
FREEZE_FILE = '/app/code/code_zhang2025/data/bwm_release.csv'
DATALIMIT = '/app/data/DATALIMIT_SUBSET.csv'

# Reference params (src/0_data_caching.py)
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100

# Reference trial mask (utils/ibl_data_utils.py::load_trials_and_mask)
MIN_RT, MAX_RT, MAX_TRIAL_LEN = 0.08, 2.0, 10.0
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']

NON_GREY = ('root', 'void')

# The data paper restricts analyses to regions that "contained at least five well-isolated
# neurons per session". Applied here at the session level: a session must retain at least 5
# well-isolated grey-matter neurons to be usable. This also removes the degenerate 1-neuron
# sessions whose sparse trials are entirely empty.
MIN_UNITS_PER_SESSION = 5

INPUT_NAMES = ['time_from_stim_onset', 'stim_onset', 'trial_in_block']
OUTPUT_NAMES = ['choice', 'prior_prob_left', 'wheel_speed', 'whisker_motion_energy']
OUTPUT_VALUES = [['left', 'right'],
                 ['p_left_0.2', 'p_left_0.5', 'p_left_0.8'],
                 ['low', 'medium', 'high'],
                 ['low', 'medium', 'high']]


# --------------------------------------------------------------------------------------
# ONE helpers
# --------------------------------------------------------------------------------------
def get_one():
    """Build an offline ONE against the rebuilt local index.

    The shipped release tables are older than the staged files and do not index the
    revision folders the data actually lives in, which makes `load_trials()` silently
    return a single column. `build_one_cache.py` regenerates the index from disk.
    """
    from one.api import ONE
    if not os.path.exists(os.path.join(TABLES_DIR, 'datasets.pqt')):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache'))
        import build_one_cache
        build_one_cache.main()
    return ONE(base_url='https://openalyx.internationalbrainlab.org', mode='local',
               cache_dir=CACHE_DIR, tables_dir=TABLES_DIR)


def session_list():
    """Sessions to convert: the BWM release freeze, as used by the reference."""
    bwm = pd.read_csv(FREEZE_FILE, index_col=0)
    if os.path.exists(DATALIMIT):
        keep = pd.read_csv(DATALIMIT)
        col = 'eid' if 'eid' in keep.columns else keep.columns[0]
        bwm = bwm[bwm.eid.isin(set(keep[col].astype(str)))]
        print(f'DATALIMIT_SUBSET.csv present -> restricted to {bwm.eid.nunique()} sessions')
    return bwm


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_session_spikes(one, eid, probes):
    """Load and merge spike sorting across a session's probes.

    Mirrors `prepare_data` + `merge_probes`: cluster ids are offset per probe so they stay
    unique, and the clusters tables are concatenated in the same order.
    """
    from brainbox.io.one import SpikeSortingLoader
    times, clus, tables, offset = [], [], [], 0
    for _, r in probes.iterrows():
        ssl = SpikeSortingLoader(pid=r.pid, one=one, eid=eid, pname=r.probe_name)
        spikes, clusters, channels = ssl.load_spike_sorting()
        if len(spikes) == 0 or 'times' not in spikes:
            continue
        df = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
        times.append(spikes['times'])
        clus.append(spikes['clusters'] + offset)
        offset += int(df.index.max()) + 1
        tables.append(df)
    if not tables:
        raise RuntimeError('no spike data')
    return (np.concatenate(times), np.concatenate(clus),
            pd.concat(tables, ignore_index=True))


def good_grey_units(clusters):
    """Well-isolated (`label >= 1`) units in grey matter, with their Beryl acronyms.

    `label` is the fraction of the three RIGOR single-unit metrics passed (amplitude,
    noise cut-off, refractory-period violation); `label >= 1` means all three passed and
    reproduces the paper's 75,708 well-isolated neurons exactly.
    """
    from iblatlas.regions import BrainRegions
    beryl = np.asarray(BrainRegions().acronym2acronym(
        clusters['acronym'].to_numpy(), mapping='Beryl'), dtype=object)
    keep = (clusters['label'].to_numpy() >= 1) & ~np.isin(beryl, NON_GREY)
    return np.where(keep)[0], beryl


def trials_and_mask(one, eid):
    """Trials table plus the reference inclusion mask (load_trials_and_mask)."""
    from brainbox.io.one import SessionLoader
    sl = SessionLoader(one=one, eid=eid)   # kw-only in ibllib 4.0.1
    sl.load_trials()
    tr = sl.trials
    if 'stimOn_times' not in tr.columns:
        raise RuntimeError(f'trials table has no stimOn_times (cols={list(tr.columns)})')
    rt = tr['firstMovement_times'] - tr['stimOn_times']
    bad = (rt < MIN_RT) | (rt > MAX_RT)
    bad |= (tr['feedback_times'] - tr['goCue_times']) > MAX_TRIAL_LEN
    bad |= tr['choice'] == 0
    for col in NAN_EXCLUDE:
        bad |= tr[col].isna()
    return tr, (~bad).to_numpy(), sl


def load_behaviour(sl):
    """Wheel speed and whisker motion energy traces.

    Matches `load_target_behavior`: wheel speed is |velocity|; whisker ME prefers the left
    camera and falls back to the right (`bin_behaviors`).
    """
    out = {}
    try:
        sl.load_wheel()
        out['wheel_speed'] = (sl.wheel['times'].to_numpy(),
                              np.abs(sl.wheel['velocity'].to_numpy()))
    except Exception:
        out['wheel_speed'] = None
    me = None
    for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
        try:
            sl.load_motion_energy(views=[view])
            df = sl.motion_energy[cam]
            vals = df['whiskerMotionEnergy'].to_numpy()
            if np.isfinite(vals).any():
                me = (df['times'].to_numpy(), vals, view)
                break
        except Exception:
            continue
    out['whisker_motion_energy'] = me
    return out


# --------------------------------------------------------------------------------------
# Binning
# --------------------------------------------------------------------------------------
def bin_spikes(spike_times, spike_clusters, unit_idx, align_times):
    """Spike counts in (n_trials, n_units, N_BINS), aligned to `align_times`.

    Vectorised equivalent of `bin_spiking_data`/`get_spike_data_per_interval`: spikes are
    restricted to the selected units, sorted once, and each trial's window is sliced with
    searchsorted before a single bincount per trial.
    """
    remap = np.full(int(spike_clusters.max()) + 2, -1, dtype=np.int64)
    remap[unit_idx] = np.arange(len(unit_idx))
    sel = np.isin(spike_clusters, unit_idx)
    st, sc = spike_times[sel], remap[spike_clusters[sel]]
    order = np.argsort(st, kind='stable')
    st, sc = st[order], sc[order]

    begs = align_times + TIME_WINDOW[0]
    ends = align_times + TIME_WINDOW[1]
    i0 = np.searchsorted(st, begs, side='left')
    i1 = np.searchsorted(st, ends, side='left')

    n_units = len(unit_idx)
    out = np.zeros((len(align_times), n_units, N_BINS), dtype=np.float32)
    for k in range(len(align_times)):
        if i1[k] <= i0[k]:
            continue
        s = slice(i0[k], i1[k])
        tb = ((st[s] - begs[k]) / BINSIZE).astype(np.int64)
        np.clip(tb, 0, N_BINS - 1, out=tb)
        flat = np.bincount(sc[s] * N_BINS + tb, minlength=n_units * N_BINS)
        out[k] = flat.reshape(n_units, N_BINS)
    return out


def bin_behaviour(times, values, align_times):
    """Interpolate a behaviour trace onto the trial bin grid.

    Follows `get_behavior_per_interval`: evaluate a linear interpolant at
    `linspace(beg + binsize, end, n_bins)` (bin end times) and mark a trial invalid when
    the trace is missing or does not cover the window to within one bin.
    """
    n = len(align_times)
    out = np.full((n, N_BINS), np.nan)
    ok = np.zeros(n, dtype=bool)
    if times is None or len(times) == 0:
        return out, ok
    begs = align_times + TIME_WINDOW[0]
    ends = align_times + TIME_WINDOW[1]
    i0 = np.searchsorted(times, begs, side='right')
    i1 = np.searchsorted(times, ends, side='left')
    for k in range(n):
        if not np.isfinite(begs[k]) or not np.isfinite(ends[k]):
            continue
        if i1[k] - i0[k] < 2:
            continue
        t = times[i0[k]:i1[k]]
        v = values[i0[k]:i1[k]]
        if abs(begs[k] - t[0]) > BINSIZE or abs(ends[k] - t[-1]) > BINSIZE:
            continue
        finite = np.isfinite(v)
        if finite.sum() < 2:
            continue
        grid = np.linspace(begs[k] + BINSIZE, ends[k], N_BINS)
        out[k] = interp1d(t[finite], v[finite], kind='linear',
                          fill_value='extrapolate')(grid)
        ok[k] = True
    return out, ok


def tertile_bins(values):
    """Discretise into 3 balanced classes using per-session tertiles.

    DEVIATION (required): the reference treats wheel speed and whisker ME as continuous
    regression targets, but the task requires categorical outputs. Tertiles are computed
    per session because these signals are in session-specific units (camera gain / ROI,
    wheel rig), so a global threshold would largely encode session identity. The reference
    likewise rescales behaviour per session before decoding.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.int16), (np.nan, np.nan)
    lo, hi = np.percentile(finite, [100 / 3, 200 / 3])
    if not (hi > lo):  # degenerate (e.g. a near-constant trace)
        hi = lo + 1e-12
    return (np.digitize(values, [lo, hi]).astype(np.int16), (float(lo), float(hi)))


def trial_in_block(prob_left):
    """Index of each trial within its probabilityLeft block (0-based).

    Computed over the full trial sequence before masking, so that removing trials does not
    corrupt the count.
    """
    pl = np.asarray(prob_left, dtype=float)
    new = np.ones(len(pl), dtype=bool)
    new[1:] = pl[1:] != pl[:-1]
    idx = np.arange(len(pl))
    return idx - np.maximum.accumulate(np.where(new, idx, 0))


# --------------------------------------------------------------------------------------
# Per-session conversion
# --------------------------------------------------------------------------------------
def convert_session(eid, probes, show=False, outdir='/app'):
    t0 = time.time()
    one = get_one()
    timings = {}

    t = time.time()
    spike_times, spike_clusters, clusters = load_session_spikes(one, eid, probes)
    timings['load_spikes'] = time.time() - t

    unit_idx, beryl = good_grey_units(clusters)
    if len(unit_idx) < MIN_UNITS_PER_SESSION:
        raise RuntimeError(f'only {len(unit_idx)} well-isolated grey-matter units '
                           f'(minimum {MIN_UNITS_PER_SESSION})')

    t = time.time()
    trials, mask, sl = trials_and_mask(one, eid)
    timings['load_trials'] = time.time() - t

    t = time.time()
    beh = load_behaviour(sl)
    timings['load_behaviour'] = time.time() - t
    if beh['whisker_motion_energy'] is None:
        raise RuntimeError('no whisker motion energy (left or right)')
    if beh['wheel_speed'] is None:
        raise RuntimeError('no wheel data')

    tib_all = trial_in_block(trials['probabilityLeft'].to_numpy())
    keep = np.where(mask)[0]
    align = trials[ALIGN_TIME].to_numpy()[keep]
    finite = np.isfinite(align)
    keep, align = keep[finite], align[finite]

    t = time.time()
    wt, wv = beh['wheel_speed']
    wheel, ok_w = bin_behaviour(wt, wv, align)
    mt, mv, me_side = beh['whisker_motion_energy']
    whisk, ok_m = bin_behaviour(mt, mv, align)
    timings['bin_behaviour'] = time.time() - t

    valid = ok_w & ok_m
    if valid.sum() < 2:
        raise RuntimeError(f'only {valid.sum()} trials with complete behaviour')
    keep, align = keep[valid], align[valid]
    wheel, whisk = wheel[valid], whisk[valid]

    t = time.time()
    counts = bin_spikes(spike_times, spike_clusters, unit_idx, align)
    timings['bin_spikes'] = time.time() - t

    # ---- outputs -------------------------------------------------------------------
    choice = trials['choice'].to_numpy()[keep]
    y_choice = np.where(choice > 0, 0, 1).astype(np.int16)   # +1 left -> 0, -1 right -> 1
    pl = trials['probabilityLeft'].to_numpy()[keep]
    y_prior = np.select([np.isclose(pl, 0.2), np.isclose(pl, 0.5), np.isclose(pl, 0.8)],
                        [0, 1, 2], default=-1).astype(np.int16)
    if (y_prior < 0).any():
        okp = y_prior >= 0
        keep, align, counts, wheel, whisk = (keep[okp], align[okp], counts[okp],
                                             wheel[okp], whisk[okp])
        y_choice, y_prior = y_choice[okp], y_prior[okp]
    y_wheel, thr_w = tertile_bins(wheel)
    y_whisk, thr_m = tertile_bins(whisk)

    n_tr = len(keep)
    outputs = np.empty((n_tr, 4, N_BINS), dtype=np.int16)
    outputs[:, 0, :] = y_choice[:, None]
    outputs[:, 1, :] = y_prior[:, None]
    outputs[:, 2, :] = y_wheel
    outputs[:, 3, :] = y_whisk

    # ---- inputs --------------------------------------------------------------------
    # bin end times, matching the behaviour grid; bin i covers (t_i - binsize, t_i]
    tgrid = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
    # bin_spikes uses tb = floor((t - beg)/binsize), so bin i spans
    # [off_start + i*binsize, off_start + (i+1)*binsize) and a spike exactly at stimulus
    # onset lands in bin round(-off_start/binsize) (= 25 here, the first bin at t >= 0).
    onset = np.zeros(N_BINS, dtype=np.float32)
    onset_bin = int(round(-TIME_WINDOW[0] / BINSIZE))
    onset[onset_bin] = 1.0
    tib = tib_all[keep].astype(np.float32)
    inputs = np.empty((n_tr, 3, N_BINS), dtype=np.float32)
    inputs[:, 0, :] = tgrid.astype(np.float32)
    inputs[:, 1, :] = onset
    inputs[:, 2, :] = tib[:, None]

    result = dict(
        eid=eid,
        subject=str(probes.iloc[0].subject),
        lab=str(probes.iloc[0].lab),
        neural=[counts[i] for i in range(n_tr)],
        input=[inputs[i] for i in range(n_tr)],
        output=[outputs[i] for i in range(n_tr)],
        regions=[str(x) for x in beryl[unit_idx]],
        n_units_total=int(len(clusters)),
        n_units_good=int((clusters['label'].to_numpy() >= 1).sum()),
        n_units_kept=int(len(unit_idx)),
        n_trials_raw=int(len(trials)),
        n_trials_mask=int(mask.sum()),
        n_trials_kept=int(n_tr),
        me_side=me_side,
        thr_wheel=thr_w,
        thr_whisker=thr_m,
        n_probes=int(len(probes)),
        timings=timings,
        elapsed=time.time() - t0,
    )
    if show:
        plot_processing(result, trials, keep, align, wheel, whisk, wt, wv, mt, mv,
                        spike_times, spike_clusters, unit_idx, outdir)
    return result


# --------------------------------------------------------------------------------------
# Diagnostic plots
# --------------------------------------------------------------------------------------
def plot_processing(res, trials, keep, align, wheel, whisk, wt, wv, mt, mv,
                    spike_times, spike_clusters, unit_idx, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tgrid = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
    neural = np.stack(res['neural'])
    out = np.stack(res['output'])
    inp = np.stack(res['input'])
    k = min(3, len(align) - 1)

    fig, ax = plt.subplots(4, 2, figsize=(15, 15))

    # 1. raw spike raster vs binned counts for one trial
    t0 = align[k] + TIME_WINDOW[0]
    t1 = align[k] + TIME_WINDOW[1]
    sel = (spike_times >= t0) & (spike_times < t1) & np.isin(spike_clusters, unit_idx)
    remap = {c: i for i, c in enumerate(unit_idx)}
    ax[0, 0].plot(spike_times[sel] - align[k],
                  [remap[c] for c in spike_clusters[sel]], '|k', ms=2)
    ax[0, 0].axvline(0, color='r')
    ax[0, 0].set(title=f'raw spikes, trial {k} (aligned)', xlabel='time from stim onset (s)',
                 ylabel='unit')
    im = ax[0, 1].imshow(neural[k], aspect='auto', origin='lower', cmap='Greys',
                         extent=[tgrid[0], tgrid[-1], 0, neural.shape[1]])
    ax[0, 1].axvline(0, color='r')
    ax[0, 1].set(title='binned counts (same trial) -- must match raster',
                 xlabel='time from stim onset (s)', ylabel='unit')
    plt.colorbar(im, ax=ax[0, 1])

    # 2. PSTH
    psth = neural.mean(axis=(0, 1)) / BINSIZE
    ax[1, 0].plot(tgrid, psth)
    ax[1, 0].axvline(0, color='r')
    ax[1, 0].set(title='population PSTH (expect a stimulus-onset response)',
                 xlabel='time from stim onset (s)', ylabel='rate (Hz)')

    # 3. wheel: raw vs binned vs discretised
    m = (wt >= t0) & (wt < t1)
    ax[1, 1].plot(wt[m] - align[k], wv[m], 'k-', lw=.8, label='raw |velocity|')
    ax[1, 1].plot(tgrid, wheel[k], 'o-', ms=3, label='binned')
    for thr in res['thr_wheel']:
        ax[1, 1].axhline(thr, color='g', ls=':')
    ax[1, 1].legend(fontsize=7)
    ax[1, 1].set(title='wheel speed: raw vs binned (dotted = tertiles)',
                 xlabel='time from stim onset (s)')
    ax[2, 0].step(tgrid, out[k, 2], where='mid')
    ax[2, 0].set(title='wheel speed discretised', ylim=(-.5, 2.5),
                 xlabel='time from stim onset (s)', ylabel='class')

    # 4. whisker ME
    m = (mt >= t0) & (mt < t1)
    ax[2, 1].plot(mt[m] - align[k], mv[m], 'k-', lw=.8, label='raw ME')
    ax[2, 1].plot(tgrid, whisk[k], 'o-', ms=3, label='binned')
    for thr in res['thr_whisker']:
        ax[2, 1].axhline(thr, color='g', ls=':')
    ax[2, 1].legend(fontsize=7)
    ax[2, 1].set(title=f"whisker ME ({res['me_side']} camera): raw vs binned",
                 xlabel='time from stim onset (s)')
    ax[3, 0].step(tgrid, out[k, 3], where='mid')
    ax[3, 0].set(title='whisker ME discretised', ylim=(-.5, 2.5),
                 xlabel='time from stim onset (s)', ylabel='class')

    # 5. inputs
    ax[3, 1].plot(tgrid, inp[k, 0], label='time from onset')
    ax[3, 1].plot(tgrid, inp[k, 1], label='stim onset indicator')
    ax[3, 1].plot(tgrid, inp[k, 2] / max(1, inp[:, 2].max()), label='trial in block (scaled)')
    ax[3, 1].axvline(0, color='r')
    ax[3, 1].legend(fontsize=7)
    ax[3, 1].set(title='inputs', xlabel='time from stim onset (s)')

    fig.suptitle(f"{res['eid']}  subject={res['subject']}  "
                 f"{res['n_units_kept']} units, {res['n_trials_kept']} trials")
    fig.tight_layout()
    path = os.path.join(outdir, f"processing_{res['eid']}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f'  wrote {path}')


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def _worker(args):
    eid, probes, show, outdir = args
    try:
        return convert_session(eid, probes, show=show, outdir=outdir)
    except Exception as exc:
        return {'eid': eid, 'error': f'{type(exc).__name__}: {exc}'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='save per-step diagnostic plots for up to 2 sessions')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    t_start = time.time()
    # Build the rebuilt ONE index once in the parent; otherwise every worker process
    # discovers it missing and regenerates it redundantly.
    get_one()
    bwm = session_list()
    eids = list(dict.fromkeys(bwm.eid))
    if args.sample:
        eids = eids[:2]
    print(f'converting {len(eids)} sessions with {args.workers} workers')

    outdir = os.path.dirname(os.path.abspath(args.outfile)) or '/app'
    jobs = [(e, bwm[bwm.eid == e], args.show_processing and i < 2, outdir)
            for i, e in enumerate(eids)]

    results, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, j): j[0] for j in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if 'error' in r:
                errors.append(r)
                print(f'[{n}/{len(jobs)}] SKIP {r["eid"]}: {r["error"]}')
            else:
                results.append(r)
                el = time.time() - t_start
                print(f'[{n}/{len(jobs)}] {r["eid"]} {r["n_units_kept"]:4d}u '
                      f'{r["n_trials_kept"]:4d}tr {r["elapsed"]:5.1f}s '
                      f'(elapsed {el/60:.1f}m, eta {el/n*(len(jobs)-n)/60:.1f}m)')

    results.sort(key=lambda r: r['eid'])
    print(f'\nconverted {len(results)} sessions, skipped {len(errors)}')
    for e in errors:
        print(f'  skipped {e["eid"]}: {e["error"]}')

    subjects = sorted({r['subject'] for r in results})
    sub_idx = {s: i for i, s in enumerate(subjects)}
    regions = sorted({x for r in results for x in r['regions']})
    reg_idx = {s: i for i, s in enumerate(regions)}

    data = {
        'neural': [r['neural'] for r in results],
        'input': [r['input'] for r in results],
        'output': [r['output'] for r in results],
        'subjects': subjects,
        'subject_idx': np.array([sub_idx[r['subject']] for r in results], dtype=int),
        'brain_regions': regions,
        'brain_region_idx': [np.array([reg_idx[x] for x in r['regions']], dtype=int)
                             for r in results],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description': (
                'IBL brain-wide map: head-fixed mice report the side of a visual grating by '
                'turning a wheel. Decode choice (left/right), the block prior probability that '
                'the stimulus appears on the left (0.2/0.5/0.8), and 3-way discretised wheel '
                'speed and whisker motion energy, from stimulus-onset-aligned spike counts.'),
            'time_bin_size': BINSIZE * 1000.0,
            'temporal_alignment_event': 'stimulus onset (trials.stimOn_times)',
            'off_start': TIME_WINDOW[0],
            'off_end': TIME_WINDOW[1],
            'n_timepoints': N_BINS,
            'neural_units': 'spike counts per 20 ms bin (float32)',
            'alignment_note': (
                'bin i spans [off_start + i*20ms, off_start + (i+1)*20ms); the input '
                'time_from_stim_onset reports the bin end time t_i = linspace(-0.48, 1.5, 100), '
                'and behaviour is interpolated at those same t_i, as in the reference '
                'get_behavior_per_interval. The stim_onset indicator marks bin 25, the first '
                'bin containing t >= 0.'),
            'neuron_curation': ('well-isolated units (clusters.label >= 1, i.e. all three RIGOR '
                                'single-unit metrics passed) in grey matter (Beryl acronym not '
                                'root/void); probes merged per session; sessions require at least '
                                f'{MIN_UNITS_PER_SESSION} such neurons'),
            'trial_curation': ('reference load_trials_and_mask: no NaN in stimOn_times, choice, '
                               'feedback_times, probabilityLeft, firstMovement_times, '
                               'feedbackType; 0.08 s <= RT <= 2 s; feedback - goCue <= 10 s; '
                               'choice != 0; plus complete wheel and whisker-ME coverage'),
            'discretisation': ('wheel_speed and whisker_motion_energy split at per-session '
                               'tertiles of all retained (trial, bin) samples'),
            'session_info': [
                {'eid': r['eid'], 'subject': r['subject'], 'lab': r['lab'],
                 'n_probes': r['n_probes'], 'n_units_total': r['n_units_total'],
                 'n_units_good': r['n_units_good'], 'n_units_kept': r['n_units_kept'],
                 'n_trials_raw': r['n_trials_raw'], 'n_trials_mask': r['n_trials_mask'],
                 'n_trials_kept': r['n_trials_kept'], 'me_camera': r['me_side'],
                 'wheel_tertiles': r['thr_wheel'], 'whisker_tertiles': r['thr_whisker']}
                for r in results],
            'source': 'IBL brain-wide map (bwm_release.csv freeze), ONE local cache',
            'reference_code': 'Zhang et al. 2025, src/0_data_caching.py + utils/ibl_data_utils.py',
        },
    }

    n_tr = sum(len(s) for s in data['neural'])
    n_un = sum(len(b) for b in data['brain_region_idx'])
    print(f'\nsessions {len(results)} | subjects {len(subjects)} | regions {len(regions)} '
          f'| trials {n_tr} | neurons {n_un}')

    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f'wrote {args.outfile} '
          f'({os.path.getsize(args.outfile)/1e9:.2f} GB) in {(time.time()-t_start)/60:.1f} min')

    agg = {}
    for r in results:
        for k, v in r['timings'].items():
            agg[k] = agg.get(k, 0.0) + v
    print('cumulative stage timings (s):',
          {k: round(v, 1) for k, v in sorted(agg.items(), key=lambda x: -x[1])})


if __name__ == '__main__':
    main()
