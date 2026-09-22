"""
Convert the Allen Brain Observatory Visual Behavior 2P dataset into the decoder format.

Design (see /app/CONVERSION_NOTES.md for full justification):
  * A decoder *session* is one `ophys_session_id`. Multiscope sessions contain several
    simultaneously recorded imaging planes (experiments); their neurons are concatenated.
  * Only *active* (non-passive) behavior sessions are used; passive sessions have no licks
    or rewards so trial outcome would be degenerate.
  * Trials are the change-detection `go` and `catch` trials. `aborted` and `auto_rewarded`
    trials are excluded.
  * Each trial is aligned to `change_time` (the real change on go trials, the sham change on
    catch trials), which coincides exactly with an image flash onset.
  * Time bins are the 750 ms image presentation intervals used by Piet et al.: 3 flashes
    before the change through 4 flashes after = 8 bins per trial.
  * Neural activity is the detected calcium event magnitude summed within each bin.

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

DATA_DIR = '/app/data/visual-behavior-ophys-1.1.0/'
NWB_FMT = DATA_DIR + 'behavior_ophys_experiments/behavior_ophys_experiment_%d.nwb'
META_DIR = DATA_DIR + 'project_metadata/'

# ---- trial / binning parameters -------------------------------------------------------
BIN_SIZE = 0.75          # s, one image presentation interval (250 ms image + 500 ms gray)
N_PRE = 3                # flashes before the change flash
N_POST = 4               # flashes after the change flash
N_BINS = N_PRE + 1 + N_POST   # = 8
OFF_START = -N_PRE * BIN_SIZE        # -2.25 s
OFF_END = (N_POST + 1) * BIN_SIZE    # +3.75 s is end of last bin; start of last bin = +3.0
N_QUANTILES = 5
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']


def bin_edges_from_flashes(flash_starts):
    """Bin k spans [flash_start[k], flash_start[k] + BIN_SIZE)."""
    return flash_starts, flash_starts + BIN_SIZE


def binned_sum(values, timestamps, t0, t1):
    """Sum `values` (n_signals, n_samples) over each [t0[i], t1[i]) window.

    Uses a cumulative sum so the cost is O(n_signals * n_samples) once plus O(n_bins) lookups.
    """
    csum = np.concatenate([np.zeros((values.shape[0], 1)), np.cumsum(values, axis=1)], axis=1)
    i0 = np.searchsorted(timestamps, t0, side='left')
    i1 = np.searchsorted(timestamps, t1, side='left')
    return csum[:, i1] - csum[:, i0], (i1 - i0)


def binned_mean_1d(values, timestamps, t0, t1):
    """Mean of a 1-D signal in each window; empty windows fall back to linear interpolation
    at the window centre so that no NaN is produced."""
    s, n = binned_sum(values[None, :], timestamps, t0, t1)
    s = s[0]
    n = n.astype(float)
    out = np.empty_like(s)
    ok = n > 0
    out[ok] = s[ok] / n[ok]
    if np.any(~ok):
        centres = 0.5 * (t0[~ok] + t1[~ok])
        out[~ok] = np.interp(centres, timestamps, values)
    return out


def interpolate_nans(x):
    """Linearly interpolate NaNs (blinks) in a 1-D array; edges are filled with nearest value."""
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if bad.all():
        return None
    if bad.any():
        idx = np.arange(len(x))
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def quantile_bin(x, n=N_QUANTILES):
    """Discretise into n equal-percentile bins. Returns integer labels in [0, n-1].

    Percentile edges are computed on the values being discretised, so each class holds ~1/n
    of the samples. Ties (e.g. a mouse that is stationary for >20% of the time) can make
    edges coincide; np.searchsorted then simply assigns fewer samples to the collapsed class.
    """
    edges = np.quantile(x, np.linspace(0, 1, n + 1)[1:-1])
    return np.searchsorted(edges, x, side='right').astype(np.int64), edges


def get_session_table(sample=False):
    """Active experiments that are actually present on disk, grouped by ophys_session_id."""
    et = pd.read_csv(META_DIR + 'ophys_experiment_table.csv')
    have = set(int(f.split('_')[-1].split('.')[0])
               for f in os.listdir(DATA_DIR + 'behavior_ophys_experiments'))
    et = et[et.ophys_experiment_id.isin(have)]
    et = et[~et.passive]                       # active behavior sessions only
    et = et.sort_values(['ophys_session_id', 'ophys_experiment_id'])
    sessions = [(int(sid), list(map(int, g.ophys_experiment_id.values)))
                for sid, g in et.groupby('ophys_session_id')]
    if sample:
        # one single-plane and one Multiscope session, so both rig types are exercised
        single = [s for s in sessions if len(s[1]) == 1]
        multi = [s for s in sessions if len(s[1]) > 1]
        sessions = single[:1] + multi[:1]
    return sessions, et


def process_session(args):
    """Convert one ophys session. Returns a dict, or None if the session must be dropped."""
    sid, eids, show = args
    from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
        BehaviorOphysExperiment)
    t_start = time.time()
    timings = {}

    # ---- load every imaging plane of this session -------------------------------------
    t0 = time.time()
    planes = []
    for eid in eids:
        planes.append((eid, BehaviorOphysExperiment.from_nwb_path(NWB_FMT % eid)))
    timings['load'] = time.time() - t0

    ref = planes[0][1]          # behavior streams are identical across planes
    md = ref.metadata

    # ---- behaviour streams --------------------------------------------------------------
    sp = ref.stimulus_presentations
    sp = sp[sp.stimulus_block_name.str.contains('change_detection', na=False)]
    sp = sp.sort_values('start_time')
    flash_start = sp.start_time.values.astype(float)
    flash_image = sp.image_name.values.astype(object)
    flash_omitted = sp.omitted.values.astype(bool)
    flash_ischange = sp.is_change.values.astype(bool)
    # omitted flashes carry image_name 'omitted' already in this release; enforce it
    flash_image = np.where(flash_omitted, 'omitted', flash_image)

    trials = ref.trials
    keep = (trials.go.values.astype(bool) | trials['catch'].values.astype(bool))
    trials = trials[keep]
    if len(trials) < 2:
        return None

    # ---- eye tracking: required for the pupil output -----------------------------------
    eye = ref.eye_tracking
    if len(eye) == 0:
        print(f'  session {sid}: DROPPED (no eye tracking data)', flush=True)
        return None
    pupil = interpolate_nans(eye.pupil_width.values)
    if pupil is None:
        print(f'  session {sid}: DROPPED (pupil all NaN)', flush=True)
        return None
    eye_t = eye.timestamps.values.astype(float)
    order = np.argsort(eye_t)
    eye_t, pupil = eye_t[order], pupil[order]

    run = ref.running_speed
    run_t = run.timestamps.values.astype(float)
    run_v = run.speed.values.astype(float)
    order = np.argsort(run_t)
    run_t, run_v = run_t[order], run_v[order]
    run_v = interpolate_nans(run_v)

    # ---- locate each trial's change flash ----------------------------------------------
    t0 = time.time()
    change_times = trials.change_time.values.astype(float)
    j = np.searchsorted(flash_start, change_times - 1e-6, side='left')
    good = np.ones(len(trials), dtype=bool)
    good &= np.isfinite(change_times)
    j_clipped = np.clip(j, 0, len(flash_start) - 1)
    good &= np.abs(flash_start[j_clipped] - change_times) < 1e-4   # change is a flash onset
    good &= (j - N_PRE >= 0) & (j + N_POST < len(flash_start))     # window inside the block

    # outcome must be one of the four mutually exclusive categories
    oc = np.full(len(trials), -1, dtype=np.int64)
    for k, name in enumerate(OUTCOMES):
        oc[trials[name].values.astype(bool)] = k
    good &= oc >= 0

    idx = np.nonzero(good)[0]
    if len(idx) < 2:
        return None
    n_dropped = int((~good).sum())
    j = j[idx]
    timings['align'] = time.time() - t0

    # flash indices for every trial: (n_trials, N_BINS)
    bin_flash = j[:, None] + np.arange(-N_PRE, N_POST + 1)[None, :]
    bt0 = flash_start[bin_flash]                 # bin start times
    bt1 = bt0 + BIN_SIZE                         # bin end times
    flat_t0 = bt0.ravel()
    flat_t1 = bt1.ravel()
    n_trials = len(idx)

    # ---- neural: sum detected calcium events in each bin, per plane --------------------
    t0 = time.time()
    neural_planes, region_idx, n_cells_plane = [], [], []
    for eid, ds in planes:
        ev = np.vstack(ds.events.events.values).astype(np.float64)   # (n_cells, n_frames)
        ts = ds.ophys_timestamps.astype(float)
        s, _ = binned_sum(ev, ts, flat_t0, flat_t1)
        neural_planes.append(s.reshape(ev.shape[0], n_trials, N_BINS))
        struct = ds.metadata['targeted_structure']
        region_idx += [struct] * ev.shape[0]
        n_cells_plane.append(ev.shape[0])
    neural = np.concatenate(neural_planes, axis=0).astype(np.float32)   # (n_cells, n_tr, T)
    timings['neural'] = time.time() - t0

    # ---- outputs -------------------------------------------------------------------------
    t0 = time.time()
    img = flash_image[bin_flash]                                  # (n_trials, T) of str
    chg = flash_ischange[bin_flash].astype(np.int64)

    run_binned = binned_mean_1d(run_v, run_t, flat_t0, flat_t1).reshape(n_trials, N_BINS)
    pup_binned = binned_mean_1d(pupil, eye_t, flat_t0, flat_t1).reshape(n_trials, N_BINS)

    # five equal-percentile bins computed within this session: pupil is measured in camera
    # pixels and running propensity varies enormously between animals, so a within-session
    # percentile makes the class labels mean the same thing (relative level) in every session.
    run_cls, run_edges = quantile_bin(run_binned.ravel())
    pup_cls, pup_edges = quantile_bin(pup_binned.ravel())
    run_cls = run_cls.reshape(n_trials, N_BINS)
    pup_cls = pup_cls.reshape(n_trials, N_BINS)

    outcome = oc[idx]
    timings['outputs'] = time.time() - t0

    # ---- consistency assertions ---------------------------------------------------------
    tr_sel = trials.iloc[idx]
    is_go = tr_sel.go.values.astype(bool)
    # `is_change` marks only REAL image changes. Catch trials are sham changes where the
    # image does not change, so they must be 0 everywhere -- which is exactly the semantics
    # required for the image_change output ('1 right after a change in image identity').
    assert np.all(chg[is_go, N_PRE] == 1), 'go trial change bin is not flagged is_change'
    assert np.all(chg[~is_go, N_PRE] == 0), 'catch trial change bin is wrongly flagged'
    assert np.all(chg[:, :N_PRE] == 0), 'a change appears before the aligned change bin'
    assert np.array_equal(img[:, N_PRE], tr_sel.change_image_name.values.astype(object)), \
        'image at change bin != trials.change_image_name'
    pre = img[:, N_PRE - 1]
    init = tr_sel.initial_image_name.values.astype(object)
    notom = pre != 'omitted'
    assert np.all(pre[notom] == init[notom]), 'image before change != trials.initial_image_name'
    assert np.isfinite(neural).all() and np.isfinite(run_binned).all() and np.isfinite(pup_binned).all()

    result = dict(
        session_id=sid, experiment_ids=eids, mouse=str(md['mouse_id']),
        neural=neural, image=img, change=chg, run_cls=run_cls, pupil_cls=pup_cls,
        outcome=outcome, regions=region_idx,
        run_cont=run_binned, pupil_cont=pup_binned,
        run_edges=run_edges, pupil_edges=pup_edges,
        n_trials=n_trials, n_dropped=n_dropped, n_cells_plane=n_cells_plane,
        session_type=md['session_type'], project=md['project_code'], cre=md['cre_line'],
        rate=float(md['ophys_frame_rate']), timings=timings, total_time=time.time() - t_start,
        trial_change_times=change_times[idx], is_go=is_go,
    )

    if show:
        try:
            plot_processing(result, planes, flash_start, flash_image, flash_ischange,
                            run_t, run_v, eye_t, pupil, bt0, bt1)
        except Exception as e:
            print('  plotting failed:', e, flush=True)

    for _, ds in planes:
        del ds
    print(f'  session {sid}: {len(eids)} plane(s), {neural.shape[0]} neurons, '
          f'{n_trials} trials (dropped {n_dropped}), {time.time() - t_start:.1f}s', flush=True)
    return result


def plot_processing(res, planes, flash_start, flash_image, flash_ischange,
                    run_t, run_v, eye_t, pupil, bt0, bt1):
    """Show every processing step for a session so alignment/binning can be eyeballed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sid = res['session_id']
    fig, ax = plt.subplots(5, 2, figsize=(22, 18))

    # --- column 0: a 30 s raw stretch around the first trial's change -------------------
    ct = res['trial_change_times'][0]
    w0, w1 = ct - 8, ct + 8
    ds = planes[0][1]
    ts = ds.ophys_timestamps
    ev = np.vstack(ds.events.events.values)
    m = (ts >= w0) & (ts <= w1)

    a = ax[0, 0]
    for k in range(min(6, ev.shape[0])):
        a.plot(ts[m], ev[k][m] + k * 0.5, lw=0.8)
    for s0 in flash_start[(flash_start >= w0) & (flash_start <= w1)]:
        a.axvspan(s0, s0 + 0.25, color='0.85', zorder=0)
    a.axvline(ct, color='r', lw=2)
    a.set_title(f'session {sid}: raw detected events (grey = image flash, red = change)')
    a.set_xlabel('time (s)')

    a = ax[1, 0]
    a.plot(run_t[(run_t >= w0) & (run_t <= w1)], run_v[(run_t >= w0) & (run_t <= w1)], 'k')
    a.axvline(ct, color='r', lw=2)
    for e0, e1 in zip(bt0[0], bt1[0]):
        a.axvspan(e0, e1, color='b', alpha=0.08)
    a.set_title('raw running speed with the 8 trial bins shaded')
    a.set_ylabel('cm/s')

    a = ax[2, 0]
    me = (eye_t >= w0) & (eye_t <= w1)
    a.plot(eye_t[me], pupil[me], 'k')
    a.axvline(ct, color='r', lw=2)
    for e0, e1 in zip(bt0[0], bt1[0]):
        a.axvspan(e0, e1, color='b', alpha=0.08)
    a.set_title('raw pupil width (blinks interpolated)')

    a = ax[3, 0]
    a.plot(res['run_cont'][0], 'o-', label='binned mean speed')
    a.plot(res['pupil_cont'][0], 's-', label='binned mean pupil')
    a.axvline(N_PRE, color='r')
    a.legend(); a.set_title('trial 0: binned continuous signals'); a.set_xlabel('bin')

    a = ax[4, 0]
    a.imshow(res['neural'][:, 0, :], aspect='auto', interpolation='nearest')
    a.axvline(N_PRE - 0.5, color='r')
    a.set_title('trial 0: binned events (neurons x bins)'); a.set_xlabel('bin')

    # --- column 1: distributions and alignment across all trials ------------------------
    a = ax[0, 1]
    a.imshow(res['change'], aspect='auto', interpolation='nearest')
    a.set_title('image_change output (should be a single column at bin %d)' % N_PRE)
    a.set_xlabel('bin'); a.set_ylabel('trial')

    a = ax[1, 1]
    vocab = sorted(set(res['image'].ravel()))
    code = np.vectorize({v: i for i, v in enumerate(vocab)}.get)(res['image'])
    a.imshow(code, aspect='auto', interpolation='nearest', cmap='tab20')
    a.axvline(N_PRE - 0.5, color='r')
    a.set_title('image identity per bin (change at red line)')
    a.set_xlabel('bin'); a.set_ylabel('trial')

    a = ax[2, 1]
    a.hist(res['run_cont'].ravel(), bins=60, color='0.6')
    for e in res['run_edges']:
        a.axvline(e, color='r')
    a.set_title('binned running speed with quintile edges'); a.set_yscale('log')

    a = ax[3, 1]
    a.hist(res['pupil_cont'].ravel(), bins=60, color='0.6')
    for e in res['pupil_edges']:
        a.axvline(e, color='r')
    a.set_title('binned pupil width with quintile edges')

    a = ax[4, 1]
    cls_counts = [np.mean(res['run_cls'] == k) for k in range(N_QUANTILES)]
    pup_counts = [np.mean(res['pupil_cls'] == k) for k in range(N_QUANTILES)]
    out_counts = [np.mean(res['outcome'] == k) for k in range(len(OUTCOMES))]
    x = np.arange(N_QUANTILES)
    a.bar(x - 0.2, cls_counts, 0.4, label='running class')
    a.bar(x + 0.2, pup_counts, 0.4, label='pupil class')
    a.axhline(0.2, color='k', ls=':')
    a.set_title('class fractions (dotted = 20%%); outcomes: ' +
                ', '.join(f'{n}={f:.2f}' for n, f in zip(OUTCOMES, out_counts)))
    a.legend()

    fig.tight_layout()
    fig.savefig(f'/app/processing_{sid}.png', dpi=110)
    plt.close(fig)
    print(f'  wrote /app/processing_{sid}.png', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--workers', type=int, default=24)
    args = ap.parse_args()

    t_all = time.time()
    sessions, et = get_session_table(sample=args.sample)
    print(f'{len(sessions)} sessions to process '
          f'({sum(len(e) for _, e in sessions)} imaging planes)', flush=True)

    jobs = [(sid, eids, args.show_processing and i < 2) for i, (sid, eids) in enumerate(sessions)]
    if args.sample or args.workers <= 1:
        results = [process_session(j) for j in jobs]
    else:
        with Pool(min(args.workers, len(jobs))) as pool:
            results = pool.map(process_session, jobs)
    results = [r for r in results if r is not None]
    print(f'{len(results)} sessions converted in {time.time() - t_all:.1f}s', flush=True)

    # ---- shared categorical vocabularies ------------------------------------------------
    images = sorted(set(v for r in results for v in np.unique(r['image'])))
    images = [v for v in images if v != 'omitted'] + ['omitted']
    img_map = {v: i for i, v in enumerate(images)}
    regions = sorted(set(v for r in results for v in r['regions']))
    subjects = sorted(set(r['mouse'] for r in results))

    data = dict(neural=[], input=[], output=[], subjects=subjects, subject_idx=[],
                brain_regions=regions, brain_region_idx=[],
                input_names=[], output_names=['image_identity', 'image_change',
                                             'running_speed', 'pupil_diameter',
                                             'trial_outcome'],
                output_values=[images, ['no_change', 'change'],
                               [f'speed_q{k+1}' for k in range(N_QUANTILES)],
                               [f'pupil_q{k+1}' for k in range(N_QUANTILES)],
                               OUTCOMES])
    session_info = []
    for r in results:
        n_tr = r['n_trials']
        img_idx = np.vectorize(img_map.get)(r['image']).astype(np.int64)
        data['neural'].append([np.ascontiguousarray(r['neural'][:, t, :]) for t in range(n_tr)])
        data['input'].append([np.zeros((0, N_BINS), dtype=np.float32) for _ in range(n_tr)])
        data['output'].append([
            np.stack([img_idx[t], r['change'][t], r['run_cls'][t], r['pupil_cls'][t],
                      np.full(N_BINS, r['outcome'][t], dtype=np.int64)]).astype(np.int64)
            for t in range(n_tr)])
        data['subject_idx'].append(subjects.index(r['mouse']))
        data['brain_region_idx'].append(np.array([regions.index(x) for x in r['regions']],
                                                 dtype=np.int64))
        session_info.append(dict(session_id=r['session_id'], experiment_ids=r['experiment_ids'],
                                 mouse=r['mouse'], session_type=r['session_type'],
                                 project_code=r['project'], cre_line=r['cre'],
                                 ophys_frame_rate=r['rate'], n_neurons=r['neural'].shape[0],
                                 n_trials=n_tr, n_trials_dropped=r['n_dropped']))
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)

    data['metadata'] = dict(
        task_description=(
            'Allen Brain Observatory Visual Behavior 2P: head-fixed mice perform a go/no-go '
            'visual change detection task on a series of natural images flashed for 250 ms '
            'every 750 ms. Decoded from the neural activity are the identity of the image '
            'shown in each 750 ms image presentation interval, whether that interval '
            'contained an image change, the binned running speed and pupil diameter, and '
            'the outcome of the trial (hit / miss / false alarm / correct reject).'),
        time_bin_size=BIN_SIZE * 1000.0,
        temporal_alignment_event=('image change (go trials) or sham image change (catch '
                                  'trials), i.e. trials.change_time, which coincides exactly '
                                  'with an image flash onset'),
        off_start=OFF_START,
        off_end=N_POST * BIN_SIZE,
        n_time_bins=N_BINS,
        neural_signal=('detected calcium event magnitude (allensdk events.events) summed '
                       'within each 750 ms bin'),
        trial_types_included='go and catch',
        trial_types_excluded='aborted, auto_rewarded, and all passive sessions',
        neuron_curation='valid ROIs only (allensdk exclude_invalid_rois=True)',
        discretization=('running speed and pupil diameter are binned into 5 equal-percentile '
                        'classes computed within each session'),
        session_info=session_info,
        dataset='visual-behavior-ophys-1.1.0',
    )

    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    n_neu = sum(len(b) for b in data['brain_region_idx'])
    n_tr = sum(len(s) for s in data['neural'])
    print(f'\nWrote {args.outfile}: {len(data["neural"])} sessions, {n_neu} neurons, '
          f'{n_tr} trials, {len(subjects)} mice, regions={regions}')
    print(f'image vocabulary ({len(images)}): {images}')
    print(f'total time {time.time() - t_all:.1f}s')


if __name__ == '__main__':
    main()
