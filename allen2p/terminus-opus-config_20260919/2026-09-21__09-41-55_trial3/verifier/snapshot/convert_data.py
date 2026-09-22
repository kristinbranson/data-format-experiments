"""
Convert the Allen Brain Observatory Visual Behavior 2P dataset (visual-behavior-ophys-1.1.0)
into the decoder-compatible pickle format.

Usage:
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

See /app/CONVERSION_NOTES.md for all design decisions.

Summary of the conversion
-------------------------
* Data selection : active (behaving) sessions with the FAMILIAR image set (image set A),
                   i.e. session_type in {OPHYS_1_images_A, OPHYS_3_images_A}.
                   All simultaneously recorded imaging planes of one ophys_session_id are
                   merged into a single "session".
* Neural         : detected calcium `events` (SDK event detection), averaged within time bins.
* Trials         : trials.(go | catch) & ~auto_rewarded  (aborted and auto-rewarded excluded)
* Alignment      : trials.change_time (real change on GO trials, sham change on CATCH trials)
* Window         : [-2.0, +4.0] s, 250 ms bins -> 24 bins per trial
* Inputs         : none (0-dimensional)
* Outputs        : image identity (8), image change (2), running-speed quintile (5),
                   pupil-diameter quintile (5), trial outcome (4)
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

CACHE_DIR = '/app/data'
NWB_DIR = os.path.join(CACHE_DIR, 'visual-behavior-ophys-1.1.0', 'behavior_ophys_experiments')

# ------------------------- conversion parameters -------------------------
BIN_SIZE = 0.750          # s, time bin size = one image-presentation interval
OFF_START = -2.25         # s, relative to change_time (3 image intervals before the change)
OFF_END = 3.75            # s, relative to change_time (5 image intervals from the change)
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 8
FAMILIAR_SESSION_TYPES = ('OPHYS_1_images_A', 'OPHYS_3_images_A')
PUPIL_INTERP_MAX_GAP = 1.0   # s, maximum blink gap that is linearly interpolated
NQUANTILES = 5               # quintiles for running speed / pupil diameter
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def get_cache():
    import allensdk.brain_observatory.behavior.behavior_project_cache as bpc
    return bpc.VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=CACHE_DIR)


def local_experiment_table():
    """Experiment (imaging-plane) metadata table restricted to the NWB files present locally."""
    import re
    import glob
    ids = sorted(int(re.search(r'(\d+)\.nwb', f).group(1))
                 for f in glob.glob(os.path.join(NWB_DIR, '*.nwb')))
    et = get_cache().get_ophys_experiment_table()
    et = et.loc[et.index.intersection(ids)]
    return et


def select_experiments():
    """Apply the data-selection rules and return the filtered experiment table."""
    et = local_experiment_table()
    n0 = len(et)
    et = et[~et.passive]                                     # active behavior only
    n1 = len(et)
    et = et[et.session_type.isin(FAMILIAR_SESSION_TYPES)]    # familiar image set only
    n2 = len(et)
    print(f'[select] local experiments {n0} -> active {n1} -> familiar {n2}')
    print(f'[select] {et.ophys_session_id.nunique()} ophys sessions, '
          f'{et.mouse_id.nunique()} mice, {et.targeted_structure.value_counts().to_dict()}')
    return et


def bin_sum_count(values, timestamps, edges_flat):
    """Sum and sample-count of `values` (..., T) within bins given by monotone `edges_flat`.

    values: (n, T) or (T,) array, timestamps: (T,) monotone increasing.
    edges_flat: (ntrials, nbins+1) array of bin edges (each row monotone increasing).
    Returns sums (n, ntrials, nbins) and counts (ntrials, nbins).
    """
    v = np.atleast_2d(values).astype(np.float64)
    idx = np.searchsorted(timestamps, edges_flat)            # (ntrials, nbins+1)
    cs = np.concatenate([np.zeros((v.shape[0], 1)), np.cumsum(v, axis=1)], axis=1)
    sums = cs[:, idx[:, 1:]] - cs[:, idx[:, :-1]]            # (n, ntrials, nbins)
    counts = (idx[:, 1:] - idx[:, :-1]).astype(np.float64)   # (ntrials, nbins)
    return sums, counts


def bin_mean_1d(values, timestamps, edges, nan_aware=False):
    """Mean of a 1-D time series within bins. NaN-aware if requested."""
    if nan_aware:
        valid = np.isfinite(values)
        sums, _ = bin_sum_count(np.where(valid, values, 0.0), timestamps, edges)
        cnts, _ = bin_sum_count(valid.astype(np.float64), timestamps, edges)
        with np.errstate(invalid='ignore', divide='ignore'):
            out = sums[0] / cnts[0]
        out[cnts[0] == 0] = np.nan
        return out
    sums, counts = bin_sum_count(values, timestamps, edges)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = sums[0] / counts
    out[counts == 0] = np.nan
    return out


def interpolate_short_gaps(t, x, max_gap):
    """Linearly interpolate NaNs in x(t) across gaps shorter than max_gap seconds.

    Longer gaps (tracking failures) are left as NaN so that the affected trials can be dropped.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    good = np.isfinite(x)
    if good.sum() < 2:
        return x
    xi = np.interp(t, t[good], x[good])
    # identify NaN runs and only fill the short ones
    bad = ~good
    if bad.any():
        d = np.diff(bad.astype(np.int8))
        starts = list(np.nonzero(d == 1)[0] + 1)
        ends = list(np.nonzero(d == -1)[0] + 1)
        if bad[0]:
            starts = [0] + starts
        if bad[-1]:
            ends = ends + [len(bad)]
        for s, e in zip(starts, ends):
            if s == 0 or e == len(bad):
                continue      # edge gaps have no two-sided support: leave NaN
            if t[e - 1] - t[s] <= max_gap:
                x[s:e] = xi[s:e]
    return x


def quantile_bin(values, nq=NQUANTILES):
    """Discretise into nq equal-percentile bins; returns labels (ints) and the edges used."""
    v = np.asarray(values, dtype=np.float64)
    finite = v[np.isfinite(v)]
    qs = np.linspace(0, 100, nq + 1)[1:-1]
    edges = np.percentile(finite, qs)
    # np.digitize with monotone-nondecreasing edges (ties -> some classes may be empty)
    labels = np.digitize(v, edges, right=False)
    return labels.astype(np.int64), edges


# --------------------------------------------------------------------------
# per-session conversion
# --------------------------------------------------------------------------
def convert_session(args):
    """Convert one ophys session (all of its imaging planes) into the target format."""
    session_id, exp_ids, meta, show_processing, signal, binparams, normalize = args
    global BIN_SIZE, OFF_START, OFF_END, NBINS
    BIN_SIZE, OFF_START, OFF_END, NBINS = binparams
    t_start = time.time()
    timings = {}
    try:
        bc = get_cache()

        # ---------------- load all imaging planes of this session ----------------
        t0 = time.time()
        planes = []
        beh = None
        for eid in exp_ids:
            ds = bc.get_behavior_ophys_experiment(int(eid))
            ev = np.vstack(ds.events[signal].values).astype(np.float64)   # (ncells, nframes)
            ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
            assert ev.shape[1] == ts.size, 'events / timestamps length mismatch'
            planes.append(dict(eid=int(eid), events=ev, ts=ts,
                               cell_ids=list(ds.events.index.values),
                               structure=ds.metadata['targeted_structure'],
                               depth=ds.metadata['imaging_depth'],
                               frame_rate=ds.metadata['ophys_frame_rate']))
            if beh is None:
                # behavior streams are identical across simultaneously recorded planes
                beh = dict(trials=ds.trials.copy(),
                           stim=ds.stimulus_presentations.copy(),
                           run=ds.running_speed.copy(),
                           eye=None if ds.eye_tracking is None else ds.eye_tracking.copy(),
                           behavior_session_id=ds.behavior_session_id)
        timings['load'] = time.time() - t0

        if beh['eye'] is None or len(beh['eye']) == 0:
            return dict(session_id=session_id, skip='no eye tracking', timings=timings)

        # ---------------- trials: go | catch, excluding auto-rewarded ----------------
        tr = beh['trials']
        keep = (tr['go'].astype(bool) | tr['catch'].astype(bool)) & ~tr['auto_rewarded'].astype(bool)
        tr = tr[keep]
        assert tr['change_time'].notna().all(), 'missing change_time on a go/catch trial'
        # outcome flags partition the selected trials
        outcome_mat = np.stack([tr[c].astype(bool).values for c in OUTCOME_NAMES], axis=1)
        assert np.all(outcome_mat.sum(axis=1) == 1), 'trial outcomes do not partition trials'
        outcome = np.argmax(outcome_mat, axis=1).astype(np.int64)
        change_times = tr['change_time'].values.astype(np.float64)
        ntrials = len(tr)

        # trial windows must lie inside the trial
        assert np.all(change_times - tr['start_time'].values >= -OFF_START), 'window precedes trial start'
        assert np.all(tr['stop_time'].values - change_times >= OFF_END), 'window exceeds trial stop'

        # ---------------- bin edges ----------------
        offs = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
        edges = change_times[:, None] + offs[None, :]         # (ntrials, NBINS+1)
        centers = edges[:, :-1] + BIN_SIZE / 2.0

        # ---------------- neural: mean detected-event magnitude per bin ----------------
        t0 = time.time()
        neural_blocks, region_list, depth_list, cellid_list = [], [], [], []
        for p in planes:
            sums, counts = bin_sum_count(p['events'], p['ts'], edges)
            assert counts.min() > 0, 'empty time bin (no imaging frame): bin size too small'
            neural_blocks.append((sums / counts[None, :, :]).astype(np.float32))
            region_list += [p['structure']] * p['events'].shape[0]
            depth_list += [p['depth']] * p['events'].shape[0]
            cellid_list += list(p['cell_ids'])
        neural = np.concatenate(neural_blocks, axis=0)       # (ncells, ntrials, NBINS)
        # Per-neuron scaling: detected calcium events have cell-specific magnitudes spanning
        # orders of magnitude, and the decoder initialises its projection from a raw SVD, so a
        # handful of large-amplitude cells would otherwise dominate every principal component.
        # Scaling is monotone per neuron, so it changes no event time or relative time course.
        if normalize in ('std', 'zscore'):
            flat = neural.reshape(neural.shape[0], -1)
            mu = flat.mean(axis=1, keepdims=True)
            sd = flat.std(axis=1, keepdims=True)
            sd[sd == 0] = 1.0
            if normalize == 'zscore':
                flat = (flat - mu) / sd
            else:
                flat = flat / sd
            neural = flat.reshape(neural.shape).astype(np.float32)
        timings['neural'] = time.time() - t0

        # ---------------- stimulus: image identity and image change ----------------
        t0 = time.time()
        stim = beh['stim']
        stim = stim[stim.stimulus_block_name.astype(str).str.contains('change_detection')]
        stim = stim.sort_values('start_time')
        fstart = stim['start_time'].values.astype(np.float64)
        omitted = stim['omitted'].fillna(False).astype(bool).values
        is_change = stim['is_change'].fillna(False).astype(bool).values
        names = stim['image_name'].astype(str).values
        # forward-fill image identity across omissions (an omission replaces a *repeat*,
        # so the ongoing image identity is the previous image)
        names_ff = pd.Series(np.where(omitted, None, names)).ffill().bfill().values
        image_names = sorted(set(names_ff[~pd.isna(names_ff)]) - {'omitted'})
        name_to_idx = {n: i for i, n in enumerate(image_names)}
        img_idx = np.array([name_to_idx[n] for n in names_ff], dtype=np.int64)

        # index of the image-presentation interval (750 ms) containing each bin centre
        j = np.searchsorted(fstart, centers, side='right') - 1
        assert j.min() >= 0, 'bin centre precedes the first flash'
        image_identity = img_idx[j]                                   # (ntrials, NBINS)
        image_change = is_change[j].astype(np.int64)                  # (ntrials, NBINS)
        timings['stim'] = time.time() - t0

        # ---------------- running speed ----------------
        t0 = time.time()
        run = beh['run']
        run_t = run['timestamps'].values.astype(np.float64)
        run_v = run['speed'].values.astype(np.float64)
        run_binned = bin_mean_1d(run_v, run_t, edges, nan_aware=True)  # (ntrials, NBINS)

        # ---------------- pupil diameter ----------------
        eye = beh['eye']
        eye_t = eye['timestamps'].values.astype(np.float64)
        # pupil_area = pi * max(width, height)^2  (SDK compute_circular_area)
        pupil_diam = 2.0 * np.sqrt(eye['pupil_area'].values.astype(np.float64) / np.pi)
        pupil_diam = interpolate_short_gaps(eye_t, pupil_diam, PUPIL_INTERP_MAX_GAP)
        pupil_binned = bin_mean_1d(pupil_diam, eye_t, edges, nan_aware=True)
        timings['behavior'] = time.time() - t0

        # ---------------- drop trials with missing behavior data ----------------
        bad = (~np.isfinite(run_binned).all(axis=1)) | (~np.isfinite(pupil_binned).all(axis=1))
        nbad = int(bad.sum())
        good = ~bad
        if good.sum() < 2:
            return dict(session_id=session_id, skip=f'<2 usable trials ({int(good.sum())})',
                        timings=timings)

        neural = neural[:, good, :]
        image_identity = image_identity[good]
        image_change = image_change[good]
        run_binned = run_binned[good]
        pupil_binned = pupil_binned[good]
        outcome = outcome[good]
        trial_ids = np.asarray(tr.index.values)[good]
        change_times_good = change_times[good]

        # ---------------- assemble per-trial arrays ----------------
        # NOTE: running speed and pupil diameter are discretised later (in `finalize`), so that
        # the percentile edges can be computed either within session or across the whole dataset.
        n_good = int(good.sum())
        neural_trials = [np.ascontiguousarray(neural[:, i, :]) for i in range(n_good)]
        input_trials = [np.zeros((0, NBINS), dtype=np.float32) for _ in range(n_good)]
        output_trials = []
        for i in range(n_good):
            output_trials.append(np.stack([
                image_identity[i],
                image_change[i],
                np.zeros(NBINS, dtype=np.int64),   # placeholder: running quintile
                np.zeros(NBINS, dtype=np.int64),   # placeholder: pupil quintile
                np.full(NBINS, outcome[i], dtype=np.int64),
            ]).astype(np.int64))

        regions = np.array(region_list)
        result = dict(
            session_id=int(session_id),
            skip=None,
            neural=neural_trials,
            input=input_trials,
            output=output_trials,
            regions=regions,
            run_binned=run_binned.astype(np.float32),
            pupil_binned=pupil_binned.astype(np.float32),
            depths=np.array(depth_list),
            cell_ids=np.array(cellid_list),
            mouse_id=str(meta['mouse_id']),
            image_names=image_names,
            info=dict(ophys_session_id=int(session_id),
                      ophys_experiment_ids=[int(e) for e in exp_ids],
                      behavior_session_id=int(beh['behavior_session_id']),
                      mouse_id=str(meta['mouse_id']),
                      cre_line=meta['cre_line'],
                      session_type=meta['session_type'],
                      experience_level=meta['experience_level'],
                      equipment_name=meta['equipment_name'],
                      project_code=meta['project_code'],
                      image_set=meta['image_set'],
                      frame_rate=float(planes[0]['frame_rate']),
                      n_neurons=int(neural.shape[0]),
                      n_trials=n_good,
                      n_trials_dropped_behavior=nbad,
                      n_trials_selected=int(ntrials),
                      trial_ids=[int(x) for x in trial_ids],
                      structures=sorted(set(region_list)),
                      n_hit=int((outcome == 0).sum()), n_miss=int((outcome == 1).sum()),
                      n_false_alarm=int((outcome == 2).sum()), n_correct_reject=int((outcome == 3).sum()),
                      ),
            timings=timings,
        )

        # ---------------- sanity checks (per session) ----------------
        checks = sanity_checks(result, beh, tr[good.tolist() if False else good], stim,
                               change_times_good, offs)
        result['checks'] = checks

        if show_processing:
            result['_plotargs'] = dict(planes=planes, stim=stim, centers=centers,
                                       image_identity=image_identity, image_change=image_change,
                                       image_names=image_names, pupil_diam=pupil_diam, eye_t=eye_t,
                                       run_t=run_t, run_v=run_v, change_times=change_times_good)

        timings['total'] = time.time() - t_start
        return result
    except Exception as ex:
        import traceback
        return dict(session_id=int(session_id), skip='ERROR: ' + repr(ex),
                    traceback=traceback.format_exc(), timings=timings)


def sanity_checks(result, beh, tr, stim, change_times, offs):
    """Per-session consistency checks comparing converted arrays with the source tables."""
    out = {}
    fstart = stim['start_time'].values.astype(np.float64)
    # 1. every change time coincides with a flash onset (as asserted in the reference code)
    out['change_times_are_flash_onsets'] = bool(
        np.all(np.min(np.abs(change_times[:, None] - fstart[None, :]), axis=1) < 1e-6))
    # 2. bin containing t=0 has the change image; bin at t=-0.25 s has the initial image
    bin0 = int(np.searchsorted(offs, 0.0, side='right') - 1)
    img_names = result['image_names']
    ident = np.stack([o[0] for o in result['output']])
    chg = np.stack([o[1] for o in result['output']])
    go = tr['go'].astype(bool).values
    change_img = tr['change_image_name'].astype(str).values
    init_img = tr['initial_image_name'].astype(str).values
    ok_change_img = [img_names[ident[i, bin0]] == change_img[i] for i in range(len(tr)) if go[i]]
    out['bin0_image_is_change_image_go'] = float(np.mean(ok_change_img)) if ok_change_img else None
    ok_init_img = [img_names[ident[i, bin0 - 1]] == init_img[i] for i in range(len(tr)) if go[i]]
    out['prebin_image_is_initial_image_go'] = float(np.mean(ok_init_img)) if ok_init_img else None
    # 3. image_change is 1 exactly in the 750 ms interval starting at the change on GO trials,
    #    and never on CATCH trials
    expected = np.zeros(chg.shape[1], dtype=int)
    expected[bin0:bin0 + int(round(0.75 / BIN_SIZE))] = 1
    out['change_pattern_go'] = float(np.mean([np.array_equal(chg[i], expected)
                                              for i in range(len(tr)) if go[i]])) if go.any() else None
    out['change_zero_catch'] = float(np.mean([chg[i].sum() == 0
                                              for i in range(len(tr)) if not go[i]])) if (~go).any() else None
    # 4. neural: finite, non-negative
    neural = np.stack(result['neural'])
    out['neural_finite'] = bool(np.isfinite(neural).all())
    out['neural_nonneg'] = bool((neural >= 0).all())  # False by design if z-scored
    out['neural_mean'] = float(neural.mean())
    # 5. behaviour streams are finite after binning/interpolation
    out['running_binned_finite'] = bool(np.isfinite(result['run_binned']).all())
    out['pupil_binned_finite'] = bool(np.isfinite(result['pupil_binned']).all())
    out['running_range'] = [float(result['run_binned'].min()), float(result['run_binned'].max())]
    out['pupil_range'] = [float(result['pupil_binned'].min()), float(result['pupil_binned'].max())]
    return out


# --------------------------------------------------------------------------
# plotting for --show-processing
# --------------------------------------------------------------------------
def plot_processing(result, run_edges, pupil_edges):
    """Six-panel visualisation of every processing step of one session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pa = result['_plotargs']
    planes = pa['planes']; stim = pa['stim']; centers = pa['centers']
    image_identity = pa['image_identity']; image_change = pa['image_change']
    image_names = pa['image_names']; pupil_diam = pa['pupil_diam']; eye_t = pa['eye_t']
    run_t = pa['run_t']; run_v = pa['run_v']; change_times = pa['change_times']
    run_binned = result['run_binned']; pupil_binned = result['pupil_binned']

    sid = result['session_id']
    itr = min(5, len(result['neural']) - 1)          # trial to display
    t_change = change_times[itr]
    ed = change_times[itr] + OFF_START + BIN_SIZE * np.arange(NBINS + 1)
    cen = ed[:-1] + BIN_SIZE / 2
    t0, t1 = ed[0] - 1.0, ed[-1] + 1.0

    fig, axs = plt.subplots(6, 1, figsize=(16, 18), sharex=False)

    # --- 1. neural: raw event trace vs binned ---
    ax = axs[0]
    p = planes[0]
    m = (p['ts'] >= t0) & (p['ts'] <= t1)
    ncell_plot = min(3, p['events'].shape[0])
    for c in range(ncell_plot):
        ax.plot(p['ts'][m], p['events'][c][m] + c * 0.5, lw=0.8, alpha=0.6,
                label=f'cell {c} raw events')
        ax.step(cen, result['neural'][itr][c] + c * 0.5, where='mid', lw=2,
                label=f'cell {c} binned ({int(BIN_SIZE*1000)} ms)')
    for e in ed:
        ax.axvline(e, color=[.85, .85, .85], lw=0.5, zorder=0)
    ax.axvline(t_change, color='r', ls='--', label='change_time (t=0)')
    ax.set_title(f'session {sid}: detected calcium events, raw (thin) vs {int(BIN_SIZE*1000)} ms bins (thick); trial {itr}')
    ax.legend(fontsize=7, ncol=3); ax.set_xlim(t0, t1); ax.set_ylabel('event magnitude')

    # --- 2. stimulus: flashes, image identity, image change ---
    ax = axs[1]
    sub = stim[(stim.start_time >= t0) & (stim.start_time <= t1)]
    cmap = plt.get_cmap('tab10')
    for _, r in sub.iterrows():
        if bool(r['omitted']):
            ax.axvspan(r['start_time'], r['start_time'] + 0.25, color='k', alpha=0.15)
        else:
            ci = image_names.index(str(r['image_name']))
            ax.axvspan(r['start_time'], r['start_time'] + 0.25, color=cmap(ci), alpha=0.6)
        if bool(r['is_change']):
            ax.axvline(r['start_time'], color='r', lw=2)
    ax.step(cen, image_identity[itr], where='mid', color='k', lw=2, label='binned image identity')
    ax.step(cen, image_change[itr] * 7, where='mid', color='m', lw=2, label='binned image change x7')
    ax.axvline(t_change, color='r', ls='--')
    ax.set_title('stimulus: image flashes (colour = image identity, grey = omitted, red line = change)')
    ax.legend(fontsize=7); ax.set_xlim(t0, t1); ax.set_ylabel('image index')

    # --- 3. running speed raw vs binned + quintile edges ---
    ax = axs[2]
    m = (run_t >= t0) & (run_t <= t1)
    ax.plot(run_t[m], run_v[m], color=[.5, .5, .5], lw=0.8, label='raw running speed (60 Hz)')
    ax.step(cen, run_binned[itr], where='mid', color='b', lw=2, label='binned mean')
    for e in run_edges:
        ax.axhline(e, color='g', ls=':', lw=1)
    ax2 = ax.twinx()
    ax2.step(cen, result['output'][itr][2], where='mid', color='r', lw=2, alpha=0.6,
             label='quintile class')
    ax2.set_ylim(-0.5, 4.5); ax2.set_ylabel('quintile class', color='r')
    ax.axvline(t_change, color='r', ls='--')
    ax.set_title('running speed: raw, binned, and quintile discretisation (green dotted = quintile edges)')
    ax.legend(fontsize=7, loc='upper left'); ax.set_xlim(t0, t1); ax.set_ylabel('cm/s')

    # --- 4. pupil diameter raw vs binned + quintile edges ---
    ax = axs[3]
    m = (eye_t >= t0) & (eye_t <= t1)
    ax.plot(eye_t[m], pupil_diam[m], color=[.5, .5, .5], lw=0.8, label='pupil diameter (px)')
    ax.step(cen, pupil_binned[itr], where='mid', color='b', lw=2, label='binned mean')
    for e in pupil_edges:
        ax.axhline(e, color='g', ls=':', lw=1)
    ax2 = ax.twinx()
    ax2.step(cen, result['output'][itr][3], where='mid', color='r', lw=2, alpha=0.6)
    ax2.set_ylim(-0.5, 4.5); ax2.set_ylabel('quintile class', color='r')
    ax.axvline(t_change, color='r', ls='--')
    ax.set_title('pupil diameter: raw, binned, and quintile discretisation')
    ax.legend(fontsize=7, loc='upper left'); ax.set_xlim(t0, t1); ax.set_ylabel('px')

    # --- 5. trial-averaged neural activity aligned to change, by outcome ---
    ax = axs[4]
    neural = np.stack(result['neural'])           # (ntrials, ncells, NBINS)
    outc = np.array([o[4, 0] for o in result['output']])
    tt = OFF_START + BIN_SIZE * (np.arange(NBINS) + 0.5)
    for c, nm in enumerate(OUTCOME_NAMES):
        if (outc == c).sum() == 0:
            continue
        ax.plot(tt, neural[outc == c].mean(axis=(0, 1)), label=f'{nm} (n={int((outc==c).sum())})')
    ax.axvline(0, color='r', ls='--')
    for k in np.arange(-3, 6) * 0.75:
        ax.axvline(k, color=[.9, .9, .9], lw=0.5, zorder=0)
    ax.set_title('population-mean event magnitude aligned to change_time, split by trial outcome '
                 '(grey lines = 750 ms flash cycle)')
    ax.set_xlabel('time from change (s)'); ax.legend(fontsize=7)

    # --- 6. output class distributions over the whole session ---
    ax = axs[5]
    labels = ['image_identity', 'image_change', 'running_quintile', 'pupil_quintile', 'trial_outcome']
    nclass = [len(image_names), 2, NQUANTILES, NQUANTILES, len(OUTCOME_NAMES)]
    allout = np.stack(result['output'])            # (ntrials, 5, NBINS)
    off = 0
    for k in range(5):
        v = allout[:, k, :].ravel()
        fr = [(v == c).mean() for c in range(nclass[k])]
        ax.bar(np.arange(nclass[k]) + off, fr, label=labels[k])
        off += nclass[k] + 1
    ax.set_ylabel('fraction of bins'); ax.set_title('output class distributions (whole session)')
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(f'/app/processing_{sid}.png', dpi=110)
    plt.close(fig)
    print(f'[plot] wrote /app/processing_{sid}.png')


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    global BIN_SIZE, OFF_START, OFF_END, NBINS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile', type=str)
    ap.add_argument('--full', action='store_true', help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='save per-step visualisations for up to 2 sessions')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--neural-signal', type=str, default='events',
                    choices=['events', 'filtered_events'])
    ap.add_argument('--nsessions', type=int, default=0, help='limit number of sessions (debug)')
    ap.add_argument('--normalize', type=str, default='zscore', choices=['none', 'std', 'zscore'],
                    help='per-neuron scaling of the binned event magnitudes')
    ap.add_argument('--bin-size', type=float, default=BIN_SIZE)
    ap.add_argument('--off-start', type=float, default=OFF_START)
    ap.add_argument('--off-end', type=float, default=OFF_END)
    ap.add_argument('--quantile-scope', type=str, default='global',
                    choices=['global', 'session'],
                    help='compute running/pupil percentile bin edges over the whole dataset (global) or within each session')
    args = ap.parse_args()

    BIN_SIZE, OFF_START, OFF_END = args.bin_size, args.off_start, args.off_end
    NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
    print(f'[params] bin size {BIN_SIZE*1000:.0f} ms, window [{OFF_START}, {OFF_END}] s, {NBINS} bins')

    t_start = time.time()
    et = select_experiments()

    # group imaging planes into sessions
    sessions = []
    for sid, grp in et.groupby('ophys_session_id'):
        meta = grp.iloc[0][['mouse_id', 'cre_line', 'session_type', 'experience_level',
                            'equipment_name', 'project_code', 'image_set']].to_dict()
        meta['mouse_id'] = str(meta['mouse_id'])
        sessions.append((int(sid), list(grp.index.values), meta))
    sessions.sort(key=lambda s: s[0])

    if args.sample:
        # one single-plane and one multi-plane session, for coverage
        multi = [s for s in sessions if len(s[1]) > 1]
        single = [s for s in sessions if len(s[1]) == 1]
        sessions = ([multi[0]] if multi else []) + single[:2]
        sessions = sessions[:2]
        print(f'[main] SAMPLE mode: sessions {[s[0] for s in sessions]}')
    print(f'[main] converting {len(sessions)} sessions '
          f'({sum(len(s[1]) for s in sessions)} imaging planes)')

    nplot = 2 if args.show_processing else 0
    if args.nsessions:
        sessions = sessions[:args.nsessions]
        print(f'[main] limiting to {len(sessions)} sessions')
    jobs = [(sid, eids, meta, i < nplot, args.neural_signal,
             (BIN_SIZE, OFF_START, OFF_END, NBINS), args.normalize)
            for i, (sid, eids, meta) in enumerate(sessions)]

    results = []
    if args.workers > 1 and len(jobs) > 1:
        with Pool(min(args.workers, len(jobs))) as pool:
            for i, r in enumerate(pool.imap_unordered(convert_session, jobs)):
                results.append(r)
                el = time.time() - t_start
                print(f'[{i+1}/{len(jobs)}] session {r["session_id"]} '
                      f'{"SKIP: " + str(r["skip"]) if r.get("skip") else "ok"} '
                      f'({el:.0f}s elapsed, {el/(i+1):.1f}s/session)', flush=True)
    else:
        for i, job in enumerate(jobs):
            r = convert_session(job)
            results.append(r)
            el = time.time() - t_start
            print(f'[{i+1}/{len(jobs)}] session {r["session_id"]} '
                  f'{"SKIP: " + str(r["skip"]) if r.get("skip") else "ok"} '
                  f'({el:.0f}s elapsed, {el/(i+1):.1f}s/session)', flush=True)

    results.sort(key=lambda r: r['session_id'])
    skipped = [(r['session_id'], r['skip']) for r in results if r.get('skip')]
    for sid, why in skipped:
        print(f'[skip] session {sid}: {why}')
        for r in results:
            if r['session_id'] == sid and 'traceback' in r:
                print(r['traceback'])
    good = [r for r in results if not r.get('skip')]
    print(f'[main] {len(good)} sessions converted, {len(skipped)} skipped')

    # ---------------- discretise running speed and pupil diameter ----------------
    # "five equal percentile bins": edges are the 20/40/60/80th percentiles, computed either
    # over all time bins of the whole dataset (global, default) or within each session.
    if args.quantile_scope == 'global':
        run_all = np.concatenate([r['run_binned'].ravel() for r in good])
        pupil_all = np.concatenate([r['pupil_binned'].ravel() for r in good])
        _, run_edges_g = quantile_bin(run_all)
        _, pupil_edges_g = quantile_bin(pupil_all)
        print(f'[quantile] global running-speed edges (cm/s): {np.round(run_edges_g, 4).tolist()}')
        print(f'[quantile] global pupil-diameter edges (px) : {np.round(pupil_edges_g, 3).tolist()}')
    for r in good:
        if args.quantile_scope == 'global':
            run_edges, pupil_edges = run_edges_g, pupil_edges_g
            run_cls = np.digitize(r['run_binned'], run_edges)
            pupil_cls = np.digitize(r['pupil_binned'], pupil_edges)
        else:
            run_cls, run_edges = quantile_bin(r['run_binned'].ravel())
            run_cls = run_cls.reshape(r['run_binned'].shape)
            pupil_cls, pupil_edges = quantile_bin(r['pupil_binned'].ravel())
            pupil_cls = pupil_cls.reshape(r['pupil_binned'].shape)
        for i, o in enumerate(r['output']):
            o[2] = run_cls[i]
            o[3] = pupil_cls[i]
        r['info']['running_quintile_edges'] = [float(x) for x in run_edges]
        r['info']['pupil_quintile_edges'] = [float(x) for x in pupil_edges]
        r['checks']['running_class_fracs'] = [float((run_cls == c).mean()) for c in range(NQUANTILES)]
        r['checks']['pupil_class_fracs'] = [float((pupil_cls == c).mean()) for c in range(NQUANTILES)]
        if '_plotargs' in r:
            plot_processing(r, run_edges, pupil_edges)
            del r['_plotargs']
        del r['run_binned'], r['pupil_binned']

    # ---------------- assemble the dataset ----------------
    image_names = good[0]['image_names']
    for r in good:
        assert r['image_names'] == image_names, 'image sets differ across sessions'
    subjects = sorted({r['mouse_id'] for r in good})
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    brain_regions = sorted({str(reg) for r in good for reg in set(r['regions'])})
    reg_to_idx = {s: i for i, s in enumerate(brain_regions)}

    data = dict(
        neural=[r['neural'] for r in good],
        input=[r['input'] for r in good],
        output=[r['output'] for r in good],
        subjects=subjects,
        subject_idx=np.array([subj_to_idx[r['mouse_id']] for r in good], dtype=np.int64),
        brain_regions=brain_regions,
        brain_region_idx=[np.array([reg_to_idx[str(x)] for x in r['regions']], dtype=np.int64)
                          for r in good],
        input_names=[],
        output_names=['image_identity', 'image_change', 'running_speed_quintile',
                      'pupil_diameter_quintile', 'trial_outcome'],
        output_values=[list(image_names),
                       ['no_change', 'change'],
                       [f'speed_q{i+1}' for i in range(NQUANTILES)],
                       [f'pupil_q{i+1}' for i in range(NQUANTILES)],
                       list(OUTCOME_NAMES)],
    )

    ntrials = [len(x) for x in data['neural']]
    nneurons = [x[0].shape[0] for x in data['neural']]
    data['metadata'] = dict(
        task_description=(
            'Allen Brain Observatory Visual Behavior 2P: head-fixed mice perform a go/no-go '
            'visual change-detection task with flashed natural images (250 ms image, 500 ms grey, '
            '750 ms cycle, 5% omissions) while 2-photon calcium imaging is performed in visual '
            'cortex. Trials are GO (image identity changes) or CATCH (sham change); aborted and '
            'auto-rewarded (free-reward) trials are excluded. Neural activity is the magnitude of '
            'detected calcium events, averaged in 250 ms bins. The decoder predicts, from neural '
            'activity alone: the identity of the currently presented image (8 classes), whether an '
            'image change just occurred (binary), the running speed and pupil diameter '
            'discretised into within-session quintiles, and the trial outcome '
            '(hit / miss / false alarm / correct reject, constant within a trial).'),
        time_bin_size=BIN_SIZE * 1000.0,
        temporal_alignment_event=('trials.change_time: onset of the image-change flash on GO '
                                  'trials, or of the sham change on CATCH trials (aligned to the '
                                  'ophys/sync clock; always coincides with a flash onset)'),
        off_start=OFF_START,
        off_end=OFF_END,
        n_time_bins=NBINS,
        neural_normalization=args.normalize,
        neural_signal=(f'detected calcium events (allensdk BehaviorOphysExperiment.events '
                       f'column "{args.neural_signal}"): mean event magnitude per time bin, '
                       f'then per-neuron normalisation within session ({args.normalize})'),
        dataset='visual-behavior-ophys-1.1.0 (Allen Brain Observatory Visual Behavior 2P)',
        data_selection=('active (behaving) sessions with the familiar image set (session_type '
                        'OPHYS_1_images_A / OPHYS_3_images_A); all simultaneously recorded imaging '
                        'planes of one ophys session merged into one session; only valid ROIs '
                        '(as returned by the AllenSDK); trials (go|catch) & ~auto_rewarded; '
                        'sessions without eye tracking excluded'),
        trial_types_included=['go', 'catch'],
        trial_types_excluded=['aborted', 'auto_rewarded'],
        n_sessions=len(good),
        n_trials_total=int(np.sum(ntrials)),
        n_neurons_total=int(np.sum(nneurons)),
        session_info=[r['info'] for r in good],
        session_checks=[r['checks'] for r in good],
        image_set='A (familiar)',
        quantile_scope=args.quantile_scope,
        output_notes=dict(
            image_identity='index into output_values[0]; held over the whole 750 ms image-presentation '
                           'interval; omitted flashes keep the identity of the repeating image',
            image_change='1 for the 750 ms image-presentation interval starting at the image change '
                         '(3 bins), 0 elsewhere; always 0 on CATCH (sham change) trials',
            running_speed_quintile='mean filtered running speed (cm/s) per bin, discretised at the '
                                   f'20/40/60/80th percentiles ({args.quantile_scope} scope)',
            pupil_diameter_quintile='pupil diameter = 2*sqrt(pupil_area/pi) px (SDK circular-area '
                                    'convention), blink NaNs interpolated across gaps <= 1 s, mean per '
                                    'bin, discretised at the '
                                    f'20/40/60/80th percentiles ({args.quantile_scope} scope)',
            trial_outcome='static per trial (broadcast over time bins)'),
    )

    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    # ---------------- summary ----------------
    print('\n================ conversion summary ================')
    print(f'sessions                : {len(good)}')
    print(f'subjects (mice)         : {len(subjects)}')
    print(f'brain regions           : {brain_regions}')
    print(f'neurons total           : {int(np.sum(nneurons))} '
          f'(min {np.min(nneurons)}, median {int(np.median(nneurons))}, max {np.max(nneurons)})')
    print(f'trials total            : {int(np.sum(ntrials))} '
          f'(min {np.min(ntrials)}, mean {np.mean(ntrials):.1f}, max {np.max(ntrials)})')
    dropped = sum(r['info']['n_trials_dropped_behavior'] for r in good)
    sel = sum(r['info']['n_trials_selected'] for r in good)
    print(f'trials dropped (missing running/pupil): {dropped} / {sel} ({100.0*dropped/max(sel,1):.1f}%)')
    allout = np.concatenate([np.stack(r['output']).transpose(1, 0, 2).reshape(5, -1) for r in good], axis=1)
    for k, nm in enumerate(data['output_names']):
        nc = len(data['output_values'][k])
        fr = [float((allout[k] == c).mean()) for c in range(nc)]
        print(f'output {k} {nm:26s} classes={nc} fractions={np.round(fr,4).tolist()}')
    ntr_out = np.array([[r['info'][k] for k in ('n_hit', 'n_miss', 'n_false_alarm', 'n_correct_reject')] for r in good]).sum(axis=0)
    print(f'trial outcomes (per trial): hit {ntr_out[0]}, miss {ntr_out[1]}, '
          f'false_alarm {ntr_out[2]}, correct_reject {ntr_out[3]} '
          f'-> fractions {np.round(ntr_out/ntr_out.sum(),4).tolist()}')
    # aggregate sanity checks
    print('\n---- sanity checks (across sessions) ----')
    keys = ['change_times_are_flash_onsets', 'bin0_image_is_change_image_go',
            'prebin_image_is_initial_image_go', 'change_pattern_go', 'change_zero_catch',
            'neural_finite', 'neural_nonneg']
    for k in keys:
        vals = [r['checks'][k] for r in good if r['checks'].get(k) is not None]
        print(f'{k:36s}: min={np.min(vals)} mean={np.mean(vals):.5f}')
    nm = np.array([r['checks']['neural_mean'] for r in good])
    print(f'{"neural_mean (event magnitude)":36s}: min={nm.min():.5f} mean={nm.mean():.5f} max={nm.max():.5f}')
    print(f'\ntotal time: {time.time()-t_start:.1f} s -> {args.outfile} '
          f'({os.path.getsize(args.outfile)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
