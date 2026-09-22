"""
Convert the Allen Brain Observatory *Visual Behavior 2P* dataset into the
decoder-ready dictionary format described in the task.

Overview of the decisions implemented here (see README-style notes inline):

SESSION SELECTION
  * Only the single-plane "VisualBehavior" 2-photon dataset variant is used
    (rigs CAM2P.3/4/5, `project_code == 'VisualBehavior'`).  Those experiments
    are all sampled at the same ophys frame rate (31 Hz, dt = 32.32 ms) and have
    exactly one imaging plane per ophys session, so "session == experiment" is
    unambiguous and the time bin is identical for every trial of every session,
    as the target format requires.  The Multiscope (multi-plane) experiments in
    the download are acquired at 10.7 Hz and therefore cannot share a common
    time bin with the single-plane data.
  * Only *active behavior* sessions are used (`passive == False`).  In passive
    sessions the lick spout is retracted, so there is no behavioral report and
    the trial-outcome output would be undefined (every trial is a miss /
    correct-reject by construction).
  * Both familiar and novel image sets are kept.  The reference paper restricted
    its neural analysis to familiar images because its scientific question was
    about behavioral strategy and novelty was a confound; that restriction is
    not relevant to the decoding task specified here, so all active sessions of
    the change-detection task are converted.

  * Three experiments (795953296, 806456687, 833631914) are dropped because the
    release contains no eye-tracking data for them, so the required pupil output
    cannot be produced.

NEURAL DATA
  * `dff_traces`, the detrended dF/F traces that the Allen pipeline produces for
    every ROI (demixing -> neuropil subtraction -> dF/F -> detrending; this is
    the neural processing the technical whitepaper documents, reproduced in full
    in methods.txt).
  * This is a deliberate departure from the Vip-Sst reference paper, which
    analysed the deconvolved calcium `events` instead.  That paper works at the
    resolution of 750 ms image-presentation intervals, where events (which are
    non-zero in only ~0.3 % of neuron x frame samples) are still informative.
    This decoder classifies every single ophys frame (32 ms), and at that
    resolution the event traces are nearly empty: converted with `events` /
    `filtered_events`, 1594 of 42467 trials (3.8 %) contain no activity at all
    in any neuron, which the format checker flags, and validation balanced
    accuracy drops for every output (e.g. image identity 0.40 -> 0.21, image
    change 0.61 -> 0.55, trial outcome 0.30 -> 0.28).  The task allows
    discrepancies from the reference paper where they are required by the task
    of training a neural decoder, and this is one: the dF/F trace carries the
    per-frame graded signal the per-time-bin decoder needs.  Set
    VB_TRACE=filtered_events to reproduce the event-based conversion.
  * All ROIs released in the NWB file are kept: the Allen pipeline has already
    applied its ROI filtering / demixing / neuropil-correction QC (every ROI in
    the release has `valid_roi == True`), and the reference paper applies no
    further neuron-level curation.

TEMPORAL ALIGNMENT
  * The ophys timestamps are the master clock, as instructed.  Time bins are the
    ophys frames themselves (~32.32 ms); the stimulus, running and pupil streams
    are aligned onto those frame times, so no resampling of the neural data is
    needed.  Running speed (60 Hz) and pupil diameter (30 Hz) are linearly
    interpolated onto the ophys frame times.

TRIALS
  * Trials are the change-detection trials defined by the experiment
    (`BehaviorOphysExperiment.trials`), from `start_time` to `stop_time`.
    "Go" and "Catch" trials are kept; "Aborted" and "Auto-rewarded" trials are
    dropped, as specified.  Trial length varies (the change time is drawn from a
    truncated exponential), so the number of time bins varies per trial; the bin
    size does not.

OUTPUTS (five categorical variables, all emitted as time series)
  1. image_identity - the image of the 750 ms "image presentation interval" that
     each time bin falls in, i.e. the image shown during the non-grey part of the
     flash cycle, held through the grey screen that follows it until the next
     flash.  This is the interval definition used by the reference paper
     ("By image presentation interval we refer to the 750 ms interval beginning
     with each image presentation.  For image omissions we used the 750 ms
     following the time of the omission").  Omitted flashes get their own
     category.  16 images (8 in image set A + 8 in set B) + 'omitted'.
  2. image_change - 1 through the presentation interval of the flash at which the
     image identity changed (go trials), 0 everywhere else.  Catch trials contain
     a sham change and therefore carry no 1s.
  3. running_speed_bin / 4. pupil_diameter_bin - quintiles of the running speed
     and of the pupil diameter.  The bin edges are the 20/40/60/80th percentiles
     of the pooled distribution over every time bin of the whole converted
     dataset, so a bin index means the same absolute range everywhere.
  5. trial_outcome - hit / miss / false_alarm / correct_reject, constant within a
     trial (go trials are hit or miss, catch trials false_alarm or
     correct_reject).  Held across the trial's time bins rather than stored as a
     1-D per-trial value, since the format asks for time-varying outputs where
     possible.

INPUTS
  * None, as specified: every trial carries a (0, n_time_bins) array.
"""

import argparse
import os
import pickle
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

DATA_ROOT = '/app/data/visual-behavior-ophys-1.1.0'
NWB_DIR = os.path.join(DATA_ROOT, 'behavior_ophys_experiments')
META_DIR = os.path.join(DATA_ROOT, 'project_metadata')
CACHE_DIR = '/app/_conversion_cache'
OUT_PATH = '/app/converted_data.pkl'

# Trace used as neural activity.  'dff' is the default (see the module
# docstring); 'events' / 'filtered_events' select the deconvolved event traces
# and are kept so the comparison behind that choice can be reproduced.
TRACE = os.environ.get('VB_TRACE', 'dff')
CACHE_DIR = CACHE_DIR + ('' if TRACE == 'dff' else '_' + TRACE)

_TRACE_DESCRIPTIONS = {
    'dff': ('dff: detrended dF/F of each segmented cell body, sampled at the '
            'ophys frame rate (Allen pipeline: motion correction -> '
            'segmentation -> demixing of overlapping ROIs -> neuropil '
            'subtraction -> dF/F against a 600 s median-filtered baseline -> '
            'detrending)'),
    'events': ('events: calcium events deconvolved from the detrended dF/F '
               'traces by the Allen pipeline, sampled at the ophys frame rate'),
    'filtered_events': ('filtered_events: calcium events deconvolved from the '
                        'detrended dF/F traces by the Allen pipeline, '
                        'convolved with a causal half-normal kernel '
                        '(sigma = 2/31 s), sampled at the ophys frame rate'),
}

# outcome categories (mutually exclusive and exhaustive for go/catch trials)
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']

NBINS = 5  # quintiles for running speed and pupil diameter


# --------------------------------------------------------------------------- #
# experiment selection
# --------------------------------------------------------------------------- #
def select_experiments():
    """Return the metadata rows of the experiments that will be converted."""
    exp = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    available = set()
    for fname in os.listdir(NWB_DIR):
        m = re.match(r'behavior_ophys_experiment_(\d+)\.nwb$', fname)
        if m:
            available.add(int(m.group(1)))
    exp = exp[exp.ophys_experiment_id.isin(available)]

    # single-plane "VisualBehavior" variant -> one common ophys frame rate and
    # one imaging plane per session
    exp = exp[exp.project_code == 'VisualBehavior']
    # active behavior only (passive sessions have no behavioral report)
    exp = exp[~exp.passive.astype(bool)]
    return exp.sort_values('ophys_experiment_id').reset_index(drop=True)


# --------------------------------------------------------------------------- #
# per-experiment conversion
# --------------------------------------------------------------------------- #
def _flash_labels(stim, ts):
    """Assign every ophys frame to the flash ("image presentation interval") it
    falls in.

    Following the reference paper, an image presentation interval is the 750 ms
    beginning with an image onset (and, for an omitted flash, the 750 ms during
    which the image should have appeared).  A frame is therefore labelled with
    the image of the most recent flash onset, so that the label covers the
    non-grey (250 ms image) part of the cycle *and* the grey screen that follows
    it, up to the next flash.

    Returns
        image_name: object array, per frame, None before the first flash
        is_change: bool array, per frame, True inside the flash at which the
                   image identity changed
    """
    starts = stim['start_time'].values
    # index of the most recent flash onset at or before each ophys frame
    idx = np.searchsorted(starts, ts, side='right') - 1
    valid = idx >= 0
    idx_clipped = np.where(valid, idx, 0)

    names = stim['image_name'].values.astype(object)
    changes = stim['is_change'].values.astype(bool)

    image_name = np.where(valid, names[idx_clipped], None)
    is_change = np.where(valid, changes[idx_clipped], False)
    return image_name, is_change


def _pupil_diameter(eye_tracking):
    """Pupil diameter (pixels) and its timestamps, blinks removed.

    The SDK already sets `pupil_area` to NaN on frames flagged `likely_blink`;
    those frames are dropped here and the signal is linearly interpolated onto
    the ophys clock by the caller.  Diameter is derived from the fitted pupil
    ellipse area as 2*sqrt(area/pi), which is more robust than either single
    ellipse axis.
    """
    if eye_tracking is None or len(eye_tracking) == 0:
        return None, None
    t = eye_tracking['timestamps'].values
    area = eye_tracking['pupil_area'].values.astype(float)
    good = np.isfinite(area) & np.isfinite(t) & (area > 0)
    if good.sum() < 100:
        return None, None
    return t[good], 2.0 * np.sqrt(area[good] / np.pi)


def convert_experiment(ophys_experiment_id):
    """Extract everything needed for one experiment; cache it to disk.

    Returns the path of the cache file, or None if the experiment has to be
    dropped.
    """
    cache_path = os.path.join(CACHE_DIR, f'{ophys_experiment_id}.pkl')
    if os.path.exists(cache_path):
        return cache_path

    from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
        BehaviorOphysExperiment,
    )

    nwb_path = os.path.join(
        NWB_DIR, f'behavior_ophys_experiment_{ophys_experiment_id}.nwb')
    ds = BehaviorOphysExperiment.from_nwb_path(nwb_path)
    md = ds.metadata

    ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)

    # ---- neural activity ------------------------------------------------- #
    if TRACE == 'dff':
        src = ds.dff_traces
        col = 'dff'
    else:
        src = ds.events
        col = TRACE
    cell_ids = src.index.values
    traces = np.vstack([np.asarray(v, dtype=np.float32)
                        for v in src[col].values])
    assert traces.shape[1] == ts.shape[0]
    # the released traces are finite, but guard against the occasional gap
    np.nan_to_num(traces, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- trials ---------------------------------------------------------- #
    trials = ds.trials
    keep = ((trials['go'].astype(bool) | trials['catch'].astype(bool))
            & ~trials['aborted'].astype(bool)
            & ~trials['auto_rewarded'].astype(bool))
    trials = trials[keep]
    if len(trials) < 2:
        return None

    outcome = np.full(len(trials), -1, dtype=np.int64)
    for i, name in enumerate(OUTCOME_NAMES):
        outcome[trials[name].astype(bool).values] = i
    if np.any(outcome < 0):
        # every go/catch trial is exactly one of hit/miss/FA/CR
        ok = outcome >= 0
        trials = trials[ok]
        outcome = outcome[ok]
        if len(trials) < 2:
            return None

    # ---- stimulus --------------------------------------------------------- #
    stim = ds.stimulus_presentations
    if 'stimulus_block_name' in stim.columns:
        stim = stim[stim['stimulus_block_name'].astype(str)
                    .str.contains('change_detection')]
    else:  # pragma: no cover - fallback for older files
        stim = stim[stim['active'].astype(bool)]
    stim = stim.sort_values('start_time')
    image_name, is_change = _flash_labels(stim, ts)

    # ---- running speed ---------------------------------------------------- #
    run = ds.running_speed
    run_t = np.asarray(run['timestamps'].values, dtype=np.float64)
    run_v = np.asarray(run['speed'].values, dtype=np.float64)
    good = np.isfinite(run_t) & np.isfinite(run_v)
    running = np.interp(ts, run_t[good], run_v[good])

    # ---- pupil ------------------------------------------------------------ #
    pt, pv = _pupil_diameter(ds.eye_tracking)
    if pt is None:
        # without pupil data the session cannot supply one of the required
        # outputs, so it is dropped
        return None
    pupil = np.interp(ts, pt, pv)

    # ---- slice into trials ------------------------------------------------ #
    starts = np.searchsorted(ts, trials['start_time'].values, side='left')
    stops = np.searchsorted(ts, trials['stop_time'].values, side='left')

    neural_trials = []
    image_trials = []
    change_trials = []
    running_trials = []
    pupil_trials = []
    outcome_trials = []
    for k in range(len(trials)):
        i0, i1 = int(starts[k]), int(stops[k])
        if i1 - i0 < 2:
            continue
        if np.any(image_name[i0:i1] == None):  # noqa: E711 - before 1st flash
            continue
        neural_trials.append(np.ascontiguousarray(traces[:, i0:i1]))
        image_trials.append(image_name[i0:i1].copy())
        change_trials.append(is_change[i0:i1].astype(np.int64))
        running_trials.append(running[i0:i1].astype(np.float64))
        pupil_trials.append(pupil[i0:i1].astype(np.float64))
        outcome_trials.append(int(outcome[k]))

    if len(neural_trials) < 2:
        return None

    out = {
        'ophys_experiment_id': int(ophys_experiment_id),
        'ophys_session_id': int(md['ophys_session_id']),
        'behavior_session_id': int(md['behavior_session_id']),
        'mouse_id': str(md['mouse_id']),
        'cre_line': str(md['cre_line']),
        'session_type': str(md['session_type']),
        'equipment_name': str(md['equipment_name']),
        'targeted_structure': str(md['targeted_structure']),
        'imaging_depth': int(md['imaging_depth']),
        'ophys_frame_rate': float(md['ophys_frame_rate']),
        'dt': float(np.median(np.diff(ts))),
        'cell_specimen_ids': np.asarray(cell_ids),
        'neural': neural_trials,
        'image_name': image_trials,
        'is_change': change_trials,
        'running_speed': running_trials,
        'pupil_diameter': pupil_trials,
        'trial_outcome': np.asarray(outcome_trials, dtype=np.int64),
    }
    tmp = cache_path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(out, f, protocol=4)
    os.replace(tmp, cache_path)
    return cache_path


def _worker(eid):
    try:
        return eid, convert_experiment(eid), None
    except Exception:  # pragma: no cover
        import traceback
        return eid, None, traceback.format_exc()


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def assemble(cache_paths):
    sessions = []
    for p in cache_paths:
        with open(p, 'rb') as f:
            sessions.append(pickle.load(f))
    sessions.sort(key=lambda s: s['ophys_experiment_id'])

    # --- global quintile edges for running speed and pupil diameter -------- #
    # "five equal percentile bins" are taken over the pooled distribution of all
    # converted time bins, so that a bin index means the same absolute range of
    # running speed / pupil diameter in every session.
    run_all = np.concatenate([np.concatenate(s['running_speed']) for s in sessions])
    pup_all = np.concatenate([np.concatenate(s['pupil_diameter']) for s in sessions])
    qs = np.linspace(0, 100, NBINS + 1)[1:-1]
    run_edges = np.percentile(run_all, qs)
    pup_edges = np.percentile(pup_all, qs)
    print('running speed quintile edges (cm/s):', np.round(run_edges, 3))
    print('pupil diameter quintile edges (pix):', np.round(pup_edges, 3))
    del run_all, pup_all

    # --- categorical vocabularies ------------------------------------------ #
    images = sorted({name for s in sessions for tr in s['image_name']
                     for name in np.unique(tr)} - {'omitted'})
    image_values = images + ['omitted']
    image_to_idx = {name: i for i, name in enumerate(image_values)}

    subjects = sorted({s['mouse_id'] for s in sessions})
    subject_to_idx = {m: i for i, m in enumerate(subjects)}

    brain_regions = sorted({s['targeted_structure'] for s in sessions})
    region_to_idx = {r: i for i, r in enumerate(brain_regions)}

    neural, inputs, outputs = [], [], []
    subject_idx, brain_region_idx, session_info = [], [], []

    for s in sessions:
        ntrials = len(s['neural'])
        nneurons = s['neural'][0].shape[0]
        sess_neural, sess_input, sess_output = [], [], []
        for k in range(ntrials):
            T = s['neural'][k].shape[1]
            img = np.array([image_to_idx[n] for n in s['image_name'][k]],
                           dtype=np.int64)
            chg = s['is_change'][k].astype(np.int64)
            run = np.searchsorted(run_edges, s['running_speed'][k],
                                  side='right').astype(np.int64)
            pup = np.searchsorted(pup_edges, s['pupil_diameter'][k],
                                  side='right').astype(np.int64)
            out = np.full(T, s['trial_outcome'][k], dtype=np.int64)
            sess_neural.append(np.asarray(s['neural'][k], dtype=np.float32))
            sess_input.append(np.zeros((0, T), dtype=np.float32))
            sess_output.append(np.stack([img, chg, run, pup, out], axis=0))
        neural.append(sess_neural)
        inputs.append(sess_input)
        outputs.append(sess_output)
        subject_idx.append(subject_to_idx[s['mouse_id']])
        brain_region_idx.append(
            np.full(nneurons, region_to_idx[s['targeted_structure']], dtype=np.int64))
        session_info.append({
            'ophys_experiment_id': s['ophys_experiment_id'],
            'ophys_session_id': s['ophys_session_id'],
            'behavior_session_id': s['behavior_session_id'],
            'mouse_id': s['mouse_id'],
            'cre_line': s['cre_line'],
            'session_type': s['session_type'],
            'equipment_name': s['equipment_name'],
            'targeted_structure': s['targeted_structure'],
            'imaging_depth': s['imaging_depth'],
            'ophys_frame_rate': s['ophys_frame_rate'],
            'n_neurons': int(nneurons),
            'n_trials': int(ntrials),
        })

    dts = np.array([s['dt'] for s in sessions])
    time_bin_size = float(np.median(dts) * 1000.0)

    def _edge_labels(edges, unit):
        lab = [f'<{edges[0]:.3g}{unit}']
        for a, b in zip(edges[:-1], edges[1:]):
            lab.append(f'{a:.3g}-{b:.3g}{unit}')
        lab.append(f'>{edges[-1]:.3g}{unit}')
        return lab

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
        'subject_idx': np.asarray(subject_idx, dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': [],
        'output_names': ['image_identity', 'image_change',
                         'running_speed_bin', 'pupil_diameter_bin',
                         'trial_outcome'],
        'output_values': [
            image_values,
            ['no_change', 'change'],
            _edge_labels(run_edges, 'cm/s'),
            _edge_labels(pup_edges, 'pix'),
            OUTCOME_NAMES,
        ],
        'metadata': {
            'task_description': (
                'Allen Brain Observatory Visual Behavior 2-photon change '
                'detection task. Head-fixed mice view a continuous series of '
                'flashed natural images (250 ms image, 500 ms grey screen) and '
                'are rewarded for licking when the image identity changes. '
                'Trials are the change-detection trials defined by the '
                'experiment; only Go (real change) and Catch (sham change) '
                'trials are included, Aborted and Auto-rewarded trials are '
                'excluded. Nothing is given to the decoder as input; it must '
                'predict, from the dF/F activity of the simultaneously '
                'recorded population, (1) the identity of the image of the '
                'current 750 ms flash cycle (including omitted flashes), '
                '(2) whether the current flash is the image change, (3) the '
                'quintile of the running speed, (4) the quintile of the pupil '
                'diameter, and (5) the behavioral outcome of the trial.'),
            'time_bin_size': time_bin_size,
            'temporal_alignment_event': (
                'trial start_time (onset of the change-detection trial, i.e. '
                'the first ophys frame at or after trials.start_time); all data '
                'streams are sampled on the ophys frame clock'),
            'off_start': 0.0,
            'off_end': None,
            'trial_window_note': (
                'Each trial spans trials.start_time to trials.stop_time, so the '
                'number of time bins varies from trial to trial (the change time '
                'is drawn from a truncated exponential). off_end is therefore '
                'None.'),
            'neural_data_type': _TRACE_DESCRIPTIONS[TRACE],
            'brain_region_note': (
                'Allen CCF names of the targeted visual areas; VISp = primary '
                'visual cortex (V1). The single-plane Visual Behavior rigs in '
                'this release only targeted VISp, so every neuron is in VISp.'),
            'running_speed_bin_edges_cm_per_s': run_edges.tolist(),
            'pupil_diameter_bin_edges_pixels': pup_edges.tolist(),
            'dataset': 'visual-behavior-ophys-1.1.0',
            'project_code': 'VisualBehavior (single-plane 2-photon)',
            'session_info': session_info,
        },
    }
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--limit', type=int, default=None,
                    help='convert only the first N experiments (for testing)')
    ap.add_argument('--out', type=str, default=OUT_PATH)
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    exp_table = select_experiments()
    eids = exp_table.ophys_experiment_id.tolist()
    if args.limit:
        eids = eids[:args.limit]
    print(f'{len(eids)} experiments selected '
          f'({exp_table.mouse_id.nunique()} mice)', flush=True)

    paths, dropped = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_worker, e) for e in eids]
        for i, fut in enumerate(as_completed(futs)):
            eid, path, err = fut.result()
            if err is not None:
                print(f'[{i + 1}/{len(eids)}] {eid} FAILED\n{err}', flush=True)
                dropped.append(eid)
            elif path is None:
                print(f'[{i + 1}/{len(eids)}] {eid} dropped', flush=True)
                dropped.append(eid)
            else:
                paths.append(path)
            if (i + 1) % 20 == 0:
                print(f'  ... {i + 1}/{len(eids)} done', flush=True)

    print(f'converted {len(paths)} experiments, dropped {len(dropped)}: {dropped}',
          flush=True)

    data = assemble(sorted(paths))
    print('assembled; writing pickle ...', flush=True)
    with open(args.out, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f'wrote {args.out} '
          f'({os.path.getsize(args.out) / 1e9:.2f} GB)', flush=True)


if __name__ == '__main__':
    main()
