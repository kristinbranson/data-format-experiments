"""Convert Allen Institute Visual Behavior 2P data into decoder-ready format.

Design decisions
----------------
* Task: the change-detection (Visual Behavior) task; only active behavior sessions
  are used (in passive sessions the lick spout is retracted, so there are no
  choices, rewards or trial outcomes to decode).
* Experience level: familiar image set only, following the reference paper, which
  restricted its neural analysis to familiar stimuli.  Every familiar active session
  in this release used image set A, so the eight image identities are the same in
  every session and share one set of output labels.
* Sessions: one converted session per ophys session.  For Multiscope (multi-plane)
  sessions the simultaneously imaged planes are merged into a single session, with
  each neuron labeled by the brain region of its plane (VISp / VISl).
* Trials: rows of the SDK trials table with go == True or catch == True.  This
  excludes the aborted and auto-rewarded trials by construction.
* Temporal alignment: trials are aligned to the change time (the sham change time on
  catch trials) and binned on a fixed 250 ms grid (the duration of one image flash)
  with a bin edge exactly at the change time, so bin edges are locked to the
  stimulus flash cycle.  The grid is shared by all data streams; each stream is
  binned from its own native timestamps (ophys frame times for the neural data,
  stimulus times for the images, behavior/eye camera times for running and pupil).
  A fixed bin also makes single-plane (~31 Hz) and Multiscope (~11 Hz) recordings
  commensurate.
* Neural data: detected calcium events (discrete events regressed from the df/f
  traces by the AllenSDK event detection), as used in the reference paper, summed
  within each time bin.
* Stimulus variables follow the reference paper's convention of assigning data to
  the 750 ms image presentation interval (250 ms flash + 500 ms gray screen), so the
  image identity is held across the gray screen and across omitted flashes.
"""

import argparse
import glob
import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

CACHE_DIR = '/app/data'
BIN_SIZE = 0.25          # s, duration of one image flash
FLASH_INTERVAL = 0.75    # s, image presentation interval (250 ms image + 500 ms gray)
NQUANTILES = 5
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']


def get_experiment_table():
    tbl = pd.read_csv(os.path.join(
        CACHE_DIR, 'visual-behavior-ophys-1.1.0', 'project_metadata',
        'ophys_experiment_table.csv'))
    have = [int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in glob.glob(
        os.path.join(CACHE_DIR, 'visual-behavior-ophys-1.1.0',
                     'behavior_ophys_experiments', '*.nwb'))]
    return tbl[tbl.ophys_experiment_id.isin(have)]


def select_sessions():
    """Return {ophys_session_id: [ophys_experiment_id, ...]} for the sessions used."""
    tbl = get_experiment_table()
    sel = tbl[(tbl.behavior_type == 'active_behavior') &      # mouse doing the task
              (tbl.experience_level == 'Familiar')]           # familiar image set
    sel = sel.sort_values(['ophys_session_id', 'ophys_experiment_id'])
    return sel.groupby('ophys_session_id').ophys_experiment_id.apply(list).to_dict()


def interp_nans(x):
    """Linearly interpolate NaNs (blinks / lost tracking) in a 1d time series."""
    x = np.asarray(x, dtype=float)
    bad = ~np.isfinite(x)
    if bad.all():
        return None
    if bad.any():
        idx = np.arange(len(x))
        x = x.copy()
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def quantile_bins(values, nbins=NQUANTILES):
    """Discretize into nbins equal-percentile bins; returns int codes 0..nbins-1."""
    edges = np.quantile(values, np.linspace(0, 1, nbins + 1)[1:-1])
    return np.searchsorted(edges, values, side='right').astype(np.int64)


def bin_sum(values_cumsum, timestamps, edges):
    """Sum of values within each bin of `edges`, plus the sample count per bin."""
    idx = np.searchsorted(timestamps, edges)
    if values_cumsum.ndim == 1:
        return values_cumsum[idx[1:]] - values_cumsum[idx[:-1]], np.diff(idx)
    return (values_cumsum[:, idx[1:]] - values_cumsum[:, idx[:-1]]), np.diff(idx)


def convert_session(osid, oeids, neural_key='events'):
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorOphysProjectCache)
    cache = VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=CACHE_DIR)

    planes = []
    for oeid in oeids:
        ds = cache.get_behavior_ophys_experiment(int(oeid))
        ts = np.asarray(ds.ophys_timestamps)
        ev = ds.events
        if len(ev) == 0:
            continue
        act = np.vstack(ev[neural_key].values).astype(np.float64)
        assert act.shape[1] == len(ts)
        planes.append({
            'oeid': int(oeid), 'ds': ds, 'ts': ts,
            'cum': np.concatenate([np.zeros((act.shape[0], 1)),
                                   np.cumsum(act, axis=1)], axis=1),
            'ncells': act.shape[0],
            'cell_ids': [int(c) for c in ev.index.values],
            'region': str(ds.metadata['targeted_structure']),
            'depth': int(ds.metadata['imaging_depth']),
            'frame_rate': float(1.0 / np.median(np.diff(ts))),
        })
    if not planes:
        return None

    ds = planes[0]['ds']     # behavior streams are shared by all planes of a session

    # ---- behavior: running speed and pupil diameter on their native clocks ----
    run = ds.running_speed
    run_t = run.timestamps.values
    run_v = interp_nans(run.speed.values)
    if run_v is None:
        return None
    run_cum = np.concatenate([[0.0], np.cumsum(run_v)])

    try:
        eye = ds.eye_tracking
    except Exception:
        eye = None
    if eye is None or len(eye) == 0:
        return None              # no eye tracking -> pupil output cannot be made
    pupil_area = interp_nans(eye.pupil_area.values)     # NaN on likely-blink frames
    if pupil_area is None:
        return None
    pupil_v = 2.0 * np.sqrt(np.maximum(pupil_area, 0.0) / np.pi)   # diameter, pixels
    eye_t = eye.timestamps.values
    pupil_cum = np.concatenate([[0.0], np.cumsum(pupil_v)])

    # ---- stimulus presentations of the change detection block ----
    sp = ds.stimulus_presentations
    sp = sp[sp.stimulus_block_name == 'change_detection_behavior']
    image_names = sorted(set(sp.image_name.unique()) - {'omitted'})
    shown = sp[(~sp.omitted) & (sp.image_name != 'omitted')].sort_values('start_time')
    pres_start = shown.start_time.values
    pres_img = np.array([image_names.index(n) for n in shown.image_name.values])
    change_start = pres_start[shown.is_change.values]

    # ---- trials: go and catch only (excludes aborted and auto-rewarded) ----
    tr = ds.trials
    tr = tr[(tr.go | tr.catch) & tr.change_time.notna()]

    tmin = max([p['ts'][0] for p in planes] + [run_t[0], eye_t[0]])
    tmax = min([p['ts'][-1] for p in planes] + [run_t[-1], eye_t[-1]])

    trials = []
    for _, trial in tr.iterrows():
        oc = [k for k in OUTCOME_NAMES if bool(trial[k])]
        if len(oc) != 1:
            continue
        # fixed 250 ms bins with an edge exactly at the (sham) change time
        k0 = int(np.ceil((trial.start_time - trial.change_time) / BIN_SIZE))
        k1 = int(np.floor((trial.stop_time - trial.change_time) / BIN_SIZE))
        edges = trial.change_time + BIN_SIZE * np.arange(k0, k1 + 1)
        if len(edges) < 3 or edges[0] < tmin or edges[-1] > tmax:
            continue             # trial not fully covered by all data streams
        centers = 0.5 * (edges[:-1] + edges[1:])

        # neural: calcium events summed within each bin, planes stacked
        neural_parts, ok = [], True
        for p in planes:
            vals, nfr = bin_sum(p['cum'], p['ts'], edges)
            if np.any(nfr < 1):
                ok = False
                break
            neural_parts.append(vals)
        if not ok:
            continue
        neural_trial = np.vstack(neural_parts).astype(np.float32)

        # behavior: mean within each bin
        rsum, rn = bin_sum(run_cum, run_t, edges)
        psum, pn = bin_sum(pupil_cum, eye_t, edges)
        if np.any(rn < 1) or np.any(pn < 1):
            continue
        run_trial = rsum / rn
        pupil_trial = psum / pn

        # stimulus: identity of the most recent image flash, held over the 750 ms
        # image presentation interval and over omitted flashes
        j = np.searchsorted(pres_start, centers, side='right') - 1
        if np.any(j < 0):
            continue
        image_trial = pres_img[j].astype(np.int64)
        # image change: 1 during the presentation interval of the changed image
        jc = np.searchsorted(change_start, centers, side='right') - 1
        change_trial = np.zeros(len(centers), dtype=np.int64)
        good = jc >= 0
        change_trial[good] = (
            (centers[good] - change_start[jc[good]]) < FLASH_INTERVAL).astype(np.int64)

        trials.append({'neural': neural_trial, 'run': run_trial, 'pupil': pupil_trial,
                       'image': image_trial, 'change': change_trial,
                       'outcome': OUTCOME_NAMES.index(oc[0])})

    if len(trials) < 2:
        return None

    # Running speed and pupil diameter are discretized into five equal-percentile
    # bins computed within each session, over the time bins included in the trials.
    # Mice differ strongly in running and pupil statistics and pupil size is measured
    # in camera pixels, so its scale is session specific; a session-wise quantization
    # is therefore the meaningful one.
    run_code = quantile_bins(np.concatenate([t['run'] for t in trials]))
    pupil_code = quantile_bins(np.concatenate([t['pupil'] for t in trials]))

    neural, output = [], []
    pos = 0
    for t in trials:
        T = t['neural'].shape[1]
        neural.append(np.ascontiguousarray(t['neural']))
        output.append(np.stack([
            t['image'],
            t['change'],
            run_code[pos:pos + T],
            pupil_code[pos:pos + T],
            np.full(T, t['outcome'], dtype=np.int64),
        ]).astype(np.int64))
        pos += T

    regions = sum([[p['region']] * p['ncells'] for p in planes], [])
    md = ds.metadata
    info = {
        'ophys_session_id': int(osid),
        'ophys_experiment_ids': [p['oeid'] for p in planes],
        'behavior_session_id': int(md['behavior_session_id']),
        'mouse_id': str(md['mouse_id']),
        'cre_line': str(md['cre_line']),
        'session_type': str(md['session_type']),
        'equipment_name': str(md['equipment_name']),
        'project_code': str(md['project_code']),
        'imaging_depths': [p['depth'] for p in planes],
        'ophys_frame_rates': [p['frame_rate'] for p in planes],
        'n_planes': len(planes),
        'n_neurons': len(regions),
        'n_trials': len(neural),
        'cell_specimen_ids': sum([p['cell_ids'] for p in planes], []),
    }
    return {'neural': neural, 'output': output, 'info': info,
            'regions': regions, 'image_names': image_names}


def _worker(args):
    osid, oeids, neural_key = args
    try:
        return convert_session(osid, oeids, neural_key)
    except Exception as exc:
        print('FAILED session %s: %s: %s' % (osid, type(exc).__name__, exc), flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/app/converted_data.pkl')
    ap.add_argument('--neural', default='events', choices=['events', 'filtered_events'])
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()

    sessions = select_sessions()
    items = [(osid, oeids, args.neural) for osid, oeids in sessions.items()]
    if args.limit:
        items = items[:args.limit]
    print('converting %d ophys sessions' % len(items), flush=True)

    from multiprocessing import Pool
    with Pool(args.workers) as pool:
        results = pool.map(_worker, items)
    results = [r for r in results if r is not None]
    print('kept %d sessions' % len(results), flush=True)

    image_names = results[0]['image_names']
    assert all(r['image_names'] == image_names for r in results), 'image sets differ'

    subjects = sorted({r['info']['mouse_id'] for r in results})
    brain_regions = sorted({reg for r in results for reg in r['regions']})

    data = {
        'neural': [r['neural'] for r in results],
        'input': [[np.zeros((0, t.shape[1]), dtype=np.float32) for t in r['neural']]
                  for r in results],
        'output': [r['output'] for r in results],
        'subjects': subjects,
        'subject_idx': np.array([subjects.index(r['info']['mouse_id'])
                                 for r in results], dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': [np.array([brain_regions.index(reg) for reg in r['regions']],
                                      dtype=np.int64) for r in results],
        'input_names': [],
        'output_names': ['image_identity', 'image_change', 'running_speed_bin',
                         'pupil_diameter_bin', 'trial_outcome'],
        'output_values': [
            list(image_names),
            ['no_change', 'change'],
            ['running_quintile_%d' % (i + 1) for i in range(NQUANTILES)],
            ['pupil_quintile_%d' % (i + 1) for i in range(NQUANTILES)],
            list(OUTCOME_NAMES),
        ],
        'metadata': {
            'task_description': (
                'Allen Brain Observatory Visual Behavior 2-photon change detection task. '
                'Head-fixed mice viewed a continuous series of natural images (250 ms '
                'flashes separated by 500 ms of gray screen) and earned water rewards by '
                'licking when the image identity changed. Trials are the go (real change) '
                'and catch (sham change) trials of each session; aborted and auto-rewarded '
                'trials are excluded. From the calcium-event activity of visual cortex '
                'neurons the decoder predicts the identity of the image currently being '
                'presented, whether an image change just occurred, quintile-binned running '
                'speed, quintile-binned pupil diameter, and the behavioral outcome of the '
                'trial.'),
            'time_bin_size': BIN_SIZE * 1000.0,
            'temporal_alignment_event': (
                'image change time of the trial (sham change time on catch trials): the '
                '250 ms time bins are laid out with an edge exactly at the change time, '
                'and every data stream (ophys calcium events, stimulus, running, pupil) is '
                'binned from its own native timestamps onto that common grid'),
            'off_start': -3.75,   # typical (median) trial start relative to the change
            'off_end': 4.0,       # typical (median) trial end relative to the change
            'off_start_range': [-11.0, -3.0],
            'off_end_range': [3.75, 4.25],
            'trial_definition': (
                'each trial covers the 250 ms bins fully inside [start_time, stop_time) of '
                'a go or catch trial of the SDK trials table; the change (or sham change) '
                'occurs 3-7 s after trial start and the trial ends 4.25 s later, so the '
                'number of bins varies between trials (median 31 bins of 250 ms)'),
            'neural_data_type': (
                '%s: detected calcium events extracted from the df/f traces by the '
                'AllenSDK event detection, summed within each 250 ms time bin' % args.neural),
            'data_source': 'Allen Institute Visual Behavior Ophys dataset, version 1.1.0',
            'selection_criteria': (
                'active behavior (non-passive) sessions with the familiar image set (image '
                'set A, session types OPHYS_1_images_A and OPHYS_3_images_A), from both the '
                'single-plane and the Multiscope rigs; the imaging planes of a Multiscope '
                'session are merged into one session; sessions without eye tracking are '
                'dropped; all ROIs in the released data already passed the Allen Institute '
                'cell segmentation and ROI-filtering QC'),
            'output_descriptions': {
                'image_identity': 'identity of the natural image currently presented, held '
                                  'over the 750 ms image presentation interval and over '
                                  'omitted flashes (8 classes, image set A)',
                'image_change': '1 during the 750 ms presentation interval of an image that '
                                'differs from the preceding one, 0 otherwise',
                'running_speed_bin': 'running speed (cm/s) on the running disk averaged in '
                                     'each time bin, discretized into five equal-percentile '
                                     'bins computed per session',
                'pupil_diameter_bin': 'pupil diameter (2*sqrt(pupil_area/pi) in camera '
                                      'pixels, blinks linearly interpolated) averaged in '
                                      'each time bin, discretized into five equal-percentile '
                                      'bins computed per session',
                'trial_outcome': 'hit / miss (go trials) or false_alarm / correct_reject '
                                 '(catch trials), constant within a trial',
            },
            'session_info': [r['info'] for r in results],
        },
    }

    with open(args.out, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    ntrials = sum(len(x) for x in data['neural'])
    nneurons = sum(x[0].shape[0] for x in data['neural'])
    print('saved %s: %d sessions, %d trials, %d neurons, %d mice'
          % (args.out, len(data['neural']), ntrials, nneurons, len(subjects)))


if __name__ == '__main__':
    main()
