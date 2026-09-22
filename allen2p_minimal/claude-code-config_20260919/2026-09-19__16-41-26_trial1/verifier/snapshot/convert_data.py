"""
Convert the Allen Brain Observatory Visual Behavior 2-photon dataset into the
trial-structured format expected by train_decoder.py.

Dataset / task
--------------
Head-fixed mice perform a go/no-go change detection task while 2-photon calcium
imaging is performed in visual cortex ("Visual Behavior" task, see
whitepaper.pdf and Garrett et al.).  A continuous series of natural images is
flashed (250 ms image, 500 ms gray, i.e. a 750 ms image-presentation cycle) and
the mouse earns water by licking when the image identity changes.

Decisions (see README-style notes in each section):

Sessions
  * project_code == 'VisualBehavior'  -- the single-plane 2p recordings of the
    Visual Behavior task.  Every experiment of this project code is present in
    /app/data.  (The handful of VisualBehaviorMultiscope experiments that are
    also present come from a single mouse, use a different imaging rig and a
    ~3x slower per-plane frame rate; mixing them in would break the "same time
    bin size for all sessions" requirement, so they are not used.)
  * passive sessions (OPHYS_2/OPHYS_5) are excluded: the lick spout is
    retracted and the mouse is satiated, so the "trial outcome" output that the
    decoder has to predict is not defined behaviourally (every go trial is a
    miss by construction).  The reference paper likewise analyses only active
    behavior sessions.
  * both familiar and novel image sets are kept.  The reference paper
    restricted itself to familiar images because its scientific question was
    about novelty; nothing in the decoding task requires that restriction and
    the novel sessions roughly double the amount of data and contribute the
    second image set.
  * for each session all cells contained in the released NWB file are used.
    The published files only contain ROIs that passed the Allen ROI-filtering /
    QC pipeline (valid_roi == True), so no further neuron curation is applied.

Trials
  * the experiment's own trial definition is used (BehaviorOphysExperiment.trials).
    Only "go" and "catch" trials are kept; "aborted" trials (mouse licked
    before the change, no stimulus change happened) and "auto_rewarded" trials
    (free rewards at session start / after 10 consecutive misses) are dropped,
    as requested.
  * each trial is aligned to its change time: the onset of the changed image
    flash on go trials, and the onset of the sham-change flash on catch trials
    (trials.change_time holds both, and it coincides exactly with the onset of
    the corresponding stimulus presentation).  A fixed window of [-2, +2] s
    around that event is taken.  Every go/catch trial is at least 3.0 s long
    before the change and 4.2 s long after it, so this window always lies
    inside the experiment-defined trial.

Time base
  * the ophys frames are the time base ("temporally align based on ophys
    timestamp"): a trial is the T = round(4 s / frame interval) = 124
    consecutive ophys frames starting at the first frame at/after
    change_time - 2 s.  All single-plane sessions run at 31 Hz
    (bin = 32.3 ms), so T and the bin size are identical for every trial of
    every session.  All other data streams are resampled onto these frame
    times.

Neural data
  * the detected calcium events ("filtered_events" in the SDK events table),
    as used for all neural analysis in the reference paper (Piet et al.).  The
    events are the L0-regularised deconvolution of the dF/F traces, which
    removes the slow GCaMP6f decay.  The SDK's half-gaussian-filtered version
    of the very same events is used because the raw event train is nonzero in
    only ~0.1% of the 32 ms bins, which carries almost no signal at the
    single-bin resolution at which this decoder has to make a prediction.

Outputs (all as a (5, T) integer array per trial)
  0 image_identity  which of the 16 natural images (8 in set A, 8 in set B) is
                    currently being shown.  Held over the whole 750 ms
                    image-presentation cycle, i.e. also over the gray period
                    that follows the 250 ms flash and over omitted flashes
                    (5% of flashes are omitted; an omission is a continuation
                    of the gray screen and is never a change or a pre-change
                    flash, so the withheld image is always the current image).
  1 image_change    1 during the 750 ms image-presentation cycle that starts
                    with a real image change, 0 elsewhere.  Sham changes on
                    catch trials are 0 (the image does not change).
  2 running_speed   running speed (cm/s) resampled to the ophys frame times and
                    discretised into 5 equal-count percentile bins.
  3 pupil_diameter  pupil diameter (2*sqrt(pupil_area/pi), in camera pixels)
                    from the eye-tracking ellipse fit, blinks removed and
                    linearly interpolated, resampled to the ophys frame times
                    and discretised into 5 equal-count percentile bins.
  4 trial_outcome   hit / miss / false_alarm / correct_reject, constant over
                    the trial.
  The percentile bin edges for running speed and pupil diameter are computed
  once over all trials of all sessions, so that a given bin index means the
  same physical value in every session (the decoder shares one output head
  across sessions).

Inputs
  * none, as specified by the decoder task.  input[session][trial] is an
    empty (0, T) array.

Usage:  python convert_data.py [--out /app/converted_data.pkl] [--workers 12]
"""

import argparse
import os
import pickle
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

DATA_DIR = '/app/data/visual-behavior-ophys-1.1.0'
EXPERIMENT_DIR = os.path.join(DATA_DIR, 'behavior_ophys_experiments')
METADATA_DIR = os.path.join(DATA_DIR, 'project_metadata')
CACHE_DIR = '/app/cache_sessions'

PROJECT_CODE = 'VisualBehavior'

# trial window relative to the (sham) change time, in seconds
OFF_START = -2.0
OFF_END = 2.0

# duration of one image presentation cycle (250 ms image + 500 ms gray)
IMAGE_CYCLE = 0.75

# acceptable ophys frame rate range (single plane imaging runs at 31 Hz)
MIN_FRAME_RATE = 30.0
MAX_FRAME_RATE = 32.0

# number of percentile bins for the continuous behavioral outputs
NBINS_BEHAVIOR = 5

OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']


def session_table():
    """Table of the sessions to convert, one row per session."""
    et = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    available = set()
    for fn in os.listdir(EXPERIMENT_DIR):
        if fn.endswith('.nwb'):
            available.add(int(fn.split('_')[-1].split('.')[0]))
    et = et[et.ophys_experiment_id.isin(available)]
    et = et[(et.project_code == PROJECT_CODE) & (~et.passive.astype(bool))]
    # single plane imaging: one imaging plane (experiment) per session
    assert et.ophys_session_id.nunique() == len(et)
    et = et.sort_values('ophys_experiment_id').reset_index(drop=True)
    return et


def extract_session(oeid):
    """Pull the trial-aligned data out of one session's NWB file.

    Returns a dict with the per-trial neural activity and the raw (not yet
    discretised) output variables, or a dict with 'error' set.
    """
    from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
        BehaviorOphysExperiment)

    path = os.path.join(EXPERIMENT_DIR,
                        f'behavior_ophys_experiment_{oeid}.nwb')
    ds = BehaviorOphysExperiment.from_nwb_path(path)

    # ---- time base: the ophys frames -------------------------------------
    ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
    dt = float(np.median(np.diff(ts)))
    frame_rate = 1.0 / dt
    if not (MIN_FRAME_RATE <= frame_rate <= MAX_FRAME_RATE):
        return {'oeid': oeid, 'error': f'frame rate {frame_rate:.2f} Hz'}
    T = int(round((OFF_END - OFF_START) / dt))

    # ---- neural activity: detected calcium events ------------------------
    events = ds.events
    neural_all = np.stack(events['filtered_events'].values).astype(np.float32)
    if neural_all.shape[1] != len(ts):
        return {'oeid': oeid, 'error': 'events / timestamps length mismatch'}
    if neural_all.shape[0] == 0:
        return {'oeid': oeid, 'error': 'no cells'}
    cell_specimen_ids = np.asarray(events.index.values)

    # ---- stimulus: image identity and image changes ----------------------
    sp = ds.stimulus_presentations
    sp = sp[sp.stimulus_block_name == 'change_detection_behavior']
    sp = sp.sort_values('start_time')
    shown = sp[~sp.omitted.astype(bool)]
    flash_start = shown.start_time.values.astype(np.float64)
    flash_image = shown.image_name.values.astype(str)
    change_start = sp[sp.is_change.astype(bool)].start_time.values.astype(np.float64)

    # ---- behavior: running speed and pupil diameter ----------------------
    run = ds.running_speed
    run_t = run.timestamps.values.astype(np.float64)
    run_v = run.speed.values.astype(np.float64)
    good = np.isfinite(run_t) & np.isfinite(run_v)
    run_t, run_v = run_t[good], run_v[good]
    if len(run_t) == 0:
        return {'oeid': oeid, 'error': 'no running data'}

    eye = ds.eye_tracking
    if len(eye) == 0:
        return {'oeid': oeid, 'error': 'no eye tracking'}
    pupil_t = eye.timestamps.values.astype(np.float64)
    # diameter of a disc with the fitted pupil area; blinks are already NaN in
    # pupil_area, drop them (and any other non-finite sample) and interpolate.
    pupil_d = 2.0 * np.sqrt(eye.pupil_area.values.astype(np.float64) / np.pi)
    good = (np.isfinite(pupil_t) & np.isfinite(pupil_d)
            & ~eye.likely_blink.values.astype(bool))
    pupil_t, pupil_d = pupil_t[good], pupil_d[good]
    if len(pupil_t) == 0:
        return {'oeid': oeid, 'error': 'no valid pupil data'}

    # ---- trials ----------------------------------------------------------
    trials = ds.trials
    keep = ((trials.go.astype(bool) | trials.catch.astype(bool))
            & ~trials.aborted.astype(bool)
            & ~trials.auto_rewarded.astype(bool)
            & trials.change_time.notna())
    trials = trials[keep]

    neural, image_name, change, running, pupil, outcome = [], [], [], [], [], []
    n_dropped_edge = 0
    n_outside_trial = 0
    for _, tr in trials.iterrows():
        change_time = float(tr.change_time)
        i0 = int(np.searchsorted(ts, change_time + OFF_START, side='left'))
        if i0 + T > len(ts):
            n_dropped_edge += 1
            continue
        tt = ts[i0:i0 + T]
        if tt[0] < tr.start_time or tt[-1] > tr.stop_time:
            # window would leave the experiment-defined trial; never happens
            # with the [-2, 2] s window, counted for the report
            n_outside_trial += 1

        # image identity: held from each flash onset to the next flash onset
        k = np.searchsorted(flash_start, tt, side='right') - 1
        if np.any(k < 0):
            n_dropped_edge += 1
            continue
        image_name.append(flash_image[k])

        # image change: the 750 ms presentation cycle starting at the change
        k = np.searchsorted(change_start, tt, side='right') - 1
        ch = np.zeros(T, dtype=np.int16)
        valid = k >= 0
        ch[valid] = (tt[valid] - change_start[k[valid]] < IMAGE_CYCLE)
        change.append(ch)

        running.append(np.interp(tt, run_t, run_v))
        pupil.append(np.interp(tt, pupil_t, pupil_d))

        if tr.hit:
            o = 0
        elif tr.miss:
            o = 1
        elif tr.false_alarm:
            o = 2
        elif tr.correct_reject:
            o = 3
        else:
            n_dropped_edge += 1
            image_name.pop(); change.pop(); running.pop(); pupil.pop()
            continue
        outcome.append(o)
        neural.append(neural_all[:, i0:i0 + T])

    if len(neural) < 2:
        return {'oeid': oeid, 'error': f'only {len(neural)} usable trials'}

    return {
        'oeid': oeid,
        'neural': np.stack(neural),                      # (ntrials, ncells, T)
        'image_name': np.stack(image_name),              # (ntrials, T) str
        'change': np.stack(change).astype(np.int16),     # (ntrials, T)
        'running': np.stack(running).astype(np.float32),  # (ntrials, T)
        'pupil': np.stack(pupil).astype(np.float32),     # (ntrials, T)
        'outcome': np.asarray(outcome, dtype=np.int16),  # (ntrials,)
        'cell_specimen_ids': cell_specimen_ids,
        'frame_rate': frame_rate,
        'dt': dt,
        'T': T,
        'n_dropped_edge': n_dropped_edge,
        'n_outside_trial': n_outside_trial,
        'n_trials_available': int(len(trials)),
    }


def cache_path(oeid):
    return os.path.join(CACHE_DIR, f'{oeid}.npz')


def extract_and_cache(oeid):
    """Worker: extract one session and write it to the cache directory."""
    out = cache_path(oeid)
    if os.path.exists(out):
        with np.load(out, allow_pickle=True) as f:
            return {'oeid': oeid, 'cached': True,
                    'error': str(f['error']) if 'error' in f else None,
                    'ntrials': int(f['outcome'].shape[0]) if 'outcome' in f else 0}
    try:
        res = extract_session(oeid)
    except Exception:
        res = {'oeid': oeid, 'error': traceback.format_exc(limit=3)}
    tmp = out + '.tmp.npz'
    np.savez(tmp, **res)
    os.replace(tmp, out)
    return {'oeid': oeid, 'cached': False, 'error': res.get('error'),
            'ntrials': int(res['outcome'].shape[0]) if 'outcome' in res else 0}


def percentile_bins(values, nbins):
    """Equal-count percentile bin edges (interior edges only)."""
    qs = np.linspace(0, 100, nbins + 1)[1:-1]
    return np.percentile(values, qs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='/app/converted_data.pkl')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--limit', type=int, default=None,
                        help='only convert this many sessions (debugging)')
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    table = session_table()
    if args.limit:
        table = table.iloc[:args.limit]
    oeids = list(table.ophys_experiment_id.values)
    print(f'{len(oeids)} sessions to convert '
          f'({table.mouse_id.nunique()} mice)', flush=True)

    # ---- 1. per-session extraction (parallel, cached on disk) ------------
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(extract_and_cache, oeids)):
            tag = 'cached' if res['cached'] else 'done'
            msg = res['error'] if res['error'] else f"{res['ntrials']} trials"
            print(f'[{i + 1}/{len(oeids)}] {res["oeid"]} {tag}: {msg}',
                  flush=True)

    # ---- 2. load the cache and assemble ---------------------------------
    sessions = []
    skipped = []
    for oeid in oeids:
        with np.load(cache_path(oeid), allow_pickle=True) as f:
            if 'error' in f:
                skipped.append((oeid, str(f['error'])))
                continue
            sessions.append({k: f[k] for k in f.files})
    print(f'{len(sessions)} sessions kept, {len(skipped)} skipped')
    for oeid, err in skipped:
        print(f'  skipped {oeid}: {err}')

    kept_ids = [int(s['oeid']) for s in sessions]
    table = table.set_index('ophys_experiment_id').loc[kept_ids]

    # time bins must be identical across sessions
    Ts = np.array([int(s['T']) for s in sessions])
    dts = np.array([float(s['dt']) for s in sessions])
    assert np.all(Ts == Ts[0]), f'inconsistent number of time bins: {set(Ts)}'
    T = int(Ts[0])
    print(f'T = {T} bins of {np.mean(dts) * 1000:.3f} ms '
          f'(range {dts.min() * 1000:.3f}-{dts.max() * 1000:.3f} ms)')

    # ---- 3. global percentile bins for running speed and pupil ----------
    all_running = np.concatenate([s['running'].ravel() for s in sessions])
    all_pupil = np.concatenate([s['pupil'].ravel() for s in sessions])
    run_edges = percentile_bins(all_running, NBINS_BEHAVIOR)
    pupil_edges = percentile_bins(all_pupil, NBINS_BEHAVIOR)
    print('running speed bin edges (cm/s):', np.round(run_edges, 3))
    print('pupil diameter bin edges (px):', np.round(pupil_edges, 3))
    del all_running, all_pupil

    # ---- 4. global image list -------------------------------------------
    images = sorted({str(im) for s in sessions
                     for im in np.unique(s['image_name'])})
    print(f'{len(images)} images: {images}')
    image_to_idx = {im: i for i, im in enumerate(images)}

    # ---- 5. assemble the output structure -------------------------------
    subjects = sorted(table.mouse_id.astype(str).unique().tolist())
    subject_idx = np.array([subjects.index(str(m)) for m in table.mouse_id],
                           dtype=np.int64)
    brain_regions = sorted(table.targeted_structure.unique().tolist())

    neural, inputs, outputs = [], [], []
    brain_region_idx = []
    session_info = []
    empty_input = np.zeros((0, T), dtype=np.float32)
    for s, (_, row) in zip(sessions, table.iterrows()):
        ntrials = s['neural'].shape[0]
        ncells = s['neural'].shape[1]
        neural.append([np.ascontiguousarray(s['neural'][i]) for i in range(ntrials)])
        inputs.append([empty_input] * ntrials)

        img = np.vectorize(image_to_idx.__getitem__)(s['image_name'])
        run_bin = np.searchsorted(run_edges, s['running'], side='right')
        pupil_bin = np.searchsorted(pupil_edges, s['pupil'], side='right')
        out = []
        for i in range(ntrials):
            out.append(np.stack([
                img[i].astype(np.int16),
                s['change'][i].astype(np.int16),
                run_bin[i].astype(np.int16),
                pupil_bin[i].astype(np.int16),
                np.full(T, s['outcome'][i], dtype=np.int16),
            ]))
        outputs.append(out)

        region = brain_regions.index(row.targeted_structure)
        brain_region_idx.append(np.full(ncells, region, dtype=np.int64))
        session_info.append({
            'ophys_experiment_id': int(s['oeid']),
            'ophys_session_id': int(row.ophys_session_id),
            'behavior_session_id': int(row.behavior_session_id),
            'ophys_container_id': int(row.ophys_container_id),
            'mouse_id': str(row.mouse_id),
            'cre_line': str(row.cre_line),
            'indicator': str(row.indicator),
            'session_type': str(row.session_type),
            'experience_level': str(row.experience_level),
            'image_set': str(row.image_set),
            'targeted_structure': str(row.targeted_structure),
            'imaging_depth': int(row.imaging_depth),
            'equipment_name': str(row.equipment_name),
            'n_neurons': int(ncells),
            'n_trials': int(ntrials),
            'ophys_frame_rate': float(s['frame_rate']),
        })

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
        'subject_idx': subject_idx,
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': [],
        'output_names': ['image_identity', 'image_change', 'running_speed',
                         'pupil_diameter', 'trial_outcome'],
        'output_values': [
            images,
            ['no_change', 'change'],
            [f'running_speed_bin{i}' for i in range(NBINS_BEHAVIOR)],
            [f'pupil_diameter_bin{i}' for i in range(NBINS_BEHAVIOR)],
            OUTCOMES,
        ],
        'metadata': {
            'task_description':
                'Visual Behavior 2-photon change detection task (Allen Brain '
                'Observatory). Head-fixed mice view a continuous series of '
                'flashed natural images (250 ms image, 500 ms gray) and are '
                'rewarded for licking within 150-750 ms of a change in image '
                'identity. Trials are the go (real change) and catch (sham '
                'change) trials of the experiment; aborted and auto-rewarded '
                'trials are excluded. Decoded from the neural activity are the '
                'identity of the image being shown, whether an image change '
                'just occurred, the binned running speed, the binned pupil '
                'diameter, and the behavioral outcome of the trial '
                '(hit / miss / false alarm / correct reject).',
            'time_bin_size': float(np.mean(dts) * 1000.0),
            'temporal_alignment_event':
                'image change time (onset of the changed image flash on go '
                'trials, onset of the sham-change flash on catch trials)',
            'off_start': OFF_START,
            'off_end': OFF_END,
            'neural_data_type':
                'detected calcium events (L0-regularized deconvolution of the '
                'dF/F traces, half-gaussian filtered: AllenSDK '
                'events["filtered_events"]), sampled at the ophys frame times',
            'neural_units': 'calcium event magnitude (dF/F units)',
            'dataset': 'Allen Brain Observatory Visual Behavior 2P, '
                       'visual-behavior-ophys-1.1.0, project code '
                       f'{PROJECT_CODE} (single-plane imaging), '
                       'active behavior sessions only',
            'session_description':
                'one session = one 2-photon imaging plane recorded during one '
                'behavior session; sessions differ in Cre line (Slc17a7 '
                'excitatory, Sst, Vip), cortical area, depth and image set',
            'running_speed_bin_edges': run_edges.tolist(),
            'pupil_diameter_bin_edges': pupil_edges.tolist(),
            'behavior_bin_note':
                'running speed (cm/s) and pupil diameter (2*sqrt(area/pi), in '
                'eye-camera pixels) were resampled onto the ophys frame times '
                'and discretized with percentile bin edges computed over all '
                'trials of all sessions',
            'image_identity_note':
                'the identity of the flashed image, held over the whole 750 ms '
                'image presentation cycle (image + following gray screen) and '
                'over omitted flashes',
            'image_change_note':
                '1 for the 750 ms image presentation cycle that starts with a '
                'real image change, 0 otherwise (sham changes on catch trials '
                'are 0)',
            'session_info': session_info,
        },
    }

    ntrials_total = sum(len(x) for x in neural)
    nneurons_total = sum(x[0].shape[0] for x in neural)
    print(f'{len(neural)} sessions, {ntrials_total} trials, '
          f'{nneurons_total} neurons, {len(subjects)} mice')

    with open(args.out, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'wrote {args.out} '
          f'({os.path.getsize(args.out) / 1e9:.2f} GB)')


if __name__ == '__main__':
    sys.exit(main())
