"""
Convert the Allen Brain Observatory "Visual Behavior 2P" dataset into the trial-structured
dictionary consumed by /app/train_decoder.py.

Overview of the decisions taken here (see README-style notes in each section):

Data selection
    * project_code == 'VisualBehavior'  -- the single-plane 2P change-detection dataset.
      Single-plane sessions contain exactly one imaging plane ("experiment") per session,
      so a session has one well defined population of simultaneously recorded neurons, and
      every session is sampled at the same 31 Hz ophys frame rate (required because the
      target format asks for one common time-bin size across all trials/sessions).
      The only other project code present in this data release
      (VisualBehaviorMultiscope) is excluded: it is 11 Hz, has up to 8 planes per session,
      and in this release contains a single mouse.
    * behaviour sessions only (passive == False).  Passive sessions have the lick spout
      retracted, so "trial outcome" (hit/miss/false alarm/correct reject) is undefined.
    * experience_level == 'Familiar'.  This follows the reference paper ("we restricted our
      analysis to familiar stimuli", "For neural analysis we used neurons recorded during
      familiar image set presentations").  It also means every included session used the
      same 8 natural images (image set A), so the categorical "image identity" output has
      the same meaning in every session.
    * sessions without eye-tracking data are dropped, since pupil diameter is a required
      decoder output.

Neural data
    * detected calcium events, as in the reference paper ("For all analysis of neural data
      we used the detected calcium events ... thus removing the slow decay dynamics of the
      calcium indicator").  We use the SDK's `filtered_events` (events convolved with a
      half-Gaussian): the raw event train is nonzero on only ~0.1% of 31 Hz frames, which
      carries essentially no information at single-timepoint resolution, whereas the
      filtered trace preserves event times and magnitudes in a form a per-timepoint decoder
      can use.  All ROIs released in the NWB files already passed the pipeline's ROI
      filtering / QC (valid_roi is True for every cell), so no further neuron curation.

Trials
    * the experiment's own trial definition (SDK `trials` table).  "Go" and "catch" trials
      are kept; "aborted" (mouse licked before the change) and "auto-rewarded" (free
      reward) trials are excluded, as instructed.
    * each trial spans [trials.start_time, trials.stop_time).

Temporal alignment
    * the ophys frame times (`ophys_timestamps`) are the time base: one time bin per
      2-photon frame (~32.3 ms).  Stimulus, running and pupil streams are aligned onto
      those frame times.
"""

import argparse
import os
import pickle
import re
import warnings
from multiprocessing import Pool

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

DATA_DIR = '/app/data/visual-behavior-ophys-1.1.0'
NWB_DIR = os.path.join(DATA_DIR, 'behavior_ophys_experiments')
META_DIR = os.path.join(DATA_DIR, 'project_metadata')
CACHE_DIR = '/app/cache_sessions'
OUT_PATH = '/app/converted_data.pkl'

NQUANTILES = 5          # running speed / pupil diameter are cut into 5 equal-count bins
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']


# ----------------------------------------------------------------------------------
# session selection
# ----------------------------------------------------------------------------------
def select_experiments():
    """Experiment table rows for the sessions we convert (see module docstring)."""
    available = sorted(int(re.search(r'(\d+)\.nwb', f).group(1))
                       for f in os.listdir(NWB_DIR) if f.endswith('.nwb'))
    tbl = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    sel = tbl[tbl.ophys_experiment_id.isin(available)
              & (tbl.project_code == 'VisualBehavior')
              & (~tbl.passive)
              & (tbl.experience_level == 'Familiar')]
    return sel.sort_values('ophys_experiment_id').reset_index(drop=True)


# ----------------------------------------------------------------------------------
# per-session extraction
# ----------------------------------------------------------------------------------
def pupil_diameter(eye_tracking):
    """Pupil diameter in pixels, from the SDK's fitted pupil area.

    `pupil_area` is already NaN on frames flagged `likely_blink`; the remaining NaNs are
    linearly interpolated over (and held constant at the edges) so that every ophys frame
    gets a value.  Returns None if the session has no usable eye tracking at all.
    """
    if eye_tracking is None or len(eye_tracking) == 0:
        return None, None
    area = eye_tracking['pupil_area'].to_numpy(dtype=float)
    area[eye_tracking['likely_blink'].to_numpy(dtype=bool)] = np.nan
    good = np.isfinite(area)
    if good.sum() < 0.5 * len(area):
        # less than half the session is usable pupil data
        return None, None
    t = eye_tracking['timestamps'].to_numpy(dtype=float)
    diam = 2.0 * np.sqrt(area[good] / np.pi)
    return t[good], diam


def extract_session(row):
    """Extract one imaging session; returns a dict of per-trial arrays (or None to skip)."""
    from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
        BehaviorOphysExperiment)

    oeid = int(row['ophys_experiment_id'])
    path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{oeid}.nwb')
    expt = BehaviorOphysExperiment.from_nwb_path(path)

    ts = np.asarray(expt.ophys_timestamps, dtype=float)       # time base for everything
    nframes = len(ts)

    # ---- neural: detected calcium events, (n_neurons, n_frames) -------------------
    events = expt.events
    cell_ids = list(events.index)
    traces = np.stack([np.asarray(x, dtype=np.float32)
                       for x in events['filtered_events'].values])
    assert traces.shape == (len(cell_ids), nframes), (traces.shape, nframes)

    # ---- behaviour streams resampled onto the ophys frame times -------------------
    run = expt.running_speed
    running = np.interp(ts, run['timestamps'].to_numpy(dtype=float),
                        run['speed'].to_numpy(dtype=float)).astype(np.float32)

    try:
        eye = expt.eye_tracking
    except Exception:
        eye = None
    et, ed = pupil_diameter(eye)
    if et is None:
        return {'oeid': oeid, 'skip': 'no eye tracking'}
    pupil = np.interp(ts, et, ed).astype(np.float32)

    # ---- stimulus: image on the screen at each ophys frame ------------------------
    stim = expt.stimulus_presentations
    stim = stim[stim['active'].astype(bool)]                  # change-detection block only
    shown = stim[~stim['omitted'].astype(bool)]
    image_names = sorted(set(shown['image_name'].dropna()))
    # 0 is reserved for "gray screen" (inter-stimulus interval and omitted flashes)
    img_lookup = {name: i + 1 for i, name in enumerate(image_names)}

    image_id = np.zeros(nframes, dtype=np.int8)
    starts = np.searchsorted(ts, shown['start_time'].to_numpy(dtype=float), side='left')
    stops = np.searchsorted(ts, shown['end_time'].to_numpy(dtype=float), side='left')
    codes = shown['image_name'].map(img_lookup).to_numpy(dtype=np.int8)
    for i0, i1, c in zip(starts, stops, codes):
        image_id[i0:i1] = c

    # image changes: 1 while the flash whose identity differs from the preceding one is on
    # the screen, i.e. over exactly the same frames that carry that image in `image_id`.
    # Sham changes on catch trials are *not* changes in image identity and stay 0.
    change = np.zeros(nframes, dtype=np.int8)
    is_chg = shown['is_change'].astype(bool).to_numpy()
    for i0, i1 in zip(starts[is_chg], stops[is_chg]):
        change[i0:i1] = 1

    # ---- trials -------------------------------------------------------------------
    trials = expt.trials
    keep = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
    trials = trials[keep]

    out = {'oeid': oeid, 'cell_ids': cell_ids, 'image_names': image_names,
           'neural': [], 'running': [], 'pupil': [], 'image_id': [], 'change': [],
           'outcome': [], 'trial_ids': [], 'n_frames_session': nframes}

    for tid, tr in trials.iterrows():
        i0 = np.searchsorted(ts, tr['start_time'], side='left')
        i1 = np.searchsorted(ts, tr['stop_time'], side='left')
        if i1 - i0 < 2:
            continue
        outcome = [tr['hit'], tr['miss'], tr['false_alarm'], tr['correct_reject']]
        if sum(bool(x) for x in outcome) != 1:
            continue                                   # not a scoreable go/catch trial
        out['neural'].append(np.ascontiguousarray(traces[:, i0:i1]))
        out['running'].append(running[i0:i1])
        out['pupil'].append(pupil[i0:i1])
        out['image_id'].append(image_id[i0:i1])
        out['change'].append(change[i0:i1])
        out['outcome'].append(int(np.argmax(outcome)))
        out['trial_ids'].append(int(tid))

    out['dt'] = float(np.median(np.diff(ts)))
    return out


def extract_cached(args):
    """extract_session + an on-disk cache, so the (slow) NWB reads happen only once."""
    idx, row = args
    cache = os.path.join(CACHE_DIR, f"{int(row['ophys_experiment_id'])}.pkl")
    if os.path.exists(cache):
        try:
            with open(cache, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    res = extract_session(row)
    tmp = cache + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(res, f, protocol=4)
    os.replace(tmp, cache)
    return res


# ----------------------------------------------------------------------------------
# assembly
# ----------------------------------------------------------------------------------
def quantile_bins(values, nbins):
    """Edges cutting `values` into `nbins` equal-count bins (interior edges only)."""
    qs = np.linspace(0, 100, nbins + 1)[1:-1]
    return np.percentile(values, qs)


def bin_names(label, edges, unit):
    lo = np.concatenate([[-np.inf], edges])
    hi = np.concatenate([edges, [np.inf]])
    names = []
    for i, (a, b) in enumerate(zip(lo, hi)):
        a_s = '-inf' if not np.isfinite(a) else f'{a:.2f}'
        b_s = 'inf' if not np.isfinite(b) else f'{b:.2f}'
        names.append(f'{label}_q{i + 1} [{a_s},{b_s}) {unit}')
    return names


def main(limit=None, workers=16, out_path=OUT_PATH):
    os.makedirs(CACHE_DIR, exist_ok=True)
    expts = select_experiments()
    if limit:
        expts = expts.iloc[:limit]
    print(f'{len(expts)} candidate experiments', flush=True)

    rows = [(i, r) for i, r in expts.iterrows()]
    if workers > 1:
        with Pool(min(workers, len(rows))) as pool:
            results = pool.map(extract_cached, rows, chunksize=1)
    else:
        results = [extract_cached(r) for r in rows]

    # ---- keep only usable sessions ------------------------------------------------
    sessions, meta_rows, skipped = [], [], []
    for res, (_, row) in zip(results, rows):
        if res.get('skip') or len(res.get('neural', [])) < 2:
            skipped.append((int(row['ophys_experiment_id']),
                            res.get('skip', 'fewer than 2 usable trials')))
            continue
        sessions.append(res)
        meta_rows.append(row)
    for oeid, why in skipped:
        print(f'  skipping experiment {oeid}: {why}', flush=True)

    # ---- global quantile bins for running speed and pupil diameter ----------------
    # Bins are computed on the pooled distribution over every timepoint that ends up in
    # the dataset, so each bin holds ~20% of the data and a bin index means the same
    # thing in every session.
    all_run = np.concatenate([np.concatenate(s['running']) for s in sessions])
    all_pupil = np.concatenate([np.concatenate(s['pupil']) for s in sessions])
    run_edges = quantile_bins(all_run, NQUANTILES)
    pupil_edges = quantile_bins(all_pupil, NQUANTILES)
    print('running speed bin edges (cm/s):', np.round(run_edges, 3))
    print('pupil diameter bin edges (pix):', np.round(pupil_edges, 3))

    # ---- image identity categories ------------------------------------------------
    image_names = sorted({n for s in sessions for n in s['image_names']})
    print(f'{len(image_names)} images: {image_names}')
    # per-session image codes are 1..n_images_in_that_session; remap them onto the global
    # ordering (0 stays "gray screen") so a category means the same image everywhere.
    for s in sessions:
        remap = np.zeros(len(s['image_names']) + 1, dtype=np.int8)
        for i, name in enumerate(s['image_names']):
            remap[i + 1] = image_names.index(name) + 1
        if not np.array_equal(remap, np.arange(len(remap), dtype=np.int8)):
            s['image_id'] = [remap[a] for a in s['image_id']]

    # ---- build the output dictionary ----------------------------------------------
    neural, inputs, outputs = [], [], []
    subject_list, subject_idx = [], []
    region_list, brain_region_idx = [], []
    session_info = []

    for s, row in zip(sessions, meta_rows):
        n_sess, i_sess, o_sess = [], [], []
        for k in range(len(s['neural'])):
            x = s['neural'][k]
            T = x.shape[1]
            out = np.empty((5, T), dtype=np.int8)
            out[0] = s['image_id'][k]
            out[1] = s['change'][k]
            out[2] = np.digitize(s['running'][k], run_edges)
            out[3] = np.digitize(s['pupil'][k], pupil_edges)
            out[4] = s['outcome'][k]
            n_sess.append(x)
            i_sess.append(np.zeros((0, T), dtype=np.float32))   # no decoder inputs
            o_sess.append(out)
        neural.append(n_sess)
        inputs.append(i_sess)
        outputs.append(o_sess)

        mouse = str(row['mouse_id'])
        if mouse not in subject_list:
            subject_list.append(mouse)
        subject_idx.append(subject_list.index(mouse))

        region = str(row['targeted_structure'])
        if region not in region_list:
            region_list.append(region)
        brain_region_idx.append(np.full(len(s['cell_ids']), region_list.index(region),
                                        dtype=np.int64))

        session_info.append({
            'ophys_experiment_id': int(row['ophys_experiment_id']),
            'ophys_session_id': int(row['ophys_session_id']),
            'behavior_session_id': int(row['behavior_session_id']),
            'ophys_container_id': int(row['ophys_container_id']),
            'mouse_id': mouse,
            'cre_line': str(row['cre_line']),
            'full_genotype': str(row['full_genotype']),
            'session_type': str(row['session_type']),
            'experience_level': str(row['experience_level']),
            'image_set': str(row['image_set']),
            'targeted_structure': region,
            'imaging_depth': int(row['imaging_depth']),
            'equipment_name': str(row['equipment_name']),
            'date_of_acquisition': str(row['date_of_acquisition']),
            'n_neurons': len(s['cell_ids']),
            'n_trials': len(s['neural']),
            'cell_specimen_ids': [int(c) for c in s['cell_ids']],
            'trial_ids': s['trial_ids'],
            'ophys_frame_interval_s': s['dt'],
        })

    dt_ms = float(np.mean([s['dt'] for s in sessions]) * 1000.0)

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subject_list,
        'subject_idx': np.array(subject_idx, dtype=np.int64),
        'brain_regions': region_list,
        'brain_region_idx': brain_region_idx,
        'input_names': [],
        'output_names': ['image_identity', 'image_change', 'running_speed_bin',
                         'pupil_diameter_bin', 'trial_outcome'],
        'output_values': [
            ['gray_screen'] + list(image_names),
            ['no_change', 'change'],
            bin_names('running_speed', run_edges, 'cm/s'),
            bin_names('pupil_diameter', pupil_edges, 'pix'),
            list(OUTCOMES),
        ],
        'metadata': {
            'task_description': (
                'Allen Brain Observatory Visual Behavior 2-photon change-detection task. '
                'Head-fixed mice view a continuous series of flashed natural images '
                '(250 ms image, 500 ms gray inter-stimulus interval; 5% of flashes '
                'omitted) and are rewarded for licking when the image identity changes. '
                'Trials are the go (real change) and catch (sham change) trials of the '
                'session; aborted and auto-rewarded trials are excluded. Decoded from '
                'the neural activity are: the identity of the image on the screen '
                '(8 natural images of image set A, or gray screen), whether an image '
                'change just occurred, the mouse running speed and pupil diameter '
                '(each in 5 equal-count bins), and the behavioral outcome of the trial '
                '(hit / miss / false alarm / correct reject).'),
            'time_bin_size': dt_ms,
            'temporal_alignment_event': (
                'trial start (trials.start_time in the AllenSDK trials table); each time '
                'bin is one 2-photon imaging frame (ophys_timestamps), and the stimulus, '
                'running-speed and pupil streams are resampled onto those frame times'),
            'off_start': 0.0,
            'off_end': None,
            'trial_window': (
                'trials.start_time (inclusive) to trials.stop_time (exclusive); trials '
                'have variable length because the change time is drawn from a truncated '
                'exponential distribution'),
            'neural_data_type': (
                'detected calcium events (AllenSDK events table, "filtered_events" '
                'column: detected events convolved with a half-Gaussian), dF/F-derived, '
                'per 2-photon frame'),
            'dataset': 'Allen Brain Observatory Visual Behavior 2P (visual-behavior-ophys-1.1.0)',
            'project_code': 'VisualBehavior (single-plane 2-photon)',
            'experience_level': 'Familiar (image set A, familiar to the mouse)',
            'behavior_type': 'active_behavior',
            'indicator': 'GCaMP6f',
            'ophys_frame_rate_hz': 1000.0 / dt_ms,
            'running_speed_bin_edges_cm_per_s': run_edges.tolist(),
            'pupil_diameter_bin_edges_pix': pupil_edges.tolist(),
            'image_change_definition': (
                'binary, 1 on the ophys frames during which the changed image is on the '
                'screen (the ~250 ms flash whose identity differs from the preceding '
                'flash), 0 otherwise; sham changes on catch trials are 0 because the '
                'image identity does not change'),
            'excluded_experiments': skipped,
            'n_sessions': len(sessions),
            'n_trials': int(sum(len(x) for x in neural)),
            'n_neurons': int(sum(len(b) for b in brain_region_idx)),
            'session_info': session_info,
        },
    }

    print(f"sessions={len(neural)} trials={data['metadata']['n_trials']} "
          f"neurons={data['metadata']['n_neurons']} mice={len(subject_list)} "
          f"time_bin_size={dt_ms:.3f} ms", flush=True)

    with open(out_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f'wrote {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--out', type=str, default=OUT_PATH)
    args = ap.parse_args()
    main(limit=args.limit, workers=args.workers, out_path=args.out)
