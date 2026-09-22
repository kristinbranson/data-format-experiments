"""Convert Allen Brain Observatory Visual Behavior 2P data into the decoder format.

Data selection:
  * Only experiments from the change-detection ("Visual Behavior") task performed
    actively by the mouse (behavior_type == 'active_behavior'); passive-viewing
    sessions contain no trials/behavioral report and are therefore excluded.
  * Only familiar image sets, following Garrett/Bennett et al. ("we restricted our
    analysis to familiar stimuli").
  * Only the single-plane VisualBehavior project code. All those experiments are
    imaged at ~31 Hz, so that every session shares the same time-bin size (as
    required by the target format). The few Multiscope (11 Hz) sessions available
    locally (4 sessions from a single mouse) would break that requirement.
  * Trials: the "go" and "catch" trials defined by the Allen trials table, i.e.
    aborted and auto-rewarded trials are excluded.

Neural data: detected calcium events (the paper: "For all analysis of neural data we
used the detected calcium events"). We use the AllenSDK 'filtered_events' trace,
which is exactly those detected events convolved with a causal half-normal kernel:
the raw event trace is nonzero in only ~0.3% of the 32 ms frames, which carries
almost no information in a single time bin, while the filtered trace keeps the event
times but spreads each event over the following few frames. Each cell is then divided
by the standard deviation of its own trace over the session, so that the per-session
projection learned by the decoder is not dominated by a few very active cells.

All data streams are put on one common clock: the ophys frame timestamps of the
imaging plane ("temporally align based on ophys timestamp"). The stimulus table,
running speed (60 Hz) and pupil (30 Hz eye-tracking camera) are resampled onto those
frame times, so one time bin = one ophys frame (~32.3 ms).
"""

import argparse
import os
import pickle
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

DATA_DIR = '/app/data/visual-behavior-ophys-1.1.0'
EXP_DIR = os.path.join(DATA_DIR, 'behavior_ophys_experiments')
META = os.path.join(DATA_DIR, 'project_metadata', 'ophys_experiment_table.csv')

OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
NBINS = 5  # number of percentile bins for running speed and pupil diameter


def select_experiments():
    meta = pd.read_csv(META)
    have = set()
    for f in os.listdir(EXP_DIR):
        if f.endswith('.nwb'):
            have.add(int(f.split('_')[-1].split('.')[0]))
    sel = meta[meta.ophys_experiment_id.isin(have)
               & (meta.behavior_type == 'active_behavior')
               & (meta.experience_level == 'Familiar')
               & (meta.project_code == 'VisualBehavior')].copy()
    sel = sel.sort_values('ophys_experiment_id').reset_index(drop=True)
    return sel


def process_experiment(args):
    eid, signal = args
    from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
        BehaviorOphysExperiment)
    exp = BehaviorOphysExperiment.from_nwb_path(
        os.path.join(EXP_DIR, f'behavior_ophys_experiment_{eid}.nwb'))

    ts = np.asarray(exp.ophys_timestamps, dtype=np.float64)
    nframes = len(ts)

    # ---- neural: detected calcium events on the ophys frame clock -------------
    ev_table = exp.events
    cell_ids = list(ev_table.index.values)
    neural = np.stack([np.asarray(x, dtype=np.float32)
                       for x in ev_table[signal].values])  # (ncells, nframes)
    assert neural.shape[1] == nframes
    # Scale each cell by the standard deviation of its trace over the whole session.
    # Event magnitudes are in units of dF/F and vary by orders of magnitude between
    # cells (and between cre lines / imaging depths); without this, the per-session
    # projection of the decoder is dominated by a few high-variance cells.
    sd = neural.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    neural = neural / sd

    # ---- stimulus: image identity and image change ----------------------------
    sp = exp.stimulus_presentations
    flashes = sp[(sp.active == True) & (sp.omitted == False)]  # noqa: E712
    image_names = sorted(flashes.image_name.unique())
    image_id = np.zeros(nframes, dtype=np.int64)   # 0 == gray screen / omitted
    change = np.zeros(nframes, dtype=np.int64)
    starts = np.searchsorted(ts, flashes.start_time.values, side='left')
    stops = np.searchsorted(ts, flashes.end_time.values, side='left')
    idx = np.array([image_names.index(n) + 1 for n in flashes.image_name.values])
    is_change = flashes.is_change.values.astype(bool)
    for a, b, i, c in zip(starts, stops, idx, is_change):
        image_id[a:b] = i
        if c:
            change[a:b] = 1

    # ---- running speed --------------------------------------------------------
    rs = exp.running_speed
    running = np.interp(ts, rs.timestamps.values, rs.speed.values)

    # ---- pupil diameter -------------------------------------------------------
    et = exp.eye_tracking
    if len(et) == 0:
        return dict(eid=eid, skip='no eye tracking')
    area = et.pupil_area.values.astype(np.float64).copy()
    area[et.likely_blink.values.astype(bool)] = np.nan  # blinks are not measurements
    diam = 2.0 * np.sqrt(area / np.pi)
    good = np.isfinite(diam)
    if good.sum() < 100:
        return dict(eid=eid, skip='no valid pupil samples')
    et_ts = et.timestamps.values.astype(np.float64)
    # interpolate the (blink) gaps, on the eye-tracking clock, then onto ophys frames
    pupil = np.interp(ts, et_ts[good], diam[good])
    # frames whose nearest valid pupil sample is far away are flagged as missing
    nearest = np.abs(ts - et_ts[good][np.clip(
        np.searchsorted(et_ts[good], ts), 0, good.sum() - 1)])
    nearest_prev = np.abs(ts - et_ts[good][np.clip(
        np.searchsorted(et_ts[good], ts) - 1, 0, good.sum() - 1)])
    gap = np.minimum(nearest, nearest_prev)
    pupil_valid = gap < 0.5  # within half a second of a real measurement

    # ---- trials ---------------------------------------------------------------
    tr = exp.trials
    gc = tr[(tr.go | tr.catch)]
    trials = []
    for tid, row in gc.iterrows():
        a = int(np.searchsorted(ts, row.start_time, side='left'))
        b = int(np.searchsorted(ts, row.stop_time, side='left'))
        if b - a < 2:
            continue
        if np.mean(pupil_valid[a:b]) < 0.5:
            continue  # pupil not tracked for most of this trial (blink / lost eye)
        oc = [o for o in OUTCOMES if bool(row[o])]
        if len(oc) != 1:
            continue
        trials.append(dict(
            neural=neural[:, a:b].copy(),
            image_id=image_id[a:b].copy(),
            change=change[a:b].copy(),
            running=running[a:b].copy(),
            pupil=pupil[a:b].copy(),
            outcome=OUTCOMES.index(oc[0]),
            trial_id=int(tid),
            change_time=float(row.change_time),
            start_time=float(row.start_time),
            stop_time=float(row.stop_time),
        ))

    md = exp.metadata
    return dict(
        eid=eid, skip=None, trials=trials, cell_ids=cell_ids,
        image_names=image_names, dt=float(np.median(np.diff(ts))),
        mouse_id=str(md['mouse_id']), structure=str(md['targeted_structure']),
        cre_line=str(md['cre_line']), session_type=str(md['session_type']),
        ophys_session_id=int(md['ophys_session_id']),
        imaging_depth=int(md['imaging_depth']),
        frame_rate=float(md['ophys_frame_rate']),
        ntrials_total=int(len(gc)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/app/converted_data.pkl')
    ap.add_argument('--signal', default='filtered_events',
                    choices=['events', 'filtered_events'])
    ap.add_argument('--nsessions', type=int, default=0,
                    help='limit number of sessions (0 = all), for pilot runs')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--binning', default='session', choices=['session', 'global'])
    args = ap.parse_args()

    sel = select_experiments()
    if args.nsessions:
        sel = sel.iloc[:args.nsessions]
    eids = [int(x) for x in sel.ophys_experiment_id.values]
    print(f'processing {len(eids)} experiments with signal={args.signal}')

    with Pool(args.workers) as pool:
        results = pool.map(process_experiment, [(e, args.signal) for e in eids])

    results = [r for r in results if r.get('skip') is None and len(r['trials']) >= 2]
    print(f'{len(results)} sessions kept')

    # ---- percentile bins for running speed and pupil diameter -----------------
    # Pupil size is measured in camera pixels and its absolute value differs by up
    # to ~3x between sessions/mice (eye size, camera position), and mice also differ
    # strongly in how much they run. The decoder has a separate projection per
    # session, so only within-session variation can be decoded. Bin edges are
    # therefore computed per session (equal-occupancy quintiles within a session),
    # unless --binning global is requested.
    qs = np.linspace(0, 100, NBINS + 1)[1:-1]
    if args.binning == 'global':
        run_all = np.concatenate([t['running'] for r in results for t in r['trials']])
        pup_all = np.concatenate([t['pupil'] for r in results for t in r['trials']])
        edges = {r['eid']: (np.percentile(run_all, qs), np.percentile(pup_all, qs))
                 for r in results}
    else:
        edges = {}
        for r in results:
            run_all = np.concatenate([t['running'] for t in r['trials']])
            pup_all = np.concatenate([t['pupil'] for t in r['trials']])
            edges[r['eid']] = (np.percentile(run_all, qs), np.percentile(pup_all, qs))
    ex = edges[results[0]['eid']]
    print('example bin edges: running (cm/s)', np.round(ex[0], 3),
          ' pupil (pix)', np.round(ex[1], 3))

    image_names = sorted({n for r in results for n in r['image_names']})
    print('images:', image_names)

    subjects = sorted({r['mouse_id'] for r in results})
    regions = sorted({r['structure'] for r in results})

    data = dict(neural=[], input=[], output=[], subjects=subjects,
                subject_idx=[], brain_regions=regions, brain_region_idx=[],
                input_names=[],
                output_names=['image_identity', 'image_change',
                              'running_speed_bin', 'pupil_diameter_bin',
                              'trial_outcome'],
                output_values=[
                    ['gray_screen'] + image_names,
                    ['no_change', 'change'],
                    [f'run_q{i+1}' for i in range(NBINS)],
                    [f'pupil_q{i+1}' for i in range(NBINS)],
                    ['hit', 'miss', 'false_alarm', 'correct_reject']],
                metadata={})
    session_info = []
    for r in results:
        run_edges, pup_edges = edges[r['eid']]
        img_map = np.array([0] + [image_names.index(n) + 1
                                  for n in r['image_names']])
        neural_s, input_s, output_s = [], [], []
        for t in r['trials']:
            T = t['neural'].shape[1]
            out = np.zeros((5, T), dtype=np.int64)
            out[0] = img_map[t['image_id']]
            out[1] = t['change']
            out[2] = np.searchsorted(run_edges, t['running'], side='right')
            out[3] = np.searchsorted(pup_edges, t['pupil'], side='right')
            out[4] = t['outcome']
            neural_s.append(t['neural'].astype(np.float32))
            input_s.append(np.zeros((0, T), dtype=np.float32))
            output_s.append(out)
        data['neural'].append(neural_s)
        data['input'].append(input_s)
        data['output'].append(output_s)
        data['subject_idx'].append(subjects.index(r['mouse_id']))
        data['brain_region_idx'].append(
            np.full(len(r['cell_ids']), regions.index(r['structure']), dtype=np.int64))
        session_info.append(dict(
            ophys_experiment_id=r['eid'], ophys_session_id=r['ophys_session_id'],
            mouse_id=r['mouse_id'], cre_line=r['cre_line'],
            session_type=r['session_type'], targeted_structure=r['structure'],
            imaging_depth=r['imaging_depth'], frame_rate=r['frame_rate'],
            n_neurons=len(r['cell_ids']), n_trials=len(r['trials']),
            n_trials_go_catch=r['ntrials_total'],
            running_speed_bin_edges=[float(x) for x in run_edges],
            pupil_diameter_bin_edges=[float(x) for x in pup_edges],
            cell_specimen_ids=[int(c) for c in r['cell_ids']]))
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)

    dt = float(np.mean([r['dt'] for r in results]))
    data['metadata'] = dict(
        task_description=(
            'Allen Brain Observatory Visual Behavior 2P: head-fixed mice perform a '
            'go/no-go visual change detection task with familiar natural images '
            '(250 ms flashes separated by 500 ms of gray screen, 5% of flashes '
            'omitted) while 2-photon calcium imaging is performed in VISp. Trials '
            'are the go (image change) and catch (sham change) trials of the Allen '
            'trials table; aborted and auto-rewarded trials are excluded. The '
            'decoder predicts, from the population calcium events, the identity of '
            'the image on the screen, whether an image change just occurred, '
            'quintile-binned running speed, quintile-binned pupil diameter, and the '
            'outcome of the trial (hit / miss / false alarm / correct reject).'),
        time_bin_size=dt * 1000.0,
        temporal_alignment_event=(
            'trial start (start_time of the go/catch trial in the Allen trials '
            'table); all data streams are resampled onto the ophys frame '
            'timestamps of the imaging plane'),
        off_start=0.0,
        off_end=None,
        trial_definition=('trials run from trials.start_time to trials.stop_time; '
                          'the stimulus change (or sham change for catch trials) '
                          'occurs ~3-8 s after trial start and the trial ends '
                          '~4.2 s after the change, so trial length varies'),
        neural_signal=('detected calcium events (AllenSDK \'%s\'), sampled at the '
                       'native ophys frame rate and divided, per cell, by the '
                       'standard deviation of that cell over the session'
                       % args.signal),
        dataset='visual-behavior-ophys-1.1.0',
        project_code='VisualBehavior (single-plane, ~31 Hz)',
        experience_level='Familiar (image set A)',
        behavior_type='active_behavior',
        binning=('per-session equal-occupancy quintiles' if args.binning == 'session'
                 else 'global equal-occupancy quintiles'),
        pupil_measure='diameter = 2*sqrt(pupil_area/pi) in camera pixels, blinks '
                      'removed and interpolated',
        session_info=session_info,
    )

    ntr = sum(len(s) for s in data['neural'])
    nneur = sum(len(b) for b in data['brain_region_idx'])
    print(f"sessions={len(data['neural'])} trials={ntr} neurons={nneur} "
          f"mice={len(subjects)} dt={dt*1000:.2f} ms")
    with open(args.out, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
