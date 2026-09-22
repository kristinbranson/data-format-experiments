"""
Convert the MAP brain-wide Neuropixels dataset (Chen et al., "Brain-wide neural activity
underlying memory-guided movement"; NWB files in /app/data) into the dictionary format
required by /app/train_decoder.py.

Decisions follow the reference papers / code as closely as the decoder task allows:

Session curation (data paper, STAR Methods):
  * behavioural performance > 65 % (fraction correct of control trials that were
    responded to: no photostimulation, no early lick, no auto/free water),
  * at least 50 correct lick-left and 50 correct lick-right control trials,
  * CCF annotation / QC classification present (one session lacks both).

Unit curation (data paper + Chen, Liu et al. spike-sorting white paper):
  * units labelled 'good' by the region-specific QC classifiers (units/classification),
  * with a CCF annotation (units/anno_name), i.e. histologically localized.
    This reproduces the published counts exactly (69,453 units, e.g. thalamus 12,808,
    striatum 7,664, midbrain 7,495, medulla 2,928).

Trial curation:
  * only trials during which the units were actually recorded are kept: units/obs_intervals
    lists the trial intervals in which each unit was observed, and in 8 sessions the
    recording covers only part of the behavioural session,
  * sessions left with fewer than 50 usable trials are dropped,
  * auto-water and free-water trials are dropped (water is not contingent on the
    animal's choice; the reference code drops them as well),
  * photostimulation, early-lick, error ('miss') and no-response ('ignore') trials are
    KEPT because the decoder task requires photostimulation as an input and
    early lick / outcome as outputs (the reference analyses drop them),
  * trials whose [-2.5, 1.5] s window contains less than half of the expected video
    frames are dropped (tongue output would be undefined).

Alignment / binning (decoder task):
  * trials aligned to the go cue, window -2.5 s .. +1.5 s, 50 ms bins (80 bins),
  * neural data are spike counts / bin width = firing rate (Hz).
    Spikes are only stored inside each trial's recorded interval
    (units/obs_intervals == trial start/stop); bins outside that interval contain no
    spikes and are therefore 0 (this mostly affects the last ~0.8 s of error trials).
"""

import argparse
import glob
import json
import os
import pickle
import sys
from multiprocessing import Pool

import h5py
import numpy as np

DATA_DIR = '/app/data'
OUT_FILE = '/app/converted_data.pkl'
TMP_DIR = '/app/tmp_sessions'
ONTOLOGY_FILE = '/app/allen_structure_graph.json'

OFF_START = -2.5          # s, relative to go cue
OFF_END = 1.5             # s, relative to go cue
BIN_SIZE = 0.05           # s
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))

LIKELIHOOD_THRESH = 0.9   # DeepLabCut confidence for 'tongue visible'
VIDEO_COVERAGE_MIN = 0.5  # minimum fraction of expected video frames in the window
ML_MIDLINE = 5700.0       # CCF x of the midline (as in the reference preprocessing code)

MIN_PERFORMANCE = 0.65
MIN_CORRECT_PER_DIRECTION = 50
MIN_TRIALS_PER_SESSION = 50   # sessions with fewer usable trials are dropped

OUTPUT_NAMES = ['choice', 'outcome', 'early_lick', 'tongue_y']
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['low (<40th pctile)', 'middle (40-60th pctile)', 'high (>60th pctile)', 'not visible'],
]
INPUT_NAMES = ['time from tone onset (s)', 'photostimulation on']


# ---------------------------------------------------------------- ontology ---
ONTOLOGY_URL = 'http://api.brain-map.org/api/v2/structure_graph_download/1.json'


def load_ontology():
    """name -> list of ancestor names (root first) from the Allen CCF structure graph.

    Recording locations in this dataset are annotated with Allen CCF (v3) structure
    names; the ontology is used to group them into the coarse regions used in the
    papers (ALM, striatum, thalamus, midbrain, medulla, ...).
    """
    if not os.path.exists(ONTOLOGY_FILE):
        import urllib.request
        urllib.request.urlretrieve(ONTOLOGY_URL, ONTOLOGY_FILE)
    with open(ONTOLOGY_FILE) as f:
        root = json.load(f)['msg'][0]
    name2path = {}

    def walk(node, path):
        p = path + [node['name']]
        name2path[node['name']] = p
        for c in node['children']:
            walk(c, p)

    walk(root, [])
    return name2path


ONTOLOGY = load_ontology()


def coarse_region(anno, probe_target):
    """Map a CCF annotation to one of the coarse regions used in the reference code."""
    path = ONTOLOGY.get(anno)
    if path is None:
        return None
    inpath = set(path)
    for key, name in [('Medulla', 'Medulla'), ('Pons', 'Pons'), ('Midbrain', 'Midbrain'),
                      ('Cerebellum', 'Cerebellum'), ('Thalamus', 'Thalamus'),
                      ('Hypothalamus', 'Hypothalamus'), ('Striatum', 'Striatum'),
                      ('Pallidum', 'Pallidum'), ('Hippocampal formation', 'Hippocampus'),
                      ('Olfactory areas', 'Olfactory'), ('Cortical subplate', 'CorticalSubplate')]:
        if key in inpath:
            return name
    if 'Isocortex' in inpath:
        is_motor = ('Primary motor area' in inpath) or ('Secondary motor area' in inpath)
        if is_motor and probe_target is not None and 'ALM' in probe_target:
            # ALM was defined functionally and overlaps the anterior parts of primary and
            # secondary motor cortex; probes of this dataset targeted at ALM sample it.
            return 'ALM'
        if 'Orbital area' in inpath:
            return 'Orbital'
        return 'OtherCortex'
    if 'Cerebral cortex' in inpath:
        return 'OtherCortex'
    return 'Other'


# ------------------------------------------------------------------ helpers ---
def to_str(arr):
    """Decode an hdf5 string column; NaN (missing) becomes ''."""
    return np.array([x.decode() if isinstance(x, bytes) else ('' if isinstance(x, float) else str(x))
                     for x in arr])


def session_name(path):
    base = os.path.basename(path)
    sub = base.split('_')[0].replace('sub-', '')
    ses = base.split('_')[1].replace('ses-', '')
    return sub, ses


# --------------------------------------------------------------- conversion ---
def process_session(path):
    subject, ses = session_name(path)
    info = {'file': os.path.basename(path), 'subject': subject, 'session': ses}
    with h5py.File(path, 'r') as f:
        trials = f['intervals/trials']
        start = trials['start_time'][:]
        stop = trials['stop_time'][:]
        outcome = to_str(trials['outcome'][:])
        early = to_str(trials['early_lick'][:])
        instruction = to_str(trials['trial_instruction'][:])
        auto_water = trials['auto_water'][:].astype(bool)
        free_water = trials['free_water'][:].astype(bool)
        photostim_trial = to_str(trials['photostim_onset'][:]) != 'N/A'
        ntrials_all = len(start)

        events = f['acquisition/BehavioralEvents']
        go = events['go_start_times/timestamps'][:]
        sample_on = events['sample_start_times/timestamps'][:]
        stim_on = events['photostim_start_times/timestamps'][:]
        stim_off = events['photostim_stop_times/timestamps'][:]
        if len(go) != ntrials_all:
            info['skip'] = 'go cue count does not match trial count'
            return info

        # ---- trials during which the units were actually recorded ----
        # units/obs_intervals lists, for every unit, the trial intervals in which it was
        # observed. In a few sessions the recording covers only part of the behavioural
        # session; those trials contain no spikes at all and must be dropped.
        units0 = f['units']
        cls0 = to_str(units0['classification'][:])
        anno0 = to_str(units0['anno_name'][:])
        good0 = np.where((cls0 == 'good') & (anno0 != ''))[0]
        if len(good0) == 0:
            info['skip'] = 'no QC-passing units with CCF annotation'
            return info
        oii = units0['obs_intervals_index'][:]
        obs = units0['obs_intervals'][:]
        recorded = np.ones(ntrials_all, dtype=bool)
        for uid in good0:
            o0 = 0 if uid == 0 else oii[uid - 1]
            ints = obs[o0:oii[uid]]
            idx = np.searchsorted(start, ints[:, 0] + 1e-6) - 1
            idx = idx[(idx >= 0) & (idx < ntrials_all)]
            m = np.zeros(ntrials_all, dtype=bool)
            m[idx] = True
            recorded &= m
        if recorded.sum() < 2:
            info['skip'] = 'fewer than 2 trials with ephys coverage'
            return info

        # ---- session selection (data paper criteria, on recorded trials) ----
        control = recorded & (~photostim_trial) & (early == 'no early') & (~auto_water) & (~free_water)
        responded = control & (outcome != 'ignore')
        performance = float(np.mean(outcome[responded] == 'hit')) if responded.sum() else 0.0
        n_correct_left = int(np.sum(control & (outcome == 'hit') & (instruction == 'left')))
        n_correct_right = int(np.sum(control & (outcome == 'hit') & (instruction == 'right')))
        info.update(performance=performance, n_correct_left=n_correct_left,
                    n_correct_right=n_correct_right, n_trials_total=int(ntrials_all),
                    n_trials_recorded=int(recorded.sum()))
        if performance <= MIN_PERFORMANCE:
            info['skip'] = 'performance <= 65%%: %.3f' % performance
            return info
        if min(n_correct_left, n_correct_right) < MIN_CORRECT_PER_DIRECTION:
            info['skip'] = 'fewer than 50 correct trials in one direction'
            return info

        # ---- unit curation ----
        units = f['units']
        classification = to_str(units['classification'][:])
        anno = to_str(units['anno_name'][:])
        good = (classification == 'good') & (anno != '')
        if good.sum() == 0:
            info['skip'] = 'no QC-passing units with CCF annotation'
            return info

        # unit position (CCF) via its (first) electrode, and the targeted brain region
        electrodes = f['general/extracellular_ephys/electrodes']
        ex = electrodes['x'][:]
        target = np.array([json.loads(loc)['brain_regions'] for loc in to_str(electrodes['location'][:])])
        eidx = units['electrodes_index'][:]
        efirst = np.concatenate([[0], eidx[:-1]]).astype(int)
        uelec = units['electrodes'][:][efirst]
        ux = ex[uelec]
        utarget = target[uelec]

        unit_ids = np.where(good)[0]
        regions = []
        keep_unit = []
        for i in unit_ids:
            reg = coarse_region(anno[i], utarget[i])
            x = ux[i]
            if reg is None or not np.isfinite(x):
                keep_unit.append(False)
                regions.append(None)
                continue
            side = 'left' if x >= ML_MIDLINE else 'right'
            keep_unit.append(True)
            regions.append('%s %s' % (side, reg))
        keep_unit = np.array(keep_unit)
        unit_ids = unit_ids[keep_unit]
        regions = [r for r, k in zip(regions, keep_unit) if k]
        if len(unit_ids) == 0:
            info['skip'] = 'no localized QC-passing units'
            return info

        # ---- trial curation ----
        keep = recorded & (~auto_water) & (~free_water)

        # video (side-view DeepLabCut tongue tracking)
        tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
        vts = tongue['timestamps'][:]
        vdata = tongue['data'][:]
        tongue_y = vdata[:, 1]
        visible = vdata[:, 2] > LIKELIHOOD_THRESH
        # per-session discretization thresholds, over frames where the tongue is visible
        if visible.sum() > 0:
            p40, p60 = np.percentile(tongue_y[visible], [40, 60])
        else:
            p40 = p60 = np.nan

        win_lo = go + OFF_START
        win_hi = go + OFF_END
        nframes = np.searchsorted(vts, win_hi) - np.searchsorted(vts, win_lo)
        expected_frames = (OFF_END - OFF_START) * 300.0
        keep &= nframes >= VIDEO_COVERAGE_MIN * expected_frames
        if not np.isfinite(p40):
            info['skip'] = 'no visible tongue frames in session'
            return info

        trial_idx = np.where(keep)[0]
        if len(trial_idx) < MIN_TRIALS_PER_SESSION:
            info['skip'] = 'only %d usable trials (mostly missing video)' % len(trial_idx)
            return info

        ntrials = len(trial_idx)
        nneurons = len(unit_ids)

        # ---- neural: spike counts in 50 ms bins ----
        neural = np.zeros((ntrials, nneurons, NBINS), dtype=np.float32)
        spike_times = units['spike_times']
        sidx = units['spike_times_index'][:]
        edges_lo = win_lo[trial_idx]
        for n, uid in enumerate(unit_ids):
            s0 = 0 if uid == 0 else sidx[uid - 1]
            sp = spike_times[s0:sidx[uid]]
            lo = np.searchsorted(sp, edges_lo)
            hi = np.searchsorted(sp, win_hi[trial_idx])
            for t in range(ntrials):
                if hi[t] <= lo[t]:
                    continue
                b = ((sp[lo[t]:hi[t]] - edges_lo[t]) / BIN_SIZE).astype(np.int64)
                np.clip(b, 0, NBINS - 1, out=b)
                neural[t, n] = np.bincount(b, minlength=NBINS)[:NBINS]
        neural /= BIN_SIZE  # spikes/s

        # ---- inputs ----
        bin_centers = OFF_START + (np.arange(NBINS) + 0.5) * BIN_SIZE   # rel. to go cue
        # tone (sample epoch) onset: last sample-epoch start before the go cue; with
        # early licks the sample epoch is replayed, so the last one is the relevant one
        tone_idx = np.searchsorted(sample_on, go) - 1
        tone_rel_go = sample_on[np.maximum(tone_idx, 0)] - go   # negative (before go cue)

        inputs = np.zeros((ntrials, 2, NBINS), dtype=np.float32)
        for k, t in enumerate(trial_idx):
            inputs[k, 0] = bin_centers - tone_rel_go[t]

        # photostimulation: 1 in bins overlapping a photostim interval
        stim_trial = np.searchsorted(start, stim_on) - 1
        for on, off, tr in zip(stim_on, stim_off, stim_trial):
            if tr < 0 or tr >= ntrials_all or not keep[tr]:
                continue
            k = int(np.searchsorted(trial_idx, tr))
            b0 = int(np.floor((on - win_lo[tr]) / BIN_SIZE))
            b1 = int(np.ceil((off - win_lo[tr]) / BIN_SIZE))
            b0 = max(b0, 0)
            b1 = min(b1, NBINS)
            if b1 > b0:
                inputs[k, 1, b0:b1] = 1.0

        # ---- outputs ----
        outputs = np.zeros((ntrials, 4, NBINS), dtype=np.int64)
        # choice: hit -> instructed side, miss (error) -> opposite side, ignore -> no lick
        choice_map = {'left': 0, 'right': 1}
        for k, t in enumerate(trial_idx):
            if outcome[t] == 'hit':
                ch = choice_map[instruction[t]]
            elif outcome[t] == 'miss':
                ch = choice_map['right' if instruction[t] == 'left' else 'left']
            else:
                ch = 2
            outputs[k, 0, :] = ch
            outputs[k, 1, :] = {'ignore': 0, 'miss': 1, 'hit': 2}[outcome[t]]
            outputs[k, 2, :] = 1 if early[t] == 'early' else 0

        # tongue y position class per time bin
        for k, t in enumerate(trial_idx):
            lo = np.searchsorted(vts, win_lo[t])
            hi = np.searchsorted(vts, win_hi[t])
            if hi <= lo:
                outputs[k, 3, :] = 3
                continue
            tt = vts[lo:hi]
            yy = tongue_y[lo:hi]
            vv = visible[lo:hi]
            b = ((tt - win_lo[t]) / BIN_SIZE).astype(np.int64)
            np.clip(b, 0, NBINS - 1, out=b)
            cls = np.full(NBINS, 3, dtype=np.int64)
            for bi in range(NBINS):
                m = (b == bi) & vv
                if not np.any(m):
                    continue
                y = float(np.median(yy[m]))
                cls[bi] = 0 if y < p40 else (1 if y <= p60 else 2)
            outputs[k, 3, :] = cls

    info.update(n_trials_used=int(ntrials), n_neurons=int(nneurons),
                n_photostim_trials=int(np.sum(photostim_trial[trial_idx])),
                n_early_trials=int(np.sum(early[trial_idx] == 'early')))
    result = {
        'info': info,
        'neural': [neural[k] for k in range(ntrials)],
        'input': [inputs[k] for k in range(ntrials)],
        'output': [outputs[k] for k in range(ntrials)],
        'regions': regions,
        'subject': subject,
    }
    out_path = os.path.join(TMP_DIR, '%s_%s.pkl' % (subject, ses))
    with open(out_path, 'wb') as fh:
        pickle.dump(result, fh, protocol=4)
    info['out_path'] = out_path
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='only process this many sessions')
    ap.add_argument('--nproc', type=int, default=16)
    ap.add_argument('--out', type=str, default=OUT_FILE)
    args = ap.parse_args()

    os.makedirs(TMP_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
    if args.limit:
        files = files[:args.limit]
    print('processing %d sessions' % len(files), flush=True)

    with Pool(args.nproc) as pool:
        infos = []
        for info in pool.imap_unordered(process_session, files):
            infos.append(info)
            msg = info.get('skip', 'kept %d trials x %d neurons' %
                           (info.get('n_trials_used', 0), info.get('n_neurons', 0)))
            print('%s: %s' % (info['file'], msg), flush=True)

    kept = sorted([i for i in infos if 'out_path' in i], key=lambda i: i['file'])
    print('kept %d / %d sessions' % (len(kept), len(infos)), flush=True)

    data = {'neural': [], 'input': [], 'output': [], 'brain_region_idx': []}
    subjects, subject_idx, brain_regions, session_info = [], [], [], []
    for info in kept:
        with open(info['out_path'], 'rb') as fh:
            sess = pickle.load(fh)
        data['neural'].append(sess['neural'])
        data['input'].append(sess['input'])
        data['output'].append(sess['output'])
        for r in sess['regions']:
            if r not in brain_regions:
                brain_regions.append(r)
        data['brain_region_idx'].append(
            np.array([brain_regions.index(r) for r in sess['regions']], dtype=np.int64))
        s = sess['subject']
        if s not in subjects:
            subjects.append(s)
        subject_idx.append(subjects.index(s))
        session_info.append(sess['info'])

    brain_regions_sorted = sorted(brain_regions)
    remap = np.array([brain_regions_sorted.index(r) for r in brain_regions], dtype=np.int64)
    data['brain_region_idx'] = [remap[idx] for idx in data['brain_region_idx']]
    data['brain_regions'] = brain_regions_sorted
    data['subjects'] = subjects
    data['subject_idx'] = np.array(subject_idx, dtype=np.int64)
    data['input_names'] = INPUT_NAMES
    data['output_names'] = OUTPUT_NAMES
    data['output_values'] = OUTPUT_VALUES
    data['metadata'] = {
        'task_description': (
            'Head-fixed mice performed an auditory delayed-response task: a pure tone '
            '(3 or 12 kHz, played 3x150 ms) during the sample epoch instructed licking '
            'left or right, a 1.2 s delay epoch followed, and an auditory go cue released '
            'the animal to report its choice by licking one of two lick ports. ALM was '
            'photoinhibited bilaterally or unilaterally during the last 0.5 s of the delay '
            'on ~25% of trials in VGAT-ChR2-EYFP mice. Decoder inputs are the time from '
            'tone (sample epoch) onset and whether photostimulation is on; outputs are the '
            'lick direction choice, trial outcome, whether the animal licked early, and a '
            'per-session discretization of the DeepLabCut tongue y-position.'),
        'time_bin_size': BIN_SIZE * 1000.0,
        'temporal_alignment_event': 'auditory go cue onset (end of the delay epoch)',
        'off_start': OFF_START,
        'off_end': OFF_END,
        'neural_units': 'firing rate (spikes/s), spike counts per 50 ms bin / bin size',
        'session_info': session_info,
        'dataset': ('Chen, Kang et al., "Brain-wide neural activity underlying memory-guided '
                    'movement" (Cell 2024); DANDI 000363 NWB files'),
        'unit_curation': ('units labelled good by the region-specific QC classifiers '
                          '(units/classification == good) and with a CCF annotation'),
        'session_curation': ('behavioural performance > 65% on control trials that were '
                             'responded to and >= 50 correct lick-left and lick-right control '
                             'trials, as in the data paper; sessions without CCF annotations '
                             'excluded'),
        'trial_curation': ('auto-water and free-water trials excluded; trials with less than '
                           '50% video coverage of the analysis window excluded; sessions with '
                           'fewer than 50 usable trials excluded; '
                           'photostimulation, early-lick, error and no-response trials kept '
                           'because they are decoder inputs/outputs'),
        'notes': ('Spikes are only stored within each trial\'s recorded interval '
                  '(units/obs_intervals == trial start/stop). Error trials typically end '
                  '~0.8 s after the go cue, so the last bins of those trials contain no '
                  'spikes and are zero.'),
        'tongue_discretization': ('per-session 40th/60th percentiles of the side-view '
                                  'DeepLabCut tongue y-position over frames with likelihood '
                                  '> %.2f; bins with no confident detection are class '
                                  '"not visible"' % LIKELIHOOD_THRESH),
    }

    print('writing %s' % args.out, flush=True)
    with open(args.out, 'wb') as fh:
        pickle.dump(data, fh, protocol=4)
    ntr = sum(len(s) for s in data['neural'])
    nneu = sum(len(b) for b in data['brain_region_idx'])
    print('sessions %d, subjects %d, trials %d, neurons %d, regions %d' %
          (len(data['neural']), len(subjects), ntr, nneu, len(brain_regions_sorted)), flush=True)


if __name__ == '__main__':
    main()
