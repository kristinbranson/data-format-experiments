"""
Convert the Mesoscale Activity Map (MAP) NWB dataset (DANDI 000363; Chen et al.,
"Brain-wide neural activity underlying memory-guided movement") into the
dictionary format required by /app/train_decoder.py.

Processing choices follow the reference papers / code
(MapVideoAnalysis: VideoAnalysisUtils/preprocessing_DJ_2022Aug.py) wherever they
are compatible with the decoding task defined in the task description.

See README-style notes in `main()` docstring and in the metadata of the saved file.
"""
import os
import sys
import glob
import json
import pickle
import argparse
from collections import OrderedDict
from multiprocessing import Pool

import numpy as np
import h5py

# ---------------------------------------------------------------- parameters
DATA_DIR = '/app/data'
OUT_FILE = '/app/converted_data.pkl'
TMP_DIR = '/tmp/converted_sessions'

OFF_START = -2.5           # s relative to go cue
OFF_END = 1.5              # s relative to go cue
BIN_SIZE = 0.05            # s (50 ms bins, as required by the task)
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))

# session inclusion criteria from the data paper (Chen et al.):
#   overall behavioural performance > 65 % and at least 50 correct lick-left and
#   50 correct lick-right trials (computed on control, non-early-lick trials).
MIN_PERFORMANCE = 0.65
MIN_CORRECT_PER_DIRECTION = 50

# DeepLabCut likelihood above which the tongue is considered visible.
# The likelihood distribution is strongly bimodal (mass at ~0 and at ~1), so the
# exact value in between has almost no influence.
TONGUE_LIKELIHOOD_THRESH = 0.9
TONGUE_LOW_PCTL = 40.0
TONGUE_HIGH_PCTL = 60.0

OUTPUT_NAMES = ['choice', 'outcome', 'early_lick', 'tongue_y_position']
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['<40th pctl', '40-60th pctl', '>60th pctl', 'not visible'],
]
INPUT_NAMES = ['time_from_tone_onset', 'photostim_on']

# ------------------------------------------------------------ region mapping
# Coarse brain areas, following the grouping used by the reference preprocessing
# code (preprocessing_DJ_2022Aug.py: ALM, Medulla, Midbrain, Striatum, Thalamus,
# Pons, Cerebellum, Hypothalamus, Hippocampus, Orbital, OtherCortex, Olfactory,
# CorticalSubplate, Pallidum).  Each unit is assigned from its CCF annotation
# (units/anno_name).  ALM is a functionally defined part of the frontal motor
# cortex, so motor-cortical units recorded with an ALM-targeted probe are
# labelled ALM (the probe target is stored in the electrode `location` field).
REGION_KEYS = [
    ('Cerebellum', ['lobul', 'lingula', 'declive', 'folium', 'pyramus', 'uvula', 'nodulus',
                    'culmen', 'ansiform', 'crus 1', 'crus 2', 'paramedian lobule',
                    'copula pyramidis', 'paraflocculus', 'flocculus', 'fastigial',
                    'interposed nucleus', 'dentate nucleus', 'cerebell', 'vermal',
                    'simple lobule', 'infracerebellar']),
    ('Medulla', ['gigantocellular', 'intermediate reticular', 'parvicellular reticular',
                 'medullary reticular', 'magnocellular reticular', 'paragigantocellular',
                 'solitary tract', 'parasolitary', 'parapyramidal',
                 'spinal nucleus of the trigeminal', 'facial motor nucleus', 'hypoglossal',
                 'inferior olivary', 'external cuneate', 'cuneate nucleus', 'gracile',
                 'nucleus ambiguus', 'lateral reticular', 'prepositus', 'raphe obscurus',
                 'raphe magnus', 'raphe pallidus', 'vestibular nucle', 'medial vestibular',
                 'spinal vestibular', 'lateral vestibular', 'superior vestibular', 'nucleus x',
                 'nucleus y', 'nucleus of roller', 'area postrema', 'intercalated nucleus',
                 'linear nucleus of the medulla', 'dorsal cochlear', 'ventral cochlear',
                 'cochlear nucle', 'accessory abducens', 'abducens', 'efferent vestibular',
                 'perihypoglossal', 'trigeminal reticular', 'vagus nerve', 'medulla']),
    ('Pons', ['pontine', 'pons', 'parabrachial', 'koelliker-fuse', 'locus ceruleus',
              'superior olivary', 'raphe pontis', 'tegmental reticular nucleus', 'barrington',
              'laterodorsal tegmental', 'sublaterodorsal',
              'principal sensory nucleus of the trigeminal', 'motor nucleus of trigeminal',
              'supratrigeminal', 'lateral lemniscus', 'dorsal tegmental nucleus',
              'nucleus incertus', 'subceruleus']),
    ('Midbrain', ['superior colliculus', 'inferior colliculus', 'midbrain', 'substantia nigra',
                  'periaqueductal gray', 'red nucleus', 'pretectal', 'nucleus of the optic tract',
                  'oculomotor', 'edinger', 'ventral tegmental area', 'interpeduncular',
                  'cuneiform', 'parabigeminal', 'sagulum', 'accessory optic tract',
                  'darkschewitsch', 'interstitial nucleus of cajal', 'dorsal nucleus raphe',
                  'raphe linearis', 'pedunculopontine', 'trochlear', 'anterior tegmental',
                  'nucleus of the brachium', 'magnocellular nucleus of the midbrain']),
    ('Hypothalamus', ['hypothalam', 'preoptic', 'mammillary', 'subthalamic', 'tuberal',
                      'supramammillary', 'arcuate', 'zona incerta', 'fields of forel',
                      'lateral terminal']),
    ('Thalamus', ['thalam', 'geniculate', 'habenula', 'paracentral nucleus',
                  'central lateral nucleus', 'central medial nucleus', 'parafascicular',
                  'reuniens', 'rhomboid', 'submedial', 'subparafascicular', 'peripeduncular',
                  'anteroventral nucleus', 'anteromedial nucleus', 'anterodorsal nucleus',
                  'interanterodorsal', 'intermediodorsal', 'ethmoid', 'posterior limiting',
                  'xiphoid', 'perireunensis', 'triangular nucleus of septum']),
    ('Hippocampus', ['field ca', 'dentate gyrus', 'subiculum', 'hippocamp', 'entorhinal',
                     'ammon', 'fasciola']),
    ('Striatum', ['caudoputamen', 'nucleus accumbens', 'fundus of striatum', 'striatum',
                  'olfactory tubercle', 'lateral septal', 'septofimbrial', 'medial amygdalar',
                  'intercalated amygdalar', 'central amygdalar', 'anterior amygdalar']),
    ('Pallidum', ['globus pallidus', 'substantia innominata', 'medial septal', 'diagonal band',
                  'bed nucle', 'pallidum', 'magnocellular nucleus']),
    ('CorticalSubplate', ['basolateral amygdalar', 'basomedial amygdalar', 'lateral amygdalar',
                          'posterior amygdalar', 'endopiriform', 'claustrum',
                          'cortical subplate']),
    ('Olfactory', ['anterior olfactory', 'piriform', 'olfactory areas', 'taenia tecta',
                   'olfactory bulb', 'lateral olfactory tract', 'cortical amygdalar',
                   'postpiriform', 'nucleus of the lateral olfactory']),
    ('Orbital', ['orbital area']),
    ('FiberTracts', ['fiber tracts', 'corpus callosum', 'capsule', 'commissure', 'peduncle',
                     'alveus', 'fimbria', 'tract', 'ventricle', 'lemniscus', 'white matter',
                     'fasciculus', 'optic chiasm', 'corticospinal']),
]
ALM_ANNOTATION_PREFIXES = ('secondary motor area', 'primary motor area', 'frontal pole')


def coarse_region(anno, probe_targets_alm):
    """Map a CCF annotation string to one of the coarse brain areas."""
    al = anno.strip().lower()
    if al == '':
        return 'Unknown'
    if probe_targets_alm and al.startswith(ALM_ANNOTATION_PREFIXES):
        return 'ALM'
    for name, keys in REGION_KEYS:
        for k in keys:
            if k in al:
                return name
    if 'area' in al or 'cortex' in al or 'field' in al:
        return 'OtherCortex'
    return 'Unknown'


# ------------------------------------------------------------------- helpers
def _dec(arr):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])


def _event_times(f, name):
    grp = f['acquisition/BehavioralEvents']
    if name not in grp:
        return np.array([])
    return np.asarray(grp[name]['timestamps'][:])


def session_info(path):
    """Read the small tables needed to decide whether a session is used."""
    with h5py.File(path, 'r') as f:
        t = f['intervals/trials']
        info = dict(
            path=path,
            subject=f['general/subject/subject_id'][()].decode(),
            session_id=f['identifier'][()].decode(),
            start=np.asarray(t['start_time'][:]),
            stop=np.asarray(t['stop_time'][:]),
            outcome=_dec(t['outcome'][:]),
            early=_dec(t['early_lick'][:]),
            instruction=_dec(t['trial_instruction'][:]),
            photostim_onset=_dec(t['photostim_onset'][:]),
            free_water=np.asarray(t['free_water'][:]).astype(bool),
            auto_water=np.asarray(t['auto_water'][:]).astype(bool),
            ngood=int((f['units']['classification'][:] == b'good').sum()),
        )
    return info


def session_is_used(info):
    """Data-paper session selection criteria (Chen et al. STAR Methods).

    Performance (fraction correct) is computed on control (no photostimulation)
    trials, excluding early-lick trials, and the session must additionally have
    at least 50 correct lick-left and 50 correct lick-right trials.
    """
    ctrl = (info['photostim_onset'] == 'N/A') & (info['early'] == 'no early')
    if ctrl.sum() == 0:
        return False, 0.0, 0, 0
    perf = float(np.mean(info['outcome'][ctrl] == 'hit'))
    ncl = int(np.sum(ctrl & (info['outcome'] == 'hit') & (info['instruction'] == 'left')))
    ncr = int(np.sum(ctrl & (info['outcome'] == 'hit') & (info['instruction'] == 'right')))
    used = (perf > MIN_PERFORMANCE and ncl >= MIN_CORRECT_PER_DIRECTION
            and ncr >= MIN_CORRECT_PER_DIRECTION and info['ngood'] > 0)
    return used, perf, ncl, ncr


def go_cue_times(f, start, stop):
    """Go cue time for every trial (NaN if the trial has no go cue)."""
    go = _event_times(f, 'go_start_times')
    idx = np.searchsorted(start, go, 'right') - 1
    out = np.full(len(start), np.nan)
    for i, g in zip(idx, go):
        if 0 <= i < len(out) and np.isnan(out[i]):
            out[i] = g
    return out


def tone_onset_times(f, start, go):
    """Onset of the instruction tone (sample epoch) for every trial.

    Licking during the sample/delay epoch triggers a replay of the epoch, so a
    trial can contain several sample-epoch onsets; we use the last one before
    the go cue, i.e. the tone the animal actually had to remember.
    """
    samp = _event_times(f, 'sample_start_times')
    idx = np.searchsorted(start, samp, 'right') - 1
    out = np.full(len(start), np.nan)
    for i, s in zip(idx, samp):
        if i < 0 or i >= len(out) or np.isnan(go[i]) or s > go[i]:
            continue
        if np.isnan(out[i]) or s > out[i]:
            out[i] = s
    return out


def observed_trials(f, start):
    """Boolean mask of the trials for which spikes were actually recorded."""
    u = f['units']
    oix = np.asarray(u['obs_intervals_index'][:])
    if len(oix) == 0:
        return np.zeros(len(start), dtype=bool)
    counts = np.diff(np.concatenate([[0], oix]))
    # all units of a session share the same observation intervals in this dataset
    assert len(np.unique(counts)) == 1, 'units differ in their observation intervals'
    oi = np.asarray(u['obs_intervals'][0:oix[0]])
    mask = np.zeros(len(start), dtype=bool)
    idx = np.searchsorted(start, oi[:, 0] + 1e-6, 'right') - 1
    idx = idx[(idx >= 0) & (idx < len(start))]
    mask[idx] = True
    return mask


def process_session(path):
    """Convert one NWB session.  Returns None if the session is not used."""
    info = session_info(path)
    used, perf, ncl, ncr = session_is_used(info)
    if not used:
        return None

    start, stop = info['start'], info['stop']
    with h5py.File(path, 'r') as f:
        go = go_cue_times(f, start, stop)
        tone = tone_onset_times(f, start, go)

        # ---------------- trial selection ----------------
        # All behaviour types required by the decoding task are kept (photostim,
        # early-lick, error and no-response trials are decoder inputs/outputs).
        # Trials in which the animal received water independently of its choice
        # (free-water / auto-water trials) are not normal task trials and are
        # dropped, as in the reference code's regular-trial mask.
        keep = (~np.isnan(go)) & (~np.isnan(tone)) & (~info['free_water']) & (~info['auto_water'])
        # In several sessions the electrophysiology covers only part of the
        # behavioural session.  The units' observation intervals (identical for
        # all units of a session) give exactly the trials during which spikes
        # were recorded; trials outside them contain no data at all and are
        # dropped instead of being converted into all-zero firing rates.
        keep &= observed_trials(f, start)
        trials = np.where(keep)[0]
        if len(trials) < 2:
            return None

        g = go[trials]
        # bin edges / centres, aligned to the go cue
        edges = g[:, None] + OFF_START + np.arange(N_BINS + 1)[None, :] * BIN_SIZE
        centers = 0.5 * (edges[:, :-1] + edges[:, 1:])

        # ---------------- neural ----------------
        u = f['units']
        good = np.where(u['classification'][:] == b'good')[0]
        sti = np.asarray(u['spike_times_index'][:])
        anno = _dec(u['anno_name'][:])
        # probe target region (used only to identify ALM recordings)
        loc = f['general/extracellular_ephys/electrodes']['location'][:]
        uel = np.asarray(u['electrodes'][:])
        targets = np.array([json.loads(loc[i].decode()).get('brain_regions', '') for i in uel])

        rates = np.zeros((len(good), len(trials), N_BINS), dtype=np.float32)
        flat_edges = edges.ravel()
        for k, iu in enumerate(good):
            a = 0 if iu == 0 else sti[iu - 1]
            spikes = np.asarray(u['spike_times'][a:sti[iu]])
            if spikes.size and np.any(np.diff(spikes) < 0):
                spikes = np.sort(spikes)
            pos = np.searchsorted(spikes, flat_edges).reshape(len(trials), N_BINS + 1)
            rates[k] = np.diff(pos, axis=1).astype(np.float32) / BIN_SIZE

        regions = [str(coarse_region(anno[i], 'ALM' in targets[i])) for i in good]

        # a handful of trials at the very end of a recording contain no spikes
        # at all (the recording stopped); they carry no neural information
        nonempty = rates.sum(axis=(0, 2)) > 0
        if not np.all(nonempty):
            trials = trials[nonempty]
            rates = rates[:, nonempty, :]
            edges = edges[nonempty]
            centers = centers[nonempty]
            g = g[nonempty]
            if len(trials) < 2:
                return None

        # ---------------- inputs ----------------
        # (1) time (s) since the instruction-tone onset, (2) photostimulation on
        dt_tone = (centers - tone[trials][:, None]).astype(np.float32)
        stim_on = np.zeros((len(trials), N_BINS), dtype=np.float32)
        ps = _event_times(f, 'photostim_start_times')
        pe = _event_times(f, 'photostim_stop_times')
        for s_, e_ in zip(ps, pe):
            ov = (edges[:, 1:] > s_) & (edges[:, :-1] < e_)
            stim_on[ov] = 1.0
        inputs = np.stack([dt_tone, stim_on], axis=1)  # (ntrials, 2, nbins)

        # ---------------- outputs ----------------
        outcome = info['outcome'][trials]
        instruction = info['instruction'][trials]
        early = info['early'][trials]

        # choice: the lick port the animal chose in the response epoch.  A 'hit'
        # means it licked the instructed port, a 'miss' the opposite one, and an
        # 'ignore' trial means the animal did not lick.
        choice = np.full(len(trials), 2, dtype=np.int64)          # 2 = no lick
        hit = outcome == 'hit'
        miss = outcome == 'miss'
        choice[hit & (instruction == 'left')] = 0
        choice[hit & (instruction == 'right')] = 1
        choice[miss & (instruction == 'left')] = 1
        choice[miss & (instruction == 'right')] = 0

        outcome_code = np.zeros(len(trials), dtype=np.int64)      # 0 = ignore
        outcome_code[miss] = 1
        outcome_code[hit] = 2

        early_code = (early == 'early').astype(np.int64)

        tongue_code = tongue_y_classes(f, edges)

        outputs = np.stack([np.repeat(choice[:, None], N_BINS, axis=1),
                            np.repeat(outcome_code[:, None], N_BINS, axis=1),
                            np.repeat(early_code[:, None], N_BINS, axis=1),
                            tongue_code], axis=1)                 # (ntrials, 4, nbins)

        session = dict(
            neural=[np.ascontiguousarray(rates[:, i, :]) for i in range(len(trials))],
            input=[np.ascontiguousarray(inputs[i]) for i in range(len(trials))],
            output=[np.ascontiguousarray(outputs[i]) for i in range(len(trials))],
            regions=regions,
            subject=info['subject'],
            session_id=info['session_id'],
            performance=perf,
            ncorrect_left=ncl, ncorrect_right=ncr,
            ntrials_total=len(start), ntrials_used=len(trials),
            # fraction of the analysis window that is inside the trial's
            # observation interval (spike times are only stored between the
            # trial start and stop times in this NWB release)
            frac_observed=float(np.mean(np.clip(
                (np.minimum(stop[trials], g + OFF_END) - np.maximum(start[trials], g + OFF_START))
                / (OFF_END - OFF_START), 0, 1))),
        )
    return session


def tongue_y_classes(f, edges):
    """Discretised tongue y-position for every trial and time bin.

    The side-view DeepLabCut tongue marker is used (the paper only analyses the
    side view).  A video frame counts as 'tongue visible' when the DeepLabCut
    likelihood exceeds TONGUE_LIKELIHOOD_THRESH; the likelihood distribution is
    strongly bimodal so the precise threshold is immaterial.  The y-position of
    a time bin is the mean over the visible frames that fall in it; bins without
    a visible frame are labelled 'not visible' (class 3).  Percentile
    boundaries are computed per session over all visible frames.
    """
    ntrials, nedges = edges.shape
    nbins = nedges - 1
    grp = f['acquisition/BehavioralTimeSeries']
    key = 'Camera0_side_TongueTracking'
    if key not in grp:
        return np.full((ntrials, nbins), 3, dtype=np.int64)
    data = np.asarray(grp[key]['data'][:])
    ts = np.asarray(grp[key]['timestamps'][:])
    y = data[:, 1]
    vis = data[:, 2] > TONGUE_LIKELIHOOD_THRESH
    vis &= np.isfinite(y)
    lo, hi = np.percentile(y[vis], [TONGUE_LOW_PCTL, TONGUE_HIGH_PCTL]) if vis.sum() else (0., 0.)

    yv = np.where(vis, y, 0.0)
    cum_y = np.concatenate([[0.0], np.cumsum(yv)])
    cum_n = np.concatenate([[0], np.cumsum(vis.astype(np.int64))])

    pos = np.searchsorted(ts, edges.ravel()).reshape(ntrials, nedges)
    n = cum_n[pos[:, 1:]] - cum_n[pos[:, :-1]]
    s = cum_y[pos[:, 1:]] - cum_y[pos[:, :-1]]
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_y = np.where(n > 0, s / np.maximum(n, 1), np.nan)

    code = np.full((ntrials, nbins), 3, dtype=np.int64)
    seen = n > 0
    code[seen & (mean_y < lo)] = 0
    code[seen & (mean_y >= lo) & (mean_y <= hi)] = 1
    code[seen & (mean_y > hi)] = 2
    return code


# ---------------------------------------------------------------------- main
def _worker(path):
    try:
        out = process_session(path)
    except Exception as exc:  # pragma: no cover - defensive
        print('FAILED %s: %r' % (path, exc), flush=True)
        return None
    if out is None:
        print('skipped %s' % os.path.basename(path), flush=True)
        return None
    fn = os.path.join(TMP_DIR, os.path.basename(path) + '.pkl')
    with open(fn, 'wb') as fh:
        pickle.dump(out, fh, protocol=4)
    print('done %s: %d neurons x %d trials' %
          (os.path.basename(path), out['neural'][0].shape[0], len(out['neural'])), flush=True)
    return fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=None, help='only convert the first N sessions')
    parser.add_argument('--nproc', type=int, default=16)
    parser.add_argument('--out', type=str, default=OUT_FILE)
    args = parser.parse_args()

    os.makedirs(TMP_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
    if args.limit:
        paths = paths[:args.limit]
    print('found %d nwb sessions' % len(paths))

    with Pool(args.nproc) as pool:
        files = pool.map(_worker, paths)
    files = [x for x in files if x is not None]
    print('converted %d sessions' % len(files))

    data = {k: [] for k in ['neural', 'input', 'output', 'brain_region_idx']}
    subjects, subject_idx, regions, sessions_info = [], [], [], []
    for fn in files:
        with open(fn, 'rb') as fh:
            s = pickle.load(fh)
        data['neural'].append(s['neural'])
        data['input'].append(s['input'])
        data['output'].append(s['output'])
        for r in s['regions']:
            if r not in regions:
                regions.append(r)
        data['brain_region_idx'].append(np.array([regions.index(r) for r in s['regions']],
                                                 dtype=np.int64))
        if s['subject'] not in subjects:
            subjects.append(s['subject'])
        subject_idx.append(subjects.index(s['subject']))
        sessions_info.append(dict(session_id=s['session_id'], subject=s['subject'],
                                  n_neurons=int(s['neural'][0].shape[0]),
                                  n_trials=int(len(s['neural'])),
                                  n_trials_in_session=int(s['ntrials_total']),
                                  behavioral_performance=float(s['performance']),
                                  n_correct_left=int(s['ncorrect_left']),
                                  n_correct_right=int(s['ncorrect_right']),
                                  frac_window_observed=float(s['frac_observed'])))

    data['subjects'] = subjects
    data['subject_idx'] = np.array(subject_idx, dtype=np.int64)
    data['brain_regions'] = regions
    data['input_names'] = INPUT_NAMES
    data['output_names'] = OUTPUT_NAMES
    data['output_values'] = OUTPUT_VALUES
    data['metadata'] = {
        'task_description':
            'Head-fixed mice performed an auditory delayed-response task (Chen et al., '
            '"Brain-wide neural activity underlying memory-guided movement", DANDI '
            '000363).  An instruction tone (3 kHz or 12 kHz, three 150 ms pips) during '
            'the sample epoch instructed the mouse to lick the left or the right port; '
            'after a 1.2 s delay an auditory go cue opened the 1.5 s response epoch.  On '
            '~25% of randomly interleaved trials ALM was photoinhibited during the last '
            '0.5 s of the delay.  The decoder receives the spike rates of all QC-passing '
            'units together with the time since the instruction-tone onset and the '
            'photostimulation state, and predicts (1) the licking direction chosen by the '
            'animal, (2) the trial outcome, (3) whether the trial contained an early lick '
            'and (4) the discretised vertical position of the tongue.',
        'time_bin_size': BIN_SIZE * 1000.0,
        'temporal_alignment_event': 'onset of the auditory go cue (end of the delay epoch)',
        'off_start': OFF_START,
        'off_end': OFF_END,
        'n_timepoints': N_BINS,
        'neural_units': 'spikes/s (spike count per 50 ms bin divided by the bin width)',
        'input_descriptions': [
            'time in seconds from the onset of the instruction tone (sample epoch); '
            'negative before the tone.  When the sample epoch was replayed because of an '
            'early lick, the last tone onset before the go cue is used.',
            'binary: 1 while ALM photostimulation was on (last 0.5 s of the delay epoch '
            'on photostimulation trials), 0 otherwise',
        ],
        'output_descriptions': [
            'licking direction chosen in the response epoch: left / right / no lick '
            '(derived from the instructed direction and the trial outcome)',
            'trial outcome: ignore (no response) / miss (licked the wrong port) / hit '
            '(licked the correct port)',
            'whether the animal licked during the sample or delay epoch (early lick)',
            'tongue y-position from the side-view DeepLabCut marker, discretised per '
            'session: 0 = below the 40th percentile, 1 = 40th-60th percentile, 2 = above '
            'the 60th percentile of the y-positions of all frames of the session in which '
            'the tongue was visible (DeepLabCut likelihood > %.2f), 3 = tongue not visible '
            'in that time bin' % TONGUE_LIKELIHOOD_THRESH,
        ],
        'unit_selection':
            'units labelled "good" by the region-specific quality-control classifiers of '
            'the accompanying spike-sorting white paper (units/classification in the NWB '
            'files), as used in both reference papers',
        'session_selection':
            'sessions with behavioural performance > %d%% on control (non-photostimulation) '
            'trials excluding early-lick trials and with at least %d correct lick-left and '
            '%d correct lick-right trials, as in the data paper; this yields the 106 '
            'sessions analysed in the method paper, of which 105 are converted here '
            '(one session contains no unit that passes the spike-sorting quality '
            'control)' %
            (int(MIN_PERFORMANCE * 100), MIN_CORRECT_PER_DIRECTION, MIN_CORRECT_PER_DIRECTION),
        'trial_selection':
            'all task trials of the selected sessions except trials in which water was '
            'delivered independently of the animal\'s choice (free-water and auto-water '
            'trials).  Photostimulation, early-lick, error and no-response trials are kept '
            'because they are decoder inputs/outputs in this task.',
        'known_caveat':
            'In this NWB release spike times are only stored inside each trial\'s '
            'observation interval (trial start to trial stop).  Parts of the -2.5 to 1.5 s '
            'window that fall outside that interval therefore contain no spikes and are '
            'binned as zero rate; this mainly affects the last ~0.8 s of error (miss) '
            'trials, which end when the animal licks the wrong port.  The '
            'frac_window_observed field of session_info quantifies this per session.',
        'session_info': sessions_info,
        'source': 'DANDI:000363 (Chen et al. 2023), NWB files in /app/data',
    }

    print('sessions: %d, subjects: %d, regions: %d' %
          (len(data['neural']), len(subjects), len(regions)))
    print('total trials: %d, total neurons: %d' %
          (sum(len(s) for s in data['neural']),
           sum(len(r) for r in data['brain_region_idx'])))
    with open(args.out, 'wb') as fh:
        pickle.dump(data, fh, protocol=4)
    print('saved to %s' % args.out)


if __name__ == '__main__':
    main()
