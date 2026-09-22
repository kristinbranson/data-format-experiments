"""Convert the MAP mesoscale activity map NWB dataset (DANDI 000363) into the
decoder-ready pickle format.

Source data
-----------
Chen, Liu, Kang, Chen, Liu, Liu, Svoboda, Li, Druckmann (2024) "Brain-wide neural
activity underlying memory-guided movement" (`datapaper.pdf`), released on DANDI as
dandiset 000363.  The same data are analysed in Wang, Kurgyis et al. (2025)
"Brain-wide analysis reveals movement encoding structured across and within brain
areas" (`methodpaper.pdf`), whose code is in `/app/code`.

Processing choices follow those two papers and `/app/code` wherever the decoder
task does not force something else; every deviation is commented at the point it
is made.

Decoder task (fixed by the assignment)
--------------------------------------
* align on the go cue, window -2.5 s .. +1.5 s, 50 ms bins (80 bins/trial)
* inputs : time from tone (sample-epoch) onset [s], photostimulation on/off
* outputs: lick direction, trial outcome, early lick, binned tongue y-position

Usage
-----
    python convert_data.py [--data-dir DIR] [--out FILE] [--workers N]
"""

import argparse
import glob
import json
import os
import pickle
import sys
import traceback
from multiprocessing import Pool

import h5py
import numpy as np


# --------------------------------------------------------------------------- #
# Allen CCF annotation -> coarse brain-region group
#
# The group list follows the region loop in the reference preprocessing code
# (MapVideoAnalysis/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py::process_one_sess):
# ALM, Medulla, Midbrain, Striatum, Thalamus, Pons, Cerebellum, Hypothalamus,
# Hippocampus, Orbital, OtherCortex, Olfactory, CorticalSubplate, Pallidum.
# Cross-check against the data paper, which reports good-unit counts per group:
# thalamus 12808 (this mapping: 12808), medulla 2928 (2928), striatum 7664 (7666),
# midbrain 7495 (7475).
# --------------------------------------------------------------------------- #

def _any(name, keys):
    return any(k in name for k in keys)


CEREBELLUM = ('lobule', 'crus 1', 'crus 2', 'copula pyramidis', 'nodulus',
              'uvula', 'declive', 'pyramus', 'lingula', 'folium', 'culmen',
              'ansiform', 'flocculus', 'fastigial nucleus', 'interposed nucleus',
              'dentate nucleus', 'cerebell', 'vermal', 'hemispheric')

HYPOTHALAMUS = ('hypothalam', 'zona incerta', 'fields of forel', 'preoptic',
                'subthalamic nucleus', 'mammillary', 'arcuate hypothalamic',
                'median eminence', 'tuberal nucleus')

THALAMUS = ('thalam', 'habenula', 'geniculate', 'paracentral nucleus',
            'parafascicular nucleus', 'anterodorsal nucleus',
            'anteromedial nucleus', 'anteroventral nucleus',
            'suprageniculate nucleus', 'subparafascicular',
            'peripeduncular nucleus', 'rhomboid nucleus',
            'perireunensis nucleus', 'nucleus of reuniens', 'xiphoid')

MIDBRAIN = ('midbrain', 'colliculus', 'pretectal', 'accessory optic tract',
            'substantia nigra', 'red nucleus', 'ventral tegmental area',
            'pedunculopontine nucleus', 'periaqueductal gray', 'cuneiform',
            'nucleus of the optic tract', 'interpeduncular nucleus',
            'oculomotor nucleus', 'edinger-westphal', 'darkschewitsch',
            'trochlear nucleus', 'anterior tegmental nucleus',
            'midbrain trigeminal nucleus', 'retrorubral')

PONS = ('pons', 'pontine reticular nucleus', 'tegmental reticular nucleus',
        'parabrachial', 'lateral lemniscus', 'locus ceruleus',
        'koelliker-fuse', 'nucleus sagulum', 'superior olivary complex',
        'pontine gray', 'raphe pontis', 'supratrigeminal',
        'laterodorsal tegmental', 'sublaterodorsal',
        'motor nucleus of trigeminal', 'principal sensory nucleus of the trigeminal')

MEDULLA = ('medulla', 'medullary reticular', 'gigantocellular',
           'magnocellular reticular', 'parvicellular reticular',
           'intermediate reticular nucleus', 'lateral reticular nucleus',
           'vestibular nucleus', 'spinal nucleus of the trigeminal',
           'solitary tract', 'cuneate', 'hypoglossal', 'facial motor nucleus',
           'vagus nerve', 'olivary complex', 'raphe magnus', 'raphe obscurus',
           'raphe pallidus', 'parapyramidal', 'nucleus x', 'nucleus y',
           'nucleus of roller', 'cochlear nucleus', 'nucleus ambiguus',
           'area postrema', 'infracerebellar', 'perihypoglossal',
           'linear nucleus of the medulla', 'paragigantocellular', 'parasolitary')

STRIATUM = ('caudoputamen', 'nucleus accumbens', 'fundus of striatum',
            'olfactory tubercle', 'striatum', 'lateral septal',
            'septofimbrial', 'triangular nucleus of septum',
            'central amygdalar nucleus', 'intercalated amygdalar nucleus',
            'medial amygdalar nucleus', 'anterior amygdalar area')

PALLIDUM = ('globus pallidus', 'substantia innominata', 'pallidum',
            'bed nuclei of the stria terminalis', 'medial septal nucleus',
            'diagonal band nucleus', 'magnocellular nucleus')

CORTICAL_SUBPLATE = ('claustrum', 'endopiriform', 'lateral amygdalar nucleus',
                     'basolateral amygdalar nucleus', 'basomedial amygdalar nucleus',
                     'posterior amygdalar nucleus', 'cortical subplate')

HIPPOCAMPUS = ('field ca1', 'field ca2', 'field ca3', 'dentate gyrus',
               'subiculum', 'entorhinal area', 'hippocamp', 'fasciola cinerea',
               'induseum griseum')

OLFACTORY = ('olfactory', 'piriform', 'taenia tecta', 'dorsal peduncular area',
             'cortical amygdalar area', 'nucleus of the lateral olfactory tract')

MOTOR_CORTEX = ('primary motor area', 'secondary motor area')


def coarse_region(anno_name, probe_target=''):
    """Return the coarse region group for one unit.

    Args:
        anno_name: Allen CCF annotation of the unit (``units/anno_name``).
        probe_target: targeted brain region of the probe the unit was recorded
            on, e.g. ``'left ALM'``.  Only used to separate ALM from the rest of
            motor cortex: ALM is not an Allen CCF structure, so motor-cortex
            units are called ALM when the probe was aimed at ALM, as in the
            reference pipeline where ALM was a targeting-based grouping.
    """
    name = anno_name.strip().lower()
    if not name:
        return None
    # Hypothalamus first: 'hypothalamic'/'subthalamic' contain 'thalam'.
    if _any(name, HYPOTHALAMUS):
        return 'Hypothalamus'
    if _any(name, CEREBELLUM) and 'infracerebellar' not in name:
        return 'Cerebellum'
    if _any(name, MEDULLA):
        return 'Medulla'
    if _any(name, PONS):
        return 'Pons'
    if _any(name, MIDBRAIN):
        return 'Midbrain'
    if _any(name, THALAMUS):
        return 'Thalamus'
    if _any(name, CORTICAL_SUBPLATE):
        return 'CorticalSubplate'
    if _any(name, HIPPOCAMPUS):
        return 'Hippocampus'
    if _any(name, PALLIDUM):
        return 'Pallidum'
    if _any(name, STRIATUM):
        return 'Striatum'
    if _any(name, OLFACTORY):
        return 'Olfactory'
    if 'orbital area' in name:
        return 'Orbital'
    if _any(name, MOTOR_CORTEX):
        return 'ALM' if 'alm' in probe_target.lower() else 'OtherCortex'
    return 'OtherCortex'

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Trial window and binning, fixed by the decoder task.  The reference pipeline
# (MapVideoAnalysis/Sherlock/preprocess_all_ephys.py) uses a -3 .. +3 s window
# around the go cue with 40 ms bins slid by 3.4 ms; here the assignment asks for
# -2.5 .. +1.5 s and 50 ms bins, so bins are laid down back-to-back (stride =
# width) and tile the window exactly.
T_START = -2.5
T_STOP = 1.5
BIN_WIDTH = 0.05
N_BINS = int(round((T_STOP - T_START) / BIN_WIDTH))       # 80
BIN_EDGES = T_START + BIN_WIDTH * np.arange(N_BINS + 1)   # relative to go cue
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2

# DeepLabCut confidence above which the tongue counts as visible.  The tongue
# likelihood in this dataset is essentially binary (median ~5e-5 when the tongue
# is in the mouth, >0.999 when it is out), so any cutoff in (1e-3, 0.99) gives
# the same answer; 0.5 is the DeepLabCut default.
TONGUE_P_CUTOFF = 0.5

# Tongue y-position percentiles for the 3-way split required by the task.
TONGUE_PCTILES = (40.0, 60.0)

# CCF medio-lateral midline in micrometres, used to assign a unit to a
# hemisphere.  Same constant and same sign convention as
# preprocessing_DJ_2022Aug.helper_get_neuron_id_area (``ML_mid_coor = 5700``).
ML_MIDLINE = 5700.0

# Session selection, from the data paper (methods.txt):  "We selected
# experimental sessions for analysis based on following criteria: overall
# behavioral performance (> 65%), and at least 50 correct lick left and lick
# right trials each."  Performance is "the fraction of correct control trials
# (i.e. no photostimulation), excluding any early lick trials".
MIN_PERFORMANCE = 0.65
MIN_CORRECT_PER_DIRECTION = 50

CHOICE_VALUES = ['left', 'right', 'no lick']
OUTCOME_VALUES = ['ignore', 'miss', 'hit']
EARLY_LICK_VALUES = ['no', 'yes']
TONGUE_VALUES = ['y < 40th pctile', '40th-60th pctile', 'y > 60th pctile',
                 'not visible']
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}

INPUT_NAMES = ['time_from_tone_onset', 'photostim_on']
OUTPUT_NAMES = ['lick_direction', 'outcome', 'early_lick', 'tongue_y_position']


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _str(arr):
    """Decode an HDF5 array of variable-length strings to a numpy unicode array."""
    return np.asarray(arr).astype(str)


def _parse_float(value):
    """Parse the string-valued trial columns; 'N/A' (and anything else that is
    not a number) becomes NaN."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def probe_targets(nwb):
    """Map electrode-group name -> targeted brain region, e.g. 'left ALM'.

    The target is stored as a JSON blob in the ElectrodeGroup ``location``
    attribute and is the region the probe was aimed at (Table S2 of the data
    paper).
    """
    groups = nwb['general/extracellular_ephys']
    out = {}
    for key in groups:
        if key == 'electrodes':
            continue
        try:
            out[key] = json.loads(groups[key].attrs['location'])['brain_regions']
        except Exception:
            out[key] = ''
    return out


def unit_regions(nwb, good):
    """Hemisphere-qualified coarse brain region for every good unit.

    Region names follow the grouping used by the reference preprocessing code
    (``preprocessing_DJ_2022Aug.process_one_sess`` loops over
    ALM/Medulla/Midbrain/Striatum/Thalamus/Pons/Cerebellum/Hypothalamus/
    Hippocampus/Orbital/OtherCortex/Olfactory/CorticalSubplate/Pallidum and over
    the two hemispheres, naming each group ``side_region``).  The group is
    derived from the unit's own Allen CCF annotation (``units/anno_name``) and
    the hemisphere from the CCF ML coordinate of its peak electrode.
    """
    units = nwb['units']
    electrodes = nwb['general/extracellular_ephys/electrodes']
    group_name = _str(electrodes['group_name'][:])
    ml = electrodes['x'][:]                      # CCF ML coordinate, micrometres
    targets = probe_targets(nwb)

    elec_idx = units['electrodes'][:][good]
    anno = _str(units['anno_name'][:])[good]
    target = np.array([targets.get(g, '') for g in group_name[elec_idx]])
    ml_unit = ml[elec_idx]

    # Fall back on the probe's targeted hemisphere if the unit has no CCF
    # coordinate (does not happen in this release, but keeps the code total).
    side = np.where(np.isnan(ml_unit),
                    np.array([t.split(' ')[0] if t else 'left' for t in target]),
                    np.where(ml_unit >= ML_MIDLINE, 'left', 'right'))

    regions = []
    for a, t, s in zip(anno, target, side):
        group = coarse_region(a, t)
        if group is None:
            raise ValueError('good unit without a CCF annotation')
        regions.append('%s %s' % (s, group))
    return regions


def observed_trials(nwb, good, trial_start):
    """Trials during which every good unit was actually being recorded.

    ``units/obs_intervals`` lists, per unit, the intervals over which its spike
    train is defined; in this release those intervals are exactly the trial
    windows of the trials the probe was recording during.  Most sessions cover
    every trial, but in a handful the ephys recording covers only the first part
    of the behavioural session, and the trials beyond it would otherwise look
    like trials in which the whole population fell silent.
    """
    units = nwb['units']
    obs = units['obs_intervals'][:]
    index = units['obs_intervals_index'][:]
    starts = np.concatenate([[0], index[:-1]])
    covered = np.ones(len(trial_start), dtype=bool)
    for u in np.flatnonzero(good):
        covered &= np.isin(trial_start, obs[starts[u]:index[u], 0])
    return covered


def units_good_on_trials(nwb, good, trial_idx):
    """Mask over the good units that are quality-flagged on every analysed trial.

    ``units/is_good_trials`` marks, per unit and trial, whether the unit passed
    quality control on that trial.  A unit has to occupy one row of a fixed
    neuron-by-time matrix on every trial, so units that fail on any analysed
    trial are dropped instead (0.8% of units across the release).
    """
    flags = nwb['units']['is_good_trials'][:][good][:, trial_idx]
    return flags.all(axis=1)


def bin_spikes(nwb, good, go_times):
    """Binned firing rates, shape (n_units, n_trials, N_BINS), in spikes/s.

    Spikes are counted in the fixed window around each trial's go cue, exactly as
    the reference pipeline does (``preprocessing_DJ_2022Aug.process_one_area``
    bins the go-cue-aligned spike train over a fixed window and divides by the
    bin width, irrespective of how long the trial itself lasted).

    Caveat worth stating explicitly: in this NWB release the spike train of every
    unit is observed only inside ``[trial start_time, trial stop_time]``
    (``units/obs_intervals`` is exactly the trial table).  Error ("miss") trials
    are terminated early by the timeout, so their recorded data typically end
    ~0.4 s after the go cue and the remaining bins of the window contain no
    spikes.  Those bins are reported as a rate of zero, which is what the
    reference pipeline produces as well; no imputation is invented here.
    """
    units = nwb['units']
    spike_index = units['spike_times_index'][:]
    starts = np.concatenate([[0], spike_index[:-1]])
    unit_ids = np.flatnonzero(good)

    n_trials = len(go_times)
    # (n_trials, N_BINS+1) absolute bin edges, flattened for one searchsorted per unit
    edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()

    rates = np.empty((len(unit_ids), n_trials, N_BINS), dtype=np.float32)
    spike_times = units['spike_times']
    for i, u in enumerate(unit_ids):
        st = spike_times[starts[u]:spike_index[u]]
        idx = np.searchsorted(st, edges).reshape(n_trials, N_BINS + 1)
        rates[i] = np.diff(idx, axis=1) / BIN_WIDTH
    return rates


def tone_onsets(nwb, trial_start, trial_stop, go_times):
    """Absolute time of the instruction tone that preceded each go cue.

    The instruction is a train of pure tones played over the sample epoch, so the
    sample-epoch onset is the tone onset.  Licking during the sample or delay
    epoch replays the epoch (data paper), which is why there are more
    ``sample_start_times`` than trials; the tone the animal actually answered is
    the last one before the go cue, so that is the one used here.
    """
    sample = nwb['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
    onsets = np.full(len(go_times), np.nan)
    lo = np.searchsorted(sample, trial_start, side='left')
    hi = np.searchsorted(sample, go_times, side='right')
    for i in range(len(go_times)):
        if hi[i] > lo[i]:
            onsets[i] = sample[hi[i] - 1]
    return onsets


def photostim_series(trials, trial_start, go_times):
    """Binary (n_trials, N_BINS) mask of when the laser was on.

    ``photostim_onset`` is given relative to the start of the trial and
    ``photostim_duration`` in seconds; both are 'N/A' on control trials.  A bin is
    marked as stimulated when it overlaps the stimulation interval.
    """
    onset = np.array([_parse_float(v) for v in _str(trials['photostim_onset'][:])])
    duration = np.array([_parse_float(v) for v in _str(trials['photostim_duration'][:])])

    stim = np.zeros((len(go_times), N_BINS), dtype=np.float32)
    on = trial_start + onset - go_times          # relative to the go cue
    off = on + duration
    valid = np.isfinite(on) & np.isfinite(off)
    for i in np.flatnonzero(valid):
        stim[i] = ((BIN_EDGES[1:] > on[i]) & (BIN_EDGES[:-1] < off[i])).astype(np.float32)
    return stim


def tongue_classes(nwb, go_times):
    """Discretised tongue y-position, shape (n_trials, N_BINS), values 0..3.

    Per the task: 0 below the session's 40th percentile, 1 between the 40th and
    60th, 2 above the 60th, 3 when the tongue is not visible.  Percentiles are
    taken over the binned y-values of the whole session (all bins of all analysed
    trials in which the tongue was visible), i.e. over the same quantity that is
    being discretised, so that the three visible classes hold 40%/20%/40% of the
    visible bins as intended.

    A bin is "visible" when at least one of its video frames has DeepLabCut
    likelihood above ``TONGUE_P_CUTOFF``; its y-value is the mean over those
    frames.  Bins with no video frames at all (the side-view video, like the
    spike train, only covers the trial itself) are likewise "not visible".
    """
    track = nwb['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
    ts = track['timestamps'][:]
    data = track['data'][:]

    n_trials = len(go_times)
    y_binned = np.full((n_trials, N_BINS), np.nan)
    for i, go in enumerate(go_times):
        lo, hi = np.searchsorted(ts, [go + T_START, go + T_STOP])
        if hi <= lo:
            continue
        chunk = data[lo:hi, :]
        y = chunk[:, 1]
        visible = chunk[:, 2] >= TONGUE_P_CUTOFF
        if not visible.any():
            continue
        which = np.searchsorted(BIN_EDGES, ts[lo:hi] - go, side='right') - 1
        ok = visible & (which >= 0) & (which < N_BINS)
        if not ok.any():
            continue
        counts = np.bincount(which[ok], minlength=N_BINS)
        sums = np.bincount(which[ok], weights=y[ok], minlength=N_BINS)
        seen = counts > 0
        y_binned[i, seen] = sums[seen] / counts[seen]

    seen = np.isfinite(y_binned)
    classes = np.full((n_trials, N_BINS), 3, dtype=np.int8)
    if seen.any():
        p40, p60 = np.percentile(y_binned[seen], TONGUE_PCTILES)
        vals = y_binned[seen]
        classes[seen] = np.where(vals < p40, 0, np.where(vals <= p60, 1, 2)).astype(np.int8)
    else:
        p40 = p60 = np.nan
    return classes, float(p40), float(p60), float(seen.mean())


# --------------------------------------------------------------------------- #
# Per-session conversion
# --------------------------------------------------------------------------- #

def process_session(path, out_dir):
    """Convert one NWB session; returns a summary dict (or None if rejected)."""
    name = os.path.basename(path).replace('.nwb', '')
    with h5py.File(path, 'r') as nwb:
        subject = nwb['general/subject/subject_id'][()].decode()
        trials = nwb['intervals/trials']
        trial_start = trials['start_time'][:]
        trial_stop = trials['stop_time'][:]
        outcome = _str(trials['outcome'][:])
        instruction = _str(trials['trial_instruction'][:])
        early = _str(trials['early_lick'][:]) == 'early'
        auto_water = trials['auto_water'][:] != 0
        free_water = trials['free_water'][:] != 0
        stim_trial = _str(trials['photostim_onset'][:]) != 'N/A'

        go_times = nwb['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
        assert len(go_times) == len(trial_start), 'go cue count != trial count'
        assert np.all((go_times >= trial_start) & (go_times <= trial_stop))

        # ---- trial curation ------------------------------------------------ #
        # The method paper excludes "water administration regardless of the
        # animals' choice (free water trials)" because the outcome on those
        # trials does not reflect the animal's decision; the same holds for
        # auto-water trials, so both are dropped.
        #
        # That paper also excludes photoinhibition, early-lick and ignore
        # trials.  Those three cannot be dropped here: photostimulation is a
        # required decoder input and early lick / no-lick are required decoder
        # output classes, so all three are kept.  This is the only deliberate
        # departure from the reference curation and it is forced by the task.
        #
        # Trials outside the ephys recording (see observed_trials) are dropped
        # as well; the quality-control classification is read first because the
        # coverage is defined per unit.
        good = _str(nwb['units']['classification'][:]) == 'good'
        keep = ~(auto_water | free_water)
        if good.any():
            keep &= observed_trials(nwb, good, trial_start)

        # ---- session curation ---------------------------------------------- #
        # Data paper: overall performance > 65% and >= 50 correct lick-left and
        # lick-right trials, measured on control (no photostimulation) trials
        # excluding early-lick trials.
        control = keep & ~stim_trial & ~early
        hit = outcome == 'hit'
        miss = outcome == 'miss'
        responded = (hit | miss) & control
        performance = hit[responded].mean() if responded.any() else 0.0
        n_left = int(np.sum(hit & control & (instruction == 'left')))
        n_right = int(np.sum(hit & control & (instruction == 'right')))
        rejected = (performance <= MIN_PERFORMANCE
                    or n_left < MIN_CORRECT_PER_DIRECTION
                    or n_right < MIN_CORRECT_PER_DIRECTION)
        summary = dict(session=name, subject=subject, path=path,
                       n_trials_total=int(len(trial_start)),
                       n_trials_kept=int(keep.sum()),
                       performance=float(performance),
                       n_correct_left=n_left, n_correct_right=n_right,
                       rejected=bool(rejected))
        if rejected:
            return summary

        trial_idx = np.flatnonzero(keep)
        go = go_times[trial_idx]

        # ---- neurons -------------------------------------------------------- #
        # ``classification`` holds the output of the region-specific quality
        # control classifiers described in the spike-sorting white paper: units
        # labelled 'good' are exactly the ones the data paper analyses ("lists of
        # units that were labeled as 'good' and were used for the analysis in
        # this paper").  They are also exactly the units carrying a CCF
        # annotation in this release.
        summary['n_units_total'] = int(len(good))
        summary['n_units_good'] = int(good.sum())
        if good.sum() == 0 or len(trial_idx) < 2:
            summary['rejected'] = True
            summary['reject_reason'] = 'no good units or fewer than 2 trials'
            return summary
        good[good] = units_good_on_trials(nwb, good, trial_idx)
        summary['n_units'] = int(good.sum())
        if good.sum() == 0:
            summary['rejected'] = True
            summary['reject_reason'] = 'no unit passes quality control on all trials'
            return summary

        regions = unit_regions(nwb, good)
        rates = bin_spikes(nwb, good, go)                 # (units, trials, bins)

        # ---- decoder inputs -------------------------------------------------- #
        tone = tone_onsets(nwb, trial_start, trial_stop, go_times)[trial_idx]
        missing_tone = ~np.isfinite(tone)
        if missing_tone.any():
            # No sample-epoch onset recorded between trial start and go cue:
            # fall back to the nominal 0.65 s sample + 1.2 s delay structure.
            tone[missing_tone] = go[missing_tone] - 1.85
        summary['n_trials_missing_tone'] = int(missing_tone.sum())
        time_from_tone = (go[:, None] + BIN_CENTERS[None, :]
                          - tone[:, None]).astype(np.float32)
        stim = photostim_series(trials, trial_start, go_times)[trial_idx]

        # ---- decoder outputs -------------------------------------------------- #
        # Lick direction: 'hit' means the animal licked the instructed spout,
        # 'miss' the opposite one and 'ignore' that it did not lick.  Verified
        # against the recorded left/right lick times (perfect agreement).
        other = np.where(instruction == 'left', 'right', 'left')
        licked = np.where(outcome == 'hit', instruction,
                          np.where(outcome == 'miss', other, 'no lick'))
        choice = np.array([CHOICE_VALUES.index(v) for v in licked], dtype=np.int8)
        outcome_code = np.array([OUTCOME_CODE[v] for v in outcome], dtype=np.int8)
        early_code = early.astype(np.int8)

        tongue, p40, p60, frac_visible = tongue_classes(nwb, go)
        summary.update(tongue_p40=p40, tongue_p60=p60,
                       frac_tongue_visible=frac_visible)

        choice = choice[trial_idx]
        outcome_code = outcome_code[trial_idx]
        early_code = early_code[trial_idx]

        # ---- assemble per-trial arrays ----------------------------------------- #
        neural, inputs, outputs = [], [], []
        ones = np.ones(N_BINS, dtype=np.int8)
        for j in range(len(trial_idx)):
            neural.append(np.ascontiguousarray(rates[:, j, :]))
            inputs.append(np.stack([time_from_tone[j], stim[j]]).astype(np.float32))
            outputs.append(np.stack([choice[j] * ones, outcome_code[j] * ones,
                                     early_code[j] * ones, tongue[j]]).astype(np.int8))

        summary.update(
            n_neurons=len(regions),
            n_trials=len(trial_idx),
            regions=regions,
            n_photostim_trials=int(np.sum(stim.any(axis=1))),
            choice_counts=np.bincount(choice, minlength=3).tolist(),
            outcome_counts=np.bincount(outcome_code, minlength=3).tolist(),
            early_counts=np.bincount(early_code, minlength=2).tolist(),
            tongue_counts=np.bincount(tongue.ravel(), minlength=4).tolist(),
            mean_rate=float(rates.mean()),
        )

    out_path = os.path.join(out_dir, name + '.pkl')
    with open(out_path, 'wb') as fh:
        pickle.dump({'neural': neural, 'input': inputs, 'output': outputs,
                     'regions': regions, 'subject': subject, 'session': name},
                    fh, protocol=4)
    summary['cache'] = out_path
    return summary


def _worker(args):
    path, out_dir = args
    try:
        return process_session(path, out_dir)
    except Exception:
        traceback.print_exc()
        return dict(session=os.path.basename(path), rejected=True,
                    error=traceback.format_exc())


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='/app/data')
    parser.add_argument('--out', default='/app/converted_data.pkl')
    parser.add_argument('--cache-dir', default='/app/work/sessions')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--limit', type=int, default=None,
                        help='process only the first N sessions (for testing)')
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
    if args.limit:
        files = files[:args.limit]
    print('found %d NWB sessions' % len(files), flush=True)

    jobs = [(f, args.cache_dir) for f in files]
    summaries = []
    with Pool(args.workers) as pool:
        for i, s in enumerate(pool.imap(_worker, jobs)):
            summaries.append(s)
            print('[%3d/%d] %s %s' % (i + 1, len(files), s['session'],
                                      'REJECTED' if s.get('rejected') else
                                      'kept %d trials, %d units'
                                      % (s.get('n_trials', 0), s.get('n_neurons', 0))),
                  flush=True)

    kept = [s for s in summaries if not s.get('rejected')]
    print('\nkeeping %d of %d sessions' % (len(kept), len(summaries)), flush=True)

    brain_regions = sorted({r for s in kept for r in s['regions']})
    region_index = {r: i for i, r in enumerate(brain_regions)}
    subjects = sorted({s['subject'] for s in kept})
    subject_index = {s: i for i, s in enumerate(subjects)}

    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': subjects,
        'subject_idx': np.array([subject_index[s['subject']] for s in kept], dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': [np.array([region_index[r] for r in s['regions']], dtype=np.int64)
                             for s in kept],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': [CHOICE_VALUES, OUTCOME_VALUES, EARLY_LICK_VALUES, TONGUE_VALUES],
    }

    for i, s in enumerate(kept):
        with open(s['cache'], 'rb') as fh:
            sess = pickle.load(fh)
        data['neural'].append(sess['neural'])
        data['input'].append(sess['input'])
        data['output'].append(sess['output'])
        if (i + 1) % 20 == 0:
            print('  loaded %d/%d sessions' % (i + 1, len(kept)), flush=True)

    session_info = [{k: v for k, v in s.items()
                     if k not in ('regions', 'cache', 'path')} for s in kept]

    data['metadata'] = {
        'task_description':
            'Head-fixed mice performed an auditory delayed-response task (Chen et al. '
            '2024, DANDI 000363). An instruction tone (3 kHz or 12 kHz, three 150 ms '
            'pips) played during a 0.65 s sample epoch told the mouse which lick port '
            'to lick; after a 1.2 s delay epoch an auditory go cue released the '
            'response. Licking the instructed port gave water (hit), licking the other '
            'port gave a timeout (miss), not licking within the 1.5 s answer period '
            'was an ignore trial, and licking during the sample or delay epoch was an '
            'early lick (which replayed the epoch). On ~25% of trials ALM was '
            'photoinhibited during the delay. Spiking was recorded brain-wide with '
            'Neuropixels probes and the face was filmed from the side at ~294 Hz and '
            'tracked with DeepLabCut. The decoder is given binned firing rates, the '
            'time since the instruction tone and whether the laser is on, and must '
            'predict the lick direction, the outcome, whether the trial had an early '
            'lick, and the discretised tongue y-position.',
        'temporal_alignment_event': 'auditory go cue onset (end of the delay epoch)',
        'off_start': T_START,
        'off_end': T_STOP,
        'time_bin_size': BIN_WIDTH * 1000.0,
        'neural_units': 'firing rate, spikes/s (spike count per 50 ms bin / 0.05 s)',
        'n_sessions': len(kept),
        'n_sessions_available': len(summaries),
        'n_subjects': len(subjects),
        'source': 'DANDI 000363, Mesoscale Activity Map Dataset (Chen et al. 2024)',
        'references': [
            'Chen S. et al. Brain-wide neural activity underlying memory-guided '
            'movement. Cell (2024).',
            'Wang Z.A., Kurgyis B. et al. Brain-wide analysis reveals movement '
            'encoding structured across and within brain areas. Nat Neurosci (2025).',
            'Chen S., Liu Y. et al. Spike sorting and quality control white paper, '
            'doi:10.25378/janelia.24066108.v1',
        ],
        'unit_quality_control':
            "units/classification == 'good', i.e. the units kept by the "
            'region-specific quality-control classifiers of the spike-sorting white '
            'paper; these are the units analysed in the data paper.',
        'session_selection':
            'data-paper criteria on control (non-photostimulation, non-early-lick) '
            'trials: performance > 65%% and >= %d correct lick-left and lick-right '
            'trials each.' % MIN_CORRECT_PER_DIRECTION,
        'trial_selection':
            'auto-water and free-water trials excluded (outcome does not reflect the '
            'animal\'s choice) and trials outside the units\' observation intervals '
            'excluded (the ephys recording covers only part of the behavioural '
            'session in a few files); photostimulation, early-lick and ignore trials '
            'kept because the decoder task requires them, although the method paper '
            'excludes them from its analyses.',
        'neuron_selection':
            "units/classification == 'good', further restricted to units flagged good "
            'on every analysed trial (units/is_good_trials).',
        'brain_region_definition':
            'coarse groups of the reference preprocessing pipeline '
            '(ALM/Medulla/Midbrain/Striatum/Thalamus/Pons/Cerebellum/Hypothalamus/'
            'Hippocampus/Orbital/OtherCortex/Olfactory/CorticalSubplate/Pallidum), '
            "derived from each unit's Allen CCF annotation, prefixed by the "
            'hemisphere of its peak electrode (CCF ML midline 5700 um). ALM is not a '
            'CCF structure, so motor-cortex units recorded on ALM-targeted probes are '
            'labelled ALM and other motor-cortex units OtherCortex.',
        'tongue_discretisation':
            'per-session 40th/60th percentiles of the binned tongue y-position over '
            'all bins in which DeepLabCut likelihood >= %.2f; bins with no confident '
            'detection (including bins with no video) are class "not visible".'
            % TONGUE_P_CUTOFF,
        'known_limitation':
            'Spikes and video in this release exist only between each trial\'s '
            'start_time and stop_time. Error (miss) trials are cut short by the '
            'timeout, so the last ~1 s of their -2.5..1.5 s window holds no recorded '
            'spikes and is binned as a zero rate, as in the reference pipeline.',
        'session_info': session_info,
    }

    print('writing %s' % args.out, flush=True)
    with open(args.out, 'wb') as fh:
        pickle.dump(data, fh, protocol=4)

    n_trials = sum(len(x) for x in data['neural'])
    n_neurons = sum(len(x) for x in data['brain_region_idx'])
    print('sessions: %d, subjects: %d, trials: %d, neurons: %d, regions: %d'
          % (len(kept), len(subjects), n_trials, n_neurons, len(brain_regions)))
    with open(os.path.join(os.path.dirname(args.out) or '.',
                           'conversion_summary.json'), 'w') as fh:
        json.dump(session_info, fh, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
