#!/usr/bin/env python3
"""
Convert the DANDI:000363 "Mesoscale Activity Map" NWB dataset (Chen et al. 2024)
into the decoder-ready dictionary format described in the task specification.

Usage
-----
    python -u convert_data.py <outpicklefile> [--full | --sample]
                              [--show-processing] [--jobs N]

Options
-------
    --full             process all sessions (default)
    --sample           process only 2 sessions (quick test)
    --show-processing  save a multi-panel figure of every processing step for
                       up to 2 sessions, as processing_<session_id>.png
    --jobs N           number of worker processes (default: 12)

Design decisions and their justification are documented in CONVERSION_NOTES.md
(Step 5).  In brief:

  * neural   : firing rate (Hz) of every classifier-QC 'good' unit, in 80
               non-overlapping 50 ms bins spanning [-2.5, +1.5] s around the Go cue
  * input    : [time since the instruction-tone onset (s), photostimulation on/off]
  * output   : [choice, outcome, early lick, discretised tongue y position]
  * trials   : only trials with ephys coverage and without auto/free water
  * neurons  : only units with units.classification == 'good'
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time
import warnings
from collections import OrderedDict

import numpy as np

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#                               configuration                                  #
# --------------------------------------------------------------------------- #

DATA_DIR = "/app/data"

T_PRE = 2.5          # s before the Go cue
T_POST = 1.5         # s after the Go cue
BIN_SIZE = 0.05      # s, non-overlapping bins
N_BINS = int(round((T_PRE + T_POST) / BIN_SIZE))          # 80
BIN_EDGES_REL = -T_PRE + np.arange(N_BINS + 1) * BIN_SIZE  # (81,) relative to Go
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2.0      # (80,)

TONGUE_LIKELIHOOD_THRESH = 0.9   # DeepLabCut likelihood above which the tongue is "visible"
TONGUE_PCT_LOW = 40.0            # percentile boundaries of the per-session discretisation
TONGUE_PCT_HIGH = 60.0

# CCF geometry (micrometres). Matches the reference code
# (VideoAnalysisUtils: ML midline 5700, global_offset_vec = (5700, 0, 5400)).
CCF_ML_MIDLINE = 5700.0
CCF_AP_BREGMA = 5400.0
ALM_MIN_AP_MM = 1.75   # ALM is centred on AP +2.5 mm, ML 1.5 mm (datapaper)
ALM_MAX_ML_MM = 2.0

INPUT_NAMES = ["time_from_tone_onset", "photostim_on"]
OUTPUT_NAMES = ["choice", "outcome", "early_lick", "tongue_y"]
OUTPUT_VALUES = [
    ["left", "right", "no lick"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["<40th pct", "40-60th pct", ">60th pct", "not visible"],
]

# The 14 coarse brain areas used by the reference preprocessing pipeline
# (VideoAnalysisUtils/preprocessing_DJ_2022Aug.py, process_one_sess).
BRAIN_REGIONS = [
    "ALM", "Orbital", "OtherCortex", "Olfactory", "Hippocampus", "CorticalSubplate",
    "Striatum", "Pallidum", "Thalamus", "Hypothalamus", "Midbrain", "Pons",
    "Medulla", "Cerebellum",
]
BRAIN_REGION_IDX = {r: i for i, r in enumerate(BRAIN_REGIONS)}


# --------------------------------------------------------------------------- #
#                 CCF annotation -> coarse brain area mapping                  #
# --------------------------------------------------------------------------- #
# Substring rules over the Allen CCFv3 structure names stored in units.anno_name.
# Validated against the per-area good-unit counts reported in the data paper
# (Medulla 2928 exact, Midbrain -0.2 %, Striatum +1.0 %, Thalamus +1.6 %, ALM +1.9 %).

_MEDULLA = [
    'Gigantocellular reticular nucleus', 'Intermediate reticular nucleus',
    'Magnocellular reticular nucleus', 'Parvicellular reticular nucleus',
    'Medullary reticular nucleus', 'Paragigantocellular reticular nucleus',
    'Parapyramidal nucleus', 'Spinal nucleus of the trigeminal', 'Medial vestibular nucleus',
    'Lateral vestibular nucleus', 'Spinal vestibular nucleus', 'Superior vestibular nucleus',
    'Inferior olivary complex', 'Nucleus of the solitary tract', 'Parasolitary nucleus',
    'Facial motor nucleus', 'Hypoglossal nucleus', 'External cuneate nucleus',
    'Cuneate nucleus', 'Gracile nucleus', 'Nucleus raphe magnus', 'Nucleus raphe obscurus',
    'Nucleus raphe pallidus', 'Lateral reticular nucleus', 'Nucleus x', 'Nucleus y',
    'Nucleus of Roller', 'Dorsal motor nucleus of the vagus nerve', 'Nucleus ambiguus',
    'Area postrema', 'Linear nucleus of the medulla', 'Infracerebellar nucleus', 'Medulla',
    'Dorsal cochlear nucleus', 'Ventral cochlear nucleus', 'Nucleus prepositus',
    'Nucleus of the trapezoid body', 'Efferent vestibular nucleus', 'Perihypoglossal nuclei',
    'Intercalated nucleus', 'Abducens nucleus',
]
_PONS = [
    'Pontine reticular nucleus', 'Tegmental reticular nucleus', 'Parabrachial nucleus',
    'Koelliker-Fuse subnucleus', 'Locus ceruleus', 'Nucleus of the lateral lemniscus',
    'Superior olivary complex', 'Pons', 'Principal sensory nucleus of the trigeminal',
    'Motor nucleus of trigeminal', 'Supratrigeminal nucleus', 'Laterodorsal tegmental nucleus',
    'Sublaterodorsal nucleus', 'Pontine gray', 'Nucleus incertus', 'Nucleus raphe pontis',
    'Barrington', 'Dorsal tegmental nucleus', 'Subceruleus nucleus', 'Peritrigeminal zone',
    'Nucleus sagulum', 'Superior central nucleus raphe',
]
_CEREBELLUM = [
    'Lobule', 'Lobules', 'Simple lobule', 'Crus 1', 'Crus 2', 'Paramedian lobule',
    'Copula pyramidis', 'Nodulus', 'Uvula', 'Pyramus', 'Declive', 'Folium', 'Culmen',
    'Central lobule', 'Lingula', 'Ansiform', 'Paraflocculus', 'Flocculus',
    'Fastigial nucleus', 'Interposed nucleus', 'Dentate nucleus', 'Vermal regions',
    'Hemispheric regions', 'Cerebellum', 'Cerebellar cortex', 'Cerebellar nuclei', 'Arbor vitae',
]
_MIDBRAIN = [
    'Superior colliculus', 'Inferior colliculus', 'Midbrain reticular nucleus',
    'Substantia nigra', 'Red nucleus', 'Periaqueductal gray', 'Ventral tegmental area',
    'Pedunculopontine nucleus', 'Anterior pretectal nucleus', 'Posterior pretectal nucleus',
    'Nucleus of the optic tract', 'Nucleus of the brachium of the inferior colliculus',
    'Midbrain', 'Cuneiform nucleus', 'Oculomotor nucleus', 'Trochlear nucleus',
    'Edinger-Westphal nucleus', 'Interpeduncular nucleus', 'Dorsal nucleus raphe',
    'Nucleus raphe', 'Midbrain trigeminal nucleus', 'Olivary pretectal nucleus',
    'Medial terminal nucleus of the accessory optic tract',
    'Dorsal terminal nucleus of the accessory optic tract', 'Nucleus of Darkschewitsch',
    'Interstitial nucleus of Cajal', 'Nucleus of the posterior commissure',
    'Peripeduncular nucleus', 'Midbrain reticular nucleus, retrorubral area',
    'Lateral terminal nucleus', 'Parabigeminal nucleus', 'Subcommissural organ',
    'Midbrain, sensory related', 'Midbrain, motor related', 'Midbrain, behavioral state related',
]
_THALAMUS = [
    'thalamus', 'thalamic', 'Thalamus', 'habenula', 'Subparafascicular',
    'Parafascicular nucleus', 'Medial geniculate complex', 'lateral geniculate complex',
    'Anteroventral nucleus', 'Anteromedial nucleus', 'Anterodorsal nucleus',
    'Interanterodorsal nucleus', 'Lateral dorsal nucleus', 'Paracentral nucleus',
    'Central medial nucleus', 'Central lateral nucleus', 'Rhomboid nucleus',
    'Reuniens nucleus', 'Perireunensis nucleus', 'Nucleus of reuniens', 'Submedial nucleus',
    'Posterior complex', 'Suprageniculate nucleus', 'Ethmoid nucleus',
    'Xiphoid thalamic nucleus', 'Intermediodorsal nucleus', 'Mediodorsal nucleus',
    'Paraventricular nucleus of the thalamus', 'Parataenial nucleus',
    'Posterior intralaminar thalamic nucleus', 'Medial habenula', 'Lateral habenula',
    'Reticular nucleus of the thalamus',
]
_HYPOTHALAMUS = [
    'hypothalamic', 'Hypothalamus', 'Lateral hypothalamic area',
    'Posterior hypothalamic nucleus', 'Tuberomammillary nucleus', 'Mammillary',
    'Preoptic area', 'Lateral preoptic area', 'Medial preoptic', 'Zona incerta',
    'Fields of Forel', 'Subthalamic nucleus', 'Parasubthalamic nucleus',
    'Arcuate hypothalamic nucleus', 'Dorsomedial nucleus', 'Ventromedial hypothalamic nucleus',
    'Supraoptic nucleus', 'Anterior hypothalamic nucleus', 'Periventricular',
    'Paraventricular hypothalamic nucleus', 'Zona', 'Median eminence', 'Tuberal nucleus',
    'Lateral mammillary nucleus', 'Supramammillary nucleus', 'Posterior periventricular nucleus',
]
_STRIATUM = [
    'Caudoputamen', 'Nucleus accumbens', 'Fundus of striatum', 'Olfactory tubercle',
    'Lateral septal nucleus', 'Septofimbrial nucleus', 'Septohippocampal nucleus',
    'Medial septal nucleus', 'Striatum', 'Central amygdalar nucleus',
    'Intercalated amygdalar nucleus', 'Anterior amygdalar area', 'Medial amygdalar nucleus',
    'Striatum-like amygdalar nuclei', 'Striatum dorsal region', 'Striatum ventral region',
    'Diagonal band nucleus', 'Triangular nucleus of septum',
]
_PALLIDUM = [
    'Globus pallidus', 'Substantia innominata', 'Pallidum', 'Magnocellular nucleus',
    'Bed nucleus of the anterior commissure', 'Medial septal complex',
    'Nucleus of the diagonal band', 'Bed nuclei of the stria terminalis',
]
_HIPPOCAMPUS = [
    'Field CA1', 'Field CA2', 'Field CA3', 'Dentate gyrus', 'Subiculum', 'Postsubiculum',
    'Presubiculum', 'Parasubiculum', 'Prosubiculum', 'Entorhinal area',
    'Hippocampal formation', 'Hippocampal region', 'Induseum griseum', 'Fasciola cinerea',
    'Retrohippocampal region', 'Ammon',
]
_OLFACTORY = [
    'Anterior olfactory nucleus', 'Piriform area', 'Olfactory areas', 'Accessory olfactory bulb',
    'Main olfactory bulb', 'Taenia tecta', 'Dorsal peduncular area',
    'Nucleus of the lateral olfactory tract', 'Cortical amygdalar area',
    'Piriform-amygdalar area', 'Postpiriform transition area', 'Olfactory bulb',
]
_CORTICAL_SUBPLATE = [
    'Claustrum', 'Endopiriform nucleus', 'Lateral amygdalar nucleus',
    'Basolateral amygdalar nucleus', 'Basomedial amygdalar nucleus',
    'Posterior amygdalar nucleus', 'Cortical subplate',
]
_ORBITAL = ['Orbital area']
# Frontal motor cortex; only the anterolateral part of it is ALM (coordinate test below).
_FRONTAL_MOTOR = ['Secondary motor area', 'Primary motor area', 'Frontal pole']

_GROUPS = [
    ('Cerebellum', _CEREBELLUM), ('Medulla', _MEDULLA), ('Pons', _PONS),
    ('Midbrain', _MIDBRAIN), ('Thalamus', _THALAMUS), ('Hypothalamus', _HYPOTHALAMUS),
    ('Striatum', _STRIATUM), ('Pallidum', _PALLIDUM), ('Hippocampus', _HIPPOCAMPUS),
    ('Olfactory', _OLFACTORY), ('CorticalSubplate', _CORTICAL_SUBPLATE),
    ('Orbital', _ORBITAL), ('__FRONTAL_MOTOR__', _FRONTAL_MOTOR),
]

_ANNO_CACHE = {}


def _coarse_group(anno):
    """Coarse group for one CCF annotation string (before the ALM coordinate test)."""
    if anno in _ANNO_CACHE:
        return _ANNO_CACHE[anno]
    a = anno.strip().lower()
    out = 'OtherCortex'
    for name, keys in _GROUPS:
        if any(k.lower() in a for k in keys):
            out = name
            break
    _ANNO_CACHE[anno] = out
    return out


def assign_brain_regions(anno_names, xyz):
    """
    Assign each unit to one of BRAIN_REGIONS.

    anno_names : (n_units,) array of CCF annotation strings
    xyz        : (n_units, 3) CCF coordinates in micrometres (x=ML, y=DV, z=AP)
    """
    groups = np.array([_coarse_group(a) for a in anno_names], dtype=object)
    ap_mm = (CCF_AP_BREGMA - np.asarray(xyz[:, 2], dtype=float)) / 1000.0
    ml_mm = np.abs(np.asarray(xyz[:, 0], dtype=float) - CCF_ML_MIDLINE) / 1000.0
    is_frontal = groups == '__FRONTAL_MOTOR__'
    # Coordinates are NaN for a handful of units; treat those as failing the ALM test.
    with np.errstate(invalid='ignore'):
        is_alm = is_frontal & (ap_mm > ALM_MIN_AP_MM) & (ml_mm < ALM_MAX_ML_MM)
    groups[is_alm] = 'ALM'
    groups[is_frontal & ~is_alm] = 'OtherCortex'
    return np.array([BRAIN_REGION_IDX[g] for g in groups], dtype=np.int64)


# --------------------------------------------------------------------------- #
#                              helper functions                                #
# --------------------------------------------------------------------------- #

def bin_spike_rates(spike_times_list, go_times):
    """
    Bin spike times into firing rates, aligned to the Go cue.

    Equivalent to VideoAnalysisUtils.preprocessing_DJ_2022Aug.sliding_histogram with
    bin_width == stride == BIN_SIZE and rate=True, i.e. non-overlapping 50 ms bins
    covering [-T_PRE, +T_POST] relative to each trial's Go cue.

    spike_times_list : list of n_neurons sorted 1-D arrays of session-clock spike times
    go_times         : (n_trials,) session-clock Go cue times

    Returns (n_neurons, n_trials, N_BINS) float32 firing rate in Hz.
    """
    n_trials = len(go_times)
    edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()   # (n_trials*81,)
    out = np.empty((len(spike_times_list), n_trials, N_BINS), dtype=np.float32)
    for i, st in enumerate(spike_times_list):
        idx = np.searchsorted(st, edges).reshape(n_trials, N_BINS + 1)
        out[i] = np.diff(idx, axis=1)
    out /= np.float32(BIN_SIZE)
    return out


def bin_visible_mean(times, values, visible, go_times):
    """
    Average `values` over the frames inside each 50 ms bin that are marked `visible`.

    times   : (n_frames,) sorted session-clock frame times
    values  : (n_frames,) values to average (tongue y)
    visible : (n_frames,) bool
    Returns (mean_value, n_visible) each (n_trials, N_BINS); mean is NaN where n_visible==0.
    """
    n_trials = len(go_times)
    tv = times[visible]
    yv = values[visible]
    edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()
    idx = np.searchsorted(tv, edges).reshape(n_trials, N_BINS + 1)
    csum = np.concatenate([[0.0], np.cumsum(yv.astype(np.float64))])
    counts = np.diff(idx, axis=1)
    sums = np.diff(csum[idx], axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return mean, counts


def discretise_tongue(mean_y, counts):
    """
    Per-session discretisation of the tongue y position.
      0 : y  < 40th percentile of the session's visible y values
      1 : 40th <= y <= 60th percentile
      2 : y  > 60th percentile
      3 : tongue not visible in this bin
    """
    vis = counts > 0
    out = np.full(mean_y.shape, 3, dtype=np.int64)
    if not np.any(vis):
        return out, (np.nan, np.nan)
    p40, p60 = np.percentile(mean_y[vis], [TONGUE_PCT_LOW, TONGUE_PCT_HIGH])
    y = mean_y[vis]
    cls = np.where(y < p40, 0, np.where(y > p60, 2, 1))
    out[vis] = cls
    return out, (float(p40), float(p60))


def _read_ragged(col, index):
    """Read one row of a pynwb ragged (VectorIndex-backed) column."""
    return np.asarray(col[index])


# --------------------------------------------------------------------------- #
#                            per-session conversion                            #
# --------------------------------------------------------------------------- #

def process_session(path, show_processing=False, plot_dir="/app"):
    """Convert one NWB session. Returns a dict, or None if the session is unusable."""
    from pynwb import NWBHDF5IO

    t_start = time.time()
    timing = OrderedDict()

    with NWBHDF5IO(path, 'r', load_namespaces=True) as io:
        nwb = io.read()
        session_id = nwb.identifier
        subject = nwb.subject.description or nwb.subject.subject_id

        units = nwb.units
        if units is None or len(units) == 0:
            return {'session_id': session_id, 'skipped': 'no units'}
        classification = np.asarray(units['classification'][:])
        good = np.where(classification == 'good')[0]
        if len(good) == 0:
            return {'session_id': session_id, 'skipped': 'no good units'}

        # ---- trials -------------------------------------------------------- #
        trials = nwb.trials.to_dataframe()
        n_trials_all = len(trials)

        # Ephys coverage.  units.obs_intervals lists, for each unit, the intervals the
        # unit was observed in; every row matches exactly one trial's
        # (start_time, stop_time) and all good units of a session share the same list
        # (verified over all 173 sessions).  In 9 sessions it covers only part of the
        # behavioural trials -- usually a prefix, but in SC026_20190807_134913_s20 a
        # block starting at trial 126 -- so match by time rather than assuming a prefix.
        obs = np.asarray(units['obs_intervals'][int(good[0])], dtype=np.float64)

        be = nwb.acquisition['BehavioralEvents']
        go_all = np.asarray(be['go_start_times'].timestamps[:], dtype=np.float64)
        assert len(go_all) == n_trials_all, \
            f"{session_id}: {len(go_all)} go cues for {n_trials_all} trials"

        start_all = trials['start_time'].values.astype(np.float64)
        stop_all = trials['stop_time'].values.astype(np.float64)
        outcome_all = np.asarray(trials['outcome'].values, dtype=object).astype(str)
        instr_all = np.asarray(trials['trial_instruction'].values, dtype=object).astype(str)
        early_all = np.asarray(trials['early_lick'].values, dtype=object).astype(str)
        auto_all = np.asarray(trials['auto_water'].values).astype(int)
        free_all = np.asarray(trials['free_water'].values).astype(int)

        # ---- trial curation ------------------------------------------------- #
        ephys_covered = np.zeros(n_trials_all, dtype=bool)
        j = np.clip(np.searchsorted(start_all, obs[:, 0] + 1e-6) - 1, 0, n_trials_all - 1)
        matched = np.isclose(start_all[j], obs[:, 0]) & np.isclose(stop_all[j], obs[:, 1])
        ephys_covered[j[matched]] = True
        n_ephys = int(ephys_covered.sum())

        keep = ephys_covered.copy()                 # ephys coverage
        keep &= (auto_all == 0) & (free_all == 0)   # no auto / free water (reference)
        trial_idx = np.where(keep)[0]
        n_trials = len(trial_idx)
        if n_trials < 2:
            return {'session_id': session_id, 'skipped': f'only {n_trials} usable trials'}

        go = go_all[trial_idx]
        timing['trials'] = time.time() - t_start

        # ---- neural --------------------------------------------------------- #
        t0 = time.time()
        spike_lists = []
        for i in good:
            st = np.asarray(units['spike_times'][int(i)], dtype=np.float64)
            if st.size > 1 and not np.all(np.diff(st) >= 0):
                st = np.sort(st)
            spike_lists.append(st)
        timing['read_spikes'] = time.time() - t0

        t0 = time.time()
        rates = bin_spike_rates(spike_lists, go)      # (n_neurons, n_trials, N_BINS)
        timing['bin_spikes'] = time.time() - t0
        del spike_lists

        # A trial in which the entire simultaneously-recorded population fires zero
        # spikes carries no neural data at all: it happens when the recording stops
        # part-way through the session (typically the single last ephys trial).
        has_spikes = rates.sum(axis=(0, 2)) > 0
        n_empty = int(np.sum(~has_spikes))
        if n_empty:
            rates = rates[:, has_spikes, :]
            trial_idx = trial_idx[has_spikes]
            go = go_all[trial_idx]
            n_trials = len(trial_idx)
            if n_trials < 2:
                return {'session_id': session_id,
                        'skipped': f'only {n_trials} trials with spikes'}

        # ---- brain regions --------------------------------------------------- #
        t0 = time.time()
        anno = np.asarray(units['anno_name'][:])[good]
        elec_rows = np.asarray(units['electrodes'].target.data[:])[good]
        etab = nwb.electrodes.to_dataframe()
        xyz = etab[['x', 'y', 'z']].values[elec_rows]
        region_idx = assign_brain_regions(anno, xyz)
        timing['regions'] = time.time() - t0

        # ---- input 1: time from instruction-tone onset ------------------------ #
        t0 = time.time()
        sample_t = np.asarray(be['sample_start_times'].timestamps[:], dtype=np.float64)
        sample_t = np.sort(sample_t)
        # last sample-epoch onset at or before the Go cue (an early lick replays the
        # sample+delay epochs; the final presentation is the instructive one)
        si = np.searchsorted(sample_t, go, side='right') - 1
        tone_onset = np.where(si >= 0, sample_t[np.clip(si, 0, len(sample_t) - 1)], np.nan)
        bad_tone = ~np.isfinite(tone_onset) | (tone_onset < start_all[trial_idx])
        if np.any(bad_tone):
            # fall back to the median Go-to-tone interval of this session
            fallback = np.nanmedian(go[~bad_tone] - tone_onset[~bad_tone]) if np.any(~bad_tone) else 1.85
            tone_onset[bad_tone] = go[bad_tone] - fallback
        go_minus_tone = go - tone_onset                                   # (n_trials,)
        time_from_tone = (BIN_CENTERS_REL[None, :] + go_minus_tone[:, None]).astype(np.float32)

        # ---- input 2: photostimulation on/off --------------------------------- #
        ps_start = np.asarray(be['photostim_start_times'].timestamps[:], dtype=np.float64)
        ps_stop = np.asarray(be['photostim_stop_times'].timestamps[:], dtype=np.float64)
        order = np.argsort(ps_start)
        ps_start, ps_stop = ps_start[order], ps_stop[order]
        photostim = np.zeros((n_trials, N_BINS), dtype=np.float32)
        if len(ps_start):
            abs_centers = go[:, None] + BIN_CENTERS_REL[None, :]           # (n_trials, N_BINS)
            k = np.searchsorted(ps_start, abs_centers, side='right') - 1
            valid = k >= 0
            kk = np.clip(k, 0, len(ps_start) - 1)
            on = valid & (abs_centers >= ps_start[kk]) & (abs_centers < ps_stop[kk])
            photostim[on] = 1.0
        timing['inputs'] = time.time() - t0

        # ---- outputs 0-2: choice / outcome / early lick ------------------------ #
        t0 = time.time()
        outcome = outcome_all[trial_idx]
        instr = instr_all[trial_idx]
        early = early_all[trial_idx]

        opposite = np.where(instr == 'left', 'right', 'left')
        choice_str = np.where(outcome == 'ignore', 'no lick',
                              np.where(outcome == 'hit', instr, opposite))
        choice = np.select([choice_str == 'left', choice_str == 'right'], [0, 1], default=2).astype(np.int64)
        outcome_code = np.select([outcome == 'ignore', outcome == 'miss'], [0, 1], default=2).astype(np.int64)
        early_code = (early == 'early').astype(np.int64)

        # ---- output 3: tongue y ------------------------------------------------ #
        tongue = nwb.acquisition['BehavioralTimeSeries']['Camera0_side_TongueTracking']
        ttimes = np.asarray(tongue.timestamps[:], dtype=np.float64)
        tdata = np.asarray(tongue.data[:], dtype=np.float64)
        if ttimes.size > 1 and not np.all(np.diff(ttimes) >= 0):
            # two sessions store every trial's frames twice -> timestamps not monotone
            o = np.argsort(ttimes, kind='stable')
            ttimes, tdata = ttimes[o], tdata[o]
        visible = tdata[:, 2] > TONGUE_LIKELIHOOD_THRESH
        mean_y, vis_counts = bin_visible_mean(ttimes, tdata[:, 1], visible, go)
        tongue_class, pcts = discretise_tongue(mean_y, vis_counts)
        timing['outputs'] = time.time() - t0

        # ---- assemble ----------------------------------------------------------- #
        t0 = time.time()
        neural_trials = [np.ascontiguousarray(rates[:, t, :]) for t in range(n_trials)]
        input_trials = [np.stack([time_from_tone[t], photostim[t]]).astype(np.float32)
                        for t in range(n_trials)]
        output_trials = [
            np.stack([
                np.full(N_BINS, choice[t], dtype=np.int64),
                np.full(N_BINS, outcome_code[t], dtype=np.int64),
                np.full(N_BINS, early_code[t], dtype=np.int64),
                tongue_class[t],
            ])
            for t in range(n_trials)
        ]
        timing['assemble'] = time.time() - t0

        info = {
            'session_id': session_id,
            'subject': subject,
            'subject_id': nwb.subject.subject_id,
            'session_start_time': str(nwb.session_start_time),
            'n_trials_all': int(n_trials_all),
            'n_trials_ephys': int(n_ephys),
            'n_obs_intervals_unmatched': int(len(obs) - matched.sum()),
            'n_trials_kept': int(n_trials),
            'n_trials_empty_dropped': n_empty,
            'n_neurons': int(len(good)),
            'n_units_total': int(len(units)),
            'n_probes': int(len(nwb.electrode_groups)),
            'tongue_pct40': pcts[0],
            'tongue_pct60': pcts[1],
            'frac_tongue_visible': float(np.mean(vis_counts > 0)),
            'frac_photostim_trials': float(np.mean(photostim.max(axis=1) > 0)),
            'go_minus_tone_median': float(np.median(go_minus_tone)),
            'timing': {k: round(v, 3) for k, v in timing.items()},
            'total_time': round(time.time() - t_start, 2),
        }

        if show_processing:
            _plot_processing(plot_dir, session_id, go, trial_idx, rates, time_from_tone,
                             photostim, choice, outcome_code, early_code, tongue_class,
                             mean_y, vis_counts, pcts, ttimes, tdata, visible,
                             ps_start, ps_stop, tone_onset, region_idx,
                             start_all[trial_idx], stop_all[trial_idx])

        return {
            'session_id': session_id,
            'subject': subject,
            'neural': neural_trials,
            'input': input_trials,
            'output': output_trials,
            'brain_region_idx': region_idx,
            'info': info,
        }


# --------------------------------------------------------------------------- #
#                            processing-step plots                             #
# --------------------------------------------------------------------------- #

def _plot_processing(plot_dir, session_id, go, trial_idx, rates, time_from_tone,
                     photostim, choice, outcome_code, early_code, tongue_class,
                     mean_y, vis_counts, pcts, ttimes, tdata, visible,
                     ps_start, ps_stop, tone_onset, region_idx, tstart, tstop):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_trials = rates.shape[1]
    # pick example trials: a hit, a miss, an ignore, a photostim trial, an early lick
    picks = []
    for label, mask in [('hit', outcome_code == 2), ('miss', outcome_code == 1),
                        ('ignore', outcome_code == 0), ('photostim', photostim.max(axis=1) > 0),
                        ('early lick', early_code == 1)]:
        w = np.where(mask)[0]
        if len(w):
            picks.append((label, int(w[len(w) // 2])))
    if not picks:
        picks = [('trial 0', 0)]

    fig, axes = plt.subplots(6, len(picks), figsize=(5.2 * len(picks), 19), squeeze=False)
    tc = BIN_CENTERS_REL

    for c, (label, t) in enumerate(picks):
        g = go[t]

        # 1: raw spike raster is not re-read here; show binned population raster
        ax = axes[0, c]
        show_n = min(80, rates.shape[0])
        sel = np.linspace(0, rates.shape[0] - 1, show_n).astype(int)
        ax.imshow(rates[sel, t, :], aspect='auto', origin='lower', cmap='magma',
                  extent=[tc[0] - 0.025, tc[-1] + 0.025, 0, show_n],
                  vmin=0, vmax=max(1.0, np.percentile(rates[sel, t, :], 99)))
        ax.axvline(0, color='c', lw=1.5)
        ax.set_title(f'{label} (trial #{trial_idx[t] + 1})\nbinned firing rate (Hz), {show_n} neurons')
        ax.set_ylabel('neuron')

        # 2: population mean rate + trial-window coverage
        ax = axes[1, c]
        ax.plot(tc, rates[:, t, :].mean(axis=0), 'k')
        ax.axvline(0, color='c', lw=1.5, label='go cue')
        ax.axvspan(tstart[t] - g, tstop[t] - g, color='green', alpha=0.08,
                   label='trial interval (data available)')
        ax.axvline(tone_onset[t] - g, color='m', ls='--', label='tone onset')
        ax.set_ylabel('mean rate (Hz)')
        ax.legend(fontsize=7)

        # 3: input 1
        ax = axes[2, c]
        ax.plot(tc, time_from_tone[t], 'b.-')
        ax.axhline(0, color='m', ls='--')
        ax.axvline(0, color='c', lw=1.5)
        ax.axvline(tone_onset[t] - g, color='m', ls='--')
        ax.set_ylabel('input 0: time from\ntone onset (s)')

        # 4: input 2 vs. raw photostim event
        ax = axes[3, c]
        ax.step(tc, photostim[t], 'r', where='mid')
        for a, b in zip(ps_start, ps_stop):
            if g - 3 < a < g + 3:
                ax.axvspan(a - g, b - g, color='r', alpha=0.2)
        ax.axvline(0, color='c', lw=1.5)
        ax.set_ylim(-0.1, 1.2)
        ax.set_ylabel('input 1: photostim\n(shading = raw event)')

        # 5: raw tongue trace + binned mean + percentile thresholds
        ax = axes[4, c]
        m = (ttimes >= g - 2.5) & (ttimes < g + 1.5)
        ax.plot(ttimes[m] - g, np.where(visible[m], tdata[m, 1], np.nan), '.', ms=2,
                color='0.5', label='raw y (visible)')
        ax.plot(tc, mean_y[t], 'o-', ms=3, color='tab:orange', label='binned mean y')
        if np.isfinite(pcts[0]):
            ax.axhline(pcts[0], color='b', ls=':', label='40th pct')
            ax.axhline(pcts[1], color='g', ls=':', label='60th pct')
        ax.axvline(0, color='c', lw=1.5)
        ax.set_ylabel('tongue y (px)')
        ax.legend(fontsize=7)

        # 6: outputs
        ax = axes[5, c]
        ax.step(tc, tongue_class[t], where='mid', color='tab:orange', label='tongue class')
        ax.axhline(choice[t], color='tab:blue', ls='-', label=f'choice={OUTPUT_VALUES[0][choice[t]]}')
        ax.axhline(outcome_code[t], color='tab:red', ls='--',
                   label=f'outcome={OUTPUT_VALUES[1][outcome_code[t]]}')
        ax.axhline(early_code[t], color='tab:green', ls=':',
                   label=f'early={OUTPUT_VALUES[2][early_code[t]]}')
        ax.axvline(0, color='c', lw=1.5)
        ax.set_ylim(-0.3, 3.3)
        ax.set_xlabel('time from go cue (s)')
        ax.set_ylabel('output class')
        ax.legend(fontsize=7)

    fig.suptitle(f'Processing steps: {session_id}   '
                 f'({rates.shape[0]} neurons, {n_trials} trials, '
                 f'{len(np.unique(region_idx))} areas)')
    fig.tight_layout()
    out = os.path.join(plot_dir, f'processing_{session_id}.png')
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f'  wrote {out}', flush=True)

    # session-level summary figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].hist(mean_y[vis_counts > 0].ravel(), bins=60, color='tab:orange')
    for p, cstr in zip(pcts, ['b', 'g']):
        if np.isfinite(p):
            axes[0].axvline(p, color=cstr, ls=':')
    axes[0].set_title('binned visible tongue y\n(session percentiles)')
    axes[1].bar(np.arange(4), [np.mean(tongue_class == k) for k in range(4)], color='tab:orange')
    axes[1].set_xticks(np.arange(4)); axes[1].set_xticklabels(OUTPUT_VALUES[3], rotation=30, ha='right')
    axes[1].set_title('tongue class fractions')
    axes[2].bar(np.arange(3), [np.mean(outcome_code == k) for k in range(3)])
    axes[2].set_xticks(np.arange(3)); axes[2].set_xticklabels(OUTPUT_VALUES[1])
    axes[2].set_title('outcome fractions')
    axes[3].plot(tc, rates.mean(axis=(0, 1)), 'k')
    axes[3].axvline(0, color='c'); axes[3].set_title('session mean firing rate (Hz)')
    axes[3].set_xlabel('time from go cue (s)')
    fig.tight_layout()
    out = os.path.join(plot_dir, f'processing_{session_id}_summary.png')
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f'  wrote {out}', flush=True)


# --------------------------------------------------------------------------- #
#                                   driver                                     #
# --------------------------------------------------------------------------- #

def _worker(args):
    path, show, plot_dir = args
    try:
        return process_session(path, show_processing=show, plot_dir=plot_dir)
    except Exception as exc:       # keep the whole run alive, report at the end
        import traceback
        return {'session_id': os.path.basename(path), 'error': repr(exc),
                'traceback': traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outfile', type=str)
    ap.add_argument('--full', action='store_true', help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='save per-step plots for up to 2 sessions')
    ap.add_argument('--jobs', type=int, default=12)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
    print(f'Found {len(files)} NWB files in {DATA_DIR}', flush=True)
    if args.sample:
        # one early session and one late session, so the sample covers different rigs
        files = [files[1], files[-3]]
        print(f'--sample: using {[os.path.basename(f) for f in files]}', flush=True)

    plot_dir = os.path.dirname(os.path.abspath(args.outfile)) or '.'
    show_flags = [args.show_processing and i < 2 for i in range(len(files))]

    t0 = time.time()
    results = []
    if args.jobs > 1 and len(files) > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.jobs) as pool:
            for i, res in enumerate(
                    pool.imap(_worker, [(f, s, plot_dir) for f, s in zip(files, show_flags)])):
                results.append(res)
                _report(i, len(files), res, t0)
    else:
        for i, (f, s) in enumerate(zip(files, show_flags)):
            res = _worker((f, s, plot_dir))
            results.append(res)
            _report(i, len(files), res, t0)

    errors = [r for r in results if 'error' in r]
    for r in errors:
        print(f"ERROR in {r['session_id']}: {r['error']}\n{r['traceback']}", flush=True)
    skipped = [r for r in results if 'skipped' in r]
    for r in skipped:
        print(f"SKIPPED {r['session_id']}: {r['skipped']}", flush=True)

    sessions = [r for r in results if 'neural' in r]
    print(f'\nConverted {len(sessions)} sessions '
          f'({len(skipped)} skipped, {len(errors)} errors) in {time.time() - t0:.1f}s',
          flush=True)

    subjects = sorted({s['subject'] for s in sessions})
    subj_idx = {s: i for i, s in enumerate(subjects)}

    data = {
        'neural': [s['neural'] for s in sessions],
        'input': [s['input'] for s in sessions],
        'output': [s['output'] for s in sessions],
        'subjects': subjects,
        'subject_idx': np.array([subj_idx[s['subject']] for s in sessions], dtype=np.int64),
        'brain_regions': list(BRAIN_REGIONS),
        'brain_region_idx': [s['brain_region_idx'] for s in sessions],
        'input_names': list(INPUT_NAMES),
        'output_names': list(OUTPUT_NAMES),
        'output_values': [list(v) for v in OUTPUT_VALUES],
        'metadata': {
            'dataset': 'DANDI:000363 Mesoscale Activity Map Dataset (Chen, Nguyen, Li, Svoboda 2023)',
            'papers': [
                'Chen et al. 2024, Brain-wide neural activity underlying memory-guided movement, Cell 187:676',
                'Wang, Kurgyis et al. 2025, Brain-wide analysis reveals movement encoding structured '
                'across and within brain areas, Nat Neurosci',
            ],
            'task_description':
                'Head-fixed mice performed an auditory memory-guided directional licking task. A 3 kHz or '
                '12 kHz instruction tone train (0.65 s) during the sample epoch was followed by a 1.2 s '
                'delay epoch and an auditory Go cue, after which the mouse reported the instruction by '
                'licking the left or right lick port (1.5 s answer period). On ~20% of trials ALM was '
                'photoinhibited (473 nm, 5.5 mW, 0.5 s) during the delay, ending before the Go cue. '
                'Decoded outputs are the lick-direction choice (left / right / no lick), the trial outcome '
                '(ignore / miss / hit), whether the mouse licked early (no / yes), and the vertical '
                'position of the tongue tracked by DeepLabCut from 300 Hz side-view video, discretised '
                'per session into <40th percentile / 40-60th percentile / >60th percentile / not visible.',
            'neural_data_description':
                'Firing rate (Hz) of Kilosort2 single units that passed the region-specific '
                'quality-control classifier (units.classification == "good"), binned into '
                'non-overlapping 50 ms bins.',
            'time_bin_size': BIN_SIZE * 1000.0,          # ms
            'temporal_alignment_event': 'Go cue onset (auditory, 6 kHz, 0.1 s), BehavioralEvents/go_start_times',
            'off_start': -T_PRE,
            'off_end': T_POST,
            'n_time_bins': N_BINS,
            'bin_centers_s': BIN_CENTERS_REL.tolist(),
            'input_descriptions': [
                'Seconds elapsed since the onset of the instruction-tone (sample) epoch of that trial; '
                'the last sample onset at or before the Go cue is used, because an early lick replays '
                'the sample+delay epochs.',
                'Binary indicator that ALM photoinhibition was on during the bin (bin centre inside '
                '[photostim_start, photostim_stop)).',
            ],
            'output_descriptions': [
                'Lick direction chosen by the mouse, from trials.outcome and trials.trial_instruction '
                '(hit -> instructed side, miss -> opposite side, ignore -> no lick). Constant within a trial.',
                'Trial outcome from trials.outcome. Constant within a trial.',
                'Whether the mouse licked during the sample/delay epoch (trials.early_lick). Constant within a trial.',
                'Tongue vertical (y) position from the side-view DeepLabCut tracking, averaged over the '
                'frames in each 50 ms bin with likelihood > 0.9; bins with no such frame are class '
                '"not visible". Visible bins are split by the 40th and 60th percentile of all visible '
                'binned y values of that session.',
            ],
            'neuron_curation': 'units.classification == "good" (region-specific logistic-regression QC '
                               'classifier of Chen, Liu et al. 2023).',
            'trial_curation': 'Trials within the range covered by the ephys recording '
                              '(units.obs_intervals) and with auto_water == 0 and free_water == 0. '
                              'Photostimulation, early-lick and no-response (ignore) trials are kept '
                              'because the decoder task requires them as an input / output classes.',
            'session_curation': 'All released sessions with at least one good unit (173 of 174 files).',
            'known_limitations':
                'The NWB export stores spikes and video frames only inside [trial.start_time, '
                'trial.stop_time]. Error (miss) trials are terminated by the time-out ~0.8 s after the '
                'Go cue, so the last ~0.7 s of the requested window contains no spikes (firing rate 0) '
                'and no video (tongue class "not visible") on those trials. 81.5% of trials cover the '
                'full [-2.5, +1.5] s window. Two sessions have partially missing video '
                '(SC066_20210413_112028_s6, SC065_20210505_170309_s6).',
            'ccf_note': 'Brain areas are derived from the per-unit Allen CCFv3 annotation '
                        '(units.anno_name); frontal motor cortex counts as ALM only when the unit is '
                        f'> {ALM_MIN_AP_MM} mm anterior to bregma and < {ALM_MAX_ML_MM} mm from the midline.',
            'session_info': [s['info'] for s in sessions],
        },
    }

    # ---- summary ---------------------------------------------------------- #
    n_neurons = [len(s['brain_region_idx']) for s in sessions]
    n_trials = [len(s['neural']) for s in sessions]
    print(f'  sessions            : {len(sessions)}')
    print(f'  subjects            : {len(subjects)}')
    print(f'  total neurons       : {sum(n_neurons)}')
    print(f'  neurons/session     : mean {np.mean(n_neurons):.1f}, median {np.median(n_neurons):.0f}, '
          f'range {min(n_neurons)}-{max(n_neurons)}')
    print(f'  total trials        : {sum(n_trials)}')
    print(f'  trials/session      : mean {np.mean(n_trials):.1f}, median {np.median(n_trials):.0f}, '
          f'range {min(n_trials)}-{max(n_trials)}')
    counts = np.bincount(np.concatenate([s['brain_region_idx'] for s in sessions]),
                         minlength=len(BRAIN_REGIONS))
    print('  neurons per region  : ' + ', '.join(f'{r}={c}' for r, c in zip(BRAIN_REGIONS, counts)))

    t0 = time.time()
    with open(args.outfile, 'wb') as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\nWrote {args.outfile} '
          f'({os.path.getsize(args.outfile) / 1e9:.2f} GB) in {time.time() - t0:.1f}s', flush=True)


def _report(i, n, res, t0):
    el = time.time() - t0
    tag = res.get('session_id', '?')
    if 'error' in res:
        print(f'[{i + 1}/{n}] {tag}: ERROR', flush=True)
    elif 'skipped' in res:
        print(f'[{i + 1}/{n}] {tag}: skipped ({res["skipped"]})', flush=True)
    else:
        info = res['info']
        print(f'[{i + 1}/{n}] {tag}: {info["n_neurons"]} neurons, '
              f'{info["n_trials_kept"]}/{info["n_trials_all"]} trials, '
              f'{info["total_time"]}s  {info["timing"]}  '
              f'[elapsed {el:.0f}s, eta {el / (i + 1) * (n - i - 1):.0f}s]', flush=True)


if __name__ == '__main__':
    main()
