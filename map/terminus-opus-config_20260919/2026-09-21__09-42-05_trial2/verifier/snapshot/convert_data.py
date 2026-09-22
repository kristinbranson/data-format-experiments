#!/usr/bin/env python3
"""
Convert the MAP dataset (DANDI 000363, Chen et al. 2024 "Brain-wide neural activity
underlying memory-guided movement") from NWB into the decoder pickle format.

Processing follows the reference pipeline of Wang/Kurgyis et al. 2025
(`/app/code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py`,
`/app/code/Sherlock/align_markers.py`) wherever the decoder specification allows:

  * neuron curation  : classifier QC 'good' units with a CCF annotation
                       (reference `qc_mode='classifier'`)
  * alignment        : go cue onset (reference aligns spikes and video to the go cue)
  * firing rates     : spike count / bin width, in Hz (reference `sliding_histogram(rate=True)`)
  * regions          : coarse CCF group x hemisphere, midline at CCF ML = 5700 um
                       (reference `helper_get_neuron_id_area`)
  * trial curation   : auto-water / free-water trials removed (reference
                       `get_regular_trial_mask`).  Early-lick / no-response /
                       photostim trials are KEPT because they are decoder targets
                       or inputs.

Deviations required by the decoder specification are documented in CONVERSION_NOTES.md.

Usage:  python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np

# ----------------------------------------------------------------------------- config
DATA_DIR = '/app/data'
OFF_START = -2.5          # s, signed time from go cue to start of the extracted window
OFF_END = 1.5             # s, signed time from go cue to end of the extracted window
BIN_SIZE = 0.05           # s, bin width == stride (non overlapping), decoder spec
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))     # 80
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(NBINS + 1)  # (81,) relative to go cue
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2.0            # (80,)
ML_MIDLINE = 5700.0       # um, reference `helper_get_neuron_id_area`
TONGUE_LIKELIHOOD_THRESHOLD = 0.9   # DLC likelihood; distribution is strongly bimodal
DEFAULT_SAMPLE_TO_GO = 1.85         # s, 0.65 s sample + 1.2 s delay (fallback only)

INPUT_NAMES = ['time_from_tone_onset', 'photostim_on']
OUTPUT_NAMES = ['choice', 'outcome', 'early_lick', 'tongue_y_position']
OUTPUT_VALUES = [
    ['no lick', 'left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['low (<40th pctile)', 'middle (40-60th pctile)', 'high (>60th pctile)', 'not visible'],
]

# ------------------------------------------------------------------- CCF region mapping
# Coarse groups are the ones used by the reference preprocessing code
# (ALM, Medulla, Midbrain, Striatum, Thalamus, Pons, Cerebellum, Hypothalamus,
#  Hippocampus, Orbital, OtherCortex, Olfactory, CorticalSubplate, Pallidum).
CEREBELLUM = ['Lingula', 'Lobule', 'Lobules', 'Declive', 'Folium', 'Tuber', 'Pyramus', 'Uvula',
              'Nodulus', 'Simple lobule', 'Crus 1', 'Crus 2', 'Paramedian lobule',
              'Copula pyramidis', 'Paraflocculus', 'Flocculus', 'Fastigial', 'Interposed',
              'Dentate nucleus', 'Cerebellum', 'Infracerebellar', 'Ansiform', 'Central lobule',
              'Culmen', 'Vermal', 'Cerebellar']
MEDULLA = ['Medulla', 'Medullary reticular', 'Gigantocellular', 'Magnocellular reticular',
           'Parvicellular reticular', 'Intermediate reticular', 'Paragigantocellular',
           'Lateral reticular', 'Inferior olivary', 'Nucleus of the solitary',
           'Spinal nucleus of the trigeminal', 'Nucleus ambiguus', 'Hypoglossal',
           'Facial motor', 'Dorsal motor nucleus of the vagus', 'Nucleus raphe magnus',
           'Nucleus raphe obscurus', 'Nucleus raphe pallidus', 'External cuneate', 'Cuneate',
           'Gracile', 'Vestibular', 'Nucleus x', 'Nucleus y', 'Nucleus of Roller',
           'Parasolitary', 'Nucleus prepositus', 'Intercalated nucleus', 'Perihypoglossal',
           'Linear nucleus', 'Dorsal cochlear', 'Ventral cochlear',
           'Nucleus of the trapezoid', 'Superior olivary', 'Paratrigeminal', 'Area postrema',
           'Interfascicular nucleus of the hypoglossal']
PONS = ['Pons', 'Pontine reticular', 'Pontine gray', 'Tegmental reticular', 'Koelliker-Fuse',
        'Parabrachial', 'Nucleus of the lateral lemniscus', 'Locus ceruleus',
        'Laterodorsal tegmental', 'Nucleus incertus', 'Barrington', 'Dorsal nucleus raphe',
        'Superior central raphe', 'Sublaterodorsal', 'Pontine central gray',
        'Nucleus raphe pontis', 'Supratrigeminal',
        'Principal sensory nucleus of the trigeminal', 'Motor nucleus of trigeminal']
MIDBRAIN = ['Midbrain', 'Superior colliculus', 'Inferior colliculus', 'Periaqueductal gray',
            'Red nucleus', 'Substantia nigra', 'Ventral tegmental', 'Pedunculopontine',
            'Anterior pretectal', 'Posterior pretectal', 'Nucleus of the optic tract',
            'terminal nucleus of the accessory optic tract',
            'Nucleus of the brachium of the inferior colliculus', 'Nucleus sagulum',
            'Cuneiform', 'Interpeduncular', 'Edinger-Westphal', 'Oculomotor', 'Trochlear',
            'Midbrain reticular', 'Olivary pretectal', 'Nucleus of the posterior commissure',
            'Parabigeminal', 'Medial terminal nucleus', 'Subcommissural',
            'Retroparafascicular', 'Intercollicular', 'Midbrain trigeminal']
THALAMUS = ['Thalamus', 'nucleus of the thalamus', 'nucleus of thalamus', 'Anteroventral nucleus',
            'Anterodorsal nucleus', 'Anteromedial nucleus', 'Lateral dorsal nucleus',
            'Mediodorsal nucleus', 'Paraventricular nucleus of the thalamus', 'Parataenia',
            'Reuniens', 'Rhomboid', 'Perireunensis', 'Xiphoid', 'Intermediodorsal',
            'Interanterodorsal', 'Central lateral', 'Central medial', 'Paracentral',
            'Parafascicular', 'Posterior complex', 'Posterior limiting', 'Suprageniculate',
            'Medial geniculate', 'lateral geniculate complex', 'Ventral anterior-lateral complex',
            'Ventral medial nucleus', 'Ventral posterolateral nucleus',
            'Ventral posteromedial nucleus', 'Submedial nucleus',
            'Reticular nucleus of the thalamus', 'Lateral habenula', 'Medial habenula',
            'Subparafascicular', 'Fields of Forel', 'Ethmoid',
            'Lateral posterior nucleus', 'Nucleus of reuniens', 'Peripeduncular nucleus']
HYPOTHALAMUS = ['Hypothalam', 'Lateral hypothalamic area', 'Tuberomammillary', 'Mammillary',
                'Zona incerta',  # Allen CCF: ZI belongs to the hypothalamus
                'Preoptic', 'Paraventricular hypothalamic', 'Arcuate',
                'Dorsomedial nucleus of the hypothalamus', 'Ventromedial hypothalamic',
                'Subthalamic', 'Parasubthalamic', 'Lateral mammillary', 'Medial mammillary',
                'Supramammillary', 'Periventricular', 'Anterior hypothalamic',
                'Posterior hypothalamic']
HIPPOCAMPUS = ['Field CA1', 'Field CA2', 'Field CA3', 'Dentate gyrus', 'Subiculum',
               'Postsubiculum', 'Presubiculum', 'Parasubiculum', 'Hippocamp',
               'Entorhinal area', 'Fasciola cinerea', 'Induseum griseum', 'Prosubiculum']
STRIATUM = ['Striatum', 'Caudoputamen', 'Nucleus accumbens', 'Fundus of striatum',
            'Olfactory tubercle', 'Lateral septal', 'Septofimbrial', 'Septohippocampal',
            'Triangular nucleus of septum', 'Central amygdalar', 'Medial amygdalar',
            'Intercalated amygdalar', 'Anterior amygdalar', 'Bed nuclei of the stria terminalis',
            'Posterior amygdalar nucleus']
PALLIDUM = ['Pallidum', 'Globus pallidus', 'Substantia innominata', 'Magnocellular nucleus',
            'Medial septal', 'Diagonal band', 'Bed nucleus of the anterior commissure']
OLFACTORY = ['Olfactory areas', 'Olfactory bulb', 'Anterior olfactory', 'Piriform',
             'Taenia tecta', 'Dorsal peduncular', 'Nucleus of the lateral olfactory tract',
             'Cortical amygdalar', 'Piriform-amygdalar', 'Postpiriform']
CORTICAL_SUBPLATE = ['Claustrum', 'Endopiriform', 'Lateral amygdalar', 'Basolateral amygdalar',
                     'Basomedial amygdalar', 'Cortical subplate']
ORBITAL = ['Orbital area']
ALM = ['Primary motor area', 'Secondary motor area']

GROUPS = [('Cerebellum', CEREBELLUM), ('Medulla', MEDULLA), ('Pons', PONS), ('Midbrain', MIDBRAIN),
          ('Hypothalamus', HYPOTHALAMUS), ('Thalamus', THALAMUS), ('Hippocampus', HIPPOCAMPUS),
          ('Striatum', STRIATUM), ('Pallidum', PALLIDUM), ('Olfactory', OLFACTORY),
          ('CorticalSubplate', CORTICAL_SUBPLATE), ('Orbital', ORBITAL), ('ALM', ALM)]
GROUP_NAMES = [g for g, _ in GROUPS] + ['OtherCortex']


def map_annotation(name):
    """Map a CCF leaf annotation to one of the coarse groups used by the reference code."""
    n = name.strip()
    if n == '':
        return None
    low = n.lower()
    for gname, keys in GROUPS:
        for k in keys:
            if k.lower() in low:
                return gname
    return 'OtherCortex'


# --------------------------------------------------------------------------- utilities
def _decode(arr):
    """h5py object (bytes) array -> numpy array of str."""
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])


def _spike_slices(spike_times_index):
    """Start/stop indices into the ragged units/spike_times dataset."""
    stop = np.asarray(spike_times_index)
    start = np.concatenate([[0], stop[:-1]])
    return start, stop


def bin_spikes(spike_times, starts, stops, unit_ids, edges_abs):
    """Firing rates in Hz.

    Reference `sliding_histogram(..., rate=True)` computes
    (number of spikes in the bin) / bin_width; here the bins are the 50 ms
    non-overlapping bins required by the decoder specification.

    Args:
        spike_times: (n_spikes,) absolute spike times of the whole session
        starts/stops: index ranges of each unit inside spike_times
        unit_ids: indices of the units to extract
        edges_abs: (n_trials, NBINS+1) absolute bin edges
    Returns:
        (n_units, n_trials, NBINS) float32 firing rates in Hz
    """
    ntrials = edges_abs.shape[0]
    flat = edges_abs.ravel()
    out = np.empty((len(unit_ids), ntrials, NBINS), dtype=np.float32)
    for i, k in enumerate(unit_ids):
        sp = spike_times[starts[k]:stops[k]]
        idx = np.searchsorted(sp, flat).reshape(ntrials, NBINS + 1)
        out[i] = np.diff(idx, axis=1) / BIN_SIZE
    return out


def tone_onset_times(trial_start, go, sample_start):
    """Onset of the (last) instruction tone of each trial.

    The sample epoch is replayed when the animal licks early, so there can be
    several `sample_start_times` events inside one trial.  The tone that the
    animal finally responded to is the last sample start before the go cue.
    """
    ntrials = len(go)
    tone = np.full(ntrials, np.nan)
    if len(sample_start):
        idx = np.searchsorted(trial_start, sample_start, side='right') - 1
        for i, s in zip(idx, sample_start):
            if 0 <= i < ntrials and s < go[i]:
                if np.isnan(tone[i]) or s > tone[i]:
                    tone[i] = s
    missing = np.isnan(tone)
    if missing.any():                                   # fallback, should be rare
        tone[missing] = go[missing] - DEFAULT_SAMPLE_TO_GO
    return tone


def photostim_intervals(f, trial_start, go):
    """(n_stim, 2) array of absolute [on, off] photostimulation times."""
    be = f['acquisition/BehavioralEvents']
    if 'photostim_start_times' in be and len(be['photostim_start_times/timestamps']) > 0:
        on = be['photostim_start_times/timestamps'][:]
        off = be['photostim_stop_times/timestamps'][:]
        n = min(len(on), len(off))
        return np.stack([on[:n], off[:n]], axis=1)
    # fallback: reconstruct from the trials table (onset is relative to trial start)
    t = f['intervals/trials']
    onset = _decode(t['photostim_onset'][:])
    dur = _decode(t['photostim_duration'][:])
    iv = []
    for i in range(len(trial_start)):
        if onset[i] != 'N/A' and dur[i] != 'N/A':
            a = trial_start[i] + float(onset[i])
            iv.append([a, a + float(dur[i])])
    return np.array(iv).reshape(-1, 2)


def observed_trial_mask(units, unit_ids, trial_start, trial_stop):
    """Trials that lie inside the obs_intervals of every selected unit.

    `units/obs_intervals` holds, for each unit, the time windows during which the
    unit was recorded; in this dataset they are exactly the [trial start, trial stop]
    windows of the trials that the probe recorded.  Trials outside these windows have
    no spike data (they would be converted to all-zero firing rates).
    """
    oi_index = units['obs_intervals_index'][:]
    oi = units['obs_intervals'][:]
    starts = np.concatenate([[0], oi_index[:-1]])
    ntrials = len(trial_start)
    counts = np.zeros(ntrials, dtype=np.int64)
    for k in unit_ids:
        iv = oi[starts[k]:oi_index[k]]
        if len(iv) == 0:
            continue
        idx = np.searchsorted(trial_start, iv[:, 0] + 1e-6) - 1
        idx = idx[(idx >= 0) & (idx < ntrials)]
        counts[np.unique(idx)] += 1
    return counts == len(unit_ids)


def process_session(path, make_plots=False, plot_dir='/app'):
    """Convert one NWB session. Returns a dict, or None if the session is unusable."""
    t_start = time.time()
    timing = {}
    with h5py.File(path, 'r') as f:
        sess_id = f['identifier'][()].decode()
        subject = f['general/subject/subject_id'][()].decode()
        session_start = f['session_start_time'][()].decode()

        # ---------------- neuron curation (reference: classifier QC 'good' units) -----
        u = f['units']
        classification = u['classification'][:]
        anno = _decode(u['anno_name'][:])
        good = (classification == b'good')
        # reference additionally requires a CCF annotation (histology present)
        good &= np.array([a.strip() != '' for a in anno])
        unit_ids = np.where(good)[0]
        if len(unit_ids) == 0:
            print('%s: no good units, skipping' % sess_id, flush=True)
            return None

        # hemisphere from the CCF ML coordinate of the unit's electrode
        el = f['general/extracellular_ephys/electrodes']
        ex = el['x'][:]
        e_of_unit = u['electrodes'][:][np.concatenate([[0], u['electrodes_index'][:][:-1]])]
        ml = ex[e_of_unit]
        region_names = []
        for k in unit_ids:
            grp = map_annotation(anno[k])
            side = 'left' if (not np.isnan(ml[k]) and ml[k] >= ML_MIDLINE) else 'right'
            region_names.append('%s %s' % (side, grp))

        # ---------------- trials -----------------------------------------------------
        t = f['intervals/trials']
        trial_start = t['start_time'][:]
        trial_stop = t['stop_time'][:]
        outcome = _decode(t['outcome'][:])
        early = _decode(t['early_lick'][:])
        instruction = _decode(t['trial_instruction'][:])
        auto_water = t['auto_water'][:] > 0
        free_water = t['free_water'][:] > 0
        ntrials_file = len(trial_start)

        go_all = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
        if len(go_all) != ntrials_file:
            print('%s: %d go cues for %d trials, skipping' % (sess_id, len(go_all), ntrials_file),
                  flush=True)
            return None

        # reference `get_regular_trial_mask` also removes auto/free water trials; early
        # lick / no-response / photostim trials are decoder targets or inputs and kept
        keep = ~(auto_water | free_water)
        keep &= np.isfinite(go_all)

        # In 8 of the 173 sessions the ephys recording covers only part of the
        # behavioural session: the units' obs_intervals then span a contiguous subset
        # of trials and the remaining trials contain no spikes at all.  Keep only
        # trials that are inside the observation window of every good unit.
        keep &= observed_trial_mask(u, unit_ids, trial_start, trial_stop)
        trial_idx = np.where(keep)[0]
        if len(trial_idx) < 2:
            print('%s: <2 usable trials, skipping' % sess_id, flush=True)
            return None

        go = go_all[trial_idx]
        edges_abs = go[:, None] + BIN_EDGES[None, :]
        centers_abs = go[:, None] + BIN_CENTERS[None, :]

        # ---------------- neural ------------------------------------------------------
        t0 = time.time()
        spikes = u['spike_times'][:]
        s0, s1 = _spike_slices(u['spike_times_index'][:])
        fr = bin_spikes(spikes, s0, s1, unit_ids, edges_abs)     # (n_units, n_trials, NBINS)
        del spikes

        # A handful of trials sit at the very end of a recording: they are listed in
        # obs_intervals but the probe was already switched off, so not a single spike
        # was recorded in the extracted window.  Such trials carry no neural data.
        has_spikes = fr.sum(axis=(0, 2)) > 0
        if not has_spikes.all():
            fr = fr[:, has_spikes, :]
            trial_idx = trial_idx[has_spikes]
            go = go[has_spikes]
            edges_abs = edges_abs[has_spikes]
            centers_abs = centers_abs[has_spikes]
        if len(trial_idx) < 2:
            print('%s: <2 usable trials after spike check, skipping' % sess_id, flush=True)
            return None
        timing['neural'] = time.time() - t0

        # ---------------- inputs ------------------------------------------------------
        t0 = time.time()
        sample_start = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:] \
            if 'sample_start_times' in f['acquisition/BehavioralEvents'] else np.array([])
        tone = tone_onset_times(trial_start, go_all, sample_start)[trial_idx]
        time_from_tone = (centers_abs - tone[:, None]).astype(np.float32)   # (n_trials, NBINS)

        stim_iv = photostim_intervals(f, trial_start, go_all)
        photostim = np.zeros((len(trial_idx), NBINS), dtype=np.float32)
        if len(stim_iv):
            for a, b in stim_iv:
                # a bin is 'on' if the stimulation overlaps the bin
                photostim[(edges_abs[:, 1:] > a) & (edges_abs[:, :-1] < b)] = 1.0
        timing['input'] = time.time() - t0

        # ---------------- outputs -----------------------------------------------------
        t0 = time.time()
        oc = outcome[trial_idx]
        instr = instruction[trial_idx]
        el_tr = early[trial_idx]

        # choice: 0 no lick, 1 left, 2 right.  hit -> instructed side, miss -> other side,
        # ignore -> no lick (identical definition to reference `behavior_report` x trial type)
        choice = np.zeros(len(trial_idx), dtype=np.int64)
        licked_left = ((oc == 'hit') & (instr == 'left')) | ((oc == 'miss') & (instr == 'right'))
        licked_right = ((oc == 'hit') & (instr == 'right')) | ((oc == 'miss') & (instr == 'left'))
        choice[licked_left] = 1
        choice[licked_right] = 2

        outcome_code = np.select([oc == 'ignore', oc == 'miss', oc == 'hit'], [0, 1, 2],
                                 default=0).astype(np.int64)
        early_code = (el_tr == 'early').astype(np.int64)

        tongue_y, tongue_visible = tongue_y_per_bin(f, edges_abs)
        tongue_class = discretize_tongue(tongue_y, tongue_visible)
        timing['output'] = time.time() - t0

    # ---------------- assemble ---------------------------------------------------------
    neural = [np.ascontiguousarray(fr[:, i, :]) for i in range(fr.shape[1])]
    inputs = [np.stack([time_from_tone[i], photostim[i]]).astype(np.float32)
              for i in range(len(trial_idx))]
    outputs = []
    for i in range(len(trial_idx)):
        o = np.empty((4, NBINS), dtype=np.int64)
        o[0] = choice[i]
        o[1] = outcome_code[i]
        o[2] = early_code[i]
        o[3] = tongue_class[i]
        outputs.append(o)

    region_names = np.array(region_names)
    result = {
        'session_id': sess_id,
        'subject': subject,
        'session_start': session_start,
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'region_names': region_names,
        'n_units_total': int(len(classification)),
        'n_trials_file': int(ntrials_file),
        'trial_idx': trial_idx,
        'timing': timing,
        'duration': time.time() - t_start,
    }
    if make_plots:
        plot_processing(result, path, go, tone, edges_abs, tongue_y, tongue_visible, plot_dir)
    print('%s: %d units, %d/%d trials, %.1fs' %
          (sess_id, len(unit_ids), len(trial_idx), ntrials_file, result['duration']), flush=True)
    return result


# ------------------------------------------------------------------------- tongue
def tongue_y_per_bin(f, edges_abs):
    """Mean tongue y position per 50 ms bin, and whether the tongue was visible.

    The side-camera DeepLabCut markers are sampled every 3.4 ms.  As in the
    reference alignment script the marker stream is aligned to the go cue; here
    all frames falling inside a bin are averaged (the reference used the last
    frame of each 3.4 ms bin, which for a 3.4 ms bin is the same operation).
    Frames with DLC likelihood <= threshold are 'tongue not visible'.
    """
    bt = f['acquisition/BehavioralTimeSeries']
    key = None
    for k in ('Camera0_side_TongueTracking', 'Camera3_side_TongueTracking'):
        if k in bt:
            key = k
            break
    ntrials = edges_abs.shape[0]
    if key is None:
        return (np.full((ntrials, NBINS), np.nan), np.zeros((ntrials, NBINS), dtype=bool))

    data = bt[key]['data'][:]
    ts = bt[key]['timestamps'][:]
    y = data[:, 1]
    lik = data[:, 2]
    visible = (lik > TONGUE_LIKELIHOOD_THRESHOLD) & np.isfinite(y)

    # cumulative sums let us average the frames inside every bin without a loop
    yv = np.where(visible, y, 0.0)
    cs_y = np.concatenate([[0.0], np.cumsum(yv)])
    cs_n = np.concatenate([[0], np.cumsum(visible.astype(np.int64))])
    idx = np.searchsorted(ts, edges_abs)                  # (ntrials, NBINS+1)
    lo, hi = idx[:, :-1], idx[:, 1:]
    nvis = cs_n[hi] - cs_n[lo]
    sumy = cs_y[hi] - cs_y[lo]
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_y = np.where(nvis > 0, sumy / np.maximum(nvis, 1), np.nan)
    return mean_y, nvis > 0


def discretize_tongue(mean_y, visible):
    """0: < 40th pctile, 1: 40-60th pctile, 2: > 60th pctile, 3: not visible.

    Percentiles are computed per session over all bins in which the tongue is visible.
    """
    cls = np.full(mean_y.shape, 3, dtype=np.int64)
    vals = mean_y[visible]
    if vals.size == 0:
        return cls
    p40, p60 = np.percentile(vals, [40, 60])
    cls[visible & (mean_y < p40)] = 0
    cls[visible & (mean_y >= p40) & (mean_y <= p60)] = 1
    cls[visible & (mean_y > p60)] = 2
    return cls


# ------------------------------------------------------------------------- plotting
def plot_processing(res, path, go, tone, edges_abs, tongue_y, tongue_visible, plot_dir):
    """Visualise every processing step for one session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sess = res['session_id']
    neural = res['neural']
    inputs = res['input']
    outputs = res['output']
    ntrials = len(neural)
    tt = BIN_CENTERS

    fig, ax = plt.subplots(4, 2, figsize=(15, 14))

    # 1. raw spike raster vs binned rates for one trial / few neurons
    with h5py.File(path, 'r') as f:
        u = f['units']
        s0, s1 = _spike_slices(u['spike_times_index'][:])
        classification = u['classification'][:]
        anno = _decode(u['anno_name'][:])
        good = np.where((classification == b'good') & np.array([a.strip() != '' for a in anno]))[0]
        tr = min(5, ntrials - 1)
        show = good[:30]
        spikes_all = u['spike_times']
        for j, k in enumerate(show):
            sp = spikes_all[s0[k]:s1[k]]
            sp = sp[(sp >= edges_abs[tr, 0]) & (sp < edges_abs[tr, -1])] - go[tr]
            ax[0, 0].plot(sp, np.full(len(sp), j), '|', color='k', ms=4)
    ax[0, 0].set_title('%s trial %d: raw spikes (aligned to go cue)' % (sess, tr))
    ax[0, 0].set_xlabel('time from go cue (s)')
    ax[0, 0].set_ylabel('neuron')
    ax[0, 0].set_xlim(OFF_START, OFF_END)
    ax[0, 0].axvline(0, color='r')

    im = ax[0, 1].imshow(neural[tr][:30], aspect='auto', origin='lower',
                         extent=[OFF_START, OFF_END, -0.5, min(30, len(show)) - 0.5])
    ax[0, 1].set_title('binned firing rate (Hz), same trial/neurons')
    ax[0, 1].set_xlabel('time from go cue (s)')
    plt.colorbar(im, ax=ax[0, 1])

    # 2. population PSTH sorted by choice
    ch = np.array([o[0, 0] for o in outputs])
    for c, lab, col in [(1, 'lick left', 'b'), (2, 'lick right', 'r'), (0, 'no lick', 'gray')]:
        m = np.where(ch == c)[0]
        if len(m) == 0:
            continue
        psth = np.mean([neural[i].mean(axis=0) for i in m], axis=0)
        ax[1, 0].plot(tt, psth, col, label='%s (n=%d)' % (lab, len(m)))
    ax[1, 0].axvline(0, color='k', ls='--')
    ax[1, 0].set_title('population mean rate by choice')
    ax[1, 0].set_xlabel('time from go cue (s)')
    ax[1, 0].set_ylabel('Hz')
    ax[1, 0].legend()

    # 3. inputs
    for i in range(min(8, ntrials)):
        ax[1, 1].plot(tt, inputs[i][0], alpha=0.7)
    ax[1, 1].axvline(0, color='k', ls='--')
    ax[1, 1].set_title('input 0: time from tone onset (s), 8 trials')
    ax[1, 1].set_xlabel('time from go cue (s)')

    stim = np.array([inp[1] for inp in inputs])
    im = ax[2, 0].imshow(stim, aspect='auto', origin='lower', interpolation='nearest',
                         extent=[OFF_START, OFF_END, 0, ntrials])
    ax[2, 0].set_title('input 1: photostim on (%d stim trials)' % int((stim.max(axis=1) > 0).sum()))
    ax[2, 0].set_xlabel('time from go cue (s)')
    ax[2, 0].set_ylabel('trial')
    plt.colorbar(im, ax=ax[2, 0])

    # 4. tongue: continuous y and discretisation
    vals = tongue_y[tongue_visible]
    if vals.size:
        p40, p60 = np.percentile(vals, [40, 60])
        ax[2, 1].hist(vals, bins=60)
        ax[2, 1].axvline(p40, color='r', label='40th pctile')
        ax[2, 1].axvline(p60, color='g', label='60th pctile')
        ax[2, 1].legend()
    ax[2, 1].set_title('tongue y over visible bins (session percentiles)')

    tcls = np.array([o[3] for o in outputs])
    im = ax[3, 0].imshow(tcls, aspect='auto', origin='lower', interpolation='nearest',
                         extent=[OFF_START, OFF_END, 0, ntrials], vmin=0, vmax=3)
    ax[3, 0].set_title('output 3: tongue y class (3 = not visible)')
    ax[3, 0].set_xlabel('time from go cue (s)')
    ax[3, 0].set_ylabel('trial')
    plt.colorbar(im, ax=ax[3, 0])

    # 5. per-trial outputs
    oc = np.array([o[1, 0] for o in outputs])
    ea = np.array([o[2, 0] for o in outputs])
    ax[3, 1].plot(ch, '.', label='choice (0 none,1 L,2 R)')
    ax[3, 1].plot(oc + 0.1, '.', label='outcome (0 ig,1 miss,2 hit)')
    ax[3, 1].plot(ea + 0.2, '.', label='early lick')
    ax[3, 1].set_xlabel('trial')
    ax[3, 1].legend(fontsize=7)
    ax[3, 1].set_title('per-trial outputs')

    fig.suptitle('Processing summary %s' % sess)
    fig.tight_layout()
    out = os.path.join(plot_dir, 'processing_%s.png' % sess)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print('  wrote', out, flush=True)


# ------------------------------------------------------------------------- main
def _worker(args):
    path, make_plots = args
    try:
        return process_session(path, make_plots=make_plots)
    except Exception as exc:                                   # keep going on bad files
        import traceback
        print('ERROR processing %s: %s' % (path, exc), flush=True)
        traceback.print_exc()
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile', type=str)
    ap.add_argument('--full', action='store_true', help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='save processing plots for up to 2 sessions')
    ap.add_argument('--nproc', type=int, default=16)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
    if args.sample:
        files = files[:2]
    print('Processing %d sessions with %d workers' % (len(files), args.nproc), flush=True)

    t0 = time.time()
    plot_flags = [args.show_processing and i < 2 for i in range(len(files))]
    if args.nproc > 1 and len(files) > 1:
        with Pool(min(args.nproc, len(files))) as pool:
            results = pool.map(_worker, list(zip(files, plot_flags)), chunksize=1)
    else:
        results = [_worker(a) for a in zip(files, plot_flags)]
    results = [r for r in results if r is not None]
    print('Processed %d sessions in %.1f s (%.2f s/session)'
          % (len(results), time.time() - t0, (time.time() - t0) / max(len(results), 1)), flush=True)

    # ---- assemble the final structure ----
    subjects = sorted({r['subject'] for r in results})
    subject_index = {s: i for i, s in enumerate(subjects)}
    all_regions = sorted({str(rn) for r in results for rn in r['region_names']})
    region_index = {r: i for i, r in enumerate(all_regions)}

    data = {
        'neural': [r['neural'] for r in results],
        'input': [r['input'] for r in results],
        'output': [r['output'] for r in results],
        'subjects': subjects,
        'subject_idx': np.array([subject_index[r['subject']] for r in results], dtype=np.int64),
        'brain_regions': all_regions,
        'brain_region_idx': [np.array([region_index[str(x)] for x in r['region_names']], dtype=np.int64)
                             for r in results],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description':
                'Auditory delayed-response (memory-guided movement) task (Chen et al. 2024, '
                'DANDI 000363). A 3 kHz or 12 kHz instruction tone (3 x 150 ms) during the sample '
                'epoch is followed by a 1.2 s delay; an auditory go cue then allows the mouse to '
                'report the instruction by licking the left or right lick port. Decoded variables: '
                'lick direction choice (no lick / left / right), trial outcome (ignore / miss / hit), '
                'early lick (no / yes) and the discretized side-view tongue y position '
                '(<40th pctile / 40-60th pctile / >60th pctile / not visible). Decoder inputs are the '
                'time from instruction-tone onset and whether ALM photostimulation is on.',
            'time_bin_size': BIN_SIZE * 1000.0,
            'temporal_alignment_event': 'Go cue onset (auditory go cue, end of the delay epoch)',
            'off_start': OFF_START,
            'off_end': OFF_END,
            'n_timepoints': NBINS,
            'bin_centers_sec': BIN_CENTERS.tolist(),
            'neural_units': 'firing rate (spikes/s), spike count per 50 ms bin / 0.05 s',
            'neuron_curation': "units/classification == 'good' (region-specific QC classifiers, "
                               'Chen, Liu et al. 2023 white paper) and a non-empty CCF annotation',
            'trial_curation': 'auto-water and free-water trials removed (reference '
                              'get_regular_trial_mask); early-lick, no-response and photostimulation '
                              'trials kept because they are decoder targets/inputs',
            'session_info': [{'session_id': r['session_id'], 'subject': r['subject'],
                              'session_start_time': r['session_start'],
                              'n_neurons': len(r['region_names']),
                              'n_units_in_file': r['n_units_total'],
                              'n_trials': len(r['neural']),
                              'n_trials_in_file': r['n_trials_file']} for r in results],
            'source': 'DANDI 000363 (Chen et al., Cell 2024); processing follows '
                      'druckmann-lab/MapVideoAnalysis (Wang, Kurgyis et al., Nat Neurosci 2025)',
        },
    }

    n_neurons = sum(len(r['region_names']) for r in results)
    n_trials = sum(len(r['neural']) for r in results)
    print('Sessions: %d | subjects: %d | neurons: %d | trials: %d | regions: %d'
          % (len(results), len(subjects), n_neurons, n_trials, len(all_regions)), flush=True)

    t0 = time.time()
    with open(args.outfile, 'wb') as fh:
        pickle.dump(data, fh, protocol=4)
    print('Wrote %s (%.2f GB) in %.1f s'
          % (args.outfile, os.path.getsize(args.outfile) / 1e9, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
