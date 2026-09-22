#!/usr/bin/env python3
"""Convert the MAP (DANDI 000363) NWB dataset to the decoder pickle format.

Usage:  python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Data source : Chen et al. 2024 (Cell) "Brain-wide neural activity underlying
              memory-guided movement"; analysed by Wang*, Kurgyis* et al. 2025 (Nat Neurosci).
Reference   : /app/code (MapVideoAnalysis).  Key reference functions reproduced here:
              preprocessing_DJ_2022Aug.process_one_sess / process_one_area / sliding_histogram
              (go-cue alignment, QC-classifier good units, hemisphere split at CCF ML 5700 um),
              Sherlock/align_markers.py (DeepLabCut side-camera marker alignment to the go cue),
              population_decoding_utils.get_regular_trial_mask (trial curation).
"""
import argparse
import json
import os
import pickle
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pynwb import NWBHDF5IO

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------- constants
DATA_DIR = '/app/data'
BIN_SIZE = 0.05           # s, 50 ms bins (decoder task specification)
OFF_START = -2.5          # s relative to go cue
OFF_END = 1.5             # s relative to go cue
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 80
BIN_EDGES_REL = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
ML_MIDLINE = 5700.0       # um, CCF ML midline used by the reference code
TONGUE_LIK_THRESH = 0.5   # DeepLabCut likelihood above which the tongue is 'visible'
VELOCITY_SIGMA = 5.0      # 5-sigma velocity outlier rejection (method paper)

INPUT_NAMES = ['time_from_tone_onset', 'photostim_on']
OUTPUT_NAMES = ['choice', 'outcome', 'early_lick', 'tongue_y']
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['<40th pct', '40-60th pct', '>60th pct', 'not visible'],
]

# ------------------------------------------------------- CCF annotation -> coarse region
# Coarse groups follow the reference preprocessing loop in preprocessing_DJ_2022Aug.py:
#   ['ALM','Medulla','Midbrain','Striatum','Thalamus','Pons','Cerebellum','Hypothalamus',
#    'Hippocampus','Orbital','OtherCortex','Olfactory','CorticalSubplate','Pallidum']
# Rules are applied in order; validated against the per-area good-unit counts reported in
# the data paper (striatum 7664, midbrain 7495, thalamus 12808, medulla 2928 -- all exact).
REGION_RULES = [
 ('Hypothalamus', ['Hypothalamus', 'hypothalamic area', 'hypothalamic nucleus', 'Zona incerta',
   'Mammillary', 'mammillary', 'Tuberomammillary', 'Supramammillary', 'Subthalamic nucleus',
   'preoptic', 'Preoptic', 'Periventricular', 'Median eminence', 'Parasubthalamic',
   'Fields of Forel', 'Suprachiasmatic', 'Subfornical organ', 'Supraoptic', 'Retrochiasmatic',
   'Tuberal nucleus', 'Arcuate hypothalamic', 'Dorsomedial nucleus of the hypothalamus',
   'Ventromedial hypothalamic', 'Posterior hypothalamic', 'Anterior hypothalamic',
   'Lateral hypothalamic']),
 ('Medulla', ['Gigantocellular reticular', 'Intermediate reticular', 'Parvicellular reticular',
   'Magnocellular reticular', 'Medullary reticular', 'Nucleus of the solitary', 'Parasolitary',
   'Nucleus ambiguus', 'Inferior olivary', 'Spinal nucleus of the trigeminal',
   'vestibular nucleus', 'Vestibular nuclei', 'Nucleus prepositus', 'cochlear nucleus',
   'Cochlear nuclei', 'Nucleus raphe magnus', 'Nucleus raphe obscurus', 'Nucleus raphe pallidus',
   'Paragigantocellular', 'Lateral reticular nucleus', 'Gracile nucleus', 'Cuneate nucleus',
   'External cuneate', 'Medulla', 'Linear nucleus of the medulla', 'Intercalated nucleus',
   'Facial motor nucleus', 'Dorsal motor nucleus of the vagus', 'Hypoglossal nucleus',
   'abducens', 'Abducens', 'Perihypoglossal', 'Area postrema', 'Nucleus x', 'Nucleus y',
   'Efferent vestibular', 'Efferent cochlear', 'Nucleus of Roller', 'Paramedian reticular',
   'Parapyramidal', 'Infracerebellar', 'Dorsal column nuclei']),
 ('Pons', ['Pontine reticular', 'Pontine gray', 'Tegmental reticular nucleus', 'Pons',
   'Parabrachial', 'Superior olivary', 'lateral lemniscus', 'Barrington', 'Locus ceruleus',
   'Laterodorsal tegmental', 'Sublaterodorsal', 'Supratrigeminal',
   'Principal sensory nucleus of the trigeminal', 'Motor nucleus of trigeminal',
   'Dorsal tegmental nucleus', 'Nucleus raphe pontis', 'Superior central nucleus raphe',
   'Pontine central gray', 'Koelliker-Fuse', 'trapezoid body', 'Nucleus incertus']),
 ('Midbrain', ['Midbrain', 'Superior colliculus', 'Inferior colliculus', 'Substantia nigra',
   'Periaqueductal gray', 'pretectal', 'Pretectal', 'Red nucleus', 'Ventral tegmental area',
   'Pedunculopontine', 'Nucleus of the brachium', 'Cuneiform nucleus', 'Interpeduncular nucleus',
   'Oculomotor nucleus', 'Trochlear nucleus', 'Edinger-Westphal', 'Darkschewitsch',
   'Interstitial nucleus of Cajal', 'Dorsal nucleus raphe', 'Midbrain trigeminal',
   'Parabigeminal', 'Olivary pretectal', 'Nucleus of the optic tract',
   'Nucleus of the posterior commissure', 'accessory optic tract', 'Nucleus sagulum',
   'Subcommissural organ', 'Retrorubral', 'Rostral linear nucleus raphe',
   'Central linear nucleus raphe', 'Intercollicular', 'Ventral tegmental nucleus',
   'Anterior tegmental nucleus']),
 ('Thalamus', ['thalamus', 'thalamic', 'Thalamus', 'geniculate', 'Reticular nucleus of the thalamus',
   'habenula', 'Habenula', 'Subparafascicular', 'Peripeduncular', 'Parafascicular',
   'Central lateral nucleus', 'Central medial nucleus', 'Paracentral nucleus', 'Intermediodorsal',
   'Nucleus of reuniens', 'Rhomboid nucleus', 'Submedial nucleus', 'Suprageniculate',
   'Anteromedial nucleus', 'Anteroventral nucleus', 'Anterodorsal nucleus', 'Interanterodorsal',
   'Interanteromedial', 'Lateral dorsal nucleus', 'Lateral posterior nucleus', 'Ethmoid nucleus',
   'Xiphoid', 'Perireunensis', 'Parataenial']),
 ('Striatum', ['Caudoputamen', 'Striatum', 'Nucleus accumbens', 'Fundus of striatum',
   'Olfactory tubercle', 'Lateral septal', 'Septofimbrial', 'Septohippocampal',
   'Medial amygdalar', 'Central amygdalar', 'Intercalated amygdalar', 'Anterior amygdalar',
   'accessory olfactory tract']),
 ('Pallidum', ['Pallidum', 'Globus pallidus', 'Substantia innominata', 'stria terminalis',
   'Magnocellular nucleus', 'Medial septal nucleus', 'Diagonal band',
   'Triangular nucleus of septum']),
 ('Hippocampus', ['Field CA1', 'Field CA2', 'Field CA3', 'Dentate gyrus', 'Ammon', 'Subiculum',
   'subiculum', 'Entorhinal area', 'Hippocamp', 'Induseum griseum', 'Fasciola cinerea',
   'Retrohippocampal']),
 ('CorticalSubplate', ['Claustrum', 'Endopiriform', 'Lateral amygdalar nucleus',
   'Basolateral amygdalar', 'Basomedial amygdalar', 'Posterior amygdalar', 'Cortical subplate']),
 ('Olfactory', ['Anterior olfactory nucleus', 'Piriform area', 'Olfactory areas', 'Taenia tecta',
   'lateral olfactory tract', 'Cortical amygdalar area', 'Piriform-amygdalar', 'Postpiriform',
   'olfactory bulb', 'Dorsal peduncular']),
 ('Cerebellum', ['Cerebell', 'Lobule', 'Uvula', 'Nodulus', 'Pyramus', 'Declive', 'Culmen',
   'Central lobule', 'Lingula', 'Flocculus', 'Paraflocculus', 'Simple lobule', 'Ansiform',
   'Paramedian lobule', 'Copula pyramidis', 'Crus 1', 'Crus 2', 'Fastigial nucleus',
   'Interposed nucleus', 'Dentate nucleus', 'Vermal regions', 'Hemispheric regions',
   'arbor vitae']),
 ('ALM', ['Secondary motor area', 'Primary motor area']),
 ('Orbital', ['Orbital area']),
]
CORTEX_KEYWORDS = ['area', 'areas', 'cortex', 'Cortex', 'Field', 'Retrosplenial', 'Gustatory',
                   'Visceral', 'Auditory', 'Visual', 'Somatosensory', 'Frontal pole', 'Prelimbic',
                   'Infralimbic', 'Anterior cingulate', 'Agranular insular', 'Ectorhinal',
                   'Perirhinal', 'Temporal association', 'Posterior parietal', 'layer', 'Layer']


def map_annotation(name):
    """Map a fine CCF annotation string to one of the 14 coarse reference groups."""
    n = str(name).strip()
    for group, terms in REGION_RULES:
        for t in terms:
            if t in n:
                return group
    if any(k in n for k in CORTEX_KEYWORDS):
        return 'OtherCortex'
    return 'Unknown'


# ------------------------------------------------------------------------ helper routines
def bin_spikes(spike_times_list, go_times):
    """Spike counts -> firing rate (Hz) in non-overlapping BIN_SIZE bins around the go cue.

    Analogue of the reference `sliding_histogram(..., rate=True)` with stride == bin width.

    Args:
        spike_times_list: list of (n_spikes,) sorted arrays, one per neuron (session clock).
        go_times: (n_trials,) go-cue times in session clock.
    Returns:
        (n_neurons, n_trials, NBINS) float32 firing rates in Hz.
    """
    ntr = len(go_times)
    edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()
    out = np.empty((len(spike_times_list), ntr, NBINS), dtype=np.float32)
    for k, st in enumerate(spike_times_list):
        idx = np.searchsorted(st, edges).reshape(ntr, NBINS + 1)
        out[k] = np.diff(idx, axis=1)
    out /= np.float32(BIN_SIZE)
    return out


def tone_onset_times(sample_starts, trial_starts, go_times):
    """Last sample(tone)-epoch start at or before the go cue, for each trial.

    Early licking triggers a replay of the sample/delay epoch, so a trial can contain several
    `sample_start_times`; the *last* one before the go cue is the tone the animal responded to.
    """
    ntr = len(go_times)
    out = np.full(ntr, np.nan)
    if len(sample_starts):
        tidx = np.searchsorted(trial_starts, sample_starts, side='right') - 1
        ok = (tidx >= 0) & (tidx < ntr)
        tidx, ev = tidx[ok], sample_starts[ok]
        ok2 = ev <= go_times[tidx]
        tidx, ev = tidx[ok2], ev[ok2]
        order = np.argsort(ev, kind='stable')          # ascending -> last write wins
        out[tidx[order]] = ev[order]
    return out


def photostim_binary(stim_on, stim_off, trial_starts, go_times):
    """(n_trials, NBINS) binary array: is photostimulation on during this bin?"""
    ntr = len(go_times)
    ps = np.zeros((ntr, NBINS), dtype=np.float32)
    if len(stim_on) == 0:
        return ps
    tidx = np.searchsorted(trial_starts, stim_on, side='right') - 1
    for i, on, off in zip(tidx, stim_on, stim_off):
        if i < 0 or i >= ntr:
            continue
        on_rel, off_rel = on - go_times[i], off - go_times[i]
        if off_rel < on_rel:                      # guard against corrupt entries
            on_rel, off_rel = off_rel, on_rel
        # a bin is 'on' if the stimulation interval overlaps the bin
        overlap = (BIN_EDGES_REL[:-1] < off_rel) & (BIN_EDGES_REL[1:] > on_rel)
        ps[i, overlap] = 1.0
    return ps


def bin_tongue(ts, y, lik, go_times):
    """Mean visible tongue y-position per 50 ms bin, NaN where the tongue is not visible.

    DeepLabCut side-camera tracking (`Camera0_side_TongueTracking`), aligned to the go cue as in
    Sherlock/align_markers.py.  Following the method paper, frames with a 5-sigma velocity
    outlier are discarded; unlike the method paper (which imputes the mean for occluded frames)
    the decoder task asks for an explicit 'not visible' class, so bins without a valid visible
    frame are left as NaN and encoded as class 3 downstream.
    """
    ntr = len(go_times)
    visible = lik > TONGUE_LIK_THRESH
    valid = visible.copy()
    if visible.sum() > 10:
        iv = np.where(visible)[0]
        dy = np.diff(y[iv]) / np.maximum(np.diff(ts[iv]), 1e-6)
        s = np.std(dy)
        if s > 0:
            bad = np.abs(dy) > VELOCITY_SIGMA * s
            # a large jump invalidates the frame after the jump
            valid[iv[1:][bad]] = False
    res = np.full((ntr, NBINS), np.nan, dtype=np.float32)
    t0 = go_times + OFF_START
    lo = np.searchsorted(ts, t0)
    hi = np.searchsorted(ts, go_times + OFF_END)
    for i in range(ntr):
        a, b = lo[i], hi[i]
        if b <= a:
            continue
        tsel = ts[a:b]
        vsel = valid[a:b]
        if not vsel.any():
            continue
        bi = np.floor((tsel[vsel] - t0[i]) / BIN_SIZE).astype(int)
        np.clip(bi, 0, NBINS - 1, out=bi)
        cnt = np.bincount(bi, minlength=NBINS)
        sm = np.bincount(bi, weights=y[a:b][vsel], minlength=NBINS)
        nz = cnt > 0
        res[i, nz] = (sm[nz] / cnt[nz]).astype(np.float32)
    return res


def discretize_tongue(y_binned):
    """Per-session discretisation of tongue y into 4 classes (task specification)."""
    vis = np.isfinite(y_binned)
    cls = np.full(y_binned.shape, 3, dtype=np.int64)     # 3 = not visible
    if vis.sum() > 0:
        p40, p60 = np.percentile(y_binned[vis], [40, 60])
        v = y_binned[vis]
        c = np.where(v < p40, 0, np.where(v <= p60, 1, 2))
        cls[vis] = c
    else:
        p40 = p60 = np.nan
    return cls, float(p40), float(p60)


# ------------------------------------------------------------------- per-session conversion
def convert_session(path, show_processing=False):
    """Convert one NWB session file; returns a dict or None if the session is dropped."""
    t_start = time.time()
    timing = {}
    with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
        nwb = io.read()
        ident = nwb.identifier
        subject = str(nwb.subject.subject_id)
        subject_name = str(nwb.subject.description)

        # ---------------- units: QC-classifier 'good' units only (reference neuron curation)
        t0 = time.time()
        units = nwb.units
        classification = np.asarray(units['classification'][:]).astype(str)
        good = np.where(classification == 'good')[0]
        if len(good) == 0:
            return {'identifier': ident, 'dropped': 'no good units'}
        anno = np.asarray(units['anno_name'][:]).astype(str)[good]
        # CCF coordinates of each unit's first electrode (fast VectorIndex access)
        vi = units['electrodes']
        ends = np.asarray(vi.data[:])
        starts = np.concatenate([[0], ends[:-1]])
        elec_ids = np.asarray(vi.target.data[:])[starts][good]
        edf = nwb.electrodes.to_dataframe()
        ccf_x = edf['x'].values.astype(float)[elec_ids]
        ccf_y = edf['y'].values.astype(float)[elec_ids]
        ccf_z = edf['z'].values.astype(float)[elec_ids]
        side = np.where(ccf_x >= ML_MIDLINE, 'left', 'right')   # reference midline convention
        groups = np.array([map_annotation(a) for a in anno])
        region_labels = np.array(['%s %s' % (s, g) for s, g in zip(side, groups)])
        timing['units_meta'] = time.time() - t0

        # ---------------- trials
        t0 = time.time()
        tr = nwb.trials
        trial_start = np.asarray(tr['start_time'][:], dtype=float)
        trial_stop = np.asarray(tr['stop_time'][:], dtype=float)
        outcome = np.asarray(tr['outcome'][:]).astype(str)
        instruction = np.asarray(tr['trial_instruction'][:]).astype(str)
        early = np.asarray(tr['early_lick'][:]).astype(str)
        auto_water = np.asarray(tr['auto_water'][:]).astype(int)
        free_water = np.asarray(tr['free_water'][:]).astype(int)
        stim_flag_tbl = np.asarray(tr['photostim_power'][:]).astype(str) != 'N/A'

        be = nwb.acquisition['BehavioralEvents'].time_series
        go = np.asarray(be['go_start_times'].timestamps[:], dtype=float)
        sample_starts = np.asarray(be['sample_start_times'].timestamps[:], dtype=float)
        stim_on = np.asarray(be['photostim_start_times'].timestamps[:], dtype=float)
        stim_off = np.asarray(be['photostim_stop_times'].timestamps[:], dtype=float)
        assert len(go) == len(trial_start), 'go cue count != trial count in %s' % ident
        timing['trials'] = time.time() - t0

        # trial curation: drop auto-water / free-water trials (reference get_regular_trial_mask);
        # early-lick, ignore/miss and photostim trials are KEPT because they are decoder outputs
        # / inputs in this task.
        keep = (auto_water == 0) & (free_water == 0)
        n_dropped_water = int((~keep).sum())
        kidx = np.where(keep)[0]
        ntr = len(kidx)
        if ntr < 2:
            return {'identifier': ident, 'dropped': 'fewer than 2 usable trials'}

        # ---------------- neural: spikes -> firing rates aligned to the go cue
        t0 = time.time()
        spike_lists = [np.asarray(units['spike_times'][int(i)]) for i in good]
        timing['spike_read'] = time.time() - t0
        t0 = time.time()
        fr = bin_spikes(spike_lists, go[kidx])                    # (n_neurons, ntr, NBINS)
        timing['binning'] = time.time() - t0

        # --- exclude trials with no ephys at all (acquisition gaps).
        # In some sessions the probes stop before the behaviour does, or single trials are
        # missing from the ephys stream; such trials contain zero spikes across *every* good
        # unit in the whole analysis window.  They carry no neural information and would be
        # pure noise for the decoder, so they are treated as invalid data periods and removed.
        spikes_per_trial = fr.sum(axis=(0, 2))
        rec = spikes_per_trial > 0
        n_dropped_norec = int((~rec).sum())
        if n_dropped_norec:
            fr = fr[:, rec, :]
            kidx = kidx[rec]
            ntr = int(rec.sum())
        if ntr < 2:
            return {'identifier': ident, 'dropped': 'fewer than 2 trials with ephys'}

        # ---------------- inputs
        t0 = time.time()
        tone = tone_onset_times(sample_starts, trial_start, go)[kidx]
        tone_rel_go = tone - go[kidx]                             # negative, ~ -1.85 s
        # time since tone onset at each bin centre (s)
        time_from_tone = (BIN_CENTERS_REL[None, :] - tone_rel_go[:, None]).astype(np.float32)
        ps_all = photostim_binary(stim_on, stim_off, trial_start, go)
        photostim = ps_all[kidx]
        timing['inputs'] = time.time() - t0

        # ---------------- outputs
        t0 = time.time()
        out_k = outcome[kidx]
        instr_k = instruction[kidx]
        # choice: hit -> instructed side, miss -> opposite side, ignore -> no lick
        choice = np.where(out_k == 'ignore', 2,
                          np.where(out_k == 'hit',
                                   np.where(instr_k == 'left', 0, 1),
                                   np.where(instr_k == 'left', 1, 0))).astype(np.int64)
        outcome_code = np.where(out_k == 'ignore', 0, np.where(out_k == 'miss', 1, 2)).astype(np.int64)
        early_code = (early[kidx] == 'early').astype(np.int64)

        bts = nwb.acquisition['BehavioralTimeSeries'].time_series
        tongue = bts['Camera0_side_TongueTracking']
        tt = np.asarray(tongue.timestamps[:], dtype=float)
        tdata = np.asarray(tongue.data[:], dtype=float)
        ty_binned = bin_tongue(tt, tdata[:, 1], tdata[:, 2], go[kidx])
        tongue_cls, p40, p60 = discretize_tongue(ty_binned)
        timing['outputs'] = time.time() - t0

        # ---------------- assemble per-trial arrays
        neural_trials = [np.ascontiguousarray(fr[:, i, :]) for i in range(ntr)]
        input_trials = [np.stack([time_from_tone[i], photostim[i]]).astype(np.float32)
                        for i in range(ntr)]
        output_trials = [np.stack([np.full(NBINS, choice[i]),
                                   np.full(NBINS, outcome_code[i]),
                                   np.full(NBINS, early_code[i]),
                                   tongue_cls[i]]).astype(np.int64) for i in range(ntr)]

        # ---------------- bookkeeping / diagnostics
        obs_frac_start = float(np.mean(trial_start[kidx] - go[kidx] <= OFF_START))
        obs_frac_end = float(np.mean(trial_stop[kidx] - go[kidx] >= OFF_END))
        stim_flag_evt = photostim.max(axis=1) > 0
        stim_mismatch = int(np.sum(stim_flag_evt != stim_flag_tbl[kidx]))

        info = {
            'identifier': ident,
            'subject': subject,
            'subject_name': subject_name,
            'session_start_time': str(nwb.session_start_time),
            'n_neurons': int(len(good)),
            'n_units_total': int(len(classification)),
            'n_trials': int(ntr),
            'n_trials_raw': int(len(trial_start)),
            'n_trials_dropped_water': n_dropped_water,
            'n_trials_dropped_no_ephys': n_dropped_norec,
            'frac_trials_observed_at_start': obs_frac_start,
            'frac_trials_observed_at_end': obs_frac_end,
            'tongue_p40': p40, 'tongue_p60': p60,
            'frac_bins_tongue_visible': float(np.mean(np.isfinite(ty_binned))),
            'median_tone_onset_re_go': float(np.nanmedian(tone_rel_go)),
            'n_photostim_trials': int(stim_flag_evt.sum()),
            'photostim_table_mismatch': stim_mismatch,
            'timing': timing,
        }

        result = {
            'identifier': ident,
            'subject': subject,
            'neural': neural_trials,
            'input': input_trials,
            'output': output_trials,
            'region_labels': region_labels,
            'ccf': np.stack([ccf_x, ccf_y, ccf_z], axis=1).astype(np.float32),
            'anno': anno,
            'info': info,
        }

        if show_processing:
            make_processing_plots(ident, nwb, kidx, go, trial_start, trial_stop, tone,
                                  spike_lists, fr, time_from_tone, photostim, tt, tdata,
                                  ty_binned, tongue_cls, p40, p60, choice, outcome_code,
                                  early_code, region_labels)
        info['total_time'] = time.time() - t_start
        return result


# ------------------------------------------------------------------------ diagnostics plots
def make_processing_plots(ident, nwb, kidx, go, trial_start, trial_stop, tone, spike_lists,
                          fr, time_from_tone, photostim, tt, tdata, ty_binned, tongue_cls,
                          p40, p60, choice, outcome_code, early_code, region_labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ntr = len(kidx)
    stim_trials = np.where(photostim.max(axis=1) > 0)[0]
    itrial = int(stim_trials[0]) if len(stim_trials) else 0
    gidx = kidx[itrial]
    g = go[gidx]

    fig, ax = plt.subplots(4, 2, figsize=(20, 18))

    # (0,0) raster of raw spikes for one trial vs binned rates
    a = ax[0, 0]
    nshow = min(40, len(spike_lists))
    for k in range(nshow):
        s = spike_lists[k]
        s = s[(s >= g + OFF_START) & (s < g + OFF_END)] - g
        a.plot(s, np.full(len(s), k), '|', color='k', ms=3)
    a.axvline(0, color='r', label='go cue')
    a.axvline(tone[itrial] - g, color='g', label='tone onset')
    a.axvline(trial_start[gidx] - g, color='b', ls='--', label='trial start')
    a.axvline(trial_stop[gidx] - g, color='b', ls=':', label='trial stop')
    a.set(xlabel='time re go cue (s)', ylabel='neuron', title='%s trial %d: raw spikes' % (ident, itrial))
    a.legend(fontsize=7)

    # (0,1) binned rates of the same neurons (image) -- checks alignment/binning
    a = ax[0, 1]
    im = a.imshow(fr[:nshow, itrial, :], aspect='auto', origin='lower',
                  extent=[OFF_START, OFF_END, 0, nshow], cmap='magma')
    a.axvline(0, color='r'); a.axvline(tone[itrial] - g, color='g')
    a.set(xlabel='time re go cue (s)', ylabel='neuron', title='binned firing rate (Hz), same trial')
    plt.colorbar(im, ax=a)

    # (1,0) population PSTH: raw spike histogram vs converted rates
    a = ax[1, 0]
    raw = np.zeros(NBINS)
    sub = min(60, len(spike_lists))
    for k in range(sub):
        s = spike_lists[k]
        for i in kidx[:200]:
            m = s[(s >= go[i] + OFF_START) & (s < go[i] + OFF_END)] - go[i]
            raw += np.histogram(m, bins=BIN_EDGES_REL)[0]
    raw = raw / (min(200, ntr) * sub * BIN_SIZE)
    a.plot(BIN_CENTERS_REL, raw, 'k', label='independent histogram of raw spikes')
    a.plot(BIN_CENTERS_REL, fr[:sub, :200, :].mean(axis=(0, 1)), 'r--', label='converted neural mean')
    a.axvline(0, color='r', alpha=.3); a.axvline(-1.85, color='g', alpha=.3)
    a.set(xlabel='time re go cue (s)', ylabel='rate (Hz)', title='population PSTH sanity check')
    a.legend(fontsize=8)

    # (1,1) inputs for the example trial
    a = ax[1, 1]
    a.plot(BIN_CENTERS_REL, time_from_tone[itrial], label='input0: time from tone onset (s)')
    a.plot(BIN_CENTERS_REL, photostim[itrial], label='input1: photostim on')
    a.axvline(0, color='r', alpha=.3, label='go cue')
    a.axhline(0, color='grey', lw=.5)
    a.axvline(tone[itrial] - g, color='g', alpha=.5, label='tone onset (raw)')
    be = nwb.acquisition['BehavioralEvents'].time_series
    son = np.asarray(be['photostim_start_times'].timestamps[:])
    sof = np.asarray(be['photostim_stop_times'].timestamps[:])
    m = (son >= trial_start[gidx]) & (son <= trial_stop[gidx])
    for s0, s1 in zip(son[m] - g, sof[m] - g):
        a.axvspan(s0, s1, color='orange', alpha=.3, label='raw photostim interval')
    a.set(xlabel='time re go cue (s)', title='inputs, trial %d' % itrial)
    a.legend(fontsize=7)

    # (2,0) tongue tracking raw vs binned for the example trial
    a = ax[2, 0]
    m = (tt >= g + OFF_START) & (tt < g + OFF_END)
    vis = tdata[m, 2] > TONGUE_LIK_THRESH
    a.plot(tt[m] - g, np.where(vis, tdata[m, 1], np.nan), '.', ms=2, label='raw tongue y (visible)')
    a.plot(BIN_CENTERS_REL, ty_binned[itrial], 'r.-', label='binned mean y')
    a.axhline(p40, color='g', ls='--', label='40th pct')
    a.axhline(p60, color='b', ls='--', label='60th pct')
    a.axvline(0, color='r', alpha=.3)
    a.set(xlabel='time re go cue (s)', ylabel='y (px)', title='tongue tracking, trial %d' % itrial)
    a.legend(fontsize=7)

    # (2,1) resulting tongue classes
    a = ax[2, 1]
    a.step(BIN_CENTERS_REL, tongue_cls[itrial], where='mid', label='tongue class')
    a.set(xlabel='time re go cue (s)', ylabel='class (0,1,2 = y pct; 3 = not visible)',
          title='discretised tongue output', ylim=[-0.5, 3.5])
    a.axvline(0, color='r', alpha=.3)
    a.legend(fontsize=7)

    # (3,0) per-trial outputs across the session
    a = ax[3, 0]
    a.plot(choice, '.', label='choice (0 L, 1 R, 2 none)')
    a.plot(outcome_code + 3.2, '.', label='outcome+3.2 (0 ign, 1 miss, 2 hit)')
    a.plot(early_code + 6.4, '.', label='early lick+6.4')
    a.set(xlabel='trial', title='per-trial outputs')
    a.legend(fontsize=7)

    # (3,1) tongue class fractions over time + region composition
    a = ax[3, 1]
    for c, lab in enumerate(OUTPUT_VALUES[3]):
        a.plot(BIN_CENTERS_REL, np.mean(tongue_cls == c, axis=0), label=lab)
    a.axvline(0, color='r', alpha=.3)
    a.set(xlabel='time re go cue (s)', ylabel='fraction of trials', title='tongue class vs time')
    a.legend(fontsize=7)

    fig.suptitle('Processing diagnostics: %s (%d neurons, %d trials, regions: %s)' %
                 (ident, fr.shape[0], ntr, ', '.join(sorted(set(region_labels))[:6])))
    fig.tight_layout()
    fname = 'processing_%s.png' % ident
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    print('  wrote %s' % fname)


# ------------------------------------------------------------------------------------ main
def list_sessions():
    import glob
    return sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))


def _worker(args):
    path, show = args
    try:
        return convert_session(path, show_processing=show)
    except Exception as e:  # pragma: no cover - defensive
        import traceback
        return {'identifier': os.path.basename(path), 'error': traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    files = list_sessions()
    if args.sample:
        files = files[:2]
    print('Converting %d sessions -> %s' % (len(files), args.outfile))
    t_all = time.time()

    show_n = 2 if args.show_processing else 0
    tasks = [(f, i < show_n) for i, f in enumerate(files)]

    results = []
    if args.workers > 1 and len(files) > 1:
        with ProcessPoolExecutor(min(args.workers, len(files))) as ex:
            for i, r in enumerate(ex.map(_worker, tasks)):
                results.append(r)
                if (i + 1) % 10 == 0 or i == len(files) - 1:
                    print('  %3d/%d done (%.1f s elapsed)' % (i + 1, len(files), time.time() - t_all))
    else:
        for i, t in enumerate(tasks):
            results.append(_worker(t))
            print('  %d/%d %s (%.1f s)' % (i + 1, len(files), results[-1].get('identifier'),
                                           time.time() - t_all))

    errors = [r for r in results if 'error' in r]
    for r in errors:
        print('ERROR in %s:\n%s' % (r['identifier'], r['error']))
    dropped = [r for r in results if r.get('dropped')]
    for r in dropped:
        print('DROPPED %s: %s' % (r['identifier'], r['dropped']))
    good = [r for r in results if 'neural' in r]
    print('Converted %d sessions (%d dropped, %d errors)' % (len(good), len(dropped), len(errors)))

    subjects = sorted({r['subject'] for r in good})
    subject_idx = np.array([subjects.index(r['subject']) for r in good], dtype=np.int64)
    brain_regions = sorted({lab for r in good for lab in r['region_labels']})
    region_index = {lab: i for i, lab in enumerate(brain_regions)}
    brain_region_idx = [np.array([region_index[l] for l in r['region_labels']], dtype=np.int64)
                        for r in good]

    session_info = [r['info'] for r in good]
    n_neurons = int(sum(i['n_neurons'] for i in session_info))
    n_trials = int(sum(i['n_trials'] for i in session_info))

    data = {
        'neural': [r['neural'] for r in good],
        'input': [r['input'] for r in good],
        'output': [r['output'] for r in good],
        'subjects': subjects,
        'subject_idx': subject_idx,
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'dataset': 'Mesoscale Activity Map (DANDI:000363), Chen et al. 2024 Cell; '
                       'analysed in Wang*, Kurgyis* et al. 2025 Nat Neurosci',
            'task_description': (
                'Head-fixed mice perform an auditory delayed-response (memory-guided movement) '
                'task. A tone (3 kHz or 12 kHz, three 150 ms pips) instructs lick-left or '
                'lick-right; after a 1.2 s delay an auditory go cue (6 kHz, 0.1 s) permits the '
                'response. Outputs decoded from neural activity: lick direction choice '
                '(left/right/no lick), trial outcome (ignore/miss/hit), early lick (no/yes) and '
                'discretised tongue y-position from side-view DeepLabCut tracking. Decoder '
                'inputs: time since tone (sample epoch) onset and whether ALM photoinhibition '
                'is on.'),
            'time_bin_size': BIN_SIZE * 1000.0,
            'time_bin_size_units': 'ms',
            'temporal_alignment_event': 'auditory go cue onset (BehavioralEvents go_start_times)',
            'off_start': OFF_START,
            'off_end': OFF_END,
            'bin_centers_re_go_cue_s': BIN_CENTERS_REL.tolist(),
            'neural_units': 'firing rate (Hz), spike counts per 50 ms bin / 0.05 s',
            'n_sessions': len(good),
            'n_subjects': len(subjects),
            'n_neurons_total': n_neurons,
            'n_trials_total': n_trials,
            'neuron_curation': (
                "units with units.classification == 'good' (output of the region-specific QC "
                'classifiers of Chen, Liu et al. 2023 white paper, the same good-unit lists used '
                'by the reference pipeline). These units also all carry a CCF annotation, which '
                "reproduces the reference's requirement of joint ephys+histology."),
            'trial_curation': (
                'trials with zero spikes from every good unit across the whole analysis window \n'
                '(ephys acquisition gaps) removed; auto-water and free-water trials removed (reward independent of the action, as '
                'in the reference get_regular_trial_mask). Early-lick, no-response (ignore), '
                'error (miss) and photostimulation trials are KEPT because the decoder task '
                'defines them as outputs/inputs.'),
            'session_curation': 'sessions without any good unit are dropped (1 session).',
            'tongue_discretisation': (
                'per-session percentiles (40th, 60th) of the binned, visible tongue y-position; '
                'bins with no frame of likelihood > 0.5 (or only 5-sigma velocity outliers) are '
                "class 3 = 'not visible'"),
            'video_tracking': 'DeepLabCut side-view camera (Camera0_side_TongueTracking), 3.4 ms frames',
            'notes_unobserved_time': (
                'spikes are recorded only inside each trial interval (units.obs_intervals == '
                'trials start/stop); 96.9 % of trials cover go-2.5 s and 84.4 % cover go+1.5 s '
                '(error/miss trials terminate early). Bins outside the recorded interval contain '
                'zero spikes by construction; per-session coverage is reported in session_info.'),
            'session_info': session_info,
        },
    }

    print('Writing pickle ...')
    t0 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print('wrote %s (%.2f GB) in %.1f s' % (args.outfile,
                                            os.path.getsize(args.outfile) / 1e9, time.time() - t0))

    # ------------------------------------------------------------------ summary statistics
    print('\n==== SUMMARY ====')
    print('sessions %d  subjects %d  neurons %d  trials %d' % (len(good), len(subjects), n_neurons, n_trials))
    print('neurons/session: mean %.1f  range %d-%d' % (
        np.mean([i['n_neurons'] for i in session_info]),
        min(i['n_neurons'] for i in session_info), max(i['n_neurons'] for i in session_info)))
    print('trials/session: mean %.1f  range %d-%d' % (
        np.mean([i['n_trials'] for i in session_info]),
        min(i['n_trials'] for i in session_info), max(i['n_trials'] for i in session_info)))
    reg = Counter()
    for r in good:
        reg.update(r['region_labels'].tolist())
    coarse = Counter()
    for k, v in reg.items():
        coarse[k.split(' ', 1)[1]] += v
    print('neurons per coarse region:', dict(coarse.most_common()))
    outs = {n: Counter() for n in OUTPUT_NAMES}
    for r in good:
        for o in r['output']:
            outs['choice'][int(o[0, 0])] += 1
            outs['outcome'][int(o[1, 0])] += 1
            outs['early_lick'][int(o[2, 0])] += 1
            for c, n in zip(*np.unique(o[3], return_counts=True)):
                outs['tongue_y'][int(c)] += int(n)
    for k, v in outs.items():
        tot = sum(v.values())
        print('%s distribution:' % k, {OUTPUT_VALUES[OUTPUT_NAMES.index(k)][c]: round(n / tot, 4)
                                       for c, n in sorted(v.items())})
    tf = np.concatenate([r['input'][0][0] for r in good])
    print('input time_from_tone_onset range [%.3f, %.3f]' % (tf.min(), tf.max()))
    nstim = sum(i['n_photostim_trials'] for i in session_info)
    print('photostim trials: %d (%.1f %%); table/event mismatches: %d' % (
        nstim, 100 * nstim / n_trials, sum(i['photostim_table_mismatch'] for i in session_info)))
    print('median tone onset re go cue: %.3f s' % np.median([i['median_tone_onset_re_go'] for i in session_info]))
    print('mean frac trials observed at window start %.4f, end %.4f' % (
        np.mean([i['frac_trials_observed_at_start'] for i in session_info]),
        np.mean([i['frac_trials_observed_at_end'] for i in session_info])))
    tt_ = [i['timing'] for i in session_info]
    for k in tt_[0]:
        print('timing %-12s mean %.2f s/session' % (k, np.mean([t[k] for t in tt_])))
    print('total elapsed %.1f s' % (time.time() - t_all))


if __name__ == '__main__':
    main()
