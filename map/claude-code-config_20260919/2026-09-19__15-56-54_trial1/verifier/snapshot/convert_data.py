#!/usr/bin/env python3
"""
Convert the Mesoscale Activity Map (MAP) dataset (DANDI:000363, Chen et al. 2024)
from NWB into the decoder-ready pickle format.

Usage
-----
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Options
-------
    --full             process all sessions (default)
    --sample           process only 2 sessions (for testing)
    --show-processing  produce diagnostic plots for up to 2 sessions
                       (processing_<session_id>.png)
    --workers N        number of parallel worker processes (default: 12)

Processing summary (see CONVERSION_NOTES.md for the full justification)
-----------------------------------------------------------------------
* alignment     : go-cue onset (acquisition/BehavioralEvents/go_start_times)
* window        : [-2.5, +1.5) s, 80 non-overlapping 50 ms bins
* neural        : firing rate (spikes/s) of every classifier-"good" unit that has a
                  CCF annotation  (= the reference pipeline's QC, qc_mode='classifier')
* inputs        : [time from tone onset (s), photostimulation on/off]
* outputs       : [choice, outcome, early lick, tongue y-position class]
* trial curation: trials covered by the ephys recording, excluding auto-water and
                  free-water trials (reference `get_regular_trial_mask`, minus the
                  early-lick / no-response / photostim terms which are decoder targets)
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time
from collections import OrderedDict

import h5py
import numpy as np

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
DATA_DIR = '/app/data'

OFF_START = -2.5          # s, signed time from go cue to start of the extracted window
OFF_END = 1.5             # s, signed time from go cue to end of the extracted window
BIN_SIZE = 0.05           # s, width of the (non-overlapping) spike-count bins
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))     # 80
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)  # (81,) relative to go cue
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])      # (80,)

TONGUE_LIKELIHOOD_THRESH = 0.5   # DLC likelihood is bimodal; any value in (0.01,0.99) works
TONGUE_PCTL = (40.0, 60.0)       # percentiles used to discretise tongue y-position

# CCF landmarks (Allen CCFv3, micrometres), used only for the ALM definition
CCF_MIDLINE_ML = 5700.0     # same value as the reference `helper_get_neuron_id_area`
CCF_BREGMA_AP = 5400.0      # AP coordinate of bregma
ALM_MIN_AP_MM = 2.0         # ALM = motor/frontal-pole cortex >= 2 mm anterior to bregma

INPUT_NAMES = ['time_from_tone_onset', 'photostim_on']
OUTPUT_NAMES = ['choice', 'outcome', 'early_lick', 'tongue_y_position']
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['<40th pct', '40th-60th pct', '>60th pct', 'not visible'],
]

# The 14 major brain-area groups used in Chen et al. 2024 (Fig. 1E / 2F) and in the
# reference code (`preprocessing_DJ_2022Aug.process_one_sess`).
BRAIN_REGIONS = ['ALM', 'Orbital', 'OtherCortex', 'Olfactory', 'Hippocampus',
                 'CorticalSubplate', 'Striatum', 'Pallidum', 'Thalamus', 'Hypothalamus',
                 'Midbrain', 'Pons', 'Medulla', 'Cerebellum']
REGION_IDX = {r: i for i, r in enumerate(BRAIN_REGIONS)}


# --------------------------------------------------------------------------------------
# CCF annotation -> major brain area
# --------------------------------------------------------------------------------------
# Substring rules mapping the 293 distinct `units/anno_name` strings in this release onto
# the Allen CCFv3 top-level divisions.  Validated against the per-area good-unit counts
# reported in the data paper (11/14 reproduce exactly, see CONVERSION_NOTES.md Step 9).
_PONS = ['Pontine reticular nucleus', 'Parabrachial nucleus', 'Nucleus of the lateral lemniscus',
         'Tegmental reticular nucleus', 'Locus ceruleus', 'Koelliker-Fuse subnucleus', 'Pons']
_MEDULLA = [
    'Gigantocellular reticular nucleus', 'Intermediate reticular nucleus',
    'Magnocellular reticular nucleus', 'Medial vestibular nucleus',
    'Spinal nucleus of the trigeminal', 'Parvicellular reticular nucleus', 'Medulla',
    'Inferior olivary complex', 'Paragigantocellular reticular nucleus', 'Facial motor nucleus',
    'Superior vestibular nucleus', 'Lateral vestibular nucleus', 'Medullary reticular nucleus',
    'Spinal vestibular nucleus', 'External cuneate nucleus', 'Nucleus of the solitary tract',
    'Dorsal motor nucleus of the vagus nerve', 'Hypoglossal nucleus', 'Nucleus raphe magnus',
    'Lateral reticular nucleus', 'Parasolitary nucleus', 'Parapyramidal nucleus',
    'Nucleus raphe obscurus', 'Nucleus x', 'Infracerebellar nucleus', 'Nucleus of Roller',
    'Cuneate nucleus', 'Gracile nucleus', 'Nucleus raphe pallidus', 'Nucleus prepositus',
    'Nucleus ambiguus', 'Area postrema', 'Accessory facial motor nucleus',
    'Dorsal cochlear nucleus', 'Ventral cochlear nucleus', 'Nucleus of the trapezoid body',
    'Efferent vestibular nucleus', 'Linear nucleus of the medulla', 'Perihypoglossal nuclei',
    'Paratrigeminal nucleus', 'Vestibular nuclei']
_CEREBELLUM = [
    'Lobule', 'Lobules', 'Simple lobule', 'Nodulus', 'Copula pyramidis', 'Uvula',
    'Interposed nucleus', 'Paramedian lobule', 'Declive', 'Fastigial nucleus', 'Crus 1',
    'Crus 2', 'Pyramus', 'Lingula', 'Culmen', 'Folium-tuber vermis', 'Ansiform lobule',
    'Flocculus', 'Paraflocculus', 'Dentate nucleus', 'Vermal regions', 'Hemispheric regions',
    'Cerebellum', 'Cerebellar cortex', 'Cerebellar nuclei', 'Central lobule', 'Arbor vitae']
_THALAMUS = [
    'thalamus', 'thalamic', 'Thalamus', 'Mediodorsal nucleus', 'Posterior complex',
    'Lateral posterior nucleus', 'Lateral dorsal nucleus', 'Paracentral nucleus',
    'Anteroventral nucleus', 'Anteromedial nucleus', 'Anterodorsal nucleus',
    'Parafascicular nucleus', 'Lateral habenula', 'Medial habenula', 'Habenula',
    'Medial geniculate complex', 'Dorsal part of the lateral geniculate complex',
    'Ventral part of the lateral geniculate complex', 'Rhomboid nucleus',
    'Suprageniculate nucleus', 'Subparafascicular nucleus', 'Subparafascicular area',
    'Perireunensis nucleus', 'Nucleus of reuniens', 'Peripeduncular nucleus', 'Reuniens',
    'Xiphoid thalamic nucleus', 'Ethmoid nucleus', 'Intermediate geniculate nucleus',
    'Posterior intralaminar thalamic nucleus']
_HYPOTHALAMUS = [
    'Zona incerta', 'Fields of Forel', 'Hypothalamus', 'Lateral hypothalamic area',
    'Posterior hypothalamic nucleus', 'Subthalamic nucleus', 'Tuberomammillary nucleus',
    'Paraventricular hypothalamic nucleus', 'Lateral preoptic area', 'Parasubthalamic nucleus',
    'Medial preoptic area', 'Medial preoptic nucleus', 'Anterior hypothalamic nucleus',
    'Dorsomedial nucleus of the hypothalamus', 'Ventromedial hypothalamic nucleus',
    'Arcuate hypothalamic nucleus', 'Mammillary body', 'Medial mammillary nucleus',
    'Supramammillary nucleus', 'Periventricular', 'Preparasubthalamic nucleus',
    'Retrochiasmatic area', 'Suprachiasmatic nucleus', 'Supraoptic nucleus',
    'Anteroventral preoptic nucleus', 'Anteroventral periventricular nucleus',
    'Ventral premammillary nucleus', 'Dorsal premammillary nucleus', 'Tuberal nucleus']
_MIDBRAIN = [
    'Midbrain', 'Superior colliculus', 'Anterior pretectal nucleus', 'Substantia nigra',
    'Red nucleus', 'Pedunculopontine nucleus', 'Inferior colliculus',
    'Nucleus of the optic tract', 'Posterior pretectal nucleus', 'Ventral tegmental area',
    'Nucleus sagulum', 'Nucleus of the brachium of the inferior colliculus',
    'Dorsal terminal nucleus', 'Medial terminal nucleus', 'Periaqueductal gray',
    'Olivary pretectal nucleus', 'Nucleus of the posterior commissure',
    'Edinger-Westphal nucleus', 'Oculomotor nucleus', 'Cuneiform nucleus',
    'Dorsal nucleus raphe', 'Interpeduncular nucleus', 'Lateral terminal nucleus',
    'Midbrain trigeminal nucleus', 'Trochlear nucleus', 'Parabigeminal nucleus',
    'Magnocellular nucleus', 'Midbrain reticular nucleus', 'Pretectal region',
    'Anterior tegmental nucleus', 'Ventral tegmental nucleus', 'Subcommissural organ',
    'Rostral linear nucleus raphe', 'Central linear nucleus raphe',
    'Supraoculomotor periaqueductal gray']
_STRIATUM = [
    'Caudoputamen', 'Fundus of striatum', 'Nucleus accumbens', 'Striatum', 'Olfactory tubercle',
    'Lateral septal nucleus', 'Central amygdalar nucleus', 'Intercalated amygdalar nucleus',
    'Medial amygdalar nucleus', 'Anterior amygdalar area', 'Septohippocampal nucleus',
    'Striatum-like amygdalar nuclei', 'Striatum dorsal region', 'Striatum ventral region',
    'Lateral septal complex', 'Islands of Calleja', 'Major island of Calleja',
    'Septofimbrial nucleus']
_PALLIDUM = [
    'Globus pallidus', 'Pallidum', 'Substantia innominata',
    'Bed nuclei of the stria terminalis', 'Triangular nucleus of septum',
    'Medial septal nucleus', 'Diagonal band nucleus', 'Magnocellular nucleus',
    'Bed nucleus of the anterior commissure']
_HIPPOCAMPUS = [
    'Field CA1', 'Field CA2', 'Field CA3', 'Dentate gyrus', 'Subiculum', 'Postsubiculum',
    'Presubiculum', 'Parasubiculum', 'Entorhinal area', 'Hippocampal formation',
    'Hippocampal region', 'Prosubiculum', 'Induseum griseum', 'Fasciola cinerea', 'Ammon',
    'Retrohippocampal region', 'Hippocampo-amygdalar transition area', 'Area prostriata']
_CTXSP = [
    'Claustrum', 'Endopiriform nucleus', 'Basolateral amygdalar nucleus',
    'Lateral amygdalar nucleus', 'Basomedial amygdalar nucleus', 'Posterior amygdalar nucleus',
    'Cortical subplate']
_OLFACTORY = [
    'Anterior olfactory nucleus', 'Piriform area', 'Olfactory areas', 'Taenia tecta',
    'Accessory olfactory bulb', 'Main olfactory bulb', 'Dorsal peduncular area',
    'Nucleus of the lateral olfactory tract', 'Cortical amygdalar area',
    'Piriform-amygdalar area', 'Postpiriform transition area', 'Olfactory bulb']
_ORBITAL = ['Orbital area']
_CORTEX = [
    'Primary motor area', 'Secondary motor area', 'Primary somatosensory area',
    'Supplemental somatosensory area', 'Gustatory areas', 'Visceral area',
    'Agranular insular area', 'Anterior cingulate area', 'Prelimbic area', 'Infralimbic area',
    'Frontal pole', 'Retrosplenial area', 'Temporal association areas', 'Perirhinal area',
    'Ectorhinal area', 'Primary auditory area', 'Dorsal auditory area', 'Ventral auditory area',
    'Posterior auditory area', 'Primary visual area', 'posteromedial visual area',
    'Anteromedial visual area', 'Anterolateral visual area', 'Lateral visual area',
    'Laterointermediate area', 'Rostrolateral area', 'Posterolateral visual area',
    'Anterior area', 'Postrhinal area', 'Isocortex', 'Somatomotor areas',
    'Somatosensory areas', 'Auditory areas', 'Visual areas',
    'Posterior parietal association areas', 'Cerebral cortex']

_GROUPS = [('Pons', _PONS), ('Medulla', _MEDULLA), ('Cerebellum', _CEREBELLUM),
           ('Hypothalamus', _HYPOTHALAMUS), ('Thalamus', _THALAMUS), ('Midbrain', _MIDBRAIN),
           ('Pallidum', _PALLIDUM), ('Striatum', _STRIATUM), ('Hippocampus', _HIPPOCAMPUS),
           ('CorticalSubplate', _CTXSP), ('Olfactory', _OLFACTORY), ('Orbital', _ORBITAL),
           ('OtherCortex', _CORTEX)]

# cortical annotations that can belong to ALM (anterior-lateral motor cortex)
_ALM_CANDIDATE_PREFIXES = ('Primary motor area', 'Secondary motor area', 'Frontal pole')


def annotation_to_group(name):
    """Map a single CCF annotation string to one of the 14 major groups (or None)."""
    for gname, keys in _GROUPS:
        for k in keys:
            if k in name:
                return gname
    return None


def assign_brain_regions(anno, ccf_ap):
    """Assign each unit to one of BRAIN_REGIONS.

    Args:
        anno: (n_units,) array of CCF annotation strings.
        ccf_ap: (n_units,) CCF AP coordinate (micrometres, Allen CCFv3 `z`).

    Returns:
        (n_units,) int array of indices into BRAIN_REGIONS.
    """
    ap_mm = (CCF_BREGMA_AP - np.asarray(ccf_ap, dtype=float)) / 1000.0
    out = np.empty(len(anno), dtype=np.int64)
    for i, name in enumerate(anno):
        grp = annotation_to_group(name)
        if grp is None:
            raise ValueError(f'Unmapped CCF annotation: {name!r}')
        if grp == 'OtherCortex' and name.startswith(_ALM_CANDIDATE_PREFIXES) \
                and ap_mm[i] >= ALM_MIN_AP_MM:
            grp = 'ALM'
        out[i] = REGION_IDX[grp]
    return out


# --------------------------------------------------------------------------------------
# Per-session conversion
# --------------------------------------------------------------------------------------
def _decode(arr):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])


def bin_spikes(spike_times, spike_index, unit_idx, go_times):
    """Bin spikes of the selected units into go-cue-aligned firing rates.

    Mirrors `preprocessing_DJ_2022Aug.sliding_histogram(..., rate=True)` with
    stride == bin width: spikes are counted in half-open bins [edge_k, edge_{k+1}).

    Args:
        spike_times: (n_spikes,) concatenated spike times of all units, session time (s).
        spike_index: (n_units,) end offsets into `spike_times` (NWB ragged index).
        unit_idx: indices of the units to bin.
        go_times: (n_trials,) go-cue times of the trials to extract, session time (s).

    Returns:
        (n_units_sel, n_trials, N_BINS) float32 array of firing rates in spikes/s.
    """
    n_trials = len(go_times)
    # absolute bin edges, (n_trials, N_BINS+1)
    edges = go_times[:, None] + BIN_EDGES[None, :]
    flat_edges = np.ascontiguousarray(edges.ravel())

    fr = np.empty((len(unit_idx), n_trials, N_BINS), dtype=np.float32)
    for i, u in enumerate(unit_idx):
        lo = spike_index[u - 1] if u > 0 else 0
        st = spike_times[lo:spike_index[u]]
        pos = np.searchsorted(st, flat_edges).reshape(n_trials, N_BINS + 1)
        fr[i] = (pos[:, 1:] - pos[:, :-1]).astype(np.float32)
    fr /= BIN_SIZE                      # spike counts -> spikes/s
    return fr


def tongue_per_bin(track_data, track_ts, go_times):
    """Aggregate DeepLabCut tongue tracking into the go-cue-aligned bins.

    Args:
        track_data: (n_frames, 3) array of (x, y, likelihood).
        track_ts: (n_frames,) frame times in session time (s), sorted.
        go_times: (n_trials,) go-cue times.

    Returns:
        y: (n_trials, N_BINS) mean y over the visible frames of the bin (NaN if none).
        visible: (n_trials, N_BINS) bool, True if the bin contains a visible frame.
    """
    n_trials = len(go_times)
    y = np.full((n_trials, N_BINS), np.nan, dtype=np.float64)
    visible = np.zeros((n_trials, N_BINS), dtype=bool)

    ycol = track_data[:, 1]
    vis_all = track_data[:, 2] > TONGUE_LIKELIHOOD_THRESH

    lo = np.searchsorted(track_ts, go_times + OFF_START, side='left')
    hi = np.searchsorted(track_ts, go_times + OFF_END, side='left')
    for t in range(n_trials):
        a, b = lo[t], hi[t]
        if b <= a:
            continue
        rel = track_ts[a:b] - go_times[t]
        b_idx = np.floor((rel - OFF_START) / BIN_SIZE).astype(np.int64)
        np.clip(b_idx, 0, N_BINS - 1, out=b_idx)
        v = vis_all[a:b]
        if not v.any():
            continue
        cnt = np.bincount(b_idx[v], minlength=N_BINS)
        tot = np.bincount(b_idx[v], weights=ycol[a:b][v], minlength=N_BINS)
        m = cnt > 0
        y[t, m] = tot[m] / cnt[m]
        visible[t, m] = True
    return y, visible


def discretize_tongue(y, visible):
    """Discretise per-bin tongue y-position using per-session percentiles.

    0: y < 40th percentile, 1: 40th-60th percentile, 2: y > 60th percentile,
    3: tongue not visible.  Percentiles are taken over the visible bins of the session
    (y is meaningless where the tongue is not visible).
    """
    cls = np.full(y.shape, 3, dtype=np.int64)
    if visible.any():
        vals = y[visible]
        p40, p60 = np.percentile(vals, TONGUE_PCTL)
        cls[visible & (y < p40)] = 0
        cls[visible & (y >= p40) & (y <= p60)] = 1
        cls[visible & (y > p60)] = 2
    else:
        p40 = p60 = np.nan
    return cls, (p40, p60)


def process_session(fname, want_raw=False):
    """Convert a single NWB session file.

    Returns None if the session has no usable units/trials.
    """
    t0 = time.time()
    with h5py.File(fname, 'r') as f:
        ident = f['identifier'][()].decode()
        mouse = f['general/subject/description'][()].decode().strip()
        subject_id = f['general/subject/subject_id'][()].decode().strip()

        tr = f['intervals/trials']
        n_trials_table = len(tr['id'])
        start_time = tr['start_time'][:]
        stop_time = tr['stop_time'][:]
        outcome = _decode(tr['outcome'][:])
        early = _decode(tr['early_lick'][:])
        instruction = _decode(tr['trial_instruction'][:])
        auto_water = tr['auto_water'][:]
        free_water = tr['free_water'][:]
        ps_onset = _decode(tr['photostim_onset'][:])
        ps_dur = _decode(tr['photostim_duration'][:])

        go = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
        sample_start = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
        assert len(go) == n_trials_table, f'{ident}: go cue count != trial count'

        # ---- units --------------------------------------------------------------
        u = f['units']
        classification = _decode(u['classification'][:])
        anno = _decode(u['anno_name'][:])
        keep = (classification == 'good') & (anno != '')
        unit_idx = np.where(keep)[0]
        if len(unit_idx) == 0:
            return None

        # CCF coordinates of each unit's electrode
        el = u['electrodes'][:]
        etab = f['general/extracellular_ephys/electrodes']
        ccf_ml = etab['x'][:][el[unit_idx]]
        ccf_ap = etab['z'][:][el[unit_idx]]

        # ---- which trials did the ephys recording actually cover? ---------------
        oii = u['obs_intervals_index'][:]
        n_obs = np.diff(np.concatenate([[0], oii]))
        n_obs_good = n_obs[unit_idx]
        observed = np.zeros(n_trials_table, dtype=bool)
        if np.all(n_obs_good == n_trials_table):
            observed[:] = True
        else:
            # all units of a session share the same contiguous observed trial set;
            # derive it from the union of their observation intervals to be safe.
            oi = u['obs_intervals']
            for k in unit_idx:
                o = oi[(oii[k] - n_obs[k]):oii[k]]
                idx = np.searchsorted(start_time, o[:, 0] + 1e-9) - 1
                observed[idx[idx >= 0]] = True

        # ---- trial curation ------------------------------------------------------
        valid = observed & (auto_water == 0) & (free_water == 0)
        trials = np.where(valid)[0]
        if len(trials) < 2:
            return None

        go_v = go[trials]

        # ---- neural --------------------------------------------------------------
        spike_index = u['spike_times_index'][:]
        spike_times = u['spike_times'][:]
        fr = bin_spikes(spike_times, spike_index, unit_idx, go_v)
        del spike_times

        # Drop trials in which not one of the (>=90) simultaneously recorded neurons
        # fired a single spike in the whole 4 s window.  This is a recording-coverage
        # failure, not biology: in this release `units/obs_intervals` occasionally
        # lists one more trial than the spike record actually extends to (e.g.
        # SC015_20190208_133600 trial 159, whose window starts after the last spike
        # in the file).
        has_data = fr.sum(axis=(0, 2)) > 0
        n_zero_trials = int((~has_data).sum())
        if n_zero_trials:
            fr = fr[:, has_data, :]
            trials = trials[has_data]
            go_v = go_v[has_data]
            if len(trials) < 2:
                return None

        # ---- inputs --------------------------------------------------------------
        # tone onset: last sample-epoch start before the go cue of that trial
        j = np.searchsorted(sample_start, go_v - 1e-9) - 1
        assert np.all(j >= 0), f'{ident}: trial without a preceding tone onset'
        tone_time = sample_start[j]
        # time from tone onset at every bin centre, (n_trials, N_BINS)
        time_from_tone = (BIN_CENTERS[None, :] + (go_v - tone_time)[:, None]).astype(np.float32)

        photostim = np.zeros((len(trials), N_BINS), dtype=np.float32)
        has_ps = ps_onset[trials] != 'N/A'
        if has_ps.any():
            on = np.array([float(x) for x in ps_onset[trials][has_ps]])
            dur = np.array([float(x) for x in ps_dur[trials][has_ps]])
            ps_t0 = start_time[trials][has_ps] + on - go_v[has_ps]   # rel. to go cue
            ps_t1 = ps_t0 + dur
            # a bin is "on" if it overlaps the stimulation interval
            ov = (BIN_EDGES[None, :-1] < ps_t1[:, None]) & (BIN_EDGES[None, 1:] > ps_t0[:, None])
            photostim[has_ps] = ov.astype(np.float32)

        # ---- outputs -------------------------------------------------------------
        oc = outcome[trials]
        ins = instruction[trials]
        choice = np.full(len(trials), 2, dtype=np.int64)              # 2 = no lick
        licked_left = ((oc == 'hit') & (ins == 'left')) | ((oc == 'miss') & (ins == 'right'))
        licked_right = ((oc == 'hit') & (ins == 'right')) | ((oc == 'miss') & (ins == 'left'))
        choice[licked_left] = 0
        choice[licked_right] = 1

        outcome_c = np.select([oc == 'ignore', oc == 'miss', oc == 'hit'], [0, 1, 2],
                              default=-1).astype(np.int64)
        assert np.all(outcome_c >= 0), f'{ident}: unexpected outcome value'
        early_c = (early[trials] == 'early').astype(np.int64)

        tongue_key = 'acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking'
        tdata = f[tongue_key + '/data'][:]
        tts = f[tongue_key + '/timestamps'][:]
        ty, tvis = tongue_per_bin(tdata, tts, go_v)
        tongue_c, pctl = discretize_tongue(ty, tvis)

        raw = None
        if want_raw:
            raw = dict(track_data=tdata, track_ts=tts, go=go_v, tone=tone_time,
                       start_time=start_time[trials], stop_time=stop_time[trials],
                       ty=ty, tvis=tvis, pctl=pctl, outcome=oc, instruction=ins,
                       spike_index=spike_index, unit_idx=unit_idx, fname=fname)

    region_idx = assign_brain_regions(anno[unit_idx], ccf_ap)

    # ---- assemble ---------------------------------------------------------------
    n_tr = len(trials)
    neural = [np.ascontiguousarray(fr[:, t, :]) for t in range(n_tr)]
    inputs = [np.stack([time_from_tone[t], photostim[t]]) for t in range(n_tr)]
    outputs = [np.stack([np.full(N_BINS, choice[t]), np.full(N_BINS, outcome_c[t]),
                         np.full(N_BINS, early_c[t]), tongue_c[t]]).astype(np.int64)
               for t in range(n_tr)]

    info = dict(
        session_id=ident, mouse=mouse, subject_id=subject_id, file=os.path.basename(fname),
        n_units_total=len(classification), n_units_good=int((classification == 'good').sum()),
        n_neurons=len(unit_idx), n_trials_table=n_trials_table, n_trials=n_tr,
        n_trials_observed=int(observed.sum()),
        n_excluded_water=int((observed & ((auto_water != 0) | (free_water != 0))).sum()),
        n_excluded_no_spikes=n_zero_trials,
        trial_index=trials.astype(np.int32),   # 0-based index into intervals/trials
        mean_rate=float(fr.mean()), tongue_pctl=[float(pctl[0]), float(pctl[1])],
        frac_tongue_visible=float(tvis.mean()),
        hemisphere_left_frac=float(np.mean(ccf_ml >= CCF_MIDLINE_ML)),
        proc_time=time.time() - t0,
    )
    return dict(neural=neural, input=inputs, output=outputs, region_idx=region_idx,
                info=info, raw=raw)


# --------------------------------------------------------------------------------------
# Diagnostic plots
# --------------------------------------------------------------------------------------
def make_processing_plot(res, fname_out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    raw = res['raw']
    info = res['info']
    neural, inputs, outputs = res['neural'], res['input'], res['output']
    n_tr = len(neural)
    tt = BIN_CENTERS

    # pick a trial that has a visible tongue and (if possible) photostimulation
    score = np.array([(outputs[t][3] < 3).sum() + 100 * (inputs[t][1].max() > 0)
                      for t in range(n_tr)])
    ti = int(np.argmax(score))

    fig, ax = plt.subplots(4, 2, figsize=(22, 18))

    # (0,0) raw spike raster + binned rates for the example trial ------------------
    a = ax[0, 0]
    with h5py.File(raw['fname'], 'r') as f:
        st_all = f['units/spike_times']
        si = raw['spike_index']
        nshow = min(25, len(raw['unit_idx']))
        sel = np.linspace(0, len(raw['unit_idx']) - 1, nshow).astype(int)
        for row, k in enumerate(sel):
            uu = raw['unit_idx'][k]
            lo = si[uu - 1] if uu > 0 else 0
            s = st_all[lo:si[uu]]
            s = s[(s >= raw['go'][ti] + OFF_START) & (s < raw['go'][ti] + OFF_END)] - raw['go'][ti]
            a.plot(s, np.full(len(s), row), '|', color='k', ms=4)
            a.step(tt, row + 0.8 * neural[ti][k] / max(1e-9, neural[ti][k].max()),
                   where='mid', color='tab:blue', lw=0.8)
    a.axvline(0, color='r', lw=1.5, label='go cue')
    a.axvline(raw['tone'][ti] - raw['go'][ti], color='g', lw=1.5, label='tone onset')
    a.axvline(raw['start_time'][ti] - raw['go'][ti], color='gray', ls='--', label='trial start')
    a.axvline(raw['stop_time'][ti] - raw['go'][ti], color='gray', ls=':', label='trial stop')
    a.set_xlim(OFF_START, OFF_END)
    a.set_title(f'{info["session_id"]} trial {ti}: raw spikes (ticks) vs binned rate (blue)')
    a.set_xlabel('time from go cue (s)'); a.set_ylabel('neuron'); a.legend(fontsize=7)

    # (0,1) neural heatmap ---------------------------------------------------------
    a = ax[0, 1]
    im = a.imshow(neural[ti], aspect='auto', origin='lower',
                  extent=[OFF_START, OFF_END, 0, neural[ti].shape[0]], cmap='magma')
    a.axvline(0, color='c', lw=1.5)
    a.set_title(f'binned firing rate (spikes/s), trial {ti}, n={neural[ti].shape[0]} neurons')
    a.set_xlabel('time from go cue (s)'); a.set_ylabel('neuron')
    fig.colorbar(im, ax=a)

    # (1,0) inputs for the example trial -------------------------------------------
    a = ax[1, 0]
    a.plot(tt, inputs[ti][0], '-o', ms=3, label='time from tone onset (s)')
    a.plot(tt, inputs[ti][1], '-s', ms=3, label='photostim on')
    a.axvline(0, color='r', lw=1.5)
    a.axvline(raw['tone'][ti] - raw['go'][ti], color='g', lw=1.5)
    a.axhline(0, color='k', lw=0.5)
    a.set_title('decoder inputs (green = tone onset, red = go cue); input 0 must cross 0 at green')
    a.set_xlabel('time from go cue (s)'); a.legend()

    # (1,1) photostim raster across trials -----------------------------------------
    a = ax[1, 1]
    ps = np.array([inputs[t][1] for t in range(n_tr)])
    a.imshow(ps, aspect='auto', origin='lower', extent=[OFF_START, OFF_END, 0, n_tr],
             cmap='Greys', interpolation='nearest')
    a.axvline(0, color='r', lw=1.5)
    a.set_title(f'photostim input, all trials ({int((ps.max(1) > 0).sum())} stim trials)')
    a.set_xlabel('time from go cue (s)'); a.set_ylabel('trial')

    # (2,0) tongue tracking, raw vs binned vs discretised --------------------------
    a = ax[2, 0]
    m = (raw['track_ts'] >= raw['go'][ti] + OFF_START) & (raw['track_ts'] < raw['go'][ti] + OFF_END)
    rel = raw['track_ts'][m] - raw['go'][ti]
    yy = raw['track_data'][m, 1].copy()
    vv = raw['track_data'][m, 2] > TONGUE_LIKELIHOOD_THRESH
    yy[~vv] = np.nan
    a.plot(rel, yy, '.', ms=3, color='0.5', label='raw visible frames (y)')
    yb = raw['ty'][ti].copy(); yb[~raw['tvis'][ti]] = np.nan
    a.step(tt, yb, where='mid', color='tab:blue', label='per-bin mean y')
    a.axhline(raw['pctl'][0], color='g', ls='--', label='session 40th pct')
    a.axhline(raw['pctl'][1], color='m', ls='--', label='session 60th pct')
    a.axvline(0, color='r', lw=1.5)
    a2 = a.twinx()
    a2.step(tt, outputs[ti][3], where='mid', color='tab:orange', label='class')
    a2.set_ylabel('tongue class (0/1/2/3)', color='tab:orange')
    a.set_title('tongue y-position: raw -> per-bin -> discretised')
    a.set_xlabel('time from go cue (s)'); a.set_ylabel('tongue y (px)'); a.legend(fontsize=7)

    # (2,1) outputs of the example trial -------------------------------------------
    a = ax[2, 1]
    for d, nm in enumerate(OUTPUT_NAMES):
        a.step(tt, outputs[ti][d] + 5 * d, where='mid', label=nm)
    a.axvline(0, color='r', lw=1.5)
    a.set_title(f'decoder outputs, trial {ti} '
                f'(outcome={raw["outcome"][ti]}, instruction={raw["instruction"][ti]})')
    a.set_xlabel('time from go cue (s)'); a.legend(fontsize=8)

    # (3,0) population PSTH split by choice ----------------------------------------
    a = ax[3, 0]
    ch = np.array([outputs[t][0][0] for t in range(n_tr)])
    for c, nm, col in [(0, 'lick left', 'tab:blue'), (1, 'lick right', 'tab:red'),
                       (2, 'no lick', 'tab:green')]:
        if (ch == c).sum() < 2:
            continue
        psth = np.mean([neural[t].mean(0) for t in np.where(ch == c)[0]], axis=0)
        a.plot(tt, psth, color=col, label=f'{nm} (n={(ch == c).sum()})')
    a.axvline(0, color='r', lw=1.5)
    a.axvline(-1.85, color='g', lw=1, ls='--')
    a.set_title('population mean firing rate by choice (alignment check)')
    a.set_xlabel('time from go cue (s)'); a.set_ylabel('spikes/s'); a.legend()

    # (3,1) tongue visibility & output class fractions over time --------------------
    a = ax[3, 1]
    a.plot(tt, raw['tvis'].mean(0), color='k', label='fraction of trials tongue visible')
    tc = np.array([outputs[t][3] for t in range(n_tr)])
    for c in range(4):
        a.plot(tt, (tc == c).mean(0), label=f'tongue class {c}')
    a.axvline(0, color='r', lw=1.5)
    a.set_title('tongue visibility / class fractions vs time (should rise after go cue)')
    a.set_xlabel('time from go cue (s)'); a.legend(fontsize=7)

    fig.suptitle(f'Processing checks — {info["session_id"]} '
                 f'({info["n_neurons"]} neurons, {n_tr} trials)', fontsize=15)
    fig.tight_layout()
    fig.savefig(fname_out, dpi=110)
    plt.close(fig)
    print(f'  wrote {fname_out}')


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def _worker(args):
    fname, want_raw = args
    try:
        return process_session(fname, want_raw=want_raw)
    except Exception as exc:        # keep the run alive, report loudly
        import traceback
        traceback.print_exc()
        return {'error': f'{fname}: {exc}'}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outfile', type=str)
    ap.add_argument('--full', action='store_true', help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='write processing_<session_id>.png for up to 2 sessions')
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
    print(f'found {len(files)} NWB files')
    if args.sample:
        files = files[:2]
        print(f'--sample: processing {len(files)} sessions')

    want_raw = [args.show_processing and i < 2 for i in range(len(files))]

    t_start = time.time()
    results = [None] * len(files)
    if args.workers > 1 and len(files) > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(min(args.workers, len(files))) as pool:
            for i, res in enumerate(pool.imap(_worker, list(zip(files, want_raw)))):
                results[i] = res
                _report(i, files, res, t_start)
    else:
        for i, fn in enumerate(files):
            results[i] = _worker((fn, want_raw[i]))
            _report(i, files, results[i], t_start)

    errors = [r['error'] for r in results if r is not None and 'error' in r]
    if errors:
        print('ERRORS:')
        for e in errors:
            print('  ', e)
        sys.exit(1)

    # ---- plots -------------------------------------------------------------------
    if args.show_processing:
        for r in results:
            if r is not None and r.get('raw') is not None:
                make_processing_plot(r, f'processing_{r["info"]["session_id"]}.png')

    # ---- assemble the dataset ----------------------------------------------------
    kept = [r for r in results if r is not None]
    dropped = [f for f, r in zip(files, results) if r is None]
    print(f'\nkept {len(kept)} sessions, dropped {len(dropped)}')
    for f in dropped:
        print('  dropped (no good units / <2 trials):', os.path.basename(f))

    subjects = sorted({r['info']['mouse'] for r in kept})
    sub_index = {s: i for i, s in enumerate(subjects)}

    data = {
        'neural': [r['neural'] for r in kept],
        'input': [r['input'] for r in kept],
        'output': [r['output'] for r in kept],
        'subjects': subjects,
        'subject_idx': np.array([sub_index[r['info']['mouse']] for r in kept], dtype=np.int64),
        'brain_regions': list(BRAIN_REGIONS),
        'brain_region_idx': [r['region_idx'] for r in kept],
        'input_names': list(INPUT_NAMES),
        'output_names': list(OUTPUT_NAMES),
        'output_values': [list(v) for v in OUTPUT_VALUES],
        'metadata': {
            'dataset': 'Mesoscale Activity Map (MAP), DANDI:000363 (Chen, Nguyen, Li, Svoboda 2023)',
            'references': [
                'Chen et al., Brain-wide neural activity underlying memory-guided movement, Cell 2024',
                'Wang, Kurgyis et al., Brain-wide analysis reveals movement encoding structured '
                'across and within brain areas, Nat Neurosci 2025',
            ],
            'task_description': (
                'Head-fixed mice perform an auditory delayed-response task. During the sample '
                'epoch (0.65 s) one of two pure tones (3 kHz / 12 kHz, 3 x 150 ms pulses) '
                'instructs the animal to lick left or right. After a delay epoch (usually 1.2 s) '
                'an auditory go cue (6 kHz, 0.1 s) releases the animal, which reports its choice '
                'by licking one of two lick ports during a 1.5 s answer period; a correct lick is '
                'rewarded with water. On ~20% of trials ALM was photoinhibited for 0.5 s during '
                'the delay epoch. Decoded outputs are the lick-direction choice (left/right/no '
                'lick), the trial outcome (ignore/miss/hit), whether the animal licked early '
                '(during the sample or delay epoch), and a per-session tertile-style '
                'discretisation of the tracked tongue y-position (plus a "not visible" class).'),
            'time_bin_size': BIN_SIZE * 1000.0,      # ms
            'temporal_alignment_event': 'onset of the auditory go cue (t = 0)',
            'off_start': OFF_START,
            'off_end': OFF_END,
            'n_time_bins': N_BINS,
            'bin_centers_s': BIN_CENTERS.tolist(),
            'neural_units': 'firing rate, spikes/s (spike count per 50 ms bin / 0.05 s)',
            'input_descriptions': {
                'time_from_tone_onset': (
                    'seconds elapsed since the onset of the instruction tone (sample-epoch start) '
                    'at each bin centre; negative before the tone. If an early lick triggered a '
                    'replay of the sample epoch, the last tone before the go cue is used.'),
                'photostim_on': (
                    '1 if the 0.5 s ALM photoinhibition stimulus overlapped the bin, else 0.'),
            },
            'output_descriptions': {
                'choice': 'lick direction reported after the go cue, from outcome x instruction',
                'outcome': 'ignore = no lick in the answer period, miss = incorrect lick, hit = correct lick',
                'early_lick': 'whether the animal licked during the sample or delay epoch',
                'tongue_y_position': (
                    'per-bin mean y-position of the DeepLabCut-tracked tongue (side camera), '
                    'discretised with the 40th/60th percentiles of the visible bins of that '
                    'session; class 3 = tongue not visible in the bin'),
            },
            'neuron_curation': (
                "units labelled 'good' by the region-specific spike-sorting QC classifiers "
                "(units/classification) and having a CCF histology annotation"),
            'trial_curation': (
                'trials covered by the ephys recording (units/obs_intervals and at least one '
                'spike in the window across all neurons), excluding auto-water and free-water '
                'trials. Early-lick, no-response and photostimulation trials are kept because '
                'they are decoder inputs/outputs.'),
            'session_curation': 'all sessions with at least one good unit and at least 2 trials',
            'known_limitations': (
                'The NWB export stores spikes only within [trial start, trial stop]. On error '
                '(miss) trials the recorded interval ends ~0.8 s after the go cue, so the last '
                'bins of those trials contain no spikes; 3% of trials likewise lack data before '
                '~-2.2 s. Those bins are reported as a firing rate of 0.'),
            'session_info': [r['info'] for r in kept],
        },
    }

    print(f'\nwriting {args.outfile} ...')
    t = time.time()
    with open(args.outfile, 'wb') as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  wrote {os.path.getsize(args.outfile) / 1e9:.2f} GB in {time.time() - t:.1f} s')

    _summary(data)
    print(f'\ntotal elapsed {time.time() - t_start:.1f} s')


def _report(i, files, res, t_start):
    el = time.time() - t_start
    if res is None:
        print(f'[{i + 1}/{len(files)}] {os.path.basename(files[i])}: SKIPPED '
              f'(no good units / <2 trials)  [{el:.0f}s]', flush=True)
    elif 'error' in res:
        print(f'[{i + 1}/{len(files)}] ERROR {res["error"]}  [{el:.0f}s]', flush=True)
    else:
        n = res['info']
        print(f'[{i + 1}/{len(files)}] {n["session_id"]}: {n["n_neurons"]} neurons, '
              f'{n["n_trials"]}/{n["n_trials_table"]} trials, '
              f'{n["mean_rate"]:.2f} spikes/s, {n["proc_time"]:.1f}s  [{el:.0f}s]', flush=True)


def _summary(data):
    nses = len(data['neural'])
    ntr = [len(s) for s in data['neural']]
    nn = [s[0].shape[0] for s in data['neural']]
    print('\n================ conversion summary ================')
    print(f'sessions            : {nses}')
    print(f'subjects            : {len(data["subjects"])}')
    print(f'trials (total)      : {sum(ntr)}  (mean {np.mean(ntr):.1f}, '
          f'range {min(ntr)}-{max(ntr)})')
    print(f'neurons (total)     : {sum(nn)}  (mean {np.mean(nn):.1f}, range {min(nn)}-{max(nn)})')
    print(f'time bins per trial : {data["neural"][0][0].shape[1]}')
    counts = np.zeros(len(data['brain_regions']), dtype=int)
    for idx in data['brain_region_idx']:
        counts += np.bincount(idx, minlength=len(counts))
    print('neurons per brain region:')
    for r, c in zip(data['brain_regions'], counts):
        print(f'   {r:18s} {c:6d}')
    # input ranges
    allin = np.concatenate([np.concatenate([t[:, None, :] for t in s], axis=1)
                            for s in data['input']], axis=1)
    for i, nm in enumerate(data['input_names']):
        print(f'input {i} ({nm}): [{allin[i].min():.3f}, {allin[i].max():.3f}]')
    # output distributions
    allout = np.concatenate([np.concatenate([t[:, None, :] for t in s], axis=1)
                             for s in data['output']], axis=1)
    for i, nm in enumerate(data['output_names']):
        vals, cnt = np.unique(allout[i], return_counts=True)
        frac = cnt / cnt.sum()
        s = ', '.join(f'{data["output_values"][i][int(v)]}={p:.4f}' for v, p in zip(vals, frac))
        print(f'output {i} ({nm}): {s}')
    print('====================================================')


if __name__ == '__main__':
    main()
