#!/usr/bin/env python3
"""
Convert MAP dataset (NWB format) to decoder-compatible pickle format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

import os
import sys
import time
import pickle
import argparse
import warnings
import numpy as np
from collections import OrderedDict

import pynwb
warnings.filterwarnings('ignore', category=UserWarning, module='hdmf')

# ============================================================
# Constants
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
BIN_WIDTH = 0.05  # 50 ms bins (decoder task spec)
ALIGN_EVENT = 'go_cue'
T_START = -2.5    # seconds before go cue
T_END = 1.5       # seconds after go cue
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80 bins

# Session selection criteria from paper
MIN_CORRECT_RATE = 0.65
MIN_CORRECT_LEFT = 50
MIN_CORRECT_RIGHT = 50

# Brain region mapping from detailed CCF annotations to broad categories
# Based on the paper's Figure 1J and reference code (preprocessing_DJ_2022Aug.py)
REGION_MAPPING = {
    # ALM = Secondary Motor Area in CCF
    'Secondary motor area': 'ALM',
    # Orbital areas
    'Orbital area': 'Orbital',
    'Agranular insular area': 'Orbital',  # grouped with orbital in paper
    # Primary motor / somatosensory / visual / retrosplenial / frontal pole = OtherCortex
    'Primary motor area': 'OtherCortex',
    'Primary somatosensory area': 'OtherCortex',
    'Primary visual area': 'OtherCortex',
    'Retrosplenial area': 'OtherCortex',
    'Frontal pole': 'OtherCortex',
    'Supplemental somatosensory area': 'OtherCortex',
    'Anterior cingulate area': 'OtherCortex',
    'Prelimbic area': 'OtherCortex',
    'Infralimbic area': 'OtherCortex',
    'Gustatory areas': 'OtherCortex',
    'Visceral area': 'OtherCortex',
    'Temporal association areas': 'OtherCortex',
    'Perirhinal area': 'OtherCortex',
    'Ectorhinal area': 'OtherCortex',
    'Entorhinal area': 'OtherCortex',
    'Posterior parietal association areas': 'OtherCortex',
    'Auditory areas': 'OtherCortex',
    # Striatum
    'Caudoputamen': 'Striatum',
    'Striatum': 'Striatum',
    'Fundus of striatum': 'Striatum',
    'Nucleus accumbens': 'Striatum',
    'Lateral septal nucleus': 'Striatum',
    # Pallidum
    'Pallidum': 'Pallidum',
    'Globus pallidus': 'Pallidum',
    'Substantia innominata': 'Pallidum',
    'Medial septal nucleus': 'Pallidum',
    'Diagonal band nucleus': 'Pallidum',
    'Bed nuclei of the stria terminalis': 'Pallidum',
    # Hippocampus
    'Field CA1': 'Hippocampus',
    'Field CA2': 'Hippocampus',
    'Field CA3': 'Hippocampus',
    'Subiculum': 'Hippocampus',
    'Postsubiculum': 'Hippocampus',
    'Presubiculum': 'Hippocampus',
    'Parasubiculum': 'Hippocampus',
    'Dentate gyrus': 'Hippocampus',
    'Fasciola cinerea': 'Hippocampus',
    'Induseum griseum': 'Hippocampus',
    # Thalamus (various nuclei)
    'Mediodorsal nucleus of thalamus': 'Thalamus',
    'Mediodorsal nucleus of the thalamus': 'Thalamus',
    'Submedial nucleus of the thalamus': 'Thalamus',
    'Paracentral nucleus': 'Thalamus',
    'Central lateral nucleus of the thalamus': 'Thalamus',
    'Central medial nucleus of the thalamus': 'Thalamus',
    'Ventral medial nucleus of the thalamus': 'Thalamus',
    'Ventral anterior-lateral complex of the thalamus': 'Thalamus',
    'Ventral posterolateral nucleus of the thalamus': 'Thalamus',
    'Ventral posteromedial nucleus of the thalamus': 'Thalamus',
    'Lateral dorsal nucleus of thalamus': 'Thalamus',
    'Lateral posterior nucleus of the thalamus': 'Thalamus',
    'Posterior complex of the thalamus': 'Thalamus',
    'Lateral habenula': 'Thalamus',
    'Medial habenula': 'Thalamus',
    'Reticular nucleus of the thalamus': 'Thalamus',
    'Thalamus': 'Thalamus',
    'Anterodorsal nucleus': 'Thalamus',
    'Anteroventral nucleus of thalamus': 'Thalamus',
    'Anteromedial nucleus': 'Thalamus',
    'Parafascicular nucleus': 'Thalamus',
    'Peripeduncular nucleus': 'Thalamus',
    'Posterior limiting nucleus of the thalamus': 'Thalamus',
    'Reuniens nucleus': 'Thalamus',
    'Rhomboid nucleus': 'Thalamus',
    'Nucleus of reuniens': 'Thalamus',
    'Paraventricular nucleus of the thalamus': 'Thalamus',
    'Intergeniculate leaflet of the lateral geniculate complex': 'Thalamus',
    'Medial geniculate complex': 'Thalamus',
    'Dorsal part of the lateral geniculate complex': 'Thalamus',
    'Ventral part of the lateral geniculate complex': 'Thalamus',
    # Hypothalamus
    'Hypothalamus': 'Hypothalamus',
    'Posterior hypothalamic nucleus': 'Hypothalamus',
    'Lateral hypothalamic area': 'Hypothalamus',
    'Zona incerta': 'Hypothalamus',
    'Subthalamic nucleus': 'Hypothalamus',
    'Mammillary body': 'Hypothalamus',
    'Supramammillary nucleus': 'Hypothalamus',
    'Tuberomammillary nucleus': 'Hypothalamus',
    'Lateral preoptic area': 'Hypothalamus',
    'Medial preoptic area': 'Hypothalamus',
    'Dorsomedial nucleus of the hypothalamus': 'Hypothalamus',
    'Ventromedial hypothalamic nucleus': 'Hypothalamus',
    'Paraventricular hypothalamic nucleus': 'Hypothalamus',
    # Midbrain
    'Midbrain reticular nucleus': 'Midbrain',
    'Red nucleus': 'Midbrain',
    'Superior colliculus': 'Midbrain',
    'Midbrain': 'Midbrain',
    'Substantia nigra': 'Midbrain',
    'Ventral tegmental area': 'Midbrain',
    'Interpeduncular nucleus': 'Midbrain',
    'Periaqueductal gray': 'Midbrain',
    'Pretectal region': 'Midbrain',
    'Anterior pretectal nucleus': 'Midbrain',
    'Inferior colliculus': 'Midbrain',
    'Pedunculopontine nucleus': 'Midbrain',
    'Cuneiform nucleus': 'Midbrain',
    'Dorsal nucleus raphe': 'Midbrain',
    'Central linear nucleus raphe': 'Midbrain',
    'Parabigeminal nucleus': 'Midbrain',
    'Nucleus of the posterior commissure': 'Midbrain',
    'Nucleus of Darkschewitsch': 'Midbrain',
    'Edinger-Westphal nucleus': 'Midbrain',
    'Oculomotor nucleus': 'Midbrain',
    'Trochlear nucleus': 'Midbrain',
    # Pons
    'Pontine reticular nucleus': 'Pons',
    'Pons': 'Pons',
    'Tegmental reticular nucleus': 'Pons',
    'Parabrachial nucleus': 'Pons',
    'Locus ceruleus': 'Pons',
    'Barrington\'s nucleus': 'Pons',
    'Pontine gray': 'Pons',
    'Pontine central gray': 'Pons',
    'Superior central nucleus raphe': 'Pons',
    'Nucleus raphe pontis': 'Pons',
    # Medulla
    'Medulla': 'Medulla',
    'Gigantocellular reticular nucleus': 'Medulla',
    'Paragigantocellular reticular nucleus': 'Medulla',
    'Magnocellular reticular nucleus': 'Medulla',
    'Parvicellular reticular nucleus': 'Medulla',
    'Intermediate reticular nucleus': 'Medulla',
    'Lateral reticular nucleus': 'Medulla',
    'Medullary reticular nucleus': 'Medulla',
    'Spinal nucleus of the trigeminal': 'Medulla',
    'Nucleus of the solitary tract': 'Medulla',
    'Dorsal motor nucleus of the vagus nerve': 'Medulla',
    'Hypoglossal nucleus': 'Medulla',
    'Nucleus ambiguus': 'Medulla',
    'Facial motor nucleus': 'Medulla',
    'Superior vestibular nucleus': 'Medulla',
    'Lateral vestibular nucleus': 'Medulla',
    'Medial vestibular nucleus': 'Medulla',
    'Spinal vestibular nucleus': 'Medulla',
    'Cochlear nuclei': 'Medulla',
    'Inferior olivary complex': 'Medulla',
    'Nucleus prepositus': 'Medulla',
    'Nucleus of the trapezoid body': 'Medulla',
    'Superior olivary complex': 'Medulla',
    'Cuneate nucleus': 'Medulla',
    'Gracile nucleus': 'Medulla',
    'External cuneate nucleus': 'Medulla',
    'Nucleus raphe magnus': 'Medulla',
    'Nucleus raphe pallidus': 'Medulla',
    'Nucleus raphe obscurus': 'Medulla',
    # Cerebellum
    'Cerebellum': 'Cerebellum',
    'Cerebellar cortex': 'Cerebellum',
    'Cerebellar nuclei': 'Cerebellum',
    'Purkinje layer': 'Cerebellum',
    'Molecular layer': 'Cerebellum',
    'Granular layer': 'Cerebellum',
    'Lobule': 'Cerebellum',
    'Vermal regions': 'Cerebellum',
    'Hemispheric regions': 'Cerebellum',
    'Ansiform lobule': 'Cerebellum',
    'Fastigial nucleus': 'Cerebellum',
    'Interposed nucleus': 'Cerebellum',
    'Dentate nucleus': 'Cerebellum',
    'Flocculus': 'Cerebellum',
    'Paraflocculus': 'Cerebellum',
    'Nodulus': 'Cerebellum',
    'Uvula': 'Cerebellum',
    'Lingula': 'Cerebellum',
    'Central lobule': 'Cerebellum',
    'Culmen': 'Cerebellum',
    'Declive': 'Cerebellum',
    'Folium-tuber vermis': 'Cerebellum',
    'Pyramus': 'Cerebellum',
    'Copula pyramidis': 'Cerebellum',
    'Simple lobule': 'Cerebellum',
    # Olfactory
    'Anterior olfactory nucleus': 'Olfactory',
    'Olfactory areas': 'Olfactory',
    'Main olfactory bulb': 'Olfactory',
    'Accessory olfactory bulb': 'Olfactory',
    'Piriform area': 'Olfactory',
    'Olfactory tubercle': 'Olfactory',
    'Taenia tecta': 'Olfactory',
    'Cortical amygdalar area': 'Olfactory',
    'Piriform-amygdalar area': 'Olfactory',
    'Postpiriform transition area': 'Olfactory',
    'Nucleus of the lateral olfactory tract': 'Olfactory',
    # Cortical Subplate
    'Claustrum': 'CorticalSubplate',
    'Endopiriform nucleus': 'CorticalSubplate',
    'Lateral amygdalar nucleus': 'CorticalSubplate',
    'Basolateral amygdalar nucleus': 'CorticalSubplate',
    'Basomedial amygdalar nucleus': 'CorticalSubplate',
    'Central amygdalar nucleus': 'CorticalSubplate',
    'Medial amygdalar nucleus': 'CorticalSubplate',
    'Intercalated amygdalar nucleus': 'CorticalSubplate',
    'Posterior amygdalar nucleus': 'CorticalSubplate',
}

def map_anno_to_region(anno_name):
    """Map a detailed CCF annotation to a broad brain region category."""
    if not anno_name or anno_name == '':
        return None
    # Try exact match first
    if anno_name in REGION_MAPPING:
        return REGION_MAPPING[anno_name]
    # Try prefix matching (handles layer suffixes like ", layer 5")
    for prefix, region in REGION_MAPPING.items():
        if anno_name.startswith(prefix):
            return REGION_MAPPING[prefix]
    # Try substring matching for complex names
    anno_lower = anno_name.lower()
    if 'thalamus' in anno_lower or 'thalamic' in anno_lower:
        return 'Thalamus'
    if 'hypothal' in anno_lower:
        return 'Hypothalamus'
    if 'cerebel' in anno_lower:
        return 'Cerebellum'
    if 'medulla' in anno_lower or 'medullar' in anno_lower:
        return 'Medulla'
    if 'midbrain' in anno_lower:
        return 'Midbrain'
    if 'hippocam' in anno_lower:
        return 'Hippocampus'
    if 'olfact' in anno_lower or 'piriform' in anno_lower:
        return 'Olfactory'
    if 'striatum' in anno_lower or 'putamen' in anno_lower:
        return 'Striatum'
    if 'pallid' in anno_lower or 'pallidum' in anno_lower:
        return 'Pallidum'
    if 'cortex' in anno_lower or 'cortical' in anno_lower:
        return 'OtherCortex'
    if 'amygdal' in anno_lower:
        return 'CorticalSubplate'
    if 'pons' in anno_lower or 'pontine' in anno_lower:
        return 'Pons'
    # Fallback
    print(f'  WARNING: Unmapped annotation: "{anno_name}"')
    return 'OtherCortex'


# ============================================================
# Data loading functions
# ============================================================

def list_nwb_files(data_dir):
    """List all NWB files organized by subject."""
    subjects = sorted([d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('sub-')])
    all_files = []
    for sub in subjects:
        sub_dir = os.path.join(data_dir, sub)
        nwb_files = sorted([os.path.join(sub_dir, f)
                           for f in os.listdir(sub_dir) if f.endswith('.nwb')])
        for nwb_file in nwb_files:
            all_files.append((sub, nwb_file))
    return all_files


def compute_firing_rates_vectorized(spike_times_list, go_time, t_start, t_end, bin_width, n_bins):
    """
    Compute firing rates for all neurons from spike times.

    Args:
        spike_times_list: list of arrays, one per neuron (absolute times)
        go_time: float, go cue time (absolute)
        t_start, t_end: window relative to go_time
        bin_width: bin width in seconds
        n_bins: number of bins

    Returns:
        fr: (n_neurons, n_bins) firing rate array
    """
    n_neurons = len(spike_times_list)
    fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
    bin_edges_start = go_time + t_start + np.arange(n_bins) * bin_width
    bin_edges_end = bin_edges_start + bin_width

    for i, spk in enumerate(spike_times_list):
        if len(spk) == 0:
            continue
        # Only keep spikes in our window
        mask = (spk >= go_time + t_start) & (spk < go_time + t_end)
        spk_window = spk[mask]
        if len(spk_window) == 0:
            continue
        # Assign spikes to bins
        bin_idx = np.floor((spk_window - (go_time + t_start)) / bin_width).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        np.add.at(fr[i], bin_idx, 1)

    fr /= bin_width  # convert to firing rate (Hz)
    return fr


def process_session(nwb_path, show_processing=False, session_idx=0):
    """
    Process one NWB session file.

    Returns:
        dict with session data, or None if session should be skipped.
    """
    t0 = time.time()

    io = pynwb.NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()

    # Extract subject info
    subject_id = nwb.subject.subject_id if nwb.subject else os.path.basename(nwb_path).split('_')[0]
    session_id = nwb.identifier

    print(f'  Processing {session_id}...')

    # ------ Extract trial info ------
    trials = nwb.trials
    n_trials = len(trials)
    trial_starts = trials['start_time'][:]
    trial_stops = trials['stop_time'][:]
    instructions = trials['trial_instruction'][:]  # 'left' or 'right'
    outcomes = trials['outcome'][:]  # 'hit', 'miss', 'ignore'
    early_licks = trials['early_lick'][:]  # 'early' or 'no early'
    auto_water = trials['auto_water'][:]  # 0 or 1
    free_water = trials['free_water'][:]  # 0 or 1
    photostim_onset = trials['photostim_onset'][:]  # time or 'N/A'
    photostim_duration = trials['photostim_duration'][:]  # duration or 'N/A'

    # ------ Get go cue times ------
    be = nwb.acquisition['BehavioralEvents']
    go_times = be.time_series['go_start_times'].timestamps[:]

    assert len(go_times) == n_trials, f"Go times ({len(go_times)}) != trials ({n_trials})"

    # ------ Get sample (tone) onset times for each trial ------
    sample_starts = be.time_series['sample_start_times'].timestamps[:]

    # Map sample_start to each trial: find the last sample_start before the go cue within each trial
    tone_onset_per_trial = np.full(n_trials, np.nan)
    for i in range(n_trials):
        # Find sample events within this trial's window
        in_trial = sample_starts[(sample_starts >= trial_starts[i]) & (sample_starts <= go_times[i])]
        if len(in_trial) > 0:
            # Take the LAST sample onset (in case of replays from early licking)
            tone_onset_per_trial[i] = in_trial[-1]

    # ------ Session selection criteria (applied to ALL behavioral trials) ------
    # These criteria use the full session behavioral data, per the paper
    behav_valid = (auto_water == 0) & (free_water == 0)
    # Paper's "regular trial" mask: exclude early lick, stim, auto/free water, AND ignore trials
    # Correct rate = hits / (hits + misses) on regular trials only
    regular_mask = behav_valid.copy()
    for i in range(n_trials):
        if photostim_onset[i] != 'N/A':
            regular_mask[i] = False
        if early_licks[i] == 'early':
            regular_mask[i] = False
    regular_mask &= (outcomes != 'ignore')  # exclude ignore from denominator

    n_regular = np.sum(regular_mask)
    if n_regular > 0:
        correct_regular = np.sum(regular_mask & (outcomes == 'hit'))
        correct_rate = correct_regular / n_regular
    else:
        correct_rate = 0.0

    correct_left = np.sum(regular_mask & (outcomes == 'hit') & (instructions == 'left'))
    correct_right = np.sum(regular_mask & (outcomes == 'hit') & (instructions == 'right'))

    if correct_rate < MIN_CORRECT_RATE:
        print(f'  SKIP: correct rate {correct_rate:.2f} < {MIN_CORRECT_RATE}')
        io.close()
        return None

    if correct_left < MIN_CORRECT_LEFT or correct_right < MIN_CORRECT_RIGHT:
        print(f'  SKIP: correct left={correct_left}, right={correct_right} (need >={MIN_CORRECT_LEFT} each)')
        io.close()
        return None

    # ------ Determine recording coverage ------
    units = nwb.units
    classification = units['classification'][:]
    good_mask_units = classification == 'good'
    anno_names = units['anno_name'][:]
    good_mask_units &= np.array([a != '' and a is not None for a in anno_names])
    good_indices_units = np.where(good_mask_units)[0]

    if len(good_indices_units) == 0:
        print(f'  SKIP: no good neurons')
        io.close()
        return None

    # Get max recording time from obs_intervals of a representative unit
    obs_intervals = units['obs_intervals'][good_indices_units[0]]
    max_recording_time = obs_intervals[-1, 1] if len(obs_intervals) > 0 else 0

    # ------ Trial filtering (behavioral + recording coverage) ------
    valid_mask = behav_valid.copy()
    valid_mask &= ~np.isnan(tone_onset_per_trial)

    # Exclude trials beyond the neural recording period
    for i in range(n_trials):
        trial_end_abs = go_times[i] + T_END
        if trial_end_abs > max_recording_time + 1.0:
            valid_mask[i] = False

    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 2:
        print(f'  SKIP: too few valid trials after recording filter ({len(valid_indices)})')
        io.close()
        return None

    # ------ Extract good neurons ------
    good_indices = good_indices_units
    spike_times_all = units['spike_times']
    spike_times_good = [spike_times_all[idx] for idx in good_indices]

    annos_good = anno_names[good_mask_units]
    neuron_regions = [map_anno_to_region(a) for a in annos_good]

    n_neurons = len(good_indices)
    print(f'  {n_neurons} good neurons, {len(valid_indices)} valid trials (of {n_trials}), correct rate={correct_rate:.2f}')

    # ------ Compute firing rates per trial ------
    neural_trials = []
    input_trials = []
    output_trials = []

    # Get tongue tracking data for this session
    bts = nwb.acquisition['BehavioralTimeSeries']
    has_tongue = 'Camera0_side_TongueTracking' in bts.time_series
    if has_tongue:
        tongue_data = bts.time_series['Camera0_side_TongueTracking'].data[:]  # (n_frames, 3): x, y, likelihood
        tongue_ts = bts.time_series['Camera0_side_TongueTracking'].timestamps[:]
        tongue_y = tongue_data[:, 1]
        tongue_likelihood = tongue_data[:, 2]

    # Compute per-session tongue y percentiles for discretization
    # Use all time points where tongue is visible (high likelihood)
    if has_tongue:
        tongue_visible = tongue_likelihood > 0.5
        if np.sum(tongue_visible) > 100:
            visible_y = tongue_y[tongue_visible]
            p40 = np.percentile(visible_y, 40)
            p60 = np.percentile(visible_y, 60)
        else:
            # Not enough visible tongue data
            p40 = np.percentile(tongue_y, 40)
            p60 = np.percentile(tongue_y, 60)

    # Time bin centers relative to go cue
    bin_centers = T_START + np.arange(N_BINS) * BIN_WIDTH + BIN_WIDTH / 2

    for trial_idx in valid_indices:
        go_time = go_times[trial_idx]

        # --- Neural data: firing rates ---
        fr = compute_firing_rates_vectorized(
            spike_times_good, go_time, T_START, T_END, BIN_WIDTH, N_BINS
        )  # (n_neurons, n_bins)
        neural_trials.append(fr)

        # --- Input 0: time from tone onset (seconds) ---
        tone_time = tone_onset_per_trial[trial_idx]
        tone_relative = tone_time - go_time  # tone onset relative to go cue (negative)
        time_from_tone = bin_centers - tone_relative  # time since tone onset at each bin

        # --- Input 1: photostimulation on/off ---
        photostim = np.zeros(N_BINS, dtype=np.float32)
        if photostim_onset[trial_idx] != 'N/A':
            ps_onset = float(photostim_onset[trial_idx])
            ps_duration = float(photostim_duration[trial_idx])
            # photostim_onset is relative to trial start, need to convert to absolute then go-relative
            # Actually check: is it absolute or relative?
            # From NWB: photostim_onset is in seconds - need to check
            # Let's check: photostim_start_times in BehavioralEvents has absolute times
            # But trials table photostim_onset may be relative to go cue or absolute
            # Let me compute relative to go cue
            # The trial table stores these as strings, and from exploring data they look like
            # they might be durations from trial start or absolute times
            # Looking at the data: photostim_onset values are small (1.8) suggesting relative to trial start
            ps_onset_abs = trial_starts[trial_idx] + ps_onset
            ps_end_abs = ps_onset_abs + ps_duration
            ps_onset_rel = ps_onset_abs - go_time
            ps_end_rel = ps_end_abs - go_time

            for b in range(N_BINS):
                bc = bin_centers[b]
                if ps_onset_rel <= bc < ps_end_rel:
                    photostim[b] = 1.0

        input_data = np.stack([time_from_tone.astype(np.float32), photostim], axis=0)  # (2, n_bins)
        input_trials.append(input_data)

        # --- Output 0: choice (left=0, right=1) ---
        instr = instructions[trial_idx]
        outcome = outcomes[trial_idx]
        if outcome == 'hit':
            choice = 0 if instr == 'left' else 1
        elif outcome == 'miss':
            choice = 1 if instr == 'left' else 0  # wrong lick = opposite
        else:  # ignore
            choice = 0 if instr == 'left' else 1  # assign instruction direction

        # --- Output 1: outcome (ignore=0, miss=1, hit=2) ---
        outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outcome]

        # --- Output 2: early lick (no=0, yes=1) ---
        early_val = 0 if early_licks[trial_idx] == 'no early' else 1

        # --- Output 3: tongue y position (discretized, time-varying) ---
        if has_tongue:
            tongue_y_trial = np.zeros(N_BINS, dtype=np.float32)
            for b in range(N_BINS):
                bc_abs = go_time + bin_centers[b]
                # Find closest tongue frame
                t_idx = np.searchsorted(tongue_ts, bc_abs)
                t_idx = min(t_idx, len(tongue_ts) - 1)
                ty = tongue_y[t_idx]
                if ty < p40:
                    tongue_y_trial[b] = 0
                elif ty < p60:
                    tongue_y_trial[b] = 1
                else:
                    tongue_y_trial[b] = 2
        else:
            tongue_y_trial = np.ones(N_BINS, dtype=np.float32)  # default to middle

        # Combine outputs: per-trial values + time-varying tongue
        # Use int values for categorical outputs (required by decoder)
        output_data = np.array([
            np.full(N_BINS, choice, dtype=np.int64),
            np.full(N_BINS, outcome_val, dtype=np.int64),
            np.full(N_BINS, early_val, dtype=np.int64),
            tongue_y_trial.astype(np.int64),
        ], dtype=np.int64)  # (4, n_bins)
        output_trials.append(output_data)

    io.close()

    elapsed = time.time() - t0
    print(f'  Done in {elapsed:.1f}s')

    return {
        'neural_trials': neural_trials,
        'input_trials': input_trials,
        'output_trials': output_trials,
        'neuron_regions': neuron_regions,
        'subject_id': subject_id,
        'session_id': session_id,
        'n_neurons': n_neurons,
        'n_trials': len(valid_indices),
        'correct_rate': correct_rate,
    }


def make_processing_plots(session_data, session_idx, nwb_path):
    """Generate processing visualization plots for a session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    neural = session_data['neural_trials']
    inputs = session_data['input_trials']
    outputs = session_data['output_trials']
    session_id = session_data['session_id']

    n_trials = len(neural)
    n_neurons = neural[0].shape[0] if n_trials > 0 else 0

    bin_centers = T_START + np.arange(N_BINS) * BIN_WIDTH + BIN_WIDTH / 2

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle(f'Session: {session_id}\n{n_neurons} neurons, {n_trials} trials', fontsize=14)

    # 1. Mean firing rate across neurons and trials
    if n_trials > 0 and n_neurons > 0:
        all_fr = np.stack(neural)  # (n_trials, n_neurons, n_bins)
        mean_fr = np.mean(all_fr, axis=(0, 1))
        axes[0, 0].plot(bin_centers, mean_fr)
        axes[0, 0].axvline(0, color='r', linestyle='--', label='Go cue')
        axes[0, 0].set_title('Mean firing rate')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('FR (Hz)')
        axes[0, 0].legend()

        # 2. Single trial neural activity (heatmap)
        trial_idx = min(5, n_trials - 1)
        im = axes[0, 1].imshow(neural[trial_idx][:min(50, n_neurons)],
                               aspect='auto', extent=[T_START, T_END, 0, min(50, n_neurons)])
        axes[0, 1].axvline(0, color='r', linestyle='--')
        axes[0, 1].set_title(f'Trial {trial_idx} neural activity')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Neuron')
        plt.colorbar(im, ax=axes[0, 1])

        # 3. FR distribution
        all_fr_flat = all_fr.flatten()
        axes[0, 2].hist(all_fr_flat[all_fr_flat > 0], bins=50, log=True)
        axes[0, 2].set_title('Firing rate distribution (>0)')
        axes[0, 2].set_xlabel('FR (Hz)')

    # 4. Input: time from tone onset
    for t in range(min(10, n_trials)):
        axes[1, 0].plot(bin_centers, inputs[t][0], alpha=0.3)
    axes[1, 0].axvline(0, color='r', linestyle='--')
    axes[1, 0].set_title('Input: time from tone onset')
    axes[1, 0].set_xlabel('Time (s)')

    # 5. Input: photostim
    stim_trials = [t for t in range(n_trials) if np.any(inputs[t][1] > 0)]
    for t in stim_trials[:10]:
        axes[1, 1].plot(bin_centers, inputs[t][1], alpha=0.5)
    axes[1, 1].set_title(f'Input: photostim ({len(stim_trials)} stim trials)')
    axes[1, 1].set_xlabel('Time (s)')

    # 6. Output distributions
    choices = [outputs[t][0, 0] for t in range(n_trials)]
    outcomes_arr = [outputs[t][1, 0] for t in range(n_trials)]
    early_arr = [outputs[t][2, 0] for t in range(n_trials)]

    axes[1, 2].bar(['left', 'right'], [choices.count(0), choices.count(1)])
    axes[1, 2].set_title('Choice distribution')

    axes[2, 0].bar(['ignore', 'miss', 'hit'],
                   [outcomes_arr.count(0), outcomes_arr.count(1), outcomes_arr.count(2)])
    axes[2, 0].set_title('Outcome distribution')

    axes[2, 1].bar(['no early', 'early'], [early_arr.count(0), early_arr.count(1)])
    axes[2, 1].set_title('Early lick distribution')

    # 7. Tongue y discretized
    if n_trials > 0:
        for t in range(min(5, n_trials)):
            axes[2, 2].plot(bin_centers, outputs[t][3], alpha=0.5)
        axes[2, 2].set_title('Output: tongue y (discretized)')
        axes[2, 2].set_xlabel('Time (s)')
        axes[2, 2].set_yticks([0, 1, 2])

    # 8. Brain region distribution
    from collections import Counter
    region_counts = Counter(session_data['neuron_regions'])
    regions_sorted = sorted(region_counts.items(), key=lambda x: -x[1])
    axes[3, 0].barh([r[0] for r in regions_sorted], [r[1] for r in regions_sorted])
    axes[3, 0].set_title('Neurons per brain region')

    # 9. Mean FR by trial type (hit vs miss)
    if n_trials > 0 and n_neurons > 0:
        hit_frs = [neural[t] for t in range(n_trials) if outputs[t][1, 0] == 2]
        miss_frs = [neural[t] for t in range(n_trials) if outputs[t][1, 0] == 1]
        if hit_frs:
            axes[3, 1].plot(bin_centers, np.mean(np.stack(hit_frs), axis=(0, 1)), label='hit')
        if miss_frs:
            axes[3, 1].plot(bin_centers, np.mean(np.stack(miss_frs), axis=(0, 1)), label='miss')
        axes[3, 1].axvline(0, color='r', linestyle='--')
        axes[3, 1].set_title('Mean FR by outcome')
        axes[3, 1].legend()

    axes[3, 2].axis('off')
    axes[3, 2].text(0.1, 0.5, f'Neurons: {n_neurons}\nTrials: {n_trials}\n'
                    f'Correct rate: {session_data["correct_rate"]:.2f}\n'
                    f'Stim trials: {len(stim_trials)}',
                    fontsize=12, transform=axes[3, 2].transAxes, va='center')

    plt.tight_layout()
    plt.savefig(f'processing_{session_id}.png', dpi=100)
    plt.close()


def build_dataset(session_results):
    """Build the final dataset dictionary from session results."""

    # Collect unique subjects and brain regions
    subjects = []
    subject_idx = []
    all_brain_regions = set()

    for sess in session_results:
        if sess['subject_id'] not in subjects:
            subjects.append(sess['subject_id'])
        subject_idx.append(subjects.index(sess['subject_id']))
        all_brain_regions.update(sess['neuron_regions'])

    # Remove None from brain regions
    all_brain_regions.discard(None)
    brain_regions = sorted(list(all_brain_regions))

    # Build neural, input, output lists
    neural = []
    inputs = []
    outputs = []
    brain_region_idx = []

    for sess in session_results:
        neural.append(sess['neural_trials'])
        inputs.append(sess['input_trials'])
        outputs.append(sess['output_trials'])

        # Map neuron regions to indices
        region_indices = np.array([
            brain_regions.index(r) if r in brain_regions else 0
            for r in sess['neuron_regions']
        ])
        brain_region_idx.append(region_indices)

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
        'subject_idx': np.array(subject_idx),
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_tone_onset', 'photostim_on'],
        'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y'],
        'output_values': [
            ['left', 'right'],           # choice
            ['ignore', 'miss', 'hit'],   # outcome
            ['no', 'yes'],               # early_lick
            ['low', 'mid', 'high'],      # tongue_y
        ],
        'metadata': {
            'task_description': 'Auditory delayed response task: mice lick left or right port based on tone frequency after a delay period',
            'time_bin_size': BIN_WIDTH * 1000,  # in ms
            'temporal_alignment_event': 'Go cue onset',
            'off_start': T_START,
            'off_end': T_END,
            'dataset': 'MAP (Mesoscale Activity Map) - DANDI:000363',
            'paper': 'Chen et al., 2024, Cell',
            'n_sessions': len(session_results),
            'n_subjects': len(subjects),
        }
    }

    return data


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Convert MAP NWB data to decoder format.')
    parser.add_argument('outfile', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true', help='Generate processing plots')
    args = parser.parse_args()

    if args.sample:
        args.full = False

    print('=' * 60)
    print('MAP Dataset Conversion')
    print(f'Output: {args.outfile}')
    print(f'Mode: {"sample (2 sessions)" if args.sample else "full"}')
    print(f'Show processing: {args.show_processing}')
    print('=' * 60)

    t_total = time.time()

    # List all NWB files
    all_files = list_nwb_files(DATA_DIR)
    print(f'Found {len(all_files)} NWB files from {len(set(s for s, _ in all_files))} subjects')

    if args.sample:
        # Pick 2 sessions from different subjects that are likely to pass selection
        # Skip first session per subject (often training), take second sessions from 2 subjects
        selected = []
        seen_subjects = set()
        for sub, fpath in all_files:
            if sub in seen_subjects:
                if len(selected) < 2:
                    selected.append((sub, fpath))
                    seen_subjects.add(sub + '_done')
            else:
                seen_subjects.add(sub)
        all_files = selected[:2]
        print(f'Sample mode: processing {len(all_files)} sessions')

    # Process sessions
    session_results = []
    n_skipped = 0

    for i, (subject, nwb_path) in enumerate(all_files):
        print(f'\n[{i+1}/{len(all_files)}] {os.path.basename(nwb_path)}')
        t_sess = time.time()

        try:
            result = process_session(nwb_path,
                                    show_processing=args.show_processing,
                                    session_idx=i)
        except Exception as e:
            print(f'  ERROR: {e}')
            import traceback
            traceback.print_exc()
            n_skipped += 1
            continue

        if result is None:
            n_skipped += 1
            continue

        session_results.append(result)

        if args.show_processing and len(session_results) <= 2:
            make_processing_plots(result, len(session_results) - 1, nwb_path)

        elapsed = time.time() - t_sess
        # Estimate remaining time
        avg_time = (time.time() - t_total) / (i + 1)
        remaining = avg_time * (len(all_files) - i - 1)
        print(f'  Elapsed: {elapsed:.1f}s, Est remaining: {remaining/60:.1f} min')

    print(f'\n{"=" * 60}')
    print(f'Processed: {len(session_results)} sessions, Skipped: {n_skipped}')

    if len(session_results) == 0:
        print('ERROR: No sessions processed!')
        sys.exit(1)

    # Build final dataset
    print('Building final dataset...')
    data = build_dataset(session_results)

    # Print summary statistics
    total_neurons = sum(s['n_neurons'] for s in session_results)
    total_trials = sum(s['n_trials'] for s in session_results)
    mean_trials = np.mean([s['n_trials'] for s in session_results])
    mean_neurons = np.mean([s['n_neurons'] for s in session_results])

    print(f'\nDataset Summary:')
    print(f'  Subjects: {len(data["subjects"])}')
    print(f'  Sessions: {len(session_results)}')
    print(f'  Total neurons: {total_neurons}')
    print(f'  Mean neurons/session: {mean_neurons:.1f}')
    print(f'  Total trials: {total_trials}')
    print(f'  Mean trials/session: {mean_trials:.1f}')
    print(f'  Brain regions: {data["brain_regions"]}')

    # Region neuron counts
    from collections import Counter
    all_region_counts = Counter()
    for sess in session_results:
        all_region_counts.update(sess['neuron_regions'])
    print(f'\n  Neurons per region:')
    for region in sorted(all_region_counts.keys()):
        print(f'    {region}: {all_region_counts[region]}')

    # Output distributions
    all_choices = []
    all_outcomes = []
    all_early = []
    for sess_outputs in data['output']:
        for trial_out in sess_outputs:
            all_choices.append(int(trial_out[0, 0]))
            all_outcomes.append(int(trial_out[1, 0]))
            all_early.append(int(trial_out[2, 0]))

    print(f'\n  Output distributions:')
    print(f'    Choice: left={all_choices.count(0)}, right={all_choices.count(1)}')
    print(f'    Outcome: ignore={all_outcomes.count(0)}, miss={all_outcomes.count(1)}, hit={all_outcomes.count(2)}')
    print(f'    Early lick: no={all_early.count(0)}, yes={all_early.count(1)}')

    # Save
    print(f'\nSaving to {args.outfile}...')
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    file_size = os.path.getsize(args.outfile)
    print(f'Saved: {file_size / 1e6:.1f} MB')

    total_time = time.time() - t_total
    print(f'\nTotal time: {total_time/60:.1f} minutes')


if __name__ == '__main__':
    main()
