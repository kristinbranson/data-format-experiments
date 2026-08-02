#!/usr/bin/env python3
"""
Convert MAP dataset (NWB format) to decoder-compatible pickle format.

Usage:
    python -u convert_data.py <outpicklefile> [--sample] [--full] [--show-processing]

References:
    - Chen et al. 2024, Cell: Brain-wide neural activity underlying memory-guided movement
    - Wang et al. 2025, Nature Neuroscience: Brain-wide analysis reveals movement encoding
"""

import os
import sys
import time
import glob
import pickle
import json
import argparse
import warnings
import numpy as np
from collections import OrderedDict

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
DATA_DIR = '/app/data'
BIN_WIDTH = 0.050  # 50 ms bins (decoder task spec)
ALIGN_START = -2.5  # seconds before go cue
ALIGN_END = 1.5     # seconds after go cue
N_BINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)  # 80 bins
TONGUE_CONFIDENCE_THRESHOLD = 0.9  # for occlusion detection
MIN_CORRECT_LEFT = 50   # session selection criterion
MIN_CORRECT_RIGHT = 50  # session selection criterion
MIN_PERFORMANCE = 0.65  # session selection criterion (65%)

# ============================================================
# Brain region mapping from CCF annotations to major regions
# Following the reference code's 14 major regions
# ============================================================

# Keywords for mapping CCF annotations to major brain regions
# Based on the Allen CCF hierarchy and reference code grouping
REGION_MAPPING = {
    'ALM': [
        'Secondary motor area',
        'Frontal pole',
    ],
    'OtherCortex': [
        'Primary motor area',
        'Primary somatosensory area',
        'Supplemental somatosensory area',
        'Retrosplenial area',
        'Agranular insular area',
        'Visceral area',
        'Gustatory area',
        'Anterior cingulate area',
        'Prelimbic area',
        'Infralimbic area',
        'Temporal association',
        'Perirhinal area',
        'Ectorhinal area',
        'Posterior parietal association',
        'Visual area',
        'Auditory area',
        'Entorhinal area',
        'Dorsal peduncular area',
    ],
    'Orbital': [
        'Orbital area',
    ],
    'Striatum': [
        'Caudoputamen',
        'Striatum',
        'Nucleus accumbens',
        'Fundus of striatum',
        'Lateral septal nucleus',
        'Septofimbrial nucleus',
    ],
    'Thalamus': [
        'thalamus',
        'Thalamus',
        'Zona incerta',
        'Reticular nucleus of the thalamus',
        'Lateral geniculate',
        'Medial geniculate',
        'Lateral habenula',
        'Medial habenula',
        'Habenular',
        'Paracentral nucleus',
        'Central lateral nucleus',
        'Central medial nucleus',
        'Ventral anterior-lateral complex',
        'Ventral medial nucleus',
        'Ventral posterolateral nucleus',
        'Ventral posteromedial nucleus',
        'Posterior limiting nucleus',
        'Laterodorsal nucleus',
        'Lateroposterior nucleus',
        'Mediodorsal nucleus',
        'Anteroventral nucleus',
        'Anteromedial nucleus',
        'Anterodorsal nucleus',
        'Parafascicular nucleus',
        'Reuniens',
        'Rhomboid nucleus',
        'Submedial nucleus',
        'Peripeduncular nucleus',
        'Posterior complex',
        'Pulvinar',
    ],
    'Hypothalamus': [
        'hypothalamic',
        'Hypothalamus',
        'Lateral hypothalamic',
        'Subthalamic nucleus',
        'Fields of Forel',
        'Mammillary',
        'Tuberal nucleus',
        'Dorsomedial nucleus of the hypothalamus',
        'Ventromedial hypothalamic',
        'Anterior hypothalamic',
        'Medial preoptic',
        'Lateral preoptic',
        'Periventricular hypothalamic',
        'Paraventricular hypothalamic',
        'Suprachiasmatic',
        'Supraoptic',
    ],
    'Midbrain': [
        'Midbrain',
        'midbrain',
        'Superior colliculus',
        'Inferior colliculus',
        'Substantia nigra',
        'Ventral tegmental',
        'Red nucleus',
        'Periaqueductal',
        'pretectal',
        'Pretectal',
        'Anterior pretectal',
        'Posterior pretectal',
        'Nucleus of Darkschewitsch',
        'Interstitial nucleus',
        'Edinger-Westphal',
        'Oculomotor',
        'Trochlear',
        'Pedunculopontine',
        'Cuneiform',
        'Interpeduncular',
        'Dorsal raphe',
        'Median raphe',
        'Parabigeminal',
        'Nucleus sagulum',
        'Midbrain reticular nucleus',
        'Nucleus of the brachium of the inferior colliculus',
    ],
    'Pons': [
        'Pons',
        'pons',
        'Pontine',
        'pontine',
        'Parabrachial',
        'Locus ceruleus',
        'Barrington',
        'Dorsal tegmental',
        'Laterodorsal tegmental',
        'Pontine central gray',
        'Pontine reticular',
        'Superior central nucleus raphe',
        'Tegmental reticular',
        'Motor nucleus of trigeminal',
        'Sensory nucleus of trigeminal',
        'Principal sensory nucleus',
        'Supratrigeminal',
        'Nucleus of the lateral lemniscus',
    ],
    'Medulla': [
        'Medulla',
        'medulla',
        'Nucleus of the solitary tract',
        'Dorsal motor nucleus',
        'Hypoglossal',
        'Facial motor',
        'Spinal nucleus of the trigeminal',
        'Cochlear',
        'Vestibular',
        'Inferior olive',
        'Lateral reticular',
        'Medullary reticular',
        'Gigantocellular',
        'Paragigantocellular',
        'Parvicellular reticular',
        'Intermediate reticular',
        'Raphe magnus',
        'Raphe obscurus',
        'Raphe pallidus',
        'Nucleus ambiguus',
        'Nucleus prepositus',
        'External cuneate',
        'Gracile nucleus',
        'Cuneate nucleus',
    ],
    'Cerebellum': [
        'Cerebellum',
        'cerebellum',
        'Cerebellar',
        'cerebellar',
        'Purkinje',
        'Fastigial',
        'Interposed',
        'Dentate nucleus',
        'Flocculus',
        'Paraflocculus',
        'Nodulus',
        'Uvula',
        'Lingula',
        'Culmen',
        'Declive',
        'Folium',
        'Tuber',
        'Pyramis',
        'Copula',
        'Simple lobule',
        'Ansiform lobule',
        'Paramedian lobule',
    ],
    'Hippocampus': [
        'Field CA',
        'Dentate gyrus',
        'Subiculum',
        'Presubiculum',
        'Parasubiculum',
        'Hippocampal',
        'hippocampal',
        'Prosubiculum',
        'Fasciola cinerea',
        'Indusium griseum',
    ],
    'Olfactory': [
        'Olfactory',
        'olfactory',
        'Piriform',
        'Taenia tecta',
        'Anterior olfactory',
    ],
    'CorticalSubplate': [
        'Claustrum',
        'Endopiriform',
        'Lateral amygdalar',
        'Basolateral amygdalar',
        'Basomedial amygdalar',
        'Central amygdalar',
        'Medial amygdalar',
        'Cortical amygdalar',
        'Intercalated amygdalar',
        'Posterior amygdalar',
        'Bed nuclei of the stria terminalis',
    ],
    'Pallidum': [
        'Pallidum',
        'pallidum',
        'Globus pallidus',
        'Substantia innominata',
        'Magnocellular',
        'Medial septal',
        'Diagonal band',
        'Triangular nucleus',
    ],
}


def map_anno_to_region(anno_name):
    """Map a CCF annotation name to a major brain region."""
    if not anno_name or anno_name.strip() == '':
        return None
    for region, keywords in REGION_MAPPING.items():
        for kw in keywords:
            if kw.lower() in anno_name.lower():
                return region
    return None  # unmapped


def get_nwb_files():
    """Get list of all NWB files sorted by subject then session."""
    return sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*_ses-*.nwb')))


def compute_session_performance(trials_data):
    """Compute session performance metrics for selection criteria.

    Following paper criteria:
    - Overall behavioral performance > 65% (fraction correct of control trials, excluding early lick)
    - At least 50 correct lick left and lick right trials each
    """
    outcomes = trials_data['outcome']
    instructions = trials_data['trial_instruction']
    early_lick = trials_data['early_lick']
    auto_water = trials_data['auto_water']
    free_water = trials_data['free_water']
    photostim_onset = trials_data['photostim_onset']

    # Control trials: no photostimulation, no auto water, no free water
    control_mask = np.ones(len(outcomes), dtype=bool)
    for i in range(len(outcomes)):
        if auto_water[i] == 1 or free_water[i] == 1:
            control_mask[i] = False
        if photostim_onset[i] != 'N/A':
            control_mask[i] = False

    # Exclude early lick trials for performance computation
    no_early_mask = np.array([el == 'no early' for el in early_lick])

    # Performance = fraction correct among control, non-early-lick, non-ignore trials
    eval_mask = control_mask & no_early_mask
    eval_outcomes = outcomes[eval_mask]
    eval_instructions = instructions[eval_mask]

    # Exclude ignore trials for performance computation
    non_ignore = eval_outcomes != 'ignore'
    eval_outcomes = eval_outcomes[non_ignore]
    eval_instructions = eval_instructions[non_ignore]

    if len(eval_outcomes) == 0:
        return 0.0, 0, 0

    n_correct = np.sum(eval_outcomes == 'hit')
    performance = n_correct / len(eval_outcomes)

    # Count correct left and right
    correct_left = np.sum((eval_outcomes == 'hit') & (eval_instructions == 'left'))
    correct_right = np.sum((eval_outcomes == 'hit') & (eval_instructions == 'right'))

    return performance, int(correct_left), int(correct_right)


def load_nwb_session(nwb_path):
    """Load all needed data from a single NWB file."""
    import pynwb

    io = pynwb.NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()

    # === Subject info ===
    subject_id = nwb.subject.subject_id
    subject_desc = nwb.subject.description  # e.g., "SC015"

    # === Units (neural data) ===
    units = nwb.units
    n_units = len(units)

    classifications = units['classification'][:]
    good_mask = classifications == 'good'
    anno_names = units['anno_name'][:]

    # Get spike times for good units
    # units['spike_times'] is a VectorIndex wrapping VectorData
    # .target.data contains all actual spike time values
    # .data contains cumulative end indices for each unit
    spike_times_vi = units['spike_times']  # VectorIndex
    all_spike_times = np.array(spike_times_vi.target.data[:])  # actual spike times
    all_st_idx = np.array(spike_times_vi.data[:])  # end indices per unit

    # Build spike times list for good units
    good_indices = np.where(good_mask)[0]
    good_spike_times = []

    for ui in good_indices:
        start_idx = 0 if ui == 0 else int(all_st_idx[ui - 1])
        end_idx = int(all_st_idx[ui])
        good_spike_times.append(all_spike_times[start_idx:end_idx])

    good_anno_names = anno_names[good_mask]

    # === Trials ===
    trials = nwb.trials
    n_trials = len(trials)
    trials_data = {
        'start_time': trials['start_time'][:],
        'stop_time': trials['stop_time'][:],
        'trial_instruction': trials['trial_instruction'][:],
        'outcome': trials['outcome'][:],
        'early_lick': trials['early_lick'][:],
        'auto_water': trials['auto_water'][:],
        'free_water': trials['free_water'][:],
        'photostim_onset': trials['photostim_onset'][:],
        'photostim_power': trials['photostim_power'][:],
        'photostim_duration': trials['photostim_duration'][:],
    }

    # === Behavioral events ===
    be = nwb.acquisition['BehavioralEvents']
    go_times = be.time_series['go_start_times'].timestamps[:]

    # Sample start times (may have more entries than trials due to early lick replays)
    sample_start_ts = be.time_series['sample_start_times'].timestamps[:]

    # Photostim events
    photostim_start_ts = be.time_series['photostim_start_times'].timestamps[:]
    photostim_stop_ts = be.time_series['photostim_stop_times'].timestamps[:]

    # Lick times
    left_lick_ts = be.time_series['left_lick_times'].timestamps[:]
    right_lick_ts = be.time_series['right_lick_times'].timestamps[:]

    # === Tongue tracking ===
    bts = nwb.acquisition['BehavioralTimeSeries']
    tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
    tongue_data = tongue_ts_obj.data[:]  # (n_frames, 3): x, y, confidence
    tongue_timestamps = tongue_ts_obj.timestamps[:]

    io.close()

    return {
        'subject_id': subject_id,
        'subject_desc': subject_desc,
        'nwb_path': nwb_path,
        'good_spike_times': good_spike_times,
        'good_anno_names': good_anno_names,
        'good_indices': good_indices,
        'n_total_units': n_units,
        'trials_data': trials_data,
        'n_trials': n_trials,
        'go_times': go_times,
        'sample_start_ts': sample_start_ts,
        'photostim_start_ts': photostim_start_ts,
        'photostim_stop_ts': photostim_stop_ts,
        'left_lick_ts': left_lick_ts,
        'right_lick_ts': right_lick_ts,
        'tongue_data': tongue_data,
        'tongue_timestamps': tongue_timestamps,
    }


def get_tone_onset_for_trials(go_times, sample_start_ts, trial_starts, trial_stops):
    """Get the last sample start time before go cue for each trial.

    The sample epoch may replay due to early licks, so we take the LAST
    sample start event within each trial's time range (before go cue).
    """
    n_trials = len(go_times)
    tone_onsets = np.full(n_trials, np.nan)

    for i in range(n_trials):
        # Find sample starts within this trial, before the go cue
        mask = (sample_start_ts >= trial_starts[i]) & (sample_start_ts <= go_times[i])
        matching = sample_start_ts[mask]
        if len(matching) > 0:
            tone_onsets[i] = matching[-1]  # last sample start

    return tone_onsets


def bin_spikes(spike_times_list, go_times, align_start, align_end, bin_width, n_bins):
    """Bin spike times into firing rates for each trial.

    Vectorized implementation: processes all trials for each neuron at once.

    Args:
        spike_times_list: list of arrays, one per neuron (absolute session times)
        go_times: array of go cue times (one per trial)
        align_start: time before go cue (negative)
        align_end: time after go cue (positive)
        bin_width: bin width in seconds
        n_bins: number of bins

    Returns:
        list of (n_neurons, n_bins) arrays, one per trial
    """
    n_neurons = len(spike_times_list)
    n_trials = len(go_times)

    # Precompute bin edges relative to alignment
    bin_edges = np.linspace(align_start, align_end, n_bins + 1)

    # Pre-allocate all trial matrices at once: (n_trials, n_neurons, n_bins)
    all_matrices = np.zeros((n_trials, n_neurons, n_bins), dtype=np.float32)

    for n in range(n_neurons):
        st = spike_times_list[n]
        if len(st) == 0:
            continue

        # For each go time, compute relative spike times and bin them
        # Vectorize across trials using searchsorted
        for t in range(n_trials):
            go_t = go_times[t]
            # Get spikes in window using searchsorted (binary search)
            abs_start = go_t + align_start
            abs_end = go_t + align_end
            idx_lo = np.searchsorted(st, abs_start, side='left')
            idx_hi = np.searchsorted(st, abs_end, side='left')

            if idx_hi > idx_lo:
                rel_spikes = st[idx_lo:idx_hi] - go_t
                counts, _ = np.histogram(rel_spikes, bins=bin_edges)
                all_matrices[t, n, :] = counts / bin_width

    # Convert to list of (n_neurons, n_bins) per trial
    return [all_matrices[t] for t in range(n_trials)]


def compute_tongue_y_per_trial(tongue_data, tongue_timestamps, go_times,
                                align_start, align_end, bin_width, n_bins,
                                confidence_threshold=TONGUE_CONFIDENCE_THRESHOLD):
    """Compute tongue y-position time series for each trial, binned.

    When tongue is occluded (confidence below threshold), set to session mean.
    Reference: "when the tongue was occluded... we set the tongue position to its mean value"

    Vectorized implementation for efficiency.
    """
    tongue_y = tongue_data[:, 1].astype(np.float64)
    tongue_conf = tongue_data[:, 2].astype(np.float64)

    # Compute session mean of tongue y when visible
    visible_mask = tongue_conf >= confidence_threshold
    if np.sum(visible_mask) > 0:
        session_mean_y = np.mean(tongue_y[visible_mask])
    else:
        session_mean_y = np.mean(tongue_y)

    # Replace non-visible tongue y with session mean (vectorized)
    tongue_y_imputed = tongue_y.copy()
    tongue_y_imputed[~visible_mask] = session_mean_y

    n_trials = len(go_times)
    bin_edges = np.linspace(align_start, align_end, n_bins + 1)

    tongue_y_trials = []
    for t in range(n_trials):
        go_t = go_times[t]

        # Get tongue data in the trial window
        window_start = go_t + align_start
        window_end = go_t + align_end
        mask = (tongue_timestamps >= window_start) & (tongue_timestamps < window_end)
        trial_ts = tongue_timestamps[mask] - go_t  # relative to go cue
        trial_y = tongue_y_imputed[mask]

        if len(trial_ts) == 0:
            tongue_y_trials.append(np.full(n_bins, session_mean_y, dtype=np.float32))
            continue

        # Bin the tongue y values using digitize (vectorized)
        bin_indices = np.digitize(trial_ts, bin_edges) - 1  # 0-indexed bins
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        trial_tongue_y = np.full(n_bins, session_mean_y, dtype=np.float64)
        # Use bincount for mean computation
        for b in range(n_bins):
            in_bin = trial_y[bin_indices == b]
            if len(in_bin) > 0:
                trial_tongue_y[b] = np.mean(in_bin)

        tongue_y_trials.append(trial_tongue_y.astype(np.float32))

    return tongue_y_trials, session_mean_y


def discretize_tongue_y(tongue_y_trials, session_mean_y):
    """Discretize tongue y-position per session.

    0: < 40th percentile of y-position over the session
    1: 40th to 60th percentile of y-position over the session
    2: > 60th percentile of y-position over the session

    Percentiles are computed over ALL time bins (including imputed values).
    """
    # Collect all tongue y values across all trials and time bins
    all_values = np.concatenate([t for t in tongue_y_trials])

    p40 = np.percentile(all_values, 40)
    p60 = np.percentile(all_values, 60)

    # Ensure thresholds differ to get 3 categories
    # If p40 == p60 (common when tongue is mostly at mean), adjust slightly
    if np.isclose(p40, p60):
        # Use small offset to create 3 categories
        eps = max(1e-6, abs(p40) * 1e-4)
        p40 = p40 - eps
        p60 = p60 + eps

    discretized = []
    for trial_y in tongue_y_trials:
        d = np.zeros(len(trial_y), dtype=np.int64)
        d[trial_y >= p40] = 1
        d[trial_y >= p60] = 2
        discretized.append(d)

    return discretized


def compute_photostim_input(go_times, photostim_start_ts, photostim_stop_ts,
                            align_start, align_end, bin_width, n_bins):
    """Compute binary photostimulation time series for each trial.

    1 when photostim is active, 0 otherwise.
    """
    n_trials = len(go_times)
    bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)

    photostim_trials = []
    for t in range(n_trials):
        go_t = go_times[t]
        ps = np.zeros(n_bins, dtype=np.float32)

        for si in range(len(photostim_start_ts)):
            ps_start = photostim_start_ts[si] - go_t
            ps_stop = photostim_stop_ts[si] - go_t

            # Check if this photostim event overlaps with our window
            if ps_stop < align_start or ps_start > align_end:
                continue

            # Mark bins where photostim is active
            for b in range(n_bins):
                bc = bin_centers[b]
                if bc >= ps_start and bc < ps_stop:
                    ps[b] = 1.0

        photostim_trials.append(ps)

    return photostim_trials


def compute_tone_onset_input(go_times, tone_onsets, align_start, align_end, bin_width, n_bins):
    """Compute time from tone onset for each trial and time bin.

    This is a continuous, time-varying input: time (in seconds) since the
    tone (sample stimulus) began.
    """
    n_trials = len(go_times)
    bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)

    tone_onset_input = []
    for t in range(n_trials):
        go_t = go_times[t]
        tone_t = tone_onsets[t]

        if np.isnan(tone_t):
            # If no tone onset found, set to 0 (shouldn't happen for valid trials)
            tone_input = np.zeros(n_bins, dtype=np.float32)
        else:
            # Time from tone onset = (absolute time of bin center) - tone_onset
            # = (go_t + bin_center) - tone_t
            # = bin_center - (tone_t - go_t)
            tone_rel = tone_t - go_t  # tone onset relative to go cue (negative)
            tone_input = (bin_centers - tone_rel).astype(np.float32)

        tone_onset_input.append(tone_input)

    return tone_onset_input


def process_session(nwb_path, show_processing=False, session_idx=0):
    """Process a single NWB session into decoder format.

    Returns None if session doesn't meet selection criteria.
    """
    t0 = time.time()
    sess_name = os.path.basename(nwb_path).replace('.nwb', '')
    print(f'  Loading {sess_name}...')

    data = load_nwb_session(nwb_path)
    t_load = time.time() - t0

    # === Session selection criteria ===
    performance, correct_left, correct_right = compute_session_performance(data['trials_data'])
    print(f'    Performance: {performance:.1%}, Correct L/R: {correct_left}/{correct_right}')

    if performance < MIN_PERFORMANCE:
        print(f'    SKIP: performance {performance:.1%} < {MIN_PERFORMANCE:.0%}')
        return None
    if correct_left < MIN_CORRECT_LEFT:
        print(f'    SKIP: correct left {correct_left} < {MIN_CORRECT_LEFT}')
        return None
    if correct_right < MIN_CORRECT_RIGHT:
        print(f'    SKIP: correct right {correct_right} < {MIN_CORRECT_RIGHT}')
        return None

    # === Trial filtering: exclude auto_water and free_water ===
    td = data['trials_data']
    trial_mask = np.ones(data['n_trials'], dtype=bool)
    trial_mask[td['auto_water'] == 1] = False
    trial_mask[td['free_water'] == 1] = False

    # Get trial indices
    trial_indices = np.where(trial_mask)[0]
    n_valid_trials = len(trial_indices)

    if n_valid_trials < 2:
        print(f'    SKIP: only {n_valid_trials} valid trials')
        return None

    go_times = data['go_times'][trial_indices]

    # === Neuron filtering: only good units with valid brain region ===
    good_anno = data['good_anno_names']
    region_labels = []
    neuron_mask = []
    for i, anno in enumerate(good_anno):
        region = map_anno_to_region(anno)
        if region is not None:
            region_labels.append(region)
            neuron_mask.append(i)

    neuron_mask = np.array(neuron_mask)
    n_neurons = len(neuron_mask)

    if n_neurons < 1:
        print(f'    SKIP: no neurons with valid brain region')
        return None

    spike_times_list = [data['good_spike_times'][i] for i in neuron_mask]

    # === Exclude trials beyond recording range ===
    # Some sessions have behavioral trials before recording starts or after it ends.
    # Find the min/max spike time across all selected neurons.
    max_spike_time = max(
        (st[-1] if len(st) > 0 else 0.0) for st in spike_times_list
    )
    min_spike_time = min(
        (st[0] if len(st) > 0 else float('inf')) for st in spike_times_list
    )
    # A trial needs neural data in [go_time + ALIGN_START, go_time + ALIGN_END].
    # Exclude trials where the window doesn't overlap with recorded data.
    recording_mask = (go_times + ALIGN_START <= max_spike_time) & \
                     (go_times + ALIGN_END >= min_spike_time)
    if not np.all(recording_mask):
        n_dropped = np.sum(~recording_mask)
        print(f'    Excluding {n_dropped} trials beyond recording range (max spike: {max_spike_time:.1f}s)')
        # Update trial_indices and go_times
        valid_positions = np.where(recording_mask)[0]
        trial_indices = trial_indices[valid_positions]
        go_times = go_times[valid_positions]
        n_valid_trials = len(trial_indices)
        if n_valid_trials < 2:
            print(f'    SKIP: only {n_valid_trials} valid trials after recording filter')
            return None

    print(f'    Neurons: {n_neurons}, Trials: {n_valid_trials} (loaded in {t_load:.1f}s)')

    # === Bin spikes into firing rates ===
    t1 = time.time()
    neural_trials = bin_spikes(spike_times_list, go_times, ALIGN_START, ALIGN_END, BIN_WIDTH, N_BINS)
    t_bin = time.time() - t1
    print(f'    Spike binning: {t_bin:.1f}s')

    # === Compute inputs ===
    # 1. Time from tone onset
    tone_onsets = get_tone_onset_for_trials(
        go_times, data['sample_start_ts'],
        td['start_time'][trial_indices], td['stop_time'][trial_indices]
    )
    tone_onset_input = compute_tone_onset_input(go_times, tone_onsets, ALIGN_START, ALIGN_END, BIN_WIDTH, N_BINS)

    # 2. Photostimulation
    photostim_input = compute_photostim_input(go_times, data['photostim_start_ts'],
                                               data['photostim_stop_ts'],
                                               ALIGN_START, ALIGN_END, BIN_WIDTH, N_BINS)

    # Combine inputs: (2, n_bins)
    input_trials = []
    for t in range(n_valid_trials):
        inp = np.stack([tone_onset_input[t], photostim_input[t]], axis=0).astype(np.float32)
        input_trials.append(inp)

    # === Compute outputs ===
    # 1. Choice: left=0, right=1
    instructions = td['trial_instruction'][trial_indices]
    choices = np.array([0 if ins == 'left' else 1 for ins in instructions], dtype=np.int64)

    # 2. Outcome: ignore=0, miss=1, hit=2
    outcomes_raw = td['outcome'][trial_indices]
    outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
    outcomes = np.array([outcome_map.get(o, 0) for o in outcomes_raw], dtype=np.int64)

    # 3. Early lick: no=0, yes=1
    early_lick_raw = td['early_lick'][trial_indices]
    early_licks = np.array([1 if el == 'early' else 0 for el in early_lick_raw], dtype=np.int64)

    # 4. Tongue y-position (time-varying, discretized)
    t2 = time.time()
    tongue_y_trials, session_mean_y = compute_tongue_y_per_trial(
        data['tongue_data'], data['tongue_timestamps'], go_times,
        ALIGN_START, ALIGN_END, BIN_WIDTH, N_BINS
    )
    tongue_y_discrete = discretize_tongue_y(tongue_y_trials, session_mean_y)
    t_tongue = time.time() - t2
    print(f'    Tongue processing: {t_tongue:.1f}s')

    # Combine outputs: per-trial for first 3, time-varying for tongue
    output_trials = []
    for t in range(n_valid_trials):
        out = np.array([
            np.full(N_BINS, choices[t], dtype=np.int64),         # choice (per-trial, broadcast)
            np.full(N_BINS, outcomes[t], dtype=np.int64),        # outcome (per-trial, broadcast)
            np.full(N_BINS, early_licks[t], dtype=np.int64),     # early lick (per-trial, broadcast)
            tongue_y_discrete[t].astype(np.int64),                # tongue y (time-varying)
        ], dtype=np.int64)
        output_trials.append(out)

    # === Plotting ===
    if show_processing:
        plot_processing(sess_name, neural_trials, input_trials, output_trials,
                       go_times, tone_onsets, tongue_y_trials, tongue_y_discrete,
                       region_labels, session_idx)

    total_time = time.time() - t0
    print(f'    Total: {total_time:.1f}s')

    return {
        'neural': neural_trials,          # list of (n_neurons, n_bins)
        'input': input_trials,            # list of (2, n_bins)
        'output': output_trials,          # list of (4, n_bins)
        'subject_id': data['subject_id'],
        'subject_desc': data['subject_desc'],
        'region_labels': region_labels,   # list of str, length n_neurons
        'n_neurons': n_neurons,
        'n_trials': n_valid_trials,
        'performance': performance,
        'correct_left': correct_left,
        'correct_right': correct_right,
        'nwb_path': nwb_path,
    }


def plot_processing(sess_name, neural_trials, input_trials, output_trials,
                   go_times, tone_onsets, tongue_y_raw, tongue_y_discrete,
                   region_labels, session_idx):
    """Plot visualizations of processing steps."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    bin_centers = np.linspace(ALIGN_START + BIN_WIDTH/2, ALIGN_END - BIN_WIDTH/2, N_BINS)

    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    fig.suptitle(f'Processing: {sess_name}', fontsize=14)

    # Pick example trial
    n_trials = len(neural_trials)
    trial_idx = min(5, n_trials - 1)

    # 1. Neural activity heatmap for example trial
    ax = axes[0, 0]
    neural = neural_trials[trial_idx]
    n_show = min(50, neural.shape[0])
    ax.imshow(neural[:n_show], aspect='auto', extent=[ALIGN_START, ALIGN_END, n_show, 0])
    ax.axvline(0, color='r', linestyle='--', label='Go cue')
    ax.set_title(f'Neural (trial {trial_idx}, first {n_show} neurons)')
    ax.set_xlabel('Time from go cue (s)')
    ax.set_ylabel('Neuron')
    ax.legend()

    # 2. Mean firing rate across neurons
    ax = axes[0, 1]
    mean_fr = np.mean([np.mean(t, axis=0) for t in neural_trials], axis=0)
    ax.plot(bin_centers, mean_fr)
    ax.axvline(0, color='r', linestyle='--', label='Go cue')
    ax.set_title('Mean firing rate (all trials, all neurons)')
    ax.set_xlabel('Time from go cue (s)')
    ax.set_ylabel('Firing rate (Hz)')
    ax.legend()

    # 3. Input: time from tone onset
    ax = axes[1, 0]
    for t in range(min(10, n_trials)):
        ax.plot(bin_centers, input_trials[t][0], alpha=0.5)
    ax.axvline(0, color='r', linestyle='--')
    ax.set_title('Input: Time from tone onset')
    ax.set_xlabel('Time from go cue (s)')
    ax.set_ylabel('Time from tone (s)')

    # 4. Input: photostimulation
    ax = axes[1, 1]
    ps_any = np.any([input_trials[t][1] for t in range(n_trials)], axis=0)
    n_stim = sum(1 for t in range(n_trials) if np.any(input_trials[t][1] > 0))
    ax.plot(bin_centers, np.mean([input_trials[t][1] for t in range(n_trials)], axis=0))
    ax.axvline(0, color='r', linestyle='--')
    ax.set_title(f'Input: Photostim (mean, {n_stim}/{n_trials} stim trials)')
    ax.set_xlabel('Time from go cue (s)')
    ax.set_ylabel('P(stim on)')

    # 5. Output: choice distribution
    ax = axes[2, 0]
    choices = [output_trials[t][0, 0] for t in range(n_trials)]
    ax.bar(['Left (0)', 'Right (1)'], [choices.count(0), choices.count(1)])
    ax.set_title('Output: Choice distribution')

    # 6. Output: outcome distribution
    ax = axes[2, 1]
    outcomes = [int(output_trials[t][1, 0]) for t in range(n_trials)]
    labels = ['Ignore (0)', 'Miss (1)', 'Hit (2)']
    ax.bar(labels, [outcomes.count(0), outcomes.count(1), outcomes.count(2)])
    ax.set_title('Output: Outcome distribution')

    # 7. Output: early lick distribution
    ax = axes[3, 0]
    early = [int(output_trials[t][2, 0]) for t in range(n_trials)]
    ax.bar(['No (0)', 'Yes (1)'], [early.count(0), early.count(1)])
    ax.set_title('Output: Early lick distribution')

    # 8. Output: tongue y example trial
    ax = axes[3, 1]
    ax.plot(bin_centers, tongue_y_raw[trial_idx], label='Raw y')
    ax.plot(bin_centers, tongue_y_discrete[trial_idx], label='Discretized', linestyle='--')
    ax.axvline(0, color='r', linestyle='--')
    ax.set_title(f'Output: Tongue y (trial {trial_idx})')
    ax.set_xlabel('Time from go cue (s)')
    ax.legend()

    # 9. Brain region distribution
    ax = axes[4, 0]
    from collections import Counter
    rc = Counter(region_labels)
    regions_sorted = sorted(rc.keys(), key=lambda x: -rc[x])
    ax.barh(range(len(regions_sorted)), [rc[r] for r in regions_sorted])
    ax.set_yticks(range(len(regions_sorted)))
    ax.set_yticklabels(regions_sorted)
    ax.set_title(f'Brain regions ({len(region_labels)} neurons)')
    ax.set_xlabel('Count')

    # 10. Temporal alignment check
    ax = axes[4, 1]
    # Check that tone onset - go cue is roughly consistent
    tone_rel = tone_onsets - go_times
    tone_rel = tone_rel[~np.isnan(tone_rel)]
    if len(tone_rel) > 0:
        ax.hist(tone_rel, bins=30)
        ax.axvline(np.median(tone_rel), color='r', linestyle='--',
                   label=f'Median: {np.median(tone_rel):.3f}s')
        ax.set_title('Tone onset relative to go cue')
        ax.set_xlabel('Time (s)')
        ax.legend()

    fig.tight_layout()
    fig.savefig(f'processing_{session_idx}.png', dpi=100)
    plt.close(fig)
    print(f'    Saved processing_{session_idx}.png')


def convert_all(outfile, sample=False, show_processing=False):
    """Main conversion function."""
    t_start = time.time()

    nwb_files = get_nwb_files()
    print(f'Found {len(nwb_files)} NWB files')

    if sample:
        # Pick 2 sessions from different subjects (middle of dataset for better chance of passing criteria)
        idx1 = len(nwb_files) // 3
        idx2 = 2 * len(nwb_files) // 3
        nwb_files = [nwb_files[idx1], nwb_files[idx2]]
        print(f'Sample mode: processing {len(nwb_files)} sessions')

    # Process sessions
    all_sessions = []
    subjects_set = OrderedDict()
    all_regions_set = set()

    for i, nwb_path in enumerate(nwb_files):
        print(f'\n[{i+1}/{len(nwb_files)}] Processing {os.path.basename(nwb_path)}')

        result = process_session(
            nwb_path,
            show_processing=show_processing and i < 2,
            session_idx=i
        )

        if result is None:
            continue

        all_sessions.append(result)
        sid = result['subject_id']
        if sid not in subjects_set:
            subjects_set[sid] = len(subjects_set)

        for r in result['region_labels']:
            all_regions_set.add(r)

    n_sessions = len(all_sessions)
    print(f'\n=== Processed {n_sessions} sessions ===')

    if n_sessions == 0:
        print('ERROR: No valid sessions')
        sys.exit(1)

    # === Build output data structure ===
    subjects = list(subjects_set.keys())
    brain_regions = sorted(all_regions_set)
    brain_region_to_idx = {r: i for i, r in enumerate(brain_regions)}

    neural = []
    inputs = []
    outputs = []
    subject_idx = []
    brain_region_idx = []

    for sess in all_sessions:
        neural.append(sess['neural'])
        inputs.append(sess['input'])
        outputs.append(sess['output'])
        subject_idx.append(subjects_set[sess['subject_id']])

        # Map region labels to indices
        ridx = np.array([brain_region_to_idx[r] for r in sess['region_labels']], dtype=np.int64)
        brain_region_idx.append(ridx)

    subject_idx = np.array(subject_idx, dtype=np.int64)

    # === Construct final data dict ===
    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
        'subject_idx': subject_idx,
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_tone_onset', 'photostim_on'],
        'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
        'output_values': [
            ['left', 'right'],                    # choice: 0=left, 1=right
            ['ignore', 'miss', 'hit'],            # outcome: 0=ignore, 1=miss, 2=hit
            ['no', 'yes'],                        # early_lick: 0=no, 1=yes
            ['low', 'mid', 'high'],               # tongue_y: 0=<40th, 1=40-60th, 2=>60th
        ],
        'metadata': {
            'task_description': 'Auditory delayed-response directional licking task. Mice lick left or right based on tone frequency (3kHz or 12kHz) after a 1.2s delay following a go cue.',
            'time_bin_size': BIN_WIDTH * 1000,  # in ms
            'temporal_alignment_event': 'Go cue onset',
            'off_start': ALIGN_START,  # -2.5s
            'off_end': ALIGN_END,      # +1.5s
            'n_sessions': n_sessions,
            'n_subjects': len(subjects),
            'session_info': [
                {
                    'nwb_path': sess['nwb_path'],
                    'subject_id': sess['subject_id'],
                    'subject_desc': sess['subject_desc'],
                    'n_neurons': sess['n_neurons'],
                    'n_trials': sess['n_trials'],
                    'performance': sess['performance'],
                }
                for sess in all_sessions
            ],
        },
    }

    # === Print summary ===
    total_neurons = sum(s['n_neurons'] for s in all_sessions)
    total_trials = sum(s['n_trials'] for s in all_sessions)
    neurons_per_session = [s['n_neurons'] for s in all_sessions]
    trials_per_session = [s['n_trials'] for s in all_sessions]

    print(f'\n=== Summary ===')
    print(f'Subjects: {len(subjects)}')
    print(f'Sessions: {n_sessions}')
    print(f'Total neurons: {total_neurons}')
    print(f'Neurons/session: mean={np.mean(neurons_per_session):.1f}, median={np.median(neurons_per_session):.1f}')
    print(f'Total trials: {total_trials}')
    print(f'Trials/session: mean={np.mean(trials_per_session):.1f}, median={np.median(trials_per_session):.1f}')
    print(f'Brain regions: {brain_regions}')
    print(f'Time bins: {N_BINS} x {BIN_WIDTH*1000:.0f}ms = {ALIGN_START}s to {ALIGN_END}s')

    # Save
    print(f'\nSaving to {outfile}...')
    with open(outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    fsize = os.path.getsize(outfile) / (1024**2)
    total_time = time.time() - t_start
    print(f'Saved {outfile} ({fsize:.1f} MB)')
    print(f'Total conversion time: {total_time:.1f}s ({total_time/60:.1f}min)')

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MAP dataset to decoder format')
    parser.add_argument('outfile', type=str, help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Save processing plots')

    args = parser.parse_args()

    if args.sample:
        print('=== SAMPLE MODE ===')
    else:
        print('=== FULL MODE ===')

    convert_all(args.outfile, sample=args.sample, show_processing=args.show_processing)
