#!/usr/bin/env python3
"""
Convert MAP dataset NWB files to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--sample] [--full] [--show-processing]
"""

import os
import sys
import glob
import time
import pickle
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Constants
# ============================================================
BIN_WIDTH = 0.05  # 50ms bins
ALIGN_START = -2.5  # seconds before go cue
ALIGN_END = 1.5    # seconds after go cue
N_TIMEBINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)  # 80 bins
TONGUE_CONFIDENCE_THRESHOLD = 0.5  # threshold for tongue visibility
LICK_RESPONSE_WINDOW = 1.5  # seconds after go cue to look for first lick

DATA_DIR = '/app/data'

def get_valid_trial_mask(spike_times_list, go_times, align_start=ALIGN_START, align_end=ALIGN_END):
    """Determine which trials have valid neural recording coverage.
    
    A trial is valid if the recording window [go_time + align_start, go_time + align_end]
    is covered by the spike recording period for at least some neurons.
    We use the overall max spike time across all good units as the recording end.
    """
    # Find max and min spike times across all good units
    max_spike = 0
    min_spike = float('inf')
    for spikes in spike_times_list:
        if len(spikes) > 0:
            max_spike = max(max_spike, spikes[-1])
            min_spike = min(min_spike, spikes[0])
    
    if max_spike == 0:
        return np.zeros(len(go_times), dtype=bool)
    
    # A trial is valid if the entire window is within the recording period
    # Add a small buffer (1 bin width) for edge effects
    valid = (go_times + align_start >= min_spike - BIN_WIDTH) & (go_times + align_end <= max_spike + BIN_WIDTH)
    return valid



# ============================================================
# Brain region mapping
# ============================================================
def get_coarse_region(anno_name):
    """Map fine CCF annotation to coarse brain region.
    
    Based on the 5 major brain areas used in the spike sorting QC paper:
    cortex, striatum, thalamus, midbrain, medulla.
    Plus additional regions found in the data.
    """
    if not anno_name or anno_name.strip() == '':
        return 'unknown'
    
    name = anno_name.strip().lower()
    
    # ALM (anterior lateral motor cortex) - Secondary motor area
    if 'secondary motor area' in name:
        return 'ALM'
    if 'primary motor area' in name:
        return 'M1'
    if 'somatosensory' in name:
        return 'SSp'
    if 'frontal pole' in name:
        return 'FRP'
    if 'orbital area' in name:
        return 'ORB'
    if 'agranular insular' in name or 'insular' in name:
        return 'AI'
    if 'retrosplenial' in name:
        return 'RSP'
    if 'visual' in name:
        return 'VIS'
    if 'anterior cingulate' in name:
        return 'ACA'
    if 'prelimbic' in name or 'infralimbic' in name:
        return 'PL'
    if 'gustatory' in name:
        return 'GU'
    if 'visceral' in name:
        return 'VISC'
    if 'temporal' in name or 'auditory' in name:
        return 'AUD'
    if 'piriform' in name or 'olfactory' in name:
        return 'OLF'
    if 'entorhinal' in name:
        return 'ENT'
    # Cortical subplate
    if 'claustrum' in name or 'endopiriform' in name:
        return 'CLA'
    # Hippocampus
    if 'hippocampal' in name or 'subiculum' in name or 'dentate' in name or 'field ca' in name or 'ammon' in name:
        return 'HPF'
    # Striatum
    if 'caudoputamen' in name or 'striatum' in name or 'nucleus accumbens' in name:
        return 'STR'
    # Pallidum
    if 'pallidum' in name or 'globus pallidus' in name or 'substantia innominata' in name or 'bed nuclei' in name or 'diagonal band' in name:
        return 'PAL'
    # Thalamus  
    if any(x in name for x in [
        'thalamus', 'thalamic', 'geniculate', 'habenula',
        'ventral anterior-lateral', 'ventral medial', 'ventral posterolateral',
        'ventral posteromedial', 'posterior complex', 'lateral posterior',
        'mediodorsal', 'parafascicular', 'reticular nucleus of the thalamus',
        'anterodorsal', 'anteromedial', 'anteroventral',
        'central medial', 'central lateral', 'paracentral',
        'reuniens', 'rhomboid', 'submedial',
        'lateral dorsal nucleus', 'zona incerta', 'subthalamic',
        'paraventricular nucleus of the thalamus',
        'nucleus of reuniens'
    ]):
        return 'TH'
    # Hypothalamus
    if 'hypothal' in name or 'lateral hypothalamic' in name or 'lateral preoptic' in name or 'medial preoptic' in name:
        return 'HY'
    # Midbrain
    if any(x in name for x in [
        'superior colliculus', 'inferior colliculus', 'periaqueductal',
        'midbrain reticular', 'substantia nigra', 'ventral tegmental',
        'red nucleus', 'anterior pretectal', 'nucleus of the optic',
        'pedunculopontine', 'cuneiform', 'interpeduncular',
        'midbrain', 'pretectal', 'nucleus of darkschewitsch',
        'interstitial nucleus'
    ]):
        return 'MB'
    # Pons
    if any(x in name for x in [
        'pontine', 'pons', 'parabrachial', 'locus ceruleus',
        'tegmental nucleus', 'barrington', 'pontine reticular',
        'superior olivary', 'motor nucleus of trigeminal',
        'principal sensory nucleus', 'nucleus raphe',
        'supratrigeminal', 'koelliker-fuse'
    ]):
        return 'P'
    # Medulla
    if any(x in name for x in [
        'medulla', 'nucleus of the solitary', 'dorsal column',
        'spinal nucleus of the trigeminal', 'facial motor',
        'vestibular', 'cochlear', 'inferior olivary',
        'gigantocellular', 'paragigantocellular',
        'lateral reticular', 'ambiguus', 'hypoglossal',
        'nucleus prepositus', 'external cuneate',
        'nucleus of roller', 'nucleus x', 'parvicellular reticular'
    ]):
        return 'MY'
    # Cerebellum
    if 'cerebell' in name or 'purkinje' in name or 'floccul' in name:
        return 'CB'
    # Generic cortex catch-all
    if any(x in name for x in ['cortex', 'cortical', 'layer', 'area']):
        return 'CTX_other'
    
    return 'other'


# ============================================================
# Fast NWB loading using h5py
# ============================================================
def load_session_h5py(nwb_path):
    """Load session data from NWB file using h5py for speed."""
    data = {}
    
    with h5py.File(nwb_path, 'r') as f:
        # Subject info
        data['subject_id'] = f['general']['subject']['subject_id'][()].decode() if isinstance(f['general']['subject']['subject_id'][()], bytes) else str(f['general']['subject']['subject_id'][()])
        data['subject_desc'] = f['general']['subject']['description'][()].decode() if isinstance(f['general']['subject']['description'][()], bytes) else str(f['general']['subject']['description'][()])
        
        # Units
        units_grp = f['units']
        data['classification'] = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in units_grp['classification'][()]])
        data['anno_name'] = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in units_grp['anno_name'][()]])
        
        # Spike times - use the ragged array structure
        spike_times_data = units_grp['spike_times'][()]
        spike_times_index = units_grp['spike_times_index'][()]
        
        # Build spike times list
        n_units = len(data['classification'])
        all_spike_times = []
        prev_idx = 0
        for i in range(n_units):
            end_idx = spike_times_index[i]
            all_spike_times.append(spike_times_data[prev_idx:end_idx])
            prev_idx = end_idx
        data['all_spike_times'] = all_spike_times
        
        # Trials
        trials_grp = f['intervals']['trials']
        data['trial_start'] = trials_grp['start_time'][()]
        data['trial_stop'] = trials_grp['stop_time'][()]
        data['trial_instruction'] = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in trials_grp['trial_instruction'][()]])
        data['outcome'] = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in trials_grp['outcome'][()]])
        data['early_lick'] = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in trials_grp['early_lick'][()]])
        
        # Behavioral Events
        acq = f['acquisition']
        be = acq['BehavioralEvents']
        
        data['go_times'] = be['go_start_times']['timestamps'][()]
        data['sample_starts'] = be['sample_start_times']['timestamps'][()]
        data['left_lick_times'] = be['left_lick_times']['timestamps'][()]
        data['right_lick_times'] = be['right_lick_times']['timestamps'][()]
        
        if 'photostim_start_times' in be:
            data['photostim_starts'] = be['photostim_start_times']['timestamps'][()]
            data['photostim_stops'] = be['photostim_stop_times']['timestamps'][()]
        else:
            data['photostim_starts'] = np.array([])
            data['photostim_stops'] = np.array([])
        
        # Tongue tracking
        bts = acq['BehavioralTimeSeries']
        data['tongue_data'] = bts['Camera0_side_TongueTracking']['data'][()]
        data['tongue_timestamps'] = bts['Camera0_side_TongueTracking']['timestamps'][()]
    
    return data


# ============================================================
# Vectorized processing functions
# ============================================================
def compute_firing_rates_all_trials(spike_times_list, go_times, 
                                     bin_width=BIN_WIDTH,
                                     align_start=ALIGN_START, 
                                     align_end=ALIGN_END):
    """
    Compute firing rates for all neurons across all trials.
    Vectorized for speed.
    
    Returns:
        list of (n_neurons, n_timebins) arrays, one per trial
    """
    n_neurons = len(spike_times_list)
    n_trials = len(go_times)
    n_bins = int((align_end - align_start) / bin_width)
    
    # Bin edges relative to alignment
    bin_edges_rel = np.arange(n_bins + 1) * bin_width + align_start
    
    results = []
    
    for t in range(n_trials):
        go_time = go_times[t]
        abs_bin_edges = bin_edges_rel + go_time
        
        fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
        
        for i, spikes in enumerate(spike_times_list):
            if len(spikes) == 0:
                continue
            # Quick bounds check
            if spikes[-1] < abs_bin_edges[0] or spikes[0] > abs_bin_edges[-1]:
                continue
            # Find spikes in window using searchsorted
            left = np.searchsorted(spikes, abs_bin_edges[0])
            right = np.searchsorted(spikes, abs_bin_edges[-1])
            spikes_in_window = spikes[left:right]
            if len(spikes_in_window) == 0:
                continue
            counts, _ = np.histogram(spikes_in_window, bins=abs_bin_edges)
            fr[i] = counts / bin_width
        
        results.append(fr)
    
    return results


def compute_tongue_y_all_trials(tongue_data, tongue_timestamps, go_times,
                                 bin_width=BIN_WIDTH, align_start=ALIGN_START,
                                 align_end=ALIGN_END):
    """
    Compute discretized tongue y-position for all trials.
    Vectorized using searchsorted.
    
    Returns:
        list of (1, n_timebins) int arrays
        p40, p60: percentile thresholds
    """
    n_bins = int((align_end - align_start) / bin_width)
    n_trials = len(go_times)
    
    tongue_y = tongue_data[:, 1]
    tongue_conf = tongue_data[:, 2]
    
    # Session-wide percentiles from visible frames
    visible_mask = tongue_conf > TONGUE_CONFIDENCE_THRESHOLD
    y_visible = tongue_y[visible_mask]
    
    if len(y_visible) > 0:
        p40 = np.percentile(y_visible, 40)
        p60 = np.percentile(y_visible, 60)
    else:
        p40 = 0.0
        p60 = 0.0
    
    bin_edges_rel = np.arange(n_bins + 1) * bin_width + align_start
    
    # Precompute stride for tongue timestamps
    # Tongue data is at regular intervals (0.0034s)
    dt = np.median(np.diff(tongue_timestamps[:100]))
    t0_tongue = tongue_timestamps[0]
    
    results = []
    for t in range(n_trials):
        go_time = go_times[t]
        abs_bin_edges = bin_edges_rel + go_time
        
        trial_tongue_y = np.full(n_bins, 3, dtype=np.int64)  # default: not visible
        
        for b in range(n_bins):
            # Use searchsorted for fast index lookup
            idx_start = np.searchsorted(tongue_timestamps, abs_bin_edges[b])
            idx_end = np.searchsorted(tongue_timestamps, abs_bin_edges[b + 1])
            
            if idx_start >= idx_end or idx_start >= len(tongue_timestamps):
                continue
            
            bin_conf = tongue_conf[idx_start:idx_end]
            bin_y = tongue_y[idx_start:idx_end]
            
            visible_frac = np.mean(bin_conf > TONGUE_CONFIDENCE_THRESHOLD)
            if visible_frac > 0.5:
                vis_mask = bin_conf > TONGUE_CONFIDENCE_THRESHOLD
                mean_y = np.mean(bin_y[vis_mask])
                if mean_y < p40:
                    trial_tongue_y[b] = 0
                elif mean_y < p60:
                    trial_tongue_y[b] = 1
                else:
                    trial_tongue_y[b] = 2
        
        results.append(trial_tongue_y.reshape(1, -1))
    
    return results, p40, p60


def get_lick_choices(go_times, left_lick_times, right_lick_times, 
                     response_window=LICK_RESPONSE_WINDOW):
    """Compute lick choice for all trials. Returns array of int."""
    choices = np.full(len(go_times), 2, dtype=np.int64)  # default: no lick
    
    for t, go_time in enumerate(go_times):
        # Find first left lick after go
        left_idx = np.searchsorted(left_lick_times, go_time)
        first_left = left_lick_times[left_idx] if left_idx < len(left_lick_times) and left_lick_times[left_idx] < go_time + response_window else float('inf')
        
        # Find first right lick after go
        right_idx = np.searchsorted(right_lick_times, go_time)
        first_right = right_lick_times[right_idx] if right_idx < len(right_lick_times) and right_lick_times[right_idx] < go_time + response_window else float('inf')
        
        if first_left < first_right:
            choices[t] = 0  # left
        elif first_right < float('inf'):
            choices[t] = 1  # right
    
    return choices


def get_outcome_codes(outcomes):
    """Map outcome strings to integer codes."""
    mapping = {'ignore': 0, 'miss': 1, 'hit': 2}
    return np.array([mapping.get(o, 0) for o in outcomes], dtype=np.int64)


def get_early_lick_codes(early_licks):
    """Map early_lick strings to integer codes."""
    return np.array([0 if e == 'no early' else 1 for e in early_licks], dtype=np.int64)


def find_sample_starts_for_trials(trial_starts, trial_stops, sample_starts, go_times):
    """Find tone onset time relative to go cue for each trial."""
    n_trials = len(trial_starts)
    tone_rel_go = np.full(n_trials, -1.85, dtype=np.float64)  # default
    
    for t in range(n_trials):
        # Find sample starts within this trial
        idx_start = np.searchsorted(sample_starts, trial_starts[t])
        idx_end = np.searchsorted(sample_starts, trial_stops[t])
        if idx_start < idx_end:
            tone_rel_go[t] = sample_starts[idx_start] - go_times[t]
    
    return tone_rel_go


def compute_inputs_all_trials(go_times, tone_rel_go, photostim_starts, photostim_stops,
                               bin_width=BIN_WIDTH, align_start=ALIGN_START,
                               align_end=ALIGN_END):
    """Compute input arrays for all trials."""
    n_bins = int((align_end - align_start) / bin_width)
    n_trials = len(go_times)
    
    bin_centers_rel = np.arange(n_bins) * bin_width + align_start + bin_width / 2
    
    results = []
    for t in range(n_trials):
        go_time = go_times[t]
        
        # Time from tone onset
        time_from_tone = (bin_centers_rel - tone_rel_go[t]).astype(np.float32)
        
        # Photostim
        photostim = np.zeros(n_bins, dtype=np.float32)
        abs_bin_centers = bin_centers_rel + go_time
        for ps_start, ps_stop in zip(photostim_starts, photostim_stops):
            mask = (abs_bin_centers >= ps_start) & (abs_bin_centers < ps_stop)
            photostim[mask] = 1.0
        
        input_data = np.vstack([time_from_tone.reshape(1, -1), 
                                photostim.reshape(1, -1)])
        results.append(input_data)
    
    return results


def compute_outputs_all_trials(choices, outcome_codes, early_lick_codes, tongue_y_list):
    """Combine all outputs into per-trial arrays."""
    n_trials = len(choices)
    n_bins = tongue_y_list[0].shape[1]
    
    results = []
    for t in range(n_trials):
        output_data = np.zeros((4, n_bins), dtype=np.int64)
        output_data[0, :] = choices[t]
        output_data[1, :] = outcome_codes[t]
        output_data[2, :] = early_lick_codes[t]
        output_data[3, :] = tongue_y_list[t][0, :]
        results.append(output_data)
    
    return results


# ============================================================
# Main processing
# ============================================================
def process_session(nwb_path, show_processing=False, session_idx=0):
    """Process a single NWB session file."""
    t0 = time.time()
    basename = os.path.basename(nwb_path)
    
    # Load data
    data = load_session_h5py(nwb_path)
    t_load = time.time()
    
    # Filter good units
    good_mask = data['classification'] == 'good'
    n_good = np.sum(good_mask)
    
    if n_good == 0:
        print(f"  Skipping {basename}: no good units")
        return None
    
    good_indices = np.where(good_mask)[0]
    spike_times_good = [data['all_spike_times'][i] for i in good_indices]
    anno_names = data['anno_name'][good_indices]
    coarse_regions = [get_coarse_region(a) for a in anno_names]
    
    n_trials = len(data['trial_start'])
    go_times = data['go_times']
    
    t_filter = time.time()
    
    # Determine valid trials (with neural recording coverage)
    valid_mask = get_valid_trial_mask(spike_times_good, go_times)
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)
    
    if n_valid < 2:
        print(f"  Skipping {basename}: only {n_valid} valid trials")
        return None
    
    if n_valid < n_trials:
        print(f"  Filtering: {n_valid}/{n_trials} trials have neural coverage")
    
    # Filter to valid trials
    go_times_valid = go_times[valid_indices]
    trial_starts_valid = data['trial_start'][valid_indices]
    trial_stops_valid = data['trial_stop'][valid_indices]
    outcome_valid = data['outcome'][valid_indices]
    early_lick_valid = data['early_lick'][valid_indices]
    
    # Compute firing rates for valid trials only
    neural_trials = compute_firing_rates_all_trials(spike_times_good, go_times_valid)
    t_neural = time.time()
    
    # Compute tone onset times
    tone_rel_go = find_sample_starts_for_trials(
        trial_starts_valid, trial_stops_valid, data['sample_starts'], go_times_valid
    )
    
    # Compute inputs
    input_trials = compute_inputs_all_trials(
        go_times_valid, tone_rel_go, data['photostim_starts'], data['photostim_stops']
    )
    t_input = time.time()
    
    # Compute outputs
    choices = get_lick_choices(go_times_valid, data['left_lick_times'], data['right_lick_times'])
    outcome_codes = get_outcome_codes(outcome_valid)
    early_lick_codes = get_early_lick_codes(early_lick_valid)
    
    tongue_y_list, p40, p60 = compute_tongue_y_all_trials(
        data['tongue_data'], data['tongue_timestamps'], go_times_valid
    )
    
    output_trials = compute_outputs_all_trials(
        choices, outcome_codes, early_lick_codes, tongue_y_list
    )
    t_output = time.time()
    
    print(f"  {basename}: {n_good} good units, {n_trials} trials")
    print(f"    load={t_load-t0:.1f}s, filter={t_filter-t_load:.1f}s, "
          f"neural={t_neural-t_filter:.1f}s, input={t_input-t_neural:.1f}s, "
          f"output={t_output-t_input:.1f}s, total={t_output-t0:.1f}s")
    
    # Show processing plots
    if show_processing:
        plot_processing(basename, neural_trials, input_trials, output_trials,
                       go_times, spike_times_good, data['tongue_data'], 
                       data['tongue_timestamps'], p40, p60, session_idx)
    
    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'coarse_regions': coarse_regions,
        'anno_names': anno_names,
        'subject_id': data['subject_id'],
        'subject_desc': data['subject_desc'],
        'n_good': n_good,
        'n_trials': n_valid,
    }


def plot_processing(basename, neural_trials, input_trials, output_trials,
                    go_times, spike_times_all, tongue_data, tongue_timestamps,
                    p40, p60, session_idx):
    """Plot processing visualizations for a session."""
    fig, axes = plt.subplots(4, 2, figsize=(20, 16))
    fig.suptitle(f'Processing: {basename}', fontsize=14)
    
    n_bins = N_TIMEBINS
    time_axis = np.linspace(ALIGN_START, ALIGN_END, n_bins, endpoint=False) + BIN_WIDTH/2
    
    # Plot 1: Neural activity heatmap for a trial
    trial_idx = min(5, len(neural_trials) - 1)
    ax = axes[0, 0]
    fr = neural_trials[trial_idx]
    n_show = min(50, fr.shape[0])
    im = ax.imshow(fr[:n_show], aspect='auto', 
                   extent=[ALIGN_START, ALIGN_END, n_show, 0])
    ax.set_title(f'Neural activity (trial {trial_idx}, first {n_show} neurons)')
    ax.set_xlabel('Time from go cue (s)')
    ax.set_ylabel('Neuron')
    ax.axvline(0, color='r', linestyle='--', label='Go cue')
    plt.colorbar(im, ax=ax, label='FR (Hz)')
    
    # Plot 2: Mean firing rate across neurons for several trials
    ax = axes[0, 1]
    for t in range(min(5, len(neural_trials))):
        ax.plot(time_axis, np.mean(neural_trials[t], axis=0), alpha=0.7, label=f'Trial {t}')
    ax.set_title('Population mean FR over time')
    ax.set_xlabel('Time from go cue (s)')
    ax.set_ylabel('Mean FR (Hz)')
    ax.axvline(0, color='r', linestyle='--')
    ax.legend(fontsize=8)
    
    # Plot 3: Time from tone onset input
    ax = axes[1, 0]
    for t in range(min(5, len(input_trials))):
        ax.plot(time_axis, input_trials[t][0], alpha=0.7, label=f'Trial {t}')
    ax.set_title('Input: Time from tone onset')
    ax.set_xlabel('Time from go cue (s)')
    ax.set_ylabel('Time from tone (s)')
    ax.axvline(0, color='r', linestyle='--')
    ax.legend(fontsize=8)
    
    # Plot 4: Photostim input
    ax = axes[1, 1]
    stim_trials = [t for t in range(len(input_trials)) if np.any(input_trials[t][1] > 0)]
    if stim_trials:
        for t in stim_trials[:5]:
            ax.plot(time_axis, input_trials[t][1] + t*0.05, alpha=0.7, label=f'Trial {t}')
        ax.legend(fontsize=8)
    ax.set_title(f'Input: Photostim ({len(stim_trials)} stim trials)')
    ax.set_xlabel('Time from go cue (s)')
    ax.axvline(0, color='r', linestyle='--')
    
    # Plot 5: Choice distribution
    ax = axes[2, 0]
    choices = [output_trials[t][0, 0] for t in range(len(output_trials))]
    ax.hist(choices, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Left', 'Right', 'No lick'])
    ax.set_title('Output: Choice distribution')
    
    # Plot 6: Outcome distribution
    ax = axes[2, 1]
    outcomes = [output_trials[t][1, 0] for t in range(len(output_trials))]
    ax.hist(outcomes, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Ignore', 'Miss', 'Hit'])
    ax.set_title('Output: Outcome distribution')
    
    # Plot 7: Tongue y position over time
    ax = axes[3, 0]
    for t in range(min(5, len(output_trials))):
        ax.plot(time_axis, output_trials[t][3], alpha=0.7, label=f'Trial {t}')
    ax.set_title('Output: Tongue Y (discretized)')
    ax.set_xlabel('Time from go cue (s)')
    ax.set_ylabel('Category')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['<p40', 'p40-p60', '>p60', 'not visible'])
    ax.axvline(0, color='r', linestyle='--')
    ax.legend(fontsize=8)
    
    # Plot 8: Early lick distribution
    ax = axes[3, 1]
    early = [output_trials[t][2, 0] for t in range(len(output_trials))]
    ax.hist(early, bins=[-0.5, 0.5, 1.5], rwidth=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No early', 'Early'])
    ax.set_title('Output: Early lick distribution')
    
    plt.tight_layout()
    plt.savefig(f'processing_{session_idx}.png', dpi=150)
    plt.close(fig)
    print(f"  Saved processing plot: processing_{session_idx}.png")


def main():
    parser = argparse.ArgumentParser(description='Convert MAP dataset to decoder format')
    parser.add_argument('outfile', type=str, help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()
    
    if not args.sample:
        args.full = True
    
    # Find all NWB files
    nwb_files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
    print(f"Found {len(nwb_files)} NWB files")
    
    if args.sample:
        # Pick 2 sessions from different subjects
        nwb_files = [nwb_files[0], nwb_files[20]]
        print(f"Sample mode: processing {len(nwb_files)} sessions")
    
    # Process all sessions
    all_neural = []
    all_input = []
    all_output = []
    all_brain_region_idx = []
    all_subject_idx = []
    
    subjects_list = []
    subject_to_idx = {}
    brain_regions_list = []
    region_to_idx = {}
    
    total_t0 = time.time()
    
    for i, nwb_path in enumerate(nwb_files):
        print(f"\nProcessing session {i+1}/{len(nwb_files)}: {os.path.basename(nwb_path)}")
        
        show = args.show_processing and i < 2
        result = process_session(nwb_path, show_processing=show, session_idx=i)
        
        if result is None:
            continue
        
        # Track subjects
        subj_id = result['subject_id']
        if subj_id not in subject_to_idx:
            subject_to_idx[subj_id] = len(subjects_list)
            subjects_list.append(subj_id)
        
        # Track brain regions
        neuron_region_idx = np.zeros(result['n_good'], dtype=np.int64)
        for j, region in enumerate(result['coarse_regions']):
            if region not in region_to_idx:
                region_to_idx[region] = len(brain_regions_list)
                brain_regions_list.append(region)
            neuron_region_idx[j] = region_to_idx[region]
        
        all_neural.append(result['neural'])
        all_input.append(result['input'])
        all_output.append(result['output'])
        all_brain_region_idx.append(neuron_region_idx)
        all_subject_idx.append(subject_to_idx[subj_id])
        
        elapsed = time.time() - total_t0
        rate = elapsed / (i + 1)
        remaining = rate * (len(nwb_files) - i - 1)
        print(f"  Elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s")
    
    total_time = time.time() - total_t0
    print(f"\nTotal processing time: {total_time:.1f}s")
    
    # Build output dictionary
    output_data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': subjects_list,
        'subject_idx': np.array(all_subject_idx, dtype=np.int64),
        'brain_regions': brain_regions_list,
        'brain_region_idx': all_brain_region_idx,
        'input_names': ['time_from_tone_onset', 'photostim_on'],
        'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
        'output_values': [
            ['left', 'right', 'no_lick'],
            ['ignore', 'miss', 'hit'],
            ['no', 'yes'],
            ['below_p40', 'p40_to_p60', 'above_p60', 'not_visible'],
        ],
        'metadata': {
            'task_description': 'Audio delayed-response licking task. Mice hear a tone during the sample period, wait through a delay period, then lick left or right after the go cue.',
            'time_bin_size': BIN_WIDTH * 1000,
            'temporal_alignment_event': 'Go cue onset',
            'off_start': ALIGN_START,
            'off_end': ALIGN_END,
            'dataset': 'Mesoscale Activity Map (MAP) Dataset, DANDI:000363',
            'bin_width_s': BIN_WIDTH,
            'n_timebins': N_TIMEBINS,
        }
    }
    
    # Save
    print(f"\nSaving to {args.outfile}...")
    with open(args.outfile, 'wb') as f:
        pickle.dump(output_data, f)
    
    file_size = os.path.getsize(args.outfile) / (1024**2)
    print(f"Saved {args.outfile} ({file_size:.1f} MB)")
    
    # Print summary
    n_sessions = len(all_neural)
    total_trials = sum(len(s) for s in all_neural)
    total_neurons = sum(len(all_brain_region_idx[s]) for s in range(n_sessions))
    print(f"\nSummary:")
    print(f"  Sessions: {n_sessions}")
    print(f"  Subjects: {len(subjects_list)}")
    print(f"  Total trials: {total_trials}")
    print(f"  Total neurons: {total_neurons}")
    print(f"  Brain regions: {brain_regions_list}")
    print(f"  Time bins per trial: {N_TIMEBINS}")


if __name__ == '__main__':
    main()
