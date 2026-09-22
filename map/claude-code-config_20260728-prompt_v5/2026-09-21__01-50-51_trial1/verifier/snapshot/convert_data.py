#!/usr/bin/env python3
"""
Convert MAP dataset NWB files to decoder-compatible pickle format.

Usage:
    python -u convert_data.py <outpicklefile> [--sample] [--show-processing]
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pynwb

# ============================================================
# Brain region mapping from CCF anno_name to broad regions
# ============================================================

def classify_brain_region(anno_name):
    """Map a CCF annotation name to a broad brain region."""
    a = anno_name.lower().strip()
    if a == '' or a == 'nan':
        return 'Unknown'

    # ALM = Secondary motor area
    if 'secondary motor area' in a:
        return 'ALM'
    # Orbital cortex
    if 'orbital area' in a:
        return 'Orbital'
    # Olfactory
    if any(kw in a for kw in ['olfactory', 'piriform', 'taenia tecta',
                                'dorsal peduncular']):
        return 'Olfactory'
    # Hippocampus
    if any(kw in a for kw in ['field ca1', 'field ca2', 'field ca3',
                                'dentate gyrus', 'subiculum', 'postsubiculum',
                                'hippocampal', 'entorhinal']):
        return 'Hippocampus'
    # Cortical subplate (amygdala, claustrum, etc.)
    if any(kw in a for kw in ['cortical subplate', 'claustrum',
                                'endopiriform',
                                'basolateral amygdalar', 'basomedial amygdalar',
                                'lateral amygdalar', 'central amygdalar',
                                'medial amygdalar', 'posterior amygdalar',
                                'anterior amygdalar', 'intercalated amygdalar']):
        return 'CorticalSubplate'
    # Striatum
    if any(kw in a for kw in ['caudoputamen', 'nucleus accumbens',
                                'fundus of striatum', 'olfactory tubercle',
                                'lateral septal nucleus', 'septofimbrial',
                                'triangular nucleus of septum']):
        return 'Striatum'
    if a == 'striatum':
        return 'Striatum'
    # Pallidum (Allen CCF: includes globus pallidus, substantia innominata, BNST)
    if any(kw in a for kw in ['globus pallidus', 'pallidum',
                                'substantia innominata',
                                'bed nuclei of the stria terminalis']):
        return 'Pallidum'
    # Thalamus
    if any(kw in a for kw in ['nucleus of the thalamus', 'nucleus of thalamus',
                                'ventral anterior-lateral complex',
                                'ventral medial nucleus',
                                'ventral posterolateral nucleus',
                                'ventral posteromedial nucleus',
                                'posterior complex of the thalamus',
                                'mediodorsal nucleus of thalamus',
                                'lateral dorsal nucleus of thalamus',
                                'lateral posterior nucleus of the thalamus',
                                'anterodorsal nucleus',
                                'anteromedial nucleus',
                                'anteroventral nucleus',
                                'reticular nucleus of the thalamus',
                                'paraventricular nucleus of the thalamus',
                                'dorsal part of the lateral geniculate',
                                'medial geniculate complex',
                                'suprageniculate nucleus',
                                'posterior limiting nucleus',
                                'lateral habenula', 'medial habenula',
                                'submedial nucleus',
                                'rhomboid nucleus', 'perireunensis',
                                'paracentral nucleus',
                                'nucleus of the optic tract',
                                'peripeduncular nucleus',
                                'subparafascicular']):
        return 'Thalamus'
    if a == 'thalamus':
        return 'Thalamus'
    # Hypothalamus (Allen CCF: includes subthalamus - ZI, fields of Forel, STN)
    if any(kw in a for kw in ['hypothal', 'lateral preoptic',
                                'parasubthalamic', 'tuberomammillary',
                                'lateral hypothalamic', 'posterior hypothalamic',
                                'zona incerta', 'fields of forel',
                                'subthalamic nucleus']):
        return 'Hypothalamus'
    if a == 'hypothalamus':
        return 'Hypothalamus'
    # Midbrain
    if any(kw in a for kw in ['superior colliculus', 'inferior colliculus',
                                'midbrain reticular', 'periaqueductal gray',
                                'red nucleus', 'substantia nigra',
                                'ventral tegmental', 'pedunculopontine',
                                'anterior pretectal', 'posterior pretectal',
                                'dorsal terminal nucleus',
                                'medial terminal nucleus',
                                'nucleus of the brachium',
                                'nucleus of the lateral lemniscus',
                                'nucleus sagulum',
                                'parafascicular nucleus']):
        return 'Midbrain'
    if a == 'midbrain':
        return 'Midbrain'
    # Pons
    if any(kw in a for kw in ['pontine reticular', 'tegmental reticular',
                                'parabrachial', 'koelliker-fuse',
                                'locus ceruleus']):
        return 'Pons'
    if a == 'pons':
        return 'Pons'
    # Cerebellum
    if any(kw in a for kw in ['cerebellum', 'lobule', 'lobules',
                                'simple lobule', 'crus 1', 'crus 2',
                                'declive', 'pyramus', 'uvula',
                                'copula pyramidis', 'paramedian lobule',
                                'nodulus', 'lingula', 'fastigial',
                                'interposed nucleus', 'infracerebellar']):
        return 'Cerebellum'
    # Medulla
    if any(kw in a for kw in ['medulla', 'medullary reticular',
                                'gigantocellular', 'magnocellular reticular',
                                'parvicellular reticular',
                                'paragigantocellular', 'parapyramidal',
                                'intermediate reticular',
                                'lateral reticular nucleus',
                                'inferior olivary', 'nucleus raphe',
                                'nucleus of the solitary',
                                'dorsal motor nucleus of the vagus',
                                'hypoglossal', 'facial motor',
                                'external cuneate', 'nucleus of roller',
                                'nucleus x', 'parasolitary',
                                'spinal nucleus of the trigeminal',
                                'vestibular nucleus', 'spinal vestibular',
                                'superior vestibular', 'lateral vestibular',
                                'medial vestibular']):
        return 'Medulla'
    if a == 'medulla':
        return 'Medulla'
    # Other cortex
    if any(kw in a for kw in ['motor area', 'somatosensory', 'visual area',
                                'auditory area', 'retrosplenial',
                                'cingulate', 'prelimbic', 'infralimbic',
                                'frontal pole', 'agranular insular',
                                'gustatory', 'visceral area',
                                'supplemental somatosensory',
                                'temporal association',
                                'perirhinal', 'ectorhinal',
                                'posteromedial visual']):
        return 'OtherCortex'
    if 'olfactory' in a:
        return 'Olfactory'
    return 'Unknown'


def get_brain_region_for_unit(anno_name, electrode_group_location):
    """Get brain region for a unit, using anno_name primarily, falling back to electrode group."""
    region = classify_brain_region(anno_name)
    if region == 'Unknown' and electrode_group_location:
        try:
            loc = json.loads(electrode_group_location)
            eg_region = loc.get('brain_regions', '').replace('left ', '').replace('right ', '')
            if eg_region:
                return eg_region
        except (json.JSONDecodeError, KeyError):
            pass
    return region


# ============================================================
# Optimized spike binning
# ============================================================

def bin_spikes_all_trials(spike_times_unit, go_cue_times, bin_edges):
    """Bin spike times for one unit across all trials at once.

    Args:
        spike_times_unit: 1D array of absolute spike times for one unit
        go_cue_times: 1D array of go cue times for valid trials
        bin_edges: 1D array of bin edges relative to go cue

    Returns:
        fr_matrix: (n_trials, n_bins) firing rates
    """
    n_trials = len(go_cue_times)
    n_bins = len(bin_edges) - 1
    bin_width = bin_edges[1] - bin_edges[0]
    fr = np.zeros((n_trials, n_bins), dtype=np.float32)

    # For each trial, find spikes in the window and histogram them
    begin = bin_edges[0]
    end = bin_edges[-1]

    for t in range(n_trials):
        gc = go_cue_times[t]
        # Get spikes in window
        mask = (spike_times_unit >= gc + begin) & (spike_times_unit < gc + end)
        if np.any(mask):
            aligned = spike_times_unit[mask] - gc
            counts, _ = np.histogram(aligned, bins=bin_edges)
            fr[t] = counts / bin_width

    return fr


def bin_tongue_all_trials(tongue_timestamps, tongue_data, go_cue_times, bin_edges,
                          tongue_y_p40, tongue_y_p60, likelihood_thresh=0.9):
    """Bin tongue y-position for all trials at once using searchsorted.

    Returns: (n_trials, n_bins) array of discretized tongue y categories
    """
    n_trials = len(go_cue_times)
    n_bins = len(bin_edges) - 1
    result = np.full((n_trials, n_bins), 3, dtype=np.int64)  # default: not visible

    for t in range(n_trials):
        gc = go_cue_times[t]
        abs_edges = gc + bin_edges

        # Find frame indices for this trial's window
        i_start = np.searchsorted(tongue_timestamps, abs_edges[0])
        i_end = np.searchsorted(tongue_timestamps, abs_edges[-1])

        if i_start >= i_end:
            continue

        trial_ts = tongue_timestamps[i_start:i_end]
        trial_data = tongue_data[i_start:i_end]

        # Assign each frame to a bin
        bin_idx = np.searchsorted(abs_edges, trial_ts, side='right') - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        for b in range(n_bins):
            frame_mask = bin_idx == b
            if not np.any(frame_mask):
                continue
            frames = trial_data[frame_mask]
            avg_likelihood = np.mean(frames[:, 2])
            if avg_likelihood >= likelihood_thresh:
                avg_y = np.mean(frames[:, 1])
                if avg_y < tongue_y_p40:
                    result[t, b] = 0
                elif avg_y <= tongue_y_p60:
                    result[t, b] = 1
                else:
                    result[t, b] = 2

    return result


def process_session(nwb_path, bin_edges, n_bins, show_processing=False, session_idx=0):
    """Process one NWB session file. Optimized version."""
    t0 = time.time()
    io = pynwb.NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()

    subject_id = nwb.subject.subject_id
    subject_desc = nwb.subject.description

    # --- Units ---
    classification = nwb.units['classification'][:]
    good_mask = classification == 'good'
    good_indices = np.where(good_mask)[0]
    n_good = len(good_indices)

    if n_good == 0:
        io.close()
        return None

    anno_names = nwb.units['anno_name'][:]
    electrode_groups = nwb.units['electrode_group'][:]

    # Brain region for each good unit
    unit_regions = []
    for idx in good_indices:
        an = anno_names[idx]
        eg = electrode_groups[idx]
        eg_loc = eg.location if hasattr(eg, 'location') else ''
        region = get_brain_region_for_unit(an, eg_loc)
        unit_regions.append(region)

    # --- Trials ---
    auto_water = nwb.trials['auto_water'][:]
    free_water = nwb.trials['free_water'][:]
    trial_instruction = nwb.trials['trial_instruction'][:]
    early_lick_arr = nwb.trials['early_lick'][:]
    outcome_arr = nwb.trials['outcome'][:]

    valid_trial_mask = (auto_water == 0) & (free_water == 0)

    # --- Timing ---
    be = nwb.acquisition['BehavioralEvents']
    go_cue_times_all = be.time_series['go_start_times'].timestamps[:]
    sample_start_times = be.time_series['sample_start_times'].timestamps[:]

    # Filter trials to those within the recording observation interval
    # Use obs_intervals of the first good unit to determine the recording window
    obs_intervals = nwb.units['obs_intervals'][good_indices[0]]
    obs_start = obs_intervals[0, 0]
    obs_end = obs_intervals[-1, 1]

    # Only include trials whose go cue + analysis window falls within obs range
    begin_time = bin_edges[0]
    end_time = bin_edges[-1]
    within_obs = ((go_cue_times_all + begin_time) >= obs_start) & \
                 ((go_cue_times_all + end_time) <= obs_end)
    valid_trial_mask = valid_trial_mask & within_obs

    valid_trial_indices = np.where(valid_trial_mask)[0]
    n_valid_trials = len(valid_trial_indices)

    if n_valid_trials < 2:
        io.close()
        return None

    go_cue_times = go_cue_times_all[valid_trial_indices]

    # Photostim
    has_photostim = 'photostim_start_times' in be.time_series
    if has_photostim:
        ps_start_times = be.time_series['photostim_start_times'].timestamps[:]
        ps_stop_times = be.time_series['photostim_stop_times'].timestamps[:]
    else:
        ps_start_times = np.array([])
        ps_stop_times = np.array([])

    # --- Tongue tracking ---
    bts = nwb.acquisition['BehavioralTimeSeries']
    tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
    tongue_data = tongue_ts_obj.data[:]
    tongue_timestamps = tongue_ts_obj.timestamps[:]

    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    TONGUE_LIKELIHOOD_THRESH = 0.9

    # Per-session tongue y percentiles (visible tongue only)
    visible_mask_all = tongue_data[:, 2] >= TONGUE_LIKELIHOOD_THRESH
    if np.sum(visible_mask_all) > 0:
        visible_y = tongue_data[visible_mask_all, 1]
        tongue_y_p40 = np.percentile(visible_y, 40)
        tongue_y_p60 = np.percentile(visible_y, 60)
    else:
        tongue_y_p40 = 0.0
        tongue_y_p60 = 0.0

    t_read = time.time()

    # --- Bin spikes for all good units across all valid trials ---
    # Shape: (n_good, n_valid_trials, n_bins)
    all_spike_times_obj = nwb.units['spike_times']
    fr_all = np.zeros((n_good, n_valid_trials, n_bins), dtype=np.float32)

    for i, unit_idx in enumerate(good_indices):
        st = all_spike_times_obj[unit_idx]
        fr_all[i] = bin_spikes_all_trials(st, go_cue_times, bin_edges)

    t_spike = time.time()

    # --- Bin tongue for all valid trials ---
    tongue_disc = bin_tongue_all_trials(tongue_timestamps, tongue_data, go_cue_times,
                                         bin_edges, tongue_y_p40, tongue_y_p60,
                                         TONGUE_LIKELIHOOD_THRESH)
    t_tongue = time.time()

    # --- Build input/output for each trial ---
    # Pre-compute tone onset relative to go cue for each valid trial
    tone_onsets_rel = np.zeros(n_valid_trials, dtype=np.float64)
    for t, trial_idx in enumerate(valid_trial_indices):
        gc = go_cue_times_all[trial_idx]
        candidates = sample_start_times[sample_start_times <= gc + 0.01]
        if len(candidates) > 0:
            tone_onsets_rel[t] = candidates[-1] - gc
        else:
            tone_onsets_rel[t] = -1.85  # fallback

    # Pre-compute photostim binary for each valid trial
    photostim_all = np.zeros((n_valid_trials, n_bins), dtype=np.float32)
    if len(ps_start_times) > 0:
        for t in range(n_valid_trials):
            gc = go_cue_times[t]
            for ps_s, ps_e in zip(ps_start_times, ps_stop_times):
                ps_s_rel = ps_s - gc
                ps_e_rel = ps_e - gc
                if ps_s_rel < bin_edges[-1] and ps_e_rel > bin_edges[0]:
                    photostim_all[t] += ((bin_centers >= ps_s_rel) & (bin_centers < ps_e_rel)).astype(np.float32)
        photostim_all = np.clip(photostim_all, 0, 1)

    # Pre-compute choice, outcome, early_lick for valid trials
    choices = np.zeros(n_valid_trials, dtype=np.float32)
    outcomes = np.zeros(n_valid_trials, dtype=np.float32)
    early_vals = np.zeros(n_valid_trials, dtype=np.float32)

    outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}

    for t, trial_idx in enumerate(valid_trial_indices):
        instr = trial_instruction[trial_idx]
        out = outcome_arr[trial_idx]

        if out == 'hit':
            choices[t] = 0 if instr == 'left' else 1
        elif out == 'miss':
            choices[t] = 1 if instr == 'left' else 0
        else:
            choices[t] = 2  # no lick

        outcomes[t] = outcome_map.get(out, 0)
        early_vals[t] = 1 if early_lick_arr[trial_idx] == 'early' else 0

    # --- Assemble per-trial lists ---
    neural_trials = []
    input_trials = []
    output_trials = []

    for t in range(n_valid_trials):
        neural_trials.append(fr_all[:, t, :])  # (n_good, n_bins)

        time_from_tone = (bin_centers - tone_onsets_rel[t]).astype(np.float32)
        input_data = np.stack([time_from_tone, photostim_all[t]], axis=0)
        input_trials.append(input_data)

        output_data = np.stack([
            np.full(n_bins, int(choices[t]), dtype=np.int64),
            np.full(n_bins, int(outcomes[t]), dtype=np.int64),
            np.full(n_bins, int(early_vals[t]), dtype=np.int64),
            tongue_disc[t].astype(np.int64)
        ], axis=0)
        output_trials.append(output_data)

    io.close()

    t1 = time.time()
    print(f'  {os.path.basename(nwb_path)}: {n_good} units, {n_valid_trials} trials, '
          f'{t1-t0:.1f}s (read:{t_read-t0:.1f}s spike:{t_spike-t_read:.1f}s '
          f'tongue:{t_tongue-t_spike:.1f}s assemble:{t1-t_tongue:.1f}s)')

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'subject_id': subject_id,
        'subject_desc': subject_desc,
        'unit_regions': unit_regions,
        'n_good': n_good,
        'n_trials': n_valid_trials,
        'nwb_path': nwb_path,
        'tongue_y_p40': tongue_y_p40,
        'tongue_y_p60': tongue_y_p60,
    }


def show_processing_plots(session_result, bin_edges, session_idx, nwb_path):
    """Generate processing visualization plots for one session."""
    if session_result is None:
        return

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    n_trials = len(session_result['neural'])
    n_neurons = session_result['n_good']

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle(f'Processing: {os.path.basename(nwb_path)}\n'
                 f'{n_neurons} neurons, {n_trials} trials', fontsize=12)

    if n_trials > 0:
        # Trial-averaged firing rate heatmap
        avg_fr = np.mean(np.stack(session_result['neural'][:50], axis=0), axis=0)
        ax = axes[0, 0]
        im = ax.imshow(avg_fr[:min(50, n_neurons)], aspect='auto',
                       extent=[bin_centers[0], bin_centers[-1], 0, min(50, n_neurons)])
        ax.axvline(0, color='r', linestyle='--', label='Go cue')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Neuron')
        ax.set_title('Trial-avg FR (first 50 neurons)')
        plt.colorbar(im, ax=ax, label='Hz')

        # Single trial firing rate
        ax = axes[0, 1]
        im = ax.imshow(session_result['neural'][0][:min(50, n_neurons)], aspect='auto',
                       extent=[bin_centers[0], bin_centers[-1], 0, min(50, n_neurons)])
        ax.axvline(0, color='r', linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Neuron')
        ax.set_title('Single trial FR (trial 0)')
        plt.colorbar(im, ax=ax, label='Hz')

        # FR histogram
        all_fr = np.concatenate([t.flatten() for t in session_result['neural']])
        ax = axes[0, 2]
        ax.hist(all_fr, bins=50, range=(0, np.percentile(all_fr, 99)))
        ax.set_xlabel('Firing rate (Hz)')
        ax.set_title('FR distribution')

        # Time from tone onset
        ax = axes[1, 0]
        for i in range(min(5, n_trials)):
            ax.plot(bin_centers, session_result['input'][i][0], alpha=0.5)
        ax.axvline(0, color='r', linestyle='--', label='Go cue')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Time from tone (s)')
        ax.set_title('Input: time from tone onset')

        # Photostim
        ax = axes[1, 1]
        for i in range(min(n_trials, 200)):
            if np.any(session_result['input'][i][1] > 0):
                ax.plot(bin_centers, session_result['input'][i][1], alpha=0.5)
                break
        ax.axvline(0, color='r', linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_title('Input: photostim')

        # Choice distribution
        choices = [int(session_result['output'][i][0, 0]) for i in range(n_trials)]
        outcomes = [int(session_result['output'][i][1, 0]) for i in range(n_trials)]
        early = [int(session_result['output'][i][2, 0]) for i in range(n_trials)]

        ax = axes[1, 2]
        ax.bar([0, 1, 2], [choices.count(0), choices.count(1), choices.count(2)])
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['left', 'right', 'no lick'])
        ax.set_title('Choice dist')

        ax = axes[2, 0]
        ax.bar([0, 1, 2], [outcomes.count(0), outcomes.count(1), outcomes.count(2)])
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['ignore', 'miss', 'hit'])
        ax.set_title('Outcome dist')

        ax = axes[2, 1]
        ax.bar([0, 1], [early.count(0), early.count(1)])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['no', 'yes'])
        ax.set_title('Early lick dist')

        # Tongue y time-varying
        ax = axes[2, 2]
        for i in range(min(n_trials, 50)):
            ty = session_result['output'][i][3]
            if np.any(ty < 3):
                ax.plot(bin_centers, ty, '.', markersize=2)
                ax.axvline(0, color='r', linestyle='--')
                ax.set_yticks([0, 1, 2, 3])
                ax.set_yticklabels(['<p40', 'p40-60', '>p60', 'invis'])
                ax.set_title(f'Tongue y (trial {i})')
                break

        # Brain region distribution
        ax = axes[3, 0]
        regions = session_result['unit_regions']
        unique_regions, counts = np.unique(regions, return_counts=True)
        sort_idx = np.argsort(-counts)
        ax.barh(range(len(unique_regions)), counts[sort_idx])
        ax.set_yticks(range(len(unique_regions)))
        ax.set_yticklabels(unique_regions[sort_idx], fontsize=7)
        ax.set_title('Brain regions')

        ax = axes[3, 1]
        ax.text(0.1, 0.5, f"p40: {session_result['tongue_y_p40']:.1f}\n"
                           f"p60: {session_result['tongue_y_p60']:.1f}",
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Tongue y percentiles')

    axes[3, 2].axis('off')
    fig.tight_layout()
    sess_name = os.path.basename(nwb_path).replace('.nwb', '')
    fig.savefig(f'/app/processing_{sess_name}.png', dpi=100)
    plt.close(fig)
    print(f'  Saved processing plot for {sess_name}')


def main():
    parser = argparse.ArgumentParser(description='Convert MAP NWB data to decoder format.')
    parser.add_argument('outfile', type=str, help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Generate processing plots')
    args = parser.parse_args()

    BEGIN_TIME = -2.5
    END_TIME = 1.5
    BIN_WIDTH = 0.05
    n_bins = int(round((END_TIME - BEGIN_TIME) / BIN_WIDTH))
    bin_edges = np.linspace(BEGIN_TIME, END_TIME, n_bins + 1)

    print(f'Parameters: [{BEGIN_TIME}, {END_TIME}]s, {n_bins} bins of {BIN_WIDTH*1000:.0f}ms')

    nwb_files = sorted(glob.glob('/app/data/sub-*/*.nwb'))
    print(f'Found {len(nwb_files)} NWB files')

    if args.sample:
        nwb_files = nwb_files[:2]
        print(f'Sample mode: {len(nwb_files)} sessions')

    all_results = []
    total_t0 = time.time()

    for i, nwb_path in enumerate(nwb_files):
        print(f'[{i+1}/{len(nwb_files)}] {os.path.basename(nwb_path)}')
        result = process_session(nwb_path, bin_edges, n_bins,
                                 show_processing=args.show_processing, session_idx=i)
        if result is not None:
            all_results.append(result)
            if args.show_processing and i < 2:
                show_processing_plots(result, bin_edges, i, nwb_path)
        else:
            print(f'  SKIPPED')

    total_t1 = time.time()
    print(f'\nProcessed {len(all_results)} sessions in {total_t1-total_t0:.1f}s '
          f'({(total_t1-total_t0)/len(all_results):.1f}s/session)')

    # --- Build output ---
    print('Building output...')

    subject_ids = sorted(set(r['subject_id'] for r in all_results))
    subjects = [str(sid) for sid in subject_ids]
    subject_id_to_idx = {sid: i for i, sid in enumerate(subject_ids)}

    all_regions = set()
    for r in all_results:
        all_regions.update(r['unit_regions'])
    all_regions.discard('Unknown')
    brain_regions = sorted(all_regions)
    if any('Unknown' in r['unit_regions'] for r in all_results):
        brain_regions.append('Unknown')
    region_to_idx = {r: i for i, r in enumerate(brain_regions)}

    neural = []
    inputs = []
    outputs = []
    subject_idx = []
    brain_region_idx = []
    total_neurons = 0
    total_trials = 0

    for r in all_results:
        neural.append(r['neural'])
        inputs.append(r['input'])
        outputs.append(r['output'])
        subject_idx.append(subject_id_to_idx[r['subject_id']])
        region_indices = np.array([region_to_idx[reg] for reg in r['unit_regions']], dtype=np.int64)
        brain_region_idx.append(region_indices)
        total_neurons += r['n_good']
        total_trials += r['n_trials']

    subject_idx = np.array(subject_idx, dtype=np.int64)

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
            ['left', 'right', 'no_lick'],
            ['ignore', 'miss', 'hit'],
            ['no', 'yes'],
            ['below_p40', 'p40_to_p60', 'above_p60', 'not_visible'],
        ],
        'metadata': {
            'task_description': 'Auditory delayed-response task: mice lick left or right based on tone frequency after a delay',
            'time_bin_size': BIN_WIDTH * 1000,
            'temporal_alignment_event': 'Go cue onset',
            'off_start': BEGIN_TIME,
            'off_end': END_TIME,
            'dataset': 'MAP (Mesoscale Activity Project)',
            'n_sessions': len(all_results),
            'n_subjects': len(subjects),
            'n_neurons_total': total_neurons,
            'n_trials_total': total_trials,
        }
    }

    # Print summary
    print(f'\n=== Summary ===')
    print(f'Sessions: {len(all_results)}')
    print(f'Subjects: {len(subjects)}')
    print(f'Total neurons: {total_neurons}')
    print(f'Total trials: {total_trials}')
    print(f'Brain regions ({len(brain_regions)}): {brain_regions}')

    region_counts = {}
    for r in all_results:
        for reg in r['unit_regions']:
            region_counts[reg] = region_counts.get(reg, 0) + 1
    for reg in brain_regions:
        print(f'  {reg}: {region_counts.get(reg, 0)}')

    all_choices = []
    all_outcomes = []
    all_early = []
    for sess in outputs:
        for trial in sess:
            all_choices.append(int(trial[0, 0]))
            all_outcomes.append(int(trial[1, 0]))
            all_early.append(int(trial[2, 0]))
    print(f'\nChoice: left={all_choices.count(0)}, right={all_choices.count(1)}, no_lick={all_choices.count(2)}')
    print(f'Outcome: ignore={all_outcomes.count(0)}, miss={all_outcomes.count(1)}, hit={all_outcomes.count(2)}')
    print(f'Early lick: no={all_early.count(0)}, yes={all_early.count(1)}')

    print(f'\nSaving to {args.outfile}...')
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    fsize = os.path.getsize(args.outfile)
    print(f'Saved ({fsize / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
