#!/usr/bin/env python3
"""
Convert MAP dataset (NWB format) to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--sample] [--full] [--show-processing]
"""

import os
import sys
import time
import pickle
import argparse
import json
import numpy as np
import pynwb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Constants
# ============================================================
BIN_WIDTH = 0.050  # 50 ms bins
TIME_BEFORE = 2.5  # seconds before go cue
TIME_AFTER = 1.5   # seconds after go cue
TONE_OFFSET = -1.85  # tone onset relative to go cue (sample=0.65s + delay=1.2s)
N_BINS = int((TIME_BEFORE + TIME_AFTER) / BIN_WIDTH)  # 80
BIN_EDGES = np.linspace(-TIME_BEFORE, TIME_AFTER, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
DLC_LIKELIHOOD_THRESH = 0.9  # DeepLabCut confidence threshold for tongue visibility
DATA_DIR = '/app/data'

# ============================================================
# Brain region mapping
# ============================================================
def get_high_level_region(anno_name, electrode_group_target):
    """Map CCF annotation name to high-level brain region.

    Based on Allen Brain Atlas CCF hierarchy and groupings used in Chen et al. 2024.
    """
    anno = anno_name.lower().strip()
    target = electrode_group_target.lower() if electrode_group_target else ''

    if not anno:
        return 'Unknown'

    # ALM: Secondary motor area from ALM-targeting probes
    if 'secondary motor area' in anno and 'alm' in target:
        return 'ALM'

    # Orbital cortex
    if 'orbital area' in anno:
        return 'Orbital'

    # Striatum (includes fundus of striatum)
    if anno in ('caudoputamen', 'striatum') or 'caudate' in anno or 'nucleus accumbens' in anno or 'fundus of striatum' in anno:
        return 'Striatum'

    # Pallidum (includes substantia innominata which is part of ventral pallidum)
    if 'pallidum' in anno or 'globus pallidus' in anno or 'substantia innominata' in anno:
        return 'Pallidum'

    # Hippocampus
    if any(x in anno for x in ['field ca', 'hippocampal', 'subiculum', 'dentate',
                                'postsubiculum', 'septofimbrial', 'triangular nucleus of septum']):
        return 'Hippocampus'

    # Thalamus (many nuclei)
    thalamus_keywords = ['thalamus', 'thalamic', 'geniculate', 'zona incerta',
                         'nucleus of the thalamus', 'reticular nucleus',
                         'paracentral nucleus', 'submedial nucleus',
                         'lateral dorsal nucleus', 'mediodorsal nucleus',
                         'anteroventral nucleus', 'ventral anterior-lateral',
                         'ventral medial nucleus', 'habenula',
                         'parafascicular nucleus', 'anteromedial nucleus',
                         'anterodorsal nucleus', 'rhomboid nucleus',
                         'perireunensis nucleus', 'fields of forel']
    if any(x in anno for x in thalamus_keywords):
        return 'Thalamus'

    # Hypothalamus (includes preoptic areas, mammillary, subparafascicular)
    if any(x in anno for x in ['hypothalamus', 'hypothalamic', 'preoptic',
                                'tuberomammillary', 'subparafascicular']):
        return 'Hypothalamus'

    # Midbrain
    midbrain_keywords = ['midbrain', 'superior colliculus', 'inferior colliculus',
                         'substantia nigra', 'ventral tegmental', 'red nucleus',
                         'pretectal', 'periaqueductal', 'nucleus of the optic tract',
                         'peripeduncular', 'terminal nucleus of the accessory optic',
                         'nucleus sagulum', 'nucleus of the lateral lemniscus']
    if any(x in anno for x in midbrain_keywords):
        return 'Midbrain'

    # Pons
    if any(x in anno for x in ['pons', 'pontine', 'parabrachial', 'locus ceruleus',
                                'koelliker-fuse']):
        return 'Pons'

    # Medulla (many brainstem nuclei)
    medulla_keywords = ['medulla', 'vestibular nucleus', 'spinal nucleus of the trigeminal',
                        'inferior olivary', 'facial motor nucleus', 'hypoglossal',
                        'nucleus of the solitary', 'dorsal motor nucleus of the vagus',
                        'nucleus raphe', 'external cuneate', 'parasolitary',
                        'parapyramidal', 'nucleus of roller', 'nucleus x']
    if any(x in anno for x in medulla_keywords):
        return 'Medulla'

    # Cerebellum (lobules, nuclei, etc.)
    cerebellum_keywords = ['cerebellum', 'cerebellar', 'lobule', 'lobules',
                           'simple lobule', 'nodulus', 'copula', 'uvula',
                           'paramedian lobule', 'declive', 'fastigial',
                           'interposed nucleus', 'crus 1', 'crus 2',
                           'pyramus', 'lingula', 'dentate nucleus']
    if any(x in anno for x in cerebellum_keywords):
        return 'Cerebellum'

    # Olfactory (includes piriform area)
    if any(x in anno for x in ['olfactory', 'taenia tecta', 'piriform']):
        return 'Olfactory'

    # CorticalSubplate (amygdala, bed nuclei, claustrum, etc.)
    subplate_keywords = ['bed nuclei', 'claustrum', 'endopiriform', 'lateral septal',
                         'amygdal', 'cortical subplate']
    if any(x in anno for x in subplate_keywords):
        return 'CorticalSubplate'

    # OtherCortex
    cortex_keywords = ['motor area', 'somatosensory', 'visual area', 'frontal pole',
                       'agranular insular', 'retrosplenial', 'prelimbic', 'infralimbic',
                       'insular', 'cortex', 'cingulate', 'auditory area',
                       'anterior cingulate', 'supplemental', 'visceral area',
                       'temporal association', 'ectorhinal', 'gustatory',
                       'entorhinal', 'perirhinal', 'dorsal peduncular']
    if any(x in anno for x in cortex_keywords):
        return 'OtherCortex'

    return 'Other'


# ============================================================
# Core processing functions (optimized)
# ============================================================

def compute_firing_rates_fast(spike_times_list, go_cue_times, n_neurons):
    """Compute binned firing rates - vectorized version.

    Returns list of (n_neurons, n_bins) arrays, one per trial.
    """
    n_trials = len(go_cue_times)
    trial_data = []

    for t_idx in range(n_trials):
        gc = go_cue_times[t_idx]
        t_start = gc - TIME_BEFORE
        t_end = gc + TIME_AFTER
        edges = BIN_EDGES + gc  # absolute time edges

        fr_matrix = np.zeros((n_neurons, N_BINS), dtype=np.float32)

        for n_idx in range(n_neurons):
            st = spike_times_list[n_idx]
            # Binary search for spikes in window
            i_start = np.searchsorted(st, t_start, side='left')
            i_end = np.searchsorted(st, t_end, side='left')
            spikes_in_window = st[i_start:i_end]

            if len(spikes_in_window) > 0:
                counts = np.histogram(spikes_in_window, bins=edges)[0]
                fr_matrix[n_idx, :] = counts / BIN_WIDTH

        trial_data.append(fr_matrix)

    return trial_data


def compute_tongue_y_fast(tongue_y, tongue_ts, tongue_lik, go_cue_times, p40, p60):
    """Compute discretized tongue y-position for all trials - vectorized.

    Returns list of (1, n_bins) int arrays.
    """
    n_trials = len(go_cue_times)
    results = []

    # Pre-sort tongue timestamps (should already be sorted)
    for t_idx in range(n_trials):
        gc = go_cue_times[t_idx]
        tongue_y_binned = np.full(N_BINS, 3, dtype=np.int64)  # default: not visible

        # Find tongue data range for this trial
        t_start = gc + BIN_EDGES[0]
        t_end = gc + BIN_EDGES[-1]
        idx_start = np.searchsorted(tongue_ts, t_start, side='left')
        idx_end = np.searchsorted(tongue_ts, t_end, side='right')

        if idx_end <= idx_start:
            results.append(tongue_y_binned.reshape(1, -1))
            continue

        # Get trial-relevant tongue data
        t_slice = tongue_ts[idx_start:idx_end]
        y_slice = tongue_y[idx_start:idx_end]
        l_slice = tongue_lik[idx_start:idx_end]

        # Compute bin assignments using digitize
        # Convert to go-cue-relative time
        t_rel = t_slice - gc
        bin_assignments = np.digitize(t_rel, BIN_EDGES) - 1  # 0-indexed bins

        # Process each bin
        for b in range(N_BINS):
            in_bin = bin_assignments == b
            if not np.any(in_bin):
                continue

            y_bin = y_slice[in_bin]
            l_bin = l_slice[in_bin]
            visible = l_bin >= DLC_LIKELIHOOD_THRESH

            if np.any(visible):
                mean_y = np.mean(y_bin[visible])
                if mean_y < p40:
                    tongue_y_binned[b] = 0
                elif mean_y <= p60:
                    tongue_y_binned[b] = 1
                else:
                    tongue_y_binned[b] = 2

        results.append(tongue_y_binned.reshape(1, -1))

    return results


def get_choice(trial_instruction, outcome):
    """Determine lick direction choice."""
    if outcome == 'ignore':
        return 2  # no_lick
    elif outcome == 'hit':
        return 0 if trial_instruction == 'left' else 1
    elif outcome == 'miss':
        return 1 if trial_instruction == 'left' else 0
    return 2


def compute_photostim_input_fast(photostim_starts, photostim_stops, go_cue_times):
    """Compute photostim binary input for all trials - vectorized.

    Returns list of (n_bins,) float32 arrays.
    """
    n_trials = len(go_cue_times)
    results = []

    if photostim_starts is None or len(photostim_starts) == 0:
        for _ in range(n_trials):
            results.append(np.zeros(N_BINS, dtype=np.float32))
        return results

    for t_idx in range(n_trials):
        gc = go_cue_times[t_idx]
        photostim_binary = np.zeros(N_BINS, dtype=np.float32)

        # Find photostim events that overlap with this trial window
        trial_start = gc - TIME_BEFORE
        trial_end = gc + TIME_AFTER

        for ps_start, ps_stop in zip(photostim_starts, photostim_stops):
            if ps_stop < trial_start or ps_start > trial_end:
                continue
            # Mark bins where photostim is active
            ps_start_rel = ps_start - gc
            ps_stop_rel = ps_stop - gc
            active = (BIN_CENTERS >= ps_start_rel) & (BIN_CENTERS < ps_stop_rel)
            photostim_binary[active] = 1.0

        results.append(photostim_binary)

    return results


def process_session(nwb_path, subject_id, show_processing=False, session_idx=0):
    """Process a single NWB session."""
    t0 = time.time()

    io = pynwb.NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()
    session_id = nwb.identifier

    # === Get units ===
    units = nwb.units
    classification = np.array(units['classification'][:])
    good_mask = classification == 'good'
    n_good = int(np.sum(good_mask))

    if n_good == 0:
        print(f"  Skipping {session_id}: 0 good units")
        io.close()
        return None

    good_indices = np.where(good_mask)[0]

    # Get spike times for good units (sorted for searchsorted)
    spike_times_list = []
    for i in good_indices:
        st = np.sort(np.array(units['spike_times'][i], dtype=np.float64))
        spike_times_list.append(st)

    # Get brain region info
    anno_names = []
    electrode_group_targets = []
    for i in good_indices:
        anno = str(units['anno_name'][i])
        anno_names.append(anno)
        eg = units['electrode_group'][i]
        try:
            loc = json.loads(eg.location)
            target = loc.get('brain_regions', '')
        except:
            target = ''
        electrode_group_targets.append(target)

    regions = [get_high_level_region(a, t) for a, t in zip(anno_names, electrode_group_targets)]

    t_units = time.time()
    print(f"  Units loaded: {n_good} good ({t_units - t0:.1f}s)")

    # === Get trials ===
    trials = nwb.trials
    n_trials_total = len(trials)
    trial_instructions = trials['trial_instruction'][:]
    outcomes = trials['outcome'][:]
    early_licks = trials['early_lick'][:]
    auto_water = np.array(trials['auto_water'][:])
    free_water = np.array(trials['free_water'][:])

    # Filter: exclude auto_water and free_water
    trial_mask = (auto_water == 0) & (free_water == 0)
    valid_trial_indices = np.where(trial_mask)[0]
    n_valid = len(valid_trial_indices)

    if n_valid < 2:
        print(f"  Skipping {session_id}: only {n_valid} valid trials")
        io.close()
        return None

    # Get go cue times
    be = nwb.acquisition['BehavioralEvents']
    go_start_times = be.time_series['go_start_times'].timestamps[:]
    go_cues_valid = go_start_times[valid_trial_indices]

    # Get photostim times
    photostim_starts = None
    photostim_stops = None
    if 'photostim_start_times' in be.time_series:
        photostim_starts = be.time_series['photostim_start_times'].timestamps[:]
        photostim_stops = be.time_series['photostim_stop_times'].timestamps[:]

    # Get tongue tracking
    bts = nwb.acquisition['BehavioralTimeSeries']
    has_tongue = 'Camera0_side_TongueTracking' in bts.time_series

    t_load = time.time()
    print(f"  Data loaded: {n_valid}/{n_trials_total} trials ({t_load - t_units:.1f}s)")

    # === Compute firing rates ===
    trial_neural = compute_firing_rates_fast(spike_times_list, go_cues_valid, n_good)
    t_fr = time.time()
    print(f"  Firing rates ({t_fr - t_load:.1f}s)")

    # === Compute inputs ===
    # Time from tone onset (same for all trials)
    time_from_tone = (BIN_CENTERS - TONE_OFFSET).astype(np.float32)

    # Photostim
    photostim_inputs = compute_photostim_input_fast(photostim_starts, photostim_stops, go_cues_valid)

    trial_inputs = []
    for t_idx in range(n_valid):
        input_data = np.stack([time_from_tone, photostim_inputs[t_idx]], axis=0)  # (2, N_BINS)
        trial_inputs.append(input_data)

    t_inp = time.time()
    print(f"  Inputs ({t_inp - t_fr:.1f}s)")

    # === Compute tongue y ===
    p40, p60 = 0.0, 0.0
    if has_tongue:
        tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
        tongue_data_all = tongue_ts_obj.data[:]
        tongue_timestamps_all = tongue_ts_obj.timestamps[:]
        tongue_y_all = tongue_data_all[:, 1]
        tongue_lik_all = tongue_data_all[:, 2]

        # Session percentiles (only visible frames)
        visible_mask = tongue_lik_all >= DLC_LIKELIHOOD_THRESH
        if np.sum(visible_mask) >= 10:
            visible_y = tongue_y_all[visible_mask]
            p40 = float(np.percentile(visible_y, 40))
            p60 = float(np.percentile(visible_y, 60))

        trial_tongue = compute_tongue_y_fast(
            tongue_y_all, tongue_timestamps_all, tongue_lik_all,
            go_cues_valid, p40, p60
        )
    else:
        trial_tongue = [np.full((1, N_BINS), 3, dtype=np.int64) for _ in range(n_valid)]

    t_tongue = time.time()
    print(f"  Tongue ({t_tongue - t_inp:.1f}s)")

    # === Compute per-trial outputs ===
    outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
    early_map = {'no early': 0, 'early': 1}

    trial_outputs = []
    for t_idx, trial_idx in enumerate(valid_trial_indices):
        instruction = trial_instructions[trial_idx]
        outcome = outcomes[trial_idx]

        choice_val = get_choice(instruction, outcome)
        outcome_val = outcome_map.get(outcome, 0)
        early_val = early_map.get(early_licks[trial_idx], 0)

        # Build output: (4, N_BINS) - first 3 per-trial (constant), last time-varying
        output_combined = np.zeros((4, N_BINS), dtype=np.int64)
        output_combined[0, :] = choice_val
        output_combined[1, :] = outcome_val
        output_combined[2, :] = early_val
        output_combined[3, :] = trial_tongue[t_idx][0, :]

        trial_outputs.append(output_combined)

    # === Show processing plots ===
    if show_processing:
        _plot_processing(session_id, session_idx, trial_neural, trial_inputs,
                        trial_outputs, go_cues_valid, spike_times_list,
                        tongue_y_all if has_tongue else None,
                        tongue_timestamps_all if has_tongue else None,
                        tongue_lik_all if has_tongue else None,
                        p40, p60, regions)

    io.close()

    t_end = time.time()
    print(f"  Done: {n_good} neurons, {n_valid} trials ({t_end - t0:.1f}s)")

    return {
        'neural': trial_neural,
        'input': trial_inputs,
        'output': trial_outputs,
        'subject_id': subject_id,
        'session_id': session_id,
        'regions': regions,
        'n_neurons': n_good,
        'n_trials': n_valid,
    }


def _plot_processing(session_id, session_idx, trial_neural, trial_inputs, trial_outputs,
                     go_cue_times, spike_times_list, tongue_y, tongue_ts, tongue_lik,
                     p40, p60, regions):
    """Plot processing steps for visual verification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 2, figsize=(20, 16))
    fig.suptitle(f'Processing: {session_id}', fontsize=14)

    n_trials = len(trial_neural)
    show_trials = [min(5, n_trials-1), min(n_trials//2, n_trials-1)]

    for col, t_idx in enumerate(show_trials):
        gc = go_cue_times[t_idx]

        # Row 0: Neural heatmap
        ax = axes[0, col]
        neural = trial_neural[t_idx]
        n_show = min(30, neural.shape[0])
        ax.imshow(neural[:n_show, :], aspect='auto', extent=[-TIME_BEFORE, TIME_AFTER, n_show, 0])
        ax.axvline(0, color='r', linestyle='--', label='Go cue')
        ax.axvline(TONE_OFFSET, color='g', linestyle='--', label='Tone onset')
        ax.set_title(f'Trial {t_idx}: Neural (top {n_show})')
        ax.set_ylabel('Neuron')
        if col == 0: ax.legend(fontsize=8)

        # Row 1: Time from tone onset
        ax = axes[1, col]
        inp = trial_inputs[t_idx]
        ax.plot(BIN_CENTERS, inp[0, :], 'b-')
        ax.axvline(0, color='r', linestyle='--')
        ax.axvline(TONE_OFFSET, color='g', linestyle='--')
        ax.set_ylabel('Time from tone (s)')
        ax.set_title('Input: time from tone onset')

        # Row 2: Photostim
        ax = axes[2, col]
        ax.plot(BIN_CENTERS, inp[1, :], 'r-')
        ax.axvline(0, color='r', linestyle='--')
        ax.set_ylabel('Photostim on')
        ax.set_ylim(-0.1, 1.1)
        ax.set_title('Input: photostim')

        # Row 3: Output values
        ax = axes[3, col]
        out = trial_outputs[t_idx]
        ax.plot(BIN_CENTERS, out[3, :], 'k-', label='Tongue Y')
        ax.axvline(0, color='r', linestyle='--')
        ax.set_ylim(-0.5, 3.5)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['low', 'mid', 'high', 'not_vis'])
        choice_nm = ['left', 'right', 'no_lick']
        outcome_nm = ['ignore', 'miss', 'hit']
        early_nm = ['no', 'yes']
        ax.set_title(f'choice={choice_nm[out[0,0]]}, outcome={outcome_nm[out[1,0]]}, early={early_nm[out[2,0]]}')

        # Row 4: Raw tongue tracking
        ax = axes[4, col]
        if tongue_y is not None:
            t_start = gc - TIME_BEFORE
            t_end = gc + TIME_AFTER
            idx_s = np.searchsorted(tongue_ts, t_start)
            idx_e = np.searchsorted(tongue_ts, t_end)
            t_rel = tongue_ts[idx_s:idx_e] - gc
            y_vals = tongue_y[idx_s:idx_e]
            lik = tongue_lik[idx_s:idx_e]
            vis = lik >= DLC_LIKELIHOOD_THRESH
            ax.scatter(t_rel[vis], y_vals[vis], s=1, c='blue', alpha=0.5, label='visible')
            ax.scatter(t_rel[~vis], y_vals[~vis], s=1, c='red', alpha=0.2, label='not vis')
            ax.axhline(p40, color='orange', linestyle='--', label=f'p40={p40:.1f}')
            ax.axhline(p60, color='green', linestyle='--', label=f'p60={p60:.1f}')
            ax.legend(fontsize=7)
        ax.axvline(0, color='r', linestyle='--')
        ax.set_xlabel('Time from go cue (s)')
        ax.set_ylabel('Tongue Y (px)')
        ax.set_title('Raw tongue Y')

    plt.tight_layout()
    plt.savefig(f'/app/processing_{session_idx}.png', dpi=100)
    plt.close()
    print(f"  Plot saved: processing_{session_idx}.png")


# ============================================================
# Main
# ============================================================

def get_all_nwb_files(data_dir, sample=False):
    """Get list of all NWB files."""
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    all_files = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for f in nwb_files:
            all_files.append((subj, os.path.join(subj_dir, f)))

    if sample:
        selected = []
        seen_subjects = set()
        for subj, fpath in all_files:
            if subj not in seen_subjects and len(selected) < 2:
                selected.append((subj, fpath))
                seen_subjects.add(subj)
        return selected
    return all_files


def main():
    parser = argparse.ArgumentParser(description='Convert MAP dataset to decoder format')
    parser.add_argument('outfile', type=str, help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Generate processing plots')
    args = parser.parse_args()

    sample_mode = args.sample

    print(f"=== MAP Dataset Conversion ===")
    print(f"Output: {args.outfile}")
    print(f"Mode: {'sample (2 sessions)' if sample_mode else 'full'}")
    print(f"Bins: {N_BINS} x {BIN_WIDTH*1000:.0f}ms = [{-TIME_BEFORE}, {TIME_AFTER}]s")
    print()

    nwb_files = get_all_nwb_files(DATA_DIR, sample=sample_mode)
    print(f"Found {len(nwb_files)} NWB files to process")

    session_data_list = []
    all_regions_set = set()
    t_start = time.time()

    for file_idx, (subj, fpath) in enumerate(nwb_files):
        print(f"\n[{file_idx+1}/{len(nwb_files)}] {os.path.basename(fpath)}")
        result = process_session(fpath, subj,
                                 show_processing=args.show_processing and file_idx < 2,
                                 session_idx=file_idx)
        if result is None:
            continue
        session_data_list.append(result)
        for r in result['regions']:
            all_regions_set.add(r)

        # Estimate remaining time
        elapsed = time.time() - t_start
        avg_per_session = elapsed / (file_idx + 1)
        remaining = avg_per_session * (len(nwb_files) - file_idx - 1)
        print(f"  ETA: {remaining/60:.1f} min remaining")

    t_process = time.time()
    print(f"\n=== Processing complete: {len(session_data_list)} sessions in {t_process - t_start:.1f}s ===")

    # Build brain region list
    brain_regions = sorted(list(all_regions_set))
    region_to_idx = {r: i for i, r in enumerate(brain_regions)}

    # Build subject list
    seen_subjects = {}
    subject_list = []
    for sd in session_data_list:
        subj = sd['subject_id']
        if subj not in seen_subjects:
            seen_subjects[subj] = len(seen_subjects)
            subject_list.append(subj)

    # Assemble final data
    all_neural = []
    all_input = []
    all_output = []
    all_subject_idx = []
    all_brain_region_idx = []
    all_session_ids = []

    for sd in session_data_list:
        all_neural.append(sd['neural'])
        all_input.append(sd['input'])
        all_output.append(sd['output'])
        all_subject_idx.append(seen_subjects[sd['subject_id']])
        all_brain_region_idx.append(
            np.array([region_to_idx[r] for r in sd['regions']], dtype=np.int64)
        )
        all_session_ids.append(sd['session_id'])

    # Summary
    total_neurons = sum(sd['n_neurons'] for sd in session_data_list)
    total_trials = sum(sd['n_trials'] for sd in session_data_list)
    print(f"\nDataset summary:")
    print(f"  Subjects: {len(subject_list)}")
    print(f"  Sessions: {len(session_data_list)}")
    print(f"  Total neurons: {total_neurons}")
    print(f"  Total trials: {total_trials}")
    print(f"  Mean neurons/session: {total_neurons/len(session_data_list):.1f}")
    print(f"  Mean trials/session: {total_trials/len(session_data_list):.1f}")
    print(f"  Brain regions: {brain_regions}")

    # Region neuron counts
    print(f"\nNeuron counts by region:")
    region_counts = {r: 0 for r in brain_regions}
    for sd in session_data_list:
        for r in sd['regions']:
            region_counts[r] += 1
    for r in sorted(region_counts.keys(), key=lambda x: -region_counts[x]):
        print(f"  {r}: {region_counts[r]}")

    # Output distributions
    print(f"\nOutput distributions:")
    for out_idx, out_name in enumerate(['choice', 'outcome', 'early_lick']):
        counts = {}
        for sd in session_data_list:
            for trial_out in sd['output']:
                val = int(trial_out[out_idx, 0])
                counts[val] = counts.get(val, 0) + 1
        total = sum(counts.values())
        print(f"  {out_name}: " + ", ".join(f"{v}: {c} ({c/total*100:.1f}%)" for v, c in sorted(counts.items())))

    # Build data dict
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': subject_list,
        'subject_idx': np.array(all_subject_idx, dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': all_brain_region_idx,
        'input_names': ['time_from_tone_onset', 'photostim_on'],
        'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y'],
        'output_values': [
            ['left', 'right', 'no_lick'],
            ['ignore', 'miss', 'hit'],
            ['no', 'yes'],
            ['low', 'mid', 'high', 'not_visible'],
        ],
        'metadata': {
            'task_description': 'Auditory delayed response task: mice hear tones (3kHz or 12kHz) during sample epoch, then after 1.2s delay lick left or right port',
            'time_bin_size': BIN_WIDTH * 1000,  # 50 ms
            'temporal_alignment_event': 'Go cue onset',
            'off_start': -TIME_BEFORE,
            'off_end': TIME_AFTER,
            'session_ids': all_session_ids,
            'bin_centers': BIN_CENTERS.tolist(),
            'tone_offset_from_go_cue': TONE_OFFSET,
        }
    }

    print(f"\nSaving to {args.outfile}...")
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    file_size = os.path.getsize(args.outfile) / (1024**2)
    print(f"Saved: {file_size:.1f} MB")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
