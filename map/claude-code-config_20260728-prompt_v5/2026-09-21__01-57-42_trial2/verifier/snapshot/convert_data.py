#!/usr/bin/env python3
"""
Convert MAP dataset (NWB files) to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

import argparse
import glob
import os
import pickle
import sys
import time

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────── Constants ───────────────────────
BIN_WIDTH = 0.05       # 50 ms bins
T_START = -2.5         # seconds before go cue
T_END = 1.5            # seconds after go cue
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2

TONGUE_LIKELIHOOD_THRESH = 0.9  # threshold for tongue visibility

# Major brain region mapping
# Maps detailed CCF annotation prefixes to major regions
MAJOR_REGION_MAP = {
    'Anterior lateral motor area': 'ALM',
    'Primary motor area': 'MOp',
    'Secondary motor area': 'MOs',
    'Frontal pole': 'FRP',
    'Orbital area': 'ORB',
    'Agranular insular area': 'AI',
    'Primary somatosensory area': 'SSp',
    'Supplemental somatosensory area': 'SSs',
    'Gustatory areas': 'GU',
    'Visceral area': 'VISC',
    'Retrosplenial area': 'RSP',
    'Posterior parietal association areas': 'PTLp',
    'Temporal association areas': 'TEa',
    'Perirhinal area': 'PERI',
    'Ectorhinal area': 'ECT',
    'Entorhinal area': 'ENT',
    'Piriform area': 'PIR',
    'Prelimbic area': 'PL',
    'Infralimbic area': 'ILA',
    'Anterior cingulate area': 'ACA',
    'Caudoputamen': 'CP',
    'Nucleus accumbens': 'ACB',
    'Fundus of striatum': 'FS',
    'Pallidum': 'PAL',
    'Globus pallidus': 'GP',
    'Substantia innominata': 'SI',
    'Bed nuclei of the stria terminalis': 'BST',
    'Field CA1': 'CA1',
    'Field CA2': 'CA2',
    'Field CA3': 'CA3',
    'Dentate gyrus': 'DG',
    'Subiculum': 'SUB',
    'Hypothalamus': 'HY',
    'Zona incerta': 'ZI',
    'Lateral hypothalamic area': 'LHA',
    'Subthalamic nucleus': 'STN',
}

# Broader grouping for thalamus and other areas
THALAMUS_KEYWORDS = [
    'Thalamus', 'thalamus', 'Ventral anterior-lateral',
    'Ventral posteromedial', 'Ventral posterolateral',
    'Posterior complex', 'Lateral posterior', 'Lateral dorsal',
    'Mediodorsal', 'Lateral habenula', 'Medial habenula',
    'Reticular nucleus', 'Ventral medial', 'Anterodorsal',
    'Anteromedial', 'Anteroventral', 'Central medial',
    'Central lateral', 'Paracentral', 'Parafascicular',
    'Reuniens', 'Rhomboid', 'Submedial', 'Paraventricular',
    'Interanterodorsal', 'Intermediodorsal',
    'geniculate', 'Geniculate',
    'Peripeduncular',
]

MIDBRAIN_KEYWORDS = [
    'Midbrain', 'midbrain', 'Superior colliculus',
    'Inferior colliculus', 'Periaqueductal',
    'Substantia nigra', 'Ventral tegmental',
    'Red nucleus', 'Interpeduncular',
    'Anterior pretectal', 'Medial pretectal',
    'Nucleus of the optic tract', 'Olivary pretectal',
    'Posterior pretectal', 'Precommissural',
    'Dorsal nucleus of the lateral lemniscus',
    'Nucleus of the lateral lemniscus',
    'Pedunculopontine', 'Cuneiform',
    'Nucleus of Darkschewitsch', 'Edinger-Westphal',
    'Trochlear', 'Oculomotor',
    'Parabigeminal', 'Sagulum',
    'Anterior tegmental', 'Laterodorsal tegmental',
    'Tegmental reticular',
    'Posterior commissure',
    'Retroambiguus',
]

PONS_KEYWORDS = [
    'Pons', 'pons', 'Pontine',
    'Parabrachial', 'Barrington',
    'Dorsal tegmental', 'Laterodorsal tegmental',
    'Pontine reticular', 'Tegmental reticular',
    'Superior central', 'Locus ceruleus',
    'Superior olivary', 'Nucleus raphe pontis',
    'Supratrigeminal',
    'Koelliker-Fuse', 'Principal sensory nucleus of the trigeminal',
    'Motor nucleus of the trigeminal',
]

MEDULLA_KEYWORDS = [
    'Medulla', 'medulla', 'Inferior olivary',
    'Nucleus of the solitary tract', 'Dorsal column nuclei',
    'Cuneate', 'Gracile', 'Spinal nucleus of the trigeminal',
    'Facial motor', 'Abducens', 'Hypoglossal',
    'Dorsal motor nucleus', 'Ambiguus',
    'Lateral reticular', 'Magnocellular reticular',
    'Parvicellular reticular', 'Intermediate reticular',
    'Gigantocellular reticular', 'Paragigantocellular',
    'Raphe', 'Vestibular', 'Cochlear',
    'Nucleus prepositus', 'Nucleus of the trapezoid',
    'Superior olivary',
    'External cuneate',
]

CEREBELLUM_KEYWORDS = [
    'Cerebellum', 'cerebellum', 'Cerebellar',
    'Purkinje', 'Lobul', 'Flocculus',
    'Fastigial', 'Interposed', 'Dentate nucleus',
    'Nodulus', 'Uvula', 'Vermis',
    'Paraflocculus', 'Copula',
    'Ansiform', 'Simple lobule',
    'Central lobule', 'Culmen', 'Decliv',
    'Folium', 'Tuber', 'Pyramid',
    'Lingula',
]

HIPPOCAMPUS_KEYWORDS = [
    'Hippocampus', 'hippocampus', 'Field CA',
    'Dentate gyrus', 'Subiculum', 'Entorhinal',
    'Fasciola cinerea', 'Indusium griseum',
]

OLFACTORY_KEYWORDS = [
    'Olfactory', 'olfactory', 'Piriform',
    'Taenia tecta', 'Dorsal peduncular',
    'Anterior olfactory', 'Main olfactory',
    'Accessory olfactory',
]

CORTICAL_SUBPLATE_KEYWORDS = [
    'Claustrum', 'Endopiriform',
    'Lateral amygdalar', 'Basolateral amygdalar',
    'Basomedial amygdalar', 'Posterior amygdalar',
    'Cortical amygdalar', 'Medial amygdalar',
    'Central amygdalar', 'Anterior amygdalar',
    'Intercalated', 'Amygdala',
]


def classify_brain_region(anno_name):
    """Classify a detailed CCF annotation into a major brain region."""
    if not anno_name or anno_name == '':
        return 'unknown'

    # Check specific mappings first
    for prefix, region in MAJOR_REGION_MAP.items():
        if anno_name.startswith(prefix):
            return region

    # Check keyword-based groups
    for kw in THALAMUS_KEYWORDS:
        if kw in anno_name:
            return 'TH'
    for kw in MIDBRAIN_KEYWORDS:
        if kw in anno_name:
            return 'MB'
    for kw in PONS_KEYWORDS:
        if kw in anno_name:
            return 'Pons'
    for kw in MEDULLA_KEYWORDS:
        if kw in anno_name:
            return 'MY'
    for kw in CEREBELLUM_KEYWORDS:
        if kw in anno_name:
            return 'CB'
    for kw in HIPPOCAMPUS_KEYWORDS:
        if kw in anno_name:
            return 'HPF'
    for kw in OLFACTORY_KEYWORDS:
        if kw in anno_name:
            return 'OLF'
    for kw in CORTICAL_SUBPLATE_KEYWORDS:
        if kw in anno_name:
            return 'CTXsp'

    # Cortex catch-all (layer annotations)
    if 'layer' in anno_name.lower() or 'Layer' in anno_name:
        return 'CTX_other'

    # Striatum catch-all
    if 'striatum' in anno_name.lower() or 'Striatum' in anno_name:
        return 'STR'

    return 'unknown'


def extract_unit_spike_times(spike_times_data, spike_times_index, unit_indices):
    """Extract spike times for multiple units at once.
    Returns list of arrays, one per unit."""
    starts = np.zeros(len(unit_indices), dtype=np.int64)
    ends = spike_times_index[unit_indices].astype(np.int64)
    for i, idx in enumerate(unit_indices):
        if idx == 0:
            starts[i] = 0
        else:
            starts[i] = int(spike_times_index[idx - 1])
    return [spike_times_data[starts[i]:ends[i]] for i in range(len(unit_indices))]


def compute_firing_rates_fast(unit_spike_times_list, go_times, bin_edges):
    """Compute firing rates for all units and trials.
    Optimized: iterate over units (outer) and vectorize across trials.
    Returns list of (n_units, n_bins) arrays, one per trial."""
    n_units = len(unit_spike_times_list)
    n_bins = len(bin_edges) - 1
    bin_width = bin_edges[1] - bin_edges[0]
    n_trials = len(go_times)

    # Pre-allocate result as 3D array, then split
    all_rates = np.zeros((n_units, n_trials, n_bins), dtype=np.float32)

    for ni in range(n_units):
        st = unit_spike_times_list[ni]
        if len(st) == 0:
            continue

        for ti in range(n_trials):
            go_t = go_times[ti]
            t_lo = go_t + bin_edges[0]
            t_hi = go_t + bin_edges[-1]
            i_lo = np.searchsorted(st, t_lo)
            i_hi = np.searchsorted(st, t_hi)
            if i_lo >= i_hi:
                continue
            # np.histogram is C-optimized and faster than searchsorted+add.at
            counts, _ = np.histogram(st[i_lo:i_hi], bins=bin_edges + go_t)
            all_rates[ni, ti] = counts

    all_rates /= bin_width

    # Split into list of per-trial arrays
    return [all_rates[:, ti, :] for ti in range(n_trials)]


# Task protocol defines fixed timing: sample epoch (0.65s) + delay (1.2s) = 1.85s before go cue.
# Early lick replays create extra sample_start events in the NWB data that confuse event-based
# matching (~12.5% of trials affected). Since the task timing is fixed, we use the protocol-defined
# offset for all trials.
TONE_ONSET_REL_GO = -1.85  # seconds before go cue


def get_photostim_intervals(f, n_trials, go_times, trial_start_times):
    """Get photostim intervals relative to go cue for each trial.
    Returns list of (onset_rel, offset_rel) or None for each trial."""
    photostim_onset = f['intervals']['trials']['photostim_onset'][:]
    photostim_dur = f['intervals']['trials']['photostim_duration'][:]

    intervals = []
    for i in range(n_trials):
        if photostim_onset[i] == b'N/A':
            intervals.append(None)
        else:
            onset_rel_trial = float(photostim_onset[i])
            dur = float(photostim_dur[i])
            # Convert from trial-relative to go-cue-relative
            onset_abs = onset_rel_trial + trial_start_times[i]
            onset_rel_go = onset_abs - go_times[i]
            offset_rel_go = onset_rel_go + dur
            intervals.append((onset_rel_go, offset_rel_go))
    return intervals


def compute_tongue_y_all_trials(tongue_ts, tongue_data, go_times, bin_edges,
                                 likelihood_thresh=TONGUE_LIKELIHOOD_THRESH):
    """Compute discretized tongue y-position for all trials using vectorized operations.
    Returns (tongue_discretized_list, p40, p60).
    Categories: 0=<40th, 1=40-60th, 2=>60th, 3=not visible
    """
    tongue_y = tongue_data[:, 1]
    tongue_lh = tongue_data[:, 2]
    n_bins = len(bin_edges) - 1
    n_trials = len(go_times)
    bin_width = bin_edges[1] - bin_edges[0]

    # Pre-compute bin assignments for tongue timestamps
    # For each trial, we need tongue data in [go_time + T_START, go_time + T_END]
    # Use searchsorted for fast indexing

    # First pass: collect visible y values for percentile computation
    visible_y_all = []
    # For each trial, find tongue data indices
    trial_tongue_ranges = []
    for ti in range(n_trials):
        go_t = go_times[ti]
        t_lo = go_t + bin_edges[0]
        t_hi = go_t + bin_edges[-1]
        i_lo = np.searchsorted(tongue_ts, t_lo)
        i_hi = np.searchsorted(tongue_ts, t_hi)
        trial_tongue_ranges.append((i_lo, i_hi))

        if i_lo < i_hi:
            vis_mask = tongue_lh[i_lo:i_hi] >= likelihood_thresh
            if vis_mask.any():
                visible_y_all.append(tongue_y[i_lo:i_hi][vis_mask])

    if len(visible_y_all) == 0:
        p40, p60 = None, None
        return [np.full(n_bins, 3, dtype=np.int64) for _ in range(n_trials)], p40, p60

    all_visible = np.concatenate(visible_y_all)
    p40 = np.percentile(all_visible, 40)
    p60 = np.percentile(all_visible, 60)

    # Second pass: discretize tongue y per trial and bin
    results = []
    for ti in range(n_trials):
        go_t = go_times[ti]
        i_lo, i_hi = trial_tongue_ranges[ti]
        trial_bins = np.full(n_bins, 3, dtype=np.int64)

        if i_lo >= i_hi:
            results.append(trial_bins)
            continue

        # Get tongue data for this trial
        trial_ts = tongue_ts[i_lo:i_hi]
        trial_y = tongue_y[i_lo:i_hi]
        trial_lh = tongue_lh[i_lo:i_hi]

        # Assign each tongue sample to a bin
        rel_ts = trial_ts - go_t
        bin_idx = np.searchsorted(bin_edges, rel_ts, side='right') - 1
        valid = (bin_idx >= 0) & (bin_idx < n_bins)

        if not valid.any():
            results.append(trial_bins)
            continue

        bin_idx_v = bin_idx[valid]
        trial_y_v = trial_y[valid]
        trial_lh_v = trial_lh[valid]

        # Compute mean y and mean likelihood per bin using bincount
        for b in range(n_bins):
            b_mask = bin_idx_v == b
            if not b_mask.any():
                continue
            avg_lh = trial_lh_v[b_mask].mean()
            if avg_lh >= likelihood_thresh:
                avg_y = trial_y_v[b_mask].mean()
                if avg_y < p40:
                    trial_bins[b] = 0
                elif avg_y <= p60:
                    trial_bins[b] = 1
                else:
                    trial_bins[b] = 2

        results.append(trial_bins)
    return results, p40, p60


def process_session(nwb_path, show_processing=False):
    """Process a single NWB session file.
    Returns session data dict or None if session should be skipped."""
    t0 = time.time()
    basename = os.path.basename(nwb_path)

    f = h5py.File(nwb_path, 'r')

    # ── Subject info ──
    subject_id = f['general']['subject']['subject_id'][()].decode()

    # ── Trial info ──
    n_trials_total = f['intervals']['trials']['id'].shape[0]
    outcome = np.array([x.decode() for x in f['intervals']['trials']['outcome'][:]])
    early_lick = np.array([x.decode() for x in f['intervals']['trials']['early_lick'][:]])
    trial_instruction = np.array([x.decode() for x in f['intervals']['trials']['trial_instruction'][:]])
    auto_water = f['intervals']['trials']['auto_water'][:]
    free_water = f['intervals']['trials']['free_water'][:]
    photostim_power = f['intervals']['trials']['photostim_power'][:]
    trial_start_times = f['intervals']['trials']['start_time'][:]

    # ── Session selection ──
    # The DANDI archive already contains sessions selected by the paper authors
    # (performance > 65%, >= 50 correct L and R trials). One session has no good units.
    is_auto_or_free = (auto_water == 1) | (free_water == 1)

    # Compute performance for metadata (on control non-early-lick trials)
    is_control = (photostim_power == b'N/A') & (early_lick == 'no early') & ~is_auto_or_free
    n_control_responded = np.sum((outcome[is_control] == 'hit') | (outcome[is_control] == 'miss'))
    n_correct = np.sum(outcome[is_control] == 'hit')
    performance = n_correct / n_control_responded if n_control_responded > 0 else 0

    # ── Trial filtering: remove auto_water and free_water ──
    trial_mask = ~is_auto_or_free
    trial_indices = np.where(trial_mask)[0]
    n_trials = len(trial_indices)

    if n_trials < 2:
        print(f"  SKIP {basename}: only {n_trials} valid trials")
        f.close()
        return None

    # ── Go cue times ──
    go_start_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
    assert len(go_start_times) == n_trials_total, f"Go cue count mismatch: {len(go_start_times)} vs {n_trials_total}"
    go_times = go_start_times[trial_indices]
    trial_starts = trial_start_times[trial_indices]

    # ── Good units ──
    classification = f['units']['classification'][:]
    good_mask = classification == b'good'
    good_indices = np.where(good_mask)[0]
    n_good = len(good_indices)

    if n_good == 0:
        print(f"  SKIP {basename}: no good units")
        f.close()
        return None

    # ── Brain region annotations ──
    anno_names = f['units']['anno_name'][:]
    good_annos = [anno_names[i].decode() if isinstance(anno_names[i], bytes) else str(anno_names[i])
                  for i in good_indices]
    good_regions = [classify_brain_region(a) for a in good_annos]

    # ── Spike times ──
    spike_times_data = f['units']['spike_times'][:]
    spike_times_index = f['units']['spike_times_index'][:]

    # Extract spike times for all good units once
    unit_spike_times = extract_unit_spike_times(spike_times_data, spike_times_index, good_indices)

    # Compute firing rates for all good units and trials
    neural_data = compute_firing_rates_fast(unit_spike_times, go_times, BIN_EDGES)

    # ── Inputs ──
    # Input 0: Time from tone onset (fixed task protocol timing)
    time_from_tone = BIN_CENTERS - TONE_ONSET_REL_GO  # same for all trials

    # Input 1: Photostim active
    photostim_intervals = get_photostim_intervals(f, n_trials_total, go_start_times, trial_start_times)

    input_data = []
    for ti, trial_idx in enumerate(trial_indices):
        go_t = go_times[ti]

        # Photostim binary time series
        photostim_ts = np.zeros(N_BINS, dtype=np.float32)
        ps_interval = photostim_intervals[trial_idx]
        if ps_interval is not None:
            onset_rel, offset_rel = ps_interval
            photostim_ts[(BIN_CENTERS >= onset_rel) & (BIN_CENTERS < offset_rel)] = 1.0

        trial_input = np.stack([time_from_tone.astype(np.float32), photostim_ts], axis=0)  # (2, n_bins)
        input_data.append(trial_input)

    # ── Outputs ──
    # Filtered trial data
    trial_outcomes = outcome[trial_indices]
    trial_early_lick = early_lick[trial_indices]
    trial_instructions = trial_instruction[trial_indices]

    # Tongue y position
    tongue_ts = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['timestamps'][:]
    tongue_data_raw = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['data'][:]

    tongue_discretized, p40, p60 = compute_tongue_y_all_trials(
        tongue_ts, tongue_data_raw, go_times, BIN_EDGES)

    output_data = []
    for ti in range(n_trials):
        # Choice: 0=left, 1=right, 2=no_lick
        out = trial_outcomes[ti]
        inst = trial_instructions[ti]
        if out == 'hit':
            choice = 0 if inst == 'left' else 1
        elif out == 'miss':
            choice = 1 if inst == 'left' else 0  # opposite of instruction
        else:  # ignore
            choice = 2

        # Outcome: 0=ignore, 1=miss, 2=hit
        outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[out]

        # Early lick: 0=no, 1=yes
        early_val = 0 if trial_early_lick[ti] == 'no early' else 1

        # Tongue y: time-varying (n_bins,)
        tongue_y_disc = tongue_discretized[ti]

        # Stack per-trial outputs with time-varying tongue
        # Per-trial values broadcast to all time points
        trial_output = np.stack([
            np.full(N_BINS, choice, dtype=np.int64),
            np.full(N_BINS, outcome_val, dtype=np.int64),
            np.full(N_BINS, early_val, dtype=np.int64),
            tongue_y_disc,
        ], axis=0)  # (4, n_bins)
        output_data.append(trial_output)

    f.close()

    elapsed = time.time() - t0
    print(f"  {basename}: {n_good} good units, {n_trials} trials, perf={performance:.1%}, time={elapsed:.1f}s")

    # ── Processing visualization ──
    if show_processing:
        plot_processing(basename, neural_data, input_data, output_data,
                       go_times, BIN_CENTERS, good_regions, trial_outcomes, trial_instructions)

    return {
        'subject_id': subject_id,
        'session_name': basename,
        'neural': neural_data,
        'input': input_data,
        'output': output_data,
        'brain_regions': good_regions,
        'n_good_units': n_good,
        'n_trials': n_trials,
        'performance': performance,
    }


def plot_processing(basename, neural_data, input_data, output_data,
                    go_times, bin_centers, regions, outcomes, instructions):
    """Plot processing steps for visual verification."""
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle(f'Processing: {basename}', fontsize=14)

    n_trials = len(neural_data)

    # 1. Neural activity heatmap for first trial
    ax = axes[0, 0]
    if n_trials > 0:
        trial_neural = neural_data[0]
        im = ax.imshow(trial_neural[:min(50, trial_neural.shape[0])], aspect='auto',
                       extent=[bin_centers[0], bin_centers[-1], 0, min(50, trial_neural.shape[0])],
                       cmap='viridis')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Neuron')
        ax.set_title('Neural: Trial 0')
        ax.axvline(0, color='r', linestyle='--', label='Go cue')
        plt.colorbar(im, ax=ax, label='FR (Hz)')

    # 2. Mean firing rate across trials
    ax = axes[0, 1]
    if n_trials > 0:
        all_rates = np.array([nd.mean(axis=0) for nd in neural_data])
        ax.plot(bin_centers, all_rates.mean(axis=0))
        ax.fill_between(bin_centers,
                        all_rates.mean(axis=0) - all_rates.std(axis=0),
                        all_rates.mean(axis=0) + all_rates.std(axis=0), alpha=0.3)
        ax.axvline(0, color='r', linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Mean FR (Hz)')
        ax.set_title('Population avg FR')

    # 3. Brain region distribution
    ax = axes[0, 2]
    from collections import Counter
    reg_counts = Counter(regions)
    regs = sorted(reg_counts.keys())
    ax.barh(range(len(regs)), [reg_counts[r] for r in regs])
    ax.set_yticks(range(len(regs)))
    ax.set_yticklabels(regs, fontsize=7)
    ax.set_xlabel('Count')
    ax.set_title('Brain regions')

    # 4. Time from tone onset for a few trials
    ax = axes[1, 0]
    for i in range(min(5, n_trials)):
        ax.plot(bin_centers, input_data[i][0], alpha=0.5, label=f'Trial {i}')
    ax.axvline(0, color='r', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Time from tone (s)')
    ax.set_title('Input: Time from tone onset')
    ax.legend(fontsize=6)

    # 5. Photostim for a few trials
    ax = axes[1, 1]
    for i in range(min(10, n_trials)):
        if input_data[i][1].max() > 0:
            ax.plot(bin_centers, input_data[i][1] + i * 0.1, alpha=0.7)
    ax.axvline(0, color='r', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_title('Input: Photostim active')

    # 6. Outcome distribution
    ax = axes[1, 2]
    outcome_vals = [od[1, 0] for od in output_data]  # per-trial outcome
    unique, counts = np.unique(outcome_vals, return_counts=True)
    ax.bar(unique, counts)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['ignore', 'miss', 'hit'])
    ax.set_title('Output: Outcome distribution')

    # 7. Choice distribution
    ax = axes[2, 0]
    choice_vals = [od[0, 0] for od in output_data]
    unique, counts = np.unique(choice_vals, return_counts=True)
    ax.bar(unique, counts)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['left', 'right', 'no_lick'])
    ax.set_title('Output: Choice distribution')

    # 8. Early lick distribution
    ax = axes[2, 1]
    early_vals = [od[2, 0] for od in output_data]
    unique, counts = np.unique(early_vals, return_counts=True)
    ax.bar(unique, counts)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['no', 'yes'])
    ax.set_title('Output: Early lick distribution')

    # 9. Tongue y discretized for a few trials
    ax = axes[2, 2]
    for i in range(min(5, n_trials)):
        ax.plot(bin_centers, output_data[i][3] + i * 0.1, alpha=0.5)
    ax.axvline(0, color='r', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_title('Output: Tongue y discretized')

    # 10. Trial count summary
    ax = axes[3, 0]
    ax.text(0.1, 0.8, f'Total trials: {n_trials}', transform=ax.transAxes, fontsize=12)
    ax.text(0.1, 0.6, f'Neurons: {neural_data[0].shape[0] if n_trials > 0 else 0}', transform=ax.transAxes, fontsize=12)
    ax.text(0.1, 0.4, f'Time bins: {N_BINS}', transform=ax.transAxes, fontsize=12)
    ax.set_title('Summary')
    ax.axis('off')

    # 11-12. Empty
    axes[3, 1].axis('off')
    axes[3, 2].axis('off')

    fig.tight_layout()
    session_id = basename.replace('.nwb', '')
    fig.savefig(f'processing_{session_id}.png', dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Convert MAP dataset to decoder format')
    parser.add_argument('outfile', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()

    if args.sample:
        args.full = False

    # Find all NWB files
    nwb_files = sorted(glob.glob('/app/data/sub-*/*.nwb'))
    print(f"Found {len(nwb_files)} NWB files")

    if args.sample:
        # Pick files from different subjects to ensure diversity
        # Use files that are likely to pass session criteria
        sample_files = []
        seen_subjects = set()
        for nf in nwb_files:
            subj = os.path.basename(os.path.dirname(nf))
            if subj not in seen_subjects:
                seen_subjects.add(subj)
                sample_files.append(nf)
            if len(sample_files) >= 5:  # try up to 5 to get at least 2 passing
                break
        nwb_files = sample_files
        print(f"Sample mode: processing up to {len(nwb_files)} files")

    # Process all sessions
    all_sessions = []
    t_total_start = time.time()

    for i, nwb_file in enumerate(nwb_files):
        print(f"[{i+1}/{len(nwb_files)}] Processing {os.path.basename(nwb_file)}...")
        try:
            result = process_session(nwb_file, show_processing=args.show_processing)
            if result is not None:
                all_sessions.append(result)
        except Exception as e:
            print(f"  ERROR processing {nwb_file}: {e}")
            import traceback
            traceback.print_exc()

    t_total = time.time() - t_total_start
    print(f"\nProcessed {len(all_sessions)} sessions in {t_total:.1f}s ({t_total/max(len(all_sessions),1):.1f}s/session)")

    if len(all_sessions) == 0:
        print("ERROR: No sessions processed successfully!")
        sys.exit(1)

    # ── Assemble output ──
    print("\nAssembling output data structure...")

    # Subjects
    all_subject_ids = sorted(set(s['subject_id'] for s in all_sessions))
    subject_to_idx = {sid: i for i, sid in enumerate(all_subject_ids)}

    # Brain regions
    all_region_names = sorted(set(
        r for s in all_sessions for r in s['brain_regions']
    ))
    region_to_idx = {r: i for i, r in enumerate(all_region_names)}

    # Build data structure
    neural_list = []
    input_list = []
    output_list = []
    subject_idx_list = []
    brain_region_idx_list = []

    for sess in all_sessions:
        neural_list.append(sess['neural'])
        input_list.append(sess['input'])
        output_list.append(sess['output'])
        subject_idx_list.append(subject_to_idx[sess['subject_id']])
        brain_region_idx_list.append(
            np.array([region_to_idx[r] for r in sess['brain_regions']], dtype=np.int64)
        )

    data = {
        'neural': neural_list,
        'input': input_list,
        'output': output_list,
        'subjects': all_subject_ids,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),
        'brain_regions': all_region_names,
        'brain_region_idx': brain_region_idx_list,
        'input_names': ['time_from_tone_onset', 'photostim_active'],
        'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
        'output_values': [
            ['left', 'right', 'no_lick'],
            ['ignore', 'miss', 'hit'],
            ['no', 'yes'],
            ['below_40pctl', '40_to_60pctl', 'above_60pctl', 'not_visible'],
        ],
        'metadata': {
            'task_description': 'Auditory delayed response task: mice hear tones, wait through delay, then lick left/right port to report instruction',
            'time_bin_size': BIN_WIDTH * 1000,  # 50 ms
            'temporal_alignment_event': 'Go cue onset',
            'off_start': T_START,
            'off_end': T_END,
            'bin_centers': BIN_CENTERS.tolist(),
            'session_info': [
                {
                    'session_name': s['session_name'],
                    'subject_id': s['subject_id'],
                    'n_good_units': s['n_good_units'],
                    'n_trials': s['n_trials'],
                    'performance': s['performance'],
                }
                for s in all_sessions
            ],
        },
    }

    # Print summary stats
    total_neurons = sum(s['n_good_units'] for s in all_sessions)
    total_trials = sum(s['n_trials'] for s in all_sessions)
    print(f"\n=== Summary ===")
    print(f"Subjects: {len(all_subject_ids)}")
    print(f"Sessions: {len(all_sessions)}")
    print(f"Total good neurons: {total_neurons}")
    print(f"Total trials: {total_trials}")
    print(f"Mean neurons/session: {total_neurons/len(all_sessions):.1f}")
    print(f"Mean trials/session: {total_trials/len(all_sessions):.1f}")
    print(f"Brain regions: {all_region_names}")

    # Save
    print(f"\nSaving to {args.outfile}...")
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = os.path.getsize(args.outfile)
    print(f"Saved {file_size / 1e6:.1f} MB")
    print("Done!")


if __name__ == '__main__':
    main()
