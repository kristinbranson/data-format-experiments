"""
Convert NWB data from the brain-wide neural activity dataset to the decoder format.

Reference: "Brain-wide analysis reveals movement encoding structured across and within brain areas"
Data: "Brain-wide neural activity underlying memory-guided movement"

Usage:
    python convert_data.py [--sample] [--output OUTPUT_PATH]

Options:
    --sample: Only process first 5 sessions (for quick testing)
    --output: Output pickle file path (default: converted_data.pkl)
"""

import numpy as np
import h5py
import glob
import pickle
import argparse
import os
import sys
from collections import OrderedDict

# === PARAMETERS ===
BIN_WIDTH = 0.05  # 50 ms bins
T_START = -2.5    # 2.5 s before go cue
T_END = 1.5       # 1.5 s after go cue
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80 bins
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2

# Session filtering criteria from methods
MIN_PERFORMANCE = 0.65  # > 65% correct on control trials
MIN_CORRECT_PER_DIRECTION = 50  # at least 50 correct left and 50 correct right

# === BRAIN REGION MAPPING ===
# Maps detailed Allen Brain Atlas annotation names to high-level region categories
# matching the 14 categories used in the reference code
REGION_MAPPING = {
    # ALM = Secondary motor area (Anterior Lateral Motor cortex)
    'ALM': [
        'Secondary motor area',
    ],
    # Other cortical areas
    'OtherCortex': [
        'Primary motor area',
        'Primary somatosensory area',
        'Supplemental somatosensory area',
        'Anterior cingulate area',
        'Prelimbic area',
        'Infralimbic area',
        'Retrosplenial area',
        'Primary visual area',
        'posteromedial visual area',
        'Anteromedial visual area',
        'Primary auditory area',
        'Dorsal auditory area',
        'Ventral auditory area',
        'Temporal association areas',
        'Gustatory areas',
        'Visceral area',
        'Frontal pole',
        'Dorsal peduncular area',
        'Perirhinal area',
        'Ectorhinal area',
        'Entorhinal area',
        'Agranular insular area',
    ],
    'Orbital': [
        'Orbital area',
    ],
    'Striatum': [
        'Caudoputamen',
        'Striatum',
        'Fundus of striatum',
        'Nucleus accumbens',
    ],
    'Pallidum': [
        'Pallidum',
        'Globus pallidus',
        'Substantia innominata',
        'Bed nuclei of the stria terminalis',
    ],
    'Thalamus': [
        'Mediodorsal nucleus of thalamus',
        'Mediodorsal nucleus of the thalamus',
        'Posterior complex of the thalamus',
        'Ventral anterior-lateral complex of the thalamus',
        'Lateral posterior nucleus of the thalamus',
        'Lateral dorsal nucleus of thalamus',
        'Ventral medial nucleus of the thalamus',
        'Ventral posteromedial nucleus of the thalamus',
        'Ventral posterolateral nucleus of the thalamus',
        'Central lateral nucleus of the thalamus',
        'Anteroventral nucleus of thalamus',
        'Reticular nucleus of the thalamus',
        'Central medial nucleus of the thalamus',
        'Posterior limiting nucleus of the thalamus',
        'Submedial nucleus of the thalamus',
        'Paracentral nucleus',
        'Parafascicular nucleus',
        'Intermediodorsal nucleus of the thalamus',
        'Thalamus',
        'Suprageniculate nucleus',
        'Lateral geniculate',
        'Dorsal part of the lateral geniculate complex',
        'Medial geniculate complex',
        'Paraventricular nucleus of the thalamus',
        'Rhomboid nucleus',
        'Anterodorsal nucleus',
        'Anteromedial nucleus',
        'Interanterodorsal nucleus of the thalamus',
        'Perireunensis nucleus',
        'Lateral habenula',
        'Medial habenula',
        'Fields of Forel',
        'Subparafascicular nucleus',
        'Subparafascicular area',
        'Nucleus of the optic tract',
        'Anterior pretectal nucleus',
        'Posterior pretectal nucleus',
        'Dorsal terminal nucleus of the accessory optic tract',
        'Medial terminal nucleus of the accessory optic tract',
    ],
    'Hypothalamus': [
        'Hypothalamus',
        'Lateral hypothalamic area',
        'Zona incerta',
        'Subthalamic nucleus',
        'Posterior hypothalamic nucleus',
        'Paraventricular hypothalamic nucleus',
        'Tuberomammillary nucleus',
        'Lateral preoptic area',
        'Parasubthalamic nucleus',
    ],
    'Hippocampus': [
        'Field CA1',
        'Field CA2',
        'Field CA3',
        'Dentate gyrus',
        'Subiculum',
        'Postsubiculum',
        'Hippocampal formation',
    ],
    'Midbrain': [
        'Midbrain reticular nucleus',
        'Midbrain',
        'Superior colliculus',
        'Red nucleus',
        'Substantia nigra',
        'Ventral tegmental area',
        'Inferior colliculus',
        'Pedunculopontine nucleus',
        'Periaqueductal gray',
        'Peripeduncular nucleus',
        'Nucleus sagulum',
        'Nucleus of the brachium of the inferior colliculus',
    ],
    'Pons': [
        'Pons',
        'Pontine reticular nucleus',
        'Parabrachial nucleus',
        'Tegmental reticular nucleus',
        'Locus ceruleus',
        'Koelliker-Fuse subnucleus',
        'Nucleus of the lateral lemniscus',
    ],
    'Medulla': [
        'Medulla',
        'Gigantocellular reticular nucleus',
        'Intermediate reticular nucleus',
        'Magnocellular reticular nucleus',
        'Parvicellular reticular nucleus',
        'Inferior olivary complex',
        'Medial vestibular nucleus',
        'Superior vestibular nucleus',
        'Lateral vestibular nucleus',
        'Spinal vestibular nucleus',
        'Spinal nucleus of the trigeminal',
        'Facial motor nucleus',
        'Hypoglossal nucleus',
        'Dorsal motor nucleus of the vagus nerve',
        'Nucleus of the solitary tract',
        'Paragigantocellular reticular nucleus',
        'Medullary reticular nucleus',
        'External cuneate nucleus',
        'Parapyramidal nucleus',
        'Nucleus raphe magnus',
        'Nucleus raphe obscurus',
        'Parasolitary nucleus',
        'Nucleus of Roller',
        'Nucleus x',
        'Infracerebellar nucleus',
    ],
    'Cerebellum': [
        'Cerebellum',
        'Lobule',
        'Lobules',
        'Simple lobule',
        'Copula pyramidis',
        'Interposed nucleus',
        'Fastigial nucleus',
        'Declive',
        'Paramedian lobule',
        'Crus',
        'Nodulus',
        'Uvula',
        'Pyramus',
        'Lingula',
    ],
    'Olfactory': [
        'Anterior olfactory nucleus',
        'Piriform area',
        'Olfactory areas',
        'Olfactory tubercle',
        'Taenia tecta',
        'Accessory olfactory bulb',
        'Endopiriform nucleus',
    ],
    'CorticalSubplate': [
        'Cortical subplate',
        'Claustrum',
        'Basolateral amygdalar nucleus',
        'Central amygdalar nucleus',
        'Lateral amygdalar nucleus',
        'Posterior amygdalar nucleus',
        'Basomedial amygdalar nucleus',
        'Anterior amygdalar area',
        'Intercalated amygdalar nucleus',
        'Medial amygdalar nucleus',
        'Lateral septal nucleus',
        'Septofimbrial nucleus',
        'Triangular nucleus of septum',
    ],
}


def map_anno_name_to_region(anno_name):
    """Map a detailed Allen Brain Atlas annotation name to a high-level region category."""
    if not anno_name or anno_name.strip() == '':
        return None
    for region, prefixes in REGION_MAPPING.items():
        for prefix in prefixes:
            if anno_name.startswith(prefix) or anno_name.startswith(prefix + ',') or anno_name.startswith(prefix + '/'):
                return region
    return None


def compute_firing_rates(spike_times, go_cue_time, bin_edges):
    """Compute firing rates for a single unit in a single trial.

    Args:
        spike_times: array of spike times (absolute)
        go_cue_time: go cue time (absolute)
        bin_edges: bin edges relative to go cue

    Returns:
        firing_rates: array of firing rates (Hz), shape (n_bins,)
    """
    # Align spike times to go cue
    rel_times = spike_times - go_cue_time
    # Only consider spikes in the window
    mask = (rel_times >= bin_edges[0]) & (rel_times < bin_edges[-1])
    rel_times = rel_times[mask]
    # Count spikes in each bin
    counts, _ = np.histogram(rel_times, bins=bin_edges)
    # Convert to firing rates
    bin_width = bin_edges[1] - bin_edges[0]
    return counts.astype(np.float64) / bin_width


def get_tone_onset_relative_to_go(sample_start_times, go_cue_time):
    """Find the tone onset time relative to go cue for a single trial.

    The tone onset is the last sample_start before the go cue
    (accounts for early lick replays).
    """
    before = sample_start_times[sample_start_times < go_cue_time]
    if len(before) > 0:
        return before[-1] - go_cue_time  # negative value
    return None


def extract_tongue_y_for_trial(tongue_timestamps, tongue_y,
                                go_cue_time, bin_edges):
    """Extract tongue y-position in time bins for a single trial.

    Uses all tongue tracking data (no confidence filtering) to compute
    mean y-position per bin. NaN for bins with no tracking data.

    Returns:
        tongue_y_binned: array of mean y-positions per bin, shape (n_bins,)
    """
    n_bins = len(bin_edges) - 1
    tongue_y_binned = np.full(n_bins, np.nan)

    # Get absolute time window
    t_abs_start = go_cue_time + bin_edges[0]
    t_abs_end = go_cue_time + bin_edges[-1]

    # Find tongue data in this window
    idx = np.searchsorted(tongue_timestamps, [t_abs_start, t_abs_end])
    ts_window = tongue_timestamps[idx[0]:idx[1]]
    y_window = tongue_y[idx[0]:idx[1]]

    if len(ts_window) == 0:
        return tongue_y_binned

    # Align to go cue
    ts_rel = ts_window - go_cue_time

    # For each bin, compute mean y
    for b in range(n_bins):
        bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
        if np.any(bin_mask):
            tongue_y_binned[b] = np.mean(y_window[bin_mask])

    return tongue_y_binned


def process_session(nwb_path, verbose=True):
    """Process a single NWB session file.

    Returns:
        session_data: dict with neural, input, output, metadata for this session,
                      or None if session doesn't meet criteria
    """
    if verbose:
        print(f"Processing: {os.path.basename(nwb_path)}")

    f = h5py.File(nwb_path, 'r')

    try:
        # === Extract trial information ===
        n_trials = len(f['intervals/trials/id'][:])
        outcomes = np.array([o.decode() if isinstance(o, bytes) else o
                            for o in f['intervals/trials/outcome'][:]])
        early_lick = np.array([e.decode() if isinstance(e, bytes) else e
                              for e in f['intervals/trials/early_lick'][:]])
        instructions = np.array([i.decode() if isinstance(i, bytes) else i
                                for i in f['intervals/trials/trial_instruction'][:]])
        auto_water = f['intervals/trials/auto_water'][:]
        free_water = f['intervals/trials/free_water'][:]

        # Photostim info from trial table
        ps_onset_str = np.array([p.decode() if isinstance(p, bytes) else p
                                for p in f['intervals/trials/photostim_onset'][:]])
        ps_dur_str = np.array([p.decode() if isinstance(p, bytes) else p
                              for p in f['intervals/trials/photostim_duration'][:]])

        # Go cue times
        go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]

        # Trial start times (needed for photostim onset conversion)
        trial_start_times = f['intervals/trials/start_time'][:]

        # Sample start times (for tone onset calculation)
        sample_start_times = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]

        assert len(go_cue_times) == n_trials, f"Go cue count ({len(go_cue_times)}) != trial count ({n_trials})"

        # === Load units ===
        classification = np.array([c.decode() if isinstance(c, bytes) else c
                                   for c in f['units/classification'][:]])
        anno_names = np.array([a.decode() if isinstance(a, bytes) else a
                              for a in f['units/anno_name'][:]])

        # Filter by quality (classifier-based QC)
        good_mask = classification == 'good'

        # Map brain regions and filter units with valid regions
        unit_regions = []
        unit_indices = []
        for i in range(len(classification)):
            if not good_mask[i]:
                continue
            region = map_anno_name_to_region(anno_names[i])
            if region is not None:
                unit_regions.append(region)
                unit_indices.append(i)

        if len(unit_indices) == 0:
            if verbose:
                print(f"  Skipping: no good units with valid regions")
            return None

        unit_indices = np.array(unit_indices)
        n_units = len(unit_indices)

        # === Determine valid trials based on recording observation window ===
        # Units may only have data for a subset of trials if the recording
        # didn't span the entire behavioral session
        obs_intervals = f['units/obs_intervals'][:]
        obs_intervals_index = f['units/obs_intervals_index'][:]

        # Find the minimum observation end time across all selected units
        min_obs_end = np.inf
        max_obs_start = -np.inf
        for ui in unit_indices:
            oi_start = 0 if ui == 0 else obs_intervals_index[ui - 1]
            oi_end = obs_intervals_index[ui]
            unit_obs = obs_intervals[oi_start:oi_end]
            if len(unit_obs) > 0:
                min_obs_end = min(min_obs_end, unit_obs[-1, 1])
                max_obs_start = max(max_obs_start, unit_obs[0, 0])

        # Only include trials where the go cue (and surrounding window) falls
        # within the recording observation window
        recording_valid = ((go_cue_times + T_START) >= max_obs_start - 0.1) & \
                          ((go_cue_times + T_END) <= min_obs_end + 0.1)

        if verbose and np.sum(~recording_valid) > 0:
            print(f"  Recording window limits: {np.sum(recording_valid)}/{n_trials} trials have neural data")

        # === Session filtering ===
        # Use ALL trials for session-level performance calculation (as in the paper)
        # Control trials: no photostim, no early lick, no auto_water, no free_water
        is_control = ((early_lick == 'no early') &
                      (auto_water == 0) &
                      (free_water == 0) &
                      (ps_onset_str == 'N/A'))

        control_hits_left = np.sum(is_control & (outcomes == 'hit') & (instructions == 'left'))
        control_hits_right = np.sum(is_control & (outcomes == 'hit') & (instructions == 'right'))

        # Performance on control trials (excluding ignore/no-response)
        control_responding = is_control & (outcomes != 'ignore')
        n_control_responding = np.sum(control_responding)
        if n_control_responding == 0:
            if verbose:
                print(f"  Skipping: no responding control trials")
            return None

        performance = np.sum(is_control & (outcomes == 'hit')) / n_control_responding

        if verbose:
            print(f"  Performance: {performance:.1%}, Correct L/R: {control_hits_left}/{control_hits_right}")

        if performance <= MIN_PERFORMANCE:
            if verbose:
                print(f"  Skipping: performance {performance:.1%} <= {MIN_PERFORMANCE:.0%}")
            return None

        if control_hits_left < MIN_CORRECT_PER_DIRECTION or control_hits_right < MIN_CORRECT_PER_DIRECTION:
            if verbose:
                print(f"  Skipping: insufficient correct trials (L={control_hits_left}, R={control_hits_right})")
            return None

        # === Select trials ===
        # Exclude auto_water and free_water trials (as in reference code)
        # Also must have valid neural recording data
        trial_mask = (auto_water == 0) & (free_water == 0) & recording_valid
        trial_indices = np.where(trial_mask)[0]

        if len(trial_indices) < 2:
            if verbose:
                print(f"  Skipping: fewer than 2 valid trials")
            return None

        if verbose:
            print(f"  Good units: {n_units}")

        # === Load spike times ===
        spike_times_data = f['units/spike_times'][:]
        spike_times_index = f['units/spike_times_index'][:]

        # Build per-unit spike time arrays
        unit_spike_times = []
        for ui in unit_indices:
            start_idx = 0 if ui == 0 else spike_times_index[ui - 1]
            end_idx = spike_times_index[ui]
            unit_spike_times.append(spike_times_data[start_idx:end_idx])

        # === Load tongue tracking ===
        tongue_ts = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
        tongue_data = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
        tongue_y = tongue_data[:, 1]  # y-position

        # === Process each trial ===
        neural_trials = []
        input_trials = []
        output_choice = []
        output_outcome = []
        output_early_lick = []
        tongue_y_all_trials = []

        for ti in trial_indices:
            gc = go_cue_times[ti]

            # --- Neural data: firing rates ---
            fr_matrix = np.zeros((n_units, N_BINS), dtype=np.float64)
            for j, st in enumerate(unit_spike_times):
                fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
            neural_trials.append(fr_matrix)

            # --- Input: time from tone onset ---
            tone_onset_rel = get_tone_onset_relative_to_go(sample_start_times, gc)
            if tone_onset_rel is None:
                # Fallback: use typical value
                tone_onset_rel = -1.85
            time_from_tone = BIN_CENTERS - tone_onset_rel  # time since tone onset at each bin

            # --- Input: photostim on/off ---
            photostim_binary = np.zeros(N_BINS, dtype=np.float64)
            if ps_onset_str[ti] != 'N/A':
                try:
                    ps_onset_val = float(ps_onset_str[ti])
                    ps_dur = float(ps_dur_str[ti])
                    # photostim_onset is relative to trial start
                    ps_abs_onset = trial_start_times[ti] + ps_onset_val
                    # Convert to relative to go cue
                    ps_start_rel = ps_abs_onset - gc
                    ps_end_rel = ps_start_rel + ps_dur
                    # Mark bins where photostim is on
                    photostim_binary = ((BIN_CENTERS >= ps_start_rel) &
                                       (BIN_CENTERS < ps_end_rel)).astype(np.float64)
                except (ValueError, TypeError):
                    pass

            input_trial = np.stack([time_from_tone, photostim_binary], axis=0)  # (2, n_bins)
            input_trials.append(input_trial)

            # --- Output: choice ---
            if instructions[ti] == 'left':
                choice = 0
            else:
                choice = 1
            output_choice.append(choice)

            # --- Output: outcome ---
            if outcomes[ti] == 'ignore':
                outcome = 0
            elif outcomes[ti] == 'miss':
                outcome = 1
            else:  # hit
                outcome = 2
            output_outcome.append(outcome)

            # --- Output: early lick ---
            el = 1 if early_lick[ti] == 'early' else 0
            output_early_lick.append(el)

            # --- Tongue y-position (continuous, will discretize later) ---
            ty = extract_tongue_y_for_trial(tongue_ts, tongue_y, gc, BIN_EDGES)
            tongue_y_all_trials.append(ty)

        # === Discretize tongue y-position per session ===
        # Collect all valid (non-NaN) tongue y values across all trials in this session
        all_tongue_y_vals = np.concatenate(tongue_y_all_trials)
        valid_tongue_y = all_tongue_y_vals[~np.isnan(all_tongue_y_vals)]

        if len(valid_tongue_y) > 0:
            p40 = np.percentile(valid_tongue_y, 40)
            p60 = np.percentile(valid_tongue_y, 60)

            tongue_y_discrete_trials = []
            for ty in tongue_y_all_trials:
                ty_disc = np.ones(N_BINS, dtype=np.int64)  # default to middle category
                valid_mask = ~np.isnan(ty)
                ty_disc[valid_mask & (ty < p40)] = 0
                ty_disc[valid_mask & (ty >= p40) & (ty <= p60)] = 1
                ty_disc[valid_mask & (ty > p60)] = 2
                tongue_y_discrete_trials.append(ty_disc)
        else:
            # No valid tongue tracking - all middle category
            tongue_y_discrete_trials = [np.ones(N_BINS, dtype=np.int64) for _ in range(len(trial_indices))]

        # === Build output arrays ===
        output_trials = []
        for i in range(len(trial_indices)):
            out = np.array([
                np.full(N_BINS, output_choice[i], dtype=np.int64),       # choice (per-trial, broadcast)
                np.full(N_BINS, output_outcome[i], dtype=np.int64),      # outcome (per-trial, broadcast)
                np.full(N_BINS, output_early_lick[i], dtype=np.int64),   # early lick (per-trial, broadcast)
                tongue_y_discrete_trials[i],              # tongue y (time-varying)
            ], dtype=np.int64)  # (4, n_bins)
            output_trials.append(out)

        # Extract subject ID from path
        basename = os.path.basename(nwb_path)
        subject_id = basename.split('_ses-')[0].replace('sub-', '')

        if verbose:
            print(f"  Kept {len(trial_indices)} trials, {n_units} neurons")

        return {
            'neural': neural_trials,
            'input': input_trials,
            'output': output_trials,
            'subject_id': subject_id,
            'unit_regions': unit_regions,
            'session_file': basename,
        }

    finally:
        f.close()


def main():
    parser = argparse.ArgumentParser(description='Convert NWB data to decoder format')
    parser.add_argument('--sample', action='store_true', help='Only process first 5 sessions')
    parser.add_argument('--output', type=str, default='converted_data.pkl', help='Output file path')
    args = parser.parse_args()

    # Find all NWB files
    nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
    print(f"Found {len(nwb_files)} NWB files")

    if args.sample:
        nwb_files = nwb_files[:5]
        print(f"Sample mode: processing first {len(nwb_files)} files")

    # Process all sessions
    all_neural = []
    all_input = []
    all_output = []
    all_subject_ids = []
    all_unit_regions = []
    all_session_files = []

    n_skipped = 0
    n_processed = 0

    for nwb_path in nwb_files:
        result = process_session(nwb_path)
        if result is None:
            n_skipped += 1
            continue

        all_neural.append(result['neural'])
        all_input.append(result['input'])
        all_output.append(result['output'])
        all_subject_ids.append(result['subject_id'])
        all_unit_regions.append(result['unit_regions'])
        all_session_files.append(result['session_file'])
        n_processed += 1

    print(f"\nProcessed: {n_processed} sessions, Skipped: {n_skipped}")

    # Build subjects list and index
    unique_subjects = sorted(set(all_subject_ids))
    subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])

    # Build brain_regions list and index
    all_region_names = set()
    for regions in all_unit_regions:
        all_region_names.update(regions)
    brain_regions = sorted(all_region_names)

    brain_region_idx = []
    for regions in all_unit_regions:
        idx = np.array([brain_regions.index(r) for r in regions])
        brain_region_idx.append(idx)

    # Build final data structure
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,

        'subjects': unique_subjects,
        'subject_idx': subject_idx,

        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,

        'input_names': ['time_from_tone_onset', 'photostim_on'],
        'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
        'output_values': [
            ['left', 'right'],           # choice: 0=left, 1=right
            ['ignore', 'miss', 'hit'],   # outcome: 0=ignore, 1=miss, 2=hit
            ['no', 'yes'],               # early_lick: 0=no, 1=yes
            ['low', 'middle', 'high'],   # tongue_y: 0=<p40, 1=p40-p60, 2=>p60
        ],

        'metadata': {
            'task_description': 'Auditory delayed response task: mice heard tones (3kHz or 12kHz) during sample epoch, maintained memory during delay, then reported by licking left or right port after go cue.',
            'time_bin_size': BIN_WIDTH * 1000,  # in ms
            'temporal_alignment_event': 'Go cue onset',
            'off_start': T_START,
            'off_end': T_END,
            'n_sessions': n_processed,
            'n_subjects': len(unique_subjects),
            'session_files': all_session_files,
            'neuron_qc': 'classifier-based quality control (classification=good)',
            'session_filter': f'Performance > {MIN_PERFORMANCE:.0%}, >= {MIN_CORRECT_PER_DIRECTION} correct trials per direction on control trials',
            'trial_filter': 'Excluded auto_water and free_water trials',
            'firing_rate_method': f'{BIN_WIDTH*1000:.0f}ms non-overlapping bins, spike count / bin_width',
            'tongue_tracking': 'DeepLabCut side-view tongue y-position, all tracking data used for percentile computation',
        },
    }

    # Print summary statistics
    total_neurons = sum(len(r) for r in brain_region_idx)
    total_trials = sum(len(s) for s in all_neural)
    neurons_per_session = [all_neural[i][0].shape[0] for i in range(len(all_neural))]
    trials_per_session = [len(s) for s in all_neural]

    print(f"\n=== Dataset Summary ===")
    print(f"Sessions: {n_processed}")
    print(f"Subjects: {len(unique_subjects)}")
    print(f"Total neurons (across sessions): {total_neurons}")
    print(f"Total trials: {total_trials}")
    print(f"Neurons per session: mean={np.mean(neurons_per_session):.1f}, "
          f"range=[{np.min(neurons_per_session)}, {np.max(neurons_per_session)}]")
    print(f"Trials per session: mean={np.mean(trials_per_session):.1f}, "
          f"range=[{np.min(trials_per_session)}, {np.max(trials_per_session)}]")
    print(f"Time bins: {N_BINS} ({BIN_WIDTH*1000:.0f}ms each)")
    print(f"Time window: [{T_START}, {T_END}]s relative to go cue")
    print(f"Brain regions: {brain_regions}")

    # Region distribution
    region_counts = {}
    for regions in all_unit_regions:
        for r in regions:
            region_counts[r] = region_counts.get(r, 0) + 1
    print(f"\nNeurons per region:")
    for r in sorted(region_counts, key=lambda x: -region_counts[x]):
        print(f"  {r}: {region_counts[r]}")

    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    file_size = os.path.getsize(args.output)
    print(f"Saved ({file_size / 1e9:.2f} GB)")

    return data


if __name__ == '__main__':
    main()
