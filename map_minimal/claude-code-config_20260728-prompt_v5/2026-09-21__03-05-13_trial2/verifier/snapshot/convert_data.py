"""
Convert NWB data from the MAP (Mesoscale Activity Project) dataset into the
standardized decoder format.

Data source: "Brain-wide neural activity underlying memory-guided movement"
Processing follows: "Brain-wide analysis reveals movement encoding structured
across and within brain areas"

Key decisions:
- Neuron filtering: Only "good" units (classifier-based QC per the data paper).
- Session filtering: >65% correct on control trials (hit/(hit+miss), excluding
  early lick, auto_water, free_water, photostim), and >=50 correct lick-left
  and >=50 correct lick-right trials (per the data paper methods).
- Trial filtering: Exclude auto_water and free_water trials (not real behavioral
  trials). Keep early lick and no-response (ignore) trials since early_lick and
  outcome are decoder outputs. Keep photostim trials since photostim is a
  decoder input.
- Brain region: Use anno_name (CCF annotation) mapped to major categories;
  fall back to electrode group target region for units without annotation.
- Temporal alignment: Go cue onset, [-2.5, 1.5] s, 50 ms bins (80 bins).
- Firing rates: spike counts per bin / bin_width (same approach as reference
  code's sliding_histogram with non-overlapping bins).
- Tone onset: last sample_start before each trial's go cue (the successful
  presentation that led to delay/go, not earlier replays from early licking).
- Photostimulation: binary time series, 1 during photostim on period.
- Tongue y-position: discretized per session using DLC confidence > 0.9 for
  visibility, percentiles computed over all visible positions in the session.
"""

import os
import glob
import json
import pickle
import numpy as np
import h5py

# ---- Parameters ----
TIME_BEFORE = 2.5
TIME_AFTER = 1.5
BIN_SIZE = 0.05
N_BINS = int(round((TIME_BEFORE + TIME_AFTER) / BIN_SIZE))  # 80
BIN_EDGES = np.linspace(-TIME_BEFORE, TIME_AFTER, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2

MIN_PERFORMANCE = 0.65
MIN_CORRECT_PER_SIDE = 50

DLC_CONFIDENCE_THRESHOLD = 0.9

DATA_DIR = '/app/data'
OUTPUT_FILE = '/app/converted_data.pkl'

# ---- Brain region mapping ----
ANNO_TO_REGION = {
    'Secondary motor area': 'ALM',
    'Primary motor area': 'OtherCortex',
    'Primary somatosensory area': 'OtherCortex',
    'Primary visual area': 'OtherCortex',
    'Agranular insular area': 'OtherCortex',
    'Anterior cingulate area': 'OtherCortex',
    'Retrosplenial area': 'OtherCortex',
    'Prelimbic area': 'OtherCortex',
    'Dorsal peduncular area': 'OtherCortex',
    'Frontal pole': 'OtherCortex',
    'Infralimbic area': 'OtherCortex',
    'Supplemental somatosensory area': 'OtherCortex',
    'Visceral area': 'OtherCortex',
    'Gustatory areas': 'OtherCortex',
    'Temporal association areas': 'OtherCortex',
    'Perirhinal area': 'OtherCortex',
    'Ectorhinal area': 'OtherCortex',
    'Entorhinal area': 'OtherCortex',
    'Posterior parietal association areas': 'OtherCortex',
    'Postsubiculum': 'OtherCortex',
    'Orbital area': 'Orbital',
    'Caudoputamen': 'Striatum',
    'Striatum': 'Striatum',
    'Nucleus accumbens': 'Striatum',
    'Fundus of striatum': 'Striatum',
    'Pallidum': 'Pallidum',
    'Globus pallidus': 'Pallidum',
    'Substantia innominata': 'Pallidum',
    'Bed nuclei of the stria terminalis': 'Pallidum',
    'Lateral septal nucleus': 'Pallidum',
    'Septofimbrial nucleus': 'Pallidum',
    'Triangular nucleus of septum': 'Pallidum',
    'Medial septal nucleus': 'Pallidum',
    'Thalamus': 'Thalamus',
    'Anteromedial nucleus': 'Thalamus',
    'Anteroventral nucleus of thalamus': 'Thalamus',
    'Central lateral nucleus of the thalamus': 'Thalamus',
    'Lateral dorsal nucleus of thalamus': 'Thalamus',
    'Lateral posterior nucleus of the thalamus': 'Thalamus',
    'Mediodorsal nucleus of thalamus': 'Thalamus',
    'Paracentral nucleus': 'Thalamus',
    'Posterior complex of the thalamus': 'Thalamus',
    'Reticular nucleus of the thalamus': 'Thalamus',
    'Submedial nucleus of the thalamus': 'Thalamus',
    'Ventral anterior-lateral complex of the thalamus': 'Thalamus',
    'Ventral medial nucleus of the thalamus': 'Thalamus',
    'Ventral posterolateral nucleus of the thalamus': 'Thalamus',
    'Ventral posteromedial nucleus of the thalamus': 'Thalamus',
    'Anterodorsal nucleus': 'Thalamus',
    'Lateral habenula': 'Thalamus',
    'Medial habenula': 'Thalamus',
    'Nucleus of reuniens': 'Thalamus',
    'Parafascicular nucleus': 'Thalamus',
    'Peripeduncular nucleus': 'Thalamus',
    'Posterior limiting nucleus of the thalamus': 'Thalamus',
    'Ventral part of the lateral geniculate complex': 'Thalamus',
    'Dorsal part of the lateral geniculate complex': 'Thalamus',
    'Medial geniculate complex': 'Thalamus',
    'Intergeniculate leaflet of the lateral geniculate complex': 'Thalamus',
    'Hypothalamus': 'Hypothalamus',
    'Paraventricular hypothalamic nucleus': 'Hypothalamus',
    'Zona incerta': 'Hypothalamus',
    'Lateral hypothalamic area': 'Hypothalamus',
    'Subthalamic nucleus': 'Hypothalamus',
    'Fields of Forel': 'Hypothalamus',
    'Dentate gyrus': 'Hippocampus',
    'Field CA1': 'Hippocampus',
    'Field CA2': 'Hippocampus',
    'Field CA3': 'Hippocampus',
    'Subiculum': 'Hippocampus',
    'Midbrain': 'Midbrain',
    'Midbrain reticular nucleus': 'Midbrain',
    'Superior colliculus': 'Midbrain',
    'Inferior colliculus': 'Midbrain',
    'Nucleus of the brachium of the inferior colliculus': 'Midbrain',
    'Pedunculopontine nucleus': 'Midbrain',
    'Substantia nigra': 'Midbrain',
    'Ventral tegmental area': 'Midbrain',
    'Red nucleus': 'Midbrain',
    'Periaqueductal gray': 'Midbrain',
    'Anterior pretectal nucleus': 'Midbrain',
    'Cuneiform nucleus': 'Midbrain',
    'Interpeduncular nucleus': 'Midbrain',
    'Pons': 'Pons',
    'Nucleus of the lateral lemniscus': 'Pons',
    'Nucleus sagulum': 'Pons',
    'Pontine reticular nucleus': 'Pons',
    'Parabrachial nucleus': 'Pons',
    'Tegmental reticular nucleus': 'Pons',
    'Pontine gray': 'Pons',
    'Medulla': 'Medulla',
    'Facial motor nucleus': 'Medulla',
    'Spinal nucleus of the trigeminal': 'Medulla',
    'Gigantocellular reticular nucleus': 'Medulla',
    'Intermediate reticular nucleus': 'Medulla',
    'Paragigantocellular reticular nucleus': 'Medulla',
    'Parvicellular reticular nucleus': 'Medulla',
    'Superior olivary complex': 'Medulla',
    'Nucleus ambiguus': 'Medulla',
    'Cerebellum': 'Cerebellum',
    'Cerebellar cortex': 'Cerebellum',
    'Cerebellar nuclei': 'Cerebellum',
    'Anterior olfactory nucleus': 'Olfactory',
    'Olfactory areas': 'Olfactory',
    'Piriform area': 'Olfactory',
    'Taenia tecta': 'Olfactory',
    'Main olfactory bulb': 'Olfactory',
    'Claustrum': 'CorticalSubplate',
    'Endopiriform nucleus': 'CorticalSubplate',
    'Lateral amygdalar nucleus': 'CorticalSubplate',
    'Basolateral amygdalar nucleus': 'CorticalSubplate',
    'Basomedial amygdalar nucleus': 'CorticalSubplate',
    'Central amygdalar nucleus': 'CorticalSubplate',
    'Medial amygdalar nucleus': 'CorticalSubplate',
}

TARGET_TO_REGION = {
    'ALM': 'ALM',
    'Striatum': 'Striatum',
    'Thalamus': 'Thalamus',
    'Midbrain': 'Midbrain',
    'Medulla': 'Medulla',
    'BLA': 'CorticalSubplate',
    'ECT': 'OtherCortex',
}


def get_unit_brain_region(anno_name, elec_target):
    """Map a unit to its major brain region using anno_name or electrode target."""
    if anno_name and anno_name != '' and str(anno_name) != 'nan':
        base = str(anno_name).split(',')[0].strip()
        if base in ANNO_TO_REGION:
            return ANNO_TO_REGION[base]
    return elec_target


def get_electrode_target_region(location_json):
    """Extract major region from electrode location JSON."""
    d = json.loads(location_json)
    brain_regions = d['brain_regions']
    parts = brain_regions.split()
    side = parts[0]
    region_name = ' '.join(parts[1:])
    major = TARGET_TO_REGION.get(region_name, region_name)
    return side, major


def compute_firing_rates_all_trials(spike_times, spike_idx, unit_indices, go_times, trial_indices):
    """Compute firing rates for all trials at once using vectorized searchsorted.

    Returns list of (n_neurons, n_bins) arrays.
    """
    n_neurons = len(unit_indices)
    n_trials = len(trial_indices)
    go_subset = go_times[trial_indices]

    # Pre-extract spike times for each good unit (sorted)
    unit_spikes = []
    for uid in unit_indices:
        start = spike_idx[uid - 1] if uid > 0 else 0
        end = spike_idx[uid]
        unit_spikes.append(spike_times[start:end])

    # For each unit, compute counts for all trials at once using searchsorted
    # rates_all shape: (n_neurons, n_trials, N_BINS)
    rates_all = np.zeros((n_neurons, n_trials, N_BINS), dtype=np.float32)

    for i, st in enumerate(unit_spikes):
        if len(st) == 0:
            continue
        # For each trial and each bin edge, compute absolute time
        # abs_edges[t, b] = go_subset[t] + BIN_EDGES[b]
        # We need counts between consecutive edges for each trial
        for t_idx in range(n_trials):
            go_t = go_subset[t_idx]
            abs_edges = go_t + BIN_EDGES
            # searchsorted gives indices where edges would be inserted
            edge_counts = np.searchsorted(st, abs_edges)
            # counts per bin = difference between consecutive edge positions
            rates_all[i, t_idx, :] = np.diff(edge_counts) / BIN_SIZE

    # Convert to list of (n_neurons, n_bins) arrays
    return [rates_all[:, t, :] for t in range(n_trials)]


def process_session(fpath):
    """Process one NWB session file. Returns session data dict or None."""
    with h5py.File(fpath, 'r') as f:
        trials = f['intervals/trials']
        n_trials = len(trials['id'][:])

        outcomes = np.array([x.decode() if isinstance(x, bytes) else str(x)
                            for x in trials['outcome'][:]])
        early_licks = np.array([x.decode() if isinstance(x, bytes) else str(x)
                               for x in trials['early_lick'][:]])
        instructions = np.array([x.decode() if isinstance(x, bytes) else str(x)
                                for x in trials['trial_instruction'][:]])
        auto_water = trials['auto_water'][:].astype(int)
        free_water = trials['free_water'][:].astype(int)
        photostim_onset_raw = np.array([x.decode() if isinstance(x, bytes) else str(x)
                                        for x in trials['photostim_onset'][:]])

        # ---- Session filtering ----
        is_control = (photostim_onset_raw == 'N/A') & (auto_water == 0) & (free_water == 0)
        is_not_early = (early_licks == 'no early')
        control_regular = is_control & is_not_early

        control_hits = np.sum(outcomes[control_regular] == 'hit')
        control_misses = np.sum(outcomes[control_regular] == 'miss')
        control_responded = control_hits + control_misses
        if control_responded == 0:
            print(f"  Skip: no responded control trials")
            return None
        performance = control_hits / control_responded
        if performance < MIN_PERFORMANCE:
            print(f"  Skip: perf {performance:.2f} < {MIN_PERFORMANCE}")
            return None

        correct_left = np.sum((outcomes == 'hit') & (instructions == 'left') & control_regular)
        correct_right = np.sum((outcomes == 'hit') & (instructions == 'right') & control_regular)
        if correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
            print(f"  Skip: correct L={correct_left}, R={correct_right}")
            return None

        # ---- Trial mask: exclude auto_water and free_water ----
        trial_mask = (auto_water == 0) & (free_water == 0)
        trial_indices = np.where(trial_mask)[0]
        if len(trial_indices) < 2:
            print(f"  Skip: too few valid trials")
            return None

        # ---- Go cue times ----
        go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]

        # ---- Sample start times (tone onset) ----
        sample_starts = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]

        # For each trial, find the last sample_start before go cue
        tone_onsets = np.full(n_trials, np.nan)
        for i in range(n_trials):
            go_t = go_times[i]
            lower = go_times[i - 1] if i > 0 else 0.0
            mask_s = (sample_starts > lower) & (sample_starts < go_t)
            candidates = sample_starts[mask_s]
            if len(candidates) > 0:
                tone_onsets[i] = candidates[-1]

        # ---- Photostimulation events ----
        ps_start_abs = np.array([])
        ps_stop_abs = np.array([])
        if 'photostim_start_times' in f['acquisition/BehavioralEvents']:
            ps_ts = f['acquisition/BehavioralEvents/photostim_start_times/timestamps']
            if ps_ts.shape[0] > 0:
                ps_start_abs = ps_ts[:]
                ps_stop_abs = f['acquisition/BehavioralEvents/photostim_stop_times/timestamps'][:]

        # ---- Units ----
        units = f['units']
        unit_quality = np.array([x.decode() if isinstance(x, bytes) else str(x)
                                for x in units['unit_quality'][:]])
        good_indices = np.where(unit_quality == 'good')[0]
        if len(good_indices) == 0:
            print(f"  Skip: no good units")
            return None

        # ---- Brain regions ----
        anno_names = np.array([x.decode() if isinstance(x, bytes) else str(x)
                              for x in units['anno_name'][:]])
        electrodes_ref = units['electrodes'][:]

        elec_table = f['general/extracellular_ephys/electrodes']
        elec_locations = np.array([x.decode() if isinstance(x, bytes) else str(x)
                                  for x in elec_table['location'][:]])

        # Cache electrode -> region mapping per unique location
        elec_region_cache = {}
        unit_regions = []
        for uid in good_indices:
            elec_idx = int(electrodes_ref[uid])
            if elec_idx not in elec_region_cache:
                side, region = get_electrode_target_region(elec_locations[elec_idx])
                elec_region_cache[elec_idx] = (side, region)
            _, target_region = elec_region_cache[elec_idx]
            region = get_unit_brain_region(anno_names[uid], target_region)
            unit_regions.append(region)

        # ---- Spike times (load all at once) ----
        all_spike_times = units['spike_times'][:]
        spike_times_index = units['spike_times_index'][:]

        # ---- Determine neural recording range ----
        # Find min/max spike time across all good units
        min_spike_time = np.inf
        max_spike_time = 0.0
        for uid in good_indices:
            start = spike_times_index[uid - 1] if uid > 0 else 0
            end = spike_times_index[uid]
            if end > start:
                min_spike_time = min(min_spike_time, all_spike_times[start])
                max_spike_time = max(max_spike_time, all_spike_times[end - 1])

        # Filter trials to those within neural recording range
        # Trial window is [go_t - TIME_BEFORE, go_t + TIME_AFTER]
        valid_trial_mask = np.array([
            (go_times[ti] - TIME_BEFORE >= min_spike_time - 1.0) and
            (go_times[ti] + TIME_AFTER <= max_spike_time + 1.0)
            for ti in trial_indices
        ])
        trial_indices = trial_indices[valid_trial_mask]

        if len(trial_indices) < 2:
            print(f"  Skip: too few trials within recording range")
            return None

        # ---- Compute firing rates for all trials ----
        neural_trials = compute_firing_rates_all_trials(
            all_spike_times, spike_times_index, good_indices,
            go_times, trial_indices
        )

        # ---- Tongue tracking ----
        tongue_ts = None
        tongue_y = None
        tongue_conf = None
        tongue_y_p40 = None
        tongue_y_p60 = None
        if 'Camera0_side_TongueTracking' in f['acquisition/BehavioralTimeSeries']:
            tt = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
            tongue_ts = tt['timestamps'][:]
            tdata = tt['data'][:]
            tongue_y = tdata[:, 1]
            tongue_conf = tdata[:, 2]

            vis = tongue_conf > DLC_CONFIDENCE_THRESHOLD
            if np.any(vis):
                tongue_y_p40 = np.percentile(tongue_y[vis], 40)
                tongue_y_p60 = np.percentile(tongue_y[vis], 60)

        # ---- Build inputs and outputs for each trial ----
        input_trials = []
        output_trials = []

        for t_i, trial_idx in enumerate(trial_indices):
            go_t = go_times[trial_idx]

            # --- Input 1: Time from tone onset ---
            tone_t = tone_onsets[trial_idx]
            if np.isnan(tone_t):
                time_from_tone = BIN_CENTERS + 1.85
            else:
                tone_rel = tone_t - go_t
                time_from_tone = BIN_CENTERS - tone_rel

            # --- Input 2: Photostimulation ---
            photostim_vec = np.zeros(N_BINS, dtype=np.float32)
            if len(ps_start_abs) > 0:
                # Find photostim events within this trial's time window
                win_start = go_t + BIN_EDGES[0]
                win_end = go_t + BIN_EDGES[-1]
                for ps_s, ps_e in zip(ps_start_abs, ps_stop_abs):
                    if ps_e < win_start or ps_s > win_end:
                        continue
                    ps_s_rel = ps_s - go_t
                    ps_e_rel = ps_e - go_t
                    on_mask = (BIN_CENTERS >= ps_s_rel) & (BIN_CENTERS < ps_e_rel)
                    photostim_vec[on_mask] = 1.0

            inputs = np.stack([time_from_tone.astype(np.float32), photostim_vec], axis=0)
            input_trials.append(inputs)

            # --- Output 1: Choice ---
            outcome = outcomes[trial_idx]
            instruction = instructions[trial_idx]

            if outcome == 'ignore':
                choice = 2  # no lick
            elif outcome == 'hit':
                choice = 0 if instruction == 'left' else 1
            elif outcome == 'miss':
                choice = 1 if instruction == 'left' else 0
            else:
                choice = 2

            # --- Output 2: Outcome ---
            outcome_val = {'hit': 0, 'miss': 1, 'ignore': 2}.get(outcome, 2)

            # --- Output 3: Early lick ---
            early_val = 0 if early_licks[trial_idx] == 'no early' else 1

            # --- Output 4: Tongue y-position (time-varying) ---
            tongue_y_disc = np.full(N_BINS, 3, dtype=np.int64)
            if tongue_ts is not None and tongue_y_p40 is not None:
                # Vectorized: get absolute times for all bins
                t_abs_all = go_t + BIN_CENTERS
                # Find nearest tongue frame for each bin
                idx_all = np.searchsorted(tongue_ts, t_abs_all)
                idx_all = np.clip(idx_all, 0, len(tongue_ts) - 1)
                # Also check idx-1 for closer match
                idx_prev = np.clip(idx_all - 1, 0, len(tongue_ts) - 1)
                dist_cur = np.abs(tongue_ts[idx_all] - t_abs_all)
                dist_prev = np.abs(tongue_ts[idx_prev] - t_abs_all)
                best_idx = np.where(dist_prev < dist_cur, idx_prev, idx_all)
                best_dist = np.minimum(dist_cur, dist_prev)

                # Within one frame period (~3.4ms) and confident
                close_enough = best_dist < 0.005
                confident = tongue_conf[best_idx] > DLC_CONFIDENCE_THRESHOLD
                valid = close_enough & confident
                y_vals = tongue_y[best_idx]

                tongue_y_disc[valid & (y_vals < tongue_y_p40)] = 0
                tongue_y_disc[valid & (y_vals >= tongue_y_p40) & (y_vals < tongue_y_p60)] = 1
                tongue_y_disc[valid & (y_vals >= tongue_y_p60)] = 2

            outputs = np.zeros((4, N_BINS), dtype=np.int64)
            outputs[0, :] = choice
            outputs[1, :] = outcome_val
            outputs[2, :] = early_val
            outputs[3, :] = tongue_y_disc
            output_trials.append(outputs)

        basename = os.path.basename(fpath)
        subject_id = basename.split('_')[0]

        return {
            'neural': neural_trials,
            'input': input_trials,
            'output': output_trials,
            'subject_id': subject_id,
            'unit_regions': unit_regions,
            'n_good_units': len(good_indices),
            'n_trials': len(trial_indices),
            'session_name': basename,
        }


def main():
    nwb_files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*.nwb')))
    print(f"Found {len(nwb_files)} NWB files")

    all_sessions = []
    all_subjects = []

    for i, fpath in enumerate(nwb_files):
        print(f"[{i+1}/{len(nwb_files)}] {os.path.basename(fpath)}")
        result = process_session(fpath)
        if result is not None:
            all_sessions.append(result)
            if result['subject_id'] not in all_subjects:
                all_subjects.append(result['subject_id'])
            print(f"  -> {result['n_good_units']} units, {result['n_trials']} trials")

    print(f"\n{len(all_sessions)} sessions passed filtering")
    print(f"{len(all_subjects)} subjects")

    # Collect all unique brain regions
    all_brain_regions = sorted(set(
        r for sess in all_sessions for r in sess['unit_regions']
    ))
    region_to_idx = {r: i for i, r in enumerate(all_brain_regions)}

    data = {
        'neural': [sess['neural'] for sess in all_sessions],
        'input': [sess['input'] for sess in all_sessions],
        'output': [sess['output'] for sess in all_sessions],
        'subjects': all_subjects,
        'subject_idx': np.array([all_subjects.index(s['subject_id']) for s in all_sessions], dtype=np.int64),
        'brain_regions': all_brain_regions,
        'brain_region_idx': [
            np.array([region_to_idx[r] for r in sess['unit_regions']], dtype=np.int64)
            for sess in all_sessions
        ],
        'input_names': ['time_from_tone_onset', 'photostimulation'],
        'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
        'output_values': [
            ['left', 'right', 'no_lick'],
            ['hit', 'miss', 'ignore'],
            ['no', 'yes'],
            ['below_40pct', '40_to_60pct', 'above_60pct', 'not_visible'],
        ],
        'metadata': {
            'task_description': 'Auditory delayed response task: mice hear a tone (3kHz or 12kHz) during sample epoch, maintain memory during delay, then report by licking left or right after go cue',
            'time_bin_size': BIN_SIZE * 1000,
            'temporal_alignment_event': 'Go cue onset',
            'off_start': -TIME_BEFORE,
            'off_end': TIME_AFTER,
            'bin_centers': BIN_CENTERS.tolist(),
            'dataset': 'MAP (Mesoscale Activity Project)',
            'paper': 'Brain-wide neural activity underlying memory-guided movement',
            'n_sessions': len(all_sessions),
            'n_subjects': len(all_subjects),
        }
    }

    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(data, f)

    total_neurons = sum(len(s['unit_regions']) for s in all_sessions)
    total_trials = sum(s['n_trials'] for s in all_sessions)
    print(f"Total sessions: {len(all_sessions)}")
    print(f"Total neurons: {total_neurons}")
    print(f"Total trials: {total_trials}")
    print(f"Brain regions: {all_brain_regions}")
    print("Done!")


if __name__ == '__main__':
    main()
