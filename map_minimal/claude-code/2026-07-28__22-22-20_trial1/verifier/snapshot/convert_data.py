"""
Convert NWB data from the MAP dataset (Li et al.) to the decoder-compatible format.

Reference papers:
- "Brain-wide neural activity underlying memory-guided movement" (data paper)
- "Brain-wide analysis reveals movement encoding structured across and within brain areas" (method paper)

Processing decisions:
- Align to go cue onset
- Extract -2.5s to +1.5s relative to go cue
- 50ms time bins for firing rates (spike counts / bin_width)
- QC filtering: use 'good' classification from NWB (matches classifier-based QC in paper)
- Trial filtering: exclude early lick and no-response (ignore) trials
- Session selection: >65% correct rate on control (no photostim) non-early-lick trials,
  and at least 50 correct lick-left and lick-right trials each
- Sessions with 0 good units are skipped
"""

import os
import sys
import pickle
import numpy as np
import pynwb
from collections import OrderedDict
import argparse
import warnings
warnings.filterwarnings('ignore')

# ---- Parameters ----
BIN_WIDTH = 0.05  # 50 ms bins
BEGIN_TIME = -2.5  # relative to go cue
END_TIME = 1.5    # relative to go cue
MIN_PERF = 0.65   # >65% correct
MIN_CORRECT_PER_SIDE = 50  # at least 50 correct per side

DATA_DIR = '/app/data'

def compute_firing_rates(spike_times_by_neuron, go_cue_time, begin_time, end_time, bin_width):
    """
    Compute firing rates for a single trial.

    Args:
        spike_times_by_neuron: list of arrays, spike times for each neuron (absolute times)
        go_cue_time: float, absolute go cue time for this trial
        begin_time: float, start of window relative to go cue
        end_time: float, end of window relative to go cue
        bin_width: float, bin width in seconds

    Returns:
        fr: (n_neurons, n_bins) array of firing rates
        bin_centers: (n_bins,) array of bin centers relative to go cue
    """
    n_bins = int(round((end_time - begin_time) / bin_width))
    bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0 - go_cue_time

    n_neurons = len(spike_times_by_neuron)
    fr = np.zeros((n_neurons, n_bins), dtype=np.float32)

    for i, st in enumerate(spike_times_by_neuron):
        if len(st) == 0:
            continue
        # Only consider spikes in the window
        mask = (st >= bin_edges[0]) & (st < bin_edges[-1])
        st_window = st[mask]
        if len(st_window) > 0:
            counts, _ = np.histogram(st_window, bins=bin_edges)
            fr[i, :] = counts.astype(np.float32) / bin_width

    return fr, bin_centers


def preload_spike_times(nwb, unit_indices):
    """
    Preload all spike times for good units into memory.
    Returns list of arrays, one per unit.
    """
    return [nwb.units['spike_times'][uid] for uid in unit_indices]


def get_spike_times_for_trial_from_cache(all_spike_times, obs_intervals, obs_idx):
    """
    Get spike times for a specific obs_interval index using preloaded data.
    """
    spike_times_list = []
    t_start, t_stop = obs_intervals[obs_idx]
    for st in all_spike_times:
        mask = (st >= t_start) & (st < t_stop)
        spike_times_list.append(st[mask])
    return spike_times_list


def build_obs_to_trial_map(nwb, good_indices):
    """
    Build mapping from obs_intervals indices to trial table indices.
    Returns:
        obs_to_trial: list of trial indices, where obs_to_trial[obs_idx] = trial_idx
    """
    trials = nwb.trials
    n_trials = len(trials)
    units = nwb.units

    # Use the first good unit's obs_intervals (all units have the same)
    obs = units['obs_intervals'][good_indices[0]]
    n_obs = obs.shape[0]

    if n_obs == n_trials:
        # Perfect match - obs[i] corresponds to trial[i]
        return list(range(n_trials))

    # Match by start time
    trial_starts = np.array([trials['start_time'][i] for i in range(n_trials)])
    obs_to_trial = []
    for i in range(n_obs):
        diffs = np.abs(trial_starts - obs[i, 0])
        min_idx = np.argmin(diffs)
        if diffs[min_idx] < 0.5:  # 0.5s tolerance
            obs_to_trial.append(int(min_idx))
        else:
            obs_to_trial.append(-1)  # no match

    return obs_to_trial


def process_session(nwb_path, sample_mode=False):
    """
    Process one NWB session file.

    Returns None if session doesn't pass criteria, otherwise returns a dict with:
        - neural: list of (n_neurons, n_bins) arrays per trial
        - input: list of (n_input, n_bins) arrays per trial
        - output: list of (n_output,) arrays per trial
        - brain_region_labels: list of str per neuron
        - subject_id: str
        - session_id: str
        - tongue_y_all: list of tongue_y values for percentile computation
    """
    io = pynwb.NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()

    trials = nwb.trials
    n_trials = len(trials)
    units = nwb.units
    n_units = len(units)

    subject_id = nwb.subject.description  # e.g. 'SC015' - mouse name from description
    session_id = nwb.session_id if nwb.session_id else os.path.basename(nwb_path)

    # ---- 1. Find good units ----
    good_indices = []
    for i in range(n_units):
        if units['classification'][i] == 'good':
            good_indices.append(i)

    if len(good_indices) == 0:
        print(f"  Skipping {os.path.basename(nwb_path)}: 0 good units")
        io.close()
        return None

    # ---- 2. Build obs_intervals to trial mapping ----
    obs_to_trial = build_obs_to_trial_map(nwb, good_indices)
    # Set of trial indices that have neural data
    trials_with_neural = set(t for t in obs_to_trial if t >= 0)
    # Build reverse mapping: trial_idx -> obs_idx
    trial_to_obs = {}
    for obs_idx, trial_idx in enumerate(obs_to_trial):
        if trial_idx >= 0:
            trial_to_obs[trial_idx] = obs_idx

    # ---- 3. Get behavioral events ----
    be = nwb.acquisition['BehavioralEvents']
    go_start_times = be.time_series['go_start_times'].timestamps[:]
    sample_start_times = be.time_series['sample_start_times'].timestamps[:]

    # ---- 4. Check session selection criteria ----
    # Compute performance on control (no photostim) non-early-lick trials
    # Use ALL trials for session selection (not just those with neural data)
    control_no_early = 0
    correct_control = 0
    correct_left = 0
    correct_right = 0

    for i in range(n_trials):
        is_control = trials['photostim_onset'][i] == 'N/A'
        is_no_early = trials['early_lick'][i] == 'no early'
        is_hit = trials['outcome'][i] == 'hit'
        instruction = trials['trial_instruction'][i]

        if is_control and is_no_early:
            control_no_early += 1
            if is_hit:
                correct_control += 1
                if instruction == 'left':
                    correct_left += 1
                else:
                    correct_right += 1

    if control_no_early == 0:
        print(f"  Skipping {os.path.basename(nwb_path)}: no control non-early trials")
        io.close()
        return None

    perf = correct_control / control_no_early
    if perf <= MIN_PERF or correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
        print(f"  Skipping {os.path.basename(nwb_path)}: perf={perf:.1%}, L={correct_left}, R={correct_right}")
        io.close()
        return None

    # ---- 5. Filter trials ----
    # Per methods: "Early lick trials and no response trials were excluded for analysis"
    # Also must have neural data (be in obs_intervals)
    valid_trial_indices = []
    for i in range(n_trials):
        # Must have neural data
        if i not in trials_with_neural:
            continue

        outcome = trials['outcome'][i]
        early = trials['early_lick'][i]

        # Exclude early lick and ignore (no response) trials
        if early != 'no early':
            continue
        if outcome == 'ignore':
            continue

        valid_trial_indices.append(i)

    if len(valid_trial_indices) < 2:
        print(f"  Skipping {os.path.basename(nwb_path)}: <2 valid trials after filtering")
        io.close()
        return None

    if sample_mode and len(valid_trial_indices) > 50:
        valid_trial_indices = valid_trial_indices[:50]

    # ---- 6. Get brain region labels for good units ----
    brain_region_labels = []
    for i in good_indices:
        anno = units['anno_name'][i]
        if anno is None or anno == '' or anno == 'nan':
            anno = 'unknown'
        brain_region_labels.append(str(anno))

    # ---- 7. Preload spike times for good units (for speed) ----
    print(f"  Loading spike times for {len(good_indices)} good units...")
    all_spike_times = preload_spike_times(nwb, good_indices)
    obs_intervals = units['obs_intervals'][good_indices[0]]

    # ---- 8. Get tongue tracking data ----
    bts = nwb.acquisition['BehavioralTimeSeries']
    tongue_ts = bts.time_series['Camera0_side_TongueTracking']
    tongue_data = tongue_ts.data[:]  # (n_frames, 3): x, y, likelihood
    tongue_times = tongue_ts.timestamps[:]
    tongue_y = tongue_data[:, 1]
    tongue_likelihood = tongue_data[:, 2]

    # ---- 9. Process each valid trial ----
    neural_list = []
    input_list = []
    output_list = []
    tongue_y_trial_values = []  # for computing session percentiles

    for trial_idx in valid_trial_indices:
        go_cue = go_start_times[trial_idx]
        obs_idx = trial_to_obs[trial_idx]

        # Get spike times for good units in this trial
        spike_times_trial = get_spike_times_for_trial_from_cache(all_spike_times, obs_intervals, obs_idx)

        # Compute firing rates
        fr, bin_centers = compute_firing_rates(
            spike_times_trial, go_cue, BEGIN_TIME, END_TIME, BIN_WIDTH
        )

        # ---- Inputs ----
        # 1. Time from tone onset (continuous, time-varying)
        # Find the last sample_start before this go cue (the actual sample epoch)
        t_start = trials['start_time'][trial_idx]
        valid_samples = sample_start_times[
            (sample_start_times >= t_start) & (sample_start_times < go_cue)
        ]
        if len(valid_samples) > 0:
            tone_onset = valid_samples[-1]  # last sample start = actual tone onset
        else:
            tone_onset = go_cue - 1.85  # fallback: typical sample-delay duration

        tone_onset_rel = tone_onset - go_cue  # relative to go cue (should be ~-1.85)
        time_from_tone = bin_centers - tone_onset_rel  # time since tone onset at each bin

        # 2. Whether photostimulation is on at every time point (binary)
        photostim_on = np.zeros(len(bin_centers), dtype=np.float32)
        photostim_onset_trial = trials['photostim_onset'][trial_idx]
        if photostim_onset_trial != 'N/A':
            onset_val = float(photostim_onset_trial)
            dur_val = float(trials['photostim_duration'][trial_idx])
            # photostim_onset is relative to trial start
            photostim_start_abs = t_start + onset_val
            photostim_stop_abs = photostim_start_abs + dur_val
            # Convert to relative to go cue
            ps_start_rel = photostim_start_abs - go_cue
            ps_stop_rel = photostim_stop_abs - go_cue
            photostim_on[(bin_centers >= ps_start_rel) & (bin_centers < ps_stop_rel)] = 1.0

        input_trial = np.stack([time_from_tone.astype(np.float32), photostim_on], axis=0)  # (2, n_bins)

        # ---- Outputs ----
        # 1. Choice: left=0, right=1
        instruction = trials['trial_instruction'][trial_idx]
        outcome = trials['outcome'][trial_idx]
        if outcome == 'hit':
            choice = 0 if instruction == 'left' else 1
        elif outcome == 'miss':
            choice = 1 if instruction == 'left' else 0  # miss means licked wrong side
        else:
            choice = 0  # shouldn't happen after filtering

        # 2. Outcome: ignore=0, miss=1, hit=2
        outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
        outcome_val = outcome_map.get(outcome, 0)

        # 3. Early lick: no=0, yes=1 (should be all 0 after filtering)
        early_lick_val = 0 if trials['early_lick'][trial_idx] == 'no early' else 1

        # 4. Tongue y-position (discretized per session) - get raw value first
        # Get tongue y values during this trial window using vectorized operations
        trial_window_start = go_cue + BEGIN_TIME - BIN_WIDTH
        trial_window_end = go_cue + END_TIME + BIN_WIDTH
        tw_mask = (tongue_times >= trial_window_start) & (tongue_times < trial_window_end)
        tw_times = tongue_times[tw_mask]
        tw_y = tongue_y[tw_mask]
        tw_lh = tongue_likelihood[tw_mask]

        # Filter by likelihood > 0.5
        lh_mask = tw_lh > 0.5
        tw_times_good = tw_times[lh_mask]
        tw_y_good = tw_y[lh_mask]

        tongue_y_bins = np.full(len(bin_centers), np.nan, dtype=np.float32)
        if len(tw_times_good) > 0:
            # Use digitize for fast binning
            abs_bin_edges = np.linspace(BEGIN_TIME, END_TIME, len(bin_centers) + 1) + go_cue
            bin_assignments = np.digitize(tw_times_good, abs_bin_edges) - 1
            for b in range(len(bin_centers)):
                b_mask = bin_assignments == b
                if np.any(b_mask):
                    tongue_y_bins[b] = np.mean(tw_y_good[b_mask])

        tongue_y_trial_values.append(tongue_y_bins)

        output_trial = np.array([choice, outcome_val, early_lick_val], dtype=np.int64)  # (3,) - tongue added later

        neural_list.append(fr)
        input_list.append(input_trial)
        output_list.append(output_trial)

    io.close()

    return {
        'neural': neural_list,
        'input': input_list,
        'output': output_list,
        'brain_region_labels': brain_region_labels,
        'subject_id': subject_id,
        'session_id': session_id,
        'tongue_y_trial_values': tongue_y_trial_values,
        'n_good_units': len(good_indices),
        'n_valid_trials': len(valid_trial_indices),
        'perf': perf,
    }


def map_region_to_major(anno_name):
    """
    Map detailed CCF annotation to major brain region.
    Based on the reference code regions: ALM, Medulla, Midbrain, Striatum,
    Thalamus, Pons, Cerebellum, Hypothalamus, Hippocampus, OtherCortex, etc.
    """
    anno = str(anno_name).strip().lower()

    # ALM: Secondary motor area (MOs) and nearby
    if any(x in anno for x in ['secondary motor area', 'primary motor area', 'frontal pole']):
        return 'ALM'

    # Striatum
    if any(x in anno for x in ['caudoputamen', 'striatum', 'nucleus accumbens', 'striatal']):
        return 'Striatum'

    # Pallidum
    if any(x in anno for x in ['pallidum', 'globus pallidus', 'substantia innominata',
                                 'bed nuclei', 'diagonal band']):
        return 'Pallidum'

    # Thalamus
    if any(x in anno for x in ['thalamus', 'geniculate', 'habenula',
                                 'zona incerta', 'reticular nucleus']):
        if 'hypothalamus' not in anno:
            return 'Thalamus'

    # Hypothalamus
    if 'hypothalamus' in anno or 'hypothalamic' in anno:
        return 'Hypothalamus'

    # Midbrain
    if any(x in anno for x in ['midbrain', 'superior colliculus', 'inferior colliculus',
                                 'substantia nigra', 'ventral tegmental',
                                 'periaqueductal', 'red nucleus', 'pretectal',
                                 'anterior pretectal', 'pedunculopontine']):
        return 'Midbrain'

    # Pons
    if any(x in anno for x in ['pons', 'pontine', 'parabrachial', 'locus ceruleus',
                                 'barrington', 'tegmental nucleus', 'raphe']):
        return 'Pons'

    # Medulla
    if any(x in anno for x in ['medulla', 'medullary', 'facial motor', 'hypoglossal',
                                 'vestibular', 'cochlear', 'spinal trigeminal',
                                 'gigantocellular', 'paragigantocellular',
                                 'lateral reticular', 'inferior olive',
                                 'dorsal column']):
        return 'Medulla'

    # Cerebellum
    if any(x in anno for x in ['cerebellum', 'cerebellar', 'purkinje', 'lobul']):
        return 'Cerebellum'

    # Hippocampus
    if any(x in anno for x in ['hippocampus', 'hippocampal', 'dentate', 'subiculum',
                                 'entorhinal', 'field ca', 'ammon']):
        return 'Hippocampus'

    # Olfactory
    if any(x in anno for x in ['olfactory', 'piriform', 'olfact']):
        return 'Olfactory'

    # Cortical subplate
    if any(x in anno for x in ['claustrum', 'endopiriform', 'cortical subplate',
                                 'basolateral amygdal', 'lateral amygdal',
                                 'basomedial amygdal', 'central amygdal',
                                 'medial amygdal', 'amygdal']):
        return 'CorticalSubplate'

    # Orbital cortex
    if any(x in anno for x in ['orbital area', 'orbital cortex']):
        return 'Orbital'

    # Other cortex (somatosensory, visual, auditory, insular, etc.)
    if any(x in anno for x in ['cortex', 'cortical', 'somatosensory', 'visual',
                                 'auditory', 'insular', 'agranular', 'retrosplenial',
                                 'cingulate', 'prelimbic', 'infralimbic',
                                 'anterior cingulate', 'layer', 'gustatory',
                                 'temporal association', 'perirhinal',
                                 'ectorhinal', 'visceral']):
        return 'OtherCortex'

    # Fiber tracts
    if any(x in anno for x in ['fiber tract', 'corpus callosum', 'internal capsule',
                                 'cerebral peduncle', 'anterior commissure',
                                 'optic', 'tract']):
        return 'FiberTract'

    if anno == 'unknown' or anno == '' or anno == 'nan':
        return 'unknown'

    return 'OtherCortex'  # default


def discretize_tongue_y(tongue_y_values, p40, p60):
    """
    Discretize tongue y values:
    0: < 40th percentile
    1: 40th to 60th percentile
    2: > 60th percentile
    NaN values get class 0 (below 40th percentile - represents no tongue visible)
    """
    result = np.zeros(len(tongue_y_values), dtype=np.int64)
    for i, val in enumerate(tongue_y_values):
        if np.isnan(val):
            result[i] = 0  # no tongue visible = low position
        elif val < p40:
            result[i] = 0
        elif val <= p60:
            result[i] = 1
        else:
            result[i] = 2
    return result


def convert_data(data_dir, sample_mode=False, max_sessions=None):
    """Main conversion function."""

    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])

    all_sessions = []
    all_subject_ids = []

    session_count = 0

    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])

        if sample_mode:
            files = files[:1]  # only first session per subject in sample mode

        for fname in files:
            if max_sessions and session_count >= max_sessions:
                break

            fpath = os.path.join(subj_dir, fname)
            print(f"Processing {subj}/{fname}...")

            result = process_session(fpath, sample_mode=sample_mode)
            if result is not None:
                all_sessions.append(result)
                session_count += 1

        if max_sessions and session_count >= max_sessions:
            break

    if len(all_sessions) == 0:
        print("ERROR: No valid sessions found!")
        return None

    # ---- Build the data dictionary ----

    # Subjects
    unique_subjects = list(OrderedDict.fromkeys(s['subject_id'] for s in all_sessions))
    subject_idx = np.array([unique_subjects.index(s['subject_id']) for s in all_sessions], dtype=np.int64)

    # Brain regions
    all_region_labels = set()
    for s in all_sessions:
        for label in s['brain_region_labels']:
            all_region_labels.add(map_region_to_major(label))

    # Remove 'unknown' and 'FiberTract' if present, sort for consistency
    brain_regions = sorted([r for r in all_region_labels if r not in ('unknown', 'FiberTract')])
    # Add unknown and FiberTract at the end if they exist
    if 'FiberTract' in all_region_labels:
        brain_regions.append('FiberTract')
    if 'unknown' in all_region_labels:
        brain_regions.append('unknown')

    brain_region_idx = []
    for s in all_sessions:
        idx = np.array([brain_regions.index(map_region_to_major(label))
                        for label in s['brain_region_labels']], dtype=np.int64)
        brain_region_idx.append(idx)

    # ---- Discretize tongue y per session ----
    neural_all = []
    input_all = []
    output_all = []

    for s in all_sessions:
        # Compute session-level percentiles for tongue y
        all_tongue_y = np.concatenate([v for v in s['tongue_y_trial_values']])
        valid_tongue = all_tongue_y[~np.isnan(all_tongue_y)]

        if len(valid_tongue) > 0:
            p40 = np.percentile(valid_tongue, 40)
            p60 = np.percentile(valid_tongue, 60)
        else:
            p40 = 0.0
            p60 = 0.0

        neural_session = s['neural']
        input_session = s['input']
        output_session = []

        for t_idx in range(len(s['output'])):
            base_output = s['output'][t_idx]  # (3,): choice, outcome, early_lick
            tongue_y_bins = s['tongue_y_trial_values'][t_idx]
            tongue_y_disc = discretize_tongue_y(tongue_y_bins, p40, p60)

            # Make tongue y a time-varying output: (1, n_bins)
            # Other outputs are per-trial: (3,)
            # We need to combine: make all outputs together
            # Per-trial outputs: choice, outcome, early_lick -> shape (n_output,)
            # Time-varying output: tongue_y -> shape (1, n_bins)
            # Since some are per-trial and one is time-varying, we make all time-varying
            n_bins = tongue_y_disc.shape[0]
            output_trial = np.zeros((4, n_bins), dtype=np.int64)
            output_trial[0, :] = base_output[0]  # choice (constant across time)
            output_trial[1, :] = base_output[1]  # outcome (constant across time)
            output_trial[2, :] = base_output[2]  # early_lick (constant across time)
            output_trial[3, :] = tongue_y_disc   # tongue y (time-varying)

            output_session.append(output_trial)

        neural_all.append(neural_session)
        input_all.append(input_session)
        output_all.append(output_session)

    # ---- Build metadata ----
    n_bins = int(round((END_TIME - BEGIN_TIME) / BIN_WIDTH))

    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,

        'subjects': unique_subjects,
        'subject_idx': subject_idx,

        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,

        'input_names': ['time_from_tone_onset', 'photostimulation'],
        'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
        'output_values': [
            ['left', 'right'],           # choice
            ['ignore', 'miss', 'hit'],   # outcome
            ['no', 'yes'],               # early_lick
            ['low', 'mid', 'high'],      # tongue_y_position
        ],

        'metadata': {
            'task_description': 'Auditory delayed response task: mice hear tones during sample epoch, wait through delay, then lick left or right at go cue',
            'time_bin_size': BIN_WIDTH * 1000,  # in ms
            'temporal_alignment_event': 'Go cue onset',
            'off_start': BEGIN_TIME,
            'off_end': END_TIME,
            'n_bins': n_bins,
            'bin_width_s': BIN_WIDTH,
            'session_selection': f'Performance >{MIN_PERF*100}%, >= {MIN_CORRECT_PER_SIDE} correct per side',
            'trial_filtering': 'Excluded early lick and no-response (ignore) trials',
            'qc_method': 'Classifier-based QC (good classification in NWB)',
            'reference_paper': 'Li et al., Brain-wide analysis reveals movement encoding structured across and within brain areas',
        }
    }

    return data


def print_stats(data):
    """Print sanity check statistics."""
    n_sessions = len(data['neural'])
    n_trials_total = sum(len(data['neural'][s]) for s in range(n_sessions))
    n_neurons_total = sum(data['neural'][s][0].shape[0] for s in range(n_sessions))
    n_subjects = len(data['subjects'])

    print(f"\n{'='*60}")
    print(f"CONVERSION STATISTICS")
    print(f"{'='*60}")
    print(f"Sessions: {n_sessions}")
    print(f"Subjects: {n_subjects}")
    print(f"Total trials: {n_trials_total}")
    print(f"Total neurons (sum across sessions): {n_neurons_total}")
    print(f"Trials per session: min={min(len(data['neural'][s]) for s in range(n_sessions))}, "
          f"max={max(len(data['neural'][s]) for s in range(n_sessions))}, "
          f"mean={n_trials_total/n_sessions:.1f}")

    neurons_per_session = [data['neural'][s][0].shape[0] for s in range(n_sessions)]
    print(f"Neurons per session: min={min(neurons_per_session)}, max={max(neurons_per_session)}, "
          f"mean={np.mean(neurons_per_session):.1f}")

    n_bins = data['neural'][0][0].shape[1]
    print(f"Time bins per trial: {n_bins}")
    print(f"Bin width: {data['metadata']['time_bin_size']} ms")
    print(f"Time window: {data['metadata']['off_start']} to {data['metadata']['off_end']} s")

    # Brain region distribution
    print(f"\nBrain region distribution:")
    region_counts = {r: 0 for r in data['brain_regions']}
    for s in range(n_sessions):
        for idx in data['brain_region_idx'][s]:
            region_counts[data['brain_regions'][idx]] += 1
    for region, count in sorted(region_counts.items(), key=lambda x: -x[1]):
        print(f"  {region}: {count}")

    # Sessions per subject
    print(f"\nSessions per subject:")
    from collections import Counter
    subj_counts = Counter(data['subject_idx'])
    for subj_idx, count in sorted(subj_counts.items()):
        print(f"  {data['subjects'][subj_idx]}: {count}")

    # Output distribution
    print(f"\nOutput distributions:")
    for out_dim in range(len(data['output_names'])):
        all_vals = []
        for s in range(n_sessions):
            for t in range(len(data['output'][s])):
                out = data['output'][s][t]
                if out.ndim == 1:
                    all_vals.append(out[out_dim])
                else:
                    all_vals.extend(out[out_dim, :].tolist())
        vals, counts = np.unique(all_vals, return_counts=True)
        total = sum(counts)
        print(f"  {data['output_names'][out_dim]}:")
        for v, c in zip(vals, counts):
            name = data['output_values'][out_dim][int(v)] if int(v) < len(data['output_values'][out_dim]) else f'val{int(v)}'
            print(f"    {name} ({int(v)}): {c} ({c/total*100:.1f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert NWB data to decoder format')
    parser.add_argument('--sample', action='store_true', help='Create a small sample dataset')
    parser.add_argument('--max-sessions', type=int, default=None, help='Maximum number of sessions to process')
    parser.add_argument('--output', type=str, default=None, help='Output pickle file path')
    args = parser.parse_args()

    if args.sample:
        print("Creating sample dataset...")
        output_file = args.output or '/app/sample_data.pkl'
        data = convert_data(DATA_DIR, sample_mode=True, max_sessions=5)
    else:
        print("Creating full dataset...")
        output_file = args.output or '/app/converted_data.pkl'
        data = convert_data(DATA_DIR, sample_mode=False, max_sessions=args.max_sessions)

    if data is not None:
        print_stats(data)

        print(f"\nSaving to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved successfully. File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
    else:
        print("Conversion failed!")
        sys.exit(1)
