"""
Convert NWB data from the MAP dataset (brain-wide Neuropixels recordings during
an auditory delayed-response task) into the standardised dictionary format
expected by the neural decoder pipeline.

Key processing decisions (aligned with the reference papers and code):
- Unit filtering: classification == 'good' (classifier-based QC from ChenLiuEtAl2023)
- Session filtering: >65% correct on control trials (no stim, no early lick,
  no auto/free water, response present), and >=50 correct left + >=50 correct right.
- Trial filtering: exclude auto_water and free_water trials only. Keep early-lick,
  no-response, and photostim trials so the decoder can predict those output categories.
- Temporal alignment: go cue onset (time 0). Extract -2.5 s to +1.5 s.
- Bin width: 50 ms non-overlapping bins for firing rates.
- Brain regions: CCF annotations (anno_name) mapped to broad categories matching
  the reference code (ALM, Striatum, Thalamus, Midbrain, Medulla, etc.).
- Tongue y-position: from side-camera DLC tracking, likelihood > 0.9 = visible.
  Per-session percentile discretisation (40th/60th) over visible positions.
"""

import os
import sys
import json
import pickle
import numpy as np
import h5py
from collections import OrderedDict

# ── constants ──────────────────────────────────────────────────────────────────
DATA_DIR = '/app/data'
SAVE_PATH = '/app/converted_data.pkl'

BIN_WIDTH = 0.05       # 50 ms
T_START = -2.5         # seconds relative to go cue
T_END = 1.5
TONGUE_LIKELIHOOD_THRESH = 0.9

# ── CCF annotation → broad region mapping ─────────────────────────────────────
# Based on the Allen CCF ontology and the 14 broad categories used in the
# reference code (preprocessing_DJ_2022Aug.py): ALM, Striatum, Thalamus,
# Midbrain, Medulla, Pons, Cerebellum, Hypothalamus, Hippocampus, Orbital,
# OtherCortex, Olfactory, CorticalSubplate, Pallidum.

_REGION_KEYWORDS = OrderedDict([
    # Order matters: first match wins. More specific before more general.
    ('Secondary motor area', 'ALM'),

    # Orbital cortex
    ('Orbital area', 'Orbital'),

    # Olfactory areas
    ('Anterior olfactory nucleus', 'Olfactory'),
    ('Accessory olfactory bulb', 'Olfactory'),
    ('Olfactory tubercle', 'Olfactory'),
    ('Olfactory areas', 'Olfactory'),
    ('Piriform area', 'Olfactory'),
    ('Taenia tecta', 'Olfactory'),
    ('Dorsal peduncular area', 'Olfactory'),
    ('Endopiriform nucleus', 'Olfactory'),

    # Hippocampal formation
    ('Field CA', 'Hippocampus'),
    ('Dentate gyrus', 'Hippocampus'),
    ('Subiculum', 'Hippocampus'),
    ('Postsubiculum', 'Hippocampus'),
    ('Hippocampal formation', 'Hippocampus'),
    ('Entorhinal area', 'Hippocampus'),

    # Cortical subplate (amygdala, claustrum)
    ('Cortical subplate', 'CorticalSubplate'),
    ('Claustrum', 'CorticalSubplate'),
    ('amygdalar nucleus', 'CorticalSubplate'),
    ('amygdalar area', 'CorticalSubplate'),
    ('Anterior amygdalar', 'CorticalSubplate'),
    ('Intercalated amygdalar', 'CorticalSubplate'),
    ('Posterior amygdalar', 'CorticalSubplate'),
    ('Lateral amygdalar', 'CorticalSubplate'),
    ('Medial amygdalar', 'CorticalSubplate'),

    # Striatum
    ('Caudoputamen', 'Striatum'),
    ('Nucleus accumbens', 'Striatum'),
    ('Fundus of striatum', 'Striatum'),
    ('Striatum', 'Striatum'),
    ('Lateral septal nucleus', 'Striatum'),
    ('Septofimbrial nucleus', 'Striatum'),
    ('Triangular nucleus of septum', 'Striatum'),

    # Pallidum
    ('Globus pallidus', 'Pallidum'),
    ('Pallidum', 'Pallidum'),
    ('Substantia innominata', 'Pallidum'),
    ('Bed nuclei of the stria terminalis', 'Pallidum'),

    # Thalamus (must come before Hypothalamus substring match)
    ('thalamus', 'Thalamus'),
    ('Thalamus', 'Thalamus'),
    ('habenula', 'Thalamus'),
    ('Zona incerta', 'Thalamus'),
    ('Fields of Forel', 'Thalamus'),
    ('Reticular nucleus of the thalamus', 'Thalamus'),
    ('Paraventricular nucleus of the thalamus', 'Thalamus'),
    ('Rhomboid nucleus', 'Thalamus'),
    ('Perireunensis nucleus', 'Thalamus'),
    ('geniculate complex', 'Thalamus'),
    ('geniculate nucleus', 'Thalamus'),
    ('Suprageniculate nucleus', 'Thalamus'),
    ('Peripeduncular nucleus', 'Thalamus'),

    # Hypothalamus
    ('hypothalamic', 'Hypothalamus'),
    ('Hypothalamus', 'Hypothalamus'),
    ('Lateral hypothalamic', 'Hypothalamus'),
    ('Lateral preoptic', 'Hypothalamus'),
    ('Posterior hypothalamic', 'Hypothalamus'),
    ('Tuberomammillary', 'Hypothalamus'),
    ('Parasubthalamic', 'Hypothalamus'),
    ('Subthalamic nucleus', 'Hypothalamus'),

    # Midbrain
    ('Superior colliculus', 'Midbrain'),
    ('Inferior colliculus', 'Midbrain'),
    ('Red nucleus', 'Midbrain'),
    ('Substantia nigra', 'Midbrain'),
    ('Ventral tegmental area', 'Midbrain'),
    ('Periaqueductal gray', 'Midbrain'),
    ('Midbrain reticular nucleus', 'Midbrain'),
    ('Midbrain', 'Midbrain'),
    ('Anterior pretectal', 'Midbrain'),
    ('Posterior pretectal', 'Midbrain'),
    ('Pedunculopontine nucleus', 'Midbrain'),
    ('Nucleus of the optic tract', 'Midbrain'),
    ('Nucleus of the brachium', 'Midbrain'),
    ('Nucleus sagulum', 'Midbrain'),
    ('Dorsal terminal nucleus', 'Midbrain'),
    ('Medial terminal nucleus', 'Midbrain'),
    ('Nucleus of the lateral lemniscus', 'Midbrain'),
    ('Subparafascicular', 'Midbrain'),

    # Pons
    ('Pons', 'Pons'),
    ('Pontine reticular', 'Pons'),
    ('Locus ceruleus', 'Pons'),
    ('Parabrachial nucleus', 'Pons'),
    ('Tegmental reticular nucleus', 'Pons'),
    ('Koelliker-Fuse', 'Pons'),

    # Cerebellum
    ('Cerebellum', 'Cerebellum'),
    ('Fastigial nucleus', 'Cerebellum'),
    ('Interposed nucleus', 'Cerebellum'),
    ('Infracerebellar', 'Cerebellum'),
    ('Copula pyramidis', 'Cerebellum'),
    ('Crus 1', 'Cerebellum'),
    ('Crus 2', 'Cerebellum'),
    ('Declive', 'Cerebellum'),
    ('Lingula', 'Cerebellum'),
    ('Lobule', 'Cerebellum'),
    ('Lobules', 'Cerebellum'),
    ('Nodulus', 'Cerebellum'),
    ('Pyramus', 'Cerebellum'),
    ('Simple lobule', 'Cerebellum'),
    ('Paramedian lobule', 'Cerebellum'),
    ('Uvula', 'Cerebellum'),

    # Medulla
    ('Medulla', 'Medulla'),
    ('Medullary reticular', 'Medulla'),
    ('Gigantocellular', 'Medulla'),
    ('Magnocellular reticular', 'Medulla'),
    ('Facial motor nucleus', 'Medulla'),
    ('Hypoglossal', 'Medulla'),
    ('Dorsal motor nucleus of the vagus', 'Medulla'),
    ('Nucleus of the solitary tract', 'Medulla'),
    ('Nucleus raphe', 'Medulla'),
    ('Spinal nucleus of the trigeminal', 'Medulla'),
    ('vestibular nucleus', 'Medulla'),
    ('Vestibular nucleus', 'Medulla'),
    ('Lateral reticular nucleus', 'Medulla'),
    ('Intermediate reticular nucleus', 'Medulla'),
    ('Parvicellular reticular', 'Medulla'),
    ('Paragigantocellular', 'Medulla'),
    ('Parapyramidal', 'Medulla'),
    ('Parasolitary', 'Medulla'),
    ('External cuneate', 'Medulla'),
    ('Inferior olivary', 'Medulla'),
    ('Nucleus of Roller', 'Medulla'),
    ('Nucleus x', 'Medulla'),
    ('Spinal vestibular', 'Medulla'),
    ('Superior vestibular', 'Medulla'),
    ('Lateral vestibular', 'Medulla'),
    ('Medial vestibular', 'Medulla'),
])


def map_anno_to_region(anno_name):
    """Map a CCF annotation string to a broad brain region category."""
    if not anno_name:
        return 'Unknown'
    for keyword, region in _REGION_KEYWORDS.items():
        if keyword in anno_name:
            return region
    # Fallback: any remaining cortical areas
    cortex_keywords = [
        'area', 'cortex', 'motor', 'somatosensory', 'visual', 'auditory',
        'cingulate', 'retrosplenial', 'prelimbic', 'infralimbic', 'gustatory',
        'visceral', 'temporal', 'ectorhinal', 'perirhinal', 'Frontal pole',
        'Agranular insular', 'Supplemental somatosensory',
    ]
    for kw in cortex_keywords:
        if kw.lower() in anno_name.lower():
            return 'OtherCortex'
    return 'Unknown'


def bin_spike_times(spike_times, t_start, t_end, bin_width):
    """Bin spike times into non-overlapping bins and return firing rates.

    Args:
        spike_times: 1-D array of spike times (already aligned to event).
        t_start, t_end: window boundaries.
        bin_width: bin width in seconds.

    Returns:
        firing_rates: 1-D array of shape (n_bins,).
    """
    n_bins = int(round((t_end - t_start) / bin_width))
    edges = np.linspace(t_start, t_end, n_bins + 1)
    counts, _ = np.histogram(spike_times, bins=edges)
    return counts / bin_width


def get_bin_centers(t_start, t_end, bin_width):
    n_bins = int(round((t_end - t_start) / bin_width))
    edges = np.linspace(t_start, t_end, n_bins + 1)
    return (edges[:-1] + edges[1:]) / 2


def process_session(nwb_path):
    """Load and process one NWB session file.

    Returns None if the session doesn't meet selection criteria.
    Otherwise returns a dict with processed data for this session.
    """
    with h5py.File(nwb_path, 'r') as f:
        # ── trial info ─────────────────────────────────────────────────
        n_trials = len(f['intervals/trials/id'])
        outcome = np.array([x.decode() for x in f['intervals/trials/outcome'][:]])
        early_lick = np.array([x.decode() for x in f['intervals/trials/early_lick'][:]])
        trial_instruction = np.array([x.decode() for x in f['intervals/trials/trial_instruction'][:]])
        auto_water = f['intervals/trials/auto_water'][:]
        free_water = f['intervals/trials/free_water'][:]
        photostim_onset_str = np.array(
            [x.decode() for x in f['intervals/trials/photostim_onset'][:]])
        photostim_dur_str = np.array(
            [x.decode() for x in f['intervals/trials/photostim_duration'][:]])

        # ── session selection criteria ─────────────────────────────────
        # Control trials: no auto/free water, no early lick, no photostim,
        # response present (not ignore)
        has_stim = photostim_onset_str != 'N/A'
        control_mask = (
            (auto_water == 0) & (free_water == 0) &
            (early_lick == 'no early') & (~has_stim) &
            (outcome != 'ignore')
        )
        n_control = control_mask.sum()
        if n_control == 0:
            return None

        correct_control = (outcome == 'hit') & control_mask
        performance = correct_control.sum() / n_control
        n_correct_left = ((trial_instruction == 'left') & correct_control).sum()
        n_correct_right = ((trial_instruction == 'right') & correct_control).sum()

        if performance <= 0.65 or n_correct_left < 50 or n_correct_right < 50:
            return None

        # ── trial filtering: keep all except auto/free water ───────────
        trial_mask = (auto_water == 0) & (free_water == 0)
        trial_indices = np.where(trial_mask)[0]
        if len(trial_indices) < 2:
            return None

        # ── timing ─────────────────────────────────────────────────────
        go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
        trial_start_times = f['intervals/trials/start_time'][:]

        # Tone onset: find last sample_start before each go cue
        sample_start_ts = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
        tone_onset_rel_go = np.full(n_trials, np.nan)
        for i in range(n_trials):
            before = sample_start_ts[sample_start_ts < go_times[i]]
            if len(before) > 0:
                tone_onset_rel_go[i] = before[-1] - go_times[i]

        # ── unit filtering ─────────────────────────────────────────────
        clf_raw = f['units/classification'][:]
        if clf_raw.dtype == object:
            classification = np.array([x.decode() if isinstance(x, bytes) else str(x)
                                       for x in clf_raw])
            good_mask = classification == 'good'
        else:
            # No classifier QC available (all NaN) – skip session
            return None
        n_units_total = len(classification)
        good_indices = np.where(good_mask)[0]

        if len(good_indices) == 0:
            return None

        # ── brain regions from anno_name ───────────────────────────────
        anno_names = np.array(
            [x.decode() if isinstance(x, bytes) else str(x)
             for x in f['units/anno_name'][:]])
        unit_regions = [map_anno_to_region(anno_names[i]) for i in good_indices]

        # ── spike times ────────────────────────────────────────────────
        spike_times_flat = f['units/spike_times'][:]
        spike_times_index = f['units/spike_times_index'][:]

        bin_centers = get_bin_centers(T_START, T_END, BIN_WIDTH)
        n_bins = len(bin_centers)
        n_good = len(good_indices)

        # Pre-extract spike times per good unit
        unit_spikes = []
        for uid in good_indices:
            start_idx = 0 if uid == 0 else spike_times_index[uid - 1]
            end_idx = spike_times_index[uid]
            unit_spikes.append(spike_times_flat[start_idx:end_idx])

        # ── photostim timing per trial ─────────────────────────────────
        # Compute photostim on/off times relative to go cue
        photostim_on_rel = np.full(n_trials, np.nan)
        photostim_off_rel = np.full(n_trials, np.nan)
        for i in range(n_trials):
            if photostim_onset_str[i] != 'N/A':
                onset_from_trial_start = float(photostim_onset_str[i])
                duration = float(photostim_dur_str[i])
                go_from_trial_start = go_times[i] - trial_start_times[i]
                photostim_on_rel[i] = onset_from_trial_start - go_from_trial_start
                photostim_off_rel[i] = onset_from_trial_start + duration - go_from_trial_start

        # ── tongue tracking ────────────────────────────────────────────
        tongue_data = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
        tongue_ts = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
        tongue_y = tongue_data[:, 1]
        tongue_likelihood = tongue_data[:, 2]

        # Per-session percentiles over all visible tongue positions
        visible_mask = tongue_likelihood > TONGUE_LIKELIHOOD_THRESH
        if visible_mask.sum() > 0:
            y_visible = tongue_y[visible_mask]
            pct40 = np.percentile(y_visible, 40)
            pct60 = np.percentile(y_visible, 60)
        else:
            pct40 = pct60 = 0.0  # fallback, all will be "not visible"

        # ── build trial data ───────────────────────────────────────────
        neural_trials = []
        input_trials = []
        output_trials = []

        for trial_idx in trial_indices:
            go_t = go_times[trial_idx]

            # Neural: bin spike times for each good unit
            fr_matrix = np.zeros((n_good, n_bins))
            for u, spks in enumerate(unit_spikes):
                aligned = spks - go_t
                fr_matrix[u, :] = bin_spike_times(aligned, T_START, T_END, BIN_WIDTH)
            neural_trials.append(fr_matrix)

            # Input 1: time from tone onset (continuous, time-varying)
            if np.isnan(tone_onset_rel_go[trial_idx]):
                # Fallback: use approximate value (-1.85s before go cue)
                tone_rel = -1.85
            else:
                tone_rel = tone_onset_rel_go[trial_idx]
            time_from_tone = bin_centers - tone_rel  # positive = after tone

            # Input 2: photostim on (binary, time-varying)
            photostim_binary = np.zeros(n_bins)
            if not np.isnan(photostim_on_rel[trial_idx]):
                on_t = photostim_on_rel[trial_idx]
                off_t = photostim_off_rel[trial_idx]
                photostim_binary = ((bin_centers >= on_t) & (bin_centers < off_t)).astype(float)

            input_data = np.stack([time_from_tone, photostim_binary], axis=0)  # (2, n_bins)
            input_trials.append(input_data)

            # Output 1: choice (0=left, 1=right, 2=no lick)
            out = outcome[trial_idx]
            instr = trial_instruction[trial_idx]
            if out == 'ignore':
                choice = 2  # no lick
            elif out == 'hit':
                choice = 0 if instr == 'left' else 1
            else:  # miss
                choice = 1 if instr == 'left' else 0

            # Output 2: outcome (0=ignore, 1=miss, 2=hit)
            outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[out]

            # Output 3: early lick (0=no, 1=yes)
            early_val = 0 if early_lick[trial_idx] == 'no early' else 1

            # Output 4: tongue y-position (time-varying, discretized)
            tongue_y_disc = np.full(n_bins, 3)  # default: not visible
            for b, tc in enumerate(bin_centers):
                abs_t = go_t + tc
                # Find closest video frame
                frame_idx = np.searchsorted(tongue_ts, abs_t)
                frame_idx = min(frame_idx, len(tongue_ts) - 1)
                if abs(tongue_ts[frame_idx] - abs_t) > 0.01:
                    # Also check previous frame
                    if frame_idx > 0 and abs(tongue_ts[frame_idx - 1] - abs_t) < abs(tongue_ts[frame_idx] - abs_t):
                        frame_idx = frame_idx - 1
                if tongue_likelihood[frame_idx] > TONGUE_LIKELIHOOD_THRESH:
                    y_val = tongue_y[frame_idx]
                    if y_val < pct40:
                        tongue_y_disc[b] = 0
                    elif y_val <= pct60:
                        tongue_y_disc[b] = 1
                    else:
                        tongue_y_disc[b] = 2

            # Stack per-trial outputs: 3 scalars + 1 time series
            # choice, outcome, early_lick are per-trial (scalar)
            # tongue_y is time-varying
            output_data = np.zeros((4, n_bins))
            output_data[0, :] = choice
            output_data[1, :] = outcome_val
            output_data[2, :] = early_val
            output_data[3, :] = tongue_y_disc
            output_trials.append(output_data)

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'unit_regions': unit_regions,
        'n_good_units': n_good,
    }


def main():
    subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])

    all_neural = []
    all_input = []
    all_output = []
    all_subject_idx = []
    all_brain_region_idx_per_session = []
    all_unit_regions_per_session = []

    subject_names = []
    sessions_processed = 0
    sessions_skipped = 0

    for sub in subjects:
        sub_dir = os.path.join(DATA_DIR, sub)
        nwb_files = sorted([f for f in os.listdir(sub_dir) if f.endswith('.nwb')])
        sub_has_session = False

        for nf in nwb_files:
            nwb_path = os.path.join(sub_dir, nf)
            print(f'Processing {nf}...', end=' ', flush=True)
            result = process_session(nwb_path)

            if result is None:
                print('SKIPPED (criteria not met)')
                sessions_skipped += 1
                continue

            print(f'OK ({result["n_good_units"]} units, {len(result["neural"])} trials)')

            if not sub_has_session:
                subject_names.append(sub)
                sub_has_session = True

            sub_idx = subject_names.index(sub)
            all_subject_idx.append(sub_idx)
            all_neural.append(result['neural'])
            all_input.append(result['input'])
            all_output.append(result['output'])
            all_unit_regions_per_session.append(result['unit_regions'])
            sessions_processed += 1

    print(f'\nTotal: {sessions_processed} sessions, {sessions_skipped} skipped, '
          f'{len(subject_names)} subjects')

    # ── build brain_regions and brain_region_idx ───────────────────────────
    all_regions_set = set()
    for regions_list in all_unit_regions_per_session:
        all_regions_set.update(regions_list)
    brain_regions = sorted(all_regions_set)

    brain_region_idx = []
    for regions_list in all_unit_regions_per_session:
        idx = np.array([brain_regions.index(r) for r in regions_list])
        brain_region_idx.append(idx)

    # ── assemble final dict ────────────────────────────────────────────────
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,

        'subjects': subject_names,
        'subject_idx': np.array(all_subject_idx),

        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,

        'input_names': ['time_from_tone_onset', 'photostim_on'],
        'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
        'output_values': [
            ['left', 'right', 'no_lick'],
            ['ignore', 'miss', 'hit'],
            ['no', 'yes'],
            ['below_40pct', '40_to_60pct', 'above_60pct', 'not_visible'],
        ],

        'metadata': {
            'task_description': (
                'Auditory delayed-response task: mice hear instruction tones '
                '(3 or 12 kHz), wait through a 1.2s delay, then lick left or '
                'right after an auditory go cue. ~25% of trials include '
                'optogenetic photoinhibition of ALM.'
            ),
            'time_bin_size': BIN_WIDTH * 1000,  # in ms
            'temporal_alignment_event': 'Go cue onset',
            'off_start': T_START,
            'off_end': T_END,
            'session_selection': (
                'Performance > 65% on control trials (no photostim, no early lick, '
                'no auto/free water, response present); >=50 correct left and '
                '>=50 correct right control trials.'
            ),
            'unit_selection': (
                'classification == good (classifier-based QC per Chen, Liu et al. 2023)'
            ),
            'trial_selection': (
                'Excluded auto_water and free_water trials. Kept early-lick, '
                'no-response, and photostim trials as they are decoder output/input categories.'
            ),
            'tongue_y_discretization': (
                'Side-camera DLC tongue tracking. Visible = likelihood > 0.9. '
                'Per-session percentiles: 0=<40th, 1=40-60th, 2=>60th, 3=not visible.'
            ),
        },
    }

    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nSaved to {SAVE_PATH}')


if __name__ == '__main__':
    main()
