#!/usr/bin/env python3
"""
Convert the DANDI:000363 "Mesoscale Activity Map" NWB dataset (Chen et al.) into the
decoder-ready pickle format described in the task specification.

Usage
-----
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Options
-------
    --full             process all sessions that pass the paper's session criteria (default)
    --sample           process only the first 2 selected sessions
    --show-processing  save diagnostic plots (processing_<session_id>.png) for up to 2 sessions
    --njobs N          number of worker processes (default: min(16, cpu_count))

Design decisions and their justification are documented in CONVERSION_NOTES.md.
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time
from collections import OrderedDict

import h5py
import numpy as np

# --------------------------------------------------------------------------------------
# Constants (see CONVERSION_NOTES.md Step 5)
# --------------------------------------------------------------------------------------

DATA_DIR = '/app/data'
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
ONTOLOGY_PATH = os.path.join(CACHE_DIR, 'allen_structure_graph.json')

# Trial window relative to the go cue and bin size (mandated by the Decoder Task).
OFF_START = -2.5          # s, signed time from go cue to start of the extracted trial
OFF_END = 1.5             # s, signed time from go cue to end of the extracted trial
BIN_SIZE = 0.05           # s, 50 ms bins
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))            # 80
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)         # (81,)
BIN_CENTERS = OFF_START + BIN_SIZE * (np.arange(N_BINS) + 0.5)   # (80,)

# Session inclusion criteria from the data paper ("Behavior and video tracking"):
#   overall behavioral performance (> 65%) and at least 50 correct lick left and
#   lick right trials each.
MIN_PERFORMANCE = 0.65
MIN_CORRECT_PER_DIRECTION = 50

# The decoder needs at least two trials per session to be able to train and validate.
MIN_TRIALS_PER_SESSION = 2

# DeepLabCut likelihood above which the tongue counts as visible.  The likelihood is
# strongly bimodal (~10.5% of frames > 0.9, the rest < 1e-3), so the exact value is
# immaterial; 0.9 is the DeepLabCut default cut-off.
TONGUE_LIKELIHOOD_THRESHOLD = 0.9

# Tongue y-position percentile boundaries (Decoder Task).
TONGUE_PCTL_LOW = 40.0
TONGUE_PCTL_HIGH = 60.0

# CCF anterior-posterior coordinate (um) anterior to which a cortical unit is called ALM.
# Bregma is at CCF (ML, DV, AP) = (5700, 0, 5400) um (cf. reference
# population_decoding_utils.get_ventral_medial_mask), and AP increases posteriorly, so
# "2.0 mm anterior to bregma" is z < 3400 um.  Reproduces the paper's 8717 ALM units.
BREGMA_CCF = np.array([5700.0, 0.0, 5400.0])   # (ML, DV, AP)
ALM_AP_MIN_MM = 2.0

# Coarse brain areas of datapaper Fig. 1J, matched against the Allen CCF structure graph
# by ancestor acronym.  Order matters: the first matching ancestor wins.
COARSE_GROUPS = [
    ('Medulla', 'MY'),
    ('Pons', 'P'),
    ('Midbrain', 'MB'),
    ('Cerebellum', 'CB'),
    ('Thalamus', 'TH'),
    ('Hypothalamus', 'HY'),
    ('Striatum', 'STR'),
    ('Pallidum', 'PAL'),
    ('Hippocampus', 'HPF'),
    ('Olfactory', 'OLF'),
    ('CorticalSubplate', 'CTXsp'),
    ('Orbital', 'ORB'),
    ('OtherCortex', 'Isocortex'),
]
BRAIN_REGIONS = ['ALM'] + [name for name, _ in COARSE_GROUPS] + ['Other']

INPUT_NAMES = ['time_from_tone_onset', 'photostim']
OUTPUT_NAMES = ['choice', 'outcome', 'early_lick', 'tongue_y']
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['<40th pctile', '40-60th pctile', '>60th pctile', 'not visible'],
]

CHOICE_LEFT, CHOICE_RIGHT, CHOICE_NOLICK = 0, 1, 2
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}


# --------------------------------------------------------------------------------------
# Allen CCF ontology -> coarse area
# --------------------------------------------------------------------------------------

def load_ontology(path=ONTOLOGY_PATH):
    """Return {structure name: [acronyms from root down to the structure]}."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Allen CCF structure graph not found at {path}. Download it with:\n'
            "  curl -o %s 'http://api.brain-map.org/api/v2/structure_graph_download/1.json'"
            % path)
    root = json.load(open(path))['msg'][0]
    by_name = {}

    def walk(node, acr_path):
        p = acr_path + [node['acronym']]
        by_name[node['name']] = p
        for child in node['children']:
            walk(child, p)

    walk(root, [])
    return by_name


def assign_brain_regions(anno_names, ccf_xyz, ontology):
    """Map each unit to an index into BRAIN_REGIONS.

    Args:
        anno_names: (n_units,) array of Allen CCF structure names.
        ccf_xyz: (n_units, 3) CCF coordinates (ML, DV, AP) in um.
        ontology: dict from load_ontology().

    Returns:
        (n_units,) int array of indices into BRAIN_REGIONS.
    """
    name_to_idx = {n: i for i, n in enumerate(BRAIN_REGIONS)}
    ap_mm = (BREGMA_CCF[2] - np.asarray(ccf_xyz)[:, 2]) / 1000.0   # +ve = anterior
    out = np.full(len(anno_names), name_to_idx['Other'], dtype=np.int64)
    for i, anno in enumerate(anno_names):
        path = ontology.get(anno)
        if path is None:
            continue
        group = 'Other'
        for name, acr in COARSE_GROUPS:
            if acr in path:
                group = name
                break
        # ALM is carved out of (non-orbital) cortex by its anterior-posterior position.
        if group == 'OtherCortex' and ap_mm[i] > ALM_AP_MIN_MM:
            group = 'ALM'
        out[i] = name_to_idx[group]
    return out


# --------------------------------------------------------------------------------------
# small NWB helpers
# --------------------------------------------------------------------------------------

def _dec(arr):
    """Decode an HDF5 object (bytes) column into a numpy array of str."""
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])


# --------------------------------------------------------------------------------------
# Session-level loading
# --------------------------------------------------------------------------------------

def read_trial_table(f):
    """Read the fields of intervals/trials that we need, plus the go cue times."""
    t = f['intervals/trials']
    tr = {
        'start_time': t['start_time'][:],
        'stop_time': t['stop_time'][:],
        'outcome': _dec(t['outcome'][:]),
        'instruction': _dec(t['trial_instruction'][:]),
        'early_lick': _dec(t['early_lick'][:]),
        'auto_water': t['auto_water'][:].astype(int),
        'free_water': t['free_water'][:].astype(int),
        'photostim_onset': _dec(t['photostim_onset'][:]),
        'photostim_duration': _dec(t['photostim_duration'][:]),
        'photostim_power': _dec(t['photostim_power'][:]),
    }
    go = f['acquisition/BehavioralEvents/go_start_times']['timestamps'][:]
    ntrials = len(tr['start_time'])
    if len(go) != ntrials:
        raise ValueError('number of go cues (%d) != number of trials (%d)' % (len(go), ntrials))
    tr['go_time'] = go
    # Tone (sample epoch) onset: the last sample-epoch start before the go cue.  A trial
    # whose sample/delay epoch was replayed after an early lick has several; the one that
    # actually preceded the go cue is the last.
    sample_start = np.sort(f['acquisition/BehavioralEvents/sample_start_times']['timestamps'][:])
    idx = np.searchsorted(sample_start, go, side='right') - 1
    tone = np.where(idx >= 0, sample_start[np.maximum(idx, 0)], np.nan)
    # Fall back to the nominal 0.65 s sample + 1.2 s delay if a trial has no preceding tone.
    tone = np.where(np.isnan(tone), go - 1.85, tone)
    tr['tone_time'] = tone
    return tr


def session_performance(tr):
    """Behavioral performance as defined in the data paper.

    "Overall performance was computed as the fraction of correct control trials
    (i.e. no photostimulation), excluding any early lick trials."
    Auto-water / free-water trials are excluded as well since the animal is rewarded
    regardless of its action.
    """
    control = ((tr['early_lick'] == 'no early') &
               (tr['photostim_power'] == 'N/A') &
               (tr['auto_water'] == 0) &
               (tr['free_water'] == 0))
    responded = control & (tr['outcome'] != 'ignore')
    if responded.sum() == 0:
        return 0.0
    return float((tr['outcome'][control] == 'hit').sum() / responded.sum())


def session_passes(f, tr):
    """Data-paper session inclusion criteria, plus the decoder's minimum requirements."""
    perf = session_performance(tr)
    ncorrect_left = int(((tr['outcome'] == 'hit') & (tr['instruction'] == 'left')).sum())
    ncorrect_right = int(((tr['outcome'] == 'hit') & (tr['instruction'] == 'right')).sum())
    ngood = int((_dec(f['units/classification'][:]) == 'good').sum())
    nkept = int(trial_mask(f, tr).sum())

    reason = ''
    if perf <= MIN_PERFORMANCE:
        reason = 'performance <= %.2f' % MIN_PERFORMANCE
    elif min(ncorrect_left, ncorrect_right) < MIN_CORRECT_PER_DIRECTION:
        reason = 'fewer than %d correct trials in one direction' % MIN_CORRECT_PER_DIRECTION
    elif ngood == 0:
        reason = 'no units passed quality control'
    elif nkept < MIN_TRIALS_PER_SESSION:
        reason = 'fewer than %d usable trials' % MIN_TRIALS_PER_SESSION
    return reason == '', perf, ncorrect_left, ncorrect_right, reason


def observed_trial_mask(f, tr):
    """Trials during which the units were actually being recorded.

    `units/obs_intervals` lists, per unit, the [start, stop] interval of every trial the
    unit was observed in.  In 9 of the 174 sessions the electrophysiology covers only a
    contiguous block of the behavioral session (as few as 160 of 480 trials); the
    remaining trials have no spikes at all and would otherwise be converted into
    all-zero firing rate matrices.  The intervals are identical for every unit of a
    session, so we read them for the first and the last unit and check that they agree.
    """
    u = f['units']
    index = u['obs_intervals_index'][:].astype(np.int64)
    first = u['obs_intervals'][0:index[0]]
    if len(index) > 1:
        last = u['obs_intervals'][index[-2]:index[-1]]
        if last.shape != first.shape or not np.allclose(first, last):
            raise ValueError('obs_intervals differ between units')

    start = tr['start_time']
    pos = np.searchsorted(start, first[:, 0])
    pos = np.clip(pos, 0, len(start) - 1)
    if not (np.allclose(start[pos], first[:, 0]) and
            np.allclose(tr['stop_time'][pos], first[:, 1])):
        raise ValueError('obs_intervals do not line up with the trial table')
    mask = np.zeros(len(start), dtype=bool)
    mask[pos] = True
    return mask


def trial_mask(f, tr):
    """Trials kept for conversion.

    All trials are kept except
      * auto-water and free-water trials, on which water is delivered irrespective of the
        animal's action so that `outcome` and `choice` do not reflect a decision (the
        reference `get_regular_trial_mask` excludes them too), and
      * trials outside the units' observation intervals, which carry no spikes.
    Early-lick, no-response and photostimulation trials are deliberately kept because the
    decoder must predict / receive exactly those variables.
    """
    return ((tr['auto_water'] == 0) & (tr['free_water'] == 0) &
            observed_trial_mask(f, tr))


# --------------------------------------------------------------------------------------
# Neural data
# --------------------------------------------------------------------------------------

def load_good_units(f, ontology):
    """Return spike time arrays and brain-region indices for the session's good units.

    Neuron curation follows the paper: units labelled 'good' by the region-specific
    quality-control classifier (units/classification), which also guarantees a CCF
    annotation (units/anno_name).
    """
    u = f['units']
    classification = _dec(u['classification'][:])
    good = np.flatnonzero(classification == 'good')

    anno = _dec(u['anno_name'][:])[good]
    e = f['general/extracellular_ephys/electrodes']
    exyz = np.stack([e['x'][:], e['y'][:], e['z'][:]], axis=1)
    eidx = u['electrodes'][:][good]
    ccf_xyz = exyz[eidx]

    region_idx = assign_brain_regions(anno, ccf_xyz, ontology)

    spike_times = u['spike_times'][:]
    spike_index = u['spike_times_index'][:]
    starts = np.concatenate([[0], spike_index[:-1]]).astype(np.int64)
    ends = spike_index.astype(np.int64)
    spikes = [spike_times[starts[i]:ends[i]] for i in good]

    return spikes, region_idx, anno, ccf_xyz


def bin_spikes(spikes, go_times):
    """Bin spikes into firing rates aligned to the go cue.

    Args:
        spikes: list of n_units sorted arrays of absolute spike times (s).
        go_times: (ntrials,) absolute go cue times (s).

    Returns:
        (ntrials, n_units, N_BINS) float32 array of firing rates in Hz.
    """
    ntrials = len(go_times)
    edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()    # (ntrials*81,)
    fr = np.empty((ntrials, len(spikes), N_BINS), dtype=np.float32)
    for i, st in enumerate(spikes):
        pos = np.searchsorted(st, edges).reshape(ntrials, N_BINS + 1)
        fr[:, i, :] = np.diff(pos, axis=1)
    fr /= BIN_SIZE
    return fr


# --------------------------------------------------------------------------------------
# Decoder inputs
# --------------------------------------------------------------------------------------

def build_inputs(tr, keep):
    """Build the (ntrials, 2, N_BINS) float32 decoder input array.

    input[0] = time from tone (sample epoch) onset, in seconds, at each bin centre
    input[1] = 1 while ALM photostimulation is on, 0 otherwise
    """
    go = tr['go_time'][keep]
    tone = tr['tone_time'][keep]
    n = len(go)

    inp = np.zeros((n, 2, N_BINS), dtype=np.float32)
    # Time since the tone onset at each bin centre (both expressed relative to the go cue).
    inp[:, 0, :] = (BIN_CENTERS[None, :] - (tone - go)[:, None]).astype(np.float32)

    onset_str = tr['photostim_onset'][keep]
    dur_str = tr['photostim_duration'][keep]
    start = tr['start_time'][keep]
    has_stim = onset_str != 'N/A'
    if has_stim.any():
        onset = np.zeros(n)
        dur = np.zeros(n)
        onset[has_stim] = np.array([float(x) for x in onset_str[has_stim]])
        dur[has_stim] = np.array([float(x) for x in dur_str[has_stim]])
        # photostim_onset is relative to the trial start_time; express relative to go cue.
        on_rel = start + onset - go
        off_rel = on_rel + dur
        inside = ((BIN_CENTERS[None, :] >= on_rel[:, None]) &
                  (BIN_CENTERS[None, :] < off_rel[:, None]))
        inp[:, 1, :] = (inside & has_stim[:, None]).astype(np.float32)
    return inp


# --------------------------------------------------------------------------------------
# Decoder outputs
# --------------------------------------------------------------------------------------

def choice_codes(tr, keep):
    """Lick direction chosen by the animal.

    hit    -> the animal licked the instructed port
    miss   -> the animal licked the other port
    ignore -> no lick
    (Equivalent to the reference's `behavior_report` x `task_trial_type` combination;
    verified against the first lick after the go cue in Step 10.)
    """
    outcome = tr['outcome'][keep]
    instr = tr['instruction'][keep]
    instr_code = np.where(instr == 'left', CHOICE_LEFT, CHOICE_RIGHT)
    opposite = np.where(instr == 'left', CHOICE_RIGHT, CHOICE_LEFT)
    choice = np.where(outcome == 'hit', instr_code,
                      np.where(outcome == 'miss', opposite, CHOICE_NOLICK))
    return choice.astype(np.int64)


def bin_tongue(f, go_times):
    """Bin the side-view tongue y-position onto the trial grid.

    Returns:
        y_bin: (ntrials, N_BINS) float array, mean y over the *visible* frames of the bin
               (NaN if no visible frame falls in the bin)
        visible: (ntrials, N_BINS) bool array
    """
    tt = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
    data = tt['data']
    ts = tt['timestamps'][:]
    y = data[:, 1]
    lik = data[:, 2]
    vis = lik > TONGUE_LIKELIHOOD_THRESHOLD

    ntrials = len(go_times)
    y_bin = np.full((ntrials, N_BINS), np.nan)
    lo = np.searchsorted(ts, go_times + OFF_START)
    hi = np.searchsorted(ts, go_times + OFF_END)
    for i in range(ntrials):
        sl = slice(lo[i], hi[i])
        if sl.stop <= sl.start:
            continue
        rel = ts[sl] - go_times[i]
        v = vis[sl]
        if not v.any():
            continue
        b = np.floor((rel[v] - OFF_START) / BIN_SIZE).astype(np.int64)
        np.clip(b, 0, N_BINS - 1, out=b)
        counts = np.bincount(b, minlength=N_BINS)
        sums = np.bincount(b, weights=y[sl][v], minlength=N_BINS)
        nz = counts > 0
        y_bin[i, nz] = sums[nz] / counts[nz]
    return y_bin, ~np.isnan(y_bin)


def discretize_tongue(y_bin):
    """Per-session discretisation of the binned tongue y-position (Decoder Task).

    0: y < 40th percentile of the session's visible y values
    1: 40th - 60th percentile
    2: y > 60th percentile
    3: tongue not visible
    """
    visible = ~np.isnan(y_bin)
    out = np.full(y_bin.shape, 3, dtype=np.int64)
    if visible.sum() == 0:
        return out, (np.nan, np.nan)
    vals = y_bin[visible]
    p40, p60 = np.percentile(vals, [TONGUE_PCTL_LOW, TONGUE_PCTL_HIGH])
    code = np.where(y_bin < p40, 0, np.where(y_bin > p60, 2, 1))
    out[visible] = code[visible]
    return out, (float(p40), float(p60))


def build_outputs(f, tr, keep):
    """Build the (ntrials, 4, N_BINS) int64 decoder output array."""
    go = tr['go_time'][keep]
    n = len(go)
    out = np.empty((n, 4, N_BINS), dtype=np.int64)

    out[:, 0, :] = choice_codes(tr, keep)[:, None]
    out[:, 1, :] = np.array([OUTCOME_CODE[o] for o in tr['outcome'][keep]])[:, None]
    out[:, 2, :] = (tr['early_lick'][keep] == 'early').astype(np.int64)[:, None]

    y_bin, _ = bin_tongue(f, go)
    tongue, pctls = discretize_tongue(y_bin)
    out[:, 3, :] = tongue
    return out, y_bin, pctls


# --------------------------------------------------------------------------------------
# One session
# --------------------------------------------------------------------------------------

def process_session(filepath, show_processing=False, ontology=None):
    """Convert a single NWB session.  Returns None if the session fails the criteria."""
    t0 = time.time()
    if ontology is None:
        ontology = load_ontology()

    with h5py.File(filepath, 'r') as f:
        session_id = f['identifier'][()].decode()
        subject_id = f['general/subject/subject_id'][()].decode()
        mouse = session_id.split('_')[0]

        tr = read_trial_table(f)
        ok, perf, ncl, ncr, reason = session_passes(f, tr)
        if not ok:
            return {'session_id': session_id, 'excluded': True, 'performance': perf,
                    'ncorrect_left': ncl, 'ncorrect_right': ncr, 'reason': reason,
                    'ntrials_total': len(tr['start_time'])}

        keep = trial_mask(f, tr)
        t_read = time.time()

        spikes, region_idx, anno, ccf_xyz = load_good_units(f, ontology)
        t_units = time.time()

        go = tr['go_time'][keep]
        fr = bin_spikes(spikes, go)

        # A 4 s window in which not one of several hundred simultaneously recorded
        # neurons fires is not biologically possible; it means the window is outside the
        # ephys recording (the last trial listed in obs_intervals can extend past the end
        # of the recording).  Drop those trials.
        silent = fr.sum(axis=(1, 2)) == 0
        n_silent = int(silent.sum())
        if n_silent:
            kept_positions = np.flatnonzero(keep)
            keep[kept_positions[silent]] = False
            fr = fr[~silent]
        t_bin = time.time()

        inp = build_inputs(tr, keep)
        out, y_bin, pctls = build_outputs(f, tr, keep)
        t_io = time.time()

        result = {
            'session_id': session_id,
            'excluded': False,
            'subject': mouse,
            'subject_id': subject_id,
            'neural': [np.ascontiguousarray(fr[i]) for i in range(fr.shape[0])],
            'input': [np.ascontiguousarray(inp[i]) for i in range(inp.shape[0])],
            'output': [np.ascontiguousarray(out[i]) for i in range(out.shape[0])],
            'brain_region_idx': region_idx,
            'performance': perf,
            'ncorrect_left': ncl,
            'ncorrect_right': ncr,
            'ntrials_total': int(len(tr['start_time'])),
            'ntrials_kept': int(keep.sum()),
            'nneurons': int(len(spikes)),
            'tongue_percentiles': pctls,
            # diagnostics: number of trials dropped because no neuron fired in them
            'n_silent_trials': n_silent,
            'mean_rate_hz': float(fr.mean()),
            'timing':{'read': t_read - t0, 'units': t_units - t_read,
                       'bin': t_bin - t_units, 'io': t_io - t_bin,
                       'total': t_io - t0},
        }

        if show_processing:
            plot_processing(f, tr, keep, spikes, fr, inp, out, y_bin, pctls,
                            region_idx, session_id)

    return result


# --------------------------------------------------------------------------------------
# Diagnostic plots
# --------------------------------------------------------------------------------------

def plot_processing(f, tr, keep, spikes, fr, inp, out, y_bin, pctls, region_idx,
                    session_id):
    """Plot every processing step for one session so that it can be checked visually."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    go_all = tr['go_time']
    go = go_all[keep]
    ntrials = len(go)
    kept_idx = np.flatnonzero(keep)

    fig = plt.figure(figsize=(22, 26))
    gs = fig.add_gridspec(7, 3, hspace=0.45, wspace=0.25)

    # ---- 1. raw spike raster vs binned rate, for one example trial -------------------
    # pick a trial with a lick and a neuron with a decent rate
    rate_mean = fr.mean(axis=(0, 2))
    ineuron = int(np.argmax(rate_mean))
    # choose a trial with a well-tracked tongue after the go cue
    vis_after = (out[:, 3, N_BINS // 2:] < 3).sum(axis=1)
    itrial = int(np.argmax(vis_after))

    ax = fig.add_subplot(gs[0, :])
    st = spikes[ineuron]
    rel = st[(st >= go[itrial] + OFF_START) & (st < go[itrial] + OFF_END)] - go[itrial]
    ax.eventplot(rel, colors='k', lineoffsets=1.0, linelengths=0.8)
    ax2 = ax.twinx()
    ax2.step(BIN_CENTERS, fr[itrial, ineuron], where='mid', color='tab:red')
    ax2.set_ylabel('binned rate (Hz)', color='tab:red')
    for e in BIN_EDGES[::4]:
        ax.axvline(e, color='0.9', lw=0.5, zorder=0)
    ax.axvline(0, color='b', lw=1.5)
    ax.set_xlim(OFF_START, OFF_END)
    ax.set_title(f'{session_id}: STEP 1 raw spike times (black ticks) vs 50 ms binned '
                 f'firing rate (red), neuron {ineuron}, trial {itrial}; '
                 f'blue line = go cue (t=0)')
    ax.set_xlabel('time from go cue (s)')

    # ---- 2. spike count check for that trial ----------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    counts_direct = np.histogram(rel, bins=BIN_EDGES)[0] / BIN_SIZE
    ax.plot(BIN_CENTERS, counts_direct, 'k-', lw=3, label='np.histogram of raw times')
    ax.plot(BIN_CENTERS, fr[itrial, ineuron], 'r--', lw=1.5, label='converted')
    ax.legend(fontsize=8)
    ax.set_title('STEP 1 check: independent re-binning matches\n'
                 f'max |diff| = {np.abs(counts_direct - fr[itrial, ineuron]).max():.3g}')
    ax.set_xlabel('time from go cue (s)')
    ax.set_ylabel('rate (Hz)')

    # ---- 3. population PSTH aligned to go cue ---------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    psth = fr.mean(axis=(0, 1))
    ax.plot(BIN_CENTERS, psth, 'k')
    ax.axvline(0, color='b')
    ax.axvline(np.median(tr['tone_time'][keep] - go), color='g', ls='--')
    ax.set_title('STEP 2: population PSTH (all neurons, all trials)\n'
                 'blue = go cue, green = median tone onset')
    ax.set_xlabel('time from go cue (s)')
    ax.set_ylabel('mean rate (Hz)')

    # ---- 4. neuron x time heatmap of trial-averaged rates ---------------------------
    ax = fig.add_subplot(gs[1, 2])
    m = fr.mean(axis=0)
    z = (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-9)
    order = np.argsort(np.argmax(z, axis=1))
    im = ax.imshow(z[order], aspect='auto', vmin=-3, vmax=3, cmap='RdBu_r',
                   extent=[OFF_START, OFF_END, len(order), 0])
    ax.axvline(0, color='k')
    ax.set_title('STEP 2: z-scored trial-averaged rate\n(neurons sorted by peak time)')
    ax.set_xlabel('time from go cue (s)')
    ax.set_ylabel('neuron')
    plt.colorbar(im, ax=ax)

    # ---- 5. inputs -------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    for i in range(min(8, ntrials)):
        ax.plot(BIN_CENTERS, inp[i, 0], alpha=0.7)
    ax.axvline(0, color='b')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title('STEP 3 input[0]: time from tone onset (s)\n'
                 '0-crossing should be near t = -1.85 s')
    ax.set_xlabel('time from go cue (s)')

    ax = fig.add_subplot(gs[2, 1])
    stim_trials = np.flatnonzero(inp[:, 1].any(axis=1))
    im = ax.imshow(inp[:, 1], aspect='auto', cmap='Greys',
                   extent=[OFF_START, OFF_END, ntrials, 0], interpolation='nearest')
    ax.axvline(0, color='b')
    ax.set_title('STEP 3 input[1]: photostim on (%d/%d trials)\n'
                 'must always end at or before the go cue' % (len(stim_trials), ntrials))
    ax.set_xlabel('time from go cue (s)')
    ax.set_ylabel('trial')

    # cross-check against the raw photostim event times
    ax = fig.add_subplot(gs[2, 2])
    ev_on = f['acquisition/BehavioralEvents/photostim_start_times']['timestamps'][:]
    ev_off = f['acquisition/BehavioralEvents/photostim_stop_times']['timestamps'][:]
    if len(ev_on):
        # assign each raw photostim event to the trial whose [start, stop] contains it
        j = np.clip(np.searchsorted(tr['start_time'], ev_on, side='right') - 1,
                    0, len(go_all) - 1)
        okj = (ev_on >= tr['start_time'][j]) & (ev_on <= tr['stop_time'][j])
        ax.hist(ev_on[okj] - go_all[j[okj]], bins=40, alpha=0.6, label='raw event onset')
        derived = [BIN_EDGES[np.flatnonzero(inp[i, 1] > 0)[0]] for i in stim_trials]
        if derived:
            ax.hist(derived, bins=40, alpha=0.6, label='converted bin onset')
        ax.legend(fontsize=8)
    ax.set_title('STEP 3 check: photostim onset re. go cue\n(raw events vs converted)')
    ax.set_xlabel('time from go cue (s)')

    # ---- 6. tongue tracking ----------------------------------------------------------
    tt = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
    ts = tt['timestamps'][:]
    ydat = tt['data'][:, 1]
    lik = tt['data'][:, 2]
    p40, p60 = pctls

    ax = fig.add_subplot(gs[3, :])
    lo, hi = np.searchsorted(ts, [go[itrial] + OFF_START, go[itrial] + OFF_END])
    relt = ts[lo:hi] - go[itrial]
    visraw = lik[lo:hi] > TONGUE_LIKELIHOOD_THRESHOLD
    ax.plot(relt[~visraw], ydat[lo:hi][~visraw], '.', ms=2, color='0.8',
            label='raw y (not visible)')
    ax.plot(relt[visraw], ydat[lo:hi][visraw], '.', ms=3, color='tab:blue',
            label='raw y (visible)')
    ax.plot(BIN_CENTERS, y_bin[itrial], 'o-', color='tab:orange', ms=4,
            label='binned mean y (visible frames)')
    ax.axhline(p40, color='g', ls='--', label='40th pctile')
    ax.axhline(p60, color='r', ls='--', label='60th pctile')
    ax.axvline(0, color='b')
    for b in range(N_BINS):
        ax.text(BIN_CENTERS[b], ax.get_ylim()[0], str(out[itrial, 3, b]),
                fontsize=5, ha='center', va='bottom')
    ax.set_xlim(OFF_START, OFF_END)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title(f'STEP 4: tongue y tracking -> discretised class (digits at the bottom), '
                 f'trial {itrial}. Classes: 0 = <p40, 1 = p40-p60, 2 = >p60, 3 = not visible')
    ax.set_xlabel('time from go cue (s)')
    ax.set_ylabel('tongue y (px)')

    ax = fig.add_subplot(gs[4, 0])
    vals = y_bin[~np.isnan(y_bin)]
    ax.hist(vals, bins=60, color='0.6')
    ax.axvline(p40, color='g', ls='--')
    ax.axvline(p60, color='r', ls='--')
    frac = [np.mean(out[:, 3] == k) for k in range(4)]
    ax.set_title('STEP 4: session distribution of visible binned y\n'
                 'class fractions (incl. class 3): %s' %
                 np.array2string(np.array(frac), precision=3))
    ax.set_xlabel('tongue y (px)')

    ax = fig.add_subplot(gs[4, 1])
    im = ax.imshow(out[:, 3], aspect='auto', cmap='viridis', vmin=0, vmax=3,
                   extent=[OFF_START, OFF_END, ntrials, 0], interpolation='nearest')
    ax.axvline(0, color='w')
    ax.set_title('STEP 4: tongue class per trial x bin\n(class 3 = not visible, yellow)')
    ax.set_xlabel('time from go cue (s)')
    ax.set_ylabel('trial')
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(gs[4, 2])
    ax.plot(BIN_CENTERS, (out[:, 3] < 3).mean(axis=0), 'k')
    # lick rate for comparison
    L = f['acquisition/BehavioralEvents/left_lick_times']['timestamps'][:]
    R = f['acquisition/BehavioralEvents/right_lick_times']['timestamps'][:]
    allL = np.sort(np.concatenate([L, R]))
    lick = np.zeros(N_BINS)
    for g in go:
        a, b = np.searchsorted(allL, [g + OFF_START, g + OFF_END])
        if b > a:
            idx = np.clip(((allL[a:b] - g - OFF_START) / BIN_SIZE).astype(int), 0, N_BINS - 1)
            np.add.at(lick, idx, 1)
    ax2 = ax.twinx()
    ax2.plot(BIN_CENTERS, lick / ntrials / BIN_SIZE, color='tab:red')
    ax2.set_ylabel('lick rate (Hz)', color='tab:red')
    ax.axvline(0, color='b')
    ax.set_title('STEP 4 check: P(tongue visible) (black) vs lick rate (red)\n'
                 'both must rise right after the go cue')
    ax.set_xlabel('time from go cue (s)')

    # ---- 7. per-trial outputs --------------------------------------------------------
    ax = fig.add_subplot(gs[5, 0])
    for k, nm in enumerate(OUTPUT_NAMES[:3]):
        vals, cnts = np.unique(out[:, k, 0], return_counts=True)
        ax.bar(np.array(vals) + 4 * k, cnts / ntrials, width=0.8,
               label=nm)
    ax.legend(fontsize=8)
    ax.set_xticks([0, 1, 2, 4, 5, 6, 8, 9])
    ax.set_xticklabels(OUTPUT_VALUES[0] + OUTPUT_VALUES[1] + OUTPUT_VALUES[2],
                       rotation=60, fontsize=7)
    ax.set_title('STEP 5: per-trial output class fractions')

    ax = fig.add_subplot(gs[5, 1])
    # choice cross-check with the first lick after the go cue
    derived, observed = [], []
    for i, g in enumerate(go):
        a, b = np.searchsorted(allL, [g, g + 3.0])
        l = L[(L >= g) & (L < g + 3.0)]
        r = R[(R >= g) & (R < g + 3.0)]
        if len(l) == 0 and len(r) == 0:
            o = CHOICE_NOLICK
        elif len(r) == 0 or (len(l) > 0 and l[0] < r[0]):
            o = CHOICE_LEFT
        else:
            o = CHOICE_RIGHT
        observed.append(o)
        derived.append(out[i, 0, 0])
    derived = np.array(derived); observed = np.array(observed)
    cm = np.zeros((3, 3))
    for a_, b_ in zip(derived, observed):
        cm[a_, b_] += 1
    im = ax.imshow(cm, cmap='Blues')
    for a_ in range(3):
        for b_ in range(3):
            ax.text(b_, a_, int(cm[a_, b_]), ha='center', va='center', fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(OUTPUT_VALUES[0], fontsize=7)
    ax.set_yticks(range(3)); ax.set_yticklabels(OUTPUT_VALUES[0], fontsize=7)
    ax.set_xlabel('observed (first lick after go cue)')
    ax.set_ylabel('converted choice')
    ax.set_title('STEP 5 check: choice vs licks\nagreement = %.4f' %
                 float((derived == observed).mean()))

    ax = fig.add_subplot(gs[5, 2])
    counts = np.bincount(region_idx, minlength=len(BRAIN_REGIONS))
    nz = counts > 0
    ax.bar(np.arange(nz.sum()), counts[nz])
    ax.set_xticks(np.arange(nz.sum()))
    ax.set_xticklabels(np.array(BRAIN_REGIONS)[nz], rotation=60, fontsize=7)
    ax.set_title('STEP 6: good units per brain region (n = %d)' % len(region_idx))

    # ---- 8. final arrays for a few trials -------------------------------------------
    for j, itr in enumerate(np.linspace(0, ntrials - 1, 3).astype(int)):
        ax = fig.add_subplot(gs[6, j])
        ax.imshow(fr[itr], aspect='auto', cmap='magma',
                  extent=[OFF_START, OFF_END, fr.shape[1], 0],
                  vmax=np.percentile(fr[itr], 99.5) + 1e-6, interpolation='nearest')
        ax.axvline(0, color='c')
        ax.set_title('FINAL neural[%d] (%d x %d)\nchoice=%s outcome=%s early=%s' %
                     (itr, fr.shape[1], N_BINS,
                      OUTPUT_VALUES[0][out[itr, 0, 0]],
                      OUTPUT_VALUES[1][out[itr, 1, 0]],
                      OUTPUT_VALUES[2][out[itr, 2, 0]]), fontsize=9)
        ax.set_xlabel('time from go cue (s)')
        if j == 0:
            ax.set_ylabel('neuron')

    fig.suptitle('Processing steps for session %s' % session_id, fontsize=14)
    fname = 'processing_%s.png' % session_id
    fig.savefig(fname, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print('  wrote %s' % fname, flush=True)


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

_WORKER_ONTOLOGY = None


def _worker_init():
    global _WORKER_ONTOLOGY
    _WORKER_ONTOLOGY = load_ontology()


def _worker(args):
    filepath, show = args
    try:
        return process_session(filepath, show_processing=show,
                               ontology=_WORKER_ONTOLOGY)
    except Exception as exc:                      # pragma: no cover - defensive
        import traceback
        traceback.print_exc()
        return {'session_id': os.path.basename(filepath), 'excluded': True,
                'error': str(exc)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile', type=str)
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true', default=True)
    g.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--njobs', type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args()

    t_start = time.time()
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
    print('Found %d NWB files' % len(files), flush=True)

    ontology = load_ontology()

    # --- pass 1: session selection (cheap: only the trials table is read) -------------
    selected = []
    excluded = []
    for fp in files:
        with h5py.File(fp, 'r') as f:
            sid = f['identifier'][()].decode()
            tr = read_trial_table(f)
            ok, perf, ncl, ncr, reason = session_passes(f, tr)
        if ok:
            selected.append(fp)
        else:
            excluded.append((sid, perf, ncl, ncr, reason))
    print('Session selection (performance > %.2f and >= %d correct per direction):'
          % (MIN_PERFORMANCE, MIN_CORRECT_PER_DIRECTION), flush=True)
    print('  selected: %d, excluded: %d' % (len(selected), len(excluded)), flush=True)
    for sid, perf, ncl, ncr, reason in excluded:
        print('    excluded %s: performance=%.3f correct L/R=%d/%d  [%s]'
              % (sid, perf, ncl, ncr, reason), flush=True)
    print('  pass 1 took %.1f s' % (time.time() - t_start), flush=True)

    if args.sample:
        selected = selected[:2]
        print('--sample: restricting to %d sessions' % len(selected), flush=True)

    nshow = 2 if args.show_processing else 0
    jobs = [(fp, i < nshow) for i, fp in enumerate(selected)]

    # --- pass 2: full conversion ------------------------------------------------------
    results = []
    t_convert = time.time()
    if args.njobs > 1 and len(jobs) > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.njobs, initializer=_worker_init) as pool:
            for i, res in enumerate(pool.imap(_worker, jobs)):
                results.append(res)
                el = time.time() - t_convert
                print('[%3d/%3d] %s  ntrials=%d/%d nneurons=%d rate=%.2fHz silent=%d '
                      '(%.1fs; elapsed %.1fs, eta %.1fs)'
                      % (i + 1, len(jobs), res['session_id'], res.get('ntrials_kept', 0),
                         res.get('ntrials_total', 0), res.get('nneurons', 0),
                         res.get('mean_rate_hz', 0), res.get('n_silent_trials', 0),
                         res.get('timing', {}).get('total', 0),
                         el, el / (i + 1) * (len(jobs) - i - 1)), flush=True)
    else:
        for i, job in enumerate(jobs):
            res = process_session(job[0], show_processing=job[1], ontology=ontology)
            results.append(res)
            el = time.time() - t_convert
            print('[%3d/%3d] %s  ntrials=%d/%d nneurons=%d rate=%.2fHz silent=%d '
                  '(%.1fs; elapsed %.1fs)'
                  % (i + 1, len(jobs), res['session_id'], res.get('ntrials_kept', 0),
                     res.get('ntrials_total', 0), res.get('nneurons', 0),
                     res.get('mean_rate_hz', 0), res.get('n_silent_trials', 0),
                     res.get('timing', {}).get('total', 0), el), flush=True)

    results = [r for r in results if not r.get('excluded', False)]
    dropped = [r for r in results
               if len(r['neural']) < MIN_TRIALS_PER_SESSION or r['nneurons'] == 0]
    for r in dropped:
        print('  dropping %s after conversion: %d trials, %d neurons'
              % (r['session_id'], len(r['neural']), r['nneurons']), flush=True)
    results = [r for r in results
               if len(r['neural']) >= MIN_TRIALS_PER_SESSION and r['nneurons'] > 0]
    results.sort(key=lambda r: r['session_id'])
    print('Converted %d sessions in %.1f s' % (len(results), time.time() - t_convert),
          flush=True)

    # --- assemble -----------------------------------------------------------------------
    subjects = sorted({r['subject'] for r in results})
    sub_to_idx = {s: i for i, s in enumerate(subjects)}
    # the mouse name and the DANDI subject id must be in one-to-one correspondence
    pairs = {(r['subject'], r['subject_id']) for r in results}
    assert len(pairs) == len(subjects), 'mouse name / subject id mapping is not 1:1'

    data = {
        'neural': [r['neural'] for r in results],
        'input': [r['input'] for r in results],
        'output': [r['output'] for r in results],
        'subjects': subjects,
        'subject_idx': np.array([sub_to_idx[r['subject']] for r in results], dtype=np.int64),
        'brain_regions': list(BRAIN_REGIONS),
        'brain_region_idx': [r['brain_region_idx'] for r in results],
        'input_names': list(INPUT_NAMES),
        'output_names': list(OUTPUT_NAMES),
        'output_values': [list(v) for v in OUTPUT_VALUES],
        'metadata': {
            'task_description':
                'Mice performed an auditory delayed-response (memory-guided directional '
                'licking) task. A series of three pure tones (3 kHz -> lick right, '
                '12 kHz -> lick left; 0.65 s sample epoch) instructed the upcoming lick '
                'direction; after a 1.2 s delay epoch an auditory go cue (6 kHz, 0.1 s) '
                'released the animal to lick one of two lick ports during a 1.5 s answer '
                'period. Licking the correct port gave a water reward. Licking during the '
                'sample/delay epoch ("early lick") triggered a replay of the epoch. On '
                '~20% of randomly interleaved trials, ALM was bilaterally or unilaterally '
                'photoinhibited (5.5 mW, 0.5 s, always ending at or before the go cue). '
                'Decoder outputs are the lick direction chosen (left / right / no lick), '
                'the trial outcome (ignore / miss / hit), whether the trial contained an '
                'early lick (no / yes), and the discretised side-view tongue y-position '
                '(below 40th pctile / 40-60th pctile / above 60th pctile / not visible). '
                'Decoder inputs are the time since the instruction-tone onset and a binary '
                'photostimulation indicator.',
            'time_bin_size': BIN_SIZE * 1000.0,            # ms
            'temporal_alignment_event': 'go cue onset (auditory go cue, 6 kHz, 0.1 s), '
                                        'NWB acquisition/BehavioralEvents/go_start_times',
            'off_start': OFF_START,
            'off_end': OFF_END,
            'bin_centers': BIN_CENTERS.copy(),             # s, relative to the go cue
            'neural_units': 'firing rate (spikes/s), spike counts in 50 ms '
                            'non-overlapping bins divided by the bin width',
            'input_units': ['s', 'binary (1 = photostimulation on)'],
            'dataset': 'DANDI:000363 Mesoscale Activity Map Dataset '
                       '(Chen, Nguyen, Li & Svoboda 2023), doi:10.48324/dandi.000363',
            'references': [
                'Chen et al., Brain-wide neural activity underlying memory-guided movement',
                'Wang, Kurgyis et al., Brain-wide analysis reveals movement encoding '
                'structured across and within brain areas (Nat Neurosci 2025)',
            ],
            'neuron_curation': "units/classification == 'good' (region-specific "
                               'logistic-regression quality-control classifier of the '
                               'Chen/Liu et al. 2023 spike-sorting white paper)',
            'trial_curation': 'all trials except auto-water and free-water trials',
            'session_curation':
                'behavioral performance on control (no photostim, no early lick, no '
                'auto/free water) trials > 65%% and at least 50 correct lick-left and 50 '
                'correct lick-right trials (data paper criteria); %d of %d sessions kept'
                % (len(results), len(files)),
            'tongue_likelihood_threshold': TONGUE_LIKELIHOOD_THRESHOLD,
            'tongue_percentiles': [TONGUE_PCTL_LOW, TONGUE_PCTL_HIGH],
            'session_info': [
                {'session_id': r['session_id'], 'subject': r['subject'],
                 'subject_id': r['subject_id'], 'performance': r['performance'],
                 'ntrials_total': r['ntrials_total'], 'ntrials_kept': r['ntrials_kept'],
                 'nneurons': r['nneurons'],
                 'tongue_percentile_values': r['tongue_percentiles']}
                for r in results
            ],
            'excluded_sessions': [
                {'session_id': sid, 'performance': perf, 'ncorrect_left': ncl,
                 'ncorrect_right': ncr, 'reason': reason}
                for sid, perf, ncl, ncr, reason in excluded
            ],
        },
    }

    # --- summary ------------------------------------------------------------------------
    ntrials = [len(s) for s in data['neural']]
    nneurons = [r['nneurons'] for r in results]
    print('\n=== conversion summary ===')
    print('sessions: %d' % len(results))
    print('subjects: %d' % len(subjects))
    print('trials: total %d, per session mean %.1f (range %d-%d)'
          % (sum(ntrials), np.mean(ntrials), min(ntrials), max(ntrials)))
    print('neurons: total %d, per session mean %.1f (range %d-%d)'
          % (sum(nneurons), np.mean(nneurons), min(nneurons), max(nneurons)))
    nsilent = sum(r['n_silent_trials'] for r in results)
    print('trials dropped because no neuron fired: %d' % nsilent)
    print('grand mean firing rate: %.3f Hz'
          % float(np.average([r['mean_rate_hz'] for r in results],
                             weights=[len(s) for s in data['neural']])))
    allreg = np.concatenate([r['brain_region_idx'] for r in results])
    for i, name in enumerate(BRAIN_REGIONS):
        c = int((allreg == i).sum())
        if c:
            print('  %-18s %6d' % (name, c))
    for k, name in enumerate(OUTPUT_NAMES):
        vals = np.concatenate([np.concatenate([o[k] for o in r['output']])
                               for r in results])
        frac = np.bincount(vals, minlength=len(OUTPUT_VALUES[k])) / len(vals)
        print('  output %-12s fractions %s' % (name, np.array2string(frac, precision=4)))
    for k, name in enumerate(INPUT_NAMES):
        vals = np.concatenate([np.concatenate([o[k] for o in r['input']])
                               for r in results])
        print('  input  %-12s range [%.3f, %.3f] mean %.4f'
              % (name, vals.min(), vals.max(), vals.mean()))

    # --- save ---------------------------------------------------------------------------
    print('\nwriting %s ...' % args.outfile, flush=True)
    t_save = time.time()
    with open(args.outfile, 'wb') as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print('wrote %s (%.2f GB) in %.1f s'
          % (args.outfile, os.path.getsize(args.outfile) / 1e9, time.time() - t_save))
    print('TOTAL time %.1f s' % (time.time() - t_start))


if __name__ == '__main__':
    main()
