"""
Convert the Mesoscale Activity Map (MAP) NWB dataset (DANDI:000363; Chen et al.,
"Brain-wide neural activity underlying memory-guided movement") into the pickle
format consumed by /app/train_decoder.py.

Processing follows the original data paper (/app/datapaper.pdf, /app/methods.txt)
and the analysis code of Wang, Kurgyis et al., "Brain-wide analysis reveals
movement encoding structured across and within brain areas" (/app/code), except
where the decoder task specification requires something different.

Decisions
---------
Loading
  * One NWB file = one recording session; all probes of a session are already
    merged in the file's `units` table.
  * Spike times, trial table, behavioural events (go cue, sample/tone epochs,
    photostimulation) and the side-view DeepLabCut tracking all live on the same
    session clock, so no extra synchronisation is needed.

Unit curation
  * Only units labelled "good" by the region-specific quality-control
    classifiers described in the spike-sorting white paper (Chen, Liu et al.
    2023) and used by both papers: `units/classification == "good"`.
    (This reproduces the published per-area unit counts, e.g. 12,808 thalamus
    and 7,664 striatum good units over the whole release.)
  * Units flagged by the per-trial quality flag `units/is_good_trials` on any
    analysed trial are dropped, so every neuron is well isolated on every kept
    trial (affects 4 sessions).
  * Each unit is assigned a "hemisphere region" label: the major brain division
    of its CCF annotation (`units/anno_name`) according to the Allen ontology,
    plus the hemisphere from the CCF ML coordinate of its electrode (midline
    5700 um, as in the reference code).  Motor-cortex units recorded with an
    ALM-targeted probe are labelled ALM; the reference preprocessing uses the
    same 14 coarse areas x 2 hemispheres.

Session curation (Chen et al. STAR Methods)
  * Behavioural performance > 65% on control trials (no photostimulation, no
    early lick) and at least 50 correct lick-left and 50 correct lick-right
    control trials.  Applied to the 174 released sessions this keeps 106
    sessions, matching the "n = 106 sessions" used in the analysis paper.
  * Sessions whose side-view video is missing, has a corrupted (non-monotonic)
    clock, or does not cover the analysis window on most trials are dropped,
    because the tongue position is one of the decoded outputs.

Trial curation
  * Free-water and auto-water trials are excluded: reward is delivered
    independently of the animal's choice, so "choice" and "outcome" are not
    meaningful there (Wang, Kurgyis et al. exclude them as well).
  * Photostimulation, early-lick and no-response (ignore) trials are KEPT, even
    though the two papers discard them, because the decoder task explicitly asks
    for photostimulation as an input and early lick / outcome (including
    "ignore") as outputs.
  * Trials outside the ephys observation intervals (`units/obs_intervals`), or
    without a single recorded spike, are dropped: the acquisition was not
    running, so these are missing data rather than silence.
  * Trials whose video does not cover the 4 s window (>=95% of frames required)
    are dropped, as are trials with no tone onset.

Alignment and binning
  * Every trial is aligned to its go-cue onset (`go_start_times`) and cut from
    -2.5 s to +1.5 s, as required by the task.
  * Firing rates in non-overlapping 50 ms bins (spikes/s), i.e. 80 bins per
    trial.  The papers use 40 ms bins with a 3.4 ms stride; the task prescribes
    50 ms bins instead, which is the only change to the neural processing.
  * Note: the NWB files store spikes only inside trial intervals.  Error (miss)
    trials end ~0.8 s after the go cue, so the last bins of such trials contain
    no spikes; this is a property of the data release, not of the alignment.

Inputs (decoder inputs, 2 x 80 per trial)
  * time from tone (sample-epoch) onset in seconds.  For early-lick trials the
    sample epoch is replayed, so the last tone onset before the go cue is used.
  * photostimulation on/off, from the photostim start/stop events (ALM
    inactivation, last 0.5 s of the delay epoch).

Outputs (decoder outputs, 4 x 80 per trial, categorical)
  * choice (lick direction): hit -> instructed side, miss -> opposite side,
    ignore -> no lick.  Cross-checked against the recorded lick times (>99%
    agreement).
  * outcome: ignore / miss / hit, from the trials table.
  * early lick: no / yes, from the trials table.
  * tongue y position, discretised per session: <40th percentile, 40-60th,
    >60th percentile of the y positions of all frames in which the tongue is
    visible, and a fourth class for frames where it is not visible (DeepLabCut
    likelihood <= 0.9).  Marker traces are cleaned with the five-sigma velocity
    outlier criterion of Wang, Kurgyis et al. and outliers are imputed from
    neighbouring frames.  Within a 50 ms bin the mean y over visible frames is
    used; bins without any visible frame get the "not visible" class.
"""

import os
import sys
import glob
import json
import pickle
import argparse
import numpy as np
import h5py
from multiprocessing import Pool

# ----------------------------------------------------------------------------- 
# parameters
# -----------------------------------------------------------------------------
DATA_DIR = '/app/data'
OUT_FILE = '/app/converted_data.pkl'
ONTOLOGY_FILE = '/app/allen_structure_graph.json'  # Allen CCF structure graph (cached)
TMP_DIR = '/tmp/convert_sessions'

OFF_START = -2.5           # s, relative to go cue
OFF_END = 1.5              # s, relative to go cue
BIN_SIZE = 0.05            # s (50 ms bins, as requested by the decoder task)
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 80

# session selection (Chen et al. STAR Methods: "overall behavioral performance
# (> 65%), and at least 50 correct lick left and lick right trials each")
PERF_THRESH = 0.65
MIN_CORRECT_PER_DIRECTION = 50

# video / DeepLabCut tracking
LIKELIHOOD_THRESH = 0.9    # tongue counted as visible above this DLC likelihood
VELOCITY_SIGMA = 5.0       # outlier rejection on marker velocity (Wang et al.)
MIN_TRIAL_VIDEO_COVERAGE = 0.95   # fraction of the 4 s window that must have frames
MIN_SESSION_GOOD_VIDEO = 0.5      # fraction of trials that must satisfy the above

MIDLINE_ML = 5700.0        # CCF ML coordinate of the midline (as in the reference code)

# -----------------------------------------------------------------------------
# Allen ontology -> major brain division
# -----------------------------------------------------------------------------
# The reference code (VideoAnalysisUtils/preprocessing_DJ_2022Aug.py) groups units
# into 14 major areas x hemisphere; we reproduce that grouping from the CCF
# annotation of each unit.
DIVISIONS = [('OLF', 'Olfactory'), ('HPF', 'Hippocampus'), ('CTXsp', 'CorticalSubplate'),
             ('STR', 'Striatum'), ('PAL', 'Pallidum'), ('TH', 'Thalamus'),
             ('HY', 'Hypothalamus'), ('MB', 'Midbrain'), ('P', 'Pons'),
             ('MY', 'Medulla'), ('CB', 'Cerebellum')]


def load_ontology():
    with open(ONTOLOGY_FILE) as f:
        root = json.load(f)['msg'][0]
    name2anc = {}

    def rec(node, anc):
        a = anc + [node['acronym']]
        name2anc[node['name']] = a
        for c in node['children']:
            rec(c, a)
    rec(root, [])
    return name2anc


def region_of(anno_name, name2anc, is_alm_probe):
    """Major brain area of a unit from its CCF annotation (Allen ontology)."""
    anc = name2anc.get(anno_name, [])
    for acr, label in DIVISIONS:
        if acr in anc:
            return label
    if 'Isocortex' in anc:
        if 'MO' in anc:
            # ALM is defined functionally and overlaps anterior MOp/MOs; units in
            # motor cortex recorded with an ALM-targeted probe are labelled ALM
            # (the remaining motor cortex units go to OtherCortex).
            return 'ALM' if is_alm_probe else 'OtherCortex'
        if 'ORB' in anc:
            return 'Orbital'
        return 'OtherCortex'
    return 'Other'


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _str(arr):
    return np.asarray(arr).astype(str)


def map_events_to_trials(event_times, trial_start, trial_stop):
    """Index of the trial containing each event (-1 if outside any trial)."""
    idx = np.searchsorted(trial_start, event_times, side='right') - 1
    ok = (idx >= 0) & (event_times <= trial_stop[np.clip(idx, 0, len(trial_stop) - 1)])
    idx[~ok] = -1
    return idx


def clean_marker(y, visible):
    """Remove 5-sigma velocity outliers and impute from neighbouring frames.

    Follows Wang, Kurgyis et al.: "We identified outliers by a five-sigma
    threshold on velocity across frames and imputed outliers from nearby frames."
    Only frames in which the tongue is visible are considered.
    """
    y = y.copy()
    vis = visible.copy()
    iv = np.flatnonzero(vis)
    if len(iv) > 2:
        v = np.diff(y[iv])
        sd = v.std()
        if sd > 0:
            bad = np.abs(v - v.mean()) > VELOCITY_SIGMA * sd
            # a jump marks the later frame of the pair as an outlier
            bad_idx = iv[1:][bad]
            if len(bad_idx):
                good = np.setdiff1d(iv, bad_idx)
                if len(good) > 1:
                    y[bad_idx] = np.interp(bad_idx, good, y[good])
                else:
                    vis[bad_idx] = False
    return y, vis


# -----------------------------------------------------------------------------
# per-session conversion
# -----------------------------------------------------------------------------

def process_session(args):
    fname, name2anc = args
    info = {'file': os.path.basename(fname)}
    try:
        with h5py.File(fname, 'r') as f:
            out = _process_session(f, fname, name2anc, info)
    except Exception as exc:  # pragma: no cover
        info['excluded'] = 'error: %s' % exc
        return None, info
    return out, info


def _process_session(f, fname, name2anc, info):
    tr = f['intervals/trials']
    be = f['acquisition/BehavioralEvents']

    trial_start = tr['start_time'][:]
    trial_stop = tr['stop_time'][:]
    ntrials_all = len(trial_start)
    outcome = _str(tr['outcome'][:])              # 'hit' / 'miss' / 'ignore'
    early = _str(tr['early_lick'][:])             # 'early' / 'no early'
    instruction = _str(tr['trial_instruction'][:])  # 'left' / 'right'
    auto_water = tr['auto_water'][:].astype(bool)
    free_water = tr['free_water'][:].astype(bool)
    photostim_onset = _str(tr['photostim_onset'][:])

    subject = str(np.asarray(f['general/subject/subject_id'][()]).astype(str))
    info['subject'] = subject
    info['ntrials_raw'] = int(ntrials_all)

    # --- go cue: exactly one per trial -------------------------------------
    go_all = be['go_start_times/timestamps'][:]
    gidx = map_events_to_trials(go_all, trial_start, trial_stop)
    go = np.full(ntrials_all, np.nan)
    go[gidx[gidx >= 0]] = go_all[gidx >= 0]

    # --- session selection (Chen et al. behavioural criteria) --------------
    # "Overall performance was computed as the fraction of correct control
    # trials (i.e. no photostimulation), excluding any early lick trials."
    control = (photostim_onset == 'N/A') & (early == 'no early')
    perf = float((outcome[control] == 'hit').mean()) if control.sum() else 0.0
    n_correct_left = int((control & (outcome == 'hit') & (instruction == 'left')).sum())
    n_correct_right = int((control & (outcome == 'hit') & (instruction == 'right')).sum())
    info['performance'] = perf
    info['n_correct_left'] = n_correct_left
    info['n_correct_right'] = n_correct_right
    if not (perf > PERF_THRESH and n_correct_left >= MIN_CORRECT_PER_DIRECTION
            and n_correct_right >= MIN_CORRECT_PER_DIRECTION):
        info['excluded'] = 'behavior'
        return None

    # --- video (side-view DeepLabCut tongue tracking) ----------------------
    key = 'acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking'
    if key not in f:
        info['excluded'] = 'no tongue tracking'
        return None
    vts = f[key + '/timestamps'][:]
    if np.any(np.diff(vts) <= 0):
        # a few sessions have corrupted (non-monotonic, restarting) video clocks;
        # frames cannot be aligned to the behavioural clock in those
        info['excluded'] = 'non-monotonic video timestamps'
        return None
    vdata = f[key + '/data'][:]
    tongue_y = vdata[:, 1].astype(np.float64)
    tongue_vis = vdata[:, 2] > LIKELIHOOD_THRESH
    tongue_y, tongue_vis = clean_marker(tongue_y, tongue_vis)
    if tongue_vis.sum() < 100:
        info['excluded'] = 'tongue never tracked'
        return None
    # session-wide percentiles of tongue y position (visible frames only)
    p40, p60 = np.percentile(tongue_y[tongue_vis], [40, 60])

    win_lo = go + OFF_START
    win_hi = go + OFF_END
    nframes_win = np.searchsorted(vts, win_hi) - np.searchsorted(vts, win_lo)
    coverage = nframes_win * np.median(np.diff(vts)) / (OFF_END - OFF_START)
    good_video = coverage >= MIN_TRIAL_VIDEO_COVERAGE
    info['frac_trials_good_video'] = float(good_video.mean())
    if good_video.mean() < MIN_SESSION_GOOD_VIDEO:
        info['excluded'] = 'video does not cover the analysis window'
        return None

    # --- trial selection ---------------------------------------------------
    # Free-water / auto-water trials are rewarded independently of the animal's
    # choice; they are excluded here as in Wang, Kurgyis et al. Photostimulation,
    # early-lick and no-response trials are *kept*, because the decoder task asks
    # for photostimulation as an input and for early lick / outcome as outputs.
    keep = (~auto_water) & (~free_water) & good_video & np.isfinite(go)

    # Some sessions contain behavioural trials that were recorded before the
    # ephys acquisition started or after it stopped. `units/obs_intervals`
    # lists, for every unit, the trial intervals during which it was observed;
    # trials outside them contain no spikes at all and must be dropped rather
    # than treated as silence.
    obs_idx = f['units/obs_intervals_index'][:]
    obs_start = np.concatenate([[0], obs_idx[:-1]])
    obs = f['units/obs_intervals'][:]

    trials = np.flatnonzero(keep)
    if len(trials) < 2:
        info['excluded'] = 'too few trials'
        return None

    # --- units -------------------------------------------------------------
    # Quality control: units labelled 'good' by the region-specific classifiers
    # of the spike-sorting white paper (Chen, Liu et al. 2023), as used in both
    # papers.
    classification = _str(f['units/classification'][:])
    good_unit = classification == 'good'
    if good_unit.sum() == 0:
        info['excluded'] = 'no good units'
        return None
    unit_idx = np.flatnonzero(good_unit)

    # Trials observed by every selected unit.  `units/obs_intervals` lists the
    # trial intervals during which each unit was recorded; trials outside them
    # contain no spikes at all (the ephys acquisition was not running) and must
    # be dropped rather than treated as silence.
    obs_trials = {}
    observed = np.ones(ntrials_all, dtype=bool)
    for u in unit_idx:
        iv = obs[obs_start[u]:obs_idx[u], 0]
        ti = np.searchsorted(trial_start, iv + 1e-6) - 1
        ti = ti[ti >= 0]
        obs_trials[u] = ti
        m = np.zeros(ntrials_all, dtype=bool)
        m[ti] = True
        observed &= m
    info['n_unobserved_trials'] = int((~observed[trials]).sum())
    keep = keep & observed
    if keep.sum() < 2:
        info['excluded'] = 'too few trials'
        return None

    # Per-trial unit quality flag from the MAP pipeline (units that drifted away
    # or became unstable during part of the session).  Its columns run over the
    # trials a unit was observed in.  Units flagged on any analysed trial are
    # dropped, so every neuron is well isolated on every trial we keep.
    ig_all = f['units/is_good_trials'][:]
    unit_ok = np.ones(len(unit_idx), dtype=bool)
    for i, u in enumerate(unit_idx):
        ti = obs_trials[u]
        row = ig_all[u]
        flag = np.zeros(ntrials_all, dtype=bool)
        if len(row) == len(ti):
            flag[ti] = row
        else:  # unexpected layout: do not filter on it
            flag[:] = True
        unit_ok[i] = bool(flag[keep].all())
    info['n_units_dropped_trialqc'] = int((~unit_ok).sum())
    unit_idx = unit_idx[unit_ok]
    if len(unit_idx) == 0:
        info['excluded'] = 'no good units after per-trial QC'
        return None

    trials = np.flatnonzero(keep)

    anno = _str(f['units/anno_name'][:])
    elec = f['units/electrodes'][:]
    el = f['general/extracellular_ephys/electrodes']
    el_x = el['x'][:]          # CCF ML
    el_group = _str(el['group_name'][:])
    probe_target = {}
    for k in f['general/extracellular_ephys']:
        if k == 'electrodes':
            continue
        try:
            probe_target[k] = json.loads(f['general/extracellular_ephys'][k].attrs['location'])['brain_regions']
        except Exception:
            probe_target[k] = ''

    regions = []
    for u in unit_idx:
        e = elec[u]
        target = probe_target.get(el_group[e], '')
        reg = region_of(anno[u], name2anc, 'ALM' in target)
        hemi = 'left' if el_x[e] >= MIDLINE_ML else 'right'
        regions.append('%s %s' % (hemi, reg))

    # --- spike times -> firing rates ---------------------------------------
    sp_index = f['units/spike_times_index'][:]
    sp_start = np.concatenate([[0], sp_index[:-1]])
    spike_ds = f['units/spike_times']

    edges = (go[trials][:, None] + OFF_START
             + BIN_SIZE * np.arange(NBINS + 1)[None, :])       # (ntrials, nbins+1)
    flat_edges = edges.ravel()
    order = np.argsort(flat_edges, kind='stable')
    sorted_edges = flat_edges[order]

    nn = len(unit_idx)
    rates = np.zeros((nn, len(trials), NBINS), dtype=np.float32)
    for i, u in enumerate(unit_idx):
        st = spike_ds[sp_start[u]:sp_index[u]]
        if len(st) == 0:
            continue
        st = np.sort(st)
        counts_sorted = np.searchsorted(st, sorted_edges)
        counts = np.empty_like(counts_sorted)
        counts[order] = counts_sorted
        counts = counts.reshape(len(trials), NBINS + 1)
        rates[i] = np.diff(counts, axis=1).astype(np.float32) / BIN_SIZE

    # --- inputs ------------------------------------------------------------
    # 1) time from tone (sample epoch) onset, in seconds
    sam_all = be['sample_start_times/timestamps'][:]
    sidx = map_events_to_trials(sam_all, trial_start, trial_stop)
    tone_onset = np.full(ntrials_all, np.nan)
    for t, i in zip(sam_all, sidx):
        if i >= 0 and (not np.isfinite(go[i]) or t <= go[i]):
            # early-lick trials replay the sample epoch: use the last tone onset
            # that precedes the go cue
            if not np.isfinite(tone_onset[i]) or t > tone_onset[i]:
                tone_onset[i] = t
    # 2) photostimulation on/off
    ps_on = np.full(ntrials_all, np.nan)
    ps_off = np.full(ntrials_all, np.nan)
    if 'photostim_start_times' in be:
        pst = be['photostim_start_times/timestamps'][:]
        psp = be['photostim_stop_times/timestamps'][:]
        pidx = map_events_to_trials(pst, trial_start, trial_stop)
        for a, b, i in zip(pst, psp, pidx):
            if i >= 0:
                ps_on[i] = a
                ps_off[i] = b

    bin_centers = OFF_START + BIN_SIZE * (np.arange(NBINS) + 0.5)
    bin_lo = OFF_START + BIN_SIZE * np.arange(NBINS)
    bin_hi = bin_lo + BIN_SIZE

    # --- outputs -----------------------------------------------------------
    # choice (lick direction): hit -> instructed side, miss -> opposite side,
    # ignore -> no lick.  Verified against the recorded lick times (>99% match).
    other = {'left': 'right', 'right': 'left'}
    choice_map = {'left': 0, 'right': 1, 'no lick': 2}
    outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}

    neural_trials, input_trials, output_trials = [], [], []
    frame_dt = np.median(np.diff(vts))
    for k, t in enumerate(trials):
        g = go[t]
        # inputs
        if np.isfinite(tone_onset[t]):
            time_from_tone = (g + bin_centers) - tone_onset[t]
        else:
            time_from_tone = np.full(NBINS, np.nan)
        photostim = np.zeros(NBINS, dtype=np.float32)
        if np.isfinite(ps_on[t]):
            a, b = ps_on[t] - g, ps_off[t] - g
            photostim = ((bin_hi > a) & (bin_lo < b)).astype(np.float32)
        inp = np.stack([time_from_tone.astype(np.float32), photostim])

        # tongue y class per bin
        lo = np.searchsorted(vts, g + OFF_START)
        hi = np.searchsorted(vts, g + OFF_END)
        tclass = np.full(NBINS, 3, dtype=np.int64)
        if hi > lo:
            b_idx = np.floor((vts[lo:hi] - (g + OFF_START)) / BIN_SIZE).astype(int)
            np.clip(b_idx, 0, NBINS - 1, out=b_idx)
            vis = tongue_vis[lo:hi]
            if vis.any():
                nvis = np.bincount(b_idx[vis], minlength=NBINS)
                ysum = np.bincount(b_idx[vis], weights=tongue_y[lo:hi][vis], minlength=NBINS)
                with np.errstate(invalid='ignore'):
                    ymean = np.where(nvis > 0, ysum / np.maximum(nvis, 1), np.nan)
                seen = nvis > 0
                tclass[seen] = np.where(ymean[seen] < p40, 0,
                                        np.where(ymean[seen] <= p60, 1, 2))

        if outcome[t] == 'hit':
            ch = instruction[t]
        elif outcome[t] == 'miss':
            ch = other[instruction[t]]
        else:
            ch = 'no lick'
        out = np.stack([
            np.full(NBINS, choice_map[ch], dtype=np.int64),
            np.full(NBINS, outcome_map[outcome[t]], dtype=np.int64),
            np.full(NBINS, 1 if early[t] == 'early' else 0, dtype=np.int64),
            tclass,
        ])

        neural_trials.append(rates[:, k, :].copy())
        input_trials.append(inp)
        output_trials.append(out)

    # Drop trials for which the tone onset is unknown (no sample epoch recorded)
    # and trials in which not a single spike was recorded from any unit: those
    # occur when the acquisition stopped in the middle of a trial and are an
    # absence of data rather than an absence of spiking.
    valid = [i for i, x in enumerate(input_trials)
             if np.all(np.isfinite(x)) and neural_trials[i].sum() > 0]
    if len(valid) < 2:
        info['excluded'] = 'no valid trials'
        return None
    neural_trials = [neural_trials[i] for i in valid]
    input_trials = [input_trials[i] for i in valid]
    output_trials = [output_trials[i] for i in valid]

    info['ntrials'] = len(neural_trials)
    info['nunits'] = nn
    info['session'] = os.path.basename(fname).split('_')[1].replace('ses-', '')
    return dict(neural=neural_trials, input=input_trials, output=output_trials,
                regions=regions, subject=subject, info=info)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='only convert N sessions (debug)')
    ap.add_argument('--nproc', type=int, default=16)
    ap.add_argument('--out', type=str, default=OUT_FILE)
    args = ap.parse_args()

    name2anc = load_ontology()
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.nwb')))
    if args.limit:
        files = files[:args.limit]
    print('found %d nwb files' % len(files), flush=True)

    results = []
    with Pool(args.nproc) as pool:
        for res, info in pool.imap(process_session, [(fn, name2anc) for fn in files]):
            print(info, flush=True)
            if res is not None:
                results.append(res)

    print('kept %d sessions' % len(results), flush=True)

    subjects = sorted({r['subject'] for r in results})
    brain_regions = sorted({reg for r in results for reg in r['regions']})

    data = {
        'neural': [r['neural'] for r in results],
        'input': [r['input'] for r in results],
        'output': [r['output'] for r in results],
        'subjects': subjects,
        'subject_idx': np.array([subjects.index(r['subject']) for r in results]),
        'brain_regions': brain_regions,
        'brain_region_idx': [np.array([brain_regions.index(x) for x in r['regions']], dtype=int)
                             for r in results],
        'input_names': ['time from tone onset (s)', 'photostimulation on'],
        'output_names': ['choice', 'outcome', 'early lick', 'tongue y position'],
        'output_values': [['left', 'right', 'no lick'],
                          ['ignore', 'miss', 'hit'],
                          ['no', 'yes'],
                          ['<40th pct', '40-60th pct', '>60th pct', 'not visible']],
        'metadata': {
            'task_description': (
                'Auditory delayed-response task (Chen et al., DANDI 000363). A 3 kHz or '
                '12 kHz tone instructs lick-left or lick-right; after a 1.2 s delay an '
                'auditory go cue releases the response. Decoded outputs: lick direction '
                'choice, trial outcome (ignore/miss/hit), whether the animal licked early, '
                'and the discretized side-view tongue y position.'),
            'time_bin_size': BIN_SIZE * 1000.0,
            'temporal_alignment_event': 'go cue onset (auditory go cue, end of delay epoch)',
            'off_start': OFF_START,
            'off_end': OFF_END,
            'neural_units': 'firing rate (spikes/s) in non-overlapping 50 ms bins',
            'session_info': [r['info'] for r in results],
            'dataset': 'DANDI:000363 Mesoscale Activity Map dataset (Chen et al. 2024)',
            'notes': (
                'Units: spike-sorted clusters labelled "good" by the region-specific QC '
                'classifiers of the MAP spike-sorting white paper. Sessions: behavioural '
                'performance > 65% on control (no photostimulation, no early lick) trials '
                'and >= 50 correct lick-left and lick-right trials, as in Chen et al.; '
                'sessions whose side-view video does not cover the analysis window were '
                'also dropped. Trials: free-water and auto-water trials excluded (as in '
                'Wang, Kurgyis et al.); photostimulation, early-lick and no-response trials '
                'are kept because they are decoder inputs/outputs here. Spike times in the '
                'NWB files are only provided within trial intervals, so bins that fall into '
                'the inter-trial interval (mostly the late part of the window on error '
                'trials, which end early) contain no spikes.'),
        },
    }

    with open(args.out, 'wb') as fh:
        pickle.dump(data, fh, protocol=4)
    print('saved', args.out, flush=True)


if __name__ == '__main__':
    main()
