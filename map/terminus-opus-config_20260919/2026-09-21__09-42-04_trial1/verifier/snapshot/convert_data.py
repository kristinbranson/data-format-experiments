"""
Convert the Mesoscale Activity Map (MAP) dataset (DANDI:000363, Chen et al.) from NWB
into the decoder-ready pickle format.

Processing follows the reference pipeline of the MapVideoAnalysis repository
(`VideoAnalysisUtils/preprocessing_DJ_2022Aug.py`, `Sherlock/align_markers.py`):
  * neurons: units labelled `good` by the region-specific QC classifiers
    (NWB column `units/classification`), which additionally have a CCF annotation,
  * trials aligned to the **go cue**,
  * firing rate = spike count / bin width,
  * behavioural markers from the side-view camera (`Camera0_side_*Tracking`),
  * auto-water / free-water trials removed (reference `get_regular_trial_mask`).

Differences from the reference, required by the decoder specification:
  * 50 ms non-overlapping bins spanning [-2.5, +1.5] s around the go cue
    (reference: 40 ms sliding window, 3.4 ms stride, [-3, +3] s),
  * early-lick, no-response (`ignore`) and photostimulation trials are kept, because
    they are the decoder outputs / inputs requested by the task.

Usage:
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
import h5py

# ----------------------------------------------------------------------------- config
DATA_DIR = '/app/data'
BIN_SIZE = 0.05           # s, width of the (non-overlapping) time bins
OFF_START = -2.5          # s, start of the trial window relative to the go cue
OFF_END = 1.5             # s, end of the trial window relative to the go cue
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 80
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])

TONGUE_LIKELIHOOD_THRESH = 0.9   # DeepLabCut likelihood above which the tongue is visible
CCF_ML_MIDLINE = 5700.0          # um, as in the reference code (helper_get_neuron_id_area)
CCF_AP_BREGMA = 5400.0           # um, CCF z of bregma
ALM_AP_MIN = 2000.0              # um anterior of bregma, lower bound of the ALM craniotomy

BRAIN_REGIONS = ['ALM', 'Orbital', 'OtherCortex', 'Striatum', 'Pallidum', 'Thalamus',
                 'Hypothalamus', 'Midbrain', 'Pons', 'Medulla', 'Cerebellum',
                 'Hippocampus', 'Olfactory', 'CorticalSubplate', 'Unknown']
REGION_INDEX = {r: i for i, r in enumerate(BRAIN_REGIONS)}

INPUT_NAMES = ['time_from_tone_onset', 'photostim_on']
OUTPUT_NAMES = ['lick_direction', 'outcome', 'early_lick', 'tongue_y_position']
OUTPUT_VALUES = [['left', 'right', 'no lick'],
                 ['ignore', 'miss', 'hit'],
                 ['no', 'yes'],
                 ['<40th pct', '40-60th pct', '>60th pct', 'not visible']]


# ------------------------------------------------------------------- region assignment
def classify_region(name, ap_from_bregma):
    """Map an Allen CCF structure name onto one of the coarse groups used by the
    reference pipeline (`process_one_sess` loops over exactly these groups).

    Args:
        name: CCF structure name (`units/anno_name`).
        ap_from_bregma: anterior-posterior position of the unit in um relative to bregma
            (positive = anterior). Only used to separate ALM from the rest of motor cortex.
    """
    n = name.lower().strip()
    if n == '':
        return 'Unknown'
    if any(k in n for k in ['cerebell', 'crus', 'lobule', 'declive', 'culmen', 'nodulus',
                            'uvula', 'pyramus', 'copula pyramidis', 'folium', 'ansiform',
                            'flocculus', 'fastigial', 'interposed', 'dentate nucleus',
                            'lingula', 'simple lobule', 'arbor vitae']):
        return 'Cerebellum'
    if any(k in n for k in ['field ca', 'dentate gyrus', 'subiculum', 'ammon', 'hippocamp',
                            'entorhinal', 'fasciola cinerea', 'induseum griseum']):
        return 'Hippocampus'
    if any(k in n for k in ['olfactory', 'piriform area', 'taenia tecta', 'nucleus of the lateral olfactory tract']):
        return 'Olfactory'
    if any(k in n for k in ['endopiriform', 'claustrum', 'basolateral amygdalar',
                            'basomedial amygdalar', 'lateral amygdalar', 'posterior amygdalar',
                            'cortical amygdalar', 'cortical subplate']):
        return 'CorticalSubplate'
    if any(k in n for k in ['caudoputamen', 'striatum', 'nucleus accumbens', 'olfactory tubercle',
                            'fundus of striatum', 'central amygdalar', 'intercalated amygdalar',
                            'medial amygdalar', 'bed nuclei of the stria terminalis',
                            'lateral septal', 'septofimbrial', 'septohippocampal']):
        return 'Striatum'
    if any(k in n for k in ['pallidum', 'globus pallidus', 'substantia innominata',
                            'diagonal band', 'medial septal', 'triangular nucleus of septum',
                            'bed nucleus of the anterior commissure', 'magnocellular nucleus']):
        return 'Pallidum'
    if any(k in n for k in ['hypothalam', 'mammillary', 'preoptic', 'supramammillary',
                            'zona incerta', 'fields of forel', 'subthalamic', 'arcuate',
                            'tuberal nucleus', 'periventricular']):
        return 'Hypothalamus'
    if any(k in n for k in ['thalam', 'geniculate', 'habenula', 'parafascicular', 'paracentral nucleus',
                            'central lateral nucleus', 'central medial nucleus', 'submedial',
                            'posterior complex', 'peripeduncular', 'intergeniculate',
                            'subparafascicular', 'reuniens', 'rhomboid', 'xiphoid',
                            'anteromedial nucleus', 'anteroventral nucleus', 'anterodorsal nucleus',
                            'interanteromedial', 'intermediodorsal', 'lateral posterior nucleus',
                            'lateral dorsal nucleus', 'nucleus of reuniens', 'ethmoid nucleus',
                            'perireunensis', 'posterior limiting nucleus', 'parataenial',
                            'paraventricular nucleus of the thalamus', 'reticular nucleus of the thalamus']):
        return 'Thalamus'
    if any(k in n for k in ['midbrain', 'superior colliculus', 'inferior colliculus', 'substantia nigra',
                            'red nucleus', 'periaqueductal', 'pretectal', 'oculomotor', 'trochlear',
                            'ventral tegmental', 'interpeduncular', 'edinger', 'anterior tegmental',
                            'cuneiform', 'parabigeminal', 'nucleus sagulum', 'nucleus of the brachium',
                            'nucleus of the lateral lemniscus', 'pedunculopontine',
                            'accessory optic tract', 'nucleus of the optic tract',
                            'nucleus of the posterior commissure', 'olivary pretectal',
                            'midbrain reticular', 'nucleus of darkschewitsch', 'interstitial nucleus of cajal']):
        return 'Midbrain'
    if any(k in n for k in ['pons', 'pontine', 'parabrachial', 'locus ceruleus', 'superior olivary',
                            'nucleus raphe pontis', 'tegmental reticular', 'barrington',
                            'laterodorsal tegmental', 'sublaterodorsal', 'dorsal tegmental nucleus',
                            'koelliker', 'motor nucleus of trigeminal', 'supratrigeminal',
                            'principal sensory nucleus of the trigeminal', 'nucleus incertus',
                            'nucleus raphe pallidus' if False else 'peritrigeminal']):
        return 'Pons'
    if any(k in n for k in ['medull', 'gigantocellular', 'vestibular', 'solitary', 'cuneate', 'gracile',
                            'hypoglossal', 'facial motor', 'inferior olivary',
                            'spinal nucleus of the trigeminal', 'raphe magnus', 'raphe obscurus',
                            'raphe pallidus', 'parvicellular reticular', 'intermediate reticular',
                            'lateral reticular', 'paragigantocellular', 'ambiguus',
                            'dorsal motor nucleus of the vagus', 'abducens', 'area postrema',
                            'nucleus x', 'nucleus y', 'nucleus of roller', 'linear nucleus of the medulla',
                            'perihypoglossal', 'nucleus prepositus', 'magnocellular reticular',
                            'parapyramidal', 'efferent vestibular', 'efferent cochlear',
                            'cochlear nuclei', 'dorsal cochlear', 'ventral cochlear',
                            'nucleus of the trapezoid']):
        return 'Medulla'
    # remaining structures are isocortex / cortical plate
    if n.startswith('orbital area'):
        return 'Orbital'
    is_motor = n.startswith('secondary motor area') or n.startswith('primary motor area') or n.startswith('frontal pole')
    if is_motor and ap_from_bregma is not None and ap_from_bregma > ALM_AP_MIN:
        return 'ALM'
    if 'layer' in n or 'area' in n or 'cortex' in n or 'frontal pole' in n:
        return 'OtherCortex'
    return 'Unknown'


# --------------------------------------------------------------------------- utilities
def _decode(x):
    if isinstance(x, bytes):
        return x.decode()
    if isinstance(x, str):
        return x
    return ''


def _tofloat(x, default=np.nan):
    try:
        return float(_decode(x))
    except Exception:
        return default


def bin_spikes(spike_times, spike_index, unit_idx, go_times):
    """Bin spikes of the selected units into firing rates aligned to the go cue.

    Args:
        spike_times: concatenated spike times of all units (session clock, seconds).
        spike_index: `units/spike_times_index`, end index of each unit's spikes.
        unit_idx: indices of the units to keep.
        go_times: (ntrials,) go-cue times in session-clock seconds.

    Returns:
        (n_units, n_trials, N_BINS) float32 array of firing rates in Hz.
    """
    ntrials = len(go_times)
    # absolute bin edges for every trial: (ntrials, N_BINS+1) -> flattened, monotonic
    edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()
    out = np.zeros((len(unit_idx), ntrials, N_BINS), dtype=np.float32)
    starts = np.concatenate([[0], spike_index[:-1]])
    for k, u in enumerate(unit_idx):
        st = spike_times[starts[u]:spike_index[u]]
        if st.size == 0:
            continue
        pos = np.searchsorted(st, edges).reshape(ntrials, N_BINS + 1)
        out[k] = np.diff(pos, axis=1)
    out /= BIN_SIZE       # spike count -> firing rate (Hz), as in sliding_histogram(rate=True)
    return out


def tongue_class_per_bin(ts, y, vis, go_times, thr_lo, thr_hi):
    """Discretised tongue y-position for every trial and time bin.

    For each bin the **last** visible video frame inside the bin is used (the same
    convention as `align_markers_between_lims` in the reference code). Bins without a
    visible frame get class 3 (not visible).
    """
    ntrials = len(go_times)
    cls = np.full((ntrials, N_BINS), 3, dtype=np.int64)
    t0 = go_times + OFF_START
    t1 = go_times + OFF_END
    lo = np.searchsorted(ts, t0, side='left')
    hi = np.searchsorted(ts, t1, side='left')
    for i in range(ntrials):
        a, b = lo[i], hi[i]
        if b <= a:
            continue
        v = vis[a:b]
        if not np.any(v):
            continue
        tt = ts[a:b][v]
        yy = y[a:b][v]
        idx = np.floor((tt - t0[i]) / BIN_SIZE).astype(int)
        keep = (idx >= 0) & (idx < N_BINS)
        idx, yy = idx[keep], yy[keep]
        if idx.size == 0:
            continue
        ybin = np.full(N_BINS, np.nan)
        ybin[idx] = yy               # frames are ordered, so the last one in a bin wins
        good = ~np.isnan(ybin)
        c = np.where(ybin[good] < thr_lo, 0, np.where(ybin[good] > thr_hi, 2, 1))
        cls[i, good] = c
    return cls


# ------------------------------------------------------------------- session conversion
def convert_session(filepath, show_processing=False):
    """Convert a single NWB session. Returns a dict or None if the session is dropped."""
    t_start = time.time()
    timings = {}
    with h5py.File(filepath, 'r') as f:
        subject = _decode(f['general/subject/subject_id'][()])
        identifier = _decode(f['identifier'][()])

        # ---------------- trials -------------------------------------------------
        tr = f['intervals/trials']
        start_time = tr['start_time'][:]
        stop_time = tr['stop_time'][:]
        outcome = np.array([_decode(x) for x in tr['outcome'][:]])
        early = np.array([_decode(x) for x in tr['early_lick'][:]])
        instruction = np.array([_decode(x) for x in tr['trial_instruction'][:]])
        auto_water = tr['auto_water'][:].astype(int)
        free_water = tr['free_water'][:].astype(int)
        ps_onset = np.array([_tofloat(x) for x in tr['photostim_onset'][:]])
        ps_dur = np.array([_tofloat(x) for x in tr['photostim_duration'][:]])
        ps_power = np.array([_tofloat(x, 0.0) for x in tr['photostim_power'][:]])
        ntrials_all = len(start_time)

        be = f['acquisition/BehavioralEvents']
        go_all = be['go_start_times']['timestamps'][:]
        sample_all = be['sample_start_times']['timestamps'][:]
        if len(go_all) != ntrials_all:
            # assign each go event to its trial; trials without a go cue are dropped
            gi = np.searchsorted(start_time, go_all, side='right') - 1
            go = np.full(ntrials_all, np.nan)
            go[gi] = go_all
        else:
            go = go_all

        # tone (sample epoch) onset = last sample-epoch start before the go cue.
        # Early licks trigger a replay of the sample epoch, so the last one is the tone
        # the animal responded to.
        j = np.searchsorted(sample_all, go, side='right') - 1
        tone = np.where(j >= 0, sample_all[np.clip(j, 0, len(sample_all) - 1)], np.nan)
        # fall back to the nominal 1.85 s (0.65 s sample + 1.2 s delay) if missing
        bad_tone = ~np.isfinite(tone) | (tone < start_time) | (tone > go)
        tone = np.where(bad_tone, go - 1.85, tone)

        # ---------------- trials covered by the ephys recording -----------------
        # Spikes are only exported for the trials during which the probes were recording
        # (`units/obs_intervals`, one interval per recorded trial). In 8/173 sessions the
        # recording covers only a contiguous subset of the behavioural trials, so trials
        # outside it must be dropped (they would otherwise appear as zero firing rates).
        u_pre = f['units']
        classification_pre = np.array([_decode(x) for x in u_pre['classification'][:]])
        anno_pre = np.array([_decode(x) for x in u_pre['anno_name'][:]])
        good_pre = (classification_pre == 'good') & (anno_pre != '')
        oii = u_pre['obs_intervals_index'][:]
        oi_starts = np.concatenate([[0], oii[:-1]])
        observed = np.ones(ntrials_all, dtype=bool)
        for ui in np.where(good_pre)[0]:
            iv = u_pre['obs_intervals'][oi_starts[ui]:oii[ui]]
            if len(iv) == ntrials_all:
                continue
            m = np.zeros(ntrials_all, dtype=bool)
            k = np.searchsorted(start_time, iv[:, 0] + 1e-6) - 1
            k = k[(k >= 0) & (k < ntrials_all)]
            m[k] = True
            observed &= m

        # ---------------- trial curation ----------------------------------------
        # Reference `get_regular_trial_mask` additionally removes early-lick, ignore and
        # photostim trials; those are decoder variables here so they are kept.
        keep = (auto_water == 0) & (free_water == 0) & np.isfinite(go) & observed
        trial_idx = np.where(keep)[0]
        if len(trial_idx) < 2:
            return None

        # ---------------- units ---------------------------------------------------
        u = f['units']
        classification = np.array([_decode(x) for x in u['classification'][:]])
        anno = np.array([_decode(x) for x in u['anno_name'][:]])
        good = (classification == 'good') & (anno != '')
        unit_idx = np.where(good)[0]
        if len(unit_idx) == 0:
            return None

        eidx = u['electrodes'][:]
        el = f['general/extracellular_ephys/electrodes']
        ex, ey, ez = el['x'][:], el['y'][:], el['z'][:]
        ux, uy, uz = ex[eidx], ey[eidx], ez[eidx]
        ap = CCF_AP_BREGMA - uz                      # + anterior of bregma (um)
        hemi = np.where(ux >= CCF_ML_MIDLINE, 'left', 'right')
        regions = np.array([classify_region(anno[i], ap[i]) for i in unit_idx])
        region_idx = np.array([REGION_INDEX[r] for r in regions], dtype=np.int64)

        t0 = time.time()
        spike_times = u['spike_times'][:]
        spike_index = u['spike_times_index'][:]
        timings['read_spikes'] = time.time() - t0

        go_keep = go[trial_idx]
        t0 = time.time()
        fr = bin_spikes(spike_times, spike_index, unit_idx, go_keep)   # (nunits, ntr, nbins)
        timings['bin_spikes'] = time.time() - t0

        # Drop trials in which not a single spike was recorded from any of the hundreds of
        # QC-passing neurons: the amplifier was not running for that trial (this happens for
        # the last trial of a recording that was stopped mid-trial).
        nonempty = fr.sum(axis=(0, 2)) > 0
        if not np.all(nonempty):
            fr = fr[:, nonempty, :]
            trial_idx = trial_idx[nonempty]
            go_keep = go_keep[nonempty]
        if len(trial_idx) < 2:
            return None

        # ---------------- inputs ---------------------------------------------------
        # 0: signed time from tone (sample epoch) onset, in seconds, per bin
        time_from_tone = (go_keep - tone[trial_idx])[:, None] + BIN_CENTERS[None, :]
        # 1: photostimulation on/off per bin
        photostim = np.zeros((len(trial_idx), N_BINS), dtype=np.float32)
        has_stim = (ps_power > 0) & np.isfinite(ps_onset) & np.isfinite(ps_dur)
        for k, i in enumerate(trial_idx):
            if not has_stim[i]:
                continue
            s0 = start_time[i] + ps_onset[i] - go[i]     # relative to the go cue
            s1 = s0 + ps_dur[i]
            overlap = (BIN_EDGES[1:] > s0) & (BIN_EDGES[:-1] < s1)
            photostim[k, overlap] = 1.0

        # ---------------- outputs ---------------------------------------------------
        oc = outcome[trial_idx]
        ins = instruction[trial_idx]
        el_tr = early[trial_idx]
        # lick direction: hit -> instructed side, miss -> opposite side, ignore -> no lick
        licked_left = np.where(oc == 'hit', ins == 'left', ins == 'right')
        choice = np.where(oc == 'ignore', 2, np.where(licked_left, 0, 1)).astype(np.int64)
        outcome_code = np.where(oc == 'ignore', 0, np.where(oc == 'miss', 1, 2)).astype(np.int64)
        early_code = np.array([0 if e == 'no early' else 1 for e in el_tr], dtype=np.int64)

        # tongue y position
        t0 = time.time()
        tt_ds = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
        vdata = tt_ds['data'][:]
        vts = tt_ds['timestamps'][:]
        vis = vdata[:, 2] > TONGUE_LIKELIHOOD_THRESH
        yv = vdata[:, 1]
        if np.any(vis):
            thr_lo, thr_hi = np.percentile(yv[vis], [40, 60])
        else:
            thr_lo, thr_hi = np.nan, np.nan
        tongue = tongue_class_per_bin(vts, yv, vis, go_keep, thr_lo, thr_hi)
        timings['tongue'] = time.time() - t0

    ntr = len(trial_idx)
    neural = [np.ascontiguousarray(fr[:, i, :]) for i in range(ntr)]
    inputs = [np.stack([time_from_tone[i].astype(np.float32), photostim[i]]) for i in range(ntr)]
    outputs = [np.stack([np.full(N_BINS, choice[i]), np.full(N_BINS, outcome_code[i]),
                         np.full(N_BINS, early_code[i]), tongue[i]]).astype(np.int64)
               for i in range(ntr)]

    sess = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subject': subject,
        'session_id': identifier,
        'file': os.path.basename(filepath),
        'brain_region_idx': region_idx,
        'hemisphere': hemi[unit_idx],
        'n_units_total': len(classification),
        'n_trials_total': ntrials_all,
        'trial_idx': trial_idx,
        'tongue_thresholds': (float(thr_lo), float(thr_hi)),
        'timings': timings,
        'elapsed': time.time() - t_start,
    }

    if show_processing:
        plot_processing(sess, go_keep, tone[trial_idx], vts, yv, vis, thr_lo, thr_hi)
    return sess


def plot_processing(sess, go_keep, tone_keep, vts, yv, vis, thr_lo, thr_hi):
    """Visualise every processing step for one session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ntr = len(sess['neural'])
    fig, ax = plt.subplots(4, 2, figsize=(16, 14))
    fr_all = np.stack(sess['neural'])        # (ntr, nunits, nbins)

    # 1. population PSTH aligned to the go cue
    ax[0, 0].plot(BIN_CENTERS, fr_all.mean(axis=(0, 1)), 'k')
    ax[0, 0].axvline(0, color='r', label='go cue')
    ax[0, 0].axvline(-1.85, color='b', label='tone onset (median)')
    ax[0, 0].axvline(-1.2, color='g', ls='--', label='delay onset (median)')
    ax[0, 0].set_xlabel('time from go cue (s)'); ax[0, 0].set_ylabel('firing rate (Hz)')
    ax[0, 0].set_title('population mean firing rate'); ax[0, 0].legend(fontsize=7)

    # 2. raster-like image of one example trial
    im = ax[0, 1].imshow(sess['neural'][0], aspect='auto', interpolation='nearest',
                         extent=[OFF_START, OFF_END, 0, sess['neural'][0].shape[0]])
    plt.colorbar(im, ax=ax[0, 1], label='Hz')
    ax[0, 1].axvline(0, color='r')
    ax[0, 1].set_title('trial 0: neurons x time'); ax[0, 1].set_xlabel('time from go cue (s)')

    # 3. input 0 (time from tone onset)
    inp = np.stack(sess['input'])
    for i in range(min(20, ntr)):
        ax[1, 0].plot(BIN_CENTERS, inp[i, 0], alpha=0.5)
    ax[1, 0].axvline(0, color='r')
    ax[1, 0].axhline(0, color='k', ls=':')
    ax[1, 0].set_title('input 0: time from tone onset (s)\n(0 crossing = tone onset, red = go cue)')
    ax[1, 0].set_xlabel('time from go cue (s)')

    # 4. input 1 (photostim)
    ax[1, 1].imshow(inp[:, 1], aspect='auto', interpolation='nearest',
                    extent=[OFF_START, OFF_END, ntr, 0])
    ax[1, 1].axvline(0, color='r')
    ax[1, 1].set_title('input 1: photostim on (trials x time)')
    ax[1, 1].set_xlabel('time from go cue (s)'); ax[1, 1].set_ylabel('trial')

    # 5/6/7. outputs
    out = np.stack(sess['output'])
    for k, name in enumerate(OUTPUT_NAMES):
        r, c = divmod(k + 4, 2)
        imk = ax[r, c].imshow(out[:, k], aspect='auto', interpolation='nearest',
                              extent=[OFF_START, OFF_END, ntr, 0])
        plt.colorbar(imk, ax=ax[r, c])
        ax[r, c].axvline(0, color='r')
        ax[r, c].set_title(f'output {k}: {name} {OUTPUT_VALUES[k]}')
        ax[r, c].set_xlabel('time from go cue (s)'); ax[r, c].set_ylabel('trial')

    fig.suptitle(sess['session_id'])
    fig.tight_layout()
    fig.savefig(f"processing_{sess['session_id']}.png", dpi=110)
    plt.close(fig)

    # second figure: raw tongue trace vs discretisation for a few trials
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for a, i in zip(axes, range(min(3, ntr))):
        t0 = go_keep[i] + OFF_START
        t1 = go_keep[i] + OFF_END
        m = (vts >= t0) & (vts < t1)
        a.plot(vts[m] - go_keep[i], np.where(vis[m], yv[m], np.nan), '.', ms=2, label='tongue y (visible)')
        a.axhline(thr_lo, color='g', ls='--', label='40th pct')
        a.axhline(thr_hi, color='m', ls='--', label='60th pct')
        a2 = a.twinx()
        a2.step(BIN_CENTERS, sess['output'][i][3], where='mid', color='k', alpha=0.6, label='class')
        a2.set_ylim(-0.2, 3.2); a2.set_ylabel('class (0,1,2=vis; 3=not visible)')
        a.axvline(0, color='r')
        a.axvline(-(go_keep[i] - tone_keep[i]), color='b', label='tone onset')
        a.set_ylabel('tongue y (px)')
        a.legend(fontsize=7, loc='upper left')
    axes[-1].set_xlabel('time from go cue (s)')
    fig.suptitle(f"{sess['session_id']}: tongue discretisation")
    fig.tight_layout()
    fig.savefig(f"processing_{sess['session_id']}_tongue.png", dpi=110)
    plt.close(fig)


def _worker(args):
    fp, show = args
    try:
        return convert_session(fp, show_processing=show)
    except Exception as e:  # pragma: no cover
        import traceback
        print(f'ERROR processing {fp}: {e}')
        traceback.print_exc()
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--nproc', type=int, default=12)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
    if args.sample:
        files = files[:2]
    print(f'Converting {len(files)} sessions with {args.nproc} processes', flush=True)

    t0 = time.time()
    show = [args.show_processing and i < 2 for i in range(len(files))]
    if args.nproc > 1 and not args.show_processing:
        with Pool(args.nproc) as pool:
            results = []
            for i, r in enumerate(pool.imap(_worker, list(zip(files, show)))):
                results.append(r)
                if r is not None:
                    print(f"[{i+1}/{len(files)}] {r['session_id']}: "
                          f"{len(r['neural'])} trials, {r['neural'][0].shape[0]} neurons, "
                          f"{r['elapsed']:.1f}s ({time.time()-t0:.0f}s elapsed)", flush=True)
                else:
                    print(f'[{i+1}/{len(files)}] dropped {os.path.basename(files[i])}', flush=True)
    else:
        results = []
        for i, fp in enumerate(files):
            r = _worker((fp, show[i]))
            results.append(r)
            if r is not None:
                print(f"[{i+1}/{len(files)}] {r['session_id']}: {len(r['neural'])} trials, "
                      f"{r['neural'][0].shape[0]} neurons, {r['elapsed']:.1f}s "
                      f"(timings {r['timings']})", flush=True)
            else:
                print(f'[{i+1}/{len(files)}] dropped {os.path.basename(files[i])}', flush=True)

    sessions = [r for r in results if r is not None]
    print(f'Kept {len(sessions)}/{len(files)} sessions in {time.time()-t0:.0f}s', flush=True)

    subjects = sorted({s['subject'] for s in sessions})
    subject_index = {s: i for i, s in enumerate(subjects)}

    data = {
        'neural': [s['neural'] for s in sessions],
        'input': [s['input'] for s in sessions],
        'output': [s['output'] for s in sessions],
        'subjects': subjects,
        'subject_idx': np.array([subject_index[s['subject']] for s in sessions]),
        'brain_regions': BRAIN_REGIONS,
        'brain_region_idx': [s['brain_region_idx'] for s in sessions],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description': (
                'Head-fixed mice performed an auditory delayed-response task (Chen et al., '
                'Mesoscale Activity Map dataset, DANDI:000363). A 3 kHz or 12 kHz tone during the '
                'sample epoch instructed licking left or right; after a 1.2 s delay an auditory go '
                'cue released the response. Decoded outputs are the lick direction (left / right / '
                'no lick), the trial outcome (ignore / miss / hit), whether the animal licked early, '
                'and the discretised y-position of the tongue tracked by DeepLabCut in the side-view '
                'video (terciles 40/60 percentile of the session, plus a class for not visible).'),
            'time_bin_size': BIN_SIZE * 1000.0,
            'temporal_alignment_event': 'go cue onset (auditory go cue ending the delay epoch)',
            'off_start': OFF_START,
            'off_end': OFF_END,
            'neural_units': 'firing rate in spikes/s (spike count per 50 ms bin / 0.05 s)',
            'neuron_curation': ("units/classification == 'good' (region-specific logistic-regression "
                                'quality-control classifiers of Chen, Liu et al. 2023) and a non-empty '
                                'CCF annotation'),
            'trial_curation': ('auto-water and free-water trials removed (reference '
                               'get_regular_trial_mask); early-lick, no-response and photostimulation '
                               'trials retained because they are decoder variables'),
            'session_curation': 'sessions with no quality-controlled unit or fewer than 2 trials removed',
            'input_descriptions': [
                'signed time in seconds from the onset of the instruction tone (sample epoch) to the bin centre',
                'binary: 1 while ALM photoinhibition laser is on during the bin'],
            'output_descriptions': [
                'direction of the response lick (from outcome and instructed trial type)',
                'trial outcome: ignore (no response), miss (error lick), hit (correct lick)',
                'whether the animal licked during the sample/delay epoch',
                'tongue y position in the side-view video, discretised by the 40th/60th percentile '
                'of the session, class 3 = tongue not visible (DeepLabCut likelihood <= 0.9)'],
            'session_info': [{'session_id': s['session_id'], 'subject': s['subject'],
                              'file': s['file'], 'n_trials': len(s['neural']),
                              'n_neurons': int(s['neural'][0].shape[0]),
                              'n_units_total': s['n_units_total'],
                              'n_trials_total': s['n_trials_total'],
                              'tongue_pct40_pct60': s['tongue_thresholds'],
                              'hemisphere': list(s['hemisphere'])}
                             for s in sessions],
            'source': 'DANDI:000363 Mesoscale Activity Map dataset (Chen et al.)',
            'reference_code': 'https://github.com/druckmann-lab/MapVideoAnalysis',
        },
    }

    t0 = time.time()
    with open(args.outfile, 'wb') as fh:
        pickle.dump(data, fh, protocol=4)
    print(f'Wrote {args.outfile} ({os.path.getsize(args.outfile)/1e9:.2f} GB) in {time.time()-t0:.0f}s')

    ntr = sum(len(s['neural']) for s in sessions)
    nneur = sum(s['neural'][0].shape[0] for s in sessions)
    print(f'Sessions: {len(sessions)}, subjects: {len(subjects)}, trials: {ntr}, neurons: {nneur}')


if __name__ == '__main__':
    main()
