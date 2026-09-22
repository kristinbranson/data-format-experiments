"""
Convert the MAP (Mesoscale Activity Map, DANDI:000363) NWB dataset into the
decoder-ready dictionary format.

Usage:
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Processing follows the reference pipeline of Wang, Kurgyis et al. 2025
(/app/code: VideoAnalysisUtils/preprocessing_DJ_2022Aug.py, Sherlock/align_markers.py,
VideoAnalysisUtils/population_decoding_utils.py) adapted to the NWB release and to the
decoder specification (50 ms bins, -2.5..+1.5 s around the go cue).

All data are read with the pynwb API.
"""

import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np
from pynwb import NWBHDF5IO

# ----------------------------------------------------------------------------- config
BIN_SIZE = 0.05            # s, decoder specification
T_START = -2.5             # s relative to go cue, decoder specification
T_STOP = 1.5               # s relative to go cue
NBINS = int(round((T_STOP - T_START) / BIN_SIZE))   # 80
BIN_EDGES_REL = T_START + BIN_SIZE * np.arange(NBINS + 1)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2

LIKELIHOOD_THRESH = 0.9    # DeepLabCut likelihood for "tongue visible"
ML_MIDLINE = 5700.0        # um; hemisphere split used by the reference code

INPUT_NAMES = ['time_from_tone_onset', 'photostim']
OUTPUT_NAMES = ['choice', 'outcome', 'early_lick', 'tongue_y_position']
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['<40th pct', '40-60th pct', '>60th pct', 'not visible'],
]

# 14 coarse region groups used by the reference preprocessing code
COARSE_REGIONS = ['ALM', 'Medulla', 'Midbrain', 'Striatum', 'Thalamus', 'Pons', 'Cerebellum',
                  'Hypothalamus', 'Hippocampus', 'Orbital', 'OtherCortex', 'Olfactory',
                  'CorticalSubplate', 'Pallidum']
BRAIN_REGIONS = ['%s %s' % (side, reg) for reg in COARSE_REGIONS for side in ('left', 'right')]
BRAIN_REGION_INDEX = {name: i for i, name in enumerate(BRAIN_REGIONS)}


# ------------------------------------------------------------------- CCF region mapping
def coarse_region(anno):
    """Map a fine CCF annotation (units.anno_name) onto one of the 14 coarse groups that
    the reference code (preprocessing_DJ_2022Aug.process_one_sess) uses."""
    a = str(anno).strip()
    al = a.lower()
    if a == '':
        return None
    if any(k in al for k in ['lobule', 'lobules', 'copula pyramidis', 'nodulus', 'uvula', 'declive',
                             'folium', 'pyramus', 'flocculus', 'paramedian lobule', 'crus 1', 'crus 2',
                             'ansiform', 'cerebell', 'fastigial', 'interposed nucleus', 'dentate nucleus',
                             'central lobule', 'culmen', 'vermal', 'hemispheric', 'lingula',
                             'infracerebellar']):
        return 'Cerebellum'
    if any(k in al for k in ['medulla', 'medullary reticular', 'gigantocellular', 'paragigantocellular',
                             'magnocellular reticular', 'intermediate reticular', 'parvicellular reticular',
                             'spinal nucleus of the trigeminal', 'spinal vestibular', 'medial vestibular',
                             'lateral vestibular', 'inferior olivary', 'nucleus of roller', 'nucleus x',
                             'nucleus y', 'nucleus prepositus', 'raphe obscurus', 'raphe pallidus',
                             'raphe magnus', 'hypoglossal', 'facial motor nucleus', 'nucleus ambiguus',
                             'cuneate nucleus', 'gracile nucleus', 'solitary', 'dorsal motor nucleus',
                             'linear nucleus of the medulla', 'parasolitary', 'perihypoglossal',
                             'intercalated nucleus', 'lateral reticular nucleus', 'nucleus of the trapezoid',
                             'paratrigeminal', 'parapyramidal', 'area postrema', 'dorsal cochlear',
                             'ventral cochlear', 'efferent cochlear', 'vestibulocochlear']):
        return 'Medulla'
    if any(k in al for k in ['pons', 'pontine', 'koelliker-fuse', 'parabrachial', 'superior vestibular',
                             'locus ceruleus', 'lateral lemniscus', 'superior olivary', 'tegmental reticular',
                             'barrington', 'laterodorsal tegmental', 'sublaterodorsal', 'supratrigeminal',
                             'principal sensory nucleus of the trigeminal', 'motor nucleus of trigeminal',
                             'dorsal tegmental nucleus', 'nucleus incertus', 'pontine central gray',
                             'nucleus raphe pontis', 'peritrigeminal']):
        return 'Pons'
    if any(k in al for k in ['midbrain', 'superior colliculus', 'inferior colliculus', 'substantia nigra',
                             'red nucleus', 'pretectal', 'periaqueductal gray', 'pedunculopontine',
                             'accessory optic tract', 'nucleus of the optic tract', 'cuneiform',
                             'ventral tegmental area', 'interpeduncular', 'oculomotor', 'trochlear',
                             'edinger-westphal', 'nucleus of darkschewitsch', 'interstitial nucleus of cajal',
                             'midbrain trigeminal', 'dorsal raphe', 'nucleus sagulum', 'parabigeminal',
                             'retrorubral', 'anterior tegmental nucleus', 'nucleus of the brachium',
                             'subcommissural', 'periventricular gray']):
        return 'Midbrain'
    if any(k in al for k in ['field ca1', 'field ca2', 'field ca3', 'dentate gyrus', 'subiculum',
                             'postsubiculum', 'presubiculum', 'parasubiculum', 'prosubiculum',
                             'hippocamp', "ammon's horn", 'induseum griseum', 'fasciola cinerea']):
        return 'Hippocampus'
    if ('thalam' in al) or any(k in al for k in ['habenula', 'geniculate', 'parafascicular', 'paracentral nucleus',
                             'anteroventral nucleus', 'anterodorsal nucleus', 'anteromedial nucleus',
                             'rhomboid nucleus', 'submedial nucleus', 'suprageniculate', 'subparafascicular',
                             'nucleus of reunions', 'nucleus reuniens', 'reuniens', 'peripeduncular',
                             'intergeniculate', 'ethmoid nucleus', 'xiphoid', 'parataenial',
                             'interanteromedial', 'intermediodorsal', 'nucleus of the field',
                             'posterior triangular', 'perireunensis']):
        return 'Thalamus'
    if any(k in al for k in ['hypothalam', 'zona incerta', 'fields of forel', 'mammillary', 'supramammillary',
                             'tuberal', 'preoptic', 'subthalamic', 'arcuate hypothalamic',
                             'periventricular hypothalamic', 'median eminence', 'posterior hypothalamic']):
        return 'Hypothalamus'
    if any(k in al for k in ['globus pallidus', 'pallidum', 'substantia innominata',
                             'bed nuclei of the stria terminalis', 'bed nucleus of the anterior commissure',
                             'magnocellular nucleus', 'medial septal nucleus', 'diagonal band',
                             'triangular nucleus of septum']):
        return 'Pallidum'
    if any(k in al for k in ['caudoputamen', 'striatum', 'nucleus accumbens', 'fundus of striatum',
                             'central amygdalar', 'medial amygdalar', 'intercalated amygdalar',
                             'lateral septal', 'septofimbrial', 'septohippocampal',
                             'anterior amygdalar area', 'olfactory tubercle']):
        return 'Striatum'
    if any(k in al for k in ['claustrum', 'endopiriform', 'cortical subplate', 'basolateral amygdalar',
                             'basomedial amygdalar', 'lateral amygdalar', 'posterior amygdalar']):
        return 'CorticalSubplate'
    if any(k in al for k in ['olfactory', 'piriform', 'taenia tecta', 'lateral olfactory tract',
                             'cortical amygdalar area', 'postpiriform']):
        return 'Olfactory'
    if al.startswith('orbital area'):
        return 'Orbital'
    if al.startswith('secondary motor area') or al.startswith('primary motor area'):
        # probes were targeted at ALM (anterior lateral motor cortex); the reference groups
        # recorded motor-cortex units as ALM
        return 'ALM'
    if any(k in al for k in ['area', 'cortex', 'layer', 'field', 'frontal pole', 'ectorhinal', 'perirhinal',
                             'entorhinal', 'gustatory', 'visceral', 'insular', 'retrosplenial', 'cingulate',
                             'prelimbic', 'infralimbic', 'dorsal peduncular']):
        return 'OtherCortex'
    return None


# ------------------------------------------------------------------------- session code
def process_session(path, make_plots=False):
    """Convert one NWB session. Returns a dict with per-session converted data, or None."""
    t0 = time.time()
    timing = {}
    with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
        nwb = io.read()
        session_id = nwb.identifier
        subject_id = str(nwb.subject.subject_id)
        mouse_name = str(nwb.subject.description)

        # ---------------- trials ----------------
        trials = nwb.intervals['trials']
        df = trials.to_dataframe()
        n_trials_all = len(df)
        start_time = df['start_time'].values.astype(float)
        stop_time = df['stop_time'].values.astype(float)
        outcome_str = np.asarray(df['outcome'].values, dtype=object)
        early_str = np.asarray(df['early_lick'].values, dtype=object)
        instruction = np.asarray(df['trial_instruction'].values, dtype=object)
        auto_water = np.asarray(df['auto_water'].values).astype(int)
        free_water = np.asarray(df['free_water'].values).astype(int)

        be = nwb.acquisition['BehavioralEvents'].time_series
        go_times = np.asarray(be['go_start_times'].timestamps[:], dtype=float)
        sample_times = np.asarray(be['sample_start_times'].timestamps[:], dtype=float)
        left_licks = np.asarray(be['left_lick_times'].timestamps[:], dtype=float)
        right_licks = np.asarray(be['right_lick_times'].timestamps[:], dtype=float)
        if 'photostim_start_times' in be:
            stim_on = np.asarray(be['photostim_start_times'].timestamps[:], dtype=float)
            stim_off = np.asarray(be['photostim_stop_times'].timestamps[:], dtype=float)
        else:
            stim_on = np.zeros(0)
            stim_off = np.zeros(0)

        if len(go_times) != n_trials_all:
            # defensive: keep only trials that contain exactly one go cue
            raise RuntimeError('%s: %d go cues for %d trials' % (session_id, len(go_times), n_trials_all))

        # ---- trial curation (reference get_regular_trial_mask, minus the exclusions that
        #      the decoder task requires us to keep: early lick, no-response, photostim) ----
        keep = (auto_water == 0) & (free_water == 0)
        # the go cue must lie inside the trial (sanity)
        keep &= (go_times >= start_time) & (go_times <= stop_time)

        # (bin edges are computed after the unit-based trial mask below)

        # ---------------- units ----------------
        units = nwb.units
        classification = np.asarray([str(c) for c in units['classification'][:]])
        anno = np.asarray([str(a) for a in units['anno_name'][:]])
        good = classification == 'good'
        if good.sum() == 0:
            return None

        # unit -> electrode (exactly one electrode per unit) -> CCF ML coordinate
        vi = units['electrodes']
        offs = np.asarray(vi.data[:])
        starts = np.concatenate([[0], offs[:-1]])
        elec_row = np.asarray(vi.target.data[:])[starts]
        electrodes = nwb.electrodes.to_dataframe()
        ml = electrodes['x'].values.astype(float)[elec_row]

        region_idx = np.full(len(classification), -1, dtype=np.int64)
        for i in np.where(good)[0]:
            reg = coarse_region(anno[i])
            if reg is None or not np.isfinite(ml[i]):
                continue
            side = 'left' if ml[i] >= ML_MIDLINE else 'right'   # reference helper_get_neuron_id_area
            region_idx[i] = BRAIN_REGION_INDEX['%s %s' % (side, reg)]
        use_unit = good & (region_idx >= 0)
        unit_ids = np.where(use_unit)[0]
        n_units = len(unit_ids)
        if n_units == 0:
            return None

        # ---- electrophysiology coverage ----
        # units.obs_intervals lists the trials during which each unit was actually recorded.
        # In 8 of the 173 sessions the ephys recording stops before the behavioural session
        # ends, so the later trials have no spike data at all; those trials must be dropped
        # rather than turned into all-zero firing rates.
        obs_trial = np.ones(n_trials_all, dtype=bool)
        obs_index = units['obs_intervals']
        for ui in unit_ids:
            oi = np.asarray(obs_index[int(ui)], dtype=float)
            m = np.zeros(n_trials_all, dtype=bool)
            if oi.size:
                rows = np.searchsorted(start_time, oi[:, 0] + 1e-6) - 1
                rows = rows[(rows >= 0) & (rows < n_trials_all)]
                m[rows] = True
            obs_trial &= m
        n_trials_unobserved = int(np.sum(~obs_trial))
        keep &= obs_trial

        trial_idx = np.where(keep)[0]
        n_trials = len(trial_idx)
        if n_trials < 2:
            return None

        go = go_times[trial_idx]
        tstart = start_time[trial_idx]
        tstop = stop_time[trial_idx]

        edges = go[:, None] + BIN_EDGES_REL[None, :]           # (n_trials, NBINS+1)
        edges_clipped = np.clip(edges, tstart[:, None], tstop[:, None])
        observed = (edges[:, 1:] <= tstop[:, None]) & (edges[:, :-1] >= tstart[:, None])
        timing['setup'] = time.time() - t0

        # ---------------- neural: spike counts -> firing rate (Hz) ----------------
        t1 = time.time()
        flat_edges = edges_clipped.ravel()
        trial_bounds = np.stack([tstart, tstop], axis=1).ravel()
        rates = np.zeros((n_units, n_trials, NBINS), dtype=np.float32)
        spikes_in_trial = np.zeros(n_trials, dtype=np.int64)
        spike_index = units['spike_times']
        for k, ui in enumerate(unit_ids):
            st = np.asarray(spike_index[int(ui)], dtype=float)
            pos = np.searchsorted(st, flat_edges).reshape(n_trials, NBINS + 1)
            rates[k] = np.diff(pos, axis=1).astype(np.float32)
            tb = np.searchsorted(st, trial_bounds).reshape(n_trials, 2)
            spikes_in_trial += tb[:, 1] - tb[:, 0]
        rates /= BIN_SIZE                                        # spikes/s, as in the reference

        # A trial in which none of the simultaneously recorded good units fires a single
        # spike over its whole ~5 s duration is not physiological: it means the ephys
        # recording had already stopped (obs_intervals can list one such trailing trial).
        has_spikes = spikes_in_trial > 0
        n_trials_nospike = int(np.sum(~has_spikes))
        if n_trials_nospike:
            trial_idx = trial_idx[has_spikes]
            go = go[has_spikes]
            tstart = tstart[has_spikes]
            tstop = tstop[has_spikes]
            edges = edges[has_spikes]
            edges_clipped = edges_clipped[has_spikes]
            observed = observed[has_spikes]
            rates = rates[:, has_spikes, :]
            n_trials = len(trial_idx)
            if n_trials < 2:
                return None
        timing['neural'] = time.time() - t1

        # ---------------- inputs ----------------
        t1 = time.time()
        # (a) time from tone (sample epoch) onset. Early-lick trials replay the sample epoch,
        #     so the tone that preceded a given go cue is the LAST sample onset before it.
        pos = np.searchsorted(sample_times, go, side='left') - 1
        tone_onset = np.where(pos >= 0, sample_times[np.clip(pos, 0, len(sample_times) - 1)], np.nan)
        bad_tone = ~np.isfinite(tone_onset) | (tone_onset < tstart - 1e-9)
        n_bad_tone = int(bad_tone.sum())
        tone_onset = np.where(bad_tone, go - 1.85, tone_onset)   # fallback: standard structure
        time_from_tone = (go - tone_onset)[:, None] + BIN_CENTERS_REL[None, :]   # (n_trials, NBINS)

        # (b) photostimulation on/off at each time point
        photostim_b = np.zeros((n_trials, NBINS), dtype=bool)
        if len(stim_on):
            which = np.searchsorted(tstart, stim_on, side='right') - 1
            for s_on, s_off, w in zip(stim_on, stim_off, which):
                if w < 0 or w >= n_trials:
                    continue
                if s_on < tstart[w] or s_on > tstop[w]:
                    continue
                ov = (edges[w, :-1] < s_off) & (edges[w, 1:] > s_on)
                photostim_b[w] |= ov
        photostim = photostim_b.astype(np.float32)
        inputs = np.stack([time_from_tone.astype(np.float32), photostim], axis=1)  # (n_trials, 2, NBINS)
        timing['input'] = time.time() - t1

        # ---------------- outputs ----------------
        t1 = time.time()
        # (a) lick direction choice: direction of the first lick in the answer period.
        #     The methods define a 1.5 s answer period after the go cue in which the mouse
        #     reports its choice; this also coincides with the end of the decoding window.
        answer_end = np.minimum(tstop, go + 1.5)
        choice = np.full(n_trials, 2, dtype=np.int64)            # 2 = no lick
        for i in range(n_trials):
            l = left_licks[(left_licks >= go[i]) & (left_licks <= answer_end[i])]
            r = right_licks[(right_licks >= go[i]) & (right_licks <= answer_end[i])]
            if len(l) == 0 and len(r) == 0:
                continue
            if len(r) == 0:
                choice[i] = 0
            elif len(l) == 0:
                choice[i] = 1
            else:
                choice[i] = 0 if l[0] < r[0] else 1

        # (b) outcome
        omap = {'ignore': 0, 'miss': 1, 'hit': 2}
        outcome = np.array([omap[str(o)] for o in outcome_str[trial_idx]], dtype=np.int64)

        # (c) early lick
        early = np.array([1 if str(e) == 'early' else 0 for e in early_str[trial_idx]], dtype=np.int64)

        # (d) tongue y position (DeepLabCut side camera, 300 Hz)
        ts_obj = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
        vts = np.asarray(ts_obj.timestamps[:], dtype=float)
        vdata = np.asarray(ts_obj.data[:], dtype=float)
        tongue_y = vdata[:, 1]
        visible = vdata[:, 2] > LIKELIHOOD_THRESH
        ysum = np.concatenate([[0.0], np.cumsum(np.where(visible, tongue_y, 0.0))])
        vcnt = np.concatenate([[0], np.cumsum(visible.astype(np.int64))])
        vidx = np.searchsorted(vts, edges_clipped)               # (n_trials, NBINS+1)
        s_y = ysum[vidx[:, 1:]] - ysum[vidx[:, :-1]]
        n_y = vcnt[vidx[:, 1:]] - vcnt[vidx[:, :-1]]
        with np.errstate(invalid='ignore', divide='ignore'):
            ybin = np.where(n_y > 0, s_y / np.maximum(n_y, 1), np.nan)
        vis_vals = ybin[np.isfinite(ybin)]
        if vis_vals.size >= 10:
            p40, p60 = np.percentile(vis_vals, [40, 60])
        else:
            p40 = p60 = np.nan
        tongue_cls = np.full((n_trials, NBINS), 3, dtype=np.int64)    # 3 = not visible
        if np.isfinite(p40):
            fin = np.isfinite(ybin)
            tongue_cls[fin & (ybin < p40)] = 0
            tongue_cls[fin & (ybin >= p40) & (ybin <= p60)] = 1
            tongue_cls[fin & (ybin > p60)] = 2

        outputs = np.empty((n_trials, 4, NBINS), dtype=np.int64)
        outputs[:, 0, :] = choice[:, None]
        outputs[:, 1, :] = outcome[:, None]
        outputs[:, 2, :] = early[:, None]
        outputs[:, 3, :] = tongue_cls
        timing['output'] = time.time() - t1

        result = {
            'session_id': session_id,
            'subject_id': subject_id,
            'mouse_name': mouse_name,
            'path': path,
            'neural': [np.ascontiguousarray(rates[:, i, :]) for i in range(n_trials)],
            'input': [np.ascontiguousarray(inputs[i]) for i in range(n_trials)],
            'output': [np.ascontiguousarray(outputs[i]) for i in range(n_trials)],
            'brain_region_idx': region_idx[unit_ids].astype(np.int64),
            'n_units_total': int(len(classification)),
            'n_units_good': int(good.sum()),
            'n_units_used': int(n_units),
            'n_trials_all': int(n_trials_all),
            'n_trials_used': int(n_trials),
            'n_trials_autowater': int((auto_water > 0).sum()),
            'n_trials_freewater': int((free_water > 0).sum()),
            'frac_fully_observed': float(np.mean(observed.all(axis=1))),
            'frac_bins_observed': float(np.mean(observed)),
            'trial_index': trial_idx.astype(np.int32),
            'go_times': go.astype(np.float64),
            'n_bad_tone': n_bad_tone,
            'n_trials_unobserved': n_trials_unobserved,
            'n_trials_nospike': n_trials_nospike,
            'tongue_pct': (float(p40), float(p60)),
            'timing': timing,
            'total_time': time.time() - t0,
        }

        if make_plots:
            plot_session(result, dict(go=go, tstart=tstart, tstop=tstop, tone_onset=tone_onset,
                                      vts=vts, tongue_y=tongue_y, visible=visible,
                                      ybin=ybin, p40=p40, p60=p60, spike_index=spike_index,
                                      unit_ids=unit_ids, edges=edges, observed=observed,
                                      left_licks=left_licks, right_licks=right_licks,
                                      stim_on=stim_on, stim_off=stim_off, choice=choice,
                                      outcome=outcome, early=early))
    return result


# ------------------------------------------------------------------------------ plotting
def plot_session(res, aux):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    go = aux['go']
    n_trials = len(go)
    # pick a few illustrative trials: one hit, one miss, one ignore, one photostim if possible
    picks = []
    for cond, name in [((aux['outcome'] == 2), 'hit'), ((aux['outcome'] == 1), 'miss'),
                       ((aux['outcome'] == 0), 'ignore'),
                       ((np.array([res['input'][i][1].any() for i in range(n_trials)])), 'photostim')]:
        idx = np.where(cond)[0]
        if len(idx):
            picks.append((int(idx[len(idx) // 2]), name))
    if not picks:
        picks = [(0, 'trial0')]

    fig, axes = plt.subplots(5, len(picks), figsize=(6 * len(picks), 16), squeeze=False)
    for col, (tr, name) in enumerate(picks):
        g = go[tr]
        # --- 1. spike raster (raw spike times relative to go cue) ---
        ax = axes[0][col]
        nshow = min(40, len(aux['unit_ids']))
        for j, ui in enumerate(aux['unit_ids'][:nshow]):
            st = np.asarray(aux['spike_index'][int(ui)], dtype=float) - g
            st = st[(st >= T_START) & (st < T_STOP)]
            ax.plot(st, np.full(len(st), j), '|', color='k', markersize=3)
        ax.axvline(0, color='r', label='go cue')
        ax.axvline(aux['tone_onset'][tr] - g, color='b', label='tone onset')
        ax.axvline(aux['tstop'][tr] - g, color='gray', ls='--', label='trial end')
        ax.set_xlim(T_START, T_STOP)
        ax.set_title('%s / trial %d (%s)\nraw spikes (40 units)' % (res['session_id'], tr, name))
        ax.set_ylabel('unit')
        ax.legend(fontsize=6, loc='upper left')

        # --- 2. binned firing rates ---
        ax = axes[1][col]
        im = ax.imshow(res['neural'][tr][:nshow], aspect='auto', origin='lower',
                       extent=[T_START, T_STOP, 0, nshow], cmap='magma')
        ax.axvline(0, color='c')
        ax.set_title('binned firing rate (Hz), 50 ms bins')
        ax.set_ylabel('unit')
        plt.colorbar(im, ax=ax, fraction=0.03)

        # --- 3. inputs ---
        ax = axes[2][col]
        ax.plot(BIN_CENTERS_REL, res['input'][tr][0], '.-', label='time from tone onset (s)')
        ax.plot(BIN_CENTERS_REL, res['input'][tr][1], '.-', label='photostim')
        ax.axvline(0, color='r')
        ax.axvline(aux['tone_onset'][tr] - g, color='b')
        for s_on, s_off in zip(aux['stim_on'], aux['stim_off']):
            if aux['tstart'][tr] <= s_on <= aux['tstop'][tr]:
                ax.axvspan(s_on - g, s_off - g, color='violet', alpha=0.3)
        ax.set_title('inputs (vertical lines: go cue red, tone blue; shading: laser)')
        ax.legend(fontsize=7)

        # --- 4. tongue tracking ---
        ax = axes[3][col]
        m = (aux['vts'] >= g + T_START) & (aux['vts'] < g + T_STOP)
        ax.plot(aux['vts'][m] - g, np.where(aux['visible'][m], aux['tongue_y'][m], np.nan),
                '.', ms=2, color='g', label='raw tongue y (visible frames)')
        ax.plot(BIN_CENTERS_REL, aux['ybin'][tr], 'o-', color='k', ms=3, label='binned mean y')
        ax.axhline(aux['p40'], color='orange', ls='--', label='40th pct')
        ax.axhline(aux['p60'], color='red', ls='--', label='60th pct')
        ax.axvline(0, color='r')
        ax.set_title('tongue y position and session percentiles')
        ax.legend(fontsize=6)

        # --- 5. outputs ---
        ax = axes[4][col]
        for k in range(4):
            ax.step(BIN_CENTERS_REL, res['output'][tr][k] + 5 * k, where='mid',
                    label='%s (+%d)' % (OUTPUT_NAMES[k], 5 * k))
        for t in aux['left_licks'][(aux['left_licks'] > g + T_START) & (aux['left_licks'] < g + T_STOP)]:
            ax.axvline(t - g, color='b', alpha=0.25)
        for t in aux['right_licks'][(aux['right_licks'] > g + T_START) & (aux['right_licks'] < g + T_STOP)]:
            ax.axvline(t - g, color='m', alpha=0.25)
        ax.axvline(0, color='r')
        ax.set_title('outputs (blue=left licks, magenta=right licks)')
        ax.set_xlabel('time from go cue (s)')
        ax.legend(fontsize=6)

    fig.tight_layout()
    fname = 'processing_%s.png' % res['session_id']
    fig.savefig(fname, dpi=110)
    plt.close(fig)

    # --- session-level summary figure ---
    fig, axes = plt.subplots(1, 4, figsize=(22, 4))
    psth = np.mean(np.stack([n.mean(axis=0) for n in res['neural']]), axis=0)
    axes[0].plot(BIN_CENTERS_REL, psth)
    axes[0].axvline(0, color='r')
    axes[0].set_title('population mean firing rate (Hz)')
    axes[0].set_xlabel('time from go cue (s)')
    outs = np.stack(res['output'])
    axes[1].plot(BIN_CENTERS_REL, (outs[:, 3, :] != 3).mean(axis=0))
    axes[1].axvline(0, color='r')
    axes[1].set_title('fraction of trials with tongue visible')
    axes[2].plot(BIN_CENTERS_REL, np.stack(res['input'])[:, 1, :].mean(axis=0))
    axes[2].axvline(0, color='r')
    axes[2].set_title('fraction of trials with photostim on')
    vals, cnts = np.unique(outs[:, 3, :], return_counts=True)
    axes[3].bar(vals, cnts / cnts.sum())
    axes[3].set_xticks(range(4))
    axes[3].set_xticklabels(OUTPUT_VALUES[3], rotation=30)
    axes[3].set_title('tongue class distribution')
    fig.tight_layout()
    fig.savefig('processing_%s_summary.png' % res['session_id'], dpi=110)
    plt.close(fig)


def _worker(args):
    path, make_plots = args
    try:
        res = process_session(path, make_plots=make_plots)
    except Exception as exc:  # noqa
        import traceback
        traceback.print_exc()
        return {'path': path, 'error': repr(exc)}
    if res is None:
        return {'path': path, 'skipped': True}
    return res


def main():
    ap = argparse.ArgumentParser(description='Convert MAP NWB dataset for decoding.')
    ap.add_argument('outfile', type=str)
    ap.add_argument('--full', action='store_true', help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true', help='plot every processing step')
    ap.add_argument('--nproc', type=int, default=16)
    ap.add_argument('--data-dir', type=str, default='/app/data')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
    if args.sample:
        files = files[:2]
    print('Converting %d sessions' % len(files), flush=True)

    plot_n = 2 if args.show_processing else 0
    jobs = [(f, i < plot_n) for i, f in enumerate(files)]

    t0 = time.time()
    results = []
    if args.nproc > 1 and len(files) > 1:
        from multiprocessing import Pool
        with Pool(min(args.nproc, len(files))) as pool:
            for i, r in enumerate(pool.imap(_worker, jobs)):
                results.append(r)
                el = time.time() - t0
                print('[%3d/%3d] %-28s %.1fs elapsed (%.1fs/session, eta %.1f min)' % (
                    i + 1, len(files), r.get('session_id', os.path.basename(r['path']))[:28],
                    el, el / (i + 1), (len(files) - i - 1) * el / (i + 1) / 60), flush=True)
    else:
        for i, job in enumerate(jobs):
            r = _worker(job)
            results.append(r)
            el = time.time() - t0
            print('[%3d/%3d] %-28s %.1fs elapsed' % (i + 1, len(files),
                  r.get('session_id', os.path.basename(r['path']))[:28], el), flush=True)

    errors = [r for r in results if 'error' in r]
    skipped = [r for r in results if r.get('skipped')]
    sessions = [r for r in results if 'neural' in r]
    for r in errors:
        print('ERROR', r['path'], r['error'])
    for r in skipped:
        print('SKIPPED (no good units / <2 trials):', os.path.basename(r['path']))
    print('Converted %d sessions (%d skipped, %d errors) in %.1f s' % (
        len(sessions), len(skipped), len(errors), time.time() - t0), flush=True)

    # -------------------- assemble --------------------
    subjects = sorted(set(r['subject_id'] for r in sessions))
    subject_index = {s: i for i, s in enumerate(subjects)}
    data = {
        'neural': [r['neural'] for r in sessions],
        'input': [r['input'] for r in sessions],
        'output': [r['output'] for r in sessions],
        'subjects': subjects,
        'subject_idx': np.array([subject_index[r['subject_id']] for r in sessions], dtype=np.int64),
        'brain_regions': BRAIN_REGIONS,
        'brain_region_idx': [r['brain_region_idx'] for r in sessions],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'dataset': 'Mesoscale Activity Map (MAP), DANDI:000363 (Chen et al. 2023)',
            'papers': ['Chen et al., Brain-wide neural activity underlying memory-guided movement',
                       'Wang, Kurgyis et al. 2025, Brain-wide analysis reveals movement encoding '
                       'structured across and within brain areas'],
            'task_description': (
                'Head-fixed mice performed an auditory delayed-response task. A tone (3 or 12 kHz, '
                '3 x 150 ms pips, 0.65 s sample epoch) instructed lick-left vs lick-right; after a '
                '1.2 s delay an auditory go cue (6 kHz, 0.1 s) opened a 1.5 s answer period in which '
                'the mouse licked one of two ports. Bilateral/unilateral ALM photoinhibition occurred '
                'on ~20% of randomly interleaved trials during the last 0.5 s of the delay. '
                'Decoder inputs: time from tone (sample epoch) onset, and whether photostimulation is '
                'on at each time point. Decoder outputs: lick direction choice (left/right/no lick), '
                'trial outcome (ignore/miss/hit), early lick (no/yes) and discretized tongue '
                'y-position from the side-view DeepLabCut tracking (<40th pct / 40-60th pct / >60th pct '
                'of the session distribution / not visible).'),
            'time_bin_size': BIN_SIZE * 1000.0,
            'temporal_alignment_event': 'go cue onset (auditory Go cue, BehavioralEvents.go_start_times)',
            'off_start': T_START,
            'off_end': T_STOP,
            'bin_centers': BIN_CENTERS_REL.tolist(),
            'neural_units': 'firing rate in spikes/s (spike counts per 50 ms bin / 0.05 s)',
            'neuron_curation': ("units.classification == 'good' (region-specific QC classifiers of "
                                'Chen, Liu et al. 2023 white paper) with a valid CCF annotation'),
            'trial_curation': ('auto-water and free-water trials excluded (reward not contingent on '
                               'choice), as in the reference get_regular_trial_mask; early-lick, '
                               'no-response (ignore) and photostimulation trials are KEPT because they '
                               'are required decoder outputs/inputs'),
            'session_curation': 'sessions with no good units excluded',
            'partial_observation': ('spikes and video frames exist only within trial intervals '
                                    '(obs_intervals == trial start/stop). Bins of the [-2.5, 1.5] s '
                                    'window that fall outside the trial are therefore empty: firing '
                                    'rate 0 and tongue class 3 (not visible). Mean fraction of fully '
                                    'observed trials: %.3f' % float(np.mean([r['frac_fully_observed'] for r in sessions]))),
            'video_tracking': ('side-view camera, 300 Hz, DeepLabCut tongue marker; a frame counts as '
                               'visible when likelihood > %.2f; the per-bin value is the mean y over '
                               'visible frames in the bin' % LIKELIHOOD_THRESH),
            'session_info': [{'session_id': r['session_id'], 'subject_id': r['subject_id'],
                              'mouse_name': r['mouse_name'], 'n_units': r['n_units_used'],
                              'n_trials': r['n_trials_used'], 'n_trials_in_file': r['n_trials_all'],
                              'frac_fully_observed': r['frac_fully_observed'],
                              'tongue_pct_40_60': r['tongue_pct'],
                              'nwb_file': r['path'],
                              'trial_index': r['trial_index'],
                              'go_times': r['go_times']} for r in sessions],
        },
    }

    # -------------------- summary / sanity --------------------
    n_neurons = sum(len(r['brain_region_idx']) for r in sessions)
    n_trials = sum(len(r['neural']) for r in sessions)
    print('\n=== SUMMARY ===')
    print('sessions: %d   subjects: %d   neurons: %d   trials: %d' % (
        len(sessions), len(subjects), n_neurons, n_trials))
    print('neurons/session: mean %.1f range %d-%d' % (
        np.mean([r['n_units_used'] for r in sessions]),
        min(r['n_units_used'] for r in sessions), max(r['n_units_used'] for r in sessions)))
    print('trials/session: mean %.1f range %d-%d' % (
        np.mean([r['n_trials_used'] for r in sessions]),
        min(r['n_trials_used'] for r in sessions), max(r['n_trials_used'] for r in sessions)))
    allregions = np.concatenate([r['brain_region_idx'] for r in sessions])
    print('neurons per brain region:')
    for i, name in enumerate(BRAIN_REGIONS):
        c = int(np.sum(allregions == i))
        if c:
            print('   %-22s %6d' % (name, c))
    coarse = {}
    for i, name in enumerate(BRAIN_REGIONS):
        coarse[name.split(' ', 1)[1]] = coarse.get(name.split(' ', 1)[1], 0) + int(np.sum(allregions == i))
    print('coarse regions:', {k: v for k, v in sorted(coarse.items(), key=lambda kv: -kv[1])})
    outs = np.concatenate([np.stack(r['output'])[:, :, 0] for r in sessions])   # per-trial labels
    for k in range(3):
        vals, cnts = np.unique(outs[:, k], return_counts=True)
        print('%s distribution: %s' % (OUTPUT_NAMES[k],
              {OUTPUT_VALUES[k][int(v)]: round(c / cnts.sum(), 4) for v, c in zip(vals, cnts)}))
    tongue = np.concatenate([np.stack(r['output'])[:, 3, :].ravel() for r in sessions])
    vals, cnts = np.unique(tongue, return_counts=True)
    print('tongue_y_position distribution: %s' % (
        {OUTPUT_VALUES[3][int(v)]: round(c / cnts.sum(), 4) for v, c in zip(vals, cnts)}))
    inp = np.concatenate([np.stack(r['input'])[:, 0, :].ravel() for r in sessions])
    print('time_from_tone_onset range: [%.3f, %.3f]' % (inp.min(), inp.max()))
    stim = np.concatenate([np.stack(r['input'])[:, 1, :] for r in sessions])
    print('photostim: %.4f of bins, %.4f of trials' % (stim.mean(), np.mean(stim.any(axis=1))))
    print('mean fraction of fully observed trials: %.4f' % np.mean([r['frac_fully_observed'] for r in sessions]))
    print('trials with no tone onset found (fallback used): %d' % sum(r['n_bad_tone'] for r in sessions))
    print('trials dropped for lack of ephys coverage: %d' % sum(r.get('n_trials_unobserved', 0) for r in sessions))
    print('trials dropped for zero spikes across all units: %d' % sum(r.get('n_trials_nospike', 0) for r in sessions))

    t1 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print('Saved %s (%.2f GB) in %.1f s' % (args.outfile, os.path.getsize(args.outfile) / 1e9,
                                            time.time() - t1))


if __name__ == '__main__':
    main()
