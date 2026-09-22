"""
Convert the Mesoscale Activity Map (MAP) NWB dataset (DANDI 000363, Chen et al. 2024)
into the decoder-compatible pickle format.

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Processing follows the reference pipeline of Wang*, Kurgyis* et al. 2025
(/app/code, VideoAnalysisUtils/preprocessing_DJ_2022Aug.py, Sherlock/align_markers.py)
and the data paper (Chen et al. 2024) wherever applicable; deviations that are required
by the decoder task are documented in /app/CONVERSION_NOTES.md.

Key choices
-----------
* alignment event : go cue (`BehavioralEvents/go_start_times`)
* window          : -2.5 s to +1.5 s around the go cue
* bins            : 80 non-overlapping 50-ms bins, firing rate in Hz (counts / bin width)
* units           : `units/classification == 'good'` (NWB encoding of the white-paper QC classifier)
* sessions        : data-paper criteria (performance > 65%, >= 50 correct left and right trials)
* trials          : auto-water / free-water trials removed; video must cover the window
"""

import argparse
import json
import os
import pickle
import sys
import time
import glob
import traceback
from collections import Counter
from multiprocessing import Pool

import h5py
import numpy as np

# ----------------------------------------------------------------------------- constants
DATA_DIR = '/app/data'
ALIGN_EVENT = 'go cue onset'
T_START = -2.5           # s relative to go cue
T_END = 1.5              # s relative to go cue
BIN_SIZE = 0.05          # s
NBINS = int(round((T_END - T_START) / BIN_SIZE))   # 80
BIN_EDGES_REL = T_START + BIN_SIZE * np.arange(NBINS + 1)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2

LIKELIHOOD_THRESH = 0.9  # DeepLabCut likelihood above which the tongue counts as visible
ML_MIDLINE = 5700.0      # um, CCF ML coordinate of the midline (reference code)
AP_BREGMA = 5400.0       # um, CCF AP coordinate of bregma (reference code global offset)
ALM_AP_MIN = 2000.0      # um anterior of bregma for a MOs/FRP unit to count as ALM

MIN_PERFORMANCE = 0.65   # data paper session selection
MIN_CORRECT_PER_SIDE = 50
IGT_TRIAL_FRAC = 0.9     # keep trial if >= 90% of good units are flagged good on it

# ------------------------------------------------------------------ CCF -> coarse region
THAL = ['thalam', 'geniculate', 'habenula', 'Parafascicular', 'Subparafascicular',
        'Paracentral nucleus', 'Anterodorsal nucleus', 'Anteromedial nucleus',
        'Anteroventral nucleus', 'Rhomboid nucleus', 'Reuniens', 'Perireunensis',
        'Interanterodorsal', 'Intermediodorsal', 'Suprageniculate', 'Posterior limiting',
        'Peripeduncular', 'Submedial', 'Ethmoid nucleus', 'Posterior intralaminar',
        'Xiphoid', 'Lateral dorsal nucleus', 'Ventral posterior complex',
        'Central medial nucleus', 'Central lateral nucleus', 'Reticular nucleus of the thalamus',
        'Mediodorsal nucleus', 'Parataenial', 'Posterior complex', 'Lateral posterior nucleus',
        'Ventral medial nucleus', 'Ventral anterior-lateral', 'Ventral posterolateral',
        'Ventral posteromedial', 'Nucleus of the optic tract']
MID = ['Midbrain', 'Superior colliculus', 'Inferior colliculus', 'Substantia nigra',
       'Red nucleus', 'pretectal', 'Pretectal', 'Periaqueductal', 'Ventral tegmental',
       'Nucleus sagulum', 'Anterior tegmental', 'Edinger', 'Interpeduncular',
       'Nucleus of the brachium', 'Cuneiform', 'Oculomotor', 'Trochlear',
       'Nucleus of Darkschewitsch', 'Intercollicular', 'Parabigeminal', 'Pedunculopontine',
       'Nucleus of the posterior commissure', 'Retrorubral', 'Medial terminal nucleus',
       'Subcommissural', 'Rostral linear', 'Interstitial nucleus of Cajal',
       'Midbrain trigeminal', 'accessory optic tract']
PONS = ['Pons', 'Pontine', 'Parabrachial', 'Superior olivary', 'lateral lemniscus',
        'Tegmental reticular', 'Locus ceruleus', 'Laterodorsal tegmental', 'Barrington',
        'Dorsal nucleus raphe', 'Nucleus raphe pontis', 'Sublaterodorsal',
        'Principal sensory nucleus of the trigeminal', 'Motor nucleus of trigeminal',
        'Supratrigeminal', 'Superior central raphe', 'Nucleus incertus', 'Peritrigeminal',
        'Koelliker']
MED = ['Medulla', 'Gigantocellular', 'Intermediate reticular', 'Magnocellular reticular',
       'Parvicellular reticular', 'Paragigantocellular', 'Medullary reticular',
       'solitary tract', 'Spinal nucleus of the trigeminal', 'Vestibular nucleus',
       'cuneate', 'Cuneate', 'Gracile', 'Inferior olivary', 'Facial motor nucleus',
       'Nucleus ambiguus', 'Hypoglossal', 'vagus nerve', 'Nucleus prepositus',
       'Lateral reticular nucleus', 'Nucleus of Roller', 'Nucleus x', 'Nucleus y',
       'Raphe magnus', 'Raphe obscurus', 'Raphe pallidus', 'Parasolitary', 'Area postrema',
       'Linear nucleus of the medulla', 'Perihypoglossal', 'trapezoid body',
       'Efferent vestibular', 'Efferent cochlear', 'Cochlear nucleus', 'cochlear nucleus',
       'Paramedian reticular', 'Abducens', 'Parapyramidal']
CB = ['Cerebell', 'Lobule', 'Declive', 'Uvula', 'Nodulus', 'Pyramus', 'Culmen', 'Folium',
      'Lingula', 'Crus 1', 'Crus 2', 'Paramedian lobule', 'Copula', 'Flocculus',
      'Paraflocculus', 'Simple lobule', 'Ansiform', 'Fastigial', 'Interposed',
      'Dentate nucleus', 'arbor vitae']
HIP = ['Field CA1', 'Field CA2', 'Field CA3', 'Dentate gyrus', 'ubiculum', 'Entorhinal',
       'Hippocamp', 'Induseum', 'Fasciola']
OLF = ['Olfactory', 'olfactory', 'Piriform', 'Taenia tecta', 'Postpiriform',
       'Accessory olfactory', 'Cortical amygdalar area']
CSP = ['Cortical subplate', 'Endopiriform', 'Claustrum', 'amygdalar nucleus',
       'Intercalated amygdalar', 'Basolateral', 'Basomedial', 'Posterior amygdalar']
STR = ['Striatum', 'Caudoputamen', 'Nucleus accumbens', 'Olfactory tubercle',
       'Fundus of striatum', 'Lateral septal', 'Septofimbrial', 'Septohippocampal',
       'stria terminalis', 'Medial amygdalar', 'Striatum-like']
PAL = ['Globus pallidus', 'Pallidum', 'pallidal', 'Substantia innominata',
       'Magnocellular nucleus', 'Medial septal', 'Diagonal band',
       'Triangular nucleus of septum', 'anterior commissure']
HY = ['Hypothalamus', 'hypothalamic', 'Subthalamic', 'ammillary', 'Tuberal', 'Preoptic',
      'preoptic', 'Periventricular', 'Arcuate hypothalamic', 'Zona incerta',
      'Fields of Forel', 'Parasubthalamic', 'Posterior hypothalamic', 'Lateral hypothalamic']
ORB = ['Orbital area']
MOTOR = ['Secondary motor area', 'Primary motor area', 'Frontal pole']


def _has(name, keys):
    low = name.lower()
    return any(k.lower() in low for k in keys)


def ccf_to_region(anno_name, ap_um):
    """Map an Allen CCF annotation to one of the coarse regions used by the reference code.

    `ap_um` is the AP coordinate in um relative to bregma (positive = anterior); it is used
    only to separate ALM (anterior MOs / frontal pole) from the rest of the cortex, following
    the ALM definition of the data paper (AP 2.5 mm, ML 1.5 mm).
    """
    n = anno_name.strip()
    if n == '':
        return 'unknown'
    if _has(n, CB):
        return 'Cerebellum'
    if _has(n, MED):
        return 'Medulla'
    if _has(n, PONS):
        return 'Pons'
    if _has(n, MID):
        return 'Midbrain'
    if _has(n, THAL):
        return 'Thalamus'
    if _has(n, HY):
        return 'Hypothalamus'
    if _has(n, HIP):
        return 'Hippocampus'
    if _has(n, CSP):
        return 'CorticalSubplate'
    if _has(n, PAL):
        return 'Pallidum'
    if _has(n, STR):
        return 'Striatum'
    if _has(n, OLF):
        return 'Olfactory'
    if _has(n, ORB):
        return 'Orbital'
    if _has(n, MOTOR):
        if (n.startswith('Secondary motor area') or n.startswith('Frontal pole')) and ap_um >= ALM_AP_MIN:
            return 'ALM'
        return 'OtherCortex'
    if 'area' in n.lower() or 'cortex' in n.lower() or 'layer' in n.lower():
        return 'OtherCortex'
    return 'unknown'


def dec(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def decode_array(arr):
    return np.array([dec(x) for x in arr])


def to_float(x):
    try:
        return float(dec(x))
    except Exception:
        return np.nan


# ------------------------------------------------------------------ per-session conversion
def process_session(args):
    """Process one NWB session file. Returns a dict (or None if the session is rejected)."""
    fname, show_processing = args
    t0 = time.time()
    info = {'file': os.path.basename(fname), 'timing': {}}
    try:
        with h5py.File(fname, 'r') as f:
            # --------------------------------------------------- trials / behaviour
            tr = f['intervals/trials']
            start_time = tr['start_time'][:]
            stop_time = tr['stop_time'][:]
            outcome = decode_array(tr['outcome'][:])
            early_lick = decode_array(tr['early_lick'][:])
            instruction = decode_array(tr['trial_instruction'][:])
            auto_water = tr['auto_water'][:].astype(int)
            free_water = tr['free_water'][:].astype(int)
            ps_onset = np.array([to_float(x) for x in tr['photostim_onset'][:]])
            ps_dur = np.array([to_float(x) for x in tr['photostim_duration'][:]])
            ps_power = np.array([to_float(x) for x in tr['photostim_power'][:]])
            ntrials_all = len(start_time)

            be = f['acquisition/BehavioralEvents']
            go = be['go_start_times/timestamps'][:]
            assert len(go) == ntrials_all, 'go cue count != trial count'
            sample_on = be['sample_start_times/timestamps'][:]
            left_lick = be['left_lick_times/timestamps'][:]
            right_lick = be['right_lick_times/timestamps'][:]

            # --------------------------------------------------- session-level selection
            is_stim = np.isfinite(ps_power) & (ps_power > 0)
            control = (~is_stim) & (early_lick == 'no early') & (auto_water == 0) & (free_water == 0)
            n_hit = int(np.sum(control & (outcome == 'hit')))
            n_miss = int(np.sum(control & (outcome == 'miss')))
            performance = n_hit / max(n_hit + n_miss, 1)
            n_correct_left = int(np.sum(control & (outcome == 'hit') & (instruction == 'left')))
            n_correct_right = int(np.sum(control & (outcome == 'hit') & (instruction == 'right')))
            info.update(performance=performance, n_correct_left=n_correct_left,
                        n_correct_right=n_correct_right, ntrials_all=ntrials_all)

            if performance <= MIN_PERFORMANCE:
                info['reject'] = 'performance %.3f <= %.2f' % (performance, MIN_PERFORMANCE)
                return None, info
            if n_correct_left < MIN_CORRECT_PER_SIDE or n_correct_right < MIN_CORRECT_PER_SIDE:
                info['reject'] = 'too few correct trials (L %d, R %d)' % (n_correct_left, n_correct_right)
                return None, info

            # --------------------------------------------------- units and QC
            u = f['units']
            classification = decode_array(u['classification'][:])
            good = np.where(classification == 'good')[0]
            info['n_units_all'] = len(classification)
            info['n_units_good'] = len(good)
            if len(good) == 0:
                info['reject'] = 'no QC-good units'
                return None, info

            anno = decode_array(u['anno_name'][:])[good]
            # electrode of each unit -> CCF coordinates
            el_index = u['electrodes_index'][:]
            el_ids = u['electrodes'][:]
            first_el = el_ids[np.concatenate([[0], el_index[:-1]])]
            el = f['general/extracellular_ephys/electrodes']
            ex, ey, ez = el['x'][:], el['y'][:], el['z'][:]
            ux = ex[first_el][good]
            uz = ez[first_el][good]
            ap_um = AP_BREGMA - uz                    # positive = anterior of bregma
            hemi = np.where(ux >= ML_MIDLINE, 'left', 'right')   # reference convention
            regions = np.array(['%s %s' % (h, ccf_to_region(a, p))
                                for h, a, p in zip(hemi, anno, ap_um)])

            # ---- per-unit trial coverage -------------------------------------------
            # `units/obs_intervals` lists, for every unit, the trial intervals during which the
            # unit was actually recorded. In 9/174 sessions the probes were only run for a
            # subset of the behavioural trials, so trials outside these intervals contain no
            # spikes at all and must not be treated as silence.
            oi_index = u['obs_intervals_index'][:]
            oi_all = u['obs_intervals'][:]
            oi_starts = np.concatenate([[0], oi_index[:-1]])
            observed = np.zeros((len(good), ntrials_all), dtype=bool)
            for i, ui in enumerate(good):
                o = oi_all[oi_starts[ui]:oi_index[ui], 0]
                if len(o) == 0:
                    continue
                j = np.searchsorted(start_time, o)
                j = np.clip(j, 0, ntrials_all - 1)
                ok = np.isclose(start_time[j], o, atol=1e-6)
                observed[i, j[ok]] = True

            # is_good_trials curation (per-probe manual annotation of stable trials). In the
            # same 9 sessions this array only has as many columns as there are observed trials,
            # so it is expanded onto the observed trials of each unit.
            igt_raw = u['is_good_trials'][:][good]
            igt = np.ones((len(good), ntrials_all), dtype=bool)
            if igt_raw.shape[1] == ntrials_all:
                igt = igt_raw
                info['igt_used'] = 'direct'
            else:
                ok_expand = True
                for i in range(len(good)):
                    obs_idx = np.where(observed[i])[0]
                    if len(obs_idx) == igt_raw.shape[1]:
                        igt[i, obs_idx] = igt_raw[i]
                    else:
                        ok_expand = False
                info['igt_used'] = 'expanded' if ok_expand else 'ignored'
                info['igt_shape'] = list(igt_raw.shape)

            usable = observed & igt
            trial_stable = usable.mean(axis=0) >= IGT_TRIAL_FRAC

            # --------------------------------------------------- trial mask
            keep = (auto_water == 0) & (free_water == 0) & trial_stable

            # video coverage of the analysis window
            tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
            vts = tongue['timestamps'][:]
            vdata = tongue['data'][:]
            vy = vdata[:, 1]
            vlik = vdata[:, 2]
            visible_all = vlik > LIKELIHOOD_THRESH
            if visible_all.sum() < 100:
                info['reject'] = 'tongue never tracked in this session'
                return None, info
            y_p40, y_p60 = np.percentile(vy[visible_all], [40, 60])

            win_lo = go + T_START
            win_hi = go + T_END
            # number of video frames per bin, per trial
            edges_abs = go[:, None] + BIN_EDGES_REL[None, :]        # (ntrials, NBINS+1)
            vidx = np.searchsorted(vts, edges_abs)                  # video frames are sorted
            frames_per_bin = np.diff(vidx, axis=1)
            video_ok = np.all(frames_per_bin > 0, axis=1)
            keep = keep & video_ok

            trials = np.where(keep)[0]
            info['n_trials_kept'] = int(len(trials))
            info['n_trials_drop_water'] = int(np.sum((auto_water > 0) | (free_water > 0)))
            info['n_trials_drop_video'] = int(np.sum(~video_ok))
            info['n_trials_drop_igt'] = int(np.sum(~trial_stable))
            if len(trials) < 10:
                info['reject'] = 'fewer than 10 usable trials'
                return None, info

            # units that are still flagged bad on a retained trial are dropped
            unit_ok = usable[:, trials].all(axis=1)
            if unit_ok.sum() == 0:
                info['reject'] = 'no units stable over the retained trials'
                return None, info
            good = good[unit_ok]
            regions = regions[unit_ok]
            anno = anno[unit_ok]
            info['n_units_used'] = int(len(good))
            info['n_units_drop_igt'] = int(np.sum(~unit_ok))

            # --------------------------------------------------- neural: binned firing rates
            t1 = time.time()
            st_index = u['spike_times_index'][:]
            st_all = u['spike_times'][:]
            starts = np.concatenate([[0], st_index[:-1]])
            edges_kept = edges_abs[trials]                   # (ntrials_kept, NBINS+1)
            flat_edges = edges_kept.ravel()
            nkept = len(trials)
            rates = np.zeros((len(good), nkept, NBINS), dtype=np.float32)
            for i, ui in enumerate(good):
                sp = st_all[starts[ui]:st_index[ui]]
                # spike_times are stored sorted per unit; make sure
                counts = np.searchsorted(sp, flat_edges).reshape(nkept, NBINS + 1)
                rates[i] = np.diff(counts, axis=1).astype(np.float32)
            rates /= BIN_SIZE                                  # Hz

            # Occasionally a trial is listed in `obs_intervals` although the recording had
            # already stopped (e.g. the last trial of a truncated session): no unit fires a
            # single spike in the whole window. Such trials carry no neural information and
            # are removed.
            nonempty = rates.sum(axis=(0, 2)) > 0
            if not nonempty.all():
                info['n_trials_drop_nospikes'] = int((~nonempty).sum())
                rates = rates[:, nonempty, :]
                trials = trials[nonempty]
                edges_kept = edges_kept[nonempty]
                nkept = len(trials)
                if nkept < 10:
                    info['reject'] = 'fewer than 10 trials with spikes'
                    return None, info
            info['timing']['neural'] = time.time() - t1

            # --------------------------------------------------- inputs
            t1 = time.time()
            # tone (sample) onset: last sample epoch start before the go cue
            si = np.searchsorted(sample_on, go) - 1
            tone_abs = np.where(si >= 0, sample_on[np.clip(si, 0, len(sample_on) - 1)], np.nan)
            tone_rel = tone_abs - go                          # negative, ~ -1.85 s
            assert np.all(np.isfinite(tone_rel[trials])), 'missing tone onset'
            time_from_tone = BIN_CENTERS_REL[None, :] - tone_rel[trials][:, None]

            # photostimulation
            stim_input = np.zeros((nkept, NBINS), dtype=np.float32)
            if 'photostim_start_times' in be:
                pstart = be['photostim_start_times/timestamps'][:]
                pstop = be['photostim_stop_times/timestamps'][:]
            else:
                pstart = np.zeros(0)
                pstop = np.zeros(0)
            stim_rel = np.full((ntrials_all, 2), np.nan)
            if len(pstart):
                pt = np.searchsorted(start_time, pstart, side='right') - 1
                ok = (pt >= 0) & (pt < ntrials_all)
                stim_rel[pt[ok], 0] = pstart[ok] - go[pt[ok]]
                stim_rel[pt[ok], 1] = pstop[ok] - go[pt[ok]]
            # fall back to the trials table where the event stream is missing
            miss = is_stim & ~np.isfinite(stim_rel[:, 0])
            if miss.any():
                stim_rel[miss, 0] = start_time[miss] + ps_onset[miss] - go[miss]
                stim_rel[miss, 1] = stim_rel[miss, 0] + ps_dur[miss]
            for k, ti in enumerate(trials):
                if np.isfinite(stim_rel[ti, 0]):
                    stim_input[k] = ((BIN_CENTERS_REL >= stim_rel[ti, 0]) &
                                     (BIN_CENTERS_REL <= stim_rel[ti, 1])).astype(np.float32)
            info['timing']['input'] = time.time() - t1

            # --------------------------------------------------- outputs
            t1 = time.time()
            # choice: direction of the first lick after the go cue (within the trial)
            choice = np.full(ntrials_all, 2, dtype=np.int64)    # 2 = no lick
            for ti in trials:
                lo, hi = go[ti], stop_time[ti]
                l0 = left_lick[np.searchsorted(left_lick, lo)] if np.searchsorted(left_lick, lo) < len(left_lick) else np.inf
                r0 = right_lick[np.searchsorted(right_lick, lo)] if np.searchsorted(right_lick, lo) < len(right_lick) else np.inf
                l0 = l0 if l0 <= hi else np.inf
                r0 = r0 if r0 <= hi else np.inf
                if np.isinf(l0) and np.isinf(r0):
                    choice[ti] = 2
                elif l0 <= r0:
                    choice[ti] = 0
                else:
                    choice[ti] = 1
            # sanity: agreement with outcome x instruction
            expected = np.where(outcome == 'hit', instruction,
                                np.where(instruction == 'left', 'right', 'left'))
            expected = np.where(outcome == 'ignore', 'none', expected)
            ch_str = np.array(['left', 'right', 'none'])[choice]
            info['choice_agreement'] = float(np.mean(ch_str[trials] == expected[trials]))

            outcome_code = np.select([outcome == 'ignore', outcome == 'miss', outcome == 'hit'],
                                     [0, 1, 2], default=0).astype(np.int64)
            early_code = (early_lick == 'early').astype(np.int64)

            # tongue y position, discretized per session
            tongue_class = np.full((nkept, NBINS), 3, dtype=np.int64)
            for k, ti in enumerate(trials):
                i0, i1 = np.searchsorted(vts, [go[ti] + T_START, go[ti] + T_END])
                if i1 <= i0:
                    continue
                ft = vts[i0:i1] - go[ti]
                fy = vy[i0:i1]
                fvis = visible_all[i0:i1]
                if not fvis.any():
                    continue
                b = np.clip(((ft - T_START) / BIN_SIZE).astype(int), 0, NBINS - 1)
                cnt = np.bincount(b[fvis], minlength=NBINS)
                ysum = np.bincount(b[fvis], weights=fy[fvis], minlength=NBINS)
                has = cnt > 0
                ymean = np.zeros(NBINS)
                ymean[has] = ysum[has] / cnt[has]
                cls = np.full(NBINS, 3, dtype=np.int64)
                cls[has] = np.digitize(ymean[has], [y_p40, y_p60])   # 0,1,2
                tongue_class[k] = cls
            info['timing']['output'] = time.time() - t1

            # --------------------------------------------------- assemble
            neural_list, input_list, output_list = [], [], []
            for k, ti in enumerate(trials):
                neural_list.append(rates[:, k, :])
                input_list.append(np.stack([time_from_tone[k].astype(np.float32),
                                            stim_input[k]]).astype(np.float32))
                out = np.empty((4, NBINS), dtype=np.int64)
                out[0] = choice[ti]
                out[1] = outcome_code[ti]
                out[2] = early_code[ti]
                out[3] = tongue_class[k]
                output_list.append(out)

            session = {
                'file': os.path.basename(fname),
                'subject': os.path.basename(fname).split('_')[0].replace('sub-', ''),
                'session_id': os.path.basename(fname).split('_')[1].replace('ses-', ''),
                'neural': neural_list,
                'input': input_list,
                'output': output_list,
                'regions': regions,
                'anno': anno,
                'performance': performance,
                'n_trials_all': ntrials_all,
                'trial_indices': trials,
                'unit_indices': good,
                'tongue_pctl': (float(y_p40), float(y_p60)),
                'tone_rel': tone_rel[trials],
                'stim_rel': stim_rel[trials],
                'mean_rate': float(rates.mean()),
            }
            info['mean_rate_hz'] = session['mean_rate']
            info['timing']['total'] = time.time() - t0

            if show_processing:
                try:
                    make_processing_plot(session, f, trials, go, rates, vts, vy, visible_all,
                                         y_p40, y_p60, tone_rel, stim_rel, choice, outcome,
                                         early_lick, tongue_class)
                except Exception:
                    traceback.print_exc()

        return session, info
    except Exception as e:
        info['error'] = '%s: %s' % (type(e).__name__, e)
        info['traceback'] = traceback.format_exc()
        return None, info


# ------------------------------------------------------------------ diagnostics plot
def make_processing_plot(session, f, trials, go, rates, vts, vy, visible_all,
                         y_p40, y_p60, tone_rel, stim_rel, choice, outcome,
                         early_lick, tongue_class):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sid = session['file'].replace('.nwb', '')
    u = f['units']
    st_index = u['spike_times_index'][:]
    st_all = u['spike_times'][:]
    starts = np.concatenate([[0], st_index[:-1]])
    classification = decode_array(u['classification'][:])
    good_idx = np.where(classification == 'good')[0]

    # pick an example trial with photostim if possible, otherwise a hit trial
    ex = None
    for k, ti in enumerate(trials):
        if np.isfinite(stim_rel[ti, 0]) and outcome[ti] == 'hit':
            ex = k
            break
    if ex is None:
        ex = int(np.argmax(np.array([outcome[ti] == 'hit' for ti in trials])))
    ti = trials[ex]

    fig, axes = plt.subplots(6, 1, figsize=(11, 15), sharex=True)

    # 1. raw spike raster of 40 units, aligned to go cue
    ax = axes[0]
    show_units = good_idx[:40]
    for j, ui in enumerate(show_units):
        sp = st_all[starts[ui]:st_index[ui]] - go[ti]
        sp = sp[(sp >= T_START) & (sp <= T_END)]
        ax.plot(sp, np.full(len(sp), j), '|', color='k', markersize=3)
    ax.set_ylabel('unit (raw spikes)')
    ax.set_title('%s - trial %d (outcome=%s, early=%s, choice=%d)' %
                 (sid, ti, outcome[ti], early_lick[ti], choice[ti]))

    # 2. binned firing rates (all units)
    ax = axes[1]
    im = ax.imshow(rates[:, ex, :], aspect='auto', origin='lower',
                   extent=[T_START, T_END, 0, rates.shape[0]], cmap='magma')
    ax.set_ylabel('neuron (binned Hz)')
    plt.colorbar(im, ax=ax, fraction=0.03, label='Hz')

    # 3. binned rate of the 40 plotted units vs raster (alignment check)
    ax = axes[2]
    sel = np.arange(min(40, rates.shape[0]))
    ax.plot(BIN_CENTERS_REL, rates[sel, ex, :].mean(axis=0), '-o', ms=3,
            label='mean rate of first 40 good units')
    ax.axvline(0, color='r', label='go cue')
    ax.axvline(tone_rel[ti], color='g', label='tone onset')
    ax.set_ylabel('Hz')
    ax.legend(fontsize=7)

    # 4. inputs
    ax = axes[3]
    inp = session['input'][ex]
    ax.plot(BIN_CENTERS_REL, inp[0], label='time from tone onset (s)')
    ax.plot(BIN_CENTERS_REL, inp[1], label='photostim on')
    ax.axvline(0, color='r')
    ax.axvline(tone_rel[ti], color='g')
    if np.isfinite(stim_rel[ti, 0]):
        ax.axvspan(stim_rel[ti, 0], stim_rel[ti, 1], color='b', alpha=0.2, label='photostim (raw)')
    ax.set_ylabel('inputs')
    ax.legend(fontsize=7)

    # 5. tongue trace and discretization
    ax = axes[4]
    i0, i1 = np.searchsorted(vts, [go[ti] + T_START, go[ti] + T_END])
    ft = vts[i0:i1] - go[ti]
    fy = vy[i0:i1].copy()
    fv = visible_all[i0:i1]
    ax.plot(ft[fv], fy[fv], '.', ms=2, label='tongue y (visible)')
    ax.axhline(y_p40, color='orange', ls='--', label='session p40')
    ax.axhline(y_p60, color='purple', ls='--', label='session p60')
    ax.axvline(0, color='r')
    ax.set_ylabel('tongue y (px)')
    ax.legend(fontsize=7)

    # 6. outputs
    ax = axes[5]
    out = session['output'][ex]
    for i, nm in enumerate(['choice', 'outcome', 'early_lick', 'tongue_y']):
        ax.step(BIN_CENTERS_REL, out[i] + 0.05 * i, where='mid', label=nm)
    ax.axvline(0, color='r')
    ax.set_ylabel('outputs (class)')
    ax.set_xlabel('time from go cue (s)')
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig('processing_%s.png' % sid, dpi=110)
    plt.close(fig)

    # second figure: session-level summaries
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.hist(vy[visible_all], bins=100)
    ax.axvline(y_p40, color='orange'); ax.axvline(y_p60, color='purple')
    ax.set_title('session tongue y (visible frames)'); ax.set_xlabel('y (px)')

    ax = axes[0, 1]
    frac = np.stack([(tongue_class == c).mean(axis=0) for c in range(4)])
    for c in range(4):
        ax.plot(BIN_CENTERS_REL, frac[c], label='tongue class %d' % c)
    ax.axvline(0, color='r'); ax.legend(fontsize=7)
    ax.set_title('tongue class fraction over time'); ax.set_xlabel('time from go (s)')

    ax = axes[1, 0]
    ax.plot(BIN_CENTERS_REL, rates.mean(axis=(0, 1)))
    ax.axvline(0, color='r')
    ax.set_title('population mean firing rate (Hz)'); ax.set_xlabel('time from go (s)')

    ax = axes[1, 1]
    stim_any = np.array([np.isfinite(stim_rel[t, 0]) for t in trials])
    ax.plot(BIN_CENTERS_REL, np.stack([session['input'][k][1] for k in range(len(trials))]).mean(axis=0))
    ax.set_title('fraction of trials with photostim on (n stim trials = %d)' % stim_any.sum())
    ax.set_xlabel('time from go (s)')
    fig.tight_layout()
    fig.savefig('processing_%s_summary.png' % sid, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile', type=str)
    ap.add_argument('--full', action='store_true', help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='save diagnostic plots for up to 2 sessions')
    ap.add_argument('--nproc', type=int, default=16)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
    print('found %d NWB files' % len(files))
    if args.sample:
        files = [f for f in files if os.path.basename(f).startswith(('sub-440956_ses-20190210',
                                                                     'sub-484677_ses-20210420'))]
        print('sample mode: %d sessions' % len(files))

    nplot = 2 if args.show_processing else 0
    tasks = [(fn, i < nplot) for i, fn in enumerate(files)]

    t0 = time.time()
    results = []
    if len(tasks) == 1 or args.nproc <= 1:
        for t in tasks:
            results.append(process_session(t))
    else:
        with Pool(min(args.nproc, len(tasks))) as pool:
            for i, res in enumerate(pool.imap(process_session, tasks)):
                results.append(res)
                sess, info = res
                msg = 'rejected: %s' % info.get('reject') if sess is None else \
                      'kept %d trials x %d neurons (%.1f s)' % (len(sess['neural']),
                                                                sess['neural'][0].shape[0],
                                                                info['timing']['total'])
                if 'error' in info:
                    msg = 'ERROR %s' % info['error']
                print('[%3d/%3d] %s: %s' % (i + 1, len(tasks), info['file'], msg), flush=True)
    print('session processing took %.1f s' % (time.time() - t0))

    sessions = [s for s, _ in results if s is not None]
    infos = [i for _, i in results]
    errors = [i for i in infos if 'error' in i]
    if errors:
        print('!! %d sessions raised errors' % len(errors))
        for e in errors:
            print(e['file'], e['error'])
            print(e['traceback'])

    if not sessions:
        print('no sessions survived curation - aborting')
        sys.exit(1)

    # ---------------------------------------------------- global bookkeeping
    subjects = sorted({s['subject'] for s in sessions})
    subject_idx = np.array([subjects.index(s['subject']) for s in sessions], dtype=np.int64)
    all_regions = sorted({str(r) for s in sessions for r in s['regions']})
    region_lookup = {r: i for i, r in enumerate(all_regions)}
    brain_region_idx = [np.array([region_lookup[str(r)] for r in s['regions']], dtype=np.int64)
                        for s in sessions]

    data = {
        'neural': [s['neural'] for s in sessions],
        'input': [s['input'] for s in sessions],
        'output': [s['output'] for s in sessions],
        'subjects': subjects,
        'subject_idx': subject_idx,
        'brain_regions': all_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_tone_onset', 'photostim_on'],
        'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y'],
        'output_values': [
            ['left', 'right', 'no lick'],
            ['ignore', 'miss', 'hit'],
            ['no', 'yes'],
            ['<40th pctile', '40th-60th pctile', '>60th pctile', 'not visible'],
        ],
        'metadata': {
            'task_description': (
                'Mice performed an auditory delayed-response (memory-guided directional licking) task '
                '(Chen et al. 2024, DANDI 000363). A 3 kHz or 12 kHz tone during the sample epoch '
                'instructed a lick to the right or left lick port after a 1.2 s delay and an auditory go cue. '
                'Decoder outputs: lick direction choice (left / right / no lick), trial outcome '
                '(ignore / miss / hit), early lick (no / yes) and the discretized vertical position of the '
                'tongue tracked with DeepLabCut in the side-view video.'),
            'time_bin_size': BIN_SIZE * 1000.0,
            'temporal_alignment_event': ALIGN_EVENT,
            'off_start': T_START,
            'off_end': T_END,
            'neural_units': 'firing rate in Hz (spike count per 50 ms bin / 0.05 s)',
            'n_time_bins': NBINS,
            'bin_centers_s': BIN_CENTERS_REL.tolist(),
            'session_info': [
                {'file': s['file'], 'subject': s['subject'], 'session_id': s['session_id'],
                 'n_trials': len(s['neural']), 'n_neurons': int(s['neural'][0].shape[0]),
                 'performance': s['performance'], 'n_trials_in_file': s['n_trials_all'],
                 'tongue_y_percentiles': s['tongue_pctl'],
                 'trial_indices': [int(x) for x in s['trial_indices']],
                 'unit_indices': [int(x) for x in s['unit_indices']],
                 'mean_firing_rate_hz': s['mean_rate']}
                for s in sessions],
            'curation': {
                'units': "units/classification == 'good' (white-paper QC classifier); "
                         'units flagged bad by units/is_good_trials on a retained trial are dropped',
                'sessions': 'behavioural performance > 65%% and >= 50 correct lick-left and lick-right '
                            'trials (data paper criteria); at least one QC-good unit; tongue tracking present',
                'trials': 'auto-water and free-water trials removed; trials must have video frames in every '
                          'bin of the [-2.5, 1.5] s window; trials flagged unstable for >10%% of units removed. '
                          'Early-lick, ignore/miss and photostimulation trials are KEPT because they are decoder '
                          'outputs/inputs (this deviates from the reference analyses, which excluded them).',
            },
            'source': 'DANDI 000363 (Chen et al., Cell 2024); processing follows Wang*, Kurgyis* et al., '
                      'Nat Neurosci 2025 (github druckmann-lab/MapVideoAnalysis)',
        },
    }

    # ---------------------------------------------------- summary
    ntr = np.array([len(s['neural']) for s in sessions])
    nne = np.array([s['neural'][0].shape[0] for s in sessions])
    perf = np.array([s['performance'] for s in sessions])
    agree = np.array([i['choice_agreement'] for i in infos if 'choice_agreement' in i])
    print('\n==== converted dataset ====')
    print('sessions %d, subjects %d' % (len(sessions), len(subjects)))
    print('trials: total %d, mean/session %.1f (range %d-%d)' % (ntr.sum(), ntr.mean(), ntr.min(), ntr.max()))
    print('neurons: total %d, mean/session %.1f (range %d-%d)' % (nne.sum(), nne.mean(), nne.min(), nne.max()))
    print('behavioural performance: mean %.3f (range %.3f-%.3f)' % (perf.mean(), perf.min(), perf.max()))
    print('choice/outcome agreement: mean %.4f min %.4f' % (agree.mean(), agree.min()))
    print('brain regions (%d): %s' % (len(all_regions), all_regions))
    rc = Counter(r for s in sessions for r in s['regions'])
    print('neurons per region:', dict(sorted(rc.items(), key=lambda kv: -kv[1])))
    outs = np.concatenate([o for s in sessions for o in s['output']], axis=1)
    for i, nm in enumerate(data['output_names']):
        vals, cnts = np.unique(outs[i], return_counts=True)
        print('output %s distribution:' % nm,
              {int(v): round(float(c) / outs.shape[1], 4) for v, c in zip(vals, cnts)})
    ins = np.concatenate([np.atleast_2d(x) for s in sessions for x in s['input']], axis=1)
    for i, nm in enumerate(data['input_names']):
        print('input %s range: [%.3f, %.3f] mean %.3f' % (nm, ins[i].min(), ins[i].max(), ins[i].mean()))
    allrates = np.array([s['mean_rate'] for s in sessions])
    print('mean firing rate across sessions: %.2f Hz (range %.2f-%.2f)' %
          (allrates.mean(), allrates.min(), allrates.max()))

    t1 = time.time()
    with open(args.outfile, 'wb') as fh:
        pickle.dump(data, fh, protocol=4)
    print('wrote %s (%.2f GB) in %.1f s' % (args.outfile,
                                            os.path.getsize(args.outfile) / 1e9, time.time() - t1))
    with open(args.outfile + '.info.json', 'w') as fh:
        json.dump([{k: v for k, v in i.items() if k != 'traceback'} for i in infos], fh, indent=1)
    print('total time %.1f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
