"""
Convert the Sosa, Plitt & Giocomo (2025) hippocampal CA1 2P + VR dataset
(DANDI:001361, NWB) into the decoder-ready pickle format.

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Options:
    --full              process all sessions (default)
    --sample            process only 2 sessions (for testing)
    --show-processing   save per-step diagnostic plots for up to 2 sessions
                        as processing_<session_id>.png

Processing follows the reference repository (`/app/code/src/reward_relative`) and
the paper methods (`/app/methods.txt`):

  * neural signal  : raw suite2p F/Fneu  ->  dF/F (per-trial maximin baseline,
                     neuropil subtraction with coefficient 0.7, sigma=2-sample
                     Gaussian smoothing), exactly as
                     `reward_relative.preprocessing.dff(...,
                     neuropil_method='subtract', baseline_method='maximin')` is
                     called from `reward_relative.utilities.multi_anim_sess`.
                     dF/F is the default neural signal written to the pickle;
                     `--neural-signal events` instead writes the OASIS
                     deconvolution of that dF/F (the reference `deconvolve=True`
                     branch, used for the paper's own decoder).
  * neuron curation: suite2p `iscell` (manual curation, stored in the NWB
                     PlaneSegmentation) then putative-interneuron exclusion
                     (Pearson r(dF/F, speed) > 0.5), as in
                     `reward_relative.spatial.is_putative_interneuron`.
  * trials         : [trial_start_ind, teleport_ind)  (the teleport sample is
                     excluded; its interpolated position is meaningless).
  * trial curation : trials with lick-sensor error (>30% of frames with a
                     cumulative lick count > 2) are dropped
                     (`reward_relative.behavior.correct_lick_sensor_error`).
  * reward zones   : from the scene name in the NWB identifier, switching at
                     trial 30 (`reward_relative.behavior.get_reward_zones`).
"""

import argparse
import glob
import multiprocessing
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage

from pynwb import NWBHDF5IO
from suite2p.extraction import dcnv

# --------------------------------------------------------------------------
# Constants taken from the reference code / paper methods
# --------------------------------------------------------------------------

# reward_relative.behavior.reward_zone_dict; get_reward_zones maps scene
# 'Location A/B/C' -> dict keys 'X'/'Y'/'Z' (paper: zone A 80-130, B 200-250,
# C 320-370 cm)
REWARD_ZONE_CM = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
RZ_LABELS = ['A', 'B', 'C']
CHANGE_TRIAL = 30           # reward zone switches after 30 trials on switch days

NEU_COEF = 0.7              # neuropil coefficient (default_dff_method)
MAXIMIN_WIN = 300           # samples (~20 s at 15.5 Hz), from preprocessing.dff
BASELINE_SMOOTH_SIG = 15    # samples, from preprocessing.dff  (nansmooth [0,15])
DFF_SMOOTH_SIG = 2          # samples (~0.129 s), from preprocessing.dff
TAU = 0.7                   # suite2p ops['tau'] used by the authors
OASIS_BATCH = 2000
INT_R_THRESH = 0.5          # dayData.int_thresh, putative interneuron cutoff
LICK_ERROR_THRESH = 0.3     # paper: ">30% of samples with cumulative lick count >2"

TRACK_LENGTH = 450.0

# reward_relative.teleport_metadata.teleport_sessions: experiment days on which
# the laser was NOT blanked during the inter-trial interval, so the ITI
# fluorescence is usable for the per-trial dF/F baseline.
TELEPORT_SESSIONS = {
    'm10': [1, 7, 8, 14, 15],
    'm11': [1, 7, 8, 14, 15],
    'm12': [1, 7, 8, 14, 15],
    'm13': [1, 7, 8, 14, 15],
    'm14': [1, 7, 8, 14, 15],
    'm15': [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17],
    'm17': [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17],
    'm18': [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17],
    'm19': [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17],
}

INPUT_NAMES = ['time_from_trial_start_s', 'environment', 'trial_number',
               'previous_trial_outcome']
OUTPUT_NAMES = ['distance_to_reward_zone', 'position', 'speed', 'lick',
                'reward_zone_location', 'reward_outcome']
OUTPUT_VALUES = [
    ['< -50 cm', '-50 to -10 cm', '-10 to 0 cm', 'in reward zone (0 cm)',
     '0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
    ['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm'],
    ['< 2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '> 40 cm/s'],
    ['no lick', 'lick'],
    ['zone A (80-130 cm)', 'zone B (200-250 cm)', 'zone C (320-370 cm)'],
    ['omitted', 'rewarded'],
]


# --------------------------------------------------------------------------
# Helpers copied / adapted from the reference repo
# --------------------------------------------------------------------------

def nansmooth(a, sig, axis=-1):
    """Gaussian smoothing that does not propagate NaNs.

    Port of ``TwoPUtils.utilities.nansmooth`` / ``reward_relative.utilities.nansmooth``
    (TwoPUtils is not installed here). A list ``sig`` applies an N-d Gaussian
    filter (as the reference does with ``[0, 15]``); a scalar ``sig`` applies a
    1-d filter along ``axis``.
    """
    nan_inds = np.isnan(a)
    a_nanless = np.array(a, dtype=np.float64, copy=True)
    a_nanless[nan_inds] = 0
    one = np.ones(a.shape)
    one[nan_inds] = 0.001
    if np.isscalar(sig):
        a_nanless = ndimage.gaussian_filter1d(a_nanless, sig, axis=axis)
        one = ndimage.gaussian_filter1d(one, sig, axis=axis)
    else:
        a_nanless = ndimage.gaussian_filter(a_nanless, sig)
        one = ndimage.gaussian_filter(one, sig)
    return a_nanless / one


def scene_reward_zones(scene, n_trials):
    """Reward-zone [start, stop] coordinates and labels per trial, from the scene name.

    Port of ``reward_relative.behavior.get_reward_zones`` restricted to the
    scenes that occur in this dataset (Env{1,2}_Location{A,B,C} and the
    within-session switch scenes ``*_X_to_*_Y``). The zone changes after
    ``CHANGE_TRIAL`` (=30) trials on switch days.
    """
    def _zone_of(label):
        return REWARD_ZONE_CM[label]

    # Scenes are like 'Env1_LocationA', 'Env1_LocationA_to_C', 'Env1_A_to_Env2_B'
    if '_to_' in scene:
        # first zone: last letter of the token before '_to_'
        before, after = scene.split('_to_')
        first = before[-1]
        second = after[-1]
        assert first in REWARD_ZONE_CM and second in REWARD_ZONE_CM, scene
        labels = np.array([first] * min(CHANGE_TRIAL, n_trials) +
                          [second] * max(0, n_trials - CHANGE_TRIAL))
    else:
        lab = scene[-1]
        assert lab in REWARD_ZONE_CM, scene
        labels = np.array([lab] * n_trials)
    coords = np.array([_zone_of(l) for l in labels], dtype=np.float64)
    return coords, labels


def compute_dff_events(F, Fneu, windows, fs, deconvolve=True):
    """dF/F and OASIS-deconvolved 'events', following ``preprocessing.dff``.

    Args:
        F: (ncells, nframes) raw suite2p fluorescence
        Fneu: (ncells, nframes) neuropil fluorescence
        windows: list of (start, stop) sample indices defining the periods over
            which dF/F is computed (one per trial; includes the preceding ITI
            when the laser was not blanked)
        fs: imaging rate per plane, Hz
        deconvolve: whether to also run the OASIS deconvolution

    Returns:
        dff, events: (ncells, nframes) float32 arrays, NaN outside `windows`
        (events is None when deconvolve is False)
    """
    f_ = np.full(F.shape, np.nan, dtype=np.float32)
    fneu_ = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in windows:
        f_[:, s:e] = F[:, s:e]
        fneu_[:, s:e] = Fneu[:, s:e]

    nanmask = ~np.isnan(f_[0, :])

    # neuropil subtraction
    f_ -= NEU_COEF * fneu_

    flow = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in windows:
        # add back the neuropil mean within the window so dF/F values stay close
        # to true dF/F (reference comment)
        f_[:, s:e] += NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
        # maximin baseline: smooth, then 20 s minimum filter, then 20 s maximum filter
        x = nansmooth(f_[:, s:e], [0, BASELINE_SMOOTH_SIG])
        x = ndimage.minimum_filter1d(x, MAXIMIN_WIN, axis=-1)
        flow[:, s:e] = ndimage.maximum_filter1d(x, MAXIMIN_WIN, axis=-1)

    dff = np.full(F.shape, np.nan, dtype=np.float32)
    dff[:, nanmask] = ((f_[:, nanmask] - flow[:, nanmask])
                       / np.abs(flow[:, nanmask]))

    events = np.full(F.shape, np.nan, dtype=np.float32) if deconvolve else None
    for s, e in windows:
        dff[:, s:e] = nansmooth(dff[:, s:e], DFF_SMOOTH_SIG, axis=1)
        if deconvolve:
            events[:, s:e] = dcnv.oasis(np.ascontiguousarray(dff[:, s:e]),
                                        OASIS_BATCH, TAU, fs)
    return dff, events, nanmask


def speed_correlation(dff, speed, nanmask):
    """Pearson r between each cell's dF/F and running speed over valid samples.

    Vectorised equivalent of ``reward_relative.spatial.is_putative_interneuron``
    with ``method='speed'``.
    """
    D = dff[:, nanmask].astype(np.float64)
    S = np.asarray(speed, dtype=np.float64)[nanmask]
    Dm = D - D.mean(axis=1, keepdims=True)
    Sm = S - S.mean()
    denom = np.sqrt((Dm ** 2).sum(axis=1)) * np.sqrt((Sm ** 2).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (Dm @ Sm) / denom
    return r


# --------------------------------------------------------------------------
# Discretisation of the decoder outputs
# --------------------------------------------------------------------------

def discretize_reward_distance(d):
    """Signed distance (cm) to the nearest point of the reward zone -> 7 bins."""
    out = np.empty(d.shape, dtype=np.int64)
    out[:] = 3                                  # d == 0: inside the reward zone
    out[(d >= -10) & (d < 0)] = 2
    out[(d >= -50) & (d < -10)] = 1
    out[d < -50] = 0
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    return out


def discretize_position(pos):
    """Absolute track position (cm) -> 5 equal bins spanning the 450 cm track."""
    return np.clip((pos / (TRACK_LENGTH / 5)).astype(np.int64), 0, 4)


def discretize_speed(speed):
    """Running speed (cm/s) -> 5 bins."""
    return np.digitize(speed, [2.0, 10.0, 20.0, 40.0]).astype(np.int64)


# --------------------------------------------------------------------------
# Per-session conversion
# --------------------------------------------------------------------------

def load_session(path):
    """Read everything needed for one session out of the NWB file with pynwb."""
    io = NWBHDF5IO(path, 'r', load_namespaces=True)
    nwb = io.read()

    subject = nwb.subject.subject_id
    exp_day = int(nwb.session_id)
    scene = nwb.identifier.rstrip('/').split('/')[-1]
    date = nwb.identifier.rstrip('/').split('/')[-2]

    beh = nwb.processing['behavior']['BehavioralTimeSeries']
    get = lambda k: np.asarray(beh[k].data[:], dtype=np.float64)
    frame_times = np.asarray(beh['position'].timestamps[:], dtype=np.float64)
    nframes = len(frame_times)

    behavior = dict(
        time=frame_times,
        pos=get('position'),
        speed=get('speed'),
        lick=get('lick'),
        rzone=get('reward_zone'),
        trial_start=get('trial_start'),
        teleport=get('teleport'),
        trialnum=get('trial number'),
        morph=get('environment'),
        autoreward=get('autoreward'),
    )
    # 'Reward' is a sparse series: one timestamp per delivered reward. Convert to
    # a per-frame binary, equivalent to the vr_data['reward'] column.
    reward_times = np.asarray(beh['Reward'].timestamps[:], dtype=np.float64)
    reward_frames = np.searchsorted(frame_times, reward_times)
    reward_frames = np.clip(reward_frames, 0, nframes - 1)
    reward = np.zeros(nframes)
    np.add.at(reward, reward_frames, 1.0)
    behavior['reward'] = reward

    # --- ophys: raw F and neuropil for suite2p-curated cells, all planes pooled
    ophys = nwb.processing['ophys']
    seg = ophys['ImageSegmentation']['PlaneSegmentation']
    iscell = np.asarray(seg['iscell'].data)[:, 0] > 0
    plane_idx = np.asarray(seg['planeIdx'].data).astype(int)

    F_list, Fneu_list, plane_list = [], [], []
    rate = None
    for key in sorted(ophys['Fluorescence'].roi_response_series.keys()):
        p = int(key.replace('plane', ''))
        mask = iscell[plane_idx == p]
        rrs = ophys['Fluorescence'][key]
        rate = rrs.rate
        # NWB stores (nframes, nroi); the reference works with (ncells, nframes).
        # Some sessions have exactly one more imaging frame than behaviour rows
        # (the "one frame correction" in vr_align_to_2P) -> truncate.
        F_list.append(np.asarray(rrs.data[:nframes, :])[:, mask].T.astype(np.float32))
        Fneu_list.append(np.asarray(ophys['Neuropil'][key].data[:nframes, :])[:, mask].T
                         .astype(np.float32))
        plane_list.append(np.full(int(mask.sum()), p, dtype=int))

    F = np.concatenate(F_list, axis=0)
    Fneu = np.concatenate(Fneu_list, axis=0)
    planes = np.concatenate(plane_list, axis=0)
    n_planes = len(F_list)
    io.close()

    return dict(path=path, subject=subject, exp_day=exp_day, scene=scene, date=date,
                behavior=behavior, F=F, Fneu=Fneu, planes=planes,
                n_planes=n_planes, rate=rate, nframes=nframes)


def convert_session(path, show_processing=False, plot_dir='/app', neural_signal='dff'):
    """Convert one NWB session into per-trial neural / input / output arrays."""
    t0 = time.time()
    sess = load_session(path)
    t_load = time.time() - t0

    beh = sess['behavior']
    nframes = sess['nframes']
    fs = sess['rate'] / sess['n_planes']          # per-plane imaging rate (Hz)

    tstart_inds = np.where(beh['trial_start'] > 0)[0]
    teleport_inds = np.where(beh['teleport'] > 0)[0]
    assert len(tstart_inds) == len(teleport_inds), \
        f"{path}: {len(tstart_inds)} trial starts vs {len(teleport_inds)} teleports"
    assert np.all(teleport_inds > tstart_inds), f"{path}: teleport before trial start"
    n_trials_raw = len(tstart_inds)

    # --- dF/F windows (reference `dff`): per trial, optionally including the
    # preceding ITI on sessions where the laser was not blanked.
    keep_teleports = sess['exp_day'] in TELEPORT_SESSIONS.get(sess['subject'], [])
    if keep_teleports:
        windows = [(int(tstart_inds[0]), int(teleport_inds[0]))]
        for i in range(1, n_trials_raw):
            # start 1 sample after the previous teleport (the teleport sample
            # itself is excluded, as in the reference)
            windows.append((int(min(teleport_inds[i - 1] + 1, tstart_inds[i])),
                            int(teleport_inds[i])))
    else:
        windows = [(int(s), int(e)) for s, e in zip(tstart_inds, teleport_inds)]

    t1 = time.time()
    dff, events, nanmask = compute_dff_events(sess['F'], sess['Fneu'], windows, fs,
                                              deconvolve=(neural_signal == 'events'
                                                          or show_processing))
    t_dff = time.time() - t1

    # --- neuron curation: putative interneurons (speed-correlated) ------------
    r_speed = speed_correlation(dff, beh['speed'], nanmask)
    is_int = np.nan_to_num(r_speed, nan=0.0) > INT_R_THRESH
    keep_cells = ~is_int
    n_cells_iscell = len(keep_cells)
    neural_full = (events if neural_signal == 'events' else dff)[keep_cells]
    planes = sess['planes'][keep_cells]

    # --- per-trial behaviour --------------------------------------------------
    rz_coords, rz_labels = scene_reward_zones(sess['scene'], n_trials_raw)

    # reward_relative.behavior.get_trial_types: rewarded iff reward delivered AND
    # the reward-zone flag was set within the trial
    isreward = np.zeros(n_trials_raw, dtype=np.int64)
    morph = np.zeros(n_trials_raw, dtype=np.int64)
    trialnum = np.zeros(n_trials_raw, dtype=np.int64)
    lick_error = np.zeros(n_trials_raw, dtype=bool)
    for i, (s, e) in enumerate(zip(tstart_inds, teleport_inds)):
        isreward[i] = int(np.any(beh['reward'][s:e] > 0)
                          and np.any(beh['rzone'][s:e] > 0))
        m = np.unique(beh['morph'][s:e])
        morph[i] = int(np.round(m[0]))
        tn = np.unique(beh['trialnum'][s:e])
        trialnum[i] = int(tn[0])
        L = beh['lick'][s:e]
        lick_error[i] = np.mean(L > 2) > LICK_ERROR_THRESH

    # previous trial outcome (0 = omitted, 1 = rewarded); undefined (=0) for the
    # first trial of a session
    prev_outcome = np.concatenate([[0], isreward[:-1]]).astype(np.int64)

    # --- build per-trial arrays ----------------------------------------------
    neural_trials, input_trials, output_trials = [], [], []
    kept_trial_idx = []
    n_dropped_lick, n_dropped_nan = 0, 0
    for i in range(n_trials_raw):
        if lick_error[i]:
            n_dropped_lick += 1
            continue
        s, e = int(tstart_inds[i]), int(teleport_inds[i])
        pos = beh['pos'][s:e]
        speed = beh['speed'][s:e]
        lick = beh['lick'][s:e]
        t = beh['time'][s:e] - beh['time'][s]
        neural = neural_full[:, s:e]

        if (not np.all(np.isfinite(neural)) or not np.all(np.isfinite(pos))
                or not np.all(np.isfinite(speed)) or not np.all(np.isfinite(lick))):
            n_dropped_nan += 1
            continue

        rz_start, rz_end = rz_coords[i]
        d = np.zeros_like(pos)
        d[pos < rz_start] = pos[pos < rz_start] - rz_start
        d[pos > rz_end] = pos[pos > rz_end] - rz_end

        out = np.stack([
            discretize_reward_distance(d),
            discretize_position(pos),
            discretize_speed(speed),
            (lick > 0).astype(np.int64),
            np.full(len(pos), RZ_LABELS.index(rz_labels[i]), dtype=np.int64),
            np.full(len(pos), isreward[i], dtype=np.int64),
        ], axis=0)

        inp = np.stack([
            t,
            np.full(len(pos), float(morph[i])),
            np.full(len(pos), float(trialnum[i])),
            np.full(len(pos), float(prev_outcome[i])),
        ], axis=0).astype(np.float32)

        neural_trials.append(np.ascontiguousarray(neural, dtype=np.float32))
        input_trials.append(inp)
        output_trials.append(out)
        kept_trial_idx.append(i)

    info = dict(
        path=path, subject=sess['subject'], exp_day=sess['exp_day'],
        scene=sess['scene'], date=sess['date'],
        session_id=f"{sess['subject']}_day{sess['exp_day']:02d}",
        n_trials_raw=n_trials_raw, n_trials_kept=len(neural_trials),
        n_dropped_lick_error=n_dropped_lick, n_dropped_nan=n_dropped_nan,
        n_rois_iscell=n_cells_iscell, n_interneurons=int(is_int.sum()),
        n_cells=int(keep_cells.sum()), n_planes=sess['n_planes'],
        keep_teleports=bool(keep_teleports), fs=float(fs),
        frac_rewarded=float(np.mean(isreward)),
        rz_labels=list(rz_labels), morph=morph.tolist(),
        t_load=t_load, t_dff=t_dff,
    )

    if show_processing:
        plot_processing(sess, beh, tstart_inds, teleport_inds, dff, events,
                        r_speed, keep_cells, rz_coords, rz_labels, isreward,
                        kept_trial_idx, neural_trials, input_trials,
                        output_trials, info, plot_dir)

    info['t_total'] = time.time() - t0
    return neural_trials, input_trials, output_trials, planes, info


# --------------------------------------------------------------------------
# Diagnostic plots
# --------------------------------------------------------------------------

def plot_processing(sess, beh, tstart_inds, teleport_inds, dff, events,
                    r_speed, keep_cells, rz_coords, rz_labels, isreward,
                    kept_trial_idx, neural_trials, input_trials, output_trials,
                    info, plot_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sid = info['session_id']
    fig, axes = plt.subplots(8, 1, figsize=(18, 26))

    # window of ~6 trials to display
    t0i, t1i = int(tstart_inds[0]), int(teleport_inds[min(5, len(teleport_inds) - 1)])
    sl = slice(t0i, t1i)
    tt = beh['time'][sl]

    ax = axes[0]
    cells = np.argsort(-np.nanmax(sess['F'], axis=1))[:3]
    for c in cells:
        ax.plot(tt, sess['F'][c, sl], lw=0.8, label=f'F cell {c}')
        ax.plot(tt, sess['Fneu'][c, sl], lw=0.6, alpha=0.6, label=f'Fneu cell {c}')
    ax.set_title(f'{sid}: step 1 - raw suite2p fluorescence (first ~6 trials)')
    ax.set_ylabel('F (a.u.)')
    ax.legend(fontsize=6, ncol=3)

    ax = axes[1]
    for c in cells:
        ax.plot(tt, dff[c, sl], lw=0.8, label=f'dF/F cell {c}')
    for s, e in zip(tstart_inds[:6], teleport_inds[:6]):
        ax.axvspan(beh['time'][s], beh['time'][e], color='k', alpha=0.05)
    ax.set_title('step 2 - dF/F (per-trial maximin baseline, neuropil subtracted, '
                 'sigma=2 smoothing). Shading = trials; gaps = inter-trial intervals (NaN)')
    ax.set_ylabel('dF/F')
    ax.legend(fontsize=6, ncol=3)

    ax = axes[2]
    for c in cells:
        ax.plot(tt, events[c, sl], lw=0.8, label=f'events cell {c}')
    ax.set_title('step 3 - OASIS-deconvolved activity ("events"); dF/F above is the '
                 'neural signal written to the pickle by default')
    ax.set_ylabel('events')
    ax.legend(fontsize=6, ncol=3)

    ax = axes[3]
    ax.hist(r_speed[np.isfinite(r_speed)], bins=50)
    ax.axvline(INT_R_THRESH, color='r')
    ax.set_title(f'step 4 - neuron curation: r(dF/F, speed); '
                 f'{int((~keep_cells).sum())}/{len(keep_cells)} cells excluded as '
                 f'putative interneurons (r > {INT_R_THRESH})')
    ax.set_xlabel('Pearson r with speed')

    # behaviour + outputs over the same window
    ax = axes[4]
    ax.plot(tt, beh['pos'][sl], 'k', lw=1, label='position')
    for j, i in enumerate(range(0, 6)):
        s, e = int(tstart_inds[i]), int(teleport_inds[i])
        ax.axhspan(rz_coords[i][0], rz_coords[i][1],
                   xmin=(beh['time'][s] - tt[0]) / (tt[-1] - tt[0]),
                   xmax=(beh['time'][e] - tt[0]) / (tt[-1] - tt[0]),
                   color='g', alpha=0.2)
        ax.text(beh['time'][s], 460, f"tr{i} {rz_labels[i]} "
                f"{'rew' if isreward[i] else 'omit'}", fontsize=7)
    rw = np.where(beh['reward'][sl] > 0)[0]
    ax.plot(tt[rw], beh['pos'][sl][rw], 'b*', ms=10, label='reward')
    lk = np.where(beh['lick'][sl] > 0)[0]
    ax.plot(tt[lk], beh['pos'][sl][lk], 'r.', ms=3, label='lick')
    ax.set_title('step 5 - behaviour: position, reward zone (green), rewards, licks')
    ax.set_ylabel('position (cm)')
    ax.legend(fontsize=7)

    # concatenated converted trials, to verify alignment / discretisation
    ntr_plot = min(6, len(neural_trials))
    cat_out = np.concatenate(output_trials[:ntr_plot], axis=1)
    cat_in = np.concatenate(input_trials[:ntr_plot], axis=1)
    cat_neural = np.concatenate(neural_trials[:ntr_plot], axis=1)
    bounds = np.cumsum([0] + [o.shape[1] for o in output_trials[:ntr_plot]])
    x = np.arange(cat_out.shape[1])

    ax = axes[5]
    pos_cat = np.concatenate([beh['pos'][int(tstart_inds[i]):int(teleport_inds[i])]
                              for i in kept_trial_idx[:ntr_plot]])
    ax.plot(x, pos_cat, 'k', lw=1, label='position (cm)')
    ax.plot(x, cat_out[1] * 90 + 45, 'r', lw=1, alpha=0.7,
            label='position bin (x90+45)')
    for b in bounds:
        ax.axvline(b, color='b', ls=':')
    ax.set_title('step 6 - converted trials: position vs. its discretisation '
                 '(blue dotted = trial boundaries)')
    ax.legend(fontsize=7)

    ax = axes[6]
    d_cat = []
    for i in kept_trial_idx[:ntr_plot]:
        p = beh['pos'][int(tstart_inds[i]):int(teleport_inds[i])]
        rs, re = rz_coords[i]
        d = np.zeros_like(p)
        d[p < rs] = p[p < rs] - rs
        d[p > re] = p[p > re] - re
        d_cat.append(d)
    d_cat = np.concatenate(d_cat)
    ax.plot(x, d_cat, 'k', lw=1, label='distance to reward zone (cm)')
    ax.plot(x, (cat_out[0] - 3) * 40, 'r', lw=1, alpha=0.7,
            label='distance bin (centred, x40)')
    for lvl in [-50, -10, 0, 10, 50]:
        ax.axhline(lvl, color='g', lw=0.5, ls='--')
    for b in bounds:
        ax.axvline(b, color='b', ls=':')
    ax.set_title('step 7 - distance to reward zone vs. its discretisation '
                 '(green dashed = bin edges)')
    ax.legend(fontsize=7)

    ax = axes[7]
    spd_cat = np.concatenate([beh['speed'][int(tstart_inds[i]):int(teleport_inds[i])]
                              for i in kept_trial_idx[:ntr_plot]])
    ax.plot(x, spd_cat, 'k', lw=1, label='speed (cm/s)')
    ax.plot(x, cat_out[2] * 10, 'r', lw=1, alpha=0.7, label='speed bin (x10)')
    ax.plot(x, cat_out[3] * 5 - 10, 'm', lw=1, label='lick (x5 - 10)')
    ax.plot(x, cat_in[0], 'c', lw=1, label='input: time from trial start (s)')
    ax.plot(x, cat_neural.mean(axis=0) * 200, 'y', lw=1,
            label='mean neural signal (x200)')
    for b in bounds:
        ax.axvline(b, color='b', ls=':')
    ax.set_title('step 8 - speed / licks / decoder time input / mean neural activity, '
                 'all on the converted trial time base')
    ax.legend(fontsize=7)

    fig.tight_layout()
    out = os.path.join(plot_dir, f'processing_{sid}.png')
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f'  wrote {out}', flush=True)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _worker(args):
    path, show, plot_dir, neural_signal = args
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return convert_session(path, show_processing=show, plot_dir=plot_dir,
                               neural_signal=neural_signal)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile', type=str)
    ap.add_argument('--full', action='store_true', help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='save diagnostic plots for up to 2 sessions')
    ap.add_argument('--data-dir', type=str, default='/app/data')
    ap.add_argument('--nworkers', type=int, default=12)
    ap.add_argument('--neural-signal', type=str, default='dff',
                    choices=['events', 'dff'])
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
    # order sessions by subject then experiment day
    files.sort(key=lambda p: (p.split('/')[-2], int(p.split('ses-')[1][:2])))
    if args.sample:
        # two sessions from different mice, one of them a reward-switch session
        files = [f for f in files if 'sub-m11_ses-04' in f or 'sub-m13_ses-03' in f]
    print(f'Converting {len(files)} sessions with {args.nworkers} workers '
          f'(neural signal: {args.neural_signal})', flush=True)

    nplot = 2 if args.show_processing else 0
    jobs = [(f, i < nplot, os.path.dirname(os.path.abspath(args.outfile)) or '.',
             args.neural_signal) for i, f in enumerate(files)]

    data = {k: [] for k in ['neural', 'input', 'output', 'brain_region_idx']}
    subjects, subject_idx, session_info = [], [], []

    t0 = time.time()
    nworkers = min(args.nworkers, len(files))
    # 'spawn': suite2p/numba initialise OpenMP at import time, and forking such a
    # process aborts.
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=nworkers, mp_context=ctx) as ex:
        for k, (neural, inp, out, planes, info) in enumerate(ex.map(_worker, jobs)):
            sub = info['subject']
            if sub not in subjects:
                subjects.append(sub)
            subject_idx.append(subjects.index(sub))
            data['neural'].append(neural)
            data['input'].append(inp)
            data['output'].append(out)
            data['brain_region_idx'].append(np.zeros(len(planes), dtype=np.int64))
            info['plane_idx'] = planes.tolist()
            session_info.append(info)
            el = time.time() - t0
            print(f"[{k+1}/{len(files)}] {info['session_id']} {info['scene']:24s} "
                  f"cells {info['n_cells']:5d} (iscell {info['n_rois_iscell']:5d}, "
                  f"int {info['n_interneurons']:3d})  trials {info['n_trials_kept']:3d}"
                  f"/{info['n_trials_raw']:3d} "
                  f"(lick-err {info['n_dropped_lick_error']}, nan {info['n_dropped_nan']}) "
                  f"rew {info['frac_rewarded']:.2f} kt={int(info['keep_teleports'])} "
                  f"| load {info['t_load']:.1f}s dff {info['t_dff']:.1f}s "
                  f"tot {info['t_total']:.1f}s | elapsed {el:.0f}s", flush=True)

    data['subjects'] = subjects
    data['subject_idx'] = np.array(subject_idx, dtype=np.int64)
    data['brain_regions'] = ['CA1']
    data['input_names'] = INPUT_NAMES
    data['output_names'] = OUTPUT_NAMES
    data['output_values'] = OUTPUT_VALUES

    ntrials = sum(len(s) for s in data['neural'])
    ncells = sum(s[0].shape[0] for s in data['neural'] if len(s) > 0)
    nframes = sum(t.shape[1] for s in data['neural'] for t in s)
    data['metadata'] = {
        'task_description': (
            'Head-fixed mice run down a 450 cm virtual linear track for water reward '
            'delivered operantly for licking in a hidden 50 cm reward zone (zone A '
            '80-130 cm, B 200-250 cm, C 320-370 cm; only one active at a time). '
            'Reward is randomly omitted on ~15% of trials. On switch sessions the '
            'reward zone moves to a new location after 30 trials, sometimes together '
            'with a switch to a second virtual environment (ENV1/ENV2). Each trial is '
            'one lap, ending with a teleport through a variable-length grey inter-trial '
            'interval. Neural data are the dF/F of two-photon-imaged CA1 pyramidal '
            'neurons at each imaging frame. The decoder predicts, '
            'from CA1 activity plus trial context, the signed distance to the reward '
            'zone, absolute track position, running speed and licking at each imaging '
            'frame, and the reward-zone identity and reward outcome of the trial.'),
        'time_bin_size': 1000.0 / 15.5078125,   # ms per imaging frame (64.484 ms)
        'temporal_alignment_event': (
            'start of trial (VR trial_start: the animal enters the linear track at '
            'position 0 cm)'),
        'off_start': 0.0,
        'off_end': None,   # trials run to the teleport; duration varies by trial
        'trial_end_event': 'teleport (end of the track; the teleport sample is excluded)',
        'neural_signal': args.neural_signal,
        'neural_signal_description': (
            'dF/F computed per trial with a maximin baseline (20 s window) after '
            'neuropil subtraction (coef 0.7) and smoothed with a 2-sample (~0.129 s) '
            'Gaussian kernel, exactly as in Sosa et al. 2025'
            + (' , then deconvolved with OASIS (tau=0.7) to give "events"'
               if args.neural_signal == 'events' else '')),
        'sampling_rate_hz': 15.5078125,
        'n_sessions': len(files),
        'n_subjects': len(subjects),
        'n_trials': ntrials,
        'n_neurons_total': ncells,
        'n_timepoints_total': nframes,
        'neuron_curation': ('suite2p iscell (manual curation) then exclusion of '
                            'putative interneurons with Pearson r(dF/F, speed) > 0.5'),
        'trial_curation': ('trials with lick-sensor error (>30% of frames with a '
                           'cumulative lick count > 2) excluded'),
        'input_descriptions': [
            'time since trial start (s)',
            'virtual environment: 0 = ENV1, 1 = ENV2 (constant within a trial)',
            'trial number within the session, 0-indexed (constant within a trial)',
            'outcome of the previous trial: 0 = omitted, 1 = rewarded '
            '(0 for the first trial of a session)'],
        'output_descriptions': [
            'signed distance from the animal to the nearest point of the active reward '
            'zone (0 while inside the zone, negative before it, positive after it), '
            'discretised into 7 bins',
            'absolute position on the 450 cm track, 5 equal bins',
            'running speed, 5 bins',
            'licking (any lick detected in the imaging frame)',
            'active reward zone location (A/B/C), constant within a trial',
            'reward delivered on this trial (1) or omitted (0), constant within a trial'],
        'session_info': session_info,
        'source': 'DANDI:001361, Sosa, Plitt & Giocomo 2025, Nature Neuroscience',
        'brain_region_note': ('all neurons are dorsal CA1 pyramidal cells; m17 and m18 '
                              'were imaged in two planes (deep/superficial CA1), pooled '
                              'here as in the paper. Per-neuron imaging plane is stored '
                              'in metadata["session_info"][i]["plane_idx"].'),
    }

    print(f'\nTotal: {len(files)} sessions, {len(subjects)} subjects, {ntrials} trials, '
          f'{ncells} neurons, {nframes} timepoints', flush=True)
    print(f'Conversion time: {time.time() - t0:.1f}s', flush=True)

    t1 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {args.outfile} '
          f'({os.path.getsize(args.outfile) / 1e9:.2f} GB) in {time.time() - t1:.1f}s')


if __name__ == '__main__':
    main()
