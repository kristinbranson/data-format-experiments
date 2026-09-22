#!/usr/bin/env python3
"""
Convert the Sosa, Plitt & Giocomo (2025) hippocampal VR dataset (DANDI:001361, NWB)
into the decoder-compatible pickle format.

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Processing follows the reference repository `GiocomoLab/Sosa_et_al_2024`
(`/app/code/src/reward_relative`) and the paper Methods:

  * neurons        : suite2p `iscell` curation, then putative interneurons
                     (Pearson r(dF/F, speed) > 0.5) removed
                     -> spatial.is_putative_interneuron
  * neural signal  : dF/F recomputed exactly as preprocessing.dff, i.e.
                     neuropil subtraction (0.7), per-trial neuropil-mean add-back,
                     per-trial maximin baseline (Gaussian sigma=15 frames, then a
                     300-sample minimum then maximum filter ~= 20 s), (F-F0)/|F0|,
                     per-trial Gaussian sigma=2 smoothing
  * trials         : [trial_start_inds[t], teleport_inds[t])  (behavior.* convention)
  * trial curation : lick-sensor-error trials removed
                     -> behavior.correct_lick_sensor_error, correction_thr=0.3
  * reward zones   : behavior.get_reward_zones (scene name + switch after 30 trials)
  * reward outcome : behavior.get_trial_types (any(reward) and any(rzone))
"""

import argparse
import glob
import os
import pickle
import sys
import time
import warnings
from multiprocessing import Pool

import h5py
import numpy as np
import scipy.ndimage as ndi

# --------------------------------------------------------------------------------------
# Constants taken from the reference code / Methods
# --------------------------------------------------------------------------------------

DATA_ROOT = '/app/data'

# reward_relative.behavior.reward_zone_dict, via map_labels A->X, B->Y, C->Z
REWARD_ZONE_COORDS = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}

CHANGE_TRIAL = 30          # behavior.get_reward_zones default; "each switch occurred after 30 trials"
NEU_COEF = 0.7             # preprocessing.dff / utilities.default_dff_method
BASELINE_SMOOTH_SIGMA = 15  # frames, nansmooth(f_, [0, 15]) in preprocessing.dff
BASELINE_FILTER_WIN = 300  # frames, ~20 s minimum/maximum filter ("maximin")
DFF_SMOOTH_SIGMA = 2       # frames, "two-sample (~0.129 s) s.d. Gaussian kernel"
INTERNEURON_R_THRESH = 0.5  # dayData.int_thresh
LICK_CORRECTION_THR = 0.3  # Methods: ">30% of the ... samples ... cumulative lick count >2"

TRACK_LENGTH = 450.0
N_POSITION_BINS = 5        # decoder task: 5 equal bins over the 450 cm track
SPEED_BIN_EDGES = [2.0, 10.0, 20.0, 40.0]

INPUT_NAMES = ['time_from_trial_start', 'environment', 'trial_number', 'prev_trial_outcome']
OUTPUT_NAMES = ['reward_zone_distance', 'position', 'speed', 'lick',
                'reward_zone_location', 'reward_outcome']
OUTPUT_VALUES = [
    ['< -50 cm', '-50 to -10 cm', '-10 to <0 cm', '0 cm (in zone)',
     '>0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
    ['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm'],
    ['< 2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '> 40 cm/s'],
    ['no lick', 'lick'],
    ['zone A (80-130 cm)', 'zone B (200-250 cm)', 'zone C (320-370 cm)'],
    ['omitted', 'rewarded'],
]

BEHAVIOR_PATH = 'processing/behavior/BehavioralTimeSeries'
OPHYS_PATH = 'processing/ophys'


# --------------------------------------------------------------------------------------
# Reference-code helpers
# --------------------------------------------------------------------------------------

def get_reward_zone_labels(scene, ntrials, change_trial=CHANGE_TRIAL):
    """Per-trial reward zone label ('A'/'B'/'C'), replicating behavior.get_reward_zones.

    Non-switch scenes end in 'LocationX'; switch scenes contain '<first>_to' and end
    with the second zone letter (e.g. 'Env1_LocationB_to_A', 'Env1_B_to_Env2_C').
    """
    for lab in 'ABC':
        if scene.endswith('Location' + lab):
            return np.array([lab] * ntrials)
    for first in 'ABC':
        if f'{first}_to' in scene:
            second = scene[-1]
            if second not in 'ABC':
                break
            n0 = min(change_trial, ntrials)
            return np.array([first] * n0 + [second] * (ntrials - n0))
    raise NotImplementedError(f'Reward zones not defined for scene {scene}')


def nan_gaussian_filter1d(a, sigma, axis=-1):
    """utilities.nansmooth / TwoPUtils.utilities.nansmooth.

    Within a trial slice there are never NaNs, so this reduces exactly to
    scipy.ndimage.gaussian_filter1d (the weight array is all ones and the
    normalisation is 1 everywhere).
    """
    return ndi.gaussian_filter1d(a, sigma, axis=axis)


def compute_dff_trials(F, Fneu, starts, stops):
    """dF/F per trial, replicating reward_relative.preprocessing.dff.

    Reference call (utilities.multi_anim_sess, single-channel branch):
        pp.dff(F, trial_starts, teleports, f_neu=Fneu, neuropil_method='subtract',
               baseline_method='maximin', subtract_baseline=True, regress_ts=None,
               neu_coef=0.7, keep_teleports=False)
    which slices each trial as [start-1, stop-1).  We pass starts+1/stops+1 so the
    sliced window is exactly [trial_start, teleport), matching every behavioural
    function in the reference.

    Args:
        F, Fneu: (ncells, nframes) float arrays (raw fluorescence and neuropil)
        starts, stops: (ntrials,) int arrays; trial t is [starts[t], stops[t])

    Returns:
        list of (ncells, T_t) float32 dF/F arrays, one per trial
    """
    out = []
    for s, e in zip(starts, stops):
        f = F[:, s:e].astype(np.float64)
        fneu = Fneu[:, s:e].astype(np.float64)

        # neuropil subtraction, then add back the trial's mean neuropil so that the
        # baseline is not a near-zero number (preprocessing.dff)
        f = f - NEU_COEF * fneu
        f = f + NEU_COEF * np.mean(fneu, axis=1, keepdims=True)

        # "maximin" baseline: smooth, minimum-filter over ~20 s, then maximum-filter
        flow = nan_gaussian_filter1d(f, BASELINE_SMOOTH_SIGMA, axis=-1)
        flow = ndi.minimum_filter1d(flow, BASELINE_FILTER_WIN, axis=-1)
        flow = ndi.maximum_filter1d(flow, BASELINE_FILTER_WIN, axis=-1)

        dff = (f - flow) / np.abs(flow)
        dff = nan_gaussian_filter1d(dff, DFF_SMOOTH_SIGMA, axis=-1)
        out.append(dff.astype(np.float32))
    return out


def find_putative_interneurons(dff_trials, speed_trials, r_thresh=INTERNEURON_R_THRESH):
    """spatial.is_putative_interneuron(method='speed'): Pearson r(dF/F, speed) > thresh.

    The reference correlates the whole dF/F timeseries against speed over the samples
    where dF/F is not NaN, which is exactly the union of the trial windows.  Computed
    here from per-trial accumulators to avoid materialising the concatenation.
    """
    ncells = dff_trials[0].shape[0]
    n = 0
    sx = np.zeros(ncells)
    sxx = np.zeros(ncells)
    sxy = np.zeros(ncells)
    sy = 0.0
    syy = 0.0
    for d, v in zip(dff_trials, speed_trials):
        d64 = d.astype(np.float64)
        n += d64.shape[1]
        sx += d64.sum(axis=1)
        sxx += np.einsum('ij,ij->i', d64, d64)
        sxy += d64 @ v
        sy += v.sum()
        syy += v @ v
    cov = sxy - sx * sy / n
    vx = sxx - sx ** 2 / n
    vy = syy - sy ** 2 / n
    with np.errstate(invalid='ignore', divide='ignore'):
        r = cov / np.sqrt(vx * vy)
    return np.nan_to_num(r, nan=0.0) > r_thresh, r


# --------------------------------------------------------------------------------------
# Output discretisation (decoder task specification)
# --------------------------------------------------------------------------------------

def bin_reward_distance(dist):
    """0: <-50 | 1: [-50,-10) | 2: [-10,0) | 3: 0 | 4: (0,10] | 5: (10,50] | 6: >50"""
    b = np.zeros(dist.shape, dtype=np.int8)
    b[(dist >= -50) & (dist < -10)] = 1
    b[(dist >= -10) & (dist < 0)] = 2
    b[dist == 0] = 3
    b[(dist > 0) & (dist <= 10)] = 4
    b[(dist > 10) & (dist <= 50)] = 5
    b[dist > 50] = 6
    return b


def bin_position(pos):
    """5 equal bins spanning the 450 cm track (90 cm each)."""
    b = np.floor(pos / (TRACK_LENGTH / N_POSITION_BINS))
    return np.clip(b, 0, N_POSITION_BINS - 1).astype(np.int8)


def bin_speed(speed):
    """0: <2 | 1: 2-10 | 2: 10-20 | 3: 20-40 | 4: >40 cm/s"""
    return np.digitize(speed, SPEED_BIN_EDGES).astype(np.int8)


def reward_zone_distance(pos, lo, hi):
    """Signed distance to the nearest point of the reward zone; 0 inside the zone."""
    return np.where(pos < lo, pos - lo, np.where(pos > hi, pos - hi, 0.0))


# --------------------------------------------------------------------------------------
# Per-session conversion
# --------------------------------------------------------------------------------------

def read_behavior(f):
    """Read the behavioural streams and the trial boundaries from an open NWB file."""
    B = f[BEHAVIOR_PATH]
    beh = {k: B[f'{k}/data'][:] for k in
           ['position', 'speed', 'lick', 'reward_zone', 'environment',
            'trial_start', 'teleport', 'scanning']}
    beh['time'] = B['position/timestamps'][:]

    # Reward is stored as an event series; map each event back onto its imaging frame
    # to recover the per-frame binary vr_data['reward'] the reference code uses.
    rt = B['Reward/timestamps'][:]
    ts = beh['time']
    idx = np.clip(np.searchsorted(ts, rt), 0, len(ts) - 1)
    left = np.clip(idx - 1, 0, len(ts) - 1)
    idx = np.where(np.abs(ts[left] - rt) < np.abs(ts[idx] - rt), left, idx)
    reward = np.zeros(len(ts))
    reward[idx] = 1.0
    beh['reward'] = reward

    starts = np.where(beh['trial_start'] > 0)[0]
    stops = np.where(beh['teleport'] > 0)[0]
    assert len(starts) == len(stops) and np.all(stops > starts), 'bad trial boundaries'
    return beh, starts, stops


def read_fluorescence(f):
    """Pooled (ncells, nframes) F and Fneu over all imaging planes, iscell-curated."""
    seg = f[f'{OPHYS_PATH}/ImageSegmentation/PlaneSegmentation']
    iscell = seg['iscell'][:, 0] > 0
    planes = sorted(f[f'{OPHYS_PATH}/Fluorescence'].keys(),
                    key=lambda p: int(p.replace('plane', '')))
    Fs, Ns = [], []
    for p in planes:
        rois = f[f'{OPHYS_PATH}/Fluorescence/{p}/rois'][:]
        keep = iscell[rois]
        Fs.append(f[f'{OPHYS_PATH}/Fluorescence/{p}/data'][:][:, keep])
        Ns.append(f[f'{OPHYS_PATH}/Neuropil/{p}/data'][:][:, keep])
    F = np.ascontiguousarray(np.concatenate(Fs, axis=1).T)
    Fneu = np.ascontiguousarray(np.concatenate(Ns, axis=1).T)
    return F, Fneu, int(iscell.sum()), len(planes)


def convert_session(path, show_processing=False, outdir='/app'):
    """Convert one NWB session. Returns a dict with per-trial neural/input/output."""
    t0 = time.time()
    timing = {}
    with h5py.File(path, 'r') as f:
        beh, starts, stops = read_behavior(f)
        scene = f['identifier'][()].decode().split('/')[-1]
        subject = f['general/subject/subject_id'][()].decode()
        session_id = f['general/session_id'][()].decode()
        region = f['general/optophysiology/ImagingPlane/location'][()].decode()
        # ImagingPlane/imaging_rate is the total scan rate (31.0 Hz for the two
        # 2-plane mice m17/m18); the effective per-plane sampling rate -- which is
        # what every timeseries in the file is sampled at -- is 15.5078125 Hz for
        # every session.  Derive it from the frame timestamps.
        frame_rate = float(1.0 / np.median(np.diff(beh['time'])))
        t1 = time.time()
        timing['read_behavior'] = t1 - t0
        F, Fneu, n_iscell, nplanes = read_fluorescence(f)
    t2 = time.time()
    timing['read_fluorescence'] = t2 - t1

    ntrials = len(starts)

    # ---- per-trial behaviour -----------------------------------------------------
    pos_tr = [beh['position'][s:e] for s, e in zip(starts, stops)]
    speed_tr = [beh['speed'][s:e] for s, e in zip(starts, stops)]
    lick_tr = [beh['lick'][s:e] for s, e in zip(starts, stops)]
    rz_tr = [beh['reward_zone'][s:e] for s, e in zip(starts, stops)]
    rew_tr = [beh['reward'][s:e] for s, e in zip(starts, stops)]
    env_tr = [beh['environment'][s:e] for s, e in zip(starts, stops)]
    time_tr = [beh['time'][s:e] - beh['time'][s] for s, e in zip(starts, stops)]

    # behavior.get_trial_types
    isreward = np.array([float(np.any(r > 0) and np.any(z > 0))
                         for r, z in zip(rew_tr, rz_tr)])
    env = np.array([np.unique(e)[0] for e in env_tr])
    for e in env_tr:
        assert len(np.unique(e)) == 1, 'environment changes within a trial'

    # behavior.get_reward_zones
    labels = get_reward_zone_labels(scene, ntrials)

    # behavior.correct_lick_sensor_error -> trials to drop
    lick_error = np.array([(l > 2).sum() / len(l) > LICK_CORRECTION_THR for l in lick_tr])

    # ---- neural ------------------------------------------------------------------
    dff_trials = compute_dff_trials(F, Fneu, starts, stops)
    t3 = time.time()
    timing['dff'] = t3 - t2

    is_int, speed_r = find_putative_interneurons(dff_trials, [v.astype(np.float64)
                                                             for v in speed_tr])
    keep_cells = ~is_int
    dff_trials = [d[keep_cells] for d in dff_trials]
    t4 = time.time()
    timing['interneurons'] = t4 - t3

    n_nonfinite = int(sum(np.sum(~np.isfinite(d)) for d in dff_trials))
    if n_nonfinite:
        warnings.warn(f'{os.path.basename(path)}: {n_nonfinite} non-finite dF/F values '
                      f'set to 0')
        dff_trials = [np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
                      for d in dff_trials]

    # ---- assemble ----------------------------------------------------------------
    neural, inputs, outputs, kept_trials = [], [], [], []
    for t in range(ntrials):
        if lick_error[t]:
            continue
        T = len(pos_tr[t])
        lo, hi = REWARD_ZONE_COORDS[labels[t]]
        dist = reward_zone_distance(pos_tr[t], lo, hi)

        inp = np.empty((4, T), dtype=np.float32)
        inp[0] = time_tr[t]
        inp[1] = env[t]
        inp[2] = t                                  # original index within the session
        inp[3] = isreward[t - 1] if t > 0 else 0.0  # no known preceding lap for trial 0

        out = np.empty((6, T), dtype=np.int8)
        out[0] = bin_reward_distance(dist)
        out[1] = bin_position(pos_tr[t])
        out[2] = bin_speed(speed_tr[t])
        out[3] = (lick_tr[t] > 0).astype(np.int8)
        out[4] = 'ABC'.index(labels[t])
        out[5] = int(isreward[t])

        neural.append(dff_trials[t])
        inputs.append(inp)
        outputs.append(out)
        kept_trials.append(t)

    timing['total'] = time.time() - t0
    info = dict(
        path=path, subject=subject, session_id=session_id, scene=scene,
        region=region, frame_rate=frame_rate, nplanes=nplanes,
        n_iscell=n_iscell, n_interneurons=int(is_int.sum()),
        nneurons=int(keep_cells.sum()), ntrials_raw=ntrials,
        ntrials=len(neural), n_lick_error=int(lick_error.sum()),
        n_nonfinite=n_nonfinite, kept_trials=kept_trials,
        reward_rate=float(isreward.mean()), timing=timing,
    )

    if show_processing:
        plot_processing(path, beh, starts, stops, F, Fneu, dff_trials, keep_cells,
                        labels, isreward, env, lick_error, neural, inputs, outputs,
                        kept_trials, info, outdir)

    return dict(neural=neural, input=inputs, output=outputs, info=info)


# --------------------------------------------------------------------------------------
# Diagnostic plots (--show-processing)
# --------------------------------------------------------------------------------------

def plot_processing(path, beh, starts, stops, F, Fneu, dff_trials, keep_cells,
                    labels, isreward, env, lick_error, neural, inputs, outputs,
                    kept_trials, info, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sid = f"{info['subject']}_ses-{info['session_id']}"
    ntr = len(starts)
    demo = [t for t in range(ntr) if t in kept_trials][:4]
    if len(demo) < 4:
        demo = list(range(min(4, ntr)))

    fig, axes = plt.subplots(7, 1, figsize=(18, 22))

    # (1) session overview: position with trial windows
    ax = axes[0]
    n_show = min(len(beh['position']), stops[min(9, ntr - 1)] + 200)
    ax.plot(np.arange(n_show), beh['position'][:n_show], 'k-', lw=0.8, label='position')
    for t in range(min(10, ntr)):
        ax.axvspan(starts[t], stops[t], color='tab:blue', alpha=0.15)
        ax.axvline(starts[t], color='g', lw=0.8)
        ax.axvline(stops[t], color='r', lw=0.8)
    ax.set_title(f'{sid} ({info["scene"]}) - step 1: trial windows [trial_start, teleport)'
                 f' (green=start, red=teleport), first 10 trials')
    ax.set_xlabel('imaging frame')
    ax.set_ylabel('position (cm)')

    # (2) dF/F pipeline for one example cell over the demo trials
    ax = axes[1]
    cell_full = np.where(keep_cells)[0]
    # pick the cell with the largest dF/F range for visibility
    rng = np.array([np.ptp(d[:, :]) for d in [dff_trials[demo[0]]]])[0] if False else None
    c = int(np.argmax(dff_trials[demo[0]].std(axis=1)))
    craw = cell_full[c]
    off = 0
    for t in demo:
        s, e = starts[t], stops[t]
        x = np.arange(off, off + (e - s))
        f = F[craw, s:e].astype(np.float64) - 0.7 * Fneu[craw, s:e].astype(np.float64)
        f = f + 0.7 * Fneu[craw, s:e].mean()
        flow = ndi.gaussian_filter1d(f, BASELINE_SMOOTH_SIGMA)
        flow = ndi.maximum_filter1d(ndi.minimum_filter1d(flow, BASELINE_FILTER_WIN),
                                    BASELINE_FILTER_WIN)
        ax.plot(x, F[craw, s:e], color='0.7', lw=0.7,
                label='raw F' if t == demo[0] else None)
        ax.plot(x, f, 'b-', lw=0.7,
                label='F - 0.7*Fneu (+mean Fneu)' if t == demo[0] else None)
        ax.plot(x, flow, 'r-', lw=1.2, label='maximin baseline' if t == demo[0] else None)
        ax.axvline(off, color='k', ls=':', lw=0.7)
        off += (e - s)
    ax.legend(fontsize=8)
    ax.set_title(f'step 2: dF/F computation, example cell {craw} '
                 f'(trials {demo}) - concatenated trials')
    ax.set_ylabel('fluorescence (a.u.)')

    ax = axes[2]
    off = 0
    for t in demo:
        d = dff_trials[t][c]
        ax.plot(np.arange(off, off + len(d)), d, 'k-', lw=0.8)
        ax.axvline(off, color='k', ls=':', lw=0.7)
        off += len(d)
    ax.set_title('step 2 (cont.): resulting dF/F for the same cell')
    ax.set_ylabel('dF/F')

    # (3) neural raster for the demo trials
    ax = axes[3]
    mat = np.concatenate([neural[kept_trials.index(t)] for t in demo
                          if t in kept_trials], axis=1)
    lim = np.percentile(mat, 99)
    ax.imshow(mat[:min(200, mat.shape[0])], aspect='auto', vmin=0, vmax=max(lim, 1e-3),
              cmap='magma', interpolation='nearest')
    ax.set_title(f'step 3: converted neural (dF/F) matrix, first <=200 of '
                 f'{mat.shape[0]} curated neurons')
    ax.set_ylabel('neuron')

    # (4) inputs
    ax = axes[4]
    off = 0
    for t in demo:
        if t not in kept_trials:
            continue
        i = kept_trials.index(t)
        x = np.arange(off, off + inputs[i].shape[1])
        for k, nm in enumerate(INPUT_NAMES):
            ax.plot(x, inputs[i][k], lw=1.0,
                    color=f'C{k}', label=nm if t == demo[0] else None)
        ax.axvline(off, color='k', ls=':', lw=0.7)
        off += inputs[i].shape[1]
    ax.legend(fontsize=8, ncol=4)
    ax.set_title('step 4: decoder inputs (time from trial start, environment, '
                 'trial number, previous outcome)')

    # (5) position + reward zone + discretised distance
    ax = axes[5]
    ax2 = ax.twinx()
    off = 0
    for t in demo:
        if t not in kept_trials:
            continue
        i = kept_trials.index(t)
        p = beh['position'][starts[t]:stops[t]]
        x = np.arange(off, off + len(p))
        lo, hi = REWARD_ZONE_COORDS[labels[t]]
        ax.plot(x, p, 'k-', lw=1.0, label='position (cm)' if t == demo[0] else None)
        ax.fill_between(x, lo, hi, color='tab:green', alpha=0.25,
                        label=f'reward zone' if t == demo[0] else None)
        ax2.plot(x, outputs[i][0], 'm-', lw=1.0, drawstyle='steps-post',
                 label='reward_zone_distance bin' if t == demo[0] else None)
        ax2.plot(x, outputs[i][1], 'c--', lw=1.0, drawstyle='steps-post',
                 label='position bin' if t == demo[0] else None)
        ax.axvline(off, color='k', ls=':', lw=0.7)
        off += len(p)
    ax.legend(fontsize=8, loc='upper left')
    ax2.legend(fontsize=8, loc='upper right')
    ax.set_ylabel('position (cm)')
    ax2.set_ylabel('output bin')
    ax.set_title('step 5: position, active reward zone and the discretised outputs')

    # (6) speed, licks, per-trial outputs
    ax = axes[6]
    ax2 = ax.twinx()
    off = 0
    for t in demo:
        if t not in kept_trials:
            continue
        i = kept_trials.index(t)
        v = beh['speed'][starts[t]:stops[t]]
        x = np.arange(off, off + len(v))
        ax.plot(x, v, color='0.5', lw=0.8, label='speed (cm/s)' if t == demo[0] else None)
        ax2.plot(x, outputs[i][2], 'b-', lw=1.0, drawstyle='steps-post',
                 label='speed bin' if t == demo[0] else None)
        ax2.plot(x, outputs[i][3], 'r-', lw=1.2, drawstyle='steps-post',
                 label='lick' if t == demo[0] else None)
        ax2.plot(x, outputs[i][4], 'g:', lw=1.5,
                 label='reward zone location' if t == demo[0] else None)
        ax2.plot(x, outputs[i][5] + 0.05, 'y-.', lw=1.5,
                 label='reward outcome' if t == demo[0] else None)
        ax.axvline(off, color='k', ls=':', lw=0.7)
        off += len(v)
    ax.legend(fontsize=8, loc='upper left')
    ax2.legend(fontsize=8, loc='upper right')
    ax.set_ylabel('speed (cm/s)')
    ax2.set_ylabel('output value')
    ax.set_title('step 6: speed / lick / per-trial outputs')
    ax.set_xlabel('concatenated frames of the example trials')

    fig.tight_layout()
    fn = os.path.join(outdir, f'processing_{sid}.png')
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    print(f'  wrote {fn}', flush=True)


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

_WORKER_ARGS = {}


def _worker(path):
    try:
        return convert_session(path, **_WORKER_ARGS)
    except Exception as exc:  # pragma: no cover
        import traceback
        traceback.print_exc()
        raise RuntimeError(f'failed on {path}: {exc}')


def _init(kwargs):
    _WORKER_ARGS.update(kwargs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', default=True,
                    help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true',
                    help='process only 2 sessions for testing')
    ap.add_argument('--show-processing', action='store_true',
                    help='save per-step diagnostic plots for up to 2 sessions')
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
    if args.sample:
        # one small single-plane session and one large 2-plane session
        files = ['/app/data/sub-m11/sub-m11_ses-03_behavior+ophys.nwb',
                 '/app/data/sub-m17/sub-m17_ses-05_behavior+ophys.nwb']
    print(f'Converting {len(files)} sessions', flush=True)

    plot_files = set(files[:2]) if args.show_processing else set()
    t_start = time.time()

    results = []
    if args.workers > 1 and not args.show_processing:
        with Pool(args.workers, initializer=_init, initargs=({},)) as pool:
            for i, res in enumerate(pool.imap(_worker, files, chunksize=1)):
                results.append(res)
                _report(i, len(files), res, t_start)
    else:
        for i, path in enumerate(files):
            res = convert_session(path, show_processing=path in plot_files)
            results.append(res)
            _report(i, len(files), res, t_start)

    # order sessions by mouse number then experiment day
    results.sort(key=lambda r: (int(r['info']['subject'][1:]),
                                int(r['info']['session_id'])))

    subjects = sorted({r['info']['subject'] for r in results},
                      key=lambda s: int(s[1:]))
    subject_idx = np.array([subjects.index(r['info']['subject']) for r in results],
                           dtype=np.int64)
    brain_regions = ['CA1']
    brain_region_idx = [np.zeros(r['info']['nneurons'], dtype=np.int64) for r in results]

    frame_rates = np.array([r['info']['frame_rate'] for r in results])
    assert np.allclose(frame_rates, frame_rates[0], rtol=1e-6), \
        f'inconsistent frame rates: {np.unique(frame_rates)}'
    bin_ms = 1000.0 / float(np.mean(frame_rates))

    data = {
        'neural': [r['neural'] for r in results],
        'input': [r['input'] for r in results],
        'output': [r['output'] for r in results],
        'subjects': subjects,
        'subject_idx': subject_idx,
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description': (
                'Head-fixed mice ran laps on a 450 cm virtual linear track (ENV 1 or ENV 2) '
                'containing a hidden 50 cm reward zone at one of three locations '
                '(A 80-130 cm, B 200-250 cm, C 320-370 cm). Sucrose reward was delivered '
                'operantly for licking inside the zone and was randomly omitted on ~15% of '
                'trials. On switch sessions the zone moved to a new location after 30 trials. '
                'Two-photon calcium imaging (GCaMP7f) of hippocampal CA1 at ~15.5 Hz. '
                'Decoder outputs: distance to the active reward zone (7 bins), absolute '
                'track position (5 bins), running speed (5 bins), licking (binary), '
                'reward zone identity (A/B/C) and reward outcome (omitted/rewarded).'),
            'time_bin_size': bin_ms,
            'temporal_alignment_event': (
                'start of trial: the frame the animal enters the linear track at 0 cm '
                '(vr_data "tstart" flag / trial_start_inds)'),
            'off_start': 0.0,
            'off_end': None,
            'off_end_note': (
                'Trials have different lengths; each trial ends at the teleport frame '
                '(end of the 450 cm track, exclusive), i.e. the full on-track lap.'),
            'neural_signal': (
                'dF/F, computed as in reward_relative.preprocessing.dff: neuropil '
                'subtraction (coefficient 0.7) with the per-trial neuropil mean added '
                'back, per-trial "maximin" baseline (Gaussian sigma = 15 frames, then a '
                '300-frame ~20 s minimum filter followed by a 300-frame maximum filter), '
                'dF/F = (F - F0)/|F0|, smoothed with a 2-frame s.d. Gaussian.'),
            'neural_signal_note': (
                'The NWB "Deconvolved" array is suite2p spks computed from raw F, not '
                'the events the paper uses (which are deconvolved from this dF/F with a '
                'per-session suite2p tau that is not in the NWB export), so it is not '
                'used. Empirically, dF/F also decodes better than OASIS events '
                '(tau = 0.7) for every output.'),
            'neuron_curation': (
                'suite2p + manual curation (iscell == 1), then putative interneurons '
                'removed (Pearson r between dF/F and running speed > 0.5, as in '
                'reward_relative.spatial.is_putative_interneuron).'),
            'trial_curation': (
                'Trials with capacitive-lick-sensor errors removed '
                '(>30% of frames with cumulative lick count > 2, as in '
                'reward_relative.behavior.correct_lick_sensor_error with thr = 0.3).'),
            'sampling_rate_hz': 1000.0 / bin_ms,
            'session_info': [
                {k: r['info'][k] for k in
                 ['subject', 'session_id', 'scene', 'nplanes', 'n_iscell',
                  'n_interneurons', 'nneurons', 'ntrials_raw', 'ntrials',
                  'n_lick_error', 'reward_rate']}
                for r in results],
            'source': 'DANDI:001361 (Sosa, Plitt & Giocomo 2025, Nat Neurosci)',
            'reference_code': 'https://github.com/GiocomoLab/Sosa_et_al_2024',
        },
    }

    _summarise(data, results)

    t = time.time()
    with open(args.outfile, 'wb') as fh:
        pickle.dump(data, fh, protocol=4)
    print(f'Wrote {args.outfile} '
          f'({os.path.getsize(args.outfile) / 1e9:.2f} GB) in {time.time() - t:.1f} s')
    print(f'Total elapsed: {time.time() - t_start:.1f} s')


def _report(i, n, res, t_start):
    info = res['info']
    el = time.time() - t_start
    print(f"[{i + 1}/{n}] {info['subject']} ses-{info['session_id']} {info['scene']}: "
          f"{info['nneurons']} neurons ({info['n_iscell']} iscell, "
          f"{info['n_interneurons']} interneurons), "
          f"{info['ntrials']}/{info['ntrials_raw']} trials "
          f"(-{info['n_lick_error']} lick err), "
          f"t={info['timing']['total']:.1f}s "
          f"[read {info['timing']['read_fluorescence']:.1f} dff {info['timing']['dff']:.1f} "
          f"int {info['timing']['interneurons']:.1f}] "
          f"| elapsed {el:.0f}s, eta {el / (i + 1) * (n - i - 1):.0f}s", flush=True)


def _summarise(data, results):
    print('\n================ CONVERSION SUMMARY ================')
    nses = len(data['neural'])
    ntr = sum(len(s) for s in data['neural'])
    nneur = sum(s[0].shape[0] for s in data['neural'])
    nframes = sum(t.shape[1] for s in data['neural'] for t in s)
    print(f'sessions: {nses}  subjects: {len(data["subjects"])}  trials: {ntr}')
    print(f'neurons (sum over sessions): {nneur}  '
          f'mean/session: {nneur / nses:.1f}  '
          f'min: {min(s[0].shape[0] for s in data["neural"])}  '
          f'max: {max(s[0].shape[0] for s in data["neural"])}')
    print(f'total timepoints: {nframes}  '
          f'mean trial length: {nframes / ntr:.1f} frames '
          f'({nframes / ntr * data["metadata"]["time_bin_size"] / 1000:.1f} s)')
    print(f'trials removed for lick-sensor error: '
          f'{sum(r["info"]["n_lick_error"] for r in results)}')
    print(f'putative interneurons removed: '
          f'{sum(r["info"]["n_interneurons"] for r in results)} of '
          f'{sum(r["info"]["n_iscell"] for r in results)} iscell '
          f'({100 * sum(r["info"]["n_interneurons"] for r in results) / sum(r["info"]["n_iscell"] for r in results):.2f}%)')
    frac_int = np.array([r['info']['n_interneurons'] / r['info']['n_iscell']
                         for r in results]) * 100
    print(f'  per-session interneuron %: {frac_int.mean():.2f} +/- {frac_int.std():.2f}')

    allin = np.concatenate([t for s in data['input'] for t in s], axis=1)
    print('\ninput ranges:')
    for k, nm in enumerate(data['input_names']):
        print(f'  {k} {nm}: [{allin[k].min():.3f}, {allin[k].max():.3f}]')
    print('\noutput distributions (fraction of timepoints):')
    allout = np.concatenate([t for s in data['output'] for t in s], axis=1)
    for k, nm in enumerate(data['output_names']):
        vals = np.bincount(allout[k].astype(int),
                           minlength=len(data['output_values'][k]))
        print(f'  {k} {nm}: ' +
              ', '.join(f'{data["output_values"][k][j]}={vals[j] / vals.sum():.4f}'
                        for j in range(len(vals))))
    # per-trial statistics for the trial-constant outputs
    zone = np.array([t[4, 0] for s in data['output'] for t in s])
    rew = np.array([t[5, 0] for s in data['output'] for t in s])
    print(f'\nper-trial reward zone A/B/C: '
          f'{np.bincount(zone, minlength=3) / len(zone)}')
    print(f'per-trial reward rate: {rew.mean():.4f}')
    neural_min = min(t.min() for s in data['neural'] for t in s)
    neural_max = max(t.max() for s in data['neural'] for t in s)
    print(f'\ndF/F range: [{neural_min:.3f}, {neural_max:.3f}]')
    print('====================================================\n')


if __name__ == '__main__':
    main()
