"""
Convert the Sosa, Plitt & Giocomo (2025) hippocampal CA1 2P + VR dataset
(DANDI:001361) into the decoder-compatible pickle format.

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Processing follows the reference code (https://github.com/GiocomoLab/Sosa_et_al_2024,
copy in /app/code) and the paper methods (/app/methods.txt):

  neural  : raw suite2p F/Fneu -> manual curation (iscell) -> dF/F with per-trial
            maximin baseline (reward_relative.preprocessing.dff) -> 2-sample gaussian
            smoothing -> OASIS deconvolution (suite2p dcnv.oasis, tau=0.7)
            = the paper's `events` timeseries, sliced into trials
  inputs  : time from trial start, environment (ENV1/ENV2), trial number,
            previous trial outcome
  outputs : distance to reward zone, absolute position, speed, lick (time-varying),
            reward zone identity and reward outcome (per trial, broadcast in time)
"""

import argparse
import glob
import os
import pickle
import re
import sys
import time
import warnings
import multiprocessing as mp

import h5py
import numpy as np
from scipy import ndimage

# ----------------------------------------------------------------------------- constants
FRAME_RATE = 15.5078125            # Hz, imaging frame rate per plane (NWB ImagingPlane)
DT = 1.0 / FRAME_RATE              # 64.48 ms
NEU_COEF = 0.7                     # neuropil coefficient (reference dff default)
TAU = 0.7                          # GCaMP7f tau used in suite2p ops (example_m12.ipynb)
BASELINE_SMOOTH_SIGMA = 15         # frames, reference: nansmooth(f, [0, 15])
BASELINE_FILTER_WIN = 300          # frames ~ 20 s, reference: min/max filter 300
DFF_SMOOTH_SIGMA = 2               # frames (~0.129 s), reference: nansmooth(dff, 2)
INTERNEURON_R_THRESH = 0.5         # Pearson r(dF/F, speed) > 0.5 -> putative interneuron
LICK_ERROR_FRAC = 0.30             # >30% of frames with cumulative lick count > 2
LICK_ERROR_COUNT = 2
CHANGE_TRIAL = 30                  # reward zone / environment switch trial (behavior.py)
TRACK_LENGTH = 450.0
REWARD_ZONES = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
ZONE_TO_IDX = {'A': 0, 'B': 1, 'C': 2}

INPUT_NAMES = ['time_from_trial_start', 'environment', 'trial_number',
               'previous_trial_outcome']
OUTPUT_NAMES = ['reward_zone_distance', 'position', 'speed', 'lick',
                'reward_zone_location', 'reward_outcome']
OUTPUT_VALUES = [
    ['< -50 cm', '-50 to -10 cm', '-10 to <0 cm', 'in zone (0 cm)',
     '>0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
    ['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm'],
    ['< 2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '> 40 cm/s'],
    ['no lick', 'lick'],
    ['A (80-130 cm)', 'B (200-250 cm)', 'C (320-370 cm)'],
    ['omitted', 'rewarded'],
]


# ----------------------------------------------------------------------------- helpers
def nansmooth(a, sig, axis=-1):
    """Gaussian smoothing that does not propagate NaNs.

    Copied from reward_relative.utilities.nansmooth / TwoPUtils.utilities.nansmooth.
    """
    nan_inds = np.isnan(a)
    a_nanless = np.copy(a)
    a_nanless[nan_inds] = 0
    one = np.ones(a.shape, dtype=a.dtype)
    one[nan_inds] = 0.001
    a_nanless = ndimage.gaussian_filter1d(a_nanless, sig, axis=axis)
    one = ndimage.gaussian_filter1d(one, sig, axis=axis)
    return a_nanless / one


def parse_scene(scene):
    """Parse a VR scene name into per-condition reward zone labels / environments.

    Mirrors reward_relative.behavior.get_reward_zones (zone from scene name, switch
    after CHANGE_TRIAL trials) and behavior.env_morph_dict (Env1 -> 0, Env2 -> 1).

    Returns (zone1, zone2, env1, env2); zone2/env2 are None for non-switch sessions.
    """
    m = re.match(r'^Env(\d)_Location([ABC])$', scene)
    if m:
        return m.group(2), None, int(m.group(1)) - 1, None
    m = re.match(r'^Env(\d)_Location([ABC])_to_([ABC])$', scene)
    if m:
        e = int(m.group(1)) - 1
        return m.group(2), m.group(3), e, e
    m = re.match(r'^Env(\d)_([ABC])_to_Env(\d)_([ABC])$', scene)
    if m:
        return m.group(2), m.group(4), int(m.group(1)) - 1, int(m.group(3)) - 1
    raise ValueError(f'unrecognized scene name: {scene}')


def zone_per_trial(scene, ntrials):
    """Per-trial reward zone label and environment index (0 = ENV1, 1 = ENV2)."""
    z1, z2, e1, e2 = parse_scene(scene)
    if z2 is None:
        return [z1] * ntrials, [e1] * ntrials
    n1 = min(CHANGE_TRIAL, ntrials)
    return ([z1] * n1 + [z2] * (ntrials - n1),
            [e1] * n1 + [e2] * (ntrials - n1))


def signed_distance_to_zone(pos, zone):
    """Signed distance (cm) from position to the nearest point of the reward zone.

    0 inside the zone, negative before the zone, positive past the zone.
    (Reference GLM code uses pos - zone_start; here we use the distance to *any*
    location in the zone, as required by the decoder output specification.)
    """
    lo, hi = zone
    d = np.zeros_like(pos)
    before = pos < lo
    after = pos > hi
    d[before] = pos[before] - lo
    d[after] = pos[after] - hi
    return d


def bin_distance(d):
    out = np.full(d.shape, -1, dtype=np.int64)
    out[d < -50] = 0
    out[(d >= -50) & (d < -10)] = 1
    out[(d >= -10) & (d < 0)] = 2
    out[d == 0] = 3
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    assert np.all(out >= 0)
    return out


def bin_position(pos):
    return np.clip(np.floor(pos / (TRACK_LENGTH / 5.0)), 0, 4).astype(np.int64)


def bin_speed(speed):
    return np.digitize(speed, [2.0, 10.0, 20.0, 40.0]).astype(np.int64)


# ----------------------------------------------------------------------------- loading
def load_behavior(f):
    """Load the VR behavior streams (already interpolated onto imaging frames)."""
    b = f['processing/behavior/BehavioralTimeSeries']
    beh = {k: b[k]['data'][:] for k in
           ['position', 'environment', 'lick', 'reward_zone', 'speed',
            'teleport', 'trial_start', 'trial number', 'autoreward', 'scanning']}
    beh['t'] = b['position']['timestamps'][:]
    # `Reward` is a sparse TimeSeries (one sample per delivered reward)
    rew_t = b['Reward']['timestamps'][:]
    rew_bin = np.zeros_like(beh['position'])
    if len(rew_t):
        idx = np.searchsorted(beh['t'], rew_t)
        idx = np.clip(idx, 0, len(rew_bin) - 1)
        rew_bin[idx] = 1
    beh['reward'] = rew_bin
    return beh


def load_fluorescence(f):
    """Pooled F / Fneu of manually curated cells (iscell), planes concatenated."""
    seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
    iscell = seg['iscell'][:, 0].astype(bool)
    plane_idx = seg['planeIdx'][:]
    Fs, Fns = [], []
    for p in np.unique(plane_idx):
        Fs.append(f[f'processing/ophys/Fluorescence/plane{p}/data'][:].T)
        Fns.append(f[f'processing/ophys/Neuropil/plane{p}/data'][:].T)
    F = np.concatenate(Fs, axis=0).astype(np.float32)
    Fneu = np.concatenate(Fns, axis=0).astype(np.float32)
    assert F.shape[0] == len(iscell), (F.shape, len(iscell))
    return F[iscell], Fneu[iscell], plane_idx[iscell]


# ----------------------------------------------------------------------------- dF/F
def compute_dff(F, Fneu, starts, stops):
    """dF/F exactly as reward_relative.preprocessing.dff(..., 'maximin', 'subtract').

    Samples outside trials are NaN (keep_teleports=False behaviour). Baseline is
    computed within each trial: gaussian smoothing (sigma 15 frames), minimum filter
    then maximum filter over 300 frames (~20 s), dff = (F - base) / |base|, then
    smoothed with a 2-sample gaussian kernel within each trial.
    """
    f_ = np.full(F.shape, np.nan, dtype=np.float32)
    fneu_ = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        f_[:, s:e] = F[:, s:e]
        fneu_[:, s:e] = Fneu[:, s:e]

    # neuropil subtraction
    f_ -= NEU_COEF * fneu_

    flow = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        # add back the per-trial mean neuropil so dF/F is close to true dF/F
        f_[:, s:e] += NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
        seg = nansmooth(f_[:, s:e], BASELINE_SMOOTH_SIGMA, axis=-1)
        seg = ndimage.minimum_filter1d(seg, BASELINE_FILTER_WIN, axis=-1)
        seg = ndimage.maximum_filter1d(seg, BASELINE_FILTER_WIN, axis=-1)
        flow[:, s:e] = seg

    mask = ~np.isnan(f_[0, :])
    dff = np.full(F.shape, np.nan, dtype=np.float32)
    dff[:, mask] = (f_[:, mask] - flow[:, mask]) / np.abs(flow[:, mask])

    for s, e in zip(starts, stops):
        dff[:, s:e] = nansmooth(dff[:, s:e], DFF_SMOOTH_SIGMA, axis=1)
    return dff


def deconvolve(dff, starts, stops):
    """OASIS deconvolution of dF/F per trial (reference: dcnv.oasis in dff())."""
    from suite2p.extraction import dcnv  # lazy import: keeps OpenMP out of the parent
    spks = np.zeros(dff.shape, dtype=np.float32)
    for s, e in zip(starts, stops):
        spks[:, s:e] = dcnv.oasis(
            np.ascontiguousarray(dff[:, s:e], dtype=np.float32),
            2000, TAU, FRAME_RATE)
    return spks


# ----------------------------------------------------------------------------- session
def convert_session(fname, show_processing=False, neural_signal='dff'):
    """Convert one NWB session. Returns a dict (or None if nothing usable)."""
    t0 = time.time()
    timings = {}
    with h5py.File(fname, 'r') as f:
        scene = f['identifier'][()].decode().split('/')[-1]
        subject = f['general/subject/subject_id'][()].decode()
        session_id = f['general/session_id'][()].decode()
        region = f['general/optophysiology/ImagingPlane/location'][()].decode()
        rate = float(f['general/optophysiology/ImagingPlane/imaging_rate'][()])
        beh = load_behavior(f)
        t1 = time.time()
        F, Fneu, plane_idx = load_fluorescence(f)
    timings['load'] = time.time() - t0

    # Occasionally the VR/behavior stream is 1 sample longer than the imaging stream
    # ("one frame correction ... scan stopping mid frame" in the reference
    # TwoPUtils/preprocessing.vr_align_to_2P). Truncate both to the common length.
    n_beh = len(beh['position'])
    n_ophys = F.shape[1]
    nframes = min(n_beh, n_ophys)
    if n_beh != n_ophys:
        assert abs(n_beh - n_ophys) <= 5, (n_beh, n_ophys)
        print(f'  {os.path.basename(fname)}: behavior has {n_beh} samples, imaging '
              f'{n_ophys}; truncating both to {nframes} (reference one-frame correction)')
        for k in list(beh.keys()):
            beh[k] = beh[k][:nframes]
        F = F[:, :nframes]
        Fneu = Fneu[:, :nframes]

    # ---- trials: [trial_start, teleport); the teleport frame has an interpolated position
    starts = np.where(beh['trial_start'] > 0)[0]
    stops = np.where(beh['teleport'] > 0)[0]
    assert len(starts) == len(stops) and np.all(stops > starts)
    # a trial whose teleport was cut off by the truncation above is dropped
    valid = stops < nframes
    if not np.all(valid):
        print(f'  {os.path.basename(fname)}: dropping {int((~valid).sum())} trial(s) '
              f'truncated at the end of the recording')
        starts, stops = starts[valid], stops[valid]
    ntrials_raw = len(starts)

    # ---- dF/F and deconvolved activity (the paper's `events`)
    t1 = time.time()
    dff = compute_dff(F, Fneu, starts, stops)
    timings['dff'] = time.time() - t1

    # ---- neuron curation: exclude putative interneurons (r(dF/F, speed) > 0.5)
    in_trial = np.zeros(nframes, dtype=bool)
    for s, e in zip(starts, stops):
        in_trial[s:e] = True
    d = dff[:, in_trial]
    sp = beh['speed'][in_trial].astype(np.float32)
    dz = d - d.mean(axis=1, keepdims=True)
    sz = sp - sp.mean()
    denom = (np.sqrt((dz ** 2).sum(axis=1)) * np.sqrt((sz ** 2).sum()))
    with np.errstate(invalid='ignore', divide='ignore'):
        r_speed = (dz @ sz) / denom
    r_speed = np.nan_to_num(r_speed, nan=0.0)
    keep_cells = r_speed <= INTERNEURON_R_THRESH
    n_interneurons = int((~keep_cells).sum())

    t1 = time.time()
    dff_kept = dff[keep_cells]
    events = deconvolve(dff_kept, starts, stops)
    timings['deconv'] = time.time() - t1
    activity = events if neural_signal == 'events' else np.nan_to_num(dff_kept)

    # ---- per-trial task variables
    zones, envs_scene = zone_per_trial(scene, ntrials_raw)
    env_stream = np.array([np.median(beh['environment'][s:e])
                           for s, e in zip(starts, stops)])
    # reward outcome, as in behavior.get_trial_types: reward delivered AND zone entered
    isreward = np.array([(beh['reward'][s:e].sum() > 0) and
                         (beh['reward_zone'][s:e].sum() > 0)
                         for s, e in zip(starts, stops)], dtype=np.int64)
    # lick-sensor error trials (paper: >30% of frames with cumulative lick count > 2)
    lick_bad = np.array([(beh['lick'][s:e] > LICK_ERROR_COUNT).mean() > LICK_ERROR_FRAC
                         for s, e in zip(starts, stops)])

    neural, inputs, outputs = [], [], []
    kept_trials = []
    for i, (s, e) in enumerate(zip(starts, stops)):
        if lick_bad[i]:
            continue
        T = e - s
        pos = beh['position'][s:e].astype(np.float64)
        speed = beh['speed'][s:e].astype(np.float64)
        licks = beh['lick'][s:e].astype(np.float64)
        tt = beh['t'][s:e] - beh['t'][s]

        zone = REWARD_ZONES[zones[i]]
        dist = signed_distance_to_zone(pos, zone)

        inp = np.empty((len(INPUT_NAMES), T), dtype=np.float32)
        inp[0] = tt
        inp[1] = env_stream[i]
        inp[2] = i
        inp[3] = isreward[i - 1] if i > 0 else 1   # see notes: warm-up trials precede

        out = np.empty((len(OUTPUT_NAMES), T), dtype=np.int64)
        out[0] = bin_distance(dist)
        out[1] = bin_position(pos)
        out[2] = bin_speed(speed)
        out[3] = (licks > 0).astype(np.int64)
        out[4] = ZONE_TO_IDX[zones[i]]
        out[5] = isreward[i]

        act = np.ascontiguousarray(activity[:, s:e], dtype=np.float32)
        assert np.all(np.isfinite(act)) and np.all(np.isfinite(inp))
        neural.append(act)
        inputs.append(inp)
        outputs.append(out)
        kept_trials.append(i)

    if len(neural) < 2:
        print(f'  !! {os.path.basename(fname)}: only {len(neural)} usable trials, skipping')
        return None

    result = dict(
        file=os.path.basename(fname), subject=subject, session_id=session_id,
        scene=scene, region=region, imaging_rate=rate,
        neural=neural, input=inputs, output=outputs,
        n_neurons=int(events.shape[0]), n_interneurons=n_interneurons,
        n_roi_iscell=int(F.shape[0]), plane_idx=plane_idx[keep_cells],
        ntrials_raw=ntrials_raw, kept_trials=np.array(kept_trials),
        n_lick_bad=int(lick_bad.sum()),
        isreward=isreward, zones=zones, env_stream=env_stream, envs_scene=envs_scene,
        timings=timings, total_time=time.time() - t0,
    )

    if show_processing:
        plot_processing(result, beh, F, Fneu, dff_kept, events, starts, stops)

    print(f'  {os.path.basename(fname)}: {result["n_neurons"]} cells '
          f'({n_interneurons} interneurons excluded), {len(neural)}/{ntrials_raw} trials '
          f'({int(lick_bad.sum())} lick-error), {result["total_time"]:.1f} s '
          f'(load {timings["load"]:.1f}, dff {timings["dff"]:.1f}, '
          f'deconv {timings["deconv"]:.1f})', flush=True)
    return result


# ----------------------------------------------------------------------------- plotting
def plot_processing(res, beh, F, Fneu, dff, events, starts, stops):
    """Visualize every processing step for one session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sid = f'{res["subject"]}_ses{res["session_id"]}'
    fig, ax = plt.subplots(7, 1, figsize=(16, 20), sharex=True)
    # first 6 trials
    s0, s1 = starts[0], stops[min(5, len(stops) - 1)]
    tt = beh['t'][s0:s1]
    ncell = min(4, F.shape[0])
    for c in range(ncell):
        ax[0].plot(tt, F[c, s0:s1], lw=0.6, label=f'F cell{c}')
        ax[0].plot(tt, Fneu[c, s0:s1], lw=0.4, alpha=0.5)
    ax[0].set_ylabel('raw F / Fneu')
    ax[0].legend(fontsize=6, ncol=4)
    ax[0].set_title(f'{sid} ({res["scene"]}): processing steps, first 6 trials')
    for c in range(min(4, dff.shape[0])):
        ax[1].plot(tt, dff[c, s0:s1], lw=0.6)
    ax[1].set_ylabel('dF/F (maximin)')
    for c in range(min(4, events.shape[0])):
        ax[2].plot(tt, events[c, s0:s1], lw=0.6)
    ax[2].set_ylabel('deconvolved (events)')
    ax[3].plot(tt, beh['position'][s0:s1], 'k', lw=1)
    for i, (s, e) in enumerate(zip(starts, stops)):
        if s >= s1:
            break
        ax[3].axvline(beh['t'][s], color='g', lw=0.8)
        ax[3].axvline(beh['t'][e], color='r', lw=0.8, ls='--')
        lo, hi = REWARD_ZONES[res['zones'][i]]
        ax[3].fill_between([beh['t'][s], beh['t'][e - 1]], lo, hi, color='gold', alpha=.4)
    ax[3].set_ylabel('position (cm)\n+reward zone')
    ax[4].plot(tt, beh['speed'][s0:s1], 'b', lw=1)
    ax[4].axhline(2, color='k', ls=':')
    ax[4].set_ylabel('speed (cm/s)')
    ax[5].plot(tt, beh['lick'][s0:s1], 'm', lw=1)
    ax[5].plot(tt, beh['reward'][s0:s1] * 5, 'c', lw=1)
    ax[5].set_ylabel('lick count / reward')
    # distance to zone and its binning
    dist_all = np.full(len(beh['position']), np.nan)
    for i, (s, e) in enumerate(zip(starts, stops)):
        dist_all[s:e] = signed_distance_to_zone(beh['position'][s:e],
                                                REWARD_ZONES[res['zones'][i]])
    ax[6].plot(tt, dist_all[s0:s1], 'k', lw=1)
    ax6b = ax[6].twinx()
    db = np.full(len(tt), np.nan)
    m = ~np.isnan(dist_all[s0:s1])
    db[m] = bin_distance(dist_all[s0:s1][m])
    ax6b.plot(tt, db, 'r.', ms=2)
    ax6b.set_ylabel('distance bin (red)')
    ax[6].set_ylabel('distance to reward zone (cm)')
    ax[6].set_xlabel('time (s)')
    fig.tight_layout()
    fig.savefig(f'processing_{sid}.png', dpi=110)
    plt.close(fig)

    # --- second figure: converted trial matrices + discretization checks
    fig, ax = plt.subplots(4, 3, figsize=(18, 12))
    for j, tr in enumerate([0, 1, min(len(res['neural']) - 1, 40)]):
        act = res['neural'][tr]
        inp = res['input'][tr]
        out = res['output'][tr]
        ax[0, j].imshow(act[:min(80, act.shape[0])], aspect='auto',
                        interpolation='nearest', cmap='magma')
        ax[0, j].set_title(f'trial {tr} (orig {res["kept_trials"][tr]}): events')
        for k in range(inp.shape[0]):
            ax[1, j].plot(inp[k], label=INPUT_NAMES[k])
        ax[1, j].legend(fontsize=6)
        ax[1, j].set_title('inputs')
        for k in range(out.shape[0]):
            ax[2, j].step(np.arange(out.shape[1]), out[k] + 8 * k, where='post',
                          label=OUTPUT_NAMES[k])
        ax[2, j].legend(fontsize=6)
        ax[2, j].set_title('outputs (offset by 8)')
        # verify discretization: position vs position bin
        s, e = starts[res['kept_trials'][tr]], stops[res['kept_trials'][tr]]
        ax[3, j].plot(beh['position'][s:e], 'k', lw=1, label='pos (cm)')
        ax[3, j].plot(out[1] * 90, 'r--', lw=1, label='pos bin x 90')
        z = REWARD_ZONES[res['zones'][res['kept_trials'][tr]]]
        ax[3, j].axhspan(z[0], z[1], color='gold', alpha=.3)
        ax[3, j].plot(np.where(out[0] == 3, 450, np.nan), 'g.', ms=3,
                      label='in-zone bin')
        ax[3, j].legend(fontsize=6)
        ax[3, j].set_xlabel('frame in trial')
    fig.suptitle(f'{sid} converted trials')
    fig.tight_layout()
    fig.savefig(f'processing_{sid}_trials.png', dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- driver
def _worker(args):
    fname, show, neural_signal = args
    warnings.simplefilter('ignore', category=RuntimeWarning)
    try:
        return convert_session(fname, show_processing=show, neural_signal=neural_signal)
    except Exception as exc:  # keep going, but report loudly
        import traceback
        print(f'  !! FAILED {fname}: {exc}')
        traceback.print_exc()
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true',
                    help='process only 2 sessions (for testing)')
    ap.add_argument('--show-processing', action='store_true',
                    help='save per-step processing plots for up to 2 sessions')
    ap.add_argument('--nproc', type=int, default=8)
    ap.add_argument('--neural-signal', choices=['events', 'dff'], default='dff',
                    help="neural stream to save: 'events' = deconvolved dF/F (paper's "
                         "default for population analyses), 'dff' = dF/F")
    ap.add_argument('--sessions', type=str, default=None,
                    help='comma-separated substrings selecting specific sessions')
    ap.add_argument('--data-dir', default='/app/data')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
    if args.sessions:
        pats = args.sessions.split(',')
        files = [f for f in files if any(p in f for p in pats)]
    if args.sample:
        # one small single-plane session and one large two-plane session
        pick = [f for f in files if 'sub-m11_ses-03' in f or 'sub-m17_ses-08' in f]
        files = pick if len(pick) == 2 else files[:2]
    print(f'Converting {len(files)} sessions with {args.nproc} workers')

    show_n = 2 if args.show_processing else 0
    jobs = [(f, i < show_n, args.neural_signal) for i, f in enumerate(files)]

    t0 = time.time()
    results = []
    if args.nproc > 1 and len(files) > 1:
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.nproc, maxtasksperchild=1) as pool:
            for i, res in enumerate(pool.imap(_worker, jobs)):
                if res is not None:
                    results.append(res)
                el = time.time() - t0
                print(f'[{i+1}/{len(files)}] elapsed {el/60:.1f} min, '
                      f'projected total {el/(i+1)*len(files)/60:.1f} min', flush=True)
    else:
        for i, job in enumerate(jobs):
            res = _worker(job)
            if res is not None:
                results.append(res)
            el = time.time() - t0
            print(f'[{i+1}/{len(files)}] elapsed {el/60:.1f} min, '
                  f'projected total {el/(i+1)*len(files)/60:.1f} min', flush=True)

    print(f'Converted {len(results)} sessions in {(time.time()-t0)/60:.1f} min')

    # ---- assemble the target structure
    results.sort(key=lambda r: (r['subject'], r['session_id']))
    subjects = sorted({r['subject'] for r in results},
                      key=lambda s: int(re.sub(r'\D', '', s)))
    brain_regions = ['CA1']

    data = {
        'neural': [r['neural'] for r in results],
        'input': [r['input'] for r in results],
        'output': [r['output'] for r in results],
        'subjects': subjects,
        'subject_idx': np.array([subjects.index(r['subject']) for r in results],
                                dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': [np.zeros(r['n_neurons'], dtype=np.int64) for r in results],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description': (
                'Head-fixed mice run laps on a 450 cm virtual linear track with a hidden '
                '50 cm reward zone at one of three locations (A 80-130, B 200-250, '
                'C 320-370 cm) in one of two visually distinct environments (ENV1/ENV2). '
                'Reward is delivered operantly for licking in the zone and randomly '
                'omitted on ~15% of trials; on switch days the zone (and on day 8/14 the '
                'environment) changes after 30 trials. Decoder outputs: distance to the '
                'reward zone, absolute track position, running speed, licking, reward zone '
                'identity and trial reward outcome. Neural data: deconvolved dF/F '
                '(OASIS) of CA1 pyramidal cells imaged with 2-photon calcium imaging.'),
            'time_bin_size': 1000.0 / FRAME_RATE,          # ms (64.48 ms imaging frame)
            'temporal_alignment_event': (
                'start of trial = entry into the virtual linear track at position 0 cm '
                '(NWB behavior `trial_start` event); trials end at the `teleport` event '
                '(entry into the inter-trial teleport period), which is excluded'),
            'off_start': 0.0,
            'off_end': None,   # trials have variable length (self-paced laps)
            'neural_signal': (
                ('dF/F: neuropil-subtracted (0.7 x Fneu), per-trial maximin baseline '
                 'over a 20 s sliding window, (F - baseline)/|baseline|, smoothed with '
                 'a 2-sample (~0.129 s) s.d. Gaussian kernel (Sosa et al. Methods, '
                 'reward_relative.preprocessing.dff)')
                if args.neural_signal == 'dff' else
                ('deconvolved dF/F ("events"): the dF/F above, deconvolved per trial '
                 'with OASIS (suite2p dcnv.oasis, tau = 0.7)')),
            'imaging_rate_hz': FRAME_RATE,
            'neural_signal_type': args.neural_signal,
            'dataset': 'DANDI:001361 (Sosa, Plitt & Giocomo 2025, Nat Neurosci)',
            'session_info': [
                {'file': r['file'], 'subject': r['subject'], 'day': r['session_id'],
                 'scene': r['scene'], 'n_neurons': r['n_neurons'],
                 'n_interneurons_excluded': r['n_interneurons'],
                 'n_cells_iscell': r['n_roi_iscell'],
                 'n_trials': len(r['neural']), 'n_trials_raw': r['ntrials_raw'],
                 'n_trials_lick_error': r['n_lick_bad'],
                 'reward_rate': float(np.mean(r['isreward'])),
                 'brain_region': r['region'],
                 'planes': int(len(np.unique(r['plane_idx'])))}
                for r in results],
            'curation': {
                'neurons': ('suite2p manual curation (iscell==1) as in the paper, then '
                            'exclusion of putative interneurons with Pearson '
                            'r(dF/F, speed) > 0.5'),
                'trials': ('trials = [trial_start, teleport); trials with lick-sensor '
                           'errors (>30% of frames with cumulative lick count > 2) '
                           'dropped (81 of 12,216 trials, as in the paper)'),
                'samples': ('all within-trial samples kept; the paper additionally '
                            'excludes samples with speed < 2 cm/s for spatial analyses, '
                            'which is not applicable here because the time series must '
                            'stay contiguous and speed is a decoded output'),
            },
        },
    }

    # ---- sanity checks
    nsess = len(data['neural'])
    ntrials = sum(len(x) for x in data['neural'])
    print(f'\nSanity checks: {nsess} sessions, {ntrials} trials, '
          f'{len(subjects)} subjects')
    assert len(data['input']) == nsess and len(data['output']) == nsess
    for i in range(nsess):
        assert len(data['input'][i]) == len(data['neural'][i]) == len(data['output'][i])
        nn = data['neural'][i][0].shape[0]
        for tr in range(len(data['neural'][i])):
            a, ii, oo = (data['neural'][i][tr], data['input'][i][tr],
                         data['output'][i][tr])
            assert a.shape[0] == nn
            assert a.shape[1] == ii.shape[1] == oo.shape[1]
            assert ii.shape[0] == len(INPUT_NAMES) and oo.shape[0] == len(OUTPUT_NAMES)
        assert len(data['brain_region_idx'][i]) == nn
    tot_neurons = sum(r['n_neurons'] for r in results)
    tot_intern = sum(r['n_interneurons'] for r in results)
    tot_lickbad = sum(r['n_lick_bad'] for r in results)
    print(f'  neurons total {tot_neurons} (excluded {tot_intern} putative interneurons, '
          f'{100*tot_intern/(tot_neurons+tot_intern):.2f}%)')
    print(f'  lick-error trials dropped: {tot_lickbad}')
    print(f'  neurons/session: min {min(r["n_neurons"] for r in results)}, '
          f'max {max(r["n_neurons"] for r in results)}, '
          f'mean {tot_neurons/nsess:.1f}')
    tps = [len(x) for x in data['neural']]
    print(f'  trials/session: min {min(tps)}, max {max(tps)}, '
          f'mean {np.mean(tps):.2f}, sd {np.std(tps):.2f}')
    rr = np.concatenate([np.stack([o[5, 0] for o in sess]) for sess in data['output']])
    print(f'  reward rate: {rr.mean():.4f}')
    for k, name in enumerate(OUTPUT_NAMES):
        vals = np.concatenate([np.concatenate([o[k] for o in sess])
                               for sess in data['output']])
        frac = np.bincount(vals, minlength=len(OUTPUT_VALUES[k])) / len(vals)
        print(f'  output {k} {name}: ' + ', '.join(f'{v:.4f}' for v in frac))
    for k, name in enumerate(INPUT_NAMES):
        lo = min(sess_tr[k].min() for sess in data['input'] for sess_tr in sess)
        hi = max(sess_tr[k].max() for sess in data['input'] for sess_tr in sess)
        print(f'  input {k} {name}: [{lo:.3f}, {hi:.3f}]')

    t1 = time.time()
    with open(args.outfile, 'wb') as fh:
        pickle.dump(data, fh, protocol=4)
    print(f'Wrote {args.outfile} '
          f'({os.path.getsize(args.outfile)/1e9:.2f} GB) in {time.time()-t1:.1f} s')


if __name__ == '__main__':
    main()
