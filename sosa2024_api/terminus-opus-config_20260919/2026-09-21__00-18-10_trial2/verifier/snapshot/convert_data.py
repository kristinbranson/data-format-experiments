"""
Convert Sosa, Plitt & Giocomo (2025) hippocampal CA1 2-photon VR dataset (DANDI 001361)
into the decoder-ready pickle format.

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Processing follows the reference code (github.com/GiocomoLab/Sosa_et_al_2024):
  * trials = frames between `trial_start` and `teleport` (teleport/ITI excluded)
  * dF/F   = neuropil subtraction (0.7) with per-trial neuropil mean added back,
             maximin baseline (gaussian sigma 15 frames -> 300-frame min -> 300-frame max),
             dF/F = (F - F0) / |F0|, gaussian smoothing with 2-frame s.d.
             (reward_relative/preprocessing.py :: dff)
  * events = OASIS deconvolution of dF/F (suite2p dcnv.oasis, tau = 0.7)
  * cells  = suite2p iscell == 1, minus putative interneurons (corr(dF/F, speed) > 0.5)
  * trials with lick-sensor errors (>30% of frames with cumulative lick count > 2) dropped
"""

import os
import re
import sys
import time
import pickle
import argparse
import warnings
from glob import glob

import numpy as np
import scipy.ndimage as ndi

warnings.filterwarnings('ignore')

DATA_ROOT = '/app/data'

# ---------------------------------------------------------------- constants
NEU_COEF = 0.7            # neuropil coefficient (reference default)
BASELINE_WIN = 300        # frames (~19.3 s ~ 20 s) for maximin min/max filters
BASELINE_SMOOTH = 15      # gaussian sigma (frames) before maximin
DFF_SMOOTH = 2            # gaussian sigma (frames) applied to dF/F
TAU = 0.7                 # GCaMP7f decay constant used by the reference suite2p ops
INTERNEURON_R = 0.5       # dF/F vs speed correlation threshold for interneuron exclusion
LICK_ERROR_FRAC = 0.30    # >30% of frames with cumulative lick > 2 => lick sensor error
LICK_ERROR_COUNT = 2
TRACK_LENGTH = 450.0
REWARD_ZONES = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
ZONE_TO_IDX = {'A': 0, 'B': 1, 'C': 2}
CHANGE_TRIAL = 30         # reward zone switches on trial index 30 (31st trial)

INPUT_NAMES = ['time_from_trial_start_s', 'environment', 'trial_number', 'prev_trial_rewarded']
OUTPUT_NAMES = ['dist_to_reward_zone', 'position', 'speed', 'lick',
                'reward_zone_location', 'reward_outcome']
OUTPUT_VALUES = [
    ['< -50 cm', '-50 to -10 cm', '-10 to 0 cm', '0 cm (in zone)',
     '0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
    ['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm'],
    ['< 2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '> 40 cm/s'],
    ['no lick', 'lick'],
    ['A (80-130 cm)', 'B (200-250 cm)', 'C (320-370 cm)'],
    ['omitted', 'rewarded'],
]


# ---------------------------------------------------------------- helpers
def nansmooth1d(a, sig, axis=-1):
    """Gaussian smoothing that ignores NaNs (reward_relative.utilities.nansmooth)."""
    nan_inds = np.isnan(a)
    a_ = np.where(nan_inds, 0.0, a)
    one = np.ones(a.shape, dtype=a.dtype)
    one[nan_inds] = 0.001
    a_ = ndi.gaussian_filter1d(a_, sig, axis=axis)
    one = ndi.gaussian_filter1d(one, sig, axis=axis)
    return a_ / one


def nansmooth_nd(a, sig):
    """N-d version used by the reference for the baseline (TwoPUtils.utilities.nansmooth)."""
    nan_inds = np.isnan(a)
    a_ = np.where(nan_inds, 0.0, a)
    one = np.ones(a.shape, dtype=a.dtype)
    one[nan_inds] = 0.001
    a_ = ndi.gaussian_filter(a_, sig)
    one = ndi.gaussian_filter(one, sig)
    return a_ / one


def scene_reward_zones(scene, ntrials, change_trial=CHANGE_TRIAL):
    """Per-trial reward zone label from the VR scene name.

    Mirrors reward_relative.behavior.get_reward_zones: fixed-zone scenes keep one
    zone for the whole session; switch scenes (``..._X_to_..._Y``) use zone X for the
    first `change_trial` trials and zone Y afterwards.
    """
    m = re.search(r'^Env\d_Location([ABC])$', scene)
    if m:
        return np.array([m.group(1)] * ntrials)
    m = re.search(r'([ABC])_to_(?:Env\d_)?(?:Location)?([ABC])$', scene)
    if m:
        n0 = min(change_trial, ntrials)
        return np.array([m.group(1)] * n0 + [m.group(2)] * (ntrials - n0))
    raise NotImplementedError(f'Reward zones not defined for scene {scene}')


def empirical_reward_zones(pos, rzone, starts, teles):
    """Zone label per trial inferred from the animal position at reward-zone entry.

    Returns an array with '?' for trials in which the zone was never triggered
    (omission trials).
    """
    starts_cm = np.array([REWARD_ZONES[z][0] for z in 'ABC'])
    labels = []
    for a, b in zip(starts, teles):
        idx = np.where(rzone[a:b] > 0)[0]
        if len(idx) == 0:
            labels.append('?')
        else:
            p = pos[a:b][idx[0]]
            labels.append('ABC'[int(np.argmin(np.abs(starts_cm - p)))])
    return np.array(labels)


def discretize_distance(d):
    """Distance (cm) to the reward zone -> 7 categories (see OUTPUT_VALUES[0])."""
    out = np.full(d.shape, 3, dtype=np.int8)        # 3 == inside the zone (d == 0)
    out[d < -50] = 0
    out[(d >= -50) & (d < -10)] = 1
    out[(d >= -10) & (d < 0)] = 2
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    return out


def discretize_position(pos):
    """Position (cm) -> 5 equal bins spanning the 450 cm track."""
    p = np.clip(pos, 0.0, TRACK_LENGTH - 1e-9)
    return np.floor(p / 90.0).astype(np.int8)


def discretize_speed(speed):
    return np.digitize(speed, [2.0, 10.0, 20.0, 40.0]).astype(np.int8)


def signed_distance_to_zone(pos, zone_start, zone_end):
    """Signed distance (cm) to the nearest point of the reward zone; 0 inside."""
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start
    d[after] = pos[after] - zone_end
    return d


# ---------------------------------------------------------------- loading
def load_session(fn):
    """Load one NWB session with pynwb; returns a dict of raw arrays and metadata."""
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(fn, 'r', load_namespaces=True) as io:
        nwb = io.read()
        subject = nwb.subject.subject_id
        scene = nwb.identifier.split('/')[-1]
        date = nwb.identifier.split('/')[-2]

        beh = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series
        t = np.asarray(beh['position'].timestamps[:], dtype=np.float64)
        pos = np.asarray(beh['position'].data[:], dtype=np.float64)
        speed = np.asarray(beh['speed'].data[:], dtype=np.float64)
        lick = np.asarray(beh['lick'].data[:], dtype=np.float64)
        rzone = np.asarray(beh['reward_zone'].data[:], dtype=np.float64)
        env = np.asarray(beh['environment'].data[:], dtype=np.float64)
        trial_start = np.asarray(beh['trial_start'].data[:], dtype=np.float64)
        teleport = np.asarray(beh['teleport'].data[:], dtype=np.float64)
        reward_t = np.asarray(beh['Reward'].timestamps[:], dtype=np.float64)

        oph = nwb.processing['ophys'].data_interfaces
        Fseries = oph['Fluorescence'].roi_response_series
        Nseries = oph['Neuropil'].roi_response_series
        plane_keys = sorted(Fseries.keys())
        nframes = len(t)
        F = np.concatenate([np.asarray(Fseries[k].data[:nframes, :], dtype=np.float32).T
                            for k in plane_keys], axis=0)
        Fneu = np.concatenate([np.asarray(Nseries[k].data[:nframes, :], dtype=np.float32).T
                               for k in plane_keys], axis=0)
        rate = float(Fseries[plane_keys[0]].rate)
        n_planes = len(plane_keys)

        ps = oph['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
        iscell = np.asarray(ps['iscell'].data[:])[:, 0] == 1
        plane_idx = np.asarray(ps['planeIdx'].data[:]).astype(int)

    starts = np.where(trial_start > 0)[0]
    teles = np.where(teleport > 0)[0]
    assert len(starts) == len(teles) and np.all(teles > starts), f'bad trial indices in {fn}'

    # reward delivery as a binary frame series (Reward has timestamps only)
    reward = np.zeros(nframes, dtype=np.float64)
    if len(reward_t):
        ridx = np.searchsorted(t, reward_t)
        ridx = np.clip(ridx, 0, nframes - 1)
        reward[ridx] = 1.0

    return dict(file=os.path.basename(fn), subject=subject, scene=scene, date=date,
                t=t, pos=pos, speed=speed, lick=lick, rzone=rzone, env=env,
                reward=reward, starts=starts, teles=teles, F=F, Fneu=Fneu,
                iscell=iscell, plane_idx=plane_idx, rate=rate, n_planes=n_planes)


def compute_dff_events(F, Fneu, starts, teles, fs):
    """dF/F and deconvolved events following reward_relative.preprocessing.dff.

    Frames outside of trials stay NaN (the reference's keep_teleports=False).
    """
    from suite2p.extraction import dcnv

    f_ = np.full(F.shape, np.nan, dtype=np.float32)
    fneu_ = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, teles):
        f_[:, s:e] = F[:, s:e]
        fneu_[:, s:e] = Fneu[:, s:e]
    nanmask = ~np.isnan(f_[0, :])

    f_ -= NEU_COEF * fneu_                                  # neuropil subtraction

    flow = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, teles):
        # add back the trial-mean neuropil so dF/F is not divided by tiny numbers
        f_[:, s:e] += NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
        seg = nansmooth_nd(f_[:, s:e], [0, BASELINE_SMOOTH])
        seg = ndi.minimum_filter1d(seg, BASELINE_WIN, axis=-1)
        seg = ndi.maximum_filter1d(seg, BASELINE_WIN, axis=-1)
        flow[:, s:e] = seg

    dff = np.full(F.shape, np.nan, dtype=np.float32)
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])

    events = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, teles):
        dff[:, s:e] = nansmooth1d(dff[:, s:e], DFF_SMOOTH, axis=1)
        events[:, s:e] = dcnv.oasis(np.ascontiguousarray(dff[:, s:e]), 2000, TAU, fs)

    return dff, events


# ---------------------------------------------------------------- per session
def process_session(fn, show_processing=False, plot_dir='/app'):
    t0 = time.time()
    S = load_session(fn)
    t_load = time.time() - t0

    starts, teles = S['starts'], S['teles']
    ntrials_raw = len(starts)
    fs = S['rate'] / S['n_planes']

    # ---- neurons: suite2p curation
    keep = np.where(S['iscell'])[0]
    F = S['F'][keep]
    Fneu = S['Fneu'][keep]

    t0 = time.time()
    dff, events = compute_dff_events(F, Fneu, starts, teles, fs)
    t_dff = time.time() - t0

    # ---- neurons: exclude putative interneurons (corr(dF/F, speed) > 0.5)
    onmask = ~np.isnan(dff[0, :])
    sp = S['speed'][onmask]
    D = dff[:, onmask]
    Dz = D - D.mean(axis=1, keepdims=True)
    spz = sp - sp.mean()
    denom = (np.sqrt((Dz ** 2).sum(axis=1)) * np.sqrt((spz ** 2).sum()))
    corr_speed = np.where(denom > 0, (Dz @ spz) / np.maximum(denom, 1e-12), 0.0)
    is_interneuron = corr_speed > INTERNEURON_R
    cells = np.where(~is_interneuron)[0]
    n_interneurons = int(is_interneuron.sum())

    # ---- per-trial behaviour variables
    pos, speed, lick = S['pos'], S['speed'], S['lick']
    rzone, env, reward, t = S['rzone'], S['env'], S['reward'], S['t']

    zone_labels = scene_reward_zones(S['scene'], ntrials_raw)
    emp = empirical_reward_zones(pos, rzone, starts, teles)
    valid = emp != '?'
    zone_agree = float(np.mean(emp[valid] == zone_labels[valid])) if valid.any() else np.nan
    if zone_agree < 1.0:
        print(f"  WARNING {S['file']}: scene-derived zones disagree with observed zone "
              f"entries on {(1 - zone_agree) * 100:.1f}% of trials -- using observed zones")
        fixed = zone_labels.copy()
        fixed[valid] = emp[valid]
        zone_labels = fixed

    isreward = np.array([bool((reward[a:b] > 0).any() and (rzone[a:b] > 0).any())
                         for a, b in zip(starts, teles)])
    morph = np.array([np.unique(env[a:b])[0] for a, b in zip(starts, teles)])
    assert np.all(np.isin(morph, [0.0, 1.0])), f"unexpected environment values in {S['file']}"

    # lick-sensor error trials (paper: 81/12,376 trials removed)
    lick_err = np.array([(lick[a:b] > LICK_ERROR_COUNT).mean() > LICK_ERROR_FRAC
                         for a, b in zip(starts, teles)])

    # ---- build per-trial arrays
    neural, inputs, outputs = [], [], []
    trial_ids = []
    for i, (a, b) in enumerate(zip(starts, teles)):
        if lick_err[i]:
            continue
        sl = slice(a, b)
        T = b - a
        tt = (t[sl] - t[a]).astype(np.float32)
        zl = zone_labels[i]
        z0, z1 = REWARD_ZONES[zl]
        d = signed_distance_to_zone(pos[sl], z0, z1)

        inp = np.empty((4, T), dtype=np.float32)
        inp[0] = tt
        inp[1] = morph[i]
        inp[2] = i
        inp[3] = 1.0 if i == 0 else float(isreward[i - 1])

        out = np.empty((6, T), dtype=np.int8)
        out[0] = discretize_distance(d)
        out[1] = discretize_position(pos[sl])
        out[2] = discretize_speed(speed[sl])
        out[3] = (lick[sl] > 0).astype(np.int8)
        out[4] = ZONE_TO_IDX[zl]
        out[5] = int(isreward[i])

        neu = events[np.ix_(cells, np.arange(a, b))].astype(np.float32)
        assert not np.isnan(neu).any(), f"NaNs in neural data, trial {i} of {S['file']}"
        neural.append(neu)
        inputs.append(inp)
        outputs.append(out)
        trial_ids.append(i)

    info = dict(file=S['file'], subject=S['subject'], scene=S['scene'], date=S['date'],
                n_trials_raw=ntrials_raw, n_trials=len(neural),
                n_lick_error_trials=int(lick_err.sum()),
                n_rois=int(len(S['iscell'])), n_iscell=int(S['iscell'].sum()),
                n_interneurons=n_interneurons, n_neurons=len(cells),
                n_planes=S['n_planes'], frame_rate=fs,
                frac_rewarded=float(isreward.mean()), zone_agreement=zone_agree,
                zone_labels=''.join(zone_labels), trial_ids=trial_ids,
                t_load=t_load, t_dff=t_dff)

    if show_processing:
        plot_processing(S, dff, events, cells, zone_labels, isreward, lick_err,
                        neural, inputs, outputs, plot_dir)

    return dict(neural=neural, input=inputs, output=outputs,
                subject=S['subject'], n_neurons=len(cells), info=info)


# ---------------------------------------------------------------- plotting
def plot_processing(S, dff, events, cells, zone_labels, isreward, lick_err,
                    neural, inputs, outputs, plot_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sid = S['file'].replace('_behavior+ophys.nwb', '')
    starts, teles = S['starts'], S['teles']
    t = S['t']
    tr0, tr1 = 0, min(6, len(starts))
    a, b = starts[tr0], teles[tr1 - 1]

    fig, axes = plt.subplots(8, 1, figsize=(15, 20), sharex=True)
    ex = cells[:3] if len(cells) >= 3 else cells

    ax = axes[0]
    for c in ex:
        ax.plot(t[a:b], S['F'][np.where(S['iscell'])[0][c], a:b], lw=0.6)
    ax.set_ylabel('raw F')
    ax.set_title(f"{sid} ({S['scene']}) - processing steps, first {tr1} trials")

    ax = axes[1]
    for c in ex:
        ax.plot(t[a:b], S['Fneu'][np.where(S['iscell'])[0][c], a:b], lw=0.6)
    ax.set_ylabel('neuropil')

    ax = axes[2]
    for c in ex:
        ax.plot(t[a:b], dff[c, a:b], lw=0.6)
    ax.set_ylabel('dF/F')

    ax = axes[3]
    for c in ex:
        ax.plot(t[a:b], events[c, a:b], lw=0.6)
    ax.set_ylabel('events\n(OASIS)')

    ax = axes[4]
    ax.plot(t[a:b], S['pos'][a:b], 'k-', lw=1)
    for i in range(tr0, tr1):
        z0, z1 = REWARD_ZONES[zone_labels[i]]
        ax.fill_between([t[starts[i]], t[teles[i] - 1]], z0, z1,
                        color='g' if isreward[i] else 'r', alpha=0.25)
    ax.set_ylabel('position (cm)\nzone shaded')

    ax = axes[5]
    # distance to reward zone and its discretization
    dd = np.full(b - a, np.nan)
    for i in range(tr0, tr1):
        s_, e_ = starts[i], teles[i]
        z0, z1 = REWARD_ZONES[zone_labels[i]]
        dd[s_ - a:e_ - a] = signed_distance_to_zone(S['pos'][s_:e_], z0, z1)
    ax.plot(t[a:b], dd, 'b-', lw=1)
    ax.axhline(0, color='k', ls=':')
    for lev in [-50, -10, 10, 50]:
        ax.axhline(lev, color='gray', ls=':', lw=0.5)
    ax2 = ax.twinx()
    ax2.plot(t[a:b], discretize_distance(dd), 'r.', ms=2)
    ax2.set_ylabel('bin (red)')
    ax.set_ylabel('dist to zone (cm)')

    ax = axes[6]
    ax.plot(t[a:b], S['speed'][a:b], 'k-', lw=1)
    for lev in [2, 10, 20, 40]:
        ax.axhline(lev, color='gray', ls=':', lw=0.5)
    ax2 = ax.twinx()
    ax2.plot(t[a:b], discretize_speed(S['speed'][a:b]), 'r.', ms=2)
    ax2.set_ylabel('speed bin (red)')
    ax.set_ylabel('speed (cm/s)')

    ax = axes[7]
    ax.plot(t[a:b], S['lick'][a:b], 'k-', lw=0.8, label='cum lick/frame')
    ax.plot(t[a:b], (S['lick'][a:b] > 0).astype(float), 'r-', lw=0.8, alpha=0.6, label='lick output')
    ax.plot(t[a:b], S['reward'][a:b], 'g-', lw=1.2, label='reward')
    ax.legend(fontsize=7)
    ax.set_ylabel('licks / reward')
    ax.set_xlabel('session time (s)')

    for ax in axes:
        for i in range(tr0, tr1):
            ax.axvline(t[starts[i]], color='b', lw=0.8, alpha=0.6)
            ax.axvline(t[teles[i] - 1], color='m', lw=0.8, alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'processing_{sid}.png'), dpi=110)
    plt.close(fig)

    # ---- second figure: trial-aligned summary of converted arrays
    ntr = len(neural)
    maxT = max(o.shape[1] for o in outputs)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    titles = ['dist_to_reward_zone', 'position', 'speed', 'lick']
    for k in range(4):
        M = np.full((ntr, maxT), np.nan)
        for i, o in enumerate(outputs):
            M[i, :o.shape[1]] = o[k]
        im = axes[0, k].imshow(M, aspect='auto', interpolation='nearest', origin='lower')
        axes[0, k].set_title(titles[k] + ' (trial x time)')
        axes[0, k].set_xlabel('frame from trial start')
        axes[0, k].set_ylabel('trial')
        plt.colorbar(im, ax=axes[0, k])

    per_trial = np.array([[o[4, 0], o[5, 0]] for o in outputs])
    axes[1, 0].plot(per_trial[:, 0], 'o-', ms=3)
    axes[1, 0].set_title('reward_zone_location (0=A,1=B,2=C)')
    axes[1, 0].set_xlabel('trial')
    axes[1, 1].plot(per_trial[:, 1], 'o-', ms=3)
    axes[1, 1].set_title('reward_outcome')
    axes[1, 1].set_xlabel('trial')
    inp = np.array([[x[1, 0], x[2, 0], x[3, 0]] for x in inputs])
    axes[1, 2].plot(inp[:, 0], 'o-', ms=3, label='environment')
    axes[1, 2].plot(inp[:, 2], 's-', ms=3, label='prev rewarded')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].set_title('per-trial inputs')
    axes[1, 2].set_xlabel('trial')
    mean_act = np.array([n.mean() for n in neural])
    axes[1, 3].plot(mean_act, 'o-', ms=3)
    axes[1, 3].set_title('mean event rate per trial')
    axes[1, 3].set_xlabel('trial')
    fig.suptitle(f'{sid}: converted trial arrays ({ntr} trials, {neural[0].shape[0]} neurons)')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'processing_{sid}_trials.png'), dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- driver
def _worker(args):
    fn, show, plot_dir = args
    try:
        res = process_session(fn, show_processing=show, plot_dir=plot_dir)
    except Exception as e:  # keep going, report at the end
        import traceback
        traceback.print_exc()
        return dict(error=str(e), file=os.path.basename(fn))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='save processing plots for up to 2 sessions')
    ap.add_argument('--nproc', type=int, default=12)
    args = ap.parse_args()

    files = sorted(glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
    if args.sample:
        files = [f for f in files if 'sub-m11_ses-03' in f or 'sub-m3_ses-05' in f]
    print(f'Processing {len(files)} sessions with {args.nproc} workers')

    show_files = set(files[:2]) if args.show_processing else set()
    jobs = [(f, f in show_files, '/app') for f in files]

    t_start = time.time()
    results = []
    if args.nproc > 1 and len(files) > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.nproc, maxtasksperchild=1) as pool:
            for i, res in enumerate(pool.imap(_worker, jobs)):
                results.append(res)
                el = time.time() - t_start
                if 'error' in res:
                    print(f"[{i+1}/{len(files)}] ERROR {res['file']}: {res['error']}")
                else:
                    inf = res['info']
                    print(f"[{i+1}/{len(files)}] {inf['file']} {inf['scene']} "
                          f"trials={inf['n_trials']}/{inf['n_trials_raw']} "
                          f"neurons={inf['n_neurons']} (iscell {inf['n_iscell']}, "
                          f"interneurons {inf['n_interneurons']}) "
                          f"rew={inf['frac_rewarded']:.2f} "
                          f"t_load={inf['t_load']:.1f}s t_dff={inf['t_dff']:.1f}s "
                          f"| elapsed {el/60:.1f} min, eta {el/(i+1)*(len(files)-i-1)/60:.1f} min")
    else:
        for i, job in enumerate(jobs):
            res = _worker(job)
            results.append(res)
            inf = res.get('info')
            if inf:
                print(f"[{i+1}/{len(files)}] {inf['file']} trials={inf['n_trials']} "
                      f"neurons={inf['n_neurons']} t_load={inf['t_load']:.1f}s t_dff={inf['t_dff']:.1f}s")

    errors = [r for r in results if 'error' in r]
    results = [r for r in results if 'error' not in r]
    if errors:
        print('ERRORS in sessions:', [r['file'] for r in errors])

    subjects = sorted({r['subject'] for r in results}, key=lambda s: int(s[1:]))
    subject_idx = np.array([subjects.index(r['subject']) for r in results], dtype=np.int64)

    data = {
        'neural': [r['neural'] for r in results],
        'input': [r['input'] for r in results],
        'output': [r['output'] for r in results],
        'subjects': subjects,
        'subject_idx': subject_idx,
        'brain_regions': ['CA1'],
        'brain_region_idx': [np.zeros(r['n_neurons'], dtype=np.int64) for r in results],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description': (
                'Head-fixed mice run down a 450 cm virtual linear track containing a hidden '
                '50 cm reward zone at one of three locations (A 80-130, B 200-250, C 320-370 cm) '
                'in one of two visual environments (ENV1/ENV2). Sucrose reward is delivered '
                'operantly for licking in the zone and is randomly omitted on ~15% of trials. '
                'On switch days the zone moves to a new location on trial 31. Decoder predicts, '
                'from CA1 deconvolved calcium activity: distance to the reward zone, absolute '
                'track position, running speed, licking, the active reward zone and whether the '
                'trial was rewarded.'),
            'time_bin_size': 1000.0 / (15.5078125),  # ms per imaging frame (per plane)
            'temporal_alignment_event': 'trial start (entry to the linear track at position 0 cm)',
            'off_start': 0.0,
            'off_end': None,
            'trial_end_event': 'teleport (exit from the track into the inter-trial interval)',
            'neural_signal': ('deconvolved calcium activity (OASIS, tau=0.7) computed from '
                              'per-trial maximin dF/F of suite2p ROIs, as in Sosa et al. 2025'),
            'neuron_curation': ('suite2p iscell==1 after manual curation, minus putative '
                                'interneurons with Pearson r(dF/F, speed) > 0.5'),
            'trial_curation': ('trials with lick-sensor errors (>30% of frames with cumulative '
                               'lick count > 2) removed, as in the paper'),
            'imaging_rate_hz': 15.5078125,
            'species': 'Mus musculus',
            'brain_region_detail': 'dorsal hippocampus CA1, GCaMP7f',
            'reward_zones_cm': {k: list(v) for k, v in REWARD_ZONES.items()},
            'track_length_cm': TRACK_LENGTH,
            'source': ('Sosa, Plitt & Giocomo 2025, Nature Neuroscience; DANDI:001361; '
                       'code: github.com/GiocomoLab/Sosa_et_al_2024'),
            'session_info': [r['info'] for r in results],
        },
    }

    # summary
    ntr = [len(r['neural']) for r in results]
    nneu = [r['n_neurons'] for r in results]
    print(f'\nSessions: {len(results)}  subjects: {len(subjects)}')
    print(f'Trials: total {sum(ntr)}, mean {np.mean(ntr):.1f}, min {min(ntr)}, max {max(ntr)}')
    print(f'Neurons: total {sum(nneu)}, mean {np.mean(nneu):.1f}, min {min(nneu)}, max {max(nneu)}')
    print(f'Dropped lick-error trials: {sum(r["info"]["n_lick_error_trials"] for r in results)}')
    print(f'Excluded interneurons: {sum(r["info"]["n_interneurons"] for r in results)}')

    t0 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f'Wrote {args.outfile} ({os.path.getsize(args.outfile)/1e9:.2f} GB) in {time.time()-t0:.1f}s')
    print(f'Total time {(time.time()-t_start)/60:.1f} min')


if __name__ == '__main__':
    main()
