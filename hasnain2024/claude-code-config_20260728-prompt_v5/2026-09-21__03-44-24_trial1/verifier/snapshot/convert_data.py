#!/usr/bin/env python3
"""
Convert Hasnain, Birnbaum et al (Nature Neuroscience 2024) data to decoder format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

import argparse
import glob
import os
import pickle
import sys
import time

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.signal.windows import gaussian
from scipy.stats import mode as scipy_mode

# ============================================================================
# PARAMETERS (matching reference code: WorkingWithDataObjs.m section 2.1)
# ============================================================================
ALIGN_EVENT = 'goCue'
TMIN = -2.5   # seconds before alignment event
TMAX = 2.5    # seconds after alignment event
DT = 1.0/100  # 10 ms bins
LOW_FR = 1.0  # Hz, minimum mean firing rate
MIN_UNITS = 10  # minimum units per session
SMOOTH_WINDOW = 15  # causal Gaussian kernel window
BC_TYPE = 'reflect'  # boundary condition for smoothing

# Quality labels to EXCLUDE (from findClusters.m)
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}

# Time bin edges and centers
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME = EDGES[:-1] + DT / 2
N_TIMEBINS = len(TIME)

# Probe assignments from reference loading scripts
PROBE_MAP = {
    # Ephys_Behavior
    'EKH1_2021-08-07': [2], 'EKH3_2021-08-11': [2],
    'JEB6_2021-04-18': [2], 'JEB7_2021-04-29': [1], 'JEB7_2021-04-30': [1],
    'JGR2_2021-11-16': [1], 'JGR2_2021-11-17': [1], 'JGR3_2021-11-18': [1],
    'JEB13_2022-09-13': [2], 'JEB13_2022-09-14': [2],
    'JEB13_2022-09-21': [1], 'JEB13_2022-09-24': [1], 'JEB13_2022-09-25': [1],
    'JEB14_2022-08-22': [1], 'JEB14_2022-08-23': [1],
    'JEB14_2022-08-24': [1], 'JEB14_2022-08-25': [1],
    'JEB15_2022-07-26': [1, 2], 'JEB15_2022-07-27': [1, 2],
    'JEB15_2022-07-28': [1, 2], 'JEB15_2022-07-29': [2],
    'JEB19_2023-04-18': [1], 'JEB19_2023-04-19': [1],
    'JEB19_2023-04-20': [1], 'JEB19_2023-04-21': [1],
    # RandomizedDelay_Ephys_Behavior
    'JEB11_2022-05-10': [1], 'JEB11_2022-05-11': [1],
    'JEB12_2022-05-12': [1], 'JEB12_2022-05-13': [1],
    'JEB23_2023-10-10': [1], 'JEB23_2023-10-11': [1],
    'JEB23_2023-10-12': [1], 'JEB23_2023-10-13': [1],
    'JEB23_2023-10-18': [1], 'JEB23_2023-10-19': [1],
    'JEB23_2023-10-21': [1],
    'JEB24_2023-10-23': [1], 'JEB24_2023-10-24': [1],
    'JEB24_2023-10-25': [1], 'JEB24_2023-10-26': [1],
    'JEB24_2023-10-27': [1], 'JEB24_2023-10-31': [1],
    'JEB24_2023-11-02': [1], 'JEB24_2023-11-03': [1],
}


# ============================================================================
# SMOOTHING (matching mySmooth.m: causal Gaussian kernel)
# ============================================================================
def make_causal_gaussian_kernel(N):
    """Create causal Gaussian kernel matching mySmooth.m"""
    kern = gaussian(N, std=(N-1)/(2*2.5))  # MATLAB gausswin(N) default alpha=2.5
    kern[:N//2] = 0  # causal: zero out first half
    kern = kern / kern.sum()
    return kern

SMOOTH_KERNEL = make_causal_gaussian_kernel(SMOOTH_WINDOW)


def smooth_signal(x, kernel=SMOOTH_KERNEL, bc_type=BC_TYPE):
    """Smooth signal with causal Gaussian kernel, matching mySmooth.m"""
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    else:
        squeeze = False

    N = len(kernel)
    if bc_type == 'reflect':
        x_padded = np.concatenate([x[:N], x], axis=0)
        trim = N
    elif bc_type == 'zeropad':
        x_padded = np.concatenate([np.zeros((N, x.shape[1])), x], axis=0)
        trim = N
    else:
        x_padded = x
        trim = 0

    out = np.zeros_like(x_padded)
    for j in range(x_padded.shape[1]):
        out[:, j] = np.convolve(x_padded[:, j], kernel, mode='same')
    out = out[trim:]

    if squeeze:
        out = out[:, 0]
    return out


# ============================================================================
# DATA LOADING HELPERS
# ============================================================================
def is_h5_format(filepath):
    """Check if file is HDF5/MATLAB v7.3 format"""
    with open(filepath, 'rb') as f:
        header = f.read(20).decode('ascii', errors='ignore')
    return '7.3' in header


class H5Session:
    """Wrapper for HDF5 (v7.3) format .mat files"""
    def __init__(self, filepath):
        self.f = h5py.File(filepath, 'r')
        self.obj = self.f['obj']

    def close(self):
        self.f.close()

    def _deref(self, ref):
        return self.f[ref]

    def _get_string(self, ref):
        arr = np.array(self.f[ref]).flatten()
        return ''.join([chr(int(c)) for c in arr]).strip()

    def get_ntrials(self):
        return int(np.array(self.obj['bp']['Ntrials']).flatten()[0])

    def get_bp_field(self, name):
        return np.array(self.obj['bp'][name]).flatten()

    def get_ev_field(self, name):
        return np.array(self.obj['bp']['ev'][name]).flatten()

    def has_stim(self):
        return 'stim' in self.obj['bp']

    def get_stim_enable(self):
        if self.has_stim():
            return np.array(self.obj['bp']['stim']['enable']).flatten()
        return np.zeros(self.get_ntrials())

    def get_early(self):
        return self.get_bp_field('early')

    def get_n_probes(self):
        return self.obj['clu'].shape[0]

    def get_probe_units(self, probe_idx):
        """Get unit data for a probe (0-indexed)"""
        clu_ref = self.obj['clu'][probe_idx, 0]
        clu = self.f[clu_ref]
        n_units = clu['quality'].shape[0]
        units = []
        for i in range(n_units):
            quality = self._get_string(clu['quality'][i, 0]).lower()
            if quality in EXCLUDE_QUALITIES:
                continue
            trialtm = np.array(self.f[clu['trialtm'][i, 0]]).flatten()
            trial = np.array(self.f[clu['trial'][i, 0]]).flatten().astype(int)
            units.append({
                'quality': quality,
                'trialtm': trialtm,
                'trial': trial,
            })
        return units

    def get_sglx_fs(self):
        return float(np.array(self.obj['sglx']['fs']).flatten()[0])

    def get_bitcode_bitstart(self):
        return np.array(self.obj['sglx']['bitcode']['bitstart']).flatten()

    def get_video_offset(self):
        """Compute video offset matching findVideoOffset.m: mode, not median"""
        try:
            fs = self.get_sglx_fs()
            bitstart_sglx = self.get_bitcode_bitstart()
            bitstart_ev = self.get_ev_field('bitStart')
            vidshift = scipy_mode(bitstart_sglx, keepdims=False).mode / fs - scipy_mode(bitstart_ev, keepdims=False).mode
            return vidshift
        except Exception:
            return 0.5  # default

    def get_traj_data(self, view, trial_idx):
        """Get trajectory data for a specific view and trial"""
        traj_ref = self.obj['traj'][view, 0]
        traj = self.f[traj_ref]

        # Get feature names (from first trial)
        feat_ref = traj['featNames'][0, 0]
        feat_ds = self.f[feat_ref]
        feat_names = []
        for j in range(feat_ds.shape[1]):
            ref = feat_ds[0, j]
            feat_names.append(self._get_string(ref).lower())

        # Get ts, frameTimes for trial
        ts_ref = traj['ts'][trial_idx, 0]
        ts = np.array(self.f[ts_ref])
        # h5py transposes: (n_feats, 3, n_frames) -> need (n_frames, 3, n_feats)
        ts = ts.transpose(2, 1, 0)

        ft_ref = traj['frameTimes'][trial_idx, 0]
        frame_times = np.array(self.f[ft_ref]).flatten()

        # Check NdroppedFrames
        ndf_ref = traj['NdroppedFrames'][trial_idx, 0]
        ndf = np.array(self.f[ndf_ref]).flatten()
        is_valid = not np.isnan(ndf[0]) if len(ndf) > 0 else True

        return feat_names, ts, frame_times, is_valid

    def get_n_traj_trials(self, view):
        traj_ref = self.obj['traj'][view, 0]
        traj = self.f[traj_ref]
        return traj['ts'].shape[0]


class V5Session:
    """Wrapper for v5 format .mat files"""
    def __init__(self, filepath):
        self.data = sio.loadmat(filepath, squeeze_me=False)
        self.obj = self.data['obj'][0, 0]

    def close(self):
        pass

    def get_ntrials(self):
        return int(self.obj['bp'][0, 0]['Ntrials'].flatten()[0])

    def get_bp_field(self, name):
        return self.obj['bp'][0, 0][name].flatten().astype(float)

    def get_ev_field(self, name):
        return self.obj['bp'][0, 0]['ev'][0, 0][name].flatten().astype(float)

    def has_stim(self):
        bp = self.obj['bp'][0, 0]
        return 'stim' in bp.dtype.names

    def get_stim_enable(self):
        if self.has_stim():
            return self.obj['bp'][0, 0]['stim'][0, 0]['enable'].flatten().astype(float)
        return np.zeros(self.get_ntrials())

    def get_early(self):
        return self.get_bp_field('early')

    def get_n_probes(self):
        return self.obj['clu'].shape[1]

    def get_probe_units(self, probe_idx):
        """Get unit data for a probe (0-indexed)"""
        probe_data = self.obj['clu'][0, probe_idx]
        n_units = probe_data.shape[1]
        units = []
        for i in range(n_units):
            unit = probe_data[0, i]
            quality = str(unit['quality'][0]).strip().lower()
            if quality in EXCLUDE_QUALITIES:
                continue
            trialtm = unit['trialtm'].flatten()
            trial = unit['trial'].flatten().astype(int)
            units.append({
                'quality': quality,
                'trialtm': trialtm,
                'trial': trial,
            })
        return units

    def get_sglx_fs(self):
        return float(self.obj['sglx'][0, 0]['fs'].flatten()[0])

    def get_bitcode_bitstart(self):
        return self.obj['sglx'][0, 0]['bitcode'][0, 0]['bitstart'].flatten().astype(float)

    def get_video_offset(self):
        """Compute video offset matching findVideoOffset.m: mode, not median"""
        try:
            fs = self.get_sglx_fs()
            bitstart_sglx = self.get_bitcode_bitstart()
            bitstart_ev = self.get_ev_field('bitStart')
            vidshift = scipy_mode(bitstart_sglx, keepdims=False).mode / fs - scipy_mode(bitstart_ev, keepdims=False).mode
            return vidshift
        except Exception:
            return 0.5

    def get_traj_data(self, view, trial_idx):
        """Get trajectory data for a specific view and trial"""
        traj_view = self.obj['traj'][0, 0]  # (2,1) cell -> (nViews,)
        # Actually for v5 with squeeze_me=False: traj is (1,1) object array
        # Each element is struct array (nTrials,) with fields
        traj = self.obj['traj']

        # traj shape varies: could be (2,1) or nested
        try:
            if traj.shape == (2, 1):
                traj_v = traj[view, 0]
            elif traj.shape == (1, 2):
                traj_v = traj[0, view]
            else:
                traj_v = traj[0, 0][view, 0]
        except Exception:
            traj_v = traj[view, 0]

        trial_data = traj_v[0, trial_idx] if traj_v.ndim == 2 else traj_v[trial_idx]

        # Feature names
        feat_names_raw = trial_data['featNames']
        if feat_names_raw.ndim == 0:
            feat_names_raw = feat_names_raw.item()
        feat_names = []
        if hasattr(feat_names_raw, 'flatten'):
            for fn in feat_names_raw.flatten():
                if hasattr(fn, 'item'):
                    fn = fn.item()
                feat_names.append(str(fn).strip().lower())
        else:
            feat_names.append(str(feat_names_raw).strip().lower())

        # ts
        ts = trial_data['ts']
        if ts.ndim == 0:
            ts = ts.item()
        if isinstance(ts, np.ndarray) and ts.ndim < 3:
            if ts.ndim == 2:
                ts = ts[:, :, np.newaxis]

        # frameTimes
        ft = trial_data['frameTimes']
        if ft.ndim == 0:
            ft = ft.item()
        frame_times = np.array(ft).flatten()

        # NdroppedFrames
        try:
            ndf = trial_data['NdroppedFrames']
            if ndf.ndim == 0:
                ndf = ndf.item()
            ndf_val = np.array(ndf).flatten()
            is_valid = not np.isnan(ndf_val[0]) if len(ndf_val) > 0 else True
        except Exception:
            is_valid = True

        return feat_names, ts, frame_times, is_valid

    def get_n_traj_trials(self, view):
        traj = self.obj['traj']
        try:
            if traj.shape == (2, 1):
                traj_v = traj[view, 0]
            elif traj.shape == (1, 2):
                traj_v = traj[0, view]
            else:
                traj_v = traj[0, 0][view, 0]
        except Exception:
            traj_v = traj[view, 0]
        if traj_v.ndim == 2:
            return traj_v.shape[1]
        return traj_v.shape[0]


def open_session(filepath):
    """Open a session file, returning appropriate wrapper"""
    if is_h5_format(filepath):
        return H5Session(filepath)
    else:
        return V5Session(filepath)


# ============================================================================
# MOTION ENERGY LOADING
# ============================================================================
def load_motion_energy(me_filepath):
    """Load motion energy file. Handles multiple formats:
    - HDF5 v7.3: me.data is cell array of refs
    - v5 struct: me.data.data is (nTrials,1) object array (nested struct)
    - v5 plain: me is (nTrials,1) object array (no struct, no moveThresh)
    """
    try:
        if is_h5_format(me_filepath):
            f = h5py.File(me_filepath, 'r')
            me = f['me']
            data_ds = me['data']
            n_trials = data_ds.shape[1]
            me_data = []
            for i in range(n_trials):
                ref = data_ds[0, i]
                d = np.array(f[ref]).flatten()
                me_data.append(d)
            thresh = float(np.array(me['moveThresh']).flatten()[0])
            f.close()
        else:
            d = sio.loadmat(me_filepath, squeeze_me=False)
            me_raw = d['me']

            if me_raw.dtype.names and 'data' in me_raw.dtype.names:
                # Struct format: me.data may be nested struct with its own 'data' field
                inner = me_raw[0, 0]['data']
                if hasattr(inner, 'dtype') and inner.dtype.names and 'data' in inner.dtype.names:
                    # Nested: me.data.data is (nTrials,1) object array
                    data_arr = inner[0, 0]['data']
                    me_data = [np.array(data_arr[i, 0]).flatten() for i in range(data_arr.shape[0])]
                    thresh = float(inner[0, 0]['moveThresh'].flatten()[0])
                else:
                    # Simple struct: me.data is (nTrials,1) object array
                    if inner.ndim == 2:
                        me_data = [np.array(inner[i, 0]).flatten() for i in range(inner.shape[0])]
                    else:
                        me_data = [np.array(x).flatten() for x in inner.flatten()]
                    thresh = float(me_raw[0, 0]['moveThresh'].flatten()[0])
            elif me_raw.dtype == object:
                # Plain cell array: me is (nTrials,1) object array, no moveThresh
                me_data = [np.array(me_raw[i, 0]).flatten() for i in range(me_raw.shape[0])]
                thresh = None  # not available
            else:
                raise ValueError(f"Unexpected ME format: dtype={me_raw.dtype}, shape={me_raw.shape}")

        return me_data, thresh
    except Exception as e:
        print(f"  Warning: Failed to load motion energy: {e}")
        return None, None


# ============================================================================
# CORE PROCESSING
# ============================================================================
def process_session(session_key, data_filepath, me_filepath, probes, show_processing=False):
    """Process a single session, returning neural + behavioral data."""
    t0 = time.time()
    anm = session_key.split('_')[0]
    date = session_key.split('_')[1]

    print(f"\nProcessing {session_key} (probes={probes})...")
    sess = open_session(data_filepath)

    n_trials = sess.get_ntrials()
    print(f"  Trials: {n_trials}")

    # ---- Get behavioral variables ----
    hit = sess.get_bp_field('hit')
    miss = sess.get_bp_field('miss')
    no = sess.get_bp_field('no')
    early = sess.get_early()
    autowater = sess.get_bp_field('autowater')
    R = sess.get_bp_field('R')
    L = sess.get_bp_field('L')
    stim_enable = sess.get_stim_enable()
    goCue = sess.get_ev_field('goCue')

    # ---- Determine valid trials ----
    # Exclude stim trials (photoinactivation disrupts neural activity)
    # Exclude early lick trials per paper/code conventions
    # The condition strings in reference code: '~stim.enable&~early'
    valid_mask = (stim_enable == 0) & (early == 0)
    valid_trials = np.where(valid_mask)[0]  # 0-indexed

    if len(valid_trials) < 2:
        print(f"  SKIP: Only {len(valid_trials)} valid trials")
        sess.close()
        return None

    print(f"  Valid trials (non-stim, non-early): {len(valid_trials)} / {n_trials}")

    # ---- Collect units from specified probes ----
    all_units = []
    for p in probes:
        units = sess.get_probe_units(p - 1)  # 0-indexed
        all_units.extend(units)

    print(f"  Units (quality-filtered): {len(all_units)}")

    if len(all_units) < MIN_UNITS:
        print(f"  SKIP: Only {len(all_units)} units (min={MIN_UNITS})")
        sess.close()
        return None

    # ---- Align spikes and bin ----
    # Matching alignSpikes.m: trialtm_aligned = trialtm - goCue(trial)
    # Matching getSeq.m: histc into edges, divide by dt, smooth
    n_units = len(all_units)
    # Single-trial data: (n_timebins, n_units, n_trials)
    trialdat = np.zeros((N_TIMEBINS, n_units, n_trials), dtype=np.float32)

    for i, unit in enumerate(all_units):
        trialtm = unit['trialtm']
        trial = unit['trial']

        for j in range(n_trials):
            trial_num = j + 1  # 1-indexed trial number
            spk_mask = trial == trial_num
            if not np.any(spk_mask):
                continue

            # Align to goCue
            spk_times = trialtm[spk_mask] - goCue[j]

            # Bin spikes
            counts, _ = np.histogram(spk_times, bins=EDGES)

            # Convert to firing rate and smooth
            rate = counts.astype(np.float32) / DT
            trialdat[:, i, j] = smooth_signal(rate)

    # ---- Remove low FR units (matching removeLowFRClusters.m) ----
    # Mean FR = mean across all conditions and time of trial-averaged PSTH
    # In reference code: meanFRs = mean(mean(obj.psth{prbnum},3,'omitnan'),'omitnan')
    # psth is (time, units, conditions). mean over conditions, then mean over time
    # For our case, we compute mean FR from single trial data of all valid trials
    mean_fr = np.mean(np.mean(trialdat[:, :, valid_trials], axis=2), axis=0)
    keep_units = mean_fr > LOW_FR

    n_kept = np.sum(keep_units)
    print(f"  Units after FR filter (>{LOW_FR} Hz): {n_kept}")

    if n_kept < MIN_UNITS:
        print(f"  SKIP: Only {n_kept} units after FR filter (min={MIN_UNITS})")
        sess.close()
        return None

    trialdat = trialdat[:, keep_units, :]

    # ---- Exclude trials with no neural data (recording ended early) ----
    # Some sessions have behavioral trials beyond the end of ephys recording
    trial_has_spikes = np.any(trialdat > 0, axis=(0, 1))
    no_spike_trials = np.where(~trial_has_spikes)[0]
    if len(no_spike_trials) > 0:
        valid_before = len(valid_trials)
        valid_trials = np.array([t for t in valid_trials if trial_has_spikes[t]])
        if len(valid_trials) < valid_before:
            print(f"  Excluded {valid_before - len(valid_trials)} trials with no neural data (recording ended early)")

    # ---- Extract only valid trials ----
    # Now subset to valid (non-stim, non-early) trials
    trialdat_valid = trialdat[:, :, valid_trials]  # (time, units, n_valid)

    # ---- Per-trial outputs ----
    # Lick direction: 0=left, 1=right, 2=none
    lick_dir = np.full(len(valid_trials), 2, dtype=int)  # default: none
    for vi, ti in enumerate(valid_trials):
        if L[ti] == 1:
            if hit[ti] == 1:
                lick_dir[vi] = 0  # left lick, correct
            elif miss[ti] == 1:
                lick_dir[vi] = 0  # left trial, animal licked wrong side - but L means it was a left trial
                # Actually: L=1 means it's a LEFT trial (stimulus side). hit means correct response.
                # For lick direction, we need to determine which side the animal actually licked.
                # R/L indicate trial type (stimulus side), not lick direction.
                # hit = correct lick (same as stimulus), miss = wrong lick (opposite side)
                pass  # Will redo below

    # Re-do lick direction properly:
    # R=1 means right trial (correct answer is right)
    # L=1 means left trial (correct answer is left)
    # hit=1 means correct response
    # miss=1 means incorrect response (licked wrong side)
    # no=1 means no response (ignore)
    # So actual lick direction:
    # - R trial + hit -> licked right
    # - R trial + miss -> licked left (wrong side)
    # - L trial + hit -> licked left
    # - L trial + miss -> licked right (wrong side)
    # - no -> no lick
    lick_dir = np.full(len(valid_trials), 2, dtype=int)  # default: none (no lick)
    for vi, ti in enumerate(valid_trials):
        if no[ti] == 1:
            lick_dir[vi] = 2  # no response
        elif R[ti] == 1 and hit[ti] == 1:
            lick_dir[vi] = 1  # right lick
        elif R[ti] == 1 and miss[ti] == 1:
            lick_dir[vi] = 0  # left lick (error on right trial)
        elif L[ti] == 1 and hit[ti] == 1:
            lick_dir[vi] = 0  # left lick
        elif L[ti] == 1 and miss[ti] == 1:
            lick_dir[vi] = 1  # right lick (error on left trial)
        # WC trials: autowater=1, hit=1 means consumed water
        # For WC: R/L still indicates which port had water

    # Behavioral context: 0=WC, 1=DR
    context = np.array([1 if autowater[ti] == 0 else 0 for ti in valid_trials], dtype=int)

    # Outcome: 0=incorrect, 1=correct, 2=ignore
    outcome = np.full(len(valid_trials), 2, dtype=int)
    for vi, ti in enumerate(valid_trials):
        if hit[ti] == 1:
            outcome[vi] = 1  # correct
        elif miss[ti] == 1:
            outcome[vi] = 0  # incorrect
        elif no[ti] == 1:
            outcome[vi] = 2  # ignore

    # ---- Video-based outputs (tongue velocity, paw velocity, motion energy) ----
    vidshift = sess.get_video_offset()
    print(f"  Video offset: {vidshift:.4f}s")

    taxis = TIME.copy()

    # Initialize arrays
    tongue_vel = np.full((N_TIMEBINS, len(valid_trials)), np.nan, dtype=np.float32)
    tongue_visible = np.zeros((N_TIMEBINS, len(valid_trials)), dtype=bool)
    paw_vel = np.full((N_TIMEBINS, len(valid_trials)), np.nan, dtype=np.float32)
    paw_visible = np.zeros((N_TIMEBINS, len(valid_trials)), dtype=bool)

    try:
        # Get feature names once
        side_feats, _, _, _ = sess.get_traj_data(0, 0)
        bottom_feats, _, _, _ = sess.get_traj_data(1, 0)

        tongue_idx_side = None
        for fi, fn in enumerate(side_feats):
            if fn == 'tongue':
                tongue_idx_side = fi
                break

        paw_indices_bottom = [fi for fi, fn in enumerate(bottom_feats) if 'paw' in fn]

        n_traj_trials_side = sess.get_n_traj_trials(0)
        n_traj_trials_bottom = sess.get_n_traj_trials(1)

        for vi, ti in enumerate(valid_trials):
            # ---- Tongue velocity from side cam ----
            if tongue_idx_side is not None and ti < n_traj_trials_side:
                try:
                    _, ts, frame_times, is_valid = sess.get_traj_data(0, ti)
                    if is_valid and len(frame_times) > 1 and not np.all(np.isnan(frame_times)):
                        tx = ts[:, 0, tongue_idx_side].astype(float)
                        ty = ts[:, 1, tongue_idx_side].astype(float)
                        vis_raw = ~(np.isnan(tx) | np.isnan(ty))
                        aligned_times = frame_times - vidshift - goCue[ti]

                        # Interpolate visibility mask to neural time
                        vis_interp = np.interp(taxis, aligned_times, vis_raw.astype(float)) > 0.5
                        tongue_visible[:, vi] = vis_interp

                        # For tongue positions: set NaN to baseline position before interp
                        # (matching setTongueBaselinePosition in reference code)
                        if np.any(vis_raw):
                            # Baseline = mean initial tongue position at start of each visible bout
                            baseline_x = np.nanmean(tx[vis_raw])
                            baseline_y = np.nanmean(ty[vis_raw])
                            tx_filled = tx.copy()
                            ty_filled = ty.copy()
                            tx_filled[~vis_raw] = baseline_x
                            ty_filled[~vis_raw] = baseline_y

                            # Interpolate to neural time
                            tx_i = np.interp(taxis, aligned_times, tx_filled)
                            ty_i = np.interp(taxis, aligned_times, ty_filled)

                            # Compute velocity (gradient, matching findVelocity.m)
                            vx = np.gradient(tx_i)
                            vy = np.gradient(ty_i)

                            # Euclidean speed
                            speed = np.sqrt(vx**2 + vy**2)

                            # Set velocity to 0 where tongue not visible
                            # (matching findVelocity.m: tongue velocity = 0 if not visible)
                            speed[~vis_interp] = 0
                            tongue_vel[:, vi] = speed
                except Exception:
                    pass

            # ---- Paw velocity from bottom cam ----
            if len(paw_indices_bottom) > 0 and ti < n_traj_trials_bottom:
                try:
                    _, ts, frame_times, is_valid = sess.get_traj_data(1, ti)
                    if is_valid and len(frame_times) > 1 and not np.all(np.isnan(frame_times)):
                        aligned_times = frame_times - vidshift - goCue[ti]
                        paw_speeds = []
                        paw_vis_list = []

                        for pidx in paw_indices_bottom:
                            px = ts[:, 0, pidx].astype(float)
                            py = ts[:, 1, pidx].astype(float)
                            vis_raw = ~np.isnan(px)

                            if not np.any(vis_raw):
                                continue

                            # Fill NaN with nearest (matching findPosition.m for non-tongue)
                            valid_idx = np.where(vis_raw)[0]
                            px_filled = np.interp(np.arange(len(px)), valid_idx, px[valid_idx])
                            py_filled = np.interp(np.arange(len(py)), valid_idx, py[valid_idx])

                            # Interpolate to neural time
                            px_i = np.interp(taxis, aligned_times, px_filled)
                            py_i = np.interp(taxis, aligned_times, py_filled)

                            # Velocity with baseline subtract (matching findVelocity.m)
                            vx = np.gradient(px_i)
                            vy = np.gradient(py_i)
                            base_vx = np.median(np.diff(px_i))
                            base_vy = np.median(np.diff(py_i))
                            vx = vx - base_vx
                            vy = vy - base_vy

                            # Fill NaN in velocity with nearest (matching findVelocity.m)
                            speed = np.sqrt(vx**2 + vy**2)
                            speed[np.isnan(speed)] = 0

                            paw_speeds.append(speed)

                            # Track visibility at neural time resolution
                            vis_i = np.interp(taxis, aligned_times, vis_raw.astype(float)) > 0.5
                            paw_vis_list.append(vis_i)

                        if paw_speeds:
                            paw_vel[:, vi] = np.mean(paw_speeds, axis=0)
                            paw_visible[:, vi] = np.any(paw_vis_list, axis=0)
                except Exception:
                    pass
    except Exception as e:
        print(f"  Warning: Failed to process video data: {e}")

    # ---- Motion energy ----
    me_data_aligned = np.full((N_TIMEBINS, len(valid_trials)), np.nan, dtype=np.float32)
    me_available = True

    if me_filepath and os.path.exists(me_filepath):
        me_data, me_thresh = load_motion_energy(me_filepath)
        if me_data is not None:
            for vi, ti in enumerate(valid_trials):
                if ti < len(me_data):
                    try:
                        me_trial = np.array(me_data[ti]).flatten().astype(float)
                        if len(me_trial) < 2:
                            continue
                        # Get frame times for this trial from side cam
                        _, _, frame_times, is_valid = sess.get_traj_data(0, ti)
                        if not is_valid or len(frame_times) < 2:
                            continue

                        aligned_times = frame_times - vidshift - goCue[ti]

                        # ME data corresponds 1:1 with frame_times
                        # If lengths differ, use min length
                        n_me = min(len(me_trial), len(aligned_times))

                        # Interpolate ME to neural time (matching loadMotionEnergy.m)
                        me_interp = np.interp(taxis, aligned_times[:n_me], me_trial[:n_me])

                        # Fill NaN with nearest (matching loadMotionEnergy.m)
                        nan_mask = np.isnan(me_interp)
                        if np.any(~nan_mask) and np.any(nan_mask):
                            valid_idx = np.where(~nan_mask)[0]
                            me_interp = np.interp(np.arange(len(me_interp)), valid_idx,
                                                me_interp[valid_idx])
                        me_data_aligned[:, vi] = me_interp
                    except Exception:
                        pass
        else:
            me_available = False
    else:
        me_available = False

    # ---- Discretize continuous outputs ----
    # Tongue velocity: per-session 50th percentile of VISIBLE values
    tongue_vel_disc = np.full((N_TIMEBINS, len(valid_trials)), 2, dtype=int)  # default: not visible
    # Get all visible tongue velocity values for threshold computation
    visible_tongue_vals = tongue_vel[tongue_visible & ~np.isnan(tongue_vel)]
    if len(visible_tongue_vals) > 0:
        thresh_tongue = np.percentile(visible_tongue_vals, 50)
        print(f"  Tongue vel threshold (p50): {thresh_tongue:.3f}, visible fraction: {tongue_visible.mean():.3f}")
        for vi in range(len(valid_trials)):
            vis = tongue_visible[:, vi]
            valid = vis & ~np.isnan(tongue_vel[:, vi])
            tongue_vel_disc[valid, vi] = (tongue_vel[valid, vi] >= thresh_tongue).astype(int)
            tongue_vel_disc[~vis, vi] = 2  # not visible
    else:
        print(f"  Warning: No visible tongue data")

    # Paw velocity: per-session 50th percentile of VISIBLE values
    paw_vel_disc = np.full((N_TIMEBINS, len(valid_trials)), 2, dtype=int)  # default: not visible
    visible_paw_vals = paw_vel[paw_visible & ~np.isnan(paw_vel)]
    if len(visible_paw_vals) > 0:
        thresh_paw = np.percentile(visible_paw_vals, 50)
        print(f"  Paw vel threshold (p50): {thresh_paw:.3f}, visible fraction: {paw_visible.mean():.3f}")
        for vi in range(len(valid_trials)):
            vis = paw_visible[:, vi]
            valid = vis & ~np.isnan(paw_vel[:, vi])
            paw_vel_disc[valid, vi] = (paw_vel[valid, vi] >= thresh_paw).astype(int)
            paw_vel_disc[~vis, vi] = 2  # not visible
    else:
        print(f"  Warning: No visible paw data")

    # Motion energy: per-session 50th percentile
    me_disc = np.full((N_TIMEBINS, len(valid_trials)), 2, dtype=int)  # default: no video
    valid_me = ~np.isnan(me_data_aligned)
    if np.any(valid_me) and me_available:
        me_vals = me_data_aligned[valid_me]
        thresh_me = np.percentile(me_vals, 50)
        print(f"  ME threshold (p50): {thresh_me:.3f}, valid fraction: {valid_me.mean():.3f}")
        for vi in range(len(valid_trials)):
            valid_t = ~np.isnan(me_data_aligned[:, vi])
            me_disc[valid_t, vi] = (me_data_aligned[valid_t, vi] >= thresh_me).astype(int)
            me_disc[~valid_t, vi] = 2
    else:
        print(f"  Warning: No motion energy data")

    # ---- Build trial-level data ----
    neural_trials = []
    input_trials = []
    output_trials = []

    n_valid = len(valid_trials)
    n_neurons = trialdat_valid.shape[1]

    for vi in range(n_valid):
        # Neural: (n_neurons, n_timepoints)
        neural_trials.append(trialdat_valid[:, :, vi].T.astype(np.float32))

        # Input: time from go cue, shape (1, n_timepoints)
        input_trials.append(TIME.reshape(1, -1).astype(np.float32))

        # Output: (n_output, n_timepoints) for time-varying, (n_output,) for per-trial
        # Per-trial outputs: lick_direction, behavioral_context, outcome
        # Time-varying: tongue_velocity, paw_velocity, motion_energy
        out = np.zeros((6, N_TIMEBINS), dtype=int)
        out[0, :] = lick_dir[vi]  # per-trial, broadcast
        out[1, :] = context[vi]   # per-trial, broadcast
        out[2, :] = outcome[vi]   # per-trial, broadcast
        out[3, :] = tongue_vel_disc[:, vi]  # time-varying
        out[4, :] = paw_vel_disc[:, vi]     # time-varying
        out[5, :] = me_disc[:, vi]          # time-varying
        output_trials.append(out)

    elapsed = time.time() - t0
    print(f"  Done: {n_neurons} neurons, {n_valid} trials, {elapsed:.1f}s")

    # ---- Plotting ----
    if show_processing:
        plot_session_processing(session_key, neural_trials, input_trials, output_trials,
                              tongue_vel, paw_vel, me_data_aligned, valid_trials,
                              tongue_vel_disc, paw_vel_disc, me_disc,
                              goCue, hit, miss, no, autowater, R, L)

    sess.close()

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'n_neurons': n_neurons,
        'animal': anm,
        'date': date,
        'session_key': session_key,
        'elapsed': elapsed,
    }


def plot_session_processing(session_key, neural_trials, input_trials, output_trials,
                           tongue_vel, paw_vel, me_data, valid_trials,
                           tongue_disc, paw_disc, me_disc,
                           goCue, hit, miss, no, autowater, R, L):
    """Plot processing visualizations for a session."""
    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    fig.suptitle(f'Processing: {session_key}', fontsize=14)

    n_trials_plot = min(5, len(neural_trials))

    # Row 0: Neural activity heatmap for first trial
    ax = axes[0, 0]
    if neural_trials:
        im = ax.imshow(neural_trials[0], aspect='auto', extent=[TMIN, TMAX, 0, neural_trials[0].shape[0]])
        ax.axvline(0, color='r', linestyle='--', label='Go cue')
        ax.set_title('Neural activity (trial 0)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Neuron')

    # Row 0: Mean firing rate distribution
    ax = axes[0, 1]
    all_fr = [n.mean() for n in neural_trials]
    ax.hist(all_fr, bins=30)
    ax.set_title('Mean FR per trial')
    ax.set_xlabel('FR (Hz)')

    # Row 0: Output distributions
    ax = axes[0, 2]
    out_names = ['lick_dir', 'context', 'outcome']
    for oi, name in enumerate(out_names):
        vals = [output_trials[t][oi, 0] for t in range(len(output_trials))]
        unique, counts = np.unique(vals, return_counts=True)
        ax.bar(unique + oi*3.5, counts/len(vals), label=name)
    ax.set_title('Per-trial output distributions')
    ax.legend(fontsize=8)

    # Row 1: Tongue velocity (raw + discretized)
    ax = axes[1, 0]
    if tongue_vel is not None and not np.all(np.isnan(tongue_vel)):
        for ti in range(min(3, tongue_vel.shape[1])):
            ax.plot(TIME, tongue_vel[:, ti], alpha=0.5)
        ax.axvline(0, color='r', linestyle='--')
        ax.set_title('Tongue velocity (raw)')

    ax = axes[1, 1]
    if tongue_disc is not None:
        for ti in range(min(3, tongue_disc.shape[1])):
            ax.plot(TIME, tongue_disc[:, ti] + ti*0.1, alpha=0.7)
        ax.set_title('Tongue velocity (discretized)')

    # Row 2: Paw velocity
    ax = axes[2, 0]
    if paw_vel is not None and not np.all(np.isnan(paw_vel)):
        for ti in range(min(3, paw_vel.shape[1])):
            ax.plot(TIME, paw_vel[:, ti], alpha=0.5)
        ax.axvline(0, color='r', linestyle='--')
        ax.set_title('Paw velocity (raw)')

    ax = axes[2, 1]
    if paw_disc is not None:
        for ti in range(min(3, paw_disc.shape[1])):
            ax.plot(TIME, paw_disc[:, ti] + ti*0.1, alpha=0.7)
        ax.set_title('Paw velocity (discretized)')

    # Row 3: Motion energy
    ax = axes[3, 0]
    if me_data is not None and not np.all(np.isnan(me_data)):
        for ti in range(min(3, me_data.shape[1])):
            ax.plot(TIME, me_data[:, ti], alpha=0.5)
        ax.axvline(0, color='r', linestyle='--')
        ax.set_title('Motion energy (raw)')

    ax = axes[3, 1]
    if me_disc is not None:
        for ti in range(min(3, me_disc.shape[1])):
            ax.plot(TIME, me_disc[:, ti] + ti*0.1, alpha=0.7)
        ax.set_title('Motion energy (discretized)')

    # Summary stats
    ax = axes[1, 2]
    ax.axis('off')
    stats_text = f"Session: {session_key}\n"
    stats_text += f"Neurons: {neural_trials[0].shape[0] if neural_trials else 0}\n"
    stats_text += f"Trials: {len(neural_trials)}\n"
    lick_vals = [output_trials[t][0, 0] for t in range(len(output_trials))]
    ctx_vals = [output_trials[t][1, 0] for t in range(len(output_trials))]
    out_vals = [output_trials[t][2, 0] for t in range(len(output_trials))]
    stats_text += f"Lick dir: L={lick_vals.count(0)}, R={lick_vals.count(1)}, None={lick_vals.count(2)}\n"
    stats_text += f"Context: WC={ctx_vals.count(0)}, DR={ctx_vals.count(1)}\n"
    stats_text += f"Outcome: err={out_vals.count(0)}, corr={out_vals.count(1)}, ign={out_vals.count(2)}\n"
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='center',
            fontfamily='monospace')

    for ax in [axes[2, 2], axes[3, 2]]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'processing_{session_key}.png', dpi=100)
    plt.close()
    print(f"  Saved processing_{session_key}.png")


# ============================================================================
# MAIN
# ============================================================================
def get_session_list():
    """Get list of all sessions to process with their file paths"""
    sessions = []

    data_dirs = [
        '/app/data/Ephys_Behavior/',
        '/app/data/RandomizedDelay_Ephys_Behavior/',
    ]

    for dpath in data_dirs:
        files = sorted(glob.glob(os.path.join(dpath, 'data_structure_*.mat')))
        for fp in files:
            fn = os.path.basename(fp)
            key = fn.replace('data_structure_', '').replace('.mat', '')

            if key not in PROBE_MAP:
                continue

            # Find motion energy file
            anm = key.split('_')[0]
            date = key.split('_')[1]
            me_fn = f'motionEnergy_{anm}_{date}.mat'
            me_fp = os.path.join(dpath, me_fn)
            if not os.path.exists(me_fp):
                me_fp = None

            sessions.append({
                'key': key,
                'data_path': fp,
                'me_path': me_fp,
                'probes': PROBE_MAP[key],
            })

    return sessions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--show-processing', action='store_true')
    args = parser.parse_args()

    if args.sample:
        args.full = False

    t_start = time.time()

    sessions = get_session_list()
    print(f"Found {len(sessions)} sessions to process")

    if args.sample:
        sessions = sessions[:2]
        print(f"Sample mode: processing {len(sessions)} sessions")

    # Process all sessions
    all_neural = []
    all_input = []
    all_output = []
    all_subjects = []
    subject_idx = []
    brain_region_idx = []
    session_info = []

    for sess_info in sessions:
        result = process_session(
            sess_info['key'],
            sess_info['data_path'],
            sess_info['me_path'],
            sess_info['probes'],
            show_processing=args.show_processing,
        )

        if result is None:
            continue

        all_neural.append(result['neural'])
        all_input.append(result['input'])
        all_output.append(result['output'])

        # Track subjects
        anm = result['animal']
        if anm not in all_subjects:
            all_subjects.append(anm)
        subject_idx.append(all_subjects.index(anm))

        # Brain region: all ALM
        brain_region_idx.append(np.zeros(result['n_neurons'], dtype=int))

        session_info.append({
            'animal': result['animal'],
            'date': result['date'],
            'n_neurons': result['n_neurons'],
            'n_trials': len(result['neural']),
        })

    # Build output dictionary
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': all_subjects,
        'subject_idx': np.array(subject_idx),
        'brain_regions': ['ALM'],
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_go_cue'],
        'output_names': ['lick_direction', 'behavioral_context', 'outcome',
                        'tongue_velocity', 'paw_velocity', 'motion_energy'],
        'output_values': [
            ['left', 'right', 'none'],
            ['WC', 'DR'],
            ['incorrect', 'correct', 'ignore'],
            ['below_p50', 'above_p50', 'not_visible'],
            ['below_p50', 'above_p50', 'not_visible'],
            ['below_p50', 'above_p50', 'no_video'],
        ],
        'metadata': {
            'task_description': 'Two-context task: delayed-response (DR) and water-cued (WC) licking tasks with ALM recordings',
            'time_bin_size': DT * 1000,  # in ms
            'temporal_alignment_event': 'Go cue onset',
            'off_start': TMIN,
            'off_end': TMAX,
            'session_info': session_info,
            'smoothing': f'Causal Gaussian, window={SMOOTH_WINDOW} bins, bc={BC_TYPE}',
            'quality_filter': f'Exclude {EXCLUDE_QUALITIES}',
            'fr_filter': f'Mean FR > {LOW_FR} Hz',
            'trial_filter': 'Exclude stim and early lick trials',
        },
    }

    # Print summary
    total_neurons = sum(si['n_neurons'] for si in session_info)
    total_trials = sum(si['n_trials'] for si in session_info)
    print(f"\n{'='*60}")
    print(f"CONVERSION COMPLETE")
    print(f"Sessions: {len(all_neural)}")
    print(f"Subjects: {len(all_subjects)} ({', '.join(all_subjects)})")
    print(f"Total neurons: {total_neurons}")
    print(f"Total trials: {total_trials}")
    print(f"Time bins: {N_TIMEBINS} ({DT*1000:.0f} ms bins)")
    print(f"Time window: [{TMIN}, {TMAX}] s from go cue")

    # Output distributions (for per-trial outputs use first timepoint, for time-varying use all)
    per_trial_outputs = {0, 1, 2}  # lick_dir, context, outcome
    for oi, oname in enumerate(data['output_names']):
        all_vals = []
        for sess in all_output:
            for trial in sess:
                if oi in per_trial_outputs:
                    all_vals.append(trial[oi, 0])
                else:
                    all_vals.extend(trial[oi, :].tolist())
        vals, counts = np.unique(all_vals, return_counts=True)
        fracs = counts / counts.sum()
        print(f"  {oname}: " + ", ".join([f"{data['output_values'][oi][int(v)]}={f:.3f}" for v, f in zip(vals, fracs)]))

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    fsize = os.path.getsize(args.output) / (1024**2)
    print(f"Saved: {fsize:.1f} MB")


if __name__ == '__main__':
    main()
