"""
Convert neural data from Hasnain, Birnbaum et al. (Nature Neuroscience 2024)
into the standardized decoder format.

Paper: "Separating cognitive and motor processes in the behaving mouse"
Data: ALM recordings during delayed-response (DR) and water-cued (WC) licking tasks.

Processing follows the reference MATLAB code (DataLoadingScripts/).
"""

import numpy as np
import h5py
import scipy.io
import pickle
import os
import sys
import glob
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

# ---- Parameters matching getDefaultParams.m ----
ALIGN_EVENT = 'goCue'
TMIN = -2.5
TMAX = 2.5
DT = 1 / 200  # 5 ms bins
SMOOTH_MS = 15  # causal Gaussian smoothing kernel width in ms
LOW_FR = 0.5  # minimum firing rate threshold (Hz)
# Time axis (center of bins)
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
N_TIMEBINS = len(TIME_AXIS)

DATA_DIRS = [
    '/app/data/Ephys_Behavior',
    '/app/data/RandomizedDelay_Ephys_Behavior',
]


def causal_gaussian_smooth(x, kernel_width_samples):
    """Causal Gaussian smoothing matching mySmooth.m.

    The MATLAB code creates a gausswin(N), zeros the left half (making it causal),
    normalizes, then convolves with 'same'.
    """
    if kernel_width_samples <= 1:
        return x
    N = kernel_width_samples
    # Create Gaussian window
    n = np.arange(N)
    center = (N - 1) / 2
    sigma = N / 6  # MATLAB gausswin default: alpha=2.5, sigma = (N-1)/(2*alpha)
    kern = np.exp(-0.5 * ((n - center) / sigma) ** 2)
    # Make causal: zero out left half
    kern[:N // 2] = 0
    kern = kern / kern.sum()

    if x.ndim == 1:
        return np.convolve(x, kern, mode='same')
    else:
        out = np.zeros_like(x)
        for j in range(x.shape[1]):
            out[:, j] = np.convolve(x[:, j], kern, mode='same')
        return out


def h5_deref(f, ref):
    """Dereference an HDF5 object reference."""
    return f[ref]


def h5_read_string(f, dataset):
    """Read a string from an HDF5 dataset (MATLAB char array stored as uint16)."""
    data = dataset[()]
    if data.dtype == np.uint16 or data.dtype == np.uint8:
        return ''.join(chr(c) for c in data.flatten())
    if isinstance(data, bytes):
        return data.decode('utf-8')
    return str(data)


def h5_read_scalar(dataset):
    """Read a scalar from an HDF5 dataset."""
    val = dataset[()]
    if hasattr(val, 'flatten'):
        return val.flatten()[0]
    return val


def load_session_data_v5(filepath):
    """Load session data from MATLAB v5/v7 format using scipy.io."""
    mat = scipy.io.loadmat(filepath, squeeze_me=False)
    obj = mat['obj']
    session = {}

    bp = obj['bp'][0, 0]
    Ntrials = int(bp['Ntrials'][0, 0].flatten()[0])
    session['Ntrials'] = Ntrials
    session['R'] = bp['R'][0, 0].flatten().astype(float)
    session['L'] = bp['L'][0, 0].flatten().astype(float)
    session['hit'] = bp['hit'][0, 0].flatten().astype(float)
    session['miss'] = bp['miss'][0, 0].flatten().astype(float)
    session['early'] = bp['early'][0, 0].flatten().astype(float)
    session['autowater'] = bp['autowater'][0, 0].flatten().astype(float)

    if 'stim' in bp.dtype.names:
        stim = bp['stim'][0, 0]
        session['stim_enable'] = stim['enable'][0, 0].flatten().astype(float)
    else:
        session['stim_enable'] = np.zeros(Ntrials)

    session['no'] = bp['no'][0, 0].flatten().astype(float) if 'no' in bp.dtype.names else np.zeros(Ntrials)

    ev = bp['ev'][0, 0]
    session['goCue'] = ev['goCue'][0, 0].flatten()
    session['sample'] = ev['sample'][0, 0].flatten() if 'sample' in ev.dtype.names else np.zeros(Ntrials)

    # Meta/animal info
    meta_key = 'meta' if 'meta' in obj.dtype.names else 'ex'
    if meta_key in obj.dtype.names:
        meta = obj[meta_key][0, 0]
        try:
            session['animal'] = str(meta['anm'][0, 0].flatten()[0]).strip()
        except:
            session['animal'] = 'unknown'
        try:
            session['date'] = str(meta['day'][0, 0].flatten()[0]).strip()
        except:
            session['date'] = 'unknown'
    else:
        session['animal'] = 'unknown'
        session['date'] = 'unknown'

    # Cluster data
    if 'clu' not in obj.dtype.names:
        raise ValueError("No clu field - behavior-only session")

    clu_data = obj['clu'][0, 0]
    if clu_data.size == 0:
        raise ValueError("Empty clu field")

    session['clusters'] = []
    session['n_probes'] = clu_data.shape[1] if clu_data.ndim >= 2 else 1

    for probe_idx in range(session['n_probes']):
        try:
            probe = clu_data[0, probe_idx]
            if probe.size == 0 or not hasattr(probe.dtype, 'names') or probe.dtype.names is None:
                session['clusters'].append([])
                continue

            quality_arr = probe['quality'].flatten()
            trialtm_arr = probe['trialtm'].flatten()
            trial_arr = probe['trial'].flatten()
            n_units = len(quality_arr)

            probe_clusters = []
            for unit_idx in range(n_units):
                unit = {}
                try:
                    q = quality_arr[unit_idx]
                    if hasattr(q, 'flatten'):
                        q = q.flatten()
                    unit['quality'] = str(q[0]).strip() if len(q) > 0 else ''
                except:
                    unit['quality'] = ''

                unit['trialtm'] = trialtm_arr[unit_idx].flatten()
                unit['trial'] = trial_arr[unit_idx].flatten().astype(int)
                probe_clusters.append(unit)

            session['clusters'].append(probe_clusters)
        except Exception as e:
            session['clusters'].append([])

    # Probe locations
    session['probe_locations'] = []
    if meta_key in obj.dtype.names:
        meta = obj[meta_key][0, 0]
        if 'probe' in meta.dtype.names:
            probe_meta = meta['probe'][0, 0]
            if 'loc' in probe_meta.dtype.names:
                locs = probe_meta['loc'][0, 0].flatten()
                for loc in locs:
                    try:
                        session['probe_locations'].append(str(loc.flatten()[0]).strip())
                    except:
                        session['probe_locations'].append('ALM')

    # Trajectory data
    session['traj'] = []
    if 'traj' in obj.dtype.names:
        traj_data = obj['traj'][0, 0]
        n_cams = traj_data.shape[1] if traj_data.ndim >= 2 else 0
        for cam_idx in range(n_cams):
            try:
                cam = traj_data[0, cam_idx]
                # cam is (1, n_trials) structured array
                n_traj_trials = cam.shape[1] if cam.ndim >= 2 else (cam.shape[0] if cam.ndim == 1 else 0)

                feat_names = []
                try:
                    # featNames is nested: access first trial
                    t0 = cam.flatten()[0]
                    fnames = t0['featNames'].flatten()
                    for fn in fnames:
                        if hasattr(fn, 'flatten'):
                            fn_arr = fn.flatten()
                            if len(fn_arr) > 0:
                                feat_names.append(str(fn_arr[0]).strip())
                except:
                    pass

                cam_trials = []
                for trial_idx in range(n_traj_trials):
                    trial_data = {}
                    try:
                        t = cam.flatten()[trial_idx]

                        # ts shape in v5: (n_frames, 3, n_features)
                        ts_raw = t['ts'].flatten()
                        if len(ts_raw) > 0:
                            ts = ts_raw[0]
                            if ts.ndim == 3 and ts.shape[1] == 3:
                                # (n_frames, 3, n_features) -> (n_features, 3, n_frames)
                                trial_data['ts'] = np.transpose(ts, (2, 1, 0))
                            elif ts.ndim == 3:
                                trial_data['ts'] = ts
                            else:
                                trial_data['ts'] = None
                        else:
                            trial_data['ts'] = None
                    except:
                        trial_data['ts'] = None

                    try:
                        ft = t['frameTimes'].flatten()
                        if len(ft) > 0:
                            ft_val = ft[0] if isinstance(ft[0], np.ndarray) else ft
                            trial_data['frameTimes'] = ft_val.flatten()
                        else:
                            trial_data['frameTimes'] = None
                    except:
                        trial_data['frameTimes'] = None

                    try:
                        ndf = t['NdroppedFrames'].flatten()
                        trial_data['NdroppedFrames'] = float(ndf[0]) if len(ndf) > 0 else 0
                    except:
                        trial_data['NdroppedFrames'] = 0

                    cam_trials.append(trial_data)

                session['traj'].append({
                    'feat_names': feat_names,
                    'trials': cam_trials
                })
            except Exception as e:
                pass

    # Video offset
    try:
        sglx = obj['sglx'][0, 0]
        fs = float(sglx['fs'][0, 0].flatten()[0])
        bitstart_sglx = float(np.nanmedian(sglx['bitcode'][0, 0]['bitstart'][0, 0].flatten()))
        bitStart_bp = float(np.nanmedian(ev['bitStart'][0, 0].flatten()))
        session['vidshift'] = bitstart_sglx / fs - bitStart_bp
    except:
        session['vidshift'] = 0.0

    return session


def load_session_data(filepath):
    """Load a single session's data_structure .mat file.
    Tries HDF5 (v7.3) first, falls back to scipy.io (v5/v7).
    """
    # Try HDF5 first
    try:
        with h5py.File(filepath, 'r') as f:
            # Check if it has obj/clu
            if 'obj' not in f:
                raise ValueError("No obj field")
            obj = f['obj']
            if 'clu' not in obj:
                raise ValueError("No clu field - behavior-only session")
        return _load_session_data_h5(filepath)
    except (OSError, ValueError) as e:
        pass

    # Fall back to scipy.io
    return load_session_data_v5(filepath)


def _load_session_data_h5(filepath):
    """Load a single session's data_structure .mat file (HDF5 format).

    Returns a dict with the relevant fields extracted.
    """
    session = {}

    with h5py.File(filepath, 'r') as f:
        obj = f['obj']
        bp = obj['bp']

        # --- Basic behavioral variables ---
        Ntrials = int(h5_read_scalar(bp['Ntrials']))
        session['Ntrials'] = Ntrials
        session['R'] = bp['R'][()].flatten().astype(float)  # (Ntrials,)
        session['L'] = bp['L'][()].flatten().astype(float)
        session['hit'] = bp['hit'][()].flatten().astype(float)
        session['miss'] = bp['miss'][()].flatten().astype(float)
        session['early'] = bp['early'][()].flatten().astype(float)
        session['autowater'] = bp['autowater'][()].flatten().astype(float)

        # Stim enable
        if 'stim' in bp:
            session['stim_enable'] = bp['stim']['enable'][()].flatten().astype(float)
        else:
            session['stim_enable'] = np.zeros(Ntrials)

        # No-response indicator
        if 'no' in bp:
            session['no'] = bp['no'][()].flatten().astype(float)
        else:
            session['no'] = np.zeros(Ntrials)

        # --- Event times ---
        ev = bp['ev']
        session['goCue'] = ev['goCue'][()].flatten()
        session['sample'] = ev['sample'][()].flatten() if 'sample' in ev else np.zeros(Ntrials)
        session['delay'] = ev['delay'][()].flatten() if 'delay' in ev else np.zeros(Ntrials)

        # --- Metadata ---
        if 'meta' in obj and isinstance(obj['meta'], h5py.Group):
            try:
                session['animal'] = h5_read_string(f, obj['meta']['anm'])
            except:
                session['animal'] = 'unknown'
            try:
                session['date'] = h5_read_string(f, obj['meta']['day'])
            except:
                session['date'] = 'unknown'
        else:
            session['animal'] = 'unknown'
            session['date'] = 'unknown'

        # --- Probe/cluster info ---
        # obj.clu is (nprobes, 1) array of references
        clu_refs = obj['clu'][()].flatten()
        session['n_probes'] = len(clu_refs)
        session['clusters'] = []

        for probe_idx in range(len(clu_refs)):
            try:
                probe_group = f[clu_refs[probe_idx]]
                # Must be a Group with quality/trialtm/trial fields
                if not isinstance(probe_group, h5py.Group):
                    session['clusters'].append([])
                    continue
                if 'quality' not in probe_group or 'trialtm' not in probe_group:
                    session['clusters'].append([])
                    continue

                n_units = probe_group['quality'].shape[0]

                probe_clusters = []
                for unit_idx in range(n_units):
                    unit = {}

                    # Quality
                    q_ref = probe_group['quality'][unit_idx, 0]
                    try:
                        unit['quality'] = h5_read_string(f, f[q_ref])
                    except:
                        unit['quality'] = ''

                    # Spike times (within trial, relative to trial start)
                    tm_ref = probe_group['trialtm'][unit_idx, 0]
                    unit['trialtm'] = f[tm_ref][()].flatten()

                    # Trial assignment for each spike
                    trial_ref = probe_group['trial'][unit_idx, 0]
                    unit['trial'] = f[trial_ref][()].flatten().astype(int)

                    probe_clusters.append(unit)

                session['clusters'].append(probe_clusters)
            except Exception as e:
                print(f"  Warning: Could not load probe {probe_idx}: {e}")
                session['clusters'].append([])

        # --- Probe location info ---
        session['probe_locations'] = []
        if 'meta' in obj and isinstance(obj['meta'], h5py.Group) and 'probe' in obj['meta']:
            probe_meta = obj['meta']['probe']
            if 'loc' in probe_meta:
                loc_data = probe_meta['loc'][()].flatten()
                for loc_ref in loc_data:
                    try:
                        loc_str = h5_read_string(f, f[loc_ref])
                        session['probe_locations'].append(loc_str)
                    except:
                        session['probe_locations'].append('ALM')

        # --- Trajectory data for kinematics ---
        session['traj'] = []
        if 'traj' in obj:
            traj_refs = obj['traj'][()].flatten()
            for cam_idx in range(len(traj_refs)):
                cam_group = f[traj_refs[cam_idx]]
                cam_data = []

                # Each camera has per-trial data
                n_traj_trials = cam_group['ts'].shape[0]

                # Get feature names from first trial
                # featNames is (n_trials, 1) refs, each ref -> dataset of (1, n_feats) refs
                feat_names = []
                try:
                    fn_ref = cam_group['featNames'][0, 0]
                    fn_dataset = f[fn_ref][()]  # (1, n_feats) array of refs
                    for ref in fn_dataset.flatten():
                        feat_names.append(h5_read_string(f, f[ref]))
                except:
                    feat_names = []

                for trial_idx in range(n_traj_trials):
                    trial_data = {}

                    # ts: (n_features, 3, n_frames) - MATLAB column-major stored in HDF5
                    try:
                        ts_ref = cam_group['ts'][trial_idx, 0]
                        ts = f[ts_ref][()]  # (n_features, 3, n_frames)
                        trial_data['ts'] = ts
                    except:
                        trial_data['ts'] = None

                    # frameTimes
                    try:
                        ft_ref = cam_group['frameTimes'][trial_idx, 0]
                        ft = f[ft_ref][()].flatten()
                        trial_data['frameTimes'] = ft
                    except:
                        trial_data['frameTimes'] = None

                    # NdroppedFrames
                    try:
                        ndf_ref = cam_group['NdroppedFrames'][trial_idx, 0]
                        ndf_val = f[ndf_ref][()].flatten()[0]
                        trial_data['NdroppedFrames'] = ndf_val
                    except:
                        trial_data['NdroppedFrames'] = 0

                    cam_data.append(trial_data)

                session['traj'].append({
                    'feat_names': feat_names,
                    'trials': cam_data
                })

        # --- Video offset computation ---
        # vidshift = mode(sglx.bitcode.bitstart) / sglx.fs - mode(bp.ev.bitStart)
        try:
            bitStart_bp = np.nanmedian(ev['bitStart'][()].flatten())
            sglx = obj['sglx']
            fs = h5_read_scalar(sglx['fs'])
            bitstart_sglx = np.nanmedian(sglx['bitcode']['bitstart'][()].flatten())
            session['vidshift'] = bitstart_sglx / fs - bitStart_bp
        except:
            session['vidshift'] = 0.0

    return session


def load_motion_energy(filepath):
    """Load motion energy from .mat file (v5/v7 or v7.3/HDF5 format)."""
    try:
        # Try scipy first (v5/v7)
        mat = scipy.io.loadmat(filepath, squeeze_me=False)
        me = mat['me']

        # Extract data and threshold
        data_field = me['data'][0, 0]
        thresh_field = me['moveThresh'][0, 0]

        # data is an object array with per-trial arrays
        me_data = []
        for i in range(data_field.shape[0]):
            for j in range(data_field.shape[1]):
                trial_me = data_field[i, j].flatten().astype(float)
                me_data.append(trial_me)

        threshold = float(thresh_field.flatten()[0])
        return {'data': me_data, 'moveThresh': threshold}
    except NotImplementedError:
        # v7.3 format - use h5py
        pass
    except Exception as e:
        # Try h5py as fallback
        pass

    # Try HDF5 format
    try:
        with h5py.File(filepath, 'r') as f:
            me_group = f['me']
            # data is (n_trials, 1) array of refs
            data_refs = me_group['data'][()].flatten()
            me_data = []
            for ref in data_refs:
                trial_data = f[ref][()].flatten().astype(float)
                me_data.append(trial_data)

            threshold = float(me_group['moveThresh'][()].flatten()[0])
            return {'data': me_data, 'moveThresh': threshold}
    except Exception as e:
        print(f"  Warning: Could not load motion energy from {filepath}: {e}")
        return None


def filter_clusters(clusters, quality_filter='all'):
    """Filter clusters by quality, matching findClusters.m."""
    excluded = {'garbage', 'gabrga', 'noisy', 'real?'}
    indices = []
    for i, unit in enumerate(clusters):
        q = unit.get('quality', '').strip().lower()
        if q not in excluded and q != '':
            indices.append(i)
        elif q == '':
            # Include unlabeled units (matching MATLAB 'all' behavior which includes empty strings)
            indices.append(i)
    return indices


def align_and_bin_spikes(clusters, cluster_indices, go_cue_times, trial_mask,
                         edges, dt, smooth_samples):
    """Align spikes to goCue, bin, smooth, and compute firing rates.

    Returns:
        trialdat: (n_timebins, n_units, n_trials) firing rate array
        mean_frs: (n_units,) mean firing rate per unit
    """
    valid_trials = np.where(trial_mask)[0]
    n_trials = len(valid_trials)
    n_units = len(cluster_indices)
    n_timebins = len(edges) - 1

    trialdat = np.zeros((n_timebins, n_units, n_trials), dtype=np.float32)

    for ui, ci in enumerate(cluster_indices):
        unit = clusters[ci]
        spike_times = unit['trialtm']
        spike_trials = unit['trial']

        for ti, trial_idx in enumerate(valid_trials):
            trial_num = trial_idx + 1  # MATLAB 1-indexed
            go_time = go_cue_times[trial_idx]

            if np.isnan(go_time) or go_time == 0:
                continue

            # Find spikes for this trial
            spike_mask = spike_trials == trial_num
            if not np.any(spike_mask):
                continue

            # Align to go cue
            aligned_times = spike_times[spike_mask] - go_time

            # Bin spikes
            counts, _ = np.histogram(aligned_times, bins=edges)
            counts = counts[:n_timebins].astype(float)

            # Convert to firing rate and smooth
            fr = counts / dt
            fr_smooth = causal_gaussian_smooth(fr, smooth_samples)
            trialdat[:, ui, ti] = fr_smooth

    # Compute mean firing rate per unit (across all trials and time)
    # Matching removeLowFRClusters.m: mean of mean across conditions
    # For simplicity, use mean across all time and trials
    mean_frs = np.mean(np.mean(trialdat, axis=2), axis=0)

    return trialdat, mean_frs


def compute_velocity(positions, dt_video=1/400):
    """Compute velocity magnitude from x,y positions.

    positions: (n_frames, 2) array of [x, y]
    Returns: (n_frames,) velocity magnitude
    """
    if positions is None or len(positions) < 2:
        return None

    # Compute velocity as first derivative
    dx = np.gradient(positions[:, 0], dt_video)
    dy = np.gradient(positions[:, 1], dt_video)
    velocity = np.sqrt(dx**2 + dy**2)
    return velocity


def extract_kinematic_feature(session, cam_idx, feat_name, valid_trials, go_cue_times):
    """Extract and interpolate a kinematic feature velocity to neural time axis.

    ts from HDF5 has shape (n_features, 3, n_frames) where 3=[x, y, likelihood].
    For tongue features, positions are NaN when tongue is not visible.

    Returns:
        velocities: (n_timebins, n_trials) array or None
    """
    if cam_idx >= len(session['traj']):
        return None

    cam = session['traj'][cam_idx]
    feat_names = cam['feat_names']

    # Find exact feature index
    feat_idx = None
    for i, fn in enumerate(feat_names):
        if fn.lower() == feat_name.lower():
            feat_idx = i
            break

    if feat_idx is None:
        return None

    is_tongue = 'tongue' in feat_name.lower()
    vidshift = session.get('vidshift', 0.0)
    n_trials = len(valid_trials)
    velocities = np.full((N_TIMEBINS, n_trials), np.nan, dtype=np.float32)

    for ti, trial_idx in enumerate(valid_trials):
        if trial_idx >= len(cam['trials']):
            continue

        trial_data = cam['trials'][trial_idx]
        ts = trial_data.get('ts')
        frame_times = trial_data.get('frameTimes')

        if ts is None or ts.ndim != 3:
            continue

        # Handle NdroppedFrames
        ndf = trial_data.get('NdroppedFrames', 0)
        if isinstance(ndf, float) and np.isnan(ndf):
            continue

        go_time = go_cue_times[trial_idx]
        if np.isnan(go_time) or go_time == 0:
            continue

        # ts shape: (n_features, 3, n_frames) from HDF5
        n_features, three, n_frames = ts.shape
        if three != 3 or feat_idx >= n_features:
            continue

        # Extract x, y for the feature: (n_frames,) each
        x_pos = ts[feat_idx, 0, :].copy()
        y_pos = ts[feat_idx, 1, :].copy()

        # Fill missing values for non-tongue features (matching MATLAB fillmissing nearest)
        if not is_tongue:
            for pos in [x_pos, y_pos]:
                mask = np.isnan(pos)
                if mask.any() and not mask.all():
                    valid_idx = np.where(~mask)[0]
                    pos[mask] = np.interp(np.where(mask)[0], valid_idx, pos[valid_idx])

        # Compute velocity magnitude
        dt_vid = 1 / 400
        dx = np.gradient(x_pos, dt_vid)
        dy = np.gradient(y_pos, dt_vid)
        vel = np.sqrt(dx**2 + dy**2)

        # For tongue, set velocity to 0 where tongue not visible
        if is_tongue:
            nan_mask = np.isnan(x_pos) | np.isnan(y_pos)
            vel[nan_mask] = 0.0
            vel[np.isnan(vel)] = 0.0

        # Create frame times if not available
        if frame_times is None or np.all(np.isnan(frame_times)):
            frame_times = np.arange(1, n_frames + 1) / 400.0

        # Align to go cue and interpolate to neural time axis
        aligned_frame_times = frame_times[:n_frames] - vidshift - go_time

        try:
            interp_fn = interp1d(aligned_frame_times, vel,
                                kind='linear', bounds_error=False, fill_value=np.nan)
            velocities[:, ti] = interp_fn(TIME_AXIS)
        except:
            continue

    # Fill NaN with nearest valid values (or 0 for tongue)
    for ti in range(n_trials):
        col = velocities[:, ti]
        mask = np.isnan(col)
        if mask.any() and not mask.all():
            valid_idx = np.where(~mask)[0]
            col[mask] = np.interp(np.where(mask)[0], valid_idx, col[valid_idx])
            velocities[:, ti] = col
        elif mask.all():
            velocities[:, ti] = 0.0

    return velocities


def interpolate_motion_energy(me_data, session, valid_trials, go_cue_times):
    """Interpolate motion energy to neural time axis, aligned to go cue.

    Matching loadMotionEnergy.m logic.
    """
    vidshift = session.get('vidshift', 0.0)
    n_trials = len(valid_trials)
    me_interp = np.full((N_TIMEBINS, n_trials), np.nan, dtype=np.float32)

    for ti, trial_idx in enumerate(valid_trials):
        if trial_idx >= len(me_data):
            continue

        me_trial = me_data[trial_idx]
        go_time = go_cue_times[trial_idx]

        if np.isnan(go_time) or go_time == 0:
            continue

        if me_trial is None or len(me_trial) == 0:
            continue

        n_frames = len(me_trial)

        # Get frame times
        # Try to use traj frameTimes, otherwise fall back
        frame_times = None
        if len(session['traj']) > 0 and trial_idx < len(session['traj'][0]['trials']):
            ft = session['traj'][0]['trials'][trial_idx].get('frameTimes')
            if ft is not None and not np.all(np.isnan(ft)):
                frame_times = ft

        if frame_times is None:
            frame_times = np.arange(1, n_frames + 1) / 400.0
            aligned_frame_times = frame_times - 0.5 - go_time  # fallback offset
        else:
            # Trim/pad to match me_trial length
            if len(frame_times) > n_frames:
                frame_times = frame_times[:n_frames]
            elif len(frame_times) < n_frames:
                # Extend frame times
                dt_vid = 1/400
                extra = np.arange(len(frame_times), n_frames) * dt_vid + frame_times[-1] + dt_vid
                frame_times = np.concatenate([frame_times, extra])
            aligned_frame_times = frame_times - vidshift - go_time

        try:
            interp_fn = interp1d(aligned_frame_times, me_trial,
                                kind='linear', bounds_error=False, fill_value=np.nan)
            me_interp[:, ti] = interp_fn(TIME_AXIS)
        except:
            continue

    # Fill NaN with nearest
    for ti in range(n_trials):
        col = me_interp[:, ti]
        mask = np.isnan(col)
        if mask.any() and not mask.all():
            valid_idx = np.where(~mask)[0]
            col[mask] = np.interp(np.where(mask)[0], valid_idx, col[valid_idx])
            me_interp[:, ti] = col
        elif mask.all():
            me_interp[:, ti] = 0.0

    return me_interp


def discretize_per_session(values, percentile=50):
    """Discretize values into 2 bins using per-session threshold.

    values: (n_timebins, n_trials)
    Returns: (n_timebins, n_trials) with 0 (< threshold) and 1 (>= threshold)

    The threshold is the 50th percentile (median). Per the decoder task spec:
    0: < 50th percentile, 1: >= 50th percentile.
    """
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.zeros_like(values, dtype=np.int32)

    threshold = np.percentile(valid, percentile)
    discretized = (values >= threshold).astype(np.int32)
    return discretized


def process_session(session_file, me_file, session_idx_global):
    """Process a single session and return formatted data."""

    basename = os.path.basename(session_file)
    parts = basename.replace('data_structure_', '').replace('.mat', '').split('_')
    animal = parts[0]
    date = '_'.join(parts[1:])

    print(f"\n--- Processing session {session_idx_global}: {animal}_{date} ---")

    # Load session data
    session = load_session_data(session_file)
    Ntrials = session['Ntrials']
    print(f"  Total trials: {Ntrials}")

    # Load motion energy
    me = load_motion_energy(me_file) if me_file else None

    # --- Trial filtering ---
    # Include: non-stim, non-early, responding (hit or miss) trials
    # Matching: ~stim.enable & ~early & (hit | miss)
    trial_mask = (
        (session['stim_enable'] == 0) &
        (session['early'] == 0) &
        ((session['hit'] == 1) | (session['miss'] == 1))
    )

    valid_trials = np.where(trial_mask)[0]
    n_valid = len(valid_trials)
    print(f"  Valid trials (no stim, no early, hit|miss): {n_valid}")

    if n_valid < 2:
        print(f"  SKIPPING: fewer than 2 valid trials")
        return None

    # --- Cluster filtering and spike processing ---
    # Process all probes
    all_cluster_indices = []
    all_clusters = []
    probe_region_labels = []

    for probe_idx in range(session['n_probes']):
        clusters = session['clusters'][probe_idx]

        # Get probe location
        if probe_idx < len(session['probe_locations']):
            loc = session['probe_locations'][probe_idx]
        else:
            loc = 'ALM'

        # Only include ALM probes (paper: "We recorded activity extracellularly in the ALM")
        loc_upper = loc.strip().upper()
        if 'ALM' not in loc_upper:
            print(f"  Probe {probe_idx} ({loc}): skipping non-ALM probe")
            continue

        # Filter by quality (matching findClusters.m with quality='all')
        valid_units = filter_clusters(clusters)

        if len(valid_units) == 0:
            continue

        # Compute firing rates for this probe
        trialdat_probe, mean_frs = align_and_bin_spikes(
            clusters, valid_units, session['goCue'], trial_mask,
            EDGES, DT, SMOOTH_MS
        )

        # Filter by firing rate (matching removeLowFRClusters.m)
        fr_mask = mean_frs > LOW_FR
        kept_units = [valid_units[i] for i in range(len(valid_units)) if fr_mask[i]]
        trialdat_probe = trialdat_probe[:, fr_mask, :]

        print(f"  Probe {probe_idx} ({loc}): {len(clusters)} total -> {len(valid_units)} quality-filtered -> {len(kept_units)} FR-filtered units")

        all_clusters.append(trialdat_probe)
        for _ in kept_units:
            probe_region_labels.append(loc)

    if len(all_clusters) == 0:
        print(f"  SKIPPING: no valid clusters")
        return None

    # Concatenate across probes: (n_timebins, n_total_units, n_trials)
    neural_data = np.concatenate(all_clusters, axis=1)
    n_units = neural_data.shape[1]
    n_trials = neural_data.shape[2]
    print(f"  Neural data shape: {neural_data.shape} (time, units, trials)")

    if n_units == 0:
        print(f"  SKIPPING: 0 valid units after filtering")
        return None

    # --- Extract behavioral variables ---
    lick_direction = session['R'][valid_trials].astype(np.int32)  # R=1, L=0
    context = (1 - session['autowater'][valid_trials]).astype(np.int32)  # DR=1, WC=0
    outcome = session['hit'][valid_trials].astype(np.int32)  # correct=1, incorrect=0

    print(f"  Lick direction: R={np.sum(lick_direction==1)}, L={np.sum(lick_direction==0)}")
    print(f"  Context: DR={np.sum(context==1)}, WC={np.sum(context==0)}")
    print(f"  Outcome: correct={np.sum(outcome==1)}, incorrect={np.sum(outcome==0)}")

    # --- Extract kinematic features ---
    go_cue_times = session['goCue']

    # Tongue velocity from side camera (cam 0), feature 'tongue'
    tongue_vel = extract_kinematic_feature(session, 0, 'tongue', valid_trials, go_cue_times)

    # Paw velocity from bottom camera (cam 1)
    # Average top_paw and bottom_paw velocities when both available
    paw_vel = None
    if len(session['traj']) > 1:
        paw_top = extract_kinematic_feature(session, 1, 'top_paw', valid_trials, go_cue_times)
        paw_bot = extract_kinematic_feature(session, 1, 'bottom_paw', valid_trials, go_cue_times)

        if paw_top is not None and paw_bot is not None:
            paw_vel = (paw_top + paw_bot) / 2.0
        elif paw_top is not None:
            paw_vel = paw_top
        elif paw_bot is not None:
            paw_vel = paw_bot

    # Motion energy
    me_interp = None
    if me is not None:
        me_interp = interpolate_motion_energy(me['data'], session, valid_trials, go_cue_times)

    # --- Discretize continuous outputs ---
    if tongue_vel is not None:
        tongue_vel_disc = discretize_per_session(tongue_vel)
    else:
        print(f"  Warning: No tongue velocity data, using zeros")
        tongue_vel_disc = np.zeros((N_TIMEBINS, n_trials), dtype=np.int32)

    if paw_vel is not None:
        paw_vel_disc = discretize_per_session(paw_vel)
    else:
        print(f"  Warning: No paw velocity data, using zeros")
        paw_vel_disc = np.zeros((N_TIMEBINS, n_trials), dtype=np.int32)

    if me_interp is not None:
        me_disc = discretize_per_session(me_interp)
    else:
        print(f"  Warning: No motion energy data, using zeros")
        me_disc = np.zeros((N_TIMEBINS, n_trials), dtype=np.int32)

    # --- Format per-trial data ---
    neural_trials = []
    input_trials = []
    output_trials = []

    for ti in range(n_trials):
        # Neural: (n_units, n_timebins)
        neural_trial = neural_data[:, :, ti].T.astype(np.float32)
        neural_trials.append(neural_trial)

        # Input: time from go cue (1, n_timebins)
        input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
        input_trials.append(input_trial)

        # Output: (6, n_timebins) - per-trial values tiled, time-varying ones as-is
        output_trial = np.zeros((6, N_TIMEBINS), dtype=np.int32)
        output_trial[0, :] = lick_direction[ti]  # per-trial
        output_trial[1, :] = context[ti]  # per-trial
        output_trial[2, :] = outcome[ti]  # per-trial
        output_trial[3, :] = tongue_vel_disc[:, ti]  # time-varying
        output_trial[4, :] = paw_vel_disc[:, ti]  # time-varying
        output_trial[5, :] = me_disc[:, ti]  # time-varying
        output_trials.append(output_trial)

    # --- Brain region info ---
    # Clean up region labels
    region_labels_clean = []
    for label in probe_region_labels:
        label_clean = label.strip()
        # Normalize common variations
        if 'ALM' in label_clean.upper():
            label_clean = 'ALM'
        elif 'TJM1' in label_clean.upper() or 'M1' in label_clean.upper():
            label_clean = 'tjM1'
        region_labels_clean.append(label_clean)

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'animal': animal,
        'date': date,
        'region_labels': region_labels_clean,
        'n_units': n_units,
        'n_trials': n_trials,
    }


def find_session_files(data_dirs):
    """Find all data_structure and motionEnergy file pairs."""
    sessions = []

    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            continue

        data_files = sorted(glob.glob(os.path.join(data_dir, 'data_structure_*.mat')))

        for df in data_files:
            basename = os.path.basename(df)
            # Extract animal_date from data_structure_ANIMAL_DATE.mat
            parts = basename.replace('data_structure_', '').replace('.mat', '')
            me_file = os.path.join(data_dir, f'motionEnergy_{parts}.mat')

            if not os.path.exists(me_file):
                me_file = None

            sessions.append({
                'data_file': df,
                'me_file': me_file,
                'animal_date': parts,
            })

    return sessions


def convert_data(data_dirs=DATA_DIRS, max_sessions=None, output_file='converted_data.pkl'):
    """Main conversion function."""

    # Find all session files
    session_files = find_session_files(data_dirs)
    print(f"Found {len(session_files)} sessions")

    if max_sessions is not None:
        session_files = session_files[:max_sessions]
        print(f"Processing first {max_sessions} sessions")

    # Process each session
    all_neural = []
    all_input = []
    all_output = []
    all_animals = []
    all_dates = []
    all_region_labels = []

    for idx, sf in enumerate(session_files):
        try:
            result = process_session(sf['data_file'], sf['me_file'], idx)
        except Exception as e:
            print(f"\n  ERROR processing session {idx} ({sf['animal_date']}): {e}")
            import traceback
            traceback.print_exc()
            continue

        if result is None:
            continue

        all_neural.append(result['neural'])
        all_input.append(result['input'])
        all_output.append(result['output'])
        all_animals.append(result['animal'])
        all_dates.append(result['date'])
        all_region_labels.append(result['region_labels'])

    n_sessions = len(all_neural)
    print(f"\n=== Successfully processed {n_sessions} sessions ===")

    if n_sessions == 0:
        print("ERROR: No sessions processed successfully")
        return None

    # --- Build subject index ---
    unique_subjects = sorted(set(all_animals))
    subject_idx = np.array([unique_subjects.index(a) for a in all_animals], dtype=np.int64)

    # --- Build brain region index ---
    all_regions_flat = set()
    for labels in all_region_labels:
        all_regions_flat.update(labels)
    brain_regions = sorted(all_regions_flat)

    brain_region_idx = []
    for labels in all_region_labels:
        idx_arr = np.array([brain_regions.index(l) for l in labels], dtype=np.int64)
        brain_region_idx.append(idx_arr)

    # --- Build output ---
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': unique_subjects,
        'subject_idx': subject_idx,
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_go_cue'],
        'output_names': [
            'lick_direction',
            'behavioral_context',
            'outcome',
            'tongue_velocity',
            'paw_velocity',
            'motion_energy',
        ],
        'output_values': [
            ['left', 'right'],
            ['WC', 'DR'],
            ['incorrect', 'correct'],
            ['low', 'high'],
            ['low', 'high'],
            ['low', 'high'],
        ],
        'metadata': {
            'task_description': 'Mice perform delayed-response (DR) and water-cued (WC) directional licking tasks with ALM recordings',
            'time_bin_size': DT * 1000,  # in ms
            'temporal_alignment_event': 'Go cue onset',
            'off_start': TMIN,  # -2.5 s before go cue
            'off_end': TMAX,  # +2.5 s after go cue
            'paper': 'Hasnain, Birnbaum et al., Nature Neuroscience 2024',
            'recording_region': 'ALM (anterior lateral motor cortex)',
            'n_sessions': n_sessions,
            'sessions': [f"{a}_{d}" for a, d in zip(all_animals, all_dates)],
            'smooth_ms': SMOOTH_MS,
            'low_fr_threshold': LOW_FR,
        },
    }

    # Save
    with open(output_file, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    print(f"\nSaved converted data to {output_file}")
    print(f"  Sessions: {n_sessions}")
    print(f"  Subjects: {unique_subjects}")
    print(f"  Brain regions: {brain_regions}")
    total_trials = sum(len(s) for s in all_neural)
    total_neurons = sum(s[0].shape[0] for s in all_neural if len(s) > 0)
    print(f"  Total trials: {total_trials}")
    print(f"  Total neurons: {total_neurons}")

    return data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert neural data to decoder format.')
    parser.add_argument('--max-sessions', type=int, default=None,
                        help='Max number of sessions to process (for testing)')
    parser.add_argument('--output', type=str, default='converted_data.pkl',
                        help='Output pickle file path')
    parser.add_argument('--data-dirs', type=str, nargs='+', default=DATA_DIRS,
                        help='Data directories to search')

    args = parser.parse_args()

    convert_data(
        data_dirs=args.data_dirs,
        max_sessions=args.max_sessions,
        output_file=args.output,
    )
