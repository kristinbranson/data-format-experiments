"""
Convert the Majnik et al. 2025 (Track2p) longitudinal 2p calcium imaging dataset into
the standard decoding dictionary format.

Decoding task: predict the animal's motion energy (discretized into 5 equal-percentile
bins per session) from the activity of the tracked barrel-cortex neurons, with time
elapsed since the beginning of the session as an additional decoder input.

Processing decisions (following the paper and the track2p repository):
  * Neural data: raw Suite2p traces (F.npy) of the neurons tracked across all days of a
    mouse (these are the only ROIs saved in the released data, all with iscell==1).
    dF/F is computed exactly as in track2p's GUI helper `F_processing`
    (track2p/gui/data_management.py), which reproduces Suite2p's default baseline
    correction: neuropil coefficient 0.0, 'maximin' baseline with sig_baseline=10
    frames and win_baseline=60 s, i.e. F - maximin_baseline(F).
  * Denoising: as described in the paper's decoding methods, dF/F and behaviour traces
    are averaged in bins of 10 consecutive frames (30 Hz -> 3 Hz, 333.33 ms bins).
  * Each neuron's binned trace is z-scored within the session (as done in track2p's
    raster preprocessing) so that neurons/sessions are on a comparable scale for the
    decoder (fluorescence units are arbitrary and differ across days).
  * Neuron curation: ROIs whose trace is identically zero (empty ROI on that day) are
    dropped. Because neurons are matched across days within a mouse (row i is the same
    cell on every day), a neuron is dropped from *all* sessions of that mouse if it is
    all-zero on any day, exactly as track2p's GUI removes such 'zero rows' across days.
  * Behaviour: motion_energy_glob.npy is sampled at the imaging frame rate (camera was
    triggered by the microscope). When camera frames were dropped (length < n imaging
    frames), the dropped frames are localised with interframe_int.npy (intervals ~2x the
    median) and filled by linear interpolation, as suggested in the dataset README.
  * Trials: each session is cut into consecutive non-overlapping 60 s blocks
    (180 bins of 333.33 ms). Sessions are 20 min (20 trials) or 30 min (30 trials).
  * Output: binned motion energy discretized into 5 equal-percentile bins (quintiles)
    computed per session.
  * All 6 mice and all 41 sessions are kept (all have >= 6 consecutive imaging days,
    the inclusion criterion of the paper).
"""

import os
import glob
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter1d, maximum_filter1d

DATA_DIR = '/app/data'
OUT_FILE = '/app/converted_data.pkl'

BIN_FRAMES = 10          # frames averaged together (paper: bins of 10 timestamps)
TRIAL_SEC = 60.0         # trial duration in seconds
N_QUANTILES = 5          # number of equal-percentile motion-energy bins


def f_processing(F, Fneu, fs, neucoeff=0.0, baseline='maximin',
                 sig_baseline=10.0, win_baseline=60.0):
    """Baseline-corrected fluorescence (dF/F), copied from track2p gui/data_management.py."""
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    if baseline == 'maximin':
        Flow = gaussian_filter(Fc, [0., sig_baseline])
        Flow = minimum_filter1d(Flow, win)
        Flow = maximum_filter1d(Flow, win)
    else:
        Flow = 0.
    return Fc - Flow


def align_motion_energy(me, ifi, nframes):
    """Put motion energy on the imaging frame grid, interpolating dropped camera frames.

    The camera was triggered by the microscope, so camera frame i corresponds to imaging
    frame i unless frames were dropped. Dropped frames show up both as a shorter
    motion-energy trace and as interframe intervals that are multiples of the median
    interval (see the dataset README). We insert exactly as many missing samples as the
    length mismatch, at the positions with the longest interframe intervals, and fill
    them by linear interpolation.
    """
    me = me.astype(np.float64)
    gap = int(nframes - len(me))
    if gap <= 0:
        return me[:nframes]
    med = np.median(ifi)
    nmiss = np.maximum(np.round(ifi / med).astype(int) - 1, 0)
    if nmiss.sum() > gap:
        # keep only the largest gaps, up to the number of actually missing frames
        order = np.argsort(-(ifi / med))
        keepmask = np.zeros(len(ifi), dtype=int)
        remaining = gap
        for i in order:
            if remaining <= 0:
                break
            take = min(nmiss[i], remaining)
            keepmask[i] = take
            remaining -= take
        nmiss = keepmask
    out = [me[0]]
    for i in range(len(ifi)):
        for _ in range(int(nmiss[i])):
            out.append(np.nan)
        out.append(me[i + 1])
    arr = np.array(out, dtype=np.float64)
    nanmask = np.isnan(arr)
    if nanmask.any():
        idx = np.arange(len(arr))
        arr[nanmask] = np.interp(idx[nanmask], idx[~nanmask], arr[~nanmask])
    if len(arr) > nframes:
        arr = arr[:nframes]
    elif len(arr) < nframes:
        arr = np.concatenate([arr, np.full(nframes - len(arr), arr[-1])])
    return arr


def bin_time(x, bin_size):
    """Average consecutive time bins. x: (..., T) -> (..., T//bin_size)."""
    T = x.shape[-1]
    nb = T // bin_size
    x = x[..., :nb * bin_size]
    return x.reshape(*x.shape[:-1], nb, bin_size).mean(axis=-1)


def main():
    subjects = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, 'jm*')))

    neural_all, input_all, output_all = [], [], []
    subject_idx, brain_region_idx = [], []
    session_info = []

    for si, sub in enumerate(subjects):
        sub_dir = os.path.join(DATA_DIR, sub)
        sess_dirs = sorted([f.path for f in os.scandir(sub_dir) if f.is_dir()])

        # --- first pass: find ROIs that are all-zero on any day of this mouse ---
        bad = None
        for sd in sess_dirs:
            F = np.load(os.path.join(sd, 'suite2p', 'plane0', 'F.npy'))
            z = np.all(F == 0, axis=1) | (F.std(axis=1) == 0)
            bad = z if bad is None else (bad | z)
        keep = ~bad
        print(f'{sub}: {keep.sum()}/{len(keep)} neurons kept ({bad.sum()} empty ROIs dropped)')

        for sd in sess_dirs:
            s2p = os.path.join(sd, 'suite2p', 'plane0')
            F = np.load(os.path.join(s2p, 'F.npy'))[keep]
            Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))[keep]
            ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
            fs = float(ops['fs'])
            nframes = int(ops['nframes'])

            # dF/F (track2p / Suite2p default baseline correction)
            dff = f_processing(F.astype(np.float64), Fneu.astype(np.float64), fs)

            # behaviour
            mv = os.path.join(sd, 'move_deve')
            me = np.load(os.path.join(mv, 'motion_energy_glob.npy'))
            ifi = np.load(os.path.join(mv, 'interframe_int.npy'))
            me = align_motion_energy(me, ifi, nframes)

            # denoise by averaging bins of 10 frames
            dff_b = bin_time(dff, BIN_FRAMES)
            me_b = bin_time(me, BIN_FRAMES)
            nbins = dff_b.shape[1]
            bin_sec = BIN_FRAMES / fs

            # z-score each neuron within session
            dff_b = (dff_b - dff_b.mean(axis=1, keepdims=True)) / dff_b.std(axis=1, keepdims=True)

            # time from session start (bin centres), in seconds
            tvec = (np.arange(nbins) + 0.5) * bin_sec

            # discretize motion energy into equal-percentile bins (per session)
            edges = np.percentile(me_b, np.linspace(0, 100, N_QUANTILES + 1)[1:-1])
            me_cat = np.digitize(me_b, edges).astype(np.int64)

            # cut into 60 s trials
            bins_per_trial = int(round(TRIAL_SEC / bin_sec))
            ntrials = nbins // bins_per_trial
            neural_sess, input_sess, output_sess = [], [], []
            for t in range(ntrials):
                sl = slice(t * bins_per_trial, (t + 1) * bins_per_trial)
                neural_sess.append(dff_b[:, sl].astype(np.float32))
                input_sess.append(tvec[sl][None, :].astype(np.float32))
                output_sess.append(me_cat[sl][None, :])

            neural_all.append(neural_sess)
            input_all.append(input_sess)
            output_all.append(output_sess)
            subject_idx.append(si)
            brain_region_idx.append(np.zeros(dff_b.shape[0], dtype=int))
            session_info.append({
                'subject': sub,
                'session': os.path.basename(sd),
                'day_index': sess_dirs.index(sd),
                'n_neurons': int(dff_b.shape[0]),
                'n_trials': int(ntrials),
                'imaging_rate_hz': fs,
            })
            print(f'  {sub} {os.path.basename(sd)}: {dff_b.shape[0]} neurons, '
                  f'{ntrials} trials x {bins_per_trial} bins')

    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,
        'subjects': list(subjects),
        'subject_idx': np.array(subject_idx, dtype=int),
        'brain_regions': ['S1 (barrel cortex), layer 2/3'],
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_session_start_s'],
        'output_names': ['motion_energy_quintile'],
        'output_values': [['very low', 'low', 'medium', 'high', 'very high']],
        'metadata': {
            'task_description': (
                'Spontaneous activity in layer 2/3 of mouse barrel cortex (S1) recorded with '
                '2-photon calcium imaging (GCaMP8m, 30 Hz) during the second postnatal week '
                '(P7-P14), daily for >=6 consecutive days per mouse, with the same neurons '
                'tracked across days using Track2p. Mice were head-fixed on a non-motorised '
                'treadmill in the dark with no sensory stimulation; spontaneous movement was '
                'quantified from videography as motion energy (sum of squared pixelwise '
                'differences between consecutive frames). The decoder predicts the animal\'s '
                'motion energy, discretized into 5 equal-percentile bins (quintiles computed '
                'within each session), from the neural activity plus the time elapsed since '
                'the start of the session.'),
            'time_bin_size': 1000.0 * BIN_FRAMES / 30.0,
            'temporal_alignment_event': (
                'start of each 60 s block of the continuous recording (blocks are cut '
                'consecutively from the session onset; there is no trial structure in this '
                'spontaneous-activity experiment)'),
            'off_start': 0.0,
            'off_end': 60.0,
            'neural_data_type': (
                'baseline-corrected fluorescence (dF/F, Suite2p maximin baseline as in '
                'track2p F_processing, neuropil coefficient 0), averaged in bins of 10 frames '
                'and z-scored per neuron within each session'),
            'output_data_type': (
                'motion energy averaged in bins of 10 frames, discretized into quintiles '
                'using per-session percentiles'),
            'session_info': session_info,
            'reference': ('Majnik et al. 2025, eLife 14:RP107540, "Longitudinal tracking of '
                          'neuronal activity from the same cells in the developing brain '
                          'using Track2p"'),
        },
    }

    with open(OUT_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f'saved {OUT_FILE}: {len(neural_all)} sessions, {len(subjects)} subjects')


if __name__ == '__main__':
    main()
