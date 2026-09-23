# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every `.nwb` file one level below `/app/data` is processed: `sorted(glob.glob('/app/data/*/*.nwb'))` finds all 152 files (11 subject directories `sub-m3` ... `sub-m19`). Each file is one session and is opened directly with `h5py` (not `pynwb`), reading the raw HDF5 datasets: the behaviour time series under `processing/behavior/BehavioralTimeSeries` (`position`, `speed`, `lick`, `environment`, `reward_zone`, `trial_start`, `teleport`, plus `position/timestamps` and `Reward/timestamps`), the imaging traces under `processing/ophys/Fluorescence/<plane>` and `processing/ophys/Neuropil/<plane>`, the ROI curation flags under `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell`, and the identifiers `identifier`, `general/subject/subject_id`, `general/session_id`. Sessions are processed independently in a `multiprocessing` (spawn) pool, each worker writing a per-session pickle to `/app/sessions_out/`; a second pass (`assemble()`) reads those back and concatenates them into `/app/converted_data.pkl`. All 152 sessions survive to the final file.

ii.
```python
DATA_DIR = '/app/data'
...
def process_session(fn):
    with h5py.File(fn, 'r') as f:
        ident = f['identifier'][()].decode()
        scene = ident.split('/')[-1]
        subject = f['general/subject/subject_id'][()].decode()
        session_id = f['general/session_id'][()].decode()
        b = f['processing/behavior/BehavioralTimeSeries']
        get = lambda k: b[k + '/data'][()]
        pos = get('position')
        speed = get('speed')
        lick = get('lick')
        env = get('environment')
        rzone_ts = get('reward_zone')
        trial_start = get('trial_start')
        teleport = get('teleport')
        tstamps = b['position/timestamps'][()]
        reward_times = b['Reward/timestamps'][()]

        # imaging: concatenate planes (pooled for all analyses in the paper)
        planes = sorted(f['processing/ophys/Fluorescence'].keys())
        F = np.concatenate([f['processing/ophys/Fluorescence/%s/data' % p][()] for p in planes], axis=1).T
        Fneu = np.concatenate([f['processing/ophys/Neuropil/%s/data' % p][()] for p in planes], axis=1).T
        ps = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
        iscell = ps['iscell'][:, 0] > 0
```
```python
if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.nwb')))
    ...
    ctx = mp.get_context('spawn')
    with ctx.Pool(nproc, maxtasksperchild=1) as p:
        for _ in p.imap_unordered(process_session, files):
            pass
    assemble()
```

iii. From the trajectory: the agent first listed `/app/data`, dumped the full HDF5 tree of one file to `struct.txt`, and read `dandiset.yaml` and the NWB dataset attributes/descriptions before deciding what to read. It then ran a metadata-only scan (`scan.py`/`scan2.py`) over all 152 files and confirmed "152 sessions total, 87 GB of NWB ... All sessions are single-plane [except m17/m18], ~80 trials each, 15.5 Hz sampling, 152 sessions, 11 mice". It chose raw `h5py` over `pynwb` for speed after finding the run was I/O-bound, and chose per-session pickles plus a `spawn` pool after diagnosing a fork/OpenMP deadlock ("Fork + OpenMP problem. Use 'spawn' start method").

## 1-b. How are the data split into subjects?

i. Subject identity is read from each NWB file's `general/subject/subject_id` field rather than from the directory name. `assemble()` builds `data['subjects']` as the list of unique ids in file order (which, because the files are sorted, is `m11, m12, m13, m14, m15, m17, m18, m19, m3, m4, m7`), and records `subject_idx` per session by looking the id up in that list. 11 subjects result, matching the 11 mice in the paper.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
...
    m = r['meta']
    if m['subject'] not in subjects:
        subjects.append(m['subject'])
    ...
    data['subject_idx'].append(subjects.index(m['subject']))
data['subjects'] = subjects
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. The agent's exploration script printed the subject metadata for a file and confirmed the id matches the `sub-<id>` directory name and the `GCAMP<N>` animal name embedded in `identifier` (e.g. `/data/InVivoDA/GCAMP17/26_03_2024/Env2_LocationB` for `sub-m17`). Its scan across all files reported "152 sessions, 11 mice", matching the paper's 11 switch mice.

## 1-c. How are the data split into sessions?

i. One `.nwb` file = one session. No merging across days and no cross-day ROI alignment is attempted (the paper's `multiDayROIAlign` is ignored). The session number is taken from `general/session_id` and stored in the metadata as `session_id`/`exp_day`. A session is dropped from the final file only if it would contribute fewer than 2 trials (this never triggers: all 152 sessions are kept).

ii.
```python
session_id = f['general/session_id'][()].decode()
...
meta = dict(file=os.path.basename(fn), subject=subject, session_id=session_id,
            exp_day=int(session_id), scene=scene, identifier=ident, ...)
```
```python
    if len(r['neural']) < 2:
        print('skipping session with <2 trials:', pkl)
        continue
```

iii. The file naming `sub-<id>_ses-<NN>_behavior+ophys.nwb` and the `session_id` field make the one-file-one-session mapping explicit; the agent's scan confirmed one scene name and one contiguous behaviour timeline per file. The `< 2 trials` guard is the format requirement stated in the instructions ("There needs to be at least two trials within each session").

## 1-d. How are the data split into trials?

i. A trial is one lap of the virtual track: it starts at the sample where the `trial_start` time series is positive and ends at (exclusive of) the sample where the `teleport` time series first goes positive. The inter-trial teleport period is therefore excluded from the trial entirely. The code asserts that the number of starts equals the number of teleports and that every teleport follows its start.

ii.
```python
starts = np.where(trial_start > 0)[0]
stops = np.where(teleport > 0)[0]
assert len(starts) == len(stops) and np.all(stops > starts)
n_trials = len(starts)
...
for t in range(n_trials):
    s, e = starts[t], stops[t]
```

iii. Docstring: "Trials run from trial_start to teleport (the VR linear track lap); the inter-trial 'teleport' period is excluded". The agent took this directly from the paper's code, where `sess.trial_start_inds` / `sess.teleport_inds` are used as the lap window everywhere (`behavior.get_trial_types`, `preprocessing.dff`, `glmUtils.get_timeseries_data`). It explicitly compared against the `trial number` time series during exploration and checked trial-length statistics ("trial len frames: min ... median ... max 3359") and per-trial positions before committing.

## 1-e. How are trials filtered based on quality controls?

i. Two trial-level filters, both applied after the neural signal is computed:
- **Lick-sensor error trials**: a trial is dropped if more than 30% of its imaging samples have a cumulative lick count > 2. This is the paper's `behavior.correct_lick_sensor_error` criterion with the threshold stated in the Methods (>30%). It removes exactly 81 trials, the number the Methods report.
- **First trial of each session**: dropped, because the decoder input "previous trial outcome" is undefined for it.
No minimum trial-length filter is applied (the shortest surviving trial is 96 samples). Result: 11,983 trials kept of 12,216.

ii.
```python
LICK_ERR_THRESH = 0.3      # fraction of samples with cumulative lick count > 2
...
    lk = lick[s:e]
    lick_err = np.sum(lk > 2) / len(lk) > LICK_ERR_THRESH
    n_lick_err += int(lick_err)
...
for t in range(n_trials):
    tr = trials[t]
    if t == 0 or tr['lick_err']:
        continue  # no previous-trial outcome / unusable lick data
```

iii. Docstring: "Trials with lick-sensor errors (>30% of imaging samples in the trial with a cumulative lick count > 2) are dropped, as those lick data are unusable (the paper sets them to NaN; 81/12216 trials here, matching the paper)" and "The first trial of each session is dropped because the decoder input 'previous trial outcome' is undefined for it." The agent read `behavior.correct_lick_sensor_error`, noted the Methods sentence ("n = 81 out of 12,376 trials"), and verified after conversion: "Conversion matches paper: 81 lick-error trials (paper: 81)". It reasoned that because lick is a required decoder output, the paper's NaN-masking is not representable, so the whole trial is removed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Raw suite2p traces: `processing/ophys/Fluorescence/<plane>/data` (F) and `processing/ophys/Neuropil/<plane>/data` (Fneu). The NWB `Deconvolved` array is deliberately **not** used. For the two-plane animals (m17, m18) the two planes are concatenated along the cell axis and pooled.

ii.
```python
planes = sorted(f['processing/ophys/Fluorescence'].keys())
F = np.concatenate([f['processing/ophys/Fluorescence/%s/data' % p][()] for p in planes], axis=1).T
Fneu = np.concatenate([f['processing/ophys/Neuropil/%s/data' % p][()] for p in planes], axis=1).T
```

iii. Docstring: "Neural activity = deconvolved calcium 'events' computed exactly as in reward_relative.preprocessing.dff ... This 'events' timeseries is what the paper used for decoding (glmUtils.get_timeseries_data)." The agent inspected the stored `Deconvolved` array and concluded "NWB Deconvolved appears to be suite2p's raw-F deconvolution (values ~300, no NaNs outside trials), not the paper's dF/F-based pipeline", so it recomputes from F/Fneu instead.

## 2-b. How is the `neural` data processed?

i. A re-implementation of the paper's `preprocessing.dff`, applied independently within each lap `[trial_start, teleport)`:
1. neuropil subtraction `F - 0.7*Fneu`, then the trial-mean neuropil `+ 0.7*mean(Fneu)` is added back so the ratio is a true dF/F;
2. maximin baseline: Gaussian smoothing with sigma = 15 samples along time, then a 300-sample running minimum followed by a 300-sample running maximum (~20 s at 15.5 Hz);
3. `dF/F = (F - F0)/|F0|`;
4. Gaussian smoothing with sigma = 2 samples;
5. `nan_to_num` (NaN/±inf → 0);
6. OASIS deconvolution via `suite2p.extraction.dcnv.oasis(d, 2000, tau=0.7, fs)` with `fs = 1/median(diff(timestamps))` ≈ 15.5 Hz (the per-plane rate, which is half the 31 Hz scanner rate on the two-plane sessions).
Samples outside laps are left NaN and never emitted. The baseline window is **always** restricted to the lap (`keep_teleports` is never enabled).

ii.
```python
def compute_events(F, Fneu, starts, stops, fs):
    """Per-trial dF/F and OASIS-deconvolved events, following preprocessing.dff."""
    n_cells, n_frames = F.shape
    dff = np.full((n_cells, n_frames), np.nan, dtype=np.float32)
    events = np.full((n_cells, n_frames), np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        f = F[:, s:e].astype(np.float64)
        fneu = Fneu[:, s:e].astype(np.float64)
        f = f - NEU_COEF * fneu + NEU_COEF * np.mean(fneu, axis=1, keepdims=True)
        flow = gaussian_filter1d(f, BASELINE_SMOOTH, axis=1)
        flow = minimum_filter1d(flow, BASELINE_WIN, axis=-1)
        flow = maximum_filter1d(flow, BASELINE_WIN, axis=-1)
        d = (f - flow) / np.abs(flow)
        d = gaussian_filter1d(d, DFF_SMOOTH, axis=1)
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        dff[:, s:e] = d
        events[:, s:e] = dcnv.oasis(d, 2000, TAU, fs)
    return dff, events
```
```python
NEU_COEF = 0.7; TAU = 0.7; BASELINE_SMOOTH = 15; BASELINE_WIN = 300; DFF_SMOOTH = 2
...
dt = float(np.median(np.diff(tstamps)))
fs = 1.0 / dt
...
dff, events = compute_events(F, Fneu, starts, stops, fs)
```

iii. Docstring: "per-trial maximin baseline (smooth sigma=15 samples, 300-sample min then max filter ~ 20 s), dF/F = (F-F0)/|F0|, 2-sample Gaussian smoothing, then OASIS deconvolution (tau=0.7, per-plane frame rate ~15.5 Hz)". The agent read `preprocessing.dff` and `utilities.multi_anim_sess`, found `neu_coef = 0.7` and `baseline_method = 'maximin'` in `make_multi_anim_sess`, and grepped the repo for `tau` to get 0.7. It also inspected `TwoPUtils.nansmooth` to reproduce the smoothing. On `keep_teleports` the docstring asserts "keep_teleports=False, the default used for the paper" — the agent did not look at `teleport_metadata.py`/`make_multi_anim_sess`, where the paper in fact sets `keep_teleports=True` per animal and experiment day.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two cell-level filters, both the paper's:
- **suite2p `iscell`**: only ROIs with `iscell[:,0] > 0` (manually curated) are kept. This is applied to F/Fneu *before* dF/F is computed, using the global `PlaneSegmentation` table, whose row order matches the plane0-then-plane1 concatenation order.
- **Putative interneurons**: cells whose dF/F correlates with running speed at Pearson r > 0.5, computed over all in-trial samples pooled across the session, are dropped. The correlation is computed in a single vectorised expression rather than a per-cell loop.
Result: 138,276 cells kept of 138,678 `iscell` ROIs (402 interneurons, 0.29%).

ii.
```python
ps = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
iscell = ps['iscell'][:, 0] > 0
...
F = F[iscell]
Fneu = Fneu[iscell]
...
# exclude putative interneurons: dF/F correlated with running speed
in_trial = np.zeros(len(pos), dtype=bool)
for s, e in zip(starts, stops):
    in_trial[s:e] = True
sp_ = speed[in_trial]
d_ = dff[:, in_trial]
d_c = d_ - d_.mean(axis=1, keepdims=True)
s_c = sp_ - sp_.mean()
denom = np.sqrt((d_c ** 2).sum(axis=1) * (s_c ** 2).sum())
r = np.divide((d_c * s_c).sum(axis=1), denom, out=np.zeros(d_.shape[0]), where=denom > 0)
keep_cells = r <= INT_R_THRESH
events = events[keep_cells]
```

iii. Docstring: "Cells: suite2p iscell==1 (manually curated in the paper), minus putative interneurons (Pearson r between dF/F and running speed > 0.5, as in dayData int_thresh=0.5 / spatial.is_putative_interneuron)." The agent grepped for `r_thresh`/`int_thresh`, found `spatial.is_putative_interneuron` defaults to 0.3 but that `dayData` and all the analysis notebooks pass `int_thresh = 0.5`, and that the Methods quote 0.5. After conversion it checked the result against the paper: "0.29% interneurons excluded (paper: 0.42±0.85%)".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Nothing beyond the trial split is needed: the alignment event is trial start, and each trial's neural matrix is exactly `events[:, trial_start : teleport]`, so sample 0 of every trial is the trial-start sample. No padding, no pre-trial window (`off_start = 0.0`, `off_end = None` because trials have variable duration).

ii.
```python
s, e = tr['s'], tr['e']
T = e - s
neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
```
```python
temporal_alignment_event='trial start (entry into the virtual linear track at 0 cm)',
off_start=0.0,
off_end=None,
trial_definition=('each trial spans trial_start to teleport (one lap of the track); '
                  'trials have variable duration so off_end is not fixed'),
```

iii. The instructions say "Temporally align based on start of the trial"; since the trial window already begins at trial start, the split *is* the alignment. Imaging and behaviour share the same sample grid, so all streams are sliced with the same `[s:e)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling. Data are kept at the native imaging/VR sampling rate of ~15.507 Hz, i.e. a bin of 64.4836 ms. `time_bin_size` in the metadata is the mean across sessions of `median(diff(position/timestamps)) * 1000`, which comes out to 64.4836 ms; the sampling interval is identical to ~1e-12 s across all sessions and both single- and two-plane recordings (for two planes the scanner runs at 31.0156 Hz, so the per-plane rate is again 15.5 Hz).

ii.
```python
dt = float(np.median(np.diff(tstamps)))
fs = 1.0 / dt
...
dts = [s['dt'] for s in session_info]
data['metadata'] = dict(
    ...
    time_bin_size=float(np.mean(dts)) * 1000.0,
    ...
    imaging_rate_hz=float(1.0 / np.mean(dts)),
```

iii. The agent verified during its scan that every session samples at 15.5 Hz / 64.5 ms bins and considered coarser binning purely for decoder runtime ("Consider binning time to e.g. 5 samples (~322 ms). Hmm, paper uses native frames"), then decided against it after checking the decoder's memory and runtime behaviour, keeping the paper's native resolution.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. From `processing/behavior/BehavioralTimeSeries/position/timestamps`: only its median inter-sample interval `dt` is used, and the within-trial time axis is reconstructed as a uniform grid `0, dt, 2dt, ...` of length equal to the trial's sample count.

ii.
```python
tstamps = b['position/timestamps'][()]
...
dt = float(np.median(np.diff(tstamps)))
...
time_s = np.arange(T, dtype=np.float32) * dt
```

iii. The agent's exploration confirmed that all behaviour series share one timestamp vector sampled at the imaging frame rate; it therefore treated the grid as uniform. (Checked here: the timestamp vector really is uniform to ~1e-12 s and `position/timestamps` is identical to `trial number/timestamps`, so the reconstructed axis equals the true elapsed time.)

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The time axis is regenerated per trial as `np.arange(T) * dt`, i.e. it is zero at the trial-start sample and increments by the constant frame period. Equivalent to subtracting the first timestamp of the trial, but built from the nominal period rather than the stored timestamps. It is stored as input row 0, in seconds, time-varying.

ii.
```python
time_s = np.arange(T, dtype=np.float32) * dt
inp.append(np.stack([
    time_s,
    np.full(T, tr['env'], dtype=np.float32),
    np.full(T, t, dtype=np.float32),
    np.full(T, trials[t - 1]['rewarded'], dtype=np.float32),
]).astype(np.float32))
```

iii. Not explicitly discussed in the trajectory beyond the sanity check at step 87, where the agent printed `input trial0 t0..5 = [0., 0.06448363, 0.12896726]` and confirmed the first input row starts at 0 and steps by one frame period.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Behaviour and imaging are on the same sample grid, so no resampling or shifting is done: the input array is built with the same length `T = e - s` as the neural slice. The only alignment work is a length reconciliation at load time — in 10 of the two-plane sessions the imaging arrays have exactly one extra frame relative to the VR series, and F/Fneu are truncated to the common length.

ii.
```python
# a few sessions have one extra imaging frame relative to the VR timeseries;
# truncate to the common length so imaging and behavior stay aligned
n_common = min(F.shape[1], len(pos))
F = F[:, :n_common]
Fneu = Fneu[:, :n_common]
```

iii. The agent hit this while running the full conversion, then scanned every file for the mismatch: "Only off-by-one mismatches (imaging has 1 extra frame) in a few 2-plane sessions. Truncate imaging to behavior length." It listed the 10 affected sessions (m17 ses-04/06, m18 ses-01/05/07/10/11/12/13/14) before patching.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behaviour time series (values 0 = ENV1, 1 = ENV2).

ii.
```python
env = get('environment')
...
trials.append(dict(..., env=int(np.round(np.median(env[s:e])))))
```

iii. The agent's per-trial scan printed `np.unique(env[s:e])` for sample trials and recorded `env=float(np.median(env[s:e]))` per trial across all 152 sessions, establishing that the variable is binary and constant within a trial, and that it changes at trial 30 in the `Env1_X_to_Env2_Y` switch sessions.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The per-trial value is the rounded median of `environment` over the trial's samples, then broadcast as a constant across all timepoints of the trial (input row 1). Using the median makes the value robust to any single-sample glitch; because the variable is in fact constant within every trial (verified: 0 trials with non-constant `environment`), this is identical to copying the raw time series.

ii.
```python
env=int(np.round(np.median(env[s:e])))
...
np.full(T, tr['env'], dtype=np.float32),
```

iii. Implied by the scan showing one unique environment value per trial; the task specifies environment as a per-trial binary input.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is the within-session index of the trial in the `trial_start`/`teleport` sequence — the loop counter `t`, not the NWB `trial number` time series.

ii.
```python
for t in range(n_trials):
    ...
    np.full(T, t, dtype=np.float32),
```

iii. The agent examined the stored `trial number` series during exploration (it prints `np.unique(tn[s:e])` per trial) but built trial boundaries from `trial_start`/`teleport` as the paper does, so the lap index is the natural and self-consistent trial number.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond broadcasting the integer index across the trial's timepoints (input row 2). The index is taken over *all* trials in the session, so it is unaffected by the trials that are subsequently dropped; because the first trial of each session is always dropped, the emitted values run 1 … n_trials-1 (observed range 1–99).

ii.
```python
np.full(T, t, dtype=np.float32),
```

iii. Not separately justified; it is the sequential lap index within a session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the reward event timestamps, `processing/behavior/BehavioralTimeSeries/Reward/timestamps`. Those event times are mapped onto behaviour-sample indices with `np.searchsorted` against the behaviour timestamp vector, and a trial counts as rewarded if any reward index falls in `[trial_start, teleport)`.

ii.
```python
reward_times = b['Reward/timestamps'][()]
...
reward_idx = np.searchsorted(tstamps, reward_times)
...
    rewarded = int(np.any((reward_idx >= s) & (reward_idx < e)))
    trials.append(dict(..., rewarded=rewarded, ...))
```

iii. The agent's exploration established that `Reward` is an event series with its own timestamps rather than a per-sample signal, so it converted event times to sample indices. Its scan also cross-checked rewards against reward-zone entries ("rz active but no reward", "reward but no rz") and confirmed that ~15% of trials have no reward, matching the paper's stated omission rate.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each emitted trial `t`, the value is the `rewarded` flag of trial `t-1` in the full (unfiltered) trial list, broadcast constant across all timepoints (input row 3). There is no imputed value for the first trial of a session: that trial is dropped instead.

ii.
```python
np.full(T, trials[t - 1]['rewarded'], dtype=np.float32),
...
if t == 0 or tr['lick_err']:
    continue  # no previous-trial outcome / unusable lick data
```

iii. Docstring: "The first trial of each session is dropped because the decoder input 'previous trial outcome' is undefined for it." Using `trials[t-1]` from the unfiltered list means the "previous trial" is the trial that actually preceded it in time, even if that trial was itself removed for a lick-sensor error.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` behaviour time series together with the trial's reward-zone identity. The zone identity is derived from the NWB `identifier` string, whose last path component is the paper's scene name (e.g. `Env1_LocationB_to_A`, `Env1_C_to_Env2_A`, `Env2_LocationC`): the character before and after `_to_` give the pre-switch and post-switch zone letters, and the switch happens at trial index 30. The zone coordinates are the paper's `behavior.reward_zone_dict` entries X/Y/Z: A = [80, 130], B = [200, 250], C = [320, 370] cm. The `reward_zone` time series is used only as an independent cross-check, not as the source.

ii.
```python
ZONES = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
ZONE_IDX = {'A': 0, 'B': 1, 'C': 2}
CHANGE_TRIAL = 30          # reward zone switches on trial 30 (0-indexed)

def scene_zones(scene):
    """Reward zone label(s) for a scene name, e.g. Env1_LocationB_to_A or Env1_C_to_Env2_A."""
    if '_to_' in scene:
        left, right = scene.split('_to_')
        return left[-1], right[-1]
    return scene[-1], scene[-1]
...
z0, z1 = scene_zones(scene)
zone_label = [z0 if t < CHANGE_TRIAL else z1 for t in range(n_trials)]
```
Cross-check against the `reward_zone` time series:
```python
    # sanity check the scene-derived zone against reward-zone entry times
    m = rzone_ts[s:e] > 0
    if m.sum() > 0:
        p_in = pos[s:e][m]
        lo, hi = ZONES[zone_label[t]]
        if not (lo - 15 <= p_in.min() <= hi + 15):
            n_zone_mismatch += 1
```

iii. Docstring: "Reward zones: A=[80,130], B=[200,250], C=[320,370] cm (behavior.reward_zone_dict), assigned from the scene name with the switch at trial index 30 (behavior.get_reward_zones change_trial=30)." The agent read `behavior.get_reward_zones` and `reward_zone_dict` (noting the paper's `A→X`, `B→Y`, `C→Z` label mapping), found that the scene name is preserved in the NWB `identifier`, grepped the repo for any `change_reward_trial` override and found none, and then empirically validated the rule on all 152 sessions using reward-zone entry positions: the scan found the switch at index 30 in 64 switch sessions and at 31/32 in 13 more (consistent with the mouse simply not triggering the zone on trial 30), and the in-conversion cross-check reported "zone_mismatch 0" across all 12,216 trials.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the signed linear distance from the animal's position to the nearest edge of that trial's reward zone: negative before the zone (`pos - lo`), exactly 0 while inside `[lo, hi]`, positive after (`pos - hi`). This is computed in centimetres on the raw `position` values, then discretised (7-c).

ii.
```python
p = pos[s:e]
lo, hi = ZONES[tr['zone']]
d = np.zeros(T)
d[p < lo] = p[p < lo] - lo
d[p > hi] = p[p > hi] - hi
```

iii. Docstring notes the deliberate deviation from the paper's own analysis: "The paper decoded reward-relative position in circular (radian) coordinates; here the task specifies signed linear distance (cm) to the reward zone, discretised into the seven bins listed in the task description."

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Into the seven bins specified in the instructions, by explicit boolean masks with default value 3 (in-zone):
0: d < -50; 1: -50 ≤ d < -10; 2: -10 ≤ d < 0; 3: d == 0 (inside the zone); 4: 0 < d ≤ 10; 5: 10 < d ≤ 50; 6: d > 50.

ii.
```python
# 0: <-50, 1: [-50,-10), 2: [-10,0), 3: 0, 4: (0,10], 5: (10,50], 6: >50
rd = np.full(T, 3, dtype=np.int64)
rd[d < -50] = 0
rd[(d >= -50) & (d < -10)] = 1
rd[(d >= -10) & (d < 0)] = 2
rd[(d > 0) & (d <= 10)] = 4
rd[(d > 10) & (d <= 50)] = 5
rd[d > 50] = 6
```
with labels
```python
['< -50 cm', '-50 to -10 cm', '-10 to 0 cm', 'in reward zone (0 cm)',
 '0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
```

iii. The bin edges are copied verbatim from the Decoder Output specification in the instructions. The agent sanity-checked the result at step 87 by cross-tabulating distance bin against position bin and confirming that bin 3 only occurs at positions inside the zone.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is sliced with the same `[s:e)` index range as the neural data, on the shared behaviour/imaging sample grid, so alignment is automatic and the output array has the same `T` as the neural matrix.

ii.
```python
s, e = tr['s'], tr['e']
T = e - s
neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
...
p = pos[s:e]
```

iii. Guaranteed by the length reconciliation at load (`n_common`), which keeps the imaging and VR arrays on a common index base; the agent verified the relationship between lick, position bin and distance bin on a converted session as a post-hoc alignment check.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behaviour time series (cm along the 450 cm virtual corridor).

ii.
```python
pos = get('position')
...
p = pos[s:e]
```

iii. Directly the VR position variable; the agent checked its per-trial range during exploration (`posmin`, `posmax` in the scan) and confirmed the track spans 0–450 cm within laps.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The raw values are clipped to `[0, 450)` and then digitised. The clip only affects the handful of samples marginally outside the track (it pushes them into the first/last bin, which is where open-ended edges would put them anyway).

ii.
```python
TRACK_LENGTH = 450.0
...
pbin = np.digitize(np.clip(p, 0, TRACK_LENGTH - 1e-6), [90.0, 180.0, 270.0, 360.0]).astype(np.int64)
```

iii. The 450 cm track length is from the Methods; the instructions ask for "5 equal-sized bins spanning the 450 cm track".

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90 cm bins over the 450 cm track: 0: < 90, 1: 90–180, 2: 180–270, 3: 270–360, 4: ≥ 360 cm, via `np.digitize` with interior edges `[90, 180, 270, 360]`.

ii.
```python
pbin = np.digitize(np.clip(p, 0, TRACK_LENGTH - 1e-6), [90.0, 180.0, 270.0, 360.0]).astype(np.int64)
...
['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm'],
```

iii. Copied from the Decoder Output specification.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[s:e)` slice as the neural data on the shared sample grid; no additional alignment.

ii.
```python
p = pos[s:e]
```

iii. Same reasoning as 7-d.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behaviour time series (a per-sample cumulative lick count).

ii.
```python
lick = get('lick')
...
lk = (lick[s:e] > 0).astype(np.int64)
```

iii. The agent inspected the value range of `lick` during exploration (values can exceed 1, and stuck-sensor trials show sustained counts > 2), which is what motivated both the binarisation and the lick-error trial filter.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarisation: any sample with a positive lick count becomes 1, otherwise 0. Trials where the sensor was stuck (the >30% / count>2 criterion) are not corrected in place — the whole trial is removed (see 1-e), so no NaN handling is required.

ii.
```python
lk = (lick[s:e] > 0).astype(np.int64)
...
    lick_err = np.sum(lk > 2) / len(lk) > LICK_ERR_THRESH
```

iii. The instructions require a binary lick output. The paper's Methods likewise say "Remaining lick counts were converted to a binary vector".

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[s:e)` slice on the shared sample grid.

ii.
```python
lk = (lick[s:e] > 0).astype(np.int64)
```

iii. The agent verified alignment functionally at step 87 by checking that lick fraction is highest in the position bin containing the reward zone (e.g. for `m11 ses-03`, lick fraction 0.321 in position bin 0 for a zone-A session, falling to 0.015 in bin 4).

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Same source as 7-a: the scene name parsed out of the NWB `identifier`, with the switch at trial index 30, mapped through the paper's `reward_zone_dict`; independently cross-checked against the `reward_zone` time series and `position`.

ii.
```python
z0, z1 = scene_zones(scene)
zone_label = [z0 if t < CHANGE_TRIAL else z1 for t in range(n_trials)]
...
ZONE_IDX = {'A': 0, 'B': 1, 'C': 2}
```

iii. See 7-a: this reproduces `behavior.get_reward_zones(sess, change_trial=30)` from the paper's own repository, and the in-code cross-check against reward-zone entry positions reported zero mismatches over all 12,216 trials.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The zone letter is mapped to 0/1/2 for A/B/C and broadcast as a constant across all timepoints of the trial (output row 4), so it is stored as a time-varying row with a constant value.

ii.
```python
out.append(np.stack([
    rd, pbin, sbin, lk,
    np.full(T, ZONE_IDX[tr['zone']], dtype=np.int64),
    np.full(T, tr['rewarded'], dtype=np.int64),
]).astype(np.int64))
...
['zone A (80-130 cm)', 'zone B (200-250 cm)', 'zone C (320-370 cm)'],
```

iii. The instructions specify a per-trial categorical output "0 = A, 1 = B, 2 = C", and also ask that outputs be time-varying "if at all possible", hence the broadcast. The resulting class balance (0.331 / 0.336 / 0.332) is close to uniform, as expected from the design.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` event timestamps (`processing/behavior/BehavioralTimeSeries/Reward/timestamps`), converted to behaviour-sample indices with `np.searchsorted`.

ii.
```python
reward_times = b['Reward/timestamps'][()]
...
reward_idx = np.searchsorted(tstamps, reward_times)
```

iii. Same as 6-a: the agent determined from exploration that reward is an event series with its own timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is rewarded (1) if at least one reward event index falls within `[trial_start, teleport)`, otherwise 0 (omitted). The flag is broadcast across all timepoints of the trial (output row 5). The same flag is reused, shifted by one trial, for the "previous trial outcome" input.

ii.
```python
rewarded = int(np.any((reward_idx >= s) & (reward_idx < e)))
...
np.full(T, tr['rewarded'], dtype=np.int64),
...
['omitted', 'rewarded'],
```

iii. The instructions define reward outcome as a per-trial binary. The agent's scan cross-checked reward events against reward-zone entries and against the ~15% omission rate described in the paper; the converted data give 15.7% omitted / 84.3% rewarded.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several defensive behaviours:
- **Imaging/behaviour length mismatch**: imaging arrays are truncated to `min(n_imaging_frames, len(position))`. In practice this is always a one-frame excess on the imaging side in 10 two-plane sessions.
- **Trial structure**: an assertion requires the number of `trial_start` events to equal the number of `teleport` events and every teleport to follow its start; a violation aborts that session's worker.
- **Undefined dF/F**: `np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)` replaces any NaN/±inf produced by a zero or near-zero baseline with 0. This is silent — nothing is counted or reported.
- **Unusable lick data**: trials flagged by the stuck-sensor criterion are removed entirely rather than NaN-masked.
- **Missing previous-trial information**: the first trial of each session is removed rather than given an imputed outcome.
- **Sessions that would yield too little data**: sessions contributing fewer than 2 trials, and sessions whose per-session pickle is missing, are skipped with a printed message.
- **Consistency diagnostics**: `n_zone_mismatch` counts trials whose scene-derived reward zone disagrees with the observed reward-zone entry position; it is stored in the metadata (it is 0 everywhere) but does not change any data.

ii.
```python
n_common = min(F.shape[1], len(pos))
F = F[:, :n_common]
Fneu = Fneu[:, :n_common]
...
assert len(starts) == len(stops) and np.all(stops > starts)
...
d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
...
    if not os.path.exists(pkl):
        print('missing', pkl)
        continue
    ...
    if len(r['neural']) < 2:
        print('skipping session with <2 trials:', pkl)
        continue
```

iii. The agent discovered the length mismatch empirically during the first full run, scanned all files to bound its magnitude, and patched the code ("Only off-by-one mismatches (imaging has 1 extra frame) in a few 2-plane sessions. Truncate imaging to behavior length."). It also ran a pre-conversion check over all files for NaNs in F, non-monotonic trial boundaries, and `scanning != 1` inside trials, finding no problems other than the two-plane sessions.

## 13-a. What are the most time-consuming steps of the code?

i. In rough order:
1. **Reading the NWB arrays** — 87 GB of HDF5 across 152 files, mostly F and Fneu; the agent measured the first (serial-ish) attempt as I/O bound ("the run used almost no CPU (17s user in 5.4 min)").
2. **dF/F + OASIS deconvolution** (`compute_events`): a Python loop over ~80 trials per session, each doing a Gaussian smooth, a 300-sample min filter, a 300-sample max filter and an OASIS call on an (n_cells × T) block. This is the dominant CPU cost, ~0.5–3 s per session.
3. **Writing and then re-reading the 152 per-session pickles** (8.9 GB) and **writing the final 9.5 GB `converted_data.pkl`**.
4. Process startup: `maxtasksperchild=1` with the `spawn` context re-imports numpy/suite2p for every one of the 152 sessions.
The whole conversion runs in roughly 5 minutes wall-clock with 16 worker processes.

ii.
```python
    ctx = mp.get_context('spawn')
    with ctx.Pool(nproc, maxtasksperchild=1) as p:
        for _ in p.imap_unordered(process_session, files):
            pass
```
```python
    with open(os.path.join(OUT_DIR, os.path.basename(fn).replace('.nwb', '.pkl')), 'wb') as fo:
        pickle.dump(res, fo, protocol=4)
```

iii. The agent profiled this directly: it interrupted a stalled run, timed a raw HDF5 read ("Reading is fast"), timed `process_session` inline ("Single-session processing takes only 0.5 s; the Pool run hung (nearly no CPU)"), diagnosed a fork/OpenMP deadlock, and switched to the `spawn` start method.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining Python-level loops are:
- `compute_events`: `for s, e in zip(starts, stops)` — one iteration per trial, each doing four array ops plus OASIS. Because the maximin baseline must not cross trial boundaries, this cannot be fully vectorised without padding/masking tricks; the per-iteration arrays are large so the loop overhead is negligible.
- The per-trial metadata loop and the per-trial array-building loop in `process_session` — `lick_err`, `rewarded`, the zone check, `np.digitize` on position/speed and the distance computation could all be done once on the whole-session arrays and then split, but variable trial lengths make this awkward and the work is small relative to the deconvolution.
- `for s, e in zip(starts, stops): in_trial[s:e] = True` — could be built with a cumulative-sum trick, but it is trivially cheap.
The interneuron speed correlation, which the paper's own `is_putative_interneuron` computes with a per-cell `np.corrcoef` loop, *is* fully vectorised here into a handful of array operations.

ii.
```python
d_c = d_ - d_.mean(axis=1, keepdims=True)
s_c = sp_ - sp_.mean()
denom = np.sqrt((d_c ** 2).sum(axis=1) * (s_c ** 2).sum())
r = np.divide((d_c * s_c).sum(axis=1), denom, out=np.zeros(d_.shape[0]), where=denom > 0)
```

iii. Not discussed in the trajectory; the agent's efficiency effort went into parallelism (a 16-way spawn pool) rather than into removing the remaining per-trial loops.

## 13-c. What processing does the code repeat multiple times?

i. - **Disk round-trip**: every session is serialised to `/app/sessions_out/<session>.pkl` and then immediately read back by `assemble()` in the same run, so ~9 GB is written and re-read unnecessarily when the pipeline is run end-to-end (the split does buy restartability and lets `assemble` be re-run cheaply, which the agent used when it revised the metadata).
- **Per-trial loop run twice**: `process_session` loops over all trials once to build the `trials` metadata list (lick error, reward, zone check, environment) and a second time to build the neural/input/output arrays.
- **Exploration passes**: before conversion the agent ran two separate full scans of all 152 files (`scan.py`, `scan2.py`) that re-read the behaviour series later read again by `convert_data.py`. These are separate scripts, not part of the deliverable.
- `sorted(glob.glob(...))` over the data directory is computed twice (once in `__main__`, once in `assemble`).

ii.
```python
    with open(os.path.join(OUT_DIR, os.path.basename(fn).replace('.nwb', '.pkl')), 'wb') as fo:
        pickle.dump(res, fo, protocol=4)
...
def assemble():
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.nwb')))
    for fn in files:
        pkl = os.path.join(OUT_DIR, os.path.basename(fn).replace('.nwb', '.pkl'))
        ...
        with open(pkl, 'rb') as fi:
            r = pickle.load(fi)
```

iii. The two-stage design is a consequence of the multiprocessing choice: workers cannot cheaply return multi-hundred-MB arrays through the pool, so they write to disk instead. The agent exploited this at the end, re-running only `assemble` ("re-assemble the final pickle (fast, no reprocessing)") after editing the metadata.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. - **dF/F and OASIS events are computed for all trials, including the ~233 trials that are then dropped** (152 first trials + 81 lick-error trials), and **for all `iscell` cells, including the 402 later removed as putative interneurons**. The interneuron removal is unavoidable in this order (the dF/F is what the speed correlation is computed on), but the dropped trials could have been excluded before deconvolution.
- **The full-session `dff` array is materialised** (n_cells × n_frames, float32) purely to compute the speed correlation, and is then deleted; only `events` is kept.
- **8.9 GB of intermediate per-session pickles are left on disk** after `assemble()` finishes; they are an exact duplicate of what is in the 9.5 GB final file and are never cleaned up.
- **Diagnostics that do not affect the output**: `n_zone_mismatch` (the reward-zone cross-check runs on every trial of every session), `n_cells_iscell`, `n_interneurons`, `kept_trials`, `identifier` — all stored in `metadata['session_info']` and unused by the decoder.
- The `--limit` / `--nproc` / `assemble` CLI plumbing and `sample_trials.png` plotting are development aids.
Nothing computed is *wrong*, but the pipeline does roughly 2% more deconvolution than needed and doubles its disk footprint.

ii.
```python
    dff, events = compute_events(F, Fneu, starts, stops, fs)
    del F, Fneu
    ...
    events = events[keep_cells]
    del dff, d_, d_c
```
```python
        m = rzone_ts[s:e] > 0
        if m.sum() > 0:
            p_in = pos[s:e][m]
            lo, hi = ZONES[zone_label[t]]
            if not (lo - 15 <= p_in.min() <= hi + 15):
                n_zone_mismatch += 1
```
```python
    meta = dict(file=..., n_trials_total=n_trials, n_trials_kept=len(kept_trials),
                n_lick_error_trials=n_lick_err, n_zone_mismatch=n_zone_mismatch,
                n_cells_iscell=int(iscell.sum()), n_cells_kept=int(keep_cells.sum()),
                n_interneurons=int((~keep_cells).sum()), dt=dt, kept_trials=kept_trials,
                n_planes=len(planes))
```

iii. The agent used the diagnostics as its validation evidence ("Conversion matches paper: 81 lick-error trials (paper: 81), 0.29% interneurons excluded (paper: 0.42±0.85%), no reward-zone mismatches"), so the extra bookkeeping was a deliberate cost. It did reduce the output footprint where it mattered by storing `neural` as float32 and `input`/`output` as compact stacked arrays, and by deleting `F`, `Fneu` and `dff` as soon as they were no longer needed.
