# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the session inventory in `SESSION_META`, with one tuple per session containing animal, date, source folder, and ALM probe list. For each session it opens `data_structure_<sess>.mat`, constructs the matching `motionEnergy_<sess>.mat` path, and dispatches to separate HDF5 and v5 loaders.

ii.
```python
SESSION_META = [
    ('EKH1', '2021-08-07', 'Ephys_Behavior', [2]),
    ...
    ('JEB24', '2023-11-03', 'RandomizedDelay_Ephys_Behavior', [1]),
]

data_fpath = os.path.join(DATA_DIR, dataset_dir, f'data_structure_{sess_id}.mat')
me_fpath = os.path.join(DATA_DIR, dataset_dir, f'motionEnergy_{sess_id}.mat')
fdata, fmt = load_mat_file(data_fpath)
```

iii. The notes say the 44 sessions should all be included and that both HDF5 v7.3 and MATLAB v5 files must be handled. The trajectory shows the agent extracted probe assignments from the authors' loading scripts rather than globbing blindly.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `anm` field in each `SESSION_META` entry and carried through each processed session record. The final `subjects` list is the sorted unique animal ids, and `subject_idx` maps each session to its animal.

ii.
```python
return {
    'anm': anm,
    'date': date,
    'sess_id': sess_id,
}
...
all_animals = sorted(set(r['anm'] for r in all_results))
subjects = all_animals
subject_idx = np.array([subjects.index(r['anm']) for r in all_results])
```

iii. The notes explicitly say there are 14 animals total and treat the animal name in the filename/loading-script metadata as the subject id.

## 1-c. How are the data split into sessions?

i. Each `SESSION_META` row is treated as one session. The code keeps fixed-delay and randomized-delay sessions in the same unified list and appends one processed result per session into the top-level `neural`, `input`, and `output` lists.

ii.
```python
for i, (anm, date, dataset_dir, alm_probes) in enumerate(sessions_to_process):
    result = process_session(anm, date, dataset_dir, alm_probes,
                            time_edges, kernel, args.show_processing)
    if result is not None:
        all_results.append(result)
```

iii. The notes justify this by saying all 44 sessions from both datasets should be included, with the probe assignments transcribed from the reference loading scripts.

## 1-d. How are the data split into trials?

i. Trials are taken directly from the per-trial behavioral arrays and `Ntrials`. The code builds a boolean `valid_trials_mask` over all trial indices, then uses those indices to slice neural and behavioral outputs. Spike times remain associated with trials through each cluster's `trial` vector.

ii.
```python
ntrials = int(f['obj/bp/Ntrials'][0, 0])
valid_trials_mask = ~early & ~stim_enable
valid_trial_indices = np.where(valid_trials_mask)[0]
...
trial = f[trial_ref][:].flatten().astype(int)  # 1-indexed
...
for tr_idx in valid_trial_indices:
    neural_trials.append(trialdat[:, :, tr_idx].T.copy())
```

iii. The trajectory shows the agent concluded that trials are defined directly by `obj.bp` and that spike data already carry trial numbers, so trial boundaries did not need to be reconstructed.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out early-lick and stimulation trials, and it skips whole sessions if fewer than 2 valid trials remain. It does not implement the reference's extra removal of trials that extend past the end of the ephys recording.

ii.
```python
valid_trials_mask = ~early & ~stim_enable
valid_trial_indices = np.where(valid_trials_mask)[0]

if len(valid_trial_indices) < 2:
    print(f"  {sess_id}: Too few valid trials ({len(valid_trial_indices)})")
    return None
```

iii. The notes explicitly say "Exclude early lick + stim trials. Keep ignore trials." They also mention warnings for sessions with zero neural data later, which is consistent with not dropping post-recording trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from each selected probe's cluster table: `quality`, `trialtm`, and `trial`, plus the behavioral `goCue` times for alignment.

ii.
```python
quality_refs = probe['quality']
trialtm_ref = probe['trialtm'][i, 0]
trial_ref = probe['trial'][i, 0]
trialtm = f[trialtm_ref][:].flatten()
trial = f[trial_ref][:].flatten().astype(int)
goCue = f['obj/bp/ev/goCue'][:].flatten()
```

iii. The notes and trajectory both describe `obj.clu` as the spike source and `obj.bp.ev.goCue` as the alignment event.

## 2-b. How is the `neural` data processed?

i. For each cluster and trial, aligned spike times are histogrammed into 10 ms bins, converted to firing rates by dividing by `DT`, then smoothed with a 15-sample causal Gaussian-like kernel using a custom reflect-padding convolution.

ii.
```python
DT = 1.0 / 100  # 10ms bins
SMOOTH_WIN = 15
...
counts, _ = np.histogram(spk_t, bins=time_edges)
fr = counts.astype(np.float32) / DT
trialdat[:, i, tr_idx] = smooth_with_bc(fr, kernel, BOUNDARY_CONDITION)
```

iii. The notes justify this as matching `WorkingWithDataObjs.m`: 10 ms bins and a 15-sample causal Gaussian. The trajectory shows the agent chose the tutorial's `dt=1/100` instead of the default parameter file's 5 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code excludes clusters only if their quality label exactly matches one of `garbage`, `gabrga`, `noisy`, or `real?`, with case-sensitive matching. It then removes clusters whose mean firing rate across all time bins and all trials is `<= 0.5` Hz, and it drops sessions with fewer than 10 remaining neurons.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 0.5
...
label = ''.join([chr(c) for c in qdata]).strip()
if label in EXCLUDED_QUALITIES:
    continue
...
mean_fr = trialdat.mean(axis=(0, 2))
keep_mask = mean_fr > LOW_FR
```

iii. The notes explicitly defend a case-sensitive MATLAB-style quality filter and a 0.5 Hz threshold from `getDefaultParams.m`, even though they note the paper text mentions 1 Hz.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned by subtracting the go cue time from that spike's own trial. No interpolation or clock correction is applied to the neural data.

ii.
```python
aligned_times = np.empty_like(trialtm)
for t_idx in range(len(trialtm)):
    tr = trial[t_idx] - 1
    if 0 <= tr < ntrials:
        aligned_times[t_idx] = trialtm[t_idx] - goCue[tr]
```

iii. The notes say this matches `alignSpikes.m`, and the trajectory explicitly states that neural alignment is a direct subtraction to go cue onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 10 ms bins from `-2.5` to `2.5` s around go cue, giving 500 bins per trial. No later temporal rebinning is applied.

ii.
```python
DT = 1.0 / 100  # 10ms bins
...
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_centers = time_edges[:-1] + DT / 2
print(f"Time axis: {TMIN} to {TMAX}s, dt={DT*1000:.0f}ms, {n_time} bins")
```

iii. The notes repeatedly justify 10 ms by citing `WorkingWithDataObjs.m`, and the sample/full logs in the notes report 500 time bins.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read from a dedicated raw variable. It is constructed from the analysis time axis defined around go cue using `TMIN`, `TMAX`, and `DT`, with go cue serving as the conceptual alignment event.

ii.
```python
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_centers = time_edges[:-1] + DT / 2
inp = time_centers.astype(np.float32).reshape(1, -1)
```

iii. The notes describe the decoder input as a continuous time axis relative to go cue rather than as a measured raw signal.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code computes the centers of uniformly spaced bins across the fixed window and stores the same 1-by-time vector for every trial.

ii.
```python
time_centers = time_edges[:-1] + DT / 2
...
inp = time_centers.astype(np.float32).reshape(1, -1)
input_trials.append(inp)
```

iii. The notes call this a direct continuous time axis. No additional processing beyond generating bin centers is described.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the same `time_centers` grid that the spike counts use for histogram bins, so its samples correspond to the neural bins' centers.

ii.
```python
counts, _ = np.histogram(spk_t, bins=time_edges)
...
inp = time_centers.astype(np.float32).reshape(1, -1)
```

iii. The notes say the input is "time from goCue" and the code reuses the session-wide bin grid for both neural and input streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the trial outcome flags `hit` and `miss` together with the instructed side flags `R` and `L`.

ii.
```python
hit = f['obj/bp/hit'][:].flatten().astype(bool)
miss = f['obj/bp/miss'][:].flatten().astype(bool)
R = f['obj/bp/R'][:].flatten().astype(bool)
L = f['obj/bp/L'][:].flatten().astype(bool)
```

iii. The trajectory explicitly says the agent discovered that `R/L` encode the correct side, not the actual lick, so actual lick direction had to be derived from `hit/miss` plus `R/L`.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The code maps hit-right to right, hit-left to left, miss-right to left, miss-left to right, and leaves all other trials as `none`/ignore.

ii.
```python
lick_dir = np.full(ntrials, 2, dtype=int)
lick_dir[hit & R] = 1
lick_dir[hit & L] = 0
lick_dir[miss & R] = 0
lick_dir[miss & L] = 1
```

iii. The notes say this was a deliberate fix after initially using `R/L` directly; the trajectory records the same correction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived from `autowater`, which the AI interprets as the WC-vs-DR context flag.

ii.
```python
autowater = f['obj/bp/autowater'][:].flatten().astype(bool)
```

iii. The notes explicitly state `autowater=1 -> WC` and `autowater=0 -> DR`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code relabels `autowater=True` as `WC` (`0`) and all other trials as `DR` (`1`).

ii.
```python
context = np.zeros(ntrials, dtype=int)
context[~autowater] = 1
context[autowater] = 0
```

iii. The notes call this a direct mapping from the reference task structure.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the `hit` and `miss` trial flags; trials that are neither are treated as ignore/no-response.

ii.
```python
hit = f['obj/bp/hit'][:].flatten().astype(bool)
miss = f['obj/bp/miss'][:].flatten().astype(bool)
no_resp = f['obj/bp/no'][:].flatten().astype(bool)
```

iii. The notes define outcome as `incorrect/correct/ignore`. Although `no_resp` is loaded, the actual decision is still driven by `hit` and `miss`, with the remainder treated as ignore.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hits are labeled `1` (correct), misses `0` (incorrect), and all remaining trials `2` (ignore).

ii.
```python
outcome = np.full(ntrials, 2, dtype=int)
outcome[hit] = 1
outcome[miss] = 0
```

iii. The notes and trajectory both describe this as the requested three-way outcome coding.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The final script derives tongue velocity only from the side-camera DeepLabCut feature named `tongue`, using that trial's `ts` coordinates, `frameTimes`, per-frame confidence, `goCue`, and the session video offset.

ii.
```python
side_ref = traj[0, 0]
side_cam = f[side_ref]
...
if name == 'tongue':
    tongue_idx = j
...
tongue_x = ts[tongue_idx, 0, :]
tongue_y = ts[tongue_idx, 1, :]
tongue_conf = ts[tongue_idx, 2, :]
```

iii. The notes broadly describe "DLC velocity from x,y coordinates," but the final code does not implement the notes' implied two-camera tongue combination.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial, the code computes frame-to-frame velocity magnitude from raw x/y differences divided by a single median frame interval, marks frames below confidence 0.9 as NaN, interpolates the surviving values onto the session `time_centers`, and later discretizes them at the session median. It does not smooth x/y first, split contiguous valid runs, bin by averaging frames, or combine the two tongue views.

ii.
```python
dt_frames = np.median(np.diff(aligned_ft))
if dt_frames > 0:
    vel = compute_velocity_from_xy(tongue_x, tongue_y, aligned_ft, dt_frames)
    vel[tongue_conf < 0.9] = np.nan
    valid = ~np.isnan(vel)
    if valid.sum() > 2:
        tongue_vel[:, tr_idx] = np.interp(time_centers, aligned_ft[valid], vel[valid],
                                          left=np.nan, right=np.nan)
```

iii. The notes justify confidence-thresholding and a per-session median split, but they do not justify the omission of the second tongue camera or the change from run-wise smoothing/binning to direct interpolation.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A single per-session threshold is computed as the median of all non-NaN tongue velocity values from valid trials. Each time bin is then labeled `0` below threshold, `1` at or above threshold, and `2` when the tongue velocity is NaN/not visible.

ii.
```python
valid_data = vel_all[:, valid_trial_indices]
flat_valid = valid_data[~np.isnan(valid_data)]
threshold = np.median(flat_valid)
...
disc[valid_mask & (trial_vel < threshold)] = 0
disc[valid_mask & (trial_vel >= threshold)] = 1
```

iii. The notes explicitly say the movement outputs are discretized at the 50th percentile per session with a separate not-visible class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are shifted by a session-level video offset estimated from the median bitcode timing difference, then by the trial's go cue time, and the resulting trace is interpolated onto the same `time_centers` grid used for neural bins.

ii.
```python
bitStart = np.nanmedian(f['obj/bp/ev/bitStart'][:].flatten())
sglx_bitstart = np.nanmedian(f['obj/sglx/bitcode/bitstart'][:].flatten())
sglx_fs = f['obj/sglx/fs'][0, 0]
vidshift = sglx_bitstart / sglx_fs - bitStart
...
aligned_ft = frame_times - vidshift - goCue[tr_idx]
tongue_vel[:, tr_idx] = np.interp(time_centers, aligned_ft[valid], vel[valid],
                                  left=np.nan, right=np.nan)
```

iii. The notes say video offset correction is required and cite the bitcode alignment logic, but they do not explain the switch from mode-based offset and binning to median-based offset and interpolation.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-camera DeepLabCut paw features whose names contain `paw`, along with bottom-camera `frameTimes`, per-frame confidence, `goCue`, and the session video offset.

ii.
```python
bottom_ref = traj[1, 0]
bottom_cam = f[bottom_ref]
...
for j, name in enumerate(bottom_feats):
    if 'paw' in name.lower():
        paw_indices.append(j)
```

iii. The notes only say "traj tongue/output[3]" and "traj paw/output[4]." The final script chooses all paw features rather than just `top_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each bottom-camera paw feature, the code computes frame-to-frame speed from unsmoothed x/y differences, masks low-confidence frames, averages the resulting paw traces together, interpolates that average to `time_centers`, and discretizes at the session median.

ii.
```python
paw_vels = []
for pidx in paw_indices:
    px = ts[pidx, 0, :]
    py = ts[pidx, 1, :]
    pc = ts[pidx, 2, :]
    vel = compute_velocity_from_xy(px, py, aligned_ft, dt_frames)
    vel[pc < 0.9] = np.nan
    paw_vels.append(vel)

avg_vel = np.nanmean(paw_vels, axis=0)
paw_vel[:, tr_idx] = np.interp(time_centers, aligned_ft[valid], avg_vel[valid],
                               left=np.nan, right=np.nan)
```

iii. The notes justify generic DLC-derived velocity with confidence thresholding and a 50th-percentile split, but they do not justify averaging `top_paw` and `bottom_paw`.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity uses the same `discretize_velocity` helper as tongue velocity: a per-session median split over all non-NaN valid-trial samples, with `2` reserved for NaN/not-visible bins.

ii.
```python
paw_disc = discretize_velocity(paw_vel_all, valid_trial_indices, not_visible_val=2)
...
disc[valid_mask & (trial_vel < threshold)] = 0
disc[valid_mask & (trial_vel >= threshold)] = 1
```

iii. The notes explicitly prescribe a per-session 50th-percentile threshold with a not-visible class.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. As with tongue velocity, paw frame times are corrected by the session video offset and the trial's go cue, then interpolated to the neural `time_centers` grid.

ii.
```python
aligned_ft = frame_times - vidshift - goCue[tr_idx]
...
paw_vel[:, tr_idx] = np.interp(time_centers, aligned_ft[valid], avg_vel[valid],
                               left=np.nan, right=np.nan)
```

iii. The notes say video and neural streams are aligned with the same go-cue-centered time axis after offset correction.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from the separate `motionEnergy_<sess>.mat` file. The code reads either `me.data` from a struct or treats `me` itself as the per-trial object array, then pairs each trial's motion-energy trace with side-camera frame times and go cue for alignment.

ii.
```python
me_data = sio.loadmat(me_fpath, squeeze_me=False)
me_raw = me_data['me']
if hasattr(me_raw, 'dtype') and me_raw.dtype.names and 'data' in me_raw.dtype.names:
    me_trials = me_raw['data'][0, 0]
elif me_raw.dtype == object:
    me_trials = me_raw
```

iii. The notes say motion energy comes from separate files and that the implementation had to support both struct-wrapped and direct-array formats.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code interpolates each per-frame motion-energy trace to `time_centers`, then fills internal NaNs by nearest-neighbor interpolation before applying the same session-median discretization helper used for tongue and paw velocity.

ii.
```python
aligned_ft = frame_times - vidshift - goCue[tr_idx]
valid = ~np.isnan(me_trial) & ~np.isnan(aligned_ft)
if valid.sum() > 2:
    me_all[:, tr_idx] = np.interp(time_centers, aligned_ft[valid], me_trial[valid])
...
col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), col[~nans])
```

iii. The trajectory shows the agent copied the idea that the original MATLAB loader interpolates motion energy to the analysis axis and fills edge NaNs; the notes also mention motion energy loading/alignment as a special case.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is thresholded with the same session-level median split as the other movement outputs. Non-NaN bins below median are `0`, at or above median are `1`, and remaining NaN bins are assigned `2`.

ii.
```python
me_disc = discretize_velocity(me_all, valid_trial_indices, not_visible_val=2)
...
output_values = [
    ...
    ['below_median', 'above_median', 'no_video'],
]
```

iii. The notes explicitly say motion energy should be discretized with a per-session 50th-percentile threshold. The output label names that third class `no_video`.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion-energy samples are aligned using side-camera frame times, a session-level video offset, and per-trial go cue times, then interpolated onto the same `time_centers` grid as neural activity.

ii.
```python
ft_ref = side_cam['frameTimes'][tr_idx, 0]
frame_times = f[ft_ref][:].flatten()
aligned_ft = frame_times - vidshift - goCue[tr_idx]
me_all[:, tr_idx] = np.interp(time_centers, aligned_ft[valid], me_trial[valid])
```

iii. The notes treat motion energy as a video-derived stream that should share the common go-cue-centered timeline after offset correction.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code mostly handles irregularities by falling back or silently skipping. It wraps many trial-level video operations in `try/except: continue`, falls back to a default 0.5 s video offset if bitcode-based alignment fails, marks NaN tongue/paw samples as not visible during discretization, and fills missing motion-energy bins by nearest-neighbor interpolation when at least some values are present.

ii.
```python
except:
    vidshift = VIDEO_OFFSET_DEFAULT
...
except Exception as e:
    continue
...
if nans.any() and not nans.all():
    col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), col[~nans])
```

iii. The trajectory mentions a hard-coded 0.5 s offset as a known reference behavior and explicitly notes that the original MATLAB motion-energy loader fills missing bins with the nearest value.

## 11-a. What are the most time-consuming steps of the code?

i. The most expensive processing in the shipped script is the nested spike loop over neurons and trials, followed by the per-trial DLC and motion-energy interpolation loops. The code bins and smooths each neuron one trial at a time instead of counting all trials at once.

ii.
```python
for i in range(n_neurons_raw):
    ...
    for tr_idx in range(ntrials):
        ...
        counts, _ = np.histogram(spk_t, bins=time_edges)
        fr = counts.astype(np.float32) / DT
        trialdat[:, i, tr_idx] = smooth_with_bc(fr, kernel, BOUNDARY_CONDITION)
```

iii. The notes do not explicitly rank runtime bottlenecks, but they report a full run time of about 132 s. The code structure itself makes the neuron-by-trial loop the clearest hot path.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could have been vectorized: the per-spike loop that subtracts `goCue`, the per-neuron/per-trial spike histogramming loop, and the repeated per-trial interpolation loops for tongue, paw, and motion energy. The final script duplicates these patterns separately for HDF5 and v5 files.

ii.
```python
for t_idx in range(len(trialtm)):
    tr = trial[t_idx] - 1
    ...
for i in range(n_neurons_raw):
    ...
    for tr_idx in range(ntrials):
        ...
for tr_idx in range(ntrials):
    ...
    tongue_vel[:, tr_idx] = np.interp(...)
```

iii. There is no explicit written justification for leaving these loops scalar. The code appears to favor straightforwardness over matching the more vectorized reference approach.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats a few computations unnecessarily: it re-creates the same input time vector for every trial, computes the video offset separately inside each DLC loader and motion-energy loader, and maintains nearly duplicated HDF5/v5 session-processing implementations.

ii.
```python
inp = time_centers.astype(np.float32).reshape(1, -1)
input_trials.append(inp)
...
vidshift = sglx_bitstart / sglx_fs - bitStart
...
def process_session_h5(...):
...
def process_session_v5(...):
```

iii. The notes do not defend these repetitions. They are implementation artifacts of keeping HDF5 and v5 code paths separate.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script performs some work whose products are thrown away: it computes `out_per_trial` and `out_time_varying` in `process_session_h5` but never uses them, loads fields like `no_resp`, `has_video`, and `has_me` without using them downstream, and reads all paw tracks only to collapse them immediately into a median-thresholded category output.

ii.
```python
out_per_trial = np.array([lick_dir[tr_idx], context[tr_idx], outcome[tr_idx]], dtype=np.float32)
out_time_varying = np.stack([
    tongue_disc[i],
    paw_disc[i],
    me_disc[i]
], axis=0).astype(np.float32)
...
tongue_vel_all, paw_vel_all, has_video = load_dlc_velocities_h5(...)
me_all, has_me = load_motion_energy(...)
```

iii. No explicit justification for these discarded intermediates appears in the notes or trajectory. They look like leftovers from an earlier version of the implementation.
