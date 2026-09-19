# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-coded a 25-session `SESSION_META` list and loaded only `data/Ephys_Behavior`, not both ephys folders. Each session is opened as an HDF5 `.mat` file with `h5py.File`, while motion energy is loaded separately from a sibling `.mat` file with `scipy.io.loadmat`. It did not implement the reference loader that supports both MATLAB v7.3 and v5 for the main data structure.

ii. ```python
DATA_DIR = 'data/Ephys_Behavior'

SESSION_META = [
    ('EKH1', '2021-08-07', [1]),
    ('EKH3', '2021-08-11', [1, 2]),
    ...
    ('JGR3', '2021-11-18', [1]),
]

session_id = f'{animal}_{date}'
data_file = os.path.join(DATA_DIR, f'data_structure_{session_id}.mat')
me_file = os.path.join(DATA_DIR, f'motionEnergy_{session_id}.mat')
...
f = h5py.File(data_file, 'r')
```

```python
import scipy.io as sio
me_raw = sio.loadmat(me_file)
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly chose `Ephys_Behavior` only because it believed that directory was the DR+WC dataset and sufficient for the context output. The trajectory also shows it decided to ignore `RandomizedDelay_Ephys_Behavior` after counting 25 fixed-delay sessions and focusing on the two-context task.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken from the `animal` element of each hard-coded session tuple, then deduplicated and sorted when assembling the output. `subject_idx` is the index of each session's animal within that sorted unique subject list.

ii. ```python
result = {
    ...
    'session_id': session_id,
    'animal': animal,
}
```

```python
subjects = sorted(list(set(r['animal'] for r in all_results)))
...
subj_idx = subjects.index(result['animal'])
subject_idx_list.append(subj_idx)
```

iii. The notes describe the dataset as 25 sessions from 10 subjects and consistently use the animal name embedded in `SESSION_META` as the subject identifier.

## 1-c. How are the data split into sessions?

i. One session is one `(animal, date, probes)` tuple in `SESSION_META`, corresponding to one `data_structure_<animal>_<date>.mat` file in `data/Ephys_Behavior`. Each processed session becomes one entry in `neural`, `input`, `output`, `subject_idx`, and `brain_region_idx`.

ii. ```python
SESSION_META = [
    ('EKH1', '2021-08-07', [1]),
    ...
]

for idx, (animal, date, probes) in enumerate(sessions):
    result = process_session(animal, date, probes, ...)
    if result is not None:
        all_results.append(result)
```

```python
neural_list.append(result['neural_trials'])
input_list.append(session_inputs)
output_list.append(session_outputs)
```

iii. The trajectory shows the AI transcribed probe assignments from the MATLAB loader scripts, but only for the 25 `Ephys_Behavior` sessions, and treated those as the full session set for conversion.

## 1-d. How are the data split into trials?

i. Trials are indexed by the per-trial Bpod arrays of length `Ntrials`. The AI creates `valid_trials` as integer trial indices into those arrays and then packages one neural matrix, one input array, and one output array per kept trial.

ii. ```python
bp = f['obj']['bp']
Ntrials = int(bp['Ntrials'][0, 0])
...
valid_trials = np.where(valid_mask)[0]  # 0-indexed
```

```python
for trial_idx in valid_trials:
    neural_trials.append(trialdat[:, :, trial_idx].T.astype(np.float32))
    ...
    session_inputs.append(time_input)
    session_outputs.append(output_array)
```

iii. The notes frame the task as using Bpod trial variables plus spike/video data aligned per trial, and the code follows that structure directly.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered with `valid_mask = (early == 0) & (no == 0) & (stim_enable == 0) & ((hit == 1) | (miss == 1))`. That excludes early-lick trials, ignore/no-response trials, and stimulation trials, and keeps only hit or miss trials. The AI does not drop post-recording trials by checking for the last trial containing spikes.

ii. ```python
valid_mask = (early == 0) & (no == 0) & (stim_enable == 0) & ((hit == 1) | (miss == 1))
valid_trials = np.where(valid_mask)[0]

if len(valid_trials) < 2:
    print(f'    Skipping {session_id}: only {len(valid_trials)} valid trials')
    ...
```

iii. The notes explicitly say: 'Include hit and miss trials' and 'Exclude early, no, stim trials.' The trajectory shows the AI justified this from the paper's omission of early and ignore trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the cluster-level `trialtm` and `trial` arrays in `obj.clu`, plus `bp.ev.goCue` for alignment. The code also reads `tm` for absolute spike times, but never uses it in the final computation.

ii. ```python
tm_ref = clu_group['trialtm'][clu_idx, 0]
trial_ref = clu_group['trial'][clu_idx, 0]
...
spike_trialtm = f[tm_ref][:].flatten()
spike_trial = f[trial_ref][:].flatten().astype(int)
...
goCue = bp['ev']['goCue'][:].flatten()
```

```python
tm_abs_ref = clu_group['tm'][clu_idx, 0]
spike_tm_abs = f[tm_abs_ref][:].flatten()
```

iii. The notes summarize the neural mapping as 'Spike times (clu) -> neural' with alignment to go cue and spike-rate conversion.

## 2-b. How is the `neural` data processed?

i. For each kept cluster and each trial, the AI subtracts the trial's go cue from `trialtm`, histograms spikes into 10 ms bins from -2.5 to 2.5 s, converts counts to spikes/s, and applies a causal Gaussian smoothing kernel with a 15-sample window. The result is stored as `(time, neurons, trials)` and then transposed per trial to `(neurons, time)`.

ii. ```python
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
...
spike_goCue = goCue[spike_trial_valid - 1]
spike_aligned = spike_trialtm_valid - spike_goCue
...
counts, _ = np.histogram(trial_spikes, bins=EDGES)
fr = counts / PARAMS['dt']
fr_smooth = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])
trialdat[:, ci, trial_num - 1] = fr_smooth
```

iii. The notes and trajectory repeatedly say the AI chose 10 ms bins, causal Gaussian smoothing, and firing-rate units because it believed that matched the main analysis scripts and paper.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural QC has three parts in the AI code: cluster-quality filtering, firing-rate filtering, and session-level minimum-unit filtering. Cluster qualities are read as strings and all clusters labeled `garbage`, `gabrga`, `noisy`, or `real?` are excluded; then units with mean firing rate `<= 1 Hz` are dropped; then sessions with fewer than 10 remaining neurons are skipped entirely.

ii. ```python
'quality_exclude': ['garbage', 'gabrga', 'noisy', 'real?'],
'lowFR': 1.0,
```

```python
cluid = find_clusters(qualities, PARAMS['quality_exclude'])
...
meanFRs = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
fr_mask = meanFRs > PARAMS['lowFR']
...
if n_neurons < 10:
    print(f'    Skipping {session_id}: only {n_neurons} neurons (need >= 10)')
    ...
```

iii. The notes list all three choices explicitly: `lowFR = 1 Hz`, the `findClusters`-style quality filter, and a minimum of 10 units per session taken from the paper/methods summary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to go cue by subtracting `goCue[trial]` from each spike's `trialtm` within its own trial. No additional interpolation or offset correction is applied to the neural data.

ii. ```python
spike_trial_valid = spike_trial[valid_spike_mask]
spike_trialtm_valid = spike_trialtm[valid_spike_mask]
spike_goCue = goCue[spike_trial_valid - 1]
spike_aligned = spike_trialtm_valid - spike_goCue
```

iii. The notes and trajectory both describe this as directly matching `alignSpikes.m` for go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 10 ms bins (`dt = 1/100`) over -2.5 to 2.5 s, yielding 500 time bins. The AI does not apply any further rebinning after histogramming/interpolating onto that grid.

ii. ```python
PARAMS = {
    ...
    'dt': 1/100,
}
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
N_TIMEBINS = len(TIME_AXIS)
```

iii. The notes repeatedly justify `dt = 10ms` as the 'standard across analysis scripts', even though the reference solution uses 5 ms bins.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not read directly from a raw field. The AI constructs it from the global binning parameters (`tmin`, `tmax`, `dt`) that define time relative to go cue, using the same axis onto which spikes and video are aligned.

ii. ```python
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
```

```python
time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
session_inputs.append(time_input)
```

iii. The notes map 'Time axis -> input[0] -> Time from goCue in seconds' and treat it as a constructed decoder input rather than a raw Bpod variable.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computes bin centers from a uniformly spaced 10 ms grid spanning -2.5 to 2.5 s, then repeats that same 1 x T vector for every trial. There is no additional transformation beyond building the bin-center array.

ii. ```python
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
...
time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
```

iii. The trajectory shows the AI chose the 10 ms grid after deciding that `dt = 1/100` was the appropriate standard.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is exactly the neural binning grid: `TIME_AXIS` is used as the per-trial input, and neural spikes are histogrammed into `EDGES` derived from the same parameters. This makes the input and neural arrays share the same per-bin time reference.

ii. ```python
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
...
counts, _ = np.histogram(trial_spikes, bins=EDGES)
...
time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
```

iii. The notes say the input time axis should match the neural processing window and the code does that directly.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction only from the per-trial `R` flag, which indicates instructed right-vs-left trials. It does not use `hit`, `miss`, or `no` to infer actual lick direction or a no-lick class.

ii. ```python
R = bp['R'][:].flatten()  # right trials
...
lick_direction = R.copy()  # 1=right, 0=left
```

iii. In the notes, the AI mapped `R (right trial)` directly to `output[0]` as `left=0, right=1`, and omitted any third class for no lick.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. There is almost no processing: the AI simply copies the binary `R` array and later repeats that value across all time bins for each kept trial. Misses and ignores are not converted into opposite-side or no-lick labels.

ii. ```python
lick_direction = R.copy()  # 1=right, 0=left
...
lick_dir = np.full(N_TIMEBINS, info['lick_direction'], dtype=np.int64)
```

iii. The trajectory and notes show the AI treated the task variable as trial side, not realized lick direction, likely to keep the output binary and decoder-friendly.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the per-trial `autowater` flag in `obj.bp`. The AI interprets `autowater == 1` as WC and `autowater == 0` as DR.

ii. ```python
autowater = bp['autowater'][:].flatten()  # WC trials
...
context = 1 - autowater  # DR=1, WC=0
```

iii. The notes explicitly map `autowater` to the context output and justify it as the WC-vs-DR indicator.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI applies a direct binary relabeling: `context = 1 - autowater`, so autowater trials become WC (0) and all others become DR (1). It then repeats that per-trial value across time bins.

ii. ```python
context = 1 - autowater  # DR=1, WC=0 (autowater=1 means WC)
...
context = np.full(N_TIMEBINS, info['context'], dtype=np.int64)
```

iii. This follows the mapping written in the notes: `WC=0, DR=1`, per trial.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived only from the `hit` flag after ignore trials have already been filtered out. The AI does not use `miss` except implicitly as the complement of hit among the kept trials.

ii. ```python
hit = bp['hit'][:].flatten()  # correct trials
miss = bp['miss'][:].flatten()  # error trials
no = bp['no'][:].flatten()  # ignore/no-response trials
...
outcome = hit.copy()  # 1=correct, 0=incorrect
```

iii. The notes map `hit` to outcome and justify trial filtering by excluding `no` trials up front, which removes the need for an explicit ignore class in their formulation.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI converts outcome to a binary correct-vs-incorrect label by copying `hit` and keeping only hit/miss trials. It then repeats that label across all time bins of the trial.

ii. ```python
valid_mask = (early == 0) & (no == 0) & (stim_enable == 0) & ((hit == 1) | (miss == 1))
...
outcome = hit.copy()  # 1=correct, 0=incorrect
...
outcome = np.full(N_TIMEBINS, info['outcome'], dtype=np.int64)
```

iii. The trajectory shows the AI intentionally excluded ignore trials because the paper omitted them from most analyses, so outcome became binary in its converted dataset.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `obj.traj` view 1 only, using the feature named `tongue`, plus that trial's `frameTimes`, `goCue`, and a session-wide video offset from `bp.ev.bitStart` and `sglx.bitcode.bitstart`. It does not use the bottom-camera tongue tracking.

ii. ```python
traj_ref = f['obj']['traj'][0, 0]  # view 1 (side)
...
for i, name in enumerate(feat_names):
    if name == 'tongue':
        tongue_idx = i
...
aligned_times = frameTimes - vidshift - goCue[trix]
```

iii. The notes explicitly say 'Tongue velocity: Side view (view 1), tongue feature,' and the trajectory shows the AI settled on the side camera after inspecting DLC feature names.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial, the AI interpolates side-view tongue x/y positions from frame times onto the 10 ms neural time axis, takes `np.gradient` of the interpolated x and y traces, sets NaN gradients to zero, and computes speed as `sqrt(xvel**2 + yvel**2)`. It does not threshold by DLC likelihood, smooth tracked positions, split into contiguous valid runs, or combine both camera views.

ii. ```python
fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
fy = interp1d(aligned_times[valid], y[valid], bounds_error=False, fill_value=np.nan)
xpos[:, trix] = fx(taxis)
ypos[:, trix] = fy(taxis)
...
xv = np.gradient(xpos[:, trix])
yv = np.gradient(ypos[:, trix])
xv[np.isnan(xv)] = 0
yv[np.isnan(yv)] = 0
...
speed = np.sqrt(xvel**2 + yvel**2)
```

iii. The notes claim this 'matches findVelocity.m', and the trajectory shows the AI focused on the gradient-of-position idea, but it simplified away the reference's visibility and two-view handling.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Tongue velocity is discretized with the generic `discretize_continuous` function. The AI pools all per-session values, computes the 50th percentile threshold, and if that threshold is exactly zero it recomputes the threshold from positive values only. Bins are then coded as `1` if `v >= threshold` and `0` otherwise; there is no `not visible` class.

ii. ```python
def discretize_continuous(values_per_trial, threshold_percentile=50):
    all_values = np.concatenate([v.flatten() for v in values_per_trial])
    all_values = all_values[~np.isnan(all_values)]
    threshold = np.percentile(all_values, threshold_percentile)
    if threshold == 0:
        pos_values = all_values[all_values > 0]
        if len(pos_values) > 0:
            threshold = np.percentile(pos_values, threshold_percentile)
    ...
    disc = (v >= threshold).astype(np.float32)
```

```python
tongue_disc = discretize_continuous(result['tongue_vel'])
```

iii. The trajectory documents this as an explicit edge-case fix after the AI discovered that the median of mostly-zero tongue velocities produced all-ones outputs.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue tracking is aligned by subtracting a session-wide video offset and the trial's go cue from frame times, then interpolating x/y positions directly onto the neural `TIME_AXIS`. The final speed trace therefore has one value per neural time bin.

ii. ```python
vidshift = find_video_offset(f)
...
aligned_times = frameTimes - vidshift - goCue[trix]
...
fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
fy = interp1d(aligned_times[valid], y[valid], bounds_error=False, fill_value=np.nan)
xpos[:, trix] = fx(taxis)
ypos[:, trix] = fy(taxis)
```

iii. The notes state there is 'no time shift between neural and movement data' beyond the video offset, and the code implements alignment by interpolation to the same bin centers used for neural activity.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from `obj.traj` view 2 only, using every feature whose name contains `'paw'` (typically both `top_paw` and `bottom_paw`), along with frame times, go cue, and the video offset. It does not restrict itself to `top_paw`.

ii. ```python
traj_ref = f['obj']['traj'][1, 0]  # view 2 (top)
...
paw_indices = []
for i, name in enumerate(feat_names):
    if 'paw' in name.lower():
        paw_indices.append(i)
```

iii. The notes explicitly say 'top_paw + bottom_paw averaged,' and the trajectory shows the AI chose to use all paw-like tracked features from the top view.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature and trial, the AI interpolates x/y positions to the 10 ms neural grid, fills missing positions by nearest interpolation, computes gradients, subtracts a per-trial baseline derivative, fills missing velocities by interpolation, converts x/y velocity to speed, and averages the resulting speeds across paw features. It does not threshold by DLC likelihood and does not preserve missingness as a separate state.

ii. ```python
fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
fy = interp1d(aligned_times[valid], y[valid], bounds_error=False, fill_value=np.nan)
xp = fx(taxis)
yp = fy(taxis)
...
xp = np.interp(indices, indices[mask], xp[mask])
...
xv = np.gradient(xpos[:, trix])
yv = np.gradient(ypos[:, trix])
basederiv_x = np.nanmedian(np.diff(xpos[:, trix]))
basederiv_y = np.nanmedian(np.diff(ypos[:, trix]))
...
speed = np.sqrt(xvel**2 + yvel**2)
avg_speed = np.mean(all_speeds, axis=0)
```

iii. The notes say the AI was trying to match `findVelocity.m`, but its actual implementation adds nearest-filling, baseline subtraction, and averaging across two paw features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity uses the same `discretize_continuous` helper as tongue velocity: one per-session threshold at the 50th percentile, with a positive-only fallback if the threshold is zero, and binary labels `0/1` only. There is no `not visible` class.

ii. ```python
paw_disc = discretize_continuous(result['paw_vel'])
...
disc = (v >= threshold).astype(np.float32)
```

iii. The AI reused a single discretizer for all three continuous outputs and documented the zero-threshold fix as a generic edge-case rule.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. As with tongue velocity, the AI subtracts a session-wide video offset and the trial's go cue from paw frame times, then interpolates paw positions to the same 10 ms `TIME_AXIS` used for neural data. Velocity is then computed on that aligned grid.

ii. ```python
aligned_times = frameTimes - vidshift - goCue[trix]
...
xp = fx(taxis)
yp = fy(taxis)
```

iii. The notes and trajectory present all movement variables as aligned to the neural time axis through the video offset plus go-cue subtraction.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<session>.mat` file, specifically `me['data']` (with an extra nested-struct unwrap when needed), plus side-camera `frameTimes`, go cue, and the session-wide video offset. The AI does not use `obj.me`.

ii. ```python
me_raw = sio.loadmat(me_file)
me_struct = me_raw['me'][0, 0]
me_data = me_struct['data']
if me_data.dtype.names is not None and 'data' in me_data.dtype.names:
    me_data = me_data[0, 0]['data']
...
traj_ref = f['obj']['traj'][0, 0]
...
aligned_times = frameTimes - vidshift - goCue[trix]
```

iii. The trajectory shows the AI discovered and fixed the nested `me.data.data` case after comparing against the MATLAB reference comment in `loadMotionEnergy.m`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI loads one motion-energy trace per trial, aligns side-camera frame times to go cue, linearly interpolates the trace onto the 10 ms neural grid, and nearest-fills NaNs where interpolation produced gaps. It does not simply average existing frame values into bins.

ii. ```python
f_interp = interp1d(aligned_times[valid], me_trial[valid],
                   bounds_error=False, fill_value=np.nan)
me_interp = f_interp(taxis)
...
me_interp = np.interp(indices, indices[mask], me_interp[mask])
me_aligned[:, trix] = me_interp
```

iii. The notes say motion energy is 'interpolated to neural time axis using video frameTimes,' which is the guiding justification the AI used throughout.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized with the same per-session 50th-percentile rule as tongue and paw, including the zero-threshold fallback to positive-only values. The output is binary `0/1`; there is no separate `no video` category.

ii. ```python
me_disc = discretize_continuous(result['motion_energy'])
...
if threshold == 0:
    pos_values = all_values[all_values > 0]
    if len(pos_values) > 0:
        threshold = np.percentile(pos_values, threshold_percentile)
```

iii. The trajectory shows this logic was introduced to avoid degenerate all-high outputs when the pooled median was zero.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by subtracting the same session-wide video offset and the trial go cue from side-camera frame times, then interpolating onto the neural `TIME_AXIS`. The aligned trace is therefore one motion-energy value per neural time bin.

ii. ```python
vidshift = find_video_offset(f)
...
aligned_times = frameTimes - vidshift - goCue[trix]
...
me_interp = f_interp(taxis)
```

iii. The AI treated all video-derived outputs as streams that should be resampled to the neural grid rather than binned from raw frames.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or awkward data mostly with fallbacks and silent continuation. If `frameTimes` cannot be read, it synthesizes them as `np.arange(...)/400`; if frame times or dropped-frame metadata are NaN, the trial's positions often remain NaN and then become zeros (tongue) or are nearest-filled (paw, motion energy); if video offset calculation fails, it uses `0.0`; if a modality is `None`, it fills the whole output with zeros for that trial. Many exceptions are swallowed with bare `except`/`continue`.

ii. ```python
except Exception as e:
    print(f'    Warning: Could not compute video offset: {e}')
    return 0.0
```

```python
except:
    n_frames = ts_data.shape[-1] if ts_data.ndim == 3 else ts_data.shape[0]
    frameTimes = np.arange(1, n_frames + 1) / 400.0
```

```python
xv[np.isnan(xv)] = 0
...
xp = np.interp(indices, indices[mask], xp[mask])
...
me_interp = np.interp(indices, indices[mask], me_interp[mask])
```

```python
if tongue_vel is not None:
    tongue_vel_trials.append(tongue_vel[:, trial_idx].astype(np.float32))
else:
    tongue_vel_trials.append(np.zeros(N_TIMEBINS, dtype=np.float32))
```

iii. The trajectory shows the AI favored keeping sessions/trials usable for the decoder and patching edge cases such as zero medians and nested motion-energy structs, rather than preserving missingness as explicit classes.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming work in the AI code is the explicit nested looping: spike binning and smoothing for every cluster and every trial, plus trial-by-trial interpolation and velocity computation for tongue, paw, and motion energy. The code also repeatedly reads HDF5 objects inside those loops.

ii. ```python
for ci, clu_idx in enumerate(cluid):
    ...
    for trial_num in range(1, Ntrials + 1):
        ...
        counts, _ = np.histogram(trial_spikes, bins=EDGES)
        fr_smooth = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])
```

```python
for trix in range(Ntrials):
    ...
    xpos[:, trix] = fx(taxis)
    ypos[:, trix] = fy(taxis)
```

iii. The notes estimate roughly 6-7 seconds per session and the implementation makes the main computational hotspots visible: the neural cluster/trial loops and the per-trial video resampling loops.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest avoidable loops are the per-trial neural histogram loop inside each cluster, the repeated convolution over each single-trial firing-rate vector, and the per-trial interpolation loops for tongue, paw, and motion energy. The code could also avoid repeatedly scanning feature names and HDF5 references inside session-level loops.

ii. ```python
for ci, clu_idx in enumerate(cluid):
    ...
    for trial_num in range(1, Ntrials + 1):
        trial_mask = spike_trial_valid == trial_num
        ...
        counts, _ = np.histogram(trial_spikes, bins=EDGES)
```

```python
for trix in range(Ntrials):
    ...
    fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
    fy = interp1d(aligned_times[valid], y[valid], bounds_error=False, fill_value=np.nan)
```

```python
for j in range(x_padded.shape[1]):
    result[:, j] = convolve(x_padded[:, j], kern, mode='same')
```

iii. The trajectory mentions efficiency goals, but the final code still uses straightforward scalar/trial loops instead of the vectorized spike counting and binning strategy seen in the reference solution.

## 11-c. What processing does the code repeat multiple times?

i. The code recomputes the video offset separately inside tongue, paw, and motion-energy processing instead of once per session. It also discretizes the continuous outputs multiple times: once while building the final dataset and again inside `plot_processing`. More broadly, it repeats trial-wise interpolation logic independently in each modality-specific function.

ii. ```python
vidshift = find_video_offset(f)
```

```python
def compute_tongue_velocity(...):
    ...
    vidshift = find_video_offset(f)
```

```python
def compute_paw_velocity(...):
    ...
    vidshift = find_video_offset(f)
```

```python
def load_motion_energy(...):
    ...
    vidshift = find_video_offset(f)
```

```python
tongue_disc = discretize_continuous(result['tongue_vel'])
paw_disc = discretize_continuous(result['paw_vel'])
me_disc = discretize_continuous(result['motion_energy'])
```

```python
tongue_disc = discretize_continuous(tongue)
paw_disc = discretize_continuous(paw)
me_disc = discretize_continuous(me)
```

iii. The AI prioritized getting each modality working independently; the resulting code repeats shared alignment and discretization work rather than centralizing it.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads absolute spike times `tm` for every cluster but never uses them. It also accumulates `all_cluid` but never consumes it, reads `L` despite not using it, and computes/retains raw continuous movement traces only to discard them after discretization in the saved dataset. In `--show-processing` mode it additionally recomputes discretizations purely for plots.

ii. ```python
L = bp['L'][:].flatten()  # left trials
```

```python
tm_abs_ref = clu_group['tm'][clu_idx, 0]
spike_tm_abs = f[tm_abs_ref][:].flatten()
```

```python
all_cluid = []
...
all_cluid.append(cluid)
```

```python
tongue_vel = compute_tongue_velocity(f, goCue, Ntrials)
paw_vel = compute_paw_velocity(f, goCue, Ntrials)
motion_energy = load_motion_energy(f, me_file, goCue, Ntrials)
...
tongue_disc = discretize_continuous(result['tongue_vel'])
```

iii. These are by-products of the AI's simpler implementation strategy and its plotting/debugging workflow, not data needed in the final saved format.
