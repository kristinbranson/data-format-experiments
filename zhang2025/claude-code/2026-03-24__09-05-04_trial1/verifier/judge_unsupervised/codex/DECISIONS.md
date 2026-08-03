# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the session list from `bwm_release.csv`, reduces it to one row per `eid`, then iterates session-by-session. For each session it resolves a cache path from `(lab, subject, date, number=001)`, loads the trials parquet, loads every `probe*` under that session, merges probes, and then loads wheel and whisker files from the same `alf` directory. This is a direct filesystem implementation of the reference ONE-cache workflow rather than calling ONE/`SessionLoader` directly.

ii. 
```python
BWM_CSV = Path('/app/code/code_zhang2025/data/bwm_release.csv')

def find_session_path(lab, subject, date, number=1):
    sess_path = DATA_DIR / lab / 'Subjects' / subject / date / f'{number:03d}' / 'alf'

def get_session_list(bwm_df):
    sessions = bwm_df.groupby('eid').first().reset_index()
    return sessions

bwm_df = pd.read_csv(BWM_CSV, index_col=0)
sessions = get_session_list(bwm_df)

for idx in range(len(sessions)):
    sess = sessions.iloc[idx]
    result = process_session(sess, br, show_processing=args.show_processing)
```

iii. In `CONVERSION_NOTES.md`, Step 4 says the agent chose to "Use BWM CSV as ground truth for session list." Step 6 says it intentionally used "Direct file loading (no ONE API needed, works offline from cache)." The trajectory summary also says it wanted to match the reference code while working offline from the mounted cache.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the `subject` column in the per-session metadata. After session processing, the agent builds a sorted unique subject list and a `subject_idx` array that maps each retained session to one entry in that list.

ii.
```python
all_subjects = sorted(set(r['subject'] for r in all_results))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}

for r in all_results:
    subject_idx.append(subject_to_idx[r['subject']])

data = {
    'subjects': all_subjects,
    'subject_idx': np.array(subject_idx),
}
```

iii. The notes repeatedly describe the target format as session-major with separate `subjects` and `subject_idx`, so the agent preserved subject identity at the session level rather than nesting by mouse. This matches the requested output structure in the instructions.

## 1-c. How are the data split into sessions?

i. The session boundary is one unique `eid` from `bwm_release.csv`, processed into one entry in `all_results`. Inside each session, all probes are merged before any trial-level outputs are built, so the converted dataset uses session-level rather than probe-level recordings.

ii.
```python
def get_session_list(bwm_df):
    sessions = bwm_df.groupby('eid').first().reset_index()
    return sessions

probe_dirs = sorted([d.name for d in alf_path.iterdir() if d.is_dir() and d.name.startswith('probe')])

spikes, clusters = merge_probes(spikes_list, clusters_list)

for r in all_results:
    neural_list.append(r['neural'])
    input_list.append(r['input'])
    output_list.append(r['output'])
```

iii. In Step 1 of the notes, the agent recorded that the reference code merges probes across a session. In Step 5 it explicitly chose "Session list: Use BWM release CSV (459 sessions) as ground truth."

## 1-d. How are the data split into trials?

i. Trials are defined from rows of the `_ibl_trials.table.pqt` DataFrame. The agent first applies a trial mask, then defines one aligned interval per surviving row, bins neural/behavioral time series into that interval, and finally stores each remaining trial as one `(neurons, time)` neural array plus one input array and one output array.

ii.
```python
trials = load_trials(alf_path)
mask = create_trial_mask(trials)
valid_trials = trials[mask].reset_index(drop=True)

align_times = valid_trials[ALIGN_TIME].values
interval_starts = align_times + TIME_WINDOW[0]
interval_ends = align_times + TIME_WINDOW[1]

final_trials = valid_trials[combined_mask].reset_index(drop=True)

for t in range(len(final_spikes)):
    neural_trials.append(final_spikes[t].astype(np.uint8))
```

iii. The notes describe `load_trials_and_mask()` and trial-aligned binning as the core reference behavior. The trajectory report on `0_data_caching.py` likewise identifies `trials_df` plus `bin_spiking_data()` and `bin_behaviors()` as the reference trial construction path.

## 1-e. How are trials filtered based on quality controls?

i. There are two layers. First, `create_trial_mask()` removes trials with NaNs in key event columns, reaction times outside `[0.08, 2.0]`, no-choice trials, and overly long trials. Second, `combined_mask` removes any remaining trial that lacks fully interpolated wheel speed or whisker motion energy across the entire 2 s window.

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']

for event in NAN_EXCLUDE:
    if event in trials.columns:
        mask &= ~trials[event].isna()

rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
mask &= (trials['choice'] != 0)

trial_len = trials['feedback_times'] - trials['goCue_times']
mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()

combined_mask &= ~np.any(np.isnan(wheel_speed_binned), axis=1)
combined_mask &= ~np.any(np.isnan(me_binned), axis=1)
```

iii. Step 1 and Step 3 of `CONVERSION_NOTES.md` list the same RT/NaN/no-choice/max-length rules and say they were taken from the reference code/papers. The notes also say the final dataset only keeps trials where all required signals are available so the decoder has matching neural and output data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from raw spike timestamps and spike cluster assignments from every `probe*/pykilosort` directory in the session, plus cluster-to-channel and channel-to-brain-region files for metadata.

ii.
```python
spikes = {
    'times': np.load(rev_dir / 'spikes.times.npy').flatten(),
    'clusters': np.load(rev_dir / 'spikes.clusters.npy').flatten(),
}

clusters_channels = np.load(rev_dir / 'clusters.channels.npy').flatten()
chan_brain_ids = np.load(rev_dir / 'channels.brainLocationIds_ccf_2017.npy').flatten()
```

iii. The Step 5 variable map in the notes says `neural` comes from `spikes.times + spikes.clusters`, binned into 20 ms windows. The reference-code summary in the trajectory also names `load_spiking_data()` and `merge_probes()` as the source path.

## 2-b. How is the `neural` data processed?

i. The agent merges probes, sorts spikes by time, maps each cluster to a Beryl brain region, then bins spike counts into a fixed 2 s window around stimulus onset using 20 ms bins. Per trial, the final neural matrix is stored as integer spike counts cast to `uint8`.

ii.
```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
cluster_regions = get_brain_regions(clusters, br)

binned_spikes = bin_spikes_vectorized(
    spikes['times'], spikes['clusters'], n_clusters,
    interval_starts, interval_ends, BINSIZE, N_BINS
)

for t in range(len(final_spikes)):
    neural_trials.append(final_spikes[t].astype(np.uint8))
```

iii. Step 5 of the notes says "Bin into 20ms windows aligned to stimOn, shape `(n_neurons, 100)`" and Step 6 says the agent wrote a vectorized spike-binning implementation to mirror the reference pipeline. The trajectory report also states that the reference uses spike counts, not rates or dF/F.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent does not apply neuron quality filtering. It loads `clusters.metrics.pqt`, but keeps all clusters from all probes and uses trial-level filtering instead of neuron-level QC.

ii.
```python
metrics_file = rev_dir / 'clusters.metrics.pqt'
if metrics_file.exists():
    metrics = pd.read_parquet(metrics_file)
else:
    metrics = None

n_clusters = len(clusters['channels'])
binned_spikes = bin_spikes_vectorized(..., n_clusters, ...)
```

iii. The notes explicitly say "qc=None: ALL clusters used (not filtered by quality label)" and later repeat "Use all clusters (qc=None): Matches reference code, not just well-isolated neurons." The trajectory step summary says the same.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every trial is aligned to `stimOn_times`. The agent builds `[stimOn - 0.5 s, stimOn + 1.5 s]` intervals and bins spikes inside those intervals.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)

align_times = valid_trials[ALIGN_TIME].values
interval_starts = align_times + TIME_WINDOW[0]
interval_ends = align_times + TIME_WINDOW[1]
```

iii. Step 4 of the notes says the agent resolved the reference/task conflict by following the task specification: "ALL alignment to stimulus onset." The trajectory also notes that it saw the methods-paper difference for dynamic behaviors but decided to keep stimulus-onset alignment in the conversion.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use fixed 20 ms bins and 100 bins per 2 s trial. No secondary rebinning is applied after the initial spike/behavior interpolation/binning.

ii.
```python
BINSIZE = 0.02
TIME_WINDOW = (-0.5, 1.5)
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```

iii. The notes say the reference code uses `binsize=0.02` and that the agent followed 20 ms for all variables. The methods excerpt in the trajectory also states "2-s trials ... 20-ms bins ... T = 100 time steps."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not computed from a raw trial column on a per-trial basis. Instead, the agent derives it from the chosen alignment definition itself: `ALIGN_TIME='stimOn_times'`, `TIME_WINDOW=(-0.5, 1.5)`, and `BINSIZE=0.02`, producing the same relative time vector for every retained trial.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02

time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. In Step 5 the notes say "Time since stimOn ... `np.linspace(-0.5, 1.48, 100)`, same for all trials." The trajectory shows the agent interpreting this as a decoder input defined by the aligned trial axis rather than by an additional raw sensor stream.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The processing is minimal: generate a 100-sample uniformly spaced vector from `-0.48` s to `1.5` s and copy that same vector into every trial's input matrix as row 0.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)

inp = np.array([
    time_since_stim,
    np.full(N_BINS, trial_in_block_final[t], dtype=np.float32),
], dtype=np.float32)
```

iii. The notes state that this input was "manual construction" rather than loaded from the raw files. The justification was that the decoder needed a continuous, time-varying representation on the same bin grid as the neural data.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is perfectly locked to the neural binning grid because it uses the exact same `TIME_WINDOW`, `BINSIZE`, and `N_BINS` as spike binning and is inserted trial-by-trial with the neural arrays.

ii.
```python
binned_spikes = bin_spikes_vectorized(..., interval_starts, interval_ends, BINSIZE, N_BINS)
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. Step 5 of the notes explicitly ties the time input to the same 100-bin stimulus-aligned window as the neural data. The trajectory likewise notes that the agent wanted the input and neural matrices on identical trial axes.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `trials['probabilityLeft']`, using changes in that column to identify block boundaries.

ii.
```python
prob_left_all = trials['probabilityLeft'].values
trial_in_block_all = compute_trial_in_block(prob_left_all)
```

iii. The notes say "Trial number in block: Computed from probabilityLeft changes" and "resets at block boundary." That is the only raw field the agent uses for this input.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The agent scans the full session's `probabilityLeft` sequence, resets a counter whenever the value changes, stores a 1-based trial index within the current block, then applies the same trial mask and final behavior-availability mask used elsewhere. The scalar value is then broadcast across all 100 time bins for that trial.

ii.
```python
def compute_trial_in_block(prob_left):
    trial_in_block = np.zeros(len(prob_left), dtype=np.float32)
    count = 1
    for i in range(len(prob_left)):
        if i > 0 and prob_left[i] != prob_left[i-1]:
            count = 1
        trial_in_block[i] = count
        count += 1
    return trial_in_block

trial_in_block_valid = trial_in_block_all[mask.values]
trial_in_block_final = trial_in_block_valid[combined_mask]
np.full(N_BINS, trial_in_block_final[t], dtype=np.float32)
```

iii. Step 5 of the notes explicitly says the value is a per-trial scalar that "resets at block boundary." The rationale was to satisfy the requested decoder input "Trial number in block" using the observable block schedule in the raw trials table.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. `Choice` is derived directly from `trials['choice']` after trial filtering.

ii.
```python
choice = final_trials['choice'].values.copy()
```

iii. The notes map `trials.choice` to output 0 and describe it as the IBL left/right choice variable. The trajectory's methods excerpt also identifies choice as one of the four decoded behaviors.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent converts IBL coding `-1=left, 1=right` to `0=left, 1=right`, excludes `choice == 0` earlier in trial filtering, and then broadcasts the per-trial class across all 100 bins in the output array.

ii.
```python
mask &= (trials['choice'] != 0)

choice_encoded = ((choice + 1) // 2).astype(int)  # -1->0, 1->1

np.full(N_BINS, choice_encoded[t], dtype=int)
```

iii. The notes say "Choice: left(-1)->0, right(1)->1" and explicitly cite the task's required label mapping as the reason for overriding the reference code's native `-1/1` representation.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The agent derives it from the trial table column `probabilityLeft`, i.e. the task block probability, not from a separate inferred latent prior estimate.

ii.
```python
prob_left = final_trials['probabilityLeft'].values
```

iii. Step 4 of the notes says the agent resolved the ambiguity by following the task spec: "Prior encoded as categorical (3 classes from probabilityLeft)." Step 5 repeats the mapping from `trials.probabilityLeft`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code performs a direct categorical remap: `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`. Like choice, it then broadcasts the trial label across all 100 bins.

ii.
```python
prior_encoded = np.zeros(len(prob_left), dtype=int)
prior_encoded[prob_left == 0.2] = 0
prior_encoded[prob_left == 0.5] = 1
prior_encoded[prob_left == 0.8] = 2

np.full(N_BINS, prior_encoded[t], dtype=int)
```

iii. The notes say this choice was driven by the explicit task requirement "`0.2->0, 0.5->1, 0.8->2`." The trajectory notes that this differs from the methods paper's description of a continuous prior estimate, but the agent chose the task's categorical version.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
wheel_pos_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.position.npy')).flatten()
wheel_ts_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.timestamps.npy')).flatten()
wheel_times, wheel_speed = interpolate_wheel(wheel_ts_raw, wheel_pos_raw)
```

iii. Step 2 of the notes lists the wheel position and timestamp files, and Step 6 says wheel velocity was computed to match ibllib's wheel processing.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The agent linearly interpolates wheel position to 1 kHz, applies an 8th-order 20 Hz Butterworth low-pass filter, differentiates to velocity, takes the absolute value to get speed, and then interpolates that continuous speed signal onto the trial-aligned 20 ms grid.

ii.
```python
t = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
pos_interp = interp1d(timestamps, position, kind='linear')(t)
sos = signal.butter(N=order, Wn=corner_freq / fs * 2, btype='lowpass', output='sos')
vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos_interp)), 0, 0) * fs
return t, np.abs(vel).astype(np.float32)

wheel_speed_binned, wheel_good = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

iii. The notes say "Wheel velocity via Butterworth filter matching ibllib (1kHz interp, 20Hz corner, order 8)." The trajectory's extracted `SessionLoader.load_wheel()` summary reports the same reference defaults.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The agent discretizes wheel speed into three equal-frequency bins using quantiles computed over all wheel-speed time points from all retained trials within a session.

ii.
```python
def discretize_to_bins(values, n_bins=N_DISCRETE_BINS):
    quantiles = np.linspace(0, 100, n_bins + 1)
    boundaries = np.percentile(valid, quantiles)
    result = np.digitize(values, boundaries[1:-1])

all_wheel_flat = final_wheel.flatten()
wheel_disc, wheel_boundaries = discretize_to_bins(all_wheel_flat, N_DISCRETE_BINS)
wheel_disc_2d = wheel_disc.reshape(final_wheel.shape).astype(int)
```

iii. Step 5 of the notes explicitly says "equal-frequency (quantile) bins across all trials within a session, 3 bins." The justification was the task's requirement to discretize an otherwise continuous signal.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. After continuous wheel speed is computed, it is interpolated onto the same stimulus-aligned `[stimOn - 0.5 s, stimOn + 1.5 s]` bins used for neural activity. Trials with incomplete wheel coverage across that entire window are dropped.

ii.
```python
wheel_speed_binned, wheel_good = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, interval_starts, interval_ends, BINSIZE, N_BINS
)

combined_mask &= ~np.any(np.isnan(wheel_speed_binned), axis=1)
```

iii. In Step 4 of the notes, the agent explicitly says it overrode the methods-paper first-movement alignment because "Task spec says stimOn for all." The trajectory contains the same reasoning.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` plus `_ibl_leftCamera.times.npy`, with fallback to the corresponding right-camera files if the left camera is unavailable.

ii.
```python
if side == 'left':
    me_file = find_latest_revision(alf_path, 'leftCamera.ROIMotionEnergy.npy')
    times_file = find_latest_revision(alf_path, '_ibl_leftCamera.times.npy')
else:
    me_file = find_latest_revision(alf_path, 'rightCamera.ROIMotionEnergy.npy')
    times_file = find_latest_revision(alf_path, '_ibl_rightCamera.times.npy')
```

iii. Step 6 of the notes says "Motion energy loading from leftCamera (fallback to rightCamera)." The trajectory's extracted reference summary also says the reference code tries left whisker motion energy first and then right.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The code loads the precomputed ROI motion-energy trace, truncates timestamps and values to their shared minimum length when they differ, and interpolates the resulting continuous signal onto the 20 ms trial bins.

ii.
```python
me = np.load(me_file).flatten()
times = np.load(times_file).flatten()
min_len = min(len(me), len(times))
return times[:min_len], me[:min_len].astype(np.float32)

me_binned, me_good = interpolate_behavior_to_bins(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

iii. The notes mention a "length mismatch (common in IBL data)" and that interpolation follows the same reference-style `interp1d` trial binning used for other continuous behaviors.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Exactly like wheel speed, it is flattened across the retained session and discretized into three quantile bins labeled low/medium/high.

ii.
```python
all_me_flat = final_me.flatten()
me_disc, me_boundaries = discretize_to_bins(all_me_flat, N_DISCRETE_BINS)
me_disc_2d = me_disc.reshape(final_me.shape).astype(int)
```

iii. Step 5 of the notes explicitly says whisker motion energy uses equal-frequency three-bin discretization for the task, parallel to wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The motion-energy trace is interpolated onto the same stimulus-aligned 100-bin grid used for spikes, and trials lacking full coverage of that window are discarded.

ii.
```python
me_binned, me_good = interpolate_behavior_to_bins(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
)

combined_mask &= ~np.any(np.isnan(me_binned), axis=1)
```

iii. As with wheel speed, Step 4 of the notes says the agent intentionally kept stimulus-onset alignment for this signal because the task specification overrode the paper's first-movement alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent uses the latest revision of each file when multiple revisions exist, ignores absent files by skipping that probe/session, truncates whisker motion energy and camera timestamps to their common length when they mismatch, and drops any trial with incomplete interpolated wheel or whisker coverage. If an entire session lacks a required component or yields fewer than two usable trials, the session is skipped.

ii.
```python
def find_latest_revision(base_path, filename_pattern):
    ...
    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]

if me_file is None or times_file is None:
    return None, None

min_len = min(len(me), len(times))
return times[:min_len], me[:min_len].astype(np.float32)

except Exception as e:
    print(f"  Session {eid}: error loading wheel: {e}")

if n_valid < 2:
    print(f"  Session {eid}: fewer than 2 trials with all data, skipping")
    return None
```

iii. The notes explicitly mention handling common motion-energy length mismatches and accepting skipped sessions as reasonable when files are missing. The trajectory summary also states that many sessions were skipped because of path/file availability.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive steps are per-session filesystem IO, loading and merging all probe spike arrays, trial-by-trial spike binning, and trial-by-trial interpolation of wheel and whisker signals. These are the only operations that repeatedly touch large arrays across all sessions and all trials.

ii.
```python
for probe_name in probe_dirs:
    spk, clu = load_spike_sorting(alf_path, probe_name)

binned_spikes = bin_spikes_vectorized(...)

wheel_speed_binned, wheel_good = interpolate_behavior_to_bins(...)
me_binned, me_good = interpolate_behavior_to_bins(...)
```

iii. Step 7 of the notes estimates about 20 s per session and Step 6 says the code's main optimization effort went into spike binning. That implies the heavy work was the repeated per-session loading/binning path rather than final dataset assembly.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still loops in Python over trials in spike binning, over trials in behavior interpolation, over trials while converting arrays to lists, and over trials while building input/output arrays. Those loops could be batched or represented as whole-session arrays for speed and memory efficiency.

ii.
```python
for trial_idx in range(n_trials):
    ...
    np.add.at(binned[trial_idx].ravel(), flat_idx, 1)

for trial_idx in range(n_trials):
    ...
    y_interp = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')(x_interp)

for t in range(len(final_spikes)):
    neural_trials.append(final_spikes[t].astype(np.uint8))

for t in range(len(final_trials)):
    input_trials.append(inp)
    output_trials.append(out)
```

iii. The notes say the agent added a vectorized spike-binning routine, but the actual implementation still iterates trial-by-trial in Python. The trajectory also framed efficiency as an open issue, so these remaining loops are consistent with that.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly scans revision directories for every requested file, repeatedly constructs interpolation objects per trial, and repeatedly broadcasts per-trial scalars (`choice`, `prior`, `trial_in_block`) across 100 bins. It also recomputes session-level discretization separately for wheel and whisker by flattening all retained values.

ii.
```python
find_latest_revision(alf_path, '_ibl_trials.table.pqt')
find_latest_revision(alf_path, '_ibl_wheel.position.npy')
find_latest_revision(alf_path, '_ibl_wheel.timestamps.npy')

for trial_idx in range(n_trials):
    y_interp = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')(x_interp)

np.full(N_BINS, trial_in_block_final[t], dtype=np.float32)
np.full(N_BINS, choice_encoded[t], dtype=int)
np.full(N_BINS, prior_encoded[t], dtype=int)
```

iii. This follows from the implementation itself; the notes do not defend these repetitions beyond saying the code was written for robustness and offline compatibility rather than maximal speed.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `clusters.metrics.pqt` and `clusters.depths.npy` without using them for filtering or outputs, computes `wheel_good`, `me_good`, `wheel_boundaries`, and `me_boundaries` but never uses them downstream, imports `ProcessPoolExecutor`/`as_completed` without using them, and optionally makes plots that are not part of the converted pickle. It also loads all cluster brain-region strings before later remapping them to integer indices.

ii.
```python
from concurrent.futures import ProcessPoolExecutor, as_completed

clusters_depths = np.load(rev_dir / 'clusters.depths.npy').flatten()
metrics = pd.read_parquet(metrics_file)

wheel_speed_binned, wheel_good = interpolate_behavior_to_bins(...)
me_binned, me_good = interpolate_behavior_to_bins(...)

wheel_disc, wheel_boundaries = discretize_to_bins(...)
me_disc, me_boundaries = discretize_to_bins(...)
```

iii. The notes say the agent explored QC metrics and plotting for validation, but the final converted dataset uses neither the cluster QC metrics nor the quantile boundaries. Those computations therefore support debugging/documentation more than downstream decoding.
