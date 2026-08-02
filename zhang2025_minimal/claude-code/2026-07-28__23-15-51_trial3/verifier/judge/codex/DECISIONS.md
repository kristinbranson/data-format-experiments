# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent uses the BWM release table as the session inventory, but then loads data by scanning the local ONE cache on disk rather than using the reference `ONE` loader pipeline. It reconstructs each session path from `lab / subject / date / session_number`, groups rows by `eid`, and then loads session-level trials, per-probe spike files, wheel files, and whisker motion-energy files from the filesystem. Only sessions whose cache directories exist are considered.

ii.
```python
def find_session_paths(cache_dir):
    bwm_df = pd.read_csv('/app/code/code_zhang2025/data/bwm_release.csv', index_col=0)
    ...
    session_path = Path(cache_dir) / lab / 'Subjects' / subject / date / '001'
    ...
    if session_path.exists():
        if eid not in sessions:
            sessions[eid] = {
                'path': session_path,
                'probes': [],
                'subject': subject,
                'lab': lab,
                'date': date,
                'eid': eid,
            }
        sessions[eid]['probes'].append(probe_name)

def convert_data(cache_dir, max_sessions=None, sample=False):
    sessions = find_session_paths(cache_dir)
```

iii. In `CONVERSION_NOTES.md`, the agent says it is using the BWM release and the Zhang et al. reference code. In the trajectory, it explicitly decided to mirror `0_data_caching.py`'s session inventory and then wrote a standalone local-cache loader instead of using `ONE`.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column of the BWM release rows associated with each `eid`. After session processing, the script builds an ordered unique `subject_set`, then records one `subject_idx` per retained session.

ii.
```python
if eid not in sessions:
    sessions[eid] = {
        'path': session_path,
        'probes': [],
        'subject': subject,
        ...
    }

...
subject = result['subject']
if subject not in subject_set:
    subject_set.append(subject)
subj_idx = subject_set.index(subject)
all_subject_idx.append(subj_idx)
```

iii. The trajectory summary states that the output should include subjects and session-to-subject indexing in the target format. No deeper justification was recorded beyond matching the requested output schema.

## 1-c. How are the data split into sessions?

i. Sessions are keyed by unique experiment ID (`eid`). Multiple probe rows in the release table are grouped into one session object, and later all probes for that `eid` are merged into a single session-wide neural population.

ii.
```python
if eid not in sessions:
    sessions[eid] = {
        'path': session_path,
        'probes': [],
        'subject': subject,
        ...
    }
sessions[eid]['probes'].append(probe_name)

...
for i, (eid, session_info) in enumerate(sorted(sessions.items())):
    result = process_session(session_info, cache_dir, br)
```

iii. In `CONVERSION_NOTES.md`, the agent says this matches the reference code because probes from the same session are not treated as independent and should be merged.

## 1-d. How are the data split into trials?

i. Trials are taken from rows of the session trial table. After applying the trial mask, the script iterates over each remaining row of `valid_trials`, treating each row as one trial and extracting a fixed 2 s window around stimulus onset.

ii.
```python
trials, mask = load_trials(session_path)
...
valid_trials = trials[mask].copy()

for trial_idx, (df_idx, trial) in enumerate(valid_trials.iterrows()):
    stim_on = trial[ALIGN_TIME]
    t_start = stim_on + TIME_WINDOW[0]
    t_end = stim_on + TIME_WINDOW[1]
```

iii. The notes justify this as matching the trial-aligned decoder setup: align to `stimOn_times`, use a `(-0.5, 1.5)` s window, and produce one trial matrix per trial.

## 1-e. How are trials filtered based on quality controls?

i. The agent applies a boolean mask that removes trials with missing required events, reaction times outside 0.08-2.0 s, and no-choice trials (`choice == 0`). It then applies a second effective filter by dropping any masked trial whose wheel or whisker data cannot be interpolated cleanly across the whole aligned window or contains NaNs.

ii.
```python
for event in NAN_EXCLUDE:
    if event in trials.columns:
        mask &= ~trials[event].isna()

rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= (rt >= MIN_RT) & (rt <= MAX_RT)

if 'choice' in trials.columns:
    mask &= (trials['choice'] != 0)

...
if ws is None or wm is None:
    continue
if np.any(np.isnan(ws)) or np.any(np.isnan(wm)):
    continue
```

iii. `CONVERSION_NOTES.md` says this matches `load_trials_and_mask()` in the reference utilities. The trajectory also states the intended mask was: NaN required events, RT in range, and exclude no-choice trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from per-probe spike timestamps and cluster IDs. Region annotations are derived indirectly from cluster-to-channel assignments and channel brain-location IDs.

ii.
```python
spike_times = np.load(revision_path / 'spikes.times.npy')
spike_clusters = np.load(revision_path / 'spikes.clusters.npy')
clusters_channels = np.load(revision_path / 'clusters.channels.npy')
chan_brain_ids = np.load(revision_path / 'channels.brainLocationIds_ccf_2017.npy')
```

iii. The notes say the script follows the Zhang reference loader, which uses all spike-sorted clusters and maps regions through Beryl labels.

## 2-b. How is the `neural` data processed?

i. The agent merges probes within a session, remaps merged cluster IDs to a dense 0-based index, then bins spikes into non-overlapping 20 ms count bins over a 2 s stimulus-aligned window. The resulting per-trial neural matrix has shape `(n_clusters, 100)`.

ii.
```python
spike_times, spike_clusters, all_channels, all_brain_ids, all_labels = \
    merge_probes(spikes_list, channels_list, brain_ids_list, labels_list)

cluster_ids = np.unique(spike_clusters)
cluster_id_to_idx = {cid: idx for idx, cid in enumerate(cluster_ids)}
spike_clusters_remapped = np.array([cluster_id_to_idx[c] for c in spike_clusters])

neural = bin_spikes_trial(
    spike_times, spike_clusters_remapped, n_clusters,
    t_start, t_end, BINSIZE, N_BINS
)
```

iii. `CONVERSION_NOTES.md` explicitly justifies: merge probes, use 20 ms bins, and bin spikes over `(-0.5, 1.5)` around `stimOn_times`, citing `0_data_caching.py`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent does not filter neural units by quality label. It loads all clusters, keeps sessions if they have spike data and at least two remaining trials, and does not apply the paper's well-isolated-neuron or minimum-five-good-neuron restriction in code.

ii.
```python
# Load cluster metrics for quality labels
metrics_files = list(revision_path.glob('clusters.metrics.pqt'))
cluster_labels = None
if metrics_files:
    metrics = pd.read_parquet(metrics_files[0])
    cluster_labels = metrics['label'].values

...
if not spikes_list:
    print(f"    Skipping: no spike data")
    return None
```

iii. The notes and trajectory both justify this by pointing to the reference caching code's call path: `prepare_data()` uses `load_spiking_data(..., qc=None)`, so the agent concluded that loading all clusters was the intended behavior.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to stimulus onset (`stimOn_times`). The script defines a per-trial interval from 0.5 s before to 1.5 s after that event and bins spikes within that interval.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)

...
stim_on = trial[ALIGN_TIME]
t_start = stim_on + TIME_WINDOW[0]
t_end = stim_on + TIME_WINDOW[1]
```

iii. The notes say this follows both the task instruction "Temporally align based on stimulus onset" and the reference caching parameters in `0_data_caching.py`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use fixed 20 ms bins throughout, yielding 100 bins per 2 s trial. No additional temporal rebinning is applied after the initial spike binning and behavior interpolation.

ii.
```python
BINSIZE = 0.02  # 20ms bins
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]  # 2.0s
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 bins
```

iii. `CONVERSION_NOTES.md` says the agent chose 20 ms because `0_data_caching.py` uses `binsize: 0.02` even though the methods paper discusses 50 ms for some static targets.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the stimulus-onset event and the hard-coded alignment window/bin size, not from an independent raw sampled signal. The only raw trial variable it depends on is `stimOn_times`.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02

stim_on = trial[ALIGN_TIME]
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. The trajectory says the decoder task required a continuous time-since-stimulus input, so the agent synthesized it directly from the alignment definition.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The script constructs a deterministic 100-sample vector spanning the aligned trial window, starting one bin after the left edge and ending at 1.5 s. It stores this same vector in every trial of every session.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
input_data = np.stack([
    time_since_stim,
    np.full(N_BINS, trial_num, dtype=np.float32)
], axis=0)
```

iii. The justification recorded in the notes is that behavioral interpolation in the reference utilities also uses uniformly spaced sample points across the aligned interval, so the agent used the same convention.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is aligned by construction: the input vector has the same `N_BINS` and same stimulus-centered time window as the neural matrix for that trial.

ii.
```python
neural = bin_spikes_trial(
    spike_times, spike_clusters_remapped, n_clusters,
    t_start, t_end, BINSIZE, N_BINS
)

time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. The notes justify this as matching the reference code's shared interval/bin definition for aligned signals.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the trial table's `probabilityLeft` column after trial masking. The code infers block identity from runs of equal `probabilityLeft`.

ii.
```python
def compute_trial_number_in_block(trials_df, mask):
    trials_masked = trials_df[mask].copy()
    prob_left = trials_masked['probabilityLeft'].values
```

iii. The trajectory does not discuss this variable explicitly. The apparent justification is to use `probabilityLeft` because that is the block-defining quantity described in the task and papers.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The script looks for change points in the masked `probabilityLeft` sequence and assigns 0-based trial counters within each constant-probability run. Because it operates after masking, excluded trials are removed before the within-block count is computed.

ii.
```python
block_changes = np.concatenate([[0], np.where(np.diff(prob_left) != 0)[0] + 1])
trial_in_block = np.zeros(len(prob_left), dtype=int)

for i in range(len(block_changes)):
    start = block_changes[i]
    end = block_changes[i + 1] if i + 1 < len(block_changes) else len(prob_left)
    trial_in_block[start:end] = np.arange(end - start)
```

iii. No explicit justification was recorded in the notes or trajectory. The code suggests the agent wanted a simple per-block counter from the filtered trial sequence.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived directly from the trial-table `choice` field.

ii.
```python
choice_val = 0 if trial['choice'] == 1 else 1
```

iii. `CONVERSION_NOTES.md` says this follows the decoder task's requested left/right coding while respecting the IBL sign convention.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The script assumes the IBL encoding `1 = left`, `-1 = right`, filters out `choice == 0` trials earlier, and remaps the remaining values to decoder labels `left = 0`, `right = 1`. It then broadcasts the per-trial label across all 100 time bins.

ii.
```python
if 'choice' in trials.columns:
    mask &= (trials['choice'] != 0)

...
choice_val = 0 if trial['choice'] == 1 else 1

output = np.stack([
    np.full(N_BINS, choice_val, dtype=np.int64),
    ...
], axis=0)
```

iii. The notes explicitly justify the remapping as required by the decoder task specification.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior is derived directly from the trial-table `probabilityLeft` field.

ii.
```python
prob_left = trial['probabilityLeft']
```

iii. The notes point to the task specification and the experimental block structure in the papers: the latent block prior is represented by `probabilityLeft`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The agent discretizes the raw floating-point probabilities using a fixed mapping `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`, with a silent fallback to `1` if any other value appears. It then broadcasts the result across time bins.

ii.
```python
if prob_left == 0.2:
    prior_val = 0
elif prob_left == 0.5:
    prior_val = 1
elif prob_left == 0.8:
    prior_val = 2
else:
    prior_val = 1  # fallback
```

iii. `CONVERSION_NOTES.md` says this exact remapping was chosen because the decoder task requested those category labels.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from raw wheel position samples and raw wheel timestamps.

ii.
```python
wheel_pos_files = list(session_path.rglob('_ibl_wheel.position.npy'))
wheel_ts_files = list(session_path.rglob('_ibl_wheel.timestamps.npy'))

wheel_pos = np.load(wheel_pos_files[0])
wheel_ts = np.load(wheel_ts_files[0])
```

iii. The notes justify wheel-speed extraction by referring to the reference utility `load_target_behavior(..., 'wheel-speed')`, but the implementation was rewritten directly against the underlying wheel arrays.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The agent computes a simple finite-difference wheel velocity, takes its absolute value to obtain speed, assigns midpoint timestamps to the derivative samples, then linearly interpolates speed onto the trial bins.

ii.
```python
dt = np.diff(wheel_ts)
dp = np.diff(wheel_pos)
vel = dp / dt
speed = np.abs(vel)
vel_ts = wheel_ts[:-1] + dt / 2

...
ws = interpolate_behavior_to_bins(wheel_ts, wheel_speed, t_start, t_end, BINSIZE, N_BINS)
```

iii. `CONVERSION_NOTES.md` says the intent was to match reference wheel speed, but it explicitly acknowledges a limitation: the reference code uses `SessionLoader.load_wheel()` and Gaussian-smoothed velocity, whereas this script uses a simple finite difference.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. After all valid trials in a session are processed, the script pools all wheel-speed values across those trials, computes session-specific quantile edges, and digitizes each time point into 3 equal-count bins labeled low/medium/high.

ii.
```python
quantiles = np.linspace(0, 100, n_bins + 1)
edges = np.percentile(all_vals, quantiles)
edges[0] = -np.inf
edges[-1] = np.inf

...
wheel_disc, wheel_edges = discretize_to_bins(wheel_speed_raw, N_WHEEL_BINS)
```

iii. The notes justify quantile discretization as a practical choice because the decoder task required categorical wheel speed but the reference materials only describe continuous wheel-speed decoding.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same stimulus-centered trial window as the neural data. The script linearly interpolates the continuous wheel-speed trace onto 100 regularly spaced samples spanning the neural interval.

ii.
```python
bin_centers = np.linspace(t_start + binsize, t_end, n_bins)
...
ws = interpolate_behavior_to_bins(wheel_ts, wheel_speed, t_start, t_end, BINSIZE, N_BINS)
```

iii. The notes justify this by saying the decoder task required stimulus-onset alignment and the reference `get_behavior_per_interval()` uses linear interpolation onto aligned bins.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from precomputed ROI motion-energy arrays and their camera timestamps, preferring left-camera files and falling back to right-camera files.

ii.
```python
left_me_files = list(session_path.rglob('leftCamera.ROIMotionEnergy.npy'))
left_times_files = list(session_path.rglob('_ibl_leftCamera.times.npy'))
...
right_me_files = list(session_path.rglob('rightCamera.ROIMotionEnergy.npy'))
right_times_files = list(session_path.rglob('_ibl_rightCamera.times.npy'))
```

iii. `CONVERSION_NOTES.md` says this matches the reference behavior loader, which tries left whisker motion energy first and then right.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The agent does not recompute motion energy from video frames. It loads the precomputed motion-energy trace, adjusts for the possible off-by-one length difference between frame times and motion-energy samples, prefers left view and falls back to right, and later interpolates the trace to the neural bins.

ii.
```python
if left_me_files and left_times_files:
    me = np.load(left_me_files[0])
    times = np.load(left_times_files[0])
    if len(me) == len(times):
        return times, me
    elif len(me) == len(times) - 1:
        return times[:-1], me
```

iii. The notes justify left-camera preference from the reference code and the data paper's description of the camera setup.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is thresholded exactly like wheel speed: the script pools all session-valid whisker-motion-energy values, computes session-level quantile edges, and digitizes each time point into 3 bins.

ii.
```python
whisker_disc, whisker_edges = discretize_to_bins(whisker_me_raw, N_WHISKER_BINS)

output = np.stack([
    ...,
    whisker_disc[i].astype(np.int64),
], axis=0)
```

iii. The notes give the same justification as wheel speed: the decoder task required discrete outputs but the source materials describe continuous signals, so the agent chose quantile binning.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The whisker trace is aligned to the same stimulus-centered trial interval as the neural data and interpolated onto the same 100 regularly spaced samples.

ii.
```python
wm = interpolate_behavior_to_bins(me_ts, me_values, t_start, t_end, BINSIZE, N_BINS)
...
output = np.stack([
    ...,
    whisker_disc[i].astype(np.int64),
], axis=0)
```

iii. The notes justify this the same way as wheel speed: shared stimulus-onset alignment and linear interpolation to the neural bin grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed data are handled mostly by skipping. The script skips sessions with missing trials, missing spikes, missing wheel data, or fewer than two retained trials; skips trials when wheel/whisker interpolation coverage is insufficient or contains NaNs; and silently maps unexpected `probabilityLeft` values to the middle prior category.

ii.
```python
if not trials_files:
    return None, None
...
if not spikes_list:
    print(f"    Skipping: no spike data")
    return None
...
if wheel_speed is None:
    print(f"    Skipping: no wheel data")
    return None
...
if ws is None or wm is None:
    continue
if np.any(np.isnan(ws)) or np.any(np.isnan(wm)):
    continue
...
else:
    prior_val = 1  # fallback
```

iii. The notes justify some of these skips as necessary to keep aligned neural/behavioral trials, but there is no explicit recorded defense of the silent `prior_val = 1` fallback or of skipping whole sessions instead of using the reference loader to fetch missing data.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive steps are the nested session-by-session, trial-by-trial spike binning and behavior interpolation, plus repeated recursive filesystem searches with `rglob()`. Trial spike binning touches every spike window separately; behavior interpolation creates a fresh interpolator for each trial and modality.

ii.
```python
for i, (eid, session_info) in enumerate(sorted(sessions.items())):
    result = process_session(session_info, cache_dir, br)

...
for trial_idx, (df_idx, trial) in enumerate(valid_trials.iterrows()):
    neural = bin_spikes_trial(...)
    ws = interpolate_behavior_to_bins(...)
    wm = interpolate_behavior_to_bins(...)
```

iii. No explicit performance justification was recorded. This is an assessment of the code path the agent chose.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python loops could have been vectorized or moved to array operations: remapping cluster IDs with a Python list comprehension, computing cluster-region IDs one cluster at a time, computing trial-in-block via a Python block loop, rebuilding `time_since_stim` every trial, and converting per-session region names to indices using repeated `list.index()` calls.

ii.
```python
spike_clusters_remapped = np.array([cluster_id_to_idx[c] for c in spike_clusters])

for cid in cluster_ids:
    if cid < len(all_channels):
        ch = int(all_channels[cid])
        ...

for i in range(len(block_changes)):
    ...
    trial_in_block[start:end] = np.arange(end - start)

idx = np.array([all_regions_flat.index(r) for r in regions])
```

iii. No explicit justification was recorded for keeping these loops in Python.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly rebuilds the same time-since-stim vector per trial, repeatedly constructs interpolation objects per trial for wheel and whisker signals, repeatedly scans the filesystem with `rglob()` for each session and modality, and recomputes ordered unique-region lookups using repeated membership/index checks.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)

f = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')
return f(bin_centers)

wheel_pos_files = list(session_path.rglob('_ibl_wheel.position.npy'))
...
left_me_files = list(session_path.rglob('leftCamera.ROIMotionEnergy.npy'))
```

iii. No explicit justification was recorded; these repetitions appear to be artifacts of a straightforward implementation.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads or computes several values that are not used meaningfully downstream: cluster quality labels are loaded but not used for filtering, `clusters_depths` is loaded and discarded immediately, `valid_indices`, `good_trial_indices`, `all_subjects`, and `all_brain_region_idx` are accumulated or created without affecting the saved dataset, and wheel/whisker bin edges are returned from `process_session()` but not stored in the final dataset.

ii.
```python
clusters_depths = np.load(revision_path / 'clusters.depths.npy')
...
cluster_labels = metrics['label'].values
...
valid_indices = valid_trials.index.tolist()
...
good_trial_indices = []
...
all_subjects = []
all_brain_region_idx = []
...
return {
    ...,
    'wheel_edges': wheel_edges,
    'whisker_edges': whisker_edges,
}
```

iii. No explicit justification was recorded for these discarded computations.
