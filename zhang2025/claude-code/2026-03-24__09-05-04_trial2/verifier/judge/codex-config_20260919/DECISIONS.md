# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively discovers local session directories under `data/one_cache`, keeps directories that appear to contain spikes, trials, wheel, and camera motion-energy files, then directly loads NumPy and Parquet files. It does not use ONE's release index or session identifiers.

ii.
```python
session_dirs = sorted(glob.glob(str(DATA_ROOT / '*/Subjects/*/*/001')))
...
if has_spikes and has_trials and has_wheel and has_me:
    valid.append(sdir)
```

iii. The notes say the cache has a predictable directory layout and report choosing the 393 locally complete sessions. They explicitly describe direct processing from the cache, rather than API-based resolution.

## 1-b. How are the data split into subjects?

i. Subject names are parsed from the directory component immediately after `Subjects`. A first-seen subject list is maintained, and each successful session is mapped to its index in that list.

ii.
```python
subject = parts[sub_idx + 1]
...
if subject not in all_subjects:
    all_subjects.append(subject)
subject_idx = np.array([all_subjects.index(s) for s in subject_per_session], dtype=np.int32)
```

iii. The agent regarded the cache path as carrying the mouse identity. The notes describe the layout as `<lab>/Subjects/<subject>/<date>/001`.

## 1-c. How are the data split into sessions?

i. Each matching `<subject>/<date>/001` directory is treated as one session and processed independently. The displayed session ID is constructed from subject and date.

ii.
```python
session_dirs = sorted(glob.glob(str(DATA_ROOT / '*/Subjects/*/*/001')))
session_id = f"{subject}_{date}"
```

iii. The notes identify that directory level as the session organization and report processing the resulting local directories sequentially.

## 1-d. How are the data split into trials?

i. The trials Parquet table is loaded as one row per trial. Every row's `stimOn_times` defines a two-second interval, and retained rows index the binned arrays.

ii.
```python
trials = pd.read_parquet(trial_files[-1])
stim_on = trials[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

iii. The agent treated the released trials table as already trial-wise, consistent with its dataset exploration.

## 1-e. How are trials filtered based on quality controls?

i. Trials must have reaction time from 80 ms through 2 s, nonzero choice, no NaNs in listed task fields, and (when available) go-cue-to-feedback duration no greater than 10 s. Wheel and camera streams must also cover the window. Sessions with fewer than two retained trials are skipped.

ii.
```python
mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
mask &= (trials['choice'] != 0)
for col in NAN_EXCLUDE:
    mask &= ~trials[col].isna()
combined_mask = mask.values & wheel_mask & whisker_mask
```

iii. The notes attribute the RT, no-choice, NaN, and 10-second rules to `load_trials_and_mask`, and behavior coverage to reference alignment logic.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from each probe's `spikes.times.npy` and `spikes.clusters.npy`. `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy` provide per-cluster Beryl region labels.

ii.
```python
spike_times = np.load(st_file).flatten()
spike_clusters = np.load(sc_file).flatten()
cluster_channels = np.load(cc_file).flatten()
channel_brain_ids = np.load(cb_file).flatten()
```

iii. The notes map spike times and cluster assignments to neural binning and the channel files to anatomy.

## 2-b. How is the `neural` data processed?

i. Probe cluster IDs are offset and merged, spikes are stably time-sorted, and spikes are counted per cluster in 100 non-overlapping 20-ms bins. Counts are clipped to 255 and stored as `uint8`; they are not divided by bin width into Hz.

ii.
```python
spike_clusters_offset = spike_clusters + cluster_offset
...
counts = np.bincount(flat_idx, minlength=n_clusters * N_BINS)
binned[trial_idx] = counts[:n_clusters * N_BINS].reshape(n_clusters, N_BINS)
...
np.clip(neural_trials[i], 0, 255).astype(np.uint8)
```

iii. The agent says this matches `bin_spiking_data`; it chose `uint8` to reduce the full pickle from the much larger float32 footprint, asserting counts rarely exceed 255.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No cluster-quality filtering is applied. All clusters with the four required files are retained, including `root` and `void` units.

ii.
```python
"""Following reference code: no QC filtering (qc=None)."""
...
n_clusters = len(cluster_channels)
```

iii. The notes distinguish the data paper's well-isolated population from `prepare_data(qc=None)` and choose all clusters to match the methods code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial spans `stimOn_times - 0.5` through `stimOn_times + 1.5` seconds. Sorted spikes in that absolute-clock interval are assigned bins relative to its beginning.

ii.
```python
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
...
bin_idx = ((times_trial - t_beg) / BINSIZE).astype(np.int32)
```

iii. The agent resolved the paper/code alignment discrepancy in favor of the task instruction and caching code, both of which require stimulus-onset alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 20 ms, producing 100 bins in a two-second window. Raw spike events are binned once; there is no later temporal rebinning or smoothing.

ii.
```python
BINSIZE = 0.02
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

iii. The notes select the uniform 20-ms configuration used by the reference caching code and required for the time-varying outputs.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is constructed from the `stimOn_times` alignment convention and the fixed `(-0.5, 1.5)` window, rather than copied from a raw trace.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
```

iii. The notes describe it as a decoder variable defined by the stimulus-aligned bin grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A 100-point float32 linear grid from -0.49 to 1.49 seconds is generated and reused for every trial.

ii.
```python
time_input = np.linspace(
    TIME_WINDOW[0] + BINSIZE / 2,
    TIME_WINDOW[1] - BINSIZE / 2,
    N_BINS
).astype(np.float32)
```

iii. The agent calls these the centers of the 20-ms neural bins.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Entry `j` is the center of neural spike-count bin `j`, with both defined relative to the same stimulus onset.

ii.
```python
bin_idx = ((times_trial - t_beg) / BINSIZE).astype(np.int32)
time_input = np.linspace(-0.49, 1.49, N_BINS).astype(np.float32)
```

iii. The notes state that all arrays share stimulus onset, window, and 20-ms resolution.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft`, treating each contiguous run of equal retained values as a block.

ii.
```python
prob_left = trials_good['probabilityLeft'].values
trial_num_in_block = compute_trial_num_in_block(prob_left)
```

iii. The notes say the prior is constant within a block and therefore identifies block boundaries.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. After all filtering, the agent scans retained trials, resets the counter to 1 whenever the prior changes, and otherwise increments it. Thus numbering is one-based and excluded trials do not advance the count.

ii.
```python
trial_nums = np.ones(len(prob_left), dtype=np.int32)
for i in range(1, len(prob_left)):
    if prob_left[i] == prob_left[i - 1]:
        trial_nums[i] = trial_nums[i - 1] + 1
```

iii. The notes merely describe “position within contiguous block”; they do not discuss the consequences of computing it after filtering or choosing one-based indexing.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the trials table's `choice` column after zero/no-response trials are excluded.

ii.
```python
choice = trials_good['choice'].values.copy()
choice_binary = np.where(choice == 1, 1, 0).astype(np.int32)
```

iii. The notes claim the intended mapping is `-1 (left) -> 0, +1 (right) -> 1`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The implementation maps raw `+1` to category 1 and every retained other value (`-1`) to 0, then broadcasts that category across all 100 time bins.

ii.
```python
choice_binary = np.where(choice == 1, 1, 0).astype(np.int32)
np.full(N_BINS, choice_binary[i], dtype=np.int64)
```

iii. The agent believed the raw convention was `-1=left, +1=right`; that premise is opposite the IBL convention used by the human reference.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `probabilityLeft` in the trials table.

ii.
```python
prob_left = trials_good['probabilityLeft'].values
```

iii. The notes identify this field as the block prior requested by the decoder task.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values near 0.2, 0.5, and 0.8 are encoded as 0, 1, and 2 respectively, then broadcast across time. The code defaults any retained unmatched value to category 1.

ii.
```python
prior = np.full(len(prob_left), 1, dtype=np.int32)
prior[np.isclose(prob_left, 0.2, atol=0.05)] = 0
prior[np.isclose(prob_left, 0.8, atol=0.05)] = 2
```

iii. The mapping is directly specified by the task; the tolerance was used as a permissive numeric comparison.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived directly from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
wh_pos = np.load(os.path.join(sdir, 'alf/_ibl_wheel.position.npy')).flatten()
wh_times = np.load(os.path.join(sdir, 'alf/_ibl_wheel.timestamps.npy')).flatten()
```

iii. The notes say this is intended to reproduce the absolute velocity returned by `SessionLoader.load_wheel()`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is linearly interpolated to 1 kHz, differentiated with `np.gradient`, converted to absolute velocity, and linearly interpolated again to trial sampling points. No low-pass filter is actually applied.

ii.
```python
t_uniform = np.arange(wh_times[0], wh_times[-1], 0.001)
pos_interp = np.interp(t_uniform, wh_times, wh_pos)
velocity = np.gradient(pos_interp, 0.001)
speed = np.abs(velocity)
```

iii. The comments claim this matches brainbox/SessionLoader and refer to Gaussian smoothing, but the implementation contains only interpolation and finite differencing.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The 33.33rd and 66.67th percentiles over all non-NaN retained wheel samples in a session define categories 0, 1, and 2.

ii.
```python
boundaries = np.percentile(flat, [100 / 3, 200 / 3])
result = np.digitize(values, boundaries).astype(np.int32)
```

iii. The agent chose session-wide terciles to make approximately balanced categorical targets, as documented in the notes.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It uses the same absolute trial bounds, but samples at `linspace(t_beg + 0.02, t_end, 100)`, i.e. from -0.48 through +1.50 s relative to stimulus onset. Those are bin right edges, not the neural centers (-0.49 through +1.49).

ii.
```python
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
values[trial_idx] = interp_func(x_interp).astype(np.float32)
```

iii. The agent says this matches `get_behavior_per_interval` and regards the outputs as stimulus-aligned, although its time coordinate differs by half a bin from `time_input`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses released `leftCamera.ROIMotionEnergy.npy` and `_ibl_leftCamera.times.npy` when available, otherwise the corresponding right-camera files.

ii.
```python
if left_me_files and left_time_files:
    me = np.load(left_me_files[-1]).flatten()
...
if right_me_files and right_time_files:
```

iii. The notes explicitly say left-camera preference with right fallback matches the reference code.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw trace and timestamps are truncated to their shorter length, then linearly interpolated to each trial's sampling grid. No filtering or normalization is applied.

ii.
```python
min_len = min(len(me), len(times))
return times[:min_len], me[:min_len]
...
interp_func = interp1d(beh_t, beh_v, kind='linear', fill_value='extrapolate')
```

iii. The agent says released motion energy should be used directly and interpolation should follow the behavioral helper in the methods code.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Session-wide 33.33rd and 66.67th percentiles over retained non-NaN samples define categories 0, 1, and 2.

ii.
```python
whisker_discrete = discretize_to_bins(whisker_trials, n_bins=3)
```

iii. The notes choose the same balanced-tercile rule as for wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera timestamps use the same stimulus-aligned absolute interval, but motion energy is sampled on the right-edge grid (-0.48 through +1.50 s), half a bin later than neural-bin centers.

ii.
```python
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
values[trial_idx] = interp_func(x_interp).astype(np.float32)
```

iii. The agent attributes this grid to the reference behavior interpolation and states that all behavior is stimulus-aligned.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Discovery excludes sessions lacking required file patterns. Timestamp/value length mismatches are silently truncated; trials without behavior coverage are masked; sessions with fewer than two valid trials are skipped; any other session exception is logged and the whole session is skipped. The full run lost 57 sessions due to missing wheel data and one for too few trials.

ii.
```python
min_len = min(len(me), len(times))
return times[:min_len], me[:min_len]
...
except Exception as e:
    print(f"  ERROR processing {session_id}: {e}", flush=True)
    return None
```

iii. The notes describe this as graceful missing-data handling, accepting a local-data subset and skipped sessions.

## 10-a. What are the most time-consuming steps of the code?

i. The code times spike binning explicitly, while full-run notes show memory accumulation and decoder preparation/training as major practical costs. Per-session raw spike I/O, spike binning, 1-kHz wheel interpolation, and behavior interpolation dominate conversion.

ii.
```python
t_bin = time.time()
binned_spikes = bin_spikes_vectorized(...)
print(f"  Spike binning: {time.time() - t_bin:.1f}s")
```

iii. Sample notes estimate 0.2–0.3 s for spike binning, 0.5–1 s for wheel/whisker work, and about 2.5 s total per session; full decoder float conversion caused a separate memory bottleneck.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial spike-binning and behavioral interpolation loops could be batched/vectorized. Trial-number computation could use change detection plus grouped cumulative counts. Output/input list construction and region/subject index lookup could also be vectorized or use dictionaries.

ii.
```python
for trial_idx in range(n_trials):
    ...
for i in range(1, len(prob_left)):
    ...
for i in range(n_trials):
    inp = np.stack([...])
```

iii. The agent labels spike binning “vectorized” because counting inside each trial uses flat indices and `bincount`, but it retains the outer trial loop for interval selection.

## 10-c. What processing does the code repeat multiple times?

i. It scans the cache with many glob calls, searches/slices every interval separately for spikes, wheel, and whisker, constructs identical time and constant per-trial rows repeatedly, and repeatedly performs linear subject lookup. It also bins spikes for all trials before applying the already-known task-quality mask.

ii.
```python
for trial_idx in range(n_trials):
    i_start = np.searchsorted(...)
...
subject_idx = np.array([all_subjects.index(s) for s in subject_per_session])
```

iii. The notes emphasize incremental memory handling, not repeated work; no explicit rationale is given for processing rejected trials first.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Spikes are binned for every raw trial and only later subset to valid trials. Continuous behavior arrays are retained long enough for plotting/discretization and then discarded. `lab`, the passed `BrainRegions` object, interval-end plotting arguments, and some detailed session identity are computed or passed but absent from the final data. The optional plots are also downstream-independent.

ii.
```python
binned_spikes = bin_spikes_vectorized(... interval_begs, interval_ends)
...
neural_trials = binned_spikes[good_indices]
...
lab, subject, date = parse_session_info(sdir)
```

iii. The agent intentionally bins before masking to mirror the reference alignment flow and retains optional plots for sanity checking; it does not justify the unused arguments/metadata.
