# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Data are loaded using the AllenSDK `VisualBehaviorOphysProjectCache.from_local_cache()`. The AI first builds a local experiment table restricted to NWB files actually present on disk (`local_experiment_table()`), then applies two selection filters: (1) exclude passive sessions (`~et.passive`), and (2) keep only familiar image set sessions (`session_type in ('OPHYS_1_images_A', 'OPHYS_3_images_A')`). For each session, each imaging plane is loaded via `bc.get_behavior_ophys_experiment(eid)`.

ii.
```python
def get_cache():
    import allensdk.brain_observatory.behavior.behavior_project_cache as bpc
    return bpc.VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=CACHE_DIR)

def local_experiment_table():
    ids = sorted(int(re.search(r'(\d+)\.nwb', f).group(1))
                 for f in glob.glob(os.path.join(NWB_DIR, '*.nwb')))
    et = get_cache().get_ophys_experiment_table()
    et = et.loc[et.index.intersection(ids)]
    return et

def select_experiments():
    et = local_experiment_table()
    et = et[~et.passive]                                     # active behavior only
    et = et[et.session_type.isin(FAMILIAR_SESSION_TYPES)]    # familiar image set only
    return et
```

iii. The AI justifies familiar-only filtering by citing the paper ("we restricted our analysis to familiar stimuli") and noting that all familiar sessions use the same 8 images (image set A), so image identity is a consistent 8-way categorical output. Active-only filtering is justified because passive sessions lack Go/Catch trial structure. The `from_local_cache` approach avoids S3 downloads by using locally available NWB files.

## 1-b. How are the data split into subjects?

i. Subjects correspond to unique `mouse_id` values extracted from the filtered experiment table. Subjects are sorted and indexed for consistent ordering.

ii.
```python
subjects = sorted({r['mouse_id'] for r in good})
subj_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. `mouse_id` is the SDK's unique identifier for each animal. The AI's approach yields 38 mice from the locally available data.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `ophys_session_id` values. Multiple imaging planes (experiments) within the same session are grouped together. The AI groups experiments by `ophys_session_id`, creating one combined session per unique ID. Sessions are further filtered to familiar image set and active behavior only.

ii.
```python
sessions = []
for sid, grp in et.groupby('ophys_session_id'):
    meta = grp.iloc[0][['mouse_id', 'cre_line', 'session_type', ...]].to_dict()
    sessions.append((int(sid), list(grp.index.values), meta))
sessions.sort(key=lambda s: s[0])
```

iii. Grouping by `ophys_session_id` ensures all simultaneously recorded imaging planes are treated as one session. The AI filters to OPHYS_1_images_A and OPHYS_3_images_A (familiar active sessions) based on the paper's restriction to familiar stimuli. This yields 91 sessions (92 minus 1 dropped for no eye tracking).

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's `trials` table. The AI selects trials where `(go | catch) & ~auto_rewarded`. For each trial, a fixed-length window of [-2.25, +3.75] seconds around `change_time` is used, divided into 8 bins of 750 ms each. This gives fixed-length trials (8 time bins) for all trials.

ii.
```python
tr = beh['trials']
keep = (tr['go'].astype(bool) | tr['catch'].astype(bool)) & ~tr['auto_rewarded'].astype(bool)
tr = tr[keep]
# ...
offs = OFF_START + BIN_SIZE * np.arange(NBINS + 1)  # OFF_START=-2.25, BIN_SIZE=0.75, NBINS=8
edges = change_times[:, None] + offs[None, :]       # (ntrials, NBINS+1)
```

iii. The AI justifies the 750 ms bin size by noting it matches the paper's image-presentation interval (250 ms image + 500 ms grey), and that bins align exactly with flash onsets since `change_time` is always a flash onset and 2.25 s = 3 x 750 ms. The window [-2.25, +3.75] s is the largest multiple of 750 ms that fits inside every trial (measured min pre-change 2.79 s, min post-change 4.20 s).

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) keeping only go and catch trials (excluding aborted), (2) excluding auto-rewarded trials, (3) dropping trials where running speed or pupil diameter bins contain NaN values (missing behavioral data), (4) dropping entire sessions without eye tracking data (1 session), (5) dropping sessions with fewer than 2 usable trials.

ii.
```python
keep = (tr['go'].astype(bool) | tr['catch'].astype(bool)) & ~tr['auto_rewarded'].astype(bool)
# ...
bad = (~np.isfinite(run_binned).all(axis=1)) | (~np.isfinite(pupil_binned).all(axis=1))
good = ~bad
if good.sum() < 2:
    return dict(session_id=session_id, skip=f'<2 usable trials ({int(good.sum())})', ...)
# ...
if beh['eye'] is None or len(beh['eye']) == 0:
    return dict(session_id=session_id, skip='no eye tracking', timings=timings)
```

iii. Per instructions, aborted and auto-rewarded trials are excluded. The AI additionally drops trials with missing behavioral data (728 trials, 3.2%) rather than imputing values, and drops 1 session without eye tracking because pupil diameter is a required output. The AI verifies `change_time` is always present for go/catch trials via assertion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `events` column of the SDK's `experiment.events` table, which contains detected calcium-event magnitudes (one value per ophys frame per cell).

ii.
```python
ev = np.vstack(ds.events[signal].values).astype(np.float64)   # (ncells, nframes)
ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
```

iii. The AI cites the paper: "For all analysis of neural data we used the detected calcium events." The SDK provides both `events` (detected calcium-event magnitudes) and `filtered_events` (smoothed version "for visualization"). The AI empirically compared both and found small, inconsistent differences in decoder accuracy, choosing `events` based on the paper's explicit statement.

## 2-b. How is the `neural` data processed?

i. Neural data is processed in three steps: (1) detected calcium events from all imaging planes of a session are concatenated along the neuron axis, (2) events are averaged within 750 ms time bins using a vectorized cumsum+searchsorted approach, (3) per-neuron z-scoring is applied within each session (mean and std computed over all selected trials, then neural = (neural - mean) / std).

ii.
```python
# Binning: mean event magnitude per bin
sums, counts = bin_sum_count(p['events'], p['ts'], edges)
neural_blocks.append((sums / counts[None, :, :]).astype(np.float32))
neural = np.concatenate(neural_blocks, axis=0)  # (ncells, ntrials, NBINS)

# Z-scoring
if normalize in ('std', 'zscore'):
    flat = neural.reshape(neural.shape[0], -1)
    mu = flat.mean(axis=1, keepdims=True)
    sd = flat.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    if normalize == 'zscore':
        flat = (flat - mu) / sd
    neural = flat.reshape(neural.shape).astype(np.float32)
```

iii. The AI justifies the mean (not sum) within bins as making values comparable between 31 Hz (single-plane) and 11 Hz (Mesoscope) rigs. Z-scoring is justified because event magnitudes span orders of magnitude across neurons, and the decoder's SVD initialization would otherwise be dominated by a few high-amplitude cells. The AI documents that z-scoring is a monotone per-neuron affine transform that preserves event timing and relative time courses.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied beyond the SDK's built-in filtering (only `valid_roi=True` cells are returned). The AI verified that all cells have valid ROI status, there are no all-zero cells, no NaN values, and QC-failed planes/sessions are already removed from the release.

ii. N/A (no filtering code)

iii. The AI cites the SDK documentation: "Table only contains roi_valid = True entries, as invalid ROIs / non-cell segmented objects have been filtered out." The paper applies no further per-cell filtering, so none is added.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to `change_time` from the trials table. The alignment event is the onset of the image change on Go trials, or the sham change on Catch trials. Bin edges are computed as offsets from `change_time`, spanning [-2.25, +3.75] seconds in 750 ms increments.

ii.
```python
change_times = tr['change_time'].values.astype(np.float64)
offs = OFF_START + BIN_SIZE * np.arange(NBINS + 1)  # [-2.25, -1.50, ..., +3.75]
edges = change_times[:, None] + offs[None, :]         # (ntrials, NBINS+1)
```

iii. The AI justifies change_time alignment by citing the reference code (`save_trial_response_df.py` aligns to `change_time`) and verifying that every `change_time` coincides exactly with a flash onset. The window [-2.25, +3.75] s ensures bin edges always fall on flash onsets (since 2.25 = 3 x 0.75 and 3.75 = 5 x 0.75), making each bin correspond to exactly one image-presentation interval.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 750 ms per bin (8 bins per trial). This is a rebinning from the native ophys frame rate (~31 Hz for single-plane, ~11 Hz for Mesoscope). The bin size matches the paper's image-presentation interval (250 ms image + 500 ms grey = 750 ms cycle).

ii.
```python
BIN_SIZE = 0.750          # s, time bin size = one image-presentation interval
OFF_START = -2.25         # s, relative to change_time
OFF_END = 3.75            # s, relative to change_time
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 8
```

iii. The AI justifies 750 ms bins by: (a) it matches the paper's unit of analysis (750 ms image-presentation interval), (b) detected calcium events are sparse (0.02-0.15 events/s/cell), so smaller bins contain mostly zeros which hurts decoder performance, (c) empirically, 750 ms bins improved decoder accuracy for all outputs compared to 250 ms bins (e.g., image_identity 0.334 -> 0.427).

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations` (filtered to the `change_detection` stimulus block), specifically the `image_name` column. Omitted flashes are forward-filled with the previous image identity, since omissions replace repeats.

ii.
```python
stim = beh['stim']
stim = stim[stim.stimulus_block_name.astype(str).str.contains('change_detection')]
stim = stim.sort_values('start_time')
fstart = stim['start_time'].values.astype(np.float64)
omitted = stim['omitted'].fillna(False).astype(bool).values
names = stim['image_name'].astype(str).values
names_ff = pd.Series(np.where(omitted, None, names)).ffill().bfill().values
image_names = sorted(set(names_ff[~pd.isna(names_ff)]) - {'omitted'})
name_to_idx = {n: i for i, n in enumerate(image_names)}
img_idx = np.array([name_to_idx[n] for n in names_ff], dtype=np.int64)
```

iii. The AI uses stimulus_presentations rather than the trials table because it provides exact flash timing for every presentation, enabling precise time-varying image identity. Forward-filling omissions is justified because an omission replaces a repeat of the current image, so the ongoing image identity is unchanged.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image identity is computed per time bin by finding which image-presentation interval (750 ms) contains each bin center. The index of the last flash onset before each bin center is found via `searchsorted`, and the image name at that flash is used as the bin's image identity. Image names are mapped to integer indices via a sorted global mapping.

ii.
```python
# index of the image-presentation interval containing each bin centre
j = np.searchsorted(fstart, centers, side='right') - 1
image_identity = img_idx[j]                                   # (ntrials, NBINS)
```

iii. Using bin centers to determine which flash interval a bin belongs to is justified because each 750 ms bin corresponds to exactly one image-presentation interval (since bin edges are aligned to flash onsets). The sorted global mapping ensures consistent integer codes across sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned by using the same bin edges (derived from change_time + offsets) that are used for neural data binning. Each bin's image identity is determined by the bin center's position relative to stimulus flash onsets.

ii.
```python
centers = edges[:, :-1] + BIN_SIZE / 2.0
j = np.searchsorted(fstart, centers, side='right') - 1
image_identity = img_idx[j]
```

iii. Since both neural data and image identity use the same bin edges (aligned to change_time), they are inherently aligned. The bin center approach ensures each bin is assigned the image identity of the flash interval it falls within.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column of `stimulus_presentations` (filtered to the change_detection block). For each time bin, the `is_change` value of the flash interval containing the bin center is used.

ii.
```python
is_change = stim['is_change'].fillna(False).astype(bool).values
# ...
image_change = is_change[j].astype(np.int64)                  # (ntrials, NBINS)
```

iii. The `is_change` flag from stimulus_presentations directly indicates whether a flash is a change or repeat/sham. For catch trials, `is_change` is False (it's a sham change, not a real image identity change), so image_change is 0 throughout, matching the expected behavior described in the instructions.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` boolean is mapped to integer (0/1) via the same bin-center-to-flash-interval lookup used for image identity. The change indicator is 1 for the 750 ms bin containing the change flash and 0 elsewhere. On catch trials (sham change), `is_change` is False, so image_change is always 0.

ii.
```python
image_change = is_change[j].astype(np.int64)
```

iii. This produces a binary time-varying output where 1 indicates the image-presentation interval containing the change. Since bins are 750 ms and aligned to flash onsets, exactly one bin per Go trial has image_change=1 (the change interval), and 0 bins on Catch trials.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1), derived directly from the boolean `is_change` flag. No thresholding is applied.

ii.
```python
image_change = is_change[j].astype(np.int64)
```

iii. The `is_change` flag is already boolean; converting to int gives the binary categories directly.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same alignment as image identity: both use the same bin edges derived from change_time offsets. The `is_change` value of the flash interval containing each bin center is used.

ii. See 4-a.

iii. Same bin-edge alignment as neural data and all other outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `running_speed` (the filtered/processed version), specifically the `speed` column (cm/s) and `timestamps` column.

ii.
```python
run = beh['run']
run_t = run['timestamps'].values.astype(np.float64)
run_v = run['speed'].values.astype(np.float64)
```

iii. The `running_speed` attribute provides the SDK's standard 10 Hz low-pass filtered running speed from the wheel encoder.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is averaged within each 750 ms time bin (NaN-aware mean), then discretized into 5 equal percentile bins computed globally across all sessions. Bin edges are the 20th, 40th, 60th, and 80th percentiles of all running speed values in the dataset.

ii.
```python
run_binned = bin_mean_1d(run_v, run_t, edges, nan_aware=True)  # (ntrials, NBINS)
# ...
run_all = np.concatenate([r['run_binned'].ravel() for r in good])
_, run_edges_g = quantile_bin(run_all)
run_cls = np.digitize(r['run_binned'], run_edges)
```

iii. NaN-aware mean handles any gaps in running speed data. Global percentile bins ensure consistent category meanings across sessions. The AI initially used per-session percentile bins but switched to global after noting that the shared decoder readout requires consistent class semantics.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized using `np.digitize` with 4 edges (20th/40th/60th/80th percentiles), producing 5 categories (0-4). Edges are computed from all valid running speed bin values across the entire dataset.

ii.
```python
def quantile_bin(values, nq=NQUANTILES):
    finite = v[np.isfinite(v)]
    qs = np.linspace(0, 100, nq + 1)[1:-1]  # [20, 40, 60, 80]
    edges = np.percentile(finite, qs)
    labels = np.digitize(v, edges, right=False)
    return labels.astype(np.int64), edges
```

iii. `np.digitize` with 4 interior edges creates 5 bins. The percentile-based edges ensure roughly equal class counts across the dataset.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is binned using the same time bin edges as neural data (derived from change_time + offsets). The `bin_mean_1d` function computes the mean running speed within each bin using searchsorted on the running speed timestamps.

ii.
```python
run_binned = bin_mean_1d(run_v, run_t, edges, nan_aware=True)
```

iii. Using the same bin edges guarantees alignment with neural data. The running speed timestamps are hardware-synced with the ophys clock via the 100 kHz sync board.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `eye_tracking['pupil_area']`, converted to diameter via `2 * sqrt(pupil_area / pi)`. This uses the SDK's circular area convention where `pupil_area = pi * max(width, height)^2`.

ii.
```python
eye = beh['eye']
eye_t = eye['timestamps'].values.astype(np.float64)
pupil_diam = 2.0 * np.sqrt(eye['pupil_area'].values.astype(np.float64) / np.pi)
```

iii. The AI verified numerically that `pupil_area = pi * max(width, height)^2` (not `pi * width * height`), so `diameter = 2 * sqrt(area / pi) = 2 * max(width, height)` pixels. This gives the diameter of the circle with the computed area.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter processing: (1) compute diameter from pupil_area, (2) linearly interpolate NaN values (blinks) across gaps shorter than 1 second, (3) compute NaN-aware mean within each 750 ms time bin, (4) discretize into 5 equal percentile bins computed globally across all sessions. Trials with remaining NaN bins (from long blinks or tracking failures) are dropped.

ii.
```python
pupil_diam = 2.0 * np.sqrt(eye['pupil_area'].values.astype(np.float64) / np.pi)
pupil_diam = interpolate_short_gaps(eye_t, pupil_diam, PUPIL_INTERP_MAX_GAP)  # max_gap=1.0s
pupil_binned = bin_mean_1d(pupil_diam, eye_t, edges, nan_aware=True)
# ...
bad = ... | (~np.isfinite(pupil_binned).all(axis=1))
```

iii. Short-gap interpolation (<=1s) fills brief blinks while preserving longer tracking failures as NaN. The 1s threshold was chosen because most blinks are brief. Trials with remaining NaN bins after interpolation are dropped rather than imputed, as a conservative approach to data quality.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: `np.digitize` with 4 edges (20th/40th/60th/80th percentiles computed globally), producing 5 categories (0-4).

ii.
```python
pupil_all = np.concatenate([r['pupil_binned'].ravel() for r in good])
_, pupil_edges_g = quantile_bin(pupil_all)
pupil_cls = np.digitize(r['pupil_binned'], pupil_edges)
```

iii. Same justification as running speed: global percentile bins for consistent category semantics across sessions with a shared decoder readout.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed: pupil diameter is binned using the same time bin edges as neural data.

ii.
```python
pupil_binned = bin_mean_1d(pupil_diam, eye_t, edges, nan_aware=True)
```

iii. Using the same bin edges guarantees alignment. Eye tracking timestamps are hardware-synced via the 100 kHz sync board.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
outcome_mat = np.stack([tr[c].astype(bool).values for c in OUTCOME_NAMES], axis=1)
assert np.all(outcome_mat.sum(axis=1) == 1), 'trial outcomes do not partition trials'
outcome = np.argmax(outcome_mat, axis=1).astype(np.int64)
```

iii. The AI verifies that the four outcome flags exactly partition the selected trials (each trial has exactly one True flag). This assertion confirms no edge cases are missed. The mapping is hit=0, miss=1, false_alarm=2, correct_reject=3.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is mapped to integer codes (0-3) via `np.argmax` over the outcome flag matrix, then broadcast as a constant value across all time bins of the trial.

ii.
```python
outcome = np.argmax(outcome_mat, axis=1).astype(np.int64)
# ...
np.full(NBINS, outcome[i], dtype=np.int64),  # in output array
```

iii. The outcome is static per trial (same value in every time bin), as required by the task specification. Broadcasting over time bins maintains the (n_output, n_timepoints) format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Sessions without eye tracking**: Skipped entirely (1 session), since pupil diameter is a required output.
- **Missing behavioral data in bins**: Trials with any NaN in running speed or pupil diameter bins are dropped (728 trials, 3.2%).
- **Blinks/tracking failures**: Short gaps (<=1s) in pupil data are linearly interpolated; longer gaps remain NaN and cause the trial to be dropped.
- **Failed sessions**: Try/except wraps each session conversion; failures are logged and skipped.
- **Data type issues**: Pandas nullable types (`<NA>`) are explicitly cast (`fillna(False).astype(bool)`); brain region labels cast to plain `str`.
- **Edge cases**: Assertions verify change_time is always present, outcome flags partition trials, no empty time bins, and bin centers don't precede the first flash.

ii.
```python
if beh['eye'] is None or len(beh['eye']) == 0:
    return dict(session_id=session_id, skip='no eye tracking', ...)
# ...
bad = (~np.isfinite(run_binned).all(axis=1)) | (~np.isfinite(pupil_binned).all(axis=1))
good = ~bad
# ...
pupil_diam = interpolate_short_gaps(eye_t, pupil_diam, PUPIL_INTERP_MAX_GAP)
```

iii. The AI documents every dropped session and trial count in the conversion funnel. The approach is conservative: rather than imputing missing data with default values, affected trials are dropped. This prevents potentially misleading behavioral labels from entering the dataset.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each experiment's NWB file via `bc.get_behavior_ophys_experiment()`, which reads large neural and behavioral data arrays from disk. This is I/O bound.

ii. N/A (profiling output from the conversion logs shows load times dominate)

iii. Each NWB file contains full-session data for all neurons, running speed, eye tracking, and stimulus presentations. The AI mitigates this with 8-process multiprocessing over sessions.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the main computational bottleneck: binning is done via cumulative sums + searchsorted over all neurons, trials, and bins simultaneously (`bin_sum_count`), with no Python loops over trials or bins. The remaining loops are over imaging planes (typically 1-7, inherently sequential for loading) and sessions (parallelized via multiprocessing).

ii.
```python
def bin_sum_count(values, timestamps, edges_flat):
    cs = np.concatenate([np.zeros((v.shape[0], 1)), np.cumsum(v, axis=1)], axis=1)
    sums = cs[:, idx[:, 1:]] - cs[:, idx[:, :-1]]
    counts = (idx[:, 1:] - idx[:, :-1]).astype(np.float64)
    return sums, counts
```

iii. The cumsum+searchsorted approach processes all neurons x all trials x all bins in 3 array operations, yielding 10-50x speedup over per-trial loops.

## 9-c. What processing does the code repeat multiple times?

i. No processing is meaningfully repeated. Each NWB file is opened exactly once per session. Behavioral streams (trials, running speed, eye tracking) are read once from the first plane's dataset. The only repeated operation is calling `get_cache()` per session in multiprocessing mode, which re-instantiates the cache object but doesn't reload data.

ii. N/A

iii. The multiprocessing architecture requires each worker to create its own cache connection, but this is a lightweight metadata operation.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes extensive per-session sanity checks (`sanity_checks()` function) and stores them in `metadata['session_checks']`. While valuable for validation, these checks are not used by the decoder. The AI also stores detailed `session_info` metadata per session (behavior_session_id, equipment_name, experience_level, etc.) that is not used downstream. The `interpolate_short_gaps` function for pupil data performs interpolation that may be unnecessary if those time bins would have been dropped anyway due to NaN.

ii.
```python
def sanity_checks(result, beh, tr, stim, change_times, offs):
    # Multiple consistency checks stored in result['checks']
    ...
```

iii. The sanity checks serve as validation during development. While they add computational overhead, they are quick compared to NWB loading and provide assurance of data integrity.
