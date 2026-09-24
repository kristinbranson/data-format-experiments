# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the frozen BWM release CSV, groups its probe rows by `eid`, constructs each session's ALF path, and directly loads revisioned parquet/NumPy files. Full mode considers all listed sessions; sessions missing mandatory streams are caught and excluded.

ii.
```python
bwm = pd.read_csv(RELEASE_CSV, index_col=0)
grouped = bwm.groupby("eid", sort=False)
path = resolve_latest(session_path / "alf", "**/_ibl_trials.table.pqt")
return pd.read_parquet(path)
```

iii. The notes say the 459-session frozen CSV matches the public release and direct ALF loading was chosen because `SessionLoader` could not be imported and local ONE access encountered permission issues.

## 1-b. How are the data split into subjects?

i. Subject identifiers come from the release CSV. The final unique subject list is sorted and each retained session receives the corresponding index.

ii.
```python
subject=str(row["subject"])
subject_names = sorted({ps.spec.subject for ps in prepared_sessions})
subject_to_idx = {subject: i for i, subject in enumerate(subject_names)}
```

iii. The notes treat frozen-release metadata as authoritative and state that `subject_idx` follows final session order.

## 1-c. How are the data split into sessions?

i. Rows are grouped by release `eid`; each group is one session and supplies its probes. Retained sessions become elements of the outer lists.

ii.
```python
for eid, df in grouped:
    probe_names = tuple(df["probe_name"].tolist())
    sessions.append(SessionSpec(eid=eid, ..., probe_names=probe_names))
```

iii. The agent justified the CSV as the frozen 459-session/699-insertion release, avoiding unrelated sessions in newer local registries.

## 1-d. How are the data split into trials?

i. Each trials-table row is treated as a trial. Kept alignment times define trial windows, and neural, input, and output lists are built one retained row/window at a time.

ii.
```python
kept_align_times = prepared.trials.loc[prepared.keep_mask, ALIGN_EVENT].to_numpy(dtype=np.float64)
for trial_idx in range(len(neural_trials)):
    session_input.append(inp)
    session_output.append(out)
```

iii. The notes describe one common stimulus-aligned grid and preserving the original trial sequence before filtering.

## 1-e. How are trials filtered based on quality controls?

i. Trials require six nonmissing fields, reaction time 0.08–2 s, response within 10 s of go cue, and nonzero choice. Wheel and whisker streams must fully cover the window without NaNs; sessions need a good unit and at least two surviving trials.

ii.
```python
mask = (~trials["stimOn_times"].isnull() & ...
        & (rt >= 0.08) & (rt <= 2.0)
        & ((trials["feedback_times"] - trials["goCue_times"]) <= 10.0)
        & (trials["choice"] != 0))
keep_mask = trial_mask & wheel_mask & whisker_mask
```

iii. The notes attribute the missing-event and RT mask to the paper and the no-choice and 10-second cuts to the supplied reference-code path; complete behavior coverage avoids fabricated values.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data use each probe's `spikes.times`, `spikes.clusters`, cluster QC `label`, cluster-to-channel indices, and channel CCF region IDs.

ii.
```python
metrics_path = resolve_latest(probe_dir, "**/clusters.metrics.pqt")
spikes_times_path = resolve_latest(probe_dir, "**/spikes.times.npy")
spikes_clusters_path = resolve_latest(probe_dir, "**/spikes.clusters.npy")
```

iii. The notes say these are the ALF-native equivalents needed for good-unit selection, spike binning, and Beryl region mapping.

## 2-b. How is the `neural` data processed?

i. Good clusters from all release probes are renumbered and merged, spikes are time-sorted, and per-trial spikes are counted into 100 20-ms bins. Counts are stored as `float16`; they are not divided by bin width.

ii.
```python
flat = spike_clusters[i0:i1][valid] * n_bins + bin_idx[valid]
counts = np.bincount(flat, minlength=n_units * n_bins).reshape(n_units, n_bins)
out.append(counts.astype(np.float16))
```

iii. The notes describe stimulus-aligned spike counts and say float16 was adopted to make the dense full-release pickle tractable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `metrics.label >= 1` are retained. Unlike the reference, there is no explicit exclusion of Beryl `void`; invalid channel indices become region ID zero and are mapped along with other units.

ii.
```python
good_rows = metrics["label"].to_numpy(copy=False) >= 1
region_ids = np.zeros(cluster_channels.shape[0], dtype=np.int64)
cluster_regions = brain_regions.id2acronym(region_ids, mapping="Beryl").astype(str)
```

iii. The notes emphasize that `label >= 1` exactly reproduces the paper's 75,708 well-isolated-neuron statistic. They do not justify retaining `void` units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to `stimOn_times` over `[-0.5, 1.5)` seconds. Absolute spike windows are found with `searchsorted`, and bin indices are measured from each window start.

ii.
```python
interval_begs = align_times + time_window[0]
start_idx = np.searchsorted(spike_times, interval_begs, side="left")
rel = spike_times[i0:i1] - interval_begs[i]
bin_idx = np.floor(rel / binsize).astype(np.int64)
```

iii. The notes identify stimulus onset as an explicit task override where the methods paper uses other events for some outputs.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural stream is rebinned to nonoverlapping 20-ms bins, producing 100 bins across two seconds. No smoothing is applied.

ii.
```python
BINSIZE = 0.02
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
bin_idx = np.floor(rel / binsize).astype(np.int64)
```

iii. The notes cite the method paper's 20-ms dynamic-behavior/choice grid and the common-grid requirement.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is constructed from the fixed window and bin size associated with raw `stimOn_times`, rather than read as a raw variable.

ii.
```python
ALIGN_EVENT = "stimOn_times"
COMMON_RELATIVE_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32)
```

iii. The notes say a common relative-time vector serves every trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. One hundred equally spaced values from -0.48 through 1.50 seconds are generated; these are right bin edges, not bin centers.

ii.
```python
np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32)
```

iii. The agent describes this as the common 20-ms relative-time grid but does not discuss the right-edge versus center choice.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The vector has the same 100 columns as neural data, but labels each neural interval by its right edge: neural bin 0 covers [-0.50,-0.48), while input column 0 is -0.48.

ii.
```python
inp = np.vstack([COMMON_RELATIVE_TIMES,
                 np.full(NBINS, prepared.trial_number_in_block[trial_idx], dtype=np.float32)])
```

iii. The notes claim the streams use a single common grid; they do not identify this half-bin difference from the reference's bin centers.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from sequential changes in the unfiltered trials table's `probabilityLeft`.

ii.
```python
trial_number_in_block = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy())
```

iii. The notes explain that the trials table lacks a block identifier, so prior changes define block boundaries.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A Python loop resets the counter to 1 whenever the prior changes and increments otherwise. It is computed before filtering and repeated over all time bins.

ii.
```python
if i == 0 or current != prev:
    counter = 1
else:
    counter += 1
out[i] = counter
```

iii. Computing before filtering is justified as preserving the animal's true block position. The agent chose one-based indexing without a specific justification.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from the trials-table `choice` column after trial filtering.

ii.
```python
choice_raw=trials.loc[keep_mask, "choice"].to_numpy()
```

iii. The notes state that IBL choice coding was checked against high-contrast correct trials.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Raw +1 is mapped to left/0 and -1 to right/1, then the per-trial class is repeated across 100 bins.

ii.
```python
mapped[raw_choice == 1] = 0
mapped[raw_choice == -1] = 1
np.full(NBINS, choice[trial_idx], dtype=np.int8)
```

iii. This follows the requested category mapping; no-choice trials are removed first.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from the retained trials' `probabilityLeft` column.

ii.
```python
prior_raw=trials.loc[keep_mask, "probabilityLeft"].to_numpy()
```

iii. The notes identify it as the per-trial block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values are rounded to one decimal and mapped 0.2→0, 0.5→1, 0.8→2, then repeated across time.

ii.
```python
mapper = {0.2: 0, 0.5: 1, 0.8: 2}
key = round(float(val), 1)
out[i] = mapper[key]
```

iii. The mapping is explicitly required by the task; repetition supplies a uniform output matrix.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
timestamps = np.load(resolve_latest(alf_path, "**/_ibl_wheel.timestamps.npy"))
position = np.load(resolve_latest(alf_path, "**/_ibl_wheel.position.npy"))
```

iii. The notes call this the SessionLoader-equivalent raw ALF path.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is interpolated to 1 kHz, filtered/differentiated with the IBL helper (20-Hz corner, order 8), converted to absolute velocity, linearly interpolated per trial, then categorized.

ii.
```python
interp_pos, interp_times = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
y_interp = np.interp(x_interp, curr_times, curr_vals)
```

iii. The notes say this preserves the reference wheel processing while direct-loading ALF files.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Two robust tertile edges are computed over every retained wheel timepoint from every prepared session, and shared globally. `np.digitize` returns 0/1/2; degenerate thresholds collapse to zero.

ii.
```python
all_wheel = np.concatenate([ps.wheel_cont.reshape(-1) for ps in prepared_sessions])
wheel_edges = robust_tertile_edges(all_wheel)
return np.digitize(values, bins=np.array([low, high]), right=False).astype(np.int8)
```

iii. The notes argue global thresholds make labels comparable across sessions; this deliberately differs from session-specific reference percentiles.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel is interpolated at 100 absolute times from onset-0.48 through onset+1.50, whereas neural bins cover intervals beginning at onset-0.50; thus samples are at neural-bin right edges rather than centers.

ii.
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = np.interp(x_interp, curr_times, curr_vals)
```

iii. The agent intended a shared stimulus-onset grid and complete coverage, but did not justify choosing right edges.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses `<view>Camera.ROIMotionEnergy.npy` and `_ibl_<view>Camera.times.npy`, preferring left and falling back to right.

ii.
```python
for view in ("left", "right"):
    me_candidates = sorted(alf_path.glob(f"**/{view}Camera.ROIMotionEnergy.npy"))
    ts_candidates = sorted(alf_path.glob(f"**/_ibl_{view}Camera.times.npy"))
```

iii. The notes say this follows supplied code and maximizes session retention.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy values are otherwise unchanged. If timestamps are longer, leading timestamps are removed; the trace is linearly interpolated per trial and categorized.

ii.
```python
if video_timestamps.shape[0] > video_data.shape[0]:
    video_timestamps = video_timestamps[-video_data.shape[0]:]
y_interp = np.interp(x_interp, curr_times, curr_vals)
```

iii. The notes identify timestamp trimming as reference-style repair and specify no extra normalization/filtering.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Global tertile edges over all retained whisker samples are computed and shared across sessions; digitization yields 0/1/2.

ii.
```python
all_whisker = np.concatenate([ps.whisker_cont.reshape(-1) for ps in prepared_sessions])
whisker_edges = robust_tertile_edges(all_whisker)
```

iii. The notes justify global bins as consistent category semantics across sessions, knowingly differing from session-level reference thresholds.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is sampled on the same right-edge grid as wheel (-0.48 to +1.50 relative to stimulus), giving 100 columns but not the neural-bin centers used by the reference.

ii.
```python
whisker_interp, whisker_mask = interpolate_behavior_per_trial(whisker_times, whisker_motion, align_times)
```

iii. The notes intend common stimulus-onset alignment and reject padding when coverage is incomplete.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing files exclude sessions; missing trial fields, behavior NaNs, or incomplete stream coverage exclude trials. Excess camera timestamps are trimmed; shorter/empty timestamp streams error. Invalid spike cluster indices are masked, invalid channels receive region ID zero, and fewer than two valid trials or zero good units excludes a session.

ii.
```python
except FileNotFoundError:
    return spec.eid, None, "missing_required_stream"
valid_spikes = spikes_clusters < n_clusters
if keep_mask.sum() < 2:
    return None
```

iii. The notes favor dropping unusable observations rather than padding/fabricating data and document the camera repair as matching reference behavior.

## 10-a. What are the most time-consuming steps of the code?

i. The notes identify pass-2 spike-stream loading, copying/merging/sorting, and dense per-trial spike binning/storage as the main bottleneck; recursive revision discovery and behavior preparation add overhead.

ii.
```python
spikes_times = np.load(spikes_times_path, mmap_mode="r")
merged_spike_times = np.concatenate(spike_times_all)
order = np.argsort(merged_spike_times)
```

iii. Full-run profiling reportedly showed spike reload and dtype copies dominated, motivating memory mapping, float16, and session threading.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial-wise behavior interpolation, trial-wise spike binning, output assembly, prior mapping, block counting, and channel-to-region mapping use Python loops/list comprehensions and could be further vectorized, though spike counting inside each trial already uses `bincount`.

ii.
```python
for i in range(len(align_times)):
    y_interp = np.interp(x_interp, curr_times, curr_vals)
for trial_idx in range(len(neural_trials)):
    session_input.append(inp)
```

iii. The notes claim spike binning is vectorized via `searchsorted`/`bincount`; more precisely, only its per-trial inner counting is vectorized.

## 10-c. What processing does the code repeat multiple times?

i. It recursively resolves revisioned paths repeatedly; reads cluster metrics once to count good units and again while loading spikes; constructs identical time rows and constant arrays per trial; and creates `BrainRegions` per session payload.

ii.
```python
n_good_units = count_good_units(spec)
# later:
metrics = pd.read_parquet(metrics_path, columns=["label"])
brain_regions = BrainRegions()
```

iii. The notes acknowledge repeated recursive path discovery and a two-pass design; the second pass is required for global thresholds, but repeated QC/path work was not eliminated.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Pass 1 computes/stores continuous behavior only to derive global edges and later categorical outputs; good-unit metrics are counted then reread; spikes are globally sorted after probe merging; static inputs/outputs are redundantly expanded to 100 columns; plotting-only work is optional.

ii.
```python
wheel_cont=np.stack(...)
all_wheel = np.concatenate([ps.wheel_cont.reshape(-1) for ps in prepared_sessions])
np.full(NBINS, choice[trial_idx], dtype=np.int8)
```

iii. The notes defend the two-pass behavior storage as necessary for global bins and the repeated static columns as shape consistency, but downstream decoding does not need independent copies of invariant values.
