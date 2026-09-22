# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the paper freeze CSV, groups its rows by session EID, constructs each session's local ONE-cache path, and directly loads Parquet/NPY files. It selects the newest/preferred revision with recursive globbing rather than using ONE loaders. Full mode attempts all 459 freeze sessions; unusable sessions are recorded as skipped.

ii.
```python
freeze = pd.read_csv(FREEZE_CSV)
for eid, group in freeze.groupby("eid", sort=False):
    sessions.append({"eid": str(eid), ...,
                     "pids": group["pid"].astype(str).tolist(),
                     "probe_names": group["probe_name"].astype(str).tolist()})
...
path = newest_file(alf, "_ibl_trials.table.pqt", "#2025-03-03#")
trials = pd.read_parquet(path)
```

iii. The agent says this preserves the supplied 459-session/699-probe freeze identity, revisions, probe order, and works offline. It regarded direct local ALF resolution as equivalent to the reference loaders and used memory mapping for large arrays.

## 1-b. How are the data split into subjects?

i. Subject IDs come from the freeze CSV. Sessions retain their subject string; assembly creates a sorted unique subject list and maps every retained session to its subject index.

ii.
```python
"subject": str(first["subject"]),
...
subjects = sorted({r["info"]["subject"] for r in results})
subject_idx = np.asarray(
    [subject_lookup[r["info"]["subject"]] for r in results], dtype=np.int32)
```

iii. The freeze provides an explicit subject identity, so no filename inference is needed. The notes report 136 subjects among the 444 retained sessions.

## 1-c. How are the data split into sessions?

i. Rows of the freeze are grouped by `eid` without sorting, producing one session record with all probe IDs/names. Results are restored to freeze order after parallel processing.

ii.
```python
for eid, group in freeze.groupby("eid", sort=False):
    ...
results = sorted(results, key=lambda r: r["index"])
```

iii. The agent identifies EID as the native unique session key and preserves paper-freeze order for reproducibility.

## 1-d. How are the data split into trials?

i. Each row of `_ibl_trials.table.pqt` is treated as one trial. Retained row indices identify trials; stimulus-aligned windows are made from each row's `stimOn_times`.

ii.
```python
code_indices = np.flatnonzero(code_mask)
source_indices = code_indices[keep_in_code]
stim_times = trials["stimOn_times"].to_numpy(dtype=float)[source_indices]
starts = stim_times + OFF_START
ends = stim_times + OFF_END
```

iii. The trial table already defines trials. Source indices are saved so converted trials can be audited against raw rows.

## 1-e. How are trials filtered based on quality controls?

i. Trials must have finite stimulus, choice, feedback, prior, first movement and feedback type; reaction time must be 0.08–2.0 s inclusive; choice must be nonzero; go-cue-to-feedback duration must not exceed 10 s. They must also have valid wheel, camera, and common-across-probes neural coverage. Sessions with fewer than two survivors are skipped.

ii.
```python
for column in required:
    good &= trials[column].notna().to_numpy()
good &= reaction_time >= 0.08
good &= reaction_time <= 2.0
good &= trials["choice"].to_numpy() != 0
good &= ~(duration > 10.0)
...
stream_good = wheel_good & motion_good & neural_good
```

iii. The notes say the first mask reproduces `load_trials_and_mask(..., max_trial_len=10)`. Stream coverage prevents extrapolated/missing targets, and neural coverage was added after validation found three all-zero trials occurring after ephys ended.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural arrays come from each probe's `spikes.times.npy` and `spikes.clusters.npy`. `clusters.metrics.pqt`, `clusters.channels.npy`, and channel atlas IDs determine cluster ordering, QC counts, and Beryl region labels, although QC labels are not used to remove neurons.

ii.
```python
spike_times = np.load(probe["spikes_times_path"], mmap_mode="r")
spike_clusters = np.load(probe["spikes_clusters_path"], mmap_mode="r")
...
metrics = pd.read_parquet(metrics_path)
cluster_channels = np.load(channels_path)
channel_atlas_ids = np.load(atlas_path)
```

iii. The agent chose the common cache representation used by the method code: all sorted clusters with spike time and cluster identity, while retaining anatomy for metadata.

## 2-b. How is the `neural` data processed?

i. For every probe and trial, spikes are assigned to 100 half-open 20-ms bins using floor indices and counted with `np.bincount`. Probe populations are concatenated. Values remain raw float32 spike counts; they are not divided by bin width or smoothed.

ii.
```python
bins = np.floor((times - starts[trial]) / BIN_SIZE).astype(np.int64)
code = ((local_trial * n_clusters + clusters) * N_BINS + bins).astype(np.int64)
counts = np.bincount(flat, minlength=...).reshape(...)
output[first:last] = counts
```

iii. The agent states that `code_zhang2025` uses half-open 20-ms spike-count bins and no smoothing. Chunked bincounting was chosen to bound temporary memory.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not filtered: every Kilosort cluster is retained, including non-good labels and `void`/`root` regions. The code computes how many labels pass `label >= 1`, but uses that only for diagnostics.

ii.
```python
"n_clusters": n_clusters,
"n_good": int((metrics["label"].to_numpy() >= 1).sum()),
...
n_neurons = sum(p["n_clusters"] for p in probes)
```

iii. The agent explicitly chose `qc=None` to match the decoder reference code, distinguishing its 621,733 all-cluster population from the paper's 75,708 stringent-QC neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`, using the absolute interval `[stimulus onset - 0.5, stimulus onset + 1.5)`. Spike bin zero starts at -0.5 s.

ii.
```python
stim_times = trials["stimOn_times"].to_numpy(dtype=float)[source_indices]
starts = stim_times + OFF_START
ends = stim_times + OFF_END
```

iii. The agent says this exactly follows the reference's stimulus-onset event and two-second decoding window on synchronized session clocks.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 20 ms: 100 bins over two seconds. Spikes are newly binned into those intervals; there is no smoothing or subsequent temporal rebinning.

ii.
```python
BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = 100
```

iii. The papers/reference code use 20-ms bins, and metadata reports `time_bin_size: 20.0` ms.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is defined from `stimOn_times` plus the fixed window/bin grid, rather than calculated from another measured stream.

ii.
```python
REL_SAMPLE_TIMES = np.linspace(
    OFF_START + BIN_SIZE, OFF_END, N_BINS).astype(np.float32)
```

iii. The agent interprets the reference behavior grid as bin-right-edge times from -0.48 through +1.50 s.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A 100-value evenly spaced right-edge grid is created once and broadcast unchanged to every retained trial.

ii.
```python
time_input = np.broadcast_to(REL_SAMPLE_TIMES, (len(source_indices), N_BINS))
inputs = np.stack((time_input, block_input), axis=1).astype(np.float32, copy=True)
```

iii. This avoids recomputing an identical relative time vector and follows the agent's claimed right-edge convention.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Each value labels the right edge of the corresponding half-open neural bin: -0.48 labels `[-0.50,-0.48)`, and +1.50 labels the last bin.

ii.
```python
"behavior_sample_convention": "right edge of each neural time bin",
```

iii. The notes explicitly say behavior samples label the preceding spike-count bin end.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from contiguous runs of the trials-table `probabilityLeft` column.

ii.
```python
raw_block_numbers = trial_number_in_block(
    trials["probabilityLeft"].to_numpy())
```

iii. The trials table has no explicit block ID, while prior probability is constant within each block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A prior change marks a block start; the zero-based distance from the most recent start is computed before filtering. Retained values are then broadcast across time.

ii.
```python
changes[1:] = ~np.isclose(probability_left[1:], probability_left[:-1], ...)
starts = np.maximum.accumulate(np.where(changes, np.arange(len(probability_left)), 0))
return (np.arange(len(probability_left)) - starts).astype(np.float32)
```

iii. Computing before filtering preserves the animal's true experimental position even when intervening trials are removed.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes directly from the trials-table `choice` field.

ii.
```python
raw_choice = trials["choice"].to_numpy()[source_indices]
```

iii. IBL already records the behavioral choice per trial; zero/no-choice trials are excluded.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Native +1 (left) becomes 0 and -1 (right) becomes 1. The categorical value is repeated across all 100 time bins.

ii.
```python
choices = (raw_choice == -1).astype(np.int8)
choice_output = np.broadcast_to(choices[:, None], (len(source_indices), N_BINS))
```

iii. This implements the requested left=0/right=1 mapping while fitting the common time-varying output shape.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from the trials-table `probabilityLeft` field.

ii.
```python
raw_prior = trials["probabilityLeft"].to_numpy(dtype=float)[source_indices]
```

iii. This is the block prior requested by the decoder task.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 are tolerance-matched to categories 0, 1, and 2 and repeated across time.

ii.
```python
for value, category in ((0.2, 0), (0.5, 1), (0.8, 2)):
    priors[np.isclose(raw_prior, value, rtol=0.0, atol=1e-8)] = category
prior_output = np.broadcast_to(priors[:, None], (len(source_indices), N_BINS))
```

iii. This is the exact categorical mapping required by the instructions.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
raw_times = np.load(timestamp_path, mmap_mode="r")
raw_position = np.load(position_path, mmap_mode="r")
```

iii. These are the released wheel time and angular-position streams used by `SessionLoader`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is interpolated to 1 kHz, velocity is computed with a 20-Hz order-8 filtered derivative, absolute velocity gives speed, and linear interpolation samples it at the 100 right-edge times. Nonfinite samples, if present, are median-imputed before categorization.

ii.
```python
position, times = interpolate_position(raw_times, raw_position, freq=1000)
velocity, _ = velocity_filtered(position, fs=1000, corner_frequency=20, order=8)
speed = np.abs(velocity)
sampled[trial] = _linear_interp_extrapolate(tx, vy, query).astype(np.float32)
```

iii. The agent intended to reproduce `SessionLoader` and reference behavior interpolation exactly; imputation handles isolated allowed NaNs rather than dropping an otherwise valid session.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Within each retained session, all finite trial-time samples jointly define empirical 1/3 and 2/3 quantiles. `np.digitize` maps values to low/medium/high (0/1/2).

ii.
```python
thresholds = np.quantile(values[finite], [1 / 3, 2 / 3])
clean = np.where(finite, values, median)
categories = np.digitize(clean, thresholds, right=False).astype(np.int8)
```

iii. Session tertiles balance classes while preserving within-session movement rank; discretization is required by the task.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Speed is queried at `stimOn_times + REL_SAMPLE_TIMES`; each query is the right edge of its corresponding neural count bin.

ii.
```python
query = stim_times[trial] + REL_SAMPLE_TIMES.astype(np.float64)
```

iii. Both streams share the session clock; the agent uses the reference's asserted right-edge sampling convention.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses `leftCamera.ROIMotionEnergy.npy` and `_ibl_leftCamera.times.npy`, falling back to the corresponding right-camera files. If timestamps are longer, their leading excess is trimmed by retaining the last `len(values)` entries.

ii.
```python
for view in ("left", "right"):
    value_path = newest_file(alf, f"{view}Camera.ROIMotionEnergy.npy")
    time_path = newest_file(alf, f"_ibl_{view}Camera.times.npy")
...
if len(times) > len(values):
    times = times[-len(values):]
```

iii. Left preference/right fallback follows the reference logic and permits sessions with only one usable side view.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy values are otherwise unfiltered and unnormalized. They are linearly interpolated at right-edge trial times; isolated nonfinite values are median-imputed before categorization.

ii.
```python
sampled, good = sample_behavior(times, values, stim_times)
...
clean = np.where(finite, values, median)
```

iii. The agent says the released ROI motion energy should be used directly and only aligned/discretized as required downstream.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The same session-wide finite-sample 1/3 and 2/3 quantiles and `np.digitize` rule used for wheel speed create low/medium/high categories.

ii.
```python
motion_categories, motion_thresholds, motion_imputed, motion_median = \
    discretize_tertiles(motion_values)
```

iii. This provides approximately balanced within-session classes; the notes acknowledge small deviations when values tie at thresholds.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera motion energy is interpolated at `stimOn_times + REL_SAMPLE_TIMES`, so its samples label neural-bin right edges.

ii.
```python
query = stim_times[trial] + REL_SAMPLE_TIMES.astype(np.float64)
sampled[trial] = _linear_interp_extrapolate(tx, vy, query).astype(np.float32)
```

iii. Camera and ephys timestamps are treated as synchronized; coverage checks reject trial windows without nearby endpoints.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing required columns/files, malformed arrays, insufficient trials, or unusable streams cause a session to be skipped and logged. Individual uncovered trials are dropped. Camera timestamp excess is trimmed, isolated nonfinite behavior samples are median-imputed, and unexpected choices/priors raise errors. Fourteen sessions lacking motion energy and one without joint coverage were omitted.

ii.
```python
if len(times) > len(values):
    times = times[-len(values):]
...
clean = np.where(finite, values, median)
...
except Exception as exc:
    return {"ok": False, "error": f"{type(exc).__name__}: {exc}", ...}
```

iii. The agent avoids fabricating an entirely missing requested target, but considers median imputation reasonable for isolated nonfinite samples. It records exclusions and source paths in metadata for auditability.

## 10-a. What are the most time-consuming steps of the code?

i. Spike loading/binning and writing the roughly 106-GB pickle dominate. Wheel interpolation/filtering and behavior alignment are secondary. Sessions are processed concurrently.

ii.
```python
spike_times = np.load(probe["spikes_times_path"], mmap_mode="r")
...
neural[:, offset:offset + n] = bin_probe(probe, starts, ends)
```

iii. The notes estimate neural and behavior work at minutes with workers, and pickle assembly/write at 7–9 minutes; memory mapping and batched counts reduce I/O and allocation cost.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. `sample_behavior` still loops over trials, and `bin_probe` loops over trial chunks and trials within each chunk. Probe and session loops also remain. Input/output construction, block numbering, and spike counting within chunks are vectorized.

ii.
```python
for trial in range(len(stim_times)):
    ...
for first in range(0, n_trials, trials_per_chunk):
    for local_trial, trial in enumerate(range(first, last)):
```

iii. The agent says per-trial search/slicing is simple and memory-safe, while batching encoded spikes into one `bincount` captures the important speedup without multi-gigabyte temporaries.

## 10-c. What processing does the code repeat multiple times?

i. For every trial it repeatedly binary-searches behavior and spike timestamps and calls cluster-ID mapping; probe metadata creates a new `BrainRegions` object per probe. Spike files are opened once for coverage and again for binning. Plot mode also copies and revisits selected data.

ii.
```python
lo = int(np.searchsorted(spike_times, starts[trial], side="left"))
hi = int(np.searchsorted(spike_times, ends[trial], side="left"))
clusters = map_spike_clusters(raw_clusters, probe["cluster_ids"])
```

iii. The agent documents avoiding many other redundancies, but accepts repeated bounded searches/loads to keep memory use controlled and processing auditable.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads QC metrics and computes `n_good` although all clusters are retained; loads/records extensive diagnostics, timing, source paths, imputation statistics, and optional plot payloads not used by decoder training. `feedbackType` and feedback/go-cue times are used only for filtering. Continuous behavior arrays and thresholds are discarded after categorical targets (except diagnostic summaries/plots).

ii.
```python
"n_good": int((metrics["label"].to_numpy() >= 1).sum()),
...
wheel_categories, wheel_thresholds, wheel_imputed, wheel_median = \
    discretize_tertiles(wheel_values)
```

iii. These extras support validation, provenance, and plots rather than the final decoder tensors. The notes emphasize auditability and sanity checking despite the additional work.
