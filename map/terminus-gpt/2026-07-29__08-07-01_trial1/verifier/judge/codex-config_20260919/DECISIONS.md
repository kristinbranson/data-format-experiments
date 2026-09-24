# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively discovers every NWB file below `data`, sorts the paths, and opens each file once with `h5py`. Within each file it reads the trial table, units, behavioral events, and tongue tracking arrays. `--sample` limits processing to the first two files; otherwise all files are used.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
files = choose_sessions(files, sample=args.sample)
...
for fp in files:
    sess = load_session(fp, brain_regions)
```
```python
with h5py.File(nwb_path, 'r') as f:
    trials = f['intervals/trials']
    ...
    spikes_flat = f['units/spike_times'][:]
```

iii. The notes say the data are organized as per-subject directories containing NWB session files and identify the relevant NWB groups. Sorting provides deterministic order. The full run found all 174 NWB files, although the agent acknowledged that this did not match the paper's 173 analyzed sessions.

## 1-b. How are the data split into subjects?

i. Subject identity is inferred from the parent directory name (for example, `sub-440956`), rather than read from the NWB subject object. A first-seen-order subject list and map are built while sessions are appended; `subject_idx` records the corresponding index for each retained session.

ii.
```python
def get_subject_id(nwb_path):
    return nwb_path.parent.name
```
```python
if subj not in subject_map:
    subject_map[subj] = len(subjects)
    subjects.append(subj)
data['subject_idx'].append(subject_map[subj])
```

iii. The notes describe the per-subject directory organization and report 28 subjects. No explicit justification is given for preferring the directory label over `nwb.subject.subject_id`.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Its filename stem is used as the session identifier, and one session-level entry is appended to each output list if at least two trials survive.

ii.
```python
def get_session_id(nwb_path):
    return nwb_path.stem
```
```python
if len(sess_neural) < 2:
    return None
...
data['neural'].append(sess['neural'])
```

iii. The agent recognized the one-file-per-session NWB layout. It retained 174 sessions and explicitly noted the discrepancy with the paper/reference count of 173, but did not resolve it.

## 1-d. How are the data split into trials?

i. Trial rows and go-cue events are paired by their array index up to `min(number of trial rows, number of go cues)`. A trial is retained only if it has a finite go cue, the complete requested window lies within the trial table's start/stop bounds, categorical fields are recognized, and a sample event can be found within that trial before the go cue.

ii.
```python
n_match = min(n_trials, len(go_times))
for i in range(n_match):
    go = go_times[i]
    if not np.isfinite(go):
        continue
    if go + T_START < trial_start[i] or go + T_END > trial_stop[i]:
        continue
```

iii. The notes say to keep trials with a valid go cue and sufficient aligned data. They also record a correction from treating the global sample-event stream as trial-indexed to selecting the last sample event within each trial.

## 1-e. How are trials filtered based on quality controls?

i. The implemented filters are finite go cue, full window inside trial boundaries, recognized choice/outcome/early-lick strings, an available sample event, and at least two retained trials per session. Although `auto_water` and `free_water` are loaded and the notes discuss excluding them, the code never uses either field. It also does not use `units/obs_intervals` to remove behavior-only trials lacking spikes.

ii.
```python
auto_water = trials['auto_water'][:]
free_water = trials['free_water'][:]
...
if go + T_START < trial_start[i] or go + T_END > trial_stop[i]:
    continue
if choice[i] not in CHOICE_MAP or outcome[i] not in OUTCOME_MAP or early[i] not in EARLY_MAP:
    continue
...
if not np.isfinite(sample_time):
    continue
```

iii. The planning notes say regular trials exclude early lick, auto water, free water, and no-response trials, but also note that decoder outputs require early-lick and ignore trials. The final code keeps those required categories, but silently fails to implement the planned free/auto-water filtering and does not discover the reference solution's `obs_intervals` criterion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from the ragged `units/spike_times` arrays and each trial's `go_start_times`. Unit selection additionally uses `units/unit_quality`, `units/electrodes`, and electrode `location` strings.

ii.
```python
spikes_flat = f['units/spike_times'][:]
spikes_index = f['units/spike_times_index'][:]
...
go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```

iii. The notes identify spike times and go-cue alignment as the neural source and state that curated good units should be used.

## 2-b. How is the `neural` data processed?

i. For every retained trial and unit, spikes in the four-second go-centered window are assigned to 80 non-overlapping bins with `floor`, counted with `bincount`, and divided by 0.05 s to produce unsmoothed firing rates in Hz.

ii.
```python
rel = spike_times[a:b] - lo
bins = np.floor(rel / BIN_SIZE).astype(np.int64)
counts = np.bincount(bins, minlength=N_BINS).astype(np.float32)
return counts / BIN_SIZE
```

iii. The agent planned 50 ms go-cue-aligned spike binning and optimized its initial implementation with `searchsorted` plus `bincount`, reporting a sample runtime reduction from about 30.6 s to 11.0 s.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are retained when the older `unit_quality` field equals `good` and their electrode's broad JSON region label belongs to ten hand-selected left/right ALM, Striatum, Thalamus, Midbrain, or Medulla regions. The newer classifier verdict in `units/classification` is not used. Sessions with zero retained units are not explicitly rejected.

ii.
```python
unit_quality = decode_arr(f['units/unit_quality'][:])
...
keep_units = (unit_quality == 'good') & np.isin(unit_regions, list(MAJOR_REGIONS))
kept_idx = np.where(keep_units)[0]
```

iii. The notes acknowledge that `unit_quality == "good"` gives far too many units, then justify restricting to major broad regions because that was thought likely to reproduce the paper total. The full result still had 142,233 units versus 69,943 reported, and the agent documented the mismatch without changing the rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute trial go-cue time is added to the fixed relative start/end offsets. `searchsorted` crops each unit's absolute spike times to that interval, and subtracting `go + T_START` places spikes on the common relative grid.

ii.
```python
lo = go_time + T_START
hi = go_time + T_END
a = np.searchsorted(spike_times, lo, side='left')
b = np.searchsorted(spike_times, hi, side='left')
rel = spike_times[a:b] - lo
```

iii. The notes repeatedly identify go-cue-centered alignment as required by both the instructions and reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50 ms. Raw point-process spike times are rebinned into 80 non-overlapping bins spanning -2.5 to +1.5 seconds around the go cue; no smoothing or further rebinning is performed.

ii.
```python
BIN_SIZE = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int(round((T_END - T_START) / BIN_SIZE))
```

iii. These values are taken directly from the decoder task instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times`, the trial's `start_time`/`stop_time`, and its go-cue time. The last sample event inside the trial and no later than the go cue is selected as tone onset.

ii.
```python
sample_time = find_last_event_within_trial(
    sample_event_times, trial_start[i], trial_stop[i], before_time=go)
```

iii. The notes explain that `sample_start_times` is a global event stream and that per-trial matching was fixed by selecting the last sample event within the trial before go.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Tone onset is expressed relative to go, subtracted from every bin center, and all negative values (bins before tone onset) are clipped to zero.

ii.
```python
tone_rel = sample_time - go_time
rel = BIN_CENTERS - tone_rel
rel[rel < 0] = 0.0
return rel.astype(np.float32)
```

iii. The notes say the range was “corrected” after fixing event matching, but do not justify clipping pre-tone times. The documented sample range starts at 0.0, showing this was intentional or at least accepted.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the same 80 go-cue-relative bin centers used for the neural firing-rate bins. Thus element `k` corresponds to neural bin `k`, although pre-tone centers are all collapsed to zero.

ii.
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
...
rel = BIN_CENTERS - tone_rel
```

iii. The agent's mapping plan says every modality should use the common go-centered 50 ms grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from the session-global `BehavioralEvents/photostim_start_times` and `photostim_stop_times`, restricted to events inside each trial using trial start/stop times, then related to the trial's go cue.

ii.
```python
photo_starts = f['acquisition/BehavioralEvents/photostim_start_times/timestamps'][:]
photo_stops = f['acquisition/BehavioralEvents/photostim_stop_times/timestamps'][:]
...
ps = photo_starts[(photo_starts >= trial_start[i]) & (photo_starts <= trial_stop[i])]
```

iii. The notes considered both event streams and trial fields, and chose event timestamps because they directly provide stimulation starts/stops on the session clock.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Starts and stops are paired by position after truncating to the shorter list. For each interval, every 50 ms bin having any temporal overlap with the interval is set to 1; all other bins are 0.

ii.
```python
m = min(len(ps), len(pe))
inp1 = build_photostim_series(ps[:m], pe[:m], go)
```
```python
overlap = (BIN_EDGES[:-1] < re) & (BIN_EDGES[1:] > rs)
x[overlap] = 1.0
```

iii. The notes say to build a binary time series and verify that stimulation ends before go. They do not justify the “any overlap” convention or silent truncation of unmatched event arrays.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation event times are converted to offsets from the same go cue as the neural window, then compared with the same relative bin edges.

ii.
```python
rs = s - go_time
re = e - go_time
overlap = (BIN_EDGES[:-1] < re) & (BIN_EDGES[1:] > rs)
```

iii. The notes identify go-cue alignment and report a raw-to-converted reconstruction check that passed.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The agent uses only `trials/trial_instruction` and treats the instructed side as the animal's actual lick choice. Loaded left/right lick timestamps and trial outcome are not used to correct misses or identify no-lick trials.

ii.
```python
choice = decode_arr(trials['trial_instruction'][:])
...
np.full(N_BINS, CHOICE_MAP[choice[i]], dtype=np.int64)
```

iii. The mapping plan explicitly equates `trial_instruction` with choice and says “Map left=0, right=1.” No justification addresses the distinction between instructed and chosen side.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Strings `left` and `right` are mapped to 0 and 1 and repeated over all 80 time bins. There is no `no lick` category; ignored trials are still labeled with the instructed side.

ii.
```python
CHOICE_MAP = {'left': 0, 'right': 1}
...
'output_values': [
    ['left', 'right'],
```

iii. The agent states that trial-level outputs matched the raw trial table, but that sanity check merely reproduced `trial_instruction`; it did not establish that this field represented actual choice.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is read directly from the trial table's `outcome` field.

ii.
```python
outcome = decode_arr(trials['outcome'][:])
```

iii. The notes correctly identify the field as already containing the requested categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. `ignore`, `miss`, and `hit` are mapped to 0, 1, and 2, respectively, and the per-trial value is repeated across all 80 bins.

ii.
```python
OUTCOME_MAP = {'ignore': 0, 'miss': 1, 'hit': 2}
...
np.full(N_BINS, OUTCOME_MAP[outcome[i]], dtype=np.int64)
```

iii. This follows the requested category ordering; the notes report a direct raw-to-converted check.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It is read directly from the trial table's `early_lick` field.

ii.
```python
early = decode_arr(trials['early_lick'][:])
```

iii. The mapping plan identifies this explicit trial-table flag, and the agent deliberately keeps early-lick trials because early lick is a requested decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` maps to 0 and `early` maps to 1; the value is repeated across all time bins.

ii.
```python
EARLY_MAP = {'no early': 0, 'early': 1}
...
np.full(N_BINS, EARLY_MAP[early[i]], dtype=np.int64)
```

iii. This matches the requested no/yes coding and is included in the agent's raw-output sanity check.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses column 1 of `Camera0_side_TongueTracking/data` as y-position and the matching camera timestamps. It reads but ignores neither explicitly nor indirectly the third tracking-likelihood column, so visibility is not determined.

ii.
```python
tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_t = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_y_all = tongue[:, 1].astype(np.float32)
```

iii. The notes identify tongue y and mention a need to handle low-likelihood/missing tracking, but the implementation never completes that part.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each trial, all session camera y-values are linearly interpolated at the 80 bin centers relative to go. Values outside the camera timestamp range become NaN. There is no likelihood filtering and no averaging of frames within a 50 ms bin.

ii.
```python
return np.interp(BIN_CENTERS, rel_t[valid], track_y[valid],
                 left=np.nan, right=np.nan).astype(np.float32)
```

iii. The notes planned interpolation/alignment and percentile discretization, but explicitly left low-likelihood/missing handling unresolved. They later reported that selected aligned values matched their own interpolation reconstruction.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The 40th and 60th percentiles are computed from all finite interpolated values in retained trial windows for the session. Values below q40 become 0, values above q60 become 2, and everything else—including NaNs—is initialized to category 1. Only three labels (`low`, `mid`, `high`) exist; the required category 3, `not visible`, is absent.

ii.
```python
q40, q60 = np.percentile(all_tongue, [40, 60])
disc = np.full(N_BINS, 1, dtype=np.int64)
disc[ty < q40] = 0
disc[ty > q60] = 2
```

iii. The agent followed the requested per-session 40th/60th percentiles but failed to implement its own note to handle low-likelihood or missing tracking, and the validation did not flag the missing fourth class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are converted to time relative to each trial's go cue, and y is interpolated at the neural bin centers. This gives 80 index-aligned samples, but each is a point interpolation rather than a mean over the corresponding neural bin interval.

ii.
```python
rel_t = track_t - go_time
...
np.interp(BIN_CENTERS, rel_t[valid], track_y[valid], ...)
```

iii. The notes cite reference go-cue marker alignment and specify a shared time base. They do not discuss the mismatch between center interpolation and interval-based neural binning.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Byte/string decoding ignores invalid UTF-8; malformed region JSON falls back to the original string. Nonfinite go cues, incomplete trial windows, unknown categories, and trials without a sample event are skipped. Sessions with fewer than two trials are dropped. Tracking with fewer than two finite samples returns NaNs, but those NaNs later become the middle tongue class. If a whole session has no finite tongue samples, synthetic `[0, 1]` values define thresholds. Event start/stop count mismatches are silently truncated. There is no handling of the unclassified session or spike observation intervals.

ii.
```python
except Exception:
    regs.append(s)
...
m = min(len(ps), len(pe))
...
else np.array([0,1], dtype=np.float32)
```

iii. The notes document several sanity checks and acknowledge unresolved count discrepancies. Missing tongue visibility and unmatched photostimulation events were not critically reviewed.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant implemented work is nested trial-by-unit spike binning: each trial performs `searchsorted` and `bincount` for every retained unit. Repeated full-camera validity/interpolation work per trial, NWB array I/O, accumulation of a 20 GB pickle, and serialization are also expensive. The full conversion took about 1,423 seconds.

ii.
```python
for i in range(n_match):
    ...
    neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
```

iii. The notes identify spike binning as initially slow and report an optimization, but the full conversion still took about 23.7 minutes, above the 15-minute target. They also recognize the 20 GB output as memory-heavy.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-by-unit neural loop could vectorize the trial dimension by searching one flattened array of all trial edges per unit, as in the reference. The per-trial tongue interpolation repeatedly builds the same session-wide validity mask and relative time array and could instead assign frames to all go-centered bins in a more batched operation. Region-index lookup also uses repeated linear list searches, though this is minor.

ii.
```python
neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
...
for r in unit_regions[kept_idx]:
    if r not in brain_regions:
        brain_regions.append(r)
    region_idx.append(brain_regions.index(r))
```

iii. The notes mention the `searchsorted`/`bincount` speedup but do not identify the remaining opportunity to process all trials' edges at once for each unit.

## 10-c. What processing does the code repeat multiple times?

i. For every trial, `interp_tracking_to_bins` recomputes `track_t - go_time`, scans the entire session arrays for finite values, and invokes interpolation. Spike searches are also repeated separately for every trial/unit rather than searching all trial edges once per unit. Photostimulation streams are scanned per trial.

ii.
```python
def interp_tracking_to_bins(track_t, track_y, go_time):
    rel_t = track_t - go_time
    valid = np.isfinite(track_y) & np.isfinite(rel_t)
```
```python
ps = photo_starts[(photo_starts >= trial_start[i]) & ...]
```

iii. The notes do not discuss this repeated processing; they only record the earlier spike-binning optimization.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads `left_lick_times`, `right_lick_times`, `auto_water`, and `free_water` but never uses them. It computes and returns q40/q60 and a session id that are not assembled into the output metadata. The accepted `--show-processing` flag is never consulted. Region parsing is necessary for the chosen QC rule, but that rule itself is unsupported by the reference.

ii.
```python
left_lick = f['acquisition/BehavioralEvents/left_lick_times/timestamps'][:]
right_lick = f['acquisition/BehavioralEvents/right_lick_times/timestamps'][:]
auto_water = trials['auto_water'][:]
free_water = trials['free_water'][:]
```

iii. The notes admit that `--show-processing` does not emit plots. They do not mention the unused lick/water arrays; notably, those arrays could have exposed or fixed important choice and trial-curation errors.
