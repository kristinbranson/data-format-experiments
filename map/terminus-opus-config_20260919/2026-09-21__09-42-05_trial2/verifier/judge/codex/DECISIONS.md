# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the dataset by globbing all NWB files under `/app/data/sub-*/*.nwb`, sorting the paths, and processing each file as one session. Inside each file it reads the NWB contents directly with `h5py`, including the subject id, trials table, units table, behavioral events, and behavioral time series. It parallelizes session processing with a `multiprocessing.Pool`.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
...
if args.nproc > 1 and len(files) > 1:
    with Pool(min(args.nproc, len(files))) as pool:
        results = pool.map(_worker, list(zip(files, plot_flags)), chunksize=1)
```

```python
with h5py.File(path, 'r') as f:
    sess_id = f['identifier'][()].decode()
    subject = f['general/subject/subject_id'][()].decode()
    ...
    u = f['units']
    t = f['intervals/trials']
    go_all = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```

iii. The header comment and `CONVERSION_NOTES.md` Step 6 say the script converts NWB directly and aims to follow the reference preprocessing “wherever the decoder specification allows.” Step 6 also justifies multiprocessing as a wall-clock speedup for per-session conversion.

## 1-b. How are the data split into subjects?

i. The AI treats `general/subject/subject_id` from each NWB file as the subject identifier. After per-session processing it builds a sorted unique `subjects` list and a `subject_idx` array mapping each session to its subject.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
```

```python
subjects = sorted({r['subject'] for r in results})
subject_index = {s: i for i, s in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.array([subject_index[r['subject']] for r in results], dtype=np.int64),
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly maps `general/subject/subject_id` to `subjects` and `subject_idx` and states that there are 28 mice.

## 1-c. How are the data split into sessions?

i. The AI uses one NWB file as one session. Session order is the sorted file list, and each processed session carries the NWB `identifier` plus `session_start_time`.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
sess_id = f['identifier'][()].decode()
session_start = f['session_start_time'][()].decode()
...
result = {
    'session_id': sess_id,
    'subject': subject,
    'session_start': session_start,
    ...
}
```

iii. `CONVERSION_NOTES.md` Step 2 describes the DANDI layout as one NWB per session, and Step 9 reports 173 kept sessions from 174 files after unit-based filtering.

## 1-d. How are the data split into trials?

i. The AI uses the NWB trials table rows as trials and checks that the number of go cues equals the number of trial rows. It then applies a boolean `keep` mask and uses `trial_idx` to select the retained trials.

ii.
```python
t = f['intervals/trials']
trial_start = t['start_time'][:]
trial_stop = t['stop_time'][:]
...
ntrials_file = len(trial_start)

go_all = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
if len(go_all) != ntrials_file:
    print('%s: %d go cues for %d trials, skipping' % (sess_id, len(go_all), ntrials_file),
          flush=True)
    return None
```

```python
trial_idx = np.where(keep)[0]
...
go = go_all[trial_idx]
```

iii. `CONVERSION_NOTES.md` Step 4 says the DataJoint/NWB mapping for trials and go cues was verified, and the trajectory notes repeatedly state there is one go cue per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops `auto_water` and `free_water` trials, drops trials with non-finite go-cue timestamps, keeps only trials inside the `obs_intervals` coverage of every retained unit, and then removes any remaining trials whose extracted neural window has zero spikes across all selected units. It keeps early-lick, ignore, miss, and photostim trials.

ii.
```python
auto_water = t['auto_water'][:] > 0
free_water = t['free_water'][:] > 0
...
keep = ~(auto_water | free_water)
keep &= np.isfinite(go_all)
keep &= observed_trial_mask(u, unit_ids, trial_start, trial_stop)
trial_idx = np.where(keep)[0]
```

```python
has_spikes = fr.sum(axis=(0, 2)) > 0
if not has_spikes.all():
    fr = fr[:, has_spikes, :]
    trial_idx = trial_idx[has_spikes]
    go = go[has_spikes]
    edges_abs = edges_abs[has_spikes]
    centers_abs = centers_abs[has_spikes]
```

iii. The header comment and `CONVERSION_NOTES.md` Steps 5-6 justify keeping early-lick, ignore, and photostim trials because they are decoder targets or inputs. Step 6 says the `obs_intervals` filter was added after validation revealed all-zero trials in partially recorded sessions, and the trajectory records that finding explicitly.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `units/spike_times` and `units/spike_times_index`, using the go-cue timestamps to place the per-trial bin edges. Unit selection additionally depends on `units/classification`, `units/anno_name`, and `units/electrodes`.

ii.
```python
classification = u['classification'][:]
anno = _decode(u['anno_name'][:])
...
spikes = u['spike_times'][:]
s0, s1 = _spike_slices(u['spike_times_index'][:])
fr = bin_spikes(spikes, s0, s1, unit_ids, edges_abs)
```

```python
go_all = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
...
edges_abs = go[:, None] + BIN_EDGES[None, :]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `units/spike_times` to `neural` and says the conversion follows the reference `sliding_histogram` rate definition.

## 2-b. How is the `neural` data processed?

i. The AI bins absolute spike times into 80 non-overlapping 50 ms bins spanning `[-2.5, 1.5]` s around the go cue, counts spikes with `np.searchsorted`, and divides by bin width to produce firing rates in Hz. No smoothing, baseline subtraction, or normalization is applied.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_SIZE = 0.05
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2.0
```

```python
def bin_spikes(spike_times, starts, stops, unit_ids, edges_abs):
    ntrials = edges_abs.shape[0]
    flat = edges_abs.ravel()
    out = np.empty((len(unit_ids), ntrials, NBINS), dtype=np.float32)
    for i, k in enumerate(unit_ids):
        sp = spike_times[starts[k]:stops[k]]
        idx = np.searchsorted(sp, flat).reshape(ntrials, NBINS + 1)
        out[i] = np.diff(idx, axis=1) / BIN_SIZE
    return out
```

iii. The file header and `CONVERSION_NOTES.md` Steps 1, 5, and 6 all state that the intended match to the reference is “spike count / bin width, in Hz,” with the decoder-required change from 40 ms sliding bins to 50 ms non-overlapping bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `classification == b'good'` and a non-empty CCF annotation (`anno_name`). It drops sessions with zero such units. It does not apply a firing-rate threshold.

ii.
```python
classification = u['classification'][:]
anno = _decode(u['anno_name'][:])
good = (classification == b'good')
good &= np.array([a.strip() != '' for a in anno])
unit_ids = np.where(good)[0]
if len(unit_ids) == 0:
    print('%s: no good units, skipping' % sess_id, flush=True)
    return None
```

iii. The file header and `CONVERSION_NOTES.md` Steps 1 and 5 justify this as matching the reference classifier-based QC (`qc_mode='classifier'`) plus the reference histology requirement that units have a CCF annotation. Step 5 explicitly rejects the 2 Hz firing-rate cutoff as specific to a different analysis.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to go-cue onset. For each kept trial it adds the fixed relative bin edges to the absolute go-cue time, then bins spikes against those absolute edges.

ii.
```python
go = go_all[trial_idx]
edges_abs = go[:, None] + BIN_EDGES[None, :]
centers_abs = go[:, None] + BIN_CENTERS[None, :]
```

```python
fr = bin_spikes(spikes, s0, s1, unit_ids, edges_abs)
```

iii. The script header states “alignment: go cue onset,” and `CONVERSION_NOTES.md` repeatedly says the NWB timestamps are on a common absolute session clock so go-cue alignment is achieved by building per-trial absolute windows around each go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins, giving 80 bins per trial across 4 seconds. The AI does not apply any extra temporal rebinning after the initial 50 ms binning.

ii.
```python
BIN_SIZE = 0.05
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2.0
```

iii. The script and notes justify this as a deliberate deviation from the paper’s 40 ms / 3.4 ms preprocessing because the task instructions explicitly require 50 ms bins for the decoder.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `sample_start_times` in `BehavioralEvents`, together with trial starts and go-cue times. It assigns each sample-start event to a trial and keeps the last sample start before that trial’s go cue as the tone onset.

ii.
```python
def tone_onset_times(trial_start, go, sample_start):
    ntrials = len(go)
    tone = np.full(ntrials, np.nan)
    if len(sample_start):
        idx = np.searchsorted(trial_start, sample_start, side='right') - 1
        for i, s in zip(idx, sample_start):
            if 0 <= i < ntrials and s < go[i]:
                if np.isnan(tone[i]) or s > tone[i]:
                    tone[i] = s
```

```python
sample_start = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:] \
    if 'sample_start_times' in f['acquisition/BehavioralEvents'] else np.array([])
tone = tone_onset_times(trial_start, go_all, sample_start)[trial_idx]
```

iii. `CONVERSION_NOTES.md` Step 5 says early licks replay the sample epoch, so the last sample start before the go cue is the behaviorally relevant tone onset.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes the value at each neural time bin as the absolute center time of that bin minus the trial’s inferred tone onset. If no tone is found, it falls back to `go - 1.85 s`.

ii.
```python
DEFAULT_SAMPLE_TO_GO = 1.85
...
missing = np.isnan(tone)
if missing.any():
    tone[missing] = go[missing] - DEFAULT_SAMPLE_TO_GO
```

```python
time_from_tone = (centers_abs - tone[:, None]).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 justifies the last-sample-start rule from the task structure, and Step 10 says the 1.85 s fallback was included as a defensive fallback and was never triggered in the final dataset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The AI computes time-from-tone on the same 80 go-aligned bin centers used for neural firing rates, so each input sample corresponds to the same time bin as the neural data.

ii.
```python
edges_abs = go[:, None] + BIN_EDGES[None, :]
centers_abs = go[:, None] + BIN_CENTERS[None, :]
...
time_from_tone = (centers_abs - tone[:, None]).astype(np.float32)
```

iii. The notes describe this as using the shared go-cue-relative time grid for both neural and input streams.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI primarily derives photostimulation from `BehavioralEvents/photostim_start_times` and `photostim_stop_times`. If those event series are missing, it falls back to reconstructing intervals from the trials-table `photostim_onset` and `photostim_duration` values plus trial start times.

ii.
```python
def photostim_intervals(f, trial_start, go):
    be = f['acquisition/BehavioralEvents']
    if 'photostim_start_times' in be and len(be['photostim_start_times/timestamps']) > 0:
        on = be['photostim_start_times/timestamps'][:]
        off = be['photostim_stop_times/timestamps'][:]
        n = min(len(on), len(off))
        return np.stack([on[:n], off[:n]], axis=1)
```

```python
    t = f['intervals/trials']
    onset = _decode(t['photostim_onset'][:])
    dur = _decode(t['photostim_duration'][:])
    ...
    a = trial_start[i] + float(onset[i])
    iv.append([a, a + float(dur[i])])
```

iii. `CONVERSION_NOTES.md` Step 4 maps both the event-stream timestamps and the trials-table photostim fields to the reference `task_stimulation` variable, and Step 10 says event timestamps are used first with the trial table as a fallback.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts photostimulation into a binary time-varying input. For each stimulation interval, it marks a bin as 1 if the interval overlaps that bin at all; otherwise the bin is 0.

ii.
```python
stim_iv = photostim_intervals(f, trial_start, go_all)
photostim = np.zeros((len(trial_idx), NBINS), dtype=np.float32)
if len(stim_iv):
    for a, b in stim_iv:
        photostim[(edges_abs[:, 1:] > a) & (edges_abs[:, :-1] < b)] = 1.0
```

iii. `CONVERSION_NOTES.md` Step 5 describes the target as “1 if the bin overlaps a stim interval else 0,” and Step 7 says this matches the expected 0.5 s block before the go cue on stimulated trials.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI keeps photostimulation in absolute session time and compares each stimulation interval to the exact absolute neural bin edges for each kept trial. That makes the photostim raster use the same bins as the neural firing rates.

ii.
```python
go = go_all[trial_idx]
edges_abs = go[:, None] + BIN_EDGES[None, :]
...
photostim[(edges_abs[:, 1:] > a) & (edges_abs[:, :-1] < b)] = 1.0
```

iii. The file header and notes both state that spikes, events, and video are all aligned on the absolute NWB session clock, so alignment is done by applying the same go-cue-centered time grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from the trials-table `outcome` and `trial_instruction` fields. It does not derive choice from lick timestamps.

ii.
```python
outcome = _decode(t['outcome'][:])
instruction = _decode(t['trial_instruction'][:])
...
oc = outcome[trial_idx]
instr = instruction[trial_idx]
```

```python
licked_left = ((oc == 'hit') & (instr == 'left')) | ((oc == 'miss') & (instr == 'right'))
licked_right = ((oc == 'hit') & (instr == 'right')) | ((oc == 'miss') & (instr == 'left'))
```

iii. `CONVERSION_NOTES.md` Step 5 says this matches the reference `behavior_report x task_trial_type` logic, and the trajectory notes say it was cross-checked against first-lick-after-go and matched on more than 99% of trials.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes choice as a three-class per-trial output repeated across all 80 bins: `0 = no lick`, `1 = left`, `2 = right`. Ignore trials stay at 0, hit trials take the instructed side, and miss trials take the opposite side.

ii.
```python
OUTPUT_VALUES = [
    ['no lick', 'left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['low (<40th pctile)', 'middle (40-60th pctile)', 'high (>60th pctile)', 'not visible'],
]
```

```python
choice = np.zeros(len(trial_idx), dtype=np.int64)
licked_left = ((oc == 'hit') & (instr == 'left')) | ((oc == 'miss') & (instr == 'right'))
licked_right = ((oc == 'hit') & (instr == 'right')) | ((oc == 'miss') & (instr == 'left'))
choice[licked_left] = 1
choice[licked_right] = 2
...
o[0] = choice[i]
```

iii. The code comment says this is the same logic as the reference `behavior_report x trial type` definition. `CONVERSION_NOTES.md` Step 5 justifies keeping a separate “no lick” class because ignore trials are required by the decoder task.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The AI uses the trials-table `outcome` field directly.

ii.
```python
outcome = _decode(t['outcome'][:])
...
oc = outcome[trial_idx]
```

iii. `CONVERSION_NOTES.md` Step 4 identifies `intervals/trials/outcome` as the NWB equivalent of the reference behavioral report variable.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore`, `miss`, and `hit` to integer classes `0`, `1`, and `2`, then repeats that per-trial value across all 80 bins.

ii.
```python
outcome_code = np.select([oc == 'ignore', oc == 'miss', oc == 'hit'], [0, 1, 2],
                         default=0).astype(np.int64)
...
o[1] = outcome_code[i]
```

iii. The notes describe outcome as a direct categorical decoder target with the paper’s native three trial outcomes.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The AI uses the trials-table `early_lick` field directly.

ii.
```python
early = _decode(t['early_lick'][:])
...
el_tr = early[trial_idx]
```

iii. `CONVERSION_NOTES.md` Step 4 maps this directly to the reference `behavior_early_report` variable.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI converts `early` to `1` and `no early` to `0`, then repeats that per-trial value across all 80 bins.

ii.
```python
early_code = (el_tr == 'early').astype(np.int64)
...
o[2] = early_code[i]
```

iii. The notes justify keeping early-lick trials because early lick is itself one of the decoder outputs.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI derives tongue y-position from the DeepLabCut tongue tracking series, preferring `Camera0_side_TongueTracking` and falling back to `Camera3_side_TongueTracking` if necessary. It uses `data[:, 1]` as tongue y and `data[:, 2]` as the tracking likelihood, together with the camera timestamps.

ii.
```python
bt = f['acquisition/BehavioralTimeSeries']
key = None
for k in ('Camera0_side_TongueTracking', 'Camera3_side_TongueTracking'):
    if k in bt:
        key = k
        break
...
data = bt[key]['data'][:]
ts = bt[key]['timestamps'][:]
y = data[:, 1]
lik = data[:, 2]
```

iii. `CONVERSION_NOTES.md` Steps 2, 4, and 5 say the NWB tongue stream matches the reference marker variables and note the fallback for sessions that might include a second camera.

## 8-b. How is `output` *Tongue y-position* processed?

i. The AI first marks frames visible only when `likelihood > 0.9`. It then averages visible tongue y values within each trial/bin on the go-aligned 50 ms grid, and computes per-session 40th and 60th percentiles from all visible binned values collected across the extracted windows of that session. Those percentiles define the class boundaries.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
...
visible = (lik > TONGUE_LIKELIHOOD_THRESHOLD) & np.isfinite(y)
```

```python
idx = np.searchsorted(ts, edges_abs)
lo, hi = idx[:, :-1], idx[:, 1:]
nvis = cs_n[hi] - cs_n[lo]
sumy = cs_y[hi] - cs_y[lo]
with np.errstate(invalid='ignore', divide='ignore'):
    mean_y = np.where(nvis > 0, sumy / np.maximum(nvis, 1), np.nan)
```

```python
vals = mean_y[visible]
if vals.size == 0:
    return cls
p40, p60 = np.percentile(vals, [40, 60])
```

iii. `CONVERSION_NOTES.md` Step 5 justifies the 0.9 threshold by saying the DLC likelihood is strongly bimodal, so the threshold choice is practically insensitive. Step 5 also says percentiles are computed “over the visible bins of that session’s extracted windows,” and the trajectory notes present this as a task-driven “per-session discretization.”

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses four classes: `0` for below the 40th percentile, `1` for values between the 40th and 60th percentiles inclusive, `2` for above the 60th percentile, and `3` for bins with no visible tongue frames.

ii.
```python
def discretize_tongue(mean_y, visible):
    cls = np.full(mean_y.shape, 3, dtype=np.int64)
    vals = mean_y[visible]
    if vals.size == 0:
        return cls
    p40, p60 = np.percentile(vals, [40, 60])
    cls[visible & (mean_y < p40)] = 0
    cls[visible & (mean_y >= p40) & (mean_y <= p60)] = 1
    cls[visible & (mean_y > p60)] = 2
    return cls
```

iii. `CONVERSION_NOTES.md` Step 5 states that this follows the decoder specification’s 40/20/40 visible-bin split plus a fourth “not visible” class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue y to the same go-cue-centered 50 ms bins used for neural activity by applying `np.searchsorted` on camera timestamps with the same absolute bin edges `edges_abs`.

ii.
```python
edges_abs = go[:, None] + BIN_EDGES[None, :]
...
idx = np.searchsorted(ts, edges_abs)
lo, hi = idx[:, :-1], idx[:, 1:]
```

iii. The script header and notes both say that video and neural data are aligned to the go cue on the shared NWB absolute time base.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases explicitly. Sessions with no good annotated units are skipped. Trials with non-finite go cues, outside observation windows, or with zero extracted spikes are removed. Missing tone detections fall back to `go - 1.85 s`. Missing tongue tracking for a bin yields the “not visible” class, and missing tongue tracking for an entire session yields all bins as not visible. If photostim event streams are absent, the code reconstructs photostim from the trial table instead. `_worker` also catches exceptions and drops bad files instead of aborting the full run.

ii.
```python
if len(unit_ids) == 0:
    print('%s: no good units, skipping' % sess_id, flush=True)
    return None
...
keep &= np.isfinite(go_all)
keep &= observed_trial_mask(u, unit_ids, trial_start, trial_stop)
```

```python
missing = np.isnan(tone)
if missing.any():
    tone[missing] = go[missing] - DEFAULT_SAMPLE_TO_GO
```

```python
if key is None:
    return (np.full((ntrials, NBINS), np.nan), np.zeros((ntrials, NBINS), dtype=bool))
...
cls = np.full(mean_y.shape, 3, dtype=np.int64)
```

```python
def _worker(args):
    ...
    except Exception as exc:
        import traceback
        print('ERROR processing %s: %s' % (path, exc), flush=True)
        traceback.print_exc()
        return None
```

iii. `CONVERSION_NOTES.md` Steps 6 and 10 justify the observation-window and all-zero-trial filters as fixes for real data defects found during validation. The tone fallback and camera fallback are justified in Step 5 as defensive handling for rare missing cases.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies session I/O, spike-time loading/binning, and video binning as the dominant costs, with pickle writing also nontrivial at the end. The code is structured to measure per-stage timing inside each session and to parallelize across sessions.

ii.
```python
timing = {}
...
t0 = time.time()
spikes = u['spike_times'][:]
...
timing['neural'] = time.time() - t0
```

```python
t0 = time.time()
sample_start = ...
stim_iv = photostim_intervals(f, trial_start, go_all)
...
timing['input'] = time.time() - t0
```

```python
t0 = time.time()
tongue_y, tongue_visible = tongue_y_per_bin(f, edges_abs)
tongue_class = discretize_tongue(tongue_y, tongue_visible)
timing['output'] = time.time() - t0
```

iii. `CONVERSION_NOTES.md` Steps 6, 7, and 9 explicitly identify spike I/O, per-unit binning, video processing, and final pickle writing as the expensive parts and motivate multiprocessing as the main runtime optimization.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorizes much of the heavy work, but the code still leaves a few loops: the per-unit loop in `bin_spikes`, the per-unit loop assigning region names, the per-stimulation-interval loop when filling the photostim input, and the per-trial loop when assembling output arrays. The notes emphasize that earlier nested loops were deliberately replaced with `searchsorted` and cumulative-sum vectorization.

ii.
```python
for i, k in enumerate(unit_ids):
    sp = spike_times[starts[k]:stops[k]]
    idx = np.searchsorted(sp, flat).reshape(ntrials, NBINS + 1)
    out[i] = np.diff(idx, axis=1) / BIN_SIZE
```

```python
for a, b in stim_iv:
    photostim[(edges_abs[:, 1:] > a) & (edges_abs[:, :-1] < b)] = 1.0
```

```python
for i in range(len(trial_idx)):
    o = np.empty((4, NBINS), dtype=np.int64)
    o[0] = choice[i]
    o[1] = outcome_code[i]
    o[2] = early_code[i]
    o[3] = tongue_class[i]
    outputs.append(o)
```

iii. `CONVERSION_NOTES.md` Step 6 explicitly says the agent replaced slower nested Python loops with one `searchsorted` per unit and cumsum-based video binning, because those were the loops that mattered most.

## 10-c. What processing does the code repeat multiple times?

i. The AI does not intentionally recompute major conversion products multiple times during a normal run. Each session is read once, binned once, and assembled once. The fixed time grid is defined once at module scope and reused for all trials and sessions.

ii.
```python
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2.0
```

```python
with Pool(min(args.nproc, len(files))) as pool:
    results = pool.map(_worker, list(zip(files, plot_flags)), chunksize=1)
...
data = {
    'neural': [r['neural'] for r in results],
    'input': [r['input'] for r in results],
    'output': [r['output'] for r in results],
```

iii. `CONVERSION_NOTES.md` Step 6 frames the conversion as a single pass over sessions, and the trajectory reflects that the agent’s optimizations were specifically meant to avoid repeated HDF5 reads and repeated nested looping.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI does a small amount of extra bookkeeping that is not preserved in the final output dataset: per-session `timing` and `duration` are computed and stored in the intermediate `result` dict but dropped during final assembly. Optional `--show-processing` plots also trigger extra rereads and visualization work that are purely diagnostic. Apart from that, most computed quantities are either emitted directly or summarized into metadata.

ii.
```python
result = {
    'session_id': sess_id,
    'subject': subject,
    ...
    'timing': timing,
    'duration': time.time() - t_start,
}
```

```python
data = {
    'neural': [r['neural'] for r in results],
    'input': [r['input'] for r in results],
    'output': [r['output'] for r in results],
    ...
}
```

```python
if make_plots:
    plot_processing(result, path, go, tone, edges_abs, tongue_y, tongue_visible, plot_dir)
```

iii. `CONVERSION_NOTES.md` justifies the timing data and plots as validation/debugging aids. It does not claim that downstream decoder training needs those intermediate artifacts.
