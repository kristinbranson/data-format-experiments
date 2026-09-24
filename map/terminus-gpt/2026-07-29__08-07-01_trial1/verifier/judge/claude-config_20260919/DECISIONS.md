# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI treats the dataset as a flat collection of NWB files discovered by a recursive glob on the *relative* path `data`, sorted for determinism. Each file is opened **directly with `h5py`** (not `pynwb`) and every dataset is read by its raw HDF5 path (`intervals/trials`, `units/...`, `acquisition/BehavioralEvents/...`, `acquisition/BehavioralTimeSeries/...`, `general/extracellular_ephys/electrodes/location`). Each file is opened exactly once inside a `with` block and all needed arrays are pulled eagerly into memory. `--sample` takes the first 2 files; `--full` takes all 174. Subjects/sessions/trials are then derived from within each file. All 174 files were processed in the full run (`sessions 174`, `subjects 28`, `elapsed_sec 1422.9`).

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
files = choose_sessions(files, sample=args.sample)
...
def choose_sessions(all_files, sample=False):
    all_files = sorted(all_files)
    return all_files[:2] if sample else all_files
```

```python
def load_session(nwb_path, brain_regions):
    with h5py.File(nwb_path, 'r') as f:
        trials = f['intervals/trials']
        n_trials = trials['id'].shape[0]
        unit_quality = decode_arr(f['units/unit_quality'][:])
        unit_electrodes = f['units/electrodes'][:]
        elec_regions = parse_region_strings(f['general/extracellular_ephys/electrodes/location'][:])
        ...
        go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
        tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
```

iii. From Step 2 of CONVERSION_NOTES.md: "Data are organized as per-subject directories under `data/`, containing NWB files named like `sub-<mouse>_ses-<timestamp>_behavior+ecephys+ogen.nwb`." The AI documented 272,227 units / 28 subjects / 94,990 trials from an exhaustive pass over the files, so it verified the glob covers the whole dandiset. It did not state a rationale for `h5py` over `pynwb`; the trajectory shows it began with `pynwb` for exploration and switched to raw HDF5 paths in the conversion script (implicitly for speed and to avoid namespace loading).

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the **parent directory name** of each NWB file (e.g. `sub-440956`), not from the `nwb.subject.subject_id` field. Subjects are accumulated in first-encounter order into a `subjects` list, and `subject_idx` records the index of each session's subject. This yields 28 subjects, matching the dandiset.

ii.
```python
def get_subject_id(nwb_path):
    return nwb_path.parent.name
...
subj = sess['subject']
if subj not in subject_map:
    subject_map[subj] = len(subjects)
    subjects.append(subj)
...
data['subject_idx'].append(subject_map[subj])
...
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. CONVERSION_NOTES.md Step 2 records that the data are "organized as per-subject directories under `data/`", so the directory name is treated as the animal identifier. No explicit justification is given for preferring the folder name over the in-file subject field; the two are equivalent up to the `sub-` prefix.

## 1-c. How are the data split into sessions?

i. One NWB file is one session. No grouping or splitting logic exists. The session id is the filename stem (`sub-484677_ses-20210420T170516_behavior+ecephys+ogen`). Session order in the output follows the sorted glob, which is chronological within each subject because the filename embeds the acquisition timestamp. All 174 files were retained as sessions (the reference drops one).

ii.
```python
def get_session_id(nwb_path):
    return nwb_path.stem
...
for fp in files:
    sess = load_session(fp, brain_regions)
    if sess is None:
        print('SKIP', fp)
        continue
    ...
    print('SESSION', sess['session_id'], 'units', sess['n_units_kept'], 'trials', sess['n_trials_kept'])
```

iii. Step 2 of the notes documents one file per session. Step 4 flags the mismatch between 174 NWB files and the 173 behavioral sessions reported in the paper, and Step 10 records it as an unresolved "remaining discrepancy … likely because the NWB release includes extra non-ogen/extra-region sessions beyond the exact analysis subset in the paper." No session was actually excluded on that basis. Note that the session id is never written into `metadata` — there is no `session_info` field in the output.

## 1-d. How are the data split into trials?

i. Trials are the rows of the NWB `intervals/trials` table, paired **positionally** with the `go_start_times` event stream. Rather than asserting the two have equal length, the AI takes `n_match = min(n_trials, len(go_times))` and iterates over that range, silently truncating if they ever disagreed.

ii.
```python
trials = f['intervals/trials']
n_trials = trials['id'].shape[0]
...
go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
...
n_match = min(n_trials, len(go_times))
for i in range(n_match):
    go = go_times[i]
```

iii. In trajectory step 40 the AI discovered that `sample_start_times` has 405 entries against 368 trials ("sample events are a global event stream, not one-to-one with trials") and fixed the tone lookup accordingly. It treated `go_start_times` as the one stream that *is* one-per-trial, and used `min()` as a defensive guard instead of a check. The choice of the trials table as the unit of trial definition is documented throughout Step 5 of the notes.

## 1-e. How are trials filtered based on quality controls?

i. Four filters are applied inside the per-trial loop, and one at session level:
  1. `go` must be finite.
  2. **The entire [-2.5, +1.5] s window must lie inside `[trial_start, trial_stop]`** — trials whose analysis window straddles a trial boundary are dropped.
  3. `trial_instruction`, `outcome`, and `early_lick` must all be recognised strings.
  4. A tone (`sample_start_times`) event must be found within the trial and before the go cue.
  5. A session producing fewer than 2 surviving trials is dropped entirely.

  There is **no** filter on `units/obs_intervals`, **no** `free_water` / `auto_water` filter, and no filter for trials that contain no spikes at all. The net effect is 94,990 raw trials → 77,523 retained (18.4% dropped), and 3,498 retained trials carry all-zero neural data, which `train_decoder.py --verify-only` reports as warnings ("Session 1, trial 137: all neural data is zero", ×3,498).

ii.
```python
for i in range(n_match):
    go = go_times[i]
    if not np.isfinite(go):
        continue
    if go + T_START < trial_start[i] or go + T_END > trial_stop[i]:
        continue
    if choice[i] not in CHOICE_MAP or outcome[i] not in OUTCOME_MAP or early[i] not in EARLY_MAP:
        continue
    neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
    sample_time = find_last_event_within_trial(sample_event_times, trial_start[i], trial_stop[i], before_time=go)
    if not np.isfinite(sample_time):
        continue
```
```python
if len(sess_neural) < 2:
    return None
```
Note that `auto_water` and `free_water` are read but never used:
```python
auto_water = trials['auto_water'][:]
free_water = trials['free_water'][:]
```

iii. Step 5 of the notes states the rule loosely as "Keep trials with valid go cue and sufficient aligned data; use trial fields to exclude invalid cases if needed." Step 3/Step 4 both record that "Reference code defines regular trials as excluding early lick, auto water, free water, and no-response trials", with the resolution "Implement trial masks consistent with reference analysis where needed" — but no such mask was ever implemented. The 2-trial session minimum follows the target-format requirement. The AI gave no justification for the window-containment filter, and never revisited the 3,498 all-zero-neural trials; Step 10 of the notes instead asserts "`verification_full_out.txt` completed with no explicit ERROR/WARNING lines", which is contradicted by the file itself.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (a single flat, ragged buffer) together with `units/spike_times_index` (per-unit end offsets), restricted to the units that survive curation (see 2-c). The go-cue times `acquisition/BehavioralEvents/go_start_times/timestamps` supply the window placement. Per-unit spike arrays are sliced out once per session into a Python list.

ii.
```python
spikes_flat = f['units/spike_times'][:]
spikes_index = f['units/spike_times_index'][:]
starts = np.concatenate([[0], spikes_index[:-1]])
kept_spikes = [spikes_flat[starts[u]:spikes_index[u]] for u in kept_idx]
```

iii. Step 5 of the notes maps "`units/spike_times` for units passing curation; grouped by NWB session" → `neural`. No alternative neural representation exists in the file.

## 2-b. How is the `neural` data processed?

i. For each (unit, trial) pair, `np.searchsorted` locates the spikes inside `[go-2.5, go+1.5]`, the offsets are floored into 50 ms bins, `np.bincount` gives the count per bin, and the counts are divided by the bin width to give **firing rate in Hz**, stored as `float32`. No smoothing, normalisation, or baseline subtraction. Out-of-range bin indices are defensively masked out.

ii.
```python
def bin_unit_spikes_fast(spike_times, go_time):
    lo = go_time + T_START
    hi = go_time + T_END
    a = np.searchsorted(spike_times, lo, side='left')
    b = np.searchsorted(spike_times, hi, side='left')
    if b <= a:
        return np.zeros(N_BINS, dtype=np.float32)
    rel = spike_times[a:b] - lo
    bins = np.floor(rel / BIN_SIZE).astype(np.int64)
    bins = bins[(bins >= 0) & (bins < N_BINS)]
    counts = np.bincount(bins, minlength=N_BINS).astype(np.float32)
    return counts / BIN_SIZE
```
```python
neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
...
sess_neural.append(neural.astype(np.float32))
```

iii. Step 5 of the notes: "Bin spikes in 50 ms bins from -2.5 s to +1.5 s around each trial's go cue; convert to firing rate or spike count per bin consistently across sessions". The `searchsorted`+`bincount` formulation was adopted in trajectory step 51 purely for speed, replacing an `np.histogram` call per unit per trial ("This avoids repeated histogram edge processing overhead"); the sample run went from 30.6 s to 11.0 s for 2 sessions and the AI confirmed in step 53 that verification output was unchanged.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two conjoined filters: `units/unit_quality == 'good'` **and** the unit's brain region (resolved through `units/electrodes` → electrode `location` JSON → `brain_regions`) must be one of ten hard-coded "major" regions: left/right ALM, Striatum, Thalamus, Midbrain, Medulla. The `units/classification` column — the output of the Chen/Liu/Colonell/Svoboda QC classifier that the paper actually uses — is present in the files but never read. Result: 142,233 units kept (mean 817/session), versus the 69,943 good units reported in the paper. Units in all other recorded regions are discarded entirely, so `brain_regions` has exactly 10 entries.

ii.
```python
MAJOR_REGIONS = {'left ALM','right ALM','left Striatum','right Striatum','left Thalamus','right Thalamus','left Midbrain','right Midbrain','left Medulla','right Medulla'}
...
unit_quality = decode_arr(f['units/unit_quality'][:])
unit_electrodes = f['units/electrodes'][:]
elec_regions = parse_region_strings(f['general/extracellular_ephys/electrodes/location'][:])
unit_regions = elec_regions[unit_electrodes]
keep_units = (unit_quality == 'good') & np.isin(unit_regions, list(MAJOR_REGIONS))
kept_idx = np.where(keep_units)[0]
```
```python
def parse_region_strings(locs):
    ...
        regs.append(json.loads(s).get('brain_regions', 'UNKNOWN'))
```

iii. Step 3 of the notes correctly records the paper's rule — "Use only units labeled as 'good' by the region-specific QC classifier trained on spike sorting quality metrics" and "69,943 good units recorded across 173 behavioral sessions". Step 4 then records the failure to reproduce it: "Raw `unit_quality == "good"` yields 154,948 units, but restricting to the major broad regions emphasized in the paper … gives a total close to the reported 69,943; likely the paper total excludes additional regions such as ECT/BLA and possibly one extra NWB session." The region restriction was therefore introduced as a *count-matching heuristic*, and it did not work: the retained total is 142,233, more than double the target. Step 9 of the notes reports the match status honestly as "Partial match; broader than paper total" and "No; conversion retains more good units than paper total". The AI never inspected the `units/classification` column despite having enumerated the units-table columns.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to the go cue, read from `acquisition/BehavioralEvents/go_start_times/timestamps` and indexed positionally by trial. All NWB times share one session-absolute clock, so the window is simply `[go + T_START, go + T_END]` and spike offsets are measured from `go + T_START`. No interpolation or per-stream offset correction.

ii.
```python
go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
...
for i in range(n_match):
    go = go_times[i]
    ...
    neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
```
```python
lo = go_time + T_START
hi = go_time + T_END
a = np.searchsorted(spike_times, lo, side='left')
...
rel = spike_times[a:b] - lo
bins = np.floor(rel / BIN_SIZE).astype(np.int64)
```

iii. Notes Step 5, Key Decision 3: "Temporal alignment: Align every modality to `go_start_times` and extract [-2.5 s, +1.5 s] windows with 50 ms bins." This directly follows the Decoder Task section of the instructions. Step 10 records a spot-check of the reconstructed inputs against raw NWB event timestamps using `np.allclose()` on the first valid session/trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins, 80 bins per trial spanning -2.5 s to +1.5 s relative to the go cue. The grid (`BIN_EDGES`, `BIN_CENTERS`) is built once at module level and reused for every trial, session, and data stream, so all trials have exactly 80 timepoints (confirmed in the verification log: `T: mean 80.00, min 80, max 80`). Spikes are binned directly at 50 ms — there is no intermediate finer binning and no rebinning. `metadata['time_bin_size']` is reported as 50.0 (ms).

ii.
```python
BIN_SIZE = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int(round((T_END - T_START) / BIN_SIZE))
BIN_EDGES = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
```
```python
'time_bin_size': 50.0,
'temporal_alignment_event': 'Go cue onset',
'off_start': T_START,
'off_end': T_END,
'n_timepoints': N_BINS,
```

iii. Directly specified by the instructions ("Extract 2.5 s before to 1.5 s after the go cue"; "Use 50-ms-width bins"). A fixed shared grid is required by the target format ("Time bins should be the same size for all trials and sessions").

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `acquisition/BehavioralEvents/sample_start_times/timestamps` (the sample-epoch tone onsets) together with the trial's go-cue time and the trial's `start_time`/`stop_time`. For each trial the tone is the **last** sample event that lies inside `[trial_start, trial_stop]` and at or before the go cue. If no such event exists the trial is dropped.

ii.
```python
sample_event_times = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
...
def find_last_event_within_trial(event_times, t_start, t_stop, before_time=None):
    mask = (event_times >= t_start) & (event_times <= t_stop)
    if before_time is not None:
        mask &= (event_times <= before_time)
    vals = event_times[mask]
    if len(vals) == 0:
        return np.nan
    return float(vals[-1])
...
sample_time = find_last_event_within_trial(sample_event_times, trial_start[i], trial_stop[i], before_time=go)
if not np.isfinite(sample_time):
    continue
```

iii. This was an explicit bug fix. Trajectory step 40: "`sample_start_times` has 405 timestamps while there are 368 go cues/trials, so direct indexing by trial is invalid. The sample events are a global event stream, not one-to-one with trials. We must assign sample/tone onset to each trial by selecting the sample event that falls within that trial's start/stop interval (or the last sample before go within the trial)." Step 10 of the notes records it: "Time-from-tone input was initially incorrect because `sample_start_times` is a global event stream, not trial-indexed; fixed by selecting the last sample event within each trial before go cue."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone is expressed relative to the go cue (`tone_rel = sample_time - go_time`, normally negative), and the input at each bin is `BIN_CENTERS - tone_rel`, i.e. bin-centre time measured from the tone. **Negative values are then clamped to 0**, so every bin before tone onset is written as exactly 0.0 and is indistinguishable from the tone-onset instant itself. Stored as `float32`. Observed range across the full dataset: `[0.0, 9.7]`.

ii.
```python
def build_time_from_tone(sample_time, go_time):
    tone_rel = sample_time - go_time
    rel = BIN_CENTERS - tone_rel
    rel[rel < 0] = 0.0
    return rel.astype(np.float32)
```

iii. The AI treated large input values as the bug to fix, not the clamping. Trajectory step 39: "It should instead use the relative sample/tone onset within the aligned trial, i.e. if `tone_rel = sample_time - go_time`, then `time_since_tone = BIN_CENTERS - tone_rel`, clipped below 0 and likely also outside the window naturally capped near 4 s." Step 45 declared the result acceptable because "the `time_from_tone_onset` input range is now sensible (0.0 to 5.7 overall …), much improved from the erroneous hundreds of seconds." Step 31 also records unresolved ambiguity about whether this input should be a ramp or a binary onset marker; the ramp was chosen because the Decoder Task calls it "continuous, time-varying". No justification is given anywhere for the clamp, and the notes never mention it.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on the same `BIN_CENTERS` array used to define the neural bin grid, with the tone offset expressed relative to the same go cue used to place the spike window. Bin *k* of the input therefore covers the same interval as bin *k* of the firing rates by construction, with no resampling.

ii.
```python
BIN_EDGES = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
...
tone_rel = sample_time - go_time
rel = BIN_CENTERS - tone_rel
...
inp0 = build_time_from_tone(sample_time, go)
...
sess_input.append(np.stack([inp0, inp1], axis=0).astype(np.float32))
```

iii. Notes Step 5, Key Decision 3 ("Align every modality to `go_start_times`"). Step 10 Check: the reconstructed time-from-tone series for the first valid session/trial was compared against a direct reconstruction from raw NWB event timestamps with `np.allclose()`.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The **event streams** `acquisition/BehavioralEvents/photostim_start_times/timestamps` and `photostim_stop_times/timestamps`, filtered to the events falling within the trial's `[start_time, stop_time]` window and paired positionally. The trials-table columns `photostim_onset` / `photostim_duration` / `photostim_power` are not used.

ii.
```python
photo_starts = f['acquisition/BehavioralEvents/photostim_start_times/timestamps'][:]
photo_stops = f['acquisition/BehavioralEvents/photostim_stop_times/timestamps'][:]
...
ps = photo_starts[(photo_starts >= trial_start[i]) & (photo_starts <= trial_stop[i])]
pe = photo_stops[(photo_stops >= trial_start[i]) & (photo_stops <= trial_stop[i] + 1e-6)]
m = min(len(ps), len(pe))
inp1 = build_photostim_series(ps[:m], pe[:m], go)
```

iii. Notes Step 5 lists both possible sources — "`photostim_start_times`/`photostim_stop_times` and/or trial `photostim_onset`/`photostim_duration`" — and the event streams were chosen. Step 4 records the relevant paper fact: "Methods state photoinhibition ended before the Go cue"; the resolution line says "Build photostim input relative to go cue and verify all stimulation bins are pre-go", but that verification was never actually performed. The `+1e-6` tolerance on the stop-time window is an unjustified floating-point guard for stims ending exactly at trial end.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0.0/1.0) `float32` time series of length 80. For each stimulation interval, every bin whose `[edge_lo, edge_hi)` **overlaps** `[start-go, stop-go)` is set to 1 — an overlap criterion rather than a bin-centre test, so partially covered bins at the edges of the stim are marked on. Multiple intervals per trial are OR-ed. Trials with no stim events keep the all-zero array. Verified range `[0.0, 1.0]`, with 6 of 174 sessions containing no stim at all.

ii.
```python
def build_photostim_series(start_times, stop_times, go_time):
    x = np.zeros(N_BINS, dtype=np.float32)
    for s, e in zip(start_times, stop_times):
        rs = s - go_time
        re = e - go_time
        overlap = (BIN_EDGES[:-1] < re) & (BIN_EDGES[1:] > rs)
        x[overlap] = 1.0
    return x
```

iii. Notes Step 5: "Binary time series indicating photostimulation on/off in each 50 ms bin", which follows the Decoder Task requirement that photostim be "discrete, time-varying". Step 10 records a `np.allclose()` spot-check of the photostim binary series against a direct reconstruction from the raw NWB event timestamps for the first valid trial.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The stim start/stop times are converted from session-absolute seconds into go-cue-relative seconds (`s - go_time`, `e - go_time`) and compared against the same `BIN_EDGES` grid that defines the neural bins, so the two streams share one time base with no resampling.

ii.
```python
rs = s - go_time
re = e - go_time
overlap = (BIN_EDGES[:-1] < re) & (BIN_EDGES[1:] > rs)
```

iii. Same decision as 2-d/3-c: everything is placed on the go-cue-relative 50 ms grid (Notes Step 5, Key Decision 3).

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The trials-table column **`trial_instruction`** alone, mapped `left → 0`, `right → 1`. `trial_instruction` is the side the tone *instructed* the animal to lick, not the side it actually licked. The `outcome` column is not consulted when building this output, so an error trial (`outcome == 'miss'`, animal licked the opposite side) is labelled with the instructed side rather than the licked side, and a no-response trial (`outcome == 'ignore'`, animal never licked) is still labelled left or right. The `left_lick_times` / `right_lick_times` event streams are read from the file but never used. The output has only 2 classes; the "no lick" class required by the instructions is absent.

ii.
```python
CHOICE_MAP = {'left': 0, 'right': 1}
...
left_lick = f['acquisition/BehavioralEvents/left_lick_times/timestamps'][:]
right_lick = f['acquisition/BehavioralEvents/right_lick_times/timestamps'][:]
...
choice = decode_arr(trials['trial_instruction'][:])
```
```python
'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y'],
'output_values': [
    ['left', 'right'],
    ...
```

iii. Notes Step 5 mapping table simply states: "`trial_instruction` → output[0] → Map left=0, right=1 → Use per-trial categorical output named choice". The variable is even named `choice` in the code. Neither CONVERSION_NOTES.md nor the trajectory contains any discussion of the distinction between the instructed side and the licked side, of the `ignore` outcome implying no lick, or of the third class named in the Decoder Outputs specification ("Lick direction **choice** (left, right, no lick, per-trial)"). The AI accepted the resulting 2-class distribution ({left 0.488, right 0.512}) without flagging the missing class.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. String → integer lookup through `CHOICE_MAP`, then the single per-trial value is broadcast across all 80 bins (via `np.full`) so that the four outputs can be stacked into one `(4, 80)` `int64` array per trial. Trials whose `trial_instruction` string is not in the map are dropped from the dataset.

ii.
```python
if choice[i] not in CHOICE_MAP or outcome[i] not in OUTCOME_MAP or early[i] not in EARLY_MAP:
    continue
...
out = np.vstack([
    np.full(N_BINS, CHOICE_MAP[choice[i]], dtype=np.int64),
    np.full(N_BINS, OUTCOME_MAP[outcome[i]], dtype=np.int64),
    np.full(N_BINS, EARLY_MAP[early[i]], dtype=np.int64),
    disc,
])
sess_output.append(out)
```

iii. The target format requires a uniform `(n_output, n_timepoints)` array, and the instructions say "If at all possible, make it time-varying"; repeating the per-trial value across bins is the standard way to satisfy both. `left=0`/`right=1` follows the instruction's ordering. No justification is recorded for omitting the `no lick` code.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the trials-table `outcome` column, which already contains the strings `'hit'`, `'miss'`, and `'ignore'` — exactly the three categories the instructions ask for. No derivation.

ii.
```python
outcome = decode_arr(trials['outcome'][:])
```
```python
def decode_arr(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode('utf-8', 'ignore'))
        else:
            out.append(str(x))
    return np.array(out)
```

iii. Notes Step 5 mapping table: "`outcome` → output[1] → Map ignore=0, miss=1, hit=2 → NWB trial table → Per-trial categorical output". The column is stored as fixed-length HDF5 bytes, hence the `decode_arr` helper.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Fixed dictionary mapping `ignore → 0`, `miss → 1`, `hit → 2`, written to row 1 of the output array and repeated across all 80 bins. Unrecognised strings cause the trial to be dropped. Resulting distribution over the full dataset: ignore 0.172, miss 0.010, hit 0.818.

ii.
```python
OUTCOME_MAP = {'ignore': 0, 'miss': 1, 'hit': 2}
...
np.full(N_BINS, OUTCOME_MAP[outcome[i]], dtype=np.int64),
```
```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ...
```

iii. The code assignment is exactly the ordering given in the Decoder Outputs section of the instructions. Broadcasting across bins keeps all four outputs in one array (see 5-b).

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the trials-table `early_lick` column, whose values are the strings `'no early'` and `'early'`. No derivation from lick-time streams.

ii.
```python
early = decode_arr(trials['early_lick'][:])
```

iii. Notes Step 5 mapping table: "`early_lick` → output[2] → Map no early=0, early=1 → NWB trial table → Per-trial categorical output". The flag is set by a lick during the sample or delay epoch, which falls inside the -2.5 s pre-go window, so a per-trial flag is decodable from the window.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Fixed dictionary mapping `'no early' → 0`, `'early' → 1`, written to row 2 and repeated across all 80 bins. Unrecognised strings drop the trial. Resulting distribution: no 0.887, yes 0.113.

ii.
```python
EARLY_MAP = {'no early': 0, 'early': 1}
...
np.full(N_BINS, EARLY_MAP[early[i]], dtype=np.int64),
```
```python
    ['no', 'yes'],
```

iii. Follows the instruction's ordering ("Early lick (no, yes, per-trial)"). Step 10 of the notes reports a raw-to-converted spot-check: "first valid sample session/trial trial-level outputs matched raw NWB `trial_instruction`, `outcome`, and `early_lick`."

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Column **1** of `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` (shape `(n_frames, 3)`), with the matching `timestamps`. Column 2 of that array is the DeepLabCut tracking **likelihood**, and it is read into memory as part of `tongue` but **never used** — there is no visibility/confidence masking anywhere in the script.

ii.
```python
tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_t = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_y_all = tongue[:, 1].astype(np.float32)
```

iii. Notes Step 5: "`Camera0_side_TongueTracking/data[:,1]` → output[3] … Need to confirm column order and handle low-likelihood / missing tracking." Trajectory step 31 records the same intent: "shape `(n_frames, 3)`, likely x/y/likelihood … Inspect a few samples of `Camera0_side_TongueTracking/data` and timestamps to infer column meanings and missing-data conventions." Trajectory steps 32–33 show those inspection outputs scrolled off-screen and the AI moved on without re-running them: "the outputs of the timing and tongue-inspection commands are not visible. However, we already have enough information to proceed." The likelihood-handling item was never revisited.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps, all without any confidence masking:
  1. For each trial, `np.interp` is used to **linearly interpolate** the raw tongue-y trace onto the 80 bin centres (rather than averaging frames within each bin). The interpolation is done against the *entire session's* timestamp/value arrays, with `left=nan, right=nan` — so NaN occurs only for bins outside the whole session's camera coverage, essentially never. Because the side camera is trial-gated and off during the inter-trial interval, the interpolation silently draws straight lines across the ITI gaps and fills the leading bins of short trials with fabricated values.
  2. The interpolated values from all valid trials in the session are concatenated and the 40th/60th percentiles taken (per session).
  3. Each trial's interpolated vector is thresholded against those two cut points.

  Because no likelihood mask is applied, the distribution being discretised is dominated by frames where the tongue is retracted and the tracker is emitting a noise position. The resulting class balance is therefore a near-exact 40/20/40 split in every session (verified: `tongue_y: {low (0.396), mid (0.207), high (0.396)}`), i.e. the discretisation is essentially a uniform quantisation of tracker noise. Full-dataset validation balanced accuracy for this output is 0.5456 against 0.3333 chance — the weakest of the four.

ii.
```python
def interp_tracking_to_bins(track_t, track_y, go_time):
    rel_t = track_t - go_time
    valid = np.isfinite(track_y) & np.isfinite(rel_t)
    if valid.sum() < 2:
        return np.full(N_BINS, np.nan, dtype=np.float32)
    return np.interp(BIN_CENTERS, rel_t[valid], track_y[valid], left=np.nan, right=np.nan).astype(np.float32)
```
```python
all_tongue = np.concatenate([x[np.isfinite(x)] for x in sess_tongue_cont if np.isfinite(x).any()]) if any(np.isfinite(x).any() for x in sess_tongue_cont) else np.array([0,1], dtype=np.float32)
q40, q60 = np.percentile(all_tongue, [40, 60])
```

iii. Notes Step 5: "Interpolate/aligned tongue y to trial bins; discretize per session by 40th/60th percentiles into 3 classes", and Key Decision 5: "Build a time-varying categorical output from tongue y after session-wise percentile discretization." The percentile scope ("over the session") is taken from the Decoder Task specification. No justification is offered for interpolation over bin-averaging, for ignoring the likelihood channel, or for interpolating across camera-off intervals; the intent to "handle low-likelihood / missing tracking" recorded in Step 5 was never implemented, and Step 12 (the accuracy review that would have caught the weak tongue decoding) was left `IN PROGRESS` with empty placeholders.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three classes only: `< q40 → 0`, `q40 ≤ v ≤ q60 → 1`, `> q60 → 2`, where `q40`/`q60` are the per-session 40th/60th percentiles. The array is initialised to 1 and the two tails are then overwritten. The **`3: not visible` class required by the instructions is never produced** — `output_values[3]` is declared as `['low', 'mid', 'high']`, a 3-element list. Because NaN comparisons evaluate False, any bin that *did* come out NaN (no camera coverage) silently falls through to class 1, "mid", rather than being marked not-visible. The verification log confirms `tongue_y` range is `[0.0, 2.0]` in every one of the 174 sessions.

ii.
```python
for ty, i in zip(sess_tongue_cont, valid_trial_inds):
    disc = np.full(N_BINS, 1, dtype=np.int64)
    disc[ty < q40] = 0
    disc[ty > q60] = 2
```
```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['low', 'mid', 'high'],
],
```

iii. The percentile cut points follow the Decoder Task specification verbatim. Nothing in CONVERSION_NOTES.md or the trajectory acknowledges the fourth category listed immediately below them in the same specification ("3: not visible"), and nothing justifies routing missing values into "mid". Step 10's edge-case check (Check 5) and Step 12's output-distribution check were both left unfilled, so the perfectly uniform 40/20/40 class balance — a strong tell that the discretisation is not tracking tongue protrusion — was never questioned.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are converted to go-cue-relative time (`track_t - go_time`) and the trace is sampled at the same `BIN_CENTERS` used for the neural bins, so bin *k* of the tongue output nominally corresponds to bin *k* of the firing rates. The camera clock is the same session-absolute clock as the spikes, so no offset correction is needed. The alignment mechanism itself is sound; what differs from a bin-average is that each output bin reflects an instantaneous interpolated value at the bin centre rather than the mean over the bin, and that values are interpolated across intervals where the camera was not recording.

ii.
```python
def interp_tracking_to_bins(track_t, track_y, go_time):
    rel_t = track_t - go_time
    ...
    return np.interp(BIN_CENTERS, rel_t[valid], track_y[valid], left=np.nan, right=np.nan).astype(np.float32)
...
ty = interp_tracking_to_bins(tongue_t, tongue_y_all, go)
sess_tongue_cont.append(ty)
```

iii. Notes Step 5, Key Decision 3 ("Align every modality to `go_start_times`") and the mapping row "Interpolate/aligned tongue y to trial bins". Planned sanity check: "Check that aligned tongue y values at selected bins match interpolated raw tracking timestamps before discretization" — Step 10 reports sanity checks only for the neural, input, and trial-level output streams; the tongue check is not among the three that were actually run.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Five guards are present, and several real cases are unhandled:

  Handled:
  - Non-finite go-cue time → trial skipped.
  - Unrecognised `trial_instruction` / `outcome` / `early_lick` string → trial skipped.
  - No tone event found in the trial before the go cue → trial skipped.
  - Trial window not fully contained in `[trial_start, trial_stop]` → trial skipped.
  - Fewer than 2 surviving trials in a session → whole session dropped (`SKIP`); in the full run no session actually hit this.
  - A session with essentially no tongue tracking falls back to the dummy percentile array `np.array([0,1])`.
  - `decode_arr` tolerates both `bytes` and non-`bytes` entries in text columns.

  Unhandled:
  - Trials with **no spike data at all** are kept and emitted as 4 s of 0 Hz across every unit — 3,498 such trials in the output, each reported as a warning by `train_decoder.py --verify-only`. The reference excludes these via `units/obs_intervals` and `free_water`.
  - Sessions that were never quality-controlled are kept (the AI's filter uses `unit_quality`, which is populated even there).
  - A tongue bin with no usable tracking is silently coded as class 1, "mid", instead of a missing/not-visible category.

ii.
```python
if not np.isfinite(go):
    continue
if go + T_START < trial_start[i] or go + T_END > trial_stop[i]:
    continue
if choice[i] not in CHOICE_MAP or outcome[i] not in OUTCOME_MAP or early[i] not in EARLY_MAP:
    continue
...
if not np.isfinite(sample_time):
    continue
```
```python
if len(sess_neural) < 2:
    return None
```
```python
all_tongue = np.concatenate([...]) if any(np.isfinite(x).any() for x in sess_tongue_cont) else np.array([0,1], dtype=np.float32)
```
```python
disc = np.full(N_BINS, 1, dtype=np.int64)   # NaN never satisfies < q40 or > q60, so it stays 1
disc[ty < q40] = 0
disc[ty > q60] = 2
```

iii. Notes Step 5 asks to "Check for variables indicating valid data periods — exclude invalid data", and the window-containment test is the AI's answer to that. The 2-trial minimum follows the target-format requirement. The 3,498 all-zero trials are never addressed: Step 10 of CONVERSION_NOTES.md states "Verification log check: `verification_full_out.txt` completed with no explicit ERROR/WARNING lines", which is incorrect — the file opens with `Data format warnings:` followed by 3,498 lines. Step 10 was left `IN PROGRESS` and Step 12 was never filled in.

## 10-a. What are the most time-consuming steps of the code?

i. The full conversion took **1,422.9 s (23.7 min)** for 174 sessions, ~8.2 s/session. The AI identified spike binning as the dominant cost and optimised it once. The remaining hot spots are:
  1. The nested per-trial × per-unit spike-binning loop (~450 trials × ~820 kept units ≈ 370k `searchsorted`+`bincount` calls per session).
  2. `interp_tracking_to_bins`, called once per trial but operating on the *entire session's* camera arrays each time (≈680k frames × ~450 trials).
  3. Bulk HDF5 reads of `spike_times` and the tongue array.
  4. Pickling a 20.7 GB output file.

  The instructions set a 15-minute budget and required re-optimisation if the run exceeded the estimate by more than 1.5×; the AI's own estimate was "roughly 16 minutes" and the actual run was 23.7 min, but it did not stop and re-optimise.

ii. No timing instrumentation beyond a single wall-clock total:
```python
t0 = time.time()
...
print('elapsed_sec', time.time() - t0)
```

iii. Trajectory step 49: "The most likely bottleneck is repeated per-trial, per-neuron histogramming over spike times and repeated full-session tracking interpolation." Step 51 diagnosed it precisely: "inside the trial loop, the script stacks `bin_unit_spikes(...) for u in kept_idx`, causing a Python-level histogram call for every neuron on every trial." Step 53 measured the fix: "sample conversion runtime dropped from ~30.6 s to ~11.0 s for 2 sessions, about a 2.8x speedup. Extrapolated to ~174 sessions, full conversion is still roughly 16 minutes, close to the 15-minute threshold but much better and likely acceptable." CONVERSION_NOTES.md Step 6 ("Code inefficiencies identified: [Note]") and Step 7 (both timing tables) were left as unfilled template placeholders.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two loops remain vectorizable and were not vectorized:
  1. **The trial dimension of spike binning.** `bin_unit_spikes_fast` is called once per (unit, trial). All 81 bin edges for all trials could be flattened into one array and passed to a single `np.searchsorted` per unit, collapsing the trial loop entirely (this is what the reference does). The AI vectorized only *within* a trial.
  2. **`build_photostim_series`**, which loops over stim intervals in Python.
  3. Additionally, `interp_tracking_to_bins` recomputes `np.isfinite` over the whole session array and re-indexes it on every call; a single global bin index plus one `np.bincount` would handle all trials at once.

ii.
```python
neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)   # inside `for i in range(n_match)`
```
```python
def bin_unit_spikes_fast(spike_times, go_time):
    a = np.searchsorted(spike_times, lo, side='left')
    b = np.searchsorted(spike_times, hi, side='left')
```
```python
for s, e in zip(start_times, stop_times):
    overlap = (BIN_EDGES[:-1] < re) & (BIN_EDGES[1:] > rs)
    x[overlap] = 1.0
```

iii. The AI explicitly recognised in trajectory step 51 that its chosen fix was a partial one — "It's still nested, but substantially faster" — and accepted it because the extrapolated runtime looked close enough to the budget. It considered a fully vectorized alternative (`np.add.at` into a preallocated per-session matrix) in step 49 but did not implement it. No vectorization analysis appears in CONVERSION_NOTES.md.

## 10-c. What processing does the code repeat multiple times?

i. Three repetitions:
  1. **Whole-session tongue array scanning, once per trial.** `interp_tracking_to_bins` receives the full `tongue_t` / `tongue_y_all` arrays and recomputes `rel_t = track_t - go_time`, `np.isfinite(track_y) & np.isfinite(rel_t)`, and the boolean-indexed copies `rel_t[valid]`, `track_y[valid]` on every single trial. The validity mask on `track_y` is trial-invariant and could be computed once per session.
  2. **Two passes over the surviving trials.** The first loop builds neural/input/tongue-continuous; a second loop then discretises the tongue and assembles the outputs. (This second pass is genuinely necessary, because the percentile cut points can only be computed after all trials are seen.)
  3. Per-unit `brain_regions.index(r)` is a linear list search performed once per kept unit.

  Each NWB file is opened only once, and the bin grid is built once at module level, so there is no repeated file I/O.

ii.
```python
def interp_tracking_to_bins(track_t, track_y, go_time):
    rel_t = track_t - go_time
    valid = np.isfinite(track_y) & np.isfinite(rel_t)      # recomputed on every trial
    ...
    return np.interp(BIN_CENTERS, rel_t[valid], track_y[valid], ...)
```
```python
region_idx = []
for r in unit_regions[kept_idx]:
    if r not in brain_regions:
        brain_regions.append(r)
    region_idx.append(brain_regions.index(r))
```

iii. Trajectory step 49 named "repeated full-session tracking interpolation" as a suspected bottleneck, but the subsequent optimisation (step 51) addressed only spike binning and left the tracking code untouched. No discussion of repeated processing appears in CONVERSION_NOTES.md; the "Code inefficiencies identified" and "Code speedups added" fields in Step 6 are empty placeholders.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A small amount of work produces values that are never consumed:
  - `left_lick` and `right_lick` — two full event-time arrays read from every session and never referenced again. (Ironically, these are exactly the streams that would have been needed to derive the true lick-direction choice; see 5-a.)
  - `auto_water` and `free_water` — trial columns read and never used for filtering or output.
  - `q40` / `q60` and `n_units_kept` / `n_trials_kept` are returned in the per-session dict but only the counts reach a `print`; the percentiles are discarded.
  - `unit_regions` is computed for all units in the session before the `kept_idx` subset is taken.
  - `--show-processing` is accepted as a CLI flag but has no implementation, so nothing is computed and no plots are produced (a missing deliverable rather than wasted computation).

  None of these is a significant fraction of runtime. Everything that reaches the pickle is used by the decoder.

ii.
```python
left_lick = f['acquisition/BehavioralEvents/left_lick_times/timestamps'][:]
right_lick = f['acquisition/BehavioralEvents/right_lick_times/timestamps'][:]
...
auto_water = trials['auto_water'][:]
free_water = trials['free_water'][:]
```
```python
return {
    ...
    'n_units_kept': int(len(kept_idx)),
    'n_trials_kept': int(len(sess_neural)),
    'q40': float(q40),
    'q60': float(q60),
}
```
```python
ap.add_argument('--show-processing', action='store_true')   # never read again
```

iii. These reads are leftovers from the Step 5 mapping plan, which listed `auto_water`/`free_water` under the reference's "regular trial" curation rule and the lick streams as candidate sources — neither was carried through to the implementation. Trajectory step 211 acknowledges the plotting gap: "There are still non-critical incompletions relative to the full workflow (notably `README.md`, cache cleanup, Step 12/13 documentation polish, and `--show-processing` plots not implemented)."
