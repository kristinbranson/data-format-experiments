# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is one NWB file per session under `data/sub-<subject>/`. The AI finds every session with a single relative-path glob and opens each file directly with `h5py` (raw HDF5 access) rather than with `pynwb`. Within a file it reads the trials table from `/intervals/trials/*`, unit metadata and spikes from `/units/*`, task events from `/acquisition/BehavioralEvents/*`, and video tracking from `/acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`. String columns are decoded from bytes explicitly. Each file is opened once, processed in `process_session()`, and closed in a `finally` block. 174 files are found; 144 survive filtering.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
print(f"Found {len(nwb_files)} NWB files")
...
for nwb_path in nwb_files:
    result = process_session(nwb_path)
```
```python
f = h5py.File(nwb_path, 'r')
try:
    n_trials = len(f['intervals/trials/id'][:])
    outcomes = np.array([o.decode() if isinstance(o, bytes) else o
                        for o in f['intervals/trials/outcome'][:]])
    ...
    go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
    ...
    classification = np.array([c.decode() if isinstance(c, bytes) else c
                               for c in f['units/classification'][:]])
    spike_times_data = f['units/spike_times'][:]
    spike_times_index = f['units/spike_times_index'][:]
finally:
    f.close()
```

iii. From the trajectory, the AI first had a sub-agent map the NWB internal layout (steps 16-17, 20-28) and then read fields by HDF5 path. It never states why it chose `h5py` over `pynwb`; the implicit reason is direct, dependency-light access to the ragged `spike_times`/`obs_intervals` datasets and their `_index` vectors, which it inspected explicitly (steps 40-44). Sorting the glob makes session order deterministic; the AI verified the file count (174) and the resulting good-unit count (69,453 before its own filters) against the white paper's 69,943 / 173 sessions (step 29).

## 1-b. How are the data split into subjects?

i. Subject identity is parsed from the **file name** (`sub-440956_ses-...nwb` → `'440956'`) rather than read from the NWB `subject` group. At assembly the unique ids are sorted into `subjects` and each session gets an index into that list. This yields the same 28 subjects as the reference.

ii.
```python
basename = os.path.basename(nwb_path)
subject_id = basename.split('_ses-')[0].replace('sub-', '')
```
```python
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. No explicit justification is given in the trajectory. The directory/file naming convention (`sub-<subject_id>`) is DANDI-standard and is derived from the NWB `subject_id`, so the parsed string is identical to `nwb.subject.subject_id`. The verification log confirms 28 subjects with 2-10 sessions each.

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no grouping or splitting is performed, and session order follows the sorted file list. Each kept session is recorded by file name in `metadata['session_files']`. **In addition, the AI applies a behavioural session-level filter taken from the data paper's methods**: a session is kept only if performance on control trials is > 65% *and* there are ≥ 50 correct left and ≥ 50 correct right control trials. Control trials are defined as no photostim, no early lick, no auto-water, no free-water. Performance is computed as hits / (control trials that responded), i.e. `ignore` trials are excluded from the denominator. This drops 30 of 174 sessions (20 for performance, 7 for too few correct trials per direction, 1 for having no QC-good units, 2 for other reasons), leaving 144.

ii.
```python
is_control = ((early_lick == 'no early') &
              (auto_water == 0) &
              (free_water == 0) &
              (ps_onset_str == 'N/A'))

control_hits_left = np.sum(is_control & (outcomes == 'hit') & (instructions == 'left'))
control_hits_right = np.sum(is_control & (outcomes == 'hit') & (instructions == 'right'))

control_responding = is_control & (outcomes != 'ignore')
n_control_responding = np.sum(control_responding)
...
performance = np.sum(is_control & (outcomes == 'hit')) / n_control_responding

if performance <= MIN_PERFORMANCE:
    return None
if control_hits_left < MIN_CORRECT_PER_DIRECTION or control_hits_right < MIN_CORRECT_PER_DIRECTION:
    return None
```

iii. The AI quotes the data paper's methods verbatim as its justification (read at step 9, plan at step 18): *"We selected experimental sessions for analysis based on following criteria: overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each."* `CONVERSION_NOTES.md` lists this under "Session Filtering — **Matches reference**: Methods text states ...", and notes that the resulting unit count (57,925) is below the paper's 69,943 "due to session filtering". It also sanity-checks mean session performance (~83%) against the paper's 84%.

## 1-d. How are the data split into trials?

i. Trials come directly from the NWB trials table; the number of rows defines the trial count, and the AI asserts that there is exactly one `go_start_times` event per row so that trial *i* maps to go cue *i*. All per-trial variables (`outcome`, `early_lick`, `trial_instruction`, `auto_water`, `free_water`, `photostim_onset`, `photostim_duration`, `start_time`) are read as parallel arrays indexed by trial.

ii.
```python
n_trials = len(f['intervals/trials/id'][:])
...
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
assert len(go_cue_times) == n_trials, f"Go cue count ({len(go_cue_times)}) != trial count ({n_trials})"
```

iii. The AI examined the event streams (steps 20-24) and established that the sample epoch consists of 3 tones (0.65 s total) followed by a 1.2 s delay, and that `sample_start_times` can have **more** than one entry per trial because an early lick replays the sample epoch — hence it uses the trials table plus the one-per-trial go cue as the trial definition, and treats `sample_start_times` as a multi-event stream to be looked up per trial (see 3-a).

## 1-e. How are trials filtered based on quality controls?

i. Two filters, both aimed at data availability rather than behaviour:
- **Reward-delivery trials**: `auto_water == 0 & free_water == 0` (the reference excludes only `free_water`).
- **Recording coverage**: for the selected units, the AI takes the intersection of the observation spans (`max` of each unit's first `obs_intervals` start, `min` of each unit's last `obs_intervals` stop) and keeps a trial only if its whole analysis window `[go − 2.5 s, go + 1.5 s]` falls inside that span, with a 0.1 s slack on each side.
- A session with fewer than 2 surviving trials is dropped.
- Early-lick, `ignore` (no-response) and photostim trials are deliberately **kept**.

74,759 trials survive over 144 sessions (mean 519/session, range 160-796).

ii.
```python
obs_intervals = f['units/obs_intervals'][:]
obs_intervals_index = f['units/obs_intervals_index'][:]

min_obs_end = np.inf
max_obs_start = -np.inf
for ui in unit_indices:
    oi_start = 0 if ui == 0 else obs_intervals_index[ui - 1]
    oi_end = obs_intervals_index[ui]
    unit_obs = obs_intervals[oi_start:oi_end]
    if len(unit_obs) > 0:
        min_obs_end = min(min_obs_end, unit_obs[-1, 1])
        max_obs_start = max(max_obs_start, unit_obs[0, 0])

recording_valid = ((go_cue_times + T_START) >= max_obs_start - 0.1) & \
                  ((go_cue_times + T_END) <= min_obs_end + 0.1)
...
trial_mask = (auto_water == 0) & (free_water == 0) & recording_valid
trial_indices = np.where(trial_mask)[0]
if len(trial_indices) < 2:
    return None
```

iii. The observation-window filter was added empirically after the decoder verifier flagged large blocks of all-zero trials (steps 37-48): the AI found that in some files spikes stop at ~1103 s while go cues run to ~3707 s, and that `is_good_trials` had only 160 columns for a 480-trial session, i.e. "the NWB file can contain more trials than the recording covers". It chose `obs_intervals` over `is_good_trials` because the latter is indexed differently from the trials table. Keeping early-lick and `ignore` trials is justified explicitly at step 18: the methods text says to exclude them, "but the decoder outputs include these as variables... excluding those trials would make those outputs meaningless. The reference paper's exclusion was for their specific analysis, but our decoder specification overrides that". `auto_water`/`free_water` exclusion is described in `CONVERSION_NOTES.md` as following the reference preprocessing code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From the ragged spike-time store `/units/spike_times` together with `/units/spike_times_index`, restricted to QC-good, region-mappable units, plus `/acquisition/BehavioralEvents/go_start_times/timestamps` to position the window. Per-unit spike arrays are sliced out once per session.

ii.
```python
spike_times_data = f['units/spike_times'][:]
spike_times_index = f['units/spike_times_index'][:]

unit_spike_times = []
for ui in unit_indices:
    start_idx = 0 if ui == 0 else spike_times_index[ui - 1]
    end_idx = spike_times_index[ui]
    unit_spike_times.append(spike_times_data[start_idx:end_idx])
```

iii. Spike times are the only neural representation in the files. The AI verified the ragged-index convention directly against the data (steps 40-44) after seeing all-zero trials, and confirmed the per-unit spike ranges against `obs_intervals`.

## 2-b. How is the `neural` data processed?

i. Spike counts per 50 ms bin, converted to Hz by dividing by the bin width. For each trial and each unit, spike times are shifted by the trial's go cue, masked to the window, histogrammed with `np.histogram` over the 81 fixed edges, and divided by 0.05. No smoothing, no normalisation, no baseline subtraction. Results are stored as `float64` `(n_units, 80)` arrays, one per trial.

ii.
```python
def compute_firing_rates(spike_times, go_cue_time, bin_edges):
    rel_times = spike_times - go_cue_time
    mask = (rel_times >= bin_edges[0]) & (rel_times < bin_edges[-1])
    rel_times = rel_times[mask]
    counts, _ = np.histogram(rel_times, bins=bin_edges)
    bin_width = bin_edges[1] - bin_edges[0]
    return counts.astype(np.float64) / bin_width
```
```python
for ti in trial_indices:
    gc = go_cue_times[ti]
    fr_matrix = np.zeros((n_units, N_BINS), dtype=np.float64)
    for j, st in enumerate(unit_spike_times):
        fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
    neural_trials.append(fr_matrix)
```

iii. `CONVERSION_NOTES.md`: "50ms non-overlapping bins (as specified in decoder task); spike counts divided by bin width (0.05s) to get rates in Hz; 80 time bins per trial. Reference code uses 100ms bandwidth with 50ms stride (Gaussian-weighted), but decoder task specifies 50ms bins." So the AI knowingly departs from the method paper's smoothed sliding histogram because the task instructions prescribe 50 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two unit-level filters:
- **QC**: keep only units with `units/classification == 'good'` (the spike-sorting classifier verdict of Chen, Liu et al. 2023). No individual metric thresholds.
- **Region annotation**: of those, keep only units whose `anno_name` matches one of ~300 hard-coded Allen-atlas prefixes belonging to 14 coarse regions (ALM, OtherCortex, Orbital, Striatum, Pallidum, Thalamus, Hypothalamus, Hippocampus, Midbrain, Pons, Medulla, Cerebellum, Olfactory, CorticalSubplate). Units with empty or unmatched annotations are dropped (~490 units, <1%).
- A session with no surviving units is skipped (this catches the one never-QC'd session whose `classification`/`anno_name` fields are empty).

57,925 units are kept over the 144 retained sessions (mean 402/session, range 90-923).

ii.
```python
good_mask = classification == 'good'

unit_regions = []
unit_indices = []
for i in range(len(classification)):
    if not good_mask[i]:
        continue
    region = map_anno_name_to_region(anno_names[i])
    if region is not None:
        unit_regions.append(region)
        unit_indices.append(i)

if len(unit_indices) == 0:
    print(f"  Skipping: no good units with valid regions")
    return None
```
```python
def map_anno_name_to_region(anno_name):
    if not anno_name or anno_name.strip() == '':
        return None
    for region, prefixes in REGION_MAPPING.items():
        for prefix in prefixes:
            if anno_name.startswith(prefix) or ...:
                return region
    return None
```

iii. `CONVERSION_NOTES.md`: "Classifier-based QC (`classification='good'`)... **Matches reference**: The reference code uses `qc_mode='classifier'` which applies trained logistic regression classifiers per brain area." The AI checked the resulting count (69,453 good units over 174 files) against the white paper's 69,943 / 173 before its session filter (step 29). The coarse region mapping was built by hand (steps 27-29) because the Allen hierarchy spreadsheet used by the reference code was unavailable; the AI chose the 14 categories to match the method paper's figure grouping and mapped "Secondary motor area" → ALM because these recordings target ALM.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go-cue onset. All NWB times share one session-absolute clock, so alignment is a subtraction: each unit's spike times minus that trial's `go_start_times` timestamp, then binned on the fixed go-cue-relative edge grid `[-2.5, 1.5]`. No resampling or interpolation.

ii.
```python
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)   # T_START=-2.5, T_END=1.5
...
rel_times = spike_times - go_cue_time
mask = (rel_times >= bin_edges[0]) & (rel_times < bin_edges[-1])
counts, _ = np.histogram(rel_times[mask], bins=bin_edges)
```

iii. The instructions specify go-cue alignment and the -2.5/+1.5 s window; `CONVERSION_NOTES.md` states "Aligned to go cue onset (t=0), window -2.5s to +1.5s... Reference code uses -3.0 to +3.0s or +3.5s, but decoder task specifies -2.5 to +1.5s." The AI confirmed that `go_start_times` has exactly one entry per trial before relying on it.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins per trial spanning -2.5 s to +1.5 s, identical for every trial and session. The grid is built once at module level with `np.linspace`. No rebinning, no overlapping/sliding windows, no Gaussian weighting — spikes go straight from raw times to the final bins. `metadata['time_bin_size'] = 50.0` ms, `off_start = -2.5`, `off_end = 1.5`.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
T_START = -2.5
T_END = 1.5
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

iii. Directly from the decoder task specification. The AI records the deviation from the method paper (100 ms bandwidth / 50 ms stride sliding histogram) in `CONVERSION_NOTES.md` under "Discrepancies from Reference", justified by the explicit instruction.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `/acquisition/BehavioralEvents/sample_start_times/timestamps` (the sample-epoch/tone onsets) together with the trial's go-cue time. Because an early lick replays the sample epoch, a trial can be preceded by several `sample_start` events; the AI takes the **last one before the go cue**.

ii.
```python
def get_tone_onset_relative_to_go(sample_start_times, go_cue_time):
    """The tone onset is the last sample_start before the go cue
    (accounts for early lick replays)."""
    before = sample_start_times[sample_start_times < go_cue_time]
    if len(before) > 0:
        return before[-1] - go_cue_time  # negative value
    return None
```

iii. Step 24: "the sample epoch has 3 tones (150 ms each, 100 ms ITI) = 650 ms; then delay = 1.2 s; so tone onset to go cue = 1.85 s (matches!). But early lick trials can have replayed sample epochs, so the last sample_start before go is the correct tone onset." The AI also verified that per-trial tone-to-go intervals vary because of replays.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A continuous, time-varying trace: for each bin, value = bin centre (relative to go cue) minus the tone onset (relative to go cue), i.e. seconds elapsed since tone onset, negative before the tone. It is stacked as row 0 of the `(2, 80)` input array and stored as `float64`. If no `sample_start` precedes the go cue, a fallback of -1.85 s (the nominal tone-to-go interval) is used. Observed range across the dataset: -1.5 to 11.9 s.

ii.
```python
tone_onset_rel = get_tone_onset_relative_to_go(sample_start_times, gc)
if tone_onset_rel is None:
    # Fallback: use typical value
    tone_onset_rel = -1.85
time_from_tone = BIN_CENTERS - tone_onset_rel  # time since tone onset at each bin
...
input_trial = np.stack([time_from_tone, photostim_binary], axis=0)  # (2, n_bins)
```

iii. Step 24: "Since I'm already aligned to go cue, if tone onset is at -1.85 s relative to go cue, then at any timepoint t the elapsed time since tone onset is t + 1.85. So at t = -2.5 s the value would be -0.65 s, meaning we're still 0.65 s before tone onset." The -1.85 s fallback is the nominal 0.65 s sample + 1.2 s delay derived from the methods text.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on `BIN_CENTERS`, the centres of exactly the same go-cue-relative 50 ms grid used to bin the spikes, so input bin *k* covers the same interval as neural bin *k* by construction. No separate alignment step or interpolation.

ii.
```python
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
...
time_from_tone = BIN_CENTERS - tone_onset_rel
```

iii. Not separately justified — it follows from using one shared bin grid for every stream, all expressed relative to the same trial go-cue time on the NWB global clock.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From the trials-table columns `photostim_onset` and `photostim_duration` (both stored as strings, with `'N/A'` on non-stimulated trials), plus `start_time` (photostim onset is measured from trial start) and the go cue (to re-express onset on the bin axis).

ii.
```python
ps_onset_str = np.array([p.decode() if isinstance(p, bytes) else p
                        for p in f['intervals/trials/photostim_onset'][:]])
ps_dur_str = np.array([p.decode() if isinstance(p, bytes) else p
                      for p in f['intervals/trials/photostim_duration'][:]])
trial_start_times = f['intervals/trials/start_time'][:]
```

iii. Found by direct inspection (steps 78-83). The AI initially treated `photostim_onset` as absolute, saw `photostim_on` stuck at 0 for all sample sessions, investigated ("the onset values are small (~2 s), but the go cue times are large (~134 s)... `photostim_onset` is probably relative to the trial start"), and confirmed that `trial_start + photostim_onset` puts the stimulus at ≈ -1.2 s relative to the go cue, i.e. in the late delay epoch as described in the papers.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1) time-varying trace: onset and offset are converted to go-cue-relative seconds and a bin is 1 when its centre lies in `[onset, offset)`. Trials with `'N/A'` onset stay all-zero; any parse failure is caught and also leaves the trace at zero. Stored as `float64` row 1 of the input array.

ii.
```python
photostim_binary = np.zeros(N_BINS, dtype=np.float64)
if ps_onset_str[ti] != 'N/A':
    try:
        ps_onset_val = float(ps_onset_str[ti])
        ps_dur = float(ps_dur_str[ti])
        ps_abs_onset = trial_start_times[ti] + ps_onset_val
        ps_start_rel = ps_abs_onset - gc
        ps_end_rel = ps_start_rel + ps_dur
        photostim_binary = ((BIN_CENTERS >= ps_start_rel) &
                           (BIN_CENTERS < ps_end_rel)).astype(np.float64)
    except (ValueError, TypeError):
        pass
```

iii. `CONVERSION_NOTES.md`: "Binary time series: 1 when photostim is on, 0 otherwise... Photostim typically occurs during late delay epoch (-1.2 s to -0.7 s relative to go cue). Duration: 0.5 s (last 0.5 s of delay including 100 ms ramp-down)." The instructions require photostim as a discrete, time-varying input. After the fix the AI re-verified that `photostim_on` spans [0, 1] rather than being identically zero (steps 93-95).

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Onset/offset are converted onto the go-cue-relative axis (`trial_start + onset − go_cue`) and compared against the same `BIN_CENTERS` used for the firing rates, so the binary trace is on exactly the neural bin grid.

ii.
```python
ps_abs_onset = trial_start_times[ti] + ps_onset_val
ps_start_rel = ps_abs_onset - gc
ps_end_rel = ps_start_rel + ps_dur
photostim_binary = ((BIN_CENTERS >= ps_start_rel) & (BIN_CENTERS < ps_end_rel)).astype(np.float64)
```

iii. Same rationale as 3-c: a single shared go-cue-relative grid for all streams. The AI validated the placement against the expected delay-epoch timing (-1.2 s).

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. From the trials-table column `trial_instruction` **alone** (`'left'`/`'right'`), i.e. the side the tone instructed the animal to lick. The `outcome` column is not consulted, so a `miss` trial (animal licked the *other* port) is labelled with the instructed side, and an `ignore` trial (animal never licked) is also labelled with the instructed side. Only two classes exist (`left`, `right`); there is no "no lick" class.

ii.
```python
instructions = np.array([i.decode() if isinstance(i, bytes) else i
                        for i in f['intervals/trials/trial_instruction'][:]])
...
# --- Output: choice ---
if instructions[ti] == 'left':
    choice = 0
else:
    choice = 1
output_choice.append(choice)
```

iii. `CONVERSION_NOTES.md`: "Choice (lick direction): 0 = left, 1 = right. Based on `trial_instruction` field (which port the animal should lick). Per-trial, broadcast across time bins." The AI's own plan (step 18) lists "Choice (left=0, right=1)" following the instruction wording it was given, and no step in the trajectory considers whether the animal's actual lick direction differs from the instruction. The AI did reason about `trial_instruction` elsewhere (step 29) only in the context of counting correct left/right trials for the session filter, where it correctly pairs instruction with `hit`.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The instruction string is mapped to 0/1 per trial and broadcast unchanged across all 80 bins into row 0 of the `(4, 80)` `int64` output array. `output_values[0] = ['left', 'right']`. The resulting class balance is 49.4% / 50.6%.

ii.
```python
out = np.array([
    np.full(N_BINS, output_choice[i], dtype=np.int64),       # choice (per-trial, broadcast)
    ...
], dtype=np.int64)  # (4, n_bins)
```
```python
'output_values': [
    ['left', 'right'],           # choice: 0=left, 1=right
    ...
]
```

iii. The 0/1 coding follows the instruction text the AI was given. Broadcasting per-trial values across bins keeps all four outputs in one `(n_output, n_timepoints)` array, satisfying the "make it time-varying if at all possible" formatting guidance. Integer dtype was adopted after the verifier failed on float outputs being used as indices into `output_values` (steps 52-54).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the trials-table `outcome` column, which already contains the strings `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcomes = np.array([o.decode() if isinstance(o, bytes) else o
                    for o in f['intervals/trials/outcome'][:]])
```

iii. No derivation needed — the stored categories are exactly the three the instructions ask for. The AI used the same column for its session-level performance computation.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. `'ignore'` → 0, `'miss'` → 1, anything else → 2 (`hit`), per trial, broadcast across all 80 bins into row 1. `output_values[1] = ['ignore', 'miss', 'hit']`. Dataset distribution: ignore 10.8%, miss 15.3%, hit 73.9%.

ii.
```python
if outcomes[ti] == 'ignore':
    outcome = 0
elif outcomes[ti] == 'miss':
    outcome = 1
else:  # hit
    outcome = 2
output_outcome.append(outcome)
```

iii. The 0/1/2 coding is given verbatim in the instructions. Because ignore/miss/hit are required output classes, the AI explicitly overrode the data paper's rule of discarding no-response trials (step 18).

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the trials-table `early_lick` column (strings `'no early'` / `'early'`).

ii.
```python
early_lick = np.array([e.decode() if isinstance(e, bytes) else e
                      for e in f['intervals/trials/early_lick'][:]])
```

iii. The flag is stored explicitly per trial, so no derivation is required; the same column is reused to define control trials for the session filter. Early-lick trials are retained rather than discarded because the instructions require early lick as a decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'early'` → 1, everything else → 0, per trial, broadcast across all 80 bins into row 2. `output_values[2] = ['no', 'yes']`. Distribution: 88.5% no, 11.5% yes.

ii.
```python
el = 1 if early_lick[ti] == 'early' else 0
output_early_lick.append(el)
```

iii. The 0/1 coding comes straight from the instructions; the value is per-trial so it is repeated across bins like the other trial-level outputs.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `/acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: the `(n_frames, 3)` DeepLabCut array (`tongue_x`, `tongue_y`, `tongue_likelihood`) with matching `timestamps`. The AI takes **column 1 only** (`tongue_y`); the likelihood column is read but never used.

ii.
```python
tongue_ts = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_data = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_y = tongue_data[:, 1]  # y-position
```

iii. This is the only tongue measurement in the files; the AI verified that all 174 sessions have tongue tracking (step 31-32) and identified the column layout from the series description. The methods text confirms DeepLabCut side-view tracking of tongue, jaw and nose at 300 Hz.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Per trial, frames within `[go − 2.5 s, go + 1.5 s]` are located with `searchsorted`, shifted to go-cue-relative time, and the **mean y of all frames** falling in each 50 ms bin is taken; bins containing no frame are `NaN`. **No likelihood/confidence filtering is applied** — frames where the tongue is retracted and DeepLabCut is reporting a low-confidence placeholder position are averaged in as if they were real tongue positions.

ii.
```python
def extract_tongue_y_for_trial(tongue_timestamps, tongue_y, go_cue_time, bin_edges):
    """Uses all tongue tracking data (no confidence filtering) to compute
    mean y-position per bin. NaN for bins with no tracking data."""
    tongue_y_binned = np.full(n_bins, np.nan)
    idx = np.searchsorted(tongue_timestamps, [t_abs_start, t_abs_end])
    ts_window = tongue_timestamps[idx[0]:idx[1]]
    y_window = tongue_y[idx[0]:idx[1]]
    ts_rel = ts_window - go_cue_time
    for b in range(n_bins):
        bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
        if np.any(bin_mask):
            tongue_y_binned[b] = np.mean(y_window[bin_mask])
    return tongue_y_binned
```

iii. This was a deliberate reversal made mid-run (steps 57-63). The AI first filtered at likelihood > 0.5, found ~79% of bins fell in the "middle" class because non-visible bins were defaulted there, and investigated: "Only 14% of tongue tracking data has high confidence... When using ALL data: p40 = 280.99, p60 = 281.24 (very narrow band)... When using high-conf only: p40 = 276.58, p60 = 290.01 (much wider band). So using confidence-filtered data for percentile computation is the right approach." It then reversed itself because with only three permitted classes it had nowhere to put non-visible bins: "the percentile definition guarantees a 40/20/40 split across the full session... I'm going to remove the confidence threshold and recompute using all available tongue y-position data." `CONVERSION_NOTES.md` records "All tracking data used (no confidence filtering)". The stated goal was an even 40/20/40 class balance, which it achieved (0.390 / 0.219 / 0.390).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three classes with per-session thresholds: the binned y-values from all kept trials of the session are pooled, the 40th and 60th percentiles of the non-NaN values become the two cut points, and each bin is labelled 0 (`< p40`), 1 (`p40 ≤ y ≤ p60`) or 2 (`> p60`). **Bins with no tracking data (NaN) are assigned class 1, "middle"** — there is no "not visible" class. If a session has no valid tongue values at all, every bin becomes class 1.

ii.
```python
all_tongue_y_vals = np.concatenate(tongue_y_all_trials)
valid_tongue_y = all_tongue_y_vals[~np.isnan(all_tongue_y_vals)]

if len(valid_tongue_y) > 0:
    p40 = np.percentile(valid_tongue_y, 40)
    p60 = np.percentile(valid_tongue_y, 60)

    tongue_y_discrete_trials = []
    for ty in tongue_y_all_trials:
        ty_disc = np.ones(N_BINS, dtype=np.int64)  # default to middle category
        valid_mask = ~np.isnan(ty)
        ty_disc[valid_mask & (ty < p40)] = 0
        ty_disc[valid_mask & (ty >= p40) & (ty <= p60)] = 1
        ty_disc[valid_mask & (ty > p60)] = 2
        tongue_y_discrete_trials.append(ty_disc)
else:
    tongue_y_discrete_trials = [np.ones(N_BINS, dtype=np.int64) for _ in range(len(trial_indices))]
```
```python
'output_values': [..., ['low', 'middle', 'high']]
```

iii. The 40th/60th percentile split and the per-session scope are taken verbatim from the instructions; the AI notes percentiles are "computed over all tongue y-position values across all trials in the session". Percentiles are taken over the 50 ms bin means (the same quantity that is discretised) rather than raw frames. Defaulting NaN bins to "middle" is described in the code comment as the fallback and was kept because, after dropping confidence filtering, "there should be very few NaN bins (only if no tracking data falls in a time bin)" (step 70). The AI validated the final class balance as "a perfect 40/20/40 split" (step 73).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock as spikes and events, so the trial's frame range is found by `searchsorted` at `go + T_START` and `go + T_END`, timestamps are converted to go-cue-relative time, and frames are assigned to bins using the same `BIN_EDGES` used for the firing rates. Bin *k* of the tongue output therefore covers the same interval as bin *k* of the neural matrix. No interpolation or offset correction.

ii.
```python
t_abs_start = go_cue_time + bin_edges[0]
t_abs_end = go_cue_time + bin_edges[-1]
idx = np.searchsorted(tongue_timestamps, [t_abs_start, t_abs_end])
ts_rel = tongue_timestamps[idx[0]:idx[1]] - go_cue_time
for b in range(n_bins):
    bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
```

iii. Not explicitly argued in the trajectory; it follows from the shared global clock and the single go-cue-relative bin grid used for every stream. The AI was aware that video is trial-gated (it handles the empty-window case by returning all-NaN).

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Five distinct cases:
- **Session never quality-controlled** (`classification`/`anno_name` empty): no units pass, session returns `None` and is skipped.
- **Trials outside the ephys recording**: excluded via the `obs_intervals` window test (1-e), after the AI discovered blocks of all-zero trials.
- **Units with missing/unrecognised anatomical annotation**: silently dropped (~490 units).
- **Missing tongue frames in a bin**: bin mean is `NaN` and is then relabelled as class 1 ("middle") — missing data is folded into a real category rather than flagged.
- **Missing tone onset / unparsable photostim strings**: tone onset falls back to the nominal -1.85 s; photostim parse errors are swallowed by `except: pass`, leaving the trial's stimulus trace all-zero.

ii.
```python
if len(unit_indices) == 0:
    print(f"  Skipping: no good units with valid regions")
    return None
```
```python
if tone_onset_rel is None:
    tone_onset_rel = -1.85          # fallback: typical value
```
```python
    except (ValueError, TypeError):
        pass
```
```python
ty_disc = np.ones(N_BINS, dtype=np.int64)  # default to middle category
valid_mask = ~np.isnan(ty)
```
```python
if len(ts_window) == 0:
    return tongue_y_binned          # all-NaN
```

iii. The AI's stated rationale is that data gaps must not surface as fabricated zeros in the neural matrix (steps 38-48, where it traced all-zero trials to the recording window and added the `obs_intervals` filter), while small behavioural/tracking gaps should not cost a whole trial. For the tongue NaNs it argued (step 70) that after removing confidence filtering "there should be very few NaN bins", so the middle-class default would be immaterial. The one residual verifier warning ("Session 0, trial 159: all neural data is zero") is dismissed in step 57 as "likely a genuine edge case at the boundary of the observation window".

## 10-a. What are the most time-consuming steps of the code?

i. The AI did not profile or document runtime. From the code and the run logs, the dominant cost is the nested per-trial × per-unit firing-rate loop: `compute_firing_rates` is called once per (trial, unit) pair — ≈ 74,759 × 402 ≈ 30 million calls — and each call subtracts the go cue from and masks the **entire** spike train of that unit before histogramming, so the cost is O(n_trials × n_units × n_spikes_per_unit) rather than O(total spikes). Secondary costs are the per-trial, per-bin Python loop in `extract_tongue_y_for_trial` (80 boolean masks over each trial's ~1,200 frames), the whole-array reads of `units/spike_times` (up to ~11 M doubles) and the tongue array (~930 k × 3), and pickling the 19.71 GB result. The full conversion ran from 02:51:44 to ≤ 03:15:06, i.e. ≈ 23 minutes for 144 sessions (~9.7 s/session), against 247 s (~1.4 s/session) for the reference implementation.

ii.
```python
for ti in trial_indices:
    ...
    for j, st in enumerate(unit_spike_times):
        fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
```
```python
def compute_firing_rates(spike_times, go_cue_time, bin_edges):
    rel_times = spike_times - go_cue_time          # touches every spike, every trial
    mask = (rel_times >= bin_edges[0]) & (rel_times < bin_edges[-1])
    counts, _ = np.histogram(rel_times[mask], bins=bin_edges)
```

iii. The AI never raises performance as a concern. It ran the full conversion as a background task and simply polled for completion (steps 96-108), and there is no reasoning in the trajectory about the cost of the nested loop or about alternatives such as `searchsorted` over a flattened edge array.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several, all of them avoidable:
- The **trial × unit firing-rate loop**: a single `np.searchsorted(spike_times, all_trial_edges)` per unit, followed by `np.diff`, would replace ~519 histogram calls per unit with one binary search over all 81 × n_trials edges (the approach the reference uses), removing both the per-trial rescan of the full spike train and the Python-level inner loop.
- The **per-bin loop inside `extract_tongue_y_for_trial`**: the 80 boolean masks could be replaced by one `np.floor((t − t0)/BIN)` index plus `np.bincount` for sums and counts.
- The **per-unit classification/region loop** in `process_session`: a vectorised `np.isin`/dictionary lookup over the `anno_name` array would do.
- The **per-unit `obs_intervals` loop**: the first/last interval per unit can be gathered by fancy indexing on the index vector.
- `unique_subjects.index(s)` and `brain_regions.index(r)` perform linear list searches inside loops; a dict lookup is O(1) (the reference uses dicts).
- Rebuilding `time_from_tone` and the `(4, 80)` output array inside the trial loop rather than as whole-session array operations.

ii.
```python
for b in range(n_bins):
    bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
    if np.any(bin_mask):
        tongue_y_binned[b] = np.mean(y_window[bin_mask])
```
```python
for i in range(len(classification)):
    if not good_mask[i]:
        continue
    region = map_anno_name_to_region(anno_names[i])
```
```python
idx = np.array([brain_regions.index(r) for r in regions])
```

iii. No justification appears in the trajectory — the loops are written in the most direct form and performance is never discussed. The AI's priority throughout was format validity and decoder accuracy, and it accepted the ~23-minute conversion without comment.

## 10-c. What processing does the code repeat multiple times?

i. Real repeated work, mostly inside the trial loop:
- Each unit's **entire spike train is re-shifted and re-masked on every trial** (`rel_times = spike_times - go_cue_time` over all spikes, ~519 times per unit per session) instead of being searched once against all trial edges.
- `bin_width = bin_edges[1] - bin_edges[0]` is recomputed on each of ~30 M calls, and the bin-edge comparison arrays are rebuilt per trial for the tongue binning.
- The trials table columns are read once (good), but `is_control`, hit counts and performance are computed over all trials and then the trial mask is recomputed separately.
- `unique_subjects.index()` / `brain_regions.index()` re-scan the name lists once per session / per unit.
- Each NWB file is opened only once, and the tongue percentile thresholds are computed in the same pass, so there is no second pass over the data.

ii.
```python
def compute_firing_rates(spike_times, go_cue_time, bin_edges):
    rel_times = spike_times - go_cue_time      # recomputed for every trial
    ...
    bin_width = bin_edges[1] - bin_edges[0]    # recomputed every call
```
```python
for b in range(n_bins):
    bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
```

iii. Not discussed in the trajectory. The structure follows from writing the conversion trial-by-trial rather than session-at-a-time; correctness of the per-trial result is unaffected.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Little computation is wasted, but some is:
- The full tongue array (`n_frames × 3`) is loaded and the **likelihood column is read and then never used** (after the AI removed confidence filtering); `tongue_x` is likewise loaded and discarded.
- `np.searchsorted` bounds and the per-bin masks are computed for bins that always end up overwritten by the "middle" default when empty.
- The session-level performance statistics (`is_control`, `control_hits_left/right`, `performance`) are computed for every session including those already doomed by other criteria, and the end-of-run region histogram / summary statistics are printed but not stored.
- Memory rather than compute: neural rates are kept as `float64` and the outputs as `int64`, where `float32`/`int8` would suffice. That roughly doubles the payload (19.71 GB written, versus 11.9 GB for the equivalent reference output) and makes the pickle write substantially longer.
- Nothing computed is left out of the saved dictionary, and no field is computed twice for the output.

ii.
```python
tongue_data = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_y = tongue_data[:, 1]  # y-position   (columns 0 and 2 unused)
```
```python
fr_matrix = np.zeros((n_units, N_BINS), dtype=np.float64)
...
out = np.array([...], dtype=np.int64)  # (4, n_bins)
```

iii. Not discussed in the trajectory. The likelihood column became dead weight only after the mid-run decision to stop confidence-filtering (steps 57-66); the dtype choices are defaults that were never revisited, and the AI's only dtype change was float → int on the outputs, forced by the verifier's indexing requirement.
