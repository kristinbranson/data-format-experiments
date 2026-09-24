# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is one NWB file per session, stored under `/app/data/sub-<subject_id>/`. The AI enumerates the `sub-*` directories with `os.listdir`, sorts them, then sorts the `*.nwb` files inside each one, producing a list of `(subject_dir_name, path)` tuples. This list (174 files, 28 subjects) is iterated once, and each file is opened with `pynwb.NWBHDF5IO(path, 'r')` and read in full. Within a file, everything is pulled from the standard NWB containers: `nwb.units` (spike times, `classification`, `anno_name`, `obs_intervals`), `nwb.trials` (per-trial table), `nwb.acquisition['BehavioralEvents']` (go cue, sample onsets, photostim start/stop) and `nwb.acquisition['BehavioralTimeSeries']` (tongue tracking). `--sample` truncates the file list to the first 2 entries; `--full` (default) uses all of them.

ii.
```python
def get_nwb_files(data_dir):
    """Get list of all NWB files organized by subject."""
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    nwb_files = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for fname in files:
            nwb_files.append((subj, os.path.join(subj_dir, fname)))
    return nwb_files
```

```python
nwb_files = get_nwb_files(DATA_DIR)
print(f"Found {len(nwb_files)} NWB files from {len(set(s for s, _ in nwb_files))} subjects")
...
for i, (subject_id, nwb_path) in enumerate(nwb_files):
    result = process_session(nwb_path, subject_id)
```

```python
io = NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
```

iii. From CONVERSION_NOTES.md Step 2: "NWB files organized by subject (sub-XXXXXX directories); Each NWB file = one session". The AI records in Step 0 that the data directory contains "28 subjects (sub-440956 through sub-484677), 174 NWB files total" and cross-checks this against `dandiset.yaml` and against the papers' "173 behavioral sessions / 28 mice". The reference code (`preprocessing_DJ_2022Aug.py`) reads `.mat` DataJoint exports; the AI notes in Step 1 that "Reference code processes .mat files from DataJoint export; our data is in NWB format", so it uses `pynwb` as the equivalent reader rather than reusing the reference loader.

## 1-b. How are the data split into subjects?

i. The subject identity of a session is taken from its containing directory name (`sub-440956`), not from `nwb.subject.subject_id`. Subjects are registered in a dict in order of first appearance while iterating the (already alphabetically sorted) file list, so `subjects` ends up sorted. Each kept session appends its subject's integer index to `subject_idx`. Because directories are sorted and each subject's files are contiguous, this reproduces the sorted-unique ordering. Result: 28 subjects, 3–10 sessions each.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
...
nwb_files.append((subj, os.path.join(subj_dir, fname)))
```

```python
# Track subjects
if subject_id not in subjects_seen:
    subjects_seen[subject_id] = len(subjects_seen)
    subjects_list.append(subject_id)
subj_idx = subjects_seen[subject_id]
...
subject_idx_list.append(subj_idx)
```

```python
'subjects': subjects_list,
'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. CONVERSION_NOTES.md Step 5 maps `subject_id → subjects/subject_idx` with the note "From NWB metadata". The directory name is derived from the NWB `subject_id` (the DANDI `sub-<id>` convention), so the AI treats the folder as the canonical animal grouping. It validates the result against the papers: Step 9 records "Subjects | 28 | 28 | 28 | YES", and the decoder verification log lists 28 subjects with 3–10 sessions each.

## 1-c. How are the data split into sessions?

i. One NWB file is one session; no grouping or boundary inference is done. Session order in the output is the sorted file order (subject directory, then filename, which embeds the acquisition timestamp, so sessions are chronological within a subject). Each session is identified by the NWB file's basename, stored as `session_name` in the per-session dict. 173 of 174 files reach the output; the one dropped (`sub-440958_ses-20190216T162508`) has no `classification == 'good'` units. Note: `session_name` is used only for the plot filenames and console output — the AI's `metadata` dict does **not** contain a `session_info` field.

ii.
```python
files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
```

```python
return {
    ...
    'session_name': os.path.basename(nwb_path),
}
```

```python
'metadata': {
    ...
    'n_sessions': len(all_sessions),
    'n_subjects': len(subjects_list),
```

iii. CONVERSION_NOTES.md Step 2: "Each NWB file = one session". Step 2 also flags "1 session has 0 good units (sub-440958_ses-20190216T162508) - will be excluded", and Step 9 records the consistency check "Sessions | 173 | 174 NWB (1 no good units) | 173 | YES" against the data paper's "173 behavioral sessions". Step 4 explicitly decides *not* to apply the data paper's behavioural session-selection criteria (>65% correct, ≥50 correct trials each direction), on the grounds that only 151 sessions would pass, whereas the paper reports 173, so those criteria are analysis-specific rather than dataset-inclusion criteria.

## 1-d. How are the data split into trials?

i. Trials are the rows of the NWB trials table (`nwb.trials`). The per-trial scalar columns (`outcome`, `trial_instruction`, `early_lick`, `auto_water`, `free_water`) are read as whole arrays, and the go cue for trial `ti` is taken as `go_start_times[ti]` — i.e. the code assumes a strict one-to-one, order-preserving correspondence between rows of the trials table and entries of the `go_start_times` event stream. No assertion checks this. Everything downstream (firing rates, inputs, outputs) is computed per surviving trial index in a `for ti in trial_indices:` loop.

ii.
```python
n_trials_total = len(nwb.trials)
...
trials = nwb.trials
outcomes = trials['outcome'][:]
instructions = trials['trial_instruction'][:]
early_lick = trials['early_lick'][:]
auto_water = trials['auto_water'][:]
free_water = trials['free_water'][:]

events = nwb.acquisition['BehavioralEvents']
go_start_times = events.time_series['go_start_times'].timestamps[:]
sample_start_times = events.time_series['sample_start_times'].timestamps[:]
```

```python
for ti in trial_indices:
    go_cue = go_start_times[ti]
```

iii. CONVERSION_NOTES.md Step 2 lists `trials` as a top-level NWB container with one row per behavioural trial. The AI explicitly verified in the trajectory (step 54) that `go_start_times` has exactly as many entries as trials ("go_start_times has 368 entries matching the trial count"), unlike `sample_start_times` (405) and `delay_start` (395), which have extra entries because "early lick trials cause replays". Because the go-cue stream is 1:1 with trials, positional indexing is safe, and the AI relies on that rather than re-deriving trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. Three filters, intersected:

1. **No spike data** — trials outside the units' `obs_intervals`. For each good unit the AI reads `obs_intervals`, groups them by *number* of intervals (assuming units with the same count share the same interval set), and keeps a trial only if its `start_time` is within 1.0 s of some interval start, for **every** distinct interval set.
2. **`free_water == 0`** — free-reward trials removed.
3. **`auto_water == 0`** — auto-reward trials removed as well.

A session with fewer than 2 surviving trials is dropped. No behavioural filter is applied: early-lick, `ignore` (no-response) and photostim trials are all deliberately kept because they are required decoder outputs/inputs. Result: 89,546 trials kept out of 94,990 (5.7% removed).

ii.
```python
def get_valid_trial_indices(nwb, good_unit_indices, n_trials):
    trial_starts = nwb.trials['start_time'][:]
    obs_sets = {}
    for idx in good_unit_indices:
        obs = nwb.units['obs_intervals'][idx]
        n_obs = len(obs)
        if n_obs not in obs_sets:
            obs_sets[n_obs] = obs

    valid_trials = np.ones(n_trials, dtype=bool)
    for n_obs, obs in obs_sets.items():
        obs_starts = obs[:, 0]
        diffs = np.abs(trial_starts[:, None] - obs_starts[None, :])
        min_diffs = diffs.min(axis=1)
        covered = min_diffs < 1.0
        valid_trials &= covered
    return np.where(valid_trials)[0]
```

```python
# ---- Filter trials: exclude auto_water, free_water, and trials without recording ----
trial_mask = (auto_water == 0) & (free_water == 0)
recording_mask = np.zeros(n_trials_total, dtype=bool)
recording_mask[valid_trial_idx] = True
trial_mask = trial_mask & recording_mask
trial_indices = np.where(trial_mask)[0]

if len(trial_indices) < 2:
    io.close()
    return None
```

iii. CONVERSION_NOTES.md Step 3/4/5: the reference analysis function `get_regular_trial_mask` excludes "early lick, auto water, free water, no response, stim". The AI's Step 4 resolution: "For our decoder: keep early lick (output), photostim (input), ignore (outcome=0). Exclude only auto_water and free_water", and Step 3 adds the rationale that auto/free water trials "confound behavior, not decoder variables". The `obs_intervals` filter was added after sample validation: Step 7 records "Key fix: Added `obs_intervals` filtering to exclude trials without valid neural recording." The <2-trial session drop follows the target-format requirement ("There needs to be at least two trials within each session"). Step 9 reports 2 remaining all-zero-neural trials out of 89,546 ("Negligible").

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `nwb.units['spike_times']` — absolute (session-clock) sorted spike times — read one unit at a time for the units that pass QC. The second input is `BehavioralEvents/go_start_times`, which sets the window for each trial. `units['classification']` selects which units contribute, and `units['anno_name']` is used for the brain-region index (and, in principle, as a second neuron filter).

ii.
```python
classification = nwb.units['classification'][:]
good_mask = np.array([c == 'good' for c in classification])
...
good_unit_indices = np.where(good_mask)[0]
...
spike_times_per_unit = []
for idx in good_unit_indices:
    st = nwb.units['spike_times'][idx]
    spike_times_per_unit.append(st)
```

```python
go_start_times = events.time_series['go_start_times'].timestamps[:]
```

iii. CONVERSION_NOTES.md Step 2: "Spike times are ABSOLUTE (not aligned to go cue)". Step 5 maps "units.spike_times (good) → neural | Align to go cue, bin at 50ms, -2.5 to 1.5s | `sliding_histogram`". Spike times are the only neural representation stored in the NWB files, so firing rates are computed directly from them.

## 2-b. How is the `neural` data processed?

i. Spike counts per 50 ms bin, converted to Hz. For each trial, and within that for each good unit, the code boolean-masks the unit's **entire** session spike train to the trial window `[go-2.5, go+1.5)`, converts the surviving spike times to bin indices by floor division, accumulates counts with `np.add.at`, and finally divides the whole `(n_neurons, 80)` matrix by the bin width. No smoothing, no normalisation, no baseline subtraction, no firing-rate threshold. Output dtype is `float32`.

ii.
```python
def compute_firing_rates(spike_times_list, go_cue_time, t_start, t_end, bin_size):
    n_neurons = len(spike_times_list)
    n_bins = int((t_end - t_start) / bin_size)
    fr = np.zeros((n_neurons, n_bins), dtype=np.float32)

    abs_start = go_cue_time + t_start
    abs_end = go_cue_time + t_end

    for i, st in enumerate(spike_times_list):
        if len(st) == 0:
            continue
        mask = (st >= abs_start) & (st < abs_end)
        spikes_in_window = st[mask]
        if len(spikes_in_window) == 0:
            continue
        bin_indices = ((spikes_in_window - abs_start) / bin_size).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        np.add.at(fr[i], bin_indices, 1.0)

    fr /= bin_size
    return fr
```

```python
for ti in trial_indices:
    go_cue = go_start_times[ti]
    fr = compute_firing_rates(spike_times_per_unit, go_cue, T_START, T_END, BIN_SIZE_S)
    neural_trials.append(fr)
```

iii. CONVERSION_NOTES.md Step 3 records the reference preprocessing parameters ("bw=0.04s (40ms), stride=0.0034s, begin=-3.0s, end=3.0s") and Step 4 resolves the difference: "Our task uses 50ms bins (non-overlapping). Different from reference but required by task spec." The reference `sliding_histogram(..., rate=True)` likewise returns counts divided by bin width, so the Hz convention matches. Step 3 notes the method paper's 2 Hz firing-rate cut-off and Step 4 rejects it: "This is analysis-specific for video prediction. Not applying to our decoder." Step 10 sanity-checks the result: "Mean 9.2 Hz, max 940 Hz (reasonable for 50ms bins)".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two neuron-level filters:

1. `units['classification'] == 'good'` — the spike-sorting QC classifier verdict. Units labelled `'unlabelled'` (or with a non-string/NaN entry, which compares False) are dropped. This keeps 69,453 of 272,227 units (25.5%), mean 401 per session, range 90–923.
2. **Histology/annotation filter** — each good unit's `anno_name` is passed through a hand-written substring mapper onto 14 coarse regions; a unit whose annotation matches nothing is marked invalid and dropped. In practice this removed 0 units (the 14 region counts sum exactly to 69,453).

A session with zero good units is dropped entirely (1 session). No thresholds are applied to any individual QC metric (SNR, ISI violations, presence ratio, etc.), and no firing-rate threshold is applied.

ii.
```python
classification = nwb.units['classification'][:]
good_mask = np.array([c == 'good' for c in classification])
n_good = good_mask.sum()

if n_good == 0:
    io.close()
    return None
```

```python
anno_names = nwb.units['anno_name'][:][good_mask]

region_indices = []
valid_neuron_mask = np.ones(n_good, dtype=bool)
for i, anno in enumerate(anno_names):
    region = map_anno_to_region(str(anno))
    if region is None:
        valid_neuron_mask[i] = False
        region_indices.append(-1)
    else:
        region_indices.append(COARSE_REGIONS.index(region))
region_indices = np.array(region_indices)

good_unit_indices = np.where(good_mask)[0]
good_unit_indices = good_unit_indices[valid_neuron_mask]
region_indices = region_indices[valid_neuron_mask]
n_neurons = len(good_unit_indices)

if n_neurons == 0:
    io.close()
    return None
```

iii. CONVERSION_NOTES.md Step 1: "**QC mode**: 'classifier' - region-specific logistic regression classifiers trained on manual labels... In NWB: `classification == 'good'` corresponds to QC classifier pass". Step 3 curation rules: "QC classifier pass: `classification == 'good'`... Must have histology (anno_name not empty) → this gives CCF brain region", mirroring the reference `helper_get_neuron_id_area`, which filters neurons by region + side + QC classifier. Step 2 notes `unit_quality` ('good'/'multi', the Kilosort label) exists but is not used. The AI checked that "all good units have anno_name" (trajectory step 86), and reconciled 69,453 against the methods' 69,943 as a "0.7% diff... acceptable".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB streams share one session-absolute clock, so alignment is just a window lookup: for each trial, the go-cue time is `go_start_times[ti]`, and the absolute window is `[go_cue - 2.5, go_cue + 1.5)`. Spikes inside that window are re-expressed as offsets from `go_cue + T_START` and binned. No interpolation, no resampling, no per-stream offset correction.

ii.
```python
go_cue = go_start_times[ti]
fr = compute_firing_rates(spike_times_per_unit, go_cue, T_START, T_END, BIN_SIZE_S)
```

```python
abs_start = go_cue_time + t_start
abs_end   = go_cue_time + t_end
mask = (st >= abs_start) & (st < abs_end)
spikes_in_window = st[mask]
bin_indices = ((spikes_in_window - abs_start) / bin_size).astype(int)
```

```python
'temporal_alignment_event': 'Go cue onset',
'off_start': T_START,
'off_end': T_END,
```

iii. The Decoder Task section of the instructions specifies "Temporally align based on **Go cue onset**". CONVERSION_NOTES.md Step 2 establishes that "Spike times are ABSOLUTE (not aligned to go cue)" whereas "in the original .mat data [they are] already aligned to go cue" (Step 1) — so in NWB the alignment must be done explicitly. Step 3 lays out the task structure relative to the go cue (presample ≈ −2.56 s, sample/tone ≈ −1.85 s, delay 1.2 s, go cue 0, answer window 1.5 s), confirming that a −2.5 → +1.5 s window covers sample, delay and response.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins per trial spanning −2.5 s to +1.5 s relative to the go cue. The grid is identical for every trial and every session, and is shared by the neural data, both inputs, and the tongue output. This is a **change from the reference pipeline**, which used a sliding 40 ms window with a 3.4 ms stride over −3 → +3 s; the AI does not rebin the reference output but recomputes directly from raw spike times at the new resolution. `metadata['time_bin_size']` is recorded in ms.

ii.
```python
BIN_SIZE_S = 0.05  # 50 ms bins
T_START = -2.5     # seconds before go cue
T_END = 1.5        # seconds after go cue
N_TIMEBINS = int((T_END - T_START) / BIN_SIZE_S)  # 80
```

```python
'time_bin_size': BIN_SIZE_S * 1000,  # in ms
'off_start': T_START,
'off_end': T_END,
```

iii. CONVERSION_NOTES.md Step 3: "**Spike binning (reference)**: 40ms Gaussian kernel width, 3.4ms stride (sliding histogram, NOT Gaussian smoothing). **Our task requires**: 50ms bins, aligned -2.5s to 1.5s relative to go cue → 80 time bins". Step 4 records the discrepancy and its resolution ("Different from reference but required by task spec"), and Step 5 Key Decisions 4 and 5 restate it. The verification log confirms "T: mean: 80.00, median: 80.00, min: 80, max: 80".

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents/sample_start_times` (the sample-epoch/tone onsets) together with the trial's go-cue time. Because `sample_start_times` has more entries than trials (early licks replay the sample epoch), the tone for a trial is defined as the **last** sample onset strictly before that trial's go cue.

ii.
```python
sample_start_times = events.time_series['sample_start_times'].timestamps[:]
```

```python
def find_last_sample_before_go(sample_start_times, go_cue_time):
    """Find the last sample onset time before a given go cue time."""
    valid = sample_start_times[sample_start_times < go_cue_time]
    if len(valid) == 0:
        return None
    return valid[-1]  # Last one before go cue
```

```python
tone_onset = find_last_sample_before_go(sample_start_times, go_cue)
if tone_onset is None:
    tone_onset = go_cue - 1.85
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 6: "**Tone onset**: Last sample_start_time before each trial's go cue (accounting for early lick replays)." The trajectory (step 54) documents the discovery: "The sample_start_times has more entries than trials (405 vs 368) - this is because early lick trials cause replays, so there are more sample events than trials", and step 76 reasons that the trial "eventually completes, and I need to match each trial's go_cue to its final sample_start_time before that cue". The `go_cue - 1.85` fallback comes from the task structure documented in Step 3 ("Sample epoch (tone)... starts ~-1.85s").

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A continuous, time-varying value: for each of the 80 bins, the bin centre (relative to the go cue) minus the tone's offset from the go cue. Equivalently, `bin_centre + (go_cue − tone_onset)`. Bin centres are `arange(80)*0.05 - 2.5 + 0.025`. Values are negative before the tone and positive after; stored as `float32`. Observed range over the full dataset: [−1.5, 11.9] s (large positive values arise on trials where the last sample onset is many seconds before the go cue, e.g. after repeated replays).

ii.
```python
def compute_time_from_tone(go_cue_time, tone_onset_time, t_start, t_end, bin_size):
    n_bins = int((t_end - t_start) / bin_size)
    bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
    tone_rel = tone_onset_time - go_cue_time  # tone onset relative to go cue
    time_from_tone = bin_centers - tone_rel   # time since tone onset
    return time_from_tone.astype(np.float32)
```

```python
input_data = np.stack([time_from_tone, photostim_ts], axis=0)
input_trials.append(input_data)
...
'input_names': ['time_from_tone_onset', 'photostim_on'],
```

iii. The instructions list the input as "Time from **tone onset** in seconds (continuous, time-varying)". CONVERSION_NOTES.md Step 5 maps "sample_start_times → input[0]: time_from_tone | Continuous time from last sample onset before go cue | Time-varying". The AI sanity-checked the range in Step 5 ("Time from tone onset values should be roughly -1.85 to 1.5s relative to go cue") and flagged the outliers in Step 10: "**time_from_tone_onset max=7.94**: Some trials have very early or misdetected tone onsets. Not critical for decoder." (The final full-dataset max is 11.9 s.)

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on exactly the same grid as the firing rates — the 80 bin centres of the go-cue-aligned window — so bin *k* of the input covers the same interval as bin *k* of the neural matrix. No separate alignment step or interpolation is needed; the go cue is the common origin and the tone offset is simply added.

ii.
```python
bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
```
(identical expression to the neural grid `abs_start = go_cue + t_start`, `bin_indices = (spikes - abs_start)/bin_size`)

```python
time_from_tone = bin_centers - tone_rel
```

```python
time_axis = np.arange(N_TIMEBINS) * BIN_SIZE_S + T_START + BIN_SIZE_S / 2   # plotting
```

iii. Not separately justified in CONVERSION_NOTES.md beyond Step 5's statement that the input is "Time-varying" on the go-cue-aligned window. The `--show-processing` plots overlay the input trace on the same `time_axis` as the neural raster with a dashed line at the go cue, which the AI used as the visual check that there is no temporal misalignment (Step 7: "Processing plots saved for both sessions. Visual inspection OK.").

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times` — absolute session-clock timestamps of each photostimulation epoch, one pair per stimulated trial. The AI does **not** use the trials-table columns `photostim_onset` / `photostim_duration` (which store the same information as strings relative to trial start, with `'N/A'` on non-stim trials). The two representations are exactly equivalent: `start_time + float(photostim_onset)` reproduces `photostim_start_times` to the printed precision.

ii.
```python
# Photostim event times (absolute)
photostim_start_abs = events.time_series['photostim_start_times'].timestamps[:]
photostim_stop_abs  = events.time_series['photostim_stop_times'].timestamps[:]
```

```python
trial_abs_start = go_cue + T_START
trial_abs_end = go_cue + T_END
stim_mask = (photostim_stop_abs > trial_abs_start) & (photostim_start_abs < trial_abs_end)
ps_starts = photostim_start_abs[stim_mask]
ps_stops = photostim_stop_abs[stim_mask]
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 8: "**Photostim input**: Binary time series - 1 during photostim, 0 otherwise. Use absolute photostim_start/stop_times aligned to go cue." Step 2 notes that the trials-table onsets are "relative to trial start", and the trajectory (step 54) shows the AI computing `start_time + photostim_onset` and cross-checking it against the go cue ("this puts photostim around 1.16 seconds before the go cue"), i.e. it was aware of both sources and chose the already-absolute event stream to avoid the relative-time conversion and the `'N/A'` string handling.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1) time series over the 80 bins, not a per-trial flag. For each stim epoch overlapping the trial window, bins whose **centre** lies in `[start, stop]` are set to 1. Non-stimulated trials get an all-zero row (the loop body never executes). Stored as `float32` alongside the time-from-tone row in a `(2, 80)` array. 20.0% of retained trials carry photostimulation.

ii.
```python
def compute_photostim_timeseries(go_cue_time, photostim_starts, photostim_stops,
                                  t_start, t_end, bin_size):
    n_bins = int((t_end - t_start) / bin_size)
    bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
    abs_centers = bin_centers + go_cue_time

    stim = np.zeros(n_bins, dtype=np.float32)
    for start, stop in zip(photostim_starts, photostim_stops):
        mask = (abs_centers >= start) & (abs_centers <= stop)
        stim[mask] = 1.0
    return stim
```

iii. The instructions require "Whether **photostimulation** is on at every time point (discrete, time-varying)" and "If an input is a time such as onset of some stimulus, represent it as a binary time series." CONVERSION_NOTES.md Step 5 Key Decision 8 states exactly this. Step 3 records the expected timing and prevalence from the papers — "**Photostim**: last 0.5s of delay epoch (~-0.5s to 0s), bilateral/unilateral ALM silencing" and "Photostim fraction | ~25% of trials (in subset of 17 VGAT mice)" — and Step 9 checks the result: "Photostim trials | ~25% (VGAT mice) | 20.0% | Reasonable".

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The 80 bin centres (relative to the go cue) are converted to absolute times by adding the trial's go-cue time, and compared directly against the absolute photostim start/stop timestamps. This is the same grid used for the firing rates, just expressed on the absolute clock instead of relative to the go cue, so bin *k* again covers the same interval in both streams. A pre-filter keeps only stim epochs that overlap `[go-2.5, go+1.5)`.

ii.
```python
abs_centers = bin_centers + go_cue_time
...
mask = (abs_centers >= start) & (abs_centers <= stop)
```

```python
stim_mask = (photostim_stop_abs > trial_abs_start) & (photostim_start_abs < trial_abs_end)
```

iii. All NWB streams are on one session clock (CONVERSION_NOTES.md Step 2), so no offset correction is needed. Step 5 Key Decision 8: "Use absolute photostim_start/stop_times aligned to go cue." The `--show-processing` plot draws the photostim trace as a filled region on the shared `time_axis` under the neural raster, which is the AI's visual alignment check.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. **Only** `trials['trial_instruction']`, the string `'left'`/`'right'` describing which side the tone *instructed* the animal to lick. The `outcome` column is not consulted when computing choice, and no lick-time stream (`left_lick_times` / `right_lick_times`, which the AI catalogued in Step 2) is used. As a consequence the variable stored under the name `choice` is the instructed direction, not the animal's actual lick direction, and there is no `'no lick'` category.

ii.
```python
instructions = trials['trial_instruction'][:]
```

```python
# Output 0: choice (left=0, right=1)
choice = 0 if instructions[ti] == 'left' else 1
```

```python
'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
'output_values': [
    ['left', 'right'],
    ...
```

iii. CONVERSION_NOTES.md Step 5 variable mapping: "trial_instruction | output[0]: choice | left=0, right=1 | `trial_type` in ref code | Per-trial". The justification given is the correspondence with the reference code's `trial_type` variable. Neither CONVERSION_NOTES.md nor the trajectory contains any discussion of the instructions' third choice category ("no lick"), nor of the fact that on `miss` trials the animal licks the side opposite to the instruction, nor of `ignore` trials where no lick occurs. Step 5's planned sanity check is only "Choice: left ~48.5%, right ~51.5%", i.e. a check on the instruction distribution rather than on lick direction.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. A single ternary expression per trial: `'left' → 0`, anything else → `1`. The scalar is then broadcast across all 80 bins of row 0 of the `(4, 80)` int64 output array, so it is stored as a constant time series. Two categories are declared in `output_values[0]`. Full-dataset distribution: left 48.6%, right 51.4%.

ii.
```python
choice = 0 if instructions[ti] == 'left' else 1
```

```python
# Output shape: (4, n_timebins) - all time-varying
# Per-trial outputs are replicated across time
full_output = np.zeros((4, N_TIMEBINS), dtype=np.int64)
full_output[0, :] = out_dict['choice']
```

```python
'output_values': [
    ['left', 'right'],
```

iii. The instructions say outputs "Can be time-varying or discrete values per trial. If at all possible, make it time-varying", and require a single `(n_output, n_timepoints)` array; the AI therefore replicates the per-trial scalar across bins so that all four outputs live in one array (comment in the code: "Per-trial outputs are replicated across time"). Step 9 checks the distribution against the data ("Choice L/R | ~50/50 | 48.5/51.5% | 48.6/51.4% | YES"). Step 12 interprets the resulting accuracy (0.71 validation) as "Reasonable for whole-brain decoding", comparing it to the reference paper's ~0.90 AUC for choice decoding in ALM.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `trials['outcome']` directly — the trials table already stores one of the three strings `'hit'`, `'miss'`, `'ignore'` per trial, which is exactly the three-way categorisation the instructions ask for.

ii.
```python
outcomes = trials['outcome'][:]
...
outcome_str = outcomes[ti]
```

iii. CONVERSION_NOTES.md Step 2 lists "outcome (hit/miss/ignore)" as a trials-table column, and Step 5 maps "outcome | output[1]: outcome | ignore=0, miss=1, hit=2 | `correctness` in ref code | Per-trial" — i.e. the AI identifies it with the reference code's `correctness` variable. No derivation is needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary `{'ignore': 0, 'miss': 1, 'hit': 2}` applied with `.get(outcome_str, 0)`, so any unexpected string would silently fall back to `0` (`ignore`). The resulting code is broadcast across all 80 bins into row 1 of the output array. Distribution: ignore 14.8%, miss 16.7%, hit 68.5%.

ii.
```python
# Output 1: outcome (ignore=0, miss=1, hit=2)
outcome_str = outcomes[ti]
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}.get(outcome_str, 0)
```

```python
full_output[1, :] = out_dict['outcome']
```

```python
'output_values': [
    ...
    ['ignore', 'miss', 'hit'],
```

iii. The code assignment follows the instructions verbatim ("**Outcome** (ignore, miss, hit, per-trial)"). The broadcast-across-bins convention is the same as for the other per-trial outputs. Step 9 validates the distribution against the data paper's 84% correct rate: "Hit rate (all trials) | 84% correct | 68.7% overall | 68.5% overall" and "Hit rate (non-ignore) | 84% | 80.4% | Close (incl. early lick)" — the AI attributes the residual gap to keeping early-lick trials that the paper excludes.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. `trials['early_lick']`, a string column with values `'no early'` / `'early'` flagging whether the animal licked during the sample or delay epoch.

ii.
```python
early_lick = trials['early_lick'][:]
...
early_val = 0 if early_lick[ti] == 'no early' else 1
```

iii. CONVERSION_NOTES.md Step 2 lists `early_lick` as a trials-table column; Step 5 maps "early_lick | output[2]: early_lick | no=0, yes=1 | `early_lick_trials` in ref code | Per-trial". The flag is stored explicitly, so no derivation from lick times is needed. Step 3/4 also record the decision to *keep* these trials rather than exclude them as the reference analysis does, precisely because early lick is a required decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'no early' → 0`, anything else → `1`; broadcast across all 80 bins into row 2 of the output array. Distribution: no 88.4%, yes 11.6%.

ii.
```python
# Output 2: early lick (no=0, yes=1)
early_val = 0 if early_lick[ti] == 'no early' else 1
```

```python
full_output[2, :] = out_dict['early_lick']
```

```python
'output_values': [
    ...
    ['no', 'yes'],
```

iii. Code assignment follows the instructions ("**Early lick** (no, yes, per-trial)"). Step 5 planned sanity check "Early lick fraction ~11.4%" and Step 9 confirms "Early lick | ~11% | 11.4% | 11.6% | YES". Step 12 notes the class imbalance is handled by the decoder's balanced loss: "Strong despite high class imbalance (88.4% no early lick)."

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`. Its `data` is `(n_frames, 3)` = `(tongue_x, tongue_y, tongue_likelihood)` with matching `timestamps` at ~294 Hz on the session clock. Column 1 (`tongue_y`) supplies the value and column 2 (the DeepLabCut likelihood) gates whether a frame counts. Column 0 (`tongue_x`) is read but unused.

ii.
```python
tongue_ts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts.data[:]  # (n_frames, 3): x, y, likelihood
tongue_timestamps = tongue_ts.timestamps[:]
tongue_y_all = tongue_data_all[:, 1].astype(np.float32)
tongue_likelihood = tongue_data_all[:, 2].astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 2: "Tongue tracking at ~300 Hz, 3 columns: (tongue_x, tongue_y, tongue_likelihood)" — the layout is taken from the series' own `description` attribute rather than assumed. Trajectory step 54: "The tongue tracking has 3 columns: x, y, likelihood. Column 2 is the likelihood (0-1) from DeepLabCut." Step 5 Key Decision 7: "**Tongue y-position**: Use column 1 (tongue_y) from TongueTracking."

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Per trial: frames in `[go-2.5, go+1.5)` are selected, frames with `likelihood < 0.9` are set to NaN, and the surviving values are averaged into the 80 bins with `np.bincount` (sum / count), leaving NaN in bins with no valid frame. These per-trial bin means are then accumulated across the whole session into a flat Python list, and the 40th and 60th percentiles of that pooled distribution are taken as the two class edges. Note the percentile population is the set of *visible-tongue bin means inside the analysed trial windows*, not all frames and not the whole session clock.

ii.
```python
mask = (tongue_timestamps >= abs_start) & (tongue_timestamps < abs_end)
if mask.sum() == 0:
    return np.full(n_bins, np.nan, dtype=np.float32)

t_in_window = tongue_timestamps[mask]
y_in_window = tongue_data[mask]
like_in_window = tongue_likelihood[mask]

# Low likelihood → set to NaN
y_in_window = y_in_window.copy()
y_in_window[like_in_window < 0.9] = np.nan

tongue_y_binned = np.full(n_bins, np.nan, dtype=np.float32)
bin_indices = np.digitize(t_in_window, bin_edges) - 1
bin_indices = np.clip(bin_indices, 0, n_bins - 1)

valid_mask = ~np.isnan(y_in_window)
if valid_mask.any():
    valid_bins = bin_indices[valid_mask]
    valid_vals = y_in_window[valid_mask]
    sums = np.bincount(valid_bins, weights=valid_vals, minlength=n_bins)
    counts = np.bincount(valid_bins, minlength=n_bins)
    has_data = counts > 0
    tongue_y_binned[has_data] = (sums[has_data] / counts[has_data]).astype(np.float32)
```

```python
valid_y = tongue_y_trial[~np.isnan(tongue_y_trial)]
if len(valid_y) > 0:
    tongue_y_session.extend(valid_y.tolist())
...
if len(tongue_y_session) > 0:
    tongue_y_arr = np.array(tongue_y_session)
    p40 = np.percentile(tongue_y_arr, 40)
    p60 = np.percentile(tongue_y_arr, 60)
else:
    p40, p60 = 0, 0
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 7: "Discretize per-session using 40th/60th percentiles over ALL valid time points." The likelihood gate is motivated by the DeepLabCut confidence being meaningful only when the tongue is actually protruding (Step 3 notes the method paper's DLC tracking with outlier correction). Step 12 explains the resulting skew: "this is because during the pre-lick period the tongue is at rest" / "the tongue is usually in the mouth (default position) during most of the trial, and only moves during licking (after go cue)". Empirically the 0.9 threshold is almost interchangeable with a lower one — the likelihood is nearly binary (10.52% of frames ≥ 0.9 vs 10.59% ≥ 0.5 in the first session).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. **Three** categories only: `0` for bin mean `< p40`, `1` for `p40 ≤ mean ≤ p60`, `2` for `mean > p60`. The instructions' fourth category (`3: not visible`) is **not implemented**. The discretised array is initialised with `np.zeros(...)` and only written where `~isnan`, so every bin with no visible-tongue frame — about 75% of all bins — silently receives class `0`, which `output_values` declares to mean "below_40th". `output_values[3]` lists only three names. The resulting distribution is `{below_40th: 0.853, 40th_to_60th: 0.049, above_60th: 0.098}`, consistent with ≈75% not-visible bins collapsed into class 0 plus 40%/20%/40% of the remaining ≈25% visible bins.

ii.
```python
final_outputs = []
for out_dict in output_trials_raw:
    tongue_y_raw = out_dict['tongue_y_raw']
    tongue_y_disc = np.zeros(N_TIMEBINS, dtype=np.float32)
    valid = ~np.isnan(tongue_y_raw)
    tongue_y_disc[valid & (tongue_y_raw < p40)] = 0
    tongue_y_disc[valid & (tongue_y_raw >= p40) & (tongue_y_raw <= p60)] = 1
    tongue_y_disc[valid & (tongue_y_raw > p60)] = 2
    ...
    full_output[3, :] = tongue_y_disc.astype(np.int64)
```

```python
'output_values': [
    ...
    ['below_40th', '40th_to_60th', 'above_60th'],
],
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 7 states only three classes: "Discretize per-session using 40th/60th percentiles" with planned sanity check "Tongue y discretization: ~40% class 0, ~20% class 1, ~40% class 2". The observed 85/5/10 split contradicts that planned check, but the AI reinterpreted it rather than treating it as a bug — Step 12: "**Tongue y-position (0.74 val)**: Excellent for 3-class with severe imbalance (85.3% class 0)"; trajectory step 136: "The tongue_y_position distribution is very skewed (86% below_40th). This is expected because the tongue is usually in the mouth." There is no mention anywhere in CONVERSION_NOTES.md or the trajectory of the instructions' class `3: not visible`.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are on the same session-absolute clock as the spikes and go cues, so alignment is done by building the same 81 bin edges used for the firing rates (`arange(81)*0.05 - 2.5 + go_cue`) and assigning each in-window camera frame to a bin with `np.digitize`. Frames are selected with the same half-open window `[go-2.5, go+1.5)`. Bin *k* of the tongue output therefore covers the same interval as bin *k* of the neural matrix. No interpolation or offset correction.

ii.
```python
n_bins = int((t_end - t_start) / bin_size)
bin_edges = np.arange(n_bins + 1) * bin_size + t_start + go_cue_time

abs_start = go_cue_time + t_start
abs_end = go_cue_time + t_end
mask = (tongue_timestamps >= abs_start) & (tongue_timestamps < abs_end)
...
bin_indices = np.digitize(t_in_window, bin_edges) - 1
bin_indices = np.clip(bin_indices, 0, n_bins - 1)
```

iii. Not separately argued in CONVERSION_NOTES.md; it follows from Step 2's finding that all NWB streams share one clock. The `--show-processing` figure plots the discretised tongue trace on the shared `time_axis` directly beneath the neural raster and the input traces, which is the AI's stated visual check that "Discretization of continuous outputs is correct" and that there are "no temporal misalignments" (Step 7: "Visual inspection OK").

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Six cases, handled in two distinct styles — exclusion for structurally missing data, and *silent substitution of a default* for everything else:

**Excluded:**
- **Session never quality-controlled** (`classification`/`anno_name` are NaN): `c == 'good'` is False for every unit, `n_good == 0`, session returns `None`. Drops 1 session → 173.
- **Trials with no spike data**: removed by the `obs_intervals` coverage test (see 1-e).
- **Session with <2 surviving trials**: returns `None`.
- **Unit with an unmappable `anno_name`**: dropped via `valid_neuron_mask` (0 units in practice).

**Silently defaulted:**
- **No tone onset before the go cue**: falls back to `tone_onset = go_cue - 1.85`, the nominal sample-epoch offset.
- **Unrecognised `outcome` string**: `.get(outcome_str, 0)` → `ignore`. Likewise `trial_instruction` not `'left'` → `right`, and `early_lick` not `'no early'` → `early`.
- **Tongue frames with low likelihood, and bins/trials with no camera frames at all**: become NaN, and NaN is then mapped to class `0` ("below_40th") by the zero-initialised discretisation array (see 8-c).

Two all-zero neural trials survive all filters and are reported by the decoder's format checker as warnings; the AI judged them negligible.

ii.
```python
tone_onset = find_last_sample_before_go(sample_start_times, go_cue)
if tone_onset is None:
    tone_onset = go_cue - 1.85
```

```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}.get(outcome_str, 0)
```

```python
mask = (tongue_timestamps >= abs_start) & (tongue_timestamps < abs_end)
if mask.sum() == 0:
    return np.full(n_bins, np.nan, dtype=np.float32)
...
tongue_y_disc = np.zeros(N_TIMEBINS, dtype=np.float32)
valid = ~np.isnan(tongue_y_raw)
```

```python
if n_good == 0:
    io.close()
    return None
...
if len(trial_indices) < 2:
    io.close()
    return None
```

iii. CONVERSION_NOTES.md Step 2 flags the unlabelled session in advance ("1 session has 0 good units... will be excluded"); Step 7 records the `obs_intervals` fix; Step 10 lists "2 all-zero neural trials: Negligible (0.002% of trials)" and "obs_intervals filtering: Properly excludes trials outside recording periods." The tone fallback is justified by Step 3's task timing ("Sample epoch (tone)... starts ~-1.85s"). The defaults in the categorical mappings and the NaN→class-0 behaviour are not discussed anywhere in CONVERSION_NOTES.md or the trajectory.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant cost is `compute_firing_rates`, which is called once per trial and, inside, loops over every good unit, boolean-masking that unit's **entire session-long** spike train against the trial window. For a 500-trial, 400-unit session that is 200,000 full-array scans of arrays up to ~10^5 spikes each. Secondary costs are the ragged per-unit HDF5 reads (`nwb.units['spike_times'][idx]` and `nwb.units['obs_intervals'][idx]`, one read per unit), reading the ~680k×3 tongue array, the `tongue_y_session.extend(...tolist())` Python-list accumulation, and pickling the 11.3 GB result.

Measured: the full conversion took **2300.8 s (~38 min)** for 174 sessions, i.e. ~13.2 s/session. This is ~9× the human reference (247 s). The AI's own Step 7 estimate was "~6.5s/session → ~19 min", so the actual run was ~2× the estimate and ~2.5× the 15-minute budget the instructions set — and the instructions' rule ("If it is much longer than your previous estimate (> 1.5x), kill the process, optimize bottlenecks, and repeat") was not applied.

ii.
```python
for ti in trial_indices:                       # ~500 iterations
    fr = compute_firing_rates(spike_times_per_unit, go_cue, T_START, T_END, BIN_SIZE_S)
```
```python
    for i, st in enumerate(spike_times_list):  # ~400 iterations, each a full-array scan
        mask = (st >= abs_start) & (st < abs_end)
```
```python
for idx in good_unit_indices:
    st = nwb.units['spike_times'][idx]         # one ragged HDF5 read per unit
    spike_times_per_unit.append(st)
```

iii. CONVERSION_NOTES.md Step 7 gives the estimate ("Load + process | ~6.5s | ~19 min for 174 sessions") and Step 9 records the actual ("Processing time: 2300.8s (~38 min)"), but no reconciliation of the 2× gap is documented. The trajectory (step 152) shows the AI profiling: "The main issue is `get_valid_trial_indices()` which does O(n_units * n_trials) comparisons... Let me also check `compute_firing_rates` and `get_tongue_y_for_trial`" — and then, at step 154, explicitly concluding "`compute_firing_rates` - already uses np.add.at, seems OK", i.e. it misdiagnosed the true bottleneck and optimised the other two instead.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three:

1. **The trial × neuron double loop in `compute_firing_rates`** — by far the most important. The trial dimension can be eliminated entirely: build one flat array of `n_trials × 81` absolute bin edges and call `np.searchsorted` once per unit against that unit's sorted spike train, then difference adjacent positions (the human reference does exactly this). That replaces ~200,000 full-array boolean scans with ~400 binary searches per session. The AI left this loop unvectorised.
2. **The per-trial tongue loop.** `get_tongue_y_for_trial` was partially vectorised (the inner per-bin loop became `np.bincount`), but it is still invoked once per trial and re-slices the full session camera array each time; a single global bin index over all frames plus one `bincount` would handle every trial at once.
3. **The photostim loop** `for start, stop in zip(photostim_starts, photostim_stops)` plus the recomputation of `bin_centers` per trial — the whole photostim input can be built as one `(n_trials, 80)` broadcast comparison.

Additionally, `spike_times_per_unit` could be loaded as one contiguous buffer via `units['spike_times'].target.data` with the per-unit offsets, rather than one HDF5 read per unit.

ii.
```python
for i, st in enumerate(spike_times_list):
    mask = (st >= abs_start) & (st < abs_end)
    spikes_in_window = st[mask]
    bin_indices = ((spikes_in_window - abs_start) / bin_size).astype(int)
    np.add.at(fr[i], bin_indices, 1.0)
```

```python
for start, stop in zip(photostim_starts, photostim_stops):
    mask = (abs_centers >= start) & (abs_centers <= stop)
    stim[mask] = 1.0
```

```python
# already vectorized by the AI:
diffs = np.abs(trial_starts[:, None] - obs_starts[None, :])
min_diffs = diffs.min(axis=1)
```

iii. CONVERSION_NOTES.md Step 6 does not fill in the template's "Code inefficiencies identified" / "Code speedups added" sections with detail, and Step 7's timing table records only a single aggregate figure. The trajectory (steps 152–156) documents the two optimisations that *were* done — "1. `get_valid_trial_indices` - vectorize the matching; 2. `get_tongue_y_for_trial` - the per-bin loop is slow, vectorize it; 3. `compute_firing_rates` - already uses np.add.at, seems OK" — and the explicit judgement that the firing-rate path needed no work.

## 10-c. What processing does the code repeat multiple times?

i. Several quantities are recomputed inside the per-trial loop that are trial-invariant or already available:

- **Full-session spike-train scans**: each unit's complete spike array is boolean-masked once per trial, so the same ~10^5-element array is rescanned ~500 times per unit per session.
- **`bin_centers`**: rebuilt from scratch in `compute_time_from_tone`, in `compute_photostim_timeseries`, and as `bin_edges` in `get_tongue_y_for_trial`, once per trial each — three identical grids recomputed ~1,500 times per session, when one module-level constant would do.
- **`find_last_sample_before_go`**: a full boolean scan of `sample_start_times` per trial, where one `np.searchsorted` over all go cues would resolve every trial at once.
- **`n_bins = int((t_end - t_start) / bin_size)`**: recomputed in four helper functions despite `N_TIMEBINS` being a module constant.
- **Camera array re-slicing**: `tongue_timestamps >= abs_start) & (< abs_end)` scans the full ~680k-frame timestamp array once per trial.

ii.
```python
def compute_time_from_tone(...):
    n_bins = int((t_end - t_start) / bin_size)
    bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
```
```python
def compute_photostim_timeseries(...):
    n_bins = int((t_end - t_start) / bin_size)
    bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
```
```python
def get_tongue_y_for_trial(...):
    n_bins = int((t_end - t_start) / bin_size)
    bin_edges = np.arange(n_bins + 1) * bin_size + t_start + go_cue_time
```
```python
def find_last_sample_before_go(sample_start_times, go_cue_time):
    valid = sample_start_times[sample_start_times < go_cue_time]
```

iii. Not identified in CONVERSION_NOTES.md. Step 6 lists the script's features without an inefficiency audit, and the trajectory's profiling pass (steps 152–156) addressed only `get_valid_trial_indices` and the tongue inner loop. The AI's justification for leaving the rest is implicit in its (mistaken) conclusion that `compute_firing_rates` "already uses np.add.at, seems OK".

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Modest but real:

- **`tongue_y_session` Python list**: every valid per-trial bin mean is converted with `.tolist()` and appended to a growing Python list (tens of thousands of boxed floats per session), used only to obtain two scalars (`p40`, `p60`). A running array or a session-level `bincount` would avoid it entirely.
- **`tongue_x`**: column 0 of the tracking array is read and never used (the read itself is unavoidable since the array is stored as `(n,3)`, but the AI also never considers the x channel as an output).
- **`valid_neuron_mask` / `-1` region sentinels**: a full per-unit Python loop with placeholder `-1` indices that are then filtered — this never removes a single unit in the whole dataset.
- **`int64` outputs**: the output array holds values in `{0,1,2}` but is stored as `int64`, 8× larger than needed; combined with `float32` neural data this contributes to the 11.3 GB pickle.
- **`auto_water`/`free_water` and the outcome/instruction/early-lick columns are read in full for every session**, including for trials that are then filtered out — trivial, but it is work on discarded rows.
- **Console summary work**: `Counter` passes over all 89,546 trials and the per-region tallies at the end of `main` are printed and discarded.
- **`metadata` omissions rather than excess**: the AI does *not* emit `session_info`, so the per-session identity it computes (`session_name`) is discarded rather than saved.

ii.
```python
valid_y = tongue_y_trial[~np.isnan(tongue_y_trial)]
if len(valid_y) > 0:
    tongue_y_session.extend(valid_y.tolist())
```

```python
full_output = np.zeros((4, N_TIMEBINS), dtype=np.int64)
```

```python
valid_neuron_mask = np.ones(n_good, dtype=bool)
for i, anno in enumerate(anno_names):
    region = map_anno_to_region(str(anno))
    if region is None:
        valid_neuron_mask[i] = False
        region_indices.append(-1)
```

```python
all_choices = []
for session_outputs in data['output']:
    for trial_out in session_outputs:
        all_choices.append(int(trial_out[0, 0]))
```

iii. Not discussed in CONVERSION_NOTES.md. The AI's Step 13 lists the output files and their sizes without commenting on the dtype choice or on redundant intermediate work. The `--show-processing` plotting path is gated behind a flag and only runs for 2 sessions, so it is not a cost in the full run.
