# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script finds all NWB files with a glob over `data/sub-*/sub-*_ses-*.nwb`, sorts them, and processes each file as one session with `process_session()`. Inside each session it opens the NWB file with `NWBHDF5IO`, reads the trials table, behavioral event time series, behavioral video time series, and unit table.

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*/sub-*_ses-*.nwb')))

for i, nwb_file in enumerate(nwb_files):
    result = process_session(nwb_file, verbose=verbose)
```

```python
io = NWBHDF5IO(nwb_file, 'r')
nwb = io.read()

trials = nwb.trials.to_dataframe()
be = nwb.acquisition['BehavioralEvents']
bt = nwb.acquisition['BehavioralTimeSeries']
units = nwb.units
```

iii. `CONVERSION_NOTES.md` says the source data are 174 NWB files. The trajectory summary also says the agent decided that one NWB file corresponds to one behavioral/ephys session and that all files should be iterated in sorted order.

## 1-b. How are the data split into subjects?

i. Subjects are split by NWB subject metadata. The per-session subject label used in the final dataset is `nwb.subject.description` (for example `SC015`), and `subject_idx` is built by indexing sessions into the ordered unique list of those subject descriptions.

ii.
```python
subject_id = nwb.subject.subject_id
subject_desc = nwb.subject.description
```

```python
if sub_desc not in all_subjects:
    all_subjects[sub_desc] = result['subject_id']

subjects = list(all_subjects.keys())
subject_idx.append(subjects.index(sess['subject_desc']))
```

iii. The code and notes consistently treat the mouse code in `subject.description` as the subject identity. The trajectory summary explicitly says the converted output uses `subject_desc` values such as `SC015` as the subject list.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. `process_session()` returns one session-level dict if the file passes filtering, and `convert_all()` appends that dict as one element in the top-level `neural`, `input`, `output`, and `brain_region_idx` session lists.

ii.
```python
def process_session(nwb_file, verbose=True):
    ...
    return {
        'neural': neural_data,
        'input': input_data,
        'output': output_data,
        ...
        'sess_name': sess_name,
    }
```

```python
for sess in all_sessions:
    neural.append(sess['neural'])
    inputs.append(sess['input'])
    outputs.append(sess['output'])
    brain_region_idx.append(region_indices)
```

iii. The notes report session counts such as “144 of 174 sessions passed,” which shows that the agent treated each file as a session and then filtered entire sessions in or out.

## 1-d. How are the data split into trials?

i. Trials come from rows of the NWB `trials` table and the matching `go_start_times` timestamps. After filtering, each remaining trial index becomes one `(n_neurons, 80)` neural matrix, one `(2, 80)` input matrix, and one `(4, 80)` output matrix.

ii.
```python
trials = nwb.trials.to_dataframe()
n_trials_total = len(trials)
go_start_times = be.time_series['go_start_times'].timestamps[:]
assert len(go_start_times) == n_trials_total
```

```python
valid_indices = np.where(valid_trial_mask)[0]

for trial_idx in valid_indices:
    go_time = go_start_times[trial_idx]
    ...
    neural_data.append(trial_fr)
    input_data.append(input_trial)
    output_data.append(output_trial)
```

iii. In the trajectory the agent explicitly reasoned that extra `sample_start_times` reflect early-lick replays, not extra trials, so it used the NWB trial table plus go-cue timestamps as the authoritative trial split.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering has two layers. First, session-level quality control is computed on “control” trials from the full session: no `auto_water`, no `free_water`, no early lick, no `ignore`, and no photostimulation. Second, trials actually included in the converted dataset are restricted to those covered by neural recording according to `obs_intervals`, and then filtered only by `auto_water == 0` and `free_water == 0`. Early-lick and ignore trials are kept in the converted dataset.

ii.
```python
all_valid = (auto_water == 0) & (free_water == 0)
control_mask_all = (all_valid &
                    (early_lick == 'no early') &
                    (outcome != 'ignore') &
                    no_photostim)
```

```python
recorded_trial_indices = np.arange(first_recorded_trial, first_recorded_trial + n_obs_trials)
...
for ti in recorded_trial_indices:
    if auto_water[ti] == 0 and free_water[ti] == 0:
        valid_trial_mask[ti] = True
```

iii. The notes and trajectory both say the agent intentionally kept early-lick and ignore trials because the decoder specification required predicting `early_lick` and the three-way `outcome`. The same sources say session filtering still follows the paper’s behavioral criteria on control trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are derived from raw spike times in `nwb.units.get_unit_spike_times(ui)`, aligned with `BehavioralEvents/go_start_times`, and filtered by the unit-table fields `classification`, `anno_name`, and `obs_intervals`.

ii.
```python
go_start_times = be.time_series['go_start_times'].timestamps[:]
classification = units['classification'].data[:]
anno_names = units['anno_name'].data[:]
obs_0 = units.get_unit_obs_intervals(good_indices[0])
```

```python
for ui in good_indices:
    st = units.get_unit_spike_times(ui)
    all_spike_times.append(st)
```

iii. The trajectory says the agent confirmed that NWB spike times are in absolute session time and therefore must be aligned by subtracting the go-cue timestamp for each trial.

## 2-b. How is the `neural` data processed?

i. For each kept trial, the code extracts spikes in a `[-2.5, 1.5]` s window around the go cue, converts spike times to go-cue-relative time, bins them into 50 ms non-overlapping bins, and divides by bin width to convert counts to firing rates in spikes/s.

ii.
```python
BIN_WIDTH = 0.05
BEGIN_TIME = -2.5
END_TIME = 1.5
BIN_EDGES = np.linspace(BEGIN_TIME, END_TIME, N_BINS + 1)
```

```python
abs_start = go_time + BEGIN_TIME
abs_end = go_time + END_TIME
lo = np.searchsorted(st, abs_start)
hi = np.searchsorted(st, abs_end)
if hi > lo:
    rel_spikes = st[lo:hi] - go_time
    counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
    trial_fr[i, :] = counts / BIN_WIDTH
```

iii. `CONVERSION_NOTES.md` states “Spike counts per bin, converted to firing rates (spikes/s)” with a 50 ms bin size, and the trajectory says the agent chose direct spike-time binning around go cue because the NWB data are not pre-aligned like the reference `.mat` export.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept only if `classification == 'good'` and `anno_name` can be mapped into one of the agent’s 14 broad brain regions. Sessions are then dropped if they fail behavioral criteria or have no good mapped units. Trials are restricted to the portion of the session covered by the neural recording according to `obs_intervals`.

ii.
```python
for ui in range(n_units_total):
    if classification[ui] != 'good':
        continue
    region = map_anno_to_region(anno_names[ui])
    if region is not None:
        good_indices.append(ui)
        unit_regions.append(region)
```

```python
if n_good == 0:
    ...
    return None

obs_0 = units.get_unit_obs_intervals(good_indices[0])
...
recorded_trial_indices = np.arange(first_recorded_trial, first_recorded_trial + n_obs_trials)
```

iii. The notes cite the classifier-derived `classification == 'good'` label as the quality-control criterion and explain that units without a valid mapped `anno_name` are discarded. The trajectory also records that `obs_intervals` were added after the agent discovered zero-filled trials when neural recording only covered part of a session.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to go-cue onset. For each trial the code uses that trial’s `go_start_times` timestamp as time zero and stores 80 bins spanning `-2.5` s to `+1.5` s relative to the go cue.

ii.
```python
go_start_times = be.time_series['go_start_times'].timestamps[:]
...
go_time = go_start_times[trial_idx]
abs_start = go_time + BEGIN_TIME
abs_end = go_time + END_TIME
rel_spikes = st[lo:hi] - go_time
```

iii. This choice is explicit in the header comment, metadata, notes, and trajectory. The agent states repeatedly that the required temporal alignment event is go-cue onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 50 ms bins. There is no later rebinning or smoothing step; the code bins raw spikes directly into the final 80-bin trial representation.

ii.
```python
BIN_WIDTH = 0.05
N_BINS = int(round((END_TIME - BEGIN_TIME) / BIN_WIDTH))
BIN_EDGES = np.linspace(BEGIN_TIME, END_TIME, N_BINS + 1)
```

```python
counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
trial_fr[i, :] = counts / BIN_WIDTH
```

iii. The notes say “50ms non-overlapping bins -> 80 time bins.” The trajectory also contrasts this with the reference code’s 40 ms / 3.4 ms preprocessing and says the decoder task instructions overrode that earlier representation.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. In code, it is not derived from a per-trial raw NWB timestamp. It is derived from the constants `BIN_CENTERS` and `TONE_ONSET_REL = -1.85`. The agent justified `-1.85` s from the task structure and by comparing `sample_start_times` and `go_start_times` outside the final code path.

ii.
```python
TONE_ONSET_REL = -1.85
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
```

```python
input_trial = np.stack([TIME_FROM_TONE, photostim_binary], axis=0)
```

iii. `CONVERSION_NOTES.md` says tone onset is consistently 1.85 s before go cue and attributes that to the 0.65 s sample period plus 1.2 s delay. The trajectory shows the agent inspecting `sample_start_times` versus `go_start_times`, then deciding to encode tone timing as a constant offset.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The code precomputes one shared 80-element vector of bin-center times relative to tone onset by subtracting the fixed tone-onset offset from each go-cue-relative bin center. The same vector is reused for every trial.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
```

```python
input_trial = np.stack([TIME_FROM_TONE, photostim_binary], axis=0)
```

iii. The trajectory says the agent deliberately allowed negative values before tone onset and positive values after it. The justification given there is that the time-from-tone feature should be a continuous aligned covariate, not a clamped indicator.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is aligned by construction to the same 80 time bins used for neural data. The vector `TIME_FROM_TONE` has one value per neural bin center and is stacked into every trial’s `input` array alongside the photostimulation trace.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
```

```python
input_trial = np.stack([TIME_FROM_TONE, photostim_binary], axis=0)
```

iii. The notes say this input is “same for all trials (deterministic from bin structure).” The trajectory likewise describes it as one per-bin covariate aligned to the go-cue-centered neural bins.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The photostimulation input is derived from the behavioral-event time series `photostim_start_times` and `photostim_stop_times`. A separate raw variable, `trials['photostim_power']`, is only used to define control trials for session filtering.

ii.
```python
photostim_power = trials['photostim_power'].values
no_photostim = np.array([str(p) == 'N/A' or str(p) == 'nan' or
                         (isinstance(p, (int, float)) and p == 0)
                         for p in photostim_power])
```

```python
photostim_start_ts = be.time_series['photostim_start_times'].timestamps[:]
photostim_stop_ts = be.time_series['photostim_stop_times'].timestamps[:]
```

iii. The notes identify a binary photostimulation input and the trajectory says the agent decided to mark bins using onset/offset event times, while still using non-photostim control trials for behavioral session QC.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial the code starts with an all-zero 80-bin vector, then marks bins as `1.0` if the bin center falls inside any photostimulation interval overlapping that trial’s `[-2.5, 1.5]` s window. It ignores power and stimulation type, collapsing the signal to “on/off”.

ii.
```python
photostim_binary = np.zeros(N_BINS, dtype=np.float32)
trial_start_abs = go_time + BEGIN_TIME
trial_end_abs = go_time + END_TIME
```

```python
for ps_start, ps_stop in zip(photostim_start_ts, photostim_stop_ts):
    if ps_stop > trial_start_abs and ps_start < trial_end_abs:
        rel_start = ps_start - go_time
        rel_stop = ps_stop - go_time
        mask = (BIN_CENTERS >= rel_start) & (BIN_CENTERS < rel_stop)
        photostim_binary[mask] = 1.0
```

iii. `CONVERSION_NOTES.md` describes this input as binary on/off. The trajectory also says the agent wanted a time-varying indicator over the go-cue-centered bins rather than a trial-level label.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is aligned in absolute session time and then converted to the same go-cue-relative 80-bin grid as the neural data by comparing each trial’s bin centers to `photostim_start_times` and `photostim_stop_times`.

ii.
```python
trial_start_abs = go_time + BEGIN_TIME
trial_end_abs = go_time + END_TIME
rel_start = ps_start - go_time
rel_stop = ps_stop - go_time
mask = (BIN_CENTERS >= rel_start) & (BIN_CENTERS < rel_stop)
```

iii. The trajectory says the agent explicitly converted absolute photostim event times into go-cue-relative trial coordinates so the binary stimulation trace would line up with the neural bins.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The code does not use explicit lick-direction event data. Instead it derives `choice` from `trial_instruction` and `outcome`: hits are assigned to the instructed side, misses to the opposite side, and ignores also to the instructed side.

ii.
```python
trial_instruction = trials['trial_instruction'].values
outcome = trials['outcome'].values
```

```python
if outc == 'hit':
    choice = 0 if instr == 'left' else 1
elif outc == 'miss':
    choice = 1 if instr == 'left' else 0
else:  # ignore
    choice = 0 if instr == 'left' else 1
```

iii. The trajectory shows the agent debating how to encode `choice` for ignore trials because no lick occurred, then deciding to infer choice from task instruction and outcome rather than from actual lick-direction variables.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. `choice` is coded as `0` for left and `1` for right. The value is a per-trial scalar inferred from `trial_instruction` and `outcome`, then broadcast across all 80 time bins as a constant categorical trace.

ii.
```python
if outc == 'hit':
    choice = 0 if instr == 'left' else 1
elif outc == 'miss':
    choice = 1 if instr == 'left' else 0
else:  # ignore
    choice = 0 if instr == 'left' else 1
```

```python
output_trial = np.stack([
    np.full(N_BINS, choice, dtype=np.int64),
    ...
], axis=0)
```

iii. The trajectory says the agent chose this rule because the decoder output schema requires a binary choice even for ignore trials, where there is no observed lick choice to encode directly.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `Outcome` is derived from the NWB trial-table column `outcome`.

ii.
```python
outcome = trials['outcome'].values
...
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outc]
```

iii. The notes and trajectory both describe a direct three-way mapping from the trial `outcome` label to the decoder output categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The string labels are converted to integer classes `ignore = 0`, `miss = 1`, and `hit = 2`, then broadcast across all 80 time bins as a constant per-trial target.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outc]
```

```python
output_trial = np.stack([
    np.full(N_BINS, choice, dtype=np.int64),
    np.full(N_BINS, outcome_val, dtype=np.int64),
    ...
], axis=0)
```

iii. The notes explicitly list the same three categories, and the trajectory says the agent kept ignore trials so that the `outcome` output would retain all three specified classes.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. This output does not exist in the agent’s code. The converted dataset contains `outcome`, not `distance to reward zone`. For `outcome`, alignment is only through broadcasting the per-trial scalar outcome class across the same 80-bin trial grid as the neural data.

ii.
```python
output_names = ['choice', 'outcome', 'early_lick', 'tongue_y_position']
```

```python
output_trial = np.stack([
    np.full(N_BINS, choice, dtype=np.int64),
    np.full(N_BINS, outcome_val, dtype=np.int64),
    np.full(N_BINS, early_lick_val, dtype=np.int64),
    tongue_y_disc.astype(np.int64),
], axis=0)
```

iii. The task instructions define `Outcome` but not any reward-zone-distance output. Nothing in the notes or trajectory mentions such a variable; this appears to be a typo in the question list.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. `Early lick` is derived from the NWB trial-table column `early_lick`.

ii.
```python
early_lick = trials['early_lick'].values
...
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
```

iii. The notes and trajectory both describe this as a direct trial-table readout, kept specifically because the decoder was asked to predict early-lick status.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The string label is binarized as `1` for `'early'` and `0` otherwise, then repeated across all 80 bins as a constant per-trial categorical output.

ii.
```python
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
```

```python
output_trial = np.stack([
    ...
    np.full(N_BINS, early_lick_val, dtype=np.int64),
    tongue_y_disc.astype(np.int64),
], axis=0)
```

iii. The trajectory shows this was an intentional divergence from the paper’s analysis-trial exclusions: early-lick trials were preserved so that this output would remain meaningful.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `Tongue y-position` is derived from the side-camera DeepLabCut time series `BehavioralTimeSeries/Camera0_side_TongueTracking`, specifically column 1 of `tongue_ts.data`, together with the corresponding timestamps.

ii.
```python
bt = nwb.acquisition['BehavioralTimeSeries']
tongue_ts = bt.time_series['Camera0_side_TongueTracking']
tongue_y_all_data = tongue_ts.data[:, 1]
tongue_timestamps = tongue_ts.timestamps[:]
```

iii. The notes identify DeepLabCut tongue tracking as the source, and the trajectory says the agent confirmed that the tracking array contains `(x, y, confidence)` columns before writing the conversion.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The code computes session-wide 40th and 60th percentiles of all tongue `y` samples, then for each neural bin chooses the nearest tongue-tracking frame in time and reads its `y` value. No confidence or missing-data mask is applied in the final code.

ii.
```python
tongue_y_p40 = np.percentile(tongue_y_all_data, 40)
tongue_y_p60 = np.percentile(tongue_y_all_data, 60)
```

```python
abs_times = go_time + BIN_CENTERS
tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
tongue_indices = np.clip(tongue_indices, 0, len(tongue_timestamps) - 1)
prev_indices = np.clip(tongue_indices - 1, 0, len(tongue_timestamps) - 1)
dist_curr = np.abs(tongue_timestamps[tongue_indices] - abs_times)
dist_prev = np.abs(tongue_timestamps[prev_indices] - abs_times)
use_prev = dist_prev < dist_curr
tongue_indices[use_prev] = prev_indices[use_prev]
tongue_y_values = tongue_y_all_data[tongue_indices]
```

iii. The trajectory says the agent initially considered filtering low-confidence tracking points, but the final code does not do that. `CONVERSION_NOTES.md` is inconsistent here: it claims a likelihood filter and 33rd/67th thresholds, but the actual code uses all samples and 40th/60th thresholds.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The code discretizes each aligned tongue `y` sample into three classes using session-wide thresholds: class `0` for values below the 40th percentile, class `1` for values from the 40th percentile up to and including the 60th percentile, and class `2` for values above the 60th percentile.

ii.
```python
tongue_y_disc = np.zeros(N_BINS, dtype=np.float32)
tongue_y_disc[tongue_y_values >= tongue_y_p40] = 1
tongue_y_disc[tongue_y_values > tongue_y_p60] = 2
```

iii. This matches the decoder task instructions. The notes, however, incorrectly document 33rd/67th percentiles; that documented justification does not match the code.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. It is aligned by nearest-neighbor resampling: for each neural bin center, the code picks the closest tongue-camera timestamp in absolute time and assigns that frame’s discretized `y` category to the corresponding neural bin.

ii.
```python
abs_times = go_time + BIN_CENTERS
tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
...
dist_curr = np.abs(tongue_timestamps[tongue_indices] - abs_times)
dist_prev = np.abs(tongue_timestamps[prev_indices] - abs_times)
use_prev = dist_prev < dist_curr
tongue_indices[use_prev] = prev_indices[use_prev]
```

iii. The trajectory says the agent intended to “resample it to match the neural bins.” The final implementation does that with nearest-frame lookup rather than averaging within each 50 ms bin.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The code mainly handles irregularities by skipping data rather than repairing them. Entire sessions are skipped on exceptions, on missing control trials, on low performance, on too few correct trials, on having no good mapped units, or on having fewer than two usable trials. Units with blank or unmapped `anno_name` are dropped. Partial-session neural coverage is handled by matching `obs_intervals` to the first recorded trial via `start_time`. There is no explicit NaN handling for tongue tracking or other missing continuous data in the final code.

ii.
```python
if n_control == 0:
    ...
    return None
...
if performance < MIN_PERFORMANCE:
    ...
    return None
...
if n_good == 0:
    ...
    return None
```

```python
first_obs_start = obs_0[0, 0]
first_recorded_trial = np.argmin(np.abs(trial_starts - first_obs_start))
recorded_trial_indices = np.arange(first_recorded_trial, first_recorded_trial + n_obs_trials)
```

```python
try:
    result = process_session(nwb_file, verbose=verbose)
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
    continue
```

iii. The trajectory documents one concrete “data mistake” the agent fixed: it originally assumed `obs_intervals` started at trial 0 and later changed to timestamp-based matching. The notes also mention dropping unmapped annotations and tolerating a few all-zero edge trials rather than removing them.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant cost is per-session neural extraction: nested loops over trials and units, with `searchsorted` and `np.histogram` for every unit-trial pair. Secondary costs are repeated per-trial scans over all photostimulation intervals and repeated nearest-neighbor tongue alignment for every trial.

ii.
```python
for trial_idx in valid_indices:
    ...
    for i, st in enumerate(all_spike_times):
        lo = np.searchsorted(st, abs_start)
        hi = np.searchsorted(st, abs_end)
        if hi > lo:
            rel_spikes = st[lo:hi] - go_time
            counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
            trial_fr[i, :] = counts / BIN_WIDTH
```

```python
for ps_start, ps_stop in zip(photostim_start_ts, photostim_stop_ts):
    ...
abs_times = go_time + BIN_CENTERS
tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
```

iii. The trajectory explicitly says the agent found the conversion “slow” and identified per-unit spike extraction and firing-rate computation as the main bottleneck, then mentioned tongue and photostim alignment as smaller optimization targets.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-by-unit histogram loop could be vectorized or replaced by a more batch-oriented spike-binning strategy. The photostimulation interval loop is repeated for every trial and could be pre-indexed or converted to a vectorized rasterization step. The `valid_trial_mask` loop over `recorded_trial_indices` is also trivially vectorizable.

ii.
```python
for ti in recorded_trial_indices:
    if auto_water[ti] == 0 and free_water[ti] == 0:
        valid_trial_mask[ti] = True
```

```python
for trial_idx in valid_indices:
    ...
    for i, st in enumerate(all_spike_times):
        ...
```

```python
for ps_start, ps_stop in zip(photostim_start_ts, photostim_stop_ts):
    ...
```

iii. The trajectory shows the agent itself calling out vectorization opportunities for spike extraction, photostim alignment, and tongue processing while trying to reduce runtime.

## 10-c. What processing does the code repeat multiple times?

i. The code loops over all kept trials twice: once to build `neural_data` and once to build `input_data` and `output_data`. Within those loops it repeatedly recomputes trial window bounds, repeatedly scans all photostim intervals, and repeatedly performs nearest-neighbor tongue-frame selection.

ii.
```python
neural_data = []
for trial_idx in valid_indices:
    go_time = go_start_times[trial_idx]
    ...
    neural_data.append(trial_fr)
```

```python
input_data = []
output_data = []
for trial_idx in valid_indices:
    go_time = go_start_times[trial_idx]
    ...
    input_data.append(input_trial)
    output_data.append(output_trial)
```

iii. The agent’s own runtime notes in the trajectory mention that the script revisits the same trials multiple times and redoes alignment work separately for spikes, photostimulation, and tongue data.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary work is the temporary session bookkeeping that is only used for filtering or console summaries and then discarded from the final output, for example `performance`, `correct_left`, `correct_right`, `sess_name`, and `subject_id` inside the per-session dict. The code also expands inherently per-trial scalar targets (`choice`, `outcome`, `early_lick`) into full 80-bin constant vectors, which increases computation and storage without adding time structure.

ii.
```python
return {
    'neural': neural_data,
    'input': input_data,
    'output': output_data,
    'unit_regions': unit_regions,
    'subject_id': subject_id,
    'subject_desc': subject_desc,
    'n_good': n_good,
    'n_trials': n_trials,
    'performance': performance,
    'correct_left': correct_left,
    'correct_right': correct_right,
    'sess_name': sess_name,
}
```

```python
output_trial = np.stack([
    np.full(N_BINS, choice, dtype=np.int64),
    np.full(N_BINS, outcome_val, dtype=np.int64),
    np.full(N_BINS, early_lick_val, dtype=np.int64),
    tongue_y_disc.astype(np.int64),
], axis=0)
```

iii. The trajectory does not frame these as bugs, but it does show the agent focusing on runtime and output format rather than minimizing intermediate bookkeeping or storage overhead. Those values are mainly used during conversion and validation, not by the downstream decoder itself.
