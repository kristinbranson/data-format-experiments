# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files under `/app/data` by recursively globbing `*.nwb`, sorting the paths, and processing each file once with `pynwb.NWBHDF5IO`. Within each file it reads `nwb.trials`, `nwb.units`, `BehavioralEvents`, and `BehavioralTimeSeries`.

ii. 
```python
files = sorted(Path('/app/data').rglob('*.nwb'))
...
with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
    nwb = io.read(); trials = nwb.trials; units = nwb.units
```

iii. In `CONVERSION_NOTES.md`, the AI states that the dataset contains 174 NWB files organized one per session and that all source access must use `pynwb`. It also notes that direct NWB access is required instead of the legacy processed files used by the paper code.

## 1-b. How are the data split into subjects?

i. Subjects are taken directly from `nwb.subject.subject_id` for each session. After all sessions are processed, the AI builds a sorted unique subject list and maps each session to an integer `subject_idx`.

ii.
```python
return dict(...,
            subject=str(nwb.subject.subject_id), ...)
...
subjects=sorted({r['subject'] for r in results}); subject_map={x:i for i,x in enumerate(subjects)}
'subject_idx':np.asarray([subject_map[r['subject']] for r in results],dtype=np.int64),
```

iii. The notes say the canonical subject identifier is the NWB `subject_id`, and that the session order follows the sorted NWB file order.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. No extra grouping is done. The session identifier is `nwb.identifier`, and session order follows the sorted path list.

ii.
```python
files = sorted(Path('/app/data').rglob('*.nwb'))
...
session_id = nwb.identifier
```

iii. The AI’s notes explicitly say the archive contains one session per NWB file and that the zero-good-unit session should be skipped, leaving 173 usable sessions.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB trials table. The AI checks that `go_start_times` has the same length as the trial table and then indexes trials with a retained trial index array.

ii.
```python
all_go = np.asarray(ev['go_start_times'].timestamps[:], dtype=np.float64)
if len(all_go) != len(trials):
    raise ValueError(f'{session_id}: go/trial count mismatch')
...
trial_idx = np.flatnonzero(trial_keep)
```

iii. In the notes, the AI records that `go_start_times` matches trial count in every session and uses this as the exact per-trial alignment event.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials that are not `auto_water`, not `free_water`, and valid for every retained classifier-good unit according to `is_good_trials`. When `is_good_trials` is shorter than the trial table, it maps those validity flags onto trials covered by that unit’s `obs_intervals`. After spike binning, it also drops any trial whose entire neural matrix is zero.

ii.
```python
auto = as_strings(trials['auto_water']) == '1'
free = as_strings(trials['free_water']) == '1'
trial_keep = ~(auto | free)
...
for ui in neuron_idx:
    stored = np.asarray(units['is_good_trials'][ui], dtype=bool)
    ...
    trial_keep &= valid
trial_idx = np.flatnonzero(trial_keep)
```

```python
nonzero_neural = np.any(rate_cube != 0, axis=(1, 2))
if not np.all(nonzero_neural):
    trial_idx = trial_idx[nonzero_neural]
    ...
```

iii. The AI justifies this in the notes as avoiding false zero-firing trials and preserving a fixed neuron set per session. It says four sessions had invalid unit-trial pairs, eight sessions had short validity vectors that needed `obs_intervals` mapping, and 127 all-zero trials were removed or fixed during later review.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from `units['spike_times']` for retained units, with `BehavioralEvents/go_start_times` used to position the trial-aligned bin edges.

ii.
```python
spikes = [np.asarray(units['spike_times'][int(i)], dtype=np.float64) for i in neuron_idx]
all_go = np.asarray(ev['go_start_times'].timestamps[:], dtype=np.float64)
abs_edges = go[:, None] + EDGES_REL[None, :]
```

iii. The notes describe the source as absolute spike timestamps from NWB plus go-cue event times on the same session clock.

## 2-b. How is the `neural` data processed?

i. The AI bins each neuron’s spike times into 80 non-overlapping 50 ms bins from -2.5 s to +1.5 s relative to go cue, using `np.searchsorted` over a flattened edge array. Counts are divided by bin width to produce firing rates in Hz.

ii.
```python
def bin_spikes(spike_times, absolute_edges):
    ...
    for j, spikes in enumerate(spike_times):
        cumulative = np.searchsorted(spikes, flat_edges, side='left').reshape(n_trials, n_edges)
        rates[:, j, :] = np.diff(cumulative, axis=1) / BIN_S
```

iii. The notes say this matches the requested 50 ms decoder bins, even though the paper code used 100 ms windows with 50 ms stride.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered to `classification == 'good'`. In addition, the AI uses per-unit `is_good_trials` and `obs_intervals` to remove trials where any retained unit is invalid, and it rejects sessions with zero retained good units.

ii.
```python
classes = as_strings(units['classification'])
neuron_idx = np.flatnonzero(classes == 'good')
if len(neuron_idx) == 0:
    print(f'SKIP {session_id}: zero classifier-good units', flush=True)
    return None
```

```python
for ui in neuron_idx:
    stored = np.asarray(units['is_good_trials'][ui], dtype=bool)
    ...
    trial_keep &= valid
```

iii. The notes justify `classification == 'good'` as the paper’s published QC classifier and justify the added trial-validity filter as a way to avoid encoding invalid recording periods as zero firing.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to go cue onset by adding the shared relative edge grid to each trial’s absolute go time and binning spikes against those absolute edges.

ii.
```python
abs_edges = go[:, None] + EDGES_REL[None, :]
abs_centers = go[:, None] + CENTERS_REL[None, :]
rate_cube = bin_spikes(spikes, abs_edges)
```

iii. The AI’s notes say all streams share the same session clock, so go-cue alignment only requires using the NWB go timestamps directly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 80 bins of width 50 ms covering `[-2.5, 1.5)` seconds relative to go cue. Spike times are rebinned into these non-overlapping bins.

ii.
```python
OFF_START, OFF_END, BIN_S = -2.5, 1.5, 0.05
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
N_TIME = 80
```

iii. The notes explicitly say the task specification overrides the paper’s 100 ms sliding windows and requires 50 ms non-overlapping bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives tone time from trial `start_time`, `BehavioralEvents/go_start_times`, `sample_start_times`, `sample_stop_times`, `delay_start_times`, and `delay_stop_times`. It selects the sample state linked to the final pre-go delay rather than simply taking the last sample start before go.

ii.
```python
sample_starts = np.asarray(ev['sample_start_times'].timestamps[:], dtype=np.float64)
sample_stops = np.asarray(ev['sample_stop_times'].timestamps[:], dtype=np.float64)
delay_starts = np.asarray(ev['delay_start_times'].timestamps[:], dtype=np.float64)
delay_stops = np.asarray(ev['delay_stop_times'].timestamps[:], dtype=np.float64)
all_tone = choose_tone_onsets(starts, all_go, sample_starts, sample_stops,
                              delay_starts, delay_stops)
```

iii. The notes and trajectory say repeated sample states occur in early-lick/retry trials, so the AI preferred the sample state whose stop is nearest the final delay start, with that delay chosen as the one whose stop is nearest go.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After choosing a tone onset per trial, the AI computes time from tone onset as absolute bin center time minus the selected tone onset.

ii.
```python
abs_centers = go[:, None] + CENTERS_REL[None, :]
tone_time = abs_centers - tone[:, None]
```

iii. The AI’s notes describe this as a continuous time-varying input defined at the 50 ms bin centers.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the same go-aligned 50 ms bin centers used for the neural data, so each timepoint corresponds to the same trial bin as the firing rates.

ii.
```python
abs_edges = go[:, None] + EDGES_REL[None, :]
abs_centers = go[:, None] + CENTERS_REL[None, :]
tone_time = abs_centers - tone[:, None]
```

iii. The notes say all signals share one session clock, so no additional offset correction is needed.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`, together with each trial’s `start_time` and `stop_time` to assign events to trials.

ii.
```python
ps = np.asarray(ev['photostim_start_times'].timestamps[:], dtype=np.float64)
pe = np.asarray(ev['photostim_stop_times'].timestamps[:], dtype=np.float64)
for k, src_i in enumerate(trial_idx):
    a, b = starts[src_i], float(trials['stop_time'][src_i])
    hits = np.flatnonzero((ps >= a) & (ps <= b))
```

iii. The notes say trial-table photostim fields include string values such as `N/A`, so absolute event timestamps are the safer source for a time-varying input.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each retained trial, the AI finds at most one photostimulation interval inside the trial and marks a bin as 1 when its center lies between that interval’s absolute start and stop times.

ii.
```python
stim = np.zeros((len(trial_idx), N_TIME), dtype=np.float32)
...
if len(hits) == 1:
    h = hits[0]
    stim[k] = ((abs_centers[k] >= ps[h]) & (abs_centers[k] < pe[h])).astype(np.float32)
```

iii. The notes justify this as a bin-center representation of a time-varying on/off input on the same clock as the neural bins.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is aligned by comparing the absolute photostim interval directly to the absolute go-aligned bin centers used for the neural data.

ii.
```python
abs_centers = go[:, None] + CENTERS_REL[None, :]
stim[k] = ((abs_centers[k] >= ps[h]) & (abs_centers[k] < pe[h])).astype(np.float32)
```

iii. The AI’s notes say all event streams use the same absolute session clock, so alignment is direct.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from the trial table columns `trial_instruction` and `outcome`.

ii.
```python
instruction = as_strings(trials['trial_instruction'])
outcome = as_strings(trials['outcome'])
```

```python
def trial_choice(instruction, outcome):
    if outcome == 'ignore':
        return 2
    if outcome == 'hit':
        return 0 if instruction == 'left' else 1
    if outcome == 'miss':
        return 1 if instruction == 'left' else 0
```

iii. The notes say there is no direct choice column, so instructed side plus outcome is used to recover left, right, or no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes choice as `0` left, `1` right, `2` no lick and repeats that value across all 80 bins of the first output row.

ii.
```python
arr = np.empty((4, N_TIME), dtype=np.int64)
arr[0] = trial_choice(instruction[src_i], outcome[src_i])
```

iii. The notes state that per-trial outputs were repeated over time so all four outputs could share a `(4, 80)` array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the `trials['outcome']` column.

ii.
```python
outcome = as_strings(trials['outcome'])
```

iii. The notes say the NWB trials table already stores the needed categories `ignore`, `miss`, and `hit`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore -> 0`, `miss -> 1`, and `hit -> 2`, then repeats the per-trial value across all 80 bins.

ii.
```python
omap = {'ignore': 0, 'miss': 1, 'hit': 2}
...
arr[1] = omap[outcome[src_i]]
```

iii. The notes say this order matches the requested output categories.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the `trials['early_lick']` column.

ii.
```python
early = as_strings(trials['early_lick'])
```

iii. The notes describe early lick as a required decoder target even though the paper’s “regular trial” analyses excluded those trials.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early -> 0` and `early -> 1`, then repeats that per-trial label across all 80 bins.

ii.
```python
arr[2] = 0 if early[src_i] == 'no early' else 1
```

iii. The notes say repeated per-trial labels were used so all outputs share the same shape.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`, using `timestamps`, the y coordinate in column 1, and the tracking likelihood in column 2.

ii.
```python
tongue = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
video_t = np.asarray(tongue.timestamps[:], dtype=np.float64)
video_d = np.asarray(tongue.data[:], dtype=np.float64)
```

iii. The notes say this stream is present in all sessions and explicitly stores `(x, y, likelihood)` at about 294 Hz.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI uses a visibility threshold of `likelihood >= 0.9`, computes session percentiles from all visible raw tongue-y frames, assigns each decoder bin the nearest video frame to the bin center, and then classifies that frame’s y-value into one of four categories.

ii.
```python
VISIBILITY_THRESHOLD = 0.9
...
visible_session = video_d[:, 2] >= VISIBILITY_THRESHOLD
q40, q60 = np.percentile(video_d[visible_session, 1], [40, 60])
frame_idx = nearest_indices(video_t, abs_centers.ravel()).reshape(len(trial_idx), N_TIME)
ty = video_d[frame_idx, 1]; tl = video_d[frame_idx, 2]
```

iii. The notes justify the `0.9` threshold by saying the likelihood distribution is strongly bimodal and that low-confidence frames should not be treated as visible tongue. It also says visible-only percentiles are necessary because low-confidence placeholder coordinates distort thresholds.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Category `3` means not visible. Otherwise, the nearest frame’s y-value is thresholded using per-session 40th and 60th percentiles computed from visible frames: `< q40 -> 0`, `q40..q60 -> 1`, `> q60 -> 2`.

ii.
```python
tongue_class = np.full((len(trial_idx), N_TIME), 3, dtype=np.int64)
vis = tl >= VISIBILITY_THRESHOLD
tongue_class[vis & (ty < q40)] = 0
tongue_class[vis & (ty >= q40) & (ty <= q60)] = 1
tongue_class[vis & (ty > q60)] = 2
```

iii. The notes say the thresholding is per session and based on visible tongue samples only, with a separate “not visible” class for low-confidence bins.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue output by evaluating the nearest camera frame at each go-aligned neural bin center, rather than averaging camera samples over the full 50 ms neural bin.

ii.
```python
abs_centers = go[:, None] + CENTERS_REL[None, :]
frame_idx = nearest_indices(video_t, abs_centers.ravel()).reshape(len(trial_idx), N_TIME)
ty = video_d[frame_idx, 1]; tl = video_d[frame_idx, 2]
```

iii. The notes say all streams share one clock and that nearest-frame assignment over bin centers is sufficient for alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several cases by exclusion or explicit categorization: sessions with zero classifier-good units are skipped; trials are removed if they are `auto_water`, `free_water`, invalid for any retained unit, or all-zero neurally; short `is_good_trials` vectors are mapped via `obs_intervals`; tongue samples with low likelihood become “not visible”; and missing anatomy in classifier-good units triggers an error.

ii.
```python
neuron_idx = np.flatnonzero(classes == 'good')
if len(neuron_idx) == 0:
    ... return None
```

```python
if len(stored) == len(trials):
    valid = stored
else:
    obs = np.asarray(units['obs_intervals'][ui], dtype=np.float64).reshape(-1, 2)
    ...
```

```python
visible_session = video_d[:, 2] >= VISIBILITY_THRESHOLD
...
tongue_class = np.full((len(trial_idx), N_TIME), 3, dtype=np.int64)
```

iii. The notes describe these choices as conservative handling of missing recording coverage and low-confidence tracking, with later review specifically focused on fixing all-zero-neural trials caused by misread validity masks.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive steps are opening and reading each NWB file, loading spike and video arrays, per-unit spike binning with `searchsorted`, and serializing the large pickle at the end.

ii.
```python
with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
    ...
spikes = [np.asarray(units['spike_times'][int(i)], dtype=np.float64) for i in neuron_idx]
rate_cube = bin_spikes(spikes, abs_edges)
...
with out.open('wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. In the notes, the AI reports a 194 s full conversion, with vectorized spike binning used specifically to avoid much slower nested histogram loops.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI still leaves several Python loops that could be vectorized further: the per-trial tone-selection loop in `choose_tone_onsets`, the per-unit spike-binning loop in `bin_spikes`, the per-trial photostimulation assignment loop, and the per-trial output-construction loop.

ii.
```python
for i, (start, go) in enumerate(zip(trial_starts, go_times)):
    ...
for j, spikes in enumerate(spike_times):
    ...
for k, src_i in enumerate(trial_idx):
    ...
for k, src_i in enumerate(trial_idx):
    arr = np.empty((4, N_TIME), dtype=np.int64)
```

iii. The notes emphasize that the AI already vectorized the expensive trial dimension of spike binning, but the code still repeats several per-trial scans over event arrays and per-trial output assembly.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly rescans session-level event arrays on a trial-by-trial basis. `choose_tone_onsets` searches all sample and delay events separately for every trial, and the photostimulation code searches all photostim intervals separately for every retained trial.

ii.
```python
for i, (start, go) in enumerate(zip(trial_starts, go_times)):
    didx = np.flatnonzero((delay_starts >= start) & (delay_starts <= go))
    sidx = np.flatnonzero((sample_starts >= start) & (sample_starts < go))
```

```python
for k, src_i in enumerate(trial_idx):
    a, b = starts[src_i], float(trials['stop_time'][src_i])
    hits = np.flatnonzero((ps >= a) & (ps <= b))
```

iii. The trajectory shows the AI iterating on correctness around tone selection and validity handling; in the final code, the main repeated work left inside a single run is these repeated scans over the same session-level event vectors.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In the core conversion path, very little is discarded, but there is optional diagnostic work that is not needed for downstream decoding: `--show-processing` produces plots, the script computes and logs extra session diagnostics such as `q40`, `q60`, and `elapsed`, and metadata stores some provenance fields that are not required by the decoder itself.

ii.
```python
def plot_session(...):
    ...
    fig.tight_layout(); fig.savefig(f'/app/processing_{session_id}.png', dpi=140); plt.close(fig)
```

```python
return dict(...,
            source_file=str(path), source_trial_idx=trial_idx.tolist(),
            q40=float(q40), q60=float(q60), elapsed=elapsed)
```

iii. The notes describe these as review and validation aids rather than essential decoder inputs. Unlike the neural, input, and output arrays, they are not used by downstream model training.
