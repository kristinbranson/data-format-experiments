# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is one NWB file per session under `/app/data/sub-<id>/`. The AI finds every session with a single sorted glob over that layout and opens each file once with `pynwb.NWBHDF5IO`. All per-session content (subject id, units table, trials table, behavioral event streams, video tracking series) is pulled out inside one `load_nwb_session()` call into a plain dict, the file is closed, and the rest of the conversion works on numpy arrays. Sessions are processed serially in a single pass (no parallelism, no caching between runs). 174 files are found; 144 survive filtering.

ii.
```python
DATA_DIR = '/app/data'

def get_nwb_files():
    """Get list of all NWB files sorted by subject then session."""
    return sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*_ses-*.nwb')))
```
```python
def load_nwb_session(nwb_path):
    """Load all needed data from a single NWB file."""
    import pynwb

    io = pynwb.NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()
    ...
    units = nwb.units
    ...
    trials = nwb.trials
    ...
    be = nwb.acquisition['BehavioralEvents']
    go_times = be.time_series['go_start_times'].timestamps[:]
    ...
    bts = nwb.acquisition['BehavioralTimeSeries']
    tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
    ...
    io.close()
```
```python
for i, nwb_path in enumerate(nwb_files):
    print(f'\n[{i+1}/{len(nwb_files)}] Processing {os.path.basename(nwb_path)}')
    result = process_session(nwb_path, show_processing=show_processing and i < 2, session_idx=i)
```

iii. From CONVERSION_NOTES Step 2: the dataset is "28 subject directories (`sub-XXXXXX/`), 174 NWB files total", one session per file, so the directory listing is the complete set of sessions. The AI notes in Step 4 that the reference code loads DataJoint `.mat` files that are not shipped here, and resolves this as "Use NWB; same underlying data". The ragged `spike_times` column is read once per file through the `VectorIndex`/`VectorData` pair rather than one read per unit ("Fixed spike times loading: NWB uses VectorIndex wrapping VectorData; need `.target.data` for actual spikes").

## 1-b. How are the data split into subjects (mice)?

i. Each file's animal is read from `nwb.subject.subject_id` (the numeric DANDI id, e.g. `'440956'`); `nwb.subject.description` (the lab mouse name, e.g. `SC015`) is also carried along and stored in `metadata['session_info']` but is not used for grouping. `subjects` is built as the list of unique ids in order of first appearance (an `OrderedDict` used as an ordered set), and `subject_idx` indexes each session into that list. No re-grouping of sessions is needed because each file names its own subject. All 28 subjects survive filtering.

ii.
```python
subject_id = nwb.subject.subject_id
subject_desc = nwb.subject.description  # e.g., "SC015"
```
```python
subjects_set = OrderedDict()
...
    sid = result['subject_id']
    if sid not in subjects_set:
        subjects_set[sid] = len(subjects_set)
...
subjects = list(subjects_set.keys())
...
    subject_idx.append(subjects_set[sess['subject_id']])
subject_idx = np.array(subject_idx, dtype=np.int64)
```

iii. CONVERSION_NOTES Step 3/9 uses the subject count as an explicit sanity check against the paper: "Subjects | 28 | 28 | YES". `subject_id` is the canonical animal identifier in the NWB file and matches the containing `sub-*` folder name, so no inference is required.

## 1-c. How are the data split into sessions?

i. One NWB file is one session; no grouping or splitting is performed. Session order in the output is the sorted file order (subject, then acquisition timestamp encoded in the filename). Sessions are identified in metadata by their file path plus subject id. Sessions are *dropped* (not split) by three rules, applied in `process_session`: the data paper's behavioral selection criteria (overall performance > 65%, ≥50 correct left and ≥50 correct right trials), having no good units with a mappable brain region, and having fewer than 2 surviving trials. 174 files → 144 sessions kept (23 dropped for performance, 4 for correct-left, 2 for correct-right, 1 for no mapped good units).

ii.
```python
MIN_CORRECT_LEFT = 50   # session selection criterion
MIN_CORRECT_RIGHT = 50  # session selection criterion
MIN_PERFORMANCE = 0.65  # session selection criterion (65%)
```
```python
performance, correct_left, correct_right = compute_session_performance(data['trials_data'])
if performance < MIN_PERFORMANCE:
    print(f'    SKIP: performance {performance:.1%} < {MIN_PERFORMANCE:.0%}')
    return None
if correct_left < MIN_CORRECT_LEFT:
    ...
if correct_right < MIN_CORRECT_RIGHT:
    ...
```
```python
'session_info': [
    {'nwb_path': sess['nwb_path'], 'subject_id': sess['subject_id'],
     'subject_desc': sess['subject_desc'], 'n_neurons': sess['n_neurons'],
     'n_trials': sess['n_trials'], 'performance': sess['performance']}
    for sess in all_sessions
],
```

iii. The file boundary is the session boundary in this dandiset, so nothing has to be inferred. For the exclusions, CONVERSION_NOTES Step 5 lists "Session selection: Apply paper's criteria (>65% performance, >=50 correct each direction)", quoting the data paper: "We selected experimental sessions for analysis based on following criteria: overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each." Step 9 then reconciles the counts: "174 NWB files total; 173 have good units (matches paper's '173 behavioral sessions'); 144 pass selection criteria... Paper's '173' = total sessions with good units, not post-selection."

## 1-d. How are the data split into trials?

i. Trials are the rows of the NWB trials table (`nwb.trials`), and the go cue of trial *i* is element *i* of `BehavioralEvents/go_start_times`. Every per-trial variable is indexed by the same `trial_indices` array produced by the trial filters, so the trials table, the go-cue vector and the derived inputs/outputs stay in register. The AI explicitly verified the 1:1 correspondence between go cues and trial rows across sessions before relying on it (trajectory step 199: "Good, go_times and trials always match"), but did not leave an assertion in the shipped script.

ii.
```python
trials = nwb.trials
n_trials = len(trials)
trials_data = {
    'start_time': trials['start_time'][:],
    ...
}
...
go_times = be.time_series['go_start_times'].timestamps[:]
```
```python
trial_indices = np.where(trial_mask)[0]
n_valid_trials = len(trial_indices)
...
go_times = data['go_times'][trial_indices]
```

iii. CONVERSION_NOTES Step 3/5 records that the trials table holds one row per behavioral trial with the task variables needed as outputs, and that `sample_start_times` (unlike `go_start_times`) may contain several entries per trial because an early lick replays the sample epoch — so the go cue is used as the per-trial anchor. Trajectory step 59: "Go start times map 1:1 with trials (368 each)".

## 1-e. How are trials filtered based on quality controls?

i. Three filters, in this order:
1. **Reward-delivery trials**: `auto_water == 1` or `free_water == 1` are dropped (water given independently of the animal's choice).
2. **Trials outside the ephys recording**: after neuron selection, the earliest and latest spike time across the retained units defines the recorded interval; a trial is dropped unless its [go−2.5 s, go+1.5 s] window overlaps that interval.
3. **Session minimum**: if fewer than 2 trials survive, the whole session is dropped (target-format requirement).

No behavioral quality filter is applied at the trial level — early-lick, `ignore` and photostimulation trials are all kept because they are required decoder inputs/outputs. Sessions themselves are filtered on behavior (see 1-c). The final dataset has 74,768 trials in 144 sessions.

ii.
```python
# === Trial filtering: exclude auto_water and free_water ===
td = data['trials_data']
trial_mask = np.ones(data['n_trials'], dtype=bool)
trial_mask[td['auto_water'] == 1] = False
trial_mask[td['free_water'] == 1] = False

trial_indices = np.where(trial_mask)[0]
n_valid_trials = len(trial_indices)

if n_valid_trials < 2:
    print(f'    SKIP: only {n_valid_trials} valid trials')
    return None
```
```python
# === Exclude trials beyond recording range ===
# Some sessions have behavioral trials before recording starts or after it ends.
max_spike_time = max((st[-1] if len(st) > 0 else 0.0) for st in spike_times_list)
min_spike_time = min((st[0] if len(st) > 0 else float('inf')) for st in spike_times_list)
recording_mask = (go_times + ALIGN_START <= max_spike_time) & \
                 (go_times + ALIGN_END >= min_spike_time)
if not np.all(recording_mask):
    n_dropped = np.sum(~recording_mask)
    print(f'    Excluding {n_dropped} trials beyond recording range (max spike: {max_spike_time:.1f}s)')
    valid_positions = np.where(recording_mask)[0]
    trial_indices = trial_indices[valid_positions]
    go_times = go_times[valid_positions]
```

iii. CONVERSION_NOTES Step 4/5: the reference analysis "Excludes early lick, auto water, free water, ignore, stim" but "For our decoder: only exclude auto_water + free_water", because early lick, outcome and photostimulation are the very variables the decoder must predict or receive. The recording-range filter was added in Step 10 after the format checker reported 1,056 all-zero-neural trials; the AI traced one case to the ephys stopping mid-session (trajectory step 201: "Spike max is 1107.4s but go times go up to 3707.7s ... the recording ended at ~1107s but the behavioral session continued... This is genuine: the neural data truly doesn't exist for those trials"). After the fix the re-run reported 7 affected sessions, ~1,056 trials removed, and zero all-zero trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (session-absolute seconds, stored ragged as a `VectorIndex` over a flat `VectorData` buffer), restricted to units with `units/classification == 'good'` and a mappable `units/anno_name`. The go-cue times `BehavioralEvents/go_start_times` provide the per-trial alignment. `units/anno_name` additionally provides the brain-region label per neuron.

ii.
```python
spike_times_vi = units['spike_times']  # VectorIndex
all_spike_times = np.array(spike_times_vi.target.data[:])  # actual spike times
all_st_idx = np.array(spike_times_vi.data[:])  # end indices per unit

good_indices = np.where(good_mask)[0]
good_spike_times = []
for ui in good_indices:
    start_idx = 0 if ui == 0 else int(all_st_idx[ui - 1])
    end_idx = int(all_st_idx[ui])
    good_spike_times.append(all_spike_times[start_idx:end_idx])
```

iii. Spike times are the only neural representation in the NWB files (CONVERSION_NOTES Step 2 lists the units table columns). Step 6 notes the ragged-storage subtlety explicitly as a bug that had to be fixed.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz. For each neuron and each trial, `np.searchsorted` locates the spikes inside [go−2.5 s, go+1.5 s], the spikes are re-expressed relative to the go cue, `np.histogram` counts them into the 80 fixed 50 ms bins, and the counts are divided by the bin width. No smoothing, normalization, baseline subtraction, or trial averaging. Results are stored `float32` as one `(n_neurons, 80)` matrix per trial. (I verified numerically on `sub-456774_ses-20191021` that this produces exactly the same rates as the reference's flattened-edges `searchsorted` implementation: `np.allclose` → True, max abs diff 0.0.)

ii.
```python
bin_edges = np.linspace(align_start, align_end, n_bins + 1)
all_matrices = np.zeros((n_trials, n_neurons, n_bins), dtype=np.float32)

for n in range(n_neurons):
    st = spike_times_list[n]
    if len(st) == 0:
        continue
    for t in range(n_trials):
        go_t = go_times[t]
        abs_start = go_t + align_start
        abs_end = go_t + align_end
        idx_lo = np.searchsorted(st, abs_start, side='left')
        idx_hi = np.searchsorted(st, abs_end, side='left')
        if idx_hi > idx_lo:
            rel_spikes = st[idx_lo:idx_hi] - go_t
            counts, _ = np.histogram(rel_spikes, bins=bin_edges)
            all_matrices[t, n, :] = counts / bin_width

return [all_matrices[t] for t in range(n_trials)]
```

iii. CONVERSION_NOTES Step 1 identifies the reference's `sliding_histogram(..., rate=True)` (counts divided by bin width) as the corresponding function, and Step 5 keeps the rate semantics while replacing the reference's 40 ms/3.4 ms sliding window with the non-overlapping 50 ms bins required by the decoder task ("50ms bins: Task specification overrides reference code's 40ms/3.4ms sliding histogram").

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. (1) `units/classification == 'good'` — the verdict of the spike-sorting QC classifier; no individual metric (ISI violation, amplitude cutoff, drift) is thresholded separately. (2) Units whose `anno_name` does not match the AI's hand-written CCF→major-region keyword table are silently dropped as well, because `region_labels` and the spike-time list are built in the same loop. A session with zero surviving units is dropped. Result: 56,890 neurons over 144 sessions, median 390/session. On three spot-checked sessions the second filter removes ~3.5% of the classifier-good units (all "Lobules IV-V", i.e. cerebellar cortex missing from the keyword table).

ii.
```python
classifications = units['classification'][:]
good_mask = classifications == 'good'
anno_names = units['anno_name'][:]
```
```python
# === Neuron filtering: only good units with valid brain region ===
good_anno = data['good_anno_names']
region_labels = []
neuron_mask = []
for i, anno in enumerate(good_anno):
    region = map_anno_to_region(anno)
    if region is not None:
        region_labels.append(region)
        neuron_mask.append(i)

neuron_mask = np.array(neuron_mask)
n_neurons = len(neuron_mask)
if n_neurons < 1:
    print(f'    SKIP: no neurons with valid brain region')
    return None
```
```python
def map_anno_to_region(anno_name):
    """Map a CCF annotation name to a major brain region."""
    if not anno_name or anno_name.strip() == '':
        return None
    for region, keywords in REGION_MAPPING.items():
        for kw in keywords:
            if kw.lower() in anno_name.lower():
                return region
    return None  # unmapped
```

iii. CONVERSION_NOTES Step 1/3/4: "QC filtering uses classifier-trained logistic regression per brain area group... In NWB: `classification == 'good'` corresponds to the classifier QC output", cross-checked against the white-paper statistics ("QC pass rate 25.9%", "69,943 good units", "median = 393"). Step 9 reports median neurons/session = 390 vs the paper's 393 as a passing sanity check. The region requirement is justified only indirectly, via Step 5's decision to "Map detailed `anno_name` CCF annotations to these major regions" following the reference code's 14 area groups; the notes' "Neuron curation rules" section lists only the `classification == 'good'` rule and never states that unmapped units are discarded.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. To the go cue. Spike times and `go_start_times` are already on the same session-absolute clock, so alignment is a subtraction: for trial *t* the window is [go_t − 2.5 s, go_t + 1.5 s] and spikes inside it are re-expressed as `st - go_t` before binning on a fixed relative grid. No resampling, interpolation, or per-stream offset. Every other stream (tone, photostim, tongue) is put on the same go-cue-relative grid, so bin *k* means the same interval in all of them.

ii.
```python
ALIGN_START = -2.5  # seconds before go cue
ALIGN_END = 1.5     # seconds after go cue
```
```python
go_t = go_times[t]
abs_start = go_t + align_start
abs_end = go_t + align_end
idx_lo = np.searchsorted(st, abs_start, side='left')
idx_hi = np.searchsorted(st, abs_end, side='left')
if idx_hi > idx_lo:
    rel_spikes = st[idx_lo:idx_hi] - go_t
    counts, _ = np.histogram(rel_spikes, bins=bin_edges)
```
```python
'temporal_alignment_event': 'Go cue onset',
'off_start': ALIGN_START,  # -2.5s
'off_end': ALIGN_END,      # +1.5s
```

iii. Required by the Decoder Task section ("Temporally align based on Go cue onset. Extract 2.5 s before to 1.5 s after"). CONVERSION_NOTES Step 4 records the format difference from the reference code, where spikes were already stored go-cue-aligned in the `.mat` files: "Spike alignment | Already aligned to go cue in .mat | Absolute session time in NWB | Align using go_start_times". The `--show-processing` plots include a "Temporal alignment check" panel (histogram of tone-onset minus go cue, median ≈ −1.85 s = 0.65 s sample + 1.2 s delay) as a visual verification.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins per trial spanning −2.5 s to +1.5 s, identical for every trial and session; `metadata['time_bin_size'] = 50.0` ms. Since the source is continuous spike times, there is no rebinning of an existing grid — the grid is imposed once. The other streams are brought onto the same grid: tongue video (~294 Hz) is *downsampled* by averaging frames within each 50 ms bin, and the two inputs are evaluated at the 80 bin centers. The reference code's 40 ms/3.4 ms sliding histogram is deliberately not reproduced.

ii.
```python
BIN_WIDTH = 0.050  # 50 ms bins (decoder task spec)
ALIGN_START = -2.5  # seconds before go cue
ALIGN_END = 1.5     # seconds after go cue
N_BINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)  # 80 bins
```
```python
bin_edges = np.linspace(align_start, align_end, n_bins + 1)
```
```python
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
```
```python
'time_bin_size': BIN_WIDTH * 1000,  # in ms
```

iii. CONVERSION_NOTES Step 4 lists this as an intentional, documented discrepancy: "Bin width | bw=40ms, stride=3.4ms | ... | 'bin width of 40 ms' | Decoder task specifies 50ms bins; use 50ms", and Step 5 Key Decision 1 repeats it. The 80-bin constant length satisfies the target format's requirement that all trials share the same bin grid.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents/sample_start_times` (the onsets of the instructing tone) together with `go_start_times` and the trials table's `start_time`. For each trial the AI takes the **last** sample-start event that falls inside [trial start, go cue].

ii.
```python
def get_tone_onset_for_trials(go_times, sample_start_ts, trial_starts, trial_stops):
    """Get the last sample start time before go cue for each trial.

    The sample epoch may replay due to early licks, so we take the LAST
    sample start event within each trial's time range (before go cue).
    """
    n_trials = len(go_times)
    tone_onsets = np.full(n_trials, np.nan)

    for i in range(n_trials):
        mask = (sample_start_ts >= trial_starts[i]) & (sample_start_ts <= go_times[i])
        matching = sample_start_ts[mask]
        if len(matching) > 0:
            tone_onsets[i] = matching[-1]  # last sample start
    return tone_onsets
```

iii. CONVERSION_NOTES Step 5 maps "Sample start time relative to go cue → input[0]: time_from_tone_onset ... Use last sample_start before go cue for each trial". The reason for "last" is documented in Step 3/trajectory step 59: "Sample events can be repeated due to early licks (trial 1 has 2, trial 4 has 4) — need the LAST sample start before go cue", i.e. an early lick replays the sample epoch and the final replay is the tone the animal actually used. (I checked three sessions: this selection is identical to the reference's `sample[searchsorted(sample, go) - 1]` on every trial, and never returns NaN.)

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A continuous, time-varying value per bin: the bin center's absolute time minus the tone onset, i.e. `bin_center - (tone_onset - go_cue)`. Because the tone precedes the go cue by ≈1.85 s (0.65 s sample + 1.2 s delay), the values run from about −0.6 s in the first bin to about +3.3 s at the go cue and +4.8 s at the end of the window, longer on trials with replayed sample epochs. Values are stored `float32` as row 0 of the `(2, 80)` input array. If no tone onset were found, the row would be filled with zeros.

ii.
```python
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)

for t in range(n_trials):
    go_t = go_times[t]
    tone_t = tone_onsets[t]
    if np.isnan(tone_t):
        # If no tone onset found, set to 0 (shouldn't happen for valid trials)
        tone_input = np.zeros(n_bins, dtype=np.float32)
    else:
        # Time from tone onset = (absolute time of bin center) - tone_onset
        #                      = bin_center - (tone_t - go_t)
        tone_rel = tone_t - go_t  # tone onset relative to go cue (negative)
        tone_input = (bin_centers - tone_rel).astype(np.float32)
    tone_onset_input.append(tone_input)
```
```python
inp = np.stack([tone_onset_input[t], photostim_input[t]], axis=0).astype(np.float32)
```

iii. The Decoder Task asks for "Time from tone onset in seconds (continuous, time-varying)", so no discretization is applied. CONVERSION_NOTES Step 7/9 uses the resulting range as a sanity check (sample range [−1.5, 9.4]; full-data range [−1.5, 11.9]), consistent with a 1.85 s tone-to-go gap plus long replayed sample epochs on early-lick trials.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on exactly the same go-cue-relative grid used to bin the spikes: the 80 bin centers are the 80 spike-bin centers, offset by the trial's tone-to-go interval. Alignment therefore needs no extra step — bin *k* of the input covers the same interval as bin *k* of the firing rates.

ii.
```python
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
...
tone_input = (bin_centers - tone_rel).astype(np.float32)
```
```python
bin_edges = np.linspace(align_start, align_end, n_bins + 1)   # same grid used for spikes
```

iii. Everything in the NWB file shares one session-absolute clock (CONVERSION_NOTES Step 4), so the only alignment needed is per-trial subtraction of the go-cue time. The `--show-processing` plot "Input: Time from tone onset" overlays 10 trials on the go-cue axis with the go cue marked, and the alignment-check histogram confirms a tight ≈−1.85 s tone-to-go offset.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The measured event streams `BehavioralEvents/photostim_start_times` and `photostim_stop_times` (session-absolute timestamps of each stimulation epoch), not the trials-table `photostim_onset`/`photostim_duration` columns. The trials-table columns are loaded and used only in the session-performance computation, to identify control trials. (I checked one session: the two sources give bit-identical binary time series — 96 stim trials, 960 stim bins, 100% bin agreement.)

ii.
```python
# Photostim events
photostim_start_ts = be.time_series['photostim_start_times'].timestamps[:]
photostim_stop_ts = be.time_series['photostim_stop_times'].timestamps[:]
```
```python
photostim_input = compute_photostim_input(go_times, data['photostim_start_ts'],
                                          data['photostim_stop_ts'],
                                          ALIGN_START, ALIGN_END, BIN_WIDTH, N_BINS)
```

iii. CONVERSION_NOTES Step 5 maps "Photostim start/stop times → input[1]: photostim_on | Binary time-varying, 1 when photostim active | From photostim_start/stop_times". Step 3 records the expected structure from the paper ("~25% of trials randomly interleaved (17 VGAT-ChR2-EYFP mice), 40Hz sinusoidal, 5mW, during late delay (last 0.5s)"), which the event stream reproduces directly (0.5 s duration ending at the go cue), so the measured onsets/offsets are used rather than reconstructing them from the string-typed trials columns.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1) time series per trial: for every trial the code walks all photostim events in the session, skips those that cannot overlap the trial window, and sets to 1 every bin whose center falls in [start, stop) of a surviving event. Non-stimulated trials stay all-zero. Stored as `float32` in row 1 of the input array.

ii.
```python
def compute_photostim_input(go_times, photostim_start_ts, photostim_stop_ts,
                            align_start, align_end, bin_width, n_bins):
    """Compute binary photostimulation time series for each trial.
    1 when photostim is active, 0 otherwise.
    """
    bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
    photostim_trials = []
    for t in range(n_trials):
        go_t = go_times[t]
        ps = np.zeros(n_bins, dtype=np.float32)
        for si in range(len(photostim_start_ts)):
            ps_start = photostim_start_ts[si] - go_t
            ps_stop = photostim_stop_ts[si] - go_t
            if ps_stop < align_start or ps_start > align_end:
                continue
            for b in range(n_bins):
                bc = bin_centers[b]
                if bc >= ps_start and bc < ps_stop:
                    ps[b] = 1.0
        photostim_trials.append(ps)
    return photostim_trials
```

iii. The Decoder Task requires "Whether photostimulation is on at every time point (discrete, time-varying)", and the target format says "If an input is a time such as onset of some stimulus, represent it as a binary time series". CONVERSION_NOTES Step 7/9 checks the resulting range is exactly [0, 1] and the `--show-processing` plot reports the fraction of stimulated trials per session.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The absolute event times are converted to go-cue-relative times by subtracting the same `go_t` used for the spikes, then compared against the same 80 bin centers. Bin *k* of the photostim row therefore covers the same interval as bin *k* of the firing rates; stimulation appears in the last ~10 bins before the go cue, as expected for late-delay inhibition.

ii.
```python
ps_start = photostim_start_ts[si] - go_t
ps_stop = photostim_stop_ts[si] - go_t
...
if bc >= ps_start and bc < ps_stop:
    ps[b] = 1.0
```

iii. Same single-clock argument as 2-d/3-c; no interpolation or offset correction is needed. The mean photostim trace is plotted against the go-cue axis in the `--show-processing` figure so the late-delay timing can be verified visually.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. From a single trials-table column, `trial_instruction` (`'left'`/`'right'`) — the side the tone *instructed*. The animal's actual response is not consulted: `outcome` is not combined in, and `left_lick_times`/`right_lick_times` are loaded from the NWB file but never used for this output. There is consequently no "no lick" class; `output_values[0] = ['left', 'right']`.

ii.
```python
# 1. Choice: left=0, right=1
instructions = td['trial_instruction'][trial_indices]
choices = np.array([0 if ins == 'left' else 1 for ins in instructions], dtype=np.int64)
```
```python
'output_values': [
    ['left', 'right'],                    # choice: 0=left, 1=right
    ...
```
```python
# Lick times  (loaded, never used)
left_lick_ts = be.time_series['left_lick_times'].timestamps[:]
right_lick_ts = be.time_series['right_lick_times'].timestamps[:]
```

iii. CONVERSION_NOTES Step 5 states the mapping flatly: "`trial_instruction` | `output[0]`: choice | left=0, right=1 (per-trial)". The trajectory (step 59) shows the same reasoning — "converting trial_instruction from 'left'/'right' strings to the decoder's left=0, right=1 format" — with no consideration of miss trials (where the animal licked the *other* side) or ignore trials (no lick). The notes' Step 3 table does record that "Left lick vs Right lick (instructed by tone frequency)" and that outcomes are hit/miss/ignore, but the two are never combined, and the required third class is never mentioned anywhere in the notes or trajectory.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The string is mapped to `0`/`1` by a list comprehension (anything that is not the literal `'left'` becomes `1`), and the single per-trial value is broadcast across all 80 bins so that all four outputs can live in one `(4, 80)` `int64` array per trial.

ii.
```python
choices = np.array([0 if ins == 'left' else 1 for ins in instructions], dtype=np.int64)
...
out = np.array([
    np.full(N_BINS, choices[t], dtype=np.int64),         # choice (per-trial, broadcast)
    np.full(N_BINS, outcomes[t], dtype=np.int64),        # outcome (per-trial, broadcast)
    np.full(N_BINS, early_licks[t], dtype=np.int64),     # early lick (per-trial, broadcast)
    tongue_y_discrete[t].astype(np.int64),                # tongue y (time-varying)
], dtype=np.int64)
```

iii. `left = 0`, `right = 1` follows the ordering given in the Decoder Task. Broadcasting per-trial values across bins is the AI's way of satisfying the target format's "If at all possible, make it time-varying" while keeping a single rectangular output array. The resulting distribution is reported in Step 9/verification as ≈49.5% left / 50.5% right, which the notes accept as reasonable — a balance that is expected for the *instruction* but not for the animal's choice.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the trials-table `outcome` column, which already contains the strings `'hit'`, `'miss'` and `'ignore'`.

ii.
```python
'outcome': trials['outcome'][:],
```
```python
outcomes_raw = td['outcome'][trial_indices]
```

iii. CONVERSION_NOTES Step 2/3 records the column and its three values ("Outcomes: hit (correct), miss (error), ignore (no response)"), which are exactly the three categories the Decoder Task asks for, so no derivation is needed. Step 4 notes the reference code's internal coding (hit=1/miss=0/ignore=−1) and that it must be remapped to the decoder's ordering.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary maps `ignore→0`, `miss→1`, `hit→2` (with `0` as the fallback for any unexpected string), and the per-trial value is broadcast across the 80 bins into row 1 of the output array. `output_values[1] = ['ignore', 'miss', 'hit']`.

ii.
```python
# 2. Outcome: ignore=0, miss=1, hit=2
outcomes_raw = td['outcome'][trial_indices]
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcomes = np.array([outcome_map.get(o, 0) for o in outcomes_raw], dtype=np.int64)
```
```python
np.full(N_BINS, outcomes[t], dtype=np.int64),        # outcome (per-trial, broadcast)
```

iii. The code assignment follows the Decoder Task ordering ("ignore, miss, hit"). Trajectory step 59: "hit corresponds to correct (1), miss to error (0), and ignore to no response (−1) in the reference code, but the decoder expects ignore=0, miss=1, hit=2". The resulting distribution (ignore 10.7%, miss 15.2%, hit 74.1%) is reported in Step 9.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the trials-table `early_lick` column, whose values are the strings `'no early'` and `'early'`.

ii.
```python
'early_lick': trials['early_lick'][:],
```
```python
early_lick_raw = td['early_lick'][trial_indices]
```

iii. The flag is stored explicitly per trial (CONVERSION_NOTES Step 2), so nothing has to be derived. Step 3 records the behavioral meaning — "Early lick: licking during sample/delay triggers replay" — which also explains why the event itself falls inside the −2.5 s pre-go window even though the label is per-trial.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to `1` if the string is exactly `'early'`, else `0`, then broadcast across the 80 bins into row 2 of the output array. `output_values[2] = ['no', 'yes']`.

ii.
```python
# 3. Early lick: no=0, yes=1
early_licks = np.array([1 if el == 'early' else 0 for el in early_lick_raw], dtype=np.int64)
```
```python
np.full(N_BINS, early_licks[t], dtype=np.int64),     # early lick (per-trial, broadcast)
```

iii. Coding follows the Decoder Task ("Early lick (no, yes, per-trial)"). Early-lick trials are deliberately *kept* in the dataset (CONVERSION_NOTES Step 5 Key Decision 2: "Keep early lick/ignore/stim trials: These are decoder outputs/inputs, not filtered") even though the reference analyses exclude them. Resulting distribution: 88.6% no / 11.4% yes.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, a `(n_frames, 3)` array of `(tongue_x, tongue_y, confidence)` sampled at ~294 Hz with its own timestamps. Column 1 is the y-position; column 2 is the DeepLabCut likelihood used to decide whether the tongue is visible.

ii.
```python
bts = nwb.acquisition['BehavioralTimeSeries']
tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts_obj.data[:]  # (n_frames, 3): x, y, confidence
tongue_timestamps = tongue_ts_obj.timestamps[:]
```
```python
tongue_y = tongue_data[:, 1].astype(np.float64)
tongue_conf = tongue_data[:, 2].astype(np.float64)
```

iii. CONVERSION_NOTES Step 2 identifies the series and its channel layout ("TongueTracking (x, y, confidence) at ~294 Hz"); the side view is the one the method paper uses ("For analysis, only the side-view frames were used"). It is the only tongue measurement in the file.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Four steps. (1) A session-wide "visible" mask is formed from `confidence >= 0.9`, and the mean y over visible frames is computed. (2) Every non-visible frame's y is **replaced by that session mean** (imputation, not exclusion). (3) For each trial, frames whose timestamps fall in [go−2.5 s, go+1.5 s) are assigned to bins and averaged within each 50 ms bin; a bin with no frames keeps the session mean, and a trial with no frames at all becomes a constant session-mean vector. (4) The binned continuous values are then discretized (8-c). Because only ~8–10% of frames are visible, ~74% of all output bins end up at exactly the session mean.

ii.
```python
TONGUE_CONFIDENCE_THRESHOLD = 0.9  # for occlusion detection
```
```python
visible_mask = tongue_conf >= confidence_threshold
if np.sum(visible_mask) > 0:
    session_mean_y = np.mean(tongue_y[visible_mask])
else:
    session_mean_y = np.mean(tongue_y)

# Replace non-visible tongue y with session mean (vectorized)
tongue_y_imputed = tongue_y.copy()
tongue_y_imputed[~visible_mask] = session_mean_y
```
```python
for t in range(n_trials):
    go_t = go_times[t]
    window_start = go_t + align_start
    window_end = go_t + align_end
    mask = (tongue_timestamps >= window_start) & (tongue_timestamps < window_end)
    trial_ts = tongue_timestamps[mask] - go_t  # relative to go cue
    trial_y = tongue_y_imputed[mask]

    if len(trial_ts) == 0:
        tongue_y_trials.append(np.full(n_bins, session_mean_y, dtype=np.float32))
        continue

    bin_indices = np.digitize(trial_ts, bin_edges) - 1  # 0-indexed bins
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    trial_tongue_y = np.full(n_bins, session_mean_y, dtype=np.float64)
    for b in range(n_bins):
        in_bin = trial_y[bin_indices == b]
        if len(in_bin) > 0:
            trial_tongue_y[b] = np.mean(in_bin)
```

iii. The imputation is taken verbatim from the method paper, which the AI quoted in the docstring and in CONVERSION_NOTES Step 4/5: "When the tongue was occluded while it was in the mouth, as was typically the case before the response epoch, we set the tongue position to its mean value" (Key Decision 4: "Tongue occlusion handling: Set tongue position to session mean when confidence < 0.9"). The 0.9 threshold is not separately justified; the likelihood is in practice near-binary, so 0.5 and 0.9 select nearly the same frames (I measured 8.21% vs 8.11% of frames on one session).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, the 40th and 60th percentiles are taken over **all binned values of that session, imputed bins included**, and each bin is assigned `0` if below p40, `1` if in [p40, p60), `2` if ≥ p60. Because the imputed session mean dominates, p40 and p60 collapse to the same value in essentially every session; the code detects this and pushes them apart by a small epsilon, which makes class 1 the "value == session mean" class, i.e. de facto "tongue not visible in this bin". There is no fourth class: `output_values[3] = ['low', 'mid', 'high']`. Resulting distribution over the full dataset: 15.9% / 74.3% / 9.7%.

ii.
```python
def discretize_tongue_y(tongue_y_trials, session_mean_y):
    """...
    Percentiles are computed over ALL time bins (including imputed values).
    """
    all_values = np.concatenate([t for t in tongue_y_trials])

    p40 = np.percentile(all_values, 40)
    p60 = np.percentile(all_values, 60)

    # Ensure thresholds differ to get 3 categories
    # If p40 == p60 (common when tongue is mostly at mean), adjust slightly
    if np.isclose(p40, p60):
        eps = max(1e-6, abs(p40) * 1e-4)
        p40 = p40 - eps
        p60 = p60 + eps

    discretized = []
    for trial_y in tongue_y_trials:
        d = np.zeros(len(trial_y), dtype=np.int64)
        d[trial_y >= p40] = 1
        d[trial_y >= p60] = 2
        discretized.append(d)
    return discretized
```

iii. The 40/60 split and the per-session scope follow the Decoder Task spec. The epsilon hack is documented in CONVERSION_NOTES Step 6 ("Fixed tongue y discretization: when p40==p60 (common with imputed values), add small offset for 3 categories"). The trajectory shows the AI diagnosing the root cause correctly — "When the tongue is occluded, I'm filling it with the mean value, which then becomes extremely common... I should compute the percentile thresholds only from the visible tongue positions, not the imputed ones, so the discretization boundaries actually reflect real tongue movement" (step 121) — but the edit it then applied (step 123) kept the percentiles over all imputed values and only added the epsilon, and the notes were updated to describe that behaviour rather than the diagnosed fix.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. On the same go-cue-relative grid as everything else: camera timestamps share the session-absolute clock, so the trial's frames are selected with `[go−2.5 s, go+1.5 s)` and placed into bins by `np.digitize` against the same `bin_edges` used for the spikes (clipped to [0, 79]). Bin *k* of the tongue output therefore spans the same interval as bin *k* of the firing rates. Where the video is off (the camera is trial-gated, so bins before trial start contain no frames), the bin falls back to the session mean and thus to class 1.

ii.
```python
bin_edges = np.linspace(align_start, align_end, n_bins + 1)
...
mask = (tongue_timestamps >= window_start) & (tongue_timestamps < window_end)
trial_ts = tongue_timestamps[mask] - go_t  # relative to go cue
...
bin_indices = np.digitize(trial_ts, bin_edges) - 1  # 0-indexed bins
bin_indices = np.clip(bin_indices, 0, n_bins - 1)
```

iii. Same one-clock argument as 2-d. The `--show-processing` figure plots the raw binned tongue y and the discretized trace on the same go-cue axis for an example trial so that the alignment and the discretization can be checked visually.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Five cases:
- **Session never quality-controlled** (`classification`/`anno_name` NaN): the `== 'good'` comparison yields nothing, no unit maps to a region, and the session is dropped with "SKIP: no neurons with valid brain region" (exactly 1 session).
- **Behavioral trials outside the ephys recording**: detected from the first/last spike time and dropped (7 sessions, ~1,056 trials).
- **Units with no spikes**: skipped in the binning loop (`if len(st) == 0: continue`), leaving their rows at 0 Hz.
- **Occluded tongue frames / empty tongue bins / trials with no video**: imputed with the session mean (see 8-b).
- **No tone onset found for a trial**: the input row is filled with zeros. (I checked three sessions; this never triggers — the tone selection matches the reference on every trial.)
- **Unexpected `outcome` strings**: `outcome_map.get(o, 0)` silently codes them as `ignore`.
The AI also adjusted its CCF keyword table twice after finding mis- and un-mapped annotations ("Midbrain reticular nucleus" matching the Thalamus rule; "Nucleus of the lateral lemniscus", "Septofimbrial nucleus", "Dorsal peduncular area", "Nucleus of the brachium of the inferior colliculus" unmapped).

ii.
```python
if n_neurons < 1:
    print(f'    SKIP: no neurons with valid brain region')
    return None
```
```python
st = spike_times_list[n]
if len(st) == 0:
    continue
```
```python
if np.isnan(tone_t):
    # If no tone onset found, set to 0 (shouldn't happen for valid trials)
    tone_input = np.zeros(n_bins, dtype=np.float32)
```
```python
if len(trial_ts) == 0:
    tongue_y_trials.append(np.full(n_bins, session_mean_y, dtype=np.float32))
    continue
```
```python
outcomes = np.array([outcome_map.get(o, 0) for o in outcomes_raw], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 10 documents the two data problems that were actually found (all-zero neural trials, brain-region mis-mapping) and their fixes, and Step 9 claims the outcome: "No zero-neural-data trials (fixed by excluding trials beyond recording range)". The general principle the AI applies is: drop the unit of data where nothing was recorded (session, trial), and impute where the measurement exists but the feature is absent (occluded tongue), the latter following the method paper's stated imputation policy.

## 10-a. What are the most time-consuming steps of the code?

i. The script prints per-step timers, and the full run took 1,171 s (19.5 min) for 174 files. Per kept session: NWB loading ~1.5–2.3 s, spike binning ~3–4 s, tongue processing ~1.5–2.2 s. Spike binning is the single largest cost, followed by file I/O (reading the whole flat `spike_times` buffer plus the ~700k×3 tongue array), then the tongue stage. The final pickle is 9.3 GB and writing it adds tens of seconds. For comparison, the reference conversion of the same 174 files takes ~247 s, so the AI's script is ~4.7× slower and exceeds the 15-minute budget the instructions set.

ii.
```python
t0 = time.time()
...
data = load_nwb_session(nwb_path)
t_load = time.time() - t0
...
t1 = time.time()
neural_trials = bin_spikes(spike_times_list, go_times, ALIGN_START, ALIGN_END, BIN_WIDTH, N_BINS)
t_bin = time.time() - t1
print(f'    Spike binning: {t_bin:.1f}s')
...
t_tongue = time.time() - t2
print(f'    Tongue processing: {t_tongue:.1f}s')
```

iii. CONVERSION_NOTES Step 7 tabulates exactly these three stages with an estimated total of ~22 min, and Step 6 lists the two optimizations that were applied: "Optimized tongue processing: vectorized using np.digitize instead of per-bin loop (142s -> 1.5s)" and "Optimized spike binning: use np.searchsorted for binary search (14s -> 3s per session)". The AI accepted the resulting ~20 min runtime without a further optimization pass.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python loops remain that numpy could absorb:
- `bin_spikes`: a nested neuron × trial loop calling `np.histogram` once per (neuron, trial) — ~375 × 500 ≈ 190k calls per session. The reference collapses the trial dimension by flattening all 81 edges of all trials into one array and issuing a single `searchsorted` per neuron.
- `compute_photostim_input`: a triple loop over trials × all session photostim events × 80 bins, i.e. ~500 × 100 × 80 = 4M pure-Python iterations per session, where a single broadcast comparison against `bin_centers` would do.
- `compute_tongue_y_per_trial`: a per-trial boolean mask over the *entire* ~700k-frame timestamp array (O(n_trials × n_frames)) plus an inner per-bin loop `trial_y[bin_indices == b]` that rescans the index array 80 times per trial; `searchsorted` + `np.bincount` replaces both.
- `get_tone_onset_for_trials` and `compute_tone_onset_input`: per-trial full-array masking and per-trial array construction, both replaceable by one `searchsorted` and one broadcast.

Two of these functions carry docstrings claiming they are already vectorized, which they are not.

ii.
```python
def bin_spikes(...):
    """...
    Vectorized implementation: processes all trials for each neuron at once.
    """
    for n in range(n_neurons):
        ...
        for t in range(n_trials):
            ...
            counts, _ = np.histogram(rel_spikes, bins=bin_edges)
```
```python
    for t in range(n_trials):
        ...
        for si in range(len(photostim_start_ts)):
            ...
            for b in range(n_bins):
                bc = bin_centers[b]
                if bc >= ps_start and bc < ps_stop:
                    ps[b] = 1.0
```
```python
def compute_tongue_y_per_trial(...):
    """...
    Vectorized implementation for efficiency.
    """
    for t in range(n_trials):
        mask = (tongue_timestamps >= window_start) & (tongue_timestamps < window_end)
        ...
        for b in range(n_bins):
            in_bin = trial_y[bin_indices == b]
```

iii. CONVERSION_NOTES Step 6 records partial vectorization work ("vectorized using np.digitize instead of per-bin loop", "use np.searchsorted for binary search") and Step 7 concludes the ~22 min estimate is acceptable, so no further loop removal was attempted. The instructions asked for optimization if the estimate exceeded 15 minutes; the AI's own estimate exceeded it and the realized runtime was 19.5 min.

## 10-c. What processing does the code repeat multiple times?

i. Mostly small, but several things are recomputed:
- The bin grid: `np.linspace` for edges or centers is rebuilt inside `bin_spikes`, `compute_tongue_y_per_trial`, `compute_photostim_input`, `compute_tone_onset_input` and `plot_processing`, once per call per session, instead of being a module-level constant as in the reference.
- `map_anno_to_region` rescans the whole 14-region keyword table for every neuron, calling `anno_name.lower()` once per keyword (hundreds of `str.lower()` calls per neuron, ~400 neurons per session) with no memoization over the handful of distinct annotation strings.
- The tongue values are traversed repeatedly: once for the session mask/mean, once per trial for window masking, 80 times per trial in the bin loop, and once more concatenated in `discretize_tongue_y` (which rebuilds the full value array that was just produced).
- `compute_session_performance` loops over trials in Python to build the control mask before the vectorized part.
Each NWB file is, correctly, opened and read only once; there is no second pass over the data.

ii.
```python
bin_edges = np.linspace(align_start, align_end, n_bins + 1)          # bin_spikes
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)  # x3 functions
```
```python
for region, keywords in REGION_MAPPING.items():
    for kw in keywords:
        if kw.lower() in anno_name.lower():
```
```python
all_values = np.concatenate([t for t in tongue_y_trials])
```
```python
for i in range(len(outcomes)):
    if auto_water[i] == 1 or free_water[i] == 1:
        control_mask[i] = False
    if photostim_onset[i] != 'N/A':
        control_mask[i] = False
```

iii. Not discussed in CONVERSION_NOTES; the notes' efficiency discussion (Step 6/7) covers only the two loop optimizations and the total runtime estimate. None of these repetitions is a first-order cost — the dominant cost is the per-(neuron, trial) histogram call identified in 10-b.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several loaded or computed quantities never reach the output and are not needed for any decision:
- `left_lick_times` and `right_lick_times` are read out of the NWB file on every session and never used (real I/O cost, and ironically these are the streams that would have given the actual lick direction for the choice output).
- `photostim_power`, `stop_time`, `trial_stops` (passed to `get_tone_onset_for_trials` and unused), `good_indices`, `n_total_units` are loaded/passed and unused.
- `tongue_data[:, 0]` (tongue x) is read as part of the `(n_frames, 3)` block but unused; the imputation is applied to the whole session's frames although only the ~3–4% of frames inside trial windows are ever binned.
- The continuous binned tongue y is computed and returned, then only its discretized form is stored (the continuous version is used only by the optional plotting).
- Outputs are stored as `int64` where values are in 0–3, 8× larger than the `int8` the reference uses; this inflates the 9.3 GB pickle by a few hundred MB.
- Session performance is computed for every session including ones that will be dropped for other reasons (unavoidable, since it is the drop criterion itself).

ii.
```python
# Lick times
left_lick_ts = be.time_series['left_lick_times'].timestamps[:]
right_lick_ts = be.time_series['right_lick_times'].timestamps[:]
```
```python
def get_tone_onset_for_trials(go_times, sample_start_ts, trial_starts, trial_stops):
    ...  # trial_stops never referenced
```
```python
tongue_y_imputed = tongue_y.copy()
tongue_y_imputed[~visible_mask] = session_mean_y   # whole session, incl. inter-trial frames
```
```python
out = np.array([...], dtype=np.int64)   # values only ever 0..3
```

iii. Not discussed in CONVERSION_NOTES. The lick-time and photostim-power loads look like leftovers from the exploration phase, when the AI was surveying available variables (CONVERSION_NOTES Step 2 enumerates all behavioral event streams); none of them was subsequently removed during the Step 13 cleanup.
