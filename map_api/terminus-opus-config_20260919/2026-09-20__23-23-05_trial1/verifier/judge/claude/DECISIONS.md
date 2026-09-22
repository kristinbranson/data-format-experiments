# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset (DANDI:000363) is one NWB file per session under `/app/data/sub-<subject_id>/`. The AI enumerates every file with a single sorted `glob` over that layout and processes each file exactly once, in a `multiprocessing.Pool` (16 workers by default). Each file is opened with `pynwb.NWBHDF5IO(..., load_namespaces=True)` inside a `with` block, and everything needed (trials table, behavioural event time series, units table, electrodes, DeepLabCut video series) is read from that single handle. `--sample` forces serial execution and stops once 2 sessions have passed curation; `--full` (default) processes all 174 files.

ii.
```python
DATA_DIR = '/app/data'
...
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
if args.sample:
    # keep the first files needed to obtain 2 sessions that pass curation
    args.nproc = 1
...
jobs = [(f, args.show_processing and (args.sample or i < 2), plot_dir) for i, f in enumerate(files)]
with Pool(args.nproc) as pool:
    for i, res in enumerate(pool.imap_unordered(process_session, jobs)):
        results.append(res)
```

```python
with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
    nwb = io.read()
    sess_name = nwb.identifier
    subject = nwb.subject.description or nwb.subject.subject_id
    ...
    trials = nwb.trials
    bev = nwb.acquisition['BehavioralEvents'].time_series
    units = nwb.units
    bts = nwb.acquisition['BehavioralTimeSeries'].time_series
```

iii. From CONVERSION_NOTES Step 2: "`/app/data` is a DANDI download of dandiset 000363 ... organised as `sub-<subject_id>/sub-<subject_id>_ses-...nwb`. **174 NWB files** = 174 sessions, **28 subjects**, 50 GB total. Every file read with `pynwb.NWBHDF5IO(..., load_namespaces=True)`." The instructions mandate `pynwb` rather than `h5py`, and since the DANDI layout stores one session per file the directory listing is by construction the complete set of sessions. Multiprocessing was added as an efficiency measure (Step 6/7: "16-way multiprocessing over sessions ~10× wall-clock"); the full conversion takes 35 s.

## 1-b. How are the data split into subjects?

i. Each session's subject is taken from `nwb.subject.description`, which holds the mouse nickname used in the papers (e.g. `SC015`), falling back to `nwb.subject.subject_id` (the numeric DANDI id, e.g. `440956`) if the description is empty. At assembly, `subjects` is the sorted set of unique names and `subject_idx` is each kept session's index into that list. This yields 28 subjects, all of which are represented in the final dataset.

ii.
```python
subject = nwb.subject.description or nwb.subject.subject_id
```

```python
kept.sort(key=lambda r: (r['subject'], r['sess']))
...
subjects = sorted({r['subject'] for r in kept})
subject_idx = np.array([subjects.index(r['subject']) for r in kept], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 2 records that `nwb.subject` carries both `subject_id` (e.g. 440956) and `description` = mouse nickname (e.g. `SC015`). The Step 5 mapping table justifies the choice: "`nwb.subject.description` (e.g. `SC015`) → `subjects`, `subject_idx` ... matches how the papers name mice" (and the reference `.mat` exports in `/app/code`). Sessions are sorted by `(subject, session)` before assembly so that a subject's sessions are contiguous and chronological.

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no grouping or boundary inference is needed. The session is identified by `nwb.identifier` (e.g. `SC015_20190207_120657_s1`, encoding mouse/date/time/session number). The AI then applies **session-level curation**, keeping only 138 of the 174 sessions:
  1. at least one QC-`good` unit (drops 1 session, `SC017_20190216_162508_s4`);
  2. the data paper's behavioural session-selection criteria — overall performance > 65 % **and** ≥ 50 correct lick-left and ≥ 50 correct lick-right control trials (drops 29 sessions);
  3. side-view video must cover the analysis window on at least 50 % of observed trials (drops 6 sessions whose video stops at/near the go cue);
  4. at least 2 usable trials after trial filtering.

ii.
```python
MIN_PERFORMANCE = 0.65
MIN_CORRECT_PER_DIRECTION = 50
MIN_SESSION_VIDEO_FRAC = 0.5  # a session must have video for at least this fraction of trials
```

```python
hit = outcome == 'hit'
miss = outcome == 'miss'
ctrl = ((early_lick == 'no early') & (auto_water == 0) & (free_water == 0)
        & ~np.isin(np.arange(n_trials), np.searchsorted(start_time, photostim_on, side='right') - 1))
responded = (hit | miss) & ctrl
performance = hit[responded].sum() / max(responded.sum(), 1)
n_correct_left = int((hit & ctrl & (instruction == 'left')).sum())
n_correct_right = int((hit & ctrl & (instruction == 'right')).sum())
...
if n_good == 0:
    info['skipped'] = 'no good units'
    return info
if not (performance > MIN_PERFORMANCE
        and n_correct_left >= MIN_CORRECT_PER_DIRECTION
        and n_correct_right >= MIN_CORRECT_PER_DIRECTION):
    info['skipped'] = 'behavioural session criteria'
    return info
```

```python
if np.mean(coverage[observed] >= MIN_VIDEO_COVERAGE) < MIN_SESSION_VIDEO_FRAC:
    info['skipped'] = 'video does not cover the analysis window'
    return info
...
if nk < 2:
    info['skipped'] = 'fewer than 2 usable trials'
    return info
```

iii. CONVERSION_NOTES Step 5, Key Decision 3: "Session curation (138 of 174 kept): ≥ 1 good unit (removes `SC017...`; leaves 173, exactly the papers' session count); the papers' behavioural session-selection criteria: performance > 65 % and ≥ 50 correct lick-left and ≥ 50 correct lick-right trials (145 sessions; over these sessions the mean performance is 84.0 % and the maximum number of responded trials is 785, both exactly as reported); video available for the analysis window (removes 6 sessions in which the side-view video stops at the go cue, so the tongue output would be undefined for the whole post-go period)." Step 9 adds: "Applying the papers' own criteria is the consistent choice; the remaining 35 sessions are excluded for the same reasons the papers excluded them." The AI's supporting evidence is that recomputing performance over its 145 passing sessions gives 84.0 % (paper: 84 %, range 65–99 %) and max 785 responded trials (paper: range 130–785), whereas over all 174 sessions the mean is 80.6 %.

## 1-d. Are the data correctly split into trials?

i. Trials come from the NWB trials table (`nwb.trials`), one row per behavioural trial. Rather than assuming a 1:1 ordering, the AI maps each `go_start_times` event to the trial whose `[start_time, ...)` interval contains it, and fails the session if any trial lacks a go cue. The same interval search is used to assign tone onsets, photostim events and observation intervals to trials. The AI separately verified (Step 2) that `go_start_times` has exactly one entry per trial in all 174 files.

ii.
```python
trials = nwb.trials
n_trials = len(trials)
start_time = np.asarray(trials['start_time'].data[:])
...
go_times_all = np.asarray(bev['go_start_times'].timestamps[:])
# one go cue per trial (verified for every session in the dataset)
go = np.full(n_trials, np.nan)
gi = np.searchsorted(start_time, go_times_all, side='right') - 1
ok = (gi >= 0) & (gi < n_trials)
go[gi[ok]] = go_times_all[ok]
if np.any(np.isnan(go)):
    return dict(sess=sess_name, skipped='missing go cue')
```

iii. CONVERSION_NOTES Step 2: "Exactly **one go cue per trial** (`go_start_times` length == n_trials in all 174 sessions)" and "`sample_start_times` can occur **several times per trial** (tone epoch is replayed after an early lick)". Because some event streams are *not* one-per-trial, the AI chose to bin every event stream into trials by interval containment rather than by position, and kept an explicit guard for a trial with no go cue.

## 1-e. How are trials filtered based on quality controls?

i. Within a kept session, three trial filters are applied (4,540 of 76,791 trials, 5.9 %):
  1. **Not observed by the ephys** — `units/obs_intervals` lists the trials during which each unit was recorded; the AI intersects the observed-trial masks over *all* good units and drops trials outside the intersection (1,056 trials, concentrated in 8 sessions where recording started late).
  2. **Insufficient video** — a trial is dropped if the side-view camera covers < 90 % of the −2.5…+1.5 s window (344 trials). A trial with no tone onset before the go cue would also be dropped (`np.isfinite(tone_onset)`), though this never occurs.
  3. **No spikes at all** — after binning, trials with zero spikes from every good unit are dropped (2,150 trials) as genuine acquisition dropouts. This filter also removes free-water trials, which carry no spikes.
  No behavioural trial filter is applied: photostimulation, early-lick, ignore and auto/free-water trials are all deliberately kept.

ii.
```python
MIN_VIDEO_COVERAGE = 0.9 # fraction of the window that must contain video frames
```

```python
# Spikes are only stored inside the NWB trial intervals and each unit carries
# `obs_intervals` listing the trials during which it was recorded. In 8 sessions
# the ephys covers only part of the behavioural session, so trials outside
# `obs_intervals` contain no neural data at all and must be dropped.
observed = np.ones(n_trials, dtype=bool)
for i in good:
    obs = np.asarray(units['obs_intervals'][int(i)])
    oi = np.searchsorted(start_time, obs[:, 0], side='right') - 1
    oi = oi[(oi >= 0) & (oi < n_trials)]
    m = np.zeros(n_trials, dtype=bool)
    m[oi] = True
    observed &= m
```

```python
vlo = np.searchsorted(vtime, win0)
vhi = np.searchsorted(vtime, win1)
coverage = (vhi - vlo) / ((T_END - T_START) / VIDEO_DT)
...
keep_mask = observed & (coverage >= MIN_VIDEO_COVERAGE) & np.isfinite(tone_onset)
```

```python
# trials without a single spike from any good unit: recording interrupted
nonempty = neural_all.any(axis=(1, 2))
n_dropped_empty = int((~nonempty).sum())
keep_trials = keep_trials[nonempty]
neural_all = neural_all[nonempty]
```

iii. CONVERSION_NOTES Step 5, Key Decision 4: "all trial types are **kept** (photostim, early lick, ignore, auto/free water) because photostimulation is a required decoder input and early-lick/ignore are required decoder outputs — a deliberate, documented deviation from `get_regular_trial_mask`." Step 6 records the two bug fixes that produced filters 1 and 3: "units carry `obs_intervals`; in 8 sessions the ephys covers only part of the behavioural session ... so the remaining trials were being emitted as all-zero neural data"; and "in one session the final trial listed in `obs_intervals` contains no spikes at all (recording interrupted)". Step 10 Check 1 documents the verification of the empty-trial case by hand in `SC066_20210416_140326_s9` ("49 of 680 trials have 0 spikes from all 317 good units although their duration (median 4.98 s) and outcomes are normal, i.e. a genuine acquisition dropout, not a conversion bug"). The video criterion is justified as "trials are dropped only when the video does not cover ≥ 90 % of the −2.5…+1.5 s window ..., i.e. when the tongue output cannot be computed."

## 2-a. What variables in the raw data is the final `neural` data derived from?

i. `units/spike_times` (session-clock seconds), restricted to units with `units/classification == 'good'`. The go-cue times from `BehavioralEvents/go_start_times` are the other input, used to place the trial windows. Spike times are read once per unit per session and sliced per trial with `searchsorted`.

ii.
```python
units = nwb.units
classification = np.asarray(units['classification'].data[:])
good = np.where(classification == 'good')[0]
n_good = len(good)
...
spikes = [np.asarray(units['spike_times'][int(i)]) for i in good]
```

```python
go_times_all = np.asarray(bev['go_start_times'].timestamps[:])
```

iii. CONVERSION_NOTES Step 2: "`nwb.units` ... `spike_times` (session clock) ... `classification` (`good` / `unlabelled` — the QC-classifier label from the Chen–Liu white paper)". Step 5 mapping table: "`units.spike_times` (session clock) for units with `units.classification == 'good'` → `neural[session][trial]`". Spike times are the only neural representation in the file.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz (spikes/s). For each good unit, `searchsorted` gives the slice of spikes falling in every trial's absolute window at once; the selected spikes are expressed relative to that trial's go cue, floor-divided into 50 ms bins, and accumulated with a single `np.bincount` over a flattened `(trial, bin)` index. Counts are then divided by the bin width. No smoothing, normalisation, or baseline subtraction is applied. Output dtype is `float32`.

ii.
```python
# For every unit, all trial windows are extracted at once with searchsorted and
# binned with a single bincount (equivalent to the reference sliding_histogram
# with bin_width == stride == BIN_SIZE and rate=True).
neural_all = np.zeros((nk, n_good, N_BINS), dtype=np.float32)
w0 = gk + T_START
w1 = gk + T_END
for ui, st in enumerate(spikes):
    if st.size == 0:
        continue
    lo = np.searchsorted(st, w0)
    hi = np.searchsorted(st, w1)
    cnt = hi - lo
    tot = int(cnt.sum())
    if tot == 0:
        continue
    trial_ids = np.repeat(np.arange(nk), cnt)
    offsets = np.repeat(lo - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt)
    pos = np.arange(tot) + offsets
    rel = st[pos] - gk[trial_ids]
    bidx = np.floor((rel - T_START) / BIN_SIZE).astype(np.int64)
    np.clip(bidx, 0, N_BINS - 1, out=bidx)
    flat = trial_ids * N_BINS + bidx
    counts = np.bincount(flat, minlength=nk * N_BINS).reshape(nk, N_BINS)
    neural_all[:, ui, :] = counts
neural_all /= BIN_SIZE      # spikes / s
```

iii. CONVERSION_NOTES Step 3/Step 5, Key Decision 6: "**Rates, not counts**: firing rate in spikes/s (count / 0.05 s), as in `sliding_histogram(..., rate=True)`." Step 10 Check 3(d): "Reference: `sliding_histogram(..., rate=True)`: counts/bin_width ... This conversion: `np.bincount` on `floor((t−T0)/0.05)`, rate = counts/0.05 s — same operation, bin size and window set by the task specification." The vectorised form was introduced for speed and verified bit-identical against the naive version (Step 12: "verified bit-identical (`np.allclose`, max diff 0.0)").

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are retained — the output of the region-specific logistic-regression QC classifiers described in the Chen/Liu white paper. No thresholds are applied to any of the 15 individual quality metrics, and no firing-rate threshold is applied. The older `unit_quality` (`good`/`multi`) column is not used. `units.is_good_trials` is explicitly examined and deliberately not used. Sessions with zero good units are dropped. 55,437 units survive in the 138 kept sessions (69,453 exist across all 174 files).

ii.
```python
units = nwb.units
classification = np.asarray(units['classification'].data[:])
good = np.where(classification == 'good')[0]
n_good = len(good)
...
if n_good == 0:
    info['skipped'] = 'no good units'
    return info
```

iii. CONVERSION_NOTES Step 5, Key Decisions 1–2: "**Neuron curation = `classification == 'good'`**: this is exactly the QC-classifier output described in the white paper and used for every analysis in both papers (69,453 units, 25.5 % of clusters ≈ the reported 69,943 / 25.9 %). No further QC-metric thresholds are applied because the classifier already integrates the 15 metrics." and "**No firing-rate threshold**: the method paper's 2 Hz cut is specific to its encoding (R²) analysis, not to dataset curation; dropping low-rate neurons would only remove information from a population decoder." Step 10 Check 5 on `is_good_trials`: "present in only 4 of 174 sessions ... the reference pipeline never uses this column, and the flagged trials have normal firing rates ... so they are **not** excluded; excluding them per-unit is impossible anyway without ragged neuron dimensions."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to the go-cue onset (`BehavioralEvents/go_start_times`). Spike times, event times and camera timestamps all live on the same session-absolute clock, so alignment is a per-trial subtraction: the absolute window `[go + T_START, go + T_END)` is located in each unit's sorted spike train with `searchsorted`, and the selected spike times have their trial's go-cue time subtracted before binning. No resampling or interpolation is performed.

ii.
```python
T_START = -2.5           # s relative to go cue (task specification)
T_END = 1.5              # s relative to go cue
```

```python
w0 = gk + T_START
w1 = gk + T_END
for ui, st in enumerate(spikes):
    ...
    lo = np.searchsorted(st, w0)
    hi = np.searchsorted(st, w1)
    ...
    rel = st[pos] - gk[trial_ids]
    bidx = np.floor((rel - T_START) / BIN_SIZE).astype(np.int64)
```

iii. CONVERSION_NOTES Step 3: "**Temporal alignment**: everything is aligned to the **go cue** (t = 0). The reference `.mat` export already stored spike times relative to the go cue; in the NWB files spike times are in session time, so we subtract `go_start_times`." Step 4 discrepancy table confirms this was checked: "reference `.mat` spike times were **already** go-cue-aligned / NWB spike times are in **session** time → Subtract `go_start_times` per trial. Same end result." Alignment correctness was further verified visually (`processing_*.png`, raster vs binned rates) and quantitatively by reproducing the published per-bin choice-decoding AUC time course (Step 12 Check 2).

## 2-e. How is the `neural` data temporally binned/resampled?

i. 80 non-overlapping 50 ms bins spanning −2.5 s to +1.5 s relative to the go cue. The grid (81 edges / 80 centres) is defined once at module level as offsets from the go cue and reused for every trial and every session, so every trial in the output has exactly 80 timepoints. No rebinning, smoothing, or sliding/overlapping windows are used, and no second temporal resolution exists anywhere in the pipeline — the tongue video and the photostim/tone inputs are placed on the identical grid.

ii.
```python
T_START = -2.5           # s relative to go cue (task specification)
T_END = 1.5              # s relative to go cue
BIN_SIZE = 0.05          # s (task specification: 50 ms bins)
N_BINS = int(round((T_END - T_START) / BIN_SIZE))          # 80
BIN_EDGES = T_START + BIN_SIZE * np.arange(N_BINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
```

```python
'time_bin_size': BIN_SIZE * 1000.0,   # ms
'temporal_alignment_event': 'auditory go cue onset (BehavioralEvents/go_start_times)',
'off_start': T_START,
'off_end': T_END,
```

iii. CONVERSION_NOTES Step 3/Step 4: "Reference used bw = 40 ms with 3.4 ms stride (method paper) or bw = 100 ms/stride 50 ms in the module `__main__`. **Our task specifies 50 ms bins**, so we use non-overlapping 50-ms bins (stride = width = 50 ms), which matches the spirit (rate in spikes/s) while satisfying the decoder spec." and "Trial window: reference used −3.0 → 3.0 / 3.5 s around the go cue; **our task specifies −2.5 → +1.5 s**." Both are explicitly listed in the Step 9 consistency table as deviations required by the task specification. Verification confirms 80 bins for every trial in every session.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents/sample_start_times` (the sample-epoch tone onsets), together with each trial's go-cue time. Because an early lick replays the sample epoch, a trial can carry several sample onsets; the AI takes the **last** sample onset occurring before the trial's go cue.

ii.
```python
sample_times = np.asarray(bev['sample_start_times'].timestamps[:])
```

```python
# tone onset = last sample-epoch onset before the go cue (epoch is replayed
# after an early lick, so the last one is the instructing tone)
tone_onset = np.full(n_trials, np.nan)
si = np.searchsorted(start_time, sample_times, side='right') - 1
for tr_i, s in zip(si, sample_times):
    if 0 <= tr_i < n_trials and s < go[tr_i]:
        if np.isnan(tone_onset[tr_i]) or s > tone_onset[tr_i]:
            tone_onset[tr_i] = s
```

iii. CONVERSION_NOTES Step 5, Key Decision 7: "**Tone onset** = last `sample_start_times` before the go cue (the tone actually instructing this trial; the epoch is replayed after early licks). Median go − tone = 1.85 s." Step 2 quantifies the replay: "mean 3.06 sample events on early-lick trials vs exactly 1 on other trials". Step 10 Check 5 records that "every trial in the dataset has at least one tone onset before its go cue (0 trials without)".

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone onset is expressed relative to the go cue (`tone_rel = tone_onset − go`, typically ≈ −1.85 s), and the input value for each bin is the bin centre (go-cue-relative) minus that offset — i.e. seconds elapsed since the instructing tone at the centre of the bin. It is a continuous, time-varying `float32` input, stored as row 0 of a `(2, 80)` array per trial. Values range over [−1.5, 11.9] s in the full dataset (large values arise on replay trials).

ii.
```python
tone_rel = tone_onset[keep_trials] - gk
input_all = np.zeros((nk, 2, N_BINS), dtype=np.float32)
input_all[:, 0, :] = (BIN_CENTERS[None, :] - tone_rel[:, None]).astype(np.float32)
```

```python
'input_descriptions': {
    'time_from_tone_onset': 'seconds from the onset of the instructing sample tone (last sample '
                            'epoch onset before the go cue) to the centre of the time bin',
    ...
```

iii. CONVERSION_NOTES Step 5 mapping table: "`BehavioralEvents.sample_start_times` (last one before the go cue) → `input[0]` = `time_from_tone_onset_s`; bin-centre time − tone-onset time (both relative to go cue); continuous, time-varying". No further processing is needed once the instructing tone is identified. Step 7 notes the plot check: "the `time_from_tone_onset` input crosses zero exactly at the marked tone onset".

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It uses exactly the same go-cue-aligned grid as the firing rates: the values are computed from `BIN_CENTERS`, the same 80 bin centres (−2.475 … +1.475 s re the go cue) used to bin the spikes. So bin *k* of the input covers precisely the same interval as bin *k* of the neural data; no separate alignment or interpolation step exists.

ii.
```python
BIN_EDGES = T_START + BIN_SIZE * np.arange(N_BINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
```

```python
input_all[:, 0, :] = (BIN_CENTERS[None, :] - tone_rel[:, None]).astype(np.float32)
```

iii. All NWB streams share one session clock (CONVERSION_NOTES Step 2), so both the spikes and the tone event are placed on the same go-cue-relative axis by subtraction. The `--show-processing` plots overlay the input trace with a vertical marker at the tone onset to demonstrate the alignment (Step 7: "the `time_from_tone_onset` input crosses zero exactly at the marked tone onset").

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`, which are absolute session-clock timestamps. Each event pair is assigned to the trial whose start-time interval contains the onset. The AI deliberately used these event streams rather than the trials-table columns (`photostim_onset` / `photostim_duration`, which are strings relative to trial start), after verifying that the two agree exactly.

ii.
```python
photostim_on = np.asarray(bev['photostim_start_times'].timestamps[:])
photostim_off = np.asarray(bev['photostim_stop_times'].timestamps[:])
```

```python
# photostim windows per trial (relative to the go cue)
stim_on = np.full(n_trials, np.nan)
stim_off = np.full(n_trials, np.nan)
pi = np.searchsorted(start_time, photostim_on, side='right') - 1
for tr_i, on, off in zip(pi, photostim_on, photostim_off):
    if 0 <= tr_i < n_trials:
        stim_on[tr_i] = on - go[tr_i]
        stim_off[tr_i] = off - go[tr_i]
```

iii. CONVERSION_NOTES Step 4 discrepancy table: "`trials.photostim_onset/power/duration` (strings, relative to trial start) **and** `BehavioralEvents.photostim_start/stop_times` in session time; the two agree exactly (`np.allclose`) → Use the event timestamps (already absolute) → go-cue-relative on/off. Data: 18,588 stim trials (19.6 %) in 168 sessions." Step 2 records the same check with atol 1e-3.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. It is a binary, time-varying input: a bin is 1.0 if its centre lies within `[stim_on, stim_off]` relative to the go cue, else 0.0. Trials with no photostimulation keep NaN bounds and are simply left at zero (the code guards with `has_stim = np.isfinite(son)`). It is stored as row 1 of the same `(2, 80)` float32 input array. Photostim is on in 3.3 % of bins overall and on ~20 % of trials.

ii.
```python
son = stim_on[keep_trials]
soff = stim_off[keep_trials]
has_stim = np.isfinite(son)
if has_stim.any():
    input_all[has_stim, 1, :] = (
        (BIN_CENTERS[None, :] >= son[has_stim][:, None])
        & (BIN_CENTERS[None, :] <= soff[has_stim][:, None])).astype(np.float32)
```

```python
'photostim_on': '1 if ALM photoinhibition was on during the time bin, else 0'
```

iii. The instructions require "Whether photostimulation is on at every time point (discrete, time-varying)" and "If an input is a time such as onset of some stimulus, represent it as a binary time series", so the AI represented it as a per-bin indicator rather than a per-trial flag. CONVERSION_NOTES Step 10 Check 4 validates the result against the paper: "of 18,588 stimulation events, none ends more than **0.5 ms** after the go cue (max overshoot 0.0005 s, i.e. timestamp rounding), matching 'photoinhibition always ended before the Go cue'", and Step 9 reports "on only in bins before t = 0".

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostim onset/offset are converted to go-cue-relative seconds (`on − go[tr]`, `off − go[tr]`) and then compared against `BIN_CENTERS`, the identical grid used for the firing rates. Alignment is therefore exact by construction, with no interpolation.

ii.
```python
stim_on[tr_i] = on - go[tr_i]
stim_off[tr_i] = off - go[tr_i]
```

```python
input_all[has_stim, 1, :] = (
    (BIN_CENTERS[None, :] >= son[has_stim][:, None])
    & (BIN_CENTERS[None, :] <= soff[has_stim][:, None])).astype(np.float32)
```

iii. Same rationale as 3-c: all streams share the session clock, so the go-cue subtraction puts the stimulation window on the same axis as the neural bins. Step 7 records the visual confirmation: "the `photostim_on` input is 1 exactly over the shaded photostim interval" in the per-trial processing plots.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no explicit choice column in the NWB file. The AI derives it from two trials-table columns, `trials['outcome']` (`hit`/`miss`/`ignore`) and `trials['trial_instruction']` (`left`/`right`): a hit means the animal licked the instructed side, a miss means it licked the opposite side, and an `ignore` means it never licked. `BehavioralEvents/left_lick_times` and `right_lick_times` were used only as an independent cross-check, not in the conversion.

ii.
```python
outcome = np.asarray(trials['outcome'].data[:])
instruction = np.asarray(trials['trial_instruction'].data[:])
```

```python
def choice_from_trial(outcome, instruction):
    """Lick direction the animal actually chose: 0 left, 1 right, 2 no lick."""
    if outcome == 'ignore':
        return 2
    right = (instruction == 'right')
    if outcome == 'miss':          # error trial -> licked the other spout
        right = not right
    return 1 if right else 0
```

iii. CONVERSION_NOTES Step 5 mapping table: "`trials.outcome` + `trials.trial_instruction` → `output[0]` = `lick_direction_choice` ∈ {0 left, 1 right, 2 no lick}; hit → instruction; miss → opposite of instruction; ignore → no lick", mapped onto the reference code's `behavior_report` (1/0/−1) + `task_trial_type` ('l'/'r'). Step 10 Check 4 validates: "the choice derived from `outcome` + `trial_instruction` agrees with the *first lick after the go cue* on **80,694/80,895 = 99.75 %** of responded trials across all 174 sessions ... the `outcome`-based definition is the one the papers use (`behavior_report`)", and "**Ignore trials**: 97.7 % have no lick at all in the response window".

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The derived code (0 = left, 1 = right, 2 = no lick) is a single value per trial, broadcast across all 80 bins of row 0 of an `(4, 80)` `int64` output array so that all four outputs share one time-varying array. `output_values[0] = ['left', 'right', 'no lick']` names the three classes. Full-dataset distribution: left 44.7 %, right 44.2 %, no lick 11.0 %.

ii.
```python
OUTPUT_VALUES = [['left', 'right', 'no lick'],
                 ...
```

```python
output_trials = []
for k, tr in enumerate(keep_trials):
    out = np.zeros((4, N_BINS), dtype=np.int64)
    out[0] = choice_from_trial(outcome[tr], instruction[tr])
    ...
```

iii. The class codes follow the instructions' ordering ("Lick direction choice (left, right, no lick, per-trial)"). CONVERSION_NOTES Step 5: "All four outputs are emitted as a **time-varying (4, 80)** array: the three per-trial variables are constant across the 80 bins (the target format requires a single array per trial, and the tongue class is genuinely time-varying)."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the NWB trials table, which already stores exactly the three required categories as strings `'hit'`, `'miss'`, `'ignore'`. No derivation is needed.

ii.
```python
outcome = np.asarray(trials['outcome'].data[:])
```

iii. CONVERSION_NOTES Step 2 documents the trials-table columns including "`outcome` (`hit`/`miss`/`ignore`)", and Step 4 confirms the 1-to-1 mapping to the reference code's variable: "`outcome=='hit'`↔`report==1`, `'miss'`↔0, `'ignore'`↔−1".

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped through a fixed dictionary to 0 = ignore, 1 = miss, 2 = hit, and written to row 1 of the output array, constant across all 80 bins. Full-dataset distribution: ignore 11.0 %, miss 15.3 %, hit 73.7 %.

ii.
```python
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}
```

```python
out[1] = OUTCOME_CODE[outcome[tr]]
```

```python
OUTPUT_VALUES = [...,
                 ['ignore', 'miss', 'hit'],
                 ...]
```

iii. CONVERSION_NOTES Step 5 mapping table: "`trials.outcome` → `output[1]` = `outcome` ∈ {0 ignore, 1 miss, 2 hit}; direct map; **ordering given in the task description**." As with choice, the per-trial value is repeated across bins to keep all outputs in a single `(n_output, n_timepoints)` array.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the `early_lick` column of the NWB trials table, which holds the strings `'early'` and `'no early'`.

ii.
```python
early_lick = np.asarray(trials['early_lick'].data[:])
```

iii. CONVERSION_NOTES Step 2 lists "`early_lick` (`early`/`no early`)" among the trials-table columns, and Step 4 maps it onto the reference code's `behavior_early_report`. The flag is per-trial in the raw data; the lick that sets it occurs during the sample/delay epoch, i.e. before the go cue.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'early'` → 1, anything else → 0, written to row 2 of the output array and held constant across all 80 bins. `output_values[2] = ['no', 'yes']`. Full-dataset distribution: no 88.4 %, yes 11.6 % (dataset-wide raw value 11.4 %).

ii.
```python
out[2] = 1 if early_lick[tr] == 'early' else 0
```

```python
OUTPUT_VALUES = [..., ['no', 'yes'], ...]
```

iii. The codes follow the instructions ("Early lick (no, yes, per-trial)"). CONVERSION_NOTES Step 12 Check 1 adds a behavioural justification of why the decoder cannot do better than ~0.75 on this output and verifies the label is meaningful: "the tongue is visible in **25.2 %** of pre-go bins on early-lick trials versus **4.6 %** on normal trials (5.5×), and the pre-go population rate is higher (8.5 vs 7.4 spikes/s)."

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, a DeepLabCut side-view marker series sampled at ~300 Hz with session-clock `timestamps` and `data` of shape `(n_frames, 3)` = (tongue_x, tongue_y, likelihood). Column 1 (`y`) is the value; column 2 (`likelihood`) determines whether the tongue is visible in that frame.

ii.
```python
bts = nwb.acquisition['BehavioralTimeSeries'].time_series
tongue = bts['Camera0_side_TongueTracking']
vtime = np.asarray(tongue.timestamps[:])
vdata = np.asarray(tongue.data[:])            # (n_frames, 3): x, y, likelihood
tongue_y = vdata[:, 1]
tongue_vis = vdata[:, 2] > LIKELIHOOD_THRESH
```

iii. CONVERSION_NOTES Step 2: "`nwb.acquisition['BehavioralTimeSeries']` | DeepLabCut side-view tracking at 300 Hz (dt = 0.0034 s): `Camera0_side_TongueTracking`, ... each (n_frames, 3) = (x, y, likelihood), with **session-clock timestamps**" and "Sessions with side-view tongue tracking | 174 (all)". This is the only tongue measurement in the file, and it is the same side-view DLC marker the reference `Sherlock/align_markers.py` aligns.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with DLC `likelihood <= 0.9` are treated as "tongue not visible" and excluded (the tracker still emits a position when the tongue is retracted). The y-values of the surviving visible frames within a trial's window are averaged into the same 50 ms bins as the neural data, giving one mean y per (trial, bin); bins containing no visible frame are left as NaN and become class 3.

ii.
```python
LIKELIHOOD_THRESH = 0.9  # DLC likelihood above which the tongue counts as visible
```

```python
tongue_bin_y = np.full((nk, N_BINS), np.nan)
for k in range(nk):
    g = gk[k]
    lo = np.searchsorted(vtime, g + T_START)
    hi = np.searchsorted(vtime, g + T_END)
    if hi <= lo:
        continue
    sel = tongue_vis[lo:hi]
    if not sel.any():
        continue
    vt = vtime[lo:hi][sel] - g
    vy = tongue_y[lo:hi][sel]
    bidx = np.floor((vt - T_START) / BIN_SIZE).astype(np.int64)
    np.clip(bidx, 0, N_BINS - 1, out=bidx)
    sums = np.bincount(bidx, weights=vy, minlength=N_BINS)
    cnts = np.bincount(bidx, minlength=N_BINS)
    nz = cnts > 0
    tongue_bin_y[k, nz] = sums[nz] / cnts[nz]
```

iii. CONVERSION_NOTES Step 5, Key Decision 8: "**Tongue visibility threshold**: DLC `likelihood > 0.9`. The likelihood is strongly bimodal (≈ 1 vs ≈ 1e-4), so the exact threshold is immaterial; 11.7 % of frames are 'visible' session-wide." Step 3/4 explain the deviation from the papers: "Outliers ... imputed; when the tongue is occluded (inside the mouth) they set the tongue position to its mean value. For our decoder the occluded state is its own output class (`not visible`), so we detect occlusion from the DLC `likelihood` rather than imputing."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, the 40th and 60th percentiles are computed over **all visible binned mean y-values of that session's kept trials** (requiring at least 10 visible bins). Each bin is then classified: `< p40` → 0, `p40 ≤ y ≤ p60` → 1, `> p60` → 2, and bins with no visible frame → 3 ("not visible"). If a session somehow has fewer than 10 visible bins, every bin becomes class 3. By construction the visible bins split 40/20/40; the full-dataset distribution is 10.6 / 5.3 / 10.6 / 73.6 %.

ii.
```python
visible = np.isfinite(tongue_bin_y)
if visible.sum() >= 10:
    p40, p60 = np.percentile(tongue_bin_y[visible], [40, 60])
else:
    p40 = p60 = np.nan
tongue_class = np.full(tongue_bin_y.shape, 3, dtype=np.int64)   # 3 = not visible
if np.isfinite(p40):
    tongue_class[visible & (tongue_bin_y < p40)] = 0
    tongue_class[visible & (tongue_bin_y >= p40) & (tongue_bin_y <= p60)] = 1
    tongue_class[visible & (tongue_bin_y > p60)] = 2
```

```python
OUTPUT_VALUES = [..., ['<40th pct', '40-60th pct', '>60th pct', 'not visible']]
```

iii. CONVERSION_NOTES Step 5, Key Decision 9: "**Percentiles for the tongue classes** are computed **per session** over the visible binned y-values inside the extracted windows (the data actually used), as required by the task ('per-session discretization')." Step 7 records the sanity check: "The tongue classes 0/1/2 are in a 40/20/40 ratio **of the visible bins** (0.103/0.051/0.103 → 40.2 %/19.9 %/40.2 %), exactly as the percentile definition requires", and the plots show "the 40th/60th percentile lines fall where the class boundaries change".

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. This is the only genuinely time-varying output. The camera timestamps are on the same session-absolute clock as the spikes, so each trial's frame range is located with `searchsorted(vtime, go ± window)` and each frame is assigned to a bin by its offset from the go cue using the identical `floor((t − T_START)/BIN_SIZE)` rule used for the spikes. Trials whose video covers < 90 % of the window are dropped rather than emitted with mostly-class-3 bins.

ii.
```python
lo = np.searchsorted(vtime, g + T_START)
hi = np.searchsorted(vtime, g + T_END)
...
vt = vtime[lo:hi][sel] - g
bidx = np.floor((vt - T_START) / BIN_SIZE).astype(np.int64)
np.clip(bidx, 0, N_BINS - 1, out=bidx)
```

```python
vlo = np.searchsorted(vtime, win0)
vhi = np.searchsorted(vtime, win1)
coverage = (vhi - vlo) / ((T_END - T_START) / VIDEO_DT)
```

iii. CONVERSION_NOTES Step 4: "reference `align_markers.py`: dt = 0.0034, frame time = index*dt − go_time ... tracking timestamps already in session clock, 300 Hz, one block per trial → Equivalent; I use the stored timestamps − go time (more robust, handles dropped frames)." Step 2 notes the video is trial-gated ("one continuous 300 Hz block per trial (n_gaps = n_trials − 1), starting at each trial start"), which motivates the coverage criterion. Step 7 confirms alignment visually: "the raw 300 Hz tongue trace and the binned mean overlay perfectly" and "tongue class 3 (not visible) dominates before the go cue (96–100 % of trials) and drops to 7–12 % immediately after it".

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Six cases, handled by either exclusion (where nothing was recorded) or an explicit category (where the measurement legitimately has no value):
  - **Session never quality-controlled** (`classification` empty for all units) → no `'good'` units → session dropped.
  - **Trial outside `obs_intervals`** (ephys not running) → trial dropped.
  - **Trial with zero spikes from every good unit** (acquisition dropout; also covers free-water trials) → trial dropped.
  - **Trial with missing/short video** (< 90 % window coverage) → trial dropped; **session with video stopping at the go cue** → session dropped.
  - **Frames with the tongue retracted/occluded** (low DLC likelihood) → excluded from the bin mean; a bin with no visible frame becomes class 3 "not visible".
  - **Trial with no go cue or no preceding tone onset** → session/trial dropped (guards that never fire in this dataset).
  Post-go bins on short (error) trials, where the trial interval ends before +1.5 s, are kept as 0 spikes/s and the fraction of such bins is reported in the metadata.

ii.
```python
if np.any(np.isnan(go)):
    return dict(sess=sess_name, skipped='missing go cue')
...
if n_good == 0:
    info['skipped'] = 'no good units'
    return info
```

```python
keep_mask = observed & (coverage >= MIN_VIDEO_COVERAGE) & np.isfinite(tone_onset)
...
nonempty = neural_all.any(axis=(1, 2))
keep_trials = keep_trials[nonempty]
```

```python
tongue_vis = vdata[:, 2] > LIKELIHOOD_THRESH
...
tongue_class = np.full(tongue_bin_y.shape, 3, dtype=np.int64)   # 3 = not visible
```

```python
# how much of the window lies after the end of the recorded trial interval
n_unobserved = int(np.sum(BIN_CENTERS[None, :] > (stop_time[keep_trials] - gk)[:, None]))
...
'frac_unobserved_neural_bins': frac_unobs,
'note_unobserved': ('spikes are stored only within the NWB trial intervals; on error (miss) '
                    'trials the interval ends ~0.8 s after the go cue, so later bins contain '
                    'no spikes'),
```

iii. CONVERSION_NOTES Step 6 documents the two bugs found during development (unobserved trials and one empty trial) and Step 10 Check 1 documents the hand verification of the empty-trial case as a genuine acquisition dropout, not a conversion bug. Step 5, Key Decision 5: "Unobserved neural periods: spikes are stored only inside trial intervals ... On error (`miss`) trials the trial ends ~0.8 s after the go cue, so the last part of the window has no recorded spikes and is binned as 0 spikes/s. This matches the reference pipeline ... The fraction of unobserved bins is recorded in `metadata`." Step 10 Check 5 enumerates the remaining edge cases (windows overlapping the previous trial, replayed sample epochs, `is_good_trials`, sessions with < 2 trials).

## 10-a. What are the most time-consuming steps of the code?

i. The code instruments every stage and prints per-session timings (`open`, `trials`, `units_meta`, `video_read`, `spikes_read`, `obs_intervals`, `bin_trials`). The dominant costs are NWB I/O: opening the file (~0.16 s), reading the ragged `spike_times` per good unit (0.08–0.5 s, scaling with unit count), reading the `obs_intervals` of every good unit, and reading the ~300 Hz video array. The actual binning/input/output construction is 0.04–0.10 s per session. End-to-end the conversion is 35 s with 16 workers (~1.2 s/session serially), plus ~15 s to pickle the 9.6 GB result.

ii.
```python
t_open = time.time()
timings = {}
with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
    ...
    timings['open'] = time.time() - t_open
    ...
    timings['trials'] = time.time() - t0
    ...
    timings['units_meta'] = time.time() - t0
    ...
    timings['video_read'] = time.time() - t0
    ...
    timings['spikes_read'] = time.time() - t0
    ...
    timings['obs_intervals'] = time.time() - t0
    ...
    timings['bin_trials'] = time.time() - t0
```

```python
print('    timings: ' + ', '.join(f'{k}={v:.2f}s' for k, v in res.get('timings', {}).items()), flush=True)
```

iii. CONVERSION_NOTES Step 7 tabulates the per-step timings and the extrapolated totals ("open NWB 0.16 s/session → 28 s; read spike times 0.08–0.14 s (up to ~0.5 s for 900-unit sessions) → ~60 s; read video 0.03 s → 6 s; obs_intervals 0.01–0.05 s → 9 s; binning + inputs + outputs 0.04–0.10 s → ~30 s; total serial ~4 min; with 16 workers well under 5 min"). The instructions required an estimate under 15 minutes; the AI met it comfortably.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python loops remain:
  - **per-unit spike binning** (`for ui, st in enumerate(spikes)`) — this one is already vectorised *across all trials at once*; it cannot be collapsed further because `spike_times` is ragged (a different number of spikes per unit), so there is no single sorted array to search.
  - **per-unit `obs_intervals` read** (`for i in good`) — reads and intersects one interval table per good unit; this is both a loop *and* redundant I/O, since the intervals are identical across units in this dataset.
  - **per-trial tongue binning** (`for k in range(nk)`) — could be collapsed into one global bin index plus a single `bincount`, as the reference-style vectorised neural path does.
  - **per-event loops for tone onset and photostim** (`for tr_i, s in zip(si, sample_times)` and `for tr_i, on, off in zip(pi, photostim_on, photostim_off)`) — these iterate over raw event timestamps in Python and could be replaced by `np.maximum.at` / fancy indexing.
  - **per-trial output assembly** (`for k, tr in enumerate(keep_trials)`) — builds one `(4, 80)` array per trial in Python, which could be a single broadcast assignment.

ii.
```python
for i in good:
    obs = np.asarray(units['obs_intervals'][int(i)])
    oi = np.searchsorted(start_time, obs[:, 0], side='right') - 1
    ...
    observed &= m
```

```python
for tr_i, s in zip(si, sample_times):
    if 0 <= tr_i < n_trials and s < go[tr_i]:
        if np.isnan(tone_onset[tr_i]) or s > tone_onset[tr_i]:
            tone_onset[tr_i] = s
```

```python
for k in range(nk):
    ...
    sums = np.bincount(bidx, weights=vy, minlength=N_BINS)
    cnts = np.bincount(bidx, minlength=N_BINS)
```

```python
for k, tr in enumerate(keep_trials):
    out = np.zeros((4, N_BINS), dtype=np.int64)
    out[0] = choice_from_trial(outcome[tr], instruction[tr])
```

iii. CONVERSION_NOTES Step 6/7 records that the original version "looped over bins inside Python and re-read spike times per trial", and that the speed-ups were "`np.bincount` binning instead of per-bin comparisons (~5×)", "single read of spike times / video per session + `searchsorted` slicing (~2×)", "fully vectorised per-unit binning over all trials at once (0.5–0.9 s → 0.04–0.10 s per session, ~10×)", and "16-way multiprocessing over sessions (~10× wall-clock)". The remaining loops run over units/trials/events rather than spikes and are not a measurable share of the 35 s total, so the AI left them.

## 10-c. What processing does the code repeat multiple times?

i. Each NWB file is opened once and the spike times and video array are each read once per session, so the heavy reads are not repeated. Two things are repeated:
  - **`obs_intervals` is read and intersected once per good unit** (up to 923 times per session) even though the intervals are identical across units of a session — this is the only materially redundant I/O in the code.
  - **Neural binning is performed for all `keep_mask` trials and then a subset is discarded** by the `nonempty` filter, so the binning work for those ~2,150 trials is thrown away. (Detecting empty trials requires binning them first, so this is inherent to the chosen order.)
  Also, `coverage` is computed over all trials and then re-tested twice (once for the session criterion, once for the trial mask), which is cheap.

ii.
```python
observed = np.ones(n_trials, dtype=bool)
for i in good:
    obs = np.asarray(units['obs_intervals'][int(i)])
    oi = np.searchsorted(start_time, obs[:, 0], side='right') - 1
    oi = oi[(oi >= 0) & (oi < n_trials)]
    m = np.zeros(n_trials, dtype=bool)
    m[oi] = True
    observed &= m
```

```python
nonempty = neural_all.any(axis=(1, 2))
keep_trials = keep_trials[nonempty]
neural_all = neural_all[nonempty]
```

```python
if np.mean(coverage[observed] >= MIN_VIDEO_COVERAGE) < MIN_SESSION_VIDEO_FRAC:
    ...
keep_mask = observed & (coverage >= MIN_VIDEO_COVERAGE) & np.isfinite(tone_onset)
```

iii. CONVERSION_NOTES Step 6: "Spike times are read once per unit (`units['spike_times'][i]`) and sliced per trial with `np.searchsorted` ... The video array is read once per session and sliced per trial." The repeated `obs_intervals` read is not flagged in the notes; it is a deliberate robustness choice (intersecting per-unit masks rather than trusting that all units share one interval table), but it costs 0.01–0.05 s per session of avoidable I/O.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items of work produce nothing that reaches the decoder:
  - **Dead code**: `bin_spike_times()` is defined (a full non-vectorised binning implementation) but never called anywhere — the inline vectorised path replaced it.
  - **Work on sessions that are later dropped**: the trials table, behavioural events, and the session performance statistics are computed for all 174 files, including the 36 that are then skipped; for the 6 video-rejected sessions the entire video array is read first.
  - **Neural binning for trials that are then dropped** by the `nonempty` filter (~2,150 trials).
  - **Diagnostics that only land in metadata**: `stop_time`, `n_unobserved` / `frac_unobserved_bins`, `n_auto_water`, `n_free_water`, `n_photostim`, `coverage`, `tongue_p40`/`p60`, per-stage `timings`, and the per-session `performance` / `n_correct_left` / `n_correct_right`. These are useful documentation, not decoder inputs.
  - `nwb.electrodes['x']` and the electrode-group JSON targets are read to build hemisphere-prefixed region labels, which populate `brain_region_idx` but are not used by `train_decoder.py`.

ii.
```python
def bin_spike_times(spike_times_rel, n_neurons):
    """Bin go-cue-aligned spike times into firing rates.
    ...
    """
    out = np.zeros((n_neurons, N_BINS), dtype=np.float32)
    for i, st in enumerate(spike_times_rel):
        ...
    out /= BIN_SIZE
    return out
# (never called)
```

```python
n_unobserved = int(np.sum(BIN_CENTERS[None, :] > (stop_time[keep_trials] - gk)[:, None]))
...
n_auto_water=int(auto_water[keep_trials].sum()),
n_free_water=int(free_water[keep_trials].sum()),
timings=timings,
```

```python
target_of_group = {}
for gname, grp in nwb.electrode_groups.items():
    try:
        target_of_group[gname] = json.loads(grp.location)['brain_regions']
    except Exception:
        target_of_group[gname] = ''
```

iii. The notes do not identify any of this as waste; the diagnostics are deliberate and are cited throughout CONVERSION_NOTES Steps 9–12 as evidence for consistency with the papers (e.g. "Mean fraction of unobserved neural bins: 0.0254", the per-session performance table, "Good units per coarse region ... reproduce the per-area counts published in methods.txt"). The `brain_region_idx` fields are required by the target data format even though the reference decoder does not consume them. The `bin_spike_times` dead function is an unflagged leftover from the pre-vectorisation version described in Step 6 ("the initial version looped over bins inside Python").
