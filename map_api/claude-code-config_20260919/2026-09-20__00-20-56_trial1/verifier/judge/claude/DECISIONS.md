# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs all NWB files under `data/sub-*/` and processes each file with `pynwb.NWBHDF5IO`. Each file is one session. Trials, units, and behavioral events are read from within each NWB file. Multiprocessing (spawn pool, default 12 workers) is used to parallelize across sessions.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
# ...
with NWBHDF5IO(path, 'r', load_namespaces=True) as io:
    nwb = io.read()
    units = nwb.units
    trials = nwb.trials.to_dataframe()
    be = nwb.acquisition['BehavioralEvents']
    go_all = np.asarray(be['go_start_times'].timestamps[:], dtype=np.float64)
```

iii. The AI noted that NWB is the published format and one file per session means a glob is sufficient. The AI confirmed 174 files and 28 subjects match `dandiset.yaml`.

## 1-b. How are the data split into subjects?

i. The AI uses `nwb.subject.description` (falling back to `nwb.subject.subject_id`) as the subject identifier. This gives lab mouse IDs like `SC015` rather than numeric IDs like `440956`.

ii.
```python
subject = nwb.subject.description or nwb.subject.subject_id
# ...
subjects = sorted({s['subject'] for s in sessions})
subj_idx = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([subj_idx[s['subject']] for s in sessions], dtype=np.int64),
```

iii. The AI's CONVERSION_NOTES.md documents using the lab mouse ID (e.g. `SC015`), noting `nwb.subject.description` provides this. This gives 28 unique subjects.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. No grouping or splitting is needed. Session order follows the sorted file list. Sessions are identified by `nwb.identifier`.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
session_id = nwb.identifier
```

iii. The AI noted that the dandiset stores one session per file, making file boundaries session boundaries. 173 of 174 files are retained (one dropped for having no good units).

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.trials.to_dataframe()`), one row per behavioral trial. The row count is asserted to match the number of go-cue events.

ii.
```python
trials = nwb.trials.to_dataframe()
n_trials_all = len(trials)
go_all = np.asarray(be['go_start_times'].timestamps[:], dtype=np.float64)
assert len(go_all) == n_trials_all, \
    f"{session_id}: {len(go_all)} go cues for {n_trials_all} trials"
```

iii. The AI verified that `go_start_times` has exactly one event per trial in all 174 sessions.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials in three stages:
1. **Ephys coverage**: Only trials within `units.obs_intervals` (matched by `searchsorted` + `np.isclose` on start/stop times).
2. **Auto-water and free-water exclusion**: `auto_water == 0 & free_water == 0`.
3. **Zero-spike filter**: Trials where the entire population fires zero spikes are dropped (2 trials total).
A session is dropped if fewer than 2 trials survive.

ii.
```python
ephys_covered = np.zeros(n_trials_all, dtype=bool)
j = np.clip(np.searchsorted(start_all, obs[:, 0] + 1e-6) - 1, 0, n_trials_all - 1)
matched = np.isclose(start_all[j], obs[:, 0]) & np.isclose(stop_all[j], obs[:, 1])
ephys_covered[j[matched]] = True

keep = ephys_covered.copy()
keep &= (auto_all == 0) & (free_all == 0)
# ...
has_spikes = rates.sum(axis=(0, 2)) > 0
if n_empty:
    rates = rates[:, has_spikes, :]
```

iii. The AI justified keeping photostim, early-lick, and ignore trials (since they are decoder inputs/outputs). The auto-water and free-water exclusion follows the reference paper's statement that "free water trials were excluded from all analyses." The zero-spike filter was added after discovering that some trials have no recorded spikes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units.spike_times` (sorted spike times per unit in session-absolute seconds). Only units with `classification == 'good'` contribute. Go-cue times from `BehavioralEvents/go_start_times` are used for alignment.

ii.
```python
classification = np.asarray(units['classification'][:])
good = np.where(classification == 'good')[0]
# ...
spike_lists = []
for i in good:
    st = np.asarray(units['spike_times'][int(i)], dtype=np.float64)
    spike_lists.append(st)
rates = bin_spike_rates(spike_lists, go)
```

iii. The AI noted that `spike_times` is the only neural representation in the file.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50 ms bins spanning [-2.5, +1.5] s relative to the go cue. Counts are divided by bin width (0.05 s) to get firing rates in Hz. No smoothing, normalization, or baseline subtraction is applied. Per-unit spike times are read individually (with a sort check), then `np.searchsorted` is used with flattened trial edges.

ii.
```python
def bin_spike_rates(spike_times_list, go_times):
    edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()
    out = np.empty((len(spike_times_list), n_trials, N_BINS), dtype=np.float32)
    for i, st in enumerate(spike_times_list):
        idx = np.searchsorted(st, edges).reshape(n_trials, N_BINS + 1)
        out[i] = np.diff(idx, axis=1)
    out /= np.float32(BIN_SIZE)
    return out
```

iii. The AI noted this matches the reference code's `sliding_histogram(..., rate=True)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units.classification == 'good'` are kept. No additional thresholds (e.g., firing rate) are applied. A session with no good units is dropped. This retains 69,453 of 272,227 units (25.5%).

ii.
```python
classification = np.asarray(units['classification'][:])
good = np.where(classification == 'good')[0]
if len(good) == 0:
    return {'session_id': session_id, 'skipped': 'no good units'}
```

iii. The AI justified this as the classifier-based QC described in the papers/white paper. The 2 Hz firing rate threshold from the method paper was deliberately not applied as it was analysis-specific.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times and go-cue times are on the same session-absolute clock. Bin edges are computed as go-cue time + relative offsets. No resampling or interpolation is needed.

ii.
```python
go_all = np.asarray(be['go_start_times'].timestamps[:], dtype=np.float64)
edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()
```

iii. Everything in the NWB file shares one global clock, so alignment is simply looking up each trial's go-cue time and placing the bin window around it.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins spanning [-2.5 s, +1.5 s] relative to the go cue. The bin grid is defined once as 81 edges and reused for every trial and session.

ii.
```python
T_PRE = 2.5
T_POST = 1.5
BIN_SIZE = 0.05
N_BINS = int(round((T_PRE + T_POST) / BIN_SIZE))  # 80
BIN_EDGES_REL = -T_PRE + np.arange(N_BINS + 1) * BIN_SIZE
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2.0
```

iii. The window and 50 ms bin width are set by the task instructions. No rebinning from a finer resolution is needed since spike times are continuous.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `BehavioralEvents/sample_start_times` (tone onsets) and the go-cue times. The last sample onset at or before the go cue is used as the tone onset for each trial.

ii.
```python
sample_t = np.asarray(be['sample_start_times'].timestamps[:], dtype=np.float64)
sample_t = np.sort(sample_t)
si = np.searchsorted(sample_t, go, side='right') - 1
tone_onset = np.where(si >= 0, sample_t[np.clip(si, 0, len(sample_t) - 1)], np.nan)
```

iii. An early lick replays the sample epoch, so a trial can have multiple tone onsets. The last one before the go cue is the instructive one. A fallback to the session median go-to-tone interval is included if no valid tone onset is found.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the time from tone onset at each bin center is: `bin_center_relative_to_go + (go_time - tone_onset_time)`.

ii.
```python
go_minus_tone = go - tone_onset
time_from_tone = (BIN_CENTERS_REL[None, :] + go_minus_tone[:, None]).astype(np.float32)
```

iii. No additional processing beyond computing the time difference.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same bin-center grid defined relative to the go cue. Since the neural data bins and the time-from-tone values share the same time axis, they are inherently aligned.

ii.
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2.0
time_from_tone = (BIN_CENTERS_REL[None, :] + go_minus_tone[:, None]).astype(np.float32)
```

iii. N/A

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times` (session-level event timestamps with laser power as data values).

ii.
```python
ps_start = np.asarray(be['photostim_start_times'].timestamps[:], dtype=np.float64)
ps_stop = np.asarray(be['photostim_stop_times'].timestamps[:], dtype=np.float64)
order = np.argsort(ps_start)
ps_start, ps_stop = ps_start[order], ps_stop[order]
```

iii. The AI used the session-level photostim event timestamps rather than the per-trial `trials.photostim_onset`/`photostim_duration` columns.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A bin is 1.0 if its absolute center time falls within any `[photostim_start, photostim_stop)` interval, else 0.0. Uses `searchsorted` to efficiently match bin centers to the nearest preceding start event.

ii.
```python
photostim = np.zeros((n_trials, N_BINS), dtype=np.float32)
if len(ps_start):
    abs_centers = go[:, None] + BIN_CENTERS_REL[None, :]
    k = np.searchsorted(ps_start, abs_centers, side='right') - 1
    valid = k >= 0
    kk = np.clip(k, 0, len(ps_start) - 1)
    on = valid & (abs_centers >= ps_start[kk]) & (abs_centers < ps_stop[kk])
    photostim[on] = 1.0
```

iii. The AI's approach tests each bin center against the sorted list of photostim events, which handles the case of multiple events or events spanning trial boundaries.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Bin centers are expressed as absolute session times (`go + BIN_CENTERS_REL`), and the photostim start/stop are already in session-absolute time. The result is placed in the same 80-bin grid as the neural data.

ii.
```python
abs_centers = go[:, None] + BIN_CENTERS_REL[None, :]
```

iii. N/A

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trials.outcome` and `trials.trial_instruction`. A hit means the animal licked the instructed side, a miss means it licked the opposite side, and ignore means no lick.

ii.
```python
opposite = np.where(instr == 'left', 'right', 'left')
choice_str = np.where(outcome == 'ignore', 'no lick',
                      np.where(outcome == 'hit', instr, opposite))
choice = np.select([choice_str == 'left', choice_str == 'right'], [0, 1], default=2).astype(np.int64)
```

iii. The AI verified that lick direction is not stored directly but can be derived from instruction and outcome. Also verified against actual lick timestamps with 99.7% agreement.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded as 0=left, 1=right, 2=no lick. Constant within a trial, repeated across all 80 bins.

ii.
```python
output_trials = [
    np.stack([
        np.full(N_BINS, choice[t], dtype=np.int64),
        # ...
    ])
    for t in range(n_trials)
]
```

iii. Three categories matching the instructions.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From `trials.outcome` directly, which holds `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
outcome = outcome_all[trial_idx]
outcome_code = np.select([outcome == 'ignore', outcome == 'miss'], [0, 1], default=2).astype(np.int64)
```

iii. The trials table stores outcome explicitly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to 0=ignore, 1=miss, 2=hit. Constant within a trial, repeated across all 80 bins.

ii.
```python
outcome_code = np.select([outcome == 'ignore', outcome == 'miss'], [0, 1], default=2).astype(np.int64)
# ...
np.full(N_BINS, outcome_code[t], dtype=np.int64),
```

iii. Direct mapping from the three string values.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From `trials.early_lick`, which holds `'no early'` and `'early'`.

ii.
```python
early = early_all[trial_idx]
early_code = (early == 'early').astype(np.int64)
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0=no, 1=yes. Constant within a trial, repeated across all 80 bins.

ii.
```python
early_code = (early == 'early').astype(np.int64)
# ...
np.full(N_BINS, early_code[t], dtype=np.int64),
```

iii. Simple binary mapping.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which has shape `(n_frames, 3)` = `(tongue_x, tongue_y, tongue_likelihood)` with corresponding `timestamps`. Column 1 (tongue_y) is the value, column 2 (likelihood) determines visibility.

ii.
```python
tongue = nwb.acquisition['BehavioralTimeSeries']['Camera0_side_TongueTracking']
ttimes = np.asarray(tongue.timestamps[:], dtype=np.float64)
tdata = np.asarray(tongue.data[:], dtype=np.float64)
visible = tdata[:, 2] > TONGUE_LIKELIHOOD_THRESH  # TONGUE_LIKELIHOOD_THRESH = 0.9
```

iii. This is the only tongue measurement in the file. The AI noted the channel layout is described in the series' own description attribute.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Four steps:
1. Frames with likelihood <= 0.9 are marked as not visible.
2. Visible frames are averaged into 50 ms bins for the whole session, and the 40th/60th percentiles of all visible binned y-values give two class edges.
3. Each trial window is binned and digitized against those edges (0/1/2).
4. Bins with no visible frame get class 3 ("not visible").

ii.
```python
TONGUE_LIKELIHOOD_THRESH = 0.9
visible = tdata[:, 2] > TONGUE_LIKELIHOOD_THRESH
mean_y, vis_counts = bin_visible_mean(ttimes, tdata[:, 1], visible, go)
tongue_class, pcts = discretise_tongue(mean_y, vis_counts)
```

```python
def discretise_tongue(mean_y, counts):
    vis = counts > 0
    out = np.full(mean_y.shape, 3, dtype=np.int64)
    p40, p60 = np.percentile(mean_y[vis], [TONGUE_PCT_LOW, TONGUE_PCT_HIGH])
    y = mean_y[vis]
    cls = np.where(y < p40, 0, np.where(y > p60, 2, 1))
    out[vis] = cls
    return out, (float(p40), float(p60))
```

iii. The AI noted the likelihood distribution is strongly bimodal (85.6% < 0.01, 14.0% > 0.99), so the threshold is insensitive in the range 0.1–0.99. The percentiles are computed over binned y-values (not raw frames).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session: the 40th and 60th percentiles of all visible binned tongue-y values across the session define two boundaries. Bins with mean y < 40th percentile get class 0, between 40th-60th get class 1, > 60th get class 2, and no visible frames get class 3.

ii.
```python
p40, p60 = np.percentile(mean_y[vis], [TONGUE_PCT_LOW, TONGUE_PCT_HIGH])
cls = np.where(y < p40, 0, np.where(y > p60, 2, 1))
```

iii. The AI computed percentiles over the per-trial binned means (from the `bin_visible_mean` function applied to go-aligned trial windows), not over the whole-session grid. The `discretise_tongue` function takes `mean_y` which is `(n_trials, N_BINS)` from the trial windows.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock as spikes and go cues. The `bin_visible_mean` function uses go-cue-aligned bin edges (same as neural data) to bin the visible frames.

ii.
```python
def bin_visible_mean(times, values, visible, go_times):
    edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()
    idx = np.searchsorted(tv, edges).reshape(n_trials, N_BINS + 1)
    # ...
```

iii. Same bin grid as the neural data ensures alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Session with no good units**: Skipped (1 session).
- **Trials without ephys coverage**: Excluded via `obs_intervals` matching.
- **Auto-water/free-water trials**: Excluded.
- **Zero-spike trials**: Dropped (2 trials).
- **Non-monotonic video timestamps**: Sorted before processing (2 sessions with duplicated frames).
- **Missing/truncated video**: Tongue class defaults to "not visible".
- **Unsorted spike times**: Sorted if not monotonic.
- **Trials extending past recording**: Binned as zero rate / not visible (documented as known limitation).

ii.
```python
if ttimes.size > 1 and not np.all(np.diff(ttimes) >= 0):
    o = np.argsort(ttimes, kind='stable')
    ttimes, tdata = ttimes[o], tdata[o]
# ...
if st.size > 1 and not np.all(np.diff(st) >= 0):
    st = np.sort(st)
```

iii. The AI documented each edge case and its handling in CONVERSION_NOTES.md Step 10. The approach is to exclude data that cannot be meaningfully processed, and use explicit "not visible" categories where measurements are absent.

## 10-a. What are the most time-consuming steps of the code?

i. Reading NWB files and extracting spike times dominate. Per-session total is ~1.3 s. With 14 parallel workers, the full 174-session conversion takes ~42 s wall clock. Pickling the 11.89 GB result takes ~15 s additional.

ii. N/A (timing information from CONVERSION_NOTES.md Step 7)

iii. The AI profiled each step and estimated total times. The multiprocessing approach brought wall-clock time well below the 15-minute budget.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two loops remain:
1. Per-unit loop in `bin_spike_rates`: one `searchsorted` per neuron (but all trials vectorized via flattened edges). Cannot be vectorized because each unit has a different number of spikes (ragged storage).
2. No per-trial tongue loop exists in the AI's code -- tongue binning uses cumulative sums (`bin_visible_mean`) which is vectorized.

ii.
```python
for i, st in enumerate(spike_times_list):
    idx = np.searchsorted(st, edges).reshape(n_trials, N_BINS + 1)
    out[i] = np.diff(idx, axis=1)
```

iii. The per-unit loop is inherent to ragged spike-time storage and cannot be further vectorized. The AI's tongue binning is fully vectorized via cumulative sums.

## 10-c. What processing does the code repeat multiple times?

i. Nothing is recomputed. Each NWB file is opened once and every quantity is derived in a single pass. The bin grid is defined once at module level.

ii. N/A

iii. The single-pass design avoids redundant computation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores extensive per-session metadata (`info` dict with timing, tongue percentiles, fraction of visible tongue bins, etc.) that goes into `metadata['session_info']` but is not used by the decoder. The brain region assignment involves an elaborate 14-region CCF mapping with coordinate checks that produces more detailed region information than the reference's simple label extraction. The `_plot_processing` function (for `--show-processing` mode) is extra but only runs when requested.

ii.
```python
info = {
    'session_id': session_id,
    'subject': subject,
    'subject_id': nwb.subject.subject_id,
    'session_start_time': str(nwb.session_start_time),
    # ... many fields
    'timing': {k: round(v, 3) for k, v in timing.items()},
    'total_time': round(time.time() - t_start, 2),
}
```

iii. The metadata is informational and doesn't add significant computation time. The brain region mapping is more complex than needed but produces a valid result.
