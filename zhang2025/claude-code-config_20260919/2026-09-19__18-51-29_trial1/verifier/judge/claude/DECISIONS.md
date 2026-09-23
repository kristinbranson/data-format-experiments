# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the `bwm_release.csv` freeze file from the reference code repository to enumerate all sessions (459 eids). For each session, it uses `SpikeSortingLoader` for neural data, `SessionLoader` for trials/wheel/motion energy, and `load_trials_and_mask` (imported from the reference code) for trial curation. The sessions are processed in parallel via `ProcessPoolExecutor`.

ii.
```python
BWM_FREEZE = '/app/code/code_zhang2025/data/bwm_release.csv'
bwm = pd.read_csv(BWM_FREEZE, index_col=0)
eids = list(dict.fromkeys(bwm.eid.tolist()))
# ...
for i, eid in enumerate(eids):
    sub = bwm[bwm.eid == eid]
    jobs.append((eid, sub.pid.tolist(), sub.probe_name.tolist(),
                 sub.subject.iloc[0], sub.lab.iloc[0], show, out_dir))
```

iii. The AI chose to use the `bwm_release.csv` freeze file directly, which lists all 459 sessions with their probe IDs and subject/lab metadata. This avoids the need for ONE API searching and dataset availability checks, since the freeze file is the authoritative list from the reference code.

## 1-b. How are the data split into subjects?

i. Subject names are read from the `bwm_release.csv` file (one subject per session row). At assembly, unique sorted subject names form the `subjects` list, and `subject_idx` maps each session to its subject.

ii.
```python
subjects = sorted({r['subject'] for r in results})
subj_lookup = {s: i for i, s in enumerate(subjects)}
# ...
'subject_idx': np.array([subj_lookup[r['subject']] for r in results], dtype=np.int64),
```

iii. Subjects are already identified in the freeze file metadata; no parsing of paths is needed.

## 1-c. How are the data split into sessions?

i. Each `eid` in the freeze file is a session. The freeze file lists probe insertions (pids) per session, so all probes for a given eid are grouped and processed together as one session.

ii.
```python
eids = list(dict.fromkeys(bwm.eid.tolist()))
# Each eid processed by convert_session()
```

iii. Sessions are the natural unit of the BWM release; no splitting is required.

## 1-d. How are the data split into trials?

i. The trials table (loaded via `load_trials_and_mask`) has one row per trial. Each trial's neural/behavioral data is extracted using the trial's `stimOn_times` as the alignment event.

ii.
```python
trials, ref_mask = load_trials_and_mask(one=one, eid=eid,
                                        max_trial_len=MAX_TRIAL_LEN,
                                        sess_loader=sess_loader)
```

iii. The trials table is already organized as one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI uses the reference code's `load_trials_and_mask` function directly (imported, not reimplemented), which applies: RT in [0.08, 2.0] s, feedback_times - goCue_times <= 10 s, no NaN in {stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType}, and choice != 0. On top of this, the AI additionally requires: `probabilityLeft` in {0.2, 0.5, 0.8}, `choice` in {-1, +1}, finite `stimOn_times`, the trial window within the ephys recording span, and wheel + whisker motion energy coverage of the trial window.

ii.
```python
trials, ref_mask = load_trials_and_mask(one=one, eid=eid,
                                        max_trial_len=MAX_TRIAL_LEN,
                                        sess_loader=sess_loader)
mask = ref_mask.to_numpy().astype(bool)
prior_code = map_prior(trials['probabilityLeft'].to_numpy())
mask &= prior_code >= 0
choice_raw = trials['choice'].to_numpy(dtype=float)
mask &= np.isin(choice_raw, (-1.0, 1.0))
align_times = trials[PARAMS['align_time']].to_numpy(dtype=float)
mask &= np.isfinite(align_times)
# ...
with np.errstate(invalid='ignore'):
    mask &= (interval_begs >= rec_span[0]) & (interval_begs + BINSIZE * NBINS <= rec_span[1])
# ...
mask &= wheel_ok & me_ok
```

iii. The AI justified using `load_trials_and_mask` as "literally the reference implementation." The additional filters (valid prior, valid choice, ephys coverage, behavior coverage) are documented in CONVERSION_NOTES as necessary to ensure all required data streams cover each trial.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` from all probes of a session, loaded via `SpikeSortingLoader.load_spike_sorting()`. The cluster table provides quality labels and anatomical locations for filtering.

ii.
```python
ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
sp, cl, ch = ssl.load_spike_sorting()
cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
```

iii. These are the standard spike sorting outputs from the IBL pipeline.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 100 non-overlapping 20 ms bins over the 2 s window (-0.5 to +1.5 s around stimulus onset). Spike counts are stored as raw counts (float32), NOT converted to firing rates. When a session has multiple probes, units are merged using the reference code's `merge_probes` function.

ii.
```python
BINSIZE = 0.02
NBINS = int(np.ceil((WIN[1] - WIN[0]) / BINSIZE))  # 100
# ...
b = ((tt - begs[k]) / BINSIZE).astype(np.int64)
np.clip(b, 0, NBINS - 1, out=b)
counts = np.bincount(uu * NBINS + b, minlength=n_units * NBINS)
out[k] = counts.reshape(n_units, NBINS)
```

iii. The AI notes in CONVERSION_NOTES: "Neural data are spike counts per 20 ms bin, stored unnormalised; normalisation is a decoder-side step. So the converted dataset should hold raw counts."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` (well-isolated neurons) are kept. Additionally, clusters whose Beryl acronym is `root` or `void` are excluded. Sessions need at least 5 such neurons.

ii.
```python
QC_LABEL = 1.0
NON_GREY = ('root', 'void')
MIN_NEURONS = 5
# ...
beryl = np.asarray(br.acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl'))
label = clusters['label'].to_numpy(dtype=float)
keep = (label >= QC_LABEL) & ~np.isin(beryl, NON_GREY)
```

iii. The AI documented this as a "deliberate deviation from the reference code in favour of the reference paper." The data paper states 75,708 well-isolated neurons (108/probe) as the inclusion criterion. The AI also excludes `root` (white matter/unassigned), noting this follows the data paper's "restricted to regions designated grey matter."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`. The interval begins at `stimOn_times + (-0.5)` and ends at `stimOn_times + 1.5`. Spikes within this window are binned relative to the interval start.

ii.
```python
PARAMS = {'align_time': 'stimOn_times', 'time_window': (-0.5, 1.5)}
align_times = trials[PARAMS['align_time']].to_numpy(dtype=float)
interval_begs = align_times + WIN[0]
# In bin_spikes:
b = ((tt - begs[k]) / BINSIZE).astype(np.int64)
```

iii. This matches both the reference code parameters and the Decoder Task instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial. No rebinning is applied; spikes are directly counted into 20 ms bins.

ii.
```python
BINSIZE = PARAMS['binsize']  # 0.02
NBINS = int(np.ceil((WIN[1] - WIN[0]) / BINSIZE))  # 100
```

iii. Matches the reference code's `binsize: 0.02` and the methods paper's "20-ms bins, T = 100".

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from the bin grid itself. The time input is the left edge of each 20 ms bin, computed as `WIN[0] + BINSIZE * i` for i = 0..99, giving values from -0.50 to 1.48.

ii.
```python
BIN_LEFT_EDGES = WIN[0] + BINSIZE * np.arange(NBINS)  # -0.50 ... 1.48
# In convert_session:
inputs[:, 0, :] = BIN_LEFT_EDGES[None, :]
```

iii. The time input represents the time of each bin relative to stimulus onset. The AI uses bin left edges.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing beyond computing the bin edge positions. The same 100 values are used for every trial.

ii.
```python
BIN_LEFT_EDGES = WIN[0] + BINSIZE * np.arange(NBINS)
inputs[:, 0, :] = BIN_LEFT_EDGES[None, :]
```

iii. The time input is defined by the binning grid, not derived from data.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input values are the left edges of the same bins used for spike counting, so they are aligned by construction.

ii.
```python
BIN_LEFT_EDGES = WIN[0] + BINSIZE * np.arange(NBINS)  # same grid as bin_spikes
```

iii. Both neural binning and the time input use the same bin grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from the `probabilityLeft` column of the trials table. Block boundaries are detected where `probabilityLeft` changes value.

ii.
```python
def trial_number_in_block(probability_left):
    p = np.asarray(probability_left, dtype=float)
    out = np.zeros(len(p), dtype=np.int64)
    count = 0
    for i in range(len(p)):
        if i > 0 and not (p[i] == p[i - 1] or (np.isnan(p[i]) and np.isnan(p[i - 1]))):
            count = 0
        out[i] = count
        count += 1
    return out
```

iii. The trials table carries no explicit block identifier, so blocks are inferred from transitions in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A 0-based counter that resets at each block boundary (change in `probabilityLeft`). Computed on the complete trials table before any trial exclusion, so the count reflects the animal's actual position in the block. The value is broadcast across all 100 time bins for each trial.

ii.
```python
block_idx = trial_number_in_block(trials['probabilityLeft'].to_numpy())
# ...
inputs[:, 1, :] = block_idx[mask][:, None]
```

iii. Computing on the full trials table before filtering preserves the animal's real block position.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column from the trials table, which takes values +1 (left), -1 (right), and 0 (no response).

ii.
```python
choice_raw = trials['choice'].to_numpy(dtype=float)
mask &= np.isin(choice_raw, (-1.0, 1.0))
choice_out = ((1 - choice_raw[mask]) / 2).astype(np.int64)  # +1 -> 0 (left), -1 -> 1 (right)
```

iii. The IBL convention is +1 for left, -1 for right. No-response trials (choice=0) are excluded.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Recoding: +1 (left) -> 0, -1 (right) -> 1 via the formula `(1 - choice) / 2`. The value is broadcast across all 100 time bins.

ii.
```python
choice_out = ((1 - choice_raw[mask]) / 2).astype(np.int64)
outputs[:, 0, :] = choice_out[:, None]
```

iii. Matches the Decoder Task specification: left = 0, right = 1.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column from the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
PRIOR_MAP = {0.2: 0, 0.5: 1, 0.8: 2}
prior_code = map_prior(trials['probabilityLeft'].to_numpy())
```

iii. These are the block prior probabilities from the IBL task design.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Recoding: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. Trials with other values are excluded. Broadcast across 100 time bins.

ii.
```python
def map_prior(probability_left):
    p = np.asarray(probability_left, dtype=float)
    out = np.full(len(p), -1, dtype=np.int64)
    for value, code in PRIOR_MAP.items():
        out[np.isclose(p, value)] = code
    return out
```

iii. Matches the Decoder Task specification: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.position` and `_ibl_wheel.timestamps`, loaded via `SessionLoader.load_wheel()`. The loader interpolates position to 1 kHz, applies a Butterworth low-pass filter, and differentiates to get velocity. Speed is `abs(velocity)`.

ii.
```python
def load_wheel_speed(sess_loader):
    if sess_loader.wheel is None or len(sess_loader.wheel) == 0:
        sess_loader.load_wheel()
    return (sess_loader.wheel['times'].to_numpy(dtype=float),
            np.abs(sess_loader.wheel['velocity'].to_numpy(dtype=float)))
```

iii. Same approach as the reference code's `load_target_behavior('wheel-speed')`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps: (1) `SessionLoader.load_wheel()` interpolates position to 1 kHz and computes velocity via Butterworth low-pass filter; (2) the speed (absolute velocity) is resampled onto the bin right edges of each trial via `behavior_per_interval`; (3) the resampled values are discretized into 3 classes using within-session tertiles (33.33/66.67 percentiles).

ii.
```python
# Resampling onto bin right edges:
q = begs[ok][:, None] + BIN_RIGHT_EDGES[None, :] - WIN[0]
interp = np.interp(q.ravel(), target_times, target_vals).reshape(q.shape)
# Discretization:
def discretize_tertiles(values):
    flat = values.ravel()
    edges = np.percentile(flat, [100.0 / N_DISCRETE_BINS * i
                                 for i in range(1, N_DISCRETE_BINS)])
    binned = np.searchsorted(edges, flat, side='right').reshape(values.shape)
    return binned.astype(np.int64), edges
```

iii. The AI explicitly follows the reference code's `get_behavior_per_interval` interpolation grid (bin right edges) and coverage checks. Discretization into tertiles is the AI's own design choice for the decoder task.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Within-session tertiles: 33.33 and 66.67 percentiles over all retained trials x time bins of that session. Uses `np.searchsorted` with `side='right'`, producing classes 0 (low), 1 (medium), 2 (high).

ii.
```python
edges = np.percentile(flat, [100.0 / N_DISCRETE_BINS * i
                             for i in range(1, N_DISCRETE_BINS)])
binned = np.searchsorted(edges, flat, side='right').reshape(values.shape)
```

iii. The AI argues that both signals are in session-specific units, so global thresholds would be inappropriate. Per-session tertiles give equal class priors.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is resampled onto the bin right edges of the same 20 ms grid used for neural data, measured from the same stimulus onset.

ii.
```python
q = begs[ok][:, None] + BIN_RIGHT_EDGES[None, :] - WIN[0]
interp = np.interp(q.ravel(), target_times, target_vals).reshape(q.shape)
```

iii. Using the same temporal grid as the neural data ensures alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `{left,right}Camera.ROIMotionEnergy` with its frame times `_ibl_{left,right}Camera.times`. Left camera preferred, right as fallback. Loaded via `SessionLoader.load_motion_energy()`.

ii.
```python
def load_whisker_me(sess_loader):
    for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
        try:
            sess_loader.load_motion_energy(views=[view])
            df = sess_loader.motion_energy[key]
            t = df['times'].to_numpy(dtype=float)
            v = df['whiskerMotionEnergy'].to_numpy(dtype=float)
            # ...
            return t[finite], v[finite], view
        except Exception:
            continue
```

iii. Same preference order as the reference code's `bin_behaviors`.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is (no filtering or normalization). It is resampled onto the bin right edges via `behavior_per_interval`, then discretized into 3 within-session tertiles, same as wheel speed.

ii.
```python
mt, mv, me_view = load_whisker_me(sess_loader)
me_vals, me_ok = behavior_per_interval(mt, mv, interval_begs)
me_bin, me_edges = discretize_tertiles(me_kept)
```

iii. No additional processing beyond resampling and discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same method as wheel speed: within-session tertiles at 33.33/66.67 percentiles.

ii.
```python
me_bin, me_edges = discretize_tertiles(me_kept)
```

iii. Same rationale as wheel speed; ensures equal class priors per session.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: resampled onto the bin right edges of the neural data grid.

ii.
```python
me_vals, me_ok = behavior_per_interval(mt, mv, interval_begs)
```

iii. Same temporal grid ensures alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple layers of handling: (1) Probes with no spike sorting are skipped. (2) NaN camera timestamps are dropped before interpolation. (3) Trials not covered by wheel/whisker data are excluded. (4) Trials outside the ephys recording span are excluded. (5) Sessions with <5 neurons or <2 usable trials are skipped and recorded in `failed_sessions`. (6) Sessions with degenerate (tied) tertile edges are rejected. (7) NaN anywhere in resampled behavior rejects the trial.

ii.
```python
if len(sp) == 0 or 'times' not in sp or len(sp['times']) == 0:
    continue
# ...
if len(keep_idx) < MIN_NEURONS:
    raise RuntimeError(f'only {len(keep_idx)} well-isolated grey-matter units')
# ...
if n_trials_kept < MIN_TRIALS:
    raise RuntimeError(f'only {n_trials_kept} usable trials')
# ...
for name, edges in (('wheel speed', wheel_edges), ('whisker motion energy', me_edges)):
    if not np.all(np.diff(edges) > 0):
        raise RuntimeError(f'{name} is degenerate')
```

iii. The AI documented each missing-data scenario it encountered and the resolution applied, including cases found during critical review iterations.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting from disk is the dominant cost (~8 s/session), as the spike arrays can be hundreds of megabytes per probe. The AI measured per-step timing for each session.

ii.
```python
timing['load_spikes'] = time.time() - t  # ~8 s/session
timing['load_trials'] = time.time() - t   # ~0.5 s/session
timing['load_behavior'] = time.time() - t  # ~1.0 s/session
timing['bin_spikes'] = time.time() - t     # ~0.7 s/session
```

iii. The AI identified file I/O as the main bottleneck and used parallel processing (24 workers) to overlap NFS reads across sessions.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The `bin_spikes` function still loops over trials (one `np.bincount` per trial). This could be vectorized by offsetting each spike's index by its trial number. Similarly, `trial_number_in_block` uses a Python loop. However, the per-trial loop in `bin_spikes` costs ~0.7 s/session, which is small compared to I/O.

ii.
```python
for k in range(n_trials):
    if not np.isfinite(begs[k]) or i1[k] <= i0[k]:
        continue
    tt = spike_times[i0[k]:i1[k]]
    uu = spike_units[i0[k]:i1[k]]
    b = ((tt - begs[k]) / BINSIZE).astype(np.int64)
    np.clip(b, 0, NBINS - 1, out=b)
    counts = np.bincount(uu * NBINS + b, minlength=n_units * NBINS)
    out[k] = counts.reshape(n_units, NBINS)
```

iii. The AI noted the loop is dominated by I/O and "there is nothing to gain" from further vectorization at the processing level.

## 10-c. What processing does the code repeat multiple times?

i. The `SessionLoader` is created once and reused for trials, wheel, and motion energy loading, so there is no repeated processing. However, `behavior_per_interval` is called twice (once for wheel, once for whisker), each time computing `searchsorted` on the interval beginnings - these could share the interval computation.

ii.
```python
wt, wv = load_wheel_speed(sess_loader)
wheel_vals, wheel_ok = behavior_per_interval(wt, wv, interval_begs)
mt, mv, me_view = load_whisker_me(sess_loader)
me_vals, me_ok = behavior_per_interval(mt, mv, interval_begs)
```

iii. The repeated searchsorted is minimal overhead compared to I/O.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores extensive per-session metadata (`session_info`) including cluster UUIDs, kept trial indices, fraction correct, mean firing rate, tertile edges, and timing information. This metadata is not used by the decoder but serves documentation/provenance purposes. The AI also computes `rec_span` (intersection of probe recording spans) for every session even when there is only one probe.

ii.
```python
'info': {
    'eid': eid, 'subject': subject, 'lab': lab,
    'n_probes': len(pids), 'n_clusters_all': int(n_all_clusters),
    'n_clusters_label_good': n_label_good, 'n_neurons': int(len(keep_idx)),
    'n_trials_raw': int(n_trials_raw), 'n_trials_kept': int(n_trials_kept),
    # ... extensive metadata
}
```

iii. The metadata is for verification and documentation, not for the decoder itself.
