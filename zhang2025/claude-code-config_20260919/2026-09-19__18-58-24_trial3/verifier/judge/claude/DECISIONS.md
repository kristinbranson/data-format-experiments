# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `bwm_release.csv` (shipped with the Zhang reference code) to enumerate all 459 released sessions and 699 probe insertions, then reads the ONE release tables (`sessions.pqt`, `datasets.pqt`) from disk. Because the shipped release tables are stale (trials only exist in revision folders), the AI rebuilds the ONE `datasets` cache table by walking each session's `alf` directory tree (`build_dataset_table`). A local ONE client is constructed with `make_one()`. Each session is then loaded through `SessionLoader` (trials, wheel, motion energy) and `SpikeSortingLoader` (spikes, clusters, channels).

ii.
```python
BWM_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
SESSIONS_PQT = f'{ROOT}/Brainwidemap/sessions.pqt'

def get_session_table():
    sess = pd.read_parquet(SESSIONS_PQT)
    bwm = pd.read_csv(BWM_CSV, index_col=0)
    paths = {}
    for eid in bwm.eid.unique():
        r = sess.loc[eid]
        paths[eid] = (f"{ROOT}/{r['lab']}/Subjects/{r['subject']}/"
                      f"{str(r['date'])}/{int(r['number']):03d}")
    return bwm, sess, paths

def build_dataset_table(session_paths_map):
    rows = []
    for eid, sp in session_paths_map.items():
        for dirpath, _, filenames in os.walk(os.path.join(sp, 'alf')):
            rel = os.path.relpath(dirpath, sp)
            for f in filenames:
                rows.append(...)
    ...

one = make_one({e: paths[e] for e in eids})
```

iii. The AI identified that the shipped ONE release tables are stale (trials only exist in revision folders, not at the unrevised path listed in the tables) and would cause `SessionLoader` to silently return a 1-column trials frame. Rebuilding the dataset table from the filesystem ensures ONE resolves exactly the files present. Probe IDs come from `bwm_release.csv` because `one.eid2pid` requires a network connection.

## 1-b. How are the data split into subjects?

i. Subject names come from the `subject` column of `bwm_release.csv`. At assembly, unique subjects are sorted and `subject_idx` maps each session to its subject.

ii.
```python
sub = bwm[bwm.eid == eid]
info['subject'] = sub.subject.iloc[0]
...
subjects = sorted({r['subject'] for r in kept})
subj_idx = np.array([subjects.index(r['subject']) for r in kept], dtype=np.int64)
```

iii. The subject name is directly available from the release CSV; no parsing of paths is needed.

## 1-c. How are the data split into sessions?

i. Sessions are the unique `eid` values from `bwm_release.csv`. Each session is processed independently.

ii.
```python
eids = list(bwm.eid.unique())
```

iii. A session is already the unit the release is organised by; each eid identifies one session.

## 1-d. How are the data split into trials?

i. The trials table (loaded via `SessionLoader.load_trials()`) has one row per trial. Trials are indexed by their row in this table.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
trials, mask = load_trials_and_mask(sl)
```

iii. The trials table is already one row per trial; no splitting is needed.

## 1-e. How are trials filtered based on quality controls?

i. The AI transcribes the reference `load_trials_and_mask` function from the Zhang code with several exclusion criteria: (1) NaN in any of 6 key trial events (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType); (2) reaction time (firstMovement_times - stimOn_times) outside [0.08, 2.0] s; (3) trial length (feedback_times - goCue_times) > 10 s; (4) choice == 0 (no response). Additionally, trials whose wheel or whisker trace does not cover the full trial window, or contains NaN, are dropped by `bin_behavior`. Finally, trials with zero total spike counts (neural dropouts) are also dropped.

ii.
```python
def load_trials_and_mask(sess_loader, min_rt=0.08, max_rt=2., nan_exclude='default',
                         min_trial_len=None, max_trial_len=10.0,
                         exclude_unbiased=False, exclude_nochoice=True):
    if nan_exclude == 'default':
        nan_exclude = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
                       'firstMovement_times', 'feedbackType']
    query = f'(firstMovement_times - stimOn_times < {min_rt})' if min_rt is not None else ''
    if max_rt is not None:
        query += f' | (firstMovement_times - stimOn_times > {max_rt})'
    if max_trial_len is not None:
        query += f' | (feedback_times - goCue_times > {max_trial_len})'
    for event in nan_exclude:
        query += f' | {event}.isnull()'
    if exclude_nochoice:
        query += ' | (choice == 0)'
    mask = ~sess_loader.trials.eval(query)
    ...

# Behavior coverage:
wheel_vals, wheel_good = bin_behavior(*traces['wheel-speed'], align_times)
whisk_vals, whisk_good = bin_behavior(*traces['whisker-motion-energy'], align_times)
beh_good = wheel_good & whisk_good

# Neural coverage:
neural_covered = binned.sum(axis=(1, 2)) > 0
keep = beh_good & neural_covered
```

iii. The trial mask is a verbatim transcription of the Zhang reference code's `load_trials_and_mask` with the same default parameters, including `max_trial_len=10.0` which the caching script passes. The behavior coverage and neural dropout checks are additional safety measures. The AI documented that the BWM data paper's trial exclusion criteria are the same as the reference code's defaults.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times` and `spikes.clusters`, loaded via `SpikeSortingLoader.load_spike_sorting()`. The cluster table supplies the quality label and anatomical location.

ii.
```python
loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = loader.load_spike_sorting()
clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
```

iii. Spike times and cluster assignments are the fundamental data for electrophysiology; the loaders are the same as the Zhang reference code.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the trial window [-0.5, 1.5] s relative to stimulus onset, giving one count per cluster per bin. The bin assignment uses `floor((t - t_beg) / binsize)`, matching the reference's `bincount2D`. The counts are stored as **raw spike counts** (float32), NOT divided by the bin width. When a session has multiple probes, their clusters are merged into one population with re-indexed cluster IDs.

ii.
```python
def bin_spiking_data(spike_times, spike_clusters, n_clusters, align_times):
    ...
    idx = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)
    keep = (idx >= 0) & (idx < NBINS)
    flat = c[keep].astype(np.int64) * NBINS + idx[keep]
    counts = np.bincount(flat, minlength=n_clusters * NBINS)
    out[k] = counts.reshape(n_clusters, NBINS)
    return out  # float32 counts, NOT divided by BINSIZE
```

iii. The AI states: "Neural values: raw spike counts (float32), not rates or z-scores -- the reference caches counts and z-scoring is a model-side step." This is justified by the Zhang reference code which stores raw counts in its cache and applies z-scoring only in the model's data loader.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` (QC_LABEL = 1.0) are kept, matching the BWM paper's "well-isolated neurons" criterion. This yields 75,708 of 621,733 units across the release. Spikes with NaN times are also removed. The AI does NOT filter out neurons whose Beryl acronym is `void` (channels placed outside the brain by histology) or `root`.

ii.
```python
QC_LABEL = 1.0

iok = clusters_labeled['label'] >= qc
selected_clusters = clusters_labeled[iok]
spike_idx, ib = ismember(spikes['clusters'], selected_clusters.index)
selected_spikes = {k: v[spike_idx] for k, v in spikes.items()}

# NaN spike times:
finite = np.isfinite(spikes['times'])
st = spikes['times'][finite]
```

iii. The AI applies the same QC threshold as the BWM data paper and the Zhang code's `load_spiking_data(qc=1)`. The AI explicitly chose not to filter `void`/`root` neurons, reasoning that "No region filtering: whole-session decoding, matching the reference's `region='all'` setting."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spike window begins at `stimOn_times + TIME_WINDOW[0]` (-0.5 s) and ends at `stimOn_times + TIME_WINDOW[1]` (+1.5 s). Spikes within this window are binned relative to the window start using `floor((t - t_beg) / binsize)`.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)

align_times = trials_sel[ALIGN_TIME].to_numpy()
begs = align_times + TIME_WINDOW[0]
ends = align_times + TIME_WINDOW[1]
```

iii. All streams share the same session clock (IBL synchronization), so subtracting stimulus onset from spike times is sufficient alignment. This matches both the reference code and the decoder task requirement.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial, covering the 2 s window. No rebinning or interpolation is applied.

ii.
```python
BINSIZE = 0.02
NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```

iii. Matches the reference code's `'binsize': 0.02` and the methods paper: "split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` in the trials table, which defines the alignment event. The time input is the centre of each of the 100 bins in the trial window.

ii.
```python
tvec = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
```

iii. The time vector is defined by the binning parameters and the alignment event; it is the same for every trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing of raw data; the variable is defined analytically as the bin centres of the decoding window.

ii.
```python
tvec = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
# = -0.49, -0.47, ..., 1.49
```

iii. N/A -- the variable is defined by the binning grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input is the centre of each neural bin, so it is inherently aligned with the neural data bin for bin.

ii.
```python
tvec = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
inp[0] = tvec  # same 100-element vector for every trial
```

iii. The bin centres define both the time input and the neural binning grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. Since `probabilityLeft` is constant within a block, a change in its value marks a new block boundary.

ii.
```python
def trial_number_in_block(probability_left):
    p = np.asarray(probability_left, dtype=float)
    newblock = np.ones(len(p), dtype=bool)
    if len(p) > 1:
        same = (p[1:] == p[:-1])
        newblock[1:] = ~same
    block_id = np.cumsum(newblock) - 1
    ...
```

iii. The trials table carries no explicit block identifier, so blocks must be recovered from transitions in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Block boundaries are detected from changes in `probabilityLeft` (NaN-safe: NaN starts a new block). Within each block, trials are numbered from 0. The computation is done on the **full** trials table before any trial exclusion, so the block position reflects the animal's true position in the block.

ii.
```python
tnib_all, block_id_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
# ... later, after filtering:
tnib = tnib_all[mask]
tnib_keep = tnib[keep]
```

iii. Computing before trial exclusion ensures that a dropped trial still advances the block count, preserving the animal's real position.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, where IBL codes +1 for left, -1 for right, and 0 for no response.

ii.
```python
choice_raw = trials_keep['choice'].to_numpy()
choice = (choice_raw < 0).astype(np.int8)
```

iii. The IBL convention is documented in the data paper and the data architecture document.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Recoded to 0 for left (+1 in IBL) and 1 for right (-1 in IBL). No-response trials (choice == 0) are excluded by the trial mask.

ii.
```python
choice = (choice_raw < 0).astype(np.int8)  # +1 (left) -> False -> 0, -1 (right) -> True -> 1
```

iii. Matches the decoder task specification: "left = 0, right = 1".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
pleft = trials_keep['probabilityLeft'].to_numpy()
prior = np.full(ntrials, -1, dtype=np.int8)
for val, lab in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior[np.isclose(pleft, val)] = lab
```

iii. The three values correspond to the block prior; the mapping follows the decoder task specification.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Recoded from continuous values to categorical: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. Uses `np.isclose` for floating-point comparison.

ii.
```python
for val, lab in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior[np.isclose(pleft, val)] = lab
if np.any(prior < 0):
    info['skip_reason'] = f'unexpected probabilityLeft values {np.unique(pleft)}'
    return None, info
```

iii. Matches the decoder task specification: "0.2 -> 0, 0.5 -> 1, 0.8 -> 2". The `np.isclose` comparison and validation of unexpected values adds robustness.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The wheel position and timestamps (`_ibl_wheel.position`, `_ibl_wheel.timestamps`), loaded via `SessionLoader.load_wheel()` which resamples to 1 kHz, applies a Butterworth low-pass filter, and computes velocity. The speed is the absolute value of velocity.

ii.
```python
sess_loader.load_wheel()
out['wheel-speed'] = (sess_loader.wheel['times'].to_numpy(),
                      np.abs(sess_loader.wheel['velocity'].to_numpy()))
```

iii. Follows the reference code's `load_target_behavior('wheel-speed')` which uses `abs(velocity)` from `SessionLoader`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps: (1) `SessionLoader.load_wheel()` interpolates wheel position to a 1 kHz grid and applies a 20 Hz Butterworth filter to compute velocity; speed is `|velocity|`. (2) The speed trace is linearly interpolated onto the **right edges** of each 20 ms neural bin for each trial, using `np.linspace(t_beg + binsize, t_end, NBINS)`. (3) The trace is discretized into 3 equal-occupancy classes using within-session tertile edges (quantiles at 1/3 and 2/3 of the flattened session data).

ii.
```python
# Resampling at bin right edges:
def bin_behavior(target_times, target_values, align_times):
    ...
    x = np.linspace(begs[k] + BINSIZE, ends[k], NBINS)
    values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
    ...

# Discretization:
def discretize_tertiles(values):
    flat = values.ravel()
    edges = np.quantile(flat, [1. / 3., 2. / 3.])
    labels = np.digitize(values, edges).astype(np.int8)
    return labels, edges
```

iii. The AI explicitly followed the Zhang reference code's `get_behavior_per_interval`, which samples behavior at `np.linspace(t_beg + binsize, t_end, n_bins)` -- the right edges of the neural bins. The tertile discretization ensures equal class occupancy within each session, making chance exactly 1/3 for balanced accuracy.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Within-session tertiles: the 1/3 and 2/3 quantiles of all wheel speed values across all retained trials and timepoints of that session define two thresholds. `np.digitize` maps values below the lower threshold to 0 (low), between to 1 (medium), and above to 2 (high).

ii.
```python
def discretize_tertiles(values):
    flat = values.ravel()
    edges = np.quantile(flat, [1. / 3., 2. / 3.])
    if not edges[1] > edges[0]:
        # degenerate case handling
        ...
    labels = np.digitize(values, edges).astype(np.int8)
    return labels, edges
```

iii. Equal-occupancy binning ensures classes are balanced and handles the fact that wheel speed distributions vary across sessions and animals. The AI also handles the degenerate case where a mostly-constant trace yields identical edges.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is sampled at the **right edges** of each neural bin (`t_beg + binsize` to `t_end` in NBINS steps), so it is aligned to the neural binning but shifted by half a bin width compared to the bin centres.

ii.
```python
x = np.linspace(begs[k] + BINSIZE, ends[k], NBINS)
values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
```

iii. This follows the Zhang reference code's `get_behavior_per_interval` which uses `np.linspace(t_beg + binsize, t_end, n_bins)`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The whisker-pad ROI motion energy from the side camera: `leftCamera.ROIMotionEnergy` preferred, falling back to `rightCamera.ROIMotionEnergy` when the left camera is unavailable.

ii.
```python
def load_behavior_traces(sess_loader):
    ...
    for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
        try:
            sess_loader.load_motion_energy(views=[view])
            df = sess_loader.motion_energy[cam]
            whisker = (df['times'].to_numpy(), df['whiskerMotionEnergy'].to_numpy())
            break
        except Exception:
            continue
```

iii. Follows the reference code logic: left camera first, right as fallback.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion energy trace is used as-is (no filtering or normalization). It is resampled onto the **right edges** of the neural bins via linear interpolation (same as wheel speed), then discretized into 3 equal-occupancy classes using within-session tertile edges.

ii.
```python
whisk_vals, whisk_good = bin_behavior(*traces['whisker-motion-energy'], align_times)
whisk_lab, whisk_edges = discretize_tertiles(whisk_vals)
```

iii. Same processing pipeline as wheel speed. The AI notes that whisker motion energy is in arbitrary camera-dependent units, so no fixed threshold transfers across sessions.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identical to wheel speed: within-session tertiles at the 1/3 and 2/3 quantiles of all values across the session.

ii.
```python
whisk_lab, whisk_edges = discretize_tertiles(whisk_vals)
```

iii. Same justification as wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: sampled at the right edges of the neural bins.

ii.
```python
whisk_vals, whisk_good = bin_behavior(*traces['whisker-motion-energy'], align_times)
# bin_behavior uses np.linspace(begs[k] + BINSIZE, ends[k], NBINS)
```

iii. Same alignment approach as wheel speed, following the Zhang reference code.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several layers of handling: (1) Sessions without whisker motion energy are skipped (14 sessions). (2) Sessions with unusable camera timestamps (trace doesn't span trial windows) are skipped (1 session). (3) Sessions with fewer than 5 well-isolated neurons (`MIN_NEURONS = 5`) are skipped (2 sessions). (4) Trials where the behavior trace doesn't cover the full window, contains NaN, or has no data are marked as bad by `bin_behavior`. (5) Trials with zero total spike counts (recording dropouts) are dropped. (6) Sessions with fewer than 2 surviving trials are dropped. (7) NaN spike times are filtered before binning. (8) The `build_dataset_table` function works around stale ONE release tables.

ii.
```python
# Behavior coverage in bin_behavior:
if len(vv) == 0: continue
if np.isnan(vv).any(): continue
if np.abs(begs[k] - tt[0]) > BINSIZE: continue
if np.abs(ends[k] - tt[-1]) > BINSIZE: continue

# Neural coverage:
neural_covered = binned.sum(axis=(1, 2)) > 0

# Session filtering:
if n_clusters < MIN_NEURONS:
    info['skip_reason'] = ...
    return None, info
```

iii. The AI documented each type of data issue and its handling in CONVERSION_NOTES.md. The handling follows the reference code's patterns (behavior coverage checks from `get_behavior_per_interval`, trial mask from `load_trials_and_mask`) with additions for neural dropouts and minimum neuron counts.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk is the dominant cost. The AI reports ~1.5-3 s per probe for loading spike data, compared to ~0.2 s for binning and ~1.5 s for behavior loading. With parallel processing (24 workers), the full conversion runs in ~162 s for 459 sessions.

ii.
```python
loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = loader.load_spike_sorting()
```

iii. Spike sorting files can be hundreds of megabytes per probe; the cost is mainly disk I/O.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two per-trial loops: (1) `bin_spiking_data` loops over trials to bin spikes, using `np.searchsorted` + `np.bincount` per trial. (2) `bin_behavior` loops over trials to interpolate behavior traces. The `trial_number_in_block` function also has a loop over unique blocks. These loops could theoretically be vectorized by offsetting indices across trials.

ii.
```python
# Per-trial spike binning:
for k in range(ntrials):
    a, b = i0[k], i1[k]
    ...
    flat = c[keep].astype(np.int64) * NBINS + idx[keep]
    counts = np.bincount(flat, minlength=n_clusters * NBINS)
    out[k] = counts.reshape(n_clusters, NBINS)

# Per-trial behavior interpolation:
for k in range(ntrials):
    ...
    values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
```

iii. The AI notes these are fast enough (~0.2 s for binning per session) that vectorization isn't needed.

## 10-c. What processing does the code repeat multiple times?

i. No significant repeated processing is evident. Each session is processed once, each probe is loaded once, and the per-trial loops within a session run once per trial.

ii. N/A

iii. N/A

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads and stores extensive metadata per session (full exclusion counts, timing info, tertile edges, etc.) in `metadata['session_info']`, which is not used by the decoder. The `build_dataset_table` scans every file in every session's `alf` tree even though only a subset is needed. The code also checks for NaN spike times (`np.isfinite(spikes['times'])`) even though none exist in this release.

ii.
```python
# Extensive per-session metadata:
session_info.append({
    'eid': r['eid'], 'subject': r['subject'], 'lab': r['lab'],
    'n_probes': r['n_probes'],
    'n_units_total': i['n_units_total'],
    'n_trials_total': i['n_trials_total'],
    'n_trials_after_trialmask': i['n_trials_after_trialmask'],
    'n_trials_after_behmask': i['n_trials_after_behmask'],
    'n_trials_no_spikes': i['n_trials_no_spikes'],
    ...
})
```

iii. The metadata is useful for debugging and documentation but is not consumed by the decoder.
