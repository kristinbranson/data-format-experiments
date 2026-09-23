# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads `bwm_release.csv` (from the reference code directory `/app/code/code_zhang2025/data/`) to get the list of session eids, subjects, and labs. It then uses the ONE API (`ONE(base_url=..., silent=True, cache_dir=...)`) to load data for each session. `SessionLoader` loads trials, wheel, and motion energy; `SpikeSortingLoader` loads spike sorting per probe. The ONE client works offline using cached Alyx REST responses.

ii.
```python
bwm = pd.read_csv(BWM_RELEASE, index_col=0)
sess_df = bwm.drop_duplicates('eid')[['eid', 'subject', 'lab']].sort_values('eid')
```

```python
one = ONE(base_url=BASE_URL, silent=True, cache_dir=CACHE_DIR)
```

iii. The AI chose to use `bwm_release.csv` as the canonical session list because it is used by the reference caching script (`0_data_caching.py`). The ONE client is constructed identically to the reference code.

## 1-b. How are the data split into subjects?

i. Subject names come from the `subject` column of `bwm_release.csv`. Sessions are grouped by subject via `sess_df` and later assembled with sorted unique subject names.

ii.
```python
subjects = sorted({r['subject'] for r in good})
sub_idx = np.array([subjects.index(r['subject']) for r in good], dtype=np.int64)
```

iii. Subject information is directly available in `bwm_release.csv`, so no parsing or derivation is needed.

## 1-c. How are the data split into sessions?

i. Each row in `bwm_release.csv` (after deduplication on `eid`) is one session. The AI processes each eid independently.

ii.
```python
sess_df = bwm.drop_duplicates('eid')[['eid', 'subject', 'lab']].sort_values('eid')
```

iii. Sessions are the natural unit of the release; each eid uniquely identifies a session.

## 1-d. How are the data split into trials?

i. The trials table loaded by `SessionLoader.load_trials()` has one row per trial. The AI uses `load_trials_and_mask` from the reference code to get the trials DataFrame and a boolean mask.

ii.
```python
trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0,
                                    sess_loader=sess_loader)
```

iii. The trials table is already one row per trial; no splitting is needed.

## 1-e. How are trials filtered based on quality controls?

i. The AI uses the reference function `load_trials_and_mask` unchanged, which excludes: trials with NaN in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, or `feedbackType`; reaction time outside [0.08, 2.0] s; `choice == 0` (no-go); feedback-goCue > 10 s. Additionally, the AI requires the behavioral streams (wheel speed and whisker motion energy) to cover the trial's 2 s window (using `bin_behavior` validity checks). Finally, trials with zero spikes in the entire window are dropped.

ii.
```python
trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0,
                                    sess_loader=sess_loader)
# ...
keep = mask & finite_align & ws_valid & me_valid
# ...
nonempty = counts.sum(axis=(1, 2)) > 0
```

iii. The AI justifies using the reference `load_trials_and_mask` as it exactly matches the paper's curation criteria. The additional behavioral coverage filter ensures no NaN or missing behavioral data enters the converted dataset. The zero-spike filter removes trials in recording gaps.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` loaded via `SpikeSortingLoader.load_spike_sorting()` for each probe, plus `clusters.label` and `clusters.acronym` for filtering.

ii.
```python
loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = loader.load_spike_sorting()
clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
```

iii. The spike times and cluster assignments are the standard inputs for computing spike counts per neuron.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the 2 s trial window (100 bins). The AI stores **raw spike counts** (not firing rates). When a session has multiple probes, they are merged using the reference `merge_probes` function, which concatenates and re-indexes clusters.

ii.
```python
def bin_spikes(spike_times, spike_clusters, t0s, n_neurons, nbins=NBINS, binsize=BINSIZE):
    out = np.zeros((ntrials, n_neurons, nbins), dtype=np.float32)
    # ...
    bins = ((spike_times[a:b] - t0s[k]) / binsize).astype(np.int64)
    np.clip(bins, 0, nbins - 1, out=bins)
    idx = spike_clusters[a:b] * nbins + bins
    counts = np.bincount(idx, minlength=n_neurons * nbins)
    out[k] = counts.reshape(n_neurons, nbins).astype(np.float32)
```

iii. The AI stores raw spike counts rather than firing rates, noting that the reference caching script also stores counts and that z-scoring happens in the decoder itself.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` (well-isolated, passing all three RIGOR single-unit metrics) are kept. Additionally, units whose Beryl-mapped acronym is `root` or `void` are dropped. Sessions with fewer than 5 remaining units are skipped entirely.

ii.
```python
QC_LABEL = 1.0
NON_GREY = ('root', 'void')
MIN_NEURONS = 5
# ...
iok = clusters_labeled['label'] >= qc
# ...
beryl = BrainRegions().acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl')
keep = ~np.isin(beryl, NON_GREY)
# ...
if n_units < MIN_NEURONS:
    return {'eid': eid, 'skip': f'only {n_units} well-isolated grey-matter units'}
```

iii. The AI justifies label >= 1 as matching the data paper's "well-isolated neurons" definition (75,708 of 621,733). Dropping `root` and `void` follows the reference multi-region loader. The MIN_NEURONS threshold is justified as ensuring meaningful decoding.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to `stimOn_times`. The trial window starts at `stimOn - 0.5 s` and ends at `stimOn + 1.5 s`. Spike times are binned relative to this window start.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
# ...
align = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
t0s_all = align + TIME_WINDOW[0]
# ...
bins = ((spike_times[a:b] - t0s[k]) / binsize).astype(np.int64)
```

iii. This follows both the reference code (`align_time='stimOn_times'`, `time_window=(-0.5, 1.5)`) and the instructions ("Temporally align based on stimulus onset").

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20 ms, producing 100 bins over the 2 s window. No rebinning or interpolation is applied to the neural data.

ii.
```python
BINSIZE = 0.02
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
```

iii. Matches the reference code's `binsize: 0.02` and the methods paper's description of "20-ms bins, producing T = 100 time steps".

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From the bin geometry itself. The time input is computed as the center of each 20 ms bin in the [-0.5, 1.5] s window around `stimOn_times`.

ii.
```python
bin_centers = (TIME_WINDOW[0] + (np.arange(NBINS) + 0.5) * BINSIZE).astype(np.float32)
```

iii. This is a constructed variable representing time since stimulus onset, identical for every trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing of raw data. The bin centers are computed from the bin geometry: `t_start + (k + 0.5) * binsize` for k = 0..99, giving values from -0.49 to 1.49 s.

ii.
```python
bin_centers = (TIME_WINDOW[0] + (np.arange(NBINS) + 0.5) * BINSIZE).astype(np.float32)
# ...
inputs.append(np.stack([bin_centers, np.full(NBINS, tib[i], dtype=np.float32)]))
```

iii. This is defined by the decoder task specification; no raw data processing is involved.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input values are the bin centers of the same 20 ms bins used for neural data, so they are inherently aligned.

ii.
```python
bin_centers = (TIME_WINDOW[0] + (np.arange(NBINS) + 0.5) * BINSIZE).astype(np.float32)
```

iii. Since both neural and time input use the same bin grid, alignment is guaranteed by construction.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `trials.probabilityLeft`, which is constant within a block. A change in its value indicates a new block.

ii.
```python
def trial_number_in_block(prob_left):
    pl = np.asarray(prob_left, dtype=np.float64)
    changed = np.ones(len(pl), dtype=bool)
    changed[1:] = ~(pl[1:] == pl[:-1])
    block_id = np.cumsum(changed) - 1
    idx = np.zeros(len(pl), dtype=np.int64)
    for b in np.unique(block_id):
        m = block_id == b
        idx[m] = np.arange(m.sum())
    return idx, block_id
```

iii. The trials table has no block identifier, so it must be inferred from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Blocks are identified by detecting changes in `probabilityLeft`. NaN comparisons yield False, starting a new block. Within each block, trials are numbered 0, 1, 2, ... The count is computed over all trials before any filtering, then indexed into the kept trials.

ii.
```python
tib_all, _ = trial_number_in_block(trials['probabilityLeft'].to_numpy())
tib = tib_all[keep].astype(np.float32)
```

iii. Computing the block index over all trials before filtering ensures the animal's actual position in the block is preserved.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From `trials.choice`, where +1 = left choice, -1 = right choice, 0 = no-go.

ii.
```python
choice = trials['choice'].to_numpy()[keep]
choice_cls = (choice < 0).astype(np.int32)
```

iii. IBL convention is +1 for left, -1 for right.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Left (+1) is mapped to 0, right (-1) is mapped to 1, using `(choice < 0).astype(int)`. No-go trials (choice == 0) are excluded by the trial mask.

ii.
```python
choice_cls = (choice < 0).astype(np.int32)
```

iii. The mapping follows the instructions: "left = 0, right = 1".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `trials.probabilityLeft`, which takes values 0.2, 0.5, or 0.8.

ii.
```python
pleft = trials['probabilityLeft'].to_numpy()[keep]
prior_cls = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                       np.isclose(pleft, 0.8)], [0, 1, 2], default=-1).astype(np.int32)
```

iii. These three values correspond to the block prior probabilities in the IBL task.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The three probability values are mapped to class labels: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. The AI uses `np.isclose` for floating-point comparison and `np.select` for the mapping.

ii.
```python
prior_cls = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                       np.isclose(pleft, 0.8)], [0, 1, 2], default=-1).astype(np.int32)
```

iii. Follows the instructions: "0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position` and `_ibl_wheel.timestamps`, loaded via `SessionLoader.load_wheel()`. The velocity is computed internally by `SessionLoader` (interpolation to 1 kHz, Butterworth low-pass filtering, differentiation), and the speed is the absolute value.

ii.
```python
sess_loader.load_wheel()
wheel_times = sess_loader.wheel['times'].to_numpy()
wheel_speed = np.abs(sess_loader.wheel['velocity'].to_numpy())
```

iii. Follows the reference code's `load_target_behavior('wheel-speed')` which computes `np.abs(velocity)`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel speed (absolute velocity) is linearly interpolated onto the **right edges** of each 20 ms bin for each trial. The interpolation grid is `t0 + (1..100) * binsize`, matching the reference code's `get_behavior_per_interval` which uses `linspace(t_beg + binsize, t_end, n_bins)`. The interpolated values are then discretized into 3 classes at the per-session 33.3 and 66.7 percentiles.

ii.
```python
grid = t0s[:, None] + (np.arange(1, nbins + 1)[None, :]) * binsize
# ...
interp = np.interp(grid.ravel(), times[finite], values[finite]).reshape(grid.shape)
# ...
def discretize_tertiles(values):
    edges = np.percentile(values, [100.0 / 3.0, 200.0 / 3.0])
    classes = np.digitize(values, edges, right=False).astype(np.int32)
    return classes, edges
```

iii. The AI explicitly follows the reference code's `get_behavior_per_interval` which interpolates at bin right edges, not bin centers.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Per-session tertiles: the 33.3rd and 66.7th percentiles of all wheel speed values across all kept trials and time bins of a session define the boundaries between low (0), medium (1), and high (2).

ii.
```python
def discretize_tertiles(values):
    edges = np.percentile(values, [100.0 / 3.0, 200.0 / 3.0])
    classes = np.digitize(values, edges, right=False).astype(np.int32)
    return classes, edges
```

iii. Per-session tertiles give approximately equal class frequencies (~1/3 each), which is confirmed in the verification output.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated at the right edges of the same 20 ms bins used for neural data, so the two are on the same time grid, though offset by half a bin (right edge vs bin center for neural).

ii.
```python
grid = t0s[:, None] + (np.arange(1, nbins + 1)[None, :]) * binsize
```

iii. The AI notes this matches the reference code's `get_behavior_per_interval` interpolation grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy` (with its timestamps `_ibl_leftCamera.times`), falling back to the right camera when the left is unavailable. When both cameras exist, the one covering more trials is used (with left preferred on ties).

ii.
```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sess_loader.load_motion_energy(views=[view])
        df = sess_loader.motion_energy[cam]
        cameras[cam] = (df['times'].to_numpy(), df['whiskerMotionEnergy'].to_numpy())
    except Exception:
        continue
```

iii. Follows the reference code's camera preference logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy trace is used as-is (no filtering or normalization). It is linearly interpolated onto the bin right edges (same grid as wheel speed) and then discretized into 3 per-session tertile bins.

ii.
```python
me_c, valid_c = bin_behavior(cameras[cam][0], cameras[cam][1], safe_t0)
# ...
me_cls, me_edges = discretize_tertiles(me_k)
```

iii. Same processing pipeline as wheel speed: interpolation at bin right edges, then tertile discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: per-session 33.3rd and 66.7th percentiles over all kept trials and time bins.

ii.
```python
me_cls, me_edges = discretize_tertiles(me_k)
```

iii. Consistent with the wheel speed discretization approach.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated at bin right edges of the same 20 ms bins.

ii.
```python
me_c, valid_c = bin_behavior(cameras[cam][0], cameras[cam][1], safe_t0)
```

iii. Same alignment approach as wheel speed, following the reference `get_behavior_per_interval`.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple layers of handling: (1) The trial mask from `load_trials_and_mask` drops trials with NaN in key fields. (2) `bin_behavior` marks trials as invalid if behavioral data doesn't cover the window. (3) Trials with non-finite `stimOn_times` are excluded. (4) Trials with zero spikes (recording gaps) are dropped. (5) Sessions with no whisker motion energy, no QC-passing units, or fewer than 2/5 usable trials are skipped entirely. (6) Probes with no spike sorting are skipped.

ii.
```python
keep = mask & finite_align & ws_valid & me_valid
# ...
nonempty = counts.sum(axis=(1, 2)) > 0
# ...
if n_units < MIN_NEURONS:
    return {'eid': eid, 'skip': ...}
```

iii. The AI documents handling of each edge case, including recording gaps and missing camera data.

## 10-a. What are the most time-consuming steps of the code?

i. Loading the spike sorting from disk dominates at 3.5-8.3 s per session. The AI reports total processing of 4-9 s per session with binning taking only 0.01-0.03 s.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
```

iii. The cost is dominated by file I/O for the large spike arrays.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `bin_spikes` iterates over trials to bin spikes one trial at a time. The `bin_behavior` function also loops per trial for validity checking. Both could potentially be fully vectorized by offsetting indices across trials.

ii.
```python
for k in range(ntrials):
    a, b = i0[k], i1[k]
    # ...
    bins = ((spike_times[a:b] - t0s[k]) / binsize).astype(np.int64)
```

```python
for k in range(ntrials):
    a, b = i_beg[k], i_end[k]
    if b <= a: continue
    # ...
```

iii. The per-trial loops are needed because each trial involves a different slice of the continuous data. The cost is negligible compared to I/O.

## 10-c. What processing does the code repeat multiple times?

i. The AI loads both left and right camera motion energy when both exist (to pick the one with better coverage), even though typically only one is used. `BrainRegions()` is instantiated inside `load_session_neural` for every session rather than being shared.

ii.
```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sess_loader.load_motion_energy(views=[view])
```

```python
beryl = BrainRegions().acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl')
```

iii. The repeated camera loading is for robustness; the BrainRegions instantiation is a minor inefficiency.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads and bins behavioral data for all trials (including those that will fail the mask), computing validity masks after the fact. It also loads both cameras when only one is needed. The timing information and detailed metadata per session is computed and stored but not used by the decoder.

ii.
```python
safe_t0 = np.where(finite_align, t0s_all, np.nanmedian(t0s_all[finite_align]))
ws, ws_valid = bin_behavior(wheel_times, wheel_speed, safe_t0)
```

iii. Binning all trials first and then filtering is simpler than pre-filtering, even if some computation is wasted.
