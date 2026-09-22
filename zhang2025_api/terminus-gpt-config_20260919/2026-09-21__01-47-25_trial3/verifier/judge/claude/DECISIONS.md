# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything scientific is read through `one.api.ONE` (two clients, both `mode='local'`): one pointed at the `Brainwidemap` release tables (core ephys, trials, wheel, camera timestamps) and one pointed at the `2025_Q3_IBL_et_al_BWM` update tables (camera `ROIMotionEnergy`, which only carries a usable revision in the Q3 table). The candidate session list is *not* obtained with `one.search`; instead the AI reads ONE's in-memory dataset index (`one._cache['datasets']`) and intersects the sets of eids whose `rel_path` strings contain `spikes.times`, `_ibl_wheel.timestamps.npy`, and a paired camera motion-energy/timestamps pair. Per session it then uses `one.list_datasets`, `one.load_dataset`, `one.load_object`, `one.list_collections`, `one.eid2ref` and `brainbox.io.one.SpikeSortingLoader`. Because the shipped release tables have stale/stripped revision components, the AI inspects the cache *directory names* (`one.eid2path(eid)/'alf'` glob for `#<revision>#`) and rewrites the `rel_path` of the trial-table record inside ONE's in-memory index before calling `one.load_dataset(...)`; spike sorting is pinned to `revision='2024-05-06'`. 445 candidate sessions were found and all 445 converted (0 failures).

ii.
```python
def make_ones():
    return (ONE(cache_dir=CACHE, tables_dir=BASE_TABLES, mode='local'),
            ONE(cache_dir=CACHE, tables_dir=UPDATE_TABLES, mode='local'))

def candidate_eids(base, update):
    """Core ephys sessions with wheel and at least one paired camera stream."""
    bd, ud = base._cache['datasets'], update._cache['datasets']
    bp, up = bd.rel_path.astype(str), ud.rel_path.astype(str)
    def es(d, p, text):
        return set(d[p.str.contains(text, regex=False)].index.get_level_values('eid'))
    core = es(bd, bp, 'spikes.times') & es(bd, bp, '_ibl_wheel.timestamps.npy')
    left = es(ud, up, 'leftCamera.ROIMotionEnergy.npy') & es(bd, bp, '_ibl_leftCamera.times.npy')
    right = es(ud, up, 'rightCamera.ROIMotionEnergy.npy') & es(bd, bp, '_ibl_rightCamera.times.npy')
    return sorted(core & (left | right), key=str), left, right
```

```python
    # Cache release updates may leave stale/stripped revisions in table records.
    # Inspect path metadata only, then perform the scientific read through ONE.
    alf = one.eid2path(eid) / 'alf'
    matches = sorted(alf.glob('#*#/_ibl_trials.table.pqt'))
    ...
    d.loc[mask, 'rel_path'] = f'alf/#{rev}#/_ibl_trials.table.pqt'
    return one.load_dataset(eid, '_ibl_trials.table.pqt', collection='alf',
                            revision=rev, check_hash=False)
```

iii. CONVERSION_NOTES Step 2/Step 5 (decision 11): "the combined table omits revision components from some `rel_path` records although files live under revisions … Loading was kept within ONE by using the release table with revision metadata or correcting stale paths only in ONE's in-memory index"; "never directly read scientific files." Step 4 justifies starting from the complete 459-session BWM release rather than the reference repo's 10-session example list, and reducing to 445 because 14 sessions lack a paired whisker stream required by the decoder outputs.

## 1-b. How are the data split into subjects?

i. The subject name is taken from ONE per session (`one.eid2ref(eid).subject`); no path parsing. At assembly the `subjects` vocabulary is the sorted set of unique names and `subject_idx` is each session's index into it. Result: 136 subjects over 445 sessions.

ii.
```python
    ref = base.eid2ref(eid)
    out = dict(eid=str(eid), subject=str(ref.subject), ...)
```
```python
    subjects = sorted({s['subject'] for s in sessions}); smap={x:i for i,x in enumerate(subjects)}
    ...
    'subject_idx':np.array([smap[s['subject']] for s in sessions],dtype=np.int32),
```

iii. Step 5 mapping table: "Unique sorted subject strings; session index mapping — ONE session table/API." Step 9 explains 136 vs the paper's 139: "3 subjects lost because all their sessions lack required motion energy."

## 1-c. How are the data split into sessions?

i. A session is the natural unit of the release: one eid per session. The eid list is sorted deterministically (`sorted(..., key=str)`), sessions are processed independently in a process pool, and results are re-sorted by eid before assembly, so session order in the pickle is deterministic.

ii.
```python
    return sorted(core & (left | right), key=str), left, right
```
```python
        sessions.sort(key=lambda x:x['eid'])
```

iii. Step 5 mapping table: "Session order deterministic by EID." No further splitting decision is needed because ONE already indexes by session.

## 1-d. How are the data split into trials?

i. The `_ibl_trials.table.pqt` aggregate has one row per trial, so the split is given by the data. Each retained row becomes one entry in `neural`/`input`/`output`; the original row index is preserved in `metadata['session_info'][i]['trial_indices']`.

ii.
```python
    trials = load_trial_table(base, eid)
    ...
    mask = trial_mask(trials)
    original_idx = np.flatnonzero(mask)
    align = trials.stimOn_times.to_numpy(float)[mask]
```

iii. Step 2 documents the 13-column trial table; no decision was needed beyond keeping the mapping back to raw trial indices for auditability ("No trial/session is dropped silently", Step 5 decision 10).

## 1-e. How are trials filtered based on quality controls?

i. Four stages. (1) `trial_mask` requires finite values for `choice`, `probabilityLeft`, `feedbackType`, `feedback_times`, `stimOn_times`, `firstMovement_times`; `choice ∈ {-1,+1}` (drops no-response trials); `probabilityLeft ∈ {0.2,0.5,0.8}`; and reaction time (`firstMovement_times − stimOn_times`) in [0.08, 2.00] s. (2) A behavioural-coverage filter: wheel and camera traces are interpolated with `left=nan, right=nan`, and any trial with a non-finite sample in its 2 s window is dropped. (3) Trials in which *no curated neuron* fired a single spike over the whole window are dropped (39 trials dataset-wide). (4) Sessions retaining fewer than two trials, or zero curated neurons, raise and are skipped. 190,239 trials survive out of ~293,662 raw.

ii.
```python
def trial_mask(trials):
    required = ['choice', 'probabilityLeft', 'feedbackType', 'feedback_times',
                'stimOn_times', 'firstMovement_times']
    ...
    m = np.ones(len(trials), dtype=bool)
    for c in required:
        m &= np.isfinite(trials[c].to_numpy(dtype=float))
    ...
    m &= np.isin(choice, [-1., 1.])
    m &= np.isin(prior, [.2, .5, .8])
    m &= (rt >= .08) & (rt <= 2.0)
    return m
```
```python
    complete = np.isfinite(wheel).all(1) & np.isfinite(whisk).all(1)
    ...
    if len(align) < 2:
        raise RuntimeError(f'fewer than two complete valid trials ({len(align)})')
    neural, regions, uuids, nspikes = load_binned_neural(base, eid, align)
    # Trials with no spikes from any curated unit contain no neural observation
    # and are rejected by the supplied validator. Remove them synchronously.
    neural_valid = np.any(neural != 0, axis=(1, 2))
```

iii. Step 3/Step 4: the mask reproduces the data paper's stated exclusions ("trials were excluded if one of the following trial events could not be detected: choice, probabilityLeft, feedbackType, feedback times, stimON times and firstMovement times … outside the range of 0.08–2.00 s") and the reference `load_trials_and_mask` defaults (`min_rt=0.08, max_rt=2, exclude_nochoice=True`, same `nan_exclude` list). 0.5-prior trials are deliberately kept "because prior=0.5 is a required decoder class". The zero-neural drop is justified in Step 9/10 purely as removing validator warnings ("39 trials with no spikes from any curated neuron … all other arrays were filtered synchronously").

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` from every `alf/probe*/pykilosort` collection of the session, loaded with `SpikeSortingLoader.load_spike_sorting(revision='2024-05-06')`. The merged cluster table (`merge_clusters`) supplies `label` (quality) and `acronym` (Allen location) used only for selection/labelling, plus `uuids` stored as provenance.

ii.
```python
        ssl = SpikeSortingLoader(eid=eid, pname=pname, one=base)
        spikes, clusters, channels = ssl.load_spike_sorting(revision='2024-05-06')
        merged = ssl.merge_clusters(spikes, clusters, channels).to_df()
        label = merged['label'].to_numpy() if 'label' in merged else np.zeros(len(merged))
        allen_acr = merged['acronym'].astype(str).to_numpy() ...
```

iii. Step 1 identifies `SpikeSortingLoader.load_spike_sorting` / `merge_clusters` as the reference loading/processing path; Step 10 check 4 states "both use ONE, `SpikeSortingLoader.load_spike_sorting`, and `merge_clusters`".

## 2-b. How is the `neural` data processed?

i. Spikes of the retained clusters are binned with `brainbox.singlecell.bin_spikes2D` into 100 non-overlapping 20 ms bins spanning [-0.5, +1.5] s about each trial's `stimOn_times`. All probes of a session are binned separately and then concatenated along the neuron axis, so a session is one pooled population. The stored values are **raw spike counts per 20 ms bin** (not firing rates, not smoothed, not z-scored), cast to float32; the array is transposed from `(trials, neurons, bins)` to a per-trial `(n_neurons, 100)` matrix.

ii.
```python
        binned, tscale = bin_spikes2D(np.asarray(spikes['times'])[selected_spikes],
                                    spike_clusters[selected_spikes], ids, align,
                                    pre_time=.5, post_time=1.5, bin_size=DT)
        # brainbox returns trials x clusters x bins.
        arrays.append(np.asarray(binned))
    ...
    x = np.concatenate(arrays, axis=1)
    if x.shape != (len(align), len(regions), N_BINS):
        raise RuntimeError(f'unexpected neural shape {x.shape}')
    return x.astype(np.float32), regions, uuids, spike_total
```

iii. Step 5 mapping table: "Merge probes; … histogram counts in 100 bins from -0.5 to +1.5 s around stimulus onset; shape `(n_neurons,100)`, integer counts — Preserve counts (not rates or z-scores) so no information is lost." Step 3: "Multiple probes in a session are combined because they are not independent sessions" (data-paper decoding methods). Step 5 decision 9: float32 because the supplied validator/trainer expects float32 while counts stay exact.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two per-cluster criteria, applied before binning: IBL quality `label >= 1` (the "well-isolated" units: amplitude > 50 µV, noise cutoff < 20 µV, refractory-period criterion), and an anatomical criterion — the Allen acronym is remapped to Beryl and the unit is dropped if the Beryl acronym is in `{'void','root','nan','None',''}`. Note this drops `root` as well as `void`. 62,761 units survive across the 445 sessions (mean 141/session, range 1–516), versus 75,708 `label>=1` units release-wide.

ii.
```python
BAD_REGIONS = {'void', 'root', 'nan', 'None', ''}
...
        acr = BrainRegions().acronym2acronym(allen_acr, mapping='Beryl').astype(str)
        good = (label >= 1) & ~np.isin(acr, list(BAD_REGIONS))
        ids = merged.index.to_numpy()[good]
        if not len(ids):
            continue
        spike_clusters = np.asarray(spikes['clusters'])
        selected_spikes = np.isin(spike_clusters, ids)
```

iii. Step 4 discrepancy table: the Zhang reference `prepare_data` caches *all* clusters and only stores `label>=1` as metadata, while the data paper analyses use well-isolated units; resolution — "Use `label >= 1` well-isolated units: this follows the data-paper curation explicitly required by the task, reduces noisy regressors, and makes full conversion tractable." Step 5 decision 3: "Exclude units with invalid/void/root atlas labels" as a gray-matter criterion; Step 10 records the earlier bug "Fine Allen white-matter labels retained: remapped to Beryl and excluded `root`/`void`, matching reference ontology processing."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams are already on one synchronised session clock, so alignment is subtraction of `stimOn_times`. `bin_spikes2D` is given the array of `stimOn_times` of the retained trials as `align_times` with `pre_time=0.5`, `post_time=1.5`; internally each spike's bin is `floor((t − (align − 0.5))/0.02)`, i.e. bins are half-open `[align−0.5+k·0.02, align−0.5+(k+1)·0.02)`. The same `align` vector is used for the wheel and camera interpolation, so every stream shares the grid.

ii.
```python
    align = trials.stimOn_times.to_numpy(float)[mask]
    ...
    wheel, wheel_raw = load_wheel_speed(base, eid, align)
    whisk, camera, whisk_raw = load_whisker(base, update, eid, align, left_set, right_set)
    ...
    neural, regions, uuids, nspikes = load_binned_neural(base, eid, align)
```

iii. Step 4: "Task requirement overrides target-specific alignments. Use the Zhang common stimulus-aligned 2 s/20 ms representation." Step 10 check 4: "`bin_spikes2D` boundary behavior was explicitly tested (left edge included, final +1.5 s edge excluded)." Step 5 decision 5: "Same edges are used for spikes, wheel, and whisker streams, preventing temporal drift."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial, window [-0.5, +1.5] s; `metadata['time_bin_size'] = 20.0` ms. Spikes are binned once directly at 20 ms — there is no rebinning of an intermediate resolution. The only resampling is on the behavioural side (wheel position → 1 kHz grid inside brainbox, then point-sampled at the 20 ms bin centres; camera motion energy point-sampled at the same centres).

ii.
```python
DT = 0.020
OFF_START, OFF_END = -0.5, 1.5
N_BINS = 100
BIN_CENTERS = (OFF_START + DT / 2 + np.arange(N_BINS) * DT).astype(np.float32)
```
```python
'time_bin_size':20.0,'temporal_alignment_event':'visual stimulus onset (stimOn_times)',
'off_start':OFF_START,'off_end':OFF_END,'neural_representation':'spike counts per 20 ms bin',
```

iii. Step 3: "Zhang stimulus-aligned window -0.5 to +1.5 s around `stimOn_times`" with `interval_len=2`, `binsize=.02`; the data paper's wheel decoder also uses 20 ms bins. Step 4 resolves the method paper's 50 ms choice window in favour of the reference code's common 20 ms/2 s representation.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` only, indirectly: the input is the fixed vector of the 100 bin centres relative to `stimOn_times`, i.e. −0.49 … +1.49 s in 0.02 s steps. The same vector is used for every trial and every session.

ii.
```python
BIN_CENTERS = (OFF_START + DT / 2 + np.arange(N_BINS) * DT).astype(np.float32)
```
```python
            ins.append(np.vstack((BIN_CENTERS, np.full(N_BINS,s['trial_number'][j],np.float32))))
```

iii. Step 5 mapping table: "Bin centers relative to `stimOn_times` → `input[0,:]`, constant float32 vector `[-0.49, ..., 1.49]` s for every trial (Zhang `create_intervals`); continuous time since stimulus onset."

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the constant bin-centre vector; it is a definition, not a measurement. It is emitted as a continuous (not binary) time-varying input, in seconds, float32.

ii.
```python
BIN_CENTERS = (OFF_START + DT / 2 + np.arange(N_BINS) * DT).astype(np.float32)
```
An internal assertion checks it is identical in every trial:
```python
            assert np.allclose(x[0],BIN_CENTERS)
```

iii. Step 5 planned sanity check: "Verify bin centers are identical across all trials and span -0.49 to +1.49 s"; Step 10 notes a test-only fix where the expected vector was changed "from cumulative `arange` to the converter's stable indexed formula".

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It *is* the neural binning grid: `bin_spikes2D` uses `pre_time=0.5/post_time=1.5/bin_size=0.02` about the same `align` vector, whose bin k covers `[−0.5+0.02k, −0.5+0.02(k+1))` with centre `−0.49+0.02k = BIN_CENTERS[k]`. Column k of `neural`, `input` and `output` therefore describes the same instant.

ii.
```python
        binned, tscale = bin_spikes2D(..., align, pre_time=.5, post_time=1.5, bin_size=DT)
```
```python
BIN_CENTERS = (OFF_START + DT / 2 + np.arange(N_BINS) * DT).astype(np.float32)
```

iii. Step 10 check 4: "Alignment/binning: matches Zhang stimulus onset [-0.5,+1.5] s and 20 ms spike bins. `bin_spikes2D` boundary behavior was explicitly tested (left edge included, final +1.5 s edge excluded)."

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table: the table has no block identifier, so a block boundary is inferred wherever `probabilityLeft` changes from the previous trial.

ii.
```python
def trial_number_in_block(prior):
    out = np.zeros(len(prior), dtype=np.float32)
    for i in range(1, len(prior)):
        out[i] = 0 if prior[i] != prior[i - 1] else out[i - 1] + 1
    return out
```

iii. Step 5 mapping table: "Trial order within contiguous `probabilityLeft` block → `input[1,:]`… Derived from trial table."

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based counter that resets to 0 at each change of `probabilityLeft`, computed on the **full, unfiltered** trial table and only then subset by the trial mask, so dropped trials still advance the animal's true position in the block. It is broadcast as a constant across the 100 bins of the trial and stored as float32. Observed range 0–98.

ii.
```python
    trials = load_trial_table(base, eid)
    block_num = trial_number_in_block(trials.probabilityLeft.to_numpy(float))
    mask = trial_mask(trials)
    ...
    block_num = block_num[mask]
```
```python
            ins.append(np.vstack((BIN_CENTERS, np.full(N_BINS,s['trial_number'][j],np.float32))))
```

iii. Step 5 mapping table: "Zero-based counter reset to 0 whenever prior changes; broadcast across 100 bins … computed before trial filtering so skipped trials do not renumber the experimental block." Step 9 sanity: range "[0,98] — Sensible (nominal blocks ~90 trials)".

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is +1, −1 or 0. Trials with 0 (no response) are already dropped by `trial_mask`. The AI maps raw `choice == +1 → 1` and `choice == −1 → 0`, and labels class 0 "left" and class 1 "right" in `output_values`, on the stated assumption that raw −1 means a leftward choice.

ii.
```python
    m &= np.isin(choice, [-1., 1.])
```
```python
    choice = (choice_raw == 1).astype(np.uint8)
```
```python
          'output_values':[['left','right'], ...]
```

iii. CONVERSION_NOTES Step 4: "Choice mapping follows the task, not raw IBL sign semantics: raw `choice=-1` (left) → 0 and `choice=+1` (right) → 1." Step 5 mapping table: "Raw -1 (left)→0, +1 (right)→1; broadcast across bins — Task-specified semantics. Raw 0/no-choice excluded."

## 5-b. What processing is involved in computing `output` *Choice*?

i. None beyond the ±1 → {0,1} recoding above, cast to uint8 and broadcast as a constant over all 100 bins so the per-trial variable is time-varying-shaped like the rest of the outputs.

ii.
```python
            outs.append(np.vstack((np.full(N_BINS,s['choice'][j],np.uint8),
                                   np.full(N_BINS,s['prior'][j],np.uint8),wc[j],mc[j])))
```

iii. Step 5 decision 8: "Broadcast choice, prior, and trial number across all 100 bins. The supplied decoder flattens trial timepoints; broadcasting makes per-trial targets dimensionally consistent with dynamic outputs and avoids missing targets."

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, restricted to exactly {0.2, 0.5, 0.8} by the trial mask and recoded 0.2→0, 0.5→1, 0.8→2.

ii.
```python
    m &= np.isin(prior, [.2, .5, .8])
```
```python
    prior = np.select([prior_raw == .2, prior_raw == .5, prior_raw == .8], [0, 1, 2]).astype(np.uint8)
```

iii. Step 4/Step 5: mapping given by the Decoder Task spec; "Retain unbiased block" (0.5) "because prior=0.5 is a required decoder class, even though reference code can optionally exclude the initial unbiased block."

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. None beyond the recoding; uint8, broadcast constant over the 100 bins. Resulting distribution 0.418 / 0.140 / 0.442 (matching the expectation that the unbiased 0.5 block is only the first ~90 trials of each session).

ii.
```python
            outs.append(np.vstack((np.full(N_BINS,s['choice'][j],np.uint8),
                                   np.full(N_BINS,s['prior'][j],np.uint8),wc[j],mc[j])))
```

iii. Step 9 consistency table: "Prior distribution … approximately [0.418,0.140,0.442] — Yes" against the expectation that 0.5 blocks are smaller.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The ALF `wheel` object loaded through ONE — `_ibl_wheel.timestamps` and `_ibl_wheel.position` — from which a filtered velocity is computed with the brainbox wheel helpers; speed is its absolute value (rad/s).

ii.
```python
def load_wheel_speed(base, eid, align):
    w = base.load_object(eid, 'wheel', collection='alf')
    ts = np.asarray(w['timestamps'], float)
    pos = np.asarray(w['position'], float)
    # Reference brainbox processing: 1 kHz interpolation then filtered derivative.
    ipos, its = wheellib.interpolate_position(ts, pos, freq=1000)
    vel, _ = wheellib.velocity_filtered(ipos, fs=1000)
    return interp_trials(its, np.abs(vel), align), (ts, pos, its, vel)
```

iii. Step 5 mapping table: "Brainbox interpolation to uniform samples, low-pass filtered velocity, absolute value for speed … Speed is magnitude, not signed velocity", citing `brainbox.behavior.wheel.interpolate_position` / `velocity_filtered` and the Zhang wheel loader.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps. (1) The irregularly sampled wheel position is linearly interpolated onto a uniform 1 kHz grid (`interpolate_position(freq=1000)`), then differentiated with a 20 Hz, 8th-order Butterworth low-pass (`velocity_filtered(fs=1000)`, brainbox defaults — identical to what `SessionLoader.load_wheel` does internally); the absolute value gives speed. (2) The speed trace is **point-sampled by linear interpolation at the 100 bin centres** of each trial, with `left=nan, right=nan` so that trials not fully covered by the wheel stream are flagged and dropped; non-finite and duplicate timestamps are removed first. (3) The trace is then discretized (see 7-c).

ii.
```python
def interp_trials(times, values, align):
    """Linear interpolation at common bin centers; no extrapolation."""
    ...
    ok = np.isfinite(times) & np.isfinite(values)
    times, values = times[ok], values[ok]
    ...
    keep = np.r_[True, np.diff(times) > 0]
    times, values = times[keep], values[keep]
    query = align[:, None] + BIN_CENTERS[None, :]
    out = np.interp(query.ravel(), times, values, left=np.nan, right=np.nan)
    return out.reshape(len(align), N_BINS).astype(np.float32)
```

iii. Step 5 decision 6: "Linear interpolation is used only to bridge the high-rate wheel position onto a uniform grid before reference filtering; no extrapolation beyond available stream support. Missing/empty behavior bins invalidate that trial rather than being imputed." (Note: the Step 5 mapping table text still says "average within each 20 ms trial bin", which the implemented code and the final metadata string — "absolute speed sampled at bin centers" — supersede.)

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into three classes by **global** tertiles: after every session is processed, all retained wheel-speed bin values from all 445 sessions are pooled, the 1/3 and 2/3 quantiles are taken once, and the same two thresholds ([0.01477, 0.39721] rad/s) are applied to every session with `np.digitize`. The thresholds are stored in metadata; a guard rejects degenerate (non-increasing) tertiles.

ii.
```python
    wheel_vals = np.concatenate([s['wheel'].ravel() for s in sessions])
    ...
    thresholds = {'wheel': np.quantile(wheel_vals, [1/3, 2/3]).astype(float),
                  'whisker': np.quantile(whisk_vals, [1/3, 2/3]).astype(float)}
    for k,v in thresholds.items():
        if not np.isfinite(v).all() or v[0] >= v[1]:
            raise RuntimeError(f'invalid {k} tertiles {v}')
    ...
        wc=np.digitize(s['wheel'],thresholds['wheel']).astype(np.uint8)
```

iii. Step 5 decision 7: "Compute 1/3 and 2/3 quantiles over all finite time-bin values from retained trials/sessions after processing. Apply fixed global thresholds to every session for comparable class meanings. Store physical-unit thresholds and class definitions in metadata."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is evaluated at exactly the same 100 bin centres, measured from the same `stimOn_times` vector used to bin the spikes, so the two share a time axis bin for bin. No lag or shift is introduced.

ii.
```python
    query = align[:, None] + BIN_CENTERS[None, :]
    out = np.interp(query.ravel(), times, values, left=np.nan, right=np.nan)
```

iii. Step 5 decision 5: "Same edges are used for spikes, wheel, and whisker streams, preventing temporal drift." Step 7 plot review: "Common stimulus-aligned traces cover the entire [-0.5,+1.5] s interval without NaNs or edge truncation" (the `--show-processing` figure overlays the raw filtered wheel trace and the 20 ms samples).

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The precomputed side-camera ROI motion energy `<side>Camera.ROIMotionEnergy.npy` (read from the 2025 Q3 release table, which retains its revision) paired with `_ibl_<side>Camera.times.npy` from the base table. The left camera is preferred and the right used only when left is absent or fails to load; 432 sessions use left, 13 right. Lengths of the two arrays are checked for equality.

ii.
```python
def load_whisker(base, update, eid, align, left_set, right_set):
    """Load preferred left camera, falling back to right on absence/load failure."""
    ...
    for camera, available in [('left', left_set), ('right', right_set)]:
        if uid not in available:
            continue
        try:
            me = load_motion_energy(update, eid, camera)
            times = base.load_dataset(eid, f'_ibl_{camera}Camera.times.npy', collection='alf')
            if len(me) != len(times):
                raise RuntimeError(f'{camera} motion/timestamp length mismatch ...')
            return interp_trials(times, me, align), camera, (times, me)
```

iii. Step 4: "Reference loader prefers left and falls back to right … Prefer left to match code; use right only when left is unavailable, with matched camera timestamps." Step 3: motion energy "is mean absolute pixel difference between adjacent frames within a DLC-defined whisker-pad ROI … It is already computed."

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is — no filtering, normalisation or per-session scaling. It is linearly interpolated at the 100 bin centres of each trial with `left=nan, right=nan` (same `interp_trials` helper as the wheel), so trials whose window is not covered by the camera are dropped; then discretized (8-c).

ii.
```python
            return interp_trials(times, me, align), camera, (times, me)
```
```python
    complete = np.isfinite(wheel).all(1) & np.isfinite(whisk).all(1)
```

iii. Step 3: "the DLC likelihood ≥0.9 rule applies to landmark estimates used in analyses, not to postcomputed motion-energy samples", so no additional curation was applied. (As with the wheel, the Step 5 mapping table's "average samples in each 20 ms bin" is superseded by the implemented bin-centre sampling and the metadata string "sampled at bin centers".)

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: **global** tertiles computed once over the pooled motion-energy bin values of all 445 sessions (thresholds [2.7814, 7.8115] in raw motion-energy units), applied with `np.digitize` to every session, including the 13 right-camera sessions whose units/frame rate differ from the left camera.

ii.
```python
    whisk_vals = np.concatenate([s['whisker'].ravel() for s in sessions])
    thresholds = {..., 'whisker': np.quantile(whisk_vals, [1/3, 2/3]).astype(float)}
    ...
        mc=np.digitize(s['whisker'],thresholds['whisker']).astype(np.uint8)
```

iii. Step 4/Step 5 decision 7 (same as the wheel): "Apply fixed global thresholds to every session for comparable class meanings. Store physical-unit thresholds and class definitions in metadata. If tied quantiles occur, investigate rather than force arbitrary classes." Step 9 reports the resulting global distribution as exactly one third per class.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Sampled at the same 100 bin centres relative to the same `stimOn_times`, on the shared IBL session clock, so it is bin-for-bin aligned with the neural matrix and the other outputs.

ii.
```python
    query = align[:, None] + BIN_CENTERS[None, :]
    out = np.interp(query.ravel(), times, values, left=np.nan, right=np.nan)
```

iii. Same as 7-d: Step 5 decision 5 (one common grid for all streams); the `--show-processing` plots overlay the raw camera samples against the binned trace around t = 0.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing/awkward data are dropped and logged rather than imputed. Per session: stale or missing revision metadata for the trial table is repaired in ONE's index (and the session fails loudly if no cached table is found); the motion-energy loader tries the left camera and falls back to right "on absence/load failure"; a motion-energy/timestamps length mismatch raises; non-finite and duplicate behavioural timestamps are removed before interpolation; a stream with < 2 samples yields an all-NaN trace. Per trial: any non-finite required trial field, or any NaN in the interpolated 2 s wheel/camera window (i.e. incomplete coverage), or an all-zero neural window drops the trial. Per session: < 2 usable trials or 0 curated neurons raises, the exception is caught in the pool loop, the session is skipped and counted in a `FAILURES` summary. The full run reported 0 failures out of 445 candidates.

ii.
```python
    if len(times) < 2:
        return np.full((len(align), N_BINS), np.nan, np.float32)
```
```python
    if len(align) < 2:
        raise RuntimeError(f'fewer than two complete valid trials ({len(align)})')
```
```python
                try: sessions.append(fut.result())
                except Exception as exc:
                    failures.append((eid,type(exc).__name__,str(exc)))
                    print(f'SKIP {eid} {type(exc).__name__}: {exc}',flush=True)
    ...
    print('FAILURES',len(failures),Counter(x[1] for x in failures),flush=True)
```

iii. Step 5 decision 10: "Exclude sessions with <2 valid trials or zero curated neurons. Log every reason and count. No trial/session is dropped silently." Step 9: "Revision review recovered six initially skipped sessions: two stale trial-table records and four left-camera records requiring right-camera fallback."

## 10-a. What are the most time-consuming steps of the code?

i. Loading the spike sorting (tens of millions of spike times/clusters per session) and binning it; the AI measured ~4.2 s of processing per session serially and identified spike I/O plus `bin_spikes2D` as the bottleneck. This was mitigated with an 8-process pool, bringing the full 445-session run to 371 s.

ii.
```python
        workers=min(8, os.cpu_count() or 1)
        print(f'PARALLEL workers={workers}',flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs={ex.submit(_process_worker,str(eid)):str(eid) for eid in eids}
```
```python
    print(f"SESSION {eid} subject={out['subject']} trials={len(trials)}->{len(align)} "
          f"neurons={neural.shape[1]} camera={camera} spikes={nspikes} time={time.time()-t0:.2f}s", flush=True)
```

iii. Step 6: "Largest cost is loading tens of millions of spike events per session and brainbox binning"; Step 7 run-time table: serial 4.2 s/session ⇒ ~31 min for 445 sessions, parallel estimate 5–8 min, actual 6.18 min.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain. (1) `trial_number_in_block` is an explicit Python `for` loop over all trials of a session — it is trivially vectorizable (`np.arange(n) - np.maximum.accumulate(np.where(change, np.arange(n), 0))`). (2) The per-trial assembly loop in `build_output` allocates two small `np.vstack`/`np.full` arrays per trial for 190,239 trials; the time and trial-number rows are identical or constant and could be built in bulk (or broadcast once). (3) `np.isin(spike_clusters, ids)` over ~50 M spikes per probe does a sort/search where a boolean lookup table indexed by cluster id would be O(n); and `bin_spikes2D` itself loops over trials internally. None of these dominate, so the impact is small. The trial-level interpolation *was* vectorized (a single `np.interp` over all trials' query points at once), which is a genuine improvement over a per-trial loop.

ii.
```python
def trial_number_in_block(prior):
    out = np.zeros(len(prior), dtype=np.float32)
    for i in range(1, len(prior)):
        out[i] = 0 if prior[i] != prior[i - 1] else out[i - 1] + 1
    return out
```
```python
        for j in range(s['n_trials_valid']):
            ns.append(s['neural'][j])
            ins.append(np.vstack((BIN_CENTERS, np.full(N_BINS,s['trial_number'][j],np.float32))))
            outs.append(np.vstack((np.full(N_BINS,s['choice'][j],np.uint8), ...)))
```
```python
        selected_spikes = np.isin(spike_clusters, ids)
```

iii. Step 6: "Code speedups added: Vectorized trial masks/interpolation, compact uint8 neural arrays, and one-time global quantile calculation." The residual loops above are not discussed in CONVERSION_NOTES; the AI's stated position is that after parallelisation the run comfortably met the 15-minute budget.

## 10-c. What processing does the code repeat multiple times?

i. Two substantive repetitions, neither identified in the notes. (1) `_process_worker` — the function submitted once per session to the process pool — rebuilds **both** ONE clients and recomputes `candidate_eids` (which scans the full `Brainwidemap` dataset index of ~76,563 records with several `str.contains` passes) on *every* one of the 445 tasks, instead of doing it once per worker via a pool `initializer`. (2) `BrainRegions()` — which loads the Allen/Beryl ontology tables — is constructed inside the per-probe loop in `load_binned_neural`, so once per probe (~700 times) rather than once per process. Smaller repetitions: `base.to_eid(eid)` / `one.list_datasets` are called again inside the per-dataset loaders, and the whole-session wheel trace is re-interpolated to 1 kHz even though only trial windows are used.

ii.
```python
def _process_worker(eid_str):
    """Independent process worker; ONE instances are intentionally not pickled."""
    base, update = make_ones()
    eids, left, right = candidate_eids(base, update)
    return process_session(base, update, base.to_eid(eid_str), left, right, False)[0]
```
```python
        acr = BrainRegions().acronym2acronym(allen_acr, mapping='Beryl').astype(str)
```

iii. CONVERSION_NOTES does not mention these; Step 6 only claims "one-time global quantile calculation" as a de-duplication. The implicit justification for re-creating ONE in the worker is given in the docstring: "ONE instances are intentionally not pickled" (they cannot cross a fork) — correct, but a `ProcessPoolExecutor(initializer=...)` would have paid that cost once per worker instead of once per session.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Modest amounts. Per-neuron `uuids` and per-trial `trial_indices` are collected for all 445 sessions and pickled inside `metadata['session_info']`, though the decoder never reads them; `spike_total` counts every spike of every probe purely for a log line; `velocity_filtered` computes and discards the acceleration; `interpolate_position` builds a 1 kHz position grid for the entire (hour-long) session when only 2 s windows around retained trials are sampled; `load_trial_table`/`load_motion_energy` call `one.list_datasets` only to assert a row count; and `make_plot`'s raw streams are carried out of `process_session` in sample mode. The neural array is stored as float32 spike counts (max observed 25), so 4 bytes/bin are used where 1–2 would suffice — an 11.28 GB pickle — but this was a deliberate choice to satisfy the supplied validator.

ii.
```python
        if 'uuids' in merged:
            uuids.extend(merged.loc[good, 'uuids'].astype(str).tolist())
        ...
        spike_total += len(spikes['times'])
```
```python
    vel, _ = wheellib.velocity_filtered(ipos, fs=1000)
```
```python
        session_info.append({k:s[k] for k in ['eid','subject','camera','n_trials_raw','n_trials_valid','n_zero_neural','trial_indices','uuids']})
```

iii. Step 5 decision 12 justifies the extra metadata as provenance: "plus EIDs, subjects, trial indices, cameras used, quantile thresholds, filtering counts, and source release"; Step 5 decision 9 justifies float32: "the supplied validator/trainer expects float32 (integer counts remain exact)" — an earlier uint8 version produced a validator warning (Step 7).
