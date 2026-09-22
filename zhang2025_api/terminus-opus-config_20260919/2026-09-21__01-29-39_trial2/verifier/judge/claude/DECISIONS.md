# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every data array is read through the ONE API with the brainbox loaders; no file under `/app/data` is opened
directly. The set of sessions to process is taken from the reference repository's release freeze file
`/app/code/code_zhang2025/data/bwm_release.csv` (the same file the reference `src/0_data_caching.py` uses),
which lists 459 `eid`s together with their `subject` and `lab`; `one.search` is not used. ONE is instantiated
per worker process in `mode='remote'` against the staged cache (the AI determined that `mode='local'` cannot
resolve the dataset revisions actually present on disk, e.g. `alf/#2025-03-03#/`, whereas the cached REST
responses in `one_cache/.rest` let the remote path resolve them fully offline). From an `eid`, `SessionLoader`
supplies the trials table, the wheel and the camera motion energy, `one.eid2pid` gives the probe insertions and
`SpikeSortingLoader` gives the spikes/clusters/channels of each probe. All 459 sessions are processed, 440 are
kept (187,547 trials, 62,701 neurons, 136 subjects).

ii.
```python
def get_one():
    """ONE instance backed by the staged local cache (works fully offline)."""
    from one.api import ONE
    return ONE(base_url=ONE_BASE_URL, mode='remote')
```
```python
bwm = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
eid2subject = bwm.groupby('eid').subject.first().to_dict()
eid2lab = bwm.groupby('eid').lab.first().to_dict()
eids = list(bwm.eid.unique())
```
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
...
pids, pnames = one.eid2pid(eid)
for pid, pname in zip(pids, pnames):
    ssl = SpikeSortingLoader(pid=str(pid), one=one, eid=eid, pname=pname)
    spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. From CONVERSION_NOTES.md Step 0/Step 1: the `brainwidemap` package is not installed, so the BWM helpers
(`bwm_query`, `load_good_units`, `load_trials_and_mask`) were re-implemented from the copies inside the
reference repo, and the repo's own `bwm_release.csv` freeze list is used to enumerate sessions exactly as
`0_data_caching.py` does. `mode='remote'` is documented as required because the staged parquet tables list
pre-revision paths while the files on disk are under dated revision folders; it is served entirely from cached
REST responses so no network access occurs. Each worker builds its own `ONE` object because the client is not
fork-safe.

## 1-b. How are the data split into subjects?

i. Subject identity is not derived from paths; it is read from the `subject` column of `bwm_release.csv`, keyed
by `eid`. After conversion, `subjects` is the sorted list of unique subject names over the *kept* sessions and
`subject_idx` is the index of each session's subject into that list. 136 subjects survive (3 of the release's
139 lose their only session).

ii.
```python
eid2subject = bwm.groupby('eid').subject.first().to_dict()
...
subjects = sorted({eid2subject[r['eid']] for r in ok})
subj_index = {s: i for i, s in enumerate(subjects)}
...
'subject_idx': np.array([subj_index[eid2subject[r['eid']]] for r in ok], dtype=np.int64),
```

iii. CONVERSION_NOTES.md Step 5 maps "`bwm_release.csv` `subject` -> `subjects`, `subject_idx`" and Step 4
verifies that the file lists exactly the 139 mice / 459 sessions / 699 insertions the data paper reports, so the
subject labels are taken as authoritative and need no derivation. Step 9 documents 136 subjects in the output
and explains the loss of 3.

## 1-c. How are the data split into sessions?

i. No splitting is done: the session (`eid`) is the natural unit of the release, one row group per `eid` in
`bwm_release.csv`, and the conversion iterates one `eid` at a time, `convert_session(eid)` returning one entry
per session. Sessions are processed in parallel (12 workers) and then sorted by `eid` so the output order is
deterministic.

ii.
```python
eids = list(bwm.eid.unique())
...
with mp.Pool(args.n_workers) as pool:
    for i, r in enumerate(pool.imap_unordered(_worker, [(e, False) for e in eids])):
        results.append(r)
...
ok.sort(key=lambda r: r['eid'])
```

iii. Nothing is documented as a decision here beyond following the reference loop in
`src/0_data_caching.py`, which also iterates over `include_eids` one session at a time; the AI only adds
session-level parallelism, noted in Step 6 as the place where the real cost (spike-sorting I/O) lies.

## 1-d. How are the data split into trials?

i. The trials table returned by `SessionLoader.load_trials()` has one row per trial, so the split is given by
the data. A trial is then defined as the fixed 2 s window `stimOn_times + (-0.5, +1.5) s`, following the
reference `params`, rather than by the trial's own start/end times.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)          # seconds relative to the alignment event
BINSIZE = 0.02                     # seconds
NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
...
align_all = trials[ALIGN_TIME].to_numpy()
...
align = align_all[idx_trials]
```

iii. Step 5 "Trial definition (from reference `src/0_data_caching.py`)" records the alignment event, window and
bin size verbatim from the reference `params` dict, cross-checked in Step 3 against the method paper
("Recordings are split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps").

## 1-e. How are trials filtered based on quality controls?

i. Five filters, applied in sequence.
(1) The reference `load_trials_and_mask` query is reproduced exactly: reaction time
`firstMovement_times - stimOn_times` must lie in [0.08, 2.0] s; trial length `feedback_times - goCue_times`
must be <= 10 s; no NaN in `stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times,
feedbackType`; `choice != 0` (no-response trials dropped).
(2) The whole 2 s window must lie inside the spike-sorted recording of *every* probe
(`max(probe t_min) .. min(probe t_max)`) — added after the AI found a session whose ephys ended before the
behaviour did.
(3) The wheel and the whisker traces must span the window (start/end within one bin) and contain no NaN — the
`get_behavior_per_interval` "target data starts too late / ends too early" checks.
(4) Trials in which no neuron in the population fired at all in the 2 s window are dropped (13 trials in the
whole dataset).
(5) Trials whose `probabilityLeft` is not in {0.2, 0.5, 0.8} are dropped (none found).
Sessions are dropped entirely if fewer than 2 trials survive any of these stages, if fewer than 5 well-isolated
neurons remain, or if neither camera has motion energy. Overall 195,781 trials pass the reference mask and
187,547 survive all filters.

ii.
```python
def compute_trials_mask(trials, min_rt=MIN_RT, max_rt=MAX_RT, max_trial_len=MAX_TRIAL_LEN,
                        nan_exclude=NAN_EXCLUDE, exclude_nochoice=True):
    query = f'(firstMovement_times - stimOn_times < {min_rt})'
    query += f' | (firstMovement_times - stimOn_times > {max_rt})'
    if max_trial_len is not None:
        query += f' | (feedback_times - goCue_times > {max_trial_len})'
    for event in nan_exclude:
        query += f' | {event}.isnull()'
    if exclude_nochoice:
        query += ' | (choice == 0)'
    return ~trials.eval(query)
```
```python
rec_t0, rec_t1 = max(probe_t0), min(probe_t1)
covered = (align_all + TIME_WINDOW[0] >= rec_t0) & (align_all + TIME_WINDOW[1] <= rec_t1)
mask = mask & covered
...
wheel_binned, wheel_valid = bin_behavior(wheel_times, wheel_speed_raw, align)
me_binned, me_valid = bin_behavior(me_times, me_vals, align)
good = wheel_valid & me_valid
...
nonzero = binned_spikes.sum(axis=(1, 2)) > 0
```
```python
if np.abs(begs[k] - tt[0]) > binsize or np.abs(ends[k] - tt[-1]) > binsize:
    valid[k] = False      # target data starts too late / ends too early
```

iii. Step 4 resolves the discrepancy between the data paper (NaN + 0.08–2 s RT rules) and the reference code
(which additionally uses `max_trial_len=10` and `exclude_nochoice=True`) in favour of the reference code, "since
the reference code is the more specific description of this pipeline". The behaviour-coverage rule is described
as copying `get_behavior_per_interval`'s own validity test. The ephys-coverage rule, the >= 5 neuron rule and the
zero-spike rule are documented in Step 10 as fixes for the 16 "all neural data is zero" warnings the first full
run produced: investigation (`cache/zero_check.py`) showed session `8c2f7f4d`'s spike sorting ends at 1779 s
while behaviour continues past 1800 s, and the remaining cases were genuinely silent windows in low-yield
sessions. The trajectory (step 127) shows the AI weighing "accept and document" against "drop"; it chose to drop
because such trials "carry no neural information", while noting it "would bias the dataset slightly".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes['times']` and `spikes['clusters']` of every probe of the session, loaded with
`SpikeSortingLoader.load_spike_sorting()`. The cluster table produced by `merge_clusters` supplies `label`
(IBL unit QC) and `acronym` (Allen region of the peak channel), which are used only for curation and for
`brain_region_idx`, not for the activity itself.

ii.
```python
ssl = SpikeSortingLoader(pid=str(pid), one=one, eid=eid, pname=pname)
spikes, clusters, channels = ssl.load_spike_sorting()
clu = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
probe_t0.append(np.nanmin(spikes['times']))
probe_t1.append(np.nanmax(spikes['times']))
spike_times_l.append(spikes['times'])
spike_clu_l.append(spikes['clusters'] + offset)
acronyms_l.append(clu['acronym'].to_numpy())
labels_l.append(clu['label'].to_numpy())
```

iii. Step 1 documents `load_spiking_data` in the reference as `SpikeSortingLoader(pid).load_spike_sorting()` +
`merge_clusters`, and Step 5 maps "`spikes.times`, `spikes.clusters` (all probes merged) -> `neural`". The AI
also notes that the reference's extra call to `raw_electrophysiology(...).fs` (used only to record the AP
sampling rate) is deliberately skipped because it would require streaming raw ephys.

## 2-b. How is the `neural` data processed?

i. Spikes of all probes of a session are pooled into one population (cluster ids of the second probe offset by
the number of clusters of the first, as in the reference `merge_probes`), the merged spike train is re-sorted by
time, non-surviving clusters are removed and the survivors renumbered 0..N-1, and spikes are counted into 100
non-overlapping 20 ms bins covering `[stimOn-0.5 s, stimOn+1.5 s)`. The stored value is the raw **spike count**
per neuron and bin in `float32`; it is *not* divided by the bin width and is not smoothed or z-scored (the AI
notes the decoder does its own PCA).

ii.
```python
spike_clu_l.append(spikes['clusters'] + offset)
...
offset += int(clu.index.max()) + 1
spike_times = np.concatenate(spike_times_l)
spike_clusters = np.concatenate(spike_clu_l)
srt = np.argsort(spike_times, kind='stable')
spike_times = spike_times[srt]
spike_clusters = spike_clusters[srt]
...
remap = -np.ones(offset, dtype=np.int64)
remap[keep_ids] = np.arange(len(keep_ids))
sel = np.isin(spike_clusters, keep_ids)
sp_t = spike_times[sel]
sp_c = remap[spike_clusters[sel]]
```
```python
i0 = np.searchsorted(spike_times, begs, side='left')
i1 = np.searchsorted(spike_times, ends, side='left')
for k in range(n_trials):
    s, e = i0[k], i1[k]
    if e <= s:
        continue
    b = ((spike_times[s:e] - begs[k]) / binsize).astype(np.int64)
    np.clip(b, 0, n_bins - 1, out=b)
    idx = spike_clusters[s:e].astype(np.int64) * n_bins + b
    cnt = np.bincount(idx, minlength=n_clusters * n_bins)
    out[k] = cnt.reshape(n_clusters, n_bins)
```

iii. Step 5 decision 6: "Sessions with two probes are merged (paper: 'neurons in the same session and region
were combined across probes'), using the reference `merge_probes`". Step 3 records "spike counts per (neuron,
bin), no smoothing". Step 6 documents that the binning is a vectorised equivalent of the reference
`bin_spiking_data`/`bincount2D`: the reference bins with a per-trial multiprocessing pool dominated by pickling
overhead, whereas two `searchsorted` calls plus one `bincount` per trial give ~0.01 s per session. Step 10
Check 3 marks the binning "YES (verified against raw spike times)", and Check 2 reports 8 random
(trial, neuron, bin) spot checks against counts recomputed directly from raw `spikes.times`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two unit-level filters, applied together before any spike is binned: the IBL unit-QC `label` must be 1 (the
unit passes all three RIGOR single-unit metrics: amplitude > 50 µV, noise cut-off < 20 µV, refractory-period
violation), and the Beryl acronym of the unit must be neither `void` nor `root`. A session is dropped if fewer
than 5 such units remain. Across the release the `label == 1` cut alone reproduces the paper's 75,708 of
621,733 units; after also removing `void`/`root` units and the dropped sessions, 62,701 neurons are kept
(mean 142.5 per session).

ii.
```python
QC_LABEL = 1.0                     # IBL unit QC: 1 == passes all three RIGOR single-unit metrics
EXCLUDE_REGIONS = ('void', 'root')  # not grey matter / outside the brain
MIN_NEURONS = 5      # BWM paper: analyses require >= 5 well-isolated neurons
...
beryl = br.acronym2acronym(acronyms, mapping='Beryl')
keep = (labels >= QC_LABEL) & (~np.isin(beryl, EXCLUDE_REGIONS))
keep_ids = np.where(keep)[0]
if len(keep_ids) < MIN_NEURONS:
    return dict(eid=eid, skipped='too few good neurons')
```

iii. Step 4 records the discrepancy explicitly: the reference `prepare_data` calls `load_spiking_data` **without**
a `qc` argument, so it bins all clusters and only records `good_clusters = label >= 1` in metadata, while the BWM
data paper states analyses use only well-isolated neurons. The AI resolves it in favour of `label == 1`,
arguing the data paper is explicit and that "8x more noisy/MUA units mostly add noise and huge memory cost".
Dropping `void`/`root` is justified as the paper's grey-matter restriction; the reference repo does the same in
`utils/data_loader_utils.py` (`[roi for roi in ... if roi not in ['root', 'void']]`). The `>= 5 neurons`
threshold was raised from 2 in Step 10 to match the BWM paper's ">= 5 well-isolated neurons" criterion and to
remove all-zero-trial warnings. Step 4 reports the strong sanity check that the AI's loading reproduces
621,733 total and 75,708 `label == 1` units exactly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to `trials.stimOn_times` (the instruction's "temporally align based on stimulus onset"),
with the window running from -0.5 s to +1.5 s around it. Alignment is a pure subtraction: spike times, trial
event times, wheel timestamps and camera frame times are all already on the same session clock, so bin `j` of a
trial is `[stimOn - 0.5 + 0.02j, stimOn - 0.5 + 0.02(j+1))`, obtained by `searchsorted` on the sorted spike
times and `floor((t - beg)/binsize)`.

ii.
```python
begs = align_times + t_start
ends = align_times + t_end
i0 = np.searchsorted(spike_times, begs, side='left')
i1 = np.searchsorted(spike_times, ends, side='left')
...
b = ((spike_times[s:e] - begs[k]) / binsize).astype(np.int64)
```

iii. Step 5 and the metadata record `temporal_alignment_event = 'stimulus onset (trials.stimOn_times)'`,
`off_start = -0.5`, `off_end = 1.5`, taken from the reference `params` and confirmed by the method paper ("For
choice, we align trials to the stimulus onset, considering neural activity from 0.5 s before to 1.5 s
post-onset"). Step 10 Check 5 states there is no off-by-one: bins are half-open `[t, t+20 ms)` starting exactly
at `stimOn - 0.5 s` and the 100th bin ends exactly at `stimOn + 1.5 s`, verified by the spot checks; the
`--show-processing` figures overlay the raw raster and the binned matrix with stimulus onset marked.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins (`time_bin_size = 20.0` ms in the metadata), 100 bins per trial, identical for every trial and
session. No rebinning, resampling or smoothing of the neural data is applied: spikes are counted once, directly
onto the final grid. (The behavioural traces are resampled onto the same 100-bin grid — see 7-b/8-b.)

ii.
```python
BINSIZE = 0.02                     # seconds
NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
...
'time_bin_size': BINSIZE * 1000.,
'n_timepoints': NBINS,
'neural_data_type': 'spike counts per 20 ms bin (not normalised)',
```

iii. Step 1 copies the reference `params = {'interval_len': 2, 'binsize': 0.02, ..., 'time_window': (-.5, 1.5)}`
and Step 3 corroborates it with the method paper's "each divided into 20-ms bins, producing T = 100 time steps".
The verification log confirms T = 100 for every trial of all 440 sessions.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from a raw variable at all: it is defined by the analysis window around
`trials.stimOn_times`. The value in bin `j` is the centre of that bin relative to stimulus onset, so the same
vector (-0.49, -0.47, ..., 1.49) s is used for every trial of every session.

ii.
```python
bin_centres = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)   # -0.49 ... 1.49
```
```python
'input_descriptions': [
    'time of the bin centre relative to stimulus onset, in seconds (-0.49 ... 1.49)',
```

iii. Step 5 maps "time within trial -> `input[0]` 'time_from_stim_on' (time-varying), bin-centre time in
seconds, -0.49 ... 1.49; required by the decoder-task spec". The window and bin size come from the reference
`params`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None. The vector of bin centres is computed once and broadcast into every trial's input array as row 0, cast
to `float32`.

ii.
```python
inputs.append(np.stack([bin_centres.astype(np.float32),
                        np.full(NBINS, tnb[k], dtype=np.float32)], axis=0))
```

iii. N/A — the variable is defined by the alignment window rather than measured. Step 10 Check 2 verifies
independently that "input 0 == bin-centre times -0.49...1.49".

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is the centre of the same bin grid the spikes are counted into: bin `j` of the neural matrix covers
`[stimOn - 0.5 + 0.02j, stimOn - 0.5 + 0.02(j+1))` and `input[0][j] = -0.5 + 0.02(j + 0.5)` is its midpoint, so
column `j` of the input and column `j` of the neural matrix describe the same 20 ms of the trial.

ii.
```python
b = ((spike_times[s:e] - begs[k]) / binsize).astype(np.int64)   # neural bin index
...
bin_centres = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
```

iii. Step 10 Check 5 ("Bin edges: half-open `[t, t+20 ms)` bins starting exactly at `stimOn - 0.5 s`") and the
`--show-processing` figures, which plot the raster, the binned matrix and the bin grid on a common axis.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `trials.probabilityLeft`. The trials table carries no block identifier, so blocks are recovered as runs
of constant `probabilityLeft`: a change of value starts a new block.

ii.
```python
def trial_number_in_block(prob_left):
    """0-based index of each trial within its block, computed on the raw trial sequence."""
    pl = np.asarray(prob_left, dtype=float)
    newblock = np.ones(len(pl), dtype=bool)
    newblock[1:] = pl[1:] != pl[:-1]
```

iii. Step 5 maps "trial index within block -> `input[1]`, count of trials since the last change of
`probabilityLeft`"; Step 4 confirms `probabilityLeft` takes exactly the three values {0.2, 0.5, 0.8} and that
each session opens with 90 unbiased (0.5) trials, matching the data paper's block structure.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A 0-based counter that resets at every block boundary, computed over the **raw** trial sequence (before any
trial filtering) so that excluded trials still advance the count and the number reflects the animal's true
position in the block. The per-trial value is then broadcast across the 100 time bins as a constant row of the
input array, as `float32`. Observed range is [0, 98].

ii.
```python
idx = np.zeros(len(pl), dtype=np.int64)
c = 0
for i in range(len(pl)):
    c = 0 if newblock[i] else c + 1
    idx[i] = c
return idx
```
```python
tnb_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
...
tnb = tnb_all[idx_trials].astype(np.float32)
...
inputs.append(np.stack([bin_centres.astype(np.float32),
                        np.full(NBINS, tnb[k], dtype=np.float32)], axis=0))
```

iii. Step 5 decision 5 and Step 10 Check 5: "Trial-in-block index is computed on the **raw** trial sequence so
that excluded trials do not corrupt the count." Step 10 Check 2 verifies input 1 against an independent
recomputation of the block index from the raw trials table.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. `trials.choice`, which is +1, -1 or 0. +1 (leftward choice) maps to 0, -1 (rightward) maps to 1; `choice == 0`
(no response) trials are removed by the trial mask, as are NaN choices.

ii.
```python
choice_raw = trials['choice'].to_numpy()[idx_trials]         # +1 left, -1 right
choice_lab = (choice_raw < 0).astype(np.int64)               # left -> 0, right -> 1
```

iii. Step 4 "Additional consistency checks" records an empirical verification of the sign convention: on correct
trials with a left stimulus `choice == +1` and with a right stimulus `choice == -1`, so +1 is a left choice.
The instruction's mapping is left = 0, right = 1.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Nothing beyond the recoding above. The per-trial label is broadcast across all 100 bins so that the output is
time-varying in shape (as the instructions prefer), stored as `int64`. The observed distribution is
0.508 left / 0.492 right.

ii.
```python
outputs.append(np.stack([np.full(NBINS, choice_lab[k], dtype=np.int64),
                         np.full(NBINS, prior_lab[k], dtype=np.int64),
                         wheel_lab[k], me_lab[k]], axis=0))
```

iii. Step 5: "All four outputs are stored time-varying (shape (4, 100)), as the task instructions prefer:
per-trial variables (choice, prior) are constant across the 100 bins." Step 10 Check 2 verifies output 0 against
the raw trials table for every trial of the two sample sessions.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. `trials.probabilityLeft`, recoded 0.2 -> 0, 0.5 -> 1, 0.8 -> 2 exactly as the instructions specify. Any other
value would be flagged and its trials dropped; none were found.

ii.
```python
pleft_raw = trials['probabilityLeft'].to_numpy()[idx_trials]
prior_lab = np.select([pleft_raw == 0.2, pleft_raw == 0.5, pleft_raw == 0.8], [0, 1, 2],
                      default=-1).astype(np.int64)
if np.any(prior_lab < 0):
    keep_pl = prior_lab >= 0
    ...
```

iii. Step 5 maps `trials.probabilityLeft -> output[1]` with the instruction's mapping, identifying it with the
reference's `block` behaviour variable in `bin_behaviors`. Step 4 verifies the variable takes only the three
expected values.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Only the recoding, plus broadcasting across the 100 bins. Distribution in the converted data is
0.418 / 0.140 / 0.442 for 0.2 / 0.5 / 0.8, consistent with a session of 90 unbiased trials followed by
alternating biased blocks.

ii.
```python
outputs.append(np.stack([np.full(NBINS, choice_lab[k], dtype=np.int64),
                         np.full(NBINS, prior_lab[k], dtype=np.int64),
                         wheel_lab[k], me_lab[k]], axis=0))
```

iii. Step 9's consistency table compares the converted prior distribution (0.418/0.140/0.442) with an
independent survey of the raw data (0.412/0.155/0.432) and marks it consistent; Step 10 Check 2 verifies the
per-trial labels against the raw table.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.position` / `_ibl_wheel.timestamps`, accessed through `SessionLoader.load_wheel()`, which returns
a dataframe of `times, position, velocity, acceleration`. Speed is the absolute value of the returned velocity,
exactly as the reference's `load_target_behavior(..., 'wheel-speed')` does.

ii.
```python
sl.load_wheel()
wheel_times = sl.wheel['times'].to_numpy()
wheel_speed_raw = np.abs(sl.wheel['velocity'].to_numpy())
```

iii. Step 1 documents `load_target_behavior` as "`SessionLoader.load_wheel()` -> wheel-speed = abs(velocity)
(Gaussian-smoothed, uniformly sampled)", and Step 5 maps "wheel `velocity` -> `output[2]`, `abs(velocity)`,
interpolated onto the bin grid, then discretized".

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three stages. (1) `SessionLoader.load_wheel()` interpolates the (event-driven) wheel position onto a uniform
1 kHz grid and differentiates it with smoothing to give velocity; the absolute value is taken. (2) For each
trial the trace is sliced to the 2 s window and linearly interpolated onto 100 sample points
`linspace(beg + binsize, end, 100)` — i.e. the *right edge* of each bin — reproducing the reference's
`get_behavior_per_interval`; a trial whose trace starts late, ends early or contains NaN is marked invalid and
dropped. (3) The resulting (n_trials, 100) matrix is discretized into three classes (see 7-c). No additional
filtering or normalisation is applied.

ii.
```python
grid = (np.linspace(binsize, t_end - t_start, n_bins)[None, :] + begs[:, None])
for k in range(n_trials):
    ib, ie = idx_beg[k], idx_end[k]
    if ie <= ib:
        valid[k] = False
        continue
    tt = times[ib:ie]
    vv = values[ib:ie]
    if np.abs(begs[k] - tt[0]) > binsize or np.abs(ends[k] - tt[-1]) > binsize:
        valid[k] = False      # target data starts too late / ends too early
        continue
    if np.any(np.isnan(vv)):
        valid[k] = False      # NaNs would break the decoder
        continue
    out[k] = np.interp(grid[k], tt, vv)
```

iii. Step 6's function table states `bin_behavior` is the equivalent of `get_behavior_per_interval`: "linear
interpolation onto `linspace(beg+binsize, end, n_bins)` + validity checks", and Step 3 records that the
interpolation grid is at bin *end* times. The velocity computation itself is not a choice — it is what
`SessionLoader` does by default and what the reference relies on. Step 10 Check 2 recomputes the interpolated
and thresholded wheel values independently from the raw ALF objects and reports a `np.allclose` pass.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into three classes by **per-session tertiles**: the 1/3 and 2/3 quantiles are computed over all retained
trials × all 100 bins of that session, and each value is labelled 0 (low), 1 (medium), 2 (high). The thresholds
are recorded per session in the metadata (`wheel_tertiles`). This yields class fractions of 0.333/0.333/0.333 by
construction.

ii.
```python
def discretize_tertiles(x, valid_rows):
    """Discretize a (n_trials, n_bins) signal into 3 classes using per-session tertiles."""
    ref = x[valid_rows].ravel()
    q1, q2 = np.quantile(ref, [1. / 3., 2. / 3.])
    lab = np.zeros(x.shape, dtype=np.int64)
    lab[x > q1] = 1
    lab[x > q2] = 2
    return lab, (float(q1), float(q2))
```

iii. Step 5 decision 4: the task spec asks for 3 bins; both signals have arbitrary session-specific scales
(whisker ME depends on camera gain, lighting and ROI size; wheel speed scale depends on the mouse), so a global
threshold "would make classes wildly unbalanced across sessions"; tertiles give balanced classes so chance is
exactly 1/3 and balanced accuracy is interpretable.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is evaluated at 100 points inside the same window measured from the same `stimOn_times`, one
point per neural bin, so column `j` of the wheel output corresponds to neural bin `j`. The sample point used is
the bin's right edge (`stimOn - 0.5 + 0.02(j+1)`) rather than its centre, following the reference
`get_behavior_per_interval`; this leaves a 10 ms offset between the behavioural sample and the centre of the
neural bin it is paired with.

ii.
```python
begs = align_times + t_start
...
grid = (np.linspace(binsize, t_end - t_start, n_bins)[None, :] + begs[:, None])
...
out[k] = np.interp(grid[k], tt, vv)
```

iii. Step 3/Step 6 state that the reference interpolates "onto `np.linspace(t_beg + binsize, t_end, n_bins)`,
i.e. values at bin *end* times", and the AI copies that convention deliberately. The `--show-processing` plot
explicitly draws the interpolated points at `bin_centres + 0.5*BINSIZE` on top of the raw trace, and Step 7
reports "the interpolated points lie on the raw traces" with no visible offset.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<side>Camera.ROIMotionEnergy` with the camera frame times, loaded via
`SessionLoader.load_motion_energy(views=[view])`, which returns a dataframe with `times` and
`whiskerMotionEnergy`. The left camera (60 Hz) is used when available, the right camera (150 Hz) as fallback;
sessions with neither are dropped (14 sessions).

ii.
```python
for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sl.load_motion_energy(views=[view])
        df = sl.motion_energy[key]
        me_times = df['times'].to_numpy()
        me_vals = df['whiskerMotionEnergy'].to_numpy()
        camera_used = view
        break
    except Exception:
        continue
if me_times is None:
    return dict(eid=eid, skipped='no whisker motion energy')
```

iii. Step 4's discrepancy table records "reference tries left camera, falls back to right ... Same as reference:
left preferred, right as fallback", and notes the left camera is the 60 Hz stream the method paper describes.
The camera actually used is recorded per session in the metadata.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is, with no filtering or normalisation. It is passed through the same
`bin_behavior` routine as the wheel: sliced to the 2 s window, checked for coverage and NaNs, and linearly
interpolated onto the 100 bin right edges; then discretized into three classes.

ii.
```python
me_binned, me_valid = bin_behavior(me_times, me_vals, align)
good = wheel_valid & me_valid
...
me_lab, me_q = discretize_tertiles(me_binned, allrows)
```

iii. Step 5 maps "`leftCamera.ROIMotionEnergy` (`whiskerMotionEnergy`; right camera as fallback) ->
`output[3]`, interpolated onto bin grid, then discretized into 3 bins", citing
`load_target_behavior('left-whisker-motion-energy')` and `get_behavior_per_interval` as the reference
counterparts. Step 10 Check 2 verifies the labels against an independent recomputation.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: per-session tertiles (1/3 and 2/3 quantiles over all retained trials and bins of
that session), labels 0/1/2, thresholds recorded in the metadata as `me_tertiles`. Resulting fractions
0.334/0.333/0.333.

ii.
```python
wheel_lab, wheel_q = discretize_tertiles(wheel_binned, allrows)
me_lab, me_q = discretize_tertiles(me_binned, allrows)
```

iii. Same rationale as 7-c (Step 5 decision 4): session-specific scale of the motion-energy signal makes a
global threshold meaningless, and tertiles keep chance at exactly 1/3.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same mechanism as the wheel: the camera trace is interpolated at 100 points measured from the same
`stimOn_times`, one per neural bin, at the right edge of each bin. Camera frame times are already on the session
clock shared with the spikes, so no further alignment is needed. The 10 ms bin-edge-versus-bin-centre offset
noted in 7-d applies here too.

ii.
```python
me_binned, me_valid = bin_behavior(me_times, me_vals, align)
```
```python
grid = (np.linspace(binsize, t_end - t_start, n_bins)[None, :] + begs[:, None])
```

iii. As for 7-d: the grid is copied from `get_behavior_per_interval`, and the `--show-processing` figure overlays
the raw motion-energy trace, the interpolated samples and the tertile thresholds for an example trial to
demonstrate no offset (Step 7 and Step 12 "Temporal alignment verified visually").

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or defective data is detected and the affected unit of data dropped, at every level.
*Trial level*: NaNs in any of the six required trial fields are excluded by the reference mask; trials whose
wheel or camera trace does not span the window or contains NaN are dropped; trials outside the spike-sorted
recording interval are dropped; trials with zero population spikes are dropped; trials whose `probabilityLeft`
is not one of the three expected values would be dropped.
*Session level*: a session with no motion energy from either camera, with fewer than 5 good units, with fewer
than 2 usable trials at any stage, or that raises any exception is skipped and the reason printed (19 of 459
skipped, 14 of them for missing motion energy).
*Subject/region level*: subjects and regions are derived from what survives, so empty ones simply do not appear.
One session was lost because a single probe's spike sorting is unreleased (`clusters` is `None`): the resulting
`AttributeError` is caught by the top-level per-session handler and the whole session is skipped rather than
falling back to the session's other probe.

ii.
```python
def _worker(args):
    eid, show = args
    try:
        return convert_session(eid, show_processing=show)
    except Exception as e:
        traceback.print_exc()
        return dict(eid=eid, skipped=f'exception: {e}')
```
```python
if np.any(np.isnan(vv)):
    valid[k] = False      # NaNs would break the decoder
```
```python
rec_t0, rec_t1 = max(probe_t0), min(probe_t1)
covered = (align_all + TIME_WINDOW[0] >= rec_t0) & (align_all + TIME_WINDOW[1] <= rec_t1)
mask = mask & covered
if mask.sum() < MIN_TRIALS:
    return dict(eid=eid, skipped='too few trials inside the ephys recording')
```

iii. Step 10 Check 5 enumerates the edge cases handled (two-probe merging, missing left camera, trials at the
edges of the wheel/video/ephys coverage, unexpected `probabilityLeft`, block counting on raw trials, bin-edge
off-by-one, `clusters is None`). Step 12 lists the issues found and resolved; the log
`conversion_full_out.txt` prints every skipped session with its reason, and the AI cross-checks that the 14
sessions without motion energy are exactly the 459 − 445 found by its independent dataset survey.

## 10-a. What are the most time-consuming steps of the code?

i. Reading the spike sorting off disk, by a wide margin: 3–6 s per session (1–2 probes) out of a 4–7 s total,
against 0.3–0.5 s for the trials table, ~0.5 s for wheel + motion energy and ~0.03 s for all binning. The
script instruments each stage into a `timing` dict and prints it in serial mode; the full run over 459 sessions
with 12 worker processes took 408 s. A contributing factor the AI does not flag is that
`load_spike_sorting()` is called with the default `SPIKES_ATTRIBUTES = ['clusters', 'times', 'amps', 'depths']`,
so twice the necessary spike data is read.

ii.
```python
t0 = time.time()
pids, pnames = one.eid2pid(eid)
...
    spikes, clusters, channels = ssl.load_spike_sorting()
...
timing['spikes'] = time.time() - t0
```
```python
| spike sorting load | 3-6 s (1-2 probes) | dominant cost |
```

iii. Step 7's run-time table attributes the dominant cost to spike-sorting load and estimates
459 × 5.5 s / 12 workers ≈ 3.5–6 min, which the 408 s full run confirmed. Step 6 states that parallelism is
applied at the session level "which is where the real cost is (spike-sorting I/O, ~3-6 s per probe)".

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four remain, all cheap relative to I/O. (1) The per-trial loop in `bin_spikes`, which could be replaced by a
single `bincount` over trial-offset flat indices. (2) The per-trial loop in `bin_behavior`, which could be a
single `np.interp` over a concatenated query vector. (3) The explicit Python `for i in range(len(pl))` loop in
`trial_number_in_block`, which is straightforwardly vectorisable (e.g. `arange - maximum.accumulate(block start
index)`, or a `groupby(...).cumcount()`); this is the one loop with no counterpart in the reference and it runs
over every raw trial of every session. (4) The final per-trial assembly loop that stacks the input and output
arrays. In addition, `np.isin(spike_clusters, keep_ids)` sorts/searches over every spike in the session where a
boolean lookup array indexed by cluster id would be O(n) — the same information `remap` already encodes.
The AI's own documentation only discusses the loops it *removed* relative to the reference; it does not identify
remaining vectorisation opportunities.

ii.
```python
for k in range(n_trials):
    s, e = i0[k], i1[k]
    ...
    cnt = np.bincount(idx, minlength=n_clusters * n_bins)
```
```python
for i in range(len(pl)):
    c = 0 if newblock[i] else c + 1
    idx[i] = c
```
```python
sel = np.isin(spike_clusters, keep_ids)
```

iii. Step 6 records the vectorisations that were made: "The reference bins spikes with a `multiprocessing` pool
**per trial**, which is dominated by pickling overhead. Here all trials of a session are binned with two
`np.searchsorted` calls plus one `np.bincount` per trial, giving ~0.01 s per session instead of tens of
seconds"; likewise the behaviour interpolation. Since binning costs ~0.03 s per session against 4–7 s of I/O,
further vectorisation would not change the run time.

## 10-c. What processing does the code repeat multiple times?

i. Per-session setup that could be per-worker: `convert_session` constructs a fresh `ONE` client **and** a fresh
`iblatlas.regions.BrainRegions()` (which reads the Allen/Beryl region tables) on every one of the 459 calls,
rather than once per worker process via a pool initializer as the reference does. `SessionLoader.load_trials`,
`load_wheel` and `load_motion_energy` each re-resolve datasets through ONE for the same session, and
`load_motion_energy` is attempted for the left camera and then the right when the left is absent. Within a
probe, `merge_clusters` recomputes the full cluster metrics table although only `label` and `acronym` are used.
The AI does not document any of this as repeated work.

ii.
```python
def convert_session(eid, show_processing=False, outdir='/app'):
    ...
    one = get_one()
    br = BrainRegions()
```
```python
for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sl.load_motion_energy(views=[view])
```

iii. Step 6 explains only the deliberate part of this: "Each worker builds its own `ONE` instance (the ONE object
is not fork-safe for concurrent use)" — correct in principle, but the object is rebuilt per session rather than
per worker. Total run time (408 s) was well inside the 15-minute budget, so the repetition was never chased.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Mostly I/O and bookkeeping. (1) `load_spike_sorting()` is called with the default attribute list, so
`spikes.amps` and `spikes.depths` are read from disk for every probe and never used — roughly doubling the
dominant cost of the pipeline (the human reference explicitly sets
`bio.SPIKES_ATTRIBUTES = ['clusters', 'times']` to avoid this). (2) `merge_clusters(...).to_df()` builds the
full cluster metrics dataframe, of which only `acronym` and `label` are read. (3) `SessionLoader.load_wheel()`
computes and returns acceleration, which is discarded. (4) Continuous `wheel_binned` / `me_binned` matrices are
computed in `float64` and kept for the whole session although only their tertile labels are stored.
(5) `bin_spikes` bins all masked trials and then discards the zero-spike ones. (6) The `valid_rows` argument of
`discretize_tertiles` is always `allrows`, so the masking it implements is dead. (7) Per-session extras stored in
the result dict (`trial_idx`, `n_trials_raw`, `timing`) are either dropped or only used for metadata; the
per-session `timing` dict is collected for all 459 sessions but only printed in the serial code path.
None of this is documented by the AI.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
clu = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
```
```python
allrows = np.ones(len(align), dtype=bool)
wheel_lab, wheel_q = discretize_tertiles(wheel_binned, allrows)
```
```python
nonzero = binned_spikes.sum(axis=(1, 2)) > 0
...
binned_spikes = binned_spikes[nonzero]
```

iii. The AI documents only the unnecessary processing it *avoided* — Step 1 notes that the reference's
`raw_electrophysiology(band='ap', stream=True).fs` call "requires streaming raw ephys and is not needed for
conversion, so it is skipped here" — and states in Step 6 that memory/efficiency effort was directed at the
binning and at session-level parallelism. It offers no justification for the unused spike attributes or the
retained float64 behaviour matrices, presumably because the total run time (408 s) and peak memory were
acceptable.
