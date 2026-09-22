# Decisions

> **Scope note.** The AI agent (codex / gpt-5.6-sol, 43 steps) stopped during **Step 2 (Dataset
> Exploration)** and declared the task blocked. The only artifact it produced is
> `/app/CONVERSION_NOTES.md` (Steps 0 and 1 marked COMPLETE, Step 2 IN PROGRESS, Steps 3–13 left as
> the unfilled template). **`/app/convert_data.py` does not exist**, and neither do
> `/app/converted_data.pkl`, `/app/sample_data.pkl`, `/app/README.md`, or any of the
> conversion/verification/training log files listed in the task description. Because there is no
> conversion script, section **ii** of every question below cannot quote `convert_data.py`; instead
> it quotes the closest available evidence — the agent's throwaway exploration code from
> `/logs/agent/trajectory.json` and the text of `CONVERSION_NOTES.md`.
>
> The agent's terminating decision was:
>
> > "Blocked at Step 2: ONE metadata lists trial tables, but they are inaccessible through the
> > required ONE API for all 459 sessions. Choice, stimulus onset, prior, and behavior alignment
> > data therefore cannot be loaded. … The cache must be restaged/restored before conversion can
> > continue without violating the ordered workflow."
>
> This premise is false. I verified in this same container that the trials table loads fine through
> the ONE API when an Alyx-backed client is used (`mode='remote'`, no `password=` argument so the
> cached auth token is reused, plus `One.load_cache(one, tables_dir=...)`), which is exactly what
> the human reference `connect()` does and documents:
>
> ```
> (569, 20) ['goCueTrigger_times', 'intervals_bpod_0', ..., 'choice', 'stimOn_times',
>  'contrastLeft', 'contrastRight', 'feedback_times', 'feedbackType', 'rewardVolume',
>  'probabilityLeft', 'firstMovement_times', 'intervals_0', 'intervals_1']
> ```
>
> The agent used `mode='local'`, which returns a plain `One` that cannot resolve the revisioned
> dataset paths (`alf/#2024-02-22#/_ibl_trials.table.pqt`) that are actually on disk, so only the
> un-revisioned `_ibl_trials.goCueTrigger_times.npy` was found. It made exactly one attempt at an
> Alyx-backed client (trajectory step 34) and passed `password='international'`, which forces
> network re-authentication; that failed with `ConnectionRefusedError`, and the agent abandoned the
> approach rather than retrying without the password. It also saw the direct hint
> `eid2pid error NotImplementedError('Converting to probe ID requires remote connection')` and did
> not act on it.

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. No loading pipeline was built. The agent explored the cache with `one.api.ONE` in `mode='local'`,
pointing `cache_dir` at `/app/data/one_cache` and `tables_dir` at one of the three release-table
directories it found (`2022_Q4_IBL_et_al_BWM`, `2025_Q3_IBL_et_al_BWM`, `Brainwidemap`). It
discovered that `tables_dir` must be an absolute path (a bare name yields "No cache tables found")
and then enumerated 354 / 459 / 480 sessions respectively. It never chose which release to convert,
never built a session-selection function, and never loaded a single complete session. It correctly
avoided reading files in `/app/data` directly. It reached `SpikeSortingLoader` successfully for one
probe, but `SessionLoader.load_trials()` returned only a single column, and it concluded the cache
was broken and stopped.

ii. No `convert_data.py` exists. The nearest equivalent is the exploration code in the trajectory
(step 21 / step 25 / step 27):

```python
from one.api import ONE
for td in ['2022_Q4_IBL_et_al_BWM','2025_Q3_IBL_et_al_BWM','Brainwidemap']:
    path = '/app/data/one_cache/' + td
    one = ONE(cache_dir='/app/data/one_cache', tables_dir=path, mode='local')
    eids = one.search()
    print(td, len(eids), eids[:3], len(one.list_datasets()))
# 2022_Q4_IBL_et_al_BWM 354 ... | 2025_Q3_IBL_et_al_BWM 459 ... | Brainwidemap 480 ...
```

```python
sets = ['_ibl_trials.table.pqt','_ibl_wheel.position.npy','_ibl_wheel.timestamps.npy',
        '_ibl_leftCamera.times.npy','leftCamera.ROIMotionEnergy.npy',
        'spikes.times.npy','spikes.clusters.npy','clusters.brainLocationAcronyms_ccf_2017.npy']
for d in sets:
    print(d, len(one.search(datasets=d)))
print('all behavior', len(one.search(datasets=sets[:5])))   # 432
```

```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()          # -> (569, 1)  ['goCueTrigger_times']
ssl = SpikeSortingLoader(one=one, eid=eid, pname=pname)
spikes, clusters, channels = ssl.load_spike_sorting()
merged = ssl.merge_clusters(spikes, clusters, channels)
```

The one Alyx-backed attempt, which failed and was never retried without `password`:

```python
one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True, cache_dir='/app/data/one_cache')
# ConnectionRefusedError: [Errno 111] Connection refused
```

iii. From `CONVERSION_NOTES.md`, Step 2: "Exploration used `ONE(cache_dir='/app/data/one_cache',
tables_dir=..., mode='local')`, `ONE.search`, `ONE.list_datasets`, `ONE.get_details`,
`ONE.load_object`, and brainbox loaders; no source data file was read directly." And:
"**Blocking cache integrity finding:** ONE metadata says `_ibl_trials.table.pqt` exists for 459
sessions, but the files are not reachable through ONE. … This is a source-cache staging problem
rather than a conversion-code issue. Per the ordered workflow, Step 2 remains IN PROGRESS until the
registered trial tables are made accessible through ONE." The agent also noted in Step 1 that the
reference caching script "randomly selects subjects and takes the first release-table EID per
selected subject; the supplied task instead asks for the complete locally staged dataset" — i.e. it
intended to convert everything, but never implemented that.

## 1-b. How are the data split into subjects?

i. No split was implemented. The agent obtained subject identity from the ONE session details
(`one.get_details(eid)['subject']`) and reported per-release subject counts and
sessions-per-subject statistics in the Step 2 table: 143 subjects / 3.36 sessions per subject
(mean), min 1, max 13 for the composite `Brainwidemap` tables; 139 / 3.30 for `2025_Q3`; 115 / 3.08
for `2022_Q4`. It never chose which release defines the dataset, never built the `subjects` list,
and never built `subject_idx`.

ii. No `convert_data.py`. Trajectory step 41:

```python
eids = one.search()
details = [one.get_details(e) for e in eids]
subjects = sorted({d['subject'] for d in details})
labs = sorted({d['lab'] for d in details})
print(td, 'sessions', len(eids), 'subjects', len(subjects), 'labs', len(labs))
```

iii. Not stated beyond the descriptive Step 2 table. The agent gave no rationale for a subject split
because it never reached the mapping step (Step 5 is an empty template).

## 1-c. How are the data split into sessions?

i. No split was implemented. The agent observed that ONE's session index already enumerates
sessions one `eid` at a time (`one.search()` returning 480 / 459 / 354 eids depending on the release
table) and that filtering by required datasets narrows this (e.g. 432 sessions have trials + wheel +
left-camera times + left-camera ROI motion energy). It never selected a session set for conversion
and never processed a session end to end.

ii. No `convert_data.py`. Trajectory step 25:

```python
print('all behavior', len(one.search(datasets=sets[:5])))     # 432
print('all requirements', len(one.search(datasets=sets)))     # 24
```

iii. No rationale recorded. Notably the agent saw that requiring
`clusters.brainLocationAcronyms_ccf_2017.npy` collapsed the session count from 432 to 24, but did
not investigate or resolve this (the reference gets histology via
`SpikeSortingLoader.merge_clusters`, not that dataset).

## 1-d. How are the data split into trials?

i. Not addressed. The agent identified `_ibl_trials.table.pqt` as the one-row-per-trial table in the
ONE index, but was never able to read it (it only ever got `goCueTrigger_times`, e.g. shapes
`(569,)` and `(885,)`). It reported "Trials (total): Blocked: registered trial tables inaccessible
through ONE" and "Trials / session: Blocked" in the Step 2 size table. No trial splitting code was
written.

ii. No `convert_data.py`. Trajectory steps 29–37, all failing:

```python
one.load_dataset(eid, '_ibl_trials.table.pqt')            # ALFObjectNotFound('Dataset not found')
one.load_dataset(eid, 'alf/_ibl_trials.table.pqt')        # ALFObjectNotFound('Dataset not found')
one.load_object(eid, 'trials', collection='alf', attribute=['table'])
#   ALFObjectNotFound('ALF object "trials" not found on disk')
one.load_object(eid, 'trials', collection='alf', attribute=None)
#   ['goCueTrigger_times'] {'goCueTrigger_times': (569,)}
```

iii. `CONVERSION_NOTES.md`, Step 2: "Therefore native trial dimensions/dtypes and total trials cannot
yet be measured, and wheel/whisker trial alignment cannot be performed."

## 1-e. How are trials filtered based on quality controls?

i. No filtering was implemented. In Step 1 the agent correctly summarized the reference code's trial
mask but never adopted it as its own decision and never coded it. Its Step 1 table records for
`load_trials_and_mask`: "Excludes RT <0.08 s or >2 s, missing required events, no-choice trials, and
(in `prepare_data`) trial duration >10 s. Keeps unbiased probabilityLeft=0.5 trials." It also noted
"The required target structure retains all valid trials; downstream `train_decoder.py` performs its
own split." Nothing was decided about dropping trials whose wheel/camera coverage is incomplete, and
Step 5 (where such a curation rule would be specified) is blank.

ii. No `convert_data.py`. The only record is the Step 1 notes table row quoted above.

iii. Implicit: match the reference. The agent never wrote out a rationale or committed to
thresholds, because it stopped before Step 5.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Not decided in code, but correctly identified in exploration: pykilosort spike sorting reached
through `SpikeSortingLoader`, i.e. `spikes.times` / `spikes.clusters` plus merged cluster/channel
metadata via `merge_clusters`. The agent listed the available spike objects
(`spikes.times.npy`, `spikes.clusters.npy`, `spikes.depths.npy`, `spikes.amps.npy`,
`spikes.templates.npy`, `_phy_spikes_subset.*`) and successfully loaded one probe (80,867,860 spikes,
1,557 clusters). It never narrowed the loaded attributes or wrote a `load_neural` equivalent.

ii. No `convert_data.py`. Trajectory step 27:

```python
cols = one.list_collections(eid, filename='spikes.times.npy')   # ['alf/probe00/pykilosort']
for c in cols:
    pname = c.split('/')[1]
    ssl = SpikeSortingLoader(one=one, eid=eid, pname=pname)
    spikes, clusters, channels = ssl.load_spike_sorting()
    merged = ssl.merge_clusters(spikes, clusters, channels)
```

iii. `CONVERSION_NOTES.md`, Step 2: "A test spike-sorting load through `SpikeSortingLoader` succeeded
(80,867,860 spikes and 1,557 clusters for one probe), showing the ephys objects are reachable."

## 2-b. How is the `neural` data processed?

i. No processing was implemented. The agent documented the reference behaviour in Step 1:
`bin_spiking_data` / `get_spike_data_per_interval` produce "Stimulus-aligned spike counts in fixed
bins; caching parameters use [-0.5,1.5) s and 20 ms (100 bins)", and `merge_probes` "Reindexes
clusters across probes, concatenates, and stable-sorts spikes by time." It also noted the required
transposition: "Spike arrays in the reference are trial × time × neuron; the requested pickle
requires each trial transposed to neuron × time." It did not decide whether to emit counts or rates
(Hz), and wrote no binning, merging, or smoothing code.

ii. No `convert_data.py`. Only the Step 1 notes table rows quoted above.

iii. `CONVERSION_NOTES.md`, Step 1 Notes: "Core reference parameters are stimulus onset alignment, a
2 s window from -0.5 to +1.5 s, and 20 ms bins. It merges all probes per session and operates on
extracellular spikes, so delta-F/F is not applicable."

## 2-c. How is the `neural` data filtered based on quality controls?

i. **Explicitly deferred and never decided.** The agent correctly spotted the tension between the
reference code (which keeps every sorted cluster) and the data paper (which uses stringent unit QC),
and postponed the choice: `load_spiking_data` — "`qc=None` keeps all clusters; optional `qc=1` would
retain labels >=1, but caching calls the default." No decision was made about the `label >= 1`
criterion, and nothing was decided or said about excluding units whose Beryl acronym is `void`
(outside the brain). No filtering code exists.

ii. No `convert_data.py`. Only the Step 1 notes table row quoted above.

iii. `CONVERSION_NOTES.md`, Step 1 Notes: "Although good-unit labels are saved, the reference caching
call does not filter clusters (`qc=None`); this will be reconciled against the data-paper BWM
curation in later steps." Those later steps were never reached.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Not implemented. The agent recorded that the reference aligns to stimulus onset and that the
decoder task also specifies stimulus onset, but it never obtained `stimOn_times` (the trials table
was never read) and wrote no alignment code. It never examined whether spike times, wheel timestamps
and camera times share one session clock.

ii. No `convert_data.py`. Step 1 notes only: "Stimulus-aligned spike counts in fixed bins".

iii. `CONVERSION_NOTES.md`, Step 1 Notes: "Core reference parameters are stimulus onset alignment, a
2 s window from -0.5 to +1.5 s, and 20 ms bins."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. There is no converted data, so there is no temporal resolution. The agent did document the
reference configuration — 20 ms bins, 100 bins spanning [-0.5, +1.5) s around stimulus onset — and
said nothing about rebinning or interpolating the neural stream. No window constants, edge grid, or
binning code were written, and no `time_bin_size` metadata field exists.

ii. No `convert_data.py`. The only record is the Step 1 notes text:
"caching parameters use [-0.5,1.5) s and 20 ms (100 bins)".

iii. `CONVERSION_NOTES.md`, Step 1: reference parameters are quoted as fact; no adoption statement,
no rationale, and the Step 3 table where the paper's binning would be cross-checked is empty.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Not addressed at all. The agent never reached input construction. It never identified
`stimOn_times` as the source (it could not read the trials table), and the Step 5 variable-mapping
table row `| | input[0] | | |` is left blank.

ii. No `convert_data.py`; no code or notes exist for this input.

iii. None recorded.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. Not addressed. No decision about representing time as bin centres versus edges, no window, no
per-trial versus shared representation, no dtype.

ii. No `convert_data.py`; nothing exists.

iii. None recorded.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Not addressed. Since neither the neural binning grid nor the time input exists, no alignment was
defined.

ii. No `convert_data.py`; nothing exists.

iii. None recorded.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Not addressed. The agent mentioned `probabilityLeft` only in the context of the reference trial
mask ("Keeps unbiased probabilityLeft=0.5 trials"). It never noted that the trials table carries no
block identifier, and never identified that blocks must be recovered from changes in
`probabilityLeft`.

ii. No `convert_data.py`; nothing exists.

iii. None recorded.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Not addressed. No decision on counting from zero versus one, and — importantly — no decision on
whether the count is taken before or after trial filtering (the reference counts before filtering so
that the number reflects the animal's true position in the block).

ii. No `convert_data.py`; nothing exists.

iii. None recorded.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Partially identified, not implemented. The agent noted that IBL's native `choice` takes values
-1 / +1 / 0 and that the task requires a left=0 / right=1 encoding, but it did not state which sign
is left, did not decide how to treat `choice == 0` (no-response) trials beyond echoing the
reference's `exclude_nochoice`, and wrote no code. The `choice` column was never actually read.

ii. No `convert_data.py`. The only record is the Step 1 notes sentence quoted below.

iii. `CONVERSION_NOTES.md`, Step 1 Notes: "Choice values in native IBL are typically -1/+1 (with 0
no-choice); the task explicitly remaps left/right to 0/1."

## 5-b. What processing is involved in computing `output` *Choice*?

i. Not implemented. No mapping dictionary, no decision on whether choice is emitted per-trial or
broadcast across the 100 time bins (the instructions prefer time-varying), and no dtype decision.

ii. No `convert_data.py`; nothing exists.

iii. None recorded beyond the sentence quoted in 5-a.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Not implemented. `probabilityLeft` appears in the agent's notes only as part of the reference
trial mask description. The required 0.2→0 / 0.5→1 / 0.8→2 recoding from the task specification is
never mentioned anywhere in `CONVERSION_NOTES.md` or the trajectory.

ii. No `convert_data.py`; nothing exists.

iii. None recorded.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Not addressed. No recoding, no handling of values outside {0.2, 0.5, 0.8}, no decision on
per-trial versus time-varying representation.

ii. No `convert_data.py`; nothing exists.

iii. None recorded.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Correctly identified in the abstract, never implemented. The agent recorded that the reference's
`load_target_behavior` "Uses brainbox `SessionLoader`; wheel speed is absolute smoothed wheel
velocity", and its dataset search confirmed `_ibl_wheel.position.npy` and
`_ibl_wheel.timestamps.npy` exist for all 459 sessions. It never called `SessionLoader.load_wheel()`
and wrote no code.

ii. No `convert_data.py`. Trajectory step 25 only checked availability:

```python
sets = [..., '_ibl_wheel.position.npy', '_ibl_wheel.timestamps.npy', ...]
for d in sets:
    print(d, len(one.search(datasets=d)))     # _ibl_wheel.position.npy 459
```

iii. `CONVERSION_NOTES.md`, Step 1 table, `load_target_behavior` row: "wheel speed is absolute
smoothed wheel velocity".

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Not implemented. The agent noted only that the reference interpolates continuous behavioural
traces linearly and that `get_behavior_per_interval` "Selects samples around event and linearly
interpolates to bin endpoints (`align+start+binsize` through end); flags inadequate coverage". It
made no decision about the `SessionLoader` interpolation frequency, the 20 Hz Butterworth filter, the
absolute value, or resampling onto the neural bin grid.

ii. No `convert_data.py`. Step 1 table rows only.

iii. `CONVERSION_NOTES.md`, Step 1 Notes: "Continuous behavioral traces are linearly interpolated."

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. **Not decided.** The agent flagged that a decision was needed but never made one: it observed
that the reference treats wheel speed as a continuous regression target while the task requires three
categorical bins. No thresholding rule (percentile-based, fixed-value, per-session versus global) was
chosen and no code exists.

ii. No `convert_data.py`; nothing exists.

iii. `CONVERSION_NOTES.md`, Step 1 Notes: "The reference paper treats wheel speed and whisker motion
energy as continuous regression targets, whereas this task explicitly requires three categorical
bins." That is the whole of the record — a statement of the problem, not a decision.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Not addressed. The agent recorded the reference's per-interval interpolation approach but chose
no alignment target (bin centres, bin endpoints, or edges) and wrote no code. Note that the
reference code it quoted interpolates to bin *endpoints*, whereas the human reference solution uses
bin *centres*; the agent never noticed or resolved this difference.

ii. No `convert_data.py`; nothing exists.

iii. None recorded beyond the `get_behavior_per_interval` table row.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Correctly identified in the abstract, never implemented. The agent recorded that the reference's
`load_target_behavior` "prefers left camera then right fallback", and its dataset census found
`_ibl_leftCamera.times.npy` in 436 sessions and `leftCamera.ROIMotionEnergy.npy` in 432. It never
loaded the motion-energy trace and wrote no camera-selection function.

ii. No `convert_data.py`. Trajectory step 25:

```python
print('_ibl_leftCamera.times.npy',      len(one.search(datasets='_ibl_leftCamera.times.npy')))      # 436
print('leftCamera.ROIMotionEnergy.npy', len(one.search(datasets='leftCamera.ROIMotionEnergy.npy'))) # 432
```

iii. `CONVERSION_NOTES.md`, Step 1 table, `load_target_behavior` row: "whisker motion energy prefers
left camera then right fallback."

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Not implemented. No decision on whether the released ROI motion-energy trace is used as-is,
filtered, or normalised; no resampling scheme; no code.

ii. No `convert_data.py`; nothing exists.

iii. None recorded beyond the generic "Continuous behavioral traces are linearly interpolated."

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. **Not decided**, exactly as for wheel speed. The need for three categorical bins was noted; no
threshold rule was chosen and no code exists.

ii. No `convert_data.py`; nothing exists.

iii. `CONVERSION_NOTES.md`, Step 1 Notes: "…whereas this task explicitly requires three categorical
bins."

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Not addressed. No alignment decision, no code. The agent explicitly stated that
"wheel/whisker trial alignment cannot be performed" given its (mistaken) conclusion that stimulus
onset times were unavailable.

ii. No `convert_data.py`; nothing exists.

iii. `CONVERSION_NOTES.md`, Step 2: "Therefore native trial dimensions/dtypes and total trials cannot
yet be measured, and wheel/whisker trial alignment cannot be performed."

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. No handling exists. The agent's only encounter with imperfect data was the revisioned-dataset
situation in the cache (`alf/#2022-10-31#/_ibl_trials.table.pqt`, `alf/#2024-02-22#/…`,
`alf/#2024-05-06#/spikes.*`), and it mis-handled it: rather than recognising that a revision-aware
(Alyx-backed) client is needed, it classified the situation as corruption — "This is a source-cache
staging problem rather than a conversion-code issue" — and halted. It had also seen the
`ALFWarning: No default revision for dataset … using most recent` messages during the successful
spike-sorting load, which was direct evidence that revision resolution, not file availability, was
the issue. No rules were defined for missing camera/wheel coverage, unreleased probes, sessions with
zero surviving units, or sessions with fewer than two trials.

ii. No `convert_data.py`. Trajectory step 29, where revisions were probed and then dropped:

```python
for rev in (None, '2022-10-31', '2024-02-22'):
    o = one.load_object(eid, 'trials', revision=rev)
    print('rev', rev, list(o.keys()))
# rev None       ['goCueTrigger_times']
# rev 2022-10-31 ['goCueTrigger_times']
# rev 2024-02-22 ['goCueTrigger_times']
```

iii. `CONVERSION_NOTES.md`, Step 2, "Blocking cache integrity finding"; and the final message: "The
cache must be restaged/restored before conversion can continue without violating the ordered
workflow."

## 10-a. What are the most time-consuming steps of the code?

i. There is no conversion code, so there is no profile and no timing instrumentation (the
instructions required printing timing information and estimating full-run duration; Step 7 of
`CONVERSION_NOTES.md` is an empty template). The only measured cost in the whole session is
incidental: the single `SpikeSortingLoader.load_spike_sorting()` call took ~19.6 s wall time for one
probe, which does point at spike-sorting I/O as the dominant cost — but the agent never drew that
conclusion or recorded it.

ii. No `convert_data.py`. Trajectory step 27 observation: "Script completed / Wall time 19.6
seconds" for the one-probe spike-sorting load (plus atlas downloads).

iii. None recorded.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Not applicable — no loops exist because no conversion code exists. The agent identified no
vectorization opportunities, in its own code or the reference's. The "Code inefficiencies
identified" and "Code speedups added" fields under Step 6 of `CONVERSION_NOTES.md` are left as
`[Note]`.

ii. No `convert_data.py`; nothing exists.

iii. None recorded.

## 10-c. What processing does the code repeat multiple times?

i. Not applicable — no conversion code. Within the exploration itself, the agent did repeatedly
rebuild `ONE` clients and re-run `one.search()` across the three release-table directories in
successive subprocesses (steps 20, 21, 22, 23, 25, 28, 29, 41), and re-attempted essentially the same
failing `load_dataset`/`load_object` call under six different spellings (steps 30–37, 40) without
changing the underlying client mode. That is redundant exploration, not redundant conversion
processing.

ii. No `convert_data.py`. Example of the repeated failing pattern (steps 30, 31, 36):

```python
one.load_dataset(eid, '_ibl_trials.table.pqt')
one.load_dataset(eid, 'alf/_ibl_trials.table.pqt')
one.load_dataset(eid, '_ibl_trials.table.pqt', collection='alf')
```

iii. None recorded.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Not applicable — no conversion code, so nothing is computed and nothing is discarded. (The
reference solution's answer to this question is also "N/A".) The nearest thing to wasted work in the
session is that `SpikeSortingLoader.load_spike_sorting()` was called with default attributes,
pulling `spikes.amps`, `spikes.depths` and `spikes.templates` that a conversion would not need — the
human reference avoids this by setting `bio.SPIKES_ATTRIBUTES = ['clusters', 'times']`. The agent did
not notice or comment on this.

ii. No `convert_data.py`; nothing exists.

iii. None recorded.
