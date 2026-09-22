# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent intended to use a local IBL ONE cache and brainbox loaders. It instantiated `ONE(..., mode='local')`, searched release tables, used `SessionLoader` for trials and `SpikeSortingLoader` for ephys, but stopped after concluding the trial tables were inaccessible. It never loaded all subjects, sessions, or trials.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The instructions required ONE/brainbox, and the notes justify this choice as compliant. The agent reported 459 registered trial-table sessions but treated `ALFObjectNotFound` as a cache-staging blocker. The reference instead configures the Alyx-backed ONE client against the cached REST metadata and successfully resolves the revisions; thus the agent’s blocker diagnosis prevented the required complete load.

## 1-b. How are the data split into subjects?

i. No conversion split was implemented. During exploration, the agent used `ONE.get_details(eid)['subject']` to count unique subjects in each release (143 composite, 139 in 2025_Q3, 115 in 2022_Q4), but created neither `subjects` nor `subject_idx`.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The notes recognized that ONE session details carry the subject identifier, consistent with the reference conceptually, but the mapping and assembly were never implemented.

## 1-c. How are the data split into sessions?

i. No conversion split was implemented. The agent treated each ONE EID as a session and counted EIDs per release, but produced no session-indexed `neural`, `input`, or `output` lists.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The exploratory model of one EID per session matches the reference, but work stopped before session selection, processing, and assembly.

## 1-d. How are the data split into trials?

i. No trial splitting was implemented. The agent attempted `SessionLoader.load_trials()` and `ONE.load_object(..., 'trials')`, observed only `goCueTrigger_times` or errors, and stopped.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The agent correctly expected the trials table to provide one row per trial, but it never accessed the table via the working reference connection setup and never emitted per-trial arrays.

## 1-e. How are trials filtered based on quality controls?

i. No filter was implemented. The notes extracted the reference rules: reaction time 0.08–2 s, required-event presence, no-choice exclusion, trial duration ≤10 s, and behavior-presence masks.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The justification came from `load_trials_and_mask`, `prepare_data`, and `align_spike_behavior`. However, the agent did not settle or apply the exact target mask; notably the reference solution uses reaction-time, choice/prior validity, and wheel/camera coverage.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. No neural output was created. The notes identify spike times, spike cluster assignments, and merged cluster/channel metadata loaded by `SpikeSortingLoader`; one probe load yielded 80,867,860 spikes and 1,557 clusters.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. This source-variable identification is consistent with the reference, but remains exploratory rather than a conversion decision.

## 2-b. How is the `neural` data processed?

i. No processing was implemented. The agent planned to merge probes, stable-sort spikes, bin stimulus-aligned activity over [-0.5, 1.5) s in 20 ms bins, and transpose reference trial×time×neuron arrays to neuron×time.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The plan follows the methods repository and target shape, but never specifies/implements the reference solution’s conversion from counts to Hz or builds trial matrices.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No unit filter was chosen or implemented. The notes explicitly left a discrepancy unresolved: reference caching uses `qc=None` (all clusters), while an optional `qc=1` and the data-paper curation suggest good-unit filtering.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The human solution resolves this by keeping cluster label ≥1 and excluding Beryl `void`. The agent deferred the decision to later steps that never occurred.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No alignment was implemented. The notes planned alignment to `stimOn_times` with a window from -0.5 to +1.5 s.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The chosen event/window matches the reference, but no spike timestamps were shifted/binned per onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent planned 20 ms bins and 100 bins across a 2 s window; no converted data exists, so no temporal rebinning was actually applied.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The planned resolution matches the reference paper/code, but there is no implementation or metadata.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. No input was created. The notes identify `stimOn_times` as the alignment event and imply a fixed relative-time grid.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. This matches the reference source conceptually, but was not converted.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No computation was implemented. The exploratory plan implies constructing the 100-point relative-time grid for the -0.5 to +1.5 s window.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The reference uses 20 ms bin centers; the agent did not explicitly decide centers versus edges or generate the array.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. No alignment was implemented. The plan was for the time input and neural bins to share stimulus-onset alignment and the same 20 ms grid.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. This is directionally consistent with the reference but unexecuted.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. No mapping was made. The notes mention `probabilityLeft` as available/required trial data but never state that block boundaries should be inferred from changes in it.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The reference derives blocks from `probabilityLeft`; this required decision is absent.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. No computation was decided or implemented.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The reference counts trials from zero within each pre-filter block. The agent never addressed whether counting occurs before filtering, a consequential detail.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. No output was created. The notes identify native IBL `choice` values (-1/+1, with 0 for no choice).

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The source variable matches the reference.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The notes planned to exclude no-choice trials and remap left/right to 0/1, but no mapping code was written.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The plan matches the required/reference recoding (+1 left→0, -1 right→1), though the precise mapping is not written in the notes and no output exists.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. No output was created. The agent identified `probabilityLeft` among the required trial attributes.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The raw source matches the reference.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. No processing decision or code was produced for mapping 0.2, 0.5, and 0.8 to classes 0, 1, and 2.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The instruction explicitly supplied the mapping, so omitting it leaves this decision incomplete.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. No output was created. The notes say `SessionLoader` supplies wheel speed as the absolute value of smoothed wheel velocity, sourced from wheel position/timestamps.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. This identifies the same source and derived quantity as the reference.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. No processing was implemented. The notes planned absolute smoothed wheel velocity and linear interpolation of continuous behavior to the neural time grid.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. This broadly matches the reference loader/interpolation behavior, but no trial traces were formed.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The notes only say the task requires three categorical bins; no thresholds, percentile scope, or digitization rule were selected.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The reference uses session-wide 33rd/67th percentiles. The essential decision is absent.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. No alignment was implemented. The notes planned behavior interpolation around stimulus onset and removal of trials failing behavior-presence masks.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The reference interpolates wheel speed to the same neural bin centers. The agent did not define or execute the exact sampling grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. No output was created. The notes identify camera ROI motion energy and camera timestamps, preferring left camera with right-camera fallback.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. This matches the reference source/view rule.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. No processing was implemented. The notes planned linear interpolation of the continuous motion-energy trace.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The interpolation concept matches the reference, but no session/trial arrays were formed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The notes only acknowledge the required three bins; they do not choose thresholds or implement categorization.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The reference uses session-wide 33rd/67th percentiles. This decision is absent.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. No alignment was implemented. The notes planned event-relative interpolation and behavior-presence filtering.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The reference samples motion energy at the same bin centers as neural activity. Exact implementation is absent.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent treated inaccessible trials tables as a hard blocker and stopped. It noted that the reference drops trials failing required-event or behavior-presence masks, but implemented no missing-data policy.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The reference handles missing streams/trials locally (coverage masks, skipped unreleased probes, and dropping sessions with fewer than two trials or zero units) rather than abandoning conversion. The agent’s diagnosis and response are incomplete.

## 10-a. What are the most time-consuming steps of the code?

i. There is no conversion code to profile. The agent observed that a test probe contains 80,867,860 spikes, but did not identify measured runtime bottlenecks.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The reference identifies spike-sorting disk I/O as dominant. The agent did not reach critical review or timing.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. No conversion loops exist and the notes leave the inefficiency section as a placeholder.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The reference identifies per-trial spike binning and behavior interpolation loops as candidates. The agent made no assessment.

## 10-c. What processing does the code repeat multiple times?

i. No conversion code exists, and the agent made no repeated-processing assessment.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The human reference reports none material; the agent nevertheless failed to answer the requested review question.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No conversion code exists, and the agent made no discarded-processing assessment.

ii. No snippet is available: the agent never created `/app/convert_data.py`. The trajectory contains only exploratory commands such as:

```python
one = ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
one.load_object(eid, 'trials', ...)
```

iii. The human reference reports none; the agent did not reach or document this review.


