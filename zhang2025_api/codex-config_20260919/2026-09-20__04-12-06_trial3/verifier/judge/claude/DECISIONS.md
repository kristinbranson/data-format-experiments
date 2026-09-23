# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI never wrote a conversion script. During Step 1 (reference code exploration), the AI identified that the reference code uses the ONE API with `SpikeSortingLoader` and `SessionLoader` to load data per session, keyed by `eid`. During Step 2 (dataset exploration), the AI attempted to use the ONE API in `mode='local'` to explore the data, but was unable to load trial tables (`_ibl_trials.table.pqt`) through `ONE.load_object`. The AI got blocked here and never progressed to writing conversion code.

ii. No conversion code was written. The AI's exploration used:
```python
ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')
```
which failed to access trial tables.

iii. From CONVERSION_NOTES.md Step 2: "ONE metadata says `_ibl_trials.table.pqt` exists for 459 sessions, but the files are not reachable through ONE. `SessionLoader.load_trials()` returned only one column (`goCueTrigger_times`) for tested sessions." The AI concluded this was a "source-cache staging problem" and halted.

## 1-b. How are the data split into subjects?

i. No decision was implemented. During Step 1, the AI noted the reference code uses `one.search()` which returns subject names alongside each `eid`, and that subjects are grouped accordingly. No code was written.

ii. No code was written.

iii. The AI documented in Step 1 that the reference code groups sessions by subject, but never implemented this.

## 1-c. How are the data split into sessions?

i. No decision was implemented. The AI understood from Step 1 that a session corresponds to a single `eid` returned by `one.search()`, but never wrote code.

ii. No code was written.

iii. The AI noted the ONE API returns sessions individually, but did not implement session handling.

## 1-d. How are the data split into trials?

i. No decision was implemented. The AI understood from Step 1 that the trials table has one row per trial, but could not load the trials table during Step 2.

ii. No code was written.

iii. The AI was blocked by inability to load trial data.

## 1-e. How are trials filtered based on quality controls?

i. No decision was implemented. During Step 1, the AI identified the reference code's trial filtering: reaction time bounds (0.08-2.0 s), exclusion of no-choice trials, and other quality checks from `load_trials_and_mask`. No filtering code was written.

ii. No code was written.

iii. From CONVERSION_NOTES.md Step 1, the AI documented: "Excludes RT <0.08 s or >2 s, missing required events, no-choice trials, and (in `prepare_data`) trial duration >10 s."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. No decision was implemented. The AI identified in Step 1 that neural data comes from `spikes.times` and `spikes.clusters` loaded via `SpikeSortingLoader`, and verified one probe could be loaded (80M spikes, 1557 clusters). No conversion code was written.

ii. No code was written.

iii. From CONVERSION_NOTES.md Step 1: "Uses `ONE.eid2pid`, `SpikeSortingLoader`, merges all probes."

## 2-b. How is the `neural` data processed?

i. No decision was implemented. The AI identified in Step 1 that the reference uses 20 ms bins over [-0.5, 1.5] s from stimulus onset (100 bins), spike counts per bin, with probes merged per session. No processing code was written.

ii. No code was written.

iii. From CONVERSION_NOTES.md Step 1: "Stimulus-aligned spike counts in fixed bins; caching parameters use [-0.5,1.5) s and 20 ms (100 bins)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No decision was implemented. The AI identified in Step 1 that the reference caching call uses `qc=None` (no quality filtering), but noted the data paper's "stringent quality control" uses `label >= 1`. The AI flagged this discrepancy but never resolved it or wrote code.

ii. No code was written.

iii. From CONVERSION_NOTES.md Step 1: "`qc=None` keeps all clusters; optional `qc=1` would retain labels >=1, but caching calls the default."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No decision was implemented. The AI identified stimulus onset alignment from the reference code, but never wrote alignment code.

ii. No code was written.

iii. From CONVERSION_NOTES.md Step 1: "stimulus onset alignment, a 2 s window from -0.5 to +1.5 s."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No decision was implemented. The AI identified 20 ms bins from the reference code.

ii. No code was written.

iii. From CONVERSION_NOTES.md Step 1: "20 ms bins."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. No decision was implemented. No code was written.

ii. No code was written.

iii. No justification provided; the AI never reached the mapping/implementation phase.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No decision was implemented.

ii. No code was written.

iii. No justification provided.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. No decision was implemented.

ii. No code was written.

iii. No justification provided.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. No decision was implemented.

ii. No code was written.

iii. No justification provided.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. No decision was implemented.

ii. No code was written.

iii. No justification provided.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. No decision was implemented. The AI noted in Step 1 that "Choice values in native IBL are typically -1/+1 (with 0 no-choice); the task explicitly remaps left/right to 0/1."

ii. No code was written.

iii. Observation only from reference code exploration; no implementation.

## 5-b. What processing is involved in computing `output` *Choice*?

i. No decision was implemented.

ii. No code was written.

iii. No justification provided.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. No decision was implemented.

ii. No code was written.

iii. No justification provided.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. No decision was implemented.

ii. No code was written.

iii. No justification provided.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. No decision was implemented. The AI noted the reference uses `SessionLoader` wheel data with absolute smoothed velocity.

ii. No code was written.

iii. From CONVERSION_NOTES.md Step 1: "wheel speed is absolute smoothed wheel velocity."

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. No decision was implemented. The AI noted the reference interpolates behavioral traces and that the task requires 3 categorical bins (unlike the reference which uses continuous regression).

ii. No code was written.

iii. From CONVERSION_NOTES.md Step 1: "The reference paper treats wheel speed and whisker motion energy as continuous regression targets, whereas this task explicitly requires three categorical bins."

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. No decision was implemented.

ii. No code was written.

iii. No justification provided.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. No decision was implemented.

ii. No code was written.

iii. No justification provided.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. No decision was implemented. The AI noted "whisker motion energy prefers left camera then right fallback" from the reference code.

ii. No code was written.

iii. From CONVERSION_NOTES.md Step 1 only.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. No decision was implemented.

ii. No code was written.

iii. No justification provided.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. No decision was implemented.

ii. No code was written.

iii. No justification provided.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. No decision was implemented.

ii. No code was written.

iii. No justification provided.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. No decision was implemented. The AI's own experience was being blocked by what it described as a "source-cache staging problem" with inaccessible trial tables. It never reached the point of handling data edge cases.

ii. No code was written.

iii. No justification provided.

## 10-a. What are the most time-consuming steps of the code?

i. No conversion code was written, so no performance analysis was possible.

ii. No code was written.

iii. No justification provided.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. No code was written.

ii. No code was written.

iii. No justification provided.

## 10-c. What processing does the code repeat multiple times?

i. No code was written.

ii. No code was written.

iii. No justification provided.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No code was written.

ii. No code was written.

iii. No justification provided.
