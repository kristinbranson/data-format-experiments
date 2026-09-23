# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI never created `/app/convert_data.py`. In its notes it said the data should be accessed through the local ONE cache using `ONE.search`, `ONE.list_datasets`, `ONE.load_object`, `SessionLoader`, and `SpikeSortingLoader`. It stopped after concluding that the trials tables were inaccessible through ONE.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The justification in `CONVERSION_NOTES.md` is that the task required the ONE API and brainbox loaders. The trajectory ends with a block report saying `_ibl_trials.table.pqt` could not be loaded via ONE, so the agent did not proceed to implementation.

## 1-b. How are the data split into subjects?

i. No final implementation was written. The notes only show that the agent counted subjects from the release tables and treated subject identity as metadata available through ONE rather than something to infer from paths.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The notes list subject counts for the composite and release-specific tables, which implies the agent planned to use ONE session metadata for subject grouping, but it never documented final assembly logic such as `subjects` or `subject_idx`.

## 1-c. How are the data split into sessions?

i. No final implementation was written. The notes treat sessions as the ONE release-table session units, identified by `eid`.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The justification is implicit in the Step 2 notes: the agent explored the release tables and ONE metadata and reported session counts, but it never wrote code that enumerated or stored sessions.

## 1-d. How are the data split into trials?

i. No final trial-splitting logic was written. The agent expected trials to come from the ONE/brainbox trials table, but it stopped when that table could not be loaded.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The trajectory records repeated `ALFObjectNotFound` failures for the trials object and then a block message saying choice, stimulus onset, prior, and behavior-alignment data could not be loaded.

## 1-e. How are trials filtered based on quality controls?

i. The only documented plan is from Step 1 reference-code notes: the reference `load_trials_and_mask` excludes reaction times below 0.08 s or above 2 s, missing required events, no-choice trials, and trial duration above 10 s in `prepare_data`. The AI did not document a final task-specific trial mask.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The justification comes from the Step 1 table in `CONVERSION_NOTES.md`. The agent never reached a point where it reconciled those reference notes with the downstream decoder task or with the missing wheel/camera coverage checks used by the human reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. No final code was written, but the notes identify extracellular spike data loaded per probe through `SpikeSortingLoader` and reference `load_spiking_data`. The AI therefore appears to have intended to derive neural data from spike-sorting outputs rather than any analog or imaging signal.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The justification is the Step 1 exploration table, which names `prepare_data`, `load_spiking_data`, `merge_probes`, and `SpikeSortingLoader` as the relevant reference loading path for electrophysiology data.

## 2-b. How is the `neural` data processed?

i. The notes say the core reference parameters are stimulus-onset alignment, a 2 s window from -0.5 s to +1.5 s, and 20 ms bins. They also note that probes are merged per session and that the requested pickle would transpose trial arrays to neuron by time.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The justification is the Step 1 reference summary: `bin_spiking_data` / `get_spike_data_per_interval` were identified as the reference processing path, with fixed-bin spike counts around stimulus onset.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI did not make a final neuron-QC decision. Its notes explicitly say the reference caching call uses `qc=None`, while an optional `qc=1` would retain clusters with labels `>= 1`, and that this tension with the data-paper curation would be reconciled later.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The justification is quoted directly in Step 1 notes: the agent recognized that QC was unresolved and deferred the decision to later workflow steps that it never reached.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The notes clearly state that the intended alignment event was stimulus onset, using a window from -0.5 s to +1.5 s.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The justification is explicit in Step 1: “Core reference parameters are stimulus onset alignment, a 2 s window from -0.5 to +1.5 s, and 20 ms bins.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI’s notes specify 20 ms bins over the 2 s stimulus-aligned window, which would give 100 time bins. No further temporal rebinning was documented.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The justification is the Step 1 reference-code summary, which explicitly identifies the 20 ms binning parameters used by the reference pipeline.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. No final implementation was written. From the notes, this input was only tied generally to stimulus-onset alignment and the 20 ms trial grid; the raw trials field was not named in the AI’s own decision document.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The justification is only implicit: the notes repeatedly refer to stimulus-onset alignment but stop short of documenting the exact source field for this decoder input.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No final processing logic was documented beyond using the stimulus-aligned 20 ms time grid required by the reference window.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The Step 1 notes identify the alignment window and bin size, but the agent never documented whether this input would be bin centers, edges, or some other representation.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The only documented intention is that this input would live on the same stimulus-aligned 20 ms grid as the neural data.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The justification follows from the notes about a shared stimulus-aligned 2 s window, but the agent never implemented or explicitly described the final alignment mechanism.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. No final implementation was written. The notes only say the reference `bin_behaviors` constructs a probability-left block variable, implying this input would come from block structure in the trials table, but they do not name the exact raw field in a final decision.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The justification is limited to the Step 1 exploration table, which identifies `bin_behaviors` but does not record the full derivation of trial-within-block for this task.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. No final processing rule was documented. The agent never wrote whether it would recover blocks from `probabilityLeft`, whether counting would start at zero, or whether dropped trials would still advance the block counter.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The agent stopped before Step 5 mapping and Step 6 script development, so this decision remained unstated.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The notes explicitly say that native IBL choice values are typically `-1/+1` with `0` for no choice, and that the task requires remapping left/right to `0/1`.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The justification appears in the Step 1 notes, which distinguish the native IBL coding from the decoder task’s required coding.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The only documented processing is recoding native choice values to the task’s binary convention. The notes also imply no-choice trials would be excluded because that is how the reference trial mask behaves.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The justification comes from the Step 1 note about remapping choice to `0/1` and from the Step 1 summary of `load_trials_and_mask`, which excludes no-choice trials.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. No final implementation was written. The notes say the reference `bin_behaviors` constructs a probability-left block variable and the task requires the prior-probability output, so the AI appears to have intended to use the trials-table prior field, but it never documented the exact final derivation.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The justification is indirect and incomplete: the Step 1 notes mention “probability-left block” and the task specification, but there is no completed mapping section.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. No final processing rule was documented beyond the task requirement that the three prior values become decoder categories.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The agent never reached the mapping or implementation steps where it would have had to commit to the `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2` recoding.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The notes state that `load_target_behavior` uses brainbox `SessionLoader` and that wheel speed is absolute smoothed wheel velocity. That implies derivation from wheel position/timestamp data loaded through `SessionLoader`.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The justification is explicit in the Step 1 table entry for `load_target_behavior`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The notes say wheel speed is absolute smoothed wheel velocity and that continuous behavioral traces are linearly interpolated. No final task-specific processing beyond that was implemented.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The Step 1 notes identify both the upstream wheel-velocity computation path in `SessionLoader` and the linear interpolation behavior used by the reference continuous-behavior loaders.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI did not document any thresholding rule. It only noted that the task required three categorical bins rather than the continuous regression target used by the reference paper.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The justification is simply that the workflow ended before mapping or code generation, so the agent never chose percentile thresholds or any other discretization rule.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The only documented intent is that continuous behavioral traces would be linearly interpolated on the same stimulus-aligned time base used by the neural data.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. This follows from the Step 1 notes about stimulus-onset alignment, 20 ms bins, and interpolation of continuous behavior, but the final alignment rule was never implemented.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The notes say `load_target_behavior` loads whisker motion energy through `SessionLoader` and prefers the left camera with right-camera fallback.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The justification is the Step 1 summary of `load_target_behavior`, which explicitly mentions left-camera preference and right fallback.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The notes only document that continuous behavioral traces are linearly interpolated; no additional whisker-specific filtering or normalization decision was written.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The Step 1 notes characterize the reference continuous-behavior processing at a high level but stop short of a final task-specific whisker pipeline.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The AI did not document any thresholding rule. It only recognized that the task required a 3-bin categorical output instead of the reference continuous target.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The agent never reached the planning or implementation stage where category thresholds would have been chosen.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The only documented intent is that whisker motion energy would be interpolated onto the same stimulus-aligned time base as the neural data.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The justification follows from the same Step 1 notes used for wheel speed: stimulus-onset alignment plus interpolation of continuous behavioral traces.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The only concrete handling decision was to stop when trials tables were inaccessible through ONE rather than attempt a non-ONE workaround. No policy for smaller issues such as partial trial coverage, unreleased probes, or missing continuous-data segments was implemented.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The Step 2 notes frame the problem as a cache-integrity block and say conversion cannot continue until the registered trial tables are accessible through ONE.

## 10-a. What are the most time-consuming steps of the code?

i. No conversion code was produced, so there is no implemented performance profile. The only timing-relevant observation in the notes is that a test spike-sorting load succeeded and was large.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The agent stopped before Step 6 script development and never ran sample or full conversion, so there is no empirical bottleneck analysis.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. No conversion code was produced, so there are no implemented loops to evaluate.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The notes never progressed beyond generic workflow requirements about vectorization and efficiency.

## 10-c. What processing does the code repeat multiple times?

i. No conversion code was produced, so no repeated processing was identified.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The agent never reached an implementation stage where repeated work could be inspected.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No conversion code was produced, so no unnecessary discarded processing was identified.

ii. No code snippet is available from `/app/convert_data.py`; the file was never created.

iii. The agent never wrote or ran the conversion pipeline, so there was nothing to audit here.
