# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent never implemented a loader. From `CONVERSION_NOTES.md` and the trajectory, its partial plan was to treat `data/spk/` as the canonical session list, load behavior from `data/beh/` condition files, and match behavior keys back to the spike-session ids, including handling `_swap` variants by base session id. It did not document using `beh/Imaging_Exp_info.npy`, did not document loading retinotopy, and never wrote executable loading code.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. This comes from Step 2 notes (`data/spk/` and `data/beh/` only) and the attempted Step 4 edit in the trajectory: “Use neural session list as canonical sessions and map behavior by base session ID, handling `_swap` variants separately.”

## 1-b. How are the data split into subjects (mice)?

i. Subjects were partially inferred from spike-file names. The notes count 19 subjects and list sessions per subject from the prefixes of the 89 spike filenames. No final `subjects` or `subject_idx` construction was implemented.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. The Step 2 notes explicitly summarize 19 subjects and per-subject session counts from `data/spk/` filenames rather than from the master experiment index.

## 1-c. How are the data split into sessions?

i. Sessions were partially treated as the 89 `*_neural_data.npy` files in `data/spk/`, with behavior matched back to those session ids. The agent also noted that extra behavior keys came from swap variants and aggregate labels. It never implemented the reference session construction from `(mname, datexp, blk)` via `Imaging_Exp_info.npy`.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. The trajectory records the Step 4 note: all 89 spike sessions are present in the behavior base-session keys, and the neural session list should be treated as canonical.

## 1-d. How are the data split into trials?

i. The agent identified candidate trial-boundary variables but never chose a trial-splitting procedure. It inspected `ntrials`, `ft_trInd`, `StartFr`, `GrayFr`, and `EndFr`, but it did not document the reference rule of using frames where `ft_trInd == trial` and `ft_CorrSpc` is true.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. The only evidence is raw-data inspection in the trajectory; Step 5 mapping planning was never completed.

## 1-e. How are trials filtered based on quality controls?

i. No trial-quality filter was specified. The closest partial idea was trimming behavior to the neural frame count because of a possible one-frame mismatch, but no explicit per-trial QC rule was documented or implemented.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. The notes and trajectory never reach a final trial-filtering decision.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The agent correctly identified `spks` in each session’s neural `.npy` dict as the source of the neural data. It also inferred that `spks` is a list of three large arrays, likely corresponding to planes or area groups.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. Step 2 notes say each neural file is a dict with key `spks`, and the attempted Step 4 edit interprets `spks` as “three area/plane-specific neuron-by-time arrays.”

## 2-b. How is the `neural` data processed?

i. No final neural-processing pipeline was written. The agent’s partial understanding was that the three `spks` arrays should be interpreted as neuron-by-time arrays and that neural and behavior streams may need trimming to a common frame length. It never documented concatenation across planes, per-trial extraction, padding, or output dtype.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. The Step 4 discrepancy note in the trajectory discusses `spks` interpretation and a possible one-frame alignment trim, but there is no Step 5 mapping or Step 6 code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neural quality-control filter was implemented. The agent did not document the retinotopy file, did not assign brain regions, and did not specify any keep/drop rule for neurons.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. Neither `CONVERSION_NOTES.md` nor the late trajectory contains a neuron-filtering rule; the work stopped before mapping or script development.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial neural alignment was not decided. The only partial statement was that neural arrays and behavior `ft` are aligned up to a possible one-frame offset and should be trimmed to a common length. The agent never defined corridor-entry alignment or a fixed per-trial window.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. This comes from the attempted Step 4 “Neural-behavior alignment” note in the trajectory.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent recognized that the data are frame-based deconvolved traces, but it left the exact time bin unresolved (“exact dt pending”) and never stated whether rebinning would occur.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. Step 3 notes explicitly say “Neural data time bin | frame-based deconvolved traces (exact dt pending).”

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. No final source-variable mapping was documented. The agent inspected both `SoundFr` and `ft`, which are the obvious candidate variables for a frame-aligned time-to-cue signal, but it never committed to them in Step 5 or code.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. The behavior-field dump in the trajectory shows `SoundFr` and `ft` were discovered during dataset inspection.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. No computation was specified for time to sound cue. There is no documented interpolation, subtraction, padding rule, or sign convention.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. The notes never progress past field discovery for this variable.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. No explicit alignment rule was written beyond the generic idea that neural and behavior frames are on a common frame grid after trimming.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. This is only implied by the partial “Neural-behavior alignment” note in the trajectory.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The agent never documented a raw source for day of training. It recorded paper facts about training days, but it did not define day-of-training from session order, session dates, or any other field.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. Step 3 notes summarize the behavioral paradigm, but Step 5 mapping never converts that into a variable definition.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. No processing rule was given for computing day of training. There is no session-ordering logic, per-subject counting, or broadcasting decision.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. No Step 5 mapping or Step 6 code exists for this variable.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. No final source-variable mapping was documented, but the agent did inspect `StartFr` and `ft`, which are the natural candidates for time since trial start.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. The behavior-field inspection in the trajectory lists both `StartFr` and `ft`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. No computation was specified for time since trial start. The agent did not describe interpolation from `StartFr`, subtraction from frame times, sign convention, or padding.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. The work stopped before Step 5 mapping was filled in.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. No explicit alignment rule was documented beyond the generic idea of trimming neural and behavior to a common frame axis.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. Only the partial Step 4 alignment note exists.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. The agent identified `isRew` in the behavior structure, but it never explicitly mapped that field to decoder input reward availability.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. The Step 2 notes list `isRew` among the trial-level behavior variables.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No processing rule was documented. There is no statement that the variable would simply be cast to 0/1 and broadcast over time bins.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. The work never reached explicit variable mapping.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The agent identified `WallName` and also noted `TrialStim` as available trial-level fields, but it never chose which raw field to use as the output stimulus category.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. The Step 2 notes list both `WallName` and `TrialStim` in the behavior payload.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. No stimulus-category processing rule was written. There is no mapping from many wall labels to four categories, and no decision about broadcasting the per-trial label across time.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. The agent never completed Step 5 mapping for visual stimulus.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. The agent discovered several licking-related fields (`LickFr`, `LickTime`, `LickTrind`, `LickPos`) but never chose a final source variable for the output licking signal.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. These fields appear in the behavior-structure dump captured in the trajectory.

## 8-b. What processing is involved in computing `output` *Licking*?

i. No licking-processing rule was written. The agent never specified whether licks would be converted into a framewise binary series, how fractional frame numbers would be handled, or how padding would be encoded.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. There is no Step 5 mapping or Step 6 implementation.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. No explicit licking-alignment rule was written beyond the generic notion of a shared frame grid after trimming.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. Only the partial common-frame alignment note survives in the trajectory.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. The agent identified `ft_Pos` and `VRpos` as available position variables, but it never selected the final raw source for the decoder output position-in-corridor.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. Both fields appear in the raw behavior inspection recorded in the trajectory.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. No position-processing rule was documented. There is no statement about using framewise corridor position, clipping to textured-corridor frames, or padding.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. The notes never progress beyond raw field discovery.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. No categorical thresholding rule was specified. The agent never described the reference four 1 m bins or any alternative binning.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. No Step 5 or Step 6 artifact contains a thresholding decision.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. No explicit position-alignment rule was written beyond the generic possibility of using the trimmed common frame axis.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. This is only indirectly implied by the Step 4 alignment note.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. The agent identified `ft_RunSpeed` as the relevant running-speed variable in the raw behavior stream, but it never completed the mapping.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. The Step 2 notes list `ft_RunSpeed` among the frame-level fields.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. No running-speed processing rule was documented. The agent never described sessionwise quantiles, rank-based quartiles, or padding.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. The work stopped before explicit output processing was designed.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. No thresholding rule was given for running speed. The agent never specified four equal-frequency bins or any other discretization scheme.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. There is no completed Step 5 mapping and no conversion code.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. No explicit alignment rule for running speed was documented beyond the partial common-frame trimming idea.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. Only the trajectory’s partial neural-behavior alignment note exists.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The only concrete data-quality handling the agent documented was a possible trim to a common neural/behavior frame length because one inspected session had 23193 neural frames versus 23194 behavior `ft` frames. It did not document dropping out-of-range licks, removing empty trials, or any other minor-data-error handling.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. This comes from the attempted Step 4 “Neural-behavior alignment” note in the trajectory and the inspected session summary for `TX108_2023_01_05_2`.

## 12-a. What are the most time-consuming steps of the code?

i. The agent never profiled or even implemented the conversion script, so it did not identify actual runtime bottlenecks.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. Work stopped before Step 6; no timing output or conversion run exists.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. No vectorization opportunities were analyzed because no conversion code was written.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. There is no Step 6 implementation section beyond the untouched template placeholders.

## 12-c. What processing does the code repeat multiple times?

i. No repeated processing was identified because the conversion logic was never implemented.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. The notes never reach a concrete algorithm whose repeated work could be assessed.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No unnecessary discarded processing was identified because the conversion logic was never implemented.

ii. No snippet is available from `convert_data.py`. `/app/convert_data.py` is absent in this environment, and the trajectory never shows a successful write to that file.

iii. Again, Step 6 was never reached and `convert_data.py` is missing.
