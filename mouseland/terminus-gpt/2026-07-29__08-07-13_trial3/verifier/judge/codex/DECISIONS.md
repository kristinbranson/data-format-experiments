# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI never reached a concrete loading design. The closest recorded decision is in `CONVERSION_NOTES.md`: behavior was thought to live in `data/beh/*.npy` and neural data in `data/spk/*_neural_data.npy`. It did not document the master index `Imaging_Exp_info.npy`, the per-session retinotopy files, or the reference solution's behavior-file grouping and per-session spike/retinotopy reads.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. `CONVERSION_NOTES.md` leaves `Step 5: Mapping Planning` as `NOT STARTED` and `Step 6: Script Development` as `NOT STARTED`. The trajectory ends with repeated messages that the shell was stuck in continuation mode and no further commands could execute.

## 1-b. How are the data split into subjects?

i. No subject-splitting logic was implemented. The notes do record `Subjects | 19` and session counts per animal, which implies the AI inferred subject identity from session/file naming, but it never documented a rule equivalent to the reference solution's grouping by `mname` from the master index.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. The only support is the Step 2 notes listing 19 subjects and per-subject session counts. No mapping or assembly code was written.

## 1-c. How are the data split into sessions?

i. The AI partially identified that neural files are session-wise and named `<session_id>_neural_data.npy`, and that behavior dictionaries are keyed by session IDs. It never specified the reference session definition `(mname, datexp, blk)`, never handled duplicate listings across experiment types, and never implemented session iteration.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. This comes from the Step 2 notes only. The reference-style session bookkeeping was never planned or coded.

## 1-d. How are the data split into trials?

i. No trial-splitting decision was made. The notes list `ntrials` and several trial-level variables, but they explicitly say exact trial definition and filtering still had to be extracted.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. The Step 3 notes say: `Trials are defined by corridor traversals ... exact trial filtering still to be extracted.` The agent never progressed to Step 5 mapping or Step 6 implementation.

## 1-e. How are trials filtered based on quality controls?

i. No trial-quality-control rule was decided. The AI never discovered or documented the reference solution's removal of empty trials and trials above the global 99th-percentile length threshold.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. The notes explicitly leave trial curation unresolved, and no later artifact fills it in.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI identified only part of the raw source: it noted that each neural file is an object with key `spks` whose value is a list. It did not identify the retinotopy `iarea` variable as part of the neural/region derivation.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. Step 2 notes say the neural files are dicts with key `spks`. Step 4 adds a tentative interpretation that `spks` is a list of three area/plane-specific arrays, but there is no mention of `iarea` or any implementation.

## 2-b. How is the `neural` data processed?

i. The AI never reached a stable processing decision. Its notes are internally inconsistent: Step 2 says the data are "spike-based" and "not calcium imaging", while Step 3 says the paper uses deconvolved fluorescence traces, and Step 4 reinterprets `spks` as three area/plane-specific neuron-by-time arrays. It never decided on the reference solution's actual processing, which is simply concatenation across planes and slicing per trial.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. The trajectory and notes show unresolved confusion about what `spks` contains, and Step 6 script development never started.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neural QC rule was implemented. The AI tentatively wrote that there was "No obvious ΔF/F computation or spike-sorting quality filtering" and left exact inclusion/exclusion criteria unresolved. It never recovered the reference solution's visual-area filter based on `iarea`.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. Step 1 notes speculate about missing QC filters; Step 3 notes say exact neuron curation rules still had to be extracted. No later decision exists.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The only partial decision is very general: the AI noted that trial-start alignment seemed central and the decoder task specified corridor entry as the alignment event. It never translated that into a concrete per-trial neural window such as the reference solution's `ft_trInd & ft_CorrSpc` frame selection.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. The Step 1 notes mention "trial-start alignment", but Step 5 mapping never began and no alignment code was written.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No final time-bin decision was made. The AI only wrote that the neural data seemed "frame-based" and that the exact time step was still pending.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. Step 3 notes say `Neural data time bin | frame-based deconvolved traces (exact dt pending)`. There is no resolution of this question in the trajectory.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. No mapping decision was recorded. The notes list some sound-related variables such as `SoundTime` and `SoundTimeDelay`, but the AI never identified the reference variables `SoundFr` and `ft`.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. Step 2 only inventories available behavior fields. Step 5 mapping never started.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. No processing decision was made. The AI never described interpolation onto the frame-time axis or subtraction from per-frame time.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. There is no recorded design beyond listing possible raw variables in the notes.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. No alignment rule was implemented. The AI never specified any shared frame window for this input and the neural data.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. The agent never reached variable-level temporal alignment decisions.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. No decision was recorded. The AI never documented the reference solution's derivation from per-mouse session ordering in the session IDs/index.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. Although the notes count sessions per subject, they never define a day-of-training variable.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. No processing decision was made. The AI never described counting sessions per mouse in chronological order or broadcasting the value across frames.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. Step 5 mapping is untouched, so this variable was never planned in detail.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. No decision was recorded. The notes mention trial start times generally (`Trial_start_time`), but the AI never identified the reference variables `StartFr` and `ft`.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. The notes inventory fields but do not map them to decoder inputs.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. No processing decision was made. The AI never described interpolating the start frame onto frame times and subtracting it from each neural bin time.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. No mapping or implementation exists for this variable.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. No alignment rule was implemented. The AI never specified using the same per-trial frame window as the neural data.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. The notes never reach this level of detail.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. The AI did not make an explicit mapping decision, but it did at least inventory `isRew` as an available trial-level variable. It never stated that this should be the decoder input for reward availability.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. The only evidence is the Step 2 variable list containing `isRew`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No processing decision was made. The AI never stated whether the value should be used directly, cast to binary, or broadcast across time bins.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. No mapping or implementation exists for this variable.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The AI did not make an explicit mapping decision, but it did inventory `WallName` and `TrialStim` as trial-level fields. It never resolved which one should drive the decoder output or how swap sessions should be handled.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. The Step 2 notes list both variables; the reference-style choice between them was never made.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. No processing decision was made. The AI never defined the four-category texture collapse, categorical encoding, or per-trial broadcasting used by the reference solution.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. There is no mapping plan or implementation for this output.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. No decision was recorded. The notes never identify `LickFr` specifically.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. The AI recognized licking as behavior of interest in the reference code, but not the raw variable needed for conversion.

## 8-b. What processing is involved in computing `output` *Licking*?

i. No processing decision was made. The AI never described constructing a binary per-frame lick series from lick-frame events.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. No variable-level decoder-output mapping was written.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. No alignment rule was implemented. The AI never described indexing the lick time series with the same per-trial frame window as the neural data.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. The notes and trajectory never reach this detail.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. The AI did not make an explicit mapping decision, but it did inventory `ft_Pos` and `ft_PosCum` as frame-level variables, which are plausible position sources.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. The Step 2 notes list the raw variables only; no decision chooses between them.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. No processing decision was made. The AI never described converting decimeters into four 1 m bins or using the within-corridor frames only.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. There is no mapping or implementation for this output.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. No thresholding/categorization decision was recorded.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. The agent never reached the discretization stage described in Step 5.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. No alignment rule was implemented. The AI never specified using the same per-trial frame window as the neural data.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. Alignment remained at the level of a vague "trial-start alignment" note.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. The AI did not make an explicit mapping decision, but it did inventory `ft_RunSpeed` as a frame-level variable.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. The Step 2 notes list the raw variable but never define the decoder output.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. No processing decision was made. The AI never described quartile binning, rank-based handling of ties, or session-specific thresholds over kept frames.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. No discretization plan was documented.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. No thresholding/categorization decision was recorded.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. The agent never reached the required discretization stage.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. No alignment rule was implemented. The AI never specified using the same per-trial frame window as the neural data.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. There is no output-alignment implementation because no script was produced.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. No concrete data-cleaning policy was implemented. The AI never documented truncating behavior streams to imaged frames, dropping out-of-range licks, or removing empty trials.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. The notes leave trial curation unresolved and never discuss imaged-frame truncation.

## 12-a. What are the most time-consuming steps of the code?

i. No completed code exists, so there is no real runtime profile. The Step 6 section has placeholders for inefficiencies and speedups, but they were never filled in.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. Because the script was never written or run, any claim about bottlenecks would be speculative.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. No implementation exists, so no actual loops were identified for vectorization.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. Step 6 remains a blank template with `Code inefficiencies identified: [Note]`.

## 12-c. What processing does the code repeat multiple times?

i. No repeated processing was documented, because there is no completed conversion script.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. The agent never progressed beyond exploratory notes.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No unnecessary downstream-discarded processing was documented, because there is no completed conversion script.

ii. `convert_data.py` was never created, so there is no AI code snippet for this decision point.

iii. The script-development and profiling stages were never reached.
