# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent identified session-wise neural files in `data/spk/` and condition-grouped behavior dictionaries in `data/beh/`, but never decided or implemented a complete loading procedure. It did not document use of `Imaging_Exp_info.npy`, behavior-key construction, retinotopy loading, or deduplication across experiment types.

ii. No executable conversion snippet exists. The notes say the neural files are `<session_id>_neural_data.npy` objects with `spks`, and behavior files are dictionaries keyed by session IDs. The trajectory stopped during consistency checking, before mapping or script development.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 1-b. How are the data split into subjects?

i. The agent counted 19 subjects and inferred subject identities from session/file identifiers, but did not specify or implement `subjects` and `subject_idx`.

ii. No executable conversion snippet exists. Its notes list per-subject session counts and report 19 subjects, but contain no conversion logic.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 1-c. How are the data split into sessions?

i. The agent treated the 89 neural session files as 89 sessions and checked that they overlap behavior base-session identifiers. It did not resolve behavior-condition duplicates into an explicit session enumeration algorithm.

ii. No executable conversion snippet exists. The notes report 89 sessions; trajectory step 313 says all 89 neural sessions occur in the behavior base-session set.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 1-d. How are the data split into trials?

i. No decision was made. The agent found fields such as `ntrials`, `trInd`, `ft_trInd`, and trial timing variables, but never chose a trial window or wrote splitting logic.

ii. No snippet exists; `/app/convert_data.py` was never created.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 1-e. How are trials filtered based on quality controls?

i. No trial-quality filtering decision was made.

ii. No snippet exists; the notes leave trial curation unresolved.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The agent correctly concluded that neural activity comes from the `spks` list in each session's neural `.npy` dictionary and that its arrays are neuron-by-frame deconvolved fluorescence traces. It did not identify `iarea` as the companion source for neuron regions.

ii. No executable conversion snippet exists. Notes state: `Each neural file is a dict with key spks` and later resolve the arrays as neuron-by-frame deconvolved traces.

iii. This conclusion is supported by the exploratory notes, but the agent never implemented it because work stopped before mapping and script development.

## 2-b. How is the `neural` data processed?

i. The agent decided only at an exploratory level that the supplied values are already deconvolved fluorescence and do not require dF/F or spike processing. It never decided trial extraction, concatenation of planes, dtype, or padding policy.

ii. No snippet exists. Notes quote the paper that analyses use deconvolved fluorescence traces.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron filtering decision was made. The notes say exact inclusion/exclusion criteria still needed extraction.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent observed near one-to-one neural/behavior frame alignment and recognized trial-start/corridor alignment as important, but never chose or implemented the per-trial window.

ii. No snippet exists. Trajectory step 313 reports neural 23,193 frames versus behavior 23,194 frames for one session.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No final bin-size or rebinning decision was made; the notes explicitly leave exact neural and behavior time bins pending.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The agent discovered `SoundTime`, `SoundTimeDelay`, and frame timestamps, but did not choose the reference variables `SoundFr` and `ft`.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. No computation or sign convention was decided.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. No alignment decision was made.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The agent noted session dates and per-subject session counts but did not define day of training.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. No counting convention or broadcasting decision was made.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The agent found trial timing and frame-level variables but did not select `StartFr` and `ft`.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. No computation was decided.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. No alignment decision was made.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. The agent identified `isRew` as an available trial-level behavior field and described rewarded versus unrewarded conditions, but never formally mapped it.

ii. No snippet exists; `isRew` appears in the notes' behavior-field inventory.

iii. This conclusion is supported by the exploratory notes, but the agent never implemented it because work stopped before mapping and script development.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No casting or per-frame broadcasting decision was made.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The agent identified `WallName`, `TrialStim`, `StimTrial`, and `StimFrame` as possible stimulus variables but never selected one.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. No mapping from wall variants to four base texture categories was designed.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. The agent recognized licking as behavior relevant to the decoder but did not identify or map `LickFr`.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 8-b. What processing is involved in computing `output` *Licking*?

i. No binary frame-series construction was decided.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. No alignment decision was made.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. The agent identified `ft_Pos` and `ft_PosCum` as available frame-level variables but did not choose one.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. No position transform was implemented or documented.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. No thresholds were decided.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. No alignment decision was made.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. The agent identified `ft_RunSpeed` as the frame-level speed variable but did not formally map it.

ii. No snippet exists.

iii. This conclusion is supported by the exploratory notes, but the agent never implemented it because work stopped before mapping and script development.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. No discretization method was decided.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. No thresholds or tie-handling rule were decided.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. No alignment decision was made.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent noticed an example one-frame mismatch between behavior and neural streams but did not decide how to handle it or any other missing/out-of-range data.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 12-a. What are the most time-consuming steps of the code?

i. No conversion code or runtime profiling was produced. The notes left runtime estimates blank.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. No conversion loops were written or analyzed.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 12-c. What processing does the code repeat multiple times?

i. No decision or analysis was made.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No decision or analysis was made.

ii. No snippet exists.

iii. The agent's notes/trajectory provide no completed justification beyond preliminary exploration; the workflow stopped before mapping planning and script development.

