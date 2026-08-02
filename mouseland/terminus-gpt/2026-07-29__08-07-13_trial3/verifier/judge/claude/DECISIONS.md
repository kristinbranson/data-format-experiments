# Decisions

**Critical finding**: The AI agent (gpt-5.4 via terminus-2) became permanently stuck in a shell heredoc/continuation prompt around trajectory step ~318 out of 3196 total steps. It completed Steps 0-4 (setup, code exploration, data exploration, reference text reading, consistency checking) but **never progressed to Step 5 (mapping planning) or beyond**. No `convert_data.py` was written, no `converted_data.pkl` was produced, and no decoder was trained. The remaining ~2,878 steps consisted of the agent repeatedly noting the shell was stuck and waiting for recovery that never came.

All answers below reflect the agent's **planned understanding** from its exploration phase (documented in CONVERSION_NOTES.md and trajectory messages), since no implementation code exists.

---

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. **No code was written.** The agent identified that neural data is stored as per-session `.npy` files in `data/spk/` (89 files, one per session), and behavioral data is stored in grouped `.npy` files in `data/beh/` organized by experimental condition (e.g., `Beh_sup_train1_after_learning.npy`). Each behavior file is a dict keyed by session IDs. The agent confirmed all 89 neural sessions are represented in the behavior data. However, no loading code was implemented.

ii. No code snippets available — `convert_data.py` was never created.

iii. From CONVERSION_NOTES.md and trajectory: The agent noted that behavior files load as `np.load(path, allow_pickle=True).item()` yielding dicts keyed by session ID, and neural files load similarly yielding a dict with key `'spks'`. The agent identified the need to iterate over all behavior condition files and match sessions to neural files.

## 1-b. How are the data split into subjects (mice)?

i. **No code was written.** The agent identified 19 mice from session ID prefixes (e.g., `TX108`, `DR10`, `VR2`) and confirmed this matches the paper's stated "89 recordings in 19 mice." The agent listed sessions per subject in CONVERSION_NOTES.md but never implemented subject splitting.

ii. No code snippets available.

iii. The agent documented subject counts: DR10:6, DR15:5, LZ13:4, LZ16:4, TX104:2, TX105:5, TX108:7, TX109:6, TX119:8, TX123:8, TX124:3, TX139:2, TX140:1, TX60:5, TX61:5, TX83:3, TX85:2, TX88:6, VR2:7.

## 1-c. How are the data split into sessions?

i. **No code was written.** The agent identified that each neural file corresponds to one session (89 total), with session IDs of the form `<mouse>_<date>_<block>`. Behavior data for each session is found by matching session IDs across behavior condition files. The `Imaging_Exp_info.npy` file provides metadata linking sessions to experimental conditions.

ii. No code snippets available.

iii. The agent confirmed 89 sessions from both neural file count and behavior data keys, consistent with the paper.

## 1-d. How are the data split into trials?

i. **No code was written.** The agent identified that each session's behavior dict contains `ntrials` (scalar), `trInd` (trial index array), `StartFr`/`EndFr`/`GrayFr` (frame indices for trial boundaries), and `ft_trInd` (frame-level trial index assignment). Trial counts range from 84 to 789 per session. The agent did not implement trial extraction from the continuous neural/behavioral streams.

ii. No code snippets available.

iii. From trajectory step 313: "behavior `ft` length is (23194,), a near-perfect match with only a 1-frame difference" vs neural timepoints, confirming frame-level alignment between neural and behavioral data within each session.

## 1-e. How are trials filtered based on quality controls?

i. **No code was written.** The agent noted from the paper/methods that "We only considered timepoints during running for analysis, which removed time periods when the task mice stopped to collect water rewards." The behavior dict has `ft_isMoving` (boolean per frame) for filtering. The agent did not identify or implement specific trial-level exclusion criteria beyond running-only filtering.

ii. No code snippets available.

iii. From CONVERSION_NOTES.md Step 3: "Trial curation rules: Trials are defined by corridor traversals with trial-level timing variables available in behavior dicts; exact trial filtering still to be extracted."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. **No code was written.** The agent correctly identified that neural data comes from `data/spk/<session_id>_neural_data.npy`, each containing a dict with key `'spks'` whose value is a list of 3 numpy arrays (one per visual area/plane). Each array has shape `(n_neurons_in_area, n_timepoints)` with float32 dtype, representing deconvolved fluorescence traces from Suite2p.

ii. No code snippets available.

iii. From trajectory step 310-311: "three arrays per session, almost identical neuron counts, same timepoints within session. Summing the first dimensions yields ~48k-64k neurons per recording, squarely within the paper's reported range."

## 2-b. How is the `neural` data processed?

i. **No code was written.** The agent identified that the raw data consists of deconvolved fluorescence traces (not raw calcium signals), so ΔF/F computation is not needed. The agent noted from methods.txt that "All our analyses were based on deconvolved fluorescence traces" with a decay timescale of 0.75s. No processing pipeline was implemented.

ii. No code snippets available.

iii. From CONVERSION_NOTES.md Step 1: "No obvious ΔF/F computation or spike-sorting quality filtering identified yet from the inspected code; likely preprocessing may already be embedded in packaged data files."

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No code was written.** The agent noted Suite2p performs "ROI detection, cell classification, neuropil correction and spike deconvolution" but did not identify specific neuron quality filtering criteria from the reference code or paper. No filtering was implemented.

ii. No code snippets available.

iii. From CONVERSION_NOTES.md Step 3: "Neuron curation rules: Paper methods mention Suite2p processing, ROI detection, cell classification, neuropil correction, and spike deconvolution; exact inclusion/exclusion criteria still to be extracted."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. **No code was written.** The instructions specify alignment to "trial start (corridor entry)." The agent identified `StartFr` (start frame per trial) and `EndFr` (end frame per trial) in the behavior dict as the relevant alignment variables. The agent confirmed neural timepoints match behavior frame count (within 1 frame). No alignment code was implemented.

ii. No code snippets available.

iii. From trajectory step 313: "neural arrays have shape (22851, 23193) for each of 3 areas, while behavior ft length is (23194,), a near-perfect match with only a 1-frame difference."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **No code was written.** The agent identified frame-based temporal resolution from the `ft` (frame times) array but did not compute the actual time bin size or implement any rebinning. The agent noted this was pending: "Neural data time bin: frame-based deconvolved traces (exact dt pending)."

ii. No code snippets available.

iii. From CONVERSION_NOTES.md Step 3: "Neural data time bin: frame-based deconvolved traces (exact dt pending)."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. **No code was written.** The agent identified relevant variables: `SoundPos` (corridor position of sound cue per trial), `SoundTime` (time of sound cue per trial), `SoundFr` (frame index of sound cue per trial), and `ft` (frame times). These would be the basis for computing time to sound cue. No implementation was done.

ii. No code snippets available.

iii. From trajectory data exploration: The agent found `SoundPos`, `SoundTime`, `SoundTimeDelay`, `SoundFr`, `SoundDelayFr`, `SoundDelPos` in the behavior dict.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. **No code was written.** No processing decisions were made or documented for this variable.

ii. No code snippets available.

iii. No justification available.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. **No code was written.** No alignment decisions were made.

ii. No code snippets available.

iii. No justification available.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. **No code was written.** The agent identified that the `Imaging_Exp_info.npy` file contains session metadata including `sess#` (session number) which could indicate day/sequence of training. The behavior condition file names (e.g., `Beh_sup_train1_before_learning`) also encode experimental phase. No implementation was done.

ii. No code snippets available.

iii. From trajectory: The agent noted `sess#` values in experiment info (e.g., 0 for before_learning, 1 for after_learning).

## 4-b. What processing is involved in computing `input` *Day of training*?

i. **No code was written.** No processing decisions were made.

ii. No code snippets available.

iii. No justification available.

## 4-a (duplicate). What variables in the raw data is `input` *Environment type* derived from?

i. **No code was written.** The instructions do not list "Environment type" as a decoder input. The specified decoder inputs are: Time to sound cue, Day of training, Time since trial start, and Reward availability. This question appears to be erroneous or refers to an unlisted variable. The agent did not address this.

ii. No code snippets available.

iii. No justification available. Note: "Environment type" is not specified as a decoder input in the instructions. The instructions specify "Reward availability: 1 if in rewarded corridor, 0 if not" which is related but distinct.

## 4-b (duplicate). What processing is involved in computing `input` *Environment type*?

i. **No code was written.** See above — this variable is not listed in the decoder inputs specification.

ii. No code snippets available.

iii. No justification available.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. **No code was written.** The agent identified `Trial_start_time` (per trial), `ft` (frame times), and `StartFr` (start frame index per trial) as relevant variables for computing time since trial start. No implementation was done.

ii. No code snippets available.

iii. From trajectory data exploration: The agent documented these timing variables in CONVERSION_NOTES.md.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. **No code was written.** No processing decisions were made.

ii. No code snippets available.

iii. No justification available.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. **No code was written.** No alignment decisions were made.

ii. No code snippets available.

iii. No justification available.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. **No code was written.** The agent identified `isRew` (boolean per trial — whether the trial is in a rewarded corridor) in the behavior dict. This is the natural source variable for reward availability. No implementation was done.

ii. No code snippets available.

iii. From CONVERSION_NOTES.md: The agent noted `isRew` among the key behavior variables.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. **No code was written.** The `isRew` field is already a boolean per trial, so minimal processing would be needed — just conversion to 0/1 format. No implementation was done.

ii. No code snippets available.

iii. No justification available.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. **No code was written.** The agent identified `TrialStim` (string per trial, e.g., 'circle1', 'leaf1', 'leaf2', 'circle2'), `StimTrial` (dict mapping stimulus names to boolean arrays per trial), and `WallName` (string per trial) as relevant variables. No implementation was done.

ii. No code snippets available.

iii. From trajectory data exploration: The agent confirmed `TrialStim` contains values like 'circle1', 'leaf1', matching the instruction's example "e.g. circle1, leaf2, etc."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. **No code was written.** No processing decisions were made. The string labels would need to be converted to integer categorical codes.

ii. No code snippets available.

iii. No justification available.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. **No code was written.** The agent identified `LickFr` (frame indices of licks), `LickTrind` (trial indices of licks), `LickTime` (lick times), and `LickPos` (corridor position of licks) as relevant variables.

ii. No code snippets available.

iii. From CONVERSION_NOTES.md and trajectory: The agent documented these licking variables during data exploration.

## 8-b. What processing is involved in computing `output` *Licking*?

i. **No code was written.** The instructions specify "binary, time-varying. 0 = not licking, 1 = licking." The `LickFr` contains frame indices where licks occurred, which would need to be converted to a binary time series. No implementation was done.

ii. No code snippets available.

iii. No justification available.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. **No code was written.** No alignment decisions were made.

ii. No code snippets available.

iii. No justification available.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. **No code was written.** The agent identified `ft_Pos` (frame-level position within corridor, range 0-60 VR units corresponding to 0-4 m), `ft_PosCum` (cumulative position), and `run_pos` (trial x position bins matrix, shape ntrials x 60).

ii. No code snippets available.

iii. From trajectory data exploration: The agent documented `ft_Pos` and related position variables.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. **No code was written.** The instructions specify "discretized into 4 equal-length, 1-m-long spatial bins." The 4 m corridor would be split into bins [0-1m], [1-2m], [2-3m], [3-4m]. No implementation was done.

ii. No code snippets available.

iii. No justification available.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. **No code was written.** The instructions specify 4 equal-length 1-m bins. Given the corridor is 4 m (60 VR units), the bins would be at 0-15, 15-30, 30-45, 45-60 VR units. No implementation was done.

ii. No code snippets available.

iii. No justification available.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. **No code was written.** No alignment decisions were made.

ii. No code snippets available.

iii. No justification available.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. **No code was written.** The agent identified `ft_RunSpeed` (frame-level running speed) and `ft_isMoving` (boolean frame-level movement flag) as the relevant variables.

ii. No code snippets available.

iii. From trajectory data exploration: The agent documented these variables in its data structure analysis.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. **No code was written.** The instructions specify "discretized into 4 bins, each corresponding to 25% of the data" (quartile-based binning). No implementation was done.

ii. No code snippets available.

iii. No justification available.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. **No code was written.** Quartile binning would require computing the 25th, 50th, and 75th percentiles of running speed across the data and assigning each timepoint to the corresponding bin. No implementation was done.

ii. No code snippets available.

iii. No justification available.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. **No code was written.** No alignment decisions were made.

ii. No code snippets available.

iii. No justification available.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. **No code was written.** The agent identified a 1-frame offset between neural and behavior data (neural shape 23,193 vs behavior ft length 23,194 for one session), noting this requires "careful trimming to common length." No handling code was implemented.

ii. No code snippets available.

iii. From trajectory step 313: "neural arrays have shape (22851, 23193) for each of 3 areas, while behavior ft length is (23194,), a near-perfect match with only a 1-frame difference."

## 12-a. What are the most time-consuming steps of the code?

i. **No code was written.** No performance analysis was possible.

ii. No code snippets available.

iii. No justification available.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. **No code was written.** No efficiency analysis was possible.

ii. No code snippets available.

iii. No justification available.

## 12-c. What processing does the code repeat multiple times?

i. **No code was written.** No analysis was possible.

ii. No code snippets available.

iii. No justification available.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. **No code was written.** No analysis was possible.

ii. No code snippets available.

iii. No justification available.
