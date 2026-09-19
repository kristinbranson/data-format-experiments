# Decisions

**Note**: The AI agent (gpt-5.4 / terminus-2) got its terminal shell stuck in a continuation prompt (`>`) around step ~50 of 3196 total steps. It only completed Steps 0-3 of the 13-step conversion workflow (setup, code exploration, data exploration, reference text reading). It **never wrote `convert_data.py`**, never produced `converted_data.pkl`, and never trained a decoder. The CONVERSION_NOTES.md is mostly empty template from Step 5 onward. All answers below reflect this failure.

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI identified the data directory structure (`data/beh/`, `data/spk/`, `data/retinotopy/`) and noted that behavior files are `.npy` object dictionaries keyed by session IDs, and neural files are per-session `.npy` files. However, no loading code was written.

ii. No code exists — `convert_data.py` was never created.

iii. From CONVERSION_NOTES.md Step 2: "data/spk/: session-wise neural arrays in .npy files... data/beh/: behavior dictionaries in .npy object files grouped by experimental condition." The trajectory shows the AI explored file structures using `find` and `np.load` but never formalized a loading strategy into code.

## 1-b. How are the data split into subjects?

i. The AI identified 19 subjects from the data files and listed per-subject session counts (e.g., DR10:6, DR15:5, etc.) in CONVERSION_NOTES.md. No splitting code was written.

ii. No code exists.

iii. From CONVERSION_NOTES.md Step 2: "Subjects: 19" with session counts per subject listed.

## 1-c. How are the data split into sessions?

i. The AI noted sessions are identified by session IDs in filenames and behavior dictionary keys. It noted 89 recordings across 19 mice. No session-splitting code was written.

ii. No code exists.

iii. From CONVERSION_NOTES.md Step 3: "We performed 89 recordings in 19 mice..." (quoted from paper).

## 1-d. How are the data split into trials?

i. The AI identified trial-level variables in the behavior dictionaries (`ntrials`, `Trial_start_time`, `Trial_end_time`, `ft_trInd`) but did not implement trial splitting.

ii. No code exists.

iii. From CONVERSION_NOTES.md Step 2: behavior files include "trial-level variables (ntrials, Trial_start_time, Trial_end_time, SoundTime, SoundTimeDelay, RewTime, isRew, WallName, TrialStim), frame-level variables (ft, ft_trInd, ft_Pos, ft_PosCum, ft_RunSpeed, ft_isMoving, etc.)."

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering decisions were made. The AI did not reach the mapping/planning stage.

ii. No code exists.

iii. No justification provided.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI identified that neural data comes from `data/spk/<session_id>_neural_data.npy` files containing `spks` (a list of arrays). In the consistency check (Step 4), the AI resolved that `spks` is a list of 3 plane-specific neuron-by-time arrays.

ii. No code exists.

iii. From CONVERSION_NOTES.md Step 4: "Interpret spks as three area/plane-specific neuron-by-time arrays; sum of first dimensions per session matches paper neuron-count range."

## 2-b. How is the `neural` data processed?

i. No processing decisions were made. The AI noted the paper states "All our analyses were based on deconvolved fluorescence traces" but did not plan or implement processing.

ii. No code exists.

iii. From CONVERSION_NOTES.md Step 3: "Paper methods state analyses used deconvolved fluorescence traces processed with Suite2p, not raw calcium traces."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neural filtering decisions were made. The AI noted Suite2p processing was used for ROI detection and cell classification but did not specify filtering criteria.

ii. No code exists.

iii. From CONVERSION_NOTES.md Step 3: "Paper methods mention Suite2p processing, ROI detection, cell classification, neuropil correction, and spike deconvolution; exact inclusion/exclusion criteria still to be extracted."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No alignment decisions were made.

ii. No code exists.

iii. No justification provided. The AI noted "trial-start alignment" in reference code analysis but did not plan implementation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal resolution decisions were made. The AI noted the data is "frame-based deconvolved traces" but did not determine the frame rate or bin size.

ii. No code exists.

iii. From CONVERSION_NOTES.md Step 3: "Neural data time bin: frame-based deconvolved traces (exact dt pending)."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The AI identified `SoundTime` and `SoundTimeDelay` as relevant variables in the behavior dictionaries but did not determine the specific frame-level variable to use.

ii. No code exists.

iii. From CONVERSION_NOTES.md Step 2: behavior files include "SoundTime, SoundTimeDelay."

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. No processing decisions were made.

ii. No code exists.

iii. No justification provided.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. No alignment decisions were made.

ii. No code exists.

iii. No justification provided.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. No specific variable mapping was decided. The AI noted training days exist but did not plan how to derive them.

ii. No code exists.

iii. From CONVERSION_NOTES.md Step 3: "Behavior-only training: 5 total days, with passive reward on day 1 and active reward on days 2-5."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. No processing decisions were made.

ii. No code exists.

iii. No justification provided.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The AI identified `Trial_start_time` and `ft` as potentially relevant but did not make a specific decision.

ii. No code exists.

iii. No justification provided.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. No processing decisions were made.

ii. No code exists.

iii. No justification provided.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. No alignment decisions were made.

ii. No code exists.

iii. No justification provided.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. The AI identified `isRew` as the relevant variable. From CONVERSION_NOTES.md Step 2: behavior files include "isRew."

ii. No code exists.

iii. From CONVERSION_NOTES.md Step 3: "reward delivered if a lick was detected after the sound cue in the rewarded corridor."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No processing decisions were made.

ii. No code exists.

iii. No justification provided.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The AI identified `WallName` and `TrialStim` as relevant variables.

ii. No code exists.

iii. From CONVERSION_NOTES.md Step 2: behavior files include "WallName, TrialStim."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. No processing decisions were made.

ii. No code exists.

iii. No justification provided.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. The AI identified licking-related variables but did not specify which raw variable to use.

ii. No code exists.

iii. From CONVERSION_NOTES.md Step 1: the reference code includes "first_lick_position" functions that compute "first-lick probability as function of corridor position."

## 8-b. What processing is involved in computing `output` *Licking*?

i. No processing decisions were made.

ii. No code exists.

iii. No justification provided.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. No alignment decisions were made.

ii. No code exists.

iii. No justification provided.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. The AI identified `ft_Pos` and `ft_PosCum` as position variables in the behavior data.

ii. No code exists.

iii. From CONVERSION_NOTES.md Step 2: behavior files include "ft_Pos, ft_PosCum."

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. No processing decisions were made.

ii. No code exists.

iii. No justification provided.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. No thresholding decisions were made.

ii. No code exists.

iii. No justification provided.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. No alignment decisions were made.

ii. No code exists.

iii. No justification provided.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. The AI identified `ft_RunSpeed` and `ft_isMoving` as running speed variables.

ii. No code exists.

iii. From CONVERSION_NOTES.md Step 2: behavior files include "ft_RunSpeed, ft_isMoving."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. No processing decisions were made.

ii. No code exists.

iii. No justification provided.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. No thresholding decisions were made.

ii. No code exists.

iii. No justification provided.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. No alignment decisions were made.

ii. No code exists.

iii. No justification provided.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. No data handling decisions were made.

ii. No code exists.

iii. No justification provided.

## 12-a. What are the most time-consuming steps of the code?

i. No code was written, so no performance analysis was possible.

ii. No code exists.

iii. No justification provided.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. No code was written.

ii. No code exists.

iii. No justification provided.

## 12-c. What processing does the code repeat multiple times?

i. No code was written.

ii. No code exists.

iii. No justification provided.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No code was written.

ii. No code exists.

iii. No justification provided.
