# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI never wrote a conversion script (`convert_data.py`). It completed Steps 0-3 of the workflow (setup, code exploration, data exploration, reference text reading) and was partway through Step 4 (consistency check) when its terminal became stuck in a heredoc/continuation mode. It spent the remaining ~3000+ trajectory steps unable to recover. No data loading code was ever written.

From CONVERSION_NOTES.md, the AI identified the data structure: `data/spk/` for neural data, `data/beh/` for behavior files keyed by session ID, and `data/retinotopy/` for visual area assignments. It noted the master index `Imaging_Exp_info.npy` and behavior files named `Beh_<exp_type>.npy`.

ii. No code was written.

iii. The AI's trajectory shows it explored the data directory structure and identified file organization but never progressed to writing conversion code due to terminal issues.

## 1-b. How are the data split into subjects?

i. No conversion code was written. From CONVERSION_NOTES.md, the AI identified 19 subjects with varying session counts per subject.

ii. No code was written.

iii. N/A - no code produced.

## 1-c. How are the data split into sessions?

i. No conversion code was written. The AI noted 89 recordings across 19 mice from the paper and identified session IDs in behavior files.

ii. No code was written.

iii. N/A - no code produced.

## 1-d. How are the data split into trials?

i. No conversion code was written. The AI noted trial-level variables in behavior dicts including `ntrials`, `Trial_start_time`, `Trial_end_time`, etc.

ii. No code was written.

iii. N/A - no code produced.

## 1-e. How are trials filtered based on quality controls?

i. No conversion code was written. The AI noted trial curation rules were "still to be extracted" in CONVERSION_NOTES.md.

ii. No code was written.

iii. N/A - no code produced.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. No conversion code was written. The AI identified `spks` in the neural data files as deconvolved fluorescence traces, interpreting `spks` as a list of plane-specific neuron-by-time arrays.

ii. No code was written.

iii. From CONVERSION_NOTES.md Step 4, the AI resolved that `spks` is a list of 3 arrays (per imaging plane), with the sum of first dimensions matching the paper's neuron count range (20,547-89,577).

## 2-b. How is the `neural` data processed?

i. No conversion code was written. No processing decisions were documented.

ii. No code was written.

iii. N/A - no code produced.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No conversion code was written. The AI noted Suite2p processing was used for cell classification but did not document specific filtering criteria.

ii. No code was written.

iii. N/A - no code produced.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No conversion code was written. No alignment decisions were documented.

ii. No code was written.

iii. N/A - no code produced.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No conversion code was written. The AI noted "frame-based deconvolved traces (exact dt pending)" in CONVERSION_NOTES.md.

ii. No code was written.

iii. N/A - no code produced.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. No conversion code was written. The AI identified `SoundTime` and `SoundTimeDelay` in the behavior dicts but did not map them to the input variable.

ii. No code was written.

iii. N/A - no code produced.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. No conversion code was written.

ii. No code was written.

iii. N/A - no code produced.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. No conversion code was written.

ii. No code was written.

iii. N/A - no code produced.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. No conversion code was written.

ii. No code was written.

iii. N/A - no code produced.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. No conversion code was written.

ii. No code was written.

iii. N/A - no code produced.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. No conversion code was written.

ii. No code was written.

iii. N/A - no code produced.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. No conversion code was written.

ii. No code was written.

iii. N/A - no code produced.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. No conversion code was written.

ii. No code was written.

iii. N/A - no code produced.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. No conversion code was written. The AI identified `isRew` in the behavior dicts.

ii. No code was written.

iii. N/A - no code produced.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No conversion code was written.

ii. No code was written.

iii. N/A - no code produced.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. No conversion code was written. The AI identified `WallName` and `TrialStim` in the behavior dicts.

ii. No code was written.

iii. N/A - no code produced.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. No conversion code was written.

ii. No code was written.

iii. N/A - no code produced.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. No conversion code was written. The AI identified licking-related variables in behavior dicts.

ii. No code was written.

iii. N/A - no code produced.

## 8-b. What processing is involved in computing `output` *Licking*?

i. No conversion code was written.

ii. No code was written.

iii. N/A - no code produced.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. No conversion code was written.

ii. No code was written.

iii. N/A - no code produced.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. No conversion code was written. The AI identified `ft_Pos` and `ft_PosCum` in the behavior dicts.

ii. No code was written.

iii. N/A - no code produced.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. No conversion code was written.

ii. No code was written.

iii. N/A - no code produced.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. No conversion code was written.

ii. No code was written.

iii. N/A - no code produced.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. No conversion code was written.

ii. No code was written.

iii. N/A - no code produced.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. No conversion code was written. The AI identified `ft_RunSpeed` in the behavior dicts.

ii. No code was written.

iii. N/A - no code produced.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. No conversion code was written.

ii. No code was written.

iii. N/A - no code produced.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. No conversion code was written.

ii. No code was written.

iii. N/A - no code produced.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. No conversion code was written.

ii. No code was written.

iii. N/A - no code produced.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. No conversion code was written. No error handling decisions were documented.

ii. No code was written.

iii. N/A - no code produced.

## 12-a. What are the most time-consuming steps of the code?

i. No conversion code was written.

ii. No code was written.

iii. N/A - no code produced.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. No conversion code was written.

ii. No code was written.

iii. N/A - no code produced.

## 12-c. What processing does the code repeat multiple times?

i. No conversion code was written.

ii. No code was written.

iii. N/A - no code produced.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No conversion code was written.

ii. No code was written.

iii. N/A - no code produced.
