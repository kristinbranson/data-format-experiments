# Decisions

**Critical Note**: The AI agent (gpt-5.4, terminus-2) never produced a `convert_data.py` script or any converted data. The agent got stuck in a broken shell state (continuation prompt `>`) around step ~10 of 3196 total steps, and spent the remaining ~3186 steps repeatedly waiting for a terminal reset that never came. Only `CONVERSION_NOTES.md` was partially filled (Steps 0-4 partially complete; Steps 5-13 remain as unfilled template). Therefore, for all decision questions below, there is **no code to evaluate** and **no implementation decisions were made**.

---

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. No conversion code was written. The agent explored the data directory structure in CONVERSION_NOTES.md Steps 1-2 and identified that neural data lives in `data/spk/<session_id>_neural_data.npy` files and behavioral data in `data/beh/Beh_<condition>.npy` files (dictionaries keyed by session ID). However, no loading code was implemented.

ii. No code exists (`convert_data.py` was never created).

iii. The agent noted in CONVERSION_NOTES.md Step 1 that the reference code uses `utils.load_dat` to load preprocessed data from pickle. In Step 2, it identified the data file organization but never progressed to implementing a loader.

---

## 1-b. How are the data split into subjects?

i. No implementation. The agent identified 19 subjects from the data files and listed per-subject session counts in CONVERSION_NOTES.md Step 2 (e.g., DR10:6, DR15:5, LZ13:4, etc.). The subject name is extracted from the session ID prefix (e.g., `TX83` from `TX83_2022_08_17_1`). However, no code was written to implement this split.

ii. No code exists.

iii. The agent's CONVERSION_NOTES.md Step 2 documents the subject identification approach.

---

## 1-c. How are the data split into sessions?

i. No implementation. The agent identified that sessions correspond to individual neural data files (`data/spk/<session_id>_neural_data.npy`) and behavior dictionary keys. No code was written.

ii. No code exists.

iii. CONVERSION_NOTES.md Step 2 documents session identification from file names and behavior dictionary keys.

---

## 1-d. How are the data split into trials?

i. No implementation. The agent identified trial-level variables in the behavior dictionaries (e.g., `ntrials`, `Trial_start_time`, `Trial_end_time`) but never wrote code to split data into trials.

ii. No code exists.

iii. CONVERSION_NOTES.md Step 2 notes that behavior files contain trial-level timing variables.

---

## 1-e. How are trials filtered based on quality controls?

i. No implementation. The agent did not determine or implement any trial filtering criteria. CONVERSION_NOTES.md Step 3 notes "Trial curation rules: Trials are defined by corridor traversals with trial-level timing variables available in behavior dicts; exact trial filtering still to be extracted."

ii. No code exists.

iii. The agent explicitly noted that trial filtering criteria were still to be determined.

---

## 2-a. What variables in the raw data is the `neural` data derived from?

i. No implementation. The agent identified that neural data is stored in `data/spk/<session_id>_neural_data.npy` files. In Step 4 (Consistency Check), it noted that each file contains a dict with key `spks`, whose value is a list of 3 arrays (interpreted as area/plane-specific neuron-by-time arrays). The paper states these are deconvolved fluorescence traces from Suite2p.

ii. No code exists.

iii. From CONVERSION_NOTES.md Step 4: "Interpret `spks` as three area/plane-specific neuron-by-time arrays; sum of first dimensions per session matches paper neuron-count range."

---

## 2-b. How is the `neural` data processed?

i. No implementation. No processing decisions were made. The agent noted that the paper uses deconvolved fluorescence traces but did not determine what processing steps to apply.

ii. No code exists.

iii. CONVERSION_NOTES.md Step 3 notes "All our analyses were based on deconvolved fluorescence traces" but no processing pipeline was designed.

---

## 2-c. How is the `neural` data filtered based on quality controls?

i. No implementation. CONVERSION_NOTES.md Step 3 states: "Neuron curation rules: Paper methods mention Suite2p processing, ROI detection, cell classification, neuropil correction, and spike deconvolution; exact inclusion/exclusion criteria still to be extracted."

ii. No code exists.

iii. The agent explicitly noted that neuron filtering criteria were still to be determined.

---

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No implementation. The instructions specify alignment to trial start (corridor entry). The agent noted in Step 1 that reference code emphasizes trial-start alignment (via `sort_trialstart` function), but no alignment code was written.

ii. No code exists.

iii. CONVERSION_NOTES.md Step 1 notes the `sort_trialstart` function for alignment.

---

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No implementation. CONVERSION_NOTES.md Step 3 notes "Neural data time bin: frame-based deconvolved traces (exact dt pending)." No temporal resolution was determined or rebinning implemented.

ii. No code exists.

iii. The agent noted that the exact time bin was still pending determination.

---

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. No implementation. The agent identified `SoundTime` and `SoundTimeDelay` in the behavior dictionary keys (CONVERSION_NOTES.md Step 2) but never designed or implemented the input variable.

ii. No code exists.

iii. No justification provided for the specific variable choice.

---

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. No implementation. No processing was designed or coded.

ii. No code exists.

iii. No justification provided.

---

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. No implementation.

ii. No code exists.

iii. No justification provided.

---

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. No implementation. The agent noted behavior files contain session metadata (date, experimental condition) but did not map these to a "day of training" variable.

ii. No code exists.

iii. No justification provided.

---

## 4-b. What processing is involved in computing `input` *Day of training*?

i. No implementation.

ii. No code exists.

iii. No justification provided.

---

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. No implementation. The agent identified `Trial_start_time` in behavior dictionaries but did not implement the input.

ii. No code exists.

iii. No justification provided.

---

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. No implementation.

ii. No code exists.

iii. No justification provided.

---

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. No implementation.

ii. No code exists.

iii. No justification provided.

---

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. No implementation. The agent identified `isRew` and `RewTime` in behavior dictionary keys, and noted in Step 3 that "reward delivered if a lick was detected after the sound cue in the rewarded corridor." No code was written.

ii. No code exists.

iii. No justification provided.

---

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No implementation.

ii. No code exists.

iii. No justification provided.

---

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. No implementation. The agent identified `WallName`, `TrialStim`, and `stim_id` fields in the data, plus `StimTrial` and `StimFrame` masks. No mapping was implemented.

ii. No code exists.

iii. No justification provided.

---

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. No implementation.

ii. No code exists.

iii. No justification provided.

---

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. No implementation. The agent identified licking-related variables in the reference code (e.g., `first_lick_position` function) but did not identify the specific raw variable or implement any code.

ii. No code exists.

iii. No justification provided.

---

## 8-b. What processing is involved in computing `output` *Licking*?

i. No implementation.

ii. No code exists.

iii. No justification provided.

---

## 8-c. How is `output` *Licking* aligned with the neural data?

i. No implementation.

ii. No code exists.

iii. No justification provided.

---

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. No implementation. The agent identified `ft_Pos` and `ft_PosCum` in the behavior dictionary keys (Step 2) but did not implement the output.

ii. No code exists.

iii. No justification provided.

---

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. No implementation.

ii. No code exists.

iii. No justification provided.

---

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. No implementation. The instructions specify discretization into 4 equal-length, 1-m-long spatial bins, but no code was written.

ii. No code exists.

iii. No justification provided.

---

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. No implementation.

ii. No code exists.

iii. No justification provided.

---

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. No implementation. The agent identified `ft_RunSpeed` and `ft_isMoving` in the behavior dictionary keys (Step 2) but did not implement the output.

ii. No code exists.

iii. No justification provided.

---

## 10-b. What processing is involved in computing `output` *Running speed*?

i. No implementation.

ii. No code exists.

iii. No justification provided.

---

## 10-c. How is `output` *Running speed* thresholded into categories?

i. No implementation. The instructions specify discretization into 4 bins, each corresponding to 25% of the data. No code was written.

ii. No code exists.

iii. No justification provided.

---

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. No implementation.

ii. No code exists.

iii. No justification provided.

---

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. No implementation. No missing data handling was designed or coded.

ii. No code exists.

iii. No justification provided.

---

## 12-a. What are the most time-consuming steps of the code?

i. No code exists, so no performance analysis is possible.

ii. No code exists.

iii. No justification provided.

---

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. No code exists.

ii. No code exists.

iii. No justification provided.

---

## 12-c. What processing does the code repeat multiple times?

i. No code exists.

ii. No code exists.

iii. No justification provided.

---

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No code exists.

ii. No code exists.

iii. No justification provided.
