# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent never produced `convert_data.py`. The only concrete loading plan it documented was to read per-session neural `.npy` files from `data/spk/` and behavior `.npy` dictionaries from `data/beh/`, then match behavior entries to the neural session list.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent in the current workspace, and the trajectory does not contain any write to that path.

iii. `CONVERSION_NOTES.md` Step 2 says `data/spk/` contains session-wise neural files and `data/beh/` contains behavior dictionaries keyed by session IDs. The stuck Step 4/5 trajectory buffer adds: use the neural session list as canonical sessions and map behavior by base session ID, handling `_swap` variants separately.

## 1-b. How are the data split into subjects?

i. The agent split subjects by taking the mouse name prefix from each session ID or filename, e.g. `TX108` from `TX108_2023_03_25_1`.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. `CONVERSION_NOTES.md` Step 2 lists 19 subjects derived from the session files, and the trajectory shows repeated subject counting from session-name prefixes.

## 1-c. How are the data split into sessions?

i. The agent treated each neural file in `data/spk/` as one canonical session and planned to attach behavior by matching the same base session ID across behavior dictionaries.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. Step 2 in `CONVERSION_NOTES.md` describes session-wise neural files named `<session_id>_neural_data.npy`. The Step 4/5 buffer says all 89 neural sessions are present in behavior base-session keys and that the neural session list should be canonical.

## 1-d. How are the data split into trials?

i. The agent did not finalize a trial-splitting procedure. It only identified trial-level arrays such as `ntrials`, `Trial_start_time`, `Trial_end_time`, `TrialStim`, `WallName`, and frame-level trial indices such as `ft_trInd`.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. Step 2 in `CONVERSION_NOTES.md` explicitly lists both trial-level and frame-level variables, but Step 5 mapping was never completed.

## 1-e. How are trials filtered based on quality controls?

i. No trial quality-control rule was documented or implemented.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. `CONVERSION_NOTES.md` leaves trial curation unresolved, and the trajectory never reaches a completed mapping or script.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The agent identified the neural source variable as the `spks` entry inside each per-session object loaded from `data/spk/*_neural_data.npy`.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. Step 2 in `CONVERSION_NOTES.md` says each neural file is a dict with key `spks`.

## 2-b. How is the `neural` data processed?

i. The only documented neural-processing decision was a tentative interpretation: `spks` is a list of three neuron-by-time arrays, likely corresponding to three simultaneously recorded areas or planes, and neuron counts should be taken from the first dimension.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. Step 4 in `CONVERSION_NOTES.md` resolves the earlier confusion about `spks` by stating that the summed first-axis counts match the paper’s per-recording neuron totals. No downstream per-trial extraction, stacking, trimming, or normalization code was written.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neural quality-control filtering rule was documented. The notes explicitly say no obvious ROI or neuron filtering rule had yet been identified from the explored reference code.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. Step 1 and Step 3 in `CONVERSION_NOTES.md` both leave neuron curation unresolved.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent never reached a trial-alignment implementation. Its only concrete alignment conclusion was that the full-session neural arrays appear to match the behavior frame stream to within about one frame and would need trimming to a common length.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. The Step 4/5 trajectory buffer says a matched session had neural length 23,193 versus behavior `ft` length 23,194 and therefore likely needed careful trimming. It never specifies alignment to corridor entry or trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent did not document a converted time-bin size or any temporal rebinning rule.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. Step 3 says the data are frame-based deconvolved traces and leaves the exact bin size pending; no final choice appears anywhere else.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The agent never made a final mapping, but it identified candidate raw variables including `SoundTime`, `SoundTimeDelay`, `SoundFr`, and the frame time vector `ft`.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. These variables are listed in the Step 2 notes and trajectory inspection output, but Step 5 mapping was never completed.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. No computation rule for time-to-cue was documented.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. The agent stopped before writing the source-to-target mapping required for this variable.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. No explicit alignment rule was documented for this input beyond the tentative full-session frame-stream matching noted in Step 4.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. The trajectory never reaches a per-trial, frame-by-frame input construction stage.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The agent did not make a final decision. It only discovered candidate metadata in `Imaging_Exp_info.npy`, including condition-group membership, `sess#`, mouse name, date, block, and reward/training descriptors.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. `Imaging_Exp_info.npy` was inspected in the notes/trajectory, but no mapping from those fields to a per-trial day-of-training value was written.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. No processing rule was documented for converting metadata to a training-day variable.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. The agent never completed Step 5, where this mapping should have been specified.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The agent never finalized this mapping. The candidate variables it surfaced were `ft`, `ft_trInd`, `Trial_start_time`, and `StartFr`.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. These are the relevant trial/frame variables listed in Step 2 and the trajectory inspection output.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. No computation procedure was documented.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. The agent never moved from variable discovery to implemented time-series construction.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. No explicit per-trial alignment rule was documented. The only recorded conclusion was that neural frames and behavior `ft` were nearly matched at the session level.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. The Step 4/5 buffer mentions only common-length trimming between full-session neural arrays and behavior frames.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. The agent did not make a final mapping. It only identified possible source variables such as `isRew`, `RewTime`, `RewPos`, `WallName`, and reward/training metadata.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. Those variables are listed in Step 2 and the exploratory trajectory output, but no decoder-input definition was completed.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No processing rule was documented.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. The notes never specify whether reward availability should track rewarded corridor identity, cue timing, or actual reward delivery.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The agent identified `TrialStim` as the raw per-trial stimulus-category variable.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. `TrialStim` is explicitly listed in Step 2 notes and the trajectory’s variable inspection output.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. No processing rule beyond identifying `TrialStim` was documented. The agent never specified whether it would use the raw string labels directly, re-index them, or harmonize them across sessions.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. Step 5 mapping never happened, so the categorical encoding decision is missing.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. The agent identified licking-related variables including `LickTrind`, `LickTime`, `LickPos`, `Lick_wallName`, and `LickFr`.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. These variables were enumerated during behavior-structure inspection and summarized in the trajectory.

## 8-b. What processing is involved in computing `output` *Licking*?

i. No processing rule was documented for converting the raw lick event variables into a binary time-varying output.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. The agent stopped before defining event-to-frame conversion or per-trial extraction.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. No explicit lick-to-neural alignment rule was documented.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. The only alignment evidence in the notes is the coarse session-level frame-stream match between neural arrays and `ft`.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. The agent identified `ft_Pos` and `ft_PosCum` as the relevant frame-level corridor-position variables.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. Those fields are named in Step 2 notes and the trajectory inspection output.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. No processing rule was documented beyond discovering the relevant variables.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. The agent never specified whether to use position within the textured corridor, cumulative position, or trial-referenced position.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. No binning rule was documented.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. The instruction-required four equal 1 m bins were never written into a mapping plan or script.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. The only documented alignment idea was to use the behavior frame stream, which appears nearly matched to the neural time axis, and trim to a common length if needed.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. This comes from the Step 4/5 trajectory buffer; there is no per-trial alignment implementation.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. The agent identified `ft_RunSpeed` as the relevant frame-level running-speed variable.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. `ft_RunSpeed` appears in the Step 2 notes and trajectory variable inspection output.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. No processing rule was documented beyond identifying `ft_RunSpeed`.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. The agent never specified any trial extraction, masking, or discretization logic for this variable.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. No thresholding rule was documented.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. The instruction-required 25% quantile binning was never mapped or implemented.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The only documented alignment idea was to use the frame-level behavior stream and trim to the common neural/behavior length.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. This comes from the Step 4/5 trajectory buffer rather than any written conversion code.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. No missing-data policy was documented. The agent observed that some reward variables contain `NaN`, but it never specified how to propagate, impute, mask, or drop such values.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. The behavior inspection output in the trajectory shows `NaN` values in reward-related arrays, but no handling rule appears in the notes or code.

## 12-a. What are the most time-consuming steps of the code?

i. The agent never wrote or profiled conversion code, so it never identified actual bottlenecks. At most, its notes acknowledge that large per-session neural arrays would need efficient handling.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. Step 6 in `CONVERSION_NOTES.md` remains `NOT STARTED`, and there are no timing outputs for a conversion script.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. No conversion code exists, so no loops were identified or optimized.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. The agent never reached implementation or optimization.

## 12-c. What processing does the code repeat multiple times?

i. No conversion code exists, so no repeated processing was documented.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. The agent never reached the script-development stage.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No conversion code exists, so there is no implemented downstream-discarded processing to inspect.

ii. No `convert_data.py` snippet is available. `/app/convert_data.py` is absent.

iii. The agent never produced `convert_data.py`, sample outputs, or full outputs.
