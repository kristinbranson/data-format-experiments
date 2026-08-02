# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent never implemented a loader. Its documented plan was to treat `/app/data/spk/*_neural_data.npy` as the canonical session list, then map behavior from `/app/data/beh/*.npy` by session ID/base session ID.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. From `CONVERSION_NOTES.md` and the trajectory, the agent concluded there were 89 neural sessions in `data/spk/`, 19 subjects from session-name prefixes, and behavior dictionaries in `data/beh/` keyed by session IDs plus extra condition/swap keys.

## 1-b. How are the data split into subjects?

i. The agent inferred subjects from the session-ID prefix before the first underscore, e.g. `TX108` from `TX108_2023_01_05_2`.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The trajectory explicitly reports 19 subjects and lists counts per prefix in `CONVERSION_NOTES.md`.

## 1-c. How are the data split into sessions?

i. The agent treated each neural file in `data/spk/` as one session and planned to use the neural session list as canonical, with behavior matched by the same session ID and `_swap` suffixes stripped when necessary.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. In Step 4 notes, the agent wrote that all 89 `spk` sessions were present in the behavior base-session keys and that extra behavior keys came from swap variants and aggregate labels.

## 1-d. How are the data split into trials?

i. The agent did not implement trial splitting. It only identified candidate behavior fields such as `ntrials`, `Trial_start_time`, `Trial_end_time`, and `ft_trInd`, which would allow frame-to-trial assignment.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The trajectory shows the agent inspected `ft_trInd` and trial timing fields but never converted them into per-trial neural/input/output arrays.

## 1-e. How are trials filtered based on quality controls?

i. No trial-quality filtering decision was implemented.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. `CONVERSION_NOTES.md` leaves trial curation unresolved, and the trajectory contains no completed trial QC rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The agent identified neural data as coming from the `spks` entry inside each `data/spk/*_neural_data.npy` file.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The trajectory records repeated inspection of `np.load(...).item()['spks']`.

## 2-b. How is the `neural` data processed?

i. The agent’s final interpretation was that each session contains three area/plane-specific arrays shaped `(n_neurons, n_timepoints)`, and that these are already deconvolved fluorescence traces rather than raw calcium or literal spike counts. No further processing was implemented.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The justification came from reconciling the paper’s “20,547 to 89,577 neurons per recording” with observed `spks` shapes like `(22851, 23193)` repeated three times per session.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neural QC/filtering rule was implemented.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The notes mention Suite2p/cell classification in the paper, but the agent never derived an inclusion rule for the released data.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The task required alignment to trial start/corridor entry. The agent did not implement this. Its only concrete alignment decision was that neural frames and behavior frames appear matched up to a possible 1-frame offset.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. Step 4 notes cite session `TX108_2023_01_05_2`, where neural arrays had 23193 frames and behavior `ft` had length 23194.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No converted time bin size was chosen, and no rebinning was implemented. The agent only noted that the neural data were frame-based.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. `CONVERSION_NOTES.md` says “frame-based deconvolved traces (exact dt pending).”

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. No final decision was implemented. The agent identified `SoundTime` and `SoundTimeDelay` as candidate trial-level variables.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. Those variables were listed during behavior-dictionary inspection in the trajectory and notes.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. No computation was implemented.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The agent never moved beyond identifying likely source fields.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. No sound-cue alignment procedure was implemented.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The only alignment result in the record is whole-session frame matching between neural arrays and behavior `ft`.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. No day-of-training source variable was identified.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The notes discuss training stages in the paper but do not map any raw field to per-session day.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. No processing rule was implemented.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The trajectory never reaches a mapping plan for this variable.

## 4-c. What variables in the raw data is `input` *Environment type* derived from?

i. No final raw variable was selected. The agent only noted condition-file names and stimulus/corridor descriptors from the paper.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The notes refer to supervised/unsupervised/grating conditions at the file level, but no implemented mapping exists.

## 4-d. What processing is involved in computing `input` *Environment type*?

i. No processing rule was implemented.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. Step 5 mapping never happened.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. No final implementation exists. The likely raw candidates the agent identified were `Trial_start_time`, `ft`, and `ft_trInd`.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. These fields were explicitly inspected, but never converted into a time-varying input.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. No computation was implemented.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The agent never produced per-trial frame-relative timestamps.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. No trial-start-aligned input was implemented.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The agent only observed that neural and behavior frames are near-equal in count within a session.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. No final choice was implemented. The agent identified candidate fields including `isRew`, `RewTime`, `Reward_Mode`, and paper-level reward-zone descriptions.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. These variables appear in the inspected behavior dictionaries and in the paper summary.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No computation was implemented.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The notes only restate that rewarded corridors deliver reward after the cue in the paper.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The agent identified `WallName`, `TrialStim`, `StimTrial`, and `StimFrame` as candidate raw variables for stimulus category.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The trajectory shows the agent inspecting these exact keys and noting stimulus labels such as `circle1`, `circle2`, `leaf1`, and `leaf2`.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. No final processing rule was implemented, and the agent never decided whether the output should be per-trial (`TrialStim`/`WallName`) or framewise (`StimFrame`).

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The notes stop at candidate-variable discovery.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. No final raw-variable decision was documented. The available candidates include `LickTime`, `LickTrind`, `LickPos`, and `LickFr`.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The agent trajectory did not reach a licking-mapping decision, though these keys exist in the behavior payload.

## 8-b. What processing is involved in computing `output` *Licking*?

i. No processing was implemented.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. There is no documented binarization/alignment of licks to frames or trials.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. No lick alignment was implemented.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The only relevant alignment observation is session-level neural-versus-`ft` frame matching.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. The agent identified `ft_Pos`, `ft_PosCum`, `VRpos`, and `run_pos` as relevant position-related raw variables.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. Those keys were inspected directly from behavior dictionaries.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. No processing rule was implemented.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The agent never selected between framewise position (`ft_Pos`) and pre-binned trial summaries (`run_pos`).

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. No thresholding into the required four 1-m bins was implemented.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The trajectory never reaches this step.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. No position alignment procedure was implemented beyond the session-level observation that neural frames and behavior `ft` are nearly matched.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. There is no per-trial or per-frame output construction.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. The agent identified `ft_RunSpeed` as the obvious framewise running-speed variable, with `ft_isMoving` as related metadata.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. These keys were explicitly listed during behavior inspection.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. No processing rule was implemented.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The agent never computed decoder-ready speed categories.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. No quartile thresholding was implemented.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The task requirement appears in the instructions, but the trajectory never reaches it.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. No speed-alignment implementation exists, apart from the session-level observation that neural and behavior frame streams are nearly coextensive.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The agent did not generate framewise per-trial output arrays.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. No general missing-data policy was implemented. The only concrete handling idea in the notes was that neural and behavior streams might need trimming to a common length because of a possible 1-frame mismatch, and that `ft_trInd` contains `NaN` outside trials.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. This came from the Step 4 discrepancy note and direct inspection of `ft_trInd`.

## 12-a. What are the most time-consuming steps of the code?

i. There is no conversion code. In the trajectory, the most time-consuming operations were repeated full-session loads of large neural arrays, which repeatedly stalled the shell.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The agent repeatedly noted that loading many large `spks` arrays across all 89 sessions was expensive.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. No conversion script exists, so no code loops can be assessed. The trajectory suggests that repeated per-session file loading and ad hoc iteration would have been the main efficiency risk.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. This is inferred from the exploratory commands, not from implemented code.

## 12-c. What processing does the code repeat multiple times?

i. No conversion code exists. The trajectory repeatedly reloads behavior files and large neural files while re-deriving the same session statistics.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The shell history shows repeated inspection of `spks`, session counts, and shape summaries.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No conversion code exists, so there is no downstream-discarded processing to audit.

ii. No `convert_data.py` was present in `/app`, so no code snippet is available.

iii. The agent never reached a working converter.
