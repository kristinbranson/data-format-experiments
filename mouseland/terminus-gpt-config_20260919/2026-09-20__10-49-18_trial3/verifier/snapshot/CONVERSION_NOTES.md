# Dataset Conversion Notes

## Overview
- **Dataset**: Unsupervised pretraining in biological neural networks
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment verification:
- Python 3.13.15
- NumPy 2.4.4
- PyTorch 2.6.0+cu124

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `code/`
- `data/`
- `decoder.py`
- `docker-compose.yaml`
- `methods.txt`
- `paper.pdf`
- `train_decoder.py`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| notebook workflow | `data_process_script.ipynb` | PROCESSING | Loads experiment metadata/behavior, creates spatially interpolated neural arrays, then paper-specific stimulus and reward analyses. |
| `load_exp_beh` | `utils.py` | LOADING | Loads `beh/Beh_<experiment>.npy` as a Python dictionary. |
| `load_spk` | `utils.py` | LOADING | Loads per-session `<mouse>_<date>_<block>_spks.npy` neuron-by-frame activity. |
| `load_retino` | `utils.py` | LOADING | Loads per-session retinotopy and neuron area assignments. |
| `get_cat_id` | `utils.py` | PROCESSING | Maps wall/visual identity and reward condition to categorical stimulus IDs. |
| `spk_pos_interp` | `utils.py` | PROCESSING | Interpolates each neuron's frame activity onto target position samples. |
| `get_interpPos_spk` | `utils.py` | PROCESSING | Splits continuous activity by trial/cumulative position and produces neuron x trial x position arrays (normally 60 bins). |
| `get_lick_raster` | `utils.py` | PROCESSING | Builds trial-position lick rasters and reward/cue-aligned behavioral summaries. |
| `Get_dprime_selective_neuron` | `utils.py` | CURATION | Computes stimulus d-prime using valid frames; used for figure analyses, not universal cell-quality exclusion. |
| `Get_coding_direction` | `utils.py` | PROCESSING | Selects stimulus-discriminating tails (5th percentile default) and projects interpolated activity. |
| `Get_sort_spk` | `utils.py` | PROCESSING | Selects neurons on training trials and sorts peak position using held-out odd/even trials. |
| `spk_2_firstLick`, `spk_2_cue` | `utils.py` | PROCESSING | Extract fixed frame windows around first lick or sound cue. |

### Notes
- `README.md` identifies `data_process_script.ipynb` as the reference intermediate-data processing workflow; the other notebook/scripts primarily recreate figures.
- Native neural input used by the code is a neuron-by-imaging-frame `spks` array. The code does not compute delta-F/F; activity is already supplied as inferred/deconvolved spike activity and should be loaded directly.
- Session identity is `<mname>_<datexp>_<blk>`. Multiple sessions from one mouse are retained and disambiguated.
- Reference spatial processing uses cumulative VR position and interpolation to 60 equal spatial samples across the corridor. Analyses mask invalid frames (`fr_valid`) and commonly restrict to VR movement frames (`VRmove`).
- Stimulus IDs documented in code are: 0 circle1, 1 circle2, 2 leaf1, 3 leaf2, 4 leaf3, 5 leaf1_swap1, 6 leaf1_swap2. Wall identity together with reward status determines category where applicable.
- Behavioral streams include trial wall/category, reward status, cumulative position, sound position, lick times/positions, reward position, and validity/movement indicators.
- Retinotopy contains per-neuron area indices. Some visual-cortex map analyses exclude area IDs -1 (unassigned/outside) and 7; this is analysis-specific. No universal neuron quality filter (`iscell`, SNR threshold, etc.) was found.
- Stimulus-selective neuron thresholds and train/test trial splitting are specific to paper figure analyses. They must not be applied as general recording-quality curation for the requested all-neural-activity decoder unless later text/data evidence requires it.
- Reward-response code uses sound/cue position and interpolated position bins (typically positions 5:40 for comparisons), confirming that cue timing/position is a core experimental variable.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/spk/`: 89 `*_neural_data.npy` files (434.2 GB decimal; 405 GiB). Each is a pickled scalar dictionary with key `spks`, whose value is a list of three 2-D float arrays (ROI x imaging frame). Reference `load_spk` concatenates all three arrays on the ROI axis.
- `/app/data/beh/`: 23 `Beh_<experiment>.npy` behavior dictionaries plus `Imaging_Exp_info.npy` and one example file (6.6 GiB total). Top-level behavior keys are session IDs; values contain trial and frame streams.
- `/app/data/retinotopy/`: 90 `*_trans.npz` files (170 MiB) containing transformed coordinates (`xy_t`, `xpos`, `ypos`) and flat per-ROI cortical area code `iarea`. There are 90 files because one date-level retinotopy can exist beyond or be shared relative to the 89 block-level neural sessions.
- `/app/data/process_data/`: empty; intended for generated reference intermediate products.
- `Imaging_Exp_info.npy` has 23 experiment groups and 142 analysis entries, but only 89 unique physical session IDs. Several recordings appear in multiple task analyses (e.g. test1/test2/train2), and test3 may expose duplicate `_swap1`/`_swap2` dictionary views. These are not additional recordings or trials.

### Available Variables
- Trial level: `ntrials`, `trInd`, start/end time, wall identity/type, `WallName`, `TrialStim`, `isRew`, sound time/position, reward time/position, lick trial/time/position, reward mode/delay.
- Frame level: imaging frame time `ft`, trial index `ft_trInd`, corridor/gray-space masks, position/cumulative position, movement and running speed, wall ID, cue-relative masks, and stimulus-frame masks.
- Neural: three planes/parts of inferred spike activity; reference concatenates them into one neuron axis. No raw fluorescence or delta-F/F conversion is required.
- Anatomy: per-neuron transformed XY position and `iarea`, converted by reference `neu_area_ID` to All/V1/medial/anterior/lateral groupings.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 4,691,034 ROI traces across sessions (sum of three arrays/session); aligned exactly to raw retinotopy `iarea` lengths |
| Neurons / session | mean 52,708.2; range 20,547–89,577 after concatenating three arrays |
| Subjects | 19: DR10, DR15, LZ13, LZ16, TX104, TX105, TX108, TX109, TX119, TX123, TX124, TX139, TX140, TX60, TX61, TX83, TX85, TX88, VR2 |
| Sessions / subject | 89 total, variable by subject |
| Trials (total) | 38,110 unique physical-session trials before validity/decoder filtering |
| Trials / session | mean 428.2, median 429, range 84–789 |
| Neural frames | 2,025,155 total; mean 22,754.6/session; range 14,570–34,228 |
| Behavior frames | 2,025,281 total; reference truncates behavior streams to neural `nfr` |

### Observed Distributions and Edge Cases
- All corridors report length 60 source units (the experiment uses a 4 m corridor; source position is scaled 0–60).
- Rewarded trials: 4,336/38,110 = 11.38%. `RewardFr` is NaN for 33,774 unrewarded trials, exactly as expected; start, end, and sound frame arrays have no missing values.
- `TrialStim` counts include 2,361 literal `stimulus_of_trial` placeholders. `WallName` retains concrete visual identity on these trials and is the safer raw visual-category source, with metadata stimulus mappings used for renamed texture sets (rock/wood versus circle/leaf).
- Neural arrays have 1–2 fewer frames than behavior arrays in many sessions. Reference code explicitly slices behavior masks/streams to neural `nfr`; conversion must do the same.
- The third `spks` component has 0–2 more rows than each of the first two in some recordings. This is valid: raw retinotopy length equals the sum of all three components, and reference code concatenates all three.
- All trial-sized arrays checked (`StartFr`, `EndFr`, `SoundFr`, `WallName`, `isRew`) match `ntrials`.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Up to ~90,000 simultaneously per recording | “populations of up to 90,000 neurons simultaneously” |
| Neurons / session | 20,547–89,577 | “Suite2p ... activity traces from 20,547 to 89,577 neurons in each recording” |
| Subjects | 19 imaging mice | “We performed 89 recordings in 19 mice...” |
| Sessions / subject | 89 recordings total | Same Methods statement; exact match to unique raw files |
| Trials (total) | Not reported globally | Raw unique-session total is 38,110 |
| Trials / session | Not reported globally | Raw mean 428.2, range 84–789 |
| Neural data time bin | Native imaging frame; raw `ft` timestamps determine interval | Imaging acquired with ScanImage/two-photon mesoscope; no temporal rebinning specified in reference analysis |
| Behavior data time bin | Behavior interpolated/aligned to imaging-frame `ft` | Reference code truncates frame behavior to neural `nfr` |
| Reward rate | Not reported globally | Raw unique-session rate 11.38% |
| Corridor length | 4 m | Closed-loop virtual linear corridor described in Methods |
| Sound cue position | Uniformly random 0.5–3.5 m | Methods: cue time chosen per trial from positions 0.5–3.5 m |
| Imaging processing | Suite2p; 0.75 s decay | Motion correction, ROI detection, cell classification, neuropil correction, non-negative spike deconvolution; analyses use deconvolved traces |

### Processing Details
- Animals ran a closed-loop virtual linear corridor containing visual textures. Imaging captured V1 and higher visual areas simultaneously with a custom two-photon mesoscope and ScanImage.
- Imaging mice received a sound cue in every trial type. In task mice the cue marked reward availability in rewarded corridors; reward followed a post-cue lick, or was passively delayed in some animals. Unsupervised mice heard the cue but received no rewards.
- Source corridor position is represented on a 0–60 scale, while the physical corridor is 4 m; therefore physical metres are `source_position / 15`.
- Paper analyses use the supplied non-negative deconvolved fluorescence/spike traces directly. No delta-F/F recomputation is appropriate.
- Reference spatial analyses interpolate each trial to 60 corridor-position bins and use valid neural-frame-length behavior streams. The requested decoder differs by requiring temporal alignment to corridor entry and time-varying outputs, so native frame timing should be retained rather than replacing time with spatial bins.
- Training schedules differ by cohort: naive, unsupervised exposure, supervised/task training, second-stimulus learning, and grating exposure. `sess#` is heterogeneous and sometimes absent; date and experiment stage must be combined carefully for “day of training.”

### Curation Steps

**Neuron curation rules**:
Suite2p performed cell classification, ROI extraction, neuropil correction, and deconvolution before these released files. The paper states all analyses used deconvolved traces. The released/reference loader concatenates all three supplied arrays; no additional SNR or activity threshold is applied globally. Region-specific figure analyses later select visual areas or d-prime-defined cells, but these are analysis selections rather than recording-quality curation.

**Trial curation rules**:
Reference code uses neural-frame-length truncation and frame validity/VR movement masks for relevant analyses. Duplicate behavior experiment views are not duplicate physical trials and must be deduplicated by recording ID. Trials have complete start/end/cue fields; unrewarded trials legitimately have missing reward frames.

### Decoders Trained
| Decoded variable | Accuracy |
|------------------|----------|
| No directly comparable neural decoder for visual category, licking, position, or speed | Not reported |

The paper reports d-prime selectivity, coding-direction projections, sequence correlations, and similarity indices rather than classification accuracies. Therefore decoder accuracy must be assessed against categorical chance and internal validation, not a nonexistent paper benchmark.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Recording count | Metadata has 142 group entries | 89 unique neural files/physical IDs | 89 recordings in 19 mice | Deduplicate behavior views by physical `<mouse>_<date>_<block>` ID; retain 89 sessions. |
| Neural components | `load_spk` concatenates every array in `spks` | Three arrays/session; third may have 0–2 extra rows | 20,547–89,577 traces/recording | Concatenate all three exactly; resulting range matches paper exactly and retinotopy length. |
| Behavior versus neural frames | Reference slices behavior arrays to `nfr` | Behavior has a small trailing excess in many sessions | No conflicting statement | Truncate all frame streams to neural frame count before trial extraction. |
| Trial boundary | Reference uses frame-indexed trial streams | Fractional `StartFr`; floor can belong to previous trial | Corridor entry is alignment event | Start at `ceil(StartFr)`; end exclusively at `ceil(GrayFr)`. Spot checks show all included `ft_trInd` values belong to the target trial. |
| Duplicate stimulus views | Test3 uses swap-specific dictionary keys | Same physical trials appear in multiple views | Test3 compares swap variants | Merge concrete `TrialStim` labels across views. There are zero conflicts; 309 unresolved placeholders are consistently raw `circle3` and retain that label. |
| Position units | Spatial helper uses 60 source bins | `Corridor_Length=60`, texture corridor ends near 40 | Physical visual corridor is 4 m | Convert by 15 source units/metre; use 0–10, 10–20, 20–30, 30–40 source bins for 1 m categories. |
| Frame timestamps | `ft` is used for alignment | Median difference ~3.64e-6 | Imaging is mesoscope/ScanImage | `ft` is MATLAB datenum days; multiply differences by 86,400 to obtain ~0.315 s/frame (~3.18 Hz). |
| Training day | No single loader rule | `sess#` is missing for 8 sessions and conflicts across duplicate views | Cohorts have heterogeneous schedules | Use continuous elapsed calendar days from each subject's earliest included recording. Preserve experiment-stage metadata separately. |
| Anatomy | `neu_area_ID` maps V1/mHV/lHV/aHV | Some `iarea` values are outside mapped sets | Recordings span V1/HVAs | Use exact reference mapping and an explicit `unassigned` class rather than dropping ROIs. |
| Trial duration | No global duration filter | Median 23 frames; rare stalls up to 5,607 frames | No exclusion rule stated | Retain all 38,110 valid windows; do not invent a duration filter. |

### Final Understanding
- Paper, code, and data agree exactly on 19 mice, 89 physical recordings, and 20,547–89,577 concatenated ROI traces/session.
- Released neural values are Suite2p non-negative deconvolved traces already cell-classified and neuropil-corrected. No delta-F/F or additional universal quality threshold is required.
- The decoder-specific temporal representation legitimately differs from the paper's 60-position-bin figure analyses: it preserves native imaging frames and aligns each trial to corridor entry, as explicitly required by the task.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spks` list | `neural` | Concatenate three ROI x frame arrays; slice `[ceil(StartFr):ceil(GrayFr))`; store float32 | `load_spk` | No activity normalization, dF/F, or selectivity filtering. |
| `SoundFr`, frame index | `input[0]` time to sound cue | `(SoundFr - frame_index) * session_dt_seconds`, signed continuous series | `spk_2_cue` establishes cue-frame alignment | Positive before cue, zero near cue, negative after. |
| recording date + mouse | `input[1]` day of training | Elapsed calendar days from that subject's earliest included recording; broadcast per trial | metadata loader | Chosen over inconsistent/missing `sess#`. |
| frame index, `StartFr` | `input[2]` time since trial start | `(frame_index - StartFr) * session_dt_seconds`, continuous series | frame-aligned behavior | Starts near zero; uses fractional event timing. |
| `isRew` | `input[3]` reward availability | Boolean 0/1 broadcast across trial | behavior dictionary | Means rewarded corridor, not actual reward delivery. |
| merged `TrialStim`; fallback `WallName=circle3` | `output[0]` visual stimulus category | Eight integer categories | `get_cat_id`, metadata stimulus IDs | circle1, circle2, circle3, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2. |
| `LickFr` | `output[1]` licking | Binary raster; event assigned to `floor(LickFr)`, multiple events/frame collapse to 1 | `get_lick_raster` | Time-varying. |
| `ft_Pos` | `output[2]` corridor position | Clip physical corridor positions and digitize source units at 10,20,30 into classes 0–3 | `get_interpPos_spk` | Four equal 1 m bins over physical 4 m corridor. |
| `ft_RunSpeed` | `output[3]` running speed | Global corridor-frame quartile thresholds 0, 8.3812388, 30.1893780; classes 0–3 | frame behavior stream | Quartiles computed over all retained unique-session trial frames; repeated zero values make exact 25% class balance impossible. |
| mouse ID | `subjects`, `subject_idx` | Sorted unique mouse IDs and per-session indices | session metadata | 19 subjects. |
| retinotopy `iarea` | `brain_regions`, `brain_region_idx` | Exact `neu_area_ID`: V1=8; mHV=0/1/2/9; lHV=5/6; aHV=3/4; otherwise unassigned | `neu_area_ID` | Length must equal concatenated neural rows. |

### Key Decisions
1. **Temporal window**: Use visual corridor entry through visual-corridor exit, `[ceil(StartFr), ceil(GrayFr))`, aligned to corridor entry. This excludes inter-trial/gray-space activity and preserves the full 4 m trial requested.
2. **Native temporal sampling**: Keep one bin per imaging frame (~315 ms), because neural and behavior are already synchronized at this rate and no paper temporal rebinning is specified.
3. **All variables time-aligned**: Broadcast per-trial day/reward/stimulus values to trial length because the validator and trainer require equal neural/input/output time axes and score concatenated timepoints.
4. **Stimulus reconciliation**: Merge all duplicate behavior views before choosing category. Concrete labels never conflict. Use raw `circle3` only for the 309 trials where all views contain placeholders.
5. **Day definition**: Calendar days since each mouse's earliest recording is reproducible, continuous, and complete. `sess#` cannot be used because it conflicts for the same physical recording and is missing for eight sessions.
6. **Lick rasterization**: `LickFr` is a fractional frame coordinate; `floor` selects the imaging interval containing the lick. Use binary presence, not lick count.
7. **Speed discretization**: Compute global thresholds once from all retained corridor frames, matching “each corresponding to 25% of the data.” Keep negative estimated speeds in the lowest class rather than altering raw behavior.
8. **No invented filtering**: Retain all 89 sessions, all 38,110 valid trials, all concatenated Suite2p traces, and rare long/stationary trials because neither code nor paper specifies exclusion.
9. **Storage**: Use float32 for neural/input and compact integer outputs/indices. Conversion loads and writes one session at a time conceptually, but final pickle necessarily contains all sessions.

### Planned Sanity Checks
- [ ] `np.allclose` converted neural trial slices to a direct raw `np.concatenate(spks)[:, start:end]` spot check.
- [ ] `np.allclose` converted input timing/reward series to direct raw event/frame calculations.
- [ ] `np.allclose` converted output position/speed/lick/category to direct raw behavior calculations.
- [ ] Assert 19 subjects, 89 sessions, 38,110 trials before any unavoidable validator filtering.
- [ ] Assert each trial has identical neural/input/output time length and every session has at least two trials.
- [ ] Assert retinotopy region vector length equals concatenated neuron count for every session.
- [ ] Compare converted neuron/session range to paper's exact 20,547–89,577.
- [ ] Check class ranges/distributions, finite values, cue crossing, and quartile thresholds.

---

## Step 6: Script Development
**Status**: COMPLETE

Implementation notes:
- Implemented `/app/convert_data.py` with required `--full`, `--sample`, and `--show-processing` modes, modular loading/mapping/validation functions, one-session-at-a-time source loading, timing output, and diagnostic plots.
- First sample iteration failed after shape/speed scanning because `rsplit('_', 2)` misparsed underscore-delimited dates in physical session IDs. No output pickle was written. Fixed by adding a strict regex parser for `<mouse>_<YYYY>_<MM>_<DD>_<block>` and unit-checking representative IDs.

Code inefficiencies identified:
- Native object NPY files cannot be memory-mapped and total 434 GB; eager loading all sessions would be wasteful.
- Trial copies dominate output size, while repeated behavior views can be merged before neural loading.

Code speedups added:
- Load and release one source neural session at a time. The initial full run projected roughly 24 minutes because it concatenated/copied each full multi-GB session before trial slicing; it was stopped after seven sessions as required (>1.5x estimate). Optimized to concatenate the three reference components only for retained trial slices, avoiding a redundant full-session copy, and to pre-rasterize licks once per session.
- Use vectorized frame arrays/digitization, compact int16 outputs/regions, float32 neural/input arrays, and protocol-4 pickle.
- Initial sample mode selected the two smallest source sessions and converted 953 trials in 13.4 s (2.114 GB), but both were unrewarded/no-lick naive sessions. Although verification was warning-free, this made two decoder targets constant and was unsuitable for Step 8.
- Revised sample mode selects the two smallest estimated converted sessions that each contain both reward classes and corridor licking (TX60_2021_04_10_1 and TX109_2023_03_27_1), preserving computational efficiency while testing every target.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total across sessions) | 112,357 ROI traces |
| Neurons / session | 48,732; 63,625 |
| Subjects | 2 (TX60, TX109) |
| Sessions / subject | 1 each |
| Trials (total) | 321 |
| Trials / session | 237; 84 |
| Trial frames | mean of session means 77.45; range 12–1,285 |
| Time to cue range | [-333.1, 400.1] s |
| Day input range | [0, 11] calendar days from subject first recording |
| Reward availability | Both 0 and 1 in each session |
| Visual category distribution | circle1 0.615, leaf1 0.385 by frame |
| Licking distribution | not licking 0.897, licking 0.103 |
| Position distribution | [0.317, 0.237, 0.205, 0.242] |
| Speed-bin distribution | [0.120, 0.380, 0.250, 0.250] |

### Format Validation
- `/app/verification_sample_out.txt`: “Data format is valid, no errors or warnings.”
- Neural/input/output dimensions and trial lengths are consistent.
- Every requested output varies. The revised representative sample replaced an initial structurally valid but all-unrewarded/no-lick sample.
- The speed 25th percentile is exactly zero in this sample. Tied zero values cannot be split without arbitrary/random tie breaking, hence classes 0/1 are 12%/38% while upper classes are exactly 25% each. This is expected and preserves deterministic value-based bins.

### Processing Plots Review
- Two 1560×1820 diagnostic PNGs were generated, one per sample session.
- Plots show raw deconvolved traces, signed cue timing crossing zero, monotonically increasing time from corridor entry, binary lick events, monotonic four-bin position progression, and valid speed classes. No temporal offset or discretization anomaly was found.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| One-session-at-a-time source loading and immediate source release | Avoids loading 434 GB simultaneously |
| Vectorized digitization/time construction and compact dtypes | Conversion of 321 trials completed in 27.8 s |
| Global behavior merge before neural processing | Avoids repeated physical-session conversion |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Shape/source scan | ~2–8 s/session depending file size | ~7.5 min based on measured all-session scan |
| Trial conversion and plots excluded | ~7 s/session for representative sample | ~10 min conservative including serialization; expected under 15 min |

Sample output size is 4.415 GB. Full size is expected to be large (hundreds of GB) because all 4.69 million released traces and all retained trial frames are intentionally preserved.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings from format verifier: None
- Training emitted sklearn's “y_pred contains classes not in y_true” warning because the global eight-class vocabulary contains visual classes absent from this two-session sample/split. This is unavoidable without falsifying the global class schema and does not occur because of invalid labels.

### Training Progress
- CUDA training completed all 200 epochs.
- Loss decreased from 5,234.60 (epoch 1) to 137.52 (epoch 200); test loss 50.50.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-----------------------|-------------------------|
| Visual stimulus category | 0.9076 | 0.8038 |
| Licking | 0.7476 | 0.6756 |
| Corridor position bin | 0.7537 | 0.5901 |
| Running speed quartile | 0.6439 | 0.6099 |

All validation accuracies exceed uniform chance (0.125, 0.5, 0.25, and 0.25 respectively). The particularly strong category/position/speed performance and above-chance lick decoding support correct temporal alignment.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 296.435 GB decimal (277 GiB), created successfully.
- `conversion_full_out.txt`: created; local validation passed 89 sessions and 38,110 trials.
- `verification_full_out.txt`: created; “Data format is valid, no errors or warnings” and “Data verification complete.”

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons/ROI traces | up to ~90k/recording | concatenate all 3 `spks` arrays | 4,691,034 session-ROI traces | 4,691,034 | Yes |
| Mean neurons/session | not reported | all released arrays | 52,708.25 | 52,708.25 | Yes |
| Neuron range/session | 20,547–89,577 | all released arrays | 20,547–89,577 | 20,547–89,577 | Exact |
| Subjects | 19 | metadata | 19 | 19 | Exact |
| Sessions | 89 recordings | 89 unique physical IDs | 89 | 89 | Exact |
| Trials (total) | not global | unique physical sessions | 38,110 | 38,110 | Exact |
| Trials/session | not global | no global trial filter | mean 428.2; 84–789 | same | Exact |
| Retained temporal frames | not global | corridor trial windows | 1,375,142 | 1,375,142 | Exact |
| Rewarded trials | not global | `isRew` | 4,336/38,110 (11.38%) | preserved | Yes |
| Position bins | four physical 1-m bins required | source position 0–40 | thresholds 10/20/30 | fractions 0.285/0.233/0.235/0.247 | Yes |
| Speed bins | quartiles required | raw `ft_RunSpeed` | thresholds 0/8.3812/30.1894 | all classes 0–3 | Yes |
| Visual categories | circle/leaf and test variants | `TrialStim`, metadata | 8 merged classes | all 8 present | Yes |
| Brain regions | V1 and HVAs | `neu_area_ID` | exact per-ROI codes | V1/mHV/lHV/aHV/unassigned | Yes |

### Runtime and Optimization
- First full attempt was stopped after seven sessions because projected runtime (~24 min plus serialization) exceeded 1.5× the estimate.
- Replaced full-session neural concatenation with mathematically equivalent per-trial concatenation of component slices and pre-rasterized licks once/session. Representative conversion fell from 27.8 s to 12.1 s with identical verified output.
- Optimized full conversion took 1,609.8 s; final serialization of the unavoidable 296.435 GB complete dataset took 434.1 s. The runtime exceeded 15 minutes because preserving all released traces creates a 296 GB target, not because of redundant source work.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Searched `verification_full_out.txt` for errors, warnings, tracebacks, and invalid values. Result: none; validator reports valid data and completes successfully.
2. **Independent raw neural sanity check**: Loaded original `spks` files directly, concatenated the three arrays, and used `np.allclose` against converted trial 5 and each session's final trial for TX60_2021_04_10_1 and TX109_2023_03_27_1. Result: exact agreement, including specific neuron/timepoint spot checks.
3. **Independent raw input sanity check**: Recomputed cue-relative seconds, subject-day broadcast, fractional-start-relative seconds, and reward availability directly from raw behavior. `np.allclose` passed for all tested trials.
4. **Independent raw output sanity check**: Recomputed visual labels, floor-rasterized licks, 10/20/30 position thresholds, and metadata-recorded speed thresholds directly from raw behavior. `np.allclose` passed for all tested trials.
5. **Independent anatomy check**: Loaded raw retinotopy NPZ files and applied the explicit reference `iarea` mapping. `np.allclose` passed for both complete sample-session region vectors.
6. **Key-statistics comparison**: Converted 89 sessions/19 mice and exact 20,547–89,577 ROI range match the paper. Converted 38,110 trials, 4,691,034 session-ROI traces, and region counts match independently aggregated raw data.
7. **Edge audit**: Audited all 38,110 original trial windows for positive length, finite position/speed, frame trial identity, cue placement, and session trial count. All sessions have at least 84 trials. Rare long trials were inspected rather than filtered; they reflect prolonged low-movement/stalled traversal, not indexing across trials.

### Reference Code Comparison
| Processing stage | Reference code/method | Conversion implementation | Comparison/result |
|------------------|-----------------------|---------------------------|-------------------|
| Data loading | `load_spk` loads `*_neural_data.npy` and concatenates every `spks` array | `convert_session` loads same file and concatenates the same component slices in identical neuron order | Equivalent; raw `np.allclose` passed. |
| Neuron filtering | Released Suite2p cell-classified deconvolved traces; no global code filter | All released rows retained; no invented SNR/activity filter | Match. Figure-specific d-prime/area selection intentionally not applied. |
| Trial filtering | Reference uses physical session behavior and neural-length frame streams | Duplicate analysis views deduplicated; all positive corridor windows retained | Match physical data; no unsupported exclusion. |
| Temporal alignment | Behavior events are fractional imaging-frame indices; code uses frame masks/truncation | `ceil(StartFr)` through exclusive `ceil(GrayFr)`, clipped to neural frames | Spot checks show floor would include previous trial; chosen boundary is correct. |
| Binning | Reference spatial figures interpolate to 60 position bins | Native ~315 ms imaging frames retained | Required difference: decoder task is temporally aligned, not spatially resampled. |
| Input construction | Cue, reward condition, frame position/time available in behavior | Signed cue time, subject-relative calendar day, elapsed trial time, `isRew` | Matches requested decoder inputs; day rationale documented because `sess#` is inconsistent. |
| Output construction | Reference derives category from stimulus/wall, lick events, position, speed | Merged nonconflicting semantic labels; binary lick frames; four 1-m position bins; global speed quartiles | Matches requested categorical outputs and source semantics. |

### Key Statistics and Edge Cases
- Full output fractions: visual category = circle1 0.331, circle2 0.055, circle3 0.007, leaf1 0.346, leaf2 0.164, leaf3 0.047, swap1 0.025, swap2 0.026; licking = 0.035 positive; position = 0.285/0.233/0.235/0.247; speed = 0.098/0.402/0.250/0.250.
- Speed lower-bin imbalance is caused by a point mass at exactly zero, equal to the 25th-percentile threshold. Deterministic value bins cannot split tied zeros; no random tie-breaking was introduced.
- Region counts: V1 1,833,035; mHV 1,108,860; lHV 495,318; aHV 668,180; unassigned 585,641; sum 4,691,034.
- Extremely long trial windows are genuine stalled runs. They are retained because frame trial IDs remain correct and no paper/code curation rule excludes them.
- Full trial end indices are at least 26 frames before neural stream end; there is no final-trial clipping loss.

### Issues Found and Resolved
- **Session ID parsing**: Initial sample attempt misparsed underscore-delimited dates; fixed with strict regex and reran conversion/validation.
- **Unrepresentative initial sample**: Two smallest files lacked reward/lick variation; replaced with two small rewarded sessions and reran all sample steps.
- **Full conversion efficiency**: Initial implementation copied full session arrays; stopped per protocol, changed to equivalent retained-slice concatenation, benchmarked identical verified outputs, and reran all full steps.
- No unresolved data or validation warnings remain. The sklearn sample warning concerns absent global visual classes in a two-session split, not conversion validity.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Initial CUDA setup completed all 89 session projections but exhausted GPU memory before optimization. The provided decoder automatically retried on CPU.
- CPU fallback completed all 89 session projections and all 200 epochs.
- Loss decreasing: Yes. Epoch 1 = 18,642.40; epoch 100 = 480.10; epoch 180 = 181.59; epoch 200 = 193.89. The small late fluctuation does not alter the approximately 99% overall reduction.
- Test loss: 146.76.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-----------------------|-------------------------|-------|
| Visual stimulus category | 0.5360 | 0.4939 | 3.95× uniform chance |
| Licking | 0.8886 | 0.8565 | 1.71× chance |
| Corridor position bin | 0.3338 | 0.3270 | 1.31× chance; investigated in Step 12 |
| Running speed quartile | 0.4326 | 0.4308 | 1.72× chance |

All requested outputs are above chance. Training finished successfully and `/app/train_decoder_full_out.txt` contains the complete run.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Achieved Validation Accuracy | Chance | Multiple of Chance | Expectation from Paper |
|----------|------------------------------|--------|--------------------|------------------------|
| Visual stimulus category | 0.4939 | 0.125 | 3.95× | No comparable classifier accuracy reported |
| Licking | 0.8565 | 0.500 | 1.71× | No comparable classifier accuracy reported |
| Corridor position bin | 0.3270 | 0.250 | 1.31× | No comparable classifier accuracy reported |
| Running speed quartile | 0.4308 | 0.250 | 1.72× | No comparable classifier accuracy reported |

### Checks Performed
1. **Accuracy versus chance**: Every output is above chance. Visual, licking, and speed exceed 1.5× chance. Position is 1.31× chance and was investigated in detail.
2. **Accuracy comparison to paper**: Exhaustive paper search found no neural classification accuracy for these requested variables. The paper reports d-prime, coding directions, sequence correlations, and similarity indices; it would be misleading to equate those values with balanced classification accuracy.
3. **Train-validation gap**: Train/validation ratios are visual 1.085, licking 1.037, position 1.021, and speed 1.004. None approaches the 1.5× overfitting threshold.
4. **Three-trial raw position audit**: Directly loaded TX60 trial 5, TX109 trial 5, and TX123 trial 100. Included frame IDs equal their target trials, position is monotonic in each, raw positions span the 0–40 corridor, and digitization yields expected 0–3 classes.
5. **Global variation**: Direct raw counts are 391,309 / 320,416 / 323,595 / 339,822 (fractions 0.28456 / 0.23301 / 0.23532 / 0.24712), exactly matching converted validation statistics. No 99%-dominant position class exists.
6. **Temporal alignment**: Cue positions fall inside reviewed corridor windows; `ceil(StartFr)` avoids the previous-trial frame. Independent neural/output `np.allclose` checks already passed in Step 10.
7. **Neural filtering/reference processing**: All released Suite2p cell-classified deconvolved traces are retained exactly as the reference loader specifies.

### Interpretation of Position Accuracy
The 0.3270 validation balanced accuracy is above chance but below 1.5× chance. No label, binning, trial-boundary, variation, filtering, or alignment bug was found. Position is the finest-grained time-varying target and the provided decoder compresses each 20,547–89,577-neuron session through a random 2,000-dimensional projection; modest accuracy is therefore plausible. Changing labels, leaking time inputs into decoder predictors, filtering slow trials, or spatially resampling neural data solely to inflate accuracy would violate the requested conversion.

### Issues Found and Resolved
- CUDA exhausted memory after session initialization; the provided script automatically retried on CPU and completed all 200 epochs.
- No Step 12 conversion issue was found. No reconversion was necessary.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
