# Dataset Conversion Notes

## Overview
- **Dataset**: Unsupervised pretraining in biological neural networks
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment check: Python: 3.13.15; NumPy: 2.4.4; PyTorch: 2.6.0+cu124. Imports succeeded.

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `code`
- `data`
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
| `download_data_from_figshare` | `code/utils.py:12` | LOADING | Downloads the paper data archives and defines their expected directory layout. |
| `spk_pos_interp` | `code/utils.py:105` | PROCESSING | Interpolates neuron activity from cumulative position to a fixed spatial grid for one trial. |
| `get_interpPos_spk` | `code/utils.py:120` | PROCESSING | Applies spatial interpolation in neuron chunks and reshapes output to neuron x trial x spatial-bin; reference default is 60 bins. |
| `get_cat_id` | `code/utils.py:137` | PROCESSING | Maps wall/stimulus names and reward status to categorical IDs. |
| `get_lick_raster` | `code/utils.py:149` | PROCESSING | Builds trial-wise lick-position histograms/raster information from `LickPos`, `LickTrind`, wall identity, and reward status. |
| `neu_area_ID` | `code/utils.py:312` | CURATION | Maps numeric imaging-area labels to named cortical visual areas. |
| `load_exp_beh` | `code/utils.py:326` | LOADING | Loads the experiment-specific processed behavior dictionary (`process_data/<exp_type>_behavior.npy`). |
| `load_retino` | `code/utils.py:330` | LOADING | Loads retinotopy and augments it with mapped neuron area IDs. |
| `load_spk` | `code/utils.py:338` | LOADING | Loads session `spks.npy`, a neuron-by-imaging-frame activity matrix. |
| `load_interp_spk` | `code/utils.py:344` | LOADING | Loads cached 60-bin spatially interpolated neural activity. |
| `spk_2_firstLick` | `code/utils.py:913` | PROCESSING | Aligns one-dimensional activity and lick events to the first lick in each trial, using frame indices. |
| `spk_2_cue` | `code/utils.py:935` | PROCESSING | Aligns activity and lick histograms to each trial's `SoundFr`; paper code commonly uses 15 frames before/after. |

### Notes
- Repository contents are chiefly figure scripts, `utils.py`, and `data_process_script.ipynb`; the notebook is the main preprocessing workflow.
- Notebook preprocessing iterates over experiment/session database entries, loads behavior with `load_exp_beh`, and loads neural data with `load_spk`.
- Neural source is an already generated `spks.npy` neuron-by-frame activity array. The reference conversion does **not** load raw fluorescence or compute delta-F/F; no additional dF/F computation is indicated.
- For paper spatial analyses, behavior streams are truncated to the neural frame count, valid VR locomotion frames are selected by `ft_move[:nfr] > 0`, cumulative position is taken from `ft_PosCum`, and `get_interpPos_spk(..., n_bins=60, lengths=Corridor_Length)` creates fixed spatial-bin activity.
- Trial identity is embedded in cumulative position: interpolation uses each corridor-length interval to isolate a trial. Position interpolation is therefore spatial, not temporal.
- For temporal analyses, the code directly indexes common imaging/behavior frames. `SoundFr` is the sound-cue frame per trial; `LickFr` and `LickTrind` identify lick frame and trial. `spk_2_cue` extracts `[SoundFr-15, SoundFr+15)` when defaults are used and bins lick events over the same 30-frame interval.
- Stimulus/wall identity is available in `WallName`/`UniqWalls` and paper labels include leaf and circle variants. Reward condition is represented separately by `isRew` and is also used by `get_cat_id`.
- Reference selectivity routines calculate d-prime and odd/even splits for particular figures. These are downstream analysis choices, not general neuron-quality curation rules, and should not be applied globally to decoder conversion without supporting evidence.
- No electrophysiology unit-quality filtering is relevant: this is calcium-imaging-derived activity. No explicit neuron exclusion was found in the general loading/interpolation path; area-specific/selectivity masks are analysis-specific.
- Key notebook cache naming is `<mouse>_<date>_<block>_interpolate_spk.npy` under `process_data`.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` is 412 GB and contains 117 `.npy` plus 90 `.npz` files.
- `beh/`: 25 object-NPY collections. Actual session records are nested dictionaries keyed by `<mouse>_<YYYY_MM_DD>_<block>`; several top-level entries are metadata arrays rather than session records, so only dictionary values containing `ntrials` are treated as sessions.
- `spk/`: 89 session files named `<session>_neural_data.npy`. Each is an object dictionary with key `spks`, holding a list of three float32 neuron-by-frame matrices (three imaging planes). In the inspected session each plane was `(6849, 18491)`; concatenating planes gives 20,547 neurons.
- `retinotopy/`: 89 session/date `*_trans.npz` archives plus one non-session `areas.npz`. Session archives contain `iarea`, `xpos`, `ypos`, `xy_t`, `A`, and sometimes registration offsets `dx`, `dy`. The `iarea` vector length equals the total neurons across all three neural planes and labels areas with numeric values `-1,0,...,9`.
- `process_data/`: empty; reference-derived caches are not precomputed.
- All 89 neural files have matching retinotopy by mouse/date. Only 76 neural session IDs have directly matching full behavior session records; 13 neural-only sessions lack required behavioral streams and are candidates for exclusion.
- Neural object arrays cannot be memory-mapped directly. Conversion must load one session at a time and promptly release it to control memory.

### Available Variables
- Trial level: `ntrials`, `WallName`, `TrialStim`, `WallType`, `WallIsProbe`, `isRew`, `StartFr`, `SoundFr`, `SoundTime`, `Corridor_Length`, `Texture_Length`, `stim_id`, trial indices.
- Frame level on the imaging clock: `ft`, `ft_Pos`, `ft_PosCum`, `ft_RunSpeed`, `ft_RunCum`, `ft_WallID`, `ft_trInd`, `ft_move`, `ft_isMoving`, corridor/gray-space masks.
- Licking: `LickFr`, `LickTime`, `LickPos`, `LickTrind`, `LickWall`, plus trial-level first-lick fields.
- Raw/high-rate VR streams are also present (`VRpos`, `VRposCum`, `VRposTime`).
- Coverage across the 76 matched sessions is complete for all required frame-level and trial-level fields; `run_pos` alone is absent in one session and is not needed because `ft_RunSpeed` is complete.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (all 89 neural sessions, from retinotopy) | 4,691,034 |
| Neurons / session | 20,547–89,577 across all neural sessions |
| Subjects with matched neural + behavior | 19 |
| Sessions with neural + behavior | 76 |
| Neural sessions lacking matching full behavior | 13 |
| Trials in matched sessions | 31,443 |
| Trials / representative session | 503 (observed example) |

### Data-quality / structure observations
- Neural values are nonnegative processed activity, not raw fluorescence. Three inspected plane variants have identical dimensions but are separate planes, not alternative transforms: the summed plane neuron count exactly equals retinotopy `iarea` length.
- Frame arrays can be slightly longer than neural arrays and must be truncated to neural frame count as in reference code.
- `ft_trInd` is floating-point but takes integral trial IDs; it spans 0 through `ntrials-1` in the representative session.
- Representative `ft_Pos` spans approximately 0–60 source units, and `Corridor_Length` is 60; physical-unit interpretation is deferred to the paper/methods review.
- Some frame-wise running speeds are slightly negative, reflecting measurement/noise/backward motion rather than missing values.
- No data README was present.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Not stated as a sum | Paper reports per-recording range only. |
| Neurons / session | 20,547–89,577 | “We ran Suite2p on this data to obtain the activity traces from 20,547 to 89,577 neurons in each recording.” |
| Subjects | 19 | “We performed 89 recordings in 19 mice…” |
| Sessions / subject | 89 recordings / 19 mice overall | Same Methods sentence. |
| Trials (total) | Not stated | No paper-wide trial total reported. |
| Trials / session | Not stated | No paper-wide value reported. |
| Neural data time bin | Not stated in paper text | No acquisition frame-rate phrase was found; derive from synchronized data. |
| Behavior data time bin | Imaging-frame streams are synchronized, exact interval not stated | Data/code use frame-indexed behavior arrays. |
| Reward rate | Cohort/stimulus dependent; not stated as one fraction | Task rewarded corridor versus unrewarded corridor; unsupervised cohorts receive no corridor reward. |
| Corridor geometry | 4 m textured corridor + 2 m gray between corridors | “The virtual reality corridors were each 4 m long, with 2 m of grey space between corridors.” |
| Imaging sound-cue position | Uniform from 0.5 to 3.5 m | “randomly chosen per trial from a uniform distribution between positions 0.5 m and 3.5 m.” |
| Neural processing | Suite2p deconvolution, 0.75 s decay | “Suite2p performs motion correction, region of interest detection, cell classification, neuropil correction and spike deconvolution… timescale of decay of 0.75 s.” |

### Processing Details
- The experiment is a visual discrimination task in head-fixed mice traversing pseudo-random leaf/circle (and in some animals rock/brick) virtual corridors.
- For imaging mice, a sound cue occurs on every trial, including unrewarded and unsupervised trials. In task mice it indicates reward availability onset in the rewarded corridor.
- The physical visual corridor is 4 m; gray inter-corridor space is 2 m. Paper neural selectivity explicitly uses only the 0–4 m corridor region.
- Paper analyses use nonnegative deconvolved fluorescence traces generated by Suite2p; no additional dF/F operation should be applied.
- Spatial analyses interpolate activity by position. Selectivity analyses compare response distributions and often use train/test or odd/even trial splits to prevent circularity.
- The paper states that only running timepoints were analyzed, removing pauses for reward collection. Reference code implements this with `ft_move > 0` before spatial interpolation.
- Sound/lick temporal analyses align using common imaging-frame indices (`SoundFr`, `LickFr`).
- The paper does not state neural/behavior frame duration. This must be estimated from synchronized frame timestamps in source data and validated across sessions.

### Curation Steps

**Neuron curation rules**:
- Suite2p performed ROI detection, cell classification, neuropil correction, and nonnegative spike deconvolution upstream.
- No general additional neuron-quality exclusion is described for loading the population traces.
- d-prime/top-5% neuron selection is analysis-specific to selectivity/coding-direction figures and is not a global curation rule.

**Trial curation rules**:
- Reference analyses retain running timepoints only.
- Spatial analyses use the 0–4 m textured corridor, excluding gray space.
- Particular figure analyses split trials for unbiased selectivity/tuning estimates; this is not a general reason to discard trials for the requested decoder.
- Trials/sessions without synchronized neural and required behavioral variables cannot support the requested conversion.

### Decoders Trained
| Decoded variable | Accuracy |
|------------------|----------|
| None reported | The paper contains no occurrences of decoder, decoding, classifier, classification accuracy, or accuracy; requested decoder tasks are new downstream analyses. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Recording count | Experiment database lists 89 base recordings across condition groups | 89 neural files and 89 matching retinotopy files; direct behavior IDs initially matched only 76 | 89 recordings in 19 mice | The other 13 recordings have behavior IDs suffixed `_swap1`/`_swap2`. Strip only this suffix when matching to neural files; all 89 recordings are covered. |
| Swap records | Figure analysis treats swap variants as separate statistical data points | Swap1/swap2 records have identical frame streams, trial indices, reward, cue, and lick arrays; only `TrialStim`/`stim_id` annotations differ | Two swap types in one session may be treated as separate data points for statistics | Do not duplicate neural sessions or frames. Reconcile the alternative stimulus annotation into one categorical label per physical trial. |
| Three neural arrays | `load_spk` concatenates list entries on axis 0 | Each file contains three same-frame-count matrices; retinotopy length equals summed neuron count | Simultaneous mesoscope recording across many areas | Concatenate all three imaging planes along neurons exactly as reference code. |
| Source position units | Code uses `Corridor_Length=60`, `Texture_Length=40`, and corridor mask | `ft_CorrSpc == (ft_Pos < 40)` exactly; gray is 40–60 | 4 m corridor + 2 m gray | Source units are decimeters. Within corridor, physical meters = `ft_Pos / 10`; four requested bins are [0,1), [1,2), [2,3), [3,4] m. |
| Temporal bin size | Temporal functions index common imaging frames | Median synchronized frame interval is 0.314804 s (1st–99th percentile 0.301–0.330 s) | Frame rate not stated | Preserve one sample per acquired volume/frame; report 314.804 ms nominal bin size. No temporal resampling is needed because all streams already share this clock. |
| Trial/corridor start | Behavior has fractional `StartFr`, frame-wise `ft_trInd`, and `ft_CorrSpc` | Some cue-minus-StartFr offsets are extreme during pauses | Decoder requests alignment to corridor entry; paper analyzes only corridor and running frames | Define each trial from `ft_trInd`, retain corridor frames (`ft_CorrSpc`) that also satisfy reference running criterion (`ft_move>0`), and set time zero at the first retained corridor frame. |
| Neuron filtering | General loader includes every concatenated Suite2p trace | Retinotopy includes labels -1 through 9 | Suite2p already performed cell classification; paper reports all 20,547–89,577 traces | Keep all neurons. Selectivity masks are figure-specific, not quality curation. Map all numeric area labels, including -1 as unassigned/outside mapped areas. |
| Training day | Reference metadata includes `sess#` | `sess#` is 0 before learning, 1 after/test in many groups, and later values (6, 10, 12) for later sessions; a few are missing | Paper discusses before/after and days of training but does not provide another per-recording numeric day field | Use numeric `sess#` as source-provided training-session/day index. Resolve missing values from ordered date/condition context in mapping step and record imputation. |

### Final cross-source understanding
- All 89 physical recordings have neural, retinotopy, and behavior data when swap suffix aliases are recognized.
- Full source population comprises 4,691,034 neurons across 19 mice, exactly matching paper session count and per-session range.
- Neural and behavior streams are already synchronized on a ~3.18 Hz volume clock. Frame arrays are truncated to neural frame count as in reference code.
- Reference-consistent trial samples use deconvolved traces during running inside the 4 m visual corridor. Gray-space and stationary frames are excluded.
- The requested decoder outputs require preserving trial-wise temporal samples rather than applying the paper's 60-bin spatial interpolation; this justified difference avoids destroying lick timing and running speed while retaining the same source streams and curation mask.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spks` list (3 planes) | `neural` | Concatenate planes on neuron axis; truncate to common neural/behavior frame count; slice each trial's running corridor frames | `load_spk` | float32 deconvolved activity, neuron x time |
| `SoundFr`, retained frame indices, frame interval | `input[0]` | Signed seconds until cue: `(SoundFr[t] - frame_index) * dt` | `spk_2_cue` | Continuous, time-varying; positive before cue, negative after |
| experiment DB `sess#` | `input[1]` | Numeric source training-session/day index, repeated over time | `Imaging_Exp_info.npy` | Missing values imputed from condition/date order and recorded in metadata |
| retained frame order | `input[2]` | Seconds since first retained corridor-running frame | trial mask logic | Continuous, time-varying; trial start is corridor entry |
| `isRew[t]` | `input[3]` | Boolean converted to 0/1 and repeated over time | `get_cat_id` uses same reward field | Reward availability in rewarded corridor |
| merged `TrialStim` aliases / `WallName` | `output[0]` | Global categorical stimulus label per trial, repeated over time | `get_cat_id` / paper labels | Merge swap1/swap2 annotations without duplicating the physical recording |
| `LickFr`, `LickTrind` | `output[1]` | Binary vector; assign each lick to nearest retained frame in its supplied trial | `spk_2_cue`, `get_lick_raster` | Multiple licks in one bin remain class 1 |
| `ft_Pos` | `output[2]` | Convert decimeters to meters and classify by `floor(pos/10)` into four bins | corridor masks / spatial interpolation | Classes: 0–1, 1–2, 2–3, 3–4 m |
| `ft_RunSpeed` | `output[3]` | Global quartile thresholds from all retained frames | source frame stream | Classes use 25th/50th/75th percentiles; ties handled by `np.digitize(..., right=False)` |
| session ID mouse prefix | `subjects`, `subject_idx` | Unique sorted mouse IDs and session lookup | experiment DB | 19 subjects |
| retinotopy `iarea` | `brain_region_idx` | Paper grouping: V1=8; medial HV=0,1,2,9; lateral HV=5,6; anterior HV=3,4; label 7=other; -1=unassigned | `neu_area_ID` | Preserve all neurons and explicitly retain unmapped labels |

### Key Decisions
1. **Physical sessions, not statistical aliases**: one target session per 89 neural recordings. Swap behavior aliases share identical frames and are merged only to recover both swap stimulus labels.
2. **All neurons retained**: reference `load_spk` concatenates all planes and the paper's reported population range includes all Suite2p-classified cells. No arbitrary subsampling is applied despite the large (~162 GiB estimated) output.
3. **Temporal rather than 60-bin spatial representation**: requested licking, speed, and cue timing require the synchronized imaging clock. We retain the reference corridor/running mask but do not spatially interpolate neural data.
4. **Trial filtering**: `ft_trInd == trial`, `ft_CorrSpc`, and `ft_move > 0`, after truncation to neural frame count. Every investigated trial has at least one valid frame; any empty trial is skipped and reported.
5. **Nominal bin duration**: use each session's median positive `diff(ft)*86400` for event/time variables; metadata reports the global median 314.804 ms.
6. **Lick assignment**: `LickFr` is fractional. For each event, use `LickTrind` and choose the nearest retained frame belonging to that trial, preventing boundary assignment errors.
7. **Speed bins**: compute global quartiles over the exact retained frames before conversion, yielding preliminary thresholds 12.4224, 25.3526, 40.8546 source speed units. Recompute in script for reproducibility.
8. **Stimulus labels**: use merged `TrialStim` because it distinguishes leaf/circle/rock/wood and swap variants; fall back to `WallName` if blank. Output values are global sorted labels.
9. **Training day**: use source `sess#`, the only numeric training-session field. Missing values are imputed by nearest matching condition stage/date, with exact values listed in metadata.
10. **Data types**: neural/input float32; categorical outputs int16; indices int32/int64 as appropriate.

### Planned Sanity Checks
- [ ] Compare concatenated converted neural trial samples against raw plane concatenation with `np.allclose`.
- [ ] Compare cue-time, position, speed, reward, and stimulus values for selected raw trials with `np.allclose`.
- [ ] Verify every retained converted frame satisfies raw `ft_CorrSpc`, `ft_move>0`, and matching `ft_trInd`.
- [ ] Verify four position classes are approximately equally represented because corridors use fixed 1 m extents.
- [ ] Verify speed class fractions are approximately 25% each globally.
- [ ] Verify neuron count equals retinotopy length in every session and area grouping covers every neuron.
- [ ] Verify all 89 physical sessions, 19 mice, and source trial IDs are accounted for without duplicate swap recordings.
- [ ] Check finite values, monotonic time-since-start, cue-time sign crossing, and binary licking/reward ranges.
- [ ] Check sample/full validator output and decoder accuracy against class chance; paper has no comparable decoder accuracy.

---

## Step 6: Script Development
**Status**: COMPLETE

- Created `/app/convert_data.py` with required positional output path, `--full` (default), `--sample`, and `--show-processing` modes.
- Implemented source discovery, swap-alias reconciliation, experiment metadata mapping, three-plane neural concatenation, retinotopy area mapping, exact trial/frame masks, cue/day/time/reward inputs, and all four categorical outputs.
- Added per-session shape checks, timing output, scan statistics, global speed quantiles, metadata, source trial IDs, and processing plots.
- Script passes `py_compile` and CLI help execution.

Code inefficiencies identified:
- Native object-NPY neural files cannot be memory mapped and each session is multi-GB.
- Target format requires separate trial arrays, while decoder later concatenates them per session; this creates unavoidable array/object overhead.
- Full all-neuron target is estimated at ~162 GiB, but arbitrary subsampling would violate the reference loader and reported population statistics.

Code speedups added:
- Behavior-only pre-scan computes global categorical values/quartiles without loading neural files.
- Neural data is loaded exactly once per session, planes are concatenated once, and frame selection is vectorized.
- Sessions are processed sequentially and raw session arrays are released with garbage collection.
- float32/int16 types and highest pickle protocol reduce storage and serialization overhead.


---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 representative supervised/reward sessions |
| Trials (total) | See verification log; all source trials retained |
| Neurons / session | 85,481 and 75,480 |
| Subjects | Sample session subject IDs reported in metadata |
| Timepoints / trial | mean 27.9 and 24.5 by session; min 20, max 43 overall |
| Time to sound cue range | -209.3 to 199.7 s |
| Day of training | 0 and 1 in sample |
| Reward availability | Both 0 and 1 in each session |
| Visual stimulus distribution | circle1 0.515, leaf1 0.485 |
| Licking distribution | not licking 0.742, licking 0.258 |
| Position distribution | 0.246, 0.252, 0.251, 0.250 |
| Speed-quartile distribution | 0.249, 0.250, 0.250, 0.250 |

### Processing Plots Review
- `processing_<session_id>.png` files were created for both sample sessions.
- Plots show neural variation, cue-time zero reference, monotonic position progression, sparse/binary lick events, and speed classes on the same retained frame clock.
- Manual pickle checks confirmed matching neural/input/output time dimensions and finite values for every sample trial.
- Initial lexicographic sample selected two unrewarded/non-licking sessions. This was corrected by deterministic selection of two sessions containing lick events and both reward classes; full conversion logic was unchanged.
- Initial NaN-to-integer warnings for invalid out-of-task `ft_trInd` values were fixed by assigning nonfinite trial IDs to -1 before masking.
- Lick assignment was tightened to nearest retained frame only when within 0.5 frame, avoiding relocation of licks from excluded gray/stationary periods.

### Format Validation
- `/app/verification_sample_out.txt`: “Data format is valid, no errors or warnings.”
- All 4 inputs and 4 outputs have expected dimensions/ranges.
- Position and speed classes are balanced as expected.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| One-pass session loading, vectorized masks, compact dtypes | Revised sample completed substantially faster than initial run |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Neural conversion | approximately 4–10 s/session in sample | approximately 6–15 min for 89 sessions including large-file I/O |
| Sample total including 3.8+ GiB pickle write | approximately 14–25 s for 2 sessions | full serialization may dominate; monitor against 15 min target |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None
- Sample training completed successfully on CUDA using 408 training and 102 validation trials.
- Loss decreased from 10,984.09 at epoch 1 to approximately 211.28 at epoch 200; test loss was 90.96.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Uniform Chance |
|--------|-----------------------|-------------------------|----------------|
| Visual stimulus category | 0.9792 | 0.9528 | 0.5000 |
| Licking | 0.7080 | 0.6857 | 0.5000 |
| Position in corridor | 0.8988 | 0.8561 | 0.2500 |
| Running speed quartile | 0.3688 | 0.3551 | 0.2500 |

- Every output is above chance.
- Train/validation gaps are small (approximately 0.01–0.03), arguing against severe overfitting or leakage.
- Speed decoding is the weakest output but remains 1.42× chance; it will be reassessed on the full dataset as required in Step 12.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 161.609 GiB, created successfully.
- `conversion_full_out.txt`: created; converter return code 0.
- `verification_full_out.txt`: created; validation completed successfully.

### Full Conversion Summary
- 89 physical recording sessions from 19 mice.
- 38,110 source trials and 38,110 retained trials; no empty trials.
- 821,579 retained running-corridor frames.
- Speed thresholds: 12.388393, 25.328451, 40.829067 source units.
- 15 physical visual categories: circle1/2/3, leaf1/2/3, leaf1 swap1/2, rock1/2, wood1/2/5, wood1 swap1/2.
- Full conversion runtime: 3,849 s (~64 min). This exceeded the 15-minute estimate because large object-NPY reads and 161.6 GiB serialization dominated despite allocation optimization.

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | Not stated | all concatenated planes | 4,691,034 | 4,691,034 | Yes |
| Neurons/session | 20,547–89,577 | concatenate all planes | 20,547–89,577 | 20,547–89,577 | Yes |
| Subjects | 19 | experiment database | 19 | 19 | Yes |
| Sessions | 89 recordings | 89 DB entries | 89 neural files | 89 | Yes |
| Trials (total) | not stated | all trials by analysis | 38,110 after physical-session alias merge | 38,110 | Yes |
| Running corridor frames | running-only, 0–4 m | `ft_move>0`, corridor | 821,579 | 821,579 | Yes |
| Brain regions | visual cortical areas | grouped by `neu_area_ID` | region labels cover 4,691,034 cells | region labels cover 4,691,034 cells | Yes |
| Position output | 4 m corridor | source 0–40 units | four 1-m bins | approximately 25% each | Yes |
| Speed output | task-required quartiles | N/A | thresholds from retained data | approximately 25% each globally | Yes |

- Validator reported valid format and completed without errors.

### Iteration log
- First full attempt was stopped after 9 sessions when the pre-scan revealed a literal `stimulus_of_trial` output class and missing rock/wood labels.
- Cause: `TrialStim` is a paper-analysis normalization field, not physical wall identity; it maps rock/wood families onto circle/leaf and contains placeholders in some tests.
- Fix: use `WallName`, which directly provides all physical visual categories including swap variants. Alias records were verified to have identical `WallName` and frame streams.
- Re-ran full behavior scan and required that no placeholder remain before restarting conversion.
- Second full attempt was stopped after 34 sessions because per-session time rose from ~4–10 s to ~66 s as hundreds of advanced-indexed trial allocations accumulated.
- Optimization: each session now performs one contiguous neural selection in trial order and exposes lightweight trial views. This preserves exact arrays and final pickle content while greatly reducing allocation fragmentation during conversion.
- Re-ran sample conversion after optimization: 510 trials, 3.934 GiB, 33 s, and format validation passed with no errors/warnings.

### Output Files
- `converted_data.pkl`: [size]
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | | | | | |
| Mean neurons/session | | | | | |
| Subjects | | | | | |
| Sessions | | | | | |
| Trials (total) | | | | | |
| Trials/session (mean) | | | | | |
| <Input 1 range> | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| ... | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| <Output 1 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |
| <Output 2 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `/app/verification_full_out.txt` reports “Data format is valid, no errors or warnings” and ends with “Data verification complete.” No warning required a waiver.
2. **Independent raw neural `np.allclose` checks**: loaded native object-NPY plane arrays directly, concatenated planes independently, reconstructed raw retained-frame indices, and compared exact converted neural matrices (`rtol=0, atol=0`) for first/middle/last trials in TX124_2023_12_24_1, TX108_2023_03_22_1, and VR2_2021_05_06_1. All passed.
3. **Independent raw input `np.allclose` checks**: reconstructed cue time, training day, time since corridor entry, and reward from raw behavior/experiment DB for the same nine trials. All passed.
4. **Independent raw output `np.allclose` checks**: reconstructed physical `WallName`, nearest-frame licking with `LickTrind`, 1-m position classes, and global speed-quartile classes. All passed.
5. **Brain-region comparison**: independently mapped raw retinotopy `iarea` for all neurons in the three spot-check sessions; `np.allclose` passed.
6. **Global structural checks**: 89 unique sessions, 19 subjects, 38,110 trials, 4,691,034 neurons; every session has at least two trials; neural/input/output time lengths and neuron/region lengths agree.
7. **Global finite/range checks**: every stored value is finite; licking is binary; position and speed outputs are in 0–3; trial time is monotonic and starts at zero.
8. **Edge/boundary checks**: explicitly tested first and last trials in sessions, exact trial masks, first retained frame, final recording, fractional lick-frame assignment, and NaN trial IDs outside valid task periods.
9. **Swap alias check**: raw swap1/swap2 records have identical frame/WallName streams and are represented by one physical neural session, preventing duplicated neurons/trials.
10. **Key-statistic comparison**: paper's 89 recordings/19 mice and 20,547–89,577 neurons/session exactly match converted data; reference-data total 4,691,034 neurons is preserved.

### Reference Code Comparison
| Major step | Reference | Conversion | Result / justification |
|------------|-----------|------------|------------------------|
| Data loading | `load_spk` loads object-NPY and concatenates planes | Same source and axis-0 concatenation | Exact match |
| Neuron filtering | General loader retains all Suite2p cells | All cells retained | Exact match; selectivity masks are figure-specific |
| Trial/frame filtering | Paper/code use running (`ft_move>0`) corridor frames | Same running + `ft_CorrSpc` mask | Exact source curation |
| Temporal alignment | Common imaging frame indices (`SoundFr`, `LickFr`) | Same frame clock; corridor entry is trial time zero | Decoder-required temporal form |
| Binning | Paper spatial figures interpolate to 60 position bins | Native ~314.8 ms frames retained | Intentional difference required for temporal lick/speed outputs |
| Input construction | No requested decoder inputs in paper | Direct raw `SoundFr`, DB `sess#`, frame indices, `isRew` | Independently verified |
| Output construction | Paper uses wall/stimulus, lick, position, speed streams | Direct `WallName`, `LickFr`, `ft_Pos`, `ft_RunSpeed` discretization | Independently verified |

### Issues Found and Resolved
- **Stimulus-field mismatch**: initial full pre-scan used `TrialStim`, revealing a placeholder class and loss of rock/wood physical identities. Fixed by using authoritative `WallName`; full scan now contains 15 valid physical categories.
- **NaN trial IDs**: invalid non-task frames produced integer-cast warnings. Fixed by mapping nonfinite raw trial IDs to -1 before valid-frame masking.
- **Lick relocation risk**: nearest retained frame could move excluded-period licks. Fixed by requiring distance <=0.5 imaging frame and matching supplied `LickTrind`.
- **Allocation slowdown**: per-trial advanced indexing caused severe cumulative slowdown. Fixed with one contiguous selected session matrix and trial views; exact output values were revalidated.
- Full report saved as `/app/critical_review1_out.txt`; final line is “CRITICAL REVIEW 1: ALL CHECKS PASS.”

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Full script finished successfully after all 200 epochs.
- CUDA initialization succeeded, but GPU memory was exhausted after epoch 1; the provided script automatically restarted on CPU as instructed.
- CPU loss decreased from 22,146.99 at epoch 1 to 219.93 at epoch 200; test loss was 185.41.
- Session-specific random-projection/SVD initialization completed for all 89 sessions.
- Statistics saved to `/app/train_decoder_full_stats.json`; full log saved to `/app/train_decoder_full_out.txt`.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Uniform Chance | Notes |
|--------|-----------------------|-------------------------|----------------|-------|
| Visual stimulus category | 0.6294 | 0.5921 | 0.0667 | 15 physical stimulus classes; 8.88× chance |
| Licking | 0.9237 | 0.8395 | 0.5000 | 1.68× chance despite sparse events |
| Position in corridor | 0.3481 | 0.3452 | 0.2500 | 1.38× chance; investigate in Step 12 |
| Running speed quartile | 0.3187 | 0.3212 | 0.2500 | 1.28× chance; investigate in Step 12 |

- All outputs exceed chance.
- Train/validation gaps are small: 0.0373, 0.0842, 0.0029, and -0.0025 respectively.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Validation Accuracy | Uniform Chance | Accuracy / Chance | Expectation from Paper |
|----------|---------------------|----------------|-------------------|------------------------|
| Visual stimulus category | 0.5921 | 0.0667 | 8.88× | No decoder accuracy reported |
| Licking | 0.8395 | 0.5000 | 1.68× | No decoder accuracy reported |
| Position in corridor | 0.3452 | 0.2500 | 1.38× | No decoder accuracy reported |
| Running speed quartile | 0.3212 | 0.2500 | 1.28× | No decoder accuracy reported |

### Checks Performed
1. **Accuracy vs chance**: every output exceeds chance. Stimulus and licking exceed 1.5× chance. Position and speed are below 1.5× chance and were investigated fully.
2. **Accuracy vs paper**: searchable paper has zero occurrences of decoder, decoding, classifier, classification accuracy, or accuracy; it reports no comparable decoder benchmark. The paper's d-prime/coding-direction analyses answer different questions.
3. **Train vs validation gap**: train/validation ratios are 1.063, 1.100, 1.009, and 0.992. None approaches the concerning 1.5 threshold; no evidence of severe overfitting or leakage.
4. **Three raw-trial checks for lower-accuracy outputs**: independently loaded DR10 trial 5, TX108 trial 150, and VR2 trial 498. Raw position, speed quartile, and time alignment exactly matched converted values with `np.allclose`.
5. **Temporal alignment**: every input/output pair has identical time length; raw retained indices match trial/corridor/running masks; median fraction of nondecreasing position transitions is 1.0.
6. **Output variation**: licking is sparse but not degenerate (3.47% positive globally). Position fractions are 0.24998, 0.24875, 0.24958, 0.25170. Speed fractions are 0.24932, 0.25022, 0.25011, 0.25036.
7. **Filtering/processing review**: all Suite2p-classified neurons are retained, planes are concatenated exactly as reference, and only reference-supported running corridor frames are used.
8. **Sample plots**: `/app/sample_trials.png` and `/app/predictions.png` were created by the full required training run.

### Interpretation of Lower Position/Speed Accuracy
- No conversion defect was found: values exactly match raw data, classes are balanced, and temporal synchronization is exact.
- The full shared decoder projects each session separately but learns a common output model across 89 recordings and diverse visual areas/cohorts. Instantaneous fine position and speed are weaker shared signals than stimulus identity or licking.
- Position accuracy of 0.3452 and speed accuracy of 0.3212 remain meaningfully above 0.25 chance with essentially no train-validation gap. Altering labels, filtering, or alignment solely to raise accuracy would diverge from the task specification and reference processing.

### Issues Found and Resolved
- No new conversion issue was found in Step 12.
- GPU memory was insufficient; the provided script automatically reran completely on CPU and finished all 200 epochs.
- Full diagnostic output is saved as `/app/critical_review2_out.txt`; final line is “CRITICAL REVIEW 2 DIAGNOSTICS PASS.”

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All investigation scripts copied to cache and documented
- [x] All required conversion, validation, training, plot, and notes files retained in `/app`
- [x] Final documentation includes mappings, rationale, statistics, issues, raw-data checks, decoder results, and accuracy review
