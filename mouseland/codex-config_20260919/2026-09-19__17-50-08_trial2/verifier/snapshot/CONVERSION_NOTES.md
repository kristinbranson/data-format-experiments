# Dataset Conversion Notes

## Overview
- **Dataset**: Unsupervised pretraining in biological neural networks (provided paper/code/data)
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment verification:
- Python 3.13.15 runs successfully.
- NumPy 2.4.4 and PyTorch 2.6.0+cu124 import successfully.
- Checkpoint passed: `/app/CONVERSION_NOTES.md` exists.

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
| `np.load(..., allow_pickle=True).item()` | `data_process_script.ipynb` | LOADING | Load experiment registry and per-experiment behavior dictionaries. |
| `load_exp_beh` | `utils.py:326` | LOADING | Load `beh/Beh_<exp_type>.npy` as a session-keyed dictionary. |
| `load_spk` | `utils.py:338` | LOADING | Load `<mouse>_<date>_<block>_neural_data.npy` and concatenate plane-wise `spks` over neurons. |
| `load_retino` / `neu_area_ID` | `utils.py:312,330` | LOADING | Load transformed neuron coordinates/area labels and group Allen-style IDs into V1, medial HV, lateral HV, anterior HV. |
| `interp_value` / `spk_pos_interp` | `utils.py:100,105` | PROCESSING | Linear interpolation of frame activity against cumulative VR position. |
| `get_interpPos_spk` | `utils.py:120` | PROCESSING | Convert neurons x frames to neurons x trials x 60 spatial bins (0.1 m each across a 6 m trial). |
| `get_lick_raster` | `utils.py:149` | PROCESSING | Group lick positions by trial/stimulus, with first-lick extraction and cue/time sorting. |
| `lickCount` / `lick_response` | `utils.py:217,238` | PROCESSING | Compute per-trial binary licking in reference-defined temporal or spatial windows. |
| `Get_dprime_selective_neuron` | `utils.py:418` | CURATION | Compute stimulus selectivity using only moving frames in textured corridor (`ft_move > 0 & ft_CorrSpc`). |
| `Get_coding_direction` | `utils.py:503` | PROCESSING | Spatially interpolate activity, define selectivity from odd trials, normalize relative to gray-space activity and stimulus SD, and project held-out trials. |
| `Get_sort_spk` | `utils.py:599` | PROCESSING | Train/test split for stimulus-selective neurons and spatial sequence sorting. |
| `get_kfold_reward_response` | `utils.py:814` | PROCESSING | Ten-fold reward-response analysis aligned to cue/first lick using spatially interpolated activity. |

### Notes
- README identifies `data_process_script.ipynb` as the authoritative intermediate-data pipeline and `Figures.ipynb`/`fig*.py` as plotting code.
- The calcium recording frame rate stated in the processing notebook is 3.17 Hz. Behavior contains raw timestamps plus values already aligned to each neural frame (`ft_*`, `StartFr`, `EndFr`, `SoundFr`, `LickFr`, etc.).
- Neural files already contain plane-wise `spks`; the reference loader concatenates these without computing delta-F/F. Thus no new dF/F computation is indicated. The `spks` are processed/deconvolved calcium-event estimates, not electrophysiology, so spike-quality filtering is inapplicable.
- Reference spatial processing keeps frames for which `ft_move > 0`, interpolates activity versus cumulative position, and uses 60 bins per 6 m corridor. Step 4 resolved an initial concern about elapsed-time gaps: the converter will follow this explicit running-frame curation, while calculating temporal inputs from original timestamps so gaps are represented rather than silently compressed.
- Reference analyses truncate behavior frame arrays to neural `nfr`, use `ft_trInd`/frame indices for alignment, and classify texture trials using `WallName`, `UniqWalls`, and `stim_id` (0 circle1, 1 circle2, 2 leaf1, 3 leaf2, 4 leaf3, 5/6 leaf1 swaps).
- Area grouping: V1=`iarea==8`; medial HV=`0,1,2,9`; lateral HV=`5,6`; anterior HV=`3,4`. Code explicitly excludes `iarea==-1` and `7` only in density-map analyses; general decoder mapping should preserve recorded neurons and represent additional labels explicitly unless the source registry provides a principled exclusion.
- The original notebook reports its complete preprocessing can take ~8 hours on the full 1.3-TB release; the provided subset must therefore be inventoried before choosing a conversion strategy.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `beh/Imaging_Exp_info.npy`: dictionary with 23 experiment-stage entries and session records (`mname`, date, block, condition, stimulus IDs, and stage/day fields).
- `beh/Beh_<experiment>.npy`: 23 session-keyed pickled dictionaries. Across files there are 142 registry entries but many intentional aliases/reuses (e.g. the same recording serves as test1 and train2-before-learning; swap1/swap2 give alternate labels for the same frames). Collapsing by physical `<mouse>_<date>_<block>` yields exactly 89 behavior sessions, matching all neural files, with 38,110 unique trials.
- `spk/<mouse>_<date>_<block>_neural_data.npy`: 89 pickled dictionaries. Each contains only `spks`, a list of three float32 neuron x frame arrays (one per imaging plane). Reference loading concatenates planes over neurons. Example `TX124_2023_12_24_1`: three `(6849, 18491)` arrays, hence 20,547 neurons.
- `retinotopy/<mouse>_<date>_trans.npz`: 89 matching files with affine transform, raw/transformed coordinates, and integer `iarea` per concatenated neuron. `areas.npz` contains region outlines.
- `process_data/`: empty in the provided data; derived spatial arrays must not be assumed available.
- No README/documentation file is present inside `data`; variable definitions are supplied by the reference processing notebook and were independently confirmed from native arrays.
- Behavior dictionaries contain 59 fields. Important native streams are trial timestamps/labels; licks with time, position, trial and neural-frame stamps; VR/run signals; and frame-aligned `ft_*` arrays. Example behavior types: timestamps/positions float64, masks bool, stimulus strings Unicode, and per-trial index arrays int/float. Neural activity is float32.
- Physical stimulus names present: circle1/2/3, leaf1/2/3 and leaf swaps, plus analogous rock/wood stimuli for later cohorts. Reward modes are `Passive` and `Active after cue`; unsupervised sessions encode no reward using the same active-mode string but `isRew` is false.
- Alignment edge case confirmed: neural and behavior frame lengths can differ by one frame (example: 18,491 neural vs 18,492 behavior frames). Reference code always truncates behavior frame arrays to neural `nfr`; the converter must use their common length.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 4,691,034 recorded ROIs; 4,105,393 after excluding reference outside/unknown IDs -1 and 7 |
| Neurons / session | 20,547–89,577; median 54,741; mean 52,708.25 (unfiltered) |
| Subjects | 19: DR10, DR15, LZ13, LZ16, TX104, TX105, TX108, TX109, TX119, TX123, TX124, TX139, TX140, TX60, TX61, TX83, TX85, TX88, VR2 |
| Sessions / subject | 1–8; DR10 6, DR15 5, LZ13 4, LZ16 4, TX104 2, TX105 5, TX108 7, TX109 6, TX119 8, TX123 8, TX124 3, TX139 2, TX140 1, TX60 5, TX61 5, TX83 3, TX85 2, TX88 6, VR2 7 |
| Trials (total) | 38,110 unique trials across physical neural sessions (63,177 if intentional aliases are incorrectly double-counted) |
| Trials / session | 84–789; median 429; mean 428.20 |

Additional area-ID counts before curation: -1 unknown/outside 397,310; 0 275,623; 1 147,071; 2 139,295; 3 568,741; 4 99,439; 5 342,053; 6 153,265; 7 outside visual cortex 188,331; 8 V1 1,833,035; 9 546,871.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Not reported as a sum | Paper reports per-recording range, not unique/summed cells. | 
| Neurons / session | 20,547–89,577 | “activity traces from 20,547 to 89,577 neurons in each recording” |
| Subjects | 19 imaging mice | “We performed 89 recordings in 19 mice…” |
| Sessions / subject | 89 total recordings; multiple sessions/mouse | “We performed 89 recordings…” and notes that mice can have multiple sessions. |
| Trials (total) | Not reported | No aggregate trial count in paper/methods. |
| Trials / session | Not reported | No per-session range in paper/methods. |
| Neural data time bin | 3.17 Hz (~315.46 ms/frame) | Processing notebook: “Calcium signal recording frame rate: fs = 3.17Hz”. |
| Behavior data time bin | Native high-rate timestamps; interpolated to imaging frames where needed | Methods: running speed “was interpolated to the timepoints of the imaging frames using scipy.interpolate.interp1d.” |
| Reward rate | Task-dependent; rewards only in designated rewarded corridor, absent in unsupervised sessions | Sound cue “was followed by the availability of water in rewarded trials only”; unsupervised cohort “did not receive water rewards.” |
| Texture corridor geometry | 4 m texture + 2 m gray | “corridors were each 4 m long, with 2 m of grey space between corridors.” |
| Sound cue location | Uniform random 0.5–3.5 m | “randomly chosen per trial from a uniform distribution between positions 0.5 m and 3.5 m.” |
| VR motion | 60 cm/s while running exceeds 6 cm/s | Methods specifies threshold 6 cm/s and constant corridor motion 60 cm/s. |


### Processing Details
- Suite2p performed motion correction, ROI detection, cell classification, neuropil correction, and non-negative spike deconvolution (decay time 0.75 s). All paper analyses used the provided deconvolved traces.
- Neural selectivity used original, non-interpolated deconvolved traces only within the textured 0–4 m corridor and only while the animal was running.
- Spatial analyses linearly interpolated neural activity to 0.1-m positions (60 points over the 6-m texture+gray cycle). Running speed was interpolated to imaging-frame timestamps for time-aligned analyses.
- Trials are naturally aligned to entry into each textured corridor (`Trial_start_time`, `StartFr`, or the transition in `ft_trInd`/`ft_CorrSpc`). The requested four 1-m position classes imply restricting decoder timepoints to the 0–4 m textured corridor, not the following 2-m gray space.
- Sound is presented in every imaging trial. In task mice it opens reward availability only in the rewarded visual corridor; it remains present without reward in unsupervised sessions.
- Licking is event-like and natively supplied with timestamps and imaging-frame indices, so it can be represented as a per-frame binary series without temporal interpolation.

### Curation Steps

**Neuron curation rules**:
Use Suite2p-classified cells and supplied deconvolved traces. The paper does not impose a firing-rate/activity threshold. Area analyses consider recorded V1 and grouped higher visual areas; source code excludes `iarea==-1` and `iarea==7` as outside mapped visual cortex in density-map totals.

**Trial curation rules**:
Paper neural analyses keep textured-corridor timepoints when mice are running and omit stopping/reward-consumption periods. The converter will follow the exact operational reference mask (`ft_CorrSpc & (ft_move > 0)`) and use original frame timestamps for temporal inputs. This also permits four non-degenerate running-speed quartiles. Only trials lacking at least one common curated neural/behavior frame will be excluded.

### Decoders Trained
| Decoded variable | Accuracy |
| None matching the supplied neural-decoder outputs | The paper reports selectivity, coding-direction similarity, correlations, and inferential statistics, but no classifier accuracy for visual category, licking, position bins, or speed bins. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Dataset scope | Registry contains 23 overlapping experiment labels | Exactly 89 physical neural/retinotopy/behavior session bases | 89 recordings in 19 mice | Collapse aliases by `<mouse>_<date>_<block>` and include each physical recording once; use `WallName` directly so swap aliases are unnecessary. |
| Neuron counts | `load_spk` concatenates plane-wise `spks` | 20,547–89,577 ROIs/session; 4,691,034 total | Same per-session range | Exact match; load all planes in reference order. |
| Brain regions | `neu_area_ID` groups IDs into V1/mHV/lHV/aHV; density code excludes -1 and 7 | 397,310 ID -1 and 188,331 ID 7 ROIs | Analyses concern V1 and HVAs | Retain only mapped V1/mHV/lHV/aHV neurons (4,105,393); exclude -1/7 as outside/unknown visual cortex. |
| Calcium processing | Load supplied `spks` directly | float32 deconvolved streams only | All analyses use Suite2p deconvolved fluorescence | No dF/F or further neural normalization for the generic decoder. |
| Frame curation | Major analyses use `ft_CorrSpc & (ft_move>0)` | 821,579 unique textured running frames versus 1,373,170 all textured frames | Only running timepoints considered | Follow exact reference mask. Preserve actual elapsed time in temporal inputs using `ft` timestamps, so missing stationary intervals remain explicit. |
| Trial boundaries | Reference selects 0–4 m texture for selectivity; spatial products also include 2 m gray | `ft_Pos` spans [0,40) in `ft_CorrSpc` | Corridor is 4 m texture plus inter-corridor 2 m gray | Decoder trial is 0–4 m visual corridor, aligned to its entry; exclude gray. Four position labels are [0,1), [1,2), [2,3), [3,4) m. |
| Neural/behavior length | Code truncates frame behavior to neural `nfr` | At least one verified one-frame mismatch | Not discussed | Always truncate every frame-aligned behavior stream and neural data to common minimum length before masking. |
| Reward variable | Analyses distinguish task reward corridor using `isRew` | Some unsupervised files say active mode but all `isRew=False`; 4,336/38,110 raw trials are rewarded corridors | No rewards in unsupervised cohort | Use per-trial `isRew`, never the session mode string. |
| Stimulus labels | `stim_id` maps exemplars but has NaNs and alternate swap mappings | Native labels include circle/leaf/rock/wood exemplars | Paper describes circle/leaf/rock/brick categories | Derive physical category from alphabetic prefix of `WallName`; preserve native `wood` label (paper's brick texture) and pool exemplar/swap variants into four categories. |
| Running speed bins | No decoder binning in reference | All-frame Q1 is tied at zero; running-mask quartiles are non-degenerate | Analyses curate running frames | Compute global quartiles over all included running frames: preliminary behavior-only thresholds 12.4224, 25.3526, 40.8546 (native speed units); final script recomputes after common-length curation. |
| Training day | No single helper; registry uses `days` for later learned sessions and `sess#` elsewhere | Alias-specific `sess#` conflicts occur for shared physical recordings | Paper describes stages/~2 weeks, not a per-recording day table | Use explicit `days` when present; otherwise the minimum `sess#` among aliases (avoids treating swap label 2 or naive reuse as another physical day). Record both source fields and rule in metadata. |
| Decoder accuracy | Not implemented | N/A | No matching classifier accuracy reported | Compare only to chance and training behavior; paper values are not directly comparable. |

Consistency checks passed: all 89 neural bases have exactly one retinotopy file and at least one behavior alias; collapsing aliases gives 38,110 trials with no trial lacking textured frames before neural-length truncation; raw dataset counts and neuron range exactly reproduce paper statements.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spks` (3 planes) | `neural` | Concatenate planes over cells, retain mapped visual-cortex cells, then select common-length frames satisfying trial ID, `ft_CorrSpc`, and `ft_move>0`; float32 | `load_spk`, `Get_dprime_selective_neuron` | One neurons x selected-time array per trial. No normalization or dF/F. |
| `SoundTime`, `ft` | `input[0]` time to sound cue | `(SoundTime[trial] - ft[selected_frame]) * 86400`, seconds; signed positive before cue | notebook alignment fields | Time-varying float32. Uses timestamps rather than fractional `SoundFr`. |
| Registry `days` / `sess#` | `input[1]` day of training | Explicit `days`, else minimum alias `sess#`; repeat over trial frames | experiment registry | Continuous numeric session-level covariate; heterogeneous source convention disclosed in metadata. |
| `Trial_start_time`, `ft` | `input[2]` time since trial start | `(ft[selected_frame] - Trial_start_time[trial]) * 86400`, seconds | notebook alignment fields | Time-varying float32 and retains real gaps removed by running mask. |
| `isRew` | `input[3]` reward availability | Boolean 0/1 repeated over trial frames | reward analyses | Means “rewarded corridor,” not whether water was actually delivered. |
| `WallName` | `output[0]` visual stimulus category | Alphabetic leading category pooled across exemplar/swap suffixes; classes circle, leaf, rock, wood | `get_cat_id` concept; direct native labels | Repeated over time because mixed per-trial/time-varying fields share one output matrix. |
| `LickFr` | `output[1]` licking | Reference-compatible `LickFr.astype(int)` event frames, binary at selected frames | `spk_2_firstLick`, `spk_2_cue` | Multiple licks in a frame remain 1. |
| `ft_Pos` | `output[2]` corridor position | floor(position decimeters / 10), clipped 0–3 | 60-position reference interpolation | Four equal 1-m bins spanning textured 0–4 m corridor. |
| `ft_RunSpeed` | `output[3]` running speed quartile | Global `np.quantile` thresholds over all common-length curated frames; `np.searchsorted(..., side='right')` | paper running-speed interpolation/curation | Four approximately equal-frequency classes with one physical set of thresholds across sessions. |
| retinotopy `iarea` | `brain_region_idx` | 8→V1; 0/1/2/9→mHV; 5/6→lHV; 3/4→aHV; remove -1/7 | `neu_area_ID`, `Get_density_map` | Region array is filtered identically to neural rows. |

### Key Decisions
1. **Physical-session scope**: Include all 89 paper recordings exactly once. Behavior aliases are alternate analyses of identical frames, not additional sessions.
2. **Running-frame curation**: Apply the paper/code mask `ft_move>0` within `ft_CorrSpc`. This best satisfies the critical reference-matching constraint and avoids stationary reward-consumption confounds.
3. **Temporal bin interpretation**: Native imaging bins are nominally 315.46 ms. Selected samples remain native frames but stationary intervals are omitted; actual timestamp-derived inputs preserve elapsed timing. Metadata will explicitly distinguish native frame interval from curated sampling gaps.
4. **Neurons**: Keep all Suite2p-classified cells in mapped visual cortex; do not select neurons based on the target labels, activity, d-prime, or firing rate, avoiding leakage and matching the absence of a general quality threshold in the reference.
5. **Speed quartiles**: Use global thresholds so class meanings are comparable across recordings. Preliminary reference-mask thresholds are [12.4224, 25.3526, 40.8546], to be recomputed after per-session common-length truncation.
6. **Categorical values**: `output_values` = `[['circle','leaf','rock','wood'], ['not_licking','licking'], ['0-1 m','1-2 m','2-3 m','3-4 m'], ['Q1','Q2','Q3','Q4']]` and arrays use matching zero-based integer codes.
7. **Missing/invalid data**: Truncate all frame streams to common neural/behavior length; omit only trials with no remaining curated frame. Fail loudly on non-finite neural/input/output values, unknown stimulus prefixes, region-length mismatch, or sessions with fewer than two included trials.
8. **Dtypes/storage**: Neural and inputs float32; outputs int8; indices integer arrays. Do not compress/quantize neural values, preserving exact supplied float32 data and decoder accuracy.
9. **Sample selection**: Deterministically choose two post-learning task sessions that contain reward and licking, while favoring smaller files, so sample validation exercises every requested variable.
10. **Validator compatibility**: Use 2-D `(4,T)` arrays for both input and output on every trial. This avoids mixed dimensionality and ensures every timepoint aligns exactly with neural columns.

### Planned Sanity Checks
- [ ] Inventory: 89 physical sessions, 19 subjects, raw neuron range 20,547–89,577, all neural/behavior/retinotopy bases matched.
- [ ] Neural raw-to-converted spot checks with `np.allclose` for selected cells/frames in at least three sessions.
- [ ] Input raw-to-converted checks with `np.allclose` for timestamp formulas, day value, and reward flag.
- [ ] Output raw-to-converted checks with `np.allclose` for category code, lick-event mask, position bin, and speed bin.
- [ ] Confirm all trial matrices have identical time lengths across neural/input/output and ≥2 trials/session.
- [ ] Confirm no NaN/Inf and all output labels are in declared ranges.
- [ ] Confirm speed-class fractions are ~0.25 globally and position/category/reward/lick distributions are plausible.
- [ ] Confirm sample plots show monotonic position progression (allowing real stationary-gap omission), cue-time zero crossing, elapsed-time increase, and exact lick alignment.
- [ ] Compare converted counts/ranges to paper, registry, and raw files; explain any trial omission.
- [ ] Re-run provided format verification and decoder after every material fix.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with `--full` (default), `--sample`, and `--show-processing`. The script:
- validates 89-way neural/behavior/registry/retinotopy correspondence;
- collapses behavior aliases without relying on ambiguous `stim_id` views;
- loads supplied plane-wise spks into a preallocated mapped-neuron matrix;
- applies common-length truncation and the documented texture/running mask;
- creates aligned float32 inputs and int8 categorical outputs;
- computes global speed quartiles after collecting all small speed vectors, avoiding a second neural-data pass;
- performs strict finite/range/shape validation before pickle writing;
- records detailed per-session provenance and counts in metadata; and
- creates 12-panel processing diagnostics for up to two sessions.

Syntax compilation and command-line option checks pass under Python 3.13.

Code inefficiencies identified:
Naively concatenating all raw planes before region filtering would allocate an additional complete session matrix. Computing exact global speed thresholds in a separate pass would also reread roughly 400 GB of neural files unnecessarily.

Code speedups added:
Preallocate only retained neurons and fill plane-by-plane; process each neural file once; retain only small speed arrays until quartile thresholds are known; release full-session arrays immediately after trial extraction; use highest-protocol pickle and float32/int8 storage. Parallel session loading was intentionally avoided because sequential reads are fast and parallel copies would substantially raise peak memory without clear throughput benefit.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 113,120 session-ROIs after mapping |
| Neurons / session | 51,339; 61,781 |
| Subjects | 2 (TX60, TX108) |
| Sessions / subject | 1 each |
| Trials (total) | 705/705 retained |
| Trials / session | 303; 402 |
| Native selected T / trial | overall summary mean 20.25; min 11; max 36 |
| Time to cue range | [-59.5, 202.1] s |
| Training day/stage range | [1, 6] |
| Time since entry range | [0.0, 204.4] s |
| Reward availability range | [0, 1] |
| Visual category distribution | [circle .177, leaf .180, rock .222, wood .422] |
| Licking distribution | [not licking .892, licking .108] |
| Position distribution | [.246, .252, .254, .248] |
| Speed quartile distribution | [.250, .250, .250, .250] |

### Processing Plots Review
`processing_TX60_2021_05_04_1.png` and `processing_TX108_2023_04_01_1.png` show correct area filtering, non-negative deconvolved activity, raw 0–6 m cycles, retained samples only in the rising 0–4 m texture segment, monotonically increasing four-bin position labels, sensible global speed boundaries, and exact binary lick events. Cue-time crosses zero as elapsed time rises. TX60 shows a real stationary gap early in the first trial; timestamp inputs correctly preserve it. No temporal offset or discretization anomaly was found.

The provided validator reported: valid format, no errors, no warnings. Manual reload confirmed required keys, `(neuron,T)/(4,T)/(4,T)` alignment, brain-region vector lengths, declared thresholds, and zero excluded trials.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Single neural-file pass with deferred speed labeling | Avoids rereading 404 GiB of spk files (estimated >10 min) |
| Preallocated mapped-neuron matrix | Avoids one full raw concatenation/copy per session |
| Float32 neural and int8 output | Preserves source exactly while minimizing output and pickle I/O |

| Step | Time / Session | Estimated Total Time |
| Sample neural load+conversion | 6.99 s/session mean | ~10.4 min scaled by 404-GiB/full versus 8.1-GiB/sample raw bytes |
| Pickle writing | 2.35 s for 3.154 GiB | ~2 min for an estimated 155–170 GiB full pickle |
| Catalog/validation overhead | ~3.5 s total sample | <1 min full |
| Overall | 20.49 s for 2 sessions including two plots | approximately 13–14 min full, below the 15-min optimization threshold |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None
- Training completed on CUDA for all 200 epochs. Loss decreased monotonically in the reported checkpoints from 7,569.29 (epoch 1) to 140.08 (epoch 200); held-out test loss was 47.55.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Visual stimulus category | 0.9985 | 0.9782 |
| Licking | 0.8122 | 0.7317 |
| Corridor position bin | 0.9239 | 0.8938 |
| Running speed quartile | 0.3834 | 0.3709 |

Every held-out result exceeds uniform chance (0.25 for four-class outputs; 0.50 for licking). Visual category, licking, and position are far above chance. Speed reaches 1.48x chance in the sample; this is plausible for instantaneous physical running speed from calcium activity and will be re-evaluated on all sessions in Steps 11–12.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 141.126 GiB (shown as 142G by `ls -lh`)
- `verification_full_out.txt`: created; valid with no errors or warnings
- `conversion_full_out.txt`: created; 742.95 s total (12.38 min), below estimate

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | Sum not reported | Exclude outside/unknown for visual-region analyses | 4,691,034 raw; 4,105,393 mapped | 4,105,393 | Yes |
| Mean neurons/session | Range 20,547–89,577 raw | Concatenate planes; mapped areas grouped | raw mean 52,708.25 | mapped mean 46,128.01; raw range reproduced | Yes |
| Subjects | 19 | registry | 19 | 19 | Yes |
| Sessions | 89 recordings | registry aliases overlap | 89 physical | 89 | Yes |
| Trials (total) | not reported | `ntrials` per behavior session | 38,110 physical-session trials | 38,110 | Yes; none lost |
| Trials/session (mean) | not reported | behavior arrays | 428.20 (84–789) | 428.20 (84–789) | Yes |
| Time to cue range | cue at 0.5–3.5 m; no time range | timestamps | signed, variable with stops | [-1763.3, 723.5] s | Formula matches; long gaps reviewed in Step 10 |
| Day/stage input | stages/~2 weeks | `days`/`sess#` | [0,15] | [0,15] | Yes |
| Time since trial start | variable | timestamps | nonnegative | [0,1765.2] s | Formula matches; long gaps reviewed in Step 10 |
| Reward availability | only rewarded task corridors | `isRew` | binary; 4,336/38,110 source trials | [0,1] | Yes |
| Visual category distribution | four physical image categories | `WallName` | circle/leaf/rock/wood | [.311,.470,.085,.135] by timepoint | Yes |
| Licking distribution | anticipatory licking in task cohort | `LickFr.astype(int)` | sparse events | [not .963, lick .037] by curated timepoint | Yes |
| Position distribution | 0–4 m texture corridor | `ft_Pos`, `ft_CorrSpc` | four 1-m bins | [.250,.249,.250,.252] | Yes |
| Speed distribution | running timepoints only | `ft_move>0` | global thresholds 12.4224/25.3526/40.8546 | [.250,.250,.250,.250] | Yes |

Spot checks during conversion included the raw minimum-neuron (20,547) and maximum-neuron (89,577) recordings, one-frame neural/behavior truncation cases, reward/no-reward cohorts, and swap-session aliases. Every session passed finite/range/shape assertions and retained every trial.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Searched complete conversion and verification logs for error/warning/invalid/NaN/Inf indicators. The only match is the validator statement “valid, no errors or warnings.” No action required.
2. **Independent raw-to-converted `np.allclose` tests**: `/app/sanity_checks.py` deliberately does not import conversion code. It loaded original behavior, retinotopy, and plane-wise spks for DR10_2022_07_12_1 (unsupervised), TX108_2023_03_25_1 (rewarded, many neurons), TX124_2023_12_24_1 (minimum-neuron session), and TX88_2022_07_19_1 (extreme stationary-gap case). For a middle trial and first/middle/last retained frame, exact (`rtol=atol=0`) neural vectors and region arrays matched. Timestamp inputs matched at 1e-6 tolerance. Category, lick, position, and speed outputs matched exactly. Trial counts and every aligned shape matched. All tests passed.
3. **Data loading comparison**: Converter uses the reference `np.load(..., allow_pickle=True).item()['spks']` and concatenated plane order. It preallocates the filtered result rather than calling `np.concatenate`, which is memory-equivalent and was proven by exact neural comparisons.
4. **Neuron/trial filtering comparison**: Suite2p deconvolved cells are used with no activity/d-prime target selection. IDs -1/7 are excluded exactly where reference density analyses define outside visual cortex; grouped area logic matches `neu_area_ID`. Trial frames use the exact reference `ft_CorrSpc & (ft_move>0)` logic. No trial lacks a retained frame, so all 38,110 remain.
5. **Temporal alignment comparison**: Both paths truncate frame behavior to neural `nfr`; all 89 behavior arrays are 1–2 frames longer than neural and were safely truncated. Trial stamps come from `ft_trInd`; trial zero/last and common-length edges were checked. Task-specific time inputs use original `ft`, `Trial_start_time`, and `SoundTime`, preserving raw stationary gaps.
6. **Binning comparison**: The reference uses 60 spatial bins across 6 m for spatial figures. The decoder requirement instead uses native 3.17-Hz frames and four requested 1-m output bins over the reference 0–4 m textured segment. Global speed quartiles contain 205,394–205,395 frames each. All position labels are 0–3.
7. **Input construction comparison**: Inputs are new decoder-task variables, derived only from documented reference fields. Independent formulas for cue time, day/stage, elapsed time, and `isRew` passed `np.allclose`.
8. **Output construction comparison**: Direct `WallName` avoids ambiguous swap `stim_id`; `LickFr.astype(int)` is exactly the cast used in reference cue/first-lick helpers; position uses `ft_Pos`; speed uses `ft_RunSpeed`. Independent reconstruction passed.
9. **Key statistics**: Reproduced paper counts 89 recordings, 19 mice, and 20,547–89,577 raw ROIs/session. Raw total is 4,691,034; the mapped output has 4,105,393. Trials exactly match raw 38,110. Output distributions and ranges are documented in Step 9.
10. **Edge cases**: Verified all 89 one/two-frame stream mismatches; 49 trials with elapsed duration >300 s; monotonic elapsed time in every trial; position range; minimum/maximum cells; reward/no-reward; aliases; and first/last selected frames. The longest raw trial resumes after a 1,737-s stationary interval with the same trial stamp and monotonic position. This is an authentic recorded pause, not misalignment; following the paper, stationary samples are omitted but the trial is not discarded.

### Issues Found and Resolved
- **Behavior aliases could double-count 25,067 trials**: resolved by physical session base; raw/paper session count now matches 89.
- **Every behavior frame stream exceeds its neural stream by 1–2 frames**: resolved with the reference code's common-length truncation. No selected trial was lost.
- **Long elapsed-time values looked suspicious in the summary**: direct raw inspection showed real continuous frame timestamps separated by stationary gaps while `ft_trInd` and position remain coherent. Kept and documented; elapsed time is monotonic and exact.
- **Stationary-frame curation versus temporal alignment**: resolved by keeping the explicit reference mask while deriving time inputs from original timestamps, so no false uniform-time compression is introduced.
- **No unresolved issues remain.**

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. Reported loss fell from 18,863.35 at epoch 1 to 181.63 at epoch 200 (99.0% reduction). Held-out test loss was 129.03.
- Device: CUDA throughout; no OOM fallback.
- Full split: 30,457 training trials and 7,653 validation trials.
- Execution completed normally and produced `sample_trials.png` and `predictions.png`.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Visual stimulus category | 0.7097 | 0.7007 | 2.80x uniform chance |
| Licking | 0.9395 | 0.8539 | 1.71x uniform chance despite 3.7% positive frames |
| Corridor position bin | 0.4053 | 0.4002 | 1.60x uniform chance |
| Running speed quartile | 0.3566 | 0.3573 | 1.43x uniform chance; investigated in Step 12 |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| Visual stimulus category | 0.7007 validation (chance .25; 2.80x) | No matching decoder accuracy reported; paper's coding-direction analyses establish strong category information. |
| Licking | 0.8539 validation (chance .50; 1.71x) | No matching decoder accuracy; paper analyzes anticipatory licking and neural/lick relationships. |
| Corridor position bin | 0.4002 validation (chance .25; 1.60x) | No matching decoder accuracy; paper reports spatial tuning/sequences, so above-chance position information is expected. |
| Running speed quartile | 0.3573 validation (chance .25; 1.43x) | No matching decoder accuracy; paper reports similar running distributions, not decoding. |

**Accuracy versus chance**: every validation accuracy is above chance. Visual category, licking, and position exceed the 1.5x-chance review threshold. Speed is 1.43x chance and therefore received the full low-accuracy audit below.

**Speed audit**:
1. Independent raw loading checked speed output for four specific trials/sessions, exceeding the requested three-trial spot check. `np.searchsorted` of raw `ft_RunSpeed` with stored thresholds matched every converted speed label exactly using `np.allclose`.
2. Processing figures overlay raw speed, thresholds, and converted classes on single trials; transitions are synchronized with the neural columns and position/lick outputs.
3. Variation is ideal rather than degenerate: global counts are [205395,205394,205395,205395], each 25.0%.
4. Neural and frame filtering follows supplied deconvolved Suite2p traces and exact reference `ft_CorrSpc & ft_move>0` curation.
5. The source variable is correctly `ft_RunSpeed`, which the reference describes as running speed interpolated to imaging frames. Replacing it with VR motion (`ft_move`) or including speed as an input would change the task/leak the answer and is not justified.
6. The sample decoder independently obtained a similar validation score (0.3709). Instantaneous physical ball speed is noisy relative to slow GCaMP6s/deconvolved activity at 3.17 Hz, so 0.357 is a plausible biological ceiling for this architecture rather than evidence of conversion error.

**Accuracy comparison to paper**: exhaustive PDF search found no decoding/classification accuracies for any of the four requested variables. The paper reports d-prime, coding-direction similarity, correlations, and statistical P values; these are not commensurate with balanced classifier accuracy. Thus no paper accuracy is lower/higher in a comparable sense.

**Train-validation gaps**: train/validation ratios are visual 1.013, licking 1.100, position 1.013, and speed 0.998. None approaches the 1.5 overfitting threshold. The near-identical results support correct held-out trial alignment and no material leakage.

**Plot review**: `sample_trials.png` shows coherent monotonic position steps, sparse licks, variable speed, fixed per-trial category, and sensible neural variability. `predictions.png` shows held-out predictions aligned to the same columns; weaker instantaneous speed/position traces are consistent with aggregate accuracies. Global scaling makes time inputs visually flat in random plots because a few authentic long-pause trials define the range, but raw processing plots and `np.allclose` checks verify their values.

### Issues Found and Resolved
- **Speed below 1.5x chance**: fully investigated as above. No formatting, alignment, variation, filtering, or source-field bug was found; changing the mapping would be scientifically unjustified.
- **No paper classifier benchmark**: documented as unavailable rather than substituting incomparable statistics.
- **No unresolved accuracy or leakage issue remains.**

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with dataset description, loading instructions, format, preprocessing, statistics, decoder scores, and reproduction commands.
- [x] cache/ folder created with `README_CACHE.md`, independent sanity-check script/output, and generated bytecode.
- [x] All files organized. Required deliverables and logs exist; investigation-only code is outside the project root; no incomplete workflow status or template placeholder remains.

Final deliverable audit:
- Full pickle: 151,532,439,865 bytes; sample pickle: 3,386,583,546 bytes.
- Full conversion and format verification completed without error/warning.
- Full 200-epoch decoder training completed successfully and prediction/sample plots exist.
- All 13 workflow steps are complete.
