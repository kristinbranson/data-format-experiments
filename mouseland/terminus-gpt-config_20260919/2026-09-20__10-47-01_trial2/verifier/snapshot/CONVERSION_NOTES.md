# Dataset Conversion Notes

## Overview
- **Dataset**: Unsupervised pretraining in biological neural networks (provided paper/code/data)
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment verification:
- Python 3.13.15
- NumPy 2.4.4
- PyTorch 2.6.0+cu124

Directory contents:
- `.manifest` (file)
- `CONVERSION_NOTES.md` (file)
- `Dockerfile` (file)
- `app/` (directory)
- `code/` (directory)
- `data/` (directory)
- `decoder.py` (file)
- `docker-compose.yaml` (file)
- `methods.txt` (file)
- `paper.pdf` (file)
- `train_decoder.py` (file)

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_spk` | `code/utils.py` | LOADING | Loads `<mouse>_<date>_<block>_neural_data.npy`, reads dictionary key `spks`, and concatenates component arrays on the neuron axis to produce neurons × frames. |
| `load_exp_beh` | `code/utils.py` | LOADING | Loads `beh/Beh_<experiment_type>.npy` as a dictionary keyed by session identifier. |
| `get_interpPos_spk` / `spk_pos_interp` | `code/utils.py` | PROCESSING | Linearly interpolates neurons × moving frames against cumulative position into neurons × trials × 60 spatial bins. |
| `get_cat_id` | `code/utils.py` | PROCESSING | Assigns wall/stimulus category IDs relative to rewarded identity: nonreward exemplars 0/1 and reward exemplars 2/3. |
| `get_lick_raster` / `pretrain_exp_lick_raster` | `code/utils.py` | PROCESSING | Groups lick-position events by trial using `LickTrind`; computes first-lick and stimulus-conditioned rasters. |
| `neu_area_ID` / `load_retino` | `code/utils.py` | PROCESSING | Maps concatenated neurons to visual-area labels from retinotopy metadata. |
| `dprime` / `Get_dprime_selective_neuron` | `code/utils.py` | CURATION | Computes stimulus selectivity for figure analyses; thresholds are analysis-specific, not raw recording-quality filters. |
| `Get_dprime_rewPred_neuron` | `code/utils.py` | CURATION | Identifies reward-prediction neurons among stimulus-selective cells for paper analyses. |
| `Get_coding_direction`, `Get_sort_spk`, `get_stimNeu_and_sorted` | `code/utils.py` | PROCESSING | Trial-split coding-direction and sequence analyses on spatially interpolated activity. |

### Notes
- `code/README.md` identifies `data_process_script.ipynb` as the authoritative workflow for processing and saving intermediate results.
- Experiment metadata are loaded from `beh/Imaging_Exp_info.npy`. Session metadata use `mname`, `datexp`, `blk`, and sometimes `stimtype`; behavior dictionary keys are built from those fields.
- Neural activity is already provided as `spks`. The reference loader performs no fluorescence baseline or delta-F/F calculation. Therefore conversion should not recompute dF/F.
- The reference spatial workflow loads `spks`, truncates behavior to the neural frame count, retains frames with `ft_move > 0`, and interpolates activity using `ft_PosCum`. It uses 60 bins over a 6 m corridor (0.1 m/bin). This is appropriate for the paper's position-based analyses, while the requested decoder explicitly requires temporal alignment and time-varying outputs; this distinction must be resolved in Step 5.
- Available stimulus names explicitly listed in the workflow are `circle1`, `circle2`, `leaf1`, `leaf2`, and `leaf3`.
- Behavior fields observed in code include `ntrials`, `Corridor_Length`, `ft_move`, `ft_PosCum`, `WallName`, `UniqWalls`, `isRew`, `stim_id`, `SoundPos`/`SoundDelPos`, `RewPos`, `LickPos`, and `LickTrind`.
- The reference uses visual areas `V1`, `mHV`, `lHV`, and `aHV` in downstream analyses.
- No generic low-quality-neuron exclusion was found in the reference processing workflow. D-prime thresholds (for example 0.3), trial halves/quarters, k-fold splits, and stimulus-selective/reward-prediction masks are analysis-specific safeguards against circularity and should not be mistaken for source-data quality curation.
- No generic trial rejection was found in the processing notebook. Moving-frame selection is used only to ensure monotonic position for spatial interpolation.
- Extracted notebook code and static inspection reports were temporarily written within `/app/code`; they will be moved to `/app/cache` during cleanup.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` is a read-only 411.08 GiB source tree. Exploration artifacts are stored in `/app/cache`.
- `beh/Imaging_Exp_info.npy`: scalar object-array containing experiment metadata dictionaries. Rows identify sessions by `mname`, `datexp`, and `blk`, with optional `stimtype`.
- `beh/Beh_<experiment_type>.npy`: scalar object-arrays containing dictionaries keyed by session identifier. There are 142 experiment-membership entries but only 89 unique physical recordings; repeated memberships point to identical behavior records. Conversion must deduplicate by mouse/date/block and include each neural file once.
- `spk/<mouse>_<date>_<block>_neural_data.npy`: 89 scalar object-arrays. Each dictionary has `spks`, a list of three float32 arrays (three imaging planes/collections) with common frame count. Reference loading concatenates them on neuron axis.
- `retinotopy/<mouse>_<date>_trans.npz`: one transform per physical session date, with `xpos`, `ypos`, transformed `xy_t`, and one numeric `iarea` per concatenated neuron. `areas.npz` contains ten atlas-boundary polygons.
- `process_data/` is empty; paper-derived spatially interpolated products are not supplied.
- Every retinotopy vector length exactly matches the corresponding concatenated neural count.
- Behavior streams have 1–3 more terminal frames than neural data (59 sessions: +1, 23: +2, 7: +3). This matches reference code that slices behavior to neural `nfr`.

### Available Variables
- Trial identity/context: `WallName`, `TrialStim`, `UniqWalls`, `stim_id`, `isRew`, `ntrials`, `trInd`, `Reward_Mode`.
- Frame streams: `ft_trInd`, `ft_Pos`, `ft_PosCum`, `ft_move`, `ft_RunSpeed`, `ft_isMoving`, `ft_CorrSpc`, `ft_GraySpc`, `ft_WallID`, `BefCueFr`, `AftCueFr`.
- Trial event coordinates: `StartFr`, `GrayFr`, `EndFr`, `SoundFr`, `SoundDelayFr`, `RewardFr`; positions include `SoundPos`, cumulative `SoundDelPos`, `RewPos`, and `LickPos` with `LickTrind`.
- Geometry: `Corridor_Length=60`, `Gray_Space_length=20`, `Texture_Length=40` in native decimeter units; `run_pos` is a trial × 60 precomputed spatial representation.
- The representative unsupervised session has no licks/rewards, while rewarded sessions populate those fields.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 4,691,034 |
| Neurons / session | mean 52,708.25; median 54,741; range 20,547–89,577 |
| Subjects | 19 (`DR10`, `DR15`, `LZ13`, `LZ16`, `TX104`, `TX105`, `TX108`, `TX109`, `TX119`, `TX123`, `TX124`, `TX139`, `TX140`, `TX60`, `TX61`, `TX83`, `TX85`, `TX88`, `VR2`) |
| Sessions / subject | range 1–8; 89 total |
| Trials (total) | 38,110 |
| Trials / session | mean 428.20; median 429; range 84–789 |
| Neural frames (total) | 2,025,155; range 14,570–34,228/session |
| Rewarded trials | 4,336 / 38,110 = 11.38% |
| Stimulus/wall identities | circle1–3, leaf1–3, leaf swap variants, rock1–2, wood1/2/5 and wood swap variants |

### Region Mapping
Reference `neu_area_ID` maps `iarea==8` to V1; `0,1,2,9` to mHV; `5,6` to lHV; and `3,4` to aHV. Label 7 is not assigned by that function and must be represented as an unmapped/outside-reference-area class rather than silently dropped. Exact mapped totals will be retained in cached statistics.

### Exploration Checks
- Sequentially loaded all 89 multi-GB neural files in isolated subprocesses; all had only `spks`, three float32 components, and consistent frame counts within session.
- Compared all neural counts to retinotopy lengths: 0 mismatches.
- Compared behavior and neural frame counts: the only discrepancy is 1–3 trailing behavior frames, handled exactly as reference code by truncation to neural length.
- Confirmed all 89 neural filenames have metadata-linked behavior and vice versa after physical-session deduplication.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote / Interpretation |
|-----------|-------|-------------------------------|
| Neurons (total) | Not reported for the full released collection | Paper reports figure-specific mouse/session subsets rather than a global cell count; use exact source-data total from Step 2. |
| Neurons / session | Not globally reported | Figure analyses use simultaneously recorded populations; exact source values are in Step 2. |
| Subjects | Figure-specific subsets (for example task, unsupervised, naive, and grating groups) | The paper captions report different `n` for each comparison; the released imaging metadata contain 19 unique mice. |
| Sessions / subject | Varies by experiment and figure | Paper notes that naive mice can contribute more than one naturalistic-stimulus imaging session and that swap stimuli may be pooled as separate analysis points. |
| Trials (total) | Not globally reported | Exact released-data count is 38,110 unique physical-session trials. |
| Trials / session | Not globally reported | Exact released range is 84–789. |
| Neural data time bin | One imaging frame; exact rate not stated in provided methods excerpt | Neural arrays and all `*Fr` event coordinates share imaging-frame coordinates. Metadata-derived timing must be checked in mapping. |
| Behavior data time bin | Behavior is synchronized/interpolated to imaging frames | Paper states running speed was interpolated to imaging-frame timepoints. |
| Reward rate | Not globally reported | Released data: 11.38% rewarded trials; many unsupervised/naive sessions intentionally have no reward. |
| Visual corridor | 4 m texture region | Paper analyses select the 0–4 m texture region; native position also includes gray space to 6 m. |
| Sound cue | Uniform random position from 0.5–3.5 m | Cue is presented in all imaging trial types, including unrewarded/unsupervised trials. |
| Movement threshold | 6 cm/s | VR motion is triggered when running exceeds 6 cm/s; a figure-specific running analysis additionally selects periods above this threshold. |

### Processing Details
- Calcium imaging was processed with Suite2p: motion correction, ROI detection, cell classification, neuropil correction, and non-negative spike deconvolution with decay timescale 0.75 s. All paper analyses use the deconvolved fluorescence traces supplied as `spks`; no new dF/F should be computed.
- The paper's spatial activity analyses use the 0–4 m texture/corridor region. Reference code constructs 60 bins over the full 6 m native trial (0.1 m/bin) after selecting `ft_move > 0`; the 2 m gray segment serves as baseline in some analyses.
- Sound cue location is randomized independently by trial and is used instead of reward location for reward-prediction analyses because it is present in every corridor.
- Lick response in the paper is often defined as at least one lick inside the corridor before the cue to avoid reward-delivery sensory confounds. The decoder task instead explicitly requests binary time-varying licking, so all lick events must be represented.
- Stimulus selectivity uses d-prime and held-out trials. Coding-direction analyses use independent trials, sequence analyses split odd/even trials, and reward response uses tenfold cross-validation. These prevent circular figure analyses but are not general source-data curation.
- No neural-decoder/classifier accuracy is reported in the paper. There is therefore no directly comparable paper accuracy for the requested decoder outputs.

### Curation Steps

**Neuron curation rules**:
Suite2p performs cell classification before release. The paper uses all supplied deconvolved traces for general processing, then applies analysis-specific selectivity masks (for example d-prime ≥0.3 or top 5%) only for individual figures. These masks should not be imposed on a general decoder dataset.

**Trial curation rules**:
No global bad-trial exclusion is described. Some figure analyses restrict to moving periods, rewarded trials with first lick after 2 m, or held-out subsets; occasional mice are excluded only when a required trial type is absent. Such figure-specific restrictions do not justify dropping valid decoder trials.

### Decoders Trained
| Decoded variable | Accuracy |
|------------------|----------|
| No supervised neural decoder reported | N/A |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Session identity | Metadata are organized by experiment type | 142 memberships map to 89 physical neural files; duplicate records are identical | A session/stimulus may appear in several analyses | Deduplicate strictly by mouse/date/block and include each neural file once. |
| Neural representation | `load_spk` concatenates three `spks` arrays | Three float32 deconvolved arrays/session | Analyses use Suite2p deconvolved fluorescence | Concatenate on neuron axis; do not calculate dF/F. |
| Behavior/neural length | Reference slices behavior to `nfr` | Behavior has 1–3 extra trailing frames in every session | Streams are synchronized to imaging | Truncate all frame behavior to neural frame count. |
| Trial boundary | Reference relies on `ft_trInd` | `StartFr` is fractional between prior/new trial: ceil matches new trial 100%; `EndFr` floor matches current trial 99.992% | Alignment requested at corridor entry | Extract finite `ft_trInd==trial`; equivalently ceil StartFr through floor EndFr. Exclude 10,028 unassigned inter-trial frames. |
| Corridor entry/exit | Spatial code uses cumulative position | First trial frame is at native position ~0; `GrayFr` is at ~40 | Texture corridor is 0–4 m, followed by gray space | Align at first assigned frame/ceil StartFr. Restrict decoder trials to native `ft_Pos<40` (0–4 m). `GrayFr` is corridor exit, not entry. |
| Position units | Reference uses 60 bins over length 60 | `ft_Pos` spans 0–60 and `Texture_Length=40` | Texture region is 0–4 m | Native unit is 0.1 m. Convert meters=`ft_Pos/10`; classes are [0,1), [1,2), [2,3), [3,4] m. |
| Time bin | Not hard-coded in reference | MATLAB-datenum `ft` median difference is 3.6436e-6 days = 314.81 ms (~3.1765 Hz) | Behavior interpolated to imaging-frame times | Retain native frames and record 314.8 ms representative bin; compute time values from actual `ft` timestamps to handle small jitter. |
| Movement filtering | `ft_move>0` ensures monotonic spatial interpolation | Stationary periods create long but valid trials | Some analyses select >6 cm/s | Do not globally remove stationary frames: temporal decoder requires time and speed, and long trials are valid behavior. Restrict only to corridor period. |
| Stimulus identity | Code uses normalized IDs and wall names | `TrialStim` usually normalized but has 3,068 placeholders; `WallName` is always concrete | Paper distinguishes naturalistic category/exemplar identities | Use valid `TrialStim`; for placeholders derive category from session `StimTrial`/wall mapping and retain concrete wall identity in metadata. |
| Brain areas | Four analysis regions are V1, mHV, lHV, aHV | `iarea` labels exactly match neuron counts; label 7 is not mapped by reference helper | Four visual-area groups analyzed | Preserve four mappings plus `unmapped` for label 7; never drop silently. |
| Curation | D-prime/selectivity masks are figure-specific | No general bad-cell/trial flags beyond Suite2p preprocessing | Held-out/cross-validation rules prevent circular figure analyses | Keep all supplied Suite2p cells and all valid corridor trials. Do not impose figure-specific selection. |
| Dataset size | Reference intermediate spatial data use 60 bins/trial | Native full float32 traces would be ~404 GiB; corridor-only ~270 GiB | Requested decoder needs temporal rather than spatial alignment | Preserve native temporal frames in corridor, use float16 only as a storage encoding after quantified error checks; decoder converts to float32. |

### Final Understanding
- The source, code, and paper agree once experiment memberships are distinguished from physical recordings and figure-specific analyses from general preprocessing.
- Every one of the 38,110 trials has valid 0–4 m corridor frames. Median corridor duration is 6.93 s (median 23 frames); long trials reflect stopping/slow running and are scientifically meaningful.
- Cue frame coordinates always belong to the indicated trial. Cue time is behavior-dependent because cue location, not elapsed time, is randomized.
- Float16 suitability was spot-checked across small/median/large neural files: no overflow; sampled MAE 0.0004–0.0028 and maximum quantization error <0.64 compared with trace maxima 981–2,520. This optimization is documented rather than treated as neural preprocessing.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spks` list | `neural` | Concatenate three components on neuron axis; slice native frames belonging to each trial with `ft_Pos<40`; store float16 | `load_spk` | Shape neurons × native imaging timepoints; no dF/F. |
| `SoundFr`, `StartFr`, `ft` | `input[0]` | Signed seconds until sound cue: `(SoundFr-current_frame)*session_median_frame_dt`; equivalently cue time minus current time | frame synchronization in behavior | Continuous, time-varying as explicitly requested. |
| session date (`datexp`) | `input[1]` | Calendar days elapsed from that subject's earliest supplied imaging date; repeat across trial | experiment metadata | Continuous per-trial training-day proxy; stage labels retained in metadata. |
| `ft`, first retained frame | `input[2]` | Actual elapsed seconds from retained corridor-entry frame | synchronized imaging timestamps | Continuous, time-varying. |
| `isRew` | `input[3]` | Boolean to 0/1 and repeat across time | behavior dictionaries | 1 means rewarded corridor, independent of whether reward was delivered. |
| `WallName` | `output[0]` | Global categorical index over concrete visual identities | `get_cat_id` uses same wall identity source | Per-trial category repeated across time. `WallName` is complete; `TrialStim` has placeholders. |
| `LickFr`, `LickTrind` | `output[1]` | Binary vector; each event assigned to nearest retained frame within its declared trial | `get_lick_raster` groups by `LickTrind` | Multiple licks in one imaging bin remain binary 1. Events outside 0–4 m window are excluded. |
| `ft_Pos` | `output[2]` | Native decimeters /10; bins `[0,1)`, `[1,2)`, `[2,3)`, `[3,4]` m -> classes 0–3 | paper/reference spatial geometry | Time-varying, four equal 1 m bins. |
| `ft_RunSpeed` | `output[3]` | Global quantile edges `[0, 8.32701545, 30.1567626]` cm/s via `searchsorted(..., side='right')`, clipped 0–3 | paper interpolates speed to imaging frames | Four empirical quartile classes. Exact 25% counts are impossible due to 20.38% exact-zero ties; deterministic quantile convention documented. |
| `mname` | `subjects`, `subject_idx` | Sorted unique mouse IDs and per-session index | metadata | 19 subjects. |
| retinotopy `iarea` | `brain_region_idx` | 8→V1; 0/1/2/9→mHV; 5/6→lHV; 3/4→aHV; 7→unmapped | `neu_area_ID`, `load_retino` | One index per concatenated neuron; do not discard label 7. |

### Output Vocabulary
- Visual stimulus categories (sorted concrete `WallName` identities): `circle1`, `circle2`, `circle3`, `leaf1`, `leaf1_swap1`, `leaf1_swap2`, `leaf2`, `leaf3`, `rock1`, `rock2`, `wood1`, `wood1_swap1`, `wood1_swap2`, `wood2`, `wood5`.
- Licking: `not licking`, `licking`.
- Position: `0-1 m`, `1-2 m`, `2-3 m`, `3-4 m`.
- Running-speed bins: `<0 cm/s`, `[0,8.327) cm/s`, `[8.327,30.157) cm/s`, `>=30.157 cm/s` under the implemented right-sided threshold convention (metadata stores exact edges).

### Key Decisions
1. **Physical-session deduplication**: Include all 89 unique neural files once. Experiment dictionaries are views/labels, not independent recordings.
2. **Temporal alignment**: Corridor entry is the first finite trial-assigned frame (ceil `StartFr`), not `GrayFr`. Retain native timestamps until `ft_Pos<40` corridor exit.
3. **Native temporal sampling**: Do not spatially interpolate because the decoder explicitly requests time-varying cue time, elapsed time, licking, and speed. Small frame jitter is represented in time inputs; metadata reports the representative 314.81 ms bin.
4. **No movement filtering**: Reference `ft_move>0` is needed only for monotonic spatial interpolation. Retaining stops is necessary to decode temporal behavior and speed.
5. **Neural float16 storage**: Required to make the full temporal pickle practical (~135 GiB estimated corridor-only versus ~270 GiB float32). Spot checks found no overflow and negligible error. The validator/trainer converts tensors to float32. This is storage encoding, not normalization.
6. **Visual identity**: Use concrete `WallName`, the complete trial variable. Do not use placeholder-containing `TrialStim` or collapse wood/rock variants, because the requested output is the presented visual stimulus category.
7. **Training day**: Use elapsed calendar days from each subject's first supplied imaging date. This is continuous, objective, and preserves gaps between before/after-learning recordings; metadata also records experiment-stage memberships.
8. **Lick binning**: Assign each lick to the nearest retained frame in its declared trial, avoiding fractional-boundary off-by-one errors. Licks outside the requested corridor window are intentionally absent.
9. **Speed quartiles**: Compute once globally over all finite valid corridor frames after behavior-to-neural truncation. Negative speeds and stops are valid behavior and remain in the lowest class.
10. **Trial/session inclusion**: All 38,110 trials have at least 11 corridor frames, satisfying the two-trial/session requirement. No source quality flag justifies further filtering.

### Planned Sanity Checks
- [ ] Direct `np.allclose` comparison of converted neural trial slices against raw concatenated `spks` at selected session/trial/neuron/time indices (with float16 tolerance).
- [ ] Direct `np.allclose` comparison of all four converted inputs against independently computed raw behavior values for selected trials.
- [ ] Direct `np.allclose` comparison of output category, lick, position, and speed classes against independently computed raw values.
- [ ] Verify 89 sessions, 19 subjects, 38,110 trials, 4,691,034 neurons, and exact per-session retinotopy/neuron lengths.
- [ ] Verify each neural/input/output trial has identical time length, finite values, four input rows, and four output rows.
- [ ] Verify first time-since-start is zero, position classes are 0–3, speed classes 0–3, and reward/lick values are binary.
- [ ] Verify cue-time input changes by the negative of elapsed time and crosses zero near `SoundFr`.
- [ ] Verify visual-category and reward distributions against direct source counts.
- [ ] Plot neural activity, position, speed, lick events, and cue timing for up to two sessions.

---

## Step 6: Script Development
**Status**: COMPLETE

- Created `/app/convert_data.py`; syntax compilation passes.
- CLI supports required output path, mutually exclusive `--full`/`--sample`, and `--show-processing`.
- Conversion loads canonical behavior metadata once, but loads and releases each multi-GB neural session sequentially.
- Neural components are concatenated exactly as reference `load_spk`; behavior is truncated to neural frame length; trial corridor indices are vectorized before slicing.
- Every session/trial undergoes shape and value-domain assertions before serialization.
- Processing plots show neural traces, cue/time alignment, position/speed discretization, and licking for up to two sessions.
- Sample mode deterministically selects one unsupervised session (`DR10_2022_07_12_1`) and one supervised/reward-task session (`TX60_2021_06_07_1`) so all outputs can be tested, rather than two no-lick unsupervised sessions.

Code inefficiencies identified:
- The source object format requires materializing each full neural session before selecting corridor frames.
- Trial-list pickle representation requires copying selected neural columns; full output is expected to be large.
- Per-lick assignment loops only over 74,483 sparse events and is negligible compared with neural I/O.

Code speedups added:
- Session-at-a-time loading and immediate garbage collection bound peak memory.
- Vectorized frame masks, output discretization, and direct NumPy slicing.
- Float16 neural serialization halves output size with quantified negligible error and no overflow.
- Highest pickle protocol minimizes serialization overhead.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Output size | 3.401 GiB |
| Sessions | 2 (`DR10_2022_07_12_1`, `TX60_2021_06_07_1`) |
| Neurons (total across sessions) | 106,009 |
| Neurons / session | 58,224; 47,785 |
| Subjects | 2 |
| Trials (total) | 988 |
| Trials / session | 503; 485 |
| Retained timepoints | 34,037 total; trial range 12–362 |
| Time to cue | [-70.65, 93.62] s |
| Training day | [0, 58] elapsed days |
| Time since start | [0, 113.66] s |
| Reward availability | both 0 and 1 |
| Visual output | circle1 41.3%, circle2 4.7%, leaf1 49.1%, leaf2 4.9% (timepoint-weighted) |
| Licking | 7.3% licking time bins |
| Position bins | 25.5%, 23.1%, 24.8%, 26.6% |
| Speed classes | 5.2%, 40.5%, 39.4%, 14.9% in this nonrepresentative two-session sample |

### Format Validation
- `verification_sample_out.txt`: **Data format is valid, no errors or warnings.**
- Neural dtype is float16; input dtype float32; outputs are integer categorical arrays.
- Neural, input, and output time lengths match for every manually checked trial; converter asserts this for every trial.
- All inputs are finite and categorical outputs lie in declared vocabularies.

### Processing Plots Review
- Created `processing_DR10_2022_07_12_1.png` and `processing_TX60_2021_06_07_1.png` (valid PNGs, 273 KiB and 254 KiB).
- Plots include native neural traces, time-to-cue crossing relative to elapsed time, monotonic position-bin progression, speed classes, and lick bins.
- No shape/alignment anomalies were reported; first elapsed-time sample is zero for all inspected trials.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| float16 serialization | Approximately 50% output I/O/storage versus float32 |
| Session-at-a-time loading | Bounded peak memory; avoids retaining raw sessions |
| Vectorized masks/classes | Behavioral processing negligible relative to neural I/O |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Sample conversion | 16.55 s mean (33.10 s / 2) | Naive 24.5 min for 89 sessions |
| Full conversion revised estimate | Source-size weighted; likely 15–25 min | Must monitor first full sessions and optimize/abort if >1.5× estimate |

The dominant cost is copying/serializing a scientifically required ~135 GiB temporal dataset. Before full conversion, neural casting/slicing order and safe parallelism will be reviewed; parallel disk reads may not help on shared storage.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings from format verifier: None
- Training-time sklearn warning: `y_pred contains classes not in y_true`. This is unavoidable for the two-session sample because `output_values` correctly declares all 15 full-dataset visual categories while only four categories occur in the sample. It is not a malformed-label warning.

### Training Progress
- Completed all 200 epochs on CUDA.
- Loss decreased from 8168.748 at epoch 1 to 141.787 at epoch 200.
- Test loss: 66.765.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Uniform Chance |
|--------|-----------------------|-------------------------|----------------|
| visual stimulus category | 0.9273 | 0.8519 | 0.0667 (15 declared classes) |
| licking | 0.8688 | 0.8764 | 0.5000 |
| position in corridor | 0.6541 | 0.5832 | 0.2500 |
| running speed quartile | 0.6384 | 0.6130 | 0.2500 |

All outputs exceed chance, loss decreases strongly, and train/validation gaps are small. The especially strong visual and licking accuracy supports correct trial identity and temporal alignment.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 137.855 GiB (reported as 138G by `ls`)
- `conversion_full_out.txt`: created; all 89 session lines plus timing
- `verification_full_out.txt`: created

### Runtime
- Full conversion plus serialization: 668.10 s (11.14 min), below the 15-minute target.
- Pickle serialization: 135.41 s.
- Three bounded NumPy worker threads reduced conversion wall time without extra process serialization.

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | Not globally reported | Concatenate all `spks` components | 4,691,034 | 4,691,034 | Yes |
| Mean neurons/session | Not globally reported | — | 52,708.25 | 52,708.25 | Yes |
| Subjects | Figure-specific subsets | metadata `mname` | 19 | 19 | Yes |
| Sessions | Figure-specific subsets | physical mouse/date/block | 89 | 89 | Yes |
| Trials (total) | Not globally reported | behavior `ntrials` | 38,110 | 38,110 | Yes |
| Trials/session (mean) | Not globally reported | — | 428.20 | 428.20 | Yes |
| Trial timepoints | Native imaging frames in 0–4 m | temporal task-specific conversion | range 11–5,607 | range 11–5,607; mean 40.40 | Yes |
| Visual categories | Naturalistic and swap stimuli | wall identity | 15 concrete identities | all 15; fractions in verifier | Yes |
| Licking | Sparse, absent in non-task sessions | event positions/trials | 74,483 source events; corridor subset expected sparse | 3.5% of retained bins | Sensible |
| Position bins | Four requested 1 m bins | 0–4 m texture region | expected traversal coverage | 28.5%, 23.3%, 23.6%, 24.6% | Yes |
| Speed classes | Requested percentile bins | global edges | edges 0, 8.3270, 30.1568 cm/s | 9.8%, 40.2%, 25.0%, 25.0% | Yes with ties |
| Brain regions | V1/mHV/lHV/aHV | `neu_area_ID` | 1,833,035 / 1,108,860 / 495,318 / 668,180; label-7 585,641 | exact same | Yes |

### Verification Result
`Data format is valid, no errors or warnings.` All required global and per-session summaries were produced. The speed-bin imbalance in the first two classes is not an error: 20.38% of valid frames have speed exactly zero, making exact equal-size value intervals impossible without arbitrarily assigning identical values to different classes. The global quantile edges are reproducible; upper classes are exactly 25% each.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output Log Verification
- Read `/app/verification_full_out.txt` in full and searched for error/warning/invalid/failed/traceback.
- Result: `Data format is valid, no errors or warnings.` No warning requires waiver.
- Global counts and ranges are valid; all categorical values are in declared vocabularies.

### Check 2: Independent Raw-Source `np.allclose` Checks
`/app/cache/critical_review1.py` loads original files directly and does **not** import conversion code. It loaded the full converted pickle and checked first/middle/last trials in sessions 0, 44, and 88:

| Session | Trials checked | Neural | Inputs | Outputs | Result |
|---------|----------------|--------|--------|---------|--------|
| `DR10_2022_07_12_1` | 0, 251, 502 | exact float16 roundtrip at 3 neurons × 3 times | all four | all four | PASS |
| `TX119_2024_01_06_1` | 0, 236, 472 | exact float16 roundtrip at 3 neurons × 3 times | all four | all four | PASS |
| `VR2_2021_05_06_1` | 0, 249, 498 | exact float16 roundtrip at 3 neurons × 3 times | all four | all four | PASS |

- Neural criteria: `np.allclose(..., rtol=0, atol=0)` after independent raw float32→float16 cast.
- Input criteria: independently reconstructed cue time, elapsed time, day, and reward availability; `np.allclose` with float timing tolerance.
- Output criteria: independently reconstructed concrete stimulus index, nearest-frame lick vector, 1 m position bins, and global speed classes; `np.allclose` exact for categorical values.
- Result log ends with `ALL_CRITICAL_REVIEW_CHECKS_PASS`.

### Check 3: Reference-Code Comparison
| Major step | Conversion | Reference | Comparison / justified difference |
|------------|------------|-----------|-----------------------------------|
| Data loading | Loads object dictionary `spks`; concatenates list on axis 0 | `utils.load_spk` does exactly this | Exact match. |
| Neuron/trial filtering | Keeps all Suite2p cells and valid trials | No global quality exclusion; selectivity masks are figure-specific | Exact general-curation match; no d-prime filtering. |
| Temporal alignment | Truncates behavior to neural `nfr`; finite trial frames; entry at ceil StartFr | Notebook truncates behavior to `nfr`; behavior synchronized to imaging | Exact synchronization; temporal entry alignment required by decoder task. |
| Binning | Keeps native imaging frames; categorical 1 m position and global speed quantiles | Paper spatial analyses interpolate to 0.1 m bins | Deliberate decoder-required difference: outputs are time varying, so spatial interpolation would destroy elapsed-time/lick timing. |
| Input construction | Raw behavior/event fields synchronized to imaging frames | Reference uses the same fields (`SoundFr`, position, reward state) in analyses | Match in source variables; new combinations are decoder specification. |
| Output construction | `WallName`, lick events, position, speed | Reference uses wall identity, `LickTrind`/`LickPos`, position, and interpolated speed | Match in source variables; categorical transforms follow decoder task. |

### Check 4: Key Statistics
- Exact matches: 19 mice, 89 physical sessions, 38,110 trials, 4,691,034 neurons, per-session neuron range 20,547–89,577, and all region counts.
- All 15 raw concrete `WallName` values occur in converted visual output.
- Rewarded source fraction is 11.38%; converted reward availability contains both classes and is repeated over valid time bins as intended.
- Position output spans all four bins in every session and has plausible traversal-weighted fractions.
- Speed global quantile edges exactly match direct raw computation. Fractions are 9.8/40.2/25.0/25.0% because identical zero values cannot be split between equal-count value intervals.

### Check 5: Edge Cases
- Behavior has 1–3 trailing frames beyond neural data; conversion truncates to neural frame count exactly as reference.
- 10,028 inter-trial frames have NaN trial IDs and are excluded.
- Fractional boundaries: ceil StartFr belongs to the new trial 100%; floor EndFr belongs to current trial 99.992%; direct finite trial IDs are used to avoid off-by-one errors.
- Every trial has 11 or more retained corridor frames; no empty/one-frame trials.
- Very long trials (maximum 5,607 retained frames) are valid stopped/slow-running behavior and remain included.
- Lick events use declared `LickTrind` and nearest frame, avoiding boundary assignment errors.
- Retinotopy label 7 is retained as `unmapped`; all 4,691,034 neurons have a region index.
- Corrected documentation interval notation for right-sided speed thresholds; code/data were already correct.

### Issues Found and Resolved
- **Documentation only**: initial prose mislabeled the exact inclusivity of speed-bin edges. Corrected to `<0`, `[0,8.327)`, `[8.327,30.157)`, `>=30.157` cm/s. No data change needed.
- No conversion mismatch or validation error was found; therefore the iteration protocol did not require reconversion.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Required command completed fully with `--plot-samples` and ended with `train_decoder.py finished successfully.`
- Initial CUDA setup completed all 89 session projections but GPU memory was insufficient for the network; the script automatically retried on CPU as permitted.
- CPU retry completed all 89 session SVD initializations and all 200 epochs.
- Loss decreasing: **Yes**. Epoch 1 = 18,297.713; epoch 100 = 407.274; epoch 200 = 185.612 (small late stochastic fluctuations do not alter the ~99% overall reduction).
- Test loss: 136.269.
- Split: 30,457 training trials; 7,653 validation trials.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Uniform Chance | Notes |
|--------|-----------------------|-------------------------|----------------|-------|
| visual stimulus category | 0.5770 | 0.5087 | 0.0667 | 7.63× chance validation |
| licking | 0.8854 | 0.8635 | 0.5000 | 1.73× chance validation |
| position in corridor | 0.3155 | 0.3196 | 0.2500 | 1.28× chance; investigated in Step 12 |
| running speed quartile | 0.4100 | 0.4159 | 0.2500 | 1.66× chance validation |

Training and validation accuracies are closely matched; validation is slightly higher for position and speed. This argues against overfitting or leakage. The paper reports no directly comparable supervised decoder accuracies.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Validation Accuracy | Chance | Accuracy / Chance | Expectation from Paper |
|----------|---------------------|--------|-------------------|------------------------|
| visual stimulus category | 0.5087 | 0.0667 | 7.63× | No supervised decoder reported |
| licking | 0.8635 | 0.5000 | 1.73× | No supervised decoder reported |
| position in corridor | 0.3196 | 0.2500 | 1.28× | No supervised decoder reported |
| running speed quartile | 0.4159 | 0.2500 | 1.66× | No supervised decoder reported |

### Check 1: Accuracy vs Chance
- Every validation accuracy is above chance.
- Visual identity is far above chance, strongly supporting correct trial identity and session mapping.
- Licking and speed exceed 1.5× chance.
- Position is above chance but below the 1.5× investigation threshold, so all prescribed debugging checks were reviewed rather than accepting it without analysis.

### Position Debugging Investigation
1. **Three specific raw trials**: Step 10 independently reconstructed position outputs for nine trials (first/middle/last in three sessions) from raw `ft_Pos`; every value matched with `np.allclose`. This exceeds the requested three-trial check.
2. **Temporal alignment**: `processing_DR10_2022_07_12_1.png` and `processing_TX60_2021_06_07_1.png` jointly plot temporal neural activity and position classes. Independent boundary analysis showed first retained frame at corridor entry, cue frames in the correct trial 100%, and `GrayFr` at the 4 m exit. No shift or off-by-one error was found.
3. **Variation**: Full position fractions are 28.5%, 23.3%, 23.6%, and 24.6%; no dominant 99% class and every session contains all four classes.
4. **Neural filtering**: Supplied Suite2p cell classification/deconvolution is retained exactly. No reference quality filter was omitted; d-prime masks are figure-specific and would create inappropriate target-dependent selection.
5. **Reference processing**: Raw loading and frame synchronization match reference code. Spatial interpolation was deliberately not used because it would replace temporal sampling and invalidate the requested elapsed-time, lick, and speed streams.
6. **Internal comparison**: The same representation achieved 0.5832 position validation accuracy in the two-session sample. The lower full result occurs when one shared decoder handles 89 heterogeneous session-specific populations and long stopped trials; it is not reproduced as a raw/converted mismatch.
7. **Train/validation behavior**: Position training accuracy is 0.3155 and validation is 0.3196. The absence of a train advantage rules out overfitting and makes a label-alignment bug unlikely; a systematic bug would also have failed exact raw comparisons.

Conclusion: the lower full position accuracy reflects the available signal and heterogeneous full-dataset model, not a correctable conversion defect. Altering labels, spatially warping time, or filtering stationary periods merely to increase accuracy would violate the decoder specification and reference data semantics.

### Check 2: Accuracy Comparison to Paper
- Exhaustive paper-text search found no supervised neural decoding/classification accuracy. The paper reports d-prime, coding-direction projections, sequence correlations, kernel fits, and behavioral performance instead.
- Therefore no paper accuracy can be numerically compared without inventing a benchmark. This is documented rather than attributed vaguely to architecture.

### Check 3: Train vs Validation Gap
| Variable | Train / Validation Ratio | Result |
|----------|--------------------------|--------|
| visual stimulus | 1.134 | below 1.5; no concerning gap |
| licking | 1.025 | no concerning gap |
| position | 0.987 | validation slightly higher |
| speed | 0.986 | validation slightly higher |

No output has a >1.5× train/validation ratio, and there is no evidence of leakage or strong overfitting.

### Issues Found and Resolved
- GPU OOM occurred after CUDA initialization; the provided script automatically restarted on CPU and completed successfully. This is a compute-capacity issue, not a data issue.
- Position accuracy triggered detailed investigation, but no conversion mismatch was found. No reconversion was scientifically justified, so the iteration protocol did not require rerunning affected steps.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created with loading instructions, format, mappings, statistics, and validation results.
- [x] `cache/` folder created with `README_CACHE.md` and investigation artifacts.
- [x] Temporary notebook extraction/static-inspection files moved from `/app/code` to `/app/cache/code_inspection`; original reference source remains unchanged.
- [x] Required conversion, validation, training, plot, script, data, and documentation files audited.
- [x] `CONVERSION_NOTES.md` finalized with all decisions, checks, warnings, and decoder results.

### Final Deliverables
- Full data: `/app/converted_data.pkl` (137.855 GiB)
- Sample data: `/app/sample_data.pkl` (3.401 GiB)
- Reproducible converter: `/app/convert_data.py`
- Full validation: no errors or warnings
- Full decoder: completed 200 epochs after automatic CPU fallback; every output above chance
- Decoder plot: `/app/predictions.png` (the trainer's `--plot-samples` output; this version does not emit a separate `samples.png`)
