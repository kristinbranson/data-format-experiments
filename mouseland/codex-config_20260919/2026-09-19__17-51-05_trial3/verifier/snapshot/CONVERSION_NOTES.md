# Dataset Conversion Notes

## Overview
- **Dataset**: Unsupervised pretraining in biological neural networks (provided paper/code/data)
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

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

Environment verification: Python 3.13.15; NumPy 2.4.4; PyTorch 2.6.0+cu124. Checkpoint confirmed with `ls -la /app/CONVERSION_NOTES.md`.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_exp_beh` | `code/utils.py` | LOADING | Loads `Beh_<exp_type>.npy` as a session-keyed behavior dictionary. |
| `load_spk` | `code/utils.py` | LOADING | Loads `<mouse>_<date>_<block>_neural_data.npy`, extracts the `spks` list, and concatenates imaging planes along the neuron axis. |
| `load_retino` / `neu_area_ID` | `code/utils.py` | LOADING | Loads transformed cell locations/area labels and maps raw `iarea` codes to V1, medial higher visual (mHV), lateral higher visual (lHV), and anterior higher visual (aHV). |
| `interp_value` / `spk_pos_interp` / `get_interpPos_spk` | `code/utils.py` | PROCESSING | Linearly interpolates moving-frame activity against cumulative VR position into 60 bins/trial (0.1 m/bin over a 6 m corridor). |
| `Get_dprime_selective_neuron` | `code/utils.py` | CURATION | Computes stimulus d-prime using raw imaging frames restricted to corridor frames with positive VR movement. |
| `Get_coding_direction` | `code/utils.py` | PROCESSING | Uses odd trials for selectivity, normalizes spatial activity relative to gray-space activity, and projects held-out trials. |
| `Get_sort_spk` | `code/utils.py` | PROCESSING | Z-scores position-interpolated activity, selects area-specific stimulus-responsive cells, and separates odd/even trials for cross-validation. |
| `lickCount` / `lick_response` | `code/utils.py` | PROCESSING | Converts lick timestamps to trial responses in paper-defined temporal or spatial windows. |
| `spk_2_cue` / `spk_2_firstLick` | `code/utils.py` | PROCESSING | Extracts 30 imaging frames around cue/first lick and bins licks on that imaging-frame grid. |
| `get_kfold_reward_response` | `code/utils.py` | CURATION | Selects reward-prediction neurons using 10-fold trial cross-validation and area/selectivity criteria. |

### Notes
- `code/README.md` identifies `data_process_script.ipynb` as the processing workflow and `Figures.ipynb` as plotting only.
- The notebook documents the calcium imaging sampling rate as 3.17 Hz and gives the behavioral field definitions. It loads experiment metadata from `beh/Imaging_Exp_info.npy`, behavior from `beh/Beh_<experiment>.npy`, neural activity through `load_spk`, and retinotopy through `load_retino`.
- Neural files contain a list named `spks`; reference loading concatenates that list across planes. These are the released processed calcium-event/deconvolved activity values used directly by all analyses. No reference function computes ΔF/F from fluorescence, so no new ΔF/F calculation is warranted.
- Reference temporal alignment uses behavior variables already sampled/aligned to neural frames (`ft*`, `StartFr`, `EndFr`, `SoundFr`, `LickFr`, etc.). The spatial-analysis interpolation first retains `ft_move > 0`, then maps activity against `ft_PosCum` into 60 equally spaced 0.1 m bins across the full 6 m trial (4 m textured corridor plus 2 m gray space).
- Paper analyses sometimes restrict to moving textured-corridor frames and sometimes select stimulus-responsive neurons. Those restrictions support particular figure analyses; their applicability to a general neural decoder will be decided after reconciling the provided decoder specification with the raw variables and methods.
- Area curation in density-map analyses excludes raw codes `-1` and `7` as outside visual cortex. The named-region mapping itself covers raw codes V1=8, mHV={0,1,2,9}, lHV={5,6}, aHV={3,4}; unmapped codes will require explicit handling after dataset inspection.
- No generic cell-quality threshold or electrophysiology quality filtering appears: this is two-photon calcium imaging, not electrophysiology. Reference selectivity thresholds (`d' >= 0.3` or percentiles) are analysis-specific rather than acquisition-quality filters.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Total data footprint: approximately 412 GB apparent size.
- `data/beh/` (28 files, 6.6 GB): `Imaging_Exp_info.npy` is a dictionary with 23 experiment-group keys and 142 group/session descriptors. Each `Beh_<group>.npy` is a pickled dictionary keyed by `<mouse>_<YYYY_MM_DD>_<block>` (with an added swap stimulus suffix where needed), containing trial timestamps/categories, lick events, continuous VR/running streams, imaging-frame-aligned behavior streams, and precomputed event frame indices. The three `Unsupervised_pretraining_behavior/` files contain behavior-only cohorts and have no matching neural files, so they are documented but are not candidate decoder sessions.
- `data/spk/` (89 files, 405 GB): one `<mouse>_<date>_<block>_neural_data.npy` per physical imaging session. Inspection of a representative file showed a dictionary whose only key is `spks`; its value is a list of three float32 arrays (one per imaging plane), each shaped `(n_cells_plane, n_imaging_frames)`. Reference loading concatenates these on the cell axis.
- `data/retinotopy/` (89 session transform files plus `areas.npz`, 170 MB): each `<mouse>_<date>_trans.npz` provides `iarea` (one raw area code per cell) and `xy_t` transformed cell coordinates. Counts/order match concatenated `spks` for the inspected session. `areas.npz` provides atlas outlines for plotting.
- The 142 experiment-group entries refer to 89 unique physical neural recordings because the same session can serve multiple analyses (for example test1 and train2-before-learning) and swap analyses can have multiple behavior keys. Duplicate representations had identical summary dimensions. The unique physical session ID is mouse + date + block; it will prevent duplicate neural recordings in conversion.
- Available behavior includes trial-level `WallName`, `isRew`, cue/reward/start/end times and imaging-frame indices; lick event time/position/trial/frame; imaging-frame-level position, running speed, movement/corridor masks and trial IDs; and 60-bin spatial running arrays. All audited trial fields had length `ntrials`, all frame fields had length `ft`, and every `run_pos` had shape `(ntrials, 60)`.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 4,691,034 neuron-session units across the 89 unique recordings |
| Neurons / session | min 20,547; mean 52,708.25; median 54,741; max 89,577 |
| Subjects | 19: DR10, DR15, LZ13, LZ16, TX104, TX105, TX108, TX109, TX119, TX123, TX124, TX139, TX140, TX60, TX61, TX83, TX85, TX88, VR2 |
| Sessions / subject | min 1, mean 4.68, max 8; exact counts: DR10 6, DR15 5, LZ13 4, LZ16 4, TX104 2, TX105 5, TX108 7, TX109 6, TX119 8, TX123 8, TX124 3, TX139 2, TX140 1, TX60 5, TX61 5, TX83 3, TX85 2, TX88 6, VR2 7 |
| Sessions (unique physical recordings) | 89 (versus 142 appearances across experiment-group indices) |
| Trials (total) | 38,110 unique-session trials |
| Trials / session | min 84; mean 428.20; median 429; max 789 |
| Imaging frames / session | min 14,571; mean 22,755.97; max 34,230 |
| Licks | 74,483 events |
| Rewarded trials | 4,336 / 38,110 = 11.38% in the unique-session inventory |
| Corridor geometry | 60 source position units = 6 m; texture 40 units = 4 m; gray 20 units = 2 m in all 89 sessions |
| Native stimulus labels | circle1/2/3, leaf1/2/3, leaf1_swap1/2, rock1/2, wood1/2/5, wood1_swap1/2 |

Raw retinotopy-code counts: `-1`: 397,310; `0`: 275,623; `1`: 147,071; `2`: 139,295; `3`: 568,741; `4`: 99,439; `5`: 342,053; `6`: 153,265; `7`: 188,331; `8`: 1,833,035; `9`: 546,871. Codes `-1` and `7` are outside visual cortex per the reference; named mapped codes account for the remaining cells.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Not explicitly totaled in text; data inventory gives 4,691,034 neuron-session units | Paper reports a range per recording, not a sum. | 
| Neurons / session | 20,547–89,577 | “We ran Suite2p on this data to obtain the activity traces from 20,547 to 89,577 neurons in each recording.” |
| Subjects | 19 imaging mice (13 male, 6 female); 23 additional behavior-only mice | “We performed 89 recordings in 19 mice…” and “We also used 23 C57 female mice for behaviour-only experiments.” |
| Sessions / subject | Not tabulated; 89 recordings / 19 mice overall | “We performed 89 recordings in 19 mice…” |
| Trials (total) | Not reported in text; data inventory gives 38,110 unique imaging-session trials | No paper total. |
| Trials / session | Not reported in text; data range 84–789 | No paper range. |
| Neural data time bin | Imaging frames at 3.17 Hz = approximately 315.46 ms | Processing notebook: “Calcium signal recording frame rate: fs = 3.17Hz”. |
| Behavior data time bin | Native behavior has higher-rate timestamps; paper interpolates behavior to imaging frames where needed | Methods: running speed was “interpolated to the timepoints of the imaging frames using … `scipy.interpolate.interp1d`.” |
| Reward rate | Not reported for imaging recordings; empirical unique-session rate is 4,336/38,110 = 11.38% | Paper describes trial contingencies but gives no aggregate rate. |
| Recordings | 89 | “We performed 89 recordings in 19 mice…” |
| Corridor geometry | 4 m textured corridor + 2 m gray interval | “The virtual reality corridors were each 4 m long, with 2 m of grey space between corridors.” |
| VR movement | Triggered when mouse speed exceeded 6 cm/s; VR then advanced at fixed 60 cm/s | Methods visual-stimuli section. |
| Imaging cue location | Uniformly randomized from 0.5–3.5 m on every imaging trial type | Methods behavioral-training section. |
| Calcium preprocessing | Suite2p cell classification, neuropil correction, and non-negative spike deconvolution with 0.75 s decay | Methods calcium-imaging section. |


### Processing Details
- All published analyses used the released Suite2p deconvolved fluorescence traces; no new ΔF/F calculation is described.
- Neural selectivity used original, non-interpolated imaging frames, restricted to the textured 0–4 m corridor and frames in which the mouse was running. The paper pooled positions/trials within stimulus when calculating d-prime.
- Spatial tuning/coding-direction analyses interpolated activity to every position, matching the released code's 60 bins across 6 m (0.1 m/bin). Coding-direction responses were baseline-centered to the gray portion and scaled by the average standard deviation in the two reference corridors.
- For running analyses, the paper used periods faster than 6 cm/s for at least 66 ms and interpolated running speed to imaging-frame timestamps with `scipy.interpolate.interp1d`; position analyses further interpolated speed at 0.1 m spacing from 0–6 m.
- Cue location was randomized trial-by-trial. For task mice it marked reward-zone onset in the rewarded corridor; it was also presented in unrewarded and unsupervised trials. Active reward followed a post-cue lick; some mice instead received passive reward 1 or 1.5 s after cue.
- Trials are naturally aligned to corridor entry via `Trial_start_time` / `StartFr`, matching the requested alignment event.

### Curation Steps

**Neuron curation rules**:
Suite2p performed ROI detection, cell classification, neuropil correction, and deconvolution upstream. The released `spks` contain these processed cell traces. No additional generic quality threshold is specified. Paper-specific analyses selected neurons by d-prime (`|d'| >= 0.3` or top/bottom 5%) and/or region; these are hypothesis-specific selections, not acquisition-quality curation suitable for an all-output decoder.

**Trial curation rules**:
No whole-session/trial exclusion rule is reported for the general dataset. Individual figure analyses held out odd/even trials, used tenfold cross-validation, or excluded trials/mice for specific first-lick/leaf2 questions. The general neural analysis excluded non-running timepoints and gray-space timepoints when computing selectivity, but the downstream temporal decoder requires a contiguous corridor-entry-aligned stream and predicts running speed/position/licking; applicability is resolved in Steps 4–5.

### Decoders Trained
| Decoded variable | Accuracy |
| No categorical neural decoder reported | N/A. The paper reports d-prime, sequence correlation, coding-direction projections/similarity indices, and statistical tests—not classification accuracy. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Recording count | Experiment index has 142 entries across 23 analysis groups | 89 physical neural files and 89 retinotopy files; 33 identities recur across groups | 89 recordings in 19 mice | Convert each mouse/date/block physical recording exactly once (89 sessions); do not duplicate recordings merely because they support multiple paper analyses. |
| Swap annotations | Separate `stimtype=swap1/swap2` descriptors support pooled statistics | For unsupervised/naive swap sessions the two behavior keys have identical timestamps, trial labels, and frame alignment; only `stim_id` marks one swap or the other | Two swap types occurred in the same session for unsupervised/naive mice and were treated as separate statistical data points | Use one physical session and its unambiguous `WallName` per trial; do not duplicate neural data. |
| Neural preprocessing | `load_spk` directly concatenates released `spks` | Neural files contain float32 processed `spks` arrays | All analyses use Suite2p deconvolved fluorescence with 0.75 s decay | Use released `spks` directly; do not calculate ΔF/F or deconvolve again. |
| Cell quality | No extra generic threshold in loading; area analyses use named-area masks | Retinotopy exactly matches cell count/order; raw codes −1 and 7 comprise 585,641 cells | Suite2p did cell classification; reported recording range includes all released cells | Suite2p classification is the quality curation. For target regional neural matrices, exclude raw codes −1 and 7 because reference code explicitly labels them outside visual cortex and no valid target brain-region name exists; retain the 4,105,393 cells in V1/mHV/lHV/aHV. Report both raw and retained counts. |
| Trial interval | Frame-aligned behavior offers `StartFr`, `GrayFr`, `EndFr`, `ft_CorrSpc`, and `ft_trInd` | Every trial has valid aligned arrays; full start-to-gray windows contain extreme stationary intervals (up to ~29 min) | Main neural analyses use textured 0–4 m corridor and only running timepoints | Use textured-corridor frames and the reference running criterion (`ft_CorrSpc & (ft_move > 0)`) for neural/behavior samples. This also makes four 1-m position bins meaningful and prevents water-consumption pauses from dominating. Retain actual frame timestamps as time inputs so skipped stationary intervals remain explicit. |
| Alignment boundaries | Reference code truncates behavior to neural length and uses frame-aligned masks/indices | `StartFr`/`GrayFr` are fractional imaging-frame coordinates; first/last frame positions may differ slightly from exact boundaries | Trial start is corridor entry; analysis region is 0–4 m | Select frames by `ft_trInd == trial`, `ft_CorrSpc`, and movement rather than rounding fractional endpoints. This avoids off-by-one gray frames and is exactly the reference masking logic. |
| Reward label | Reference uses `isRew` to identify the rewarded stimulus, then compares by wall name | Active sessions have 1,037 delivered rewards but 1,147 trials in the rewarded corridor; passive sessions match 3,299/3,299 | Reward is available after cue in the rewarded corridor; active delivery requires licking | Derive the single rewarded `WallName` from any `isRew=True` trial, then mark every trial with that wall as availability=1. Sessions with no delivered rewards have no rewarded corridor and remain 0. Result: 4,446 availability-positive trials, not 4,336 delivered-reward trials. |
| “Day of training” metadata | Descriptors primarily use numeric `sess#`; later train2-after sessions use `days` or a day-like `sess#` | Every physical session has at least one numeric value via `days` or `sess#`; duplicates agree at the physical-session level for the chosen primary stage | Paper describes before/after stages and approximately 2-week phases but does not list every session day | Use `days` when present, otherwise `sess#`, from the primary experiment descriptor. This is the only released continuous per-session training-stage/day field; store source group/value in metadata. |
| Stimulus naming | Code canonicalizes stimuli differently for each analysis using `stim_id`; figures call natural categories circle/leaf even for some rock/brick mice | Native `WallName` has 15 exemplar labels across four image families (circle, leaf, rock, wood) | Paper distinguishes four texture images: circle, leaf, rock, and brick, while often relabeling pairwise analyses for simplicity | Decoder target says visual stimulus *category*, so collapse suffix/exemplar/swap labels to four raw families: circle, leaf, rock, wood (wood is the native-data name corresponding to the paper's brick family). Preserve native `WallName` mappings in metadata. |
| Decoder accuracy | No decoder implementation in reference code | N/A | No categorical decoding accuracy is reported | Do not invent a paper benchmark; use chance-level comparisons and the supplied decoder. |

Final reconciled understanding: the candidate dataset is all 89 unique imaging recordings only; behavior-only cohorts lack neural input and are excluded. Each trial is sampled on original 3.17 Hz imaging frames that lie in the 0–4 m textured corridor while VR advances. Released deconvolved traces, aligned frame-level behavior, named visual-area neurons, and all otherwise valid trials are retained.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| Neural file `spks` list | `neural` | Concatenate imaging planes in released order, retain named-area cells, then index original imaging frames selected per trial; float32 `(n_neurons, T)` | `load_spk`; `Get_dprime_selective_neuron` | Released non-negative deconvolved fluorescence; no ΔF/F, smoothing, temporal binning, interpolation, or z-scoring. |
| `ft_trInd`, `ft_CorrSpc`, `ft_move` | trial timepoint selection | For trial `t`, retain indices where `ft_trInd == t`, `ft_CorrSpc`, and `ft_move > 0` | `Get_dprime_selective_neuron`, `Get_coding_direction`, `Get_sort_spk` | Exact original frames in 0–4 m while VR advances; every native trial has >=2 (observed min 11). |
| `SoundTime`, `ft` | `input[0]` | `(SoundTime[trial] - ft[selected_frames]) * 86400`, signed seconds to cue; float32 | Frame alignment variables documented in processing notebook; cue alignment in `spk_2_cue` | Positive before cue, zero at cue, negative after cue. Decoder Task explicitly requests continuous time-to-cue, so this is not replaced with a binary onset pulse. |
| experiment descriptor `days`, else `sess#` | `input[1]` | Numeric session training day/stage, repeated across trial frames; float32 | `Imaging_Exp_info.npy` used by processing notebook | `days` has priority because it is explicit; `sess#` is the released fallback. Source field/group stored in session metadata. |
| `Trial_start_time`, `ft` | `input[2]` | `(ft[selected_frames] - Trial_start_time[trial]) * 86400`, seconds since corridor entry; float32 | Frame alignment documented in notebook | Preserves actual elapsed time across omitted stationary frames. |
| `WallName`, `isRew` | `input[3]` | Identify the unique rewarded wall from `WallName[isRew]`; mark all trials of that wall 1, others 0; repeat across frames; float32 | `get_cat_id` similarly identifies rewarded stimulus by wall | Encodes reward *availability*, not delivery. No-reward sessions are all 0. |
| `WallName` prefix | `output[0]` | Map `circle*→0`, `leaf*→1`, `rock*→2`, `wood*→3`; repeat trial category across frames; int16 | Paper's category definitions; native `WallName` | Exemplar number and swap do not change visual category. Values: circle, leaf, rock, wood (paper calls the last image family brick). |
| `LickFr`, `LickTrind` | `output[1]` | Reference-style integer/floor frame assignment; 1 where at least one lick from that trial lands on selected frame, else 0; int16 | `spk_2_firstLick`, `spk_2_cue` cast `LickFr.astype(int)` | Binary even if multiple licks occupy one frame. Licks during excluded stationary/gray frames are outside the converted interval. |
| `ft_Pos` | `output[2]` | `floor(position_source_units / 10)` for selected 0–40 source-unit frames; clip defensively to 0–3; int16 | `get_interpPos_spk` establishes 10 source units = 1 m | Four equal physical bins: 0–1, 1–2, 2–3, 3–4 m. Direct `ft_Pos` avoids cumulative-position offsets observed in 21 boundary samples. |
| `ft_RunSpeed` | `output[3]` | Compute 25th/50th/75th percentiles over all selected timepoints in the converted session set; `np.digitize` into 0–3; int16 | Paper running-speed section uses imaging-frame interpolation; notebook defines `ft_RunSpeed` | Full-data preliminary edges: about 12.4224, 25.3526, 40.8546 (native speed units); script recomputes exact edges for full/sample modes and stores them. |
| retinotopy `iarea` | `brain_region_idx` and neural row filter | Keep codes other than −1,7; map 8→V1, {0,1,2,9}→mHV, {5,6}→lHV, {3,4}→aHV | `neu_area_ID`; `Get_density_map` | `brain_regions = ['V1','mHV','lHV','aHV']`; row order remains exactly the released concatenated cell order. |
| mouse name in experiment descriptor | `subjects`, `subject_idx` | Deterministic subject list and per-session integer lookup | Experiment-index loading | Sample mode lists only subjects present in its two sessions; full mode contains all 19. |

### Key Decisions
1. **Session identity**: Use the first experiment-index occurrence of each mouse/date/block and convert 89 physical recordings. Repeated group labels and swap annotations do not create new neural sessions.
2. **Temporal representation**: Preserve original imaging-frame activity rather than spatially interpolating to 60 bins. The target requires a common time-bin duration and temporal cue/lick variables; spatial interpolation would replace time with position. The nominal bin is `1000/3.17 = 315.4574 ms`, while actual elapsed times are carried explicitly because excluded stationary frames create gaps.
3. **Reference running restriction**: Retain only advancing-VR textured-corridor frames, matching all main neural-response analyses and preventing reward-consumption pauses from dominating. Preliminary audit: 821,579 selected observations, 11–178 per trial (median 21), with every trial retained.
4. **Neural values**: Preserve float32 deconvolved activity without normalization. Reference d-prime uses raw deconvolved frames; decoder code learns per-session projections and does not require conversion-time z-scoring.
5. **Area curation**: Exclude raw area −1/7 cells because the reference says they are outside visual cortex. No stimulus-selectivity filtering is applied: d-prime thresholds are analysis-specific and would leak the requested visual output into neuron selection.
6. **Trial curation**: Keep every imaging trial because all have valid aligned running/corridor frames and all sessions have at least 84 trials. Behavior-only sessions are excluded solely because they lack neural input.
7. **Mixed variable shapes**: Store all inputs as float32 `(4,T)` and outputs as integer `(4,T)`, broadcasting per-trial day/reward/category. A single trial array cannot otherwise combine scalar and time-varying dimensions, and this is the validator/trainer's documented convention.
8. **Speed binning**: Use global selected-observation quartiles, not per-session quartiles, so labels have a consistent meaning for the shared decoder and each class represents approximately 25% of full data.
9. **Metadata**: `off_start=0.0` because conceptual trial start is corridor entry; `off_end=None` because elapsed end time varies. Document moving-frame selection/gaps, speed edges, full native/retained counts, category mapping, and per-session source details.
10. **Storage/efficiency**: Load/process one multi-GB neural session at a time; filter planes before concatenation; use float32 neural/input and compact integer outputs; release raw arrays immediately. Behavior prepass is small and computes global speed thresholds before neural I/O.

### Planned Sanity Checks
- [ ] Structural: 89 unique full sessions (2 sample), aligned neural/input/output trial counts, every trial `T>=2`, constant neurons within session, no NaN/Inf.
- [ ] Native neural spot-check: load a raw neural file independently and use `np.allclose` on specific retained cells/frames versus converted trials.
- [ ] Native input spot-check: independently recompute cue time, training day, elapsed time, and reward availability for specific frames and compare with `np.allclose`.
- [ ] Native output spot-check: independently recompute category, lick event, 1-m position bin, and quartile speed bin for specific frames and compare with `np.allclose`.
- [ ] Alignment: selected raw indices exactly satisfy trial ID, textured-corridor, and movement masks; plot neural/activity plus cue, lick, position, speed after alignment.
- [ ] Curation: raw cells = 4,691,034; retained named-area cells = 4,105,393; per-session raw range matches paper 20,547–89,577.
- [ ] Dataset size: match paper's 89 recordings/19 mice and native 38,110 trials; no physical session duplicated or lost.
- [ ] Outputs: category/lick ranges valid; position bins cover 0–3 and correspond exactly to meter intervals; speed class fractions each approximately 0.25.
- [ ] Reward: availability positives equal all trials of the identified rewarded wall (preliminary 4,446), exceeding delivered rewards (4,336) only in active sessions as expected.
- [ ] Reference comparison: raw imaging-frame values (not spatial interpolation) and movement/corridor masks match the applicable reference functions; all deliberate differences documented.

---

## Step 6: Script Development
**Status**: COMPLETE

[Implemented `/app/convert_data.py` with the required positional output path, default `--full`, `--sample`, and `--show-processing`. The script builds a deduplicated 89-session manifest; validates behavior geometry/shapes; performs a behavior-only global speed-quartile prepass; loads, area-filters, aligns, and releases one neural session at a time; constructs typed trial arrays; records rich metadata; and writes with the highest pickle protocol. It compiles successfully and its CLI was checked.]

Code inefficiencies identified:
[A naive implementation would concatenate all raw planes (including excluded cells), retain raw and filtered copies, repeatedly load behavior groups, recompute quartiles during neural I/O, and build temporary lists during concatenation. At full scale those choices would add tens to hundreds of GB of transient memory and redundant I/O.]

Code speedups added:
[Behavior files are loaded once per relevant group in each pass; speed thresholds are computed before neural loading; retained neural matrices are preallocated and filled plane-by-plane without first concatenating excluded cells; trials use direct vectorized indexing/stacking; outputs use int16; raw session arrays are garbage-collected promptly; timing and rolling ETA are printed. Parallel neural-file loading was intentionally not used because sequential access avoids I/O contention and bounds peak memory for 1.5–10 GB source files.]

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 86,097 retained named-area neuron-session units (106,665 raw) |
| Neurons / session | [32,387, 53,710]; mean 43,048.5 |
| Subjects | 2: TX83, TX88 |
| Sessions / subject | 1 each |
| Trials (total) | 682 |
| Trials / session | [400, 282] |
| Selected timepoints | 15,845; T min 20, mean across session means 23.46, max 41 |
| Time to sound cue range | [-79.9, 64.7] s |
| Day of training range | [0, 0] (expected for both pre-learning sessions) |
| Time since trial start range | [0.0, 83.2] s |
| Reward availability range | [0, 0] (expected unsupervised sessions) |
| Visual category distribution | circle 0.527, leaf 0.473 |
| Licking distribution | not licking 1.000 (expected in these two no-reward pre-learning sessions) |
| Position-bin distribution | [0.253, 0.253, 0.249, 0.245] |
| Speed-quartile distribution | [0.250, 0.250, 0.250, 0.250] |

### Processing Plots Review
`processing_TX83_2022_08_17_1.png` and `processing_TX88_2022_06_13_2.png` were visually inspected. Region bars show raw versus retained curation; red selected frames coincide with valid native activity; the retained heatmaps show nonzero variable neural activity without striping/misalignment; position increases from 0–4 m after corridor-entry alignment; cue lines occur at plausible variable positions/times; any pauses appear as elapsed-time gaps/plateaus rather than reordered samples; position histograms cover four equal bins; speed thresholds divide the pooled sample; and final input/output traces match their source panels. No anomaly required correction.

`train_decoder.py --verify-only` reported: “Data format is valid, no errors or warnings.” Manual inspection confirmed 682 source trials became 682 converted trials and metadata matches source session IDs/counts.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Filter raw planes directly into a preallocated retained matrix | Avoids one full all-cell concatenated copy per session |
| One behavior prepass; vectorized masks/discretization | Prepass took only 0.17 s for 682 trials |
| Sequential per-session release and protocol-5 pickle | Bounds working memory and wrote 2.468 GiB in 2.29 s |

| Step | Time / Session | Estimated Total Time |
| Behavior prepass | 0.085 s/sample session | <10 s full |
| Neural conversion including diagnostic plot | ~9.9 s/session | plots are disabled for full; conservatively ~12–14 min conversion |
| Serialization | 2.29 s for 2.468 GiB | approximately 2–3 min if full output is 120–170 GiB |
| Total | 22.43 s for sample with two plots | approximately 14–15 min; near the guideline but no avoidable full-matrix copy/I/O remains |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: No data-format warnings. Scikit-learn emitted its expected single-label confusion-matrix warning for licking because both deliberately sampled unsupervised pre-learning sessions contain no licks on retained frames. Full data contain both licking classes and will test this output nontrivially.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Visual stimulus category | 0.8594 | 0.8555 |
| Licking | 1.0000 | 1.0000 |
| Position (1-m bin) | 0.7893 | 0.7796 |
| Running-speed quartile | 0.4496 | 0.4505 |

Training completed on CUDA. Loss decreased from 7,322.07 at epoch 1 to 107.90 at epoch 200 (minor late fluctuations but a clear sustained decline); test loss was 21.10. The supplied script's stated chance levels were 0.25, 0.50, 0.25, and 0.25, respectively, and every validation result exceeded its stated chance. Stimulus also exceeds the stricter 0.50 two-class chance applicable to this sample.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 141.129 GiB
- `verification_full_out.txt`: created

Conversion completed all 89 sessions without a failed assertion. In-memory conversion took 1,393.93 s, pickle writing took 123.12 s, and total runtime was 1,517.05 s (25.28 min). This exceeded the sample-derived guideline estimate because cumulative multi-GB neural reads and retained float32 copies scaled less favorably than the first two cached files; no semantics-preserving way exists to avoid reading source activity or writing target activity. The run remained below the 1.5× abort threshold relative to the conservative upper estimate, so it was allowed to finish.

`train_decoder.py --verify-only` loaded and scanned the complete file and reported “Data format is valid, no errors or warnings.” Its scan itself provides an all-session integrity check: consistent neurons per session, matching time axes, categorical outputs, finite values, and aligned trial counts.

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | Not totaled; raw per-recording range reported | `Get_density_map` excludes raw −1/7 outside cortex | 4,691,034 raw; 4,105,393 named-area | 4,105,393 | Yes, exact after documented reference area curation |
| Mean neurons/session | Not reported | Named areas V1/mHV/lHV/aHV | 52,708.25 raw; 46,128.01 retained | 46,128.01 | Yes |
| Raw neurons/session range | 20,547–89,577 | `load_spk` all released cells | 20,547–89,577 | Metadata retains raw range; converted named-area range 17,363–78,815 | Yes |
| Subjects | 19 imaging mice | experiment index | 19 | 19 | Yes |
| Sessions | 89 recordings | 89 physical neural files, 142 analysis appearances | 89 | 89 | Yes |
| Trials (total) | Not reported | behavior `ntrials` | 38,110 | 38,110 | Yes |
| Trials/session (mean) | Not reported | behavior `ntrials` | 428.20 | 428.20 | Yes |
| Selected timepoints | Running textured-corridor frames | `ft_CorrSpc & (ft_move>0)` | 821,579 | 821,579 | Yes |
| Input: time to cue | Cue randomly 0.5–3.5 m; elapsed ranges not reported | imaging-aligned timestamps/frames | [-1763.3, 723.5] s after preserving pauses | [-1763.3, 723.5] s | Yes |
| Input: day | Stages and ~2-week periods; exact days not all listed | `days`, otherwise `sess#` | [0,15] | [0,15] | Yes |
| Input: time since start | Corridor-entry timestamps | `ft`, `Trial_start_time` | [0,1765.2] s | [0,1765.2] s | Yes |
| Input: reward availability | Reward available after cue in rewarded wall | rewarded wall identified from `isRew` | [0,1]; 4,446 positive trials | [0,1]; same mapping | Yes |
| Visual category distribution | Four natural texture families described | Native wall names / group-specific canonical IDs | timepoint-weighted [0.311,0.470,0.085,0.135] | same | Yes |
| Licking distribution | Anticipatory licking in task sessions; absent in unsupervised | `LickFr.astype(int)` | timepoint-weighted [0.963,0.037] | [0.963,0.037] | Yes |
| Position-bin distribution | 0–4 m texture region | 0.1-m positional representation available | [0.250,0.249,0.250,0.252] | same | Yes |
| Speed-bin distribution | Running frames; speed interpolated to imaging frames | `ft_RunSpeed` | quartiles [0.250,0.250,0.250,0.250] | exact | Yes |
| Brain-region counts | V1 plus medial/lateral/anterior visual areas | `neu_area_ID` | V1 1,833,035; mHV 1,108,860; lHV 495,318; aHV 668,180 | exact | Yes |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Read `verification_full_out.txt`. Result: validator explicitly reported valid format with no errors or warnings. All dimensions/ranges/classes are sensible. Extreme elapsed/cue times are genuine trials with long pauses; because stationary observations are removed but exact timestamps are preserved, these values are expected and documented rather than silently compressed.
2. **Independent raw neural/input/output `np.allclose` tests**: Created and ran `cache/sanity_checks.py` without importing conversion code; output is in `cache/sanity_check_out.txt`. It loaded original neural, behavior, and retinotopy files for sessions 0, 18, 65, and 88 and checked trials 0, 5, and the last trial in each. Eleven evenly spaced cells × up to seven timepoints per trial matched raw float32 neural values exactly (`rtol=atol=0`). All four input rows matched independent timestamp/day/reward reconstruction (`np.allclose`), and all four output rows matched independent category/lick/position/speed reconstruction exactly. All checks passed.
3. **All-session alignment/curation scan**: Independently reconstructed the raw mask for every trial of every session. Every one of 38,110 converted trial lengths equals its native selected-frame count; all region-index arrays match independent raw maps. Spot-checked boundaries contain the exact trial ID, no gray-space frame, no stationary frame, and positions only in `[0,40)`. First/middle/last trials prevent beginning/end off-by-one errors from escaping. All passed.
4. **Key-statistics comparison from originals**: Independently obtained 89 unique IDs, 19 subjects, 38,110 trials, 821,579 observations, 4,691,034 raw cells, 4,105,393 retained cells, brain-region counts `[1,833,035, 1,108,860, 495,318, 668,180]`, 4,336 reward deliveries, and 4,446 availability-positive trials. All equal notes, metadata, and converted structure. Speed bins are each 0.25 within `2e-6`; all four stimulus/position/speed classes and both lick classes occur.
5. **Reference processing comparison**:

| Stage | Reference | Conversion | Comparison / rationale |
|-------|-----------|------------|------------------------|
| Loading | `utils.load_spk` concatenates `spks` planes; notebook loads `Beh_<group>` and experiment index | `load_filtered_neural`, `build_session_records`, grouped behavior loading | Same files, plane order, physical identity; deduplicates analysis aliases as required for sessions. |
| Neuron/trial filtering | Suite2p upstream cell classification; `Get_density_map` excludes raw area −1/7; neural analyses use moving corridor frames | `raw_area_to_region_idx`; `selected_frames_by_trial` | Same area exclusion and exact `ft_CorrSpc & ft_move>0` mask. No d-prime selection because selecting on stimulus would leak a requested decoder output. All valid trials retained. |
| Temporal alignment | Notebook supplies `ft`, `ft_trInd`, event frames/times; paper uses original frames for d-prime and cue alignment | Selected original frame indices grouped by `ft_trInd`; exact elapsed timestamps relative to corridor entry/cue | Same aligned native samples, with exact time carried through gaps. |
| Binning | Reference temporal analyses use native imaging frames; spatial figures interpolate 60 position bins | Native 3.17 Hz samples; no spatial resampling; categorical 1-m position and global speed quartiles only as required | Correct difference: target is temporal and explicitly requests categorical position/speed. Spatial interpolation would violate constant time-bin meaning. |
| Input construction | Cue/start/reward/wall/frame variables defined in notebook; `get_cat_id` identifies rewarded wall | Signed cue time, released day/stage, elapsed start time, rewarded-wall availability | Same native variables; delivery corrected to availability per task specification. |
| Output construction | `LickFr.astype(int)` in cue/lick functions; `ft_Pos`, `ft_RunSpeed`, `WallName` available; paper defines texture families | Same lick frame convention; four 1-m bins; global speed quartiles; four visual families | Exact requested transformations and raw spot checks pass. |

6. **Edge cases**: Verified fractional `StartFr`/`GrayFr` are not rounded (mask used instead); cumulative-position offsets are avoided by direct `ft_Pos`; active missed-reward trials remain availability-positive; duplicate swap keys produce one physical session; neural arrays one frame shorter than some behavior arrays never lose a selected frame; min/max raw recordings and min/max trials were included; every session has at least two trials and every trial at least 11 selected frames.

### Issues Found and Resolved
- No mismatch was found in the converted artifact during this review, so no conversion rerun was required.
- Earlier planning-stage issues already resolved before conversion: `isRew` undercounted active reward availability (fixed by rewarded-wall mapping); cumulative position had rare per-session offset ambiguity (fixed by direct `ft_Pos`); 142 analysis descriptors duplicated 89 physical sessions (fixed by physical ID deduplication); raw codes −1/7 lacked named regions (excluded exactly as reference regional analysis does).
- Full conversion runtime exceeded the preferred 15-minute guideline but not due to redundant computation: the artifact requires reading 405 GB of neural sources and writing 141 GB of exact float32 trial data. The script already avoids full raw concatenation and loads one session at a time.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples 2>&1 | tee /app/train_decoder_full_out.txt`
- Exit status: 0 (`train_decoder.py finished successfully.`)
- Device: CUDA.
- Loss decreasing: Yes. Training loss fell from 17,866.954370 at epoch 1 to 163.714855 at epoch 200 (99.08% reduction). It declined strongly overall, with only small late-stage stochastic fluctuations (for example, 213.338214 at epoch 130 and 247.523560 at epoch 140) before reaching a new minimum at epoch 200.
- Held-out test loss: 99.824343.
- Plot outputs: `/app/sample_trials.png` and `/app/predictions.png` were created.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Visual stimulus category | 0.8232 | 0.8131 | Chance = 0.25 |
| Licking | 0.9327 | 0.8482 | Chance = 0.50; class-balanced loss/metric used |
| Position (1-m bin) | 0.3124 | 0.3104 | Chance = 0.25 |
| Running speed quartile | 0.3500 | 0.3487 | Chance = 0.25 |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Validation balanced accuracy | Uniform chance | Accuracy / chance | Train / validation | Expectation from paper |
|----------|------------------------------|----------------|-------------------|--------------------|------------------------|
| Visual stimulus category | 0.8131 | 0.2500 | 3.2524x | 1.0124x | No categorical decoding accuracy reported |
| Licking | 0.8482 | 0.5000 | 1.6964x | 1.0996x | No categorical decoding accuracy reported |
| Position (1-m bin) | 0.3104 | 0.2500 | 1.2416x | 1.0064x | No categorical decoding accuracy reported |
| Running speed quartile | 0.3487 | 0.2500 | 1.3948x | 1.0037x | No categorical decoding accuracy reported |

#### Check 1: Accuracy versus chance

- Every validation score is above its balanced-accuracy chance level. Visual category and licking exceed 1.5x chance. Position and speed are above chance but below the requested 1.5x-chance screening threshold, so both were investigated in the checks below.
- The output distributions provide adequate variation: position fractions are `[0.250, 0.249, 0.250, 0.252]`; speed fractions are `[0.250, 0.250, 0.250, 0.250]`. Licking is naturally sparse (`0.963/0.037`) but the supplied class-balanced loss and balanced-accuracy metric prevent the majority class from inflating the score.

#### Check 2: Accuracy comparison to the paper

- A full-text search of all 25 pages of `paper.pdf`, together with `methods.txt` and the reference code, found no categorical decoder or classification-accuracy result for any of these four requested variables. The paper instead reports d-prime selectivity, sequence correlations, coding-direction projections/similarity indices, and statistical tests. Consequently, there is no paper accuracy number to compare without inventing a benchmark.
- The applicable benchmark is therefore the supplied decoder's uniform balanced-accuracy chance level, shown above.

#### Check 3: Train versus validation gap

- Train/validation ratios are 1.0124, 1.0996, 1.0064, and 1.0037. None approaches the `>1.5x` threshold. Absolute gaps are 0.0101, 0.0845, 0.0020, and 0.0013, respectively. This rules out material overfitting and provides no evidence of leakage.

#### Targeted investigation of position and speed

1. **Raw-output verification**: The independent `/app/cache/sanity_checks.py` audit loaded the original behavior and neural files directly rather than calling conversion functions. For `TX83_2022_08_17_1` trials 0, 5, and 399, `np.allclose` passed for all four output rows; the same exact checks also passed for nine trials from three widely separated sessions (`VR2_2021_03_20_1`, `TX61_2021_06_25_1`, and `TX139_2024_05_31_1`). Position, speed, licking, and stimulus labels are therefore not shifted or miscomputed in the tested trials.
2. **Temporal alignment**: `sample_trials.png` shows the expected monotone 0-to-3 corridor-bin sequence and framewise speed variation on the same time axes as neural activity. `predictions.png` shows position predictions tracking the ordered bin steps in held-out trials. Step 10 additionally verified native selected-frame IDs and trial lengths for every session, and exact timestamps/arrays in the spot checks.
3. **Variation**: Neither target is dominated by a class; the full-data fractions are approximately or exactly quartile-balanced as reported above.
4. **Neural filtering**: Released Suite2p deconvolved `spks` are retained in reference plane order; raw retinotopy codes `-1` and `7` are excluded; frames use the reference textured-corridor and positive-movement mask. All checks passed. Stimulus-selectivity filtering was deliberately not applied because it is an analysis-specific criterion that would leak the requested stimulus target.
5. **Processing/reference match**: Position uses the native aligned `ft_Pos` and exact 1-m bins; speed uses native aligned `ft_Speed` and full-dataset quartile edges. These values matched independently reconstructed raw targets exactly.
6. **Decoder/task interpretation**: The supplied decoder makes framewise linear predictions after a learned 100-D session projection; it has no recurrent/convolutional temporal context. Instantaneous speed is correspondingly difficult, and elapsed time predicts position only imperfectly because running rates vary. Supplying raw position or speed as decoder inputs would trivially improve results but would violate the requested input/output split. Altering the supplied decoder architecture would not diagnose the conversion. Thus, the modest position/speed scores are task/model limitations, not evidence supporting a conversion change.

### Issues Found and Resolved
- **Position and speed below 1.5x chance screening level**: Fully investigated with raw-file equality tests, alignment plots, distribution checks, filtering/reference comparison, and train/validation-gap analysis. No conversion error was found, so no data change or rerun was justified.
- **Late loss fluctuations**: Epoch loss briefly rose at epochs 140 and 170, then reached its minimum at epoch 200. Overall loss decreased 99.08%; this is normal late-stage stochastic behavior and not a failed convergence criterion.
- No iteration was required because all diagnostic checks passed and no mismatch was discovered.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created with dataset description, loading example, variable definitions, key statistics, reproduction commands, and decoder results.
- [x] `cache/` folder created with `README_CACHE.md`; the independent audit script/log and generated root bytecode cache were moved there without deleting evidence.
- [x] All files organized. Required artifacts are present and nonempty; `convert_data.py` and `cache/sanity_checks.py` pass `py_compile`; all 14 workflow steps are complete; and the root contains only reference materials, runtime/conversion code, required datasets/logs, documentation, and diagnostic plots.

### Final deliverable audit

- Complete data: `/app/converted_data.pkl` (142 GiB as displayed by `ls -lh`; 141.129 GiB exact conversion report).
- Sample data: `/app/sample_data.pkl` (2.5 GiB as displayed by `ls -lh`; 2.468 GiB exact conversion report).
- Validation: sample and full verification logs report no errors or warnings.
- Training: sample and full decoder logs finish successfully; full loss and all train/validation accuracies are recorded in Steps 11–12.
- Visual checks: two processing plots, `sample_trials.png`, and `predictions.png` are present.
