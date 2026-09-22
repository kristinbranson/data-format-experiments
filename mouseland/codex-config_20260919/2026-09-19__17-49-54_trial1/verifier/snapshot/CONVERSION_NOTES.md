# Dataset Conversion Notes

## Overview
- **Dataset**: Unsupervised pretraining in biological neural networks (provided paper, code, and data)
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

Environment checks: Python 3.13.15, NumPy 2.4.4, and PyTorch 2.6.0+cu124 import successfully. Checkpoint confirmed `/app/CONVERSION_NOTES.md` exists.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_exp_beh` | `code/utils.py` | LOADING | Loads an experiment-type behavior dictionary from `beh/Beh_<exp_type>.npy`. |
| `load_spk` | `code/utils.py` | LOADING | Loads `<mouse>_<date>_<block>_neural_data.npy`, takes its `spks` list, and concatenates planes/cell groups along the neuron axis to produce neurons x imaging frames. |
| `load_retino` / `neu_area_ID` | `code/utils.py` | LOADING | Loads retinotopic registration and maps numeric area labels to V1, medial higher visual (mHV), lateral higher visual (lHV), and anterior higher visual (aHV). |
| `spk_pos_interp` / `get_interpPos_spk` | `code/utils.py` | PROCESSING | Linearly interpolates neural activity against cumulative VR position into 60 position bins per 6-m trial (1 dm/bin); only frames with `ft_move > 0` are passed by the processing notebook. |
| `dprime` | `code/utils.py` | PROCESSING | Computes per-neuron `2*(mean1-mean2)/(std1+std2)` from neurons x frames arrays. |
| `Get_dprime_selective_neuron` | `code/utils.py` | CURATION | Computes stimulus selectivity only from frames inside the textured corridor (`ft_CorrSpc`) while VR is moving; this is analysis-specific neuron characterization, not a global removal of cells. |
| `Get_coding_direction` | `code/utils.py` | PROCESSING | Uses alternating trials to estimate selectivity, requires corridor response above gray-space response for coding-direction neuron selection, normalizes to gray activity/condition SD, and projects held-out trials. |
| `Get_sort_spk` | `code/utils.py` | PROCESSING | Z-scores position-interpolated activity, selects top/bottom 5% stimulus-selective corridor-responsive neurons by area, and separates trial subsets for sorting/testing. |
| `lickCount` / `lick_response` | `code/utils.py` | PROCESSING | Constructs per-trial binary lick responses in experiment-defined temporal or spatial windows. |
| `spk_2_firstLick` / `spk_2_cue` | `code/utils.py` | PROCESSING | Aligns continuous neural activity and binned lick events to first lick or sound-cue imaging frames. |
| `get_kfold_reward_response` | `code/utils.py` | PROCESSING | Selects stimulus/reward-prediction neurons within 10-fold held-out trials and produces position-, lick-, and cue-aligned responses. |

### Notes
- `code/README.md` identifies `data_process_script.ipynb` as the canonical intermediate-data processing pipeline. The notebook loads `Imaging_Exp_info.npy`, behavior dictionaries, neural `spks`, and retinotopy registrations.
- The source is two-photon imaging activity already stored under `spks`. The reference code does **not** compute fluorescence dF/F and exposes no raw fluorescence processing or cell-quality filter. Therefore dF/F should not be recomputed during conversion unless later data/text evidence contradicts this.
- Canonical spatial processing removes stationary frames (`ft_move > 0`) and interpolates across cumulative position to 60 bins/trial. That spatial interpolation is appropriate for position-based paper analyses, but the requested downstream task explicitly requires temporal alignment to corridor entry; whether to retain raw imaging-frame sampling or temporally bin it will be resolved from data and methods in Steps 2-5.
- Reference behavior-to-neural alignment is by global imaging-frame indices and frame-wise behavior arrays (`ft_trInd`, `ft_WallID`, `ft_PosCum`, `ft_move`, `ft_CorrSpc`, `ft_GraySpc`); cue indices are `SoundFr`, and lick indices are `LickFr`/`LickTrind`.
- Stimulus IDs used by the code are `circle1=0`, `circle2=1`, `leaf1=2`, `leaf2=3`, `leaf3=4`, `leaf1_swap1=5`, and `leaf1_swap2=6` (availability depends on experiment).
- Region mapping in `neu_area_ID`: V1=`iarea==8`; mHV=`iarea in {0,1,2,9}`; lHV=`iarea in {5,6}`; aHV=`iarea in {3,4}`. Values `-1` and `7` are explicitly excluded only from cortical density-map analyses. The decoder mapping decision awaits inspection of the actual registrations.
- No globally applicable trial deletion is present in the canonical loader. Alternating/held-out trial restrictions are safeguards for specific selectivity and coding-direction analyses, not acquisition-quality curation.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Total on-disk size: approximately 412 GB.
- `data/beh/Imaging_Exp_info.npy`: small dictionary indexing 23 imaging experiment labels. It contains 142 label records but only 89 unique physical recording keys; repeated labels deliberately reuse recordings for different paper contrasts (for example `sup_test1` and `sup_train2_before_learning`). Swap experiment behavior dictionaries can also provide two labeled views of one recording.
- `data/beh/Beh_<experiment>.npy`: pickled dictionaries keyed by session (sometimes with `_swap1`/`_swap2`). Each value has 59 fields. Core per-trial arrays include `WallName`, `isRew`, `StartFr`, `EndFr`, `SoundFr`, `SoundPos`, `RewPos`, and trial timestamps; global event arrays include `LickFr`, `LickTrind`, and `LickPos`; imaging-frame arrays include `ft`, `ft_trInd`, `ft_Pos`, `ft_PosCum`, `ft_RunSpeed`, `ft_move`, `ft_WallID`, `ft_CorrSpc`, and `ft_GraySpc`. Every one of these fields is present for all 89 unique recordings.
- `data/spk/*_neural_data.npy`: 89 scalar object `.npy` files, one per physical recording. The object contains `spks`, a list of float32 neurons x frames matrices. The reference concatenates list elements along neurons. A directly inspected example (`TX124_2023_12_24_1`) contains three `(6849, 18491)` arrays; their concatenation has 20,547 neurons.
- `data/retinotopy/*_trans.npz`: 89 files, one per physical recording date, with registered cell positions and one `iarea` label per concatenated neuron. The `iarea` length exactly supplies the reference-loader neuron count. `areas.npz` contains cortical outline polygons.
- `data/process_data/` is empty; the provided source data therefore require loading raw deconvolved `spks`, not precomputed spatial interpolation products.
- `data/beh/Unsupervised_pretraining_behavior/` contains three behavior-only cohort files, and `example_bef_and_aft_learning_behavior.npy` is a figure example. These have no matched neural recording index and therefore are descriptive/reference behavior rather than decoder sessions.
- Neural/behavior frame counts have a systematic acquisition offset: neural arrays contain 1 fewer frame in 59 sessions, 2 fewer in 23 sessions, and 3 fewer in 7 sessions. This exactly explains why the reference code truncates frame-wise behavior arrays to `nfr`; conversion must do the same.
- Imaging timestamps are MATLAB datenums. Across sessions, the median frame interval is about 0.315 s (approximately 3.18 Hz; sampled range about 0.258-0.403 s, with rare within-file timestamp anomalies). Position is recorded in decimeters: `Texture_Length=40` (4 m corridor) and `Corridor_Length=60` (4 m textured corridor plus 2 m gray space).
- Repeated experiment labels were not double-counted in the physical-recording totals below. Across all labels there are 142 behavior representations and 63,177 represented trials; deduplication by neural-file key yields the authoritative 89 sessions and 38,110 acquired trials.
- Available wall/stimulus names are circle1/2/3, leaf1/2/3, leaf1_swap1/2, and matched alternate texture-family names rock1/2 and wood1/2/5/wood1_swap1/2. `stim_id` assigns paper-level category IDs but differs between paper contrasts for reused sessions; literal `WallName` remains the acquisition-level source of truth.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 4,691,034 across 89 unique recordings |
| Neurons / session | mean 52,708.25; median 54,741; range 20,547-89,577 |
| Subjects | 19 (`DR10`, `DR15`, `LZ13`, `LZ16`, `TX104`, `TX105`, `TX108`, `TX109`, `TX119`, `TX123`, `TX124`, `TX139`, `TX140`, `TX60`, `TX61`, `TX83`, `TX85`, `TX88`, `VR2`) |
| Sessions / subject | range 1-8; 89 total |
| Trials (total) | 38,110 unique acquired trials (after deduplicating repeated paper labels, before task-specific filtering) |
| Trials / session | mean 428.20; median 429; range 84-789 |

Additional native-data statistics: 2,025,281 behavior/imaging timestamps; 4,336 rewarded trials (11.38% of all unique trials; 37.61% within the 28 sessions containing rewards); 74,483 lick events, with no recorded licks in 61 non-rewarded sessions. Retinotopy assigns 4,105,393 neurons (87.52%) to the four reference visual-region groups and 585,641 to `iarea=-1` or `iarea=7`, which the paper excludes from cortical density-map analyses. Region counts before grouping are: -1=397,310; 0=275,623; 1=147,071; 2=139,295; 3=568,741; 4=99,439; 5=342,053; 6=153,265; 7=188,331; 8=1,833,035; 9=546,871.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Not reported as a summed total | Paper reports populations “up to 90,000 neurons simultaneously”; Methods reports the per-recording range. | 
| Neurons / session | 20,547-89,577 | “We ran Suite2p on this data to obtain the activity traces from 20,547 to 89,577 neurons in each recording.” |
| Subjects | 19 imaging mice | “We performed 89 recordings in 19 mice…” |
| Sessions / subject | 89 recordings / 19 mice overall; group/session counts vary by analysis | Methods and figure legends; for example naive test 1 has 9 mice/11 sessions and grating has 3 mice/5 sessions. |
| Trials (total) | Not reported | No paper-wide trial total is supplied. |
| Trials / session | Not reported | Trial counts are shown for examples but no global summary is stated. |
| Neural data time bin | Original deconvolved imaging frames; no temporal binning reported | “These data points are computed from original estimated deconvolved traces without interpolation” for selectivity. |
| Behavior data time bin | Continuous acquisition; spatial analyses interpolate at 0.1 m | Running speed was interpolated “for every position (0–6 m, with a 0.1-m step size).” |
| Reward rate | Not reported as a fraction | Water was available only after the sound cue in the rewarded corridor; corridors were pseudo-random. |
| Corridor geometry | 4 m visual corridor + 2 m gray space | “The virtual reality corridors were each 4 m long, with 2 m of grey space between corridors.” |
| Sound-cue position | Uniformly randomized from 0.5-3.5 m for imaging mice | Methods, Behavioral training. |
| VR motion / running threshold | VR moves at 60 cm/s while mouse speed exceeds 6 cm/s; otherwise stationary | Methods, Visual stimuli. |
| Training interval | Approximately 2 weeks for imaging task timeline | Main text and Fig. 1 timeline. |


### Processing Details
- Calcium data were processed in Suite2p (motion correction, ROI detection, cell classification, neuropil correction, and non-negative spike deconvolution) with a 0.75-s decay timescale. All paper analyses use the supplied deconvolved fluorescence traces. No dF/F recomputation is appropriate.
- The paper's primary selectivity calculation uses original (non-interpolated) deconvolved frames, restricted to the textured 0-4 m portion and timepoints when the animal is running. Means and standard deviations are pooled across eligible positions/frames for each corridor, with selectivity `d' = 2*(mu1-mu2)/(sigma1+sigma2)` and a typical threshold of absolute d' >= 0.3.
- Position-resolved analyses interpolate activity to 0.1-m spatial samples. Coding-direction analyses subtract gray-space baseline and divide by the average standard deviation in the two reference corridors. Trial splits prevent using the same trials for selection and evaluation.
- Reward-prediction analysis interpolates single-neuron activity by position, divides rewarded-category trials into early- versus late-cue groups, and uses 10-fold held-out neuron selection. Cue position is used because it is present in every corridor and strongly correlated with reward position in rewarded trials.
- Sound cue is present in all imaging trial types, including unsupervised/no-reward trials. In task mice it starts reward availability in the rewarded corridor.
- Running speed analyses interpolate behavior to imaging frames. The paper considers periods above 6 cm/s (sustained at least 66 ms) and analyzes 0-4 m for textured-corridor summaries.
- No neural-decoder accuracy is reported anywhere in the paper: searches for decoder/decoding/decode/accuracy return no instances. The paper reports selectivity, correlations, coding-direction similarity indices, and statistical tests; those are not directly comparable accuracy benchmarks for the supplied neural decoder.

### Curation Steps

**Neuron curation rules**:
Suite2p cell classification was already applied upstream. The paper does not describe an additional global cell-quality threshold. Analysis-specific selection is based on d', cortical region, and sometimes corridor response above gray response. For density maps, neurons outside mapped visual cortex are excluded; the four pooled analysis regions are V1, medial, lateral, and anterior.

**Trial curation rules**:
There is no global acquisition-level trial deletion. Paper analyses restrict frames/trials to what their hypothesis requires: running frames and 0-4 m for selectivity; held-out/alternating trials for unbiased tuning/coding; rewarded trials with first lick after 2 m for first-lick reward-prediction analyses; and occasional mouse exclusion when the required trial subtype is absent. Those latter restrictions should not be generalized to the decoder task.

### Decoders Trained
| Decoded variable | Accuracy |
| None (paper did not train/report a categorical neural decoder) | N/A |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Recording count | Experiment index contains 142 label records | 89 neural files and 89 retinotopy files | 89 recordings in 19 mice | One target session per physical neural-file key. Do not duplicate recordings reused under several analysis labels or the two swap views. |
| Neuron count | `load_spk` concatenates all `spks` list elements | Retinotopy length gives 4,691,034 total, 20,547-89,577/session | 20,547-89,577/session | Exact match. Retinotopy length is a safe pre-load count and concatenated `spks` is authoritative neural order. |
| Neural signal | Loader uses `spks` directly | Float32 non-negative `spks` matrices | Suite2p deconvolved fluorescence, 0.75-s decay | Use supplied `spks` without dF/F or additional deconvolution. |
| Frame alignment | Reference slices behavior arrays to neural `nfr` | Neural files are systematically 1-3 frames shorter | Analyses use imaging-aligned behavior | Truncate every frame-wise behavior vector and neural stream to their common minimum length before trial extraction. |
| Spatial versus temporal sampling | Notebook interpolates running frames to 60 x 0.1-m bins for position analyses; selectivity uses raw frames | Raw frame timestamps, positions, and trial IDs are available | Selectivity explicitly uses original frames; positional analyses interpolate | The requested trial-start temporal alignment overrides spatial resampling. Use original frame-aligned traces and uniform **temporal** resampling; preserve reference 0-4 m/running validity criteria where applicable. |
| Running/corridor validity | `ft_CorrSpc & (ft_move>0)` for selectivity | Both masks exist in all recordings | Only running timepoints in 0-4 m are analyzed | Eligible decoder samples are textured-corridor, running samples. Pauses/reward collection and 2-m gray intervals are invalid for this task. |
| Brain regions | Analyses use V1/mHV/lHV/aHV masks; density plots exclude `iarea=-1,7`, but `load_spk` and initial d-prime calculations load all cells | 4,105,393 mapped neurons; 585,641 labels -1 or 7 | Recorded-neuron range counts the full Suite2p population; conclusions are summarized in four visual groups | Preserve all recorded cells to avoid losing neural data. Use exactly the four reference groups and assign -1/7 to a fifth `unmapped/non-visual` group; downstream users can reproduce the paper mask by excluding that group. |
| Stimulus labels | Code defines canonical IDs 0-6 (`circle1`, `circle2`, `leaf1`, `leaf2`, `leaf3`, `leaf1_swap1`, `leaf1_swap2`) | Reused label files supply complementary, non-conflicting wall-to-ID mappings; four session/wall pairs (`circle3`) remain unlabeled | Different physical texture pairs are described generically as circle/leaf roles | Merge all label views per physical session, map by canonical `stim_id`, and exclude the 390 trials whose stimulus remains genuinely unlabeled; do not infer a paper category for NaN IDs. |
| Training day | Code mostly uses analysis labels, `sess#`, and occasional `days` | Dates exist for all sessions; `days` exists only for some train-2 endpoints and `sess#` is ordinal | Imaging is described before/after approximately two weeks, without a universal numeric day field | Use elapsed calendar days from each subject's first indexed imaging session as the only complete continuous day measure; retain original experiment labels/`sess#`/`days` in metadata for auditability. This avoids fabricating missing training-duration values. |
| Trial filtering | Alternating/k-fold filters are analysis-specific | Acquisition has 38,110 unique trials | No global trial deletion described | Do not impose selectivity-analysis train/test splits. Exclude only trials lacking a canonical output label or enough valid aligned corridor samples. |
| Decoder benchmark | No decoder function | N/A | No decode/decoder/accuracy result appears in the paper | No paper accuracy can be claimed; compare later results to per-output chance and neural plausibility, while using paper statistics for data consistency. |

Final consistent understanding: each physical file is one two-photon session; supplied float32 `spks` are already Suite2p cell-classified, neuropil-corrected, deconvolved traces; frame-wise behavior is aligned globally and must be truncated to neural length; trial zero is corridor entry, textured corridor is 0-4 m, and the paper's valid neural regime is active running. All recorded neurons are retained, with four reference visual-region groups plus an explicit group for labels the paper omits from region-specific analyses. Decoder-specific temporal construction is the intentional departure from the paper's position-resolved analyses.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| Neural object `spks` list | `neural` | For each physical session, preserve float32 deconvolved values and concatenate list elements in the reference neuron order; slice each eligible trial at common neural/behavior frames satisfying `ft_trInd==trial`, `ft_CorrSpc`, and `ft_move>0`. | `load_spk`, `Get_dprime_selective_neuron` | No dF/F, z-scoring, selectivity filtering, or temporal/spatial averaging. Native imaging samples are retained. |
| `SoundTime`, `ft` | `input[0]` | Signed seconds **to** cue: `(SoundTime[trial] - ft[frame]) * 86400`, varying over time (positive before cue, negative after). | `spk_2_cue`; frame-alignment logic throughout `utils.py` | Continuous as explicitly required, rather than binary cue onset. |
| Recording date (`datexp`) and subject | `input[1]` | Elapsed integer calendar days from that subject's earliest indexed imaging session; tiled across trial frames and stored float32. | Experiment index / paper timeline | Complete continuous proxy; original `sess#`, optional `days`, and experiment labels retained in metadata. |
| `Trial_start_time`, `ft` | `input[2]` | Seconds since corridor entry: `(ft[frame] - Trial_start_time[trial]) * 86400`; varying over time. | Frame-wise alignment fields | Temporal alignment is corridor entry; minor interpolation offsets are preserved rather than rounded. |
| `isRew` | `input[3]` | Trial value 1 for rewarded corridor, 0 otherwise, tiled across frames. | Reward/lick functions | This is reward **availability**, not whether reward was delivered. |
| Merged `UniqWalls` + `stim_id`, applied to `WallName` | `output[0]` | Canonical class 0-6, tiled across frames. Merge complementary paper-label views for the same physical recording. | `get_mean_lick_response`; `Get_coding_direction` canonical stimulus list | Values: circle1, circle2, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2. Exclude 309 trials with no finite ID in any provided label view; do not guess labels. |
| `LickFr` | `output[1]` | Binary per retained imaging frame; event frames use `LickFr.astype(int)` exactly as cue/first-lick reference functions do. Multiple events in a frame remain 1. | `spk_2_firstLick`, `spk_2_cue` | Values: not licking / licking. |
| `ft_Pos` + `ft_CorrSpc` | `output[2]` | `floor(ft_Pos / 10)` for decimeter positions, clipped/validated to classes 0-3 after the corridor mask. | 0.1-m spatial interpolation functions; Methods 0-4 m rule | Four equal 1-m bins: 0-1, 1-2, 2-3, 3-4 m. |
| `ft_RunSpeed` | `output[3]` | Global quartile thresholds over every retained frame, then `searchsorted(..., side='right')` into classes 0-3. | Paper running-speed interpolation | Planned thresholds from the full raw-data pass: 12.398619, 25.284774, 40.752700 cm/s; counts differ by at most one (203,877/203,876/203,876/203,877). |
| `mname` | `subjects`, `subject_idx` | Sorted unique mouse IDs and integer session mapping. | Experiment index | 19 subjects. |
| Retinotopy `iarea` | `brain_regions`, `brain_region_idx` | Reference masks: 8 -> V1; 0/1/2/9 -> mHV; 5/6 -> lHV; 3/4 -> aHV; -1/7 -> unmapped/non-visual. | `neu_area_ID` | All 4,691,034 cells retained; fifth class preserves cells excluded only from region-specific paper plots. |
| Remaining behavior fields (`Rew*`, `Sound*`, `StimTrial`, `StimFrame`, `SubjMove`, `run_pos`, wall/probe/type fields, VR position/time, gray/corridor masks and trial parity) | `metadata.session_info` or checks | Store concise source experiment labels, dates, blocks, original `sess#`/`days`, counts, reward mode, lengths, and source key; use detailed arrays for validation but do not duplicate them in the pickle. | Behavior loaders and analysis functions | These variables are available but are neither specified decoder inputs nor outputs; duplicating large raw arrays would be redundant. |

### Key Decisions
1. **Deduplicate to 89 physical sessions**: A session is a neural file, not a paper-analysis label. Reused labels are merged only to recover canonical stimulus IDs.
2. **Use all recorded neurons**: This preserves the source and paper-reported population sizes. Region-specific paper filtering remains reproducible from `brain_region_idx`.
3. **Use native imaging samples (nominal 314.8 ms)**: The paper explicitly uses original deconvolved frames for core selectivity. No temporal averaging is needed; trial lengths may vary and the supplied decoder supports this. Metadata will call this the nominal acquisition bin and include observed interval percentiles because timestamps have ordinary acquisition jitter.
4. **Restrict samples to active 0-4 m corridor frames**: This exactly matches the paper's neural validity regime and makes the four requested 1-m position classes exhaustive. Gray intervals and stopped/reward-collection samples are not decoder observations.
5. **Use variable trial lengths**: Retaining every valid native frame avoids padding, NaNs, or truncating slow/paused trials. The temporal alignment remains corridor entry through the explicit time-since-start input.
6. **No neural normalization**: Raw supplied deconvolved values match the core paper analysis and allow the decoder's learned session projection to choose scaling. Analysis-specific z-scoring used only for selected figure panels is not generalized.
7. **Continuous cue-time sign convention**: Positive means cue is in the future, zero at cue, negative after; this is the literal “time to cue” interpretation and is documented in metadata.
8. **Continuous training-day proxy**: Calendar-day elapsed time is complete for naive/task/unsupervised/grating sessions, unlike `days` and `sess#`. It is per-session/per-trial and does not invent exact unrecorded training duration.
9. **Sample selection**: `--sample` will deliberately choose one unrewarded and one rewarded recording so all binary task variables and lick classes are exercised; choosing the first two index entries would yield no lick/reward variation.
10. **Memory-efficient session assembly**: Do not concatenate a full multi-plane recording and then copy it. Allocate each trial once and fill neuron blocks from `spks`; retain source float32 precision. Full projected neural payload is estimated at 43,145,902,753 float32 values (172.58 GB decimal) before pickle overhead.

Planned post-filter statistics from an independent behavior-only pass: 89 sessions; 37,801 trials (84-722/session); 815,506 retained frame samples; trial T range 11-178, mean 21.57, median 21; no trial lacks valid running-corridor frames; 309 trials excluded solely for missing canonical stimulus ID. Position-frame counts are 203,842 / 202,873 / 203,517 / 205,274, consistent with four equal-length bins. Reward-availability frame counts are 714,861 / 100,645 and lick counts are 785,205 / 30,301.

### Planned Sanity Checks
- [ ] Neural raw spot-check: independently load a source neural file and use `np.allclose` on selected neuron/frame blocks versus converted trial matrices.
- [ ] Input raw spot-check: independently recompute cue time, elapsed day, trial time, and reward availability for at least three trials and compare using `np.allclose`.
- [ ] Output raw spot-check: independently reconstruct stimulus ID, lick-frame binary, position class, and speed quartile for at least three trials and compare using `np.allclose`.
- [ ] Assert the neural/behavior common-frame rule for all sessions and reproduce the observed 59/23/7 distribution of 1/2/3 missing terminal neural frames.
- [ ] Assert exactly 89 unique source neural keys, no duplicate session IDs, 19 subjects, and at least two retained trials per session.
- [ ] Assert neural count per session equals retinotopy length and remains in the paper's 20,547-89,577 range.
- [ ] Assert all trial neural/input/output time dimensions match, all values are finite, all outputs are integral and within declared classes, and every brain-region vector matches neuron count.
- [ ] Assert speed-bin counts differ by at most one and position classes are near-balanced as expected from constant-speed VR traversal.
- [ ] Compare full converted subject/session/neuron/trial statistics against paper, code, and the independent raw-data inventory.
- [ ] Inspect `--show-processing` plots for neural/behavior alignment, cue-time zero crossing, licking events, monotonic trial time, position-bin transitions, and speed thresholds.

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]
- Created `/app/convert_data.py` with required positional output path, default/explicit `--full`, `--sample`, and `--show-processing` modes.
- Converter builds a deduplicated 89-session catalog, merges complementary stimulus mappings, computes full-dataset speed quartiles, loads one source session at a time, and constructs shape-checked neural/input/output trial arrays.
- `--show-processing` generates a 10-panel audit for up to two sessions covering frame/trial masks, raw neural preservation, time/cue alignment, reward/lick mapping, position and speed discretization, output distributions, trial lengths, brain regions, and an explicit `np.allclose` neural comparison.
- Syntax compilation and a non-neural full-catalog dry run passed. It reproduced 89 sessions, 19 subjects, 4,691,034 neurons, speed thresholds `[12.39861906, 25.28477401, 40.75270048]`, speed-class counts `[203877, 203876, 203876, 203877]`, native median interval 314.804167 ms, and terminal frame offsets 59/23/7 for 1/2/3 frames.
- Sample mode deterministically selects `TX83_2022_08_17_1` (unrewarded) and `VR2_2021_03_20_1` (rewarded) so reward and licking variation are tested.

Code inefficiencies identified:
Naively calling the reference `load_spk` would concatenate the complete multi-plane recording and then copy it again into trials, adding up to roughly 10 GB peak memory for the largest source file. Re-reading large behavior files per session and loading neural data merely to learn its frame count would also add avoidable I/O.

Code speedups added:
The converter validates and uses the invariant scalar-NPY payload layout to infer neural frame counts from file size during the behavior-only prepass; behavior files are loaded once per source experiment group; trial arrays are allocated once and filled plane-by-plane without a full concatenated intermediate; source and target remain float32; and only compact metadata are duplicated. Per-session timing, payload, cumulative time, and ETA are printed.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 125,532 session-specific neurons |
| Neurons / session | 44,059; 81,473 |
| Subjects | 2 represented (`TX83`, `VR2`); full 19-name subject vocabulary retained |
| Sessions / subject | 1 each in sample |
| Trials (total) | 748 |
| Trials / session | 400; 348 |
| Time to sound cue range (s) | [-301.3, 83.1] |
| Day of training range | [0, 0] (both selected baseline recordings) |
| Time since trial start range (s) | [0, 304.8] |
| Reward availability range | [0, 1] |
| Visual stimulus distribution | circle1 0.488, leaf1 0.512 |
| Licking distribution | no lick 0.961, lick 0.039 |
| Position distribution | [0.249, 0.243, 0.249, 0.260] |
| Speed-quartile distribution | [0.243, 0.183, 0.329, 0.244] (sample-specific; full thresholds are globally balanced) |

### Processing Plots Review
Both 10-panel plots were inspected. Corridor/movement masks retain only running 0-4 m samples; position bins transition at the correct 1-m boundaries; time-to-cue crosses zero at cue time; the rewarded example shows the expected binary lick pulse; speed classes match the three global thresholds; region counts include all neurons; and raw-to-converted neural comparisons show maximum absolute difference 0 with `np.allclose=True`. Long wall-clock gaps occur when a task mouse pauses—the intervening stopped frames are intentionally excluded per the paper, while the explicit time inputs preserve the gap. This explains the legitimate 304.8-s maximum rather than indicating temporal misalignment.

`train_decoder.py --verify-only` reports the format valid with no errors or warnings. Manual reload assertions confirmed all required keys, identical neural/input/output time dimensions, finite inputs, integral outputs, 400/348 trials, and retained sample counts 8,855/9,600. No trials were dropped in either representative session.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| File-size/retinotopy frame-count inference in prepass | Avoids loading all 412 GB just to obtain frame counts; full prepass completes in about 3.4 s. |
| Plane-by-plane trial assembly | Avoids a second full-session concatenated neural copy and lowers peak memory/I/O. |
| Behavior-file reuse and float32 preservation | Avoids repeated 0.1-0.4 GB behavior loads and dtype expansion. |

| Step | Time / Session | Estimated Total Time |
| Sample neural conversion (including plots) | 8.25 s mean (4.28, 12.21 s) | About 9-12 min after accounting for full/sample payload ratio and no plots beyond two sessions |
| Pickle write | 3.70 s for 4.69 GB | About 2.3 min scaled to 172.6 GB neural payload |
| Full conversion total | sample total 22.91 s | Approximately 12-14.5 min, below the 15-min optimization threshold; estimate scales by payload rather than session count because file sizes differ |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: No data-format warnings. Scikit-learn emitted `y_pred contains classes not in y_true` during accuracy calculation because the two selected sessions contain canonical stimulus classes 0 and 2 but the model's contiguous 0-2 logits can predict absent class 1. This is sample-only and not a conversion defect; full data include all seven canonical classes.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Visual stimulus category | 0.9165 | 0.9114 |
| Licking | 0.8946 | 0.6999 |
| Corridor position bin | 0.7638 | 0.7414 |
| Running speed quartile | 0.6345 | 0.6235 |

Training completed all 200 epochs on GPU. Loss decreased monotonically from 10,597.6445 (epoch 1) to 193.8721 (epoch 200), with held-out test loss 104.1736. Every validation accuracy exceeds uniform chance: visual category also exceeds the stricter two-observed-class chance of 0.5; licking exceeds 0.5; position and speed exceed 0.25. The small train-validation gaps (1.01x, 1.28x, 1.03x, 1.02x) do not indicate severe overfitting.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 172,608,341,359 bytes (172.608 GB decimal)
- `verification_full_out.txt`: created

Full conversion processed 89 sessions in 769.19 s and wrote the pickle in 151.82 s, for 921.03 s total. This was 1.06x the 14.5-min upper estimate, not the >1.5x condition requiring termination/re-optimization. The full verifier reports “Data format is valid, no errors or warnings” and completes successfully.

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | Not summed | `load_spk` retains all | 4,691,034 | 4,691,034 | Yes |
| Mean neurons/session | Range only | all concatenated `spks` | 52,708.25 | 52,708.25 | Yes |
| Neuron range | 20,547-89,577 | all concatenated `spks` | 20,547-89,577 | 20,547-89,577 | Yes |
| Subjects | 19 | experiment index | 19 | 19 | Yes |
| Sessions | 89 recordings | 89 physical keys | 89 neural/retinotopy pairs | 89 | Yes |
| Trials (total) | Not reported | no global filter | 38,110 physical acquired | 37,801; exactly 309 excluded for missing canonical stimulus IDs | Explained task-required difference |
| Trials/session (mean) | Not reported | N/A | 428.20 before output-validity filter | 424.73 | Explained |
| Timepoints/trial | Native frames | raw imaging frame operations | valid running/corridor mean 21.57 per trial | session-mean aggregate 22.25; global trial mean 21.57; range 11-178 | Yes |
| Time to cue (s) | cue at 0.5-3.5 m; pauses possible | frame/cue aligned | [-1763.3, 723.5] after running-frame filter | [-1763.3, 723.5] | Yes |
| Training day | before/after timeline | partial `sess#`/`days` | calendar proxy [0,92] | [0,92] | Yes by documented definition |
| Position distribution | 0-4 m texture | 0.1-m reference interpolation | [0.250,0.249,0.250,0.252] | [0.250,0.249,0.250,0.252] | Yes |
| Speed distribution | running timepoints | `ft_RunSpeed` | exact global quartiles [0.25,0.25,0.25,0.25] | [0.25,0.25,0.25,0.25] | Yes |
| Visual distribution | role IDs 0-6 | canonical list | [0.319,0.060,0.335,0.170,0.059,0.027,0.029] | same | Yes |
| Licking distribution | licking only recorded in task sessions | `LickFr.astype(int)` | [0.963,0.037] | [0.963,0.037] | Yes |

Spot checks of early, middle, and final sessions in the conversion/verification logs show matching trial, timepoint, and neuron counts. Examples: `TX83_2022_08_17_1` = 400 trials/44,059 neurons/8,855 samples; `TX140_2024_05_31_1` = 398/49,311/9,147; `TX139_2024_05_31_1` = 369/34,522/8,444. The verifier independently traversed every trial and found no shape, dtype, finite-value, class-integrality, or region-index issue.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` contains “Data format is valid, no errors or warnings” and terminates with “Data verification complete.” No warning was left to explain or suppress.
2. **Independent raw `np.allclose` comparisons**: `/app/cache/critical_review.py` deliberately does not import the converter. It independently loads original behavior and neural files, reconstructs all four inputs and outputs, and compares beginning/middle/end sessions:
   - `TX83_2022_08_17_1`, converted/source trial 5, T=21: neural exact (`rtol=atol=0`), inputs within float32 tolerance, outputs exact. Spot neuron[3,10]=15.79517; input=[-0.17823,0,3.44423,0]; output=[0,0,2,1].
   - `LZ13_2024_05_16_2`, trial 123, T=19: all three comparisons pass. Spot neural=0; input=[2.10388,1,3.45312,0]; output=[4,0,2,3].
   - `TX139_2024_05_31_1`, trial 200, T=22: all comparisons pass. Spot neural=122.1826; input=[-0.56064,13,3.45563,0]; output=[2,0,2,1].
3. **Reference-code comparison**:
   - Loading: reference `load_spk` concatenates the `spks` list; converter fills trial arrays in exactly that list/neuron order without a full-session intermediate.
   - Neuron/trial filtering: reference does not globally quality-filter Suite2p cells; converter retains every cell. Reference core selectivity uses `ft_CorrSpc & ft_move>0`; converter uses the same sample mask. Decoder-only exclusions are the 309 trials with no canonical target label.
   - Alignment: both use the common neural/behavior prefix, global `ft_trInd`, and frame-indexed behavior. Converter additionally represents exact datenum-derived seconds from corridor entry/cue as required.
   - Binning: reference position analyses use 0.1-m interpolation, whereas the requested temporal decoder requires original imaging frames. Converter retains those frames and only categorizes raw `ft_Pos` into the requested 1-m classes. This is a task-required difference.
   - Input construction: cue/trial times use raw timestamps; reward availability is raw `isRew`; day is the documented complete calendar proxy because no universal paper day variable exists.
   - Output construction: canonical `stim_id` list matches reference; licking uses the reference integer frame convention; position follows paper 0-4 m; speed quartiles implement the explicit decoder requirement.
4. **Key statistics**: asserted 89 sessions, 19 subjects, 37,801 converted trials, 4,691,034 neurons, mean 52,708.247/session, 20,547-89,577 range, exact five-group region counts, exact visual/lick/position/speed counts, and exact 59/23/7 terminal-frame-offset counts. All match the independent raw inventory and all paper-available statistics.
5. **Edge cases**: asserted two or more trials/session; sorted unique source trial indices within bounds; matching neural/input/output T; float32 neural/input; integral categorical output; finite values; nonnegative trial time; constant cue time relative to trial start within each trial; constant per-trial day/reward/stimulus; speed class counts differing by at most one; 309 and only 309 missing-label trial exclusions; and zero trials lacking valid running-corridor frames.
6. **Final rerun**: after audit-harness refinements, every check above was rerun and `/app/cache/critical_review_out.txt` ends with `ALL CRITICAL REVIEW 1 CHECKS PASS`.

### Issues Found and Resolved
- **Iteration 1 — position-boundary assertion**: 21 of 37,801 trials contain one apparent 3→0 position-bin transition. Direct raw inspection showed these are original acquisition-boundary samples: interpolated `ft_Pos` has wrapped to approximately 0, while reference `ft_trInd`/`ft_WallID` still assigns that imaging frame to the prior trial because fractional `StartFr` lies just after the sample. Reassigning it would diverge from reference alignment. The converter was retained; the audit now allows exactly these 21 source-proven 3→0 cases and rejects any other backward transition.
- **Iteration 2 — audit source-anchor typo**: All data checks and raw comparisons passed, but the code-comparison harness searched for `LickFr.astype(int)` instead of the actual source text `['LickFr'].astype(int)`. The harness string was corrected; no data/code regeneration was needed.
- **Iteration 3**: Complete rerun passed log, raw comparisons, reference anchors, statistics, and edge checks. No unresolved issue remains.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples 2>&1 | tee /app/train_decoder_full_out.txt`
- GPU initialization completed, but the first epoch exceeded CUDA memory; the supplied trainer automatically restarted on CPU, as permitted by the workflow.
- All 200 CPU epochs completed and the script exited successfully. Loss decreased from 20,145.866886 (epoch 1) to 187.499923 (epoch 200), a 99.07% reduction. A small late stochastic fluctuation at epoch 180 resolved to new lows by epochs 190 and 200. Test loss was 122.774470.
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Visual stimulus category | 0.4932 | 0.4753 | 7 classes; chance 0.1429 |
| Licking | 0.9239 | 0.8462 | 2 classes; chance 0.5000 |
| Corridor position bin | 0.3442 | 0.3426 | 4 classes; chance 0.2500 |
| Running speed quartile | 0.3893 | 0.3890 | 4 classes; chance 0.2500 |

The required sample and prediction plots were generated as `/app/sample_trials.png` and `/app/predictions.png`.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Train | Validation | Chance | Validation / chance | Train / validation | Expectation from Paper |
|----------|-------|------------|--------|---------------------|--------------------|------------------------|
| Visual stimulus category | 0.4932 | 0.4753 | 0.1429 | 3.326 | 1.038 | Not reported |
| Licking | 0.9239 | 0.8462 | 0.5000 | 1.692 | 1.092 | Not reported |
| Corridor position bin | 0.3442 | 0.3426 | 0.2500 | 1.370 | 1.005 | Not reported |
| Running speed quartile | 0.3893 | 0.3890 | 0.2500 | 1.556 | 1.001 | Not reported |

- **Check 1, accuracy versus chance**: every validation accuracy is above chance. Visual, licking, and speed exceed 1.5x chance. Position is 1.370x chance, so it triggered the requested low-accuracy investigation.
- **Position investigation**: (1) Step 10 independently loaded original files and reconstructed outputs for three concrete trials spanning the dataset; all output arrays, including position, pass `np.allclose`; (2) `sample_trials.png` visually shows the expected monotonic 0→1→2→3 spatial sequence aligned with trial time, apart from the 21 source-proven acquisition-boundary frames already reviewed; (3) position classes are well balanced at [0.250, 0.249, 0.250, 0.252], ruling out class collapse; (4) reference running and 0-4 m corridor masks and unfiltered Suite2p neural streams are preserved; and (5) training and validation position accuracies are nearly identical (ratio 1.005), ruling out overfit. The prediction plot shows partial spatial tracking rather than a temporal offset. The remaining modest accuracy is plausible because a 1-m bin transition depends on variable running speed while trials contain only about 22 imaging frames on average. No conversion change is supported by the evidence.
- **Check 2, paper comparison**: exhaustive extracted-paper searches found 0 occurrences of `decod`, `accuracy`, or `classifier`. The paper reports selectivity/representation analyses and no accuracy for this newly specified decoder or any directly comparable output, so there is no numeric paper benchmark to populate. This is not used to dismiss a discrepancy; none exists to compare.
- **Check 3, train/validation gap**: train/validation ratios are 1.038, 1.092, 1.005, and 1.001, all far below the 1.5 threshold. There is no large generalization gap or evidence of leakage.
- Automated review is `/app/cache/critical_review_2.py`; `/app/cache/critical_review_2_out.txt` ends with `ALL CRITICAL REVIEW 2 AUTOMATED CHECKS PASS`.

### Issues Found and Resolved
- **Iteration 1 — audit-text assertion**: accuracy, gap, loss, and paper checks passed, but the harness expected a verbose `output allclose=True` phrase not used in the existing Step 10 log. The harness was corrected to require exactly three `CHECK 2 raw np.allclose: PASS` records plus all three session IDs.
- **Iteration 2**: all automated checks passed. No data or converter issue was found, so no conversion or retraining was warranted.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with dataset description, loading example, schema, statistics, reproduction commands, and decoder results.
- [x] `cache/` folder created; investigation scripts and their output logs are documented in `cache/README_CACHE.md`.
- [x] All files organized; required conversion, validation, training, processing-plot, prediction-plot, and documentation artifacts were checked for existence and nonzero size. Python scripts compile successfully.
