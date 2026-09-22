# Dataset Conversion Notes

## Overview
- **Dataset**: Track2p longitudinal mouse barrel-cortex two-photon imaging dataset (provided paper/code/data)
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment checks:
- Python 3.13.15
- NumPy 2.4.4
- PyTorch 2.6.0+cu124
- `/app/CONVERSION_NOTES.md` existence and all 14 step headings verified.

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `code/`
- `data/`
- `decoder.py`
- `docker-compose.yaml`
- `docker-compose.yaml~`
- `methods.txt`
- `paper.pdf`
- `train_decoder.py`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| Suite2p/raw-NPY loading functions | `track2p/io/loaders.py`, `track2p/io/s2p_loaders.py` | LOADING | Load mean field-of-view images, ROI masks/statistics, fluorescence arrays, and Suite2p `iscell` values for every session/plane. |
| `run_track2p` / `Track2p` orchestration | `track2p/t2p.py` | PROCESSING | Load datasets, register consecutive sessions, match ROIs, build tracked-cell matrices, and save results. |
| registration functions | `track2p/register/loop.py`, `track2p/register/elastix.py` | PROCESSING | Register fields of view and transform ROI masks between consecutive recordings. |
| `get_all_ds_assign` | `track2p/match/loop.py` | CURATION | Assign candidate ROI pairs, score overlap by IoU, and retain pairs above an automatic Otsu or minimum threshold. |
| `get_all_pl_match_mat` | `track2p/match/loop.py` | PROCESSING | Chain pairwise ROI identities into plane-specific cell-by-session matrices; tracking stops at the first missing consecutive match. |
| `get_iou`, cost/centroid helpers, `init_all_pl_match_mat` | `track2p/match/utils.py` | PROCESSING | Compute spatial matching metrics and initialize cross-session identity matrices. |
| save functions | `track2p/io/savers.py` | PROCESSING | Save Track2p outputs, parameters, matches, and optionally Suite2p-compatible products. |

### Notes
- The repository is the Track2p longitudinal calcium-imaging ROI registration/tracking package. README examples concern seven daily barrel-cortex recordings from P8-P14.
- Supported inputs are native Suite2p directories or raw NPY (`F.npy`, `fov.npy`, `rois.npy`) datasets. The utility notebook converts Suite2p by loading `ops.npy`, `stat.npy`, `iscell.npy`, and `F.npy`, filtering `stat` and `F` with the `iscell` boolean, constructing ROI masks, and saving filtered fluorescence/ROIs/FOV.
- Default cell curation is Suite2p `iscell[:,1] > 0.50` (`DefaultTrackOps.iscell_thr=0.50`). The code notes a permissive threshold is reasonable because artifacts are unlikely to recur consistently across datasets.
- Registration can use functional channel 0 or anatomical channel 1 and affine or rigid transforms. Matching defaults to IoU, skips expensive IoU comparisons for centroids farther than 16 pixels, uses linear assignment, then automatically thresholds match IoU (default Otsu).
- Pairwise matches are chained only through consecutive sessions. A cell is counted as tracked across all days only when every match-matrix entry is present.
- Crucially, the Track2p package does **not** compute dF/F, deconvolve fluorescence, temporally bin traces, align behavioral motion energy, or split recordings into trials. It preserves/loads fluorescence traces for tracked ROIs. The appropriate neural signal and behavior alignment must therefore be determined from the supplied data and reference text in Steps 2-4.
- No electrophysiology quality filtering applies; this is two-photon calcium imaging. Spatial ROI curation (`iscell`) and longitudinal identity matching are the reference-code curation operations.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Six subject directories: `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046`; dated session folders are chronologically sortable.
- Every session contains `suite2p/plane0/` arrays: `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), `spks.npy` (Suite2p deconvolved activity), `iscell.npy`, `stat.npy` (ROI pixel dictionaries), and `ops.npy` (Suite2p settings/images).
- Every session contains `move_deve/`: `motion_energy_glob.npy`, `tstamps.npy`, and `interframe_int.npy`. Motion energy and timestamps are one-dimensional and generally one sample per neural imaging frame.
- Subject-level `ground_truth.csv` files concern ROI-tracking evaluation. `load_data.ipynb` is the supplied usage documentation.
- Arrays are NumPy NPY; `F`, `Fneu`, and `spks` are float32 neuron-by-frame matrices; behavior timestamps/interframe intervals are float64; motion energy is numeric and finite; `stat` is an object array.
- The notebook explicitly calls `F.npy` **raw fluorescence** and says proper analysis should compute dF/F as described in the paper (or alternatively use `spks.npy`). Its raster z-scoring is visualization only.
- The notebook explicitly states rows are the same Track2p-matched cells, in matching order, across all days for each subject. The constant row counts and fully passing `iscell` arrays confirm these are pre-curated/pre-tracked products.
- Suite2p metadata report one 512x512 plane, `fs=30 Hz`, and 36,000 or 54,000 neural frames/session. Raw timestamp differences are about `3.36e-5` in their stored units and endpoints about 1.21 or 1.81; the physical-unit scaling is deferred to the paper/methods consistency review.
- Neural sessions always have their nominal frame count. Behavior is usually equal length but 9 sessions lose trailing samples (1-148 samples); this is an explicit edge case requiring common-valid-period truncation/alignment after reference timing is understood.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (unique tracked cells, summed across subjects) | 2,998 |
| Neurons / session | constant within subject; 221-746 (mean unique subject count 499.7) |
| Subjects | 6 |
| Sessions / subject | 6-7 (jm040: 6; all others: 7) |
| Sessions total | 41 |
| Session-neuron recordings | 20,198 |
| Native trials | none; continuous sessions (60-second trials must be constructed) |
| Neural frames / session | 36,000 (jm031/jm032) or 54,000 (others) |
| Nominal duration / session at 30 Hz | 1,200 s or 1,800 s |
| Motion samples / session | 35,852-36,000 or 53,999-54,000 |
| Brain area | mouse barrel cortex (project context/notebook example; confirm in paper) |

### Available Variables
- Neural candidates: raw `F`, neuropil `Fneu`, and deconvolved `spks`; ROI identity/geometry and Suite2p processing metadata are also present.
- Behavioral output candidate: global motion energy with corresponding frame timestamps and interframe intervals.
- No native trial/event variable exists; data are continuous longitudinal recordings.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (tracked) | 526 ± 190 SD per mouse | “On average 526 (± 190 std) neurons per mouse were successfully tracked across all days” |
| First-day tracking fraction | 33% ± 11% SD | “…corresponding to 33% (± 11% std) of the neurons detected on the first day” |
| Subjects | 6 mice | “full dataset of 6 mice” |
| Sessions / subject | at least 6 consecutive daily sessions | “imaged daily for a minimum of 6 consecutive days” |
| Developmental age | within P7-P14; principal example P8-P14 | Methods/results |
| Imaging | layer 2/3, barrel cortex; 720×720 µm FOV, 512×512 pixels | Chronic 2-photon methods |
| Neural data time bin | native 30 Hz (33.3 ms); decoder averaged 10 timestamps (333.3 ms) | “Imaging rate was 30 Hz”; “averaging in bins of 10 consecutive timestamps” |
| Behavior data time bin | video 30 Hz, acquisition-trigger synchronized; decoder averaged 10 timestamps | Videography and decoding methods |
| Session duration | 20 minutes | “each session lasted 20 minutes” |
| Motion-energy definition | sum of squared pixel differences between consecutive video frames | Preprocessing videography |
| Decoder output in paper | continuous motion-energy trace | Results/methods |
| Paper decoder metric | R², shown in same-day/cross-day matrices and box plots | Figure 6 / Supplementary Figure 7 captions |

### Processing Details
- Suite2p was run independently per recording: motion correction, ROI detection, signal extraction, and spike deconvolution.
- ROIs with Suite2p classifier probability strictly above the default 0.5 threshold were considered cells.
- The paper calls Suite2p **baseline-corrected fluorescence traces using default Suite2p parameters** its dF/F and uses this signal in all subsequent analyses. Thus neural conversion should reproduce Suite2p's baseline correction rather than use raw `F` or merely z-score it.
- Video and microscope acquisition are both 30 Hz, with microscope acquisition triggering camera frames for direct synchronization.
- Motion energy is already supplied: consecutive-frame pixel difference, squared and summed over pixels.
- For decoding, both dF/F and behavior were denoised by non-overlapping averaging in bins of 10 consecutive timestamps, yielding a 3 Hz analysis stream.
- Reference same-day decoder: ridge linear regression, nested 5-fold CV for lambda selection/evaluation; splits are consecutive 2-minute recording blocks. Cross-day models fit on one day and evaluate all others.
- The required downstream decoder differs: session-relative elapsed time is the input and motion energy must be categorical in five within-session equal-percentile bins. Sixty-second trials replace the paper's 2-minute CV blocks as explicitly required by this task.

### Curation Steps

**Neuron curation rules**:
- Suite2p `iscell` probability > 0.5.
- Track2p spatial registration/matching across consecutive days; functional analyses use neurons successfully tracked across all included days for a mouse.

**Trial curation rules**:
- The paper has continuous sessions rather than native trials. Reference decoder split consecutive 2-minute blocks; task requires constructing consecutive 60-second trials.
- Restrict processing to synchronized/common valid neural-behavior timestamps where supplied behavior is shorter.

### Decoders Trained
| Decoded variable | Accuracy |
|---|---|
| Continuous global mouse motion energy | R² reported graphically for same-day and cross-day decoding; no single exact numeric accuracy stated in text. Performance increases in the late (>P11) epoch and same-/late-to-late day prediction is stronger than early-day prediction. |

### Other Paper Findings Relevant to Validation
- The paper reports first-day detected ROI counts `[607, 1849, 2190, 1988, 1316, 2138]` and tracking proportions `[0.47, 0.20, 0.36, 0.37, 0.41, 0.19]` in its response text. These imply approximate tracked counts but do not exactly equal the redistributed curated arrays.
- Motion energy is a proxy for global spontaneous movement/arousal, not manually labeled locomotion or whisking.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Neural signal | Track2p GUI `F_processing` computes neuropil-subtracted, maximin-baseline-subtracted fluorescence | Raw `F`, `Fneu`, and `spks` supplied; ops store `neucoeff=0.7`, `baseline=maximin`, `sig_baseline=10`, `win_baseline=60`, `fs=30` | “baseline corrected fluorescence traces as our dF/F (using default Suite2p parameters)” | Reproduce the exact Track2p/Suite2p baseline-corrected signal: `Fc=F-0.7*Fneu`; Gaussian filter sigma 10 frames; 60-s minimum then maximum filters; neural=`Fc-Flow`. Do not divide by Flow because reference code does not. |
| Cell filtering | `iscell[:,1] > 0.5`; Track2p all-day matching | Every distributed row passes >0.5 and row counts/order are constant within subject | >0.5 and successfully tracked neurons used | Do not filter again except assert the criterion; use every distributed row because files are already filtered and all-day matched. |
| Cell counts | Reference tracking logic | 2,998 cells; 499.7±180.5 population SD per mouse | 526±190 per mouse; first-day proportions imply approximate counts | Modest redistribution/version difference. Use supplied curated arrays as authoritative; no principled way to add absent cells. Counts remain close to paper and all 6 mice are represented. |
| Session duration | Ops say 30 Hz; files contain 36k or 54k frames | jm031/jm032: 20 min nominal; other mice: 30 min nominal | 20 min/session | Use complete supplied valid recordings (explicit task asks full converted dataset). Preserve 20 or 30 complete 60-s trials/session; document version discrepancy. |
| Behavior timing | No behavior code in Track2p | Behavior sample counts occasionally short; timestamp local intervals reveal internal dropped camera frames; stored timestamp scale is ×1000 to seconds, with clock drift | Camera triggered by microscope for simple synchronization, both nominally 30 Hz | Preserve trigger/frame-order alignment. Detect only local gaps where interframe interval >1.5× session median, insert missing positions according to rounded interval ratio, and linearly interpolate motion there. Avoid absolute timestamp rounding, which accumulates clock drift. Trim any unmatched tail and retain complete 60-s trials. |
| Temporal denoising | No decoder preprocessing in Track2p core | 30-Hz streams | Paper averages neural and behavior in 10 consecutive timestamps | Apply synchronized non-overlapping 10-frame means to both streams, giving 3 Hz / 333.333 ms bins. |
| Trials | None | Continuous sessions | Reference CV uses consecutive 2-min blocks | Required task overrides this with consecutive 60-s trials (180 post-bin timepoints). |
| Decoder target | N/A | Continuous motion energy | Continuous ridge target, evaluated by R² | Required task overrides with five within-session equal-percentile categorical bins. Use pooled valid 3-Hz samples per session to define quantile edges. |

### Final Consistent Understanding
- Data are pre-filtered, all-day Track2p-matched layer 2/3 barrel-cortex cells. Each dated recording is one decoder session; preserving sessions (rather than merging days) respects both target format and within-session quantile selection.
- The reference “dF/F” label means baseline-subtracted fluorescence in the provided implementation, not conventional `(F-F0)/F0`; exact code behavior takes priority and is reproduced.
- Neural and video streams are frame-trigger synchronized. Local timestamp gaps require interpolation at dropped video frames, while slow camera-clock scaling must not alter frame alignment.
- After behavior repair, both streams undergo the paper's 10-frame averaging before trial splitting and target discretization.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `F.npy`, `Fneu.npy`, per-session `ops.npy` | `neural[session][trial]` | `Fc=F-0.7Fneu`; Gaussian sigma=10 frames; 60-s min then max baseline; subtract baseline; average each 10 native frames; split into 60-s matrices `(neurons,180)`; float32 | Track2p GUI `F_processing`; Suite2p preprocessing | Exact paper/reference baseline-corrected fluorescence processing. |
| Native frame index after 10-frame bins | `input[session][trial][0,:]` | Session-relative bin-center elapsed seconds `(native_start+4.5)/30`; split with neural | task specification | One time-varying input, float32; values continue across trials rather than reset. |
| `motion_energy_glob.npy`, `tstamps.npy` | `output[session][trial][0,:]` | Repair verified missing camera frames by linear interpolation; average 10 frames; compute session quantile edges at 20/40/60/80%; encode bins 0-4 with `searchsorted(..., side='right')`; split into `(1,180)` | paper videography/decoder preprocessing plus task-required discretization | Equal-percentile bins are selected independently per dated session. |
| Subject directory name | `subjects`, `subject_idx` | sorted unique IDs and integer lookup | direct | `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`. |
| Known acquisition location | `brain_regions`, `brain_region_idx` | single region `barrel cortex (L2/3)`; zero for every cell | paper methods | All recordings are same area/layer. |
| Session paths/ops/timing diagnostics | `metadata['session_info']` | record ID, subject, source/native/analysis counts, trial count, quantile edges, repair count | direct | Supports audit and spot checks. |

### Key Decisions
1. **Session definition**: Each dated recording remains a separate session. This preserves per-day neuron matrices and obeys “selected per session” target quantiles.
2. **Neural activity**: Use paper/reference baseline-corrected fluorescence, not raw `F`, conventional divided dF/F, or `spks`. Reference code explicitly returns `Fc-Flow`.
3. **Already curated cells**: Keep all rows after asserting `iscell[:,1] > 0.5`; files are already all-day matched and consistently ordered within each mouse.
4. **Behavior repair**: When behavior is shorter, use local `diff(tstamp) / median(diff)` to locate gaps >1.5 frames and insert linear values only if inferred insertions exactly equal the neural-behavior deficit. This condition holds for every short session. If lengths already match, preserve order and ignore isolated timing glitches; do not create extra samples.
5. **Common temporal processing**: Process both streams on the same native 30-Hz grid, then non-overlapping mean over 10 consecutive frames exactly as the paper, yielding 3 Hz (`time_bin_size=333.333333 ms`).
6. **Trials**: Split post-binned data into consecutive non-overlapping 60-s trials, 180 points each. All supplied lengths become exact full trials: 20 for 36k-frame sessions and 30 for 54k-frame sessions; no incomplete trial remains.
7. **Time input**: Use physical bin-center elapsed seconds from session start. It is time-varying `(1,T)` and does not reset each 60-s trial because the requested variable is elapsed time from session beginning.
8. **Quantile target**: Derive four percentile boundaries from the full repaired/3-Hz session before trial splitting. `np.quantile` plus right-sided search yields integer classes 0-4. Ties may cause slight class imbalance, an unavoidable consequence of discrete motion-energy values; preserve values rather than jitter.
9. **Output naming**: `output_names=['motion_energy_bin']`; `output_values=[['lowest (0-20%)','low (20-40%)','middle (40-60%)','high (60-80%)','highest (80-100%)']]`.
10. **Complete dataset**: Retain all 41 sessions despite the paper saying 20 min while distributed files include 30-min sessions. Excluding valid data would violate full-data conversion and lose four subjects' final 10 minutes.
11. **Metadata alignment**: Trials are consecutive windows aligned to session start; `off_start=0.0`, `off_end=60.0`, temporal alignment event is session start / consecutive 60-s window start.

### Planned Sanity Checks
- [ ] Assert every raw `F`, `Fneu`, `iscell`, and output neural array has matching neuron/frame dimensions; assert every retained `iscell` probability >0.5.
- [ ] Compare conversion baseline-corrected samples with a direct independent call/formula using SciPy filters via `np.allclose`.
- [ ] For each short behavior stream, reconstruct from raw timestamps and assert repaired length equals neural length; compare all original samples at reconstructed indices with `np.allclose`.
- [ ] Independently average a selected raw 10-frame neural and motion segment and compare with converted trial/timepoint using `np.allclose`.
- [ ] Assert each trial has neural `(N,180)`, input `(1,180)`, output `(1,180)`, finite values, and exactly aligned counts.
- [ ] Assert session time input is strictly increasing at 1/3 s and first/last trial boundaries have no off-by-one overlap/gap.
- [ ] Check output values are integer 0-4 and report per-session/global class fractions; expect near 20% each except ties.
- [ ] Compare subject/session/cell/trial totals against data and paper: 6 subjects, 41 sessions, 2,998 unique tracked cells, 20/30 trials per session, 1,060 trials total.
- [ ] Visualize raw/repaired/binned motion, quantile edges/class trace, raw neuropil-corrected/baseline/neural traces, and final aligned trial examples for up to two sessions.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with required positional output and `--full` (default), `--sample`, and `--show-processing` options. The script:
- discovers all dated sessions deterministically;
- validates curated cells and source dimensions;
- reproduces reference neuropil and maximin-baseline subtraction;
- repairs only timestamp-verified missing video frames;
- averages synchronized streams by 10 frames;
- constructs session-elapsed-time input and within-session five-quantile motion labels;
- creates contiguous 60-second trials and complete metadata;
- emits per-session timing/statistics and strict assertions;
- saves up to two multi-panel processing plots showing every major transform.

Code inefficiencies identified:
- Full float32 neural arrays plus intermediate baseline arrays can be large.
- Repeated per-neuron filtering would add Python overhead.
- Loading whole raw datasets eagerly would exceed useful memory.

Code speedups added:
- Memory-map `F`/`Fneu` and process vectorized 64-neuron chunks.
- Apply SciPy filters across each entire chunk and vectorized reshape/mean binning.
- Discard native-rate intermediates after each session; retain only 3-Hz float32 products.
- Process/write one final pickle only, with highest pickle protocol.
- Syntax compilation and CLI help completed successfully.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (session-neuron total) | 442 |
| Neurons / session | 221, 221 |
| Subjects | 1 (`jm031`) |
| Sessions / subject | 2 |
| Trials (total) | 40 |
| Trials / session | 20, 20 |
| Trial shape | neural `(221,180)`, input `(1,180)`, output `(1,180)` |
| Time input range | `[0.15, 1199.8167]` seconds |
| Neural ranges | session 1 `[-45.11,824.29]`; session 2 `[-47.94,1144.54]` |
| Motion class distribution | `[0.200,0.200,0.200,0.200,0.200]` in each session |

### Processing Plots Review
- Created `processing_jm031_2023-10-18_a.png` and `processing_jm031_2023-10-19_a.png` (14×15-inch multi-panel figures, 735-782 KiB).
- Panels cover raw F/neuropil subtraction, maximin baseline and subtraction, 10-frame neural averaging, observed/repaired behavior alignment, and binned motion/quantile edges/classes.
- Both sample sessions required no behavior repair; observed and aligned traces overlap as expected.
- Plots have valid image dimensions/dynamic range and no detected empty/corrupt output. Numerical alignment checks show exactly 3,600 synchronized bins/session.

### Format Validation
- `/app/verification_sample_out.txt`: “Data format is valid, no errors or warnings.”
- T=180 for every trial; one input and one output; class values exactly 0-4; brain-region and subject dimensions consistent.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| Chunked/vectorized filtering and memory mapping | Avoids full native-array copies and per-neuron Python loops |
| 10-frame reduction before trial storage | 10× smaller temporal products |

| Step | Time / Session | Estimated Total Time |
|---|---:|---:|
| Sample conversion including plots | 0.90-1.14 s | ~42 s for 41 sessions (likely modestly longer for 54k-frame sessions) |
| Pickle serialization | included; total 2.05 s for 2 sessions | comfortably <15 min |

Files created and inspected: `/app/sample_data.pkl`, `/app/conversion_sample_out.txt`, `/app/verification_sample_out.txt`, and two processing PNGs.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|----------------------:|------------------------:|
| motion_energy_bin | 0.5388 | 0.2967 |

- Chance balanced accuracy is 0.2000; validation is 1.48× chance and therefore above chance.
- Training completed all 200 epochs successfully. Loss decreased from 32.336 at epoch 1 to 1.267 at epoch 200; test loss was 2.305.
- The train-validation gap (1.82×) indicates some overfitting in the very small two-session sample and will be reassessed using all 41 sessions.
- Required output log created: `/app/train_decoder_sample_out.txt`.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 395.3 MiB
- `conversion_full_out.txt`: created
- `verification_full_out.txt`: created; validator reports “Data format is valid, no errors or warnings.”

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total unique tracked neurons | 526±190/mouse | all-day Track2p rows | 2,998 across 6 mice | 2,998 unique row identities | Yes to supplied data; close to paper |
| Mean neurons/session | same tracked set per mouse | fixed rows across days | 498.66 weighted by session | 498.66 | Yes |
| Subjects | 6 | N/A | 6 | 6 | Yes |
| Sessions | ≥6/mouse | consecutive-day tracking | 41 (6-7/mouse) | 41 | Yes |
| Trials (total) | no native trials | no native trials | continuous recordings | 1,090 constructed 60-s trials | Task-defined; correct |
| Trials/session | paper uses 2-min CV blocks | N/A | 20- or 30-min sessions | 20 or 30 | Yes to full supplied durations |
| Time input range | N/A | N/A | nominal 0-1200/1800 s | bin centers 0.15-1199.82/1799.82 s | Yes |
| Motion class distribution | continuous target | N/A | continuous motion | `[0.200,0.200,0.200,0.200,0.200]` | Yes, task-required quantiles |
| Native/analysis bin | 30 Hz; average 10 | exact baseline implementation | 30 Hz | 3 Hz / 333.333 ms | Yes |

### Full Conversion Statistics
- Conversion completed in 40.67 s, well below the 15-minute threshold.
- 41 sessions, 1,090 trials, 20,445 session-neuron instances, trial T=180 throughout.
- Subject/session/trial/neuron summary: jm031 7/140/221; jm032 7/140/370; jm038 7/210/685; jm039 7/210/746; jm040 6/180/541; jm046 7/210/435.
- The prior planning estimates of 1,060 trials and 20,198 session-neuron instances were arithmetic mistakes, corrected here. Direct formulas are `14×20 + 27×30 = 1,090` trials and summing per-session neuron counts gives 20,445.
- Behavior repair inserted 264 samples in jm031/2023-10-22 and jm032/2023-10-22 combined, plus 12 samples over seven other short sessions (276 total); all repaired arrays exactly matched neural lengths. Equal-length jm046 timestamp pauses were correctly left untouched.
- Spot checks at sessions 0, 13, 14, 20, 28, 34, and 40 confirmed finite `(N,180)` trials, monotonic 1/3-s input, classes 0-4, and exact per-session class balance.
- Paper/data differences are limited to redistributed release version (tracked count 499.7±180.5 population SD vs paper 526±190) and valid 30-min files for four subjects vs textual 20 min. No available data were discarded.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Read all of `/app/verification_full_out.txt` and searched case-insensitively for error/warning/invalid/failed. The only match is the affirmative first line: “Data format is valid, no errors or warnings.” No actionable warnings exist.
2. **Independent neural raw-file sanity checks (`np.allclose`)**: `/app/sanity_checks.py` directly loads raw `F`, `Fneu`, and `ops` without importing conversion code. It independently recomputed full-neuron reference filtering and selected converted values for:
   - jm031/2023-10-18 neuron 3, bin 10: 10.9747677 = 10.9747677;
   - jm038/2023-05-04 neuron 17, bin 2345: 28.2684517 = 28.2684517;
   - jm046/2024-09-09 neuron 100, bin 5399: 15.1149340 = 15.1149340.
   All pass `np.allclose(rtol=1e-5, atol=1e-5)`.
3. **Independent input sanity checks (`np.allclose`)**: Analytically generated `(10*k+4.5)/30` bin-center seconds and compared complete session inputs in sessions 0, 14, and 40 (`atol=1e-6`). All pass. Every tested trial boundary advances one 1/3-s sample without overlap/gap.
4. **Independent output sanity check (`np.allclose`)**: Directly loaded raw behavior/timestamps for jm031/2023-10-22, independently inferred 116 missing frames, interpolated onto frame indices, averaged each 10 frames, computed quantiles `[630157.08,655469.88,744150.96,2067514.34]`, and generated labels. The entire 3,600-label stream and five explicit trial/timepoint samples exactly match converted output (`rtol=atol=0`). Observed raw behavior values are unchanged at reconstructed indices.
5. **All-session invariants**: For all 41 sessions, source `F/Fneu/iscell` dimensions match; every `iscell` probability is >0.5; converted trial count follows native frames; every neural trial is finite `(N,180)`; input/output are `(1,180)`; input is increasing; classes are exactly `{0,1,2,3,4}`; class counts are exactly equal.
6. **Key statistics**: Recomputed 6 subjects, 41 sessions, 2,998 unique subject-level tracked cells, 20,445 session-neuron instances, 1,090 trials, range 221-746 neurons/session, and global class counts `[39240]*5`. These match source-derived and validator statistics. Paper-version differences remain fully explained in Steps 4/9.

### Reference Code Comparison
| Major step | Conversion implementation | Reference implementation/method | Comparison |
|---|---|---|---|
| (a) Loading | Memory-mapped `F.npy`, `Fneu.npy`, `iscell.npy`; `ops.npy` dict; behavior NPY arrays | Track2p loaders/Suite2p files and supplied notebook use these arrays | Same source variables; conversion additionally requires Fneu for exact reference processing |
| (b) Neuron/trial filtering | Assert all distributed `iscell[:,1]>0.5`; retain all pre-matched rows; retain complete 60-s windows | Track2p default strict >0.5 and all-day match matrix; paper has continuous sessions | Same neuron rule/curated rows; task-required 60-s trials |
| (c) Temporal alignment | Trigger-order alignment; timestamp-local repair only for exact frame deficits | Paper: microscope acquisition triggers 30-Hz camera | Consistent; repairs documented dropped camera samples without drift-induced remapping |
| (d) Binning | Non-overlapping mean of 10 native frames for both neural and motion | Paper: averages both dF/F and behavior in bins of 10 timestamps | Exact match |
| (e) Input construction | Session-relative elapsed bin-center seconds | New decoder specification | Intentional task-specific addition; no analogous paper decoder input |
| (f) Output construction | Supplied global motion energy, repaired/binned, then five within-session percentile classes | Paper uses same continuous global motion energy; task requires five equal-percentile bins | Reference signal preserved; only mandated categorical transformation differs |
| Neural preprocessing | `F-0.7Fneu`; Gaussian sigma 10; 60-s min/max baseline; subtract | Track2p GUI `F_processing` lines 185-211 / Suite2p defaults | Exact logic and ops values; intentionally no division by baseline |

### Edge Cases Reviewed
- Behavior deficits of 1-148 samples are internal; all nine short sessions' timestamp-inferred insertions exactly equal deficits and reconstructed lengths equal neural lengths.
- Three equal-length jm046 sessions contain timestamp pauses. Inserting there would create excess behavior; conservative frame-order preservation correctly leaves them unchanged.
- Trial/session endpoints: all native frame counts divide exactly by 10×180, so no partial windows are dropped. First input is 0.15 s (10-frame bin center), last is 1199.8167 or 1799.8167 s; there is no off-by-one endpoint loss.
- Float32 elapsed time has maximum adjacent-step error ~8.14e-5 s from representation only; values remain strictly increasing and analytically match float32 expectations.
- Quantile ties were checked via class counts; this release yields exact 20% classes in every session.

### Issues Found and Resolved
- **Documentation arithmetic only**: Earlier planning listed 1,060 trials/20,198 session-neuron instances; corrected to 1,090/20,445 from direct source-derived sums.
- **Repair-count wording only**: Corrected to 264 insertions in the two largest-drop sessions plus 12 in seven other sessions = 276 total.
- No conversion mismatch or data error was found; no reconversion was required. Independent check log: `/app/cache/sanity_checks_out.txt`.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. Epoch 1 loss 70.8989; epoch 100 loss 1.5181; epoch 200 loss 1.0454.
- Test loss: 2.7349.
- Training used CUDA, all 200 epochs, 872 training trials and 218 held-out trials, and finished successfully.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|----------------------:|------------------------:|-------|
| motion_energy_bin | 0.6389 | 0.3118 | Chance 0.2000; validation 1.559× chance |

- Full log: `/app/train_decoder_full_out.txt`.
- Sample plotting was requested and training completed without memory failure.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Achieved Validation Balanced Accuracy | Chance | Ratio to Chance | Expectation from Paper |
|---|---:|---:|---:|---|
| motion_energy_bin | 0.3118 | 0.2000 | 1.559× | Paper predicts **continuous** motion energy with ridge regression and reports R² graphically, not five-class balanced accuracy; no numeric like-for-like benchmark exists. |

#### Check 1: Accuracy vs chance
- Validation balanced accuracy is above chance and above the requested 1.5×-chance investigation threshold.
- Training loss decreases smoothly and validation prediction is meaningfully above chance, supporting correct neural-behavior alignment.

#### Check 2: Accuracy comparison to paper
- Every paper decoding claim/metric found was reviewed. The paper uses continuous motion-energy ridge regression and R² matrices for same-day/cross-day prediction; exact values are in figures rather than text.
- The present task mandates five session-specific percentile classes and the supplied nonlinear neural classifier reports balanced accuracy. Therefore an exact numeric comparison is not valid. Qualitatively both analyses show that motion is decodable from population activity.
- This distinction is not used to dismiss low performance: raw labels, timing, filtering, and three explicit trials were independently rechecked below.

#### Check 3: Train-validation gap
- Training balanced accuracy 0.6389 / validation 0.3118 = 2.05×, exceeding 1.5×.
- Investigation found no data leakage: train/test are disjoint whole trials (872/218), outputs are derived only from behavior, session quantile edges never use neural activity, and no output samples are copied between trials. Each source frame belongs to exactly one non-overlapping trial.
- The gap is consistent with model capacity and heterogeneous sessions/animals: 221-746 neural features per session, only 16-24 training trials/session, and session-specific neural scales/motion thresholds. The sample also showed overfitting, while full validation improved from 0.2967 to 0.3118.
- Changing neural processing or labels to optimize validation would violate the reference/task mapping. No conversion fix is justified.

### Required Low-Accuracy Debugging Checks
1. **Three raw trials**: Independently regenerated entire labels from original behavior files for jm031/2023-10-18 trial 5, repaired jm031/2023-10-22 trial 10, and jm046/2024-09-09 trial 29. All labels exactly match (`np.allclose`, zero tolerance). Trials contain 28-84 class transitions, demonstrating time variation.
2. **Temporal alignment**: Frame-trigger order is preserved; repaired observed values are identical at retained indices; neural and motion use identical 10-frame boundaries. Processing plots overlay observed/repaired motion and display aligned categorical traces. Independent input/trial-boundary checks found no gap or overlap.
3. **Output variation**: Every session is exactly 20% in each class overall; individual trials have varying mixtures as expected for behavioral states, rather than artificially forcing trial-level balance.
4. **Neural filtering**: Every raw distributed row passes strict `iscell probability >0.5`; all-day tracked row identity/order is retained; exact Track2p/Suite2p baseline correction was independently matched.
5. **Reference match**: Neuropil subtraction, maximin baseline, 10-frame means, and motion-energy signal follow paper/code. Only required changes are 60-s trials, elapsed-time input, and categorical quantiles.

### Issues Found and Resolved
- No conversion issue was found in this review. The train-validation gap is documented as model overfitting/generalization under the fixed reference trainer, not leakage or misalignment.
- All Step 10 independent checks remain passing after training; no reconversion/retraining iteration was necessary.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created with dataset description, loading example, schema, processing, statistics, and reproduction commands.
- [x] `cache/` folder created.
- [x] Investigation script and output moved to `cache/`; `cache/README_CACHE.md` documents them.
- [x] Required conversion, validation, decoder logs, data pickles, and processing/training plots retained at top level for user inspection.
- [x] `CONVERSION_NOTES.md` reviewed and completed through all workflow steps.
