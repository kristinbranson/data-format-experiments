# Dataset Conversion Notes

## Overview
- **Dataset**: Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping
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

Python 3.13.15, NumPy 2.4.4, and PyTorch 2.6.0+cu124 imported successfully. The required checkpoint `ls -la /app/CONVERSION_NOTES.md` succeeded.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_dat` | `code/georepca1/src/utils.py:61` | LOADING | Loads one animal from MATLAB or joblib; converts `envs`, `position`, and `trace` to NumPy for MATLAB input. |
| `generate_behav_dict` | `code/georepca1/src/utils.py:130` | LOADING | Extracts positions, environment labels, and map shapes into a lightweight dictionary. |
| `get_env_mat` | `code/georepca1/src/utils.py:215` | PROCESSING | Maps each environment name to a binary 3x3 occupancy/geometry matrix (1 present, 0 blocked). |
| `get_rate_maps` | `code/georepca1/src/utils.py:313` | PROCESSING | Bins position into 15x15, accumulates event traces, smooths with sigma 1.5 bins, divides by occupancy, and converts to event rate at 30 Hz. |
| `get_split_half` / `get_place_cells` | `code/georepca1/src/utils.py:356,415` | CURATION | Measures within-session spatial reliability; place cells are above a circular-shuffle null at p<0.05. |
| `get_shr_within` | `code/georepca1/src/utils.py:442` | CURATION | Applies split-half place-cell classification independently to every recording day. |
| `decode_position_within` | `code/georepca1/src/utils.py:1845` | PROCESSING/CURATION | 5-fold within-day Gaussian naive Bayes decoder; position is 15x15, samples require speed >5 cm/s, and cells require >5 events in moving samples. |

### Notes
- The native neural stream is `trace`, a rise-extracted binary calcium-event series sampled at 30 Hz. Delta-F/F has already been processed upstream; it must not be recomputed.
- Arrays are stored per animal with days/sessions along axis 0: author calls use `position[day].T` (time x 2) and `trace[day].T` (time x registered cells).
- A registered cell absent on a day is represented by an all-NaN trace/map for that day. Day-specific analyses implicitly exclude such cells; explicit curation sets p-values to NaN based on `trace[0, :]` after transposition.
- The provided `maps` are already derived from the same position/trace streams. They are reference products for checks, not decoder inputs.
- Reference position decoding filters frames for speed and cells for event count. That filter will be reconciled with the requested continuous, one-minute-trial decoder in Steps 4-5; blindly dropping frames would destroy requested temporal continuity.
- README defines seven animals and fields: `SFPs`, `blocked`, `centroids`, `envs`, `maps`, `position`, and `trace`.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Seven extensionless compressed joblib files (`QLAK-CA1-*`) and duplicate MATLAB `.mat` files contain the primary data, one file per mouse. `behav_dict` is a reference-code-derived lightweight aggregate. `precomputed_results/` contains paper analysis products, including split-half reliability and within-day decoding.
- Every joblib object is `{animal_id: payload}`. Payload fields are: `SFPs` (35x35 x globally registered cells x days, float64), `blocked` (list of blocked 3x3 partition indices), `centroids` (global cells x 2 x days), `envs` (days x 1 strings), `maps` (`sampling`, `smoothed`, `unsmoothed` at 15x15 spatial resolution), `position` (days x 2 x frames, float64), and `trace` (days x global cells x frames, float64).
- Finite trace values are exactly `{0,1}`. Missing day-specific cell registrations are all-NaN traces. Position is finite for every stored frame; trace/position frame axes are exactly aligned and have no intermittent missing samples.
- Each recording lasts approximately 40 minutes (71,866-72,219 frames at 30 Hz). Six animals have 31 days (three complete 10-geometry sequences plus a final square), while QLAK-CA1-51 has 21 days (two sequences plus square).
- No separate README is present under `data/`; field semantics therefore come from the code README and are cross-checked against array contents here.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 5,413 unique within-animal CellReg identities; 69,744 day-specific registered neuron instances |
| Neurons / session | 113-564; mean 336.93; median 344 |
| Subjects | 7 (`QLAK-CA1-08`, `-30`, `-50`, `-51`, `-56`, `-74`, `-75`) |
| Sessions / subject | 31, 31, 31, 21, 31, 31, 31 (207 total) |
| Trials (total) | Not native: sessions are continuous recordings; 8,187 complete 1-min windows after floor-splitting |
| Trials / session | Not native; 39 trials for 93 sessions and 40 trials for 114 sessions |

Per-animal global registered identities: 515, 875, 942, 554, 862, 713, 952. Total frames across sessions: 14,903,019. There are only 45 day-specific registered neuron instances with zero events over a full session; this is a potential activity-curation edge case.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 5,413 unique; 69,744 rate maps/session-neuron instances | Methods: “5,413 unique neurons across 207 sessions in 10 geometries, forming 69,744 rate maps.” | 
| Neurons / session | 336.93 inferred (69,744/207); paper also reports 773 ± 68 SE unique cells/animal, range 515-952 | Results: “mean number of cells per animal = 773 ± 68 SE, minimum ... 515, maximum ... 952.” |
| Subjects | 7 | Seven animal traces/points are shown and the data/code enumerate seven IDs. |
| Sessions / subject | Up to 31; 207 total | Figure 1D: sequence repeated “for up to three total repetitions (31 days).” |
| Trials (total) | N/A natively | Sessions are continuous; one-minute trials are required by this conversion rather than the study. |
| Trials / session | N/A natively | “All sessions were 40 min.” |
| Neural data time bin | 1/30 s native | “simultaneously acquired behavioral and cellular imaging streams at 30 Hz.” |
| Behavior data time bin | 1/30 s native | Same quote; frames were timestamped for post-hoc alignment. |
| Reward rate | N/A | Free exploration with no reward outcome described. |
| Environment size | 75 x 75 cm, imagined 3x3 partitions | Methods and Figure 1A. |
| Geometries | 10 | Overview/results: 207 sessions in 10 geometries. |


### Processing Details
- Calcium and overhead behavior were simultaneously acquired at 30 Hz and timestamp-aligned. The supplied position and event vectors therefore require no lag correction.
- Upstream calcium processing: motion correction, cell segmentation, calcium-trace derivative, Gaussian smoothing (sigma 5 frames), noise-based z-scoring, threshold at z>2.5, then binarization. The supplied binary rising-phase vector is treated as firing rate in all analyses.
- Position came from DeepLabCut head tracking. Cells were longitudinally tracked using brain landmarks, spatial footprints, and/or centroids.
- Reference maps use 5x5 cm spatial pixels (15x15 over 75 cm) and a 5 cm isotropic Gaussian smoothing scale. The code implementation uses sigma 1.5 bins for some regenerated maps, but supplied maps are authoritative precomputed products.
- Paper position decoder: within-session 5-fold Gaussian naive Bayes, flat position prior, binned position one-hot target, Euclidean error on withheld samples. Figure 1F shows mean error declining approximately from 22 cm early to 11 cm late (read from graph; no exact table is printed).

### Curation Steps

**Neuron curation rules**:
Motion correction was manually inspected; spatial footprints were manually verified to remove lens artifacts. Place cells exceed the 99th percentile of 1,000 position-shuffle split-half correlations (p<0.01), but the paper explicitly says high population reliability “motivated the inclusion of all cells in subsequent analyses.” Therefore place-cell-only filtering is not the general reference rule.

**Trial curation rules**:
No event trials and no trial rejection are described. Each daily session is 40 min of free exploration. Target one-minute slicing is downstream-specific.

### Decoders Trained
| Decoded variable | Accuracy |
| 15x15 animal position (within session) | Euclidean error approximately 22 cm on early days, improving to approximately 11 cm on later days (Figure 1F); categorical accuracy is not reported. |
| Animal identity from RSM | Not above shuffled control (not comparable to this task). |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Dataset size | Seven fixed IDs; day-wise registered cells inferred from non-NaN traces | 5,413 global identities, 207 sessions, 69,744 day-cell instances | Exactly 5,413 / 207 / 69,744 | Exact agreement. |
| Session duration | Assumes 30 Hz | 71,866-72,219 frames = 39.93-40.12 min | All sessions 40 min | Small timestamp/alignment variation is expected; retain frames but use only complete fixed one-minute windows. |
| Stream alignment | Author passes matching day/time slices of `position` and `trace` | Identical time axes; no position NaNs/intermittent trace NaNs | Simultaneous timestamped 30 Hz acquisition | No resynchronization or interpolation needed. |
| Neural representation | `trace` loaded directly; maps and decoder consume it as events | Finite values exactly 0/1 | Binary rising-phase vector treated as firing rate | Use supplied `trace`; do not compute dF/F or deconvolve. |
| Neuron inclusion | Within-session decoder uses all registered cells with >5 events during moving frames; later RSA uses all cells | 45 registered day-cell instances have zero total events | Paper motivates inclusion of all cells, not just place cells | Exclude only absent all-NaN registrations and zero-event cells. The latter carry no decodable information and fail the reference decoder’s >5-event rule; a >5 threshold is considered in Step 5. |
| Environment sequence | `main.py` lists canonical environments, but analyses operate on each animal’s stored order | Each animal has a different randomized 10-shape order, repeated exactly across sequences | Randomized across mice and repeated within mouse | Preserve each stored session/day order. |
| Geometry orientation | `get_env_mat(env)` supplies one canonical shape orientation | `blocked` differs from `get_env_mat` for `t`, `l`, `bit donut`, and `glenn`; native flattening is `(y,x)`, while `position`/decoder classes are `(x,y)` | Inserted walls create specific daily geometries | Use native `blocked` as authoritative, then transpose its 3x3 matrix into the x-first position frame. This gives only 152/4,912,200 samples (0.0031%) in blocked bins, versus 17.14% without transpose. |
| Place-cell threshold | `get_place_cells` default alpha=0.05, while callers retain continuous p-values and plots apply thresholds | Precomputed shuffle p-values available | Methods define place cells at p<0.01 | No place-cell-only selection: paper explicitly includes all cells later. Thus threshold discrepancy does not affect conversion. |
| Reference position performance | Precomputed results are 15x15, moving-frame, GaussianNB Euclidean errors | Overall fold-error mean 13.49 cm (range 5.76-32.46); first/last day values improve | Figure 1F visually declines from roughly 22 to 11 cm | Exact precomputed results agree with plotted scale and trend. They are context, not directly comparable to requested 3x3 categorical balanced accuracy. |

All source descriptions now agree after distinguishing (1) unique CellReg identities from day-specific neurons and (2) environment names from exact blocked-partition orientations.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `trace[day, registered_cells, :]` | `neural[session][trial]` | Select cells finite on that day; Gaussian smooth along native time with sigma=3 frames, average non-overlapping groups of 3 frames; cast float32; slice 600-bin one-minute trials | `fit_decoder` / `test_decoder` | Matches reference decoder’s 3-frame (100 ms) smoothing/binning. Neural shape is neuron x 600. |
| `blocked[day]` | `input[session][trial]` | Fill native row-major `(y,x)` matrix, transpose to decoder `(x,y)` order, 1=blocked and 0=accessible; repeat as a static per-trial vector | Native field; contrasted with `get_env_mat` | Native field preserves orientation; transpose aligns class `x*3+y`. Input names are `blocked_x{x}_y{y}`. |
| `position[day, :, :]` | `output[session][trial]` | Average each non-overlapping 3-frame group; clip coordinates to [0,75); floor-divide by 25 cm; class=`x_bin*3+y_bin`; slice as shape (1,600), int64 | `fit_decoder` position pooling and row-major one-hot construction | Nine labels ordered `x0_y0` through `x2_y2`. Exact 75 cm samples clip into bin 2. |
| animal ID | `subjects`, `subject_idx` | Seven IDs in reference order; one subject index per day-session | `main.py` animal list | Sessions remain animal-major, then day order. |
| CA1 recording | `brain_regions`, `brain_region_idx` | `brain_regions=['CA1']`; zero index for every retained neuron | Paper/README | All recordings are dorsal CA1. |

### Key Decisions
1. **Session and trial definitions**: Each recording day is one target session. Consecutive non-overlapping 1,800-frame windows are exactly 60 s and become trials. Only the short incomplete tail is discarded. This yields 8,187 trials: 39 in each 71,866-frame session and 40 in sessions at least 72,000 frames long.
2. **100 ms bins**: Use the reference position decoder’s `temporal_bin_size=3` at 30 Hz. Gaussian smoothing (sigma 3 native frames) followed by 3-frame average pooling is applied to traces; position is mean-pooled over the identical raw frames. This preserves reference processing and makes every trial 600 timepoints.
3. **No place-cell or velocity-frame filtering**: All manually curated, day-registered cells are retained, matching the paper’s stated inclusion of all cells and preserving the reported 69,744 rate maps. The reference’s >5-moving-event cell and >5 cm/s frame masks are classifier-specific optimizations. Dropping frames would violate contiguous one-minute trial timing; omitting low-activity neurons does not improve information preservation. The 45 zero-event session-neurons are retained and documented.
4. **Geometry semantics**: Encode blocked locations from `blocked`, with 1 meaning blocked. Native indices flatten `(y,x)` while the position code and reference map accumulation index `(x,y)`, so transpose the 3x3 geometry before x-first flattening. Testing all eight grid symmetries made this unambiguous: transpose reduced position/block overlap from 17.14% to 0.0031%; the residual 152 boundary samples are tracking/bin-edge noise.
5. **Spatial class convention**: Use the reference’s x-first NumPy row-major flattening. Bins cover [0,25), [25,50), and [50,75] cm on each axis. This is the requested coarse 3x3 analogue of the reference 15x15 decoder.
6. **Alignment and tails**: Neural and behavioral samples already align one-to-one. Temporal binning begins at session frame 0. Smoothing is performed continuously over the session before window slicing, matching the physical continuous recording; metadata records discarded tail frames.
7. **Dtypes/metadata**: Neural and input are float32, output is int64. Metadata records 100 ms bins, 60 s windows aligned to each minute-window start (`off_start=0`, `off_end=60`), 30 Hz source sampling, smoothing/pooling, class convention, and per-session provenance.

### Planned Sanity Checks
- [ ] Structural: 207 sessions, 8,187 trials, 7 subjects, 69,744 total session-neurons; all trials have 600 aligned time bins and no NaN/Inf.
- [ ] Neural raw comparison: independently load a raw file and `np.allclose` a selected converted neuron/time bin to the Gaussian-smoothed mean of its exact three raw frames.
- [ ] Input raw comparison: independently reconstruct a nine-bit native `blocked` vector and `np.allclose` it to selected converted trials.
- [ ] Output raw comparison: independently mean three raw x-y samples, compute the clipped 25-cm row-major class, and `np.allclose` it to converted output.
- [ ] Geometry: for every session, occupied position classes must never correspond to a native blocked partition (allowing investigation of boundary-tracking noise rather than silently changing labels).
- [ ] Reference products: regenerated event-rate-map/occupancy spot checks and precomputed decoding trend will be used to validate stream orientation and alignment.
- [ ] Distribution: all nine classes must occur globally; per-class frequencies, geometry-bit frequencies, event ranges, neuron counts, and discarded-tail sizes will be reported.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with the required CLI modes, source-aligned smoothing/pooling, complete-minute slicing, geometry/output construction, per-session provenance, shape/value assertions, timing, and six-panel processing plots. Syntax compilation and `--help` execution succeeded.

Code inefficiencies identified:
Native float64 animal arrays are large; making both float64 source and separate smoothed copies would increase peak memory. Repeated trial-by-trial filtering would also repeat Gaussian work.

Code speedups added:
Each animal is loaded once; each selected session is cast once to float32; SciPy smoothing overwrites that working buffer via `output=`; vectorized reshape/mean performs pooling; plots are capped at two sessions; each animal payload is released after its sessions.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 338 session-neuron instances |
| Neurons / session | 185, 153 (mean 169) |
| Subjects | 7 IDs retained in schema; sample sessions both belong to QLAK-CA1-08 |
| Sessions / subject | QLAK-CA1-08: 2 sample sessions |
| Trials (total) | 78 |
| Trials / session | 39, 39 |
| Input blocked-bit range | [0, 1] (day 0 open square; day 1 center blocked) |
| Neural range | [0, 1], continuous float32 after smoothing/pooling |
| Position class distribution | [0.103526, 0.102308, 0.103697, 0.061603, 0.043761, 0.117222, 0.134893, 0.103782, 0.229209] |
| Samples in blocked partitions | 0 / 46,800 |

### Processing Plots Review
Both six-panel plots were inspected. Raw trajectories respect the open-square and center-blocked geometries; raw binary events become localized smooth 100 ms responses; pooled x/y traces overlay raw traces without lag; 3x3 classes change at the expected 25/50 cm crossings; and the static geometry matrices match the native block. No anomaly or temporal misalignment was visible.

Manual inspection confirmed shapes `(n_neurons,600)`, `(9,)`, `(1,600)` and dtypes float32/float32/int64. Independent raw-file spot checks gave `np.allclose=True` for neural smoothing/pooling, blocked input, and output discretization. Metadata contains complete processing and per-session provenance. Verification reported “valid, no errors or warnings.”

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| One animal load, in-place float32 smoothing, vectorized pooling | Processing itself was 1.40 s for two sessions; loading was 8.88 s. |

| Step | Time / Session | Estimated Total Time |
| Native load (amortized) | ~4.44 s in sample; ~0.5-0.7 s/day when amortized over full animal | ~100-120 s based on measured seven-animal exploration loads |
| Session conversion + optional plot | 0.70 s with plots; expected lower without plots | ~90-145 s |
| Save | 0.03 s for 31 MiB | ~7-20 s for expected ~3.2 GiB |
| Full total estimate | N/A | ~3-5 min, comfortably below 15 min |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| position_3x3 | 0.4429 | 0.3783 |

Chance is 1/9 = 0.1111. Validation is 3.40x chance. In the final post-fix regression run, loss decreased monotonically from 2.278805 at epoch 1 to 1.719163 at epoch 200; held-out loss was 1.806126. Training finished successfully on CUDA. (The symmetric square/center-blocked sample geometry is numerically unchanged by transpose; small score differences across reruns reflect randomized model initialization.)

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 6.205 GiB
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total session-neurons | 69,744 rate maps | finite day registrations | 69,744 | 69,744 | Yes |
| Mean neurons/session | 69,744/207 = 336.93 | day-specific finite traces | 336.93 | 336.93 | Yes |
| Subjects | 7 | 7 IDs | 7 files | 7 | Yes |
| Sessions | 207 | all animal days | 207 | 207 | Yes |
| Trials (total) | N/A (continuous sessions) | N/A | 8,187 complete native minutes | 8,187 | Yes, target-derived |
| Trials/session (mean) | 40-min sessions | N/A | 39 for 93 sessions, 40 for 114 | 39.55 (same split) | Yes |
| Geometry-bit range | N/A | binary matrices | [0,1] | [0,1] | Yes |
| Geometry/position conflict | N/A | maps index `(x,y)` | Best grid transform is transpose; 152/4,912,200 | 0.003094% | Yes |
| Position class range | N/A | requested 3x3 | [0,8] | [0,8] | Yes |
| Position distribution | N/A | N/A | [0.099663,0.075412,0.115696,0.098495,0.057266,0.141213,0.135121,0.076848,0.200287] | identical | Yes |

Full conversion took 160.21 s plus 5.39 s to save, below the 3-5 minute estimate. The first run exposed a native `(y,x)` versus decoder `(x,y)` geometry convention: naive flattening yielded 17.14% impossible overlap. Testing all eight grid transforms identified transpose (0.003094%). The script was corrected, sample conversion/verification/training were rerun, and full conversion/verification were rerun. Final verification states “valid, no errors or warnings.”

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Searched the complete final verification log for error, warning, invalid, NaN, and Inf. The only match is the affirmative first line: “Data format is valid, no errors or warnings.” There are no warnings requiring waiver.
2. **Independent raw `np.allclose` checks**: `/app/sanity_checks.py` deliberately does not import conversion code. It loads original joblib files for converted sessions 0, 65, and 206 (three animals, early/middle/final days and trials), independently smooths a nonzero raw trace, pools its exact three frames, reconstructs transposed geometry, and computes the raw-position class. Neural, input, and output all pass `np.allclose` in all three cases. Explicit frames 1797:1800 versus 1800:1803 confirm the first trial boundary has no overlap or gap. Full totals and ranges also pass.
3. **Reference occupancy orientation check**: Independently accumulated QLAK-CA1-08 day-0 position into 15x15 bins and compared with supplied `maps.sampling`. Direct `(x,y)` orientation has correlation 0.99991, versus 0.52980 after transpose, confirming source position order. Small max occupancy difference (0.67 s; total differs by one 30 Hz frame) reflects legacy edge/bin convention and does not affect the native time-series conversion.
4. **Reference-code comparison**:

| Stage | Reference implementation | Conversion implementation | Assessment |
|---|---|---|---|
| Loading | `load_dat` uses joblib nested by animal | Same joblib payload and animal list | Match |
| Neuron/trial filtering | Missing registrations are NaN; paper includes all curated cells; reference classifier additionally masks low-moving-event cells/frames | Finite day registrations retained; no place-cell/activity mask; complete 60 s windows retained | Paper-consistent; classifier-specific speed mask omitted to preserve contiguous required trials |
| Temporal alignment | Same-index 30 Hz `position` and `trace`; timestamp-aligned acquisition | Same raw indices, no interpolation/lag | Match |
| Temporal binning | `fit_decoder`/`test_decoder`: Gaussian sigma=3 then 3-frame AvgPool; position 3-frame AvgPool | Same sigma, SciPy default boundary mode, and non-overlapping 3-frame mean; complete session smoothed before slicing | Match |
| Input construction | No paper decoder geometry input; `get_env_mat` loses some orientation | Native `blocked` matrix, transposed `(y,x)->(x,y)`, binary static vector | Required downstream addition; empirically aligned to position |
| Output construction | Divide position into 15 bins/axis, mean-pool 3 frames, integer spatial class | Requested 3 bins/axis, identical mean-pool, class `x*3+y` | Match except required coarser resolution |

5. **Key statistics**: Seven animals, 207 sessions, 5,413 global CellReg identities, 69,744 session-neurons, 113-564 neurons/session, and mean 336.93 all exactly match paper/data. Unique cells/animal average 773.29 and range 515-952 match paper’s 773 ± 68 SE and range. All nine outputs occur; fractions are recorded in Step 9. Source has no reward statistic. Reference 15x15 decoder fold-error mean is 13.49 cm and follows the paper’s improving trend; this is not numerically equated to 3x3 class accuracy.
6. **Edge cases**: Verified 93 sessions yield 39 trials and 114 yield 40; per-animal discarded tails are 1,666, 219, 91, 60, or 71 frames and are in metadata. Exact 75 cm coordinates are clipped into bin 2 rather than producing class 9/12. All-NaN unregistered cells are excluded, finite registered cells contain no intermittent NaNs, and zero-event registered cells remain intentionally. Trial arrays are uniformly 600 bins. All classes are 0-8. Geometry conflict is only 152/4,912,200 bins (0.003094%).
7. **Post-fix complete rerun**: After the final continuous-session smoothing fix, sample conversion, sample verification, sample training, full conversion, full verification, and every independent sanity check were rerun. Final conversion took 161.29 s plus 5.63 s save; all checks passed.

### Issues Found and Resolved
- **Geometry axis convention found during Step 9**: Native blocked vectors flatten `(y,x)` while position classes flatten `(x,y)`. All eight dihedral transforms were compared; transpose uniquely reduced impossible position/block overlap from 17.14% to 0.0031%. Geometry conversion was fixed and all sample/full artifacts were regenerated.
- **Filter-boundary implementation/documentation mismatch**: The plan specified smoothing the continuous physical session before trial slicing, but the first implementation trimmed the incomplete tail first. It was changed to smooth every native frame, then crop/pool complete trials, preventing a synthetic filter boundary at the final retained minute. All affected artifacts and checks were regenerated successfully.
- **Final re-check result**: No unresolved issues. Independent output shows `ALL INDEPENDENT SANITY CHECKS PASSED`; final format verification has no errors or warnings.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes; 2.324551 (epoch 1) to 1.157662 (epoch 200), monotonically at every printed checkpoint. Held-out loss: 1.142679. Training completed on CUDA without fallback.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| position_3x3 | 0.6979 | 0.6083 | Chance 0.1111; validation is 5.47x chance. |

`train_decoder_full_out.txt` ends with `train_decoder.py finished successfully.` Sample and prediction plots were created and inspected. In the prediction plot, dashed predictions track sustained true-position segments and many transitions; errors are plausible local/brief confusions rather than temporal shifts or constant-class collapse.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| position_3x3 | Validation balanced accuracy 0.6083; chance 0.1111; training 0.6979 | Paper reports 15x15 GaussianNB Euclidean error rather than class accuracy: precomputed data give 22.77 cm across animals on day 1 and 11.50 cm on day 31 (six animals), matching Figure 1F’s visual trend. |
| RSM prediction across animals | Not a target output | Paper reports significant above-shuffle RSM prediction effects but no categorical accuracy value; not applicable. |
| Animal identity from RSM | Not a target output | Paper reports no difference from shuffled identity (chi-square p=1); not applicable. |

1. **Accuracy versus chance**: 0.6083 / 0.1111 = 5.47x chance, far above both the 1.0x bug threshold and 1.5x investigation threshold.
2. **Paper comparison**: The paper’s only directly relevant decoder uses a different target granularity and metric. Its exact released errors improve from 22.77 cm on day 1 to 11.50 cm on day 31. The converted decoder’s strong held-out coarse-position accuracy is consistent with, rather than below, that evidence of robust spatial information. No categorical 3x3 paper value exists to claim numerical equality.
3. **Train/validation gap**: 0.6979 / 0.6083 = 1.147, below the specified 1.5x overfitting threshold; absolute gap is 0.0896.
4. **Loss and completion**: All 29 printed loss checkpoints strictly decrease; the log has the success marker and no CUDA fallback.
5. **Target validity/alignment**: Although accuracy was not low, the requested diagnostics were already performed: three specific raw trials across three animals pass output `np.allclose`; neural/output synchronization passes exact three-frame checks; all nine classes occur and the largest is only 20.03%; reference filtering/processing comparison is in Step 10; processing and prediction plots show no lag.

No conversion change is indicated by this review.

### Issues Found and Resolved
- No new issue. The earlier geometry and filter-edge issues remained resolved under full training.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with description, format, loading example, statistics, processing, reproduction, and decoder results
- [x] cache/ folder created with `README_CACHE.md`
- [x] Investigation script/output, rendered paper page, and bytecode moved into cache; required deliverables and validation plots/logs remain at project root

Final file audit confirmed all eleven required deliverables are present and non-empty. `converted_data.pkl` is 6.205 GiB; `sample_data.pkl` is 31 MiB. Final full verification is clean and final full training ends successfully with 0.6083 held-out balanced accuracy.
