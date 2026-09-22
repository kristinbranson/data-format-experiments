# Dataset Conversion Notes

## Overview
- **Dataset**: A flexible hippocampal population code for experience relative to reward (provided paper/code/data)
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

Environment check: Python 3.13.15; NumPy 2.4.4; PyTorch 2.6.0+cu124 imported successfully.
Checkpoint: `ls -la /app/CONVERSION_NOTES.md` confirmed the notes file exists.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `create_sess` / `append_session_data` | `src/reward_relative/preprocessing.py` | LOADING | Construct TwoPUtils session, align VR to imaging, load curated Suite2p ROIs, and attach lick/reward/speed streams and spatial trial matrices. |
| `vr_align_to_2P` (external TwoPUtils; documented by repository) | TwoPUtils `preprocessing.py` | PROCESSING | Synchronize VR to imaging frames: linear interpolation for position/time, nearest for categorical state, cumulative interpolation/differencing for events, Gaussian smoothing of displacement before speed. |
| `dff` | `src/reward_relative/preprocessing.py` | PROCESSING | Keep on-track trial samples, subtract 0.7 neuropil, compute per-trial maximin baseline (15-frame smoothing; 300-frame min/max filters), calculate dF/F, smooth by 2 frames, and optionally OASIS-deconvolve into `events`. |
| `multi_anim_sess` | `src/reward_relative/utilities.py` | LOADING / PROCESSING | Load session pickles, calculate dF/F/events, trial outcome/environment/reward-zone labels, trial subsets, and optional place-cell statistics. |
| `get_trial_types` | `src/reward_relative/behavior.py` | PROCESSING | For each `[trial_start, teleport)` interval, mark reward only when reward and reward-zone signals occur; obtain environment morph. |
| `get_reward_zones` | `src/reward_relative/behavior.py` | PROCESSING | Derive per-trial A/B/C reward-zone coordinates and labels from scene and switch trial (normally trial index 30). A/B/C map to 80–130, 200–250, 320–370 cm. |
| `define_trial_subsets` | `src/reward_relative/behavior.py` | CURATION | Split switch sessions chronologically by reward-zone label; optionally split nonswitch sessions into halves. |
| `get_timeseries_data` | `src/reward_relative/glmUtils.py` | PROCESSING / CURATION | Copy synchronized deconvolved events and behavior within `start-1:stop-1`, binarize cumulative lick counts, reject sensor-error trials, optionally require speed >=2 cm/s, and mask invalid samples. |
| `calc_place_cells` / `calc_shuffle` | `src/reward_relative/spatial.py` | CURATION | Identify place cells using 10-cm spatial bins, events at speed >2 cm/s, 100 position/activity shuffles, and p<0.05. |
| `CircularRegression` / `train_vs_test_blocks` | `src/reward_relative/decode.py` | PROCESSING | Paper decoder for continuous circular reward-relative position from event activity with block cross-validation. |

### Notes
- Native data are already author-processed `multi_anim_sess` pickles, so dF/F does **not** need to be recomputed. The target neural stream should use the stored `sess.timeseries['events']`, matching the paper decoder and avoiding an incompatible second deconvolution.
- Suite2p `iscell.npy` was manually curated before session construction; loaded neural arrays therefore represent curated cells. Place-cell selection is analysis-specific and the paper decoder divides place cells into subclasses, but the downstream task asks to decode several behaviors and does not require place-cell-only activity. The mapping choice is finalized after inspecting the supplied data and paper.
- Streams are synchronized at the imaging-frame sampling rate (~15.5 Hz; ~64.5 ms per frame). Source trial slicing has a consistent historical one-index correction: `start-1:stop-1`.
- Author preprocessing preserves discrete VR events during downsampling by interpolating cumulative counts and differencing. Lick values can consequently exceed one and are binarized as `>0`/clipped to one for binary use.
- The paper position decoder uses continuous event activity, a speed threshold of 2 cm/s, and occupancy matching because it asks a different scientific question. Our trial-aligned decoder must retain stopped samples because speed and lick are explicit outputs; this necessary difference will be documented.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
The supplied DANDI 001361 dataset contains `dandiset.yaml` plus 152 NWB 2.8 files in `data/sub-m*/sub-m*_ses-*_behavior+ophys.nwb` (92.45 GB). There are 11 mice. Ten have 14 sessions; m11 has sessions 03–14 (12 sessions).

Each NWB contains:
- `processing/ophys/Deconvolved/plane{0,1}/data`: time x ROI float32 author-computed deconvolved calcium events. All sessions have plane0 and 28 sessions (m17/m18) also have plane1.
- `processing/ophys/Fluorescence` and `Neuropil`: time x ROI source fluorescence streams (not needed because author events are present).
- `processing/ophys/ImageSegmentation/PlaneSegmentation/{iscell,planeIdx}`: Suite2p cell flag/probability and the plane assignment. The concatenated segmentation order partitions exactly by `planeIdx` and matches each plane's columns.
- `processing/behavior/BehavioralTimeSeries`: frame-aligned full streams for `position`, `speed`, `lick`, `reward_zone`, `environment`, `trial number`, `trial_start`, `teleport`, `scanning`, and `autoreward`, plus sparse `Reward` delivery timestamps.
- Behavioral timestamps have a constant 64.483627 ms interval (15.5078125 Hz). Single-plane metadata report 15.5078125 Hz; dual-plane files report the scanner rate 31.015625 Hz, but their behavioral and per-plane sample interval is still 64.483627 ms as described by the repository.
- All trial starts and teleports are paired. On-track samples run from the nonzero `trial_start` frame through immediately before the paired `teleport`; trial-number transitions independently agree.
- Ten dual-plane files contain one extra final neural row relative to behavior; it is outside all trial intervals and will be safely ignored by slicing to behavioral trial endpoints.
- Environment is exactly one valid value (0 or 1) per trial; `-1` occurs only outside synchronized acquisition. Lick and reward-zone signals are cumulative counts per imaging frame and can exceed 1.
- Sparse `Reward` entries correspond to reward delivery. `reward_zone>0` occurs on rewarded trials and its position reveals zone identity. Zone identity on omission trials is not stored explicitly, but all two-zone sessions switch at trial index 30 and fixed-zone/switch identities are unambiguous from rewarded neighboring trials; this is consistent with reference code.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 138,678 curated cell-session entries after `iscell[:,0]`; 312,110 raw ROIs before curation |
| Neurons / session | curated min 155, max 2,341, mean 912.36, median 921.5 |
| Subjects | 11: m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19 |
| Sessions / subject | 14 except m11=12; 152 total |
| Trials (total) | 12,216 |
| Trials / session | min 41, max 100, mean 80.37; most sessions have 80 |

Additional native statistics: 3,610,877 behavioral frames; 10,345 sparse rewards (84.684% rewards/trials before any trial rejection); all recordings are labeled `hippocampus, CA1`; trial environments total 6,226 ENV1 and 5,990 ENV2. Available continuous or event variables are neural events/fluorescence/neuropil, time, position, speed, lick, reward delivery, reward-zone entry, autoreward, scanning, environment, trial number, trial start, and teleport.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Not reported as a session-sum because cells can be tracked/repeated across days | Paper Methods | 
| Neurons / session | 155–2,172 putative pyramidal neurons | “This approach yielded 155–2172 putative pyramidal neurons per session” |
| Subjects | 11 switch-task mice in supplied cohort | “counterbalanced across mice (n = 11 mice)” |
| Sessions / subject | Up to 14 daily sessions; m11 imaging began day 3 | “total of 14 days”; “imaging started on day 3” for m11 |
| Trials (total) | 12,376 imaged trials cited for 11 switch mice in lick-QC section | “n = 81 out of 12,376 trials removed across 11 switch mice” |
| Trials / session | 80.5 ± 7.4 across 14 mice/all imaging days; target 80–100 | “mean ± s.d., 80.5 ± 7.4 trials” |
| Neural data time bin | ~64.5 ms (~15.5 Hz per plane) | “sampling rate of ~15.5 Hz per plane” |
| Behavior data time bin | 0.0645 s, synchronized to imaging | “0.0645 s imaging frame samples”; all streams sampled at ~15.5 Hz |
| Reward rate | ~85% expected | “Reward was randomly omitted on approximately 15% of trials” |
| Lick artifact trials | 81 / 12,376 (~0.65%) | Trials with >30% frames having cumulative lick count >2 were removed from lick analysis |
| Track and reward zones | 450 cm; A=80–130, B=200–250, C=320–370 cm | Hidden reward-zone task Methods |
| Switch schedule | Switch after 30 trials on days 3,5,7,8,10,12,14 | Hidden reward-zone task Methods |


### Processing Details
- Imaging/behavior streams are synchronized at ~15.5 Hz. Dual-plane scans are interleaved at ~31 Hz, giving ~15.5 Hz per plane. The task decoder must retain this native temporal resolution and align each trial at its track-entry frame.
- Manual Suite2p ROI curation precedes analysis. dF/F uses a trial-independent maximin baseline within each trial with a 20-s window, then two-sample (~0.129-s s.d.) Gaussian smoothing. OASIS deconvolution removes calcium-kernel asymmetry; the result is explicitly not interpreted as spike rate.
- Reference spatial analyses exclude speed <2 cm/s and use 45 10-cm bins. The paper RR decoder takes synchronized deconvolved events at speed >2 cm/s and predicts circular RR position with tenfold cross-validation and occupancy matching.
- This downstream task instead explicitly predicts speed (including the <2 cm/s class), lick, absolute position, and signed distance to the active zone on the native time axis. Dropping slow frames or spatially averaging would erase required labels, so neither reference speed filtering nor spatial binning applies to the target trial-aligned representation.
- Reward zone identity is a trial context: one of A/B/C, with all switch days changing at trial index 30. Reward outcome is per trial and random omissions are expected at ~15%.
- Signed distance is interpreted relative to **any location in the 50-cm active reward-zone interval**: negative before zone start, exactly zero inside the zone, and positive after zone end. This interpretation follows the requested wording and differs appropriately from paper RR coordinates, which are centered only on zone start and circularized.

### Curation Steps

**Neuron curation rules**:
Apply the supplied manual Suite2p `iscell[:,0]` mask and pool planes, matching the paper. The paper additionally excluded speed-correlated putative interneurons (Pearson r>0.5, 0.42±0.85%) from specific downstream place-cell analyses. That analysis-specific filter is not applied to a general neural-to-behavior decoder: the NWB lacks stored dF/F, the target is not restricted to pyramidal place-cell analyses, and excluding neurons based on correlation with the speed output would directly select against a requested prediction. This rationale will be rechecked against decoder behavior.

**Trial curation rules**:
Use paired on-track trial-start through pre-teleport frames; exclude teleport/ITI and pre-acquisition samples. Reject entire trials with the paper’s lick-sensor artifact rule (>30% of on-track frames having cumulative lick count >2), because lick is a requested output and cannot be valid for those trials. Keep rewarded and omitted trials and stopped frames. Require at least two retained trials (all sessions far exceed this).

### Decoders Trained
| Decoded variable | Accuracy |
| Paper circular reward-relative position | Continuous cosine decode score, not categorical accuracy; 0=random, 1=perfect. Across 77 switch sessions, within-condition decoding was above shuffle for RR/TR/non-RR populations; cross-switch decoding was above shuffle only for RR cells (reported effect size 2.7; P=7.54e-14). |
| Target six categorical variables | Not reported in paper. Direct numeric comparison is not possible because the supplied trainer uses a different architecture, categorical balanced accuracy, all curated cells, all trials, and six labels. Chance baselines are 1/7, 1/5, 1/5, 1/2, 1/3, and 1/2. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Trial indexing | Uses `trial_start_inds`/`teleport_inds`; historical session arrays are sliced `start-1:stop-1` because those indices came from one-index-adjusted TwoPUtils objects | NWB has zero-based event frames; each start/teleport pair agrees with trial-number transitions and positions ~0→450 | On-track lap terminated by teleport | Use native NWB zero-based `[start:teleport)` slicing. Applying the pickle-specific `-1` adjustment to NWB would be an off-by-one error. |
| Trial total | Original analyses refer to 12,376 trials across 11 switch mice | Supplied ophys NWBs contain 12,216; m11 lacks two 80-trial NWBs for days 1–2 | m11 imaging began on day 3, although lick-QC text cites 12,376 | Use all 12,216 trials actually paired with neural data. The exact 160-trial difference equals the two absent m11 sessions. |
| Lick artifact cutoff | `glmUtils.get_timeseries_data` implements >35% despite a stale “>50%” comment; other behavior helpers expose configurable thresholds | >30%=81 trials, >35%=69, >50%=44 | Methods explicitly say >30% and 81 trials | Use >30%. The supplied data reproduce the paper’s 81 rejected trials exactly, decisively resolving code-version inconsistency. |
| Neuron range | Load curated Suite2p masks; later place-cell analysis optionally excludes speed-correlated interneurons | `iscell` yields 155–2,341 cells/session | Manual curation reported 155–2,172; later interneuron filter removes only 0.42±0.85% | Use authoritative per-file `iscell` masks. The small release-versus-paper maximum mismatch likely reflects updated NWB curation/export and cannot justify overriding explicit masks or inventing a cell cap. |
| Neural planes | Session/analysis code pools planes except anatomy analysis | 28 m17/m18 sessions contain plane0 and plane1 with matching `planeIdx` partitions | Two interleaved planes for m17/m18, pooled for analyses | Concatenate curated plane0+plane1 cells on the shared behavioral frame axis. |
| Sampling rate | Single-plane ~15.5 Hz; dual scan metadata can be ~31 Hz before division by `n_planes` | All behavior timestamps have 64.483627-ms spacing; dual-plane neural series metadata say 31.015625 Hz but arrays align one-to-one with behavior frames | ~15.5 Hz per plane | Trust synchronized behavioral timestamps/per-plane description: common bin size 64.483627 ms. |
| Reward outcome | Rewarded iff reward delivery and reward-zone entry occur in trial | 10,342 rewarded trials; all have `reward_zone>0`; 52 additional zone entries lack delivery; 3 trials have two sparse delivery events | ~15% random omissions | Binary outcome is whether any sparse Reward timestamp falls in `[start,teleport)`, equivalent to reference logic here. Rate 84.659%, consistent with paper. |
| Reward-zone identity | `get_reward_zones` uses scene and switch at trial 30 | Scene name/explicit label absent in NWB; zone-entry positions reveal schedules without contradictions, and every pre/post segment has observations | A/B/C coordinates and all switches after 30 trials | Infer the modal observed zone label separately for trials 0–29 and 30–end, then fill omissions. No observed label contradicts its inferred segment. |
| Slow frames | Paper spatial/RR decoder excludes <2 cm/s | Full synchronized speed available, including slight smoothing-derived negatives | Speed <2 excluded only “for all neural spatial activity analyses” | Retain all on-track frames because speed class 0 and lick during stopping are target outputs; this is a task-required difference. |

Final consistent understanding: each target session is one NWB; trials are paired track laps; neural activity is manually curated, pooled deconvolved events at the native frame rate; all behavioral labels use the same timestamps without resampling; invalid lick trials are removed wholesale; teleport periods are excluded; no session needs removal.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `processing/ophys/Deconvolved/plane*/data` + `PlaneSegmentation/{iscell,planeIdx}` | `neural` | For each plane select columns whose matching `iscell[:,0]>0.5`, transpose time×cell to cell×time, concatenate planes, and slice native zero-based `[trial_start:teleport)` | `multi_anim_sess`; paper RR decoder / `glmUtils.get_timeseries_data` | Raw author-computed events, float32, no extra normalization/deconvolution. |
| behavior timestamps | `input[0]` time from trial start | `timestamp[start:stop] - timestamp[start]`, seconds | synchronized session time | Continuous time-varying float32. |
| `environment` | `input[1]` environment type | Unique valid value within trial (0=ENV1, 1=ENV2), repeated across time | `get_trial_types` morph | Binary per-trial context represented across time for matrix consistency. |
| within-session trial index / `trial number` | `input[2]` trial number | Zero-based trial index, repeated across time | `get_timeseries_data` trial IDs | Continuous per-trial input. NWB trial number agrees with enumeration. |
| preceding trial's sparse `Reward` outcome | `input[3]` previous outcome | 0 if preceding trial omitted, 1 if rewarded; first trial=0 because no prior within-session observation; repeated across time | `get_trial_types` | Current-trial outcome is never leaked into this input. |
| `position` + inferred active `[zone_start,zone_end]` | `output[0]` signed distance to any zone location | `position-start` before zone, exactly 0 within inclusive interval, `position-end` after zone; discretize to requested 7 bins | `get_reward_zones` supplies coordinates; target-specific transform | Non-circular, time-varying. Boundary convention: class 1 includes -50 and -10; class 2 is (-10,0); class 4 is (0,10]; class 5 is (10,50]. |
| `position` | `output[1]` absolute position | Requested five 90-cm categories over 0–450 cm | paper 450-cm track | Time-varying int64. 360 is assigned to bin 3; >360 to bin 4. |
| `speed` | `output[2]` speed | Requested <2, 2–10, 10–20, 20–40, >40 cm/s categories | synchronized, Gaussian-smoothed speed in VR alignment | Retain slight negative smoothing artifacts in <2 class. Boundaries 10/20/40 stay in preceding inclusive bin. |
| `lick` | `output[3]` lick | `lick>0` after rejecting artifact trials; cumulative within-frame count becomes binary | `get_timeseries_data`; paper lick Methods | Time-varying int64, not smoothed. |
| zone-entry positions + fixed A/B/C intervals + trial-30 rule | `output[4]` reward-zone location | Infer modal label in trials 0–29 and 30–end; fill all trials; map A/B/C→0/1/2 and repeat in time | `get_reward_zones` | Every segment has observations and zero contradictions. |
| sparse `Reward/timestamps` | `output[5]` reward outcome | Nearest exact behavioral timestamp; 1 iff ≥1 event lies in trial, else 0; repeat in time | `get_trial_types` | Three trials contain two deliveries but remain binary. |
| NWB `subject_id` | `subjects`, `subject_idx` | Natural numeric subject ordering; session index points to subject | NWB metadata | 11 subjects. |
| ImagingPlane `location` | `brain_regions`, `brain_region_idx` | Normalize `hippocampus, CA1` to `CA1`; every retained neuron gets index 0 | NWB/paper | One brain region. |

### Key Decisions
1. **Native temporal bins and variable trial duration**: Preserve every synchronized imaging frame and variable `[start,teleport)` trial length. This meets common-bin-size requirements while retaining timing, speed, and licks; padding/resampling would add fabricated observations.
2. **All curated neurons, both planes**: Apply manual `iscell` but do not restrict to place cells or output-correlated interneuron filters. This is the least task-biased population and matches the general neural-decoder goal; planes are pooled as in reference analyses.
3. **Lick artifact rejection is whole-trial**: Remove the 81 trials matching the paper's >30% rule from neural/input/output together. Keeping them with an invented lick value would contaminate one required output, while masking only lick is unsupported by the target structure.
4. **Zone label recovery**: Infer schedule from actual zone-entry positions rather than hard-coding mouse sequences. Trial 30 segmentation comes from paper/code and is empirically contradiction-free.
5. **Outcome from sparse delivery, not zone entry**: Zone entry can occur on omissions; sparse `Reward` is the actual delivery event. This exactly preserves 84.66% rewarded trials.
6. **Sample selection**: `--sample` will use two switch sessions (`m11` session 03 and `m12` session 03 when available), providing all A/B/C labels across the sample and testing multi-class conversion rather than selecting two fixed-zone files.
7. **Storage types**: neural/input float32; categorical outputs int64; metadata/index arrays standard NumPy integers. This avoids float64 memory inflation without changing source float32 events.
8. **Output organization**: All four inputs and all six outputs are matrices repeated across each trial's time dimension. This keeps one unambiguous shape per field and permits the validator/trainer to align every label to neural samples.
9. **Metadata offsets**: `off_start=0.0`; `off_end=None` because trial ends/teleports are variable-duration. `time_bin_size=64.483627204...` ms from the actual shared timestamps.

### Planned Sanity Checks
- [ ] Independently load a raw NWB and use `np.allclose` to compare selected event values (multiple cells/trials/times) after mask, plane concatenation, and transpose.
- [ ] Independently calculate time-from-start, environment, trial index, and previous outcome for selected raw trials and compare inputs with `np.allclose`.
- [ ] Independently calculate position/speed/lick/zone/outcome labels for selected raw trials and compare every output row with `np.allclose`.
- [ ] Confirm 152 sessions, 12,216 native paired trials, exactly 81 lick-artifact rejections, and 12,135 converted trials; every session retains at least two.
- [ ] Confirm curated neuron totals/counts from raw masks, including both planes, equal converted shapes.
- [ ] Assert uniform 64.483627-ms bins, matched neural/behavior slices, finite values, monotonic within-trial time, allowed category ranges, and zero zone-schedule contradictions.
- [ ] Compare rewarded fraction (~85%), ENV1/ENV2 counts, A/B/C distributions, trial count mean/range, and neuron count mean/range with raw data and paper.
- [ ] Plot raw continuous variables overlaid with discrete labels and neural activity for up to two sessions, emphasizing starts/ends and all threshold boundaries.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with `--full` (default), `--sample`, and `--show-processing`. It performs native HDF5/NWB loading, explicit timestamp/trial validation, sparse reward alignment, empirical zone-schedule inference, paper lick-QC, both-plane `iscell` curation, categorical transforms, metadata construction, per-session timing, and pickle serialization. Static compilation and CLI help completed without error.

Code inefficiencies identified:
Per-trial HDF5 reads or fancy selection of thousands of noncontiguous columns would cause excessive chunk decompression and point-selection overhead. Concatenating repeated arrays before trial slicing would also add avoidable copies.

Code speedups added:
Each dense deconvolved plane is read once, its local cell mask is applied in memory, and curated planes fill one preallocated time×cell matrix. Trials are then copied once into final contiguous cell×time float32 matrices. Behavioral streams are loaded once per session and transforms are vectorized.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 1,298 curated cell-session entries (2,419 raw ROIs) |
| Neurons / session | [155, 1,143] |
| Subjects | 2 (m11, m12) |
| Sessions / subject | 1 each |
| Trials (total) | 160 retained / 160 native; 0 artifact trials in these sessions |
| Trials / session | [80, 80] |
| time from trial start range | [0, 20.9] s |
| environment range | [0, 0] (both representative sessions are ENV1) |
| trial number range | [0, 79] |
| previous outcome range | [0, 1] |
| distance class fractions | [0.240, 0.094, 0.042, 0.266, 0.022, 0.079, 0.257] |
| position class fractions | [0.213, 0.170, 0.250, 0.206, 0.160] |
| speed class fractions | [0.076, 0.065, 0.065, 0.239, 0.555] |
| lick fractions | [0.833, 0.167] |
| zone fractions A/B/C | [0.318, 0.406, 0.276] by time; 50/60/50 trials |
| outcome fractions | [0.113, 0.887] by time; 142/160 rewarded trials |

### Processing Plots Review
`processing_m11_ses-03.png` and `processing_m12_ses-03.png` were visually reviewed. Trial starts occur near 0 cm, teleports near 450 cm, neural events share the behavior frame axis, iscell reductions are plausible, position/distance/speed threshold crossings coincide with class steps, zone interiors map to distance 0/class 3, lick counts map to binary events, and contextual inputs are constant within trials. No temporal shift or discretization anomaly was visible.

Manual structure inspection found all 160 neural/input/output trial lengths exactly aligned, expected float32/float32/int64 dtypes, correct subject/region indices, and complete per-session metadata. Formal validation reported: “Data format is valid, no errors or warnings.”

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Dense once-per-plane HDF5 read and vectorized curation | Avoids ~160 trial-wise decompressions per two-session sample; sample conversion itself took 1.77 s. |
| float32 neural/input arrays | Approximately halves memory/file size versus float64; sample pickle is 62 MB. |

| Step | Time / Session | Estimated Total Time |
| Conversion excluding pickle write | 0.89 s/session on representative sample | ~2.3 min for 152 sessions after conservatively scaling 1:1 by count; allow ~3–5 min for larger dual-plane sessions and final write |
| Sample pickle write | 0.025 s/session | a few seconds to minutes depending on full output size; total safely below 15 min |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| distance to reward zone | 0.5460 | 0.3757 |
| absolute position | 0.6264 | 0.5249 |
| speed | 0.5171 | 0.3731 |
| lick | 0.7557 | 0.7115 |
| reward zone location | 0.9394 | 0.8608 |
| reward outcome | 0.7930 | 0.6031 |

Training completed on CUDA. Loss fell from 126.36 (epoch 1) to 1.29 (epoch 200); test loss was 3.74. Every validation balanced accuracy exceeded uniform chance (0.1429, 0.2, 0.2, 0.5, 0.3333, 0.5 respectively). This supports correct neural/behavior alignment and label construction.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 8.996 GiB
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | Not given as repeated cell-session sum | Apply curated iscell and pool planes | 138,678 curated / 312,110 raw ROI-session entries | 138,678 | Yes |
| Mean neurons/session | range 155–2,172 in manuscript release | Explicit per-file iscell; planes pooled | 912.36; range 155–2,341 | 912.36; range 155–2,341 | Exact data match; documented paper-release maximum discrepancy |
| Subjects | 11 switch mice | switch cohort | 11 | 11 | Yes |
| Sessions | up to 14/mouse; m11 starts day 3 | one NWB=session | 152 (14×10 + 12) | 152 | Yes |
| Trials (total) | 12,376 cited including 160 unavailable m11 day1–2 trials; 81 lick-QC removals | paired starts/teleports; paper lick filter | 12,216 available; 81 artifacts | 12,135 retained | Yes after required data availability/QC |
| Trials/session (mean) | 80.5±7.4 across broader 14-mouse cohort | retain all paired valid trials | 80.37 native | 79.84 after 81 removals | Yes |
| Time bin | ~64.5 ms / ~15.5 Hz | native synchronized imaging frames | 64.483627 ms | 64.483627 ms | Yes |
| Reward rate | ~85% | sparse Reward + zone entry per trial | 84.6595% native | 84.2% of retained timepoints (duration-weighted) | Yes |
| Environment range | ENV1/ENV2 | morph 0/1 | [0,1] | [0,1] | Yes |
| Time-from-start range | variable lap durations | timestamp subtraction | [0,216.5] s | [0,216.5] s | Yes |
| Distance output distribution | Not categorical in paper | target-specific | N/A | [0.251,0.102,0.073,0.238,0.021,0.072,0.243] | Sensible/all classes |
| Position output distribution | 45 spatial bins used for paper analyses | target-specific five bins | N/A | [0.212,0.177,0.231,0.226,0.154] | Sensible/all classes |
| Speed output distribution | <2 cm/s excluded only in spatial analyses | target-specific five bins | N/A | [0.117,0.087,0.134,0.319,0.343] | Sensible/all classes |
| Lick output distribution | binary after artifact QC | clip cumulative count to binary | N/A | [0.777,0.223] | Sensible/both classes |
| Zone output distribution | counterbalanced A/B/C | A/B/C | retained trial counts A=4,172, B=3,974, C=3,989 | time fractions [0.332,0.336,0.333] | Yes/balanced |

Full conversion took 55.73 s plus 8.41 s to write. Formal verification reported “Data format is valid, no errors or warnings.” Spot checks in the verifier covered every session's trial count, neuron count, input/output range, class fractions, and region index; all six outputs contain every expected class globally.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Read `verification_full_out.txt`. It begins “Data format is valid, no errors or warnings” and ends “Data verification complete.” There are no actionable warnings.
2. **Independent neural sanity checks**: `audit_conversion.py` loaded original NWBs directly without importing `convert_data.py`, reconstructed curated cell ordering and full trial slices, and used exact `np.allclose(..., rtol=0, atol=0)` comparisons. Tests passed for m11 session 03 trial 5 (single plane), m18 session 03 trial 31 (dual plane), and m4 session 14 trial 70 (a session containing many rejected lick-artifact trials).
3. **Independent input sanity checks**: For the same three raw trials, independently reconstructed timestamp subtraction, environment, original within-session trial index, and preceding raw-trial outcome. Full matrices passed `np.allclose` (1e-6 floating tolerance). This specifically verifies that rejection does not renumber trials or change the meaning of “previous trial.”
4. **Independent output sanity checks**: For the same trials, independently inferred zone schedules and reconstructed all six categorical output rows from raw position/speed/lick/reward streams. Every full output matrix passed exact `np.allclose`.
5. **All-file aggregate audit**: Independently opened all 152 NWBs and exactly matched converted metadata using `np.allclose`: 12,216 native trials; 12,135 retained; 81 removed; 312,110 raw ROIs; 138,678 curated cells; 10,342 rewarded trials; 2,576,026 retained timepoints. Per-session retained trial and neuron counts also matched all 152 converted sessions.
6. **Reference code comparison—loading**: Reference `create_sess`/`multi_anim_sess` loads synchronized VR and Suite2p streams; conversion loads their NWB export directly. No raw dF/F recomputation is performed because author-deconvolved events are supplied.
7. **Reference code comparison—filtering**: Conversion applies the supplied manual `iscell` flag and pools planes. It does not apply place-cell or speed-correlated-interneuron analysis filters because those are downstream, output-correlated filters inappropriate to this general decoder. Trials follow paper lick-QC exactly (>30%), reproducing 81 exclusions.
8. **Reference code comparison—alignment**: Both use paired trial-start/teleport intervals. The NWB audit confirms zero-based `[start,teleport)` is correct; the `-1` in old pickle code is specific to its index convention. Exact neural/input/output trial checks rule out a one-frame shift.
9. **Reference code comparison—binning**: Paper spatial analyses use 10-cm bins and remove speed<2, but the requested decoder requires native time-varying time/speed/lick outputs. Conversion therefore preserves native 64.483627-ms frames and applies only the requested categorical thresholds.
10. **Reference code comparison—inputs**: Environment corresponds to paper `morph`; trial index matches `get_timeseries_data`; previous outcome is independently derived from preceding trial Reward delivery and first-trial default 0. These are target-requested inputs, not paper decoder covariates.
11. **Reference code comparison—outputs**: Outcome matches `get_trial_types`; zones match `get_reward_zones`; lick binarization matches paper/code. Absolute position, interval distance, and categorical speed are target-specific transforms. Their full value ranges and distributions are valid.
12. **Key statistics**: Subjects, sessions, native trials, available-cell masks, reward rate, time bin, environment range, zone balance, output ranges, and lick artifacts were compared across paper/code/raw/converted in Steps 4 and 9. All raw-comparable quantities match exactly; manuscript-only discrepancies (160 unavailable m11 trials and updated maximum cell count) are resolved/documented.
13. **Edge cases**: Synthetic exact-boundary arrays passed expected categories for distance (-50,-10,0,10,50), position (90,180,270,360), and speed (2,10,20,40). All converted trials start at time 0, have strictly increasing time, exclude the teleport frame, contain finite activity/input, use allowed category indices, and each session retains ≥2 trials. Ten one-row-long neural streams are safely truncated only outside trial endpoints.
14. **Global distributions**: Exact timepoint counts were distance `[647274,262652,188729,614278,53356,184916,624821]`; position `[546139,455285,595609,583250,395743]`; speed `[300302,225017,345496,822252,882959]`; lick `[2002653,573373]`; zone `[854185,865244,856597]`; outcome `[406272,2169754]`.

### Issues Found and Resolved
- No new conversion issue was found in this review; all independent comparisons passed on the first iteration. Earlier source discrepancies and their resolutions are recorded in Step 4. `audit_conversion_out.txt` ends `AUDIT PASSED`.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. Epoch 1=173.0809; epoch 100=3.8200; epoch 200=1.2296; test loss=1.1465. Training completed on CUDA without fallback or error.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| distance to reward zone | 0.4182 | 0.3703 | chance 0.1429 |
| absolute position | 0.5141 | 0.4888 | chance 0.2000 |
| speed | 0.4643 | 0.4317 | chance 0.2000 |
| lick | 0.6179 | 0.6098 | chance 0.5000 |
| reward zone location | 0.8413 | 0.8137 | chance 0.3333 |
| reward outcome | 0.5533 | 0.5192 | chance 0.5000 |

All validation balanced accuracies exceed chance. `sample_trials.png` and `predictions.png` were generated and visually reviewed: neural traces are temporally aligned with orderly position/distance labels, sparse lick labels, and constant per-trial zone/outcome labels; prediction overlays show expected tracking without evidence of a label shift.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Validation balanced accuracy | Chance | Multiple of chance | Train / validation | Expectation from paper |
|----------|------------------------------|--------|--------------------|--------------------|------------------------|
| distance to reward zone | 0.3703 | 0.1429 | 2.592× | 1.129× | Closest analogue is continuous circular RR position, which paper shows above shuffle for RR cells; not the same categorical metric. |
| absolute position | 0.4888 | 0.2000 | 2.444× | 1.052× | Paper establishes robust CA1 spatial coding but does not report five-class accuracy. |
| speed | 0.4317 | 0.2000 | 2.158× | 1.076× | No paper neural decoder accuracy for speed. |
| lick | 0.6098 | 0.5000 | 1.220× | 1.013× | No paper neural decoder accuracy for licks. |
| reward zone location | 0.8137 | 0.3333 | 2.441× | 1.034× | No paper categorical zone-identity decoder. |
| reward outcome | 0.5192 | 0.5000 | 1.038× | 1.066× | No paper per-trial binary outcome decoder; omissions are randomized ~15%. |

All outputs exceed chance, and no train/validation ratio approaches the 1.5× overfitting threshold. Distance, position, speed, and zone are >2.15× chance. Lick and outcome are below the 1.5×-chance screening threshold and were investigated fully:

- Three specific raw trials (m11 session 03 trials 0 rewarded, 1 omitted, and 30 rewarded) were reloaded independently. Full lick and outcome arrays passed `np.allclose` against converted output. Their raw reward-event counts were 1, 0, and 1 respectively.
- `critical_review_low_outputs.png` overlays raw mean curated activity, position, binary lick, and outcome on the same native time axis. Visual review found no temporal offset. Earlier independent full-trial checks also covered m18 dual-plane and m4 artifact-heavy data.
- Lick has substantial variation (22.3% positive timepoints; neither class near 99%). Outcome has 15.8% omission timepoints, consistent with the experimental ~15% random omission design.
- Lick is a sparse single-frame behavioral event, making 0.6098 balanced accuracy plausible. Outcome is deliberately randomized per trial and repeated across the entire trial as required; before reward-zone entry, current outcome is not behaviorally/neuronally observable. A result only modestly above 0.5 is therefore scientifically expected, while post-zone neural signals support the observed above-chance increment.
- Changing outcome to switch only after delivery would improve temporal predictability but violate the specified **per-trial** output. Using reward-zone entry as outcome would falsely convert 52 omissions to rewards. Neither change is justified.
- Manual cell filtering, plane pooling, native timing, and author-deconvolved activity were rechecked against paper/code. No missing filter or processing mismatch was found.

### Accuracy Comparison to Paper

The paper reports only one neural decoding analysis: continuous circular RR position from selected RR/TR/non-RR place-cell populations, scored by mean cosine similarity (0=random, 1=perfect), not categorical balanced accuracy. Exact mean scores are plotted but not numerically tabulated in text/caption, so inventing point estimates would be inappropriate. Every reported numeric decoder comparison is listed below.

| Paper decoder condition | Paper report | Closest target result |
|-------------------------|--------------|-----------------------|
| Train-before/test-before, RR cells | Above shuffle; effect size 2.6 | Distance class 0.3703 vs 0.1429 chance (2.592× chance) |
| Train-before/test-before, TR cells | Above shuffle; effect size 3.3 | No cell-subclass-specific target model |
| Train-before/test-before, non-RR remapping cells | Above shuffle; effect size 3.2 | No cell-subclass-specific target model |
| Train-before/test-after, RR cells | Above shuffle; effect size 2.7; P=7.54×10^-14 across 77 sessions | Target validation splits trials within each session rather than testing cross-switch generalization |
| Train-before/test-after, TR cells | Not above shuffle; effect size -0.5; P=0.5 | Not comparable |
| Train-before/test-after, non-RR cells | Not above shuffle; effect size -0.2; P=0.3 | Not comparable |

The categorical target distance result is strongly above its own chance baseline and is directionally consistent with the paper's RR decoding result. The algorithms, labels, cell sets, and test split differ, so no paper numeric accuracy can validly serve as a one-to-one threshold.

### Issues Found and Resolved
- **Lick and outcome below 1.5× chance screen**: Raw-value, temporal-alignment, variation, curation, and reference-processing checks found no conversion bug. Their scores are above chance and consistent with sparse licking and randomized omissions. No conversion change was warranted.
- **Overfitting check**: All train/validation ratios are 1.013–1.129, so no overfitting or leakage issue was found.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

`README.md` documents loading, field order, core processing, key statistics, accuracies, and reproduction commands. Independent audit code/logs and low-output diagnostic artifacts were moved to `cache/`, whose contents are documented in `cache/README_CACHE.md`. All eleven explicitly required deliverables exist and are non-empty. Every workflow step is complete.
