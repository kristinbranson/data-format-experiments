# Dataset Conversion Notes

## Overview
- **Dataset**: A flexible hippocampal population code for experience relative to reward (provided NWB files)
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `paper.pdf`, `methods.txt`: reference manuscript and extracted methods.
- `code/`: paper repository (README, documentation, notebooks, and `reward_relative` Python source).
- `data/`: `dandiset.yaml` plus 152 `*_behavior+ophys.nwb` files under 11 subject directories (`sub-m3`, `m4`, `m7`, `m11`-`m15`, `m17`-`m19`).
- `pynwb_docs/`: local PyNWB documentation.
- `train_decoder.py`, `decoder.py`: supplied validation/training code.
- `Dockerfile`, `docker-compose.yaml`: environment setup.
- `CONVERSION_NOTES.md`: this progress record.

Environment verification: Python 3.13.15, NumPy 2.4.4, and PyTorch 2.6.0+cu124 import successfully. Checkpoint `ls -la /app/CONVERSION_NOTES.md` succeeded.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `multi_anim_sess` | `src/reward_relative/utilities.py` | LOADING/PROCESSING | Loads synchronized session objects, computes dF/F and OASIS-deconvolved events, derives trial labels/reward zones, and optionally finds place cells. |
| `dff` | `src/reward_relative/preprocessing.py` | PROCESSING | Restricts fluorescence to track epochs (`start-1:teleport-1`), subtracts 0.7 neuropil, adds trial neuropil mean, estimates a maximin baseline (15-frame smoothing and 300-frame min/max filters), computes and 2-frame-smooths dF/F, then optionally OASIS-deconvolves it. |
| `get_trial_types` | `src/reward_relative/behavior.py` | PROCESSING | Per trial, marks reward only if reward and reward-zone signals occur; extracts environment (`morph`). |
| `get_reward_zones` | `src/reward_relative/behavior.py` | PROCESSING | Maps scene and switch trial to zone coordinates/labels A, B, C (task coordinates X=[80,130], Y=[200,250], Z=[320,370] cm). |
| `define_trial_subsets` | `src/reward_relative/behavior.py` | CURATION | Splits switch sessions chronologically by reward-zone identity; optionally halves nonswitch sessions. |
| `correct_lick_sensor_error` | `src/reward_relative/behavior.py` | CURATION | Sets a trial's lick stream to NaN if cumulative lick count >2 in more than the chosen fraction of samples (0.35 used in continuous analysis). |
| `calc_place_cells` | `src/reward_relative/spatial.py` | CURATION | Finds spatial-information-significant cells, normally from deconvolved events at speed >=2 cm/s with 100 shuffles and p<0.05. |
| `get_timeseries_data` | `src/reward_relative/glmUtils.py` | PROCESSING/CURATION | Copies events/behavior only from trial start to teleport using the authors' `start-1:stop-1` convention; derives reward-relative position; clips lick counts >1; excludes corrupt-lick and <2 cm/s samples for paper decoders. |

### Notes
- This is two-photon calcium imaging, not electrophysiology. The raw synchronized `sess` lacks dF/F, but the shared NWB release may already contain processed dF/F/deconvolved event series; Step 2 will inspect this through PyNWB. If a reference-derived event series exists, it should be used rather than recomputing from raw fluorescence.
- Suite2p ROIs were manually curated through `iscell.npy` before session creation. Thus the exported ROI table/series is expected to contain curated cells; no electrophysiology unit-quality filter applies.
- Imaging/behavior are already synchronized at one row per imaging frame, approximately 15.5 Hz (~64.5 ms). Trial epochs run from entry to the track through teleport; teleport/ITI samples are normally excluded.
- Paper spatial analyses use trials x 10-cm position bins (0-450 cm). Paper continuous decoding uses deconvolved events, excludes speeds <2 cm/s and corrupt lick trials, and occupancy-matches position bins. The requested decoder instead needs complete time-varying speed/lick outputs, so low-speed frames cannot be dropped merely because of the paper decoder threshold; this expected task-driven difference will be planned explicitly.
- The Fig. 3 paper model predicts circular reward-relative position from deconvolved events using regularized circular regression and block cross-validation. It is not directly comparable to the supplied multi-output categorical neural decoder, but establishes the preferred neural representation and trial-epoch convention.
- Source-index subtlety: preprocessing and continuous decoding repeatedly use Python slices `start-1:stop-1`, whereas some behavioral helpers use `start:stop`. The NWB intervals/timestamps must be examined before choosing bounds; alignment will be validated directly rather than blindly copying an index offset from the legacy `sess` object.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- DANDI 001361, version `0.251124.0550`; local manifest reports 92,448,350,544 bytes, 152 NWB assets, and 11 mice. Each `sub-<mouse>/sub-<mouse>_ses-<day>_behavior+ophys.nwb` is one session. Mice generally have days 1-14; m11 begins at day 3 and has 12 sessions.
- All NWB inspection used `pynwb.NWBHDF5IO(...).read()`; no direct HDF5 parsing was used.
- `processing/behavior/BehavioralTimeSeries` has frame-synchronous `autoreward`, `environment`, `lick`, `position`, `reward_zone`, `scanning`, `speed`, `teleport`, `trial number`, and `trial_start`. It also has sparse `Reward`: 0.004-mL delivery values with event timestamps. Common behavior timestamps have median spacing 0.0644836 s in 15.5-Hz sessions.
- `processing/ophys/Deconvolved/plane0` is frames x ROIs and is already synchronized to behavior. `Fluorescence/plane0` and `Neuropil/plane0` contain raw ROI and neuropil fluorescence. The deconvolved series is the only ready-to-use neural activity representation in the NWB release; no dF/F series is exported.
- `processing/ophys/ImageSegmentation/PlaneSegmentation` contains `pixel_mask`, `iscell`, and `planeIdx`. `iscell` has shape (segmented ROIs, 2): manual/class flag and Suite2p probability. Only rows with `iscell[:,0] > 0` are curated neurons, matching the repository's GUI curation step. ROI response columns refer to all table rows in order, so neural conversion must subset these columns.
- `acquisition/TwoPhotonSeries` stores metadata/external-series scaffolding rather than the analysis signal. `trials` and `intervals` tables are absent; trial boundaries must be read from the frame-aligned `trial_start` and `teleport` series.
- Ophys frame rates are exactly 15.5078125 Hz in 124 sessions and 31.015625 Hz in 28 sessions (m17/m18). Behavior timestamp length, deconvolved row count, and fluorescence row count agree within each inspected session. A common decoder bin width is therefore required; planned resampling is deferred to Step 5.
- There are 73 ENV1-only, 68 ENV2-only, and 11 environment-switch sessions. Scene/reward schedules are encoded in `NWBFile.identifier`, e.g. `Env1_LocationB_to_A` or `Env1_B_to_Env2_C`. Frame environment values are 0 (ENV1) and 1 (ENV2), with -1 only outside synchronized/valid acquisition.
- Every file has equal numbers of trial-start and teleport flags. Trial numbers are zero-based within the NWB behavior stream. Track positions nominally span 0-450 cm; invalid/pre-acquisition is -500 cm and teleport intervals are near -50 cm. Lick and reward-zone signals are cumulative within an imaging frame (often >1), so binary events use `>0` per the repository documentation.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 138,678 curated session-neuron recordings (`iscell[:,0]>0`); 312,110 segmented ROI recordings before curation |
| Neurons / session | curated: min 155, median 921.5, mean 912.36, max 2,341; all ROIs: min 315, median 2,070, mean 2,053.36, max 5,085 |
| Subjects | 11 (`m3`, `m4`, `m7`, `m11`-`m15`, `m17`-`m19`) |
| Sessions / subject | 14 except m11=12; 152 total |
| Trials (total) | 12,216 |
| Trials / session | min 41, median 80, mean 80.37, max 100; all sessions have >=2 |

Additional scale: 3,610,867 imaging frames total; 14,164-51,520 frames/session. Subject-level curated session-neuron totals are m3 13,010; m4 14,488; m7 14,867; m11 2,308; m12 17,841; m13 8,759; m14 9,864; m15 16,214; m17 8,663; m18 25,720; m19 6,944.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Not reported as session-summed total | Same FOV was aligned across days so cells can recur across sessions. |
| Neurons / session | 155-2,172 after manual curation | “This approach yielded 155–2172 putative pyramidal neurons per session.” |
| Subjects | 11 switch-task mice (plus 3 fixed-condition mice not in NWB subset) | “counterbalanced across mice (n = 11 mice)” and fixed cohort n=3. |
| Sessions / subject | Up to 14 daily sessions; m11 imaging began day 3 | “total of 14 days”; “imaging started on day 3” for m11. |
| Trials (total) | 12,376 quoted for lick QC across 11 switch mice | “n = 81 out of 12,376 trials removed across 11 switch mice.” |
| Trials / session | Target 80-100; 80.5 ± 7.4 across 14 mice/all imaging days | Session ended early for disengagement or >50 min. |
| Neural data time bin | ~64.5 ms per plane (~15.5 Hz) | Single plane ~15.5 Hz; m17/m18 acquired interleaved planes at ~31 Hz, ~15.5 Hz per plane. |
| Behavior data time bin | Native VR 50-75 Hz, synchronized/downsampled to imaging at ~15.5 Hz | Unity TTLs synchronized each VR frame to imaging. |
| Reward rate | ~85% expected | Reward randomly omitted on ~15% of trials. |
| Track / reward geometry | 450 cm; A=80-130, B=200-250, C=320-370 cm | Three hidden 50-cm zones, only one active. |
| Switch schedule | switch after trial 30 on days 3,5,7,8,10,12,14 | Every switch occurred after 30 trials; first 10 new-condition trials could autoreward. |
| Corrupt lick trials | ~0.65%, 81/12,376 | >30% of 0.0645-s samples had cumulative lick count >2. |
| Interneuron exclusion | 0.42 ± 0.85% of cells | Exclude Pearson correlation >0.5 between dF/F and running speed. |


### Processing Details
- Imaging and behavior are synchronized at the per-plane imaging rate. Task trials cover the 0-450 cm track; the gray teleport/ITI is outside the requested trial and is excluded from ordinary analyses.
- dF/F is calculated separately within every trial: a 20-s maximin baseline, `(F-baseline)/abs(baseline)`, then a two-sample (~0.129-s) Gaussian smoothing. OASIS/canonical calcium-kernel deconvolution produces the event activity used for spatial information and decoding.
- Spatial analyses use 45 bins of 10 cm and exclude samples with speed <2 cm/s. The requested time-domain decoder must retain low-speed samples to produce the specified speed and lick outputs; this is a necessary downstream-task exception.
- Licks are converted from cumulative counts to binary per-frame events. Sensor-error trials are NaN/excluded from licking analysis when >30% of samples have counts >2.
- Reward-relative position in the paper is circular and anchored to reward-zone start. The requested output instead explicitly asks for signed “distance to any location in the reward zone,” so Step 5 will use signed distance to the nearest point of the closed zone: negative before the zone, zero inside, positive after.
- The paper decoder used deconvolved events at >2 cm/s, tenfold circular-linear regression, and occupancy matching. It reports a cosine decode score (1 perfect, 0 random), not categorical balanced accuracy.

### Curation Steps

**Neuron curation rules**:
1. Manual Suite2p curation removed multi-soma/dendritic ROIs, ROIs without obvious transients, overexpression, and high continuous fluctuations typical of interneurons.
2. Pool both planes for m17/m18 in ordinary analyses.
3. Exclude additional putative interneurons when dF/F-speed Pearson r >0.5 (mean exclusion 0.42%).
4. Place-cell significance is an analysis-specific filter (SI >95% of pooled-cell shuffles in either pre/post set) and should not be imposed on the general neural decoder, which should retain all curated pyramidal neurons unless the supplied validator demands otherwise.

**Trial curation rules**:
Keep complete start-to-teleport track epochs. Early session termination is valid. For the lick output, remove whole trials meeting the >30% cumulative-count>2 sensor-failure rule; no general omission-trial removal is appropriate because reward outcome is a required output and previous outcome is a required input.

### Decoders Trained
| Decoded variable | Accuracy |
| Circular reward-relative position from RR/TR/non-RR cell subsets | Cosine decode score compared with 100 circular-shift shuffles; exact categorical accuracy not reported. Pre-to-pre effect sizes: RR 2.6, TR 3.3, non-RR 3.2. Pre-to-post: RR 2.7, TR -0.5, non-RR -0.2. |
| Spatial extent of significant cross-switch RR decoding | z-scored decode >2 from -104.5 ± 20.1 cm to +152.7 ± 22.9 cm relative to reward-zone start. |

The paper contains no decoder results for absolute-position, speed-bin, lick, zone-label, or trial-outcome classification. Those supplied-decoder accuracies therefore have no direct paper comparator; chance and biological plausibility are the applicable checks.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Neural representation | Recompute trial-wise maximin dF/F and OASIS `events`; raw Suite2p `spks` is not the analysis signal. | NWB `Deconvolved` is finite/nonzero during ITIs, is 1000s-fold larger than reference-recomputed events, and correlates only moderately with recomputed events (median r=0.483 across 32 sample neurons). | Decode from trial-wise dF/F-deconvolved events. | Recompute dF/F and OASIS events from NWB `Fluorescence` and `Neuropil`; do not use NWB `Deconvolved` as final neural input. This is the largest processing correction found. |
| Trial bounds / index offset | Legacy code slices `start-1:stop-1`, reflecting its stored index convention. | NWB event flag at `start` is the first track crossing (median position 1.75 cm); `teleport` row is an interpolated/artifact transition, while `teleport-1` is median 448.94 cm. | Trial starts at entry to 0-cm track and ends on teleport entry. | Slice NWB rows `[trial_start, teleport)` and align time zero to the timestamp of the flagged start. This is semantically equivalent to reference bounds after accounting for legacy one-based indices. |
| Dual-plane rate | Use acquisition frame rate divided by number of planes (~15.5 Hz/plane). | m17/m18 ROI series metadata says 31.015625 Hz, but each plane independently has one row per behavior timestamp and those timestamps are 0.0644836 s apart. | ~31 Hz interleaved acquisition, ~15.5 Hz sampling per plane. | Trust explicit behavior timestamps and aligned row counts, not erroneous aggregate `rate`; both planes are already at 15.5 Hz and are concatenated by ROI table region. No temporal downsampling is needed. |
| ROI curation | `iscell[:,0]`, then speed-correlated interneuron exclusion for later analyses. | 138,678 manual cells; max session 2,341. Recomputed dF/F-speed filter removes 18/2,341 in max session, leaving 2,323. | 155-2,172 manual cells/session; speed filter removes 0.42 ± 0.85%. | Apply both `iscell` and r>0.5 filters. The remaining maximum mismatch is attributed to the newer DANDI release/version or a manuscript summary bound; local NWB is authoritative for conversion and the minimum agrees exactly. Do not impose an arbitrary count cutoff. |
| Trial count | Repository schedule supports variable 41-100-trial sessions. | 12,216 trials, mean 80.37 across 152 sessions. | Mean 80.5 ± 7.4; lick section says 12,376. | Use all locally present valid trials. The 160-trial difference equals two typical 80-trial sessions and likely reflects a manuscript/data-version accounting difference; m11 has no day 1-2 NWB as expected from methods. |
| Lick corruption | Correct trials where cumulative count >2 in >30-35% of frames. | Exactly 81 trials at the paper's >30% threshold (0.663%). | 81 trials (~0.65%). | Exact match. Since categorical lick labels cannot be NaN, exclude these 81 complete trials from the decoder dataset. |
| Reward omissions | Reward event plus reward-zone occurrence defines rewarded trial. | Sparse NWB reward timestamps give 10,342/12,216 rewarded =84.659%; three reward events lie outside track trials and are ignored. | Approximately 15% omitted. | Exact expected rate. Map a reward timestamp in `[start,teleport)` to outcome 1; otherwise 0. Keep omission trials. |
| Reward-zone schedule | Code maps scene A/B/C to X/Y/Z coordinates [80,130], [200,250], [320,370], with switch at trial 30. | Identifiers encode all schedules; parsed totals A=4,186, B=4,010, C=4,020 trials. Environment never changes within a trial. | Same coordinates; each switch after 30 trials. | Parse `NWBFile.identifier`; use first zone for trials 0-29 and second thereafter. |
| Speed threshold | Paper decoders/spatial analyses drop <2 cm/s samples. | Speed includes stationary and low-speed frames required by output class 0. | Threshold is analysis-specific. | Retain all trial frames because the requested decoder explicitly predicts the <2 cm/s category; document as a required task exception. |

Final consistent model: each NWB file is one CA1 session. Use curated, non-speed-correlated neurons; recompute the authors' dF/F and event activity per track trial; concatenate planes by their ROI table mappings; retain all non-corrupt trials and all within-trial speed states; derive schedules from identifiers and reward outcomes from sparse event timestamps.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `Fluorescence`, `Neuropil`, ROI `iscell` | `neural` | Select `iscell[:,0]>0`; compute trial-wise `(F-0.7*Fneu + 0.7*mean(Fneu))`, 15-frame Gaussian, 300-frame min then max baseline, dF/F, 2-frame Gaussian, and OASIS (`tau=0.7`, effective fs=15.5078125); remove dF/F-speed r>0.5 cells; split `[start:teleport)`; transpose to neurons x time; float32. | `preprocessing.dff`, `utilities.multi_anim_sess`, `spatial.is_putative_interneuron` | Pool plane series by ROI table region; use explicit behavior timestamps. No place-cell filter. |
| common behavior timestamps | `input[0]` | Timestamp minus timestamp at trial-start flag, seconds, float32 time series. | `glmUtils.get_timeseries_data` (trial alignment) | Starts exactly at 0; preserves slight timestamp jitter. |
| `environment` | `input[1]` | Unique valid per-trial value, 0=ENV1, 1=ENV2, repeated across time. | `behavior.get_trial_types` (`morph`) | Validate exactly one environment per trial. |
| `trial number` | `input[2]` | Native zero-based continuous trial index, repeated across time, float32. | `behavior.get_trial_types` / NWB documentation | Validate integer and equals chronological trial index. |
| sparse `Reward.timestamps` from raw trial i-1 | `input[3]` | Previous raw chronological trial outcome; omitted=0, rewarded=1; repeat across time. First trial=0 because no previous observation. | `behavior.get_trial_types` | If a corrupt-lick trial is excluded, the following trial still references that actual raw preceding trial. |
| `position`; active zone bounds | `output[0]` | Signed distance to nearest point in zone: `position-start` before, 0 inside closed zone, `position-end` after. Classes: d<-50:0; [-50,-10):1; [-10,0):2; 0:3; (0,10]:4; (10,50]:5; >50:6. | `behavior.get_reward_zones`; task-required transform | “Any location” is interpreted as distance to the closed 50-cm interval, hence an exact 0 class throughout the zone. |
| `position` | `output[1]` | 0: p<90; 1: [90,180); 2: [180,270); 3: [270,360]; 4: p>360. | task-required transform | Tiny boundary overshoots remain in endpoint classes; trial slicing excludes teleport artifacts. |
| `speed` | `output[2]` | 0: v<2; 1: [2,10); 2: [10,20); 3: [20,40]; 4: v>40 cm/s. | task-required transform | Small negative smoothed speeds correctly fall in class 0. No speed-based sample removal. |
| `lick` | `output[3]` | `lick>0` -> 1, else 0, int8. | paper lick quantification / `glmUtils.get_timeseries_data` | Exclude whole trial if >30% samples have cumulative count >2 (81 raw trials). |
| `NWBFile.identifier` | `output[4]` | Parse A/B/C; for `_to_` scenes, first label trials 0-29 and second label trial >=30; encode A=0, B=1, C=2; repeat across time. | `behavior.get_reward_zones` | Zone bounds A=[80,130], B=[200,250], C=[320,370] cm. |
| sparse `Reward.timestamps` | `output[5]` | Any delivery in `[start timestamp, teleport timestamp)` -> 1, else 0; repeat across time. | `behavior.get_trial_types` | Retains omission trials; three delivery events outside track epochs are ignored. |
| `subject.subject_id` | `subjects`, `subject_idx` | Natural numeric subject order; map each session. | NWB metadata | 11 subjects. |
| dorsal hippocampal imaging site | `brain_regions`, `brain_region_idx` | `brain_regions=['CA1']`; zeros for every retained neuron. | paper + DANDI metadata | Both deep/superficial planes are CA1. |

### Key Decisions
1. **Time bin**: Keep one row per synchronized behavior/imaging sample (median 64.483627 ms). Multi-plane NWB series are already resampled/aligned per plane at this interval despite their 31-Hz metadata field. This meets the common-bin requirement without interpolation artifacts.
2. **Trial filtering**: Exclude only the 81 objectively corrupt-lick trials, since lick is a required output and unknown labels cannot be represented. Keep short/long, rewarded/omitted, stationary, and early-terminated trials. Verify every retained session still has at least two trials.
3. **Neurons**: Use all manually curated, non-speed-correlated CA1 neurons, not only place cells. Place-cell/remapping categories are specific to the paper's spatial hypotheses and would discard valid speed/lick information for this broader decoder.
4. **Custom events instead of NWB `Deconvolved`**: The latter is the Suite2p export and does not reproduce the paper preprocessing. Recomputing from PyNWB-loaded F/Fneu is required for consistency.
5. **Shapes**: Every input is `float32 (4,T)` and every output is `int8 (6,T)`. Per-trial categorical/context values are repeated over T because time-varying and per-trial variables coexist in a single rectangular trial array and the validator/trainer supports this directly.
6. **Session ordering**: Natural numeric mouse then numeric session day. `--sample` uses two representative day-8 environment-switch sessions (`m3` day 8 and `m12` day 8) to cover both environments and all A/B/C zone labels across only two sessions.
7. **Neural scale**: Preserve the reference OASIS event amplitudes as float32 without z-scoring or per-cell normalization; the supplied decoder learns session projections, and altering amplitudes would depart from paper processing.
8. **Metadata**: `time_bin_size=64.483627204 ms`, alignment=`trial_start flag / entry to 0-cm track`, `off_start=0.0`, `off_end=None` because lap durations vary. Add session source IDs, raw/retained trial counts, raw/retained neuron counts, sampling intervals, and exclusion counts.

### Planned Sanity Checks
- [x] PyNWB-original versus converted neural: independently recompute selected dF/F/OASIS trials and require `np.allclose`.
- [x] PyNWB-original versus converted input: independently derive time/environment/trial/previous-outcome for selected trials and require `np.allclose`.
- [x] PyNWB-original versus converted output: independently derive all six labels for selected trials and require `np.allclose`.
- [x] Check equal neural/input/output T, finite values, fixed neuron count per session, and >=2 retained trials/session.
- [x] Confirm raw totals (152 sessions, 12,216 trials, 138,678 manually curated session-neurons), exactly 81 corrupt trials, reward rate 84.659%, and expected 11 subjects.
- [x] Confirm discretization boundary functions with synthetic exact values (-50,-10,0,10,50 cm; 90,180,270,360 cm; 2,10,20,40 cm/s).
- [x] Confirm all expected categorical values are present globally and output distributions are plausible; inspect per-session extremes.
- [x] Plot raw F, neuropil-corrected F and maximin baseline, dF/F, events, position/trial bounds, inputs, continuous values with class transitions, and class histograms for two sample sessions.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with the required CLI (`<outpicklefile>`, `--full`, `--sample`, `--show-processing`). It loads all NWB content through `pynwb.NWBHDF5IO`, translates ROI table regions across one/two planes, applies reference dF/F/OASIS processing, filters neurons/trials, constructs all fields and metadata, validates shapes/finiteness/boundaries, prints per-session timings, and writes a highest-protocol pickle. `python3 -m py_compile` and `--help` both completed successfully.

Code inefficiencies identified:
The raw dataset is 92.45 GB and recomputing dF/F can create several hundred MB of intermediate arrays in high-cell sessions. Loading uncurated ROIs or holding raw F/Fneu, dF/F, events, and final trial copies for multiple sessions simultaneously would unnecessarily increase memory and I/O. Per-neuron Python loops for speed correlation would also be costly.

Code speedups added:
Read only manually curated ROI columns; process one plane and one session at a time; vectorize filtering/baseline operations over neurons; compute speed correlations using vectorized sufficient statistics accumulated by trial; delete large plane intermediates after filtered event extraction; retain float32 neural/input and int8 outputs; use sparse reward timestamp search rather than per-frame event expansion. Full conversion remains sequential to avoid competing reads from 92-GB NWB assets; measured runtime will determine whether parallelism is needed.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 2,272 session-neurons after filter (2,277 manual before speed-correlation filter) |
| Neurons / session | 778, 1,494 |
| Subjects | 2 (m3, m12) |
| Sessions / subject | 1 each |
| Trials (total) | 160/160 raw retained |
| Trials / session | 80, 80 |
| Time input range | [0, 67.966] s |
| Environment input range | [0, 1] |
| Trial-number input range | [0, 79] |
| Previous-outcome input range | [0, 1] |
| Distance output distribution | [0.1615, 0.0698, 0.1332, 0.2927, 0.0209, 0.0836, 0.2383] |
| Position output distribution | [0.1459, 0.1220, 0.2760, 0.3009, 0.1552] |
| Speed output distribution | [0.2200, 0.0767, 0.1026, 0.2358, 0.3649] |
| Lick output distribution | [0.7712, 0.2288] |
| Reward-zone output distribution | [0.2046, 0.5160, 0.2794] |
| Reward-outcome output distribution | [0.1913, 0.8087] |

`sample_data.pkl` is 135 MiB with 34,297 trial timepoints. Validator result: valid format, no errors, no warnings; all requested classes appear globally and all neural/input/output arrays have equal T and finite values.

### Processing Plots Review
Reviewed `processing_m3_ses-08.png` and `processing_m12_ses-08.png`. Raw F, Fneu, neuropil-corrected F, stable maximin baselines, smoothed dF/F, and sparse OASIS events are visually coherent. Position begins near 0 and ends near 450 cm without teleport artifacts; time starts at zero; environment/zone transitions are internally consistent; position, distance, speed and lick classes change at the expected continuous-value boundaries. No temporal offsets or anomalies were found. m3 excluded 5/783 speed-correlated cells; m12 excluded 0/1,494.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Curated-column loading, vectorized filters/correlation, per-plane processing | Sample finished in 5.62 s; avoided reading/processing 2,616 uncurated sample ROIs. |

| Step | Time / Session | Estimated Total Time |
| Sample neural conversion and plots | 2.75 s average | 5.49 s processing for 2 sessions |
| Full neural conversion, workload-scaled | varies with neuron x track-frame count | 377 s (~6.3 min) |
| Full write/overhead allowance | sample write 0.13 s; full file much larger | <2 min estimated |

Runtime scaling used curated neuron x track-frame workload: 2.418 billion full versus 35.17 million sample units (68.75x). Estimated full total is under 9 minutes, below the 15-minute optimization threshold.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Distance to reward zone | 0.6744 | 0.5830 |
| Absolute position | 0.8255 | 0.7075 |
| Speed | 0.6623 | 0.6013 |
| Lick | 0.8118 | 0.8337 |
| Reward zone location | 0.9567 | 0.9517 |
| Reward outcome | 0.7925 | 0.5913 |

Training completed on CUDA. Loss decreased monotonically from 2.831717 (epoch 1) to 0.741961 (epoch 200); test loss was 0.743353. Every validation balanced accuracy exceeded uniform chance (respectively 1/7, 1/5, 1/5, 1/2, 1/3, 1/2), supporting correct alignment and labels. Reward outcome is expected to be the hardest: outcome is randomized on ~15% of trials and is not revealed by the allowed current-trial inputs.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 9,518,422,526 bytes (8.86 GiB)
- `conversion_full_out.txt`: created; conversion completed in 301.43 s (including 8.43 s pickle write)
- `verification_full_out.txt`: created; validator reported "Data format is valid" and no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Manual/retained neurons | 155-2,172 cells/session stated | `iscell[:,0]` then speed-correlation filter | 138,678 manual cell session-recordings, 155-2,341/session | 138,276 retained, mean 909.71/session, 154-2,323/session | Yes after 402 (0.290%) reference-defined interneuron exclusions; release maximum differs from manuscript |
| Subjects | 11 switch mice | 11 entries in `MetaLearn` | 11 | 11 | Yes |
| Sessions | Up to 14/mouse; m11 begins day 3 | 152 rows in `MetaLearn` | 152 | 152 | Yes |
| Trials (total) | 12,376 quoted for lick QC; mean 80.5/session | trial start/teleport boundaries | 12,216 in this NWB release | 12,135 | Yes: exactly the 81 paper-defined corrupt-lick trials were removed; manuscript/release accounting differs by 160 raw trials |
| Trials/session | 80.5 mean stated | no arbitrary count filter | 80.37 mean raw | 79.84 mean retained; minimum 40 | Yes after lick QC |
| Reward rate | approximately 85%; omissions approximately 15% | `reward_delivery`/Reward at trial end | 10,342/12,216 = 84.659% | 10,271/12,135 = 84.639% trial-weighted | Yes |
| Input ranges | task has ENV1/ENV2 and 30-trial blocks | native environment/trial streams | time-varying values in NWB | time [0,216.5] s; environment [0,1]; trial [0,99]; previous outcome [0,1] | Yes |
| Distance-class distribution | no categorical distribution reported | zones A/B/C: 80-130/200-250/320-370 cm | deterministically derived | [0.251,0.102,0.073,0.238,0.021,0.072,0.243] | Yes by independent derivation |
| Position-class distribution | 450-cm corridor | position stream | deterministically derived | [0.212,0.177,0.231,0.226,0.154] | Yes by independent derivation |
| Speed-class distribution | running measured continuously | speed stream | deterministically derived | [0.117,0.087,0.134,0.319,0.343] | Yes by independent derivation |
| Lick distribution | lick sensor; 81 corrupt trials | lick count stream | retained valid trials | [0.777,0.223] | Yes by independent derivation |
| Zone distribution | A/B/C zones | scene/reward-zone mapping | retained trial counts A/B/C = 4,172/3,974/3,989 | timepoint-weighted [0.332,0.336,0.333] | Yes |
| Outcome distribution | approximately 15% omissions | Reward stream | retained trial counts omitted/rewarded = 1,864/10,271 | timepoint-weighted [0.158,0.842] | Yes |

The final dataset contains 2,576,026 within-trial time bins. Trial length is 215.53 bins on average (median 195.02, range 96-3,359), using the 64.483627-ms common behavior/imaging sampling interval.

### Conversion iteration and spot checks

The first full run stopped at m17 session 4 because its ophys matrices contained one more terminal frame than the behavior streams. A PyNWB audit found this exact +1 condition in ten dual-plane sessions: m17 sessions 4 and 6, and m18 sessions 1, 5, 7, 10, 11, 12, 13, and 14. In every case both streams began at time zero and the surplus ophys row was after the last teleport, hence outside every retained trial. The loader was tightened to allow and discard only this precisely characterized single terminal row, while raising on any other mismatch. The restarted full conversion and verification both completed. Session/trial counts, all class ranges, and several early/middle/late sessions were then spot-checked; no data inside a trial were lost.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output-log verification**: Read all of `verification_full_out.txt`. It states "Data format is valid, no errors or warnings" and ends normally with "Data verification complete." All 152 per-session shape/range summaries were present. There are no warnings requiring waiver.
2. **Independent original-data comparisons**: `sanity_checks.py` loads the original files directly with `pynwb.NWBHDF5IO` and deliberately does not import `convert_data.py`. It independently recomputes trial-wise F/Fneu correction, maximin dF/F, OASIS events, speed correlations/curation, trial bounds, reward outcomes, four inputs, and six outputs. `np.testing.assert_allclose` passed for full neural/input/output arrays on 13 trials across m3 session 1, m4 session 14, m12 session 8, and dual-plane m18 session 7. These cover first/middle/last trials, a trial immediately after an excluded corrupt trial, an ENV/zone switch session, extensive lick corruption, and a +1 terminal-ophys-row session. Exact results are in `sanity_checks_out.txt`.
3. **Reference loading comparison**: Reference `utilities.multi_anim_sess` loads synchronized session arrays; conversion `process_session` instead accesses the released NWB through PyNWB, which is the required equivalent. It selects `iscell[:,0]` exactly as `multiDayROIAlign.py` and NWB ROI regions require. No direct HDF5 access occurs.
4. **Reference neuron/trial filtering comparison**: Conversion uses manual cells and the `spatial.is_putative_interneuron` criterion dF/F-speed Pearson r>0.5. It intentionally does not call `calc_place_cells`, since place-cell status is an analysis subset rather than general-quality curation. Track epochs are retained except the exact paper criterion of >30% frames with cumulative lick count >2; all 81 such raw trials are removed. Unlike paper spatial/continuous decoding, frames below 2 cm/s remain because speed class 0 and time-varying lick are required outputs.
5. **Reference temporal-alignment comparison**: Reference `preprocessing.dff` and `glmUtils.get_timeseries_data` use `start-1:stop-1` on legacy one-based event indices. Conversion uses the semantically corresponding NWB `[trial_start, teleport)` flags. Independent raw checks showed every audited trial begins below 15 cm, ends above 425 cm, has time exactly zero at its first sample, and excludes the teleport row.
6. **Reference neural binning comparison**: Conversion matches `preprocessing.dff`: 0.7 neuropil subtraction plus 0.7 trial mean, Gaussian sigma 15, 300-frame minimum then maximum baseline, division by absolute baseline, Gaussian sigma 2, and Suite2p OASIS with tau 0.7 at 15.5078125 Hz. It keeps native synchronized 64.483627-ms rows; there is no spike-count binning or interpolation. Independently recomputed complete trial matrices pass `np.allclose`.
7. **Input-construction comparison**: Time uses explicit timestamps rather than assuming perfectly uniform samples. Environment and native zero-based trial number are read from their frame streams. Previous outcome is based on the actual preceding raw trial, including when that trial is excluded for corrupt licking; m4 session 14 converted trial 12/raw trial 13 explicitly passed this edge case.
8. **Output-construction comparison**: Zones and the trial-30 switch follow `behavior.get_reward_zones`; reward events follow `behavior.get_trial_types` semantics. Position/speed/distance bins implement the requested inequalities; exact equality-side tests passed. Lick is reference-style `>0` binary. All categorical classes are globally represented.
9. **Key-statistics comparison**: Confirmed 11 mice, 152 sessions, 12,216 raw trials, exactly 81 corrupt trials, 12,135 retained trials, 138,678 manual cell session-recordings, 402 speed-correlated exclusions, 138,276 retained neurons, and 84.639% retained-trial reward rate. Paper-compatible values are 11 mice, approximately 80.5 trials/session, approximately 15% omissions, 81 corrupt trials, and a 0.42+/-0.85% interneuron exclusion. The paper/release differences (12,376 versus 12,216 trials; maximum 2,172 versus 2,341 manual cells) are upstream release/accounting differences, not conversion losses, as documented in Steps 4 and 9.
10. **Edge-case audit**: Minimum retained trials/session is 40, safely above two. Trial starts/stops are paired; source trial numbers equal raw indices; environments are constant and valid within trial; output bin boundaries were tested at equality; first-trial previous outcome is zero; omitted trials remain; removed-trial chronology remains intact; both planes are pooled; ten precisely identified sessions discard only one unmatched post-behavior terminal ophys row. All trial arrays are finite and have equal neural/input/output T.

### Issues Found and Resolved
- **NWB `Deconvolved` mismatch**: Found during cross-source review to be raw Suite2p output rather than the paper's trial-wise processing. Resolved by recomputing reference dF/F/OASIS from PyNWB-loaded Fluorescence/Neuropil; independent checks pass.
- **Dual-plane terminal length mismatch**: The initial full run stopped rather than guessing. PyNWB inspection showed an exact one-row, post-final-teleport surplus in ten sessions. Resolved by permitting only +1 terminal row and truncating it; any other mismatch remains fatal. Full conversion, validator, and independent dual-plane checks then passed.
- **No remaining issues**: Re-running the complete independent suite after adding corrupt-trial chronology coverage produced all passes. No conversion change or re-run was needed after this final review because it found no mismatch in the completed output.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. CUDA training completed all 200 epochs; loss fell monotonically from 3.291559 to 1.002306, and test loss was 0.880729. The run ended with `train_decoder.py finished successfully.` Sample/prediction plots were produced.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Distance to reward zone | 0.5649 | 0.4982 | chance 0.1429 |
| Absolute position | 0.6609 | 0.6098 | chance 0.2000 |
| Speed | 0.5841 | 0.5416 | chance 0.2000 |
| Lick | 0.7488 | 0.7249 | chance 0.5000 |
| Reward zone location | 0.8892 | 0.8462 | chance 0.3333 |
| Reward outcome | 0.7277 | 0.5523 | chance 0.5000; omissions are randomized |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Validation balanced accuracy | Chance | Multiple of chance | Expectation from paper |
|----------|------------------------------|--------|--------------------|------------------------|
| Distance to reward zone | 0.4982 | 0.1429 | 3.49x | Closest to paper RR-position decoder, but paper reports cosine decode score relative to shuffle rather than categorical accuracy; RR decoder is significantly above shuffle. |
| Absolute position | 0.6098 | 0.2000 | 3.05x | No corresponding categorical accuracy reported. |
| Speed | 0.5416 | 0.2000 | 2.71x | No corresponding decoder reported. |
| Lick | 0.7249 | 0.5000 | 1.45x | No corresponding decoder reported. |
| Reward zone location | 0.8462 | 0.3333 | 2.54x | No corresponding decoder reported. |
| Reward outcome | 0.5523 | 0.5000 | 1.10x | No corresponding decoder reported; reward omissions are randomized at approximately 15%. |

**Accuracy versus chance**: Every validation accuracy is above chance. Distance, position, speed, and zone exceed 1.5x chance. Lick is narrowly below the conservative 1.5x screen (1.45x), and outcome is below it (1.10x), so both received the complete low-accuracy investigation below.

**Every paper decoding result found**: The paper has one decoder family: circular reward-relative position from RR, track-relative, or non-RR cells. Its timepoint score is `cos(true-predicted)` (1 perfect, 0 random), evaluated against circular-shift shuffles in 77 switch sessions. The text does not tabulate raw mean scores or classification accuracies. It reports pre-to-pre effect sizes RR=2.6, TR=3.3, non-RR=3.2; pre-to-post RR=2.7, TR=-0.5, non-RR=-0.2; `P=7.54e-14` for the significant comparison; and cross-switch RR decoding z>2 over -104.5+/-20.1 to +152.7+/-22.9 cm. Because the supplied decoder's seven-class signed-distance accuracy is a different target, population, split, and metric, no numeric equality is possible. Its 0.4982 balanced accuracy (3.49x chance) is nevertheless consistent with the paper's core result that reward-relative position is neurally decodable.

**Train-validation gaps**: Ratios are distance 1.13, position 1.08, speed 1.08, lick 1.03, zone 1.05, and outcome 1.32. None exceeds the specified 1.5x threshold. There is no evidence of severe overfitting or leakage. Trial-level splitting and outcome's larger but sub-threshold gap are expected for a random, imbalanced event.

**Low-accuracy investigation**:

1. Re-ran `sanity_checks.py` after training. Thirteen named raw trials were reloaded directly with PyNWB and independently processed. Full neural, input, and output arrays pass `np.allclose`; examples include m18 session 7 raw trial 40 (omitted outcome=0, lick fraction 0.2510), raw trial 0 (rewarded, lick fraction 0.1894), and raw trial 79 (rewarded, lick fraction 0.1190). m4 session 14 raw trial 13 additionally proves correct previous-outcome chronology after an excluded trial.
2. Inspected `sample_trials.png` and `predictions.png`. Neural events and outputs share exactly the same trial axis; position/distance increase coherently, speed changes with traversal, licks occur as frame-level events, and per-trial zone/outcome remain constant. No one-bin offset or teleport artifact is visible.
3. Target variation is ample: lick classes are [0.777,0.223] and outcome classes [0.158,0.842] by timepoint, far from 99% dominance. At trial level, 1,864/12,135 are omissions and 10,271/12,135 are rewards.
4. Neural filtering was rechecked against paper code: manual `iscell`, trial-wise dF/F/OASIS, both imaging planes pooled, then r>0.5 speed-correlated cells excluded. Independent neural reconstruction passed.
5. Processing/alignment was rechecked against reference code and raw position endpoints as described in Step 10. No mismatch was found.

The remaining outcome difficulty is scientifically expected, not a label workaround: omission is randomized, the outcome label is repeated over the entire trial, and pre-reward-zone neural samples cannot contain information about a future random delivery. Lick is a sparse, rapidly changing behavioral event and still reaches 0.7249 balanced accuracy. Changing labels, removing pre-outcome samples, or injecting reward/lick into inputs would inflate scores through leakage and violate the requested mapping.

### Issues Found and Resolved
- **Potential low lick/outcome accuracy**: Fully investigated using raw-label checks, neural/output alignment plots, class variation, filtering, and reference processing. No conversion issue found; no inappropriate score-driven transformation was made.
- **Paper comparison metric mismatch**: Resolved by enumerating all reported paper decoder statistics and explicitly comparing the closest requested output on a chance-normalized basis rather than treating cosine scores/effect sizes as categorical accuracies.
- **Final result**: All accuracies are above chance, no train/validation ratio exceeds 1.5, required plots are coherent, and re-run raw-data checks all pass. No further conversion iteration is warranted.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with dataset description, loading example, format, processing, statistics, results, and reproduction commands.
- [x] cache/ folder created; `README_CACHE.md` documents the independent audit script and captured output.
- [x] All files organized. `sanity_checks.py` and its output were moved to `cache/`; required datasets, logs, code, and user-facing plots remain at the project root.

Final required-file audit found every requested artifact. `convert_data.py` and `cache/sanity_checks.py` compile successfully. The complete 9,518,422,526-byte pickle passes the supplied validator without warnings, independent PyNWB comparisons pass, and the supplied full decoder finishes all 200 epochs with every output above chance.
