# Dataset Conversion Notes

## Overview
- **Dataset**: Brain-wide neural activity underlying memory-guided movement (provided NWB data)
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `ChenLiuEtAl2023_SpikeSortingQC.pdf`
- `Dockerfile`
- `code/`
- `data/`
- `datapaper.pdf`
- `decoder.py`
- `docker-compose.yaml`
- `methodpaper.pdf`
- `methods.txt`
- `pynwb_docs/`
- `train_decoder.py`

Environment verified: Python 3.13.15; NumPy 2.4.4; PyTorch 2.6.0+cu124; PyNWB 4.1.0. Checkpoint confirmed with `ls -la /app/CONVERSION_NOTES.md`.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `process_all_sess_parallel` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING | Groups probe files by session and preprocesses sessions in parallel. |
| `process_one_sess` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING | Loads behavior/task, spikes, unit QC, histology; combines probes; aligns lick/stimulation times to go cue. |
| `helper_get_neuron_id_area` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | CURATION | Intersects classifier-selected good-unit IDs with hemisphere and valid histology/CCF annotation. |
| `sliding_histogram` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Counts spikes in half-open sliding windows and divides by bin width to obtain Hz. |
| `process_one_area` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Truncates go-cue-relative spikes, bins firing rates, and stores per-area neural/task metadata. |
| `load_session` | `code/VideoAnalysisUtils/population_decoding_utils.py` | LOADING | Concatenates processed area files across neurons into one session tensor and CCF arrays. |
| `get_regular_trial_mask` | `code/VideoAnalysisUtils/population_decoding_utils.py` and `functions_for_r2.py` | CURATION | Defines regular trials: no early lick, auto/free water, no-response, or stimulation. |
| `create_4fold_trial_type_mask` | `code/VideoAnalysisUtils/functions_for_r2.py` | PROCESSING | Stratifies trials as hit/miss crossed with left/right; pools sparse miss groups. |

### Notes
- The repository README identifies the source as DANDI 000363 and the paper as *Brain-wide analysis reveals movement encoding structured across and within brain areas*.
- This is extracellular electrophysiology, not imaging; delta-F/F is not applicable.
- Reference preprocessing uses classifier QC (`qc_mode='classifier'`). Good units are supplied by session-level QC files, restricted to units with matched ephys/histology and one of 14 broad regions, and split at CCF ML coordinate 5700 into hemispheres.
- Raw spike times are already relative to go cue. Lick and laser times originally referenced to trial start are shifted by subtracting the go-cue timestamp.
- The reusable binning definition is `[center-width/2, center+width/2)` and firing rate is spike count / width. The method-paper production run used 40-ms windows with 3.4-ms stride over -3 to +3 s (`Sherlock/preprocess_all_ephys.py`); another script default uses 100-ms width/50-ms stride. The decoder requirement overrides these with non-overlapping 50-ms bins from -2.5 to +1.5 s.
- Important source behavior fields: early report, auto/free water, lick directions/times, correctness (1 hit, 0 error, -1 no response), go cue, sample/delay durations, stimulation `[power,type,on,off]`, and instructed trial type.
- Reference population analyses generally exclude early-lick, auto-water, free-water, no-response, and stimulation trials. For this requested decoder, early lick, no-lick outcome, and photostimulation are explicit targets/inputs, so those categories cannot be discarded merely by applying the regular-trial mask; this necessary divergence will be resolved after inspecting NWB availability.
- `load_session` concatenates all areas within a session along neurons and preserves per-neuron CCF labels, matching the target session organization.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/dandiset.yaml` identifies DANDI 000363 version 0.230822.0128, 53,585,631,708 bytes (~50 GiB), mouse electrophysiology/behavior, 174 NWB assets, and 28 subjects.
- Files are organized as `sub-<subject>/sub-<subject>_ses-<timestamp>_behavior+ecephys[+ogen].nwb`. All native-data inspection was performed through `pynwb.NWBHDF5IO`; no HDF5 parser was used.
- All 174 files have the same 14-column `trials` table: start/stop, trial number, photostim onset/power/duration, globally unique trial ID, task/protocol, left/right instruction, early-lick label, outcome, auto-water, and free-water. Task is uniformly `audio delay`.
- `acquisition/BehavioralEvents` holds absolute timestamps for presample/sample/delay/go/trial-end boundaries, left/right licks, and (where present) photostimulation start/stop. Event-list counts can exceed trial counts for malformed/repeated state-machine events, so trial association must be based on trial intervals rather than positional indexing except for the one-go-per-trial stream after verification.
- `acquisition/BehavioralTimeSeries` has side-camera Jaw, Nose, and Tongue tracking in every session; each is `(samples,3)` = x, y, likelihood with explicit timestamps. Some sessions add whisker/lick-port or Camera3 duplicates; Camera0 side tongue is consistent across all sessions.
- `units` contains global session-clock spike times plus spike-sorting metrics, `classification`, detailed Allen CCF `anno_name`, per-unit valid-trial flags, observation intervals, electrode links, and waveforms. Raw units total 272,227. Classifier-good and annotated units normally coincide. One session (`sub-440958_ses-20190216T162508...`) has `classification='nan'` for all 1,852 units but all have nonempty annotations; thus 71,305 nonempty annotated units versus 69,453 literally labeled `good`. This anomaly requires reference-text/code reconciliation before final curation.
- There are four acquisition-schema variants caused only by optional tracked features/cameras; required trial, neural, event, and Camera0 tongue streams are present in all sessions. All 174 files opened successfully.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 272,227 raw; 69,453 `classification=good`; 71,305 with nonempty CCF annotation |
| Neurons / session | Raw: 493-3,191, mean 1,564.5, median 1,571.5; literal classifier-good: 0-923, mean 399.2, median 390 |
| Subjects | 28 |
| Sessions / subject | 3-10 (174 sessions total) |
| Trials (total) | 94,990 |
| Trials / session | 264-800, mean 545.9, median 534 |

Native categorical totals: outcome = 65,254 hit / 15,641 miss / 14,095 ignore; early lick = 84,185 no / 10,805 yes; instruction = 46,077 left / 48,913 right. These are pre-curation counts.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 69,943 good units (25.9% of Kilosort2 clusters) | Data paper Methods and QC white paper: “69,943 good units recorded across 173 behavioral sessions” |
| Neurons / session | Not stated as a distribution | — |
| Subjects | 28 mice | Both papers’ Methods |
| Sessions / subject | 173 behavioral sessions total; 655 probe insertions (QC white paper; data-paper text elsewhere says 660 penetrations) | QC/Methods |
| Trials (total) | Not stated | — |
| Trials / session | Mean 476; range 130–785 | Data paper Methods |
| Neural data time bin | Method paper: 40-ms width, 3.4-ms stride; data paper varies by analysis (40-ms/17-ms for video-to-neural, 200-ms/10-ms for population choice decoding) | Methods sections |
| Behavior data time bin | Video acquired at 300 Hz (~3.33 ms/frame) | Both papers |
| Reward rate | 84% correct mean, 65–99% session range on selected control/non-early trials | Data paper Methods |
| Session eligibility | >65% overall performance and >=50 correct left and >=50 correct right trials | Data paper Methods |
| Photostimulation prevalence | ~25% randomly interleaved trials in 17 VGAT-ChR2 mice | Data paper Methods |
| Bilateral photostim performance | 83.2% control to 71.7% stimulated; 17 mice, 93 sessions | Data paper Methods |


### Processing Details
- Task: three 150-ms presentations of either 3-kHz or 12-kHz tone separated by 100-ms gaps; then 1.2-s delay; 100-ms go cue; 1.5-s answer period. Early licking replays the sample/delay epoch.
- Spikes were Kilosort2 sorted. The method paper bins firing rates at 40-ms width/3.4-ms stride; requested downstream specification instead fixes 50-ms width and the -2.5-to-+1.5-s go-aligned interval.
- Video/marker analysis uses side view, 300-Hz tracking. Reference marker cleaning detects velocity outliers at five SD and imputes from nearby frames. Occluded/in-mouth tongue positions were replaced by their mean for neural-prediction analyses. The requested decoder instead explicitly requires a `not visible` class, so visibility must be retained rather than mean-imputed.
- Photoinhibition occurs in the last 0.5 s of delay and ends before go cue (including 100-ms ramp-down). This provides a strong timing sanity check for the binary photostimulation input.

### Curation Steps

**Neuron curation rules**:
Use region-specific logistic-regression QC classifiers trained from manually curated Kilosort2 clusters using 15 quality metrics. “Good” classifier units were used in paper analyses. Cortex classifier covered cortex/hippocampus/olfactory/cortical subplate; striatum classifier covered striatum/pallidum; thalamus covered thalamus/hypothalamus; midbrain covered midbrain/pons; medulla covered medulla/cerebellum. QC ROC AUC averaged >0.9. The method paper additionally excluded <2-Hz neurons for its video-to-neural regression, but this is analysis-specific and not a global dataset curation rule.

**Trial curation rules**:
Data-paper core analyses exclude early-lick and no-response trials and select high-performance sessions. Method-paper analyses exclude photoinhibition, free-water, early-lick, and ignore trials; behavioral prediction further uses only correct trials. These exclusions directly conflict with requested decoding of photostimulation, early lick, and ignore/no-lick outcomes, so retaining those trials is required unless a trial lacks the required alignment/data window. Auto/free-water trials require careful treatment because outcome/choice semantics may be non-volitional.

### Decoders Trained
| Decoded variable | Accuracy |
| Population lick choice | Logistic regression, 200-ms causal windows/10-ms step, nested 5-fold CV; the paper reports time-varying figure curves rather than a directly comparable single scalar. Chance = 0.5; strongest late-delay performance is in ALM. |
| Behavioral choice from video | ROC AUC with 20-fold stratified CV on correct trials; figure curves, not a scalar directly comparable to this four-output decoder. |

No reference decoder predicts the requested three-way outcome, binary early lick, or four-way tongue-y class in the same formulation, so there is no paper accuracy directly comparable to those outputs.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Session count | Preprocessing processes every raw session that has QC files | 174 NWBs; exactly one has no `good` classifier labels | 173 behavioral sessions | Exclude the zero-good-unit session naturally because it cannot yield a neural decoder session; remaining session count is exactly 173. |
| Good-unit count | Classifier-QC IDs intersected with valid CCF/histology | 69,453 units labeled `classification='good'`; all have nonempty `anno_name` | 69,943 good units | Use the release’s explicit classifier labels, not reverse-engineered metric thresholds or the paper aggregate. The 490-unit (0.7%) difference is attributed to data-release/code-version differences and documented. Do not include the anomalous session’s 1,852 units, whose classification is `nan`. |
| Unit region | Code saves both a broad side/area filename and detailed `ccf_label` | NWB has detailed `anno_name`; probe target is available through electrode-group location | Paper reports broad-area totals and uses Allen subregions in later analyses | Preserve detailed Allen annotation as neuron region, which is the anatomically measured location and supports later aggregation; do not substitute probe target for histology. |
| Neural binning | Reference method paper: 40-ms width / 3.4-ms stride, half-open windows | Global spike timestamps in NWB | Requested 50-ms-width bins, -2.5 to +1.5 s | Keep reference half-open counting/rate logic but use 80 non-overlapping 50-ms bins as the explicit decoder override. |
| Trial filtering | `get_regular_trial_mask`: excludes early, auto/free water, ignore, stimulation | All requested categories exist, including 10,805 early and 14,095 ignore trials | Papers exclude them for their analyses | Retain early, ignore/no-lick, and stimulated trials because they define requested labels/input. Auto/free-water semantics will be retained only if requested labels remain defined; record flags in metadata and validate distributions. |
| Tongue missingness | Method paper mean-imputes occluded tongue after five-SD velocity outlier cleanup | NWB exposes raw DLC y and likelihood | Requested class 3 = not visible | Preserve invisibility as class 3; do not mean-impute invisible values. Apply reference velocity-outlier concept only if needed, and use DLC confidence/finite position to establish visibility. |
| Alignment | Raw reference export spikes already go-aligned; lick/laser shifted to go | NWB stores all streams in absolute session time | Both papers organize epochs relative to go | Use each NWB trial’s absolute go event, then subtract it consistently from spike, event, and video timestamps. Associate events by trial intervals to avoid state-machine replay count mismatches. |
| Trial-count expectation | Code filters according to each downstream analysis | NWB raw mean 545.9, range 264–800 | Paper says mean 476, range 130–785 after experimental selection/definitions | Do not force counts to the publication aggregate: requested outputs require trials excluded by paper analyses, and the archived release differs. Report raw/retained counts and exclusion reasons explicitly. |

Final consistent understanding: each retained session is one NWB file; classifier-good, CCF-annotated units are concatenated within that session; global timestamps are aligned to the trial’s go onset; requested categories necessitate broader trial retention than either reference analysis; and the target’s 50-ms bins replace the papers’ analysis-specific sliding bins while retaining their counting convention.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `units.spike_times` for `classification == 'good'` | `neural` | Count spikes in 80 half-open, non-overlapping 50-ms bins spanning edges [-2.5,+1.5] relative to go; divide by 0.05 s; float32 `(neurons,80)` | `sliding_histogram`, `process_one_area` | One session per NWB; global timestamps shifted by trial go timestamp. |
| Last `sample_start_times` event before go in the same trial | `input[0]` | For each bin center, seconds since tone/sample onset: `(go + relative_bin_center) - tone_onset` | `process_one_sess` event alignment | Last onset handles early-lick state-machine replays and yields the final completed sample epoch; normal onset is ~1.85 s before go. |
| `photostim_start_times` / `photostim_stop_times` | `input[1]` | 1 where absolute bin center is in `[laser_on, laser_off)`, else 0 | `process_one_sess` stimulation alignment | Pair events within each trial interval; expected during final 0.5 s before go and off by go. |
| `trial_instruction` + `outcome` | `output[0]` choice | ignore -> no lick; hit -> instructed side; miss -> opposite side | Reference behavior fields in `process_one_sess` | More reliable task-defined response than isolated lick-event glitches; audited against response licks and agrees overwhelmingly. Values: left=0, right=1, no lick=2. |
| `trials.outcome` | `output[1]` outcome | Direct categorical map ignore=0, miss=1, hit=2 | `correctness` mapping in reference loader | Retain ignore trials because explicitly requested. |
| `trials.early_lick` | `output[2]` early lick | `no early`=0, `early`=1 | `early_lick_trials` in reference loader | Retain early trials because explicitly requested. |
| Camera0 side `TongueTracking[:,1:3]` | `output[3]` tongue y | Match nearest 300-Hz frame to each bin center. Visible iff finite and DLC likelihood >=0.9; visible y is discretized with session-visible 40th/60th percentiles; invisible=3. | Method paper marker preprocessing | Confidence is sharply bimodal near 0/1, making 0.9 a stable visibility threshold. Threshold percentiles exclude invisible garbage coordinates. |
| `subject.subject_id` | `subjects`, `subject_idx` | Sorted unique strings and per-session indices | — | 28 subjects expected. |
| `units.anno_name` | `brain_regions`, `brain_region_idx` | Sorted unique detailed Allen CCF labels and per-neuron indices | `ccf_label` in reference preprocessing | Classifier-good units all have annotations. |

### Key Decisions
1. **Session/unit curation**: Retain only explicit classifier-good units. This yields 173 sessions automatically and follows the reference QC rather than inventing metric thresholds. Do not apply the method-paper’s analysis-specific <2-Hz cutoff.
2. **Trial retention**: Retain every trial with exactly one go cue and all required streams; all 94,990 raw trials pass the one-go test, and the 173 neural-valid sessions contain 94,370. Requested categories would be destroyed by the reference “regular trial” mask. Auto/free-water trials remain because their task-table choice/outcome/early labels are defined; metadata will report them.
3. **Choice definition**: Use outcome crossed with instruction. Across all raw trials this agrees with the first response-period lick for >99.6%; discrepancies are predominantly auto/free-water cases or sparse event-log glitches. It also guarantees the semantically required no-lick label for ignore trials.
4. **Mixed output temporality**: Store all four outputs as `(4,80)` int arrays. Repeat per-trial choice/outcome/early labels across time and vary tongue class by bin. A single 2-D representation is required to combine per-trial and time-varying outputs in the validator/trainer.
5. **Tongue cleaning/visibility**: Use DLC likelihood >=0.9, finite y, and nearest frame. The paper’s invisible-tongue mean imputation is deliberately not used because class 3 must preserve missing visibility. Percentiles are computed once per session over visible y samples. Five-SD velocity outliers will be inspected in sample plots; because the requested target is discretized and percentile-robust, any cleaning will be added only if it changes sampled-bin classes materially.
6. **Binning/alignment**: Bin edges are exactly `np.linspace(-2.5,1.5,81)` and centers end at +1.475 s, avoiding an off-by-one endpoint. Spike counts use `[left,right)`, identical to reference logic, and are converted to Hz.
7. **Time input semantics**: Follow the explicit Decoder Task (“time from tone onset in seconds, continuous, time-varying”) rather than the generic key-format suggestion that event times be binary.
8. **Storage types**: Neural and inputs float32, outputs and region/subject indices compact integer types. This controls full-pickle size while preserving exact count-derived rates and labels.
9. **Names/value order**: `input_names = ['time from tone onset (s)', 'photostimulation on']`; `output_names = ['lick direction choice','outcome','early lick','tongue y-position']`; values respectively `[['left','right','no lick'], ['ignore','miss','hit'], ['no','yes'], ['below 40th percentile','40th to 60th percentile','above 60th percentile','not visible']]`.
10. **Metadata**: 50 ms, go-cue onset, offsets -2.5/+1.5 s, 80 centers, half-open bins, firing-rate units Hz, QC/trial retention, DLC threshold, percentile definition, event association, per-session thresholds/counts, and source paths/version.

### Planned Sanity Checks
- [ ] PyNWB-original neural spot check: independently count raw spikes for selected unit/trial/bin and compare converted Hz with `np.allclose`.
- [ ] PyNWB-original input spot check: independently compute time-since-tone and laser-center membership for selected trial/bins and compare with `np.allclose`.
- [ ] PyNWB-original output spot check: independently map trial table labels and nearest tongue frame/category using saved session thresholds; compare with `np.allclose`.
- [ ] Verify every trial has one go and at least one sample onset; paired laser events; all extraction centers within available tongue timestamp range.
- [ ] Verify 173 sessions, 28 subjects, 69,453 total session-neuron counts, and 94,370 retained trials unless a precisely logged data-quality exclusion is necessary.
- [ ] Neural checks: nonnegative finite rates, multiples of 20 Hz, plausible mean firing rates, no unexpected all-zero session/unit.
- [ ] Input checks: tone-time increases exactly 0.05 s/bin; laser is binary; laser timing primarily -0.5-to-0 s and ends by go as described.
- [ ] Output checks: values fall in declared ranges; choice is no-lick iff outcome ignore; visible tongue classes approach 40/20/40 session fractions before time-window sampling; class 3 tracks low DLC confidence.
- [ ] Shape checks: every neural/input/output trial has 80 time points; fixed neurons per session; >=2 trials/session; region indices match neuron count.
- [ ] Compare full converted counts/distributions against NWB scan and all paper statistics, documenting release/task-required deviations.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with the required CLI modes, PyNWB-only reads, classifier unit selection, go/tone/laser trial association, vectorized 50-ms spike binning, task-label construction, nearest-frame tongue discretization, per-session metadata, structural assertions, timing logs, and six-panel processing plots. Syntax compilation and CLI parsing passed.

Code inefficiencies identified:
Naive nested unit x trial x bin spike counting would invoke millions of Python operations. Retaining full raw unit/video tables beyond a session would also inflate memory.

Code speedups added:
For each unit, all absolute bin edges across trials are flattened and passed to one `np.searchsorted`, then differenced into all trial/bin counts. Video frames are matched in one vectorized nearest-neighbor search. Only one NWB/session is open at a time; compact float32/int8 arrays are retained.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 834 session-neurons |
| Neurons / session | [459, 375] |
| Subjects | 1 |
| Sessions / subject | 2 |
| Trials (total) | 527 retained from 848 NWB behavior trials |
| Trials / session | [368, 159] |
| Time from tone onset range | [-0.6, 5.7] s |
| Photostimulation range | [0, 1] |
| Choice distribution | [0.452, 0.408, 0.140] |
| Outcome distribution | [0.140, 0.298, 0.562] |
| Early lick distribution | [0.943, 0.057] |
| Tongue class distribution (timepoints) | [0.053, 0.029, 0.062, 0.857] |

### Processing Plots Review
Both required plots were created and visually inspected. Raster/rate panels contain plausible sparse 20-Hz increments with go/tone lines correctly placed; continuous tone-time rises at exactly 0.05 s/bin; laser input is binary; raw tongue values, bimodal confidence, percentile lines, and output class transitions agree visually. No temporal shift or discretization anomaly was found.

Initial validation exposed all-zero late trials in session 2: its NWB contains 480 behavioral trials but population spike recording ends after trial 159. An attempted all-unit observation-interval intersection was too strict and removed valid trials/entire miss classes because per-unit accepted intervals differ. Final resolution matches reference binning: bin all classifier-good units, then exclude only trials with zero spikes across the complete retained population/window. Re-conversion retains 368 and 159 trials and produces no verifier warnings.

Format verifier: valid, no errors or warnings; every trial has neural `(n_neurons,80)`, input `(2,80)`, output `(4,80)`.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| One `searchsorted` per unit over all trial/bin edges; vectorized nearest video frames | Sample conversion completes in 2.32 s excluding verifier, versus millions of nested count operations |

| Step | Time / Session | Estimated Total Time |
| Conversion (sample) | 1.11 s mean NWB processing | ~3.2 min for 174 files by linear session count |
| Pickle writing / retained data volume | Sample: 0.069 GiB for 527 trials | Roughly 6–12 GiB and several minutes; full run safely below 15 min |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None after excluding only all-population-zero windows

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Lick direction choice | 0.7239 | 0.6161 |
| Outcome | 0.7524 | 0.6505 |
| Early lick | 0.8385 | 0.7355 |
| Tongue y-position | 0.6638 | 0.5251 |

Training completed on GPU. Loss decreased monotonically overall from 14.825 at epoch 1 to 0.580 at epoch 200; test loss was 0.739. Every validation balanced accuracy exceeded uniform chance (0.333, 0.333, 0.5, 0.25 respectively), supporting correct alignment and label construction.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 11.042 GiB
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total neurons | 69,943 | Classifier-good + CCF | 69,453 explicit good | 69,453 | Yes to archive; paper differs 490 (0.7%) by version |
| Mean neurons/session | Not stated | All good areas concatenated | 401.46 good | 401.46 (90–923) | Yes |
| Subjects | 28 | — | 28 | 28 | Yes |
| Sessions | 173 | QC-file sessions | 174 NWB; 173 with good units | 173 | Yes |
| Trials (total) | Not stated | Analysis-dependent filters | 94,990 raw; 94,370 in usable neural sessions | 90,859 with population spikes in window | Explained data-validity exclusion |
| Trials/session (mean) | 476 (130–785) | Analysis-dependent | 545.5 raw among usable sessions | 525.2 (159–800) | Requested broad retention; differs as expected |
| Time-from-tone range | Not stated | Go-relative task events | [-1.5,11.9] at bin centers | [-1.5,11.9] s | Yes |
| Photostim range | Binary condition | Stimulation on/off | [0,1] | [0,1] | Yes |
| Choice distribution | Not stated | left/right; reference drops no-response | Raw task-derived | [0.429,0.422,0.149] | Yes |
| Outcome distribution | 84% correct on selected control trials | correctness {-1,0,1} | Raw retained counts | [0.149 ignore,0.166 miss,0.684 hit] | Not directly comparable due inclusion of stim/early/ignore |
| Early-lick distribution | Excluded in analyses | binary | Raw retained | [0.885 no,0.115 yes] | Yes |
| Tongue distribution | Not reported | Reference mean-imputes hidden | DLC + requested visibility | [0.062,0.032,0.065,0.841] | Requested transform |

Full conversion required 223.43 s. Exact class counts: choice [38,939, 38,369, 13,551]; outcome [13,551, 15,116, 62,192]; early [80,373, 10,486]; tongue timepoints [451,708, 230,742, 473,264, 6,113,006]. Verification reports valid format with no errors or warnings. Spot checks of early/middle/late sessions show 80 bins, plausible unit counts, and no all-zero retained trials.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` begins “Data format is valid, no errors or warnings.” Searching for error/warning/invalid/all-neural messages found only that success sentence.
2. **Independent original-NWB neural check**: `/app/cache/sanity_checks.py` loads the original first-session NWB directly with PyNWB (without importing conversion code), independently counts spikes for trial 5/unit 3/bin 10 in `[left,right)`, divides by 0.05, and compares to the pickle using `np.allclose`. PASS; both equal 20 Hz.
3. **Independent original-NWB input checks**: Independently associates go and last sample onset by trial interval, reconstructs all 80 continuous tone-time values, and reconstructs binary laser-center membership from raw on/off timestamps. Both `np.allclose` checks PASS.
4. **Independent original-NWB output checks**: Independently maps task-table choice/outcome/early labels and recomputes session tongue percentiles, nearest raw frame, confidence visibility, and the selected tongue class. All `np.allclose` checks PASS.
5. **Reference code comparison—loading**: Reference `process_one_sess` loads behavior/spikes/QC/histology from raw export; converter loads equivalent NWB `trials`, events, `units`, and Camera0 tracking exclusively through PyNWB.
6. **Reference code comparison—filtering**: Both select classifier-good, histology-annotated units. Converter excludes the one no-good-unit NWB and only wholly empty population windows (3,511 trials across 100 sessions; maximum 376/session). Reference downstream regular-trial exclusion is intentionally not applied because requested outputs require early/ignore/stimulation trials.
7. **Reference code comparison—alignment**: Reference raw spikes are already go-relative and shifts lick/laser events by go; converter subtracts absolute NWB go timestamps consistently from all streams and associates by trial interval.
8. **Reference code comparison—binning**: Both count `[left,right)` and divide by width. Explicit task override changes reference sliding 40-ms/3.4-ms to 80 non-overlapping 50-ms bins.
9. **Reference code comparison—inputs**: Reference aligns task stimulation; converter creates requested time-from-final-tone and per-bin stimulation state. Tone increments are 0.05 s with maximum float32 deviation `7.64e-7`.
10. **Reference code comparison—outputs**: Outcome/early fields are direct NWB equivalents of reference correctness/early report. Choice uses task-defined outcome x instruction and tongue uses side-camera DLC; hidden tongue remains class 3 as explicitly required rather than reference mean imputation.
11. **Key statistics**: Asserted 173 sessions, 28 subjects, 69,453 units, 90,859 trials, 80 timepoints, and >=2 trials/session. These match the archive and documented paper-version/task-retention differences. Session neuron range 90–923 and all 293 CCF labels map exactly.
12. **Edge cases/off-by-one**: Confirmed 81 edges/80 centers, final center +1.475 (not +1.5), half-open bins, valid class bounds, monotonic tone-time, ordered q40<=q60 in every session, constant trial outputs across time, choice no-lick iff outcome ignore, and no retained all-zero neural trial. Full check PASS.

### Issues Found and Resolved
- **Behavior extends beyond neural recording in some NWBs**: Initial sample produced hundreds of all-zero late trials. Applying the intersection of all unit observation intervals proved over-restrictive because accepted periods vary by unit/probe and removed valid populations/classes. Final resolution follows reference population binning and excludes only windows with no spike in any classifier-good unit. Re-running conversion and every Step 10 check yields zero verifier warnings and all PASS results.
- **Paper/archive aggregate mismatch**: Paper reports 69,943 units; this release explicitly labels 69,453. Using explicit archived classifier labels is reproducible and differs only 0.7%; no code change is justified.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes; 15.5517 at epoch 1, 0.6581 at epoch 200; test loss 0.6408. GPU training finished without fallback/error.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Lick direction choice | 0.7075 | 0.6779 | Uniform chance 0.3333 |
| Outcome | 0.7029 | 0.6662 | Uniform chance 0.3333 |
| Early lick | 0.7992 | 0.7557 | Uniform chance 0.5000 |
| Tongue y-position | 0.6718 | 0.6305 | Uniform chance 0.2500 |

Required `--plot-samples` run completed and created `/app/sample_trials.png` and `/app/predictions.png`. All outputs are substantially above chance; train/validation gaps are small (0.030, 0.037, 0.044, 0.041).

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Papers |
| Lick direction choice | 0.6779 balanced validation; 2.03x uniform chance | Data paper reports time-varying lick-choice logistic decoding (200-ms windows, pseudo-populations, nested CV), strongest in ALM and above 0.5 across projection regions; no single scalar directly comparable to this across-session 50-ms model. Result is directionally consistent. |
| Outcome | 0.6662; 2.00x chance | No three-class ignore/miss/hit decoder reported in either paper. |
| Early lick | 0.7557; 1.51x chance | Early trials were excluded in paper analyses; no reported decoder. |
| Tongue y-position | 0.6305; 2.52x chance | Method paper predicts continuous marker/video-related activity and reports time-varying ROC/R2 curves, not this requested four-class tongue-y decoder. Strong above-chance result is consistent with widespread movement encoding. |

**Chance analysis**: All four exceed chance; all also meet or exceed the requested 1.5x-chance investigation threshold (early lick is 1.511x). Therefore none meets the protocol’s low-accuracy trigger.

**Paper comparison search**: Both full paper texts were searched for accuracy/decoder/AUC passages and all choice-decoding Methods/Figure captions were reviewed. Reported decoding is plotted as time-dependent traces or ROC AUC/R2 under materially different trial selection, binning, regions, and targets; the papers provide no scalar for the four requested outputs. It would be misleading to digitize figure curves as direct accuracy targets. Choice result is above chance and within the broad range shown by region/time-specific curves.

**Train-validation gap**: Ratios are choice 1.044, outcome 1.055, early 1.058, tongue 1.066—far below the 1.5x overfitting trigger. Test loss (0.6408) is also slightly lower than final train loss (0.6581), with balanced-loss sampling explaining small differences; there is no leakage signature.

**Alignment/label review**: Step 10 independently verified raw neural, task input, trial labels, and tongue labels using PyNWB and `np.allclose`. Class variation is adequate (full distributions documented in Step 9). Processing plots show go/tone/laser/tongue alignment. Unit QC and bin counting match the reference except explicit task overrides. No accuracy-related conversion change is justified.

### Issues Found and Resolved
- No new issue was found. The early-lick score was specifically scrutinized because it is closest to 1.5x chance; its raw labels, variation (11.5% yes), alignment, and small generalization gap all pass, and validation remains above the threshold.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created with README_CACHE.md
- [x] Investigation scripts/output and bytecode moved to cache; all 11 required deliverables audited as present and nonempty
