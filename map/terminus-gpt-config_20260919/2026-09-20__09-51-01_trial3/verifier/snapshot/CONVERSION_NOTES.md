# Dataset Conversion Notes

## Overview
- **Dataset**: Brain-wide neural activity underlying memory-guided movement (provided NWB/source data and reference papers)
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment verification:
- Python 3.13.15
- NumPy 2.4.4
- PyTorch 2.6.0+cu124

Directory contents (`/app`, listed after notes file creation):
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
- `train_decoder.py`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `loadmat`, `loadh5mat` | `VideoAnalysisUtils/preprocessing_utils.py` | LOADING | Load conventional/HDF5 MATLAB files and normalize MATLAB structures/cluster notes. |
| `process_one_sess` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING / CURATION | Combine probe files for a session, load precomputed classifier or legacy good-unit index files, preserve unit QC/anatomy, and prepare trial-aligned spikes. |
| `sliding_histogram` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Count spikes in half-open windows `[center-width/2, center+width/2)` and divide by width for Hz; returns `(n_bins,n_trials,n_units)`. |
| `helper_get_neuron_id_area` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | CURATION | Restrict precomputed area/QC unit IDs by hemisphere, require CCF labels, and assert agreement of annotations. |
| `helper_filter_by_neuron_id` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Subset neural/anatomical arrays while propagating trial behavior, stimulation, and timing variables. |
| `check_fr` | `VideoAnalysisUtils/preprocessing_utils.py` | CURATION | Remove neurons with zero across-trial variance in any analyzed time bin. |
| `get_period` | `VideoAnalysisUtils/preprocessing_utils.py` | PROCESSING | Define go-cue-relative analysis epochs (`all` -3.0..3.5 s, sample -1.9..-1.2, delay -1.2..0, post-go 0..1). |

### Notes
- Repository README identifies this as the analysis code for *Brain-wide analysis reveals movement encoding structured across and within brain areas* and labels `Archive/` old/unused/potentially broken; current `VideoAnalysisUtils` code was prioritized.
- Data are extracellular electrophysiology spike times, not calcium imaging; delta-F/F is not applicable.
- Source behavior fields used by the reference loader include auto-learn, early-lick, auto/free-water flags, lick directions/times, report/correctness, cue/delay/sample timing, stimulation, and trial type.
- The loader subtracts each trial's go-cue time from lick times and stimulation on/off times. Saved spike times are documented as already go-cue aligned (go cue = 0).
- `behavior_report` coding is 1 correct/free-water, 0 error, -1 no response. `task_stimulation` columns are laser power, stimulation type, laser on, laser off; no-stimulation rows use power 0 and NaN timing/type fields.
- Neural binning uses centers and half-open bins `[left,right)`. Rate is spike count divided by bin width in seconds. The method repository includes processed folders named `stride3_bw40`; this is a reference-analysis choice, whereas this conversion must use the task-required non-overlapping 50-ms bins.
- Unit curation is not recomputed from inline thresholds in this code. `process_one_sess` loads precomputed good-unit indices from classifier QC (`goodunits`) or legacy QC (`DJ_GoodUnitsIdx_14regions_old_qc`), then retains QC metrics (amplitude, presence ratio, amplitude cutoff, ISI violation, average firing rate, drift metric), unit quality, and CCF anatomy. The current path defaults to classifier QC.
- Region membership is supplied by the precomputed area/QC index list. Active helper logic intersects it with hemisphere (CCF ML split at 5700 um), excludes absent CCF annotations, and checks annotation agreement. Direct ontology/voxel region re-filtering in that helper is commented out.
- `check_fr` is an analysis safeguard against zero-variance recordings; whether it is appropriate for final conversion will be reconciled with the distributed data and paper methods in later steps.
- No source trial filtering was found in the core loader: behavioral flags are preserved for downstream selection. Exact trial/session curation will therefore be resolved from data and papers before mapping.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` contains 174 NWB 2.x/HDF5 files, one file per recording session, nested under `sub-<mouse>/`. Filenames encode subject, session timestamp, and modalities (`behavior+ecephys+ogen`). No data-side README was present.
- NWB `intervals/trials` is a 1-D DynamicTable with 15 columns common to every file: start/stop time, trial number/UID, task/protocol, instruction, early lick, outcome, auto/free-water flags, and photostimulation onset/power/duration.
- NWB `units` is a ragged spike table. `spike_times` stores absolute session timestamps concatenated across units and `spike_times_index` gives cumulative unit endpoints. It also stores 43 common columns including `unit_quality`, `is_good_trials` (unit x trial Boolean), spike-sorting metrics, detailed Allen CCF `anno_name`, classifier label, electrode references, and waveforms.
- `acquisition/BehavioralEvents` contains absolute timestamp series for presample, sample, delay, go, trial end, left/right licks, and photostimulation start/stop. `go_start_times` has exactly one event per trial. Some sample/delay streams contain extra events (e.g. 405 samples for 368 trials), so trial-window/event matching rather than positional truncation is required.
- `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking` exists in all sessions. Its data are `(n_video_frames,3)` float64 columns `(tongue_x, tongue_y, tongue_likelihood)`, with explicit timestamps; representative median interval is 3.4 ms (~294 Hz). All coordinates/likelihoods are finite, including low-confidence frames, so visibility must use likelihood rather than NaN detection.
- Camera0 jaw and nose tracking also occur in all sessions. Lick-port, whisker, and Camera3 tracking are optional and sparse across sessions.
- Neural and behavioral streams are session-absolute and must be aligned using event timestamps. A representative first trial spans 0.0–4.8615 s and has go onset 3.0615 s.

### Available Variables and Native Types
| Source | Shape/type | Meaning |
|--------|------------|---------|
| `units/spike_times` + index | ragged float64 | Absolute spike timestamps per unit |
| `units/unit_quality` | string per unit | `good` single unit or `multi` multi-unit |
| `units/is_good_trials` | bool `(n_units,n_trials)` | Trial-specific drift/quality validity |
| `units/anno_name` | string per unit | Detailed Allen CCF anatomical annotation |
| `trials/outcome` | string per trial | `hit`, `miss`, `ignore` |
| `trials/early_lick` | string per trial | `early`, `no early` |
| `trials/trial_instruction` | string per trial | instructed side (`left`, `right`) |
| BehavioralEvents go/sample | timestamp series | Go cue and auditory sample/tone timing |
| BehavioralEvents photostim | timestamps/data/control | Laser on/off, power, stimulation code |
| Camera0 tongue tracking | float64 `(frames,3)` | x, y, likelihood |

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Data size | approximately 100 GB |
| Raw unit-session records | 272,227 |
| Units / session | 493–3,191; mean 1,564.52 |
| `unit_quality=good` | 154,948 (56.92%) |
| `unit_quality=multi` | 117,279 (43.08%) |
| Unit-trial validity | 146,877,235 / 147,085,032 pairs true (99.8587%) |
| Units valid on every trial | 270,401 / 272,227 |
| Subjects | 28 |
| Sessions | 174 |
| Sessions / subject | 3–10 |
| Trials (total) | 94,990 |
| Trials / session | 264–800; mean 545.92 |
| Outcome distribution | hit 65,254; miss 15,641; ignore 14,095 |
| Early-lick distribution | no early 84,185; early 10,805 |
| Trial instruction | left 46,077; right 48,913 |
| Auto-water / free-water | 1,339 / 2,450 trials |

### Data Quality / Edge Cases
- Photostimulation trial-table fields use string `N/A` on unstimulated trials, while BehavioralEvents provide numeric timestamps only for actual stim events.
- `classification` is object/string in 173 sessions but float64 NaN in one session; robust code must not assume string dtype. This field is distinct from `unit_quality`.
- `is_good_trials` is almost entirely true but is not universally true; conversion must decide how to preserve a fixed neuron set while excluding invalid unit-trial periods.
- Detailed `anno_name` labels include cortical layers and subnuclei; downstream brain-region naming may use these native labels or a documented aggregation after reconciliation with references.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote / Context |
|-----------|-------|------------------------|
| Good units (total) | 69,943 | Data-paper methods: classifier-selected good units across the analyzed dataset |
| Subjects | 28 mice | Data paper Figure 1 summary and method paper Methods |
| Sessions | 173 analyzed behavioral sessions | Data paper Figure 1 summary |
| Probe penetrations | 655 in Methods text; figure summary says 660 | Data-paper prose versus figure caption; likely counting/version distinction |
| Trials / session | mean 476, range 130–785 | Data-paper behavioral methods |
| Correct rate | mean 84%, range 65–99% | Data-paper behavioral methods |
| Neural reference bin | 40 ms width, 3.4 ms stride | Method paper: firing-rate construction for video analyses |
| Video sampling | 300 Hz | Both papers; NWB timestamps are ~294 Hz nominally |
| Response period | 1.5 s | Data-paper task description |
| Delay period | 1.2 s | Data-paper task description |
| Go cue | 6-kHz carrier, 360-Hz modulation, 0.1 s | Data-paper task description |
| Photostimulation prevalence | about 25% randomly interleaved trials | Data-paper photoinhibition methods |
| Video choice AUC pre-sample | 0.51 ± 0.06 | Method paper, 106 sessions |
| Video choice AUC sample/delay | 0.66 ± 0.12 | Method paper, 106 sessions |
| Video choice AUC post-go | 0.99 ± 0.01 | Method paper, 106 sessions |

### Processing Details
- The task is an auditory delayed-response task. The instruction tone (3 or 12 kHz) is played three times for 150 ms with 100-ms inter-tone gaps, followed by a 1.2-s delay and the go cue. Early licking can trigger replay of the sample/delay epoch, explaining extra sample/delay events relative to trials in NWB.
- Data paper analyses align task epochs to the go cue. The conversion's required window (-2.5 to +1.5 s) lies within the task and uses the task-required 50-ms bins rather than the method paper's overlapping 40-ms/3.4-ms bins.
- The method paper bins spikes into 40-ms firing-rate windows with 3.4-ms stride. Reference code counts half-open windows and divides by width. This conversion must preserve that counting convention but change width/stride to required non-overlapping 50 ms.
- DeepLabCut tracks tongue, jaw, and nose. The method paper detects marker outliers using a five-standard-deviation velocity threshold and imputes from nearby frames. For its continuous marker model, occluded in-mouth tongue positions are set to their mean. The current task instead explicitly requires a categorical `not visible` tongue class, so visibility information must be retained rather than mean-imputed.
- The method paper used side-view video only. This matches the Camera0 side-view stream present in every NWB session.
- Reference neural/video analyses excluded brain areas with fewer than 10 neurons/session and neurons firing below 2 Hz.

### Curation Steps

**Neuron curation rules**:
- Raw spikes were sorted with Kilosort2 and evaluated with 15 quality metrics.
- Five region-specific logistic-regression classifiers (cortex, striatum, thalamus, midbrain, medulla) were trained from blinded manual labels. Classifier-predicted `good` units were used in the papers; classifier false-alarm rates were reported as cortex 7.8%, striatum 6.4%, thalamus 7.3%, midbrain 5.5%, medulla 4.3%.
- The paper reports 69,943 classifier-good units. This aligns conceptually with NWB `units/classification`, not the broader Kilosort `unit_quality` field.
- Method-paper downstream analyses additionally exclude mean firing rate <2 Hz and area/session groups with <10 neurons; applicability to this decoder will be reconciled in Step 4 because preserving all classifier-good units may better match the data paper.

**Trial/session curation rules**:
- Data-paper behavior analyses exclude early-lick and no-response trials. Sessions were selected for >65% control performance and at least 50 correct left and 50 correct right trials.
- Method-paper video/neural analyses exclude photoinhibition, free-water, early-lick, and ignore/no-response trials. Some analyses use 64 held-out trials balanced for lick side and correct/error; others use stratified 5-fold or 20-fold cross-validation.
- These trial exclusions cannot be copied wholesale here: photostimulation is a required decoder input, and early lick, ignore outcome, and no-lick choice are required decoder outputs. Such trials must be retained unless their source streams are invalid.

### Decoders Trained
| Decoded variable | Accuracy |
|------------------|----------|
| Choice from behavioral video, pre-sample | ROC-AUC 0.51 ± 0.06 |
| Choice from behavioral video, sample/delay | ROC-AUC 0.66 ± 0.12 |
| Choice from behavioral video, post-go | ROC-AUC 0.99 ± 0.01 |
| Single-neuron choice / uninstructed movement | Logistic-regression ROC-AUC; AUC >0.65 used as modulation threshold, not a mean task accuracy |

### Interpretation for This Conversion
- Paper AUCs are contextual only: they decode from video or single neurons and do not directly predict expected accuracy of the supplied multi-neuron decoder.
- The decoder specification overrides paper exclusions and tongue mean-imputation where necessary, but loading, timestamps, half-open spike counting, classifier QC, and side-camera use should otherwise match the references.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Session count | Classifier-QC path expects good-unit lists | 174 NWBs; one session (`sub-440958_ses-20190216T162508`) has all-NaN `classification`; 173 have classifier labels | 173 analyzed sessions | Exclude the single session lacking classifier output; retain the 173 QC-compatible sessions. |
| Good-unit count | Load precomputed classifier-good indices | 69,453 `classification=='good'` units in 173 sessions | 69,943 good units | Use the distributed classifier labels. The 490-unit difference is a release/version discrepancy; the extra unlabeled session has 1,852 raw units and cannot explain it. |
| Quality field | Reference uses region classifiers | `unit_quality` gives 154,948 `good` and is much broader; `classification` gives paper-scale 69,453 | Region-specific classifier units used | Use `classification=='good'`, not Kilosort `unit_quality`. |
| Trial-specific QC | Reference processed good-unit lists | `is_good_trials` has 64,612 invalid pairs among classifier-good units, affecting 565 units in four sessions | Stable/high-quality units intended | Because target format needs one fixed neuron matrix shape per session, retain only classifier-good units valid on every session trial (68,888 units expected). This avoids silently treating invalid periods as zero activity. |
| Low firing rate | Method paper excludes <2 Hz for video prediction | 28,806/69,453 classifier-good units are <2 Hz over the session | Analysis-specific 2-Hz exclusion | Do not apply: it is downstream method-paper curation, not data-paper QC, and would discard 41.5% of released good units. |
| Trial exclusions | Method paper excludes stim/free-water/early/ignore | These labels are present and valid | Excluded for specific analyses | Retain all trials because the task explicitly requires photostim input and early-lick/ignore/no-lick outputs. |
| Binning | 40-ms width, 3.4-ms stride in method paper | Raw spikes permit arbitrary bins | Decoder requires 50 ms | Use reference half-open counting but task-required 50-ms non-overlapping bins. |
| Tongue occlusion | Mean-imputed for continuous marker models | Raw likelihood is available | Decoder requires class 3 not visible | Do not mean-impute invisible frames; classify visibility from DLC likelihood, then discretize visible y per session. |
| Anatomy | Reference analyses aggregate ontology areas | NWB has detailed Allen `anno_name` labels | Many areas/subareas analyzed | Preserve native `anno_name` as brain-region names; this avoids undocumented aggregation and retains full anatomical information. |
| Session behavioral criteria | Paper describes >65% performance and >=50 correct trials/side | Applying literal criteria to released labels excludes 22/174 files | Paper summary is 173 classifier-QC sessions | Treat classifier availability as authoritative released-session curation. The behavioral criteria are analysis-selection criteria and cannot supersede task-required labels. |

### Final Consistent Understanding
- The distributed NWBs are the raw synchronized source; session-absolute spikes, events, and tracking must be aligned directly by timestamps.
- The custom classifier field is the authoritative neuron QC output. The one file lacking it is unusable under paper-matched QC.
- Reference half-open spike counting and side-camera tracking are retained. Required 50-ms bins, retained special trial types, and explicit invisible-tongue category are task-mandated differences.
- No calcium processing or delta-F/F is applicable.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `units/spike_times`, `spike_times_index` | `neural` | Select classifier-good units valid on every trial; count spikes in 80 half-open 50-ms bins from go-2.5 to go+1.5; divide by 0.05 s; transpose to `(neurons,80)` float32 | `sliding_histogram` | Same half-open counting/rate convention; task-required bin width/stride |
| Last `sample_start_times` within trial before go | `input[0]` | At each bin center, absolute center time minus tone onset, in seconds | reference event alignment | Continuous time-from-tone-onset as explicitly required; replay trials use most recent sample onset |
| `photostim_start_times` / `photostim_stop_times` | `input[1]` | Boolean 1 where bin center is within a laser interval, else 0 | loader stimulation alignment | Time-varying `(80,)`; direct event timestamps avoid string `N/A` parsing |
| `trials/outcome` + `trial_instruction` | `output[0]` choice | ignore→no lick; hit→instruction side; miss→opposite of instruction | loader correctness/trial type | Expanded across 80 bins so outputs share a time-varying matrix |
| `trials/outcome` | `output[1]` outcome | map ignore/miss/hit to 0/1/2 | direct NWB label | Expanded across time |
| `trials/early_lick` | `output[2]` early lick | no early→0, early→1 | loader early-lick field | Expanded across time |
| Camera0 tongue y + likelihood | `output[3]` tongue y-position | Align nearest video frame to each bin center; likelihood <0.9→3 not visible; visible y below session 40th percentile→0, 40th–60th→1, above 60th→2 | side-view marker processing | Percentiles computed from visible session frames only; preserve required invisible class |
| `general/subject/subject_id` | `subjects`, `subject_idx` | Sorted unique IDs and per-session index | direct NWB metadata | 28 subjects expected |
| `units/anno_name` | `brain_regions`, `brain_region_idx` | Decode native Allen labels, global sorted vocabulary | anatomical helper concept | Preserve detailed source labels |

### Output Structure
- `input_names = ['time from tone onset', 'photostimulation on']`.
- `output_names = ['lick direction choice', 'outcome', 'early lick', 'tongue y-position']`.
- `output_values = [['left','right','no lick'], ['ignore','miss','hit'], ['no','yes'], ['below 40th percentile','40th to 60th percentile','above 60th percentile','not visible']]`.
- Each input trial is float32 `(2,80)`. Each output trial is int64 `(4,80)`; per-trial labels are repeated over time, while tongue class varies by bin.
- Bin centers are `-2.475, -2.425, ..., 1.475` s relative to go. Metadata records 50 ms, go-cue onset, offsets -2.5/+1.5, 80 bins, class rules, QC, and per-session source information.

### Key Decisions
1. **Session QC**: Exclude only the one NWB with missing classifier labels, yielding the paper-compatible 173 sessions.
2. **Neuron QC**: Keep `classification=='good'` units only when `is_good_trials` is true for every source trial. This gives a fixed trustworthy neuron set per session and avoids unsupported imputation.
3. **Trials retained**: Keep photostimulation, free/auto-water, early-lick, miss, and ignore trials because they are needed by specified inputs/outputs. Exclude only trials lacking complete required continuous-stream coverage.
4. **Tone event**: Use the last sample-start event between trial start and go. Early licking can replay sample/delay, so this represents the tone epoch causally associated with the final go cue.
5. **Tone representation**: Use continuous elapsed time in seconds, as explicitly requested by “time from tone onset,” rather than a binary onset pulse.
6. **Photostimulation representation**: Mark bin centers inside `[stim_on, stim_off)`; the paper states stimulation ends before go, providing a temporal sanity check.
7. **Choice**: Derive actual lick direction from the curated per-trial outcome/instruction labels: hits follow instruction, misses are opposite, ignores are no lick. A direct event comparison agreed overwhelmingly but had occasional timing/free-water edge cases, so curated labels are authoritative.
8. **Tongue visibility**: Use DLC likelihood >=0.9. Likelihood is strongly bimodal (near zero or one), and 0.9 is a standard conservative DLC confidence threshold. Percentiles use only visible frames, preventing occluded/noisy coordinates from corrupting thresholds.
9. **Tongue temporal sampling**: Nearest video frame to each 50-ms bin center is sufficient at ~300 Hz (maximum typical mismatch ~1.7 ms), avoids smoothing across appearance boundaries, and preserves categorical visibility.
10. **Trial boundaries**: `stop_time` may precede go+1.5 for ignore/aborted trials, but continuous streams remain valid and no requested window crosses the next trial. Do not clip solely to behavioral trial stop; exclude only genuine stream-coverage failures.
11. **No 2-Hz filter**: Preserve classifier-good data-paper units; method-paper 2-Hz filtering is specific to video prediction.
12. **Precision and size**: Store firing rates and inputs as float32 and outputs/indices with compact integer dtypes where accepted, to control memory without changing 20-Hz firing-rate quantization.

### Planned Sanity Checks
- [ ] Directly load an original NWB and compare one converted neural bin against manual `sum((spikes>=left)&(spikes<right))/0.05` using `np.allclose`.
- [ ] Compare converted time-from-tone and photostimulation vectors to direct event timestamps using `np.allclose`.
- [ ] Compare choice/outcome/early labels for at least three direct-source trials using `np.allclose`.
- [ ] Compare tongue classes for selected visible/invisible frames to direct y/likelihood and session percentile thresholds using `np.allclose`.
- [ ] Assert 80 bins, identical trial counts across neural/input/output, at least two trials/session, finite arrays, and one region index per neuron.
- [ ] Assert no retained requested window crosses the next trial or falls outside video timestamps.
- [ ] Confirm expected 28 subjects, 173 source-QC sessions before stream-coverage filtering, approximately 68,888 stable classifier-good units, and source trial/output distributions.
- [ ] Verify photostimulation is off by go in accordance with the paper and that tone onset precedes go.
- [ ] Compare converted statistics with the paper (28 mice, 173 sessions, paper 69,943 good units) and document known release difference.

---

## Step 6: Script Development
**Status**: COMPLETE

- Created `/app/convert_data.py` with required CLI: positional output, mutually exclusive `--full` (default) / `--sample`, and `--show-processing`.
- Implemented robust HDF5/NWB loading, classifier/session QC, fixed stable-neuron selection, event alignment, vectorized spike binning, tongue discretization, source metadata, assertions, and per-session timing.
- `--show-processing` creates `processing_<session_id>.png` for up to two sessions, showing neural alignment, inputs, output time series, and class distributions.
- Syntax compilation and help-path execution passed.

Code inefficiencies identified:
- Naive loops over trials x bins x spikes would be prohibitively slow.
- Repeatedly loading full video arrays or scanning spikes per trial would duplicate I/O.

Code speedups added:
- For each unit, one vectorized `np.searchsorted` over all trial bin edges computes all 80-bin counts.
- Session tongue arrays are loaded once; nearest video indices and all classes are vectorized.
- Float32 neural/input storage and int8 categorical output storage reduce pickle size and I/O.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 1 |
| Trials (total) | 527 |
| Trials / session | 368, 159 |
| Stable classifier-good neurons (total) | 834 |
| Neurons / session | 459, 375 |
| Time bins | 80 per trial |
| Neural dtype/range | float32, nonnegative Hz |
| Time-from-tone range | approximately -0.6 to 5.7 s |
| Photostimulation range | {0,1} |
| Choice values | all 0–2 present |
| Outcome values | all 0–2 present |
| Early-lick values | both 0–1 present |
| Tongue values | all 0–3 present |
| Output shapes | `(4,80)` for every trial |
| Pickle size | approximately 0.074 GB |

### Format Validation
- `/app/verification_sample_out.txt` created by the required `--verify-only` command.
- Validator completed with `Data verification complete`; no errors were reported.
- All dimensions, names, values, region indices, and metadata passed the supplied validator.

### Processing Plots Review
- Two `processing_<session_id>.png` files were created and opened successfully (valid RGB PNGs).
- Plots include neural heatmaps/population means centered on go=0, tone/stimulation inputs, categorical output trajectories, and output distributions.
- Neural and inputs have the expected temporal ranges; tongue is predominantly not visible before response, consistent with paper methods.

### Run Time Estimates
| Speed-up Implemented | Time Savings |
|----------------------|--------------|
| Vectorized searchsorted binning | Avoids trial x bin x spike Python loops |
| Single-load/vectorized tongue alignment | Avoids repeated frame scans |
| Compact dtypes | Reduces corrected sample to approximately 74 MB and lowers write time |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Processing | 0.86–1.29 s | ~3 minutes for 173 sessions |
| Pickle writing | small for sample; scales with data size | total conservatively <10 minutes |

### Issue Found and Fixed
- Initial sample run failed because a float32 photostimulation array was used with Boolean in-place OR. Changed the accumulator to Boolean and cast to float32 after interval construction. Recompiled and reran successfully.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Corrected sample data passed verify-only after neural observation-interval mapping was fixed.
- Errors: None
- Remaining warnings: None

### Training Progress
- Corrected sample loss decreased from approximately 8.0 at epoch 5 to 0.586226 at epoch 200.
- Test loss: 0.733773.

### Decoder Results (Corrected Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-----------------------|-------------------------|--------|
| Lick direction choice | 0.7280 | 0.6213 | 0.3333 |
| Outcome | 0.7525 | 0.6448 | 0.3333 |
| Early lick | 0.8417 | 0.7270 | 0.5000 |
| Tongue y-position | 0.6636 | 0.5291 | 0.2500 |

All corrected-sample validation accuracies exceed chance. Results supersede the initial sample run performed before the neural trial-mapping bug was discovered.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 11.645 GB
- `conversion_full_out.txt`: created
- `verification_full_out.txt`: created; ends with `Data verification complete`

### Final Full Statistics
| Statistic | Reference Papers | Reference Data | Converted Data | Match / explanation |
|-----------|------------------|----------------|----------------|---------------------|
| Subjects | 28 | 28 | 28 | Exact |
| Sessions | 173 analyzed | 174 NWB; one lacks classifier output | 173 | Exact paper-compatible QC sessions |
| Stable classifier-good neurons | 69,943 paper version | 69,453 classifier-good; 68,888 valid all recorded trials | 68,888 | Release difference plus conservative trial-validity filter |
| Mean neurons/session | not directly stated | — | 398.20 (range 90–923) | Plausible |
| Source behavioral trials in retained sessions | mean paper analysis 476/session | 94,370 | 94,370 inspected before neural/video validity | Raw distributed data include more trial types than paper analyses |
| Converted usable trials | paper excludes several required categories | — | 90,094 | Only absent/unusable streams removed; required special trial types retained |
| Trials/session | paper analysis mean 476, range 130–785 | — | mean ~520.8, range 7–800 | Includes task-required early/ignore/stim trials; one video-truncated session has 7 valid trials |
| Time bins | task requires 80 | raw timestamps | 80 in every trial | Exact |
| Photostim range | binary task input | events | {0,1} | Exact |
| Brain regions | broad and subregional analyses | 293 native Allen labels | 293 | Native annotations preserved |

### Full Output Distributions
- Choice: left/right/no-lick all represented and near-balanced between left/right.
- Outcome: ignore, miss, and hit all represented; hit is the majority as expected.
- Early lick: both no/yes represented; no is the majority.
- Tongue: all four classes represented; not-visible dominates because the tongue is usually occluded before response, matching the paper.

### Validation and Integrity
- Corrected full validator produced no errors and no all-zero-neural warnings.
- Every trial has `(n_neurons,80)`, `(2,80)`, and `(4,80)` neural/input/output shapes.
- All 173 sessions retain at least two trials; the minimum is 7 due to a genuinely truncated 49-s video stream.
- Full conversion took 154.6 s; runtime was well below the estimate and optimization threshold.

### Iteration: Neural Trial Mapping Bug Found and Fixed
- Initial full validation reported 3,470 wholly zero neural trials.
- Investigation showed `units/obs_intervals` and `is_good_trials` columns enumerate only electrophysiologically recorded trials, which can be fewer than `intervals/trials`. The initial script incorrectly treated every behavioral trial as recorded.
- Fixed by mapping per-unit observation-interval start times to behavioral trial starts, retaining only mapped trials, checking observation-count consistency, and excluding any remaining wholly zero neural window.
- An intermediate overly strict rule requiring the entire requested window inside trial observation bounds disproportionately removed short ignore trials; it was rejected. Final logic preserves recorded short trials but excludes only wholly absent neural inputs.
- Re-ran sample conversion/verification and full conversion/verification. The corrected data have zero all-neural-zero trials and validator warnings are resolved.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output Log Verification
- Read the complete corrected `/app/verification_full_out.txt`; it ends with `Data verification complete`.
- Initial warning: 3,470 wholly zero neural trials. Root cause was behavioral trials without corresponding electrophysiology observation intervals.
- Fixed mapping and reran sample/full conversion and verification. Corrected validator has no all-zero-neural warnings and no errors.

### Check 2: Independent Direct-Source Sanity Checks
`/app/step10_sanity.py` loads an original NWB directly without importing conversion functions and uses `np.allclose` for:
1. **Neural**: trial 5, selected neuron 3, bin 10 manually counted from raw spike timestamps with `[left,right)` and divided by 0.05 s.
2. **Input/time**: all 80 bin-center times minus the direct last pre-go sample timestamp.
3. **Input/photostimulation**: all 80 values reconstructed from raw stimulation on/off event intervals.
4. **Per-trial outputs**: choice, outcome, and early-lick values for three specific raw trials.
5. **Tongue output**: nearest raw video frames, likelihood threshold, and raw-session visible-y percentiles.
6. **Anatomy**: raw selected unit `anno_name` mapped through global region index.
- All direct-source `np.allclose` tests passed.

### Check 3: Reference Code Comparison
| Stage | Conversion | Reference | Result / justified difference |
|-------|------------|-----------|-------------------------------|
| Loading | Direct h5py NWB synchronized tables | MATLAB/HDF5 helper loaders | Same source concepts; NWB is distributed native format |
| Neuron filtering | classifier `good` and all `is_good_trials` true | precomputed classifier-good area indices | Matched classifier QC; fixed neuron set required by target |
| Trial mapping/filtering | map `obs_intervals` starts to trial starts; require video; remove wholly zero neural windows | saved trialized spikes/behavior | Necessary NWB-native equivalent; special task trial labels retained by specification |
| Alignment | absolute event timestamps, go=0 | subtract go cue from events/spikes | Equivalent |
| Binning | half-open bins, count/0.05 | `sliding_histogram`, half-open, count/bin width | Logic matched; 50 ms required instead of reference 40 ms |
| Inputs | last pre-go sample elapsed time; stim interval Boolean | cue/sample/stim fields retained | Task-specific representation |
| Outputs | curated labels and likelihood-based tongue classes | behavioral flags; tongue mean-imputed for continuous models | Task-required categorical outputs override exclusions/imputation |

### Check 4: Key Statistics
- 28 subjects matches the papers exactly.
- 173 classifier-QC sessions matches the paper exactly.
- 68,888 stable classifier-good units is close to the paper's 69,943; difference is explained by released classifier count (69,453 versus paper version) and removal of 565 units not valid for every recorded trial.
- 90,094 usable trials retain all required categories; trial count differs from paper analysis subsets because early, ignore, and photostimulation trials are intentionally retained.
- All input/output values fall within declared ranges and every categorical class is represented.

### Check 5: Edge Cases and Off-by-One Review
- Whole-data checks passed for 173 sessions: matching trial-list lengths, at least two trials/session, one region index/neuron, `(neurons,80)/(2,80)/(4,80)` shapes, finite/nonnegative neural data, valid class ranges, and no wholly zero neural trials.
- Confirmed bin centers are exactly `-2.475 + 0.05*k`, k=0..79, with metadata offsets -2.5/+1.5.
- Confirmed requested windows never cross the next trial start.
- One source session has classifier labels entirely missing and is excluded.
- One retained session has only seven trials because its side video ends at 49.3 s; all seven retained trials have valid neural/video data and satisfy the format minimum.
- One other session has video only through part of the behavior; only jointly observed trials are retained.
- `classification` is non-string in the excluded session and is robustly detected.

### Issues Found and Resolved
- **Photostimulation mask dtype**: Boolean OR on float accumulator failed; fixed Boolean accumulation followed by float cast.
- **Neural trial alignment**: Initially assumed all behavioral trials had ephys. Fixed using ragged `obs_intervals` and observation-count assertions.
- **Over-filtering iteration**: Requiring the entire fixed window inside trial stop/observation bounds removed short ignore trials. Rejected; final rule keeps recorded trials and removes only wholly absent neural inputs.
- Re-ran all checks after fixes; all pass.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`.
- Device: CUDA.
- Training/test trials: 72,022 / 18,072.
- Loss decreased: **Yes**, from 19.485005 at epoch 1 to 0.673147 at epoch 200.
- Test loss: 0.669909.
- Script finished successfully and sample plots were generated.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|-----------------------|-------------------------|--------|-------|
| Lick direction choice | 0.7094 | 0.6766 | 0.3333 | 2.03× chance |
| Outcome | 0.6967 | 0.6640 | 0.3333 | 1.99× chance |
| Early lick | 0.7804 | 0.7476 | 0.5000 | 1.495× chance; essentially the 1.5× heuristic |
| Tongue y-position | 0.6714 | 0.6085 | 0.2500 | 2.43× chance |

All outputs are substantially above chance. Train-validation gaps are small (0.033–0.063), with no evidence of severe overfitting or leakage.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Validation Accuracy | Chance | Ratio to Chance | Closest Paper Context |
|----------|---------------------|--------|-----------------|-----------------------|
| Choice | 0.6766 | 0.3333 | 2.03× | Video choice ROC-AUC 0.66 sample/delay; not directly comparable |
| Outcome | 0.6640 | 0.3333 | 1.99× | No matching paper decoder reported |
| Early lick | 0.7476 | 0.5000 | 1.495× | No matching paper decoder reported |
| Tongue y-position | 0.6085 | 0.2500 | 2.43× | Paper predicts neural activity from continuous video, opposite direction |

### Check 1: Accuracy vs Chance
- Every output exceeds chance. Choice, outcome, and tongue exceed 1.5× chance.
- Early lick is 0.7476 versus the 0.7500 optional 1.5× heuristic, a difference of 0.0024. This was investigated rather than dismissed.
- Direct raw checks of three trials confirm early-lick labels and time-constant expansion exactly (`np.allclose`). Both classes are well represented across the full dataset; imbalance is handled by the supplied decoder's balanced loss.
- Processing plots show go-aligned neural/input/output streams, and Step 10 independently verified timing against original NWB timestamps.
- Classifier QC, observation-interval mapping, and half-open binning all match the reconciled reference process. No conversion change is justified by the 0.0024 heuristic shortfall.

### Check 2: Accuracy Comparison to Papers
- The method paper reports video-to-choice ROC-AUC (0.51 pre-sample, 0.66 sample/delay, 0.99 post-go) and single-neuron AUC analyses. It does not report balanced accuracy for this task's neural-to-choice/outcome/early/tongue decoder.
- Choice validation balanced accuracy (0.6766) is numerically consistent with the paper's 0.66 sample/delay video ROC-AUC, while recognizing different inputs, metrics, epochs, and targets.
- No paper accuracy exists for outcome, early lick, or discretized tongue y-position in the required direction, so no valid one-to-one comparison can be made.

### Check 3: Train vs Validation Gap
- Train/validation ratios are choice 1.05, outcome 1.05, early lick 1.04, and tongue 1.10—well below the 1.5 overfitting threshold.
- Absolute gaps are 0.0328, 0.0327, 0.0328, and 0.0629 respectively.

### Low-Accuracy Debugging Checklist
1. Raw outputs checked for three specific trials: passed.
2. Temporal alignment plotted and independently reconstructed: passed.
3. Output variation checked: all classes represented; no 99% class.
4. Neural filtering checked against classifier QC and `is_good_trials`: passed.
5. Processing compared with reference code/papers: passed with task-mandated differences documented.

### Issues Found and Resolved
- No new conversion defect was found during accuracy review.
- The earlier neural observation-interval alignment issue had already been fixed before this final training run; the strong accuracies and small generalization gaps support the corrected alignment.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created with usage, mappings, curation, statistics, and decoder results
- [x] `cache/` folder created
- [x] `cache/README_CACHE.md` documents investigation artifacts
- [x] Temporary analysis scripts and extracted PDF text moved to cache
- [x] Required conversion, validation, and training outputs retained in `/app`
- [x] `CONVERSION_NOTES.md` reviewed and finalized

### Final Required Artifacts
- `/app/CONVERSION_NOTES.md`
- `/app/convert_data.py`
- `/app/converted_data.pkl`
- `/app/sample_data.pkl`
- `/app/README.md`
- `/app/conversion_sample_out.txt`
- `/app/verification_sample_out.txt`
- `/app/train_decoder_sample_out.txt`
- `/app/conversion_full_out.txt`
- `/app/verification_full_out.txt`
- `/app/train_decoder_full_out.txt`

