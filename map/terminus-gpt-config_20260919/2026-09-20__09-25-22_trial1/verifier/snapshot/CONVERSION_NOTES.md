# Dataset Conversion Notes

## Overview
- **Dataset**: Brain-wide neural activity underlying memory-guided movement
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment: Python 3.13.15; numpy 2.4.4; torch 2.6.0+cu124

Directory contents:
- `total 101632`
- `drwxr-xr-x  4 root root       57 Sep 20 13:32 .`
- `dr-xr-xr-x 19 root root      116 Sep 20 13:32 ..`
- `-rw-r--r--  1 root root     4911 Sep 20 13:31 .manifest`
- `-rw-r--r--  1 root root     5408 Sep 20 13:32 CONVERSION_NOTES.md`
- `-rw-r--r--  1 root root  5068544 Mar 22 14:34 ChenLiuEtAl2023_SpikeSortingQC.pdf`
- `-rw-r--r--  1 root root     2646 Sep 19 14:35 Dockerfile`
- `drwxr-xr-x  6 root root     4096 Mar 22 16:40 code`
- `drwxr-xr-x  2 root root     4096 Nov 26  2025 data`
- `-rw-r--r--  1 root root 26721927 Mar 22 14:34 datapaper.pdf`
- `-rw-r--r--  1 root root    89127 Sep 18 21:06 decoder.py`
- `-rw-r--r--  1 root root      644 Sep 20 03:17 docker-compose.yaml`
- `-rw-r--r--  1 root root 72136133 Mar 22 14:34 methodpaper.pdf`
- `-rw-r--r--  1 root root     8430 Mar 22 14:34 methods.txt`
- `-rw-r--r--  1 root root     7641 Sep 18 23:11 train_decoder.py`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_ephys_data` | `Notebooks/from_ccn/e2e_rewrite.ipynb` | LOADING | Loads MATLAB ephys/session structures used by the method-paper pipeline. |
| `preprocess_session` and area reorganization helpers | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Build session/area dictionaries containing firing rates, spike times, bin centers, CCF labels/coordinates, trial type, lick data, delay/sample durations, and stimulation metadata. |
| `align_embedding_vecs_between_lims` / alignment script | `Sherlock/align_embed_vecs.py` and archived getting-started notebook | PROCESSING | Align video embedding frames to a common time grid relative to go cue; nearest preceding frame is used on the requested grid. |
| marker alignment functions | `Sherlock/align_markers.py` | PROCESSING | Align tracked marker trajectories to the electrophysiology time basis and trial ordering. |
| `get_bad_trial_inds` / dataset integrity checks | archived getting-started and e2e notebooks | CURATION | Reject video trials whose frame count is inconsistent with trial end time (outside the documented 0--1 s tolerance) or whose requested sequence is incomplete. |
| `load_session`, `get_regular_trial_mask` | `VideoAnalysisUtils/population_decoding_utils.py` | LOADING / CURATION | Load a preprocessed brain-area session and select regular (non-perturbation/non-special) trials for reference population analyses. |
| nested cross-validation utilities | `VideoAnalysisUtils/population_decoding_utils.py` | PROCESSING | Decode trial type from time-resolved population firing rates using nested CV. |
| trial label collection | `Sherlock/collect_trial_type_labels_for_fig6.py` | PROCESSING | Construct trial labels/order used for movement/trial-type analyses. |

### Notes
- The recordings are extracellular electrophysiology. Neural source data are spike times and/or precomputed binned firing rates; delta-F/F is not applicable.
- Reference session arrays use firing-rate layout time x trial x neuron (`fr`), with `bin_centers`, CCF labels/coordinates, and unit identifiers. Some collection scripts also preserve raw `spike_times`.
- Behavioral/session fields propagated by preprocessing include `lick_directions` (0 right, 1 left), `lick_times`, `trial_type` (0 right, 1 left), sample and delay durations, and `stimulation` with columns `[laser_power, stim_type, laser_on_time, laser_off_time]`; no-stimulation rows use zero power and NaN timing/type.
- Reference video alignment is explicitly relative to go cue. Video embeddings/markers are aligned to ephys bins and trial indices; incomplete/mismatched video trials are identified rather than silently padded.
- Reference analyses commonly use a regular-trial mask when addressing unperturbed trial-type decoding. For this task, however, photostimulation is a required decoder input, so stimulation trials must be retained when their required streams are valid.
- Anatomical assignment uses CCF labels and coordinates; area-specific analyses select labels and can subdivide areas by coordinates.
- The method code largely consumes already preprocessed MATLAB/pickle products. For the supplied source files, equivalent transformations must be reconstructed from native fields while preserving these conventions.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` contains 174 NWB 2.x/HDF5 files organized in subject/session directories. Filenames encode subject and session and indicate behavior + ecephys + optogenetics.
- Every NWB has `intervals/trials` with 15 consistent columns: trial identifiers/times, `trial_instruction`, `outcome`, `early_lick`, task/protocol, auto/free-water flags, and photostimulation power/onset/duration.
- `acquisition/BehavioralEvents` contains timestamped go, sample, delay, presample, trial-end, left/right lick, and photostimulation start/stop events.
- `acquisition/BehavioralTimeSeries` contains side-camera jaw, nose, and tongue tracking. Tongue data columns are `(tongue_x, tongue_y, tongue_likelihood)` with explicit timestamps; values are finite, while likelihood is strongly bimodal and therefore carries visibility information.
- `units` contains raw spike times plus waveform and spike-sorting QC metrics, `unit_quality`, atlas annotation `anno_name`, electrode links, observation intervals, and `is_good_trials`.
- `is_good_trials` is a boolean unit x trial array described as manual probe-insertion-specific valid-trial annotation. In most sessions its width equals trial count; in some it is smaller, requiring reconciliation with session/probe structure before conversion.
- Electrodes include probe/electrode-group metadata and coordinates. Optogenetic sites are represented under `general/optogenetics`.
- No separate README was found in the data tree.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total, unfiltered units) | 272,227 |
| Neurons / session | 493--3,191; mean 1,564.52; median 1,571.5 |
| Subjects | 28 |
| Sessions / subject | 3--10 (174 sessions total) |
| Trials (total) | 94,990 |
| Trials / session | 264--800; mean 545.92; median 534 |

### Native Value Distributions
- Trial instruction: 48,913 right; 46,077 left.
- Outcome: 65,254 hit; 15,641 miss; 14,095 ignore.
- Early lick: 84,185 no-early; 10,805 early.
- Task: all 94,990 rows are `audio delay`, protocol 1.
- Photostimulation: 18,588 trials have 0.5-s stimulation; 76,402 use `N/A`. Power/onset/duration are stored as strings and require explicit N/A parsing.
- Unit quality: 154,948 `good`; 117,279 `multi`.
- Atlas annotation is blank for 200,922 raw units; 69,453 units have `classification=good`, indicating the anatomically registered subset. There are 2,146 detailed `anno_name` strings, often layer-resolved.
- Auto-water: 1,339 trials; free-water: 2,450 trials.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 69,943 good units (25.9% of Kilosort2 clusters) | Data-paper methods / QC white paper: region-specific classifiers yielded 69,943 good units. |
| Neurons / session | Brain areas with <10 neurons/session excluded in method-paper analyses | Method-paper reporting summary. |
| Subjects | Dataset paper reports the full cohort; supplied NWBs contain 28 subjects | Data paper and native files. |
| Sessions / subject | 173 analyzed behavioral sessions; 655 probe insertions | Data-paper methods. |
| Trials (total) | Not stated as one total | — |
| Trials / session | mean 476, range 130--785 | Data-paper methods. |
| Neural data time bin | 40-ms width, 3.4-ms stride in method paper | Method-paper Methods: “binned spikes into firing rates with a bin width of 40 ms and a stride of 3.4 ms.” |
| Behavior data time bin | video acquired at 300 Hz (~3.33 ms/frame) | Data-paper methods. |
| Reward rate | 84% correct control trials (range 65--99%) | Data-paper methods. |
| Task timing | three 150-ms tones separated by 100 ms; 1.2-s delay; 0.1-s go cue; 1.5-s answer period | `methods.txt`, Behavior and video tracking. |
| Photostimulation | late delay, final 0.5 s, ending before go cue | `methods.txt`, Photoinhibition. |
| Video choice decoding | AUC 0.51 ± 0.06 before sample; 0.66 ± 0.12 during sample/delay, n=106 sessions | Method paper, choice decoding from behavioral videos. |

### Processing Details
- Neural data were acquired at 30 kHz with Neuropixels probes and spike sorted with Kilosort2. The recent method paper computes sliding firing rates using a 40-ms window and 3.4-ms stride; this task explicitly requires 50-ms-width bins, so the target conversion must differ there.
- Task alignment is naturally defined by the auditory go cue. The delay begins 1.2 s before go cue; photoinhibition occupies the final 0.5 s of delay and ends before go.
- Side and bottom video were recorded at 300 Hz. DeepLabCut tracked tongue, jaw, and nose.
- Early-lick and no-response trials were excluded in the original behavioral/neural analyses. Session inclusion required >65% control-trial performance and at least 50 correct left and 50 correct right trials.
- Method-paper area analyses exclude area/session groups with fewer than 10 neurons.

### Curation Steps

**Neuron curation rules**:
- Fifteen quality metrics were used with five region-specific logistic-regression classifiers trained on blind manual curation labels.
- Classifier-positive `good` units were used in paper analyses. Region grouping: cortex classifier for cortex/hippocampus/olfactory/cortical subplate; striatum for striatum/pallidum; thalamus for thalamus/hypothalamus; midbrain for midbrain/pons; medulla for medulla/cerebellum.
- Reported false-alarm rates were cortex 7.8%, striatum 6.4%, thalamus 7.3%, midbrain 5.5%, medulla 4.3%.

**Trial curation rules**:
- Original analyses excluded early-lick and no-response trials and often control/special-trial subsets were selected for a particular analysis.
- For this decoder, early lick, no lick, and photostimulation are explicitly required labels/inputs, so those categories cannot be removed solely because reference analyses removed them.

### Decoders Trained
| Decoded variable | Accuracy |
|------------------|----------|
| Choice from behavioral video before sample | ROC AUC 0.51 ± 0.06 |
| Choice from behavioral video during sample/delay | ROC AUC 0.66 ± 0.12 |
| Single-neuron choice/movement analyses | Test-fold ROC AUC with regularized logistic regression; no directly comparable categorical accuracy reported |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Session count | Session dictionaries use probe-valid trial periods | 174 NWBs; one has no classifier-good units | 173 analyzed sessions | Filter `classification == good`; exclude the one session with zero retained units. This yields exactly 173 sessions. |
| Good-unit count | Reference analyses consume classifier-curated units | 69,453 units have `classification=good` and nonblank atlas labels | 69,943 good units | Use the native classifier result (69,453). The 490-unit difference is a release/version discrepancy; substituting `unit_quality` would be incorrect because it is the pre-classifier Kilosort label. |
| Unit quality field | Method code uses final curated/anatomical units | `unit_quality`: 154,948 good; `classification`: 69,453 good | Region-specific classifiers yielded final good units | Use `classification == good`. Notably, 8,475 classifier-good units have Kilosort label `multi`, confirming these fields are not interchangeable. |
| Trial validity | Reference preprocessing stores probe-specific valid trials | Ragged `obs_intervals` and `is_good_trials`; some probes cover only a subset | Analyses use valid recording periods | Reconstruct ragged observation intervals and retain trials represented by an observation row for every retained unit. Neural spikes are continuously timestamped; the observation-row stop is the behavioral trial end, not the end of neural recording. |
| Trial categories | Regular mask excludes early, auto/free water, invalid correctness, stimulation | Required labels include early lick, no lick/outcome ignore, and photostimulation | Paper often excludes early/no-response/stim trials | Retain these required categories when neural/video streams are valid; do not apply the analysis-specific regular-trial mask. |
| Neural temporal processing | Method code uses 40-ms window, 3.4-ms stride | Raw spike times available | Method paper specifies 40 ms / 3.4 ms | Task specification overrides reference here: use 50-ms-width non-overlapping bins from −2.5 to +1.5 s. |
| Tongue visibility | No explicit confidence threshold found | Likelihood is strongly bimodal; 11.87% frames ≥0.9, very little intermediate mass | DeepLabCut used, threshold unstated | Use likelihood ≥0.9 as visible. This conservative threshold is supported by the native bimodal distribution and is documented explicitly. |
| Session behavioral criteria | Regular-analysis masks and paper criteria are analysis-specific | Applying criteria directly to release labels does not select 173 sessions | >65% performance and ≥50 correct each direction | Do not additionally remove valid NWBs: observation-interval/classifier curation already reproduces 173 sessions, whereas reapplying behavioral criteria to release labels would discard 22 sessions and diverge from the released analyzed cohort. |

### Final Consistent Understanding
- Use all 173 NWBs containing classifier-good units, all classifier-good neurons, and only trials with complete requested neural coverage.
- Preserve required behavioral/stimulation categories even when original single-purpose analyses excluded them.
- Use go-event timestamps for alignment, sample/tone event timestamps for the continuous tone-time input, and native optogenetic event intervals for the stimulation input.
- Use detailed `anno_name` labels for brain-region identity, preserving the supplied atlas annotation rather than inventing coarser regions.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `units/spike_times` | `neural` | Count spikes in 80 non-overlapping 50-ms bins from −2.5 to +1.5 s around each go cue; divide by 0.05 s to Hz; float32 neuron × time | Reference preprocessing produces `fr`; task-required binning replaces 40-ms/3.4-ms sliding window | Raw spikes avoid double-smoothing. |
| Final `sample_start_times` event within each trial | `input[0]` | At each bin center, absolute time minus final sample/tone onset | Event alignment conventions | Continuous seconds from tone onset, as explicitly requested. Final onset handles early-lick replay. |
| `photostim_start_times` / `photostim_stop_times` | `input[1]` | Binary 1 when bin center lies in any stimulation interval, else 0 | Native BehavioralEvents and reference stimulation metadata | Time-varying. |
| `trial_instruction` + `outcome` | `output[0]` choice | hit → instructed side; miss → opposite side; ignore → no lick; repeat across 80 bins | Trial-label logic | Values: left=0, right=1, no lick=2. |
| `outcome` | `output[1]` | Direct categorical mapping repeated across bins | Native trials | ignore=0, miss=1, hit=2. |
| `early_lick` | `output[2]` | Direct mapping repeated across bins | Native trials / regular-mask logic | no=0, yes=1. |
| side-camera tongue y + likelihood | `output[3]` | Linear interpolation at bin centers; visible if interpolated likelihood ≥0.9; visible y discretized by session-wide visible-frame 40th/60th percentiles; invisible=3 | Marker alignment reference | Values low=0, middle=1, high=2, not visible=3. |
| NWB subject ID | `subjects`, `subject_idx` | Unique sorted strings and session index | Native NWB metadata | 28 subjects expected. |
| `units/anno_name` | `brain_regions`, `brain_region_idx` | Preserve detailed Allen annotation strings for classifier-good units | CCF label use in reference code | Global sorted region vocabulary. |

### Key Decisions
1. **Neuron curation**: retain only `classification == good`, the final region-specific classifier used by the paper; do not substitute Kilosort `unit_quality`.
2. **Session curation**: exclude the sole NWB with zero classifier-good units, reproducing 173 paper sessions.
3. **Trial validity**: reconstruct each unit's ragged `obs_intervals`, map intervals to trial starts, apply `is_good_trials`, require a mapped observation row and manual good flag for every retained unit, and intersect across retained units. This yields 92,801 trials. Observation-interval stop times are behavioral trial ends and must not be treated as neural-recording ends.
4. **Required categories retained**: early, miss/ignore, auto/free-water, and photostimulation trials remain if streams are valid because they are required outputs/inputs.
5. **Tone replay**: sample events can outnumber trials. Select the final sample-start event between trial start and go cue; no trial lacks such an event.
6. **Output temporal form**: all outputs are 4×80 for decoder compatibility. Trial-level outputs are repeated; tongue is genuinely time-varying.
7. **Tongue thresholds**: percentiles are computed from all session frames with likelihood ≥0.9, not merely retained trial windows, matching “over the session.” Boundary convention: class 0 below p40, class 1 from p40 through p60, class 2 above p60.
8. **Storage**: float32 firing rates are required by decoder.py. Full data are large; conversion is session-streamed and pickle protocol 4/5 is used.

### Planned Sanity Checks
- [ ] Raw spike spot check: independently histogram one neuron/trial with `np.histogram`; compare converted rate using `np.allclose`.
- [ ] Input spot check: independently compute bin-center minus final sample onset and stimulation interval membership; compare with `np.allclose`.
- [ ] Output spot check: independently derive three trial labels and interpolate/discretize tongue y; compare with `np.allclose`.
- [ ] Assert 80 bins, 50-ms spacing, exact −2.5/+1.5 edges, finite neural/input values, and categorical ranges.
- [ ] Confirm 173 sessions, 28 subjects, 69,453 session-units before per-trial intersection, and at least two retained trials/session.
- [ ] Compare categorical distributions and reference dataset/session/unit counts.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with required `--full`, `--sample`, and `--show-processing` options. The script uses direct h5py access, reconstructs NWB ragged arrays, validates unit/trial coverage, bins raw spikes, aligns inputs/outputs, emits processing plots, and asserts all shapes/ranges.

Code inefficiencies identified:
A naïve neuron × trial × spike histogram loop would be too slow and duplicate I/O. Full float32 payload is expected to be several GiB.

Code speedups added:
Spike assignment is vectorized across all trials within each neuron using searchsorted/bincount; NWBs are opened once per session; arrays are preallocated; output is streamed session-by-session into the result. Single-session smoke conversion took ~0.28 s before pickle serialization.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics (final corrected sample)
| Statistic | Value |
|-----------|-------|
| Neurons (total session-units) | 834 |
| Neurons / session | 459, 375 |
| Subjects | 1 |
| Sessions / subject | 2 |
| Trials (total) | 527 |
| Trials / session | 368, 159 |
| Timepoints / trial | 80 |
| Time from tone onset | continuous, replay-aware; finite |
| Photostimulation range | [0, 1] |
| Output classes present | all required classes globally |

### Processing Plots Review
Two `processing_<session_id>.png` files were created. Raster, input traces, tongue categories, and firing-rate histograms have the expected shared −2.5 to +1.5 s go-aligned axis. Final sample verification reports no errors or warnings and no all-zero population trials.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| Vectorized spike-to-trial/bin assignment per neuron | Avoids nested spike × trial histogram loops |
| Single h5py open per session and preallocation | Minimizes I/O/allocation overhead |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Conversion before serialization | generally 0.2--2.0 s/session | comfortably under 15 minutes including serialization |

Sample pickle, conversion log, verification log, and processing plots all exist.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Final Corrected Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-----------------------|-------------------------|
| Lick direction choice | 0.7337 | 0.6259 |
| Outcome | 0.7532 | 0.6495 |
| Early lick | 0.8368 | 0.7424 |
| Tongue y-position | 0.6849 | 0.5213 |

Loss decreased from ~7 to 0.569431; test loss was 0.728457. All validation accuracies exceed chance. These final results include the corrected miss-trial retention.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Iteration 1 issue and correction
The initial full conversion retained 75,645 trials but only 766 miss trials. Critical inspection showed that `obs_intervals` stop times are behavioral trial ends, not neural recording ends. Requiring go+1.5 s to precede those stops preferentially removed misses/short trials. The converter was corrected to require mapped observation-row presence and `is_good_trials`, without treating behavioral stop as neural coverage end. Completely zero population windows are excluded as missing neural segments. Corrected sample reconversion restored miss trials and passed verification; independent `np.histogram` checks matched converted rates exactly.

### Output Files
- `converted_data.pkl`: 11.17 GiB
- `verification_full_out.txt`: created; format valid with no errors or warnings

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total neurons | 69,943 good | final curated units | 69,453 `classification=good` | 69,453 | Native release exact; paper differs by 490 |
| Mean neurons/session | not stated | area analyses require ≥10/area | 401.46 classifier-good | 401.46 (range 90--923) | Yes |
| Subjects | full cohort | session-based | 28 | 28 | Yes |
| Sessions | 173 analyzed | valid preprocessed sessions | 174 NWBs, one with no classifier-good units | 173 | Yes |
| Trials (total) | not stated | analysis-specific subsets | 94,990 raw; 92,801 probe/manual-valid | 90,378 after removing 2,423 all-zero neural windows | Justified |
| Trials/session (mean) | 476 (paper analysis cohort) | subset dependent | 545.92 raw | 522.42 (range 159--800) | Reasonable; decoder retains required categories |
| Tone-time range | task dependent | go aligned | replay/variable epochs present | −1.525 to 11.894 s | Native-derived |
| Photostimulation | final 0.5 s of delay | binary trial/time metadata | 18,588 raw stimulated trials | binary [0,1], valid-trial subset | Yes |
| Choice distribution | approximately balanced instruction | trial labels | raw instructions balanced | left 42.83%, right 42.23%, no lick 14.95% | Yes |
| Outcome distribution | 84% correct control analysis subset | correctness used for regular masks | hit 68.70%, miss 16.47%, ignore 14.84% raw | hit 68.45%, miss 16.60%, ignore 14.95% | Yes |
| Early lick | excluded in many analyses | regular mask excludes it | 11.37% raw | 11.58% | Yes |
| Tongue classes | DeepLabCut tracking | aligned markers | likelihood strongly bimodal | 5.71% low, 3.19% middle, 6.48% high, 84.63% invisible | Yes |

The corrected full verification reports: `Data format is valid, no errors or warnings.`

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Corrected `/app/verification_full_out.txt` states `Data format is valid, no errors or warnings.` The initial warning about all-zero population trials was addressed by excluding those missing-neural windows.
2. **Independent raw-data sanity checks**: `/app/cache/raw_sanity_checks.py` loads original NWBs directly and does not call conversion functions. For three spot checks spanning sessions 0, 86, and 172, `np.allclose()` passed for: raw-spike histogram firing rate, time from final replay-aware sample onset, photostimulation state, choice/outcome/early labels, and interpolated/discretized tongue y. All 15 comparisons passed.
3. **Reference-code comparison**:
   - Loading: converter uses native NWB spike/event/tracking fields; reference code consumes equivalent preprocessed MATLAB/pickle fields.
   - Neuron filtering: converter uses final `classification == good`, matching the paper's region-specific classifier curation.
   - Trial filtering: converter maps `obs_intervals` rows and intersects manual `is_good_trials`; unlike analysis-specific regular masks, it retains required early/no-response/stimulation categories.
   - Alignment: go-event timestamps define zero in both converter and reference analyses.
   - Binning: converter uses required non-overlapping 50-ms raw-spike bins; reference uses 40-ms/3.4-ms sliding rates, an explicitly required difference.
   - Inputs: native sample and photostimulation events are used; final sample onset handles the documented replay behavior.
   - Outputs: native trial labels and DeepLabCut tongue tracking are used; choice is actual response, derived from instruction/outcome.
4. **Key statistics**: 28 subjects, 173 sessions, 69,453 classifier-good units, and detailed region counts exactly match the supplied curated data. The 69,943 paper count differs by 490 units and is attributed to release/version differences. Converted outcome and early distributions closely match raw distributions.
5. **Edge cases/off-by-one checks**: exactly 80 half-open 50-ms bins cover [−2.5,+1.5); stimulation uses [start,stop); every final sample onset is selected within trial start/go bounds; output boundaries use y<p40, p40≤y≤p60, y>p60; all neural/input/output arrays have fixed time length and correct dtype.
6. **Global invariant checks**: all neural arrays are float32 neuron×80, all inputs float32 2×80, all outputs integer 4×80, photostimulation is binary, and no retained population window is entirely zero.

### Issues Found and Resolved
- **Observation-interval semantics**: Initial code treated each `obs_intervals` stop as the end of continuous neural coverage. These are behavioral trial ends, causing severe preferential removal of miss trials (only 766 remained). Fixed by using interval rows to identify represented trials without requiring go+1.5≤trial stop. Corrected outcome distribution is ignore 14.95%, miss 16.60%, hit 68.45%.
- **All-zero population windows**: Corrected broad trial retention exposed 2,423 windows with no spikes in any of 90--923 curated neurons. These occurred across outcomes and indicate missing neural segments. They were excluded. Full verification then reported no warnings.
- **Iteration re-check**: sample conversion, sample verification, sample training, full conversion, full verification, all raw sanity checks, and all global checks were rerun after fixes; all passed.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes; 24.823056 at epoch 1 to 0.683991 at epoch 200
- Test loss: 0.687529
- Device: CUDA
- Split: 72,249 training trials; 18,129 validation trials

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-----------------------|-------------------------|-------|
| Lick direction choice | 0.7032 | 0.6738 | chance 0.3333 |
| Outcome | 0.6947 | 0.6553 | chance 0.3333 |
| Early lick | 0.7928 | 0.7533 | chance 0.5000 |
| Tongue y-position | 0.6765 | 0.6114 | chance 0.2500 |

The full training script finished successfully and sample plots were requested with `--plot-samples`.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Achieved Validation Balanced Accuracy | Chance | Accuracy / Chance | Expectation from Papers |
|----------|---------------------------------------|--------|-------------------|-------------------------|
| Lick direction choice | 0.6738 | 0.3333 | 2.02× | Method paper reports behavioral-video choice ROC AUC 0.66 ± 0.12 during sample/delay; metric/model differ, but achieved neural decoding is consistent with substantial choice signal. |
| Outcome | 0.6553 | 0.3333 | 1.97× | No directly comparable paper accuracy reported. |
| Early lick | 0.7533 | 0.5000 | 1.51× | No directly comparable paper accuracy reported. |
| Tongue y-position | 0.6114 | 0.2500 | 2.45× | Paper establishes widespread movement encoding but reports no comparable four-class accuracy. |

### Checks
1. **Accuracy versus chance**: Every output is above chance and at least 1.5× chance (early lick is 1.507×). No below-chance output exists.
2. **Accuracy versus papers**: The only directly relevant reported decoder is choice from behavioral video (ROC AUC 0.66 ± 0.12). Our choice balanced accuracy is 0.6738 using neural data and three classes, so it is not numerically identical but is consistent with the expected signal magnitude. No paper reports directly comparable categorical accuracies for outcome, early lick, or discretized tongue y.
3. **Train/validation gap**: Train/validation ratios are choice 1.044, outcome 1.060, early lick 1.052, tongue 1.106; all are far below the 1.5 concern threshold.
4. **Raw label verification**: Step 10 independently checked three trials spanning the dataset for all output values against original NWBs.
5. **Temporal synchronization**: Processing plots and raw `np.allclose` checks verify go alignment, replay-aware sample onset, stimulation intervals, and tongue timestamps.
6. **Class variation**: Full distributions are nondegenerate: choice 42.83/42.23/14.95%; outcome 14.95/16.60/68.45%; early 88.42/11.58%; tongue 5.71/3.19/6.48/84.63%.
7. **Filtering/processing match**: Classifier-good neuron selection, trial observation mapping, raw-spike binning, and event alignment were rechecked against papers/code and native data.

### Issues Found and Resolved
- No new conversion issue was indicated by full decoder accuracy.
- The modestly hardest required class is early lick (1.51× chance), but raw labels, class balance, temporal alignment, and filtering all passed independent checks; no unsupported conversion change was warranted.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with loading instructions, format, processing, statistics, and decoder results
- [x] cache/ folder created with independent sanity-check script/output
- [x] `cache/README_CACHE.md` documents cached investigation files
- [x] All required conversion, validation, training, plot, and documentation files verified
- [x] CONVERSION_NOTES.md reconciled with the final corrected conversion
