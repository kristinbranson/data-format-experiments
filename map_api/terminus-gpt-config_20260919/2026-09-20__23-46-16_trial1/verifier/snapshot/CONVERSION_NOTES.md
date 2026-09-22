# Dataset Conversion Notes

## Overview
- **Dataset**: Brain-wide neural activity underlying memory-guided movement
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment verification:
- Python 3.13.15
- NumPy 2.4.4
- PyTorch 2.6.0+cu124
- Required notes-file checkpoint passed (`ls -la /app/CONVERSION_NOTES.md`).

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

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `process_one_sess` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING/PROCESSING | Loads one legacy MAP export session, extracts trial/spike/unit metadata, aligns spikes to go cue, bins rates, and saves a session dictionary. |
| `sliding_histogram` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Histograms spike times over trials/neurons; optional division by bin width yields Hz. Reference defaults are 100-ms windows with 50-ms stride. |
| `helper_get_neuron_id_area` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | CURATION | Selects units by anatomical region/side and QC mode, including classifier-based QC. |
| `helper_filter_by_neuron_id` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | CURATION | Subsets firing rates, spike times, unit metadata, QC, and CCF IDs while retaining trial variables. |
| `process_all_sess_parallel` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Parallel session preprocessing driver (five workers in the reference main block). |
| `create_4fold_trial_type_mask` | `VideoAnalysisUtils/functions_for_r2.py` | PROCESSING | Builds trial strata from left/right trial type and correct/error response for cross-validation. |
| `temporal_alignment_embed_and_ephys` | `VideoAnalysisUtils/functions_for_r2.py` | PROCESSING | Aligns video embedding times with ephys firing-rate times. |
| `get_regular_trial_mask` | `VideoAnalysisUtils/functions_for_r2.py` | CURATION | Defines reference “regular” trials: no early lick, auto/free water, no-response, or stimulation. |

### Notes
- Repository: MapVideoAnalysis for Wang et al., *Brain-wide analysis reveals movement encoding structured across and within brain areas*; data source is DANDI 000363.
- This is electrophysiology, not calcium imaging; delta-F/F is not applicable.
- Reference preprocessing represents each neuron's trial spike times relative to go cue (go cue = 0), then computes firing rate arrays ordered `(time bins, trials, neurons)`.
- Reference main defaults: `bw=0.1 s`, `stride=0.05 s`, window `[-3.0, 3.5] s`, and `qc_mode='classifier'`.
- Retained trial fields include early-lick, auto-water, free-water, lick directions/times, correctness (`1` correct/free water, `0` error, `-1` no response), stimulation (`power`, type, laser on/off), trial type (`1` left, `0` right), and sample/delay duration.
- Retained unit fields include unit quality, location/electrode/shank/cell type/probe, CCF assignment, amplitude, presence ratio, amplitude cutoff, ISI violation, mean firing rate, and drift metric.
- The method-paper code uses legacy exported session structures and processed pickle files. The present task instead explicitly requires direct NWB access through `pynwb`; source concepts and filtering semantics will be matched against NWB fields.
- The paper's movement-prediction analyses use a strict regular-trial mask excluding early lick, no response, stimulation, and special-water trials. Those exclusions cannot be copied wholesale here because early lick, no lick/miss, and photostimulation are required decoder targets/inputs. This task-mandated distinction will be planned explicitly in Step 5.
- The decoder task mandates 50-ms-width bins and `[-2.5, +1.5] s`; therefore non-overlapping 50-ms bins will supersede the reference 100-ms sliding analysis window while preserving 50-ms temporal sampling.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` contains 174 NWB files organized as one subject directory per mouse (`sub-<id>/sub-<id>_ses-<timestamp>_behavior+ecephys+ogen.nwb`).
- All files were inspected exclusively with `pynwb.NWBHDF5IO(..., load_namespaces=True)`; `h5py` was not used.
- All 174 sessions share one trial/unit schema.
- `nwbfile.trials` has columns: start/stop time, trial number and UID, photostimulation onset/power/duration, task/protocol, trial instruction, early lick, outcome, auto-water, and free-water.
- `nwbfile.units` contains ragged absolute `spike_times`, ragged per-unit `is_good_trials`, legacy `unit_quality`, classifier `classification`, anatomical `anno_name`, electrode links, waveforms, and extensive QC metrics (presence ratio, amplitude cutoff, ISI violation, mean firing rate, drift, isolation metrics, etc.).
- `acquisition['BehavioralEvents']` contains start/stop timestamps for presample, sample (auditory tone), delay, go, trial end, and photostimulation, plus left/right lick timestamps. `go_start_times` has exactly one timestamp per trial in every session.
- `acquisition['BehavioralTimeSeries']['Camera0_side_TongueTracking']` exists in all sessions. Data shape is `(video frames, 3)` with columns `(tongue_x, tongue_y, tongue_likelihood)`, explicit timestamps, arbitrary spatial units, and median frame interval about 3.4 ms (~294 Hz). Coordinates are finite even when tracking confidence is tiny; visibility must therefore use the likelihood channel.
- Additional optional tracking streams include jaw/nose in all sessions, whisker in 20 sessions, Camera3 tracking in 3, and lick-port tracking in 4.
- The experiment is uniformly labeled `audio delay`.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total, unfiltered) | 272,227 |
| Neurons / session (unfiltered) | mean 1,564.5; median 1,571.5; range 493–3,191 |
| Classifier-good neurons | 69,453 |
| Legacy `unit_quality=good` neurons | 154,948 |
| Subjects | 28 |
| Sessions / subject | 3–10; 174 total sessions |
| Trials (total) | 94,990 |
| Trials / session | mean 545.9; median 534; range 264–800 |

### Native Value Distributions and Data Quality
- Trial instruction: right 48,913; left 46,077.
- Outcome: hit 65,254; miss 15,641; ignore 14,095.
- Early lick: no early 84,185; early 10,805.
- Photostimulation event intervals: 18,588 session-wide events.
- Unit classifier: good 69,453; unlabelled 200,922; NaN 1,852. Every classifier-good unit has a nonblank anatomical annotation.
- Classifier-good and legacy-quality-good overlap: 60,978 units, confirming these are distinct QC systems and supporting use of the paper's classifier QC.
- Per-unit valid trials: 565 classifier-good units have at least one false `is_good_trials` entry (64,612 invalid unit-trial pairs), heavily concentrated in four sessions. This validity information must be handled rather than silently treating spikes outside valid recording periods as zero.
- Anatomical labels comprise 295 raw Allen-style names including layer-qualified regions and broader fallback labels.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote / context |
|-----------|-------|------------------------|
| Neurons (total, classifier-good) | 69,943 | Data-paper methods: “Overall, the dataset consisted of 69,943 good units...” |
| Neurons / session | about 404 mean from paper totals | 69,943 units / 173 behavioral sessions. |
| Subjects | 28 in provided archive; paper describes the MAP mouse cohort | NWB subject census; paper methods describe head-fixed mice and multi-probe recordings. |
| Sessions | 173 behavioral sessions | Data-paper methods: “...recorded across 173 behavioral sessions...” |
| Probe insertions | 655 | Same data-paper methods sentence. |
| Trials (total) | not stated as a single paper-wide total | Native NWB census is therefore the authoritative value for conversion. |
| Neural data time bin | 100-ms window, 50-ms stride in reference code | `preprocessing_DJ_2022Aug.py` defaults `bw=0.1`, `stride=0.05`; requested decoder instead mandates 50-ms-width bins. |
| Behavior video sampling | approximately 294 Hz | NWB timestamps have median 3.4-ms spacing; method paper analyzes high-speed side-view video. |
| Task timing | 0.6-s auditory sample, delay, go cue, response | Data-paper text defines sample and response epochs and aligns population analyses to go cue. |
| Photostimulation prevalence | approximately 25% of randomly interleaved trials in 17 VGAT-ChR2 mice | `methods.txt`, photoinhibition protocol. |
| Bilateral photostimulation performance | 83.2% control to 71.7% stimulation | `methods.txt`; 17 mice, 93 sessions. |
| Unit QC yield | 25.9% of Kilosort2 clusters | Data-paper methods. |
| QC false-alarm rates | cortex 7.8%, striatum 6.4%, thalamus 7.3%, midbrain 5.5%, medulla 4.3% | Data-paper methods and QC white paper. |

### Processing Details
- Animals performed an auditory delayed-response task. Auditory tone/sample information precedes a delay; the go cue is time zero for the ephys preprocessing and response analyses.
- The data paper analyzes sample (0.6 s), delay, and response epochs. The requested conversion uses the prescribed go-aligned `[-2.5, +1.5] s` interval.
- Reference code computes rates with a 100-ms sliding histogram every 50 ms. The decoder specification requires 50-ms-width bins, so conversion will use 80 non-overlapping bins while maintaining the reference 50-ms sampling grid.
- ALM photoinhibition was randomly interleaved, typically during the final 0.5 s of delay and ended before go cue. NWB event timestamps/power provide the precise time-varying input rather than assuming protocol timing.
- The method paper tracks tongue, jaw, and nose with DeepLabCut-like marker position/confidence streams and predicts firing rates from video. The NWB tongue stream explicitly stores `(x, y, likelihood)` at each frame.
- Reported paper decoding concerns population choice decoding and single-neuron ROC-AUC/video prediction; it does not report directly comparable accuracies for the supplied multi-output decoder (choice, outcome, early lick, tongue-y category).

### Curation Steps

**Neuron curation rules**:
- Use the published region-specific logistic-regression classifier output (`classification == 'good'`), trained from 15 spike-sorting metrics and manual labels. The five classifier families map cortex/related structures, striatum/pallidum, thalamus/hypothalamus, midbrain/pons, and medulla/cerebellum.
- Do not substitute the broader legacy `unit_quality == 'good'`; the classifier labels are explicitly those used for the paper analyses.
- Respect per-unit `is_good_trials` validity. Because the target requires a fixed neuron matrix per session, units invalid on retained trials require a consistent session-level handling decision in Step 5.
- Method-paper area-wise analyses excluded brain-area/session groups with fewer than 10 neurons. This is relevant for area-stratified figures but not automatically a reason to remove units from a joint whole-session decoder.

**Trial curation rules**:
- Reference movement/video analyses define “regular trials” by excluding early lick, auto-water, free-water, no response, and photostimulation.
- The requested decoder explicitly needs early-lick, no-lick/miss, and photostimulation labels, so these trial classes must be retained. Auto/free-water trials remain candidates for exclusion because their outcomes are not standard instructed behavior.
- Trials must have valid go timestamps and enough neural/video coverage for the full requested window.

### Decoders Trained
| Decoded variable | Accuracy |
|------------------|----------|
| Population choice in data paper | Reported as time-resolved decoding curves/figures; no single value directly matching this decoder and split scheme. |
| Single-neuron choice/movement in method paper | ROC-AUC analyses; not directly comparable to categorical balanced accuracy here. |
| Video-to-firing-rate in method paper | Cross-validated explained variance (R²), not output-classification accuracy. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Sessions | Process all exported sessions | 174 NWBs, but exactly one session has zero classifier-good units; 173 have usable units | 173 behavioral sessions | Exclude the zero-good session. This exactly reconciles usable session count. |
| Classifier-good units | Use classifier QC | 69,453 across the 173 usable sessions | 69,943 | Treat as an archive-version/paper-count difference (or transposed published digits); use the labels actually stored in the supplied NWBs. Do not manipulate QC to force a paper total. |
| Unit QC field | `qc_mode='classifier'` | Both broad `unit_quality` and stricter `classification` exist | Paper says classifier-good units were analyzed | Use `classification == 'good'`; all 69,453 such units have anatomy. |
| Unit trial validity | Legacy processed arrays carry trial spikes; reference region helper does not expose a fixed-matrix solution for invalid unit-trials | 565 classifier-good units have invalid trials (64,612 pairs), concentrated in four sessions | Quality/drift control is required | For a fixed neuron set per session, retain only classifier-good units valid on every retained trial; do not encode invalid periods as zero firing. |
| Trial filtering | Movement/video “regular trial” mask excludes early lick, no response, stimulation, and special-water trials | NWB contains all these labels | Same strict mask for the paper's specific analyses | Retain early lick, ignore/no-lick, miss, and stimulation because the decoder explicitly predicts/uses them. Exclude auto/free-water because they are nonstandard assisted trials. |
| Neural bins | 100-ms window with 50-ms stride | Absolute spike timestamps support arbitrary bins | Reference analysis used sliding rates | Decoder mandate takes precedence: use non-overlapping 50-ms bins, expressed as Hz, over exactly `[-2.5, +1.5)`. |
| Go alignment | Processed spike times use go = 0 | `go_start_times` count equals trials in all 174 sessions and every go lies inside its corresponding trial | Go-aligned response analyses | Pair trial row `i` with go event `i`; this mapping is exact. |
| Tone onset | Reference epochs assume nominal sample near -1.85 s | Repeated/aborted sample states create extra timestamps in 5,534 trials | Auditory sample is 0.6–0.65 s before delay | Select the 0.65-s sample state whose stop is nearest the final delay start, where the final delay is identified by its stop at go. This resolves repeated early-lick sample states and variable delay durations. |
| Tongue visibility | Marker analyses retain x/y; no explicit confidence threshold found in supplied code | Tongue stream has `(x,y,likelihood)` and strongly bimodal likelihood | Video markers are estimated with pose tracking | Use likelihood ≥0.9 as visible. Threshold 0.95 gives nearly identical fractions, so 0.9 is conservative without materially changing labels. |
| Brain-area minimum | Method-paper area analyses exclude area/session groups with <10 neurons | Whole-session decoder uses all retained neurons jointly | Reporting summary states the <10 rule | Do not drop small regional groups: that rule was for area-level statistical analyses, whereas dropping them would discard valid neural predictors and distort the required brain-region index. |

### Final Consistent Understanding
- The supplied NWBs are a near-paper archive revision with one additional session containing no classifier-good neurons and a 490-unit difference from the published aggregate. Native classifier labels are authoritative for reproducibility.
- Neural spikes, behavioral state events, trial labels, and video coordinates share the same session clock. Go events map one-to-one by trial row; video and spike streams will be indexed by absolute time around each go.
- Published classifier QC and anatomical annotations should be used. Task-specific requested variables require broader trial retention than the method paper's regular-trial analyses.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `units.spike_times` | `neural` | Count spikes in 80 non-overlapping 50-ms bins spanning `[-2.5,+1.5)` relative to go; divide by 0.05 to Hz; transpose to `(neurons,time)` float32 | `sliding_histogram`, `process_one_sess` | Task-required width replaces reference 100-ms sliding window. |
| `BehavioralEvents.go_start_times` | alignment | Trial-row-matched absolute go timestamp | `process_one_sess` | Exactly one per trial and all lie within corresponding trial. |
| `sample_start_times` | `input[0]` | Select the sample onset whose sample stop is nearest the final pre-go delay start; for each bin center store continuous seconds since tone onset: `absolute_bin_center − tone_onset` | reference sample epoch definitions | Called `time from tone onset (s)`. |
| `photostim_start_times/stop_times` | `input[1]` | Binary at each bin center: 1 iff center is inside the trial's 0.5-s photostimulation interval | reference stimulation field | 18,588 trials have exactly one interval. |
| `trial_instruction` + `outcome` | `output[0]` choice | hit → instructed direction; miss → opposite direction; ignore → no lick. Encode left=0, right=1, no lick=2 and repeat over 80 bins | reference correctness/trial-type fields | Miss is an incorrect directional response; ignore is no response. |
| `trials.outcome` | `output[1]` | ignore=0, miss=1, hit=2; repeat over bins | reference correctness | Required order matches task wording. |
| `trials.early_lick` | `output[2]` | no early=0, early=1; repeat over bins | regular-trial metadata | Retained as requested target rather than filtered out. |
| `Camera0_side_TongueTracking` | `output[3]` | Nearest video frame to each bin center; likelihood <0.9 → 3 (not visible); otherwise y below visible-session 40th percentile →0, 40th–60th →1, above 60th →2 | marker alignment code | Percentiles use all visible frames in the session, not low-confidence placeholder coordinates. |
| `subject.subject_id` | `subjects`, `subject_idx` | Unique sorted subject IDs and session index lookup | — | Session order is sorted NWB path order. |
| classifier-good `anno_name` | `brain_regions`, `brain_region_idx` | Preserve raw Allen-style anatomical names; create global sorted vocabulary and per-session integer indices | `helper_get_neuron_id_area` | Layer-qualified labels are retained rather than lossy collapsing. |

### Key Decisions
1. **Unit QC**: Retain only `classification == 'good'`, exactly matching the classifier used in the papers. Do not use the broader legacy `unit_quality` field.
2. **Invalid recording periods**: For each session, intersect `is_good_trials` over all classifier-good units and drop trials where any retained unit is invalid. This affects only four sessions and removes 476 standard trials while preserving 565 curated neurons that would otherwise be lost.
3. **Special-water trials**: Exclude `auto_water == 1` or `free_water == 1`, because these are assisted/nonstandard trials and confound ordinary outcome/choice. The union contains 3,766 trials (23 overlap), before unit-validity filtering.
4. **Required trial classes**: Retain early-lick, ignore/no-lick, miss, hit, control, and photostimulation trials because they define requested inputs/outputs.
5. **Session inclusion**: Exclude the single session with zero classifier-good units. Require at least two retained trials; all remaining sessions satisfy this.
6. **Output shape**: Use integer `(4,80)` arrays. Choice/outcome/early-lick are repeated per-trial labels, while tongue category varies in time. This mixed representation is supported by the decoder and keeps all outputs aligned.
7. **Time coordinates**: Bin edges are `[-2.5,-2.45,...,+1.5]`; values correspond to centers `[-2.475,...,+1.475]`. Metadata records the half-open interval.
8. **Firing-rate dtype**: float32 Hz minimizes memory while preserving exact integer-count multiples of 20 Hz.
9. **Tone event repeats**: Repeated state-machine sample events are resolved from the state chain: final delay stop nearest go, then sample stop nearest that delay start. Selected sample durations are 0.65 s; this also handles variable delay durations.
10. **Tongue visibility**: likelihood ≥0.9. Likelihood is strongly bimodal and 0.95 produces nearly identical visible fractions; 0.9 avoids treating low-confidence placeholder coordinates as behavior.
11. **Photostimulation bin state**: Evaluate at bin centers, consistent with representing each 50-ms bin by one timepoint.
12. **Neural binning efficiency**: Flatten all trial edge arrays (monotonic because trial windows do not overlap), use `np.searchsorted` into each neuron's absolute spike train, reshape cumulative indices, and difference along each trial. This exactly matches direct histograms without nested trial loops.
13. **Short `is_good_trials` vectors**: Eight sessions store validity only for a contiguous neural-recording block (3,401 good units), which may begin late or end early. Map stored flags to go-cue trials inside each unit's `obs_intervals`; never interpret uncovered trials as recorded zero-spike data.

### Planned Sanity Checks
- [ ] Compare vectorized neural bins with direct `np.histogram` for selected raw NWB neuron/trial pairs using `np.allclose()`.
- [ ] Compare converted tone-time input against raw selected sample onset and absolute bin centers using `np.allclose()`.
- [ ] Compare converted photostimulation binary vectors against raw event intervals using `np.allclose()`.
- [ ] Compare converted categorical outputs with raw trial labels and raw nearest-frame tongue data using `np.allclose()`.
- [ ] Verify all trial matrices are `(n_neurons,80)`, inputs `(2,80)`, outputs `(4,80)`, finite, and categorical ranges are exact.
- [ ] Verify source/converted subject, session, trial, neuron, and class totals after documented filters.
- [ ] Verify all retained sessions have at least two trials and every retained neuron is valid on every retained trial.
- [ ] Plot neural population rate, continuous tone time, photostimulation state, tongue y/likelihood, thresholds, and categorical tongue output for up to two sessions.

---

## Step 6: Script Development
**Status**: COMPLETE

Implementation notes:
- `/app/convert_data.py` uses `pynwb.NWBHDF5IO` exclusively and supports `--full` (default), `--sample`, and `--show-processing`.
- Internal assertions validate all session/trial dimensions, finite values, dtypes, and categorical ranges before serialization.
- First runtime iteration revealed short per-unit validity vectors; this was investigated across all NWBs and fixed by treating missing trailing entries as invalid.

Code inefficiencies identified:
- A nested trial × neuron histogram would require billions of Python-level operations.
- Loading full video arrays and rates creates substantial but manageable per-session memory use; objects are released when each NWB context closes.

Code speedups added:
- Spike counts use `np.searchsorted` over one flattened, monotonic trial-edge array per neuron, vectorizing across trials.
- Inputs/outputs and nearest video-frame assignment are vectorized over all trial-bin centers.
- Data use float32 for neural/input arrays and int64 only where required for categorical decoder targets.
- Sample timing: 2 sessions, initially 514 trials and finally 513 after boundary safeguard, 834 neurons in 2.19 s including plots and writing; conversion compute was 1.27 s and 0.82 s/session.

Implementation iteration 1:
- Initial sample run converted session 1 but stopped on session 2 because `is_good_trials` had 160 entries for 480 behavioral trials.
- Initial fix treated short vectors as initial trial blocks; sample conversion then passed.

Implementation iteration 2 (Step 10 critical review):
- Full validator reported 127 all-zero-neural trials across three sessions.
- Raw `obs_intervals` showed seven short-mask sessions recorded initial blocks, but `SC026_20190807_134913_s20` recorded a late block (source trials 125–629). Right-padding had therefore misassigned 505 flags to trials 0–504.
- Tested all 3,401 short-mask classifier-good units: observation-overlapping go-cue trial counts exactly equal stored validity-vector lengths (zero mapping failures).
- Fix: map short validity vectors onto source trials whose go cues lie in each unit's `obs_intervals`; preserve full-length masks directly. Remove any residual trial whose complete retained-neuron rate matrix is zero (recording-boundary safeguard).
- This replaces the earlier initial-block assumption and will be revalidated on sample and full data.
- Tone review found 183 trials where nearest-nominal onset differed from the final sample→delay state (typically valid 0.3-s rather than 1.2-s delays). Fix: select tone from the explicit sample/delay state chain, which is semantically correct for “tone onset.”

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total session-sum) | 834 |
| Neurons / session | 459, 375 |
| Subjects | 1 (`440956`) |
| Sessions / subject | 2 |
| Trials (total) | 513 |
| Trials / session | 354, 159 |
| Neural shape/range | `(n_neurons,80)`; 0 to count-derived multiples of 20 Hz |
| Tone-time input range | session ranges approximately `[-0.6,5.7]`, `[-0.6,5.5]` s |
| Photostimulation range | `[0,1]` in both sessions |
| Choice fractions | s1 `[0.429,0.379,0.192]`; s2 `[0.512,0.475,0.013]` |
| Outcome fractions | s1 `[0.192,0.370,0.438]`; s2 `[0.013,0.138,0.850]` |
| Early-lick fractions | s1 `[0.949,0.051]`; s2 `[0.925,0.075]` |
| Tongue category fractions | s1 `[0.047,0.025,0.056,0.873]`; s2 `[0.068,0.036,0.076,0.819]` |

### Processing Plots Review
- Two PNGs were created, one per sample session.
- Population rates, tone-relative time, photostimulation state, raw nearest-frame tongue y, likelihood threshold, percentile thresholds, and final category are shown on a common go-aligned x-axis.
- No temporal discontinuity or category/threshold inconsistency was observed from generated traces.
- High not-visible fraction is expected: session-wide tongue visibility is only 10.5–14.1%, because tongue position is confidently tracked primarily during protrusion/licking.
- Atypical tone-time maxima arise from genuine restarted/aborted source state sequences; nearest canonical sample onset is selected uniquely and never occurs after go.

### Format Validation
- `/app/verification_sample_out.txt` created.
- Errors: none.
- Warnings: none.
- Decoder validator reports 80 timepoints for every trial, proper input/output ranges, and complete brain-region indexing.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| Searchsorted spike binning vectorized across trials | Avoids nested trial × neuron histogram calls |
| Vectorized video nearest-frame lookup and output construction | Avoids per-frame Python loops |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Conversion compute | 0.82–1.27 s in sample | ~3–4 minutes conservatively for 174 files, including larger sessions and I/O |
| Sample pickle write | 0.07 s for 0.068 GiB | ~10–30 s for expected ~10 GiB |
| Total full conversion estimate | scaled by neural matrix elements (~150× sample) | approximately 5–7 minutes, below 15-minute optimization threshold |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None
- Training completed all 200 epochs and `train_decoder.py` exited successfully.

### Training Progress
- Loss decreased monotonically overall from approximately 7 at initialization to 0.5788 at epoch 200.
- Held-out test loss: 1.2071.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-----------------------|-------------------------|--------|
| Lick direction choice | 0.7415 | 0.6118 | 0.3333 |
| Outcome | 0.7714 | 0.6443 | 0.3333 |
| Early lick | 0.8307 | 0.7326 | 0.5000 |
| Tongue y-position | 0.6674 | 0.5363 | 0.2500 |

All outputs exceed chance, including by at least 1.47× for early lick and over 1.8× for the other validation tasks. Train/validation ratios are 1.21, 1.20, 1.13, and 1.24 respectively, below the 1.5× overfitting review threshold.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 11.013 GiB
- `conversion_full_out.txt`: created
- `verification_full_out.txt`: created; `Data verification complete` with no errors or warnings.
- Full conversion runtime: 194.05 s (including 13.57 s serialization), substantially below the 5–7 minute conservative estimate.

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total classifier-good neurons | 69,943 | classifier QC | 69,453 | 69,453 | Matches supplied archive; 490 below paper-version aggregate, documented |
| Mean neurons/session | ~404.3 | classifier QC | 401.5 over 173 usable sessions | 401.5 | Yes for supplied data |
| Subjects | 28 | — | 28 | 28 | Yes |
| Sessions | 173 behavioral | process sessions | 174 NWBs; one has zero good units | 173 | Yes after documented unusable-session exclusion |
| Trials (total) | no paper-wide total | regular-trial analyses apply stricter exclusions | 94,990 raw | 89,068 | Yes after 3,766 special-water exclusions plus neural-validity/trailing-recording exclusions |
| Trials/session | not stated | varies by recording | raw mean 545.9 | retained mean 514.8 | Consistent with documented filtering |
| Time bins | reference 100-ms windows / 50-ms stride | `bw=.1`, `stride=.05` | absolute spikes | 80 × 50-ms bins | Required decoder-task override |
| Tone-time input | task sample normally near -1.85 s from go | sample event state | source events include restarts | continuous seconds from selected source tone | Yes |
| Photostimulation | ~25% in relevant VGAT cohort | interval/power fields | 18,588 raw stim trials | binary center-sampled intervals on retained trials | Yes |
| Choice values | left/right/no response concepts | trial type + correctness | instruction/outcome | left/right/no lick `[0,1,2]` | Yes |
| Outcome values | correct/error/no response | correctness `1/0/-1` | hit/miss/ignore | ignore/miss/hit `[0,1,2]` | Yes |
| Early lick | excluded only in regular-trial paper analyses | explicit flag | no early/early | no/yes `[0,1]` | Required retention |
| Tongue values | video marker position | aligned marker coordinates | y + likelihood at ~294 Hz | visible-session percentile classes + not-visible | Required discretization |

### Spot Checks
- Conversion printed every processed session with retained/raw trials, neurons, visible fraction, thresholds, and runtime.
- One zero-good session was skipped exactly as expected; all 173 usable sessions passed internal dimensions/range checks.
- Sessions with truncated neural recordings retained only validity-covered trial prefixes; no missing recording period was represented as zero firing.
- Official validator confirms 80 timepoints in every trial, valid ranges for all two inputs/four outputs, and complete brain-region indices.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output-log verification**: The first full validator found 127 all-zero-neural trials in three sessions. After correction and full reconversion, `verification_full_out.txt` reports no format warnings/errors and ends with `Data verification complete`.
2. **Independent raw-neural checks**: `/app/cache/critical_review_checks.py` loaded original NWBs directly with `pynwb` and used direct `np.histogram` calls. For first/middle/last neurons in 12 trials across four sessions, including late-start/truncated recordings, `np.allclose(raw_counts/0.05, converted_rates)` passed.
3. **Independent raw-input checks**: For the same trials, reconstructed absolute bin centers and source sample→delay→go event chains; continuous tone-time arrays passed `np.allclose`. Binary photostimulation arrays reconstructed from raw start/stop event intervals also passed.
4. **Independent raw-output checks**: Choice from instruction/outcome, raw outcome and early-lick labels, and nearest-frame tongue likelihood/y with raw session percentiles all passed `np.allclose` against converted outputs.
5. **Anatomy/metadata checks**: Raw classifier-good `anno_name` arrays exactly matched global `brain_regions` lookup through each session's indices. Subject/session mappings, time metadata, percentile thresholds, and source-trial indices passed.
6. **Global edge checks**: All 173 sessions have ≥2 trials; first/last trial matrices have exactly 80 bins; neural/input values are finite; totals are 89,068 trials and 69,453 session-neurons.
7. **Key statistics**: 28 animals and 173 usable sessions match the data paper. Native classifier-good total is 69,453 versus paper-version 69,943; no labels were altered to force agreement. Mean 401.46 neurons/session and range 90–923 match source labels.
8. **Output distributions**: Before the final two boundary removals, choice was approximately `[0.429,0.422,0.149]`, outcomes `[ignore .149, miss .166, hit .685]`, early lick `[no .883, yes .117]`, photostimulation ~20.0% of retained trials, and tongue `[low .062, middle .032, high .065, not visible .840]`. Removing two trials changes these only below displayed precision.
9. **Reference processing comparison**:
   - Loading: reference loads legacy exports; conversion uses required `pynwb.NWBHDF5IO` on DANDI NWBs.
   - Neuron filtering: both use published classifier-good labels; conversion additionally honors unit trial-validity alignment.
   - Trial filtering: reference regular mask removes early/no-response/stim trials; conversion deliberately retains these required decoder classes and removes only assisted-water/invalid-recording trials.
   - Alignment: both use go cue = 0; NWB go events map exactly one-to-one with trial rows.
   - Binning: reference `sliding_histogram` uses 100-ms windows/50-ms stride; conversion uses mandated non-overlapping 50-ms bins.
   - Inputs: source event clocks construct continuous time since actual tone state and photostimulation interval state.
   - Outputs: source instruction/outcome/early flags and aligned tongue marker data construct requested categories.
10. **Off-by-one checks**: Bin edges are exactly 81 values on `[-2.5,1.5]`, yielding 80 half-open bins and centers `[-2.475,1.475]`. Direct first/last trial and first/middle/last neuron histograms pass.

### Issues Found and Resolved
- **Short validity-vector assumption**: Eight sessions store 3,401 classifier-good unit masks shorter than the behavior table. Initial right-padding assumed all recordings started at trial 0. Raw `obs_intervals` proved one session started at source trial 125. Tested every short-mask unit: observation-overlapping go-cue trial count exactly matched stored mask length with zero failures. Fixed by mapping flags to observation-covered source trials.
- **All-zero neural trials**: Initial validator warned on 127 trials, primarily the misassigned pre-recording trials and two recording boundaries. Correct validity mapping removed 125; an explicit all-neuron-zero safeguard removed the two boundary trials. Corrected validator has zero such warnings.
- **Repeated tone states**: Early-lick/retry trials can contain multiple sample states. Nearest nominal onset differed from the final task-state chain in 183 trials, including valid variable-delay trials. Fixed by choosing the sample state whose stop is nearest the final delay start, where delay stop is nearest go.
- **Critical-review iteration reruns**: After fixes, sample conversion, sample verify-only, sample 200-epoch training, full conversion, full verify-only, and all raw-source checks were rerun. Sample validation accuracies remained above chance (choice 0.6151, outcome 0.6413, early lick 0.7309, tongue 0.5194).

### Warning Resolution
- No warnings remain in corrected full verification.
- High tongue not-visible frequency is a meaningful category, not missing data: low-likelihood placeholder coordinates are explicitly encoded as class 3.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes, from 21.4482 at epoch 1 to 0.6725 at epoch 200.
- Held-out test loss: 0.6923.
- Training used CUDA, 71,189 training trials and 17,879 held-out trials, with balanced class weights.
- `/app/train_decoder_full_out.txt` records complete execution through `train_decoder.py finished successfully`.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-----------------------|-------------------------|-------|
| Lick direction choice | 0.7108 | 0.6803 | Chance 0.3333; 2.04× chance validation |
| Outcome | 0.6953 | 0.6569 | Chance 0.3333; 1.97× chance validation |
| Early lick | 0.7912 | 0.7495 | Chance 0.5000; 1.499× chance validation |
| Tongue y-position | 0.6757 | 0.6194 | Chance 0.2500; 2.48× chance validation |

All outputs are above chance. Train/validation ratios are approximately 1.045, 1.058, 1.056, and 1.091; none approaches the 1.5× overfitting threshold.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Validation Balanced Accuracy | Chance | Multiple of chance | Expectation from papers |
|----------|------------------------------|--------|--------------------|-------------------------|
| Lick direction choice | 0.6803 | 0.3333 | 2.04× | Data paper reports time-resolved binary population choice decoding, using correct left/right trials, 200-ms causal windows, and 10-ms steps. It is not numerically equivalent to this three-class, all-trial decoder. Strong above-chance performance is consistent. |
| Outcome | 0.6569 | 0.3333 | 1.97× | Papers analyze reward/outcome selectivity but do not report a matching three-class decoder accuracy. Strong above-chance performance is consistent. |
| Early lick | 0.7495 | 0.5000 | 1.499× | Papers generally exclude early-lick trials from regular-trial analyses and do not report an early-lick decoder. Result is effectively the requested 1.5× review threshold. |
| Tongue y-position | 0.6194 | 0.2500 | 2.48× | Method paper reports marker/video prediction via ROC-AUC or firing-rate explained variance, not this four-class tongue decoder. High above-chance performance is consistent with movement encoding. |

### Checks Performed
1. **Accuracy versus chance**: Every output is above chance; choice/outcome/tongue substantially exceed 1.5× chance. Early lick is 1.499× due to rounding and was investigated rather than dismissed.
2. **Early-lick investigation**:
   - Raw NWB labels were checked with `np.allclose()` on specific trials and exactly match converted repeated labels.
   - Class distribution is 88.35% no / 11.65% yes, providing 10,377 positive trials—not a 99% dominant-class artifact.
   - Neural alignment is the same validated go cue and bins used by the other high-performing outputs.
   - Class-balanced training was enabled, so majority-class accuracy cannot explain the balanced score.
   - Training 0.7912 versus validation 0.7495 shows only a 0.0417 absolute gap and no problematic overfitting.
3. **Accuracy comparison to papers**: Exhaustive text search found population choice accuracy curves/figure descriptions, single-neuron choice/movement ROC-AUC, video behavioral ROC-AUC, and firing-rate R². No paper reports a single balanced-accuracy number under the same trial set, classes, temporal window, binning, architecture, and cross-validation split. Comparisons are therefore qualitative and task-matched where possible, not falsely equated.
4. **Train versus validation gap**:
   - Choice ratio 1.045, outcome 1.058, early lick 1.056, tongue 1.091.
   - All are far below the 1.5× concern threshold; held-out loss (0.6923) is close to final training loss (0.6725).
5. **Temporal/output synchronization**: Processing plots and independent raw checks show neural, event inputs, and nearest-frame tongue categories on the same absolute clock around go.
6. **Output variation**: Choice `[42.9%,42.2%,14.9%]`, outcome `[14.9%,16.6%,68.5%]`, early `[88.3%,11.7%]`, and time-resolved tongue `[6.2%,3.2%,6.5%,84.0%]`; no output is 99% one class.
7. **Filtering/processing match**: Rechecked classifier QC, source trial validity, event-chain tone alignment, 50-ms bins, and task-required trial-class retention against Step 10's reference comparison.

### Issues Found and Resolved
- No new conversion issue was found from accuracy review.
- Early-lick validation is at 1.499× chance rather than numerically above 1.5×, but all specified debugging checks passed, positive-trial count is large, raw labels are exact, and generalization gap is small. Changing labels/filtering to inflate accuracy would reduce fidelity to the source data.
- Paper metrics cannot be placed in a numeric same-task column without misrepresentation because they use different outputs and methods. This limitation is documented rather than using architecture differences as an unsupported excuse.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created with usage, schema, processing summary, statistics, and decoder results
- [x] `cache/` folder created
- [x] `cache/README_CACHE.md` documents retained investigation files
- [x] Critical-review script and final successful output retained in `cache/`
- [x] All required conversion, validation, and training logs exist
- [x] All workflow steps and implementation iterations documented
- [x] Required final artifact `/app/converted_data.pkl` exists and passes corrected full validation

Final deliverables are reproducible with `/app/convert_data.py`; all source NWB access uses `pynwb`.

