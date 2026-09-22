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

Environment verification: Python 3.13.15; NumPy 2.4.4; PyTorch 2.6.0+cu124. The required notes file was verified with `ls -la /app/CONVERSION_NOTES.md` before Step 1.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `process_one_sess` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING/CURATION | Groups probes into sessions; reads behavior/task, spike times, QC classifications, and histology; intersects ephys with histology and classifier-QC unit IDs. |
| `sliding_histogram` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Counts spikes in half-open sliding windows and divides by bin width to obtain Hz. Reference analysis uses 40-ms width, 3.4-ms stride, -3 to +3 s. |
| `process_one_area` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Truncates already-go-cue-aligned spike times, computes firing rates, and attaches unit/CCF/trial metadata. |
| `helper_get_neuron_id_area` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | CURATION | Intersects classifier-QC units with histology and hemisphere (CCF ML midpoint 5700 um), verifying CCF annotations. |
| `load_session` | `VideoAnalysisUtils/population_decoding_utils.py` | LOADING | Concatenates region files into one session-level time × trial × neuron array and carries behavioral labels. |
| `get_regular_trial_mask` | `VideoAnalysisUtils/population_decoding_utils.py` | CURATION | Reference analysis subset excludes early lick, auto/free water, no-response, and photostimulation trials. |
| `check_fr` | `VideoAnalysisUtils/preprocessing_utils.py` | CURATION | Removes a neuron if its across-trial firing-rate variance is zero at any bin (analysis utility, not invoked by the main preprocessing script). |

### Notes
- Repository corresponds to the 2025 movement-encoding paper and preprocesses DataJoint MATLAB exports, not NWB directly. Its loader therefore cannot be reused for this task; equivalent source fields must be accessed through `pynwb`.
- Data are electrophysiology spike times, so delta-F/F is not applicable.
- The main preprocessing selects the provided classifier-QC good-unit lists, requires matching histology, retains 14 broad anatomical groups, and preserves left/right region identity. QC is applied upstream through good-unit indices rather than re-thresholding individual QC metrics in this code.
- Raw/exported spike times are documented as already relative to go cue. Absolute lick and laser times are explicitly shifted by each trial's go-cue time.
- Behavioral conventions in this code: correctness 1=correct/free-water, 0=error, -1=no response; trial type is transformed to 1=left/0=right. Stimulation fields are power, type, onset, offset, with onset/offset aligned to go cue.
- The reference's `regular trial` mask is appropriate for its movement/video analyses but cannot be copied wholesale here: the requested decoder explicitly requires early-lick, miss/no-lick, and photostimulation labels, so excluding those trials would destroy required output/input classes. Auto/free-water handling remains to be resolved against NWB fields and papers.
- Target binning is mandated to non-overlapping 50-ms bins, superseding the reference paper code's overlapping 40-ms/3.4-ms firing-rate representation.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data` is DANDI:000363 v0.230822.0128 (Mesoscale Activity Map Dataset): one `dandiset.yaml` plus 174 NWB 2.x files under 28 `sub-<id>/` directories. Each file is one behavior+ecephys session (some also have optogenetics in the filename). All content inspection used `pynwb.NWBHDF5IO(..., load_namespaces=True)`; `h5py` was not used.

NWB contents are consistent across all sessions:
- `trials` columns: start/stop, trial IDs, `photostim_onset/power/duration`, task/protocol, `trial_instruction` (left/right), `early_lick` (early/no early), `outcome` (ignore/miss/hit), auto-water, free-water.
- `units`: ragged `spike_times`, `classification` (good/unlabelled/NaN), fine Allen CCF `anno_name`, electrodes, waveform and QC metrics, and a per-unit length-`n_trials` Boolean `is_good_trials` vector.
- `acquisition/BehavioralEvents`: sample (auditory tone), delay, go, trial-end, left/right lick, and photostimulation start/stop timestamps.
- `acquisition/BehavioralTimeSeries`: side-view jaw, nose, and tongue `(x, y, likelihood)` at explicit timestamps. Tongue series exists in every file.
- Example dtypes: event/tracking timestamps and positions float64, unit spikes ragged float arrays, trial IDs/int flags int32, categorical trial columns strings.
- Trial categories across native files: outcome hit 65,254, miss 15,641, ignore 14,095; instruction right 48,913, left 46,077; early 10,805, no-early 84,185.
- Unit classifier categories: 69,453 good, 200,922 unlabelled, 1,852 NaN. One session (`sub-440958_ses-20190216T162508...`) has zero classifier-good units and cannot form a neural session after curation; the other 173 do.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 272,227 raw; 69,453 classifier-good |
| Neurons / session | raw 493–3,191; good 0–923 |
| Subjects | 28 |
| Sessions / subject | 3–10 (174 total) |
| Trials (total) | 94,990 |
| Trials / session | 264–800 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 69,943 good units | Data paper STAR Methods / QC white paper: overall dataset after region-specific classifier QC |
| Neurons / session | median 393 | Data paper Fig. 1 text |
| Subjects | 28 mice | Data paper Results |
| Sessions / subject | 173 behavioral sessions total | Data paper Fig. 1 / STAR Methods |
| Trials (total) | Not stated | — |
| Trials / session | mean 476, range 130–785 | Data paper STAR Methods |
| Neural data time bin | 40-ms width, 3.4-ms stride (movement paper); choice population decoder used causal 200-ms windows/10-ms step | Movement paper Methods; data paper Methods |
| Behavior data time bin | 300-Hz side-view video (~3.33 ms) | Both papers Methods |
| Reward rate | 84% correct control trials, range 65–99% | Data paper STAR Methods |
| Photostimulation prevalence | ~25% randomly interleaved subset; late final 0.5 s of delay | Data paper STAR Methods |
| Task timing | 0.65-s tone/sample, 1.2-s delay, 0.1-s go cue, 1.5-s answer | Papers/methods excerpt |
| Major-area good units | ALM 8,717; striatum 7,664; thalamus 12,808; midbrain 7,495; medulla 2,928 | Data paper / QC white paper |


### Processing Details
- Task is a two-alternative auditory delayed response: three 150-ms 3/12-kHz tones with 100-ms gaps form the 0.65-s sample, followed by 1.2-s delay; a 0.1-s go cue begins the 1.5-s answer period. Thus sample/tone onset is nominally -1.85 s relative to go.
- The data paper excluded early-lick and no-response trials for its analyses and selected sessions with >65% control performance plus >=50 correct left and right trials. This conversion must retain those trial types because they are explicitly requested decoder targets.
- Spikes were Kilosort2 sorted. Fifteen jointly informative QC metrics fed five region-specific logistic-regression classifiers trained on blinded manual labels; individual metric thresholding is explicitly discouraged because distributions overlap.
- Movement paper bins spikes with 40-ms width/3.4-ms stride, but the decoder task mandates 50-ms width; this is a justified task override.
- Video/keypoints are synchronized at 300 Hz. The movement paper used side view; it identified >5-SD frame-to-frame velocity outliers and imputed them, and set occluded tongue positions to their mean for its regression. The present categorical target instead explicitly reserves class 3 for not visible, so occlusion must remain distinguishable rather than imputed.
- Reported decoding is not directly comparable to this neural-to-behavior architecture. Data-paper choice decoding used logistic regression, 200-neuron pseudo-populations, 200-ms/10-ms causal bins and nested 5-fold CV; plotted accuracy rose from chance (0.5) during sample/delay toward ~0.8–0.9 near go depending on region. Movement-paper video-to-choice AUC approached 0.99 after go.

### Curation Steps

**Neuron curation rules**:
Retain NWB units whose `classification == "good"`, the supplied output of the 15-metric region classifier. Do not substitute independent thresholds. Respect each retained unit's `is_good_trials` valid-trial mask when deciding session/trial usability.

**Trial curation rules**:
Reference analyses exclude early-lick/no-response and often photostimulation/auto/free-water trials, but required decoder labels demand retaining early/ignore/stimulation trials. Sessions in the released set were behaviorally curated upstream. Trials invalid for all retained neurons or lacking required go/sample/tracking support must be excluded; precise intersection policy is planned in Step 5.

### Decoders Trained
| Decoded variable | Accuracy |
| Choice (paper population decoder) | chance 0.5; approximately 0.8–0.9 near go for 200-neuron pseudo-populations, region-dependent |
| Choice from video (movement paper) | response epoch AUC ~0.99 ± 0.01 across 106 sessions |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Good-unit total | Classifier-QC lists; 14 broad regions | NWB has 69,453 `good` | Paper/white paper report 69,943 | Use release-native labels; 490-unit (0.70%) discrepancy is attributable to released-version/export differences and cannot be repaired without inventing labels. |
| Sessions | Code scans all exported sessions | 174 files; 173 with >=1 good unit | 173 behavioral sessions | Drop sole zero-good-unit session; exact match. |
| Neurons/session | Code concatenates all classifier-good regional units | median 390 among usable sessions (range 90–923) | median 393 | Consistent within release difference. |
| Trial counts | Reference analyses define task-specific masks | 94,990 native trials, 264–800/session | mean 476, range 130–785 after analysis/session criteria | Native release contains mean 546; retain required early/miss/stim classes and use validity masks, explaining broader/higher counts. |
| Trial filtering | Movement code excludes early, auto/free-water, ignore, stimulation | All requested categories exist in NWB | Data paper generally excludes early/no-response | Decoder specification overrides those exclusions; preserve classes, but record auto/free-water and unit-validity policy. |
| Unit trial validity | MATLAB reference export does not explicitly consume NWB `is_good_trials` | 68,888/69,453 good units valid on all trials; 565 have partial masks (minimum 69.6% valid) | Recording drift/QC motivates invalid periods | Retain otherwise-good units but exclude any trial not valid for every retained unit. This honors explicit valid-period metadata and avoids fabricating neural values; all sessions retain >=2 trials. |
| Tone/sample event | Code aligns spikes and laser/licks to go | 1–16 sample-start events per trial because early licking can trigger replay | Early licking triggers replay | Use the last sample onset before go as onset of the completed instruction epoch; preserves nominal relationship to go and avoids using abandoned/replayed epochs. |
| Binning | 40-ms width, 3.4-ms stride in movement analysis | Raw spike timestamps available | Same; choice analysis uses other bins | Required non-overlapping 50-ms bins supersede paper analysis bins. |

Final understanding: use all 173 sessions containing classifier-good units; align raw spikes and timestamped behavior directly to each NWB go onset; use 80 half-open 50-ms bins spanning [-2.5, 1.5) s; keep requested trial classes; represent fine Allen `anno_name` labels; derive choice consistently from instruction+outcome and verify against lick timestamps; discretize only visible tongue frames using session-wide visible-y percentiles. No calcium imaging is present.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `units.spike_times` for `classification == "good"` | `neural` | For each go-aligned trial, histogram absolute spikes into 80 half-open 50-ms bins from -2.5 to +1.5 s; divide counts by 0.05 s to Hz; float32 `(neurons,80)` | `sliding_histogram`, `process_one_area` | Same spike-count/rate principle; task-mandated bins. |
| Bin centers and final pre-go `sample_start_times` event | `input[0]` | Continuous seconds since completed tone/sample onset: `(go + bin_center) - tone_onset` | NWB event alignment; task Methods | Time-varying float32. |
| `photostim_start_times/stop_times` | `input[1]` | Binary 1 where a 50-ms bin overlaps laser-on interval, otherwise 0 | Reference shifts laser events by go cue | Time-varying float32. |
| `trial_instruction` + `outcome` | `output[0]` choice | hit -> instructed side; miss -> opposite side; ignore -> no lick; confirm against first response-period lick events | Data-paper four trial groups `<stimulus,choice>` | Values 0 left, 1 right, 2 no lick; broadcast over 80 bins. |
| `trials.outcome` | `output[1]` outcome | Map ignore/miss/hit to 0/1/2 | `correctness` conventions in preprocessing | Broadcast over 80 bins. |
| `trials.early_lick` | `output[2]` early lick | Map no early/early to 0/1 | `early_lick_trials` | Broadcast over 80 bins. |
| `Camera0_side_TongueTracking[:,1:3]` | `output[3]` tongue y-position | Nearest video sample at each bin center; likelihood <0.9 -> 3 not visible; among session-wide visible samples, y below 40th percentile ->0, 40th–60th ->1, above 60th ->2 | Movement paper side-view DeepLabCut markers | DLC likelihood is strongly bimodal near 0/1; 0.9 is a conservative standard visibility threshold. Percentiles computed before trial selection over all visible session frames, as requested. |
| `subject.subject_id` | `subjects`, `subject_idx` | Stable sorted subject list; session lookup indices | NWB metadata | 28 subjects expected. |
| good-unit `anno_name` | `brain_regions`, `brain_region_idx` | Global stable sorted unique Allen CCF annotation strings and per-session indices | Reference CCF labeling | Fine labels preserve maximum anatomical information. |

### Key Decisions
1. **Session inclusion**: Process all 174 NWBs, then omit the single session with zero classifier-good neurons; expected 173 sessions, matching the paper.
2. **Neuron curation**: Use supplied multimetric classifier label exactly (`classification == good`); no individual QC thresholds because the QC white paper explicitly rejects that strategy.
3. **Trial validity**: Keep trials for which every retained unit's `is_good_trials` is true. This excludes invalid recording periods while preserving a constant neuron set. Keep early, ignore, miss, stimulation, auto-water and free-water trials because requested targets/inputs require broad behavioral coverage; record auto/free-water prevalence in metadata.
4. **Alignment and endpoints**: edges are `np.arange(-2.5,1.5+0.05,0.05)` and bins are `[left,right)`, yielding exactly 80 centers from -2.475 to +1.475 s. This removes endpoint ambiguity.
5. **Tone onset under replay**: Early licking can produce multiple sample events. Select the last sample start at or before go within the trial—the completed instruction epoch that causally precedes that go cue.
6. **Choice derivation**: Trial table contains instruction/outcome, not explicit choice. In a two-port task, outcome deterministically maps instruction to actual choice (same side for hit, opposite for miss, none for ignore); raw response lick events provide an independent check.
7. **Outputs as time-varying matrices**: Broadcast trial-level choice/outcome/early labels across time and retain tongue as truly time-varying, producing uniform integer `(4,80)` arrays compatible with joint decoder training.
8. **Tongue visibility/discretization**: Use DLC likelihood >=0.9. Compute q40/q60 only from visible session frames (otherwise hidden placeholder coordinates would dominate), with exact boundary convention requested: class 0 `<q40`, class 1 `q40<=y<=q60`, class 2 `>q60`, class 3 hidden.
9. **Storage**: float32 firing rates/inputs and int8 outputs minimize a dataset expected to occupy several GB while retaining exact 20-Hz firing-rate increments.

### Planned Sanity Checks
- [ ] Format/shapes/dtypes: 173 sessions; every included session >=2 trials and neurons; all trials `(n,80)`, inputs `(2,80)`, outputs `(4,80)`; finite values and categorical bounds.
- [ ] Direct neural spot check: independently histogram a selected raw NWB unit/trial with `np.histogram` and require `np.allclose(converted, counts/0.05)`.
- [ ] Direct input spot check: independently reconstruct tone-time and photostimulation overlap from NWB event timestamps and require `np.allclose`.
- [ ] Direct output spot check: independently reconstruct three trial labels and tongue classes from NWB trial/tracking data and require `np.allclose`.
- [ ] Choice/lick audit: compare instruction+outcome-derived choice with first left/right lick during `[go, go+1.5)`; investigate mismatches (early/replayed/late responses).
- [ ] Distribution checks: compare 28 subjects, 173 sessions, good units near 69,943 paper value (native 69,453 before trial-period handling), median ~393/session, and paper-style control/no-early hit rate near 84%.
- [ ] Timing checks: bin centers exactly span requested window; normal completed sample onset clusters at -1.85 s; laser input agrees with late-delay timing and is off after go in standard photoinhibition trials.
- [ ] Tongue checks: visible class distribution is approximately 40/20/40 by construction over session frames; class 3 dominates pre-go and visible tongue increases after go.

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]
- Created `/app/convert_data.py`; syntax compilation and CLI help succeeded.
- NWB access is exclusively `pynwb.NWBHDF5IO`; no `h5py` import or direct HDF5 access.
- Script supports default/`--full`, `--sample`, and `--show-processing`, with session timing/ETA and validation exceptions.
- Processing plots cover firing-rate binning/alignment, both inputs, all outputs, tongue thresholds, and DLC visibility likelihood.

Code inefficiencies identified:
Naively looping over every unit × trial would require ~35 million Python-level histograms. Retaining all converted sessions in the required nested-list pickle also creates an unavoidable multi-GB in-memory result.

Code speedups added:
For each unit, a single vectorized `np.searchsorted` call evaluates all trial-bin edges, then `np.diff` yields counts. Laser overlap and tracking lookup are vectorized over trials/time. Arrays are float32/int8, and tracking/source objects are scoped to one NWB at a time.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 834 session-neurons |
| Neurons / session | 459, 375 |
| Subjects | 1 |
| Sessions / subject | 2 |
| Trials (total) | 527 |
| Trials / session | 368, 159 |
| Tone-time input range (s) | [-0.6, 5.7] |
| Photostimulation range | [0, 1] |
| Choice distribution | [0.452, 0.408, 0.140] |
| Outcome distribution | [0.140, 0.298, 0.562] |
| Early-lick distribution | [0.943, 0.057] |
| Tongue distribution | [0.053, 0.029, 0.062, 0.857] |

### Processing Plots Review
Both plots show the go line at 0, plausible event-related firing structure, linear tone-time input, correctly bounded stimulation, bimodal DLC likelihood with a clear 0.9 threshold, and tongue percentile boundaries spanning visible samples. Trial-level outputs are constant and tongue is time-varying when visible. No alignment artifact was seen.

Initial run exposed two raw-data edge cases and was rerun after fixes: (1) `is_good_trials` is insertion-local and indexed by per-unit `obs_intervals`, so it was expanded by exact trial start/stop matching; (2) one nominally observed terminal trial occurred after all units' final spikes and produced an all-zero population, so it is now excluded as an invalid recording period. The final verifier reports no errors or warnings. Choice reconstructed from instruction+outcome matched the first response-window lick for all 527 sample trials.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Vectorized edge lookup rather than unit×trial histograms | Sample conversion processing is ~1.25 s/session |

| Step | Time / Session | Estimated Total Time |
| NWB load + convert + plots | ~1.25 s/session | ~3.6 min conservatively for 174 sessions with size variation |
| Serialization | sample 74 MB in seconds | Full output expected roughly 10–15 GB and several additional minutes |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None (after excluding one terminal trial with no recorded spikes)

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Lick direction choice | 0.7284 | 0.6189 |
| Outcome | 0.7502 | 0.6487 |
| Early lick | 0.8358 | 0.7255 |
| Tongue y-position | 0.6715 | 0.5214 |

Loss decreased from 12.655 at epoch 1 to 0.586 at epoch 200 (test loss 0.722). Every validation balanced accuracy exceeds uniform chance (0.333, 0.333, 0.5, 0.25 respectively), supporting the alignment and label mappings.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 11.793 GB
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total neurons | 69,943 | classifier-QC | 69,453 | 69,453 | Native release match; -0.70% vs paper |
| Mean neurons/session | median 393 | classifier-QC session concatenation | median 390 | mean 401.46, median 390 | Yes |
| Subjects | 28 | all dataset | 28 | 28 | Yes |
| Sessions | 173 | all classifier-QC sessions | 174 files/173 nonempty | 173 | Yes |
| Trials (total) | not stated | analysis-dependent masks | 94,990 native | 90,378 valid recorded | Explained by insertion intervals and terminal recording gaps |
| Trials/session (mean) | mean 476, range 130–785 | movement analyses select usable video/trials | native mean 546, 264–800 | mean 522.4, 159–800 | Consistent given required class retention; one 800 vs paper 785 release difference |
| Tone-time range (s) | task nominally tone -1.85 from go | event timestamps | replay/variable protocols | [-1.5, 11.9] at bin centers | Yes; long positive times are early-lick replay delays |
| Photostimulation range | binary/off-on; ~25% trials in subset | timestamp intervals | timestamps | [0,1] | Yes |
| Choice distribution | not global | derived stimulus-choice convention | raw outcomes/instructions | [0.428,0.422,0.149] | Plausible/balanced sides |
| Outcome distribution | control accuracy ~84% under exclusions | correctness fields | all-native [ignore .148, miss .165, hit .687] | [.149,.166,.685] | Yes |
| Early distribution | excluded in many paper analyses | early flag | all-native [.886,.114] | [.884,.116] | Yes |
| Tongue distribution | occluded before response | side-view DLC | likelihood bimodal | [.062,.032,.065,.841] | Yes; visible subclasses close 40/20/40 conditional proportions |

Full conversion took 3.19 min. Audit: 94,370 trials in 173 neural sessions; 1,569 rejected by insertion-local good-trial masks; 2,423 all-population-zero trials rejected as recording gaps; 90,378 retained. Full verifier: valid with no errors or warnings.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` says valid with no errors or warnings. All 173 sessions have >=159 trials; no all-zero trial remains.
2. **Independent raw-data sanity tests**: `/app/cache/sanity_checks.py` loads the original NWB with `pynwb` without importing conversion code. For session 0/trial 5/neuron 3, raw `np.histogram` rates equal the pickle by `np.allclose`. Independently reconstructed tone-time, laser overlap, choice/outcome/early labels, and nearest-frame tongue category also pass `np.allclose`. First/last centers are exactly -2.475/+1.475 and 80-bin endpoint checks pass. Output is in `/app/cache/sanity_checks_out.txt`.
3. **Data loading comparison**: Reference code loads DataJoint MATLAB exports; converter uses the equivalent NWB objects through mandatory `pynwb` (`units`, trials, acquisitions). Both concatenate good units at session scope. Difference is source API only.
4. **Neuron/trial filtering comparison**: Both use the provided 15-metric region-classifier good labels rather than ad hoc thresholds. Reference movement analyses remove early/ignore/stimulation trials; converter deliberately retains them because they are requested variables. NWB insertion-local `is_good_trials` is expanded using `obs_intervals`; trials invalid for any retained unit and population-zero acquisition gaps are excluded.
5. **Temporal alignment comparison**: Reference spikes are go-relative and explicitly shifts lick/laser times by go. Converter subtracts the absolute NWB go timestamp from every stream, which is equivalent. Early-lick replay uses the last completed sample onset before go.
6. **Binning comparison**: Both count spikes in half-open windows and divide by width. Reference movement processing uses overlapping 40-ms/3.4-ms bins; required decoder bins are non-overlapping 50-ms, exactly 80 over [-2.5,+1.5).
7. **Input construction comparison**: Tone and laser use NWB event timestamps. Continuous elapsed tone time follows the explicit decoder task; binary laser is marked by positive-duration bin overlap.
8. **Output construction comparison**: Reference defines `<stimulus,choice>` trial groups; in this two-port task instruction+outcome uniquely yields choice. Outcome and early lick use NWB categorical columns. Tongue uses the paper's side-view DeepLabCut stream, but preserves occlusion as required instead of imputing it as the movement regression did.
9. **Key statistics**: 28 subjects and 173 sessions exactly match paper. 69,453 release-native good units differ from 69,943 paper count by 490 (0.70%); median neurons/session 390 versus paper 393. Converted control/no-early response accuracy is 81.67% (43,680 hit/9,801 miss), close to paper mean 84%; including ignore in the denominator would incorrectly give 68.65%. Retained outcomes and early-lick fractions closely match native release values.
10. **Edge cases/off-by-one audit**: Fixed insertion-local validity vectors; excluded 2,423 trials with no recorded population spikes despite nominal intervals; dropped the sole zero-good-unit session; verified no remaining all-zero neural trials. Choice matched first response-window lick on 90,096/90,378 trials (99.69%). The 282 mismatches are expected event ambiguity (multiple/carry-over lick events near response boundaries); authoritative task outcome+instruction is retained because it defines scored behavioral choice.

### Issues Found and Resolved
- `is_good_trials` initially assumed session length: raw inspection showed it is indexed by unit `obs_intervals`; expanded to the full trial table by exact start/stop matching and reran sample/full validation.
- One sample terminal trial and analogous full-data gaps had no spikes from any unit: added a population-zero acquisition-period exclusion; final validation has no warning and zero all-zero trials.
- Paper/native unit count differs by 0.70%: preserved release-native classifier labels rather than fabricating 490 labels; session count and median neurons remain consistent.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes; 21.587 (epoch 1) -> 0.677 (epoch 200), test loss 0.673. GPU training completed without fallback.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Lick direction choice | 0.7181 | 0.6876 | chance 0.3333 |
| Outcome | 0.6964 | 0.6632 | chance 0.3333 |
| Early lick | 0.7879 | 0.7527 | chance 0.5000 |
| Tongue y-position | 0.6793 | 0.6159 | chance 0.2500 |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Papers |
| Lick direction choice | 0.6876 validation balanced accuracy (2.06× chance) | Data-paper neural choice pseudo-populations are region/time dependent, ~0.8–0.9 near go for 200 neurons; not the same whole-window/session architecture |
| Outcome | 0.6632 (1.99× chance) | Paper reports outcome selectivity, not directly comparable decoder accuracy |
| Early lick | 0.7527 (1.51× chance) | Not decoded in papers; early trials were excluded in their primary analyses |
| Tongue y-position | 0.6159 (2.46× chance) | Movement paper predicts neural activity from video (reverse direction), not categorical tongue from neural activity |

**Accuracy versus chance**: all validation balanced accuracies exceed chance; all meet or exceed the requested 1.5×-chance investigation threshold (early lick is 1.505× and was reviewed closely). Class distributions contain ample samples: choice [38,708, 38,163, 13,507], outcome [13,507, 15,005, 61,866], early [79,915, 10,463]. The raw-output spot check in Step 10 verifies early labels directly, and no conversion change is justified.

**Paper accuracy survey**: the data paper's population-choice decoder uses 200-neuron cross-session pseudo-populations, causal 200-ms windows and decorrelated trial weighting; its figure shows accuracy rising above 0.5 during sample/delay toward roughly 0.8–0.9 near go. The movement paper reports video-to-choice performance near 0.99 ± 0.01 after go and response-epoch single-marker/embedding performance of 0.88 ± 0.01/0.96 ± 0.00 across 106 sessions. These reverse-direction or materially different decoders are contextual, not target-for-target baselines. Our 0.688 choice score averages all 80 bins from -2.5 to +1.5 s—including pre-instruction/chance-like bins—and simultaneously fits heterogeneous session projections, so it is consistent with a near-go paper curve being higher.

**Train/validation gaps**: train/validation ratios are choice 1.044, outcome 1.050, early 1.047, tongue 1.103—none approach the >1.5 concern threshold. Test loss (0.673) is nearly identical to final training loss (0.677); there is no evidence of overfitting or leakage.

**Low-accuracy debugging review**: no output is below chance or below 1.5× chance. Nevertheless, Step 10 independently verified raw labels and timing for a concrete trial, plots show synchronized go/neural/output streams, every class has variation, and classifier-QC filtering exactly follows the paper. No conversion issue or accuracy-motivated change was found.

[Analysis of any low accuracies]

### Issues Found and Resolved
- No unresolved issue. Early-lick accuracy was closest to the 1.5× threshold; raw-label checks, class support, small generalization gap, and 0.7527 validation accuracy support retaining the specified representation.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created with investigation script/output and `README_CACHE.md`
- [x] All required files audited; complete conversion/verification/training logs exist
