# Dataset Conversion Notes

## Overview
- **Dataset**: Brain-wide neural activity underlying memory-guided movement (provided NWB data)
- **Date started**: 2026-09-19
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
- `train_decoder.py`

Environment check: Python 3.13.15, NumPy 2.4.4, and PyTorch 2.6.0+cu124 import successfully.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `loadmat` / `loadh5mat` | `code/VideoAnalysisUtils/preprocessing_utils.py` | LOADING | Load MATLAB v5/v7.3 exports and recursively convert MATLAB structs/cells; decode cluster notes. |
| `process_one_sess` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING / CURATION | Load all probes in a session, combine ephys/histology, map task/behavior fields, and apply the classifier-derived good-unit indices. |
| `helper_get_neuron_id_area` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | CURATION | Intersect good-unit indices with units having CCF annotations, split hemispheres at ML=5700, and validate QC annotation identity. |
| `sliding_histogram` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Count spikes in half-open `[center-width/2, center+width/2)` bins and divide by bin width to produce Hz. |
| `process_one_area` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Truncate already go-cue-aligned spike times, compute firing rates, and save time × trial × neuron area files. |
| `load_session` | `code/VideoAnalysisUtils/population_decoding_utils.py` | LOADING | Concatenate area files along neuron dimension and retain behavior/task variables and CCF metadata. |
| `get_regular_trial_mask` | `code/VideoAnalysisUtils/population_decoding_utils.py` and `functions_for_r2.py` | CURATION | Define paper-analysis “regular” trials: no early lick, auto/free water, no-response, or stimulation. |
| `align_markers_between_lims` | `code/Sherlock/align_markers.py` | PROCESSING | Align video markers to go cue on a 3.4-ms grid over `[-3, 1.5)`, using the last video sample in each interval. |
| `temporal_alignment_embed_and_ephys` | `code/VideoAnalysisUtils/functions_for_r2.py` | PROCESSING | Crop video/ephys streams to their common time support using nearest indices. |
| `create_4fold_trial_type_mask` | `code/VideoAnalysisUtils/functions_for_r2.py` | CURATION | Stratify CV by requested lick side and correctness, pooling sparse incorrect groups when necessary. |

### Notes
- The repository README identifies the supplied code as the analysis for Wang et al., with raw MAP data from DANDI 000363. The active preprocessing is in `VideoAnalysisUtils/` and `Sherlock/`; `Archive/` is explicitly old/potentially broken.
- This is extracellular electrophysiology, so delta-F/F is not applicable. The native neural variable is per-unit spike times.
- The paper pipeline uses `qc_mode='classifier'`, reading externally generated `goodunits` indices, not ad-hoc thresholds in the analysis script. Only units that also match histology/CCF records are retained, and only 14 listed gross regions are processed.
- Raw spike times are documented as already relative to go cue. Lick and stimulation times in the MATLAB exports are converted to go-cue-relative time by subtracting `task_cue_time`.
- Published preprocessing used a 40-ms spike-count window with 3.4-ms stride over approximately `[-3, 3]`; this conversion must instead use the decoder-specified non-overlapping 50-ms bins over `[-2.5, 1.5)` while preserving the reference half-open bin convention and Hz scaling.
- Behavioral mappings exposed by the code include early report, auto/free water, response correctness (`1` correct, `0` error, `-1` no response), lick directions/times, stimulation power/type/on/off, requested trial type, and sample/delay/go timing.
- Video marker order in the code is nose x/y, tongue x/y, jaw x/y, whisker x/y. Missing full marker sets caused sessions to be skipped in the video paper, but the decoder task only requires tongue y and allows a “not visible” class, so raw NWB availability/visibility must be examined directly.
- The reference “regular trial” filter is analysis-specific and removes exactly the trial categories that are required decoder labels here (early lick, no response, photostimulation). It therefore cannot be applied wholesale; only invalid/missing-data trial exclusions should be carried forward, with this required difference documented.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/dandiset.yaml` describes DANDI:000363, version `0.230822.0128` (Mesoscale Activity Map Dataset), 53.6 GB / 174 assets / 28 mice.
- There are 28 `sub-<mouse>/` directories and 174 HDF5-backed NWB files named `sub-<mouse>_ses-<timestamp>_behavior+ecephys[+ogen].nwb`; each file is one recording session. Subject identifiers are numeric strings encoded in the directory and NWB subject metadata.
- All sessions contain these relevant NWB groups:
  - `units/`: ragged global `spike_times` + `spike_times_index`, classifier label (`classification`: `good`/`unlabelled`), Kilosort label (`unit_quality`: `good`/`multi`), detailed Allen CCF `anno_name`, electrophysiology/QC metrics, electrode references, waveform summaries, and a `(unit, trial)` `is_good_trials` mask.
  - `intervals/trials/`: one row per trial, with global `start_time`/`stop_time`, instruction (`left`/`right`), outcome (`hit`/`miss`/`ignore`), early-lick label, auto/free-water flags, and photostimulation onset/duration/power (`N/A` on unstimulated trials).
  - `acquisition/BehavioralEvents/`: global timestamps for presample, sample, delay, go, trial-end, left/right licks, and photostimulation starts/stops. All 14 event series exist in every session. The event `data` values are amplitudes/powers rather than trial IDs; events must be assigned to trial intervals by timestamps.
  - `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/`: `(frames, 3)` float64 columns documented as tongue x, tongue y, and DeepLabCut likelihood, with global float64 timestamps. Camera0 tongue/jaw/nose tracking exists in all 174 sessions. Nominal inter-frame interval is 0.0034 s, with gaps between trial clips.
- Every session is the audio-delay task. The first sample/tone onset normally follows presample onset; some trials have repeated sample/delay event starts (e.g. after early reports), so source event-to-trial assignment and “first valid sample onset” require explicit handling.
- Tongue coordinates are always numeric even when tracking confidence is essentially zero, so visibility must be determined from the likelihood column rather than finite/NaN coordinates. In an inspected session, likelihood was strongly bimodal and `10.52%` of frames were at least 0.9.
- Raw spike timestamps and behavioral/video timestamps share the NWB session clock. Unlike the older MATLAB export consumed by the reference repository, NWB spike times are not stored per trial or pre-aligned; they must be sliced by `go_start_times`.
- `is_good_trials` is explicitly described in NWB as a manual per-probe/per-unit trial-validity annotation. Across all unit-trial entries only 207,797 / 147,085,032 (`0.1413%`) are false; its structure will be handled during curation planning.
- One session (`sub-440958_ses-20190216T162508`) has 1,852 units but missing (`NaN`) classifier labels and therefore zero classifier-good units. The other classifier labels total 69,453 good and 200,922 unlabelled.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 272,227 raw sorted units; 69,453 classifier-good units |
| Neurons / session | raw: 493–3,191, mean 1,564.52, median 1,571.5; classifier-good: 0–923, mean 399.16, median 390 |
| Subjects | 28 |
| Sessions / subject | 3–10 (174 sessions total) |
| Trials (total) | 94,990 |
| Trials / session | 264–800, mean 545.92, median 534 |

Additional source distributions: 65,254 hit / 15,641 miss / 14,095 ignore; 10,805 early / 84,185 no-early; 46,077 left / 48,913 right instructions. There are 4,030,078,240 raw sorted spikes indexed across all units (before unit/trial/window filtering). Nonempty CCF annotations occur for 71,305 unit rows (69,453 classifier-good plus 1,852 rows in the session with missing classifier labels), covering 294 unique nonempty annotation strings.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 69,943 classifier-good units | Data paper Methods / QC white paper: “69,943 good units recorded across 173 behavioral sessions.” |
| Neurons / session | median 393 | Data paper Results: each session yielded hundreds of neurons, “median = 393.” |
| Subjects | 28 | Data paper Results/Methods: “Mice (n = 28).” |
| Sessions / subject | not tabulated; 173 sessions over 28 mice | Data paper Fig. 1 legend/Methods. |
| Trials (total) | not reported as a total | — |
| Trials / session | mean 476, range 130–785 (paper-selected sessions/analysis definition) | Data paper Methods. |
| Neural data time bin | 40-ms width, 3.4-ms stride in newer movement paper; 200-ms causal width, 10-ms step for data-paper population choice decoder | Method paper Methods; data paper population decoder Methods. |
| Behavior data time bin | video acquired at 300 Hz; code uses 3.4 ms | Both papers; reference code. |
| Reward rate | 84% correct control trials, range 65–99%, excluding early licks | Data paper Methods. |
| Task timing | tone/sample 0.65 s, delay 1.2 s, go cue 0.1 s, answer period 1.5 s | Data paper Fig. 1/Methods. |
| Photostimulation | ~25% randomly interleaved trials in 17 VGAT mice; final 0.5 s of delay and ends before go | Data paper Methods. |
| Units by selected region | ALM 8,717; striatum 7,664; thalamus 12,808; midbrain 7,495; medulla 2,928 | Data paper and QC white paper. |
| QC retention | 25.9% of Kilosort2 clusters | Data paper and QC white paper. |


### Processing Details
- Trials are aligned to the auditory go cue. The canonical epochs relative to go are sample `[-1.85, -1.2)` s, delay `[-1.2, 0)` s, and response `[0, 1.5)` s. Early licking can trigger replay of the sample/delay sequence, explaining repeated raw event timestamps.
- The task uses 3-kHz and 12-kHz pure tones (three 150-ms tones with 100-ms inter-tone gaps) to instruct right and left licking, respectively. A 6-kHz amplitude-modulated 0.1-s go cue marks the end of the delay.
- The data paper’s PSTHs used 1-ms spike counts smoothed with a 200-ms boxcar. Its choice population decoder used causal 200-ms windows stepped by 10 ms, logistic regression, hierarchical bootstrapping, trial balancing, and nested fivefold CV. The newer movement paper instead used 40-ms spike-rate windows at 3.4-ms stride, matching the supplied repository.
- Side-view video at 300 Hz was tracked with DeepLabCut. The newer movement paper rejected marker velocity outliers beyond five standard deviations and imputed nearby frames; when tongue was occluded/retracted, it replaced tongue position with its mean. That imputation supported continuous movement regression, whereas this task explicitly requires a categorical “not visible” state, so raw DLC likelihood must be preserved and used instead of mean imputation.
- The method paper excludes firing-rate neurons below 2 Hz for video-to-neural prediction, but the original data paper’s broader neural analyses rely on the classifier-good set. Because our downstream decoder predicts behavior from neural activity and should preserve the paper’s complete curated unit population, the 2-Hz analysis-specific threshold is not automatically applicable.

### Curation Steps

**Neuron curation rules**:
Kilosort2 clusters were characterized by 15 QC metrics. Five region-specific logistic classifiers (cortex, striatum, thalamus, midbrain, medulla), trained from blind manual `good`/`unlabelled` labels, produced the final unit list. Individual metrics overlap strongly, and the white paper explicitly rejects simple one-metric thresholds. Use the NWB `classification == "good"` label. Classifier ROC AUC was >0.9; reported false-alarm rates were cortex 7.8%, striatum 6.4%, thalamus 7.3%, midbrain 5.5%, and medulla 4.3%.

**Trial curation rules**:
For the original data paper, early-lick and no-response trials were excluded from standard analyses; sessions required >65% performance and at least 50 correct left and 50 correct right trials. The newer movement paper additionally excluded photoinhibition and auto/free-water trials from all its movement analyses, and used only correct trials for behavioral prediction. These are analysis-specific exclusions. This decoder explicitly requires early-lick, ignore/no-lick, and photostimulation variables, so excluding them would destroy required target/input classes; retain them unless raw timing/validity is unusable. Auto/free-water trials need separate evaluation because outcome semantics may be altered.

### Decoders Trained
| Decoded variable | Accuracy |
| Neural population → choice (data paper) | Accuracy starts near 0.5 before sample and rises across sample/delay; the plotted 200-neuron regional curves reach roughly 0.7–0.9 near late delay depending on region. Exact point estimates are not tabulated. |
| Behavioral-video embedding → choice (method paper) | ROC AUC 0.51 ± 0.06 before sample, 0.66 ± 0.12 over late sample+delay, and 0.99 ± 0.01 in late response (106 sessions). |
| QC metrics → good unit (white paper) | Cross-validated ROC AUC >0.9 in each major region (not a behavioral decoder). |
| Outcome / early lick / tongue category from neural activity | No directly comparable decoder accuracy reported in the supplied papers. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Session count | Reference preprocessing scans all exported sessions but only produces region files where classifier-good units exist. | 174 NWBs; one session has 1,852 units with `classification=NaN` and `anno_name=NaN`, leaving 173 labeled sessions. | 173 behavioral sessions. | Exclude only the unlabeled/no-curated-unit session. This exactly reconciles session count. |
| Good-unit count | Uses external classifier `goodunits` indices and valid CCF annotation. | 69,453 rows are explicitly `classification == "good"`; all have nonempty CCF annotation. Median 390/session. | 69,943; median 393/session. Figure-region counts sum exactly to 69,943. | The embedded release labels are authoritative for reproducibility. The 490-unit difference is not recoverable from the unlabeled session (which has 1,852 units and no annotations) and is likely a release/export or manuscript-summary discrepancy. Use 69,453, not an invented metric threshold. |
| Neural trial coverage | MATLAB exports expose per-unit/per-trial arrays, implicitly only for recorded trials. | In 9 NWBs, `is_good_trials.shape[1]` is shorter than `intervals/trials` (by 5–376 trials). Unit `obs_intervals` map the 93,310 neural trials to the exact contiguous trial-table block; one session is a suffix (rows 125–629), while the others begin at row 0. | Trial total is not reported. | Map every neural-mask column to the trial table through exact observation-interval start times. Discard the 1,060 behavioral-only rows outside those blocks. |
| Per-probe bad trials | Reference scripts do not explicitly read this NWB-only field. | Four usable sessions contain false `is_good_trials` entries among curated units; requiring all retained neurons valid would exclude 509 trials total. | White paper says unstable recordings/periods were rejected, but gives no trial-mask rule. | Honor the NWB validity annotation and retain a rectangular session population by dropping trials where any retained curated neuron is invalid. This is conservative and required by the instruction to exclude invalid periods. |
| Trial categories | Newer repository’s `get_regular_trial_mask` removes early lick, auto/free water, ignore, and stimulation. | All requested labels and photostim timing are explicitly present; raw source has 14,095 ignores and 10,805 early trials. Auto/free-water trials are irregular reward-delivery trials and many have no spike samples in the requested window. | Same analysis exclusions are described in the method paper. | Retain early/ignore/stimulation trials because they are required labels/inputs, but exclude auto/free-water trials because those variables are not requested, outcome semantics are experimentally altered, and the reference explicitly removes them. |
| Spike binning | Half-open centered windows divided by width; movement paper uses overlapping 40-ms width / 3.4-ms stride. | Global float64 spike timestamps on the NWB session clock. | 40-ms/3.4-ms for movement prediction; other analyses use 200-ms windows. | Decoder specification overrides width/stride: use 80 adjacent 50-ms half-open bins spanning `[-2.5, 1.5)` and divide counts by 0.05 s to Hz. This preserves reference counting/scaling conventions. |
| Tone/sample events | Reference task timeline uses sample onset; early lick can replay sample/delay. | Every ephys-backed trial has at least one `sample_start_times` event in `[trial start, go)`. Early trials often have 2–14; most no-early trials have one. | Tone/sample starts at canonical −1.85 s in standard trials, with replays after early licking. | Define tone onset as the first sample-start event in the behavioral trial, matching “time from tone onset” rather than the final replay. This intentionally makes elapsed time informative on replay/early-lick trials. |
| Choice label | Reference papers define choice as actual left/right lick and distinguish it from instructed stimulus using hit/error trial type. | Instruction + outcome implies choice and agrees with the first directional lick in the 1.5-s answer period on 92,976 / 93,310 ephys-backed trials (99.64%); event discrepancies include tracker/event omissions and occasional rapid opposite licks. | Choice is lick direction; ignore trials have no response. | Use the task-authoritative mapping: hit→instruction side, miss→opposite side, ignore→no lick. It is complete, matches paper definitions, and is more robust than selecting the first lick event. |
| Photostimulation timing | MATLAB preprocessing subtracts go time from stimulation onset/offset. | NWB global start/stop events match trial columns; example starts occur at −1.2 s and stop at −0.7 s relative to go. | Photoinhibition is during late delay and ends before go. | Use the NWB start/stop event pairs directly on the global clock and sample binary state at each decoder bin center. |
| Tongue processing | Newer paper removes >5-SD velocity outliers and mean-imputes occluded tongue for continuous regression. | Raw tongue x/y are numeric even at near-zero DLC likelihood; likelihood is strongly bimodal. | Occluded/retracted tongue is typically not visible before response. | For the required categorical output, low-likelihood frames must map to class 3 (“not visible”), not be mean-imputed. Apply the reference 5-SD velocity outlier idea only to otherwise visible samples, then aggregate to 50-ms bins and discretize visible y by session percentiles. |

Final consistent understanding: use 173 curated ephys sessions, source classifier-good units with CCF labels, observation-interval-mapped ephys trial blocks and per-unit validity masks, exclude auto/free-water trials, retain required early/ignore/stimulation trials, use go-cue alignment and global event/video timestamps, and apply decoder-mandated 50-ms bins. Differences from the remaining paper-analysis trial filters and continuous tongue imputation are necessary to satisfy the requested decoder variables.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `units/spike_times`, `spike_times_index`, `obs_intervals`; `classification`; `is_good_trials`; go timestamps | `neural[session][trial]` | Keep `classification == "good"`; map neural trials to trial-table rows by observation intervals; retain all-unit-valid non-water trials; count spikes in 80 adjacent half-open 50-ms bins over `[-2.5, 1.5)` around go and divide by 0.05 to float32 Hz; transpose to neuron × time. | `sliding_histogram`, `process_one_area` | Decoder width/stride overrides paper’s 40-ms/3.4-ms windows, but half-open counting and Hz scaling match reference. |
| First `sample_start_times/timestamps` event within each trial | `input[..., 0, :]` | For every 50-ms bin center, compute global bin-center time minus that trial’s first tone/sample onset, in seconds (float32). | Task-event alignment in `process_one_sess`; replay behavior in paper Methods | Continuous time-varying input explicitly requested; unlike the generic format example, this is elapsed time, not an onset indicator. |
| `photostim_start_times/timestamps` and paired stop timestamps | `input[..., 1, :]` | Binary float32 state at each global bin center, using half-open `[start, stop)` intervals. | `process_one_sess` stimulation alignment | Retain stimulated trials because photostimulation is a decoder input. |
| `trial_instruction`, `outcome` | `output[..., 0, :]` choice | `hit`→instruction side; `miss`→opposite side; `ignore`→no lick. Codes: left 0, right 1, no lick 2. Repeat the per-trial value across 80 timepoints. | Paper’s LL/LR/RL/RR definition; `create_4fold_trial_type_mask` | Task-authoritative mapping is 99.64% consistent with first response-period lick event. |
| `outcome` | `output[..., 1, :]` outcome | Direct categorical mapping: ignore 0, miss 1, hit 2; repeat across time. | `correctness` mapping in `process_one_sess` | Preserve all three required classes. |
| `early_lick` | `output[..., 2, :]` early lick | `no early`→0, `early`→1; repeat across time. | `early_lick_trials` in reference preprocessing | Preserve rather than applying paper’s exclusion. |
| `Camera0_side_TongueTracking` x/y/likelihood and timestamps | `output[..., 3, :]` tongue y | Use likelihood ≥0.9 as visible; identify >5-SD 2D velocity outliers within continuous high-confidence segments and interpolate them from adjacent valid frames as in the method paper; compute session 40th/60th percentiles from cleaned visible y frames; sample the nearest/last frame at each bin center if within one nominal frame. Codes: y<q40→0, q40≤y≤q60→1, y>q60→2, low confidence/no nearby frame→3. | `align_markers_between_lims`; method-paper marker cleaning | Raw low-confidence coordinates are finite but invalid; do not mean-impute occlusion because class 3 is explicitly required. |
| `general/subject/subject_id` / path subject | `subjects`, `subject_idx` | Deterministic sorted unique subject strings and per-session indices. | — | Expected 28 subjects among 173 sessions. |
| Curated `units/anno_name` | `brain_regions`, `brain_region_idx` | Decode/strip exact Allen CCF region strings; deterministic sorted global vocabulary; one index per retained good unit. | `helper_get_neuron_id_area` validates CCF annotations | Exact source annotations avoid unsupported heuristic remapping to 14 coarse areas. |

### Key Decisions
1. **Session curation**: Exclude `sub-440958_ses-20190216T162508` because it has no classifier or anatomical labels; retain the other 173 sessions. Sessions must retain at least two trials and one neuron after all filters.
2. **Trial curation**: Map the 93,310 `is_good_trials` columns to exact trial-table rows via `obs_intervals`; drop trials invalid for any retained classifier-good neuron (509), auto/free-water trials (3,731), and two all-zero no-coverage edge cases. Keep early/ignore/stimulation trials because those requested classes/inputs must remain. Final expected count after the review iteration is 89,068.
3. **Fixed temporal grid**: Edges are `np.arange(-2.5, 1.5 + 0.05, 0.05)` (81 edges), producing exactly 80 timepoints at centers `-2.475 ... 1.475` s. All neural/input/tongue streams use these centers/windows.
4. **Neural units and scaling**: Use classifier-good units without the movement paper’s analysis-specific 2-Hz threshold. Values are firing rates in Hz, float32, never smoothed or z-scored.
5. **Mixed output representation**: Store every trial output as a `(4, 80)` integer array. Per-trial choice/outcome/early labels are repeated across time so the time-varying tongue output can coexist in the required dense representation.
6. **Tongue visibility**: A DLC likelihood cutoff of 0.9 is justified by the strongly bimodal source distribution and conventional high-confidence DLC use. Percentiles are calculated only over visible, cleaned y samples across the whole session; invisible coordinates must not contaminate thresholds.
7. **Tongue temporal sampling**: Follow the reference marker alignment’s sample-and-hold convention (last raw frame at/before a requested time), with a one-frame tolerance. Do not average visible and invisible samples inside a 50-ms bin because that would blur the required not-visible class.
8. **Brain regions**: Preserve the 294-granularity Allen annotation vocabulary supplied with the data rather than inventing coarse parent mappings not present in the NWBs.
9. **Metadata**: Set `time_bin_size=50.0` ms, `temporal_alignment_event="auditory go cue onset"`, `off_start=-2.5`, `off_end=1.5`, and include units, bin convention, categorical codebooks, tongue rules, curation summary, and per-session source/retention/percentile information.
10. **Determinism and efficiency**: Sort source paths; precompute trial/bin timing; stream one HDF5 session at a time; use vectorized spike-to-trial/bin assignment and float32/int8 outputs; avoid loading waveforms or raw uncurated spikes unnecessarily.

### Planned Sanity Checks
- [ ] Structure: 173 sessions, matching lengths across `neural`/`input`/`output`/indices, ≥2 trials/session, 80 timepoints everywhere, and neuron counts matching `brain_region_idx`.
- [ ] Neural raw spot-check: independently load raw ragged spike timestamps for several `(session, trial, neuron)` triples; recompute `np.histogram` on go-relative edges and verify `np.allclose(raw_counts/0.05, converted)`.
- [ ] Input raw spot-check: independently locate first sample onset and stimulation interval in raw NWB; verify `np.allclose` for elapsed-time and photostim vectors.
- [ ] Output raw spot-check: independently decode instruction/outcome/early for at least three trials and verify repeated values with `np.allclose`.
- [ ] Tongue raw spot-check: independently find selected raw video frames, apply stored session thresholds, and verify categorical vectors with `np.allclose`.
- [ ] Count reconciliation: account explicitly for 94,990 source rows → 620 rows in unlabeled session → 1,060 behavioral-only rows outside observation-interval-mapped ephys blocks → 509 bad-neural trials → 3,731 auto/free-water trials → 2 all-zero no-coverage edges = 89,068 converted trials.
- [ ] Paper comparison: 28 mice; 173 sessions; 69,453 source-labeled good units versus paper 69,943 discrepancy already explained; converted median neurons/session should be 390 versus paper 393.
- [ ] Distribution checks: input elapsed-time finite; photostim binary; categorical outputs within codebooks; tongue visible classes approximate 40%/20%/40% over visible session frames (trial-center sampling can shift these fractions); no NaN/Inf neural or task arrays.
- [ ] Temporal checks: plots for two sessions will overlay trial/tone/go/stimulation timing, neural population mean, raw/cleaned tongue y and likelihood, percentile cutoffs, and final categorical output.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with the required invocation, default `--full`, `--sample`, and `--show-processing`. The script directly streams HDF5-backed NWB groups, performs classifier/session/trial curation, aligns all streams on the global NWB clock, constructs 80 time bins, discretizes tongue y per session, validates shapes/dtypes/ranges in memory, writes a highest-protocol pickle, and records detailed per-session provenance in metadata. `--show-processing` saves eight-panel `processing_<session_id>.png` diagnostics for up to two sessions. Python compilation and CLI help checks passed.

Code inefficiencies identified:
Loading via a full NWB object model would materialize irrelevant waveform/electrode data. Per-trial/per-unit histogram calls would also multiply Python overhead across approximately 69k units and 93k trials. Storing float64 firing rates would double the dominant output size.

Code speedups added:
Use direct `h5py` access; load only small metadata/video arrays and selected ragged spike slices; assign all spikes from one unit to non-overlapping trial windows with vectorized `searchsorted` + `bincount`; precompute all timing grids; keep neural/input arrays float32 and output arrays int8; process one session at a time; retain trial arrays as views over session-contiguous blocks until serialization. Per-session timing and pickle-write timing are printed for the Step 7 estimate.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 834 session-neurons |
| Neurons / session | 459, 375 (mean 417) |
| Subjects | 1 (`440956`) |
| Sessions / subject | 2 |
| Trials (total) | 513 |
| Trials / session | 354, 159 |
| Time from tone onset range | [-0.6, 5.7] s |
| Photostimulation range | [0, 1] |
| Choice distribution | [0.456 left, 0.407 right, 0.136 no lick] |
| Outcome distribution | [0.136 ignore, 0.298 miss, 0.565 hit] |
| Early-lick distribution | [0.942 no, 0.058 yes] |
| Tongue distribution | [0.053 low, 0.029 middle, 0.061 high, 0.856 not visible] |

### Processing Plots Review
Both required session plots were generated and visually inspected. Tone onsets cluster at the canonical −1.85 s with longer elapsed intervals on replay/early-lick trials; photostimulation occupies late delay and ends before go; neural population firing shows sensible response-locked structure; low tongue likelihood is mapped to class 3; visible-y thresholds partition the session distribution at the displayed 40th/60th percentiles; fixed trial labels remain constant over time. No alignment anomaly was seen.

The initial verifier warned that the final nominal ephys trial in the second sample session was all-zero across 375 curated units. Raw coverage showed that spikes ended before this window even though `is_good_trials` included it. A later full-data review also established that neural trial blocks must be mapped by `obs_intervals` rather than assumed to be trial-table prefixes, and that auto/free-water trials should be excluded in agreement with the reference. The converter and all sample artifacts were regenerated. Session 1 retains 354/368 trials after excluding 14 water trials; session 2 retains 159/160 after the no-coverage edge rejection. Verification passes with no errors or warnings.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Direct HDF5 field access and vectorized per-unit spike assignment | Final sample conversion work completed in 0.92 s across sessions; avoids per-trial histogram loops. |
| float32 neural/input and int8 output arrays | Estimated dominant output memory/disk reduced by roughly half versus float64 neural arrays. |

| Step | Time / Session | Estimated Total Time |
| Neural/task conversion | 0.34 s mean in the two sample sessions | ~111 s after scaling by full retained trial×neuron work (164× sample) |
| Diagnostic plotting | ~0.9 s/session (sample only) | not used for full conversion |
| Sample pickle serialization | 0.08 s for 0.067 GiB | final full serialization took 14.56 s for 10.83 GiB |
| Total full conversion estimate | — | approximately 2–3 minutes, well below 15 minutes |

Required files `/app/sample_data.pkl`, `/app/conversion_sample_out.txt`, and `/app/verification_sample_out.txt` exist. The final verifier reports valid format with no errors or warnings.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None after the all-zero coverage fix and reconversion

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Lick direction choice | 0.7340 | 0.6204 |
| Outcome | 0.7661 | 0.6365 |
| Early lick | 0.8422 | 0.7568 |
| Tongue y-position | 0.6749 | 0.5021 |

Final post-review sample training completed on CUDA in approximately 8 s. Loss decreased from 13.1908 (epoch 1) to 0.5733 (epoch 200); test loss was 1.0121. All validation balanced accuracies exceed uniform chance (choice/outcome 0.3333, early 0.5, tongue 0.25), supporting correct temporal alignment and label construction on the sample.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 11,624,636,855 bytes (10.826 GiB)
- `verification_full_out.txt`: created
- `conversion_full_out.txt`: created; final corrected conversion completed in 148.64 s (130.42 s summed session work + 14.56 s serialization)

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total neurons | 69,943 | external classifier good-unit lists | 69,453 explicit `classification=good` | 69,453 | Source/converted exact; paper differs by 490 as documented |
| Mean neurons/session | median 393 | hundreds/session after classifier+CCF filtering | mean 401.46, median 390, range 90–923 | mean 401.46, median 390, range 90–923 | Yes (source exact; paper median close) |
| Subjects | 28 | session mouse parsed from export | 28 | 28 | Yes |
| Sessions | 173 | processes sessions with curated area units | 174 NWBs, 173 with classifier labels | 173 | Yes |
| Trials (total) | not reported | excludes early/ignore/stim/water for method analyses | 94,990 table rows; 93,310 ephys-backed in curated sessions | 89,068 after documented validity/water/coverage exclusions while retaining required classes | Reconciled |
| Trials/session (mean) | 476 (paper analysis; range 130–785) | analysis-dependent regular trials | ephys-backed mean 539.36 | mean 514.84, median 517, range 159–796 | Difference required by retaining early/ignore/stim trials |
| Time from tone onset range | canonical tone at −1.85 s; replays possible | replay-aware event timestamps | first sample event per retained trial | [-1.5, 11.9] s over bin centers | Yes; long positive values are early-lick replay trials |
| Photostimulation range | binary presence; late delay | stimulation onset/offset aligned to go | event intervals | [0, 1] | Yes |
| Choice distribution | not globally reported | derived from instruction×correctness | required mapping | [0.429, 0.422, 0.149] left/right/no lick | Source-derived |
| Outcome distribution | 84% correct only for selected control/non-early analysis trials | correctness 1/0/−1 | three NWB labels | [0.149, 0.166, 0.685] ignore/miss/hit | Consistent given broader retained trials |
| Early-lick distribution | excluded in paper analyses | explicit early-report flag | two NWB labels | [0.884, 0.116] no/yes | Source-derived |
| Tongue distribution | occluded before response; no class proportions reported | 3.4-ms tracked marker alignment | likelihood-gated y | [0.062, 0.032, 0.065, 0.841] low/mid/high/not visible | Plausible and percentile-consistent among visible samples |

Final accounting: 94,990 raw trial-table rows − 620 rows in the unlabeled session − 1,060 behavioral-only rows outside ephys observation blocks − 509 per-unit-invalid trials − 3,731 auto/free-water trials − 2 all-zero coverage edges = 89,068 converted trials. The final verifier reports valid format with no errors or warnings. Spot checks included prefix and suffix neural blocks, stimulated/unstimulated trials, early/ignore trials, and the two no-coverage edge removals.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output-log verification**: searched `verification_full_out.txt`, `conversion_full_out.txt`, the final sample-training log, and `sanity_checks_out.txt` for errors, warnings, invalid values, NaNs, and infinities. The full verifier explicitly reports "Data format is valid, no errors or warnings." No actionable warning remains.
2. **Independent raw-data `np.allclose()` checks**: `cache/sanity_checks.py` loaded the NWBs directly (without importing `convert_data.py`) and compared three sessions/trials, including the non-prefix session whose converted trial 0 is raw trial 125. For first/third/last curated neurons it independently called `np.histogram` on raw ragged spikes and compared all 80 firing-rate bins; it also compared spike totals, time-from-tone, stimulation state, choice, outcome, early lick, all 80 tongue classes, tongue quantiles, CCF labels, and provenance indices. All comparisons passed. Transcript: `cache/sanity_checks_out.txt`.
3. **Whole-data shape/edge audit**: checked every one of 89,068 trials. All neural/input/output shapes are `(n_neurons,80)`, `(2,80)`, and `(4,80)`; all values are finite; firing rates are nonnegative multiples of 20 Hz (integer spike counts divided by 0.05 s); stimulation and output codes are in their declared alphabets; per-trial labels are constant across time; region indices are in bounds; retained source row IDs are strictly increasing and fall inside each ephys observation block. The first/last centers are exactly −2.475/+1.475 s and the minimum session size is 159 trials. No issue was found.
4. **Key-statistics comparison**: repeated the complete Step 9 comparison (animals, sessions, trials, neurons, trial/session and neuron/session summaries, input ranges, all output distributions). Converted totals exactly match the curated source counts: 28 animals, 173 usable sessions, 69,453 explicitly good units, and 89,068 fully accountable retained trials. The only paper/source discrepancy is the already documented 490-unit difference (69,943 paper versus 69,453 explicit NWB classifier labels); the authoritative row-level NWB labels were retained rather than manufacturing missing units.
5. **Boundary and missing-data checks**: verified half-open 50-ms intervals, exact 80-bin coverage, suffix as well as prefix ephys blocks, sparse ragged spike vectors, trial-table/ephys offsets, invalid per-unit observation masks, absent/low-confidence tongue frames, video-frame tolerance at bin edges, auto/free-water flags, stimulated/unstimulated trials, early/ignore labels, empty-spike edge trials, and sessions with the minimum trial count. The unlabeled 174th NWB is excluded before conversion and no retained session has fewer than two trials.

### Reference Code Comparison
| Processing stage | Reference implementation/method | Conversion implementation | Result / justified difference |
|------------------|---------------------------------|---------------------------|-------------------------------|
| Data loading | `preprocessing_DJ_2022Aug.py` loads session spike times, behavioral flags, go cue, and CCF metadata | `discover_usable_files()` and `convert_session()` stream the corresponding NWB datasets with `h5py` | Same source concepts; direct NWB access avoids lossy intermediate MATLAB exports |
| Neuron/trial filtering | supplied good-unit QC lists; `get_regular_trial_mask()` removes early, water, ignore, and stimulation for the paper's regular-trial analyses | exact NWB `classification == good`; removes invalid observation periods and water, but retains early/ignore/stim | Neuron rule matches classifier curation. Required decoder labels/input necessitate retaining early, ignore, and stimulation trials; water is still removed |
| Temporal alignment | reference spike arrays are go-cue-relative; video functions align to `go_times`, generally taking the last nearby frame | uses NWB absolute go timestamps, then subtracts/adds fixed relative edges; tongue uses last frame at/before each center within 5.1 ms | Equivalent go-cue alignment, independently verified on raw timestamps |
| Binning | `sliding_histogram()` counts half-open windows and divides by bin width; method analysis uses 40-ms windows/3.4-ms stride | adjacent half-open 50-ms bins divided by 0.05 s | Same histogram/rate definition; width/stride changed exactly as required by this decoder task |
| Input construction | task events and photostimulation are aligned to go cue | first in-trial NWB `sample_start` supplies continuous time-from-tone; NWB start/stop intervals give binary stimulation at each center | Uses native event streams and requested representations |
| Output construction | task flags come from trial arrays; video marker preprocessing aligns DLC and removes >5-SD velocity artifacts | instruction×outcome defines choice; native outcome/early flags are encoded; side-camera tongue uses likelihood ≥0.9, >5-SD velocity interpolation, session q40/q60, and explicit not-visible | Task labels match raw rows. Tongue cleaning matches the method, while explicit class 3 replaces the paper's occlusion mean-imputation as required |

### Issues Found and Resolved
- **Iteration 1 — sample all-zero neural trial**: the first sample verifier exposed an all-population-zero boundary trial with no usable spike coverage. Added a coverage guard; two such trials are removed globally. Sample conversion, verification, and training were rerun.
- **Iteration 2 — trial-block offset**: full exploration showed that one session's ephys observation block begins at raw trial 125 rather than trial 0. Prefix slicing would silently attach wrong behavior to spikes. Replaced it with exact `obs_intervals` start-time mapping to trial-table rows and added provenance metadata/checks. Independent raw comparison of this session passes.
- **Iteration 3 — water trials**: comparison to `get_regular_trial_mask()` identified auto/free-water trials as nonstandard and sometimes outside reliable neural coverage. Added explicit removal of 3,731 such rows while retaining early/ignore/stim trials needed for this task. All sample/full conversions, verification, and sample training were rerun after the fix.
- **Final re-check**: after these fixes, all five checks above were repeated. No mismatch, warning, invalid value, or unresolved edge case remains.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. Full CUDA training completed all 200 epochs; loss fell monotonically in the reported checkpoints from 18.4498 (epoch 1) to 0.6543 (epoch 200). Held-out test loss was 0.6579. `train_decoder.py` printed its successful-completion marker and generated `sample_trials.png` and `predictions.png`.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Lick direction choice | 0.7313 | 0.6950 | 3 classes; chance 0.3333 |
| Outcome | 0.7080 | 0.6627 | 3 classes; chance 0.3333 |
| Early lick | 0.7967 | 0.7567 | 2 classes; chance 0.5000 |
| Tongue y-position | 0.6693 | 0.6252 | 4 classes; chance 0.2500 |

Training used 71,189 trials and evaluation used 17,879 trials. Every held-out balanced accuracy exceeds both chance and 1.5× chance, and every training/validation ratio is below 1.08, indicating useful signal without a problematic generalization gap.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Validation balanced accuracy | Chance | Multiple of chance | Train / validation | Expectation from papers |
|----------|------------------------------|--------|--------------------|--------------------|-------------------------|
| Lick direction choice | 0.6950 | 0.3333 | 2.085× | 1.052 | Data-paper population curves are about 0.5 before sample and roughly 0.7–0.9 in informative sample/delay/response periods; method-paper video choice AUC is 0.66 in sample/delay |
| Outcome | 0.6627 | 0.3333 | 1.988× | 1.068 | No outcome decoder accuracy reported |
| Early lick | 0.7567 | 0.5000 | 1.513× | 1.053 | No early-lick decoder accuracy reported |
| Tongue y-position | 0.6252 | 0.2500 | 2.501× | 1.071 | No discretized tongue-position decoder accuracy reported |

Every output exceeds chance and 1.5× chance. No train/validation ratio approaches the specified 1.5 overfitting threshold. Early lick is closest to the 1.5× criterion, so it received extra scrutiny: the class has adequate prevalence (11.6%), its three raw-trial spot checks match exactly, its sample/full validation accuracies agree (0.7568/0.7567), and its train/validation gap is only 0.040. This is stable signal, not leakage or a conversion artifact.

### Comparison to Every Reported Decoding Accuracy
| Paper result | Reported value | Comparison and investigation |
|--------------|----------------|------------------------------|
| Data paper, population neural choice decoding (Figure 6D/S6; binary choice, 200-neuron pseudo-populations, 200-ms causal windows) | Figure curves: chance 0.5 before informative epochs and approximately 0.7–0.9 depending on area/time; no single aggregate numeric accuracy stated in text | Our harder three-class, whole-window balanced accuracy is 0.695. The raw choice labels, go alignment, spike bins, and task variation were independently checked; performance is consistent with the lower edge of the paper curves and rises well above its own 0.333 chance |
| Data paper, selected single-neuron movement-subtraction examples (Figure 3E) | AUC 0.82→0.52 and 0.91→0.87 | Selected single-neuron binary AUC examples answer a different question and are not aggregate decoder targets; included here so no reported decoder number is omitted |
| Method paper, video-to-choice before sample | AUC 0.51 ± 0.06 | Near binary chance as expected before sensory information; not directly comparable to our neural whole-window three-class score |
| Method paper, video-to-choice sample/delay | AUC 0.66 ± 0.12 | Our neural choice score is 0.695, closely consistent despite different modality, class count, and temporal aggregation |
| Method paper, video-to-choice response epoch | AUC 0.99 ± 0.01 | Higher because directional licking is visually explicit in the response epoch. We checked rather than dismissing the difference: raw labels match, response-aligned bins are correct, no-lick is an additional balanced class, and our metric aggregates the full −2.5-to-+1.5-s window |

The method paper's AUC >0.60 and AUC >0.65 values are selection/classification thresholds for defining neurons or sessions, not achieved decoder accuracies. They were therefore not treated as performance baselines.

### Diagnostic Checks

1. Reloaded and compared all four raw outputs on three specific trials in Step 10, including an ephys-offset session; all values matched with `np.allclose()`.
2. Confirmed output variation is substantial rather than a 99% majority-class artifact: choice `[0.429, 0.422, 0.149]`, outcome `[0.149, 0.166, 0.685]`, early `[0.884, 0.116]`, tongue `[0.062, 0.032, 0.065, 0.841]`. Balanced accuracy prevents the visible tongue majority class from inflating the reported result.
3. Confirmed temporal synchronization through independent raw timestamp reconstruction and visual inspection of `sample_trials.png` and `predictions.png`. Tone time ramps by exactly 50 ms/bin, stimulation is bounded and binary, per-trial labels are constant, and tongue visibility varies at plausible response-period times.
4. Rechecked filtering against the supplied classifier labels, observation intervals, water flags, reference code, and methods. No change was warranted.

### Issues Found and Resolved
- No new issue was found in this review. Because all accuracy, gap, raw-label, alignment, variation, and processing checks passed, no conversion rerun was necessary.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created with dataset description, loading example, field/code definitions, provenance summary, key statistics, reproduction commands, and decoder results.
- [x] `cache/` created; independent investigation code and its passing transcript moved there and documented in `cache/README_CACHE.md`.
- [x] All files organized. Required outputs and plots were checked for existence and nonzero size; both Python scripts compile; the cached independent sanity audit was rerun successfully after the move.

Final handoff checks: all 14 workflow steps are marked complete, the full verifier reports no errors or warnings, the 200-epoch decoder run ends with `train_decoder.py finished successfully`, and every required deliverable exists.
