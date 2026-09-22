# Dataset Conversion Notes

## Overview
- **Dataset**: Brain-wide neural activity underlying memory-guided movement (provided NWB dataset)
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

Environment verification: Python 3.13.15; NumPy 2.4.4; PyTorch 2.6.0+cu124. The required checkpoint `ls -la /app/CONVERSION_NOTES.md` succeeded.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `loadmat` | `code/VideoAnalysisUtils/preprocessing_utils.py` | LOADING | Loads MATLAB exports while recursively converting MATLAB structs/cells into Python dictionaries/lists. |
| `process_one_sess` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING / CURATION | Groups probes into a session; reads task/behavior, spike times, unit QC, histology; intersects ephys units with histology; applies classifier-QC unit indices. |
| `helper_get_neuron_id_area` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | CURATION | Intersects classifier-approved units with hemisphere (CCF ML midline 5700 um) and checks QC annotations against histology. |
| `sliding_histogram` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Counts spikes in half-open windows and divides by bin width to yield Hz, with output shape time x trial x unit. |
| `process_one_area` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Truncates already go-cue-aligned spikes, bins them, and saves neural, behavior, task, QC, and CCF fields per session/area. |
| `load_session` | `code/VideoAnalysisUtils/population_decoding_utils.py` | LOADING | Concatenates region files along the neuron axis while retaining session-wide trial variables. |
| `get_regular_trial_mask` | `code/VideoAnalysisUtils/population_decoding_utils.py` | CURATION | For the paper's regular-trial analyses excludes early lick, auto-water, free-water, no-response, and stimulation trials. |
| `align_markers_between_lims` | `code/Sherlock/align_markers.py` | PROCESSING | Aligns 294-Hz marker streams to go cue using frame times (`dt=0.0034 s`) and retains the last frame in each time cell. |
| `temporal_alignment_embed_and_ephys` | `code/VideoAnalysisUtils/functions_for_r2.py` | PROCESSING | Crops ephys and video-derived arrays to their common time interval using closest timestamps. |
| `create_4fold_trial_type_mask` | `code/VideoAnalysisUtils/functions_for_r2.py` | PROCESSING | Encodes hit/miss crossed with instructed side for stratified cross-validation. |

### Notes
- Repository is the analysis code for *Brain-wide analysis reveals movement encoding structured across and within brain areas*. The raw public data are the MAP DANDI dataset.
- This is electrophysiology, not imaging; delta-F/F is not applicable.
- The authors' current preprocessing entry point (`Sherlock/preprocess_all_ephys.py`) uses 40-ms spike-count windows, 3.4-ms stride, -3 to +3 s around go cue, and `qc_mode='classifier'`. The current decoder task explicitly overrides bin width/window to non-overlapping 50-ms bins from -2.5 to +1.5 s.
- Raw exported spike times are documented as already relative to go cue. Absolute lick and photostimulation times are shifted by `task_cue_time` before saving.
- Task fields identified: autolearn, early report, auto/free water, lick directions/times, correctness (`1` correct, `0` error, `-1` no response), cue/sample/delay times, stimulation power/type/on/off, and instructed trial type.
- Tracking fields used by the reference marker pipeline are side-camera `nose_x/y`, `tongue_x/y`, `jaw_x/y`, and `whisker_x/y`; trial indices are explicitly sorted after alignment.
- Reference “regular trial” filtering is analysis-specific and removes precisely several categories that are required decoder outputs/inputs here (early lick, no response, and stimulation). It therefore cannot be applied wholesale; neuron QC and valid-stream checks remain applicable.
- Region processing is limited to 14 coarse groups (ALM, Medulla, Midbrain, Striatum, Thalamus, Pons, Cerebellum, Hypothalamus, Hippocampus, Orbital, OtherCortex, Olfactory, CorticalSubplate, Pallidum), separated by hemisphere in saved intermediate files.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/dandiset.yaml` identifies DANDI:000363 version 0.230822.0128, “Mesoscale Activity Map Dataset.”
- There are 28 `sub-*` directories containing 174 NWB/HDF5 session files (about 50 GiB total). Subject session counts range from 3 to 10.
- `intervals/trials` contains one row per trial: `start_time`, `stop_time`, 1-based `trial`, `trial_uid`, `task`, `task_protocol`, instructed side (`trial_instruction`), `early_lick`, `outcome`, `auto_water`, `free_water`, and photostimulation onset/power/duration (stored as string-like mixed columns with `N/A`). Native trial arrays are 264–800 rows/session.
- `acquisition/BehavioralEvents` contains absolute timestamps for presample, sample/tone, delay, go, trial-end, left/right licks, and photostimulation starts/stops. Every session has exactly one go-start timestamp per trial. There are 18,588 photostimulation intervals across the dataset in 168 sessions.
- `acquisition/BehavioralTimeSeries/Camera0_side_{Jaw,Nose,Tongue}Tracking` contains dense absolute timestamps and three float64 columns per marker (x, y, likelihood). All 174 sessions have tongue tracking. The median timestamp interval is 0.003400000544 s (~294.12 Hz).
- `units` is a ragged NWB Units table. `spike_times` are absolute continuous-session seconds with `spike_times_index` row endpoints. It includes 272,227 sorted units and classifier labels (`good` or `unlabelled`). All 69,453 `good` units have nonempty `anno_name` CCF labels.
- Unit metadata include waveform/QC measurements (`unit_amp`, SNR, ISI violation, average firing rate, drift, presence ratio, amplitude cutoff, isolation distance, L-ratio, d-prime, nearest-neighbor metrics, waveform shape), electrode references, CCF annotation, and `is_good_trials` (unit x trial boolean validity).
- `general/extracellular_ephys/electrodes` contains probe/electrode identity, group, full CCF location text, and x/y/z coordinates; the units table links units to electrodes.
- Native spike times and behavioral timestamps share the same absolute session clock; this differs from the paper's MATLAB export where trial spike arrays were already go-cue-relative.
- The requested -2.5 to +1.5 s window may cross NWB trial-table start/stop boundaries (2,963 starts; 14,844 stops), but those interval boundaries are task bookkeeping rather than neural recording boundaries. Continuous spikes remain available. Tongue timestamp coverage misses only 6 requested starts and 800 requested ends; uncovered output bins can be assigned the specified “not visible” class.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 272,227 sorted; 69,453 classifier-`good` |
| Neurons / session | sorted: 493–3,191 (mean 1,564.5); classifier-good: 0–923 (mean 399.2, median 390) |
| Subjects | 28 |
| Sessions / subject | 3–10 |
| Sessions | 174 native; one (`sub-440958_ses-20190216T162508`) has 0 classifier-good units |
| Trials (total) | 94,990 |
| Trials / session | 264–800 (mean 545.9, median 534) |
| Good unit x trial validity | 37,678,473 pairs; 64,612 (0.1715%) marked false |
| CCF annotations among good units | 293 unique full annotation strings |

Native categorical totals: instructed left/right = 46,077/48,913; outcome hit/miss/ignore = 65,254/15,641/14,095; early/no-early = 10,805/84,185; auto-water = 1,339; free-water = 2,450.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 69,943 good units (25.9% of Kilosort2 clusters) | Data paper Methods / QC white paper: “69,943 good units recorded across 173 behavioral sessions.” | 
| Neurons by selected area | ALM 8,717; striatum 7,664; thalamus 12,808; midbrain 7,495; medulla 2,928 | Data paper Methods and QC white paper. |
| Neurons / session | Not tabulated | — |
| Subjects | 28 mice | Data paper Methods: “based on data from 28 mice.” |
| Sessions / subject | Not tabulated | — |
| Sessions | 173 analyzed behavioral sessions; 655 probe insertions (paper text; Figure 1 caption says 660 penetrations) | Data paper Methods / Figure 1J. |
| Trials (total) | Not tabulated | — |
| Trials / session | Mean 476, range 130–785 in selected analysis sessions | Data paper Methods. |
| Neural data time bin | Data paper: 1-ms PSTH counts smoothed 200 ms; video prediction: 40-ms width/17-ms step. Method paper: 40-ms width/3.4-ms stride. Choice decoder: causal 200-ms width/10-ms step. | Data paper and method paper Methods. |
| Behavior data time bin | Video acquired at 300 Hz; NWB timestamps give ~3.4 ms | Both papers' Methods. |
| Reward/correct rate | 84% mean, range 65–99%, computed on control trials excluding early licks | Data paper Methods. |
| Session selection | Performance >65% and at least 50 correct left and 50 correct right trials | Data paper Methods. |
| Sample epoch | 0.65 s; 3 or 12 kHz tone, three 150-ms presentations with 100-ms gaps | Data paper Methods. |
| Delay/response | 1.2-s delay; 6-kHz go cue; 1.5-s answer period | Data paper Methods. |
| Photostimulation | About 25% randomly interleaved trials in 17 mice; last 0.5 s of delay and ends before go | Data paper Methods. |


### Processing Details
- Data were recorded with Neuropixels probes at 30 kHz, spike sorted with Kilosort2, and aligned across streams via recorded synchronization.
- The original paper used correct and incorrect trials for selectivity; choice was explicitly defined as the actual lick direction and distinguished from stimulus/instruction. Four groups were stimulus-choice LL/LR/RL/RR.
- The newer method paper binned spikes into firing rates with a 40-ms window and 3.4-ms stride. Its tracking pipeline used side-view video, aligned at frame cadence, detected marker velocity outliers using a five-SD threshold, imputed outliers from nearby frames, and replaced occluded tongue positions with their mean for continuous-regression analyses.
- The present task overrides the neural bin width/stride with 50-ms bins and requires occluded tongue to be class 3, so mean-imputation is inappropriate for the output label even though outlier handling remains relevant.
- Reference trial filters differ by analysis. Both papers generally exclude early-lick and no-response trials; the newer method paper additionally excludes photoinhibition and free-water trials. However, the required decoder outputs explicitly include early lick and ignore, and the input explicitly includes photostimulation, so those categories must be retained here.
- Population choice decoding in the data paper used 200 neurons, 200-ms bins stepped 10 ms, regularized logistic regression, nested fivefold CV, and hierarchical bootstrapping with balanced LL/RR/LR/RL trials. This is useful context but is not numerically comparable to the supplied multi-output decoder.

### Curation Steps

**Neuron curation rules**:
Use only classifier-labeled `good` units. The QC classifiers are five region-specific logistic regressions trained from blinded manual labels using 15 jointly interpreted metrics. The white paper explicitly warns that simple per-metric thresholds cause unacceptable misses/false alarms. Classifier cross-validated ROC AUC exceeded 0.9; false-alarm rates were cortex 7.8%, striatum 6.4%, thalamus 7.3%, midbrain 5.5%, medulla 4.3%. The method paper later applied an additional >2-Hz threshold only for its movement-prediction regression; this is analysis-specific and not part of core dataset QC.

**Trial curation rules**:
Reference behavioral analyses selected high-performance sessions and often excluded early, ignore, free-water, and photostimulation trials. These exclusions must not remove explicitly requested decoder classes/inputs. Retain trials with valid go-cue timestamps and neural recording coverage; represent unavailable tongue frames as “not visible.” Auto/free-water remain documented nuisance flags and will be assessed before deciding whether they alter requested labels.

### Decoders Trained
| Decoded variable | Accuracy |
| Neural population choice (data paper) | Time-resolved curve, chance 0.5; no single exact aggregate value tabulated. Figure 6D visually rises well above chance through sample/delay. |
| Choice from full video embedding (method paper) | ROC AUC 0.51 ± 0.06 pre-sample, 0.66 ± 0.12 during second-half sample + delay, 0.99 ± 0.01 second-half response (106 sessions). |
| Choice from markers/full embedding in response (method paper) | Marker AUC 0.88 ± 0.01; embedding AUC 0.96 ± 0.00 (106 sessions). |
| Single-neuron choice/uninstructed movement (method paper) | AUC >0.65 used as modulation threshold, not a population decoder accuracy. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Session/unit count | Classifier-QC units with histology are retained | 174 NWBs; 69,453 explicit `good` units; one session has 0 `good` labels and every annotation is blank | 173 sessions; 69,943 good units | Exclude the zero-good session, yielding 173 usable sessions. The 490-unit numerical gap is exactly consistent with missing QC/CCF export for that excluded session (69,943 - 69,453 = 490). Do not reconstruct labels from thresholds, because the white paper says thresholding is invalid and spatial labels are unavailable. |
| Unit QC | Load external classifier-approved indices | NWB stores final `classification` directly | Use five region-specific classifier outputs | Use `classification == 'good'`; do not re-threshold the 15 metrics and do not apply the method-paper's later analysis-specific >2-Hz filter. |
| Unit-trial validity | MATLAB pipeline does not expose/use a validity mask | `is_good_trials`: 64,612 false of 37.68M good-unit/trial pairs, affecting 565 units, 509 trials, 4 sessions | No explicit per-trial unit rule | The target workflow explicitly says to honor valid-period variables. Exclude any trial for which any retained good unit is marked invalid, preserving a fixed valid neuron set per session. |
| Spike clock | MATLAB trial spike arrays already relative to go cue | NWB spikes and go events are absolute session timestamps | Analyses align to task events/go | Subtract each trial's absolute go timestamp conceptually via absolute bin edges; no further synchronization or offset is required. |
| Neural binning | Method code: 40-ms sliding window, 3.4-ms stride | Raw spike timestamps | Same in method paper; other analyses use 200-ms bins | Decoder specification overrides this: use 80 adjacent half-open 50-ms bins spanning exactly [-2.5, +1.5) s relative to go, report rates in Hz. |
| Trial filtering | “Regular” mask excludes early, ignore, stimulation, auto/free water | All required categories are present | Papers use analysis-dependent exclusions | Keep early, ignore, and stimulated trials because they are explicit outputs/inputs. Keep auto/free-water trials unless invalid by `is_good_trials`; their recorded outcome/choice remains authoritative and metadata will disclose them. |
| Actual choice | Raw code loads lick-direction lists | `trial_instruction` + `outcome`; event lick timestamps also available | Choice is actual lick direction, distinct from stimulus | Map hit to instructed side, miss to opposite side, ignore to no lick. This agrees with the first response-window lick in 94,669/94,990 native trials (99.66%); rare event conflicts/carryover licks make trial outcome the authoritative label. |
| Tone onset | Raw code loads sample time; early licks can replay sample/delay | Every trial has >=1 `sample_start_times`; 5,534 trials have repeats | Tone is the instruction sample; early lick triggers epoch replay | Select the last sample-start event before the final go cue in the trial, i.e. the tone epoch that led into the aligned delay/go. Most are go-1.85 s; protocol variants at -0.95/-2.45 s are preserved. |
| Photostimulation | Shift onset/offset from trial clock to go-relative time | Absolute start/stop events; 18,588 paired intervals, exactly matching non-`N/A` trial rows | Last 0.5 s delay, ends before go | Evaluate the paired absolute event intervals on decoder time-bin centers; do not infer from power alone. |
| Tongue tracking | Reference marker alignment uses side camera at 3.4-ms cadence | x, y, likelihood on absolute clock; likelihood is strongly bimodal | Method paper imputes occluded tongue to mean for continuous regression | For requested categorical output, use nearest frame at each bin center, consider likelihood >=0.9 visible, compute session p40/p60 from visible y values, and label missing/low-confidence values as class 3. This intentionally differs from regression imputation so “not visible” remains meaningful. |
| Brain region naming | Intermediate code groups units into 14 coarse region families/hemispheres using external QC dictionaries | NWB provides 293 complete Allen CCF annotation strings for good units, but no explicit 14-family column | Papers analyze both coarse areas and CCF subregions | Preserve the supplied full `anno_name` strings. Inventing a coarse mapping without the external ontology/QC dictionaries would discard valid anatomy and be less reproducible. |
| Trial-count summaries | Regular-trial mask yields fewer trials for specific analyses | 94,990 native trials; 94,370 in 173 usable neural sessions before validity filtering | Mean 476, range 130–785 after paper-specific behavioral selection | Different counts are expected because this decoder must retain ignored, early, and stimulation trials. Dataset-native and converted totals will both be reported. |

Final consistent understanding: this is continuous-clock NWB electrophysiology with final classifier QC already embedded. Session-wide event and tracking streams must be sampled against go-aligned 50-ms bins. Analysis-specific exclusions and bin widths from the papers are superseded only where they would remove mandated labels/inputs or conflict with the requested temporal grid.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `units/spike_times`, `spike_times_index`; `BehavioralEvents/go_start_times/timestamps` | `neural[session][trial]` | Retain `classification == good`; count absolute spike timestamps in 80 adjacent half-open bins `[go-2.5, go+1.5)` of width 0.05 s; divide counts by 0.05 to Hz; transpose to neuron x time; float32. | `sliding_histogram`, `process_one_area` | Requested bins replace reference 40-ms sliding windows. Bin centers are -2.475,…,+1.475 s. |
| Last `sample_start_times/timestamps` before trial go | `input[...,0,:]` | For every bin center, compute absolute center timestamp minus that trial's final tone/sample onset; float32 seconds. | Raw `task_sample_time`; `align_markers_between_lims` clock logic | Continuous signed time, as explicitly requested; preserves replayed sample epochs and protocol-dependent timing. |
| `photostim_start_times/timestamps`, `photostim_stop_times/timestamps` | `input[...,1,:]` | Binary 1 where absolute bin center lies in any paired half-open laser interval `[start, stop)`, else 0; float32. | `process_one_sess` stimulation alignment | Start/stop counts must match and non-N/A trial count must match interval count. |
| `trial_instruction`, `outcome` | `output[...,0,:]` (`choice`) | `hit`→instruction, `miss`→opposite instruction, `ignore`→no lick; encode left=0, right=1, no lick=2; broadcast across 80 bins. | `create_4fold_trial_type_mask`; paper LL/LR/RL/RR definition | Outcome is authoritative over rare lick-event conflicts. |
| `outcome` | `output[...,1,:]` | Encode ignore=0, miss=1, hit=2 and broadcast. | Raw `correctness` mapping in `process_one_sess` | Retains all mandated classes. |
| `early_lick` | `output[...,2,:]` | Encode `no early`=0, `early`=1 and broadcast. | `get_regular_trial_mask` (reference exclusion only) | Retained as required output. |
| Side-camera `TongueTracking/data[:,1:3]` and timestamps | `output[...,3,:]` | Use likelihood >=0.9 as visible; repair high-confidence >5-SD velocity outliers by interpolation; compute p40/p60 from all visible corrected session y values; sample nearest frame to each bin center. Encode y<p40=0, p40<=y<=p60=1, y>p60=2, unavailable/low-likelihood=3. | `align_markers_between_lims`; method-paper tracking preprocessing | Per-session thresholds stored in metadata. p40/p60 exclude invisible frames. |
| `general/subject/subject_id` / `sub-*` directory | `subjects`, `subject_idx` | Deterministic sorted unique subject IDs and per-session index. | — | Cross-check file subject and NWB subject agree. |
| Good-unit `anno_name` | `brain_regions`, `brain_region_idx` | Deterministic sorted unique full Allen CCF strings and per-unit integer lookup. | `helper_get_neuron_id_area` (reference used external coarse grouping) | Preserves all 293 source annotations. |
| Trial `obs_intervals`, `is_good_trials`; requested window bounds | trial inclusion | Map the common good-unit `obs_intervals` exactly to trial-table start/stop; retain only columns where every retained unit is good; require full requested neural window inside overall observed recording span. | Reference core QC; NWB valid-period fields | Handles nine partial-recording NWBs and four sessions with bad unit-trial flags. |
| `task`, `task_protocol`, auto/free-water, lick events, presample/delay/trial-end events; jaw/nose tracking; all QC metrics/electrode metadata | metadata / unused | Summarize in session metadata or provenance; not decoder variables. | Reference loading code | Available variables are intentionally not added because the Decoder Task specifies exactly two inputs and four outputs. |

### Key Decisions
1. **Fixed grid**: 80 bins with edges `np.linspace(-2.5, 1.5, 81)`; centers are edge midpoints. This prevents off-by-one ambiguity and uses half-open counting exactly like the reference histogram.
2. **Mixed output shape**: Every output trial is `(4, 80)`. Trial-level choice/outcome/early labels are broadcast through time so they coexist with time-varying tongue y in one dense categorical array.
3. **QC**: Use the authors' final classifier labels, not `unit_quality` or hand-selected metric thresholds. No >2-Hz filter because that was specific to video-to-neural regression and would remove potentially informative decoder inputs.
4. **Observed-trial mapping**: `is_good_trials` is not always trial-table length. In nine NWBs it corresponds to only the trials listed in each unit's ragged `obs_intervals` (for example 160 observed of 480 behavioral trials). All good units within a session share exactly the same interval sequence. Map those intervals to trial rows by exact start/stop equality before applying the boolean mask.
5. **Valid recording windows**: After observation mapping there are 93,310 observed trials; all-unit validity removes 509; full -2.5/+1.5 neural coverage removes 15 edge trials, giving 92,786 candidate trials across 173 sessions. A final population-all-zero check removes any residual source stream truncation (one such trial was detected in the sample), avoiding treatment of unavailable ephys as zero firing; final full count will be measured in Step 9.
6. **Tongue visibility**: A 0.9 likelihood cutoff is justified by the source's sharply bimodal likelihood distribution (only 11.87% of a 1/100-frame sample >=0.9; invisible mass is near 0). This also implements the requested class 3, unlike the paper's continuous-regression mean imputation.
7. **Tongue percentile scope**: Compute percentiles across all visible frames in the source session, before trial sampling, exactly matching “over the session.” Boundaries follow the specification: equality at p40/p60 is class 1.
8. **Choice**: Use task semantics rather than raw first-lick timestamps; early/carryover sensor contacts explain the 0.34% disagreement and the paper defines choice jointly from instruction and correct/error outcome.
9. **Data types**: float32 neural/input arrays and int64 categorical outputs. Dense neural payload is projected at about 11.2 GiB; available disk/RAM are sufficient. Pickle protocol 5/HIGHEST_PROTOCOL supports large arrays.
10. **Ordering**: Sort file paths, subjects, and region strings to make output reproducible. `--sample` takes the first two usable sessions under the identical logic.

### Planned Sanity Checks
- [ ] For selected raw unit/trial/bin indices, independently count raw NWB spikes with boolean comparisons and require `np.allclose` to converted Hz.
- [ ] Independently calculate tone-relative bin centers and photostimulation center membership from raw event timestamps and require `np.allclose` to both converted input rows.
- [ ] Independently derive choice/outcome/early from raw trial rows and tongue class from raw nearest-frame y/likelihood and saved percentiles; require `np.allclose` to converted output.
- [ ] Assert 173 sessions, 28 subjects, 69,453 retained neurons, planned 92,786 valid trials, 80 bins, finite arrays, categorical ranges, and >=2 trials/session.
- [ ] Assert every spike count multiplied by 0.05 is integral within float tolerance and all rates are nonnegative.
- [ ] Assert every retained trial maps uniquely to an observation interval, all retained `is_good_trials` values are true, and every neural window lies within the common recording span.
- [ ] Assert photostimulation start/stop event counts and ordering match, and event count equals non-N/A trial photostimulation rows.
- [ ] Confirm conditional tongue classes 0/1/2 approximate 40/20/40 over all visible source frames and class 3 occurs only for low-confidence/missing sampled frames.
- [ ] Compare native/converted output distributions and paper statistics; explain expected differences caused by mandated retention of early/ignore/stimulated trials.
- [ ] Plot raster/rate alignment, tone-relative time, photostimulation intervals, corrected tongue y/thresholds/classes, and categorical trial outputs for up to two sessions.

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]
- Implemented `/app/convert_data.py` with the required positional output path and `--full` (default), `--sample`, and `--show-processing` modes.
- Uses `h5py` directly, maps ragged NWB observation intervals to trial rows, filters classifier-good units and invalid unit-trial periods, bins spikes, constructs both decoder inputs and all four outputs, validates shapes/ranges, stores detailed per-session provenance, and writes protocol-5 pickle output.
- Optional figures show raw spikes versus binned rates, tone-relative time, photostimulation state, tongue correction/visibility/percentiles/classes, and final categorical outputs.
- `python3 -m py_compile /app/convert_data.py` and CLI help completed successfully.

Code inefficiencies identified:
Per-unit HDF5 reads and Python loops over every trial/bin would cause excessive I/O and interpreter overhead. The target format itself requires ~11.2 GiB of float32 neural payload.

Code speedups added:
Read each session's concatenated spike vector once; use NumPy `searchsorted` against all trial edges for each unit; vectorize event sampling, tracking sampling, categorical construction, and validity masks; process one session at a time.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 834 |
| Neurons / session | [459, 375], mean 417 |
| Subjects | 1 (`sub-440956`) |
| Sessions / subject | 2 |
| Trials (total) | 527 |
| Trials / session | [368, 159] |
| Time bins | 80, all trials |
| Time from tone range | [-0.625, 5.722] s |
| Photostimulation range | [0, 1] |
| Choice distribution | [0.4516, 0.4080, 0.1404] left/right/no lick |
| Outcome distribution | [0.1404, 0.2979, 0.5617] ignore/miss/hit |
| Early lick distribution | [0.9431, 0.0569] no/yes |
| Tongue distribution | [0.0526, 0.0286, 0.0616, 0.8573] low/mid/high/not visible |

### Processing Plots Review
Reviewed both `processing_sub-440956_*.png` files. Raw spike ticks and 50-ms rates coincide; go is at 0; tone input is a unit-slope signed ramp crossing zero at the final tone; laser state is confined to pre-go intervals; tongue bins use the plotted session percentile thresholds and low-confidence bins are class 3. Session 1's plotted first trial has no visible tongue, while session 2 shows plausible post-go visible tracking and switching among classes 0–2. No temporal shift or discretization anomaly was seen.

Initial verification warned that session 2's final retained trial was all-zero across 375 units. Direct raw inspection showed every unit's spike stream ended at or before 1107.443 s, whereas the requested trial window was [1110.140, 1114.140) s despite its NWB observation flag being true. Added explicit population-all-zero exclusion, reran conversion, and removed that one trial. The repeated verifier reported: “Data format is valid, no errors or warnings.”

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| One sequential spike-vector read/session | Avoids hundreds of random HDF5 reads/session |
| `searchsorted` on all trial edges/unit | Avoids Python loops over trials and bins |
| Vectorized event/tracking/output construction | Keeps non-neural processing subsecond/session |

| Step | Time / Session | Estimated Total Time |
| Sample conversion computation | 1.05 s/session average | ~3 minutes for 173 sessions by count |
| Scale adjustment by dense neural payload (11.2 GiB full / 0.070 GiB sample) | Conservative multiplier | ~6 minutes including heterogeneous unit/trial counts and I/O |
| Pickle write | 0.07 s for 0.070 GiB | ~12 s at measured throughput; allow several minutes for filesystem variability |
| Total full conversion estimate | — | <10 minutes, below the 15-minute optimization threshold |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None after resolving the truncated all-zero trial described in Step 7

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Lick direction choice | 0.7305 | 0.6230 |
| Outcome | 0.7523 | 0.6556 |
| Early lick | 0.8353 | 0.7355 |
| Tongue y-position | 0.6624 | 0.5191 |

Training completed on CUDA (421 train / 106 validation trials). Loss decreased monotonically in the reported epochs from 15.7935 at epoch 1 to 0.5885 at epoch 200; test loss was 0.7314. All validation balanced accuracies exceed chance (0.3333, 0.3333, 0.5, and 0.25 respectively), supporting correct labels and temporal alignment.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 11.170 GiB (173 sessions)
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total neurons | 69,943 | Classifier-good only | 69,453 explicit good | 69,453 | Data exact; paper gap explained by missing-label session |
| Mean neurons/session | Not reported | — | 401.46 usable | 401.46 | Yes |
| Subjects | 28 | — | 28 | 28 | Yes |
| Sessions | 173 | All session groups with QC units | 174 NWBs; 173 usable | 173 | Yes |
| Trials (total) | Not reported | Analysis-specific filtering | 94,990 native; 94,370 in usable sessions | 90,363 valid retained | Expected validity filtering |
| Trials/session (mean; range) | 476; 130–785 after paper selection | Depends on regular-trial mask | 545.5 native in usable sessions | 522.33; 159–800 | Expected mandated category retention + validity filtering |
| Time from tone range | Not reported | Sample timing loaded | Protocol/replay dependent | [-1.525, 11.894] s | Plausible; exact source mapping |
| Photostimulation range | Binary trial condition | onset/offset aligned to go | paired absolute intervals | [0, 1] | Yes |
| Choice distribution | Not reported | lick direction | Native before validity: [0.428?, 0.423?, 0.149?] | [0.4283, 0.4223, 0.1495] | Consistent |
| Outcome distribution | 84% correct in selected control/no-early analysis | correctness labels | Native [ignore .1484, miss .1647, hit .6870] | [.1495, .1659, .6846] | Consistent after validity filtering; paper subset differs |
| Early distribution | Excluded in paper analyses | early-report field | Native yes=.1138 | [no .8842, yes .1158] | Consistent |
| Tongue distribution | No categorical reference | marker alignment at 3.4 ms | likelihood strongly bimodal | [.0621, .0317, .0651, .8410] | Plausible; visible conditional split ~39/20/41 |

Full conversion took 194.50 s (14.52 s pickle write), substantially below the 15-minute threshold. Trial accounting: 94,370 in sessions with good units; 93,310 explicitly observed; 92,801 after all-unit per-trial QC; 92,786 after full-window coverage; 90,363 after excluding 2,423 population-all-zero windows caused by spike-stream truncation. The verifier reported no errors or warnings, all 80-bin dimensions/ranges were correct, and region totals summed to 69,453.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Read `verification_full_out.txt`. Result: “Data format is valid, no errors or warnings”; dimensions, ranges, class names, region counts, and all 173 per-session summaries completed.
2. **Independent neural sanity checks**: `/app/check_conversion.py` loads raw NWBs directly and does not import conversion code. For sessions 0, 86, and 172 it maps the converted neuron back through raw `classification`, slices raw ragged spike timestamps, independently counts `[lower, upper)` spikes, divides by 0.05, and uses `np.allclose()` against converted rates. Checked bins 10, 39, and 79 plus explicit first/last bins; all passed.
3. **Independent input sanity checks**: For the same three spots, selected the last raw sample event before go and independently reconstructed all 80 tone-relative centers; independently evaluated all paired raw photostimulation intervals at the centers. `np.allclose()` passed for both rows in all checks.
4. **Independent output sanity checks**: Re-derived choice/outcome/early from raw NWB rows, independently repeated tongue confidence, velocity interpolation, p40/p60, nearest-frame, and class logic, and compared via `np.allclose()`. All four outputs passed. A further full raw audit compared choice/outcome/early for every one of 90,363 retained trials; all passed.
5. **Global neural checks**: Every trial was finite, nonnegative, not population-all-zero, shaped neuron x 80, and satisfied `rate * 0.05 == integer spike count` by `np.allclose()`.
6. **Global input/output checks**: Every tone-time row advanced by 0.05 s/bin; every laser value was binary; all output arrays were `(4,80)`; trial labels were constant over time; categorical ranges and accumulated counts matched the conversion log.
7. **Anatomy check**: For all units in the three raw spot sessions, full raw good-unit `anno_name` arrays exactly equaled strings recovered through `brain_region_idx`; global region indices accounted for all 69,453 units.
8. **Data loading comparison**: Converter reads native NWB HDF5 instead of reference MATLAB export but retrieves the corresponding units, spikes, trials/events, tracking, QC, and anatomy. Absolute NWB clocks replace the MATLAB arrays already relative to go; raw allclose tests validate equivalence.
9. **Neuron/trial filtering comparison**: Both use final classifier-good units. Reference “regular” trial exclusion is deliberately not used because it would delete mandated early/ignore/stimulation classes. Added NWB `obs_intervals`, `is_good_trials`, full-window, and truncated-stream checks because these expose source validity unavailable/unused in the reference MATLAB analysis.
10. **Temporal alignment/binning comparison**: Both align to go and use half-open spike intervals divided by width. Requested 50-ms adjacent bins override paper 40-ms/3.4-ms or 200-ms sliding variants. Raw checks at first, middle, and last bins found no off-by-one issue.
11. **Input/output construction comparison**: Tone is the final sample epoch preceding go after any replay; photo uses real event intervals; choice uses the paper's instruction-choice/outcome semantics. Tongue uses the reference side camera, 5-SD velocity correction, and session percentiles, but preserves low-confidence occlusion as required class 3 rather than regression mean-imputation.
12. **Statistics comparison**: 28 subjects and 173 usable sessions match the paper. Converted/data good units are 69,453; the paper's 69,943 differs by exactly 490, attributable to the one NWB with no exported classifier/CCF labels. Native good-unit proportion is 25.51% versus paper 25.9%, consistent with that missing export. Paper area-specific totals cannot be exactly reconstructed from the 293 NWB leaf annotations without the absent external classifier-family dictionaries; preserving native CCF labels avoids speculative remapping.
13. **Behavior statistics comparison**: Native usable sessions have weighted control/no-early/responded accuracy 81.83% (session mean 81.01%, range 53.78–97.01%) versus paper-selected mean 84%, range 65–99%. Lower raw sessions confirm that paper behavioral selection was analysis-specific. Required conversion retains all valid behavioral categories, so its overall hit fraction 68.46% is not expected to equal selected control performance.
14. **Edge cases**: Verified nine partial trial tables map through exact start/stop pairs; one zero-good session is skipped; four sessions' false unit-trial masks are enforced; 15 incomplete recording-edge windows are removed; 2,423 residual population-zero truncated windows are removed; uncovered video centers become tongue class 3; p40/p60 equality is class 1; first/last spike bins use the same half-open rule.

### Issues Found and Resolved
- **Malformed-looking validity dimensions**: Nine `is_good_trials` arrays are shorter than the behavioral table. Found that their columns correspond exactly to ragged `obs_intervals`; mapped intervals rather than padding/truncating.
- **All-zero neural warning in initial sample**: Raw data proved all 375 unit streams ended before the flagged window. Added population-all-zero truncation filtering, regenerated sample/full outputs, and reran validation/checks successfully.
- **Paper/data count mismatch**: Did not fabricate 490 labels or threshold QC metrics; excluded the unlabeled/unanatomized session as required by the authors' classifier/histology policy and documented the difference.
- **No issues found in the final independent review**: `cache/critical_review1_out.txt` ends with `ALL CRITICAL REVIEW CHECKS PASSED`.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes; 19.9268 (epoch 1) → 2.0530 (epoch 50) → 0.9643 (epoch 100) → 0.6730 (epoch 200). Test loss 0.6681.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Lick direction choice | 0.7098 | 0.6789 | Chance 0.3333 |
| Outcome | 0.7031 | 0.6614 | Chance 0.3333 |
| Early lick | 0.7807 | 0.7484 | Chance 0.5000 |
| Tongue y-position | 0.6764 | 0.6166 | Chance 0.2500 |

Full CUDA run completed all 200 epochs on 72,237 training / 18,126 validation trials and finished successfully. `sample_trials.png` shows sensible heterogeneous firing rates, unit-slope tone time, binary pre-go photostimulation, fixed trial labels, and time-varying tongue classes. `predictions.png` shows decoded trial labels and post-go tongue visibility/class transitions broadly tracking targets.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Papers |
| Lick direction choice | Validation balanced accuracy 0.6789 (2.037x chance) | Data paper population choice decoder is >0.5 through sample/delay and approaches high accuracy near go, but gives no single aggregate scalar. Method paper video→choice AUC: 0.51±0.06 pre-sample, 0.66±0.12 sample/delay, 0.99±0.01 late response; different modality/metric. |
| Outcome | 0.6614 (1.984x chance) | No matching outcome decoder accuracy reported. |
| Early lick | 0.7484 (1.497x chance) | No matching early-lick decoder reported. |
| Tongue y-position | 0.6166 (2.466x chance) | No neural→categorical-tongue decoder reported. Method paper's marker/full-video choice AUC in response was 0.88±0.01 / 0.96±0.00, a different prediction direction and target. |

All outputs are above chance. Choice, outcome, and tongue exceed 1.5x chance. Early lick is 0.7484 versus the 0.7500 screen (difference 0.0016, or 0.2 percentage points). This borderline result was investigated rather than dismissed:

- Raw verification: Critical Review 1 already compared all 90,363 early labels to raw NWB. Critical Review 2 additionally checked concrete converted/source trials `(session, converted trial, raw trial) = (0,1,1), (86,1,1), (172,1,1)`, obtaining `early`, `no early`, `no early` exactly.
- Variation: 79,901 no-early and 10,462 early trials (88.42%/11.58%), so the minority is substantial and balanced accuracy prevents majority dominance.
- Alignment: `early_lick_alignment.png` shows go-aligned population mean activity and constant per-trial labels across the full required window. Input time advances exactly 50 ms and photo remains binary.
- Filtering/processing: classifier QC, valid trial periods, full-window checks, raw bin allclose checks, and label logic all passed. Early trials were retained because they are a mandated output, unlike reference analysis exclusions.
- Generalization: train/validation ratio is 1.043, so neither serious overfitting nor leakage is evident. Changing correct raw labels or selectively dropping difficult early trials merely to cross 0.75 would bias the dataset.

Every explicitly numerical decoding performance located in the papers was reviewed. In addition to method-paper values above, data-paper figure examples show single-neuron AUC changes 0.82→0.52 and 0.91→0.87 after subtracting video-predicted activity; these illustrative single-unit AUCs are not comparable to the supplied multi-session balanced-accuracy decoder. The data paper's main population result is a time curve rather than a scalar.

Train/validation gap audit:

| Output | Train / validation ratio |
|--------|--------------------------|
| Choice | 1.0455 |
| Outcome | 1.0630 |
| Early lick | 1.0432 |
| Tongue | 1.0970 |

All are far below the 1.5x concern threshold. Visual review of `sample_trials.png`, `predictions.png`, and `early_lick_alignment.png` found no temporal or categorical artifact.

### Issues Found and Resolved
- **Borderline early-lick 1.5x-chance screen**: Exhaustive/raw checks found no conversion defect; result is 0.2 percentage points below the heuristic threshold with a small train-validation gap. No data change is scientifically justified.
- **No below-chance or large-gap outputs**: No iteration was required. `/app/cache/critical_review2_out.txt` ends with `CRITICAL REVIEW 2 CHECKS PASSED`.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with dataset description, loading example, format, decisions, reproduction commands, statistics, and decoder results
- [x] cache/ folder created with `README_CACHE.md`
- [x] Investigation scripts/logs and generated bytecode moved to cache; required deliverables and user-facing diagnostics retained at project root
- [x] All workflow statuses complete and required output files present
