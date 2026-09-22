# Dataset Conversion Notes

## Overview
- **Dataset**: International Brain Laboratory brain-wide map data (local project copy)
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
- `dataarchitecture.pdf`
- `datapaper.pdf`
- `decoder.py`
- `docker-compose.yaml`
- `methodpaper.pdf`
- `methods.txt`
- `stage_cache.sh`
- `train_decoder.py`

Environment verified: Python 3.13.15; NumPy 2.3.5; PyTorch 2.6.0+cu124. The required checkpoint `ls -la /app/CONVERSION_NOTES.md` succeeded.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `prepare_data` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Resolves all probes for an EID, loads and merges their spikes/clusters, loads trials, applies the trial mask, and prepares cluster metadata. |
| `load_spiking_data` | same | LOADING/CURATION | Uses `SpikeSortingLoader`; merges spike sorting with channels. Default `qc=None` returns all clusters; `qc=1` would retain `label >= 1`, but caching calls the default. |
| `merge_probes` | same | PROCESSING | Re-indexes cluster IDs across probes, concatenates clusters/spikes, stable-sorts spikes by time. |
| `load_trials_and_mask` | same | CURATION | Requires non-NaN stimulus, choice, feedback, prior, first movement, and feedback type; excludes no-choice, RT < 0.08 s or > 2 s, and (as called) trial duration > 10 s. Keeps 0.5-prior trials. |
| `bin_spiking_data` / `get_spike_data_per_interval` | same | PROCESSING | Counts spikes in half-open aligned intervals; caching parameters are stimulus onset, [-0.5, 1.5) s, 0.02 s bins (100 bins). |
| `load_target_behavior` | same | LOADING | Loads absolute wheel velocity as wheel speed and left whisker motion energy, falling back to right view. |
| `get_behavior_per_interval` / `bin_behaviors` | same | PROCESSING | Extracts behavior around trial events and linearly interpolates to bin-end times from -0.48 through +1.50 s (100 samples); also supplies per-trial choice, prior (`block`), reward, contrast. |
| `align_spike_behavior` | same | CURATION | Applies the reference trial mask and drops corresponding neural/behavior trials; reshapes every target consistently. |
| `list_brain_regions` / `select_brain_regions` | same | PROCESSING | Maps native acronyms to the Beryl ontology and includes all mapped regions in the default all-region analysis. |
| `create_dataset` | `code/code_zhang2025/src/utils/dataset_utils.py` | PROCESSING | Stores trial-by-time-by-neuron spike counts sparsely along with behavior and cluster metadata. |
| `standardize_spike_data` | `code/code_zhang2025/src/utils/data_loader_utils.py` | PROCESSING | Decoder-side standardization by time bin across trials; this is training preprocessing, not part of cached data. |

### Notes
- Reference caching parameters in `0_data_caching.py`: `align_time='stimOn_times'`, `time_window=(-0.5, 1.5)`, `interval_len=2`, `binsize=0.02`; targets are choice, reward, block, wheel speed, and whisker motion energy.
- Electrophysiology, not imaging: delta-F/F is inapplicable.
- The caching call intentionally does not pass a cluster QC threshold, so all available sorted clusters are binned; `good_clusters=(label >= 1)` is retained only as metadata. This is distinct from some BWM analyses that select good units.
- Spike outputs have source orientation trial × time × neuron and are transposed to target neuron × time.
- `align_spike_behavior` uses Python list truthiness in a way that makes the explicit trial mask decisive; behavior availability is allowed to contain NaNs because `allow_nans=True`. Conversion must handle any residual non-finite behavior explicitly for categorical outputs.
- Data are randomly partitioned only for the authors' model cache. The requested target retains the full ordered trial set; downstream validation performs its own split.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`data/one_cache` is an IBL ONE cache. It contains three Parquet cache-table snapshots (`2022_Q4_IBL_et_al_BWM`, `2025_Q3_IBL_et_al_BWM`, and `Brainwidemap`) plus lab-organized data roots. The broad `Brainwidemap` table indexes 480 sessions. Native session paths are `lab/Subjects/<subject>/<date>/<number>/`.

- Trial data: `alf/#revision#/_ibl_trials.table.pqt`, one row per trial. Columns are `goCue_times`, `response_times`, `choice`, `stimOn_times`, `contrastLeft`, `contrastRight`, `probabilityLeft`, `feedback_times`, `feedbackType`, `rewardVolume`, `firstMovement_times`, `intervals_0`, `intervals_1`; numeric columns are float64.
- Neural data: `alf/probeXX/pykilosort/#2024-05-06#/spikes.{times,clusters,amps,depths}.npy`, cluster arrays/metrics, and channel arrays. There are 701 probe collections in 461 unique sessions. `clusters.metrics.pqt` exposes 22 quality/summary columns including `label`, `firing_rate`, `presence_ratio`, contamination measures, and spike count. Cluster channel indices plus `channels.brainLocationIds_ccf_2017.npy` provide anatomical labels.
- Wheel: `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`, both float64 and matched in length, for all 461 neural/trial sessions. Speed is not stored directly and must be computed from position/time as in the reference loader.
- Whisker motion energy: `leftCamera.ROIMotionEnergy.npy` with `_ibl_leftCamera.times.npy` for 439 sessions; right equivalents for 422. Left is the reference-preferred view and right is the fallback.
- The 466 trial-table files represent 461 unique sessions because five sessions have multiple cached revisions; conversion must select the ONE/default revision rather than double count.
- Physical data volume under the lab roots is approximately 571 GB, dominated by 21.15 billion spike events; memory-mapped loading and session-wise processing are required.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 622,377 sorted clusters across all 701 cached probes (all-quality inventory) |
| Neurons / session | mean 1,350.06; range 135–3,140 (all-quality inventory) |
| Subjects | 141 with neural + trial data |
| Sessions / subject | mean 3.27; range 1–13 |
| Trials (total) | 297,505 raw trials; 196,727 pass the reference code mask when applied to the available tables |
| Trials / session | mean 645.35; range 401–1,525 raw |

Additional native checks: 294,411 trials have all reference-required fields; choice counts are left (`-1`) 146,529, no-choice (`0`) 1,152, right (`+1`) 149,824. Prior counts are 0.2: 124,194; 0.5: 41,490; 0.8: 131,821. These are pre-curation counts.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 621,733 all sorted units; 75,708 well-isolated | Data paper: “621,733 units…75,708 well-isolated neurons.” | 
| Neurons / session | 889 units/probe; 108 well-isolated/probe; method paper reports ~676/session (roughly 300 to >2,000) for its analyzed subset | Data and method papers |
| Subjects | 139 (94 male, 45 female) | Data paper |
| Sessions / subject | 459 sessions / 139 mice = 3.30 mean | Data paper |
| Trials (total) | Not stated as a single total | — |
| Trials / session | Release required ≥250; main analyses retained ≥400; first 90 are unbiased | Data paper |
| Neural data time bin | Requested/common cache: 20 ms over a 2 s window (100 bins); method-paper static-variable analysis text also describes 50 ms task-specific windows | Method paper and supplied caching code |
| Behavior data time bin | Source camera ~60 Hz (left) or 150 Hz (right); reference caching code linearly resamples behavior to 20 ms bins | Method paper/code |
| Reward/correct rate | 81.4 ± 0.4% correct after training | Data paper |
| Biased block length | 20–100 trials; empirical mean 51 | Data paper |
| Trial RT exclusion | first movement − stimulus onset outside 0.08–2.00 s | Data paper |


### Processing Details
- The task here mandates stimulus-onset alignment for all variables. The compatible common window in the supplied caching code is `[-0.5, +1.5)` s with 20 ms spike-count bins (100 time bins). This exactly matches the method paper's general 2 s/100-bin tensor definition and its choice window.
- The method paper otherwise used target-specific windows: choice, stimulus onset −0.5 to +1.5 s; prior, stimulus onset −0.6 to −0.1 s with 50 ms bins; dynamic wheel/whisker, first movement 0 to +1 s with 20 ms bins. Those target-specific alignments cannot coexist in the requested one-tensor, stimulus-aligned target and are superseded by the explicit Decoder Task.
- The supplied caching code is the applicable reconciliation: it creates every requested target in the same stimulus-aligned −0.5 to +1.5 s, 20 ms representation.
- Motion energy is mean absolute adjacent-frame difference inside a whisker-pad bounding box anchored between DLC nose and eye locations. Wheel speed is absolute wheel velocity.
- ONE/ALF semantics: `*.times` are seconds relative to session onset; object attributes have matched row counts; `spikes.clusters` indexes cluster rows. Revision-aware/default dataset selection prevents duplicates.

### Curation Steps

**Neuron curation rules**:
The data paper distinguishes all Kilosort units from well-isolated neurons. Well-isolated units pass amplitude >50 µV, noise cutoff <20 µV, and refractory-period-violation criteria, represented by the released cluster `label >= 1`. However, the methods-paper cache explicitly calls `load_spiking_data(..., qc=None)` and repeatedly says it uses “all neurons”; therefore the decoder conversion should retain all clusters from the exact 699-insertion release, while recording quality metadata. This choice matches the directly supplied decoder code rather than the data paper's later region-specific analyses.

**Trial curation rules**:
Exclude trials missing choice, probabilityLeft, feedbackType, feedback time, stimulus-on time, or first-movement time; exclude RT outside 0.08–2.00 s. The supplied code additionally excludes no-choice and trial duration >10 s. It does not exclude the initial 0.5-prior block. Sessions/probes are taken from the curated release (behavioral, hardware, recording, histology, and alignment QC already applied).

### Decoders Trained
| Decoded variable | Accuracy |
| Choice | Paper evaluates accuracy/AUC; Figure 5 region-specific accuracy is approximately 0.60–0.82 across PO, LP, DG, CA1, VISa and models (chance 0.5). An example BMM-HMM improves AUC from 0.66 to 0.72 (oracle 0.79). |
| Prior | Continuous in paper: Pearson correlation, broadly ~0.3–0.7 in Figure 5; example multi-session RRR 0.76 versus ridge 0.54. Not directly comparable to required 3-class accuracy. |
| Wheel speed | Continuous in paper: R² broadly ~0.39–0.53 in Figure 5; example multi-session RRR 0.80 versus ridge 0.52. Not directly comparable to required 3-class accuracy. |
| Whisker motion energy | Continuous in paper: R² broadly ~0.38–0.50 in Figure 5; example multi-session RRR 0.86 versus ridge 0.66. Not directly comparable to required 3-class accuracy. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Release size | `bwm_release.csv` has 699 PIDs, 459 EIDs, 139 subjects | Broad live cache has 701 probes/461 sessions/141 subjects and 622,377 clusters | Frozen data paper: 699 probes, 459 sessions, 139 mice, 621,733 units | Use `bwm_release.csv` as the authoritative freeze. The two extras are KS074 (370 units) and KS075 (274 units), exactly explaining the +644-unit discrepancy. |
| Unit quality | Caching calls `load_spiking_data` with `qc=None`; all clusters; quality labels saved | Frozen 699 probes contain exactly 621,733 clusters; exactly 75,708 have `label >= 1` | Data paper reports both exact totals; region-specific analyses use well-isolated neurons, methods-paper decoder says all neurons | For this decoder conversion, retain all 621,733 units as the supplied decoder code does. Do not silently apply the data-paper analysis-only good-unit filter. |
| Sessions analyzed | BWM freeze provides 459; caching wrapper may skip a session on load error | 14 frozen sessions have no paired whisker motion energy/timestamps; 445 have a valid left-preferred/right-fallback stream | Data paper release is 459; method paper analyzes 433 | Begin from all 459 frozen sessions, then enforce requested-stream validity. Exclude sessions lacking the mandatory whisker output and trials outside valid stream coverage. The paper's 433 is an analysis subset whose complete membership/filter is not specified; it is not reproducible to discard an arbitrary 12 additional sessions. |
| Brain regions | Code maps every unit to Beryl and selects all by default | 580 native Allen acronyms map to 281 Beryl labels including `root` and `void` | Method paper reports 270 analyzed regions; data paper additionally restricts reportable gray-matter regions with coverage | Preserve every unit's Beryl label because target format requires a region per retained neuron and the cache's all-region decoder does not apply reportability filters. The 270 figure is an analyzed-region count, not a unit-loading rule. |
| Trial counts | Code applies event completeness, 0.08–2 s RT, no-choice exclusion, and ≤10 s trial duration | Frozen release: 296,090 raw; exactly 195,781 pass this mask | Paper explicitly specifies missing-event and RT filtering; code adds no-choice and duration | Apply the supplied code mask exactly, then add required behavior-coverage validity. |
| Window/binning | Cache script: stimulus onset, −0.5 to +1.5 s, 20 ms | Raw streams support this common grid | Paper uses target-specific windows, but also defines 2 s/100-bin tensors | Explicit Decoder Task requires stimulus alignment for every output, so use the supplied cache script's common stimulus-aligned grid. |
| Camera choice | Code prefers left whisker motion energy and falls back to right | 437 frozen sessions have paired left data; 8 additional sessions have only usable right; 14 have neither | Left is 60 Hz and right is 150 Hz; paper defines the same whisker ROI measure | Use left-first/right-fallback exactly as code; record view in session metadata; omit sessions with neither. |
| ONE revisions | ONE APIs choose default revisions | Five trial-table duplicates exist physically; default/latest release table is `#2025-03-03#` | Architecture paper requires versioned datasets and stable object/attribute semantics | Resolve one dataset per attribute deterministically, favoring the current revision, and never treat revisions as separate sessions. |

Final cross-source understanding: the source is frozen-release Neuropixels spike times plus synchronized trial, wheel, and camera streams. Sessions combine simultaneous probes; trials are stimulus-aligned and filtered by reference event/RT/no-choice/duration criteria; all sorted units are retained; spike counts are 20 ms counts; wheel position is resampled to 1 kHz and Butterworth-filtered before taking absolute velocity; motion energy is linearly resampled; mandatory-stream coverage adds a documented downstream validity filter.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spikes.times`, `spikes.clusters` across all frozen-release probes | `neural` | Count spikes in 100 half-open 20 ms bins spanning stimulus onset + `[-0.5, 1.5)`; concatenate probes on neuron axis; transpose to neuron × time; float32 | `merge_probes`, `bin_spiking_data`, `get_spike_data_per_interval` | All sorted clusters retained in cluster order; simultaneous probes form one session. |
| Bin ending times relative to `stimOn_times` | `input[0, :]` | `[-0.48, -0.46, ..., 1.50]` seconds, float32 | `get_behavior_per_interval` | Continuous time-varying input. These are bin-ending labels for neural intervals `[-0.50,-0.48),...`. |
| Raw trial index and `probabilityLeft` run boundaries | `input[1, :]` | Zero-based count within the original experimental block, reset when `probabilityLeft` changes; repeat across 100 bins | task design | Continuous per-trial input. Filtering does not compress block position. |
| `trials.choice` | `output[0, :]` | source −1 (left) → 0; source +1 (right) → 1; repeat across time | `bin_behaviors` plus target requirement | No-choice 0 source trials are already excluded. |
| `trials.probabilityLeft` | `output[1, :]` | 0.2 → 0, 0.5 → 1, 0.8 → 2; repeat across time | `bin_behaviors` (`block`) plus target requirement | Exact mapping required by Decoder Task. |
| `_ibl_wheel.position`, `_ibl_wheel.timestamps` | `output[2, :]` | Uniform 1 kHz position interpolation; order-8, 20 Hz Butterworth zero-phase filter; absolute differentiated velocity; linearly sample at bin ends; discretize by global tertiles | `SessionLoader.load_wheel`, `interpolate_position`, `velocity_filtered`, `load_target_behavior` | Time-varying classes low/medium/high. |
| `{left,right}Camera.ROIMotionEnergy` and matching camera times | `output[3, :]` | Prefer left, fall back right; linearly sample at bin ends; discretize by global tertiles | `load_target_behavior`, `get_behavior_per_interval` | Time-varying classes low/medium/high; sessions lacking either paired stream excluded. |
| Frozen CSV subject | `subjects`, `subject_idx` | Sorted unique subject IDs and integer lookup | `bwm_release.csv` | Only subjects represented by retained sessions remain. |
| cluster channel CCF ID | `brain_regions`, `brain_region_idx` | cluster → channel → Allen ID → native acronym → Beryl acronym; sorted global label list | `BrainRegions.acronym2acronym(..., mapping='Beryl')` | Includes `root`/`void` for retained all-unit decoder rather than fabricating anatomy. |

### Key Decisions
1. **Release freeze**: Use exactly the 699 insertions in `bwm_release.csv`; this reproduces every headline data-paper unit statistic and removes the two later cache additions.
2. **Session validity**: Start from 459 EIDs, exclude a session only when a required wheel/motion stream is unavailable or fewer than two fully valid aligned trials remain. This satisfies the target and avoids undocumented arbitrary reduction to the method paper's 433-session analysis subset.
3. **Trial validity**: Apply the reference missing-event, RT, no-choice, and duration mask. Then require both continuous streams to cover the full two-second window with boundary samples no more than one 20 ms bin away; drop individual invalid trials while preserving their original indices/block counts.
4. **All units**: Retain every Kilosort cluster because this exactly matches the supplied methods-paper cache (`qc=None`) and the method paper's “all neurons” decoder. Quality labels remain described in metadata.
5. **Common temporal grid**: The explicit stimulus-onset task overrides the paper's target-specific alignments. The cache script's −0.5 to +1.5 s, 20 ms common grid is the closest direct reference implementation.
6. **Bin-end coordinates**: Use −0.48…+1.50 s because the reference behavior interpolator labels each count bin by its right edge. Neural counts remain half-open, avoiding double counting on edges.
7. **Mixed static/dynamic fields**: A single trial array cannot mix 1D and 2D rows. Therefore input is 2×100 and output is 4×100; per-trial variables are constant repeats, preserving their per-trial semantics while satisfying shape consistency.
8. **Behavior discretization**: Use deterministic global 1/3 and 2/3 quantiles over all retained, aligned samples. Global thresholds keep class definitions physically consistent across sessions and approximately balance classes. Store exact thresholds in metadata and assert they are distinct and every class is represented.
9. **Data types**: Store spike counts and inputs as float32 (validator-native and exact for these small integer counts) and outputs as int8 categorical labels.
10. **Memory/I/O**: Load spike arrays with memory mapping and process session-wise. Use search-sorted per-trial slices plus `np.bincount`, which is exactly equivalent to reference counting without scanning every session-wide spike array once per trial.

### Planned Sanity Checks
- [x] Frozen list reproduces 139 subjects, 459 sessions, 699 probes, 621,733 clusters, and 75,708 `label >= 1` units before mandatory-output filtering.
- [x] Direct raw trial-table mask count is 195,781 before continuous-stream coverage filtering.
- [x] For at least three trials, independently compute spike counts from original arrays and require `np.allclose` to converted neural values.
- [x] For at least three trials, independently reconstruct bin-ending time and block counter from raw tables and require `np.allclose` to converted inputs.
- [x] For at least three trials, independently map choice/prior and interpolate raw behavior, apply saved thresholds, and require `np.allclose` to converted outputs.
- [x] Every session has ≥2 trials; all neural/input/output time axes are 100; all arrays are finite; neural counts are nonnegative integer-valued floats.
- [x] Choice classes are both populated; prior classes are exactly {0,1,2}; continuous outputs are exactly {0,1,2}, near tertile balance globally.
- [x] Plotted raw continuous wheel/motion traces, resampled traces, thresholds/classes, inputs, and neural raster for two sessions; visual inspection passed.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py`. It supports the required command, default full mode, `--full`, `--sample`, and `--show-processing`, plus an optional `--workers`. Compilation and CLI-help smoke tests passed.

Implementation details: frozen-release and exact cluster-count assertions; revision-aware file selection; reference trial mask; original-sequence block counter; reference 1 kHz/20 Hz/order-8 wheel filter; left-first/right-fallback motion stream; reference boundary-coverage tests and bin-end interpolation/extrapolation; global tertiles; memory-mapped spike loading; Beryl region mapping; direct raw neural spot check; full internal shape/type/finite/class checks; per-session timing; diagnostic plots.

Code inefficiencies identified:
The reference `get_spike_data_per_interval` applies a full-session Boolean comparison separately for every trial, which is prohibitively expensive for 21.15 billion source spike events. Dense target storage is also intrinsically large.

Code speedups added:
Binary-search each trial boundary in sorted memory-mapped spike times, process only spikes inside retained windows, count flattened cluster×bin indices with compiled `np.bincount`, process probes without concatenating session-wide spike arrays, and parallelize independent sessions with a bounded thread pool. Behavior is processed in a first lightweight pass so global thresholds are known before assembling categorical outputs.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 2,626 session-neurons |
| Neurons / session | 898, 1,728 (mean 1,313) |
| Subjects | 1 (NYU-11) |
| Sessions / subject | 2 |
| Trials (total) | 651 |
| Trials / session | 407, 244 |
| Time input range | [-0.48, 1.50] s (verifier rounds to [-0.5, 1.5]) |
| Block-trial input range | [0, 89] |
| Choice distribution | [0.4823, 0.5177] |
| Prior distribution | [0.4777, 0.1613, 0.3610] |
| Wheel-speed class distribution | [0.3333, 0.3333, 0.3333] |
| Whisker class distribution | [0.3333, 0.3333, 0.3333] |
| Neural mean / nonzero fraction / max | 0.13317 / 0.11031 / 17 spikes per 20 ms |
| Sample pickle size | 315,704,539 bytes (0.294 GiB) |

### Processing Plots Review
Both required `processing_<eid>.png` plots were inspected. Trial-filter bars agree with logged counts. Wheel position, Butterworth-filtered speed, and 20 ms samples are synchronized to stimulus onset; camera samples overlay raw motion energy; categorical steps change exactly at saved tertiles; spike rasters use the same −0.5 to +1.5 s axis; time and block inputs are correct. No temporal offset, clipping, missing classes, or anomalous raster artifact was found. The first plotted retained trial correctly has block position 9 because excluded raw trials are not allowed to compress the experimental block counter.

`verification_sample_out.txt` reports: valid format, no errors, no warnings; every trial has T=100; all four outputs have their full declared class range; brain-region indices and subject/session mappings are valid.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Search-sorted spike slices + `np.bincount` instead of full spike-array scan per trial | Makes full conversion tractable; work scales with spikes in retained windows. |
| Memory mapping and per-probe processing | Avoids loading/merging all session spikes and reduces peak memory. |
| Two concurrent sample workers; up to eight for full run | Sample spike stages overlapped (1.44 s and 2.54 s session work). |

| Step | Time / Session | Estimated Total Time |
| Frozen inventory | 14.0 s fixed | 14 s |
| Behavior alignment/filtering | 0.19–0.20 s in sample | ~1.5 min for 459 candidates (sequential) |
| Spike binning | 3.98 worker-s for 0.787 million trial×neurons | ~2.8–4 min with 8 workers after scaling to the release |
| Validation + ~95–100 GiB pickle write | sample write 0.24 s; size-scaled with overhead | ~1–3 min |
| **Total** | sample 16.96 s (inventory dominates) | **~5–8 min** |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

Loss decreased monotonically from 1.962143 at epoch 1 to 0.597178 at epoch 200; test loss was 0.649341. GPU training completed successfully.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Choice | 0.7105 | 0.6184 |
| Prior probability left | 0.8286 | 0.7834 |
| Wheel speed tertile | 0.6775 | 0.6417 |
| Whisker motion energy tertile | 0.7055 | 0.6700 |

All validation scores exceed uniform chance (0.5 choice; 0.3333 other outputs). The train/validation gaps are modest and do not indicate leakage or severe overfitting.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 98.796 GiB (99 GiB as reported by `ls -lh`)
- `conversion_full_out.txt`: created; exact full command completed in 455.48 s
- `verification_full_out.txt`: created; verifier exited 0 and printed `Data verification complete.`

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Session-neuron sum | 621,733 units in complete freeze | Loads all clusters, `qc=None` | 621,733 in 459 frozen sessions | 599,865 in 444 retained sessions | Yes after 15 mandatory stream exclusions |
| Mean neurons/session | 1,354.5 over all 459 | All clusters per probe | 1,354.5 | 1,351.05 | Yes; excluded sessions have a similar distribution |
| Subjects | 139 | Frozen release list | 139 | 136 | Yes after stream exclusions |
| Sessions | 459 | 433 analysis sessions in methods paper | 459 | 444 | Yes: 14 lacked paired motion streams; one had <2 covered valid trials |
| Trials (total) | Not stated as one post-curation value | Reference mask gives 195,781 over complete freeze | 296,090 raw; 195,781 reference-valid | 188,925 | Yes after stream coverage/session exclusions |
| Trials/session (mean) | N/A | Reference-valid mean 426.54 over 459 | 426.54 | 425.51 | Consistent |
| Time input range | [-0.5, 1.5] window | 20-ms grid on that window | N/A | [-0.5, 1.5] (rounded verifier display; stored centers -0.48..1.50) | Yes |
| Trial-in-block range | N/A | Derived from probability-left changes | [0, 98] retained | [0, 98] | Yes |
| Choice fractions | N/A | -1/+1 retained choices | N/A | [0.492116, 0.507884] | Plausible and balanced |
| Prior fractions | Typical biased blocks plus 0.5 initialization | Probability-left values 0.2/0.5/0.8 | N/A | [0.417435, 0.140590, 0.441974] | Plausible |
| Wheel tertile fractions | Task-required discretization | N/A | Continuous | [1/3, 1/3, 1/3] | Exact by global quantiles |
| Whisker tertile fractions | Task-required discretization | N/A | Continuous | [0.333333, 0.33333328, 0.33333339] | Exact up to ties/count indivisibility |

The verifier found no errors. It emitted three warnings for all-zero neural windows in
session index 392, retained-trial indices 241--243. These are confined to three of
188,925 trials and are investigated explicitly in Step 10 rather than silently removed.
Spot checks of the first, middle, and final converted sessions confirmed 100 time bins,
matching neuron-region lengths, valid categorical ranges, and at least two trials.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Full verifier log**: Read all of `verification_full_out.txt`. There were no
   errors and verification completed. The only warnings were three all-zero neural
   trials in session index 392 (converted trials 241--243).
2. **Independent original-file `np.allclose` checks**: `sanity_checks.py` does not
   import conversion code. It loaded raw Parquet/NPY files and independently checked
   one trial each from the first, middle, and final converted sessions. It tested raw
   spike counts at bin indices 0, 10, and 99 (counts 2, 1, and 1), both input rows,
   raw choice/prior mappings, reference wheel filtering, raw motion interpolation,
   and both saved global thresholds. Every neural/input/output comparison passed.
   Exact tuples are recorded in `sanity_checks_out.txt`.
3. **Warning audit against raw files**: Session 392 is EID
   `8c2f7f4d-7346-42a4-a715-4d37a5208535`. Direct raw counts across its sole probe
   are exactly zero for all three warned windows. The spike recording ends at
   1779.359643 s, whereas the warned stimulus onsets are 1789.999367, 1796.482300,
   and 1799.791400 s. Thus these zeros are source recording coverage, not a binning
   bug. The reference counter also returns zeros and defines no neural-coverage trial
   filter, so the three trials remain; the warning cannot be removed without changing
   reference curation.
4. **Global invariants/edges**: Rechecked all 444 sessions: matched trial list lengths;
   neuron/region lengths; 100-bin time vectors; static choice/prior rows; strictly
   increasing source indices; at least two trials; and exact first/interior/last-bin
   raw counts. All passed. The half-open `[start,end)` convention was specifically
   checked at bin 0 and bin 99.
5. **Key statistics**: Independently recomputed the frozen raw reward fraction as
   242,628/296,090 = 81.944%, close to the paper's 81.4 +/- 0.4%. Applying the exact
   reference RT/event mask raises it to 167,913/195,781 = 85.766%; stream-valid
   converted trials are 161,974/188,925 = 85.735%, showing the stream filter does not
   distort accuracy. Raw probability-left run length is mean 51.964 (paper: 51) and
   median 47; median initial unbiased run is exactly 90 trials (paper: first 90).
   Frozen inventory totals exactly match 139 mice, 459 sessions, 699 probes, 621,733
   clusters, and 75,708 good clusters. Post-validity reductions are fully accounted
   for by the 15 logged session exclusions and trial coverage masks.

### Reference Code Comparison

| Stage | Reference | Conversion | Comparison/result |
|-------|-----------|------------|-------------------|
| Loading | `ibl_data_utils.py:27,727`; ONE `SpikeSortingLoader` | `convert_data.py:110,207,292` | Same frozen PID/EID list and revisioned ALF attributes; direct files avoid network/API ambiguity. Probe concatenation preserves CSV/probe and cluster-row order. |
| Neuron/trial filtering | `load_spiking_data(qc=None)`; `load_trials_and_mask` at line 123 | `reference_trial_mask` line 150; all cluster headers loaded | Exact event, RT 0.08--2 s, no-choice, and >10 s-duration logic; no neuron QC, matching cache. Added only mandatory finite full-window wheel/whisker coverage. |
| Alignment | cache parameters plus `get_behavior_per_interval:507` | constants and `coverage_mask`/`interval_interpolate:182,193` | Same `stimOn_times`, -0.5 to +1.5 s window and right-edge behavior grid. Final-point extrapolation reproduces reference exclusion of the interval end. |
| Binning | `get_spike_data_per_interval:244`, `bin_spiking_data:313` | `bin_spikes_for_session:292` | Same 100 half-open 20-ms spike-count bins; `searchsorted` + `bincount` is an exact vectorized equivalent. |
| Input construction | No paper decoder analogue for task-added inputs | `trial_number_in_block:162`, `assemble_data:397` | Explicit task requirement: continuous bin-ending time and zero-based position within the unfiltered probability-left run. |
| Output construction | `load_target_behavior:388`, `get_behavior_per_interval:507`, `align_spike_behavior:772` | `prepare_behavior:207`, `assemble_data:397` | Same choice/prior source, exact reference wheel functions, left-first/right-fallback motion, interpolation and trial order. Only global tertile categorization and static-row repetition are task-required additions. |

### Issues Found and Resolved
- **Three zero-neural warnings**: Proven to be faithful representations of a raw
  recording that ended before the final three behavioral trials. No data edit made.
- **Initial weak neural audit points were zero**: Strengthened the audit to choose an
  active neuron independently in bins 0, 10, and 99; raw counts 2/1/1 all matched.
- **Reward-rate apparent difference after filtering**: Recomputed both stages. The
  unfiltered frozen release reproduces the paper statistic; RT/event filtering
  appropriately enriches correct trials, and the added stream-validity filter changes
  it by only 0.031 percentage points.
- **No conversion mismatch found**: Therefore no full-conversion rerun was necessary.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. The required 200-epoch CUDA run completed without OOM or
  retry. Training loss decreased monotonically from 8.509975 (epoch 1) to 1.914755
  (epoch 200); final held-out loss was 0.851791. The script exited 0 and printed
  `train_decoder.py finished successfully.` Sample plots were generated by the
  required `--plot-samples` invocation.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Choice | 0.5834 | 0.5743 | Chance 0.5000 |
| Prior probability left | 0.5849 | 0.5753 | Chance 0.3333 |
| Wheel speed tertile | 0.5808 | 0.5792 | Chance 0.3333 |
| Whisker motion energy tertile | 0.7223 | 0.7208 | Chance 0.3333 |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Validation balanced accuracy (chance; ratio) | Expectation from papers |
|----------|------------------------------------------------|-------------------------|
| Choice | 0.5743 (0.5000; 1.149x) | Figure 5 reports ordinary region/session accuracy approximately 0.60--0.82; Figure 4 example reports AUC 0.66 baseline, 0.72 BMM-HMM, 0.79 oracle, with block examples 0.51/0.56, 0.64/0.70, 0.69/0.89. These are trial-level accuracy/AUC on selected regions/sessions, not all-timepoint balanced accuracy. |
| Prior probability left | 0.5753 (0.3333; 1.726x) | Paper treats prior as continuous: example correlations 0.74 vs 0.37 and 0.76 vs 0.54 (RRR vs ridge); Figure 4 shows 0.65/0.05/0.34 for oracle/single/multi example; Figure 5 spans roughly 0.3--0.7. These correlations cannot be numerically equated to required 3-class accuracy. |
| Wheel speed tertile | 0.5792 (0.3333; 1.738x) | Paper uses continuous R-squared: example RRR/ridge pairs 0.80/0.52 and 0.65/0.52; Figure 5 means roughly 0.39--0.53. Required tertile accuracy is a different endpoint. |
| Whisker motion-energy tertile | 0.7208 (0.3333; 2.162x) | Paper uses continuous R-squared: example RRR/ridge pairs 0.86/0.66 and 0.91/0.73; Figure 5 means roughly 0.38--0.50. Required tertile accuracy is a different endpoint. |

The paper additionally reports a 2% relative multi-session choice-accuracy gain,
Figure 8 IBL/external-task R-squared values from 0.24 to 0.80, and unrelated lick
accuracies of 0.71 (linear) and 0.90 (BMM-HMM). They are catalogued for completeness
but are not matched tasks/metrics for this conversion.

### Low-Accuracy Investigation

Choice is above chance but below the requested diagnostic threshold of 1.5x chance,
and 0.026 below the lowest approximate Figure 5 bar, so it was investigated rather
than dismissed:

1. **Raw target values**: Independently checked three source rows, not via conversion
   code: EID `6713...`, raw trial 9 is choice +1 -> 1/prior 0.5; EID `2336...`, raw
   trial 238 is +1 -> 1/prior 0.2; EID `c7bd...`, raw trial 1210 is -1 -> 0/prior
   0.2. Full 4x100 converted outputs matched independent reconstructions by
   `np.allclose` for all three.
2. **Alignment**: Inspected both processing plots (raw wheel/motion, sampled values,
   discretized classes, stimulus marker, and neural raster), `sample_trials.png`, and
   `predictions.png`. Neural responses and behavior transitions are synchronized;
   predictions track dynamic transitions without a consistent lag. No time shift was
   found.
3. **Variation**: Choice is 49.2116% left / 50.7884% right, not dominated by one
   class. Prior is also well populated; tertile outputs are balanced by construction.
4. **Filtering and neural processing**: Reconfirmed `qc=None` all-unit loading, exact
   reference event/RT/no-choice/duration mask, 20-ms half-open spike bins, and direct
   raw spike counts. Applying a good-unit or reportable-region filter solely to raise
   the score would contradict the supplied caching code.
5. **Why the paper number is not an attainable identity check**: The validator scores
   every one of the 100 timepoints. Choice is necessarily repeated across time in the
   mixed static/dynamic 4x100 target, so its score includes 0.5 s of pre-stimulus
   activity where choice information is weak. Figure 5 instead uses trial-level
   target-specific models on five selected regions and ten sessions; Figure 4 reports
   AUC after behavioral state smoothing. This is visible in `predictions.png`: choice
   predictions vary most early while later neural/behavioral predictions are coherent.
   The target format cannot represent choice as a separate 1D row alongside two
   time-varying rows, making repetition the only shape-consistent faithful encoding.

These tests identify no conversion change that would legitimately improve choice.
Changing alignment, deleting low-information bins, quality-filtering neurons, or
leaking choice through inputs would violate the explicit task/reference decisions.

### Train/Validation Gap

| Output | Train | Validation | Train/validation ratio |
|--------|-------|------------|------------------------|
| Choice | 0.5834 | 0.5743 | 1.016 |
| Prior | 0.5849 | 0.5753 | 1.017 |
| Wheel | 0.5808 | 0.5792 | 1.003 |
| Whisker | 0.7223 | 0.7208 | 1.002 |

All ratios are far below the 1.5x overfitting threshold. The close held-out scores,
balanced targets, and absence of a train-only advantage argue against leakage or
overfitting. The trial-number input does not encode block side; it is the explicitly
required zero-based within-block position.

### Issues Found and Resolved
- **Choice below 1.5x chance diagnostic**: Completed all five prescribed debugging
  checks. Mapping, variation, alignment, filtering, and processing all passed; the
  remaining gap is explained by genuinely different scoring units/window/subsets.
- **No output below chance**: All four exceed chance; three multiclass outputs exceed
  1.5x chance.
- **No large train/validation gap**: Maximum ratio is 1.017; no remedial change needed.
- **No conversion issue found**: Consequently no conversion/training rerun was warranted.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

Created the user-facing README with dataset purpose, exact retained statistics,
load example, schema/mappings, reproduction commands, validation caveat, and full
decoder results. Moved the independent audit script/output, temporary paper rendering,
and generated bytecode into `cache/`; documented each in `cache/README_CACHE.md`.
Kept the conversion program, required logs/pickles, and user-facing processing/training
plots at the project root. Final nonempty-file audit confirmed every required artifact.
