# Dataset Conversion Notes

## Overview
- **Dataset**: IBL Brain-Wide Map public ONE cache (data paper release)
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `.manifest`, `Dockerfile`, `code/`, `data/`, `dataarchitecture.pdf`, `datapaper.pdf`, `decoder.py`, `docker-compose.yaml`, `ibl_docs/`, `methodpaper.pdf`, `methods.txt`, `stage_cache.sh`, `train_decoder.py`, and this notes file.

Environment verification:
- Python 3.13.15; NumPy 2.3.5; PyTorch 2.6.0+cu124 import successfully.
- Mandatory file checkpoint passed: `ls -la /app/CONVERSION_NOTES.md` confirmed the file exists.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `prepare_data` | `code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Resolves all probes with `one.eid2pid`, loads each using `SpikeSortingLoader`, merges probes, loads trials/behaviors, and returns neural, behavioral, metadata, and trial dictionaries. |
| `load_spiking_data` | `code_zhang2025/src/utils/ibl_data_utils.py` | LOADING/CURATION | Uses `SpikeSortingLoader.load_spike_sorting` and `merge_clusters`; optional `qc=1` selects clusters with `label >= 1`, but the caching pipeline calls it with default `qc=None`. |
| `load_trials_and_mask` | `code_zhang2025/src/utils/ibl_data_utils.py` | CURATION | Uses `SessionLoader.load_trials`; excludes missing required events, no-choice trials, RT outside 0.08–2 s, and (via `prepare_data`) trial duration above 10 s. It retains the initial 0.5-prior block. |
| `bin_spiking_data` / `get_spike_data_per_interval` | `code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Builds stimulus-aligned windows and bins spike counts with half-open time intervals and `bincount2D`. |
| `load_target_behavior` | `code_zhang2025/src/utils/ibl_data_utils.py` | LOADING/PROCESSING | Uses `SessionLoader`; wheel speed is absolute interpolated wheel velocity; whisker motion energy prefers left camera, falling back to right only on load failure. |
| `get_behavior_per_interval` / `bin_behaviors` | `code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Extracts behavior windows and linearly interpolates at bin right edges; default caching permits NaNs but removes missing intervals during alignment. |
| `align_spike_behavior` | `code_zhang2025/src/utils/ibl_data_utils.py` | CURATION | Deletes trials failing behavioral availability and trial masks, then verifies equal trial counts. |
| `list_brain_regions` / `select_brain_regions` | `code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Maps Allen acronyms to the Beryl atlas and selects cluster IDs. |

### Notes
- Reference preprocessing in `0_data_caching.py` uses 20 ms spike/behavior bins, a 2 s window from −0.5 to +1.5 s relative to `stimOn_times`, and merges all probes in a session.
- It caches choice, reward, prior block, wheel speed, and whisker motion energy. The requested downstream task omits reward and adds explicitly specified categorical encodings.
- This is electrophysiology, so delta-F/F is not applicable.
- Although quality labels are saved as metadata, the published caching call does **not** filter to good clusters (`qc=None`). Therefore all loaded sorted clusters are the reference choice unless later source text establishes a different BWM curation requirement.
- Neural arrays produced by reference code are trial × time-bin × cluster; target format requires transposition per trial to neuron × time-bin.
- The code contains a likely masking bug (`target_mask = target_mask and beh_mask` inside the loop retains only the final behavior mask because Python list truthiness is used). We will preserve intended semantics—logical conjunction across every behavior and trial QC—not reproduce an implementation defect.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- The cache root is `/app/data/one_cache`; data are organized in standard IBL ALF hierarchy (lab/subject/date/session/collection), with symlinked lab trees and separate ONE table snapshots `2022_Q4_IBL_et_al_BWM`, `2025_Q3_IBL_et_al_BWM`, and aggregate `Brainwidemap`.
- All dataset discovery below used `one.api.ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')`; no neuroscience arrays were read directly.
- Aggregate ONE inventory: 480 sessions, 143 subjects, and 76,439 registered datasets. The 2025 BWM snapshot has 459 sessions and 139 subjects; the older 2022 snapshot has 354 sessions and 115 subjects. Step 4 will resolve the release matching the supplied paper/code.
- Relevant ALF objects/attributes are `trials.table` (stimulus, choice, prior, movement and feedback fields), `spikes.times`, `spikes.clusters`, `clusters.*` (acronym/location/metrics), `wheel.position` + `wheel.timestamps`, camera timestamps, and left/right `Camera.ROIMotionEnergy` plus ROI definitions.
- Tables include revisioned collections. ONE resolves the appropriate/default revision; conversion must never infer revisions from filenames.
- Typical types loaded by ONE/brainbox are floating arrays for timestamps and continuous behavior, integer cluster IDs/spike counts, a pandas-like trial table, and cluster metadata arrays/tables.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | To be computed for the paper-matched release after Step 4 resolves session selection; aggregate tables contain multiple releases/revisions, so counting now would double-count. |
| Neurons / session | To be computed via `SpikeSortingLoader` for selected sessions. |
| Subjects | 139 in 2025 BWM snapshot (143 aggregate) |
| Sessions / subject | 459 / 139 = 3.30 registered sessions/subject in snapshot; reference decoder selects one session per sampled subject. |
| Trials (total) | To be computed via `SessionLoader.load_trials` after release resolution. |
| Trials / session | To be computed after reference QC mask. |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 621,733 sorted units; 75,708 well-isolated | Data paper methods: 621,733 units, 75,708 passed stringent QC. | 
| Neurons / session | 1,354 sorted units/session; 165 well-isolated/session (derived) | 621,733 / 459; 75,708 / 459. |
| Subjects | 139 (94 male, 45 female) | Data paper methods. |
| Sessions / subject | 459 total release sessions / 139 subjects | Data paper methods. |
| Trials (total) | At least 250/session by release criteria | Data paper inclusion criteria. |
| Trials / session | Sessions retained with >=250 trials; collection description elsewhere notes >=400 for primary analyses | Data paper methods passages. |
| Neural data time bin | 20 ms for supplied decoder cache | Method paper: 2-s trials divided into 20-ms bins (100 steps). |
| Behavior data time bin | Raw video 60 Hz left / 150 Hz right; interpolated to neural bins | Data/method papers and reference code. |
| Reward rate | Not explicitly reported in copied methods | Not used as requested output. | 
| Decoder session count | 433 | Method paper: models applied to 433 IBL sessions. | 
| Brain coverage | 270 regions | Method paper. |


### Processing Details
- Supplied caching implementation aligns all requested variables to stimulus onset in [−0.5, +1.5) s with 20-ms bins. The narrative method paper uses that window for choice, but variable-specific alternatives for prior (−0.6 to −0.1 s, 50-ms bins) and dynamic behavior (first movement to +1 s, 20-ms bins). The downstream task explicitly requires stimulus-onset alignment and common timepoints, so the caching implementation's common stimulus window is the applicable reference.
- Choice and prior are per-trial; wheel speed and whisker motion energy are time-varying. Whisker motion energy is mean absolute adjacent-frame difference within a whisker-pad bounding box anchored by nose and eye.
- Wheel speed is absolute wheel velocity. Reference behavioral streams are linearly interpolated to bin right edges.

### Curation Steps

**Neuron curation rules**:
- Data paper analysis definition: amplitude >50 µV, noise cutoff <20 µV, and refractory-period criterion; 75,708/621,733 pass. However, method-paper caching explicitly uses all neurons and the supplied code calls `load_spiking_data(qc=None)`. For reproducing the decoder dataset, retain all clusters while preserving atlas regions.

**Trial curation rules**:
- Exclude missing choice, probabilityLeft, feedbackType/time, stimOn time, or firstMovement time.
- Exclude first-movement latency outside 0.08–2.00 s, no-choice trials, and trials longer than 10 s in supplied code. Retain unbiased 0.5-prior trials.

### Decoders Trained
| Decoded variable | Accuracy |
| Choice | Paper reports balanced accuracy as the metric; no single global accuracy is stated in copied text. |
| Wheel speed / whisker motion energy | Papers report continuous decoding with R²; not directly comparable to requested 3-class balanced accuracy. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Release identity | `bwm_release.csv`: 699 probes, 459 EIDs, 139 subjects | 2025 ONE snapshot: 459 sessions, 139 subjects | Data paper: 459 sessions, 699 insertions, 139 mice | Use the exact CSV EIDs and probe membership; ONE/brainbox perform all actual loading. |
| Decoder availability | Reference paper analyzes 433 sessions | Some of 459 lack complete target behavior after QC | 433 sessions cited | Attempt all 459, retain only sessions with >=2 jointly valid trials and all required streams; expect 433. |
| Cluster QC | Caching passes `qc=None`; metadata only labels good clusters | Both all clusters and metrics exist | Method paper says all neurons; data paper primary analyses emphasize 75,708 good neurons | Follow decoder caching pipeline: all clusters. This explains expected 621,733-scale rather than 75,708-scale count. |
| Alignment | Cache script: stimOn, −0.5:+1.5 s, 20 ms for all outputs | Streams have shared clock timestamps | Narrative has target-specific windows | Downstream mandates stimulus alignment/common bins; use cache-script common window. |
| Whisker view | Prefer left, fall back right | Both views may exist at different rates | Both described | Match reference fallback behavior; never average views. |

Final understanding: this is the 459-session BWM release. Conversion candidates are all release EIDs, with all simultaneously recorded probes merged per EID, reference trial QC, 100 stimulus-aligned 20-ms bins, and jointly available wheel/whisker streams. The expected final count is near the 433 sessions used by the methods paper.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spikes.times`, `spikes.clusters` from every release probe | `neural` | Merge probes, count spikes in 100 half-open 20-ms bins spanning stimOn−0.5 to stimOn+1.5; transpose to neuron×time; float32 | `prepare_data`, `merge_probes`, `bin_spiking_data` | All sorted clusters retained, matching decoder cache. |
| Bin right-edge offsets | `input[0]` time since stimulus onset | `[-0.48, ..., 1.50]` s repeated identically per trial | `get_behavior_per_interval` | Continuous, time-varying; using reference interpolation coordinates ensures exact stream alignment. |
| `trials.probabilityLeft` run position | `input[1]` trial number in block | Zero-based count within each consecutive prior block, repeated over time | planned run-length transform | Computed before trial filtering so excluded trials do not collapse experimental block position. |
| `trials.choice` | `output[0]` choice | IBL −1 (left choice)→0, +1 (right choice)→1, repeated over time | `bin_behaviors` plus requested recode | No-choice 0 excluded. |
| `trials.probabilityLeft` | `output[1]` prior probability of left | 0.2→0, 0.5→1, 0.8→2, repeated over time | `bin_behaviors` plus requested recode | Exact categorical mapping. |
| absolute `SessionLoader.wheel.velocity` | `output[2]` wheel speed | Linear interpolation to bin right edges; discretize by session-wide 1/3 and 2/3 quantiles across jointly valid trial×time samples | `load_target_behavior`, `get_behavior_per_interval` | Three ordinal categories low/medium/high; per-session quantiles accommodate rig scaling and produce learnable, non-degenerate classes. |
| left (fallback right) `SessionLoader.motion_energy.*.whiskerMotionEnergy` | `output[3]` whisker motion energy | Same interpolation and per-session tertile discretization | `load_target_behavior`, `get_behavior_per_interval` | Do not average camera views. |
| cluster acronym | `brain_region_idx` | Allen acronym→Beryl acronym using `BrainRegions.acronym2acronym`; global sorted vocabulary | `list_brain_regions` | Unknown/root retained as atlas labels rather than dropping neurons. |

### Key Decisions
1. **Window/binning**: Use exactly −0.5 to +1.5 s and 20 ms (100 bins), matching supplied cache code and decoder-task alignment.
2. **Trial validity**: Apply reference event/RT/no-choice/max-duration QC and additionally require finite coverage of both requested dynamic outputs across the full window. Sessions need >=2 valid trials.
3. **Continuous-output discretization**: Tertiles are computed within session after joint-valid masking, but before categorical conversion. This is robust to between-camera/rate calibration differences and ensures all three requested categories are represented; repeated quantile edges are handled deterministically with rank-based fallback.
4. **Per-trial outputs**: Repeat choice and prior across 100 bins because the target container must have a uniform `n_output × n_timepoints` shape alongside time-varying outputs.
5. **Time coordinate**: Use bin right edges as the reference behavior code does; neural counts cover corresponding preceding half-open bins.
6. **Memory**: Store neural counts as float32 and categorical outputs as int64. Build one session at a time and serialize once.

### Planned Sanity Checks
- [ ] ONE/brainbox raw-versus-converted `np.allclose` spot checks for spike-bin counts, time input, trial-in-block, choice/prior, and interpolated continuous behavior before discretization.
- [ ] Assert every trial neural/input/output has shapes `(n_neurons,100)`, `(2,100)`, `(4,100)`; all finite; categories/ranges exact.
- [ ] Compare candidate/retained sessions, subjects, trials, neurons, regions with 459/139 release totals and 433 method-paper session expectation.
- [ ] Verify dynamic category fractions near one third per session and static class distributions plausible.
- [ ] Plot raw/interpolated/binned behavior and neural population counts for up to two sessions.

---

## Step 6: Script Development
**Status**: IN PROGRESS

[Implementation notes]
- Created `/app/convert_data.py` with the required CLI modes, ONE/brainbox-only neuroscience loading, reference trial QC, multi-probe merging, vectorized spike binning, behavior interpolation, categorical construction, validation-by-construction, timing, and processing plots.
- `python3 -m py_compile /app/convert_data.py` passes.
- **Blocking source-cache defect (2026-09-20):** the first mandated sample execution queried all 459 release EIDs through `SessionLoader`. Every session's `trials` object contained only `goCueTrigger_times` (and occasionally another legacy attribute), while the required `_ibl_trials.table.pqt` could not be loaded. ONE's cache table nevertheless reports that table with `exists=True`. The same mismatch affects registered `leftCamera.ROIMotionEnergy` arrays. Thus this is not a converter exception or a QC choice: required source datasets are absent from the staged filesystem despite metadata claiming presence.
- Evidence: all 459 sessions failed identically with missing `choice`, `feedbackType`, `feedback_times`, `firstMovement_times`, `goCue_times`, `probabilityLeft`, and `stimOn_times`. The complete log is `/app/conversion_sample_out.txt`. An authenticated remote ONE attempt was also unavailable because outbound access to OpenAlyx was refused.
- Per the required ordered workflow, Step 6 cannot be marked complete (the converter cannot complete against missing inputs), and Steps 7–13 have not been started.

Code inefficiencies identified:
- Reference code creates a multiprocessing pool per stream/session and performs redundant behavior loads.

Code speedups added:
- Single SessionLoader per session; vectorized interpolation for every trial; sorted-spike `searchsorted` windows; flattened `np.bincount`; no redundant behavior loading.

---

## Step 7: Sample Conversion and Validation
**Status**: [NOT STARTED | IN PROGRESS | COMPLETE]

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | |
| Neurons / session | |
| Subjects | |
| Sessions / subject | |
| Trials (total) | |
| Trials / session | |
| <Input statistic 1 range> | [MIN, MAX] |
| ... | [MIN, MAX] |
| <Output statistic 1 distribution> | [FRAC0,FRAC1,...] |
| ... | [FRAC0,FRAC1,...] |

### Processing Plots Review
[Notes on any anomalies]

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |

| Step | Time / Session | Estimated Total Time |
| | | |

---

## Step 8: Sample Decoder Training
**Status**: [NOT STARTED | IN PROGRESS | COMPLETE]

### Format Validation
- Errors: [None / List]
- Warnings: [None / List]

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| <Output 1> | | |
| <Output 2> | | |
| ... | | |

---

## Step 9: Full Conversion and Validation
**Status**: [NOT STARTED | IN PROGRESS | COMPLETE]

### Output Files
- `converted_data.pkl`: [size]
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | | | | | |
| Mean neurons/session | | | | | |
| Subjects | | | | | |
| Sessions | | | | | |
| Trials (total) | | | | | |
| Trials/session (mean) | | | | | |
| <Input 1 range> | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| ... | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| <Output 1 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |
| <Output 2 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |

---

## Step 10: Critical Review 1
**Status**: [NOT STARTED | IN PROGRESS | COMPLETE]

### Checks Performed
1. [Check]: [Result]

### Issues Found and Resolved
- [Issue]: [Resolution]

---

## Step 11: Full Decoder Training
**Status**: [NOT STARTED | IN PROGRESS | COMPLETE]

### Training Progress
- Loss decreasing: [Yes/No]

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| <Output 1> | | |
| <output 2> | | |
| ... | | |

---

## Step 12: Critical Review 2
**Status**: [NOT STARTED | IN PROGRESS | COMPLETE]

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from papers |
| | |

[Analysis of any low accuracies]

### Issues Found and Resolved
- [Issue]: [Resolution]

---

## Step 13: Documentation and Cleanup
**Status**: [NOT STARTED | IN PROGRESS | COMPLETE]

- [ ] README.md created
- [ ] cache/ folder created
- [ ] All files organized
