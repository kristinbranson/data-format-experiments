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

Environment check: Python 3.13.15, NumPy 2.3.5, and PyTorch 2.6.0+cu124 imported successfully. The required notes-file checkpoint (`ls -la /app/CONVERSION_NOTES.md`) passed.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `prepare_data` | `code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Resolve all probe insertions for an EID, load/merge spike sorting, load trials and behavioral streams, and package unit metadata. |
| `load_spiking_data` | `code_zhang2025/src/utils/ibl_data_utils.py` | LOADING/CURATION | Load spike sorting with `SpikeSortingLoader`; optional `qc` threshold selects `clusters.label >= qc`, but the reference pipeline calls it with `qc=None` and therefore retains all clusters. |
| `merge_probes` | `code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Reindex cluster IDs across probes, concatenate units/spikes, and stably sort spikes by time. |
| `load_trials_and_mask` | `code_zhang2025/src/utils/ibl_data_utils.py` | CURATION | Build trial mask: 0.08–2 s first-movement latency, trial duration at most 10 s, required non-NaN events, and exclude `choice == 0`; unbiased 0.5-prior trials are retained. |
| `list_brain_regions` / `select_brain_regions` | `code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Map Allen acronyms to Beryl regions and select cluster IDs; the caching script combines all Beryl regions per session. |
| `get_spike_data_per_interval` / `bin_spiking_data` | `code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Count spikes in half-open bins for trial windows; returns trial × time × cluster arrays. |
| `load_target_behavior` | `code_zhang2025/src/utils/ibl_data_utils.py` | LOADING/PROCESSING | Load uniformly sampled wheel and camera motion-energy data; wheel speed is absolute smoothed wheel velocity; whisker energy prefers left camera and falls back to right. |
| `get_behavior_per_interval` / `bin_behaviors` | `code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Linearly interpolate behavior to one sample per neural bin, aligned to the same event/window; preserve NaNs when `allow_nans=True`. Also expose choice, block (`probabilityLeft`), reward, and contrast per trial. |
| `align_spike_behavior` | `code_zhang2025/src/utils/ibl_data_utils.py` | CURATION | Delete trials rejected by behavior availability/trial mask and assert trial count agreement. |
| `get_sparse_from_binned_spikes` / `create_dataset` | `code_zhang2025/src/utils/dataset_utils.py` | PROCESSING | Store binned counts as per-trial CSR arrays plus behaviors and unit/session metadata. |
| `standardize_spike_data` | `code_zhang2025/src/utils/data_loader_utils.py` | PROCESSING | Decoder-only normalization by time point across trials/neurons using training statistics; raw cached data remain spike counts. |

### Notes
- Primary workflow: `0_data_caching.py` calls `prepare_data` → Beryl mapping/region selection → 20 ms spike binning → 20 ms behavior interpolation → trial alignment → 70/10/20 random split.
- Reference alignment parameters are `align_time='stimOn_times'`, `time_window=(-0.5, 1.5)` s, `binsize=0.02` s, yielding 100 time bins per trial.
- Neural representation is raw spike counts, not firing rates and not fluorescence; delta-F/F is inapplicable to this electrophysiology dataset.
- The reference cache retains all loaded clusters (`qc=None`). Unit quality (`label >= 1`) is saved only as `good_clusters` metadata. This differs from common BWM analyses that may explicitly use good units, and will be reconciled against the data paper/data release in Steps 3–4.
- Choice is native IBL `-1`/`+1` in the cache; the requested target conversion must remap left/right explicitly. Block is native `probabilityLeft` (0.2/0.5/0.8).
- Behavior interpolation points are `interval_start + binsize` through `interval_end`; spike bins cover half-open intervals from the interval start. The one-bin convention is therefore behavior at the right edge of each corresponding spike-count bin.
- The code's `align_spike_behavior` implementation only preserves the final behavior availability mask due to Python-list boolean semantics, while the trial mask is applied. For the requested two continuous outputs, conversion should require both wheel and whisker data to be valid, avoiding this apparent reference bug.
- The supplied method code treats wheel speed/whisker motion energy as continuous regression targets. The current task overrides only their target representation by requiring three categorical bins.
- Decoder-side code standardizes spike counts and uses classification for choice, regression for continuous behaviors; the supplied `/app/train_decoder.py` governs the requested multi-output categorical validation instead.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
The local source is an IBL ONE/ALF cache at `data/one_cache/` (149 MB inside the project plus lab-directory symlinks to the mounted dataset). It contains three ONE cache-table snapshots (`2022_Q4_IBL_et_al_BWM`, `2025_Q3_IBL_et_al_BWM`, and `Brainwidemap`) plus lab/subject/date/session ALF trees. The broad `Brainwidemap` catalog contains 480 sessions, 143 subjects, and 12 labs (2019-11-26 through 2023-10-20); 461 session directories from 141 subjects are materialized locally.

Native hierarchy: `lab/Subjects/<subject>/<date>/<number>/alf/`. Each materialized session has a trial table, wheel timestamps/position, and one or more `probe*/pykilosort` trees. Camera timestamps and motion-energy streams may be left, right, or both. Explicit revision directories (for example `#2025-03-03#`) coexist with unrevised paths; conversion must select the ONE-default/newest intended revision once per logical object rather than count duplicates.

Available variables and formats:
- Trial Parquet: 13 float64 columns in every materialized session: `goCue_times`, `response_times`, `choice`, `stimOn_times`, `contrastLeft`, `contrastRight`, `probabilityLeft`, `feedback_times`, `feedbackType`, `rewardVolume`, `firstMovement_times`, `intervals_0`, `intervals_1`.
- Wheel: same-length float NumPy vectors `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy` in all 461 sessions; velocity/speed must be derived with the reference SessionLoader processing.
- Video behavior: camera timestamp vectors plus `leftCamera.ROIMotionEnergy.npy` (439 materialized sessions) and/or right-camera counterpart (422); their union is 447/461 materialized sessions, so 14 release sessions have no whisker motion-energy stream.
- Neural: per-probe NumPy `spikes.times`, `spikes.clusters` (plus amplitudes/depths/templates), `clusters.channels`, cluster metric Parquet, channel/electrode atlas IDs and coordinates. There are 701 materialized probe trees. Cluster metrics include `label`, firing rate, presence ratio, contamination/noise and refractory-period metrics.
- Brain location is stored as Allen CCF numeric IDs at channel/electrode level; neuron acronyms/regions must be obtained by associating each cluster's peak channel and mapping atlas ID to an Allen acronym, then Beryl as in the reference code.
- Example dimensions: trial table `(n_trials, 13)`; wheel `(n_samples,)`; camera motion energy `(n_frames,)`; spike times and cluster IDs `(n_spikes,)`; cluster metrics `(n_clusters, 22)` (two older probe records omit several newer columns).

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 622,377 raw sorted clusters; 75,808 `label >= 1` clusters (12.18%) |
| Neurons / session | Raw probe-level 135–2,311, mean 887.84, median 863; session totals depend on 1–4 merged probes |
| Subjects | 143 catalogued; 141 materialized locally |
| Sessions / subject | Catalog: 1–13, mean 3.357; 480 catalogued sessions |
| Trials (total) | 297,505 native trials; 196,727 pass the reference trial-event/latency/duration/choice mask before continuous-stream availability checks |
| Trials / session | Native 401–1,525, mean 645.35, median 601; reference mask 126–1,445, mean 426.74, median 393 |

Additional raw-data statistics: 701 probe recordings; 21,145,748,401 spike events represented by the available on-disk arrays; native choice counts are left (`-1`) 146,529, right (`+1`) 149,824, no-go (`0`) 1,152; prior counts are 0.2: 124,194, 0.5: 41,490, 0.8: 131,821; raw reward fraction (`rewardVolume > 1`) is 0.8189. These trial distributions are before reference curation.

No separate README is present under `data/`; the ONE Parquet cache tables supply session/dataset paths, hashes, revisions, QC state, and subject/lab metadata.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 621,733 sorted units; 75,708 well-isolated | Data paper: “621,733 units … 75,708 well-isolated neurons.” | 
| Neurons / session | Method dataset mean 676, approximately 300 to >2,000 | Method paper STAR Methods: “average of 676 neurons per session.” |
| Subjects | 139 mice (94 male, 45 female) | Data paper: “We trained 139 mice (94 male and 45 female).” |
| Sessions / subject | Not reported as a distribution | Data paper reports 459 retained sessions across 139 mice. |
| Trials (total) | Not reported as one total | — |
| Trials / session | At least 400 for main analyses; release criterion at least 250 plus performance/hardware criteria | Data paper main text and Methods inclusion criteria. |
| Neural data time bin | Method paper overview/code: 20 ms for 2-s windows (100 bins); detailed text: 50 ms for static choice/prior and 20 ms for dynamic behaviors | Method paper: “2-s trials, each divided into 20-ms bins”; detailed STAR Methods distinguishes target-specific windows/bins. |
| Behavior data time bin | Motion signals native near 60 Hz in method paper; aligned/interpolated to neural bins in code | Method paper describes wheel and whisker as time-varying signals sampled at 60 Hz; code interpolates to 20 ms bins. |
| Reward rate | Not explicitly tabulated | Native local data are 81.89% rewarded before curation; retained as a data-derived check, not a paper expectation. |
| Sessions / insertions in public release | 459 / 699 | Data paper: “459 sessions, 699 insertions and 621,733 neurons remained.” |
| Sessions / brain regions used by method paper | 433 / 270 | Method paper: “433 IBL sessions, covering 270 brain regions.” |
| Reaction-time exclusions | Exclude <0.08 s and >2.00 s; 22.8% are under 80 ms | Data paper Fig. 1 caption and Methods. |
| No-go fraction | 1.5% illustrated | Data paper Fig. 1c. |
| Zero-contrast accuracy | 58.7 ± 0.4% across mice | Data paper Fig. 1 caption. |


### Processing Details
- Requested/common static alignment: choice is aligned to `stimOn_times` from −0.5 to +1.5 s. The method paper's prior analysis instead used −0.6 to −0.1 s and the dynamic behavior analysis used first movement to +1 s; the current decoder task explicitly requires stimulus-onset alignment for all outputs, so a common −0.5/+1.5 s window is the applicable reference-code choice.
- Spike sorting is Kilosort 2.5/custom IBL Kilosort. Neural matrices contain non-overlapping binned spike counts. Multiple probes in one session are merged because they share behavior and are not independent.
- Wheel speed is derived from wheel position/velocity; whisker motion energy is the mean pixelwise absolute difference between adjacent frames within a DLC-anchored whisker-pad ROI. Behavior is synchronized/interpolated to neural time bins.
- The architecture paper confirms ALF object/attribute invariants: `spikes.times` and `spikes.clusters` have equal rows, integer cluster assignments cross-reference cluster rows, and ONE cache metadata controls dataset paths/revisions. This supports loading logical ALF objects rather than treating every revision file as independent.
- Method decoders use full-trial neural activity for static variables and all within-trial activity for each dynamic time-bin target. The supplied conversion validator defines the actual architecture for this task.

### Curation Steps

**Neuron curation rules**:
Data-paper analyses use well-isolated units passing amplitude >50 µV, noise cutoff <20 µV, and refractory-period-violation criteria (represented by cluster `label >= 1` in the release). Some decoding method code loads all Kilosort clusters while storing this label only as metadata. Final analyses in the data paper also restrict to grey-matter regions with at least five well-isolated neurons/session and at least two sessions, but that region-level rule is analysis-specific and would discard useful full-session decoder neurons.

**Trial curation rules**:
Exclude trials missing choice, `probabilityLeft`, `feedbackType`, `feedback_times`, `stimOn_times`, or `firstMovement_times`; exclude first-movement latency outside 0.08–2.00 s. Reference code additionally excludes no-go choice and trials longer than 10 s. Continuous behavior must cover the complete requested window; trials with absent streams/coverage are invalid.

### Decoders Trained
| Decoded variable | Accuracy |
| Choice | Method paper Fig. 5: region-specific accuracy roughly 0.60–0.83 across PO/LP/DG/CA1/VISa and models; example BMM-HMM AUC 0.72 versus baseline 0.66 and oracle 0.79. |
| Prior | Method paper reports Pearson correlation (not categorical accuracy): broadly ~0.2–0.7 region/model dependent; example oracle/single/multi correlations 0.65/0.05/0.34. |
| Wheel speed | Method paper reports continuous R², roughly 0.4–0.55 in Fig. 5. |
| Whisker motion energy | Method paper reports continuous R², roughly 0.35–0.5 in Fig. 5. |

The paper metrics are not numerically interchangeable with this task's three-class prior/wheel/whisker balanced accuracy. They are useful qualitative lower-bound checks only; chance here is 0.5 for choice and 1/3 for each three-class output.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Release membership | `bwm_release.csv` has 699 probe rows, 459 EIDs, 139 subjects | Local trees contain 701 probes/461 sessions; two extra Cortexlab probe/session trees account for exactly 644 raw and 100 good units | 699 probes, 459 sessions, 139 subjects, 621,733 units, 75,708 good units | Use the supplied release CSV as the authoritative freeze. It maps without missing probes and exactly reproduces both published unit totals; exclude the two non-release trees. |
| Dataset revisions | Reference code delegates to ONE defaults | Trial/camera objects have multiple physical revisions | Official IBL release notes recommend current corrected video data; ONE docs expose default-revision selection | Use the current explicitly revised trial/video products present in the supplied cache (equivalent to ONE's intended default), once per logical object. Never double-count revision files. |
| Sessions with behavior | Caching code prefers left whisker motion energy, falls back to right, and skips failures | 437 release sessions have left ME, 8 additional have only right, 14 have neither; one right-only stream has no coverage near valid trials. After reference trial and full-window coverage checks, 444 sessions and 188,927 trials have whisker coverage (188,925 also have wheel coverage) | Method paper analyzed 433 sessions; release changelog notes video removal/correction over time | The method-paper count reflects its analysis snapshot/additional availability. For the requested full conversion using the supplied current release, retain every release session with both requested streams and at least two usable trials: 444 sessions, with the final exact intersection determined during conversion. Missing ME cannot be reconstructed and its sessions are excluded. |
| Trial selection | 0.08–2.00 s latency, required event fields, no-go exclusion, ≤10 s go-cue-to-feedback | 195,781/296,090 release trials pass this code mask; behavior coverage removes additional trials | Paper mandates event presence and 0.08–2.00 s latency; no-go trials are invalid for binary choice | Apply the reference-code superset, then require complete wheel and whisker coverage for the common window. Preserve original trial order so trial number/block structure remains meaningful. |
| Temporal alignment | Caching script: stimulus onset, −0.5/+1.5 s, 20 ms for all behaviors | Streams share session-clock seconds and cover those windows for most curated trials | Detailed method text uses target-specific alignments/bins, but task explicitly says stimulus-onset alignment and target format requires a common bin size | Follow the requested common alignment and the executable reference caching code: 100 bins of 20 ms over [−0.5,+1.5) s. |
| Neuron quality | Method caching calls `load_spiking_data(qc=None)` and saves `label>=1` only as metadata | Freeze exactly reproduces 621,733 all clusters and 75,708 label-passing clusters | Data-paper analyses use well-isolated neurons; method paper says “all neurons” and reports an intermediate average that is not consistent with merged-probe release totals | This is a genuine cross-source difference. For robust downstream decoding and explicit data-paper curation, Step 5 will use well-isolated (`label>=1`) units while documenting the intentional divergence from the method cache. |
| Brain-region definition | Map original atlas acronyms to Beryl and merge probes by session | Per-channel Allen CCF IDs plus cluster peak-channel indices are present | Paper assigns every neuron to Allen CCF and uses region inclusion rules for region-wise inference | Map each retained unit through its peak channel to Allen acronym then Beryl. Do not apply the paper's region-level ≥5-neuron/≥2-session inferential filter because target format needs neuron annotations rather than region-level hypothesis tests. |
| Wheel preprocessing | SessionLoader resamples position at 1 kHz and computes 20 Hz, order-8 Butterworth zero-phase velocity; speed is absolute velocity | Every release session has equal-length position/timestamp vectors; only two otherwise valid trials fall outside wheel coverage | Data paper bins/averages wheel values in 20 ms intervals; method code interpolates reference speed at behavior-bin timepoints | Reproduce SessionLoader velocity exactly and sample at the reference right-edge grid. |
| ME preprocessing | SessionLoader corrects excess leading timestamps, then code linearly interpolates to behavior-bin right edges | No selected valid motion-energy segment contains NaNs; timestamp-shorter-than-data/coverage checks expose invalid streams | Paper defines whisker-pad ROI frame-difference motion energy | Reproduce timestamp correction, left-first/right-fallback selection, and linear interpolation; exclude uncovered trials/sessions. |

Final source-consistent understanding: process only the 459-session/699-insertion publication freeze; merge simultaneous probes within an EID; retain valid original trial order; use release quality labels and Beryl annotations; align all requested variables to stimulus onset on the reference 20 ms grid; and make exclusions only for reference trial QC or missing required streams. Official IBL documentation confirms the `Brainwidemap` release counts and that current revisions include corrected wheel/video products.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spikes.times`, `spikes.clusters`; cluster `label`; cluster peak channel | `neural` | Select publication-freeze probes and `label >= 1` units; merge probes within EID; count spikes into 100 half-open 20 ms bins from −0.5 to +1.5 s around `stimOn_times`; store float32 `(neurons, 100)` | `load_spiking_data`, `merge_probes`, `get_spike_data_per_interval`, `bin_spiking_data` | Raw counts, no smoothing or firing-rate conversion. |
| Reference right-edge bin grid | `input[0]` | Time since stimulus onset `[-0.48, -0.46, ..., 1.50]` s, repeated identically for each trial | `get_behavior_per_interval` | Continuous, time-varying. Values are right edges of corresponding neural count bins, matching behavioral interpolation. |
| Full raw `probabilityLeft` sequence | `input[1]` | Detect block transitions before trial filtering; zero-based count since current block began; repeat the per-trial scalar over 100 bins | Task design / trial-order handling | Continuous per-trial context; excluded trials do not renumber later retained trials. |
| `_ibl_trials.choice` | `output[0]` | Native −1 (left) → 0; +1 (right) → 1; repeat across 100 bins | `bin_behaviors`; task override | No-go 0 trials excluded. |
| `_ibl_trials.probabilityLeft` | `output[1]` | 0.2 → 0; 0.5 → 1; 0.8 → 2; repeat across 100 bins | `bin_behaviors` | This task requests actual block prior probability, not the method paper's model-derived continuous subjective prior. |
| `_ibl_wheel.timestamps`, `_ibl_wheel.position` | `output[2]` | Resample position to 1 kHz; order-8, 20-Hz zero-phase Butterworth differentiation; absolute velocity; linearly sample at right-edge grid; session-wise empirical tertiles → classes 0/1/2 | `SessionLoader.load_wheel`, `interpolate_position`, `velocity_filtered`, `load_target_behavior`, `get_behavior_per_interval` | Time-varying. Session-wise thresholds parallel method decoder's per-session behavioral scaling and avoid between-rig scale confounds. |
| Left/right camera timestamps and `ROIMotionEnergy` | `output[3]` | Prefer left, fall back right; if timestamps exceed frames drop leading timestamps; linearly sample at right-edge grid; session-wise empirical tertiles → classes 0/1/2 | `SessionLoader.load_motion_energy`, `_check_video_timestamps`, `load_target_behavior`, `get_behavior_per_interval` | Time-varying; sessions/trials without full coverage excluded. |
| Release subject strings | `subjects`, `subject_idx` | Stable first-appearance unique list and per-session integer index | `bwm_release.csv` | 139 release subjects exist; retained set may be smaller after required-stream exclusion. |
| Channel `brainLocationIds_ccf_2017`, `clusters.channels` | `brain_regions`, `brain_region_idx` | Peak-channel atlas ID → Allen acronym → Beryl acronym; global sorted acronym vocabulary and per-neuron indices | `SpikeSortingLoader.merge_clusters`, `list_brain_regions` | Unit order matches merged neural rows. |
| Trial/session/probe identifiers and thresholds | `metadata.session_info` | Record EID, subject/lab/date, probe names, original retained trial indices, selected camera side, unit counts, and discretization thresholds | — | Supports audit/reconstruction. |

### Key Decisions
1. **Release freeze**: Use all and only `code/code_zhang2025/data/bwm_release.csv` entries. This exactly matches published totals and eliminates two unrelated cached sessions.
2. **Good units**: Use `clusters.metrics.label >= 1`. This follows the data-paper's explicit neural curation, improves biological validity, and keeps the dense target representation tractable. It intentionally differs from the method caching script's `qc=None`; all-cluster counts remain checked as a release-integrity invariant.
3. **Common window/bin**: Use 20 ms and [−0.5,+1.5) s around stimulus onset for every stream. This is required by the task's stimulus alignment/common time dimension and matches the executable cache script/overview, despite target-specific windows in the detailed method-paper analysis.
4. **Continuous sampling**: Use the reference code's right-edge interpolation grid rather than bin centers or per-bin behavior averages, because this is the exact supplied implementation.
5. **Continuous-output discretization**: Compute 1/3 and 2/3 quantiles separately within each retained session over all valid sampled values, then `searchsorted(..., side='right')`. This yields low/medium/high relative behavior states and mirrors reference per-session scaling. Thresholds are metadata. If quantiles tie, use deterministic stable rank tertiles as an explicit fallback so all requested classes exist.
6. **Mixed temporal types**: Store every input/output trial as a 2D array (`input`: 2×100, `output`: 4×100). Per-trial values are repeated in time because NumPy cannot combine scalar and time-varying rows otherwise; their semantics remain per-trial.
7. **Block counter**: Zero-based trial index within each native probability block, computed on all original trials before QC. This retains true experimental progression even when invalid trials are removed.
8. **Missing data**: Exclude a trial unless both continuous streams cover the complete 2-s window; exclude sessions with fewer than two jointly valid trials. Do not impute outputs. Current pre-check predicts 444 retained sessions and 188,925 trials, subject to exact processing validation.
9. **Probe merging**: Concatenate retained units from simultaneous probes in deterministic `probe_name` order; do not treat probes as independent sessions, matching both papers' session-level behavioral dependence rationale.
10. **Numeric types**: Neural/input arrays float32; output arrays int64; metadata indices int64. These satisfy validation and bound pickle size.

### Planned Sanity Checks
- [ ] Release membership/totals: 459 EIDs, 699 probes, 621,733 raw clusters, 75,708 `label>=1` units before behavior exclusions.
- [ ] Trial mask: independently recompute required-event, latency, duration, no-go, and coverage masks from raw trial/stream files; compare retained original indices with `np.allclose`.
- [ ] Neural: for at least three concrete session/trial/neuron/time-bin coordinates, independently histogram raw `spikes.times` for the matching raw cluster and require `np.allclose` to converted counts; also check per-trial count conservation over all retained units.
- [ ] Inputs: require `np.allclose` for the entire time grid and raw-sequence block counter on concrete trials.
- [ ] Outputs: require `np.allclose` for raw choice/prior mappings and for independently preprocessed/interpolated wheel/ME values after application of recorded tertile thresholds on concrete trials.
- [ ] Alignment: plot trial-average neural counts, continuous pre-discretization behavior, categorical behavior, and event time on the common grid for two sessions; inspect stimulus time zero and class transitions.
- [ ] Shapes/ranges: exactly 100 bins; counts finite/nonnegative/integer-valued; time [−0.48,1.50]; block counter nonnegative; output ranges `[0,1]`, `[0,2]`, `[0,2]`, `[0,2]`.
- [ ] Distributions: continuous-output class fractions should be approximately one third per session/global (allow deviations from ties); choice/prior distributions compared with curated raw trials.
- [ ] Region/unit order: independently map raw peak-channel atlas IDs for selected units and compare every per-session region index/label correspondence.
- [ ] Edge cases: trials at first/last recording samples, sessions with right-camera fallback, multiple probes, non-contiguous cluster IDs, tied quantiles, and sessions with missing/uncovered motion streams.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with the required positional output path, default `--full`, `--sample`, and `--show-processing`. It performs release preflight validation, reference trial filtering, exact wheel filtering, camera fallback/timestamp correction, joint coverage filtering, behavior interpolation/discretization, direct spike-memmap binning, Beryl annotation, provenance-rich metadata, structural assertions, plotting, and pickle serialization. `python3 -m py_compile` and CLI help smoke tests pass.

Code inefficiencies identified:
- Loading all 21.1 billion spikes or materializing full-session spike tables would be wasteful.
- Per-spike Python loops and one SciPy interpolator object per behavior/trial would be slow.
- Building neural counts directly as float32 doubles construction memory relative to the safe count range.
- Reprocessing sessions that lack a required motion-energy stream would waste wheel/spike work.

Code speedups added:
- Read sorted spike arrays as memmaps and use `searchsorted` to touch only requested 2-s trial windows.
- Map cluster IDs and use vectorized flat `bincount` per trial/probe.
- Construct counts as uint16 and cast each final trial to required float32.
- Use vectorized NumPy interpolation with an explicit final-sample extrapolation matching reference `interp1d` behavior.
- Reject missing/uncovered behavior before loading/binnning spikes.
- Process one session at a time to bound peak memory and print per-session/serialization timing.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 340 session-units |
| Neurons / session | 76, 264 (mean 170) |
| Subjects | 1 (`NYU-11`) |
| Sessions / subject | 2 |
| Trials (total) | 651 |
| Trials / session | 407, 244 |
| Time input range | [-0.48, 1.50] s (validator rounds display to [-0.5,1.5]) |
| Trial number in block range | [0, 89] |
| Choice distribution | [0.482335, 0.517665] |
| Prior distribution | [0.477727, 0.161290, 0.360983] |
| Wheel-speed class distribution | [0.333333, 0.333318, 0.333349] |
| Whisker-ME class distribution | [0.333333, 0.333318, 0.333349] |
| Neural count range / mean | [0, 11] / 0.15923 spikes per unit-bin |
| Neural nonzero fraction | 0.12461 |
| Time bins | Exactly 100 in every trial |

### Processing Plots Review
Reviewed both `processing_<eid>.png` files and validator `sample_trials.png`. Trial curation declines only at documented QC stages; raw wheel movement, filtered speed, ME, neural response, and categorical transitions share the intended stimulus zero; neural population averages show a plausible stimulus-locked rise; thresholds intersect their distributions sensibly; static choice/prior rows remain constant; no truncation, NaNs, empty classes, all-zero trials, or time shifts were observed. The second session's raw wheel response and neural population response begin shortly after stimulus as expected.

`verification_sample_out.txt` reports: valid format, no errors, no warnings, correct dimensions/ranges, 17 Beryl regions, and exact class support. Direct metadata counts agree with conversion log (407/565 and 244/425 retained trials; 76/898 and 264/1,728 good/raw units), so no sample object was silently lost.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Spike memmap window slicing + `bincount`; behavior-first rejection; uint16 construction | Unplotted session work was 0.27 s and 0.29 s for the two sample sessions; plotted runs were 2.00 s and 3.59 s. |
| Skip processing plots in full mode | Reduces sample conversion from 8.83 s to 2.56 s internal total (3.62 s shell wall including imports). |

| Step | Time / Session | Estimated Total Time |
| Release preflight/import | 2.0–3.1 s total | ~3 s |
| Conversion without plots | 0.28 s for sample sessions; scale by 11.84 GiB predicted neural output vs 0.038 GiB sample | ~3 minutes conservative |
| Pickle serialization | 0.04 s for 0.038 GiB | ~13–30 s for ~11.8 GiB |
| Full conversion total | size-scaled from no-plot benchmark | ~4 minutes, comfortably below 15 minutes |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Choice | 0.6213 | 0.5425 |
| Prior probability left | 0.7139 | 0.6742 |
| Wheel speed tertile | 0.5863 | 0.5749 |
| Whisker motion-energy tertile | 0.5782 | 0.5786 |

Training completed on GPU. Loss decreased monotonically from 1.785909 (epoch 1) to 0.740130 (epoch 200); test loss was 0.773312. Every validation accuracy exceeds uniform chance (0.5 for choice; 0.3333 for other outputs). Choice is only modestly above chance on this one-subject/two-session sample, while all three-class targets are well above chance; full-data behavior is evaluated in Steps 11–12.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

The first sequential full run was stopped after the observed cold-I/O rate projected to
approximately 19 minutes (>1.5x the Step 7 estimate), as required by the workflow.  Profiling
showed that spike-array reads, rather than computation, were the bottleneck.  I added a
four-worker `ThreadPoolExecutor` across sessions (with ordered result collection and no
change to numerical processing) and restarted from the beginning.  The optimized run
completed in 349.07 s, including 16.69 s to serialize the result.

### Output Files
- `converted_data.pkl`: 12.559 GiB; 444 sessions, 136 subjects, 188,925 trials,
  73,044 session-units, and 266 Beryl regions
- `conversion_full_out.txt`: created; contains per-session trial/unit counts and timings
- `verification_full_out.txt`: created; verifier completed with no errors. It reports 16
  all-zero neural trials (0.0085% of retained trials), examined further in Step 10.

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 621,733 raw; 75,708 good | all clusters loaded; good label metadata | 621,733 raw; 75,708 good | 73,044 good session-units | Exact for retained sessions |
| Mean neurons/session | 1,355 raw; 165 good (derived) | variable | 165 good (derived) | 164.51 | Subset-consistent |
| Subjects | 139 | release freeze | 139 | 136 | Three mice only in excluded sessions |
| Sessions | 459 | release freeze | 459 | 444 | 14 lack ME; one has no trial-window coverage |
| Trials (total) | not an analysis total | reference trial mask | 296,090 native; 195,781 reference-valid; 188,925 joint coverage | 188,925 | Exact |
| Trials/session (mean) | N/A | variable | 425.51 after all masks | 425.51 | Exact |
| Time input range | -0.5 to +1.5 s window | 20-ms right-edge samples | `[-0.48,1.50]` | `[-0.48,1.50]` (verifier rounds display) | Exact |
| Trial-in-block range | N/A | block sequence | `[0,98]` | `[0,98]` | Exact |
| Choice distribution | approximately balanced | left/right retained | `[0.492,0.508]` | `[0.492,0.508]` | Exact |
| Prior distribution (0.2/0.5/0.8) | biased blocks plus unbiased transitions | `probabilityLeft` | `[0.417,0.141,0.442]` | `[0.417,0.141,0.442]` | Exact |
| Wheel classes | required categorical target | continuous absolute speed | session tertiles | `[0.333,0.333,0.333]` | Exact by construction |
| Whisker classes | required categorical target | continuous left ME, right fallback | session tertiles | `[0.333,0.333,0.333]` | Exact by construction |

The 266 labels are Beryl-mapped labels represented by curated neurons in retained sessions.
This is compatible with the paper's 279 recorded Allen areas (and 241 areas meeting its
separate sufficient-data criterion): these are different selection and atlas summaries.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` ends with `Data verification
   complete` and contains no errors.  Its only warnings are 16 all-zero neural windows
   (0.0085% of 188,925 trials).  Direct raw-spike inspection confirmed these windows truly
   contain no spikes from the session's good units.  Dropping otherwise valid trials merely
   to suppress this warning would introduce neural-activity-dependent trial selection, so
   they are retained and documented.
2. **Independent raw-file `np.allclose` checks** (`cache/sanity_checks.py`): nine complete
   100-bin vectors (3 raw trials x 3 raw clusters) match neural output; the 100-point time
   grid and three native probability-block counters match inputs; raw choices and priors
   match static outputs; independently filtered/interpolated wheel and camera ME, independently
   recomputed tertiles, and 600 dynamic class labels match outputs.  A warned zero window
   was independently confirmed from its raw probe.  All checks passed.
3. **Reference-code comparison**:

   | Stage | Reference implementation | Conversion implementation | Review result |
   |---|---|---|---|
   | Loading | `prepare_data`, `load_spiking_data`, `merge_probes` | release CSV drives native ALF/parquet/npy loads; simultaneous probes concatenated | Same release identity and probe merge; direct local loading is equivalent |
   | Filtering | `load_trials_and_mask(min_rt=.08,max_rt=2,max_trial_len=10,exclude_nochoice=True)`; good labels saved but all clusters cached | identical event/RT/duration/no-choice mask; `label>=1` units | Trial logic exact; good-unit restriction intentionally follows data-paper curation and keeps the 12.6-GiB decoder tractable |
   | Alignment | `stimOn_times`, window `(-.5,1.5)` | same event/window | Exact |
   | Binning | `binsize=.02`; behavior sampled at bin right edges | floor-indexed half-open spike bins and identical right-edge interpolation/extrapolation | Raw vector checks exact |
   | Input construction | reference used task/behavior features for encoding; `probabilityLeft` available | mandated time and native trial-in-block counter | Task-required difference; block counter computed before trial filtering |
   | Output construction | choice/block plus continuous wheel/ME; wheel is absolute filtered velocity; left ME preferred | mandated choice/prior classes plus per-session behavior tertiles | Source processing exact; categorical conversion is task-required |

   `align_spike_behavior` retains only the final behavior's mask because of its loop scope;
   requiring complete coverage for **both** requested time-varying targets is an intentional
   correction, not replication of that evident bug.
4. **Key-statistics comparison** (`cache/full_audit.py`): raw cluster metric files reproduce
   459 sessions, 699 probes, 139 subjects, 621,733 raw units, and 75,708 good units exactly.
   The converted EIDs are a strict release subset; their independently summed metric labels
   reproduce 73,044 units exactly.  Exhaustive converted counts reproduce 444 sessions,
   188,925 trials, and the Step 9 class fractions.  All paper/data differences are completely
   accounted for by documented behavior availability and good-unit filtering.
5. **Edge cases**: exhaustive checks verified >=2 sorted native trial indices per session,
   indices in native bounds (including first/last retained trials), `(neurons,100)`, `(2,100)`,
   and `(4,100)` shapes, strict `-0.48` through `+1.50` time samples, constant per-trial
   input/output rows where required, matching region-vector lengths, all output classes
   present globally, and 16 expected sparse zero windows.  No off-by-one mismatch was found.

### Issues Found and Resolved
- **Audit harness path**: the first sanity-check run used `parents[3]` and looked one level
  above the session directory.  Changed it to `parents[2]`, reran the complete raw suite,
  and all checks passed.  This was isolated to the new audit script; converter output did
  not change.
- **Full-run I/O bottleneck**: resolved in Step 9 with ordered four-thread session loading;
  numerical output and ordering remain unchanged.
- **Conversion defects found**: none.  Therefore no reconversion was required after the
  complete second-pass review.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: **Yes**. Training loss fell monotonically from 2.033199 (epoch 1) to
  0.729643 (epoch 200); held-out loss was 0.754193. Training used CUDA, 150,958 training
  trials and 37,967 held-out trials. `predictions.png` was created and visually inspected;
  signals are aligned on the shared 100-bin grid and dynamic predictions track class changes.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| choice | 0.6429 | 0.6196 | chance 0.5000 |
| prior_probability_left | 0.6886 | 0.6724 | chance 0.3333 |
| wheel_speed | 0.6239 | 0.6164 | chance 0.3333 |
| whisker_motion_energy | 0.6078 | 0.6012 | chance 0.3333 |

The required command completed normally and `train_decoder_full_out.txt` ends with
`train_decoder.py finished successfully.`

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved validation balanced accuracy | Chance / ratio | Closest expectation from papers |
|---|---:|---:|---|
| Choice | 0.6196 | 0.5000 / 1.239x | Method-paper Fig. 5 shows approximately 0.60-0.83 across its five selected regions and models. Data-paper Fig. 5 shows region-wise null-corrected balanced accuracy on a 0.02-0.28 scale; this decoder's raw excess over chance is 0.1196, within that range. |
| Prior probability | 0.6724 | 0.3333 / 2.017x | Method-paper Fig. 5 reports continuous-prior Pearson correlations of roughly 0.2-0.7, not 3-class balanced accuracy; the metrics are not numerically interchangeable, but performance is of the expected magnitude. |
| Wheel speed | 0.6164 | 0.3333 / 1.849x | Method-paper Fig. 5 reports continuous-signal R2 around 0.4-0.53; data-paper Fig. 7 maps null-corrected continuous speed R2 on 0.01-0.30. The requested tertile accuracy is not directly comparable. |
| Whisker motion energy | 0.6012 | 0.3333 / 1.804x | Method-paper Fig. 5 reports continuous-signal R2 around 0.37-0.50. The requested tertile accuracy is not directly comparable. |

I reviewed every decoding scale or exact accuracy printed in the supplied papers, including
targets outside this conversion.  The data paper's maps span approximately 0.01-0.17
(stimulus null-corrected balanced accuracy), 0.02-0.28 (choice), 0.02-0.46 (feedback),
0.01-0.30 (wheel-speed R2), and 0-0.43 (wheel-velocity R2).  The method paper reports the
four relevant Figure 5 metric ranges above, a 2% relative decoding-accuracy improvement in
Figure 2, and exact lick accuracies 0.71 (linear) versus 0.90 (BMM-HMM) for an unrelated
Allen task.  Figure 5's printed relative improvements span about 0-16% for choice, -37-136%
for prior, 0-16% for wheel, and 4-13% for whisker across regions/models.  These are not
appropriate absolute thresholds for this decoder because the papers use region-selected
models and continuous targets for three of the four variables.

`cache/accuracy_review.py` reproducibly parsed the full log.  Every held-out result exceeds
chance.  Training/validation ratios are 1.038 (choice), 1.024 (prior), 1.012 (wheel), and
1.011 (whisker), far below the 1.5 overfitting threshold.

Choice is the only output below 1.5x chance, so I performed all requested debugging:

1. Raw trial-table choices were checked for three concrete trials with `np.allclose`.
2. The stimulus-aligned neural/output processing plots and held-out prediction plot were
   inspected; their shared 100-bin axes have no temporal shift.
3. Its `[0.4921, 0.5079]` class split has ample variation.
4. Raw spike vectors reproduce converted vectors exactly. `label>=1` implements the data
   paper's explicit rule excluding units that fail amplitude/noise/refractory criteria.
5. The -0.5/+1.5-s stimulus window and 20-ms binning match the method paper's choice setup.

Using all multi-unit clusters might change accuracy, but would contradict the data paper's
explicit analysis curation; moving the window to movement onset would change the mandated
stimulus alignment.  Because the achieved value also lies within the paper's reported
region-wise range, neither change is justified as a conversion fix.

### Issues Found and Resolved
- No accuracy was below chance, no train/validation gap exceeded 1.5x, and the targeted
  choice investigation found no label, alignment, variation, filtering, or processing bug.
- The apparent metric discrepancies for prior/wheel/whisker are resolved by the requested
  categorical discretization: paper values are Pearson correlation or R2, whereas the
  validator reports balanced accuracy.
- No conversion issue was found, so iteration/reconversion was not warranted.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created with loading instructions, format, mappings, statistics, and results
- [x] `cache/` folder created; `README_CACHE.md` inventories audits, logs, benchmarks, and paper renders
- [x] All investigation scripts and generated bytecode organized under `cache/`
- [x] All 11 required deliverables checked present and non-empty
- [x] All 14 workflow steps checked `COMPLETE`; no template placeholders remain

Final root contents retain source/reference materials, required datasets and logs, the
converter, user documentation, and requested processing/prediction plots. No raw project
data were modified.
