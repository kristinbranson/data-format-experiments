# Dataset Conversion Notes

## Overview
- **Dataset**: IBL brain-wide map of neural activity during complex behaviour
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `code`
- `data`
- `dataarchitecture.pdf`
- `datapaper.pdf`
- `decoder.py`
- `docker-compose.yaml`
- `ibl_docs`
- `methodpaper.pdf`
- `methods.txt`
- `stage_cache.sh`
- `train_decoder.py`

Environment verification: Python 3.13.15; NumPy 2.3.5; PyTorch 2.6.0+cu124 imported successfully.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_spiking_data` | `code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Uses `brainbox.io.one.SpikeSortingLoader`/ONE to load spikes, clusters and channels for a probe and merge cluster information. |
| `merge_probes` | `code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Concatenates spikes/clusters from all probes in one session while keeping cluster IDs unique. |
| `load_trials_and_mask` | `code_zhang2025/src/utils/ibl_data_utils.py` | LOADING/CURATION | Loads the ALF trials object through ONE and constructs a trial-validity mask, including finite event times and a maximum trial duration criterion. |
| `load_target_behavior` | `code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Loads wheel velocity/speed and camera-derived motion energy/pupil streams using ONE objects and returns timestamps plus values. |
| `load_anytime_behaviors` | `code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Loads continuous wheel/camera behavior targets, optionally in parallel. |
| `load_ibl_dataset` | `code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Loads every probe, trial table/mask, and continuous behaviors for an EID; returns neural, behavior, metadata, and trial dictionaries. |
| `bin_spiking_data` / `bin_behaviors` | `code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Aligns streams to a selected trial event and bins/interpolates them into fixed trial windows. |
| `align_spike_behavior` | `code_zhang2025/src/utils/ibl_data_utils.py` | CURATION | Removes trials missing requested behavior and checks neural/behavior trial count agreement. |
| `create_dataset` | `code_zhang2025/src/utils/dataset_utils.py` | PROCESSING | Stores binned spikes sparsely with behavior, metadata, bin size, interval length and BWM session metadata. |
| `SingleSessionDataset` | `code_zhang2025/src/utils/data_loader_utils.py` | PROCESSING | Reconstructs dense binned spikes and exposes trial samples/targets to PyTorch. |
| `SingleSessionDataModule` / `MultiSessionDataModule` | `code_zhang2025/src/utils/data_loader_utils.py` | CURATION | Builds train/validation/test splits for single- and multi-session decoders. |
| caching main program | `code_zhang2025/src/0_data_caching.py` | PROCESSING | Queries/iterates BWM sessions, loads with ONE, bins all streams, and writes decoder cache datasets. |

### Notes
- Relevant methods repository: `/app/code/code_zhang2025`; `/app/code/ibllib` provides the bundled ONE/brainbox implementation and examples.
- The caching configuration uses `interval_len=2`, `binsize=0.02` s, `align_time='stimOn_times'`, and `time_window=(-0.5, 1.5)`. Thus reference neural activity is spike counts in 100 bins of 20 ms centered relative to stimulus onset.
- Spikes and cluster metadata are loaded through `SpikeSortingLoader`; all probes returned by `one.eid2pid(eid)` are merged. Cluster acronyms provide regions.
- Reference metadata defines good clusters as `clusters['label'] >= 1`; this is the electrophysiology cell-quality filter. There is no calcium imaging and no dF/F computation.
- Trial variables include `choice`, `probabilityLeft` (called block), reward, and signed contrast. Continuous targets include wheel velocity/speed and left/right whisker motion energy.
- The decoder code is configuration-driven and evaluates held-out trials; multi-session decoding preserves per-session unit counts and EID indexing.
- Important caveat: `align_spike_behavior` uses Python list `and` rather than elementwise logical conjunction, and only the last loop-created behavior mask survives. Conversion code must not copy this defect; it should explicitly combine all validity masks with NumPy logical operations while preserving the intended curation.
- No direct filesystem data loader from the reference repository will be used for source data; the conversion will follow its ONE/brainbox loading path as required.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Data are an IBL ONE cache staged at `/app/data/one_cache` and accessed exclusively with `one.api.ONE` plus `brainbox.io.one` loaders. No source data files were opened directly.
- `ONE.load_cache(tag="Brainwidemap")` exposes release tables with 480 sessions, 143 subjects, 12 labs, and 76,563 indexed datasets (release table dated 2025-02-20). Of these, 459 sessions index trials, wheel and spikes; 445 also index camera ROI motion energy and therefore contain all requested source modalities.
- Session data use ALF objects. Trials are primarily `_ibl_trials.table.pqt`; wheel uses `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`; camera streams include `leftCamera/rightCamera.times` and `ROIMotionEnergy`; spike sorting is under `alf/probeXX/pykilosort` with spikes arrays, cluster metrics, channels and atlas locations.
- Complete trial objects contain `stimOn_times`, `choice`, `probabilityLeft`, contrasts, feedback, movement and interval fields. A representative trials table had shape `(692, 13)` before supplemental attributes were merged; `one.load_object` returned 18 aligned trial attributes.
- A representative wheel object had 890,708 timestamp/position samples; representative left/right camera timestamp streams had 299,749/742,128 samples. Continuous stream lengths vary by session.
- A representative probe contained 50,570,481 spikes and 1,239 clusters; merged cluster metadata include graded `label`, channel and atlas acronym. Label distribution was 52/522/463/202 at 0, 1/3, 2/3, 1 respectively.
- ONE release metadata has a revision anomaly: canonical unrevisioned trials tables are marked non-default while supplemental trial arrays are default. Cache-backed `remote` query mode plus an in-memory normalization of the unique trials-table `default_revision` flag makes ONE resolve the complete object. Local query mode alone returns only supplemental fields. This normalization will be explicit in conversion code; data remain loaded through ONE.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 566,533 clusters in metric tables; 68,873 with reference criterion `label >= 1` among loadable probe metrics |
| Neurons / session | all clusters mean 1,273.1, median 1,221; label-qualified mean 154.8, median 138 |
| Subjects | 143 in full release table; 136 among 445 complete-modality sessions |
| Sessions / subject | 445 complete sessions / 136 subjects (mean 3.27; variable) |
| Trials (total) | 287,040 native trials in 445 complete-modality sessions |
| Trials / session | min 401, mean 645.0, median 600, max 1,525 |

### Available Variables and Native Types
- Neural: spike `times` (float), `clusters` (integer IDs), amplitudes/depths; cluster metrics DataFrame, channel coordinates, atlas IDs/acronyms.
- Trial/task: choice values `-1, 0, +1`; probabilityLeft values `0.2, 0.5, 0.8`; contrasts, reward/feedback, stimulus/go-cue/movement/response times, intervals.
- Continuous behavior: wheel position and timestamps (wheel velocity/speed derived by brainbox/reference utilities); left/right camera ROI motion energy and timestamps; DLC features are also indexed in many sessions.
- Native choice counts across complete sessions: -1=141,904, 0/no-go=1,112, +1=144,024. Prior counts: 0.2=119,737; 0.5=40,050; 0.8=127,253.
- Thirty-seven of 676 probe metric loads raised `OSError` during the aggregate scan. These probes/sessions require explicit robust handling and logging during conversion; no silent substitution is acceptable.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 621,733 units; 75,708 well-isolated | Data paper methods: “459 sessions, 699 insertions and 621,733 neurons”; “75,708 were considered well-isolated neurons.” |
| Neurons / session | approximately 165 well-isolated on average before analysis-specific exclusions | 75,708 / 459, computed from paper totals; paper additionally requires >=5 good neurons per region/session. |
| Subjects | not stated in the extracted methods passage; release table has 143 | ONE Brainwidemap release table (Step 2); paper totals focus on sessions/insertions/neurons. |
| Sessions / subject | variable | ONE release and experimental design. |
| Trials (total) | not stated as one paper-wide total; source cohort has 287,040 before trial exclusions | Paper specifies session/trial criteria; Step 2 API scan supplies native count. |
| Trials / session | >=250 for release inclusion; source complete-modality median 600 | “Sessions were included ... if the mice performed at least 250 trials.” |
| Neural data time bin | 20 ms for time-varying wheel decoding; methods-paper code cache also uses 20 ms | “We averaged wheel values in nonoverlapping 20-ms bins ... Spike counts were similarly binned.” |
| Behavior data time bin | 20 ms after interpolation/averaging for decoding | Same wheel-decoding passage; native camera rates differ. |
| Reward rate | no single paper-wide fraction stated in inspected text | Session inclusion instead requires >=90% correct at 100% contrast in both blocks and >=3 incorrect trials. |
| Insertions | 699 | Data paper methods. |
| Sessions | 459 released | Data paper methods. |
| Choice native balance | binary left/right after excluding missing/no-go | Choice is treated as binary target; class-frequency weighting used in reference decoding. |
| Prior values | 0.2, 0.5, 0.8 | Biased-choice task and source trials table. |

### Processing Details
- Task is a two-alternative visual choice task with blockwise left-stimulus prior (`probabilityLeft`). Choice, stimulus side and feedback are binary logistic-regression targets in the data paper; balanced accuracy and inverse-class-frequency sample weights are used.
- Neural regressors are binned spike counts. For time-varying wheel targets, wheel values and spike counts are placed in non-overlapping 20 ms bins. The paper's wheel analysis aligns -0.2 to +1.0 s around first movement and uses a causal 11-bin neural history; the supplied methods-paper caching code instead uses the decoder-task-applicable stimulus alignment, -0.5 to +1.5 s around `stimOn_times`, at 20 ms resolution.
- Multiple probes in one session are not treated independently: neurons in the same session/region are combined across probes.
- Left, right and body cameras run at 60, 150 and 30 Hz respectively. Whisker motion energy is the mean absolute adjacent-frame difference in a DLC-anchored whisker-pad ROI and retains the temporal resolution of its camera before resampling.
- The task specification requires wheel speed and whisker motion energy to become three-class categorical time series. This is an intentional difference from the paper's continuous Lasso/R2 wheel decoder.

### Curation Steps

**Neuron curation rules**:
- Exclude insertions with whole-recording RIGOR failure/major artifacts, unresolved histology tract, or unresolved alignment.
- Well-isolated units satisfy all three stated RIGOR metrics: amplitude >50 µV, noise cutoff <20 µV, and refractory-period-violation criterion. The reference code operationalizes this as merged cluster `label >= 1`.
- Paper region-level analysis further restricts to Allen CCF grey matter regions with >=5 well-isolated neurons per session and recordings in >=2 sessions. For the requested session-level neural decoder, retain all well-isolated units with valid atlas acronym; do not discard a whole session solely because an individual region lacks cross-session replication unless required after consistency review.

**Trial curation rules**:
- Release sessions: >=250 trials, >=90% correct on 100% contrast trials for left and right blocks, >=3 incorrect-choice trials after exclusions, and hardware QC thresholds.
- Exclude trials missing choice, probabilityLeft, feedbackType, feedback time, stimulus-on time, or firstMovement time.
- Exclude trials where first wheel movement occurs outside 0.08–2.00 s after stimulus onset.
- Reference loader additionally enforces finite required events and maximum trial duration (10 s). No-go choice 0 is not a valid requested binary class and will be excluded.

### Decoders Trained
| Decoded variable | Accuracy |
|---|---|
| Choice/task variables in data paper | Evaluated by balanced accuracy against session-specific null distributions; no single global numerical accuracy is reported in the inspected methods text. |
| Wheel speed/velocity in data paper | Continuous Lasso evaluated by R², not directly comparable to required three-bin classification. |
| Methods paper | Compares single-session and cross-session latent/reduced-rank choice decoders; reported metrics are architecture/experiment-specific and not a direct expected value for the provided generic four-output decoder. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Release session count | Caching pipeline iterates BWM release/query sessions | Brainwidemap release table has 480 indexed sessions; 459 have trials/spikes/wheel and 445 also have camera motion energy | Final analyzed release has 459 sessions and 699 insertions | Use the 459 task/ephys sessions as the candidate cohort, then require a valid whisker stream for this decoder output; expected usable ceiling is 445 before trial/neuron QC. Log every excluded session. The extra 21 release-table sessions are passive-only or lack required task/ephys objects. |
| Probe count | Loads every `one.eid2pid` probe and merges probes within a session | 676 probe collections in the 445 complete-modality sessions; 37 metric paths raised OSError in exploration | 699 insertions across 459 analyzed sessions | Counts are consistent after removing 14 camera-incomplete sessions and accounting for failed/missing probe payloads. Conversion will use exact ONE-indexed probe collections, skip only probes that cannot be loaded, and record failures. |
| Unit count/QC | Merged cluster `label >= 1` defines good clusters | 566,533 metric rows and 68,873 label-qualified clusters loaded for complete sessions, excluding 37 failed metrics | 621,733 total; 75,708 well-isolated | Apply `label >= 1`, matching code's operational version of RIGOR. The lower preliminary totals are explained by cohort restriction and failed probe loads; final converted count must be compared again. |
| Trial validity | `load_trials_and_mask` checks required columns, finite/non-null values, finite intervals, and duration <=10 s | Native choice includes 1,112 no-go zeros; key event NaNs occur in some sessions | Also requires first movement latency 0.08–2.00 s and key event presence | Combine both sources: finite required fields, interval duration <=10 s, movement latency in [0.08,2.00], binary choice only, valid prior, and complete behavior windows. |
| Alignment/binning | Cache config uses `stimOn_times`, [-0.5,1.5] s, 20 ms; `bin_spiking_data` uses `bincount2D`; behavior is interpolated to bin endpoints | Trial and continuous timestamps share session clock | Paper's wheel decoder uses movement alignment, but decoder task explicitly requires stimulus alignment; 20 ms bins are used | Follow decoder-task alignment and methods-code cache: stimulus onset, [-0.5,+1.5] s, 20 ms, exactly 100 bins. This is a justified task-required difference from movement-aligned paper analysis. |
| Wheel target | `load_target_behavior` derives velocity from wheel position/timestamps and takes absolute value for speed | No stored velocity; position/timestamps exist in 459 sessions | Wheel speed is continuous and averaged in 20 ms bins | Use brainbox/reference velocity derivation, absolute speed, interpolate/average on the same 20 ms grid, then discretize into three classes as required. |
| Whisker target | Loads left/right camera ROI motion energy separately; generic whisker loader falls back left then right | Camera motion energy indexed for 445 sessions; camera rates differ | Whisker-pad motion energy is frame-difference mean at camera temporal resolution | Prefer left camera when valid, fall back to right exactly as reference helper; resample to common 20 ms bins before three-class discretization. Record camera side per session. |
| ONE revisions | Reference assumes `one.load_object` resolves release objects | Canonical trials table is marked non-default; local mode returns only supplemental fields | Not discussed | Stay in cache-backed remote query mode and normalize the canonical trials-table default flag in memory before ONE loads. This fixes metadata resolution without direct file access. Explicitly select latest revisioned spike/cluster paths and log warnings. |
| Region inclusion | Code carries all cluster acronyms/good labels into cache | Atlas acronyms available from channels/merged clusters | Main paper region analyses require grey matter, >=5 good units/session-region, >=2 sessions | Use good units with valid non-root atlas acronym. Do not impose analysis-specific >=5/region or >=2-session threshold because target format is session-level and should preserve neurons; document this justified difference. |

### Final Reconciled Understanding
- Load exclusively through ONE and brainbox; merge all valid probes in each session and retain only `label >= 1` clusters.
- Build fixed 100-bin trials from -0.5 to +1.5 s around stimulus onset using 20 ms spike counts and aligned behavioral summaries.
- Apply the union of paper and code trial-QC criteria, then require all four outputs to be defined for the complete window.
- Preserve per-session grouping and subject identity; map every retained unit to its Allen acronym.
- Treat paper totals as the full-release reference and explain expected reductions from missing requested modalities, trial QC, and probe-load failures.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spikes.times`, `spikes.clusters`; merged cluster QC | `neural[session][trial]` | Retain clusters with merged `label >= 1` and valid atlas acronym; merge probes with unique unit indices; histogram spike times into 100 half-open 20 ms bins from -0.5 to +1.5 s around `stimOn_times`; store `(n_neurons,100)` float32 counts | `load_spiking_data`, `merge_probes`, `bin_spiking_data` | Counts, not smoothed rates, match caching code and preserve Poisson information. Last bin includes only events before +1.5 s. |
| fixed bin centers | `input[...,0,:]` | `-0.49, -0.47, ..., 1.49` seconds relative to stimulus onset | caching config | Continuous time-varying decoder input, identical for all trials. |
| `trials.probabilityLeft` contiguous runs | `input[...,1,:]` | Trial index within each contiguous prior block, zero-based, repeated over 100 bins | task structure | Counter resets whenever probabilityLeft differs from preceding native trial. It is computed before trial filtering so omitted bad/no-go trials do not compress behavioral block position. |
| `trials.choice` | `output[...,0,:]` | IBL `-1` (left) -> 0; `+1` (right) -> 1; repeat over 100 bins; exclude choice 0/NaN | trial loader | Required binary convention. |
| `trials.probabilityLeft` | `output[...,1,:]` | `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`; repeat over 100 bins | `load_ibl_dataset` (`block`) | Reject unexpected/NaN prior values. |
| wheel timestamps/position | `output[...,2,:]` | Derive velocity with brainbox/reference wheel utility, take absolute value, summarize/interpolate on same 20 ms bins, discretize low/mid/high by session-level 1/3 and 2/3 quantiles | `load_target_behavior('wheel-speed')`, `bin_behaviors` | Quantile edges are fit using finite samples from retained trial windows only. `np.digitize(..., right=False)` produces classes 0/1/2; degenerate edges are nudged/documented. |
| left or right camera times + ROI motion energy | `output[...,3,:]` | Prefer left camera when complete; otherwise right; interpolate/bin to 20 ms grid; discretize by session-level tertiles | `load_target_behavior('*-whisker-motion-energy')`, `bin_behaviors` | One camera side per session avoids mixing scales/rates. Reject trials whose complete window is outside stream support or contains non-finite values. |
| session subject | `subjects`, `subject_idx` | Stable sorted unique subject names and integer lookup | ONE session release table | Session order is stable by EID unless a subset manifest specifies order. |
| merged cluster `acronym` | `brain_regions`, `brain_region_idx` | Stable sorted unique Allen acronyms; one index per retained unit | `SpikeSortingLoader.merge_clusters` | Exclude empty/`void`/`root`/NaN acronyms; retain fine Allen acronyms rather than collapsing regions. |

### Key Decisions
1. **Window and temporal grid**: Use the methods-code stimulus-aligned cache specification `[-0.5,+1.5)` s at 20 ms (100 bins). This follows the explicit decoder task and matches reference caching code.
2. **Neural representation**: Store raw spike counts as float32. The reference cache bins counts; no smoothing, z-scoring, or firing-rate conversion is applied before saving.
3. **Unit filtering**: Apply `label >= 1`, matching reference code's operational well-isolated-unit rule, plus valid anatomical assignment. Merge all valid probes within each session.
4. **Trial filtering**: Require finite choice, prior, feedbackType/time, stimOn and firstMovement; binary choice; prior in {0.2,0.5,0.8}; interval <=10 s; movement latency 0.08–2.00 s; and complete finite wheel/whisker window. Require at least two retained trials and at least one retained neuron per session.
5. **Trial number in block**: Count native trials since the current contiguous probabilityLeft run began, zero-based. Compute on the unfiltered trial sequence because it is an experimental variable, not an index into the converted subset.
6. **Mixed output dimensions**: Use one `(4,100)` integer array per trial. Per-trial choice/prior are repeated across time, allowing all outputs to share the required temporal dimension and avoiding object/ragged output arrays.
7. **Behavior resampling**: Use interpolation at reference bin centers/ends following `bin_behaviors`; wheel speed derives from native wheel position and timestamps. No extrapolation outside recorded support.
8. **Three-bin discretization**: The task gives class count but no physical thresholds. Session-level tertiles provide balanced classes despite rig/camera scale differences and are fit without using neural activity or validation labels. Save thresholds per session in metadata for reproducibility.
9. **Whisker camera fallback**: Prefer left, then right, matching reference generic whisker loader. Do not average cameras because rates, ROI geometry, and scale differ.
10. **ONE cache handling**: Use cache-backed remote query mode and repair only the release table's canonical trial-table default flag in memory. Resolve probe collections from ONE dataset metadata; never open scientific files directly.
11. **Subset control**: If the staged environment supplies a non-scientific run-limit manifest, it may select EIDs for computational feasibility; all scientific arrays for those EIDs still load through ONE/brainbox. The selection and counts will be reported explicitly.

### Output Schema Details
- `input_names = ['time_since_stimulus_onset', 'trial_number_in_block']`.
- `output_names = ['choice', 'prior_probability_left', 'wheel_speed_bin', 'whisker_motion_energy_bin']`.
- `output_values = [['left','right'], ['0.2','0.5','0.8'], ['low','medium','high'], ['low','medium','high']]`.
- Metadata includes 20.0 ms bins, stimulus-onset alignment, offsets -0.5/+1.5 s, 100 bins, QC criteria, bin edges, camera side, EIDs, subjects, probes, retained/native trial counts, and exclusion reasons.

### Edge Cases
- Multiple probes: remap each probe's cluster IDs into one session-global unit axis; never assume raw cluster IDs are contiguous.
- Revision ambiguity: select exact latest revision paths through ONE metadata and record the revision.
- Failed probe: log EID/probe/error; retain the session only if another valid probe provides at least one good unit.
- Missing left whisker stream: use complete right stream; exclude session if neither side works.
- Repeated quantile thresholds: create monotonic thresholds with `np.nextafter`; record class counts and warn if any class remains empty.
- Empty/short sessions after QC: exclude, with reason; decoder requires >=2 trials.
- Time boundaries: use common edges `stimOn + np.arange(-0.5,1.5+0.02,0.02)`; verify exactly 101 edges and 100 bins.

### Planned Sanity Checks
- [ ] ONE-loaded spike times vs converted neural: for at least 3 session/trial/unit cases, independently count spikes between exact bin edges and require `np.allclose(raw_counts, converted_counts)`.
- [ ] Input time: require every trial row 0 to equal bin centers with `np.allclose`; independently recompute three block counters from ONE trials and compare with row 1.
- [ ] Trial outputs: independently map raw ONE choice/prior for at least 3 trials and compare repeated converted rows with `np.allclose`.
- [ ] Continuous outputs: independently derive wheel speed and interpolate raw camera motion energy for at least 3 trials; apply saved thresholds and require `np.allclose` to converted classes.
- [ ] Shape invariant: neural time dimension, input time dimension and output time dimension are all 100 for every trial; session trial counts agree across all lists.
- [ ] Distribution checks: choice approximately balanced; all priors and all three behavior classes represented; report fractions globally and per session.
- [ ] Reference totals: reconcile candidate/retained sessions, probes, trials and units against 459/699/621,733/75,708 paper totals and Step 2 counts.
- [ ] Temporal visualization: overlay binned spike population, wheel speed/classes and whisker energy/classes on the common stimulus-aligned axis for up to two sessions.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with the required positional output path and `--full`, `--sample`, and `--show-processing` options. The script:
- Initializes the staged Brainwidemap cache with ONE and normalizes the malformed trials-table default metadata in memory.
- Uses the methods-paper freeze CSV only for curated EID/PID/probe identifiers; scientific arrays load exclusively via ONE, `SessionLoader`, and `SpikeSortingLoader`.
- Loads complete trials, reference-processed wheel velocity, camera motion energy, merged spike sorting/QC and atlas acronyms.
- Applies paper/code trial and unit QC, bins spike counts into 100 stimulus-aligned 20 ms bins, creates inputs/outputs, discretizes behavior by session tertiles, and stores detailed provenance/QC metadata.
- Produces processing plots for up to two successfully converted sessions and validates core array lengths during processing.
- Passed `python3 -m py_compile` and CLI help checks.

Code inefficiencies identified:
- Full raw release has tens of millions of spikes per probe and 287k native trials; materializing all unfiltered trial-neuron arrays would be prohibitive.
- Generic brainbox `get_spike_counts_in_bins` loops over every interval and assumes cluster-ID layout.
- Loading neural data before behavioral QC wastes work on trials later excluded.

Code speedups added:
- Behavioral and trial validity is computed before neural binning.
- Spike searches use sorted-time `searchsorted`, dense selected-cluster lookup, and `np.bincount` per retained trial.
- Trial behavior interpolation is vectorized across all trial/bin query points.
- Compact dtypes are used: uint16 spike counts, float32 inputs/continuous intermediates, uint8 categorical outputs, int32 indices.
- Probe arrays are concatenated once per session; no repeated pickle writes occur.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 340 |
| Neurons / session | 76, 264 (mean 170) |
| Subjects | 1 (`NYU-11`) |
| Sessions / subject | 2 |
| Trials (total) | 643 retained of 990 native |
| Trials / session | 402, 241 |
| time since stimulus range | [-0.49, 1.49] s at bin centers (validator rounds/displays [-0.5,1.5]) |
| trial number in block range | [0,89] |
| choice fractions | session 1: [0.428,0.572]; session 2: [0.577,0.423] |
| prior fractions | session 1: [0.445,0.142,0.413]; session 2: [0.539,0.183,0.278] |
| wheel class fractions | [0.333,0.333,0.333] per session (up to rounding) |
| whisker class fractions | [0.333,0.333,0.333] per session (up to rounding) |
| Matrix shapes | neural `(76 or 264,100)`, input `(2,100)`, output `(4,100)` |
| Dtypes | neural uint16, input float32, output uint8 |
| File size | 19.7 MB |

### Processing Plots Review
- Two `processing_<eid>.png` files were generated, one for each sample session, with population spike counts, continuous wheel traces, wheel classes and whisker classes on the common stimulus-aligned axis.
- Plot files open successfully and show the stimulus marker at zero; all panels span the same -0.5 to +1.5 s window. No dimensional or alignment anomaly was reported.
- Session-level tertile class counts are balanced and all classes are represented.

### Format Validation
- `/app/verification_sample_out.txt` was created by the required `--verify-only` command.
- Final validator result: “Data verification complete.” No errors or warnings remain.
- All stream time dimensions are exactly 100; brain-region index lengths match neuron counts; all required categorical ranges are valid.

### Issues Found and Fixed
1. Initial run failed after successful scientific processing because the ONE session table is UUID-indexed but EIDs were strings. Fixed with `uuid.UUID(eid)` and a provenance-only freeze-table fallback.
2. Initial validator run found neural matrices stored as time x neuron. Removed an erroneous transpose and added explicit assertions for `(n_neurons,100)`, `(2,100)`, and `(4,100)`.
3. Re-ran conversion and all validation after each fix; final checks pass.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| Trial/behavior QC before neural binning | Avoids binning 347 invalid sample trials |
| Vectorized behavior interpolation and compact dtypes | Controls CPU and memory use |
| Searchsorted + bincount spike binning | Avoids generic interval-by-unit loops |

| Step | Time / Session | Estimated Total Time |
|---|---:|---:|
| Sample conversion (including plots) | 6.3 s mean (4.5, 7.4 s scientific processing; 12.6 s total) | ~48 min for 459 sessions if serial |

The serial estimate exceeds 15 minutes. Before Step 9, session-level parallel processing or another safe optimization is required and will be benchmarked; full conversion will not begin with the current serial runtime.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None after orientation fix.
- Warnings: None from final format validator. ONE emitted benign revision-selection warnings during source loading; selected data were complete and dimensions were checked.

### Training Progress
- Loss decreased monotonically from >1.5 early in training to 0.743185 at epoch 200; test loss was 0.775497.
- `/app/train_decoder_sample_out.txt` exists and ends with `train_decoder.py finished successfully.`

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-----------------------|-------------------------|
| choice | 0.6219 | 0.5435 |
| prior_probability_left | 0.7096 | 0.6745 |
| wheel_speed_bin | 0.5689 | 0.5693 |
| whisker_motion_energy_bin | 0.5813 | 0.5755 |

All outputs exceed chance (0.5 for choice; 0.3333 for three-class outputs). Prior, wheel and whisker exceed 1.5x chance on validation. Choice is modestly above chance on only two sessions; no conversion/alignment issue is indicated because loss decreases and all temporally varying outputs decode strongly.

---

## Step 9: Full Conversion and Validation
**Status**: NOT STARTED

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
**Status**: NOT STARTED

### Checks Performed
1. [Check]: [Result]

### Issues Found and Resolved
- [Issue]: [Resolution]

---

## Step 11: Full Decoder Training
**Status**: NOT STARTED

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
**Status**: NOT STARTED

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from papers |
| | |

[Analysis of any low accuracies]

### Issues Found and Resolved
- [Issue]: [Resolution]

---

## Step 13: Documentation and Cleanup
**Status**: NOT STARTED

- [ ] README.md created
- [ ] cache/ folder created
- [ ] All files organized

<!-- Step 7 iteration log: Initial sample run completed scientific loading/processing for both sessions but failed at session metadata lookup because the ONE session table is UUID-indexed and EIDs were strings. Fixed lookup with `uuid.UUID(eid)` plus freeze-metadata fallback; no scientific data loading changed. -->
<!-- Step 7 iteration 2: Validator revealed neural matrices were transposed to time x neuron, producing 100/76 and 100/264 instead of neuron x 100 and mismatching brain_region_idx. Removed transpose and added explicit neural/input/output shape assertions. -->
