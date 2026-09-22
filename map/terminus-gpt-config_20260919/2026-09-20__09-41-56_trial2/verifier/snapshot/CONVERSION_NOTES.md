# Dataset Conversion Notes

## Overview
- **Dataset**: Brain-wide neural activity underlying memory-guided movement (provided NWB dataset)
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment verification:
- Python 3.13.15
- NumPy 2.4.4
- PyTorch 2.6.0+cu124
- CUDA available: True
- `/app/CONVERSION_NOTES.md` existence verified before proceeding.

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

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `loadmat` | `VideoAnalysisUtils/preprocessing_utils.py` | LOADING | Loads MATLAB files with `squeeze_me=True`, recursively converts structs/cell arrays to Python dictionaries/lists. |
| `process_one_sess` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING/CURATION | Loads all probes in a session, task/behavior fields, histology, QC indices, intersects ephys with histology, concatenates probes, and selects classifier-QC units by region and side. |
| `sliding_histogram` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Counts spikes in half-open windows centered on requested times and divides by bin width to produce spikes/s. |
| `process_one_area` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Truncates go-cue-relative spike times, computes firing rates, and packages neural/task data for each region/hemisphere. |
| `helper_get_neuron_id_area` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | CURATION | Maps precomputed session-level classifier-QC IDs to histology-matched units and validates CCF annotations/hemisphere. |
| `helper_filter_by_neuron_id` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | CURATION | Applies one neuron selection consistently to rates, spikes, CCF labels/coordinates, and unit metadata. |
| trial mask helper | `VideoAnalysisUtils/population_decoding_utils.py` | CURATION | Paper analyses commonly retain non-early, non-auto-water, non-free-water, responsive trials. |

### Notes
- Repository corresponds to Wang et al., *Brain-wide analysis reveals movement encoding structured across and within brain areas*, using the MAP DANDI dataset.
- This is electrophysiology, not imaging; delta-F/F is not applicable.
- Native reference code loads DataJoint-exported MATLAB probe files. Subject/date/session/probe are parsed from filenames. Behavior/task arrays are copied from the first probe and neural units are concatenated across all probes in a session.
- Neural curation first intersects electrophysiology unit IDs with histology unit IDs. It then uses precomputed classifier-based good-unit indices (`goodunits`) for 14 regions: ALM, Medulla, Midbrain, Striatum, Thalamus, Pons, Cerebellum, Hypothalamus, Hippocampus, Orbital, OtherCortex, Olfactory, CorticalSubplate, and Pallidum; units are also separated by left/right hemisphere. MATLAB indices are converted from 1-based to 0-based.
- Available unit QC metrics include amplitude, presence ratio, amplitude cutoff, ISI violation, average firing rate, and drift metric. The method-paper production invocation uses `qc_mode='classifier'`; therefore classifier QC rather than ad-hoc thresholds is the reference curation.
- Raw exported spike times are already relative to go cue. Lick times and stimulation on/off times are explicitly converted to go-cue-relative values by subtracting `task_cue_time`.
- Reference `sliding_histogram` uses `[left,right)` bins and reports spikes/s. The method-paper preprocessing invocation uses 40-ms windows, 3.4-ms stride, and -3 to +3 s. Its module CLI example uses 100-ms windows/50-ms stride. For this task, the explicit decoder requirement overrides both: contiguous 50-ms bins from -2.5 to +1.5 s.
- Task/behavior variables found: auto-learn status, early report, auto/free water, lick directions and times, report/correctness (1 hit/correct or free-water, 0 error, -1 no response), absolute go-cue time, sample/delay times and durations, stimulation power/type/on/off, and instructed trial type.
- Lick-direction comments conflict in two locations (one says 0 left/1 right, another says 0 right/1 left), so raw data semantics and NWB documentation must resolve this in later steps.
- Paper movement analyses often exclude early-lick, auto-water, free-water, and no-response trials. This conversion must not blindly apply that mask: early lick, no lick, and miss are explicitly required outputs. Any exclusions must be limited to structurally invalid trials and justified after source-data/paper comparison.
- No tongue-position preprocessing was present in the inspected ephys MATLAB pipeline; tongue/video data must be mapped from the provided NWB data after inspecting its native structure.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` is 50 GB and contains 174 NWB/HDF5 session files plus one DANDI metadata YAML.
- Files are organized as `/app/data/sub-<subject>/sub-<subject>_ses-<timestamp>_behavior+ecephys+ogen.nwb`: one directory per subject and one NWB per session.
- YAML identifies **Mesoscale Activity Map Dataset**, DANDI `000363`, version `0.230822.0128`, DOI `10.48324/dandi.000363/0.230822.0128`.
- `intervals/trials` is a DynamicTable. Every session has `start_time`, `stop_time`, 1-based trial number, globally unique trial ID, task/protocol, `trial_instruction` (left/right), `early_lick` (early/no early), `outcome` (ignore/miss/hit), auto/free-water flags, and photostimulation onset/power/duration (`N/A` for control trials).
- `acquisition/BehavioralEvents` contains absolute-time series for go, presample, sample, delay, trial end, left/right licks, and photostimulation start/stop.
- `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking` exists in all 174 sessions. Data shape is `(n_video_frames,3)` with description `(tongue_x, tongue_y, tongue_likelihood)` and matching absolute timestamps; the representative session had 680,500 frames spanning 0 to 2493.305 s.
- `units` contains ragged absolute `spike_times`, unit quality/QC metrics, electrode references, observation intervals, and `is_good_trials`. Representative session dimensions: 1,952 units x 368 trials; 11,495,314 spike timestamps.
- Unit QC variables include `unit_quality`, amplitude, presence ratio, amplitude cutoff, ISI violations, SNR, drift metrics, d-prime, isolation distance, L-ratio, nearest-neighbor metrics, waveform metrics, and more.
- `general/extracellular_ephys/electrodes/location` stores JSON including `brain_regions`; unit electrode indices map units to these labels and CCF x/y/z coordinates.
- Native data are electrophysiology; no calcium imaging/delta-F/F is involved.
- Nine sessions have fewer `is_good_trials` columns than trial-table rows (examples: 160 vs 480, 206 vs 582, 140 vs 620). `obs_intervals_index` has the same shorter count. This indicates unit/probe recording intervals do not necessarily cover every behavioral trial. Mapping must use observation times/trial intervals robustly rather than directly indexing this matrix as if all trials were represented.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total, raw unit rows) | 272,227 |
| Neurons labeled `unit_quality=good` | 154,948 |
| Neurons / session (raw) | min 493, mean 1,564.52, max 3,191 |
| Subjects | 28 |
| Sessions / subject | min 3, max 10; 174 total sessions |
| Trials (total) | 94,990 |
| Trials / session | min 264, mean 545.92, max 800 |

### Available Values and Distributions
- Outcome: hit 65,254 (68.70%), miss 15,641 (16.47%), ignore 14,095 (14.84%).
- Early lick: early 10,805 (11.37%); no early 84,185 (88.63%).
- Instruction: right 48,913; left 46,077. All 94,990 trials are `audio delay`.
- Photostimulation: 18,588 trials with values; 76,402 `N/A` control trials.
- Unit quality: 154,948 good; 117,279 multi.
- Electrode region labels (14): bilateral ALM, Striatum, Thalamus, Midbrain, Medulla, ECT, and BLA.
- All sessions contain side-camera tongue tracking.

### Data Quality / Edge Cases
- Event series lengths may differ from trial count (e.g. sample/delay events), so trials must be aligned using timestamps and trial intervals rather than positional assumptions.
- Tongue invisibility should use tracking likelihood according to reference methodology; low-likelihood rows can still have finite x/y coordinates.
- Direct trial-table fields already use exactly the requested outcome categories and early-lick labels.
- Brain-region names should come from electrode `location` JSON through each unit's electrode reference.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total after classifier QC) | 69,943 | Data paper/QC white paper: “Overall, the dataset consisted of 69,943 good units…” |
| Major-region good units | ALM 8,717; striatum 7,664; thalamus 12,808; midbrain 7,495; medulla 2,928 | Data paper methods |
| Subjects | 28 mice | Data paper Fig. 1J; method paper Methods |
| Sessions | 173 behavioral sessions | Data paper/QC paper |
| Probe insertions | 655 (text; figure caption says 660 penetrations) | Data paper methods/Fig. 1J |
| Trials / session | mean 476, range 130–785 | Data paper methods |
| Correct rate | mean 84%, range 65–99% | Data paper methods |
| Neural data time bin | 40 ms width, 3.4 ms stride | Method paper Methods |
| Behavior video rate | 300 Hz | Method paper Methods |
| Photoinhibition performance | 83.2% control to 71.7% bilateral ALM stimulation | Data paper methods; 17 mice, 93 sessions |
| Classifier-QC fraction | 25.9% of Kilosort2 clusters | Data paper/QC white paper |
| QC false-alarm rates | cortex 7.8%, striatum 6.4%, thalamus 7.3%, midbrain 5.5%, medulla 4.3% | Data paper methods |

### Processing Details
- Task: auditory instruction (high/low tone) during sample, followed by delay; mice lick left/right after the go cue in the response epoch.
- The method paper bins spikes into 40-ms sliding windows with 3.4-ms stride. The decoder specification overrides this only for bin geometry: use contiguous 50-ms bins from -2.5 to +1.5 s around go cue.
- Side-view video was recorded at 300 Hz. DeepLabCut tracked jaw, paws, and tongue using one model across sessions.
- Method-paper marker cleanup identifies velocity outliers above five sigma and imputes them from nearby frames. When tongue is occluded in the mouth (typically before response), its position was set to its mean. The supplied NWB also provides tracking likelihood, which can identify visibility for the task-required class 3.
- Reference movement analyses excluded photoinhibition, free-water, early-lick, and ignore trials. The data paper likewise excluded early-lick and no-response trials for behavioral analyses.
- Session selection in the data paper required >65% overall behavioral performance and at least 50 correct left and 50 correct right control trials.

### Curation Steps

**Neuron curation rules**:
- Kilosort2 clusters were manually labeled on 28 penetrations across five major areas, and 15 quality metrics trained five region-specific logistic-regression classifiers.
- Apply cortex classifier to ALM/other cortex/hippocampus/olfactory/cortical subplate; striatum classifier to striatum/pallidum; thalamus classifier to thalamus/hypothalamus; midbrain classifier to midbrain/pons; medulla classifier to medulla/cerebellum.
- Use classifier-derived `good` units, not Kilosort `unit_quality` alone. NWB contains the classifier result in `units/classification`.

**Trial curation rules**:
- Paper exclusions were analysis-specific. This decoder explicitly requires photostimulation input and early-lick, ignore/miss/hit, and no-lick output categories, so excluding those trials would destroy required targets.
- Retain all structurally valid trials with a go cue and complete requested time window; account for per-unit observation intervals. Exclude only sessions/trials that cannot support valid neural alignment or required outputs.

### Decoders Trained
| Decoded variable | Accuracy |
|------------------|----------|
| Single-neuron choice modulation | ROC AUC; method-paper criterion AUC > 0.65 |
| Single-neuron uninstructed movement modulation | ROC AUC; criterion AUC > 0.65 |
| Video behavioral predictability example | ROC AUC > 0.60 criterion |

The papers do not report directly comparable balanced accuracies for the provided multi-output neural decoder. Their AUC thresholds provide qualitative, not numerical, expectations for this task.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Unit QC | Production code uses classifier QC | NWB has both Kilosort `unit_quality` (154,948 good) and final `classification` (69,453 good) | 69,943 classifier-good units | Use `classification == good`; this closely reproduces paper QC, unlike `unit_quality`. The 490-unit difference is attributed to supplied DANDI-version/export differences and will be reported. |
| Session count | Code processes sessions with QC files | 174 NWBs; one session has all 1,852 classifications=`nan` | 173 sessions | Exclude `sub-440958_ses-20190216T162508`; it has no classifier-curated units. Remaining count exactly matches 173. |
| Trial count | Paper code often filters early/free-water/ignore/photostim trials | NWB has 94,990 behavioral trials and required labels | Papers report analysis-specific exclusions and 476 trials/session | Retain required categories, but only trials with simultaneous curated-unit observation coverage. Do not apply movement-paper exclusions that erase decoder targets. |
| Partial recordings | Older MATLAB code assumes probe/session arrays already aligned | Nine NWBs have fewer `is_good_trials` columns than behavioral trials | Not explicitly discussed | `obs_intervals` and `is_good_trials` represent a prefix of recorded trials. All curated units in each session share the same count and all represented entries are true. Restrict each session to that prefix; never fabricate neural zeros for later behavioral-only trials. |
| Go cue | Raw MATLAB spikes were already go-relative | NWB spikes/events are absolute; every trial contains exactly one go-start event | Decoder requires go alignment | Find the unique `go_start_times/timestamps` value inside each trial interval and subtract it for binning/alignment. |
| Tone/sample onset | Code exposes sample timing | Some trials contain repeated sample-start events; every trial has at least one before go | Auditory tone occurs in sample epoch | Use the final sample-start event before the unique go cue. Common tone-to-go durations (1.85, 0.95, 2.45 s) support variable protocol timing. Preserve measured timing rather than assume a constant offset. |
| Choice semantics | MATLAB comments conflict on lick direction coding | NWB has instruction and outcome strings | Hit means instructed lick, miss means incorrect lick, ignore means no response | Derive choice: `ignore -> no lick`, `hit -> instruction`, `miss -> opposite instruction`; spot-check later against post-go left/right lick timestamps. |
| Brain regions | MATLAB pipeline uses histology/QC region names | NWB electrode target JSON gives 14 bilateral labels | Paper totals use broader histological regions and 14 analysis groups | Use supplied unit-to-electrode target labels exactly (including hemisphere) as available NWB region metadata; document that these are target labels, not fine CCF annotations. |
| Tongue occlusion | Method code imputes occluded marker to mean | NWB provides x/y/likelihood, finite coordinates even at tiny likelihood | Method paper mean-imputes occluded tongue | Task explicitly requires class 3 “not visible”; use likelihood-based visibility and retain class 3 rather than mean-imputing it away. Apply reference five-sigma velocity cleanup to visible y traces before percentiles. |
| Firing-rate bins | Reference uses 40-ms width/3.4-ms stride | Raw spikes available | Decoder requires 50-ms bins | Explicit task overrides reference bin geometry. Count spikes in 80 contiguous half-open 50-ms bins over [-2.5,1.5), divide by 0.05 s. |

### Final Consistent Understanding
- Analyze 173 sessions with at least one classifier-good unit; exclude the one all-`nan` classifier session.
- Start from `classification == good`. Four sessions contain some classifier-good units with false `is_good_trials` entries; because target matrices require one fixed population per session, additionally require each retained unit to be valid on every represented trial. This removes 565 units (0.81%) while preserving all represented trials.
- Retain early-lick, miss/ignore, and photostimulation trials because they are explicitly required inputs/outputs, despite reference movement-analysis exclusions.
- Align every stream in absolute NWB time to the unique per-trial go cue. Use exact timestamps for spikes, tone onset, laser intervals, and tongue frames.
- Session target labels from electrode JSON are the only direct region labels in these NWBs; preserve them without inventing finer anatomy.
- Paper classifier count discrepancy (69,453 supplied versus 69,943 published, -0.70%) is small and traceable to the supplied export/version rather than processing logic.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `units/spike_times` ragged arrays | `neural` | Select `classification == good`; histogram absolute spikes into 80 half-open 50-ms bins at go + `[-2.5,1.5)`; divide counts by 0.05 for Hz; transpose to neurons x time | `sliding_histogram`, `process_one_area` | Task-required bins replace reference 40-ms/3.4-ms sliding bins. |
| final `sample_start_times/timestamps` before go | `input[0,:]` | For each bin center, seconds elapsed since tone onset: `(go + bin_center) - tone_onset` | task timing extraction in `process_one_sess` | Continuous, time-varying; values may be negative before tone. |
| `photostim_start_times` / `photostim_stop_times` | `input[1,:]` | 1 when bin center lies in any laser interval (`start <= t < stop`), else 0 | stimulation alignment in `process_one_sess` | Exact event intervals; trial table/event status agrees for all usable trials. |
| `trial_instruction` + `outcome` | `output[0,:]` choice | ignore -> no lick; hit -> instructed side; miss -> opposite side; tile over 80 bins | behavior fields in `process_one_sess` | Classes: left=0, right=1, no lick=2. |
| `outcome` | `output[1,:]` | Direct categorical mapping, tiled over 80 bins | correctness/report loading | Classes: ignore=0, miss=1, hit=2. |
| `early_lick` | `output[2,:]` | Direct categorical mapping, tiled over 80 bins | early-report loading | Classes: no=0, yes=1. |
| side-camera tongue x/y/likelihood + timestamps | `output[3,:]` | Clean visible y; nearest frame to each bin center; visible session y percentiles: `<q40`=0, `q40..q60`=1, `>q60`=2; likelihood <0.9/missing=3 | method-paper marker preprocessing | Time-varying; classes low/middle/high/not visible. |
| parent `sub-*` directory | `subjects`, `subject_idx` | Unique subject IDs and per-session index | filename parsing analogous to `process_one_sess` | 28 subjects expected. |
| unit electrode index -> electrode `location` JSON `brain_regions` | `brain_regions`, `brain_region_idx` | Preserve supplied bilateral target labels; global sorted lookup | `helper_filter_by_neuron_id` concept | 14 labels expected. |

### Exact Temporal Representation
- Window edges: `np.arange(-2.5, 1.5 + 0.05, 0.05)` (81 edges); centers are -2.475 through +1.475 s (80 points).
- Spike bins are left-closed/right-open. Rates are spike counts / 0.05 s in float32.
- All streams use absolute NWB timestamps before subtraction/alignment, preventing clock-offset mistakes.
- The unique go event inside each trial is the alignment timestamp.
- Tone onset is the latest sample-start event within the trial and strictly before go; this maps all source trials and preserves protocol-dependent sample/delay timing.
- Photostimulation state is sampled at bin centers from exact event start/stop intervals.
- Tongue y uses the nearest 300-Hz video frame to each center; require nearest-frame distance <=20 ms (normal distance is <2 ms), otherwise class 3.

### Tongue Processing and Discretization
- Define raw visibility as finite y and DeepLabCut likelihood >=0.9. Likelihood is extremely bimodal (median session 75th percentile about 0.00006, 90th percentile about 0.999996); 0.9 is insensitive to modest threshold changes and yields 11.68% visible frames on average.
- Match method-paper cleanup: calculate frame-to-frame y velocity among contiguous visible frames, identify absolute velocity outliers above five standard deviations, and linearly interpolate those y samples from neighboring clean visible samples. Low-likelihood samples remain “not visible” because the requested class 3 overrides the paper's mean imputation of occlusion.
- Compute q40/q60 from all cleaned visible y frames over that session, exactly matching “over the session.” Boundaries: class 0 for y < q40; class 1 for q40 <= y <= q60; class 2 for y > q60.
- Representative session check: q40=273.94, q60=288.70; sampled fractions low/middle/high/not-visible = 4.60%/2.51%/5.52%/87.37%, plausible because most of the -2.5 to +1.5-s window precedes overt licking.

### Curation and Inclusion Decisions
1. **Sessions**: Include the 173 sessions containing classifier-good units. Exclude the one all-`nan` classifier session. This exactly matches the paper session count.
2. **Neurons**: Use only `units/classification == good`, the stored region-specific classifier result. Do not substitute `unit_quality`.
3. **Trials**: For each included session use the prefix represented by `units/is_good_trials`, and retain classifier-good units whose validity is true on every represented trial. This gives 90,734 valid trials and 68,888 units. Retain early, miss, ignore, photostim, auto-water, and free-water trials because required labels/categories would otherwise be lost.
4. **Window validity**: Every represented trial has one go cue. Video and spikes are checked against the requested window. Missing video samples become tongue class 3; neural bins outside a unit's observation period would be invalid, but represented trial observation intervals and `is_good_trials` are checked to prevent this.
5. **Minimum session size**: All included sessions have at least 160 represented trials and at least 90 curated units, comfortably exceeding format requirements.
6. **Output shape**: Use `(4,80)` for every trial; tile the three per-trial categories so they can coexist with time-varying tongue in one rectangular array.

### Expected Converted Statistics
- 28 subjects, 173 sessions, 90,734 final valid trials, 68,888 classifier-good/always-valid units (69,453 before trial-validity filtering).
- Curated/always-valid units/session: min 90; aggregate 68,888. Exact mean/max will be recorded after conversion.
- Trials/session: min 160, mean 539.36, max 800.
- Choice counts before tiling: left 39,940; right 39,323; no lick 14,047.
- Outcome: ignore 14,047; miss 15,458; hit 63,805.
- Early lick: no 82,602; yes 10,708.
- Trial-level photostim status: off 75,144; on 18,166 (time-varying bin occupancy will be lower).

### Planned Sanity Checks
- [ ] Direct neural spot check: load original spike times for selected session/trial/unit, independently apply `np.histogram`, and `np.allclose` against converted rates.
- [ ] Direct input spot check: independently derive tone-relative center times and laser interval booleans from NWB event timestamps; compare via `np.allclose`.
- [ ] Direct output spot check: independently map trial table strings and nearest tongue frames/session thresholds; compare via `np.allclose`.
- [ ] Verify every session has exactly 80 bins, matching trial counts across neural/input/output, and a constant neuron count across its trials.
- [ ] Verify unique go cue and at least one pre-go sample event for every included trial.
- [ ] Verify all selected neurons are classifier-good and every represented `is_good_trials` entry is true.
- [ ] Compare aggregate subject/session/trial/neuron counts and categorical distributions to values above and paper statistics.
- [ ] Plot raster/rates, tone-relative time, photostim intervals, tongue likelihood/y/classes, and percentile boundaries for up to two sessions.

---

## Step 6: Script Development
**Status**: COMPLETE

Implementation notes:
- Created `/app/convert_data.py` with required positional output path and `--full`, `--sample`, and `--show-processing` modes.
- Implemented deterministic NWB inventory, stored classifier QC plus per-trial validity filtering, simultaneous recorded-trial checks, exact event alignment, vectorized spike-rate binning, two decoder inputs, four categorical outputs, region/subject mappings, metadata, internal validation, timing logs, and diagnostic plots.
- Syntax compilation and CLI help completed successfully. A photostimulation boolean-accumulator type issue found during code review was fixed before sample execution and covered by a direct self-check.
- Sample attempt 1 exposed four sessions with unit-specific false `is_good_trials` cells. Fixed by requiring retained classifier-good units to be valid on every represented trial (565 units removed, all 93,310 trials preserved); this is preferable to varying neuron dimensions or fabricating zeros.
- Sample attempt 2 exposed an over-strict check that required the go-centered window to fit inside NWB `obs_intervals`. Inspection showed those intervals mirror behavioral trial start/stop, while ephys is continuous and valid windows can extend into ITI. Removed this incorrect clipping assertion; retained `is_good_trials` as the validity criterion.

Code inefficiencies identified:
- Naive nested neuron x trial x bin loops would be prohibitive for 69,453 units and 93,310 trials.
- Repeated per-trial video scans and repeated HDF5 spike reads would add unnecessary overhead.

Code speedups added:
- For each unit, one vectorized `np.searchsorted` against all globally ordered trial-bin edges computes all counts.
- Go/tone mappings, tongue nearest frames, and output arrays are computed session-wise with NumPy.
- HDF5 datasets are read once per required stream/session; rates use float32 and outputs int64.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 834 (459, 375) |
| Neurons / session | mean 417 |
| Subjects | 1 (`440956`) |
| Sessions / subject | 2 |
| Trials (total) | 527 |
| Trials / session | 368, 159 |
| Neural shape | `(n_neurons, 80)` float32 per trial |
| Input shape | `(2, 80)` float32 per trial |
| Output shape | `(4, 80)` int64 per trial |
| Time-from-tone range | approximately [-0.6, 5.7] s |
| Photostimulation range | [0, 1] |
| Choice distribution | left 42.4%, right 38.0%, no lick 19.6% |
| Outcome distribution | ignore 19.6%, miss 29.8%, hit 50.6% |
| Early lick distribution | no 94.3%, yes 5.7% |
| Tongue distribution | low 5.3%, middle 2.8%, high 6.2%, not visible 85.7% |
| Mean firing rate | inspected and finite/nonnegative; sparse 50-ms multiples |

### Processing Plots Review
- Created `processing_sub-440956_ses-20190207T120657.png` and `processing_sub-440956_ses-20190208T133600.png`.
- Plots show firing rates aligned to go=0, linear time-from-tone traces, binary laser state, raw/clean tongue y with q40/q60 thresholds, tracking likelihood, and final class 3 during occlusion.
- No temporal offset or discretization anomaly was found.

### Format Validation
- `/app/verification_sample_out.txt` created; verifier completed successfully.
- No errors.
- Final rerun: no warnings. An earlier warning for session 1 source trial 159 had all-zero neural data. Direct raw NWB check found zero spikes across every one of the 375 retained units in the exact [-2.5,+1.5) s window, while source `is_good_trials` is true. Neighboring trials contain normal activity. This is an unavoidable source recording gap/anomaly, not a binning error; the trial is retained to avoid unsupported behavioral filtering.
- Long positive time-from-tone values were investigated. Values >2.5 s occur overwhelmingly on early-lick trials (4,595 cases dataset-wide), where repeated delay-state transitions postpone go cue; raw sample onset is unique and unchanged. Thus these values are valid rather than alignment errors.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| Vectorized searchsorted spike binning | Neural binning only 0.23–0.39 s/session |
| Session-wise vectorized video/input/output processing | Total processing 0.70–1.21 s/session |
| Metadata-only inventory prepass | Detects exclusions/regions without loading bulk spikes |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Session processing | about 0.96 s mean for sample | about 3 minutes for 173 sessions, allowing size variation |
| Inventory + pickle I/O | sample total 4.87 s | conservatively several additional minutes |
| Full conversion estimate | — | under 10 minutes, below 15-minute optimization threshold |

### Iterations
1. Initial run found unit-specific false `is_good_trials` values in four sessions; added always-valid-unit filtering.
2. Second run found an incorrect assertion against behavioral `obs_intervals`; direct inspection showed required windows may extend into ITI while continuous spikes remain available, so removed the assertion.
3. Third run completed, internal validation passed, plots were generated, and the provided verifier passed.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: One source-confirmed all-zero neural trial; investigated and documented in Step 7.

### Training Progress
- Training completed for all 200 epochs.
- Loss decreased monotonically from 8+ early in training to 0.583916 at epoch 200; test loss 0.694469.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Uniform Chance |
|--------|-----------------------|-------------------------|----------------|
| Lick direction choice | 0.7261 | 0.6368 | 0.3333 |
| Outcome | 0.7467 | 0.6662 | 0.3333 |
| Early lick | 0.8349 | 0.7513 | 0.5000 |
| Tongue y-position | 0.6814 | 0.5314 | 0.2500 |

All outputs are above chance, including tongue at more than twice uniform chance. Train/validation ratios are 1.14, 1.12, 1.11, and 1.28 respectively, below the 1.5x overfitting concern threshold.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 11.942 GB decimal after invalid-gap filtering
- `conversion_full_out.txt`: created
- `verification_full_out.txt`: created; verification completed without errors
- Full conversion runtime: 164.37 s, substantially faster than the conservative estimate.

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total neurons | 69,943 classifier-good | classifier QC | 69,453 classifier-good | 68,888 classifier-good and valid on every represented trial | Yes after documented 565-unit validity filter; -1.51% from paper |
| Mean neurons/session | not stated | region/session populations | 401.46 before trial-validity filter | 398.20 (min 90, max 923) | Reasonable |
| Subjects | 28 | filename-derived | 28 | 28 | Yes |
| Sessions | 173 | process sessions with QC | 174 NWBs, one lacks classifier labels | 173 | Yes |
| Trials (total) | mean 476/session in analysis-selected subset | analysis-specific exclusions | 94,990 behavioral; 93,310 represented before invalid-gap filtering | 93,310 | Yes for decoder-required inclusion |
| Trials/session | range 130–785 paper subset | varies | represented min 160, mean 539.36, max 800 | min 160, mean 539.36, max 800 | Yes to source; expected difference from paper subset |
| Time from tone range | variable sample/delay; early licks alter state timing | sample timing | approximately [-1.525, 11.8943] at bin centers | [-1.525, 11.8943] | Yes |
| Photostim input range | binary laser intervals | exact onset/offset | [0,1] | [0,1], 2.430% bin occupancy | Yes |
| Choice distribution | not directly tabulated | behavior report/direction | [0.428036, 0.421423, 0.150541] | same | Yes |
| Outcome distribution | direct NWB labels | correctness/report | [0.150541, 0.165663, 0.683796] | same | Yes |
| Early-lick distribution | analysis often excludes | early report | [0.885243, 0.114757] | same | Yes |
| Tongue classes | no categorical reference | marker processing | per-session visible percentiles | [0.061825, 0.031609, 0.064644, 0.841921] | Plausible; visibility dominates pre-response window |

### Full Validation Notes
- All dimensions, ranges, subject/region mappings, and categorical values passed the provided verifier.
- The verifier reports all-zero neural-trial warnings. A sample warning was already proven to be exactly zero in raw spikes. All full warnings are reviewed in Step 10 rather than suppressed.
- Spot checks and direct original-file comparisons follow in Step 10.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output Log Verification
- Initial full verification found 2,576 all-zero neural trials across 94 sessions and no other warning types.
- Raw NWB checks proved these were exact population-wide source spike gaps, scattered through otherwise continuously covered sessions. A four-second absence of all spikes from 90–923 units is physiologically implausible, so these are invalid acquisition periods not marked by `is_good_trials`.
- Fixed by excluding population-wide zero-spike trials after binning and applying the identical trial mask to every stream. Removed 2,576/93,310 represented before invalid-gap filtering trials (2.76%); all sessions retain at least 159 trials.
- Final `/app/verification_full_out.txt`: **“Data format is valid, no errors or warnings.”** Final dataset has 173 sessions and 90,734 trials.

### Check 2: Direct Original-Data Sanity Checks (`np.allclose`)
- Scripts: `/app/cache/direct_sanity_checks_corrected.py` and `/app/cache/direct_tongue_check.py`.
- Neural: independently loaded ragged source spike times, mapped corrected source trial indices, called `np.histogram` on exact go-relative edges, divided by 0.05, and matched converted rates for five cases across first/middle/last sessions.
- Inputs: independently selected the unique go event, latest pre-go sample onset, and laser intervals; both time-from-tone and laser state matched via `np.allclose`.
- Trial outputs: independently mapped raw outcome/instruction/early strings and matched all 80 tiled values via `np.allclose`.
- Tongue: independently repeated visibility, five-sigma velocity cleanup, interpolation, session percentiles, nearest-frame sampling, and four-class assignment; matched converted output via `np.allclose`.
- Choice semantic check: derived choice agreed with first raw post-go lick direction on 69,245/69,424 (99.742%) responsive non-early trials. Rare mismatches involve multiple/near-simultaneous events and do not invalidate authoritative trial outcome/instruction labels.

### Check 3: Reference Code Comparison
| Processing step | Conversion | Reference | Comparison / rationale |
|-----------------|------------|-----------|------------------------|
| Data loading | Direct HDF5/NWB datasets | Recursive MATLAB `loadmat` export loader | Same underlying MAP streams; direct NWB avoids export ambiguity. |
| Neuron filtering | `classification == good`, then require validity on all represented trials | Region-specific classifier QC lists | Same final classifier concept. Extra validity filter removes 565 units needed for fixed session populations. |
| Trial filtering | represented prefix, remove raw population gaps; retain required categories | Movement analyses exclude photostim/free-water/early/ignore | Intentional decoder-task override; excluding these would erase requested inputs/outputs. |
| Alignment | unique absolute go event per trial | exported spikes already go-relative; lick/laser times subtract go | Equivalent alignment in source clock. |
| Binning | 80 contiguous `[left,right)` 50-ms bins, rates in Hz | `sliding_histogram`, half-open 40-ms windows/3.4-ms stride | Same histogram/rate logic; task-required geometry overrides reference. |
| Input construction | measured tone onset and exact laser intervals | sample and stimulation timing fields | Same source concepts, represented time-varyingly as required. |
| Output construction | authoritative trial labels plus tongue tracking | behavior report/direction and marker processing | Same fields; class 3 preserves occlusion required by task rather than reference mean imputation. |

### Check 4: Key Statistics
| Statistic | Paper | Source / final |
|-----------|-------|----------------|
| Subjects | 28 | 28 / 28 |
| Sessions | 173 | 174 NWBs, one unclassified / 173 |
| Classifier-good units | 69,943 | 69,453 in supplied version; 68,888 after fixed-population validity filter |
| Trials | analysis mean 476/session after exclusions | 93,310 represented before invalid-gap filtering; 90,734 after invalid-gap removal (mean 524.47/session) |
| Choice fractions | not tabulated | left 42.847%, right 42.222%, no lick 14.930% after filtering |
| Outcome fractions | 84% correct in selected control subset | ignore 14.930%, miss 16.640%, hit 68.430% across all required categories |
| Early lick | analysis excluded | no 88.448%, yes 11.552% |
| Tongue | occluded before response | not visible dominates (~84%), as expected in mostly pre-response window |

### Check 5: Edge Cases
- Exactly one go event and at least one sample event map every represented trial.
- Repeated sample/delay state events are handled by latest pre-go sample onset; long tone-to-go intervals are overwhelmingly genuine early-lick state-machine delays.
- Nine sessions contain shorter neural trial prefixes than behavioral tables; later behavior-only trials are excluded.
- Four sessions have unit-specific false validity cells; only always-valid classifier-good units are retained.
- Half-open bin edges avoid double counting. All sessions have exactly 80 bins and >=159 trials.
- Nearest video samples require <=20 ms distance (normally <2 ms); missing/low-confidence frames map to class 3.
- Every final neural trial has at least one source spike; no fabricated zeros or NaNs remain.

### Iterations and Rechecks
1. Found unit-specific validity cells; added always-valid unit filter and reran sample checks.
2. Removed incorrect clipping to behavioral observation boundaries after raw inspection; reran sample checks.
3. Found population-wide raw spike gaps in full verification; added trial exclusion, reran full conversion/verification, all direct comparisons, key statistics, edge checks, sample conversion, sample verification, and sample training.
4. Final sample training: loss 7.76 early to 0.5892; validation balanced accuracies choice 0.6240, outcome 0.6568, early lick 0.7345, tongue 0.5355—all above chance.

All discovered issues are resolved; final validation has no errors or warnings.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Full command completed successfully on GPU with `--plot-samples`.
- 72,533 training trials and 18,201 validation trials.
- Loss decreased smoothly from 27.495369 at epoch 1 to 0.707289 at epoch 200.
- Test loss: 0.702779.
- `/app/train_decoder_full_out.txt` contains the complete run.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Uniform Chance | Notes |
|--------|-----------------------|-------------------------|----------------|-------|
| Lick direction choice | 0.6949 | 0.6645 | 0.3333 | 1.99x chance |
| Outcome | 0.6963 | 0.6547 | 0.3333 | 1.96x chance |
| Early lick | 0.7925 | 0.7519 | 0.5000 | 1.50x chance |
| Tongue y-position | 0.6548 | 0.5990 | 0.2500 | 2.40x chance |

All outputs exceed chance, loss converged, and train/validation gaps are small.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy vs Chance Analysis
| Variable | Validation Balanced Accuracy | Uniform Chance | Ratio to Chance | Finding |
|----------|------------------------------|----------------|-----------------|---------|
| Lick direction choice | 0.6645 | 0.3333 | 1.994x | Strongly above chance |
| Outcome | 0.6547 | 0.3333 | 1.964x | Strongly above chance |
| Early lick | 0.7519 | 0.5000 | 1.504x | Above the 1.5x review threshold |
| Tongue y-position | 0.5990 | 0.2500 | 2.396x | Strongly above chance |

No output is below chance or below 1.5x uniform chance. All categorical classes are present. Final timepoint-weighted class fractions are:
- Choice: left 0.428472, right 0.422223, no lick 0.149305.
- Outcome: ignore 0.149305, miss 0.166398, hit 0.684297.
- Early lick: no 0.884475, yes 0.115525.
- Tongue: low 0.062122, middle 0.031745, high 0.065107, not visible 0.841026.

### Accuracy Comparison to Papers
| Variable / analysis | Achieved | Paper reference | Interpretation |
|---------------------|----------|-----------------|----------------|
| Lick direction choice | balanced accuracy 0.6645 | Method paper uses single-neuron choice ROC AUC and threshold >0.65; one reported mean AUC is 0.66±0.12 for a different video-prediction analysis | Achieved value is numerically consistent with paper signal strength but metrics/tasks differ. |
| Outcome | balanced accuracy 0.6547 | Data paper reports outcome-selective neural activity but no directly comparable provided-decoder accuracy | Strongly above chance; direct numerical comparison unavailable. |
| Early lick | balanced accuracy 0.7519 | Early trials were excluded from paper movement analyses | No paper decoder benchmark; high accuracy supports correct retention/alignment. |
| Tongue y-position | balanced accuracy 0.5990 | Method paper predicts neural activity from video/markers and uses AUC >0.60 or >0.65 modulation criteria | Reverse decoding direction and categorical target differ; achieved 2.396x chance is strong. |
| Population choice decoding | balanced accuracy 0.6645 | Data paper reports population choice decoding curves versus neuron count, chance 0.5 for binary correct-trial choice | Our target has three classes including no lick and uses a different architecture/split, so its harder 3-class balanced accuracy is not directly comparable. |

The papers contain no balanced-accuracy result for this exact four-output neural decoder. Metric/target differences are explicitly noted rather than treating AUC as accuracy.

### Train vs Validation Gap
| Output | Train | Validation | Train/Validation | Absolute Gap |
|--------|-------|------------|------------------|--------------|
| Choice | 0.6949 | 0.6645 | 1.046 | 0.0304 |
| Outcome | 0.6963 | 0.6547 | 1.064 | 0.0416 |
| Early lick | 0.7925 | 0.7519 | 1.054 | 0.0406 |
| Tongue | 0.6548 | 0.5990 | 1.093 | 0.0558 |

All ratios are far below the 1.5x overfitting criterion. Test loss (0.702779) is close to final training loss (0.707289), with no evidence of leakage or severe overfitting.

### Required Low-Accuracy Debugging Checks
Although no output was low, the prescribed checks were performed:
1. Raw trial labels were verified on multiple explicit trials and by `np.allclose` direct-source scripts.
2. Neural/input/tongue temporal alignment was plotted and independently reconstructed from absolute NWB timestamps.
3. Every output has meaningful variation; no class approaches 99%.
4. Neural filtering uses paper classifier QC plus source trial-validity and invalid-gap filtering.
5. Loading, alignment, binning, input construction, and output construction were compared line-by-line conceptually to reference code in Step 10.

### Issues Found and Resolved
- No new conversion issue was revealed by full-decoder accuracy.
- The lowest validation/chance ratio is early lick at 1.504x, still above the requested threshold. Raw labels, class balance, and alignment had already passed direct checks; no scientifically justified conversion change would improve it without target leakage.
- Tongue validation accuracy of 0.5990 across four classes, despite 84% invisible samples and balanced scoring, supports the likelihood/percentile mapping.

All Step 12 checks pass; no further iteration is required.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with loading instructions, format, processing summary, final statistics, and decoder results.
- [x] cache/ folder created.
- [x] `cache/README_CACHE.md` documents extracted paper text and direct-source validation scripts.
- [x] Analysis/investigation scripts are stored under `/app/cache/`.
- [x] All required conversion, validation, training, plot, and documentation outputs are present.
- [x] Final full verifier reports no errors or warnings.

