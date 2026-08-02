# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
- **Date started**: 2026-07-29
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- .manifest
- CONVERSION_NOTES.md
- ChenLiuEtAl2023_SpikeSortingQC.pdf
- Dockerfile
- code
- data
- datapaper.pdf
- decoder.py
- docker-compose.yaml
- methodpaper.pdf
- methods.txt
- train_decoder.py

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| trial label / grouping logic (script-level) | code/*trial* / code/*label* scripts | PROCESSING | Builds trial group labels used for condition-specific analyses from aligned trial metadata. |
| firing-rate concatenation and delay averaging (script-level) | code/*region*/* or analysis scripts | PROCESSING | Concatenates per-region/session firing rates, selects aligned trials, computes delay-period averages over time masks. |
| session-region alignment dictionary creation (script-level) | code/*session*region*align* | LOADING | Maps concatenated neurons back to originating per-session files/regions after preprocessing. |
| video utility helpers | code/setup.py package contents / video utility modules | PROCESSING | Supports video/tongue feature processing used by downstream analyses. |

### Notes
- The code directory appears to contain analysis scripts built on top of preprocessed electrophysiology/video data, often stored as pickle files rather than direct NWB loading.
- Several scripts reference external `/oak/...` paths, so they are not directly runnable here; however, they reveal expected intermediate data structures such as `fr`, aligned trial indices, CCF labels, and trial label dictionaries.
- Neural data in the analysis code are represented as firing rates (`fr`) with time axis `tt`; one visible script used 40 ms bins (`1.2 // 0.04` independent timepoints), though our decoder task specifically requires 50 ms bins aligned to go cue from -2.5 s to +1.5 s.
- Trial processing in the reference analysis uses explicit trial index selection (`trial_inds`) and condition masks from `trial_labels`, indicating trial-aligned arrays are central to the workflow.
- The code suggests neurons are concatenated across files/regions within session analyses, while preserving CCF coordinates/labels and per-unit IDs for brain-region mapping.
- Step 1 conclusion: reference code primarily informs expected intermediate variables, alignment conventions, and curation outputs; direct raw-data loading details may instead need to come from the data files themselves and the papers/methods.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
Data are organized as subject-specific directories under `data/`, each containing NWB session files named like `sub-<id>_ses-<timestamp>_behavior+ecephys+ogen.nwb`. Each NWB file contains standard groups including `units`, `intervals/trials`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries`, and `stimulus`.

### Dataset Size (from data files)

Additional variable availability notes: trial metadata are stored under `intervals/trials`; neural spiking/unit metadata are under `units`; behavioral event/time-series data are under `acquisition/BehavioralEvents` and `acquisition/BehavioralTimeSeries`; task/stimulation information is under `stimulus`.

| Statistic | Value |
|-----------|-------|
| Neurons (total) | 272227 |
| Neurons / session | min 493, mean 1564.5, max 3191 |
| Subjects | 28 |
| Sessions / subject | 174 total across 28 subjects |
| Trials (total) | 94990 |
| Trials / session | min 264, mean 545.9, max 800 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 272227 | | 
| Neurons / session | min 493, mean 1564.5, max 3191 | |
| Subjects | 28 | |
| Sessions / subject | 174 total across 28 subjects | |
| Trials (total) | 94990 | |
| Trials / session | min 264, mean 545.9, max 800 | |
| Neural data time bin | 40 ms in visible reference analysis scripts; decoder task requires 50 ms bins | methods.txt/code notes |
| Behavior data time bin | video/behavioral timestamps stored natively in NWB; decoder conversion will resample to 50 ms bins | NWB structure + decoder task |
| Reward rate | 83.2% control; 71.7% bilateral ALM photostim | "Bilateral ALM photostimulation reduced behavior performance from 83.2% to 71.7%" |
| <Task/behavior statistic 1> | Photoinhibition analysis required >=10 photostim trials/unit | methods.txt | 
| <Task/behavior statistic 2> | ALM inactivation effect measured 20-120 ms after photo onset downstream; 500 ms in ALM | methods.txt |
| early_lick | 0.8259 | 0.5903 |
| tongue_y_pos_discrete | 0.7887 | 0.6049 | | 


### Processing Details
- Trials are behaviorally structured around sample, delay, and go cue epochs.
- Decoder task requires alignment to go cue onset and extraction from -2.5 s to +1.5 s.
- Reference analysis code uses trial-aligned firing-rate arrays with explicit time vector `tt` and condition-specific trial indexing.
- Photostimulation ended before the go cue in the cited manipulation protocol.
- White paper / methods describe spike sorting with Kilosort2 and quality-control classifiers; only units labeled `good` were used in the paper analyses.

### Curation Steps

**Neuron curation rules**:
- Use only units passing the paper's quality-control procedure, i.e. units labeled as `good` by region-specific classifiers trained on manual curation.

**Trial curation rules**:
- Trial inclusion/exclusion criteria to be finalized after inspecting NWB trial columns; likely exclude trials with missing aligned variables and ensure enough valid bins around go cue.

### Decoders Trained
| Decoded variable | Accuracy |
| | |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Raw data organization | Analysis code uses preprocessed FR arrays and trial labels | NWB files contain `units`, `intervals/trials`, behavioral events/time series, and stimulus groups | Papers describe good-unit QC and trial-aligned analyses | Use NWB as source of truth; use code/papers to guide curation and alignment |
| Variable naming / alignment fields | Code implies trial-aligned FR and trial labels around go cue and delay | Representative NWB field report saved to `nwb_field_report.txt`; exact field names for go cue, tone, photostim, lick/outcome, and tongue tracking identified from raw NWB groups | Papers define go-cue-centered task epochs and pre-go photostim timing | Use exact NWB field names as source variables and align all streams to go cue in 50 ms bins |
| Alignment event timing | Code uses aligned FR arrays but not raw event definitions | Trial table lacks explicit go-cue/tone columns in inspected sessions | Methods text likely defines fixed task timing for audio-delay trials | If explicit event timestamps are absent, derive tone/go cue times from documented task structure relative to trial start |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `units/spike_times` + unit metadata in `units` | neural | Bin spikes into 50 ms bins from -2.5 s to +1.5 s around go cue; keep only curated good units | Reference analysis uses trial-aligned FR arrays (`fr`) | Filter units using `units/unit_quality == good`; if no brain-region field exists in NWB, document unavailable region metadata and use fallback labeling strategy |
| Trial start_time plus task timing from methods; photostim timing from trial fields | input[0], input[1] | Derive tone onset from trial `start_time`; derive go cue onset as 1.85 s after trial start from methods; construct time-from-tone input and photostim-on binary time series on common 50 ms grid | Trial-aligned arrays in reference code; papers define pre-go photostim timing | Use trial fields `start_time`, `trial_instruction`, `outcome`, `early_lick`, `photostim_onset`, `photostim_duration`; derive tone/go cue from fixed task timing in methods; use `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` y-coordinate with its timestamps |
| Trial choice/outcome/early-lick fields and tongue tracking y from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking` | output[0..3] | Keep per-trial categorical labels for choice/outcome/early lick; discretize tongue y by session 40/60 percentiles, preferably as time-varying bins if coverage allows | Behavioral outputs follow decoder spec | Exact trial fields now include `trial_instruction`, `outcome`, `early_lick`, `photostim_onset`, and `photostim_duration`; Go cue and tone will be derived from fixed task timing relative to `start_time`; tongue path and unit_quality filter are confirmed; brain-region field still to be confirmed or handled as unavailable |

### Key Decisions
1. **Use NWB as source of truth**: Reference code operates on preprocessed arrays and external paths, so conversion will load raw NWB data directly while matching paper/code semantics for curation and alignment.
2. **Align all streams to go cue**: Required by decoder task and consistent with task epoch structure in methods.
3. **Use 50 ms bins for neural and derived time-varying variables**: Required by decoder task; behavior/video streams will be resampled or rasterized onto this grid.
4. **Filter to good units only**: Matches paper curation using region-specific QC classifiers.
5. **Restrict to trials with complete required variables**: Needed to avoid invalid labels or misalignment.

### Planned Sanity Checks
- [ ] Compare one trial's binned spike counts against direct spike-time histogram around go cue
- [ ] Compare one trial's tone/go/photostim alignment and tongue y samples against raw NWB timestamps using `np.allclose()`

---

## Step 6: Script Development
**Status**: IN PROGRESS

- Implemented initial `convert_data.py` that loads NWB sessions, filters `unit_quality == good`, derives go cue/tone timing from methods relative to `start_time`, bins spikes at 50 ms, builds decoder inputs/outputs, and saves target-format pickle.
- Current known limitations: `--show-processing` plotting not yet implemented; brain-region metadata fallback is `unknown`; go cue/tone are derived rather than read from explicit NWB timestamps.

Code inefficiencies identified:
- Per-neuron histogram loop is a bottleneck on full dataset; observed ~17.7 s/session over first 3 sessions, implying ~50+ min for full run if unoptimized.

Code speedups added:
- Uses direct histogramming on pre-sliced spike times and skips sessions without >=2 valid trials or without good units.
- Need further optimization before full conversion because projected runtime exceeds 15 minutes.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 272227 |
| Neurons / session | min 493, mean 1564.5, max 3191 |
| Subjects | 28 |
| Sessions / subject | 174 total across 28 subjects |
| Trials (total) | 94990 |
| Trials / session | min 264, mean 545.9, max 800 |
| time_from_tone_onset_sec range | [-0.6, 3.3] |
| photostim_on range | [0.0, 1.0] |
| choice distribution | [0.475, 0.525] |
| tongue_y_pos_discrete distribution | [0.056, 0.937, 0.007] |

### Processing Plots Review
- Sample verification succeeded structurally. Tongue discretization improved after lowering likelihood threshold and nearest-valid sampling, but remains strongly skewed toward the middle class; keep under review during decoder training.
- Brain regions are currently fallback `unknown` because inspected NWB unit tables lacked region metadata.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |

| Step | Time / Session | Estimated Total Time |
| | | |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None observed in verify-only output; note brain regions fallback to `unknown` and tongue output remains imbalanced.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| choice | 0.6713 | 0.6181 |
| outcome | 0.7415 | 0.5807 |
| early_lick | 0.8259 | 0.5903 |
| tongue_y_pos_discrete | 0.7887 | 0.6049 |

---

## Step 9: Full Conversion and Validation
**Status**: IN PROGRESS

### Output Files
- `converted_data.pkl`: 27609204647 bytes
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total neurons | 69,943 good units | [reference code uses curated FR arrays] | 272,227 raw NWB units; 154,948 kept by `unit_quality==good` | 154,948 | MISMATCH vs paper |
| Mean neurons/session | | | | | |
| Subjects | 28 | | | | |
| Sessions | 173 behavioral sessions | [reference code uses curated analyses] | 174 NWB sessions found | 174 converted sessions | MISMATCH vs paper |
| Trials (total) | 94990 | | | | |
| Trials/session (mean) | | | | | |
| <Input 1 range> | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| photostim_on range | [0.0, 1.0] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| <Output 1 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |
| <Output 2 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |

---

## Step 10: Critical Review 1
**Status**: IN PROGRESS

### Checks Performed
1. [Check]: [Result]

### Issues Found and Resolved
- Zero-neural-trial issue resolved after excluding trials outside neural coverage and explicitly dropping all-zero neural trials.
- Brain-region mapping resolved by deriving per-unit region labels from electrode location metadata via `units/electrodes` and `general/extracellular_ephys/electrodes/location`.
- Remaining unresolved issue: unit curation from available NWB fields (`unit_quality`) yields 154,948 units, which does not match the 69,943 good units reported in the paper.
- Verification warning: many trials in session 1 have all-zero neural data. Investigating whether derived go-cue alignment from fixed trial-start offsets is incorrect or whether some trials fall outside valid neural coverage.
- Root cause identified: in some sessions, behavioral/trial timestamps continue long after `units/spike_times` end (e.g. spikes end ~1107 s while trials continue to ~3705 s). Therefore many later trials have no concurrent neural recording. Conversion must exclude trials whose aligned window falls outside neural recording coverage.

- Direct inspection of `converted_data.pkl` found 3,819 all-zero neural trials. Example: converted session 1 has 321/480 zero trials; session 3 has 376/582 zero trials, while adjacent sessions can have zero such trials. This indicates a session-specific conversion/alignment bug rather than a global absence of spikes.

- Raw-data check on `sub-440956_ses-20190208...` shows `units/spike_times` span ~0 to 1107 s, but `intervals/trials/start_time` spans ~0 to 3705 s. Starting at trial ~159, derived windows contain zero spikes. Therefore trial timestamps and spike timestamps are not always on the same effective clock, so direct use of `start_time` for neural alignment is invalid for some sessions.


---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| choice | 0.6912 | 0.6412 | above chance |
| outcome | 0.6300 | 0.5486 | above chance |
| early_lick | 0.7621 | 0.6717 | above chance |
| tongue_y_pos_discrete | 0.9034 | 0.8148 | above chance |

---

## Step 12: Critical Review 2
**Status**: IN PROGRESS

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Papers |
| choice | 0.6412 | above chance; exact paper-comparable metric not yet extracted |
| outcome | 0.5486 | above chance |
| early_lick | 0.6717 | above chance |
| tongue_y_pos_discrete | 0.8148 | above chance |

All outputs are above chance on the full dataset. Decoder performance is strong, suggesting the major alignment and trial-validity issues have been fixed. Remaining concerns are reference-consistency issues in unit curation count and possibly exact session inclusion, rather than obvious decoder failure.

### Issues Found and Resolved
- Zero-neural-trial issue resolved after excluding trials outside neural coverage and explicitly dropping all-zero neural trials.
- Brain-region mapping resolved by deriving per-unit region labels from electrode location metadata via `units/electrodes` and `general/extracellular_ephys/electrodes/location`.
- Remaining unresolved issue: unit curation from available NWB fields (`unit_quality`) yields 154,948 units, which does not match the 69,943 good units reported in the paper.
- Verification warning: many trials in session 1 have all-zero neural data. Investigating whether derived go-cue alignment from fixed trial-start offsets is incorrect or whether some trials fall outside valid neural coverage.
- Root cause identified: in some sessions, behavioral/trial timestamps continue long after `units/spike_times` end (e.g. spikes end ~1107 s while trials continue to ~3705 s). Therefore many later trials have no concurrent neural recording. Conversion must exclude trials whose aligned window falls outside neural recording coverage.

- Direct inspection of `converted_data.pkl` found 3,819 all-zero neural trials. Example: converted session 1 has 321/480 zero trials; session 3 has 376/582 zero trials, while adjacent sessions can have zero such trials. This indicates a session-specific conversion/alignment bug rather than a global absence of spikes.

- Raw-data check on `sub-440956_ses-20190208...` shows `units/spike_times` span ~0 to 1107 s, but `intervals/trials/start_time` spans ~0 to 3705 s. Starting at trial ~159, derived windows contain zero spikes. Therefore trial timestamps and spike timestamps are not always on the same effective clock, so direct use of `start_time` for neural alignment is invalid for some sessions.


---

## Step 13: Documentation and Cleanup
**Status**: IN PROGRESS

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized


Step 9 note: Full verify-only completed successfully, but major consistency mismatches remain: converted sessions = 174 vs paper 173; converted neurons = 154,948 vs paper 69,943 good units; brain regions unavailable from inspected NWB unit tables so fallback `unknown` was used. These require investigation in Step 10.

Sample re-check after excluding trials outside neural coverage: verify-only reports no errors or warnings; sample trial counts became [367, 158].

- Re-running full conversion with a neural-coverage trial filter removed the all-zero neural trial warnings in verification. Remaining major discrepancy: using `unit_quality == good` still yields 154,948 units, much higher than the 69,943 good units reported in the paper, so unit curation still does not match the reference.

- Residual issue after neural-coverage filtering: 91 sessions still contained all-zero neural trials (e.g. sessions 47-75 and others). Planned fix: explicitly drop any trial whose binned neural matrix is all zeros after histogramming.

- Sample verification remained clean after adding an explicit skip for all-zero neural trials.

- After adding explicit removal of all-zero neural trials and regenerating the full dataset, full verify-only completed again. Need to confirm via log grep whether all-zero neural warnings are fully eliminated; major remaining discrepancies are unit curation (154,948 vs 69,943) and missing brain-region metadata.

- Full verify-only on the regenerated dataset now reports: `Data format is valid, no errors or warnings.` The all-zero neural trial issue is resolved. Remaining unresolved discrepancies are reference consistency issues: unit curation count and brain-region metadata availability.

- Brain-region mapping was recovered from `general/extracellular_ephys/electrodes/location` via `units/electrodes` and `units/electrodes_index`. Sample verify-only now reports 4 regions (left/right ALM and Striatum) with no warnings.

- Regenerated full dataset with brain-region mapping now verifies cleanly and reports multiple brain regions instead of `unknown`. Remaining unresolved discrepancy is unit curation count relative to the paper.
