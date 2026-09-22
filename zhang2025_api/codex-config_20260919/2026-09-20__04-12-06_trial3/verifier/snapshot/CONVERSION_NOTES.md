# Dataset Conversion Notes

## Overview
- **Dataset**: IBL brain-wide map / methods-paper decoding dataset, local ONE cache
- **Date started**: 2026-09-20
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
- `ibl_docs/`
- `methodpaper.pdf`
- `methods.txt`
- `stage_cache.sh`
- `train_decoder.py`

Environment verification: Python 3.13.15; NumPy 2.3.5; PyTorch 2.6.0+cu124. Required imports succeeded. The checkpoint `ls -la /app/CONVERSION_NOTES.md` confirmed the notes file exists.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `prepare_data` | `code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Uses `ONE.eid2pid`, `SpikeSortingLoader`, merges all probes, loads trials/mask and continuous behavior. |
| `load_spiking_data` | `code_zhang2025/src/utils/ibl_data_utils.py` | LOADING/CURATION | Loads spike sorting and merged cluster/channel metadata. `qc=None` keeps all clusters; optional `qc=1` would retain labels >=1, but caching calls the default. |
| `merge_probes` | `code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Reindexes clusters across probes, concatenates, and stable-sorts spikes by time. |
| `load_trials_and_mask` | `code_zhang2025/src/utils/ibl_data_utils.py` | CURATION | Excludes RT <0.08 s or >2 s, missing required events, no-choice trials, and (in `prepare_data`) trial duration >10 s. Keeps unbiased probabilityLeft=0.5 trials. |
| `bin_spiking_data` / `get_spike_data_per_interval` | `code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Stimulus-aligned spike counts in fixed bins; caching parameters use [-0.5,1.5) s and 20 ms (100 bins). |
| `load_target_behavior` | `code_zhang2025/src/utils/ibl_data_utils.py` | LOADING/PROCESSING | Uses brainbox `SessionLoader`; wheel speed is absolute smoothed wheel velocity; whisker motion energy prefers left camera then right fallback. |
| `get_behavior_per_interval` | `code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING/CURATION | Selects samples around event and linearly interpolates to bin endpoints (`align+start+binsize` through end); flags inadequate coverage. |
| `bin_behaviors` | `code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Constructs choice, probability-left block, reward and signed contrast; bins continuous behaviors. |
| `align_spike_behavior` | `code_zhang2025/src/utils/ibl_data_utils.py` | CURATION | Deletes trials failing trial/behavior presence masks and asserts aligned trial counts. |
| `BrainRegions.acronym2acronym(..., mapping='Beryl')` | `code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Maps fine cluster acronyms to Beryl regions. |
| `create_dataset` | `code_zhang2025/src/utils/dataset_utils.py` | PROCESSING | Stores trial spike counts as sparse uint8 matrices and attaches behavior/cluster metadata. |

### Notes
The methods repository is `code_zhang2025`. Its caching entry point is `src/0_data_caching.py`. For brain-wide-map it randomly selects subjects and takes the first release-table EID per selected subject; the supplied task instead asks for the complete locally staged dataset. Core reference parameters are stimulus onset alignment, a 2 s window from -0.5 to +1.5 s, and 20 ms bins. It merges all probes per session and operates on extracellular spikes, so delta-F/F is not applicable. Although good-unit labels are saved, the reference caching call does not filter clusters (`qc=None`); this will be reconciled against the data-paper BWM curation in later steps. Continuous behavioral traces are linearly interpolated. The reference paper treats wheel speed and whisker motion energy as continuous regression targets, whereas this task explicitly requires three categorical bins.

The caching script partitions trials randomly 70/10/20 only for its own Hugging Face output. The required target structure retains all valid trials; downstream `train_decoder.py` performs its own split. Spike arrays in the reference are trial × time × neuron; the requested pickle requires each trial transposed to neuron × time. Choice values in native IBL are typically -1/+1 (with 0 no-choice); the task explicitly remaps left/right to 0/1.

---

## Step 2: Dataset Exploration
**Status**: IN PROGRESS

### Data Structure
The data root contains an IBL ONE cache at `/app/data/one_cache`, with release tables `2022_Q4_IBL_et_al_BWM`, `2025_Q3_IBL_et_al_BWM`, and a composite `Brainwidemap` table directory. Lab/subject/date/session trees are exposed beneath the cache. Exploration used `ONE(cache_dir='/app/data/one_cache', tables_dir=..., mode='local')`, `ONE.search`, `ONE.list_datasets`, `ONE.get_details`, `ONE.load_object`, and brainbox loaders; no source data file was read directly.

The composite cache advertises ALF trial tables, wheel timestamps/positions, left/right camera timestamps and ROI motion energy, and pykilosort spike/cluster/channel objects. Common formats are `.pqt` tables, `.npy` arrays, and cluster UUID `.csv`; these were identified through `ONE.list_datasets`. A test spike-sorting load through `SpikeSortingLoader` succeeded (80,867,860 spikes and 1,557 clusters for one probe), showing the ephys objects are reachable.

**Blocking cache integrity finding:** ONE metadata says `_ibl_trials.table.pqt` exists for 459 sessions, but the files are not reachable through ONE. `SessionLoader.load_trials()` returned only one column (`goCueTrigger_times`) for tested sessions. `ONE.load_object(eid, 'trials', attribute=['choice','probabilityLeft','stimOn_times','firstMovement_times','feedback_times','feedbackType','rewardVolume'])` failed for all 459 candidate sessions; explicit `attribute='table'` reports `ALFObjectNotFound`. Therefore native trial dimensions/dtypes and total trials cannot yet be measured, and wheel/whisker trial alignment cannot be performed. This is a source-cache staging problem rather than a conversion-code issue. Per the ordered workflow, Step 2 remains IN PROGRESS until the registered trial tables are made accessible through ONE.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | Blocked: trial/session inclusion cannot be established; one successfully loaded probe had 1,557 clusters |
| Neurons / session | Blocked pending usable sessions |
| Subjects | 143 composite; 139 in 2025_Q3 release; 115 in 2022_Q4 release |
| Sessions / subject | Composite: min 1, mean 3.3566, max 13; 2025_Q3: min 1, mean 3.3022, max 13 |
| Trials (total) | Blocked: registered trial tables inaccessible through ONE |
| Trials / session | Blocked: registered trial tables inaccessible through ONE |

---

## Step 3: Reference Text Reading
**Status**: [NOT STARTED | IN PROGRESS | COMPLETE]

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | | | 
| Neurons / session | | |
| Subjects | | |
| Sessions / subject | | |
| Trials (total) | | |
| Trials / session | | |
| Neural data time bin | | |
| Behavior data time bin | | |
| Reward rate | | | | 
| <Task/behavior statistic 1> | | | | 
| <Task/behavior statistic 2> | | | |
| ... | | | | 


### Processing Details
[Document temporal alignment, filtering, etc.]

### Curation Steps

**Neuron curation rules**:
[Describe rules]

**Trial curation rules**:
[Describe rules]

### Decoders Trained
| Decoded variable | Accuracy |
| | |

---

## Step 4: Check for Consistency
**Status**: [NOT STARTED | IN PROGRESS | COMPLETE]

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| | | | | |

---

## Step 5: Mapping Planning
**Status**: [NOT STARTED | IN PROGRESS | COMPLETE]

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| | neural | | |
| | input[0] | | |
| | output[0] | | |

### Key Decisions
1. **[Decision]**: [Rationale]

### Planned Sanity Checks
- [ ] Check 1
- [ ] Check 2

---

## Step 6: Script Development
**Status**: [NOT STARTED | IN PROGRESS | COMPLETE]

[Implementation notes]

Code inefficiencies identified:
[Note]

Code speedups added:
[Note]

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
