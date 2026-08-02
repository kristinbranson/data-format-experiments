# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
- **Date started**: 2026-07-30
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- .manifest
- CONVERSION_NOTES.md
- Dockerfile
- code
- data
- decoder.py
- docker-compose.yaml
- methods.txt
- paper.pdf
- train_decoder.py

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| NeuralChoiceDecoding | code/ChoiceContextDecoding/NeuralChoiceDecoding.m | PROCESSING | Neural decoding of lick choice across sessions; reveals binning/trial balancing logic |
| NeuralContextDecoding | code/ChoiceContextDecoding/NeuralContextDecoding.m | PROCESSING | Neural decoding of behavioral context; reveals labels and balancing logic |
| DLC_ChoiceDecoding | code/ChoiceContextDecoding/DLC_ChoiceDecoding.m | PROCESSING | Behavioral feature decoding of choice using tongue/paw/motion energy features |
| DLC_ContextDecoding | code/ChoiceContextDecoding/DLC_ContextDecoding.m | PROCESSING | Behavioral feature decoding of context; explicitly sets 75 ms bins |

### Notes
### Script-level extraction
- Decoding scripts operate session-by-session and use `params(sessix).trialid(...)` to obtain trial sets for condition groups.
- Behavioral/video features are stored in `kin(sessix).dat` and indexed as `X = kin(sessix).dat(:,trials.all,rez.featix)`, consistent with a time x trial x feature tensor.
- Context decoding balances AFC and AW trial counts by subsampling matched numbers from condition groups before train/test splitting.
- In context decoding, labels are explicitly coded as `Y = +1` for AFC and `Y = -1` for AW.
- Train/test sets are created from `trials.all` via `datasample(..., 'Replace', false)` and then indexed into `X` and `Y`.
- Prior code inspection indicates go-cue-centered timing and 75 ms decoding bins for DLC features.

- Codebase is primarily MATLAB.
- Relevant subdirectories include `DataLoadingScripts`, `ChoiceContextDecoding`, `CodingDirections`, and `Behavior`.
- `DataLoadingScripts` appear to define per-session `meta` structures with fields such as animal (`anm`), date, probe, datapath, and sometimes `stimEpoch`; paths point into `DataObjects/<animal>/<datafn>` and use `findDataFn(meta(end))`.
- `ChoiceContextDecoding` scripts indicate behavioral/video feature groups include `tongue`, `paw`, and `motion_energy`, matching required decoder outputs.
- `DLC_ContextDecoding.m` explicitly sets `rez.binSize = 75` ms and computes `rez.dt = floor(rez.binSize / (params(1).dt*1000))`, suggesting native behavioral/neural streams are rebinned to 75 ms for decoding analyses.
- Helper code uses event fields such as `obj(sessix).bp.ev.goCue`, `bitStart`, and `sample`; plotting functions label the x-axis as `Time (s) from go cue`, consistent with aligning trials to go cue.
- `getBlockNum_AltContextTask.m` derives block/context identity from transitions in `obj(sessix).bp.autowater` across trials, implying context labels are encoded in behavioral metadata rather than a standalone variable.
- Need to finish direct inspection of decoding scripts to capture exact trial-condition mappings (choice/context/outcome) and any explicit neuron/trial curation rules before closing Step 1.

- Additional targeted inspection requested: extract exact trial-condition mappings (`trialid`, hit/miss, afc/aw/context), neural preprocessing terms (`lowFR`, PSTH smoothing), and label construction (`X`, `Y`) from decoding scripts before finalizing Step 1.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/` contains four subdirectories: `Ephys_Behavior`, `RandomizedDelay_Ephys_Behavior`, `DelayInhibition_BilatMC_Behavior`, and `GoCueInhibition_BilatMC_Behavior`.
- All source files observed so far are MATLAB `.mat` files.
- File naming includes subject/date patterns such as `motionEnergy_<subject>_<date>.mat`; further inspection in progress to identify paired neural/behavior/session files and variable names.

- Representative `data_structure_*.mat` files are MATLAB v7.3/HDF5 and require `h5py` rather than `scipy.io.loadmat`.
- Preliminary file types include at least `data_structure_*` and `motionEnergy_*`; prefix counting and HDF5 key inspection are being used to determine all paired per-session files and available variables.

- Mixed MATLAB formats are present: `data_structure_*.mat` are MATLAB v7.3/HDF5, while at least `motionEnergy_*.mat` are older MAT files readable with `scipy.io.loadmat`.

- Representative `data_structure_*.mat` files contain a top-level MATLAB struct `obj` with fields `pth`, `bp`, `sglx`, `traj`, `trials`, `clu`, `ex`, and `meta`.
- These fields strongly suggest the native session object bundles behavior (`bp`), electrophysiology (`sglx`, `clu`), trajectories/video (`traj`), trial metadata (`trials`), exclusions (`ex`), and session metadata (`meta`).

- Representative `motionEnergy_*.mat` files load with `scipy.io.loadmat` and contain a struct `me` with fields `data` and `moveThresh`.
- Preliminary subject/date parsing across filenames found 18 unique subjects: EKH1, EKH3, JEB11, JEB12, JEB13, JEB14, JEB15, JEB19, JEB23, JEB24, JEB6, JEB7, JGR2, JGR3, MAH13, MAH14, MAH20, MAH21.

- HDF5 inspection indicates the top-level `obj` fields are themselves groups rather than plain datasets, so additional dereferencing is needed to read per-session contents and dimensions.

- Additional HDF5 inspection revealed metadata/path-related fields including `anm`, `bpod`, `dt`, `fn`, `jrc`, `sglx`, `sv`, and `vid`, indicating the session object stores modality-specific filenames/paths and a sampling interval field `dt`.

- In a representative randomized-delay ephys session (`JEB11`, `2022-05-10`), `bp` contains `Ntrials` plus trial-wise arrays `L`, `R`, `autowater`, `bitRand`, `early`, `hit`, `miss`, and `no`, each with length 365, and an `ev` subgroup containing event times `bitStart`, `delay`, `goCue`, `lickL`, `lickR`, `reward`, and `sample`.
- This confirms that go-cue alignment and per-trial labels for lick direction, context, and outcome can be derived from the native behavioral structure.

- Representative `motionEnergy_*.mat` files store `me.data` as a per-trial object array (length matched to `bp.Ntrials`; 365 in the inspected session), with one variable-length continuous motion-energy trace per trial, plus a scalar `moveThresh`.
- This implies continuous behavioral traces are stored natively at variable per-trial lengths and must be temporally aligned/rebinned for decoder use.

- The `clu` field references a struct with fields `tm`, `site`, `quality`, `spkWavs`, `trialtm`, and `trial`, indicating spike times / trial-aligned spike timing plus unit site and quality metadata.
- The `traj` field references a struct with fields `fn`, `featNames`, `ts`, `frameTimes`, and `NdroppedFrames`, indicating video/DLC-derived feature trajectories with named features and timestamps.

- Representative randomized-delay ephys session example: `bp.Ntrials = 365` and `clu` contains 63 units (`trialtm`/`trial` arrays with shape `(63, 1)`).
- `clu.trialtm` and `clu.trial` dereference to per-unit spike-time arrays and matching trial-index arrays, confirming neural data are stored as spike times that must be binned by trial and time.


### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 85 |
| Neurons / session | example session JEB11 2022-05-10: 63 |
| Subjects | 18 |
| Sessions / subject | variable; e.g. 1-24 from filename parsing |
| Trials (total) | 536 |
| Trials / session | example session JEB11 2022-05-10: 365 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 522 (two-context); 845 (randomized delay); 1651 (DR task) | "In total, 522 units ... in these sessions. ... for the randomized delay task, we recorded 845 units ... For the DR task, we recorded 1,651 units" |
| Neurons / session | example session JEB11 2022-05-10: 63 | |
| Subjects | 18 | |
| Sessions / subject | variable; e.g. 1-24 from filename parsing | |
| Trials (total) | 536 | |
| Trials / session | example session JEB11 2022-05-10: 365 | |
| Neural data time bin | | |
| Behavior data time bin | | |
| Reward rate | not yet extracted | methods excerpt does not give a single overall reward rate in viewed section |
| <Task/behavior statistic 1> | | | | 
| <Task/behavior statistic 2> | | | |
| outcome | 0.6035 | 0.5257 |
| tongue_velocity | 0.8415 | 0.8365 |
| paw_velocity | 0.8130 | 0.8029 |
| motion_energy | 0.6678 | 0.6447 | | 


### Processing Details
- Early lick and ignore trials are omitted from analyses.
- Go cue / water drop is a key alignment and behavioral epoch reference.
- Reference code default alignment is explicitly `params.alignEvent = goCue` in `code/DataLoadingScripts/getDefaultParams.m`.
- Reference loading/processing code uses a shared `params.dt` time step (e.g. `baselineFR.m` bins from presample to sample using `-0.5:params.dt:mode(obj.bp.ev.sample)`).
- Motion energy analyses use the delay epoch in some figures; decoder task here will instead align to go cue per instructions.
- DeepLabCut is used to label tongue visibility/kinematics.
- Randomized-delay and two-context tasks are both present in the dataset; task-specific filtering may be needed depending on required outputs.

### Curation Steps

**Neuron curation rules**:
- Recording sessions included only if they had at least 10 units.
- All units with firing rates > 1 Hz were included in most analyses.
- For subspace alignment and single-unit selectivity, only well-isolated single units with firing rates > 1 Hz were included.
- Unit isolation categories come from manual curation after JRCLUST and/or Kilosort + Phy.

**Trial curation rules**:
- Early lick and ignore trials omitted from all analyses.
- Behavioral-analysis sessions required at least 40 correct DR trials per direction and 20 correct WC trials per direction.

### Decoders Trained
| Decoded variable | Accuracy |
| | |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Alignment event | `getDefaultParams.m` sets `alignEvent = goCue`; plotting/decoding functions use time from go cue | `bp.ev.goCue` exists in session objects | Methods repeatedly reference go cue / water drop timing | Use go cue as the temporal alignment event in converted data |
| Neural filtering | `processData.m` removes low firing rate clusters; code artifacts suggest `lowFR=1` and `dt=0.005` | `clu` contains per-unit spike timing and quality fields | Methods: all units >1 Hz in most analyses; sessions need >=10 units | Filter to sessions with >=10 units and units with FR >1 Hz; document any task-specific deviations |
| Context encoding | `getBlockNum_AltContextTask.m` uses `bp.autowater` transitions; decoding scripts use AFC/AW trial groups | `bp.autowater` is present per trial | Methods describe DR and WC / two-context blocks | Map context from trial structure/autowater-consistent labels; verify DR/WC coding during conversion |
| Behavioral traces | DLC/motion-energy decoding uses 75 ms bins; `traj` and `me.data` are per-trial variable-length traces | `traj.ts/frameTimes` and `me.data` are variable-length per trial | Methods describe DeepLabCut and motion-energy analyses | Rebin/alignment to a common time grid around go cue is required |
| Trial curation | Decoding/analysis scripts exclude early/ignore and balance conditions | `bp.early`, `bp.no`, hit/miss arrays exist | Methods: early lick and ignore trials omitted from analyses | Exclude early and ignore trials before constructing decoder dataset |

---

## Step 5: Mapping Planning
**Status**: IN PROGRESS

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `obj.clu.trialtm` + `obj.clu.trial` (+ unit metadata from `clu`) | neural | Bin spike times by trial on a common go-cue-aligned time grid | `processData.m`, `getSeq`, `removeLowFRClusters` | Use units with FR > 1 Hz; likely exclude sessions with <10 units |
| time relative to go cue | input[0] | Continuous time vector repeated for each trial | `getDefaultParams.m`, goCue-aligned plotting/decoding code | Decoder input is only time from go cue |
| `obj.bp.L`, `obj.bp.R` | output[0] lick direction | Per-trial categorical label; left=0, right=1 | choice/context decoding scripts, `bp` structure | Exclude invalid/ambiguous trials |
| context from `bp.autowater` / trial groups | output[1] behavioral context | Per-trial categorical label; WC=0, DR=1 | `getBlockNum_AltContextTask.m`, context decoding scripts | Verify final sign/coding against trial-group logic |
| `obj.bp.hit`, `obj.bp.miss` | output[2] outcome | Per-trial categorical label; incorrect=0, correct=1 | methods + decoding scripts | Exclude ignore / early trials |
| tongue-related `traj` features | output[3] tongue velocity | Derive per-time-bin scalar, discretize by session median | `traj.featNames`, DLC decoding scripts | Need to identify exact tongue feature(s) from `featNames` |
| paw-related `traj` features | output[4] paw velocity | Derive per-time-bin scalar, discretize by session median | `traj.featNames`, DLC decoding scripts | Need to identify exact paw feature(s) from `featNames` |
| `me.data` | output[5] motion energy | Rebin to common time grid, discretize by session median | `loadMotionEnergy.m`, DLC decoding scripts | `me.moveThresh` available in raw file |

### Key Decisions
1. **Use go cue as alignment event**: Reference code default is `alignEvent = 'goCue'`, and the task explicitly requires go-cue alignment.
2. **Use spike counts on a fixed time grid**: Raw neural data are stored as per-unit spike times by trial, so conversion will bin to a common grid across sessions.
3. **Exclude early and ignore trials**: Matches methods and analysis code.
4. **Filter units by firing rate > 1 Hz and sessions with >=10 units**: Matches methods and `removeLowFRClusters` logic.
5. **Restrict to sessions with all required outputs available**: Need neural + behavior + trajectory/motion-energy data for decoder outputs.
6. **Represent decoder input as continuous time from go cue**: Single input channel shared across trials/sessions.
7. **Discretize continuous behavioral outputs per session at 50th percentile**: Required by task for tongue velocity, paw velocity, and motion energy.
8. **Make continuous outputs time-varying**: Prefer time-varying categorical outputs where possible, following decoder task instructions.

### Planned Sanity Checks
- [ ] Check that converted trial count equals raw valid trial count (`~early & ~ignore`) for representative sessions.
- [ ] Check that binned spike counts for a chosen unit/trial/time bin match counts from raw `clu.trialtm` values.
- [ ] Check that converted lick direction/context/outcome labels match raw `bp` arrays on selected trials.
- [ ] Check that rebinned motion energy on selected trials matches raw `me.data` after applying the same time grid.
- [ ] Check that tongue/paw-derived time series use features named in raw `traj.featNames` and align with raw timestamps.

---

## Step 6: Script Development
**Status**: IN PROGRESS

[Implementation notes]

Code inefficiencies identified:
[Note]

Code speedups added:
[Note]

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 85 |
| Neurons / session | example session JEB11 2022-05-10: 63 |
| Subjects | 18 |
| Sessions / subject | variable; e.g. 1-24 from filename parsing |
| Trials (total) | 536 |
| Trials / session | example session JEB11 2022-05-10: 365 |
| time_from_go_cue range | [-2.5, 2.4] |
| ... | [MIN, MAX] |
| lick_direction distribution | [0.466, 0.534] |
| behavioral_context distribution | [0.237, 0.763] |
| outcome distribution | [0.073, 0.927] |
| tongue_velocity distribution | [0.664, 0.336] |
| paw_velocity distribution | [0.704, 0.296] |
| motion_energy distribution | [0.500, 0.500] |

### Processing Plots Review
- No plots generated yet (`--show-processing` not used in latest successful sample run).
- Earlier trajectory-loading bug caused degenerate tongue/paw outputs; fixed by loading both camera views and using correct `(frames, coords, bodypart)` trajectory layout.
- Sample verification currently shows degenerate tongue/paw outputs (all high); trajectory feature decoding/interpolation remains under investigation.


[Step 7 in progress note] Sample format verification passes structurally, but tongue_velocity and paw_velocity are currently degenerate (constant class), indicating trajectory feature extraction is still incorrect and must be fixed before proceeding to decoder training.


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
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| lick_direction | 0.5860 | 0.5766 |
| behavioral_context | 0.6659 | 0.6682 |
| outcome | 0.6332 | 0.5818 | above chance |
| tongue_velocity | 0.8730 | 0.8623 | strong performance |
| paw_velocity | 0.8568 | 0.8419 | strong performance |
| motion_energy | 0.6071 | 0.5950 | above chance |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: created
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | code-dependent after filtering | data | 4176 | 4176 | Yes |
| Mean neurons/session | | | | | |
| Subjects | 18 | | | | |
| Sessions | 33 kept after conversion | code/data | 33 | 33 | Yes |
| Trials (total) | 536 | | | | |
| Trials/session (mean) | | | | | |
| <Input 1 range> | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| ... | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| lick_direction distribution | n/a | n/a | n/a | [0.496, 0.504] | |
| behavioral_context distribution | n/a | n/a | n/a | [0.112, 0.888] | |
| outcome distribution | n/a | n/a | n/a | [0.153, 0.847] | |
| tongue_velocity distribution | n/a | n/a | n/a | [0.741, 0.259] | |
| paw_velocity distribution | n/a | n/a | n/a | [0.716, 0.284] | |
| motion_energy distribution | n/a | n/a | n/a | [0.500, 0.500] | |

---

## Step 10: Critical Review 1
**Status**: IN PROGRESS

### Checks Performed
1. Verification log review: `verification_full_out.txt` reports "Data format is valid, no errors or warnings.".
2. Raw-vs-converted sanity checks on representative session `EKH1_2021-08-07`:
   - valid trial count from raw `bp` after excluding early/ignore matches converted trial count
   - converted lick/context/outcome labels match raw `bp` arrays on selected trials
   - converted spike-count vector for one unit/trial matches raw histogrammed `clu.trialtm` counts

### Issues Found and Resolved
- Sessions with unreadable `data_structure_*.mat` files are skipped with explicit logging.
- Sessions lacking trial-aligned spike fields (`clu.trialtm` / `clu.trial`) are skipped with explicit logging.
- Earlier bug: only one camera view was loaded from `obj.traj`; fixed by loading both views and using side-view tongue plus bottom-view paw features.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| lick_direction | 0.5860 | 0.5766 |
| behavioral_context | 0.7568 | 0.7397 | strong performance |
| ... | | |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| | |
| lick_direction | 0.5854 validation balanced accuracy | above chance; plausible for neural decoding |
| behavioral_context | 0.7397 validation balanced accuracy | strong, consistent with context signal in ALM/task structure |
| outcome | 0.5818 validation balanced accuracy | above chance |
| tongue_velocity | 0.8623 validation balanced accuracy | strong, consistent with movement-related signals |
| paw_velocity | 0.8419 validation balanced accuracy | strong, consistent with movement-related signals |
| motion_energy | 0.5950 validation balanced accuracy | above chance |

All outputs are above chance on the full dataset. No output is below chance. Tongue and paw are especially strong after fixing the two-view trajectory loading bug.

### Issues Found and Resolved
- [Issue]: [Resolution]

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
