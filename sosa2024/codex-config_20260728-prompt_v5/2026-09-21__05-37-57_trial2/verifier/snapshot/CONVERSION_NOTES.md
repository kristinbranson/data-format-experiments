# Dataset Conversion Notes

## Overview
- **Dataset**: A flexible hippocampal population code for experience relative to reward
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `code/`
- `data/`
- `decoder.py`
- `docker-compose.yaml`
- `methods.txt`
- `paper.pdf`
- `train_decoder.py`

Environment checks:
- `python3` import test passed
- `numpy` version: `2.4.4`
- `torch` version: `2.6.0+cu124`
- Checkpoint verified: `ls -la /app/CONVERSION_NOTES.md` succeeded

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `create_sess` | `code/src/reward_relative/preprocessing.py` | LOADING | Creates a `TwoPUtils.sess.Session` from imaging + VR files and appends aligned scan/VR/suite2p/behavior data. |
| `append_session_data` | `code/src/reward_relative/preprocessing.py` | PROCESSING | Calls VR alignment, loads suite2p outputs, adds behavior time series (`licks`, `rewards`, `speed`) and position-binned trial matrices. |
| `multi_anim_sess` | `code/src/reward_relative/utilities.py` | PROCESSING | Loads prebuilt `sess` pickles for a given experiment day across animals, computes `dff` and optionally deconvolved `events`, adds trial matrices, and stores behavior annotations. |
| `get_trial_types` | `code/src/reward_relative/behavior.py` | PROCESSING | Computes per-trial reward outcome (`isreward`) and environment morph (`morph`) from aligned VR data between trial start and teleport. |
| `get_reward_zones` | `code/src/reward_relative/behavior.py` | PROCESSING | Maps session scene names to per-trial reward-zone coordinates and categorical labels `A/B/C`; handles switch sessions via `change_trial` or `sess.change_reward_trial`. |
| `define_trial_subsets` | `code/src/reward_relative/behavior.py` | PROCESSING | Defines pre/post-switch trial sets or splits fixed-zone sessions in half when two sets are forced. |
| `get_timeseries_data` | `code/src/reward_relative/glmUtils.py` | PROCESSING | Builds continuously sampled behavioral dataframe and neural matrix from a `sess`, aligned on imaging frames; computes relative position, trial ids, reward/omission state, speed and licks, and masks low-speed / invalid samples. |
| `get_omission_inds` | `code/src/reward_relative/rewardAnalysis.py` | PROCESSING | Finds reward-zone entry sample indices on omission trials. |
| `get_omission_trials` | `code/src/reward_relative/rewardAnalysis.py` | CURATION | Identifies “true omissions” as unrewarded trials lacking active `rzone` samples. |
| `sessions_dict` entries | `code/src/reward_relative/sessions_dict.py` | LOADING | Hard-coded session metadata per animal: date, scene, VR session, scan number, and experiment day. |

### Notes
- Reference dataset is 2-photon calcium imaging, not electrophysiology. There is no spike-sorting quality filtering step; instead, cell curation occurs earlier in Suite2p and is saved in `iscell.npy` / `sess.iscell`.
- `sess` is the base aligned object. The repo README says raw `sess` does **not** initially contain dF/F; `multi_anim_sess` computes `dff` and optionally deconvolved `events`.
- The figure-3 decoder notebook uses `glmUtils.get_timeseries_data(..., data_types=['trials', 'rel_pos', 'pos', 'speed', 'rewards'], rel_pos_type='circular', use_speed_thr=2)` and decodes from `events`, not from raw fluorescence or `dff`.
- `multi_anim_sess_README.md` states `timeseries` are sampled at imaging frames, approximately 15.5 Hz (~64.5 ms/sample), and `trial_matrices` are 10 cm position bins over a 450 cm track.
- `append_session_data` adds `licks`, `rewards`, and `speed` from `sess.vr_data` into `sess.timeseries`, then creates position-binned trial matrices. This indicates the time-aligned continuous streams are already present in `sess`.
- `get_timeseries_data` copies `sess.timeseries['events']` into a continuous neural matrix, fills valid trial segments using `trial_start_inds` and `teleport_inds`, computes:
  - `rel_pos`: position relative to reward zone, either linear or circular.
  - `pos`: absolute position on the 450 cm track.
  - `trial_ids`: per-sample trial number.
  - `was_reward`: 0 before reward delivery and 1 after reward delivery on rewarded trials.
  - `was_omission`: 0 before reward-zone entry and 1 after entry on true omission trials.
  - `licks`: thresholded to binary after sensor-error cleanup.
  - `speed`: NaN-masked below a 2 cm/s threshold in the decoder notebook.
- Important indexing detail from reference code: per-trial continuous segments are often filled using `start-1:stop-1`, so trial start and teleport indices behave like 1-based boundaries inherited from the aligned session object.
- Reward-zone labels are determined from scene names. Canonical track reward-zone labels for this task are `A`, `B`, `C`; internal coordinate dictionaries include environment-specific coordinate sets (`X/Y/Z`) that map back to these labels.
- Switch sessions use the first 30 trials as pre-switch and later trials as post-switch in the decoder notebook (`trial <= 29` vs `> 29` on zero-indexed `trial_ids`).
- External dependency noted but not inspected here to respect project constraints: `TwoPUtils.preprocessing.vr_align_to_2P` performs the original VR-to-imaging alignment and is referenced by the repo documentation as the authoritative alignment function.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` contains a DANDI-style NWB dataset with one folder per subject and one NWB file per session.
- Top-level files:
  - `dandiset.yaml`
  - subject folders: `sub-m3`, `sub-m4`, `sub-m7`, `sub-m11`, `sub-m12`, `sub-m13`, `sub-m14`, `sub-m15`, `sub-m17`, `sub-m18`, `sub-m19`
- Each NWB filename follows `sub-<mouse>_ses-<NN>_behavior+ophys.nwb`.
- No README/text docs are present inside `/app/data`; the data are self-described through NWB metadata.
- Example session metadata from NWB:
  - `subject.subject_id`: e.g. `m3`
  - `session_id`: e.g. `03`
  - `identifier`: original source path string, e.g. `/data/InVivoDA/GCAMP3/03_10_2022/Env1_LocationC_to_A`
    - this encodes the original animal ID, recording date, and scene string needed to recover reward-zone condition.
- Behavioral streams live in `processing/behavior/BehavioralTimeSeries`.
- Ophys streams live in `processing/ophys`:
  - `Fluorescence`
  - `Deconvolved`
  - `Neuropil`
  - `ImageSegmentation`
- Imaging data are stored as ROI response series with shape `(n_timepoints, n_rois)` per plane.
- Segmentation table `PlaneSegmentation` contains:
  - ROI masks (`pixel_mask` in single-plane files or `voxel_mask` in multi-plane files)
  - `iscell`: Suite2p two-column array where column 0 is the curated binary cell flag and column 1 is the confidence/probability
  - `planeIdx`: plane assignment per ROI
- Behavior variables available per frame:
  - `environment`
  - `lick`
  - `position`
  - `reward_zone`
  - `scanning`
  - `speed`
  - `teleport`
  - `trial number`
  - `trial_start`
  - `autoreward`
- Reward deliveries are also available as an event-style series `Reward` with its own timestamps and reward amounts (`0.004 mL` entries in the inspected example).
- Native behavior quirks observed directly in the files:
  - pre-synchronization samples use sentinel values like `environment = -1`, `trial number = -1`, `scanning = -1`, `position = -500`
  - `trial number` is 0-indexed once valid
  - `trial_start` and `teleport` are binary pulse series
- Ophys sampling rates observed across the dataset: `15.5078125 Hz` and `31.015625 Hz`.
- Session organization by plane:
  - 124 single-plane sessions (`plane0` only)
  - 28 multi-plane sessions (`plane0` and `plane1`)
- In multi-plane files, response-series column counts across planes sum exactly to the segmentation table ROI count.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 138,678 curated cells (`iscell[:,0] == 1`) |
| Neurons / session | mean 912.36, range 155-2341 curated cells |
| Subjects | 11 |
| Sessions / subject | 12 for `sub-m11`, 14 for all other subjects |
| Trials (total) | 12,216 (`trial_start` pulses) |
| Trials / session | mean 80.37, range 41-100 |

Additional size notes:
- Total NWB files / sessions: 152
- Total segmented ROIs before curation: 312,110 (mean 2053.36/session, range 315-5085)
- Mean frames/session: 23,755.77 (range 14,164-51,520)
- Total reward events (`Reward` series length summed across files): 10,345
- Environment codes present in valid samples: `0`, `1`
- Trial-count consistency from raw data:
  - `trial_start` sum equals `teleport` sum in all 152 sessions
  - `trial_start` sum matches `max(trial number)+1` in 151/152 sessions
  - the lone mismatch is `sub-m11_ses-03`, where `trial number` indicates 81 contiguous labeled segments but `trial_start` and `teleport` each indicate 80 full trials; inspection shows the extra trial-number segment is a post-teleport partial tunnel segment, so pulse-defined starts/stops remain the correct trial boundaries

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Not explicitly stated for all sessions | No direct total reported in provided text |
| Neurons / session | 155-2172 putative pyramidal neurons/session after manual curation | “This approach yielded 155–2172 putative pyramidal neurons per session” |
| Subjects | 11 switch-task mice | “Each mouse encountered a different starting reward zone and sequence of reward zone switches, counterbalanced across mice (n = 11 mice).” |
| Sessions / subject | 14 task days planned; m11 imaging began on day 3 | “for a total of 14 days” and “The task was imaged starting from day 1 for all mice except m11, for whom imaging started on day 3” |
| Trials (total) | Not explicitly stated | No direct total reported in provided text |
| Trials / session | 80-100 targeted; mean 80.5 ± 7.4 | “We targeted 80–100 trials per session… (mean ± s.d., 80.5 ± 7.4 trials across 14 mice, all imaging days).” |
| Neural data time bin | ~15.5 Hz imaging frames (~64.5 ms/frame) | “All behavioral and neural time series were sampled at ~15.5 Hz, the imaging frame rate.” |
| Behavior data time bin | ~15.5 Hz after alignment to imaging | “All behavioral and neural time series were sampled at ~15.5 Hz” |
| Reward rate | ~85% rewarded, ~15% omitted | “Reward was randomly omitted on approximately 15% of trials” |
| Reward zone locations | A: 80-130 cm; B: 200-250 cm; C: 320-370 cm | “zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm” |
| Track length | 450 cm | “Both environments consisted of a 450 cm linear track” |
| Switch trial within session | after 30 trials | “On day 3 (switch one), the zone was moved after 30 trials” / “Each switch occurred after 30 trials.” |
| Automatic reward after switch | first 10 trials of new condition if no lick in zone | “On the first ten trials of any new condition… the reward was automatically delivered at the end of the zone if the mouse had not yet licked within the zone” |
| Multi-plane imaging | 2 mice (m17, m18), 2 planes separated by ~27 µm, ~31 Hz interleaved / ~15.5 Hz per plane | “in two mice (m17 and m18), in two planes separated by ~27 µm, with frames bidirectionally imaged at ~31 Hz interleaved in the scan for a sampling rate of ~15.5 Hz per plane.” |
| Place cells on switch sessions | 459 ± 263 out of 954 ± 453 cells imaged across 11 mice, 7 switch sessions | “459 ± 263 place cells out of 954 ± 453 cells imaged; 48.5 ± 14.5% identified as place cells” |
| RR cell fraction | 16.3 ± 5.3% of place cells over switch days | “mean ± s.d., 16.3 ± 5.3% of place cells averaged over all switch days” |


### Processing Details
- Imaging/behavior alignment:
  - Behavioral and neural streams are sampled on the imaging frame grid (~15.5 Hz) after synchronization.
- Trial structure:
  - Trial runs from track start at 0 cm to teleport entry at track end.
  - Teleport period includes a gray jitter/tunnel interval and can extend longer after omissions or missed licks.
- Reward-relative coordinates:
  - Reward-relative position is defined relative to the **start** of the active reward zone.
  - In the GLM description, RR position spans `-pi` to `pi`, centered at reward-zone start.
- Temporal smoothing / derived neural signal:
  - dF/F baseline is computed independently within each trial using a maximin procedure with a 20 s sliding window.
  - dF/F is Gaussian-smoothed with a two-sample (~0.129 s) s.d. kernel.
  - Deconvolved activity is extracted with OASIS and is the main “activity rate” used in analyses.
- Speed-related masking:
  - For omission/reward comparisons in Fig. 6, neural data at speeds `< 2 cm/s` were excluded.
  - The reference Fig. 3 decoder notebook likewise calls `get_timeseries_data(... use_speed_thr=2)`.
- Spatial binning in reference analyses:
  - Spatial trial matrices use 10 cm linear position bins.
  - Several analyses smooth those binned maps with 10 cm Gaussian kernels.
- Trial grouping:
  - Switch sessions are analyzed as pre-switch versus post-switch, with the switch occurring after 30 trials.

### Curation Steps

**Neuron curation rules**:
- Manual Suite2p curation removed ROIs:
  - containing multiple somata or dendrites
  - lacking visually obvious transients
  - suspected of indicator overexpression
  - with high continuous fluorescence typical of putative interneurons
- Additional putative interneurons were excluded if Pearson correlation between dF/F and running speed exceeded 0.5.
- Multi-plane animals pool planes for nearly all analyses.

**Trial curation rules**:
- Omission trials occur randomly on ~15% of laps.
- Switch occurs after 30 trials on designated switch days.
- For omission analyses in the paper, a trial set needed at least 3 omission trials.
- In the reference decoder notebook, switch-day analyses focus on 77 sessions = 11 mice × 7 switch days.
- Reference continuous-time decoder downsampled occupancy by reward-relative position to match before/after trial-set occupancy.

### Decoders Trained
| Decoded variable | Accuracy |
| Reward-relative position from RR/TR/non-RR populations (Fig. 3) | Paper reports decode score relative to shuffle, not standard classification accuracy. Key claim: when trained before switch and tested after switch, only RR cells remained above shuffle. |
| Single-neuron deconvolved activity from task/movement variables (GLM, Fig. 7) | Fraction deviance explained: all place cells `0.10 ± 0.19`, TR `0.32 ± 0.13`, RR `0.29 ± 0.11`, non-RR remapping `0.29 ± 0.11`. |

Notes on decoder expectations from paper:
- Fig. 3 caption: “RR population provides accurate decoding of RR position.”
- Main text: “when we tested on trials after the reward switch, only the RR population decoded the animals’ RR position better than shuffle.”
- Main text: above-shuffle RR decoding extended over most of the environment with z-scored decode >2 from `-104.5 ± 20.1 cm` to `+152.7 ± 22.9 cm` relative to reward-zone start.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Reward-zone naming | `behavior.get_reward_zones` uses internal coordinate keys `X/Y/Z` but returns labels `A/B/C` based on scene string | NWB stores no explicit `A/B/C` label; only `identifier` contains scene name and behavior `position`/`reward_zone` time series are present | Reward zones are `A=80–130`, `B=200–250`, `C=320–370 cm` | Parse scene from `nwb.identifier`, then use the reference code’s scene-to-label logic. Treat `A/B/C` as the canonical categorical labels, not `X/Y/Z`. |
| Trial boundaries | Reference code uses `sess.trial_start_inds` and `sess.teleport_inds` arrays | NWB stores per-frame pulse series `trial_start` and `teleport`, plus `trial number` | Methods describe laps from track start to teleport and switch after 30 trials | Reconstruct trial starts/stops from pulse indices; use `trial_start`/`teleport` pulses as authoritative. The one trial-count mismatch in `sub-m11_ses-03` comes from an extra trailing `trial number` segment in the tunnel after the last completed trial, not from a missing full trial. |
| Reward delivery representation | Reference code uses `sess.timeseries['rewards']` as a frame-aligned binary series | NWB stores reward deliveries as event series `Reward` with timestamps, not a lowercase per-frame reward vector | Reward is delivered on rewarded trials; omissions ~15% | Rasterize `Reward.timestamps` onto the behavior/imaging frame timestamps. In inspected data the reward timestamps exactly match frame timestamps (`max_abs_time_diff = 0`). |
| Behavior alignment | Reference `sess.vr_data` is already aligned to imaging frames via `vr_align_to_2P` | NWB behavior streams already have one sample per imaging frame and shared timestamps | “All behavioral and neural time series were sampled at ~15.5 Hz, the imaging frame rate.” | Treat NWB behavioral streams as already aligned products of the original synchronization step; no extra interpolation is needed. |
| Multi-plane sampling rate | Code/paper describe two-plane sessions interleaved at ~31 Hz for an effective ~15.5 Hz per plane | NWB `TwoPhotonSeries.rate = 62.03125`, per-plane deconvolved series report `31.015625`, but behavior timestamps advance at `1/15.5078125 s` and deconv sample count matches behavior sample count exactly | Paper says two-plane imaging is interleaved at ~31 Hz with ~15.5 Hz per plane | Use behavior timestamps / frame count as the true decoder time base. Ignore the misleading deconvolved `rate` metadata in multi-plane exports and treat each sample row as one ~15.5 Hz frame-aligned observation per plane. |
| Subject/session counts | Code metadata define 14 experiment days per mouse, with m11 missing early imaging | NWB contains 152 sessions: 12 for `m11`, 14 for all other 10 mice | Paper says m11 imaging began on day 3 | Counts are consistent once m11’s delayed imaging start is accounted for. |
| Interneuron filter threshold | Helper `spatial.is_putative_interneuron` defaults to `r_thresh=0.3`, but `dayData` documents `int_thresh: 0.5` | NWB `iscell` only captures Suite2p/manual ROI curation, not this later speed-correlation exclusion | Methods state additional putative interneurons were excluded by speed correlation `>0.5` | If applying post-manual interneuron exclusion to match later analyses, use `0.5`, not the helper default. This is the manuscript-consistent threshold. |
| Neuron count upper bound | Reference text says 155–2172 putative pyramidal neurons/session | NWB curated `iscell` counts range 155–2341; only multi-plane mouse `m18` exceeds 2172 | Paper also says multi-plane animals were pooled across planes | Single-plane NWB sessions fall within the reported range (max 1780). The modest overage occurs only in pooled two-plane exports and likely reflects export/version differences rather than a fundamental mismatch. Keep this in mind during later statistics checks and revisit after any post-hoc interneuron exclusion. |
| Environment coding | Code uses morph / env logic (`Env1 -> 0`, `Env2 -> 1`) | NWB `environment` valid values are `0` and `1` after sync, `-1` pre-sync | Paper describes ENV 1 and ENV 2 | Use `environment` directly as binary ENV1/ENV2 after removing invalid pre-sync samples. |

Final understanding after reconciliation:
- The NWB files are already the aligned, post-Suite2p, per-frame data representation that the reference code would operate on through `sess`.
- Continuous decoder-ready signals should be reconstructed on the common behavior frame timestamps, not from any nominal ophys `rate` field in multi-plane files.
- Reward-zone label per trial must be inferred from session scene and switch schedule, using the reference code’s logic, because NWB does not store the categorical `A/B/C` label directly.
- Reward outcome per trial and lick/reward time series can be rebuilt exactly from the NWB behavior streams without approximation.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `processing/ophys/Deconvolved/roi_response_series[plane*].data` | `neural` | Concatenate planes along ROI axis if needed, keep curated cells with `iscell[:,0] == 1`, transpose per trial to `(n_neurons, n_timepoints)` | `utilities.multi_anim_sess`, `glmUtils.get_timeseries_data` | Use deconvolved activity (“events”) as the neural signal, matching the reference decoder notebook. |
| `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell[:,0]` | `neural` curation | Binary include mask | Manual curation described in Methods; `sess.iscell` in code | Minimum neuron curation is Suite2p/manual `iscell`. Post-hoc interneuron exclusion remains a candidate additional filter to evaluate in implementation. |
| `processing/ophys/ImageSegmentation/PlaneSegmentation/planeIdx` | metadata/session info | Preserve per-cell plane assignment in metadata | `sess.plane_per_cell` | Not a brain region; useful supplemental metadata for multi-plane sessions. |
| `behavior.position.timestamps` and trial start time | `input[0]` | `time_from_trial_start_s = timestamps - trial_start_timestamp` per frame | Aligned session logic in `get_timeseries_data` | Time-varying continuous input. |
| `behavior.environment` | `input[1]` | Framewise constant within a trial; use first valid value in trial, repeat across frames | `get_trial_types` / `env_morph_dict` | Binary ENV1/ENV2 code already present as 0/1. |
| `behavior.trial number` | `input[2]` | Use the pulse-paired trial order / native 0-indexed completed-trial number, repeat across frames | `glmUtils.get_timeseries_data` (`trial_ids`) | Keep 0-indexing to match reference code and switch-at-trial-30 logic; boundaries come from pulses, not trial-number transitions, because trial numbers extend into tunnel periods. |
| Reward outcome of previous trial, derived from `Reward` event timestamps grouped by trial | `input[3]` | Binary per trial, repeated across frames | `get_trial_types` conceptually | First trial has no history; set previous outcome to `0` and document as a boundary convention. |
| `behavior.position` + inferred active reward-zone interval | `output[0]` | Signed distance to nearest point in active reward zone; discretize into 7 bins | `behavior.get_reward_zones`; paper reward-zone definitions | Use zone interval, not distance to zone start. Inside zone => distance `0`. |
| `behavior.position` | `output[1]` | Discretize absolute position into 5 bins across 450 cm track | General task description | Time-varying output. |
| `behavior.speed` | `output[2]` | Discretize into 5 speed bins | `glmUtils.get_timeseries_data` uses same aligned speed stream | Keep all valid speeds; do **not** apply the paper’s <2 cm/s exclusion because speed itself is a decoder output here. |
| `behavior.lick` | `output[3]` | Convert to binary lick/no-lick per frame after sensor-error handling and clipping | `glmUtils.get_timeseries_data` | Native stream is cumulative lick count per frame; convert any positive count to `1`. |
| Active reward-zone label inferred from scene + pre/post-switch trial index | `output[4]` | Map `A/B/C` -> `0/1/2`, repeat across frames | `behavior.get_reward_zones` | Needed because NWB does not store the categorical reward-zone label directly. |
| Reward event presence per trial from `Reward` timestamps | `output[5]` | Binary per trial (`any reward event in trial`), repeat across frames | `get_trial_types` (`isreward`) | Rewarded includes auto-rewarded trials. |

### Key Decisions
1. **Neural signal = deconvolved activity, not raw fluorescence or dF/F**: The reference decoder notebook explicitly decodes from `events`, and the paper describes deconvolution as the activity-rate representation used in downstream analyses.
2. **Trial alignment event = trial start**: This is required by the user task and is compatible with the paper’s lap structure and the reference code’s use of `trial_start_inds` / `teleport_inds`.
3. **Trial window = start index inclusive, teleport index exclusive**: This reproduces the reference code’s effective `start-1:stop-1` slicing semantics after translating to 0-based NWB indices.
4. **Per-trial constants will be repeated across time**: To keep `input` and `output` arrays uniform as `(n_channels, n_timepoints)`, trial-wise variables (`environment`, `trial number`, `previous outcome`, `reward zone`, `reward outcome`) will be broadcast across all frames in the trial.
5. **Use pulse-defined trial boundaries and native completed-trial numbering**: `trial number` is useful as a label, but it extends into tunnel periods and can create extra partial segments; the reference code’s `trial_start_inds` / `teleport_inds` abstraction is better matched by the pulse streams.
6. **Scene parsing will drive reward-zone identity**: The NWB export omits categorical reward-zone labels, so I will reconstruct them from the `identifier` scene string using the same scene logic as `behavior.get_reward_zones`.
7. **Environment type will come from the aligned `environment` stream, not from scene-name prefixes alone**: This is necessary for cross-environment switch sessions, where ENV changes mid-session after trial 30.
8. **Do not exclude low-speed frames globally**: The reference RR decoder used `speed_thr=2`, but the new decoder must predict speed including the `<2 cm/s` bin, so retaining these frames is a task-required deviation.
9. **Distance-to-reward-zone will be defined relative to the active interval, not just the reward-zone start**: This best matches the user’s wording “distance to any location in the reward zone,” making all in-zone positions class `3`.
10. **Brain region will be encoded as `CA1` for all cells**: Deep/superficial plane identity is plane metadata, not a distinct named brain region in the sense of the target schema.
11. **Minimum neuron filter = `iscell[:,0] == 1`; evaluate additional speed-correlation interneuron exclusion during implementation**: This keeps conversion faithful to manual curation while leaving room to match the paper’s later analysis filter if statistics suggest it is necessary.
12. **Lick sensor errors will be handled using the reference heuristic**: Trials/samples with obvious cumulative-sensor failure should not be silently treated as real lick bursts.

### Planned Sanity Checks
- [ ] Raw-to-converted neural spot check: choose a session, curated ROI, trial, and frame index; verify converted `neural` equals the corresponding raw deconvolved NWB sample with `np.allclose()`.
- [ ] Trial-boundary check: verify that converted trial starts/stops reproduce the raw `trial_start` / `teleport` pulses and trial-number segments, including the clipped-start session.
- [ ] Reward rasterization check: verify framewise reward vector reconstructed from `Reward.timestamps` matches exact raw reward event timestamps with `np.allclose()` / equality.
- [ ] Reward-zone mapping check: compare scene-derived reward-zone start positions to empirical first `reward_zone > 0` positions in raw data on rewarded trials using `np.allclose(..., atol=<few cm>)`.
- [ ] Input broadcast check: for several trials, confirm repeated per-trial inputs are constant over frames and equal to raw trial metadata.
- [ ] Output discretization check: manually verify several raw position/speed/lick samples fall into the intended output bins.
- [ ] Dataset-statistics check: subjects, sessions, trials/session, omission fraction, and neuron/session counts should remain consistent with paper/data after conversion.

Detailed conversion plan:
- Sessions to include:
  - include all 152 imaging sessions in the NWB dataset
  - no fixed-condition mice are present in `/app/data`, so the dataset already matches the 11-mouse switch cohort
- Trial extraction:
  - identify pulse-defined trial starts from `trial_start > 0` and pulse-defined trial ends from `teleport > 0`
  - pair starts/stops in order and use `[start, stop)` as the trial slice
  - use the native trial-number stream only to label each completed trial (0-indexed), not to define boundaries
  - ignore tunnel-only partial trial-number segments that occur before the first start pulse or after the final teleport pulse
- Neural extraction:
  - for single-plane sessions, use `plane0`
  - for two-plane sessions, concatenate `plane0` then `plane1` in `planeIdx` order
  - retain only curated cells according to `iscell[:,0] == 1`
- Temporal sampling:
  - use the frame timestamps attached to behavior streams as the authoritative time axis
  - record metadata `time_bin_size` from the common frame interval (~64.4836 ms)
- Invalid data handling:
  - exclude pre-sync sentinel samples (`environment=-1`, `trial number=-1`, `position=-500`) by working only within extracted trial windows
  - handle lick sensor artifacts using the reference threshold before constructing binary lick output
- Output bin conventions:
  - absolute-position bins will be half-open `[0,90)`, `[90,180)`, `[180,270)`, `[270,360)`, `[360, inf)`
  - speed bins will be `<2`, `[2,10)`, `[10,20)`, `[20,40]`, `>40`
  - reward-distance bins will use exact `0` for in-zone samples and signed interval distance elsewhere:
    - `d < -50 -> 0`
    - `-50 <= d < -10 -> 1`
    - `-10 <= d < 0 -> 2`
    - `d == 0 -> 3`
    - `0 < d <= 10 -> 4`
    - `10 < d <= 50 -> 5`
    - `d > 50 -> 6`

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with:
- CLI: `python -u /app/convert_data.py <outpicklefile> [--sample|--full] [--show-processing]`
- NWB loading via `pynwb`
- Trial extraction from `trial_start` / `teleport` pulses
- Reward rasterization from `Reward.timestamps`
- Reward-zone reconstruction from scene strings using reference-code-equivalent logic
- Deconvolved neural extraction from curated cells only (`iscell[:,0] == 1`)
- Trial-aligned input/output construction with per-trial values repeated across frames
- Processing plots for up to two sessions

Code inefficiencies identified:
- Re-reading full sessions for plots would be expensive on the full dataset.
- Saving neural arrays as float32 would roughly double output size versus float16.
- Using trial-number transitions for trial segmentation would require extra cleanup of tunnel segments and create avoidable complexity.

Code speedups added:
- Neural data saved as `float16` to reduce output size roughly 2x while remaining numeric/floating for the decoder.
- Only curated cells are read from NWB response series; uncurated ROIs are skipped at load time.
- Trial segmentation uses pulse streams directly, avoiding extra per-frame grouping logic.
- Sample/full plots are limited to at most two sessions.
- Large-session benchmark (`sub-m18_ses-03`, 2341 neurons, 80 trials) converts in under 1 s without plotting overhead.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 323 |
| Neurons / session | 155, 168 |
| Subjects | 1 (`m11`) |
| Sessions / subject | 2 |
| Trials (total) | 160 |
| Trials / session | 80, 80 |
| `time_from_trial_start_s` range | [0.0, 30.6] |
| `environment_type` range | [0.0, 0.0] |
| `trial_number` range | [0.0, 79.0] |
| `previous_trial_outcome` range | [0.0, 1.0] |
| `distance_to_reward_zone` distribution | [0.088, 0.092, 0.058, 0.285, 0.022, 0.071, 0.384] |
| `absolute_position` distribution | [0.312, 0.207, 0.203, 0.146, 0.133] |
| `speed` distribution | [0.121, 0.062, 0.066, 0.233, 0.518] |
| `lick` distribution | [0.820, 0.180] |
| `reward_zone_location` distribution | [0.792, 0.208, 0.000] |
| `reward_outcome` distribution | [0.161, 0.839] |

### Processing Plots Review
- Created:
  - `processing_sub-m11_ses-03.png`
  - `processing_sub-m11_ses-04.png`
- No obvious temporal misalignment observed from the generated plots:
  - reward events align to positions within the active reward zone
  - trial windows cover the track traversal and exclude teleport segments
  - discretized outputs span the expected label sets for these two sessions
- No lick-sensor-corrected trials were needed in either sample session (`bad_lick_trials = 0` for session 0).

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| `float16` neural storage | ~2x smaller neural payload in output pickle |
| Curated-cell-only loading | avoids loading ~50%+ extra uncurated ROIs |
| Plotting limited to sample mode / first 2 sessions | avoids expensive repeated plotting in full conversion |
| Pulse-based trial extraction | simpler and faster than trial-number regrouping |

| Step | Time / Session | Estimated Total Time |
| | | |
| Small single-plane conversion (`sub-m11_ses-03`) | ~0.44 s/session without plotting | |
| Large two-plane conversion (`sub-m18_ses-03`) | ~0.93 s/session without plotting | |
| Full conversion (rough estimate) | weighted by large-session benchmark and total neural payload | likely a few minutes; comfortably below 15 min |

Supporting file checks:
- `sample_data.pkl` created, size `9.7M`
- `conversion_sample_out.txt` created
- `verification_sample_out.txt` created
- Sample pickle inspection:
  - session 0 trial 0 shapes: neural `(155, 290)`, input `(4, 290)`, output `(6, 290)`
  - dtypes: neural `float16`, input `float32`, output `int16`

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `distance_to_reward_zone` | 0.4262 | 0.3315 |
| `absolute_position` | 0.5077 | 0.4426 |
| `speed` | 0.3680 | 0.3324 |
| `lick` | 0.7309 | 0.6724 |
| `reward_zone_location` | 0.9644 | 0.9617 |
| `reward_outcome` | 0.6088 | 0.6677 |

Sample-training checks:
- Loss decreased from `59.397` at epoch 1 to `0.915` at epoch 200.
- Validation balanced accuracies exceeded uniform-chance baselines for every output:
  - reward distance: `0.3315` vs chance `0.1429`
  - position: `0.4426` vs chance `0.2000`
  - speed: `0.3324` vs chance `0.2000`
  - lick: `0.6724` vs chance `0.5000`
  - reward zone: `0.9617` vs chance `0.3333`
  - reward outcome: `0.6677` vs chance `0.5000`

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: `4.6G`
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | Not explicitly stated | Curated-cell workflow | 138,678 curated cells | 138,678 | Yes |
| Mean neurons/session | Not explicitly stated | Curated-cell workflow | 912.36 | 912.36 | Yes |
| Subjects | 11 switch-task mice | 11 switch-task mice in sessions metadata | 11 | 11 | Yes |
| Sessions | 14 days per mouse, except m11 starts on day 3 -> 152 imaging sessions | session metadata imply 152 imaging sessions | 152 | 152 | Yes |
| Trials (total) | Not explicitly stated | pulse-defined laps | 12,216 | 12,216 | Yes |
| Trials/session (mean) | 80.5 ± 7.4 | ~80 target | 80.37 | 80.37 | Yes |
| `time_from_trial_start_s` range | variable by behavior | trial-time input | [0.0, 216.5] | [0.0, 216.5] | Yes |
| `environment_type` range | {ENV1, ENV2} | {0,1} | [0.0, 1.0] | [0.0, 1.0] | Yes |
| `trial_number` range | ~0-99 depending session | 0-indexed in decoder code | [0.0, 99.0] | [0.0, 99.0] | Yes |
| Reward outcome yes-fraction | ~0.85 | rewarded vs omission logic | 0.843 | 0.843 | Yes |
| Reward zone distribution | counterbalanced A/B/C across mice | scene-dependent | [0.329, 0.337, 0.335] | [0.329, 0.337, 0.335] | Yes |

Notes:
- `verification_full_out.txt` reported: “Data format is valid, no errors or warnings.”
- Global output distributions are sensible:
  - distance to reward zone: `[0.253, 0.102, 0.074, 0.237, 0.021, 0.072, 0.242]`
  - absolute position: `[0.211, 0.178, 0.231, 0.227, 0.154]`
  - speed: `[0.122, 0.088, 0.134, 0.318, 0.338]`
  - lick: `[0.780, 0.220]`
- The converted subject/session/trial counts exactly match the direct raw-data audit.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Reviewed `/app/verification_full_out.txt`. Result: `train_decoder.py --verify-only` reported “Data format is valid, no errors or warnings.” No unresolved warnings remained.
2. **Raw-data sanity checks with `np.allclose()`**:
   - Neural, single-plane spot check: `sub-m11_ses-03`, trial 0, first 10 curated neurons, first 50 frames. Raw deconvolved NWB samples matched converted `neural` exactly after the expected `float32 -> float16 -> float32` round-trip: `np.allclose(expected_float16_roundtrip, converted, atol=0, rtol=0) == True`.
   - Neural, multi-plane edge-case spot check: `sub-m17_ses-04`, trial 10, first 8 curated neurons, first 40 frames. After trimming the known extra final neural frame to the behavior length, the raw concatenated plane0+plane1 activity again matched converted `neural` exactly after the same float16 round-trip check.
   - Input check: `sub-m11_ses-03`, trial 0 `time_from_trial_start_s` matched raw frame timestamps relative to the trial-start frame with `np.allclose(..., atol=1e-6) == True`.
   - Input check: `sub-m11_ses-03`, trial 40 `environment_type` matched the raw aligned environment stream over the whole trial with `np.allclose(..., atol=0) == True`.
   - Output check: `sub-m11_ses-03`, trial 40 `distance_to_reward_zone` matched bins recomputed directly from raw position and the active reward-zone interval with `np.allclose(..., atol=0) == True`.
   - Output check: `sub-m11_ses-03`, trial 41 `previous_trial_outcome` matched the reward outcome of trial 40 with `np.allclose(..., atol=0) == True`.
   - Output check: `sub-m11_ses-03`, trial 40 `reward_outcome` matched the raw rasterized reward events with `np.allclose(..., atol=0) == True`.
   - Output check: `sub-m11_ses-03`, reward-zone labels matched the scene-derived expected categorical zone (`trial 0 = B`, `trial 40 = A`) with `np.allclose(..., atol=0) == True`.
3. **Reference code comparison**:
   - Data loading: conversion uses NWB `processing/behavior` and `processing/ophys/Deconvolved`, matching the reference code’s aligned `sess.timeseries` and `events` usage.
   - Neuron filtering: conversion keeps manually curated cells with `iscell[:,0] == 1`, matching the minimum curation present in the reference workflow. I did not apply the later speed-correlation interneuron exclusion because the decoder task here is session-wide and not restricted to place-cell subsets; the converted neuron counts and decoder verification remained internally consistent.
   - Trial filtering/alignment: conversion reconstructs trials from `trial_start` and `teleport` pulses, which is the NWB analogue of `sess.trial_start_inds` and `sess.teleport_inds`. This correctly avoids counting the extra tunnel-only `trial number` segment in `sub-m11_ses-03`.
   - Binning/time base: conversion uses the shared behavior-frame timestamps at ~15.5 Hz, matching the aligned framewise time base described by the paper and code. This is preferable to relying on the misleading multi-plane `rate` metadata.
   - Input construction: `time_from_trial_start_s`, environment, trial number, and previous trial outcome are all built from the same aligned framewise/sessionwise quantities used conceptually in `glmUtils.get_timeseries_data`.
   - Output construction: reward-zone identity uses the same scene/switch logic as `behavior.get_reward_zones`; reward outcome is rebuilt from reward events; distance/position/speed/lick are direct discretizations of aligned behavioral streams required by the decoder task.
4. **Key statistics comparison**:
   - Subjects: paper 11, raw data 11, converted 11.
   - Sessions: paper-consistent 152 imaging sessions, raw data 152, converted 152.
   - Trials: raw pulse-defined total 12,216, converted 12,216.
   - Curated neurons: raw `iscell[:,0] == 1` total 138,678, converted 138,678.
   - Reward rate: paper/methods about 15% omitted, converted `reward_outcome=no` fraction `0.157`, consistent.
   - Reward-zone occupancy: converted per-frame zone fractions `[0.329, 0.337, 0.335]`, confirming the scene-derived A/B/C mapping is balanced as expected from counterbalancing across mice.
5. **Edge-case review**:
   - One-frame neural/behavior mismatch: 10 multi-plane sessions had deconvolved arrays longer than the aligned behavior streams by exactly 1 frame. Fix implemented in `convert_data.py`: trim the final neural frame only when `deconv_len == behavior_len + 1`; otherwise raise an error. After this fix, full conversion and full verify-only both succeeded.
   - Trial-number off-by-one artifact: `sub-m11_ses-03` contains 81 labeled `trial number` segments but only 80 real start/teleport-defined trials. Retaining pulse-defined trial windows prevented a false extra trial.
   - Long trials: some sessions contain very long completed laps (up to `216.5 s`, `T=3359`). These are preserved because they are present in the aligned raw data and do not violate format checks.

### Issues Found and Resolved
- **Multi-plane deconvolved arrays exceeded behavior length by 1 frame in 10 sessions**: Updated `convert_data.py` to trim exactly one trailing neural frame in this specific case and to keep raising for larger mismatches. Re-ran full conversion and full verify-only successfully.
- **Initial neural sanity check appeared to fail**: Confirmed this was expected `float16` quantization from the deliberate storage-size optimization, not a neuron-order or alignment bug. Updated the Step 10 neural checks to compare against the explicit float16 round-trip representation.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes
- Device used: `cuda`
- Final train loss: `1.225711`
- Final test loss: `1.141356`

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `distance_to_reward_zone` | 0.4481 | 0.4081 | Strongly above chance (`0.1429`) |
| `absolute_position` | 0.5316 | 0.5084 | Strongly above chance (`0.2000`) |
| `speed` | 0.3639 | 0.3479 | Above chance (`0.2000`) |
| `lick` | 0.6107 | 0.6059 | Slightly above chance (`0.5000`), low but stable |
| `reward_zone_location` | 0.8424 | 0.8006 | Strongly above chance (`0.3333`) |
| `reward_outcome` | 0.5815 | 0.5069 | Near chance (`0.5000`); investigated in Step 12 |

Training notes:
- Training completed all 200 epochs without errors.
- Loss decreased monotonically overall from `177.927862` at epoch 1 to ~`1.22` by epoch 200.
- The full run created `/app/train_decoder_full_out.txt` successfully.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| `distance_to_reward_zone` | 0.4081 val balanced acc | Closest paper analogue is reward-relative position decoding in Fig. 3; expectation is that position relative to reward should be one of the strongest decodable variables. This matched expectation. |
| `absolute_position` | 0.5084 | Paper emphasizes robust hippocampal spatial coding; absolute position should decode well. This matched expectation. |
| `speed` | 0.3479 | Paper/GLM analyses include movement covariates, so moderate decodability is expected. This matched expectation. |
| `lick` | 0.6059 | No directly comparable paper accuracy. Expectation is modest decodability because licking is sparse and partly sensor-limited. Result is modest but above chance. |
| `reward_zone_location` | 0.8006 | Session/trial context strongly determines reward zone, so high decodability is expected. This matched expectation. |
| `reward_outcome` | 0.5069 | No directly comparable paper accuracy. Low accuracy is plausible because omission trials are rare (~15.7%) and the user-required target is a per-trial constant rather than the paper’s more time-local `was_reward` / `was_omission` state. |

Chance-margin review:
- `distance_to_reward_zone`: `0.4081 / 0.1429 = 2.86x` chance
- `absolute_position`: `0.5084 / 0.2000 = 2.54x` chance
- `speed`: `0.3479 / 0.2000 = 1.74x` chance
- `lick`: `0.6059 / 0.5000 = 1.21x` chance
- `reward_zone_location`: `0.8006 / 0.3333 = 2.40x` chance
- `reward_outcome`: `0.5069 / 0.5000 = 1.01x` chance

Low-accuracy investigation:
- `lick` and `reward_outcome` were the only outputs below the 1.5x-chance heuristic.
- I verified the converted outputs directly against raw NWB data on multiple trials:
  - `sub-m11_ses-03`, trial 1: omission trial. Converted `reward_outcome` was constant `0` across the trial and matched the absence of any raw reward event exactly.
  - `sub-m11_ses-06`, trial 60: artifact-flagged lick trial (`42.0%` of raw lick frames >2). Converted `lick` matched the expected all-zero cleaned trial exactly; converted `reward_outcome` matched the raw rewarded label exactly.
  - `sub-m12_ses-10`, trial 73: artifact-flagged lick trial (`38.7%` of raw lick frames >2). Converted `lick` again matched the expected cleaned all-zero trial exactly.
- The processing plots created during sample conversion showed the expected temporal relationship between neural activity, position, reward-zone occupancy, and reward/lick outputs; no temporal shift was evident.
- Train/validation gaps did **not** indicate leakage or overfitting:
  - `distance_to_reward_zone`: train/val ratio `1.10`
  - `absolute_position`: `1.05`
  - `speed`: `1.05`
  - `lick`: `1.01`
  - `reward_zone_location`: `1.05`
  - `reward_outcome`: `1.15`

Conclusion:
- I did not find evidence of a conversion bug behind the weak `reward_outcome` score.
- The most likely explanation is target definition: by user instruction `reward_outcome` is a per-trial categorical label repeated across all frames, whereas the reference code and manuscript emphasize more local reward-related states (`was_reward`, `was_omission`, reward-zone entry) and reward omission is intentionally sparse/random.
- `lick` is also a difficult sparse binary target with sensor-artifact cleanup, but the converted labels matched raw data exactly on normal and artifact-handled trials.

### Issues Found and Resolved
- **Potential concern: `reward_outcome` validation accuracy near chance**: Investigated raw omission/rewarded trials and confirmed labels are correct. Left conversion unchanged because the weakness is explained by the user-required per-trial target rather than an alignment/labeling error.
- **Potential concern: `lick` validation accuracy only modestly above chance**: Investigated raw and artifact-handled trials and confirmed converted binary lick traces are correct. Left conversion unchanged.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

Notes:
- Created `/app/README.md` with dataset summary, format description, loading example, and validation snapshot.
- Created `/app/cache/README_CACHE.md` documenting cache usage and why required output files remain at `/app/`.
- No standalone investigation scripts existed to move; all scratch analyses were executed inline during the workflow.
