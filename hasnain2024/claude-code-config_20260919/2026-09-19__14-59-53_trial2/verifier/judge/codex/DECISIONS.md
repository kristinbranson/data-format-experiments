# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes 44 author-selected sessions (25 fixed-delay, 19 randomized-delay), their ALM probe numbers, and task folder. Each `data_structure_<mouse>_<date>.mat` is loaded with `pymatreader`; motion energy is loaded from the matching standalone file, with `obj.me` as fallback. Full conversion processes these session records in parallel.

ii.
```python
SESSIONS = ([(a, d, p, "fixed") for a, d, p in FIXED_DELAY]
            + [(a, d, p, "randomized") for a, d, p in RANDOMIZED_DELAY])
def load_obj(anm, date, task):
    from pymatreader import read_mat
    return read_mat(data_path(anm, date, task))["obj"]
```

iii. It says the list was transcribed from the authors' `load<ANM>_ALMVideo.m` scripts, excluding three files omitted there, and that `pymatreader` handles both MAT v7 and v7.3.

## 1-b. How are the data split into subjects?

i. The subject is the `anm` field in each hard-coded session tuple. Final assembly builds a sorted unique subject list and maps every retained session to it.

ii.
```python
subjects = sorted({r["info"]["anm"] for r in results})
subject_idx = np.array([subject_to_idx[r["info"]["anm"]] for r in results], dtype=np.int32)
```

iii. The notes report 14 mice and treat filename/load-script animal identifiers as authoritative.

## 1-c. How are the data split into sessions?

i. Every `(animal, date, probes, task)` tuple is one session and becomes one entry in each top-level session list after session-level curation.

ii.
```python
def process_session(args):
    anm, date, probes, task = args
    obj = load_obj(anm, date, task)
```

iii. The AI follows the paper's explicit 25+19 session lists and excludes files not referenced by the author loaders.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` sets the raw trial count. Per-trial behavior fields are sliced to that count; spike `trial` identifiers are converted from one-based to zero-based; each retained index becomes one neural/input/output element.

ii.
```python
n = int(vec(bp["Ntrials"])[0])
return np.flatnonzero(keep), n
for k, it in enumerate(trials):
    neural.append(np.ascontiguousarray(rates[:, k, :], dtype=np.float32))
```

iii. The notes verify one-based spike trial indices, behavior-field consistency, and preserve `trial_index` provenance.

## 1-e. How are trials filtered based on quality controls?

i. It removes early-lick trials, photostimulation trials, and every trial on which no cluster from a selected probe has any spike. Hit, miss, ignore, WC, and DR trials remain.

ii.
```python
keep = ~(early | stim)
if obj is not None:
    keep &= trials_with_ephys(obj, probes, n)
```

iii. Early and stimulated trials are excluded by the paper's conditions. The extra no-ephys rule was added after verification found all-zero neural trials caused by recordings ending before behavior.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from selected `obj.clu` clusters' `trial`, `trialtm`, and `quality` fields, selected probe IDs, and `obj.bp.ev.goCue`.

ii.
```python
tm = np.ravel(np.asarray(trialtm[i], dtype=np.float64))
tr = np.ravel(np.asarray(trialno[i], dtype=np.float64)).astype(np.int64) - 1
aligned = tm - align_times[tr]
```

iii. The AI maps this to `findClusters`, `alignSpikes`, and `getSeq` in the authors' pipeline.

## 2-b. How is the `neural` data processed?

i. Spikes are counted in 10 ms bins over −2.5 to +2.5 s, divided by 0.01 to obtain Hz, and smoothed with a 15-sample one-sided causal Gaussian FIR with reflected prefix. There is no normalization or baseline subtraction.

ii.
```python
rate = counts / DT
m = rate.reshape(-1, NT).T
m = my_smooth(m)
```

iii. The AI chose figure-script parameters (`dt=1/100`, causal `mySmooth`) over conflicting defaults/comments and says this exactly reconstructs `obj.trialdat`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It keeps only clusters on specified probes, rejects exactly four case-sensitive trimmed labels, then retains units with mean processed rate greater than 1 Hz. Sessions must have at least 10 retained units.

ii.
```python
QUALITY_REJECT = ("garbage", "gabrga", "noisy", "real?")
if q not in QUALITY_REJECT:
    keep.append((p - 1, i))
keep_unit = mean_fr > LOW_FR
```

iii. It claims exact fidelity to MATLAB `findClusters(...,'all')` case sensitivity and the Methods' >1 Hz/session ≥10-unit rules.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's trial-relative time has that trial's go-cue time subtracted before bin assignment.

ii.
```python
gocue = vec(bp["ev"][ALIGN_EVENT])[:ntrials_all]
aligned = tm - align_times[tr]
```

iii. This directly mirrors `alignSpikes.m` and satisfies the required go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The stored resolution is 10 ms (500 bins across five seconds). Raw spikes are binned directly; 400 Hz video signals are linearly interpolated onto those bin centers.

ii.
```python
TMIN, TMAX, DT = -2.5, 2.5, 1.0 / 100.0
EDGES = np.round(np.arange(TMIN, TMAX + DT / 2.0, DT), 10)
TIME = (EDGES[:-1] + DT / 2.0)
```

iii. The AI resolves contradictory repository parameters in favor of the 10 ms settings used by figure scripts.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a constructed time axis derived from the selected analysis window, bin width, and the go-cue alignment convention, not a varying raw trial field.

ii.
```python
TIME = (EDGES[:-1] + DT / 2.0).astype(np.float64)
INPUT_NAMES = ["time_from_go_cue"]
```

iii. The notes identify this with the reference `obj.time` axis.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code takes centers of consecutive 10 ms edges, yielding −2.495 through +2.495 s, converts to float32, and reuses the same `(1, 500)` array for every trial.

ii.
```python
input_trial = np.ascontiguousarray(TIME[None, :], dtype=np.float32)
inputs.append(input_trial)
```

iii. Reuse saves memory because the time ramp is identical across trials.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the centers of the exact edges used to bin go-cue-relative spikes, so corresponding columns represent the same intervals.

ii.
```python
b = bin_index(aligned)
TIME = (EDGES[:-1] + DT / 2.0)
```

iii. The AI reports explicit range and bin-edge sanity tests.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses `bp.R`, `bp.L`, `bp.hit`, `bp.miss`, and `bp.no`.

ii.
```python
R = vec(bp["R"])[:ntrials_all] > 0
L = vec(bp["L"])[:ntrials_all] > 0
hit = vec(bp["hit"])[:ntrials_all] > 0
miss = vec(bp["miss"])[:ntrials_all] > 0
no = vec(bp["no"])[:ntrials_all] > 0
```

iii. It follows the reference `getPrevChoice` definition and uses the explicit no-response field for the required third class.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Correct trials use the instructed side; misses invert it; ignores are `none` (2). The per-trial class is broadcast across time.

ii.
```python
lick[(L & hit) | (R & miss)] = 0
lick[(R & hit) | (L & miss)] = 1
lick[no] = 2
out[0] = lick[it]
```

iii. The notes say the explicit class replaces reference NaN because the task requires `none`.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context comes solely from `bp.autowater`.

ii.
```python
autowater = vec(bp["autowater"])[:ntrials_all] > 0
```

iii. The authors' tutorial calls it a proxy for WC versus DR blocks, and its block structure matches the Methods.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater is coded WC=0; all other trials are DR=1, then broadcast over time.

ii.
```python
context = np.where(autowater, 0, 1).astype(np.int8)
out[1] = context[it]
```

iii. This is a direct categorical relabeling specified by the decoder task.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It reads `bp.hit`, `bp.miss`, and `bp.no`, although assignment explicitly uses hit and no, leaving miss as the initialized class.

ii.
```python
hit = vec(bp["hit"])[:ntrials_all] > 0
miss = vec(bp["miss"])[:ntrials_all] > 0
no = vec(bp["no"])[:ntrials_all] > 0
```

iii. The AI verified these flags are mutually exclusive and exhaustive.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Default/miss is incorrect=0, hit is correct=1, and no-response is ignore=2; the class is broadcast across time.

ii.
```python
outcome = np.full(ntrials_all, 0, dtype=np.int8)
outcome[hit] = 1
outcome[no] = 2
```

iii. It replaces reference NaNs for ignored trials with the prompt-required category.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only side-camera `obj.traj[0].ts` x/y coordinates for feature `tongue`, that camera's `frameTimes`, behavior go cue, and the session video-clock shift.

ii.
```python
TONGUE_VIEW, TONGUE_FEAT = 0, "tongue"
i_tongue = feature_index(obj, TONGUE_VIEW, TONGUE_FEAT)
```

iii. The AI calls the side-view tongue the canonical marker used in the reference feature list.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. X/y are linearly interpolated to the neural grid. Speed is the Euclidean magnitude of `np.gradient` within each contiguous finite segment, with no baseline correction. NaNs remain where unavailable.

ii.
```python
x, y = interp_feature(ts, i_tongue, ft_side, align, taxis)
tongue_v[k] = feature_speed(x, y, subtract_baseline=False)
```

iii. The AI argues segment-wise derivatives avoid contaminating visibility gaps and preserve the prompt's not-visible class.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A session-wide median is calculated over finite values from kept trials. Finite values below the median are 0, values at/above it are 1, and nonfinite bins are 2.

ii.
```python
thr = float(np.median(v[valid]))
out[valid] = (v[valid] >= thr).astype(np.int8)
```

iii. Invalid bins are excluded so the large not-visible fraction does not determine the threshold.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The code subtracts session video shift plus trial go cue from side-camera frame times, then interpolates to the neural `TIME` centers.

ii.
```python
align = gocue[it] + vidshift
x, y = interp_feature(ts, i_tongue, ft_side, align, taxis)
```

iii. It mirrors `findVideoOffset`/`findPosition` and applies no extra lag.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera x/y tracks for both `top_paw` and `bottom_paw`, plus that view's frame times and clock alignment fields.

ii.
```python
PAW_VIEW, PAW_FEATS = 1, ("top_paw", "bottom_paw")
i_paws = [feature_index(obj, PAW_VIEW, f) for f in PAW_FEATS]
```

iii. The AI combines paws because either individual marker can have extensive dropout.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each paw is interpolated to the neural grid, differentiated in finite runs, baseline-corrected following the reference's apparent x-for-both-axes behavior, converted to speed, then averaged over whichever paws are finite.

ii.
```python
sp.append(feature_speed(x, y, subtract_baseline=True))
paw_v[k] = np.nanmean(np.stack(sp, axis=0), axis=0)
```

iii. Averaging the visible subset is justified as reducing marker-specific dropout while retaining bottom-view paw information.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. It uses the same per-session finite-value median rule: below=0, at/above=1, neither paw visible=2.

ii.
```python
paw_c, thr_paw = discretise(paw_v)
```

iii. The rule directly implements the specified 50th percentile and visibility category.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera times are corrected by video shift and trial go cue, and positions are interpolated at the neural bin centers before velocity calculation.

ii.
```python
ft_bot = frame_times(obj, PAW_VIEW, it)
x, y = interp_feature(ts, ip, ft_bot, align, taxis)
```

iii. The same time base is used for every modality, with `ADVANCE_MOVEMENT=0`.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It uses per-trial traces from `motionEnergy_<animal>_<date>.mat`, falling back to `obj.me`, plus side-camera frame times, video shift, and go cue.

ii.
```python
if os.path.exists(p):
    me = read_mat(p)["me"]
elif "me" in obj:
    me = obj["me"]
```

iii. The loader supports the several observed wrapper layouts and verifies trace/frame-length agreement.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The already spatially reduced motion-energy trace is linearly interpolated from frame times to neural bin centers; there is no smoothing or differentiation.

ii.
```python
me_v[k] = interp_trace(ft_side - align, tr, taxis)
```

iii. The AI says upstream computation already performed the paper's pixel/frame reduction.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Finite values are split at their per-session median with `<` coded 0 and `>=` coded 1; unavailable video bins are 2.

ii.
```python
me_c, thr_me = discretise(me_v)
```

iii. This implements the prompt's session 50th-percentile requirement rather than the paper's manual movement threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame time minus video shift minus trial go cue is interpolated onto `TIME`.

ii.
```python
align = gocue[it] + vidshift
me_v[k] = interp_trace(ft_side - align, tr, taxis)
```

iii. Motion energy has one value per side-camera frame, making that camera's timing the stated match.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. MATLAB singleton cells are normalized; missing/bad frame times, shape mismatches, absent motion traces, and out-of-coverage bins become category 2. No-ephys trials are removed. Empty probes and non-string quality labels are tolerated. No missing video values are imputed.

ii.
```python
if ft is None or a.size < 2 or not np.all(np.isfinite(a)):
    return None
out = np.full(v.shape, 2, dtype=np.int8)
```

iii. The notes say class 2 truthfully represents absent measurements, whereas interpolation/filling would fabricate behavior; verification motivated dropping all-zero neural trials.

## 11-a. What are the most time-consuming steps of the code?

i. Loading whole MATLAB objects is identified as dominant, particularly large unused waveform arrays; video interpolation/velocity and neural processing are separately timed.

ii.
```python
obj = load_obj(anm, date, task)
t_load = time.time() - t0
timing=dict(load=t_load, neural=t_neural, video=t_video, total=time.time() - t0)
```

iii. The notes report 6–8 seconds and about 1.4 GB for the largest whole-object read; parallel sessions reduce full wall time to 17 seconds.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The cluster loop remains, with vectorized binning across all spikes/trials per cluster. The per-trial video loop and inner two-paw loop could be further vectorized only awkwardly because trials have ragged frames and missingness.

ii.
```python
for ci, (p, i) in enumerate(cluster_ids):
    counts[ci] = np.bincount(flat, minlength=ntrials * NT).reshape(ntrials, NT)
for k, it in enumerate(trials):
```

iii. The AI explicitly avoided a much larger trial-by-neuron loop and batches smoothing across the session.

## 11-c. What processing does the code repeat multiple times?

i. It calls `trials_with_ephys` once for selection and again for metadata. It also interpolates and differentiates each paw separately and reloads a session when diagnostic plots are requested. Otherwise thresholds and offsets are session-level.

ii.
```python
trials, ntrials_all = select_trials(bp, obj, probes)
n_trials_no_ephys=int(np.sum(~trials_with_ephys(obj, probes, ntrials_all)))
```

iii. The notes emphasize reuse of the shared time input and session-parallel work, but do not call out the duplicate no-ephys scan.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `pymatreader` materializes the full object, including spike waveforms and other unused fields/features. It also reads `L`, `miss`, and all-cluster trial arrays for checks/definitions, though some assignments can be inferred from complementary flags; diagnostic plotting recomputes processing only when requested.

ii.
```python
return read_mat(data_path(anm, date, task))["obj"]
R = vec(bp["R"]); L = vec(bp["L"]); hit = vec(bp["hit"]); miss = vec(bp["miss"])
```

iii. The AI explicitly identifies whole-object loading, especially `clu.spkWavs`, as the main unavoidable inefficiency of its chosen loader.
