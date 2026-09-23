# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes 25 sessions (10 mice) from `/app/data/Ephys_Behavior`, loads each `data_structure_<animal>_<date>.mat` through a custom `Session` reader, and loads its separate motion-energy MAT file. It does not load the 19 sessions in `RandomizedDelay_Ephys_Behavior`.

ii.
```python
DATA_DIR = '/app/data/Ephys_Behavior'
SESSIONS = [('JEB6', '2021-04-18', [2]), ...]
for anm, date, probes in sessions:
    res = process_session(anm, date, probes)
```

iii. The trajectory says the agent chose the 25-session set because it regarded it as the “main dataset” used in most figure scripts and treated randomized-delay recordings as secondary. It used the authors' loader scripts to choose sessions and probes.

## 1-b. How are the data split into subjects?

i. The animal identifier in each hard-coded session tuple defines the subject. Subjects are added in first-encounter order, and each retained session receives the corresponding integer index.

ii.
```python
if res['subject'] not in data['subjects']:
    data['subjects'].append(res['subject'])
data['subject_idx'].append(data['subjects'].index(res['subject']))
```

iii. The agent relied on the animal IDs transcribed from the authors' loading scripts; its final trajectory reports 10 mice.

## 1-c. How are the data split into sessions?

i. Every `(animal, date, probes)` tuple and associated MAT file becomes one session-level element of `neural`, `input`, and `output`, unless it fails the minimum-unit or minimum-trial check.

ii.
```python
path = os.path.join(DATA_DIR, 'data_structure_%s_%s.mat' % (anm, date))
s = Session(path)
...
data['neural'].append(res['neural'])
```

iii. The trajectory says this mirrors the session/probe organization in `load<ANM>_ALMVideo.m`; the agent reports 25 retained sessions.

## 1-d. How are the data split into trials?

i. Trial indices come from the Bpod session table. Kept zero-based indices select behavior flags, while spike records' one-based trial IDs are mapped into rows of the retained-trial tensor. Each retained row is emitted as one trial matrix.

ii.
```python
trials = np.where(keep)[0]
trial_pos[trials + 1] = np.arange(len(trials))
neural = [np.ascontiguousarray(fr[:, k, :]) for k in range(len(trials))]
```

iii. The agent treated the Bpod trial table and cluster trial labels as authoritative trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. Trials with early licks, stimulation, or non-finite go-cue times are removed. Ignore/no-response trials remain. A session is skipped if fewer than two trials remain; unlike the reference, trials after electrophysiology recording ends are not explicitly removed.

ii.
```python
keep = (~early) & (~stim) & np.isfinite(gocue)
trials = np.where(keep)[0]
if fr.shape[0] < MIN_UNITS or len(trials) < 2:
    return None
```

iii. The agent cites paper conditions containing `~early & ~stim.enable`, retains ignore trials because they are a requested outcome class, and adds the finite-go-cue safeguard.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from selected probes' cluster `trial` and `trialtm` spike fields, cluster `quality`, per-trial `bp.ev.goCue`, and probe-location metadata for region labels.

ii.
```python
tr, ttm = s.clu_spikes(prb, ci)
aligned = ttm - gocue[tr - 1]
qual = s.qualities(prb)
```

iii. The agent identified these as the fields used by the paper's spike-alignment and sequence-building functions.

## 2-b. How is the `neural` data processed?

i. Spikes are counted in 10 ms bins, converted to Hz, and smoothed with a one-sided 15-bin Gaussian kernel intended to reproduce `mySmooth(...,15,'reflect')`. Selected probes are concatenated. No normalization or baseline subtraction is applied.

ii.
```python
np.add.at(counts[ii], (ti[ok], bi[ok]), 1.0)
rate = smooth_causal(counts / DT)
fr = np.concatenate(fr_list, axis=0)
```

iii. The trajectory explicitly chose `dt=1/100`, causal `gausswin(15)`, and reflect-style boundary handling based on figure scripts, and later checked the kernel with a delta test.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters whose lower-cased label is `garbage`, typo `gabrga`, `noisy`, or `real?` are excluded. Remaining units must have mean smoothed rate above 1 Hz. Sessions must retain at least 10 units.

ii.
```python
BAD_QUALITY = ('garbage', 'gabrga', 'noisy', 'real?')
cluix = [i for i, q in enumerate(qual) if q.lower() not in BAD_QUALITY]
use = fr.mean(axis=(1, 2)) > LOW_FR
```

iii. The agent cites `findClusters.m`, `removeLowFRClusters.m`, the Methods' >1 Hz criterion, and a Methods-based 10-unit session criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's within-trial time has that trial's go-cue time subtracted before binning, producing a −2.5 to +2.5 s go-cue-relative window.

ii.
```python
aligned = ttm - gocue[tr - 1]
bi = np.floor((aligned - TMIN) / DT).astype(int)
```

iii. The agent states this follows `alignSpikes.m` with `params.alignEvent='goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output uses 500 non-overlapping 10 ms bins from −2.5 to +2.5 s. Raw spikes are binned directly at 10 ms; video streams are interpolated to the 10 ms bin centers.

ii.
```python
DT = 0.01
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME = EDGES[:-1] + DT / 2
```

iii. The agent selected 10 ms from figure scripts and described it as the paper pipeline's `params.dt=1/100`, although the human reference uses 5 ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not derived from a varying raw field; it is the fixed bin-center axis implied by `TMIN`, `TMAX`, and `DT`. Raw go-cue times are used to align the neural/video streams to that axis.

ii.
```python
TIME = EDGES[:-1] + DT / 2
inp = [np.asarray(TIME, dtype=np.float32).reshape(1, NT)] * len(trials)
```

iii. The agent reasoned that the requested continuous input should be the common go-cue-relative time axis.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. It computes centers of uniform 10 ms bins and duplicates the same `(1, 500)` array reference for every trial.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME = EDGES[:-1] + DT / 2
```

iii. This supplies a continuous, time-varying decoder input on the common analysis grid.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. `TIME` contains the centers of the exact edges used for spike bin assignment, so input sample `j` corresponds to neural bin `j`.

ii.
```python
bi = np.floor((aligned - TMIN) / DT).astype(int)
TIME = EDGES[:-1] + DT / 2
```

iii. The shared axis was deliberately used for every stream.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived from Bpod `hit`, `miss`, instructed-right `R`, and instructed-left `L` flags.

ii.
```python
hit = s.bp_flag('hit'); miss = s.bp_flag('miss')
R = s.bp_flag('R'); L = s.bp_flag('L')
```

iii. The trajectory says this reconstruction was checked against recorded lick times and found consistent.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Default class 2 means none. Hits use the instructed side; misses use the opposite side. Codes are left 0 and right 1, and the per-trial result is repeated over time.

ii.
```python
lick_dir = np.full(len(trials), 2, dtype=np.int64)
lick_dir[hit[trials] & R[trials]] = 1
lick_dir[miss[trials] & R[trials]] = 0
```

iii. The agent inferred actual lick direction from instruction side plus correctness because direction is not directly stored as one field.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It uses the per-trial Bpod `autowater` flag.

ii.
```python
autowater = s.bp_flag('autowater')
```

iii. The agent interpreted autowater trials as the water-cued context.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater maps to WC (0), otherwise DR (1), repeated across all time bins.

ii.
```python
context = np.where(autowater[trials], 0, 1).astype(np.int64)
```

iii. This is a direct relabeling of the block/context flag.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It reads Bpod `hit`, `miss`, and `no` flags, though only `hit` and `no` are needed by the assignment code.

ii.
```python
hit = s.bp_flag('hit')
miss = s.bp_flag('miss')
no = s.bp_flag('no')
```

iii. The agent treated these mutually exclusive trial flags as the raw outcome labels.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. It defaults to incorrect (0), changes hits to correct (1), and `no` trials to ignore (2), then repeats the value over time.

ii.
```python
outcome = np.full(len(trials), 0, dtype=np.int64)
outcome[hit[trials]] = 1
outcome[no[trials]] = 2
```

iii. The agent retained ignore trials because ignore is explicitly requested as a class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only the side-camera DLC feature named `tongue`, its `frameTimes` and x/y tracking, plus go-cue and video/behavior bitcode timing. It does not use the bottom-camera `top_tongue` feature used by the reference.

ii.
```python
feats0 = s.traj_featnames(0)
tongue_ix = feats0.index('tongue')
ft, xy = s.traj_feat_trial(0, tr, tongue_ix)
```

iii. The trajectory explicitly settled on side-camera `tongue`; it recognized both camera views but did not combine them.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Side-camera x/y positions are linearly interpolated directly onto the neural time centers. A NaN-aware central/one-sided sample difference is computed, and its Euclidean magnitude becomes speed. There is no 5 ms positional smoothing, division by elapsed time, or two-view normalization/averaging.

ii.
```python
pos = interp_to_axis(ft, xy, taxis)
vel = nan_gradient(pos)
tongue_speed[k] = np.hypot(vel[:, 0], vel[:, 1])
```

iii. The agent introduced `nan_gradient` so visibility-bout edges would not be turned into extra missing samples; it considered this closer to visibility semantics.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A single session-wide median over all finite trial/time samples is used: finite values below it are 0, values at or above it are 1, and NaNs are 2 (`not_visible`).

ii.
```python
thr = np.percentile(x[vis], 50)
out[vis] = (x[vis] >= thr).astype(np.int64)
```

iii. The agent followed the requested per-session 50th-percentile threshold and maps untracked samples to the explicit missing-visibility class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The destination time in video-clock coordinates is neural `TIME + goCue + vidshift`; DLC positions are interpolated to it. The shift comes from matching video and behavior bitcodes.

ii.
```python
vidshift = s.video_offset()
taxis = TIME + gocue[tr] + vidshift
pos = interp_to_axis(ft, xy, taxis)
```

iii. The agent cites `findVideoOffset.m` and `frameTimes - vidshift - alignTime` as the source of this alignment.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera DLC `top_paw`, its frame times and x/y positions, plus go-cue and bitcode timing.

ii.
```python
feats1 = s.traj_featnames(1)
paw_ix = feats1.index('top_paw')
ft1, xy1 = s.traj_feat_trial(1, tr, paw_ix)
```

iii. The agent chose `top_paw` as the paper-used/reliably tracked paw feature.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Positions are interpolated to the neural grid, differentiated with the same NaN-aware sample difference, corrected by subtracting the median frame-to-frame x/y displacement, and converted to magnitude. It does not apply the reference's 5 ms pre-derivative smoothing or differentiate using actual time.

ii.
```python
pos1 = interp_to_axis(ft1, xy1, taxis)
vel1 = nan_gradient(pos1)
base = np.nanmedian(np.diff(pos1, axis=0), axis=0)
vel1 = vel1 - base[None, :]
```

iii. After initial validation, the agent added baseline subtraction to match `findVelocity.m` and retained NaN-aware differentiation to preserve bout edges.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The session-wide median over finite samples defines below (0) versus at/above (1); NaNs become not visible (2).

ii.
```python
paw_d = discretise(paw_speed)
```

iii. This implements the prompt's per-session 50th-percentile rule.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera positions are interpolated at video-clock timestamps corresponding to every go-cue-relative neural bin center.

ii.
```python
taxis = TIME + gocue[tr] + vidshift
pos1 = interp_to_axis(ft1, xy1, taxis)
```

iii. The agent uses the same bitcode-derived session shift and go-cue alignment as for tongue.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It loads the standalone `motionEnergy_<animal>_<date>.mat` file's `me.data`, unwrapping nested `data` structs, and associates each trace with side-camera frame times.

ii.
```python
medat = sio.loadmat(mepath)['me']['data'][0, 0]
while medat.dtype.names is not None and 'data' in medat.dtype.names:
    medat = medat['data'][0, 0]
```

iii. The agent discovered multiple wrapper layouts and added the loop to match the guard in `loadMotionEnergy.m`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The already-reduced per-frame trace is linearly interpolated onto neural bin centers, and interpolation NaNs are filled by nearest available values before categorization.

ii.
```python
vals = interp_to_axis(ft, m, taxis)
me_arr[k] = fill_nearest(vals)
```

iii. The agent says nearest filling follows `loadMotionEnergy.m`; no spatial energy calculation is repeated.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The per-session median across all finite retained values gives class 0 below and class 1 at/above. Remaining NaNs are class 2 (`no_video`).

ii.
```python
me_d = discretise(me_arr)
```

iii. This follows the requested session-level 50th-percentile split and explicit no-video class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Each motion-energy sample is paired with side-camera frame time, then interpolated at `TIME + goCue + vidshift`. Trials lacking valid video or with mismatched energy/frame lengths remain missing.

ii.
```python
taxis = TIME + gocue[tr] + vidshift
vals = interp_to_axis(ft, m, taxis)
```

iii. The agent regarded motion energy as synchronized to side-camera frames and used the same bitcode/go-cue correction as DLC.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing/non-finite go cues cause trial removal. Bad video trials, absent frame data, mismatched motion-energy lengths, and unavailable tracking remain NaN until mapped to category 2. Motion-energy interpolation gaps are nearest-filled. Sessions lacking enough units/trials are skipped. No explicit electrophysiology-end trial cutoff is implemented.

ii.
```python
if ndrop is not None and not np.isfinite(ndrop[tr]): continue
if ft is None or xy is None or not np.isfinite(ft).any(): continue
if ft is None or len(ft) != len(m): continue
```

iii. The agent aimed to retain usable neural/behavior trials while representing missing video with the required third class; it intentionally improved NaN-edge differentiation.

## 11-a. What are the most time-consuming steps of the code?

i. The agent did not explicitly profile the final converter. Its trajectory indicates file loading, per-unit spike accumulation/smoothing, per-trial interpolation, full conversion, and decoder training consumed most wall time.

ii.
```python
for ii, ci in enumerate(cluix):
    ...
    np.add.at(counts[ii], (ti[ok], bi[ok]), 1.0)
for k, tr in enumerate(trials):
    pos = interp_to_axis(ft, xy, taxis)
```

iii. The trajectory repeatedly waited on full-session conversion and training, but provides no benchmark that isolates conversion stages.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Unit-by-unit spike binning, trial-by-trial video/motion interpolation, trial output assembly, and repeated region-index lookup are loop based. Spike binning could use a 2-D histogram per cluster or more batched grouping; output assembly can be array-built before splitting.

ii.
```python
for ii, ci in enumerate(cluix): ... np.add.at(...)
for k, tr in enumerate(trials): ...
for k in range(len(trials)):
    o = np.empty((6, NT), dtype=np.int64)
```

iii. The agent prioritized fidelity and variable-length source records over explicit vectorization and did not document an efficiency analysis.

## 11-c. What processing does the code repeat multiple times?

i. Per trial it creates `taxis` twice (kinematics and motion energy), re-reads the side-camera tongue trajectory/frame times for motion energy, constructs an interpolation object for every stream/trial, and repeatedly fills constant trial outputs across time.

ii.
```python
ft, xy = s.traj_feat_trial(0, tr, tongue_ix)
...
ft, _ = s.traj_feat_trial(0, tr, tongue_ix)
taxis = TIME + gocue[tr] + vidshift
```

iii. The trajectory does not justify these repetitions; they are consequences of separate processing blocks.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads `miss` for outcome even though outcome assignment uses only `hit` and `no`, calculates/stores verbose session metadata not needed by the decoder, and processes tjM1 units despite describing the target as ALM-related. Most importantly, substantial source fields may be loaded by the helper even if unused.

ii.
```python
miss = s.bp_flag('miss')
info = dict(animal=anm, date=date, probes=list(probes), ...)
session_info.append(res['info'])
```

iii. The agent retained rich metadata for auditability and included probes exactly as selected in the author loaders; it did not identify discarded work in its final account.
