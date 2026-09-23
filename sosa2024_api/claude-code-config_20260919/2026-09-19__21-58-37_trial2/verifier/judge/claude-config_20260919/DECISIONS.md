# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every `*.nwb` file under `/app/data/sub-*/` is globbed (152 files), sorted by subject then
experiment day, and each one is converted in a separate worker process (`ProcessPoolExecutor`,
`spawn` context, 12 workers). Each file is opened once with `pynwb.NWBHDF5IO` and everything needed
(behaviour time series, raw fluorescence, neuropil, ROI segmentation table, subject id, session id,
scene name) is read in a single pass in `load_session()`; the file is then closed. No `h5py` use.
There is no separate "survey" pass — one read per file.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
# order sessions by subject then experiment day
files.sort(key=lambda p: (p.split('/')[-2], int(p.split('ses-')[1][:2])))
...
ctx = multiprocessing.get_context('spawn')
with ProcessPoolExecutor(max_workers=nworkers, mp_context=ctx) as ex:
    for k, (neural, inp, out, planes, info) in enumerate(ex.map(_worker, jobs)):
```
```python
def load_session(path):
    """Read everything needed for one session out of the NWB file with pynwb."""
    io = NWBHDF5IO(path, 'r', load_namespaces=True)
    nwb = io.read()
    subject = nwb.subject.subject_id
    exp_day = int(nwb.session_id)
    scene = nwb.identifier.rstrip('/').split('/')[-1]
    ...
    beh = nwb.processing['behavior']['BehavioralTimeSeries']
    ophys = nwb.processing['ophys']
    ...
    io.close()
```

iii. From CONVERSION_NOTES Step 2: the data are one NWB file per mouse-day,
`/app/data/sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb`, 11 subject directories and 152
files totalling 87 GB. The AI cross-checked this against the paper ("n = 11 mice", 14 days/mouse,
m11 imaged from day 3) and confirmed 152 = 12 + 14×10. All 152 sessions are loaded; nothing is
sub-sampled except under `--sample` (2 named sessions used only for testing).

## 1-b. How are the data split into subjects?

i. Subject identity is taken from inside each NWB file (`nwb.subject.subject_id`, e.g. `m11`), not
from the directory name. The `subjects` list is built in order of first appearance while the
per-session results stream back from the worker pool, and `subject_idx` records, for each session,
the index into that list. 11 subjects result (m11, m12, m13, m14, m15, m17, m18, m19, m3, m4, m7).

ii.
```python
subject = nwb.subject.subject_id
...
for k, (neural, inp, out, planes, info) in enumerate(ex.map(_worker, jobs)):
    sub = info['subject']
    if sub not in subjects:
        subjects.append(sub)
    subject_idx.append(subjects.index(sub))
...
data['subjects'] = subjects
data['subject_idx'] = np.array(subject_idx, dtype=np.int64)
```

iii. CONVERSION_NOTES Step 2/Step 9: "11 subject dirs ... 11 ✅". The AI also uses the animal name
embedded in `nwb.identifier` (`/data/InVivoDA/GCAMP11/...`) to map the released `m<N>` id onto the
paper's internal `GCAMP<N>` id, which is what the per-animal `teleport_metadata` table is keyed on.
`ex.map` preserves input order, so the subject grouping is deterministic.

## 1-c. How are the data split into sessions?

i. One NWB file = one session = one mouse-day. `nwb.session_id` is parsed as the integer experiment
day and stored as `exp_day`; a human-readable `session_id` (`m11_day03`) plus the scene name, date
and all per-session counts are written to `metadata['session_info']`. No cross-mouse or cross-day
neuron alignment is attempted (each session keeps its own neuron set).

ii.
```python
exp_day = int(nwb.session_id)
...
session_id=f"{sess['subject']}_day{sess['exp_day']:02d}",
```

iii. CONVERSION_NOTES Step 2: "`/app/data/sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb`, one
file per mouse-day (`ses-NN` == experiment day, 1–14)". The AI verified the day numbering against
the paper ("imaged starting from day 1 for all mice except m11, for whom imaging started on day 3")
— m11 indeed has files from `ses-03`, 12 sessions, while every other mouse has 14.

## 1-d. How are the data split into trials?

i. A trial is the lap `[trial_start_ind, teleport_ind)`: the frame indices where the behaviour
series `trial_start` is non-zero give the starts, and the frames where `teleport` is non-zero give
the ends. The teleport frame itself is **excluded** from the trial. Two assertions guard the
segmentation: the number of starts must equal the number of teleports, and every teleport must come
after its trial start.

ii.
```python
tstart_inds = np.where(beh['trial_start'] > 0)[0]
teleport_inds = np.where(beh['teleport'] > 0)[0]
assert len(tstart_inds) == len(teleport_inds), \
    f"{path}: {len(tstart_inds)} trial starts vs {len(teleport_inds)} teleports"
assert np.all(teleport_inds > tstart_inds), f"{path}: teleport before trial start"
n_trials_raw = len(tstart_inds)
...
s, e = int(tstart_inds[i]), int(teleport_inds[i])
pos = beh['pos'][s:e]
```

iii. CONVERSION_NOTES Step 2/Step 5 Decision 3: this is the window the reference code uses
everywhere (`sess.trial_start_inds[i] : sess.teleport_inds[i]` in `behavior.get_trial_types`,
`lick_pos_std`, etc.). The teleport sample is dropped because "position at the `teleport` frame is a
meaningless interpolation between end-of-track and the teleport zone (e.g. 350 cm when the previous
sample is 446 cm)". The AI verified over all 152 files that `#trial_start == #teleport` in every
session, that the per-frame counts never exceed 1 (so `np.where(>0)` is exact), and that no trial
overlaps the previous teleport. 12,216 raw trials result.

## 1-e. How are trials filtered based on quality controls?

i. One quality filter: trials with a **lick-sensor error** — more than 30% of the frames in the
trial carrying a cumulative lick count > 2 — are dropped entirely (81 of 12,216 trials). A second,
defensive filter drops any trial containing a non-finite value in neural, position, speed or lick
(it never fires: `n_dropped_nan = 0` in all 152 sessions). No minimum trial-length filter is
applied.

ii.
```python
LICK_ERROR_THRESH = 0.3     # paper: ">30% of samples with cumulative lick count >2"
...
    L = beh['lick'][s:e]
    lick_error[i] = np.mean(L > 2) > LICK_ERROR_THRESH
...
for i in range(n_trials_raw):
    if lick_error[i]:
        n_dropped_lick += 1
        continue
    ...
    if (not np.all(np.isfinite(neural)) or not np.all(np.isfinite(pos))
            or not np.all(np.isfinite(speed)) or not np.all(np.isfinite(lick))):
        n_dropped_nan += 1
        continue
```

iii. This is the paper's own trial-curation rule (`behavior.correct_lick_sensor_error`). The AI
noted that the function's default `correction_thr=0.5` and the `lick_pos_std` call site's 0.35
disagree with the Methods text, tested all three (81/69/44 trials) and used **0.30**, which
reproduces the paper's stated count exactly: "n = 81 out of 12,376 trials". The paper NaNs the licks
on those trials; the AI drops the whole trial instead because "`lick` is a decoder output and NaN is
not a permitted value" (Step 5, Decision 8). The AI also checked that the shortest trial in the
whole dataset is 96 frames (6.2 s), so no minimum-length filter was needed
(Step 10, Check 5). 12,135 trials survive.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The raw suite2p traces: `processing['ophys']['Fluorescence']['planeK']` (raw F) and
`processing['ophys']['Neuropil']['planeK']` (Fneu), restricted to `iscell` ROIs via the
`ImageSegmentation/PlaneSegmentation` table. The NWB `Deconvolved` series is deliberately **not**
used.

ii.
```python
ophys = nwb.processing['ophys']
seg = ophys['ImageSegmentation']['PlaneSegmentation']
iscell = np.asarray(seg['iscell'].data)[:, 0] > 0
plane_idx = np.asarray(seg['planeIdx'].data).astype(int)
for key in sorted(ophys['Fluorescence'].roi_response_series.keys()):
    p = int(key.replace('plane', ''))
    mask = iscell[plane_idx == p]
    rrs = ophys['Fluorescence'][key]
    rate = rrs.rate
    F_list.append(np.asarray(rrs.data[:nframes, :])[:, mask].T.astype(np.float32))
    Fneu_list.append(np.asarray(ophys['Neuropil'][key].data[:nframes, :])[:, mask].T
                     .astype(np.float32))
```

iii. CONVERSION_NOTES Step 1/Step 4: "The NWB `Deconvolved` series is suite2p's own `spks`
(non-zero before the first trial start, i.e. computed over the whole session), NOT the paper's
per-trial dFF-based `events` — so ΔF/F + OASIS must be recomputed here, exactly as the paper does."
The AI detected this by observing that the stored `Deconvolved` trace is non-zero outside trials,
whereas the reference `dff(..., deconvolve=True)` output is NaN outside trials.

## 2-b. How is the `neural` data processed?

i. A line-for-line port of the paper's `preprocessing.dff`: per-window neuropil subtraction with
`neu_coef = 0.7`, the window's neuropil mean added back, a *maximin* baseline (Gaussian σ = 15
samples along time, then a 300-sample `minimum_filter1d`, then a 300-sample `maximum_filter1d` ≈ the
Methods' 20 s window), `dF/F = (F − baseline)/|baseline|`, and a 2-sample (σ ≈ 0.129 s) Gaussian
smoothing — all NaN-safe (`nansmooth`, a local port of `TwoPUtils.utilities.nansmooth`). The windows
are the individual trials, except on the per-animal/per-day sessions listed in the reference's
`teleport_metadata.teleport_sessions`, where the laser was not blanked and the window is extended
back to one sample after the previous teleport (`keep_teleports`). Planes are pooled afterwards.

**The processing then stops at dF/F**: the deconvolved "events" are computed only on request
(`--neural-signal events`), and the delivered `/app/converted_data.pkl` contains **dF/F**, not the
paper's OASIS-deconvolved activity rate.

ii.
```python
    f_ -= NEU_COEF * fneu_
    flow = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in windows:
        f_[:, s:e] += NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
        x = nansmooth(f_[:, s:e], [0, BASELINE_SMOOTH_SIG])
        x = ndimage.minimum_filter1d(x, MAXIMIN_WIN, axis=-1)
        flow[:, s:e] = ndimage.maximum_filter1d(x, MAXIMIN_WIN, axis=-1)
    dff[:, nanmask] = ((f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask]))
    for s, e in windows:
        dff[:, s:e] = nansmooth(dff[:, s:e], DFF_SMOOTH_SIG, axis=1)
        if deconvolve:
            events[:, s:e] = dcnv.oasis(np.ascontiguousarray(dff[:, s:e]), OASIS_BATCH, TAU, fs)
```
```python
keep_teleports = sess['exp_day'] in TELEPORT_SESSIONS.get(sess['subject'], [])
if keep_teleports:
    windows = [(int(tstart_inds[0]), int(teleport_inds[0]))]
    for i in range(1, n_trials_raw):
        windows.append((int(min(teleport_inds[i - 1] + 1, tstart_inds[i])),
                        int(teleport_inds[i])))
else:
    windows = [(int(s), int(e)) for s, e in zip(tstart_inds, teleport_inds)]
...
neural_full = (events if neural_signal == 'events' else dff)[keep_cells]
```

iii. Constants are all sourced: `neu_coef=0.7` / `baseline_method='maximin'` from
`utilities.default_dff_method`, `tau=0.7` from the suite2p ops, the 300-sample window from
`preprocessing.dff` and the Methods' "20 s sliding window", σ=2 from "a two-sample (~0.129 s) s.d.
Gaussian kernel". `keep_teleports` is justified as "a baseline taken over blanked (≈0) fluorescence
would be meaningless" (Step 5, Decision 4). The choice of dF/F over events is Step 5, Decision 2:
"(a) the paper treats it as the signal 'closest to the raw data' and uses it for spatial-peak, field
and sequence analyses; (b) OASIS deconvolution is explicitly *not* interpreted as a spike rate by
the authors ...; and (c) it decodes better here — validation balanced accuracy on the 2-session
sample was higher for every one of the six outputs (e.g. position 0.625 vs 0.559, speed 0.553 vs
0.415, reward outcome 0.633 vs 0.568)." The trajectory records the same reasoning at step 110. The
AI lists this as a deliberate deviation in its Step 10 Check 3 comparison table.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both the paper's. (1) suite2p manual curation: only ROIs with `iscell[:,0] == 1`
are read at all. (2) Putative interneurons — cells whose dF/F has a Pearson r > 0.5 with running
speed over all valid (in-window) samples — are dropped. 138,678 `iscell` ROIs → 138,298 neurons
(380 interneurons, 0.32 ± 0.60% per session).

ii.
```python
iscell = np.asarray(seg['iscell'].data)[:, 0] > 0
...
def speed_correlation(dff, speed, nanmask):
    D = dff[:, nanmask].astype(np.float64)
    S = np.asarray(speed, dtype=np.float64)[nanmask]
    Dm = D - D.mean(axis=1, keepdims=True)
    Sm = S - S.mean()
    denom = np.sqrt((Dm ** 2).sum(axis=1)) * np.sqrt((Sm ** 2).sum())
    return (Dm @ Sm) / denom
...
r_speed = speed_correlation(dff, beh['speed'], nanmask)
is_int = np.nan_to_num(r_speed, nan=0.0) > INT_R_THRESH   # INT_R_THRESH = 0.5
keep_cells = ~is_int
```

iii. `iscell` is the suite2p manual curation described in the Methods; the r > 0.5 speed-correlation
criterion is `spatial.is_putative_interneuron` with `dayData.int_thresh = 0.5`, and the Methods
quote "a Pearson correlation of >0.5 ... excluding 0.42 ± 0.85% of cells". The AI's 0.32 ± 0.60%
is reported as a match, and its per-session neuron range (154–2320) is compared with the paper's
155–2172 (minimum exact; 3 m18 sessions exceed the maximum, documented in Step 4 as ROI re-curation
between the paper and the DANDI deposit). The per-cell `np.corrcoef` loop of the reference was
vectorised into one matrix product.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Nothing beyond slicing: the alignment event is trial start, and the trial slice already begins at
`trial_start_ind`. The behaviour series in the NWB file are already one sample per imaging frame
(the authors' `vr_align_to_2P` output), so the neural and behavioural streams share indices exactly.
`metadata['temporal_alignment_event']` is set to the trial start, `off_start = 0.0`,
`off_end = None` (variable trial length).

ii.
```python
s, e = int(tstart_inds[i]), int(teleport_inds[i])
pos = beh['pos'][s:e]
...
t = beh['time'][s:e] - beh['time'][s]
neural = neural_full[:, s:e]
```
```python
'temporal_alignment_event': (
    'start of trial (VR trial_start: the animal enters the linear track at position 0 cm)'),
'off_start': 0.0,
'off_end': None,   # trials run to the teleport; duration varies by trial
```

iii. Step 5, Decision 1: "The VR behaviour in the NWB file has already been interpolated onto
imaging frame times by the authors' `vr_align_to_2P`, so neural and behavioural streams are
sample-for-sample aligned with no further resampling — the safest possible alignment." The AI
checked visually (`processing_*.png`, panel 8: `time_from_trial_start` is a clean saw-tooth
resetting exactly at trial boundaries, position staircase resets with it) and numerically (Step 10
Check 2: neural/input/output re-derived from raw NWB by independent code, `np.allclose` pass).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One bin = one imaging frame = 64.484 ms (15.5078125 Hz per plane), identical for every session.
**No rebinning or resampling of any kind is applied.** For the two-plane animals (m17, m18) the
stored `rate` is the scanner rate (31.015625 Hz) and the per-plane rate `rate / n_planes` is used —
both for the time base and for the OASIS kernel.

ii.
```python
fs = sess['rate'] / sess['n_planes']          # per-plane imaging rate (Hz)
...
'time_bin_size': 1000.0 / 15.5078125,   # ms per imaging frame (64.484 ms)
'sampling_rate_hz': 15.5078125,
```

iii. Step 5, Decision 1: "Every session is sampled at 15.5078125 Hz per plane, so the bin size is
identical everywhere, as the target format requires. No additional binning was applied: it would
only blur the licking and speed outputs." Cross-checked against the Methods ("each frame is
~64.5 ms"; "~15.5 Hz") and against the data (behaviour timestamp dt = 0.0644836 s in every session;
Step 9 consistency table marks it ✅).

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The imaging-frame timestamps attached to the behaviour series — specifically
`behavior/position.timestamps`, which the AI verified are the common frame-time base for all the
behaviour series.

ii.
```python
frame_times = np.asarray(beh['position'].timestamps[:], dtype=np.float64)
nframes = len(frame_times)
behavior = dict(time=frame_times, ...)
```

iii. CONVERSION_NOTES Step 2: the `BehavioralTimeSeries` are "one sample per imaging frame
(`timestamps` = frame times, dt = 0.0644836 s = 1/15.5078125 Hz)", i.e. the authors' VR-to-2P
alignment output, so any of the series' timestamps gives the same vector.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial's first frame time is subtracted, giving seconds since trial start; the result is
stored as `input[0]` (float32) at every timepoint of the trial. Range over the full dataset:
[0.0, 216.5] s.

ii.
```python
t = beh['time'][s:e] - beh['time'][s]
...
inp = np.stack([
    t,
    np.full(len(pos), float(morph[i])),
    np.full(len(pos), float(trialnum[i])),
    np.full(len(pos), float(prev_outcome[i])),
], axis=0).astype(np.float32)
```

iii. Directly required by the Decoder Task spec ("Time from start of trial in seconds (continuous,
time-varying)"); no reference-code counterpart. The AI cross-checked the resulting range against the
trial durations it measured from the raw data (6.2–216.6 s, Step 2) — Step 9 table marks it ✅.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the same `[s:e]` frame index slice produces the neural matrix and the time
vector, and the behaviour stream is already on the imaging-frame grid. The only length adjustment is
that the ophys arrays are truncated to the behaviour length (10 sessions have exactly one extra
imaging frame — the authors' documented "one frame correction").

ii.
```python
F_list.append(np.asarray(rrs.data[:nframes, :])[:, mask].T.astype(np.float32))
...
t = beh['time'][s:e] - beh['time'][s]
neural = neural_full[:, s:e]
```

iii. Step 10 Check 5: "ophys longer than behaviour — 10 sessions have exactly one extra imaging
frame → ophys truncated to the behaviour length (the reference's 'one frame correction')". The AI
verified the resulting alignment numerically against an independent re-derivation from the NWB
(Check 2, `np.allclose` on all four inputs × all timepoints of 5 random trials in 9 sessions,
including two of the one-extra-frame sessions) and visually in `processing_*.png`.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behaviour time series (the reference's `morph`: 0 = ENV1, 1 = ENV2).

ii.
```python
behavior = dict(..., morph=get('environment'), ...)
```

iii. CONVERSION_NOTES Step 2: "`environment` (0 = ENV1, 1 = ENV2; −1 before sync)"; the reference's
`behavior.get_trial_types` reads exactly this variable as `morph`, and `env_morph_dict` in the
reference maps `Env1→0`, `Env2→1`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. One value per trial, taken as the unique value of `environment` over the trial window, rounded to
an integer, and broadcast over all timepoints of the trial as `input[1]`.

ii.
```python
    m = np.unique(beh['morph'][s:e])
    morph[i] = int(np.round(m[0]))
...
    np.full(len(pos), float(morph[i])),
```

iii. Follows `behavior.get_trial_types`, which does `np.unique(sess.vr_data['morph'][firstI:lastI])`
per trial. The AI listed "`morph` and `trial number` constant within every trial" as a planned
sanity check and marked it verified; the delivered range is [0, 1] (no −1 pre-sync values leak into
a trial).

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The NWB `trial number` behaviour time series (0-indexed within the session), read once per trial
over the trial window — *not* a loop counter. Because it is the raw index, trials dropped for
lick-sensor error leave a gap in the numbering rather than shifting the remaining trials.

ii.
```python
behavior = dict(..., trialnum=get('trial number'), ...)
...
    tn = np.unique(beh['trialnum'][s:e])
    trialnum[i] = int(tn[0])
```

iii. CONVERSION_NOTES Step 5 variable-mapping table: "`behavior/trial number` → `input[2]`,
constant per trial, 0-indexed". The AI listed "`morph` and `trial number` constant within every
trial" as a sanity check. Delivered range [0, 99], consistent with the 41–100 trials/session it
measured from the data.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond taking the per-trial value and broadcasting it over the trial's timepoints as a
float32 row of the input matrix.

ii.
```python
    np.full(len(pos), float(trialnum[i])),
```

iii. The Decoder Task spec asks for "Trial number (continuous, per trial)"; the format spec requires
per-trial inputs to be broadcast to `(d_input, n_timepoints)`, which the AI does for all three
per-trial inputs.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the per-trial rewarded flag, which is the reference's `get_trial_types` definition: a trial
counts as rewarded iff **both** a `Reward` event fell inside it **and** the `reward_zone` flag was
set inside it. The sparse `Reward` series (one timestamp per delivered reward) is first mapped onto
the imaging-frame grid with `searchsorted`.

ii.
```python
reward_times = np.asarray(beh['Reward'].timestamps[:], dtype=np.float64)
reward_frames = np.searchsorted(frame_times, reward_times)
reward_frames = np.clip(reward_frames, 0, nframes - 1)
reward = np.zeros(nframes)
np.add.at(reward, reward_frames, 1.0)
behavior['reward'] = reward
...
    isreward[i] = int(np.any(beh['reward'][s:e] > 0)
                      and np.any(beh['rzone'][s:e] > 0))
```

iii. CONVERSION_NOTES Step 1: `get_trial_types` is "Per trial (`trial_start_inds[i]:teleport_inds[i]`):
`isreward = any(reward>0) AND any(rzone>0)`". The AI checked the AND empirically over all sessions
(Step 10, Check 2): "10,342 trials with both, **0 with reward but no zone flag**, 52 with a zone
flag but no reward (omission trials on which the VR still registered zone entry — precisely why the
reference requires the AND)". Resulting omission rate 15.3% of trials vs the paper's "~15%".

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The rewarded flag of trial *i−1* (in raw trial indexing, so before any lick-error dropping) is
assigned to trial *i* and broadcast over its timepoints; the first trial of every session gets 0.

ii.
```python
# previous trial outcome (0 = omitted, 1 = rewarded); undefined (=0) for the
# first trial of a session
prev_outcome = np.concatenate([[0], isreward[:-1]]).astype(np.int64)
...
    np.full(len(pos), float(prev_outcome[i])),
```

iii. Step 5, Decision 7: "`previous_trial_outcome` = 0 for the first trial of each session (no
preceding imaged trial). This affects 152/12,135 = 1.25% of trials; dropping those trials instead
would discard real neural data for no benefit." Matches the spec's "Previous trial outcome (binary,
omitted = 0, rewarded = 1, per trial)".

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` behaviour time series and the active reward zone for that trial. The reward
zone is obtained from the **scene name** stored in `nwb.identifier` (e.g.
`/data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A`), using a port of the reference's
`behavior.get_reward_zones`: zone A = 80–130 cm, B = 200–250 cm, C = 320–370 cm, with the zone
switching after trial 30 on `X_to_Y` scenes. The `reward_zone` behaviour flag is used only for
validation, not for labelling.

ii.
```python
REWARD_ZONE_CM = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
CHANGE_TRIAL = 30           # reward zone switches after 30 trials on switch days

def scene_reward_zones(scene, n_trials):
    if '_to_' in scene:
        before, after = scene.split('_to_')
        first = before[-1]
        second = after[-1]
        assert first in REWARD_ZONE_CM and second in REWARD_ZONE_CM, scene
        labels = np.array([first] * min(CHANGE_TRIAL, n_trials) +
                          [second] * max(0, n_trials - CHANGE_TRIAL))
    else:
        lab = scene[-1]
        labels = np.array([lab] * n_trials)
    coords = np.array([_zone_of(l) for l in labels], dtype=np.float64)
    return coords, labels
...
scene = nwb.identifier.rstrip('/').split('/')[-1]
rz_coords, rz_labels = scene_reward_zones(sess['scene'], n_trials_raw)
```

iii. Step 4/Step 5 Decision 6: "Reward-zone identity from the scene name, as the reference does,
rather than from the `reward_zone` flag — the flag only fires on rewarded trials (~85%), so it
cannot label omission trials." The AI resolved the reference `reward_zone_dict`'s two competing
key sets (`'A':[175,225]` vs `'X':[80,130]`) by following `get_reward_zones`'s `map_labels`
(A→X, B→Y, C→Z), which agrees with the paper's "zone A, 80–130 cm; zone B, 200–250 cm; zone C,
320–370 cm". Validated on the data: "over all 10,394 trials with a reward-zone flag, the position at
the first flagged frame is within 0–8.5 cm of the scene-derived zone start (mean +0.9 cm, the
expected 1-frame lag at ~44 cm/s). Zero mismatches."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance to the nearest point of the zone: `pos − zone_start` when before the zone
(negative), `pos − zone_end` when past it (positive), and exactly 0 while inside the zone. Then
discretised into the 7 specified bins.

ii.
```python
rz_start, rz_end = rz_coords[i]
d = np.zeros_like(pos)
d[pos < rz_start] = pos[pos < rz_start] - rz_start
d[pos > rz_end] = pos[pos > rz_end] - rz_end
```

iii. Matches the Decoder Task wording "Distance to any location in the reward zone", i.e. distance
to the nearest point of the 50 cm zone, which is why it is 0 throughout the zone. Verified visually
in panel 7 of `processing_*.png`: "flat at 0 exactly while position is inside the green band".

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes by explicit boolean masks: `< −50 → 0`, `[−50, −10) → 1`, `[−10, 0) → 2`,
`0 → 3`, `(0, 10] → 4`, `(10, 50] → 5`, `> 50 → 6`.

ii.
```python
def discretize_reward_distance(d):
    out = np.empty(d.shape, dtype=np.int64)
    out[:] = 3                                  # d == 0: inside the reward zone
    out[(d >= -10) & (d < 0)] = 2
    out[(d >= -50) & (d < -10)] = 1
    out[d < -50] = 0
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    return out
```

iii. The bin edges are copied from the Decoder Task spec, with class 3 reserved for exactly 0
("in reward zone (0 cm)" in `output_values`). Panel 7 of the processing plots overlays the bin
edges on the continuous trace to show the transitions line up. Resulting distribution
0.251 / 0.102 / 0.073 / 0.238 / 0.021 / 0.072 / 0.243.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same `pos = beh['pos'][s:e]` slice used for the neural slice
`neural_full[:, s:e]`, so it is aligned by construction; no interpolation or shifting.

ii.
```python
pos = beh['pos'][s:e]
neural = neural_full[:, s:e]
...
out = np.stack([discretize_reward_distance(d), ...], axis=0)
```

iii. Same justification as 2-d/3-c: the NWB behaviour series are already on the imaging-frame grid.
Verified by the independent re-derivation (`np.array_equal` on all 6 outputs × all timepoints of 5
random trials in 9 sessions) and by the processing plots.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behaviour time series (cm along the 450 cm virtual track).

ii.
```python
behavior = dict(..., pos=get('position'), ...)
...
pos = beh['pos'][s:e]
```

iii. CONVERSION_NOTES Step 2 identifies `position` as "cm; −500 before the VR/2P TTL sync"; only
in-trial samples are used, whose range the AI measured as −2.74 … 451.83 cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond slicing and discretisation; the raw cm values are used directly.

ii.
```python
pos = beh['pos'][s:e]
...
    discretize_position(pos),
```

iii. Step 5 mapping table: "`behavior/position` → `output[1]`, 5 equal 90 cm bins over the 450 cm
track". No smoothing or unwrapping is applied because each trial is a single forward lap.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Integer-divided by 90 cm (= 450/5) and clipped to [0, 4], i.e. bins `<90 / 90–180 / 180–270 /
270–360 / >360`; the few samples slightly outside [0, 450] fall into the end bins.

ii.
```python
TRACK_LENGTH = 450.0

def discretize_position(pos):
    """Absolute track position (cm) -> 5 equal bins spanning the 450 cm track."""
    return np.clip((pos / (TRACK_LENGTH / 5)).astype(np.int64), 0, 4)
```

iii. The Decoder Task spec asks for "5 equal-sized bins spanning the 450 cm track". Step 10 Check 5:
"position slightly out of [0, 450] — within-trial range is −2.74 … 451.83 cm → binning clips into
bins 0 and 4". Panel 6 of the processing plots shows the staircase tracking the position ramp.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[s:e]` slice as the neural data; no additional alignment.

ii.
```python
pos = beh['pos'][s:e]
neural = neural_full[:, s:e]
```

iii. As for 7-d: the behaviour is already sampled at imaging-frame times; verified numerically and
visually.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behaviour time series (a per-frame cumulative lick count, not a binary).

ii.
```python
behavior = dict(..., lick=get('lick'), ...)
...
lick = beh['lick'][s:e]
```

iii. CONVERSION_NOTES Step 1/Step 2: `vr_align_to_2P` produces `lick` as "counts per imaging frame";
the reference's `behavior.lickrate` binarises with `licks[licks>0] = 1`.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarised: any count > 0 in the frame → 1, otherwise 0. (Trials where the sensor was stuck —
>30% of frames with count > 2 — have already been dropped, see 1-e.)

ii.
```python
    (lick > 0).astype(np.int64),
```

iii. Matches the reference's `lickrate` binarisation and the Decoder Task spec ("Lick, time-varying.
0 = no, 1 = yes"). Resulting distribution 0.777 / 0.223.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[s:e]` slice; no shifting.

ii.
```python
lick = beh['lick'][s:e]
neural = neural_full[:, s:e]
```

iii. As above; licks were also checked visually against position and reward in panel 5 of the
processing plots ("licks cluster just before and inside [the reward zone]").

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. From the scene name in `nwb.identifier` plus the trial index, exactly as in 7-a — labels A/B/C
mapped to 0/1/2, switching after trial 30 on switch sessions. The `reward_zone` behaviour flag is
used only as an independent validation.

ii.
```python
rz_coords, rz_labels = scene_reward_zones(sess['scene'], n_trials_raw)
...
    np.full(len(pos), RZ_LABELS.index(rz_labels[i]), dtype=np.int64),
```

iii. See 7-a: this is the reference's `behavior.get_reward_zones`. The AI's validation over all
10,394 flagged trials (max error 8.5 cm, zero mismatches) and the near-uniform delivered
distribution (A 0.332 / B 0.336 / C 0.333, matching the counterbalanced design) are its evidence.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The per-trial label is converted to an index into `['A','B','C']` and broadcast over the trial's
timepoints (the target format allows per-trial outputs, but the AI makes every output time-varying).

ii.
```python
RZ_LABELS = ['A', 'B', 'C']
...
    np.full(len(pos), RZ_LABELS.index(rz_labels[i]), dtype=np.int64),
```

iii. Step 5, Decision 5: "All outputs time-varying. `reward_zone_location` and `reward_outcome` are
constant within a trial but are broadcast over time, both because the format requires a single
`doutput` for every trial and because the task asks for time-varying outputs wherever possible."

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The same `isreward` flag as the previous-trial input (6-a): the sparse `Reward` timestamps mapped
to frames, AND-ed with the `reward_zone` flag over the trial window.

ii.
```python
    isreward[i] = int(np.any(beh['reward'][s:e] > 0)
                      and np.any(beh['rzone'][s:e] > 0))
```

iii. The reference's `behavior.get_trial_types` definition; see 6-a for the AI's empirical check of
the AND (0 trials with reward but no zone flag, 52 with a zone flag but no reward).

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The per-trial binary (0 = omitted, 1 = rewarded) is broadcast over all timepoints of the trial as
`output[5]`.

ii.
```python
    np.full(len(pos), isreward[i], dtype=np.int64),
```

iii. Matches the spec ("Reward outcome, per-trial. 0 = no, 1 = yes"), broadcast for the reason given
in 10-b. The resulting omission rate (15.3% of trials, 15.8% of samples) is checked against the
paper's "~15% of trials". The AI also diagnosed the modest decoding accuracy for this output
(0.602) by splitting balanced accuracy by trial phase, showing it is at chance before the reward
zone and rises through and after it — "exactly the causal structure of the task".

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Ophys/behaviour frame mismatch**: the ophys series are truncated to the behaviour length
  (10 sessions have exactly one extra imaging frame — the reference's "one frame correction").
- **Teleport-frame position artefact**: the teleport sample is excluded from every trial.
- **Lick-sensor error**: those 81 trials are dropped (1-e).
- **Non-finite values** in neural/position/speed/lick: the trial is dropped and counted
  (`n_dropped_nan`, 0 everywhere).
- **Reward timestamps past the last frame**: `searchsorted` result clipped into range.
- **`keep_teleports` window underflow** (previous teleport + 1 after the next trial start): guarded
  with `min(...)`; never occurs.
- **Position/speed slightly out of range** (−2.74…451.83 cm; −6.45…133.1 cm/s): absorbed by the open
  end bins.
- **First trial of a session** has no predecessor: `previous_trial_outcome = 0`.
- Structural assertions: `#trial_start == #teleport`, `teleport > trial_start`.

ii.
```python
F_list.append(np.asarray(rrs.data[:nframes, :])[:, mask].T.astype(np.float32))
...
reward_frames = np.clip(reward_frames, 0, nframes - 1)
...
assert len(tstart_inds) == len(teleport_inds), ...
assert np.all(teleport_inds > tstart_inds), ...
...
windows.append((int(min(teleport_inds[i - 1] + 1, tstart_inds[i])), int(teleport_inds[i])))
...
if (not np.all(np.isfinite(neural)) or not np.all(np.isfinite(pos))
        or not np.all(np.isfinite(speed)) or not np.all(np.isfinite(lick))):
    n_dropped_nan += 1
    continue
```

iii. CONVERSION_NOTES Step 10, Check 5 is a table of exactly these edge cases, each with the finding
from a scan over all 152 files and the handling. The AI states the NaN guard is defensive ("no NaNs
in behaviour ... still guarded") and that no minimum-trial-length filter is needed because the
shortest trial is 96 frames.

## 13-a. What are the most time-consuming steps of the code?

i. Measured per session and printed for every session (`t_load`, `t_dff`, `t_total`):
1. the dF/F pipeline (0.9–4 s/session, the largest single cost),
2. reading `F`/`Fneu` out of the NWB file (0.2–1.5 s/session),
3. writing the 9.63 GB pickle (9.4 s, single-threaded at the end),
4. trial assembly (<0.5 s/session).
Whole-dataset wall clock: 48.3 s conversion + 9.4 s pickle with 12 worker processes.

ii.
```python
    t0 = time.time()
    sess = load_session(path)
    t_load = time.time() - t0
    ...
    t1 = time.time()
    dff, events, nanmask = compute_dff_events(...)
    t_dff = time.time() - t1
    ...
    print(f"... | load {info['t_load']:.1f}s dff {info['t_dff']:.1f}s "
          f"tot {info['t_total']:.1f}s | elapsed {el:.0f}s", flush=True)
```

iii. CONVERSION_NOTES Step 7 has the timing table and the extrapolation ("single-threaded projection
was ~8–10 min and the 12-worker projection ~1–2 min. Actual full run: 57 s, well under the
15-minute budget"), as the instructions required.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI identified and vectorised the one expensive loop: the reference's per-cell `np.corrcoef`
interneuron test became a single matrix product (~100× faster by its estimate). Loops that remain
are (a) the per-window loops inside `compute_dff_events` (four of them: masking, baseline,
smoothing/deconvolution) — inherent to a per-trial baseline with variable-length windows, (b) the
per-trial statistics loop computing `isreward`/`morph`/`trialnum`/`lick_error`, and (c) the
per-trial assembly loop; both per-trial loops could in principle be done on whole-session arrays
with `np.add.reduceat`-style segment operations, and neither is flagged in the notes.

ii.
```python
def speed_correlation(dff, speed, nanmask):
    D = dff[:, nanmask].astype(np.float64)
    ...
    return (Dm @ Sm) / denom
```
```python
for i, (s, e) in enumerate(zip(tstart_inds, teleport_inds)):
    isreward[i] = ...
    m = np.unique(beh['morph'][s:e]); morph[i] = int(np.round(m[0]))
    tn = np.unique(beh['trialnum'][s:e]); trialnum[i] = int(tn[0])
    L = beh['lick'][s:e]; lick_error[i] = np.mean(L > 2) > LICK_ERROR_THRESH
```

iii. Step 6: "Code inefficiencies identified: naive per-cell `np.corrcoef` loop for the interneuron
test (O(n_cells) python loop) ... Code speedups added: interneuron correlations vectorised into one
matrix product (≈100× faster than the loop)." The remaining loops run over ≤100 trials and cost
<0.5 s/session, so the AI left them.

## 13-c. What processing does the code repeat multiple times?

i. Very little: each NWB file is opened and read exactly once, and the whole conversion is a single
pass (no separate survey pass). The repetitions that do exist are cheap: the trial windows are
iterated four times inside `compute_dff_events` and twice more in `convert_session` (statistics
loop, then assembly loop); under `--show-processing` the distance-to-reward-zone and the per-trial
slices are recomputed inside `plot_processing` instead of being reused; and `events` is computed in
addition to `dff` whenever plotting is on.

ii.
```python
    for s, e in windows:            # 1: mask
    ...
    for s, e in windows:            # 2: neuropil add-back + maximin baseline
    ...
    for s, e in windows:            # 3: smoothing (+ OASIS)
```
```python
    for i in kept_trial_idx[:ntr_plot]:
        p = beh['pos'][int(tstart_inds[i]):int(teleport_inds[i])]
        rs, re = rz_coords[i]
        d = np.zeros_like(p)
        d[p < rs] = p[p < rs] - rs        # recomputed for plotting
```

iii. Step 6 lists the I/O design decision: "whole `(n_frames, n_roi)` datasets read in one contiguous
call, then column-masked and transposed" and "OASIS deconvolution skipped entirely when the neural
signal is dF/F" — i.e. the AI deliberately avoided re-reading and redundant computation on the hot
path. The plotting duplication only runs for 2 sessions with `--show-processing`.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Small amounts, mostly unavoidable or deliberately avoided:
- dF/F must be computed for all `iscell` cells *before* the interneuron test (the test needs dF/F),
  so the 380 excluded cells' dF/F is computed and then discarded — unavoidable given the criterion.
- `autoreward` is read from every file and never used; `date` and per-neuron `planeIdx` are read and
  only stored in metadata.
- On `keep_teleports` sessions the dF/F is computed over the inter-trial interval as well, and those
  samples are then discarded when trials are sliced — but they are required by the reference's
  baseline definition.
- Under `--show-processing` only: OASIS events are deconvolved purely for a plot panel, and the
  distance-to-reward-zone is recomputed for plotting.
- The AI explicitly removed the one large item: OASIS deconvolution is skipped entirely in the
  default dF/F mode.
The notes do not discuss this question explicitly.

ii.
```python
    dff, events, nanmask = compute_dff_events(sess['F'], sess['Fneu'], windows, fs,
                                              deconvolve=(neural_signal == 'events'
                                                          or show_processing))
```
```python
        autoreward=get('autoreward'),   # read, never used
```

iii. Step 6, "Code speedups added": "OASIS deconvolution skipped entirely when the neural signal is
dF/F (~0.3–1.5 s/session)". No justification is given for the remaining minor items; they cost
negligible time (the whole conversion is 48 s).
