# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every NWB file under `/app/data/sub-*/` is discovered with a single `glob` and read directly with
`h5py` (not `pynwb`), one file per session, 152 files in total. Each session is read once, in a
worker process, and all three streams needed (behaviour time series, suite2p fluorescence/neuropil,
session metadata) are pulled out of that one open handle. There is no separate "survey" pass — the
reward-zone identity, which the human reference had to infer from a session-spanning survey, is read
out of the session's own `identifier` string, so one pass suffices. Sessions are afterwards sorted
by (mouse number, experiment day).

ii.
```python
DATA_ROOT = '/app/data'
...
files = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
...
with Pool(args.workers, initializer=_init, initargs=({},)) as pool:
    for i, res in enumerate(pool.imap(_worker, files, chunksize=1)):
```
```python
def convert_session(path, show_processing=False, outdir='/app'):
    with h5py.File(path, 'r') as f:
        beh, starts, stops = read_behavior(f)
        scene = f['identifier'][()].decode().split('/')[-1]
        subject = f['general/subject/subject_id'][()].decode()
        session_id = f['general/session_id'][()].decode()
        region = f['general/optophysiology/ImagingPlane/location'][()].decode()
        ...
        F, Fneu, n_iscell, nplanes = read_fluorescence(f)
```
```python
BEHAVIOR_PATH = 'processing/behavior/BehavioralTimeSeries'
OPHYS_PATH = 'processing/ophys'

def read_behavior(f):
    B = f[BEHAVIOR_PATH]
    beh = {k: B[f'{k}/data'][:] for k in
           ['position', 'speed', 'lick', 'reward_zone', 'environment',
            'trial_start', 'teleport', 'scanning']}
    beh['time'] = B['position/timestamps'][:]
```

iii. From CONVERSION_NOTES Step 2/Step 9: the DANDI export has exactly one `.nwb` per
subject/session, 152 files across 11 `sub-*` directories (12 sessions for m11, 14 for the other ten
mice), which the notes cross-check against the paper's "14 days/mouse; m11 imaged from day 3" and
"n = 11 switch mice". `h5py` was chosen over `pynwb` for speed (the notes report the whole 152-session
conversion in 38.6 s with 16 workers); the NWB paths used are documented in the Step-2 structure dump,
and Step-10 Check 2 re-derives values straight from the raw HDF5 with independently written code.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is taken from inside each NWB file (`general/subject/subject_id`, e.g. `m17`),
not from the directory name. The unique set is sorted by mouse number to form `subjects`, and each
session gets an index into that list.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
...
subjects = sorted({r['info']['subject'] for r in results}, key=lambda s: int(s[1:]))
subject_idx = np.array([subjects.index(r['info']['subject']) for r in results], dtype=np.int64)
```

iii. Step 2 of the notes records that the data directory is one level deep with 11 `sub-<id>`
folders and that the in-file `subject_id` matches the folder name; 11 mice matches the paper's
"n = 11 mice" for the switch task. The notes also record an issue found and fixed here: sessions were
first sorted by subject *string* (`m11 < m3`), which was changed to numeric mouse/day order.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. The experiment day is read from `general/session_id` and the scene
(task condition) from `identifier`; sessions are ordered by (mouse, day). No cross-session neuron
alignment is attempted, so each session's neurons are independent.

ii.
```python
session_id = f['general/session_id'][()].decode()
scene = f['identifier'][()].decode().split('/')[-1]
...
results.sort(key=lambda r: (int(r['info']['subject'][1:]), int(r['info']['session_id'])))
```

iii. Step 9's consistency table checks 152 sessions, 12 for m11 and 14 for every other mouse,
against the paper's statement that imaging ran 14 days per mouse except m11 (from day 3). Step 4
resolves the paper's 12,376-trial figure as including m11's two unimaged behaviour-only days
(12,376 − 12,216 = 160 = 2 × 80 trials).

## 1-d. How are the data split into trials?

i. A trial is the on-track lap `[trial_start_idx, teleport_idx)`: it starts at the sample where the
`trial_start` flag is set and ends (exclusive) at the first sample where the `teleport` flag is set.
Both flags are read as "any sample > 0", and the code asserts that starts and stops pair up one-to-one
and that every stop follows its start. This is the same window every behavioural function in the
reference repo uses (`get_trial_types`, `lickrate_PETH`, `correct_lick_sensor_error` all slice
`sess.trial_start_inds[t] : sess.teleport_inds[t]`). Because `pp.dff` internally slices
`start-1 : stop-1`, the notes state the intent to pass `starts+1/stops+1` so that the neural window is
the same sample range; in the actual implementation dF/F is computed directly on `[s, e)`, which is
the same window.

ii.
```python
    starts = np.where(beh['trial_start'] > 0)[0]
    stops = np.where(beh['teleport'] > 0)[0]
    assert len(starts) == len(stops) and np.all(stops > starts), 'bad trial boundaries'
    return beh, starts, stops
```
```python
    pos_tr = [beh['position'][s:e] for s, e in zip(starts, stops)]
    speed_tr = [beh['speed'][s:e] for s, e in zip(starts, stops)]
    ...
    dff_trials = compute_dff_trials(F, Fneu, starts, stops)
```

iii. Step 5 decision 2: "Trial window = `[trial_start_idx, teleport_idx)` — exactly the on-track
samples, matching every behavioural function in the reference". Step 10 Check 3 notes this
deliberately uses one window for both the neural and behavioural streams, removing the reference's
own one-sample inconsistency between `pp.dff` and its behaviour functions, and validates the trial
definition by reproducing the paper's lick-sensor-error count exactly (81 trials). The stored
`trial number` series was not used for boundaries. (Independently verified here: both flags are
0/1 impulses, and the resulting boundaries are identical to the human reference's rising-edge
detection, giving the same 12,216 raw trials.)

## 1-e. How are trials filtered based on quality controls?

i. One trial filter: trials flagged by the paper's capacitive-lick-sensor error test are dropped
entirely. A trial is flagged when more than 30 % of its frames have a cumulative lick count > 2
(`behavior.correct_lick_sensor_error` with `correction_thr = 0.3`). 81 trials of 12,216 (0.66 %) are
removed, leaving 12,135. No other trial exclusions: the notes verify that no trial overlaps the
pre-sync frames (`scanning != 1`), that every trial has a single environment value, and that the
shortest trial is 96 frames, so no minimum-length filter is applied. Note the reference only sets the
*licks* of these trials to NaN; the AI removes the whole trial (neural and all other outputs) because
NaN is not a legal categorical output.

ii.
```python
LICK_CORRECTION_THR = 0.3  # Methods: ">30% of the ... samples ... cumulative lick count >2"
...
    # behavior.correct_lick_sensor_error -> trials to drop
    lick_error = np.array([(l > 2).sum() / len(l) > LICK_CORRECTION_THR for l in lick_tr])
...
    for t in range(ntrials):
        if lick_error[t]:
            continue
```

iii. Step 5 decision 5 and Step 3 curation rules: the Methods say "n = 81 out of 12,376 trials
removed"; the AI's threshold sweep (Step 4) gives 81 trials at thr 0.3, 69 at 0.35 and 44 at 0.5, so
0.3 — the value quoted in the Methods — is used and reproduces the paper's number exactly. The notes
explicitly justify dropping rather than NaN-ing: "the reference NaNs their licks; NaN is not
representable in a categorical output".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The raw suite2p traces: `processing/ophys/Fluorescence/plane*/data` (F) and
`processing/ophys/Neuropil/plane*/data` (Fneu), restricted to the `iscell`-curated ROIs and pooled
across imaging planes. The NWB `Deconvolved` array is explicitly **not** used.

ii.
```python
def read_fluorescence(f):
    """Pooled (ncells, nframes) F and Fneu over all imaging planes, iscell-curated."""
    seg = f[f'{OPHYS_PATH}/ImageSegmentation/PlaneSegmentation']
    iscell = seg['iscell'][:, 0] > 0
    planes = sorted(f[f'{OPHYS_PATH}/Fluorescence'].keys(),
                    key=lambda p: int(p.replace('plane', '')))
    Fs, Ns = [], []
    for p in planes:
        rois = f[f'{OPHYS_PATH}/Fluorescence/{p}/rois'][:]
        keep = iscell[rois]
        Fs.append(f[f'{OPHYS_PATH}/Fluorescence/{p}/data'][:][:, keep])
        Ns.append(f[f'{OPHYS_PATH}/Neuropil/{p}/data'][:][:, keep])
    F = np.ascontiguousarray(np.concatenate(Fs, axis=1).T)
    Fneu = np.ascontiguousarray(np.concatenate(Ns, axis=1).T)
    return F, Fneu, int(iscell.sum()), len(planes)
```

iii. Step 4's discrepancy table: "NWB `Deconvolved` has F-scale amplitudes (max 10976) and no NaNs
⇒ suite2p `spks` from **raw F**", i.e. not the events the paper computes from its own dF/F, so it is
discarded and the signal is recomputed from F and Fneu with the reference's `preprocessing.dff`.
Planes are pooled because the Methods say "planes were pooled for all analyses except those in
Extended Data Fig. 7".

## 2-b. How is the `neural` data processed?

i. The paper's dF/F pipeline is re-implemented per trial: subtract `0.7 × Fneu`, add back the trial's
mean neuropil (×0.7) so the baseline is not near zero, compute a "maximin" baseline (Gaussian
smoothing with σ = 15 frames, then a 300-frame ≈20 s minimum filter followed by a 300-frame maximum
filter), form `(F − F0)/|F0|`, and smooth with a 2-frame s.d. Gaussian. **The result is stored as the
neural signal; the OASIS deconvolution step that the paper applies afterwards is deliberately not
run.** The teleport periods are always excluded from the baseline (`keep_teleports = False` for every
session — the per-animal/per-day `teleport_metadata.teleport_sessions` table the reference's
`make_multi_anim_sess` notebook uses is not applied). Non-finite dF/F values would be warned about and
zeroed; none occurred in the full run.

ii.
```python
def compute_dff_trials(F, Fneu, starts, stops):
    out = []
    for s, e in zip(starts, stops):
        f = F[:, s:e].astype(np.float64)
        fneu = Fneu[:, s:e].astype(np.float64)

        # neuropil subtraction, then add back the trial's mean neuropil so that the
        # baseline is not a near-zero number (preprocessing.dff)
        f = f - NEU_COEF * fneu
        f = f + NEU_COEF * np.mean(fneu, axis=1, keepdims=True)

        # "maximin" baseline: smooth, minimum-filter over ~20 s, then maximum-filter
        flow = nan_gaussian_filter1d(f, BASELINE_SMOOTH_SIGMA, axis=-1)
        flow = ndi.minimum_filter1d(flow, BASELINE_FILTER_WIN, axis=-1)
        flow = ndi.maximum_filter1d(flow, BASELINE_FILTER_WIN, axis=-1)

        dff = (f - flow) / np.abs(flow)
        dff = nan_gaussian_filter1d(dff, DFF_SMOOTH_SIGMA, axis=-1)
        out.append(dff.astype(np.float32))
    return out
```
```python
NEU_COEF = 0.7             # preprocessing.dff / utilities.default_dff_method
BASELINE_SMOOTH_SIGMA = 15  # frames, nansmooth(f_, [0, 15]) in preprocessing.dff
BASELINE_FILTER_WIN = 300  # frames, ~20 s minimum/maximum filter ("maximin")
DFF_SMOOTH_SIGMA = 2       # frames, "two-sample (~0.129 s) s.d. Gaussian kernel"
```

iii. Step 5 decision 1 and Step 10 Check 3 difference 4: the dF/F arithmetic and constants are taken
from `preprocessing.dff`/`utilities.multi_anim_sess`. Deconvolution is skipped on two grounds —
(a) "it needs the per-session suite2p `ops['tau']`, which is not present in the NWB export;
introducing a guessed kernel would be a worse match to the reference than using the dF/F the paper
itself defines", and (b) an explicit experiment (Step 8, `cache/compare_signal.py`) in which the
sample was rebuilt with `dcnv.oasis(dff, 2000, tau=0.7, 15.5 Hz)` and re-decoded: dF/F gave the
better validation balanced accuracy on 5 of 6 outputs (e.g. position 0.713 vs 0.547). Step 10
Check 2 also audits the dF/F distribution (per-session max |dF/F| median 4.88; 7 × 10⁻⁶ of values
exceed 10, from cells whose maximin baseline lands near zero) and keeps them because the reference
`dff` divides by `|F0|` with no guard.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both the paper's. First, ROIs are restricted to suite2p's manually curated cells
(`PlaneSegmentation/iscell[:, 0] == 1`), applied per plane through each plane's `rois` index before
the traces are pooled. Second, putative interneurons are dropped: any cell whose dF/F correlates with
running speed at Pearson r > 0.5, computed over all in-trial samples. 402 of 138,678 curated cells
(0.29 %; 0.35 ± 0.61 % per session) are removed, leaving 138,276.

ii.
```python
    seg = f[f'{OPHYS_PATH}/ImageSegmentation/PlaneSegmentation']
    iscell = seg['iscell'][:, 0] > 0
    ...
        rois = f[f'{OPHYS_PATH}/Fluorescence/{p}/rois'][:]
        keep = iscell[rois]
```
```python
def find_putative_interneurons(dff_trials, speed_trials, r_thresh=INTERNEURON_R_THRESH):
    """spatial.is_putative_interneuron(method='speed'): Pearson r(dF/F, speed) > thresh."""
    ...
    for d, v in zip(dff_trials, speed_trials):
        d64 = d.astype(np.float64)
        n += d64.shape[1]
        sx += d64.sum(axis=1); sxx += np.einsum('ij,ij->i', d64, d64); sxy += d64 @ v
        sy += v.sum(); syy += v @ v
    cov = sxy - sx * sy / n
    ...
    return np.nan_to_num(r, nan=0.0) > r_thresh, r
...
    is_int, speed_r = find_putative_interneurons(dff_trials, [v.astype(np.float64) for v in speed_tr])
    keep_cells = ~is_int
    dff_trials = [d[keep_cells] for d in dff_trials]
```

iii. Step 3 curation rules and Step 5 decision 4: `iscell` is the manual curation described in the
Methods; the interneuron rule is the Methods' "Pearson correlation of >0.5 between their dF/F
timeseries and the animal's running speed". Sanity checks in Step 9/Step 10: minimum cells per
session = 155, matching the paper's quoted range "155–2172" exactly at the low end; switch-day mean
967.5 ± 464.6 vs the paper's 954 ± 453; interneuron exclusion 0.35 ± 0.61 % vs the paper's
0.42 ± 0.85 %. The notes flag the one residual mismatch (max 2323 cells/session vs the paper's 2172,
from the 2-plane mouse m18) as unexplained but not a conversion error. The streaming-accumulator
formulation is documented as a memory optimisation that is mathematically the same correlation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to trial start, which requires nothing beyond the trial split: neural and behavioural
samples share one index base (VR is already resampled onto the imaging frame grid in the export), and
the neural slice uses exactly the same `[s, e)` window as every behavioural slice, so sample 0 of each
neural matrix is the trial-start frame. `off_start = 0.0`, `off_end = None` (variable trial lengths).

ii.
```python
    pos_tr = [beh['position'][s:e] for s, e in zip(starts, stops)]
    ...
    dff_trials = compute_dff_trials(F, Fneu, starts, stops)   # same [s, e) window
```
```python
            'temporal_alignment_event': (
                'start of trial: the frame the animal enters the linear track at 0 cm '
                '(vr_data "tstart" flag / trial_start_inds)'),
            'off_start': 0.0,
            'off_end': None,
```

iii. Step 5 decisions 2–3 and Step 10 Check 3(c): using one window for both streams removes the
reference's internal one-sample offset between `pp.dff` (`start-1:stop-1`) and its behaviour
functions (`start:stop`). Step 3 notes that "VR behaviour is already interpolated onto the imaging
frame grid (one sample per 2P frame)". Alignment is verified twice: the sanity-check script
re-derives dF/F for 18 random (session, trial) pairs from the raw NWB and matches with
`np.allclose`, and `predictions.png` from the decoder run shows predicted position/distance tracking
the true step functions "with no visible temporal lag".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning at all — the data stay at the native imaging frame, 64.4836 ms (15.5078125 Hz). The
rate is derived from the behaviour timestamps rather than from `ImagingPlane/imaging_rate`, because
for the two 2-plane mice (m17, m18) the stored rate is the 31 Hz scanner rate, twice the per-plane
sampling rate. The code asserts that all 152 sessions share the same rate before writing
`time_bin_size`.

ii.
```python
        # ImagingPlane/imaging_rate is the total scan rate (31.0 Hz for the two
        # 2-plane mice m17/m18); the effective per-plane sampling rate -- which is
        # what every timeseries in the file is sampled at -- is 15.5078125 Hz for
        # every session.  Derive it from the frame timestamps.
        frame_rate = float(1.0 / np.median(np.diff(beh['time'])))
```
```python
    frame_rates = np.array([r['info']['frame_rate'] for r in results])
    assert np.allclose(frame_rates, frame_rates[0], rtol=1e-6), \
        f'inconsistent frame rates: {np.unique(frame_rates)}'
    bin_ms = 1000.0 / float(np.mean(frame_rates))
```

iii. Step 5 decision 3 ("No resampling: time bin = the native imaging frame, 1000/15.5078125 =
64.4839 ms") and Step 10 Check 3(d): the paper's 10 cm spatial binning is for its spatial analyses
and is not applicable to a time-resolved decoder. Step 10's "Issues Found and Resolved" records that
the first full run asserted on "multiple frame rates" because of the 31 Hz `imaging_rate` for the
2-plane mice, and that deriving the rate from timestamps fixed it. The stored value,
64.48362720402656 ms, agrees with the paper's "0.0645 s imaging frame samples".

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `position` time series' own `timestamps` array (all behaviour series in the file share the same
timestamp vector).

ii.
```python
    beh['time'] = B['position/timestamps'][:]
    ...
    time_tr = [beh['time'][s:e] - beh['time'][s] for s, e in zip(starts, stops)]
```

iii. The Step-5 variable-mapping table lists `BehavioralTimeSeries/position/timestamps` →
`input[0]`, transform "t − t[tstart] (s)". Step 2 records that all behaviour series are on the same
timestamp grid, so the choice of which series to take the timestamps from is arbitrary. The Step-10
sanity check re-derives input 0 straight from `position/timestamps` and matches.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Only subtraction of the trial's first timestamp, giving seconds since trial start; stored as
float32. Range across the dataset is [0, 216.5] s (the long tail is one 3,359-frame trial in which
the animal stopped).

ii.
```python
    time_tr = [beh['time'][s:e] - beh['time'][s] for s, e in zip(starts, stops)]
...
        inp = np.empty((4, T), dtype=np.float32)
        inp[0] = time_tr[t]
```

iii. No justification beyond the decoder spec ("Time from start of trial in seconds"); Step 9's
consistency table records the resulting range and attributes the 216.5 s maximum to a single stopped
trial rather than to a boundary error.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. No alignment step is needed: the timestamps are sliced with the same `[s, e)` window as the neural
data, and the behaviour is already on the imaging frame grid, so element *k* of the time vector is the
same frame as column *k* of the neural matrix.

ii.
```python
    time_tr = [beh['time'][s:e] - beh['time'][s] for s, e in zip(starts, stops)]
    ...
    dff_trials = compute_dff_trials(F, Fneu, starts, stops)
    ...
        T = len(pos_tr[t])
        inp = np.empty((4, T), dtype=np.float32)
```

iii. Step 5 decision 2 ("giving the identical sample window for neural and behavioural streams — no
temporal misalignment") and Step 10 Check 2, where input 0 is re-derived from the raw file for
randomly chosen trials and compared with `np.allclose`. The per-trial time dimension is also asserted
equal across neural/input/output for every session in the structural checks.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behaviour time series (the export of the reference's `vr_data['morph']`):
0 = ENV1, 1 = ENV2.

ii.
```python
    env_tr = [beh['environment'][s:e] for s, e in zip(starts, stops)]
    ...
    # behavior.get_trial_types
    env = np.array([np.unique(e)[0] for e in env_tr])
    for e in env_tr:
        assert len(np.unique(e)) == 1, 'environment changes within a trial'
```

iii. Step 4's discrepancy table maps `morph` 0 → ENV1 and 1 → ENV2 following
`behavior.get_trial_types`, notes that `environment` ∈ {0, 1} inside trials (−1 only before VR/2P
sync, which no trial overlaps), and records that 11 sessions contain both environments (the day-8
`EnvX_?_to_EnvY_?` sessions) with the change always falling exactly on a trial boundary.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Take the single unique value within the trial and broadcast it across all timepoints of that trial;
an assertion guards against a trial containing more than one value. No other transformation.

ii.
```python
    env = np.array([np.unique(e)[0] for e in env_tr])
    for e in env_tr:
        assert len(np.unique(e)) == 1, 'environment changes within a trial'
    ...
        inp[1] = env[t]
```

iii. Step 4: "Use the unique per-trial value", justified by the observation that no trial has mixed
environment values, and matching `get_trial_types`, which likewise takes
`np.unique(sess.vr_data['morph'][firstI:lastI])` per trial. The decoder spec asks for a per-trial
binary variable, so broadcasting the constant is the required representation.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Not from any stored variable — it is the 0-based ordinal index of the trial within the session, i.e.
the index into the `trial_start`/`teleport` boundary arrays. The NWB `trial number` series is not used.

ii.
```python
    for t in range(ntrials):
        if lick_error[t]:
            continue
        ...
        inp[2] = t                                  # original index within the session
```

iii. Step 5 decision 9: "Trial number keeps its original within-session index even when an earlier
trial has been removed by lick-sensor curation, so that the input is the true ordinal position in the
session and `prev_trial_outcome` refers to the true preceding lap." Trial boundaries themselves come
from the `trial_start`/`teleport` flags (question 1-d), whose validity the notes establish by
reproducing the paper's 81 lick-error trials.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond assigning the loop index and broadcasting it across the trial's timepoints (float32).
Because the index is the pre-filtering index, the values in a session with dropped trials have gaps,
and the maximum observed is 99 (a 100-trial session).

ii.
```python
        inp[2] = t                                  # original index within the session
```

iii. Step 5 decision 9, as above; Step 10's sanity check verifies input 2 against "the original
within-session index of the kept trials". The Step-9 table records the resulting range [0, 99].

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the per-trial reward outcome vector `isreward`, which is built from the `Reward` event series
(its own timestamps, mapped to the nearest imaging frame) combined with the `reward_zone` series: a
trial counts as rewarded if it contains both a reward event and a reward-zone entry. The previous
trial's value is then used.

ii.
```python
    rt = B['Reward/timestamps'][:]
    ts = beh['time']
    idx = np.clip(np.searchsorted(ts, rt), 0, len(ts) - 1)
    left = np.clip(idx - 1, 0, len(ts) - 1)
    idx = np.where(np.abs(ts[left] - rt) < np.abs(ts[idx] - rt), left, idx)
    reward = np.zeros(len(ts))
    reward[idx] = 1.0
    beh['reward'] = reward
```
```python
    # behavior.get_trial_types
    isreward = np.array([float(np.any(r > 0) and np.any(z > 0))
                         for r, z in zip(rew_tr, rz_tr)])
```

iii. Step 5's mapping table cites `behavior.get_trial_types` as the source of the definition (the
reference's `(np.any(tmp_reward > 0) and np.any(tmp_rzone > 0)) * 1`). The notes explain that the
`Reward` series is an event series with its own timestamps, so each event is mapped back onto the
nearest imaging frame to reconstruct the per-frame binary `vr_data['reward']` the reference works
with; Step 10's edge-case table records that 3 of 10,345 reward events fall outside any trial window
and are ignored, "exactly as `get_trial_types` does".

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial *t* > 0 the value is `isreward[t−1]`, i.e. the outcome of the immediately preceding lap
in the raw trial indexing (so a trial dropped for lick-sensor error still provides the previous-trial
value). For the first trial of a session the value is 0. The value is broadcast across all timepoints.

ii.
```python
        inp[3] = isreward[t - 1] if t > 0 else 0.0  # no known preceding lap for trial 0
```

iii. Step 5 decision 8: "There is no previous trial inside the recording (the 30 pre-session warm-up
trials are not in the data), so 'no known preceding reward' is encoded as 0. This affects
152/12,135 = 1.25 % of trials." Decision 9 explains the use of the raw index so the reference is to
the true preceding lap. Step 10's sanity check re-derives input 3 as "`any(reward) & any(rzone)` of
the preceding lap" from the raw file.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` behaviour series and the active reward zone's coordinates. The zone identity is
**not** inferred from the animal's behaviour: it is parsed from the session's scene name (the NWB
`identifier`, e.g. `Env1_LocationB_to_A`), using the logic of `behavior.get_reward_zones` with the
switch occurring after 30 trials; the coordinates come from the reference's `reward_zone_dict`
(A = X = 80–130, B = Y = 200–250, C = Z = 320–370 cm).

ii.
```python
# reward_relative.behavior.reward_zone_dict, via map_labels A->X, B->Y, C->Z
REWARD_ZONE_COORDS = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
CHANGE_TRIAL = 30          # behavior.get_reward_zones default; "each switch occurred after 30 trials"

def get_reward_zone_labels(scene, ntrials, change_trial=CHANGE_TRIAL):
    """Per-trial reward zone label ('A'/'B'/'C'), replicating behavior.get_reward_zones."""
    for lab in 'ABC':
        if scene.endswith('Location' + lab):
            return np.array([lab] * ntrials)
    for first in 'ABC':
        if f'{first}_to' in scene:
            second = scene[-1]
            if second not in 'ABC':
                break
            n0 = min(change_trial, ntrials)
            return np.array([first] * n0 + [second] * (ntrials - n0))
    raise NotImplementedError(f'Reward zones not defined for scene {scene}')
```
```python
        lo, hi = REWARD_ZONE_COORDS[labels[t]]
        dist = reward_zone_distance(pos_tr[t], lo, hi)
```

iii. Step 5 decision 7: "Reward zone from the scene name (`behavior.get_reward_zones` logic) rather
than from the observed rzone-entry position, because the zone is defined even on trials where the
animal never enters it. Validated against the data (0/12,216 mismatches)." Step 4 records the same
check against the measured first-reward-zone-entry position, confirming both the scene parsing and
`change_trial = 30`.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from the animal's position to the nearest point of the active zone: negative before
the zone (`pos − lo`), positive after it (`pos − hi`), and exactly 0 anywhere inside. Computed
vectorised over the trial, then discretised (7-c).

ii.
```python
def reward_zone_distance(pos, lo, hi):
    """Signed distance to the nearest point of the reward zone; 0 inside the zone."""
    return np.where(pos < lo, pos - lo, np.where(pos > hi, pos - hi, 0.0))
```

iii. The decoder task asks for "distance to any location in the reward zone", which the notes read as
0 while the animal is anywhere in the 50 cm zone; the zone bounds are the paper's
(A 80–130, B 200–250, C 320–370 cm). Step 9 sanity-checks the resulting class distribution: bin 3
("0 cm") holds 23.9 % of timepoints, consistent with the animal spending extra time in the zone (the
zone is 50/450 ≈ 11 % of the track by length, but licking/slowing inflates the occupancy).

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes assigned by explicit boolean masks: 0 for < −50; 1 for [−50, −10); 2 for [−10, 0);
3 for exactly 0 (inside the zone); 4 for (0, +10]; 5 for (+10, +50]; 6 for > +50 cm. Stored as int8.

ii.
```python
def bin_reward_distance(dist):
    """0: <-50 | 1: [-50,-10) | 2: [-10,0) | 3: 0 | 4: (0,10] | 5: (10,50] | 6: >50"""
    b = np.zeros(dist.shape, dtype=np.int8)
    b[(dist >= -50) & (dist < -10)] = 1
    b[(dist >= -10) & (dist < 0)] = 2
    b[dist == 0] = 3
    b[(dist > 0) & (dist <= 10)] = 4
    b[(dist > 10) & (dist <= 50)] = 5
    b[dist > 50] = 6
    return b
```
```python
OUTPUT_VALUES = [
    ['< -50 cm', '-50 to -10 cm', '-10 to <0 cm', '0 cm (in zone)',
     '>0 to +10 cm', '+10 to +50 cm', '> +50 cm'], ...
```

iii. Step 5 decision 10: "Discretisation exactly as specified by the decoder task". The class 3
("0 cm") is defined as *exactly* zero, which by construction is the in-zone case. The
`--show-processing` figure (panel "step 5") overlays the discretised bins on the raw position and
zone extent so the thresholds can be checked visually, and Step 10's sanity check recomputes the bins
from raw `position` for random trials with exact equality.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. No separate alignment: position is sliced with the same `[s, e)` trial window as the neural data,
so the distance series is sample-for-sample aligned with the neural matrix.

ii.
```python
    pos_tr = [beh['position'][s:e] for s, e in zip(starts, stops)]
    ...
        T = len(pos_tr[t])
        dist = reward_zone_distance(pos_tr[t], lo, hi)
        out = np.empty((6, T), dtype=np.int8)
        out[0] = bin_reward_distance(dist)
```

iii. Same as 2-d/3-c: one window for all streams; verified by the sanity-check script and by the
lag-free predicted-vs-true traces in `predictions.png` (Step 11).

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behaviour time series (cm along the 450 cm virtual track), used directly.

ii.
```python
    beh = {k: B[f'{k}/data'][:] for k in
           ['position', 'speed', 'lick', 'reward_zone', 'environment',
            'trial_start', 'teleport', 'scanning']}
    ...
    pos_tr = [beh['position'][s:e] for s, e in zip(starts, stops)]
```

iii. The Step-5 mapping table maps `position` → `output[1]` with transform "5 bins of 90 cm". Step 4
notes that `position` is −500 before VR/2P sync but that no trial overlaps those frames, and Step 9
records the in-trial range as −2.7 … 451.8 cm against the paper's 450 cm track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None before discretisation — the raw cm values are binned directly.

ii.
```python
        out[1] = bin_position(pos_tr[t])
```

iii. The paper's 10 cm spatial binning is for spatial analyses only (Step 10 Check 3(d)); the decoder
task prescribes its own 5-bin discretisation, so no other transformation is applied.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90 cm bins over the 450 cm track, by `floor(pos / 90)` clipped to [0, 4], so the few
samples marginally outside the track (−2.7 cm, 451.8 cm) fall into the end bins rather than forming
extra classes.

ii.
```python
TRACK_LENGTH = 450.0
N_POSITION_BINS = 5        # decoder task: 5 equal bins over the 450 cm track

def bin_position(pos):
    """5 equal bins spanning the 450 cm track (90 cm each)."""
    b = np.floor(pos / (TRACK_LENGTH / N_POSITION_BINS))
    return np.clip(b, 0, N_POSITION_BINS - 1).astype(np.int8)
```

iii. The decoder task specifies "5 equal-sized bins spanning the 450 cm track"; the Methods give the
450 cm length. Step 10's edge-case table records "Position slightly outside [0, 450] at the trial
edges (−2.7 … 451.8 cm) → `bin_position` clips to [0, 4]". Step 9 checks that all five bins are
populated (0.212 / 0.177 / 0.231 / 0.226 / 0.154), the non-uniformity reflecting the animal's
occupancy.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[s, e)` window as the neural data; no further alignment.

ii.
```python
    pos_tr = [beh['position'][s:e] for s, e in zip(starts, stops)]
    ...
        out[1] = bin_position(pos_tr[t])
```

iii. As in 2-d; the position/neural alignment is the one explicitly checked in the decoder's
`predictions.png` and in the `--show-processing` "step 1" panel, which overlays the
`[trial_start, teleport)` windows on the raw position trace.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behaviour time series (per-frame lick counts from the capacitive sensor).

ii.
```python
    lick_tr = [beh['lick'][s:e] for s, e in zip(starts, stops)]
```

iii. The Step-5 mapping table maps `lick` → `output[3]` with transform "`lick>0 → 1`", citing
`behavior.lickrate`/`lickrate_PETH`, which likewise binarise the raw counts. The same series drives
the lick-sensor-error trial exclusion (1-e).

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarisation: any frame with a positive lick count becomes 1, everything else 0 (int8). Trials on
which the sensor was stuck are not corrected in place but removed entirely (see 1-e), so no NaN
handling is needed here. 22.3 % of retained timepoints are licks.

ii.
```python
        out[3] = (lick_tr[t] > 0).astype(np.int8)
```

iii. The decoder task specifies a binary lick output. The Methods describe the same binarisation for
the paper's lick-rate analyses ("Remaining lick counts were converted to a binary vector"). Step 5
decision 5 explains why the erroneous trials are dropped instead of NaN-ed.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[s, e)` window as the neural data; no additional alignment.

ii.
```python
    lick_tr = [beh['lick'][s:e] for s, e in zip(starts, stops)]
    ...
        out[3] = (lick_tr[t] > 0).astype(np.int8)
```

iii. As in 2-d; Step 10's sanity check recomputes output 3 from the raw `lick` array for random trials
with exact equality.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The session's scene name in the NWB `identifier` field, combined with the 30-trial switch rule —
the same source as 7-a. The `reward_zone` behaviour series is used only to validate the labels (and
for the reward-outcome definition), not to assign them.

ii.
```python
        scene = f['identifier'][()].decode().split('/')[-1]
        ...
    # behavior.get_reward_zones
    labels = get_reward_zone_labels(scene, ntrials)
```

iii. Step 5 decision 7 and Step 4: the scene-derived label was checked against the position of the
animal's first reward-zone entry on every trial — 0 of 12,216 mismatches — which simultaneously
confirms the scene parsing, the A/B/C coordinate mapping and `change_trial = 30`. Using the scene
rather than the observed entry means the zone is also defined on omission trials and on trials where
the animal never entered the zone.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The per-trial letter is mapped to an integer (A → 0, B → 1, C → 2) and broadcast across the trial's
timepoints. On switch sessions the first `min(30, ntrials)` trials get the pre-switch zone and the
rest the post-switch zone.

ii.
```python
            n0 = min(change_trial, ntrials)
            return np.array([first] * n0 + [second] * (ntrials - n0))
...
        out[4] = 'ABC'.index(labels[t])
```
```python
    ['zone A (80-130 cm)', 'zone B (200-250 cm)', 'zone C (320-370 cm)'],
```

iii. Step 10's edge-case table notes the `min(30, ntrials)` clamp is defensive (the shortest switch
session has 41 trials). Step 9 reports the resulting class balance, 0.344 / 0.327 / 0.329 per trial,
as evidence the counterbalanced design is reproduced.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` event series (delivery timestamps) together with the `reward_zone` series, exactly as
in `behavior.get_trial_types`: a trial is rewarded iff it contains at least one reward event **and**
at least one reward-zone sample.

ii.
```python
    rt = B['Reward/timestamps'][:]
    ts = beh['time']
    idx = np.clip(np.searchsorted(ts, rt), 0, len(ts) - 1)
    left = np.clip(idx - 1, 0, len(ts) - 1)
    idx = np.where(np.abs(ts[left] - rt) < np.abs(ts[idx] - rt), left, idx)
    reward = np.zeros(len(ts))
    reward[idx] = 1.0
    beh['reward'] = reward
...
    rz_tr = [beh['reward_zone'][s:e] for s, e in zip(starts, stops)]
    rew_tr = [beh['reward'][s:e] for s, e in zip(starts, stops)]
    isreward = np.array([float(np.any(r > 0) and np.any(z > 0))
                         for r, z in zip(rew_tr, rz_tr)])
```

iii. Step 5's mapping table cites `behavior.get_trial_types`. Reward events carry their own
timestamps, so each is snapped to the *nearest* imaging frame (the code compares the `searchsorted`
insertion point with its left neighbour). Step 4 checks the resulting reward rate: 10,342/12,216 =
84.7 %, against the paper's "reward was randomly omitted on ~15 % of trials".

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Per trial, the boolean above is cast to 0/1 and broadcast across all timepoints of the trial
(int8). Reward events falling outside any trial window (3 of 10,345) are ignored.

ii.
```python
        out[5] = int(isreward[t])
```
```python
    ['omitted', 'rewarded'],
```

iii. The decoder task asks for a per-trial binary outcome; the definition is the reference's
`get_trial_types`. Step 12 investigates the relatively low decoder accuracy on this output
(0.609 validation balanced accuracy) and concludes it is a property of the experiment — omissions are
random and only distinguishable from neural activity after the animal has passed the zone — rather
than a conversion error.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases, mostly caught by assertions plus a documented edge-case audit:
- **Trial boundaries**: asserted one-to-one and ordered (`len(starts) == len(stops)`, `stops > starts`).
- **Pre-sync frames** (`position = −500`, `trial number = −1`, `scanning = −1`): verified never to
  overlap a trial window, so no masking is needed.
- **Environment changing within a trial**: asserted not to happen.
- **Non-finite dF/F**: counted, warned about and replaced with 0 (none occurred in the full run).
- **Near-zero maximin baseline → very large dF/F**: kept deliberately (reference behaviour), after
  quantifying that 7 × 10⁻⁶ of values exceed 10.
- **Position slightly outside the track**: clipped into the end position bins.
- **Reward events outside any trial**: ignored.
- **First trial of a session has no predecessor**: `prev_trial_outcome = 0`.
- **Switch sessions shorter than the 30-trial switch point**: `min(change_trial, ntrials)` clamp.
- **Lick-sensor errors**: whole trial dropped (1-e).
- **2-plane sessions' `imaging_rate` being the 31 Hz scan rate**: frame rate taken from timestamps
  instead, with a cross-session equality assertion.
- **Worker failures**: re-raised with the offending file name.
Not explicitly handled: 10 sessions in which the imaging arrays have exactly one more frame than the
behaviour arrays (m17 ses-04/06, m18 ses-01/05/07/10/11/12/13/14). This is harmless here because all
trial windows are defined on behaviour indices and end before the last behaviour sample, so the extra
neural frame is simply never indexed; the human reference prints a warning and crops instead.

ii.
```python
    assert len(starts) == len(stops) and np.all(stops > starts), 'bad trial boundaries'
```
```python
    n_nonfinite = int(sum(np.sum(~np.isfinite(d)) for d in dff_trials))
    if n_nonfinite:
        warnings.warn(f'{os.path.basename(path)}: {n_nonfinite} non-finite dF/F values '
                      f'set to 0')
        dff_trials = [np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
                      for d in dff_trials]
```
```python
    for e in env_tr:
        assert len(np.unique(e)) == 1, 'environment changes within a trial'
...
        inp[3] = isreward[t - 1] if t > 0 else 0.0  # no known preceding lap for trial 0
...
    return np.clip(b, 0, N_POSITION_BINS - 1).astype(np.int8)
```
```python
def _worker(path):
    try:
        return convert_session(path, **_WORKER_ARGS)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f'failed on {path}: {exc}')
```

iii. Step 10 Check 5 is an explicit edge-case table listing each of the above with the verification
that was run (e.g. "0/12,216 trials overlap `scanning != 1`", "3 of 10,345 rewards outside trial
windows — ignored, exactly as `get_trial_types` does"). The dF/F outlier decision is justified as
"the behaviour of the reference `preprocessing.dff` itself (it divides by `|F0|` with no guard)".

## 13-a. What are the most time-consuming steps of the code?

i. The script prints a per-stage timing breakdown for every session (`read_behavior`,
`read_fluorescence`, `dff`, `interneurons`, `total`) plus a running ETA. From
`conversion_full_out.txt` the dominant costs are (1) the dF/F computation (~0.1–2.9 s/session,
typically the largest single stage), (2) reading the fluorescence/neuropil arrays out of HDF5
(~0.1–2.8 s/session, I/O bound), and (3) writing the 9.5 GB output pickle (11.3 s, ~29 % of the total
38.6 s wall clock). The interneuron correlation is negligible (≤0.3 s). Total: 152 sessions in 38.6 s
on 16 worker processes.

ii.
```python
    t3 = time.time()
    timing['dff'] = t3 - t2
    ...
    timing['total'] = time.time() - t0
```
```python
    print(f"... t={info['timing']['total']:.1f}s "
          f"[read {info['timing']['read_fluorescence']:.1f} dff {info['timing']['dff']:.1f} "
          f"int {info['timing']['interneurons']:.1f}] "
          f"| elapsed {el:.0f}s, eta {el / (i + 1) * (n - i - 1):.0f}s", flush=True)
```

iii. Step 6/Step 7 of the notes: the timing instrumentation was added to locate bottlenecks, the
estimate from the 2-session sample was carried forward to the full run, and the notes record
"full conversion of 152 sessions in 39 s (of which 11 s is writing the 9.5 GB pickle)", well inside
the 15-minute budget, so no further optimisation was pursued.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Remaining per-trial Python loops: `compute_dff_trials` loops over trials (unavoidable in spirit —
the baseline must be computed within each trial — but the seven separate behaviour slicing
comprehensions in `convert_session` traverse `zip(starts, stops)` seven times and could be one loop or
a single ragged split); the `isreward`, `env` and `lick_error` list comprehensions each re-loop over
the trials; the assembly loop over trials builds the per-trial input/output arrays element by element,
and the discretisation (`bin_position`, `bin_speed`, `bin_reward_distance`) could be applied once to
the whole session array before splitting, since only the reward-zone distance depends on the trial.
`plot_processing` loops with `kept_trials.index(t)`, a linear search inside a loop. The code already
vectorises the two loops that mattered: the interneuron correlation (streaming sufficient statistics
instead of a per-cell `np.corrcoef` loop) and the per-cell dF/F arithmetic (whole-matrix filters).

ii.
```python
    pos_tr = [beh['position'][s:e] for s, e in zip(starts, stops)]
    speed_tr = [beh['speed'][s:e] for s, e in zip(starts, stops)]
    lick_tr = [beh['lick'][s:e] for s, e in zip(starts, stops)]
    rz_tr = [beh['reward_zone'][s:e] for s, e in zip(starts, stops)]
    rew_tr = [beh['reward'][s:e] for s, e in zip(starts, stops)]
    env_tr = [beh['environment'][s:e] for s, e in zip(starts, stops)]
    time_tr = [beh['time'][s:e] - beh['time'][s] for s, e in zip(starts, stops)]
```
```python
    for d, v in zip(dff_trials, speed_trials):
        d64 = d.astype(np.float64)
        n += d64.shape[1]
        sx += d64.sum(axis=1)
        sxx += np.einsum('ij,ij->i', d64, d64)
        sxy += d64 @ v
```

iii. Step 6 lists the inefficiencies that were identified and fixed ("naively slicing
`F[s:e, keep].T` per trial does a fancy-index + transpose copy on every trial"; "building the full
NaN-padded `(ncells, nframes)` dF/F array (as the reference does) wastes the ~15 % of frames in
teleport periods and doubles peak memory"; "concatenating all trials' dF/F to correlate with speed
doubles memory") and the speed-ups added (ROI filtering before the transpose, per-trial dF/F without
NaN padding, streaming correlation accumulators, 16-process pool). The remaining per-trial loops run
on behaviour vectors only and cost a negligible fraction of the 38.6 s.

## 13-c. What processing does the code repeat multiple times?

i. Little: each NWB file is opened and read exactly once, and there is no separate survey pass. What
is repeated: the seven behaviour slicing comprehensions each re-walk the trial list; `np.unique` is
called twice per trial on `environment` (once to take the value, once inside the assertion loop);
in `--show-processing` mode `plot_processing` recomputes the neuropil subtraction and the maximin
baseline for the example cell rather than reusing the values from `compute_dff_trials`, and repeatedly
calls `kept_trials.index(t)`; dF/F is computed for the putative interneurons and then discarded
(unavoidable, since the filter is defined on dF/F). The summary pass `_summarise` concatenates all
inputs and outputs a second time after assembly.

ii.
```python
    env = np.array([np.unique(e)[0] for e in env_tr])
    for e in env_tr:
        assert len(np.unique(e)) == 1, 'environment changes within a trial'
```
```python
        f = F[craw, s:e].astype(np.float64) - 0.7 * Fneu[craw, s:e].astype(np.float64)
        f = f + 0.7 * Fneu[craw, s:e].mean()
        flow = ndi.gaussian_filter1d(f, BASELINE_SMOOTH_SIGMA)
        flow = ndi.maximum_filter1d(ndi.minimum_filter1d(flow, BASELINE_FILTER_WIN),
                                    BASELINE_FILTER_WIN)
```

iii. Step 6 records the design goal of a single pass per file; the reward-zone identity, which is what
forced the human reference into a two-pass survey, is read from the scene name instead (Step 5
decision 7), so no second read of the NWB files is needed. The recomputation inside
`plot_processing` is diagnostic-only code that runs for at most two sessions.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Minor, all cheap: the `scanning` stream is read from every file and never used afterwards (it was
used only in the one-off edge-case audit); `find_putative_interneurons` returns the full correlation
vector, which is bound to `speed_r` and never used; `frame_rate`, `nplanes`, `n_iscell`,
`n_interneurons`, `reward_rate`, `kept_trials` and the timing dict are computed per session for
bookkeeping and only some reach the metadata; `plot_processing` contains a dead expression
(`rng = ... if False else None`). dF/F is computed for the ~0.3 % of cells that are then discarded as
interneurons, which is necessary because the interneuron test is defined on dF/F. Storing dF/F as
float32 for every neuron and timepoint produces a 9.5 GB pickle, of which the decoder consumes all of
it, so nothing is wasted there.

ii.
```python
    beh = {k: B[f'{k}/data'][:] for k in
           ['position', 'speed', 'lick', 'reward_zone', 'environment',
            'trial_start', 'teleport', 'scanning']}   # 'scanning' never used again
```
```python
    is_int, speed_r = find_putative_interneurons(...)   # speed_r unused
```
```python
    rng = np.array([np.ptp(d[:, :]) for d in [dff_trials[demo[0]]]])[0] if False else None
```

iii. Not discussed as such in CONVERSION_NOTES; the `scanning` read is explained by Step 10 Check 5,
where it was used to verify that no trial overlaps the pre-sync frames, and it was left in place
afterwards. None of these items is on the critical path (the full conversion takes 38.6 s).
