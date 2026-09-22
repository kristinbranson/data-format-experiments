#!/usr/bin/env python3
"""Convert the IBL Brain-Wide Map release to the decoder interchange format.

Processing follows the supplied BWM release and Zhang et al. code: released
sessions/insertions, RIGOR well-isolated units, stimulus alignment, 20-ms bins,
and a -0.5 to 1.5 s window.  The only new processing is tertile discretization
of wheel speed and whisker motion energy, required by the decoder task.
"""
from pathlib import Path
import pickle, warnings, traceback
import numpy as np
import pandas as pd
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from brainbox.behavior.wheel import interpolate_position, velocity_filtered
from iblatlas.atlas import AllenAtlas

ROOT = Path('/app')
CACHE = ROOT / 'data/one_cache'
RELEASE = ROOT / 'code/code_zhang2025/data/bwm_release.csv'
OUT = ROOT / 'converted_data.pkl'
DT = 0.020
OFF0, OFF1 = -0.5, 1.5
EDGES = np.arange(OFF0, OFF1 + DT/2, DT)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2


def newest(path, pattern):
    """Return newest revision of a file (revision dates sort lexically)."""
    fs = list(path.rglob(pattern))
    if not fs:
        raise FileNotFoundError(f'{pattern} below {path}')
    def key(p):
        rev = next((x[1:-1] for x in p.parts if x.startswith('#') and x.endswith('#')), '')
        return (rev, str(p))
    return max(fs, key=key)


def load_trials(session_path):
    return pd.read_parquet(newest(session_path / 'alf', '*trials.table.pqt'))


def binned_mean(times, values, onsets):
    """Mean samples in each trial-relative bin; NaN where no samples exist."""
    times = np.asarray(times, float); values = np.asarray(values, float).squeeze()
    good = np.isfinite(times) & np.isfinite(values)
    times, values = times[good], values[good]
    order = np.argsort(times); times, values = times[order], values[order]
    cs = np.r_[0., np.cumsum(values)]
    out = np.full((len(onsets), len(CENTERS)), np.nan, np.float32)
    for i, onset in enumerate(onsets):
        ix = np.searchsorted(times, onset + EDGES)
        n = np.diff(ix)
        sm = cs[ix[1:]] - cs[ix[:-1]]
        np.divide(sm, n, out=out[i], where=n > 0)
    return out


def fill_short_gaps(x):
    """Interpolate occasional empty temporal bins within each trial."""
    x = np.asarray(x, np.float32)
    q = np.arange(x.shape[1])
    for row in x:
        ok = np.isfinite(row)
        if ok.sum() >= 2:
            row[~ok] = np.interp(q[~ok], q[ok], row[ok])
    return x


def tertiles(x):
    """Session-wise equal-frequency discretization, preserving NaNs as invalid."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        raise ValueError('behavior contains no finite samples')
    q = np.quantile(finite, [1/3, 2/3])
    if q[0] == q[1]:
        # deterministic fallback for unusually constant traces
        q = np.quantile(finite + np.linspace(0, 1e-7, finite.size), [1/3, 2/3])
    return np.digitize(x, q, right=False).astype(np.int8), q.tolist()


def main():
    warnings.filterwarnings('ignore', category=UserWarning)
    one = ONE(mode='local', cache_dir=CACHE)
    one.load_cache(CACHE / 'Brainwidemap')
    release = pd.read_csv(RELEASE).drop(columns=['Unnamed: 0'], errors='ignore')
    import os
    if os.environ.get('MAX_SESSIONS'):
        e = release.eid.drop_duplicates().iloc[:int(os.environ['MAX_SESSIONS'])]
        release = release[release.eid.isin(e)]
    atlas = AllenAtlas().regions
    sessions, failures = [], []

    for si, (eid, probes) in enumerate(release.groupby('eid', sort=False), 1):
        try:
            spath = one.eid2path(eid)
            tr = load_trials(spath)
            needed = ['choice','probabilityLeft','feedbackType','feedback_times',
                      'stimOn_times','firstMovement_times']
            if any(c not in tr for c in needed):
                raise ValueError('missing required trial fields')
            valid = np.ones(len(tr), bool)
            for c in needed:
                valid &= np.isfinite(tr[c].to_numpy(float))
            rt = tr.firstMovement_times.to_numpy(float) - tr.stimOn_times.to_numpy(float)
            valid &= (rt >= .08) & (rt <= 2.00)
            valid &= np.isin(tr.choice.to_numpy(float), [-1, 1])
            valid &= np.isin(np.round(tr.probabilityLeft.to_numpy(float), 1), [.2, .5, .8])
            ti = np.flatnonzero(valid)
            if len(ti) < 2:
                raise ValueError('fewer than two valid trials')
            onsets = tr.stimOn_times.to_numpy(float)[ti]

            probe_data = []
            for r in probes.itertuples(index=False):
                ssl = SpikeSortingLoader(eid=eid, pname=r.probe_name, one=one)
                spikes, clusters, channels = ssl.load_spike_sorting()
                clusters = ssl.merge_clusters(spikes, clusters, channels)
                labels = np.asarray(clusters['label'])
                ids = np.asarray(clusters['cluster_id'], int)
                atlas_ids = np.asarray(clusters['atlas_id'])
                good = (labels == 1) & np.isfinite(atlas_ids) & (atlas_ids > 0)
                if not np.any(good):
                    continue
                mapped = atlas.remap(atlas_ids[good].astype(int), source_map='Allen', target_map='Beryl')
                acr = np.asarray(atlas.get(mapped)['acronym']).astype(str)
                keep = ~np.isin(acr, ['root','void','fiber tracts'])
                gids, acr = ids[good][keep], acr[keep]
                if len(gids):
                    probe_data.append((np.asarray(spikes.times), np.asarray(spikes.clusters), gids, acr))
            if not probe_data:
                raise ValueError('no well-isolated grey-matter neurons')

            # Region criterion: >=5 good neurons in this session.
            all_regions = np.concatenate([x[3] for x in probe_data])
            vals, cnt = np.unique(all_regions, return_counts=True)
            allowed = set(vals[cnt >= 5])
            n_neurons = sum(np.count_nonzero(np.isin(x[3], list(allowed))) for x in probe_data)
            if n_neurons == 0:
                raise ValueError('no region has >=5 well-isolated neurons')
            neural = np.zeros((len(ti), n_neurons, len(CENTERS)), dtype=np.uint16)
            region_names = []
            col = 0
            for st, sc, gids, acr in probe_data:
                use = np.isin(acr, list(allowed)); gids, acr = gids[use], acr[use]
                if not len(gids):
                    continue
                # Map sorter cluster IDs directly to output rows, then bin only
                # spikes in each two-second trial slice.  This is equivalent to
                # one histogram per neuron but avoids repeatedly scanning the
                # full (often tens-of-millions long) spike vector.
                max_id = int(max(np.max(sc), np.max(gids)))
                cmap = np.full(max_id + 1, -1, np.int32)
                cmap[gids] = np.arange(col, col + len(gids), dtype=np.int32)
                order = np.argsort(st) if np.any(np.diff(st) < 0) else None
                if order is not None:
                    st, sc = st[order], sc[order]
                for j, onset in enumerate(onsets):
                    lo, hi = np.searchsorted(st, [onset + OFF0, onset + OFF1])
                    cc = cmap[sc[lo:hi]]
                    ok = cc >= 0
                    if np.any(ok):
                        bb = np.floor((st[lo:hi][ok] - onset - OFF0) / DT).astype(np.int32)
                        inside = (bb >= 0) & (bb < len(CENTERS))
                        np.add.at(neural[j], (cc[ok][inside], bb[inside]), 1)
                region_names.extend(acr.tolist()); col += len(gids)

            wheel = one.load_object(eid, 'wheel', collection='alf')
            # Standard IBL wheel processing: interpolate to 1 kHz then 20-Hz
            # low-pass filtered velocity (brainbox.behavior.wheel).
            wpos, wt = interpolate_position(np.asarray(wheel.timestamps), np.asarray(wheel.position), freq=1000)
            ws, _ = velocity_filtered(wpos, 1000)
            wheel_b = fill_short_gaps(binned_mean(wt, np.abs(ws), onsets))

            alf = spath / 'alf'
            # Left camera is the high-frame-rate side view used for whisker-pad motion.
            mef = newest(alf, 'leftCamera.ROIMotionEnergy.npy')
            tf = newest(alf, '_ibl_leftCamera.times.npy')
            me, mt = np.load(mef, mmap_mode='r'), np.load(tf, mmap_mode='r')
            n = min(len(me), len(mt))
            whisk_b = fill_short_gaps(binned_mean(mt[:n], me[:n], onsets))
            good_trials = np.isfinite(wheel_b).all(1) & np.isfinite(whisk_b).all(1)
            if good_trials.sum() < 2:
                raise ValueError('fewer than two trials with complete behavior')
            ti, onsets = ti[good_trials], onsets[good_trials]
            neural, wheel_b, whisk_b = neural[good_trials], wheel_b[good_trials], whisk_b[good_trials]
            wheel_c, wheel_q = tertiles(wheel_b)
            whisk_c, whisk_q = tertiles(whisk_b)

            choice = (tr.choice.to_numpy(float)[ti] == -1).astype(np.int8)  # IBL: +1 left, -1 right
            pleft = np.round(tr.probabilityLeft.to_numpy(float)[ti], 1)
            prior = np.array([{.2:0,.5:1,.8:2}[float(x)] for x in pleft], np.int8)
            # Trial number within probability block, zero based.
            block_trial = np.zeros(len(tr), np.float32); k = 0
            probs = np.round(tr.probabilityLeft.to_numpy(float), 1)
            for j in range(len(tr)):
                if j == 0 or probs[j] != probs[j-1] or not np.isfinite(probs[j-1]): k = 0
                else: k += 1
                block_trial[j] = k
            sessions.append(dict(eid=str(eid), subject=str(probes.subject.iloc[0]),
                trials=ti, neural=neural, regions=np.asarray(region_names),
                choice=choice, prior=prior, block_trial=block_trial[ti],
                wheel=wheel_c, whisker=whisk_c, wheel_q=wheel_q, whisk_q=whisk_q))
            print(f'[{si}/{release.eid.nunique()}] {eid}: {len(ti)} trials, {n_neurons} neurons', flush=True)
        except Exception as exc:
            failures.append((str(eid), f'{type(exc).__name__}: {exc}'))
            print(f'[{si}] SKIP {eid}: {failures[-1][1]}', flush=True)

    # Across-session region criterion from the data paper.
    prevalence = {}
    for s in sessions:
        for r in np.unique(s['regions']): prevalence[r] = prevalence.get(r, 0) + 1
    allowed_global = {r for r,n in prevalence.items() if n >= 2}

    subjects = list(dict.fromkeys(s['subject'] for s in sessions))
    subj_map = {x:i for i,x in enumerate(subjects)}
    brain_regions = sorted(allowed_global)
    reg_map = {x:i for i,x in enumerate(brain_regions)}
    neural_out=[]; input_out=[]; output_out=[]; br_out=[]; kept=[]
    for s in sessions:
        nk = np.isin(s['regions'], list(allowed_global))
        if nk.sum() == 0: continue
        kept.append(s)
        br_out.append(np.array([reg_map[x] for x in s['regions'][nk]], np.int32))
        neural_out.append([s['neural'][j,nk] for j in range(len(s['trials']))])
        input_out.append([np.vstack((CENTERS.astype(np.float32),
                            np.full(len(CENTERS),s['block_trial'][j],np.float32)))
                          for j in range(len(s['trials']))])
        output_out.append([np.vstack((np.full(len(CENTERS),s['choice'][j],np.int8),
                             np.full(len(CENTERS),s['prior'][j],np.int8),
                             s['wheel'][j],s['whisker'][j]))
                           for j in range(len(s['trials']))])

    data = dict(neural=neural_out,input=input_out,output=output_out,
        subjects=subjects,subject_idx=np.array([subj_map[s['subject']] for s in kept],np.int32),
        brain_regions=brain_regions,brain_region_idx=br_out,
        input_names=['time since stimulus onset','trial number in block'],
        output_names=['choice','prior probability of left','wheel speed','whisker motion energy'],
        output_values=[['left','right'],['0.2','0.5','0.8'],['low','medium','high'],['low','medium','high']],
        metadata=dict(task_description='Decode choice, block prior, wheel speed, and whisker motion energy from brain-wide neural activity.',
          time_bin_size=20.0,temporal_alignment_event='visual stimulus onset (stimOn_times)',
          off_start=-0.5,off_end=1.5,neural_measure='spike counts per 20-ms bin',
          release='Brain-Wide Map; bwm_release.csv (459 curated sessions, 699 curated insertions)',
          unit_filter='RIGOR well-isolated units (clusters.label == 1); grey-matter Beryl regions; >=5 units/session and >=2 sessions/region',
          trial_filter='required events finite; first movement latency 0.08-2.00 s; valid binary choice/prior; complete wheel and whisker bins',
          behavior_binning='Mean absolute wheel angular velocity and left-camera ROI motion energy in each neural time bin; session-wise tertiles.',
          session_info=[dict(eid=s['eid'],source_trial_indices=s['trials'].tolist(),wheel_tertiles=s['wheel_q'],whisker_tertiles=s['whisk_q']) for s in kept],
          skipped_sessions=failures))
    with open(OUT,'wb') as f: pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {OUT}: {len(kept)} sessions, {sum(map(len,neural_out))} trials, {len(subjects)} subjects, {len(brain_regions)} regions')

if __name__ == '__main__': main()
