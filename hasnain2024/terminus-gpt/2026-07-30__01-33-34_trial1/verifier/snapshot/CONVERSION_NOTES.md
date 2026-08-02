# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
- **Date started**: 2026-07-30
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- .
- ..
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
| NeuralChoiceDecoding | code/ChoiceContextDecoding/NeuralChoiceDecoding.m | PROCESSING | Time-resolved choice decoding from neural data across sessions/trials |
| NeuralContextDecoding | code/ChoiceContextDecoding/NeuralContextDecoding.m | PROCESSING | Time-resolved context decoding from neural data |
| DLC_ChoiceDecoding | code/ChoiceContextDecoding/DLC_ChoiceDecoding.m | PROCESSING | Choice decoding from DLC-derived behavioral features |
| DLC_ContextDecoding | code/ChoiceContextDecoding/DLC_ContextDecoding.m | PROCESSING | Context decoding from DLC-derived behavioral features |
| getCDContext_NonStationary | code/CodingDirections/funcs/getCDContext_NonStationary.m | PROCESSING | Uses `params.alignEvent` and event times from `obj(...).bp.ev` to align trials and compute neural projections |
| getCodingDimensions_2afc | code/CodingDirections/funcs/getCodingDimensions_2afc.m | PROCESSING | Coding-dimension analysis anchored to alignment event and task epochs |
| alignSpikes | code/DataLoadingScripts/alignSpikes.m | PROCESSING | Aligns spike times to behavioral events (e.g. goCue/moveOnset) in the reference loading pipeline |

### Notes
- Session-specific loader scripts in `code/DataLoadingScripts/Recording and video/` enumerate animals/dates and build `meta` entries with `anm`, `date`, `datafn`, `probe`, and `datapth`, clarifying how sessions are selected in the reference code.
- Additional upstream reference code identified in `code/DataLoadingScripts`, especially `alignSpikes.m`, which likely implements the spike-alignment logic used before decoding.
- Reference code is primarily MATLAB.
- Relevant analysis modules identified: `ChoiceContextDecoding` and `CodingDirections`.
- Both DLC and neural decoders use session-level `params(sessix).trialid(...)` condition groupings, implying trial labels are precomputed and should be preserved.
- Behavioral feature groups explicitly include `tongue`, `paw`, and `motion_energy`, matching required decoder outputs.
- Decoding scripts use `rez.binSize = 75` ms and derive sample-step count with `rez.dt = floor(rez.binSize / (params(1).dt*1000))`, indicating a 75 ms analysis bin.
- Alignment logic in coding-direction code uses `params(sessix).alignEvent` and event times from `obj(sessix).bp.ev.(params(sessix).alignEvent)`, confirming event-based trial alignment in the reference pipeline.
- Need in subsequent steps to identify the upstream source files that instantiate `obj`, `params`, `kin`, and neural trial matrices, plus any neuron/trial curation rules.

- Additional alignment clue from `getCodingDimensions_2afc.m`: `aligntimes = mode(obj(sessix).bp.ev.goCue) - 2.5`, which strongly indicates go-cue-centered timing in the 2AFC analysis code.
- This supports using go cue as the temporal alignment event for the converted dataset, consistent with the task instructions.
- Additional Step 1 scan highlights:
```
code/Behavior/LickRaster_DR_WC.m:13:params.alignEvent          = 'goCue'; % 'jawOnset' 'goCue'  'moveOnset'  'firstLick'  'lastLick'
code/Behavior/LickRaster_DR_WC.m:15:% time warping only operates on neural data for now.
code/Behavior/LickRaster_DR_WC.m:18:params.nLicks              = 20; % number of post go cue licks to calculate median lick duration for and warp individual trials to
code/Behavior/LickRaster_DR_WC.m:39:params.traj_features = {{'tongue','left_tongue','right_tongue','jaw','trident','nose'},...
code/Behavior/LickRaster_DR_WC.m:40:    {'top_tongue','topleft_tongue','bottom_tongue','bottomleft_tongue','jaw','top_paw','bottom_paw','top_nostril','bottom_nostril'}};
code/Behavior/LickRaster_DR_WC.m:100:nTrials{1} = numel(params(sessix).trialid{3});
code/Behavior/LickRaster_DR_WC.m:101:nTrials{2} = numel(params(sessix).trialid{1});
code/Behavior/LickRaster_DR_WC.m:123:        trial = params(sessix).trialid{cond2use(cix)}(trix);
code/Behavior/LickRaster_DR_WC.m:157:xlabel('Time from go cue (s)')
code/Behavior/LickRaster_DR_WC.m:168:nTrials{1} = numel(params(sessix).trialid{4});
code/Behavior/LickRaster_DR_WC.m:169:nTrials{2} = numel(params(sessix).trialid{2});
code/Behavior/LickRaster_DR_WC.m:190:        trial = params(sessix).trialid{cond2use(cix)}(trix);
code/Behavior/LickRaster_DR_WC.m:224:xlabel('Time from go cue (s)')
code/Behavior/MCStim_AlternatingContext_LickRaster.m:17:params.alignEvent          = 'goCue'; % 'jawOnset' 'goCue'  'moveOnset'  'firstLick'  'lastLick'
code/Behavior/MCStim_AlternatingContext_LickRaster.m:37:params.alignEvent = 'goCue';
code/Behavior/MCStim_AlternatingContext_LickRaster.m:42:params.traj_features = {{'tongue','left_tongue','right_tongue','jaw','trident','nose'},...
code/Behavior/MCStim_AlternatingContext_LickRaster.m:43:    {'top_tongue','topleft_tongue','bottom_tongue','bottomleft_tongue','jaw'}};
code/Behavior/plotLickRaster.m:29:        trial = params(sessix).trialid{cond2use(cix)}(trix);
code/Behavior/plotLickRaster.m:77:xlabel('Time from go cue/water drop (s)')
code/Behavior/plotPerformanceAcrossTrials_CondSchematic.m:18:params.alignEvent          = 'goCue'; % 'jawOnset' 'goCue'  'moveOnset'  'firstLick'  'lastLick'
code/Behavior/plotPerformanceAcrossTrials_CondSchematic.m:20:% time warping only operates on neural data for now.
code/Behavior/plotPerformanceAcrossTrials_CondSchematic.m:23:params.nLicks              = 20; % number of post go cue licks to calculate median lick duration for and warp individual trials to
code/Behavior/plotPerformanceAcrossTrials_CondSchematic.m:42:params.traj_features = {{'tongue','left_tongue','right_tongue','jaw','trident','nose'},...
code/Behavior/plotPerformanceAcrossTrials_CondSchematic.m:43:    {'top_tongue','topleft_tongue','bottom_tongue','bottomleft_tongue','jaw','top_paw','bottom_paw','top_nostril','bottom_nostril'}};
code/Behavior/plotPerformance_byCondition.m:19:params.alignEvent          = 'goCue'; % 'jawOnset' 'goCue'  'moveOnset'  'firstLick'  'lastLick'
code/Behavior/plotPerformance_byCondition.m:21:% time warping only operates on neural data for now.
code/Behavior/plotPerformance_byCondition.m:24:params.nLicks              = 20; % number of post go cue licks to calculate median lick duration for and warp individual trials to
code/Behavior/plotPerformance_byCondition.m:43:params.traj_features = {{'tongue','left_tongue','right_tongue','jaw','trident','nose'},...
code/Behavior/plotPerformance_byCondition.m:44:    {'top_tongue','topleft_tongue','bottom_tongue','bottomleft_tongue','jaw','top_paw','bottom_paw','top_nostril','bottom_nostril'}};
code/ChoiceContextDecoding/DLC_ChoiceDecoding.m:23:% featGroups = {{'tongue'},...
code/ChoiceContextDecoding/DLC_ChoiceDecoding.m:26:%     {'paw'},...
code/ChoiceContextDecoding/DLC_ChoiceDecoding.m:27:%     {'motion_energy'}};
code/ChoiceContextDecoding/DLC_ChoiceDecoding.m:35:    % rez.feats2use = {'motion_energy'};
code/ChoiceContextDecoding/DLC_ChoiceDecoding.m:78:        trials_cond = params(sessix).trialid(cond2use);
code/ChoiceContextDecoding/DLC_ContextDecoding.m:24:% featGroups = {{'tongue'},...
code/ChoiceContextDecoding/DLC_ContextDecoding.m:27:%     {'paw'},...
code/ChoiceContextDecoding/DLC_ContextDecoding.m:28:%     {'motion_energy'}};
code/ChoiceContextDecoding/DLC_ContextDecoding.m:36:    % rez.feats2use = {'motion_energy'};
code/ChoiceContextDecoding/DLC_ContextDecoding.m:79:        trials_cond = params(sessix).trialid(cond2use);
code/ChoiceContextDecoding/NeuralChoiceDecoding.m:2:%% choice decoding from neural data
code/ChoiceContextDecoding/NeuralChoiceDecoding.m:30:    trials_cond = params(sessix).trialid(cond2use);
code/ChoiceContextDecoding/NeuralContextDecoding.m:24:% featGroups = {{'tongue'},...
code/ChoiceContextDecoding/NeuralContextDecoding.m:27:%     {'paw'},...
code/ChoiceContextDecoding/NeuralContextDecoding.m:28:%     {'motion_energy'}};
code/ChoiceContextDecoding/NeuralContextDecoding.m:36:    % rez.feats2use = {'motion_energy'};
code/ChoiceContextDecoding/NeuralContextDecoding.m:79:        trials_cond = params(sessix).trialid(cond2use);
code/CodingDirections/CDChoice_ROC.m:29:    trials = balanceAndSplitTrials(params(isess).trialid,p.cond2use,p.train,p.test);
code/CodingDirections/codingDirections_Bootstrap.m:53:            trialid = params(sessix).trialid;
code/CodingDirections/codingDirections_Bootstrap.m:54:            for icond = 1:numel(trialid)
code/CodingDirections/codingDirections_Bootstrap.m:63:                if numel(trialid{icond}) >= nTrials2Sample % if there are enough trials, sample without replacement
code/CodingDirections/codingDirections_Bootstrap.m:64:                    samp.trialid{ianm}{isess}{icond} = randsample(trialid{icond},nTrials2Sample,false);
code/CodingDirections/codingDirections_Bootstrap.m:67:                        samp.trialid{ianm}{isess}{icond} = randsample(trialid{icond},nTrials2Sample,true);
code/CodingDirections/codingDirections_Bootstrap.m:69:                        samp.trialid{ianm}{isess}{icond} = randsample(trialid{icond},numel(trialid{icond}),true);
code/CodingDirections/codingDirections_Bootstrap.m:94:            bootparams.trialid = samp.trialid{ianm}{isess};
code/CodingDirections/codingDirections_Bootstrap.m:104:            trialid = bootparams.trialid;
code/CodingDirections/codingDirections_Bootstrap.m:105:            for icond = 1:numel(trialid)
code/CodingDirections/codingDirections_Bootstrap.m:106:                samp.trialdat{icond} = cat(3,samp.trialdat{icond},trialdat(:,:,trialid{icond}));
code/CodingDirections/codingDirections_Bootstrap.m:107:                samp.me{icond} = cat(2,samp.me{icond},me_trialdat(:,trialid{icond}));
code/CodingDirections/codingDirections_Bootstrap.m:121:    bootparams.alignEvent = params(1).alignEvent;
code/CodingDirections/funcs/concatRezAcrossSessions.m:22:% allrez.selectivity_squared = zeros(dims(1),dims(2),numel(rez)); % (time,nCDs+1,sessions), nCDs+1 because its sum.sq.sel for each of the coding directions plus full neural pop's sumsqsel
code/CodingDirections/funcs/getCDContext_NonStationary.m:33:    rez(sessix).trialid = params(sessix).trialid;
code/CodingDirections/funcs/getCDContext_NonStationary.m:34:    rez(sessix).alignEvent = params(sessix).alignEvent;
code/CodingDirections/funcs/getCDContext_NonStationary.m:38:    aligntimes = obj(sessix).bp.ev.(params(sessix).alignEvent)(cell2mat(params(sessix).trialid(cond2use)'));
code/CodingDirections/funcs/getCDContext_NonStationary.m:39:    rez(sessix).ev.(params(sessix).alignEvent) = aligntimes;
code/CodingDirections/funcs/getCDContext_NonStationary.m:40:    rez(sessix).align = mode(aligntimes);
code/CodingDirections/funcs/getCDContext_NonStationary.m:73:        e1 = mode(rez(sessix).ev.(cd_epochs{1})) + cd_times{1}(1) - rez(sessix).align;
code/CodingDirections/funcs/getCDContext_NonStationary.m:74:        e2 = mode(rez(sessix).ev.(cd_epochs{1})) + cd_times{1}(2) - rez(sessix).align;
code/CodingDirections/funcs/getCDContext_NonStationary.m:88:    % --project neural population on CDs--
code/CodingDirections/funcs/getCDContext_NonStationary.m:124:    % full neural pop
code/CodingDirections/funcs/getCDContext_NonStationary.m:127:    rez(sessix).selectivity_squared(:,5) = sum(temp,2); % full neural pop
code/CodingDirections/funcs/getCodingDimensions_2afc.m:4:% cd_epochs = {'delay',params(1).alignEvent,params(1).alignEvent};
code/CodingDirections/funcs/getCodingDimensions_2afc.m:8:% cd_epochs = {'delay',params(1).alignEvent,params(1).alignEvent};
code/CodingDirections/funcs/getCodingDimensions_2afc.m:12:% cd_epochs = {params(1).alignEvent};
code/CodingDirections/funcs/getCodingDimensions_2afc.m:40:    rez(sessix).trialid = params(sessix).trialid;
code/CodingDirections/funcs/getCodingDimensions_2afc.m:41:    rez(sessix).alignEvent = params(sessix).alignEvent;
code/CodingDirections/funcs/getCodingDimensions_2afc.m:45:    aligntimes = mode(obj(sessix).bp.ev.goCue) - 2.5;
code/CodingDirections/funcs/getCodingDimensions_2afc.m:46:    % aligntimes = obj(sessix).bp.ev.(params(sessix).alignEvent)(cell2mat(params(sessix).trialid(cond2use)'));
code/CodingDirections/funcs/getCodingDimensions_2afc.m:47:    rez(sessix).ev.(params(sessix).alignEvent) = aligntimes;
code/CodingDirections/funcs/getCodingDimensions_2afc.m:48:    rez(sessix).align = mode(aligntimes);
code/CodingDirections/funcs/getCodingDimensions_2afc.m:59:            e1_start = mode(rez(sessix).ev.(ramp_epochs{1})) + ramp_times{1}(1) - rez(sessix).align;
code/CodingDirections/funcs/getCodingDimensions_2afc.m:60:            e1_stop = mode(rez(sessix).ev.(ramp_epochs{1})) + ramp_times{1}(2) - rez(sessix).align;
code/CodingDirections/funcs/getCodingDimensions_2afc.m:63:            e2_start = mode(rez(sessix).ev.(ramp_epochs{2})) + ramp_times{2}(1) - rez(sessix).align;
code/CodingDirections/funcs/getCodingDimensions_2afc.m:64:            e2_stop = mode(rez(sessix).ev.(ramp_epochs{2})) + ramp_times{2}(2) - rez(sessix).align;
code/CodingDirections/funcs/getCodingDimensions_2afc.m:71:        e1 = mode(rez(sessix).ev.(cd_epochs{ix})) + cd_times{ix}(1) - rez(sessix).align;
code/CodingDirections/funcs/getCodingDimensions_2afc.m:72:        e2 = mode(rez(sessix).ev.(cd_epochs{ix})) + cd_times{ix}(2) - rez(sessix).align;
code/CodingDirections/funcs/getCodingDimensions_2afc.m:97:    % --project neural population on CDs--
code/CodingDirections/funcs/getCodingDimensions_2afc.m:128:%     % full neural pop
code/CodingDirections/funcs/getCodingDimensions_2afc.m:131:%     rez(sessix).selectivity_squared(:,5) = sum(temp,2); % full neural pop
code/CodingDirections/funcs/getCodingDimensions_aw.m:15:    rez(sessix).trialid = params(sessix).trialid;
code/CodingDirections/funcs/getCodingDimensions_aw.m:16:    rez(sessix).alignEvent = params(sessix).alignEvent;
code/CodingDirections/funcs/getCodingDimensions_aw.m:20:    aligntimes = mode(obj(sessix).bp.ev.goCue) - 2.5;
code/CodingDirections/funcs/getCodingDimensions_aw.m:21:    % aligntimes = obj(sessix).bp.ev.(params(sessix).alignEvent)(cell2mat(params(sessix).trialid(cond2use)'));
code/CodingDirections/funcs/getCodingDimensions_aw.m:22:    rez(sessix).ev.(params(sessix).alignEvent) = aligntimes;
code/CodingDirections/funcs/getCodingDimensions_aw.m:23:    rez(sessix).align = mode(aligntimes);
code/CodingDirections/funcs/getCodingDimensions_aw.m:33:        e1 = mode(rez(sessix).ev.(cd_epochs{ix})) + cd_times{ix}(1) - rez(sessix).align;
code/CodingDirections/funcs/getCodingDimensions_aw.m:34:        e2 = mode(rez(sessix).ev.(cd_epochs{ix})) + cd_times{ix}(2) - rez(sessix).align;
code/CodingDirections/funcs/getCodingDimensions_aw.m:48:    % --project neural population on CDs--
code/CodingDirections/funcs/getCodingDimensions_aw.m:79:    % full neural pop
code/CodingDirections/funcs/getCodingDimensions_aw.m:82:    rez(sessix).selectivity_squared(:,5) = sum(temp,2); % full neural pop
code/CodingDirections/funcs/plotCDContext_singleTrials.m:3:trialStart = mode(obj(1).bp.ev.bitStart - rez_2afc(1).align);
code/CodingDirections/funcs/plotCDContext_singleTrials.m:4:sample = mode(obj(1).bp.ev.sample - rez_2afc(1).align);
code/CodingDirections/funcs/plotCDContext_singleTrials.m:6:    %     trix = sort(cell2mat(params(sessix).trialid([6 7])')); % 2afc and aw hits
code/CodingDirections/funcs/plotCDContext_singleTrials.m:24:    ylabel(['Time (s) from ' params(sessix).alignEvent])
code/CodingDirections/funcs/plotCDProj.m:1:function plotCDProj(allrez,obj,sav,plotmiss,plotaw,alignEvent,varargin)
code/CodingDirections/funcs/plotCDProj.m:13:sample = mode(obj(1).bp.ev.sample - obj(1).bp.ev.(alignEvent));
code/CodingDirections/funcs/plotCDProj.m:14:delay = mode(obj(1).bp.ev.delay - obj(1).bp.ev.(alignEvent));
code/CodingDirections/funcs/plotCDProj.m:56:    xlabel(['Time from ' alignEvent ' (s)'])
code/CodingDirections/funcs/plotSelectivity.m:16:sample = mode(rez(1).ev.sample) - mode(rez(1).align);
code/CodingDirections/funcs/plotSelectivity.m:17:delay  = mode(rez(1).ev.delay) - mode(rez(1).align);
code/CodingDirections/funcs/plotSelectivity.m:41:xlabel('Time (s) from go cue')
code/CodingDirections/funcs/plotSelectivityCorrMatrix.m:1:function plotSelectivityCorrMatrix(obj,dat,alignEv,sav)
code/CodingDirections/funcs/plotSelectivityCorrMatrix.m:4:sample = mode(obj.bp.ev.sample) - mode(obj.bp.ev.(alignEv));
code/CodingDirections/funcs/plotSelectivityCorrMatrix.m:5:delay  = mode(obj.bp.ev.delay) - mode(obj.bp.ev.(alignEv));
code/CodingDirections/funcs/plotSelectivityCorrMatrix.m:21:xlabel('Time (s) from go cue')
code/CodingDirections/funcs/plotSelectivityCorrMatrix.m:22:ylabel('Time (s) from go cue')
code/CodingDirections/funcs/plotSelectivityExplained.m:14:sample = mode(rez(1).ev.sample) - mode(rez(1).align);
code/CodingDirections/funcs/plotSelectivityExplained.m:15:delay  = mode(rez(1).ev.delay) - mode(rez(1).align);
code/CodingDirections/funcs/plotSelectivityExplained.m:37:xlabel('Time (s) from go cue')
code/DataLoadingScripts/alignSpikes.m:1:function obj = alignSpikes(obj,params,prbnum)
code/DataLoadingScripts/alignSpikes.m:3:if strcmp(params.alignEvent,'moveOnset')
```


---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `obj.meta.root` decodes to a source path string (representative session shows `E:\JEB\Experiments`), confirming `meta` stores provenance/session metadata.
- `obj.trials.bp` and `obj.trials.sglx` contain per-session bookkeeping fields such as `N`, `haveEphys`/`haveBP`, and file-number mappings, which are likely needed to identify valid trials shared across modalities.
- `obj.traj` is a 2x1 MATLAB cell and `obj.clu` is a 1x1 MATLAB cell, so both require dereferencing to expose trajectory and cluster/neuron content.
- Representative session `JEB11_2022-05-10` has 365 trials inferred from `obj.bp.ev` event arrays (`sample`, `delay`, `goCue`, `reward`, `lickL`, `lickR`).
- `obj.bp.ev.goCue` is a per-trial float64 array of shape `(1, 365)`, confirming go-cue timestamps are directly available for alignment.
- `obj/bp`, `obj/trials`, `obj/traj`, `obj/clu`, and `obj/meta` must be explored as HDF5 groups/datasets directly; they are not all simple arrays.
- Representative `data_structure_*.mat` includes MATLAB cell field `obj.clu`, likely containing cluster/neuron information that must be dereferenced from HDF5 references.
- Representative `motionEnergy_*.mat` session has `me.data` length 365 and session-level `moveThresh` = 9 (for JEB11 2022-05-10).
- Representative `motionEnergy_*.mat` file contains struct `me` with fields `data` and `moveThresh`.
- Representative `data_structure_*.mat` file has top-level struct `obj` with fields: `pth`, `bp`, `sglx`, `traj`, `trials`, `clu`, `ex`, `meta`.
- Mixed MATLAB formats detected: representative `data_structure_*.mat` files are MATLAB v7.3 / HDF5 (`scipy.io.loadmat` fails; use `h5py`), while representative `motionEnergy_*.mat` files are older MAT files (`h5py` fails; use `scipy.io.loadmat`).
- Initial HDF5 inspection of `data_structure_*.mat` shows MATLAB internal `#refs#` objects, so conversion will require dereferencing MATLAB object references rather than naive dataset loading.
Data are organized across multiple task-specific subdirectories under `data/`. Files are per-session MATLAB `.mat` files. In at least some task directories (including `RandomizedDelay_Ephys_Behavior`), sessions have paired files such as `data_structure_<subject>_<date>.mat` and `motionEnergy_<subject>_<date>.mat`.

### Dataset Size (from data files)

Step 2 summary:
```
Subjects: ['EKH1', 'EKH3', 'JEB11', 'JEB12', 'JEB13', 'JEB14', 'JEB15', 'JEB19', 'JEB23', 'JEB24', 'JEB6', 'JEB7', 'JGR2', 'JGR3', 'MAH13', 'MAH14', 'MAH20', 'MAH21']
Sessions per subject: {'EKH1': 1, 'EKH3': 1, 'JEB11': 2, 'JEB12': 2, 'JEB13': 5, 'JEB14': 4, 'JEB15': 4, 'JEB19': 4, 'JEB23': 8, 'JEB24': 10, 'JEB6': 1, 'JEB7': 2, 'JGR2': 2, 'JGR3': 1, 'MAH13': 22, 'MAH14': 24, 'MAH20': 9, 'MAH21': 18}
Unique sessions: 120
```


Step 2 directory breakdown:
```
DelayInhibition_BilatMC_Behavior mat_files 53 unique_sessions 53
Ephys_Behavior mat_files 50 unique_sessions 25
GoCueInhibition_BilatMC_Behavior mat_files 20 unique_sessions 20
RandomizedDelay_Ephys_Behavior mat_files 42 unique_sessions 22
```

Step 2 file inventory:
```
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH13_2022-07-29.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH13_2022-08-01.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH13_2022-08-02.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH13_2022-08-03.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH13_2022-08-04.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH13_2022-08-24.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH13_2022-09-13.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH13_2022-09-14.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH13_2022-09-19.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH13_2022-09-20.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH13_2022-09-21.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH13_2022-09-22.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH13_2022-09-26.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH13_2022-09-27.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH13_2022-09-28.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH13_2022-09-30.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH13_2022-11-09.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH13_2022-11-21.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH14_2022-08-15.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH14_2022-08-16.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH14_2022-08-17.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH14_2022-08-18.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH14_2022-08-22.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH14_2022-08-23.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH14_2022-08-24.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH14_2022-09-13.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH14_2022-09-14.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH14_2022-09-19.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH14_2022-09-20.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH14_2022-09-21.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH14_2022-09-22.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH14_2022-09-26.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH14_2022-09-27.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH14_2022-09-28.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH14_2022-09-30.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH14_2022-10-03.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH14_2022-11-04.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH14_2022-11-07.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH20_2023-09-13.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH20_2023-09-18.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH20_2023-09-25.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH20_2023-09-26.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH21_2023-09-07.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH21_2023-09-12.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH21_2023-09-13.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH21_2023-09-14.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH21_2023-09-18.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH21_2023-09-19.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH21_2023-09-20.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH21_2023-09-21.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH21_2023-09-22.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH21_2023-09-25.mat
data/DelayInhibition_BilatMC_Behavior/data_structure_MAH21_2023-09-26.mat
data/Ephys_Behavior/data_structure_EKH1_2021-08-07.mat
data/Ephys_Behavior/data_structure_EKH3_2021-08-11.mat
data/Ephys_Behavior/data_structure_JEB13_2022-09-13.mat
data/Ephys_Behavior/data_structure_JEB13_2022-09-14.mat
data/Ephys_Behavior/data_structure_JEB13_2022-09-21.mat
data/Ephys_Behavior/data_structure_JEB13_2022-09-24.mat
data/Ephys_Behavior/data_structure_JEB13_2022-09-25.mat
data/Ephys_Behavior/data_structure_JEB14_2022-08-22.mat
data/Ephys_Behavior/data_structure_JEB14_2022-08-23.mat
data/Ephys_Behavior/data_structure_JEB14_2022-08-24.mat
data/Ephys_Behavior/data_structure_JEB14_2022-08-25.mat
data/Ephys_Behavior/data_structure_JEB15_2022-07-26.mat
data/Ephys_Behavior/data_structure_JEB15_2022-07-27.mat
data/Ephys_Behavior/data_structure_JEB15_2022-07-28.mat
data/Ephys_Behavior/data_structure_JEB15_2022-07-29.mat
data/Ephys_Behavior/data_structure_JEB19_2023-04-18.mat
data/Ephys_Behavior/data_structure_JEB19_2023-04-19.mat
data/Ephys_Behavior/data_structure_JEB19_2023-04-20.mat
data/Ephys_Behavior/data_structure_JEB19_2023-04-21.mat
data/Ephys_Behavior/data_structure_JEB6_2021-04-18.mat
data/Ephys_Behavior/data_structure_JEB7_2021-04-29.mat
data/Ephys_Behavior/data_structure_JEB7_2021-04-30.mat
data/Ephys_Behavior/data_structure_JGR2_2021-11-16.mat
data/Ephys_Behavior/data_structure_JGR2_2021-11-17.mat
data/Ephys_Behavior/data_structure_JGR3_2021-11-18.mat
data/Ephys_Behavior/motionEnergy_EKH1_2021-08-07.mat
data/Ephys_Behavior/motionEnergy_EKH3_2021-08-11.mat
data/Ephys_Behavior/motionEnergy_JEB13_2022-09-13.mat
data/Ephys_Behavior/motionEnergy_JEB13_2022-09-14.mat
data/Ephys_Behavior/motionEnergy_JEB13_2022-09-21.mat
data/Ephys_Behavior/motionEnergy_JEB13_2022-09-24.mat
data/Ephys_Behavior/motionEnergy_JEB13_2022-09-25.mat
data/Ephys_Behavior/motionEnergy_JEB14_2022-08-22.mat
data/Ephys_Behavior/motionEnergy_JEB14_2022-08-23.mat
data/Ephys_Behavior/motionEnergy_JEB14_2022-08-24.mat
data/Ephys_Behavior/motionEnergy_JEB14_2022-08-25.mat
data/Ephys_Behavior/motionEnergy_JEB15_2022-07-26.mat
data/Ephys_Behavior/motionEnergy_JEB15_2022-07-27.mat
data/Ephys_Behavior/motionEnergy_JEB15_2022-07-28.mat
data/Ephys_Behavior/motionEnergy_JEB15_2022-07-29.mat
data/Ephys_Behavior/motionEnergy_JEB19_2023-04-18.mat
data/Ephys_Behavior/motionEnergy_JEB19_2023-04-19.mat
data/Ephys_Behavior/motionEnergy_JEB19_2023-04-20.mat
data/Ephys_Behavior/motionEnergy_JEB19_2023-04-21.mat
data/Ephys_Behavior/motionEnergy_JEB6_2021-04-18.mat
data/Ephys_Behavior/motionEnergy_JEB7_2021-04-29.mat
data/Ephys_Behavior/motionEnergy_JEB7_2021-04-30.mat
data/Ephys_Behavior/motionEnergy_JGR2_2021-11-16.mat
data/Ephys_Behavior/motionEnergy_JGR2_2021-11-17.mat
data/Ephys_Behavior/motionEnergy_JGR3_2021-11-18.mat
data/GoCueInhibition_BilatMC_Behavior/data_structure_MAH13_2022-12-12.mat
data/GoCueInhibition_BilatMC_Behavior/data_structure_MAH13_2022-12-14.mat
data/GoCueInhibition_BilatMC_Behavior/data_structure_MAH13_2022-12-16.mat
data/GoCueInhibition_BilatMC_Behavior/data_structure_MAH13_2022-12-22.mat
data/GoCueInhibition_BilatMC_Behavior/data_structure_MAH14_2022-12-14.mat
data/GoCueInhibition_BilatMC_Behavior/data_structure_MAH14_2022-12-15.mat
data/GoCueInhibition_BilatMC_Behavior/data_structure_MAH14_2022-12-16.mat
data/GoCueInhibition_BilatMC_Behavior/data_structure_MAH14_2022-12-20.mat
data/GoCueInhibition_BilatMC_Behavior/data_structure_MAH20_2023-09-28.mat
data/GoCueInhibition_BilatMC_Behavior/data_structure_MAH20_2023-09-29.mat
data/GoCueInhibition_BilatMC_Behavior/data_structure_MAH20_2023-10-05.mat
data/GoCueInhibition_BilatMC_Behavior/data_structure_MAH20_2023-10-09.mat
data/GoCueInhibition_BilatMC_Behavior/data_structure_MAH20_2023-10-19.mat
data/GoCueInhibition_BilatMC_Behavior/data_structure_MAH21_2023-09-28.mat
data/GoCueInhibition_BilatMC_Behavior/data_structure_MAH21_2023-10-02.mat
data/GoCueInhibition_BilatMC_Behavior/data_structure_MAH21_2023-10-03.mat
data/GoCueInhibition_BilatMC_Behavior/data_structure_MAH21_2023-10-05.mat
```

| Statistic | Value |
|-----------|-------|
| Neurons (total) | pending file inspection |
| Neurons / session | pending file inspection |
| Subjects | 18 |
| Sessions / subject | 120 total sessions across 18 subjects (distribution pending) |
| Trials (total) | pending file inspection |
| Trials / session | pending file inspection |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 522 (two-context); 845 (randomized delay); 1,651 (DR) | "In total, 522 units ... two-context ... randomized delay ... 845 units ... DR task ... 1,651 units" |
| Neurons / session | | |
| Subjects | 6 (two-context), 4 (randomized delay), 9 (DR) | "In 12 sessions from six mice, animals performed the two-context task... randomized delay task ... 19 sessions using four mice ... DR task ... 25 sessions using nine mice." |
| Sessions / subject | 12 sessions / 6 mice (two-context); 19 sessions / 4 mice (randomized delay); 25 sessions / 9 mice (DR) | "In 12 sessions from six mice... 19 sessions using four mice... 25 sessions using nine mice." |
| Trials (total) | | |
| Trials / session | | |
| Neural data time bin | | |
| Behavior data time bin | | |
| Reward rate | correct lick within 600 ms used in analysis | "we first calculated the percent of trials with a correct lick within 600 ms of the go cue/water drop" |
| <Task/behavior statistic 1> | | | | 
| <Task/behavior statistic 2> | | | |
| ... | | | | 


### Processing Details
- Reference code and methods consistently reference time relative to go cue (for example, plotting functions label x-axis as time from go cue, and coding-dimension helpers use `obj.bp.ev.goCue`).
- The paper methods describe behavioral timing relative to go cue/water drop and use the 1 s after go cue for some video-derived measurements, supporting go-cue-centered alignment.
- Early lick and ignore trials are omitted from analyses.
- Behavioral inclusion for behavioral analysis: sessions had at least 40 correct DR trials for each direction and 20 correct WC trials for each direction.
- Motion energy was averaged during the delay epoch for some analyses.
- Tongue visibility and other video-derived measures were quantified relative to the 1 s after go cue/water drop.
- For electrophysiology analyses, sessions were included only if they had at least 10 units.
- For many analyses, units with firing rates > 1 Hz were included; for subspace alignment and single-unit selectivity, only well-isolated single units with firing rates > 1 Hz were included.

### Curation Steps

**Neuron curation rules**:
- Spike sorting via JRCLUST and/or Kilosort 3 with manual curation in Phy 2.
- Well-isolated single units determined by manual inspection of ISI histogram, separation from other units, and stationarity across session.
- Multiunits: manually curated units with higher ISI violation rate.
- Include sessions only if at least 10 units.
- Include units with firing rates > 1 Hz in most analyses; only well-isolated single units > 1 Hz for subspace alignment and single-unit selectivity analyses.

**Trial curation rules**:
- Omit early lick and ignore trials from analyses.
- Behavioral analyses require sufficient correct trials per direction/context (40 DR and 20 WC correct trials per direction at the session level).

### Decoders Trained
| Decoded variable | Accuracy |
| | |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
- Resolved randomized-delay session-count discrepancy: reference loading scripts enumerate 19 sessions total (JEB11:2, JEB12:2, JEB23:7 active with one commented-out session, JEB24:8), matching the paper; therefore conversion should follow the explicit loader session list rather than all 22 raw files.
- Raw directory counts exceed paper-analyzed session counts for at least the randomized-delay subset (22 raw sessions vs 19 reported), so paper-consistent session filtering is required before conversion.
- No final exclusion list has yet been extracted; Step 5 must map/filter sessions using reference code and inclusion criteria rather than blindly converting all raw files.
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Raw vs paper session counts (randomized delay) | Raw `RandomizedDelay_Ephys_Behavior` has 22 sessions from 4 subjects | Data directory count = 22 sessions, 4 subjects | Paper reports randomized delay: 19 sessions using 4 mice | Three raw sessions are likely excluded by paper/reference processing; determine exclusion rule before conversion |
| Raw vs paper session counts (DR/ephys) | Raw `Ephys_Behavior` has 25 sessions from 10 subjects | Data directory count = 25 sessions, 10 subjects | Paper reports DR task: 25 sessions using 9 mice | At least one raw session/mouse is likely excluded in the paper; identify exact filtering/session list in Step 5 |
| WC vs DR labels | Figure scripts use `autowater` for WC and `~autowater` for DR | Data `bp` likely contains `autowater` trial flag; event arrays include goCue | Methods describe WC vs DR contexts | Map WC=autowater, DR=not autowater unless contradicted by deeper inspection |
| Alignment event | Coding/plotting functions use `obj.bp.ev.goCue` and label time from go cue | Representative data file has `obj.bp.ev.goCue` shape (1,365) | Methods describe analyses relative to go cue/water drop | Use go cue onset as temporal alignment event |
| Relevant session subsets | Data contain multiple directories/tasks | `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior` both contain motionEnergy and ephys sessions | Methods report separate two-context and randomized-delay ephys datasets | Candidate conversion should likely combine the paper-relevant ephys+behavior session subsets; verify exact subset in Step 5 |
| | | | | |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `obj.clu` / aligned spike data | neural | Bin aligned neural activity to common time bins (likely 75 ms) around go cue | `alignSpikes.m`, `NeuralChoiceDecoding.m`, `NeuralContextDecoding.m` | Need exact dereference/field path for spike times or binned spikes |
| Time from go cue | input[0] | Continuous per-timepoint variable relative to go cue onset | `obj.bp.ev.goCue`, coding-direction plotting/alignment functions | Required decoder input |
| `obj.bp.L`, `obj.bp.R` | output[0] lick direction | Map left=0, right=1 | Figure/task condition code, `NeuralChoiceDecoding.m` | Per-trial |
| `obj.bp.autowater` | output[1] behavioral context | Map WC/autowater=0, DR/~autowater=1 | Figure 1 scripts, methods | Per-trial |
| `obj.bp.hit`, `obj.bp.miss`/`obj.bp.no` | output[2] outcome | Map incorrect=0, correct=1; likely hit=1 and miss/no=0 after trial filtering | Methods and behavior flags | Per-trial |
| `obj.traj` tongue-related trajectories | output[3] tongue velocity | Compute velocity, then discretize per session at 50th percentile | DLC decoding scripts (`tongue`) | Need exact field mapping from `traj` |
| `obj.traj` paw-related trajectories | output[4] paw velocity | Compute velocity, then discretize per session at 50th percentile | DLC decoding scripts (`paw`) | Need exact field mapping from `traj` |
| `me.data` | output[5] motion energy | Use provided motion-energy time series, discretize per session at 50th percentile | DLC decoding scripts (`motion_energy`) | Session-level thresholding required by task |

### Key Decisions
- Figure 3 randomized-delay scripts explicitly build `randmeta` from `loadJEB11_ALMVideo`, `loadJEB12_ALMVideo`, `loadJEB23_ALMVideo`, and `loadJEB24_ALMVideo`, confirming the randomized-delay subset uses these four mice.
1. **Use go cue alignment**: Supported by data (`obj.bp.ev.goCue`), methods (go cue/water drop timing), and code (time from go cue labels and goCue-based alignment).
2. **Exclude early and ignore/no-go trials from primary analyses**: Methods explicitly omit early lick and ignore trials; outcome should be defined on the remaining valid trials.
3. **Use paper-consistent session filtering rather than all raw files**: Raw directory counts exceed paper-analyzed session counts, so Step 6 conversion must implement filtering to match reference analyses.
4. **Use 75 ms bins as initial default**: Explicitly present in decoding scripts; verify against neural loading/alignment code during implementation.
5. **Construct time-varying outputs for kinematics/motion energy when possible**: Better matches decoder requirements and reference time-resolved analyses.

### Planned Sanity Checks
- [ ] Check that `obj.bp.Ntrials` / event-array lengths match converted trial counts per session
- [ ] Check that go-cue-aligned time axis reproduces expected sample/delay offsets from reference code
- [ ] Check that context labels from `autowater` match trial-condition splits used in reference scripts

---

## Step 6: Script Development
**Status**: IN PROGRESS

[Implementation notes]
- Sample verification passed structural checks with no format errors/warnings, but kinematic and motion-energy outputs are still degenerate placeholders (all low), so Step 6 must continue before Step 7.
- Created initial `convert_data.py` skeleton that inspects representative sessions, reads mixed-format MAT files, and writes a placeholder pickle with session summaries.
- Next implementation task: decode inner `traj` feature names/time series and reconstruct per-trial binned neural matrices from `obj.clu`.


Code inefficiencies identified:
- Full conversion exposed another motion-energy edge case: some later `motionEnergy_*.mat` files contain `mat_struct` elements inside `me.data` rather than direct numeric arrays, so the loader must normalize multiple internal representations.
- Additional mixed-format issue discovered during full conversion: not all `data_structure_*.mat` files are HDF5/v7.3; later randomized-delay sessions include older MAT files, so the converter must support both formats for `data_structure` as well as `motionEnergy`.
- Current script now produces interim neural/input/output arrays for randomized-delay sample sessions, but tongue/paw/motion-energy outputs remain placeholder and full paper-consistent filtering is still missing.
- Need robust MATLAB v7.3 dereferencing utilities for nested cell/struct fields.

Code speedups added:
[Note]

- Interim sample verification now passes with no format errors/warnings.
- Sample output distributions are non-degenerate for all six outputs after implementing tongue/paw/motion-energy binning.


---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

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
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| lick_direction | 0.6273 | 0.6028 |
| behavioral_context | 0.7261 | 0.6861 |
| outcome | 0.5814 | 0.5312 |
| tongue_velocity | 0.8542 | 0.8556 |
| paw_velocity | 0.6064 | 0.5923 |
| motion_energy | 0.7812 | 0.7852 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 276052695 bytes
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
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
**Status**: NOT STARTED

### Checks Performed
1. [Check]: [Result]

### Issues Found and Resolved
- [Issue]: [Resolution]

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| lick_direction | 0.6399 | 0.6230 | above chance |
| behavioral_context | 0.7821 | 0.7191 | above chance |
| outcome | 0.6114 | 0.5797 | above chance |
| tongue_velocity | 0.8546 | 0.8560 | above chance |
| paw_velocity | 0.6068 | 0.5869 | above chance |
| motion_energy | 0.7744 | 0.7691 | above chance |

---

## Step 12: Critical Review 2
**Status**: NOT STARTED

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| | |

[Analysis of any low accuracies]

### Issues Found and Resolved
- [Issue]: [Resolution]

---

## Step 13: Documentation and Cleanup
**Status**: NOT STARTED

- [ ] README.md created
- [ ] cache/ folder created
- [ ] All files organized


Step 0 package verification:
- python3 import check passed
- numpy import passed
- torch import passed
