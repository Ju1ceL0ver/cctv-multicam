# Door student: speed, identity and side audit, 10 October 2026

The student is now fast enough on one short training clip: **65 ms per 0.08-s tick**, including decoding, inference, tracking and the original export. The measurement is before the additional export optimization below. Production replacement remains unverified: b2 scored 0.8400 mean directional event F1 versus SAM's 0.8846 on the same ten test stretches. A geometry-derived mask quantization correction was rejected after the cached paired CPU test: 0.82235 F1, entries 13/14/20 and exits 11/12/13. The test completed in 122.08 seconds; no network rerun was needed.

All experiments here use `runs/doorseg_b/step8604.pt`. Production SAM, CRM, staff model and their settings were not replaced. The broader acceptance criterion includes segmentation, identity contamination, staff/customer behavior, resource use and sustained live speed, as well as door events. The network's 5 future frames add **400 ms of context delay**; the 80-ms target is processing time/throughput, not total event notification latency.

## Evidence and rapid experiment protocol

Owner-completed **train** stretches of 18 September: 01405, 15375, 15569, and three disjoint parts of 18720 (`0:625`, `3500:4100`, `6500:7100`). These yield 3700 cached ticks. The network supplies its own previous mask. Cache contains scores, masks, displacement predictions, and teacher identities corrected by the owner's cuts/merges; owner event TEST was not used to fit or select settings. Cached association experiments decode no video and run no network. Side features are cached separately, so repeated door-policy evaluation takes seconds.

Held development interval 3500:4100 of 18720 is excluded from association fitting, including both endpoints of any candidate pair. This is within-day development validation, not an independent test day. Detection agreement measures masks against SAM, not independent human outlines. `wrong` counts **matched observations attributed to an ID initially assigned to somebody else**, not unique identity switches or unique incorrect merges.

Machine-readable reports are in `artifacts/doorseg_research/`. Windows caches and temporary reproductions are in `data/quality_backups/doorseg_quick/`; no model or image data is embedded in Git.

## Speed changes measured separately

| Stage / comparison | Measurement | Interpretation |
|---|---:|---|
| Original b2 full evaluation | 160.13 ms/tick | 10 test stretches, historical complete pass |
| Original network after limiting CPU threads | median 36.81 ms, p90 39.25 | four training moments; not production latency |
| Vectorized postprocessing | median 35.72 ms | masks equal; maximum offset error 4.8e-7 on four moments |
| Fused network, channels first, BF16 | 23.52 ms on one moment | same four detections, mask IoU 1; larger equivalence check remains |
| Old / cached CPU channel assembly | 50.09 / 39.06 ms | exact 17-channel integer equality |
| GPU channel assembly, warmed cache | 2.38 ms | exact integer equality; bounded eleven-frame cache |
| Fast complete pass, train 01405 | 65.0 ms/tick, 625 ticks | decoding + model + tracking + export; one light clip |
| Old crop/pack / new complete compressed export | 3.742 / 0.242 s | 200 crowded training ticks, 706 masks; old timing excludes final NPZ compression |

`door_seg_fast.py` removes per-person CUDA transfers and reductions, batches mask cropping and offset averaging, reuses decoded BGR/grayscale context and assembles differences on the GPU. `door_seg_run.py --fast` limits CPU threads to two, fuses convolution/normalization layers and preserves channels-first BF16. It does not alter the default live counter.

The export now determines bounds on the 320×180 mask and expands only that crop using the **global** nearest-neighbor sampling coordinates. Round-trip tests verify identical images at 1280×720, 1920×1080 and a noninteger scale. Previous union is formed at mask resolution and expanded once.

Channels-last was **rejected for the current release candidate**: despite speed improvement, one of four sampled moments changed detection count. FP16 was not faster than the fused BF16 sample. TensorRT for this custom 17-channel model and displacement head has not yet been measured; results for the old heavy teacher do not establish performance for this student.

## Association experiments

| Variant, held 600 training ticks | Continued / opportunities | Broken | Wrong-ID observations |
|---|---:|---:|---:|
| b2 cascade | 3996 / 4090 | 94 | 77 |
| learned association, probability >= 0.7 | 3978 / 4090 | 112 | 0 |
| learned association, >= 0.8 | 3977 / 4090 | 113 | 0 |
| learned association, >= 0.9 | 3973 / 4090 | 117 | 0 |

The association classifier is a small **48-tree random forest**, not a new neural fingerprint head. Eleven observed features include forward/backward displacement errors, shifted mask overlaps, size, detection confidence and a cheap three-band masked HSV descriptor. Owner identities supply only TRAIN targets. Hard motion and six-tick gap limits still apply. The classifier fits 12174 pairs and validates on 7435; at 0.7 it accepts 4047 true pairs, one false pair, and misses nine positive pairs. Pair metrics and full tracking metrics measure different things.

Stricter ordinary costs did not solve fragmentation: on the first 400 training moments, max-cost 0.65 increased breaks 43→45; stronger ambiguity increased them 43→44. Weak-score recovery (ByteTrack-style, no births from weak detections) slightly increases matched masks but worsens identity contamination on the long training excerpt. Lowering the birth score to 0.3 increases false detections and breaks. These changes are not selected for production.

Masked color reduces wrong-ID observations on the long excerpt 145→35 at weight 0.4, but raises breaks 273→281. On the other excerpts the effect is mixed; it also adds one false exit on the sampled long training event test. Therefore “stricter appearance glue” is not a general improvement by itself.

The learned association likewise does **not** improve sampled train events: entries remain 2/2/5 and exits worsen 3/3/3→2/2/3. It is retained as an experimental tool, not a production candidate solely because its pair accuracy is high.

## Side and training-data defects

**Confirmed data-preparation bug:** `Source.day = name[:8]` parses `door_20260918_…` as `door_202`. There are 12 actual 18 September camera-1 background files and zero matching the malformed date; training used the all-days fallback for those sources. `source_day()` now extracts the recording date for both ordinary segmentation windows and door stretches. Existing weights have not been retrained after this fix. This can reduce a training/inference mismatch; the actual accuracy benefit still needs measurement.

**Student mask quantization:** door depth is the deepest mask pixel beyond the owner's line. A stride-4 binary mask expanded by nearest neighbor has a cell-scale uncertainty at its boundary. The existing immediate inside override treats small positive depth as certain. At one missed training entry, all 25 preceding observations voted inside although the median depth was only 2.83 px. An extra required inside margin is derived from the cell projected onto the line's normal: `4*(abs(dx)+abs(dy))/hypot(dx,dy)`. This is a geometry-derived correction, not a threshold chosen on TEST.

On the three disjoint excerpts of 18720, the cached correction changes b2 entries **2/2/5→3/3/5**, preserving exits **3/3/3**. With teacher identities it changes entries **2/2/5→4/4/5**. The identity control resets at every excerpt boundary; it is an optimistic diagnostic that omits unmatched detections, not deployable tracking. The cached test lacks RGB classifier features previously skipped by the original immediate inside override. `door_seg_side.Live` recomputed those features in the uncertain band in the final evaluation. That fixed policy lost one true entry on TEST (13 versus b2's 14); it is rejected, and the test is not used to choose another margin.

Another missed entry has only 3 matched high-confidence observations in the six seconds before it, versus 25 teacher observations. This can reflect missed/incorrectly shaped detections or unreliable teacher identity matching. It cannot be diagnosed as “just tracker glue” from event F1. Visual cases are retained on Windows in `side_cases.jpg`; owner annotation is not changed automatically.

The current masks are stored as binary teacher labels. Teacher-supervised training is distillation in the broad sense, but this data path **does not contain SAM probability/logit targets**. Soft-mask/logit distillation requires retaining such outputs; adding a residual loss to binary labels does not make them soft. Boundary losses, visibility/occlusion prediction and temporal memory remain hypotheses until trained and tested separately. Larger displacement horizons can help gaps, but must be trained for their own horizons; multiplying the one-tick head across unseen frames is unreliable.

## Staff/customer and deployment limits

The current role model has consumed the old annotated days. Its reported fit accuracy is not a blind test. Historical grouped validation shows staff recall is substantially weaker than customer recall. Current live feedback has 11 usable people / 88 stored views on 9 October, including four staff; 8 October has nine answered groups but no usable saved views. There is only one new usable day. Refitting on it cannot demonstrate a gain on an independent new live day.

Role inference remains asynchronous on CPU. The view bank keeps eight high-scoring views but does not enforce temporal diversity; adjacent near-identical views need a measured person-level comparison before changing this. The role code and artifacts being worked on in parallel were left intact.

The `doorseg_c_background` candidate started from finite step8604 weights and optimizer, with the baseline validation recorded. Its only training change is the recording-date/background repair. Colab training started at 05:14 UTC on 10 October and the VM disappeared by 05:48 UTC. The latest verified Windows checkpoint is step8996; the Windows GPU remains reserved for the live door during shop hours.

## Colab continuation, 10 October

The owner made Colab GPU quota available. Session `doorseg-c` has a Tesla T4 with 15360 MiB. A100 and L4 allocation were rejected. The complete unchanged data manifest is **148 training sources, six validation sources, 1216 files, 57.7477 GB**; transfer completed at 05:12:12 UTC. The exact source composition and finite weights/optimizer gate passed at 05:14:21 UTC; the trainer restored step8604 and has logged step8700 with finite losses. New validation and quality improvement have not yet been measured. Small labels/backgrounds/checkpoint travel as one 1.099-GB stored ZIP to avoid hundreds of HTTP round trips.

At 05:48:43 UTC the backend reported the session lost (404/401), after a successful kernel ping at 05:45:41. No active VM remained. The last observed training log reached step9200 without a new validation. Windows checkpoint step8996 was rechecked on CPU: finite model and optimizer, SHA256 `0a2b537b7a94b49977b74b095891f5bfe7e4de8c83601c2b6705c0e9265d0b65`, 140327101 bytes. Five pieces of a newer upload are incomplete and were not used. A new manifest and stored bundle preserve the original TRAIN/VAL and labels while replacing only run state with the verified resume. Two T4 allocation attempts returned HTTP503; L4 was rejected for quota/entitlement. The old keepalive was stopped after confirming VM loss. Recovery waits for a GPU allocation; there is no new trainer or quality result. Night-only Windows fallback is permitted after ownership/process checks, without concurrent Colab training.

The VM uses the same torch 2.13.0+cu130, torchvision 0.28.0+cu130 and Ultralytics 8.4.108 as Windows. Installation exposed overlapping CUDA-12/CUDA-13 cuDNN packages; a real convolution failed with `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`. Removing both distributions and installing only torch's required `nvidia-cudnn-cu13==9.20.0.48` fixed it. A six-example BF16 forward/backward from the finite checkpoint passed: loss 8.265064, peak allocation 5.4768 GB. This is compatibility evidence, not speed or accuracy evidence for production.

`door_seg_colab.py` verifies the source manifest, initial checkpoint SHA256, successful step and finite model/optimizer before starting the unchanged batch-six training recipe (lr 1e-4, 400 validation moments, 2000 successful steps between checks). Two data workers fit the Colab CPU. New checkpoints and progress are backed up every two minutes; large checkpoints use 24-MiB HTTP pieces. `door_seg_colab_restore.py` on Windows verifies the completed SHA256 and finite model/optimizer, refuses a rollback over a newer local checkpoint, and restores atomically. The Colab sender waits for receiver acknowledgement. `COLAB_ACTIVE.json` blocks a duplicate Windows supervisor; ownership clears only after the final backup succeeds.

The existing keepalive mechanism was found in `tennis-pose/tools/colab_keepalive.sh` and that project's AGENTS.md, lines 273 and 383: background computation does not keep the notebook kernel active, and overlapping `colab exec` commands have hung. The Mac now runs a kernel/job probe every three minutes under `caffeinate`. All executions use the serialized wrapper:

```sh
python3 tools/doorseg_colab_control.py exec --file /tmp/probe.py
```

The pinger stops when the runner exits; restart it after recovering a failed runner. Its log is `data/logs/colab_doorseg-c_keepalive.log`. The existing task heartbeat now checks every ten minutes throughout the day, stays quiet during healthy transfer/training, and retains the wider research/production acceptance requirements. Release the owned Colab session only after Windows checkpoint restoration is verified.

Storage preflight found another constraint before training: packed-mask mmap
files need **11.1759 GB**, beyond the 57.7477-GB compressed/video manifest.
Projected disk reserve was only 1.09 GB. Cache cleaning and removing unused
base-image TensorFlow/JAX reclaimed only a few MB because their base filesystem
layers remain allocated. The torch/vision/Ultralytics recipe is unchanged.

`door_seg_colab_cache.py` placed 3.2212 GB of identical packed buffers from
18 sources on `/dev/shm/doorseg_masks`, with adjacent NPY symlinks that the
existing Masks reader accepts. Each buffer is compared to its original NPZ
array; cache is bounded to 3 GiB total and leaves at least 6 GiB available RAM.
After preparation, available RAM was 8.54 GB; projected final disk reserve
was 3.90 GB. Do not remove this shared cache while the Colab trainer is active.
The runner now counts remaining dataset files, uncompressed mmap and a 3-GiB
checkpoint reserve. It attempts the bounded shared cache if necessary and
fails before training if space still does not fit. Repeated BF16 batch-six
forward/backward passed with the identical 8.265064 loss and 5.4768-GB GPU
peak, without an optimizer update. This fixes storage, not model quality.

A full release still requires: a successful selected-policy comparison on the frozen paired event cohort, clean and crowded end-to-end speed including side/role queues, mask equivalence after fusion, identity quality, then live shadow/integration checks preserving CRM IDs and format. GPU tests wait for the night window; CPU cache tests can run during the day. A fast isolated frame and a training-only event gain do not satisfy these conditions.

## Kaggle continuation on two GPUs, 10 October

The owner redirected recovery to Kaggle and approved using both GPUs for one
student with a larger global batch. A completed private probe confirmed two
15-GB Tesla T4s, driver580.178.04, BF16 support and 32.07GB available RAM.
Working output storage is only20.94GB; the full frozen manifest and packed-mask
mmap live on /kaggle/temp (over1TB free), with model/log artifacts in working.
Kernel `juice0lover/cctv-doorseg-c-background` version4 is running transfer.
Version2 failed before transfer because the base Python lacked ensurepip;
virtualenv with bundled pip seeds fixes environment creation. Version3 lost
CUDA access when isolation removed the mounted NVIDIA driver library path;
version4 preserves it and preflights both allocated GPUs. At06:47:47UTC the
exact torch2.13.0+cu130/vision0.28.0+cu130/Ultralytics8.4.108 passed finite BF16
convolution/backward on both T4s. At06:51:50–53UTC the actual full model
passed two-rank BF16 backward with six examples per GPU, global12. Losses
were8.5238256/10.3039169; averaged gradient norms matched at23.7682285 before
clip10. Peak allocation13.4544GB on each T4; retain localbatch6 pending real
training stability. No optimizer update or new checkpoint/validation occurred.
Full unchanged transfer continues; training is not yet confirmed. Credentials,
bootstrap and kernel are private.

Update08:30UTC: the full1217-file/57,747,616,016-byte transfer finished at
08:21:17UTC. Exact148TRAIN/6VAL, checkpoint/optimizer finite and SHA gates
passed at08:22:23UTC. DDP resumed from8996 on bothT4s/global12 with two
workers per rank. New successful steps9000(08:24:22UTC) and9100(08:28:40UTC)
are in log.jsonl with global_batch12 and finite losses; older Colab steps
are separate history. No new validation/quality/plateau result yet.
The latest acknowledged Windows checkpoint remains8996 pending the next
ten-minute save and receiver ACK. Stop17:43UTC leaves one hour for final
validation and backup within the12-hour allocation. Do not restart this
healthy training job because of the known nonfatal sitecustomize/DDP warnings.

First new Kaggle checkpoint acknowledged on Windows at15:35:22UTC+7:
step9196, finite model AND optimizer, SHA256
`bcf8d8bd28276abef00a5f02dbb29186b92d0fc3a1103b69f2b05b19659874df`.
A verified immutable copy lives in `data/quality_backups/doorseg_c_kaggle/`.
Recovery must use the latest verified backup and refresh manifest/bundle;
the old8996 snapshot is no longer the default recovery point. Fresh training
log reached9300 at08:37:11UTC without a new validation or numerical rejection.

First spaced validation completed09:43UTC atstep10604 on the unchanged
400-moment23September proxy cohort. F1 .9307 versus baseline .9345
(−0.38percentage points), found .9079 versus .9148; maskIoU .9013 versus
.8974 and links .9783 versus .9774. This is a mixed result, not overall
improvement or production acceptance. The reported59.1ms is T4 proxy
inference, not full RTX3060 door processing. Training continued to10700;
one new evaluation is insufficient for plateau. No labels or thresholds changed.
Windows CPU receiver verified finite model AND optimizer for last10604 and
evalstep10604; SHA-checked immutable copies and metadata are in
`data/quality_backups/doorseg_c_kaggle/step10604_verified.json`.
Last SHA `c63d05a9105013956856ffdfcef6996e027cd8c5eb3ccd48613d940143c98f9f`;
eval SHA `94c24af82f9677d17e9ef209edeb8c8b8e103c2fcb3015f2be06a20f1bc637d4`.
Recovery must check latest finite last10604 or newer, rather than the old
9196/8996 bundle. The healthy job and night CPU waiter remain running.

At10:10UTC the local Kaggle CLI lost authorization while fresh training
steps11200 and Windows finite restore11064 continued. The installed SDK's
expiry check uses now−30minutes, treating an expired access token as valid
for30minutes. Explicitly refreshing the existing OAuth credentials via
`KaggleCredentials.refresh_access_token()` restored RUNNING status without
changing the live kernel. Never interpret this CLI failure alone as VM loss;
check fresh mirrored progress and backup. No token was logged or SDK patched.

Second spaced proxy validation at11:22UTC/step12604: found .9001,
precision .9603, F1 .9292, maskIoU .9017, links .9803,915links/1021truth.
Against baseline8604: F1 −0.53percentage points, maskIoU +0.43points,
links +0.29points. Recall fell despite better precision/masks/links; overall
improvement remains unproven.57.8ms is T4 proxy inference only. Two stale
spaced evaluations followed the baseline; plateau requires three. The
healthy job continued to12700 with no STOP. No labels/thresholds changed.
Windows CPU receiver verified finite model AND optimizer12604; SHA-checked
immutable last/eval copies andstep12604_verified.json are in the existing
Kaggle backup directory. Recovery must check latest finite12604 or newer.
Last SHA `ca1c017baee7d396a82d36ea92cbaaa45a9af060731e4840527d704b5285a178`;
eval SHA `233dca6e8d8b90a1aeedbe73a7285f80559d6c52c0323673c85af0438813f371`.

Third spaced evaluation completed13:00UTC atstep14604 on the unchanged
400-moment23September proxy: found .9109, precision .9718, F1 .9403,
maskIoU .9052, links .9838,926links/1021truth,56.9ms T4 proxy.
Against baseline8604: F1 +0.58percentage points, maskIoU +0.78points,
links +0.64points. Found remains0.39points below baseline .9148; higher
precision drives the F1 gain. This measured proxy improvement does not
establish better door events, role quality or full RTX3060 deployment speed.
The gain .0058 exceeds plateau min_delta .005, so14604 becomes the new
best and resets the two stale evaluations. Training continued to14700;
no STOP or recipe/label/threshold change. The existing DDP job remains RUNNING.
Windows CPU receiver acknowledged finite model AND optimizer last14604 at
20:03:36UTC+7 and evalstep14604 at20:05:17. SHA-checked immutable copies
andstep14604_verified.json are saved in the existing Kaggle backup directory.
Last SHA `b43f0b69c08c1f20eed1cbc4b0bfeabaa8b527eb12c2ce6c87dae032c494112f`;
eval SHA `ce9b0a8498d3e69494d528c66498657e4103337929f4de9859489a3459154c66`.
Recovery must check latest finite14604 or newer, preserving weights and
optimizer rather than restoring a stale12604/8996 bundle.

At13:12UTC, step14862 was rejected by the joint DDP finite gate before
any optimizer update. BN buffers rolled back; each rank retained its actual
batch and rank0 saved a finite before_rejected checkpoint. Retrying the
same batch in FP32 passed, so later steps stay in FP32. Fresh14900–15100
losses are finite, with no STOP and the same RUNNING kernel. Do not reset
precision to BF16 or restart the healthy trainer. This is a numerical incident,
not a new quality measurement. Its specific cause and failing rank/loss versus
gradient remain unresolved; the combined gate log does not distinguish them.
The CPU receiver verified before_rejected_14862.pt weights AND optimizer,
SHA `52923c7d057baa1ad92ebd2a98d1a120e5c61501ecfc04239fef2bf1f86cbbab`;
a SHA-checked immutable copy is retained in the Kaggle backup directory.
The latest normal ACK14834 still predates the incident and is marked BF16;
recovery must verify a current checkpoint and its saved precision.
The sender intentionally skips rejected batch files. They remain on the VM;
a version4 private-output request returned no matching files while RUNNING.
Download both rejected_14862_rank0/1.pt and before_rejected on completion,
then diagnose them separately. No daytime Windows GPU reproduction occurred.

Update13:31UTC: the first post-retry FP32 last15191 was acknowledged on
Windows at20:24:20UTC+7 with finite model AND optimizer, SHA
`09d1cb7cad79a0bd08a7e439f08cce77236a405e6d8e4c7cdc3d6d0685cf0efc`.
Its FP32 metadata was inspected without torch via zip/pickletools; an
immutable SHA-verified copy andstep15191_verified.json are retained.
Recovery must use latest finite15191 or newer and preserve FP32. Five older
orphan upload parts without a completion manifest remain incomplete and
must not be used. Fresh15500 losses are finite with no further rejection.

`door_seg_ddp.py` runs one student via NCCL, six examples per GPU, twelve total.
DDP averages losses with the legacy local batch-six scale; lr1e-4 is retained,
BN remains local, and the finite step8996 weights AND optimizer are restored.
The source cohort, labels, offset heads and400-moment validation remain fixed.
Both ranks finish backward and jointly approve finite loss/gradient before any
update; rejection restores BN, captures each rank's batch and a finite state,
and permits one FP32 retry. Real CPU two-rank tests verify gradient averaging
and that NaN on either rank leaves both models/Adam/BN unchanged.56 door tests
pass; these are protocol checks, not evidence of CUDA training speed or quality.

The foreground committed kernel waits for its runner, with training ending11h
from kernel start to reserve one hour within the12h allocation for final
validation/backup/output. Large checkpoint pieces and Windows finite restore
ACK are reused. Cloud ownership remains active through preparation/failures;
no concurrent Windows/Colab trainer is allowed. The monitor follows Kaggle
status/logs and the existing Windows run, and no longer allocates Colab. The
fingerprint/ConvGRU work remains separate. Production acceptance is unchanged.

## Autonomous night work, 10 October

The owner authorized unattended RTX3060 research from21:00: latest finite
Windows training recovery if cloud C ends, or a separate speed/accuracy
experiment while Kaggle is healthy. The existing ten-minute heartbeat now
continues implementation at night, beyond monitoring. No duplicate C trainer,
no fitting on TEST, and production acceptance remains unchanged.

`door_seg_night_probe.py --wait` is actually waiting on Windows under venv_sam3,
without importing torch during the day. Its CPU preflight verified completed
TRAIN01405/15375 and refused daytime GPU use. It runs24 same-input comparisons
and120 fast processing ticks per source, with an180-second research budget,
night boundaries, foreign-process guards and initial GPU utilization gate.
Fixed B step8604 weights; only new experiment outputs are written.
`data/quality_backups/doorseg_night_20261010/report.json` will contain channel/
mask/offset comparisons and stage timings including decode and compressed
export. Guard overhead is excluded and separately recorded. Side/role queues
need their own profile. No new GPU speed/equivalence claim is made yet.
Two CPU tests cover night/foreign-process/Windows-wrapper guards and refusal
when an owner TRAIN source becomes TEST. Windows venv's redirector has its
own python PID: ignore own ancestors, never an unrelated research job.

After the short profile, implement and measure an improvement to the actual
slow stage on short TRAIN excerpts; if speed is sufficient, work on concrete
missed detections or fragmentation. Do not spend a full event TEST pass merely
on profiling. A bounded Mac caffeinate runs until11October09:40 to support
local heartbeat execution; the app and machine must remain available.

## Primary sources and relevance to this case

The owner also requested implementing the fingerprint and ConvGRU heads while
the dataset transfers. They are staged separately, with ordered TRAIN unroll
and checkpoint/context guards; see [DOOR_SEG_TEMPORAL.md](DOOR_SEG_TEMPORAL.md).
Four real TRAIN frames passed BF16 backward without updating weights. Initialized
heads preserved original predictions on a short frame check. Heads plus mask
pool averaged 3.139 ms on T4; full RTX3060 pipeline and trained quality remain
unverified. The existing C recipe has not acquired these heads.

- [ByteTrack](https://arxiv.org/abs/2110.06864): recover existing tracks with low-confidence boxes rather than discarding all low scores. Implemented as a bounded experiment; measured contamination here prevents adopting it indiscriminately.
- [OC-SORT](https://arxiv.org/abs/2203.14360): use observed motion and correct state after occlusion. Supports retaining measured motion and rejecting naive extrapolation of one-tick displacement heads over long gaps.
- [Deep SORT](https://arxiv.org/abs/1703.07402) and [Deep OC-SORT](https://arxiv.org/abs/2302.11813): appearance is an association cue with motion constraints, not sufficient evidence for unrestricted long-gap identity merges. Our appearance tests preserve hard physical limits.
- [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) and [NVIDIA sync-free code](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/sync-free-code.html): CPU/GPU transfers and data-dependent host operations can serialize an otherwise fast network. Bulk postprocessing and channel caching address measured bottlenecks here.
- [TensorRT optimization](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html): benchmark the actual precision, shapes, transfer and workload. Exporting a different model is not evidence for this student's deployment speed.

## Parallel visibility/completion work

Sections 66–67 of AGENTS also record a separate owner-authorized `amodal_lab` experiment. Its mask-only spatial visibility classifier scored 32/41 (78.05%), balanced accuracy 80.39%, on validation day 18 September; its recorded MPS network median is 0.993 ms. Rich RGB completion scored hidden IoU 0.5068 across 24×13 synthetic occlusions, versus 0.4731 without RGB. Those are the other experiment's recorded results, not Windows measurements made here. Real hidden-leg recovery and independent event gains are not established. This is useful for a future visibility signal, but does not justify filling imagined feet into the live door geometry. These files, weights and owner annotations were preserved.

Fourth fixed400-moment proxy evaluation at14:07UTC/step16604 after FP32 retry:
found .9128, precision .9668, F1 .9390, maskIoU .9036, links .9849,
927links/1021truth. This is one stale evaluation after best14604, not plateau.
Both last/eval16604 have finite model AND optimizer CPU receiver ACKs,
SHA-verified immutable snapshots and saved FP32 precision. Recovery uses
latest finite16604 or newer, preserving FP32. Best proxy remains14604.

While the night GPU probe correctly yielded to live/foreign processes,
120-tick CPU TRAIN replays on01405/15375 measured actual side feature jobs
and view-bank selection. A separate experimental door_seg_queue_fast.Live
keeps eight immutable RGB frame references and serializes preparation to
avoid duplicate work between side threads. It prepared each of240 frames
once, with329 cache hits across569 feature jobs. Feature vectors match an
uncached fresh reference bit-for-bit; side votes, IDs/stats and kept role
JPEG inputs also match. The legacy dense feature hash differs; its cause
has not been isolated. No production side/role code was changed.
Mean feature service fresh→fast was13.81→9.94ms light and19.52→18.27ms
dense under shared CPU load. These sequential timings are provisional;
repeat alternating before/after. Empty event lists do not prove event quality.
Role model service and full GPU/live scheduling are still excluded.
Artifacts: artifacts/doorseg_research/cpu_queue_20261010.json.

The initial globally locked experimental frame cache slowed the light CPU
sample in alternating tests. It now uses a per-frame Future: only metadata
lookup holds the global lock, so different frames prepare concurrently.
Immutable1280x720 RGB inputs avoid an identity-resize copy. Repeated120-tick
ABBA preserves every feature/vote/ID/view hash; dense feature service falls
18.50→15.05ms but total CPU processing improves only about2.4%, light unchanged.
A second200-tick180:380 TRAIN ABBA includes actual exits: both variants give
01405 out21.36s/ID2 and15375 out20.72s/ID1, including identical geometry.
Dense total CPU mean6.377→5.379s, about15.6%; light is nearly unchanged.
These locally reset fragments do not establish full-clip event accuracy.

Two finite CPU-only owned role workers processed identical eight TRAIN views.
Four new views cost2.82–2.95s, ~0.72s/view; repeated Roles cache calls take
36–43microseconds. Probabilities/roles match; these are model outputs, not
independent owner correctness. Workers exited, production role services intact.
The measured role service excludes live queue backlog and GPU competition.

A read-only live check found a significant backlog predating the CPU tests:
closing21:00:16 retained23547ticks, behind5685s at21:00:23. At21:30:23
fresh status reports6002s behind,233ms/tick and no skipping; it continues
draining and counting events. Research GPU waits for this live work and
foreign applications. No production restart or queue truncation occurred.
Combined evidence: artifacts/doorseg_research/cpu_queue_roles_20261010.json.
