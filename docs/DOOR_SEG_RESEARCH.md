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

A new `doorseg_c_background` candidate is staged from finite step8604 weights and optimizer, with the baseline validation recorded. Its only training change is the recording-date/background repair; no GPU training has started during shop hours.

## Colab continuation, 10 October

The owner made Colab GPU quota available. Session `doorseg-c` has a Tesla T4 with 15360 MiB. A100 and L4 allocation were rejected. The complete unchanged data manifest is **148 training sources, six validation sources, 1216 files, 57.7477 GB**; video transfer is in progress, so no new successful training step or quality improvement is claimed yet. Small labels/backgrounds/checkpoint travel as one 1.099-GB stored ZIP to avoid hundreds of HTTP round trips.

The VM uses the same torch 2.13.0+cu130, torchvision 0.28.0+cu130 and Ultralytics 8.4.108 as Windows. Installation exposed overlapping CUDA-12/CUDA-13 cuDNN packages; a real convolution failed with `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`. Removing both distributions and installing only torch's required `nvidia-cudnn-cu13==9.20.0.48` fixed it. A six-example BF16 forward/backward from the finite checkpoint passed: loss 8.265064, peak allocation 5.4768 GB. This is compatibility evidence, not speed or accuracy evidence for production.

`door_seg_colab.py` verifies the source manifest, initial checkpoint SHA256, successful step and finite model/optimizer before starting the unchanged batch-six training recipe (lr 1e-4, 400 validation moments, 2000 successful steps between checks). Two data workers fit the Colab CPU. New checkpoints and progress are backed up every two minutes; large checkpoints use 24-MiB HTTP pieces. `door_seg_colab_restore.py` on Windows verifies the completed SHA256 and finite model/optimizer, refuses a rollback over a newer local checkpoint, and restores atomically. The Colab sender waits for receiver acknowledgement. `COLAB_ACTIVE.json` blocks a duplicate Windows supervisor; ownership clears only after the final backup succeeds.

The existing keepalive mechanism was found in `tennis-pose/tools/colab_keepalive.sh` and that project's AGENTS.md, lines 273 and 383: background computation does not keep the notebook kernel active, and overlapping `colab exec` commands have hung. The Mac now runs a kernel/job probe every three minutes under `caffeinate`. All executions use the serialized wrapper:

```sh
python3 tools/doorseg_colab_control.py exec --file /tmp/probe.py
```

The pinger stops when the runner exits; restart it after recovering a failed runner. Its log is `data/logs/colab_doorseg-c_keepalive.log`. The existing task heartbeat now checks every ten minutes throughout the day, stays quiet during healthy transfer/training, and retains the wider research/production acceptance requirements. Release the owned Colab session only after Windows checkpoint restoration is verified.

A full release still requires: a successful selected-policy comparison on the frozen paired event cohort, clean and crowded end-to-end speed including side/role queues, mask equivalence after fusion, identity quality, then live shadow/integration checks preserving CRM IDs and format. GPU tests wait for the night window; CPU cache tests can run during the day. A fast isolated frame and a training-only event gain do not satisfy these conditions.

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
