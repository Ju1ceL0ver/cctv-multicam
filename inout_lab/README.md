# Camera 1 classification experiments

Use the working Mac Python environment:

```sh
cd /Users/aleksejzagorskij/cctv-multicam/inout_lab
/opt/homebrew/Caskroom/miniforge/base/bin/python train.py --day 20260918 --epochs 30
```

Network: 3 conv/ReLU/maxpool blocks (64/128/256), 2x2 final pooling,
3 dense layers (256/64/3). Nine geometry inputs; the constant camera input is removed.
`fit_net(..., flip_prob=0, balanced=False)` disables reflection and class weighting.

Dataset labels: outside=0, inside=1, doorway=2. Original annotation `unknown`
was mapped to outside during export; the NPZ does not preserve that original code.
Do not silently change labels based on predictions.

## Diagnosis

`diagnose.py`: train on seven days, use 18 September for development, keep
19 September out of training as separate validation. Geometric baselines and
CNN with/without reflection. These scores are not comparable to the earlier
leave-one-day-out result, which included 19 September when testing 18 September.

`mask_baseline.py`: adds person/floor overlap, bottom-mask/floor overlap,
mask area and centroid to the nine geometry features. ExtraTrees baseline.

`diagnose_unweighted.py`: CNN with these mask features, no reflection and
no class weights. Must run after the other two scripts complete.

`error_gallery.py`: creates an HTML error review from saved predictions.
Diagnostics and checkpoints are under `diagnostics/`. 18 September becomes a
development set once it is used for model choices; a new day is needed for a
new untouched final test after further tuning.

## Additional search (07 October)

`search_features.py` adds a coarse mask-location map, cropped-mask shape,
signed floor-distance quantiles and overlap at several lower-body fractions.
Six classifier families were tried on development18. Selected ExtraTrees:
84.08% development18, 92.62% validation19 (the latter has already been inspected).
`calibrate_mix.py` improved development to 84.71% but reduced validation to
90.84%; this is not a generalization win. `focus_net.py` explicitly pools CNN
features inside the selected person mask and lower-body mask; clean separation
still gave 79.94% / 89.57% after 30 epochs. Its augmented training accuracy
reached 99.6% during training. `ordinal_model.py` was weaker.

Original annotation codes were fetched read-only as
`diagnostics/source_labels.json` / `original_labels.npy`: 47 unknown answers,
one on 18 September and 13 on 19 September. No changed label values relative
to the export were found. Exact identical full-frame+geometry inputs with
conflicting labels: zero; this does not rule out semantic annotation noise.

`lodo_rich.py` checks each day using the other eight days, excludes unknown
answers from training and reports defined-label coverage explicitly.
Result: 1459/1581 correct overall (92.28%), 665/762 near door (87.27%).
The features/hyperparameters have already been developed on 18 September;
this aggregate is diagnostics, not an untouched final test.
`diagnostics/rich_final.pkl` is fitted on all defined labels for inference;
do not score it on those same labels as a test.
`review_lodo.py` generates `diagnostics/lodo_errors.html`, with all 122 errors
(first the 97 near the door). It does not change answers.
100% has NOT been achieved. Dataset labels remain unchanged.

## Targeted investigation: direct inside/outside mistakes

`binary_ablation.py` predicts two classes only (excludes doorway and unknown),
with leave-one-day-out. Rich features: 1376/1391 = 98.92%, 15 errors, zero on
six days. This is NOT three-class accuracy or a solved doorway classifier.
Simply removing free-floor distance did not improve the geometry baseline.
The three-class LODO also has 15 direct cross-side errors (13 overlap with
binary errors). Twelve are inside -> outside, all with negative free-floor
distance, clustered around interior fixtures/door occlusions. `rooms.py` masks
visible FREE FLOOR and explicitly excludes stands; `door_sam.foot_of` returns
the lowest visible mask boundary, not guaranteed anatomical feet under
occlusion. Negative free-floor distance cannot directly mean outside shop.

`person_context.py` tests a shared small CNN on a large person crop plus
whole-frame context (4 channels RGB+target mask, no free-floor channel/distance),
with eight geometry features and an auxiliary binary loss. 914692 parameters.
25 epochs, held-out 18 September, excluding unknown in train/eval:
three-class 81.79% (313 examples), binary auxiliary 94.42% on definite sides;
9 direct cross-side errors. It did not beat the tree baseline.

`side_state.py` is a local prototype of door-gated track membership. Only
trusted observations and real doorway contact may change established state;
uncertain/occluded observations retain it. Tested synthetic state scenarios
(initialization, wrong votes away from door, occlusion, entry, exit, stale ID).
NOT integrated into the live pipeline and NOT measured on real tracks. It
cannot fix initial wrong membership or identity switches by itself.

Relevant primary implementations:
- https://supervision.roboflow.com/latest/detection/tools/polygon_zone/
- https://supervision.roboflow.com/latest/detection/tools/line_zone/
- https://docs.pytorch.org/vision/stable/generated/torchvision.ops.roi_align.html

LineZone maintains per-track crossing state and confirms the opposite side
across observations; PolygonZone uses explicit anchors. These do not solve
an occluded-foot problem automatically. Region-of-interest pooling motivates
object-specific visual inputs rather than relying on a tiny person in a
whole-frame image. The dataset source labels were downloaded read-only;
Windows code and running processes were untouched in this investigation.

Средние координаты, Gaussian-карты отдельными каналами и kNN: см. [SPATIAL_PRIORS.md](SPATIAL_PRIORS.md). Координаты и параметры `spatial_means.json`, карты `spatial_channels.png`/`.npy`. При проверке каждого дня его метки исключены также из карт и соседей; на тренировочных строках применяется вложенный cross-fitting по дням, чтобы kNN не получал собственный ответ строки.
