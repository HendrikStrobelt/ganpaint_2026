# Reproduction report (churchoutdoor, layer4)

How closely the regenerated data matches the old demo's files (`reference/levels_demo.json`,
`reference/rankings_demo_iou_short.json`). Generator: the original `churchoutdoor_lsun.pth`
(sha256 `4cd98dcf9bb2b8598cb29ecaa0d1d748fb3052ed3bf5185187476c5a51bc38bc`). Segmenter: the original
`upp-resnet50-upernet` epoch 40, with `labels.json` rebuilt from the unifiedparsing CSVs.

| check | result | target |
|---|---|---|
| layer4 0.99 levels vs demo (generator + estimator) | Pearson 0.998, Spearman 0.984, median relative error 2%, p90 5%, max 22% | within a few % |
| `*-iou` rankings vs demo, all 30 comparable | per-unit Spearman 0.84 to 0.95 | >= 0.9 (**missed** for 5 of the 7 brush concepts) |
| brush IoU rankings, top-15 overlap | sky 14, cloud 15, tree 15, grass 15, door 15, dome 13, brick 11 (of 15) | >= 10 |
| `*-ace` rankings, top-15 overlap | tree 14, grass 13, door 13 (of 15); per-unit Spearman 0.74 / 0.66 / 0.64 | >= 10 overlap |
| final brush unit lists vs demo top-15 | tree 14, grass 13, door 13, sky 14, cloud 15, brick 11, dome 13 | >= 10 |

Notes

* Spearman over all 512 units is dominated by the many near-zero units, whose order is noise; the top-unit
  overlap is what the brushes use. The Spearman targets were set before this was known.
* ACE: the repository's `ace_loss` measures the insertion effect at the present-sample cell indices (`ploc`),
  which looks like a bug (`cloc`, the candidate cells, would be natural). Both were trained for tree
  (one seed each): `ploc` overlap 14/15, Spearman 0.737, 370 nonzero units; `cloc` 13/15, 0.693, 445
  (the demo has 329). `ploc` was kept for all three classes. The difference is small and not tested across seeds.
* Edit levels are the 0.99 quantiles from the original streaming estimator (error up to ~0.11 sigma in the
  tail), as the demo used, not exact quantiles.
* Not reproduced: `fabric-iou` and `metal-iou` rankings (no unit has them as its best label in this run),
  per-unit top-image thumbnails (`/units`), the `<class>-ace` rankings for person and window.
* The generator's stored output-layer `wscale` differs from the computed value by 5.9%; like the original
  loader, the computed value is used.
* The quad-subdivision step skips the highest-numbered connected component per image, as the original does.
  This only affects the `-t/-l/-b/-r` labels, which no brush uses.
