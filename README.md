# GANPaint (reimplementation)

Paint trees, grass, doors, sky, clouds, brick and domes into GAN-generated churches. A modern rewrite of
[CSAILVision/gandissect](https://github.com/CSAILVision/gandissect)'s GANPaint: PyTorch 2 + FastAPI backend,
Svelte 5 + Vite + TypeScript frontend. Painting edits units of ProGAN `layer4` (512 x 8 x 8): a stroke becomes
`x * (1 - a) + v * a` on the units that match the brush, with `v` the unit's 0.99-quantile activation to draw
and 0 to remove.

```
pipeline/   ganpaint_pipeline: generator loading, segmenter, dissection, ACE, export   (uv project)
server/     ganpaint_server: FastAPI app, intervention math, tests                      (uv project)
web/        Svelte UI                                                                   (pnpm project)
data/       downloads/ (inputs), churchoutdoor/ (everything the pipeline produces)
reference/  files copied from the original repo, used while porting
```

## Inputs (`data/downloads/`)

| path | what | where it came from |
|---|---|---|
| `old-gandissect/churchoutdoor_lsun.pth` | the original church ProGAN | supplied by the project owner; the CSAIL host that served it is down |
| `upp-resnet50-upernet/{encoder,decoder}_epoch_40.pth` | unified-parsing segmenter | `sceneparsing.csail.mit.edu/model/unified_perceptual_parsing/` |
| `unifiedparsing-meta/*.csv` | label vocabularies, used to rebuild the missing `labels.json` | CSAILVision/unifiedparsing `meta_file/joint_dataset` |
| `karras-tf/*.pkl` | Karras' TF church checkpoint (a *different* model; kept only as a fallback) | ProGAN Google Drive folder |

## Running it

```bash
cd pipeline
uv run ganpaint-pipeline labels      # rebuild the segmenter's labels.json
uv run ganpaint-pipeline dissect     # layer4: quantiles, IQR thresholds, IoU tables  (~3.5 min on a GB10)
uv run ganpaint-pipeline ace tree    # learn the ACE ranking; also: grass, door
uv run ganpaint-pipeline export      # data/churchoutdoor/{generator.safetensors,ganpaint.json}

cd ../web && pnpm install && pnpm run build
cd ../server
GANPAINT_DATA=../data GANPAINT_WEB=../web/dist \
  uv run uvicorn --factory ganpaint_server.app:app_from_env --port 8000
```

For UI development run `pnpm run dev` in `web/` (it proxies `/api` to port 8000). Tests:
`uv run --with pytest python -I -m pytest tests` in `pipeline/`, and `uv run python -I -m pytest tests` in `server/`.

## How it differs from the original

* Everything the demo needed besides the weights was regenerated: `labels.json`, `dissect.json`, the quantiles
  and the ACE rankings. Against the old demo's data, layer4 levels correlate at r = 0.998 and the top-15
  units of the sky/cloud/tree/grass/door/dome/brick IoU rankings overlap 11-15 of 15.
* `prroi_pool` (a CUDA extension for the old `torch.utils.ffi`) is replaced by an exact pure-PyTorch version.
* `grid_sample` is called with `align_corners=True`, which the original relied on implicitly.
* The server is stateless: the client sends the sample id and the whole stroke list on every change.
