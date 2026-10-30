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

## Quick start

Needs [uv](https://docs.astral.sh/uv/). A CUDA GPU is optional. [pnpm](https://pnpm.io/) (which provides Node) is
only needed to change the UI: the compiled UI is committed in `web/dist/`.

**1. Data.** The server needs `data/churchoutdoor/generator.safetensors` and `data/churchoutdoor/ganpaint.json`.
They are not in git. If you already have a `data/` folder from a previous run, copy it into the repo root and
skip to step 2. Otherwise generate them once (see [Inputs](#inputs-datadownloads) and
[Running it](#running-it) for the one-off steps; on a GB10 dissection takes about 4 minutes and each of the
three ACE classes 30-45 minutes).

**2. Start the server**

```bash
cd server
GANPAINT_DATA=../data GANPAINT_WEB=../web/dist \
  uv run uvicorn --factory ganpaint_server.app:app_from_env --port 8000
```

Open <http://127.0.0.1:8000/>: pick a church, choose Draw or Remove and a brush, then drag on the image.
Ctrl+Z undoes a stroke.

* The device is picked automatically (cuda, then mps, then cpu); set `GANPAINT_DEVICE=cuda|mps|cpu` to override.
  An edit takes about 10-15 ms on CUDA against about 150 ms on a CPU (GB10); on an Apple M4 Max about 5 ms on
  `mps` against about 20 ms on `cpu`.
* The server binds to localhost. Adding `--host 0.0.0.0` exposes it to your network; it has no
  authentication or rate limiting, so only do that on a trusted network.

**UI development.** Run the server as above, then `cd web && pnpm install && pnpm run dev` and open the URL Vite
prints (<http://localhost:5173/>). It hot-reloads and proxies `/api` to port 8000. The server serves the committed
`web/dist/`, so after changing the UI run `pnpm run build` and commit the updated `web/dist/` too.

If the page says "the server has no projects", `GANPAINT_DATA` does not point at the folder that contains
`churchoutdoor/ganpaint.json`.

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

cd ../server    # the UI is already built in ../web/dist
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
