"""ganpaint-pipeline command line: labels | dissect | ace | export."""
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SEG_DIR = DATA / "downloads" / "upp-resnet50-upernet"
OLD_PTH = DATA / "downloads" / "old-gandissect" / "churchoutdoor_lsun.pth"
PROJECT = DATA / "churchoutdoor"
ACE_CLASSES = ["tree", "grass", "door"]
BRUSH_LABELS = ["tree", "grass", "door", "sky", "cloud", "brick", "dome"]


def _model():
    from .convert_progan import load_old_pth

    return load_old_pth(OLD_PTH).cuda().eval()


@app.command()
def labels(meta_dir: Path = DATA / "downloads" / "unifiedparsing-meta"):
    """Rebuild the segmenter's labels.json from the unifiedparsing CSVs and check it against the decoder."""
    from .labels import write_labels

    print(write_labels(meta_dir, SEG_DIR / "decoder_epoch_40.pth", SEG_DIR / "labels.json"))


@app.command()
def dissect(size: int = 1000, outdir: Path = PROJECT / "dissect"):
    """Dissect layer4: quantiles, IQR thresholds, IoU tables, dissect.json."""
    from .dissect import dissect as run
    from .segmenter import UnifiedParsingSegmenter

    run(_model(), UnifiedParsingSegmenter(SEG_DIR, segdiv="quad"), outdir, size=size,
        extra_labels=BRUSH_LABELS, settings=dict(size=size))


@app.command()
def ace(classname: str, add_at: str = "cloc", dissect_dir: Path = PROJECT / "dissect", write: bool = True):
    """Learn the ACE unit ranking for a class; with --write, add `<class>-ace` to dissect.json."""
    from .ace import add_ace_ranking_to_dissection, run_ace
    from .segmenter import UnifiedParsingSegmenter

    seg = UnifiedParsingSegmenter(SEG_DIR, segdiv="undivided")
    scores = run_ace(_model(), seg, dissect_dir, classname, variant=add_at, add_at=add_at)
    if write:
        add_ace_ranking_to_dissection(dissect_dir, "layer4", classname, scores)


@app.command()
def export(fallback_iou: bool = False):
    """Write generator.safetensors and ganpaint.json for the server."""
    from .export import export as run

    for brush, ranking in run(PROJECT, OLD_PTH, fallback_iou=fallback_iou).items():
        print(f"{brush:6s} <- {ranking}")


if __name__ == "__main__":
    app()
