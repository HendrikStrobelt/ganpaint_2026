"""GANPaint HTTP API (FastAPI).

GET  /api/projects                          list projects
GET  /api/projects/{p}                      brushes, strengths, gallery, canvas grid for the UI
GET  /api/projects/{p}/samples/{id}.webp    an unedited sample
POST /api/projects/{p}/generate             a sample with strokes applied -> {image: data URL}
The built web UI (web/dist) is served at / when present.
"""
import os
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .project import Project, data_url, encode_image

CANVAS_GRID = 32  # strokes are painted on a 32 x 32 mask, like the original UI
MAX_MASK_CHARS = 100_000


class Stroke(BaseModel):
    brush: str
    mode: Literal["draw", "remove"]
    strength: Literal["low", "med", "high"] = "high"
    mask: str = Field(max_length=MAX_MASK_CHARS, description="PNG data URL; nontransparent pixels are painted")


class GenerateRequest(BaseModel):
    id: int = Field(ge=0)
    strokes: list[Stroke] = Field(default_factory=list, max_length=64)


class GenerateResponse(BaseModel):
    image: str


def load_projects(data_dir, device="cpu"):
    return {p.parent.name: Project.load(p.parent, device=device) for p in sorted(Path(data_dir).glob("*/ganpaint.json"))}


def create_app(projects, web_dir=None):
    app = FastAPI(title="GANPaint")

    def get_project(name):
        if name not in projects:
            raise HTTPException(404, f"unknown project {name!r}")
        return projects[name]

    @app.get("/api/projects")
    def list_projects():
        return [dict(name=n, layer=p.layer) for n, p in projects.items()]

    @app.get("/api/projects/{name}")
    def project_info(name: str):
        p = get_project(name)
        c = p.config
        return dict(
            name=name, layer=p.layer, canvas_grid=CANVAS_GRID, strengths=c["strengths"],
            default_strength=c.get("default_strength", "high"),
            brushes=[dict(name=b, label=v.get("label", b)) for b, v in c["brushes"].items()],
            gallery=[dict(id=i, thumb=f"/api/projects/{name}/samples/{i}.webp") for i in c["gallery"]],
            max_id=p.max_id,
        )

    @app.get("/api/projects/{name}/samples/{sample_id}.webp")
    def sample(name: str, sample_id: int):
        p = get_project(name)
        try:
            image = p.generate(sample_id)
        except IndexError as e:
            raise HTTPException(404, str(e))
        return Response(encode_image(image), media_type="image/webp", headers={"Cache-Control": "public, max-age=86400"})

    @app.post("/api/projects/{name}/generate", response_model=GenerateResponse)
    def generate(name: str, req: GenerateRequest):
        p = get_project(name)
        for s in req.strokes:
            if s.brush not in p.config["brushes"]:
                raise HTTPException(422, f"unknown brush {s.brush!r}")
            if s.strength not in p.config["strengths"]:
                raise HTTPException(422, f"unknown strength {s.strength!r}")
        try:
            image = p.generate(req.id, [s.model_dump() for s in req.strokes])
        except IndexError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:  # undecodable or oversized mask
            raise HTTPException(422, str(e))
        return GenerateResponse(image=data_url(encode_image(image)))

    if web_dir and Path(web_dir).is_dir():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def app_from_env():
    """uvicorn --factory ganpaint_server.app:app_from_env  (GANPAINT_DATA, GANPAINT_WEB, GANPAINT_DEVICE).

    GANPAINT_DEVICE defaults to the best available of cuda, mps, cpu.
    """
    data = os.environ.get("GANPAINT_DATA", "data")
    device = os.environ.get("GANPAINT_DEVICE") or default_device()
    return create_app(load_projects(data, device), os.environ.get("GANPAINT_WEB"))
