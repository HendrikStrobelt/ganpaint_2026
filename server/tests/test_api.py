import base64
import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from ganpaint_pipeline.proggan import ProgressiveGenerator
from ganpaint_pipeline.zdataset import z_sample
from ganpaint_server.app import create_app
from ganpaint_server.project import Project

UNITS = 16


def png_url(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


FULL_MASK = png_url(Image.new("RGBA", (32, 32), (255, 0, 0, 255)))


@pytest.fixture(scope="module")
def project():
    torch.manual_seed(0)
    gen = ProgressiveGenerator(256, sizes=[512] + [UNITS] * 7)  # z stays 512-d; tiny channel counts
    config = dict(
        layer="layer4", generator="unused", levels=[2.0 + u for u in range(UNITS)],
        strengths={"low": 2, "med": 4, "high": 6},
        brushes={"tree": dict(units=[3, 5, 7, 9, 11, 13], label="tree"), "sky": dict(units=[0, 1, 2, 4, 6, 8])},
        gallery=[0, 1, 2],
    )
    return Project("test", gen, config, z_pool=8)


@pytest.fixture
def client(project):
    return TestClient(create_app({"test": project}))


def decode(url):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")).astype(int)


def plain(project, i):
    z = z_sample(8, seed=1)[i : i + 1]
    with torch.no_grad():
        x = project.generator(z)[0]
    return ((x + 1) / 2 * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()


def test_unedited_generate_equals_plain_forward(project):
    assert np.array_equal(project.generate(3), plain(project, 3))


def test_edit_equals_manual_layer_edit(project):
    strokes = [dict(brush="tree", mode="remove", strength="med", mask=FULL_MASK)]
    got = project.generate(2, strokes)
    z = z_sample(8, seed=1)[2:3]
    with torch.no_grad():
        h = project.head(z.view(1, -1, 1, 1))
        h[:, [3, 5, 7, 9]] = 0.0  # med = 4 units, removed everywhere
        x = project.tail(h)[0]
    want = ((x + 1) / 2 * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    assert np.array_equal(got, want)
    assert not np.array_equal(got, plain(project, 2))


def test_draw_sets_level(project):
    strokes = [dict(brush="sky", mode="draw", strength="low", mask=FULL_MASK)]
    z = z_sample(8, seed=1)[1:2]
    with torch.no_grad():
        h = project.head(z.view(1, -1, 1, 1))
        h[:, 0], h[:, 1] = 2.0, 3.0  # units 0, 1 at their levels
        want = ((project.tail(h)[0] + 1) / 2 * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    assert np.array_equal(project.generate(1, strokes), want)


def test_api_roundtrip(client):
    info = client.get("/api/projects/test").json()
    assert [b["name"] for b in info["brushes"]] == ["tree", "sky"] and info["canvas_grid"] == 32
    assert client.get("/api/projects").json() == [dict(name="test", layer="layer4")]
    r = client.get("/api/projects/test/samples/1.webp")
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
    body = dict(id=1, strokes=[dict(brush="tree", mode="remove", mask=FULL_MASK)])
    r = client.post("/api/projects/test/generate", json=body)
    assert r.status_code == 200 and decode(r.json()["image"]).shape == (256, 256, 3)


@pytest.mark.parametrize("path,body,code", [
    ("/api/projects/nope", None, 404),
    ("/api/projects/test/samples/999.webp", None, 404),
    ("/api/projects/test/generate", dict(id=999), 404),
    ("/api/projects/test/generate", dict(id=-1), 422),
    ("/api/projects/test/generate", dict(id=0, strokes=[dict(brush="moon", mode="draw", mask=FULL_MASK)]), 422),
    ("/api/projects/test/generate", dict(id=0, strokes=[dict(brush="tree", mode="paint", mask=FULL_MASK)]), 422),
    ("/api/projects/test/generate", dict(id=0, strokes=[dict(brush="tree", mode="draw", mask="data:image/png;base64,@@")]), 422),
    ("/api/projects/test/generate", dict(id=0, strokes=[dict(brush="tree", mode="draw", mask=png_url(Image.new("RGBA", (600, 600))))]), 422),
])
def test_bad_requests(client, path, body, code):
    r = client.post(path, json=body) if body is not None else client.get(path)
    assert r.status_code == code, r.text
