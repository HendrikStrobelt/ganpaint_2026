import numpy as np
import torch

from ganpaint_pipeline.runningstats import RunningConditionalQuantile, RunningQuantile, RunningTopK


def test_running_quantile_close_to_exact():
    torch.manual_seed(0)
    x = torch.randn(64000, 8) * torch.arange(1, 9) + torch.arange(8)
    rq = RunningQuantile(resolution=1024)
    for i in range(0, len(x), 6400):
        rq.add(x[i : i + 6400])
    qs = [0.5, 0.9, 0.99]
    got = rq.quantiles(qs)  # (channels, len(qs))
    exact = torch.tensor(np.quantile(x.numpy(), qs, axis=0).T)
    # RunningQuantile is a sampling-based estimator (same behaviour as the original); the error grows
    # in the tail: measured max over seeds is 0.008 / 0.018 / 0.108 sigma at q = 0.5 / 0.9 / 0.99.
    err = ((got - exact).abs() / x.std(0)[:, None]).max(0).values
    assert err[0] < 0.02 and err[1] < 0.04 and err[2] < 0.15


def test_running_topk_exact():
    torch.manual_seed(1)
    x = torch.randn(500, 4)
    tk = RunningTopK(5)
    for i in range(0, 500, 100):
        tk.add(x[i : i + 100])
    vals, idx = tk.result()
    assert torch.allclose(vals, x.topk(5, dim=0).values.T)
    assert torch.equal(x.gather(0, idx.T), vals.T)


def test_conditional_quantile_conditions():
    torch.manual_seed(2)
    rcq = RunningConditionalQuantile(resolution=512)
    a, b = torch.randn(20000, 3), torch.randn(20000, 3) + 3
    rcq.add(("label", 0), a)
    rcq.add(("label", 1), b)
    ma, mb = rcq.conditional(("label", 0)).quantiles([0.5]), rcq.conditional(("label", 1)).quantiles([0.5])
    assert ma.abs().max() < 0.1 and (mb - 3).abs().max() < 0.1
