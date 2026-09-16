import json
import time

import torch
import torch_npu  # noqa: F401
from vllm_ascend.ops.triton.batch_invariant.matmul import linear_persistent
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

SHAPES = [
    (1, 257, 257),
    (7, 17, 13),
    (127, 127, 129),
    (128, 128, 256),
    (129, 65, 257),
    (255, 255, 129),
    (256, 256, 128),
    (257, 257, 129),
    (513, 129, 1025),
    (1023, 257, 129),
    (1024, 256, 128),
    (1025, 257, 257),
]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
TOLERANCES = {
    torch.float16: (2e-3, 2e-2),
    torch.bfloat16: (2e-2, 5e-2),
    torch.float32: (1e-4, 1e-4),
}


def main() -> None:
    init_device_properties_triton()
    results = []
    for dtype in DTYPES:
        for m, k, n in SHAPES:
            torch.manual_seed(42)
            x = (torch.randn((m, k), dtype=torch.float32) * 0.2).to(dtype=dtype, device="npu")
            weight = (torch.randn((n, k), dtype=torch.float32) * 0.2).to(dtype=dtype, device="npu")
            expected = torch.nn.functional.linear(x.cpu().float(), weight.cpu().float()).to(dtype)
            launches = [linear_persistent(x, weight) for _ in range(3)]
            torch.npu.synchronize()
            rtol, atol = TOLERANCES[dtype]
            for actual in launches:
                assert bool(torch.isfinite(actual).all())
                torch.testing.assert_close(actual.float().cpu(), expected.float(), rtol=rtol, atol=atol)
            for actual in launches[1:]:
                torch.testing.assert_close(actual, launches[0], rtol=0, atol=0)
            results.append({"dtype": str(dtype), "m": m, "n": n, "k": k, "launches": 3, "passed": True})
    print(json.dumps({"timestamp": time.time(), "cases": len(results), "launches": len(results) * 3, "results": results}))


if __name__ == "__main__":
    main()
