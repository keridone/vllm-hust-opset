import dis
import importlib.metadata
import os

from vllm.plugins import load_general_plugins


print("VLLM_PLUGINS=", os.getenv("VLLM_PLUGINS"))
print(
    "general_plugins=",
    [(item.name, item.value) for item in importlib.metadata.entry_points(group="vllm.general_plugins")],
)
load_general_plugins()

from vllm_ascend.ops.triton.batch_invariant import matmul


allocation_attributes = [
    instruction.argval
    for instruction in dis.get_instructions(matmul.linear_persistent)
    if instruction.opname in {"LOAD_ATTR", "LOAD_METHOD"}
    and instruction.argval in {"zeros", "empty"}
]
print("linear_persistent_allocation=", allocation_attributes)
assert allocation_attributes == ["empty"]
