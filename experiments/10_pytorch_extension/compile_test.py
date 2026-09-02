from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
import torch._dynamo as dynamo
import torch.nn.functional as F

from accelerateworld_ops import silu_mul


def custom_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Small compiler graph with the custom CUDA op kept as an opaque boundary."""

    fused = silu_mul(gate, up)
    return torch.tanh(fused) + 0.125 * fused


def native_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    fused = F.silu(gate) * up
    return torch.tanh(fused) + 0.125 * fused


def custom_loss(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return custom_forward(gate, up).square().mean()


def native_loss(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return native_forward(gate, up).square().mean()


def _make_inputs(shape: tuple[int, ...], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    gate = torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator) * 0.75
    up = torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator)
    return gate, up


def _loss_and_grads(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    gate_data: torch.Tensor,
    up_data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate = gate_data.detach().clone().requires_grad_(True)
    up = up_data.detach().clone().requires_grad_(True)
    loss = fn(gate, up)
    grad_gate, grad_up = torch.autograd.grad(loss, (gate, up))
    return loss.detach(), grad_gate.detach(), grad_up.detach()


def _assert_close_triplet(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def validate_full_opcheck(gate: torch.Tensor, up: torch.Tensor) -> dict[str, str]:
    result = torch.library.opcheck(
        torch.ops.accelerateworld.silu_mul.default,
        (gate.detach().clone().requires_grad_(True), up.detach().clone().requires_grad_(True)),
    )
    expected = {
        "test_schema",
        "test_autograd_registration",
        "test_faketensor",
        "test_aot_dispatch_dynamic",
    }
    if set(result) != expected:
        raise RuntimeError(f"unexpected opcheck result keys: {sorted(result)}")
    if any(value != "SUCCESS" for value in result.values()):
        raise RuntimeError(f"opcheck failed: {result}")
    return result


def validate_dynamo_capture(gate: torch.Tensor, up: torch.Tensor) -> None:
    explanation = dynamo.explain(custom_forward)(gate, up)
    if explanation.graph_count != 1:
        raise RuntimeError(f"expected one Dynamo graph, got {explanation.graph_count}")
    if explanation.graph_break_count != 0:
        raise RuntimeError(
            f"expected zero graph breaks, got {explanation.graph_break_count}: "
            f"{explanation.break_reasons}"
        )

    targets = [
        str(node.target)
        for node in explanation.graphs[0].graph.nodes
        if node.op == "call_function"
    ]
    if not any("accelerateworld.silu_mul" in target for target in targets):
        raise RuntimeError(f"custom operator boundary missing from Dynamo graph: {targets}")


def validate_dynamic_single_graph(shapes: tuple[tuple[int, int], ...]) -> int:
    compile_count = 0

    def counting_backend(graph_module: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
        del example_inputs
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    dynamo.reset()
    compiled = torch.compile(
        custom_forward,
        backend=counting_backend,
        fullgraph=True,
        dynamic=True,
    )

    for index, shape in enumerate(shapes):
        gate, up = _make_inputs(shape, seed=3100 + index)
        actual = compiled(gate, up)
        expected = native_forward(gate, up)
        torch.testing.assert_close(actual, expected, rtol=5e-5, atol=2e-5)

    if compile_count != 1:
        raise RuntimeError(f"dynamic shape workload recompiled {compile_count} graphs; expected one")
    return compile_count


def validate_backend_ladder(shapes: tuple[tuple[int, int], ...]) -> None:
    for backend in ("eager", "aot_eager", "inductor"):
        dynamo.reset()
        compiled_loss = torch.compile(
            custom_loss,
            backend=backend,
            fullgraph=True,
            dynamic=True,
        )
        for index, shape in enumerate(shapes):
            gate, up = _make_inputs(shape, seed=4100 + index)
            expected = _loss_and_grads(native_loss, gate, up)
            actual = _loss_and_grads(compiled_loss, gate, up)
            _assert_close_triplet(actual, expected)
        torch.cuda.synchronize()
        print(f"  {backend}: forward + backward PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=257)
    args = parser.parse_args()

    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("--rows and --cols must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    gate, up = _make_inputs((args.rows, args.cols), seed=2026)
    opcheck = validate_full_opcheck(gate, up)
    validate_dynamo_capture(gate, up)

    dynamic_shapes = ((2, 257), (4, 513), (8, 1025))
    compile_count = validate_dynamic_single_graph(dynamic_shapes)

    # The ladder isolates failures by compiler layer:
    # eager -> Dynamo, aot_eager -> AOTAutograd, inductor -> TorchInductor.
    validate_backend_ladder(((4, 257), (8, 513)))

    print("PyTorch compiler integration — fused SiLU*mul")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  opcheck: {opcheck}")
    print("  Dynamo fullgraph capture: PASS")
    print("  Custom operator retained as FX boundary: PASS")
    print(f"  Dynamic-shape Dynamo graph count: {compile_count}")
    print("  AOTAutograd training path: PASS")
    print("  TorchInductor training path: PASS")
    print("  Validation: PASS")


if __name__ == "__main__":
    main()
