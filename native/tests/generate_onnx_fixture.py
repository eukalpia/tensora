from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
    output_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])

    weight = numpy_helper.from_array(
        np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32), name="W"
    )
    bias = numpy_helper.from_array(
        np.asarray([1.0, -1.0], dtype=np.float32), name="B"
    )

    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["X", "W"], ["XW"]),
            helper.make_node("Add", ["XW", "B"], ["Y"]),
        ],
        "tensora_reference_graph",
        [input_info],
        [output_info],
        initializer=[weight, bias],
    )
    model = helper.make_model(
        graph,
        producer_name="tensora",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def build_no_input_model() -> onnx.ModelProto:
    output_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
    constant = numpy_helper.from_array(
        np.asarray([1.0], dtype=np.float32), name="constant_value"
    )

    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["Y"], value=constant)],
        "tensora_no_input_graph",
        [],
        [output_info],
    )
    model = helper.make_model(
        graph,
        producer_name="tensora",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def build_double_input_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("X", TensorProto.DOUBLE, [2, 2])
    output_info = helper.make_tensor_value_info("Y", TensorProto.DOUBLE, [2, 2])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["X"], ["Y"])],
        "tensora_double_input_graph",
        [input_info],
        [output_info],
    )
    model = helper.make_model(
        graph,
        producer_name="tensora",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def build_double_output_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
    output_info = helper.make_tensor_value_info("Y", TensorProto.DOUBLE, [2, 2])
    graph = helper.make_graph(
        [helper.make_node("Cast", ["X"], ["Y"], to=TensorProto.DOUBLE)],
        "tensora_double_output_graph",
        [input_info],
        [output_info],
    )
    model = helper.make_model(
        graph,
        producer_name="tensora",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def sibling_model_path(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}{suffix}{output.suffix}")


def no_input_model_path(output: Path) -> Path:
    return sibling_model_path(output, "-no-input")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), args.output)
    onnx.save(build_no_input_model(), no_input_model_path(args.output))
    onnx.save(
        build_double_input_model(), sibling_model_path(args.output, "-double-input")
    )
    onnx.save(
        build_double_output_model(), sibling_model_path(args.output, "-double-output")
    )


if __name__ == "__main__":
    main()
