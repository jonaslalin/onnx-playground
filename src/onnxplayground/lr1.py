import numpy as np
import numpy.typing as npt
from onnx import ModelProto, TensorProto
from onnx.checker import check_model
from onnx.defs import onnx_opset_version
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.parser import parse_model
from onnx.reference import ReferenceEvaluator


def make_lr1_model() -> ModelProto:
    X = make_tensor_value_info("X", TensorProto.FLOAT, ["M", "N"])
    A = make_tensor_value_info("A", TensorProto.FLOAT, ["N", "L"])
    B = make_tensor_value_info("B", TensorProto.FLOAT, ["M", "L"])

    node1 = make_node("MatMul", ["X", "A"], ["XA"])
    node2 = make_node("Add", ["XA", "B"], ["Y"])

    Y = make_tensor_value_info("Y", TensorProto.FLOAT, ["M", "L"])

    graph = make_graph([node1, node2], "lr1", [X, A, B], [Y])

    model = make_model(graph)
    check_model(model, full_check=True)
    return model


def make_lr1_model_v2() -> ModelProto:
    model_text = f"""
    <
        ir_version: 10,
        opset_import: ["": {onnx_opset_version()}]
    >
    lr1 (float[M, N] X, float[N, L] A, float[M, L] B) => (float[M, L] Y) {{
        XA = MatMul(X, A)
        Y = Add(XA, B)
    }}
    """
    model = parse_model(model_text)
    check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    model = make_lr1_model_v2()
    print(model)

    with open("models/lr1.onnx", "wb") as f:
        f.write(model.SerializeToString())

    X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    A = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    B = np.array([[1]], dtype=np.float32)
    print(f"X:\n{X}\nX.shape: {X.shape}")
    print(f"A:\n{A}\nA.shape: {A.shape}")
    print(f"B:\n{B}\nB.shape: {B.shape}")

    sess = ReferenceEvaluator(model)
    inputs = {"X": X, "A": A, "B": B}
    outputs = sess.run(None, inputs)

    Y: npt.NDArray[np.float32] = outputs[0]
    print(f"Y:\n{Y}\nY.shape: {Y.shape}")
