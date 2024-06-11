import numpy as np
import numpy.typing as npt
from onnx import ModelProto, TensorProto
from onnx.checker import check_model
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.numpy_helper import from_array
from onnx.reference import ReferenceEvaluator


def make_lr2_model() -> ModelProto:
    X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])

    A = from_array(np.array([[1, 2, 3], [4, 5, 6]], np.float32), "A")
    B = from_array(np.array([[1, 2, 3]], dtype=np.float32), "B")
    node1 = make_node("MatMul", ["X", "A"], ["XA"])
    node2 = make_node("Add", ["XA", "B"], ["Y"])

    Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])

    graph = make_graph([node1, node2], "lr2", [X], [Y], [A, B])

    model = make_model(graph)
    check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    model = make_lr2_model()
    print(model)

    with open("models/lr2.onnx", "wb") as f:
        f.write(model.SerializeToString())

    X = np.array([[1, 2], [3, 4]], dtype=np.float32)
    print(f"X:\n{X}\nX.shape: {X.shape}")

    sess = ReferenceEvaluator(model)
    inputs = {"X": X}
    outputs = sess.run(None, inputs)

    Y: npt.NDArray[np.float32] = outputs[0]
    print(f"Y:\n{Y}\nY.shape: {Y.shape}")
