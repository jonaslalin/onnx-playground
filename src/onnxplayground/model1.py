import numpy as np
import numpy.typing as npt
from onnx import ModelProto, TensorProto
from onnx.checker import check_model
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.reference import ReferenceEvaluator


def make_my_model() -> ModelProto:
    X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
    A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
    B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])

    node1 = make_node("MatMul", ["X", "A"], ["XA"])
    node2 = make_node("Add", ["XA", "B"], ["Y"])

    Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])

    g = make_graph([node1, node2], "g", [X, A, B], [Y])

    model = make_model(g)
    check_model(model)
    return model


if __name__ == "__main__":
    my_model = make_my_model()
    print(my_model)

    X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  # 2 x 3
    A = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)  # 3 x 2
    B = np.array([[1]], dtype=np.float32)  # 1 x 1

    sess = ReferenceEvaluator(my_model, verbose=4)
    output: npt.NDArray[np.float32] = sess.run(None, {"X": X, "A": A, "B": B})[0]
    print(output)
    print(output.shape)
