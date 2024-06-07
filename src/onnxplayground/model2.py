import numpy as np
import numpy.typing as npt
from onnx import ModelProto, TensorProto
from onnx.checker import check_model
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.numpy_helper import from_array
from onnx.reference import ReferenceEvaluator


def make_my_model() -> ModelProto:
    X = make_tensor_value_info("X", TensorProto.FLOAT, [None, 2])  # m x 2

    A = from_array(np.array([[1, 2, 3], [4, 5, 6]], np.float32), "A")  # 2 x 3
    B = from_array(np.array([[1, 2, 3]], dtype=np.float32), "B")  # 1 x 3

    node1 = make_node("MatMul", ["X", "A"], ["XA"])
    node2 = make_node("Add", ["XA", "B"], ["Y"])

    Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, 3])  # m x 3

    g = make_graph([node1, node2], "g", [X], [Y], [A, B])

    model = make_model(g)
    check_model(model)
    return model


if __name__ == "__main__":
    my_model = make_my_model()
    print(my_model)

    X = np.array([[1, 2], [3, 4]], dtype=np.float32)

    sess = ReferenceEvaluator(my_model, verbose=4)
    output: npt.NDArray[np.float32] = sess.run(None, {"X": X})[0]
    print(output)
    print(output.shape)
