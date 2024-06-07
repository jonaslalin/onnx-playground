import numpy as np
import numpy.typing as npt
from onnx import ModelProto, TensorProto
from onnx.checker import check_model
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.reference import ReferenceEvaluator


def make_concat_model() -> ModelProto:
    X1 = make_tensor_value_info("X1", TensorProto.FLOAT, [None, None])
    X2 = make_tensor_value_info("X2", TensorProto.FLOAT, [None, None])
    X3 = make_tensor_value_info("X3", TensorProto.FLOAT, [None, None])

    node = make_node("Concat", ["X1", "X2", "X3"], ["Y"], axis=-1)

    Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])

    graph = make_graph([node], "graph", [X1, X2, X3], [Y])

    model = make_model(graph)
    check_model(model)
    return model


if __name__ == "__main__":
    model = make_concat_model()
    print(model)

    X1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    X2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
    X3 = np.array([[9, 10], [11, 12]], dtype=np.float32)
    print(f"X1:\n{X1}\nX1.shape: {X1.shape}")
    print(f"X2:\n{X2}\nX2.shape: {X2.shape}")
    print(f"X3:\n{X3}\nX3.shape: {X3.shape}")

    sess = ReferenceEvaluator(model)
    inputs = {"X1": X1, "X2": X2, "X3": X3}
    outputs = sess.run(None, inputs)

    Y: npt.NDArray[np.float32] = outputs[0]
    print(f"Y:\n{Y}\nY.shape: {Y.shape}")
