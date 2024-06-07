import numpy as np
from onnx import TensorProto
from onnx.checker import check_model
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.reference import ReferenceEvaluator

X = make_tensor_value_info("X", TensorProto.FLOAT, [None, 2])
A = make_tensor_value_info("A", TensorProto.FLOAT, [2, 3])
B = make_tensor_value_info("B", TensorProto.FLOAT, [1, 1])

Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, 3])

node1 = make_node("MatMul", ["X", "A"], ["XA"])
node2 = make_node("Add", ["XA", "B"], ["Y"])

g = make_graph([node1, node2], "g", [X, A, B], [Y])

model = make_model(g)
check_model(model)
print(model)

feed = {
    "X": np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
        ],
        dtype=np.float32,
    ),
    "A": np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=np.float32,
    ),
    "B": np.array(
        [
            [1],
        ],
        dtype=np.float32,
    ),
}
sess = ReferenceEvaluator(model)
output = sess.run(None, feed)[0]
print(output)
print(output.shape)
