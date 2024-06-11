import numpy as np
import numpy.typing as npt
from onnx import ModelProto
from onnx.checker import check_model
from onnx.compose import add_prefix, merge_models
from onnx.reference import ReferenceEvaluator

from .concat import make_concat_model
from .reducesum import make_reducesum_model


def make_head_model() -> ModelProto:
    concat_model = make_concat_model()
    reducesum_model = make_reducesum_model()

    add_prefix(concat_model, "concat_", inplace=True)
    add_prefix(reducesum_model, "reducesum_", inplace=True)

    model = merge_models(concat_model, reducesum_model, [("concat_Y", "reducesum_X")], name="head", doc_string="")
    check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    model = make_head_model()
    print(model)

    with open("models/head.onnx", "wb") as f:
        f.write(model.SerializeToString())

    X1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    X2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
    X3 = np.array([[9, 10], [11, 12]], dtype=np.float32)
    print(f"X1:\n{X1}\nX1.shape: {X1.shape}")
    print(f"X2:\n{X2}\nX2.shape: {X2.shape}")
    print(f"X3:\n{X3}\nX3.shape: {X3.shape}")

    sess = ReferenceEvaluator(model)
    inputs = {"concat_X1": X1, "concat_X2": X2, "concat_X3": X3}
    outputs = sess.run(None, inputs)

    Y: npt.NDArray[np.float32] = outputs[0]
    print(f"Y:\n{Y}\nY.shape: {Y.shape}")
