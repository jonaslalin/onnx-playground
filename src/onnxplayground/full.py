import numpy as np
import numpy.typing as npt
from onnx import ModelProto
from onnx.checker import check_model
from onnx.compose import add_prefix, merge_models
from onnx.helper import make_graph, make_model
from onnx.reference import ReferenceEvaluator

from .head import make_head_model
from .lr2 import make_lr2_model


def make_full_model() -> ModelProto:
    lr21_model = make_lr2_model()
    lr22_model = make_lr2_model()
    lr23_model = make_lr2_model()
    head_model = make_head_model()

    add_prefix(lr21_model, "lr21_", inplace=True)
    add_prefix(lr22_model, "lr22_", inplace=True)
    add_prefix(lr23_model, "lr23_", inplace=True)
    add_prefix(head_model, "head_", inplace=True)

    full_model = make_model(make_graph([], "full", [], []))
    full_model = merge_models(full_model, lr21_model, [], name="full", doc_string="")
    full_model = merge_models(full_model, lr22_model, [], name="full", doc_string="")
    full_model = merge_models(full_model, lr23_model, [], name="full", doc_string="")
    full_model = merge_models(
        full_model,
        head_model,
        [
            ("lr21_Y", "head_concat_X1"),
            ("lr22_Y", "head_concat_X2"),
            ("lr23_Y", "head_concat_X3"),
        ],
        name="full",
        doc_string="",
    )
    check_model(full_model, full_check=True)
    return full_model


if __name__ == "__main__":
    model = make_full_model()
    print(model)

    with open("models/full.onnx", "wb") as f:
        f.write(model.SerializeToString())

    X1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    X2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
    X3 = np.array([[9, 10], [11, 12]], dtype=np.float32)
    print(f"X1:\n{X1}\nX1.shape: {X1.shape}")
    print(f"X2:\n{X2}\nX2.shape: {X2.shape}")
    print(f"X3:\n{X3}\nX3.shape: {X3.shape}")

    sess = ReferenceEvaluator(model)
    inputs = {"lr21_X": X1, "lr22_X": X2, "lr23_X": X3}
    outputs = sess.run(None, inputs)

    Y: npt.NDArray[np.float32] = outputs[0]
    print(f"Y:\n{Y}\nY.shape: {Y.shape}")
