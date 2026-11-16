from pathlib import Path

import numpy as np

from .audio import Frontend, coefficients, features
from .data import LABELS


def build_model(
    config: Frontend,
    channels: tuple[int, int, int],
    batch_size: int | None = None,
):
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=config.shape, batch_size=batch_size, name="log_mel"),
            tf.keras.layers.Conv2D(
                channels[0], (5, 3), strides=2, padding="same", activation="relu"
            ),
            tf.keras.layers.Conv2D(
                channels[1], (5, 3), strides=2, padding="same", activation="relu"
            ),
            tf.keras.layers.Conv2D(
                channels[2], (5, 3), strides=2, padding="same", activation="relu"
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(len(LABELS), activation="softmax"),
        ],
        name="doorbell",
    )


def float_predictions(model, x: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            model(x[start : start + 64], training=False).numpy()
            for start in range(0, len(x), 64)
        ]
    )


def inspect_model(model_bytes: bytes) -> dict:
    from tensorflow.lite.python import schema_py_generated as schema

    model = schema.Model.GetRootAsModel(model_bytes, 0)
    graph = model.Subgraphs(0)
    op_names = {
        value: name
        for name, value in vars(schema.BuiltinOperator).items()
        if isinstance(value, int)
    }
    operators = [
        {
            "name": op_names.get(model.OperatorCodes(i).BuiltinCode(), "UNKNOWN"),
            "version": model.OperatorCodes(i).Version(),
        }
        for i in range(model.OperatorCodesLength())
    ]
    allowed = {"CONV_2D", "RESHAPE", "FULLY_CONNECTED", "SOFTMAX"}
    if any(op["name"] not in allowed for op in operators):
        raise ValueError(
            f"Unexpected operators for the microcontroller model: {operators}"
        )
    for i in range(graph.TensorsLength()):
        if graph.Tensors(i).Type() not in (
            schema.TensorType.INT8,
            schema.TensorType.INT32,
        ):
            raise ValueError("Export contains a tensor that is not INT8 or INT32")

    def tensor_info(index: int) -> dict:
        tensor = graph.Tensors(index)
        quantization = tensor.Quantization()
        if tensor.Type() != schema.TensorType.INT8 or quantization.ScaleLength() != 1:
            raise ValueError("Expected per-tensor int8 model input and output")
        return {
            "shape": tensor.ShapeAsNumpy().tolist(),
            "dtype": "int8",
            "scale": float(quantization.Scale(0)),
            "zero_point": int(quantization.ZeroPoint(0)),
        }

    return {
        "input": tensor_info(graph.Inputs(0)),
        "output": tensor_info(graph.Outputs(0)),
        "operators": operators,
        "model_bytes": len(model_bytes),
        "pico_verified": False,
        "tensor_arena_bytes": None,
    }


class Int8Model:
    def __init__(self, model_bytes: bytes, metadata: dict, *, reference: bool = True):
        self.reference = reference
        if reference:
            from ai_edge_litert.interpreter import Interpreter, OpResolverType

            # XNNPACK can round quantized convolutions differently from TFLite Micro.
            self.model = Interpreter(
                model_content=model_bytes,
                experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
                num_threads=1,
            )
            self.model.allocate_tensors()
            inputs = self.model.get_input_details()
            outputs = self.model.get_output_details()
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError("Expected one model input and output")
            for key, tensor in (("input", inputs[0]), ("output", outputs[0])):
                expected = metadata[key]
                if (
                    tensor["dtype"] != np.int8
                    or expected.get("dtype") != "int8"
                    or tensor["shape"].tolist() != expected.get("shape")
                    or tensor["quantization"]
                    != (expected.get("scale"), expected.get("zero_point"))
                ):
                    raise ValueError(f"Model and metadata disagree: {key}")
            self.input_index = inputs[0]["index"]
            self.output_index = outputs[0]["index"]
        else:
            from ai_edge_litert.compiled_model import CompiledModel

            self.model = CompiledModel.from_buffer(model_bytes)
            self.inputs = self.model.create_input_buffers(0)
            self.outputs = self.model.create_output_buffers(0)
        self.input = metadata["input"]
        self.output = metadata["output"]
        if "labels" in metadata and metadata["labels"] != list(LABELS):
            raise ValueError("Model label order does not match")

    def quantize(self, sample: np.ndarray) -> np.ndarray:
        return np.clip(
            np.rint(sample / self.input["scale"] + self.input["zero_point"]), -128, 127
        ).astype(np.int8)

    def predict(self, x: np.ndarray) -> np.ndarray:
        results = []
        for sample in x:
            quantized = self.quantize(sample)[np.newaxis]
            if self.reference:
                self.model.set_tensor(self.input_index, quantized)
                self.model.invoke()
                output = self.model.get_tensor(self.output_index).astype(np.float32)
            else:
                self.inputs[0].write(quantized)
                self.model.run_by_index(0, self.inputs, self.outputs)
                output = self.outputs[0].read(len(LABELS), np.int8).astype(np.float32)
            results.append(
                (output.reshape(-1) - self.output["zero_point"]) * self.output["scale"]
            )
        return np.stack(results)


def export_model(
    model,
    x: np.ndarray,
    y: np.ndarray,
    pcm: np.ndarray,
    config: Frontend,
    output: Path,
    max_bytes: int,
    channels: tuple[int, int, int],
) -> tuple[Int8Model, dict]:
    import tensorflow as tf

    fixed_model = build_model(config, channels, batch_size=1)
    fixed_model.set_weights(model.get_weights())
    calibration = np.concatenate(
        [np.flatnonzero(y == i)[:100] for i in range(len(LABELS))]
    )
    converter = tf.lite.TFLiteConverter.from_keras_model(fixed_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: (
        [x[index : index + 1]] for index in calibration
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model_bytes = converter.convert()
    info = inspect_model(model_bytes)
    info["reference_runtime"] = "litert_builtin_reference"
    if len(model_bytes) > max_bytes:
        raise ValueError(
            f"Model is {len(model_bytes)} bytes, exceeding budget {max_bytes}"
        )
    runtime = Int8Model(model_bytes, info)
    comparison = calibration[:: max(1, len(calibration) // 30)]
    floating = float_predictions(model, x[comparison])
    quantized = runtime.predict(x[comparison])
    info["quantization_comparison"] = {
        "examples": len(comparison),
        "max_probability_difference": float(np.max(np.abs(floating - quantized))),
        "argmax_agreement": float(
            np.mean(floating.argmax(axis=1) == quantized.argmax(axis=1))
        ),
    }
    if not np.isfinite(quantized).all():
        raise ValueError("Quantized model produced non-finite outputs")
    (output / "model.int8.tflite").write_bytes(model_bytes)
    window, mel = coefficients(config)
    np.savez(output / "frontend.npz", hann=window, mel=mel)
    reference_features = features(pcm[comparison[0]], config)
    np.savez(
        output / "reference.npz",
        pcm=pcm[comparison[0]],
        features=reference_features,
        input_int8=runtime.quantize(reference_features),
        output_probabilities=runtime.predict(reference_features[np.newaxis])[0],
    )
    return runtime, info
