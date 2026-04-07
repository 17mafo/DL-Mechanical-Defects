import tf2onnx
import tensorflow as tf
import onnx
import os

class BinaryF1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision_metric = tf.keras.metrics.Precision(thresholds=threshold)
        self.recall_metric = tf.keras.metrics.Recall(thresholds=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision_metric.update_state(y_true, y_pred, sample_weight)
        self.recall_metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision_metric.result()
        recall = self.recall_metric.result()
        return 2.0 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision_metric.reset_state()
        self.recall_metric.reset_state()

path = "C:\\Users\\Marti\\Documents\\DL-Mechanical-Defects\\model_training\\saved_models"

for model_filename in os.listdir(path):
    if model_filename.endswith(".keras"):
        model_name = model_filename.replace(".keras", "")
        model_path = os.path.join(path, model_filename)
        onnx_output_path = os.path.join(path, f"{model_name}.onnx")

        print(f"Converting: {model_filename}")

        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"BinaryF1Score": BinaryF1Score}
        )

        # Handle both single and multiple inputs
        inputs = model.inputs
        input_signature = [
            tf.TensorSpec(inp.shape, tf.float32, name=inp.name)
            for inp in inputs
        ]

        print(f"Input shapes: {[s.shape for s in input_signature]}")

        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)
        onnx.save(onnx_model, onnx_output_path)

        print(f"Saved ONNX model to: {onnx_output_path}")