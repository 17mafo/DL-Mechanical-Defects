import tf2onnx, tensorflow as tf
model = tf.keras.models.load_model('modelname.h5') # replace with model path
model.save('model_tf', save_format='tf')
tf2onnx.convert.from_saved_model('model_tf', output_path='model.onnx')