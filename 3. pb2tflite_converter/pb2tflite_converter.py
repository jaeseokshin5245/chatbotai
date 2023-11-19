import tensorflow as tf

saved_model_dir = "c:\\Users\\jaese\\Desktop\\tf_model"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open(r'c:\Users\jaese\Desktop\converter\firststep\result\firststep_converted_model.tflite', 'wb').write(tflite_model)