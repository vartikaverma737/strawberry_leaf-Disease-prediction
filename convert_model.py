import tensorflow as tf

# Load your existing H5 model
model = tf.keras.models.load_model("model.h5")

# Export to SavedModel format (compatible with TFLite, TF Serving, etc.)
model.export("optimized_model")

print("✅ Model has been exported successfully to 'optimized_model/'")
