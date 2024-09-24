import os
import gradio as gr
import tensorflow as tf
import numpy as np

# Disable GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load the trained model
model = tf.keras.models.load_model('best_model.keras')

def classify_plant_disease(image):
    try:
        # Convert the image to a tensor and preprocess it
        tensor_image = tf.convert_to_tensor(image)
        resized_image = tf.image.resize(tensor_image, [256, 256])  # Adjust to match your model's input size
        normalized_image = resized_image / 255.0  # Normalize the pixel values
        input_image = tf.expand_dims(normalized_image, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(input_image)
        class_labels = ['Healthy', 'Powdery', 'Rust']

        # Get the predicted class and confidence
        predicted_class_idx = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_idx]
        confidence_score = np.round(predictions[0][predicted_class_idx] * 100, 3)

        return f"🌱 Predicted: {predicted_class} | Confidence: {confidence_score}%"
    
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Example images for demonstration
demo_images = ["Healthy.png", "Powdery.png", "Rust.png"]

# Create a Gradio interface
interface = gr.Interface(
    fn=classify_plant_disease,
    inputs=gr.Image(),  # Use numpy type for image input
    outputs="text",
    title="🌿 Plant Disease Detection",
    description="This model detects the health status of plants, identifying conditions like Powdery Mildew and Rust. <br> \
                 Trained using a Convolutional Neural Network, it evaluates images uploaded by users to provide classifications.",
    examples=demo_images,
    theme="monochrome"  # You can change this to a different theme
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch(share=True)

