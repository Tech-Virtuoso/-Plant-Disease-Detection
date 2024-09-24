Here's the updated **README.md** with additional information about using **Gradio** for the web interface:

---

# 🌿 Plant Disease Detection

## 🌱 Project Overview
This project focuses on detecting the health status of plants. The model is trained using a **Convolutional Neural Network (CNN)** to classify plant diseases. It identifies conditions like **Powdery Mildew** and **Rust** from images uploaded by users and provides the appropriate classification.

### ✨ Key Features:
- 🌾 **Classifies plant images** into one of three categories: **Healthy**, **Powdery Mildew**, or **Rust**.
- 🛠️ Built using **TensorFlow** and trained on a dataset of plant images.
- 📷 User-friendly web interface powered by **Gradio**, allowing users to easily upload images for disease detection.

## 📊 Dataset
The dataset used for training the model can be found on **Kaggle**. It includes labeled images of plants affected by different diseases and healthy plants.

- [Kaggle Dataset: Plant Disease Detection](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

## 🧠 Model Details
The model architecture consists of a **CNN** built using **TensorFlow** and **Keras**. It has been trained to detect patterns in images that correlate with different plant diseases.

## 🚀 Installation

To run this project on your local machine, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/Tech-Virtuoso/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```

### 2. Set Up a Conda Environment
Create a **conda** environment using the following command:
```bash
conda create --name plant-disease-detection python=3.8
conda activate plant-disease-detection
```

### 3. Install Required Libraries
Once the environment is activated, install the required libraries by running:
```bash
pip install -r requirements.txt
```

### 4. Running the Application with Gradio
This project uses **Gradio** to create an easy-to-use web interface where users can upload images and get predictions. 

Run the application by executing the following command:
```bash
python app.py
```

This will start the Gradio interface, which will open in your web browser automatically.

### 5. Gradio Interface Overview
- 🖼️ **Upload an Image**: Users can upload an image of a plant for disease detection.
- 🔍 The model will process the image and display the predicted health status of the plant, whether it is **Healthy**, **Powdery Mildew**, or **Rust**.

## 🌐 How to Use
1. Start the Gradio interface using the above command.
2. Upload an image of a plant.
3. The model will evaluate the image and return a classification of the plant's health status.

## 🔧 Gradio Settings
This project uses the **monochrome theme** in the Gradio interface for a clean and minimalist look. If you'd like to modify the Gradio interface further, check the [Gradio documentation](https://www.gradio.app/).

---

### 📝 Additional Notes
Feel free to contribute to this project by opening issues or submitting pull requests. Your feedback is valuable!

---
