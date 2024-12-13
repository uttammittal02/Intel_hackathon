# **Skin Cancer Detection**

This project implements a deep learning pipeline using a Vision Transformer (ViT) for detecting skin cancer. The model is trained and evaluated on the **ISIC dataset**, a benchmark dataset for skin cancer research published in recent years. The system is designed to provide accurate predictions through an easy-to-use **Gradio-based interface**.

---

## **Table of Contents**

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Setup Instructions](#setup-instructions)
4. [Model Details](#model-details)
5. [Usage](#usage)
6. [Future Work](#future-work)
7. [Acknowledgments](#acknowledgments)

---

## **Project Overview**

Skin cancer is one of the most common types of cancer, and early detection is crucial for effective treatment. This project leverages the power of Vision Transformers (ViTs) to accurately classify skin lesions into malignant and benign categories. The model is designed to run on **Intel CPUs** with optimizations for efficient processing.

The application is containerized using **Docker**, allowing seamless deployment and usage on any compatible system.

---

## **Features**

- **Deep Learning Model**: Uses Vision Transformer (ViT-Tiny) for skin lesion classification.
- **Dataset**: Trained on the ISIC dataset, known for its comprehensive and high-quality images.
- **User Interface**: A clean and intuitive Gradio web-based interface for uploading images and viewing predictions.
- **CPU Optimized**: Designed to run efficiently on systems with Intel CPUs.
- **Containerized Deployment**: Easily deploy the application using Docker.

---

## **Setup Instructions**

### **Pre-requisites**
Ensure the following are installed and set up on your system:
- **Docker**: [Install Docker](https://docs.docker.com/get-docker/) v 26.1.4
- **Python** (optional for testing the notebook).

### **Steps to Run the Application**
1. **Navigate to the `docker_setup` Directory**:
   Open a terminal and change to the `docker_setup` directory of the project.

2. **Build the Docker Image**:
   Run the following command to build the Docker image:
    ```bash
    docker build -t gradio-app .
    ```

3. **Run the Docker Container**: 
Start the application by running:

    ```bash
    docker run -p 7860:7860 gradio-app
    ```
    
4. **Access the Application**: Open your web browser and go to:

    ```bash
    http://localhost:7860
    ```
You will see the Gradio-based user interface to upload images and view predictions.
It's working demo can be seen [here](https://youtu.be/PXS2YCD3_Cs)

## **Model Details**

The project employs a **Vision Transformer (ViT-Tiny)** model for classification. The model processes input images of size **224x224** and is trained using the ISIC dataset. The details of the model architecture, preprocessing, training, and evaluation can be found in the **`vit_tiny.ipynb`** file located in the project repository.

The model leverages state-of-the-art deep learning techniques and has been optimized for **Intel CPUs** using the Intel AI Toolkit.


## **Usage**

1. Start the application using the steps outlined in the [Setup Instructions](#setup-instructions).
2. Upload an image of a skin lesion using the Gradio interface.
3. View the prediction results, including the probabilities for each class (e.g., malignant or benign).

---

## **Presentation for the Hackathon**

[INTC AI.Hackathon.2024 - vPitch.Presentation.Template.pptx (1).pdf](https://github.com/user-attachments/files/18129150/INTC.AI.Hackathon.2024.-.vPitch.Presentation.Template.pptx.1.pdf)


---

## **Future Work**

- Add support for GPU acceleration for faster inference on larger datasets.
- Implement additional optimizations for running on non-Intel CPUs.
- Extend the pipeline to include explainability (e.g., Grad-CAM visualizations) for better interpretability of predictions.
- Train on augmented datasets for improved generalization.

---

## **Acknowledgments**

- **ISIC Dataset**: [International Skin Imaging Collaboration Dataset](https://isic-archive.com/)  
  A comprehensive collection of annotated images for skin cancer research.
- **Vision Transformer**: The ViT model implementation is inspired by [Google Research](https://github.com/google-research/vision_transformer).
- **Gradio**: [Gradio](https://gradio.app/) provides the interactive web interface.

---

## **Contributions**

Contributions, issues, and feature requests are welcome!  
Feel free to fork this repository and submit pull requests.
