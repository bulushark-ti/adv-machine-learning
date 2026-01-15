--------------------------------------------------------------
Background
--------------------------------------------------------------
VQA-RAD: Visual Question Answering on Radiology Images
This project explores Visual Question Answering (VQA) in the medical domain using the VQA-RAD dataset. It compares two deep learning approaches: a traditional custom CNN-LSTM architecture and a modern ViLT (Vision-and-Language Transformer) model.

Disclaimer    : This is for education purposes only, NOT Design for final use case.
Version Code  : VUM_AL_1.1
Producer      : Gan Jian Jie (23065592) | Lim Hong Yao (24206969)

--------------------------------------------------------------
File Overview
--------------------------------------------------------------
The repository consists of three Jupyter Notebooks representing different stages of the experiment:

1. AML_CNN_PreliminaryExperiment.ipynb
Purpose: Initial data exploration and baseline modeling.
Key Features:
Loads the VQA-RAD Public dataset and corresponding images.
Performs basic data cleaning (e.g., handling missing answers).
Establishes a baseline for Closed-Ended VQA tasks.
Visualizes training and validation accuracy/loss.

2. AML_CNN_Finetune_Model.ipynb
Purpose: A refined and robust implementation of the custom CNN-LSTM architecture.
Key Features:
Reproducibility: Sets seeds for Python, NumPy, and TensorFlow to ensure consistent results.
Preprocessing: Sophisticated handling of "Yes/No" normalization, answer type separation (Open vs. Closed), and image resizing (224x224).
Architecture:
Image Encoder: A custom Convolutional Neural Network (CNN) with Conv2D and MaxPooling2D layers.
Text Encoder: An Embedding layer followed by an LSTM network to process questions.
Fusion: Concatenates image and text features before passing them to a dense classification layer.
Training: Implements EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau for optimized training.

3. AML_ViLT_Pre_and_Fine_Tuning_Code.ipynb
Purpose: Implementation of a State-of-the-Art (SOTA) Transformer model for comparison.
Key Features:
Model: Uses the ViltForQuestionAnswering architecture from Hugging Face Transformers, pre-trained on dandelin/vilt-b32-mlm.
Pipeline: leverages ViltProcessor for handling multimodal inputs (images + text).
Training: Includes a training loop (using PyTorch AdamW optimizer) to fine-tune the transformer on the VQA-RAD dataset.
Visualization: Plots the learning curve to assess feasibility.

--------------------------------------------------------------
Dataset
--------------------------------------------------------------
This project uses the VQA-RAD dataset, which consists of clinical images (X-ray, CT, MRI) and question-answer pairs divided into "Open" and "Closed" categories.
Images: Located in /VQA_RAD_Image_Folder.
Metadata: Located in VQA_RAD_Dataset_Public.xlsx.

Remark: Please Download in Kaggle before to run the script!!!
Link: https://www.kaggle.com/datasets/shashankshekhar1205/vqa-rad-visual-question-answering-radiology

--------------------------------------------------------------
Tech Stack & Requirements
--------------------------------------------------------------
The notebooks are designed to run in Google Colab.
Key Libraries:
Deep Learning: TensorFlow/Keras (for CNN models), PyTorch (for ViLT), Transformers (Hugging Face).
Data Processing: Pandas, NumPy, OpenCV.
Visualization: Matplotlib.

--------------------------------------------------------------
Usage Instructions
--------------------------------------------------------------
1. Environment Setup:
Open the notebooks in Google Colab.
Ensure the Runtime type is set to GPU (specifically required for the ViLT notebook)

2. Data Mounting:
Based on your preference, download mount to drive, or else, download from kaggle & drop into Colab Files.

3. Execution:
Run AML_CNN_PreliminaryExperiment.ipynb first to understand the data structure.
Run AML_CNN_Finetune_Model.ipynb to train the custom CNN-LSTM model.
Run AML_ViLT_Pre_and_Fine_Tuning_Code.ipynb to fine-tune the Transformer model.

END
Good Luck.
