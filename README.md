🫁 Pneumonia Detection Using Chest X-Rays
This project is a Deep Learning-based web application that classifies Chest X-ray images into two categories: Normal and Pneumonia. It uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras and is served via a Flask web interface.
🚀 Features
Automated Training Pipeline: main.py handles data loading, preprocessing (grayscale, 150x150 resizing), and model training.
Web Interface: app.py provides a user-friendly UI to upload X-ray images and get instant predictions.
Data Augmentation: Implemented to improve model generalization and prevent overfitting.
📂 Project Structure
pnemonia detection/
├── chest_xray/
│   ├── app.py             # Flask Web Server
│   ├── main.py            # Model Training Script
│   ├── templates/         # UI (HTML/CSS)
│   └── (dataset folders)  # Train/Test/Val (Not on GitHub)
├── pneumonia_model.h5     # Trained Model File (Not on GitHub)
├── .gitignore             # Excludes heavy files from Git
└── README.md              # Project Documentation
📊 Dataset
The model was trained on the Kaggle Chest X-Ray Images (Pneumonia) dataset.
To run this project locally, download the dataset and place it in the chest_xray/ folder.
🛠️ Installation & Setup
Clone the repository:
code
Bash
git clone https://github.com/Pamursahasra22/Pneumonia-Detection-Using--Chest-X-rays.git
cd "pnemonia detection"
Create a Virtual Environment:
code
Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:
code
Bash
pip install flask tensorflow opencv-python numpy pillow
⚡ How to Run
1. Training the Model
If you want to retrain the model on the dataset:
code
Bash
python chest_xray/main.py
This will generate the pneumonia_model.h5 file.
2. Launching the Web App
To start the detection interface:
code
Bash
python chest_xray/app.py
Open your browser and navigate to http://127.0.0.1:5000.
📈 Model Performance
Target Size: 150x150 (Grayscale)
Optimizer: RMSprop
Loss: Binary Crossentropy
Callbacks: ReduceLROnPlateau implemented for optimized learning rates.
