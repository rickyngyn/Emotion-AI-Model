# Emotion Detection AI

This project uses a AI to recognize human facial emotions from grayscale images. It is trained on the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) and can perform real-time emotion recognition using your webcam.


## Project Setup

1. **Clone repo**
   ```bash
   git clone https://github.com/rickyngyn/Emotion-AI-Model.git
   cd Emotion-AI-Model

2. **Setup environment and install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt


## Model Training

1. **Download FER 2013 Dataset and run create_fer2013_csv.py**
   Images are resized to 48x48 grayscale and saved in 'fer2013.csv'

2. **Train the model**
   ```bash
   python train_model.py

3. **Run the application**
   ```bash
   python real_time_detector.py



