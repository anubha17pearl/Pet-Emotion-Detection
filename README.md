# Pet-Emotion-Detection
Pet Animal emotion detections are important to analyze for human connect. So, an architecture is introduced to analyze the emotions using the recorded/extracted audios
Pet Audio Emotion Detection
Detects and classifies emotions of dogs and cats from audio signals using MFCC feature extraction and a stacked LSTM network.

Folder Structure
pet-audio-emotion-detection/
├── merged_dataset/
│   ├── dog_angry/        # angry 1.mp3 ... angry 100.mp3
│   ├── dog_sad/          # sad 1.mp3 ... sad 100.mp3
│   ├── cat_Angry/
│   ├── cat_Defense/
│   ├── cat_Fighting/
│   ├── cat_Paining/
│   └── cat_HuntingMind/
├── mergedData.py         # Main training script
├── merged_model.keras    # Saved model
├── merged_model.h5       # Model weights
├── dog_model.json        # Model architecture
└── requirements.txt
Python Libraries Required
Python • TensorFlow/Keras • Librosa • NumPy • Scikit-learn • Matplotlib

Steps to Run

Clone the repository

git clone https://github.com/your-username/pet-audio-emotion-detection.git
   cd pet-audio-emotion-detection

Install dependencies

pip install -r requirements.txt

Download the dataset and place audio files inside merged_dataset/
Update the dataset path in mergedData.py (line: os.chdir(...))
Run the training script

python mergedData.py

Model evaluates automatically — prints Accuracy, Precision, Recall, F1, Confusion Matrix

Author
Anubha Pearline
