import pandas as pd
import librosa
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def extract_features(file):
    audio, sr = librosa.load(file, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

paths = []
labels = []
extracted_features = []

os.chdir('D:/vit_2024_2025/research/ashley/ashley_Cap_paper/code/merged_dataset')

for i in range(0, 100):
    filename = "dog_angry/angry " + str(i+1) + ".mp3"
    print(i)
    label = "dog_angry"
    data = extract_features(filename)
    extracted_features.append([data, label])

for i in range(0, 100):
    filename = "dog_sad/sad " + str(i+1) + ".mp3"
    print(i)
    label = "dog_sad"
    data = extract_features(filename)
    extracted_features.append([data, label])
for i in range(0, 100):
    filename = "cat_Angry/angry " + str(i+1) + ".mp3"
    print(i)
    label = "cat_angry"
    data = extract_features(filename)
    extracted_features.append([data, label])
for i in range(0, 100):
    filename = "cat_Defense/defense " + str(i+1) + ".mp3"
    print(i)
    label = "angry"
    data = extract_features(filename)
    extracted_features.append([data, label])
for i in range(0, 100):
    filename = "cat_Fighting/fighting " + str(i+1) + ".mp3"
    print(i)
    label = "cat_Fighting"
    data = extract_features(filename)
    extracted_features.append([data, label])
for i in range(0, 100):
    filename = "cat_Paining/paining " + str(i+1) + ".mp3"
    print(i)
    label = "cat_Paining"
    data = extract_features(filename)
    extracted_features.append([data, label])
for i in range(0, 100):
    filename = "cat_HuntingMind/hunting mind " + str(i+1) + ".mp3"
    print(i)
    label = "cat_HuntingMind"
    data = extract_features(filename)
    extracted_features.append([data, label])

ex_ft_df = pd.DataFrame(extracted_features, columns=['features', 'class'])
X = np.array(ex_ft_df['features'].tolist())
y = np.array(ex_ft_df['class'].tolist())

labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Reshape the input data to match the expected shape for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(LSTM(units=128, input_shape=(X_train.shape[1], 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=True))  # Additional LSTM layer
model.add(Dropout(0.2))
model.add(LSTM(units=32))  # Additional LSTM layer
model.add(Dropout(0.2))
model.add(Dense(units=y_train.shape[1], activation='softmax'))

from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

# Convert predictions and true labels back to categorical values
y_pred_categorical = labelencoder.inverse_transform(np.argmax(y_pred, axis=1))
y_test_categorical = labelencoder.inverse_transform(np.argmax(y_test, axis=1))

# Calculate and print additional performance metrics
accuracy = accuracy_score(y_test_categorical, y_pred_categorical)
precision = precision_score(y_test_categorical, y_pred_categorical, average='weighted')
recall = recall_score(y_test_categorical, y_pred_categorical, average='weighted')
f1 = f1_score(y_test_categorical, y_pred_categorical, average='weighted')
conf_matrix = confusion_matrix(y_test_categorical, y_pred_categorical)
compiled_loss = model.compiled_loss
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

train_loss = history.history['loss']
avg_train_loss = sum(train_loss) / len(train_loss)
print(f"Average Train Loss: {avg_train_loss:.4f}")

print(f"Accuracy Score: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)
'''
def predict_emotion(file):
    new_data = extract_features(file)
    new_data = new_data.reshape(1, new_data.shape[0], 1)
    predicted_emotion = model.predict(new_data)
    predicted_emotion_label = labelencoder.inverse_transform(np.argmax(predicted_emotion, axis=1))
    return predicted_emotion_label[0]

test_audio = "C:/Users/hp/OneDrive/VIT/SEM 8 CAPSTONE/project/dataset/test.mp3"
predicted_emotion = predict_emotion(test_audio)
print(f"Predicted emotion: {predicted_emotion}")
'''
import matplotlib.pyplot as plt

# # Plot accuracy vs loss graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# # Plot accuracy vs test and train loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

current_directory = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the script's directory
os.chdir(current_directory)

model.save('merged_model.keras')
model_json = model.to_json()
with open("dog_model.json", "w") as json_file:
     json_file.write(model_json)
# # serialize weights to HDF5
model.save_weights("merged_model.h5")