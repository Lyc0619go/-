mport numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout

# 1. Load and preprocess the dataset
# Update file path to local disk
data = pd.read_csv('C:/dataset/diabetes.csv')

# Features and labels
X = data.drop(columns=['Outcome']).values
y = data['Outcome'].values

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape features to simulate "image-like" data for CNN
X_reshaped = X_scaled.reshape(-1, 8, 1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42, stratify=y)

# 2. Build the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 1), activation='relu', input_shape=(8, 1, 1)),
    MaxPooling2D(pool_size=(2, 1)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# 3. Save the model
model.save('cnn_diabetes_model.h5')

# 4. Fine-tuning with the pre-trained model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GlobalAveragePooling2D

# Load the saved model
pretrained_model = load_model('cnn_diabetes_model.h5')

# Add new layers for fine-tuning
fine_tune_model = Sequential([
    pretrained_model,
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Final output layer
])