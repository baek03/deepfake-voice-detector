import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# 1. 데이터 디렉토리 설정
# 진짜 목소리와 가짜 목소리 스펙트로그램 이미지를 저장한 폴더 경로를 설정하세요
real_spectrogram_dir = r'C:\Users\bkbk1\OneDrive\바탕 화면\인공지능 기초_발표\인공지능 기초_작품\spectrogram\REAL'
fake_spectrogram_dir = r'C:\Users\bkbk1\OneDrive\바탕 화면\인공지능 기초_발표\인공지능 기초_작품\spectrogram\FAKE'

# 2. 이미지 데이터를 로드하고 라벨을 지정
image_data = []
labels = []

for category, spectrogram_dir in zip(['REAL', 'FAKE'], [real_spectrogram_dir, fake_spectrogram_dir]):
    for img_file in os.listdir(spectrogram_dir):
        if img_file.endswith('.png'):
            img_path = os.path.join(spectrogram_dir, img_file)
            img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
            img_array = tf.keras.utils.img_to_array(img) / 255.0  # 이미지 정규화
            image_data.append(img_array)
            labels.append(category)

image_data = np.array(image_data)
labels = np.array(labels)

# 3. 레이블 인코딩
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# 4. 데이터셋 분할 (학습용 및 테스트용 데이터)
X_train, X_test, y_train, y_test = train_test_split(image_data, labels_categorical, test_size=0.2, random_state=42)

# 5. CNN 모델 정의
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 출력 레이어: 2개의 클래스 (REAL과 FAKE)
])

# 6. 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 7. 모델 학습
history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_split=0.2, verbose=2)

# 8. 모델 평가
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy:.2f}')
print(f'Test Loss: {test_loss:.2f}')

# 9. 학습 결과 시각화
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
