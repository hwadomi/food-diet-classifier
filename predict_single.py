import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os


# 1. 모델 구조 정의 함수
img_size = (224, 224)

def build_model():
    # 데이터 증강 (학습 때와 동일 구조로 맞춤)
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=img_size + (3,),
        weights="imagenet"
    )
    base_model.trainable = False  # main.py와 동일하게 freezing

    inputs = keras.Input(shape=img_size + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    return model


# 2. 모델 생성 & 가중치 로드
model = build_model()
model.load_weights("diet_cnn_weights.h5")   # main.py에서 저장한 가중치 파일

# 3. 파일 확장자 후보
EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

def find_image_file(filename):
    """확장자 없이 파일 이름만 입력해도 자동으로 이미지 파일을 찾아줌."""
    # 사용자가 확장자를 직접 입력한 경우
    if os.path.exists(filename):
        return filename

    # 확장자가 없으면 자동으로 붙여서 확인
    for ext in EXTENSIONS:
        candidate = filename + ext
        if os.path.exists(candidate):
            return candidate

    # 못 찾은 경우
    return None


# 4. 이미지 로드 & 전처리
def load_and_preprocess_image(img_path):
    img = keras.utils.load_img(img_path, target_size=img_size)
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array


# 5. 예측 함수
def predict_image(img_path):
    x = load_and_preprocess_image(img_path)
    pred = model.predict(x)[0][0]

    label = "diet (다이어트 식단)" if pred > 0.5 else "not_diet (비다이어트 식단)"

    print(f"\n이미지: {img_path}")
    print(f"예측 확률: {pred:.4f}")
    print(f"예측 라벨: {label}\n")


# 6. 메인 실행부
if __name__ == "__main__":
    user_input = input("이미지 파일 이름을 입력하세요 (확장자 생략 가능): ").strip()

    img_file = find_image_file(user_input)

    if img_file is None:
        print("\n❌ 오류: 해당 파일을 찾을 수 없습니다.")
        print("파일이 존재하는지 확인해주세요.")
    else:
        predict_image(img_file)
