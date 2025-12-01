# 1. 필요 라이브러리
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 2. 기본 설정 & 학습된 모델 불러오기
img_size = (224, 224)
model = keras.models.load_model("diet_cnn_model.h5")

# 3. 이미지 파일 전처리 함수
def load_and_preprocess_image(img_path):
    img = keras.utils.load_img(img_path, target_size=img_size)
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # (H, W, C) -> (1, H, W, C)

    return img_array

# 4. 예측 함수: diet / not_diet 분류
def predict_image(img_path):
    x = load_and_preprocess_image(img_path)
    pred = model.predict(x)[0][0]  # sigmoid 출력 (0~1)
    
    if pred > 0.5:
        label = "diet (다이어트 식단)"
    else:
        label = "not_diet (비다이어트 식단)"
    
    print(f"이미지: {img_path}")
    print(f"예측 확률: {pred:.4f}")
    print(f"예측 라벨: {label}")

# 5. 메인 함수
if __name__ == "__main__":
    # 여기에 테스트해보고 싶은 이미지 경로 넣으면 됩니다
    test_image_path = "./hamburger.png"
    predict_image(test_image_path)

