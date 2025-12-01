# 1. 필요 라이브러리
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt

# 2. 기본값 설정
data_dir = "data"              
img_size = (224, 224)         
batch_size = 32                

# 3. 데이터셋 로드
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_dir, "train"),
    image_size=img_size,
    batch_size=batch_size,
    label_mode="binary"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_dir, "val"),
    image_size=img_size,
    batch_size=batch_size,
    label_mode="binary"
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_dir, "test"),
    image_size=img_size,
    batch_size=batch_size,
    label_mode="binary",
    shuffle=False
)

# 4. 데이터 파이프라인 최적화
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# 5. 데이터 증강 & 전처리 정의
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

preprocess_input = tf.keras.applications.efficientnet.preprocess_input

# 6. CNN 기반 모델 구축축
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    input_shape=img_size + (3,),
    weights="imagenet"
)
base_model.trainable = False   # 처음에는 backbone 고정

inputs = keras.Input(shape=img_size + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

# 7. 모델 컴파일
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
)

# 8. 모델 학습
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# 9. 테스트셋으로 평가 
test_loss, test_acc, test_prec, test_rec = model.evaluate(test_ds)
print(f"Test loss:      {test_loss:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")
print(f"Test precision: {test_prec:.4f}")   # diet(1) 기준
print(f"Test recall:    {test_rec:.4f}")   # diet(1) 기준

# 10. 모델 저장
model.save("diet_cnn_model.h5")
print("모델 저장 완료: diet_cnn_model.h5")

# 11. 학습 곡선 출력
# accuracy curve 
plt.figure()
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy_curve.png")

# loss_curve
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_curve.png")

# precision curve
plt.figure()
plt.plot(history.history["precision"], label="train_precision")
plt.plot(history.history["val_precision"], label="val_precision")
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.legend()
plt.savefig("precision_curve.png")

# recall curve
plt.figure()
plt.plot(history.history["recall"], label="train_recall")
plt.plot(history.history["val_recall"], label="val_recall")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.legend()

plt.savefig("recall_curve.png")
