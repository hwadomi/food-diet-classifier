import os
import shutil

# 원본 Food-11 경로
RAW_DIR = os.path.join("data_raw", "Food-11")

# 최종 출력 경로
OUT_DIR = "data"

# === 1) 클래스 이름을 diet / not_diet 로 나누기 ===
diet_classes = {
    "Egg",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
}

not_diet_classes = {
    "Bread",
    "Dairy product",
    "Dessert",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
}

#   training   → train
#   validation → val
#   evaluation → test
split_map = {
    "training": "train",
    "validation": "val",
    "evaluation": "test",
}


def prepare_split(split_name_raw: str, split_name_out: str) -> None:
   
    src_split_dir = os.path.join(RAW_DIR, split_name_raw)

    # 폴더 0_not_diet / 1_diet 으로 생성
    not_diet_dir = os.path.join(OUT_DIR, split_name_out, "0_not_diet")
    diet_dir = os.path.join(OUT_DIR, split_name_out, "1_diet")

    os.makedirs(diet_dir, exist_ok=True)
    os.makedirs(not_diet_dir, exist_ok=True)

    # 11개 클래스 폴더(Bread, Egg, ...)를 돎
    for class_folder in os.listdir(src_split_dir):
        class_path = os.path.join(src_split_dir, class_folder)
        if not os.path.isdir(class_path):
            continue  # 파일이 아니라 폴더만 대상

        # 이 클래스가 diet 그룹인지 not_diet 그룹인지 결정
        if class_folder in diet_classes:
            target_root = diet_dir
        elif class_folder in not_diet_classes:
            target_root = not_diet_dir
        else:
            print(f"[경고] 어느 그룹에도 속하지 않는 클래스 폴더: {class_folder}")
            continue

        # 해당 클래스 폴더 안에 있는 이미지들을 복사
        for fname in os.listdir(class_path):
            # 이미지 파일만 대상
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            src_path = os.path.join(class_path, fname)

            # 혹시 이름이 겹쳐도 구분되도록 앞에 클래스 이름을 붙여줌
            new_name = f"{class_folder}_{fname}"
            dst_path = os.path.join(target_root, new_name)

            shutil.copy2(src_path, dst_path)


if __name__ == "__main__":
    for raw_name, out_name in split_map.items():
        print(f"Processing {raw_name}  -->  {out_name}")
        prepare_split(raw_name, out_name)

    print("완료! data/train, data/val, data/test 안에 0_not_diet / 1_diet 이미지가 생성되었습니다.")
