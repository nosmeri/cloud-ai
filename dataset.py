import os, json, glob
from pathlib import Path
from collections import defaultdict

IMG_DIR = "dataset/img"  # 이미지들
ANN_DIR = "dataset/ann"  # 이미지별 .json (이름 동일 가정: 0001.jpg <-> 0001.json)
OUT_CSV = "labels.csv"

# ✅ 구름 클래스 이름(데이터셋에 실제 있는 구름 클래스들을 여기에 추가)
CLOUD_CLASS_TITLES = {
    "Cirriformes": "cirriformes",
    "Cumuliformes": "cumuliformes",
    "Estratiformes": "estratiformes",
    "Estratocumuliformes": "estratocumuliformes",
}


def polygon_area(points):
    # points: [[x,y], [x,y], ...]
    if len(points) < 3:
        return 0.0
    s = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


# 이미지 기준으로 매칭되는 json 찾기
def find_json_for_image(img_path):
    stem = Path(img_path).stem
    cand = Path(ANN_DIR) / f"{stem}.jpg.json"
    return str(cand) if cand.exists() else None


def main():
    img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "**", "*.*"), recursive=True))
    rows = []
    missing_json = 0
    no_cloud = 0

    for ip in img_paths:
        jp = find_json_for_image(ip)
        if not jp:
            missing_json += 1
            continue
        with open(jp, "r", encoding="utf-8") as f:
            ann = json.load(f)

        # 라벨 후보: 클래스별 면적 합
        area_by_label = defaultdict(float)

        for obj in ann.get("objects", []):
            cls = obj.get("classTitle", "")
            if cls not in CLOUD_CLASS_TITLES:
                continue
            if obj.get("geometryType") != "polygon":
                continue
            exterior = obj.get("points", {}).get("exterior", [])
            if not exterior:
                continue
            area_by_label[CLOUD_CLASS_TITLES[cls]] += polygon_area(exterior)

        if not area_by_label:
            no_cloud += 1
            label = "unknown"  # 학습에서 제외 권장
        else:
            # 면적 최대 라벨 선택
            label = max(area_by_label.items(), key=lambda x: x[1])[0]

        rows.append(f"{ip},{label}")

    with open(OUT_CSV, "w", encoding="utf-8") as out:
        out.write("image_path,label\n")
        out.write("\n".join(rows))

    print(f"[DONE] wrote {OUT_CSV}")
    print(f" - images: {len(img_paths)}")
    print(f" - missing_json: {missing_json}")
    print(f" - no_cloud(unknown): {no_cloud}")


if __name__ == "__main__":
    main()
