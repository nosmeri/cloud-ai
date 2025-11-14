import os
import glob
import torch
from torchvision import transforms, models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

# --------------------
# 경로/상수
# --------------------
CKPT_PATH = "runs_cls/best.pt"
IMG_SIZE = 384
GALLERY_DIR = "assets"  # 예시 이미지 폴더
device = torch.device("cpu")

# --------------------
# 모델 로드
# --------------------
ckpt = torch.load(CKPT_PATH, map_location=device)
classes = ckpt[
    "classes"
]  # ["cirriformes", "cumuliformes", "estratiformes", "estratocumuliformes"]

model = models.efficientnet_b0(weights=None)
in_f = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_f, len(classes))
model.load_state_dict(ckpt["model"], strict=True)
model.eval().to(device)

# --------------------
# 전처리
# --------------------
tf = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# --------------------
# 4개 그룹 한글 이름
# --------------------
KOR_LABELS = {
    "cirriformes": "권운형 구름",
    "cumuliformes": "적운형 구름",
    "estratiformes": "층운형 구름",
    "estratocumuliformes": "층적운형 구름",
}

# --------------------
# 그룹 설명
# --------------------
GROUP_DESC = {
    "cirriformes": "상층(약 6~13 km)에서 얼음 결정으로 이루어진 매우 얇고 섬유질 모양의 구름입니다.",
    "cumuliformes": "대류로 인해 솟아오른 솜뭉치 모양의 구름으로, 수직으로 크게 발달할 수 있습니다.",
    "estratiformes": "수평으로 넓게 퍼져 하늘을 커튼처럼 덮는 구름으로, 흐리고 비 오는 날과 관련이 깊습니다.",
    "estratocumuliformes": "층운처럼 퍼져 있으면서도 적운처럼 덩어리 구조를 가진 구름입니다.",
}

# --------------------
# 하위 구름(종) 정보
# id는 폴더 이름과 동일해야 함
# --------------------
SUBTYPES = {
    "cirriformes": [
        {
            "id": "cirrus",
            "ko": "새털구름",
            "en": "Cirrus",
            "desc": "매우 가늘고 실 같은 줄기 모양으로 퍼진 상층운입니다. 주로 맑은 날 상공에 나타나며, 날씨 변화의 전조가 되기도 합니다.",
        },
        {
            "id": "cirrocumulus",
            "ko": "비늘구름",
            "en": "Cirrocumulus",
            "desc": "작고 하얀 점무늬 또는 물결무늬가 하늘을 덮는 구름으로, 비늘이나 양털처럼 보입니다.",
        },
        {
            "id": "cirrostratus",
            "ko": "흰막구름",
            "en": "Cirrostratus",
            "desc": "얇은 흰 막처럼 하늘 전반을 덮는 상층운으로, 해무리·달무리를 만드는 원인이 됩니다.",
        },
    ],
    "cumuliformes": [
        {
            "id": "cumulus",
            "ko": "뭉게구름",
            "en": "Cumulus",
            "desc": "밝고 하얀 솜뭉치 모양의 구름으로, 주로 맑은 날 낮에 볼 수 있습니다.",
        },
        {
            "id": "towering_cumulus",
            "ko": "탑적운",
            "en": "Towering Cumulus",
            "desc": "수직으로 크게 발달 중인 적운으로, 더 자라면 소나기구름(적란운)으로 변할 수 있습니다.",
        },
        {
            "id": "cumulonimbus",
            "ko": "소나기구름",
            "en": "Cumulonimbus",
            "desc": "거대한 기둥이나 산처럼 솟아오른 구름으로, 번개·천둥·소나기를 동반합니다.",
        },
    ],
    "estratiformes": [
        {
            "id": "stratus",
            "ko": "층운",
            "en": "Stratus",
            "desc": "낮은 하늘에서 회색 커튼처럼 퍼져 있는 구름으로, 이슬비나 가랑비를 내리기도 합니다.",
        },
        {
            "id": "altostratus",
            "ko": "고층운",
            "en": "Altostratus",
            "desc": "중간 높이에 나타나는 회색 구름층으로, 해가 희미하게 비칠 정도로 두껍습니다. 비나 눈이 오기 전 나타나는 경우가 많습니다.",
        },
        {
            "id": "nimbostratus",
            "ko": "난층운",
            "en": "Nimbostratus",
            "desc": "두껍고 어두운 회색 구름층으로, 오랫동안 지속되는 비나 눈을 내립니다.",
        },
    ],
    "estratocumuliformes": [
        {
            "id": "stratocumulus",
            "ko": "층쌘구름",
            "en": "Stratocumulus",
            "desc": "넓게 퍼진 판 모양의 덩어리 구름이 하늘을 뒤덮는 형태로, 대체로 약한 비 또는 비가 없는 경우가 많습니다.",
        },
        {
            "id": "altocumulus",
            "ko": "양떼구름",
            "en": "Altocumulus",
            "desc": "중간 높이에서 작은 구름 덩어리들이 무리지어 나타나는 구름으로, 양떼처럼 보입니다.",
        },
    ],
}


# --------------------
# 예측 함수
# --------------------
def predict_class(img_path: str):
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pred_idx = logits.argmax(1).item()
        prob = torch.softmax(logits, dim=1)[0, pred_idx].item()
    label = classes[pred_idx]
    return img, label, prob


# --------------------
# Tkinter GUI
# --------------------
class CloudClassifierApp:
    def __init__(self, master):
        self.master = master
        master.title("AI 구름 분류 안내기 (4종 + 세부 구름)")
        master.geometry("1100x720")

        # 좌: 입력 이미지, 우: 결과/하위 구름 갤러리
        self.left = tk.Frame(master)
        self.left.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        self.right = tk.Frame(master, width=420)
        self.right.pack(side="right", fill="both", padx=8, pady=8)

        # 좌측 미리보기
        self.preview_label = tk.Label(
            self.left, text="구름 이미지를 선택해주세요.", bg="#eeeeee"
        )
        self.preview_label.pack(fill="both", expand=True)

        # 버튼들
        self.btn_frame = tk.Frame(self.left)
        self.btn_frame.pack(fill="x", pady=8)
        self.select_btn = tk.Button(
            self.btn_frame, text="이미지 선택", command=self.select_image
        )
        self.select_btn.pack(side="left", padx=4)
        self.clear_btn = tk.Button(self.btn_frame, text="초기화", command=self.clear)
        self.clear_btn.pack(side="left", padx=4)

        # 우측 상단: 결과 요약
        self.result_title = tk.Label(
            self.right, text="예측 결과", font=("맑은 고딕", 14, "bold")
        )
        self.result_title.pack(anchor="w")

        self.result_text = tk.Label(
            self.right,
            text="이미지를 선택하면 결과가 표시됩니다.",
            justify="left",
            font=("맑은 고딕", 11),
        )
        self.result_text.pack(anchor="w", pady=(2, 6))

        # 그룹 설명
        self.group_desc_label = tk.Label(
            self.right, text="", justify="left", font=("맑은 고딕", 10), wraplength=400
        )
        self.group_desc_label.pack(anchor="w", pady=(0, 10))

        # 스크롤 가능한 캔버스 (하위 구름 카드들)
        self.sub_canvas = tk.Canvas(
            self.right, width=400, height=520, highlightthickness=0
        )
        self.sub_scroll = tk.Scrollbar(
            self.right, orient="vertical", command=self.sub_canvas.yview
        )
        self.sub_frame = tk.Frame(self.sub_canvas)

        self.sub_frame.bind(
            "<Configure>",
            lambda e: self.sub_canvas.configure(
                scrollregion=self.sub_canvas.bbox("all")
            ),
        )
        self.sub_canvas.create_window((0, 0), window=self.sub_frame, anchor="nw")
        self.sub_canvas.configure(yscrollcommand=self.sub_scroll.set)

        self.sub_canvas.pack(side="left", fill="both", expand=True)
        self.sub_scroll.pack(side="right", fill="y")

        # 이미지 참조 보관
        self._preview_tk = None
        self._subtype_tks = []  # 하위 구름 썸네일들

    def select_image(self):
        path = filedialog.askopenfilename(
            title="구름 이미지 선택",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            img, label, prob = predict_class(path)
        except Exception as e:
            messagebox.showerror("오류", str(e))
            return

        kor_group = KOR_LABELS.get(label, label)
        self.result_text.configure(
            text=f"예측된 구름군: {kor_group} ({label})\n신뢰도: {prob*100:.1f}%"
        )

        # 그룹 설명
        gdesc = GROUP_DESC.get(label, "")
        self.group_desc_label.configure(text=gdesc)

        # 좌측 미리보기
        preview = img.copy()
        preview.thumbnail((750, 600))
        self._preview_tk = ImageTk.PhotoImage(preview)
        self.preview_label.configure(image=self._preview_tk, text="")

        # 하위 구름 카드들 생성
        self.load_subtypes(label)

    def load_subtypes(self, group_label: str):
        # 기존 내용 제거
        for w in self.sub_frame.winfo_children():
            w.destroy()
        self._subtype_tks.clear()

        subtypes = SUBTYPES.get(group_label, [])
        if not subtypes:
            tk.Label(
                self.sub_frame,
                text="이 구름군에 대한 세부 정보가 없습니다.",
                font=("맑은 고딕", 10),
            ).pack(anchor="w", pady=4)
            return

        for sub in subtypes:
            card = tk.Frame(self.sub_frame, bd=1, relief="solid", padx=4, pady=4)
            card.pack(fill="x", pady=4)

            title = tk.Label(
                card,
                text=f"{sub['ko']} ({sub['en']})",
                font=("맑은 고딕", 11, "bold"),
                anchor="w",
            )
            title.pack(anchor="w")

            desc = tk.Label(
                card,
                text=sub["desc"],
                font=("맑은 고딕", 9),
                justify="left",
                wraplength=360,
            )
            desc.pack(anchor="w", pady=(2, 4))

            # 이미지 썸네일 영역
            img_frame = tk.Frame(card)
            img_frame.pack(fill="x")

            # assets/<group>/<subtype_id>/*.jpg 불러오기
            folder = os.path.join(GALLERY_DIR, group_label, sub["id"])
            files = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"):
                files.extend(glob.glob(os.path.join(folder, ext)))
            files = sorted(files)[:5]  # 각 하위 구름당 최대 5장 정도만

            if not files:
                tk.Label(
                    img_frame,
                    text=f"샘플 이미지가 없습니다.\n({folder} 폴더에 이미지를 넣어주세요.)",
                    font=("맑은 고딕", 8),
                    justify="left",
                ).pack(anchor="w")
                continue

            # 썸네일들 가로로 나열
            for i, f in enumerate(files):
                try:
                    im = Image.open(f).convert("RGB")
                    im.thumbnail((360, 280))
                    tkimg = ImageTk.PhotoImage(im)
                except Exception:
                    continue

                self._subtype_tks.append(tkimg)  # 참조 유지
                lbl = tk.Label(img_frame, image=tkimg)
                lbl.pack(side="left", padx=2, pady=2)

    def clear(self):
        self._preview_tk = None
        self._subtype_tks.clear()
        self.preview_label.configure(image="", text="구름 이미지를 선택해주세요.")
        self.result_text.configure(text="이미지를 선택하면 결과가 표시됩니다.")
        self.group_desc_label.configure(text="")
        for w in self.sub_frame.winfo_children():
            w.destroy()


# --------------------
# 실행
# --------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = CloudClassifierApp(root)
    root.mainloop()
