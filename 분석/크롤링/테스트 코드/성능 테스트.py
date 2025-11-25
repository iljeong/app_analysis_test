import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


import pandas as pd

# 1) 원본 데이터 읽기
df = pd.read_csv("/Users/seojeong-il/Desktop/내문서/데이터 분석/개인 분석/보이저엑스/vrew/reviews_for_labeling.csv")

# 2) 샘플 300개만 뽑기 (원하면 숫자 바꿔도 됨)
df_sample = df.sample(300, random_state=42).copy()

# 3) 사람이 채울 정답 컬럼 추가 (초기값은 빈 값)
df_sample["true_label"] = ""

# 4) 라벨링용 파일로 저장
df_sample.to_csv("reviews_for_labeling.csv", index=False, encoding="utf-8-sig")

print("✅ 'reviews_for_labeling.csv' 파일 생성 완료 (여기에 true_label 직접 채우면 됨)")



df = pd.read_csv("/Users/seojeong-il/Desktop/내문서/데이터 분석/개인 분석/보이저엑스/vrew/reviews_for_labeling.csv")

y_true = df["true_label"]
y_pred = df["pred_label"]

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, pos_label="NEG"))
print("Recall:", recall_score(y_true, y_pred, pos_label="NEG"))
print("F1:", f1_score(y_true, y_pred, pos_label="NEG"))

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
