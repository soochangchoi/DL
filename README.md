# 🤖 AI 통합 Flask 웹 서비스 (숫자 인식, 감정 인식, 마스크 감지, 성별 분류)

다양한 이미지 AI 모델 (숫자 인식, 감정 인식, 마스크 착용, 성별 분류)을  
Flask를 통해 통합 제공하는 AI 웹 서비스입니다.

---

## 📂 프로젝트 구조

| 파트            | 주요 기능                        | 스크립트              | 모델 파일             |
|----------------|-----------------------------|--------------------|---------------------|
| 숫자 인식 (CNN)  | 필기 숫자 인식 (0~9)             | `writer.py`         | `best_model.pt`, `1.pt` |
| 감정 인식       | 얼굴 감정 분류 (happy, sad 등)   | 통합 Flask API 예상  | `emotion.pt`        |
| 마스크 감지      | 마스크 착용 여부 감지             | 통합 Flask API 예상  | `mask_model.pt`     |
| 성별 분류       | 얼굴 성별 분류                    | 통합 Flask API 예상  | `sex_model.pt`      |

---

## ▶ 실행 방법

### 1. 필기 숫자 인식 (PyTorch CNN)
```bash
python writer.py
모델: best_model.pt

입력: ../project/0.png

결과: 예측된 숫자 클래스 (0~9)

2. 감정, 마스크, 성별 인식 (예상 Flask API 통합)
/predict_digit → 숫자 인식

/predict_emotion → 감정 인식

/predict_mask → 마스크 착용 감지

/predict_sex → 성별 분류

✅ Flask 통합 API는 app.py 등에서 라우트로 통합 가능.

💡 주요 특징
✅ PyTorch 기반 다중 AI 모델 지원

✅ Flask REST API 제공 (추가 예정)

✅ 이미지 입력 → 모델 예측 → JSON 응답 가능

✅ GPU 자동 인식 (cuda → cpu 자동 fallback)

📃 사용 모델 정보

best_model.pt	숫자 인식 CNN 최종 모델
1.pt	숫자 인식 CNN 다른 버전
emotion.pt	얼굴 감정 분류 CNN 모델
mask_model.pt	마스크 착용 감지 CNN 모델
sex_model.pt	성별 분류 CNN 모델

✅ 향후 통합 API 예시 (Flask 예상)

복사
편집
@app.route('/predict_digit', methods=['POST'])
@app.route('/predict_emotion', methods=['POST'])
@app.route('/predict_mask', methods=['POST'])
@app.route('/predict_sex', methods=['POST'])
