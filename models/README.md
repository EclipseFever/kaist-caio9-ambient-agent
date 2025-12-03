# Pre-trained Models

LangGraph Ambient Agent에서 사용하는 사전학습 모델

## Models

### 1. AutoEncoder (`autoencoder_v1.0.pth`)

Vital Sign 시계열 이상 탐지를 위한 Linear AutoEncoder

| 항목 | 값 |
|------|-----|
| Architecture | SimpleAutoEncoder |
| Input Dim | 60 (60초 시계열) |
| Latent Dim | 20 |
| Parameters | 13,328 |
| Final Loss | 8.48e-06 |
| File Size | 58 KB |

**Architecture:**
```
Encoder: 60 → 64 → 32 → 20 (latent)
Decoder: 20 → 32 → 64 → 60
Activation: ReLU + Sigmoid
```

**Usage:**
```python
import torch
from src.anomaly_detector import SimpleAutoEncoder

model = SimpleAutoEncoder(input_dim=60, latent_dim=20)
model.load_state_dict(torch.load("models/autoencoder_v1.0.pth"))
model.eval()

# 추론
signal = torch.randn(1, 60)
reconstructed = model(signal)
mse = ((signal - reconstructed) ** 2).mean().item()
```

### 2. GBM Risk Model (`gbm_risk_model.pkl`)

Vital Sign 특성 기반 위험도 예측 모델

| 항목 | 값 |
|------|-----|
| Type | MockGBMModel |
| Version | 1.0.0 |
| File Size | 316 bytes |

**Features (10개):**
- `hr_mean`, `hr_std`, `hr_min`, `hr_max`, `hr_trend`
- `br_mean`, `br_std`, `br_min`, `br_max`, `br_trend`

**Usage:**
```python
import pickle

with open("models/gbm_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

features = {
    "hr_mean": 72.0,
    "hr_std": 3.0,
    "hr_min": 68.0,
    "hr_max": 78.0,
    "hr_trend": 0.0,
    "br_mean": 16.0,
    "br_std": 1.0,
    "br_min": 14.0,
    "br_max": 18.0,
    "br_trend": 0.0,
}

risk = model.predict_proba(features)  # 0.0 ~ 1.0
```

## 모델 재생성

```bash
# 가상환경 활성화
source venv/bin/activate

# 모델 재학습 및 저장
python scripts/save_models.py
```

## 메타데이터

`model_info.json` 파일에 모델 생성 정보가 저장됨:
- 생성 일시
- 아키텍처 정보
- 학습 파라미터

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2025-12-03 | Initial release |

---

*KAIST 김재철AI대학원 CAIO 9기*
