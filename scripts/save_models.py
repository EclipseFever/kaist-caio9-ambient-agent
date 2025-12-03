#!/usr/bin/env python3
"""
사전학습 모델 생성 및 저장 스크립트
===================================

AutoEncoder와 GBM 모델을 학습하고 저장합니다.

Usage:
    python scripts/save_models.py

Author: KAIST AI Graduate School CAIO 9
"""

import sys
import os

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import pickle
import json
from datetime import datetime


class SimpleAutoEncoder(nn.Module):
    """경량 Linear AutoEncoder"""

    def __init__(self, input_dim: int = 60, latent_dim: int = 20):
        super(SimpleAutoEncoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, latent_dim),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class MockGBMModel:
    """Mock GBM 모델 (Pickle 저장용)"""

    def __init__(self):
        self.feature_weights = {
            "hr_mean": 0.15,
            "hr_std": 0.12,
            "hr_min": 0.10,
            "hr_max": 0.10,
            "hr_trend": 0.13,
            "br_mean": 0.12,
            "br_std": 0.10,
            "br_min": 0.08,
            "br_max": 0.05,
            "br_trend": 0.05,
        }
        self.version = "1.0.0"
        self.trained_at = datetime.now().isoformat()

    def predict_proba(self, features: dict) -> float:
        """위험 확률 예측"""
        risk_score = 0.0

        hr_mean = features.get("hr_mean", 72)
        if hr_mean < 50 or hr_mean > 110:
            risk_score += 0.3
        elif hr_mean < 55 or hr_mean > 100:
            risk_score += 0.15

        hr_std = features.get("hr_std", 5)
        if hr_std > 15:
            risk_score += 0.2
        elif hr_std > 10:
            risk_score += 0.1

        hr_trend = features.get("hr_trend", 0)
        if hr_trend < -1.0:
            risk_score += 0.25
        elif hr_trend < -0.5:
            risk_score += 0.1

        br_mean = features.get("br_mean", 16)
        if br_mean < 8 or br_mean > 25:
            risk_score += 0.25
        elif br_mean < 10 or br_mean > 22:
            risk_score += 0.1

        hr_min = features.get("hr_min", 70)
        hr_max = features.get("hr_max", 75)
        if hr_min < 40 or hr_max > 130:
            risk_score += 0.2

        return min(1.0, max(0.0, risk_score))


def train_autoencoder():
    """AutoEncoder 학습 및 저장"""
    print("\n[1/2] AutoEncoder 학습 중...")

    model = SimpleAutoEncoder(input_dim=60, latent_dim=20)

    # 정상 호흡 패턴 생성
    t = np.linspace(0, 4 * np.pi, 60)
    normal_pattern = np.sin(t) * 0.3 + 0.5
    normal_pattern += np.random.randn(60) * 0.02

    # 학습
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    model.train()
    epochs = 200
    for epoch in range(epochs):
        x = torch.FloatTensor(normal_pattern).unsqueeze(0)
        optimizer.zero_grad()
        reconstructed = model(x)
        loss = criterion(reconstructed, x)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    model.eval()

    # 저장
    model_path = "models/autoencoder_v1.0.pth"
    torch.save(model.state_dict(), model_path)
    print(f"    ✅ 저장됨: {model_path}")

    # 메타데이터
    return {
        "architecture": "SimpleAutoEncoder",
        "input_dim": 60,
        "latent_dim": 20,
        "parameters": sum(p.numel() for p in model.parameters()),
        "final_loss": loss.item(),
    }


def save_gbm_model():
    """GBM 모델 저장"""
    print("\n[2/2] GBM 모델 저장 중...")

    model = MockGBMModel()
    model_path = "models/gbm_risk_model.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"    ✅ 저장됨: {model_path}")

    return {
        "type": "MockGBMModel",
        "features": list(model.feature_weights.keys()),
        "version": model.version,
    }


def save_model_info(ae_info: dict, gbm_info: dict):
    """모델 메타데이터 저장"""
    info = {
        "created_at": datetime.now().isoformat(),
        "project": "kaist-caio9-ambient-agent",
        "models": {
            "autoencoder": ae_info,
            "gbm": gbm_info,
        }
    }

    info_path = "models/model_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n    ✅ 메타데이터 저장됨: {info_path}")


def main():
    print("=" * 60)
    print("  모델 생성 및 저장")
    print("  KAIST AI Graduate School CAIO 9")
    print("=" * 60)

    # 디렉토리 확인
    os.makedirs("models", exist_ok=True)

    # 모델 학습/저장
    ae_info = train_autoencoder()
    gbm_info = save_gbm_model()
    save_model_info(ae_info, gbm_info)

    print("\n" + "=" * 60)
    print("  완료!")
    print("=" * 60)

    # 파일 확인
    print("\n생성된 파일:")
    for f in os.listdir("models"):
        size = os.path.getsize(f"models/{f}")
        print(f"  - models/{f} ({size:,} bytes)")


if __name__ == "__main__":
    main()
