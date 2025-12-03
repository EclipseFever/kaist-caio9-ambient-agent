"""
AutoEncoder-based Anomaly Detection
====================================

CNN AutoEncoder를 사용한 Vital Sign 시계열 이상 탐지

이 모듈은 정상 패턴으로 학습된 AutoEncoder의 재구성 오차(Reconstruction Error)를
기반으로 이상 상황을 탐지합니다.

Author: KAIST AI Graduate School CAIO 9

References:
    [1] An & Cho (2015). Variational Autoencoder based Anomaly Detection.
    [2] kaist-caio9-autoencoder 프로젝트
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """이상 탐지 결과"""
    score: float           # 재구성 오차 (0~1 정규화)
    is_anomaly: bool       # 이상 여부
    raw_mse: float         # 원본 MSE 값
    sqi: float             # Signal Quality Index
    reconstructed: Optional[np.ndarray] = None


class SimpleAutoEncoder(nn.Module):
    """
    경량 Linear AutoEncoder

    kaist-caio9-autoencoder 프로젝트와 동일한 구조입니다.
    """

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

    def reconstruct(self, signal: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(signal).unsqueeze(0)
            reconstructed = self.forward(x)
        return reconstructed.squeeze(0).numpy()


class AnomalyDetector:
    """
    AutoEncoder 기반 이상 탐지기

    Vital Sign 시계열 데이터에서 정상 패턴을 벗어나는 이상을 탐지합니다.

    Args:
        model_path: 사전학습된 모델 경로 (None이면 새 모델 생성)
        threshold: 이상 판단 임계값 (default: 0.35)
        input_dim: 입력 시계열 길이 (default: 60)

    Example:
        >>> detector = AnomalyDetector()
        >>> result = detector.detect(hr_series=[72.0] * 60)
        >>> print(result.is_anomaly)  # False
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.35,
        input_dim: int = 60
    ):
        self.threshold = threshold
        self.input_dim = input_dim

        # 모델 초기화
        self.model = SimpleAutoEncoder(input_dim=input_dim, latent_dim=20)

        # 모델 로드 시도
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            # 정상 패턴으로 간단히 학습 (Mock)
            self._train_on_normal()
            logger.info("Model trained on synthetic normal data")

    def _load_model(self, path: str):
        """모델 로드"""
        try:
            state_dict = torch.load(path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except Exception as e:
            logger.warning(f"Failed to load model: {e}. Using new model.")
            self._train_on_normal()

    def _train_on_normal(self, epochs: int = 100):
        """정상 데이터로 학습"""
        # 정상 호흡 패턴 생성
        t = np.linspace(0, 4 * np.pi, self.input_dim)
        normal_pattern = np.sin(t) * 0.3 + 0.5
        normal_pattern += np.random.randn(self.input_dim) * 0.02

        # 학습
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        self.model.train()
        for _ in range(epochs):
            x = torch.FloatTensor(normal_pattern).unsqueeze(0)
            optimizer.zero_grad()
            reconstructed = self.model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()

        self.model.eval()

    def detect(
        self,
        hr_series: list,
        br_series: Optional[list] = None
    ) -> AnomalyResult:
        """
        이상 탐지 실행

        Args:
            hr_series: 심박수 시계열 (60 samples)
            br_series: 호흡수 시계열 (선택, 현재는 HR만 사용)

        Returns:
            AnomalyResult: 탐지 결과
        """
        # 데이터 전처리
        signal = np.array(hr_series, dtype=np.float32)

        # 정규화 (0~1 범위로)
        signal_min, signal_max = signal.min(), signal.max()
        if signal_max - signal_min > 0:
            normalized = (signal - signal_min) / (signal_max - signal_min)
        else:
            normalized = np.ones_like(signal) * 0.5

        # 길이 조정
        if len(normalized) < self.input_dim:
            # 패딩
            normalized = np.pad(normalized, (0, self.input_dim - len(normalized)), mode='edge')
        elif len(normalized) > self.input_dim:
            # 최근 데이터만 사용
            normalized = normalized[-self.input_dim:]

        # 재구성
        reconstructed = self.model.reconstruct(normalized)

        # MSE 계산
        raw_mse = float(np.mean((normalized - reconstructed) ** 2))

        # 정규화된 스코어 (0~1)
        # MSE를 sigmoid-like 함수로 변환
        score = min(1.0, raw_mse * 10)  # 스케일링

        # SQI 계산
        sqi = self._calculate_sqi(signal)

        # 이상 판단
        is_anomaly = score > self.threshold

        return AnomalyResult(
            score=round(score, 4),
            is_anomaly=is_anomaly,
            raw_mse=round(raw_mse, 6),
            sqi=round(sqi, 4),
            reconstructed=reconstructed
        )

    def _calculate_sqi(self, signal: np.ndarray) -> float:
        """
        Signal Quality Index 계산

        신호의 품질을 0~1 범위로 평가합니다.
        """
        # 1. SNR 기반 점수
        diff = np.diff(signal)
        noise_level = np.std(diff)

        if noise_level < 2.0:
            snr_score = 1.0
        elif noise_level < 5.0:
            snr_score = 0.8
        elif noise_level < 10.0:
            snr_score = 0.5
        else:
            snr_score = 0.2

        # 2. 연속성 점수
        max_jump = np.max(np.abs(diff)) if len(diff) > 0 else 0

        if max_jump < 5:
            continuity_score = 1.0
        elif max_jump < 10:
            continuity_score = 0.7
        elif max_jump < 20:
            continuity_score = 0.4
        else:
            continuity_score = 0.2

        # 3. 범위 점수 (생리학적 유효 범위: 40-150 bpm)
        in_range = np.mean((signal >= 40) & (signal <= 150))
        range_score = float(in_range)

        # 가중 평균
        sqi = 0.4 * snr_score + 0.3 * continuity_score + 0.3 * range_score

        return sqi


# =====================
# LangGraph 노드 함수
# =====================

def detect_anomaly_node(state: dict) -> dict:
    """
    LangGraph 노드: AutoEncoder 이상 탐지

    Args:
        state: AgentState 딕셔너리

    Returns:
        dict: 업데이트된 상태 필드
    """
    detector = AnomalyDetector()

    result = detector.detect(
        hr_series=state["hr_series"],
        br_series=state.get("br_series")
    )

    return {
        "ae_score": result.score,
        "ae_anomaly": result.is_anomaly,
        "ae_details": {
            "raw_mse": result.raw_mse,
            "sqi": result.sqi,
            "threshold": detector.threshold,
        },
        "message": f"AE anomaly detection completed. Score: {result.score:.3f}"
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  AutoEncoder Anomaly Detector Test")
    print("=" * 60)

    detector = AnomalyDetector()

    # 테스트 케이스
    print("\n테스트 케이스:")
    print("-" * 60)

    # Case 1: 정상 심박
    normal_hr = list(np.random.normal(72, 2, 60))
    result = detector.detect(normal_hr)
    print(f"  1. Normal HR (72±2 bpm):")
    print(f"     Score: {result.score:.4f}, Anomaly: {result.is_anomaly}, SQI: {result.sqi:.4f}")

    # Case 2: 노이즈 많은 신호
    noisy_hr = list(np.random.normal(72, 15, 60))
    result = detector.detect(noisy_hr)
    print(f"  2. Noisy HR (72±15 bpm):")
    print(f"     Score: {result.score:.4f}, Anomaly: {result.is_anomaly}, SQI: {result.sqi:.4f}")

    # Case 3: 심정지 패턴 (신호 감소)
    cardiac_arrest = list(np.linspace(72, 10, 60))
    result = detector.detect(cardiac_arrest)
    print(f"  3. Cardiac Arrest (72→10 bpm):")
    print(f"     Score: {result.score:.4f}, Anomaly: {result.is_anomaly}, SQI: {result.sqi:.4f}")

    # Case 4: 빈맥
    tachycardia = list(np.random.normal(130, 5, 60))
    result = detector.detect(tachycardia)
    print(f"  4. Tachycardia (130±5 bpm):")
    print(f"     Score: {result.score:.4f}, Anomaly: {result.is_anomaly}, SQI: {result.sqi:.4f}")

    print("\n" + "=" * 60)
    print("  Test Complete!")
    print("=" * 60)
