"""
GBM-based Risk Scoring
======================

Gradient Boosting Machine을 사용한 위험도 예측

이 모듈은 Vital Sign 특성(Feature)을 기반으로 위험 확률을 예측합니다.

Author: KAIST AI Graduate School CAIO 9

References:
    [1] Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System.
    [2] Ke et al. (2017). LightGBM: A Highly Efficient Gradient Boosting.
"""

import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging
import os
import pickle

logger = logging.getLogger(__name__)


@dataclass
class RiskResult:
    """위험도 예측 결과"""
    risk: float            # 위험 확률 (0~1)
    risk_level: str        # "LOW" | "MEDIUM" | "HIGH"
    features: Dict[str, float]  # 사용된 특성
    feature_importance: Optional[Dict[str, float]] = None


class MockGBMModel:
    """
    Mock GBM 모델

    실제 XGBoost/LightGBM 없이 규칙 기반으로 위험도를 계산합니다.
    실제 배포 시에는 학습된 모델로 교체합니다.
    """

    def __init__(self):
        # 특성 가중치 (실제 모델의 feature importance 시뮬레이션)
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

    def predict_proba(self, features: Dict[str, float]) -> float:
        """
        위험 확률 예측 (0~1)

        규칙 기반 위험도 계산:
        - 심박수 이상 (< 50 or > 110): 위험 증가
        - 심박수 변동성 높음 (std > 10): 위험 증가
        - 호흡수 이상 (< 8 or > 25): 위험 증가
        - 하향 추세: 위험 증가
        """
        risk_score = 0.0

        # 1. 심박수 평균 위험
        hr_mean = features.get("hr_mean", 72)
        if hr_mean < 50 or hr_mean > 110:
            risk_score += 0.3
        elif hr_mean < 55 or hr_mean > 100:
            risk_score += 0.15

        # 2. 심박수 변동성 위험
        hr_std = features.get("hr_std", 5)
        if hr_std > 15:
            risk_score += 0.2
        elif hr_std > 10:
            risk_score += 0.1

        # 3. 심박수 추세 위험 (음수 = 감소 추세)
        hr_trend = features.get("hr_trend", 0)
        if hr_trend < -1.0:  # 급격한 감소
            risk_score += 0.25
        elif hr_trend < -0.5:
            risk_score += 0.1

        # 4. 호흡수 평균 위험
        br_mean = features.get("br_mean", 16)
        if br_mean < 8 or br_mean > 25:
            risk_score += 0.25
        elif br_mean < 10 or br_mean > 22:
            risk_score += 0.1

        # 5. 극단값 위험
        hr_min = features.get("hr_min", 70)
        hr_max = features.get("hr_max", 75)
        if hr_min < 40 or hr_max > 130:
            risk_score += 0.2

        # 정규화 (0~1 범위)
        risk_prob = min(1.0, max(0.0, risk_score))

        return risk_prob


class RiskScorer:
    """
    GBM 기반 위험도 예측기

    Vital Sign 시계열 데이터에서 특성을 추출하고 위험 확률을 예측합니다.

    Args:
        model_path: 사전학습된 GBM 모델 경로 (None이면 Mock 모델 사용)
        threshold_high: HIGH 위험 임계값 (default: 0.6)
        threshold_medium: MEDIUM 위험 임계값 (default: 0.3)

    Example:
        >>> scorer = RiskScorer()
        >>> result = scorer.predict(hr_series=[72.0]*60, br_series=[16.0]*60)
        >>> print(result.risk_level)  # "LOW"
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold_high: float = 0.6,
        threshold_medium: float = 0.3
    ):
        self.threshold_high = threshold_high
        self.threshold_medium = threshold_medium

        # 모델 로드 또는 Mock 생성
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
            logger.info(f"GBM model loaded from {model_path}")
        else:
            self.model = MockGBMModel()
            logger.info("Using Mock GBM model")

    def _load_model(self, path: str):
        """모델 로드"""
        try:
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load GBM model: {e}. Using Mock model.")
            self.model = MockGBMModel()

    def extract_features(
        self,
        hr_series: List[float],
        br_series: List[float]
    ) -> Dict[str, float]:
        """
        시계열 데이터에서 특성 추출

        Args:
            hr_series: 심박수 시계열
            br_series: 호흡수 시계열

        Returns:
            Dict[str, float]: 추출된 특성
        """
        hr = np.array(hr_series)
        br = np.array(br_series)

        # 기본 통계
        features = {
            # 심박수 특성
            "hr_mean": float(np.mean(hr)),
            "hr_std": float(np.std(hr)),
            "hr_min": float(np.min(hr)),
            "hr_max": float(np.max(hr)),
            "hr_range": float(np.max(hr) - np.min(hr)),

            # 호흡수 특성
            "br_mean": float(np.mean(br)),
            "br_std": float(np.std(br)),
            "br_min": float(np.min(br)),
            "br_max": float(np.max(br)),
            "br_range": float(np.max(br) - np.min(br)),
        }

        # 추세 계산 (선형 회귀 기울기)
        if len(hr) > 1:
            x = np.arange(len(hr))
            hr_slope = np.polyfit(x, hr, 1)[0]
            br_slope = np.polyfit(x, br, 1)[0]
            features["hr_trend"] = float(hr_slope)
            features["br_trend"] = float(br_slope)
        else:
            features["hr_trend"] = 0.0
            features["br_trend"] = 0.0

        return features

    def predict(
        self,
        hr_series: List[float],
        br_series: List[float]
    ) -> RiskResult:
        """
        위험도 예측

        Args:
            hr_series: 심박수 시계열 (60 samples)
            br_series: 호흡수 시계열 (60 samples)

        Returns:
            RiskResult: 예측 결과
        """
        # 특성 추출
        features = self.extract_features(hr_series, br_series)

        # 위험 확률 예측
        risk_prob = self.model.predict_proba(features)

        # 위험 수준 판정
        if risk_prob >= self.threshold_high:
            risk_level = "HIGH"
        elif risk_prob >= self.threshold_medium:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Feature importance (Mock 모델인 경우)
        feature_importance = None
        if hasattr(self.model, 'feature_weights'):
            feature_importance = self.model.feature_weights

        return RiskResult(
            risk=round(risk_prob, 4),
            risk_level=risk_level,
            features=features,
            feature_importance=feature_importance
        )


# =====================
# LangGraph 노드 함수
# =====================

def calculate_risk_node(state: dict) -> dict:
    """
    LangGraph 노드: GBM 위험도 예측

    Args:
        state: AgentState 딕셔너리

    Returns:
        dict: 업데이트된 상태 필드
    """
    scorer = RiskScorer()

    result = scorer.predict(
        hr_series=state["hr_series"],
        br_series=state["br_series"]
    )

    return {
        "gbm_risk": result.risk,
        "gbm_details": {
            "risk_level": result.risk_level,
            "features": result.features,
            "feature_importance": result.feature_importance,
        },
        "message": f"GBM risk prediction completed. Risk: {result.risk:.3f} ({result.risk_level})"
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  GBM Risk Scorer Test")
    print("=" * 60)

    scorer = RiskScorer()

    # 테스트 케이스
    print("\n테스트 케이스:")
    print("-" * 60)

    # Case 1: 정상
    normal_hr = list(np.random.normal(72, 3, 60))
    normal_br = list(np.random.normal(16, 1, 60))
    result = scorer.predict(normal_hr, normal_br)
    print(f"  1. Normal (HR=72±3, BR=16±1):")
    print(f"     Risk: {result.risk:.4f}, Level: {result.risk_level}")

    # Case 2: 빈맥
    tachy_hr = list(np.random.normal(115, 5, 60))
    tachy_br = list(np.random.normal(22, 2, 60))
    result = scorer.predict(tachy_hr, tachy_br)
    print(f"  2. Tachycardia (HR=115±5, BR=22±2):")
    print(f"     Risk: {result.risk:.4f}, Level: {result.risk_level}")

    # Case 3: 서맥
    brady_hr = list(np.random.normal(45, 3, 60))
    brady_br = list(np.random.normal(10, 1, 60))
    result = scorer.predict(brady_hr, brady_br)
    print(f"  3. Bradycardia (HR=45±3, BR=10±1):")
    print(f"     Risk: {result.risk:.4f}, Level: {result.risk_level}")

    # Case 4: 심정지 패턴 (하향 추세)
    arrest_hr = list(np.linspace(72, 20, 60))
    arrest_br = list(np.linspace(16, 4, 60))
    result = scorer.predict(arrest_hr, arrest_br)
    print(f"  4. Cardiac Arrest Pattern (HR: 72→20, BR: 16→4):")
    print(f"     Risk: {result.risk:.4f}, Level: {result.risk_level}")

    print("\n특성 예시 (Case 4):")
    for k, v in result.features.items():
        print(f"     {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("  Test Complete!")
    print("=" * 60)
