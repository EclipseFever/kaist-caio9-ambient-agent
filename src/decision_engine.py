"""
Decision Engine for Final State Determination
==============================================

SPARQL 규칙, AutoEncoder 이상도, GBM 위험도를 통합하여 최종 상태를 판단

이 모듈은 3가지 신호를 융합(Fusion)하여 CRITICAL/WARNING/NORMAL 상태를 결정합니다.

Author: KAIST AI Graduate School CAIO 9

Decision Matrix:
    - CRITICAL: SPARQL alert OR (AE > 0.35 AND GBM > 0.6)
    - WARNING:  AE > 0.35 OR GBM > 0.6
    - NORMAL:   그 외
"""

from typing import Tuple, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DecisionResult:
    """의사결정 결과"""
    state: str              # "CRITICAL" | "WARNING" | "NORMAL"
    confidence: float       # 판단 신뢰도 (0~1)
    reasoning: str          # 판단 근거
    factors: Dict[str, Any] # 기여 요인


class DecisionEngine:
    """
    최종 상태 결정 엔진

    3가지 입력 신호를 통합하여 최종 상태를 결정합니다:
        1. SPARQL 규칙 기반 Alert (bool)
        2. AutoEncoder 이상도 (float, 0~1)
        3. GBM 위험도 (float, 0~1)

    Args:
        ae_critical: AE 임계값 (CRITICAL 판단용, default: 0.35)
        gbm_critical: GBM 임계값 (CRITICAL 판단용, default: 0.6)
        ae_warning: AE 임계값 (WARNING 판단용, default: 0.25)
        gbm_warning: GBM 임계값 (WARNING 판단용, default: 0.4)

    Example:
        >>> engine = DecisionEngine()
        >>> result = engine.decide(
        ...     sparql_alert=False,
        ...     ae_score=0.4,
        ...     gbm_risk=0.7
        ... )
        >>> print(result.state)  # "CRITICAL"
    """

    def __init__(
        self,
        ae_critical: float = 0.35,
        gbm_critical: float = 0.6,
        ae_warning: float = 0.25,
        gbm_warning: float = 0.4
    ):
        self.ae_critical = ae_critical
        self.gbm_critical = gbm_critical
        self.ae_warning = ae_warning
        self.gbm_warning = gbm_warning

    def decide(
        self,
        sparql_alert: bool,
        ae_score: float,
        gbm_risk: float
    ) -> DecisionResult:
        """
        최종 상태 결정

        Decision Logic:
            1. SPARQL alert → 즉시 CRITICAL (규칙 기반 이상)
            2. AE > 0.35 AND GBM > 0.6 → CRITICAL (복합 이상)
            3. AE > 0.35 OR GBM > 0.6 → WARNING (단일 이상)
            4. AE > 0.25 OR GBM > 0.4 → WARNING (경미한 이상)
            5. 그 외 → NORMAL

        Args:
            sparql_alert: SPARQL 규칙 기반 이상 여부
            ae_score: AutoEncoder 재구성 오차 (0~1)
            gbm_risk: GBM 위험 확률 (0~1)

        Returns:
            DecisionResult: 판단 결과
        """
        reasons = []
        factors = {
            "sparql_alert": sparql_alert,
            "ae_score": ae_score,
            "gbm_risk": gbm_risk,
            "ae_critical_threshold": self.ae_critical,
            "gbm_critical_threshold": self.gbm_critical,
        }

        # ==================
        # Priority 1: SPARQL Alert
        # ==================
        if sparql_alert:
            reasons.append("SPARQL rule-based alert triggered")
            return DecisionResult(
                state="CRITICAL",
                confidence=0.95,
                reasoning=" | ".join(reasons),
                factors=factors
            )

        # ==================
        # Priority 2: Compound Critical (AE + GBM)
        # ==================
        if ae_score > self.ae_critical and gbm_risk > self.gbm_critical:
            reasons.append(f"AE score ({ae_score:.3f}) > {self.ae_critical}")
            reasons.append(f"GBM risk ({gbm_risk:.3f}) > {self.gbm_critical}")
            reasons.append("Compound anomaly detected")

            # 신뢰도: 두 지표의 가중 평균
            confidence = 0.5 * min(ae_score / self.ae_critical, 1.5) + \
                         0.5 * min(gbm_risk / self.gbm_critical, 1.5)
            confidence = min(0.95, confidence / 1.5)

            return DecisionResult(
                state="CRITICAL",
                confidence=round(confidence, 3),
                reasoning=" | ".join(reasons),
                factors=factors
            )

        # ==================
        # Priority 3: Single Critical Indicator
        # ==================
        if ae_score > self.ae_critical:
            reasons.append(f"AE score ({ae_score:.3f}) > {self.ae_critical}")

            confidence = min(0.85, ae_score / self.ae_critical * 0.6)

            return DecisionResult(
                state="WARNING",
                confidence=round(confidence, 3),
                reasoning=" | ".join(reasons),
                factors=factors
            )

        if gbm_risk > self.gbm_critical:
            reasons.append(f"GBM risk ({gbm_risk:.3f}) > {self.gbm_critical}")

            confidence = min(0.85, gbm_risk / self.gbm_critical * 0.6)

            return DecisionResult(
                state="WARNING",
                confidence=round(confidence, 3),
                reasoning=" | ".join(reasons),
                factors=factors
            )

        # ==================
        # Priority 4: Warning Level
        # ==================
        if ae_score > self.ae_warning or gbm_risk > self.gbm_warning:
            if ae_score > self.ae_warning:
                reasons.append(f"AE score ({ae_score:.3f}) > warning threshold ({self.ae_warning})")
            if gbm_risk > self.gbm_warning:
                reasons.append(f"GBM risk ({gbm_risk:.3f}) > warning threshold ({self.gbm_warning})")

            # 낮은 신뢰도
            confidence = max(ae_score / self.ae_critical, gbm_risk / self.gbm_critical)
            confidence = min(0.6, confidence * 0.5)

            return DecisionResult(
                state="WARNING",
                confidence=round(confidence, 3),
                reasoning=" | ".join(reasons),
                factors=factors
            )

        # ==================
        # Priority 5: Normal
        # ==================
        reasons.append("All indicators within normal range")
        reasons.append(f"AE: {ae_score:.3f} <= {self.ae_warning}")
        reasons.append(f"GBM: {gbm_risk:.3f} <= {self.gbm_warning}")

        # 정상 상태 신뢰도: 두 지표가 임계값에서 멀수록 높음
        confidence = 1.0 - max(ae_score / self.ae_critical, gbm_risk / self.gbm_critical)
        confidence = max(0.5, min(0.95, confidence))

        return DecisionResult(
            state="NORMAL",
            confidence=round(confidence, 3),
            reasoning=" | ".join(reasons),
            factors=factors
        )


# =====================
# LangGraph 노드 함수
# =====================

def decide_node(state: dict) -> dict:
    """
    LangGraph 노드: 최종 상태 결정

    이 노드는 SPARQL, AutoEncoder, GBM 결과를 통합하여
    최종 상태를 결정합니다.

    Args:
        state: AgentState 딕셔너리

    Returns:
        dict: 업데이트된 상태 필드
    """
    engine = DecisionEngine()

    result = engine.decide(
        sparql_alert=state["sparql_alert"],
        ae_score=state["ae_score"],
        gbm_risk=state["gbm_risk"]
    )

    return {
        "final_state": result.state,
        "confidence": result.confidence,
        "message": f"Decision: [{result.state}] Confidence: {result.confidence:.2f} | {result.reasoning}"
    }


# =====================
# 라우팅 함수 (조건부 엣지용)
# =====================

def should_alert(state: dict) -> str:
    """
    조건부 엣지: 알림 전송 여부 결정

    Args:
        state: AgentState

    Returns:
        str: "send_alert" or "end"
    """
    if state["final_state"] in ["CRITICAL", "WARNING"]:
        return "send_alert"
    return "end"


if __name__ == "__main__":
    print("=" * 60)
    print("  Decision Engine Test")
    print("=" * 60)

    engine = DecisionEngine()

    # 테스트 케이스
    test_cases = [
        # (sparql_alert, ae_score, gbm_risk, expected_state)
        (False, 0.10, 0.20, "NORMAL"),
        (False, 0.40, 0.30, "WARNING"),
        (False, 0.30, 0.70, "WARNING"),
        (False, 0.40, 0.70, "CRITICAL"),
        (True,  0.10, 0.20, "CRITICAL"),  # SPARQL alert
    ]

    print("\n테스트 케이스:")
    print("-" * 60)
    print(f"  {'SPARQL':^8} {'AE':^8} {'GBM':^8} {'Expected':^10} {'Actual':^10} {'Pass':^6}")
    print("-" * 60)

    for sparql, ae, gbm, expected in test_cases:
        result = engine.decide(sparql, ae, gbm)
        passed = "Yes" if result.state == expected else "NO!"
        print(f"  {str(sparql):^8} {ae:^8.2f} {gbm:^8.2f} {expected:^10} {result.state:^10} {passed:^6}")

    # 상세 결과 출력
    print("\n상세 결과 예시 (Case: SPARQL=False, AE=0.40, GBM=0.70):")
    result = engine.decide(False, 0.40, 0.70)
    print(f"  State: {result.state}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Reasoning: {result.reasoning}")

    print("\n" + "=" * 60)
    print("  Test Complete!")
    print("=" * 60)
