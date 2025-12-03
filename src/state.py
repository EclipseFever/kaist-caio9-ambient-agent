"""
AgentState Definition for LangGraph Ambient Agent
==================================================

LangGraph StateGraph에서 사용하는 상태(State) 타입 정의

이 모듈은 Ambient Agent의 전체 워크플로우에서 공유되는
상태 정보를 TypedDict로 정의합니다.

Author: KAIST AI Graduate School CAIO 9
"""

from typing import TypedDict, Optional, List, Any
from dataclasses import dataclass
from datetime import datetime


class AgentState(TypedDict):
    """
    LangGraph Ambient Agent의 상태 정의

    이 TypedDict는 그래프의 모든 노드에서 공유되며,
    각 노드는 이 상태를 읽고 업데이트합니다.

    Workflow:
        START → fetch_sparql → detect_anomaly → calculate_risk → decide → [send_alert] → END

    Attributes:
        patient_id (str): 환자/대상자 고유 ID
        timestamp (str): 데이터 수집 시간 (ISO format)

        hr_series (List[float]): 심박수 시계열 데이터 (최근 60초)
        br_series (List[float]): 호흡수 시계열 데이터 (최근 60초)
        hr_current (float): 현재 심박수 (bpm)
        br_current (float): 현재 호흡수 (breaths/min)

        sparql_alert (bool): SPARQL 규칙 기반 이상 감지 여부
        sparql_details (Optional[dict]): SPARQL 쿼리 결과 상세

        ae_score (float): AutoEncoder 재구성 오차 (0~1)
        ae_anomaly (bool): AutoEncoder 이상 탐지 결과
        ae_details (Optional[dict]): AutoEncoder 분석 상세

        gbm_risk (float): GBM 모델 위험 확률 (0~1)
        gbm_details (Optional[dict]): GBM 예측 상세

        final_state (str): 최종 판단 상태 ("CRITICAL" | "WARNING" | "NORMAL")
        confidence (float): 판단 신뢰도 (0~1)

        alert_sent (bool): 알림 전송 완료 여부
        alert_response (Optional[dict]): 알림 서버 응답

        message (str): 상태 메시지
        errors (List[str]): 처리 중 발생한 에러 목록
    """

    # =====================
    # 입력 데이터
    # =====================
    patient_id: str
    timestamp: str

    # Vital Sign 시계열 데이터
    hr_series: List[float]  # Heart Rate series (60 samples)
    br_series: List[float]  # Breathing Rate series (60 samples)

    # 현재 Vital Sign 값
    hr_current: float  # Current Heart Rate (bpm)
    br_current: float  # Current Breathing Rate (breaths/min)

    # =====================
    # SPARQL 쿼리 결과
    # =====================
    sparql_alert: bool
    sparql_details: Optional[dict]

    # =====================
    # AutoEncoder 결과
    # =====================
    ae_score: float      # Reconstruction error (0~1)
    ae_anomaly: bool     # Is anomaly detected?
    ae_details: Optional[dict]

    # =====================
    # GBM 위험도 결과
    # =====================
    gbm_risk: float      # Risk probability (0~1)
    gbm_details: Optional[dict]

    # =====================
    # 최종 판단
    # =====================
    final_state: str     # "CRITICAL" | "WARNING" | "NORMAL"
    confidence: float    # Decision confidence (0~1)

    # =====================
    # 알림 전송
    # =====================
    alert_sent: bool
    alert_response: Optional[dict]

    # =====================
    # 메타 정보
    # =====================
    message: str
    errors: List[str]


def create_initial_state(
    patient_id: str,
    hr_series: List[float],
    br_series: List[float]
) -> AgentState:
    """
    초기 AgentState 생성

    Args:
        patient_id: 환자/대상자 ID
        hr_series: 심박수 시계열 (60 samples)
        br_series: 호흡수 시계열 (60 samples)

    Returns:
        AgentState: 초기화된 상태 딕셔너리

    Example:
        >>> state = create_initial_state(
        ...     patient_id="P001",
        ...     hr_series=[72.0] * 60,
        ...     br_series=[16.0] * 60
        ... )
    """
    return AgentState(
        # 입력 데이터
        patient_id=patient_id,
        timestamp=datetime.now().isoformat(),
        hr_series=hr_series,
        br_series=br_series,
        hr_current=hr_series[-1] if hr_series else 0.0,
        br_current=br_series[-1] if br_series else 0.0,

        # SPARQL 결과 (초기값)
        sparql_alert=False,
        sparql_details=None,

        # AutoEncoder 결과 (초기값)
        ae_score=0.0,
        ae_anomaly=False,
        ae_details=None,

        # GBM 결과 (초기값)
        gbm_risk=0.0,
        gbm_details=None,

        # 최종 판단 (초기값)
        final_state="NORMAL",
        confidence=0.0,

        # 알림 (초기값)
        alert_sent=False,
        alert_response=None,

        # 메타 정보
        message="Initialized",
        errors=[],
    )


# =====================
# 상태 유틸리티 함수
# =====================

def get_state_summary(state: AgentState) -> str:
    """상태 요약 문자열 생성"""
    return (
        f"[{state['final_state']}] Patient: {state['patient_id']} | "
        f"HR: {state['hr_current']:.1f} bpm | BR: {state['br_current']:.1f} /min | "
        f"AE: {state['ae_score']:.3f} | GBM: {state['gbm_risk']:.3f} | "
        f"Alert: {'Sent' if state['alert_sent'] else 'No'}"
    )


def is_critical(state: AgentState) -> bool:
    """CRITICAL 상태인지 확인"""
    return state["final_state"] == "CRITICAL"


def is_warning(state: AgentState) -> bool:
    """WARNING 상태인지 확인"""
    return state["final_state"] == "WARNING"


def needs_alert(state: AgentState) -> bool:
    """알림이 필요한지 확인"""
    return state["final_state"] in ["CRITICAL", "WARNING"]


if __name__ == "__main__":
    # 테스트
    import numpy as np

    print("=" * 60)
    print("  AgentState Test")
    print("=" * 60)

    # 정상 상태 테스트
    normal_hr = list(np.random.normal(72, 2, 60))
    normal_br = list(np.random.normal(16, 1, 60))

    state = create_initial_state(
        patient_id="P001",
        hr_series=normal_hr,
        br_series=normal_br
    )

    print(f"\nInitial State:")
    print(f"  Patient ID: {state['patient_id']}")
    print(f"  Timestamp: {state['timestamp']}")
    print(f"  HR Current: {state['hr_current']:.1f} bpm")
    print(f"  BR Current: {state['br_current']:.1f} /min")
    print(f"  Final State: {state['final_state']}")

    print("\n" + "=" * 60)
    print("  Test Complete!")
    print("=" * 60)
