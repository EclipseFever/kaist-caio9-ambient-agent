"""
LangGraph Ambient Agent for Elderly Care
=========================================

SPARQL 규칙, AutoEncoder 이상도, GBM 위험도를 통합하는 LangGraph 기반 Ambient Agent

이 모듈은 StateGraph를 사용하여 다음 워크플로우를 구현합니다:
    START → fetch_sparql → detect_anomaly → calculate_risk → decide → [send_alert] → END

Author: KAIST AI Graduate School CAIO 9

References:
    [1] LangGraph Documentation: https://langchain-ai.github.io/langgraph/
    [2] LangChain Ambient Agents: https://blog.langchain.com/introducing-ambient-agents/
"""

from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, START, END
import logging

# 내부 모듈 임포트
from .state import AgentState, create_initial_state, get_state_summary
from .sparql_client import fetch_sparql_node
from .anomaly_detector import detect_anomaly_node
from .risk_scorer import calculate_risk_node
from .decision_engine import decide_node, should_alert
from .alert_sender import send_alert_node

logger = logging.getLogger(__name__)


def create_ambient_agent() -> StateGraph:
    """
    LangGraph Ambient Agent 생성

    이 함수는 독거노인 Vital Sign 모니터링을 위한
    Ambient Agent StateGraph를 생성하고 반환합니다.

    Workflow:
        ```
        [START]
            ↓
        [fetch_sparql] ─→ Fuseki 쿼리로 규칙 기반 이상 확인
            ↓
        [detect_anomaly] ─→ AutoEncoder 재구성 오차 계산
            ↓
        [calculate_risk] ─→ GBM 모델로 위험도 예측
            ↓
        [decide] ─→ 3가지 신호 통합 → 최종 상태 결정
            ↓
        [conditional_edge: should_alert?]
            ├─ CRITICAL/WARNING → [send_alert] → [END]
            └─ NORMAL → [END]
        ```

    Returns:
        CompiledGraph: 컴파일된 LangGraph 에이전트

    Example:
        >>> agent = create_ambient_agent()
        >>> state = create_initial_state(
        ...     patient_id="P001",
        ...     hr_series=[72.0] * 60,
        ...     br_series=[16.0] * 60
        ... )
        >>> result = agent.invoke(state)
        >>> print(result["final_state"])  # "NORMAL"
    """

    # =====================
    # StateGraph 생성
    # =====================
    workflow = StateGraph(AgentState)

    # =====================
    # 노드 추가
    # =====================

    # Node 1: SPARQL 쿼리 (규칙 기반 이상 감지)
    workflow.add_node("fetch_sparql", fetch_sparql_node)

    # Node 2: AutoEncoder 이상 탐지
    workflow.add_node("detect_anomaly", detect_anomaly_node)

    # Node 3: GBM 위험도 예측
    workflow.add_node("calculate_risk", calculate_risk_node)

    # Node 4: 최종 상태 결정
    workflow.add_node("decide", decide_node)

    # Node 5: 알림 전송
    workflow.add_node("send_alert", send_alert_node)

    # =====================
    # 엣지 추가 (순차 연결)
    # =====================

    # START → fetch_sparql
    workflow.add_edge(START, "fetch_sparql")

    # fetch_sparql → detect_anomaly
    workflow.add_edge("fetch_sparql", "detect_anomaly")

    # detect_anomaly → calculate_risk
    workflow.add_edge("detect_anomaly", "calculate_risk")

    # calculate_risk → decide
    workflow.add_edge("calculate_risk", "decide")

    # =====================
    # 조건부 엣지 (알림 전송 여부)
    # =====================
    workflow.add_conditional_edges(
        "decide",
        should_alert,
        {
            "send_alert": "send_alert",
            "end": END
        }
    )

    # send_alert → END
    workflow.add_edge("send_alert", END)

    # =====================
    # 그래프 컴파일
    # =====================
    agent = workflow.compile()

    logger.info("Ambient Agent created and compiled successfully")

    return agent


class AmbientAgentRunner:
    """
    Ambient Agent 실행 관리자

    LangGraph Ambient Agent의 생성, 실행, 모니터링을 관리합니다.

    Example:
        >>> runner = AmbientAgentRunner()
        >>> result = runner.run(
        ...     patient_id="P001",
        ...     hr_series=[72.0] * 60,
        ...     br_series=[16.0] * 60
        ... )
        >>> print(result["final_state"])
    """

    def __init__(self):
        self.agent = create_ambient_agent()
        self._run_count = 0

    def run(
        self,
        patient_id: str,
        hr_series: list,
        br_series: list,
        verbose: bool = True
    ) -> AgentState:
        """
        에이전트 실행

        Args:
            patient_id: 환자 ID
            hr_series: 심박수 시계열 (60 samples)
            br_series: 호흡수 시계열 (60 samples)
            verbose: 상세 출력 여부

        Returns:
            AgentState: 최종 상태
        """
        self._run_count += 1

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Ambient Agent Run #{self._run_count}")
            print(f"{'='*60}")
            print(f"  Patient: {patient_id}")
            print(f"  HR samples: {len(hr_series)}, BR samples: {len(br_series)}")

        # 초기 상태 생성
        initial_state = create_initial_state(
            patient_id=patient_id,
            hr_series=hr_series,
            br_series=br_series
        )

        if verbose:
            print(f"  Current HR: {initial_state['hr_current']:.1f} bpm")
            print(f"  Current BR: {initial_state['br_current']:.1f} /min")
            print(f"{'='*60}\n")

        # 에이전트 실행
        result = self.agent.invoke(initial_state)

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Result: [{result['final_state']}]")
            print(f"{'='*60}")
            print(f"  SPARQL Alert: {result['sparql_alert']}")
            print(f"  AE Score: {result['ae_score']:.4f}")
            print(f"  GBM Risk: {result['gbm_risk']:.4f}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Alert Sent: {result['alert_sent']}")
            print(f"{'='*60}\n")

        return result

    def stream(
        self,
        patient_id: str,
        hr_series: list,
        br_series: list
    ):
        """
        스트리밍 모드로 에이전트 실행

        각 노드의 실행 결과를 실시간으로 반환합니다.

        Args:
            patient_id: 환자 ID
            hr_series: 심박수 시계열
            br_series: 호흡수 시계열

        Yields:
            Dict: 각 노드의 실행 결과
        """
        initial_state = create_initial_state(
            patient_id=patient_id,
            hr_series=hr_series,
            br_series=br_series
        )

        for event in self.agent.stream(initial_state):
            yield event


def run_demo():
    """데모 실행"""
    import numpy as np

    print("\n" + "="*70)
    print("  LangGraph Ambient Agent Demo")
    print("  KAIST AI Graduate School CAIO 9")
    print("="*70)

    runner = AmbientAgentRunner()

    # =====================
    # Case 1: 정상 상태
    # =====================
    print("\n" + "-"*70)
    print("  CASE 1: Normal Vital Signs")
    print("-"*70)

    normal_hr = list(np.random.normal(72, 3, 60))
    normal_br = list(np.random.normal(16, 1, 60))
    result = runner.run("P001", normal_hr, normal_br)

    # =====================
    # Case 2: 경미한 이상 (WARNING)
    # =====================
    print("\n" + "-"*70)
    print("  CASE 2: Elevated Vital Signs (WARNING)")
    print("-"*70)

    elevated_hr = list(np.random.normal(100, 8, 60))
    elevated_br = list(np.random.normal(22, 3, 60))
    result = runner.run("P002", elevated_hr, elevated_br)

    # =====================
    # Case 3: 심정지 패턴 (CRITICAL)
    # =====================
    print("\n" + "-"*70)
    print("  CASE 3: Cardiac Arrest Pattern (CRITICAL)")
    print("-"*70)

    arrest_hr = list(np.linspace(72, 25, 60))  # 심박수 급감
    arrest_br = list(np.linspace(16, 4, 60))   # 호흡수 급감
    result = runner.run("P003", arrest_hr, arrest_br)

    print("\n" + "="*70)
    print("  Demo Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    run_demo()
