#!/usr/bin/env python3
"""
LangGraph Ambient Agent for Elderly Care - Main Entry Point
============================================================

독거노인 안전 돌봄을 위한 LangGraph 기반 Ambient Agent 시스템

이 스크립트는 SPARQL 규칙 결과, CNN AutoEncoder 이상도, GBM 위험도를
LangGraph로 통합해 최종 판단 후 실시간 알림을 전송하는 데모를 실행합니다.

Usage:
    python main.py

Author: KAIST AI Graduate School CAIO 9

Project Structure:
    - src/state.py: AgentState 정의
    - src/sparql_client.py: SPARQL 쿼리 클라이언트
    - src/anomaly_detector.py: AutoEncoder 이상 탐지
    - src/risk_scorer.py: GBM 위험도 예측
    - src/decision_engine.py: 최종 상태 판단
    - src/alert_sender.py: 알림 전송
    - src/ambient_agent.py: LangGraph StateGraph
"""

import sys
import numpy as np
import logging
from datetime import datetime

# 내부 모듈 임포트
from src.ambient_agent import AmbientAgentRunner, create_ambient_agent
from src.state import create_initial_state, get_state_summary


def print_banner():
    """배너 출력"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║     ██╗      █████╗ ███╗   ██╗ ██████╗  ██████╗ ██████╗  █████╗ ██████╗ ██╗  ║
║     ██║     ██╔══██╗████╗  ██║██╔════╝ ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ║
║     ██║     ███████║██╔██╗ ██║██║  ███╗██║  ███╗██████╔╝███████║██████╔╝██████║
║     ██║     ██╔══██║██║╚██╗██║██║   ██║██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║
║     ███████╗██║  ██║██║ ╚████║╚██████╔╝╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║
║     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝
║                                                                               ║
║                    Ambient Agent for Elderly Care                             ║
║                                                                               ║
║              KAIST 김재철AI대학원 CAIO 9기                                    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_header(title: str):
    """섹션 헤더 출력"""
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}\n")


def generate_normal_data(n_samples: int = 60) -> tuple:
    """정상 Vital Sign 데이터 생성"""
    hr = list(np.random.normal(72, 3, n_samples))
    br = list(np.random.normal(16, 1, n_samples))
    return hr, br


def generate_warning_data(n_samples: int = 60) -> tuple:
    """경고 수준 Vital Sign 데이터 생성"""
    hr = list(np.random.normal(105, 8, n_samples))
    br = list(np.random.normal(22, 3, n_samples))
    return hr, br


def generate_critical_data(n_samples: int = 60) -> tuple:
    """위급 상황 Vital Sign 데이터 생성 (심정지 패턴)"""
    # 심박수: 72에서 시작해서 점점 감소
    hr = list(np.linspace(72, 20, n_samples))
    # 호흡수: 16에서 시작해서 점점 감소
    br = list(np.linspace(16, 4, n_samples))
    return hr, br


def generate_sparql_trigger_data(n_samples: int = 60) -> tuple:
    """SPARQL 규칙 트리거 데이터 (심박수 > 110)"""
    hr = list(np.random.normal(125, 5, n_samples))
    br = list(np.random.normal(18, 2, n_samples))
    return hr, br


def run_single_case(runner: AmbientAgentRunner, patient_id: str,
                    hr_series: list, br_series: list, case_name: str):
    """단일 케이스 실행"""
    print(f"\n{'─'*75}")
    print(f"  📊 {case_name}")
    print(f"{'─'*75}")

    result = runner.run(patient_id, hr_series, br_series, verbose=False)

    # 결과 출력
    state_emoji = {
        "NORMAL": "🟢",
        "WARNING": "🟡",
        "CRITICAL": "🔴"
    }
    emoji = state_emoji.get(result["final_state"], "⚪")

    print(f"\n  Patient ID    : {patient_id}")
    print(f"  Final State   : {emoji} {result['final_state']}")
    print(f"  Confidence    : {result['confidence']:.1%}")
    print(f"  ─────────────────────────────────────")
    print(f"  SPARQL Alert  : {'Yes ⚠️' if result['sparql_alert'] else 'No'}")
    print(f"  AE Score      : {result['ae_score']:.4f} {'(Anomaly)' if result['ae_anomaly'] else '(Normal)'}")
    print(f"  GBM Risk      : {result['gbm_risk']:.4f}")
    print(f"  Alert Sent    : {'Yes 📤' if result['alert_sent'] else 'No'}")

    return result


def main():
    """메인 실행 함수"""

    # 로깅 설정
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 배너 출력
    print_banner()

    print(f"  시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # =====================
    # Phase 1: Agent 생성
    # =====================
    print_header("Phase 1: LangGraph Ambient Agent 생성")

    print("  🔧 Agent 생성 중...")
    runner = AmbientAgentRunner()
    print("  ✅ Agent 생성 완료!\n")

    print("  Workflow:")
    print("    [START]")
    print("       ↓")
    print("    [fetch_sparql] ─→ SPARQL 규칙 기반 이상 확인")
    print("       ↓")
    print("    [detect_anomaly] ─→ AutoEncoder 재구성 오차 계산")
    print("       ↓")
    print("    [calculate_risk] ─→ GBM 모델 위험도 예측")
    print("       ↓")
    print("    [decide] ─→ 3가지 신호 통합 → 최종 상태 결정")
    print("       ↓")
    print("    ┌─ CRITICAL/WARNING → [send_alert] → [END]")
    print("    └─ NORMAL → [END]")

    # =====================
    # Phase 2: 테스트 실행
    # =====================
    print_header("Phase 2: 시나리오별 테스트 실행")

    results = []

    # Case 1: 정상 상태
    hr, br = generate_normal_data()
    result = run_single_case(runner, "P001", hr, br,
                             "Case 1: Normal Vital Signs (정상 상태)")
    results.append(("Normal", result))

    # Case 2: 경고 수준
    hr, br = generate_warning_data()
    result = run_single_case(runner, "P002", hr, br,
                             "Case 2: Elevated Vital Signs (경고 수준)")
    results.append(("Warning", result))

    # Case 3: 심정지 패턴
    hr, br = generate_critical_data()
    result = run_single_case(runner, "P003", hr, br,
                             "Case 3: Cardiac Arrest Pattern (심정지 패턴)")
    results.append(("Critical", result))

    # Case 4: SPARQL 규칙 트리거
    hr, br = generate_sparql_trigger_data()
    result = run_single_case(runner, "P004", hr, br,
                             "Case 4: SPARQL Rule Trigger (규칙 기반 이상)")
    results.append(("SPARQL", result))

    # =====================
    # Phase 3: 결과 요약
    # =====================
    print_header("Phase 3: 결과 요약")

    print("  ┌─────────────┬──────────────┬────────────┬──────────┬──────────┐")
    print("  │   Scenario  │  Final State │ Confidence │ AE Score │ GBM Risk │")
    print("  ├─────────────┼──────────────┼────────────┼──────────┼──────────┤")

    for scenario, result in results:
        state = result['final_state']
        conf = result['confidence']
        ae = result['ae_score']
        gbm = result['gbm_risk']
        emoji = {"NORMAL": "🟢", "WARNING": "🟡", "CRITICAL": "🔴"}.get(state, "⚪")
        print(f"  │ {scenario:^11} │ {emoji} {state:^8} │   {conf:>5.1%}   │  {ae:>6.4f} │  {gbm:>6.4f} │")

    print("  └─────────────┴──────────────┴────────────┴──────────┴──────────┘")

    # =====================
    # Phase 4: 스트리밍 데모
    # =====================
    print_header("Phase 4: LangGraph 스트리밍 데모")

    print("  📡 실시간 노드 실행 추적:\n")

    hr, br = generate_normal_data()
    initial_state = create_initial_state("P005", hr, br)

    for i, event in enumerate(runner.stream("P005", hr, br)):
        for node_name, output in event.items():
            if isinstance(output, dict) and "message" in output:
                print(f"    [{i+1}] 🔹 {node_name}: {output.get('message', '')[:50]}...")

    # =====================
    # 완료
    # =====================
    print_header("완료")

    print(f"  🎉 모든 테스트 완료!")
    print(f"  ⏱️  종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n  이 시스템은 KAIST 김재철AI대학원 CAIO 9기 과제로 개발되었습니다.")
    print(f"  LangGraph를 사용하여 Ambient Agent 패턴을 구현하였습니다.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
