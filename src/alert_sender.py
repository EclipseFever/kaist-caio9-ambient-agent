"""
Alert Sender for Notification Dispatch
======================================

CRITICAL/WARNING 상태 발생 시 알림을 전송하는 모듈

알림 대상:
    - 모바일 앱 (보호자)
    - 대시보드 (관제 센터)
    - 간호스테이션

Author: KAIST AI Graduate School CAIO 9
"""

import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class AlertPayload:
    """알림 페이로드"""
    patient_id: str
    state: str              # "CRITICAL" | "WARNING"
    timestamp: str
    heart_rate: float
    breathing_rate: float
    ae_score: float
    gbm_risk: float
    message: str
    targets: List[str] = field(default_factory=lambda: ["mobile", "dashboard", "nursing_station"])


@dataclass
class AlertResponse:
    """알림 전송 응답"""
    success: bool
    sent_to: List[str]
    failed: List[str]
    response_data: Optional[Dict] = None
    error_message: Optional[str] = None


class AlertSender:
    """
    알림 전송기

    CRITICAL 또는 WARNING 상태 발생 시 지정된 엔드포인트로 알림을 전송합니다.

    Args:
        endpoint: 알림 서버 엔드포인트
        use_mock: Mock 모드 (True면 콘솔 출력만)
        retry_count: 재시도 횟수
        retry_delay: 재시도 간격 (초)

    Example:
        >>> sender = AlertSender(use_mock=True)
        >>> response = sender.send(
        ...     patient_id="P001",
        ...     state="CRITICAL",
        ...     heart_rate=130,
        ...     breathing_rate=6
        ... )
        >>> print(response.success)  # True
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8000/notify",
        use_mock: bool = True,
        retry_count: int = 3,
        retry_delay: float = 1.0
    ):
        self.endpoint = endpoint
        self.use_mock = use_mock
        self.retry_count = retry_count
        self.retry_delay = retry_delay

        # 알림 대상별 엔드포인트
        self.target_endpoints = {
            "mobile": f"{endpoint}/mobile",
            "dashboard": f"{endpoint}/dashboard",
            "nursing_station": f"{endpoint}/nursing",
        }

    def send(
        self,
        patient_id: str,
        state: str,
        heart_rate: float,
        breathing_rate: float,
        ae_score: float = 0.0,
        gbm_risk: float = 0.0,
        message: str = "",
        targets: Optional[List[str]] = None
    ) -> AlertResponse:
        """
        알림 전송

        Args:
            patient_id: 환자 ID
            state: 상태 ("CRITICAL" | "WARNING")
            heart_rate: 현재 심박수
            breathing_rate: 현재 호흡수
            ae_score: AutoEncoder 이상도
            gbm_risk: GBM 위험도
            message: 추가 메시지
            targets: 알림 대상 목록

        Returns:
            AlertResponse: 전송 결과
        """
        if targets is None:
            targets = ["mobile", "dashboard", "nursing_station"]

        payload = AlertPayload(
            patient_id=patient_id,
            state=state,
            timestamp=datetime.now().isoformat(),
            heart_rate=heart_rate,
            breathing_rate=breathing_rate,
            ae_score=ae_score,
            gbm_risk=gbm_risk,
            message=message,
            targets=targets
        )

        if self.use_mock:
            return self._mock_send(payload)
        else:
            return self._real_send(payload)

    def _mock_send(self, payload: AlertPayload) -> AlertResponse:
        """
        Mock 알림 전송 (콘솔 출력)
        """
        logger.info(f"[MOCK ALERT] Sending alert for patient {payload.patient_id}")

        # 콘솔에 알림 출력
        alert_box = self._format_alert_box(payload)
        print(alert_box)

        return AlertResponse(
            success=True,
            sent_to=payload.targets,
            failed=[],
            response_data={"mock": True, "payload": payload.__dict__}
        )

    def _real_send(self, payload: AlertPayload) -> AlertResponse:
        """
        실제 HTTP 알림 전송
        """
        sent_to = []
        failed = []

        for target in payload.targets:
            endpoint = self.target_endpoints.get(target, self.endpoint)

            for attempt in range(self.retry_count):
                try:
                    response = requests.post(
                        endpoint,
                        json={
                            "patient": payload.patient_id,
                            "state": payload.state,
                            "timestamp": payload.timestamp,
                            "hr": payload.heart_rate,
                            "br": payload.breathing_rate,
                            "ae_score": payload.ae_score,
                            "gbm_risk": payload.gbm_risk,
                            "message": payload.message,
                        },
                        timeout=5
                    )
                    response.raise_for_status()
                    sent_to.append(target)
                    break

                except requests.exceptions.RequestException as e:
                    logger.warning(f"Alert to {target} failed (attempt {attempt+1}): {e}")
                    if attempt == self.retry_count - 1:
                        failed.append(target)

        return AlertResponse(
            success=len(failed) == 0,
            sent_to=sent_to,
            failed=failed,
            response_data={"payload": payload.__dict__}
        )

    def _format_alert_box(self, payload: AlertPayload) -> str:
        """알림 박스 포맷팅"""
        state_emoji = "🚨" if payload.state == "CRITICAL" else "⚠️"
        state_color = "RED" if payload.state == "CRITICAL" else "YELLOW"

        box = f"""
╔══════════════════════════════════════════════════════════════╗
║  {state_emoji} ALERT: {payload.state:^10} {state_emoji}                                    ║
╠══════════════════════════════════════════════════════════════╣
║  Patient ID  : {payload.patient_id:<45} ║
║  Timestamp   : {payload.timestamp:<45} ║
╠══════════════════════════════════════════════════════════════╣
║  Heart Rate  : {payload.heart_rate:>6.1f} bpm                                    ║
║  Breath Rate : {payload.breathing_rate:>6.1f} /min                                   ║
║  AE Score    : {payload.ae_score:>6.3f}                                         ║
║  GBM Risk    : {payload.gbm_risk:>6.3f}                                         ║
╠══════════════════════════════════════════════════════════════╣
║  Sent to: {', '.join(payload.targets):<50} ║
╚══════════════════════════════════════════════════════════════╝
"""
        return box


# =====================
# LangGraph 노드 함수
# =====================

def send_alert_node(state: dict) -> dict:
    """
    LangGraph 노드: 알림 전송

    CRITICAL 또는 WARNING 상태인 경우 알림을 전송합니다.

    Args:
        state: AgentState 딕셔너리

    Returns:
        dict: 업데이트된 상태 필드
    """
    sender = AlertSender(use_mock=True)

    response = sender.send(
        patient_id=state["patient_id"],
        state=state["final_state"],
        heart_rate=state["hr_current"],
        breathing_rate=state["br_current"],
        ae_score=state["ae_score"],
        gbm_risk=state["gbm_risk"],
        message=state.get("message", "")
    )

    return {
        "alert_sent": response.success,
        "alert_response": {
            "sent_to": response.sent_to,
            "failed": response.failed,
        },
        "message": f"Alert sent to: {', '.join(response.sent_to)}"
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  Alert Sender Test")
    print("=" * 60)

    sender = AlertSender(use_mock=True)

    # CRITICAL 알림 테스트
    print("\n[Test 1] CRITICAL Alert:")
    response = sender.send(
        patient_id="P001",
        state="CRITICAL",
        heart_rate=130,
        breathing_rate=6,
        ae_score=0.45,
        gbm_risk=0.78,
        message="Cardiac arrest suspected"
    )
    print(f"Success: {response.success}, Sent to: {response.sent_to}")

    # WARNING 알림 테스트
    print("\n[Test 2] WARNING Alert:")
    response = sender.send(
        patient_id="P002",
        state="WARNING",
        heart_rate=105,
        breathing_rate=22,
        ae_score=0.32,
        gbm_risk=0.55,
        message="Elevated vital signs"
    )
    print(f"Success: {response.success}, Sent to: {response.sent_to}")

    print("\n" + "=" * 60)
    print("  Test Complete!")
    print("=" * 60)
