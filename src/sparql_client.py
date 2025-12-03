"""
SPARQL Client for Fuseki Triple Store
======================================

Apache Jena Fuseki에서 Vital Sign 이상을 감지하는 SPARQL 쿼리 클라이언트

이 모듈은 규칙 기반(Rule-based) 이상 탐지를 수행합니다.
심박수 > 110 또는 호흡수 < 8 인 경우 Alert를 발생시킵니다.

Mock 모드를 지원하여 실제 Fuseki 서버 없이도 테스트할 수 있습니다.

Author: KAIST AI Graduate School CAIO 9

References:
    [1] Apache Jena Fuseki: https://jena.apache.org/documentation/fuseki2/
    [2] SPARQL 1.1 Query Language: https://www.w3.org/TR/sparql11-query/
"""

import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SPARQLResult:
    """SPARQL 쿼리 결과"""
    alert: bool
    bindings: List[Dict[str, Any]]
    raw_response: Optional[Dict] = None


class SPARQLClient:
    """
    Fuseki SPARQL 쿼리 클라이언트

    독거노인 Vital Sign 데이터에서 규칙 기반 이상을 탐지합니다.

    Args:
        endpoint: Fuseki SPARQL 엔드포인트 URL
        use_mock: Mock 모드 사용 여부 (default: True)
        timeout: 요청 타임아웃 (초)

    Example:
        >>> client = SPARQLClient(use_mock=True)
        >>> result = client.query_vital_alert(
        ...     patient_id="P001",
        ...     heart_rate=120,  # 이상!
        ...     breathing_rate=16
        ... )
        >>> print(result.alert)  # True
    """

    # SPARQL 쿼리 템플릿 (Fuseki용)
    VITAL_ALERT_QUERY = """
    PREFIX : <http://kaist.ac.kr/elderly#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT ?patient ?hr ?br
    WHERE {
        ?o a :VitalObservation ;
           :aboutPatient ?patient ;
           :hrValue ?hr ;
           :brValue ?br ;
           :observedAt ?t .
        FILTER(?hr > 110 || ?br < 8)
    }
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:3030/elderly/sparql",
        use_mock: bool = True,
        timeout: int = 10
    ):
        self.endpoint = endpoint
        self.use_mock = use_mock
        self.timeout = timeout

        # 규칙 기반 임계값
        self.hr_max = 110  # 심박수 상한
        self.hr_min = 50   # 심박수 하한
        self.br_max = 25   # 호흡수 상한
        self.br_min = 8    # 호흡수 하한

    def query_vital_alert(
        self,
        patient_id: str,
        heart_rate: float,
        breathing_rate: float
    ) -> SPARQLResult:
        """
        Vital Sign 이상 여부 쿼리

        규칙:
            - 심박수 > 110 bpm 또는 < 50 bpm → Alert
            - 호흡수 > 25 /min 또는 < 8 /min → Alert

        Args:
            patient_id: 환자 ID
            heart_rate: 현재 심박수 (bpm)
            breathing_rate: 현재 호흡수 (breaths/min)

        Returns:
            SPARQLResult: 쿼리 결과 (alert, bindings)
        """
        if self.use_mock:
            return self._mock_query(patient_id, heart_rate, breathing_rate)
        else:
            return self._real_query(patient_id, heart_rate, breathing_rate)

    def _mock_query(
        self,
        patient_id: str,
        heart_rate: float,
        breathing_rate: float
    ) -> SPARQLResult:
        """
        Mock 쿼리 (실제 Fuseki 없이 규칙 기반 판단)

        실제 SPARQL 쿼리를 시뮬레이션합니다.
        """
        logger.info(f"[MOCK SPARQL] Patient: {patient_id}, HR: {heart_rate}, BR: {breathing_rate}")

        # 규칙 기반 이상 탐지
        hr_alert = heart_rate > self.hr_max or heart_rate < self.hr_min
        br_alert = breathing_rate > self.br_max or breathing_rate < self.br_min
        alert = hr_alert or br_alert

        # 결과 바인딩 생성
        bindings = []
        if alert:
            bindings.append({
                "patient": {"type": "uri", "value": f"http://kaist.ac.kr/elderly#{patient_id}"},
                "hr": {"type": "literal", "value": str(heart_rate)},
                "br": {"type": "literal", "value": str(breathing_rate)},
            })

        # 상세 정보
        details = {
            "hr_alert": hr_alert,
            "br_alert": br_alert,
            "hr_range": f"{self.hr_min}-{self.hr_max}",
            "br_range": f"{self.br_min}-{self.br_max}",
            "hr_actual": heart_rate,
            "br_actual": breathing_rate,
        }

        if hr_alert:
            if heart_rate > self.hr_max:
                details["hr_reason"] = f"Tachycardia (HR > {self.hr_max})"
            else:
                details["hr_reason"] = f"Bradycardia (HR < {self.hr_min})"

        if br_alert:
            if breathing_rate < self.br_min:
                details["br_reason"] = f"Bradypnea (BR < {self.br_min})"
            else:
                details["br_reason"] = f"Tachypnea (BR > {self.br_max})"

        return SPARQLResult(
            alert=alert,
            bindings=bindings,
            raw_response={"mock": True, "details": details}
        )

    def _real_query(
        self,
        patient_id: str,
        heart_rate: float,
        breathing_rate: float
    ) -> SPARQLResult:
        """
        실제 Fuseki 서버에 SPARQL 쿼리 실행

        Note: 이 메서드는 실제 Fuseki 서버가 필요합니다.
        """
        logger.info(f"[REAL SPARQL] Querying {self.endpoint}")

        try:
            # SPARQL 쿼리 실행
            response = requests.post(
                self.endpoint,
                data={"query": self.VITAL_ALERT_QUERY},
                headers={"Accept": "application/sparql-results+json"},
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            bindings = result.get("results", {}).get("bindings", [])
            alert = len(bindings) > 0

            return SPARQLResult(
                alert=alert,
                bindings=bindings,
                raw_response=result
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"SPARQL query failed: {e}")
            # Fallback to mock
            logger.warning("Falling back to mock query")
            return self._mock_query(patient_id, heart_rate, breathing_rate)

    def check_connection(self) -> bool:
        """Fuseki 서버 연결 확인"""
        if self.use_mock:
            return True

        try:
            response = requests.get(
                self.endpoint.replace("/sparql", "/$/ping"),
                timeout=5
            )
            return response.status_code == 200
        except:
            return False


# =====================
# LangGraph 노드 함수
# =====================

def fetch_sparql_node(state: dict) -> dict:
    """
    LangGraph 노드: SPARQL 쿼리 실행

    이 함수는 LangGraph StateGraph의 노드로 사용됩니다.
    AgentState를 받아 SPARQL 쿼리를 실행하고 결과를 업데이트합니다.

    Args:
        state: AgentState 딕셔너리

    Returns:
        dict: 업데이트된 상태 필드
    """
    client = SPARQLClient(use_mock=True)

    result = client.query_vital_alert(
        patient_id=state["patient_id"],
        heart_rate=state["hr_current"],
        breathing_rate=state["br_current"]
    )

    return {
        "sparql_alert": result.alert,
        "sparql_details": result.raw_response,
        "message": f"SPARQL query completed. Alert: {result.alert}"
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  SPARQL Client Test")
    print("=" * 60)

    client = SPARQLClient(use_mock=True)

    # 테스트 케이스
    test_cases = [
        ("P001", 72, 16, "Normal"),
        ("P002", 120, 16, "Tachycardia (HR > 110)"),
        ("P003", 45, 16, "Bradycardia (HR < 50)"),
        ("P004", 72, 6, "Bradypnea (BR < 8)"),
        ("P005", 130, 5, "Critical - Multiple Alerts"),
    ]

    print("\n테스트 케이스:")
    print("-" * 60)

    for patient_id, hr, br, expected in test_cases:
        result = client.query_vital_alert(patient_id, hr, br)
        status = "ALERT" if result.alert else "NORMAL"
        print(f"  {patient_id}: HR={hr:3d}, BR={br:2d} → [{status:6s}] {expected}")

    print("\n" + "=" * 60)
    print("  Test Complete!")
    print("=" * 60)
