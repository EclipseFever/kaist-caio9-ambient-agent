# LangGraph Ambient Agent for Elderly Care

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **KAIST 김재철AI대학원 CAIO 9기 과제**
>
> SPARQL 규칙 결과, CNN AutoEncoder 이상도, GBM 위험도를 LangGraph로 통합해
> 최종 판단 후 모바일·대시보드·간호스테이션으로 실시간 알림을 전송하는 Ambient Agent 시스템

---

## 📋 목차

- [개요](#-개요)
- [시스템 아키텍처](#-시스템-아키텍처)
- [LangGraph 워크플로우](#-langgraph-워크플로우)
- [설치 및 실행](#-설치-및-실행)
- [프로젝트 구조](#-프로젝트-구조)
- [핵심 컴포넌트](#-핵심-컴포넌트)
- [의사결정 로직](#-의사결정-로직)
- [데모 실행](#-데모-실행)
- [참고 자료](#-참고-자료)

---

## 🎯 개요

### 배경

- 국내 독거노인 약 230만 명 (65세 이상 인구의 35% 이상)
- 낙상, 심정지 등 응급상황 시 골든타임 내 대응 필요
- 기존 시스템의 한계: 오경보 빈발, 수동 호출 의존

### 해결책

**LangGraph Ambient Agent**를 활용하여:

1. **SPARQL 규칙 기반 쿼리** - 명시적 임계값 초과 감지
2. **CNN AutoEncoder** - 시계열 패턴 이상 탐지
3. **GBM 위험도 모델** - 복합 특성 기반 위험 예측
4. **통합 의사결정** - 3가지 신호 융합으로 오경보 최소화

### Ambient Agent란?

> Ambient Agent는 사용자의 적극적인 개입 없이 **백그라운드에서 지속적으로 모니터링**하며,
> 중요한 상황이 감지될 때만 **사용자에게 알림**을 보내는 AI 에이전트입니다.

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        60GHz mmWave Radar Sensor                        │
│                      (MR60BHA2 - 비접촉 생체신호 측정)                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Vital Sign Stream                             │
│                    (Heart Rate, Breathing Rate @ 1Hz)                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │    SPARQL    │ │  AutoEncoder │ │     GBM      │
            │   (Fuseki)   │ │   (CNN 1D)   │ │  (XGBoost)   │
            │              │ │              │ │              │
            │ HR > 110?    │ │ Recon Error  │ │ Risk Score   │
            │ BR < 8?      │ │ (0~1)        │ │ (0~1)        │
            └──────────────┘ └──────────────┘ └──────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
            ┌─────────────────────────────────────────────────┐
            │             LangGraph Decision Engine           │
            │                                                 │
            │  if SPARQL_alert:                               │
            │      return "CRITICAL"                          │
            │  if AE > 0.35 AND GBM > 0.6:                    │
            │      return "CRITICAL"                          │
            │  if AE > 0.35 OR GBM > 0.6:                     │
            │      return "WARNING"                           │
            │  return "NORMAL"                                │
            └─────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │   Mobile     │ │  Dashboard   │ │   Nursing    │
            │    App       │ │   (관제)     │ │   Station    │
            └──────────────┘ └──────────────┘ └──────────────┘
```

---

## 🔄 LangGraph 워크플로우

### StateGraph 구조

```
[START]
    │
    ▼
┌─────────────────┐
│  fetch_sparql   │ ← SPARQL 규칙 기반 이상 확인 (HR > 110 OR BR < 8)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ detect_anomaly  │ ← AutoEncoder 재구성 오차 계산
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ calculate_risk  │ ← GBM 모델 위험도 예측
└─────────────────┘
    │
    ▼
┌─────────────────┐
│     decide      │ ← 3가지 신호 통합 → 최종 상태 결정
└─────────────────┘
    │
    ▼
┌─────────────────┐     ┌─────────────────┐
│ CRITICAL/WARNING├────►│   send_alert    │
└─────────────────┘     └─────────────────┘
    │                           │
    │ NORMAL                    │
    ▼                           ▼
[END] ◄─────────────────────────┘
```

### 코드 예시

```python
from langgraph.graph import StateGraph, START, END
from src.state import AgentState

# StateGraph 생성
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("fetch_sparql", fetch_sparql_node)
workflow.add_node("detect_anomaly", detect_anomaly_node)
workflow.add_node("calculate_risk", calculate_risk_node)
workflow.add_node("decide", decide_node)
workflow.add_node("send_alert", send_alert_node)

# 엣지 연결
workflow.add_edge(START, "fetch_sparql")
workflow.add_edge("fetch_sparql", "detect_anomaly")
workflow.add_edge("detect_anomaly", "calculate_risk")
workflow.add_edge("calculate_risk", "decide")

# 조건부 엣지
workflow.add_conditional_edges(
    "decide",
    should_alert,
    {"send_alert": "send_alert", "end": END}
)
workflow.add_edge("send_alert", END)

# 컴파일
agent = workflow.compile()
```

---

## 🚀 설치 및 실행

### 요구사항

- Python 3.11+
- pip

### 설치

```bash
# 저장소 클론
git clone https://github.com/EclipseFever/kaist-caio9-ambient-agent.git
cd kaist-caio9-ambient-agent

# 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 실행

```bash
# 전체 데모 실행
python main.py

# 개별 모듈 테스트
python -m src.sparql_client
python -m src.anomaly_detector
python -m src.risk_scorer
python -m src.decision_engine
python -m src.alert_sender
python -m src.ambient_agent
```

---

## 📁 프로젝트 구조

```
kaist-caio9-ambient-agent/
├── README.md                    # 프로젝트 문서
├── requirements.txt             # Python 의존성
├── main.py                      # 진입점 (데모 실행)
├── .gitignore
│
├── src/                         # 소스 코드
│   ├── __init__.py
│   ├── state.py                 # AgentState TypedDict 정의
│   ├── sparql_client.py         # Fuseki SPARQL 쿼리 클라이언트
│   ├── anomaly_detector.py      # AutoEncoder 이상 탐지
│   ├── risk_scorer.py           # GBM 위험도 예측
│   ├── decision_engine.py       # 최종 상태 판단 로직
│   ├── alert_sender.py          # HTTP 알림 전송
│   └── ambient_agent.py         # LangGraph StateGraph
│
├── models/                      # 사전학습 모델
│   ├── autoencoder_v1.0.pth     # AutoEncoder 가중치
│   ├── gbm_risk_model.pkl       # GBM 모델
│   └── README.md
│
├── data/                        # 데이터
│   ├── sample_vitals.csv        # 샘플 Vital Sign 데이터
│   └── ontology/
│       └── elderly_care.ttl     # RDF 온톨로지 (선택)
│
├── config/                      # 설정
│   └── settings.yaml            # 엔드포인트, 임계값 설정
│
└── assets/                      # 시각화 자료
    ├── langgraph_flow.png
    └── architecture.png
```

---

## 🧩 핵심 컴포넌트

### 1. AgentState (`src/state.py`)

LangGraph에서 사용하는 상태 정의:

```python
class AgentState(TypedDict):
    patient_id: str
    hr_series: List[float]       # 심박수 시계열 (60초)
    br_series: List[float]       # 호흡수 시계열 (60초)

    sparql_alert: bool           # SPARQL 규칙 이상 여부
    ae_score: float              # AutoEncoder 재구성 오차
    gbm_risk: float              # GBM 위험 확률

    final_state: str             # "CRITICAL" | "WARNING" | "NORMAL"
    alert_sent: bool
```

### 2. SPARQL Client (`src/sparql_client.py`)

규칙 기반 이상 감지:

```python
# HR > 110 (빈맥) 또는 BR < 8 (서호흡) → Alert
FILTER(?hr > 110 || ?br < 8)
```

### 3. Anomaly Detector (`src/anomaly_detector.py`)

AutoEncoder 기반 이상 탐지:

- 정상 패턴으로 학습된 모델
- 재구성 오차(MSE)가 높으면 이상
- 임계값: 0.35

### 4. Risk Scorer (`src/risk_scorer.py`)

GBM 기반 위험도 예측:

- 특성: HR/BR의 mean, std, min, max, trend
- 출력: 위험 확률 (0~1)
- 임계값: 0.6

### 5. Decision Engine (`src/decision_engine.py`)

3가지 신호 융합:

```python
def decide(sparql_alert, ae_score, gbm_risk):
    if sparql_alert:
        return "CRITICAL"
    if ae_score > 0.35 and gbm_risk > 0.6:
        return "CRITICAL"
    if ae_score > 0.35 or gbm_risk > 0.6:
        return "WARNING"
    return "NORMAL"
```

---

## 🧠 의사결정 로직

### Decision Matrix

| SPARQL | AE Score | GBM Risk | Final State | Action |
|--------|----------|----------|-------------|--------|
| ✅ Alert | Any | Any | 🔴 CRITICAL | 즉시 알림 |
| ❌ | > 0.35 | > 0.6 | 🔴 CRITICAL | 즉시 알림 |
| ❌ | > 0.35 | ≤ 0.6 | 🟡 WARNING | 알림 |
| ❌ | ≤ 0.35 | > 0.6 | 🟡 WARNING | 알림 |
| ❌ | ≤ 0.35 | ≤ 0.6 | 🟢 NORMAL | 모니터링 지속 |

### 왜 3가지 신호를 융합하는가?

1. **SPARQL만 사용** → 복잡한 패턴 놓침
2. **AutoEncoder만 사용** → 센서 노이즈에 민감
3. **GBM만 사용** → 급성 변화 감지 지연

**융합의 장점:**
- 오경보(False Positive) 감소
- 놓침(False Negative) 감소
- 다각도 검증으로 신뢰도 향상

---

## 🎮 데모 실행

### 기본 실행

```bash
python main.py
```

### 출력 예시

```
╔═══════════════════════════════════════════════════════════════╗
║     LangGraph Ambient Agent for Elderly Care                  ║
║              KAIST 김재철AI대학원 CAIO 9기                    ║
╚═══════════════════════════════════════════════════════════════╝

Phase 2: 시나리오별 테스트 실행
═══════════════════════════════════════════════════════════════

  📊 Case 1: Normal Vital Signs (정상 상태)
  Patient ID    : P001
  Final State   : 🟢 NORMAL
  SPARQL Alert  : No
  AE Score      : 0.0821 (Normal)
  GBM Risk      : 0.1500

  📊 Case 3: Cardiac Arrest Pattern (심정지 패턴)
  Patient ID    : P003
  Final State   : 🔴 CRITICAL
  SPARQL Alert  : Yes ⚠️
  AE Score      : 0.4532 (Anomaly)
  GBM Risk      : 0.7800
  Alert Sent    : Yes 📤
```

---

## 📚 참고 자료

### LangGraph
- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- [Ambient Agent 101](https://github.com/langchain-ai/ambient-agent-101)
- [LangChain Academy - Ambient Agents](https://academy.langchain.com/courses/ambient-agents)

### 관련 연구
- Yang et al. (2017). Vital Sign and Sleep Monitoring Using Millimeter Wave
- An & Cho (2015). Variational Autoencoder based Anomaly Detection
- Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System

### 관련 프로젝트
- [kaist-caio9-autoencoder](https://github.com/EclipseFever/kaist-caio9-autoencoder) - AutoEncoder 기반 이상 탐지

---

## 📄 라이선스

MIT License

---

## 👥 저자

**KAIST 김재철AI대학원 CAIO 9기**

---

*이 프로젝트는 KAIST AI대학원 CAIO 과정의 일환으로 개발되었습니다.*
