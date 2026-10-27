# Workspace 작업 지침

활성 기반 패키지는 `src/inspire_sweep_baseline`이다. **Holding → Final 계열이 가장 중요한 최신 개발 대상**이며, 요청된 정확한 버전을 기준으로 작업한다. 이전 환경의 코드·자산은 현재 환경이 의존하는 경우에만 확인한다.

- 작업 시작: 
    - [현재 인계](example/Contact-Rich-Manipulation-Context/01_plan/00_HANDOFF_BRIEF.md)를 읽고 요청 대상의 실제 코드를 확인한다. 모든 문서를 연쇄적으로 읽지 않는다.
    - 환경 수정·추가: [개발 지침](example/Contact-Rich-Manipulation-Context/00_guidelines/09_DEVELOPMENT_GUIDE.md)과 [Holding·Final 환경 안내](example/Contact-Rich-Manipulation-Context/02_implementation/08_ACTIVE_ENVIRONMENTS.md)를 읽는다. 
        - 기존 버전 수정 요청은 그 환경에 직접 적용하고, 새 환경 생성 요청은 지정 부모를 상속하여 부모의 구성·동작을 보존한다.

- 구성 보존: `src/inspire_sweep_baseline/inspire_sweep_baseline`의 현재 디렉토리 구조를 유지한다. 요청한 함수·수치·일정만 변경한다. 파일 추가 또는 명시적인 구조 변경 작업에만 [파일 구조 지침](example/Contact-Rich-Manipulation-Context/00_guidelines/12_ENVIRONMENT_FILE_STRUCTURE.md)을 추가로 읽는다.

- 기록: 매 수정의 근거·의도·변경 전후·영향·최종 대조·검증 결과와 미검증 범위를 문서 저장소의 개발 로그에 남긴다. 후속 에이전트 인계용이며 사용자에게 읽기나 승인을 요구하지 않는다.

- 학습: `src/inspire_sweep_baseline/scripts/train.py`를 사용한다.

- 학습 로그: 명시적 요청 없이 정리하지 않으며 `logs/rsl_rl`은 건드리지 않는다. 정리 요청에만 [로그 정리 지침](example/Contact-Rich-Manipulation-Context/00_guidelines/11_TRAINING_LOG_CLEANUP.md)을 읽는다.

- 문서: `example/Contact-Rich-Manipulation-Context`에서 관리하며 내부 작업에는 그 저장소의 `AGENTS.md`를 따른다. 경로·저장소 작업에만 [Workspace 안내](example/Contact-Rich-Manipulation-Context/00_guidelines/10_WORKSPACE_GUIDE.md)를 읽는다.

서브모듈은 독립 저장소다. 현재 커밋과 작업트리의 사용자 변경을 보존한다. 과거 환경 설명은 기본 컨텍스트에 넣지 않으며, 명시적으로 과거 이력을 조사할 때만 [정리 전 백업](example/Contact-Rich-Manipulation-Context/02_implementation/history/2026-10-27_agent_context/README.md)을 읽는다.
