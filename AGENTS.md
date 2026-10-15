# Workspace 작업 지침

현재 개발 대상은 `src/inspire_sweep_baseline`과 `src/axia80_feasibility`다.
`src/trash`는 과거 구현과 자산의 보관 위치이며 빌드 및 기본 테스트 탐색에서 제외한다.
활성 코드가 필요로 하는 보관 자산은 새 경로에서 읽기 전용으로 참조한다.

- 작업 시작: [프로젝트 인계](example/Contact-Rich-Manipulation-Context/01_plan/00_HANDOFF_BRIEF.md)와 [Workspace 안내](example/Contact-Rich-Manipulation-Context/00_guidelines/10_WORKSPACE_GUIDE.md)를 읽는다.
- 환경 추가·수정: [개발 지침](example/Contact-Rich-Manipulation-Context/00_guidelines/09_DEVELOPMENT_GUIDE.md)의 Manager 기반 OOP 구성과 로그 규칙을 따르고, [활성 환경](example/Contact-Rich-Manipulation-Context/02_implementation/08_ACTIVE_ENVIRONMENTS.md)의 상속·등록·실행 안내를 갱신한다.
- 연구 결정·개발 이력: 문서 전용 `example/Contact-Rich-Manipulation-Context`에서 관리한다. 내부 작업에는 그 저장소의 `AGENTS.md`도 적용한다.
- 참고 환경 설명: `example/Sweep-Policy/docs`에서 관리한다. 일반 SSH·VPN·ROS/Python 지식은 루트 `docs`에 간략히 기록한다.
- 검증: 현재 변경에 맞는 테스트를 실행하고 실제 실행한 명령·결과·미검증 범위를 개발 로그에 남긴다. 테스트와 실행 안내는 루트 `README.md`를 따른다.

서브모듈은 독립된 저장소다. 현재 커밋을 기준으로 읽고, 사용자 작업 중인 변경을 보존한다.

- Omni 베이스 및 모든 파생 환경(Dex Omni 포함)은 로컬 `mdp/goal_rewards.py`의 `action_rate_l2`·`joint_vel_l2` 페널티를 반드시 사용한다. Arm/Hand 네 항을 분리하고 raw Action 변화량·실제 관절 속도의 제곱합을 사용한다. 평균·정규화·Hand 입력 clamp를 보상 계산에 도입하거나 참고 환경에서 함수를 import하지 않는다. 계수는 Arm Action/Velocity −0.03/−0.03, Hand Action/Velocity −0.03/−0.01을 유지하며 사용자 명시 지시 없이 변경하지 않는다.
