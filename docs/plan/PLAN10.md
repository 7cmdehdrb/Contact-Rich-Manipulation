inspire_revision_test_v2 를 계승한 환경을 만들어줘. 
의미론적으로 계승인거지, 실제로는 상속 관계를 완전히 끊고, 독립된 환경으로 만든다. 함수를 재사용하는 것은 허용한다.

주요 변경점은 다음과 같다.

1. goal_reward_height_limit를 0.12로 변경
- 의도는, 기존 환경에서 shelf termination이 급증한 이유가 이 조건 때문으로 추정하기 때문이다. 리워드를 받으려면 낮은 높이를 유지해야 하는데, 이 과정에서 shelf 조건이 자꾸 발생하는 것으로 추정한다.

2. 성공 조건의 변경
- 기존에는 timeout 시기에 물체가 3cm 이내일 경우, 성공 취급하고 터미네이션 + 보상 지급 구조로 이루어져 있다.
- 해당 조건 때문에, 실질적으로 목표에 도달하고 이상 행동이 자꾸 유발된다고 생각하기 때문에, 물체가 3cm 이하 & 물체 속도 0.005m/s 이하 조건을 3초동안 연속적으로 만족한다면, success termination 발생 + 큰 보상 지급으로 변경한다. 보상 수준은 같다.

3. 물체 스폰 규칙의 변경
- RED Zone을 다소 확장한다. 지금 배치 구조로는 물체 주변에 너무 가깝게 배치되어 있는 경향이 있는 것으로 보인다. 이를 개선하기 위해 RED Zone을 확장, 타겟 물체와 너무 가까운 공간은 비우도록 유도한다.
- 환경 랜덤화 최적화를 수행핸다. 기존에는 매 reset 마다 매번 환경을 새로 구축하게 설정되어 있다. 이것에 최적화를 부여한다. 최초 initialize에 리셋되는 사항과, reset() 하다 리셋되는 사항을 구분해라. 가장 병목이 심하다고 판단되는 부분을 initialize 쪽으로 빼서, collection time을 1~2초 수준으로 줄이는 최적화를 수행해라.

4. Dead Zone 커리큘럼의 변경
- 10000 iter부터 증가하기 시작하여, 3000 iteration 마다 -5도씩 수행하는 규칙으로 변경한다. 이는 정책이 제대로 학습이 되기 전부터 데드존이 감소해서 차라리 실패하는 경향이 발생하는 것 같아, 이를 방지하기 위함이다.

---

Inspire-Wrench-v0 를 상속받는 -v1 환경을 만들어줘.
이 환경은 OSC에만 변경점을 가한다.

기존에는 다음과 같은 조건이 메인이였다.

```python
OperationalSpaceControllerCfg(
            target_types=["pose_rel"],
            impedance_mode="fixed",
            motion_control_axes_task=(1, 1, 1, 1, 1, 1),
            contact_wrench_control_axes_task=(0, 0, 0, 0, 0, 0),
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=True,
            motion_stiffness_task=(100.0, 100.0, 100.0, 50.0, 50.0, 50.0),
            motion_damping_ratio_task=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            nullspace_control="none",
        ),
```

motion_control_axes_task 만 사용하는데, 이를 개편하여, contact_wrench_control_axes_task 만 사용하고, target_types도 wrench로 변경한다.
closed-loop wrench control을 사용한다.
그냥 수정하는게 아니라, "IsaacLab/source/isaaclab_tasks" 안에 있는 예제라거나, 검색으로 취득한 Wrench-based OSC 를 참고하여 코드를 작성해줘. 어떤 설정 파라미터를 썼는지 등이 주요 확인점이다.

또한, 추가적으로 다음과 같은 변경사항을 반영해라.
Wrench Control에서, closed loop를 사용할 경우, current wrench를 알아야 하기 때문에, 현실 deploy를 고려하여, **OSC 제어 기준점을 F/T 센서 프레임으로 설정**한다.

IsaacLab에서는 별로 지장 없지만, 이 변경을 통해, 현실에서 F/T 센서 값을 받아서, OSC Controller에 넣을 수 있게 처리할 것이다. 이는, 기존과 같이 EEF 프레임이 기준점일 경우, 해당 위치에서의 Current Wrench를 파악하는 것이 곤란하기 때문이다.

(시뮬레이션 단에서 수정할 사항은 없을 것으로 여겨진다.) 

---

방금 작성한 -v0, -v1 환경을 각각 상속받는 -Test 환경을 추가 구현한다.
구현 내용은 매우 간단하다. 타겟 물체는 항상 실린더로 고정한다.

