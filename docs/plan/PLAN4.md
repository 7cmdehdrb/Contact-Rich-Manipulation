Contact-Sweep-Clutter-Fusion-Holding-v0와 이 파생 환경들인 -OSC, -OSC-ObjectRandom 환경에 큰 개선점을 부여한다.
단, 기존 환경은 유지하고, 변경점이 반영된 -v1 환경을 각각 구현할 것.

1. Fusion-Holding-v1

기존에 장애물 정보 인코딩 하는 부분을 변경한다. 기존의 40D -> 16D 압축을, 총 16개의 장애물까지 감당 가능하도록 수정하여, 16 * 4 = 64 -> 32 -> 16D 방식의 압축을 사용하도록 변경한다.

2. -OSC-v0

Holding-v0을 상속 받는 구조를 유지하며, 이동 방향을 Robot Base 기준으로 해석하도록 변경하고(현재는 EEF), stiffness를 200 -> 100으로 변경한다.

3. -OSC-v1

-Holding-v1을 상속받도록 설정하며, OSC 설정은 -OSC-v0과 같다.

3. -OSC-ObjectRandom-v1

-OSC-v1을 상속받도록 설정한다.

---

ObjectRandom 환경의 관측을 점검해줘.

1. 기존에 사용했던 것 처럼, 정면에서 바라볼 때, 충분히 관측 가능하다고 판단되는 물체만 관측에 포함하는 로직은 유지되며, 물체가 원통형이 아닌 다른 물체도 추가된 상황애서도 정상적으로 동작하는가?

2. 왼쪽부터 관측한 물체들을 채워넣고, 관측 가능한 물체 사이에서 확률적 스왑이 일어나는 로직은 유지되고 있는가?

---

-OSC-v2 환경을 만들어줘. Fusion-Holding-v1 환경을 상속하며, OSC 컨트롤러 부분만 변경한다.

```python
self.actions.arm_action = UR10OperationalSpaceControllerActionCfg(
    asset_name="robot",
    joint_names=[".*"],
    body_name="ee_link",
    body_offset=UR10OperationalSpaceControllerActionCfg.OffsetCfg(),
    controller_cfg=OperationalSpaceControllerCfg(
        target_types=["pose_rel"],
        impedance_mode="fixed",
        motion_control_axes_task=(1, 1, 1, 1, 1, 1),
        contact_wrench_control_axes_task=(0, 0, 0, 0, 0, 0),
        inertial_dynamics_decoupling=False,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_stiffness_task=(80.0, 80.0, 80.0, 25.0, 25.0, 25.0),
        motion_damping_ratio_task=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        nullspace_control="none",
    ),
    position_scale=0.03,
    orientation_scale=0.15,
    effort_limit_scale=1.0,
)
```
이는 UR10을 사용한 Reach 예제에 있는 OSC 컨트롤러이다. 이것에 맞추어 OSC 컨트롤러를 설정해줘.
단, 제어 기준점은 우리의 EEF Frame으로 변경되어야 한다.

