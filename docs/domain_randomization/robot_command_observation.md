# 로봇·명령·관측 Randomization

물체 물성만 바꾸면 정책이 로봇 calibration이나 sensor 오차에 여전히 과적합될 수 있다.
이 프로젝트는 초기 관절, OSC calibration, task command와 관측값도 randomize한다.

## Arm 초기 상태

기본 환경과 Independent 환경 모두 episode reset에서 arm의 default joint pose에
`[-0.04, 0.04] rad` offset을 더하고 초기 속도는 0으로 둔다.

```python
reset_arm = EventTerm(
    func=mdp.reset_joints_by_offset,
    mode="reset",
    params={
        "position_range": (-0.04, 0.04),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": ARM_CFG,
    },
)
```

구현 연결은 기본
[`EventCfg`](../../src/sweep_rl/sweep_rl/osc_sweep/env_cfg.py)와 Independent
[`EventsCfg`](../../src/sweep_rl/sweep_rl/osc_sweep_independent/env_cfg.py)다.

## OSC calibration

Independent Action term은 episode마다 environment별 scalar를 샘플링한다.

| 항목 | 범위 | 적용 위치 |
|---|---:|---|
| stiffness calibration | `0.95–1.05` | policy stiffness 변환 후 |
| damping calibration | `0.95–1.05` | OSC damping gain |
| effort calibration | `0.97–1.03` | joint effort limit |

구현은
[`IndependentSweepOscAction.reset()`](../../src/sweep_rl/sweep_rl/osc_sweep_independent/mdp/actions.py)에
있다. Action 차원은 늘리지 않고 같은 policy output이 조금 다른 실제 torque를 만들게 한다.

## Task command

Command randomization은 학습해야 할 목표 자체를 분포로 만든다.

| 계열 | 방향 | 거리 | 추가 command |
|---|---:|---:|---|
| 기본 force-command | `[-π, π]` | `0.10–0.22 m` | 힘 `8–25 N`, tolerance `3–6 N` |
| Wide/Tactile | `[-π, π]` | `0.10–0.22 m` | 힘 `8–50 N`, tolerance `3–6 N` |
| ConstantVelocity | `[-π, π]` | `0.10–0.22 m` | 속도 `0.08 m/s` |
| Independent | feasible `[-π, π]` | `0.12–0.35 m` | 속도 `0.04–0.12 m/s` |

Independent command는 먼저 shelf workspace 경계를 넘지 않는 방향과 최대 거리를 계산한
뒤 feasible 범위 안에서 샘플링한다. 구현은
[`FeasibleSweepHomeCommand`](../../src/sweep_rl/sweep_rl/osc_sweep_independent/mdp/commands.py)다.

## Observation noise

기본 정책 관측의 주요 noise는 다음과 같다.

| Observation | 기본/ConstantVelocity | Independent |
|---|---:|---:|
| joint position | `±0.002 rad` | `±0.002 rad` |
| joint velocity | `±0.01 rad/s` | `±0.01 rad/s` |
| joint effort | 없음 | `±0.5` |
| F/T force | 없음 | 축별 `±0.5 N` |
| F/T torque | 없음 | 축별 `±0.02 Nm` |
| contact point | 없음 | 유효 접촉에만 `±0.002 m` |
| initial target position | 없음 | 축별 `±0.003 m` |
| initial target rotation | 없음 | 축별 `±0.02 rad` |
| object velocity | ConstantVelocity `±0.005 m/s` | 정책에 제공하지 않음 |

Independent의 component별/masked noise 구현은
[`osc_sweep_independent/mdp/observations.py`](../../src/sweep_rl/sweep_rl/osc_sweep_independent/mdp/observations.py),
연결은 `ObservationsCfg.PolicyCfg`에 있다.

## 범위 설계 순서

1. 먼저 randomization 없는 환경에서 task가 학습되는지 확인한다.
2. 실제 장비/asset 측정 오차에 근거한 좁은 범위부터 시작한다.
3. 한 번에 한 category를 넓혀 regression 원인을 분리한다.
4. 관측 noise는 reward/termination에 쓰는 privileged simulator state에는 적용하지 않는다.
5. command 범위가 scene workspace와 충돌하지 않게 feasible sampler를 사용한다.

[Domain Randomization 문서로 돌아가기](README.md)
