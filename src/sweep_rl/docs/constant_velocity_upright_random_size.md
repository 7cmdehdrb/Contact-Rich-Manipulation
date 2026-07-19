# ConstantVelocity UprightRandomSize

새 환경 ID:

```text
Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-v0
```

기존 `Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-v0`의 Action, Observation,
속도/목표점 reward는 그대로 상속하며 기존 환경의 설정은 변경하지 않는다.

## 변경점

### 완만한 그리퍼 orientation 보상

두 contact pad는 EEF local `+Y/-Y` 위치에 있다. 새 `gripper_upright` reward는
EEF local `+Y`가 world `+Z`를 향하도록 유도한다. 따라서 ㄷ자의 한쪽 변은 테이블
쪽, 반대쪽은 천장 쪽을 향한다.

```text
raw_reward = (dot(local_+Y_in_world, world_+Z) + 1) / 2
weight = 0.75
```

완전 upright는 raw reward 1, 수평은 0.5, 뒤집힌 자세는 0이다. hard constraint나
termination으로 사용하지 않으며, endpoint cost보다 훨씬 작은 보조 보상이다.

### Cube 크기 랜덤화

- 형상: 정육면체 유지
- 한 변 길이: environment마다 uniform `0.06~0.12 m`
- 기준 `0.06 m` cube에 isotropic scale `1.0~2.0` 적용
- 서로 다른 environment가 독립적인 물리 크기를 갖도록 `replicate_physics=False`
- reset 시 `z = table_top + side_length / 2`로 계산해 바닥면을 테이블에 맞춤

Isaac Lab/PhysX는 실행 중 rigid-body scale 변경을 안전하게 지원하지 않으므로,
크기는 simulation prestartup에서 각 parallel environment마다 한 번 샘플링된다.
예를 들어 2048 environments를 사용하면 한 rollout batch가 전체 크기 구간을 계속
포함한다.

물체 크기가 달라져도 접근 reward가 물체 내부나 지나치게 먼 지점을 가리키지 않도록
pre-contact stand-off도 다음처럼 크기에 따라 바뀐다.

```text
stand_off = side_length / 2 + 0.035 m
```

물체의 초기/현재 pose에 포함된 중심 높이가 크기에 따라 달라지므로 기존 55-D
Observation만으로도 정책은 물체 크기를 식별할 수 있다. Action 차원은 기존과 같은
12-D다.

## 학습

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/train_constant_velocity_upright_random_size.py \
  --device cuda:0 \
  --num_envs 2048 \
  --headless
```

크기별 물리 scene 복제가 비활성화되므로 기존 환경보다 simulation 초기화와 physics
step 비용이 증가할 수 있다.
