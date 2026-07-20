# Sweep RL 상태와 후속 작업

이 파일은 2026-07-19 현재 구현 상태만 요약한다. 환경별 정확한 계약과 실행 명령은
[`src/sweep_rl/docs/registered_osc_sweep_environments.md`](src/sweep_rl/docs/registered_osc_sweep_environments.md)를 따른다.

## 구현 완료

- 12-D variable-stiffness OSC: stiffness 6-D + relative pose 6-D
- 5-D force-command 기본, Play, WideRandomization, TactileLocalization 환경
- 4-D speed-command ConstantVelocity 환경
- 외측 pad 접근과 gripper 내부 삽입 실패 환경
  (`...UprightRandomSize-v0`, 이름은 호환성 때문에 유지)
- 목표 정지 후 SWEEP에서 HOME으로 전환하는 HomeReturn 환경
- Home 복귀 중 전체 robot-target 접촉 페널티
- HOME 진입 위치 기준 물체 변위 보상과 강제 실패 조건
- 환경별 전용 RSL-RL PPO 설정과 학습 스크립트

## 현재 검증 포인트

- ConstantVelocity: `endpoint_error`, `progress_ratio`, success 비율
- Gripper Exclusion: `object_inside_gripper` 종료 비율
- HomeReturn: `home_phase`, `parked_displacement`, `post_goal_object_moved`, success 비율
- 공통 안전성: `excessive_wrench`, `arm_speed`, torque saturation 종료/보상

## 후속 작업

1. Isaac Sim이 설치된 환경에서 HomeReturn 4개 환경·1 iteration smoke test
2. HomeReturn 학습에서 물체 재접촉 및 `parked_displacement >= 0.015 m` 실패율 확인
3. 성공률과 안전 종료율을 기준으로 Home reward weight 및 12초 timeout 조정
4. 필요 시 55-D 기존 정책의 첫 layer를 56-D phase observation에 맞게 변환하는 도구 추가

## HomeReturn 학습

Linux:

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/train_constant_velocity_upright_random_size_home.py \
  --num_envs 2048 --device cuda:0 --headless
```

Windows PowerShell:

```powershell
.\IsaacLab\isaaclab.bat -p `
  src\sweep_rl\scripts\train_constant_velocity_upright_random_size_home.py `
  --num_envs 2048 --device cuda:0 --headless
```
