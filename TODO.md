
## 학습 및 실행 방법

```bash
./IsaacLab/isaaclab.sh -p -m pip install -e src/sweep_rl

./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Sweep-Object-UR5e-OSC-v0 \
  --headless --device cuda:0
```

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/play_sweep.py \
  --device cuda:0 \
  --task Isaac-Sweep-Object-UR5e-OSC-TactileLocalization-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/ur5e_osc_sweep_tactile_localization/2026-07-17_19-18-02/model_4700.pt
```

---

## 현재 진행 상황

- joint_pos
- joint_vel
- joint_effort  
- eef_pose 
- ft_sensor  
- contact_point  
- initial_target_pose  
- current_target_pose  
- desired_motion  
- last_action  

위 관측을 기반으로 정확하게 미는 정책을 학습함

---

## 다음 수정 사항 (진행 필요)

1. current_target_pose 를 사용하지 않은 정책 학습
2. 센서 데이터에 노이즈를 추가한 정책 학습

---

## 진행된 수정 사항

1. 미는 방위를 무제한으로 설정. 360도 전부 화전이 가능함
2. 미는 물체의 질량 범위를 상당히 넓은 범위로 랜덤화 시도할 것. 0.3kg~3.0kg

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/train_wide_randomization.py \
  --task Isaac-Sweep-Object-UR5e-OSC-WideRandomization-v0 \
  --device cuda:0 \
  --num_envs 2048 \
  --headless
```

## 진행된 수정 사항 2

current_target_pose 를 관측에서 제외
이를 통해, 미는 물체의 현재 위치는 관측되지 않고, 촉각 데이터 만으로 위치를 추정하는 구조를 구현
Reward에서는 그대로 사용.

```bash
./IsaacLab/isaaclab.sh -p  src/sweep_rl/scripts/train_tactile_localization.py  --task Isaac-Sweep-Object-UR5e-OSC-TactileLocalization-v0  --device cuda:0  --num_envs 2048  --headless
```


```bash
./IsaacLab/isaaclab.sh -p  src/sweep_rl/scripts/train_constant_velocity.py  --device cuda:0  --num_envs 2048  --headless
./IsaacLab/isaaclab.bat -p  src/sweep_rl/scripts/train_constant_velocity.py  --device cuda:0  --num_envs 2048  --headless
```
