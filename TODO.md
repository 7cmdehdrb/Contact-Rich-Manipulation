
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
  src/sweep_rl/scripts/train_wide_randomization.py \
  --task Isaac-Sweep-Object-UR5e-OSC-WideRandomization-v0 \
  --device cuda:0 \
  --num_envs 2048 \
  --headless
```

---

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/play_sweep.py \
  --device cuda:0 \
  --num_envs 1
```

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

## 다음 수정 사항 (진행 필요)

1. current_target_pose 를 사용하지 않은 정책 학습
2. 센서 데이터에 노이즈를 추가한 정책 학습
