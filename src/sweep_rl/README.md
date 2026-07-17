# Sweep RL

UR5e → virtual F/T sensor → Robotiq 2F-85 구성의 Manager-based
variable-stiffness OSC sweep 학습 패키지이다.

설치와 학습 방법은 [docs/osc_sweep_training.md](docs/osc_sweep_training.md)를
참고한다.

## Sweep 정책 재생 및 시각화

전용 재생 스크립트는 목표 물체 pose를 좌표 프레임으로 표시하고, 목표 힘을
녹색 화살표, 실제 접촉 힘을 빨간색 화살표로 표시한다. 화살표 길이는 힘의
크기에 비례하며 같은 값이 터미널에도 주기적으로 출력된다.

```bash
./IsaacLab/isaaclab.sh -p src/sweep_rl/scripts/play_sweep.py \
  --device cuda:0 --num_envs 1 \
  --checkpoint logs/rsl_rl/ur5e_osc_sweep/<run>/model_<iteration>.pt
```

`--checkpoint`를 생략하면 `logs/rsl_rl/ur5e_osc_sweep` 아래의 가장 최근
체크포인트를 자동으로 선택한다. GUI 시각화를 보려면 `--headless`를 사용하지
않는다.
