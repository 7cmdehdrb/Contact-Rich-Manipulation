# Sweep Shelve Force

`example/Sweep-Policy`의 random shelf-sweep task를 현재 Isaac Lab API에 맞춰
독립적으로 구성한 패키지다. 기존 task config나 MDP 모듈을 상속/import하지
않으며 환경 ID는 `Isaac-Sweep-Shelve-Force-v0` 하나만 등록한다.

주요 차이는 UR5e tool과 Robotiq 사이에 `VirtualFTSensor` rigid body를 삽입하고,
두 fixed articulation joint의 incoming wrench를 robot base frame 6-D F/T
observation으로 제공한다는 점이다. Policy는 `sweep_jh`와 동일한 41-D
Cartesian observation과 12-D variable-stiffness OSC action을 사용한다.

Reward manager에는 접근, force 제어, sweep 진행/정지, scene 안전, action 안정성,
실패 종료의 6개 term만 등록한다. Base-frame F/T에서 sweep 방향의 평면 force를
추출해 command의 desired force와 tolerance를 추종하며, 세부 물리량은 reward를
추가로 쪼개지 않고 TensorBoard diagnostic metric으로 기록한다.

## USD 경로

Shelf와 6개 object는 Omniverse/Nucleus USD를 직접 사용한다.

모든 USD 경로의 source of truth는 다음 파일 하나다.

```text
sweep_shelve_force/shelf_force/asset_manifest.py
```

이 파일의 `UR5E_USD_PATH`, `ROBOTIQ_USD_PATH`, `SHELF_USD_PATH`,
`OBJECT_USD_PATHS`에서 모든 asset 경로를 관리한다.

## 설치와 학습

```bash
./IsaacLab/isaaclab.sh -p -m pip install -e src/sweep_shelve_force

./IsaacLab/isaaclab.sh -p src/sweep_shelve_force/scripts/train.py \
  --num_envs 4096 --device cuda:0 --headless
```

Smoke test:

```bash
./IsaacLab/isaaclab.sh -p src/sweep_shelve_force/scripts/train.py \
  --num_envs 4 --max_iterations 2 --device cuda:0 --headless
```

TensorBoard:

```bash
tensorboard --logdir logs/rsl_rl/sweep_shelve_force
```

## 재생

```bash
./IsaacLab/isaaclab.sh -p src/sweep_shelve_force/scripts/play.py \
  --checkpoint /absolute/path/to/model.pt --num_envs 1 --device cuda:0
```
