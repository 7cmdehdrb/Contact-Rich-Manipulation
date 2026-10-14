화면에서 확인하도록 `--num_envs 1`로 설정했습니다. 각 `--checkpoint ""` 안에 체크포인트 경로를 입력하세요.

**Baseline**

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Sweep-Baseline-v0 \
  --num_envs 1 \
  --sweep-direction both \
  --checkpoint ""
```

**Realistic Physics**

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Sweep-Realistic-Physics-v0 \
  --num_envs 1 \
  --sweep-direction both \
  --checkpoint ""
```

**Realistic Physics Initial Privileged**

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Sweep-Realistic-Physics-Initial-Privileged-v0 \
  --num_envs 1 \
  --sweep-direction both \
  --checkpoint ""
```