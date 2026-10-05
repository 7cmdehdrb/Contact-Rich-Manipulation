저장소 루트에서 실행하세요.

Ubuntu 학습:

```bash
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/train.py --task Isaac-Blind-Sweep-Inspire-Approach-v2 --headless --device cuda:0 --num_envs 2048
```

Ubuntu 재생:

```bash
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/play.py --task Isaac-Blind-Sweep-Inspire-Approach-v2 --device cuda:0 --num_envs 1 --checkpoint /home/min/7cmdehdrb/grad/logs/rsl_rl/UR5e_shelf_sweep_approach_v2/2026-09-28_02-45-16/model_3300.pt
```

Windows 학습:

```powershell
.\IsaacLab\isaaclab.bat -p src\hand_manipulation_rl\scripts\train.py --task Isaac-Blind-Sweep-Inspire-Approach-v2 --headless --device cuda:0 --num_envs 2048
```

Windows 재생:

```powershell
.\IsaacLab\isaaclab.bat -p src\hand_manipulation_rl\scripts\play.py --task Isaac-Blind-Sweep-Inspire-Approach-v2 --device cuda:0 --num_envs 1 --checkpoint "C:\absolute\path\to\model.pt"
```
