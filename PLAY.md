저장소 루트에서 실행하세요.

Ubuntu 학습:

```bash
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/train.py --task Isaac-Blind-Sweep-Inspire-Approach-v0 --headless --device cuda:0 --num_envs 2048
```

Ubuntu 재생:

```bash
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/play.py --task Isaac-Blind-Sweep-Inspire-Approach-v0 --device cuda:0 --num_envs 1 --checkpoint /absolute/path/to/model.pt
```

Windows 학습:

```powershell
.\IsaacLab\isaaclab.bat -p src\hand_manipulation_rl\scripts\train.py --task Isaac-Blind-Sweep-Inspire-Approach-v0 --headless --device cuda:0 --num_envs 2048
```

Windows 재생:

```powershell
.\IsaacLab\isaaclab.bat -p src\hand_manipulation_rl\scripts\play.py --task Isaac-Blind-Sweep-Inspire-Approach-v0 --device cuda:0 --num_envs 1 --checkpoint "C:\absolute\path\to\model.pt"
```