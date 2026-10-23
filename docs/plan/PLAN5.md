"src\inspire_sweep_baseline"

이 환경의 정리를 원한다. 추가적인 환경 구현을 요청하는 것이 아니다.
현 폴더 아래는 여러 환경 폴더가 존재하고, "src\inspire_sweep_baseline\inspire_sweep_baseline" 아래에도 환경을 의미하는 파이썬 파일들이 존재한다. 즉, 구조가 통일되지 않은 상태이다.

이를 통합된 구조로 변경한다. 

이러한 환경 구축 방식은 별도로 문서를 작성하여 앞으로 추가적으로 구현하는 환경들도 같은 방식으로 파일 시스템을 작성하도록 하며, 현재의 최신 깃 커밋을 기록하여, 문제가 생겼을 때, 복구가 가능하도록 한다.

파일 시스템 구성은 아래와 같다.

inspire_sweep_baseline
    ㄴ inspire_sweep_baseline
        ㄴ agent
            ㄴ "agent" 폴더 안에는 PPO 관련 파일들만 존재합니다.
        ㄴ assets
            ㄴ "assets" 폴더 안에는 환경을 구성하기 위한 파일들만 존재합니다.
        ㄴ mdp
            ㄴ actions
                ㄴ "mdp/actions" 폴더 안에는 ActionCFG를 기록한 파이썬 파일들만 존재합니다.
            ㄴ "action_fuctions"
                ㄴ "mdp/action_functions" 폴더 안에는 ActionCFG에서 사용할 함수를 기록한 파이썬 파일들만 존재합니다.
            ㄴ rewards
            ㄴ reward_functions
            ... mdp 아래는 이런 식으로 Observation, Event, Action, Reward, Cirriculum, Termination 등을 담당하는 CFG 파일을 담당하는 폴더와, 그들이 사용할 함수를 담당하는 폴더로 나누어서 구성합니다. 모든 환경에 대하여 분리하지 말고, v1, v2, ... 이런 식으로 공통 인자를 가진 환경들은 같은 파일 안에 작성하도록 합니다.
        ㄴ envs
            ㄴ contact_sweep_clutter_fusion_holding_envs.py
            ... envs 폴더 아래에는 mdp 요소들을 모아 학습을 구동할 수 있는 EnvCfg를 담당하는 파이썬 파일들만 존재합니다. v1, v2, ... 이런 식으로 공통 인자를 가진 환경들은 같은 파일 안에 작성하도록 합니다.
        ㄴ __init__.py 


이 작업을 수행할 때, 가장 중요한 것은 임의의 수정을 가하면서 기존에 동작하던 환경이 동작하지 않게 되는 것 입니다. 파일 구조를 개선하다가 기존의 학습이 동작하지 않게 되는 현상을 최악으로 상정합니다.

특히, 가장 최신 환경인 "Contact Sweep Clutter Fusion", "Contact Sweep Clutter Fusion Holding" 계열 환경에는 특히 주의합니다.

작업 도중에 이해가 가지 않는 부분이나, 저 규칙에 해당하지 않는 파일이 존재한다면, 나에게 질문하고 작업하십시오.