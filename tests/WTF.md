다음 사항을 확인 및 반영해라. 이미 반영된 사항이라면 넘기고, 아니라면 개정하라:

1. PPO 설정
example/Sweep-Policy/sweeping_policy/config/ur5e/agents/rsl_rl_ppo_cfg_02.py 를 기본으로 따라가도록 해라.
단, 지금 너가 만든 설정은 유지하고, 새 설정 파일을 만들어서, 저 설정을 사용해라. 다만, max_iter는 10000을 유지한다.

2. F/T 센서
별도의 캔슬레이션이 없는 것으로 취급해라. 즉, Hand 자체의 무게에 의한 힘과 모멘트는 상시 감지되는 구조로 설정된다.

3. 로봇 Configuration 및 shelf 위치.
example/Sweep-Policy/sweeping_policy 기본적으로 여기에 맞춰라.
당연히 가상 EEF Point나 핸드를 포함한 구성 같은건 변경되어야 하지만, 각종 파라미터 등은 여기에 맞춰라

4. OSC
example/Sweep-Policy/sweeping_policy/config/ur5e/osc_random_sweep_env_cfg.py
OSC 파라미터는 기본적으로 여기에 맞춰라. 

5. 환경 구성

밀어야 하는 방향이 잘못 되었다.

       불가능
가능 <- 선반 -> 가능
       불가능

이런 방향의 움직임이여야 한다.

example/Sweep-Policy 를 확인해서 방향을 잡아라.

또한, 선반임을 감안해서 w는 크고 d는 짧은 경향이 있어야 하는데, 지금은 정사각형에 가까워 보인다. 이도 개선해라.
