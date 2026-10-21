"src/inspire_sweep_baseline/inspire_sweep_baseline/contact_sweep_clutter_v1_curriculum" 이 환경과 "src/inspire_sweep_baseline/inspire_sweep_baseline/contact_sweep_clutter_v2" 이 환경을 융합한 새로운 환경을 만들어줘.

기존 환경들과 가지는 상속 관계를 끊고, 새로운 CFG들을 형성한다. 함수는 그대로 끌고 와서 써도 된다.

다음과 같은 조건을 가진다:

0. 환경 이름은 적당히 설정한다. (중요하지 않음)

1. 기본적인 센서 구성 방식은 contact_sweep_clutter_v2를 따른다. 즉, 캔슬레이션 된 F/T 센서 관측을 사용하는 환경이다.
   
2. Observation을 변경한다.
    - 다른 물체의 관측을 변경한다. 관측 방식은 변하지 않지만, 정면(로봇이 존재하는 위치)에서 선반을 바라볼 때, 다른 물체에 가려져서 잘 보이지 않는 물체는 관측 대상에서 제외한다. ("src/scene_randomization"를 참고) 단, 관측 대상 물체를 선정하는 로직은 매우 빨라야하며, 1%(0.01)의 확률로 관측 여부를 flip한다. 즉, 1% 확률로 관측 안되는 물체인데 관측된 것으로 취급하고, 반대로 관측 안되야 하는 물체인데 관측되는 랜덤화를 수행한다. 
    - Target Object Width 관측을 제거한다.
    - Goal Pos를 3차원 관측에서 2차원 관측으로 변경한다. 기준 Frame을 변경하지는 않고, "얼마나 밀어야 하는가"를 의미하는 distance, "어느 방향으로 밀어야 하는가"를 의미하는 theta로 구분하여 2차원 입력욿 변경한다. 수직으로 밀어내는 것을 0.0, 끌어당기는 것을 3.14로 하여 시계 방향으로 rad 단위로 부여한다. 따라서, 가능한 수치는 0.0~2*3.14이다. 주의할 점은, 관측 방식만 바뀌는 것이지, 실질적으로 목표점이 변경되지는 않는다는 것이다.

3. Reward를 통합한다.
   - arm_action_rate, hand_action_rate 이 2개의 action rate를 합쳐, 하나의 action rate 리워드로 계산해라. 실질적인 리워드는 변경하지 않는다.
   - arm_joint_velocity, hand_joint_velocity 이 2개의 joint velocity를 합쳐, 하나의 joint velocity 리워드로 계산해라. 실질적인 리워드는 변경하지 않는다.
   - object_tilt, obstacle_displacement 이 2개의 object를 합쳐, 하나의 object 리워드로 계산해라. 실질적인 리워드는 변경하지 않는다.

4. Reward를 변경한다.
   - Contact 보상을 강화한다. 단순 Weight 변화로, Weight를 10배 증가시켜라.
   - 통합된 object 보상을 강화한다. 단순 Weight 변화로, 기존의 15배로 증가시켜라. 이는, 기존의 tilt와 displacement를 각각 15배 증가 시키는 것과 같은 효과를 만든다.

5. Reward를 추가한다.
   - EEF Linear Speed에 대한 페널티 항을 추가한다. 수식 구성 방식은 arm_joint_velocity와 같이 특정 임계점부터 exp하게 증가하는 방식이며, free speed를 1.5m/s으로 설정한다.
   - EEF의 Orientation 페널티 항을 추가한다. 지금 v1-cur 환경을 보면, 손바닥을 아래로 향하고 접촉하는 형태가 목격된다. 이는 내가 원하는 방향이 아님으로, 손바닥의 Normal이 바닥 방향에 가까울 수록 주는 페널티를 부여한다. 이는 EEF Frame의 z축이 지금은 하늘 방향을 향하는 것에 페널티를 부여하는 것이다. EEF Frame의 Z축이 가르키는 방향과 하늘 방향 벡터 사이의 코사인 각도가 특정 값부터 페널티가 시작, 0.0(일치)일 때 최대가 되는 방향의 페널티를 구현한다. 페널티는 전체 보상의 abs(1%) 수준으로 설정하고, 발산하지 않도록 유의해라. 주의-반대 방향을 가르킬 경우는 페널티 대상이 아니다. 즉, 손바닥이 하늘을 향하는 방향은 문제 없다.
   - 급격한 F/T 및 Wrench 변화에 대한 페널티를 추가해라. "outputs/clutter_ft/2026-10-21_curriculum_model12800_dataset_rates" 를 확인하여, 지나치게 큰 F/T 변화량에 대한 페널티를 부여하라. 주의할 점은, 페널티의 최대 한도가 있어야 하며(발산을 막기 위함), 지나치게 큰 경우만 막아야 한다는 점이다. 처음에 접촉이 발생할 때 정도의 페널티는 용인하는 느낌이여야 한다. 그래서 목표는, 지수함수 적으로 증가하는데, 적절한 계수를 설정하여 용인되는 변화량과 용인되지 않는 페널티를 간접적으로 구분하는 것이다.

6. Curriculum을 추가 및 변경한다.
   - pushing speed, arm speed, action rate(통합)의 커리큘럼을 변경한다.
   - 변경 주기를 500 iteration으로 통일한다.
   - pushing_speed의 free speed 경우 0.1 시작, 0.01씩 감소, 0.05 도달로 설정한다.
   - arm_speed_excess의 free speed 경우, 1.0 시작, 0.02씩 감소, 0.8 도달로 설정한다.
   - action rate weight의 경우, -0.03 시작, 0.005씩 감소, -0.06 도달로 설정한다.
   - EEF Linear Speed의 free speed 경우, 1.5 시작, 0.02씩 감소하여 0.7 도달로 설정한다.
   - Goal 관련 커리큘럼을 추가한다. 현재는 12시, 6시 방향에 고정된 DeadZone이 존재하는데, 이 DeadZone을 변경하는 커리큘럼을 추가한다. 시작에는 12시, 6시 각각 +-60도 범위로 goal을 설정하지 않지만, 1000 iteration 이후부터, 200 iteration마다 +-1도씩 감소시켜, 최종적으로 DeadZone이 삭제되는 구성을 따른다. 즉, 처음에는 좌, 우로만 밀 수 있다가 시간이 지나면서 앞 뒤까지 미는 경향성을 학습하게 만든다.
  

---

방금 구현한 환경을 설명하는 문서를 작성해줘. 다른 환경과 비교는 수행하지 않고, 독립적인 설명을 수행한다.

1. Scene 구성 방법에 대한 설명
   - Scene의 구성 요소에 대한 설명. 코드 레벨의 구현을 설명하는 것이 아닌, 어떤 의도로 구현되었는지를 설명
   - Robot
   - Hand
   - Sensor
   - Shelf
   - Object
  
2. Observation
   - 각 Observation의 의미와 차원
   - Actor/Critic이 다른 관측을 사용할 경우, 반드시 이를 설명
   - 관측 노이즈가 포함될 경우 이를 반드시 설명
   - Frame이 존재할 경우(특정 Frame에서의 관측일 경우) 이에 대한 설명을 반드시 포함

3. Action
   - Action 정의 방법
   - 최대로 나올 수 있는 Action의 크기

4. Reward
   - 각 Term의 의미와 의도
   - 의도한 각 Term이 가지는 비중 (예상 보상 수치)
   - 구성 상, 중요도가 높은 순서로 설명한다.

5. Termination
   - 각 Term의 의미와 의도
   - 매칭되는 Reward 항과 연계하여 설명한다.

6. Curriculum
   - 각 커리큘럼의 의미와 의도