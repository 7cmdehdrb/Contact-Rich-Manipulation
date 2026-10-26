FinalEnvCfg 를 상속받는 v1 환경을 구성해라. 주요 변경점은 아래와 같다.
주의! 기존 함수를 수정하다 예전 환경이 동작하지 않는 상황은 절대 있어서는 안된다. 가급적 함수를 새로 작성하는 방향으로 진행해라.

## 0. 공통 변경

리워드, 터미네이션, 커리큘럼 등의 이름을 직관적으로 변경하며, 같은 것끼리는 공통 접두사를 붙인다.
예를 들면, 장애물과 관련되면 obstacle ~~~ 이런 식으로 작명하는 것이다.
이유는 직관적 이해와, 텐서보드 상 정렬 때문이다.

## 1. Observation 변경

- 관측에 현재 Hand Configuration을 추가한다. 이는 Hand Action과 마찬가지로 [0, 1] 사이에서 정규화된 2차원 관측이다.

## 2. Reward 

- 단순 식 병합
    - arm_action_rate, hand_action_rate 를 병합하여, action_rate_penalty 라는 하나의 페널티로 변경한다. 보상식 및 Weight는 변경이 없는 구성이다.
    - arm_joint_velocity, hand_joint_velocity 를 병합하여, joint_velocity_penalty 라는 하나의 페널티로 변경한다. 보상식 및 Weight는 변경이 없는 구성이다. 
    - object, object_acceleration 를 병합하여, object_penalty 라는 하나의 페널티로 변경한다. 보상식 및 Weight는 변경이 없는 구성이다.
    - shelf_failure, shelf_collision 를 병합하여, shelf_penalty 라는 하나의 페널티로 변경한다. 보상식 및 Weight는 변경이 없는 구성이다.

- 페널티 삭제
    - arm_speed_excess 삭제
    - eef_linear_speed 삭제


## 3. Termination

- (단순 식 병합) shelf_collision, shelf_displacement 를 병합하여, shelf_termination 이라는 하나의 터미네이션으로 변경한다. 조건 자체는 변경이 없는 구성이다. 
- (이름만 변경) hand_velocity 터미네이션의 이름을 joint_velocity 로 변경한다. 
- time_out, success 관련으로, 성공률을 확인하기 위해 success termination을 만들고 싶다. 다만, 이걸 조기 종료로 걸면 리워드와 충돌해서 정확하게 밀지 않는 현상이 있다. 따라서, 10초가 초과할 때, 물체가 goal hold의 "r" 조건 (커리큘럼의 영향을 받는다) 이내면 success termination으로 취급하고, 아니라면 time_out termination으로 취급. 실질적인 Termination 조건은 변함이 없다.

## 4. Event

- (물체 크기 랜덤화) 물체의 크기 관련 이벤트를 추가해줘. 지금은 항상 고정된 물체 사이즈인데, 기존 사이즈 대비 +-10% 수준으로 하여 물체 크기를 랜덤화 해줘. 타겟과 장애물 둘 다에 해당하며, 사이즈를 키움으로 인하여 스폰 시 바닥과 겹치는 등의 이슈가 없도록 주의할 것
- (물체 종류 확장) 물체의 종류를 조금 추가해줘. 머그컵 같이, 일반적인 큐브나 실린더와 다른 형상의 물체를 2종 정도 추가했으면 좋겠어. 선반에 있을 물체여야 한다.
- (타겟 물체의 랜덤화) 타겟 물체도 다양화를 수행한다. 실린더, 큐브, 그리고 랜덤 물체로 변경한다. 랜덤 물체는 현재 있는 장애물 랜덤 물체와 추가된 랜덤 물체까지 포함한다. 
    - 추가적으로, 무엇이 타겟 물체인지 알기 위해 Physics에 영향을 주지 않는 3D Bounding Box를 추가해줘. 순수한 시각화 기능이며, 학습 과정에서는 부하를 줄이기 위해 시각화 하지 않고, play 시에만 시각화 한다.
- (물체 스폰 규칙 고도화) 물체 스폰 규칙과 관련하여, 예외 처리를 추가해줘. 
    - 이론 상 스폰할 수 있는 최대 수를 10 Gray, 10 White로 설정한다. 
    - 목표한 수치대로 스폰을 시도하되, 내가 알기로 8, 8개부터 Spawn에 실패했는데, 실패할 경우 White, Gray를 교대로 한개씩 줄여가며 스폰을 시도, 성공할 때 까지 시도하는 구성으로 잡는다. 
    - 이는 환경 초기화에 실패해서 학습이 터지는 상황을 방지하기 위함이다.

## 5. Curriculum

- 삭제된 Reward와 연계된 커리큘럼 삭제
- 주기 변경
    - obstacle_population_schedule 조정. 2000부터 시작하여, 1000마다 교대로 1개씩 증가, 최종 목표 갯수는 8, 8 
    - gray_zone_padding : 4000부터 시작하여, 10000에서 종료되도록 설정. 최종 수치는 동일, interval을 1000으로 변경.
    - wrench_free_force_change, wrench_free_moment_change의 주기를 변경. 3000부터 감소하여, 250 iteration 주기로 지속적으로  감소하여 10000에서 최종에 도달하도록 변경. 최종 수치는 동일
    - object_acceleration의 주기를 변경. free_acceleration, acceleration_weight 양 쪽에 해당. 3000부터 감소하여, 지속적으로 선형적으로 감소해서, 10000에서 최종에 도달하도록 변경. 최종 수치는 동일
    - goal_position_tolerance : 4000부터 감소 시작하여, 10000에서 0.01도달, 이후 고정으로 변경. 해당 커리큘럼은 다른 리워드에서도 사용하는 수치임에 유의할 것. 


---

FinalEnvCfg v1을 상속받는 v2 환경을 구성해라. 주요 변경점은 아래와 같다.

## Event

현재는 타겟 물체의 위치가 정 중앙에서 랜덤화인데, 너비 80%, 높이 80% 구간 내에서 (즉, 최외곽 영역만 제외) 랜덤 스폰으로 변경한다. 주의할 점은, 해당 환경에서 가능한 목표를 제시해야 한다는 점이다. 

환경 랜덤화 코드에 유의할 것.

---

**v1, v2, 그리고 기존의 모든 환경이 전부 정상 동작해야 한다.**