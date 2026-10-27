기존 -v1 환경을 상속받는, Final-GoalAccuacy-v0 환경을 만들어줘.
이 환경의 목적은, 기존의 -v1 환경이, 어느 정도는 목표 위치 까지는 밀어내지만, 내가 원하는 정확도 (1cm 미만) 을 만들지 못하는 것을 개선하는 것에 있다.

## 커리큘럼 조정

커리큘럼을 변경한다. 해당 값에 맞추어 리워드의 계수도 조정할 것

1. goal_speed_scale : 커리큘럼 삭제. 0.05 으로 고정
2. goal_position_tolerance : 커리큘럼 삭제. 0.01 (1cm) 으로 고정
3. object_free_speed : 커리큘럼 삭제. 0.1 으로 고정
4. obstacle_population : 빈도 조정. 2000부터 시작하되, 전체 500마다 1개 == 각 색 1000마다 1개 증가, 최종 8/8로 수정할 것. 최종 목표는 같고, 늘어나는 빈도가 짧아지는 구성.

## 리워드 조정

1. gold_hold : r < d < 2r 일 때의 리워드를 삭제하고, 오직 r < d 일 때만 보상을 부여하도록 조정할 것. 
2. object_speed_penalty 의 패널티 이름을, target_speed_penalty 로 변경할 것

## 기타 조정

학습 Metric을 남기도록 설정할 것. 이는 중요한 몇개의 지표를 텐서보드에 로그 형태로 남김으로서 개선 방향을 알기 쉽게 하기 위함이다.

1. Target - Goal Distance (XY Plane)
2. End-Effector Linear Speed (1D)
3. Target Object Linear Speed, Linear Accel
4. obstacle_actual_gray, obstacle_actual_white : 커리큘럼 쪽에 보였으나, 이동 시킬 것
5. Max Force/Moment (F/T 센서 관측 값)

---

위와 같은 수정을 할 때, "r" 에 따라서 영향 받는 다른 조건에 유의할 것.
Success 판정, Reaching 인정 범위 조건, 접촉 인정 범위 조건도 같이 0.01(1cm)으로 변경되어야 한다.

이러한 수정의 목적은, 원래는 인정 범위를 넓혔다 좁히면서 정확도를 높히려는 것이 목적이였으나, exploration이 감소하면서 실질적으로 정확도가 올라지 않는 것으로 판단, d < r 을 만족하는 것을 시작부터 상당히 엄격하게 제한하기 위함이다.
