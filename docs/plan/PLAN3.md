v1_0을 계승하는 v1_4 환경을 구성해라.
이 환경의 목적은 학습된 v1_0을 기반으로 리워드 및 페널티를 고정하기 위한 구성이다.

1. arm_free_speed 커리큘럼 : 
    - 기존과 동일한 감소 경향 및 목표를 가지지만, interval을 250으로 변경할 것. 최종 수치와 도달 시점은 같다.

2. eef_free_speed 커리큘럼 : 
    - 의도한 것 보다 EEF가 빠르다
    - 따라서, 감소 경향은 유지한 상태로 free speed를 0.2까지 감소시키는 것으로 변경
    - interval을 250으로 수정. 다만, 감소 경향은 유지해야 함

3. Dead Zone 커리큘럼 : 기존과 동일

4. object_free_acceleration : 
    - 초기에는 제한을 두지 않아 미는 것을 우선적으로 학습하게 하고, 후반부터는 순간적인 가속을 억지하는 방향의 학습을 유도한다. 
    - free accel을 100에서 시작하여 10000 iteration까지 유지한다. 
    - 10000 iteration부터 12000 iteration까지 50으로 유지한다. (100->50으로 계단식 조정)
    - 12000 iteration부터 18000 iteration 까지, 50에서 10으로 점진적으로 낮춘다. 

5. obstacles : 
    - 더 복잡한 환경을 만드는 것을 목표로 한다.
    - 시작 및 빈도는 같으나, 5 Gray, 5 White 스폰을 목표로 최대치를 늘릴 것
    - 최종적으로 11개를 스폰한다. (5, 5, 1)

6. pushing_free_speed : 
    - 의도보다 빠르게 밀고 있다
    - 시작 iteration 및 감소 주기는 같으나, 도달 지점을 0.02로 설정할 것
    - interval을 250으로 수정. 다만, 감소 경향은 유지해야 함

7. wrench free force/moment : 
    - 초기에는 제한을 두지 않아 미는 것을 우선적으로 학습하게 하고, 후반부터는 억지하도록 학습을 유도한다. 
    - Force는 5000, Moment는 500으로 10000 iteration까지 유지한다. 
    - 10000 iteration부터 12000 iteration까지는 Force 200, Moment 30으로 유지한다. (계단식 조정)
    - 이후, 12000 iteraion부터 18000 iteration까지 Force는 100, Moment는 20이 되도록 점진적으로 낮춘다.

8. goal_hold_tolerance
    - 5000 iteration 까지 0.0을 유지한다.
    - 5000 iteration 부터 0.01로 설정하고, 10000 iteration까지 0.005가 되도록 점진적으로 증가시킨다.
    - interval은 250 정도로 설정한다.

---

추가적으로, 아래 사항을 추가로 반영한다.

1. Success Termination을 추가한다.
    Success 조건에 들어온 상태로 3초가 지나면 Termination이 발생하도록 설정한다.

2. 물체 속도 Termination을 0.8 -> 1.0으로 변경할 것

3. goal_hold 의 Weight를 지금보다 10배 증가시켜라

4. Hand Contact의 Weight를 지금보다 5배 증가시켜라