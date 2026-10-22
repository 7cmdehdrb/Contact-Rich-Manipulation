"src/inspire_sweep_baseline/inspire_sweep_baseline/contact_sweep_clutter_fusion_v3" 이 환경을 계승(상속)한 "_v1_0" 환경을 만들어줘. 이후는 v1_1 이렇게 상속 받을 것이다.

다음과 같은 변경 사항을 가진다.

## 0. 용어의 정리

- "Net Force", "Net Moment"의 경우, SQRT(F_x **2 + F_y ** 2 + F_z ** 2) 과 같은 방식으로 계산된다.

## 1. Dead Zone 커리큘럼의 변경

기존의 Dead Zone 커리큘럼의 의도는 점점 수직 방향으로도 미는 경향을 학습하게 만드는 것이였으나, 생각보다 이 의도대로 동작하지 못하는 현상이 발생했다.

따라서, 아래와 같이 변경한다.
- start degress를 30도로 설정
- holding iteration을 3000으로 설정
- interval을 500으로 설정
- 감소 degree를 5도로 설정

이 변경의 의도는 시작부터 대각선 방향을 어느 정도 학습하는 것이다.

## 2. Goal 유지 보상의 커리큘럼 추가

기존의 Goal 유지 보상은 1cm 이내에 유지시킬 때, 보상을 부여하는 것이다.
이 보상은 의도대로 동작하나, 정책이 목표를 인식하는 것을 쉽게 하기 위해 position tolerance에 커리큘럼을 부여한다.

- 시작 tolerance를 0.05로 설정
- 1000 iteration부터 tolerance를 감소
- interval을 500으로 설정
- 감소 tolerance를 0.01로 설정
- 0.01까지 감소

## 3. Tactile 보상의 변경

기존의 Tactile 보상의 의도는, Tactile 센서가 ON되는 갯수가 많아지고, 접촉을 유지할 수록 보상을 부여하는 것이다. 하지만, 왼쪽으로 미는 케이스에서 손가락 때문에 wrist3를 돌리지 못해 실질적으로 의미가 없는 보상이 되었다. 따라서, Tactile 센서는 단순한 관측으로만 사용하고, 접촉 유지 보상은 다른 방식으로 부여한다.

- 손 전체를 Contact Sensor로 취급한다. (기존 Tactile 역할은 유지되어야 한다) 
- 손에 느껴진 Netforce가 0.01 이상이 유지되면 보상을 부여한다. (기존 Tactile ON 보상과 같은 로직)
- 5초 동안 풀로 유지될 경우, 전체 보상의 10% 수준을 유지하도록 리워드 및 Weight를 부여하라. "logs/rsl_rl/inspire_contact_sweep_clutter_fusion_v3/2026-10-21_21-55-18" 의 학습 결과 참조한다. 
- 기존의 Tactile Weight 커리큘럼과 Tacile 갯수 보상은 삭제한다.

## 4. 기대 밖의 접촉에 페널티를 추가한다.

3은, 어떤 물체든 접촉이 발생하면 주어지는 보상이지만, 이 페널티는 "타겟 물체 밖의 접촉"에 부여하는 페널티이다. 어떤 물체와 접촉했는지를 확인해서, 접촉이 타겟 이외의 장애물에 의해 발생한다면, 해당 Net Force에 exp하게 증가하는 페널티를 부여한다.

- exponential_joint_speed_cost_limit 와 같은 방식으로 부여
- free speed에 해당하는 수치는 20N 으로 설정

## 5. wrench_change 보상 구조 및 수치 변경

아래와 같은 변경을 수행한다.

- 리워드 부여에 사용하는 힘의 기준점을 관측과 같은 EEF 기준으로 변경할 것
- 페널티의 기준을 Net Force와 Net Moment를 계산한 다음, 이것의 abs(delta)를 사용할 것. 즉, 이전 스탭과 현재 스탭에서 느껴지는 Net Force/Moment의 변화량의 절댓값이다.
- exp하게 증가하는 페널티 구조를 유지하며, Net Force의 경우 50N, Moment의 경우 5.0Nm부터 증가하는 구조를 가진다
- 커리큘럼도 변경한다. 
   - Net Force의 경우 1000 iteration부터, 250 iteration마다 1N 씩 감소시켜 10N 까지 감소시킨다. 
   - Net Moment의 경우 1000 iteration부터, 250 iteration마다 0.1Nm 씩 감소시켜 1.0Nm 까지 감소시킨다.

## 6. Object 속도의 Acceleration에 페널티를 부여한다. 

- 각각의 물체에 독립적으로 부여되는 페널티이고, 최종 페널티는 해당 페널티들의 합산으로 정해진다.
- exponential_joint_speed_cost_limit 과 같이 특정 지점 이후부터 exp 하게 증가하는 방식으로 부여한다.
- 0.3 m/s^2 이상부터 페널티를 부여한다.
- 커리큘럼을 추가하여 1000 iteration 부터, 250 iteration마다 0.01 N/m^2 씩 감소시켜, 최종적으로 0.05 m/s^2 이상부터 페널티를 부여한다.

## 7. 기타

- 변경되는 리워드, 터미네이션, 커리큘럼은 새로운 CFG 클래스를 짜서 집어 넣는 방식을 사용할 것 (기존의 리워드, 터미네이션, 커리큘럼을 상속받지 않는다.)

---

추가 수정 사항 (Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_0 환경의 변경이다. 새 환경 구현이 아니다.)

## 1. Acceleration 페널티:

Acceleration의 절댓값에 비례하는 페널티를 쓴다. 즉, 급격한 감속도 페널티 대상에 포함된다.

## 2. 장애물 수를 쉽게 조정할 수 있도록 스폰 함수 조정 및 커리큘럼 추가:

- 시작 장애물은 Gray Zone 1, White Zone 1 로 시작하게 설정한다.
3000 iteration부터, 3000 iteration마다 White Zone Spawn 갯수를 1개씩 늘려, 최종적으로 3개까지 스폰하게 한다.
- 4500 iteration부터, 3000 iteration마다 Gray Zone Spawn 갯수를 1개씩 늘려, 최종적으로 3개까지 스폰하게 한다.
- 최종적으로, 3 White, 3 Gray, 1 Target이 되게 한다.

## 3. Contact 유지 보상에 수정을 가한다.

- alpha 라는 추가 계수를 Contact 유지 보상에 곱한다. 
- 이 추가 계수는 Goal 도달로 인정하는 거리에서는 0.0, Goal 도달 인정 거리 +3cm에서는 1.0이라는 수치를 가지며, 그 사이에서는 선형적으로 감소한다.
- 즉, 목표와 가까워지면 Contact를 유지할 목적을 점점 줄여가는 것에 있다.
- 주의할 점은, Goal 도달로 인정하는 거리는 해당 커리큘럼과 반드시 동기화가 되어야 한다. Goal 도달 리워드에서 해당 거리가 감소된다면, 반드시 Contact 유지 보상의 인정 거리도 감소해야 한다.

4. 전용 train.py 를 추가해라. 

```
RuntimeError: normal expects all elements of std >= 0.0

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
```

이런 식으로 PPO에 문제가 생겨 학습이 중단되는 경우가 자주 발생한다. 따라서, 전용 train.py를 설정하여, Full Iteration을 만족하지 못하고 학습이 중단될 경우, 가장 최신의 checkpoint -500 iteration을 만족하는 checkpoint를 찾아, 자동적으로 resume 되도록 train.py를 구축한다.

주의할 점은, resume시 커리큘럼도 같이 따라와야 한다는 것이다. 3000 iteration에서 터져서 2500에서 재학습 할 때, 이미 2500 iteration으로 취급되어 진행된 커리큘럼이 적용되어야 한다는 뜻이다.

이 train.py는 이후 작성할 환경에도 재사용할 수 있도록 구현하라.

---

다음을 확인해라.

가속 페널티에 상한 값이 존재하는지 체크해라. 상한이 없다면, 순간적으로 너무 큰 가속이 발생하면 해당 페널티가 발산할 가능성이 있다고 생각된다. 따라서 상한이 존재하지 않는다면, 가속도가 0.5m/s^2 수준에서 MAX가 되도록 설정해라. (음의 Weight를 무시하고 하는 말이다. 요점은, 해당 항이 발산하지 않도록 점검할 것)

또한, velocity - self._previous_velocity 방식이 아니라, IsaacLab 자체에서 가속도 값을 얻을 수는 없는지 체크해라. 만약 가능하다면, 해당 방식으로 변경하고, 불가능하다면 기존 방식을 유지해라.