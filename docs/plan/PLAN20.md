Single Contact v0을 참조하여, "src/inspire_sweep_baseline/inspire_sweep_baseline/envs" 안에 새로운 파일을 만들고 새로운 계통의 환경을 만들어줘.
환경 이름은, Inspire-Sweep-Prof-Revision-v0 으로 삼는다.

우선은, 크게 다음과 같은 변화를 준다.

1. Observation Space의 변경

- 타겟 물체 및 장애물의 변경: "src/observed_obb" 를 참조하여, "보이는 3D Bounding Box" 으로 변경한다. 다만, 장애물의 경우 지금과 유사하게, 정면에서 봤을 때, 어느 이상은 노출되어야 인정해준다.

- 타겟 물체의 관측 차원은, 관측 가능한 OBB의 중심점 (x,y,z), Dimensions (w,d,h), Orientation (yaw) 로 준다. 단, OBB 산출은 정면에서 볼 때 기준으로 1번만 수행하고, 관측 자체는 EEF Frame 기준으로 본다. 아마 중심점과 Orientation만 변경될 것이다.

- 장애물 인코더도 변경한다. 원래 [Mask, x, y, z] 였는데, [Mask, x, y, z, w, d, h, yaw] * 16 을 인코딩하는 방식을 쓴다. 128 -> 64 -> 32 -> 16 으로 잡으면 되려나? 이 부분은 어느 정도 추천에 맡긴다. 주의할 점은, "관측 가능한 OBB" 로 구성된 장애물이라는 것이다.

- EEF Pose 뒤에, 5개 손가락의 지문 정도 위치에 해당하는 위치의 Pose (EEF 기준) 을 추가해야 한다. (x, y, z, qx, qy, qz, qw) * 5 = 35차원이 추가된다.

결과적으로 이런 느낌이 된다.

    - Arm Joint Position
    - Arm Joint Velocity
    - EEF Pose (w.r.t Base)
    - Finger Pose * 5 (w.r.t EEF)
    - Target Observable OBB
    - Goal Position
    - Last Action
    - Tactile
    - Wrench
    - Encoded Obstacles

2. Reward의 변경

- Reaching 의 경우, -v1을 참고하여, "줄어든 거리"와 "늘어난 거리"에 대항하는 식으로 변경한다. 다만, 목표점은 관측 가능한 OBB의 Centriod이며, 허용 거리를 10cm 으로 설정한다.

- Contact Formation 이라는 보상을 추가한다. 이 보상은, 5개의 Finger Tip과 Bounding Box "표면"과의 최소 거리의 합에 반비례하는 구조다. 즉, 최소 거리 합이 작을 수록 보상이 크다. 유의할 점은, Centriod가 아니라, OBB의 표면이라는 것이며, Bounding Box 안으로 파고 들 경우 (물체의 실제 부피보다 BBOX가 크기 때문에) "최소 거리"는 0으로 취급한다. 보상 수준은, Reaching의 절반 수준을 기대한다. 다만, 이 절반이라는 것은 엄격하게 제한하는 것이 아니라, 수식 상으로, 그 정도 수준이 오도록 식과 Weight를 설정하라는 뜻이다.

- Sweeping 의 경우, 원래는 물체의 실제 Centriod와 목표점 사이의 XY 거리로 계산했는데, 이를 "관측 가능한 OBB의 중심점"과 목표점 사이의 XY 거리로 사용한다. 다만, 관측 가능한 OBB가 작업 도중 변경되지는 않는다. 즉 왼쪽 일부만 보였다면, 그 왼쪽 일부가 목표점에 도달하는 것이 목표인 것이다.

- Single Contact v0 대비, 정방향, 역방향 Sweeping의 보상 계수를 2배로 한다.

- 기존 Reward에 있었던, EEF Z 제한을 삭제한다.

