"src/inspire_sweep_baseline" 이 패키지 내부에 새로운 환경을 구상하고자 한다.
기본적으로 상속관계를 끊고, 독립적인 CFG들을 구성할 계획이지만, "src/inspire_sweep_baseline/inspire_sweep_baseline/envs/sweep_simple_envs.py" 안에 있는 환경들을 많이 참조하라.


1. Scene 구성 방법

- Robot System : 기본적으로 원래 사용하던 UR5e + Axia80-M8 + RH56E2 구성에, F/T 피드백 + Tactile 피드백 구성을 유지한다.
- Shelf Environment : 이 부분에서는 상당히 많은 변화를 가져온다. 
    - IK : 기존에는 선반 밖에서 왼쪽으로 미는가 오른쪽으로 미는가에 대하여만 랜덤화를 했는데, 이를 전면 수정한다. 손바닥이 왼쪽을 바라보는 느낌으로, 밀어야 하는 반대 방향에 손을 위치시키도록 IK를 풀어서 위치시킨다. 예를 들어, 목표하는 미는 방향이 5시쪽으로 밀어낸다면, 손은 1시쯤에 가있고, 손바닥이 5시 방향을 가르키는 느낌이 된다. 물체와의 거리는 물체의 지름을 어느 정도는 고려해야 하지만, 대략 5cm 정도 떨어진 지점을 예상한다. 다만, 완벽한 고정 위치는 아니고, 해당 위치에 랜덤화가 적용된다. XY(하늘 방향이 Z로 볼 때) 랜덤화, 및 Yawing 랜덤화를 어느 정도 수행한다. 이는, 다른 정책에 의하여, "그 옆 위치에 도달" 하는 것을 모사하는 행위이다.
    - Object Randomization : 기존에는 Sweeping을 위해 필요한 공간에만 스폰하도록 해있으나, 이제 IK 때문에 로봇도 선반 안에 있음으로, 밀기 위한 공간 (기존과 같음) + 로봇이 차지하는 공간이 스폰 가능 영역으로 취급된다. 로봇과 스폰되는 물체가 겹치지 않도록 유의할 것

2. Observation 구성 방법

src/surface_point 를 참조해라. 로봇이 들어가기 전을 기준으로, 정면에서 Depth Camera + Image Segmentation을 통해 얻은 대표점을 물체의 위치로서 취급한다. 다만, 해당 대표점을 EEF Frame 기준으로 Transform 하여 관측한다.

3. Reward 구성 방법

Inspire-Sweep-Simple-Finger-ForceDirection-v0 를 많이 참조한다. 다만 변경점은 다음과 같다.

    1. Reaching 보상 : 삭제한다. 왜냐면 이미 IK로 인해서 10cm 언더로 들어온 상태이기 때문이다. 
    2. Sweeping 보상 : 기존에 정방향 밀기 + 접촉 유지 보상 + 방향 보상을 결합한 새로운 형태를 제안한다. 구체적으로, 현재 물체 위치 - 목표 점 간의 XY 평면상 거리를 계측하고, 이것의 변화량을 확인한다. 그리고, (줄어든 변화량) * (접촉 카운트) * (힘에 의한 방향 유사도) 형태로 식을 짜낸다. 이 방법을 통해, "접촉을 유지하면서 올바른 방향으로 거리를 감소할 것" 을 유도한다. 거리가 증가할 경우는 별도의 식으로 처리한다. 기존과 같은데, 페널티 수준만 1.5배로 처리한다. 이는 잘못된 방향으로 거리를 증가시키면 페널티가 감소할 것을 우려한 설계이다.
    3. 그 외, Action L-2, Velocity, 그리고 충격량 등의 페널티는 유지한다. 
    4. Sweeping 이전에, Stabilize Sequence를 추가한다. 해당 시퀸스에서는, 물체를 정확하게 밀기 위한 자세를 만드는 단계로, 해당 시퀸스에서 원하는 수준이 되어야 Gate를 열고, Sweeping을 받을 수 있게 처리한다. 해당 단계는 아래를 참조해라.

4. 그 외

- Termination은 유지
- Curriculum은 Deadzone 삭제
- Action은 Action clip 적용 상태 유지 (PPO Runner 쪽 설정)


---

현재 과업에서는 **‘목표 방향으로 작은 힘을 전달하면서, 물체 중심 기준 모멘트를 최소화하는 것’**을 하나의 보상으로 묶으면 됩니다. 물체 이동과 과도한 압착은 별도 페널티로 처리합니다.

아래는 구현을 위한 제안식입니다.

**1. 보상 계산에 사용할 값**

손이 타겟에 가하는 접촉력과 질량중심 기준 모멘트를 계산합니다.

\[
\mathbf F_t=\sum_j\mathbf f_{j,t},
\qquad
\boldsymbol\tau_t
=
\sum_j(\mathbf p_{j,t}-\mathbf c_t)\times\mathbf f_{j,t}
\]

목표 방향 단위벡터 \(\mathbf d\)를 기준으로 힘을 분리합니다.

\[
F_{\parallel,t}=\mathbf d^\top\mathbf F_t,
\qquad
\mathbf F_{\perp,t}
=\mathbf F_t-F_{\parallel,t}\mathbf d
\]

여기서 **테이블 반력·중력·장애물 접촉력은 제외**하고 손–타겟 접촉만 합산합니다.

**2. 핵심 보상: 작은 목표 힘 + 작은 모멘트**

먼저 세 가지 오차를 정의합니다.

\[
e_{\mathrm{force},t}
=
\frac{F_{\parallel,t}-F_{\mathrm{test}}}{\sigma_F}
\]

\[
e_{\mathrm{direction},t}^{2}
=
\frac{\|\mathbf F_{\perp,t}\|_2^2}{\sigma_\perp^2}
\]

\[
e_{\mathrm{moment},t}^{2}
=
\frac{\tau_{z,t}^{2}}{\sigma_{\tau,z}^{2}}
+
\frac{\tau_{x,t}^{2}+\tau_{y,t}^{2}}
{\sigma_{\tau,xy}^{2}}
\]

| 항목 | 원하는 상태 |
|---|---|
| \(e_{\mathrm{force}}\) | 목표 방향 힘이 작은 기준값 \(F_{\mathrm{test}}\)와 일치 |
| \(e_{\mathrm{direction}}\) | 옆·위·아래로 전달되는 힘이 0 |
| \(e_{\mathrm{moment}}\) | 물체를 비틀거나 기울이는 모멘트가 0 |

이를 결합한 보상은 다음과 같습니다.

\[
\boxed{
R_{\mathrm{setup},t}
=
g_t
\exp\left[
-e_{\mathrm{force},t}^{2}
-e_{\mathrm{direction},t}^{2}
-e_{\mathrm{moment},t}^{2}
\right]
}
\]

단, 힘을 주지 않는 행동이 보상받지 않도록 다음 계수를 곱합니다.

\[
\boxed{
g_t=
C_t\operatorname{clip}
\left(
\frac{F_{\parallel,t}}{F_{\mathrm{test}}},
0,1
\right)
}
\]

\(C_t\)는 타겟 접촉 여부입니다.

**목표 힘을 정확한 방향으로 전달하고 모멘트가 0이면 보상은 1입니다. 접촉하지 않거나 힘을 주지 않으면 0입니다.**

힘이 기준보다 강해지면 \(e_{\mathrm{force}}\)가 증가하므로 보상이 감소합니다. 방향과 모멘트가 좋아도 지나치게 세게 누르는 행동은 최대 보상을 받지 못합니다.

**3. 물체 이동·회전 페널티**

물체를 밀어놓고 멈추는 행동을 막기 위해, 매 step 변위가 아니라 **초기 위치·자세로부터의 변화**를 사용합니다.

\[
\boxed{
P_{\mathrm{move},t}
=
\left[
\frac{\|\mathbf c_t-\mathbf c_0\|_2}{\delta_p}-1
\right]_+^2
}
\]

\[
\boxed{
P_{\mathrm{rotate},t}
=
\left[
\frac{\theta(R_0^\top R_t)}{\delta_R}-1
\right]_+^2
}
\]

여기서 \([x]_+=\max(x,0)\)입니다.

- \(\delta_p\): 허용하는 미세 이동량.
- \(\delta_R\): 허용하는 미세 회전각.
- 허용 범위 이내에서는 페널티가 없고, 초과분부터 증가합니다.

**모멘트는 ‘회전시키려는 하중’을, 회전 페널티는 ‘실제로 회전한 결과’를 평가합니다.** 따라서 두 항의 역할은 다릅니다.

**4. 과도한 압착 페널티**

여러 손가락의 힘이 서로 상쇄되는 경우를 잡기 위해 개별 접촉력의 크기를 합합니다.

\[
S_t=\sum_j\|\mathbf f_{j,t}\|_2
\]

\[
\boxed{
P_{\mathrm{squeeze},t}
=
\left[
\frac{S_t}{S_{\max}}-1
\right]_+^2
}
\]

예를 들어 양쪽에서 강하게 쥐면 전체 합력은 작아도 \(S_t\)는 커집니다. 이 항은 그런 행동을 억제합니다.

**5. 최종 보상식**

\[
\boxed{
r_t
=
\Delta t\left[
w_rR_{\mathrm{reach},t}
+
w_sR_{\mathrm{setup},t}
-
w_pP_{\mathrm{move},t}
-
w_RP_{\mathrm{rotate},t}
-
w_cP_{\mathrm{squeeze},t}
-
P_{\mathrm{other},t}
\right]
}
\]

- \(R_{\mathrm{reach}}\): 기존 접근 보상. 충분히 가까워지면 포화.
- \(R_{\mathrm{setup}}\): 이번 과업의 핵심 보상.
- \(P_{\mathrm{other}}\): 기존 F/T 급변·장애물 충돌·속도·action 페널티.

환경에서 이미 reward에 \(\Delta t\)를 곱한다면 함수 안에서 중복 적용하지 않습니다.

**6. 파라미터는 이렇게 해석하면 됩니다.**

| 파라미터 | 설정 의미 |
|---|---|
| \(F_{\mathrm{test}}\) | 물체를 움직이지 않고 접촉을 시험할 작은 힘 |
| \(\sigma_F\) | 목표 힘 크기의 허용 오차 폭 |
| \(\sigma_\perp\) | 불필요한 방향 힘의 허용 폭 |
| \(\sigma_{\tau,z}\) | yaw 모멘트의 허용 폭 |
| \(\sigma_{\tau,xy}\) | 기울어짐 모멘트의 허용 폭 |
| \(S_{\max}\) | 개별 접촉 하중 합의 상한 |

각 오차가 해당 \(\sigma\)와 같아지면, 그 항만으로 보상이 \(e^{-1}\approx0.37\)배가 됩니다. **엄격하게 만들고 싶은 항의 \(\sigma\)를 작게 설정**하면 됩니다.

처음에는 별도의 접촉 개수 보상을 추가하지 않고 이 구성으로 시작하는 편이 좋습니다. 성공은 **목표 힘·방향·모멘트와 물체 정지 조건을 일정 시간 연속 만족했는지**로 판정하면 됩니다.


---