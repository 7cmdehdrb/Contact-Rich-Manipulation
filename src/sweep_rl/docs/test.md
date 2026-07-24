



## 결론

**물체의 현재 위치를 Actor Observation에서 제외하는 것은 가능하다.** 다만 정확한 pushing을 학습하려면 다음 조건이 필요하다.

1. **시뮬레이터의 물체 위치·속도는 Reward 계산과 Critic 학습에는 사용한다.**
2. Actor는 실제 배치 가능한 센서 관측만 사용한다.
3. Actor는 단일 시점 MLP가 아니라 **관측 이력을 처리하는 GRU/LSTM 정책**을 사용한다.

물체 위치를 Reward에서도 사용하지 않으면, 마찰·미끄러짐·물체 회전에 의해 동일한 F/T 및 로봇 움직임이 서로 다른 물체 궤적으로 이어질 수 있으므로, “정확한 물체 이동”을 직접 최적화할 수 없다.

---

# 1. 현재 시스템 구성

README 기준 현재 Low-Level Policy는 다음 구조이다.

- **Robot**: UR5e
- **Controller**: Operational Space Control
- **Actor Observation**
  - Joint position, velocity, effort
  - Contact Sensor 및 Particle Filter 기반 접촉 정보
  - Wrist F/T Sensor
  - User Command
  - Initial Target Pose
  - Last Action
- **Action**
  - 6D OSC target pose
  - 6D stiffness
- **Randomization**
  - 물체 질량·관성·마찰
  - 센서 노이즈
  - 제어기 및 관절 파라미터

즉, 정책은 물체의 초기 위치는 알지만 현재 위치는 직접 관측하지 않는 **POMDP** 구조이다. fileciteturn0file0

Yang 등의 tactile pushing은 비전 없이 동작하지만, 실제 정책 입력에는 tactile contact pose와 contact-to-goal 상대 위치가 포함된다. 따라서 물체의 현재 위치를 전혀 입력하지 않는 현재 시스템보다 관측성이 높으며, 해당 논문의 Reward를 그대로 적용하기는 어렵다. fileciteturn0file5

반면 Force Push는 물체 pose를 사용하지 않고 접촉력 방향과 pusher 위치만으로 pushing을 수행한다. 여기서 핵심 피드백은 **접촉력 방향과 목표 경로 방향의 차이**이다. fileciteturn0file2

---

# 2. 권장 Task Definition

User Command가 다음을 정의한다고 가정한다.

\[
\mathcal{C}
=
\left(
\hat{\mathbf d},
v^\star,
L
\right)
\]

- \(\hat{\mathbf d}\): World frame 기준 목표 pushing 방향
- \(v^\star\): 목표 물체 이동 속도
- \(L\): 목표 pushing 거리

물체의 시뮬레이션 Ground Truth를 다음과 같이 정의한다.

- \(\mathbf p_t^o\): 물체 중심의 현재 평면 위치
- \(\mathbf v_t^o\): 물체 중심의 현재 평면 속도
- \(\mathbf p_0^o\): 초기 물체 위치
- \(\mathbf f_t\): **로봇이 물체에 가하는** 평면 접촉력
- \(c_t\): robot–object 접촉 여부

F/T 센서가 물체가 로봇에 가하는 힘을 출력한다면, World frame으로 변환한 뒤 부호를 반전하여 \(\mathbf f_t\)를 구성해야 한다.

목표 방향 및 횡방향 투영 행렬은 다음과 같다.

\[
\mathbf P_\perp
=
\mathbf I-\hat{\mathbf d}\hat{\mathbf d}^{T}
\]

\[
s_t
=
\hat{\mathbf d}^{T}
\left(
\mathbf p_t^o-\mathbf p_0^o
\right)
\]

\[
\mathbf e_{\perp,t}
=
\mathbf P_\perp
\left(
\mathbf p_t^o-\mathbf p_0^o
\right)
\]

\[
v_{\parallel,t}
=
\hat{\mathbf d}^{T}\mathbf v_t^o,
\qquad
\mathbf v_{\perp,t}
=
\mathbf P_\perp\mathbf v_t^o
\]

---

# 3. 권장 Reward Formulation

초기 순간부터 목표 속도를 요구하면 속도 추종 항과 가속도 패널티가 충돌한다. 따라서 다음과 같은 smooth velocity reference를 사용한다.

\[
v_t^{\mathrm{ref}}
=
v^\star
\left(
1-e^{-t\Delta t/\tau_r}
\right)
\]

\[
s_t^{\mathrm{ref}}
=
v^\star
\left[
t\Delta t
-
\tau_r
\left(
1-e^{-t\Delta t/\tau_r}
\right)
\right]
\]

여기서 \(\tau_r\)는 일반적으로 \(0.2\sim0.5\)초의 acceleration ramp time이다.

최종 Reward는 다음과 같이 구성하는 것이 적절하다.

\[
\boxed{
\begin{aligned}
r_t
={}&
w_{\Delta s}
\operatorname{clip}
\left(
\frac{s_t-s_{t-1}}
     {v^\star\Delta t},
-1,1
\right)
\\
&-
w_s
\left(
\frac{s_t-s_t^{\mathrm{ref}}}
     {\epsilon_s}
\right)^2
-
w_\perp
\left(
\frac{\|\mathbf e_{\perp,t}\|}
     {\epsilon_\perp}
\right)^2
\\
&-
w_v
\left(
\frac{v_{\parallel,t}-v_t^{\mathrm{ref}}}
     {\epsilon_v}
\right)^2
-
w_{v\perp}
\left(
\frac{\|\mathbf v_{\perp,t}\|}
     {\epsilon_{v\perp}}
\right)^2
\\
&-
w_a
\left(
\frac{
\|\mathbf v_t^o-\mathbf v_{t-1}^o\|
}{
a^\star\Delta t
}
\right)^2
\\
&-
w_f c_t
\left(
1-\hat{\mathbf f}_t^{T}\hat{\mathbf d}
\right)
-
w_{\mathrm{lost}}\mathbb{I}_{\mathrm{lost},t}
\\
&-
w_{\mathrm{safe}}
\left[
\operatorname{ReLU}
\left(
\frac{
\|\mathbf f_t\|-F_{\mathrm{safe}}
}{
F_{\max}-F_{\mathrm{safe}}
}
\right)
\right]^2
\\
&-
w_{\mathrm{tilt}}
\left(
1-\hat{\mathbf z}_{ee,t}^{T}
\hat{\mathbf z}_{ee}^{\mathrm{ref}}
\right)
\\
&-
w_{\Delta u}
\left\|
\tilde{\mathbf u}_t-
\tilde{\mathbf u}_{t-1}
\right\|^2
-
w_{\Delta K}
\left\|
\tilde{\mathbf K}_t-
\tilde{\mathbf K}_{t-1}
\right\|^2
+
r_t^{\mathrm{terminal}} .
\end{aligned}
}
\]

## 각 항의 의미

| 항 | 목적 |
|---|---|
| \(\Delta s\) | 물체가 실제 목표 방향으로 이동하도록 유도 |
| \(s-s^{ref}\) | 목표 속도 프로파일에 따른 longitudinal 위치 추종 |
| \(\mathbf e_\perp\) | 직선 경로에서 벗어나는 lateral displacement 억제 |
| \(v_\parallel-v^{ref}\) | 일정한 pushing 속도 유지 |
| \(\mathbf v_\perp\) | 물체의 횡방향 미끄러짐 억제 |
| \(\Delta\mathbf v^o\) | 물체 가속도 및 충격성 움직임 억제 |
| \(1-\hat{\mathbf f}^{T}\hat{\mathbf d}\) | 접촉력 방향을 목표 pushing 방향과 정렬 |
| \(\mathbb I_{\mathrm{lost}}\) | 접촉 후 contact loss 방지 |
| Over-force barrier | 특정 힘을 추종하지 않고 과도한 힘만 제한 |
| EE tilt | 손목이 꺾이거나 tool이 기울어지는 현상 방지 |
| \(\Delta u,\Delta K\) | OSC 목표와 stiffness의 급격한 변화 억제 |

Force Push의 핵심 제어식도 접촉력 각도와 목표 경로 방향의 차이를 이용하며, 접촉력이 임계값보다 작을 때 contact recovery를 수행하고 힘이 과도할 때만 admittance로 속도를 줄인다. 따라서 **고정된 목표 힘을 Reward로 설정하지 않고**, force direction과 safety barrier만 사용하는 것이 현재 목적에 더 적합하다. fileciteturn0file2

Beltran-Hernandez 등의 Reward도 목표 오차, action 크기, 접촉력, 시간 패널티, 성공 및 안전 위반으로 구성된다. 다만 해당 연구의 contact-force 항은 정밀 조립 작업에 맞춰져 있으므로, pushing에서는 고정 force tracking보다 상한 기반 패널티가 적절하다. fileciteturn0file3

---

# 4. 권장 초기 가중치

모든 오차를 허용 오차 \(\epsilon\)으로 정규화했다는 조건에서 다음 값으로 시작할 수 있다.

| 계수 | 초기값 |
|---|---:|
| \(w_{\Delta s}\) | 1.0 |
| \(w_s\) | 0.5 |
| \(w_\perp\) | 2.0 |
| \(w_v\) | 1.0 |
| \(w_{v\perp}\) | 0.5 |
| \(w_a\) | 0.05 |
| \(w_f\) | 0.25 |
| \(w_{\mathrm{lost}}\) | 1.0 |
| \(w_{\mathrm{safe}}\) | 2.0 |
| \(w_{\mathrm{tilt}}\) | 0.25 |
| \(w_{\Delta u}\) | 0.01 |
| \(w_{\Delta K}\) | 0.005 |

권장 normalization scale은 다음과 같다.

\[
\epsilon_\perp
=
\text{허용 가능한 lateral error}
\]

\[
\epsilon_s
=
0.5L_{\mathrm{object}}
\]

\[
\epsilon_v
=
\epsilon_{v\perp}
=
0.25v^\star
\]

\[
a^\star
=
\frac{v^\star}{\tau_r}
\]

\(F_{\mathrm{safe}}\)와 \(F_{\max}\)는 로봇, gripper, tool, F/T 센서 및 물체의 허용 하중을 기준으로 별도 설정해야 한다.

---

# 5. Terminal Reward와 종료 조건

\[
r_t^{\mathrm{terminal}}
=
\begin{cases}
+10,
&
s_t\ge L
\;\land\;
\|\mathbf e_{\perp,t}\|
\le\epsilon_{\mathrm{success}}
\\[2mm]
-10,
&
\text{safety violation}
\\[1mm]
-2,
&
\text{timeout}
\\[1mm]
0,
&
\text{otherwise}
\end{cases}
\]

Safety violation에는 다음을 포함한다.

- \(\|\mathbf f_t\|>F_{\max}\)
- 물체가 작업 영역 밖으로 이탈
- 물체가 테이블에서 떨어짐
- gripper 이외의 링크가 물체 또는 환경과 충돌
- joint position, velocity 또는 OSC workspace 제한 위반

물체 방향 유지가 실제 task requirement라면 다음 항을 추가한다.

\[
-
w_\psi
\left(
\frac{
\operatorname{wrap}
(\psi_t^o-\psi_0^o)
}{
\epsilon_\psi
}
\right)^2
\]

그러나 다양한 형상에 대한 sweep가 목적이라면 물체 회전이 자연스럽거나 필요한 경우가 있으므로, 기본 Reward에는 포함하지 않는 편이 낫다.

---

# 6. Reward만으로는 부족한 부분

현재 Observation은 현재 물체 pose를 포함하지 않으므로, 단일 시점 MLP 정책은 서로 다른 실제 상태를 동일한 Observation으로 인식할 수 있다. `Last Action` 하나만 추가하는 것으로는 누적 미끄러짐과 접촉 이력을 복원하기 어렵다.

권장 Actor 구조는 다음과 같다.

\[
\mathbf h_t
=
\operatorname{GRU}
(
\mathbf h_{t-1},
[\mathbf o_t,\mathbf u_{t-1}]
)
\]

\[
\mathbf u_t
=
\pi_\theta(\mathbf h_t)
\]

시뮬레이션에서만 사용할 auxiliary estimation head를 추가하는 것이 더 안정적이다.

\[
\mathcal L_{\mathrm{aux}}
=
\lambda_p
\left\|
\widehat{\Delta\mathbf p_t^o}
-
\Delta\mathbf p_t^o
\right\|^2
+
\lambda_v
\left\|
\widehat{\mathbf v_t^o}
-
\mathbf v_t^o
\right\|^2
\]

- 입력: Actor의 recurrent hidden state
- 정답: 시뮬레이터의 물체 displacement 및 velocity
- 실제 실행: auxiliary head와 Ground Truth 제거
- Actor에는 예측된 물체 위치도 직접 제공하지 않고 hidden representation만 사용 가능

BGN은 부분 관측 환경에서 관측·행동 이력으로 belief를 복원하는 auxiliary loss를 사용했으며, 실행 시에는 belief가 필요하지 않았다. 단순한 asymmetric critic보다 이력 표현을 직접 학습시키는 방식이 더 효과적일 수 있음을 보여준다. fileciteturn0file1 최근의 non-prehensile manipulation 연구에서도 force와 proprioception을 이용한 recurrent state estimator를 먼저 학습하고, 그 추정치와 불확실성을 제어 정책에 반영하는 구조가 사용되었다. citeturn282703view1turn937936view3

따라서 현재 시스템에는 다음 조합이 가장 적합하다.

> **Deployable observations 기반 recurrent Actor + object Ground Truth 기반 Reward + privileged Critic + object-motion auxiliary loss**

이 구조에서는 물체 위치가 Actor Observation으로 유출되지 않으면서도, Reward가 실제 물체 궤적의 정확도·속도·안전성을 직접 감독할 수 있다.
