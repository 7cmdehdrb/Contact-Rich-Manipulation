아래 기준으로 판정했습니다.

* **조건 1:** 단순 contact flag가 아니라, 힘을 **수치값**으로 출력해야 함.
* **조건 2:** 센서 셀을 여러 개 배치하여 **taxel/grid 형태**로 구성할 수 있어야 함.
* “그리드 배치”는 공식 API가 직접 2D tactile array를 제공하거나, 최소한 여러 센서/프로브를 격자 형태로 배치해 각 위치별 force를 받을 수 있으면 조건 충족으로 봤습니다.

| Simulator                 | 정량 힘 출력 | 그리드 배치 | 판정         | 부족한 부분                                                                                                                                          |
| ------------------------- | ------: | -----: | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Isaac Sim / Isaac Lab** |      가능 |  부분 가능 | **조건부 가능** | 기본 Contact Sensor는 rigid body 또는 body별 net contact force 중심. 진짜 tactile grid는 Isaac Lab의 Visuo-Tactile/TacSL 쪽을 써야 함.                           |
| **MuJoCo**                |      가능 |     가능 | **가능**     | 기본 `touch`는 scalar normal force만 출력. shear force field까지 원하면 직접 구현 필요. 최신 `tactile` 센서는 force가 아니라 penetration/sliding velocity 계열이라 조건 1에 부적합. |
| **Genesis**               |      가능 |     가능 | **가장 적합**  | 공식 센서 테이블상 grid/probe 기반 force가 명확함. 다만 물리적 elastomer 기반 고충실도 tactile인지, 단순 kinematic probe인지 목적에 따라 구분 필요.                                     |
| **Newton**                |      가능 |  부분 가능 | **조건부 가능** | Contact force sensor는 존재하지만, 공식 문서 기준 “2D tactile grid/taxel array”가 명확한 상위 센서로 정리되어 있지는 않음. 직접 여러 contact sensor/shape를 배치해야 할 가능성이 큼.         |

## 1. Isaac Sim / Isaac Lab

**기본 Contact Sensor는 조건 1을 만족합니다.** Isaac Sim 문서는 Contact Sensor가 PhysX Contact Report API를 사용하며, contact cell 또는 pressure-based sensor와 유사한 force reading을 제공한다고 설명합니다. 또한 min/max force threshold와 force output이 있고, 예제에서도 force output을 읽습니다. ([Isaac Sim Documentation][1])

Isaac Lab의 Contact Sensor도 **net contact force**와 **filtered force matrix**를 출력합니다. 문서 예제에서는 `net_forces_w`와 `force_matrix_w`가 CUDA tensor 형태로 출력되며, 실제 예시 값도 N 단위로 해석 가능한 3D force vector입니다. ([Isaac Sim][2])

다만 **기본 Contact Sensor만으로는 촘촘한 tactile grid라고 보기는 어렵습니다.** Isaac Lab 문서상 contact sensor는 body 단위 net force가 중심이고, filtering도 “many-to-one” 제한이 있습니다. 여러 body를 쓰면 force matrix가 `None`이 되는 경우가 명시되어 있습니다. ([Isaac Sim][2])

대신 Isaac Lab에는 **Visuo-Tactile Sensor**가 따로 있으며, TacSL과 통합되어 GelSight류 센서처럼 tactile RGB, force field distribution, intermediate tactile measurement를 제공한다고 문서화되어 있습니다. 또한 sensor resolution, force sensitivity, output data type, `enable_force_field` 등을 설정할 수 있습니다. ([Isaac Sim][3])

**결론:**
Isaac Sim/Isaac Lab은 **조건부 가능**입니다.
단순 force sensor라면 충분하지만, “그리드 tactile taxel” 목적이면 기본 Contact Sensor보다 **Isaac Lab Visuo-Tactile Sensor / TacSL 기반**을 써야 합니다.

## 2. MuJoCo

MuJoCo는 **정량 힘 출력 조건을 만족합니다.** `sensor/touch`는 site volume 안의 contact force를 포함하며, 여러 contact의 **normal force scalar 합**을 non-negative scalar로 출력한다고 문서화되어 있습니다. ([MuJoCo Documentation][4])

또한 MuJoCo에는 `sensor/force`와 `sensor/torque`도 있습니다. `sensor/force`는 child-parent body 사이 interaction force를 3축으로 출력하며, contact와 external perturbation을 포함합니다. 다만 이것은 tactile taxel이라기보다 F/T 센서 모델에 가깝습니다. ([MuJoCo Documentation][4])

**그리드 배치는 가능하다고 봅니다.** MuJoCo `touch` sensor는 site 기반이므로, 작은 site들을 표면에 격자로 배치하면 각 site별 scalar normal force를 얻을 수 있습니다. 실제로 Shadow Dexterous Hand 환경은 MuJoCo touch sensor 92개를 palm/finger phalanx에 배치해 tactile observation으로 사용합니다. ([로보틱스 문서][5])

주의할 점은 MuJoCo 최신 문서의 별도 `tactile` 센서입니다. 이 센서는 force가 아니라 **penetration depth**와 **tangent-frame sliding velocity**를 반환한다고 되어 있습니다. 따라서 사용자가 요구한 “정확한 힘 수치” 기준으로는 `tactile` 센서 자체는 부적합하고, `touch` sensor 또는 force sensor를 써야 합니다. ([MuJoCo Documentation][6])

**결론:**
MuJoCo는 **가능**입니다.
다만 grid tactile force field를 원하면 `touch` sensor를 여러 개 배치하는 방식이고, 기본 출력은 **normal scalar force**입니다. 3축 taxel force나 shear force까지 필요하면 추가 모델링이 필요합니다.

## 3. Genesis

Genesis는 네 개 중에서 조건에 가장 잘 맞습니다.

공식 문서상 Genesis는 `ContactForceSensor`를 제공하며, associated rigid link에 작용하는 **total contact force**를 local frame에서 측정한다고 설명합니다. ([Genesis World][7])

최신 센서 테이블에서는 `ContactForceSensor`가 `float32`, shape `([n_envs,] 3)`의 3D force를 반환한다고 되어 있습니다. 즉 조건 1을 만족합니다. ([Genesis World][8])

더 중요한 부분은 grid/probe 기반 tactile입니다. Genesis 문서에는 `KinematicContactProbe`가 있으며, 반환 필드가 `penetration`, `force`이고 shape가 각각 `([n_envs,] n_probes)`, `([n_envs,] n_probes, 3)`입니다. 즉 여러 probe를 taxel처럼 배치하면 위치별 3D force grid를 만들 수 있습니다. ([Genesis World][8])

또한 Genesis의 센서 목록에는 `ElastomerDisplacementSensor`와 `TemperatureGridSensor`도 있어, tactile simulation 쪽 확장성을 염두에 둔 구조입니다. ([Genesis World][8])

**결론:**
Genesis는 **가능**, 그리고 네 후보 중 **가장 직접적으로 조건을 만족**합니다.
특히 `KinematicContactProbe`의 `n_probes × 3D force` 구조가 사용자가 말한 grid tactile/force sensor 요구와 가장 잘 맞습니다.

## 4. Newton

Newton은 **정량 force 출력은 가능**합니다. Newton 예제에는 `SensorContact`가 있고, contact forces와 per-counterpart breakdown을 평가한다고 되어 있습니다. 예제 코드에서는 `total_force`, `force_matrix`를 읽고, flap이나 plate에 작용하는 contact force를 로깅합니다. ([GitHub][9])

Newton 자체는 NVIDIA Warp 기반의 GPU-accelerated, extensible, differentiable physics engine이며, MuJoCo Warp를 주요 backend로 통합한다고 설명됩니다. ([GitHub][10]) Isaac Sim 6.0 문서에서도 Newton backend가 MuJoCo Warp를 통합하고 XPBD, MuJoCo, Featherstone, SemiImplicit solver를 제공한다고 설명합니다. ([Isaac Sim Documentation][11])

하지만 **그리드 tactile sensor는 공식적으로 명확하지 않습니다.** Newton GitHub 이슈를 보면 2025년에 contact sensor 구현이 ANYmal C 환경에 필요하다는 task가 있었고, MuJoCo Warp의 touch sensor와 site support가 필요할 것 같다고 적혀 있습니다. ([GitHub][12]) 현재는 `SensorContact` 예제가 존재하지만, Isaac Lab이나 Genesis처럼 “taxel grid / n_probes force array”가 공식 센서 타입으로 정리되어 있는 수준은 확인되지 않았습니다. ([GitHub][9])

**결론:**
Newton은 **조건부 가능**입니다.
정량 contact force는 가능하지만, 사용자가 원하는 **grid tactile force sensor**는 직접 여러 contact sensor 또는 sensing shape를 격자 형태로 구성해야 할 가능성이 큽니다. 공식 문서만 보면 Genesis보다 덜 명확합니다.

## 최종 추천

연구 목적이 **tactile/force grid observation을 정책 입력으로 넣는 것**이라면 우선순위는 다음이 타당합니다.

1. **Genesis**
   `KinematicContactProbe`의 `n_probes × 3D force` 구조가 가장 직접적입니다.

2. **MuJoCo**
   `touch` sensor를 여러 site로 격자 배치하면 구현 가능합니다. 다만 기본값은 scalar normal force입니다.

3. **Isaac Lab / Isaac Sim**
   일반 contact force는 쉽지만, grid tactile은 기본 Contact Sensor보다 Visuo-Tactile/TacSL 쪽으로 가야 합니다.

4. **Newton**
   contact force는 가능하지만, grid tactile 센서로는 아직 직접 구현 부담이 커 보입니다.

논문/실험 구현 관점에서 말하면, **“정확한 힘 수치 + grid 배치”만 중요하면 Genesis 또는 MuJoCo가 가장 현실적**입니다. **GelSight류 tactile image + force field distribution**까지 필요하면 Isaac Lab Visuo-Tactile도 후보가 됩니다.

[1]: https://docs.isaacsim.omniverse.nvidia.com/latest/sensors/isaacsim_sensors_physics_contact.html "Contact sensor — Isaac Sim Documentation"
[2]: https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/contact_sensor.html "Contact Sensor — Isaac Lab Documentation"
[3]: https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/visuo_tactile_sensor.html "Visuo-Tactile Sensor — Isaac Lab Documentation"
[4]: https://mujoco.readthedocs.io/en/latest/XMLreference.html "XML Reference - MuJoCo Documentation"
[5]: https://robotics.farama.org/envs/shadow_dexterous_hand/manipulate_block_touch_sensors/?utm_source=chatgpt.com "Manipulate Block Touch Sensors"
[6]: https://mujoco.readthedocs.io/en/stable/XMLreference.html?utm_source=chatgpt.com "XML Reference - MuJoCo Documentation"
[7]: https://genesis-world.readthedocs.io/en/v.0.4.7/api_reference/sensor/contact.html "Contact Sensors — Genesis 0.4.7 documentation"
[8]: https://genesis-world.readthedocs.io/en/latest/api_reference/sensor/index.html "Sensors — Genesis 1.2.1 documentation"
[9]: https://raw.githubusercontent.com/newton-physics/newton/main/newton/examples/sensors/example_sensor_contact.py "raw.githubusercontent.com"
[10]: https://github.com/newton-physics/newton?utm_source=chatgpt.com "newton-physics/newton"
[11]: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/physics/newton_physics.html?utm_source=chatgpt.com "Newton Physics Backend - Isaac Sim Documentation"
[12]: https://github.com/newton-physics/newton/issues/67 "Implement contact sensor · Issue #67 · newton-physics/newton · GitHub"
