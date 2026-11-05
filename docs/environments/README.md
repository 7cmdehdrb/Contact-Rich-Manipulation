# Inspire Sweep 환경 통합 안내

2026-11-05 현재 `src/inspire_sweep_baseline/inspire_sweep_baseline`의 실제 등록·설정·함수 기준. **75개 등록 환경을 13개 계열 문서**로 정리했다. 기본 환경은 관측·장면/제어·보상·종료·커리큘럼을 짧게 설명하고, 파생 환경은 직접 부모와 변경점만 적는다.

## 문서 지도

| 계열 | 등록 수 | 기억할 구성 |
| --- | ---: | --- |
| [Sweep Baseline 계열](01_sweep.md) | 4 | 원통 1개, 기본 관절 위치 제어 |
| [Sweep Omni 계열](02_omni.md) | 6 | 평면 목표·거리 진전 실험 |
| [Sweep Omni Renewal 계열](03_renewal.md) | 9 | 완화된 실패 조건, Dex/Tactile/OSC 분기 |
| [Contact Sweep 계열](04_contact_sweep.md) | 1 | 57D 촉각 기반 독립 기본 환경 |
| [Contact Sweep Clutter 계열](05_clutter.md) | 5 | 장애물과 초기 장면 인코더 |
| [Contact Sweep Clutter Fusion 계열](06_fusion.md) | 11 | 55D 직접 관측·보정 F/T·다양한 비용 일정 |
| [Holding 계열](07_holding.md) | 5 | 목표 정지·유지 보상과 성공 기준 |
| [Holding OSC 계열](08_holding_osc.md) | 5 | 각 Holding 버전의 Arm 제어 교체 |
| [Holding ObjectRandom 계열](09_object_random.md) | 5 | 장애물 형상 랜덤화와 개수 증가 |
| [Final 계열](10_final.md) | 6 | 독립 OSC 구성, v1 관측/보상 통합 |
| [Final GoalAccuracy 계열](11_goal_accuracy.md) | 7 | 고정 정밀도, timeout 보너스, OBB/경량 Scene |
| [Revision 계열](12_revision.md) | 6 | 8종·물리 10개·Critic 실시간 장애물 |
| [Wrench 계열](13_wrench.md) | 5 | 배치 bank·안정 성공, v1부터 wrench 제어 |

## 읽는 기준

- **부모는 EnvCfg의 직접 Python 상속 기준**이다. Scene/보상 Cfg/runtime/PPO의 상속은 다를 수 있다. Final-v0·Revision-v0·Wrench-v0는 이전 환경을 기능 기준으로 삼았어도 각각 `ManagerBasedRLEnvCfg`에서 독립 구성한다.
- 관측 **원시 차원**은 모델에 전달하는 선택 그룹의 합이다. **MLP 차원**은 장애물 `[존재, x, y, z]` 슬롯을 16D로 압축한 뒤의 입력이다. 물리 물체 개수와 관측 슬롯 개수는 다르다.
- 초기 타겟/장애물 정보는 reset 때 정해진 값을 뜻한다. Critic의 현재 타겟은 별도 3D이며, 현재 장애물도 별도로 쓰는 계열은 Revision/Wrench다. Initial-Privileged는 예외적으로 현재 물체 상태 13D를 추가한다.
- 별도 언급이 없으면 8D 행동, 물리 dt=0.01초·decimation=2·제어 dt=0.02초(50Hz), 에피소드 최대 10초다. 초기 계열은 4096개, Revision/Wrench는 2048개 환경이 기본값이다.
- 일반 보상은 `step_dt × Σ(weight × term)`으로 합산한다. 새 최단 기록 진전과 일회 이벤트는 함수가 dt를 상쇄하므로 계수를 그대로 매초 보상으로 해석하지 않는다. `+30` 같은 이벤트와 `weight 30`인 지속 항을 구분한다.
- ObjectRandom/Final의 `gray_red_padding_fraction=0.15`는 **15%의 비율**이다. GoalAccuracy-v0_2_1의 60→69mm, Revision/Wrench의 고정 65mm 같은 OBB 절대 padding과 구분한다.
- 커리큘럼의 k는 **학습 iteration 1000회**다. 시간 step이나 에피소드 수가 아니다. 표의 계수는 기본/초기값이며 일정이 있으면 학습 중 달라진다.
- 이 문서는 현재 구현을 떠올리기 위한 요약이다. 정적 코드 확인으로 작성했으며 이번 작업에서 시뮬레이션·학습 성능을 재검증한 것은 아니다.

## 계열 간 상속

```text
ManagerBasedRLEnvCfg
├─ Sweep Baseline
│  ├─ Realistic → Initial-Privileged
│  ├─ Tactile
│  ├─ Omni → 각 Omni/Dex 변형
│  └─ Renewal → 각 Dex/Tactile/OSC 변형
├─ Contact Sweep → Clutter → 각 Clutter 변형
├─ Fusion-v0 → v2 → v3 → v1_0 → 각 v1_* 변형
│                           └─ Holding-v0 → v1 → v2 → v3 → v4
│                              각 Holding-vN → OSC-vN
│                              OSC-v0/v2/v3/v4 → ObjectRandom-v0/v1/v2/v3
│                              Holding-v2 → ObjectRandom-Test-v0
├─ Final-v0 → v1 → v1-1 / Test-v1 / v2 / GoalAccuracy-v0
│                 v2 → Test-v2
│                 GoalAccuracy-v0 → v1 / v0_1 → v0_2
│                                            v0_2 → v0_3 / v0_2_1 → v0_2_2
├─ Revision-v0 → v1 → Test-v1 / v2 → Test-v2 → Test-v2-GT
└─ Wrench-v0 → v0_1 / v0_2 / v1 → v2
```

## 전체 등록 ID 색인

각 행은 [패키지 등록 코드](../../src/inspire_sweep_baseline/inspire_sweep_baseline/__init__.py)의 실제 Gym ID다. 환경/runner/runtime 연결의 원본도 해당 파일에 있다. 약칭이 비슷한 Final-v1, GoalAccuracy-v1, Revision-v1 등을 이 표로 구분한다.

| Gym ID | EnvCfg | 직접 부모 EnvCfg | 문서 |
| --- | --- | --- | --- |
| `Isaac-Inspire-Sweep-Baseline-v0` | `InspireSweepEnvCfg` | `ManagerBasedRLEnvCfg` | [Sweep Baseline 계열](01_sweep.md) |
| `Isaac-Inspire-Sweep-Realistic-Physics-v0` | `InspireSweepRealisticPhysicsEnvCfg` | `InspireSweepEnvCfg` | [Sweep Baseline 계열](01_sweep.md) |
| `Isaac-Inspire-Sweep-Realistic-Physics-Initial-Privileged-v0` | `InspireSweepRealisticInitialPrivilegedEnvCfg` | `InspireSweepRealisticPhysicsEnvCfg` | [Sweep Baseline 계열](01_sweep.md) |
| `Isaac-Inspire-Sweep-Baseline-Tactile-v0` | `InspireSweepTactileEnvCfg` | `InspireSweepEnvCfg` | [Sweep Baseline 계열](01_sweep.md) |
| `Isaac-Inspire-Sweep-Omni-v0` | `InspireSweepOmniEnvCfg` | `InspireSweepEnvCfg` | [Sweep Omni 계열](02_omni.md) |
| `Isaac-Inspire-Sweep-Omni-v1` | `InspireSweepOmniV1EnvCfg` | `InspireSweepOmniEnvCfg` | [Sweep Omni 계열](02_omni.md) |
| `Isaac-Inspire-Sweep-Omni-v2` | `InspireSweepOmniV2EnvCfg` | `InspireSweepOmniV1EnvCfg` | [Sweep Omni 계열](02_omni.md) |
| `Isaac-Inspire-Dex-Omni-v0` | `InspireDexOmniEnvCfg` | `InspireSweepOmniV2EnvCfg` | [Sweep Omni 계열](02_omni.md) |
| `Isaac-Inspire-Sweep-Omni-v3` | `InspireSweepOmniV3EnvCfg` | `InspireSweepOmniV2EnvCfg` | [Sweep Omni 계열](02_omni.md) |
| `Isaac-Inspire-Sweep-Omni-v4` | `InspireSweepOmniV4EnvCfg` | `InspireSweepOmniV3EnvCfg` | [Sweep Omni 계열](02_omni.md) |
| `Isaac-Inspire-Sweep-Omni-Renewal-v0` | `InspireSweepOmniRenewalEnvCfg` | `InspireSweepEnvCfg` | [Sweep Omni Renewal 계열](03_renewal.md) |
| `Isaac-Inspire-Sweep-Omni-Renewal-OSC-v0` | `InspireSweepOmniRenewalOSCEnvCfg` | `InspireSweepOmniRenewalEnvCfg` | [Sweep Omni Renewal 계열](03_renewal.md) |
| `Isaac-Inspire-Dex-Sweep-Omni-Renewal-v0` | `InspireDexSweepOmniRenewalEnvCfg` | `InspireSweepOmniRenewalEnvCfg` | [Sweep Omni Renewal 계열](03_renewal.md) |
| `Isaac-Inspire-Tactile-Sweep-Omni-Renewal-v0` | `InspireTactileSweepOmniRenewalEnvCfg` | `InspireSweepOmniRenewalEnvCfg` | [Sweep Omni Renewal 계열](03_renewal.md) |
| `Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-v0` | `InspireDexTactileSweepOmniRenewalEnvCfg` | `InspireDexSweepOmniRenewalEnvCfg` | [Sweep Omni Renewal 계열](03_renewal.md) |
| `Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-v1` | `InspireDexTactileSweepOmniRenewalV1EnvCfg` | `InspireDexTactileSweepOmniRenewalEnvCfg` | [Sweep Omni Renewal 계열](03_renewal.md) |
| `Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-v2` | `InspireDexTactileSweepOmniRenewalV2EnvCfg` | `InspireDexTactileSweepOmniRenewalV1EnvCfg` | [Sweep Omni Renewal 계열](03_renewal.md) |
| `Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-v3` | `InspireDexTactileSweepOmniRenewalV3EnvCfg` | `InspireDexTactileSweepOmniRenewalV1EnvCfg` | [Sweep Omni Renewal 계열](03_renewal.md) |
| `Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-OSC-v2` | `InspireDexTactileSweepOmniRenewalV2OSCEnvCfg` | `InspireDexTactileSweepOmniRenewalV2EnvCfg` | [Sweep Omni Renewal 계열](03_renewal.md) |
| `Isaac-Inspire-Contact-Sweep-v0` | `ContactSweepEnvCfg` | `ManagerBasedRLEnvCfg` | [Contact Sweep 계열](04_contact_sweep.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-FT-Test-v0` | `ContactSweepClutterFTTestEnvCfg` | `ContactSweepClutterV1CurriculumEnvCfg` | [Contact Sweep Clutter 계열](05_clutter.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Curriculum-v1` | `ContactSweepClutterV1CurriculumEnvCfg` | `ContactSweepClutterV1EnvCfg` | [Contact Sweep Clutter 계열](05_clutter.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-v2` | `ContactSweepClutterV2EnvCfg` | `ContactSweepClutterV1EnvCfg` | [Contact Sweep Clutter 계열](05_clutter.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-v1` | `ContactSweepClutterV1EnvCfg` | `ContactSweepClutterEnvCfg` | [Contact Sweep Clutter 계열](05_clutter.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-v0` | `ContactSweepClutterEnvCfg` | `ContactSweepEnvCfg` | [Contact Sweep Clutter 계열](05_clutter.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_1_1` | `ContactSweepClutterFusionV1_1_1EnvCfg` | `ContactSweepClutterFusionV1_1EnvCfg` | [Contact Sweep Clutter Fusion 계열](06_fusion.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_1_2` | `ContactSweepClutterFusionV1_1_2EnvCfg` | `ContactSweepClutterFusionV1_1EnvCfg` | [Contact Sweep Clutter Fusion 계열](06_fusion.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_4` | `ContactSweepClutterFusionV1_4EnvCfg` | `ContactSweepClutterFusionV1_0EnvCfg` | [Contact Sweep Clutter Fusion 계열](06_fusion.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_3` | `ContactSweepClutterFusionV1_3EnvCfg` | `ContactSweepClutterFusionV1_1EnvCfg` | [Contact Sweep Clutter Fusion 계열](06_fusion.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_3_1` | `ContactSweepClutterFusionV1_3_1EnvCfg` | `ContactSweepClutterFusionV1_3EnvCfg` | [Contact Sweep Clutter Fusion 계열](06_fusion.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_1` | `ContactSweepClutterFusionV1_1EnvCfg` | `ContactSweepClutterFusionV1_0EnvCfg` | [Contact Sweep Clutter Fusion 계열](06_fusion.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_2` | `ContactSweepClutterFusionV1_2EnvCfg` | `ContactSweepClutterFusionV1_1EnvCfg` | [Contact Sweep Clutter Fusion 계열](06_fusion.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_0` | `ContactSweepClutterFusionV1_0EnvCfg` | `ContactSweepClutterFusionV3EnvCfg` | [Contact Sweep Clutter Fusion 계열](06_fusion.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v2` | `ContactSweepClutterFusionV2EnvCfg` | `ContactSweepClutterFusionEnvCfg` | [Contact Sweep Clutter Fusion 계열](06_fusion.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v3` | `ContactSweepClutterFusionV3EnvCfg` | `ContactSweepClutterFusionV2EnvCfg` | [Contact Sweep Clutter Fusion 계열](06_fusion.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v0` | `ContactSweepClutterFusionEnvCfg` | `ManagerBasedRLEnvCfg` | [Contact Sweep Clutter Fusion 계열](06_fusion.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-v0` | `FusionHoldingV0EnvCfg` | `ContactSweepClutterFusionV1_0EnvCfg` | [Holding 계열](07_holding.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-v1` | `FusionHoldingV1EnvCfg` | `FusionHoldingV0EnvCfg` | [Holding 계열](07_holding.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-v2` | `FusionHoldingV2EnvCfg` | `FusionHoldingV1EnvCfg` | [Holding 계열](07_holding.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-v3` | `FusionHoldingV3EnvCfg` | `FusionHoldingV2EnvCfg` | [Holding 계열](07_holding.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-v4` | `FusionHoldingV4EnvCfg` | `FusionHoldingV3EnvCfg` | [Holding 계열](07_holding.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-OSC-v0` | `FusionHoldingOSCV0EnvCfg` | `FusionHoldingV0EnvCfg` | [Holding OSC 계열](08_holding_osc.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-OSC-v1` | `FusionHoldingOSCV1EnvCfg` | `FusionHoldingV1EnvCfg` | [Holding OSC 계열](08_holding_osc.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-OSC-v2` | `FusionHoldingOSCV2EnvCfg` | `FusionHoldingV2EnvCfg` | [Holding OSC 계열](08_holding_osc.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-OSC-v3` | `FusionHoldingOSCV3EnvCfg` | `FusionHoldingV3EnvCfg` | [Holding OSC 계열](08_holding_osc.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-OSC-v4` | `FusionHoldingOSCV4EnvCfg` | `FusionHoldingV4EnvCfg` | [Holding OSC 계열](08_holding_osc.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-OSC-ObjectRandom-v0` | `FusionHoldingOSCObjectRandomV0EnvCfg` | `FusionHoldingOSCV0EnvCfg` | [Holding ObjectRandom 계열](09_object_random.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-OSC-ObjectRandom-v1` | `FusionHoldingOSCObjectRandomV1EnvCfg` | `FusionHoldingOSCV2EnvCfg` | [Holding ObjectRandom 계열](09_object_random.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-ObjectRandom-Test-v0` | `FusionHoldingObjectRandomTestV0EnvCfg` | `FusionHoldingV2EnvCfg` | [Holding ObjectRandom 계열](09_object_random.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-OSC-ObjectRandom-v2` | `FusionHoldingOSCObjectRandomV2EnvCfg` | `FusionHoldingOSCV3EnvCfg` | [Holding ObjectRandom 계열](09_object_random.md) |
| `Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-OSC-ObjectRandom-v3` | `FusionHoldingOSCObjectRandomV3EnvCfg` | `FusionHoldingOSCV4EnvCfg` | [Holding ObjectRandom 계열](09_object_random.md) |
| `Isaac-Inspire-Final-v0` | `FinalEnvCfg` | `ManagerBasedRLEnvCfg` | [Final 계열](10_final.md) |
| `Isaac-Inspire-Final-v1` | `FinalV1EnvCfg` | `FinalEnvCfg` | [Final 계열](10_final.md) |
| `Isaac-Inspire-Final-v2` | `FinalV2EnvCfg` | `FinalV1EnvCfg` | [Final 계열](10_final.md) |
| `Isaac-Inspire-Final-v1-1` | `FinalV1_1EnvCfg` | `FinalV1EnvCfg` | [Final 계열](10_final.md) |
| `Isaac-Inspire-Final-Test-v1` | `FinalV1TestEnvCfg` | `FinalV1EnvCfg` | [Final 계열](10_final.md) |
| `Isaac-Inspire-Final-Test-v2` | `FinalV2TestEnvCfg` | `FinalV2EnvCfg` | [Final 계열](10_final.md) |
| `Isaac-Inspire-Final-GoalAccuracy-v0` | `FinalGoalAccuracyEnvCfg` | `FinalV1EnvCfg` | [Final GoalAccuracy 계열](11_goal_accuracy.md) |
| `Isaac-Inspire-Final-GoalAccuracy-v1` | `FinalGoalAccuracyV1EnvCfg` | `FinalGoalAccuracyEnvCfg` | [Final GoalAccuracy 계열](11_goal_accuracy.md) |
| `Isaac-Inspire-Final-GoalAccuracy-v0_1` | `FinalGoalAccuracyV0_1EnvCfg` | `FinalGoalAccuracyEnvCfg` | [Final GoalAccuracy 계열](11_goal_accuracy.md) |
| `Isaac-Inspire-Final-GoalAccuracy-v0_2` | `FinalGoalAccuracyV0_2EnvCfg` | `FinalGoalAccuracyV0_1EnvCfg` | [Final GoalAccuracy 계열](11_goal_accuracy.md) |
| `Isaac-Inspire-Final-GoalAccuracy-v0_3` | `FinalGoalAccuracyV0_3EnvCfg` | `FinalGoalAccuracyV0_2EnvCfg` | [Final GoalAccuracy 계열](11_goal_accuracy.md) |
| `Isaac-Inspire-Final-GoalAccuracy-v0_2_1` | `FinalGoalAccuracyV0_2_1EnvCfg` | `FinalGoalAccuracyV0_2EnvCfg` | [Final GoalAccuracy 계열](11_goal_accuracy.md) |
| `Isaac-Inspire-Final-GoalAccuracy-v0_2_2` | `FinalGoalAccuracyV0_2_2EnvCfg` | `FinalGoalAccuracyV0_2_1EnvCfg` | [Final GoalAccuracy 계열](11_goal_accuracy.md) |
| `Isaac-Inspire-Revision-v0` | `RevisionEnvCfg` | `ManagerBasedRLEnvCfg` | [Revision 계열](12_revision.md) |
| `Isaac-Inspire-Revision-v1` | `RevisionV1EnvCfg` | `RevisionEnvCfg` | [Revision 계열](12_revision.md) |
| `Isaac-Inspire-Revision-Test-v1` | `RevisionV1TestEnvCfg` | `RevisionV1EnvCfg` | [Revision 계열](12_revision.md) |
| `Isaac-Inspire-Revision-v2` | `RevisionV2EnvCfg` | `RevisionV1EnvCfg` | [Revision 계열](12_revision.md) |
| `Isaac-Inspire-Revision-Test-v2` | `RevisionV2TestEnvCfg` | `RevisionV2EnvCfg` | [Revision 계열](12_revision.md) |
| `Isaac-Inspire-Revision-Test-v2-GT` | `RevisionV2TestGTEnvCfg` | `RevisionV2TestEnvCfg` | [Revision 계열](12_revision.md) |
| `Inspire-Wrench-v0` | `WrenchEnvCfg` | `ManagerBasedRLEnvCfg` | [Wrench 계열](13_wrench.md) |
| `Inspire-Wrench-v0_1` | `WrenchV0_1EnvCfg` | `WrenchEnvCfg` | [Wrench 계열](13_wrench.md) |
| `Inspire-Wrench-v1` | `WrenchV1EnvCfg` | `WrenchEnvCfg` | [Wrench 계열](13_wrench.md) |
| `Inspire-Wrench-v2` | `WrenchV2EnvCfg` | `WrenchV1EnvCfg` | [Wrench 계열](13_wrench.md) |
| `Inspire-Wrench-v0_2` | `WrenchV0_2EnvCfg` | `WrenchEnvCfg` | [Wrench 계열](13_wrench.md) |

## 코드와 상세 기록

구성은 [envs](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs), 장면은 [assets/scenes](../../src/inspire_sweep_baseline/inspire_sweep_baseline/assets/scenes), 관측·보상·종료·일정은 [mdp](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp), 입력 결합과 PPO는 [agent](../../src/inspire_sweep_baseline/inspire_sweep_baseline/agent), 실행 동작은 [runtime](../../src/inspire_sweep_baseline/inspire_sweep_baseline/runtime)에 있다.

학습 진입점은 [train.py](../../src/inspire_sweep_baseline/scripts/train.py)다. 환경별 상세 구현 계약·기존 검증 기록은 [문서 저장소의 환경 문서](../../example/Contact-Rich-Manipulation-Context/02_implementation/environments), 최신 변경 요약은 [현재 인계](../../example/Contact-Rich-Manipulation-Context/01_plan/00_HANDOFF_BRIEF.md)를 참고한다. 이 요약의 기준일 이후에는 실제 코드를 우선한다.
