import numpy as np
import matplotlib.pyplot as plt


def simulate_pgs(
    x0: float,
    v0: float,
    x_target: float,
    v_target: float,
    mass: float,
    kp: float,
    kd: float,
    dt: float,
    num_iterations: int,
):
    """
    PGS force drive의 position iteration을 계산한다.

    주요 식:
        a = dt * (dt * kp + kd)
        b = dt * kd * v_target
        c = 1 / (1 + a * r)

        lambda_i
        = c * (b + dt*kp*(x_target-x0) - a*v_{i-1})
          + (1-c)*lambda_{i-1}
    """

    r = 1.0 / mass

    a = dt * (dt * kp + kd)
    b = dt * kd * v_target
    c = 1.0 / (1.0 + a * r)

    velocity = v0
    impulse = 0.0

    history = {
        "iteration": [],
        "carried_position": [],
        "predicted_end_position": [],
        "velocity": [],
        "impulse": [],
        "delta_impulse": [],
    }

    for i in range(1, num_iterations + 1):
        new_impulse = (
            c * (b + dt * kp * (x_target - x0) - a * velocity) + (1.0 - c) * impulse
        )

        delta_impulse = new_impulse - impulse

        # Eq. (7): v_i = v_{i-1} + r * delta_lambda_i
        velocity += r * delta_impulse

        # Eq. (8): x_i = x_0 + v_i * dt
        predicted_end_position = x0 + velocity * dt

        impulse = new_impulse

        history["iteration"].append(i)

        # PGS는 iteration 사이에 실제 위치를 전진시키지 않는다.
        history["carried_position"].append(x0)

        history["predicted_end_position"].append(predicted_end_position)
        history["velocity"].append(velocity)
        history["impulse"].append(impulse)
        history["delta_impulse"].append(delta_impulse)

    final_position = x0 + velocity * dt

    return history, final_position, velocity, impulse


def simulate_tgs(
    x0: float,
    v0: float,
    x_target: float,
    v_target: float,
    mass: float,
    kp: float,
    kd: float,
    dt: float,
    num_iterations: int,
):
    """
    TGS force drive의 position iteration을 계산한다.

    주요 식:
        rho = dt / num_iterations

        a = rho * (dt * kp + kd)
        b = rho * kd * v_target
        c = 1 / (1 + a * r)

        delta_lambda_i
        = c * (
            b
            + rho*kp*(x_target_sub-x0)
            - rho*kp*delta_x_{i-1}
            - a*v_{i-1}
        )
    """

    r = 1.0 / mass
    rho = dt / num_iterations

    # 문서의 설명에 따라 PGS 계수에서
    # 바깥쪽 timestep dt를 rho로 치환한다.
    a = rho * (dt * kp + kd)
    b = rho * kd * v_target
    c = 1.0 / (1.0 + a * r)

    velocity = v0
    impulse = 0.0

    # 이전 TGS iteration들에서 누적된 위치 변화
    accumulated_position_change = 0.0

    history = {
        "iteration": [],
        "carried_position": [],
        "predicted_end_position": [],
        "velocity": [],
        "impulse": [],
        "delta_impulse": [],
        "substep_target": [],
    }

    for i in range(1, num_iterations + 1):
        # Eq. (41)
        substep_target = x_target - (dt - i * rho) * v_target

        # Eq. (44)에서 이번 iteration의 impulse 증가량
        delta_impulse = c * (
            b
            + rho * kp * (substep_target - x0)
            - rho * kp * accumulated_position_change
            - a * velocity
        )

        impulse += delta_impulse

        # impulse에 의한 속도 변화
        velocity += r * delta_impulse

        # Eq. (42)의 timestep 끝 예측 위치
        predicted_end_position = x0 + accumulated_position_change + velocity * dt

        # TGS에서는 각 iteration마다 rho만큼 위치를 전진시킨다.
        accumulated_position_change += rho * velocity

        carried_position = x0 + accumulated_position_change

        history["iteration"].append(i)
        history["carried_position"].append(carried_position)
        history["predicted_end_position"].append(predicted_end_position)
        history["velocity"].append(velocity)
        history["impulse"].append(impulse)
        history["delta_impulse"].append(delta_impulse)
        history["substep_target"].append(substep_target)

    final_position = x0 + accumulated_position_change

    return history, final_position, velocity, impulse


def plot_comparison():
    # ============================================================
    # 실험 파라미터
    # ============================================================

    x0 = 0.0
    v0 = 0.0

    x_target = 1.0
    v_target = 0.0

    mass = 1.0

    # 값을 높일수록 PGS와 TGS의 차이가 커진다.
    kp = 5000.0
    kd = 40.0

    dt = 1.0 / 60.0
    num_iterations = 8

    # ============================================================
    # PGS / TGS 계산
    # ============================================================

    pgs, pgs_x, pgs_v, pgs_impulse = simulate_pgs(
        x0=x0,
        v0=v0,
        x_target=x_target,
        v_target=v_target,
        mass=mass,
        kp=kp,
        kd=kd,
        dt=dt,
        num_iterations=num_iterations,
    )

    tgs, tgs_x, tgs_v, tgs_impulse = simulate_tgs(
        x0=x0,
        v0=v0,
        x_target=x_target,
        v_target=v_target,
        mass=mass,
        kp=kp,
        kd=kd,
        dt=dt,
        num_iterations=num_iterations,
    )

    iterations = np.asarray(pgs["iteration"])
    rho = dt / num_iterations

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 8),
        constrained_layout=True,
    )

    # ============================================================
    # 1. timestep 내부에서 실제로 반영되는 position
    # ============================================================

    # TGS는 각 position iteration마다 rho만큼 시간을 전진시킨다.
    tgs_time = np.arange(num_iterations + 1) * rho

    tgs_actual_position = np.concatenate(
        (
            [x0],
            np.asarray(tgs["carried_position"]),
        )
    )

    # PGS는 position iteration 동안 실제 position을 x0로 유지하고,
    # timestep 종료 시점에 최종 velocity를 이용해 한 번에 갱신한다.
    #
    # 동일한 시간 dt를 두 번 넣어서 수직 점프를 명시적으로 그린다.
    pgs_time = np.array(
        [
            0.0,
            dt,
            dt,
        ]
    )

    pgs_actual_position = np.array(
        [
            x0,
            x0,
            pgs_x,
        ]
    )

    axes[0, 0].plot(
        pgs_time,
        pgs_actual_position,
        marker="o",
        label="PGS actual position",
    )

    axes[0, 0].plot(
        tgs_time,
        tgs_actual_position,
        marker="o",
        label="TGS actual position",
    )

    axes[0, 0].axhline(
        x_target,
        linestyle="--",
        label="Target position",
    )

    # timestep 끝의 최종 위치 표시
    axes[0, 0].scatter(
        [dt],
        [pgs_x],
        s=80,
        zorder=5,
    )

    axes[0, 0].scatter(
        [dt],
        [tgs_x],
        s=80,
        zorder=5,
    )

    axes[0, 0].annotate(
        f"PGS final = {pgs_x:.4f}",
        xy=(dt, pgs_x),
        xytext=(-115, 15),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )

    axes[0, 0].annotate(
        f"TGS final = {tgs_x:.4f}",
        xy=(dt, tgs_x),
        xytext=(-115, -30),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )

    axes[0, 0].set_xlim(
        0.0,
        dt * 1.05,
    )

    axes[0, 0].set_title("Actual position update within one timestep")
    axes[0, 0].set_xlabel("Time within timestep [s]")
    axes[0, 0].set_ylabel("Position")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # ============================================================
    # 2. 각 iteration에서 예측한 timestep 끝 position
    # ============================================================

    axes[0, 1].plot(
        iterations,
        pgs["predicted_end_position"],
        marker="o",
        label="PGS predicted end position",
    )

    axes[0, 1].plot(
        iterations,
        tgs["predicted_end_position"],
        marker="o",
        label="TGS predicted end position",
    )

    axes[0, 1].axhline(
        x_target,
        linestyle="--",
        label="Target position",
    )

    axes[0, 1].set_title("Predicted end-of-timestep position")
    axes[0, 1].set_xlabel("Solver iteration")
    axes[0, 1].set_ylabel("Predicted position")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # ============================================================
    # 3. Velocity
    # ============================================================

    axes[1, 0].plot(
        iterations,
        pgs["velocity"],
        marker="o",
        label="PGS velocity",
    )

    axes[1, 0].plot(
        iterations,
        tgs["velocity"],
        marker="o",
        label="TGS velocity",
    )

    axes[1, 0].set_title("Velocity during solver iterations")
    axes[1, 0].set_xlabel("Solver iteration")
    axes[1, 0].set_ylabel("Velocity")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # ============================================================
    # 4. 누적 drive impulse
    # ============================================================

    axes[1, 1].plot(
        iterations,
        pgs["impulse"],
        marker="o",
        label="PGS accumulated impulse",
    )

    axes[1, 1].plot(
        iterations,
        tgs["impulse"],
        marker="o",
        label="TGS accumulated impulse",
    )

    axes[1, 1].set_title("Accumulated drive impulse")
    axes[1, 1].set_xlabel("Solver iteration")
    axes[1, 1].set_ylabel("Impulse")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    # ============================================================
    # 전체 제목 및 출력
    # ============================================================

    fig.suptitle(
        (
            "PGS vs TGS force drive\n"
            f"kp={kp}, kd={kd}, "
            f"dt={dt:.5f} s, "
            f"iterations={num_iterations}"
        )
    )

    plt.show()

    print("===== Final result =====")

    print(
        f"PGS: position={pgs_x:.6f}, "
        f"velocity={pgs_v:.6f}, "
        f"impulse={pgs_impulse:.6f}"
    )

    print(
        f"TGS: position={tgs_x:.6f}, "
        f"velocity={tgs_v:.6f}, "
        f"impulse={tgs_impulse:.6f}"
    )

    print(f"Position difference: " f"{tgs_x - pgs_x:.6f}")

    print(f"Velocity difference: " f"{tgs_v - pgs_v:.6f}")

    print(f"Impulse difference: " f"{tgs_impulse - pgs_impulse:.6f}")


if __name__ == "__main__":
    plot_comparison()
