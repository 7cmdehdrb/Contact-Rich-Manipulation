import rtde_control
import rtde_receive

ROBOT_IP = "192.168.56.101"
FREQUENCY = 500.0

rtde_c = rtde_control.RTDEControlInterface(
    ROBOT_IP,
    FREQUENCY,
)
rtde_r = rtde_receive.RTDEReceiveInterface(
    ROBOT_IP,
    FREQUENCY,
)

count = 0

try:
    while True:
        cycle_start = rtde_c.initPeriod()

        tcp_pose = rtde_r.getActualTCPPose()
        print("TCP pose:", tcp_pose)

        # tau_cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # success = rtde_c.moveJ(tau_cmd, True)
        # if not success:
        #     print("Failed to transmit torque command")
        #     raise RuntimeError("directTorque command transmission failed")

        # if count % 500 == 0:
        #     print("Torque command transmitted:", tau_cmd)

        # count += 1
        rtde_c.waitPeriod(cycle_start)

except KeyboardInterrupt:
    print("Keyboard interrupt detected")
    rtde_c.stopJ(10.0)
    rtde_c.stopScript()


"""

python3 src/test/rtde_osc_velocity.py \
  --robot-ip 192.168.56.101 \
  --target -0.524556 -0.441240 0.423292 0.124174  2.810813 -0.793706 \
  --duration 5 \
  --execute

"""
