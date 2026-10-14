import dashboard_client

ROBOT_IP = "192.168.56.101"

dashboard = dashboard_client.DashboardClient(ROBOT_IP)

try:
    dashboard.connect()
    print(dashboard.polyscopeVersion())
finally:
    dashboard.disconnect()
