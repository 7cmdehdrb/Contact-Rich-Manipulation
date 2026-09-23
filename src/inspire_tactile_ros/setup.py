from setuptools import setup

setup(
    name="inspire_tactile_ros", version="0.1.0", packages=["inspire_tactile_ros"],
    data_files=[("share/ament_index/resource_index/packages", ["resource/inspire_tactile_ros"]),
                ("share/inspire_tactile_ros", ["package.xml"])],
    install_requires=["setuptools"], zip_safe=True,
    maintainer="min", maintainer_email="min@example.com", license="Apache-2.0",
    description="ROS 2 relay and viewer for Inspire tactile sensors",
    entry_points={"console_scripts": [
        "bridge = inspire_tactile_ros.bridge:main", "viewer = inspire_tactile_ros.viewer:main",
    ]},
)
