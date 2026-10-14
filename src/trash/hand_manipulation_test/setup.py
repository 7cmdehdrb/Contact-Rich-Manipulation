"""Install the standalone UR5e–Inspire reaching and contact environments."""

from setuptools import find_packages, setup


setup(
    name="hand-manipulation-test",
    version="0.1.0",
    description="Manager-based UR5e–Inspire reaching and palm contact for Isaac Lab",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "hand_manipulation_test": [
            "assets/data/*.usd",
            "assets/data/ur5e_inspire_usd/*",
            "assets/data/ur5e_inspire_usd/configuration/*",
        ]
    },
    python_requires=">=3.10",
    zip_safe=False,
)
