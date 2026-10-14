"""Install the self-contained Isaac Lab blind-sweeping task."""

from setuptools import find_packages, setup


setup(
    name="hand-manipulation-rl",
    version="0.1.0",
    description="Blind bilateral-tactile sweeping environment for Isaac Lab",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "hand_manipulation_rl": [
            "assets/data/ur5e_inspire_usd/*",
            "assets/data/ur5e_inspire_usd/configuration/*",
        ]
    },
    python_requires=">=3.10",
    zip_safe=False,
)

