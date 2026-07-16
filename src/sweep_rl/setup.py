"""Editable installation for the Sweep RL Isaac Lab task package."""

from setuptools import find_packages, setup

setup(
    name="sweep-rl",
    version="0.2.0",
    description="Manager-based UR5e sweep reinforcement-learning tasks for Isaac Lab",
    packages=find_packages(),
    python_requires=">=3.10",
    zip_safe=False,
)
