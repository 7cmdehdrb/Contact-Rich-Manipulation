"""Install the Sweep-Policy / Inspire Hand shelf-sweeping task."""

from setuptools import find_packages, setup


setup(
    name="sweep-inspire-rl",
    version="0.1.0",
    description="Right-only shelf sweeping with UR5e, Axia80 and Inspire Hand",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["sweeping-policy>=0.1.0", "hand-manipulation-rl>=0.1.0"],
    python_requires=">=3.10",
    zip_safe=False,
)
