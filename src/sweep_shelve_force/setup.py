from setuptools import find_packages, setup


setup(
    name="sweep-shelve-force",
    version="0.1.0",
    description="Independent Isaac Lab shelf-sweep task with articulation-joint F/T sensing",
    packages=find_packages(),
    python_requires=">=3.10",
    zip_safe=False,
)
