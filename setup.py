from setuptools import setup, find_packages

setup(
    name="tdcr_torch",
    version="1.0.0",
    author="Navid Feizi",
    description="Tendon-Driven Continuum Robot Cosserat Model (PyTorch)",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    install_requires=["numpy", "torch", "scipy"],
)
