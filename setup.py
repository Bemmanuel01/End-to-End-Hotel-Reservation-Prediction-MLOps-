from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()
    
setup(
    name = "HR-AXIOMS",
    version = "0.1",
    author = "Manuell",
    packages = find_packages(),
    install_requires = requirements
)
    