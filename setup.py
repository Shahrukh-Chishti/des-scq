from setuptools import setup,find_packages
from pathlib import Path

root = Path(__file__).parent

def load_requirements(path):
    with open(path) as f:
        return f.read().splitlines()

setup(name='Des-Scq',
        author='Shahrukh Chishti, Carla Illmann',
        version="0.0.1",
        python_requires=">=3.11",
        packages=find_packages(),
        install_requires=load_requirements(root / "requirements.txt"),
        author_email="shahrukh.chishti@gmail.com",
        description="Optimization library to design Superconducting circuits",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        url="https://github.com/Shahrukh-Chishti/des-scq"
)
