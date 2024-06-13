from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="rssgen",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    author="X",
    author_email="X",
    description="Reasoning Shortcuts dataset generator",
    url="https://github.com",
    entry_points={
        "console_scripts": [
            "rssgen=rssgen.__main__:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Machine Learning",
        "Topic :: Scientific/Engineering :: Neuro-Symbolic AI",
        "Topic :: Scientific/Engineering :: Reasoning Shortcuts",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
    ],
)
