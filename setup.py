from setuptools import setup, find_packages

setup(
    name="ai-linker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "httpx>=0.25.0",
        "pydantic>=2.0.0",
        "rich>=13.0.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "ai-linker=ai_linker.cli.main:main",
        ],
    },
)
