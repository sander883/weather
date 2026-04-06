"""Setup script for Polymarket Weather Trading Agent."""

from setuptools import setup, find_packages

setup(
    name='polymarket-weather-agent',
    version='1.0.0',
    description='Autonomous AI agent for trading weather-based predictions on Polymarket',
    author='Weather Agent Team',
    packages=find_packages(),
    install_requires=[
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'xgboost>=2.0.0',
        'scikit-learn>=1.3.0',
        'requests>=2.31.0',
        'aiohttp>=3.8.0',
        'python-dotenv>=1.0.0',
        'pyyaml>=6.0',
        'flask>=2.3.0',
        'flask-cors>=4.0.0',
        'flask-restx>=0.5.1',
        'sqlalchemy>=2.0.0',
        'web3>=6.11.0',
        'eth-account>=0.9.5',
        'plotly>=5.16.0',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'polymarket-agent=src.main:main',
            'polymarket-dashboard=dashboard.app:main',
        ],
    },
)
