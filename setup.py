from setuptools import find_packages, setup

setup(
    name='esun-ai-2022-summer',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'absl-py==1.0.0',
        'scipy==1.4.1',
        'gunicorn==20.1.0',
        'Flask==2.0.3',
        'Flask-Caching==1.11.1',
        'redis==3.5.3',
        'torch==1.11.0',
        'transformers==4.18.0'
    ],
)
