from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='income prediction',
    author='sanjeevan thorat',
    license='MIT',
    dependency_links=['https://github.com/sanjeevan121/income-pred.git@246cdf41359cf4d0982ecb2ed6a6e82a572bf58b#egg=src']
)
