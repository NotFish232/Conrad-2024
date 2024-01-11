# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='xarm',
    version='0.1.0',
    description='Lobot xArm Controller',
    long_description=readme,
    author='Chris Courson',
    author_email='chris@chrisbot.com',
    url='https://github.com/ccourson/xArmServoController/tree/master/PC/Python/xarm',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'old')),
    install_requires=['pywinusb']
)
