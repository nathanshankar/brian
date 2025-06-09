# <your_ros2_workspace>/src/brian_rl_env/setup.py
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'brian_rl_env'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Add launch files
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][y]'))),
        # Add scripts
        (os.path.join('share', package_name, 'scripts'), glob(os.path.join('scripts', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train_brian = brian_rl_env.train_brian:main', # Make train_brian.py runnable
            'brian_gym_env = brian_rl_env.brian_gym_env:main', # If you want to run env directly for debug
        ],
    },
)