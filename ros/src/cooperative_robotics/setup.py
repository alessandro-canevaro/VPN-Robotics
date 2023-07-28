import os
from glob import glob
from setuptools import setup

package_name = 'cooperative_robotics'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    	(os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alessandro',
    maintainer_email='alessandro.canevaro@huawei.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ctrl_node = cooperative_robotics.ctrl_node:main',
            'manager_node = cooperative_robotics.manager_node:main',
            'noisy_odom = cooperative_robotics.noisy_odom:main'
        ],
    },
)
