from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'foundationpose_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ehsanullahm1',
    maintainer_email='iamehsan10@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'foundationpose_service_node = foundationpose_bridge.foundationpose_service_node:main',
            'mesh_service_client_example = foundationpose_bridge.mesh_service_client_example:main',
        ],
    },
)
