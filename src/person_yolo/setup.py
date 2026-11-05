import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'person_yolo'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    package_data={package_name: ['web/*.html']},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='matias',
    maintainer_email='ibanezignacio2005@gmail.com',
    description='Deteccion de personas con YOLO + GPS + recorrido del dron.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'person_detector = person_yolo.person_detector:main',
            'gps_node = person_yolo.gps_node:main',
            'trail_node = person_yolo.trail_node:main',
            'camera_node = person_yolo.camera_node:main',
            'fpv_node = person_yolo.fpv_node:main',
        ],
    },
)
