from setuptools import find_packages, setup

package_name = 'perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='helenlu66@gmail.com',
    description='Perception nodes for scene grounding and tactile classification.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
            'external_cam_node = perception.vision_node:main',
            'tactile_node = perception.tactile_node:main',
            'kinova_rtsp_bridge = perception.kinova_rtsp_bridge:main',
            'kinova_wrist_cam_node = perception.kinova_wrist_cam_node:main',
    ]},
)
