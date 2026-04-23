from setuptools import setup

package_name = 'perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
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
            "vision_node = perception.vision_node:main",
            "tactile_node = perception.tactile_node:main",
            "kinova_rtsp_bridge = perception.kinova_rtsp_bridge:main"
    ]},
)
