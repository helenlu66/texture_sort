from setuptools import setup

package_name = 'manipulation'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['pre_grasp.json']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='helenlu66@gmail.com',
    description='Manipulation action servers for grasp, place, and load operations.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
            "manipulation_node = manipulation.manipulation_node:main"
    ]},
)
