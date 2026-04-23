from setuptools import setup

package_name = 'bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/system_bringup.launch.py']),
        ('share/' + package_name + '/config', ['config/system_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='helenlu66@gmail.com',
    description='Launch and configuration package for the texture sorting system.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
            
    ]},
)
