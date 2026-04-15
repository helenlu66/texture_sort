from setuptools import setup

package_name = 'texture_sorting_navigation'

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
    maintainer_email='user@example.com',
    description='Navigation/delivery node for TurtleBot delivery cycles via Nav2.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
            "delivery_node = texture_sorting_navigation.delivery_node:main"
    ]},
)
