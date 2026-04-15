from setuptools import setup

package_name = 'texture_sorting_task_manager'

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
    description='High-level task manager coordinating perception, manipulation, and delivery.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
            "task_manager_node = texture_sorting_task_manager.task_manager_node:main"
    ]},
)
