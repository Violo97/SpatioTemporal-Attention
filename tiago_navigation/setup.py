from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# Fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['tiago_navigation'],
    package_dir={'': 'src'},
    #scripts=['script/test.py ' , 'src/Qlearning/prova.py']
)

setup(**setup_args)