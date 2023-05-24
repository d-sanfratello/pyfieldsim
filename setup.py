import numpy as np
from setuptools import setup
from codecs import open

try:
    import cpnest
except ImportError:
    raise Exception(
        "This package needs `cpnest`. To install it follow instructions at"
        "https://github.com/johnveitch/cpnest/tree/massively_parallel."
    )

# try:
#     import figaro
# except ImportError:
#     raise Exception(
#         "This package needs `figaro`. To install it follow instructions at"
#         "https://github.com/sterinaldi/figaro."
#     )


with open("requirements.txt") as requires_file:
    requirements = requires_file.read().split("\n")

scripts = [
    'fs-help=pyfieldsim.pipelines.help:main',
    'fs-initialize=pyfieldsim.pipelines.initialize_field:main',
    'fs-observation=pyfieldsim.pipelines.simulate_observation:main',
]
pymodules = [
    'pyfieldsim/pipelines/help',
    'pyfieldsim/pipelines/initialize_field',
    'pyfieldsim/pipelines/simulate_observation',
]

setup(
    name='pyfieldsim',
    use_scm_version=True,
    description='A package to simulate field observations with a CCD.',
    author='Daniele Sanfratello',
    author_email='d.sanfratello@studenti.unipi.it',
    url='https://github.com/d-sanfratello/pyfieldsim',
    python_requires='~=3.8.15',
    packages=['pyfieldsim'],
    install_requires=requirements,
    include_dirs=[np.get_include()],
    setup_requires=['numpy~=1.21.5', 'setuptools_scm'],
    py_modules=pymodules,
    entry_points={
        'console_scripts': scripts
    },
)
