from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='nimbus',
    url='https://github.com/sidmohite/nimbus-astro',
    author='Siddharth Mohite',
    author_email='srmohite@uwm.edu',
    # Needed to actually package something
    packages=find_packages(include=['nimbus','nimbus.*']),
    entry_points={
        "console_scripts": [
            "singlefield_calc=nimbus.scripts.singlefield_calc:main",
            "compute_field_probs=nimbus.scripts.compute_field_probs:main",
            "combine_fields=nimbus.scripts.combine_fields:main",
        ],
    },
    python_requires='>=3.7, <4',
    # Needed for dependencies
    install_requires=['numpy','scipy','astropy','pandas','healpy','pytest'],
    # *strongly* suggested for sharing
    version='1.0.0',
    # The license can be anything you like
    license='MIT',
    description='A hierarchical Bayesian inference framework to constrain\
kilonova population properties.',
    # We will also need a readme eventually (there will be a warning)
    long_description=long_description,
    long_description_content_type='text/markdown',
)
