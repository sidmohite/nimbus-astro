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
    packages=find_packages(where='nimbus'),
    scripts=['nimbus/singlefield_calc','nimbus/compute_field_probs',\
           'nimbus/combine_fields'],
    python_requires='>=3.6, <4',
    # Needed for dependencies
    install_requires=['numpy','scipy','astropy','pandas','healpy'],
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
