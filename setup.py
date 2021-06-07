from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='nimbus',
    url='https://github.com/sidmohite/nimbus-astro',
    author='Siddharth Mohite',
    author_email='srmohite@uwm.edu',
    # Needed to actually package something
    packages=['nimbus'],
    # Needed for dependencies
    install_requires=['numpy'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='A hierarchical Bayesian inference framework to 
        constrain kilonova population properties.',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
