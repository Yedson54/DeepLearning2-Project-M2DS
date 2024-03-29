from setuptools import setup, find_packages

setup(
    name='generative_model',
    version='0.1',
    author='Yedidia AGNIMO & C. Yann Éric CHOHO',
    author_email='yedidia.agnimo@ensae.fr // chohoyanneric.choho@ensae.fr',
    description='Implement basics deep learning architecture for generative models.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.10',
)
