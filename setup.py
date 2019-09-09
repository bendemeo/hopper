from setuptools import setup, find_packages

setup(
    name='treehopper',
    version='0.0',
    description='Principled data sketching',
    url='https://github.com/brianhie/treehopper',
    #download_url='https://github.com/brianhie/geosketch/archive/v1.0-beta.tar.gz',
    packages=find_packages(exclude=['bin', 'conf', 'data', 'target', 'R']),
    install_requires=[
        'numpy>=1.12.0',
        'scipy>=1.3.0',
        'scikit-learn>=0.20rc1',
    ],
)
