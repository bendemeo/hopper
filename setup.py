from setuptools import setup
setup(
  name = 'hopper', # the name of your package
  version = '0.0.1', # version number
  description = 'Hopper is a tool for sampling highly-informative cells from single-cell datasets', # short description
  author = 'Benjamin Demeo', # your name
  author_email = 'bdemeo@mit.edu', # your email
  long_description_content_type="text/markdown",
  license="MIT",
  keywords="hopper treehopper single-cell transcriptome transcriptomics",
  zip_safe=False,
  packages=['hopper'],
  url="https://github.com/bendemeo/hopper",
  classifiers=[
      "Programming Language :: Python :: 3 :: Only",
      "Programming Language :: Python :: 3",
      "Programming Language :: Python :: 3.7",
      "License :: OSI Approved :: MIT License",
      "Topic :: Scientific/Engineering :: Bio-Informatics"
  ],
  python_requires=">=3.6",
  install_requires=[
    "scanpy>=1.4.6",
    "scipy==1.4.1",
    "matplotlib==3.2.1",
    "numpy==1.18.1",
    "fbpca==1.0",
    "scikit_learn==0.23.1",
    "python-igraph",
    "leidenalg"
  ]
)