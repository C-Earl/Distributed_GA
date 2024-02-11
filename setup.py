from setuptools import setup, find_packages

setup(name='DGA',
      version='0.0.1',
      author='Christopher T. Earl',
      author_email='cearl@umass.edu',
      url='https://github.com/cearlUmass/Distributed_GA',
      packages=find_packages(),
      license="MIT",
      install_requires=['numpy', 'portalocker', 'jsbeautifier', 'matplotlib', 'PySide2'],
      python_requires=">=3.7.0"
)