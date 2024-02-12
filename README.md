# Introduction
Distributed Genetic Algorithms (DGA) is a Python based library that runs paralellized, user-customizable genetic algorithms. Given you have the appropriate hardware, the paralellization should net a boost in performance.

DGA is designed to be highly generalizable, and can run theoretically train any model with a quantifiable 'fitness'.

### IDE Note:
**VScode is the intended IDE for running distributed GA**. Should still be compatible with other IDE's, but currently only the .vscode directory is setup.

**To run on other IDEs** you will need to set the PYTHONPATH environment variable to the DGA base directory on your local machine ("~/path/to/Distributed_GA"). Setting this up will depend on your IDE, but most modern IDE's should have a way to do this.

# Installation
1. Clone repository
2. Select a Python interpreter (3.10+ recommended)
3. Build the package by running the following command in the terminal:
```bash
 python3 setup.py sdist bdist_wheel
```
4. Install the package by running the following command in the terminal:
```bash
pip install .
```

# How to run example
To see examples, navigate to the ```scripts\Vector Estimate Benchmark``` folder. Here you will find 2 files, ```vector_estimator.py``` and ```vector_estimator_(parallel).py```. The former runs a single-threaded genetic algorithm, while the latter runs a multi-threaded genetic algorithm. Note that the multi-threaded version will be less efficient, and is only included for testing purposes.

To run either, simply run the files from the terminal like this:
```bash
python3 vector_estimator.py
```

# Additional Documentation
**code_docs.pdf**
>Contains in depth info about Server, Model, and Algorithm objects and how they are used. See here for any info about how to use the libraries various objects two pre-packaged algorithms

**backend_docs.md**
>A more in depth look at how to program your own genetic algorithms and understanding how the Server executes.
