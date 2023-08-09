# Introduction
Distributed Genetic Algorithms (DGA) is a Python based module that will run a highly paralellized genetic algorithm. Given you have the appropriate hardware, the paralellization should net a boost in performance from non-paralellized versions. To briefly explain, paralellization is done by sending 'genes' (model parameters) to an asynchronous subprocess for fitness evaluation.

DGA is designed to be modular, meaning it's designed to run *any* model with a quantifiable 'fitness'. You may also write your own genetic algorithm scripts to be deployed asynchronously.

### Important term:
* '**Gene**' is synonymous with '**Parameters**'. When discussing the use of genes, assume it is about the use of parameters for a model.

### IDE Note:
**VScode is the intended IDE for running distributed GA**. Should still be compatible with other IDE's, but currently only the .vscode directory is setup.

**To run on other IDEs** you will need to set the PYTHONPATH environment variable to the base directory ("~/path/to/Distributed_GA"). Setting this up will depend on your IDE.

### Hardware Note:
Your performance will be higher if you have more cores. While you can set as many parallel processes to run as you like, note that performance will only increase if you have at least the same number of *idle* cores as processes.

**Example**: I have a 16 core machine, 1 of which is mostly utilized by the OS. I would reccomend setting no more than 15 parallel processes in this case

# Installation
1. Clone repository
2. Create a virtual environment for the project. It's reccomended to use VScode to do this for simplicity. Simply press ```ctrl+shift+P```, search for "Create Environment" and select "Python: Create Environment". In order, select 'venv', a 3.11+ Python interpreter, and ensure that requirements.txt is checked before hitting OK. 

NOTE: If you are NOT using VSCode, now is the time to set your PYTYHONPATH to ```./Distributed_GA```

# How to run example
To see examples, navigate to the ```scripts``` folder. To run the ```complex_example``` you will need to install [PyTorch](https://pytorch.org/) on your system. 
- ```simple_example```: Optimizes a size 10 array towards the origin (array of all 0's)
- ```complex_example```: Optimizes simple Artificial Neural Net for MNIST classification 

Choose one and open either ```Complex_``` or ```Simple_Example.py```. Run this file to begin the algorithm, but note that the algorithm doesn't end when the starting process finishes... 

To tell if the GA has finished running, check your process manager. While the algorithm is running, you should see ```num_parallel_processes``` (check near the bottom of the file) Python processes in your manager. These are the parallel processes testing the example model, and will 'reproduce' (create new genes & call new processes to test it). **You will know the run has finished when all the Python processes have disappeared from your process manager.**

To check the results, run the ```Evaluator.py``` file, which will print out the gene name next to that genes fitness. You should see ```num_genes``` total genes printed out in this format:

```bash
Gene: fc4843f1f6...             Fitness: -0.8420308346282214
```

# How to run your own models on the base algorithm

# How to write Genetic Algorithms scripts

# Server.py and Server description
