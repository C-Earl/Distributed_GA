# Introduction
Distributed Genetic Algorithms (DGA) is a Python based library that will run a highly paralellized genetic algorithm. Given you have the appropriate hardware, the paralellization should net a boost in performance.

DGA is designed to be highly generalizable, and can run *any* model with a quantifiable 'fitness'. You may also write your own genetic algorithm scripts.

### Important term:
* For this project, '**Gene**' is synonymous with '**Parameters**'. When discussing the use of genes, assume it is about the use of parameters for a model.

### IDE Note:
**VScode is the intended IDE for running distributed GA**. Should still be compatible with other IDE's, but currently only the .vscode directory is setup.

**To run on other IDEs** you will need to set the PYTHONPATH environment variable to the DGA base directory on your local machine ("~/path/to/Distributed_GA"). Setting this up will depend on your IDE, but most modern IDE's should have a way to do this.

### Hardware Note:
Your performance will be higher if you have more cores. While you can set as many parallel processes to run as you like, note that performance will only increase if you have at least the same number of *idle* cores as processes.

**Example**: I have a 16 core machine, 1 of which is mostly utilized by the OS. I would reccomend setting no more than 15 parallel processes in this case

# Installation
1. Clone repository
2. Create a virtual environment for the project. It's reccomended to use VScode to do this for simplicity. Simply press ```ctrl+shift+P```, search for "Create Environment" and select "Python: Create Environment". From there select 'venv', a 3.11+ Python interpreter, and ensure that requirements.txt is checked before hitting OK (requirements.txt will install all python package requirements). 

NOTE: If you are NOT using VSCode, now is the time to set your PYTYHONPATH to ```./Distributed_GA```
Creating a virtual environment through other IDEs should be similar to the VSCode process above.

# How to run example
To see examples, navigate to the ```scripts``` folder. 
- ```GA_examples```: Optimizes a vector matching task. One version ("Simple_GA_Example.py") is a generic genetic algorithm while the other ("Complex_GA_Example.py") is a more sophisticated algorithm called the 'Plateau' algorithm
- ```ANN_example```: Optimizes simple Artificial Neural Net for MNIST classification. You will need to install [PyTorch](https://pytorch.org/) on your system to run this model.
- ```SLURM_example```: Example of DGA utilizing the SLURM operating system. Works by calling nodes to test genes on a model in parallel.

I reccomend starting with the simplest example in ```GA_examples```. Open the file "Simple_GA_Example.py" and simply run the file to start the algorithm. 

To see if the algorithm is running, open your process manager. While the algorithm is running, you should see ```num_parallel_processes``` Python processes in your manager (you can find this arg. value near the bottom of the file). These are the parallel processes testing the example model and generated genes. They will automatically call new processes to test the next generation of genes until the end condition is met (it will test ```iterations``` number of genes). **You will know the run has finished when all the Python processes have disappeared from your process manager.**

To check the results, run the ```Evaluator.py``` file, which will print out the gene name next to that genes fitness. You should see ```num_genes``` total genes printed out in this format:

```bash
Gene: fc4843f1f6...             Fitness: -0.8420308346282214
Gene: dh197s7a13...             Fitness: -0.9187269126031982
Gene: nakx8461s4...             Fitness: -0.8781020980398019
Gene: ka81ntb29d...             Fitness: -0.7701928739009892
Gene: 18aksbnais...             Fitness: -0.7019288379001982
...
```

Your final values may look a bit as there is some stochasticity

# Additional Documentation
**code_docs.pdf**
>Contains in depth info about Server, Model, and Algorithm objects and how they are used. See here for any info about how to use the libraries various objects two pre-packaged algorithms

**backend_docs.md**
>A more in depth look at how to program your own genetic algorithms and understanding how the Server executes.
