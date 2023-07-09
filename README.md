# Introduction
Distributed Genetic Algorithms (DGA) is a Python based module that will run a highly paralellized genetic algorithm. Given you have the appropriate hardware, the paralellization should net a boost in performance from non-paralellized versions. To briefly explain, paralellization is done by sending 'genes' (model parameters) to an asynchronous subprocess for fitness evaluation.

DGA is designed to be modular, meaning it's designed to run *any* model with a quantifiable 'fitness'. You may also write your own genetic algorithm scripts to be deployed asynchronously.

### IDE Note:
**VScode is the intended IDE for running distributed GA**. Should still be compatible with other IDE's, but currently only the .vscode directory is setup.

**To run on other IDEs** you will need to set the PYTHONPATH environment variable to the base directory ("~/path/to/Distributed_GA"). Setting this up will depend on your IDE.

### Hardware Note:
If running on a single machine (stuff from the base DGA module), your performance will be higher if you have more cores. While you can set as many subprocesses to run as you like, note that performance will only increase if you have at least the same number of *idle* cores as subprocesses.

Example: I have a 16 core machine, 1 of which is mostly utilized by the OS. I would reccomend setting 15 subprocesses in this case

# Installation
1. Clone repository
2. Navigate to the base directory ```./Distributed_GA```
3. Install packages in ```requirements.txt``` using the command ```pip install -r requirements.txt```

NOTE: If you are NOT using VSCode, now is the time to set your PYTYHONPATH to ```./Distributed_GA```

# How to run example
This example will show you how to run a distributed genetic algorithm on your local machine using DGA module. The example script works by calling asynchronous subprocesses to test a very basic model. 

Navigate to ```./Distributed_GA/scripts/local_example/Example.py``` and open the file in vscode. To run the script, use the run button on the IDE. Alternatively, you can navigate to the same location in the VScode terminal and run ```python3 Example.py```. 

To tell if the genetic algorithm has finished running, check your process manager. While it's running, you should see 5 Python processes. These are the subprocesses testing the example model, and will 'reproduce' (call another subprocess) until the max iterations. The run has finished when all subprocess's have disappeared from your process manager.

Soon after you start, you should see a new directory called ```example_run_name```. This is where all the data (genes, fitnesses, etc.) are stored. To see the result of your run, run the ```Evaluate.py``` file next to ```Example.py```. This will print the current state of the gene pool (should be 10 genes). Outputs should look something like this:
```bash
Gene: fc4843f1f6...             Fitness: -0.8420308346282214		# gene name is hashed gene
``` 

# How to run your own models and algorithms
**Use Example.py as a template**, it has inline comments for everything covered here

1. Navigate to ```./Distributed_GA/scripts```. This is where all your personal projects should go.
2. Choose a name for your project, and make a folder with that name.
3. Choose a name for the file that will contain your genetic algorithm and model code. Create a python file with that name in your scripts folder. Your directory should look like this now:

```
├── Distributed_GA
|		...
│   └── scripts
│   	└── Project_Name
│   		└── Project_code.py
|		...
```

4. Like you see in ```Example.py```, add the following imports:
```python
from Algorithm import Algorithm
from Client import Client
from Server import Server
```
3. See the Example.py file for details on how to test your own models or implement your own algorithm.
4. Once you have written your classes, create an ```if __name__ == '__main__':``` with a single line that creates a Server object. The arguments will give the Algorithm and Client you wrote to the Server:
* ```algorithm_path (str)``` = *absolute path* of the python file containing your Algorithm class
* ```algorithm_name (str)``` = name you gave your Algorithm class
* ```client_path (str)``` = *absolute path* of the python file containing your Client class
* ```client_name (str)``` = name you gave your Client class
* ```num_parallel_processes (int)``` = # of genes being tested in parallel
* ```gene_shape (tuple)``` = shape of genes
* ```num_genes (int)``` = max # of genes stored in gene pool 
```python
if __name__ == '__main__':
  Server(run_name="test_dir", 
		 algorithm_path="Example", algorithm_name="Simple_GA",
		 client_path="Example", client_name="Simple_GA_Client",
		 num_clients=5, gene_shape=(10,), num_genes=10,arguments
		 mutation_rate=0.1, iterations=20)
```
5. Run the file. This should start the genetic algorithm.

# How to write Genetic Algorithms scripts
Writing genetic algorithms for this library will be slightly different than how you may normally write them. There are two important details: 
1. When your code will be executed 
2. How your code will be executed

When: Your code will be executed according to this flow of events. After initialization 
![Diagram of Distributed Genetic Algorithm](../diagrams/DGA_diagram.png)

# How to write Client scripts

# Server.py and Server description

```
├── run_name
│   ├── logs
│   └── pool
```