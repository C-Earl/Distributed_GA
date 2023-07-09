# Introduction
Distributed Genetic Algorithms (DGA) is a Python based module that will run a highly paralellized genetic algorithm. Given you have the appropriate hardware, the paralellization should net a boost in performance from non-paralellized versions. To briefly explain, paralellization is done by sending 'genes' (model parameters) to an asynchronous subprocess for fitness evaluation.

DGA is designed to be modular, meaning it's designed to run *any* model with a quantifiable 'fitness'. You may also write your own genetic algorithm scripts to be deployed asynchronously.

### An important term:
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
This example will show you how to run a distributed genetic algorithm on your local machine using DGA module. The example script works by calling asynchronous subprocesses to test a very basic model. 

Using VScode, navigate to ```./Distributed_GA/scripts/local_example/Example.py``` To run Example.py, simply use the run button on the IDE. Alternatively, you can navigate to the same location in the VScode terminal and run ```python3 Example.py```.

To tell if the genetic algorithm has finished running, check your process manager. While the algorithm is running, you should see 5 Python processes in your manager. These are the parallel processes testing the example model, and will 'reproduce' (create new gene & call new process to test it) until the max iterations. **You will know the run has finished when all the Python processes have disappeared from your process manager.**

Soon after you start, you should see a new directory called ```example_run_name```. This is where all the data (genes, fitnesses, etc.) are stored. To see the result of your run, run the ```Evaluate.py``` file next to ```Example.py```. This will print the current state of the gene pool (should be 10 genes). Outputs should look something like this:
```bash
Gene: fc4843f1f6...             Fitness: -0.8420308346282214		# gene name is hashed gene
```

# How to run your own models on the base algorithm
Limitations:
* Currently, the only 'gene' format supported is numpy array. Whatever model you want to test, make sure its parameters can be formatted as a numpy array.

**Use Example.py as a template**, it has inline comments for everything covered here

1. Navigate to ```./Distributed_GA/scripts```. This is where all your personal projects should go. We're now going to make a few directories:

```
├── Distributed_GA
|		...
│   └── scripts
│   	└── Project_Name
|				...
│   		└── Project_code.py
|		...
```

2. Choose a name for your project (```Project_Name```), and make a folder with that name.
3. Choose a name for the file that will contain your genetic algorithm and model code. Create a python file with that name (```Project_code.py```) in your scripts folder. Your directory should look like the above.

4. Like in ```Example.py```, add the following imports to the top of ```Project_code.py``` along with any other imports you need for your model. (Don't worry about ```Algorithm```):
```python
from DGA.Client import Client
from DGA.Server import Server
```
3. Choose a name for a new Python class (```My_Client```) which inherits ```Client```, and implement the following ```run()``` function:
```Python
class My_Client(Client):
	def run(self) -> float:
		gene = self.gene_data['gene']
		fitness = 0.0
		return fitness
```
* Brief explanation of ```run()```:
	* When a subprocess tests a gene, it will setup ```My_Client``` with the gene info it needs for testing. ```self.gene_data['gene']``` contains the gene data in a numpy array. 
	* **The ```run()``` function will contain all of *your* model and environment code.** In other words, you shouldn't need to write any major Python scripts outside of the Client class (assuming I did my job correctly...).

3. Fill in your ```run()``` function as desired. In Example.py, the fitness is based on how close all the values in the gene are to 0. Here is an example of a more complex model you could implement:
```Python
import torch				# <--- Add any additional imports
from torch.utils.data import Dataset, Dataloader
from torchvision import datasets
from torchvision.transforms import ToTensor

class ANN(nn.Module):
	...

class My_Client(Client):
	def run(self) -> float:
		### Define Neural Net Model ###
		gene = self.gene_data['gene']
		model = ANN(gene)		# <--- Sets neuron params to gene values

		### Training Phase ###
		training_data = datasets.FashionMNIST(
			root="data",
			train=True,
			download=True,
			transform=ToTensor()
		)
		train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

		for feature, label in train_dataloader:
			...

		### Testing Phase ###
		test_data = datasets.FashionMNIST(
				root="data",
				train=False,
				download=True,
				transform=ToTensor()
		)
		test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
		
		total_loss = 0
		for feature, label in test_dataloader:
			...		# <--- Loss accumulated during testing
		
		fitness = 1 / total_loss 	# More loss -> lower fitness
		return fitness
```

4. The final step is to make the file runnable. To do this, add the following to the very bottom of your file, and fill in the missing arguments:
```Python
if __name__ == '__main__':
  import os
  alg_path = os.path.abspath(__file__)
  client_path = os.path.abspath(__file__)

  Server(run_name="",
		algorithm_path="scripts/local_example/Example.py",  # Use example Algorithm
		algorithm_name="Simple_GA",
		client_path="scripts/local_example/", 	# Fill in your own Client info
		client_name="",
		num_parallel_processes=, 
		gene_shape=(,), 
		num_genes=, 
		mutation_rate=,
		iterations=
	)
```

* ```run_name``` (str) = a name for your outputs folder
* ```algorithm_path (str)``` = path from ```./Distributed_GA``` to python file containing your Algorithm class
* ```algorithm_name (str)``` = name you gave your Algorithm class
* ```client_path (str)``` = path from ```./Distributed_GA``` to python file containing your Client class
* ```client_name (str)``` = string version of the name you gave your Client class
* ```num_parallel_processes (int)``` = # of genes being tested in parallel
* ```gene_shape (tuple)``` = shape of genes
* ```num_genes (int)``` = max # of genes stored in gene pool 
* ```iterations (int)``` = how many genes each parallel processor will test. Total genes tested is ```iterations * num_parallel_processes```

5. You should now be able to run your file, just like Example.py. See the 'how to run example' section on how to tell when the algorithm has finished running. Again, it is reccomended to use VScode since the repository is preconfigured for it.

6. To see the results of your run, copy Evaluate.py to your project folder and at the bottom change this line to reflect your own ```run_name``` (This is a WIP)

```python
...	# This is at the bottom of Evaluate.py
  if len(all_args.items()) == 0:

    ### Manual Inputs ###
    all_args['run_name'] = ""	# <--- Your run_name here
```

# How to write Genetic Algorithms scripts

# Server.py and Server description
