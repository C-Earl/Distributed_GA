Note: This diagram shows an abstract representation of how the algorithm progresses, but the code implementations aren’t as straight-forward as what’s shown above.

Step 0: From a starting file, a Server object is initialized by specifying a Model to train and a Genetic Algorithm to optimize it. From here, the Server will begin the run by initializing the gene pool (Step 1).

Step 1:  
Given the state of the gene pool, the Genetic Algorithm applies logic to prune weak genes and generate new ones. A simple example is removing the gene with the lowest fitness and adding a new gene in its place. 

On initialization, the Current Gene Pool will be empty, so special logic is used to generate genes from scratch.

Step 2: 
After the algorithm has finished manipulating the pool, exactly one New Gene should be generated and returned. This New Gene should already be in the Updated Pool, but returning the name helps the Server identify the New Gene that was just made so it can send it to a new subprocess for testing (Step 3).

Step 3: Here, an asynchronous subprocess is called to test the New Gene. This subprocess will run in parallel to other subprocess’s running different genes on the same model. The only requirement for the Model is to return a numeric fitness value back to the Server (Step 4).

Step 4: Once tested, the New Gene Fitness value is returned to the server. The Gene Pool is updated with the New Gene Fitness, and thus the state of the pool is updated. This new Current Gene Pool is sent back to the Genetic Algorithm for adjustment (Back to Step 1!)


This cycle repeats until the Genetic Algorithm breaks and exits when an end condition is met. An example for an end condition is setting a maximum number of genes to be tested (for Genetic_Algorithm, this is the ‘iterations’ arg). 



Code Documentation:
The main files for the Server, Genetic Algorithm, and Model are found in “/Distributed_GA/DGA/”. I’ll describe each of the three parts below, and I recommend reading them in order (It will be easier to understand this way).

Server:
The Servers job is to handle messages from both the genetic algorithm and subprocess’s testing models, and update the state of the run accordingly. Implementation, however, is not as straight-forward. Most Cluster OS’s like SLURM don’t allow users to run a server to manage nodes, so alternative methods need to be used. To mimic the perpetually-online behavior of a server, the Server.py script automatically saves the state of the run to the disk after processing a request, and then exits. When a new request to the Server is made, Server.py will restart and pick up where it left off by reloading the state. 

The easiest way to make sense of this is to simplify the code. There are three types of requests: 
“init”, “server-callback” and “run-client”. “init” indicates a run is being initialized, “server-callback” handles when a subprocess returns a fitness, and “run-client” when the genetic algorithm returns a new gene to test. 

When a request is made, the ‘main’ function at the bottom of the file is executed first. This is where the previous run state is loaded and prepared for the Server object. The Server object is initialized.

Server’s __init__ function acts like a switch