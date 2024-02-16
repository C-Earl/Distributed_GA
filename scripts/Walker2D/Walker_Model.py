import numpy as np
import gymnasium as gym
from tqdm import tqdm
from DDPG import ReplayBuffer, TD3
from inference import evaluate_policy

from DGA.Model import Model
from DGA.Gene import Parameters, Gene, Genome


class Walker_Model(Model):
  def __init__(self):
    super().__init__()          # Add hyperparams to log
    self.log_vars.extend(['expl_noise', 'discount', 'tau', 'policy_noise', 'noise_clip'])

  def run(self, params: Parameters, **kwargs) -> float:
    ## Default hyperparameters ###
    env_name = "Walker2d-v4"  # Name of a environment (set it to any Continous environment you want)
    max_timesteps = 5e5  # Total number of iterations/timesteps
    start_timesteps = 1e4  # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
    batch_size = 100  # Size of the batch
    policy_freq = 2  # Number of iterations to wait before the policy network (Actor model) is updated

    ### Hyperparams ###
    expl_noise = params['expl_noise']  # Exploration noise - STD value of exploration Gaussian noise
    discount = params['discount']  # Discount factor gamma, used in the calculation of the total discounted reward
    tau = params['tau']  # Target network update rate
    policy_noise = params['policy_noise']  # STD of Gaussian noise added to the actions for the exploration purposes
    noise_clip = params['noise_clip']  # Maximum value of the Gaussian noise added to the actions (policy)
    print("---------------------------------------")
    print(f"expl_noise: {expl_noise}, discount: {discount}, tau: {tau}, policy_noise: {policy_noise}, noise_clip: {noise_clip}")

    # Create the PyBullet environment
    env = gym.make(env_name, healthy_z_range=(0.8, 2.0), terminate_when_unhealthy=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    ## Create the policy network (the Actor model), Replay buffer
    policy = TD3(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()

    ## We initialize the variables
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True

    ### Training ###
    with tqdm(total=max_timesteps) as pbar:
      while total_timesteps < max_timesteps:

        # If the episode is done
        if done:

          # If we are not at the very beginning, we start the training process of the model
          if total_timesteps != 0:
            # print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
            policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip,
                         policy_freq)

          # When the training step is done, we reset the state of the environment
          obs = env.reset()[0]

          # Set rewards and episode timesteps to zero
          episode_reward = 0
          episode_timesteps = 0
          episode_num += 1

        # Before 10000 timesteps, we play random actions
        if total_timesteps < start_timesteps:
          action = env.action_space.sample()
        else:  # After 10000 timesteps, we switch to the model
          action = policy.select_action(np.array(obs))
          # If the explore_noise parameter is not 0, we add noise to the action and we clip it
          if expl_noise != 0:
            action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low,
                                                                                                     env.action_space.high)

        # The agent performs the action in the environment, then reaches the next state and receives the reward
        new_obs, reward, term, done, _ = env.step(action)
        if term:  # if the agent falls, we terminate the episode
          done = True

        # We check if the episode is done
        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)

        # We increase the total reward
        episode_reward += reward

        # We store the new transition into the Experience Replay memory (ReplayBuffer)
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
        obs = new_obs
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
        pbar.update(1)

      fitness = evaluate_policy(env, policy)
      print(f"Fitness: {fitness}")
      return fitness
    # fitness = 0
    # if round(expl_noise, 2) == 0.1:
    #   fitness += 1
    # if round(discount, 4) == 0.99:
    #   fitness += 1
    # if round(tau, 3) == 0.005:
    #   fitness += 1
    # if round(policy_noise, 2) == 0.2:
    #   fitness += 1
    # if round(noise_clip, 2) == 0.5:
    #   fitness += 1
    # if policy_freq == 2:
    #   fitness += 1
    # return fitness

if __name__ == '__main__':
  env_name = "Walker2d-v4"  # Name of a environment (set it to any Continous environment you want)
  # seed = 0  # Random seed number
  start_timesteps = 1e4  # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
  eval_freq = 5e3  # How often the evaluation step is performed (after how many timesteps)
  # max_timesteps = 5e5  # Total number of iterations/timesteps
  # save_models = True  # Boolean checker whether or not to save the pre-trained model
  expl_noise = 0.1  # Exploration noise - STD value of exploration Gaussian noise
  batch_size = 100  # Size of the batch
  discount = 0.99  # Discount factor gamma, used in the calculation of the total discounted reward
  tau = 0.005  # Target network update rate
  policy_noise = 0.2  # STD of Gaussian noise added to the actions for the exploration purposes
  noise_clip = 0.5  # Maximum value of the Gaussian noise added to the actions (policy)
  policy_freq = 2  # Number of iterations to wait before the policy network (Actor model) is updated

  model = Walker_Model()
  genome = Genome()
  genome.add_gene(Gene(dtype=float, min_val=0.01, max_val=3), 'expl_noise')
  genome.add_gene(Gene(dtype=float, min_val=0.9, max_val=0.9999), 'discount')
  genome.add_gene(Gene(dtype=float, min_val=0.001, max_val=0.010), 'tau')
  genome.add_gene(Gene(dtype=float, min_val=0.05, max_val=0.5), 'policy_noise')
  genome.add_gene(Gene(dtype=float, min_val=0.1, max_val=1.0), 'noise_clip')
  genome.add_gene(Gene(dtype=int, min_val=2, max_val=10), 'policy_freq')
  params = genome.initialize(0)
  model.run(params)