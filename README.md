<h1 style="font-size:50px;">Cart-Pole Balance using Reinforcement Learning</h1>

In this project, we will be using reinforcement learning to build a model, train, test and evaluate it. We will be using stable-baselines3 with the PPO model on the classic cart-pole problem. We will run a simulation using OpenAI's gym environment.\

The actions are:
1. Move Left
2. Move Right
(Discrete Gym Space)

And the observations are:
1. Cart Position
2. Cart Velocity
3. Pole Position
4. Pole Velocity
(Box Gym Space)

We first run the simulation by passing sample values to the action. We observe that the cart moves unnaturally. After using the model, we get the returned value of 200 for majority of the trials, signifying that the model is accurate. Further details such as fps, loss, learning rate can be viewed by tensorboard from the PPO logs.

<h1>Import dependencies</h1>


```python
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
```

<h1>Load Environment</h1>


```python
environment_name = "CartPole-v0"
# Create Env
env = gym.make(environment_name)
```


```python
episodes = 20
# Loop through each episode
for episode in range(1, episodes+1):
    # Set environment state
    state = env.reset()
    done = False
    score = 0
    
    while not done:
        # Display Ca rt Window
        env.render()
        # Set action as sample
        action = env.action_space.sample()
        # Return new state, reward, done state and info
        n_state, reward, done, info = env.step(action)
        # Keep tally of score
        score += reward
    
    print("Episode: {} Score: {}".format(episode, score))
    
# Close Window
env.close()
```

    Episode: 1 Score: 16.0
    Episode: 2 Score: 16.0
    Episode: 3 Score: 10.0
    Episode: 4 Score: 13.0
    Episode: 5 Score: 13.0
    Episode: 6 Score: 12.0
    Episode: 7 Score: 17.0
    Episode: 8 Score: 14.0
    Episode: 9 Score: 13.0
    Episode: 10 Score: 10.0
    Episode: 11 Score: 36.0
    Episode: 12 Score: 34.0
    Episode: 13 Score: 17.0
    Episode: 14 Score: 20.0
    Episode: 15 Score: 14.0
    Episode: 16 Score: 15.0
    Episode: 17 Score: 23.0
    Episode: 18 Score: 12.0
    Episode: 19 Score: 34.0
    Episode: 20 Score: 13.0
    

![ss\1.png](attachment:image.png)

<h1> Understanding the Environment </h1>


```python
env.reset()
```




    array([ 0.03944678, -0.04542061, -0.04584641,  0.03873855], dtype=float32)




```python
env.action_space
```




    Discrete(2)




```python
# 0 for left action, 1 for right action
env.action_space.sample()
```




    0




```python
env.observation_space
```




    Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32)




```python
# 4 values, cart/pole position/velocity
env.observation_space.sample()
```




    array([ 4.1324692e+00,  4.9251807e+36, -2.8778794e-01,  1.8393422e+38],
          dtype=float32)




```python
env.step(1)
```




    (array([ 0.03853837,  0.15032777, -0.04507164, -0.26804963], dtype=float32),
     1.0,
     False,
     {})



!["ss\2.png"](attachment:image.png)

<h1>Algorithm</h1>

!["ss\3.png"](attachment:image.png)

<h1>Train an RL Model</h1>


```python
# MAKE YOUR DIRECTORIES FIRST
log_path = os.path.join('Training','Logs')
```


```python
log_path
```




    'Training\\Logs'




```python
# Make environment
env = gym.make(environment_name)
# Wrapper for non-vectorised env
env = DummyVecEnv([lambda: env])
# Using PPO model
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
```

    Using cpu device
    


```python
PPO??
```


```python
model.learn(total_timesteps=20000)
```

    Logging to Training\Logs\PPO_2
    -----------------------------
    | time/              |      |
    |    fps             | 2559 |
    |    iterations      | 1    |
    |    time_elapsed    | 0    |
    |    total_timesteps | 2048 |
    -----------------------------
    ------------------------------------------
    | time/                   |              |
    |    fps                  | 1673         |
    |    iterations           | 2            |
    |    time_elapsed         | 2            |
    |    total_timesteps      | 4096         |
    | train/                  |              |
    |    approx_kl            | 0.0080842525 |
    |    clip_fraction        | 0.0934       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.686       |
    |    explained_variance   | 0.00172      |
    |    learning_rate        | 0.0003       |
    |    loss                 | 6.12         |
    |    n_updates            | 10           |
    |    policy_gradient_loss | -0.0158      |
    |    value_loss           | 54.6         |
    ------------------------------------------
    -----------------------------------------
    | time/                   |             |
    |    fps                  | 1477        |
    |    iterations           | 3           |
    |    time_elapsed         | 4           |
    |    total_timesteps      | 6144        |
    | train/                  |             |
    |    approx_kl            | 0.009030341 |
    |    clip_fraction        | 0.0624      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.662      |
    |    explained_variance   | 0.0663      |
    |    learning_rate        | 0.0003      |
    |    loss                 | 15          |
    |    n_updates            | 20          |
    |    policy_gradient_loss | -0.018      |
    |    value_loss           | 43.3        |
    -----------------------------------------
    -----------------------------------------
    | time/                   |             |
    |    fps                  | 1418        |
    |    iterations           | 4           |
    |    time_elapsed         | 5           |
    |    total_timesteps      | 8192        |
    | train/                  |             |
    |    approx_kl            | 0.007265622 |
    |    clip_fraction        | 0.0698      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.636      |
    |    explained_variance   | 0.181       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 28.9        |
    |    n_updates            | 30          |
    |    policy_gradient_loss | -0.0158     |
    |    value_loss           | 61.1        |
    -----------------------------------------
    -----------------------------------------
    | time/                   |             |
    |    fps                  | 1382        |
    |    iterations           | 5           |
    |    time_elapsed         | 7           |
    |    total_timesteps      | 10240       |
    | train/                  |             |
    |    approx_kl            | 0.006497982 |
    |    clip_fraction        | 0.0593      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.609      |
    |    explained_variance   | 0.214       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 29          |
    |    n_updates            | 40          |
    |    policy_gradient_loss | -0.0154     |
    |    value_loss           | 70.8        |
    -----------------------------------------
    ------------------------------------------
    | time/                   |              |
    |    fps                  | 1364         |
    |    iterations           | 6            |
    |    time_elapsed         | 9            |
    |    total_timesteps      | 12288        |
    | train/                  |              |
    |    approx_kl            | 0.0064096954 |
    |    clip_fraction        | 0.0534       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.597       |
    |    explained_variance   | 0.38         |
    |    learning_rate        | 0.0003       |
    |    loss                 | 20.4         |
    |    n_updates            | 50           |
    |    policy_gradient_loss | -0.0115      |
    |    value_loss           | 68.8         |
    ------------------------------------------
    -----------------------------------------
    | time/                   |             |
    |    fps                  | 1347        |
    |    iterations           | 7           |
    |    time_elapsed         | 10          |
    |    total_timesteps      | 14336       |
    | train/                  |             |
    |    approx_kl            | 0.008010818 |
    |    clip_fraction        | 0.063       |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.586      |
    |    explained_variance   | 0.378       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 21.4        |
    |    n_updates            | 60          |
    |    policy_gradient_loss | -0.0112     |
    |    value_loss           | 59.9        |
    -----------------------------------------
    ------------------------------------------
    | time/                   |              |
    |    fps                  | 1340         |
    |    iterations           | 8            |
    |    time_elapsed         | 12           |
    |    total_timesteps      | 16384        |
    | train/                  |              |
    |    approx_kl            | 0.0078093545 |
    |    clip_fraction        | 0.0724       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.589       |
    |    explained_variance   | 0.781        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 2.93         |
    |    n_updates            | 70           |
    |    policy_gradient_loss | -0.0106      |
    |    value_loss           | 32.6         |
    ------------------------------------------
    ------------------------------------------
    | time/                   |              |
    |    fps                  | 1323         |
    |    iterations           | 9            |
    |    time_elapsed         | 13           |
    |    total_timesteps      | 18432        |
    | train/                  |              |
    |    approx_kl            | 0.0037150714 |
    |    clip_fraction        | 0.0175       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.58        |
    |    explained_variance   | 0.703        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 10.9         |
    |    n_updates            | 80           |
    |    policy_gradient_loss | -0.00468     |
    |    value_loss           | 38.4         |
    ------------------------------------------
    -----------------------------------------
    | time/                   |             |
    |    fps                  | 1317        |
    |    iterations           | 10          |
    |    time_elapsed         | 15          |
    |    total_timesteps      | 20480       |
    | train/                  |             |
    |    approx_kl            | 0.010578734 |
    |    clip_fraction        | 0.109       |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.573      |
    |    explained_variance   | 0.916       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 5.11        |
    |    n_updates            | 90          |
    |    policy_gradient_loss | -0.012      |
    |    value_loss           | 16.8        |
    -----------------------------------------
    




    <stable_baselines3.ppo.ppo.PPO at 0x2a34545cac0>



<h1>Saving and reloading the model</h1>


```python
PPO_path = os.path.join('Training','Saved Models','PPO_Model_Cartpole')
```


```python
PPO_path
```




    'Training\\Saved Models\\PPO_Model_Cartpole'




```python
model.save(PPO_path)
```


```python
del model
```


```python
model = PPO.load(PPO_path, env=env)
```

    Wrapping the env with a `Monitor` wrapper
    Wrapping the env in a DummyVecEnv.
    


```python
model
```




    <stable_baselines3.ppo.ppo.PPO at 0x25a19f74fd0>



<h1>Evaluation</h1>


```python
evaluate_policy(model, env, n_eval_episodes=20, render=True)
```

    C:\Users\Tanush R\anaconda3\lib\site-packages\stable_baselines3\common\evaluation.py:65: UserWarning: Evaluation environment is not wrapped with a ``Monitor`` wrapper. This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. Consider wrapping environment first with ``Monitor`` wrapper.
      warnings.warn(
    




    (200.0, 0.0)




```python
env.close()
```

<h1>Test Model</h1>


```python
episodes = 20
# Loop through each episode
for episode in range(1, episodes+1):
    # Set observation
    obs = env.reset()
    done = False
    score = 0
    
    while not done:
        # Display Cart Window
        env.render()
        # Set action to predicted value, ignore other so _
        action, _ = model.predict(obs)
        # Return new state, reward, done state and info
        obs, reward, done, info = env.step(action)
        # Keep tally of score
        score += reward
    
    print("Episode: {} Score: {}".format(episode, score))
    
# Close Window
env.close()
```

    Episode: 1 Score: [178.]
    Episode: 2 Score: [200.]
    Episode: 3 Score: [176.]
    Episode: 4 Score: [200.]
    Episode: 5 Score: [200.]
    Episode: 6 Score: [200.]
    Episode: 7 Score: [132.]
    Episode: 8 Score: [200.]
    Episode: 9 Score: [200.]
    Episode: 10 Score: [200.]
    Episode: 11 Score: [200.]
    Episode: 12 Score: [200.]
    Episode: 13 Score: [200.]
    Episode: 14 Score: [200.]
    Episode: 15 Score: [200.]
    Episode: 16 Score: [200.]
    Episode: 17 Score: [200.]
    Episode: 18 Score: [200.]
    Episode: 19 Score: [200.]
    Episode: 20 Score: [106.]
    

!["ss\1.png"](attachment:image.png)


```python
env.close()
```


```python
obs = env.reset()
```


```python
obs
```




    array([[ 0.01475481, -0.02922212, -0.00979095, -0.01112738]],
          dtype=float32)




```python
action, _ = model.predict(obs)
```


```python
env.step(action)
```




    (array([[ 0.01417036, -0.22420229, -0.0100135 ,  0.27845037]],
           dtype=float32),
     array([1.], dtype=float32),
     array([False]),
     [{}])



<h1> Viewing Logs in Tensorboard </h1>


```python
training_log_path = os.path.join(log_path, 'PPO_2')
```


```python
training_log_path
```




    'Training\\Logs\\PPO_2'




```python
!tensorboard --logdir={training_log_path}
```

!["ss\4.png"](attachment:image.png)
