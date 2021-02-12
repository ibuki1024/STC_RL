# Reinforcement Learning of Self-Triggered Control

# Requirements
`Python 3.7`<br>
`tensorflow == 1.14.0`<br>
`keras == 2.2.4`<br>
`keras-rl == 0.4.2`<br>

# Directry
```
STC_RL/
　├ module/
　│　├ gym2/ (original environment)
　│　├ keras2/ (original activation function)
　│　├ rl2/ (not used)
　│　│　├ selfcore.py (original algorithm)
　│　│　└ agents/
　│　│　 　└ selfddpg.py (not for use)
　├ notebooks/
　│　├ linear/
　│　│　├ practical.ipynb (practical implementation)
　│　│　├ ideal.ipynb (naive implementation)
　│　│　├ saved_agent/ (initial policy or proposed policy)
　│　│　│　├ *_actor.h5 (weight_of_actor)
　│　│　│　└ *_critic.h5 (weight_of_critic)
　│　└ pendulum/
　│　│　├ practical.ipynb (practical implementation)
　│　│　├ ideal.ipynb (naive implementation)
　│　│　├ saved_agent/ (initial policy or proposed policy)
　│　│　│　├ *_actor.h5 (weight_of_actor)
　│　│　│　└ *_critic.h5 (weight_of_critic)
```
