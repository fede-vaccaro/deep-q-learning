# Deep Q-Learning

From 
* [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), V. Mnih et al., 2013
* [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf) H. van Hasselt et al., 2015

The game is a simple "GridWorld" style game, where the agent has to move the green square to the red square, 
in a minimum number of steps.

<p align="center">
  <img src="https://imgur.com/btJfkjD.gif">
</p>

For using the code:
```
$ python main.py
```
Options: 
* `-d` for using Double Q-Learning 
* `--gpu` for using GPU acceleration

For testing a model:
```
$ python test.py -f FILENAME
```

Options:
* `--gpu` for using GPU acceleration
