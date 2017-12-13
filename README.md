# Pytorch - Rainbow Net

## This implementation is inspired by:
1. OpenAI Tensorflow code: https://github.com/openai/baselines/tree/master/baselines/deepq
2. https://github.com/ShangtongZhang/DeepRL
3. https://github.com/floringogianu/categorical-dqn


To run training:

```bash
python main.py
```

To modify training parameters, modify:
```
params.py
```


## Components
- [x] Dueling Architecture
- [x] n-step Double Q-learning   
- [x] Prioritized replay
- [x] Noisy Linear Layer -> Greatly improves learning
- [x] Distributional policy
- [ ] Scale training to train and benchmark in Atari environments
- [ ] Continuous counterpart
- [ ] TreeQN / ATreeC

## Disclaimer
Needs more benchmarking against several different environments, the current implementation works within the CartPole environment.

## Common mistakes
1. Flip 'dones' variable before multiplying with state value
2. Use 'Tensor.index_add_', instead of 'Tensor[i].add_'
3. For n-step learning, do optimizer update at every step, avoid aggregating losses from each step. Still not sure why that doesn't work.


