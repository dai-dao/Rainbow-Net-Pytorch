# Pytorch - Rainbow Net

## This implementation is inspired by:
1. OpenAI Tensorflow code: https://github.com/openai/baselines/tree/master/baselines/deepq
2. https://github.com/ShangtongZhang/DeepRL


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
- [ ] Parameter space noise
- [ ] Distributional policy
- [ ] Continuous counterpart
- [ ] TreeQN / ATreeC

## Disclaimer
Needs more benchmarking against several different environments, the current implementation works within the CartPole environment.

## Common mistakes
1. Flip the dones variable before multiplying with state value