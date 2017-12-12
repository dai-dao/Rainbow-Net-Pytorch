import numpy as np 
import random
from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer_NStep(object):
    def __init__(self, size, nstep):
        self._storage = []
        self._maxsize = size
        self._nstep = nstep
        self._next_idx = 0


    def __len__(self):
        return len(self._storage)


    def add(self, data):
        '''
            data: a list that contains [(ob, action, reward, ob_next, done) * nstep]
        '''
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
    

    def _encode_sample(self, idxes):
        '''
            Return:
                Return arrays of shape [batch_size, nsteps]
        '''
        obs_n, actions_n, rewards_n, obs_next_n, dones_n = [], [], [], [], []

        for i in idxes:
            data = self._storage[i]
            obs, actions, rewards, obs_next, dones = [], [], [], [], []

            for s_i, step in enumerate(data):
                ob, action, reward, ob_next, done = step

                obs.append(np.array(ob, copy=False))
                obs_next.append(np.array(ob_next, copy=False))
                actions.append(np.array(action, copy=False))
                rewards.append(np.array(reward))
                dones.append(np.array(done))

            obs_n.append(np.array(obs))
            obs_next_n.append(np.array(obs_next))
            actions_n.append(np.array(actions))
            rewards_n.append(np.array(rewards))
            dones_n.append(np.array(dones))

        return np.array(obs_n), np.array(actions_n), np.array(rewards_n), \
               np.array(obs_next_n), np.array(dones_n)


    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)



class PrioritizedReplayBuffer_NStep(ReplayBuffer_NStep):
    def __init__(self, size, nstep, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer_NStep, self).__init__(size, nstep)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0


    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    
    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res


    def sample(self, batch_size, beta):
        """
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

            for importance weights, which is used to determine how much 
            to scale the gradients of high error samples DOWN, INCREASE
            correction as training progresses
        Returns
        -------
        obs_batch: np.array
            batch of observations
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])


    def update_priorities(self, idxes, priorities):
        """
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class PrioritizedReplay(ReplayBuffer_NStep):
    """
        Use sampling with numpy -> MUCH slower when scaling up compared to using segment tree
    """

    def __init__(self, size, nstep, alpha):
        """
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        nstep: int
            Number of steps in the sequence for multi-step learning
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        --------
        """
        super(PrioritizedReplay, self).__init__(size, nstep)
        assert alpha > 0
        self._alpha = alpha
        self._max_priority = 1.0
        self.priorities = np.array([0.0 for _ in range(size)])


    def add(self, *args, **kwargs):
        """
        New samples have max priorities, most likely to be sampled first, and then 
        have its priorities updated
        """
        idx = self._next_idx
        super().add(*args, **kwargs)
        self.priorities[idx] = self._max_priority ** self._alpha
    

    def _sample_proportional(self, batch_size):
        # Sample without replacement -> no repeats
        probs = self.priorities / np.sum(self.priorities)
        return np.random.choice(self._maxsize, batch_size, p=probs, replace=False)


    def sample(self, batch_size, beta):
        assert beta > 0
        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self.priorities[np.nonzero(self.priorities)].min() / np.sum(self.priorities)
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self.priorities[idx] / np.sum(self.priorities)
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])


    def update_priorities(self, idxes, priorities):
        """
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self.priorities[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)