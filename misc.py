"""
    To store other interesting versions of algorithms
"""

# DQN version
def dqn():
    next_target_phi = self.target_model(obs_next)
    next_target_q = self.p_to_q(next_target_phi)
    _, next_target_action = next_target_q.data.max(1)
    next_target_best = next_target_phi[batch_dim, next_target_action].data


def ddqn():
    next_online_phi = self.model(obs_next)
    next_online_q = self.p_to_q(next_online_phi)
    _, next_online_action = next_online_q.data.max(1)
    next_target_phi = self.target_model(obs_next).data
    next_target_best = next_target_phi[batch_dim, next_online_action]


# Other variants of distributional update algorithm
def dist_update_v1(self, obs, actions, rewards, obs_next, dones, weights):
    obs = Variable(torch.from_numpy(obs).type(self.dtype)).view(-1, self.ob_space)
    obs_next = Variable(torch.from_numpy(obs_next).type(self.dtype)).view(-1, self.ob_space)
    weights = Variable(torch.from_numpy(weights).type(self.dtype)).view(-1, 1)
    actions = Variable(torch.from_numpy(actions.astype(int)).type(self.atype)).view(-1, 1)
    rewards = torch.from_numpy(rewards).type(self.dtype).view(-1, 1)
    dones = torch.from_numpy(dones.astype(float)).type(self.dtype).view(-1, 1)  

    batch_dim = self.atype(np.arange(obs.size(0)))
    batch_size = obs.size(0)

    # Compute p'(x', a')
    next_target_phi = self.target_model(obs_next)
    next_target_q = self.p_to_q(next_target_phi)
    _, next_target_action = next_target_q.data.max(1)
    next_target_best = next_target_phi[batch_dim, next_target_action].data

    # 
    big_r = rewards.expand(batch_size, self.nb_atoms)
    big_dones = dones.expand(batch_size, self.nb_atoms)
    big_z = self.z.expand(batch_size, -1)
    
    # Compute projection of Bellman operator
    bellman_op = torch.clamp(big_r + gamma * big_z * (1.0 - big_dones), self.v_min, self.v_max)

    # Compute categorical indices for distributing the probabilities
    b = (bellman_op - self.v_min) / self.dz
    l = b.floor().long()
    u = b.ceil().long()

    offset = torch.linspace(0, (batch_size - 1) * self.nb_atoms, batch_size).type(self.atype)
    offset = offset.unsqueeze(1).expand(batch_size, self.nb_atoms)

    # Distribute probabilities
    m = self.m.fill_(0)
    m.view(-1).index_add_(0, (l + offset).view(-1),
                            (next_target_best * (u.float() - b)).view(-1))
    m.view(-1).index_add_(0, (u + offset).view(-1),
                            (next_target_best * (b - l.float())).view(-1))

    online_phi = self.model(obs)
    q_out_selected = online_phi[batch_dim, actions.squeeze()]
    cross_entropy_losses = -(Variable(m) * torch.log(q_out_selected + 1e-5)).sum(-1)

    self.optimizer.zero_grad()
    cross_entropy_losses.mean().backward()
    nn.utils.clip_grad_norm(self.model.parameters(), self.args.grad_norm_clipping)
    self.optimizer.step()        


def dist_update_v4(self, obs, actions, rewards, obs_next, dones, weights):
    batch_size = self.args.batch_size
    batch_dim = self.atype(np.arange(batch_size))    
    gamma = self.args.gamma
    nb_atoms = self.nb_atoms

    obs = Variable(torch.from_numpy(obs).type(self.dtype)).view(-1, self.ob_space)
    obs_next = Variable(torch.from_numpy(obs_next).type(self.dtype)).view(-1, self.ob_space)
    weights = Variable(torch.from_numpy(weights).type(self.dtype)).view(-1, 1)
    actions = Variable(torch.from_numpy(actions.astype(int)).type(self.atype)).view(-1, 1)
    rewards = torch.from_numpy(rewards).type(self.dtype).view(-1, 1)
    dones = torch.from_numpy(dones.astype(float)).type(self.dtype).view(-1, 1)         

    # DDQN
    next_online_phi = self.model(obs_next)
    next_online_q = self.p_to_q(next_online_phi)
    _, next_online_action = next_online_q.data.max(1)
    next_target_phi = self.target_model(obs_next).data
    next_target_best = next_target_phi[batch_dim, next_online_action]

    m = self.m.fill_(0)
    for j in range(nb_atoms): 
        Tzj = torch.clamp(rewards + (1.0 - dones) * gamma * self.z[j], self.v_min, self.v_max)
        # Clipping to avoid numerical errors
        bj = ((Tzj - self.v_min) / self.dz).view(-1).clamp(0, nb_atoms-1)
        l = bj.floor().long()
        u = bj.ceil().long()
        m[batch_dim, l] = m[batch_dim, l] + (next_target_best[:, j] * (u.float() - bj))
        m[batch_dim, u] = m[batch_dim, u] + (next_target_best[:, j] * (bj - l.float()))

    online_phi = self.model(obs)
    q_out_selected = online_phi[batch_dim, actions.squeeze()]
    cross_entropy_losses = -(Variable(m) * torch.log(q_out_selected + 1e-5)).sum(-1)
    kl_errors = cross_entropy_losses.data.cpu().numpy().flatten()

    self.optimizer.zero_grad()
    cross_entropy_losses.mean().backward()
    nn.utils.clip_grad_norm(self.model.parameters(), self.args.grad_norm_clipping)
    self.optimizer.step() 
    return kl_errors


def dist_update_v2(self, obs, actions, rewards, obs_next, dones, weights):
    obs = Variable(torch.from_numpy(obs).type(self.dtype)).view(-1, self.ob_space)
    obs_next = Variable(torch.from_numpy(obs_next).type(self.dtype)).view(-1, self.ob_space)
    weights = Variable(torch.from_numpy(weights).type(self.dtype)).view(-1, 1)
    actions = Variable(torch.from_numpy(actions.astype(int)).type(self.atype)).view(-1, 1)
    rewards = torch.from_numpy(rewards).type(self.dtype).view(-1, 1)
    dones = torch.from_numpy(dones.astype(float)).type(self.dtype).view(-1, 1) 

    # 
    big_r = rewards.expand(batch_size, nb_atoms)
    big_dones = dones.expand(batch_size, nb_atoms)
    big_z = self.z.expand(batch_size, -1)
    
    # Compute projection of Bellman operator
    bellman_op = torch.clamp(big_r + gamma * big_z * (1.0 - big_dones), self.v_min, self.v_max)

    # Compute categorical indices for distributing the probabilities
    b = (bellman_op - self.v_min) / self.dz
    l = b.floor().long()
    u = b.ceil().long()

    m = self.m.fill_(0)

    for i in range(batch_size):
        for j in range(self.nb_atoms):
            uidx = u[i][j]
            lidx = l[i][j]
            m[i][lidx] = m[i][lidx] + next_target_best[i][j] * (uidx - b[i][j])
            m[i][uidx] = m[i][uidx] + next_target_best[i][j] * (b[i][j] - lidx)

    ''' OR '''
    for i in range(batch_size):
        m[i].index_add_(0, l[i], next_target_best[i] * (u[i].float() - b[i]))
        m[i].index_add_(0, u[i], next_target_best[i] * (b[i] - l[i].float()))


# Like this one best
def dist_update_v3(self, obs, actions, rewards, obs_next, dones, weights):
    obs = Variable(torch.from_numpy(obs).type(self.dtype)).view(-1, self.ob_space)
    obs_next = Variable(torch.from_numpy(obs_next).type(self.dtype)).view(-1, self.ob_space)
    weights = Variable(torch.from_numpy(weights).type(self.dtype)).view(-1, 1)
    actions = Variable(torch.from_numpy(actions.astype(int)).type(self.atype)).view(-1, 1)
    rewards = torch.from_numpy(rewards).type(self.dtype).view(-1, 1)
    dones = torch.from_numpy(dones.astype(float)).type(self.dtype).view(-1, 1)  

    # Online evaluation
    phi_online = self.model(obs)
    q_online = self.p_to_q(phi_online)

    # Target Q network evaluation
    phi_target_next = self.target_model(obs_next)
    q_target_next = self.p_to_q(phi_target_next)
    _, a_target_next = q_target_next.data.max(-1)

    # 
    batch_dim = self.atype(np.arange(obs.size(0)))
    phi_target_best = phi_target_next[batch_dim, a_target_next]

    #
    big_z = self.z.expand(obs.size(0), -1)
    big_r = rewards.expand(rewards.size(0), self.nb_atoms)
    big_done = dones.expand(dones.size(0), self.nb_atoms)

    # 
    Tz = torch.clamp(big_r + self.args.gamma * big_z * (1.0 - big_done), self.v_min, self.v_max)

    # TZ -> [32, 11]
    big_Tz = Tz.repeat(1, self.nb_atoms).view(-1, self.nb_atoms, self.nb_atoms)

    # big_z -> [32, 11]
    big_big_z = big_z.repeat(1, self.nb_atoms).view(-1, self.nb_atoms, self.nb_atoms)

    # https://arxiv.org/pdf/1707.06887.pdf
    Tzz = torch.abs(big_Tz - big_big_z.permute(0, 2, 1)) / self.dz
    Thz = torch.clamp(1 - Tzz, 0, 1)

    # Thz -> [32, 11, 11]
    # phi_target_best -> [32, 11]
    # ThTz -> [32, 11]
    ThTz = torch.matmul(Thz, phi_target_best.unsqueeze(2).data).squeeze()

    phi_best = phi_online[Variable(batch_dim), actions]
    cross_entropy = (-1 * Variable(ThTz) * torch.log(phi_best)).mean()

    self.optimizer.zero_grad()
    cross_entropy.backward()
    nn.utils.clip_grad_norm(self.model.parameters(), self.args.grad_norm_clipping)
    self.optimizer.step()