# Multi-Agent Arm Control

## Algorithm

The algorithm used to optimize learning was the deep deterministic policy gradient (DDPG).
This algorithm uses two sets of networks, the actor and the critical, each with a local and a target network.
The actor's function is to return an action for each state (that is deterministic part of the algorithm), and the critical returns the Q value from the state and the action that the actor returns.

Each network starts with the random weights, which induce exploration in the first episodes. At each step, an action is selected through the actor, and the environment returns the reward, the next state and if the episode ended (done). That is, at each step we have a set of state, action, reward, next state and done, this set is considered an experience and is saved in an experience buffer. As soon as we have enough experience, the system starts to learn.

Learning happens at the end of each step, in this process, a number of previous experiences are selected and the weights of the networks are updated.

The critical network is updated from the difference between the Q-value of the current state and the Q-value of the next state plus the reward.

<div style="text-align: center;">Q_targets_next = critic_target(next_state, action_next)</div>
<div style="text-align: center;">Q_target = reward + gamma * Q_targets_next</div>
<div style="text-align: center;">Q_expected = critic_local(state, action)</div>
<div style="text-align: center;">critic_loss = loss_function(Q_expected, Q_target) </div>

The actor network is updated using the gradient of the critical network when receiving the states and actions predicted by the actor network.

<div style="text-align: center;">actions_pred = actor_local(state)</div>
<div style="text-align: center;">actor_loss = -critic_local(state, actions_pred)</div>

The fact that the action is determined by one network and the Q value is calculated in another one increases the stability of the algorithm.
More details about the algorithm [here](https://arxiv.org/abs/1509.02971).

## Hyperparameters

| Hyperparameter                      | Value |
| ----------------------------------- | ----- |
| Replay buffer size                  | 1e5   |
| Batch size                          | 128   |
| Gamma                               | 0.99  |
| Tau                                 | 1e-3  |
| Learning rate actor                 | 1e-4  |
| Learning rate critic                | 1e-3  |
| Weight decay                        | 0     |
| Number of episodes                  | 1000  |

## Model architeture

### Actor

The local and the target networks have the same architecture.

Input layer: 33 linear nodes (the state size).
First hidden layer: 256 linear nodes.
Second hidden layer: 128 linear nodes.
Output layer: 4 linear nodes. (the action size).

Between each layer there is a ReLU activation function and a hyperbolic tangent after the output layer.

### Critic

The local and the target networks have the same architecture.

Input layer: 37 linear nodes (the state size + action size).
First hidden layer: 260 linear nodes (256 + action size).
Second hidden layer: 128 linear nodes.
Output layer: 1 linear node (the Q value).

Between each layer there is a ReLU activation function.