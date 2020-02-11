from policies import base_policy as bp
import numpy as np
import tensorflow as tf
from tensorflow import keras
# TODO remove these lines - used for debugging purposes...
# TODO --------------------------------------------------
from datetime import datetime
import os
import json
import pathlib
# TODO --------------------------------------------------

# The possible values are integers from -1 to 9 (inclusive) representing the different objects,
# but we do not know which number correspond to which object.
N_VALUES = 11
FIRST_VALUE = -1

# There are 3 actions - L, R and F (i.e. left, right or forward).
N_ACTIONS = len(bp.Policy.ACTIONS)


def get_window(board, position, head, window_size):
    """
    Get a squared window of size (window_size x window_size) from the given board,
    centered at the given position and rotated so that the head direction will be upwards.
    :param board: The board, which is a NumPy array containing integers from -1 to 9 (inclusive).
    :param position: A Position object, describing the position of the head of the snake.
    :param head: The direction of the snake's head, a string in {'N', 'S', 'W', 'E'}
    :param window_size: An odd integer describing the window size.
    :return: The window, which is a NumPy array of shape (window_size, window_size).
    """
    assert window_size % 2 == 1, "window_size should be odd (to enable putting the head in the center)."

    board_width, board_height = board.shape
    center_x = position[0]
    center_y = position[1]
    half_window_size = window_size // 2

    start_x = center_x - half_window_size
    end_x = center_x + half_window_size
    start_y = center_y - half_window_size
    end_y = center_y + half_window_size

    x_indices = np.arange(start_x, end_x + 1) % board_width
    y_indices = np.arange(start_y, end_y + 1) % board_height

    window = board[x_indices.reshape(-1, 1), y_indices]

    # Define the amount of times to rotate 90-degrees counter-clock-wise, according to the head's direction.
    if head == 'N':
        k = 0
    elif head == 'S':
        k = 2
    elif head == 'E':
        k = 1
    elif head == 'W':
        k = 3
    else:
        raise ValueError("Unknown head, should be one of (N, S, E, W)")

    rotated_window = np.rot90(window, k)

    return rotated_window


def get_window_3d(board, position, head, window_size):
    """
    Get a 3D stack of boolean windows, indicating the appearance of a certain object in a certain location.
    :param board: The board, which is a NumPy array containing integers from -1 to 9 (inclusive).
    :param position: A Position object, describing the position of the head of the snake.
    :param head: The direction of the snake's head, a string in {'N', 'S', 'W', 'E'}
    :param window_size: An odd integer describing the window size.
    :return: The window, which is a NumPy boolean array of shape (window_size, window_size, N_VALUES).
    """
    window = get_window(board, position, head, window_size)
    window_3d = np.empty(shape=(window_size, window_size, N_VALUES), dtype=np.bool)

    for value in range(FIRST_VALUE, FIRST_VALUE + N_VALUES):
        i = value - FIRST_VALUE
        window_3d[:, :, i] = (window == value)

    return window_3d


def get_state_array(state, window_size, shape=None, dtype=None):
    """
    Get the state 3D array, reshaped to the given shape and converted to the desired dtype.
    :param state: The state, containing the board, head position and direction.
    :param window_size: An odd integer describing the window size.
    :param shape: The desired shape, e.g. in order to get a column vector give shape (-1, 1).
    :param dtype: The desired dtype, e.g. to multiply with floating-point matrix give float32.
    :return: A NumPy array of with the given shape and dtype, describing the state of the game.
    """
    board, head = state
    head_position, head_direction = head

    state_array = get_window_3d(board, head_position, head_direction, window_size)

    if shape is not None:
        state_array = state_array.reshape(shape)

    if dtype is not None:
        state_array = state_array.astype(dtype)

    return state_array


class ReplayMemory:
    """
    This class represents the Replay-Memory, which holds the last N experiences of the game,
    and enables sampling from them and appending efficiently.
    """

    def __init__(self, state_shape, max_size, sampling_method='uniform', smallest_weight=0.01):
        """
        Initialize the object with empty NumPy arrays of the given sizes.
        :param max_size: The maximal number of experiences to hold.
        :param state_shape: The shape of the state, e.g. (window_size, window_size, N_VALUES).
        :param sampling_method: How to sample the mini-batch of experiences from the memory.
                                Should be a string in {'uniform', 'nonzero_reward', 'positive_reward'}.
        :param smallest_weight: The smallest weight (which is then converted to probability) to give the experiences
                                we wish to sample less (or not at all, if it's 0).
        """
        self.max_size = max_size
        self.sampling_method = sampling_method
        self.smallest_weight = smallest_weight

        # This is the current size of the ReplayMemory, which is 0 at the beginning until it reaches max_size.
        self.size = 0

        # This is the current index to insert the next experience, it is growing in a cyclic way.
        self.curr_index = 0

        # These are NumPy array holding the actual experiences.
        # Note that the states are stored in their original boolean values, to reduce memory usage.
        self.states = np.empty(shape=(max_size,) + state_shape, dtype=np.bool)
        self.actions = np.empty(shape=(max_size,), dtype=np.int)
        self.rewards = np.empty(shape=(max_size,), dtype=np.float32)
        self.next_states = np.empty(shape=(max_size,) + state_shape, dtype=np.bool)

    def append(self, state, action, reward, next_state):
        """
        Append a new experience to the memory.
        The new experience is stored at the curr_index, which means it replaces the experience that was already there.
        Therefore this data-structure mimics a queue (first-in-first-out).
        :param state: The state.
        :param action: The action.
        :param reward: The resulting reward.
        :param next_state: The resulting new state.
        """
        self.states[self.curr_index] = state
        self.actions[self.curr_index] = action
        self.rewards[self.curr_index] = reward
        self.next_states[self.curr_index] = next_state

        self.size = min(self.size + 1, self.max_size)
        self.curr_index = (self.curr_index + 1) % self.max_size

    def get_probabilities(self):
        """
        Get a vector of probabilities to sample the mini-batch,
        according to the sampling-method the object was initialized with.
        :return: A probability vector with self.size dimensions.
        """
        weights = np.ones(shape=self.size, dtype=np.float32)

        if self.sampling_method == 'nonzero_reward':
            weights[self.rewards[:self.size] == 0] = self.smallest_weight
        elif self.sampling_method == 'positive_reward':
            weights[self.rewards[:self.size] <= 0] = self.smallest_weight
        elif not self.sampling_method == 'uniform':
            raise ValueError("Unknown sampling_method given to ReplayMemory ({}).".format(self.sampling_method))

        weights_sum = weights.sum()

        # It might be the cases where all the weights are zero.
        # In this extreme case, return a uniform sampling weights.
        if weights_sum > 0:
            return weights / weights_sum
        else:
            return np.ones(shape=self.size, dtype=np.float32) / self.size

    def sample(self, sample_size):
        """
        Sample a mini-batch of experiences from the memory.
        As long as there are some experiences the sampling is possible (because it's done with replacements).
        However, it's not possible to call this method with an empty memory (i.e. self.size == 0).
        :param sample_size: The size of the mini-batch to sample.
        :return: A tuple with four elements, each containing a mini-batch of the corresponding variable
                 (state, action, reward and next-state).
        """
        assert self.size > 0, "Can not sample from an empty ReplayMemory."

        indices = np.random.choice(self.size, size=sample_size, p=self.get_probabilities())

        sampled_states = self.states[indices]
        sampled_actions = self.actions[indices]
        sampled_rewards = self.rewards[indices]
        sampled_next_states = self.next_states[indices]

        return sampled_states, sampled_actions, sampled_rewards, sampled_next_states

    def __len__(self):
        """
        :return: The length of the memory, which is the amount of experiences in it (which is different from max_size).
        """
        return self.size


class QFunction(keras.Model):
    """
    A model trying to approximate the Q-function for every action a, given a state as an input.
    """
    def __init__(self):
        super(QFunction, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=16, kernel_size=3, activation=tf.nn.relu)
        self.conv2 = keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu)
        self.conv3 = keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu)
        self.flatten = keras.layers.Flatten()
        self.affine1 = keras.layers.Dense(units=128, activation=tf.nn.relu)
        self.affine2 = keras.layers.Dense(units=N_ACTIONS)

    def __call__(self, inputs, *args, **kwargs):
        x = inputs

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.flatten(x)

        x = self.affine1(x)
        x = self.affine2(x)

        return x


class DQN(bp.Policy):
    """
    An agent implementing Deep Q-Learning.
    Reference:
    "Human-level control through deep reinforcement learning"
    https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
    """
    params = {
        'epsilon': 0.1,  # epsilon controls the exploration:
                         # when acting, with probability epsilon select random action.

        'decay_epsilon': True,  # How much to decay epsilon during the game.
                                # The current epsilon will be initialized to epsilon, and decay linearly
                                # to zero at the beginning of the score-scope.

        'gamma': 0.96,  # gamma controls the reward decaying:
                        # how much to prefer current reward over future rewards.

        'bs': 32,  # The mini-batch size to sample from the replay-memory
                   # in each training iteration.

        'window_size': 13,  # The size of the squared window to take around the head of the snake.

        'memory_size': 1000,  # The size of the replay-memory - how many past experiences to keep.

        'update_target_interval': 100,  # How frequently should we update the weights of the target Q-function.

        'sampling_method': 'uniform',  # How to sample mini-batches from the ReplayMemory.
        'smallest_weight': 0.5,        # The minimal weight (probability) to give samples we wish to
                                       # sample less (or not at all, if it's zero)

        'double_dqn': False,  # Whether to implement Double-Deep-Q-Learning.
                              # Reference:
                              # "Deep Reinforcement Learning with Double Q-learning"
                              # https://arxiv.org/pdf/1509.06461.pdf

        # TODO remove these lines - used for debugging purposes...
        # TODO --------------------------------------------------
        'agent_name': ''
        # TODO --------------------------------------------------
    }

    def cast_string_args(self, policy_args):
        # Initial the parameters according to their default values,
        # or according to the given arguments from the command-line.
        for param_name in DQN.params.keys():
            default_value = DQN.params[param_name]
            param_type = type(default_value)

            # If the parameter does not exist in the policy_args dictionary, add it with its default value.
            if param_name not in policy_args:
                policy_args[param_name] = default_value

            # If the parameter already exists in the policy_args dictionary, cast it to its correct type.
            policy_args[param_name] = param_type(policy_args[param_name])

        return policy_args

    def init_run(self):
        # TODO remove these lines - used for debugging purposes...
        # TODO --------------------------------------------------
        self.rewards = list()
        self.losses = list()

        # This is the sum of rewards among the last iterations, in order to print.
        self.reward_sum = 0
        # TODO --------------------------------------------------

        # This is the current epsilon, which defined the exploration probability.
        # It might change during time (e.g. decay).
        self.curr_epsilon = self.epsilon

        # This is the number of exploration steps, which are all steps except the score scope.
        self.exploration_steps = self.game_duration - self.score_scope

        # This is the replay-memory containing the past experiences (up to a maximal size).
        self.replay_memory = ReplayMemory(state_shape=(self.window_size, self.window_size, N_VALUES),
                                          max_size=self.memory_size,
                                          sampling_method=self.sampling_method,
                                          smallest_weight=self.smallest_weight)

        # Initialize the two Q-functions - one is the actual Q-function to be learned, and the second is the target
        # Q-function that will be fixed and its weights will be updated every self.update_target_interval iterations.
        # (this is inspired by the original DQN paper).
        self.q_function = QFunction()
        self.target_function = QFunction()

        self.q_function.compile(optimizer='adam', loss='mse')
        self.target_function.compile(optimizer='adam', loss='mse')

        # Avoid "lazy evaluation" in keras, which will cause the model to build its
        # training-function and predict-function in the first learning iterations.
        # This will speed up the first steps in the learning process.
        dummy_samples = np.zeros(shape=(self.bs, self.window_size, self.window_size, N_VALUES), dtype=np.float32)
        dummy_targets = np.zeros(shape=(self.bs, N_ACTIONS), dtype=np.float32)
        for model in [self.q_function, self.target_function]:
            # Predict on the dummy data, to make to model build its predict-function.
            model.predict(x=dummy_samples, batch_size=self.bs)

            # train on the dummy data, to make to model build its train-function.
            # Note that the inputs and the targets are all zeros so the gradients are also zeros and this "learning"
            # will not affect the initial weights (and even if it does,
            # they are random initial weights so it does not really matter)
            model.train_on_batch(x=dummy_samples, y=dummy_targets)

    def adjust_epsilon(self, curr_round):
        """
        Adjust the current epsilon according to the game's round number.
        The current epsilon will be initialized to epsilon, and decay linearly
        to zero at the beginning of the score-scope.
        :param curr_round: The current round-number of the game.
        """
        # If the current round number is larger than the number of exploration steps,
        # stop exploring (by setting epsilon to 0)
        if curr_round >= self.exploration_steps:
            self.curr_epsilon = 0
        else:
            # If the current round number is smaller than the number of exploration steps,
            # set the current epsilon to do down linearly from the its initial value to 0.
            self.curr_epsilon = self.epsilon * (1 - float(curr_round) / float(self.exploration_steps))

    def get_training_batch(self):
        """
        Sample a mini-batch of experiences from the replay-memory, and adjust the tensors accordingly
        to enable feeding them to the model without further processing.
        The processing contains two things:
            (*) Converting the states tensors to float32.
                (the replay memory stores them in their original boolean values to save space).
            (*) Building the target tensor for the model, which is a NumPy array of shape (batch-size, N_ACTIONS),
                which is the same as the network's output tensor.
                In each row (corresponding to a single training-sample) adjust the label of the corresponding action
                (which is the action that was taken by the agent in that state) to be the addition of
                (-) The corresponding reward (which is the reward that was given to the agent in that state when
                    performing the action).
                (-) The maximal Q(s_{t+1},a') for action a' (according to the target Q-function) multiplied by gamma.
        :return: Two NumPy arrays, one is the input to the model and the other is the labels,
        """
        # Get a mini-batch of states, action, rewards and next_states from the replay-memory.
        states, actions, rewards, next_states = self.replay_memory.sample(self.bs)

        # Convert the states to float32 to feed the models.
        states = states.astype(np.float32)
        next_states = next_states.astype(np.float32)

        # Get Q(s_{t+1},a) for all actions a, according to the target Q-function.
        next_states_actions_target_values = self.target_function.predict(next_states)

        if self.double_dqn:
            # Use the Q-function to select the greedy action - the action which maximizes the state-action value.
            next_states_actions_q_values = self.q_function.predict(next_states)
            next_states_argmax_action_value = next_states_actions_q_values.argmax(axis=1)

            # Evaluate the state-action value on this action using the target network.
            next_states_max_action_value = next_states_actions_target_values[np.arange(self.bs),
                                                                             next_states_argmax_action_value]
        else:
            # Get the maximal Q(s_{t+1},a) for an action a'.
            next_states_max_action_value = next_states_actions_target_values.max(axis=1)

        # Initialize the targets tensor to be the output of the q-function model.
        targets = self.q_function.predict(states)

        # For each training-sample, adjust the label of the corresponding action accordingly
        # (see the function's documentation for more details).
        targets[np.arange(self.bs), actions] = rewards + self.gamma * next_states_max_action_value

        # Convert the targets tensor to float32, to feed to the model.
        targets = targets.astype(np.float32)

        return states, targets

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if self.decay_epsilon:
            self.adjust_epsilon(round)

        # Verify that the ReplayMemory is not empty.
        # Should not happen, but it might if the agent's learn function will be called at the beginning of the game.
        if len(self.replay_memory) > 0:
            # Get a mini-batch of experiences and train the Q-function model on this mini-batch.
            inputs, targets = self.get_training_batch()
            loss = self.q_function.train_on_batch(inputs, targets)

        # Every self.update_target_interval iterations,
        # update the target Q-function network weights to match the Q-function weights.
        if round % self.update_target_interval == 0:
            self.target_function.set_weights(self.q_function.get_weights())

        # TODO remove these lines - used for debugging purposes...
        # TODO --------------------------------------------------
        self.losses.append(loss)
        if round % 1000 == 0:
            self.log("Q-Function network's loss = " + str(loss), 'VALUE')

            if round > self.game_duration - self.score_scope:
                self.log("Rewards in last 1000 rounds which counts towards the score: " + str(self.reward_sum), 'VALUE')
            else:
                self.log("Rewards in last 1000 rounds: " + str(self.reward_sum), 'VALUE')

            self.reward_sum = 0
        else:
            self.reward_sum += reward
        # TODO --------------------------------------------------

    def select_action(self, state):
        """
        Select an action which maximizes Q(state,action) according to the Q-function model and the given state.
        :param state: The state to select the action for.
        :return: The selected action ('L', 'R' or 'F').
        """
        # Convert the state to the corresponding tensor.
        state_array = get_state_array(state, self.window_size, dtype=np.float32)

        # Add a batch dimension of size 1.
        state_array = state_array.reshape(1, *state_array.shape)

        # Calculate Q(state,action) for all actions.
        state_actions_values = self.q_function.predict(state_array, batch_size=1).flatten()

        # Select the action which maximizes Q(state,action).
        return bp.Policy.ACTIONS[np.argmax(state_actions_values)]

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        # Do not add experiences containing None (this can happen in the first round of the game).
        if (prev_state is not None) and (prev_action is not None) and (reward is not None) and (new_state is not None):
            self.replay_memory.append(state=get_state_array(prev_state, self.window_size),
                                      action=bp.Policy.ACTIONS.index(prev_action),
                                      reward=reward,
                                      next_state=get_state_array(new_state, self.window_size))

        # TODO remove these lines - used for debugging purposes...
        # TODO --------------------------------------------------
        self.rewards.append(reward)
        if round == self.game_duration - 1:
            time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            s = '{}-id{}'.format(self.agent_name, self.id)

            out_dir = os.path.join('logs', time_str)
            pathlib.Path(out_dir).mkdir(exist_ok=True)

            np.array(self.losses, dtype=np.float32).tofile(os.path.join(out_dir, '{}_losses'.format(s)))
            np.array(self.rewards, dtype=np.float32).tofile(os.path.join(out_dir, '{}_rewards'.format(s)))

            with open(os.path.join(out_dir, '{}_params.json'.format(s)), 'w') as f:
                json.dump({k: self.__dict__[k] for k in self.__dict__.keys() if k in DQN.params.keys()}, f)
        # TODO --------------------------------------------------

        # Exploration, with probability epsilon.
        if np.random.rand() < self.curr_epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        # Exploitation, with probability (1 - epsilon).
        return self.select_action(new_state)
