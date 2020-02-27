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


def get_symmetric_windows(prev_window, prev_action, new_window):
    """
    Calculates all the symmetric windows and returns a list containing tuples of the following structure:
    (state, action, new_state), each one representing a rotation or a symmetry of the given windows.
    :param prev_window: the state at time t
    :param prev_action: the action the agent took
    :param new_window: the state at time t+1 after prev_action was taken
    :return: A list of tuples (state, action, new_state)

    """
    new_memories = list()

    if prev_action == "F":  # we can use rotations
        new_memories.append((np.rot90(prev_window, 3),
                             bp.Policy.ACTIONS.index("R"),
                             new_window))
        new_memories.append((np.rot90(prev_window, 1),
                             bp.Policy.ACTIONS.index("L"),
                             new_window))
    else:  # we can use symmetry along the advancement axis
        opposite_action = {"L": "R", "R": "L"}
        new_memories.append((np.fliplr(prev_window),
                             bp.Policy.ACTIONS.index(opposite_action[prev_action]),
                             np.fliplr(new_window)))

    return new_memories


class ReplayMemory:
    """
    This class represents the Replay-Memory, which holds the last N experiences of the game,
    and enables sampling from them and appending efficiently.
    """

    def __init__(self, state_shape, max_size, game_duration, sampling_method='uniform', smallest_weight=0.01,
                 prioritized_memory=False, epsilon=0.01, alpha=0.6, beta_0=0.4):
        """
        Initialize the object with empty NumPy arrays of the given sizes.
        :param state_shape: The shape of the state, e.g. (window_size, window_size, N_VALUES).
        :param max_size: The maximal number of experiences to hold.
        :param game_duration: The duration of the game.
                              Used to anneal beta linearly from beta_0 (at round #1) to 1 (at round #game_duration).

        :param sampling_method: How to sample the mini-batch of experiences from the memory.
                                Should be a string in {'uniform', 'nonzero_reward', 'positive_reward'}.
        :param smallest_weight: The smallest weight (which is then converted to probability) to give the experiences
                                we wish to sample less (or not at all, if it's 0).

        :param prioritized_memory: Whether to use Prioritized Experience Memory or not.
                                   The following arguments are hyper-parameters of the memory.
        :param epsilon: If prioritized_memory is True, this value indicates the amount to add to the absolute deltas in
                        order to avoid not sampling at all from experiences that have low delta.
                        This value's default is 0.01, following the original paper.
        :param alpha: If prioritized_memory is True, this value (between 0 and 1) indicates the exponent to raise the
                      priorities in order to obtain probabilities.
                      When this value is 0 it means no prioritization is done (a.k.a. uniform sampling).
                      This value's default is 0.6, following the original paper.
        :param beta_0: If prioritized_memory is True, this value (between 0 and 1) indicates the exponent to raise the
                       sample-weights (or "importance-sampling" as stated in the original paper).
                       This value's default is 0.4, following the original paper.
        """
        self.max_size = max_size
        self.sampling_method = sampling_method
        self.smallest_weight = smallest_weight
        self.game_duration = game_duration

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

        self.prioritized_memory = prioritized_memory

        self.deltas = np.zeros(shape=(max_size,), dtype=np.float32) if prioritized_memory else None
        self.epsilon = epsilon if prioritized_memory else None
        self.alpha = alpha if prioritized_memory else None
        self.beta_0 = beta_0 if prioritized_memory else None

    def append(self, state, action, reward, next_state):
        """
        Append a new experience to the memory.
        The new experience is stored at the curr_index, which means it replaces the experience that was already there.
        Therefore this data-structure mimics a queue (first-in-first-out).
        If we use a Prioritized Experience Memory, this experience is added with maximal delta
        (to make sure it will be sampled at least once, and its delta will be updated accordingly).
        :param state: The state.
        :param action: The action.
        :param reward: The resulting reward.
        :param next_state: The resulting new state.
        """
        self.states[self.curr_index] = state
        self.actions[self.curr_index] = action
        self.rewards[self.curr_index] = reward
        self.next_states[self.curr_index] = next_state

        # If we use a Prioritized Experience Memory, this experience is added with maximal delta
        # (to make sure it will be sampled at least once, and its delta will be updated accordingly).
        if self.prioritized_memory:
            # max with epsilon to avoid being zero in the beginning of the learning
            # (when all deltas are zero since they were not updated yet).
            self.deltas[self.curr_index] = max(self.deltas.max(), self.epsilon)

        self.curr_index = (self.curr_index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

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
        else:
            raise ValueError("Unknown sampling_method given to ReplayMemory ({}).".format(self.sampling_method))

        weights_sum = weights.sum()

        # It might be the cases where all the weights are zero.
        # In this extreme case, return a uniform sampling weights.
        if weights_sum > 0:
            return weights / weights_sum
        else:
            return np.ones(shape=self.size, dtype=np.float32) / self.size

    def get_prioritized_probabilities(self, round_number):
        """
        Get a vector of probabilities to sample the mini-batch,
        according to the sampling-method the object was initialized with.
        :param round_number: The round number.
                             Used to anneal beta linearly from beta_0 (at round #1) to 1 (at round #game_duration).
        :return: A probability vector and weights vector, both with self.size dimensions,
        """
        # Anneal beta linearly from beta_0 (at round #1) to 1 (at round #game_duration).
        beta = self.beta_0 + round_number * ((1 - self.beta_0) / self.game_duration)

        # Initialize the priorities to be the deltas up to the current size of the memory.
        priorities = self.deltas[:self.size]

        # Add epsilon in order to avoid not sampling at all from experiences that have low delta,
        # and raise to the power of alpha.
        priorities_pow_alpha = (priorities + self.epsilon) ** self.alpha

        # Normalize it to be a probabilities vector.
        # Note that the sum will never be 0 because we add epsilon which is positive.
        probabilities = priorities_pow_alpha / priorities_pow_alpha.sum()

        # Define the sample-weights (or "importance-sampling" as stated in the original paper).
        weights = (self.size * probabilities) ** (-beta)

        # Normalize it to be a probabilities vector.
        weights /= weights.max()

        return probabilities, weights

    def update_deltas(self, indices, deltas):
        """
        Update the deltas of the experiences in the corresponding indices.
        :param indices: The indices of the experiences.
        :param deltas: The new deltas to update.
        """
        self.deltas[indices] = deltas

    def sample(self, sample_size, round_number):
        """
        Sample a mini-batch of experiences from the memory.
        The sampling is done according to the parameters that this memory was initialized with.
        It can be sampling more from positive/non-zero rewards, or according to the prioritized memory approach.
        :param sample_size: The size of the mini-batch to sample.
        :param round_number: The round number.
                             Used to anneal beta linearly from beta_0 (at round #1) to 1 (at round #game_duration).
        :return: A tuple with 6 elements.
                 The first 4 are the mini-batch (state, action, reward and next-state).
                 The last 2 are the indices of the sampled experiences, as well as sample-weight.
                 The last 2 values are used for the prioritized memory approach.
        """
        weights = None

        if self.prioritized_memory:
            probabilities, weights = self.get_prioritized_probabilities(round_number)
            indices = np.random.choice(self.size, size=sample_size, p=probabilities)
        elif self.sampling_method == 'uniform':
            indices = np.random.choice(self.size, size=sample_size)
        else:
            probabilities = self.get_probabilities()
            indices = np.random.choice(self.size, size=sample_size, p=probabilities)

        sampled_states = self.states[indices]
        sampled_actions = self.actions[indices]
        sampled_rewards = self.rewards[indices]
        sampled_next_states = self.next_states[indices]

        return sampled_states, sampled_actions, sampled_rewards, sampled_next_states, indices, weights

    def __len__(self):
        """
        :return: The length of the memory, which is the amount of experiences in it
        (which is different from max_size at the beginning of the training, until it reaches max_size).
        """
        return self.size


def get_window_pair(state, next_state, window_size):
    """
    Returns two windows, from the previous and the current round,
    centered at the player's head at the previous round.
    :param state: The previous state, containing the board and the head.
    :param next_state: The next state, containing the board and the head.
    :param window_size: The size of the window.
    :return: Two NumPy arrays containing the windows.
    """
    board, head = state
    head_position, head_direction = head

    next_board, next_head = next_state

    window = get_window(board, head_position, head_direction, window_size)
    next_window = get_window(next_board, head_position, head_direction, window_size)

    return window, next_window


def died_snake(state, next_state, window_size, player_id):
    """
    Checks if another snake died when touching the player's snake
    between the previous round and the current one.
    :param state: The previous state, containing the board and the head.
    :param next_state: The next state, containing the board and the head.
    :param window_size: The size of the window.
    :param player_id: The id if the player, to ignore changes involving the player's snake.
    :return: True if another snake died crashing on the player's snake, False otherwise.
    """
    # The size of the minimal snake - this will define the size of a 'chain' of cells with the same values
    # to be regarded as a snake. This is set to 3 because in the original version of the game, snakes are initialized
    # with size 3, and if it'll change it will not be a disaster...
    minimal_snake_length = 3

    window, next_window = get_window_pair(state, next_state, window_size)
    n_rows, n_cols = window.shape

    # Create a mask indicating where is the player's snake.
    player_location = (next_window == player_id)

    # Create a mask indicating the differences between the window in the current state and the previous one.
    diffs = (window != next_window) & (window != player_id) & (next_window != player_id)

    # From this point onwards, we treat our boolean matrix 'diffs' as a discrete graph - the ones are the vertices,
    # and two vertices are connected if they are at the left/right/bottom/top of each other.

    # Ignore isolated vertices - cells containing 1 but none of the neighbors contain 1.
    diffs[1:-1, 1:-1] &= (np.roll(diffs, shift=1, axis=1)[1:-1, 1:-1] | np.roll(diffs, shift=-1, axis=1)[1:-1, 1:-1] |
                          np.roll(diffs, shift=1, axis=0)[1:-1, 1:-1] | np.roll(diffs, shift=-1, axis=0)[1:-1, 1:-1])

    # Find the indices where there are ones that are connected to others
    # (i.e. they belong to a connected-component of size at least 2).
    indices = np.where(diffs)
    indices_tuples = list(zip(indices[0], indices[1]))

    # Create an array that will holds the ids of the vertices.
    ids = np.zeros_like(diffs, dtype=np.int)

    # The id is initialized to 0 and increments at every new connected-component.
    curr_id = 0

    # Go over all non-visited vertices in the graph, starting DFS from each one
    # and visit the entire connected-component.
    for i, j in indices_tuples:
        # This condition verifies that this vertex was not already marked with an id.
        if ids[i, j] != 0:
            continue

        curr_id += 1    # This will be the id of the current connected-component.
        connected_component_size = 0
        touching_player = False     # Will be True if the any vertex is neighbor to the player's snake.

        # Store the next vertices to process in a stack, therefore performing depth-first-search (DFS).
        stack = [(i, j)]
        while len(stack) > 0:
            row, col = stack.pop()
            connected_component_size += 1

            ids[row, col] = curr_id     # Mark the vertex with the id of the connected-component.

            # Go over the 4 possible neighbors.
            # for neighbor in [(row, col + 1), (row, col - 1), (row - 1, col), (row + 1, col)]:
            for neighbor in [(row, (col + 1) % n_cols), (row, (col - 1) % n_cols),
                             ((row - 1) % n_rows, col), ((row + 1) % n_rows, col)]:
                neighbor_i, neighbor_j = neighbor

                # If the neighbor is out-of-bounds, continue.
                if not ((neighbor_i < n_rows) and (neighbor_i >= 0) and
                        (neighbor_j < n_cols) and (neighbor_j >= 0)):
                    continue

                # If the neighbor is the player's snake, mark it.
                if player_location[neighbor_i, neighbor_j]:
                    touching_player = True

                # If the neighbor has not been marked, append it to the stack to be processed later.
                if diffs[neighbor_i, neighbor_j] and (ids[neighbor_i, neighbor_j] == 0):
                    stack.append(neighbor)

            # If the connected-component size is greater than the size of a snake, and one of the vertices
            # is neighbor to the player's snake, return True.
            if connected_component_size >= minimal_snake_length and touching_player:
                return True

    return False    # No sufficiently big connected-component touching the player was found.


class DuelQFunction(keras.Model):
    """
    A model trying to approximate the Q-function for every action a, given a state as an input.
    """

    def __init__(self,
                 conv1_channels, conv1_kernel_size, conv1_stride,
                 conv2_channels, conv2_kernel_size, conv2_stride,
                 conv3_channels, conv3_kernel_size, conv3_stride, conv3_on,
                 affine_channels):
        super(DuelQFunction, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=conv1_channels, kernel_size=conv1_kernel_size, activation='relu', strides=conv1_stride)
        self.conv2 = keras.layers.Conv2D(filters=conv2_channels, kernel_size=conv2_kernel_size, activation='relu', strides=conv2_stride)
        
        if conv3_on == 1:
            self.conv3 = keras.layers.Conv2D(filters=conv3_channels, kernel_size=conv3_kernel_size, activation='relu', strides=conv3_stride)
        else:
            self.conv3 = None

        self.flatten = keras.layers.Flatten()
        self.advantage_affine1 = keras.layers.Dense(units=affine_channels // 2, activation='relu')
        self.advantage_affine2 = keras.layers.Dense(units=N_ACTIONS)
        self.value_affine1 = keras.layers.Dense(units=affine_channels // 2, activation='relu')
        self.value_affine2 = keras.layers.Dense(units=1)
        self.get_mean = keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=-1, keepdims=True))
        self.repeat_value = keras.layers.RepeatVector(N_ACTIONS)
        self.flatten_value = keras.layers.Flatten()
        self.repeat_mean = keras.layers.RepeatVector(N_ACTIONS)
        self.flatten_mean = keras.layers.Flatten()
        self.subtract_mean = keras.layers.Subtract()
        self.add_value_advantage = keras.layers.Add()

    def call(self, inputs, *args, **kwargs):
        x = inputs

        x = self.conv1(x)
        x = self.conv2(x)
        
        if self.conv3 is not None:
            x = self.conv3(x)

        x = self.flatten(x)

        # after the convolution we split into two streams, one of the advantage function and one of the value function
        v = self.value_affine1(x)  # value
        v = self.value_affine2(v)
        v = self.repeat_value(v)  # take scalar and repeat to match N_ACTIONS
        v = self.flatten_value(v)

        # calculate A, mean(A)
        a = self.advantage_affine1(x)  # advantage
        a = self.advantage_affine2(a)
        a_mean = self.get_mean(a)  # mean(A)
        a_mean = self.repeat_mean(a_mean)  # take scalar and repeat to match N_ACTIONS
        a_mean = self.flatten_mean(a_mean)

        # we end by calculating the duel network objective V + (A - mean(A))
        q = self.subtract_mean([a, a_mean])
        q = self.add_value_advantage([v, q])

        return q


class QFunction(keras.Model):
    """
    A model trying to approximate the Q-function for every action a, given a state as an input.
    """

    def __init__(self,
                 conv1_channels, conv1_kernel_size, conv1_stride,
                 conv2_channels, conv2_kernel_size, conv2_stride,
                 conv3_channels, conv3_kernel_size, conv3_stride, conv3_on,
                 affine_channels):
        super(QFunction, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=conv1_channels, kernel_size=conv1_kernel_size, activation='relu', strides=conv1_stride)
        self.conv2 = keras.layers.Conv2D(filters=conv2_channels, kernel_size=conv2_kernel_size, activation='relu', strides=conv2_stride)
        
        if conv3_on == 1:
            self.conv3 = keras.layers.Conv2D(filters=conv3_channels, kernel_size=conv3_kernel_size, activation='relu', strides=conv3_stride)
        else:
            self.conv3 = None

        self.flatten = keras.layers.Flatten()
        self.affine1 = keras.layers.Dense(units=affine_channels, activation='relu')
        self.affine2 = keras.layers.Dense(units=N_ACTIONS)

    def __call__(self, inputs, *args, **kwargs):
        x = inputs

        x = self.conv1(x)
        x = self.conv2(x)
        
        if self.conv3 is not None:
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
        # epsilon controls the exploration:
        # when acting, with probability epsilon select random action.
        'epsilon': 0.5,

        # How much to decay epsilon during the game.
        # The current epsilon will be initialized to epsilon, and decay linearly
        # to zero at the beginning of the score-scope.
        'decay_epsilon': True,

        # gamma controls the reward decaying:
        # how much to prefer current reward over future rewards.
        'gamma': 0.96,

        # The mini-batch size to sample from the replay-memory
        # in each training iteration.
        'bs': 32,

        # The learning-rate to initialize the Adam optimizer with.
        'lr': 0.0005,

        # How many times to decay the learning-rate.
        'lr_n_changes': 2,

        # The factor to decay the learning-rate each time.
        'lr_decay_factor': 0.5,

        # Number of channels and convolution's kernel-sizes for the neural-network.
        'conv1_channels': 16,
        'conv1_kernel_size': 3,
        'conv1_stride': 1,
        'conv2_channels': 32,
        'conv2_kernel_size': 3,
        'conv2_stride': 1,
        'conv3_channels': 64,
        'conv3_kernel_size': 3,
        'conv3_stride': 1,
        'conv3_on': 1,
        'affine_channels': 128,

        'kernel_size': 3,

        # The size of the squared window to take around the head of the snake.
        'window_size': 17,

        # The size of the replay-memory - how many past experiences to keep.
        'memory_size': 1000,

        # How frequently should we update the weights of the target Q-function.
        'update_target_interval': 100,

        # How to sample mini-batches from the ReplayMemory.
        'sampling_method': 'uniform',

        # If it's greater than 0, give a reward when another snake dies crashing on our snake.
        'kill_snakes_reward': 0,

        # If it's greater than 0, give a reward when another snake dies crashing on our snake.
        'use_symmetric_experiences': True,

        # The minimal weight (probability) to give samples we wish to
        # sample less (or not at all, if it's zero)
        'smallest_weight': 0.5,

        # Whether to implement Double-Deep-Q-Learning.
        # Reference:
        # "Deep Reinforcement Learning with Double Q-learning"
        # https://arxiv.org/pdf/1509.06461.pdf
        'double_dqn': True,

        # Whether to implement Prioritized-Experience-Memory.
        # Reference:
        # "PRIORITIZED EXPERIENCE REPLAY"
        # https://arxiv.org/pdf/1511.05952.pdf
        'prioritized_memory': False,

        # Whether to implement Dueling DQN.
        # Reference:
        # "Dueling Network Architectures for Deep Reinforcement Learning"
        # https://arxiv.org/pdf/1511.06581.pdf
        'duel_dqn': False,
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
        self.kills = list()
        # This is the sum of rewards among the last iterations, in order to print.
        self.reward_sum = 0
        # TODO --------------------------------------------------

        # This is the current epsilon, which defined the exploration probability.
        # It might change during time (e.g. decay).
        self.curr_epsilon = self.epsilon

        # This is the number of exploration steps, which are all steps except the score scope.
        self.exploration_steps = self.game_duration - self.score_scope

        # If we add symmetric experiences, we dot not want them to take the space of actual experiences,
        # so we increase the memory-size  by 3 (because at most 2 symmetric experiences are added per real experience).
        if self.use_symmetric_experiences:
            self.memory_size *= 3

        # This is the replay-memory containing the past experiences (up to a maximal size).
        self.replay_memory = ReplayMemory(state_shape=(self.window_size, self.window_size, N_VALUES),
                                          max_size=self.memory_size,
                                          game_duration=self.game_duration,
                                          sampling_method=self.sampling_method,
                                          smallest_weight=self.smallest_weight,
                                          prioritized_memory=self.prioritized_memory)

        # Initialize the two Q-functions - one is the actual Q-function to be learned, and the second is the target
        # Q-function that will be fixed and its weights will be updated every self.update_target_interval iterations.
        # (this is inspired by the original DQN paper).
        model_class = DuelQFunction if self.duel_dqn else QFunction
        model_args = (self.conv1_channels, self.conv1_kernel_size, self.conv1_stride,
                      self.conv2_channels, self.conv2_kernel_size, self.conv2_stride,
                      self.conv3_channels, self.conv3_kernel_size, self.conv3_stride,  self.conv3_on,
                      self.affine_channels)

        self.q_function = model_class(*model_args)
        self.target_function = model_class(*model_args)

        for model in [self.q_function, self.target_function]:
            if self.lr_n_changes > 0:
                lr_boundaries = list(np.linspace(start=self.game_duration / (self.lr_n_changes + 1),
                                                 stop=self.game_duration, num=self.lr_n_changes,
                                                 endpoint=False, dtype=np.float32))
                lr_values = list(self.lr * (self.lr_decay_factor ** np.arange(self.lr_n_changes + 1, dtype=np.float32)))
                lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(lr_boundaries, lr_values)
            else:
                lr_schedule = self.lr

            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
            model.compile(optimizer=optimizer, loss='mse')

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

        self.target_function.set_weights(self.q_function.get_weights())

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
            self.curr_epsilon = self.epsilon * (1 - (float(curr_round) / float(self.exploration_steps)))

    def get_inputs_and_targets(self, states, actions, rewards, next_states):
        """
        Return the inputs and the targets in order to feed the neural-network.
        It also returns the prediction of the current model (self.q_function) on the previous-states, because it
        calculates these prediction anyway (to build the targets)
        and other might use this (i.e. the prioritized-memory).
        :param states: The current states.
        :param actions: The actions that were taken in the current states.
        :param rewards: The rewards received.
        :param next_states: The next states obtained.
        :return: Three NumPy arrays, the first is the inputs (current states arrays) the second is the targets
                 (target for the output of the model, which is of shape batch-size x N_ACTIONS), and the third
                 is the prediction of the model (shape batch-size x N_ACTIONS).
        """
        batch_size = states.shape[0]

        # Convert the states to float32 to feed the models.
        states = states.astype(np.float32)
        next_states = next_states.astype(np.float32)

        # Get Q(s_{t+1},a) for all actions a, according to the target Q-function.
        next_states_actions_target_values = self.target_function.predict(next_states, batch_size)

        if self.double_dqn:
            # Use the Q-function to select the greedy action - the action which maximizes the state-action value.
            next_states_actions_q_values = self.q_function.predict(next_states, batch_size)
            next_states_argmax_action_value = next_states_actions_q_values.argmax(axis=1)

            # Evaluate the state-action value on this action using the target network.
            next_states_max_action_value = next_states_actions_target_values[np.arange(batch_size),
                                                                             next_states_argmax_action_value]
        else:
            # Get the maximal Q(s_{t+1},a) for an action a'.
            next_states_max_action_value = next_states_actions_target_values.max(axis=1)

        # First predict using the Q-function on the current states.
        states_actions_target_values = self.q_function.predict(states, batch_size)

        # Initialize the targets tensor to be the output of the q-function model.
        targets = np.copy(states_actions_target_values)

        # For each training-sample, adjust the label of the corresponding action accordingly
        # (see the function's documentation for more details).
        targets[np.arange(batch_size), actions] = rewards + self.gamma * next_states_max_action_value

        # Convert the targets tensor to float32, to feed to the model.
        targets = targets.astype(np.float32)

        return states, targets, states_actions_target_values

    def get_training_batch(self, round_number):
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
        :return: Five NumPy arrays.
                 The first is the inputs to the model.
                 The second is the labels.
                 The third is the indices of the experiences sampled from the memory.
                 The forth is sample-weights of this mini-batch.
                 The fifth is the prediction of the current model on the inputs (used in the prioritized-memory).
        """
        # Get a mini-batch of states, action, rewards and next_states from the replay-memory.
        states, actions, rewards, next_states, indices, weights = self.replay_memory.sample(self.bs, round_number)

        inputs, targets, predictions = self.get_inputs_and_targets(states, actions, rewards, next_states)

        return inputs, targets, indices, weights, predictions

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if self.decay_epsilon:
            self.adjust_epsilon(round)

        # Verify that the ReplayMemory is not empty.
        # Should not happen, but it might if the agent's learn function will be called at the beginning of the game.
        if len(self.replay_memory) > 0:
            # Get a mini-batch of experiences and train the Q-function model on this mini-batch.
            inputs, targets, indices, weights, predictions = self.get_training_batch(round)

            # If we use prioritized-memory, we need to train with the corresponding sample_weight.
            train_kwargs = {'sample_weight': weights[indices]} if self.prioritized_memory else dict()
            loss = self.q_function.train_on_batch(inputs, targets, **train_kwargs)
            # TODO remove these lines - used for debugging purposes...
            # TODO --------------------------------------------------
            self.losses.append(loss)
            # TODO --------------------------------------------------

            # If we use prioritized-memory, update the deltas
            # (Temporal-Difference errors, which contributes to the priority of the experiences).
            if self.prioritized_memory:
                deltas = np.sum(np.abs(predictions - targets), axis=1)
                self.replay_memory.update_deltas(indices, deltas)
                # TODO why like this? Maybe because the sample_weight given to the train_on_batch function?
                # assert abs(loss - np.mean(deltas ** 2)) < 0.0001, "Wrong calculation of the loss."

        # Every self.update_target_interval iterations,
        # update the target Q-function network weights to match the Q-function weights.
        if round % self.update_target_interval == 0:
            self.target_function.set_weights(self.q_function.get_weights())

        # TODO remove these lines - used for debugging purposes...
        # TODO --------------------------------------------------
        if round % 1000 == 0:
            # self.log("Q-Function network's loss = " + str(loss), 'VALUE')
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
        # If the given experience contains None (this can happen in the first round of the game),
        # return a random action.
        if (prev_state is None) or (prev_action is None) or (reward is None) or (new_state is None):
            return np.random.choice(bp.Policy.ACTIONS)

        killed_snake = False

        if (self.kill_snakes_reward > 0) and died_snake(prev_state, new_state, self.window_size, self.id):
            # if died_snake(prev_state, new_state, self.window_size, self.id):
            # self.log("I killed a snake!", 'VALUE')
            reward += self.kill_snakes_reward
            killed_snake = True

        prev_state_array = get_state_array(prev_state, self.window_size)
        new_state_array = get_state_array(new_state, self.window_size)
        action_index = bp.Policy.ACTIONS.index(prev_action)

        self.replay_memory.append(prev_state_array, action_index, reward, new_state_array)

        # Add the symmetric experiences, if the flag was set to True.
        if self.use_symmetric_experiences:
            for memory in get_symmetric_windows(prev_state_array, prev_action, new_state_array):
                sym_state, sym_action, sym_next = memory  # unpack the tuple
                self.replay_memory.append(sym_state, sym_action, reward, sym_next)

        # TODO remove these lines - used for debugging purposes...
        # TODO --------------------------------------------------
        self.rewards.append(reward if not killed_snake else reward - self.kill_snakes_reward)
        self.kills.append(killed_snake)
        if round == self.game_duration - 1:
            time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            s = 'id{}'.format(self.id)

            out_dir = os.path.join('logs', time_str)

            try:
                os.makedirs(out_dir)
            except OSError:
                if not os.path.isdir(out_dir):
                    raise

            np.array(self.losses, dtype=np.float32).tofile(os.path.join(out_dir, '{}_losses'.format(s)))
            np.array(self.rewards, dtype=np.float32).tofile(os.path.join(out_dir, '{}_rewards'.format(s)))
            np.array(self.kills, dtype=np.float32).tofile(os.path.join(out_dir, '{}_kills'.format(s)))

            with open(os.path.join(out_dir, '{}_params.json'.format(s)), 'w') as f:
                json.dump({k: self.__dict__[k] for k in self.__dict__.keys() if k in DQN.params.keys()}, f)
        # TODO --------------------------------------------------

        # Exploration, with probability epsilon.
        if np.random.rand() < self.curr_epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        # Exploitation, with probability (1 - epsilon).
        return self.select_action(new_state)
