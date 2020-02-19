from policies import base_policy as bp
import numpy as np

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


def get_other_window(last_window, next_window, last_action):
    """
    Returns a tuple of the 2 windows the agent didn't choose to go to. For example, if we
    had state s_t and the agent chose to go straight, the function will return the states s_t+1
    for the case the agent chose to go right and for the case it chose to go left.
    :param last_window: a numpy array indicating the state before the action
    :param next_window: a numpy array indicating the state after the action
    :param last_action: the action the snake took between the states
    :return: tuple adjusted for appending into the replay memory
    """
    assert last_action != "F", "No symmetry if the snake went forward"
    oposite_action = {"L":"R", "R":"L"}
    prev_window = np.fliplr(last_window)
    new_window = np.fliplr(next_window)
    new_action = bp.Policy.ACTIONS.index(oposite_action[last_action])
    return prev_window, new_action, new_window


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


class LinearQLearning(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes. It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """

    params = {
        'epsilon': 0.25,  # epsilon controls the exploration:
                          # when acting, with probability epsilon select random action.

        'decay_epsilon': True,  # How much to decay epsilon during the game.
                                # The current epsilon will be initialized to epsilon, and decay linearly
                                # to zero at the beginning of the score-scope.

        'gamma': 0.95,  # gamma controls the reward decaying:
                        # how much to prefer current reward over future rewards.

        'lr': 0.02,  # The learning-rate.

        'decay_lr': 0,  # Decay the learning-rate by half, every decay_lr iterations (if 0 - do not decay at all).

        'bs': 64,  # The mini-batch size to sample from the replay-memory
                   # in each training iteration.

        'window_size': 13,  # The size of the squared window to take around the head of the snake.

        'memory_size': 500  # The size of the replay-memory - how many past experiences to keep.
    }

    def cast_string_args(self, policy_args):
        # Initial the parameters according to their default values,
        # or according to the given arguments from the command-line.
        for param_name in LinearQLearning.params.keys():
            default_value = LinearQLearning.params[param_name]
            param_type = type(default_value)

            # If the parameter does not exist in the policy_args dictionary, add it with its default value.
            if param_name not in policy_args:
                policy_args[param_name] = default_value

            # If the parameter already exists in the policy_args dictionary, cast it to its correct type.
            policy_args[param_name] = param_type(policy_args[param_name])

        return policy_args

    def init_run(self):
        # This is the sum of rewards among the last iterations, in order to print.
        self.reward_sum = 0

        # These are the weights-matrix and bias-vector of the linear function approximating the Q-function.
        # Initialize the weight-matrix with normal distribution centered at 0 with small variance,
        # and the bias-vector with zeros.
        self.w = np.random.normal(loc=0, scale=0.01, size=(N_ACTIONS, N_VALUES * self.window_size ** 2))
        self.b = np.zeros(shape=(3,))

        # This is the current epsilon, which defined the exploration probability.
        # It might change during time (e.g. decay).
        self.curr_epsilon = self.epsilon

        # This is the current epsilon, which defined the exploration probability.
        # It might change during time (e.g. decay).
        self.curr_lr = self.lr

        # This is the number of exploration steps, which are all steps except the score scope.
        self.exploration_steps = self.game_duration - self.score_scope

        # This is the replay-memory containing the past experiences (up to a maximal size).
        self.replay_memory = ReplayMemory(state_shape=(self.window_size, self.window_size, N_VALUES),
                                          max_size=self.memory_size)

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

    def delta(self, states, actions, rewards, next_states):
        """
        Calculate 'delta' - the difference between the Q(s,a) and the desired value
        (the reward plus gamma times the maximal Q-value for the next state).
        :param states: A NumPy array of shape (d, n) containing a mini-batch of n state, each in d dimensions.
                       These are the current states (at which the action was taken, the reward was received,
                       and the next-state was observed).
        :param actions: A NumPy vector containing n integers representing the actions.
        :param rewards: A NumPy vector containing n rewards.
        :param next_states: A NumPy array of shape (d, n) containing a mini-batch of n state, each in d dimensions.
                            These are the next-states the agent observed, after being the the corresponding states
                            and taking the corresponding action.
        :return: A NumPy vector containing n values that are the results of the calculation.
        """
        n = states.shape[-1]    # This is the number of samples in the mini-batch
        assert actions.shape == rewards.shape == (n,) and next_states.shape == states.shape

        # These two are NumPy arrays with shape N_ACTIONS x n.
        # Each column i is the state-action values Q(s_i,a) for the pairs (s_i, a) for all actions a.
        states_actions_values = np.dot(self.w, states) + self.b.reshape(-1, 1)
        next_states_actions_values = np.dot(self.w, next_states) + self.b.reshape(-1, 1)

        # This is a NumPy array of shape (n,) containing the state-action values Q(s_i,a_i) for i = 0,1,...,n-1.
        states_actions_value = states_actions_values[(actions, np.arange(n))]

        return states_actions_value - (rewards + self.gamma * np.max(next_states_actions_values, axis=0))

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        # Decay linearly to zero at the beginning of the score-scope.
        if self.decay_epsilon:
            self.adjust_epsilon(round)

        # Decay the learning-rate by half, every decay_lr iterations (if 0 - do not decay at all).
        if self.decay_lr != 0:
            if round % self.decay_lr == 0:
                self.curr_lr = self.lr * 0.5 ** (round // self.decay_lr)

        # Sample a mini-batch of experiences from the ReplayMemory.
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states = self.replay_memory.sample(self.bs)

        # Transpose the shape of the state from (bs, window_size, window_size, N_VALUES) to
        # (window_size, window_size, N_VALUES, bs), to enable reshaping each state to be a column.
        sampled_states = np.transpose(sampled_states, axes=(1, 2, 3, 0))
        sampled_next_states = np.transpose(sampled_next_states, axes=(1, 2, 3, 0))

        # Reshape them to the corresponding shape, to enable multiplying by the weight-matrix and adding the bias.
        sampled_states = sampled_states.reshape(-1, self.bs)
        sampled_next_states = sampled_next_states.reshape(-1, self.bs)

        # Calculate the deltas for all experiences in the mini-batch in a vectorized way.
        deltas = self.delta(sampled_states, sampled_actions, sampled_rewards, sampled_next_states)

        # Update the weight-matrix and bias according to the gradient of the mini-batch.
        coefficient = self.curr_lr * (float(1) / float(self.bs))
        for i in range(self.bs):
            action = sampled_actions[i]
            self.w[action, :] -= coefficient * deltas[i] * sampled_states[:, i]
            self.b[action] -= coefficient * deltas[i]

        if round % 1000 == 0:
            if round > self.game_duration - self.score_scope:
                self.log("Rewards in last 1000 rounds which counts towards the score: " + str(self.reward_sum), 'VALUE')
            else:
                self.log("Rewards in last 1000 rounds: " + str(self.reward_sum), 'VALUE')

            self.reward_sum = 0
        else:
            self.reward_sum += reward

    def select_action(self, state):
        state_vec = get_state_array(state, self.window_size, shape=(-1, 1), dtype=np.float32)
        state_actions_values = np.dot(self.w, state_vec).flatten()

        return bp.Policy.ACTIONS[np.argmax(state_actions_values)]

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if (prev_state is not None) and (prev_action is not None) and (reward is not None) and (new_state is not None):
            prev_window = get_state_array(prev_state, self.window_size)
            new_window = get_state_array(new_state, self.window_size)
            self.replay_memory.append(state=prev_window,
                                      action=bp.Policy.ACTIONS.index(prev_action),
                                      reward=reward,
                                      next_state=new_window)
            if prev_action != "F":  # we can use symmetries to add more experiences to the replay memory
               sym_state, sym_action, sym_next = get_other_window(prev_window, new_window, prev_action)
               self.replay_memory.append(state=sym_state,
                                      action=sym_action,
                                      reward=reward,
                                      next_state=sym_next)

        # Exploration, with probability epsilon.
        if np.random.rand() < self.curr_epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        # Exploitation, with probability (1 - epsilon).
        return self.select_action(new_state)
