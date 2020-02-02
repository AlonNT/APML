from policies import base_policy as bp
import numpy as np

EPSILON = 0.05


def get_window(board, position, head, window_size):
    board_width, board_height = board.shape
    x = position[0]
    y = position[1]
    d = window_size // 2

    start_x = x - d
    end_x = x + d + 1
    start_y = y - d
    end_y = y + d + 1

    x_idx = np.arange(start_x, end_x) % board_width
    y_idx = np.arange(start_y, end_y) % board_height

    window = board[x_idx.reshape(-1, 1), y_idx]

    if head == 'N':
        k = 0
    elif head == 'S':
        k = 2
    elif head == 'E':
        k = 1
    elif head == 'W':
        k = 3
    else:
        raise ValueError("Unknown head.")

    return np.rot90(window, k)


# TODO


class TmpPolicy(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes. It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """

    def cast_string_args(self, policy_args):
        # TODO get args - gamma, lr, window_size
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        return policy_args

    def init_run(self):
        self.r_sum = 0

        # TODO initialize weight-matrix of size (3, window_size ** 2)
        # TODO initialize bias of size (3,)


    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        # TODO Implement the update-rule (slide 48)

        try:
            if round % 100 == 0:
                if round > self.game_duration - self.score_scope:
                    self.log("Rewards in last 100 rounds which counts towards the score: " + str(self.r_sum), 'VALUE')

                    self.log("****************************************", 'VALUE')
                    self.log(str(prev_state), 'VALUE')
                    self.log(str(prev_action), 'VALUE')
                    self.log(str(reward), 'VALUE')
                    self.log(str(too_slow), 'VALUE')
                    self.log("****************************************", 'VALUE')
                else:
                    self.log("Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')
                self.r_sum = 0
            else:
                self.r_sum += reward

        except Exception as e:
            self.log("Something Went Wrong...", 'EXCEPTION')
            self.log(e, 'EXCEPTION')

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        # board, head = new_state
        # head_pos, direction = head

        return 'L'
        # if np.random.rand() < self.epsilon:
        #     return np.random.choice(bp.Policy.ACTIONS)
        #
        # else:
        #     for a in list(np.random.permutation(bp.Policy.ACTIONS)):
        #
        #         # get a Position object of the position in the relevant direction from the head:
        #         next_position = head_pos.move(bp.Policy.TURNS[direction][a])
        #         r = next_position[0]
        #         c = next_position[1]
        #
        #         # look at the board in the relevant position:
        #         if board[r, c] > 5 or board[r, c] < 0:
        #             return a
        #
        #     # if all positions are bad:
        #     return np.random.choice(bp.Policy.ACTIONS)

