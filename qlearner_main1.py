__author__ = 'philippe'
import World
import threading
import time
import random

# parameter to discount future rewards
discount = 0.3

# epsilon-greedy policy parameter (probability of picking a random action)
epsilon = 0.1

actions = World.actions
states = []
Q = {}
for i in range(World.x):
    for j in range(World.y):
        states.append((i, j))

for state in states:
    temp = {}
    for action in actions:
        temp[action] = 0.1
        World.set_cell_action_score(state, action, temp[action])
    Q[state] = temp

for (i, j, c, w) in World.specials:
    for action in actions:
        Q[(i, j)][action] = w
        World.set_cell_action_score((i, j), action, w)


def do_action(action):
    s = World.player
    r = -World.score
    if action == actions[0]:
        World.try_move(0, -1)
    elif action == actions[1]:
        World.try_move(0, 1)
    elif action == actions[2]:
        World.try_move(-1, 0)
    elif action == actions[3]:
        World.try_move(1, 0)
    else:
        return
    s2 = World.player
    r += World.score
    return s, action, r, s2


def policy(max_act):
    if random.random() > epsilon:
        return max_act
    else:
        random_idx = random.randint(0, len(actions) - 2)
        if actions[random_idx] == max_act:
            return actions[len(actions) - 1]
        else:
            return actions[random_idx]


def max_Q(s):
    val = None
    act = None
    for a, q in Q[s].items():
        """
        For given state s:
            loop through all the (action and q) values for a given state
            and select the action that is associated with the highest q value
        """
        if val is None or (q > val):
            val = q
            act = a
    return act, val


def inc_Q(s, a, alpha, inc):
    Q[s][a] *= 1 - alpha
    Q[s][a] += alpha * inc
    World.set_cell_action_score(s, a, Q[s][a])


def run():
    global discount
    time.sleep(1)
    alpha = 1
    t = 1
    while True:
        # print("Q: {}".format(Q))
        # Pick the right action
        s = World.player
        max_act, max_val = max_Q(s)
        chosen_act = policy(max_act)
        (s, a, r, s2) = do_action(chosen_act)

        # TODO XXX ask philippe:
        # 1. Why do we calculate max_Q on the new state?
        # 2. Is this temporal difference learning?
        # 3. What is alpha ?

        # If alpha = 1, you ignore your previous Q value, and completely accept your new Q value
        # (1-alpha)Q(s,a) + alpha[r + gamma max_{a'}(Q(s',a'))]
        # If alpha = 0, keep previous Q value
        # 4. What is the -0.1 learning rate update step value
        #    -- Why are we pow(t..)?

        # "Expected reward RL" --> Where they ignore the gamma and take expectation instead

        # . 5 can learn in completely online fashion (sleep wake)
        # --> non-episodic rl

        # Maybe policy search or policy gradient is better for us
        # --> This might be better for game playinig (poker)

        # google "on policy" --> sarsa can't re-learn from dataset
        #
        #   therefore
        #
        # q-learning  is a better approach

        # Update Q
        max_act, max_val = max_Q(s2)
        increment = r + discount * max_val
        inc_Q(s, a, alpha, increment)

        # Check if the game has restarted
        t += 1.0
        if World.has_restarted():
            World.restart_game()
            time.sleep(0.01)
            t = 1.0

        # Update the learning rate
        alpha = pow(t, -0.1)

        # MODIFY THIS SLEEP IF THE GAME IS GOING TOO FAST.
        time.sleep(0.01)


t = threading.Thread(target=run)
t.daemon = True
t.start()
World.start_game()
