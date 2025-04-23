import numpy as np
import matplotlib.pyplot as plt

def epsilon_greedy_policy(state, model, epsilon, env):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        Q = model.predict(state.reshape(1, -1))
        return np.argmax(Q)

def print_policy_from_q(Q, env):
    action_arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    lake_map = env.unwrapped.desc.astype(str)
    n_rows, n_cols = lake_map.shape

    print("\nPolicy visualization:\n")
    for i in range(n_rows):
        row = ''
        for j in range(n_cols):
            tile = lake_map[i][j]
            state = i * n_cols + j

            if tile in ('H', 'G'):
                row += tile + ' '
            else:
                best_action = np.argmax(Q[state])
                row += action_arrows[best_action] + ' '
        print(row.strip())

def plot_policy_and_value(Q, V, env):
    V = np.max(Q, axis=1)
    action_arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    lake_map = env.unwrapped.desc.astype(str)
    n_rows, n_cols = lake_map.shape

    fig, ax = plt.subplots()
    V_reshaped = V.reshape((n_rows, n_cols))
    
    im = ax.imshow(V_reshaped, cmap=plt.cm.viridis)

    for i in range(n_rows):
        for j in range(n_cols):
            tile = lake_map[i][j]
            state = i * n_cols + j

            if tile == 'H':
                ax.text(j, i, 'H', ha='center', va='center', color='white', fontsize=16, fontweight='bold')
            elif tile == 'G':
                ax.text(j, i, 'G', ha='center', va='center', color='gold', fontsize=16, fontweight='bold')
            else:
                best_action = np.argmax(Q[state])
                arrow = action_arrows[best_action]
                ax.text(j, i, arrow, ha='center', va='center', color='white', fontsize=16)

    plt.title("Policy Arrows and State Values (V)")
    plt.colorbar(im, label='V[state]')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
