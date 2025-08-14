# Deep Q-Learning for Atari Breakout 🎮

This part of the repository contains an implementation of **Deep Q-Learning (DQN)** to play Atari's Breakout using reinforcement learning techniques.  It is generally based on the ideas presented in the seminal paper ["Playing Atari with Deep Reinforcement Learning"](https://arxiv.org/abs/1312.5602), adhering to the network structure and size, but with a few changes introduced, such as reward augumentation and a minor parameter tweeks.

Implementation utilizes the following structure in which two networks are used for evaluating the value function, to avoid the so-called 'moving target' issue, as is the common practice, as well as the replay buffer to avoid overfitting on earlier samples:

<img width="600" height="350" alt="DQNarchitecture-1" src="https://github.com/user-attachments/assets/f1315dd0-094f-4ec2-978c-30bb7fdc42ca" />

Structure of the network is a commonly used, consisting of 3 CNN layers, followed by a fully connected layers:

<img width="700" height="200" alt="1_yfrF2jnI3zspkZELq2rw9g" src="https://github.com/user-attachments/assets/cedb41ee-1b74-4f72-a386-a9de1bd748bf" />

Implementation also features a possibility to store and load model checkpoints, in case of timeout, if ran on limited resources, since an execution takes a fair ammount of time.

---

## 📌 Features
- **Deep Q-Network** Agent (4 CNN + 2 Dense layers)
- Uses **OpenAI Gym** Breakout environment
- Augumented reward (add reward for a score point, not only a won episode)
- Experience replay buffer
- Target network updates (avoid moving target updates) 
- Epsilon-greedy exploration strategy
- Training progress logging

---

## 🏗 Project Structure

```text
.
├── agent.py                   # DQN Agent class
├── main_test.py               # Training loop
├── replay_buffer.py           # Deque buffer class
├── environment.py             # Game environment definition 
├── check_buffer_content.py    # Debugging script which helps visualize and analyse the buffer content
├── animate_env.py             # Debugging script which helps visualize the agent behaviour
└── README.md
```

## Parameters

| Parameter              | Default Value                                            | Description |
|------------------------|----------------------------------------------------------|-------------|
| **alpha**              | `0.0005`                                                 | Learning rate for the optimizer. |
| **gamma**              | `0.95`                                                   | Discount factor for future rewards. |
| **epsilon_start**      | `0.85`                                                    | Initial epsilon value for the epsilon-greedy policy. |
| **epsilon_end**        | `0.05`                                                    | Minimum epsilon value during training. |
| **decay_rate**         | `0.9995`                                                  | Multiplicative decay factor for epsilon per episode or step. |
| **num_steps**          | `10000`                                                   | Maximum number of steps per episode. |
| **num_episodes**       | `10000`                                                   | Total number of training episodes. |
| **min_sample**         | `200`                                                     | Minimum number of experiences before training begins. |
| **no_step_train**      | `5`                                                       | Number of steps between training updates. |
| **no_step_update_target** | `5000`                                                 | Steps between target network updates. |
| **batch_size**         | `32`                                                      | Minibatch size for training updates. |
| **load_checkpoint**    | `True`                                                    | Load model weights from an existing checkpoint at startup. |
| **load_deque**         | `True`                                                    | Load saved replay buffer from a file at startup. |
| **capacity**           | `100000`                                                  | Maximum number of experiences stored in the replay buffer. |



