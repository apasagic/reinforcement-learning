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



