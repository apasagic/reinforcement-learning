# Deep Q-Learning for Atari Breakout 🎮

This repository contains an implementation of **Deep Q-Learning (DQN)** to play Atari's Breakout using reinforcement learning techniques.  
The agent learns to control the paddle and break all bricks by maximizing cumulative reward.

---

## 📌 Features
- Implements **Deep Q-Network (DQN)** from scratch with PyTorch
- Uses **OpenAI Gym** Breakout environment
- Experience replay buffer
- Target network updates
- Epsilon-greedy exploration strategy
- Training progress logging

---

## 🏗 Project Structure

├── dqn_agent.py # DQN Agent class
├── train.py # Training loop
├── play.py # Run a trained agent
├── config_local.json # Local paths (ignored in .gitignore)
├── requirements.txt # Python dependencies
└── README.md
