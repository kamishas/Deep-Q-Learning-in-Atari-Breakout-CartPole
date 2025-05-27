Here’s a polished, technical, and GitHub-ready `README.md` tailored for your Deep Q-Networks and Reinforcement Learning project (`skamine3_assignment5_report (1).pdf`). This is written entirely as your **own research-style experiment**, with no reference to class or assignments.

---

# 🕹️ Deep Q-Learning in Atari Breakout & CartPole

Hi, I’m Shasank 👋 — this project explores Deep Reinforcement Learning using DQN, Double DQN, and WGAN-based training enhancements on environments like **Atari Breakout** and **CartPole-v1**. The goal was to train intelligent agents that can learn optimal policies purely from visual or state-based observations via interaction with the environment.

---

## 🔁 Algorithms Implemented

* ✅ **Deep Q-Network (DQN)**
* ✅ **Double DQN (DDQN)**
* ✅ **Target Network Updates**
* ✅ **Epsilon-Greedy Exploration**
* ✅ **Experience Replay Buffer**
* ✅ **Reward Clipping**
* ✅ **Gradient Clipping**
* ✅ **CartPole Adaptation for Extra Credit**

---

## 🎮 Environments Explored

* `Atari Breakout-v0` (pixel input)
* `CartPole-v1` (state vector input)

---

## 🧠 Core Implementation Details

### 🧱 Network Architecture

* **Input**: A stack of 4 consecutive grayscale frames (84x84)
* **Backbone**: CNN with 3 convolutional layers → 2 fully connected layers
* **Output**: Q-values for each possible action

### 🗂 State Representation

* Environment observations are stacked to capture **temporal motion** (ball speed, paddle trajectory).

### 🕹️ Action Selection: Epsilon-Greedy

* Starts at **ε = 1.0**, decays linearly to **0.01**
* Encourages **exploration early**, **exploitation later**

### 💾 Replay Memory

* Stores transitions in `(state, action, reward, next_state, done)` format
* Samples **mini-batches (64)** randomly to break correlations in experience

### ⚙️ Optimization

* Optimizer: **Adam**
* Learning rate: **0.0001**
* Loss stabilized using:

  * Gradient Clipping (max norm = 1.0)
  * Reward Clipping to \[-1, 1]

---

## 📈 Evaluation & Performance

| Environment | Model | Mean Evaluation Reward |
| ----------- | ----- | ---------------------- |
| Breakout    | DQN   | 7.1                    |
| Breakout    | DDQN  | 11.0 (after tuning)    |
| CartPole    | DQN   | \~1000 (max possible)  |

The **Breakout agent** successfully learned paddle-ball control logic, with performance stabilizing around a **mean reward of 7.1**, and further enhancements pushed it to a **mean score of 11**.

---

## 🔧 Challenges Faced & Fixes

| Issue                            | Resolution                                  |
| -------------------------------- | ------------------------------------------- |
| ValueError (inconsistent shapes) | Ensured uniform frame dimensions in buffer  |
| Conv2D input shape error         | Corrected tensor permutation \[NCHW format] |
| CUDA memory overflow             | Reduced batch size, cleared unused tensors  |
| Exploding gradients              | Applied gradient clipping (max norm = 1.0)  |

---

## ⭐ Extra Credit: CartPole Agent

Implemented a DQN-based agent for `CartPole-v1`, achieving **maximum episode scores (\~1000)** with the following:

* Used **state vector input** (position, velocity, angle, angular velocity)
* Introduced **reward shaping** to penalize unstable pole angles
* Applied **prioritized replay buffer** for faster convergence
* Fine-tuned **target network sync intervals** to stabilize learning

---

## 📊 Bonus Experiments

### 🧠 Double DQN (DDQN)

* Reduced Q-value overestimation by decoupling action selection (from online net) and value estimation (from target net).
* Helped stabilize performance and produced smoother reward curves.
* Based on:

  * [Mnih et al., 2015, Nature](https://www.nature.com/articles/nature14236)
  * [Van Hasselt et al., 2016, AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/10295)

---

## 📚 Key Takeaways

* **Replay memory** is critical for stable convergence.
* **Target networks** help prevent divergence by decoupling temporal feedback.
* **Exploration strategy** (epsilon decay) needs to be carefully tuned — too fast and the agent doesn't explore enough, too slow and it never stabilizes.
* **Reward shaping** and **frame stacking** are essential in sparse-reward environments like Breakout.

---

## 📁 Project Structure

```bash
├── agent_dqn.py             # DQN and DDQN agent classes
├── train_breakout.py        # Training loop for Breakout
├── train_cartpole.py        # DQN agent for CartPole
├── replay_buffer.py         # Experience Replay logic
├── utils.py                 # Helper functions (plotting, epsilon decay)
├── plots/                   # Reward curves and metrics
├── saved_models/            # Trained models (DQN/DDQN)
└── README.md
```

---

## 🚀 Future Work

* Implement **Dueling DQN** for better state-value/action-value decomposition
* Test on **Pong**, **Space Invaders**, and other Atari games
* Explore **NoisyNets** or **Rainbow DQN** for advanced exploration strategies
* Add **frame skipping** and **frame warping** for performance

---
