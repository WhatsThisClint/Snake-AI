# AI Snake Game with Reinforcement Learning

This project implements an AI that learns to play the Snake game using Deep Q-Learning, a type of reinforcement learning. The AI starts with no knowledge of the game and progressively learns to play better by experiencing the game and receiving rewards for its actions.

## Project Structure

- `game.py`: Implementation of the Snake game using Pygame
- `model.py`: Neural network model architecture for the AI
- `agent.py`: AI agent that makes decisions and learns from experience
- `train.py`: Script to train the AI
- `play.py`: Script to watch the trained AI play
- `requirements.txt`: List of required Python packages
- `model/model.pth`: Saved model weights (created after training)

## Setup Instructions

1. Make sure you have Python 3.8+ installed
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## How to Use

### Training the AI
To start training the AI:
```
python train.py
```
During training, you'll see:
- A game window showing the snake's movements
- A plot window showing the score progression
- Console output with game number, current score, and record score

The AI's progress is automatically saved whenever it achieves a new high score.
You can stop training at any time by closing either window.

### Watching the Trained AI Play
To watch the trained AI play without further training:
```
python play.py
```

## How It Works

### The Game
- The snake moves around a grid trying to eat food
- Each food eaten increases the score and snake length
- Game ends if snake hits a wall or itself

### The AI
The AI uses Deep Q-Learning to make decisions:
1. **State**: The AI sees:
   - Danger positions (straight, right, left)
   - Direction of movement
   - Food location relative to head

2. **Actions**: The AI can choose to:
   - Go straight
   - Turn right
   - Turn left

3. **Rewards**:
   - +10 for eating food
   - -10 for dying
   - 0 for surviving

### The Learning Process
1. The AI starts by making mostly random moves (exploration)
2. Over time, it learns which actions lead to higher rewards
3. The neural network learns to predict which actions will give the best long-term rewards
4. As training progresses, the AI makes fewer random moves and more strategic decisions

## Files Explained

- `game.py`: Contains the SnakeGameAI class that handles:
  - Game mechanics
  - Visual rendering
  - Collision detection
  - Reward calculation

- `model.py`: Contains the neural network model that:
  - Takes game state as input
  - Predicts best action to take
  - Saves/loads learned weights

- `agent.py`: Contains the Agent class that:
  - Gets game state
  - Makes decisions
  - Remembers experiences
  - Learns from past games

- `train.py`: Handles the training process:
  - Runs game episodes
  - Collects experiences
  - Updates the model
  - Shows training progress

- `play.py`: Demonstrates learned behavior:
  - Loads trained model
  - Runs game without training
  - Shows AI performance

## Customization

You can modify various parameters in the code to change the AI's behavior:
- `BLOCK_SIZE` in `game.py`: Changes the size of the game grid
- `MAX_MEMORY` in `agent.py`: Changes how many experiences the AI remembers
- `BATCH_SIZE` in `agent.py`: Changes how many experiences the AI learns from at once
- `LR` in `agent.py`: Changes how quickly the AI learns
- `SPEED` in `game.py`: Changes the game speed
