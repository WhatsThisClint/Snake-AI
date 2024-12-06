import torch
import numpy as np
from game import SnakeGameAI
from agent import Agent, Linear_QNet
import os

def play_game():
    # Create the game and agent
    game = SnakeGameAI()
    agent = Agent()
    
    # Load the trained model if it exists
    model_folder_path = './model'
    model_path = os.path.join(model_folder_path, 'model.pth')
    
    if os.path.exists(model_path):
        model_state = torch.load(model_path)
        agent.model.load_state_dict(model_state)
        print("Loaded trained model!")
    else:
        print("No trained model found. Please train the model first.")
        return
    
    # Play the game
    while True:
        # Get current state
        state = agent.get_state(game)
        
        # Get move (no randomness in playing mode)
        with torch.no_grad():
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = agent.model(state0)
            move = torch.argmax(prediction).item()
            final_move = [0, 0, 0]
            final_move[move] = 1
        
        # Perform move
        reward, done, score = game.play_step(final_move)
        
        if done:
            game.reset()
            print(f'Game Over. Score: {score}')

if __name__ == '__main__':
    play_game()
