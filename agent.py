import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 80  # Starting with higher randomness
        self.gamma = 0.95  # Increased discount rate for better long-term planning
        self.memory = deque(maxlen=MAX_MEMORY)
        input_size = 21  # 3 danger + 4 direction + 8 vision + 3 food + 3 special features
        self.model = Linear_QNet(input_size, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        # Try to load existing training state
        self.load_state()

    def get_state(self, game):
        head = game.snake[0]
        
        # Create points for all 8 directions
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        point_ul = Point(head.x - 20, head.y - 20)
        point_ur = Point(head.x + 20, head.y - 20)
        point_dl = Point(head.x - 20, head.y + 20)
        point_dr = Point(head.x + 20, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Vision in 8 directions (distance to walls, obstacles, or body)
        vision_l = self._get_distance_to_obstacle(game, head, [-20, 0])
        vision_r = self._get_distance_to_obstacle(game, head, [20, 0])
        vision_u = self._get_distance_to_obstacle(game, head, [0, -20])
        vision_d = self._get_distance_to_obstacle(game, head, [0, 20])
        vision_ul = self._get_distance_to_obstacle(game, head, [-20, -20])
        vision_ur = self._get_distance_to_obstacle(game, head, [20, -20])
        vision_dl = self._get_distance_to_obstacle(game, head, [-20, 20])
        vision_dr = self._get_distance_to_obstacle(game, head, [20, 20])

        # Food information
        food_dist_x = game.food.x - head.x
        food_dist_y = game.food.y - head.y
        food_dist = ((food_dist_x ** 2) + (food_dist_y ** 2)) ** 0.5

        # Special food awareness
        is_special_food = 1 if hasattr(game, 'special_food') and game.special_food else 0

        # Nearby obstacles awareness
        nearby_obstacles = sum(1 for obs in game.obstacles 
                             if abs(obs.x - head.x) < 3*BLOCK_SIZE and 
                                abs(obs.y - head.y) < 3*BLOCK_SIZE)
        
        # Level awareness
        level = game.level / 10  # Normalize level

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Vision (normalized)
            vision_l / game.w,
            vision_r / game.w,
            vision_u / game.h,
            vision_d / game.h,
            vision_ul / (game.w + game.h),
            vision_ur / (game.w + game.h),
            vision_dl / (game.w + game.h),
            vision_dr / (game.w + game.h),

            # Food information (normalized)
            food_dist_x / game.w,
            food_dist_y / game.h,
            food_dist / (game.w + game.h),
            
            # Special features
            is_special_food,
            nearby_obstacles / 5,  # Normalize nearby obstacles count
            level,  # Already normalized
        ]

        return np.array(state, dtype=float)

    def _get_distance_to_obstacle(self, game, head, direction):
        current = Point(head.x, head.y)
        distance = 0
        while not game.is_collision(current) and distance < max(game.w, game.h):
            current = Point(current.x + direction[0], current.y + direction[1])
            distance += 20
        return distance

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < max(self.epsilon, 20):  # Maintain minimum exploration
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def save_state(self):
        state = {
            'n_games': self.n_games,
            'epsilon': self.epsilon,
            'model_state': self.model.state_dict()
        }
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        torch.save(state, os.path.join(model_folder_path, 'training_state.pth'))
        print(f'Game {self.n_games}: Model saved! Current epsilon: {self.epsilon}')

    def load_state(self):
        try:
            state = torch.load('./model/training_state.pth')
            self.n_games = state['n_games']
            self.epsilon = state['epsilon']
            self.model.load_state_dict(state['model_state'])
            print(f'Loaded existing model! Games played: {self.n_games}, Epsilon: {self.epsilon}')
            return True
        except:
            print('No existing model found, starting fresh!')
            return False
