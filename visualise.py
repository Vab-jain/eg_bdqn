import argparse
import time
import os
import yaml
import torch
import gymnasium as gym

from agent import EGBDQNAgent
from model import tokenize_mission

def load_agent(config_path, model_path, device="cpu"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    env_name = config["env"]["env_name"]
    env = gym.make(env_name)
    num_actions = env.action_space.n
    agent = EGBDQNAgent(config, num_actions, device)
    
    agent.load(model_path)
    # set epsilon to 0 for evaluation
    agent.eps_start = 0.0
    agent.eps_end = 0.0
    
    return agent, config, env_name

def visualise(agent, env_name, delay=0.5, episodes=5):
    # render_mode="human" for visualization
    env = gym.make(env_name, render_mode="human")
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        print(f"Episode {ep+1}/{episodes} started")
        while not done:
            env.render()
            time.sleep(delay)
            # Evaluate using get action
            obs_image = obs["image"]
            obs_mission = tokenize_mission(obs["mission"])
            
            # Since evaluate, no need for oracle, just use agent policy (epsilon 0, greedy)
            a_greedy, _ = agent.compute_uncertainty(obs_image, obs_mission)
            
            next_obs, reward, terminated, truncated, _ = env.step(a_greedy)
            done = terminated or truncated
            obs = next_obs
            steps += 1
            # make this print update itself
            print(f"Step {steps}: Action: {a_greedy}", end="\r") 
        env.render()
        print(f"Episode {ep+1} finished with reward: {reward}")
        time.sleep(1) # pause before next episode
    env.close()

def evaluate(agent, env_name, num_episodes=100):
    env = gym.make(env_name)
    successes = 0
    total_rewards = 0
    total_steps = 0
    
    print(f"Starting evaluation over {num_episodes} episodes...")
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        ep_reward = 0
        while not done:
            obs_image = obs["image"]
            obs_mission = tokenize_mission(obs["mission"])
            a_greedy, _ = agent.compute_uncertainty(obs_image, obs_mission)
            
            next_obs, reward, terminated, truncated, _ = env.step(a_greedy)
            done = terminated or truncated
            obs = next_obs
            steps += 1
            ep_reward += reward
        
        if ep_reward > 0: # Assuming positive reward means success in BabyAI
            successes += 1
        total_rewards += ep_reward
        total_steps += steps
        
    print(f"\nResults over {num_episodes} episodes:")
    print(f"Success Rate: {successes / num_episodes * 100:.2f}%")
    print(f"Average Reward: {total_rewards / num_episodes:.4f}")
    print(f"Average Steps: {total_steps / num_episodes:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Visualize or evaluate a saved model")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model", required=True, help="Path to saved model (.pt)")
    parser.add_argument("--mode", choices=["visualise", "evaluate"], default="visualise", help="Mode: visualise (with rendering) or evaluate (metrics over episodes)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between steps in visualise mode")
    parser.add_argument("--episodes", type=int, default=None, help="Episodes for evaluation (default 100) or visualize (default 5)")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent, config, env_name = load_agent(args.config, args.model, device)
    
    if args.mode == "visualise":
        episodes = 5 if args.episodes is None else args.episodes 
        visualise(agent, env_name, delay=args.delay, episodes=episodes)
    elif args.mode == "evaluate":
        episodes = 100 if args.episodes is None else args.episodes
        evaluate(agent, env_name, num_episodes=episodes)

if __name__ == "__main__":
    main()
