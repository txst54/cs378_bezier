import gym
import imageio
import numpy as np


def play_video(image_list, fps=30):
    output_video = 'tmp/temp_video.mp4'
    with imageio.get_writer(output_video, fps=fps) as writer:
        for img in image_list:
            writer.append_data(img)


def rand_control(seed):
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    obs = env.reset(seed=seed)
    total_reward = 0
    done = False
    imgs = [env.render()]
    while not done:
        action = np.random.randint(0, 2)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        imgs.append(env.render())
    print(f'reward={total_reward}')
    return imgs


def nn_control(policy, params, seed, render=True, break_in_middle=False):
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    obs = env.reset(seed=seed)[0]
    policy.seed(seed=seed)
    total_reward = 0
    done = False
    step_to_break = np.random.randint(100, 300)
    step = 0
    imgs = []
    if render:
        imgs = [env.render()]
    while not done:
        if break_in_middle and step_to_break == step:
            print(f'Reset MLP parameters at step {step_to_break}')
            param_hist = policy.mlp_params_hist
            policy.seed(seed=np.random.randint(1 << 20))
            policy.mlp_params_hist = param_hist + policy.mlp_params_hist
        action = policy(params, obs)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        step += 1
        if render:
            imgs.append(env.render())
        if step >= 1000:
            # print("early termination")
            break
    env.close()
    if render:
        print(f'reward={total_reward}')
        return imgs
    else:
        return total_reward
