from kuka_vision_grasping_env4 import KukaVisionGraspingEnv

env = KukaVisionGraspingEnv(render_mode="human")

env.reset()

for i in range(100):
    action = env.action_space.sample()
    obs, reward, done, terminated, info = env.step(action)
    # print(obs, reward, done, info)
    if done:
        env.reset()

