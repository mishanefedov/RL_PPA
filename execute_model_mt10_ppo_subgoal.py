from stable_baselines3 import PPO
from SubGoalEnv import SubGoalEnv
import csv

def execute():
    need_header = True
    
    calculate_max_success = True
    csv_file = 'results_mean.csv'
    if calculate_max_success:
        csv_file = "result_max.csv"
    
    episodes_per_task = 30
    # titles = ['reach-v2', 'push-v2', 'pick-place-v2', 'door-open-v2', 'drawer-open-v2', 'drawer-close-v2','button-press-topdown-v2', 'peg-insert-side-v2', 'window-open-v2', 'window-close-v2']
    titles = ['reach-v2', 'push-v2', 'pick-place-v2', 'door-open-v2', 'drawer-open-v2', 'drawer-close-v2','button-press-topdown-v2', 'peg-insert-side-v2', 'window-open-v2', 'window-close-v2']
    all_tasks_mean_reward = 0
    all_tasks_mean_steps = 0
    all_tasks_success_rate = 0
    for i, title in enumerate(titles):
        env = make_env(title, "rew1", 10, i)()
        models_dir = "models/teleportation/"
        # model_path = f"{models_dir}/846692352"
        model_path = f"{models_dir}/small_step_best"
        
        model = PPO.load(model_path, env=env)
        mean_rew_all_tasks = 0
        num_success = 0
        mean_steps = 0
        for ep in range(episodes_per_task):
            obs = env.reset()
            done = False
            steps = 0
            total_reward = 0
            success = False
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                steps += 1
                total_reward += reward
                if info['success']:
                    success = True
                if done and success:
                    num_success += 1
            mean_rew_all_tasks += total_reward
            mean_steps += steps
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if need_header:
                writer.writerow(['Title', 'Mean Total Reward', 'Mean Steps', 'Success Rate'])
                need_header = False

            writer.writerow([
                title,
                mean_rew_all_tasks / episodes_per_task,
                mean_steps / episodes_per_task,
                num_success / episodes_per_task
            ])
        all_tasks_mean_reward += mean_rew_all_tasks
        all_tasks_mean_steps += mean_steps
        all_tasks_success_rate += num_success
    print("\n-------all tasks:-------")
    print("mean_tot_rew:", all_tasks_mean_reward / (episodes_per_task*10))
    print("mean_steps:", all_tasks_mean_steps / (episodes_per_task*10))
    print("success rate:", all_tasks_success_rate / (episodes_per_task*10))
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "all-tasks-mean",
            all_tasks_mean_reward / (episodes_per_task * 10),
            all_tasks_mean_steps / (episodes_per_task * 10),
            all_tasks_success_rate / (episodes_per_task * 10)
        ])
        
        

def make_env(name,rew_type,number_of_one_hot_tasks, one_hot_task_index):

    def _init():
        return SubGoalEnv(env=name, rew_type=rew_type, number_of_one_hot_tasks=number_of_one_hot_tasks,
                          one_hot_task_index=one_hot_task_index,render_subactions=False)
    return _init


if __name__ == '__main__':
    execute()