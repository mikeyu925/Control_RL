
import argparse
# 读取参数
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)  # mode = 'train' or 'test'
# OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
# Note that DDPG is feasible about hyper-parameters.
# You should fine-tuning if you change to another environment.
parser.add_argument("--env_name", default="Pendulum-v1")
parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)
parser.add_argument('--max_length_of_trajectory', default=1000, type=int)


parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
parser.add_argument('--capacity', default=1000000, type=int)  # replay buffer size 经验回放池大小
parser.add_argument('--batch_size', default=100, type=int)  # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--render', default=False, type=bool)  # show UI or not
parser.add_argument('--log_interval', default=50, type=int)  #
parser.add_argument('--load', default=False, type=bool)  # load model
parser.add_argument('--render_interval', default=100, type=int)  # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=2, type=float)
parser.add_argument('--max_episode', default=100000, type=int)  # num of games
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=200, type=int)

parser.add_argument("--max_ep_steps", type=int, default=4800, help="")

# pid
# 跟踪曲线的高度
parser.add_argument("--height", type=int, default=1000, help="the highest value of the tracking curve")
# 要跟踪的曲线类型
parser.add_argument("--curve_type", type=str, default="trapezoidal", help="type of curve to be tracked, choose of ['trapezoidal', 'sine']")
# 离散时间间隔
parser.add_argument("--dt", type=float, default=0.0001, help="discrete time interval")
# 仿真开始的时间
parser.add_argument("--start_time", type=float, default=0.0, help="the start time of simulate")
# 仿真结束时间
parser.add_argument("--end_time", type=float, default=0.5, help="the end time of simulate")
# 系统改变(跳变)的时间
parser.add_argument("--change_times", type=int, default=9, help="the change times of system")
parser.add_argument("--use_jump_system", type=bool, default=False, help="select whether to choose a jump system")
parser.add_argument("--used_system_id", type=int, default=1, help="choose the system id when not use jump system. default in [1, 2, 3]")
# pid的参数
parser.add_argument("--kp", type=float, default=0.5, help="the kp parameter of PID")
parser.add_argument("--ki", type=float, default=2.985, help="the ki parameter of PID")
parser.add_argument("--kd", type=float, default=0.0, help="the kd parameter of PID")

# RL Agent 给定的参数
parser.add_argument("--kp_rl", type=float, default=0.9318, help="the kp parameter of RL Agent Given")
parser.add_argument("--ki_rl", type=float, default=1.4824, help="the ki parameter of RL Agent Given")
parser.add_argument("--kd_rl", type=float, default=0.0, help="the kd parameter of RL Agent Given")
# pid参数搜索的范围
parser.add_argument("--pid_parameter_search_action_bound", type=float, default=10.0, help="the action bound of pid search for [-10.0, 10.0]")
parser.add_argument("--electric_compensation_action_bound", type=float, default=2.0, help="the action bound of electric search for [-2.0, 2.0]")
parser.add_argument("--state_dim", type=int, default=6, help="the state dim")  # 状态维度
parser.add_argument("--pid_parameter_search_action_dim", type=int, default=2, help="the action dim of PID Parameters Adjustment") # PID参数调整的动作维度
parser.add_argument("--electric_compensation_action_dim", type=int, default=1, help="the action dim of electric compensation.")


parser.add_argument("--ep_reward_max", type=int, default=1000000, help="the ")
parser.add_argument("--run_type", type=str, default="train", help="choose from ['train', 'test']")
parser.add_argument("--choose_model", type=str, default='search_electric', help="one of ['search_pid_parameter', 'search_electric', 'class_pid']")

args = parser.parse_args()