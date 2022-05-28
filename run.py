from DDPG import *
import numpy as np
from servo_system_env import ServoSystemEnv
from utils import trapezoidal_function


class run_base(object):
    def __init__(self):
        self.var_min = 0.01
        self.env = ServoSystemEnv()  # 环境

        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 训练模式
        if args.run_type == "train":
            result_dir = os.path.join('results',
                                      'ChooseModel_{}_CurveType_{}_Height_{}_DumpSystem_{}_RunType_{}'.format(
                                          args.choose_model, args.curve_type, args.height, args.use_jump_system,
                                          args.run_type))
            self.result_dir = os.path.join(current_dir, result_dir)
        else:
            # Need to load the train model.  需要加载训练的模型
            result_dir = os.path.join('results',
                                      'ChooseModel_{}_CurveType_{}_Height_{}_DumpSystem_{}_RunType_{}'.format(
                                          args.choose_model, args.curve_type, args.height, args.use_jump_system,
                                          "train"))
            self.result_dir = os.path.join(current_dir, result_dir)

        if args.choose_model == "search_pid_parameter":
            self.max_episodes = 800
            self.action_dim = args.pid_parameter_search_action_dim  # 动作维度  默认2
            self.action_bound = args.pid_parameter_search_action_bound  # 动作范围界限  默认[-10.0, 10.0]

            self.state_dim = self.env.state_dim   # 状态维度
            self.agent = DDPG(self.state_dim, self.action_dim, self.action_bound)  # 创建一个Agent


    def train(self,times):
        pass

    def test(self,times):
        pass

    def save_result_txt(self, xs, ys, yaxis='radValues'):
        """
        # Save the data to .txt file.
        :param xs:
        :param ys:
        :param yaxis:
        """
        filename = os.path.join(self.result_dir, yaxis + '.txt')
        if isinstance(xs, np.ndarray) and isinstance(ys, list):
            if not os.path.exists(filename):
                with open(file=filename, mode="a+") as file:
                    file.write("times {}".format(yaxis))
                    file.write("\r\n")
            else:
                print("{} has already existed. added will be doing.".format(filename))
                with open(file=filename, mode="a+") as file:
                    file.write("times, {}".format(yaxis))
                    file.write("\r\n")
        else:
            pass

        with open(file=filename, mode="a+") as file:
            for index, data in enumerate(zip(xs, ys)):
                file.write("{} {}".format(str(data[0]), str(data[1])))
                file.write("\r\n")


class run_pid_parameter(run_base):
    def __init__(self):
        super(run_pid_parameter, self).__init__()
        self.action_dim = args.pid_parameter_search_action_dim  # PID参数调整的动作维度  默认2
        self.action_bound = args.pid_parameter_search_action_bound  # PID参数调整的界限 默认[-10.0, 10.0]

    def train(self,times):
        steps = []
        epRewards = []
        maxEpReward = 4000.0  # set the base max reward.

        for i in range(self.max_episodes):
            state = self.env.reset()  # 获取环境初始状态
            ep_reward = 0 # 累计奖励
            for j, sim_time in enumerate(times):
                action = self.agent.select_action(state) # 获得行为
                # 给行为添加探索的噪音
                action = (action + np.random.normal(0, args.exploration_noise, size=self.action_dim)).clip(
                    -self.action_bound, self.action_bound) # exploration_noise: default=0.1， clip相当于做了一个边界处理

                next_state, reward, done = self.env.step(action)   # 获得下一个状态
                self.agent.replay_buffer.push((state, next_state, action, reward, np.float64(done)))  # 放入经验回放池

                state = next_state  # 更新状态

                ep_reward += reward # 累加奖励

                if sim_time == times[-1]: # 到达了times的最后
                    # 打印: 第i段经历, 奖励 , 探索噪音, time
                    print("Episode: {}, Reward {}, Explore {} Steps {}".format(i, ep_reward, args.exploration_noise, j))
                    epRewards.append(ep_reward)
                    steps.append(i)

            self.agent.update()  # 更新网络

            if ep_reward >= maxEpReward:  # 保存最大的 reward 的模型
                print("Get More Episode Reward {}".format(maxEpReward))
                maxEpReward = ep_reward  # 更新最大reward值
                self.agent.save() # 保存模型

        self.save_result_txt(xs=steps,ys=epRewards,yaxis="epRewards")

def main():
    # Get Time [0,args.dt,2 * args.dt,..., ]
    times = np.arange(args.start_time, args.end_time, args.dt)[:args.max_ep_steps]
    runner = run_pid_parameter()
    if args.run_type == "train":
        if args.choose_model == "search_pid_parameter":
            runner.train(times)
    else:
        pass

if __name__ == '__main__':
    main()