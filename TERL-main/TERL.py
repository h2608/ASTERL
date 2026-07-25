import numpy as np, time, random
import utils
import gym, torch, os
import TD3
import argparse
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

render = False
parser = argparse.ArgumentParser()
parser.add_argument('-env', help='Environment Choices: (HalfCheetah-v2) (Ant-v2) (Reacher-v2) (Walker2d-v2) (Swimmer-v2) (Hopper-v2)', required=True)
parser.add_argument('-seed', default=7, type=int)
parser.add_argument('-pop_size', default=5, type=int)
args = parser.parse_args()
seed = args.seed
env_tag = args.env
time_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
run_name = f"{env_tag}__TERL__{seed}__{time_str}"
writer = SummaryWriter(f"runs/{env_tag}/{run_name}")
cwd = os.getcwd()
path = f'{cwd}/log/{env_tag}/{run_name}'
os.makedirs(path, mode=0o777)
f = open(f'{path}/log.txt', 'w')
new_steps = [0] * args.pop_size
best_f = [-9999] * args.pop_size
max_best_f = max(best_f)
last_test_point = 0
best_idx = 0
learned_steps = [0] * args.pop_size
total_learned_steps = [0] * args.pop_size
last_evo_point = [0] * args.pop_size
extra_idx = 0
stage = 1
test_individual_fitness = -9999


def source_to_target(source_net, target_net):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(source_param.data)


class Parameters:
    def __init__(self):

        # max timesteps
        if env_tag == 'Hopper-v2':
            self.max_timesteps = 4000000
        elif env_tag == 'Ant-v2':
            self.max_timesteps = 6000000
        elif env_tag == 'Walker2d-v2':
            self.max_timesteps = 8000000
        elif env_tag == 'HalfCheetah-v2' or env_tag == 'Reacher-v2' or env_tag == 'Swimmer-v2':
            self.max_timesteps = 2000000
        else:
            self.max_timesteps = 1000000

        # env info
        self.state_dim = None
        self.action_dim = None
        self.max_action = None
        if env_tag == 'Reacher-v2':
            self.max_episode_steps = 50
        elif env_tag == 'BipedalWalker-v3':
            self.max_episode_steps = 1600
        elif env_tag == 'Pendulum-v1':
            self.max_episode_steps = 200
        else:
            self.max_episode_steps = 1000

        # seed
        self.seed = args.seed
        f.write("seed=%s " % self.seed)

        # RL hyperparameter
        self.batch_size = 256
        self.expl_noise = 0.1
        self.start_timesteps = 25000

        # EA hyperparameter
        if env_tag == 'HalfCheetah-v2':
            self.stable_eval_times = 1
        else:
            self.stable_eval_times = 5

        self.ratio = 0.25
        f.write("exploration ratio=%s " % self.ratio)
        f.write("stable_eval_times=%s " % self.stable_eval_times)
        self.pop_size = args.pop_size
        f.write("population size=%s " % self.pop_size)


parameters = Parameters()


class Agent:
    def __init__(self, args, env):
        self.args = args
        self.env = env

        # initialize population
        self.pop = []
        self.V = []
        self.inertia_weight = 0
        for i in range(args.pop_size):
            if env_tag == 'Swimmer-v2':
                self.pop.append(TD3.TD3(args.state_dim, args.action_dim, args.max_action, discount=0.999))
            else:
                self.pop.append(TD3.TD3(args.state_dim, args.action_dim, args.max_action))
            params = self.pop[i].actor.state_dict()
            self.V.append([torch.zeros_like(params['l1.weight']), torch.zeros_like(params['l2.weight']),
                           torch.zeros_like(params['l3.weight']),
                           torch.zeros_like(params['l1.bias']), torch.zeros_like(params['l2.bias']),
                           torch.zeros_like(params['l3.bias'])])
            self.test_individual = TD3.Actor(args.state_dim, args.action_dim, args.max_action).to(device)
        self.replay_buffer = utils.ReplayBuffer(args.state_dim, args.action_dim)

        # steps
        self.num_games = 0
        self.timesteps = 0

    def evaluate(self, individual_idx, is_action_noise=False, store_transition=True):
        global learned_steps, total_learned_steps
        total_reward = 0.0
        if store_transition:
            policy = self.pop[individual_idx]
        else:
            policy = self.test_individual

        state = self.env.reset()

        done = False
        step = 0
        while not done:
            if self.timesteps < self.args.start_timesteps:
                action = self.env.action_space.sample()
            else:
                if is_action_noise:
                    action = (
                            policy.select_action(np.array(state))
                            + np.random.normal(0, self.args.max_action * self.args.expl_noise, size=self.args.action_dim)
                    ).clip(-self.args.max_action, self.args.max_action)
                else:
                    action = policy.select_action(np.array(state))

            next_state, reward, done, info = self.env.step(action)
            done_bool = float(done) if step < self.args.max_episode_steps - 1 else 0
            total_reward += reward

            if store_transition:
                self.timesteps += 1
                new_steps[individual_idx] += 1
                if self.timesteps >= self.args.start_timesteps:
                    total_learned_steps[individual_idx] += 1
                    learned_steps[individual_idx] += 1
                self.replay_buffer.add(state, action, next_state, reward, done_bool)
            step += 1
            state = next_state
        if store_transition:
            self.num_games += 1
            f.write("%s,%.0f " % (step, total_reward))

        return total_reward

    def pso(self):
        global stage
        # gbest
        params = self.pop[best_idx].actor.state_dict()
        gbest = [params['l1.weight'], params['l2.weight'], params['l3.weight'], params['l1.bias'],
                     params['l2.bias'], params['l3.bias']]

        # update population
        if stage == 1:
            update_frequency = 1e4
        else:
            update_frequency = 1e3
        for i in range(args.pop_size):
            if i != best_idx and learned_steps[i] - last_evo_point[i] > update_frequency:
                last_evo_point[i] = learned_steps[i]

                # initialize X and V
                params = self.pop[i].actor.state_dict()
                X = [params['l1.weight'], params['l2.weight'], params['l3.weight'], params['l1.bias'],
                     params['l2.bias'], params['l3.bias']]
                V = self.V[i]

                # update V
                for j in range(len(X)):
                    V[j] = self.inertia_weight * V[j] \
                           + torch.rand_like(X[j]) * (gbest[j] - X[j])  # pbest[j] = X[j], so pbest[j] - X[j] = 0

                # update X
                for j in range(len(X)):
                    X[j].copy_(X[j] + V[j])

                f.write(" diff%s=%.0f " % (i, self.get_difference(self.pop[i].actor, self.pop[best_idx].actor)))

    def get_difference(self, net1, net2):
        params1 = net1.state_dict()
        params2 = net2.state_dict()
        net1_W1 = params1['l1.weight']
        net1_W2 = params1['l2.weight']
        net1_W3 = params1['l3.weight']
        net2_W1 = params2['l1.weight']
        net2_W2 = params2['l2.weight']
        net2_W3 = params2['l3.weight']
        diff = 0

        diff += torch.sum(abs(net1_W1 - net2_W1))
        diff += torch.sum(abs(net1_W2 - net2_W2))
        diff += torch.sum(abs(net1_W3 - net2_W3))
        return float(diff)

    def train(self):
        global max_best_f, last_test_point, best_f, best_idx, learned_steps, total_learned_steps, extra_idx, stage, test_individual_fitness
        if self.timesteps >= self.args.max_timesteps * self.args.ratio:  # stage 2
            stage = 2
        max_eval_times = 1
        if env_tag == 'LunarLanderContinuous-v2' or env_tag == 'Pendulum-v1':  # the variation of performance in these environment is too huge
            max_eval_times = self.args.stable_eval_times

        # evaluate pop
        if learned_steps[extra_idx] > learned_steps[best_idx] or stage == 2:
            extra_idx = best_idx
        idx_list = [extra_idx] * 5 + list(range(self.args.pop_size))
        for i in idx_list:
            f.write("[%s]:" % i)
            new_steps[i] = 0
            eval_times = 0
            fitness = 0
            while eval_times < max_eval_times:
                fitness = (fitness * eval_times + self.evaluate(i, is_action_noise=True)) / (eval_times + 1)
                eval_times += 1
                if fitness < max_best_f:
                    break
            #  update best_f
            if fitness > best_f[i]:
                best_f[i] = fitness
                learned_steps[i] = 0
                last_evo_point[i] = 0
                if stage == 1:
                    extra_idx = i
                    tag = "fitness" + str(i)
                    writer.add_scalar(tag, best_f[i], total_learned_steps[i])
            #  update max_best_f, best_idx, test_individual
            if best_f[i] > max_best_f:
                max_best_f = best_f[i]
                if stage == 1:
                    best_idx = i
                    source_to_target(self.pop[i].actor, self.test_individual)
                else:  # stage 2
                    if i != best_idx:
                        source_to_target(self.pop[i].actor, self.pop[best_idx].actor)
                    # update test_individual
                    if self.args.stable_eval_times > 1 and max_eval_times == 1:
                        fitness = 0
                        for _ in range(self.args.stable_eval_times):
                            fitness += self.evaluate(i) / self.args.stable_eval_times
                        if fitness > test_individual_fitness:
                            test_individual_fitness = fitness
                            source_to_target(self.pop[i].actor, self.test_individual)
                    else:
                        source_to_target(self.pop[i].actor, self.test_individual)
            # RL train
            if self.replay_buffer.size >= self.args.start_timesteps:
                for _ in range(new_steps[i]):
                    if stage == 1:
                        self.pop[i].train(self.replay_buffer, self.args.batch_size)
                    else:
                        self.pop[best_idx].train(self.replay_buffer, self.args.batch_size)

        # pso
        self.pso()

        #  info for debug
        f.write(" %s,%.0f" % (best_idx, max_best_f))
        f.write(" pop:")
        for i in range(self.args.pop_size):
            f.write("[%s]:%.0f,%s/%s " % (i, best_f[i], learned_steps[i], total_learned_steps[i]))
        f.write("\n")

        # test
        if self.timesteps - last_test_point >= 5e3:
            last_test_point = self.timesteps
            test_score = 0.0
            for _ in range(5):
                test_score += self.evaluate(best_idx, store_transition=False) / 5.0
            writer.add_scalar("charts/test_score", test_score, self.timesteps)
        else:
            test_score = None

        return test_score


if __name__ == "__main__":
    # Create Env
    env = gym.make(env_tag)
    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]
    parameters.max_action = float(env.action_space.high[0])

    # Seed
    env.seed(parameters.seed)
    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)
    random.seed(parameters.seed)
    env.action_space.seed(parameters.seed)

    # Create Agent
    agent = Agent(parameters, env)
    f.write('Running: %s State_dim: %s Action_dim:%s\n' % (env_tag, parameters.state_dim, parameters.action_dim))

    # learning
    while agent.timesteps <= parameters.max_timesteps:
        test_score = agent.train()
        if test_score is not None:
            f.write('\n#Games:%s #Steps:%s Test_Score: %.2f ENV:%s\n\n' % (agent.num_games, agent.timesteps, test_score, env_tag))
            f.flush()












