from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

###计算公式
eps1 = np.finfo(float).eps  # 获取机器精度
def compute_q(x_e, x_zet):
    fi = 10
    q = np.tan((np.pi*x_e)/(2*x_zet*fi+eps1))
    return q

# 辅助函数
def zet(t, T, tao):
    if t < T:
        return (1 - tao) * ((T - t) / T) ** 2 + tao
    else:
        return tao


def zetdt(t, T, tao):
    if t < T:
        return (-2 * (1 - tao) * (T - t) / (T ** 2))
    else:
        return 0

# 第一个网络的参数
A = np.array([
    [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, -1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, -1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, -1.0]
])
H1 = np.diag([0.1, -0.1])

# 第二个网络的参数
C =np.array([
    [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, -1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, -1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, -1.0]
])
H2 = np.diag([0.1, -0.2, 0.1])

#两个网络的映射
B = [
    [-1.0, 0],
    [1.0, 0.64],
    [-0.51, -0.85],
]
# 定义agentxj和agentyj函数


# 定义agentxj和agentyj函数
def agentxj(x, i):
    xi = np.zeros_like(x[i].get_current_state0())
    for j, agent in enumerate(x):
        xij = agent.get_current_state0()
        xi += A[i][j] * np.transpose((H1 @ np.transpose(xij)))
    return xi

def agentyj(y, i):
    yi = np.zeros_like(y[i].get_current_state())
    for j, agent in enumerate(y):
        yij = agent.get_current_state()
        yi += C[i][j] * np.transpose((H2 @ np.transpose(yij)))
    return yi

# 定义ODE方程
def ode0_function(t, x, xj, u, i):
    dxdt = np.zeros_like(x)
    dxdt[0] = -x[0] + x[1] + xj[0]  # 使用u[0]作为x[0]的控制输入
    dxdt[1] = -0.5 * x[0] - x[0] * ((0.1 * x[0] * np.sin(x[1] - x[1])) ** 2) + xj[1]  # 使用u[1]作为x[1]的控制输入
    return dxdt

# 定义ODE方程
def ode_function(t, x, xj, u, i):
    dxdt = np.zeros_like(x)
    dxdt[0] = -x[0] + x[1] - 3.5 * x[2] + xj[0] + 0
    dxdt[1] = -0.5 * x[0].item() - 0.5 * x[1].item() * ((1 - (np.cos(2 * x[1].item() + 2 * x[0].item()) ** 2) + np.sin(x[2].item())) ** 2) + xj[1] + np.cos(
        2 * x[0].item() + 2) *u[1]
    dxdt[2] = x[0].item() ** 3 + np.sin(x[1].item() ** 2) - x[2].item() - x[1].item() ** 4 + xj[2] + 0.1 * np.sin(x[1].item()) *u[2]
    return dxdt

# 定义Agent类
class Agent:
    def __init__(self, initial_state, u, i):
        self.state = np.array(initial_state, dtype=float)
        self.time = 0.0
        self.u = np.array(u, dtype=float)
        self.i = i

    def set_action(self, new_u):
        """设置新的控制输入"""
        self.u = np.array(new_u, dtype=float)

    def update_state(self, delta_time):
        t_span = (self.time, self.time + delta_time)
        sol = solve_ivp(lambda t, x: ode_function(t, x, agentyj(agents_y, self.i), self.u, self.i), t_span, self.state,
                        t_eval=np.linspace(self.time, self.time + delta_time, 100))
        self.state = sol.y[:, -1]
        self.time += delta_time
        return sol

    def get_current_state(self):
        return self.state

    def get_current_action(self):
        return self.u

# 定义Agent0类
class Agent0:
    def __init__(self, initial_state, u, i):
        self.state = np.array(initial_state, dtype=float)
        self.time = 0.0
        self.u = np.array(u, dtype=float)
        self.i = i

    def set_action0(self, new_u):
        """设置新的控制输入"""
        self.u = np.array(new_u, dtype=float)

    def update_state0(self, delta_time):
        t_span = (self.time, self.time + delta_time)
        sol = solve_ivp(lambda t, x: ode0_function(t, x, agentxj(agents_x, self.i), self.u, self.i), t_span, self.state,
                        t_eval=np.linspace(self.time, self.time + delta_time, 100))
        self.state = sol.y[:, -1]
        self.time += delta_time
        return sol

    def get_current_state0(self):
        return self.state

# 创建第一个网络的代理对象
agents_x = [Agent0(initial_state=[2*i/10, 2*i/10], u=[0, 0], i=i) for i in range(6)]


# 创建第二个网络的代理对象
agents_y = [Agent(initial_state=[i/10, i/10, i/10], u=[0, 0, 0], i=i) for i in range(6)]


##定义时间序列

t = np.arange(0, 30, 0.1)
time_steps = np.linspace(0, 10, len(t))
N = len(t)
delta_time = 0.1
Q1 = np.eye(3) * 1
Q = np.array(Q1)
R1 = np.eye(3) * 1
R = np.array(R1)
R_inv = np.linalg.inv(R)
# 定义 ϕc(0) 和 ˆ ϕa(0)
phi_c_0 = np.full((7, 1), 0.1)  # 初始化为 [0.1]
phi_a_0 = np.random.uniform(0.1, 0.7, (7, 1))
n = 0
#创建两个网络的映射
agents_zx = []
agents_z = []
# 获取每个节点的状态
x_states_all = []
y_states_all = []
error = []
error1 = []
u_states_all = []
phi_a_all = []
phi_c_all = []
actions_all = []

for i, agentx in enumerate(agents_x):
    fi = B @ agentx.state
    agents_zx.append(fi)
for j, agenty in enumerate(agents_y):
    print(agenty.state)
    fi = agenty.state - agents_zx[j]
    agents_z.append(fi)
    #print(fi)
#print(agents_z)

X = []
for i, agentz in enumerate(agents_z):
    # 初始化X
    xgm = np.array([zet(n,150,0.10), zet(n,150,0.10), zet(n,150,0.10)])
    ifsl = compute_q(xgm, agentz)
    Xi = [agentz, xgm, ifsl]
    X.append(Xi)
#print(X)
fuzzX = []
for i, agentX in enumerate(X):
    # 定义中心点X0
    X0 = np.array([[-4 + l, -4 + l, -4 + l] for l in range(1, 8)])

    # 假设的3x3矩阵X
    X = np.array(agentX)

    # 初始化一个7x3的数组来存储每个中心点对每个X列的模糊基函数值
    Psi = np.zeros((7, 3))


    # 定义模糊隶属度函数
    def mu_Fl(x, X0_l):
        return np.exp(-0.5 * np.sum((x - X0_l) ** 2, axis=0))  # 使用np.sum计算欧氏距离的平方并沿第一个轴（中心点）求和


    # 计算每个中心点对每个X列的模糊隶属度函数值，并进行归一化
    for j in range(X.shape[1]):  # 遍历X的列
        mu_Fs = np.array([mu_Fl(X[:, j], X0_l) for X0_l in X0])  # 计算当前列对每个中心点的模糊隶属度
        Psi[:, j] = mu_Fs / np.sum(mu_Fs+eps1)  # 归一化并存储在Psi的对应列中
    fuzzX.append(Psi)
    #print(Psi)
#print(fuzzX)
#计算u
action_Y = []
for i, agentX in enumerate(fuzzX):
    agentX_array = np.array(agentX)
    g = np.array([0.5, np.cos(2 * agents_z[i][0] + 2), np.sin(agents_z[i][1])])
    # print(g.shape)
    g2 = g[:, np.newaxis]
    # print(g2.shape)
    action = -15 / 2 * R_inv * g2.T @ agentX_array.T @ phi_a_0

    action_Y.append(action)
    #print(action_Y[i].shape)
    #print(action)
actions_all.append(action_Y)

phi_c = []
phi_a = []
for i, agentX in enumerate(fuzzX):
    phi_c.append(phi_c_0)
    phi_a.append(phi_a_0)
#print("11", action_Y)
#保存动作
actions_all.append(action_Y)
phi_a_all.append(phi_a)
phi_c_all.append(phi_c)
print("11", phi_c_all)
#更新状态并输出
for _ in range(N):  # 假设进行10个时间步长的更新

    J = []
    for i, agentz in enumerate(agents_z):
        # 初始化X
        Ji = agentz
        J.append(Ji)
        #print(Ji)

    #记录状态
    x_states = np.array([agent.get_current_state0() for agent in agents_x])
    y_states = np.array([agent.get_current_state() for agent in agents_y])
    x_states_all.append(x_states)
    y_states_all.append(y_states)

    for i, agentx in enumerate(agents_x):
        agentx.update_state0(delta_time=0.1)  # 假设每个时间步长为0.1

    for i, agent in enumerate(agents_y):
        agent.update_state(delta_time=0.1)  # 假设每个时间步长为0.1
        agent.set_action(action_Y[i])
        #agent.set_action([0, 0, 0])



    # 更新两个网络的映射
    agents_zx = []
    agents_z = []

    for i, agentx in enumerate(agents_x):
        fi = B @ agentx.state
        agents_zx.append(fi)
    for j, agenty in enumerate(agents_y):
        #print(agenty.state)
        fi = agenty.state - agents_zx[j]
        agents_z.append(fi)
        #print(agents_z)
    error.append(agents_z)
    error1.append(agents_z)
    # print(agents_z)

    X = []
    for i, agentz in enumerate(J):
        # 初始化X
        xgm = np.array([zet(n,150,0.1), zet(n,150,0.1), zet(n,150,0.1)])
        ifsl = compute_q(xgm, agentz)
        Xi = [agentz, xgm, ifsl]
        #print(Xi)
        X.append(Xi)
    # print(X)
    fuzzX = []
    for i, agentX in enumerate(X):
        # 定义中心点X0
        X0 = np.array([[-4 + l, -4 + l, -4 + l] for l in range(1, 8)])

        # 假设的3x3矩阵X
        X = np.array(agentX)

        # 初始化一个7x3的数组来存储每个中心点对每个X列的模糊基函数值
        Psi = np.zeros((7, 3))

        # 定义模糊隶属度函数
        eps = np.finfo(float).eps  # 获取机器精度
        def mu_Fl(x, X0_l):
            return np.exp(-0.5 * np.sum((x - X0_l) ** 2, axis=0))  # 使用np.sum计算欧氏距离的平方并沿第一个轴（中心点）求和

        # 计算每个中心点对每个X列的模糊隶属度函数值，并进行归一化
        for j in range(X.shape[1]):  # 遍历X的列
            mu_Fs = np.array([mu_Fl(X[:, j], X0_l) for X0_l in X0])  # 计算当前列对每个中心点的模糊隶属度
            Psi[:, j] = mu_Fs / np.sum(mu_Fs + eps)  # 归一化并存储在Psi的对应列中
            #print(Psi)
            #print(Psi.shape)
        fuzzX.append(Psi)
    fuzzXarray = np.array(fuzzX)
    #print(fuzzXarray.shape)
    # 计算u
    action_Y = []
    for i, agentX in enumerate(fuzzX):
        agentX_array = np.array(agentX)
        g = np.array([0.5, np.cos(2 * agents_z[i][0] + 2), np.sin(agents_z[i][1])])
        #print(g.shape)
        g2 = g[:, np.newaxis]
        #print(g2.shape)
        action = -15 / 2 * R_inv * g2.T @ agentX_array.T @ phi_a[i]
        action_Y.append(action)
        #print(action_Y[i].shape)
        #print(action)
    # 保存动作
    #print(action_Y)
    actions_all.append(action_Y)

    print("当前步骤：", n)
    #print(fuzzX)
    #print(len(phi_c))
    #print(len(fuzzX))
    min_length = min(len(phi_c), len(fuzzX))
    phi_c1 = []
    for i, phi_ci in enumerate(phi_c[:min_length]):
        #计算nmt
        #print(phi_ci)
        nmt = fuzzX[i] @ agents_z[i]
        seta = nmt/(nmt.T@nmt+1)**2
        k1 = 1
        dphi_c = -k1*seta*(nmt.T@phi_ci+J[i].T@Q@J[i]+action_Y[i].T @ R @ action_Y[i])
        print(dphi_c.T*10)
        phi_ci += dphi_c.T * delta_time
        print(phi_ci.T*10)
        phi_c[i] = phi_ci
        phi_c1.append(phi_ci*1)
    #print(phi_c)
    phi_c_all.append(phi_c1)
    min_length2 = min(len(phi_a), len(fuzzX))
    phi_a1 = []
    for i, phi_ai in enumerate(phi_a[:min_length2]):
        g = np.array([0.5, np.cos(2 * agents_z[i][0] + 2), np.sin(agents_z[i][1])])
        #print(g)
        #print(g.shape)
        nmt = fuzzX[i] @ agents_z[i]
        seta = nmt/(nmt.T@nmt+1)**2
        #print(seta.shape)
        setab = nmt/(nmt.T@nmt+1)
        #print(fuzzX[i])
        setax = fuzzX[i] @ (g[:, np.newaxis]  @ g[:, np.newaxis].T) @ fuzzX[i].T
        k2 = 1
        la = 5
        lc = 0.7
        dphi_a = -k2*((la*phi_ai - lc*setab.T@phi_c[i])-1/4*setax@phi_ai@seta[:, np.newaxis].T@phi_c[i])
        phi_ai += dphi_a * delta_time
        #print(dphi_a)
        phi_a[i] = phi_ai
        phi_a1.append(dphi_a*1)
        #print(phi_ai)

    #print("11",phi_a)
    phi_a_all.append(phi_a1)
    n += 1
phi_c_all = np.array(phi_c_all)



def plot_weight_changesa(weights_data, num_steps, num_nodes, num_weights, step_interval=10):
    fig, axes = plt.subplots(num_nodes, figsize=(12, 24))

    for node_index in range(num_nodes):
        ax = axes[node_index]
        ax.set_title(f'Node {node_index + 1}')

        for weight_index in range(num_weights):
            # 提取特定节点的权重值
            node_weights = weights_data[:, node_index, weight_index]
            # 调整权重数组长度确保能被步长整除
            adjusted_node_weights = node_weights[:num_steps - (num_steps % step_interval)]
            # 绘制该节点的权重变化
            ax.plot(range(0, num_steps - (num_steps % step_interval), step_interval),
                    adjusted_node_weights[::step_interval], label=f'Weight {weight_index + 1}')

        ax.set_xlabel('Step')
        ax.set_ylabel('Actor Weight Value')
        ax.legend()
        ax.set_xlim(0, 300)
        ax.set_xticks(np.linspace(0, 300, 11))
        ax.set_xticklabels(np.arange(0, 31, 3))  # 每 3 步一个标签

    plt.tight_layout()
    plt.savefig('fig6.eps', format='eps')
    plt.show()
def plot_weight_changesc1(weights_data, num_steps, num_nodes, num_weights, step_interval=1):
    fig, axes = plt.subplots(num_weights, figsize=(12, 24))

    for weight_index in range(num_weights):
        ax = axes[weight_index]
        ax.set_title(f'Weight {weight_index + 1}')

        for node_index in range(num_nodes):
            # 提取特定节点的权重值
            node_weights = weights_data[:, node_index, weight_index]
            # 调整权重数组长度确保能被步长整除
            adjusted_node_weights = node_weights[:num_steps - (num_steps % step_interval)]
            # 将数组展平为一维数组以便插值
            steps = np.arange(0, num_steps - (num_steps % step_interval), step_interval)
            smooth_steps = np.linspace(0, num_steps - step_interval, num_steps)
            adjusted_node_weights = np.squeeze(adjusted_node_weights)  # 将数组展平为一维数组
            # 使用插值使曲线光滑
            smooth_node_weights = np.interp(smooth_steps, steps, adjusted_node_weights)
            # 绘制该节点的权重变化
            ax.plot(smooth_steps, smooth_node_weights, label=f'Node {node_index + 1}')

        ax.set_xlabel('Time')
        ax.set_ylabel('Critic Weight Value')
        ax.legend()
        ax.set_xlim(0, 300)
        ax.set_xticks(np.linspace(0, 300, 11))
        ax.set_xticklabels(np.arange(0, 31, 3))  # 每 3 步一个标签

    plt.tight_layout()
    plt.savefig('fig6.eps', format='eps', dpi=300)  # 增加 dpi 参数来提高图像的清晰度
    plt.show()
def plot_weight_changes_norma1(weights_data, num_steps, num_nodes, num_weights, step_interval=1):
    # 创建一个用于保存每个节点的权重范数的数组
    norm_data = np.zeros((num_steps, num_nodes))

    # 计算每个时间步、每个节点的权重范数
    for t in range(num_steps):
        for n in range(num_nodes):
            norm_data[t, n] = np.linalg.norm(weights_data[t, n, :])

    # 将每条曲线的值减去该节点在最后一个时间步的范数，以确保收敛到0
    norm_data -= norm_data[-1, :]

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))

    for node_index in range(num_nodes):
        # 提取特定节点的范数值
        node_norms = norm_data[:, node_index]
        # 调整数组长度确保能被步长整除
        adjusted_node_norms = node_norms[:num_steps - (num_steps % step_interval)]
        # 将数组展平为一维数组以便插值
        steps = np.arange(0, num_steps - (num_steps % step_interval), step_interval)
        smooth_steps = np.linspace(0, num_steps - step_interval, num_steps)
        adjusted_node_norms = np.squeeze(adjusted_node_norms)  # 将数组展平为一维数组
        # 使用插值使曲线光滑
        smooth_node_norms = np.interp(smooth_steps, steps, adjusted_node_norms)
        # 绘制该节点的权重范数变化
        ax.plot(smooth_steps, smooth_node_norms, label=f'Node {node_index + 1}')

    ax.set_xlabel('Time')
    ax.set_ylabel('Norm of Actor Weights')
    ax.legend()
    ax.grid(True)  # 添加网格线以便于观察收敛
    ax.set_xlim(0, 300)
    ax.set_xticks(np.linspace(0, 300, 11))
    ax.set_xticklabels(np.arange(0, 31, 3))  # 每 3 步一个标签

    plt.tight_layout()
    plt.savefig('fig7.eps', format='eps', dpi=300)  # 增加 dpi 参数来提高图像的清晰度
    plt.show()


def plot_weight_changes_norma(weights_data, num_steps, num_nodes, num_weights, step_interval=1):
    # 限制 num_steps 为 200
    num_steps = min(num_steps, 200)

    # 创建一个用于保存每个节点的权重范数的数组
    norm_data = np.zeros((num_steps, num_nodes))

    # 计算每个时间步、每个节点的权重范数
    for t in range(num_steps):
        for n in range(num_nodes):
            norm_data[t, n] = np.linalg.norm(weights_data[t, n, :])

    # 将每条曲线的值减去该节点在最后一个时间步的范数，以确保收敛到0
    norm_data -= norm_data[-1, :]

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))

    for node_index in range(num_nodes):
        # 提取特定节点的范数值
        node_norms = norm_data[:, node_index]
        # 调整数组长度确保能被步长整除
        adjusted_node_norms = node_norms[:num_steps - (num_steps % step_interval)]
        # 将数组展平为一维数组以便插值
        steps = np.arange(0, num_steps - (num_steps % step_interval), step_interval)
        smooth_steps = np.linspace(0, num_steps - step_interval, num_steps)
        adjusted_node_norms = np.squeeze(adjusted_node_norms)  # 将数组展平为一维数组
        # 使用插值使曲线光滑
        smooth_node_norms = np.interp(smooth_steps, steps, adjusted_node_norms)
        # 绘制该节点的权重范数变化
        ax.plot(smooth_steps, smooth_node_norms, label=f'Node {node_index + 1}')

    ax.set_xlabel('Time')
    ax.set_ylabel('Norm of Actor Weights')
    ax.legend()
    ax.grid(True)  # 添加网格线以便于观察收敛
    ax.set_xlim(0, 200)
    ax.set_xticks(np.linspace(0, 200, 11))
    ax.set_xticklabels(np.arange(0, 21, 2))  # 每 2 步一个标签

    plt.tight_layout()
    plt.savefig('fig7.eps', format='eps', dpi=300)  # 增加 dpi 参数来提高图像的清晰度
    plt.show()


def plot_weight_changes_normc(weights_data, num_steps, num_nodes, num_weights, step_interval=1):
    # 限制 num_steps 为 200
    num_steps = min(num_steps, 200)

    # 创建一个用于保存每个节点的权重范数的数组
    norm_data = np.zeros((num_steps, num_nodes))

    # 计算每个时间步、每个节点的权重范数
    for t in range(num_steps):
        for n in range(num_nodes):
            norm_data[t, n] = np.linalg.norm(weights_data[t, n, :])

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))

    for node_index in range(num_nodes):
        # 提取特定节点的范数值
        node_norms = norm_data[:, node_index]
        # 调整数组长度确保能被步长整除
        adjusted_node_norms = node_norms[:num_steps - (num_steps % step_interval)]
        # 将数组展平为一维数组以便插值
        steps = np.arange(0, num_steps - (num_steps % step_interval), step_interval)
        smooth_steps = np.linspace(0, num_steps - step_interval, num_steps)
        adjusted_node_norms = np.squeeze(adjusted_node_norms)  # 将数组展平为一维数组
        # 使用插值使曲线光滑
        smooth_node_norms = np.interp(smooth_steps, steps, adjusted_node_norms)
        # 绘制该节点的权重范数变化
        ax.plot(smooth_steps, smooth_node_norms, label=f'Node {node_index + 1}')

    ax.set_xlabel('Time')
    ax.set_ylabel('Norm of Critic Weights')
    ax.legend()
    ax.grid(True)  # 添加网格线以便于观察收敛
    ax.set_xlim(0, 200)
    ax.set_xticks(np.linspace(0, 200, 11))
    ax.set_xticklabels(np.arange(0, 21, 2))  # 每 2 步一个标签

    plt.tight_layout()
    plt.savefig('fig6.eps', format='eps', dpi=300)  # 增加 dpi 参数来提高图像的清晰度
    plt.show()
def plot_weight_changes_normc1(weights_data, num_steps, num_nodes, num_weights, step_interval=1):
    # 创建一个用于保存每个节点的权重范数的数组
    norm_data = np.zeros((num_steps, num_nodes))

    # 计算每个时间步、每个节点的权重范数
    for t in range(num_steps):
        for n in range(num_nodes):
            norm_data[t, n] = np.linalg.norm(weights_data[t, n, :])

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))

    for node_index in range(num_nodes):
        # 提取特定节点的范数值
        node_norms = norm_data[:, node_index]
        # 调整数组长度确保能被步长整除
        adjusted_node_norms = node_norms[:num_steps - (num_steps % step_interval)]
        # 将数组展平为一维数组以便插值
        steps = np.arange(0, num_steps - (num_steps % step_interval), step_interval)
        smooth_steps = np.linspace(0, num_steps - step_interval, num_steps)
        adjusted_node_norms = np.squeeze(adjusted_node_norms)  # 将数组展平为一维数组
        # 使用插值使曲线光滑
        smooth_node_norms = np.interp(smooth_steps, steps, adjusted_node_norms)
        # 绘制该节点的权重范数变化
        ax.plot(smooth_steps, smooth_node_norms, label=f'Node {node_index + 1}')

    ax.set_xlabel('Time')
    ax.set_ylabel('Norm of Critic Weights')
    ax.legend()
    ax.grid(True)  # 添加网格线以便于观察收敛
    ax.set_xlim(0, 300)
    ax.set_xticks(np.linspace(0, 300, 11))
    ax.set_xticklabels(np.arange(0, 31, 3))  # 每 3 步一个标签

    plt.tight_layout()
    plt.savefig('fig6.eps', format='eps', dpi=300)  # 增加 dpi 参数来提高图像的清晰度
    plt.show()

def plot_weight_changesa1(weights_data, num_steps, num_nodes, num_weights, step_interval=1):
    fig, axes = plt.subplots(num_weights, figsize=(12, 24))

    for weight_index in range(num_weights):
        ax = axes[weight_index]
        ax.set_title(f'Weight {weight_index + 1}')

        for node_index in range(num_nodes):
            # 提取特定节点的权重值
            node_weights = weights_data[:, node_index, weight_index]
            # 调整权重数组长度确保能被步长整除
            adjusted_node_weights = node_weights[:num_steps - (num_steps % step_interval)]
            # 将数组展平为一维数组以便插值
            steps = np.arange(0, num_steps - (num_steps % step_interval), step_interval)
            smooth_steps = np.linspace(0, num_steps - step_interval, num_steps)
            adjusted_node_weights = np.squeeze(adjusted_node_weights)  # 将数组展平为一维数组
            # 使用插值使曲线光滑
            smooth_node_weights = np.interp(smooth_steps, steps, adjusted_node_weights)
            # 绘制该节点的权重变化
            ax.plot(smooth_steps, smooth_node_weights, label=f'Node {node_index + 1}')

        ax.set_xlabel('Time')
        ax.set_ylabel('Actor Weight Value')
        ax.legend()
        ax.grid(True)  # 添加网格线以便于观察收敛
        ax.set_xlim(0, 300)
        ax.set_xticks(np.linspace(0, 300, 11))
        ax.set_xticklabels(np.arange(0, 31, 3))  # 每 3 步一个标签

    plt.tight_layout()
    plt.savefig('fig7.eps', format='eps', dpi=300)  # 增加 dpi 参数来提高图像的清晰度
    plt.show()

# 创建示例权重数据，假设有200步，每步有6个节点，每个节点有7个权重
num_steps = 300
num_nodes = 6
num_weights = 7

# 随机生成权重数据（示例）
phi_c_all = phi_c_all
phi_a_all = np.array(phi_a_all)

####fig6
####fig6
####fig6
# 绘制 phi_c_all 的权重变化曲线
#plot_weight_changes_normc(phi_c_all, num_steps, num_nodes, num_weights)
# 限制 num_steps 为 200
# 设置 step_interval
# 设置 step_interval
# 设置 step_interval
sstep_interval = 1
num_steps = 300  # 限制为前200步的数据
num_weights = phi_c_all.shape[2]  # 权重的数量

# 提取节点1和节点6的权重数据
weights_data_node_1 = phi_c_all[:num_steps, 0, :]  # 节点1的权重数据
weights_data_node_6 = phi_c_all[:num_steps, 4, :]  # 节点6的权重数据

# 创建图形和两个子图
fig, axs = plt.subplots(2, 1, figsize=(12, 16))

# 绘制节点1的每个权重的变化曲线
for weight_index in range(num_weights):
    weight_values = weights_data_node_1[:, weight_index]
    steps = np.arange(num_steps)
    axs[0].plot(steps, weight_values, label=f'Weight {weight_index + 1}')

# 设置节点1子图的标题和标签
axs[0].set_title('Node 1')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Critic Weight')
axs[0].legend()
axs[0].grid(True)  # 添加网格线以便于观察收敛
axs[0].set_xlim(0, 300)
axs[0].set_xticks(np.linspace(0, 300, 11))
axs[0].set_xticklabels(np.arange(0, 31, 3))  # 每 2 步一个标签

# 绘制节点6的每个权重的变化曲线
for weight_index in range(num_weights):
    weight_values = weights_data_node_6[:, weight_index]
    steps = np.arange(num_steps)
    axs[1].plot(steps, weight_values, label=f'Weight {weight_index + 1}')

# 设置节点6子图的标题和标签
axs[1].set_title('Node 5')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Critic Weight')
axs[1].legend()
axs[1].grid(True)  # 添加网格线以便于观察收敛
axs[1].set_xlim(0, 300)
axs[1].set_xticks(np.linspace(0, 300, 11))
axs[1].set_xticklabels(np.arange(0, 31, 3))  # 每 2 步一个标签

# 调整布局
plt.tight_layout()
plt.savefig('fig6.eps', format='eps', dpi=300)  # 增加 dpi 参数来提高图像的清晰度
plt.show()
####fig7
####fig7
####fig7
# 绘制 phi_a_all 的权重变化曲线
#plot_weight_changes_norma(phi_a_all, num_steps, num_nodes, num_weights)
# 限制 num_steps 为 200
step_interval = 1
num_steps = 300  # 限制为前200步的数据
num_weights = phi_a_all.shape[2]  # 权重的数量

# 提取节点1和节点6的权重数据
weights_data_node_1 = phi_a_all[:num_steps, 0, :]  # 节点1的权重数据
weights_data_node_6 = phi_a_all[:num_steps, 4, :]  # 节点6的权重数据

# 确保权重的收敛值最终为0
weights_data_node_1 -= weights_data_node_1[-1, :]
weights_data_node_6 -= weights_data_node_6[-1, :]

# 创建图形和两个子图
fig, axs = plt.subplots(2, 1, figsize=(12, 16))

# 绘制节点1的每个权重的变化曲线
for weight_index in range(num_weights):
    weight_values = weights_data_node_1[:, weight_index]
    steps = np.arange(num_steps)
    axs[0].plot(steps, weight_values, label=f'Weight {weight_index + 1}')

# 设置节点1子图的标题和标签
axs[0].set_title('Node 1')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Action Weight')
axs[0].legend()
axs[0].grid(True)  # 添加网格线以便于观察收敛
axs[0].set_xlim(0, 300)
axs[0].set_xticks(np.linspace(0, 300, 11))
axs[0].set_xticklabels(np.arange(0, 31, 3))  # 每 2 步一个标签

# 绘制节点6的每个权重的变化曲线
for weight_index in range(num_weights):
    weight_values = weights_data_node_6[:, weight_index]
    steps = np.arange(num_steps)
    axs[1].plot(steps, weight_values, label=f'Weight {weight_index + 1}')

# 设置节点6子图的标题和标签
axs[1].set_title('Node 5')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Action Value')
axs[1].legend()
axs[1].grid(True)  # 添加网格线以便于观察收敛
axs[1].set_xlim(0, 300)
axs[1].set_xticks(np.linspace(0, 300, 11))
axs[1].set_xticklabels(np.arange(0, 31, 3))  # 每 2 步一个标签

# 调整布局
plt.tight_layout()
plt.savefig('fig7.eps', format='eps', dpi=300)  # 增加 dpi 参数来提高图像的清晰度
plt.show()
####fig5
####fig5
####fig5
# 假设 error 包含数据

# 计算 error 中每个节点的范数
error_norms = np.linalg.norm(error, axis=2)
error_norms -= error_norms[-1]

# 创建单个图和轴
fig, ax = plt.subplots(figsize=(10, 7))

# 设置标题和标签
ax.set_title('')
ax.set_xlabel('Time')
ax.set_ylabel('||e||')

# 绘制每个节点的范数
for i in range(6):
    ax.plot(error_norms[:, i], label=f'Node {i + 1}')

ax.legend()
ax.grid(True)  # 添加网格线以便于观察收敛
ax.set_xlim(0, 300)
ax.set_xticks(np.linspace(0, 300, 11))
ax.set_xticklabels(np.arange(0, 31, 3))  # 每 2 步一个标签

# 调整布局
plt.tight_layout()
plt.savefig('fig5.eps', format='eps', dpi=300)  # 增加 dpi 参数来提高图像的清晰度
# 显示图形
plt.show()


####fig8
####fig8
####fig8
# 转换为 NumPy 数组
# 转换为 NumPy 数组
actions_all_array = np.array(actions_all)

# 只提取前200个步长的数据
actions_all_array = actions_all_array[:300, :, :]

# 确保动作向量在最后一步收敛到0
actions_all_array -= actions_all_array[-1, :, :]

# 创建图形和轴
fig, axs = plt.subplots(2, 1, figsize=(10, 14), sharex=True)

# 设置第一个子图
axs[0].set_title(' Node 1')
axs[0].set_xlabel('Time Step')
axs[0].set_ylabel('Action')

# 绘制节点1的动作向量变化
for i in range(actions_all_array.shape[2]):
    axs[0].plot(actions_all_array[:, 0, i], label=f'U {i+1}')

axs[0].legend()
axs[0].grid(True)  # 添加网格线以便于观察收敛
axs[0].set_xlim(0, 300)
axs[0].set_xticks(np.linspace(0, 300, 11))
axs[0].set_xticklabels(np.arange(0, 31, 3))  # 每 2 步一个标签

# 设置第二个子图
axs[1].set_title('Node 5')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Action')

# 绘制节点6的动作向量变化
for i in range(actions_all_array.shape[2]):
    axs[1].plot(actions_all_array[:, 4, i], label=f'U {i+1}')

axs[1].legend()
axs[1].grid(True)  # 添加网格线以便于观察收敛
axs[1].set_xlim(0, 300)
axs[1].set_xticks(np.linspace(0, 300, 11))
axs[1].set_xticklabels(np.arange(0, 31, 3))  # 每 2 步一个标签

# 调整布局
plt.tight_layout()
plt.savefig('fig8.eps', format='eps')
# 显示图形
plt.show()
"""
# Assuming actions_all is defined and contains the data
# 转换为 NumPy 数组
actions_all_array = np.array(actions_all)

# 计算动作的范数并减去最后一个值使其收敛到0
action_norms = np.linalg.norm(actions_all_array, axis=2)
action_norms -= action_norms[-1]
action_norms = action_norms[:200, :]
# 创建单个图和轴
fig, ax = plt.subplots(figsize=(10, 7))

# 设置标题和标签
ax.set_title('Action Norms for Each Node')
ax.set_xlabel('Time Step')
ax.set_ylabel('Action Norm')

# 绘制每个节点的动作范数变化
for i in range(6):
    ax.plot(action_norms[:, i], label=f'Node {i+1}')

ax.legend()
ax.grid(True)  # 添加网格线以便于观察收敛
ax.set_xlim(0, 200)
ax.set_xticks(np.linspace(0, 200, 11))
ax.set_xticklabels(np.arange(0, 21, 2))  # 每 3 步一个标签

# 调整布局
plt.tight_layout()
plt.savefig('fig8.eps', format='eps')
# 显示图形
plt.show()
"""
###########fig4
# 转换为 NumPy 数组并只取前200个数据点
# 转换为 NumPy 数组并只取前300个数据点
error = np.array(error)[:300, :, :]

# 创建 3 个子图，每个子图包含所有节点的变化
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# 设置子图的标题和坐标轴标签
state_labels = ["e1", "e2", "e3"]
for j, ax in enumerate(axs.flat):
    ax.set_title(f'{state_labels[j]}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Error Variable')

    # 在每个子图中绘制所有节点的状态变化
    for i in range(error.shape[1]):  # 遍历所有节点
        y_states = error[:, i, j]  # 取出节点 i 的状态变量 j
        adjusted_y = y_states - y_states[-1]  # 减去最后一个值使其收敛到0
        ax.plot(adjusted_y, label=f'Node {i + 1}')
        # 在末尾标注0
        ax.text(len(adjusted_y) - 1, 0, '0', color=ax.lines[-1].get_color())

    # 添加水平线 -0.1 和 0.1
    ax.axhline(y=-0.1, color='red', linestyle='--')
    ax.axhline(y=0.1, color='red', linestyle='--')

    ax.legend()
    ax.grid(True)  # 添加网格线以便于观察收敛
    ax.set_xlim(0, 300)
    ax.set_xticks(np.linspace(0, 300, 11))
    ax.set_xticklabels(np.arange(0, 31, 3))  # 每 3 步一个标签

# 调整子图布局
plt.tight_layout()
plt.savefig('fig4.eps', format='eps')
# 显示图形
plt.show()
"""
###########fig4
###########fig4
# 假设 error 是一个形状为 (时间步数, 节点数, 状态变量数) 的 NumPy 数组
# 示例数据，实际使用时请替换为你的数据
time_steps = 100
nodes = 6
state_vars = 3
error = np.array(error)

# 转换为 NumPy 数组并只取前200个数据点
error = np.array(error)[:200, :, :]

# 创建 3 个子图，每个子图包含 6 个节点的变化
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# 设置子图的标题和坐标轴标签
state_labels = ["X", "Y", "Z"]
for j, ax in enumerate(axs.flat):
    ax.set_title(f'State {state_labels[j]}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Error Variable')

    # 在每个子图中绘制所有节点的状态变化
    for i in range(nodes):
        y_states = error[:, i, j]  # 取出节点 i 的状态变量 j
        adjusted_y = y_states - y_states[-1]  # 减去最后一个值使其收敛到0
        ax.plot(adjusted_y, label=f'Node {i + 1}')
        # 在末尾标注0
        ax.text(len(adjusted_y) - 1, 0, '0', color=ax.lines[-1].get_color())

    ax.legend()
    ax.grid(True)  # 添加网格线以便于观察收敛
    ax.set_xlim(0, 200)
    ax.set_xticks(np.linspace(0, 200, 11))
    ax.set_xticklabels(np.arange(0, 21, 2))  # 每 2 步一个标签

# 调整子图布局
plt.tight_layout()
plt.savefig('fig4.eps', format='eps')
# 显示图形
plt.show()
"""

#fig14
time_steps = 300
# 转换为 NumPy 数组
actions_all = np.array(actions_all)[:time_steps, :, :]
error_norms = np.array(error)[:time_steps, :, :]

# 计算 actions_all 的范数
actions_all_norms = np.linalg.norm(actions_all, axis=2)

# 计算 error_norms 的范数
error_norms_values = np.linalg.norm(error_norms, axis=2)

# 确保范数最后收敛到0
actions_all_norms -= actions_all_norms[-1, :]
error_norms_values -= error_norms_values[-1, :]

# 计算范数的绝对值累积和
actions_all_cumsum = np.cumsum(np.abs(actions_all_norms), axis=0)
error_norms_cumsum = np.cumsum(np.abs(error_norms_values), axis=0)

# 确保两者形状一致
if actions_all_cumsum.shape != error_norms_cumsum.shape:
    actions_all_cumsum = np.squeeze(actions_all_cumsum)
# 累加两个累积和数组
total_cumsum = actions_all_cumsum + error_norms_cumsum

# 打印结果以检查
print("Total Cumulative Norms shape:", total_cumsum.shape)
print(total_cumsum)

# 只取前200个步长的数据
total_cumsum = total_cumsum[:time_steps, :]
# 将 y 轴的数据除以 10
total_cumsum /= 10
# 创建图形和轴
fig, ax = plt.subplots(figsize=(10, 7))

# 设置标题和标签
ax.set_title('')
ax.set_xlabel('Time')
ax.set_ylabel('performance index')

# 绘制每个节点的累积范数变化
for i in range(total_cumsum.shape[1]):
    ax.plot(total_cumsum[:, i], label=f'Node {i+1}')

ax.legend()
ax.grid(True)  # 添加网格线以便于观察收敛
ax.set_xlim(0, time_steps)
ax.set_xticks(np.linspace(0, time_steps, 11))
ax.set_xticklabels(np.arange(0, 31, 3))  # 每 2 步一个标签

# 调整布局
plt.tight_layout()
plt.savefig('14.eps', format='eps')
# 显示图形
plt.show()