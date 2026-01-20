import numba as nb
import numpy as np
import pickle

@nb.njit(cache=True)
def allocate_fs_block(fs_usage_uv, start_fs, end_fs, Bool):
    for fs in range(start_fs, end_fs + 1):
        fs_usage_uv[fs] = Bool

@nb.njit(cache=True)
def expected_allocated_slots(N, M, p, k):
    """
    计算到达 k 个业务后被分配时隙数的期望值

    参数:
    N: 总时隙数
    M: 最大可能的需求时隙数
    p: 列表，p[i] 表示需求量为 i 个时隙的概率 (i从1到M)
    k: 到达的业务数量

    返回:
    期望分配时隙数
    """
    # 初始化DP表: V[r][m] 表示剩余 r 个时隙，还有 m 个业务时的期望分配量
    # V = [[0.0] * (k + 1) for _ in range(N + 1)]
    V = np.zeros((N+1,k+1), dtype=float)

    # 预计算累积概率和加权期望，加速计算
    # cum_prob[t] = sum_{i=1}^{t} p(i)
    # weighted_sum[t] = sum_{i=1}^{t} i * p(i)
    cum_prob = np.zeros(M+2, dtype=float)
    weighted_sum = np.zeros(M + 2, dtype=float)
    # cum_prob = [0.0] * (M + 2)
    # weighted_sum = [0.0] * (M + 2)

    for i in range(1, M + 1):
        cum_prob[i] = cum_prob[i - 1] + p[i - 1]
        weighted_sum[i] = weighted_sum[i - 1] + i * p[i - 1]

    # 按 m 从小到大计算
    for m in range(1, k + 1):
        for r in range(0, N + 1):
            # 当 r <= 0 时，后续业务无法分配
            if r <= 0:
                V[r][m] = 0.0
                continue

            # t = min(M, r)
            t = min(M, r)

            # 接受概率
            P_acc = cum_prob[t]

            # 在可接受范围内的平均需求
            E_le_t = weighted_sum[t]

            # 计算期望
            # 第一部分: 当前业务的期望分配
            value = E_le_t

            # 第二部分: 接受后未来的期望
            future_acc = 0.0
            for i in range(1, t + 1):
                future_acc += p[i - 1] * V[r - i][m - 1]

            # 第三部分: 阻塞后未来的期望
            future_rej = (1 - P_acc) * V[r][m - 1]

            V[r][m] = value + future_acc + future_rej

    return V[N][k]


# 测试示例：均匀分布情况
if __name__ == "__main__":
    # 示例1: 均匀分布
    max_N = 800
    M = 40
    max_K = 10

    # 均匀分布: p(i) = 1/40
    p_uniform = np.array([1 / M] * M, dtype=float)

    E = np.zeros((max_N+1,max_K+1),dtype=float)
    E = [[0.0]*11 for _ in range(max_N+1)]
    for N in range(max_N+1):
        for k in range(max_K+1):
            E[N][k] = expected_allocated_slots(N, M, p_uniform, k)
            print(f"N={N}, M={M}, k={k}, 均匀分布时的期望分配时隙数: {E[N][k]:.4f}")
            # print(f"近似值 20.5*k = {20.5 * k}")
    for e in E:
        print(e)
    print('\n\n\n\n')

    E_path = 'pre_calc_E'
    with open(f'{E_path}/E_N{max_N}_M{M}_K{max_K}.pkl', 'wb') as f:
        pickle.dump(E, f)

    # 加载
    with open(f'{E_path}/E_N{max_N}_M{M}_K{max_K}.pkl', 'rb') as f:
        E_loaded = pickle.load(f)
    for e in E_loaded:
        print(e)

    # # 示例2: 几何分布(截断)
    # print("\n--- 几何分布示例 ---")
    # N = 50
    # M = 20
    # k = 8
    #
    # # 创建几何分布(截断在M): p(i) ∝ q^(i-1)
    # q = 0.8
    # geom_probs = np.array([q ** (i - 1) for i in range(1, M + 1)], dtype=float)
    # total = np.sum(geom_probs)
    # p_geom = np.array([prob / total for prob in geom_probs], dtype=float)
    #
    # result = expected_allocated_slots(N, M, p_geom, k)
    # print(f"N={N}, M={M}, k={k}, 几何分布时的期望分配时隙数: {result:.4f}")
    #
    # # 验证: 打印需求分布的统计量
    # import numpy as np
    #
    # mean_demand = sum((i + 1) * p_geom[i] for i in range(M))
    # print(f"平均需求: {mean_demand:.4f}")