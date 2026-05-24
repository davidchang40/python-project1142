# -*- coding: utf-8 -*-


def Z_ci(alpha):
t_table = {
}


chi_square_table = {
    25: {0.995: 10.520, 0.975: 13.120, 0.950: 14.611, 0.050: 37.652, 0.025: 40.646, 0.005: 46.928},
    26: {0.995: 11.160, 0.975: 13.844, 0.950: 15.379, 0.050: 38.885, 0.025: 41.923, 0.005: 48.290},
    27: {0.995: 11.808, 0.975: 14.573, 0.950: 16.151, 0.050: 40.113, 0.025: 43.195, 0.005: 49.645},
    28: {0.995: 12.461, 0.975: 15.308, 0.950: 16.928, 0.050: 41.337, 0.025: 44.461, 0.005: 50.993},
    29: {0.995: 13.121, 0.975: 16.047, 0.950: 17.708, 0.050: 42.557, 0.025: 45.722, 0.005: 52.336},
}



def confident_interval():
    stat_type = input("請輸入您要計算的是母體平均數(mu)還是母體比例(p)還是變異數(var)：")
    s     = float(input("請輸入母體標準差或樣本標準差(s)："))
    n     = float(input("請輸入樣本數："))
    alpha = float(input("請輸入信賴水準(alpha)："))

    if stat_type == 'mu':
        x_bar = float(input("請輸入母體(mu)或樣本平均數(x-bar)的值："))
        d = input("請選擇要使用z計算還是t計算：")
        if d == 'z':
            z = float(Z_ci(float(alpha)))
        elif d == 't':
            t = float(T_ci(float(alpha), int(n - 1)))
    elif stat_type == 'p':
        p = float(input("請輸入母體比例(P)或樣本比例(p)的值："))
        z = float(Z_ci(float(alpha)))
    elif stat_type == 'var':
        chi_upper = float(chi_ci_upper(float(alpha), int(n - 1)))
        chi_lower = float(chi_ci_lower(float(alpha), int(n - 1)))

    if stat_type == 'mu' and d == 'z':
        upper_bound = x_bar + z * s / n ** 0.5
        lower_bound = x_bar - z * s / n ** 0.5
    elif stat_type == 'mu' and d == 't':
        upper_bound = x_bar + t * s / n ** 0.5
        lower_bound = x_bar - t * s / n ** 0.5
    elif stat_type == 'p':
        upper_bound = p + z * s / n ** 0.5
        lower_bound = p - z * s / n ** 0.5
    elif stat_type == 'var':
        upper_bound = (n - 1) * s ** 2 / chi_upper
        lower_bound = (n - 1) * s ** 2 / chi_lower

    concluded_interval = (lower_bound, upper_bound)
    return concluded_interval


# ============================================================
# 假設檢定區
# ============================================================
import numpy as np
from scipy import stats

# --- 條件變數 ---
is_proportion       = None
is_population_known = None
is_paired           = None
is_raw_data         = None
n_samples           = None
tail_type           = None
alpha_ht            = None   # 與信賴區間的 alpha 分開命名，避免衝突

# --- 數據 ---
raw_data            = []
proportion          = []
p0                  = 0

# --- 統計量 ---
mu                  = None
x_bar               = None
sigma               = None
n_ht                = None   # 同上，避免與信賴區間的 n 衝突
x_bar2              = None
sigma2              = None
df                  = None


def _reset_hypothesis_globals():
    """每次執行假設檢定前重置全域變數，避免跨次污染"""
    global is_proportion, is_population_known, is_paired, is_raw_data
    global n_samples, tail_type, alpha_ht, raw_data, proportion, p0
    global mu, x_bar, sigma, n_ht, x_bar2, sigma2, df

    is_proportion       = None
    is_population_known = None
    is_paired           = None
    is_raw_data         = None
    n_samples           = None
    tail_type           = None
    alpha_ht            = None
    raw_data            = []
    proportion          = []
    p0                  = 0
    mu                  = None
    x_bar               = None
    sigma               = None
    n_ht                = None
    x_bar2              = None
    sigma2              = None
    df                  = None


def ask(prompt, valid_options):
    while True:
        answer = input(prompt)
        if answer in valid_options:
            return answer
        print(f"請輸入有效選項：{valid_options}")


def run_test():
    global is_proportion, is_population_known, is_paired
    global is_raw_data, n_samples, tail_type, alpha_ht
    global p0, raw_data, proportion, mu
    global x_bar, x_bar2, sigma, sigma2, n_ht

    _reset_hypothesis_globals()

    is_proportion       = (ask("數值/比例？(1=數值 / 2=比例)：",         ["1","2"])    == "2")
    n_samples           = int(ask("樣本數？(1 / 2)：",                   ["1","2"]))
    is_paired           = (n_samples == 2) and \
                          (ask("獨立/配對？(1=獨立 / 2=配對)：",         ["1","2"])    == "2")
    is_population_known = not is_proportion and \
                          (ask("母體已知？(1=是 / 2=否)：",               ["1","2"])    == "1")
    is_raw_data         = (ask("輸入方式？(1=原始數據 / 2=統計量)：",     ["1","2"])    == "1")
    alpha_ht            = {"90": 0.10, "95": 0.05, "99": 0.01}\
                          [ask("信心水準？(90/95/99)：",                  ["90","95","99"])]
    tail_type           = {"1": "left", "2": "right", "3": "two"}\
                          [ask("尾數？(1=左尾 / 2=右尾 / 3=雙尾)：",     ["1","2","3"])]

    if is_proportion:
        p0 = float(input("輸入假設比例 p₀："))
        if n_samples == 1:
            n_ht = int(input("輸入樣本數 n："))
            proportion.append(float(input("輸入樣本比例 p̂：")))
        else:
            n_ht = [int(input("輸入第1組樣本數 n₁：")), int(input("輸入第2組樣本數 n₂："))]
            for i in range(n_samples):
                proportion.append(float(input(f"輸入第 {i+1} 組樣本比例 p̂：")))
    else:
        mu = float(input("輸入假設的母體平均數 μ₀："))
        if is_raw_data:
            for i in range(n_samples):
                raw_data.append(list(map(float, input(f"輸入第 {i+1} 組數據（空格分隔）：").split(" "))))
        else:
            if n_samples == 1:
                x_bar = float(input("輸入樣本平均數 x̄："))
                sigma = float(input("輸入標準差 σ："))
                n_ht  = int(input("輸入樣本數 n："))
            else:
                x_bar  = float(input("輸入第1組樣本平均數 x̄₁："))
                x_bar2 = float(input("輸入第2組樣本平均數 x̄₂："))
                sigma  = float(input("輸入第1組標準差 σ₁："))
                sigma2 = float(input("輸入第2組標準差 σ₂："))
                n_ht   = [int(input("輸入第1組樣本數 n₁：")),
                          int(input("輸入第2組樣本數 n₂："))]

    if is_proportion:
        if n_samples == 1:
            test_stat = calculate_stats_z_one()
        else:
            test_stat = calculate_stats_z_two()
    else:
        if is_population_known:
            if n_samples == 1:
                test_stat = calculate_stats_z_one()
            else:
                test_stat = calculate_stats_z_two()
        else:
            if n_samples == 1:
                test_stat = calculate_stats_t_one()
            else:
                if is_paired:
                    test_stat = calculate_stats_t_pair()
                else:
                    test_stat = calculate_stats_t_ind()

    check_hypothesis(test_stat)


# ── 計算函式 ──

def calculate_stats_z_one():
    global x_bar, sigma, n_ht
    if is_proportion:
        x_bar = proportion[0]
        sigma = np.sqrt(p0 * (1 - p0) / n_ht)
    elif is_raw_data:
        data  = raw_data[0]
        n_ht  = len(data)
        x_bar = np.mean(data)
        sigma = np.std(data, ddof=0)
    z_stat = (x_bar - mu) / (sigma / np.sqrt(n_ht))
    return z_stat


def calculate_stats_z_two():
    global x_bar, x_bar2, sigma, sigma2, n_ht
    if is_proportion:
        x_1  = proportion[0]
        x_2  = proportion[1]
        n1, n2 = n_ht[0], n_ht[1]
        p_hat  = (x_1 * n1 + x_2 * n2) / (n1 + n2)
        D      = np.sqrt(p_hat * (1 - p_hat) * (1/n1 + 1/n2))
        return (x_1 - x_2 - mu) / D
    elif is_raw_data:
        data1, data2 = raw_data[0], raw_data[1]
        n_ht   = [len(data1), len(data2)]
        sigma  = np.std(data1, ddof=0)
        sigma2 = np.std(data2, ddof=0)
        x_bar  = np.mean(data1)
        x_bar2 = np.mean(data2)
    se = np.sqrt(sigma**2 / n_ht[0] + sigma2**2 / n_ht[1])
    return (x_bar - x_bar2 - mu) / se


def calculate_stats_t_one():
    global x_bar, sigma, n_ht, df
    if is_raw_data:
        data  = raw_data[0]
        n_ht  = len(data)
        x_bar = np.mean(data)
        sigma = np.std(data, ddof=1)
    df     = n_ht - 1
    t_stat = (x_bar - mu) / (sigma / np.sqrt(n_ht))
    return t_stat, df


def calculate_stats_t_ind():
    global x_bar, x_bar2, sigma, sigma2, n_ht, df
    if is_raw_data:
        data1, data2 = raw_data[0], raw_data[1]
        n_ht   = [len(data1), len(data2)]
        sigma  = np.std(data1, ddof=1)
        sigma2 = np.std(data2, ddof=1)
        x_bar  = np.mean(data1)
        x_bar2 = np.mean(data2)
    s2p    = ((n_ht[0]-1)*sigma**2 + (n_ht[1]-1)*sigma2**2) / (n_ht[0]+n_ht[1]-2)
    se     = np.sqrt(s2p * (1/n_ht[0] + 1/n_ht[1]))
    df     = n_ht[0] + n_ht[1] - 2
    return (x_bar - x_bar2 - mu) / se, df


def calculate_stats_t_pair():
    global x_bar, sigma, n_ht, df
    if is_raw_data:
        data1, data2 = raw_data[0], raw_data[1]
        n_ht  = len(data1)
        diff  = [data1[i] - data2[i] for i in range(n_ht)]
        x_bar = np.mean(diff)
        sigma = np.std(diff, ddof=1)
    df     = n_ht - 1
    t_stat = (x_bar - mu) / (sigma / np.sqrt(n_ht))
    return t_stat, df


# ── 檢定函式 ──

def get_critical_value():
    if is_proportion or is_population_known:
        if tail_type == "left":
            return stats.norm.ppf(alpha_ht)
        elif tail_type == "right":
            return stats.norm.ppf(1 - alpha_ht)
        else:
            return stats.norm.ppf(1 - alpha_ht / 2)
    else:
        if tail_type == "left":
            return stats.t.ppf(alpha_ht, df)
        elif tail_type == "right":
            return stats.t.ppf(1 - alpha_ht, df)
        else:
            return stats.t.ppf(1 - alpha_ht / 2, df)


def check_hypothesis(test_stat):
    stat           = test_stat[0] if isinstance(test_stat, tuple) else test_stat
    critical_value = get_critical_value()

    if tail_type == "left":
        conclusion = "拒絕 H₀" if stat < critical_value else "不拒絕 H₀"
    elif tail_type == "right":
        conclusion = "拒絕 H₀" if stat > critical_value else "不拒絕 H₀"
    else:
        conclusion = "拒絕 H₀" if abs(stat) > abs(critical_value) else "不拒絕 H₀"

    print(f"統計量：{stat:.4f}")
    print(f"臨界值：{critical_value:.4f}")
    print(conclusion)


# ============================================================
# 主選單
# ============================================================

def main():
    while True:
        print("\n======= 歡迎使用統計計算機 =======")
        print("1. 信賴區間")
        print("2. 假設檢定")
        print("3. 離開")
        c = input("請輸入選項：").strip()

        if c == "1":
            result = confident_interval()
            print(f"\n信賴區間：{result}")
        elif c == "2":
            run_test()
        elif c == "3":
            print("感謝使用！")
            break
        else:
            print("請輸入 1 / 2 / 3")

if __name__ == "__main__":
    main()