import numpy as np
from scipy import stats

# ============================================================
# 全域變數區
# ============================================================

# --- 使用者條件（由 run_test 產出）---
is_proportion       = None   # bool：True=比例型 / False=數值型
is_population_known = None   # bool：True=母體已知(Z) / False=未知(T)
is_paired           = None   # bool：True=配對 / False=獨立（n_samples==2 才有意義）
is_raw_data         = None   # bool：True=輸入原始數據 / False=直接輸入統計量
n_samples           = None   # int：1 或 2
tail_type           = None   # str：'left' / 'right' / 'two'
alpha               = None   # float：0.10 / 0.05 / 0.01

# --- 原始數據暫存 ---
raw_data            = []     # list of list：[[樣本1數據], [樣本2數據]]

# --- 處理後統計量（由 calculate_stats 寫入，供 check_hypothesis 使用）---
mu                  = None   # float：母體平均數 或 假設比例 p0
x_bar               = None   # float：樣本平均數 或 樣本比例 p_hat
sigma               = None   # float：標準差（母體已知用 σ，未知用 s）
n                   = None   # int 或 list：樣本數（雙樣本時為 [n1, n2]）

# 雙樣本專用
x_bar2              = None   # float：第二組樣本平均數
sigma2              = None   # float：第二組標準差


# ============================================================
# 啟動區
# ============================================================
def ask(prompt, valid_options):
    # """
    # 局域變數：
    #     answer  (str)  ── 使用者輸入的原始字串
    # """
    while True:
        answer = input(prompt)
        if answer in valid_options:
            return answer
        print(f"請輸入有效選項：{valid_options}")

def run_test():
    # """
    # 負責：問答流程、設定所有全域條件變數、呼叫計算函式、呼叫檢定函式

    # 使用的全域變數（全部寫入）：
    #     is_proportion, is_population_known, is_paired,
    #     is_raw_data, n_samples, tail_type, alpha, raw_data, mu

    # 局域變數：
    #     answer      (str)  ── 每一題使用者的原始輸入
    #     confidence  (str)  ── 使用者選的信心水準字串，轉換成 alpha
    # """
    global is_proportion, is_population_known, is_paired
    global is_raw_data, n_samples, tail_type, alpha

    is_proportion       = (ask("數值/比例？(1=數值 / 2=比例)：",         ["1","2"])    == "2")
    n_samples           = int(ask("樣本數？(1 / 2)：",                   ["1","2"]))
    is_paired           = (n_samples == 2) and \
                          (ask("獨立/配對？(1=獨立 / 2=配對)：",         ["1","2"])    == "2")
    is_population_known = not is_proportion and \
                          (ask("母體已知？(1=是 / 2=否)：",               ["1","2"])    == "1")
    is_raw_data         = (ask("輸入方式？(1=原始數據 / 2=統計量)：",     ["1","2"])    == "1")
    alpha               = {"90": 0.10, "95": 0.05, "99": 0.01}\
                          [ask("信心水準？(90/95/99)：",                  ["90","95","99"])]
    tail_type           = {"1": "left", "2": "right", "3": "two"}\
                          [ask("尾數？(1=左尾 / 2=右尾 / 3=雙尾)：",     ["1","2","3"])]
                          
    if is_proportion:
        # --- 比例型（一律為 Z 檢定）---
        if n_samples == 1:
            test_stat = calculate_stats_z_one()     # 情境 1：單比例 Z 檢定
        else:
            test_stat = calculate_stats_z_two()     # 情境 2：雙比例 Z 檢定
            
    else:
        # --- 數值型 ---
        if is_population_known:
            # 母體變異數已知（使用 Z 檢定）
            if n_samples == 1:
                test_stat = calculate_stats_z_one() # 情境 3：單樣本 Z 檢定
            else:
                # 實務上母體已知的雙樣本多為獨立檢定
                test_stat = calculate_stats_z_two() # 情境 4：雙樣本 Z 檢定 
                
        else:
            # 母體變異數未知（使用 T 檢定）
            if n_samples == 1:
                test_stat = calculate_stats_t_one() # 情境 5：單樣本 T 檢定
            else:
                if is_paired:
                    test_stat = calculate_stats_t_pair() # 情境 6：配對 T 檢定
                else:
                    test_stat = calculate_stats_t_ind()  # 情境 7：獨立雙樣本 T 檢定

    # 統一將計算結果交給檢定區處理
    check_hypothesis(test_stat)
        
    pass


# ============================================================
# 計算區
# ============================================================

def calculate_stats_z_one():
    """
    單樣本 Z 檢定 / 單比例 Z 檢定 的數據處理 + 統計量計算

    讀取的全域變數：
        is_raw_data, is_proportion, raw_data, mu,
        x_bar, sigma, n

    寫入的全域變數：
        x_bar, sigma, n

    局域變數：
        data        (list)   ── raw_data[0] 的捷徑
        z_stat      (float)  ── 計算出的 Z 統計量

    回傳：z_stat (float)
    """
    global x_bar, sigma, n          # 宣告要修改的全域變數
    if is_raw_data:
        data = raw_data[0]          # 取第一組（單樣本只有一組）
        n     = len(data)
        x_bar = np.mean(data)
        sigma = np.std(data, ddof=0)  # ddof=0 代表母體標準差
        # mu 由使用者另外輸入，不在這裡算
    else:
        # 使用者直接輸入，全域變數已經有值，不用動
        pass
    z_test=(x_bar-mu)/(sigma/np.sqrt(n))
    return z_test  
    pass


def calculate_stats_z_two():
    """
    雙樣本 Z 檢定 / 雙比例 Z 檢定 的數據處理 + 統計量計算

    讀取的全域變數：
        is_raw_data, is_proportion, raw_data, mu,
        x_bar, x_bar2, sigma, sigma2, n

    寫入的全域變數：
        x_bar, x_bar2, sigma, sigma2, n

    局域變數：
        data1, data2  (list)   ── raw_data[0], raw_data[1] 的捷徑
        se            (float)  ── 標準誤
        z_stat        (float)  ── 計算出的 Z 統計量

    回傳：z_stat (float)
    """
    pass


def calculate_stats_t_one():
    """
    單樣本 T 檢定 的數據處理 + 統計量計算

    讀取的全域變數：
        is_raw_data, raw_data, mu,
        x_bar, sigma, n

    寫入的全域變數：
        x_bar, sigma, n

    局域變數：
        data    (list)   ── raw_data[0] 的捷徑
        t_stat  (float)  ── 計算出的 T 統計量

    回傳：t_stat (float)
    """
    global x_bar, sigma, n          # 宣告要修改的全域變數

    if is_raw_data:
        data = raw_data[0]          # 取第一組（單樣本只有一組）
        n     = len(data)
        x_bar = np.mean(data)
        sigma = np.std(data, ddof=1)  # ddof=1 代表樣本標準差
        # mu 由使用者另外輸入，不在這裡算
    else:
        # 使用者直接輸入，全域變數已經有值，不用動
        pass

    t_stat = (x_bar - mu) / (sigma / np.sqrt(n))   # 算出 t 統計量
    return t_stat
    pass


def calculate_stats_t_ind():
    """
    獨立雙樣本 T 檢定 的數據處理 + 統計量計算

    讀取的全域變數：
        is_raw_data, raw_data,
        x_bar, x_bar2, sigma, sigma2, n

    寫入的全域變數：
        x_bar, x_bar2, sigma, sigma2, n

    局域變數：
        data1, data2  (list)   ── raw_data[0], raw_data[1] 的捷徑
        se            (float)  ── 標準誤
        df            (int)    ── 自由度
        t_stat        (float)  ── 計算出的 T 統計量

    回傳：(t_stat, df)
    """
    pass


def calculate_stats_t_pair():
    """
    配對 T 檢定 的數據處理 + 統計量計算

    讀取的全域變數：
        is_raw_data, raw_data,
        x_bar, sigma, n

    寫入的全域變數：
        x_bar, sigma, n

    局域變數：
        data1, data2  (list)   ── raw_data[0], raw_data[1] 的捷徑
        diff          (list)   ── 每對差值 (data1[i] - data2[i])
        t_stat        (float)  ── 計算出的 T 統計量

    回傳：t_stat (float)
    """
    pass


# ============================================================
# 檢定區
# ============================================================

def check_hypothesis(test_stat):
    """
    負責：根據條件查臨界值、與統計量比大小、輸出結論

    參數：
        test_stat   (float 或 tuple) ── 來自計算函式的統計量
                                        T獨立雙樣本時為 (t_stat, df)

    讀取的全域變數：
        is_proportion, is_population_known, is_paired,
        n_samples, tail_type, alpha

    局域變數：
        critical_value  (float)  ── 查表得到的臨界值
        df              (int)    ── 自由度（T檢定用）
        conclusion      (str)    ── 最終輸出結論文字

    回傳：無（直接 print 結論）
    """
    pass