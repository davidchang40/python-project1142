import numpy as np
from scipy.stats import t

# ==========================================
# 🧮 第一區：數學計算區 (只吃數字，吐出分數與自由度)
# ==========================================

def calc_1samp_mean_t_score(x_bar, mu0, s, n):
    """計算單樣本 t 檢定"""
    t_score = (x_bar - mu0) / (s / np.sqrt(n))
    df = n - 1
    return t_score, df

def calc_2samp_mean_t_score_welch(m1, s1, n1, m2, s2, n2):
    """計算獨立雙樣本 t 檢定 (Welch's 變異數不相等)"""
    # 計算 t 分數
    se1, se2 = s1**2 / n1, s2**2 / n2
    t_score = (m1 - m2) / np.sqrt(se1 + se2)
    
    # 計算 Welch-Satterthwaite 自由度
    df_numerator = (se1 + se2)**2
    df_denominator = (se1**2 / (n1 - 1)) + (se2**2 / (n2 - 1))
    df = df_numerator / df_denominator
    
    return t_score, df


# ==========================================
# 📖 第二區：P 值轉換區 (只呼叫 Scipy)
# ==========================================

def get_pvalue_from_t(t_score, df, tail):
    """將 t 分數轉換為 P 值"""
    if tail == 'greater':
        return t.sf(t_score, df)          # 右尾面積
    elif tail == 'less':
        return t.cdf(t_score, df)         # 左尾面積
    elif tail == 'two-sided':
        return t.sf(abs(t_score), df) * 2 # 雙尾面積 (取絕對值算單邊再乘2)


# ==========================================
# 🚦 第三區：交通警察主控台 (讀取參數，分發任務)
# ==========================================

def run_hypothesis_test(test_params):
    print("\n" + "="*35)
    print("📊 統計計算機：假設檢定分析報告")
    print("="*35)
    
    # 提取共用設定
    alpha = test_params['alpha']
    tail = test_params['tail']
    
    if test_params['type'] == 'mean':
        
        # --- 路徑 A：單樣本 t 檢定 ---
        if not test_params['is_two_sample']:
            mu0 = test_params['h0_value']
            m1 = test_params['group1']['mean']
            s1 = test_params['group1']['std']
            n1 = test_params['group1']['n']
            
            # 分派計算
            t_score, df = calc_1samp_mean_t_score(m1, mu0, s1, n1)
            test_name = "單樣本平均數 t 檢定"

        # --- 路徑 B：雙樣本 t 檢定 (Welch's) ---
        else:
            m1 = test_params['group1']['mean']
            s1 = test_params['group1']['std']
            n1 = test_params['group1']['n']
            m2 = test_params['group2']['mean']
            s2 = test_params['group2']['std']
            n2 = test_params['group2']['n']
            
            # 分派計算 (雙樣本 H0 通常假設兩者差異為 0，所以 mu0 沒用到)
            t_score, df = calc_2samp_mean_t_score_welch(m1, s1, n1, m2, s2, n2)
            test_name = "獨立雙樣本平均數 t 檢定 (Welch's)"

        # 統一計算 P 值
        p_val = get_pvalue_from_t(t_score, df, tail)
        
        # --- 輸出漂亮報告 ---
        tail_ch = {'two-sided':'雙尾', 'greater':'右尾', 'less':'左尾'}
        
        print(f"📌 檢定類型: {test_name}")
        print(f"📌 檢定方向: {tail_ch[tail]}")
        print(f"📌 顯著水準 (alpha): {alpha}")
        print("-" * 35)
        print(f"▶ t 統計量 (t-score): {t_score:.4f}")
        print(f"▶ 自由度 (df):        {df:.4f}")
        print(f"▶ P 值 (p-value):     {p_val:.4e}")  # 用科學記號顯示極小的P值
        print("-" * 35)
        
        # 決策判斷
        if p_val < alpha:
            print(f"✅ 結論：P 值 < {alpha}，結果達到統計顯著。")
            print("👉 統計決策：【拒絕】虛無假設 (H0)。")
        else:
            print(f"❌ 結論：P 值 >= {alpha}，結果未達統計顯著。")
            print("👉 統計決策：【無法拒絕】虛無假設 (H0)。")


# ==========================================
# 🧪 測試區：模擬來自 Main 的輸入資料
# ==========================================

if __name__ == "__main__":
    
    # 模擬情境 1：單樣本 t 檢定 (例如：測試某機器生產的零件平均長度是否大於 10cm)
    mock_1samp_params = {
        'type': 'mean',
        'is_two_sample': False,
        'group1': {'mean': 10.5, 'std': 1.2, 'n': 30},
        'h0_value': 10.0,
        'alpha': 0.05,
        'tail': 'greater'
    }
    print("\n\n>>> 執行情境 1：單樣本測試")
    run_hypothesis_test(mock_1samp_params)

    # 模擬情境 2：雙樣本 t 檢定 (例如：測試 A 班與 B 班的平均成績是否有顯著不同)
    mock_2samp_params = {
        'type': 'mean',
        'is_two_sample': True,
        'group1': {'mean': 85.0, 'std': 5.5, 'n': 40},
        'group2': {'mean': 80.0, 'std': 6.2, 'n': 35},
        'h0_value': 0.0, # 檢定差值是否為0
        'alpha': 0.05,
        'tail': 'two-sided'
    }
    print("\n\n>>> 執行情境 2：雙樣本測試")
    run_hypothesis_test(mock_2samp_params)