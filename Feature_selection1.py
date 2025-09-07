#该程序的作用是一个完整的Python脚本，用于对胰腺炎训练集数据进行LASSO回归分析的lambda最优参数求解
#并画出LASSO交叉验证AUC曲线图和LASSO系数变化趋势图
# 运行得出最优lambda值为0.000838
#程序编写于2025年8月27日，编写者：龙诗科
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings

# --------------------------
# 基础配置
# --------------------------
# 设置中文字体为SimHei，英文和数字为Times New Roman
plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
rcParams['axes.unicode_minus'] = False  # 负号显示
warnings.filterwarnings('ignore')  # 忽略警告
plt.rcParams['figure.dpi'] = 100  # 分辨率


# --------------------------
# 1. 数据加载与预处理
# --------------------------
def load_data(file_path):
    """加载数据（标签为0/1，1表示重症）"""
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    print(f"数据总形状: {df.shape}")

    # 提取特征和标签
    feature_columns = df.columns[3:-1]
    # 移除缺失值填充代码（因为Excel中没有缺失值）
    X = df[feature_columns].copy()
    y = 1 - df[df.columns[-1]].copy()  # 标签反转：确保1=重症

    # 数据信息
    print(f"特征数量: {len(feature_columns)}, 样本数量: {len(X)}")
    print(f"标签分布:\n{y.value_counts()}")
    print(f"重症(1)比例: {y.mean():.4f}")
    return X, y, feature_columns


# --------------------------
# 2. Lasso回归核心分析（计算AUC）
# --------------------------
def lasso_core_analysis(X, y):
    """
    Lasso回归，10折交叉验证计算AUC
    返回：最优λ、1SEλ、系数路径、AUC值
    """
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # 交叉验证确定最优lambda
    lasso_cv = LassoCV(
        cv=kf,
        random_state=42,
        max_iter=10000,
        n_alphas=100
    )
    lasso_cv.fit(X, y)

    # 获取所有候选alpha值
    alphas = lasso_cv.alphas_
    # 按alpha升序排列
    sorted_idx = np.argsort(alphas)
    alphas = alphas[sorted_idx]

    # 对每个alpha计算10折交叉验证的AUC
    auc_scores = []
    for alpha in alphas:
        fold_aucs = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # 训练Lasso模型
            lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
            lasso.fit(X_train, y_train)

            # 预测并计算AUC（将回归结果视为概率）
            y_pred = lasso.predict(X_test)
            auc = roc_auc_score(y_test, y_pred)
            fold_aucs.append(auc)

        auc_scores.append(np.mean(fold_aucs))

    auc_scores = np.array(auc_scores)
    # 计算AUC的标准差
    auc_std = np.array([
        np.std([
            roc_auc_score(y.iloc[test_idx],
                          Lasso(alpha=alpha, max_iter=10000).fit(X.iloc[train_idx], y.iloc[train_idx]).predict(
                              X.iloc[test_idx]))
            for train_idx, test_idx in kf.split(X)
        ])
        for alpha in alphas
    ])

    # 最优lambda（最大AUC对应的lambda）
    idx_max_auc = np.argmax(auc_scores)
    alpha_opt = alphas[idx_max_auc]

    # 1SE规则计算
    threshold_auc = auc_scores[idx_max_auc] - auc_std[idx_max_auc]
    alpha_1se = alphas[auc_scores >= threshold_auc][0]

    # 计算系数路径
    coef_path = []
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
        lasso.fit(X, y)
        coef_path.append(lasso.coef_)
    coef_path = np.array(coef_path).T

    # 输出关键参数
    print(f"\n最优λ（最大AUC）: {alpha_opt:.6f}")
    print(f"1SE规则λ: {alpha_1se:.6f}")
    print(f"最大AUC值: {auc_scores[idx_max_auc]:.4f}")
    return lasso_cv, alpha_opt, alpha_1se, alphas, auc_scores, auc_std, coef_path


# --------------------------
# 3. 图2(A)：LASSO交叉验证AUC曲线
# --------------------------
def plot_lasso_cv_auc(alphas, auc_scores, auc_std, alpha_opt, alpha_1se,
                      save_path="图2A_LASSO交叉验证AUC曲线.png"):
    """绘制AUC随λ变化的曲线"""
    log_alphas = np.log10(alphas)

    fig, ax = plt.subplots(figsize=(10, 6))
    # 绘制AUC曲线和误差带
    ax.plot(log_alphas, auc_scores, color='#1f77b4', linewidth=2.5, label='Average AUC')
    ax.fill_between(
        log_alphas,
        auc_scores - auc_std,
        auc_scores + auc_std,
        alpha=0.2,
        color='#1f77b4',
        label='AUC ± 1 standard deviation'
    )

    # 标记最优λ和1SEλ
    ax.axvline(
        x=np.log10(alpha_opt),
        color='#ff7f0e',
        linestyle='--',
        linewidth=2,
        label=f'Optimal λ: {alpha_opt:.6f}'
    )
    ax.axvline(
        x=np.log10(alpha_1se),
        color='#d62728',
        linestyle='--',
        linewidth=2,
        label=f'1SE λ: {alpha_1se:.6f}'
    )

    # 图表标签
    ax.set_xlabel('log10(λ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC Value', fontsize=12, fontweight='bold')
    #ax.set_title('LASSO回归10折交叉验证曲线（最优λ选择）', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"\n图2(A)已保存至: {save_path}")


# --------------------------
# 4. 图2(B)：特征系数随λ变化趋势
# --------------------------
def plot_lasso_coef_trend(alphas, coef_path, feature_columns, alpha_1se,
                          save_path="图2B_LASSO系数变化趋势.png"):
    """绘制特征系数随λ变化的路径"""
    log_alphas = np.log10(alphas)

    fig, ax = plt.subplots(figsize=(12, 7))
    # 绘制每个特征的系数路径
    for i in range(coef_path.shape[0]):
        ax.plot(log_alphas, coef_path[i, :], linewidth=1.2, alpha=0.8, label=feature_columns[i])

    # 标记1SEλ位置
    ax.axvline(
        x=np.log10(alpha_1se),
        color='#d62728',
        linestyle='--',
        linewidth=2.5,
        label=f'1SE λ: {alpha_1se:.6f}'
    )

    ax.set_xlabel('log10(λ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('LASSO coefficient values', fontsize=12, fontweight='bold')
    #ax.set_title('各特征LASSO系数随λ变化趋势', fontsize=14, fontweight='bold', pad=20)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)

    # 图例处理
    #if len(feature_columns) > 15:
    #    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    #else:
     #   ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图2(B)已保存至: {save_path}")


# --------------------------
# 主函数
# --------------------------
def main():
    file_path = "training_set_eng.xlsx"  # 数据文件路径
    X, y, feature_columns = load_data(file_path)

    # LASSO核心分析
    print("\n" + "=" * 50)
    print("2. 开始LASSO回归分析...")
    lasso_cv_model, alpha_opt, alpha_1se, alphas, auc_scores, auc_std, coef_path = lasso_core_analysis(X, y)

    # 绘制图2(A)（AUC曲线）
    print("\n" + "=" * 50)
    print("3. 绘制图2(A)：AUC曲线...")
    plot_lasso_cv_auc(alphas, auc_scores, auc_std, alpha_opt, alpha_1se)

    # 绘制图2(B)（系数趋势）
    print("\n" + "=" * 50)
    print("4. 绘制图2(B)：系数变化趋势...")
    plot_lasso_coef_trend(alphas, coef_path, feature_columns, alpha_1se)

    # 结果总结
    print("\n" + "=" * 50)
    print("核心结果总结:")
    print(f"- 最优λ（最大AUC）: {alpha_opt:.6f}")
    print(f"- 1SE规则λ: {alpha_1se:.6f}")
    print(f"- 图2(A)、图2(B)已保存至当前目录")
    print("=" * 50)


if __name__ == "__main__":
    main()
