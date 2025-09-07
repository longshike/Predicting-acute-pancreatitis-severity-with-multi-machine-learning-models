#该程序的作用是一个完整的Python脚本，将最优的Lasso值代入LASSO回归分析
#通过LASSO得到的特征量为18个，然后通过随机森林选择18个特征变量
# 然后通过VIF分析两者的交集特征，总共得出有11个特征量
# 并画出Lasso的各个特征系数分布，画出随机森林的各个特征的重要程度排序
#程序编写于2025年8月30日，编写者：龙诗科
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import warnings
import os

# 设置中文字体
plt.rcParams["font.family"] = ['Times New Roman', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
warnings.filterwarnings('ignore')

# 创建输出目录
if not os.path.exists('output'):
    os.makedirs('output')


# 1. 数据加载与预处理
def load_data(file_path):
    """加载Excel数据文件，确保标签反转正确：1代表重症，0代表轻症"""
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    print(f"数据形状: {df.shape}")

    # 提取特征和标签（明确反转标签：表格中0代表重症，1代表轻症）
    feature_columns = df.columns[3:-1]  # 从第4列到倒数第2列
    X = df[feature_columns]
    # 明确反转：0→1（重症），1→0（轻症）
    y = 1 - df[df.columns[-1]]

    print(f"特征数量: {len(feature_columns)}")
    print(f"标签分布（反转后）:\n{y.value_counts()}")
    print(f"重症(1)比例: {y.mean():.4f}")

    return X, y, feature_columns


# 2. LASSO特征选择（使用指定的最优lambda值）
def lasso_feature_selection(X, y, optimal_lambda=0.000838):
    """使用指定的最优lambda值通过LASSO回归进行特征选择"""
    # 使用指定的最优lambda值
    lasso = Lasso(alpha=optimal_lambda, max_iter=10000, random_state=42)
    lasso.fit(X, y)

    print(f"LASSO使用的lambda值: {optimal_lambda}")

    # 提取特征系数
    lasso_coef = pd.DataFrame({
        'feature': X.columns,
        'coefficient': lasso.coef_
    }).sort_values('coefficient', key=abs, ascending=False)

    # 选择非零系数对应的特征
    lasso_selector = SelectFromModel(lasso, prefit=True, threshold=1e-5)
    lasso_support = lasso_selector.get_support()
    lasso_features = X.columns[lasso_support].tolist()

    print(f"LASSO选择的特征数量: {len(lasso_features)}")
    print(f"LASSO选择的特征: {lasso_features}")

    # 仅保留选中特征的系数
    selected_lasso_coef = lasso_coef[lasso_coef['feature'].isin(lasso_features)]

    return lasso_features, lasso, selected_lasso_coef


# 3. 随机森林特征选择（基于基尼系数，指定选择10个变量）
def random_forest_feature_selection(X, y, top_n=10):
    """使用随机森林基于基尼系数进行特征选择，指定选择top_n个特征"""
    # 确保使用基尼系数
    rf = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',  # 基于基尼系数
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)

    # 获取特征重要性（基尼系数均值下降值）
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_  # 基尼系数均值下降值
    }).sort_values('importance', ascending=False)

    # 选择重要性最高的top_n个特征
    rf_features = feature_importance.head(top_n)['feature'].tolist()

    print(f"随机森林选择的特征数量: {len(rf_features)}")
    print(f"随机森林选择的特征: {rf_features}")

    # 仅保留选中特征的重要性
    selected_rf_importance = feature_importance[feature_importance['feature'].isin(rf_features)]

    return rf_features, rf, selected_rf_importance


# 4. 绘制图2(c)：LASSO模型选择的特征系数分布
def plot_lasso_coefficients(lasso_coef, save_path='output/图2c_LASSO特征系数分布.png'):
    """绘制LASSO模型中用于预测急性胰腺炎严重程度的特征系数分布"""
    plt.figure(figsize=(10, 8))

    # 按系数绝对值排序
    lasso_coef = lasso_coef.sort_values('coefficient', key=abs, ascending=True)

    # 绘制水平条形图
    colors = ['red' if x < 0 else 'blue' for x in lasso_coef['coefficient']]
    plt.barh(lasso_coef['feature'], lasso_coef['coefficient'], color=colors, alpha=0.7)

    # 添加零值参考线
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

    plt.xlabel('LASSO coefficient values', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    #plt.title('LASSO模型选择的特征系数分布', fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"LASSO特征系数分布图已保存至: {save_path}")


# 5. 绘制图2(d)：随机森林特征重要性分布
def plot_rf_importance(rf_importance, save_path='output/图2d_随机森林特征重要性.png'):
    """绘制随机森林构建过程中各变量对决策树的相对影响"""
    plt.figure(figsize=(10, 8))

    # 按重要性排序
    rf_importance = rf_importance.sort_values('importance', ascending=True)

    # 绘制水平条形图
    plt.barh(rf_importance['feature'], rf_importance['importance'], color='#2ca02c', alpha=0.7)

    plt.xlabel('Feature Importance (Mean Decrease in Gini Coefficient)', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    #plt.title('随机森林模型特征重要性分布（Top 10）', fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"随机森林特征重要性图已保存至: {save_path}")


# 6. 特征交集分析（文本分析，不绘图）
def analyze_feature_intersection(lasso_features, rf_features):
    """分析LASSO和随机森林选择特征的交集"""
    # 计算交集特征
    common_features = list(set(lasso_features) & set(rf_features))
    print(f"\n两种方法共同选择的特征数量: {len(common_features)}")
    print(f"共同特征: {common_features}")

    return common_features


# 7. 生成表格2：特征选择结果比较
def generate_table2(lasso_features, rf_features, lasso_coef, rf_importance,
                    save_path='output/表格2_特征选择结果比较.csv'):
    """生成比较两种方法选择特征的表格"""
    # 获取所有被至少一种方法选择的特征
    all_selected = list(set(lasso_features) | set(rf_features))

    # 创建表格数据
    table_data = []
    for feature in all_selected:
        in_lasso = 1 if feature in lasso_features else 0
        in_rf = 1 if feature in rf_features else 0

        # 获取LASSO系数
        lasso_coef_val = lasso_coef[lasso_coef['feature'] == feature]['coefficient'].values[0] if in_lasso else None

        # 获取随机森林重要性
        rf_imp_val = rf_importance[rf_importance['feature'] == feature]['importance'].values[0] if in_rf else None

        table_data.append({
            '特征名称': feature,
            'LASSO选择': '是' if in_lasso else '否',
            'LASSO系数': round(lasso_coef_val, 4) if in_lasso else '-',
            '随机森林选择': '是' if in_rf else '否',
            '随机森林重要性': round(rf_imp_val, 4) if in_rf else '-',
            '两种方法共同选择': '是' if (in_lasso and in_rf) else '否'
        })

    # 转换为DataFrame并排序
    table_df = pd.DataFrame(table_data)
    table_df = table_df.sort_values(['两种方法共同选择', 'LASSO选择', '随机森林选择'],
                                    ascending=[False, False, False])

    # 保存为CSV和Excel
    table_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    table_df.to_excel(save_path.replace('.csv', '.xlsx'), index=False)
    print(f"特征选择结果比较表格已保存至: {save_path}")

    return table_df


# 8. VIF分析去除共线性（可选步骤）
def vif_analysis(X, features, threshold=2):
    """计算VIF并去除高共线性特征"""
    if not features:
        return [], pd.DataFrame()

    # 只分析选定的特征
    X_selected = X[features].copy()

    # 添加常数项用于VIF计算
    X_with_const = add_constant(X_selected)

    # 计算VIF
    vif_data = pd.DataFrame()
    vif_data["特征"] = X_with_const.columns
    vif_data["VIF值"] = [variance_inflation_factor(X_with_const.values, i)
                        for i in range(X_with_const.shape[1])]

    # 移除常数项
    vif_data = vif_data[vif_data['特征'] != 'const']

    print("VIF分析结果:")
    print(vif_data.sort_values('VIF值', ascending=False))

    # 筛选VIF低于阈值的特征
    selected_features = vif_data[vif_data['VIF值'] <= threshold]['特征'].tolist()

    print(f"经过VIF筛选后的特征数量: {len(selected_features)}")
    print(f"筛选后的特征: {selected_features}")

    return selected_features, vif_data


# 9. 主函数
def main():
    # 加载数据
    file_path = 'training_set_eng.xlsx'
    X, y, feature_columns = load_data(file_path)

    # LASSO特征选择（使用指定的最优lambda值）
    print("\n=== LASSO特征选择 ===")
    optimal_lambda = 0.000838  # 最优lambda值
    lasso_features, lasso_model, lasso_coef = lasso_feature_selection(X, y, optimal_lambda)

    # 随机森林特征选择（指定选择10个变量）
    print("\n=== 随机森林特征选择 ===")
    rf_features, rf_model, rf_importance = random_forest_feature_selection(X, y, top_n=18)

    # 特征交集分析
    print("\n=== 特征交集分析 ===")
    common_features = analyze_feature_intersection(lasso_features, rf_features)

    # 绘制图2(c)：LASSO特征系数分布
    print("\n=== 绘制LASSO特征系数分布图 ===")
    plot_lasso_coefficients(lasso_coef)

    # 绘制图2(d)：随机森林特征重要性
    print("\n=== 绘制随机森林特征重要性图 ===")
    plot_rf_importance(rf_importance)

    # 生成表格2：特征选择结果比较
    print("\n=== 生成特征选择结果表格 ===")
    table2 = generate_table2(lasso_features, rf_features, lasso_coef, rf_importance)

    # 可选：VIF分析去除共线性
    print("\n=== VIF分析（去除共线性） ===")
    # 使用共同特征进行VIF分析
    if common_features:
        final_features, vif_results = vif_analysis(X, common_features)

        # 可视化VIF结果
        if not vif_results.empty:
            plt.figure(figsize=(10, 6))
            vif_results = vif_results.sort_values('VIF值', ascending=True)
            plt.barh(vif_results['特征'], vif_results['VIF值'])
            plt.axvline(x=10, color='r', linestyle='--', label='VIF=10')
            plt.xlabel('VIF值')
            plt.title('特征VIF值')
            plt.legend()
            plt.tight_layout()
            plt.savefig('output/VIF分析结果.png', dpi=300)
            plt.close()
    else:
        # 如果没有共同特征，使用两种方法选择的所有特征
        all_selected = list(set(lasso_features) | set(rf_features))
        final_features, vif_results = vif_analysis(X, all_selected)

    print("\n=== 分析完成 ===")
    print(f"图2(c)、图2(d)和表格2已保存至output文件夹")

    return final_features, lasso_model, rf_model


if __name__ == "__main__":
    final_features, lasso_model, rf_model = main()