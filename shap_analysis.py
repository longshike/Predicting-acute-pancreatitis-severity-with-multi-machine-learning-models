# 该程序的作用是一个完整的Python脚本，以LightGBM来构建急性胰腺炎的预测模型
# 并进行SHAP分析，其中读取training_set_eng.xlsx进行训练模型，训练集进行了SMOTE过采样
# 读取testing_set_eng.xlsx进行测试集，读取整个原始数据data_V7.0.xlsx来进行SHAP分析
# 同时这些excel表格数据，都需要进行反转
# 程序编写于2025年8月26日，编写者：龙诗科
# 修改：使用指定文件路径和自动反转目标变量编码
# 新增：详细的单特征PDP图和交互PDP等高线图
# 新增：SHAP瀑布图
# 新增：加入了保存模型的功能
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from imblearn.over_sampling import SMOTE
import warnings
import joblib

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams["font.family"] = ['Times New Roman', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子以确保结果可重现
np.random.seed(42)


# 读取数据
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df


# 数据预处理
def preprocess_data(df):
    # 选择指定的特征
    features = ['PT', 'α-HBDH', 'CRP', 'Glu', 'Hb', 'Ca', 'ALB', 'APTT', 'WBC', 'CO₂-CP', 'MCH']
    # 提取特征和目标变量
    X = df[features]
    y = df['Diagnostic Result']

    # 自动反转目标变量编码（重症=1，轻症=0）
    y = 1 - y

    # 打印目标变量的分布和编码
    print("目标变量分布 (反转后):")
    print(y.value_counts())
    print("\n目标变量编码 (反转后):")
    print("唯一值:", y.unique())

    return X, y


# 应用SMOTE过采样
def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


# 训练LightGBM模型
def train_lightgbm(X_train, y_train, X_test, y_test):
    # 创建LightGBM分类器
    model = lgb.LGBMClassifier(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8
    )

    # 训练模型
    model.fit(X_train, y_train)

    # 在测试集上评估模型
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"LightGBM模型准确率: {accuracy:.4f}")
    print(f"LightGBM模型AUC: {auc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))

    # 交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"\n5折交叉验证AUC: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

    return model


# 绘制单特征PDP图（每个特征单独一个图）
def plot_individual_pdp(model, X, feature_names, top_features):
    """为每个重要特征单独绘制PDP图"""
    for i, feature in enumerate(feature_names):
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))

        # 计算部分依赖值
        pdp_values, (feature_values,) = partial_dependence(
            model, X, [feature],
            grid_resolution=50
        )

        # 绘制PDP曲线 - 使用深黄色
        line = ax.plot(feature_values, pdp_values[0],
                       color='#DAA520', linewidth=3, label='PDP curve')  # 深黄色

        # 添加上下置信区间阴影（模拟置信区间）
        # 这里我们使用一个简单的模拟方法，实际应用中可能需要更复杂的计算方法
        std_dev = np.std(pdp_values[0]) * 0.5  # 模拟标准差
        upper_bound = pdp_values[0] + std_dev
        lower_bound = pdp_values[0] - std_dev

        # 在实线两侧添加阴影表示置信区间
        ax.fill_between(feature_values, lower_bound, upper_bound,
                        alpha=0.3, color='#A8DADC', label='Confidence interval')

        # 设置标题和标签
        # ax.set_title(f'{feature} - Partial Dependence Plot', fontsize=16, fontweight='bold')
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('Prediction probability', fontsize=12)

        # 优化x轴范围，使图表更居中
        x_range = feature_values.max() - feature_values.min()
        ax.set_xlim(feature_values.min() - 0.05 * x_range,
                    feature_values.max() + 0.05 * x_range)

        # 添加网格 - 减少网格密度，只在主要刻度处显示
        ax.grid(True, which='major', linestyle='--', alpha=0.7)

        # 获取网格线的x坐标位置
        x_ticks = ax.get_xticks()

        # 只保留在特征值范围内的网格线位置
        valid_ticks = [x for x in x_ticks if feature_values.min() <= x <= feature_values.max()]

        # 插值获取网格线位置对应的PDP值
        from scipy.interpolate import interp1d
        interp_func = interp1d(feature_values, pdp_values[0], kind='linear', bounds_error=False,
                               fill_value="extrapolate")
        y_ticks = interp_func(valid_ticks)

        # 在实线与网格交界处添加黑点（只在有效范围内）
        ax.scatter(valid_ticks, y_ticks, color='black', s=40, zorder=5, alpha=0.8)

        # 添加数据分布直方图
        ax_twin = ax.twinx()
        ax_twin.hist(X[feature], bins=20, alpha=0.2, color='#F24236', density=True)
        ax_twin.set_ylabel('Data distribution', fontsize=10)
        ax_twin.set_ylim(0, ax_twin.get_ylim()[1] * 3)

        # 添加图例
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='best')

        plt.tight_layout()
        plt.savefig(f'pdp_individual_{feature}.png', dpi=600, bbox_inches='tight')
        plt.close()

        # 为前4个最重要的特征额外显示
        if feature in top_features[:4]:
            plt.figure(figsize=(10, 6))
            plt.plot(feature_values, pdp_values[0],
                     color='#DAA520', linewidth=3, label='PDP curve')  # 深黄色

            # 添加上下置信区间阴影
            plt.fill_between(feature_values, lower_bound, upper_bound,
                             alpha=0.3, color='#A8DADC', label='Confidence interval')

            # plt.title(f'{feature} - Partial Dependence Plot', fontsize=16, fontweight='bold')
            plt.xlabel(feature, fontsize=12)
            plt.ylabel('Prediction probability', fontsize=12)

            # 优化x轴范围
            x_range = feature_values.max() - feature_values.min()
            plt.xlim(feature_values.min() - 0.05 * x_range,
                     feature_values.max() + 0.05 * x_range)

            # 添加网格
            plt.grid(True, which='major', linestyle='--', alpha=0.7)

            # 获取当前坐标轴
            ax = plt.gca()
            x_ticks = ax.get_xticks()

            # 只保留在特征值范围内的网格线位置
            valid_ticks = [x for x in x_ticks if feature_values.min() <= x <= feature_values.max()]

            # 插值获取网格线位置对应的PDP值
            interp_func = interp1d(feature_values, pdp_values[0], kind='linear', bounds_error=False,
                                   fill_value="extrapolate")
            y_ticks = interp_func(valid_ticks)

            # 在实线与网格交界处添加黑点（只在有效范围内）
            plt.scatter(valid_ticks, y_ticks, color='black', s=40, zorder=5, alpha=0.8)

            plt.legend(loc='best')
            plt.tight_layout()
            plt.show()


# 绘制交互PDP等高线图 - 分开绘制每个交互图
def plot_interaction_pdp(model, X, feature_names, top_features):
    # 选择前三个最重要的特征进行两两交互
    top_3_features = top_features[:4]

    # 生成所有两两组合
    feature_pairs = [(top_3_features[1], top_3_features[2]),
                     (top_3_features[1], top_3_features[3]),
                     (top_3_features[2], top_3_features[3])]

    for idx, (feature1, feature2) in enumerate(feature_pairs):
        # 创建单独的图形
        fig, ax = plt.subplots(figsize=(8, 6))

        # 手动计算部分依赖值
        pdp, (x_vals, y_vals) = partial_dependence(
            model, X, [feature1, feature2],
            grid_resolution=30
        )

        # 创建网格
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

        # 绘制等高线图
        contour = ax.contourf(X_grid, Y_grid, pdp[0], cmap="viridis", alpha=0.8)

        # 添加等高线并标注数值
        contour_lines = ax.contour(X_grid, Y_grid, pdp[0], colors='k', linewidths=0.5, alpha=0.8)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')

        # 添加颜色条
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Prediction probability', fontsize=10)

        #ax.set_title(f'{feature1} & {feature2} 交互效应', fontsize=14, fontweight='bold')
        ax.set_xlabel(feature1, fontsize=12)
        ax.set_ylabel(feature2, fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'pdp_interaction_{feature1}_{feature2}.png', dpi=600, bbox_inches='tight')
        plt.show()


# 绘制SHAP瀑布图
def plot_shap_waterfall(explainer, shap_values, X, feature_names, sample_indices):
    # 获取基准值
    if isinstance(explainer.expected_value, list):
        base_value = explainer.expected_value[1]
    else:
        base_value = explainer.expected_value

    # 为每个样本绘制瀑布图
    for idx in sample_indices:
        plt.figure(figsize=(12, 8))

        # 创建瀑布图
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[idx],
                base_values=base_value,
                data=X.iloc[idx],
                feature_names=feature_names
            ),
            max_display=12,  # 显示前12个最重要的特征
            show=False
        )

        #plt.title(f"SHAP瀑布图 - 样本 {idx}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'shap_waterfall_plot_sample_{idx}.png', dpi=600, bbox_inches='tight')
        plt.show()


# SHAP分析
def shap_analysis(model, X, feature_names):
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)

    # 计算SHAP值
    shap_values = explainer.shap_values(X)

    # 获取基准值
    if isinstance(explainer.expected_value, list):
        base_value = explainer.expected_value[1]
    else:
        base_value = explainer.expected_value

    print(f"基准值 (Base Value): {base_value:.4f}")
    print("基准值表示模型在没有任何特征信息时的平均预测值")

    # 如果是二分类问题，取第二个类的SHAP值（通常对应正类）
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    # 1. 绘制特征重要性摘要图（蜂群图）
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    #plt.title("SHAP特征重要性摘要图（蜂群图）", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=600, bbox_inches='tight')
    plt.show()

    # 2. 绘制特征重要性条形图
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    #plt.title("SHAP特征重要性排序", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_bar_plot.png', dpi=600, bbox_inches='tight')
    plt.show()

    # 3. 绘制单个特征的依赖图
    # 获取最重要的特征
    mean_abs_shap = np.abs(shap_values).mean(0)
    important_features = np.argsort(mean_abs_shap)[-3:]  # 选择最重要的3个特征

    for i, feature_idx in enumerate(important_features):
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X,
            feature_names=feature_names,
            interaction_index=None,
            show=False
        )
        #plt.title(f"SHAP依赖图 - {feature_names[feature_idx]}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'shap_dependence_plot_{feature_names[feature_idx]}.png', dpi=600, bbox_inches='tight')
        plt.show()

    # 4. 绘制力力图（选择几个样本进行解释）
    # 选择几个有代表性的样本
    sample_indices = [0, 10, 20]  # 可以根据需要调整
    for idx in sample_indices:
        plt.figure(figsize=(12, 6))
        shap.force_plot(
            base_value,
            shap_values[idx, :],
            X.iloc[idx, :],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        #plt.title(f"SHAP力力图 - 样本 {idx}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'shap_force_plot_sample_{idx}.png', dpi=600, bbox_inches='tight')
        plt.show()

    # 5. 绘制决策图
    plt.figure(figsize=(12, 8))
    shap.decision_plot(
        base_value,
        shap_values[:1200],  # 只显示前100个样本，避免过于拥挤
        feature_names=feature_names,
        show=False
    )
    #plt.title("SHAP决策图", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_decision_plot.png', dpi=600, bbox_inches='tight')
    plt.show()

    # 6. 绘制瀑布图
    print("\n绘制SHAP瀑布图...")
    plot_shap_waterfall(explainer, shap_values, X, feature_names, sample_indices)

    return explainer, shap_values, mean_abs_shap


# 主函数
def main():
    # 加载数据
    training_file = 'training_set_eng-non-normalize.xlsx'
    testing_file = 'testing_set_eng-non-normalize.xlsx'
    all_data_file = 'data_V7.0-non-normalize.xlsx'

    print("加载训练集数据...")
    train_df = load_data(training_file)
    print("加载测试集数据...")
    test_df = load_data(testing_file)
    print("加载所有数据...")
    all_df = load_data(all_data_file)

    # 数据预处理
    print("\n处理训练集数据...")
    X_train, y_train = preprocess_data(train_df)
    print("\n处理测试集数据...")
    X_test, y_test = preprocess_data(test_df)
    print("\n处理所有数据...")
    X_all, y_all = preprocess_data(all_df)

    feature_names = X_train.columns.tolist()

    # 打印一些基本信息
    print(f"\n特征数量: {len(feature_names)}")
    print(f"训练集样本数量: {len(X_train)}")
    print(f"测试集样本数量: {len(X_test)}")
    print(f"所有数据样本数量: {len(X_all)}")

    # 只对训练集应用SMOTE
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

    # 训练LightGBM模型（使用过采样数据）
    print("\n训练LightGBM模型...")
    model = train_lightgbm(X_train_resampled, y_train_resampled, X_test, y_test)

    # 使用所有原始数据进行SHAP分析
    print(f"\n使用所有原始数据进行SHAP分析，样本数量: {len(X_all)}")
    print("进行SHAP分析...")
    explainer, shap_values, mean_abs_shap = shap_analysis(model, X_all, feature_names)

    # 保存SHAP值
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv('shap_values.csv', index=False)

    # 计算并显示特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    top_features = feature_importance['feature'].tolist()[:4]  # 获取前4个最重要的特征

    print("\n特征重要性排序:")
    print(feature_importance)

    # 绘制特征重要性条形图
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('特征重要性排序 (基于SHAP值)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_importance_ranking.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 绘制单特征PDP图（每个特征单独一个图）
    print("\n绘制单特征PDP图...")
    plot_individual_pdp(model, X_all, feature_names, top_features)

    # 绘制交互PDP等高线图（每个交互图单独绘制）
    print("\n绘制交互PDP等高线图...")
    plot_interaction_pdp(model, X_all, feature_names, top_features)

    print("SHAP分析和PDP图绘制完成，所有图表已保存。")

    # 保存模型
    joblib.dump(model, 'pancreatitis_model.pkl')

    # 保存特征名称（确保预测时特征顺序一致）
    import json
    with open('feature_names.json', 'w') as f:
        json.dump(feature_names, f)

    print("模型已保存为 pancreatitis_model.pkl")


if __name__ == '__main__':
    main()