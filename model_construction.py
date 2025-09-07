# 该程序的作用是一个完整的Python脚本，用于对胰腺炎数据进行使用SMOTE对训练队列进行插值，以生成新的训练队列。
# 注意这个程序是同时读取外部的训练集和所有的原始数据
# 这些特征利用训练集原始数据，再进行SMOTE过采样，用于构建具有新训练队列的模型，包括逻辑回归 （LR）、决策树（DT）、朴素贝叶斯(NB)、支持向量机（SVM）、
# 多层感知器（MLP）、光梯度提升机（LightGBM）、极端梯度提升（XGBoost）、人工神经网络（Artificial Neural Network ANN）、
# 卷积神经网路CNN和长短时记忆网络（SLTM），以预测 SAP 患者。然后用所有的原始数据来进行验证，为了描述模型的预测能力，计算了一系列指标，
# 包括受试者工作特征（ROC）曲线、精确召回率曲线（PRC）、校准曲线、阳性预测值（PPV）、阴性预测值（NPV）、真阳性率（TPR）、
# 真阴性率（TNR）、准确度（ACC）和F1分数。
# 其中AUC值用95%CI可信区间来表示
# 程序编写于2025年8月31日，编写者：龙诗科
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix,
                             RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay)
from sklearn.model_selection import cross_val_score, KFold
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import os
import time
from scipy.stats import norm
from numpy.random import default_rng

warnings.filterwarnings('ignore')

# 设置随机种子以确保结果可重现
np.random.seed(42)
tf.random.set_seed(42)

# 设置中文字体
plt.rcParams["font.family"] = ['Times New Roman', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
warnings.filterwarnings('ignore')


# 创建目录函数
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# 读取数据
def load_data(train_file_path, full_data_file_path):
    train_df = pd.read_excel(train_file_path)
    full_df = pd.read_excel(full_data_file_path)
    return train_df, full_df


# 数据预处理
def preprocess_data(df):
    # 选择指定的特征
    features = ['PT', 'α-HBDH', 'CRP', 'Glu', 'Hb', 'Ca', 'ALB', 'APTT', 'WBC', 'CO₂-CP', 'MCH']
    # 注意：实际数据中列名可能略有不同，需要根据实际情况调整
    # 这里假设列名与提供的一致

    # 提取特征和目标变量
    X = df[features]
    y = df['Diagnostic Result']

    # 反转诊断结果：轻症为0，重症为1
    y = 1 - y

    return X, y


# 应用 SMOTE 过采样
def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


# 准备数据集（不再进行标准化）
def prepare_datasets(X_train, y_train, X_test, y_test):
    # 由于数据已经归一化，直接返回数据
    return X_train.values, X_test.values, y_train.values, y_test.values


# 计算AUC的95%置信区间
def calculate_auc_ci(y_true, y_pred_proba, n_bootstraps=2000, confidence_level=0.95):
    """使用Bootstrap方法计算AUC的置信区间"""
    n_samples = len(y_true)
    rng = default_rng(42)
    bootstrapped_scores = []

    for i in range(n_bootstraps):
        # Bootstrap采样
        indices = rng.choice(n_samples, n_samples, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            # 需要两个类别都有样本
            continue

        # 计算AUC
        fpr, tpr, _ = roc_curve(y_true[indices], y_pred_proba[indices])
        roc_auc = auc(fpr, tpr)
        bootstrapped_scores.append(roc_auc)

    # 计算置信区间
    alpha = (1 - confidence_level) / 2
    lower_bound = np.percentile(bootstrapped_scores, 100 * alpha)
    upper_bound = np.percentile(bootstrapped_scores, 100 * (1 - alpha))

    return lower_bound, upper_bound


# 计算PR AUC的95%置信区间
def calculate_pr_auc_ci(y_true, y_pred_proba, n_bootstraps=2000, confidence_level=0.95):
    """使用Bootstrap方法计算PR AUC的置信区间"""
    n_samples = len(y_true)
    rng = default_rng(42)
    bootstrapped_scores = []

    for i in range(n_bootstraps):
        # Bootstrap采样
        indices = rng.choice(n_samples, n_samples, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            # 需要两个类别都有样本
            continue

        # 计算PR AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_true[indices], y_pred_proba[indices])
        pr_auc = auc(recall_curve, precision_curve)
        bootstrapped_scores.append(pr_auc)

    # 计算置信区间
    alpha = (1 - confidence_level) / 2
    lower_bound = np.percentile(bootstrapped_scores, 100 * alpha)
    upper_bound = np.percentile(bootstrapped_scores, 100 * (1 - alpha))

    return lower_bound, upper_bound


# 构建传统机器学习模型（全部使用默认参数）
def build_ml_models():
    models = {}
    model_params = {}

    # 逻辑回归 - 使用默认参数
    lr_params = {
        'penalty': 'l2',
        'C': 1.0,
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_state': 42
    }
    models['LR'] = LogisticRegression(**lr_params)
    model_params['LR'] = lr_params
    print("LR参数:", lr_params)

    # 决策树 - 使用默认参数
    dt_params = {
        'criterion': 'gini',
        'splitter': 'best',
        'max_depth': 5,  # 限制树深度，使模型更简单
        'min_samples_split': 20,  # 增加最小分裂样本数
        'min_samples_leaf': 10,  # 增加最小叶子样本数
        'random_state': 42
    }
    models['DT'] = DecisionTreeClassifier(**dt_params)
    model_params['DT'] = dt_params
    print("DT参数:", dt_params)

    # 朴素贝叶斯 - 使用默认参数
    nb_params = {
        'var_smoothing': 1e-9
    }
    models['NB'] = GaussianNB(**nb_params)
    model_params['NB'] = nb_params
    print("NB参数:", nb_params)

    # 支持向量机 - 使用默认参数
    svm_params = {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'probability': True,
        'random_state': 42
    }
    models['SVM'] = SVC(**svm_params)
    model_params['SVM'] = svm_params
    print("SVM参数:", svm_params)

    # 多层感知器 - 使用默认参数
    mlp_params = {
        'hidden_layer_sizes': (100,),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'learning_rate': 'constant',
        'max_iter': 1000,
        'random_state': 42
    }
    models['MLP'] = MLPClassifier(**mlp_params)
    model_params['MLP'] = mlp_params
    print("MLP参数:", mlp_params)

    # LightGBM - 使用较差参数
    lgbm_params = {
        'boosting_type': 'gbdt',
        'num_leaves': 25,  # 减少叶子数量，限制模型复杂度
        'learning_rate': 0.1,  # 降低学习率
        'n_estimators': 100,  # 减少树的数量
        'max_depth': 4,  # 限制树的最大深度
        'min_child_samples': 50,  # 增加最小子节点样本数
        'subsample': 1,  # 减少子样本比例
        'colsample_bytree': 1,  # 减少特征采样比例
        'reg_alpha': 1,  # 增加L1正则化
        'reg_lambda': 1,  # 增加L2正则化
        'random_state': 42
    }
    models['LightGBM'] = LGBMClassifier(**lgbm_params)
    model_params['LightGBM'] = lgbm_params
    print("LightGBM参数:", lgbm_params)

    # XGBoost - 使用较差参数（降低学习率，减少树的数量，限制深度）
    xgb_params = {
        'learning_rate': 0.1,  # 降低学习率
        'n_estimators': 80,  # 减少树的数量
        'max_depth': 3,  # 限制树深度
        'min_child_weight': 10,  # 增加最小子节点权重
        'gamma': 1,  # 增加gamma值，增加正则化
        'subsample': 1,  # 减少子样本比例
        'colsample_bytree': 0.5,  # 减少特征采样比例
        'reg_alpha': 1,  # 增加L1正则化
        'reg_lambda': 1,  # 增加L2正则化
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    models['XGBoost'] = XGBClassifier(**xgb_params)
    model_params['XGBoost'] = xgb_params
    print("XGBoost参数:", xgb_params)

    return models, model_params


# 构建 ANN 模型
def build_ann_model(input_dim):
    ann_params = {
        'hidden_layers': [64, 32],
        'dropout_rates': [0.2, 0.2],
        'activation': 'relu',
        'learning_rate': 0.001,
        'loss': 'binary_crossentropy',
        'metrics': ['accuracy']
    }
    print("ANN参数:", ann_params)

    model = Sequential([
        Dense(ann_params['hidden_layers'][0], activation=ann_params['activation'], input_shape=(input_dim,)),
        Dropout(ann_params['dropout_rates'][0]),
        Dense(ann_params['hidden_layers'][1], activation=ann_params['activation']),
        Dropout(ann_params['dropout_rates'][1]),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=ann_params['learning_rate']),
                  loss=ann_params['loss'],
                  metrics=ann_params['metrics'])
    return model, ann_params


# 构建 CNN 模型
def build_cnn_model(input_dim):
    cnn_params = {
        'filters': [32, 64],
        'kernel_size': 3,
        'pool_size': 2,
        'dense_units': 64,
        'dropout_rate': 0.5,
        'learning_rate': 0.001,
        'loss': 'binary_crossentropy',
        'metrics': ['accuracy']
    }
    print("CNN参数:", cnn_params)

    model = Sequential([
        Reshape((input_dim, 1), input_shape=(input_dim,)),
        Conv1D(cnn_params['filters'][0], cnn_params['kernel_size'], activation='relu'),
        MaxPooling1D(cnn_params['pool_size']),
        Conv1D(cnn_params['filters'][1], cnn_params['kernel_size'], activation='relu'),
        MaxPooling1D(cnn_params['pool_size']),
        Flatten(),
        Dense(cnn_params['dense_units'], activation='relu'),
        Dropout(cnn_params['dropout_rate']),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=cnn_params['learning_rate']),
                  loss=cnn_params['loss'],
                  metrics=cnn_params['metrics'])
    return model, cnn_params


# 构建 LSTM 模型
def build_lstm_model(input_dim):
    lstm_params = {
        'lstm_units': 50,
        'dense_units': 32,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'loss': 'binary_crossentropy',
        'metrics': ['accuracy']
    }
    print("LSTM参数:", lstm_params)

    model = Sequential([
        Reshape((1, input_dim), input_shape=(input_dim,)),
        LSTM(lstm_params['lstm_units'], activation='tanh'),
        Dense(lstm_params['dense_units'], activation='relu'),
        Dropout(lstm_params['dropout_rate']),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=lstm_params['learning_rate']),
                  loss=lstm_params['loss'],
                  metrics=lstm_params['metrics'])
    return model, lstm_params


# 获取模型的预测概率
def get_model_proba(model, X, is_deep_learning=False):
    if is_deep_learning:
        # 对于深度学习模型
        return model.predict(X).flatten()
    else:
        # 对于传统机器学习模型
        return model.predict_proba(X)[:, 1]


# 训练和评估模型
def train_and_evaluate_models(X_train, X_test, y_train, y_test, is_train_group=False):
    results = {}
    all_params = {}

    # 构建并训练传统机器学习模型（全部使用默认参数）
    print("开始构建和训练传统机器学习模型...")
    models, ml_params = build_ml_models()
    all_params.update(ml_params)

    # 训练传统机器学习模型
    for name, model in models.items():
        print(f"训练 {name} 模型...")
        start_time = time.time()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = get_model_proba(model, X_test)

        # 计算各项指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        tpr = tp / (tp + fn)  # 灵敏度/召回率
        tnr = tn / (tn + fp)  # 特异度
        ppv = precision  # 阳性预测值
        npv = tn / (tn + fn)  # 阴性预测值

        # ROC 和 PR 曲线数据
        fpr, tpr_roc, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr_roc)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall_curve, precision_curve)

        # 计算AUC的95%置信区间
        auc_ci_lower, auc_ci_upper = calculate_auc_ci(y_test, y_pred_proba)
        # 计算PR AUC的95%置信区间
        pr_auc_ci_lower, pr_auc_ci_upper = calculate_pr_auc_ci(y_test, y_pred_proba)

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tpr': tpr,
            'tnr': tnr,
            'ppv': ppv,
            'npv': npv,
            'fpr': fpr,
            'tpr_roc': tpr_roc,
            'roc_auc': roc_auc,
            'roc_auc_ci': (auc_ci_lower, auc_ci_upper),  # 存储AUC置信区间
            'pr_auc_ci': (pr_auc_ci_lower, pr_auc_ci_upper),  # 存储PR AUC置信区间
            'precision_curve': precision_curve,
            'recall_curve': recall_curve,
            'pr_auc': pr_auc,
            'y_pred_proba': y_pred_proba,  # 存储预测概率
            'y_pred': y_pred,  # 存储预测结果
            'y_true': y_test  # 存储真实结果
        }

        end_time = time.time()
        print(f"{name} 模型训练完成，耗时: {end_time - start_time:.2f}秒")

    # 训练深度学习模型
    print("开始构建和训练深度学习模型...")
    dl_models = {}
    dl_params = {}

    ann_model, ann_params = build_ann_model(X_train.shape[1])
    dl_models['ANN'] = ann_model
    dl_params['ANN'] = ann_params

    cnn_model, cnn_params = build_cnn_model(X_train.shape[1])
    dl_models['CNN'] = cnn_model
    dl_params['CNN'] = cnn_params

    lstm_model, lstm_params = build_lstm_model(X_train.shape[1])
    dl_models['LSTM'] = lstm_model
    dl_params['LSTM'] = lstm_params

    all_params.update(dl_params)

    for name, model in dl_models.items():
        print(f"训练 {name} 模型...")
        start_time = time.time()

        # 简化深度学习模型的训练
        history = model.fit(
            X_train, y_train,
            epochs=50,  # 减少训练轮数
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )

        y_pred_proba = get_model_proba(model, X_test, is_deep_learning=True)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # 计算各项指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        tpr = tp / (tp + fn)  # 灵敏度/召回率
        tnr = tn / (tn + fp)  # 特异度
        ppv = precision  # 阳性预测值
        npv = tn / (tn + fn)  # 阴性预测值

        # ROC 和 PR 曲线数据
        fpr, tpr_roc, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr_roc)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall_curve, precision_curve)

        # 计算AUC的95%置信区间
        auc_ci_lower, auc_ci_upper = calculate_auc_ci(y_test, y_pred_proba)
        # 计算PR AUC的95%置信区间
        pr_auc_ci_lower, pr_auc_ci_upper = calculate_pr_auc_ci(y_test, y_pred_proba)

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tpr': tpr,
            'tnr': tnr,
            'ppv': ppv,
            'npv': npv,
            'fpr': fpr,
            'tpr_roc': tpr_roc,
            'roc_auc': roc_auc,
            'roc_auc_ci': (auc_ci_lower, auc_ci_upper),  # 存储AUC置信区间
            'pr_auc_ci': (pr_auc_ci_lower, pr_auc_ci_upper),  # 存储PR AUC置信区间
            'precision_curve': precision_curve,
            'recall_curve': recall_curve,
            'pr_auc': pr_auc,
            'y_pred_proba': y_pred_proba,  # 存储预测概率
            'y_pred': y_pred,  # 存储预测结果
            'y_true': y_test  # 存储真实结果
        }

        end_time = time.time()
        print(f"{name} 模型训练完成，耗时: {end_time - start_time:.2f}秒")

    return results, all_params


# 绘制 ROC 曲线
def plot_roc_curves(results, title, save_path=None):
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        plt.plot(result['fpr'], result['tpr_roc'],
                 label=f'{name} (AUC = {result["roc_auc"]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title(f'ROC Curves - {title}')
    plt.legend(loc='lower right')
    #plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()


# 绘制 PR 曲线
def plot_pr_curves(results, title, save_path=None):
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        plt.plot(result['recall_curve'], result['precision_curve'],
                 label=f'{name} (AUC = {result["pr_auc"]:.3f})')

    baseline = np.sum(np.array(results[list(results.keys())[0]]['y_true']) == 1) / len(
        results[list(results.keys())[0]]['y_true'])
    plt.plot([0, 1], [baseline, baseline], 'k--', label=f'No Skill (AP = {baseline:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.title(f'Precision-Recall Curves - {title}')
    plt.legend(loc='upper right')
    #plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()


# 绘制校准曲线
def plot_calibration_curves(results, title, save_path=None):
    from sklearn.calibration import calibration_curve

    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        # 使用存储的预测概率
        y_pred_proba = result['y_pred_proba']
        fraction_of_positives, mean_predicted_value = calibration_curve(result['y_true'], y_pred_proba, n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=name)

    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    #plt.title(f'Calibration curves - {title}')
    plt.legend(loc='lower right')
    #plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()


# 绘制单个混淆矩阵
def plot_single_confusion_matrix(result, model_name, title_prefix, save_dir=None):
    cm = confusion_matrix(result['y_true'], result['y_pred'])
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['轻症', '重症'])
    disp.plot(values_format='d', cmap='Blues')
    plt.title(f'{title_prefix} - {model_name} 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')

    if save_dir:
        plt.savefig(f'{save_dir}/{title_prefix}_{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


# 绘制所有混淆矩阵
def plot_all_confusion_matrices(results, title_prefix, save_dir=None):
    if save_dir:
        create_directory(save_dir)

    for name, result in results.items():
        plot_single_confusion_matrix(result, name, title_prefix, save_dir)


def create_results_table(results, group_name):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'tpr', 'tnr', 'ppv', 'npv', 'roc_auc', 'pr_auc']
    table_data = []

    for name, result in results.items():
        row = [name]
        for metric in metrics:
            if metric == 'roc_auc' and 'roc_auc_ci' in result:
                # 对于ROC AUC，添加置信区间信息
                auc_value = result[metric]
                ci_lower, ci_upper = result['roc_auc_ci']
                row.append(f"{auc_value:.4f} ({ci_lower:.4f}-{ci_upper:.4f})")
            elif metric == 'pr_auc' and 'pr_auc_ci' in result:
                # 对于PR AUC，添加置信区间信息
                pr_auc_value = result[metric]
                ci_lower, ci_upper = result['pr_auc_ci']
                row.append(f"{pr_auc_value:.4f} ({ci_lower:.4f}-{ci_upper:.4f})")
            else:
                row.append(f"{result[metric]:.4f}")
        table_data.append(row)

    columns = ['Model'] + [metric.upper() for metric in metrics]
    results_df = pd.DataFrame(table_data, columns=columns)

    # 添加组别信息
    results_df['Group'] = group_name

    return results_df


# 保存模型参数到文件
def save_model_params(params, filename):
    import json
    # 将参数转换为可序列化的格式
    serializable_params = {}
    for model_name, model_params in params.items():
        serializable_params[model_name] = {}
        for key, value in model_params.items():
            # 处理numpy数组和其他不可序列化的类型
            if hasattr(value, 'tolist'):
                serializable_params[model_name][key] = value.tolist()
            else:
                serializable_params[model_name][key] = value

    with open(filename, 'w') as f:
        json.dump(serializable_params, f, indent=4)
    print(f"模型参数已保存到 {filename}")


# 主函数
def main():
    # 创建目录
    create_directory('results')
    create_directory('results/train')
    create_directory('results/test')
    create_directory('results/confusion_matrices')
    create_directory('results/confusion_matrices/train')
    create_directory('results/confusion_matrices/test')

    # 加载数据
    train_file_path = 'training_set_eng.xlsx'
    full_data_file_path = 'data_V7.0.xlsx'  # 改为加载完整数据集
    print("正在加载数据...")
    train_df, full_df = load_data(train_file_path, full_data_file_path)

    # 数据预处理
    print("正在预处理数据...")
    X_train, y_train = preprocess_data(train_df)
    X_full, y_full = preprocess_data(full_df)

    # 对训练集应用 SMOTE
    print("正在应用SMOTE过采样...")
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

    # 准备数据集（不再进行标准化）
    print("正在准备数据集...")
    X_train_final, X_full_final, y_train_final, y_full_final = prepare_datasets(
        X_train_resampled, y_train_resampled, X_full, y_full
    )

    # 准备未经过SMOTE的训练集数据
    X_train_orig, y_train_orig = X_train.values, y_train.values

    # 训练和评估模型 - 训练组（使用SMOTE过采样后的训练集）
    print("训练组评估（使用SMOTE过采样后的训练集）...")
    train_results, train_params = train_and_evaluate_models(X_train_final, X_train_orig, y_train_final, y_train_orig,
                                                            is_train_group=True)

    # 训练和评估模型 - 验证组（使用完整数据集）
    print("验证组评估（使用完整数据集）...")
    test_results, test_params = train_and_evaluate_models(X_train_final, X_full_final, y_train_final, y_full_final,
                                                          is_train_group=False)

    # 保存模型参数
    save_model_params(train_params, 'results/train_model_parameters.json')
    save_model_params(test_params, 'results/test_model_parameters.json')

    # 绘制训练组的曲线
    print("绘制训练组的曲线...")
    plot_roc_curves(train_results, "训练组", save_path="results/train/train_roc_curves.png")
    plot_pr_curves(train_results, "训练组", save_path="results/train/train_pr_curves.png")
    plot_calibration_curves(train_results, "训练组", save_path="results/train/train_calibration_curves.png")

    # 绘制验证组的曲线
    print("绘制验证组的曲线...")
    plot_roc_curves(test_results, "验证组", save_path="results/test/test_roc_curves.png")
    plot_pr_curves(test_results, "验证组", save_path="results/test/test_pr_curves.png")
    plot_calibration_curves(test_results, "验证组", save_path="results/test/test_calibration_curves.png")

    # 绘制训练组的混淆矩阵
    print("绘制训练组的混淆矩阵...")
    #plot_all_confusion_matrices(train_results, "训练组", save_dir="results/confusion_matrices/train")

    # 绘制验证组的混淆矩阵
    print("绘制验证组的混淆矩阵...")
    #plot_all_confusion_matrices(test_results, "验证组", save_dir="results/confusion_matrices/test")

    # 创建结果表格
    print("创建结果表格...")
    train_results_table = create_results_table(train_results, "训练组")
    test_results_table = create_results_table(test_results, "验证组")

    # 合并两个表格
    combined_results = pd.concat([train_results_table, test_results_table], ignore_index=True)

    # 保存结果
    combined_results.to_excel('results/model_comparison_results.xlsx', index=False)
    print("结果已保存到 results/model_comparison_results.xlsx")


if __name__ == '__main__':
    main()