import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from tabpfn import TabPFNRegressor
from data_processor import DataProcessor
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # 加载数据
    df = pd.read_excel('D:/A/CSU/数据集/corn_xlsl/mp6_corn.xlsx')
    
    # 提取通道数据和目标变量
    spectra = df.iloc[:, :700].values  # 直接转换为numpy数组
    y = df.iloc[:, 700].values.ravel()  # 目标变量（水分含量）
    
    # 初始化数据处理器
    processor = DataProcessor(spectra, y)
    
        # 定义数据划分配置 - 使用字典形式
    split_config = {
        'methods': [
            {'name': 'random', 'params': {'test_size': 0.25, 'random_state': 42}},
            {'name': 'spxy', 'params': {'test_size': 0.25}},
            {'name': 'kennard-stone', 'params': {'test_size': 0.25}}
        ],
        'active_method': 'kennard-stone'  # 指定要使用的数据划分方法
    }
    
    # 根据配置加载并分割数据集
    for method_entry in split_config['methods']:
        method_name = method_entry['name']
        method_params = method_entry['params']
        
        if method_name == split_config['active_method']:
            print(f"使用数据划分方法: {method_name}")
            
            if method_name == 'random':
                processor.load_and_split_data(
                    method='random',
                    test_size=method_params['test_size'],
                    random_state=method_params['random_state']
                )
            elif method_name == 'spxy':
                processor.load_and_split_data(
                    method='spxy',
                    test_size=method_params['test_size']
                )
            elif method_name == 'kennard-stone':
                processor.load_and_split_data(
                    method='kennard-stone',
                    test_size=method_params['test_size']
                )
            break

    # 定义预处理和特征选择方法
    preprocess_methods = ['derivative']  # 可以添加其他预处理方法
    feature_selection_methods = ['uniform_sampling']  # 确保是列表类型
    
    # 定义采样数目
    count = 200  
    
    # 处理数据
    processed_data = {}
    
    # 应用预处理方法
    processed_X_cal = processor.X_cal.copy()
    processed_X_test = processor.X_test.copy()
    
    if 'airPLS' in preprocess_methods:
        print("应用airPLS基线校正...")
        processed_X_cal = processor.baseline_correction_airPLS(processed_X_cal, lambda_=100, itermax=15)
        processed_X_test = processor.baseline_correction_airPLS(processed_X_test, lambda_=100, itermax=15)
    
    if 'MSC' in preprocess_methods:
        print("应用多元散射校正...")
        processed_X_cal = processor.perform_msc(processed_X_cal)
        processed_X_test = processor.perform_msc(processed_X_test)
    
    if 'SNV' in preprocess_methods:
        print("应用标准正态变量变换...")
        processed_X_cal = processor.perform_standard_normal_variate(processed_X_cal)
        processed_X_test = processor.perform_standard_normal_variate(processed_X_test)
    
    if 'Savitzky-Golay' in preprocess_methods:
        print("应用Savitzky-Golay平滑滤波...")
        processed_X_cal = processor.perform_savgol(processed_X_cal)
        processed_X_test = processor.perform_savgol(processed_X_test)

    if 'detrend' in preprocess_methods:
        print("应用去趋势...")
        processed_X_cal = processor.perform_detrend(processed_X_cal)
        processed_X_test = processor.perform_detrend(processed_X_test)

    if 'derivative' in preprocess_methods:
        print("一阶微分")
        processed_X_cal = processor.spectral_first_order_derivative(processed_X_cal)
        processed_X_test = processor.spectral_first_order_derivative(processed_X_test)
    
    # 应用特征选择方法
    for method in feature_selection_methods:
        if method == 'uniform_sampling':
            print(f"应用均匀采样 (count={count})...")
            # 正确传递参数：X_cal, X_test, count
            sampled_cal, sampled_test = processor.uniform_sampling(
                processed_X_cal, 
                processed_X_test, 
                count=count  # 使用关键字参数确保正确传递
            )
            processed_data[f'uniform_sampling_{count}'] = {
                'cal': sampled_cal,
                'test': sampled_test
            }
        elif method == 'univariant_selection':
            print("应用单变量特征选择...")
            selected_indices = processor.perform_univariant_selection(processed_X_cal, processor.y_cal, count=count)
            processed_data['univariant_selection'] = {
                'cal': processed_X_cal[:, selected_indices],
                'test': processed_X_test[:, selected_indices]
            }
        elif method == 'recursive_elimination':
            print("应用递归特征消除...")
            selected_indices = processor.perform_recursive_elimination(processed_X_cal, processor.y_cal, count=count)
            processed_data['recursive_elimination'] = {
                'cal': processed_X_cal[:, selected_indices],
                'test': processed_X_test[:, selected_indices]
            }
        elif method == 'pca':
            print("应用主成分分析...")
            pca_cal = processor.perform_pca(processed_X_cal, count=count)
            pca_test = processor.perform_pca(processed_X_test, count=count)
            processed_data['pca'] = {
                'cal': pca_cal,
                'test': pca_test
            }
        elif method == 'uve':
            print("应用无信息变量消除...")
            selected_indices = processor.perform_uve(processed_X_cal, processor.y_cal, count=count)
            processed_data['uve'] = {
                'cal': processed_X_cal[:, selected_indices],
                'test': processed_X_test[:, selected_indices]
            }
    
    print("数据处理完成！")
    print("处理后的数据键:", list(processed_data.keys()))
    
    # 定义模型及其参数网格
    models = {
        'PLSR': {
            'estimator': PLSRegression(),
            'param_grid': {'n_components': [5, 10, 15, 20]}
        },
        'SVR': {
            'estimator': SVR(),
            'param_grid': {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
        },
        'Random Forest': {
            'estimator': RandomForestRegressor(random_state=42),
            'param_grid': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
        },
        'TabPFN': {
            'estimator': TabPFNRegressor(
                device='cpu',
                ignore_pretraining_limits=True,
                inference_precision=torch.float32,
                memory_saving_mode=True,
                fit_mode='fit_preprocessors',
                random_state=42
            ),
            'param_grid': {}  # TabPFN 模型不进行网格搜索调参
        }
    }
    
    # 创建一个字典来存储模型的预测结果和最佳参数
    predictions = {model_name: [] for model_name in models.keys()}
    best_params = {model_name: [] for model_name in models.keys()}
    mse_results = {model_name: [] for model_name in models.keys()}
    r2_results = {model_name: [] for model_name in models.keys()}
    
    # 对每个处理后的数据集进行建模和评估
    for method_name, method_data in processed_data.items():
        X_cal, X_test = method_data['cal'], method_data['test']
        y_cal, y_test = processor.y_cal, processor.y_test
        
        # 确保数据是2D数组
        if X_cal.ndim == 1:
            X_cal = X_cal.reshape(-1, 1)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        
        print(f"\n正在处理: {method_name}, 数据形状: 训练集 {X_cal.shape}, 测试集 {X_test.shape}")
        
        for model_name, model_info in models.items():
            print(f"训练模型: {model_name}...")
            model = model_info['estimator']
            param_grid = model_info['param_grid']
            
            if param_grid:  # 如果有参数网格，进行网格搜索调参
                # 调参
                grid_search = GridSearchCV(
                    model, param_grid, cv=3, 
                    scoring='neg_mean_squared_error', verbose=0, n_jobs=-1
                )
                grid_search.fit(X_cal, y_cal)
                best_model = grid_search.best_estimator_
                best_params[model_name].append(grid_search.best_params_)
            else:  # 如果没有参数网格，直接使用默认参数
                best_model = model
                best_params[model_name].append('Default params')
                # 确保模型被正确训练
                best_model.fit(X_cal, y_cal)
            
            # 预测
            y_pred = best_model.predict(X_test)
            
            # 评估
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # 存储结果
            predictions[model_name].append(y_pred)
            rmse_results[model_name].append(mse)
            r2_results[model_name].append(r2)
            
            print(f"方法: {method_name}, 模型: {model_name}, 最佳参数: {best_params[model_name][-1]}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
            # 可视化预测结果与真实值的对比
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.7)
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
            plt.xlabel('True Value')
            plt.ylabel('Predicated Value')
            plt.title(f'{method_name} - {model_name} Prediction vs True Value')
            
            # 添加评估指标文本
            plt.text(min(y_test), max(y_pred) - 0.1*(max(y_pred)-min(y_pred)), 
                    f'MSE: {mse:.4f}\nR²: {r2:.4f}', 
                    verticalalignment='top')
            plt.tight_layout()
            plt.savefig(f"{method_name}_{model_name}_prediction.png", dpi=300)
            plt.show()
    
    # 可视化不同处理方法后的光谱数据
    for method_name, method_data in processed_data.items():
        # 随机选择一个样本进行可视化
        sample_index = np.random.randint(0, method_data['cal'].shape[0])
        
        # 确保数据是1D
        sample_data = method_data['cal'][sample_index, :]
        if sample_data.ndim > 1:
            sample_data = sample_data.flatten()
        
        plt.figure(figsize=(10, 6))
        plt.plot(sample_data, label='processed spectrum', color='blue')
        plt.title(f'{method_name} processed spectrum')
        plt.xlabel('Wavelength Index')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{method_name}_processed_spectrum.png", dpi=300)
        plt.show()
    
    # 打印模型的性能结果和最佳参数
    print("\n最终模型性能评估:")
    for model_name in models.keys():
        print(f"\n模型: {model_name}")
        print(f"{'方法':<25}{'最佳参数':<45}{'RMSE':<15}{'R²':<10}")
        for i, method_name in enumerate(processed_data.keys()):
            params = best_params[model_name][i]
            rmse = rmse_results[model_name][i]
            r2 = r2_results[model_name][i]
            print(f"{method_name:<25}{str(params):<45}{rmse:<15.4f}{r2:<10.4f}")

