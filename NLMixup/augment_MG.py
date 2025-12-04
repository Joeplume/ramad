import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random
import argparse
from datetime import datetime

def import_data(file_path, substance='MG'):
    """
    导入CSV文件中的光谱数据。

    参数:
        file_path (str): CSV文件的路径。
        substance (str): 要处理的物质名称，默认为MG。

    返回:
        tuple: 包含数据DataFrame、浓度数组、光谱矩阵、波数范围和光谱列名列表。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"指定路径下未找到数据文件: {file_path}")

    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 筛选指定物质的数据
    df = df[df['Category'] == substance].copy()
    
    if len(df) == 0:
        raise ValueError(f"未找到物质 {substance} 的数据")
    
    # 获取浓度列
    if 'Conc' in df.columns:
        df.rename(columns={'Conc': 'Concentration'}, inplace=True)
    
    if 'Concentration' not in df.columns:
        raise ValueError("数据中缺少必要的浓度列 'Concentration' 或 'Conc'")
    
    # 提取光谱列（假设除了Category、Concentration等特定列之外的所有列都是光谱数据）
    non_spectral_cols = ['Category', 'Concentration', 'Title', 'Water', 'Det_Type', 'Meas_Method']
    spectra_columns = [col for col in df.columns if col not in non_spectral_cols]
    
    if not spectra_columns:
        raise ValueError("未找到光谱数据列")
    
    # 确保光谱数据为浮点型
    for col in spectra_columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    concentrations = df['Concentration'].values
    spectra = df[spectra_columns].values
    
    # 定义波数范围（根据实际情况调整，这里用列索引代替）
    x_range = np.arange(len(spectra_columns))
    
    return df, concentrations, spectra, x_range, spectra_columns

def generate_combined_spectrum(selected_spectra, coefficients):
    """
    生成组合光谱。

    参数:
        selected_spectra (list of array): 选择的光谱列表。
        coefficients (list of float): 对应的系数列表。

    返回:
        array: 组合后的光谱。
    """
    combined_spectrum = np.zeros_like(selected_spectra[0])
    for spec, coeff in zip(selected_spectra, coefficients):
        combined_spectrum += coeff * spec
    return combined_spectrum

def find_coefficients(target_conc, base_concs, tolerance=1e-3):
    """
    寻找系数，使得它们的加权和等于目标浓度，且系数之和等于1。

    参数:
        target_conc (float): 目标浓度。
        base_concs (list of float): 基础浓度列表。
        tolerance (float): 目标浓度与实际浓度的容差。

    返回:
        tuple: 系数元组。如果未找到合适的系数，返回None。
    """
    # 根据基础浓度的数量确定系数个数
    num_coeffs = len(base_concs)
    x0 = [1.0 / num_coeffs] * num_coeffs  # 初始猜测

    # 找到两个最接近目标浓度的基浓度（一个小于，一个大于）
    sorted_base = sorted(base_concs)
    lower = max([c for c in sorted_base if c <= target_conc], default=None)
    upper = min([c for c in sorted_base if c >= target_conc], default=None)

    if lower is None or upper is None:
        # 如果目标浓度超出了基础浓度的范围，返回None
        return None

    # 处理边界情况：如果目标浓度正好等于某个基浓度
    if lower == upper:
        idx = sorted_base.index(lower)
        if idx == 0:
            upper = sorted_base[idx + 1] if len(sorted_base) > 1 else None
        elif idx == len(sorted_base) - 1:
            lower = sorted_base[idx - 1] if len(sorted_base) > 1 else None
        else:
            # 选择离目标浓度最近的相邻基浓度
            next_lower = sorted_base[idx - 1]
            next_upper = sorted_base[idx + 1]
            if abs(next_lower - target_conc) >= abs(next_upper - target_conc):
                upper = next_upper
            else:
                lower = next_lower

    if lower is None or upper is None:
        return None

    try:
        idx_lower = base_concs.index(lower)
        idx_upper = base_concs.index(upper)
    except ValueError:
        return None

    # 约束条件
    constraints = [
        {'type': 'eq', 'fun': lambda x: sum(x) - 1},  # 系数和为1
        {'type': 'eq', 'fun': lambda x: sum(ci * xi for ci, xi in zip(base_concs, x)) - target_conc},  # 加权和等于目标浓度
        {'type': 'ineq', 'fun': lambda x: x[idx_lower] + x[idx_upper] - 0.65}  # 最接近的两个系数之和不小于0.65
    ]

    # 目标函数：最大化最接近的两个系数之和
    def objective(x):
        return -(x[idx_lower] + x[idx_upper])  # 负号因为使用minimize

    # 边界条件：系数都在0和1之间
    bounds = [(0, 1)] * num_coeffs

    # 优化
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        calculated_conc = sum(ci * xi for ci, xi in zip(base_concs, result.x))
        if abs(calculated_conc - target_conc) <= tolerance:
            return tuple(result.x)
    return None

def generate_spectrum_by_combination(conc, data_df, x_range, spectra_columns, base_concs, min_samples=5):
    """
    通过组合现有光谱生成新的光谱。

    参数:
        conc (float): 目标浓度。
        data_df (DataFrame): 原始数据的DataFrame，包含'Concentration'和光谱列。
        x_range (array): 波数范围。
        spectra_columns (list): 光谱列名列表。
        base_concs (list of float): 基础浓度列表。
        min_samples (int): 每个浓度至少需要的样本数。

    返回:
        array: 生成的光谱。
    """
    # 寻找指定浓度的数据
    selected_data = {}
    for target in base_concs:
        df_subset = data_df[data_df['Concentration'] == target]
        if len(df_subset) < min_samples:
            # 如果样本数不足，则减少最小样本数要求
            available = len(df_subset)
            if available == 0:
                raise ValueError(f"未找到Concentration为{target}的数据")
            min_samples_adjusted = min(available, min_samples)
            selected_data[target] = df_subset.sample(n=min_samples_adjusted, replace=True).reset_index(drop=True)
            print(f"警告: 浓度 {target} 的样本数量为 {available}，少于{min_samples}。使用替换采样。")
        else:
            selected_data[target] = df_subset.sample(n=min_samples, replace=False).reset_index(drop=True)

    # 随机选择每个浓度的一个样本
    selected_spectra = []
    for target in base_concs:
        idx = random.randint(0, len(selected_data[target]) - 1)
        spectrum = selected_data[target].iloc[idx][spectra_columns].values
        selected_spectra.append(spectrum)

    # 寻找合适的系数
    coefficients = find_coefficients(conc, base_concs)
    if coefficients is None:
        raise ValueError(f"无法找到满足条件的系数，使得基础浓度的加权和等于{conc}")

    # 生成组合光谱
    combined_spectrum = generate_combined_spectrum(selected_spectra, coefficients)

    return combined_spectrum

def augment_spectra(data_df, x_range, target_concentrations, samples_per_conc=1, spectra_columns=[], base_concs=[]):
    """
    对数据进行扩增，生成新的光谱数据。

    参数:
        data_df (DataFrame): 原始数据的DataFrame，包含'Concentration'和光谱列。
        x_range (array): 波数范围。
        target_concentrations (list of float): 目标生成的浓度列表。
        samples_per_conc (int): 每个浓度生成的样本数量。
        spectra_columns (list): 光谱列名列表。
        base_concs (list of float): 基础浓度列表。

    返回:
        list: 扩增后的数据列表，每个元素是一个字典，包含'Concentration'和'Spectrum'。
    """
    augmented_data = []
    for conc in target_concentrations:
        for _ in range(samples_per_conc):
            try:
                new_spectrum = generate_spectrum_by_combination(conc, data_df, x_range, spectra_columns, base_concs)
                augmented_data.append({
                    'Concentration': conc,
                    'Spectrum': new_spectrum
                })
            except ValueError as e:
                print(f"生成浓度{conc}的光谱时出错: {e}")
    return augmented_data

def save_to_csv(augmented_data, spectra_columns, category, output_file=None):
    """
    将扩增后的数据保存到CSV文件。

    参数:
        augmented_data (list): 扩增后的数据列表。
        spectra_columns (list): 光谱数据列名。
        category (str): 物质类别名称。
        output_file (str): 输出CSV文件名，如果为None则自动生成。
    
    返回:
        str: 保存的文件路径。
    """
    # 构建DataFrame
    data = {
        'Category': [category] * len(augmented_data),
        'Concentration': [entry['Concentration'] for entry in augmented_data]
    }

    # 添加光谱数据列
    for i, col in enumerate(spectra_columns):
        data[col] = [entry['Spectrum'][i] for entry in augmented_data]

    augmented_df = pd.DataFrame(data)

    # 如果未指定输出文件名，则自动生成
    if output_file is None:
        timestamp = datetime.now().strftime("%m%d%H%M")
        output_file = f"{category}_{len(augmented_data)}_{timestamp}.csv"

    # 保存为CSV文件
    augmented_df.to_csv(output_file, index=False)
    print(f"扩增后的数据已保存到 '{output_file}'。")
    
    return output_file

def visualize_spectra(data, x_range, num_plots=5, title='光谱可视化'):
    """
    可视化部分光谱数据。

    参数:
        data (list or DataFrame): 光谱数据列表或DataFrame。
        x_range (array): 波数范围。
        num_plots (int): 要绘制的光谱数量。
        title (str): 图表标题。
    """
    plt.figure(figsize=(12, 8))
    
    if isinstance(data, pd.DataFrame):
        # 获取光谱列
        non_spectral_cols = ['Category', 'Concentration']
        spectral_cols = [col for col in data.columns if col not in non_spectral_cols]
        
        # 随机选择num_plots个样本
        sample_indices = np.random.choice(len(data), min(num_plots, len(data)), replace=False)
        
        for idx in sample_indices:
            spectrum = data.iloc[idx][spectral_cols].values
            conc = data.iloc[idx]['Concentration']
            plt.plot(x_range, spectrum, label=f"Conc: {conc}")
    else:
        # 从列表中绘制
        sample_indices = np.random.choice(len(data), min(num_plots, len(data)), replace=False)
        for idx in sample_indices:
            spectrum = data[idx]['Spectrum']
            conc = data[idx]['Concentration']
            plt.plot(x_range, spectrum, label=f"Conc: {conc}")
    
    plt.xlabel('Raman Shift (点位)', fontsize=14)
    plt.ylabel('Intensity (a.u.)', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.tight_layout()
    
    # 保存图像
    timestamp = datetime.now().strftime("%m%d%H%M")
    plt.savefig(f"{title}_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MG物质光谱数据扩增工具')
    parser.add_argument('--csv_path', type=str, default='高质量预处理后.csv', help='CSV文件路径')
    parser.add_argument('--target_count', type=int, default=50, help='每个浓度生成的样本数量')
    parser.add_argument('--min_conc', type=float, default=7.0, help='最小目标浓度')
    parser.add_argument('--max_conc', type=float, default=10.0, help='最大目标浓度')
    parser.add_argument('--step', type=float, default=0.1, help='浓度步长')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
    parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    
    # 获取CSV文件的绝对路径
    csv_path = os.path.abspath(args.csv_path)
    if not os.path.exists(csv_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, args.csv_path)
        if not os.path.exists(csv_path):
            print(f"错误：找不到CSV文件 {args.csv_path}")
            return
    
    print("=" * 50)
    print("MG物质光谱数据扩增工具")
    print("=" * 50)
    print(f"CSV文件路径: {csv_path}")
    print(f"目标样本数量/浓度: {args.target_count}")
    print(f"浓度范围: {args.min_conc} - {args.max_conc}, 步长: {args.step}")
    print(f"随机种子: {args.random_seed}")
    print("=" * 50)
    
    substance = 'MG'
    
    try:
        # 导入数据
        data_df, concentrations, spectra, x_range, spectra_columns = import_data(csv_path, substance)
        print(f"成功导入{substance}数据。样本数: {len(concentrations)}")
    except Exception as e:
        print(f"导入数据时出错: {e}")
        return
    
    # 可视化部分原始光谱
    if args.visualize:
        visualize_spectra(data_df, x_range, num_plots=5, title=f'{substance}原始光谱')
    
    # 获取基础浓度列表（所有可用的浓度）
    base_concs = sorted(data_df['Concentration'].unique())
    print(f"检测到的基础浓度: {base_concs}")
    
    # 检查每个基础浓度的样本数量
    for bc in base_concs:
        count = len(data_df[data_df['Concentration'] == bc])
        print(f"浓度 {bc} 的样本数量: {count}")
        if count < 5:
            print(f"警告: 浓度 {bc} 的样本数量少于5个。可能导致数据扩增不稳定。")
    
    # 定义目标浓度列表
    target_concentrations = np.arange(args.min_conc, args.max_conc + args.step/2, args.step)
    target_concentrations = [round(c, 2) for c in target_concentrations]
    
    print(f"目标生成浓度: {target_concentrations}")
    print(f"共{len(target_concentrations)}个不同浓度，每个浓度生成{args.target_count}个样本")
    print(f"总目标样本数: {len(target_concentrations) * args.target_count}")
    
    # 数据扩增
    print("\n正在进行数据扩增...")
    augmented_data = augment_spectra(
        data_df,
        x_range,
        target_concentrations,
        samples_per_conc=args.target_count,
        spectra_columns=spectra_columns,
        base_concs=base_concs
    )
    print(f"数据扩增完成。成功扩增样本数：{len(augmented_data)}")
    
    # 保存扩增数据
    if augmented_data:
        output_file = save_to_csv(augmented_data, spectra_columns, substance)
        
        # 转换为DataFrame以便可视化
        if args.visualize:
            augmented_df = pd.DataFrame({
                'Category': [substance] * len(augmented_data),
                'Concentration': [entry['Concentration'] for entry in augmented_data]
            })
            
            # 添加光谱数据列
            for i, col in enumerate(spectra_columns):
                augmented_df[col] = [entry['Spectrum'][i] for entry in augmented_data]
            
            # 可视化部分扩增后的光谱
            visualize_spectra(augmented_df, x_range, num_plots=5, title=f'{substance}扩增光谱')
    else:
        print("警告：未生成任何扩增数据")
    
    print("\n处理完成!")

if __name__ == "__main__":
    main() 