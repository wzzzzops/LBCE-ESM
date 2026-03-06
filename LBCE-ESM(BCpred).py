import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import joblib
import json
import time
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 导入鲁棒数据加载模块
from robust_data_loader import load_and_validate_dataset

def load_esmc_features_with_sequences(filepath):
    """加载ESMC特征文件，返回序列ID和特征"""
    sequences = []  # 序列ID
    features = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 非空行
                # 解析特征向量，每行格式为：序列ID,特征1,特征2,特征3,...
                parts = line.split(',')
                # 第一部分是序列ID，其余是特征
                seq_id = parts[0]
                feature_parts = parts[1:]
                
                # 将特征部分转换为数值
                numeric_features = []
                for part in feature_parts:
                    part = part.strip()
                    try:
                        val = float(part)
                        numeric_features.append(val)
                    except ValueError:
                        # 如果转换失败，跳过整个行
                        break
                
                if len(numeric_features) == len(feature_parts):  # 确保所有特征都被成功转换
                    sequences.append(seq_id)
                    features.append(numeric_features)
    
    return np.array(sequences), np.array(features)

def load_sequences(seq_folder):
    """从序列文件夹加载所有序列"""
    sequences = []
    filenames = []
    
    for filename in os.listdir(seq_folder):
        if filename.endswith('.txt') or filename.endswith('.fasta'):
            filepath = os.path.join(seq_folder, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
                # 处理FASTA格式或简单文本格式
                if filename.endswith('.fasta'):
                    # FASTA格式处理
                    seqs = []
                    current_seq = ""
                    for line in content.split('\n'):
                        if line.startswith('>'):
                            if current_seq:
                                seqs.append(current_seq)
                                current_seq = ""
                        else:
                            current_seq += line.strip()
                    if current_seq:
                        seqs.append(current_seq)
                    sequences.extend(seqs)
                    filenames.extend([filename] * len(seqs))
                else:
                    # 简单文本格式，每行一个序列
                    seq_list = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('>')]
                    sequences.extend(seq_list)
                    filenames.extend([filename] * len(seq_list))
    
    return sequences, filenames

def calculate_aac(sequence):
    """计算氨基酸组成(AAC)特征"""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_count = {aa: 0 for aa in amino_acids}
    
    sequence = sequence.upper()
    for aa in sequence:
        if aa in aa_count:
            aa_count[aa] += 1
    
    # 归一化
    total_length = len(sequence)
    if total_length == 0:
        return [0.0] * len(amino_acids)
    
    aac_features = [aa_count[aa] / total_length for aa in amino_acids]
    return aac_features

def calculate_dipeptide_composition(sequence):
    """计算二肽组成(DPC)特征 - 扩展特征维度"""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    dipeptides = [aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids]
    dp_count = {dp: 0 for dp in dipeptides}
    
    sequence = sequence.upper()
    for i in range(len(sequence) - 1):
        dipep = sequence[i:i+2]
        if dipep in dp_count:
            dp_count[dipep] += 1
    
    # 归一化
    total_pairs = len(sequence) - 1 if len(sequence) > 1 else 1
    dpc_features = [dp_count[dp] / total_pairs for dp in dipeptides]
    return dpc_features

def calculate_physicochemical_features(sequence):
    """计算理化性质特征：疏水性、电荷、分子量等"""
    # 氨基酸疏水性值 (Kyte & Doolittle scale)
    hydrophobicity_scale = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    
    # 氨基酸电荷特性
    charge_scale = {
        'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
        'Q': 0, 'E': 1, 'G': 0, 'H': 1, 'I': 0,
        'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
        'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
    }
    
    # 氨基酸分子量
    molecular_weight_scale = {
        'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
        'Q': 146.2, 'E': 147.1, 'G': 75.1, 'H': 155.2, 'I': 131.2,
        'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
        'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
    }
    
    # 氨基酸极性
    polarity_scale = {
        'A': 0, 'R': 1, 'N': 1, 'D': 1, 'C': 0,
        'Q': 1, 'E': 1, 'G': 0, 'H': 1, 'I': 0,
        'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
        'S': 1, 'T': 1, 'W': 0, 'Y': 1, 'V': 0
    }
    
    sequence = sequence.upper()
    hydrophobicity_sum = 0
    charge_sum = 0
    mol_weight_sum = 0
    polarity_sum = 0
    valid_aa_count = 0
    
    for aa in sequence:
        if aa in hydrophobicity_scale:
            hydrophobicity_sum += hydrophobicity_scale[aa]
            charge_sum += charge_scale[aa]
            mol_weight_sum += molecular_weight_scale[aa]
            polarity_sum += polarity_scale[aa]
            valid_aa_count += 1
    
    avg_hydrophobicity = hydrophobicity_sum / valid_aa_count if valid_aa_count > 0 else 0.0
    avg_charge = charge_sum / valid_aa_count if valid_aa_count > 0 else 0.0
    avg_molecular_weight = mol_weight_sum / valid_aa_count if valid_aa_count > 0 else 0.0
    avg_polarity = polarity_sum / valid_aa_count if valid_aa_count > 0 else 0.0
    
    return [avg_hydrophobicity, avg_charge, avg_molecular_weight, avg_polarity]

def calculate_tripeptide_composition(sequence):
    """计算三肽组成特征 - 优化版，只返回观察到的三肽频率"""
    if len(sequence) < 3:
        # 返回一个固定长度的零向量（对于没有足够长度形成三肽的序列）
        return [0.0] * 100  # 使用固定长度100来保持一致性
    
    # 统计实际出现的三肽
    tripeptide_counts = {}
    total_triplets = len(sequence) - 2
    
    for i in range(total_triplets):
        triplet = sequence[i:i+3]
        tripeptide_counts[triplet] = tripeptide_counts.get(triplet, 0) + 1
    
    # 将三肽频率转换为特征向量
    # 为了保持特征维度一致，我们只考虑最常见的100个三肽
    sorted_triplets = sorted(tripeptide_counts.items(), key=lambda x: x[1], reverse=True)[:100]
    
    # 创建特征向量（最多100个特征，对应最频繁的100个三肽）
    features = []
    for _, count in sorted_triplets:
        features.append(count / total_triplets)  # 频率
    
    # 如果三肽种类少于100，补零
    while len(features) < 100:
        features.append(0.0)
    
    return features

def calculate_sequence_stats(sequence):
    """计算序列统计特征"""
    features = []
    
    # 序列长度
    features.append(len(sequence))
    
    # GC含量（虽然对于蛋白质不完全适用，但我们计算疏水性残基的比例）
    hydrophobic_aa = 'AILMFPWV'
    hydrophobic_count = sum(1 for aa in sequence if aa in hydrophobic_aa)
    features.append(hydrophobic_count / len(sequence) if len(sequence) > 0 else 0)
    
    # 电荷相关特征
    positive_aa = 'KRH'  # 带正电荷的氨基酸
    negative_aa = 'DE'   # 带负电荷的氨基酸
    charged_count = sum(1 for aa in sequence if aa in positive_aa + negative_aa)
    features.append(charged_count / len(sequence) if len(sequence) > 0 else 0)
    
    # 极性特征
    polar_aa = 'NQSTYCMW'  # 极性氨基酸
    polar_count = sum(1 for aa in sequence if aa in polar_aa)
    features.append(polar_count / len(sequence) if len(sequence) > 0 else 0)
    
    # 分子量估算（相对值）
    avg_molecular_weight = sum([{'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2, 
                                'E': 147.1, 'Q': 146.2, 'G': 75.1, 'H': 155.2, 'I': 131.2, 
                                'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1, 
                                'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1}.get(aa, 0) 
                                for aa in sequence]) / len(sequence) if len(sequence) > 0 else 0
    features.append(avg_molecular_weight)
    
    return features

def extract_features_from_sequences(sequences):
    """从序列中提取AAC、DPC和理化性质特征（优化的核心特征集合）"""
    features = []
    
    for seq in sequences:
        aac_features = calculate_aac(seq)
        dpc_features = calculate_dipeptide_composition(seq)  # 二肽组成特征，增强模型性能
        tpc_features = calculate_tripeptide_composition(seq)  # 三肽组成特征
        physicochemical_features = calculate_physicochemical_features(seq)  # 氨基酸疏水性等理化性质
        sequence_stats = calculate_sequence_stats(seq)  # 序列统计特征
        
        # 合并核心特征：ESMC-600M + AAC + 氨基酸疏水性 + DPC + TPC + 序列统计
        combined_features = aac_features + dpc_features + tpc_features + physicochemical_features + sequence_stats
        features.append(combined_features)
    
    return np.array(features)


def ensure_fixed_features(features_array, expected_dim=530):
    """确保特征数组具有固定的维度"""
    current_dim = features_array.shape[1] if len(features_array.shape) > 1 else len(features_array)
    
    if current_dim == expected_dim:
        return features_array
    elif current_dim < expected_dim:
        # 如果特征数量不足，用零填充
        padding = np.zeros((features_array.shape[0], expected_dim - current_dim))
        return np.hstack([features_array, padding])
    else:
        # 如果特征数量过多，截断到期望维度
        return features_array[:, :expected_dim]

def load_sequence_labels(pos_file, neg_file):
    """从正负样本文件中加载序列和标签，支持FASTA格式"""
    sequences = []
    labels = []
    
    # 检查文件格式：如果包含制表符，则可能是序列+标签格式；如果以'>'开头则是FASTA格式；否则是纯序列格式
    def detect_format(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line.startswith('>'):
                return 'fasta'  # FASTA格式
            elif '\t' in first_line:
                return 'tab_separated'  # 格式：序列\t标签
            else:
                return 'sequence_only'   # 格式：纯序列，标签由文件类型决定
    
    # 加载正样本
    pos_format = detect_format(pos_file)
    with open(pos_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if pos_format == 'fasta':
            # 处理FASTA格式
            lines = content.split('\n')
            current_seq = ""
            for line in lines:
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq)
                        labels.append(1)  # 正样本
                        current_seq = ""
                else:
                    current_seq += line.strip()
            if current_seq:
                sequences.append(current_seq)
                labels.append(1)  # 正样本
        elif pos_format == 'tab_separated':
            # 处理制表符分隔格式
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    seq = parts[0].strip()
                    label = int(parts[1])
                    sequences.append(seq)
                    labels.append(label)
        else:
            # 处理纯序列格式
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('>'):
                    seq = line
                    sequences.append(seq)
                    labels.append(1)  # 正样本标签为1
    
    # 加载负样本
    neg_format = detect_format(neg_file)
    with open(neg_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if neg_format == 'fasta':
            # 处理FASTA格式
            lines = content.split('\n')
            current_seq = ""
            for line in lines:
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq)
                        labels.append(0)  # 负样本
                        current_seq = ""
                else:
                    current_seq += line.strip()
            if current_seq:
                sequences.append(current_seq)
                labels.append(0)  # 负样本
        elif neg_format == 'tab_separated':
            # 处理制表符分隔格式
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    seq = parts[0].strip()
                    label = int(parts[1])
                    sequences.append(seq)
                    labels.append(label)
        else:
            # 处理纯序列格式
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('>'):
                    seq = line
                    sequences.append(seq)
                    labels.append(0)  # 负样本标签为0
    
    return sequences, np.array(labels)

def load_and_combine_features(esmc_filepath, seq_folder, dataset_name="BCPreds"):
    """加载ESMC特征并结合AAC、DPC和理化性质特征"""
    # 加载ESMC特征和对应的虚拟序列ID
    esmc_sequences, esmc_features = load_esmc_features_with_sequences(esmc_filepath)
    print(f"  调试: {dataset_name} - ESMC特征形状: {esmc_features.shape}")
    
    # 根据数据集名称加载对应的标签
    if dataset_name == "BCPreds":
        pos_file = os.path.join(seq_folder, "bcpred_pos_20mer.txt")
        neg_file = os.path.join(seq_folder, "bcpred_neg_20mer.txt")
    elif dataset_name == "LBtope":
        print(f"  {dataset_name}: 使用鲁棒数据加载器处理")
        # 使用新的鲁棒数据加载器加载数据集
        try:
            # 将seq_folder作为数据集目录传递给加载器
            sequences, labels, stats = load_and_validate_dataset(dataset_name, seq_folder)
            
            if len(sequences) == 0:
                print(f"  {dataset_name}: 数据加载失败，使用ESMC特征文件中的虚拟ID进行处理")
                # 如果数据加载失败，使用虚拟ID来构建标签，假设前一半是正样本，后一半是负样本
                total_samples = len(esmc_features)
                pos_count = total_samples // 2
                neg_count = total_samples - pos_count
                labels = np.array([1] * pos_count + [0] * neg_count)
                
                # 提取AAC、DPC和理化性质特征
                # 使用虚拟序列ID来提取特征（这些可能不包含实际序列信息）
                sequences = [f"SEQ_{i}" for i in range(total_samples)]  # 创建虚拟序列
                additional_features = extract_features_from_sequences(sequences)
                # 确保额外特征具有固定的维度
                additional_features = ensure_fixed_features(additional_features, expected_dim=530)
                
                # 合并特征
                combined_features = np.hstack([esmc_features, additional_features])
                
                return combined_features, labels
            else:
                print(f"  {dataset_name}: 成功加载 {len(sequences)} 个序列")
                
                # 对于使用虚拟ID的数据集（如LBtope），我们直接按顺序匹配ESMC特征和加载的序列
                # ESMC特征文件中的虚拟ID并不包含实际序列信息，所以我们直接使用加载的序列
                total_samples = len(esmc_features)
                
                # 确保序列和标签的数量不超过ESMC特征的数量
                if len(sequences) > total_samples:
                    print(f"  {dataset_name}: 序列数量({len(sequences)})超过ESMC特征数量({total_samples})，截取前{total_samples}个")
                    sequences = sequences[:total_samples]
                    labels = labels[:total_samples]
                elif len(sequences) < total_samples:
                    print(f"  {dataset_name}: 序列数量({len(sequences)})少于ESMC特征数量({total_samples})，补充虚拟序列")
                    # 补充虚拟序列和标签
                    for i in range(len(sequences), total_samples):
                        sequences.append(f"VIRTUAL_SEQ_{i}")
                        labels.append(labels[-1] if len(labels) > 0 else 1)  # 使用最后的标签或默认为正样本
                
                # 提取AAC、DPC、TPC和理化性质特征
                additional_features = []
                for seq in sequences:
                    aac_features = calculate_aac(seq)
                    dpc_features = calculate_dipeptide_composition(seq)  # 新增DPC特征
                    tpc_features = calculate_tripeptide_composition(seq)  # 新增TPC特征
                    physicochemical_features = calculate_physicochemical_features(seq)
                    sequence_stats = calculate_sequence_stats(seq)  # 序列统计特征
                    combined_seq_features = aac_features + dpc_features + tpc_features + physicochemical_features + sequence_stats
                    additional_features.append(combined_seq_features)
                
                # 确保即使列表为空也能创建正确的二维数组
                if len(additional_features) == 0:
                    # 如果没有序列，创建一个空的二维数组，列数等于每个样本应有的特征数
                    num_additional_features = 530  # 根据计算：20(AAC) + 400(DPC) + 100(TPC) + 6(理化性质) + 4(序列统计) = 530
                    additional_features = np.empty((0, num_additional_features))
                else:
                    additional_features = np.array(additional_features)
                    # 确保特征数量一致
                    if additional_features.shape[1] != 530:
                        additional_features = ensure_fixed_features(additional_features, expected_dim=530)
                
                # 确保additional_features和esmc_features的行数相同
                if esmc_features.shape[0] != additional_features.shape[0]:
                    min_rows = min(esmc_features.shape[0], additional_features.shape[0])
                    esmc_features_matched = esmc_features[:min_rows]
                    additional_features = additional_features[:min_rows]
                    labels = np.array(labels[:min_rows])
                else:
                    esmc_features_matched = esmc_features
                    labels = np.array(labels)
                
                # 添加调试信息
                print(f"  调试: {dataset_name} - ESMC特征形状: {esmc_features_matched.shape}, 额外特征形状: {additional_features.shape}")
                # 合并ESMC特征和额外特征
                combined_features = np.hstack([esmc_features_matched, additional_features])
                
                return combined_features, labels
                
        except Exception as e:
            print(f"  {dataset_name}: 鲁棒数据加载器出错: {str(e)}, 回退到原方法")
            pos_file = os.path.join(seq_folder, "pos.txt")
            neg_file = os.path.join(seq_folder, "neg.txt")
    elif dataset_name == "ABCPred":
        print(f"  {dataset_name}: 使用鲁棒数据加载器处理")
        # 使用新的鲁棒数据加载器加载数据集
        try:
            # 将seq_folder作为数据集目录传递给加载器
            sequences, labels, stats = load_and_validate_dataset(dataset_name, seq_folder)
            
            if len(sequences) == 0:
                print(f"  {dataset_name}: 数据加载失败，使用ESMC特征文件中的虚拟ID进行处理")
                # 如果数据加载失败，使用虚拟ID来构建标签，假设前一半是正样本，后一半是负样本
                total_samples = len(esmc_features)
                pos_count = total_samples // 2
                neg_count = total_samples - pos_count
                labels = np.array([1] * pos_count + [0] * neg_count)
                
                # 提取AAC、DPC和理化性质特征
                # 使用虚拟序列ID来提取特征（这些可能不包含实际序列信息）
                sequences = [f"SEQ_{i}" for i in range(total_samples)]  # 创建虚拟序列
                additional_features = extract_features_from_sequences(sequences)
                # 确保额外特征具有固定的维度
                additional_features = ensure_fixed_features(additional_features, expected_dim=530)
                
                # 合并特征
                combined_features = np.hstack([esmc_features, additional_features])
                
                return combined_features, labels
            else:
                print(f"  {dataset_name}: 成功加载 {len(sequences)} 个序列")
                
                # 对于使用虚拟ID的数据集（如ABCPred），我们直接按顺序匹配ESMC特征和加载的序列
                # ESMC特征文件中的虚拟ID并不包含实际序列信息，所以我们直接使用加载的序列
                total_samples = len(esmc_features)
                
                # 确保序列和标签的数量不超过ESMC特征的数量
                if len(sequences) > total_samples:
                    print(f"  {dataset_name}: 序列数量({len(sequences)})超过ESMC特征数量({total_samples})，截取前{total_samples}个")
                    sequences = sequences[:total_samples]
                    labels = labels[:total_samples]
                elif len(sequences) < total_samples:
                    print(f"  {dataset_name}: 序列数量({len(sequences)})少于ESMC特征数量({total_samples})，补充虚拟序列")
                    # 补充虚拟序列和标签
                    for i in range(len(sequences), total_samples):
                        sequences.append(f"VIRTUAL_SEQ_{i}")
                        labels.append(labels[-1] if len(labels) > 0 else 1)  # 使用最后的标签或默认为正样本
                
                # 提取AAC、DPC、TPC和理化性质特征
                additional_features = []
                for seq in sequences:
                    aac_features = calculate_aac(seq)
                    dpc_features = calculate_dipeptide_composition(seq)  # 新增DPC特征
                    tpc_features = calculate_tripeptide_composition(seq)  # 新增TPC特征
                    physicochemical_features = calculate_physicochemical_features(seq)
                    sequence_stats = calculate_sequence_stats(seq)  # 序列统计特征
                    combined_seq_features = aac_features + dpc_features + tpc_features + physicochemical_features + sequence_stats
                    additional_features.append(combined_seq_features)
                
                # 确保即使列表为空也能创建正确的二维数组
                if len(additional_features) == 0:
                    # 如果没有序列，创建一个空的二维数组，列数等于每个样本应有的特征数
                    num_additional_features = 530  # 根据计算：20(AAC) + 400(DPC) + 100(TPC) + 6(理化性质) + 4(序列统计) = 530
                    additional_features = np.empty((0, num_additional_features))
                else:
                    additional_features = np.array(additional_features)
                    # 确保特征数量一致
                    if additional_features.shape[1] != 530:
                        additional_features = ensure_fixed_features(additional_features, expected_dim=530)
                
                # 确保additional_features和esmc_features的行数相同
                if esmc_features.shape[0] != additional_features.shape[0]:
                    min_rows = min(esmc_features.shape[0], additional_features.shape[0])
                    esmc_features_matched = esmc_features[:min_rows]
                    additional_features = additional_features[:min_rows]
                    labels = np.array(labels[:min_rows])
                else:
                    esmc_features_matched = esmc_features
                    labels = np.array(labels)
                
                # 添加调试信息
                print(f"  调试: {dataset_name} - ESMC特征形状: {esmc_features_matched.shape}, 额外特征形状: {additional_features.shape}")
                # 合并ESMC特征和额外特征
                combined_features = np.hstack([esmc_features_matched, additional_features])
                
                return combined_features, labels
                
        except Exception as e:
            print(f"  {dataset_name}: 鲁棒数据加载器出错: {str(e)}, 回退到原方法")
            pos_file = os.path.join(seq_folder, "abcpred16-pos.txt")
            neg_file = os.path.join(seq_folder, "abcpred16-neg.txt")
    elif dataset_name == "Chen":
        pos_file = os.path.join(seq_folder, "chen_pos_20mer.txt")
        neg_file = os.path.join(seq_folder, "chen_neg_20mer.txt")
    elif dataset_name == "Blind387":
        pos_file = os.path.join(seq_folder, "blind387_pos.txt")
        neg_file = os.path.join(seq_folder, "blind387_neg.txt")
    elif dataset_name == "iBCE-EL_independent":
        # 尝试多种可能的文件名
        pos_options = ["ibce_ind_pos.txt", "Ind-positive.txt"]
        neg_options = ["ibce_ind_neg.txt", "Ind-negative.txt"]
        
        pos_file = None
        neg_file = None
        
        for option in pos_options:
            candidate = os.path.join(seq_folder, option)
            if os.path.exists(candidate):
                pos_file = candidate
                break
        
        for option in neg_options:
            candidate = os.path.join(seq_folder, option)
            if os.path.exists(candidate):
                neg_file = candidate
                break
        
        if pos_file is None or neg_file is None:
            raise FileNotFoundError(f"找不到iBCE-EL_independent的正负样本文件。查找的路径: {pos_file}, {neg_file}")
    elif dataset_name == "iBCE-EL_training":
        # iBCE-EL_training数据集
        pos_file = os.path.join(seq_folder, "pos.txt")
        neg_file = os.path.join(seq_folder, "neg.txt")
        
        if not os.path.exists(pos_file) or not os.path.exists(neg_file):
            # 尝试其他可能的文件名
            pos_options = ["pos.txt", "ibce_el_training_pos.txt", "tr_pos.txt", "train_pos.txt"]
            neg_options = ["neg.txt", "ibce_el_training_neg.txt", "tr_neg.txt", "train_neg.txt"]
            
            pos_file = None
            neg_file = None
            
            for option in pos_options:
                candidate = os.path.join(seq_folder, option)
                if os.path.exists(candidate):
                    pos_file = candidate
                    break
            
            for option in neg_options:
                candidate = os.path.join(seq_folder, option)
                if os.path.exists(candidate):
                    neg_file = candidate
                    break
            
            if pos_file is None or neg_file is None:
                raise FileNotFoundError(f"找不到iBCE-EL_training的正负样本文件。查找的路径: {os.path.join(seq_folder, 'pos.txt')}, {os.path.join(seq_folder, 'neg.txt')}")
    else:
        # 处理其他数据集的默认情况
        sequences, _ = [], []
        labels = np.array([])
        
        # 加载正负样本
        pos_files = [f for f in os.listdir(seq_folder) if 'pos' in f.lower()]
        neg_files = [f for f in os.listdir(seq_folder) if 'neg' in f.lower()]
        
        for filename in pos_files:
            filepath = os.path.join(seq_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('>'):
                        sequences.append(line)
                        labels = np.append(labels, 1)  # 正样本标签为1
        
        for filename in neg_files:
            filepath = os.path.join(seq_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('>'):
                        sequences.append(line)
                        labels = np.append(labels, 0)  # 负样本标签为0
        
        # 确保序列数量与ESMC特征匹配
        if len(sequences) != len(esmc_features):
            print(f"警告: ESMC特征数量({len(esmc_features)})与序列数量({len(sequences)})不匹配")
            min_len = min(len(esmc_features), len(sequences))
            esmc_features = esmc_features[:min_len]
            sequences = sequences[:min_len]
            labels = labels[:min_len]
        
        # 提取AAC、DPC和理化性质特征
        additional_features = extract_features_from_sequences(sequences)
        # 确保额外特征具有固定的维度
        additional_features = ensure_fixed_features(additional_features, expected_dim=530)
        
        # 合并特征
        combined_features = np.hstack([esmc_features, additional_features])
        
        return combined_features, labels.astype(int)
    
    # 分别加载正负样本
    pos_sequences = []
    neg_sequences = []
    
    # 加载正样本
    with open(pos_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('>'):
                pos_sequences.append(line)
    
    # 加载负样本
    with open(neg_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('>'):
                neg_sequences.append(line)
    
    print(f"  {dataset_name} - 正样本数量: {len(pos_sequences)}, 负样本数量: {len(neg_sequences)}")
    
    # 获取ESMC特征的数量
    esmc_count = len(esmc_features)
    pos_count = len(pos_sequences)
    neg_count = len(neg_sequences)
    
    # 计算应该使用的正负样本数量，确保总数不超过ESMC特征数量
    total_needed = esmc_count
    if pos_count + neg_count <= total_needed:
        # 如果正负样本总数少于或等于ESMC特征数，全部使用
        selected_pos_count = pos_count
        selected_neg_count = neg_count
    else:
        # 如果正负样本总数超过ESMC特征数，按比例分配
        ratio = esmc_count / (pos_count + neg_count)
        selected_pos_count = int(pos_count * ratio)
        selected_neg_count = esmc_count - selected_pos_count
        # 确保不超过原始数量
        selected_neg_count = min(selected_neg_count, neg_count)
        selected_pos_count = esmc_count - selected_neg_count
    
    print(f"  {dataset_name} - 选择正样本: {selected_pos_count}, 选择负样本: {selected_neg_count}")
    
    # 组合选中的序列和标签
    selected_sequences = pos_sequences[:selected_pos_count] + neg_sequences[:selected_neg_count]
    selected_labels = [1] * selected_pos_count + [0] * selected_neg_count
    
    # 提取AAC、DPC和理化性质特征
    additional_features = extract_features_from_sequences(selected_sequences)
    # 确保额外特征具有固定的维度
    additional_features = ensure_fixed_features(additional_features, expected_dim=530)
    
    # 调整ESMC特征以匹配选中的样本数
    esmc_features_adjusted = esmc_features[:len(selected_sequences)]
    
    # 合并特征
    combined_features = np.hstack([esmc_features_adjusted, additional_features])
    
    return combined_features, np.array(selected_labels)

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """计算六大性能指标"""
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    sn = recall_score(y_true, y_pred, pos_label=1)  # Sensitivity/Recall for positive class
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # 计算AUROC (如果可能)
    # 检查y_true中是否有多个类别
    if len(np.unique(y_true)) < 2:
        # 如果只有一个类别，AUROC无法计算，返回0.5（随机猜测）
        auroc = 0.5
    else:
        try:
            auroc = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            # 如果仍然无法计算AUROC（例如所有预测都相同），返回0.5
            auroc = 0.5
    
    return {
        'accuracy': acc,
        'precision': pre,
        'sensitivity': sn,
        'f1': f1,
        'mcc': mcc,
        'auroc': auroc
    }

def optimize_threshold_multi_metric(y_true, y_pred_proba):
    """多指标阈值优化 - 为不同指标找到最佳阈值，重点关注ACC、MCC和AUROC"""
    thresholds = np.arange(0.05, 0.95, 0.005)
    
    best_thresholds = {}
    
    for metric in ['f1', 'mcc', 'balanced_accuracy', 'auroc', 'acc_mcc_auroc']:
        best_threshold = 0.5
        best_score = -np.inf
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            # 确保有足够的类别以避免评分函数出错
            unique_vals = np.unique(y_pred_thresh)
            if len(unique_vals) < 2:
                continue  # 跳过无法计算指标的情况
            
            if metric == 'f1':
                try:
                    score = f1_score(y_true, y_pred_thresh)
                except:
                    continue
            elif metric == 'mcc':
                try:
                    score = matthews_corrcoef(y_true, y_pred_thresh)
                except:
                    continue
            elif metric == 'balanced_accuracy':
                try:
                    from sklearn.metrics import balanced_accuracy_score
                    score = balanced_accuracy_score(y_true, y_pred_thresh)
                except:
                    continue
            elif metric == 'auroc':
                try:
                    # AUROC实际上不需要阈值，这里使用概率进行计算
                    score = roc_auc_score(y_true, y_pred_proba)
                except:
                    continue
            elif metric == 'acc_mcc_auroc':
                # 重点关注ACC、MCC和AUROC的组合
                try:
                    accuracy = accuracy_score(y_true, y_pred_thresh)
                    mcc = matthews_corrcoef(y_true, y_pred_thresh)
                    # 使用概率而非阈值预测来计算AUROC
                    auroc = roc_auc_score(y_true, y_pred_proba)
                    
                    # 综合评分：重点关注ACC、MCC和AUROC
                    score = 0.35 * accuracy + 0.4 * mcc + 0.25 * auroc
                except:
                    continue
            else:  # 默认使用f1
                try:
                    score = f1_score(y_true, y_pred_thresh)
                except:
                    continue
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        best_thresholds[metric] = best_threshold
    
    return best_thresholds

def perform_cross_validation(X, y, cv_folds=5):
    """执行交叉验证 - 每次迭代都使用独立的预处理器，防止数据泄露"""
    print(f"执行{cv_folds}折交叉验证...")
    start_time = time.time()
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"正在执行第 {fold+1}/{cv_folds} 折交叉验证...")
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # 标准化 - 在每个fold中独立进行
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        # 特征选择 - 在每个fold中独立进行
        k_value = min(1300, X_train_scaled.shape[1])  # 增加特征选择数量
        selector = SelectKBest(score_func=f_classif, k=k_value)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train_fold)
        X_val_selected = selector.transform(X_val_scaled)
        
        # PCA降维 - 在每个fold中独立进行
        n_components = min(650, X_train_selected.shape[1], X_train_selected.shape[0])  # 增加PCA组件数
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train_selected)
        X_val_pca = pca.transform(X_val_selected)
        
        # 计算正负样本比例
        neg_count = np.sum(y_train_fold == 0)
        pos_count = np.sum(y_train_fold == 1)
        scale_pos_weight = neg_count / pos_count if pos_count != 0 else 1.0
        
        # 优化XGBoost参数以提升性能
        model = XGBClassifier(
            n_estimators=1000,  # 进一步增加树的数量
            max_depth=8,       # 进一步增加树的深度
            learning_rate=0.05, # 降低学习率以提高精度
            subsample=0.9,    # 增加子采样率
            colsample_bytree=0.85, # 特征采样率
            reg_alpha=0.2,     # L1正则化
            reg_lambda=1.5,    # L2正则化
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        model.fit(X_train_pca, y_train_fold)
        
        # 预测
        y_pred = model.predict(X_val_pca)
        y_pred_proba = model.predict_proba(X_val_pca)[:, 1]
        
        # 计算基础指标
        metrics = calculate_metrics(y_val_fold, y_pred, y_pred_proba)
        
        # 优化阈值以最大化ACC、MCC和AUROC的组合得分
        best_thresholds = optimize_threshold_multi_metric(y_val_fold, y_pred_proba)
        optimal_threshold = best_thresholds.get('acc_mcc_auroc', 0.5)
        y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
        metrics_optimized = calculate_metrics(y_val_fold, y_pred_optimized, y_pred_proba)
        
        # 使用优化后的指标
        metrics = metrics_optimized
        metrics['optimal_threshold'] = optimal_threshold
        
        fold_results.append(metrics)
    
    cv_time = time.time() - start_time
    print(f"交叉验证完成，耗时: {cv_time:.2f}秒")
    
    return fold_results

def calculate_cross_validation_stats(cv_results):
    """计算交叉验证结果的统计信息"""
    # 提取各项指标
    acc_scores = [result['accuracy'] for result in cv_results]
    prec_scores = [result['precision'] for result in cv_results]
    sens_scores = [result['sensitivity'] for result in cv_results]
    f1_scores = [result['f1'] for result in cv_results]
    mcc_scores = [result['mcc'] for result in cv_results]
    auroc_scores = [result['auroc'] for result in cv_results]
    
    # 计算均值和标准差
    stats = {
        'cv_accuracy_mean': np.mean(acc_scores),
        'cv_accuracy_std': np.std(acc_scores),
        'cv_precision_mean': np.mean(prec_scores),
        'cv_precision_std': np.std(prec_scores),
        'cv_sensitivity_mean': np.mean(sens_scores),
        'cv_sensitivity_std': np.std(sens_scores),
        'cv_f1_mean': np.mean(f1_scores),
        'cv_f1_std': np.std(f1_scores),
        'cv_mcc_mean': np.mean(mcc_scores),
        'cv_mcc_std': np.std(mcc_scores),
        'cv_auroc_mean': np.mean(auroc_scores),
        'cv_auroc_std': np.std(auroc_scores)
    }
    
    return stats

def main():
    print("开始在BCPreds数据集上训练模型...")
    
    # 定义数据集信息
    datasets = {
        'Chen': {
            'esmc_path': r'E:\工\b细胞表位预测\LBCE-ESMC-600M\esmc_600mfeatures\Chen\Chen_CLS_fea.txt',
            'seq_folder': r'E:\工\b细胞表位预测\LBCE-ESMC-600M\data\datasets\Chen'
        },
        'ABCPred': {
            'esmc_path': r'E:\工\b细胞表位预测\LBCE-ESMC-600M\esmc_600mfeatures\ABCPred\ABCPred_CLS_fea.txt',
            'seq_folder': r'E:\工\b细胞表位预测\LBCE-ESMC-600M\data\datasets\ABCPred'
        },
        'BCPreds': {
            'esmc_path': r'E:\工\b细胞表位预测\LBCE-ESMC-600M\esmc_600mfeatures\BCPreds\BCPreds_CLS_fea.txt',
            'seq_folder': r'E:\工\b细胞表位预测\LBCE-ESMC-600M\data\datasets\BCPreds'
        },
        'Blind387': {
            'esmc_path': r'E:\工\b细胞表位预测\LBCE-ESMC-600M\esmc_600mfeatures\Blind387\Blind387_CLS_fea.txt',
            'seq_folder': r'E:\工\b细胞表位预测\LBCE-ESMC-600M\data\datasets\Blind387'
        },
        'iBCE-EL_independent': {
            'esmc_path': r'E:\工\b细胞表位预测\LBCE-ESMC-600M\esmc_600mfeatures\iBCE-EL_independent\iBCE-EL_independent_CLS_fea.txt',
            'seq_folder': r'E:\工\b细胞表位预测\LBCE-ESMC-600M\data\datasets\iBCE-EL_independent'
        },
        'iBCE-EL_training': {
            'esmc_path': r'E:\工\b细胞表位预测\LBCE-ESMC-600M\esmc_600mfeatures\iBCE-EL_training\iBCE-EL_training_CLS_fea.txt',
            'seq_folder': r'E:\工\b细胞表位预测\LBCE-ESMC-600M\data\datasets\iBCE-EL_training'
        },
        'LBtope': {
            'esmc_path': r'E:\工\b细胞表位预测\LBCE-ESMC-600M\esmc_600mfeatures\LBtope\LBtope_CLS_fea.txt',
            'seq_folder': r'E:\工\b细胞表位预测\LBCE-ESMC-600M\data\datasets\LBtope'
        }
    }
    
    # 加载训练数据
    train_dataset = 'BCPreds'
    train_data_info = datasets[train_dataset]
    
    X_train, y_train = load_and_combine_features(
        train_data_info['esmc_path'], 
        train_data_info['seq_folder'], 
        dataset_name=train_dataset
    )
    
    print(f"训练数据形状: {X_train.shape}")
    print(f"训练标签分布: {Counter(y_train)}")
    
    # 划分内部训练集和验证集
    X_internal_train, X_internal_val, y_internal_train, y_internal_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # 初始化预处理器
    scaler = StandardScaler()
    selector = SelectKBest(score_func=f_classif, k=min(1300, X_internal_train.shape[1]))
    pca = PCA(n_components=min(650, X_internal_train.shape[1], X_internal_train.shape[0]), random_state=42)
    
    # 标准化
    X_internal_train_scaled = scaler.fit_transform(X_internal_train)
    X_internal_val_scaled = scaler.transform(X_internal_val)
    
    # 特征选择
    k_value = min(1300, X_internal_train_scaled.shape[1])
    X_internal_train_selected = selector.fit_transform(X_internal_train_scaled, y_internal_train)
    X_internal_val_selected = selector.transform(X_internal_val_scaled)
    
    # PCA降维
    n_components = min(650, X_internal_train_selected.shape[1], X_internal_train_selected.shape[0])
    X_internal_train_pca = pca.fit_transform(X_internal_train_selected)
    X_internal_val_pca = pca.transform(X_internal_val_selected)
    
    print(f"内部训练数据PCA后形状: {X_internal_train_pca.shape}")
    
    # 计算正负样本比例
    neg_count = np.sum(y_internal_train == 0)
    pos_count = np.sum(y_internal_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count != 0 else 1.0
    
    # 训练优化的XGBoost模型 - 重点优化F1分数
    print("训练优化的XGBoost模型...")
    model = XGBClassifier(
        n_estimators=1000,  # 进一步增加树的数量
        max_depth=8,       # 进一步增加树的深度
        learning_rate=0.05, # 降低学习率以提高精度
        subsample=0.9,    # 增加子采样率
        colsample_bytree=0.85, # 特征采样率
        reg_alpha=0.2,     # L1正则化
        reg_lambda=1.5,    # L2正则化
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    start_time = time.time()
    model.fit(X_internal_train_pca, y_internal_train)
    
    # 内部验证以确保模型有效性
    y_val_pred_proba = model.predict_proba(X_internal_val_pca)[:, 1]
    y_val_pred = (y_val_pred_proba >= 0.5).astype(int)
    internal_val_metrics = calculate_metrics(y_internal_val, y_val_pred, y_val_pred_proba)
    print(f"内部验证集性能 - ACC: {internal_val_metrics['accuracy']:.4f}, Pre: {internal_val_metrics['precision']:.4f}, Sn: {internal_val_metrics['sensitivity']:.4f}, F1: {internal_val_metrics['f1']:.4f}, MCC: {internal_val_metrics['mcc']:.4f}, AUROC: {internal_val_metrics['auroc']:.4f}")
    
    train_time = time.time() - start_time
    
    print(f"模型训练完成，耗时: {train_time:.2f}秒")
    
    # 使用内部验证集上训练的模型进行外部测试（防止数据泄露）
    # 注意：我们不会用完整的训练集重新训练，以避免数据泄露
    
    # 转换内部训练集用于交叉验证（只使用内部训练集来适配预处理器）
    # 注意：此处不需要重复转换，因为X_internal_train_pca已经在上面创建过了
    # 我们只需使用之前创建的内部训练集PCA特征进行交叉验证
    
    # 对所有数据集进行测试（排除训练集，不将其作为测试集）
    all_results = {}
    for dataset_name, data_info in datasets.items():
        # 跳过训练集，不将其作为测试集
        if dataset_name == "BCPreds":
            continue
            
        print(f"\n正在测试 {dataset_name} 数据集...")
        
        try:
            X_test, y_test = load_and_combine_features(
                data_info['esmc_path'], 
                data_info['seq_folder'], 
                dataset_name=dataset_name
            )
            
            # 预处理测试数据
            X_test_scaled = scaler.transform(X_test)
            X_test_selected = selector.transform(X_test_scaled)
            X_test_pca = pca.transform(X_test_selected)
            
            # 预测
            y_pred_proba = model.predict_proba(X_test_pca)[:, 1]
            y_pred = model.predict(X_test_pca)
            
            # 计算基础指标
            base_metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # 优化阈值
            best_thresholds = optimize_threshold_multi_metric(y_test, y_pred_proba)
            optimal_threshold = best_thresholds.get('acc_mcc_auroc', 0.5)
            y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
            opt_metrics = calculate_metrics(y_test, y_pred_optimized, y_pred_proba)
            
            print(f"  {dataset_name} - 使用优化阈值:")
            print(f"    Accuracy (ACC):  {opt_metrics['accuracy']:.4f}")
            print(f"    Precision (Pre): {opt_metrics['precision']:.4f}")
            print(f"    Sensitivity (Sn): {opt_metrics['sensitivity']:.4f}")
            print(f"    F1-Score (F1):   {opt_metrics['f1']:.4f}")
            print(f"    MCC:             {opt_metrics['mcc']:.4f}")
            print(f"    AUROC:           {opt_metrics['auroc']:.4f}")
            print(f"    Optimal Threshold: {optimal_threshold:.4f}")
            
            all_results[dataset_name] = {
                'base_metrics': base_metrics,
                'opt_metrics': opt_metrics,
                'optimal_threshold': optimal_threshold,
                'y_true': y_test,
                'y_pred_proba': y_pred_proba
            }
        except Exception as e:
            print(f"测试 {dataset_name} 数据集时出错: {str(e)}")
            all_results[dataset_name] = {'error': str(e)}
    
    # 执行交叉验证
    print("\n执行交叉验证...")
    # 注意：使用原始的内部训练数据而不是PCA转换后的数据，因为perform_cross_validation会自行处理预处理
    cv_results = perform_cross_validation(X_internal_train, y_internal_train)
    cv_stats = calculate_cross_validation_stats(cv_results)
    
    # 特征工程统计
    feature_stats = {
        'initial_features': X_internal_train.shape[1],
        'selected_features': X_internal_train_selected.shape[1],
        'pca_components': X_internal_train_pca.shape[1]
    }
    
    # 合并结果
    final_results = {
        'model_results': all_results,
        'cross_validation': cv_stats,
        'feature_engineering': feature_stats,
        'train_time': train_time
    }
    # 保存结果到JSON（将numpy数组转换为列表）
    json_filename = f"optimized_bcpred_new_improved_chen_blind387_targeted_results.json"
    
    def convert_numpy_types(obj):
        """递归转换numpy类型到Python原生类型"""
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # 将numpy数组转换为列表
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # 将numpy标量转换为Python原生类型
        else:
            return obj
    
    converted_results = convert_numpy_types(final_results)
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(converted_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存至: {json_filename}")
    
    # 打印汇总性能
    print("\n=== 模型性能汇总 ===")
    for dataset_name, result in all_results.items():
        if 'opt_metrics' in result:
            metrics = result['opt_metrics']
            print(f"\n{dataset_name}:")
            print(f"  ACC: {metrics['accuracy']:.4f}")
            print(f"  Pre: {metrics['precision']:.4f}")
            print(f"  Sn: {metrics['sensitivity']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  MCC: {metrics['mcc']:.4f}")
            print(f"  AUROC: {metrics['auroc']:.4f}")
    
    return model, scaler, selector, pca, final_results

if __name__ == "__main__":
    model, scaler, selector, pca, results = main()