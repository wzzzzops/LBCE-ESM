import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import re

def load_lbtope_data(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Load LBtope dataset from pos.txt and neg.txt files
    Format: One sequence per line, no labels in file (labels derived from filename)
    """
    pos_file = os.path.join(data_dir, 'pos.txt')
    neg_file = os.path.join(data_dir, 'neg.txt')
    
    sequences = []
    labels = []
    
    # Load positive sequences
    if os.path.exists(pos_file):
        with open(pos_file, 'r') as f:
            for line in f:
                seq = line.strip()
                if seq:  # Skip empty lines
                    sequences.append(seq)
                    labels.append(1)
    
    # Load negative sequences
    if os.path.exists(neg_file):
        with open(neg_file, 'r') as f:
            for line in f:
                seq = line.strip()
                if seq:  # Skip empty lines
                    sequences.append(seq)
                    labels.append(0)
    
    return sequences, labels

def load_abcpred_data(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Load ABCPred dataset from abcpred16-pos.txt and abcpred16-neg.txt files
    Format: "sequence<TAB>label" per line
    """
    pos_file = os.path.join(data_dir, 'abcpred16-pos.txt')
    neg_file = os.path.join(data_dir, 'abcpred16-neg.txt')
    
    sequences = []
    labels = []
    
    # Load positive sequences with labels
    if os.path.exists(pos_file):
        with open(pos_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    seq = parts[0].strip()
                    label = int(parts[1].strip())
                    if seq:
                        sequences.append(seq)
                        labels.append(label)
    
    # Load negative sequences with labels
    if os.path.exists(neg_file):
        with open(neg_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    seq = parts[0].strip()
                    label = int(parts[1].strip())
                    if seq:
                        sequences.append(seq)
                        labels.append(label)
    
    return sequences, labels

def load_blind387_data(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Load Blind387 dataset from blind387_pos.txt and blind387_neg.txt files
    Format: One sequence per line, no labels in file (labels derived from filename)
    """
    pos_file = os.path.join(data_dir, 'blind387_pos.txt')
    neg_file = os.path.join(data_dir, 'blind387_neg.txt')
    
    sequences = []
    labels = []
    
    # Load positive sequences
    if os.path.exists(pos_file):
        with open(pos_file, 'r') as f:
            for line in f:
                seq = line.strip()
                if seq:  # Skip empty lines
                    sequences.append(seq)
                    labels.append(1)
    
    # Load negative sequences
    if os.path.exists(neg_file):
        with open(neg_file, 'r') as f:
            for line in f:
                seq = line.strip()
                if seq:  # Skip empty lines
                    sequences.append(seq)
                    labels.append(0)
    
    return sequences, labels

def load_dataset_by_name(dataset_name: str, data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Generic function to load dataset based on its name
    """
    if dataset_name.lower() == 'lbtope':
        return load_lbtope_data(data_dir)
    elif dataset_name.lower() == 'abcpred':
        return load_abcpred_data(data_dir)
    elif dataset_name.lower() == 'blind387':
        return load_blind387_data(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def validate_sequences(sequences: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
    """
    Validate loaded sequences and remove invalid ones
    """
    valid_sequences = []
    valid_labels = []
    
    for seq, label in zip(sequences, labels):
        # Check if sequence contains only valid amino acids
        if re.match(r'^[ACDEFGHIKLMNPQRSTVWY]+$', seq.upper()):
            valid_sequences.append(seq.upper())  # Normalize to uppercase
            valid_labels.append(label)
        else:
            print(f"Warning: Invalid sequence detected and skipped: {seq}")
    
    return valid_sequences, valid_labels

def get_dataset_statistics(sequences: List[str], labels: List[int], dataset_name: str) -> dict:
    """
    Generate statistics for the loaded dataset
    """
    stats = {
        'dataset_name': dataset_name,
        'total_sequences': len(sequences),
        'positive_count': sum(labels),
        'negative_count': len(labels) - sum(labels),
        'avg_sequence_length': np.mean([len(s) for s in sequences]) if sequences else 0,
        'min_sequence_length': min([len(s) for s in sequences]) if sequences else 0,
        'max_sequence_length': max([len(s) for s in sequences]) if sequences else 0,
        'unique_sequences': len(set(sequences)),
    }
    
    return stats

def load_and_validate_dataset(dataset_name: str, data_dir: str) -> Tuple[List[str], List[int], dict]:
    """
    Main function to load and validate a dataset
    """
    print(f"Loading {dataset_name} dataset from {data_dir}")
    
    # Load raw data
    sequences, labels = load_dataset_by_name(dataset_name, data_dir)
    
    print(f"Raw data loaded: {len(sequences)} sequences")
    
    # Validate sequences
    valid_sequences, valid_labels = validate_sequences(sequences, labels)
    
    print(f"After validation: {len(valid_sequences)} sequences")
    
    # Generate statistics
    stats = get_dataset_statistics(valid_sequences, valid_labels, dataset_name)
    
    print(f"Dataset Statistics:")
    print(f"  Total Sequences: {stats['total_sequences']}")
    print(f"  Positive: {stats['positive_count']}, Negative: {stats['negative_count']}")
    print(f"  Avg Length: {stats['avg_sequence_length']:.2f}")
    print(f"  Unique Sequences: {stats['unique_sequences']}")
    
    return valid_sequences, valid_labels, stats