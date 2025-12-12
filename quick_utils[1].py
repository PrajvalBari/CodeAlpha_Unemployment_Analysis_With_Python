
"""quick_utils.py
Small utility functions for the unemployment project.
"""
import pandas as pd
import matplotlib.pyplot as plt

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding='latin1')
