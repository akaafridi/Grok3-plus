"""
Configuration module for Grok-3+ model
DOI: 10.5281/zenodo.15341810
"""

import os
from pathlib import Path

import yaml


def get_default_config_path():
    """Get the path to the default configuration file."""
    return os.path.join(os.path.dirname(__file__), "default.yaml")


def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file, or None to use default
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = get_default_config_path()
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config
