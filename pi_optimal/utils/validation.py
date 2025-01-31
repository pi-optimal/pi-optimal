import os

def validate_path(path: str) -> None:
    """Path validation utilities"""
    if not isinstance(path, str):
        raise TypeError("Path must be a string")
    if path.strip() == '':
        raise ValueError("Path cannot be empty")

def validate_agent_directory(path: str) -> None:
    """Agent directory validation utilities"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Agent directory not found at: {path}")
    if not os.path.exists(f"{path}/agent_config.json"):
        raise FileNotFoundError(f"Agent configuration file not found at: {path}/agent_config.json")
