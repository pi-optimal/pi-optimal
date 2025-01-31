import numpy as np
import json
import joblib
from typing import Any, Dict
import os

def serialize_processors(config: dict, path: str) -> dict:
    """
    Converts processor objects to their string representation and saves them.

    Args:
        config (dict): The configuration dictionary containing processor objects.
        path (str): The directory path where processors will be saved.

    Returns:
        dict: The configuration dictionary with processor paths.
    """
    serialized_config = {}
    processors_dict = {}
    
    def _process_dict(d, processors):
        processed = {}
        for key, value in d.items():
            if isinstance(value, dict):
                processed[key] = _process_dict(value, processors)
            elif key == 'processor' and hasattr(value, '__class__'):
                processor_key = f"{key}_{d['name']}"
                processors[processor_key] = value
                processed[key] = {
                    'class': value.__class__.__name__,
                    'module': value.__class__.__module__,
                    'key': processor_key
                }
            else:
                processed[key] = value
        return processed

    # Process the config and collect all processors
    serialized_config = _process_dict(config, processors_dict)
    
    # Save all processors in a single file
    processors_path = os.path.join(path, "processors.pkl")
    os.makedirs(os.path.dirname(processors_path), exist_ok=True)
    joblib.dump(processors_dict, processors_path)
    
    # Store the path to all processors
    serialized_config['_processors_path'] = "processors.pkl"
    return serialized_config

def deserialize_processors(config: dict, base_path: str) -> dict:
    """
    Converts processor strings back to objects by loading them.

    Args:
        config (dict): The configuration dictionary containing processor paths.
        base_path (str): The base directory path where processors are saved.

    Returns:
        dict: The configuration dictionary with processor objects.
    """
    # Load all processors from the single file
    processors_path = os.path.join(base_path, config.pop('_processors_path', None))
    all_processors = joblib.load(processors_path) if processors_path else {}
    
    def _process_dict(d):
        processed = {}
        for key, value in d.items():
            if isinstance(value, dict):
                if key == 'processor' and 'key' in value:
                    processed[key] = all_processors[value['key']]
                else:
                    processed[key] = _process_dict(value)
            else:
                processed[key] = value
        return processed

    deserialized_config = _process_dict(config)
    
    # Convert string keys to integers
    if 'states' in deserialized_config:
        deserialized_config['states'] = {int(k): v for k, v in deserialized_config['states'].items()}    
    if 'actions' in deserialized_config:
        deserialized_config['actions'] = {int(k): v for k, v in deserialized_config['actions'].items()}
    
    return deserialized_config

def serialize_policy_dict(policy_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize policy dictionary, handling special cases.

    Args:
        policy_dict (Dict[str, Any]): The policy dictionary to serialize.

    Returns:
        Dict[str, Any]: The serialized policy dictionary.
    """
    serialized_dict = {}
    for key, value in policy_dict.items():
        if value is None:
            serialized_dict[key] = None
        elif isinstance(value, dict):
            serialized_dict[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        elif isinstance(value, np.ndarray):
            serialized_dict[key] = value.tolist()
        else:
            serialized_dict[key] = value
    return serialized_dict

def deserialize_policy_dict(policy_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deserialize policy dictionary, converting lists back to numpy arrays.

    Args:
        policy_dict (Dict[str, Any]): The policy dictionary to deserialize.

    Returns:
        Dict[str, Any]: The deserialized policy dictionary.
    """
    deserialized_dict = {}
    for key, value in policy_dict.items():
        if value is None:
            deserialized_dict[key] = None
        elif isinstance(value, dict):
            deserialized_dict[key] = {
                k: np.array(v) if isinstance(v, list) else v
                for k, v in value.items()
            }
        elif isinstance(value, list):
            deserialized_dict[key] = np.array(value)
        else:
            deserialized_dict[key] = value
    return deserialized_dict

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for numpy types.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)
