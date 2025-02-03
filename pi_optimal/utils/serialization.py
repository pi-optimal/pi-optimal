import numpy as np
import json
import pickle
from typing import Any, Dict
import os
import datetime

def serialize_processors(config: dict, path: str) -> dict:
    """Improved processor serialization using pickle and path safety"""
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

    serialized_config = _process_dict(config, processors_dict)
    
    # Safe path handling with pickle
    processors_path = os.path.join(path, "processors.pkl")
    os.makedirs(os.path.dirname(processors_path), exist_ok=True)
    with open(processors_path, 'wb') as f:
        pickle.dump(processors_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    serialized_config['_processors_path'] = "processors.pkl"
    return serialized_config

def deserialize_processors(config: dict, base_path: str) -> dict:
    """Improved processor deserialization with error handling"""
    try:
        processors_path = os.path.join(base_path, config.pop('_processors_path'))
        with open(processors_path, 'rb') as f:
            all_processors = pickle.load(f)
    except (KeyError, FileNotFoundError):
        all_processors = {}

    def _process_dict(d):
        processed = {}
        for key, value in d.items():
            if isinstance(value, dict):
                if key == 'processor' and 'key' in value:
                    processed[key] = all_processors.get(value['key'], None)
                else:
                    processed[key] = _process_dict(value)
            else:
                processed[key] = value
        return processed

    return _process_dict(config)

def serialize_policy_dict(policy_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced policy serialization with logger exclusion and numpy support"""
    serialized_dict = {}
    for key, value in policy_dict.items():
        if key == 'logger':
            continue  # Skip logger serialization
        
        if isinstance(value, dict):
            serialized_dict[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        elif isinstance(value, np.ndarray):
            serialized_dict[key] = value.tolist()
        elif hasattr(value, '__dict__'):
            serialized_dict[key] = serialize_policy_dict(value.__dict__)
        else:
            serialized_dict[key] = value
    return serialized_dict

def deserialize_policy_dict(policy_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced policy deserialization with array restoration"""
    deserialized_dict = {}
    for key, value in policy_dict.items():
        if isinstance(value, dict):
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
    """Extended numpy support with datetime handling"""
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super().default(obj)
