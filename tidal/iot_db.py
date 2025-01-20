# tidal/iot_db.py

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
import h5py
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import threading
from datetime import datetime
import hashlib

@dataclass
class ModelMetadata:
    """Metadata for stored model weights"""
    model_id: str
    creation_date: str
    l_max: int
    m_max: int
    iot_params: Dict[str, float]
    basis_functions: List[Tuple[int, int]]
    performance_metrics: Optional[Dict[str, float]] = None

class IOTWeightMapper:
    """Maps model weights to IOT coordinates for storage"""
    
    def __init__(self, R: float, r: float):
        self.R = R
        self.r = r
        self._lock = threading.Lock()
    
    def coeffs_to_iot_coords(self, coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map complex coefficients to IOT surface coordinates"""
        n_coeffs = len(coeffs)
        
        # Use coefficient magnitudes and phases for mapping
        magnitudes = np.abs(coeffs)
        phases = np.angle(coeffs)
        
        # Normalize to [0, 2π] range
        u = 2 * np.pi * magnitudes / np.max(magnitudes)
        v = phases + np.pi  # Shift to [0, 2π]
        
        return u, v

    def iot_coords_to_coeffs(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Recover complex coefficients from IOT coordinates"""
        # Denormalize magnitudes
        magnitudes = (u * np.max(u)) / (2 * np.pi)
        # Recover phases
        phases = v - np.pi
        
        # Reconstruct complex coefficients
        return magnitudes * np.exp(1j * phases)

class DatabaseError(Exception):
    """Base exception for database operations"""
    pass

class ModelNotFoundError(DatabaseError):
    """Raised when a model cannot be found in the database"""
    pass

class VersionNotFoundError(DatabaseError):
    """Raised when a specific model version cannot be found"""
    pass

class IOTDatabase:
    """Database for storing TIDAL model weights using IOT geometry"""
    
    def __init__(self, base_path: str = "tidal_weights"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.mapper = IOTWeightMapper(R=137.035999/136, r=0.5)
        self._lock = threading.Lock()
        
        # Create and maintain an index file for faster lookups
        self._index_path = self.base_path / "model_index.json"
        self._initialize_index()
        
    def _generate_model_id(self, coeffs: np.ndarray) -> str:
        """Generate unique model ID based on coefficients and timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        coeff_hash = hashlib.sha256(coeffs.tobytes()).hexdigest()[:8]
        return f"model_{timestamp}_{coeff_hash}"
    
    def _save_metadata(self, metadata: ModelMetadata):
        """Save model metadata to JSON file"""
        metadata_path = self.base_path / f"{metadata.model_id}_metadata.json"
        with self._lock:
            with open(metadata_path, 'w') as f:
                json.dump({
                    'model_id': metadata.model_id,
                    'creation_date': metadata.creation_date,
                    'l_max': metadata.l_max,
                    'm_max': metadata.m_max,
                    'iot_params': metadata.iot_params,
                    'basis_functions': metadata.basis_functions,
                    'performance_metrics': metadata.performance_metrics
                }, f, indent=2)
    
    def save_model(self, wave_function: Any, performance_metrics: Optional[Dict[str, float]] = None,
                   version: Optional[str] = None, compress: bool = True) -> str:
        """Save model weights and metadata to database with versioning and compression"""
        coeffs = wave_function.coefficients
        model_id = self._generate_model_id(coeffs)
        
        # Map coefficients to IOT coordinates
        u, v = self.mapper.coeffs_to_iot_coords(coeffs)
        
        # Prepare metadata with version info
        metadata = ModelMetadata(
            model_id=model_id,
            creation_date=datetime.now().isoformat(),
            l_max=max(bf.l for bf in wave_function.basis_functions),
            m_max=max(abs(bf.m) for bf in wave_function.basis_functions),
            iot_params={'R': wave_function.iot.R, 'r': wave_function.iot.r},
            basis_functions=[(bf.l, bf.m) for bf in wave_function.basis_functions],
            performance_metrics=performance_metrics
        )
        
        # Save weights and coordinates with compression
        weights_path = self.base_path / f"{model_id}_weights.h5"
        with self._lock:
            compression = 'gzip' if compress else None
            compression_opts = 9 if compress else None
            
            with h5py.File(weights_path, 'w') as f:
                # Create a versioned group
                version_group = f.create_group(version or 'v1')
                
                # Save arrays with compression
                version_group.create_dataset('u_coords', data=u, 
                                         compression=compression,
                                         compression_opts=compression_opts)
                version_group.create_dataset('v_coords', data=v,
                                         compression=compression,
                                         compression_opts=compression_opts)
                version_group.create_dataset('coefficients_real', data=coeffs.real,
                                         compression=compression,
                                         compression_opts=compression_opts)
                version_group.create_dataset('coefficients_imag', data=coeffs.imag,
                                         compression=compression,
                                         compression_opts=compression_opts)
                
                # Add metadata attributes
                version_group.attrs['timestamp'] = datetime.now().isoformat()
                version_group.attrs['version'] = version or 'v1'
        
        # Save metadata
        self._save_metadata(metadata)
        
        return model_id
    
    def _initialize_index(self):
        """Initialize or load the model index"""
        if not self._index_path.exists():
            with self._lock:
                with open(self._index_path, 'w') as f:
                    json.dump({
                        'models': {},
                        'last_updated': datetime.now().isoformat()
                    }, f)

    def _update_index(self, model_id: str, metadata: ModelMetadata):
        """Update the model index with new metadata"""
        with self._lock:
            with open(self._index_path, 'r') as f:
                index = json.load(f)
            
            index['models'][model_id] = {
                'path': str(self.base_path / f"{model_id}_weights.h5"),
                'metadata_path': str(self.base_path / f"{model_id}_metadata.json"),
                'creation_date': metadata.creation_date,
                'last_updated': datetime.now().isoformat(),
                'versions': ['v1']  # Initialize with first version
            }
            index['last_updated'] = datetime.now().isoformat()
            
            with open(self._index_path, 'w') as f:
                json.dump(index, f, indent=2)

    def load_model(self, model_id: str, wave_function_class: Any, version: str = 'v1') -> Any:
        """Load model weights and reconstruct wave function"""
        try:
            # Load metadata
            metadata_path = self.base_path / f"{model_id}_metadata.json"
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
                # Convert nested dict to appropriate format
                metadata = ModelMetadata(
                    model_id=metadata_dict['model_id'],
                    creation_date=metadata_dict['creation_date'],
                    l_max=metadata_dict['l_max'],
                    m_max=metadata_dict['m_max'],
                    iot_params=metadata_dict['iot_params'],
                    basis_functions=metadata_dict['basis_functions'],
                    performance_metrics=metadata_dict.get('performance_metrics')
                )
            
            # Create IOT instance
            from tidal.core import IOT  # Import here to avoid circular imports
            iot = IOT(**metadata.iot_params)
            
            # Create basis functions
            from tidal.core import BasisFunction
            basis_functions = [BasisFunction(l, m) for l, m in metadata.basis_functions]
            
            # Create wave function instance
            wave_function = wave_function_class(iot, basis_functions)
            
            # Load weights from specific version
            weights_path = self.base_path / f"{model_id}_weights.h5"
            with h5py.File(weights_path, 'r') as f:
                if version not in f:
                    raise VersionNotFoundError(f"Version {version} not found for model {model_id}")
                    
                version_group = f[version]
                coeffs_real = version_group['coefficients_real'][:]
                coeffs_imag = version_group['coefficients_imag'][:]
                wave_function.coefficients = coeffs_real + 1j * coeffs_imag
            
            return wave_function
            
        except FileNotFoundError:
            raise ModelNotFoundError(f"Model {model_id} not found in database")
        except Exception as e:
            raise DatabaseError(f"Error loading model {model_id}: {str(e)}")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all stored models with their metadata"""
        models = []
        for metadata_file in self.base_path.glob('*_metadata.json'):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                models.append(metadata)
        return models
    
    def delete_model(self, model_id: str) -> bool:
        """Delete model weights and metadata from database"""
        try:
            with self._lock:
                # Remove weights file
                weights_path = self.base_path / f"{model_id}_weights.h5"
                weights_path.unlink()
                
                # Remove metadata file
                metadata_path = self.base_path / f"{model_id}_metadata.json"
                metadata_path.unlink()
            return True
        except FileNotFoundError:
            return False

class IOTModelLoader:
    """Utility class for loading and managing TIDAL models"""
    
    def __init__(self, db: IOTDatabase):
        self.db = db
        self._cache = {}
        self._cache_lock = threading.Lock()
    
    def load_model(self, model_id: str, wave_function_class: Any) -> Any:
        """Load model from database with caching"""
        with self._cache_lock:
            if model_id in self._cache:
                return self._cache[model_id]
            
            wave_function = self.db.load_model(model_id, wave_function_class)
            self._cache[model_id] = wave_function
            return wave_function
    
    def clear_cache(self):
        """Clear the model cache"""
        with self._cache_lock:
            self._cache.clear()