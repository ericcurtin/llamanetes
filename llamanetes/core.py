"""
Core building blocks (bricks) for llama.cpp integration.
"""

import subprocess
import json
import tempfile
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


class LlamaBrick(ABC):
    """Base class for all llama.cpp building blocks."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.inputs = {}
        self.outputs = {}
        
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the brick's functionality."""
        pass
        
    def connect(self, other_brick: 'LlamaBrick', output_key: str = 'output', input_key: str = 'input'):
        """Connect this brick's output to another brick's input."""
        if output_key in self.outputs:
            other_brick.inputs[input_key] = self.outputs[output_key]
        return other_brick
        
    def __call__(self, **kwargs):
        """Make bricks callable."""
        return self.execute(**kwargs)


class ModelBrick(LlamaBrick):
    """Brick for loading and managing llama.cpp models."""
    
    def __init__(self, model_path: str, llama_cpp_path: str = "llama-server", **kwargs):
        super().__init__("ModelBrick")
        self.model_path = Path(model_path)
        self.llama_cpp_path = llama_cpp_path
        self.server_args = kwargs
        self.process = None
        self.port = kwargs.get('port', 8080)
        
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Load the model and start llama-server if needed."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        # Check if server is already running
        if self.process and self.process.poll() is None:
            result = {"status": "running", "port": self.port, "model": str(self.model_path)}
        else:
            result = {"status": "loaded", "port": self.port, "model": str(self.model_path)}
            
        self.outputs['model_info'] = result
        return result
        
    def start_server(self) -> bool:
        """Start llama-server process."""
        try:
            cmd = [
                self.llama_cpp_path,
                "--model", str(self.model_path),
                "--port", str(self.port),
                "--host", "127.0.0.1"
            ]
            
            # Add additional server arguments
            for key, value in self.server_args.items():
                if key not in ['port', 'host'] and value is not None:
                    cmd.extend([f"--{key}", str(value)])
                    
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except Exception as e:
            print(f"Failed to start llama-server: {e}")
            return False
            
    def stop_server(self):
        """Stop llama-server process."""
        if self.process:
            self.process.terminate()
            self.process = None


class GenerationBrick(LlamaBrick):
    """Brick for text generation using llama.cpp."""
    
    def __init__(self, model_brick: ModelBrick = None, **generation_params):
        super().__init__("GenerationBrick")
        self.model_brick = model_brick
        self.generation_params = {
            'max_tokens': 100,
            'temperature': 0.8,
            'top_p': 0.9,
            'top_k': 40,
            **generation_params
        }
        
    def execute(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text from the given prompt."""
        params = {**self.generation_params, **kwargs}
        
        # If we have a model brick with a running server, use API
        if self.model_brick and self.model_brick.process:
            result = self._generate_via_api(prompt, params)
        else:
            # Fallback to direct llama-main call
            result = self._generate_direct(prompt, params)
            
        self.outputs['generated_text'] = result['text']
        self.outputs['generation_info'] = result
        return result
        
    def _generate_via_api(self, prompt: str, params: Dict) -> Dict[str, Any]:
        """Generate text via llama-server API."""
        import requests
        
        url = f"http://127.0.0.1:{self.model_brick.port}/completion"
        payload = {
            'prompt': prompt,
            **params
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return {
                'text': data.get('content', ''),
                'method': 'api',
                'status': 'success'
            }
        except Exception as e:
            return {
                'text': '',
                'method': 'api',
                'status': 'error',
                'error': str(e)
            }
            
    def _generate_direct(self, prompt: str, params: Dict) -> Dict[str, Any]:
        """Generate text via direct llama-main call."""
        if not self.model_brick or not self.model_brick.model_path.exists():
            return {
                'text': '',
                'method': 'direct',
                'status': 'error',
                'error': 'No model available'
            }
            
        try:
            # Create temporary file for prompt
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                f.write(prompt)
                prompt_file = f.name
                
            cmd = [
                'llama-main',
                '--model', str(self.model_brick.model_path),
                '--prompt', prompt,
                '--n-predict', str(params.get('max_tokens', 100)),
                '--temp', str(params.get('temperature', 0.8)),
                '--top-p', str(params.get('top_p', 0.9)),
                '--top-k', str(params.get('top_k', 40))
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Clean up temp file
            os.unlink(prompt_file)
            
            if result.returncode == 0:
                return {
                    'text': result.stdout.strip(),
                    'method': 'direct',
                    'status': 'success'
                }
            else:
                return {
                    'text': '',
                    'method': 'direct', 
                    'status': 'error',
                    'error': result.stderr
                }
                
        except Exception as e:
            if 'prompt_file' in locals():
                try:
                    os.unlink(prompt_file)
                except:
                    pass
            return {
                'text': '',
                'method': 'direct',
                'status': 'error',
                'error': str(e)
            }


class TokenizationBrick(LlamaBrick):
    """Brick for tokenization operations."""
    
    def __init__(self, model_brick: ModelBrick = None):
        super().__init__("TokenizationBrick")
        self.model_brick = model_brick
        
    def execute(self, text: str, operation: str = 'tokenize', **kwargs) -> Dict[str, Any]:
        """Perform tokenization operations."""
        if operation == 'tokenize':
            result = self._tokenize(text)
        elif operation == 'detokenize':
            result = self._detokenize(text)
        elif operation == 'count':
            result = self._count_tokens(text)
        else:
            result = {'error': f'Unknown operation: {operation}'}
            
        self.outputs['tokens'] = result
        return result
        
    def _tokenize(self, text: str) -> Dict[str, Any]:
        """Convert text to tokens."""
        if not self.model_brick or not self.model_brick.model_path.exists():
            return {'error': 'No model available for tokenization'}
            
        try:
            cmd = [
                'llama-tokenize',
                '--model', str(self.model_brick.model_path),
                '--prompt', text
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                tokens = result.stdout.strip().split()
                return {
                    'tokens': [int(t) for t in tokens if t.isdigit()],
                    'count': len(tokens),
                    'status': 'success'
                }
            else:
                return {'error': result.stderr, 'status': 'error'}
                
        except Exception as e:
            return {'error': str(e), 'status': 'error'}
            
    def _detokenize(self, tokens: Union[str, List[int]]) -> Dict[str, Any]:
        """Convert tokens back to text."""
        # This would require llama.cpp detokenization functionality
        return {'error': 'Detokenization not yet implemented', 'status': 'error'}
        
    def _count_tokens(self, text: str) -> Dict[str, Any]:
        """Count tokens in text."""
        tokenize_result = self._tokenize(text)
        if 'count' in tokenize_result:
            return {
                'count': tokenize_result['count'],
                'status': 'success'
            }
        return tokenize_result


class ConfigBrick(LlamaBrick):
    """Brick for managing configurations."""
    
    def __init__(self, config_path: str = None):
        super().__init__("ConfigBrick")
        self.config_path = Path(config_path) if config_path else None
        self.config = {}
        
    def execute(self, action: str = 'load', **kwargs) -> Dict[str, Any]:
        """Manage configuration."""
        if action == 'load':
            return self._load_config()
        elif action == 'save':
            return self._save_config(kwargs)
        elif action == 'get':
            return self._get_config(kwargs.get('key'))
        elif action == 'set':
            return self._set_config(kwargs.get('key'), kwargs.get('value'))
        else:
            return {'error': f'Unknown action: {action}'}
            
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.config_path or not self.config_path.exists():
            return {'config': {}, 'status': 'no_file'}
            
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            self.outputs['config'] = self.config
            return {'config': self.config, 'status': 'loaded'}
        except Exception as e:
            return {'error': str(e), 'status': 'error'}
            
    def _save_config(self, config: Dict) -> Dict[str, Any]:
        """Save configuration to file."""
        if not self.config_path:
            return {'error': 'No config path specified', 'status': 'error'}
            
        try:
            self.config.update(config)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return {'status': 'saved'}
        except Exception as e:
            return {'error': str(e), 'status': 'error'}
            
    def _get_config(self, key: str) -> Dict[str, Any]:
        """Get configuration value."""
        return {'value': self.config.get(key), 'status': 'success'}
        
    def _set_config(self, key: str, value: Any) -> Dict[str, Any]:
        """Set configuration value."""
        self.config[key] = value
        self.outputs['config'] = self.config
        return {'status': 'set'}