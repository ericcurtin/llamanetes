"""
Tests for LlamanetES core functionality.
"""

import pytest
import tempfile
import json
from pathlib import Path
from llamanetes.core import LlamaBrick, ModelBrick, GenerationBrick, TokenizationBrick, ConfigBrick


class MockBrick(LlamaBrick):
    """Mock brick for testing."""
    
    def execute(self, **kwargs):
        self.outputs['test_output'] = kwargs.get('input', 'test_result')
        return {'status': 'success', 'result': self.outputs['test_output']}


def test_llama_brick_base():
    """Test base LlamaBrick functionality."""
    brick = MockBrick()
    assert brick.name == 'MockBrick'
    assert brick.inputs == {}
    assert brick.outputs == {}
    
    result = brick.execute(input='test')
    assert result['status'] == 'success'
    assert result['result'] == 'test'
    assert brick.outputs['test_output'] == 'test'


def test_brick_connection():
    """Test brick connection functionality."""
    brick1 = MockBrick()
    brick2 = MockBrick()
    
    # Execute first brick
    brick1.execute(input='first')
    
    # Connect bricks
    brick1.connect(brick2, 'test_output', 'input')
    
    # Check connection
    assert brick2.inputs['input'] == 'first'


def test_model_brick():
    """Test ModelBrick functionality."""
    # Test with non-existent model
    model = ModelBrick('/nonexistent/path/model.gguf')
    
    with pytest.raises(FileNotFoundError):
        model.execute()


def test_generation_brick():
    """Test GenerationBrick functionality."""
    # Test without model
    gen = GenerationBrick()
    result = gen.execute(prompt="test")
    
    assert result['status'] == 'error'
    assert 'No model available' in result['error']


def test_tokenization_brick():
    """Test TokenizationBrick functionality."""
    # Test without model
    tok = TokenizationBrick()
    result = tok.execute(text="test", operation="tokenize")
    
    assert result['status'] == 'error'
    assert 'No model available' in result['error']


def test_config_brick():
    """Test ConfigBrick functionality."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_config = {'test_key': 'test_value', 'number': 42}
        json.dump(test_config, f)
        config_path = f.name
    
    try:
        config = ConfigBrick(config_path)
        
        # Test load
        result = config.execute(action='load')
        assert result['status'] == 'loaded'
        assert result['config'] == test_config
        
        # Test get
        result = config.execute(action='get', key='test_key')
        assert result['value'] == 'test_value'
        
        # Test set
        result = config.execute(action='set', key='new_key', value='new_value')
        assert result['status'] == 'set'
        
        # Test save
        result = config.execute(action='save', new_data={'extra': 'data'})
        assert result['status'] == 'saved'
        
        # Verify saved data
        result = config.execute(action='load')
        assert 'new_key' in result['config']
        assert result['config']['new_key'] == 'new_value'
        
    finally:
        Path(config_path).unlink()


def test_config_brick_no_file():
    """Test ConfigBrick with no file."""
    config = ConfigBrick()
    
    result = config.execute(action='load')
    assert result['status'] == 'no_file'
    assert result['config'] == {}


if __name__ == '__main__':
    pytest.main([__file__])