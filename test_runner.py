#!/usr/bin/env python3
"""
Simple test runner for LlamanetES without requiring pytest.
"""

import sys
import traceback
import tempfile
import json
from pathlib import Path

# Add the package to the path
sys.path.insert(0, '/home/runner/work/llamanetes/llamanetes')

from llamanetes.core import LlamaBrick, ModelBrick, GenerationBrick, TokenizationBrick, ConfigBrick
from llamanetes.chains import ChainBuilder, Pipeline, ParallelChain


class MockBrick(LlamaBrick):
    """Mock brick for testing."""
    
    def execute(self, **kwargs):
        self.outputs['test_output'] = kwargs.get('input', 'test_result')
        return {'status': 'success', 'result': self.outputs['test_output']}


def run_test(test_func):
    """Run a single test function."""
    try:
        test_func()
        print(f"✓ {test_func.__name__}")
        return True
    except Exception as e:
        print(f"✗ {test_func.__name__}: {e}")
        traceback.print_exc()
        return False


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
    
    try:
        model.execute()
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass  # Expected


def test_generation_brick():
    """Test GenerationBrick functionality."""
    # Test without model
    gen = GenerationBrick()
    result = gen.execute(prompt="test")
    
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
        
    finally:
        Path(config_path).unlink()


def test_chain_builder():
    """Test basic ChainBuilder functionality."""
    chain = ChainBuilder("TestChain")
    
    brick1 = MockBrick()
    brick1.name = "Brick1"
    brick2 = MockBrick()
    brick2.name = "Brick2"
    
    chain.add_brick(brick1)
    chain.add_brick(brick2)
    
    assert len(chain.bricks) == 2
    assert chain.bricks[0].name == "Brick1"
    assert chain.bricks[1].name == "Brick2"


def test_pipeline():
    """Test Pipeline functionality."""
    pipeline = Pipeline("TestPipeline")
    
    brick1 = MockBrick()
    brick1.name = "Brick1"
    brick2 = MockBrick()
    brick2.name = "Brick2"
    
    # Test fluent interface
    pipeline.pipe(brick1).pipe(brick2)
    
    assert len(pipeline.bricks) == 2
    assert len(pipeline.connections) == 1  # Auto-connected
    
    result = pipeline.execute()
    assert result['status'] == 'success'


def main():
    """Run all tests."""
    print("Running LlamanetES tests...")
    print("=" * 40)
    
    tests = [
        test_llama_brick_base,
        test_brick_connection,
        test_model_brick,
        test_generation_brick,
        test_config_brick,
        test_chain_builder,
        test_pipeline,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if run_test(test):
            passed += 1
        else:
            failed += 1
    
    print("=" * 40)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())