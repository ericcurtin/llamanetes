"""
Tests for LlamanetES chain functionality.
"""

import pytest
from llamanetes.core import LlamaBrick
from llamanetes.chains import ChainBuilder, Pipeline, ParallelChain


class TestBrick(LlamaBrick):
    """Test brick for chain testing."""
    
    def __init__(self, name, output_value="test"):
        super().__init__(name)
        self.output_value = output_value
    
    def execute(self, **kwargs):
        self.outputs['output'] = self.output_value
        return {'status': 'success', 'value': self.output_value}


def test_chain_builder():
    """Test basic ChainBuilder functionality."""
    chain = ChainBuilder("TestChain")
    
    brick1 = TestBrick("Brick1", "value1")
    brick2 = TestBrick("Brick2", "value2")
    
    chain.add_brick(brick1)
    chain.add_brick(brick2)
    
    assert len(chain.bricks) == 2
    assert chain.bricks[0].name == "Brick1"
    assert chain.bricks[1].name == "Brick2"


def test_chain_connection():
    """Test brick connections in chains."""
    chain = ChainBuilder("TestChain")
    
    brick1 = TestBrick("Brick1", "value1")
    brick2 = TestBrick("Brick2", "value2")
    
    chain.add_brick(brick1)
    chain.add_brick(brick2)
    chain.connect(brick1, brick2, 'output', 'input')
    
    assert len(chain.connections) == 1
    connection = chain.connections[0]
    assert connection['from'] == brick1
    assert connection['to'] == brick2
    assert connection['output_key'] == 'output'
    assert connection['input_key'] == 'input'


def test_chain_execution():
    """Test chain execution."""
    chain = ChainBuilder("TestChain")
    
    brick1 = TestBrick("Brick1", "value1")
    brick2 = TestBrick("Brick2", "value2")
    
    chain.add_brick(brick1)
    chain.add_brick(brick2)
    chain.connect(brick1, brick2, 'output', 'input')
    
    result = chain.execute()
    
    assert result['status'] == 'success'
    assert result['chain_name'] == 'TestChain'
    assert 'results' in result
    assert 'Brick1' in result['results']
    assert 'Brick2' in result['results']


def test_pipeline():
    """Test Pipeline functionality."""
    pipeline = Pipeline("TestPipeline")
    
    brick1 = TestBrick("Brick1", "value1")
    brick2 = TestBrick("Brick2", "value2")
    brick3 = TestBrick("Brick3", "value3")
    
    # Test fluent interface
    pipeline.pipe(brick1).pipe(brick2).pipe(brick3)
    
    assert len(pipeline.bricks) == 3
    assert len(pipeline.connections) == 2  # Auto-connected
    
    result = pipeline.execute()
    assert result['status'] == 'success'


def test_parallel_chain():
    """Test ParallelChain functionality."""
    parallel = ParallelChain("TestParallel")
    
    brick1 = TestBrick("Brick1", "value1")
    brick2 = TestBrick("Brick2", "value2")
    brick3 = TestBrick("Brick3", "value3")
    
    parallel.add_brick(brick1)
    parallel.add_brick(brick2)
    parallel.add_brick(brick3)
    
    result = parallel.execute()
    
    assert result['status'] == 'success'
    assert len(result['results']) == 3
    assert 'Brick1' in result['results']
    assert 'Brick2' in result['results']
    assert 'Brick3' in result['results']


def test_empty_chain():
    """Test empty chain execution."""
    chain = ChainBuilder("EmptyChain")
    
    result = chain.execute()
    
    assert result['status'] == 'error'
    assert 'No bricks in chain' in result['error']


if __name__ == '__main__':
    pytest.main([__file__])