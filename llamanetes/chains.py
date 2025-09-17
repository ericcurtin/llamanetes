"""
Chain builder for combining multiple bricks into workflows.
"""

from typing import List, Dict, Any, Optional
from .core import LlamaBrick


class ChainBuilder:
    """Build and execute chains of llama.cpp bricks."""
    
    def __init__(self, name: str = "Chain"):
        self.name = name
        self.bricks: List[LlamaBrick] = []
        self.connections: List[Dict] = []
        
    def add_brick(self, brick: LlamaBrick) -> 'ChainBuilder':
        """Add a brick to the chain."""
        self.bricks.append(brick)
        return self
        
    def connect(self, from_brick: LlamaBrick, to_brick: LlamaBrick, 
                output_key: str = 'output', input_key: str = 'input') -> 'ChainBuilder':
        """Connect two bricks in the chain."""
        connection = {
            'from': from_brick,
            'to': to_brick,
            'output_key': output_key,
            'input_key': input_key
        }
        self.connections.append(connection)
        return self
        
    def execute(self, initial_input: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the entire chain."""
        if not self.bricks:
            return {'error': 'No bricks in chain', 'status': 'error'}
            
        results = {}
        
        try:
            # Execute first brick with initial input
            first_brick = self.bricks[0]
            if initial_input:
                first_result = first_brick.execute(**initial_input)
            else:
                first_result = first_brick.execute()
                
            results[first_brick.name] = first_result
            
            # Execute connections in order
            for connection in self.connections:
                from_brick = connection['from']
                to_brick = connection['to']
                output_key = connection['output_key']
                input_key = connection['input_key']
                
                # Transfer output from source brick to target brick
                if output_key in from_brick.outputs:
                    to_brick.inputs[input_key] = from_brick.outputs[output_key]
                    
                # Execute target brick if it hasn't been executed yet
                if to_brick.name not in results:
                    brick_result = to_brick.execute(**to_brick.inputs)
                    results[to_brick.name] = brick_result
                    
            return {
                'chain_name': self.name,
                'status': 'success',
                'results': results
            }
            
        except Exception as e:
            return {
                'chain_name': self.name,
                'status': 'error',
                'error': str(e),
                'results': results
            }
            
    def __call__(self, **kwargs):
        """Make chain callable."""
        return self.execute(kwargs)


class Pipeline(ChainBuilder):
    """A linear pipeline of bricks."""
    
    def __init__(self, name: str = "Pipeline"):
        super().__init__(name)
        
    def add_brick(self, brick: LlamaBrick) -> 'Pipeline':
        """Add a brick to the pipeline and auto-connect to previous brick."""
        if self.bricks:
            # Auto-connect to previous brick
            self.connect(self.bricks[-1], brick)
        super().add_brick(brick)
        return self
        
    def pipe(self, brick: LlamaBrick) -> 'Pipeline':
        """Alias for add_brick with fluent interface."""
        return self.add_brick(brick)


class ParallelChain(ChainBuilder):
    """Execute multiple bricks in parallel."""
    
    def __init__(self, name: str = "ParallelChain"):
        super().__init__(name)
        
    def execute(self, initial_input: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute all bricks in parallel with same input."""
        if not self.bricks:
            return {'error': 'No bricks in chain', 'status': 'error'}
            
        results = {}
        
        try:
            for brick in self.bricks:
                if initial_input:
                    result = brick.execute(**initial_input)
                else:
                    result = brick.execute()
                results[brick.name] = result
                
            return {
                'chain_name': self.name,
                'status': 'success',
                'results': results
            }
            
        except Exception as e:
            return {
                'chain_name': self.name,
                'status': 'error',
                'error': str(e),
                'results': results
            }


class ConditionalChain(ChainBuilder):
    """Execute bricks based on conditions."""
    
    def __init__(self, name: str = "ConditionalChain"):
        super().__init__(name)
        self.conditions = {}
        
    def add_condition(self, condition_func, true_brick: LlamaBrick, 
                     false_brick: LlamaBrick = None) -> 'ConditionalChain':
        """Add a conditional execution rule."""
        self.conditions[condition_func] = {
            'true': true_brick,
            'false': false_brick
        }
        return self
        
    def execute(self, initial_input: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute bricks based on conditions."""
        results = {}
        
        try:
            for condition_func, branches in self.conditions.items():
                if initial_input and condition_func(initial_input):
                    # Execute true branch
                    if branches['true']:
                        result = branches['true'].execute(**(initial_input or {}))
                        results[branches['true'].name] = result
                else:
                    # Execute false branch
                    if branches['false']:
                        result = branches['false'].execute(**(initial_input or {}))
                        results[branches['false'].name] = result
                        
            return {
                'chain_name': self.name,
                'status': 'success',
                'results': results
            }
            
        except Exception as e:
            return {
                'chain_name': self.name,
                'status': 'error',
                'error': str(e),
                'results': results
            }