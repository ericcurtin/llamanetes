"""
LlamanetES - AIBrix-like tool for llama.cpp in Python3

A modular, building-block approach to working with llama.cpp models.
Provides LEGO-like bricks that can be combined to create complex AI workflows.
"""

__version__ = "0.1.0"
__author__ = "Eric Curtin"

from .core import LlamaBrick, ModelBrick, GenerationBrick, TokenizationBrick
from .chains import ChainBuilder
from .cli import main

__all__ = [
    "LlamaBrick",
    "ModelBrick", 
    "GenerationBrick",
    "TokenizationBrick",
    "ChainBuilder",
    "main"
]