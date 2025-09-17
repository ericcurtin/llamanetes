#!/usr/bin/env python3
"""
Command-line interface for LlamanetES.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

from .core import ModelBrick, GenerationBrick, TokenizationBrick, ConfigBrick
from .chains import Pipeline, ParallelChain, ChainBuilder


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LlamanetES - AIBrix-like tool for llama.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate text with a model
  llamanetes generate --model /path/to/model.gguf --prompt "Hello, world!"
  
  # Count tokens in text
  llamanetes tokenize --model /path/to/model.gguf --text "Hello, world!" --count
  
  # Create a pipeline
  llamanetes pipeline --config pipeline.json
  
  # Start interactive mode
  llamanetes interactive --model /path/to/model.gguf
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate text')
    gen_parser.add_argument('--model', required=True, help='Path to model file')
    gen_parser.add_argument('--prompt', required=True, help='Input prompt')
    gen_parser.add_argument('--max-tokens', type=int, default=100, help='Maximum tokens to generate')
    gen_parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for generation')
    gen_parser.add_argument('--top-p', type=float, default=0.9, help='Top-p for nucleus sampling')
    gen_parser.add_argument('--top-k', type=int, default=40, help='Top-k for sampling')
    gen_parser.add_argument('--server', action='store_true', help='Use server mode')
    gen_parser.add_argument('--port', type=int, default=8080, help='Server port')
    
    # Tokenize command
    tok_parser = subparsers.add_parser('tokenize', help='Tokenize text')
    tok_parser.add_argument('--model', required=True, help='Path to model file')
    tok_parser.add_argument('--text', required=True, help='Text to tokenize')
    tok_parser.add_argument('--count', action='store_true', help='Just count tokens')
    
    # Pipeline command
    pipe_parser = subparsers.add_parser('pipeline', help='Run a pipeline')
    pipe_parser.add_argument('--config', required=True, help='Path to pipeline config file')
    pipe_parser.add_argument('--input', help='Input data (JSON string)')
    
    # Interactive command
    int_parser = subparsers.add_parser('interactive', help='Interactive mode')
    int_parser.add_argument('--model', required=True, help='Path to model file')
    int_parser.add_argument('--server', action='store_true', help='Use server mode')
    int_parser.add_argument('--port', type=int, default=8080, help='Server port')
    
    # Config command
    conf_parser = subparsers.add_parser('config', help='Manage configuration')
    conf_parser.add_argument('--file', help='Config file path')
    conf_parser.add_argument('--get', help='Get config value')
    conf_parser.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'), help='Set config value')
    conf_parser.add_argument('--list', action='store_true', help='List all config values')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    try:
        if args.command == 'generate':
            return cmd_generate(args)
        elif args.command == 'tokenize':
            return cmd_tokenize(args)
        elif args.command == 'pipeline':
            return cmd_pipeline(args)
        elif args.command == 'interactive':
            return cmd_interactive(args)
        elif args.command == 'config':
            return cmd_config(args)
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return 1
            
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_generate(args) -> int:
    """Handle generate command."""
    model_brick = ModelBrick(args.model, port=args.port)
    gen_brick = GenerationBrick(
        model_brick, 
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )
    
    if args.server:
        print("Starting llama-server...")
        if not model_brick.start_server():
            print("Failed to start server", file=sys.stderr)
            return 1
        print(f"Server started on port {args.port}")
        
    try:
        result = gen_brick.execute(prompt=args.prompt)
        
        if result['status'] == 'success':
            print(result['text'])
            return 0
        else:
            print(f"Generation failed: {result.get('error', 'Unknown error')}", file=sys.stderr)
            return 1
            
    finally:
        if args.server:
            model_brick.stop_server()


def cmd_tokenize(args) -> int:
    """Handle tokenize command."""
    model_brick = ModelBrick(args.model)
    tok_brick = TokenizationBrick(model_brick)
    
    operation = 'count' if args.count else 'tokenize'
    result = tok_brick.execute(text=args.text, operation=operation)
    
    if result['status'] == 'success':
        if args.count:
            print(f"Token count: {result['count']}")
        else:
            print(f"Tokens: {result['tokens']}")
            print(f"Count: {result['count']}")
        return 0
    else:
        print(f"Tokenization failed: {result.get('error', 'Unknown error')}", file=sys.stderr)
        return 1


def cmd_pipeline(args) -> int:
    """Handle pipeline command."""
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Failed to load config: {e}", file=sys.stderr)
        return 1
        
    # Parse input data
    input_data = {}
    if args.input:
        try:
            input_data = json.loads(args.input)
        except Exception as e:
            print(f"Failed to parse input JSON: {e}", file=sys.stderr)
            return 1
            
    # Build pipeline from config
    pipeline = build_pipeline_from_config(config)
    if not pipeline:
        print("Failed to build pipeline from config", file=sys.stderr)
        return 1
        
    # Execute pipeline
    result = pipeline.execute(input_data)
    
    if result['status'] == 'success':
        print(json.dumps(result, indent=2))
        return 0
    else:
        print(f"Pipeline failed: {result.get('error', 'Unknown error')}", file=sys.stderr)
        return 1


def cmd_interactive(args) -> int:
    """Handle interactive command."""
    model_brick = ModelBrick(args.model, port=args.port)
    gen_brick = GenerationBrick(model_brick)
    
    if args.server:
        print("Starting llama-server...")
        if not model_brick.start_server():
            print("Failed to start server", file=sys.stderr)
            return 1
        print(f"Server started on port {args.port}")
        
    try:
        print("LlamanetES Interactive Mode")
        print("Type 'quit' or 'exit' to quit, 'help' for commands")
        print("-" * 50)
        
        while True:
            try:
                prompt = input(">>> ").strip()
                
                if prompt.lower() in ['quit', 'exit']:
                    break
                elif prompt.lower() == 'help':
                    print_interactive_help()
                    continue
                elif not prompt:
                    continue
                    
                result = gen_brick.execute(prompt=prompt)
                
                if result['status'] == 'success':
                    print(f"Generated: {result['text']}")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
                    
            except EOFError:
                break
                
        print("\nGoodbye!")
        return 0
        
    finally:
        if args.server:
            model_brick.stop_server()


def cmd_config(args) -> int:
    """Handle config command."""
    config_brick = ConfigBrick(args.file)
    
    if args.list:
        result = config_brick.execute(action='load')
        if 'config' in result:
            print(json.dumps(result['config'], indent=2))
        else:
            print("No configuration found")
        return 0
        
    elif args.get:
        config_brick.execute(action='load')
        result = config_brick.execute(action='get', key=args.get)
        print(f"{args.get}: {result['value']}")
        return 0
        
    elif args.set:
        key, value = args.set
        config_brick.execute(action='load')
        
        # Try to parse value as JSON first
        try:
            parsed_value = json.loads(value)
        except:
            parsed_value = value
            
        config_brick.execute(action='set', key=key, value=parsed_value)
        result = config_brick.execute(action='save')
        
        if result['status'] == 'saved':
            print(f"Set {key} = {parsed_value}")
            return 0
        else:
            print(f"Failed to save config: {result.get('error')}", file=sys.stderr)
            return 1
    else:
        print("Use --list, --get KEY, or --set KEY VALUE", file=sys.stderr)
        return 1


def build_pipeline_from_config(config: Dict[str, Any]) -> Pipeline:
    """Build a pipeline from configuration."""
    # This is a simplified pipeline builder
    # In a real implementation, this would be more sophisticated
    pipeline = Pipeline(config.get('name', 'ConfigPipeline'))
    
    for brick_config in config.get('bricks', []):
        brick_type = brick_config.get('type')
        brick_params = brick_config.get('params', {})
        
        if brick_type == 'model':
            brick = ModelBrick(**brick_params)
        elif brick_type == 'generation':
            brick = GenerationBrick(**brick_params)
        elif brick_type == 'tokenization':
            brick = TokenizationBrick(**brick_params)
        else:
            print(f"Unknown brick type: {brick_type}", file=sys.stderr)
            return None
            
        pipeline.add_brick(brick)
        
    return pipeline


def print_interactive_help():
    """Print help for interactive mode."""
    print("""
Interactive Commands:
  - Type any text to generate a response
  - 'help' - Show this help
  - 'quit' or 'exit' - Exit interactive mode
    """)


if __name__ == '__main__':
    sys.exit(main())