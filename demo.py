#!/usr/bin/env python3
"""
Demonstration of LlamanetES features.
This shows how the tool would work with actual llama.cpp models.
"""

from llamanetes import ModelBrick, GenerationBrick, TokenizationBrick, ConfigBrick
from llamanetes.chains import Pipeline, ParallelChain, ChainBuilder


def demo_basic_usage():
    """Demo basic brick usage."""
    print("=== Basic Brick Usage Demo ===")
    
    # This would work with an actual model file
    print("1. Model Management:")
    print("   model = ModelBrick('/path/to/model.gguf')")
    print("   model.start_server()  # Start llama-server")
    print("   model.stop_server()   # Stop server")
    
    print("\n2. Text Generation:")
    print("   generator = GenerationBrick(model, max_tokens=100, temperature=0.7)")
    print("   result = generator('Write a Python function:')")
    print("   print(result['text'])  # Generated code")
    
    print("\n3. Tokenization:")
    print("   tokenizer = TokenizationBrick(model)")
    print("   tokens = tokenizer('Hello world', operation='tokenize')")
    print("   count = tokenizer('Hello world', operation='count')")
    

def demo_pipeline():
    """Demo pipeline functionality.""" 
    print("\n=== Pipeline Demo ===")
    
    print("Creating a text analysis pipeline:")
    print("""
# Create pipeline
pipeline = Pipeline("TextAnalysis")

# Add bricks in sequence
model = ModelBrick("model.gguf")
tokenizer = TokenizationBrick(model)
generator = GenerationBrick(model, max_tokens=50)

pipeline.pipe(model).pipe(tokenizer).pipe(generator)

# Execute pipeline
result = pipeline.execute({"text": "Analyze this text"})
    """)


def demo_parallel_processing():
    """Demo parallel processing."""
    print("\n=== Parallel Processing Demo ===")
    
    print("Process text with multiple models simultaneously:")
    print("""
# Create parallel chain
parallel = ParallelChain("MultiModel")

# Add different models/configurations
model1 = ModelBrick("creative_model.gguf")
model2 = ModelBrick("factual_model.gguf")
gen1 = GenerationBrick(model1, temperature=0.9)  # Creative
gen2 = GenerationBrick(model2, temperature=0.1)  # Factual

parallel.add_brick(gen1).add_brick(gen2)

# Get responses from both models
result = parallel.execute({"prompt": "Explain quantum computing"})
creative_response = result['results']['GenerationBrick']['text']
factual_response = result['results']['GenerationBrick']['text']
    """)


def demo_cli_usage():
    """Demo CLI usage."""
    print("\n=== CLI Usage Demo ===")
    
    print("Command-line interface examples:")
    print("""
# Generate text
$ llamanetes generate --model model.gguf --prompt "Hello world" --max-tokens 50

# Count tokens
$ llamanetes tokenize --model model.gguf --text "How many tokens?" --count

# Interactive mode
$ llamanetes interactive --model model.gguf --server --port 8080

# Run pipeline from config
$ llamanetes pipeline --config workflow.json --input '{"text": "Process this"}'

# Manage configuration
$ llamanetes config --file settings.json --set temperature 0.8
$ llamanetes config --file settings.json --get temperature
    """)


def demo_configuration():
    """Demo configuration management."""
    print("\n=== Configuration Demo ===")
    
    print("JSON-based configuration:")
    print("""
{
  "name": "CreativeWritingPipeline",
  "bricks": [
    {
      "type": "model",
      "params": {
        "model_path": "/models/creative-writer.gguf",
        "port": 8080
      }
    },
    {
      "type": "generation",
      "params": {
        "max_tokens": 200,
        "temperature": 0.8,
        "top_p": 0.9
      }
    }
  ]
}
    """)


def main():
    """Run the demonstration."""
    print("LlamanetES - AIBrix-like Tool for llama.cpp")
    print("=" * 50)
    print()
    print("This tool provides LEGO-like building blocks for AI workflows:")
    print("• ModelBrick - Manage llama.cpp models")
    print("• GenerationBrick - Generate text") 
    print("• TokenizationBrick - Tokenize and analyze text")
    print("• ConfigBrick - Manage configurations")
    print("• Chains - Combine bricks into workflows")
    print()
    
    demo_basic_usage()
    demo_pipeline()
    demo_parallel_processing()
    demo_cli_usage()
    demo_configuration()
    
    print("\n=== Summary ===")
    print("LlamanetES provides:")
    print("✓ Modular architecture with reusable components")
    print("✓ Both Python API and CLI interface")
    print("✓ Support for llama.cpp server and direct execution")
    print("✓ Chain operations for complex workflows")
    print("✓ JSON configuration for reproducible setups")
    print("✓ Comprehensive documentation and examples")
    print()
    print("Ready to use with your llama.cpp models!")


if __name__ == "__main__":
    main()