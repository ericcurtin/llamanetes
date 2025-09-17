# LlamanetES

**AIBrix-like tool for llama.cpp in Python3**

LlamanetES provides a modular, building-block approach to working with llama.cpp models. Like LEGO bricks, you can combine different components to create complex AI workflows with minimal code.

## Features

- 🧱 **Modular Architecture**: Building blocks (bricks) that can be combined
- 🔗 **Chain Operations**: Connect bricks to create workflows  
- 🖥️ **CLI Interface**: Easy command-line usage
- 🐍 **Python API**: Programmatic access for developers
- 🚀 **Server Mode**: Support for llama-server API
- 🔧 **Configuration**: JSON-based configuration management

## Installation

```bash
pip install -e .
```

Or install dependencies manually:
```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Usage

```bash
# Generate text
llamanetes generate --model /path/to/model.gguf --prompt "Hello, world!"

# Count tokens
llamanetes tokenize --model /path/to/model.gguf --text "Hello, world!" --count

# Interactive mode
llamanetes interactive --model /path/to/model.gguf
```

### Python API Usage

```python
from llamanetes import ModelBrick, GenerationBrick, Pipeline

# Create bricks
model = ModelBrick("/path/to/model.gguf")
generator = GenerationBrick(model, max_tokens=50, temperature=0.7)

# Generate text
result = generator("Tell me a joke")
print(result['text'])

# Create a pipeline
pipeline = Pipeline("MyPipeline")
pipeline.add_brick(model).add_brick(generator)

# Execute pipeline
result = pipeline.execute({"prompt": "What is AI?"})
print(result)
```

## Core Concepts

### Bricks

Bricks are the fundamental building blocks:

- **ModelBrick**: Manages llama.cpp models
- **GenerationBrick**: Handles text generation
- **TokenizationBrick**: Tokenizes and counts tokens
- **ConfigBrick**: Manages configurations

### Chains

Chains combine multiple bricks:

- **Pipeline**: Linear chain of bricks
- **ParallelChain**: Execute bricks in parallel
- **ConditionalChain**: Execute based on conditions

## Examples

### Basic Text Generation

```python
from llamanetes import ModelBrick, GenerationBrick

# Setup
model = ModelBrick("model.gguf")
gen = GenerationBrick(model, max_tokens=100)

# Generate
result = gen("Write a Python function:")
print(result['text'])
```

### Token Analysis

```python
from llamanetes import ModelBrick, TokenizationBrick

model = ModelBrick("model.gguf") 
tokenizer = TokenizationBrick(model)

# Count tokens
result = tokenizer("This is a test sentence", operation="count")
print(f"Token count: {result['count']}")

# Get tokens
result = tokenizer("This is a test", operation="tokenize")
print(f"Tokens: {result['tokens']}")
```

### Pipeline Example

```python
from llamanetes import ModelBrick, GenerationBrick, Pipeline

# Create pipeline
pipeline = Pipeline("StoryGenerator")
model = ModelBrick("model.gguf")
generator = GenerationBrick(model, max_tokens=200, temperature=0.8)

pipeline.pipe(model).pipe(generator)

# Execute
result = pipeline(prompt="Once upon a time")
print(result['results']['GenerationBrick']['text'])
```

### Server Mode

```python
from llamanetes import ModelBrick, GenerationBrick

# Start server
model = ModelBrick("model.gguf", port=8080)
model.start_server()

# Use server for generation
gen = GenerationBrick(model)
result = gen("Hello, world!")

# Cleanup
model.stop_server()
```

### Configuration Management

```python
from llamanetes import ConfigBrick

config = ConfigBrick("config.json")

# Load config
config.execute(action="load")

# Set values
config.execute(action="set", key="model_path", value="/path/to/model.gguf")
config.execute(action="set", key="temperature", value=0.7)

# Save config
config.execute(action="save")
```

## CLI Commands

### Generate
```bash
llamanetes generate --model model.gguf --prompt "Hello" --max-tokens 50
```

### Tokenize
```bash
llamanetes tokenize --model model.gguf --text "Hello world" --count
```

### Pipeline
```bash
llamanetes pipeline --config pipeline.json --input '{"prompt": "Hello"}'
```

### Interactive
```bash
llamanetes interactive --model model.gguf --server --port 8080
```

### Config
```bash
llamanetes config --file config.json --set model_path /path/to/model.gguf
llamanetes config --file config.json --get model_path
llamanetes config --file config.json --list
```

## Pipeline Configuration

Create JSON configuration files for complex workflows:

```json
{
  "name": "TextAnalysisPipeline",
  "bricks": [
    {
      "type": "model",
      "params": {
        "model_path": "/path/to/model.gguf",
        "port": 8080
      }
    },
    {
      "type": "generation", 
      "params": {
        "max_tokens": 100,
        "temperature": 0.7
      }
    }
  ]
}
```

## Requirements

- Python 3.8+
- llama.cpp installed and in PATH
- Required Python packages (see requirements.txt)

## License

Apache License 2.0 - see LICENSE file for details.

## Contributing

Contributions welcome! Please read the contributing guidelines and submit pull requests.

## Roadmap

- [ ] Additional brick types (embedding, fine-tuning)
- [ ] Web interface
- [ ] Docker support
- [ ] Model zoo integration
- [ ] Advanced pipeline features
- [ ] Performance optimizations