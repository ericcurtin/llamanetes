#!/usr/bin/env python3
"""
Basic example of using LlamanetES for text generation.
"""

from llamanetes import ModelBrick, GenerationBrick


def main():
    # Note: Replace with actual model path
    model_path = "/path/to/your/model.gguf"
    
    print("LlamanetES Basic Example")
    print("=" * 30)
    
    try:
        # Create model brick
        print("Loading model...")
        model = ModelBrick(model_path)
        
        # Create generation brick with custom parameters
        generator = GenerationBrick(
            model, 
            max_tokens=100,
            temperature=0.7,
            top_p=0.9
        )
        
        # Generate text
        prompt = "The future of artificial intelligence is"
        print(f"Prompt: {prompt}")
        print("Generating...")
        
        result = generator(prompt=prompt)
        
        if result['status'] == 'success':
            print(f"Generated text: {result['text']}")
        else:
            print(f"Generation failed: {result.get('error')}")
            
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        print("Please update the model_path variable with your actual model file.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()