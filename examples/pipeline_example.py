#!/usr/bin/env python3
"""
Example of using LlamanetES pipelines for complex workflows.
"""

from llamanetes import ModelBrick, GenerationBrick, TokenizationBrick, Pipeline


def main():
    # Note: Replace with actual model path
    model_path = "/path/to/your/model.gguf"
    
    print("LlamanetES Pipeline Example")
    print("=" * 35)
    
    try:
        # Create bricks
        model = ModelBrick(model_path)
        tokenizer = TokenizationBrick(model)
        generator = GenerationBrick(
            model,
            max_tokens=50,
            temperature=0.8
        )
        
        # Create a pipeline that first counts tokens, then generates
        pipeline = Pipeline("AnalysisAndGeneration")
        
        # Note: This is a simplified example
        # In practice, you'd have more sophisticated connections
        pipeline.add_brick(model)
        pipeline.add_brick(generator)
        
        # Test input
        test_prompt = "Write a short poem about coding"
        
        print(f"Input prompt: {test_prompt}")
        
        # First, analyze tokens
        print("\n1. Token Analysis:")
        token_result = tokenizer(text=test_prompt, operation="count")
        if token_result['status'] == 'success':
            print(f"   Input has {token_result['count']} tokens")
        
        # Then generate text
        print("\n2. Text Generation:")
        pipeline_result = pipeline.execute({"prompt": test_prompt})
        
        if pipeline_result['status'] == 'success':
            gen_result = pipeline_result['results'].get('GenerationBrick', {})
            if gen_result.get('status') == 'success':
                generated_text = gen_result['text']
                print(f"   Generated: {generated_text}")
                
                # Analyze generated text tokens
                print(f"\n3. Generated Text Analysis:")
                gen_token_result = tokenizer(text=generated_text, operation="count")
                if gen_token_result['status'] == 'success':
                    print(f"   Generated text has {gen_token_result['count']} tokens")
            else:
                print(f"   Generation failed: {gen_result.get('error')}")
        else:
            print(f"Pipeline failed: {pipeline_result.get('error')}")
            
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        print("Please update the model_path variable with your actual model file.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()