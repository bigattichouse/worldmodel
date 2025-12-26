#!/usr/bin/env python3
"""
Convert WorldModel training data to llama.cpp format
"""
import json
import sys
sys.path.insert(0, 'src')

def convert_to_llama_cpp_format():
    """Convert our WorldModel training data to llama.cpp raw text format"""
    
    print("🔄 Converting WorldModel data to llama.cpp format...")
    
    # Load our enhanced training dataset
    with open('./data/worldmodel_enhanced_training.json') as f:
        data = json.load(f)
    
    examples = data['examples']
    print(f"📊 Processing {len(examples)} examples")
    
    # llama.cpp expects raw text with examples concatenated
    # We'll use the conversation format that mimics training
    
    output_lines = []
    
    for i, example in enumerate(examples):
        input_text = example['input_text']
        target_output = example['target_output']
        
        # Create a conversation format similar to what the model expects
        # This mimics the chat format but in plain text for training
        conversation = f"User: {input_text}\n\nAssistant: {target_output}\n\n"
        output_lines.append(conversation)
        
        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{len(examples)} examples...")
    
    # Save to raw text file for llama.cpp
    output_file = "./data/worldmodel_llama_cpp_training.txt"
    with open(output_file, 'w') as f:
        f.write(''.join(output_lines))
    
    # Calculate statistics
    total_chars = sum(len(line) for line in output_lines)
    total_tokens_approx = total_chars // 4  # Rough estimate: 4 chars per token
    
    print(f"✅ Conversion complete!")
    print(f"   📁 Output file: {output_file}")
    print(f"   📊 Total examples: {len(examples)}")
    print(f"   📝 Total characters: {total_chars:,}")
    print(f"   🎯 Estimated tokens: ~{total_tokens_approx:,}")
    
    # Show a sample of the output
    print(f"\n🔍 Sample output (first 500 chars):")
    with open(output_file) as f:
        sample = f.read(500)
    print(sample + "...")
    
    return output_file

if __name__ == "__main__":
    convert_to_llama_cpp_format()