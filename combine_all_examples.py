#!/usr/bin/env python3
"""Combine all training examples including new science ones"""
import json
from datetime import datetime, timezone

def combine_all_datasets():
    # Load existing comprehensive dataset
    with open('./data/worldmodel_final_training.json') as f:
        original_data = json.load(f)
    
    # Load new science examples
    with open('./data/worldmodel_science_additional.json') as f:
        science_data = json.load(f)
    
    # Combine all examples
    all_examples = original_data['examples'] + science_data['examples']
    
    # Create enhanced final dataset
    enhanced_dataset = {
        "metadata": {
            "total_examples": len(all_examples),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generator_version": "4.0_enhanced_with_science",
            "description": "Enhanced WorldModel training dataset with science and tricky problems",
            "original_examples": len(original_data['examples']),
            "science_examples": len(science_data['examples'])
        },
        "examples": all_examples
    }
    
    # Save enhanced dataset
    with open('./data/worldmodel_enhanced_training.json', 'w') as f:
        json.dump(enhanced_dataset, f, indent=2)
    
    print(f"🎯 Enhanced Dataset Created!")
    print(f"   📊 Original examples: {len(original_data['examples'])}")
    print(f"   🧪 Science examples: {len(science_data['examples'])}")
    print(f"   📈 Total examples: {len(all_examples)}")
    print(f"   💾 Saved to: ./data/worldmodel_enhanced_training.json")

if __name__ == "__main__":
    combine_all_datasets()