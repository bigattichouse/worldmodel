#!/usr/bin/env python3
"""
Interactive Chat with WASM WorldModel
====================================

Chat with the WASM WorldModel and see calculations happening in real-time.
"""

import sys
sys.path.append('src')

import torch
from src.models.qwen_wasm_adapter import QwenWASMAdapter
from src.tokenization.wat_tokenizer import WATTokenizer

class WASMChat:
    """Interactive chat interface for WASM WorldModel."""
    
    def __init__(self):
        print("🔧 Loading WASM WorldModel...")
        
        print("Loading Qwen model from /home/bigattichouse/workspace/model/Qwen3-0.6B")
        print("🔧 WASM Executor initialized:")
        print("   Internal WASM: Direct execution")  
        print("   External APIs: Direct host")
        
        # Initialize model
        self.model = QwenWASMAdapter(
            model_path="/home/bigattichouse/workspace/model/Qwen3-0.6B",
            cross_modal_layers=[3, 7, 11],
            use_sandbox=False
        )
        
        # Load trained weights from latest checkpoint
        checkpoint_path = "./wasm_worldmodel_output/checkpoint-1890/pytorch_model.bin"
        print(f"🔄 Loading weights from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        # Set WASM tokenizer
        wasm_tokenizer = WATTokenizer(vocab_size=8000)
        self.model.set_wasm_tokenizer(wasm_tokenizer)
        
        print("✅ WASM WorldModel ready!")
        print("🧠 Cross-modal computation layers: [3, 7, 11]")
        print()
    
    def process_question(self, question: str):
        """Process a user question and show WASM calculations."""
        print(f"🤔 Processing: {question}")
        print("=" * 60)
        
        # Prepare input
        input_text = f"User: {question}\nAssistant:"
        inputs = self.model.text_tokenizer(
            input_text,
            return_tensors="pt",
            max_length=50,
            truncation=True
        )
        
        # Set the current input text for the model to use in WASM context extraction
        self.model._current_input_text = question
        
        # Forward pass with WASM execution
        print("🔄 Running forward pass with WASM execution...")
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                execute_wasm=True
            )
        
        # Show WASM calculations
        execution_results = outputs.get('execution_results', [])
        if execution_results:
            print("\n🔧 WASM Calculations During Forward Pass:")
            print("-" * 40)
            
            layer_names = [3, 7, 11]
            for i, result in enumerate(execution_results):
                layer = layer_names[i] if i < len(layer_names) else f"L{i}"
                
                if result and result.get('success'):
                    computed = result.get('result')
                    if computed is not None:
                        if isinstance(computed, (int, float)):
                            print(f"   Layer {layer:2}: {float(computed):12.6f}")
                        else:
                            print(f"   Layer {layer:2}: {str(computed)[:20]}")
                    else:
                        print(f"   Layer {layer:2}: SUCCESS - None result")
                else:
                    error = result.get('error', 'Failed') if result else 'No result'
                    error_str = str(error)[:50] if error else 'Unknown error'
                    print(f"   Layer {layer:2}: ERROR - {error_str}...")
            
            print("-" * 40)
            
            # Show successful computations
            successful_results = [r for r in execution_results if r and r.get('success')]
            if successful_results:
                print(f"\n✨ Computed {len(successful_results)} results during reasoning!")
                
                # Find the most reasonable result (closest to expected answer patterns)
                results = [r['result'] for r in successful_results]
                print(f"   Possible answers: {results}")
                
                # Simple heuristic: pick the most "reasonable" result
                # For math problems, often the middle-range values are more plausible
                reasonable_result = self._pick_best_result(results, question)
                if reasonable_result is not None:
                    print(f"   🎯 Best computation: {reasonable_result}")
        else:
            print("❌ No WASM execution occurred")
        
        print("\n" + "=" * 60)
    
    def _pick_best_result(self, results, question):
        """Simple heuristic to pick the most reasonable result."""
        if not results:
            return None
        
        # For basic arithmetic, reasonable answers are usually positive and not too extreme
        reasonable_results = []
        for result in results:
            if isinstance(result, (int, float)):
                # Filter out extreme values
                if -1000 < result < 10000:
                    reasonable_results.append(result)
        
        if not reasonable_results:
            return results[0]  # Fallback to first result
        
        # For addition/multiplication, prefer larger positive values
        if any(word in question.lower() for word in ['add', '+', 'plus', 'sum']):
            return max(reasonable_results)
        elif any(word in question.lower() for word in ['multiply', '×', '*', 'times']):
            return max(reasonable_results)
        elif any(word in question.lower() for word in ['divide', '÷', '/', 'divided']):
            return min([r for r in reasonable_results if r > 0], default=reasonable_results[0])
        else:
            # Default: pick middle value
            reasonable_results.sort()
            return reasonable_results[len(reasonable_results)//2]
    
    def chat_loop(self):
        """Main chat loop."""
        print("🎮 Interactive WASM WorldModel Chat")
        print("=" * 50)
        print("Ask me mathematical questions and watch the WASM")
        print("computations happen during my reasoning process!")
        print()
        print("Examples:")
        print("• 'Calculate 15 + 27'")
        print("• 'What is 8 × 9?'") 
        print("• 'Compute 144 ÷ 12'")
        print("• 'What's 25 - 7?'")
        print()
        print("Type 'quit' to exit")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n💬 Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thanks for chatting with the WASM WorldModel!")
                    break
                
                if not user_input:
                    continue
                
                print()
                self.process_question(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error processing question: {e}")

def main():
    """Main function."""
    import sys
    
    # Check for command line arguments for non-interactive mode
    if len(sys.argv) > 1:
        # Non-interactive mode: process single question
        question = " ".join(sys.argv[1:])
        try:
            chat = WASMChat()
            chat.process_question(question)
        except Exception as e:
            print(f"❌ Failed to process question: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Interactive mode
        try:
            chat = WASMChat()
            chat.chat_loop()
        except Exception as e:
            print(f"❌ Failed to start chat: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()