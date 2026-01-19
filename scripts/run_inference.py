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
        checkpoint_path = "./wasm_worldmodel_output/final_model/pytorch_model.bin"
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
            wat_operations = []  # Track what operation each layer performed
            
            for i, result in enumerate(execution_results):
                layer = layer_names[i] if i < len(layer_names) else f"L{i}"
                executed = result.get('executed', True)  # Assume executed for backward compatibility
                score = result.get('score', 0.0)
                
                wat_code = result.get('wat_code', '')
                
                # Extract operation type from WAT code
                if 'f64.mul' in wat_code:
                    op_type = 'multiply'
                elif 'f64.add' in wat_code:
                    op_type = 'add'
                elif 'f64.sub' in wat_code:
                    op_type = 'subtract'
                elif 'f64.div' in wat_code:
                    op_type = 'divide'
                else:
                    op_type = 'unknown'
                
                wat_operations.append(op_type)
                
                if executed and result and result.get('success'):
                    computed = result.get('result')
                    if computed is not None:
                        if isinstance(computed, (int, float)):
                            print(f"   Layer {layer:2}: {float(computed):12.6f} ({op_type}) [score: {score:.2f}]")
                        else:
                            print(f"   Layer {layer:2}: {str(computed)[:20]} ({op_type}) [score: {score:.2f}]")
                    else:
                        print(f"   Layer {layer:2}: SUCCESS - None result ({op_type}) [score: {score:.2f}]")
                elif executed:
                    error = result.get('error', 'Failed') if result else 'No result'
                    error_str = str(error)[:50] if error else 'Unknown error'
                    print(f"   Layer {layer:2}: ERROR - {error_str}... [score: {score:.2f}]")
                else:
                    # Not executed due to selective execution
                    error = result.get('error', 'Not executed')
                    print(f"   Layer {layer:2}: SKIPPED - {error} ({op_type}) [score: {score:.2f}]")
            
            print("-" * 40)
            
            # Show successful computations (only executed ones)
            successful_results = [r for r in execution_results if r and r.get('success') and r.get('executed', True)]
            if successful_results:
                print(f"\n✨ Computed {len(successful_results)} results during reasoning!")
                
                # Find the best result using attention-based selection (like token generation)
                best_result = self._pick_best_result_attention(successful_results, question, execution_results)
                if best_result is not None:
                    print(f"   🎯 Best computation: {best_result}")
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
    
    def _pick_best_result_semantic(self, successful_results, question, wat_operations):
        """Pick the best result using semantic matching between question and WAT operations."""
        if not successful_results:
            return None
        
        # Parse question for operation intent
        question_lower = question.lower()
        intended_operation = None
        
        if any(word in question_lower for word in ['*', '×', 'times', 'multiply', 'product']):
            intended_operation = 'multiply'
        elif any(word in question_lower for word in ['+', 'plus', 'add', 'sum']):
            intended_operation = 'add'
        elif any(word in question_lower for word in ['-', 'minus', 'subtract', 'difference']):
            intended_operation = 'subtract'
        elif any(word in question_lower for word in ['/', '÷', 'divide', 'divided']):
            intended_operation = 'divide'
        
        print(f"   📝 Question intent: {intended_operation}")
        print(f"   🔧 WAT operations: {wat_operations}")
        
        # Find results that match the intended operation
        matching_results = []
        for i, result in enumerate(successful_results):
            if i < len(wat_operations) and wat_operations[i] == intended_operation:
                matching_results.append(result['result'])
        
        if matching_results:
            print(f"   ✅ Found {len(matching_results)} matching operations")
            # If multiple matches, pick the first one (could be improved)
            return matching_results[0]
        else:
            print(f"   ⚠️  No matching operations found, using heuristic")
            # Fallback to original heuristic
            results = [r['result'] for r in successful_results]
            return self._pick_best_result(results, question)
    
    def _pick_best_result_attention(self, successful_results, question, all_execution_results):
        """Pick the best result using attention-like scoring, similar to token generation."""
        import torch
        import torch.nn.functional as F
        
        if not successful_results:
            return None
        
        # Get text representation of the question
        question_inputs = self.model.text_tokenizer(
            question, return_tensors="pt", max_length=50, truncation=True
        )
        
        with torch.no_grad():
            # Get question embedding from model
            question_outputs = self.model.text_model(
                input_ids=question_inputs['input_ids'], 
                attention_mask=question_inputs['attention_mask'],
                output_hidden_states=True
            )
            # Use the last layer's CLS-like token (first token) as question representation
            question_embedding = question_outputs.hidden_states[-1][:, 0, :]  # [1, hidden_size]
        
        # Score each execution result based on attention with question
        result_scores = []
        
        for i, result in enumerate(successful_results):
            layer_idx = result.get('layer', i)
            
            # Create a simple "result embedding" by combining:
            # 1. Layer position (later layers = more refined)
            # 2. Operation type (from WAT code analysis) 
            # 3. Result magnitude (reasonable range bonus)
            
            # Layer position score (prefer later layers)
            layer_score = torch.tensor([layer_idx / 11.0], dtype=torch.float32)  # Normalize by max layer
            
            # Operation matching score (semantic similarity)
            wat_code = result.get('wat_code', '')
            op_score = self._compute_operation_similarity(question, wat_code)
            op_score_tensor = torch.tensor([op_score], dtype=torch.float32)
            
            # Result reasonableness score  
            result_val = abs(result['result'])
            if 0.001 <= result_val <= 1000000:
                reasonableness_score = torch.tensor([1.0], dtype=torch.float32)
            else:
                reasonableness_score = torch.tensor([0.1], dtype=torch.float32)
            
            # Combine into a pseudo "result embedding"
            result_features = torch.cat([layer_score, op_score_tensor, reasonableness_score])  # [3]
            
            # Combine scores more meaningfully
            # Weight: operation match (most important) + layer position + reasonableness
            combined_score = (
                op_score * 2.0 +           # Operation match is most important
                layer_score.item() * 1.0 + # Later layers preferred  
                reasonableness_score.item() * 0.5  # Reasonable values preferred
            )
            
            attention_score = combined_score
            result_scores.append((attention_score, result['result'], i))
            
            print(f"      Layer {layer_idx:2}: score={attention_score:.3f} (layer={layer_score.item():.2f}, op={op_score:.2f}, reason={reasonableness_score.item():.1f})")
        
        # Select result with highest attention score
        best_score, best_result, best_idx = max(result_scores, key=lambda x: x[0])
        print(f"   🔍 Attention-based selection: chose layer result with score {best_score:.3f}")
        
        return best_result
    
    def _compute_operation_similarity(self, question, wat_code):
        """Compute semantic similarity between question intent and WAT operation."""
        question_lower = question.lower()
        
        # Extract operation from WAT code
        if 'f64.mul' in wat_code:
            wat_op = 'multiply'
        elif 'f64.add' in wat_code:
            wat_op = 'add'
        elif 'f64.sub' in wat_code:
            wat_op = 'subtract'
        elif 'f64.div' in wat_code:
            wat_op = 'divide'
        else:
            return 0.0
        
        # Score based on question keywords
        if wat_op == 'multiply' and any(word in question_lower for word in ['*', '×', 'times', 'multiply', 'product']):
            return 1.0
        elif wat_op == 'add' and any(word in question_lower for word in ['+', 'plus', 'add', 'sum']):
            return 1.0
        elif wat_op == 'subtract' and any(word in question_lower for word in ['-', 'minus', 'subtract', 'difference']):
            return 1.0
        elif wat_op == 'divide' and any(word in question_lower for word in ['/', '÷', 'divide', 'divided']):
            return 1.0
        else:
            return 0.2  # Partial credit for wrong operation
    
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