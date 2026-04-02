#!/usr/bin/env python3
"""
BluePrint WorldModel Training Script with Geometric Optimization Option
===================================

Trains a model on BluePrint methodology using the thinking → blueprint token pattern.
Automatically scans training/datasets/ for JSONL files and creates progressive curriculum.
Optionally implements geometric optimization techniques from ShapeOfThought to accelerate training.

This creates a model that:
1. Generates <thinking> tokens for problem understanding
2. Generates <blueprint> tokens with proper BluePrint notation
3. Follows the collaborative specification framework

Based on docs/worldmodel-blueprint-plan.md Phase 1 implementation.
"""

import torch
from pathlib import Path
import sys
import logging
import argparse
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerControl,
    TrainerState
)
from peft import PeftModel, LoraConfig, get_peft_model
from torch.utils.data import DataLoader
import time
import subprocess
import json
from typing import List, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from training.blueprint_dataset import load_blueprint_datasets, validate_blueprint_syntax


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardware detection
print("=== BluePrint WorldModel Training with Geometric Optimization Option ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name()
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name}")
    print(f"   Memory: {total_memory:.1f}GB")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    print("❌ Using CPU")




def test_generation(model_path: str, test_query: str = "Design a temperature conversion service"):
    """Test the trained model with a sample query."""
    logger.info(f"🧪 Testing model generation from {model_path}...")

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer_test(model_path)

    # Format input
    input_text = f"User: {test_query}\n\nAssistant: "

    # Tokenize and move to device
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        # Add custom stop tokens
        stop_tokens = ["</blueprint>", "<blueprint>"]
        stop_token_ids = []
        for stop_token in stop_tokens:
            token_ids = tokenizer.encode(stop_token, add_special_tokens=False)
            if token_ids:
                stop_token_ids.extend(token_ids)

        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=1536,  # Full context usage - generous limit
            temperature=0.7,      # Balanced creativity/consistency
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,  # Light repetition penalty
            no_repeat_ngram_size=3   # Moderate n-gram prevention
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract only the assistant's response
    response_start = generated_text.find("Assistant:")
    if response_start != -1:
        response = generated_text[response_start + len("Assistant:"):].strip()
    else:
        # Fallback if Assistant: marker not found
        response = generated_text[len(input_text):]

    logger.info(f"📋 Test Query: {test_query}")
    logger.info(f"📝 Generated Response:")
    logger.info(f"---")
    logger.info(response)
    logger.info(f"---")

    # Validate format
    is_valid, errors = validate_blueprint_syntax(response)

    logger.info(f"✅ Validation:")
    logger.info(f"   Format valid: {is_valid}")
    if errors:
        for error in errors:
            logger.info(f"   ❌ {error}")


class ThermalMonitor:
    """Monitor GPU temperature and provide thermal throttling capabilities."""
    
    def __init__(self, max_temp: float = 99.0, throttle_temp: float = 85.0, cooldown_temp: float = 80.0):
        """
        Initialize thermal monitor.
        
        Args:
            max_temp: Maximum temperature before emergency stop (°C)
            throttle_temp: Temperature to start throttling (°C) 
            cooldown_temp: Temperature to resume normal operation (°C)
        """
        self.max_temp = max_temp
        self.throttle_temp = throttle_temp
        self.cooldown_temp = cooldown_temp
        self.is_throttled = False
        self.throttle_start_time = None
        
        # Detect GPU type
        self.gpu_type = self._detect_gpu_type()
        logger.info(f"🌡️ Thermal monitor initialized for {self.gpu_type} GPU")
        logger.info(f"   Max temp: {max_temp}°C | Throttle: {throttle_temp}°C | Cooldown: {cooldown_temp}°C")
    
    def _detect_gpu_type(self) -> str:
        """Detect if using NVIDIA or AMD GPU."""
        try:
            # Try ROCm first
            result = subprocess.run(['rocm-smi', '--showtemp'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return "AMD"
        except:
            pass
        
        try:
            # Try nvidia-smi
            result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return "NVIDIA"
        except:
            pass
        
        logger.warning("⚠️ Could not detect GPU type, thermal monitoring disabled")
        return "UNKNOWN"
    
    def get_gpu_temperature(self) -> Optional[float]:
        """Get current GPU temperature in Celsius."""
        try:
            if self.gpu_type == "AMD":
                result = subprocess.run(['rocm-smi', '--showtemp', '--json'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    # Extract junction temperature as it's typically the hottest
                    temp_str = data.get("card0", {}).get("Temperature (Sensor junction) (C)", "0.0")
                    return float(temp_str.strip())
            
            elif self.gpu_type == "NVIDIA":
                result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return float(result.stdout.strip())
                    
        except Exception as e:
            logger.warning(f"⚠️ Failed to get GPU temperature: {e}")
        
        return None
    
    def check_thermal_state(self) -> Dict[str, any]:
        """
        Check thermal state and return throttling recommendations.
        
        Returns:
            dict with keys: temperature, should_throttle, should_pause, emergency_stop
        """
        temp = self.get_gpu_temperature()
        
        if temp is None:
            return {
                "temperature": None,
                "should_throttle": False, 
                "should_pause": False,
                "emergency_stop": False,
                "status": "monitoring_disabled"
            }
        
        # Emergency stop
        if temp >= self.max_temp:
            return {
                "temperature": temp,
                "should_throttle": True,
                "should_pause": True,
                "emergency_stop": True,
                "status": f"emergency_stop_temp_{temp:.1f}C"
            }
        
        # Start throttling
        if temp >= self.throttle_temp and not self.is_throttled:
            self.is_throttled = True
            self.throttle_start_time = time.time()
            return {
                "temperature": temp,
                "should_throttle": True,
                "should_pause": False,
                "emergency_stop": False,
                "status": f"throttling_started_{temp:.1f}C"
            }
        
        # Continue throttling
        if self.is_throttled and temp > self.cooldown_temp:
            return {
                "temperature": temp,
                "should_throttle": True,
                "should_pause": False,
                "emergency_stop": False,
                "status": f"throttling_active_{temp:.1f}C"
            }
        
        # Stop throttling
        if self.is_throttled and temp <= self.cooldown_temp:
            throttle_duration = time.time() - self.throttle_start_time if self.throttle_start_time else 0
            self.is_throttled = False
            self.throttle_start_time = None
            logger.info(f"🌡️ Thermal throttling ended after {throttle_duration:.1f}s - temp: {temp:.1f}°C")
            return {
                "temperature": temp,
                "should_throttle": False,
                "should_pause": False,
                "emergency_stop": False,
                "status": f"throttling_ended_{temp:.1f}C"
            }
        
        # Normal operation
        return {
            "temperature": temp,
            "should_throttle": False,
            "should_pause": False,
            "emergency_stop": False,
            "status": f"normal_{temp:.1f}C"
        }
    
    def wait_for_cooldown(self, check_interval: float = 5.0) -> None:
        """Wait for GPU to cool down below throttle temperature."""
        logger.info(f"🌡️ Waiting for GPU to cool down below {self.cooldown_temp}°C...")
        
        while True:
            temp = self.get_gpu_temperature()
            if temp is None:
                logger.warning("⚠️ Cannot monitor temperature during cooldown")
                break
                
            if temp <= self.cooldown_temp:
                logger.info(f"🌡️ GPU cooled down to {temp:.1f}°C - resuming training")
                break
                
            logger.info(f"🌡️ GPU temp: {temp:.1f}°C - waiting {check_interval}s...")
            time.sleep(check_interval)


class ThermalCallback(TrainerCallback):
    """Thermal monitoring callback for HuggingFace trainer."""
    
    def __init__(self, thermal_monitor: ThermalMonitor):
        self.thermal_monitor = thermal_monitor
        self.last_temp_check = 0
        self.check_interval = 10  # Check temperature every 10 steps
    
    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Check thermal state at the beginning of each training step."""
        # Only check temperature every N steps to avoid overhead
        if state.global_step % self.check_interval != 0:
            return
            
        thermal_state = self.thermal_monitor.check_thermal_state()
        
        if thermal_state["emergency_stop"]:
            logger.error(f"🚨 EMERGENCY STOP: GPU temperature {thermal_state['temperature']:.1f}°C exceeds maximum {self.thermal_monitor.max_temp}°C")
            logger.error("Training stopped to prevent hardware damage!")
            control.should_training_stop = True
            return control
        
        if thermal_state["should_throttle"]:
            if thermal_state["status"].startswith("throttling_started"):
                logger.warning(f"🌡️ THERMAL THROTTLING: GPU temperature {thermal_state['temperature']:.1f}°C - reducing training intensity")
            
            # Aggressive throttling based on temperature
            current_temp = thermal_state['temperature']
            if current_temp >= 95.0:
                # Very hot - pause for cooldown
                logger.warning(f"🌡️ HIGH TEMPERATURE: {current_temp:.1f}°C - pausing for cooldown")
                self.thermal_monitor.wait_for_cooldown()
            elif current_temp >= 90.0:
                # Hot - long delay
                logger.info(f"🌡️ Throttling heavily: {current_temp:.1f}°C - waiting 10s")
                time.sleep(10.0)
            elif current_temp >= 85.0:
                # Warm - moderate delay
                logger.info(f"🌡️ Throttling moderately: {current_temp:.1f}°C - waiting 5s")
                time.sleep(5.0)
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Add temperature info to training logs."""
        if logs is not None:
            thermal_state = self.thermal_monitor.check_thermal_state()
            if thermal_state["temperature"] is not None:
                logs["gpu_temp"] = thermal_state["temperature"]
                logs["thermal_throttling"] = thermal_state["should_throttle"]


class GeometricOptimizer:
    """
    Implements geometric optimization techniques from ShapeOfThought:
    1. Vector space jumps (extrapolation along gradient direction)
    2. Hypersphere search (neighborhood exploration)
    """

    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        self.learning_rate = learning_rate
        self.previous_params = None
        self.jump_factor = 10.0
        self.sphere_radius = 1.0
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def to_vector(self):
        """Convert model parameters to a single flattened vector."""
        params = []
        for param in self.model.parameters():
            if param.requires_grad:  # Only include trainable parameters
                params.append(param.data.flatten())
        return torch.cat(params) if params else torch.tensor([])

    def from_vector(self, vector):
        """Load parameters from a flattened vector back to the model."""
        if len(vector) == 0:
            return

        vector = vector.to(device)
        idx = 0
        for param in self.model.parameters():
            if param.requires_grad:  # Only update trainable parameters
                param_size = param.numel()
                param.data = vector[idx:idx + param_size].reshape(param.shape)
                idx += param_size

    def capture_gradient_direction(self, batch_inputs, batch_labels, loss_fn):
        """Capture the gradient direction from a training step."""
        # Store current parameters
        current_params = self.to_vector().clone()

        # Perform a training step to compute gradient
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(**batch_inputs)

        # Reshape logits and labels for loss calculation
        # outputs.logits shape: [batch_size, seq_len, vocab_size]
        # Need to reshape for cross entropy: [batch_size*seq_len, vocab_size] and [batch_size*seq_len]
        shift_logits = outputs.logits.view(-1, outputs.logits.size(-1))  # [batch*seq, vocab]
        shift_labels = batch_labels.view(-1)  # [batch*seq]

        loss = loss_fn(shift_logits, shift_labels)

        # Backward pass
        loss.backward()

        # Apply optimizer step (but store the direction)
        self.optimizer.step()

        # Calculate gradient direction
        new_params = self.to_vector()
        gradient_direction = new_params - current_params

        # Restore original parameters for geometric optimization
        self.from_vector(current_params)

        return gradient_direction

    def vector_jump_step(self, batch_inputs, batch_labels, loss_fn, gradient_direction):
        """Perform a vector space jump along the gradient direction."""
        if torch.norm(gradient_direction) < 1e-10:  # Skip if gradient is too small
            return False, 0.0

        # Store current parameters
        original_params = self.to_vector().clone()
        current_loss = self._evaluate_loss(batch_inputs, batch_labels, loss_fn)

        # Generate candidates
        candidates = []

        # Gradient direction candidate
        jump_candidate = original_params + self.jump_factor * gradient_direction
        candidates.append(jump_candidate)

        # Random direction candidates (based on gradient magnitude)
        grad_norm = torch.norm(gradient_direction)
        for i in range(2):  # Fewer random candidates to save computation
            random_dir = torch.randn_like(original_params)
            random_norm = torch.norm(random_dir)
            if random_norm > 1e-10:  # Prevent division by zero
                random_dir = random_dir / random_norm  # Normalize
                random_candidate = original_params + grad_norm * self.jump_factor * random_dir
                candidates.append(random_candidate)

        # Evaluate candidates
        best_loss = current_loss
        best_candidate = None
        jump_accepted = False

        for candidate in candidates:
            self.from_vector(candidate)
            candidate_loss = self._evaluate_loss(batch_inputs, batch_labels, loss_fn)

            # Check for NaN or infinite loss
            if not (torch.isnan(torch.tensor(candidate_loss)) or torch.isinf(torch.tensor(candidate_loss))):
                if candidate_loss < best_loss:
                    best_loss = candidate_loss
                    best_candidate = candidate
                    jump_accepted = True

        # Apply best candidate or restore original
        if jump_accepted and best_candidate is not None:
            self.from_vector(best_candidate)
        else:
            self.from_vector(original_params)
            # Reduce jump factor if jump was rejected
            self.jump_factor = max(0.05, self.jump_factor * 0.99)

        return jump_accepted, best_loss

    def hypersphere_step(self, batch_inputs, batch_labels, loss_fn, gradient_direction):
        """Perform a hypersphere search around current parameters."""
        # Store current parameters
        original_params = self.to_vector().clone()
        current_loss = self._evaluate_loss(batch_inputs, batch_labels, loss_fn)

        # Generate candidates on hypersphere
        candidates = []

        # Gradient direction candidate on hypersphere
        if torch.norm(gradient_direction) > 1e-10:
            gradient_unit = gradient_direction / torch.norm(gradient_direction)
            gradient_candidate = original_params + self.sphere_radius * gradient_unit
            candidates.append(gradient_candidate)

        # Random directions on hypersphere
        for i in range(4):  # Fewer candidates to save computation
            random_dir = torch.randn_like(original_params)
            random_norm = torch.norm(random_dir)
            if random_norm > 1e-10:  # Prevent division by zero
                random_dir = random_dir / random_norm  # Normalize to unit vector
                random_candidate = original_params + self.sphere_radius * random_dir
                candidates.append(random_candidate)

        # Evaluate candidates
        best_loss = current_loss
        best_candidate = None
        sphere_accepted = False

        for candidate in candidates:
            self.from_vector(candidate)
            candidate_loss = self._evaluate_loss(batch_inputs, batch_labels, loss_fn)

            # Check for NaN or infinite loss
            if not (torch.isnan(torch.tensor(candidate_loss)) or torch.isinf(torch.tensor(candidate_loss))):
                if candidate_loss < best_loss:
                    best_loss = candidate_loss
                    best_candidate = candidate
                    sphere_accepted = True

        # Apply best candidate or restore original
        if sphere_accepted and best_candidate is not None:
            self.from_vector(best_candidate)
        else:
            self.from_vector(original_params)
            # Reduce sphere radius if sphere search was rejected
            self.sphere_radius = max(0.01, self.sphere_radius * 0.995)

        return sphere_accepted, best_loss

    def _evaluate_loss(self, batch_inputs, batch_labels, loss_fn):
        """Evaluate loss without gradients."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**batch_inputs)
            # Reshape logits and labels for loss calculation
            # outputs.logits shape: [batch_size, seq_len, vocab_size]
            # Need to reshape for cross entropy: [batch_size*seq_len, vocab_size] and [batch_size*seq_len]
            shift_logits = outputs.logits.view(-1, outputs.logits.size(-1))  # [batch*seq, vocab]
            shift_labels = batch_labels.view(-1)  # [batch*seq]
            loss = loss_fn(shift_logits, shift_labels)
        self.model.train()
        return loss.item()


def geometric_training_loop(model, train_dataloader, val_dataloader, epochs, learning_rate, device, thermal_monitor=None):
    """Custom training loop with geometric optimization and thermal monitoring."""
    logger.info("🚀 Starting geometric optimization training loop...")

    # Initialize geometric optimizer
    geo_optimizer = GeometricOptimizer(model, learning_rate=learning_rate)

    # Initialize thermal monitor if not provided
    if thermal_monitor is None:
        thermal_monitor = ThermalMonitor()

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Training statistics
    jump_acceptance_rate = 0.0
    sphere_acceptance_rate = 0.0
    total_jump_attempts = 0
    total_sphere_attempts = 0
    accepted_jumps = 0
    accepted_spheres = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        step_count = 0

        start_time = time.time()

        # Calculate total steps in epoch for progress tracking
        total_steps_in_epoch = len(train_dataloader)

        for step, batch in enumerate(train_dataloader):
            # Thermal monitoring - check every step
            thermal_state = thermal_monitor.check_thermal_state()
            
            if thermal_state["emergency_stop"]:
                logger.error(f"🚨 EMERGENCY STOP: GPU temperature {thermal_state['temperature']:.1f}°C exceeds maximum {thermal_monitor.max_temp}°C")
                logger.error("Training stopped to prevent hardware damage!")
                return False
            
            if thermal_state["should_throttle"]:
                if thermal_state["status"].startswith("throttling_started"):
                    logger.warning(f"🌡️ THERMAL THROTTLING: GPU temperature {thermal_state['temperature']:.1f}°C - reducing training intensity")
                
                # Aggressive throttling based on temperature
                current_temp = thermal_state['temperature']
                if current_temp >= 95.0:
                    # Very hot - pause for cooldown
                    print()  # Add newline before warning
                    logger.warning(f"🌡️ HIGH TEMPERATURE: {current_temp:.1f}°C - pausing for cooldown")
                    thermal_monitor.wait_for_cooldown()
                elif current_temp >= 90.0:
                    # Hot - long delay
                    print()  # Add newline before warning
                    logger.info(f"🌡️ Throttling heavily: {current_temp:.1f}°C - waiting 10s")
                    time.sleep(10.0)
                elif current_temp >= 85.0:
                    # Warm - moderate delay
                    print()  # Add newline before warning
                    logger.info(f"🌡️ Throttling moderately: {current_temp:.1f}°C - waiting 5s")
                    time.sleep(5.0)

            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # For causal language modeling, inputs and labels are typically the same
            # The model shifts the input internally to create labels
            batch_inputs = {k: v for k, v in batch.items() if k != 'labels'}
            batch_labels = batch['labels']

            # Capture gradient direction using standard backprop
            gradient_direction = geo_optimizer.capture_gradient_direction(
                batch_inputs, batch_labels, loss_fn
            )

            # Step 1: Vector space jump (tries to improve on standard backprop)
            jump_accepted, loss_after_jump = geo_optimizer.vector_jump_step(
                batch_inputs, batch_labels, loss_fn, gradient_direction
            )

            # Update statistics
            total_jump_attempts += 1
            if jump_accepted:
                accepted_jumps += 1

            # Step 2: Hypersphere search (tries to improve on vector jump)
            sphere_accepted, final_loss = geo_optimizer.hypersphere_step(
                batch_inputs, batch_labels, loss_fn, gradient_direction
            )

            # Update statistics
            total_sphere_attempts += 1
            if sphere_accepted:
                accepted_spheres += 1

            total_loss += final_loss
            step_count += 1

            # Print progress for every step using carriage return for live updates
            avg_loss = total_loss / step_count
            jump_rate = accepted_jumps / total_jump_attempts if total_jump_attempts > 0 else 0
            sphere_rate = accepted_spheres / total_sphere_attempts if total_sphere_attempts > 0 else 0

            # Print live progress to stdout using \r to overwrite the same line
            temp_display = f"{thermal_state['temperature']:.1f}°C" if thermal_state['temperature'] else "N/A"
            throttle_indicator = " 🔥" if thermal_state['should_throttle'] else ""
            
            progress_msg = f"\rEpoch {epoch+1}/{epochs}, Step {step+1}/{total_steps_in_epoch}, " \
                          f"Loss: {final_loss:.6f}, Avg Loss: {avg_loss:.6f}, " \
                          f"Jump Rate: {jump_rate:.3f}, " \
                          f"Sphere Rate: {sphere_rate:.3f}, " \
                          f"GPU: {temp_display}{throttle_indicator}"
            print(progress_msg, end='', flush=True)

        # Calculate epoch statistics
        avg_loss = total_loss / step_count
        epoch_time = time.time() - start_time

        # Update acceptance rates
        jump_acceptance_rate = accepted_jumps / total_jump_attempts if total_jump_attempts > 0 else 0
        sphere_acceptance_rate = accepted_spheres / total_sphere_attempts if total_sphere_attempts > 0 else 0

        # Print newline to move to next line after the live progress updates
        print()  # This moves to a new line after the \r progress updates
        logger.info(f"Epoch {epoch+1}/{epochs} completed - "
                   f"Avg Loss: {avg_loss:.6f}, "
                   f"Time: {epoch_time:.2f}s, "
                   f"Jump Acceptance: {jump_acceptance_rate:.3f}, "
                   f"Sphere Acceptance: {sphere_acceptance_rate:.3f}")

        # Validation every few epochs
        if (epoch + 1) % 5 == 0:
            val_loss = evaluate_model(model, val_dataloader, loss_fn, device)
            logger.info(f"Validation Loss after Epoch {epoch+1}: {val_loss:.6f}")

    logger.info("✅ Geometric optimization training completed!")


def evaluate_model(model, dataloader, loss_fn, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            batch_inputs = {k: v for k, v in batch.items() if k != 'labels'}
            batch_labels = batch['labels']

            outputs = model(**batch_inputs)
            # Reshape logits and labels for loss calculation
            shift_logits = outputs.logits.view(-1, outputs.logits.size(-1))  # [batch*seq, vocab]
            shift_labels = batch_labels.view(-1)  # [batch*seq]
            loss = loss_fn(shift_logits, shift_labels)

            total_loss += loss.item()
            total_samples += 1

            # Report progress every 10 validation batches
            if batch_idx % 10 == 0:
                logger.debug(f"Validation batch {batch_idx}, Loss: {loss.item():.6f}")

    return total_loss / total_samples if total_samples > 0 else float('inf')


def setup_model_and_tokenizer(model_path: str):
    """Load and prepare model and tokenizer for training."""
    logger.info(f"🧠 Loading model from {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Use float16 for LoRA
        low_cpu_mem_usage=True,
        device_map={"": 0}  # Direct device mapping
    )

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"], # Specific to Qwen models, may need adjustment for others
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Wrap the model with PEFT
    model = get_peft_model(model, lora_config)

    # Disable gradient checkpointing to resolve the RuntimeError with LoRA
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
        logger.info("✅ Gradient checkpointing: DISABLED to be compatible with LoRA")

    logger.info(f"✅ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    model.print_trainable_parameters()

    return model, tokenizer


def setup_model_and_tokenizer_test(model_path: str):
    """Load and prepare model and tokenizer for testing/inference."""
    logger.info(f"🧠 Loading model from {model_path} for testing...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try to load as a PEFT model first
    import os
    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForCausalLM

    adapter_config_path = os.path.join(model_path, "adapter_config.json")

    if os.path.exists(adapter_config_path):
        try:
            # Load the PEFT configuration to get the base model
            config = PeftConfig.from_pretrained(model_path)
            base_model_name = config.base_model_name_or_path

            # Load the base model with the correct parameters
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map={"": 0},
                ignore_mismatched_sizes=True  # Important for resized embeddings
            )

            # Load the PEFT adapter on top
            model = PeftModel.from_pretrained(base_model, model_path)
        except Exception as e:
            logger.warning(f"Failed to load as PEFT model: {e}")
            # Fallback: try loading directly (might work if it's a merged model)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map={"": 0},
                ignore_mismatched_sizes=True
            )
    else:
        # Not a PEFT model, load directly
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": 0},
            ignore_mismatched_sizes=True
        )

    logger.info(f"✅ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()

    return model, tokenizer

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="BluePrint WorldModel Training with Geometric Optimization")
    parser.add_argument("--model", required=True, help="Path to base model (e.g., Qwen/Qwen1.5-0.5B-Chat)")
    parser.add_argument("--datasets_dir", default="training/blueprint/datasets", help="Path to datasets directory")
    parser.add_argument("--output_dir", default="blueprint_model_output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--test_only", action="store_true", help="Only run generation test")
    parser.add_argument("--test_model", type=str, help="Model path for testing")
    parser.add_argument("--curriculum", choices=["foundation", "business", "technical", "domain", "advanced", "security", "complete", "all"],
                       default="all", help="Curriculum stage (default: all - trains on everything)")
    parser.add_argument("--use_geometric", action="store_true", help="Use geometric optimization techniques")
    parser.add_argument("--max_temp", type=float, default=99.0, help="Maximum GPU temperature before emergency stop (°C)")
    parser.add_argument("--throttle_temp", type=float, default=85.0, help="GPU temperature to start throttling (°C)")
    parser.add_argument("--cooldown_temp", type=float, default=80.0, help="GPU temperature to resume normal operation (°C)")
    parser.add_argument("--disable_thermal", action="store_true", help="Disable thermal monitoring")

    args = parser.parse_args()

    # Set ROCm-specific environment variable for memory allocation
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

    if args.test_only:
        test_model_path = args.test_model or args.model
        test_generation(test_model_path)
        return True

    logger.info(f"🎯 BluePrint WorldModel Training with Geometric Optimization")
    logger.info(f"   Model: {args.model}")
    logger.info(f"   Datasets: {args.datasets_dir}")
    logger.info(f"   Output: {args.output_dir}")
    logger.info(f"   Curriculum: {args.curriculum}")
    logger.info(f"   Use Geometric: {args.use_geometric}")
    logger.info(f"   Thermal Monitoring: {'Disabled' if args.disable_thermal else 'Enabled'}")

    # Initialize thermal monitor
    thermal_monitor = None
    if not args.disable_thermal:
        thermal_monitor = ThermalMonitor(
            max_temp=args.max_temp,
            throttle_temp=args.throttle_temp, 
            cooldown_temp=args.cooldown_temp
        )

    # Run training
    try:
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(args.model)

        # Load datasets
        train_dataset, val_dataset = load_blueprint_datasets(
            datasets_dir=args.datasets_dir,
            tokenizer=tokenizer,
            max_length=args.max_length,
            curriculum_stage=args.curriculum
        )

        # Resize token embeddings for special tokens
        model.resize_token_embeddings(len(tokenizer))

        if args.use_geometric:
            # Custom geometric training loop with thermal monitoring
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

            success = geometric_training_loop(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                device=device,
                thermal_monitor=thermal_monitor
            )
            
            if not success:
                logger.error("❌ Training stopped due to thermal emergency!")
                # Save model anyway to preserve training progress
                logger.info("💾 Saving model progress before exit...")
                try:
                    model.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
                    logger.info("✅ Model saved successfully despite thermal emergency")
                except Exception as save_error:
                    logger.error(f"❌ Failed to save model: {save_error}")
                return False

            # For geometric training, save the model properly with PEFT
            model.save_pretrained(args.output_dir)
        else:
            # Standard HuggingFace training
            # Calculate training parameters
            total_steps = len(train_dataset) // (args.batch_size * args.gradient_accumulation) * args.epochs
            warmup_steps = int(0.1 * total_steps)

            # Training arguments - ROCm optimized from proven script
            training_args = TrainingArguments(
                output_dir=args.output_dir,
                overwrite_output_dir=True,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation,
                learning_rate=args.learning_rate,
                weight_decay=0.01,
                warmup_steps=warmup_steps,
                lr_scheduler_type="cosine",

                # Performance monitoring
                logging_steps=50,
                eval_steps=500,
                save_steps=1000,
                eval_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,

                # ROCm optimizations from proven script
                dataloader_num_workers=4,           # Parallel loading
                dataloader_pin_memory=True,         # Memory efficiency
                dataloader_persistent_workers=True, # Reuse workers
                dataloader_drop_last=True,         # Consistent batch sizes

                # STABLE: FP16 training is recommended for LoRA
                fp16=True,
                bf16=False,                         # Disabled for ROCm stability

                # Gradient checkpointing is disabled to be compatible with LoRA
                gradient_checkpointing=False,

                # ROCm compatibility
                report_to=None,                    # Disable wandb/tensorboard
                remove_unused_columns=False,
                max_grad_norm=1.0,
            )

            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                return_tensors="pt"
            )

            # Initialize trainer with thermal callback
            callbacks = []
            if thermal_monitor is not None:
                callbacks.append(ThermalCallback(thermal_monitor))
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
                callbacks=callbacks,
            )

            # Run training
            logger.info(f"🏋️ Starting standard training...")
            trainer.train()

            # Save final model using trainer
            trainer.save_model(args.output_dir)

        # Always save tokenizer
        tokenizer.save_pretrained(args.output_dir)

        logger.info(f"🎉 Training completed!")
        logger.info(f"   Final model saved to: {args.output_dir}")

        # Test the model
        test_generation(args.output_dir)

        return True
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False



if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)