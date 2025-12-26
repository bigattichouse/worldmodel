import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import random
from collections import deque
import copy

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from ..utils.config import TrainingConfig
from ..utils.logging import get_logger
from ..training.dataGenerator import TrainingExample
from ..execution.vmInterface import VMInterface
from ..execution.requirementValidator import RequirementValidator
from ..core.tagParser import TagParser
from .sftTrainer import SFTTrainer, SFTConfig


@dataclass
class RLConfig:
    model_name: str = "google/gemma-2-2b-it"
    max_sequence_length: int = 2048
    learning_rate: float = 1e-5
    batch_size: int = 4
    num_episodes: int = 1000
    max_steps_per_episode: int = 20
    discount_factor: float = 0.99
    epsilon_start: float = 0.9
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    target_update_freq: int = 100
    save_freq: int = 500
    output_dir: str = "./rl_checkpoints"
    use_wandb: bool = False
    wandb_project: str = "worldmodel_rl"
    reward_scaling: float = 1.0
    penalty_scaling: float = 1.0

@dataclass 
class RLState:
    input_text: str
    context: str
    step: int
    max_steps: int
    requirements: List[str]
    previous_actions: List[str]
    
    def to_tensor_input(self, tokenizer) -> Dict[str, torch.Tensor]:
        """Convert state to model input format."""
        prompt = f"<user>\n{self.input_text}\n</user>\n<assistant>\n"
        if self.context:
            prompt = self.context + "\n" + prompt
        
        encoding = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - 512  # Leave room for generation
        )
        
        return encoding

@dataclass
class RLAction:
    action_type: str  # "thinking", "modeling", "requirements", "completion"
    content: str
    confidence: float = 1.0
    
    def to_text(self) -> str:
        """Convert action to text format."""
        if self.action_type == "thinking":
            return f"<think>{self.content}</think>"
        elif self.action_type == "modeling":
            return f"<model>{self.content}</model>"
        elif self.action_type == "requirements":
            return f"<requires>{self.content}</requires>"
        else:
            return self.content

@dataclass
class RLExperience:
    state: RLState
    action: RLAction
    reward: float
    next_state: Optional[RLState]
    done: bool
    metadata: Dict[str, Any]

@dataclass
class RLEpisodeResult:
    total_reward: float
    steps_taken: int
    success: bool
    final_output: str
    execution_results: List[Dict[str, Any]]
    violations: List[str]
    metadata: Dict[str, Any]

class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, experience: RLExperience):
        """Save an experience."""
        self.memory.append(experience)
    
    def sample(self, batch_size: int) -> List[RLExperience]:
        """Sample a batch of experiences."""
        return random.sample(self.memory, min(batch_size, len(self.memory)))
    
    def __len__(self):
        return len(self.memory)

class WorldModelEnvironment:
    def __init__(self, config: RLConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.vm_interface = VMInterface()
        self.requirement_validator = RequirementValidator()
        self.tag_parser = TagParser()
        
        # Episode state
        self.current_episode = None
        self.step_count = 0
        
    async def reset(self, example: TrainingExample) -> RLState:
        """Reset environment with a new example."""
        self.current_episode = example
        self.step_count = 0
        
        # Extract requirements if any exist in target output
        requirements = []
        if "<requires>" in example.target_output:
            requires_tags = self.tag_parser.parse(example.target_output).requires_tags
            requirements = [tag.content for tag in requires_tags]
        
        initial_state = RLState(
            input_text=example.input_text,
            context="",
            step=0,
            max_steps=self.config.max_steps_per_episode,
            requirements=requirements,
            previous_actions=[]
        )
        
        return initial_state
    
    async def step(self, state: RLState, action: RLAction) -> Tuple[RLState, float, bool, Dict[str, Any]]:
        """Execute an action and return new state, reward, done flag, and info."""
        
        self.step_count += 1
        reward = 0.0
        done = False
        info = {}
        
        # Update context with action
        new_context = state.context + action.to_text()
        new_actions = state.previous_actions + [action.to_text()]
        
        # Check if this is a completion action or max steps reached
        if action.action_type == "completion" or self.step_count >= state.max_steps:
            done = True
            
            # Calculate final reward based on execution and requirements
            final_reward, exec_info = await self._calculate_final_reward(
                state, new_context, action
            )
            reward = final_reward
            info.update(exec_info)
        
        else:
            # Intermediate reward for good structure
            reward = self._calculate_intermediate_reward(state, action)
        
        # Create new state
        new_state = RLState(
            input_text=state.input_text,
            context=new_context,
            step=self.step_count,
            max_steps=state.max_steps,
            requirements=state.requirements,
            previous_actions=new_actions
        ) if not done else None
        
        return new_state, reward, done, info
    
    def _calculate_intermediate_reward(self, state: RLState, action: RLAction) -> float:
        """Calculate intermediate reward for action structure."""
        reward = 0.0
        
        # Reward for proper action sequencing
        if action.action_type == "thinking":
            # Good to think before acting
            if not any("<think>" in act for act in state.previous_actions):
                reward += 0.1  # First thinking is good
            else:
                reward -= 0.05  # Too much thinking
        
        elif action.action_type == "modeling":
            # Should think before modeling
            has_thinking = any("<think>" in act for act in state.previous_actions)
            if has_thinking:
                reward += 0.2  # Good sequence
            else:
                reward -= 0.1  # No thinking first
        
        elif action.action_type == "requirements":
            # Requirements should come after modeling
            has_modeling = any("<model>" in act for act in state.previous_actions)
            if has_modeling:
                reward += 0.1
            else:
                reward -= 0.1
        
        # Penalize repetitive actions
        if len(state.previous_actions) > 0 and action.to_text() == state.previous_actions[-1]:
            reward -= 0.2
        
        return reward * self.config.reward_scaling
    
    async def _calculate_final_reward(self, state: RLState, context: str, 
                                    action: RLAction) -> Tuple[float, Dict[str, Any]]:
        """Calculate final reward based on execution results."""
        reward = 0.0
        info = {
            'execution_success': False,
            'requirement_satisfaction': 0.0,
            'violations': [],
            'execution_output': ""
        }
        
        # Parse the complete response
        full_response = context + action.to_text()
        parse_result = self.tag_parser.parse(full_response)
        
        # Reward for proper structure
        if parse_result.think_tags:
            reward += 0.2  # Has thinking
        if parse_result.model_tags:
            reward += 0.3  # Has code
        if parse_result.requires_tags:
            reward += 0.1  # Has requirements
        
        # Execute code if present
        execution_results = []
        if parse_result.model_tags:
            for model_tag in parse_result.model_tags:
                try:
                    # Extract language and code
                    language = "python"  # Default
                    if model_tag.language:
                        language = model_tag.language
                    
                    # Execute the code
                    exec_result = await self.vm_interface.execute_code(
                        language=language,
                        code=model_tag.content,
                        timeout=10
                    )
                    
                    execution_results.append({
                        'language': language,
                        'success': exec_result.success,
                        'output': exec_result.output,
                        'error': exec_result.error
                    })
                    
                    if exec_result.success:
                        reward += 0.5  # Successful execution
                        info['execution_success'] = True
                        info['execution_output'] = exec_result.output
                    else:
                        reward -= 0.3  # Failed execution
                        info['violations'].append(f"Execution failed: {exec_result.error}")
                
                except Exception as e:
                    reward -= 0.2
                    info['violations'].append(f"Execution error: {str(e)}")
        
        # Validate requirements if present
        if state.requirements and parse_result.model_tags and parse_result.requires_tags:
            try:
                # Use requirement validator
                if execution_results:
                    from ..execution.vmInterface import ExecutionResult
                    exec_result = ExecutionResult(
                        success=execution_results[0]['success'],
                        output=execution_results[0]['output'],
                        error=execution_results[0]['error'],
                        execution_time=1.0,
                        return_code=0 if execution_results[0]['success'] else 1,
                        language=execution_results[0]['language']
                    )
                    
                    validation = self.requirement_validator.validate_execution(
                        exec_result, parse_result.model_tags[0], parse_result.requires_tags
                    )
                    
                    # Reward based on requirement satisfaction
                    satisfaction = validation.accuracy_score
                    reward += satisfaction * 0.4  # Up to 0.4 for perfect satisfaction
                    info['requirement_satisfaction'] = satisfaction
                    
                    # Penalize violations
                    if validation.violations:
                        violation_penalty = len(validation.violations) * 0.1
                        reward -= violation_penalty * self.config.penalty_scaling
                        info['violations'].extend([v.description for v in validation.violations])
                
            except Exception as e:
                reward -= 0.1
                info['violations'].append(f"Validation error: {str(e)}")
        
        return reward * self.config.reward_scaling, info

class DQNPolicy(nn.Module):
    """Deep Q-Network for action selection."""
    
    def __init__(self, vocab_size: int, hidden_size: int = 512, num_actions: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        
        # Simple embedding and processing layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.action_head = nn.Linear(hidden_size, num_actions)
        self.content_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """Forward pass to get action Q-values."""
        # Embed tokens
        embedded = self.embedding(input_ids)
        
        # Process with LSTM
        if attention_mask is not None:
            # Pack padded sequences for efficiency
            lengths = attention_mask.sum(dim=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, _) = self.lstm(packed)
        else:
            lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use final hidden state for action prediction
        final_hidden = hidden[-1]  # Last layer's hidden state
        
        # Predict action type
        action_q_values = self.action_head(final_hidden)
        
        return action_q_values

class RLTrainer:
    def __init__(self, config: RLConfig, base_model_path: str = None):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize environment
        self.env = WorldModelEnvironment(config)
        
        # Initialize models
        self.base_model = None
        self.policy_net = None
        self.target_net = None
        self.tokenizer = None
        
        # Training state
        self.replay_memory = ReplayMemory(config.memory_size)
        self.episode_results: List[RLEpisodeResult] = []
        self.epsilon = config.epsilon_start
        self.optimizer = None
        
        # Load base model if provided
        if base_model_path:
            self._load_base_model(base_model_path)
        
        self.logger.info("RL Trainer initialized")
    
    def _load_base_model(self, model_path: str):
        """Load the base SFT model."""
        try:
            sft_trainer = SFTTrainer(SFTConfig())
            sft_trainer.load_model(model_path)
            
            self.base_model = sft_trainer.model
            self.tokenizer = sft_trainer.tokenizer
            
            # Initialize DQN policy
            vocab_size = len(self.tokenizer)
            self.policy_net = DQNPolicy(vocab_size)
            self.target_net = DQNPolicy(vocab_size)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Initialize optimizer
            self.optimizer = torch.optim.Adam(
                self.policy_net.parameters(),
                lr=self.config.learning_rate
            )
            
            self.logger.info(f"Loaded base model from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading base model: {e}")
            raise
    
    def _select_action(self, state: RLState) -> RLAction:
        """Select action using epsilon-greedy policy."""
        
        if random.random() < self.epsilon:
            # Random action
            action_types = ["thinking", "modeling", "requirements", "completion"]
            action_type = random.choice(action_types)
            
            if action_type == "thinking":
                content = self._generate_thinking_content(state)
            elif action_type == "modeling":
                content = self._generate_model_content(state)
            elif action_type == "requirements":
                content = self._generate_requirements_content(state)
            else:
                content = ""  # Completion
            
            return RLAction(
                action_type=action_type,
                content=content,
                confidence=0.5  # Random action has low confidence
            )
        
        else:
            # Use policy network
            return self._policy_action(state)
    
    def _policy_action(self, state: RLState) -> RLAction:
        """Select action using policy network."""
        if self.policy_net is None:
            # Fallback to random if no policy
            return self._select_action(state)  # Will hit random branch
        
        try:
            # Convert state to model input
            model_input = state.to_tensor_input(self.tokenizer)
            
            with torch.no_grad():
                q_values = self.policy_net(
                    model_input['input_ids'],
                    model_input.get('attention_mask')
                )
                
                # Select best action
                action_idx = q_values.argmax().item()
                action_types = ["thinking", "modeling", "requirements", "completion"]
                action_type = action_types[action_idx]
                
                confidence = torch.softmax(q_values, dim=-1)[action_idx].item()
                
                # Generate content for the action
                if action_type == "thinking":
                    content = self._generate_thinking_content(state)
                elif action_type == "modeling":
                    content = self._generate_model_content(state)
                elif action_type == "requirements":
                    content = self._generate_requirements_content(state)
                else:
                    content = ""
                
                return RLAction(
                    action_type=action_type,
                    content=content,
                    confidence=confidence
                )
        
        except Exception as e:
            self.logger.warning(f"Policy action failed: {e}, falling back to random")
            return self._select_action(state)  # Fallback to random
    
    def _generate_thinking_content(self, state: RLState) -> str:
        """Generate thinking content using base model."""
        if self.base_model is None:
            return f"I need to solve: {state.input_text}"
        
        try:
            prompt = f"<user>\n{state.input_text}\n</user>\n<assistant>\n<think>"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.base_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    eos_token_id=self.tokenizer.encode("</think>")[0]
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the thinking content
            if "<think>" in response:
                thinking = response.split("<think>")[-1]
                if "</think>" in thinking:
                    thinking = thinking.split("</think>")[0]
                return thinking.strip()
            
            return f"I need to approach this problem: {state.input_text}"
            
        except Exception as e:
            self.logger.warning(f"Thinking generation failed: {e}")
            return f"Let me think about: {state.input_text}"
    
    def _generate_model_content(self, state: RLState) -> str:
        """Generate code content using base model."""
        if self.base_model is None:
            return "print('Hello World')"
        
        try:
            context = state.context
            prompt = f"<user>\n{state.input_text}\n</user>\n<assistant>\n{context}<model>"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.base_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.8,
                    do_sample=True,
                    eos_token_id=self.tokenizer.encode("</model>")[0]
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract code content
            if "<model>" in response:
                code = response.split("<model>")[-1]
                if "</model>" in code:
                    code = code.split("</model>")[0]
                return code.strip()
            
            return "# Generated code would go here"
            
        except Exception as e:
            self.logger.warning(f"Model generation failed: {e}")
            return "print('Generated code')"
    
    def _generate_requirements_content(self, state: RLState) -> str:
        """Generate requirements content."""
        # Simple heuristic based on input
        input_lower = state.input_text.lower()
        
        if "python" in input_lower or "calculate" in input_lower or "math" in input_lower:
            return "python:math"
        elif "javascript" in input_lower or "js" in input_lower:
            return "javascript:general"
        elif "text" in input_lower or "string" in input_lower:
            return "python:text"
        else:
            return "python:general"
    
    async def run_episode(self, example: TrainingExample) -> RLEpisodeResult:
        """Run a single episode."""
        state = await self.env.reset(example)
        total_reward = 0.0
        steps_taken = 0
        execution_results = []
        violations = []
        final_output = ""
        
        while steps_taken < self.config.max_steps_per_episode:
            # Select action
            action = self._select_action(state)
            
            # Take step
            next_state, reward, done, info = await self.env.step(state, action)
            
            total_reward += reward
            steps_taken += 1
            
            # Store experience
            experience = RLExperience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                metadata=info
            )
            self.replay_memory.push(experience)
            
            # Update final output
            final_output += action.to_text()
            
            # Collect execution results and violations
            if 'execution_output' in info and info['execution_output']:
                execution_results.append(info)
            if 'violations' in info:
                violations.extend(info['violations'])
            
            if done:
                break
            
            state = next_state
        
        success = total_reward > 0 and len(violations) == 0
        
        result = RLEpisodeResult(
            total_reward=total_reward,
            steps_taken=steps_taken,
            success=success,
            final_output=final_output,
            execution_results=execution_results,
            violations=violations,
            metadata={'example_category': example.category}
        )
        
        return result
    
    def _train_policy(self, batch_size: int = 32):
        """Train the policy network on a batch of experiences."""
        if len(self.replay_memory) < batch_size or self.policy_net is None:
            return
        
        # Sample batch
        experiences = self.replay_memory.sample(batch_size)
        
        # Prepare batch data
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for exp in experiences:
            try:
                state_input = exp.state.to_tensor_input(self.tokenizer)
                states.append(state_input['input_ids'].squeeze())
                
                # Map action type to index
                action_types = ["thinking", "modeling", "requirements", "completion"]
                action_idx = action_types.index(exp.action.action_type)
                actions.append(action_idx)
                
                rewards.append(exp.reward)
                
                if exp.next_state is not None:
                    next_state_input = exp.next_state.to_tensor_input(self.tokenizer)
                    next_states.append(next_state_input['input_ids'].squeeze())
                else:
                    next_states.append(torch.zeros_like(states[-1]))
                
                dones.append(exp.done)
                
            except Exception as e:
                self.logger.warning(f"Error processing experience: {e}")
                continue
        
        if not states:
            return
        
        # Convert to tensors
        try:
            # Pad sequences to same length
            max_len = max(s.size(0) for s in states)
            state_batch = torch.stack([
                F.pad(s, (0, max_len - s.size(0))) for s in states
            ])
            next_state_batch = torch.stack([
                F.pad(s, (0, max_len - s.size(0))) for s in next_states
            ])
            
            action_batch = torch.tensor(actions)
            reward_batch = torch.tensor(rewards, dtype=torch.float32)
            done_batch = torch.tensor(dones, dtype=torch.bool)
            
            # Compute current Q values
            current_q_values = self.policy_net(state_batch)
            current_q_values = current_q_values.gather(1, action_batch.unsqueeze(1))
            
            # Compute target Q values
            with torch.no_grad():
                next_q_values = self.target_net(next_state_batch).max(1)[0]
                target_q_values = reward_batch + (
                    self.config.discount_factor * next_q_values * ~done_batch
                )
            
            # Compute loss
            loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
            
        except Exception as e:
            self.logger.error(f"Training step failed: {e}")
    
    async def train(self, training_examples: List[TrainingExample], 
                   num_episodes: int = None) -> Dict[str, Any]:
        """Train the RL policy."""
        
        if num_episodes is None:
            num_episodes = self.config.num_episodes
        
        if not training_examples:
            raise ValueError("No training examples provided")
        
        # Initialize wandb if enabled
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.wandb_project,
                config=asdict(self.config),
                name=f"rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        try:
            self.logger.info(f"Starting RL training for {num_episodes} episodes")
            
            for episode in range(num_episodes):
                # Select random example
                example = random.choice(training_examples)
                
                # Run episode
                result = await self.run_episode(example)
                self.episode_results.append(result)
                
                # Train policy
                if episode % 10 == 0:  # Train every 10 episodes
                    self._train_policy()
                
                # Update target network
                if episode % self.config.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                # Decay epsilon
                self.epsilon = max(
                    self.config.epsilon_end,
                    self.epsilon * self.config.epsilon_decay
                )
                
                # Logging
                if episode % 100 == 0:
                    avg_reward = np.mean([r.total_reward for r in self.episode_results[-100:]])
                    success_rate = np.mean([r.success for r in self.episode_results[-100:]])
                    
                    self.logger.info(
                        f"Episode {episode}: avg_reward={avg_reward:.3f}, "
                        f"success_rate={success_rate:.3f}, epsilon={self.epsilon:.3f}"
                    )
                    
                    if self.config.use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            'episode': episode,
                            'avg_reward': avg_reward,
                            'success_rate': success_rate,
                            'epsilon': self.epsilon
                        })
                
                # Save checkpoint
                if episode % self.config.save_freq == 0:
                    self._save_checkpoint(episode)
            
            # Final metrics
            final_metrics = self._calculate_final_metrics()
            
            self.logger.info("RL training completed")
            return final_metrics
            
        finally:
            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.finish()
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        if self.policy_net is None:
            return
        
        checkpoint_dir = Path(self.config.output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': asdict(self.config)
        }
        
        checkpoint_path = checkpoint_dir / f"checkpoint_episode_{episode}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def _calculate_final_metrics(self) -> Dict[str, Any]:
        """Calculate final training metrics."""
        if not self.episode_results:
            return {}
        
        rewards = [r.total_reward for r in self.episode_results]
        successes = [r.success for r in self.episode_results]
        steps = [r.steps_taken for r in self.episode_results]
        
        return {
            'total_episodes': len(self.episode_results),
            'avg_reward': np.mean(rewards),
            'final_avg_reward': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
            'success_rate': np.mean(successes),
            'final_success_rate': np.mean(successes[-100:]) if len(successes) >= 100 else np.mean(successes),
            'avg_steps': np.mean(steps),
            'final_epsilon': self.epsilon,
            'memory_size': len(self.replay_memory)
        }

# Convenience functions
async def quick_rl_train(examples: List[TrainingExample], 
                        base_model_path: str,
                        num_episodes: int = 500) -> RLTrainer:
    """Quickly train an RL policy with default settings."""
    config = RLConfig(
        num_episodes=num_episodes,
        batch_size=2,
        memory_size=1000
    )
    
    trainer = RLTrainer(config, base_model_path)
    await trainer.train(examples, num_episodes)
    
    return trainer