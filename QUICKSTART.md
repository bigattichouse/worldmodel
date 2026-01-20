# WorldModel BluePrint - Quick Start Guide

Get started with training a **BluePrint reasoning model** in under 30 minutes.

## 🎯 What You'll Build

A small language model (Qwen3-0.6B) that generates structured reasoning:
- `<thinking>` - Strategic problem understanding
- `<blueprint>` - Formal algorithmic design using BluePrint notation

## ⚡ Prerequisites (5 minutes)

### Hardware Requirements
- **AMD GPU**: 8GB+ VRAM (ROCm support)
- **Alternative**: NVIDIA GPU or CPU (slower)
- **RAM**: 16GB+ recommended

### Software Setup
```bash
# Clone the repository
git clone [repository_url]
cd worldmodel

# Install dependencies
pip install -r requirements.txt

# For AMD GPUs - Set up ROCm
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0
```

### Check GPU Status
```bash
# AMD GPU
rocm-smi

# NVIDIA GPU  
nvidia-smi

# CPU fallback - skip GPU setup
```

## 🚀 Quick Training (15 minutes)

### Step 1: Generate Training Data
```bash
# Create 200 high-quality thinking→blueprint examples
python tools/generate_blueprint_dataset.py \
  --examples 200 \
  --output training/blueprint_examples.json
```

### Step 2: Train the Model
```bash
# Quick training (3 epochs, ~10 minutes)
python train_blueprint_model.py \
  --data training/blueprint_examples.json \
  --epochs 3 \
  --batch-size 2 \
  --output models/blueprint_quick

# Production training (15 epochs, ~45 minutes)
python train_blueprint_model.py \
  --data training/blueprint_examples.json \
  --epochs 15 \
  --batch-size 2 \
  --output models/blueprint_production
```

### Step 3: Test the Model
```bash
# Interactive testing
python test_blueprint_model.py \
  --model models/blueprint_quick \
  --interactive

# Single query test
python test_blueprint_model.py \
  --model models/blueprint_quick \
  --query "Design a user authentication system"
```

## 📝 Expected Output

After training, your model should generate responses like:

```
User: Design a user authentication system

<thinking>
PROBLEM UNDERSTANDING:
- Need secure user authentication
- Should handle login/logout flows
- Must store user credentials safely
- Consider session management

STRATEGIC APPROACH:
- Create authentication service
- Use secure password hashing
- Implement session tokens
- Support user registration and login

DESIGN PREPARATION:
- User entity with credentials
- Authentication service with methods
- Session management system
- Security considerations
</thinking>

<blueprint>
Service AuthenticationService {
  description: "Handles user authentication and session management",
  
  methods: {
    register(email: string, password: string) -> Result<UserId>,
    login(email: string, password: string) -> Result<SessionToken>,
    logout(sessionToken: SessionToken) -> Result<void>,
    validateSession(sessionToken: SessionToken) -> Result<UserId>
  },
  
  security: {
    passwordHashing: "bcrypt with salt",
    sessionExpiry: "24 hours",
    maxLoginAttempts: 3
  }
}

Database UserDatabase {
  Table Users {
    columns: {
      id: {type: UUID, primaryKey: true},
      email: {type: String, unique: true},
      passwordHash: {type: String, sensitive: true},
      createdAt: {type: Timestamp}
    }
  },
  
  Table Sessions {
    columns: {
      token: {type: String, primaryKey: true},
      userId: {type: UUID, foreignKey: "Users.id"},
      expiresAt: {type: Timestamp}
    }
  }
}
</blueprint>
```

## 📊 Success Metrics

Your model is working well if you see:

**✅ Good Signs:**
- Both `<thinking>` and `<blueprint>` tokens generated (>95% of responses)
- BluePrint syntax is valid (>90% accuracy)
- Thinking shows strategic understanding
- Blueprint contains complete specifications

**❌ Warning Signs:**
- Missing tokens or malformed syntax
- Generic/shallow thinking content
- Incomplete blueprint specifications
- Model refusing to generate structured output

## 🔧 Troubleshooting

### Training Issues

**Zero Loss / Not Learning:**
```bash
# Check data format
python tools/validate_blueprint_data.py training/blueprint_examples.json

# Try different learning rate
python train_blueprint_model.py --learning-rate 1e-5
```

**GPU Memory Errors:**
```bash
# Reduce batch size
python train_blueprint_model.py --batch-size 1

# Use gradient accumulation
python train_blueprint_model.py --batch-size 1 --grad-accumulation 4
```

**ROCm Issues:**
```bash
# Test ROCm setup
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

# Try different GFX version
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

### Generation Issues

**Model Not Generating Tags:**
- Increase training epochs (try 10-15)
- Check training data quality
- Verify model is loading correctly

**Invalid BluePrint Syntax:**
- Review blueprint-prompt.md for proper syntax
- Check training examples for consistency
- Consider adding syntax validation during training

## 📚 Next Steps

### Evaluate Your Model
```bash
# Run comprehensive evaluation
python evaluate_blueprint_model.py \
  --model models/blueprint_production \
  --test-data evaluation/test_cases.json
```

### Scale Up Training
```bash
# Generate larger dataset
python tools/generate_blueprint_dataset.py --examples 1000

# Longer training with better monitoring
python train_blueprint_model.py \
  --epochs 20 \
  --eval-steps 100 \
  --save-steps 200
```

### Plan for Phase 2
Once your BluePrint model is working reliably (>90% success rate):
1. Review [Phase 2 roadmap](docs/worldmodel-bytelogic.md)
2. Begin computational execution integration
3. Create BluePrint→ByteLogic translation examples

## 📖 Documentation

- **Complete Guide**: [Implementation Plan](docs/worldmodel-blueprint-plan.md)
- **BluePrint Syntax**: [Methodology Guide](docs/blueprint-prompt.md)
- **GPU Setup**: [ROCm Configuration](docs/rocm/ROCm_Training_Success_Guide.md)
- **Troubleshooting**: [ROCm Issues](docs/rocm/rocm_troubleshooting.md)

## 🎉 Success!

If you can reliably generate thinking→blueprint chains, you've built a foundational world modeling capability! The model now understands problems strategically and designs solutions formally - creating the perfect foundation for future computational execution.

**Time to Phase 2**: Once BluePrint generation is solid, you're ready to add executable code generation and build the full WorldModel vision.