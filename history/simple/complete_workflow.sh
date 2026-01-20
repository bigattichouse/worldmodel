#!/bin/bash
# Complete WorldModel Workflow
# ============================

echo "🚀 WorldModel Complete Workflow"
echo "==============================="

# Step 1: Setup ROCm Environment
echo "📡 Setting up ROCm environment..."
source /home/bigattichouse/workspace/rocm/setup-rocm-env.sh
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false

echo "🌡️  GPU Status:"
rocm-smi --showtemp --showpower --showuse 2>/dev/null | head -3

# Step 2: Check if we have training data
if [ ! -f "data/worldmodel_training_1000.txt" ]; then
    echo ""
    echo "📁 Generating 1000 training examples..."
    python3 generate_1000_examples.py
fi

echo ""
echo "📊 Dataset info:"
wc -l data/worldmodel_training_1000.txt
ls -lh data/worldmodel_training_1000.txt

# Step 3: Ask user what to do
echo ""
echo "🤔 What would you like to do?"
echo "1. Quick training - Development (6 minutes, basic learning)"
echo "2. Production training - Full (3 hours, reliable WorldModel)"  
echo "3. Run inference (if model already trained)"
echo "4. Both dev training + inference test"
echo "5. Just show dataset sample"

read -p "Choose (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🔥 Starting development training (6 minutes)..."
        python3 train_worldmodel_rocm.py
        ;;
    2)
        echo ""
        echo "🏭 Starting PRODUCTION training (~3 hours)..."
        echo "⚠️  This will train for 15 epochs with proper convergence"
        echo "   Expected time: ~3-4 hours"
        echo "   Final model will be much more reliable"
        read -p "Continue? (y/N): " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            python3 train_worldmodel_production.py
        else
            echo "Production training cancelled."
        fi
        ;;
    3)
        echo ""
        echo "🧠 Starting inference..."
        # Check for production model first, then dev model
        if [ -d "worldmodel_production_training/final_model" ]; then
            echo "Using production model..."
            python3 run_worldmodel_inference.py --model worldmodel_production_training/final_model --interactive
        elif [ -d "worldmodel_rocm_output/final_model" ]; then
            echo "Using development model..."
            python3 run_worldmodel_inference.py --model worldmodel_rocm_output/final_model --interactive
        else
            echo "❌ No trained model found. Please run training first."
        fi
        ;;
    4)
        echo ""
        echo "🔥 Starting development training + inference test..."
        python3 train_worldmodel_rocm.py
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "🧠 Training complete! Testing inference..."
            python3 run_worldmodel_inference.py --model worldmodel_rocm_output/final_model "Calculate 30% of 150"
            echo ""
            echo "💬 Starting interactive session..."
            python3 run_worldmodel_inference.py --model worldmodel_rocm_output/final_model --interactive
        fi
        ;;
    5)
        echo ""
        echo "📖 Dataset sample (first 3 examples):"
        echo "====================================="
        head -50 data/worldmodel_training_1000.txt
        echo ""
        echo "📊 Total examples in dataset:"
        grep -c "^User:" data/worldmodel_training_1000.txt
        echo ""
        echo "📈 Training comparison:"
        echo "Development: 3 epochs, ~6 minutes, basic learning"
        echo "Production:  15 epochs, ~3 hours, reliable WorldModel"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "✅ Workflow complete!"