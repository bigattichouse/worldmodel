ProjectStructure {
  srcDirectory: {
    
    core: {
      inferenceEngine: {
        responsibilities: ["model loading", "tokenization", "generation", "structured output parsing"],
        dependencies: ["transformers", "torch", "llama.cpp bindings"]
      },
      
      tagParser: {
        responsibilities: ["extract <think>/<model>/<requires> blocks", "validate syntax", "parse requirements"],
        interface: "clean API for tag extraction and validation"
      },
      
      uncertaintyDetection: {
        responsibilities: ["perplexity calculation", "confidence scoring", "trigger decision"],
        interface: "detect when model should enter <think>/<model> mode"
      }
    },
    
    execution: {
      vmInterface: {
        responsibilities: ["communicate with QEMU scratchpad", "language dispatch", "result capture"],
        abstraction: "unified interface for python/js/bash/c execution"
      },
      
      requirementValidator: {
        responsibilities: ["post-execution analysis", "requirement accuracy checking", "safety violation detection"],
        feedback: "generate learning signals for RL training"
      },
      
      approvalSystem: {
        responsibilities: ["permission checking", "user prompt generation", "approval persistence"],
        policies: "configurable approval rules and categories"
      }
    },
    
    memory: {
      ragSystem: {
        responsibilities: ["embedding generation", "similarity search", "retrieval ranking"],
        storage: "file-based vector database (FAISS/ChromaDB)"
      },
      
      modelRegistry: {
        responsibilities: ["model storage", "metadata management", "history tracking"],
        versioning: "git-like versioning for model evolution"
      },
      
      embeddingManager: {
        responsibilities: ["multi-path embedding accumulation", "semantic indexing"],
        interface: "add/search/cluster embeddings"
      }
    },
    
    training: {
      dataGenerator: {
        responsibilities: ["synthetic example generation", "category coverage", "quality validation"],
        output: "structured training datasets"
      },
      
      sftTrainer: {
        responsibilities: ["supervised fine-tuning pipeline", "loss calculation", "checkpoint management"],
        integration: "work with existing model infrastructure"
      },
      
      rlTrainer: {
        responsibilities: ["reinforcement learning loop", "reward calculation", "policy optimization"],
        feedback: "execution success + requirement accuracy rewards"
      }
    },
    
    utils: {
      config: {
        responsibilities: ["configuration management", "environment setup", "path resolution"],
        scope: "centralized config for all modules"
      },
      
      logging: {
        responsibilities: ["structured logging", "experiment tracking", "debug output"],
        integration: "unified logging across all components"
      },
      
      testing: {
        responsibilities: ["test utilities", "mock objects", "integration test helpers"],
        coverage: "support unit and integration testing"
      }
    }
  }
}