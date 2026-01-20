Experiment WorldModelExperiment {
  description: "Exploratory research into augmenting small language models with computational modeling capabilities",
  
  hypothesis: "A small model can learn to recognize knowledge gaps and generate executable code to fill those gaps, creating a basic world modeling capability",
  
  baseModel: {
    name: "gemma-3-270m-it",
    source: "hugging face", 
    capabilities: ["instruction following", "tool calling"],
    size: "270M parameters"
  },
  
  approach: "cautiously optimistic scientific exploration",
  
  coreComponents: {
    thinkProcess: {
      format: "<think>reasoning content</think>",
      purpose: "explicit reasoning steps before responses"
    },
    
    modelingCapability: {
      format: "<model>language: code</model>",
      purpose: "generate executable code to model unknown concepts",
      integration: "embedded within or near think processes"
    },
    
    requirementsDeclaration: {
      format: "<requires>requirement1,requirement2</requires>",
      syntax: "language:category(optional_domain)",
      examples: ["Python:math", "Python:web(wikipedia.com),file(read_only)"]
    },
    
    uncertaintyDetection: {
      method: "perplexity-based",
      rationale: "requires no architectural changes, fastest to prototype", 
      trigger: "high perplexity suggests knowledge gaps",
      fallback: "inject prompt suggesting lookup/modeling"
    }
  },
  
  executionModel: {
    ragRetrieval: {
      trigger: "user query received, before <think>",
      pipeline: [
        "embed user query + conversation context",
        "semantic search against model registry",
        "hybrid retrieval (semantic + keyword matching)",
        "re-rank by relevance + execution success",
        "filter by compatible requirements",
        "inject top-k into context"
      ]
    },
    
    contextInjection: {
      format: "<retrieved_models>model_summaries</retrieved_models><think>...</think>"
    },
    
    toolExecution: {
      trigger: "</model> tag closes",
      environment: "QEMU VM (../scratchpad tool)",
      languageDispatch: {
        python: "execute in VM python interpreter",
        js: "node execution within VM", 
        bash: "VM shell commands",
        c: "compile and execute within VM"
      }
    },
    
    approvalSystem: {
      autoApprove: "categories in allow list",
      requireApproval: "new or restricted categories",
      granularControl: "per-requirement approval/denial"
    }
  },
  
  modelMemory: {
    embeddingAccumulation: {
      primaryEmbedding: "original reasoning + code + context",
      pathExpansion: [
        "multiple reasoning paths to same model",
        "alternative problem contexts",
        "successful adaptations/modifications",
        "user feedback patterns"
      ]
    },
    
    retrievalBehavior: {
      multiPathMatching: "query matches any associated embedding path",
      crossContextDiscovery: "find models via unexpected reasoning paths"
    },
    
    modelRegistry: {
      structure: {
        modelId: "unique identifier",
        reasoning: "original <think> content",
        code: "the <model> code block", 
        requirements: "the <requires> declaration",
        context: "user query and conversation context",
        embeddings: "vector representations for similarity search",
        executionResults: "success/failure + validation feedback"
      }
    }
  },
  
  feedbackLoop: {
    requirementValidation: {
      postExecution: "analyze actual behavior vs declared requirements",
      learningSignals: ["requirement_accuracy", "actual_requirements", "safety_violations"]
    },
    
    iterativeImprovement: {
      signatureRefinement: "learn more precise categorization",
      trustBuilding: "consistent accuracy enables broader auto-approval"
    }
  },
  
  infrastructure: {
    modelStorage: "../model/ (hugging face format)",
    ggufStorage: "../gguf/ (llama.cpp compatibility)", 
    computeBackend: "ROCm (details in ../rocm/)",
    executionEnvironment: "QEMU VM (../scratchpad)"
  },
  
  futureEnhancements: {
    clusteringMechanisms: {
      note: "detect natural model families through path analysis",
      priority: "deferred - focus on basic multi-path embedding first"
    }
  }
}