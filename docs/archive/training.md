TrainingMethodology {
  hybridApproach: {
    phase1_SFT: {
      purpose: "teach basic <think>/<model>/<requires> structure",
      dataRequirement: "2000-2500 synthetic examples",
      rationale: "need coverage across reasoning patterns, languages, requirement types"
    },
    
    phase2_RL: {
      purpose: "optimize execution success and requirement accuracy", 
      environment: "live execution feedback loop",
      rewards: ["successful execution", "accurate requirements", "helpful solutions"]
    }
  },
  
  syntheticDataEstimate: {
    recommendedStart: "2000-2500 examples",
    categories: [
      "math/computation: 300 examples",
      "data processing: 200 examples", 
      "web/api calls: 200 examples",
      "file operations: 150 examples",
      "system queries: 100 examples",
      "multi-step reasoning: 200 examples",
      "existing_tool_calls: 350 examples (wikipedia, search, file_read, etc)",
      "chemistry/physics_hybrid: 400 examples (PCM calculations, orbital mechanics, thermodynamics)",
      "explicit_reasoning: 600 examples (why tool vs model decisions)"
    ]
  },
  
  complexDomainExamples: {
    phaseChangeMaterials: {
      reasoning: "<think>Need room volume (lookup), PCM properties (lookup), then heat calculations (model)</think>",
      tools: "wikipedia for PCM data, web search for building specs",
      models: "python heat transfer calculations, BTU/joule conversions"
    },
    
    orbitalMechanics: {
      reasoning: "<think>Need Earth/rocket specs (lookup), then trajectory math (model)</think>", 
      tools: "lookup planetary data, rocket specifications",
      models: "python orbital velocity calculations, fuel requirements"
    }
  },
  
  decisionLearning: {
    toolVsModel: "teach when to use existing tools vs create new models",
    examples: [
      "factual lookup → use wikipedia tool",
      "complex calculation → create <model>python: math</model>", 
      "file parsing → use existing file tools",
      "custom data transformation → create <model>python: data_processing</model>",
      "current events → use web search tool",
      "mathematical modeling → create computational model"
    ]
  },
  
  hybridBehaviors: {
    toolThenModel: "use tool for data, then model for processing",
    modelThenTool: "create model for calculation, then tool for validation",
    decisionTree: "learn appropriate tool/model selection patterns"
  }
}