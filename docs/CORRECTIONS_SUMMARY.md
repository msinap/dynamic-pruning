# Documentation Corrections Summary

This document summarizes all corrections made to align the documentation with the master's thesis paper.

## Major Corrections

### 1. Model Architecture
- **ERROR**: Documentation initially stated xLAM-2 1B has 24 layers
- **CORRECTED**: xLAM-2 1B has **28 layers** (confirmed in paper Section 4.1)
- **Files Updated**: README.md, ARCHITECTURE.md, EXAMPLES.md, all example code snippets

### 2. Terminology: LLM vs SLM
- **ERROR**: Referred to model as "Large Language Model (LLM)"
- **CORRECTED**: Model is a **"Small Language Model (SLM)"** (confirmed in paper title and abstract)
- **Files Updated**: README.md, ARCHITECTURE.md
- **Rationale**: Paper specifically focuses on SLMs (1-15B parameters) as efficient alternatives to LLMs

### 3. Adapter Type
- **ERROR**: Generic references to "adapters"
- **CORRECTED**: Specifically **"LoRA adapters"** throughout
- **Files Updated**: README.md, ARCHITECTURE.md, EXAMPLES.md
- **Rationale**: Paper Section 3.2 specifies use of LoRA (Low-Rank Adaptation) adapters

### 4. Training Method
- **ERROR**: Vague "knowledge distillation"
- **CORRECTED**: **"White-box knowledge distillation"** with MSE loss
- **Files Updated**: README.md, ARCHITECTURE.md
- **Rationale**: Paper Section 3.2 explicitly states white-box distillation approach

### 5. Performance Metrics
- **ERROR**: Incorrect accuracy and speedup numbers
- **CORRECTED**: Aligned with paper Table 1 results:
  - Original model: 71% accuracy
  - Static 2-layer: 63% accuracy, 7.1% speedup
  - Static 3-layer: 51% accuracy, 10.7% speedup
  - Static 5-layer: 38% accuracy, 17.6% speedup
  - FBC: 52% accuracy, 16.1% speedup
  - DPO: 59% accuracy, 17.1% speedup
  - ORPO: **61% accuracy, 17.9% speedup**
- **Files Updated**: README.md

### 6. Average Layers Pruned
- **ERROR**: Documentation stated "6-7 layers pruned on average"
- **CORRECTED**: **~5.5 layers pruned on average** for router methods
- **Files Updated**: README.md, ARCHITECTURE.md
- **Rationale**: Paper Section 5.2 mentions "pruning 5.5 layers on average" for FBC

### 7. Layer Analysis
- **ERROR**: Generic statements about layer importance
- **CORRECTED**: Specific findings from paper:
  - Layer 1: MSE = 46.75 (critical for embedding transformation)
  - Middle layers (2-20): Low MSE (functional redundancy)
  - Final layers (26-27): Elevated MSE (task-specific output)
- **Files Updated**: README.md
- **Rationale**: Paper Section 5.1 and Figure 4

### 8. Speedup Calculation
- **ERROR**: Didn't mention router overhead
- **CORRECTED**: Explicitly state that **speedup includes router overhead (equivalent to 1 layer)**
- **Files Updated**: README.md, ARCHITECTURE.md
- **Rationale**: Paper Section 5.2 mentions this important detail

### 9. Loss Functions
- **ERROR**: Simplified or incorrect loss function formulations
- **CORRECTED**:
  - FBC: `L_FBC = (μ_pred - μ_target)² + BCE(scores_pred, scores_target)`
  - DPO: `L_DPO = -E[(x,y_w,y_l)][log σ(β log(π_θ(y_w|x)/π_θ(y_l|x)))]`
  - ORPO: `L_ORPO = -E[log σ(β log(π_θ(y_w|x)/π_θ(y_l|x))) + log π_θ(y_w|x)]`
- **Files Updated**: README.md
- **Rationale**: Paper Section 3.4

### 10. Preference Criterion
- **ERROR**: Vague preference description
- **CORRECTED**: Explicit criterion:
  1. Configuration A preferred over B if higher accuracy score, OR
  2. Same accuracy but A prunes more layers (more efficient)
- **Files Updated**: README.md, ARCHITECTURE.md
- **Rationale**: Paper Section 3.4.1

### 11. DFS Strategy
- **ERROR**: Basic description of DFS
- **CORRECTED**: Detailed explanation:
  - Depth-first search starting with 0 layers pruned
  - Iteratively adds layers maintaining accuracy above threshold
  - Backtracks when performance drops
  - Produces higher variance data with aggressive configurations
  - Creates clear contrasts for preference learning
- **Files Updated**: README.md, ARCHITECTURE.md
- **Rationale**: Paper Section 3.4.1

### 12. Router Architecture Details
- **ERROR**: Missing precision on router outputs
- **CORRECTED**:
  - Layer scores: prunability scores (higher = safer to prune)
  - Ratio distribution: μ via sigmoid ∈ [0, 1]
  - Log σ clipped to [-7, -3] for numerical stability
  - Inference: sample r ~ N(μ, exp(log_σ)), compute k = ⌊r × L⌋
- **Files Updated**: README.md, ARCHITECTURE.md
- **Rationale**: Paper Section 3.3 and equations

### 13. Evaluation Metric
- **ERROR**: Generic "accuracy" mention
- **CORRECTED**: **JSON Match** score used throughout (parses and compares JSON structures)
- **Files Updated**: README.md, ARCHITECTURE.md
- **Rationale**: Paper Section 4.2 specifies JSON Match as primary metric

### 14. Dataset Details
- **ERROR**: Incomplete dataset description
- **CORRECTED**: 
  - **Dataset**: Salesforce xlam-function-calling-60k (60K examples)
  - **Task**: Tool calling / function calling
  - **Format**: (query, available_tools, function_calls) tuples
- **Files Updated**: README.md
- **Rationale**: Paper Section 4.1

### 15. Training Efficiency
- **ERROR**: Missing training time information
- **CORRECTED**: Added specific timings:
  - LoRA adapter training: <1 hour
  - Offline dataset generation: 10-15 hours
  - Router training: 1-2 hours each
  - Total: <20 hours on single GPU
- **Files Updated**: README.md
- **Rationale**: Paper Section 5.3

## Minor Corrections

### 16. Differentiable Operations
- **CLARIFIED**: Both Gumbel-Softmax and Sinkhorn sorting mentioned (not just one)
- **Files Updated**: README.md
- **Rationale**: Paper Section 3.5

### 17. Offline RL Emphasis
- **CLARIFIED**: Emphasized "offline" reinforcement learning (no online environment interaction)
- **Files Updated**: README.md, ARCHITECTURE.md
- **Rationale**: Paper abstract and Section 3.4

### 18. Base Model
- **ADDED**: xLAM-2 1B is fine-tuned from Qwen2.5-1B
- **Files Updated**: ARCHITECTURE.md
- **Rationale**: Paper Section 4.1

### 19. Hardware
- **ADDED**: All experiments on single NVIDIA RTX 4090 GPU
- **Context**: Resource-constrained research environment
- **Rationale**: Paper Section 4.1

### 20. Acknowledgments
- **UPDATED**: More specific acknowledgments:
  - xLAM-2 1B (not just "xLAM Model")
  - LoRA (Hu et al.)
  - Sentence-BERT (all-MiniLM-L6-v2 specifically)
- **Files Updated**: README.md

## Consistency Improvements

### All Code Examples
- Updated all router initialization to use `num_llm_layers=28`
- Updated layer iteration ranges to account for 28 layers
- Added comments clarifying xLAM-2 1B architecture

### Terminology Consistency
- "Tool calling" vs "Function calling" - used interchangeably as in paper
- "SLM" consistently used for the base model
- "LoRA adapters" specifically, not generic "adapters"
- "Offline RL" emphasized throughout

### Mathematical Notation
- Aligned loss function notation with paper
- Used consistent symbols (μ, σ, π_θ, etc.)
- Added paper equation references where appropriate

## Verification Checklist

✅ Number of layers: 28 (not 24)
✅ Model type: SLM (not LLM)
✅ Adapter type: LoRA adapters
✅ Training: White-box knowledge distillation
✅ Results table: Matches paper Table 1
✅ Average pruning: ~5.5 layers
✅ Layer MSE values: Match paper Figure 4
✅ Speedup: Includes router overhead
✅ Loss functions: Match paper equations
✅ Preference criterion: Explicit and accurate
✅ DFS strategy: Detailed explanation
✅ Router outputs: Precise specifications
✅ Evaluation: JSON Match metric
✅ Dataset: xlam-function-calling-60k
✅ Training time: Documented
✅ Hardware: Single RTX 4090

## Files Modified

1. **README.md** - Major updates throughout
2. **docs/ARCHITECTURE.md** - Technical corrections
3. **docs/EXAMPLES.md** - Code example fixes
4. **All source file docstrings** - Already accurate

## Impact

These corrections ensure:
1. **Factual accuracy**: All technical details match the paper
2. **Reproducibility**: Correct hyperparameters and architecture
3. **Clarity**: Precise terminology and specifications
4. **Consistency**: Unified terminology across all documentation
5. **Credibility**: Documentation aligns with published research

## Notes

- No changes were needed to source code docstrings (they were already accurate)
- Script documentation headers were already correct
- CONTRIBUTING.md was deleted (user preference)
- All corrections maintain the professional tone and comprehensive nature of the original documentation

