# Bootstrap Script for hf-llm-architect Skill

## Purpose
This script creates the complete `hf-llm-architect` skill structure with all files and content.

## Usage
```bash
python3 create_hf_llm_architect.py
```

## What It Creates
- **Main Skill File**: `skills/hf-llm-architect/SKILL.md`
- **Utility Scripts** (3 PEP 723 compliant Python tools):
  - `env_health_check.py` - GPU/CUDA diagnostics
  - `stream_data_preview.py` - Dataset streaming inspector
  - `calc_vram_requirements.py` - VRAM requirement estimator
- **Reference Documentation** (9 markdown files across 5 categories):
  - Pre-training: Scaling laws, continual pre-training
  - Fine-tuning: SFT best practices, PEFT/LoRA config
  - Alignment: DPO/ORPO, GRPO reasoning
  - Distributed: Strategy matrix, DeepSpeed/FSDP config
  - Inference: Production optimization

## Requirements
- Python 3.6+
- Write access to the `skills/` directory

## Output
Creates 7 directories and 13 files totaling approximately 18KB of documentation and code.

## Verification
After running, verify with:
```bash
# Check structure exists
ls -la skills/hf-llm-architect/

# Count files (should show 13)
find skills/hf-llm-architect/ -type f | wc -l

# Test a script
python3 skills/hf-llm-architect/scripts/calc_vram_requirements.py --size 7 --mode inference
```

## Notes
- Scripts are automatically made executable (chmod 755)
- All content matches the specification exactly
- PEP 723 inline script metadata included for `uv` compatibility

## Troubleshooting
- **Permission denied**: Run with appropriate user permissions
- **Directory exists**: Script uses `mkdir -p` (safe to re-run)
- **File exists**: Will overwrite existing files with fresh content
