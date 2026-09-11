# Stages & acceptance

## Stage A — `OvisOCR2-Point` (required)

Deliverables:

- [ ] POINT dataset (synth primary, real 15–30%), multi-point + negatives  
- [ ] Held-out eval set **before** large train runs  
- [ ] QLoRA/LoRA SFT on `ATH-MaaS/OvisOCR2`, vision frozen first  
- [ ] Metrics beat zero-shot OvisOCR2+POINT prompt  
- [ ] Merged HF + GGUF (`Q4_K_M` / `Q5_K_M`) + mmproj if needed  

Go: outputs are mainly the pointed block on full-screen + crosshair cases.  
No-Go: systematic page dump → debug data/marker/negatives.

## Stage B — dual PAGE+POINT (optional)

- [ ] Stage A already Go  
- [ ] Mixed dataset with prompt switching  
- [ ] Re-eval POINT + PAGE; abort if either regresses hard  

## Compute policy

Single ~24GB GPU: SFT LoRA/QLoRA only. Skip paper’s 4B GRPO + OPD unless upgraded.
