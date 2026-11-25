# AE/VAE architecture comparisons

Question: which graph architecture supports reconstruction and useful sampling?

- Existing reusable study: `latentmol.architecture_study` and `latentmol.architecture_followup`.
- Model components: [models](../../src/latentmol/models/).
- Retained variants: `configs/arch_*`.
- Saved comparisons: [study25](../../docs/25_architecture_results_20260929.md), [study29](../../docs/29_architecture_comparison_20261001.md).

Use another config/arm in the same study before writing another trainer. Preserve
matched schema/data/budget differences in a short result note. Existing source
releases and checkpoints stay where their saved records point.
