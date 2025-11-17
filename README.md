# PEdit: Pareto-guided Editing for Text-Instruction Guided Image Editing

## 📄 Abstract
Text-instruction guided image editing should balance the competing goals of preserving the source image and achieving faithful edits. However, conventional methods that rely on fixed input conditions or optimize only the initial noise fail to dynamically control the latent states throughout the editing process, resulting in final images that are often inconsistent and misaligned with the intended instruction.

We argue that this limitation arises from an imbalance between the source image condition and the editing prompt condition across the entire editing trajectory. To balance the two heterogeneous conditions, we propose **Pareto-guided Editing (PEdit)**, a paradigm that dynamically steers the latent representation to remain near the central region of the Pareto front between source preservation and editing fidelity during denoising.

While single-objective optimization tends to overemphasize one aspect, leading to biased results toward either preservation or fidelity, the proposed Pareto-guided method achieves a **balanced trade-off** between these inherently conflicting objectives, ensuring both semantic consistency and structural stability.

The intermediate latent representations are evaluated from two complementary perspectives:

- **Editing fidelity** — measured by the cross-attention score ratio between the text and source-image conditions  
- **Structural preservation** — assessed via pixel-wise similarity to the source image signal

Depending on the latent condition, the text and image condition embeddings are **adaptively scaled within each Transformer block**, effectively guiding the latent trajectory toward the Pareto front and maintaining a stable balance between the two objectives.

Experiments on **HQ-Edit Bench** (synthetic editing dataset) and **Emu-Edit Bench** (real data) demonstrate that our method consistently improves spatial preservation and semantic faithfulness, achieving robust performance without sacrificing image quality.
