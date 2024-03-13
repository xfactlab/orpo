# **ORPO**

This is the official repository for <a class="link" href="https://arxiv.org/abs/2403.07691">**Reference-free Monolithic Preference Optimization with Odds Ratio**</a>.

&nbsp;

**`Model Checkpoints`**

Our models trained with ORPO can be found in:

- [X] **Mistral-ORPO-⍺**: 🤗 <a class="link" href="https://huggingface.co/kaist-ai/mistral-orpo-alpha">kaist-ai/mistral-orpo-alpha</a>
- [X] **Mistral-ORPO-β**: 🤗 <a class="link" href="https://huggingface.co/kaist-ai/mistral-orpo-beta">kaist-ai/mistral-orpo-beta</a>

&nbsp;

**`AlpacaEval`**

<figure>
  <img class="png" src="/assets/img/alpaca_blog.png" alt="Description of the image">
  <figcaption><b>Figure 1.</b> AlpacaEval 2.0 score for the models trained with different alignment methods.</figcaption>
</figure>

&nbsp;

**`MT-Bench`**

<figure>
  <img class="png" src="/assets/img/mtbench_hf.png" alt="Description of the image">
  <figcaption><b>Figure 2.</b> MT-Bench result by category.</figcaption>
</figure>

&nbsp;
**`IFEval`**

| **Model Type**     | **Prompt-Strict** | **Prompt-Loose** | **Inst-Strict** | **Inst-Loose** |
|--------------------|:-----------------:|:----------------:|:---------------:|----------------|
| **Llama-2-Chat (70B)** |       0.4436      |      0.5342      |      0.5468     |     0.6319     |
| **Zephyr-β (7B)** |       0.4233      |      0.4547      |      0.5492     |     0.5767     |
| **Mixtral-8X7B-Instruct-v0.1** |       0.5213      |      **0.5712**      |      0.6343     |     **0.6823**     |
| **Mistral-ORPO-⍺ (7B)** |       0.5009      |      0.5083      |      0.5995     |     0.6163     |
| **Mistral-ORPO-β (7B)** |       **0.5287**      |      0.5564      |      **0.6355**     |     0.6619     |
