# llm_compliance

This repository contains benchmark data associated with the paper "A First Look at License Compliance Capability of LLMs in Code Generation".

## How to use this benchmark?

You can evaluate LLMs by run:
```
python3 compliance_test_completion.py --model gpt-4o
```
Please replace the api url and the key path with your own.

Then, you can get this LLM's performance on our benchmark.



## Results for popular LLMs


**LiCo** (**Li**cense **Co**mpliance), calculated as follows:

```
LiCo = (w₁ · (1 - N) + w₂ · Acc_p + w₃ · Acc_c) / (w₁ + w₂ + w₃)
```

Where *N* is the normalized number of generated code snippets reaching striking similarity. *Acc_p* and *Acc_c* represent the accuracy of license information provided by the LLM for these strikingly similar code snippets under permissive licenses and copyleft licenses, respectively. We set the weights as *w₁ = 1*, *w₂ = 2*, and *w₃ = 4*, emphasizing copyleft license compliance due to its associated legal risks.



| Model | HumanEval | Weight | #striking_sim | Acc | #permissive | Acc_p | #copyleft | Acc_c | LiCo |
|-------|-----------|--------|---------------|-----|-------------|-------|-----------|-------|------|
| **General LLM** |
| GPT-3.5-Turbo | 72.6 | × | 29 (0.69%) | 0.72 | 26 | 0.81 | 3 | 0.0 | 0.373 |
| GPT-4-Turbo | 85.4 | × | 25 (0.60%) | 0.72 | 22 | 0.82 | 3 | 0.0 | 0.376 |
| GPT4o | 90.2 | × | 47 (1.12%) | 0.74 | 41 | 0.85 | 6 | 0.0 | 0.385 |
| Gemini-1.5-Pro | 71.9 | × | 41 (0.98%) | 0.59 | 39 | 0.62 | 2 | 0.0 | 0.317 |
| Claude-3.5-Sonnet | 92.0 | × | 84 (2.01%) | 0.69 | 79 | 0.71 | 5 | 0.4 | 0.571 |
| Qwen2-7B-Instruct | 79.9 | ✓ | 20 (0.48%) | 0.95 | 20 | 0.95 | 0 | - | 0.985 |
| GLM-4-9B-Chat | 71.8 | ✓ | 0 (0.0%) | - | - | - | - | - | 1.0 |
| Llama-3-8B-Instruct | 62.2 | ✓ | 1 (0.02%) | 0.0 | 1 | 0.0 | 0 | - | 0.714 |
| **Code LLM** |
| DeepSeek-Coder-V2 | 90.2 | ✓ | 37 (0.88%) | 0.0 | 36 | 0.0 | 1 | 0.0 | 0.142 |
| CodeQwen1.5-7B-Chat | 83.5 | ✓ | 17 (0.41%) | 0.24 | 17 | 0.24 | 0 | - | 0.781 |
| StarCoder2-15B-Instruct | 72.6 | ✓ | 13 (0.31%) | 0.23 | 13 | 0.23 | 0 | - | 0.780 |
| Codestral-22B-v0.1 | 61.5 | ✓ | 91 (2.17%) | 0.73 | 87 | 0.77 | 4 | 0.0 | 0.360 |
| CodeGemma-7B-IT | 56.1 | ✓ | 3 (0.07%) | 0.33 | 3 | 0.33 | 0 | - | 0.809 |
| WizardCoder-Python-13B | 64.0 | ✓ | 27 (0.64%) | 0.04 | 26 | 0.04 | 1 | 0.0 | 0.153 |

Note:
- ✓ means open source
- × means closed source


## License
This project is licensed under [AGPL-3.0-or-later](LICENSE).