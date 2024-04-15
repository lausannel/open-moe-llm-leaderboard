from src.display.utils import ModelType

TITLE = """<h1 align="center" id="space-title">OPEN-MOE-LLM-LEADERBOARD</h1>"""

INTRODUCTION_TEXT = """
The OPEN-MOE-LLM-LEADERBOARD is specifically designed to assess the performance and efficiency of various Mixture of Experts (MoE) Large Language Models (LLMs). This initiative, driven by the open-source community, aims to comprehensively evaluate these advanced MoE LLMs. We extend our gratitude to the Huggingface for the GPU community grant that supported the initial debugging process, and to [NetMind.AI](https://netmind.ai/home) for their generous GPU donation, which ensures the continuous operation of the Leaderboard.

The OPEN-MOE-LLM-LEADERBOARD includes generation and multiple choice tasks to measure the performance and efficiency of MOE LLMs.


Tasks:
- **Generation Self-consistancy** -- [SelfCheckGPT](https://github.com/potsawee/selfcheckgpt)
- **Multiple Choice Performance** -- [MMLU](https://arxiv.org/abs/2009.03300)

Columns and Metrics:
- Method: The MOE LLMs inference framework.
- E2E(s): Average End to End generation time in seconds.
- PRE(s): Prefilling Time of input prompt in seconds.
- T/s: Tokens throughout per second.
- Precision: The precison of used model.

"""
LLM_BENCHMARKS_TEXT = f"""

"""
LLM_BENCHMARKS_DETAILS = f"""

"""

FAQ_TEXT = """
---------------------------
# FAQ
## 1) Submitting a model
XXX
## 2) Model results
XXX
## 3) Editing a submission
XXX
"""

EVALUATION_QUEUE_TEXT = """

"""

CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
CITATION_BUTTON_TEXT = r"""

"""
