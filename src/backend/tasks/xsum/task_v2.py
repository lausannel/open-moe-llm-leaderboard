from lm_eval.api.task import ConfigurableTask
from lm_eval.api.instance import Instance

# from lm_eval.api.registry import register_task
from lm_eval.api.metrics import mean

import torch
import sacrebleu
from rouge_score import rouge_scorer, scoring


def bleu(refs, preds):
    """
    Returns `t5` style BLEU scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L41

    :param refs:
        A `list` of `list` of reference `str`s.
    :param preds:
        A `list` of predicted `str`s.
    """
    score = sacrebleu.corpus_bleu(
        preds,
        refs,
        smooth_method="exp",
        smooth_value=0.0,
        force=False,
        lowercase=False,
        tokenize="intl",
        use_effective_order=False,
    ).score
    return score


def rouge(refs, preds):
    """
    Returns `t5` style ROUGE scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L68

    :param refs:
        A `list` of reference `strs`.
    :param preds:
        A `list` of predicted `strs`.
    """
    rouge_types = ["rouge1", "rouge2", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(rouge_types)
    # Add newlines between sentences to correctly compute `rougeLsum`.

    def _prepare_summary(summary):
        summary = summary.replace(" . ", ".\n")
        return summary

    # Accumulate confidence intervals.
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        ref = _prepare_summary(ref)
        pred = _prepare_summary(pred)
        aggregator.add_scores(scorer.score(ref, pred))
    result = aggregator.aggregate()
    return {type: result[type].mid.fmeasure * 100 for type in rouge_types}


# @register_task("xsum_v2")
class XSumv2(ConfigurableTask):
    VERSION = 2
    DATASET_PATH = "EdinburghNLP/xsum"
    DATASET_NAME = None

    def __init__(self):
        # breakpoint()
        super().__init__(
            config={
                "metadata": {"version": self.VERSION},
                "generation_kwargs": {"do_sample": False, "temperature": 0.0, "until": ["\n", "\n\n"]},
            }
        )
        self.factkb_tokenizer = None
        self.factkb_model = None
        self.bert_score = None

    def maybe_init_factkb(self):
        if self.factkb_tokenizer is None or self.factkb_model is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            self.factkb_tokenizer = AutoTokenizer.from_pretrained(
                "roberta-base", padding="max_length", truncation=True
            )
            self.factkb_model = AutoModelForSequenceClassification.from_pretrained(
                "bunsenfeng/FactKB", num_labels=2, device_map="auto"
            )

    def maybe_init_bertscore(self):
        if self.bert_score is None:
            from evaluate import load

            self.bert_score = load("bertscore")

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    # def fewshot_delimiter(self):
    #     return "\n\n"

    # From https://arxiv.org/abs/2305.14739
    def doc_to_text(self, doc):
        return f'Article: {doc["document"]}\nSummarize the article in one sentence. Summary:'

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["document"]

    def doc_to_target(self, doc):
        return doc["summary"]

    def construct_requests(self, doc, ctx, **kwargs):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """

        return [
            Instance(
                request_type="generate_until",
                doc=doc,
                # arguments=(ctx, {"until": ["\n", "."]}),
                arguments=(ctx, {"until": ["\n"]}),
                idx=0,
                **kwargs,
            )
        ]

    def process_results(self, doc, results):
        completion = results[0]

        # breakpoint()

        document = doc["document"]
        gold_summary = doc["summary"]

        true_refs = [doc["summary"]]
        all_refs = true_refs

        # ROUGE-N
        rouge_scores = [rouge([ref], [completion]) for ref in all_refs]
        # ROUGE-1
        rouge1_scores = [score["rouge1"] for score in rouge_scores]
        # ROUGE-2
        rouge2_scores = [score["rouge2"] for score in rouge_scores]
        # ROUGE-L
        rougeL_scores = [score["rougeLsum"] for score in rouge_scores]

        self.maybe_init_factkb()
        input_factkb = [[completion, document]]
        factkb_tokens = self.factkb_tokenizer(
            input_factkb, return_tensors="pt", padding="max_length", truncation=True
        ).to(self.factkb_model.device)
        factkb_logits = self.factkb_model(**factkb_tokens).logits
        factkb_res = torch.softmax(factkb_logits, dim=1)

        self.maybe_init_bertscore()
        bert_score_res = self.bert_score.compute(
            predictions=[completion], references=[gold_summary], model_type="microsoft/deberta-xlarge-mnli", lang="en"
        )

        res = {
            "rouge1": rouge1_scores[0],
            "rouge2": rouge2_scores[0],
            "rougeL": rougeL_scores[0],
            "factKB": float(factkb_res[0][1]),
            "bertscore_precision": float(bert_score_res["precision"][0]),
            "bertscore_recall": float(bert_score_res["recall"][0]),
            "bertscore_f1": float(bert_score_res["f1"][0]),
        }

        # breakpoint()

        return res

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            k: mean
            for k in [
                "rouge1",
                "rouge2",
                "rougeL",
                "factKB",
                "bertscore_precision",
                "bertscore_recall",
                "bertscore_f1",
            ]
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            k: True
            for k in [
                "rouge1",
                "rouge2",
                "rougeL",
                "factKB",
                "bertscore_precision",
                "bertscore_recall",
                "bertscore_f1",
            ]
        }
