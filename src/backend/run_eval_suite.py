from lm_eval import evaluator
from lm_eval.tasks import TaskManager

from src.backend.manage_requests import EvalRequest

from src.backend.tasks.xsum.task import XSum
from src.backend.tasks.xsum.task_v2 import XSumv2

from src.backend.tasks.cnndm.task import CNNDM
from src.backend.tasks.cnndm.task_v2 import CNNDMv2

from src.backend.tasks.selfcheckgpt.task import SelfCheckGPT

from src.backend.huggingface_generate_until import HFLMwithChatTemplate
from src.backend.moe_infinity import MoEHFLM


def run_evaluation(
    eval_request: EvalRequest,
    task_names,
    num_fewshot,
    batch_size,
    device,
    use_cache=None,
    limit=None,
    max_nb_samples=100,
) -> dict:
    if limit:
        print("WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")

    # include_task_folder("src/backend/tasks/")
    # initialize_tasks('INFO')

    print(f"Allocating task manager for: {task_names}")

    task_manager = TaskManager(include_path="./src/backend/tasks/")
    # task_manager.initialize_tasks('INFO')

    print(f"Considered Tasks: {task_names}")
    # print(f"Allowed Tasks: {tasks.ALL_TASKS}")

    # task_names = utils.pattern_match(task_names, tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")
    print(f"Eval Request: {eval_request}")
    print(
        f"Num Fewshot: {num_fewshot}, Batch Size: {batch_size}, Device: {device}, Use Cache: {use_cache}, Limit: {limit}"
    )
    # hf-chat is implemented to use apply_chat_template
    results = evaluator.simple_evaluate(
        model=eval_request.inference_framework,  # "hf-causal-experimental",  # "hf-causal", hf-chat
        model_args=eval_request.get_model_args(),
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        max_batch_size=8,
        device=device,
        use_cache=use_cache,
        limit=limit,
        write_out=True,
        task_manager=task_manager,
        verbosity="WARNING",
    )

    results["config"]["model_dtype"] = eval_request.precision
    results["config"]["model_name"] = eval_request.model
    results["config"]["model_sha"] = eval_request.revision
    results["config"]["inference_framework"] = eval_request.inference_framework

    if max_nb_samples is not None:
        if "samples" in results:
            samples = results["samples"]
            for task_name in samples.keys():
                if len(samples[task_name]) > max_nb_samples:
                    results["samples"][task_name] = results["samples"][task_name][:max_nb_samples]

    # print(evaluator.make_table(results))

    return results
