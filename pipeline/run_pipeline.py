import torch
import random
import json
import os
import argparse

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction, get_sentiment_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    return parser.parse_args()

def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.
    
    For sentiment analysis:
    - positive: Prompts that should elicit positive sentiment (e.g., positive reviews)
    - negative: Prompts that should elicit negative sentiment (e.g., negative reviews)

    Returns:
        Tuple of datasets: (positive_train, negative_train, positive_val, negative_val)
    """
    random.seed(42)
    # Note: 'harmful' split is repurposed for negative sentiment, 'harmless' for positive sentiment
    # You may want to create dedicated sentiment datasets
    positive_train = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=True), cfg.n_train)
    negative_train = random.sample(load_dataset_split(harmtype='harmful', split='train', instructions_only=True), cfg.n_train)
    positive_val = random.sample(load_dataset_split(harmtype='harmless', split='val', instructions_only=True), cfg.n_val)
    negative_val = random.sample(load_dataset_split(harmtype='harmful', split='val', instructions_only=True), cfg.n_val)
    return positive_train, negative_train, positive_val, negative_val

def filter_data(cfg, model_base, positive_train, negative_train, positive_val, negative_val):
    """
    Filter datasets based on sentiment scores.
    
    Filters to keep:
    - positive prompts that actually score high on positive sentiment
    - negative prompts that actually score low on positive sentiment

    Returns:
        Filtered datasets: (positive_train, negative_train, positive_val, negative_val)
    """
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        # High score = positive sentiment, Low score = negative sentiment
        positive_train_scores = get_sentiment_scores(model_base.model, positive_train, model_base.tokenize_instructions_fn, model_base.positive_toks)
        negative_train_scores = get_sentiment_scores(model_base.model, negative_train, model_base.tokenize_instructions_fn, model_base.positive_toks)
        # Keep positive prompts with high sentiment score
        positive_train = filter_examples(positive_train, positive_train_scores, 0, lambda x, y: x > y)
        # Keep negative prompts with low sentiment score
        negative_train = filter_examples(negative_train, negative_train_scores, 0, lambda x, y: x < y)

    if cfg.filter_val:
        positive_val_scores = get_sentiment_scores(model_base.model, positive_val, model_base.tokenize_instructions_fn, model_base.positive_toks)
        negative_val_scores = get_sentiment_scores(model_base.model, negative_val, model_base.tokenize_instructions_fn, model_base.positive_toks)
        positive_val = filter_examples(positive_val, positive_val_scores, 0, lambda x, y: x > y)
        negative_val = filter_examples(negative_val, negative_val_scores, 0, lambda x, y: x < y)
    
    return positive_train, negative_train, positive_val, negative_val

def generate_and_save_candidate_directions(cfg, model_base, positive_train, negative_train):
    """
    Generate and save candidate sentiment directions.
    
    The direction is computed as: mean(positive) - mean(negative)
    This captures the "positivity direction" that, when ablated, reduces positive sentiment.
    """
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'generate_directions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'generate_directions'))

    mean_diffs = generate_directions(
        model_base,
        positive_train,
        negative_train,
        artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"))

    torch.save(mean_diffs, os.path.join(cfg.artifact_path(), 'generate_directions/mean_diffs.pt'))

    return mean_diffs

def select_and_save_direction(cfg, model_base, positive_val, negative_val, candidate_directions):
    """
    Select and save the best sentiment direction.
    
    Selects the direction that most effectively:
    - Reduces positive sentiment when ablated (on negative prompts)
    - Increases positive sentiment when added (on positive prompts)
    """
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'select_direction')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'select_direction'))

    pos, layer, direction = select_direction(
        model_base,
        negative_val,  # Test ablation on negative prompts
        positive_val,  # Test steering on positive prompts
        candidate_directions,
        artifact_dir=os.path.join(cfg.artifact_path(), "select_direction")
    )

    with open(f'{cfg.artifact_path()}/direction_metadata.json', "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)

    torch.save(direction, f'{cfg.artifact_path()}/direction.pt')

    return pos, layer, direction

def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'completions'))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens)
    
    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_completions.json', "w") as f:
        json.dump(completions, f, indent=4)

def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies):
    """Evaluate completions and save results for a dataset."""
    with open(os.path.join(cfg.artifact_path(), f'completions/{dataset_name}_{intervention_label}_completions.json'), 'r') as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_{intervention_label}_evaluations.json"),
    )

    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_evaluations.json', "w") as f:
        json.dump(evaluation, f, indent=4)

def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label):
    """Evaluate loss on datasets."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'loss_evals')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'loss_evals'))

    on_distribution_completions_file_path = os.path.join(cfg.artifact_path(), f'completions/harmless_baseline_completions.json')

    loss_evals = evaluate_loss(model_base, fwd_pre_hooks, fwd_hooks, batch_size=cfg.ce_loss_batch_size, n_batches=cfg.ce_loss_n_batches, completions_file_path=on_distribution_completions_file_path)

    with open(f'{cfg.artifact_path()}/loss_evals/{intervention_label}_loss_eval.json', "w") as f:
        json.dump(loss_evals, f, indent=4)

def run_pipeline(model_path):
    """
    Run the full sentiment direction pipeline.
    
    This pipeline finds a "positivity direction" that, when ablated, causes the model
    to generate more negative sentiment content (similar to how ablating a refusal
    direction causes harmful content generation).
    """
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)

    model_base = construct_model_base(cfg.model_path)

    # Load and sample datasets
    positive_train, negative_train, positive_val, negative_val = load_and_sample_datasets(cfg)
    
    # Filter datasets based on sentiment scores
    positive_train, negative_train, positive_val, negative_val = filter_data(cfg, model_base, positive_train, negative_train, positive_val, negative_val)

    # 1. Generate candidate positivity directions
    candidate_directions = generate_and_save_candidate_directions(cfg, model_base, positive_train, negative_train)
    
    # 2. Select the most effective positivity direction
    pos, layer, direction = select_and_save_direction(cfg, model_base, positive_val, negative_val, candidate_directions)

    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
    actadd_fwd_pre_hooks, actadd_fwd_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))], []

    # 3a. Generate and save completions on evaluation datasets (tests if ablation induces negative sentiment)
    for dataset_name in cfg.evaluation_datasets:
        generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', dataset_name)
        generate_and_save_completions_for_dataset(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation', dataset_name)
        generate_and_save_completions_for_dataset(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd', dataset_name)

    # 3b. Evaluate completions and save results on evaluation datasets
    for dataset_name in cfg.evaluation_datasets:
        evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
        evaluate_completions_and_save_results_for_dataset(cfg, 'ablation', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
        evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
    
    # 4a. Generate and save completions on positive sentiment dataset (tests if adding direction increases positivity)
    positive_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)

    generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', 'positive', dataset=positive_test)
    
    # Add positivity direction to boost positive sentiment
    actadd_positive_pre_hooks, actadd_positive_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=+1.0))], []
    generate_and_save_completions_for_dataset(cfg, model_base, actadd_positive_pre_hooks, actadd_positive_hooks, 'actadd', 'positive', dataset=positive_test)

    # 4b. Evaluate completions and save results on positive sentiment dataset
    evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', 'positive', eval_methodologies=cfg.refusal_eval_methodologies)
    evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', 'positive', eval_methodologies=cfg.refusal_eval_methodologies)

    # 5. Evaluate loss on positive sentiment datasets
    evaluate_loss_for_datasets(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline')
    evaluate_loss_for_datasets(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation')
    evaluate_loss_for_datasets(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd')

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path)
