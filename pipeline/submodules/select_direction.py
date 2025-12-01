import json
import torch
import functools
import math
import matplotlib.pyplot as plt
import os

from typing import List, Optional
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from einops import rearrange

from pipeline.model_utils.model_base import ModelBase
from pipeline.utils.hook_utils import add_hooks, get_activation_addition_input_pre_hook, get_direction_ablation_input_pre_hook, get_direction_ablation_output_hook

def sentiment_score(
    logits: Float[Tensor, 'batch seq d_vocab_out'],
    positive_toks: Int[Tensor, 'batch seq'],
    epsilon: Float = 1e-8,
):
    """
    Compute a score measuring the model's tendency toward positive sentiment.
    Higher score = more positive, lower score = more negative.
    """
    logits = logits.to(torch.float64)

    # we only care about the last tok position
    logits = logits[:, -1, :]

    probs = torch.nn.functional.softmax(logits, dim=-1)
    positive_probs = probs[:, positive_toks].sum(dim=-1)

    negative_probs = torch.ones_like(positive_probs) - positive_probs
    return torch.log(positive_probs + epsilon) - torch.log(negative_probs + epsilon)

def get_sentiment_scores(model, instructions, tokenize_instructions_fn, positive_toks, fwd_pre_hooks=[], fwd_hooks=[], batch_size=32):
    """
    Get sentiment scores for a batch of instructions.
    Higher score = more positive sentiment, lower score = more negative sentiment.
    """
    sentiment_score_fn = functools.partial(sentiment_score, positive_toks=positive_toks)

    sentiment_scores = torch.zeros(len(instructions), device=model.device)

    for i in range(0, len(instructions), batch_size):
        tokenized_instructions = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            logits = model(
                input_ids=tokenized_instructions.input_ids.to(model.device),
                attention_mask=tokenized_instructions.attention_mask.to(model.device),
            ).logits

        sentiment_scores[i:i+batch_size] = sentiment_score_fn(logits=logits)

    return sentiment_scores

def get_last_position_logits(model, tokenizer, instructions, tokenize_instructions_fn, fwd_pre_hooks=[], fwd_hooks=[], batch_size=32) -> Float[Tensor, "n_instructions d_vocab"]:
    if len(instructions) == 0:
        raise ValueError("Cannot get logits for empty instructions list")
    
    last_position_logits = None

    for i in range(0, len(instructions), batch_size):
        tokenized_instructions = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            logits = model(
                input_ids=tokenized_instructions.input_ids.to(model.device),
                attention_mask=tokenized_instructions.attention_mask.to(model.device),
            ).logits

        if last_position_logits is None:
            last_position_logits = logits[:, -1, :]
        else:
            last_position_logits = torch.cat((last_position_logits, logits[:, -1, :]), dim=0)

    return last_position_logits

def plot_sentiment_scores(
    sentiment_scores: Float[Tensor, 'n_pos n_layer'],
    baseline_sentiment_score: Optional[float],
    token_labels: List[str],
    title: str,
    artifact_dir: str,
    artifact_name: str,
):
    n_pos, n_layer = sentiment_scores.shape

    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(9, 5))  # width and height in inches

    # Add a trace for each position to extract
    for i in range(-n_pos, 0):
        ax.plot(
            list(range(n_layer)),
            sentiment_scores[i].cpu().numpy(),
            label=f'{i}: {repr(token_labels[i])}'
        )

    if baseline_sentiment_score is not None:
        # Add a horizontal line for the baseline
        ax.axhline(y=baseline_sentiment_score, color='black', linestyle='--')
        ax.annotate('Baseline', xy=(1, baseline_sentiment_score), xytext=(8, 10), 
                    xycoords=('axes fraction', 'data'), textcoords='offset points',
                    horizontalalignment='right', verticalalignment='center')

    ax.set_title(title)
    ax.set_xlabel('Layer source of direction (resid_pre)')
    ax.set_ylabel('Sentiment score')
    ax.legend(title='Position source of direction', loc='lower left')

    plt.savefig(f"{artifact_dir}/{artifact_name}.png")

# returns True if the direction should be filtered out
def filter_fn(sentiment_score, steering_score, kl_div_score, layer, n_layer, kl_threshold=None, induce_positive_threshold=None, prune_layer_percentage=0.20) -> bool:
    """
    Filter out candidate directions based on various criteria.
    
    Args:
        sentiment_score: Score after ablation (lower = more negative sentiment after ablation = better)
        steering_score: Score after adding direction (higher = more positive sentiment = better)
        kl_div_score: KL divergence score (lower = less disruption = better)
        layer: Source layer of the direction
        n_layer: Total number of layers
        kl_threshold: Maximum allowed KL divergence
        induce_positive_threshold: Minimum steering score to induce positive sentiment
        prune_layer_percentage: Discard directions from last X% of layers
    """
    if math.isnan(sentiment_score) or math.isnan(steering_score) or math.isnan(kl_div_score):
        return True
    if prune_layer_percentage is not None and layer >= int(n_layer * (1.0 - prune_layer_percentage)):
        return True
    if kl_threshold is not None and kl_div_score > kl_threshold:
        return True
    if induce_positive_threshold is not None and steering_score < induce_positive_threshold:
        return True
    return False

def select_direction(
    model_base: ModelBase,
    negative_instructions,
    positive_instructions,
    candidate_directions: Float[Tensor, 'n_pos n_layer d_model'],
    artifact_dir,
    kl_threshold=0.1, # directions with larger KL score are filtered out
    induce_positive_threshold=0.0, # directions with a lower inducing positive sentiment score are filtered out
    prune_layer_percentage=0.2, # discard the directions extracted from the last 20% of the model
    batch_size=32
):
    """
    Select the best direction for sentiment manipulation.
    
    The goal is to find a "positivity direction" that, when ablated, causes the model
    to generate more negative sentiment (similar to how ablating the refusal direction
    causes harmful content generation).
    
    Args:
        model_base: The model wrapper
        negative_instructions: Instructions/prompts that should elicit negative sentiment
        positive_instructions: Instructions/prompts that should elicit positive sentiment
        candidate_directions: Candidate directions to evaluate [n_pos, n_layer, d_model]
        artifact_dir: Directory to save artifacts
        kl_threshold: Maximum KL divergence allowed
        induce_positive_threshold: Minimum score for inducing positive sentiment
        prune_layer_percentage: Percentage of final layers to discard
        batch_size: Batch size for processing
    
    Returns:
        (pos, layer, direction): The best position, layer, and direction vector
    """
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    # Validate inputs
    assert len(negative_instructions) > 0, "negative_instructions is empty! Check your dataset or filtering settings."
    assert len(positive_instructions) > 0, "positive_instructions is empty! Check your dataset or filtering settings."
    
    print(f"Evaluating with {len(negative_instructions)} negative and {len(positive_instructions)} positive instructions")

    n_pos, n_layer, d_model = candidate_directions.shape

    # Baseline sentiment scores: positive_toks measures probability of positive sentiment tokens
    # Higher score = more positive sentiment
    baseline_sentiment_scores_negative = get_sentiment_scores(model_base.model, negative_instructions, model_base.tokenize_instructions_fn, model_base.positive_toks, fwd_hooks=[], batch_size=batch_size)
    baseline_sentiment_scores_positive = get_sentiment_scores(model_base.model, positive_instructions, model_base.tokenize_instructions_fn, model_base.positive_toks, fwd_hooks=[], batch_size=batch_size)

    ablation_kl_div_scores = torch.zeros((n_pos, n_layer), device=model_base.model.device, dtype=torch.float64)
    ablation_sentiment_scores = torch.zeros((n_pos, n_layer), device=model_base.model.device, dtype=torch.float64)
    steering_sentiment_scores = torch.zeros((n_pos, n_layer), device=model_base.model.device, dtype=torch.float64)

    baseline_positive_logits = get_last_position_logits(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=positive_instructions,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        fwd_pre_hooks=[],
        fwd_hooks=[],
        batch_size=batch_size
    )

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing KL for source position {source_pos}"):

            ablation_dir = candidate_directions[source_pos, source_layer]
            fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]
            fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]
            fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]

            intervention_logits: Float[Tensor, "n_instructions 1 d_vocab"] = get_last_position_logits(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                instructions=positive_instructions,
                tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                fwd_pre_hooks=fwd_pre_hooks,
                fwd_hooks=fwd_hooks,
                batch_size=batch_size
            )

            ablation_kl_div_scores[source_pos, source_layer] = kl_div_fn(baseline_positive_logits, intervention_logits, mask=None).mean(dim=0).item()

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing sentiment ablation for source position {source_pos}"):

            ablation_dir = candidate_directions[source_pos, source_layer]
            fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]
            fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]
            fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]

            # Test ablation on negative instructions: after ablating the positivity direction,
            # the model should become even more negative (lower sentiment score)
            sentiment_scores = get_sentiment_scores(model_base.model, negative_instructions, model_base.tokenize_instructions_fn, model_base.positive_toks, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
            ablation_sentiment_scores[source_pos, source_layer] = sentiment_scores.mean().item()

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing sentiment addition for source position {source_pos}"):

            positivity_vector = candidate_directions[source_pos, source_layer]
            coeff = torch.tensor(1.0)

            fwd_pre_hooks = [(model_base.model_block_modules[source_layer], get_activation_addition_input_pre_hook(vector=positivity_vector, coeff=coeff))]
            fwd_hooks = []

            # Test steering on positive instructions: adding the positivity direction
            # should make the model even more positive (higher sentiment score)
            sentiment_scores = get_sentiment_scores(model_base.model, positive_instructions, model_base.tokenize_instructions_fn, model_base.positive_toks, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
            steering_sentiment_scores[source_pos, source_layer] = sentiment_scores.mean().item()

    plot_sentiment_scores(
        sentiment_scores=ablation_sentiment_scores,
        baseline_sentiment_score=baseline_sentiment_scores_negative.mean().item(),
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='Ablating direction on negative sentiment prompts',
        artifact_dir=artifact_dir,
        artifact_name='ablation_scores'
    )

    plot_sentiment_scores(
        sentiment_scores=steering_sentiment_scores,
        baseline_sentiment_score=baseline_sentiment_scores_positive.mean().item(),
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='Adding direction on positive sentiment prompts',
        artifact_dir=artifact_dir,
        artifact_name='actadd_scores'
    )

    plot_sentiment_scores(
        sentiment_scores=ablation_kl_div_scores,
        baseline_sentiment_score=0.0,
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='KL Divergence when ablating direction on positive sentiment prompts',
        artifact_dir=artifact_dir,
        artifact_name='kl_div_scores'
    )

    filtered_scores = []
    json_output_all_scores = []
    json_output_filtered_scores = []

    for source_pos in range(-n_pos, 0):
        for source_layer in range(n_layer):

            json_output_all_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'sentiment_score': ablation_sentiment_scores[source_pos, source_layer].item(),
                'steering_score': steering_sentiment_scores[source_pos, source_layer].item(),
                'kl_div_score': ablation_kl_div_scores[source_pos, source_layer].item()
            })

            sentiment_score_val = ablation_sentiment_scores[source_pos, source_layer].item()
            steering_score = steering_sentiment_scores[source_pos, source_layer].item()
            kl_div_score = ablation_kl_div_scores[source_pos, source_layer].item()

            # We sort directions in descending order (from highest to lowest sorting score)
            # The intervention is better at inducing negative sentiment if the sentiment score is low
            # (more negative), so we multiply by -1
            sorting_score = -sentiment_score_val

            # Filter out directions based on criteria
            discard_direction = filter_fn(
                sentiment_score=sentiment_score_val,
                steering_score=steering_score,
                kl_div_score=kl_div_score,
                layer=source_layer,
                n_layer=n_layer,
                kl_threshold=kl_threshold,
                induce_positive_threshold=induce_positive_threshold,
                prune_layer_percentage=prune_layer_percentage
            )

            if discard_direction:
                continue

            filtered_scores.append((sorting_score, source_pos, source_layer))

            json_output_filtered_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'sentiment_score': ablation_sentiment_scores[source_pos, source_layer].item(),
                'steering_score': steering_sentiment_scores[source_pos, source_layer].item(),
                'kl_div_score': ablation_kl_div_scores[source_pos, source_layer].item()
            })   

    with open(f"{artifact_dir}/direction_evaluations.json", 'w') as f:
        json.dump(json_output_all_scores, f, indent=4)

    json_output_filtered_scores = sorted(json_output_filtered_scores, key=lambda x: x['sentiment_score'], reverse=False)

    with open(f"{artifact_dir}/direction_evaluations_filtered.json", 'w') as f:
        json.dump(json_output_filtered_scores, f, indent=4)

    assert len(filtered_scores) > 0, "All scores have been filtered out!"

    # sorted in descending order
    filtered_scores = sorted(filtered_scores, key=lambda x: x[0], reverse=True)

    # now return the best position, layer, and direction
    score, pos, layer = filtered_scores[0]

    print(f"Selected direction: position={pos}, layer={layer}")
    print(f"Sentiment score after ablation: {ablation_sentiment_scores[pos, layer]:.4f} (baseline: {baseline_sentiment_scores_negative.mean().item():.4f})")
    print(f"Steering score: {steering_sentiment_scores[pos, layer]:.4f} (baseline: {baseline_sentiment_scores_positive.mean().item():.4f})")
    print(f"KL Divergence: {ablation_kl_div_scores[pos, layer]:.4f}")
    
    return pos, layer, candidate_directions[pos, layer]

def masked_mean(seq, mask = None, dim = 1, keepdim = False):
    if mask is None:
        return seq.mean(dim = dim)

    if seq.ndim == 3:
        mask = rearrange(mask, 'b n -> b n 1')

    masked_seq = seq.masked_fill(~mask, 0.)
    numer = masked_seq.sum(dim = dim, keepdim = keepdim)
    denom = mask.sum(dim = dim, keepdim = keepdim)

    masked_mean = numer / denom.clamp(min = 1e-3)
    masked_mean = masked_mean.masked_fill(denom == 0, 0.)
    return masked_mean

def kl_div_fn(
    logits_a: Float[Tensor, 'batch seq_pos d_vocab'],
    logits_b: Float[Tensor, 'batch seq_pos d_vocab'],
    mask: Int[Tensor, "batch seq_pos"]=None,
    epsilon: Float=1e-6
) -> Float[Tensor, 'batch']:
    """
    Compute the KL divergence loss between two tensors of logits.
    """
    logits_a = logits_a.to(torch.float64)
    logits_b = logits_b.to(torch.float64)

    probs_a = logits_a.softmax(dim=-1)
    probs_b = logits_b.softmax(dim=-1)

    kl_divs = torch.sum(probs_a * (torch.log(probs_a + epsilon) - torch.log(probs_b + epsilon)), dim=-1)

    if mask is None:
        return torch.mean(kl_divs, dim=-1)
    else:
        return masked_mean(kl_divs, mask).mean(dim=-1)