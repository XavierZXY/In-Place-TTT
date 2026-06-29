from contextlib import contextmanager

import torch


def unwrap_model_for_attr(model):
    while hasattr(model, "module"):
        model = model.module
    return model


_MISSING = object()


def _iter_decoder_layers(model):
    root = unwrap_model_for_attr(model)
    inner_model = getattr(root, "model", None) or root
    return list(getattr(inner_model, "layers", []))


@contextmanager
def _temporary_training_mode(module, training: bool):
    module_states = [(child, child.training) for child in module.modules()]
    module.train(training)
    try:
        yield
    finally:
        for child, was_training in module_states:
            child.train(was_training)


@contextmanager
def _temporary_ttt_enabled(model, enabled: bool):
    root = unwrap_model_for_attr(model)
    inner_model = getattr(root, "model", None) or root
    saved_attrs = []

    for owner, attr in ((inner_model, "ttt_mode"), (getattr(inner_model, "config", None), "ttt_mode")):
        if owner is None or not hasattr(owner, attr):
            continue
        value = getattr(owner, attr)
        saved_attrs.append((owner, attr, value))
        setattr(owner, attr, bool(enabled) and bool(value))

    saved_layers = []
    for layer in getattr(inner_model, "layers", []):
        if not hasattr(layer, "is_ttt_layer"):
            continue
        value = getattr(layer, "is_ttt_layer")
        saved_layers.append((layer, value))
        setattr(layer, "is_ttt_layer", bool(enabled) and bool(value))

    try:
        yield
    finally:
        for owner, attr, value in saved_attrs:
            setattr(owner, attr, value)
        for layer, value in saved_layers:
            setattr(layer, "is_ttt_layer", value)


def _snapshot_ttt_monitor_stats(model):
    saved = []
    for layer in _iter_decoder_layers(model):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        saved.append((mlp, getattr(mlp, "_last_ttt_monitor_stats", _MISSING)))
    return saved


def _restore_ttt_monitor_stats(saved):
    for mlp, value in saved:
        if value is _MISSING:
            if hasattr(mlp, "_last_ttt_monitor_stats"):
                delattr(mlp, "_last_ttt_monitor_stats")
        else:
            mlp._last_ttt_monitor_stats = value


def _logits_sample(model, micro_batch, *, sample_tokens: int, sample_dim: int):
    root = unwrap_model_for_attr(model)
    inner_model = getattr(root, "model", None)
    lm_head = getattr(root, "lm_head", None)
    if inner_model is None or lm_head is None:
        return None

    forward_kwargs = {
        key: micro_batch[key]
        for key in ("input_ids", "attention_mask", "position_ids", "inputs_embeds", "cache_position")
        if key in micro_batch
    }
    forward_kwargs["use_cache"] = False
    outputs = inner_model(**forward_kwargs)
    hidden_states = outputs.last_hidden_state
    if hidden_states.shape[1] == 0:
        return None

    sample_tokens = min(sample_tokens, hidden_states.shape[1])
    hidden_states = hidden_states[:, -sample_tokens:, :]
    logits = lm_head(hidden_states)
    if sample_dim > 0:
        logits = logits[..., : min(sample_dim, logits.shape[-1])]
    return logits.detach().float()


def compute_ttt_logits_delta_sample_ratio(
    model,
    micro_batch,
    *,
    sample_tokens: int = 1,
    sample_dim: int = 0,
):
    sample_tokens = int(sample_tokens)
    sample_dim = int(sample_dim)
    if sample_tokens <= 0 or sample_dim < 0:
        return None

    root = unwrap_model_for_attr(model)
    inner_model = getattr(root, "model", None)
    if inner_model is None or not getattr(inner_model, "ttt_mode", False):
        return None
    if not any(getattr(layer, "is_ttt_layer", False) for layer in getattr(inner_model, "layers", [])):
        return None

    saved_monitor_stats = _snapshot_ttt_monitor_stats(model)
    try:
        with torch.no_grad(), _temporary_training_mode(root, False):
            with _temporary_ttt_enabled(root, True):
                ttt_logits = _logits_sample(
                    root,
                    micro_batch,
                    sample_tokens=sample_tokens,
                    sample_dim=sample_dim,
                )
            with _temporary_ttt_enabled(root, False):
                base_logits = _logits_sample(
                    root,
                    micro_batch,
                    sample_tokens=sample_tokens,
                    sample_dim=sample_dim,
                )
    finally:
        _restore_ttt_monitor_stats(saved_monitor_stats)

    if ttt_logits is None or base_logits is None:
        return None
    eps = torch.tensor(1e-12, device=ttt_logits.device, dtype=torch.float32)
    ratio = (ttt_logits - base_logits).norm() / ttt_logits.norm().clamp_min(eps)
    return torch.nan_to_num(ratio.detach().float())


def get_last_ttt_aux_loss(model_outputs, model):
    aux_loss = getattr(model_outputs, "ttt_aux_loss", None)
    if aux_loss is not None:
        return aux_loss

    root = unwrap_model_for_attr(model)
    inner_model = getattr(root, "model", None) or root
    if inner_model is None:
        return None
    return getattr(inner_model, "_last_ttt_aux_loss", None)


def collect_ttt_aux_params(model, *, require_grad_only: bool = True):
    root = unwrap_model_for_attr(model)
    inner_model = getattr(root, "model", None) or root
    if inner_model is None:
        return []

    params = []
    seen = set()
    for layer in getattr(inner_model, "layers", []):
        if not getattr(layer, "is_ttt_layer", False):
            continue
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "ttt_conv"):
            continue

        for module_name in ("ttt_conv", "ttt_proj", "ttt_key_norm"):
            module = getattr(mlp, module_name, None)
            param = getattr(module, "weight", None) if module is not None else None
            if param is None or (require_grad_only and not param.requires_grad):
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            params.append(param)
    return params


def configure_ttt_only_trainable_params(model):
    ttt_param_ids = {id(param) for param in collect_ttt_aux_params(model, require_grad_only=False)}
    if not ttt_param_ids:
        return 0

    trainable_params = 0
    for param in model.parameters():
        is_ttt_param = id(param) in ttt_param_ids
        param.requires_grad_(is_ttt_param)
        if is_ttt_param:
            trainable_params += param.numel()
    return trainable_params


def pop_ttt_monitor_stats(model):
    root = unwrap_model_for_attr(model)
    inner_model = getattr(root, "model", None) or root
    if inner_model is None:
        return {}

    values = {}
    for layer in getattr(inner_model, "layers", []):
        if not getattr(layer, "is_ttt_layer", False):
            continue
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        stats = getattr(mlp, "_last_ttt_monitor_stats", None)
        mlp._last_ttt_monitor_stats = None
        if not stats:
            continue
        for key, value in stats.items():
            if not torch.is_tensor(value):
                continue
            values.setdefault(key, []).append(torch.nan_to_num(value.detach().float()))

    if not values:
        return {}
    return {key: torch.stack(tensors).mean() for key, tensors in values.items() if tensors}


def build_ttt_optimizer_param_groups(
    model,
    *,
    base_lr: float,
    base_weight_decay: float,
    lr_multiplier: float = 1.0,
    weight_decay: float | None = None,
):
    lr_multiplier = float(lr_multiplier)
    if lr_multiplier <= 0.0:
        raise ValueError(f"ttt_param_lr_multiplier must be > 0, got {lr_multiplier}")

    ttt_weight_decay = base_weight_decay if weight_decay is None else float(weight_decay)
    if ttt_weight_decay < 0.0:
        raise ValueError(f"ttt_param_weight_decay must be >= 0, got {ttt_weight_decay}")

    if lr_multiplier == 1.0 and weight_decay is None:
        return None

    ttt_param_ids = {id(param) for param in collect_ttt_aux_params(model)}
    if not ttt_param_ids:
        return None

    base_params = []
    ttt_params = []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in ttt_param_ids:
            ttt_params.append(param)
        else:
            base_params.append(param)

    if not ttt_params:
        return None

    param_groups = []
    if base_params:
        param_groups.append(
            {
                "params": base_params,
                "lr": float(base_lr),
                "weight_decay": float(base_weight_decay),
            }
        )
    param_groups.append(
        {
            "params": ttt_params,
            "lr": float(base_lr) * lr_multiplier,
            "weight_decay": ttt_weight_decay,
        }
    )
    return param_groups


def accumulate_ttt_aux_grads(model, aux_loss, aux_loss_weight, loss_scale):
    if aux_loss is None or aux_loss_weight <= 0.0:
        return None
    if not torch.is_tensor(aux_loss) or not aux_loss.requires_grad:
        return None

    target_params = collect_ttt_aux_params(model)
    if not target_params:
        return None

    scaled_aux_loss = aux_loss * (aux_loss_weight * loss_scale)
    aux_grads = torch.autograd.grad(
        scaled_aux_loss,
        target_params,
        allow_unused=True,
        retain_graph=True,
    )
    for param, grad in zip(target_params, aux_grads):
        if grad is None:
            continue
        grad = grad.detach()
        if param.grad is None:
            param.grad = grad.clone()
        else:
            param.grad.detach().add_(grad)
    return scaled_aux_loss.detach()
