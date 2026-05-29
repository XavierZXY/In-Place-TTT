import torch


def unwrap_model_for_attr(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def get_last_ttt_aux_loss(model_outputs, model):
    aux_loss = getattr(model_outputs, "ttt_aux_loss", None)
    if aux_loss is not None:
        return aux_loss

    root = unwrap_model_for_attr(model)
    inner_model = getattr(root, "model", None)
    if inner_model is None:
        return None
    return getattr(inner_model, "_last_ttt_aux_loss", None)


def collect_ttt_aux_params(model):
    root = unwrap_model_for_attr(model)
    inner_model = getattr(root, "model", None)
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

        for module_name in ("ttt_conv", "ttt_proj"):
            module = getattr(mlp, module_name, None)
            param = getattr(module, "weight", None) if module is not None else None
            if param is None or not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            params.append(param)
    return params


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
        retain_graph=False,
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
