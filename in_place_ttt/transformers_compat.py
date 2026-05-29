def resolve_check_model_inputs(check_model_inputs_fn):
    try:
        return check_model_inputs_fn()
    except TypeError:
        return check_model_inputs_fn
