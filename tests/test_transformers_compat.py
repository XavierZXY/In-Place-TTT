from functools import wraps

from in_place_ttt.transformers_compat import resolve_check_model_inputs


def test_resolve_check_model_inputs_supports_direct_decorator_signature():
    def direct_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            kwargs["direct_seen"] = True
            return func(self, *args, **kwargs)

        return wrapper

    decorator = resolve_check_model_inputs(direct_decorator)

    @decorator
    def forward(self, **kwargs):
        return kwargs

    assert forward(object(), input_ids=[1, 2, 3]) == {"input_ids": [1, 2, 3], "direct_seen": True}


def test_resolve_check_model_inputs_supports_factory_decorator_signature():
    def factory_decorator(tie_last_hidden_states=True):
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                kwargs["factory_seen"] = tie_last_hidden_states
                return func(self, *args, **kwargs)

            return wrapper

        return decorator

    decorator = resolve_check_model_inputs(factory_decorator)

    @decorator
    def forward(self, **kwargs):
        return kwargs

    assert forward(object(), input_ids=[1, 2, 3]) == {"input_ids": [1, 2, 3], "factory_seen": True}
