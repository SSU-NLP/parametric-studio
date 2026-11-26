# The only source of truth for what's loadable. Add models here.
MODELS = [
    {"id": "Qwen/Qwen2.5-0.5B-Instruct", "label": "Qwen2.5-0.5B"},
    {"id": "Qwen/Qwen2.5-1.5B-Instruct", "label": "Qwen2.5-1.5B"},
]


def available_models():
    return MODELS
