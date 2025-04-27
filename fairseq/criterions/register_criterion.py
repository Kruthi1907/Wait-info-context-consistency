from fairseq import criterions

# Define the decorator
def register_criterion(name):
    def register_criterion_fn(cls):
        if name in criterions.CRIETERION_REGISTRY:
            raise ValueError(f"Criterion {name} is already registered.")
        criterions.CRIETERION_REGISTRY[name] = cls
        return cls
    return register_criterion_fn
