import models


def build(model_key, model_params, ann_params, ip_dim):
    """ Builds a model specified by model_key and returns it """

    model_class = getattr(models, model_key)
    model = model_class(input_dim=ip_dim, ann_params=ann_params, **model_params)
    return model
