import torch.optim
import models


def build(input_dim, config, device):
    """ Builds a model, optimizer and learning rate scheduler before wrapping them together """
    model = build_model(model_key=config.get_model_key(),
                        model_params=config.get_model_params(),
                        ann_params=config.get_ann_params(),
                        ip_dim=input_dim)

    optimizer, lr_scheduler = build_optimizer(model=model,
                                              optimizer_key=config.get_optimizer_key(),
                                              optimizer_params=config.get_optimizer_params(),
                                              lr_sched_key=config.get_lr_scheduler_key(),
                                              lr_sched_params=config.get_lr_scheduler_params())

    return build_wrapper(config.get_model_wrapper(), model, optimizer, lr_scheduler, device)


def build_model(model_key, model_params, ann_params, ip_dim):
    """ Builds a model specified by model_key and returns it """

    model_class = getattr(models, model_key)
    model = model_class(input_dim=ip_dim, ann_params=ann_params, **model_params)
    return model


def build_optimizer(model, optimizer_key, optimizer_params, lr_sched_key, lr_sched_params):
    """ Builds an optimizer with a learning rate scheduler and returns both """
    optimizer = getattr(torch.optim, optimizer_key)(model.parameters(), **optimizer_params)
    lr_scheduler = getattr(torch.optim.lr_scheduler, lr_sched_key)(optimizer, **lr_sched_params)

    return optimizer, lr_scheduler


def build_wrapper(wrapper_key, model, optimizer, lr_scheduler, device):
    """ Wraps the model, optimizer and lr_scheduler together """
    wrapper_class = getattr(models, wrapper_key)
    return wrapper_class(model, optimizer, lr_scheduler, device)