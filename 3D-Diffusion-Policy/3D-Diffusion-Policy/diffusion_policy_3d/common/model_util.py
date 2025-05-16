from termcolor import cprint

def print_params(model):
    """
    Print the number of parameters in each part of the model.
    """    
    params_dict = {}

    all_num_param = sum(p.numel() for p in model.parameters())

    for name, param in model.named_parameters():
        part_name = name.split('.')[0]
        if part_name not in params_dict:
            params_dict[part_name] = 0
        params_dict[part_name] += param.numel()

    cprint(f'----------------------------------', 'cyan')
    cprint(f'Class name: {model.__class__.__name__}', 'cyan')
    cprint(f'  Number of parameters: {all_num_param / 1e6:.4f}M', 'cyan')
    for part_name, num_params in params_dict.items():
        cprint(f'   {part_name}: {num_params / 1e6:.4f}M ({num_params / all_num_param:.2%})', 'cyan')
    cprint(f'----------------------------------', 'cyan')

def print_params_v2(model):
    """
    Print the total and learnable parameters in the model, and for each part, formatted similarly to print_params.
    """
    params_dict = {}
    learnable_params_dict = {}

    all_num_param = sum(p.numel() for p in model.parameters())
    all_learnable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for name, param in model.named_parameters():
        part_name = name.split('.')[0]
        if part_name not in params_dict:
            params_dict[part_name] = 0
            learnable_params_dict[part_name] = 0
        params_dict[part_name] += param.numel()
        if param.requires_grad:
            learnable_params_dict[part_name] += param.numel()

    cprint(f'----------------------------------', 'cyan')
    cprint(f'Class name: {model.__class__.__name__}', 'cyan')
    cprint(f'  Total params: {all_num_param}', 'cyan')
    cprint(f'  Total learnable params: {all_learnable_param}', 'cyan')
    for part_name in params_dict:
        total = params_dict[part_name]
        learnable = learnable_params_dict[part_name]
        cprint(f'   {part_name}: All: {total/ 1e6:.4f}M, Learnable: {learnable/ 1e6:.4f}M ({learnable/ all_learnable_param:.2%})', 'cyan')
    cprint(f'----------------------------------', 'cyan') 
