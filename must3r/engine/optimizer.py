# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import json


def _get_num_layer_for_vit(var_name, depth, offset=0):
    if var_name.startswith('feat_embed') or var_name.startswith("patch_embed"):
        return 0 + offset
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1 + offset
    elif var_name.startswith('norm'):  # part of the last block
        return depth + offset
    elif any(var_name.startswith(k) for k in ['head', 'prediction_head']):
        return depth + 1 + offset
    else:
        raise NotImplementedError(var_name)


def get_parameter_groups(model, offset, weight_decay, layer_decay=1.0, skip_list=(), no_lr_scale_list=[]):
    parameter_group_names = {}
    parameter_group_vars = {}
    depth = None
    # prepare layer decay values
    assert layer_decay == 1.0 or 0. < layer_decay < 1.
    if layer_decay < 1.:
        depth = model.depth
        num_layers = depth + offset
        layer_decay_values = list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        # Assign weight decay values
        if name.endswith(".bias") or 'norm' in name or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        # Assign layer ID for LR scaling
        if layer_decay < 1.:
            skip_scale = False
            layer_id = _get_num_layer_for_vit(name, depth, offset)
            group_name = "layer_%d_%s" % (layer_id, group_name)
            if name in no_lr_scale_list:
                skip_scale = True
                group_name = f'{group_name}_no_lr_scale'
        else:
            layer_id = 0
            skip_scale = True

        if group_name not in parameter_group_names:
            if not skip_scale:
                scale = layer_decay_values[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())
