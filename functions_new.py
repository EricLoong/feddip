from functools import partial
import numpy as np
import torch
import copy
import time
import torch.nn as nn
import torch.nn.functional as F
import types

def keep_k_error(error, amount=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shape = error.size()
    k = int(error.nelement() * amount)
    value, pos = torch.topk(torch.abs(error.flatten()), k=k)
    temp_err = torch.zeros(shape).flatten().to(device)
    temp_err[pos] = error.flatten()[pos]
    err_sparse = temp_err.reshape(shape)
    return err_sparse

# wap_list,resi_list,bm_list = weight_approx_sparse_prune(test_model_reg)
# new_model = recover_from_wap(wap=wap_list,pos=resi_list,bm_list=bm_list,model=copy.deepcopy(test_model_reg))
# torch.sum(new_model.state_dict()['features.3.weight']-test_model_reg.state_dict()['features.3.weight'])

def norm_whole_model(model):
    norm_sum = 0
    for keys in model.state_dict().keys():
        # print(torch.norm(model.state_dict()[keys]),torch.pow(torch.norm(model.state_dict()[keys]),2))
        norm_sum += torch.pow(torch.norm(model.state_dict()[keys].type(torch.float)), 2)

    return norm_sum


def inform_loss_norm(model1, model2):
    norm_sum_up = 0
    for keys in model1.state_dict().keys():
        # print(torch.norm(model.state_dict()[keys]),torch.pow(torch.norm(model.state_dict()[keys]),2))
        norm_sum_up += torch.pow(torch.norm(model1.state_dict()[keys].type(torch.float) -
                                            model2.state_dict()[keys].type(torch.float)), 2)

    norm_sum_down = norm_whole_model(model1)

    loss_p = norm_sum_up / norm_sum_down
    print('Additional Pruning at server with information loss percentage:', loss_p * 100, '%')
    return loss_p.cpu().detach().numpy()


# Prune Models Functions
import torch.nn.utils.prune as prune


def get_name_index(model, fc=False):
    index_list_features = list()
    index_list_classifier = list()
    if fc == False:
        for name, parameters in model.features.named_parameters():
            name_split = name.split('.')
            temp_index = eval(name_split[0])
            if temp_index not in index_list_features:
                index_list_features.append(temp_index)
    else:
        for name, parameters in model.features.named_parameters():
            name_split = name.split('.')
            temp_index = eval(name_split[0])
            if temp_index not in index_list_features:
                index_list_features.append(temp_index)
        for name, parameters in model.classifier.named_parameters():
            name_split = name.split('.')
            temp_index = eval(name_split[0])
            if temp_index not in index_list_classifier:
                index_list_classifier.append(temp_index)
    return index_list_features, index_list_classifier


def get_weight_fc_resnet(model_resnet, bias=False, only_conv=True):
    model = model_resnet
    if only_conv:
        if bias:
            keys = [key for key in model.state_dict().keys() if 'weight' in key or 'bias' in key]
        else:
            keys = [key for key in model.state_dict().keys() if ('weight' in key and 'conv' in key) or ('linear' in key
                                                                                                        and 'weight' in key)]
    else:
        if bias:
            keys = [key for key in model.state_dict().keys() if 'weight' in key or 'bias' in key]
        else:
            keys = [key for key in model.state_dict().keys() if 'weight' in key]

    return keys


def basic_generate_resnet(model_resnet, bias=True):
    keys = get_weight_fc_resnet(model_resnet=model_resnet,bias=bias)
    prune_parameters = []
    for key in keys:
        temp_key_split = key.split('.')
        temp_l = len(temp_key_split)
        temp_attr = model_resnet
        for i in range(temp_l - 1):
            temp_attr = getattr(temp_attr, temp_key_split[i])
        prune_parameters.append([temp_attr, temp_key_split[-1]])
    return prune_parameters


def basic_generate(model, name, fc=True):
    '''
    :param model: the target model
    :param name: classifier or features
    :param fc: fully connected layers
    :return:
    '''
    parameters_to_prune = list()
    if fc:
        ind_f, ind_c = get_name_index(model=model, fc=fc)
        for index in ind_f:
            temp_prune = getattr(model.features, str(index))
            parameters_to_prune.append((temp_prune, name))
        for index in ind_c:
            temp_prune = getattr(model.classifier, str(index))
            parameters_to_prune.append((temp_prune, name))
    else:
        ind_f = get_name_index(model=model, fc=fc)[0]
        for index in ind_f:
            temp_prune = getattr(model.features, str(index))
            parameters_to_prune.append((temp_prune, name))

    return parameters_to_prune


def generate_prune_param(model, fc=True, bias=True):
    basic_param = basic_generate(model=model, name='weight', fc=fc)
    if bias == True:
        basic_param = basic_param + basic_generate(model=model, name='bias', fc=fc)

    return basic_param

def prune_4_mask( model, sparsity,bias=False):
    prune_para = generate_prune_param(model=model,bias=bias,fc=True)

    prune.global_unstructured(prune_para, pruning_method=prune.L1Unstructured, amount=sparsity, )
    mask_dict = dict()
    for key in model.state_dict().keys():
        if key.endswith('_mask'):
            mask_dict[key] = model.state_dict()[key]
        # Clean the pruned model
    for module, name in prune_para:
        prune.remove(module, name)
    return mask_dict

def prune_4_mask_resnet( model, sparsity,bias=False):
    prune_para = basic_generate_resnet(model_resnet=model, bias=bias)
    prune.global_unstructured(prune_para, pruning_method=prune.L1Unstructured, amount=sparsity, )
    mask_dict = dict()
    for key in model.state_dict().keys():
        if key.endswith('_mask'):
            mask_dict[key] = model.state_dict()[key]
    for module, name in prune_para:
        prune.remove(module, name)
    return mask_dict

def prune_clean_model(param_pruned, amount=0.2):
    '''
    :param param_pruned:
    :param amount: the degree of sparsity
    :return: get a clean model with removing module like weight.ori and weight.mask
    '''
    prune.global_unstructured(param_pruned, pruning_method=prune.L1Unstructured, amount=amount)

    for module, name in param_pruned:
            prune.remove(module, name)
    return None


def print_sparsity(parameters_to_prune):
    '''
    :param parameters_to_prune: The parameters module and name ready to be pruned
    :return: print the sparsity of the model
    '''
    denominator = list()
    nominator = list()
    sparsity_list = list()
    for weight, name in parameters_to_prune:
        temp_denominator = float(torch.sum(getattr(weight, name) == 0))
        temp_nominator = float(getattr(weight, name).nelement())
        print(
            f'Layer {weight}', "Sparsity is: {:.2f}% ".format(
                100. * temp_denominator
                / temp_nominator
            ))
        sparsity_list.append(round(np.sum(temp_denominator) / np.sum(temp_nominator),4))
        denominator.append(temp_denominator)
        nominator.append(temp_nominator)
    print("Total Sparsity is: {:.2f}%".format(
        np.sum(denominator) / np.sum(nominator) * 100
    ))
    total_sp =  np.sum(denominator) / np.sum(nominator) * 100
    return sparsity_list, total_sp


def compute_sparsity(model):
    total_elements = 0
    zero_elements = 0

    for param in model.parameters():
        param_elements = torch.numel(param)
        param_zero_elements = torch.sum(param == 0).item()

        total_elements += param_elements
        zero_elements += param_zero_elements

    sparsity = zero_elements / total_elements * 100
    print(f'Sparsity is {sparsity}')
    return sparsity

def compute_sparsity_weight(weight_mask):
    total_elements = 0
    zero_elements = 0

    for key, tensor in weight_mask.items():
        param_elements = torch.numel(tensor)
        param_zero_elements = torch.sum(tensor == 0).item()

        total_elements += param_elements
        zero_elements += param_zero_elements

    sparsity = zero_elements / total_elements * 100
    print(f'Sparsity is {sparsity}')
    return sparsity


def init_mask(model,args):
    '''
    :param model: The model not contained buffer (masks)
    :return: The mask.
    '''
    if args.parallel:
        prune_para = generate_prune_param(model=model.module, fc=True)
    else:
        prune_para = generate_prune_param(model=model, fc=True)
    prune.global_unstructured(prune_para, pruning_method=prune.L1Unstructured, amount=0, )
    mask_dict = dict()
    for key in model.state_dict().keys():
        if key.endswith('_mask'):
            mask_dict[key] = model.state_dict()[key]
    for module, name in prune_para:
        prune.remove(module, name)

    return mask_dict

def generate_mask(model, args):
    '''

    :param model: A model without mask function
    :param args: The args parameters
    :return: A dict with adding _mask to the original dictionary
    '''
    if args.parallel:
        model_dict = model.module.state_dict()
    else:
        model_dict = model.state_dict()
    for key in list(model_dict.keys()):
        if key.endswith('weight'):
            model_dict[key+'_mask'] = model_dict[key].bool()
    return model_dict

def prunefl_reconfig(aggregate_gradients, layer_times, prunable_params=0.3):
    with torch.no_grad():
        importances = []
        for i, g in aggregate_gradients.items():
            if "bias" not in i:
                #g.square_()
                g = g.div(layer_times[i])  # remember change the layer_times to satisfy this
                importances.append(g)

        t = 0.2
        delta = 0
        cat_grad = torch.cat([torch.flatten(g) for key, g in aggregate_gradients.items() if "bias" not in key])
        cat_imp = torch.cat([torch.flatten(g) for key, g in aggregate_gradients.items() if "bias" not in key])
        indices = torch.argsort(cat_grad, descending=True)
        n_required = (1 - prunable_params) * cat_grad.numel()
        n_grown = 0

        masks = dict()
        for i, g in aggregate_gradients.items():
            if "bias" not in i:
                print('layer name is:', i)
                masks[i] = torch.zeros_like(g, dtype=torch.bool)

        for j, i in enumerate(indices):
            if cat_imp[i] >= delta / t or n_grown <= n_required:
                index_within_layer = i.item()
                for layer in aggregate_gradients.keys():
                    if "bias" not in layer:
                        numel = aggregate_gradients[layer].numel()
                        if index_within_layer >= numel:
                            index_within_layer -= numel
                        else:
                            break

                delta += cat_grad[i]
                t += layer_times[layer]

                shape = tuple(masks[layer].shape)
                masks[layer][np.unravel_index(index_within_layer, shape)] = 1
                n_grown += 1
            else:
                break

        print('reconfig density', n_grown / cat_imp.numel())
        masks_weight = dict()
        for key in masks.keys():
            if 'weight' in key:
                masks_weight[key+'_mask'] = masks[key]

        return masks_weight



def number_parameters(model):
    n_ele = 0
    for _,para in model.named_parameters():
        n_ele += torch.numel(para)
    return n_ele

def comm_costs_in_mb(model,sparsity, mask=False):
    n_element = number_parameters(model=model)
    if mask:
        comm_costs = int(4*(1-sparsity)*n_element)
    else:
        comm_costs = int(32*(1-sparsity)*n_element)

    return comm_costs/(1024*1024)

def measure_layer_time_exclude_bias(model, inputs):
    layer_times = {}

    def forward_features_hook(layer_name, self, input, output):
        nonlocal layer_times
        start_time = time.time()
        _ = self.forward(input[0])
        elapsed_time = time.time() - start_time
        layer_times[layer_name] = layer_times.get(layer_name, 0) + elapsed_time

    handles = []

    for layer_name, layer in model.named_modules():
        if "bias" not in layer_name:
            for state_key in model.state_dict().keys():
                if layer_name in state_key:
                    layer_key = state_key
                    break
            handle = layer.register_forward_hook(partial(forward_features_hook, layer_key))
            handles.append(handle)

    with torch.no_grad():
        _ = model(inputs)

    for handle in handles:
        handle.remove()

    return layer_times




def create_dir_partition(train_ds, num_clients, num_classes, alpha=0.5, least_samples=100):
    dataset_content, dataset_label = train_ds

    dataidx_map = {}
    min_size = 0
    K = num_classes
    N = len(dataset_label)

    while min_size < least_samples:
        idx_batch = [[] for _ in range(num_clients)]

        for k in range(K):
            idx_k = np.where(dataset_label == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_clients):
        dataidx_map[j] = idx_batch[j]

    return dataidx_map


def create_pat_partition(train_ds, num_clients, num_classes, batch_size, class_per_client=2):
    least_samples = batch_size*2
    dataset_content, dataset_label = train_ds

    dataidx_map = {}
    idxs = np.array(range(len(dataset_label)))
    idx_for_each_class = []

    for i in range(num_classes):
        idx_for_each_class.append(np.array([idx for idx, label in enumerate(dataset_label) if label == i]))

    #print("Idx for each class:", idx_for_each_class)  # Print statement 1

    class_num_per_client = [class_per_client for _ in range(num_clients)]

    for i in range(num_classes):
        selected_clients = []

        for client in range(num_clients):
            if class_num_per_client[client] > 0:
                selected_clients.append(client)
            selected_clients = selected_clients[:int(num_clients / num_classes * class_per_client)]

        print(f"Class {i}, selected clients:", selected_clients)  # Print statement 2

        num_all_samples = len(idx_for_each_class[i])
        num_selected_clients = len(selected_clients)
        num_per = num_all_samples / num_selected_clients

        if num_per > 0:
            num_samples = np.random.randint(max(int(num_per / 10), int(least_samples / num_classes), int(num_per) - 1),
                                            int(num_per), num_selected_clients - 1).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            print(f"Class {i}, num_samples:", num_samples)  # Print statement 3

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample],
                                                    axis=0)
                #print(f"Client {client}, dataidx_map[client]:", dataidx_map[client])  # Print statement 4
                idx += num_sample
                class_num_per_client[client] -= 1

    return dataidx_map




def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


def SNIP_modified(net, keep_ratio, train_dataloader, device):
    # TODO: shuffle?
    # This is directly from SNIP official code, but we generate mask with name in dictionary.
    # Grab a single batch from the training dataset
    inputs, targets = next(iter(train_dataloader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net)

    masks = {}

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

            # Override the forward methods:
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(snip_forward_conv2d, layer)

            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    outputs = net.forward(inputs)
    loss = F.nll_loss(outputs, targets)
    loss.backward()

    grads_abs = {}
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads_abs[name] = torch.abs(layer.weight_mask.grad)

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs.values()])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    for name, g in grads_abs.items():
        masks[name+'.weight_mask'] = ((g / norm_factor) >= acceptable_score).float()

    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in masks.values()])))

    return masks

def set_weight_by_mask(model, mask):
    '''
    :param model: Original model
    :param mask: Mask to be applied
    :return: A model has been applied with mask
    '''
    with torch.no_grad():
        model_dict = model.state_dict()
        new_dict = dict.fromkeys(model_dict.keys())
        for key in list(model_dict.keys()):
            key_name = key+'_mask'
            if 'weight' in key and key_name in mask.keys():
                new_dict[key] = model_dict[key]*mask[key+'_mask']
            else:
                new_dict[key] = model_dict[key]
        model.load_state_dict(new_dict)

def score_grad_local(model, dataloader, mask, device):
    """
    Compute gradients and average them over whole local data.
    Returns a dictionary with layer names as keys and their average gradients as values.
    """
    # mask is to decide which layer's importance we should consider
    # Prepare dictionary to store gradients
    gradients = {}
    model = model.to(device)
    for name, layer in model.named_parameters():
        mask_name = name + '_mask'
        if mask_name in mask.keys() and layer.grad is not None:
            gradients[name] = torch.zeros(layer.grad.shape).to(device)

    # Take a whole epoch
    for batch_idx in range(len(dataloader)):
        x, y = next(iter(dataloader))
        x = x.to(device)
        y = y.to(device)

        # Compute gradients (but don't apply them)
        model.zero_grad()
        outputs = model.forward(x)
        loss = F.nll_loss(outputs, y)
        loss.backward()

        # Store gradients
        for name, layer in model.named_parameters():
            mask_name = name + '_mask'
            if mask_name in mask.keys() and layer.grad is not None:
                gradients[name] += layer.grad

    avg_gradients = {name: grad / len(dataloader) for name, grad in gradients.items()}

    return avg_gradients


def layerwise_score(model, dataloader, device, mask):
    avg_grad = score_grad_local(model=model, dataloader=dataloader, device=device,mask=mask)
    importance_dict = {}
    # Loop through named parameters
    for name, param in model.named_parameters():
        mask_name = name + '_mask'
        if mask_name in mask.keys():
            layer_weight_grad = avg_grad[name]
            importance_dict[name] = layer_weight_grad ** 2
    return importance_dict