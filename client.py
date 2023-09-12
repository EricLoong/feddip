# import os

# os.chdir('/Users/ericlong/PycharmProjects/reuse_in_cnn/cifar10_FL')
import copy
import math
from torchmetrics import Accuracy
import warnings
from torch.utils.data.dataloader import DataLoader, Dataset
import torch.optim as optim
from functions_new import *


class separated_data(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idx_list):
        self.dataset = dataset
        self.idxs = [int(i) for i in idx_list]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        warnings.filterwarnings('ignore')
        result_img = torch.tensor(image)
        result_label = torch.tensor(label)
        return result_img, result_label



class Client(object):
    '''
    The functions that client/user uses
    '''
    def __init__(self, args, dataset, index_list, model, client_idx):
        self.dataset = dataset
        self.index_list = index_list
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        # Default criterion set to CrossEntropy loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.client_idx = client_idx
        self.trainloader, self.testloader = self.train_test_data()


    def customize_loss(self, output, true_label, current_model):
        loss_part1 = self.criterion(output, true_label)
        loss_part2 = norm_whole_model(current_model).to(self.device)
        loss = loss_part1 + self.args.lambda_shrink * loss_part2
        return loss

    def prune_4_mask(self, model, sparsity):
        prune_para = generate_prune_param(model=model,bias=False,fc=True)
        prune.global_unstructured(prune_para, pruning_method=prune.L1Unstructured, amount=sparsity, )
        mask_dict = dict()
        for key in model.state_dict().keys():
            if key.endswith('_mask'):
                mask_dict[key] = model.state_dict()[key]
        # Clean the pruned model
        for module, name in prune_para:
            prune.remove(module, name)
        return mask_dict

    def prune_4_mask_resnet(self, model, sparsity):
        prune_para = basic_generate_resnet(model_resnet=model, bias=False)
        #print('Sparsity before pruning: ')
        #print_sparsity(parameters_to_prune=prune_para)
        prune.global_unstructured(prune_para, pruning_method=prune.L1Unstructured, amount=sparsity, )
        #print('Sparsity after pruning: ')
        #print_sparsity(parameters_to_prune=prune_para)
        mask_dict = dict()
        for key in model.state_dict().keys():
            if key.endswith('_mask'):
                mask_dict[key] = model.state_dict()[key]
        # Clean the pruned model
        for module, name in prune_para:
            prune.remove(module, name)
        return mask_dict

    def download_global_model(self,model_stat_dict):
        self.model.load_state_dict(model_stat_dict)

    def upload_local_model(self):
        return self.model.state_dict()

    def train_test_data(self):
        '''

        :return: training, validation and test datasets according to the received list
        '''
        train_dataset, test_dataset = self.dataset
        train_index, test_index = self.index_list

        trainloader = DataLoader(separated_data(train_dataset, train_index),
                                 batch_size=self.args.local_batchsize, shuffle=True,num_workers=self.args.num_workers)
        testloader = DataLoader(separated_data(test_dataset, test_index),
                                batch_size=self.args.local_batchsize, shuffle=False, num_workers=self.args.num_workers)
        #print(f'local data size:{len(train_index)}')
        return trainloader, testloader

    def layerwise_prune(self, masks, weights, rounds):
        '''
        :param sparsity: the sparsity of the model by layer.
        :return: a mask dictionary
        '''
        #print('Pruning the model layer by layer')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        drop_ratio = self.args.anneal_factor / 2 * (1 + np.cos((rounds * np.pi) / self.args.epochs))
        new_masks = copy.deepcopy(masks)

        num_remove = {}
        for name in masks.keys():
            if name.endswith('mask'): # To double check only mask get pruned
                num_non_zeros = torch.sum(masks[name].to(device))
                num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
                weight_name = name[:-5]
                temp_weights = torch.where(masks[name].to(device) > 0, torch.abs(weights[weight_name].to(device)),
                                           1000000 * torch.ones_like(weights[weight_name].to(device)))
                x, idx = torch.sort(temp_weights.view(-1))
                new_masks[name].view(-1)[idx[:num_remove[name]]] = 0  # Prune the smallest weights
        for key in new_masks:
            new_masks[key] = new_masks[key].to(device)
        return new_masks, num_remove

    def layerwise_regrow(self, masks, num_remove, score_by_layers=None):
        new_masks = copy.deepcopy(masks)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for name in masks.keys():
            if name.endswith('mask'):
                weight_name = name[:-5]
                negative_tensor = -1000000 * torch.ones_like(score_by_layers[weight_name])

                temp = torch.where(masks[name].to(device) == 0,
                                       torch.abs(score_by_layers[weight_name].to(device)),
                                       negative_tensor.to(device))
                sort_temp, idx = torch.sort(temp.view(-1).to(device), descending=True)
                new_masks[name].view(-1)[idx[:num_remove[name]]] = 1

        for key in new_masks:
            new_masks[key] = new_masks[key].to(device)
        return new_masks

    def compare_models(self, model1, model2):
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            if name1 == name2:
                diff = torch.norm(param1 - param2)
                print(f"Parameter: {name1}, Difference: {diff:.4f}")
            else:
                print("Parameters do not match")
                break
    def lambda_shrink_cur(self,global_iter,num_segments=20):
        lambda_shrink = self.args.lambda_shrink
        initial_lambda = 5e-04
        epochs = self.args.epochs
        step_size = (lambda_shrink - initial_lambda) / (num_segments - 1)
        segment_length = epochs // num_segments
        step_function = []
        for i in range(num_segments):
            if i == 0:
                current_lambda = 0
            else:
                current_lambda = initial_lambda + step_size * (i - 1)
            step_function.extend([current_lambda] * segment_length)
            if i == num_segments - 1:
                # Adjust the last segment to include the remaining epochs
                step_function.extend([current_lambda] * (epochs % num_segments))
        return step_function[global_iter]

    def init_masks(self, params, sparsities):
        masks = {}
        for name in params:
            if 'weight' in name:
                mask_name = name + '_mask'
                masks[mask_name] = torch.zeros_like(params[name])
                dense_numel = int((1 - sparsities[name]) * torch.numel(masks[mask_name]))
                if dense_numel > 0:
                    temp = masks[mask_name].view(-1)
                    perm = torch.randperm(len(temp))
                    perm = perm[:dense_numel]
                    temp[perm] = 1
        return masks

    def proximal_term_compute(self, global_model, local_model):
        proximal_term = 0.0
        for w, w_t in zip(local_model.parameters(), global_model.parameters()):
            proximal_term += (w - w_t).norm(2)
        return proximal_term

    def calculate_sparsities(self, params, tabu=[], distribution="ERK", sparse=0.5):
        spasities = {}
        if distribution == "uniform":
            print('initialize by Uniform')
            for name in params:
                if name not in tabu:
                    spasities[name] = 1 - self.args.amount_sparsity
                else:
                    spasities[name] = 0
        elif distribution == "ERK":
            print('initialize by ERK')
            total_params = 0
            for name in params:
                if 'weight' in name:
                    total_params += params[name].numel()
            is_epsilon_valid = False
            # # The following loop will terminate worst case when all masks are in the
            # custom_sparsity_map. This should probably never happen though, since once
            # we have a single variable or more with the same constant, we have a valid
            # epsilon. Note that for each iteration we add at least one variable to the
            # custom_sparsity_map and therefore this while loop should terminate.
            dense_layers = set()

            density = sparse
            while not is_epsilon_valid:
                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name in params:
                    if 'weight' in name:
                        if name in tabu:
                            dense_layers.add(name)
                        n_param = np.prod(params[name].shape)
                        n_zeros = n_param * (1 - density)
                        n_ones = n_param * density

                        if name in dense_layers:
                            rhs -= n_zeros
                        else:
                            rhs += n_ones
                            raw_probabilities[name] = (
                                                              np.sum(params[name].shape) / np.prod(params[name].shape)
                                                      ) ** self.args.erk_power_scale
                            divisor += raw_probabilities[name] * n_param
                epsilon = rhs / divisor
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            (f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name in params:
                if 'weight' in name:
                    if name in dense_layers:
                        spasities[name] = 0
                    else:
                        spasities[name] = (1 - epsilon * raw_probabilities[name])

        return spasities

    def prunable_layer_norm(self, model, pruning_mask, temp_lambda):
        # Access pruning_mask to know which layers should be counted in the norm.
        norm_sum = 0
        for name, w in model.named_parameters():
            # Check if the layer has a corresponding mask in pruning_mask
            if name + '_mask' in pruning_mask:
                norm_sum += temp_lambda * torch.norm(w, p=self.args.pnorm)
        return norm_sum

    def train_model(self, global_iter, learning_rate, mask_applied=None,pruning_mask = None):
        apply_mask = self.args.prunefl
        apply_pruning = self.args.prune
        apply_prox = self.args.prox
        sparse = self.args.sparse
        model = self.model
        if apply_prox:
            global_model = copy.copy(self.model)
        else:
            global_model = None
        model.train()
        epochs = self.args.local_ep
        criterion = self.criterion
        device = self.device
        #if len(model.state_dict()) == 22:
        #    optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9)
        #else:
        #    optimizer = optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=5e-04)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,  weight_decay=5e-04)
        loss_result = list()

        sparsity_t = self.args.amount_sparsity+(self.args.init_sparsity- self.args.amount_sparsity)*((1-(global_iter/self.args.epochs))**3)
        # Train the network
        for epoch in range(epochs):
            running_loss = 0.0

            if apply_pruning:
                # This step is just to obtain the mask from the global model since the global model is pruned after 1 global iteration.
                if epoch == 0 and type(pruning_mask)==type(None):
                    if global_iter == 0:
                        if 'resnet' in self.args.model:
                            #dense_layers= []
                            dense_layers = [key for key in model.state_dict().keys() if ('bn' in key and 'weight' in key) or ('downsample' in key and 'weight' in key)]
                        else:
                            dense_layers = []
                        weight_sparsities = self.calculate_sparsities(params=model.state_dict(),tabu=dense_layers,sparse=1-self.args.init_sparsity)
                        pruning_mask = self.init_masks(params=model.state_dict(),sparsities=weight_sparsities)
                    else:
                        if 'resnet' in self.args.model:
                            pruning_mask = self.prune_4_mask_resnet(model, sparsity=sparsity_t)
                        else:
                            pruning_mask = self.prune_4_mask(model, sparsity=sparsity_t)


                #self.compare_models(model1=model,model2=model_pruned)

            for i, data in enumerate(self.trainloader, 0):
                images, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                if apply_pruning:
                    # From here start to compute gradients of pruned model
                    model_pruned = copy.deepcopy(model)
                    for name, param in model_pruned.named_parameters():
                        if name + '_mask' in pruning_mask:
                            param.data *= pruning_mask[name + '_mask'].to(device)
                    outputs_pruned = model_pruned(images)
                    loss_pruned = criterion(outputs_pruned, labels)
                    if sparse:
                        # Adopt increasing regularization power
                        # Because we have 5e-04 in optimizor, we add back the difference
                        temp_lambda = self.lambda_shrink_cur(global_iter=global_iter)
                        norm_sum = self.prunable_layer_norm(model=model_pruned,temp_lambda=temp_lambda,pruning_mask=pruning_mask)
                        loss_pruned = loss_pruned + norm_sum
                    if apply_prox:
                        proximal_term = self.proximal_term_compute(global_model=global_model,local_model=model_pruned)
                        loss_pruned = loss_pruned+(self.args.mu/2)*proximal_term
                    loss_pruned.backward()


                    # Copy gradients from the pruned model to the original model
                    with torch.no_grad():
                        for original_param, pruned_param in zip(model.named_parameters(),
                                                                model_pruned.named_parameters()):
                            original_name, original_param_data = original_param
                            pruned_name, pruned_param_data = pruned_param

                            if pruned_param_data.grad is not None:
                                original_param_data.grad = pruned_param_data.grad.clone()
                                pruned_param_data.grad.zero_()
                    # Here ends the original model has the pruned mode gradients.

                    outputs = model(images)
                    loss_ce = criterion(outputs, labels)
                    if sparse:
                        temp_lambda = self.lambda_shrink_cur(global_iter=global_iter)
                        norm_sum = self.prunable_layer_norm(model=model_pruned,temp_lambda=temp_lambda,pruning_mask=pruning_mask)
                        loss_ce += norm_sum
                    if apply_prox:
                        proximal_term = self.proximal_term_compute(global_model=global_model,local_model=model_pruned)
                        loss_ce += (self.args.mu/2)*proximal_term
                    loss = loss_ce

                else:
                    outputs = model(images)
                    loss_ce = criterion(outputs, labels)

                    if sparse:
                        #print('Regularization on use')
                        norm_sum = 0
                        for w in zip(model.parameters()):
                            norm_sum += self.args.lambda_shrink * torch.norm(w[0], p=self.args.pnorm)
                        loss_ce += norm_sum
                    if apply_prox:
                        proximal_term = self.proximal_term_compute(global_model=global_model,local_model=model)
                        loss_ce += (self.args.mu/2)*proximal_term

                    loss = loss_ce

                    loss.backward()
                optimizer.step()  # Update the original model using the optimizer

                running_loss += loss.item()
                if apply_mask and type(mask_applied)!=type(None):
                    for name, param in model.named_parameters():
                        mask_name_temp = name + '_mask'
                        if mask_name_temp in mask_applied.keys():
                            param.data *= mask_applied[name + '_mask'].to(device)

            loss_result.append(running_loss / len(self.trainloader))
            print(f'Running loss is {running_loss / len(self.trainloader)}')


        with torch.no_grad():
            loss, total, correct = 0.0, 0.0, 0.0
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Inference
                outputs = model(images)
                if sparse:
                    batch_loss = self.customize_loss(output=outputs, true_label=labels, current_model=model)
                else:
                    batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

            accuracy = correct / total  # Validation Accuracy
            print(f'The local test accuracy of global iteration(s) {global_iter} is {accuracy}')
        return accuracy, loss_result, pruning_mask

    def train_model_snip(self, global_iter,learning_rate, mask_applied):
        apply_mask_snip = self.args.snip
        model = self.model
        model.train()
        epochs = self.args.local_ep
        criterion = self.criterion
        device = self.device

        #if len(model.state_dict()) == 22:
        #    optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9)
        #else:
        #    optimizer = optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=5e-04)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=5e-04)
        loss_result = list()
        #    f"Global Iteration: {global_iter}, Local Training {epochs} epoch(s) and  {len(trainloader)} batches each epoch")
        # Train the network
        for epoch in range(epochs):
            running_loss = 0.0

            for i, data in enumerate(self.trainloader, 0):
                images, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()  # Update the original model using the optimizer

                running_loss += loss.item()

                if apply_mask_snip:
                    for name, param in model.named_parameters():
                        mask_name_temp = name + '_mask'
                        if mask_name_temp in mask_applied.keys():
                            param.data *= mask_applied[name + '_mask'].to(device)

            loss_result.append(running_loss / len(self.trainloader))
            print(f'Running loss is {running_loss / len(self.trainloader)}')



        with torch.no_grad():
            loss, total, correct = 0.0, 0.0, 0.0
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Inference
                outputs = model(images)

                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

            accuracy = correct / total  # Validation Accuracy
            print(f'The test accuracy of global iteration(s) {global_iter} is {accuracy}')


        return accuracy, loss_result

    def train_model_dst(self, global_iter,learning_rate, mask_applied):
        apply_mask_dst = self.args.feddst
        model = self.model
        model.train()
        epochs = self.args.local_ep
        criterion = self.criterion
        device = self.device

        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=5e-04)
        loss_result = list()

        # Train the network
        for epoch in range(epochs):
            running_loss = 0.0

            for i, data in enumerate(self.trainloader, 0):
                images, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()  # Update the original model using the optimizer

                running_loss += loss.item()

                if apply_mask_dst:
                    for name, param in model.named_parameters():
                        mask_name_temp = name + '_mask'
                        if mask_name_temp in mask_applied.keys():
                            param.data *= mask_applied[name + '_mask'].to(device)

            loss_result.append(running_loss / len(self.trainloader))
            print(f'Running loss is {running_loss / len(self.trainloader)}')



        with torch.no_grad():
            loss, total, correct = 0.0, 0.0, 0.0
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Inference
                outputs = model(images)

                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

            accuracy = correct / total  # Validation Accuracy
            print(f'The test accuracy of global iteration(s) {global_iter} is {accuracy}')


        return accuracy, loss_result

    def train_model_prunetrain(self, global_iter,learning_rate,power=0.01):
        model = self.model
        model.train()
        epochs = self.args.local_ep
        criterion = self.criterion
        device = self.device
        sparsity_t = self.args.amount_sparsity + (self.args.init_sparsity - self.args.amount_sparsity) * (
                    (1 - (global_iter / self.args.epochs)) ** 3)
        if 'resnet' in self.args.model:
            pruning_mask = self.prune_4_mask_resnet(model, sparsity=sparsity_t)
        else:
            pruning_mask = self.prune_4_mask(model, sparsity=sparsity_t)

        key_0 = list(pruning_mask.keys())[0]
        key_max = list(pruning_mask.keys())[-1]
        pruning_mask.pop(key_0)
        pruning_mask.pop(key_max)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=5e-04)
        loss_result = list()

        # Train the network
        for epoch in range(epochs):
            running_loss = 0.0

            for i, data in enumerate(self.trainloader, 0):
                images, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss_ce = criterion(outputs, labels)
                loss_norm = self.prunable_layer_norm(model,pruning_mask=pruning_mask,temp_lambda=power)
                loss = loss_ce+loss_norm
                loss.backward()
                optimizer.step()  # Update the original model using the optimizer

                running_loss += loss.item()

            loss_result.append(running_loss / len(self.trainloader))
            print(f'Running loss is {running_loss / len(self.trainloader)}')



        with torch.no_grad():
            loss, total, correct = 0.0, 0.0, 0.0
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Inference
                outputs = model(images)

                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

            accuracy = correct / total  # Validation Accuracy
            print(f'The test accuracy of global iteration(s) {global_iter} is {accuracy}')


        return accuracy, loss_result

    def inference(self, total_test):
        """ Returns the inference accuracy and loss.
        """
        testloader = total_test
        self.model.eval()
        loss, total, correct, correct_top5 = 0.0, 0.0, 0.0, 0.0
        top_1_acc = Accuracy().to(self.device)
        top_5_acc = Accuracy(top_k=5).to(self.device)
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = self.model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += (top_1_acc(pred_labels, labels) * torch.tensor(len(labels))).item()
            correct_top5 += (top_5_acc(outputs, labels) * torch.tensor(len(labels))).item()
            total += len(labels)

        accuracy = correct / total
        accuracy_top5 = correct_top5 / total

        if 'resnet' not in self.args.model:
            #print('Only top1 accuracy')
            return accuracy, loss
        else:
            #print('Consider top 1 and 5 accuracy')
            return accuracy, loss, accuracy_top5

