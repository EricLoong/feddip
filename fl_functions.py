# import os
# os.chdir('/Users/ericlong/PycharmProjects/reuse_in_cnn/cifar10_FL')

from functions_new import *
import torch.nn as nn
from torchmetrics import Accuracy

class ServerCollect(object):
    """
    The functions that central server use.
    """

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def customize_loss(self, output, true_label, current_model):
        loss_part1 = self.criterion(output, true_label)
        loss_part2 = norm_whole_model(current_model).to(self.device)
        loss = loss_part1 + self.args.lambda_shrink * loss_part2
        return loss

    def average_weights(self, weight, selected_clients, pctg):
        # Initialize a dictionary to store the average weights
        avg_weights = {key: torch.zeros_like(value, dtype=torch.float) for key, value in weight[0].items()}

        # Calculate the sum of percentages for the selected clients
        total_selected_pctg = sum(pctg[i] for i in selected_clients)

        # Calculate the weighted sum of the model weights based on the percentage of local sample size
        for client_idx, client_weight in zip(selected_clients, weight):
            client_pctg = pctg[client_idx] / total_selected_pctg
            for key, value in client_weight.items():
                avg_weights[key] += client_pctg * value.to(torch.float)  # cast to float before addition

        return avg_weights

    def layerwise_pruning_server(self, params, sparsities):
        masks = {}
        for name, param in params.items():
            if 'weight' in name:
                mask_name = name + '_mask'
                # Flattening the tensor
                flat_weights = param.view(-1)

                # Sorting the weights by magnitude (absolute value), excluding zeros
                non_zero_weights = flat_weights[flat_weights != 0]
                _, sorted_indices = torch.sort(torch.abs(non_zero_weights))

                # Determine the number of non-zero elements to retain based on required sparsity
                num_to_retain = int((1 - sparsities[name]) * flat_weights.numel())

                # The indices of the weights to be retained
                retain_indices = sorted_indices[-num_to_retain:]

                # Create a new mask initialized to zeros
                mask = torch.zeros_like(flat_weights)

                # Set the positions of the retained weights to 1 in the mask
                non_zero_positions = torch.nonzero(flat_weights).squeeze()
                mask[non_zero_positions[retain_indices]] = 1.0

                masks[mask_name] = mask.view(param.shape)
                actual_sparsity = 1.0 - torch.sum(mask) / mask.numel()
                print(f"{name}: Target Sparsity: {sparsities[name]}, Actual Sparsity: {actual_sparsity}")

        return masks

    def set_weight_by_mask(self, model, mask):
        '''
        :param model: Original model
        :param mask: Mask to be applied
        :return: A model has been applied with mask
        '''
        with torch.no_grad():
            model_dict = model.state_dict()
            new_dict = dict.fromkeys(model_dict.keys())
            for key in list(model_dict.keys()):
                if 'weight' in key:
                    new_dict[key] = model_dict[key]*mask[key+'_mask']
                else:
                    new_dict[key] = model_dict[key]
            model.load_state_dict(new_dict)

    def inference(self, model, total_test, sparse=1):
        """ Returns the inference accuracy and loss.
        """
        testloader = total_test
        model.to(self.device)
        model.eval()
        loss, total, correct, correct_top5 = 0.0, 0.0, 0.0, 0.0
        top_1_acc = Accuracy().to(self.device)
        top_5_acc = Accuracy(top_k=5).to(self.device)
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            if sparse == 1:
                batch_loss = self.customize_loss(output=outputs, true_label=labels, current_model=model)
            else:
                batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1).to(self.device)
            correct += (top_1_acc(pred_labels, labels) * torch.tensor(len(labels))).item()
            correct_top5 += (top_5_acc(outputs, labels) * torch.tensor(len(labels))).item()
            total += len(labels)

        accuracy = correct / total
        accuracy_top5 = correct_top5 / total

        if self.args.model not in ['vgg16', 'resnet18','resnet50']:
            print('Only top1 accuracy')
            return accuracy, loss/len(testloader)
        else:
            print('Consider top 1 and 5 accuracy')
            return accuracy, loss/len(testloader), accuracy_top5
