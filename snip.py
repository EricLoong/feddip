from client import Client
from fl_functions import *
from exp_args import *
import pandas as pd
from datasets_models import *
from functions_new import *
# import os
# os.environ["NCCL_DEBUG"] = "INFO"
import tqdm

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.enabled = False

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())

def fed_avg_snip():
    """
    :return: (sparse)(pruned) model
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device use is {device}')
    print(f'Learning rate is {args.lr}')
    # Preparation for data and clients

    data_name = args.dataset
    model_name = args.model
    data = load_data(data_name=data_name)
    index_list, data_train_test, pctg_4_avg = preprocessed_data(data_list=data_list, batch_size=args.local_batchsize,
                                                    n_users=args.num_users,
                                                    data_name=data_name, num_work=args.num_workers,partition_method=args.partition_method)
    if type(args.partition_method)!=type(None):
        print('Adopt Non-IID partition, and the clients weight for average is:', pctg_4_avg)
    else:
        print('IID partition with equal weights')
    user_index, idx_users = index_list
    train_dl, test_dl = data_train_test

    global_model = model_selected(model_list=model_list, model_name=model_name,
                                  data_name=data_name, pre_trained=args.prt)
    # Initilize Clients
    clients = []
    for c in range(args.num_users):
        cl = Client(args=args, dataset=data, index_list=idx_users[c][1], model=copy.deepcopy(global_model),
                    client_idx=c)
        clients.append(cl)
        torch.cuda.empty_cache()

    print('Finish client initilization')
    if args.parallel:
        global_model = nn.DataParallel(global_model)

    print('The number of GPU used', torch.cuda.device_count())
    global_model.to(device)
    global_model.train()
    train_loss, train_accuracy = [], []
    glo_loss, glo_acc = [], []

    top5_acc_train = []
    top5_acc_test = []
    inf_loss_record = []
    sparsity_record = []
    delta_loss_record = []
    initial_lr = copy.deepcopy(args.lr)
    comm_costs_accum = [0]
    mask_snip = SNIP_modified(net=global_model,keep_ratio=1-args.amount_sparsity,train_dataloader=clients[0].trainloader,device=device)
    # Start training
    for epoch in tqdm.tqdm(range(args.epochs)):
        local_waps, local_test_accuracy = [], []
        train_epoch_loss = list()
        global_model.train()
        user_index_ts = np.random.choice(user_index, int(args.frac * args.num_users), replace=False)
        k = 0
        local_delta_epochs = []
        args.lr = initial_lr * (0.998**epoch)

        # print('Current target sparsity is', sparsity_t)
        for index in tqdm.tqdm(user_index_ts):
            cur_client = clients[index]
            temp_model = copy.deepcopy(global_model)
            cur_client.download_global_model(model_stat_dict=temp_model.state_dict())
            # This ensure the learning rate is decreasing to involve the new learning rate every round.
            local_val_temp, local_loss_temp = cur_client.train_model_snip(global_iter=epoch, learning_rate=args.lr,mask_applied=mask_snip)
            local_test_accuracy.append([epoch, index, local_val_temp])
            train_epoch_loss.append([epoch, index, local_loss_temp])

            temp_server_wap = cur_client.upload_local_model()
            local_waps.append(temp_server_wap)
            k += 1
            print(f'Finish Local Training for {k} users. And current user is {index}')

        server = ServerCollect(args=args, device=device)
        global_receive = server.average_weights(weight=local_waps,pctg=pctg_4_avg,selected_clients=user_index_ts)
        global_model.load_state_dict(global_receive)
        sparsity_check = compute_sparsity(global_model)

        upload_costs_temp = args.num_users * args.frac * comm_costs_in_mb(model=global_model,
                                                                              sparsity=sparsity_check/100)
        download_costs_temp = args.num_users * args.frac * comm_costs_in_mb(model=global_model, sparsity=sparsity_check/100)


        temp_comm_cost = comm_costs_accum[-1] + upload_costs_temp + download_costs_temp
        comm_costs_accum.append(temp_comm_cost)

        if 'resnet' not in args.model:
            temp_train_acc, temp_train_loss = server.inference(model=copy.deepcopy(global_model), total_test=train_dl,
                                                               sparse=args.sparse)
            temp_glo_acc, temp_glo_loss = server.inference(model=copy.deepcopy(global_model), total_test=test_dl,
                                                           sparse=args.sparse)
            glo_loss.append(temp_glo_loss)
            glo_acc.append(temp_glo_acc)
            train_accuracy.append(temp_train_acc)
            train_loss.append(temp_train_loss)
            print('Accuracy Record is:', glo_acc)
        else:
            temp_train_acc, temp_train_loss, temp_top5_acc_train = server.inference(model=copy.deepcopy(global_model),
                                                                                    total_test=train_dl,
                                                                                    sparse=args.sparse)
            temp_glo_acc, temp_glo_loss, temp_top5_acc_test = server.inference(model=copy.deepcopy(global_model),
                                                                               total_test=test_dl, sparse=args.sparse)
            glo_loss.append(temp_glo_loss)
            glo_acc.append(temp_glo_acc)
            print('Accuracy Record is:', glo_acc)
            train_accuracy.append(temp_train_acc)
            train_loss.append(temp_train_loss)
            top5_acc_train.append(temp_top5_acc_train)
            top5_acc_test.append(temp_top5_acc_test)

        # print("--- %s seconds ---" % (time.time() - start_time))

    # Store model, loss and validation accuracy
    df_loss = pd.DataFrame(data=train_loss)
    df_train_acc = pd.DataFrame(data=train_accuracy)
    df_glob_loss = pd.DataFrame(data=glo_loss)
    df_glob_acc = pd.DataFrame(data=glo_acc)
    df_top5_train = pd.DataFrame(data=top5_acc_train)
    df_top5_test = pd.DataFrame(data=top5_acc_test)
    df_sparsity_record = pd.DataFrame(data=sparsity_record)
    df_infloss_record = pd.DataFrame(data=inf_loss_record)
    df_deltaloss_record = pd.DataFrame(data=delta_loss_record)
    df_comm_costs = pd.DataFrame(data=comm_costs_accum)
    print('The record of additional mask information loss is', df_infloss_record)
    print('The global model accuracy is', glo_acc)
    print('Communication Costs Accumulated:', comm_costs_accum)


    name_tail = f"Global_epochs_{args.epochs}_Local_epochs_{args.local_ep}_model_name_{model_name}_SNIP_sparsity_{args.amount_sparsity}_numofclients_{args.num_users}_fraction_{args.frac}"

    if args.tfstp:
        name_tail = name_tail + '_ctrain_' + '.csv'
    else:
        name_tail = name_tail + '.csv'
    # PS means prune and shrink
    file_name_loss = r'loss_' + name_tail
    file_name_train_acc = r'trainacc_' + name_tail
    file_name_global_loss = r'gloss_' + name_tail
    file_name_global_acc = r'gacc_' + name_tail
    file_name_global_top5train = r'top5train_' + name_tail
    file_name_global_top5test = r'top5test_' + name_tail
    file_name_sparsity_record = r'sparsity_record_' + name_tail
    file_name_infloss_record = r'infloss_record_' + name_tail
    file_name_deltaloss_record = r'delta_record_' + name_tail
    file_name_comm_costs = r'comm_cost_' + name_tail

    # first job is marked by E50U5 resnet
    df_loss.to_csv(file_name_loss, index=False)
    df_train_acc.to_csv(file_name_train_acc, index=False)
    df_glob_acc.to_csv(file_name_global_acc, index=False)
    df_glob_loss.to_csv(file_name_global_loss, index=False)
    df_sparsity_record.to_csv(file_name_sparsity_record, index=False)
    df_infloss_record.to_csv(file_name_infloss_record, index=False)
    df_deltaloss_record.to_csv(file_name_deltaloss_record, index=False)
    df_comm_costs.to_csv(file_name_comm_costs, index=False)


    if args.model == 'resnet50':
        df_top5_train.to_csv(file_name_global_top5train, index=False)
        df_top5_test.to_csv(file_name_global_top5test, index=False)

    # Global model test accuracy on full dataset
    global_model.eval()
    num_correct = 0
    num_samples = 0
    for batch_idx, (data, targets) in enumerate(test_dl):
        data = data.to(device=device)
        targets = targets.to(device=device)
        #  Forward Pass

        scores = global_model(data)
        _, predictions = scores.max(1)
        num_correct += (predictions == targets).sum()
        num_samples += predictions.size(0)
    print(
        f" Final Global Model, got {num_correct} / {num_samples} "
        f"with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
    )
    print(file_name_global_acc)

    save_path = './' + name_tail[:-4] + '.pth'

    return [global_model, save_path]


if __name__ == "__main__":
    model, save_path = fed_avg_snip()
    # torch.save(model.state_dict(), save_path)
    print("Model training is Done. Good job!")
