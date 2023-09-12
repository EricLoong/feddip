import copy
import time
from client import *
from fl_functions import *
from exp_args import *
import pandas as pd
from datasets_models import *
from functions_new import *
# import os
# os.environ["NCCL_DEBUG"] = "INFO"
import tqdm

torch.cuda.empty_cache()


# Local iteration equals to 5
# The idea of PruneFL: Firstly initialize the mask on one client until converge. For each global round, local client update
# several iterations and apply the masks.




def feddst_server():
    """
    :return: (sparse)(pruned) model accuaracy and sparsity record
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_name = args.dataset
    model_name = args.model
    data = load_data(data_name=data_name)
    index_list, data_train_test, pctg_4_avg = preprocessed_data(data_list=data_list, batch_size=args.local_batchsize,
                                                    n_users=args.num_users,
                                                    data_name=data_name, num_work=args.num_workers,partition_method=args.partition_method)
    if args.partition_method!= 'iid':
        print('Adopt Non-IID partition, and the clients weight for average is:', pctg_4_avg)
    else:
        print('IID partition with equal weights')
    user_index, idx_users = index_list
    train_dl, test_dl = data_train_test

    global_model = model_selected(model_list=model_list, model_name=model_name,
                                  data_name=data_name, pre_trained=args.prt)
    if args.parallel:
        global_model = nn.DataParallel(global_model)
    global_model.to(device)

    # No inital client and server, this is just for coding since we need to use the client class to generate the mask
    # and apply it to the global model for initialization.

    cl_init = Client(args=args, dataset=data, index_list=idx_users[0][1], model=copy.deepcopy(global_model),
                client_idx=0)
    if 'resnet' in args.model:
        # dense_layers= []
        dense_layers = [key for key in cl_init.model.state_dict().keys() if
                        ('bn' in key and 'weight' in key) or ('downsample' in key and 'weight' in key)]
    else:
        dense_layers = []
    weight_sparsities = cl_init.calculate_sparsities(params=cl_init.model.state_dict(), tabu=dense_layers,
                                                sparse=1 - args.amount_sparsity)
    glo_pruning_mask = cl_init.init_masks(params=cl_init.model.state_dict(), sparsities=weight_sparsities)
    #print(glo_pruning_mask.keys())

    server_init = ServerCollect(args=args, device=device)

    # Generated the initial mask and apply to the global model. Then create the clients.
    clients = []
    for c in range(args.num_users):
        c_client = Client(args=args, dataset=data, index_list=idx_users[c][1], model=copy.deepcopy(global_model),
                    client_idx=c)
        c_client.model.to(device)
        server_init.set_weight_by_mask(model=c_client.model, mask=glo_pruning_mask)
        clients.append(c_client)
        torch.cuda.empty_cache()

    print('Finish client initilization')

    #mask_one_initial = init_mask(model=global_model, args=args)
    init_model = copy.deepcopy(global_model)
    set_weight_by_mask(model=init_model, mask=glo_pruning_mask)
    init_model.to(device)
    init_model.train()
    args_init = copy.deepcopy(args)
    # This is for initial mask
    args_init.num_users = 100

    comm_costs_accum = [0]

    # Send the initial models and mask to all users
    comm_costs_accum.append(args.num_users*(comm_costs_in_mb(model=init_model,sparsity=args.amount_sparsity)))
    global_model.load_state_dict(init_model.state_dict())
    print('Finish initial pruning!')

    train_loss, train_accuracy = [], []
    top5_acc_train, top5_acc_test = [], []
    glo_loss, glo_acc = [], []
    glo_sp = []
    # Start training
    global_model.train()
    global_model.to(device)
    mask_reconfig_list = []
    for c in range(args.num_users):
        mask_reconfig_list.append(glo_pruning_mask)

    initial_lr = copy.deepcopy(args.lr)
    for epoch in tqdm.tqdm(range(args.epochs)):
        args.lr = initial_lr * (0.998**epoch)
        reconfig = (epoch+1) % args.reconfig_interval == 0
        local_waps = []
        glo_sp_temp = compute_sparsity(model=global_model)
        glo_sp.append(glo_sp_temp)
        user_index_ts = np.random.choice(user_index, int(args.frac * args.num_users), replace=False)

        for index in tqdm.tqdm(user_index_ts):
            cur_client = clients[index]
            temp_model = copy.deepcopy(global_model)
            cur_client.download_global_model(model_stat_dict=temp_model.state_dict())
            cur_client.train_model_dst(global_iter=epoch, learning_rate=args.lr, mask_applied=mask_reconfig_list[index])
            if reconfig:# Do reconfiguration, layerwise drop and regrow.
                mask_reconfig_temp = copy.deepcopy(mask_reconfig_list[index])
                print(f'Reconfiguration for client {index} at epoch {epoch}')
                mask_reconfig_temp, num_prune_temp = cur_client.layerwise_prune(masks=mask_reconfig_temp, weights=temp_model.state_dict(),rounds=epoch)
                score_layerwise = layerwise_score(model=cur_client.model, mask=mask_reconfig_temp,device=device,dataloader=cur_client.trainloader)
                mask_reconfig_temp = cur_client.layerwise_regrow(masks=mask_reconfig_temp, num_remove=num_prune_temp, score_by_layers=score_layerwise)
                mask_reconfig_list[index] = copy.deepcopy(mask_reconfig_temp)
            local_sparsity = compute_sparsity(model=cur_client.model)
            print('The local sparsity is:', local_sparsity)
            temp_server_wap = cur_client.upload_local_model()
            local_waps.append(temp_server_wap)


            # This appends the squared gradients

        server = ServerCollect(args=args, device=device)
        global_receive = server.average_weights(weight=local_waps,pctg=pctg_4_avg,selected_clients=user_index_ts)
        mask_global = server.layerwise_pruning_server(params=global_receive, sparsities=weight_sparsities)
        for name, mask in mask_global.items():
            unique_values = torch.unique(mask)
            print(f"Unique values in {name}: {unique_values}")

        #print(weight_sparsities.keys())
        #print(mask_global.keys())
        #print(global_receive.keys())
        global_model.load_state_dict(global_receive)
        # Apply mask
        server.set_weight_by_mask(model=global_model, mask=mask_global)
        # After applying mask

        if args.model != 'resnet18':
            temp_train_acc, temp_train_loss = server.inference(model=copy.deepcopy(global_model), total_test=train_dl,
                                                               sparse=args.sparse)
            temp_glo_acc, temp_glo_loss = server.inference(model=copy.deepcopy(global_model), total_test=test_dl,
                                                           sparse=args.sparse)
            # print(f'Current loss is{temp_glo_loss}')
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
            # print(f'Current loss is{temp_glo_loss}')
            glo_loss.append(temp_glo_loss)
            glo_acc.append(temp_glo_acc)
            print('Accuracy Record is:', glo_acc)
            train_accuracy.append(temp_train_acc)
            train_loss.append(temp_train_loss)
            top5_acc_train.append(temp_top5_acc_train)
            top5_acc_test.append(temp_top5_acc_test)
        #global_mask = generate_mask(global_model, args=args)
        #server.set_weight_by_mask(model=global_model,mask=global_mask)
        upload_costs_temp = args.num_users*args.frac*comm_costs_in_mb(model=global_model,sparsity=args.amount_sparsity)
        download_costs_temp = args.num_users*args.frac*comm_costs_in_mb(model=global_model,sparsity=args.amount_sparsity)


        temp_comm_cost = comm_costs_accum[-1] + upload_costs_temp + download_costs_temp
        comm_costs_accum.append(temp_comm_cost)
        print(f'Current loss is{temp_glo_loss}')
        #glo_loss.append(temp_glo_loss)
        #glo_acc.append(temp_glo_acc)
        train_accuracy.append(temp_train_acc)
        train_loss.append(temp_train_loss)
        print('Accuracy Record is:', glo_acc)
        print('Global Sparsity is:', glo_sp)
        print('Communication Costs Accumulated:', comm_costs_accum)
    if args.feddst:
        name_tail = "Global_epochs_" + str(args.epochs) + '_Local_epochs_' + str(
            args.local_ep) + '_model_name_' + model_name + 'FedDST_' + 'sparsity_' + str(
            args.amount_sparsity)
        df_costs = pd.DataFrame(data=comm_costs_accum)
        df_acc = pd.DataFrame(data=glo_acc)
        df_top_5_acc=pd.DataFrame(data=top5_acc_test)
        df_sp = pd.DataFrame(data=glo_sp)
        filename_acc = name_tail + '_acc''.csv'
        filename_sp = name_tail + '_sprecord_''.csv'
        filename_cost = name_tail + '_commcost_''.csv'
        filename_top5_acc = name_tail + '_top5acc_''.csv'
        df_acc.to_csv(filename_acc, index=False)
        df_sp.to_csv(filename_sp, index=False)
        df_costs.to_csv(filename_cost,index=False)
        if len(df_top_5_acc)!=0:
            df_top_5_acc.to_csv(filename_top5_acc,index=False)




if __name__ == "__main__":
    feddst_server()
