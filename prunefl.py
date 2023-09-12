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


# For comparison, we only do experiment for AlexNet.
# Local iteration equals to 5
# The idea of PruneFL: Firstly initialize the mask on one client until converge. For each global round, local client update
# several iterations and apply the masks.




def prunefl_server():
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
    # Since the selected user is randomly. We just use the first user in list to conduct initialization.
    # Initilize Clients
    clients = []
    for c in range(args.num_users):
        cl = Client(args=args, dataset=data, index_list=idx_users[c][1], model=copy.deepcopy(global_model),
                    client_idx=c)
        clients.append(cl)
        torch.cuda.empty_cache()

    print('Finish client initilization')
    inputs_4_time, _ = next(iter(train_dl))
    layer_times = measure_layer_time_exclude_bias(model=global_model, inputs=inputs_4_time)
    first_key = list(layer_times.keys())[0]
    layer_times[first_key] = 1e-10
    #mask_one_initial = init_mask(model=global_model, args=args)
    init_model = copy.deepcopy(global_model)
    init_model.to(device)
    init_model.train()
    local_delta_init_rounds = list()
    args_init = copy.deepcopy(args)
    # This is for initial mask
    args_init.num_users = 100
    index_list_init, data_train_test_init, pctg_4_avg_init = preprocessed_data(data_list=data_list, batch_size=args_init.local_batchsize,
                                                    n_users=args_init.num_users,
                                                    data_name=data_name, num_work=args_init.num_workers)
    user_index_init, idx_users_init = index_list_init
    init_user = 0
    comm_costs_accum = [0]
    init_client= Client(args=args_init, dataset=data, model=init_model, client_idx=init_user,
           index_list=idx_users_init[init_user][1])

    init_reconfig_mask= None
    for r in range(1,args.init_rounds):
        init_reconfig=(r-1)%5==0
        init_aggr_grad = []
        init_model_temp = copy.deepcopy(init_client.model)
        # Mask is not used in initial pruning
        temp_acc,_,_ = init_client.train_model(global_iter=r,learning_rate=args_init.lr,mask_applied=init_reconfig_mask)
        #if args.parallel:
        #    prune_para_local = generate_prune_param(model=init_client.model.module, fc=True,bias=False)
        #else:
        #    prune_para_local = generate_prune_param(model=init_client.model, fc=True,bias=False)
        #prune_clean_model(param_pruned=prune_para_local, amount=args.init_sparsity)
        local_grad_temp = dict()
        for name, param in init_client.model.named_parameters():
            local_grad_temp[name] = param.grad.square_()
        init_aggr_grad.append(local_grad_temp)
        if init_reconfig:
            init_server= ServerCollect(args=args, device=device)
            init_aggrted_g=init_server.average_weights(init_aggr_grad,selected_clients=[0], pctg=pctg_4_avg_init)
            init_reconfig_mask = prunefl_reconfig(init_aggrted_g,layer_times=layer_times,prunable_params=args.init_sparsity)
            init_server.set_weight_by_mask(init_client.model,mask=init_reconfig_mask)
            print('Reconfiguration at round:', r)
        #temp_delta_local_init = inform_loss_norm(model1=init_model_temp, model2=init_model)
        #local_delta_init_rounds.append(temp_delta_local_init)
        if r > 5:
            #diff_init = local_delta_init_rounds[r] - local_delta_init_rounds[r - 1]
            if data_name!='cifar100':
                if temp_acc > 0.11:
                    print(f'Initial rounds (delta){local_delta_init_rounds}')
                    break
            else:
                if temp_acc > 0.02:
                    print(f'Initial rounds (delta){local_delta_init_rounds}')
                    break


    # Send the initial models and mask to all users
    comm_costs_accum.append(args.num_users*(comm_costs_in_mb(model=init_model,sparsity=args.init_sparsity)+comm_costs_in_mb(model=init_model,sparsity=args.init_sparsity,mask=True)))
    global_model.load_state_dict(init_model.state_dict())
    print('Finish initial pruning!')
    init_model_masked = generate_mask(model=init_model, args=args)

    train_loss, train_accuracy = [], []
    top5_acc_train, top5_acc_test = [], []
    glo_loss, glo_acc = [], []
    glo_sp = []
    # Start training
    global_model.train()
    global_model.to(device)
    mask_reconfig = init_model_masked
    initial_lr = copy.deepcopy(args.lr)
    for epoch in tqdm.tqdm(range(args.epochs)):
        args.lr = initial_lr * (0.998**epoch)
        global_model.train()
        reconfig = (epoch + 1) % args.reconfig_interval == 0
        local_waps = []
        aggr_gradients = []
        glo_sp_temp = compute_sparsity(model=global_model)
        glo_sp.append(glo_sp_temp)
        user_index_ts = np.random.choice(user_index, int(args.frac * args.num_users), replace=False)

        for index in tqdm.tqdm(user_index_ts):
            cur_client = clients[index]
            temp_model = copy.deepcopy(global_model)
            cur_client.download_global_model(model_stat_dict=temp_model.state_dict())
            cur_client.train_model(global_iter=epoch, learning_rate=args.lr,mask_applied=mask_reconfig)

            temp_server_wap = cur_client.upload_local_model()
            local_waps.append(temp_server_wap)

            local_grad_temp = dict()
            for name, param in cur_client.model.named_parameters():
                local_grad_temp[name] = param.grad.square_()
            aggr_gradients.append(local_grad_temp)
            # This appends the squared gradients

        server = ServerCollect(args=args, device=device)
        global_receive = server.average_weights(weight=local_waps,pctg=pctg_4_avg,selected_clients=user_index_ts)
        if reconfig:
            global_model.load_state_dict(global_receive)
            aggrted_g = server.average_weights(aggr_gradients,pctg=pctg_4_avg,selected_clients=user_index_ts)
            prune_amount = args.amount_sparsity + (args.init_sparsity - args.amount_sparsity) * (
                    (1 - (epoch / args.epochs)) ** 3)
            mask_reconfig = prunefl_reconfig(aggregate_gradients=aggrted_g, layer_times=layer_times,
                                             prunable_params= prune_amount)
            #server.set_weight_by_mask(model=global_model, mask=mask_reconfig)
        else:
            global_model.load_state_dict(global_receive)
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
        upload_costs_temp = args.num_users*args.frac*comm_costs_in_mb(model=global_model,sparsity=(glo_sp_temp/100))
        download_costs_temp = args.num_users*args.frac*comm_costs_in_mb(model=global_model,sparsity=(glo_sp_temp/100))

        if reconfig:
            # This is because of adding the costs of gradients allocation and mask updates
            upload_costs_temp += args.num_users*args.frac*comm_costs_in_mb(model=global_model,sparsity=0)
            download_costs_temp += args.num_users*args.frac*comm_costs_in_mb(model=global_model,sparsity=(glo_sp_temp/100),mask=True)


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
    if args.prunefl:
        name_tail = "Global_epochs_" + str(args.epochs) + '_Local_epochs_' + str(
            args.local_ep) + '_model_name_' + model_name + 'PruneFL_' + 'sparsity_' + str(
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
    prunefl_server()
