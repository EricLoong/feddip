import random
import time
from client import *
from fl_functions import *
from exp_args import args_parser
import pandas as pd
from torch.utils.data import random_split
from functions import get_subsets_dict, get_clf_key, get_fe_key

tranform_train = transforms.Compose(
    [transforms.Resize((227, 227)), transforms.RandomHorizontalFlip(p=0.7), transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
tranform_test = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# preparing the train, validation and test dataset
torch.manual_seed(123)
train_ds = CIFAR10("data/", train=True, download=True, transform=tranform_train)  # 40,000 original images + transforms
val_size = 10000  # there are 10,000 test images and since there are no transforms performed on the test, we keep the validation as 10,000
train_size = len(train_ds) - val_size
train_ds_1, val_ds = random_split(train_ds,
                                [train_size, val_size])  # Extracting the 10,000 validation images from the train set
test_ds = CIFAR10("data/", train=False, download=True, transform=tranform_test)  # 10,000 images

# passing the train, val and test datasets to the dataloader
train_dl = DataLoader(train_ds_1, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

args = args_parser()
#args.group = {0: 9, 3: 9, 6: 9, 8: 9, 11: 9, 13: 9, 16: 9, 18: 9} vgg11
n_users = args.num_users
print(n_users)
user_index = list(range(n_users))
data = load_data()

idx_users = list()
n_sample_user = int(40000 / n_users)
n_sample_user_test = int(10000 / n_users)
train_list = train_ds_1.indices
test_list = list(range(10000))
random.shuffle(train_list)
random.shuffle(test_list)

for ids in user_index:
    idx_users.append([ids, [train_list[ids * n_sample_user:(ids + 1) * n_sample_user],
                      test_list[ids * n_sample_user_test:(ids + 1) * n_sample_user_test], list(range(10000))]])

start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global_model = models.alexnet(pretrained=False)
ini_server = ServerCollect(args=args, device=device)
global_model.to(device)
global_model.train()

train_loss, train_accuracy = [], []
glo_loss, glo_acc = [], []
val_acc_list, net_list = [], []
cv_loss, cv_acc = [], []
print_every = 2
val_loss_check = [0]
t_b = 10
global_model_fe = get_subsets_dict(dict_target=global_model.state_dict(),
                                   target_key=get_fe_key(global_model.state_dict()))

for epoch in range(args.epochs):
    local_waps, local_test_accuracy = [], []
    train_epoch_loss = list()
    global_model.train()
    if epoch<t_b:
        k = 0
        for index in user_index:
            local_model = ClientUpdate(args=args, dataset=data, index_list=idx_users[index][1])
            temp_model = copy.deepcopy(global_model)
            local_val_temp, local_loss_temp = local_model.train_model(model=temp_model, global_iter=epoch)
            local_test_accuracy.append([epoch, index, local_val_temp])
            train_epoch_loss.append([epoch, index, local_loss_temp])
            temp_server_wap = temp_model.state_dict()
            local_waps.append(temp_server_wap)
            k += 1
            print(f'Finish Local Training for {k} users.')

        temp_time = time.time()
        print(f'Local Trainings for global epoch {epoch} is: Done.')

        server = ServerCollect(args=args, device=device)
        global_receive = server.average_weights(weight=local_waps)
        global_model.load_state_dict(global_receive)
        global_model_fe = get_subsets_dict(dict_target=global_model.state_dict(),
                                           target_key=get_fe_key(global_model.state_dict()))
        temp_train_acc, temp_train_loss = server.inference(model=copy.deepcopy(global_model), total_test=train_dl)
        temp_glo_acc, temp_glo_loss = server.inference(model=copy.deepcopy(global_model), total_test=test_dl)
        temp_val_acc, temp_val_loss = server.inference(model=copy.deepcopy(global_model), total_test=val_dl)
        glo_loss.append(temp_glo_loss)
        glo_acc.append(temp_glo_acc)
        val_acc_list.append(temp_val_acc)
        val_loss_check.append(temp_val_loss)
        train_accuracy.append(temp_train_acc)
        train_loss.append(temp_train_loss)

    else:
        k = 0
        print('Only Send Classifier Weights')
        for index in user_index:
            local_model = ClientUpdate(args=args, dataset=data, index_list=idx_users[index][1])
            temp_model = copy.deepcopy(global_model)
            local_val_temp, local_loss_temp = local_model.train_model(model=temp_model, global_iter=epoch)
            local_test_accuracy.append([epoch, index, local_val_temp])
            train_epoch_loss.append([epoch, index, local_loss_temp])
            temp_server_wap_clf = get_subsets_dict(dict_target=temp_model.state_dict(),
                                                   target_key=get_clf_key(temp_model.state_dict()))
            local_waps.append(temp_server_wap_clf)
            k += 1
            print(f'Finish Local Training for {k} users.')

        temp_time = time.time()
        print(f'Local Trainings for global epoch {epoch} is: Done.')

        server = ServerCollect(args=args, device=device)
        fe_unite = copy.deepcopy(global_model_fe)
        global_receive = server.average_weights(weight=local_waps)
        fe_unite.update(global_receive)
        global_model.load_state_dict(fe_unite)
        temp_train_acc, temp_train_loss = server.inference(model=copy.deepcopy(global_model), total_test=train_dl)
        temp_glo_acc, temp_glo_loss = server.inference(model=copy.deepcopy(global_model), total_test=test_dl)
        temp_val_acc, temp_val_loss = server.inference(model=copy.deepcopy(global_model), total_test=val_dl)
        glo_loss.append(temp_glo_loss)
        glo_acc.append(temp_glo_acc)
        val_acc_list.append(temp_val_acc)
        val_loss_check.append(temp_val_loss)
        train_accuracy.append(temp_train_acc)
        train_loss.append(temp_train_loss)
    # calculate an error and send back to the clients
    if abs(val_loss_check[-1]-val_loss_check[-2])<5e-04 and epoch>50:
        print("--- %s seconds ---" % (time.time() - start_time))
        print(f'Finish after {epoch} Epochs')
        break

# Store model, loss and validation accuracy
df_loss = pd.DataFrame(data=train_loss)
df_train_acc = pd.DataFrame(data=train_accuracy)
df_val = pd.DataFrame(data=val_acc_list)
df_test_loss = pd.DataFrame(data=cv_loss)
df_test_acc = pd.DataFrame(data=cv_acc)
df_glob_loss = pd.DataFrame(data=glo_loss)
df_glob_acc = pd.DataFrame(data=glo_acc)

print(glo_acc)

file_name_loss = r'loss_E50U5_alexnet_flper_NP.csv'
file_name_val = r'val_E50U5_alexnet_flper_NP.csv'
file_name_test_loss = r'tloss_E50U5_alexnet_flper_NP.csv'
file_name_test_acc = r'tacc_E50U5_alexnet_flper_NP.csv'
file_name_train_acc = r'trainacc_E50U5_alexnet_flper_NP.csv'
file_name_global_loss = r'gloss_E50U5_alexnet_flper_NP.csv'
file_name_global_acc = r'gacc_E50U5_alexnet_flper_NP.csv'

#first job is marked by E50U5 resnet

df_loss.to_csv(file_name_loss, index=False)
df_val.to_csv(file_name_val, index=False)
df_test_loss.to_csv(file_name_test_loss, index=False)
df_test_acc.to_csv(file_name_test_acc, index=False)
df_train_acc.to_csv(file_name_train_acc,index=False)
df_glob_acc.to_csv(file_name_global_acc, index=False)
df_glob_loss.to_csv(file_name_global_loss, index=False)