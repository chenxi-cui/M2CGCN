import torch
from network import Network
from torch.utils.data import Dataset
import numpy as np
import argparse
from loss import Loss
from dataloader import load_data
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import pandas as pd


Dataname = 'BIC'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=624, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--tune_epochs", default=50)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=256)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if args.dataset == "AML":
    args.con_epochs = 50
    seed = 10
if args.dataset == "BIC":
    args.con_epochs = 50
    seed = 10
if args.dataset == "COAD":
    args.con_epochs = 50
    seed = 10
if args.dataset == "GBM":
    args.con_epochs = 50
    seed = 10
if args.dataset == "KIRC":
    args.con_epochs = 50
    seed = 10
if args.dataset == "LIHC":
    args.con_epochs = 50
    seed = 10
if args.dataset == "LUSC":
    args.con_epochs = 50
    seed = 10
if args.dataset == "SKCM":
    args.con_epochs = 50
    seed = 10
if args.dataset == "OV":
    args.con_epochs = 50
    seed = 10
if args.dataset == "SARC":
    args.con_epochs = 50
    seed = 10
if args.dataset == "METABRIC":
    args.con_epochs = 50
    seed = 10


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True




dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )


def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, xrs, _, ar, ars = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
            loss_list.append((criterion(torch.Tensor(ar[v]), ars[v])))#ar——KNN图 ars——重构
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print(len(data_loader))
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


def contrastive_train(epoch):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, xrs, zs, _, _ = model(xs)
        loss_list = []
        for v in range(view):
            for w in range(v+1, view):
                loss_list.append(criterion.forward_feature(hs[v], hs[w]))
                loss_list.append(criterion.forward_label(qs[v], qs[w]))
            loss_list.append(mes(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))


def make_pseudo_label(model, device):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    model.eval()
    scaler = MinMaxScaler()
    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            hs, _, _, _, _, _ = model.forward(xs)
        for v in range(view):
            hs[v] = hs[v].cpu().detach().numpy()
            hs[v] = scaler.fit_transform(hs[v])

    kmeans = KMeans(n_clusters=class_num, n_init=100)
    new_pseudo_label = []
    for v in range(view):
        Pseudo_label = kmeans.fit_predict(hs[v])
        Pseudo_label = Pseudo_label.reshape(data_size, 1)
        Pseudo_label = torch.from_numpy(Pseudo_label)
        new_pseudo_label.append(Pseudo_label)

    return new_pseudo_label


def match(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    new_y = np.zeros(y_true.shape[0])
    for i in range(y_pred.size):
        for j in row_ind:
            if y_true[i] == col_ind[j]:
                new_y[i] = row_ind[j]
    new_y = torch.from_numpy(new_y).long().to(device)
    new_y = new_y.view(new_y.size()[0])
    return new_y


def fine_tuning(epoch, new_pseudo_label):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    tot_loss = 0.
    cross_entropy = torch.nn.CrossEntropyLoss()
    for batch_idx, (xs, _, idx) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, qs, _, _, _, _ = model(xs)
        loss_list = []
        for v in range(view):
            p = new_pseudo_label[v].numpy().T[0]
            with torch.no_grad():
                q = qs[v].detach().cpu()
                q = torch.argmax(q, dim=1).numpy()
                p_hat = match(p, q)
            loss_list.append(cross_entropy(qs[v], p_hat))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

def inference(loader, model, device, view, data_size):
    """
    :return:
    total_pred: prediction among all modalities
    pred_vectors: predictions of each modality, list
    labels_vector: true label
    Hs: high-level features
    Zs: low-level features
    """
    model.eval()
    soft_vector = []
    pred_vectors = []
    Hs = []
    Zs = []
    for v in range(view):
        pred_vectors.append([])
        Hs.append([])
        Zs.append([])


    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            qs, preds = model.forward_cluster(xs)
            hs, _, _, zs, _, _ = model.forward(xs)
            q = sum(qs)/view
        for v in range(view):
            hs[v] = hs[v].detach()
            zs[v] = zs[v].detach()
            preds[v] = preds[v].detach()
            pred_vectors[v].extend(preds[v].cpu().detach().numpy())
            Hs[v].extend(hs[v].cpu().detach().numpy())
            Zs[v].extend(zs[v].cpu().detach().numpy())
        q = q.detach()
        soft_vector.extend(q.cpu().detach().numpy())
    total_pred = np.argmax(np.array(soft_vector), axis=1)
    for v in range(view):
        Hs[v] = np.array(Hs[v])
        Zs[v] = np.array(Zs[v])
        pred_vectors[v] = np.array(pred_vectors[v])
    return total_pred, pred_vectors, Hs, Zs


if not os.path.exists('./models'):
    os.makedirs('./models')
T = 1
for i in range(T):
    print("ROUND:{}".format(i+1))
    setup_seed(seed)
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
    print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)

    epoch = 1
    while epoch <= args.mse_epochs:
        pretrain(epoch)
        epoch += 1
    while epoch <= args.mse_epochs + args.con_epochs:
        contrastive_train(epoch)
        epoch += 1
    new_pseudo_label = make_pseudo_label(model, device)
    while epoch <= args.mse_epochs + args.con_epochs + args.tune_epochs:
        fine_tuning(epoch, new_pseudo_label)
        if epoch == args.mse_epochs + args.con_epochs + args.tune_epochs:
            state = model.state_dict()
            torch.save(state, './models/' + args.dataset + '.pth')
            print('Saving..')

            test_loader = DataLoader(
                dataset,
                batch_size=624,
                shuffle=False,
            )
            total_pred, pred_vectors, high_level_vectors, low_level_vectors = inference(test_loader, model, device,
                                                                                                       view, data_size)
            #np.savetxt('AML.txt', total_pred,fmt='%d')

            df = pd.DataFrame({'': range(1, len(total_pred) + 1), 'x': total_pred})
            df.to_excel('BIC.xlsx', index=False)
        epoch += 1
