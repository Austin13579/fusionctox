import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
import random
import argparse
import copy

from metrics import evaluate
from model import FusionCTox_Sum
from utils import Encode_Data


def set_random_seed(seed):
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def training(model, loader, optimizer):
    print('Training on {} samples...'.format(len(loader.dataset)))
    model.train()
    losses = []
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    for batch_idx, (fp, seq1, seq2, label) in enumerate(loader):
        fp, seq1, seq2 = fp.to(device), seq1.to(device), seq2.to(device)
        optimizer.zero_grad()

        output = model(fp, seq1, seq2)
        score = torch.squeeze(output)
        loss = loss_fn(score, label.float().to(device))

        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())

        total_preds = torch.cat((total_preds, score.cpu()), 0)
        total_labels = torch.cat((total_labels, label.flatten().cpu()), 0)
    return np.mean(losses)


def validation(model, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, (fp, seq1, seq2, label) in enumerate(loader):
            fp, seq1, seq2 = fp.to(device), seq1.to(device), seq2.to(device)
            output = model(fp, seq1, seq2)

            score = torch.squeeze(output)
            total_preds = torch.cat((total_preds, score.cpu()), 0)
            total_labels = torch.cat((total_labels, label.flatten().cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="herg", help='which dataset')
    parser.add_argument('--rs', type=int, default=0, help='which random seed')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    args = parser.parse_args()

    set_random_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_df = pd.read_csv('../dataset/datas/' + args.data + '_train' + str(args.rs) + '.csv')
    valid_df = pd.read_csv('../dataset/datas/' + args.data + '_valid' + str(args.rs) + '.csv')
    test_df60 = pd.read_csv('../dataset/' + args.data + '/' + args.data + '_eval_60.csv')
    test_df70 = pd.read_csv('../dataset/' + args.data + '/' + args.data + '_eval_70.csv')

    train_set = Encode_Data(train_df.index.values, train_df, args.data)
    valid_set = Encode_Data(valid_df.index.values, valid_df, args.data)
    test_set60 = Encode_Data(test_df60.index.values, test_df60, args.data)
    test_set70 = Encode_Data(test_df70.index.values, test_df70, args.data)

    # Build dataloader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    test_loader60 = DataLoader(test_set60, batch_size=args.batch_size, shuffle=False)
    test_loader70 = DataLoader(test_set70, batch_size=args.batch_size, shuffle=False)

    # Model
    tox_model = FusionCTox_Sum().to(device)
    optim = torch.optim.Adam(tox_model.parameters(), lr=args.lr)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    models = []
    val_results, val_models, t60_results, t70_results = [], [], [], []
    for epoch in range(args.epochs):
        print("Epoch: ", epoch + 1)
        train_loss = training(tox_model, train_loader, optim)
        print("Train Loss: ", train_loss)

        print("Validation")
        valid_true, valid_prob = validation(tox_model, valid_loader)
        res_valid = evaluate(valid_true, valid_prob)
        val_results.append(res_valid['AUROC'])
        print("Valid: ", res_valid)
        models.append(copy.deepcopy(tox_model.state_dict()))

        t60_true, t60_prob = validation(tox_model, test_loader60)
        res_60 = evaluate(t60_true, t60_prob)
        t60_results.append(res_60)
        print("Test60: ", res_60)

        t70_true, t70_prob = validation(tox_model, test_loader70)
        res_70 = evaluate(t70_true, t70_prob)
        t70_results.append(res_70)
        print("Test70:", res_70)

    best_val_result, best_index = torch.topk(torch.tensor(val_results), k=1)
    best_model = models[best_index]
    torch.save(best_model, 'ckpts/' + args.data + '_' + str(args.rs) + '_sum.pth')
    print("Best Epoch:", best_index)
    best_t60_result, best_t70_result = t60_results[best_index], t70_results[best_index]
    pd.DataFrame([best_t60_result]).to_csv('results/' + args.data + '_t60_sum_' + str(args.rs) + '.csv',
                                           columns=['Accuracy', 'AUROC', 'AUPRC', 'Recall', 'F1', 'MCC'], index=False)

    pd.DataFrame([best_t70_result]).to_csv('results/' + args.data + '_t70_sum_' + str(args.rs) + '.csv',
                                           columns=['Accuracy', 'AUROC', 'AUPRC', 'Recall', 'F1', 'MCC'], index=False)
