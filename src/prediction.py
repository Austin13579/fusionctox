import pandas as pd
import torch
from torch.utils.data import DataLoader
import argparse

from metrics import evaluate
from model import HybridCTox
from utils import Encode_Data


def validation(m1,m2,m3,m4, loader,weights):
    m1.eval()
    m2.eval()
    m3.eval()
    m4.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    m=torch.nn.Sigmoid()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, (fp, seq1,seq2,label) in enumerate(loader):
            fp, seq1,seq2 = fp.to(device), seq1.to(device), seq2.to(device)
            o1= torch.squeeze(m(m1(fp)))
            o2= torch.squeeze(m(m2(fp, seq1,seq2)))
            o3= torch.squeeze(m(m3(fp, seq1,seq2)))
            o4= torch.squeeze(m(m4(fp, seq1,seq2)))
            score=o1*weights[0]+o2*weights[1]+o3*weights[2]+o4*weights[3]
            total_preds = torch.cat((total_preds, score.cpu()), 0)
            total_labels = torch.cat((total_labels, label.flatten().cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="herg", help='which dataset')
    parser.add_argument('--rs', type=int, default=0, help='which random seed')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    args = parser.parse_args()

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    test_df60 = pd.read_csv('../dataset/'+args.data + '/' + args.data + '_eval_60.csv')
    test_df70 = pd.read_csv('../dataset/'+args.data + '/' + args.data + '_eval_70.csv')

    test_set60 = Encode_Data(test_df60.index.values, test_df60, args.data)
    test_set70 = Encode_Data(test_df70.index.values, test_df70,args.data)

    test_loader60 = DataLoader(test_set60, batch_size=args.batch_size, shuffle=False)
    test_loader70 = DataLoader(test_set70, batch_size=args.batch_size, shuffle=False)

    model1=FP_Model().to(device)
    model2=FusionCTox_Sum().to(device)
    model3=FusionCTox_Concat().to(device)
    model4=FusionCTox_Film().to(device)

    model1.load_state_dict(torch.load('ckpts/'+args.data+'_'+str(args.rs)+'_fp.pth',map_location=device))
    model2.load_state_dict(torch.load('ckpts/'+args.data+'_'+str(args.rs)+'_sum.pth',map_location=device))
    model3.load_state_dict(torch.load('ckpts/'+args.data+'_'+str(args.rs)+'_concat.pth',map_location=device))
    model4.load_state_dict(torch.load('ckpts/'+args.data+'_'+str(args.rs)+'_film.pth',map_location=device))



    weight=[0.25,0.25,0.25,0.25]
    t60_true, t60_prob = validation(model1,model2,model3,model4, test_loader60,weight)
    res_60 = evaluate(t60_true, t60_prob)

    t70_true, t70_prob = validation(model1,model2,model3,model4, test_loader70,weight)
    res_70 = evaluate(t70_true, t70_prob)


    pd.DataFrame([res_60]).to_csv('results/' + args.data + '_t60_ensemble_' + str(args.rs) + '.csv',
                                           columns=['Accuracy', 'AUROC', 'AUPRC', 'Recall', 'F1', 'MCC'], index=False)

    pd.DataFrame([res_70]).to_csv('results/' + args.data + '_t70_ensemble_' + str(args.rs) + '.csv',
                                  columns=['Accuracy', 'AUROC', 'AUPRC', 'Recall', 'F1', 'MCC'], index=False)
