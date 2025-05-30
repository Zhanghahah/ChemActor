from utils import *
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="A simple argument parser")

    parser.add_argument('--path', default=None, type=str)
    parser.add_argument('--use_tok', default=False, action='store_true')
    args = parser.parse_args()
    return args

def read_dataset(args):
    print(f'Reading {args.path}...')
    with open(args.path, 'r', encoding='utf-8') as f:
        test_tgt = [json.loads(line) for line in f.readlines()]
    print(f'{len(test_tgt)} samples read.')
    gt_list = [i['targets'] for i in test_tgt]
    pred_list = [i['predictions'] for i in test_tgt]
    return gt_list, pred_list

def read_result(args):
    path = ''
    
    results = pd.read_csv(path)
    inps = results['reaction_name'].tolist()
    gt_list = results['action_labels'].tolist()
    pred_list = results['action_preds_1'].tolist()
    calculator = Metric_calculator()
    calculator(gt_list, pred_list, args.use_tok)

if __name__ == "__main__":
    args=parse_args()
    read_result(args)