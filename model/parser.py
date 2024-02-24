import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=6000, type=int)
    parser.add_argument('--lr_drop', default=2000, type=int)
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--out_channels', default=2, type=int)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--root_path', default='/disk1/guozhanqiang/Cerebral/scribble/', type=str)

    # * Loss coefficients
    parser.add_argument('--CrossEntropy_loss_coef', default=1, type=float)
    parser.add_argument('--Avg', default=1, type=float)
    
    # dataset parameters
    parser.add_argument('--dataset', default='SSVS_XRAY_Coronary', type=str)
    parser.add_argument('--full', default=False, type=bool)
    # set your outputdir 
    parser.add_argument('--output_dir', default='./',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    return parser
