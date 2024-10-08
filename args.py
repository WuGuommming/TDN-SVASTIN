import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--dataset', type=str, help='The dataset that was attacked, UCF101 or Kinetic400', default="UCF101")
    parser.add_argument('--config_file', type=str, help='Model configuration file',
                        default='tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_kinetics400-rgb.py')
    parser.add_argument('--checkpoint_file', type=str, help='Model configuration file',
                        default='tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_kinetics400-rgb_20220906-23cff032.pth')
    parser.add_argument('--device', type=str, help='Choosing device,cuda or cpu', default="cuda")
    parser.add_argument('--models', help='Target classifiers, MVIT or SLOWFAST or TSN',
                        default='MVIT')
    parser.add_argument('--inputpath', help='', default='C:\\data\\dataset\\UCF101\\UCF-101')
    parser.add_argument('--outputpath', help='', default='')
    args = parser.parse_args()
    return args