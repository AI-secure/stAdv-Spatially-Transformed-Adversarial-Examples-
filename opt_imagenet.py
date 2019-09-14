import argparse
import os

# Training settings
def parse_opt():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
    # parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                     help='input batch size for training (default: 64)')

    parser.add_argument('--batch_size', type=int, default=1, help='batch size' )

    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--fineSize', type=int, default=224, 
                        help='how many batches to wait before logging training status')
    parser.add_argument('--t_l', type=int, default=1, help='source label')
    parser.add_argument('--s_l', type=int, default=5, help='target_label')
    parser.add_argument('--num_classes', type=int, default=1000, help='num_classes')
    parser.add_argument('--gpu_ids', type=int, default=0, help='GPU')
    # parser.add_argument('--ld_tv', type=float, default=100, help='lambda_tv')
    parser.add_argument('--ld_tv', type=float, default=50, help='lambda_tv')
    
    parser.add_argument('--ld_adv', type=float, default=0.005, help='lambda_adv')
    # parser.add_argument('--ld_adv', type=float, default=0.005, help='lambda_adv')
    # parser.add_argument('--ld_adv', type=float, default=0.001875, help='lambda_adv')
    parser.add_argument('--ld_l2', type=float, default=0.0, help='lambda_l2')
    parser.add_argument('--ld_flowm', type=float, default=0.000, help='ld_flow_mean')
    parser.add_argument('--binary_search_steps', type=int, default= 100 , help="binary_search_steps")
    parser.add_argument('--prefix', type=str, default="/raid/chaowei/stn/cifar10/" , help="prefix")
    parser.add_argument('--hingle_flag', type=int, default=1, help="hingle_flag")
    # parser.add_argument('--dataset', type=int, default=0, help="0:MNIST,1:CIFAR10,3:IMAGENET")
    # parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

    parser.add_argument('--gpu_id', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

    parser.add_argument('-d', '--dataset', default='cifar10', type=str)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('--image_path', default = 'data/dataset/')
    parser.add_argument('--save_path', default = 'data/adv_images')

    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    return args
