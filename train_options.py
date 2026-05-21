import argparse

parser = argparse.ArgumentParser(description='Training script arguments for flood detection networks')

parser.add_argument('--dataset_root', type=str, default="./Benchmark", help='Base path where all datasets are located.')
parser.add_argument('--model_name', type=str, default='SAFENet_Flood', help='Model name')
parser.add_argument('--dataset', type=str, default='S1GFloods7', choices=['S1GFloods7', 'S1GFloods', 'OmbriaS1'], 
                                                                        help='Name of the dataset to use.')

parser.add_argument('--batch_size', type=int, default=4, help='Batch size for the DataLoaders.')
parser.add_argument('--num_classes', type=int, default=2, help='number of classes, 2 for binary classification.')

parser.add_argument('--img_size', default=256, type=int, help='Input image size (H=W).')
parser.add_argument('--num_epochs', default=100, type=int, help='Number of training epochs.')
parser.add_argument('--save_dir', type=str, default="./WORK_DIR/checkpoints", help='Directory to save best model checkpoint.')
parser.add_argument('--log_folder', type=str, default="./WORK_DIR/train_history", help='Directory to save training history/logs.')

parser.add_argument('--loss', default='NoiseRobustLoss', type=str, help='Type of loss function to use.')

parser.add_argument('--alpha', default=0.1, type=float, help='Alpha parameter for NoiseRobustLoss.')
parser.add_argument('--beta', default=0.9, type=float, help='Beta parameter for NoiseRobustLoss.')

parser.add_argument('--optimizer', default='Adam', type=str, help='Optimizer to use for training.')             
parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate.')             
parser.add_argument('--beta1', default=0.9, type=float, help='Beta1 parameter for Adam/AdamW optimizer.')        
parser.add_argument('--beta2', default=0.999, type=float, help='Beta2 parameter for Adam/AdamW optimizer.')           

opt = parser.parse_args()