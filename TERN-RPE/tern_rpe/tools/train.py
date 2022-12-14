from .utils.getter import *
import argparse

parser = argparse.ArgumentParser('Training Object Detection')
parser.add_argument('--print_per_iter', type=int, default=300, help='Number of iteration to print')
parser.add_argument('--val_interval', type=int, default=2, help='Number of epoches between valing phases')
parser.add_argument('--save_interval', type=int, default=1000, help='Number of steps between saving')
parser.add_argument('--resume', type=str, default=None,
                    help='whether to load weights from a checkpoint, set None to initialize')
parser.add_argument('--saved_path', type=str, default='./weights')
parser.add_argument('--top_k', type=int, default=10, help='top k validation')
parser.add_argument('--no_visualization', action='store_false', help='whether to visualize box to ./sample when validating (for debug), default=on')

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True
seed_everything()

def train(args, config):
    # os.environ['CUDA_VISIBLE_DEVICES'] = config.globals['gpu_devices'] # UNCOMMENT ON GPU
    # num_gpus = len(config.globals['gpu_devices'].split(',')) # UNCOMMENT ON GPU

    # device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') # UNCOMMENT ON GPU or CPU
    device = torch.device('mps') # M1 Mac
    # devices_info = get_devices_info(config.globals['gpu_devices']) # UNCOMMENT ON GPU
    
    trainloader = get_instance(config.trainloader, device=device)
    valloader = get_instance(config.valloader) 

    trainset = trainloader.dataset
    valset = valloader.dataset

    net = get_instance(config.model, device=device)

    optimizer, optimizer_params = get_lr_policy(config.lr_policy)

    criterion = get_loss_fn(config.loss)

    valset1 = get_instance(config.valset1)
    valset2 = get_instance(config.valset2)

    metric = RetrievalScore(
            valset1, valset2, 
            max_distance = 1.3,
            top_k=args.top_k,
            metric_names=["R@1", "R@5", "R@10"],
            dimension=config.model['args']['d_embed'],
            save_results=True)

    model = Retriever(
            model = net,
            metrics = metric,
            criterion=criterion,
            scaler=NativeScaler(),
            optimizer= optimizer,
            optim_params = optimizer_params,     
            device = device)

    if args.resume is not None:                
        load_checkpoint(model, args.resume)
        start_epoch, start_iter, best_value = get_epoch_iters(args.resume)
    else:
        print('Not resume. Load pretrained weights...')
        start_epoch, start_iter, best_value = 0, 0, 0.0
        
    scheduler, step_per_epoch = get_lr_scheduler(
        model.optimizer, train_len=len(trainloader),
        lr_config=config.lr_scheduler,
        num_epochs=config.globals['num_epochs'])

    if args.resume is not None:                 
        old_log = find_old_log(args.resume)
    else:
        old_log = None

    args.saved_path = os.path.join(
        args.saved_path, 
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    trainer = Trainer(config,
                     model,
                     trainloader, 
                     valloader,
                     checkpoint = Checkpoint(save_per_iter=args.save_interval, path = args.saved_path),
                     best_value=best_value,
                     logger = Logger(log_dir=args.saved_path, resume=old_log),
                     scheduler = scheduler,
                     evaluate_per_epoch = args.val_interval,
                     visualize_when_val = args.no_visualization,
                     step_per_epoch = step_per_epoch)
    print()
    print("##########   DATASET INFO   ##########")
    print("Trainset: ")
    print(trainset)
    print("Valset: ")
    print(valset)
    print()
    print(trainer)
    print()
    print(config)
    # print(f'Training with {num_gpus} gpu(s): ') # UNCOMMENT ON GPU
    # print(devices_info) # UNCOMMENT ON GPU
    print(f"Start training at [{start_epoch}|{start_iter}]")
    print(f"Current best R@10: {best_value}")

    trainer.fit(start_epoch = start_epoch, start_iter = start_iter, num_epochs=config.globals['num_epochs'], print_per_iter=args.print_per_iter)

    

if __name__ == '__main__':
    
    args = parser.parse_args()
    config = Config("./tools/configs/yaml/tern.yaml")

    train(args, config)