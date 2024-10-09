import torch.optim as optim
import kornia.augmentation as K
from copy import deepcopy
from defense.ueraser import UEraser, UEraser_jpeg
import os
import itertools
import argparse
import numpy as np
import torch.nn.functional as F
from madrys import MadrysLoss
from nets import *
import torchvision.models as models
from util_sharp import *
from tensorboardX import SummaryWriter

def layer_sharpness(args, model, criterion,trainloader=None,epsilon=0.1,test_acc=0,test_loss=0,epoch=0,sharp_lr_min=0.01,sharp_lr_max=0.01):
    # if "CIFAR" in args.dataset:
    #     norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    # else:
    #     norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = model.cuda()
    if trainloader is None:
        # original dataset
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainloader = torch.utils.data.DataLoader(datasets.CIFAR10(root="../../dataset/cifar-10/", train=False, download=False, transform=transform_test), batch_size=256, shuffle=True, num_workers=4) 
    origin_total = 0
    origin_loss = test_loss
    origin_acc = test_acc/100

    if test_loss == 0:
        model.eval()
        origin_acc = 0.
        origin_loss = 0.
        with torch.no_grad():
            for inputs, targets in trainloader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                origin_total += targets.shape[0]
                origin_loss += criterion(outputs, targets).item() * targets.shape[0]
                _, predicted = outputs.max(1)
                origin_acc += predicted.eq(targets).sum().item()        
            
            origin_acc /= origin_total
            origin_loss /= origin_total
    # print('test_loss',test_loss)
    # print('origin_loss',origin_loss,origin_total)
    args.logger.info("{:25}, Loss: {:8.4f}, Acc: {:5.3f}".format("Origin", origin_loss, origin_acc*100))

    model.eval()
    layer_sharpness_dict_loss = {} 
    layer_sharpness_dict_acc = {} 
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # print(name)
            # For WideResNet
            if "sub" in name:
                continue
            layer_sharpness_dict_loss[name] = 1e10
            layer_sharpness_dict_acc[name] = 1e10

    
    for layer_name, module in model.named_parameters():
        if "weight" in layer_name and layer_name[:-len(".weight")] in layer_sharpness_dict_loss.keys():
            # print(layer_name)
            cloned_model_min = deepcopy(model)
            cloned_model_max = deepcopy(model)
            # set requires_grad sign for each layer
            for name, param in cloned_model_min.named_parameters():
                if name == layer_name:
                    param.requires_grad = True
                    init_param_min = param.detach().clone()
                else:
                    param.requires_grad = False
            for name, param in cloned_model_max.named_parameters():
                if name == layer_name:
                    param.requires_grad = True
                    init_param_max = param.detach().clone()
                else:
                    param.requires_grad = False

            optimizer_min = torch.optim.SGD(cloned_model_min.parameters(), lr=sharp_lr_min)
            optimizer_max = torch.optim.SGD(cloned_model_max.parameters(), lr=sharp_lr_max)

            max_loss = origin_loss
            min_loss = origin_loss
            min_acc = origin_acc
            max_acc = origin_acc
    
            for iter_num in range(args.iter_num):
                # =========Min=========
                # Gradient ascent
                # ori_param_min = deepcopy(cloned_model_min.state_dict())
                # conv_init_param_min = cloned_model_min.state_dict()[layer_name][conv_kernal].detach().clone()
                for inputs, targets in trainloader:
                    inputs, targets = inputs.cuda(), targets.cuda()
                    # inputs.requires_grad = True # TODO
                    optimizer_min.zero_grad()
                    outputs = cloned_model_min(inputs)
                    loss = criterion(outputs, targets) 
                    loss.backward()
                    # print('min',loss.item())
                    optimizer_min.step()
                # ori_param_min[layer_name][conv_kernal]= cloned_model_min.state_dict()[layer_name][conv_kernal]
                # cloned_model_min.load_state_dict(ori_param_min)

                sd = cloned_model_min.state_dict()
                diff = sd[layer_name] - init_param_min
                times = torch.linalg.norm(diff)/torch.linalg.norm(init_param_min)
                if times > epsilon:
                    # print('min times > epsilon')
                    diff = diff / times * epsilon
                    sd[layer_name] = deepcopy(init_param_min + diff)
                    cloned_model_min.load_state_dict(sd)

                # =========Max=========
                for inputs, targets in trainloader:
                    inputs, targets = inputs.cuda(), targets.cuda()
                    # inputs.requires_grad = True # TODO
                    optimizer_max.zero_grad()
                    outputs = cloned_model_max(inputs)
                    loss = -1 * criterion(outputs, targets) 
                    # print('max',-loss.item())
                    loss.backward()
                    optimizer_max.step()

                sd = cloned_model_max.state_dict()
                diff = sd[layer_name] - init_param_max
                times = torch.linalg.norm(diff)/torch.linalg.norm(init_param_max)
                if times > epsilon:
                    # print('max times > epsilon')
                    diff = diff / times * epsilon
                    sd[layer_name] = deepcopy(init_param_max + diff)
                    cloned_model_max.load_state_dict(sd)

                with torch.no_grad():
                    total_min = 0
                    total_max = 0
                    total_loss_min = 0.0
                    total_loss_max = 0.0
                    correct_min = 0
                    correct_max = 0
                    for inputs, targets in trainloader:
                        inputs, targets = inputs.cuda(), targets.cuda()
                        outputs_min = cloned_model_min(inputs)
                        total_min += targets.shape[0]
                        total_loss_min += criterion(outputs_min, targets).item()  * targets.shape[0]
                        _, predicted_min = outputs_min.max(1)
                        correct_min += predicted_min.eq(targets).sum().item()

                        outputs_max = cloned_model_max(inputs)
                        total_max += targets.shape[0]
                        total_loss_max += criterion(outputs_max, targets).item() * targets.shape[0]
                        _, predicted_max = outputs_max.max(1)
                        correct_max += predicted_max.eq(targets).sum().item() 

                    total_loss_min /= total_min
                    correct_min /= total_min

                    total_loss_max /= total_max
                    correct_max /= total_max

                if total_loss_min < min_loss:
                    min_loss = total_loss_min
                    max_acc = correct_min

                if total_loss_max > max_loss:
                    max_loss = total_loss_max
                    min_acc = correct_max

            s_max = max_loss - origin_loss
            s_min = origin_loss - min_loss
            dropped_acc = (origin_acc - min_acc)*100
            rise_acc = (max_acc - origin_acc)*100

            sharp_loss = s_max if abs(s_max) > abs(s_min) else s_min
            sharp_acc = dropped_acc if abs(dropped_acc) > abs(rise_acc) else rise_acc
            layer_sharpness_dict_loss[layer_name[:-len(".weight")]] = sharp_loss # max_loss - origin_loss
            layer_sharpness_dict_acc[layer_name[:-len(".weight")]] = sharp_acc # max_loss - origin_loss
            args.logger.info("{:25} L: {:8.4f} Acc: {:5.4f} S: {:8.4f} SMax: {:8.4f} SMin: {:8.4f} Dropped Acc: {:5.3f} Rise Acc: {:5.3f}".format(layer_name, origin_loss,origin_acc*100,sharp_loss,s_max,s_min,dropped_acc,rise_acc))
            args.writer.add_scalar("Sharpness_Loss/{}".format(layer_name), sharp_loss,epoch)
            args.writer.add_scalar("Sharpness_Max_loss/{}".format(layer_name), s_max,epoch)
            args.writer.add_scalar("Sharpness_Min_loss/{}".format(layer_name), s_min,epoch)
            args.writer.add_scalar("Sharpness_Acc/{}".format(layer_name), sharp_acc,epoch)
            args.writer.add_scalar("Sharpness_Drop_Acc/{}".format(layer_name), dropped_acc,epoch)
            args.writer.add_scalar("Sharpness_Rise_Acc/{}".format(layer_name), rise_acc,epoch)

    sorted_layer_sharpness_loss = sorted(layer_sharpness_dict_loss.items(), key=lambda x:x[1])
    sorted_layer_sharpness_acc = sorted(layer_sharpness_dict_acc.items(), key=lambda x:x[1])
    avg_loss = []
    avg_acc = []
    for (k, v) in sorted_layer_sharpness_loss:
        args.logger.info("{:25}, Sharpness_loss: {:8.4f}".format(k, v))
        avg_loss.append(v)
    for (k, v) in sorted_layer_sharpness_acc:
        args.logger.info("{:25}, Sharpness_acc: {:8.4f}".format(k, v))
        avg_acc.append(v)

    args.writer.add_scalar("Sharpness_Loss_avg/model", np.mean(avg_loss),epoch)
    args.writer.add_scalar("Sharpness_Acc_avg/model", np.mean(avg_acc),epoch)
    return sorted_layer_sharpness_loss



def train(model, trainloader, optimizer, criterion, device, epoch, args):
    print("Epoch: %d" % epoch)
    # model = torch.nn.DataParallel(model)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    acc = 0
    if args.cutmix:
        cutmix = K.RandomCutMixV2(data_keys=["input", "class"])
    elif args.mixup:
        mixup = K.RandomMixUpV2(data_keys=["input", "class"])

    # for batch_idx, (inputs, targets, inputs_clean) in enumerate(trainloader):
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # bs = inputs.shape[0]
        # num_poisons = bs * args.ratio // 100
        # inputs[num_poisons:] = inputs_clean[num_poisons:]
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        if args.cutmix or args.mixup:
            if args.cutmix:
                inputs, targets = cutmix(inputs, targets)
                targets = targets.squeeze(0)
            else:
                inputs, targets = mixup(inputs, targets)
            outputs = model(inputs)
            loss = loss_mix(targets, outputs)
            loss.backward()
            optimizer.step()
            total += targets.size(0)
            acc += torch.sum(acc_mix(targets, outputs))
            progress_bar(batch_idx, len(trainloader))
            continue
        elif args.ueraser:
            if args.type == "tap" or args.type == "ar":
                U = UEraser_jpeg
            else:
                U = UEraser
            result_tensor = torch.empty((5, inputs.shape[0])).to(device)
            if epoch < args.repeat_epoch:
                for i in range(5):
                    images_tmp = U(inputs)
                    output_tmp = model(images_tmp)
                    loss_tmp = F.cross_entropy(
                        output_tmp, targets, reduction="none")
                    result_tensor[i] = loss_tmp
                outputs = output_tmp
                max_values, _ = torch.max(result_tensor, dim=0)
                loss = torch.mean(max_values)
            else:
                inputs = U(inputs)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(trainloader))
            continue
        elif args.at:
            outputs, loss = MadrysLoss(epsilon=args.at_eps, distance=args.at_type)(
                model, inputs, targets, optimizer
            )
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader))
    if args.cutmix or args.mixup:
        avg_train_acc = acc * 100.0 / total
    else:
        avg_train_acc = correct * 100.0 / total
    print(f"train_acc: {avg_train_acc:.4f}")
    return_train_loss = train_loss / total * targets.size(0)
    return avg_train_acc, return_train_loss


def test(model, testloader, criterion, device,logger):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * targets.shape[0]
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader))
    avg_test_acc = correct * 100.0 / total
    test_loss /= total
    logger.info("Test Acc: {:8.4f}".format(avg_test_acc))
    return avg_test_acc,test_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="c10")
    parser.add_argument(
        "--type",
        default="em",
        type=str,
        help="ar, dc, em, rem, hypo, tap, lsp, ntga, ops",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", default=0.1, type=float, help="model learning rate")
    parser.add_argument("--param_eps", default=0.05, type=float, help="model learning rate")
    parser.add_argument("--sharp_lr_min", default=0.1, type=float, help="sharp learning rate")
    parser.add_argument("--sharp_lr_max", default=0.1, type=float, help="sharp learning rate")
    parser.add_argument("--ratio", default=100,
                        type=int, help="poisoned ratio")

    parser.add_argument("--clean", default=False, action="store_true")
    parser.add_argument("--cutout", default=False, action="store_true")
    parser.add_argument("--cutmix", default=False, action="store_true")
    parser.add_argument("--mixup", default=False, action="store_true")
    parser.add_argument("--rnoise", default=False, action="store_true")
    parser.add_argument("--pure", default=False, action="store_true")
    parser.add_argument("--jpeg", default=False, action="store_true")
    parser.add_argument("--bdr", default=False, action="store_true")
    parser.add_argument("--gray", default=False, action="store_true")
    parser.add_argument("--gaussian", default=False, action="store_true")
    parser.add_argument("--nodefense", default=True, action="store_true")

    parser.add_argument("--ueraser", default=False, action="store_true")
    parser.add_argument(
        "--repeat_epoch",
        default=300,
        type=int,
        help="0 for -lite / 50 for UEraser / 300 for -max",
    )
    parser.add_argument("--ft", default=False, action="store_true")
    parser.add_argument("--on_test", default=False, action="store_true")
    parser.add_argument("--at", default=False, action="store_true")
    parser.add_argument("--at_eps", default=8 / 255,
                        type=float, help="noise budget")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument(
        "--at_type", default="L_inf", type=str, help="noise type, [L_inf, L_2]"
    )

    parser.add_argument(
        "--arch", default="r18", type=str, help="r18, r50, se18, mv2, de121, vit, cait"
    )
    parser.add_argument('--iter_num', default=10, type=int)
    parser.add_argument('--layer', default=None, type=str, help='Trainable layer')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_acc = 0
    start_epoch = 0

    d_idx = [
        args.cutout,
        args.cutmix,
        args.mixup,
        args.pure,
        args.jpeg,
        args.bdr,
        args.gray,
        args.gaussian,
        args.nodefense,
        args.ueraser,
        args.at,
        args.nodefense,
    ]
    d_name = [
        "cutout",
        "cutmix",
        "mixup",
        "pure",
        "jpeg",
        "bdr",
        "gray",
        "gaussian",
        "nodefense",
        "ueraser",
        "at",
        "nodefense",
    ]
    defense = d_name[d_idx.index(max(d_idx))]

    # ---------------------------UD---------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    proj_name = f"defense={defense}"
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    sharp_dataset_name = 'test' if args.on_test else 'train'
    finetune_name = 'finetune' if args.ft else 'scratch'
    suffix = '{}_lr={}_epochs={}_sharplr={}-{}_iter={}-eps={}'.format(proj_name,args.lr, args.epochs, args.sharp_lr_min, args.sharp_lr_max,args.iter_num,args.param_eps)
    model_save_dir = os.path.join('exp_iclr',args.dataset,args.arch,'clean' if args.clean else args.type,sharp_dataset_name,finetune_name,suffix)
    os.makedirs(model_save_dir,exist_ok=True)

    logger = create_logger(os.path.join(model_save_dir,'output.log'))
    logger.info(args)

    args.logger = logger
    os.makedirs(os.path.join(model_save_dir,'runs'),exist_ok=True)
    writer = SummaryWriter(os.path.join(model_save_dir,'runs'))
    args.writer = writer
    # ---------------------------UD---------------------------

    # Data
    logger.info('==> Preparing data and create dataloaders...')
    transform_train = aug_train(
        args.dataset, args.jpeg, args.gray, args.bdr, args.gaussian, args.cutout, args
    )
    if args.ft:
        if not args.clean:
            args.clean = True
            train_set, test_set = get_dataset(args, transform_train,train_ratio=0.5,front=True)
            args.clean = False
            ue_trainset, ue_test_set = get_dataset(args, transform_train,train_ratio=0.5,front=False)
        else:
            train_set, test_set = get_dataset(args, transform_train,train_ratio=0.5,front=True)
            ue_trainset, ue_test_set = get_dataset(args, transform_train,train_ratio=0.5,front=False)

        train_loader, test_loader = get_loader(args, train_set, test_set)
        ue_train_loader, _ = get_loader(args, ue_trainset, ue_test_set)
    else:
        train_set, test_set = get_dataset(args, transform_train)
        train_loader, test_loader = get_loader(args, train_set, test_set)

    if args.dataset == "c100":
        num_classes = 100
    else:
        num_classes = 10
    logger.info('==> Building model...')
    logger.info(args.arch)
    if args.arch == "r18":
        model = ResNet18(num_classes)
    elif args.arch == "r50":
        model = ResNet50(num_classes)
    elif args.arch == "se18":
        model = SENet18(num_classes)
    elif args.arch == "mv2":
        model = MobileNetV2(num_classes)
    elif args.arch == "de121":
        model = DenseNet121(num_classes)
    elif args.arch == "vit":
        model = Vit_cifar(num_classes)
    elif args.arch == "cait":
        model = Cait_cifar(num_classes)

    model.to(device)
    logger.info('==> Building optimizer and learning rate scheduler...')
    criterion = nn.CrossEntropyLoss()
    if args.arch == "vit" or args.arch == "cait":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
        )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 200, 300], gamma=0.5, last_epoch=-1, verbose=False
    )

    directory = "log"
    path = os.path.join(directory, args.dataset)
    dir = os.path.join(path, 'clean' if args.clean else args.type)

    if not os.path.exists(dir):
        os.makedirs(dir)
    log_dir = os.path.join(
        dir,
        f"defense={defense}-repeat={args.repeat_epoch}-poison_ratio={args.ratio}-arch={args.arch}.pth",
    )
    # test_acc,test_loss = test(model, test_loader, criterion, device,logger)
    # layer_sharpness(args, deepcopy(model),criterion,trainloader=test_loader,epsilon=0.1,epoch=0)
    if args.ft:
        if args.arch == "vit" or args.arch == "cait":
            ft_optimizer = optim.Adam(model.parameters(), lr=args.lr)
        else:
            ft_optimizer = optim.SGD(
                model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
            )
    train_history, eval_history = [], []
    for epoch in range(start_epoch, start_epoch + args.epochs):
        if args.ft and epoch >= 5:
            train_acc,train_loss = train(
                model, ue_train_loader, ft_optimizer, criterion, device, epoch, args
            )
        else:
            train_acc,train_loss = train(
                model, train_loader, optimizer, criterion, device, epoch, args
            )            
        test_acc,test_loss = test(model, test_loader, criterion, device,logger)

        sharpness_loader = test_loader if args.on_test else train_loader
        sharpness_loss = test_loss if args.on_test else train_loss
        sharpness_acc = test_acc if args.on_test else train_acc
        writer.add_scalar("Train/train_acc", train_acc,epoch)
        writer.add_scalar("Train/train_loss", train_loss,epoch)
        writer.add_scalar("Test/test_acc", test_acc,epoch)
        writer.add_scalar("Test/test_loss", test_loss,epoch)
        logger.info("epoch: {}".format(epoch))
        logger.info("train Acc: {:8.4f}, test Acc: {:8.3f}".format(train_acc, test_acc))
        layer_sharpness(args, deepcopy(model), criterion,trainloader=sharpness_loader,epsilon=args.param_eps,test_acc=sharpness_acc,test_loss=sharpness_loss,epoch=epoch,sharp_lr_min=args.sharp_lr_min,sharp_lr_max=args.sharp_lr_max)
        
        train_history.append(train_acc)
        eval_history.append(test_acc)
        scheduler.step()

    print(" Saving...")
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_history": train_history,
        "eval_history": eval_history,
    }
    logger.info("Train Acc: {:8.4f}, Test Acc: {:8.3f}".format(train_acc, test_acc))
    torch.save(state, log_dir)


if __name__ == "__main__":
    main()
