from models import networks
import torch

def backbone_map(backbone):
    if 'resnet' in backbone:
        return 'resnet'
    elif 'clip' in backbone:
        return 'clip_encoder'
    elif 'safas' in backbone:
        return 'resnet18'

def build_model(args):
    return getattr(networks, backbone_map(args.backbone))(args)

def build_optimizer(args, net):
    print()
    if 'clip' in args.backbone:
        return torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        return torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd)