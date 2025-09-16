from data.facedataset import FaceDataset, BalanceFaceDataset
from torchvision import transforms
from torch.utils.data import DataLoader

def protocol_decoder(protocol):
    MAP = {
        'C':'CASIA',
        'I':'Idiap',
        'M':'MSU',
        'O':'OULU',
        'A':'CelebA',
        'W':'SiW',
        's':'Surf',
        'c':'CeFA',
        'w':'WMCA',
    }

    train_protocols, test_protocols = protocol.split('_to_')
    train_protocols = train_protocols.split('_')
    test_protocols = test_protocols.split('_')
    return [MAP[train_protocol] for train_protocol in train_protocols], [MAP[test_protocol] for test_protocol in test_protocols]  

def get_transform(backbone):
    if 'resnet' in backbone:
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-180, 180)),
            transforms.RandomResizedCrop(256, scale=(0.2, 1.0), ratio=(1., 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_transform = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.9, 0.9), ratio=(1., 1.)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    else:
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-180, 180)),
            transforms.RandomResizedCrop(224, scale=(0.9, 0.9), ratio=(1., 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.9, 0.9), ratio=(1., 1.)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return train_transform, test_transform

def build_datasets(args):
    train_protocol_names, test_protocol_names = protocol_decoder(args.protocol)
    train_transform, test_transform = get_transform(args.backbone)
    args.num_domain = len(train_protocol_names)

    train_dataset = BalanceFaceDataset(args.data_root, train_protocol_names, 'train', train_transform, args.max_iter * args.batch_size, not args.silence)
    test_dataset = FaceDataset(args.data_root, test_protocol_names, 'test', test_transform, not args.silence)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader