import torch.nn as nn
import torch.nn.functional as F
import torchvision, torch
import numpy as np

from models.clip.clip import load, tokenize
from collections import OrderedDict
from scipy import optimize
from utils import centroid_calculate_function

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        if n > 0:
            self.sum += val * n
            self.cnt += n
            self.avg = self.sum / self.cnt

class resnet(nn.Module):
    # setup
    def __init__(self, args):
        super(resnet, self).__init__()
        self.backbone = args.backbone
        self.verbose = not args.silence

        self.batch_size = args.batch_size
        self.num_domain = args.num_domain
        self.num_class = 2
        self.build()

    # build the network and loss
    def build(self):
        # encoder
        model = getattr(torchvision.models, f'{self.backbone}')(weights=f'ResNet{self.backbone.split("resnet")[1]}_Weights.DEFAULT')
        module_sequence = []
        for name, module in model.named_modules():
            if name != '' and '.' not in name:
                if 'fc' in name:
                    self.last_dim = module.weight.size(1)
                    break
                module_sequence.append((name, module))

        self.model = nn.Sequential(OrderedDict(module_sequence))

        in_dim, mlp_dim, out_dim = self.last_dim, 4096, 256
        self.image_mlp = nn.Sequential(
            OrderedDict(
                [
                    ("layer1", nn.Linear(in_dim, mlp_dim)),
                    ("bn1", nn.BatchNorm1d(mlp_dim)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("layer2", nn.Linear(mlp_dim, mlp_dim)),
                    ("bn2", nn.BatchNorm1d(mlp_dim)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("layer3", nn.Linear(mlp_dim, out_dim)),
                ]
            ))

        # classifiers
        self.classifier = nn.Linear(256, self.num_class, bias=False)

        # criterion
        self.celoss = nn.CrossEntropyLoss()

        # loss_info
        self.variance_loss  = AverageMeter()
        self.class_loss     = AverageMeter()
        self.info_nce_loss  = AverageMeter()
        self.class_sim_loss = AverageMeter()
        self.regular_loss   = AverageMeter()
        self.total_loss     = AverageMeter()

        # inform backbone and number of parameters
        if self.verbose:
            encoder_param = sum(p.numel() for p in self.model.parameters())
            classifier_param = sum(p.numel() for p in self.classifier.parameters())
            total_param = encoder_param + classifier_param

            print('------------------------------------------------------')
            print(f'build_backbone   \t- {self.backbone}')
            print('------------------------------------------------------')
            print(f'total_param      \t: {total_param}')
            print(f'encoder_param    \t: {encoder_param}')
            print(f'classifier       \t: {classifier_param}')

    def loss_update(self, variance_loss, class_loss, info_nce_loss, class_sim_loss, regular_loss, total_loss):
        self.variance_loss.update(variance_loss.item())
        self.class_loss.update(class_loss.item())
        self.info_nce_loss.update(info_nce_loss.item())
        self.class_sim_loss.update(class_sim_loss.item())
        self.regular_loss.update(regular_loss.item())
        self.total_loss.update(total_loss.item())

    def loss_reset(self):
        variance_loss = self.variance_loss.avg
        class_loss    = self.class_loss.avg
        info_nce_loss   = self.info_nce_loss.avg
        class_sim_loss  = self.class_sim_loss.avg
        regular_loss = self.regular_loss.avg
        total_loss    = self.total_loss.avg

        self.variance_loss.reset()
        self.class_loss.reset()
        self.info_nce_loss.reset()
        self.total_loss.reset()

        return {
            'variance_loss' :variance_loss,
            'class_loss'    :class_loss,
            'info_nce_loss' :info_nce_loss,
            'class_sim_loss':class_sim_loss,
            'regular_loss'  :regular_loss,
            'total_loss'    :total_loss,
        }
    
    def compute_class_loss(self, features, labels):
        similarity = self.classifier(self.image_mlp(features))
        mask = torch.cat([labels.eq(0).unsqueeze(1),labels.eq(1).unsqueeze(1)],dim=1)
        mask = torch.min(torch.tensor(0.86).repeat(features.size(0)).cuda(), similarity.clone().detach()[mask]).bool()
        # mask = torch.min(torch.tensor(0.7).repeat(features.size(0)).cuda(), similarity.clone().detach()[mask]).bool()
        
        if (~mask).float().sum() > 0:
            class_loss = self.celoss(similarity[mask], labels[mask]) * labels[mask].size(0) - self.celoss(similarity[~mask], labels[~mask]) * labels[~mask].size(0) * 0.01
            class_loss = class_loss / labels.size(0)
            return class_loss
        
        else:
            class_loss = self.celoss(similarity[mask], labels[mask])
            return class_loss

    # close to same video image with different augmentation methods
    def compute_info_nce_loss(self, features, labels, domains):
        class_feature = self.classifier.weight.clone().detach()
        class_feature = 0.86 * torch.cat([class_feature[a].unsqueeze(0) for a in labels],dim=0)
        # class_feature = 0.7 * torch.cat([class_feature[a].unsqueeze(0) for a in labels],dim=0)

        features = features - class_feature
        features = features / features.norm(dim=-1, keepdim=True)
        similarity_matrix = torch.matmul(features, features.T)

        index = (domains.unsqueeze(0) == domains.unsqueeze(1))
        mask   = ~torch.eye(labels.size(0)).bool()

        similarity_matrix = similarity_matrix[mask].view(domains.size(0),-1) / 0.1
        index   = index[mask].view(domains.size(0),-1).bool()

        positive_features = similarity_matrix[index].view(48, -1).sum(dim=1)
        negative_features = similarity_matrix[~index].view(48, -1).sum(dim=1)

        info_nce_loss = (positive_features / (positive_features + negative_features)).mean()
        return info_nce_loss
    
    def compute_loss(self, images, labels, domains):
        # concatenate features
        features = self.model(images).squeeze()
        features = features / features.norm(dim=-1, keepdim=True)
        
        variance_loss  = torch.max(torch.zeros(features.size(0)).cuda(), 0.04 - features.std(dim=1)).mean()
        class_loss     = self.compute_class_loss(features, labels)
        info_nce_loss  = 0.3 * self.compute_info_nce_loss(features, labels, domains)
        class_sim_loss = 0.3 * torch.matmul(self.classifier.weight[0], self.classifier.weight[1])
        regular_loss   = self.l1loss(self.classifier.weight.norm(dim=1), 1*torch.ones(2).cuda())
        total_loss     = class_loss + info_nce_loss + class_sim_loss + regular_loss
        
        self.loss_update(variance_loss, class_loss, info_nce_loss, class_sim_loss, regular_loss, total_loss)

        return total_loss

    def forward(self, input):
        features = F.normalize(self.model(input).squeeze())
        return self.classifier(features) + 0.5
    
class clip_encoder(nn.Module):
    def __init__(self, args):
        super(clip_encoder, self).__init__()
        # args
        self.params = args.params
        self.temperature = args.temperature
        self.alpha = np.log(args.num_domain)/2
        self.beta = args.beta
        self.gs = args.gs

        # load the model
        self.model, _ = load("ViT-B/16", "cuda:0")

        # define the mlp
        in_dim, mlp_dim, out_dim = 512,4096,256
        self.image_mlp = self._build_mlp(
            in_dim=in_dim, mlp_dim=mlp_dim, out_dim=out_dim
        )

        # define classifier
        self.classifier  = nn.Linear(256, 2, bias=True)

        # define loss
        self.define_losses()

        # define spoof and real templates
        spoof_templates = [
            "This is an example of a spoof face",
            "This is an example of an attack face",
            "This is not a real face",
            "This is how a spoof face looks like",
            "a photo of a spoof face",
            "a printout shown to be a spoof face",
        ]

        real_templates = [
            "This is an example of a real face",
            "This is a bonafide face",
            "This is a real face",
            "This is how a real face looks like",
            "a photo of a real face",
            "This is not a spoof face",
        ]

        # tokenize the spoof and real templates
        self.spoof_texts = tokenize(spoof_templates).cuda(non_blocking=True)  # tokenize
        self.real_texts = tokenize(real_templates).cuda(non_blocking=True)  # tokenize

    def _build_mlp(self, in_dim, mlp_dim, out_dim):
        return nn.Sequential(
            OrderedDict(
                [
                    ("layer1", nn.Linear(in_dim, mlp_dim)),
                    ("bn1", nn.BatchNorm1d(mlp_dim)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("layer2", nn.Linear(mlp_dim, mlp_dim)),
                    ("bn2", nn.BatchNorm1d(mlp_dim)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("layer3", nn.Linear(mlp_dim, out_dim)),
                    ("bn3", nn.BatchNorm1d(out_dim)),
                    ("relu3", nn.ReLU(inplace=True)),
                ]
            )
        )
    
    def define_losses(self):
        self.IT_similarity_loss = AverageMeter()
        self.II_similarity_loss = AverageMeter()
        self.class_loss = AverageMeter()
        self.domain_loss = AverageMeter()
        self.total_loss = AverageMeter()

        self.sim_features = {
            'sl':AverageMeter(),
            'cl':AverageMeter()
        }

    def loss_reset(self):
        IT_similarity_loss = self.IT_similarity_loss.avg
        II_similarity_loss = self.II_similarity_loss.avg
        class_loss = self.class_loss.avg
        domain_loss = self.domain_loss.avg
        total_loss = self.total_loss.avg
        sim_features_sl = self.sim_features['sl'].avg
        sim_features_cl = self.sim_features['cl'].avg

        self.IT_similarity_loss.reset()
        self.II_similarity_loss.reset()
        self.class_loss.reset()
        self.domain_loss.reset()
        self.total_loss.reset()
        self.sim_features['sl'].reset()
        self.sim_features['cl'].reset()
        info = {
            'loss':'IT {:.4f} II {:.4f} class {:.4f} domain {:.4f} total {:.4f} | sl : {:.4f} cl : {:.4f}'.format(IT_similarity_loss, II_similarity_loss, class_loss, domain_loss, total_loss, sim_features_sl, sim_features_cl),
        }

        return info
    
    def compute_domain_similarity(self, text_features):
        self.sim_features['sl'].update((text_features[0]@text_features[1].t()).item())
        self.sim_features['cl'].update((self.classifier.weight[0]@self.classifier.weight[1].t()).item())
    
    def scaling_estimator(self, input):
        return self.beta/(1+np.exp(-input/self.alpha)) - self.beta/2 + 1
    
    def group_wise_scaling(self, logits, labels):
        sizes  = [l.size(0) for l in labels]
        losses = [torch.nn.functional.cross_entropy(logit, label) * size / sum(sizes)
                 for logit, label, size in zip(logits, labels, sizes)]
        n_losses = np.array([loss.item() for loss in losses])
        norm_losses = (n_losses-n_losses.mean())/n_losses.std()
        return [l*self.scaling_estimator(n) for l, n in zip(losses, norm_losses)], losses
    
    def gram_schmidt_process(self, vectors, basis=None):
        if basis == None:
            orthogonal_vectors = []
            for v in vectors:
                # Copy the current vector
                ortho_v = v.clone()

                # Subtract the projection of v onto each of the orthogonal vectors
                for u in orthogonal_vectors:
                    proj = torch.dot(v, u) / torch.dot(u, u) * u
                    ortho_v -= proj

                # Add non-zero orthogonal vectors to the basis
                if torch.norm(ortho_v) > 1e-6:
                    ortho_v = ortho_v / ortho_v.norm()
                    orthogonal_vectors.append(ortho_v)
            return orthogonal_vectors
        else:
            for b in basis:
                try:
                    invariant_features += torch.inner(vectors, b).unsqueeze(1) * b
                except:
                    invariant_features = torch.inner(vectors, b).unsqueeze(1) * b

            specific_features = vectors - invariant_features.detach().clone().cuda()
            return specific_features

    def compute_loss(self, images, labels, domains):
        # encode the spoof and real templates with the text encoder
        all_spoof_class_embeddings = self.model.encode_text(self.spoof_texts)
        all_real_class_embeddings = self.model.encode_text(self.real_texts)

        # ------------------- Image-Text Ebedding Space Branch -------------------
        # Ensemble of text features
        # embed with text encoder
        spoof_class_embeddings = all_spoof_class_embeddings.mean(dim=0)
        real_class_embeddings = all_real_class_embeddings.mean(dim=0)

        # stack the text features of liveness and spoofness.
        ensemble_weights = [spoof_class_embeddings, real_class_embeddings]
        text_features = torch.stack(ensemble_weights, dim=0).cuda()

        # get the image features and features
        image_features = self.model.encode_image(images)    # image-features

        # normalized features
        norm_image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        norm_text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = torch.inner(norm_image_features, norm_text_features) * logit_scale

        group_label = labels + 2 * domains # A1 0 L1 1 A2 2 L2 3 A3 4 L3 5
        group_size = group_label.unique().size(0)

        if self.gs:
            IT_similarity_loss, _ = self.group_wise_scaling(
                [logits_per_image[group_label.eq(group)] for group in range(group_size)],
                [labels[group_label.eq(group)] for group in range(group_size)]
                )
        else:
            IT_similarity_loss = [torch.nn.functional.cross_entropy(logits_per_image, labels)]

        # IT_similarity_loss = self.params[0] * sum(IT_similarity_loss)
        IT_similarity_loss = sum(IT_similarity_loss)

        self.IT_similarity_loss.update(IT_similarity_loss.item())

        self.compute_domain_similarity(norm_text_features)

        # domain similarity
        size = images.size(0)
        mask = ~torch.eye(size).bool()

        basis = self.gram_schmidt_process(text_features, None)
        specific_features = self.gram_schmidt_process(image_features, basis)

        domain_similarity = torch.inner(specific_features, specific_features) / self.temperature
        domain_sim_mask = (domains.unsqueeze(0) == domains.unsqueeze(1)).float()

        domain_similarity = domain_similarity[mask].view(size,-1)
        domain_sim_mask = domain_sim_mask[mask].view(size,-1)
        domain_loss = -torch.mean(torch.div(domain_sim_mask * torch.nn.functional.log_softmax(domain_similarity,dim=1),
                                        domain_sim_mask.sum(dim=1).unsqueeze(1)))
    
        domain_loss = self.params[1] * domain_loss
        # domain_loss = self.param1 * domain_loss
        self.domain_loss.update(domain_loss.item())

        # ------------------- Image Embedding Space Branch -------------------
        # cosine similarity as same image different view
        embedding_features = self.image_mlp(image_features) # features
        logits = self.classifier(embedding_features)
        norm_embedding_features = embedding_features/embedding_features.norm(dim=-1,keepdim=True)
        similarity = torch.inner(norm_embedding_features, norm_embedding_features) / 0.8
        index = torch.arange(int(size/2)).repeat(2)
        index = (index.unsqueeze(0) == index.unsqueeze(1))

        similarity = similarity[mask].view(size,-1)
        index      = index[mask].view(size,-1).bool()

        similarity = torch.cat([similarity[index].view(size,-1),similarity[~index].view(size,-1)],dim=1) 
        sim_label  = torch.zeros(size).long().cuda()

        # II_similarity_loss = self.param2 * torch.nn.functional.cross_entropy(similarity, sim_label) 
        II_similarity_loss = self.params[2] * torch.nn.functional.cross_entropy(similarity, sim_label) 
    
        self.II_similarity_loss.update(II_similarity_loss.item())

        logits = self.classifier(embedding_features)

        class_loss = torch.nn.functional.cross_entropy(logits,labels)
        
        self.class_loss.update(class_loss.item())

        total_loss = IT_similarity_loss + II_similarity_loss + domain_loss + class_loss 
        self.total_loss.update(total_loss.item())
        return total_loss

    def forward(self, input):
        # encode the spoof and real templates with the text encoder
        all_spoof_class_embeddings = self.model.encode_text(self.spoof_texts)
        all_real_class_embeddings = self.model.encode_text(self.real_texts)

        # ------------------- Image-Text similarity branch -------------------
        # Ensemble of text features
        # embed with text encoder
        spoof_class_embeddings = all_spoof_class_embeddings.mean(dim=0)
        real_class_embeddings = all_real_class_embeddings.mean(dim=0)

        # stack the embeddings for image-text similarity
        ensemble_weights = [spoof_class_embeddings, real_class_embeddings]
        text_features = torch.stack(ensemble_weights, dim=0).cuda()

        # get the image features
        image_features = self.model.encode_image(input)
        embedding_features = self.image_mlp(image_features)

        # # normalized features
        norm_image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        norm_text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = norm_image_features @ norm_text_features.t() * logit_scale

        return logits_per_image, self.classifier(embedding_features)