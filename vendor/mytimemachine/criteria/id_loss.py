import torch
from torch import nn
from configs.paths_config import model_paths
from models.encoders.model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_paths['ir_se50']))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # todo: https://github.com/google/mystyle/blob/main/utils/id_utils.py#L43
        for module in [self.facenet, self.face_pool]:
            for param in module.parameters():
                param.requires_grad = False

    def extract_feats(self, x):
        if x.shape[2] != 256:
            # https://github.com/williamyang1991/GP-UNIT/blob/main/model/arcface/id_loss.py
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    # def compute_similarities(self, img1, img2):
    #     '''
    #     img: tensor of shape (n, 3, 256, 256) normalized in the range [-1, 1]
    #     return: tensor of shape (n, ) with cosine similarities
    #     '''
    #     ds = []
    #     for img_1, img_2 in zip(img1, img2):
    #         feat1 = self.extract_feats(img_1.unsqueeze(0))
    #         feat2 = self.extract_feats(img_2.unsqueeze(0))
    #         d = torch.cosine_similarity(feat1, feat2)
    #         ds.append(d)
    #     return torch.stack(ds)

    def forward(self, y_hat, y, x, label=None, weights=None):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        total_loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            diff_input = y_hat_feats[i].dot(x_feats[i])
            diff_views = y_feats[i].dot(x_feats[i])

            if label is None:
                id_logs.append({'diff_target': float(diff_target),
                                'diff_input': float(diff_input),
                                'diff_views': float(diff_views)})
            else:
                id_logs.append({f'diff_target_{label}': float(diff_target),
                                f'diff_input_{label}': float(diff_input),
                                f'diff_views_{label}': float(diff_views)})

            loss = 1 - diff_target
            if weights is not None:
                loss = weights[i] * loss

            total_loss += loss
            id_diff = float(diff_target) - float(diff_views)
            sim_improvement += id_diff
            count += 1

        return total_loss / count, sim_improvement / count, id_logs
