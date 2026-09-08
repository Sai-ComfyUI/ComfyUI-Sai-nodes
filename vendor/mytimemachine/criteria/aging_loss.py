import torch
from torch import nn
import torch.nn.functional as F

from configs.paths_config import model_paths
from models.dex_vgg import VGG
import os

class AgingLoss(nn.Module):

    def __init__(self, opts):
        super(AgingLoss, self).__init__()
        self.age_net = VGG()
        ckpt = torch.load(model_paths['age_predictor'], map_location="cpu")['state_dict']
        ckpt = {k.replace('-', '_'): v for k, v in ckpt.items()}
        self.age_net.load_state_dict(ckpt)
        self.age_net.cuda()
        self.age_net.eval()
        self.min_age = 0
        self.max_age = 100
        self.opts = opts

    def __get_predicted_age(self, age_pb):
        predict_age_pb = F.softmax(age_pb, dim=1)
        predict_age = torch.zeros(age_pb.size(0)).type_as(predict_age_pb)
        for i in range(age_pb.size(0)):
            for j in range(age_pb.size(1)):
                predict_age[i] += j * predict_age_pb[i][j]
        return predict_age

    def extract_ages(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear')
        predict_age_pb = self.age_net(x)['fc8']
        predicted_age = self.__get_predicted_age(predict_age_pb)
        return predicted_age
    
    def extract_ages_gt(self, x_path):
        '''
        Extracts the age from the ground truth image based on the file name
        '''
        def extract_age_gt(x_path):
            delimiter = ['_', '.', ' ']
            img_name = os.path.basename(x_path)
            for d in delimiter:
                img_name = img_name.replace(d, '/')
            age = int(img_name.split('/')[0])
            return age
        ages = []
        # determine single image path or batch of images
        if isinstance(x_path, str):
            age = extract_age_gt(x_path)
            ages.append(age)
        else:
            for path in x_path:
                age = extract_age_gt(path)
                ages.append(age)
        return torch.tensor(ages)
        # x = F.interpolate(x, size=(224, 224), mode='bilinear')
        # predict_age_pb = self.age_net(x)['fc8']
        # predicted_age = self.__get_predicted_age(predict_age_pb)
        # return predicted_age

    def forward(self, y_hat, y, target_ages, id_logs, label=None):
        n_samples = y.shape[0]

        if id_logs is None:
            id_logs = []

        input_ages = self.extract_ages(y) / 100.
        output_ages = self.extract_ages(y_hat) / 100.

        for i in range(n_samples):
            # if id logs for the same exists, update the dictionary
            if len(id_logs) > i:
                id_logs[i].update({f'input_age_{label}': float(input_ages[i]) * 100,
                                   f'output_age_{label}': float(output_ages[i]) * 100,
                                   f'target_age_{label}': float(target_ages[i]) * 100})
            # otherwise, create a new entry for the sample
            else:
                # print('i:', i)
                # print('input_ages:', input_ages)
                # print('output_ages:', output_ages)
                id_logs.append({f'input_age_{label}': float(input_ages[i]) * 100,
                                f'output_age_{label}': float(output_ages[i]) * 100,
                                f'target_age_{label}': float(target_ages[i]) * 100})

        loss = F.mse_loss(output_ages, target_ages)
        return loss, id_logs
