import torch
import torch.nn as nn

# class Blender(nn.Module):

#     def __init__(self, num_styles=18):
#         super(Blender, self).__init__()
#         # simple blending with theta as learnable parameters
#         # x is of shape (batch_size, num_styles, 512)
#         # initialize theta to be 0.5
#         self.theta = nn.Parameter(torch.full((1, num_styles, 1), 0.5))

#     def forward(self, local_styles, global_styles, target_ages):
#         # sanity check
#         # print('sanity check in blender')
#         # print(self.theta)
#         return local_styles * self.theta + global_styles * (1 - self.theta)


# class Blender(nn.Module):

#     def __init__(self, num_styles=18):
#         super(Blender, self).__init__()
#         # time-conditioned blending with theta as learnable parameters
#         # mlp(target_ages) -> theta, target_ages is of shape (b), theta is of shape (b, num_styles, 1)
#         self.mlp = nn.Sequential(
#             nn.Linear(1, 16),
#             nn.ReLU(),
#             nn.Linear(16, 16),
#             nn.ReLU(),
#             nn.Linear(16, num_styles)
#         )

#     def forward(self, local_styles, global_styles, target_ages):
#         target_ages = target_ages.unsqueeze(-1) # (b, 1)
#         theta = self.mlp(target_ages).unsqueeze(-1) # (b, num_styles, 1)
#         # print('sanity check:')
#         # print(theta)
#         # final style = local_style * theta + global_style * (1 - theta)
#         return local_styles * theta + global_styles * (1 - theta)


# class Blender(nn.Module):
#     # brute force mlp blending
#     def __init__(self, num_styles=18):
#         super(Blender, self).__init__()
#         # naive brute-force blending with theta as learnable parameters
#         self.mlp_time = nn.Sequential(
#             nn.Linear(1, 16),
#             nn.ReLU(),
#             nn.Linear(16, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1)
#         )
#         # global and local styles are of shape (b, 11, 18, 512)
#         self.mlp_global = nn.Sequential(
#             nn.Linear(11 * 18 * 512, 16),
#             nn.ReLU(),
#             nn.Linear(16, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1)
#         )
#         self.mlp_local = nn.Sequential(
#             nn.Linear(11 * 18 * 512, 16),
#             nn.ReLU(),
#             nn.Linear(16, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1)
#         )
#         # output of mlp_time, mlp_global, mlp_local are all of shape (b, 1)
#         self.mlp_final = nn.Sequential(
#             nn.Linear(3, 16),
#             nn.ReLU(),
#             nn.Linear(16, 16),
#             nn.ReLU(),
#             nn.Linear(16, 18)
#         )

#     def forward(self, local_style, global_style, local_styles, global_styles, target_ages):
#         # local_style: [b, 18, 512]
#         # local_styles: [b, 11, 18, 512]
#         # global_styles: [b, 11, 18, 512]

#         target_ages = target_ages.unsqueeze(-1) # (b, 1)
#         theta_time = self.mlp_time(target_ages) # (b, 1)

#         theta_global = self.mlp_global(global_styles.flatten(start_dim=1)) # (b, 1)
#         theta_local = self.mlp_local(local_styles.flatten(start_dim=1)) # (b, 1)
#         theta = self.mlp_final(torch.cat([theta_time, theta_global, theta_local], dim=1)).unsqueeze(-1) # (b, 18, 1)

#         return local_style * theta + global_style * (1 - theta)


# class Blender(gpytorch.models.ExactGP):

#     def __init__(self, train_x, train_y, likelihood):
#         super(Blender, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# class Blender(nn.Module):

#     def __init__(self, num_styles=18):
#         super(Blender, self).__init__()
#         # directly predict delta_W
#         self.delta_w = nn.Parameter(torch.zeros((1, num_styles, 512)))

#     def forward(self, local_styles, global_styles, target_ages):
#         return global_styles + self.delta_w
    


# class Blender(nn.Module):

#     def __init__(self, num_styles=18):
#         super(Blender, self).__init__()
#         for i in range(num_styles):
#             setattr(self, f'fc_{i}', nn.Linear(512, 512))

#     def forward(self, local_styles, global_styles, target_ages):
#         updated_styles = global_styles.clone()
#         for i in range(global_styles.shape[1]):
#             updated_styles[:, i] = getattr(self, f'fc_{i}')(global_styles[:, i])
#         return updated_styles



class Blender(nn.Module):
    
    def __init__(self, num_styles=18):
        super(Blender, self).__init__()
        for i in range(num_styles):
            setattr(self, f'bottleneck_{i}', nn.Sequential(
                nn.Linear(512, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
            ))
        self.global_mlp = nn.Sequential(
            nn.Linear(18 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.age_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )
        for i in range(num_styles):
            setattr(self, f'fc_{i}', nn.Sequential(
                nn.Linear(128 + 16 + 512, 512), # global_features + age_features + local_styles
                nn.ReLU(),
                nn.Linear(512, 512),
            ))
            
    def forward(self, local_styles, global_styles, target_ages, global_step=1e4):
        # global_styles are of shape (b, 18, 512)
        global_features = torch.cat([getattr(self, f'bottleneck_{i}')(global_styles[:, i]) for i in range(global_styles.shape[1])], dim=1) # (b, 18 * 32)
        global_features = self.global_mlp(global_features) # (b, 128)

        # ---------------------------- add age information --------------------------- #
        age_features = self.age_mlp(target_ages.unsqueeze(-1)) # (b, 16)
        global_features = torch.cat([global_features, age_features], dim=1)

        # -------------------------- delta_w + pretrained w -------------------------- #
        updated_styles = torch.zeros_like(global_styles)
        for i in range(global_styles.shape[1]):
            updated_styles[:, i] = getattr(self, f'fc_{i}')(torch.cat([global_features, global_styles[:, i]], dim=1)) + global_styles[:, i]

        # ------------------------------- delta_w as w ------------------------------- #
        # updated_styles = torch.zeros_like(global_styles)
        # for i in range(global_styles.shape[1]):
        #     updated_styles[:, i] = getattr(self, f'fc_{i}')(torch.cat([global_features, global_styles[:, i]], dim=1))

        # --------------------- blended delta_w and pretrained w --------------------- #
        # give global_styles alpha weight, delta_w 1-alpha weight
        # alpha is 1 at the beginning, and exponentially decrease to 0 at the end when global_step = 10000
        # updated_styles = torch.zeros_like(global_styles)
        # for i in range(global_styles.shape[1]):
        #     if global_step > 5000:
        #         alpha = 0
        #     else:
        #         alpha = 1 - global_step / 5000
        #     updated_styles[:, i] = alpha * global_styles[:, i] + (1 - alpha) * getattr(self, f'fc_{i}')(torch.cat([global_features, global_styles[:, i]], dim=1))

        # ------------------------ update only certain styles ------------------------ #
        # updated_styles = global_styles.clone().detach().requires_grad_(True)

        # todo: uncomment this line
        # i = 2
        # updated_style_i = getattr(self, f'fc_{i}')(torch.cat([global_features, global_styles[:, i]], dim=1)) + global_styles[:, i]
        # updated_style_i = getattr(self, f'fc_{i}')(torch.cat([global_features, global_styles[:, i]], dim=1))
        # updated_styles = torch.cat([updated_styles[:, :i], updated_style_i.unsqueeze(1), updated_styles[:, i+1:]], dim=1)

        # update only style 2 to 12
        # styles = []
        # for i in range(18):
        #     if 2 <= i <= 12:
        #         # directly use delta_w as final w (style)
        #         style_i = getattr(self, f'fc_{i}')(torch.cat([global_features, global_styles[:, i]], dim=1))
        #     else:
        #         style_i = global_styles[:, i]
        #     styles.append(style_i)
        # updated_styles = torch.stack(styles, dim=1)

        return updated_styles