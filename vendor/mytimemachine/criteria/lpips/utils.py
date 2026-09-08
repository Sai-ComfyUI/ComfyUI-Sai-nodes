from collections import OrderedDict

import torch

from configs.paths_config import model_paths


def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def get_state_dict(net_type: str = "alex", version: str = "0.1"):
    local_path = model_paths.get(f"lpips_{net_type}")
    if local_path:
        old_state_dict = torch.load(local_path, map_location="cpu", weights_only=True)
    else:
        # Upstream fallback for environments without a managed model path.
        url = (
            "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/"
            f"master/lpips/weights/v{version}/{net_type}.pth"
        )
        old_state_dict = torch.hub.load_state_dict_from_url(
            url,
            progress=True,
            map_location=None if torch.cuda.is_available() else torch.device("cpu"),
        )

    new_state_dict = OrderedDict()
    for key, value in old_state_dict.items():
        new_key = key.replace("lin", "").replace("model.", "")
        new_state_dict[new_key] = value
    return new_state_dict
