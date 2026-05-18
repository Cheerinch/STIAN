import torch

cropped_video_folder = r"D:\subgraph\Datastet2\allavi"
skeleton_folder = r"D:\subgraph\Datastet2\allsk"

output_model_path = "fusion_bias_transformer_17joints.pth"

num_classes = 2
batch_size = 2
epochs = 100
num_frames = 24
img_size = 224
accumulation_steps = 2

lr = 5e-5
weight_decay = 1e-3

use_amp = False
n_splits = 5
random_state = 42

num_workers = 2

rgb_dim = 512
skel_dim = 32
hidden_dim = 256
num_heads = 8
num_layers = 2

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)