import numpy as np
import torch
from torch import nn
from torch.nn import Sequential
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import math # Import math để làm tròn

class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # --- Phân bổ dimension và đảm bảo là số nguyên ---
        # Làm tròn xuống cho image_features_dim để dễ kiểm soát
        image_features_dim = math.floor(features_dim * 0.7)
        if image_features_dim <= 0 : image_features_dim = 1 # Đảm bảo > 0
        scalar_features_dim = features_dim - image_features_dim
        # Đảm bảo scalar_features_dim cũng > 0 nếu features_dim đủ lớn
        if scalar_features_dim <= 0 and features_dim > image_features_dim:
             scalar_features_dim = 1
             image_features_dim = features_dim - scalar_features_dim # Điều chỉnh lại image_dim nếu cần
        elif scalar_features_dim <= 0:
             # Trường hợp features_dim quá nhỏ, phân bổ lại
             image_features_dim = max(1, features_dim - 1)
             scalar_features_dim = features_dim - image_features_dim

        print(f"Total features_dim: {features_dim}")
        print(f"Allocated image_features_dim: {image_features_dim}")
        print(f"Allocated scalar_features_dim: {scalar_features_dim}")
        assert image_features_dim + scalar_features_dim == features_dim, \
            f"Dimension allocation error: {image_features_dim} + {scalar_features_dim} != {features_dim}"
        assert image_features_dim > 0 and scalar_features_dim > 0, "Dimensions must be positive"
        # ----------------------------------------------------

        # image extractor (CNN) - Giả định input là (N, C, H, W)
        # self.image_extractor = Sequential(
        #     # N, 3, 64, 64
        #     nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2), # Lấy số kênh từ obs space
        #     nn.ReLU(),
        #     # N, 32, 16, 16
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     # N, 32, 8, 8
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     # N, 32, 8, 8
        #     nn.Flatten(),
        #     nn.Linear(32*8*8, 512), # Cần tính toán kích thước này cẩn thận dựa trên output CNN
        #     nn.ReLU(),
        # )

        self.image_extractor = Sequential(
            # N, 1, 84, 84
            nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=2), # Lấy số kênh từ obs space
            nn.LeakyReLU(),
            # N, 16, 21, 21
            nn.MaxPool2d(kernel_size=2, stride=2),
            # N, 16, 10, 10
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            # N, 32, 10, 10
            nn.Flatten(),
            nn.Linear(32*10*10, 512), # Cần tính toán kích thước này cẩn thận dựa trên output CNN
            nn.LeakyReLU(),
        )

        # MLP xử lý đặc trưng ảnh nối
        self.image_mlp = nn.Sequential(
            nn.Linear(512*3, 512), # 3 ảnh * 512 features/ảnh
            nn.LeakyReLU(),
            nn.Linear(512, image_features_dim), # Output đúng dimension đã phân bổ
            nn.LeakyReLU(),
        )

        # MLP xử lý scalar - Giả định input là (N, 1)
        self.scalar_mlp = Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(),
            nn.Linear(64, scalar_features_dim), # Output đúng dimension đã phân bổ
            nn.LeakyReLU(),
        )

    def forward(self, obs):
        # --- Xử lý ảnh ---
        image_keys = ['wrist_camera', 'front_camera', 'side_camera']
        # Giả định obs[key] là tensor từ SB3, có thể ở format (N, H, W, C)
        images = [obs[key] for key in image_keys]
        
        # Chuẩn hóa và chuyển đổi format nếu cần
        images_normalized = []
        for img in images:
            # Chuẩn hóa về [0, 1] nếu input ở dạng [0, 255]
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            elif img.max() > 1.0:
                img = img / 255.0
            
            # Xử lý các trường hợp shape khác nhau
            if len(img.shape) == 4:
                # Batch của images
                if img.shape[-1] == 1:  # (N, H, W, 1) - cần chuyển thành (N, 1, H, W)
                    img = img.permute(0, 3, 1, 2)
                elif img.shape[1] == 1:  # (N, 1, H, W) - đã đúng format
                    pass
                elif img.shape[1] == img.shape[2]:  # Có thể là (N, H, W, C) với H=W
                    # Kiểm tra nếu chiều cuối = 1 (grayscale)
                    if img.shape[3] == 1:
                        img = img.permute(0, 3, 1, 2)  # (N, H, W, 1) -> (N, 1, H, W)
                    else:
                        print(f"Warning: Unexpected 4D shape {img.shape}, assuming (N, C, H, W)")
                else:
                    print(f"Warning: Unexpected 4D shape {img.shape}")
            elif len(img.shape) == 3:
                # Single image
                if img.shape[-1] == 1:  # (H, W, 1)
                    img = img.permute(2, 0, 1).unsqueeze(0)  # -> (1, 1, H, W)
                elif img.shape[0] == 1:  # (1, H, W)
                    img = img.unsqueeze(0)  # -> (1, 1, H, W)
                else:
                    print(f"Warning: Unexpected 3D shape {img.shape}")
                    img = img.unsqueeze(0)  # Add batch dimension
            
            images_normalized.append(img)

        # Trích xuất đặc trưng CNN
        image_cnn_features = []
        for i, img in enumerate(images_normalized):
            try:
                # Debug: print shape nếu có lỗi
                if img.shape[1] != 1:  # Expected 1 channel
                    print(f"Warning: Image {i} has unexpected shape: {img.shape}")
                    print(f"Expected format: (N, 1, H, W), got: {img.shape}")
                    # Try to fix by taking only first channel if multiple channels
                    if len(img.shape) == 4 and img.shape[1] > 1:
                        img = img[:, :1, :, :]  # Take only first channel
                
                feature = self.image_extractor(img)
                image_cnn_features.append(feature)
            except Exception as e:
                print(f"Error processing image {i} with shape {img.shape}: {e}")
                print(f"Image dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")
                raise e

        # Nối đặc trưng ảnh
        concatenated_image_features = torch.cat(image_cnn_features, dim=1) # (N, 512*3)

        # Qua MLP ảnh
        image_final_features = self.image_mlp(concatenated_image_features) # (N, image_features_dim)
        # print(f'Image features shape: {image_final_features.shape}')

        # --- Xử lý scalar ---
        scalar_input = obs['orientation_error']
        # Đảm bảo là tensor, có shape (N, 1) và kiểu float
        if not isinstance(scalar_input, torch.Tensor):
            scalar_input = torch.tensor(scalar_input, device=self.device) # Chuyển sang tensor trên đúng device
        if len(scalar_input.shape) == 0:
            scalar_input = scalar_input.unsqueeze(0).unsqueeze(1)
        elif len(scalar_input.shape) == 1:
            scalar_input = scalar_input.unsqueeze(1) # Shape (N, 1)
        scalar_input = scalar_input.float() # Đảm bảo float

        scalar_features = self.scalar_mlp(scalar_input) # (N, scalar_features_dim)
        # print(f'Scalar features shape: {scalar_features.shape}')

        # --- Nối đặc trưng cuối cùng ---
        state = torch.cat((image_final_features, scalar_features), dim=1) # (N, features_dim)
        # print(f"Combined state shape: {state.shape}")

        return state


# --- Test Block (Cần điều chỉnh để mimic input từ SB3) ---
if __name__ == '__main__':
    from kuka_vision_grasping_env4 import KukaVisionGraspingEnv
    env = KukaVisionGraspingEnv()
    try:
        obs_np, info = env.reset() # obs_np là dict các numpy array

        # --- Mimic SB3 preprocessing ---
        # Best practice: Sử dụng DummyVecEnv và VecTransposeImage từ SB3 để tiền xử lý
        # Nhưng làm thủ công để test nhanh:
        processed_obs = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Sử dụng device phù hợp

        for key, value in obs_np.items():
            tensor_value = torch.tensor(value).to(device) # Chuyển sang tensor và đưa lên device
            if key in ['wrist_camera', 'front_camera', 'side_camera']:
                # Input ảnh từ env là (H, W, C) NumPy
                # Thêm batch dim (N=1), chuyển HWC -> CHW
                tensor_value = tensor_value.unsqueeze(0).permute(0, 3, 1, 2)
                # Chuyển sang float nếu CNN cần float (thường là vậy)
                tensor_value = tensor_value.float()
            else: # Scalar orientation_error
                # Input là scalar NumPy
                # Thêm batch dim (N=1), thêm feature dim (1)
                 tensor_value = tensor_value.unsqueeze(0)
                 # Đảm bảo float cho MLP
                 tensor_value = tensor_value.float()
            processed_obs[key] = tensor_value
        # --- End Mimic ---

        print("Processed Observation Shapes for Extractor (Mimicked SB3):")
        for key in processed_obs.keys():
            print(f'processed_obs: {key}, {processed_obs[key].shape}, {processed_obs[key].dtype}')

        # Khởi tạo extractor (cần observation_space gốc từ env)
        extractor = CustomExtractor(env.observation_space, features_dim=256).to(device) # Đưa extractor lên device

        # Test forward pass với observation đã xử lý
        with torch.no_grad(): # Không cần tính gradient khi test
            features = extractor.forward(processed_obs)
        print(f"Extractor output shape: {features.shape}") # Phải là (1, features_dim)
        assert features.shape == (1, extractor.features_dim), "Output shape mismatch"

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()