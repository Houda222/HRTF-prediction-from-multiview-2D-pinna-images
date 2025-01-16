import argparse
import sofar
import torch
import numpy as np
import open3d as o3d
import cv2
import sys
sys.path.append('.')
from model.depth_anything_v2.dpt import DepthAnythingV2
from model.HRTFNet_onefreq import MultiViewHRTFPredictionModel

class HRTFPredictor:
    VIEW_OPTIONS = {
        1: [3, 6, 9],                 # Task 1: 3 views
        2: [0, 3, 6, 9, 12, 15, 18],  # Task 2: 7 views
        3: list(range(19))            # Task 3: 19 views
    }

    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load depth model
        self.depth_model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
        self.depth_model.load_state_dict(torch.load('./checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
        self.depth_model.eval().to(self.device)
        
        # Load HRTF model
        self.model = MultiViewHRTFPredictionModel().to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.mean_hrtf = torch.load('./checkpoints/mean_hrtf.pt', map_location=self.device).to(self.device)
    
        
        self.num_views_total = 19
    
    def complex_hrtf(self, hrtf_mag_phase):
        # Split the real and imaginary parts
        hrtf_magnitude, hrtf_phase = torch.chunk(hrtf_mag_phase, 2, dim=-1)
        # Combine them to form the complex hrtf
        hrtf = hrtf_magnitude * torch.exp(1j * hrtf_phase)
        return hrtf
        
    def read_image(self, path):
        rgb_origin = cv2.imread(path)
        rgb_origin = rgb_origin[:, :, ::-1]
        
        background_mask = np.all(rgb_origin == np.array([255, 255, 255]), axis=-1)
        
        input_size = (616, 1064)
        h, w = rgb_origin.shape[:2]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, 
                                cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb = (rgb - mean) / std
        rgb = rgb[None, :, :, :].to(self.device)
        
        return rgb, pad_info, background_mask

    def infer_depth(self, rgb, pad_info):
        with torch.no_grad():
            pred_depth = self.depth_model(rgb)

        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[
            pad_info[0]: pred_depth.shape[0] - pad_info[1],
            pad_info[2]: pred_depth.shape[1] - pad_info[3]
        ]

        pred_depth = torch.nn.functional.interpolate(
            pred_depth[None, None, :, :], size=(1024, 1024), mode='bilinear'
        ).squeeze()
        
        return pred_depth

    def to_point_cloud(self, depth_map, voxel_size=0.005):
        h, w = depth_map.shape
        yy, xx = np.meshgrid(
            np.linspace(-1, 1, h),
            np.linspace(-1, 1, w),
            indexing='ij'
        )
        
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()
        depth_flat = depth_map.flatten()
        mask = depth_flat > 0
        points = np.stack((xx_flat[mask], yy_flat[mask], depth_flat[mask]), axis=-1)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if voxel_size is not None:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        return np.asarray(pcd.points)

    def process_images(self, image_paths):
        point_clouds = []
        for path in image_paths:
            rgb, pad_info, background_mask = self.read_image(path)
            depth = self.infer_depth(rgb, pad_info)
            
            depth[background_mask] = 0
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            
            point_cloud = self.to_point_cloud(depth.cpu().numpy())
            indices = np.random.choice(point_cloud.shape[0], 30000, replace=True)
            point_cloud = point_cloud[indices]
            point_cloud -= np.mean(point_cloud, axis=0)
            point_cloud /= np.max(np.linalg.norm(point_cloud, axis=1))
            
            point_clouds.append(point_cloud)
            
        return point_clouds

    def determine_task_and_indices(self, num_images):
        if num_images == 3:
            task, indices = 1, self.VIEW_OPTIONS[1]
        elif num_images == 7:
            task, indices = 2, self.VIEW_OPTIONS[2]
        elif num_images == 19:
            task, indices = 3, self.VIEW_OPTIONS[3]
        else:
            task, indices = None, list(range(num_images))
        return task, indices

    def run(self, left_paths, right_paths, output_path):
        # Process images
        left_clouds = self.process_images(left_paths)
        right_clouds = self.process_images(right_paths)
        
        # Determine task and prepare input tensor
        num_images = len(left_paths)
        _, indices = self.determine_task_and_indices(num_images)
        
        # Create input tensor
        input_tensor = torch.zeros(1, 2, self.num_views_total, 30000, 3)
        
        # Fill tensor with point clouds
        for idx, (left_cloud, right_cloud) in enumerate(zip(left_clouds, right_clouds)):
            if idx >= len(indices):
                break
            input_tensor[0, 0, indices[idx]] = torch.from_numpy(left_cloud).float()
            input_tensor[0, 1, indices[idx]] = torch.from_numpy(right_cloud).float()
        
        # Run HRTF prediction
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output_magnitude_phase = self.model(input_tensor)
            
            # Process complex HRTF
            output_complex = self.complex_hrtf(output_magnitude_phase)
            batch_size = output_complex.shape[0]
            uncentered_output_complex = output_complex + self.mean_hrtf.expand(batch_size, -1, -1, -1)
            
            # Convert to numpy and prepare for SOFA
            predictions = uncentered_output_complex.cpu().numpy().squeeze()

        print(predictions.shape)
        hrir_data = np.fft.irfft(predictions, axis=-1)
        
        sofa = sofar.Sofa('SimpleFreeFieldHRIR')
        sofa.Data_IR = hrir_data
        sofa.Data_SamplingRate = 48000
        sofa.verify()
        sofar.write_sofa(output_path, sofa)

def main():
    parser = argparse.ArgumentParser(description="HRTF Prediction from Ear Images")
    parser.add_argument("-l", "--left", nargs='+', required=True, help="Left ear images")
    parser.add_argument("-r", "--right", nargs='+', required=True, help="Right ear images")
    parser.add_argument("-o", "--output_path", required=True, help="Output SOFA file path")
    parser.add_argument("--model_path", default="HRTFNet.pth", help="HRTF model weights path")
    
    args = parser.parse_args()
    
    if len(args.left) != len(args.right):
        raise ValueError("Number of left and right images must be equal")
        
    predictor = HRTFPredictor(args.model_path)
    predictor.run(args.left, args.right, args.output_path)

if __name__ == "__main__":
    main()