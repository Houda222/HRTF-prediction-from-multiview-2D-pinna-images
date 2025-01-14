from __future__ import annotations
from typing import Dict, List, Tuple
import sofar
import glob
import re
import numpy as np
from imageio.v3 import imread
import os
import torch
import tqdm
import sys
sys.path.append('/mnt/thau04b/hghallab/comp/Final model/metrics.py')
from metrics import MeanSpectralDistortion
from enum import Enum, auto
import trimesh
from typing import Dict, List, Tuple, Optional, Literal
import open3d as o3d
import pprint
from sht_utils import compute_sht_coeffs


all_tasks = [np.arange(19).tolist(), np.arange(19, step=3).tolist(), [3, 6, 9]]


class STLFormat(Enum):
    RAW = ".stl"
    WATERTIGHT = "_watertight.stl"
    PREPARED = "_wtt_prepd.stl"
    PROJECT_ASC = "_Project1.asc"
    PLY = ".ply"

class ProcessingType(Enum):
    VOXELS = auto()
    POINT_CLOUD = auto()
    MESH = auto()

class DataConfiguration:
    def __init__(
        self,
        use_2d: bool = True,
        use_3d: bool = False,
        use_hrtf: bool = True,
        use_3d_head: bool = True,
        precompute_sht: bool = True,
        stl_format: STLFormat = STLFormat.PROJECT_ASC,
        processing_type: ProcessingType = ProcessingType.POINT_CLOUD,
        voxel_resolution: int = 64,
        num_points: int = 2048
    ):
        self.use_2d = use_2d
        self.use_3d = use_3d
        self.use_hrtf = use_hrtf
        self.use_3d_head = use_3d_head
        self.precompute_sht = precompute_sht
        if use_3d_head:
            self.stl_format = stl_format
        else:
            self.stl_format = STLFormat.PLY
        self.processing_type = processing_type
        self.voxel_resolution = voxel_resolution
        self.num_points = num_points

class SonicomDatabase(torch.utils.data.Dataset):

    def __init__(
        self,
        root_dir: str,
        config: DataConfiguration,
        sht_order: int,
        hrtf_type="FreeFieldCompMinPhase",
        no_itd=True,
        sampling_rate="48kHz",
        nfft=256,
        training_data: bool = True,
        task_id: int = 0,
    ):
        """
        Args:
            root_dir: Directory with all the HRTF files in subfolder.
            hrtf_type: can be any of ['Raw','Windowed','FreeFieldComp','FreeFieldCompMinPhase'(default)]
            sampling_rate: any of 44kHz, 48kHz, 96kHz
            nfft: fft length
            training_data: if true then return training dataset
            task_id: task id determines how many images will be used for inference. Can be 0, 1, or 2. 
        """
        super().__init__()
        self.root_dir = root_dir
        self.hrtf_type = hrtf_type
        self.nfft = nfft
        self.config = config
        self.sht_order = sht_order
        self.training_data = training_data

        if no_itd:
            itd_str = "NoITD_"
        else:
            itd_str = ""

        self.hrtf_files = glob.glob(
            os.path.join(
                root_dir,
                f"P*/P*/HRTF/HRTF/{sampling_rate}/*_{hrtf_type}_{itd_str}{sampling_rate}.sofa",
            )
        )
        
        # pprint.pprint(sorted(self.hrtf_files))
        # print('Found ' + str(len(self.hrtf_files)) + ' files')

        # if training_data:
        #     self.image_dir = os.path.join(root_dir, "SONICOM_TrainingData_pics")
        #     self.task = all_tasks[0]
        # else:
        #     self.image_dir = os.path.join(root_dir, "SONICOM_TestData_pics")
        #     self.task = all_tasks[task_id]

        # self.all_image_names = [i for i in os.listdir(self.image_dir) if ".png" in i]
        # self.all_subjects = self.get_available_ids()
        
        missing_instances_list = [
            "P0234", "P0238", "P0241", "P0242", "P0246", "P0247", "P0248", "P0249", "P0250",
            "P0251", "P0252", "P0253", "P0254", "P0256", "P0257", "P0258", "P0259"
        ]
        
        not_sliced_ears = [
            "P0004", "P0038", "P0068", "P0084", "P0099", "P0102", "P0114", "P0122", "P0134", "P0135", 
            "P0136", "P0143", "P0145", "P0146", "P0150", "P0151", "P0164", "P0173", "P0175", "P0183", 
            "P0187", "P0189", "P0190", "P0191", "P0202", "P0216", "P0217", "P0220", "P0228", "P0230", 
            "P0231"
        ]
        
        not_sliced_ears = []

        if training_data:
            self.image_dir = os.path.join(root_dir, "SONICOM_TrainingData_pics")
            self.depthmap_dir = os.path.join(self.root_dir, "SONICOM_TrainingData_depthmaps")
            self.task = all_tasks[0]
            if config.use_2d:
                self.all_image_names = [i for i in os.listdir(self.image_dir) if ".png" in i]
            else:
                all_image_names = [f"P{str(i).zfill(4)}" for i in range(1, 245)]
                test_image_names = [name[:5] for name in os.listdir(os.path.join(root_dir, "SONICOM_TestData_pics")) if "png" in name] + missing_instances_list
                self.all_image_names = [img_name for img_name in all_image_names if img_name not in test_image_names]
                
        else:
            self.image_dir = os.path.join(root_dir, "SONICOM_TestData_pics")
            self.depthmap_dir = os.path.join(self.root_dir, "SONICOM_TestData_depthmaps")
            self.task = all_tasks[task_id]
            self.all_image_names = [i for i in os.listdir(self.image_dir) if ".png" in i]
        
        if not config.use_3d_head:
            self.all_image_names = [img_name for img_name in self.all_image_names if img_name[:5] not in not_sliced_ears]
            
        self.all_subjects = self.get_available_ids()
        
        # self.all_subjects = [item for item in self.all_subjects if int(item[1:5]) < 100]
        # self.all_image_names = [i for i in self.all_image_names if i[:5] in self.all_subjects]
        
        print("Number of subjects: ", len(self.all_subjects))
        print("Total number of images: ", len(self.all_image_names))

        # read one to get coordinate system information
        try:
            tmp = sofar.read_sofa(self.hrtf_files[0], verbose=False)
            self.training_data = training_data
            self.position = tmp.SourcePosition
        except (IndexError, ValueError):
            print("Check if Dataset is saved as described in the notebook.")
            return None

        if config.stl_format == STLFormat.PLY:
            self.mesh_files = glob.glob(
            os.path.join(
                root_dir,
                f"P*/P*/3DSCAN/*[0-9]*{config.stl_format.value}",
            )
            )
        else:
            self.mesh_files = glob.glob(
                os.path.join(
                    root_dir,
                    f"P*/P*/3DSCAN/*[0-9]{self.config.stl_format.value}",
                )
            )
        
        self.coeffs_cache_path = os.path.join(root_dir, f"sht_coeffs_cache{'_train' if self.training_data else '_test'}.pt")
        self.sht_coeffs = self._load_or_compute_coeffs()
        print("SHT coefficients loaded.")
        
        
    def _load_or_compute_coeffs(self):
        # if os.path.exists(self.coeffs_cache_path):
        #     return torch.load(self.coeffs_cache_path)
        
        # Compute coefficients for all HRTFs
        coeffs_dict = {}
        for subject_id in tqdm.tqdm(self.all_subjects, desc="Computing SHT coefficients"):
            hrtf, positions = self.load_subject_id_hrtf(subject_id)
            hrtf = torch.from_numpy(hrtf).cfloat()
            positions = torch.from_numpy(positions).float()
            coeffs = compute_sht_coeffs(hrtf.unsqueeze(0), positions)
            coeffs_dict[subject_id] = coeffs.squeeze(0)
        
        # Save cache
        # torch.save(coeffs_dict, self.coeffs_cache_path)
        return coeffs_dict

    def __len__(self):
        return len(self.all_subjects)

    def load_all_hrtfs(self) -> torch.Tensor:
        """
        This function loads all the HRTFs from the list of IDs.

        Returns:
            Magnitude Spectrum of HRTFs : torch.Tensor
        """
        HRTFs = torch.zeros(
            (self.__len__(), self.position.shape[0], 2, self.nfft // 2 + 1)
        )

        allids = np.unique([cur_id[:5] for cur_id in self.all_image_names])
        for idx in range(len(allids)):
            if allids[idx] == allids[idx - 1] and idx > 0:
                HRTFs[idx] = HRTFs[idx - 1]
            else:
                HRTFs[idx] = torch.from_numpy(
                    self.load_subject_id_hrtf(allids[idx])[0] ###### [0] because the function was chaged to return positions as well
                ).abs()
        return HRTFs

    def load_image(self, image_name: str) -> Tuple[np.ndarray, str, str]:
        """
        This function read all the image files in the directory, get the ID of the image, Left or Right side of the pinna.

        Args:
            image_name (str): e.g. P0002_left_0.png

        Returns:
            image: torch.Tensor
            ID: (str) Subject ID of the loaded image
            Face_Side: (str) If the image loaded is of the left ear or the right ear
        """

        image = imread(os.path.join(self.image_dir, image_name))
        ID = image_name[:5]
        Face_Side = ["left" if "left" in image_name else "right"]

        return image, ID, Face_Side

    def get_image_names_from_id(self, id: str) -> List[str]:
        """
        This function helps to get the image names from the directory.

        Args:
            id (str): Subject ID e.g. 'P0001'
        Returns:
            List of image name
        """
        return [
            x for x in os.listdir(self.image_dir) if f"{id}" in x
        ]  # glob.glob(os.path.join(self.image_dir, f'{id}*'))

    def get_available_ids(self) -> List[str]:
        """
        This function returns all unique IDs from the list of images.

        Args:
            all_images (list of str)
        Returns:
            list of unique IDs
        """
        return list({name[:5] for name in self.all_image_names})

    def _extract_number_of_image(self, image_name: str) -> List[int]:
        """
        Extracts the image number of the subject from an image filename.

        Args:
            image_name (str): Filename of the image.

        Returns:
            Optional[int]: value if successfully extracted; otherwise, None.
        """
        try:
            azi_str = image_name.split("t_")[1]
            number = int(azi_str.split(".")[0])
            return number
        except (IndexError, ValueError):
            return None

    def _get_task_subset_image_names(self, image_names: List[str]) -> Tuple[List[str], List[str]]:
        """
        Returns two Lists of left and right image names from selected subset (based on task).

        Args:
            image_names (List of str): Filenames of the images.

        Returns:
            Dict e.g. {left_0: [0, 'P0002_left_0.png'], right_0: [1, 'P0002_right_0.png']}
        """

        left_names = []
        right_names = []
        for i in image_names:
            cur_azi = self._extract_number_of_image(i)
            if cur_azi in self.task:
                if "left" in i:
                    left_names.append(i)  # channel, name
                if "right" in i:
                    right_names.append(i)

        return left_names, right_names

    def get_all_images_and_HRTF_from_id(
        self, id: str, load_images: bool = True, load_hrtf: bool = True
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Loads all the images for the subject (only the subset) and the corresponding HRTF.

        Args:
            ID of each subject (str): e.g. P0001

        Returns:
            all_images: torch.Tensor of shape (ear_idx, image_idx, height, width)
            HRTFs: torch.Tensor
        """
         
        
        if load_images:
            
            image_names = self.get_image_names_from_id(id)
            image_names.sort()
            left_images_filenames, right_images_filenames = (
                self._get_task_subset_image_names(image_names)
            )
            left_images = []
            right_images = []
            # import sys
            # print(len(image_names), len(left_images_filenames), len(right_images_filenames), "\n\n\n")
            # sys.stdout.flush()

            if not left_images_filenames or not right_images_filenames:
                raise FileNotFoundError(f"No images found for subject ID '{id}'.")       
            
            left_images = torch.from_numpy(np.stack([imread(os.path.join(self.image_dir, path)) for path in left_images_filenames]))
            right_images = torch.from_numpy(np.stack([imread(os.path.join(self.image_dir, path)) for path in right_images_filenames]))
            all_images = torch.stack((left_images, right_images))
            
            
            left_depths = torch.from_numpy(np.stack([np.load(os.path.join(self.depthmap_dir, path.split(".")[0] + ".npy")) for path in left_images_filenames]))
            right_depths = torch.from_numpy(np.stack([np.load(os.path.join(self.depthmap_dir, path.split(".")[0] + ".npy")) for path in right_images_filenames]))
            all_depth_maps = torch.stack((left_depths, right_depths))
        else:
            all_images = None
            all_depth_maps = None

        if load_hrtf:
            HRTF, positions = self.load_subject_id_hrtf(id)
            coeffs = self.sht_coeffs[id]
        else:
            HRTF, positions, coeffs = None, None, None

        return all_images, all_depth_maps, HRTF, positions, coeffs
    
    
    # def load_images_and_depth_maps(self, id: str) -> Tuple[torch.Tensor, torch.Tensor]:
    #     image_names = self.get_image_names_from_id(id)
    #     image_names.sort()
    #     left_images_filenames, right_images_filenames = self._get_task_subset_image_names(image_names)
        
    #     left_images = torch.from_numpy(np.stack([
    #         imread(os.path.join(self.image_dir, path)) for path in left_images_filenames
    #     ]))
    #     right_images = torch.from_numpy(np.stack([
    #         imread(os.path.join(self.image_dir, path)) for path in right_images_filenames
    #     ]))
    #     all_images = torch.stack((left_images, right_images))

        
    #     return all_images, depth_maps

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        This function is used by the Dataloader, it iterates through the number of subjects in the current dataset
        and provides the corresponding Images, HRTFs and Subject IDs.
        """
        id = self.all_subjects[idx]
        outputs = []
        
        # print(id)

        # if self.config.use_2d:
        #     all_images, depth_maps = self.load_images_and_depth_maps(id)
        #     outputs.append(all_images)
        #     outputs.append(depth_maps)

        all_images, all_depthmaps, hrtf, positions, coeffs = self.get_all_images_and_HRTF_from_id(id, load_images=self.config.use_2d, load_hrtf=self.config.use_hrtf)

        if self.config.use_2d:
            outputs.append(all_images)
            outputs.append(all_depthmaps)

        if self.config.use_3d:
            mesh_data = self.load_subject_id_3d(id)
            if mesh_data is not None:
                mesh_data = torch.from_numpy(mesh_data).float()
                outputs.append(mesh_data)
            else:
                # Handle missing 3D data if necessary
                print("####### Missing 3D data for subject ID: ", id)
                pass

        if self.config.use_hrtf:
            outputs.append(hrtf)
            outputs.append(positions)
            outputs.append(coeffs)
        # if self.config.use_ears:
        #     # Load ear data
        return tuple(outputs)
    
    def load_subject_id_hrtf(self, subject_id: str, return_sofa: bool = False) ->  Tuple[np.ndarray, np.ndarray]:
        """
        This function load the HRIR data for the given file name and compute the RFFT of the HRIRs then return HRTFs
        Example if the file name is P0001, this function will load the sofa file of P0001 - read it and return the HRTF data of the P001

        Args:
            subject_id (str): e.g. P0001, ..., P0200
        """

        hrtf_file_lst = [s for s in self.hrtf_files if subject_id + "_" + self.hrtf_type in s]

        if len(hrtf_file_lst) == 0:
            print(subject_id)
            pprint.pprint(sorted(self.hrtf_files))
            pprint.pprint(sorted(hrtf_file_lst))
            print(subject_id + " Not found!")
            return None
        else:
            hrtf_file = hrtf_file_lst[0]
        if return_sofa:
            return sofar.read_sofa(hrtf_file, verbose=False), None
        else:
            # hrir = self._load_hrir(hrtf_file)
            data = sofar.read_sofa(hrtf_file, verbose=False)
            hrir = data.Data_IR
            positions = data.SourcePosition
            hrtf = self._compute_HRTF(hrir)  # Shape: [793, 2, 129]
            
            positions_rad = np.deg2rad(positions[:, :2])  # [azimuth, elevation] in radians
            positions_rad = np.concatenate([positions_rad, positions[:, 2:]], axis=1)  # Append distance

            
            return hrtf, positions_rad
    
    
    # o1 suggested function
    # def load_subject_id_hrtf(self, subject_id: str) -> Tuple[np.ndarray, np.ndarray]:
    #     hrtf_file_lst = [s for s in self.hrtf_files if subject_id + "_" + self.hrtf_type in s]
    #     if len(hrtf_file_lst) == 0:
    #         raise FileNotFoundError(f"No HRTF file found for subject {subject_id}")
    #     else:
    #         hrtf_file = hrtf_file_lst[0]
    #     data = sofar.read_sofa(hrtf_file, verbose=False)
    #     hrir = data.Data_IR  # Shape: [793, 2, time_samples]
    #     positions = data.SourcePosition  # Shape: [793, 3] (azimuth, elevation, distance)
    #     hrtf = self._compute_HRTF(hrir)  # Shape: [793, 2, 129]
    #     return hrtf, positions

    def _load_hrir(self, hrtf_file: sofar.Sofa) -> np.ndarray:
        """
        This function load the HRIR data for the given filename.

        Args:
              sofa file
            Returns:
               HRIR data"""
        data = sofar.read_sofa(hrtf_file, verbose=False)
        return data.Data_IR

    def _compute_HRTF(self, hrir: np.ndarray) -> np.ndarray:
        """
        This function compute the RFFT of the given HRIRs and return HRTFs.

        Args:
              HRIRs (time domain)
            Returns:
               HRTFs (Frequency domain)"""

        return np.fft.rfft(hrir, n=self.nfft)
    
    def _convert_to_voxels(self, mesh: trimesh.Trimesh) -> np.ndarray:
        voxel_data = mesh.voxelized(pitch=1.0 / self.config.voxel_resolution)
        return voxel_data.matrix

    def _convert_to_point_cloud(self, mesh: trimesh.Trimesh) -> np.ndarray:
        points, _ = trimesh.sample.sample_surface(mesh, self.config.num_points)
        return points
        
    def _process_raw_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        return np.array(mesh.vertices)

    def load_subject_id_3d(self, subject_id: str) -> np.ndarray:
        """Loads the 3D mesh data for the given subject ID."""
        if self.config.use_3d_head:
            pattern = subject_id + self.config.stl_format.value
        else:
            pattern = subject_id
            # pattern_left = subject_id + "_left" + ".ply"
            # pattern_right = subject_id + "_right" + ".ply"
            
        
            
        if self.config.use_3d_head:
            matching_files = [
                f for f in self.mesh_files 
                if os.path.basename(f).endswith(pattern)
            ]
            if not matching_files:
                print(f"No mesh file found for subject {subject_id}")
                return None
                
            point_cloud = np.loadtxt(matching_files[0])
            head_mask = point_cloud[:, 0] > -60
            point_cloud = point_cloud[head_mask]
            
            num_desired_points = 20000 # change to 400000
            num_points_in_cloud = point_cloud.shape[0]
            
            if num_points_in_cloud >= num_desired_points:
                indices = np.random.choice(num_points_in_cloud, num_desired_points, replace=False)
            else:
                indices = np.random.choice(num_points_in_cloud, num_desired_points, replace=True)
            point_cloud = point_cloud[indices]
            
            points = point_cloud[:, :3]
            normals = point_cloud[:, 3:]
            
            centroid = np.mean(points, axis=0)
            points -= centroid
            
            max_distance = np.max(np.sqrt(np.sum(points**2, axis=1)))
            points /= max_distance
            
            return np.hstack((points, normals))
        # else:
        #     matching_left = [f for f in self.mesh_files if os.path.basename(f).endswith(pattern_left)]
        #     matching_right = [f for f in self.mesh_files if os.path.basename(f).endswith(pattern_right)]
            
        #     if not (matching_left and matching_right):
        #         print(f"Missing ear data for subject {subject_id}")
        #         return None
                
        #     # Left ear processing
        #     left_point_cloud_file = o3d.io.read_point_cloud(matching_left[0])
        #     left_point_cloud = np.asarray(left_point_cloud_file.points)
        #     left_normals = np.asarray(left_point_cloud_file.normals)
            
        #     num_desired_points = 20000
        #     num_points_in_cloud = left_point_cloud.shape[0]
            
        #     if num_points_in_cloud >= num_desired_points:
        #         indices = np.random.choice(num_points_in_cloud, num_desired_points, replace=False)
        #     else:
        #         indices = np.random.choice(num_points_in_cloud, num_desired_points, replace=True)
            
        #     left_points = left_point_cloud[indices]
        #     left_normals = left_normals[indices]
            
        #     left_centroid = np.mean(left_points, axis=0)
        #     left_points -= left_centroid
            
        #     left_max_distance = np.max(np.sqrt(np.sum(left_points**2, axis=1)))
        #     left_points /= left_max_distance
            
        #     left_point_cloud_normalized = np.hstack((left_points, left_normals))
            
        #     # Right ear processing
        #     right_point_cloud_file = o3d.io.read_point_cloud(matching_right[0])
        #     right_point_cloud = np.asarray(right_point_cloud_file.points)
        #     right_normals = np.asarray(right_point_cloud_file.normals)
                        
        #     num_points_in_cloud = right_point_cloud.shape[0]

        #     if num_points_in_cloud >= num_desired_points:
        #         indices = np.random.choice(num_points_in_cloud, num_desired_points, replace=False)
        #     else:
        #         indices = np.random.choice(num_points_in_cloud, num_desired_points, replace=True)
            
        #     right_points = right_point_cloud[indices]
        #     right_normals = right_normals[indices]
            
        #     right_centroid = np.mean(right_points, axis=0)
        #     right_points -= right_centroid
            
        #     right_max_distance = np.max(np.sqrt(np.sum(right_points**2, axis=1)))
        #     right_points /= right_max_distance
            
        #     right_point_cloud_normalized = np.hstack((right_points, right_normals))

        else:
            if self.training_data:
                directory = f"/autofs/thau04b/hghallab/comp/Huawei/TechArena20241016/data/SONICOM_TrainingData_pointclouds/{pattern}"
            else:
                directory = f"/autofs/thau04b/hghallab/comp/Huawei/TechArena20241016/data/SONICOM_TestData_pointclouds/{pattern}"
            
            def extract_index(filename):
                match = re.search(r'_(\d+)\.npy$', filename)
                return int(match.group(1)) if match else -1

            point_clouds = []
            for side in ['left', 'right']:
                files = sorted(
                    [f for f in os.listdir(directory) if f.endswith('.npy') and side in f.lower()],
                    key=extract_index
                )
                side_point_clouds = []
                for file_name in files:
                    point_cloud = np.load(os.path.join(directory, file_name))
                    indices = np.random.choice(point_cloud.shape[0], 30000, replace=False)
                    point_cloud = point_cloud[indices]
                    point_cloud -= np.mean(point_cloud, axis=0)
                    point_cloud /= np.max(np.linalg.norm(point_cloud, axis=1))
                    side_point_clouds.append(point_cloud)
                point_clouds.append(side_point_clouds)

            point_clouds = np.array(point_clouds) # Shape: (1, 2, 19, 30000, 3)

            # Convert to a numpy array
            return point_clouds

            
            return np.concatenate((left_point_cloud_normalized, right_point_cloud_normalized), axis=0)


def baseline_spectral_distortion(sd: SonicomDatabase, path_to_baseline_hrtf: str = "./data/Average_HRTFs.sofa") -> float:
    # this function calculate the spectral difference as mean square error between your ground truth HRTFs and the baseline average HRTFs
    # load all HRTFS, concat in 1 tensor, clone Average_HRTFs as many times and then find get_spectral_distortion
    """Returns:
    baseline prediction MSE in dB
    """

    all_HRTFs = sd.load_all_hrtfs()
    baseline_HRIR = sofar.read_sofa(path_to_baseline_hrtf, verbose=False).Data_IR
    baseline_HRTF = torch.from_numpy(sd._compute_HRTF(baseline_HRIR))
    baseline_HRTF = baseline_HRTF.unsqueeze(0).repeat(all_HRTFs.shape[0], 1, 1, 1)
    eval_metric = MeanSpectralDistortion()

    return eval_metric.get_spectral_distortion(all_HRTFs, baseline_HRTF)


def convert_to_HRIR(hrtfs: np.ndarray) -> np.ndarray:
    return np.fft.irfft(hrtfs, axis=-1)

def save_sofa(HRIR: np.ndarray, output_path: str, reference_sofa: sofar.Sofa):
    """
    Save the HRIR to a SOFA object file. See main() for example usage

    Args:
        HRIR (np.ndarray): HRIR of shape (793, 2, 256).
        output_path (str): Path where the SOFA file will be saved.
        reference_sofa (str): The SOFA object to copy information
    """
    hrtf = reference_sofa
    hrtf.Data_IR = HRIR
    sofar.write_sofa(output_path, hrtf, 0)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    sonicom_root = "./data"
    config = DataConfiguration(use_2d=True, use_3d=True, use_hrtf=True)
    sd = SonicomDatabase(sonicom_root, config, training_data=False, task_id=2)
    train_dataloader = DataLoader(sd, batch_size=1, shuffle=False)

    for i, (images, hrtf) in tqdm.tqdm(enumerate(train_dataloader)):
        print(f"Image size: {images.shape} and HRTF size: {hrtf.shape}")
        break

    Error = baseline_spectral_distortion(sd)
    print(Error)



# def __init__(
#         self,
#         root_dir: str,
#         config: DataConfiguration,
#         hrtf_type="FreeFieldCompMinPhase",
#         no_itd=True,
#         sampling_rate="48kHz",
#         nfft=256,
#         training_data: bool = True,
#         task_id: int = 0,
#     ):
#         """
#         Args:
#             root_dir: Directory with all the HRTF files in subfolder.
#             hrtf_type: can be any of ['Raw','Windowed','FreeFieldComp','FreeFieldCompMinPhase'(default)]
#             sampling_rate: any of 44kHz, 48kHz, 96kHz
#             nfft: fft length
#             training_data: if true then return training dataset
#             task_id: task id determines how many images will be used for inference. Can be 0, 1, or 2. 
#         """
#         super().__init__()
#         self.root_dir = root_dir
#         self.hrtf_type = hrtf_type
#         self.nfft = nfft
#         self.config = config

#         if no_itd:
#             itd_str = "NoITD_"
#         else:
#             itd_str = ""

#         self.hrtf_files = glob.glob(
#             os.path.join(
#                 root_dir,
#                 f"P*/P*/HRTF/HRTF/{sampling_rate}/*_{hrtf_type}_{itd_str}{sampling_rate}.sofa",
#             )
#         )
        
#         # pprint.pprint(sorted(self.hrtf_files))
#         # print('Found ' + str(len(self.hrtf_files)) + ' files')

#         # if training_data:
#         #     self.image_dir = os.path.join(root_dir, "SONICOM_TrainingData_pics")
#         #     self.task = all_tasks[0]
#         # else:
#         #     self.image_dir = os.path.join(root_dir, "SONICOM_TestData_pics")
#         #     self.task = all_tasks[task_id]

#         # self.all_image_names = [i for i in os.listdir(self.image_dir) if ".png" in i]
#         # self.all_subjects = self.get_available_ids()
        
#         missing_instances_list = [
#             "P0234", "P0238", "P0241", "P0242", "P0246", "P0247", "P0248", "P0249", "P0250",
#             "P0251", "P0252", "P0253", "P0254", "P0256", "P0257", "P0258", "P0259"
#         ]

#         if training_data:
#             self.image_dir = os.path.join(root_dir, "SONICOM_TrainingData_pics")
#             self.depthmap_dir = os.path.join(self.root_dir, "SONICOM_TrainingData_depthmaps")
#             self.task = all_tasks[0]
#             if config.use_2d:
#                 self.all_image_names = [i for i in os.listdir(self.image_dir) if ".png" in i]
#             else:
#                 all_image_names = [f"P{str(i).zfill(4)}" for i in range(1, 245)]
#                 test_image_names = [name[:5] for name in os.listdir(os.path.join(root_dir, "SONICOM_TestData_pics")) if "png" in name] + missing_instances_list
#                 self.all_image_names = [img_name for img_name in all_image_names if img_name not in test_image_names]
                
#         else:
#             self.image_dir = os.path.join(root_dir, "SONICOM_TestData_pics")
#             self.depthmap_dir = os.path.join(self.root_dir, "SONICOM_TestData_depthmaps")
#             self.task = all_tasks[task_id]
#             self.all_image_names = [i for i in os.listdir(self.image_dir) if ".png" in i]
            
#         self.all_subjects = self.get_available_ids()
        
#         # self.all_subjects = [item for item in self.all_subjects if int(item[1:5]) < 100]
#         # self.all_image_names = [i for i in self.all_image_names if i[:5] in self.all_subjects]
        
#         print("Number of subjects: ", len(self.all_subjects))
#         print("Total number of images: ", len(self.all_image_names))

#         # read one to get coordinate system information
#         try:
#             tmp = sofar.read_sofa(self.hrtf_files[0], verbose=False)
#             self.training_data = training_data
#             self.position = tmp.SourcePosition
#         except (IndexError, ValueError):
#             print("Check if Dataset is saved as described in the notebook.")
#             return None

#         self.mesh_files = glob.glob(
#             os.path.join(
#                 root_dir,
#                 f"P*/P*/3DSCAN/*[0-9]{self.config.stl_format.value}",
#             )
#         )

    # def load_images_and_depth_maps(self, id: str) -> Tuple[torch.Tensor, torch.Tensor]:
    #     image_names = self.get_image_names_from_id(id)
    #     image_names.sort()
    #     left_images_filenames, right_images_filenames = self._get_task_subset_image_names(image_names)
        
    #     left_images = torch.from_numpy(np.stack([
    #         imread(os.path.join(self.image_dir, path)) for path in left_images_filenames
    #     ]))
    #     right_images = torch.from_numpy(np.stack([
    #         imread(os.path.join(self.image_dir, path)) for path in right_images_filenames
    #     ]))
    #     all_images = torch.stack((left_images, right_images))

    #     left_depths = torch.from_numpy(np.stack([
    #         imread(os.path.join(self.depthmap_dir, path)) for path in left_images_filenames
    #     ]))
    #     right_depths = torch.from_numpy(np.stack([
    #         imread(os.path.join(self.depthmap_dir, path)) for path in right_images_filenames
    #     ]))
    #     depth_maps = torch.stack((left_depths, right_depths))
        
    #     return all_images, depth_maps

