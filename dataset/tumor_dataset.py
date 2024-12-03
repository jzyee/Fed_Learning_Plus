from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

class MRI_Tumor_17(Dataset):
    """
    MRI Tumor dataset with 17 classes, following iCIFAR10 structure
    Dataset structure: dataset/brain_tumor/{class_name}/img
    """
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
        test_transform: Optional[transforms.Compose] = None,
        target_test_transform: Optional[transforms.Compose] = None,
        download: bool = False
    ):
        self.root = Path(root) / "tumor"
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.test_transform = test_transform
        self.target_test_transform = target_test_transform
        
        # Initialize empty arrays for train/test data
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []
        
        # Load dataset
        self._load_dataset()
        
    def _load_dataset(self) -> None:
        """
        Load the full dataset and store as numpy arrays
        Recursively searches for .jpg files in subdirectories
        """
        if not self.root.exists():
            raise RuntimeError(
                f"Dataset directory {self.root} does not exist.\n"
                f"Please ensure the directory exists at: {self.root.absolute()}"
            )
        
        print(f"Current working directory: {Path.cwd()}")
        print(f"Looking for .jpg images in: {self.root.absolute()}")
        
        def get_class_dirs(root_dir: Path) -> List[Path]:
            """Find all directories containing .jpg files"""
            class_dirs = []
            print("\nScanning directories:")
            for path in root_dir.iterdir():
                if not path.is_dir():
                    continue
                
                print(f"Checking directory: {path}")
                jpg_files = list(path.glob("*.jpg"))
                if jpg_files:
                    print(f"Found {len(jpg_files)} .jpg files in: {path}")
                    class_dirs.append(path)
                else:
                    print(f"No .jpg files found in: {path}")
                
            return class_dirs
        
        # Get all directories containing images
        class_dirs = get_class_dirs(self.root)
        
        if not class_dirs:
            raise RuntimeError(
                f"No directories with .jpg images found in {self.root}.\n"
                f"Expected structure: {self.root}/{{class_name}}/*.jpg"
            )
        
        self.classes = sorted([str(d.name) for d in class_dirs])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"\nFound {len(self.classes)} classes:")
        for cls in self.classes:
            print(f"  - {cls}")
        
        # Use the appropriate transform based on train/test mode
        transform_to_use = self.transform if self.train else self.test_transform
        if transform_to_use is None:
            raise ValueError("Transform must be provided for data loading")
        
        # Load all data
        all_data = []
        all_targets = []
        
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            jpg_files = list(class_dir.glob("*.jpg"))
            
            for img_path in jpg_files:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = transform_to_use(img)
                    img_array = img_tensor.permute(1, 2, 0).numpy() * 255
                    img_array = img_array.astype(np.uint8)
                    all_data.append(img_array)
                    all_targets.append(class_idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        if not all_data:
            raise RuntimeError(
                f"No valid .jpg images could be loaded from {self.root}\n"
                f"Please check that your .jpg images are in the correct location."
            )
        
        print(f"Loaded {len(all_data)} .jpg images across {len(self.classes)} classes")
        print(f"Images per class:")
        for cls in self.classes:
            count = sum(1 for t in all_targets if t == self.class_to_idx[cls])
            print(f"  - {cls}: {count} images")
            
        print(f"\nShape of first few arrays in all_data:")
        for i in range(min(5, len(all_data))):
            print(f"  {i}: {all_data[i].shape}")

        # Try to identify any inconsistent shapes
        shapes = [arr.shape for arr in all_data]
        unique_shapes = set(shapes)
        if len(unique_shapes) > 1:
            print("\nWarning: Inconsistent shapes detected:")
            for shape in unique_shapes:
                count = shapes.count(shape)
                print(f"  Shape {shape}: {count} images")
                # Print an example image path with this shape
                example_idx = shapes.index(shape)
                print(f"  Example: {list(self.root.glob('**/*.jpg'))[example_idx]}")

        print("\nAttempting to create numpy array...")
        self.data = np.array(all_data)
        self.targets = np.array(all_targets)
        print(f"Final array shape: {self.data.shape}")
        print(f"Targets shape: {self.targets.shape}")

    def concatenate(self, datas: List[np.ndarray], labels: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Concatenate multiple data and label arrays"""
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def getTestData(self, classes: Tuple[int, int]) -> None:
        """Get test data for specified classes"""
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TestData, self.TestLabels = self.concatenate(datas, labels)

    def getTrainData(self, classes: List[int], exemplar_set: List, exemplar_label_set: List) -> None:
        """Get training data for specified classes including exemplars"""
        datas, labels = [], []
        if len(exemplar_set) != 0 and len(exemplar_label_set) != 0:
            datas = [exemplar for exemplar in exemplar_set]
            length = len(datas[0])
            labels = [np.full((length), label) for label in exemplar_label_set]

        for label in classes:
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)

    def getSampleData(self, classes: List[int], exemplar_set: List, exemplar_label_set: List, group: int) -> None:
        """Get sample data for specified classes and group"""
        datas, labels = [], []
        if len(exemplar_set) != 0 and len(exemplar_label_set) != 0:
            datas = [exemplar for exemplar in exemplar_set]
            length = len(datas[0])
            labels = [np.full((length), label) for label in exemplar_label_set]

        if group == 0:
            for label in classes:
                data = self.data[np.array(self.targets) == label]
                datas.append(data)
                labels.append(np.full((data.shape[0]), label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)

    def getTrainItem(self, index: int) -> Tuple[int, Image.Image, int]:
        """Get a training item"""
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return index, img, target

    def getTestItem(self, index: int) -> Tuple[int, Image.Image, int]:
        """Get a test item"""
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]

        if self.test_transform:
            img = self.test_transform(img)

        if self.target_test_transform:
            target = self.target_test_transform(target)

        return index, img, target

    def __getitem__(self, index: int) -> Tuple[int, Image.Image, int]:
        """Get an item from either training or test set"""
        if isinstance(self.TrainData, np.ndarray) and self.TrainData.size > 0:
            return self.getTrainItem(index)
        elif isinstance(self.TestData, np.ndarray) and self.TestData.size > 0:
            return self.getTestItem(index)
        else:
            raise ValueError("No data available")

    def __len__(self) -> int:
        """Get length of current dataset (train or test)"""
        if isinstance(self.TrainData, np.ndarray) and self.TrainData.size > 0:
            return len(self.TrainData)
        elif isinstance(self.TestData, np.ndarray) and self.TestData.size > 0:
            return len(self.TestData)
        return 0

    def get_image_class(self, label: int) -> np.ndarray:
        """Get all images of a specific class"""
        return self.data[np.array(self.targets) == label]