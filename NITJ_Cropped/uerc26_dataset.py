import os
import csv
import pathlib
import random
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


# Base dataset class for UERC, providing common functionality for data handling.
# Updated for NITJ_Cropped dataset structure:
#   Cropped/<subject_id>/L/Angle_variation/*.tiff
#   Cropped/<subject_id>/R/Angle_variation/*.tiff
#
# L and R are transparent — both are treated as images of the same subject.
# Subject identity is determined solely by the numeric <subject_id> folder.
class UERCBaseDataset(Dataset):
    def __init__(self, split, root_dir, full_image_list_csv, data_split_csv,
                 val_ratio=0.2, test_ratio=0.1):
        self.split = split
        self.root_dir = root_dir
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        if full_image_list_csv is None:
            raise ValueError("full_image_list_csv must be provided.")
        self.full_image_list_csv = os.path.join(self.root_dir, full_image_list_csv)

        if data_split_csv is None:
            raise ValueError("data_split_csv must be provided.")
        self.data_split_csv = os.path.join(self.root_dir, data_split_csv)

        # Check if the full image list CSV file exists, if not create it
        if not os.path.exists(self.full_image_list_csv):
            print(f"Full image list CSV file '{self.full_image_list_csv}' not found. Creating it...")
            self.__create_image_list()

        # Check if the data splits CSV file exists, if not create it
        if not os.path.exists(self.data_split_csv):
            print(f"Data splits CSV file '{self.data_split_csv}' not found. Creating it...")
            self.__create_data_split()

        self.images = self.__get_split_images()

        # Subject = first path component (numeric subject ID), L/R ignored
        subjects = set(img.replace('\\', '/').split('/')[0] for img in self.images)
        classes = sorted(subjects, key=lambda x: int(x) if x.isdigit() else x)

        self.classes_mapping = {c: i for i, c in enumerate(classes)}
        self.num_of_classes = len(self.classes_mapping)

        # Map each image path to its subject label (L/R ignored)
        self.labels = {}
        for img in self.images:
            subject = img.replace('\\', '/').split('/')[0]
            self.labels[img] = self.classes_mapping[subject]

        self.transforms = {}
        self.transforms['train'] = transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomRotation(30),
                                transforms.CenterCrop(224),
                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transforms['test'] = transforms.Compose([
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transforms['val'] = self.transforms['test']

    def __create_image_list(self):
        """Creates a CSV file with all image paths.

        Scans both L and R subdirectories — L/R is irrelevant to identity.
        Structure: <root_dir>/<subject_id>/<L|R>/Angle_variation/<file>.tiff
        Each CSV row stores the relative path from root_dir.
        """
        root = pathlib.Path(self.root_dir)

        # Scan both L and R — L/R distinction is ignored for labeling
        patterns = [
            '*/L/Angle_variation/*.tiff',
            '*/R/Angle_variation/*.tiff',
            '*/L/Angle_variation/*.png',
            '*/R/Angle_variation/*.png',
            '*/L/Angle_variation/*.jpg',
            '*/R/Angle_variation/*.jpg',
        ]

        img_list = []
        for pattern in patterns:
            img_list.extend(root.glob(pattern))

        img_list = sorted(set(img_list))

        with open(self.full_image_list_csv, "w", newline="") as csv_file:
            cw = csv.writer(csv_file, delimiter="\t")
            for image_path in img_list:
                # Store relative path using forward slashes
                rel_path = image_path.relative_to(root).as_posix()
                cw.writerow([rel_path])

        print(f"Created image list CSV with {len(img_list)} images: '{self.full_image_list_csv}'")

    def __get_full_image_list(self):
        """Parses the CSV file into a dict mapping subject_id -> [image paths].

        Subject key = numeric folder (first path component), L/R ignored.
        """
        full_image_list = {}
        with open(self.full_image_list_csv, "r") as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                img_pth = row[0]
                # Subject is the numeric top-level folder only
                subject = img_pth.replace('\\', '/').split('/')[0]

                if subject not in full_image_list:
                    full_image_list[subject] = []
                full_image_list[subject].append(img_pth)

        return full_image_list

    def __create_data_split(self):
        """Splits dataset into train/val/test.

        Test split: whole subjects held out.
        Val/train split: per-image within each subject.
        All images of a subject (L + R combined) stay together at subject level.
        """
        full_image_list = self.__get_full_image_list()
        subjects = list(full_image_list.keys())

        num_test_subjects = max(1, round(len(subjects) * self.test_ratio))

        random.shuffle(subjects)
        subjects_test = subjects[:num_test_subjects]
        subjects_train_val = subjects[num_test_subjects:]

        data = []

        for subject in subjects_train_val:
            images = full_image_list[subject]
            random.shuffle(images)

            num_val = max(1, round(len(images) * self.val_ratio))

            for img in images[:num_val]:
                data.append([img, 'val'])
            for img in images[num_val:]:
                data.append([img, 'train'])

        for subject in subjects_test:
            for img in full_image_list[subject]:
                data.append([img, 'test'])

        data = sorted(data)

        with open(self.data_split_csv, "w", newline="") as csv_file:
            cw = csv.writer(csv_file, delimiter="\t")
            for d in data:
                cw.writerow(d)

        n_train = sum(1 for d in data if d[1] == 'train')
        n_val   = sum(1 for d in data if d[1] == 'val')
        n_test  = sum(1 for d in data if d[1] == 'test')
        print(f"Created data split CSV: {len(data)} images total -- "
              f"train={n_train}, val={n_val}, test={n_test}")
        print(f"  Saved to: '{self.data_split_csv}'")

    def __get_split_images(self):
        """Reads the split CSV and returns images for the current split."""
        images = []
        with open(self.data_split_csv, "r") as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                if row[1] == self.split:
                    images.append(row[0])
        return images

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


# Dataset class for the identification task
class UERCDataset(UERCBaseDataset):
    def __init__(self, split, root_dir, full_image_list_csv, data_split_csv,
                 preload_images=False):
        super(UERCDataset, self).__init__(split, root_dir,
                                          full_image_list_csv, data_split_csv)
        self.preload_images = preload_images

        if self.preload_images:
            self.preloaded_images = {}
            for img_name in self.images:
                img_path = os.path.join(self.root_dir, img_name)
                image = Image.open(img_path).convert('RGB')
                self.preloaded_images[img_name] = image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)

        if self.preload_images:
            image = self.preloaded_images[img_name]
        else:
            image = Image.open(img_path).convert('RGB')

        if self.transforms[self.split]:
            image = self.transforms[self.split](image)

        label = self.labels[img_name]
        return image, label


# Dataset class for verification task (pairwise image comparison)
class UERCPairwiseDataset(UERCBaseDataset):
    def __init__(self, split, root_dir, full_image_list_csv, data_split_csv,
                 pairs_csv, num_pairs_per_subject=0, preload_images=False):
        super(UERCPairwiseDataset, self).__init__(split, root_dir,
                                                   full_image_list_csv, data_split_csv)

        if pairs_csv is None:
            raise ValueError("pairs_csv must be provided.")
        self.pairs_csv = os.path.join(self.root_dir, pairs_csv)

        if not os.path.exists(self.pairs_csv):
            self.__create_pairwise_pairs(num_pairs_per_subject)

        self.preload_images = preload_images
        if self.preload_images:
            self.preloaded_images = {}
            for img_name in self.images:
                img_path = os.path.join(self.root_dir, img_name)
                image = Image.open(img_path).convert('RGB')
                self.preloaded_images[img_name] = image

        self.image_pairs, self.labels = self.__load_pairwise_pairs()

    def __create_pairwise_pairs(self, num_pairs_per_subject):
        """Creates positive and negative image pairs.

        Positive pairs: two images of the same subject (L+R mixed — same identity).
        Negative pairs: two images from different subjects.
        """
        pairs = []
        labels = []

        for subject in self.classes_mapping.keys():
            images = [img for img in self.images
                      if self.labels[img] == self.classes_mapping[subject]]

            if len(images) <= 0:
                continue

            all_positive_pairs = [(images[i], images[j])
                                  for i in range(len(images))
                                  for j in range(i + 1, len(images))]

            if num_pairs_per_subject > 0 and len(images) > 1:
                selected_positive_pairs = random.sample(
                    all_positive_pairs,
                    min(num_pairs_per_subject, len(all_positive_pairs)))
            else:
                selected_positive_pairs = all_positive_pairs

            num_negative_pairs = (num_pairs_per_subject
                                  if num_pairs_per_subject > 0
                                  else len(selected_positive_pairs))

            if num_negative_pairs <= 0:
                continue

            other_subjects = [s for s in self.classes_mapping.keys() if s != subject]
            selected_negative_pairs = []
            attempts = 0
            max_attempts = num_negative_pairs * 20
            while len(selected_negative_pairs) < num_negative_pairs and attempts < max_attempts:
                attempts += 1
                other_subject = random.choice(other_subjects)
                other_images = [img for img in self.images
                                if self.labels[img] == self.classes_mapping[other_subject]]
                if not other_images:
                    continue
                img1 = random.choice(images)
                img2 = random.choice(other_images)
                pair = (img1, img2)
                if pair not in selected_negative_pairs:
                    selected_negative_pairs.append(pair)

            pairs.extend(selected_positive_pairs)
            labels.extend([1] * len(selected_positive_pairs))
            pairs.extend(selected_negative_pairs)
            labels.extend([0] * len(selected_negative_pairs))

        pairs_labels = sorted(zip(pairs, labels), key=lambda x: (x[0][0], x[0][1]))

        with open(self.pairs_csv, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            for pair, label in pairs_labels:
                img1, img2 = pair
                writer.writerow([img1, img2, label])

        n_pos = sum(1 for _, lbl in pairs_labels if lbl == 1)
        n_neg = sum(1 for _, lbl in pairs_labels if lbl == 0)
        print(f"Created pairs CSV: {len(pairs_labels)} pairs "
              f"(positive={n_pos}, negative={n_neg}) -> '{self.pairs_csv}'")

    def __load_pairwise_pairs(self):
        """Reads the pairs CSV and returns image pairs and labels."""
        pairs = []
        labels = []
        with open(self.pairs_csv, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                img1, img2, label = row
                pairs.append((img1, img2))
                labels.append(int(label))
        return pairs, labels

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        (img1_name, img2_name), label = self.image_pairs[idx], self.labels[idx]
        img1_path = os.path.join(self.root_dir, img1_name)
        img2_path = os.path.join(self.root_dir, img2_name)

        if self.preload_images:
            img1 = self.preloaded_images[img1_name]
            img2 = self.preloaded_images[img2_name]
        else:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')

        transform = self.transforms[self.split]
        if transform:
            img1 = transform(img1)
            img2 = transform(img2)

        return (img1, img2), label


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # NITJ_Cropped dataset CSV generation
    # Root dir = 'Cropped' folder containing numeric subject subfolders
    # L and R are IGNORED for identity — subject = numeric ID only
    # Both L and R images of a subject belong to the same class
    # -----------------------------------------------------------------------
    ROOT = 'Cropped'

    print("=" * 60)
    print("Generating unified CSVs (L and R treated as same subject)")
    print("=" * 60)

    # -- Identification datasets --
    train_ds = UERCDataset(
        'train', root_dir=ROOT,
        full_image_list_csv='image_list.csv',
        data_split_csv='dataset_split.csv'
    )
    print(f'Train samples: {len(train_ds)}')

    val_ds = UERCDataset(
        'val', root_dir=ROOT,
        full_image_list_csv='image_list.csv',
        data_split_csv='dataset_split.csv'
    )
    print(f'Validation samples: {len(val_ds)}')

    test_ds = UERCDataset(
        'test', root_dir=ROOT,
        full_image_list_csv='image_list.csv',
        data_split_csv='dataset_split.csv'
    )
    print(f'Test samples: {len(test_ds)}')

    print()

    # -- Pairwise (verification) datasets --
    pairs_train = UERCPairwiseDataset(
        'train', root_dir=ROOT,
        full_image_list_csv='image_list.csv',
        data_split_csv='dataset_split.csv',
        pairs_csv='pairs_train.csv',
        num_pairs_per_subject=100
    )
    print(f'Pairwise train samples: {len(pairs_train)}')

    pairs_val = UERCPairwiseDataset(
        'val', root_dir=ROOT,
        full_image_list_csv='image_list.csv',
        data_split_csv='dataset_split.csv',
        pairs_csv='pairs_val.csv',
        num_pairs_per_subject=100
    )
    print(f'Pairwise val samples: {len(pairs_val)}')

    pairs_test = UERCPairwiseDataset(
        'test', root_dir=ROOT,
        full_image_list_csv='image_list.csv',
        data_split_csv='dataset_split.csv',
        pairs_csv='pairs_test.csv',
        num_pairs_per_subject=100
    )
    print(f'Pairwise test samples: {len(pairs_test)}')

    print()
    print("=" * 60)
    print("All CSVs generated successfully in:", ROOT)
    print("  image_list.csv      - All image paths (L+R combined)")
    print("  dataset_split.csv   - Train/val/test split")
    print("  pairs_train.csv     - Pairwise verification train pairs")
    print("  pairs_val.csv       - Pairwise verification val pairs")
    print("  pairs_test.csv      - Pairwise verification test pairs")
    print("=" * 60)
