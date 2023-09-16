import os.path
import torch
import numpy as np
import tqdm
from monai.metrics.meandice import DiceMetric
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.visualize.utils import blend_images
import nibabel as nib
import glob
import json
from monai.transforms import AsDiscrete

labels_Synapse = {'Synapse': ['spleen', 'rkid', 'lkid', 'gall', 'liver', 'sto', 'aorta', 'pancreas']}
metrics_example = ['dice', 'HD95']


def analyze_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default=r"\path\to\images", type=str, help="images dir path")
    parser.add_argument("--predict_path", default=r"\path\to\predict", type=str,
                        help="predict dir path")
    parser.add_argument("--label_path", default=r"\path\to\labels", type=str,
                        help="ground truth dir path")
    parser.add_argument("--dataset_json", default=r"\path\to\dataset_json", type=str,
                        help="dataset config json file path")
    parser.add_argument("--save_path", default=r"./output", type=str,
                        help="where to save")
    parser.add_argument("--num_classes", default=14, type=int,
                        help="number of classes")
    parser.add_argument("--labels", default=None, help="if None, select all labels defined in dataset json file")
    parser.add_argument("--dataset_name", default="Synapse", help="name of dataset")
    parser.add_argument("--img_end", default=".nii.gz", help="image filename extension")
    parser.add_argument("--label_end", default=".nii.gz", help="label filename extension")
    parser.add_argument("--predict_end", default=".nii.gz", help="predict filename extension")
    args = parser.parse_args()
    labels = labels_Synapse if (args.dataset_name == "Synapse" and args.labels) is None else None
    metrics = metrics_example
    analyze = Analyze(image_path=args.image_path, predict_path=args.predict_path,
                      label_path=args.label_path, dataset_json=args.dataset_json, save_path=args.save_path,
                      num_classes=args.num_classes, labels=labels, ref_name=args.dataset_name,
                      img_end=args.img_end, label_end=args.label_end, predict_end=args.predict_end)
    analyze.compute_metrics(metrics)


class Analyze:
    def __init__(self, image_path,
                 predict_path,
                 label_path,
                 dataset_json,
                 save_path,
                 num_classes: int = 14,
                 labels=None,
                 ref_name='Synapse',
                 img_end=".nii.gz",
                 label_end=".nii.gz",
                 predict_end=".nii.gz"):
        # ref = nib.load(os.path.join(predict_path, ref_name)).get_fdta()
        # self.dataset_json = dataset_json
        self.ref = ref_name
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.save_path = save_path
        with open(dataset_json, 'r', encoding='utf-8') as f:
            self.json_file = json.load(f)
        self.label_names = list(self.json_file['labels'].keys())[1:]
        self.labels_index = range(len(self.label_names))
        if labels is not None:
            self.labels_index = [self.json_file['labels'][label] - 1 for label in labels]
            self.label_names = labels
        self.label_files = None
        self.image_files = None
        self.predict_files = None
        self.num_classes = num_classes
        self.image_path = image_path
        self.predict_path = predict_path
        self.predict_filenames = glob.glob(os.path.join(self.predict_path, "*" + predict_end))
        self.predict_fienames = self.predict_filenames.sort()
        self.image_filenames = [os.path.join(self.image_path, ((file.split('/')[-1]).split(predict_end)[0])
                                             + "_0000" + img_end)
                                for file in self.predict_filenames]
        self.label_filenames = [(file.replace(predict_path, label_path)).replace(predict_end, label_end)
                                for file in self.predict_filenames]
        self.label_path = label_path
        self.img_end = img_end
        self.label_end = label_end
        self.predict_end = predict_end
        self.loaded = False

    def load(self):
        def reader(ends):
            if ends in [".nii.gz", ".nii"]:
                return nib.load, "get_fdata"
            if ends in [".npy", ".npz"]:
                return np.load, None

        predict_loader, p_flag = reader(self.predict_end)
        image_loader, i_flag = reader(self.img_end)
        label_loader, l_flag = reader(self.label_end)
        self.predict_files = [predict_loader(filename).get_fdata() if p_flag is not None
                              else predict_loader(filename)
                              for filename in self.predict_filenames]
        self.image_files = [image_loader(filename).get_fdata() if i_flag is not None
                            else image_loader(filename)
                            for filename in self.image_filenames]
        self.label_files = [label_loader(filename).get_fdata() if l_flag is not None
                            else label_loader(filename)
                            for filename in self.label_filenames]
        self.loaded = True

    def compute_metrics(self, metrics, *args, **kwargs):
        if not self.loaded:
            self.load()
        ont_hot_transform = AsDiscrete(to_onehot=self.num_classes)
        predict_tensors = [ont_hot_transform(torch.from_numpy(file).permute(2, 0, 1)[None, :, :, :])
                           [None, :, :, :, :]  # B,C,D,H,W
                           for file in self.predict_files]
        label_tensors = [ont_hot_transform(torch.from_numpy(file).permute(2, 0, 1)[None, :, :, :])
                         [None, :, :, :, :]  # B,C,D,H,W
                         for file in self.label_files]
        valid_metrics = ['dice', 'HD', 'HD95']
        metrics_map = {'dice': DiceMetric(include_background=False, num_classes=self.num_classes, *args, **kwargs),
                       'HD': HausdorffDistanceMetric(include_background=False, percentile=100, *args, **kwargs),
                       'HD95': HausdorffDistanceMetric(include_background=False, percentile=95, *args, **kwargs)}
        result_map = {}
        for metirc in metrics:
            assert metirc in valid_metrics, "have not supported yet"
            result_map[metirc] = {}
            func = metrics_map[metirc]
            metric_result_map = result_map[metirc]
            func_results = []
            func_means = []
            for i in tqdm.tqdm(range(len(predict_tensors))):
                name = self.predict_filenames[i].split('/')[-1]
                result = func(predict_tensors[i], label_tensors[i])
                func_results.append(result)
                func_means.append(result.nanmean())
                list_result = [float(i) if not torch.isnan(i) else 0.0 for i in result.squeeze()]
                metric_result_map[name] = dict(zip(self.label_names, list_result))
            func_results = torch.cat(func_results, dim=0)
            func_results = torch.cat([(func_results[:, index])[:, None] for index in self.labels_index], dim=1)
            per_label_mean = torch.nanmean(func_results, dim=0)
            metric_result_map['per_label_mean'] = dict(zip(self.label_names,
                                                           [float(i) if not torch.isnan(i) else 0.0 for i in
                                                            per_label_mean]))
            metric_result_map['foreground_mean'] = float(torch.nanmean(per_label_mean, dim=0))
        with open(os.path.join(self.save_path, 'Analyze' + self.ref + '.json'), "w", encoding='utf-8') as f:
            json.dump(result_map, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    analyze_entry()