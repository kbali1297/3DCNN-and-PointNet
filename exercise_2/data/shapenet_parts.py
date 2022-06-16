from pathlib import Path
import json

import numpy as np
import torch


class ShapeNetParts(torch.utils.data.Dataset):
    num_classes = 50  # We have 50 parts classes to segment
    num_points = 1024
    dataset_path = Path("exercise_2/data/shapenetcore_partanno_segmentation_benchmark_v0/")  # path to point cloud data
    class_name_mapping = json.loads(Path("exercise_2/data/shape_parts_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())
    part_id_to_overall_id = json.loads(Path.read_text(Path(__file__).parent.parent / 'data' / 'partid_to_overallid.json'))

    def __init__(self, split):
        assert split in ['train', 'val', 'overfit']

        self.items = Path(f"exercise_2/data/splits/shapenet_parts/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        item = self.items[index]

        pointcloud, segmentation_labels = ShapeNetParts.get_point_cloud_with_labels(item)

        return {
            'points': pointcloud,
            'segmentation_labels': segmentation_labels
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        batch['points'] = batch['points'].to(device)
        batch['segmentation_labels'] = batch['segmentation_labels'].to(device)

    @staticmethod
    def get_point_cloud_with_labels(shapenet_id):
        """
        Utility method for reading a ShapeNet point cloud from disk, reads points from pts files on disk as 3d numpy arrays, together with their per-point part labels
        :param shapenet_id: Shape ID of the form <shape_class>/<shape_identifier>, e.g. 03001627/f913501826c588e89753496ba23f2183
        :return: tuple: a numpy array representing the point cloud, in shape 3x1024, and the segmentation labels, as numpy array in shape 1024
        """
        category_id, shape_id = shapenet_id.split('/')

        # TODO: Load point cloud and segmentation labels, subsample to 1024 points. Make sure points and labels still correspond afterwards!
        # TODO: Important: Use ShapeNetParts.part_id_to_overall_id to convert the part labels you get from the .seg files from local to global ID as they start at 0 for each shape class whereas we want to predict the overall part class.
        # ShapeNetParts.part_id_to_overall_id converts an ID in form <shapenetclass_partlabel> to and integer representing the global part class id
        

        points = []
        seg_labels = []
        shape_id_pts = shape_id + ".pts"
        shape_id_seg = shape_id + ".seg"
        fptr_pts = open(ShapeNetParts.dataset_path / category_id / "points" / shape_id_pts, "r")
        fptr_seg = open(ShapeNetParts.dataset_path / category_id / "points_label" /shape_id_seg, "r")
        
        for line in fptr_pts:
            words = line.split()
            points.append([float(words[0]), float(words[1]), float(words[2])])
        for line in fptr_seg:
            words = line.split()
            seg_labels.append(int(words[0]))
        
        n_points = 1024
        point_cloud = np.zeros((3,n_points))
        segmentation_labels = np.zeros((n_points,))

        points = np.array(points).astype('float32').transpose(1,0)
        seg_labels = np.array(seg_labels).astype('int')
        
        seg_point_ids_dict = {}
        for i,label in enumerate(seg_labels):
            if label not in seg_point_ids_dict.keys():
                seg_point_ids_dict[label] = []
            seg_point_ids_dict[label].append(i)

        n_points_per_label = {}
        #print(np.sum(total_points))
        count = 0
        for i,label in enumerate(seg_point_ids_dict.keys()):
            n_points_per_label[label] = int(np.floor(n_points * len(seg_point_ids_dict[label]) / points.shape[1] ))
            count += n_points_per_label[label]

        extra_points = n_points - int(count)
        label_list = list(seg_point_ids_dict.keys())
        n_points_per_label_total = {}
        for label in label_list:
            n_points_per_label_total[label] = len(seg_point_ids_dict[label])
        #print('File Category: ',category_id, ' File : ',shape_id)
        #print('n_points originally',n_points_per_label_total)
        #print('n_points before : ',n_points_per_label)
        #print('Extra Points to be added:',extra_points)
        for i in range(extra_points):
            label = label_list[i % len(label_list)]
            n_points_per_label[label] += 1
        
        #print('n_points after:',n_points_per_label)

        temp = 0
        #print(seg_point_ids_dict)
        for label in label_list:
            indices = np.random.choice(seg_point_ids_dict[label],n_points_per_label[label],replace=True)
            point_cloud[:,temp:temp + n_points_per_label[label]] = points[:,indices]
            part_id = category_id + "_" + str(label)
            segmentation_labels[temp: temp + n_points_per_label[label]] = int(ShapeNetParts.part_id_to_overall_id[part_id]) 
            temp += n_points_per_label[label]

        point_cloud = np.array(point_cloud).astype('float32')
        segmentation_labels = np.array(segmentation_labels).astype('int64')
        return point_cloud, segmentation_labels
        


        
        





