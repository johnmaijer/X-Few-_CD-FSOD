import json
import os
import cv2
import torch
from detectron2.structures import Boxes, Instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from tqdm import tqdm


class JsonVisualizerWithImageInfo:
    def __init__(self,
                 results_json,
                 images_json,
                 images_base_dir,
                 output_dir,
                 dataset_name="coco"):
        """
        参数说明：
        results_json: 检测结果JSON路径
        images_json: 图像信息JSON路径
        images_base_dir: 图像基础目录
        output_dir: 输出目录
        dataset_name: 数据集元数据名称
        """
        # 初始化路径参数
        self.images_base_dir = images_base_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 加载元数据
        self.metadata = MetadataCatalog.get(dataset_name)

        # 加载检测结果
        with open(results_json) as f:
            self.all_results = json.load(f)

        # 加载图像信息并创建映射表
        with open(images_json) as f:
            images_info = json.load(f)
        images = images_info['images']
        self.image_info_map = {img["id"]: img for img in images}

        # 按image_id分组检测结果
        self._group_results_by_image()

    def _group_results_by_image(self):
        """将检测结果按image_id分组"""
        self.grouped_results = {}
        for res in self.all_results:
            img_id = res["image_id"]
            if img_id not in self.grouped_results:
                self.grouped_results[img_id] = []
            self.grouped_results[img_id].append(res)

    def _create_instances(self, results, image_shape):
        """创建Detectron2 Instances对象"""
        instances = Instances(image_shape)

        boxes = []
        scores = []
        classes = []

        for res in results:
            # 转换bbox格式（支持XYWH和XYXY）
            bbox = res["bbox"]
            x, y, w, h = bbox
            x1, y1 = x, y
            x2, y2 = x + w, y + h

            boxes.append([x1, y1, x2, y2])
            scores.append(res["score"])
            classes.append(res["category_id"])

        instances.pred_boxes = Boxes(torch.tensor(boxes))
        instances.scores = torch.tensor(scores)
        instances.pred_classes = torch.tensor(classes)
        return instances

    def _get_image_path(self, img_id):
        """获取完整图像路径"""
        img_info = self.image_info_map.get(img_id)
        if not img_info:
            return None

        # 拼接完整路径
        return os.path.join(self.images_base_dir, img_info["file_name"])

    def visualize_all(self, confidence_threshold=0.3):
        """执行批量可视化"""
        print(f"开始可视化处理，共{len(self.grouped_results)}张图片...")

        for img_id, results in tqdm(self.grouped_results.items()):
            # 过滤低置信度结果
            filtered_results = [
                res for res in results
                if res["score"] >= confidence_threshold
            ]
            if not filtered_results:
                continue

            # 获取图像路径
            img_path = self._get_image_path(img_id)
            if not img_path or not os.path.exists(img_path):
                print(f"跳过未找到的图片：{img_id}")
                continue

            # 读取图像并验证尺寸
            image = cv2.imread(img_path)
            if image is None:
                print(f"无法读取图片：{img_path}")
                continue

            # 验证尺寸一致性
            img_info = self.image_info_map[img_id]
            h, w = image.shape[:2]
            if h != img_info["height"] or w != img_info["width"]:
                print(f"图片尺寸不匹配：{img_id} 预期({img_info['height']},{img_info['width']}) 实际({h},{w})")
                continue

            # 创建实例对象
            instances = self._create_instances(filtered_results, (h, w))

            # 创建可视化器
            vis = Visualizer(
                image[:, :, ::-1],  # BGR转RGB
                metadata=self.metadata,
                instance_mode=ColorMode.IMAGE
            )

            # 绘制检测结果
            vis_output = vis.draw_instance_predictions(instances)

            # 保存结果
            output_path = os.path.join(
                self.output_dir,
                f"vis_{os.path.basename(img_info['file_name'])}"
            )
            cv2.imwrite(output_path, vis_output.get_image()[:, :, ::-1])  # 转回BGR保存

        print(f"可视化完成！结果保存在 {self.output_dir}")


# 使用示例 ---------------------------------------------------
if __name__ == "__main__":
    # 配置参数
    config = {
        "results_json": "/root/autodl-tmp/CDFSOD-benchmark-main/output/dataset1_1shot.json",
        "images_json": "/root/autodl-tmp/CDFSOD-benchmark-main/datasets/dataset1/annotations/test.json",
        "images_base_dir": "/root/autodl-tmp/CDFSOD-benchmark-main/datasets/dataset1/test",
        "output_dir": "/root/autodl-tmp/CDFSOD-benchmark-main/visual_result",
        "dataset_name": "dataset1"
    }

    # 初始化可视化器
    visualizer = JsonVisualizerWithImageInfo(**config)

    # 执行可视化（可设置置信度阈值）
    visualizer.visualize_all(confidence_threshold=0.30)
