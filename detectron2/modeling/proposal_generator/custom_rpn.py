from detectron2.modeling.proposal_generator.rpn import StandardRPNHead
import torch.nn as nn


class DropoutRPNHead(StandardRPNHead):
    def __init__(self, in_channels, num_anchors, box_dim=4, dropout_prob=0.2):
        super().__init__(in_channels, num_anchors, box_dim)
        # 在分类和回归分支前添加Dropout
        self.cls_dropout = nn.Dropout2d(p=dropout_prob)
        self.bbox_dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, features):
        logits = []
        bbox_reg = []
        for x in features:
            # 原始卷积层前向传播
            cls_feat = self.conv(x)
            reg_feat = self.conv(x)
            # 添加Dropout
            cls_feat = self.cls_dropout(cls_feat)
            reg_feat = self.bbox_dropout(reg_feat)
            # 分类和回归分支
            logits.append(self.cls_logits(cls_feat))
            bbox_reg.append(self.bbox_pred(reg_feat))
        return logits, bbox_reg