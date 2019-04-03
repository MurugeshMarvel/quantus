from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from utils.bbox import BBox
from utils.nms import NMS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RegionProposalNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self._features = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self._objectness = nn.Conv2d(in_channels=512, out_channels=18, kernel_size=1)
        self._transformer = nn.Conv2d(in_channels=512, out_channels=36, kernel_size=1)

    def forward(self, features, image_width, image_height):
        anchor_bboxes = RegionProposalNetwork._generate_anchors(image_width, image_height, num_x_anchors=features.shape[3], num_y_anchors=features.shape[2]).to(device)

        features = self._features(features)
        objectnesses = self._objectness(features)
        transformers = self._transformer(features)

        objectnesses = objectnesses.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        transformers = transformers.permute(0, 2, 3, 1).contiguous().view(-1, 4)

        proposal_bboxes = RegionProposalNetwork._generate_proposals(anchor_bboxes, objectnesses, transformers, image_width, image_height)

        proposal_bboxes = proposal_bboxes[:12000 if self.training else 6000]
        keep_indices = NMS.suppress(proposal_bboxes, threshold=0.7)
        proposal_bboxes = proposal_bboxes[keep_indices]
        proposal_bboxes = proposal_bboxes[:2000 if self.training else 300]

        return anchor_bboxes, objectnesses, transformers, proposal_bboxes

    def sample(self, anchor_bboxes, anchor_objectnesses, anchor_transformers, gt_bboxes,
               image_width, image_height):
        anchor_bboxes = anchor_bboxes.cpu()
        #print(gt_bboxes.shape)
        gt_bboxes = gt_bboxes.cpu()

        # remove cross-boundary
        boundary = torch.tensor(BBox(0, 0, image_width, image_height).tolist(), dtype=torch.float)
        inside_indices = BBox.inside(anchor_bboxes, boundary.unsqueeze(dim=0)).squeeze().nonzero().view(-1)

        anchor_bboxes = anchor_bboxes[inside_indices]
        anchor_objectnesses = anchor_objectnesses[inside_indices]
        anchor_transformers = anchor_transformers[inside_indices]

        # find labels for each `anchor_bboxes`
        labels = torch.ones(len(anchor_bboxes), dtype=torch.long) * -1
        ious = BBox.iou(anchor_bboxes, gt_bboxes)
        anchor_max_ious, anchor_assignments = ious.max(dim=1)
        gt_max_ious, gt_assignments = ious.max(dim=0)
        anchor_additions = (ious == gt_max_ious).nonzero()[:, 0]
        labels[anchor_max_ious < 0.3] = 0
        labels[anchor_additions] = 1
        labels[anchor_max_ious >= 0.7] = 1

        # select 256 samples
        fg_indices = (labels == 1).nonzero().view(-1)
        bg_indices = (labels == 0).nonzero().view(-1)
        fg_indices = fg_indices[torch.randperm(len(fg_indices))[:min(len(fg_indices), 128)]]
        bg_indices = bg_indices[torch.randperm(len(bg_indices))[:256 - len(fg_indices)]]
        select_indices = torch.cat([fg_indices, bg_indices])
        select_indices = select_indices[torch.randperm(len(select_indices))]

        gt_anchor_objectnesses = labels[select_indices]
        gt_bboxes = gt_bboxes[anchor_assignments[fg_indices]]
        anchor_bboxes = anchor_bboxes[fg_indices]
        gt_anchor_transformers = BBox.calc_transformer(anchor_bboxes, gt_bboxes)

        gt_anchor_objectnesses = gt_anchor_objectnesses.to(device)
        gt_anchor_transformers = gt_anchor_transformers.to(device)

        anchor_objectnesses = anchor_objectnesses[select_indices]
        anchor_transformers = anchor_transformers[fg_indices]

        return anchor_objectnesses, anchor_transformers, gt_anchor_objectnesses, gt_anchor_transformers

    def loss(self, anchor_objectnesses, anchor_transformers, gt_anchor_objectnesses, gt_anchor_transformers):
        cross_entropy = F.cross_entropy(input=anchor_objectnesses, target=gt_anchor_objectnesses)

        # NOTE: The default of `reduction` is `elementwise_mean`, which is divided by N x 4 (number of all elements), here we replaced by N for better performance
        smooth_l1_loss = F.smooth_l1_loss(input=anchor_transformers, target=gt_anchor_transformers, reduction='sum')
        smooth_l1_loss /= len(gt_anchor_transformers)

        return cross_entropy, smooth_l1_loss

    @staticmethod
    def _generate_anchors(image_width, image_height, num_x_anchors, num_y_anchors):
        center_based_anchor_bboxes = []

        # NOTE: it's important to let `anchor_y` be the major index of list (i.e., move horizontally and then vertically) for consistency with 2D convolution
        for anchor_y in np.linspace(start=0, stop=image_height, num=num_y_anchors + 2)[1:-1]:    # remove anchor at vertical boundary
            for anchor_x in np.linspace(start=0, stop=image_width, num=num_x_anchors + 2)[1:-1]:  # remove anchor at horizontal boundary
                for ratio in [(1, 2), (1, 1), (2, 1)]:
                    for size in [128, 256, 512]:
                        center_x = float(anchor_x)
                        center_y = float(anchor_y)
                        r = ratio[0] / ratio[1]
                        height = size * np.sqrt(r)
                        width = size * np.sqrt(1 / r)
                        center_based_anchor_bboxes.append([center_x, center_y, width, height])

        center_based_anchor_bboxes = torch.tensor(center_based_anchor_bboxes, dtype=torch.float)
        anchor_bboxes = BBox.from_center_base(center_based_anchor_bboxes)

        return anchor_bboxes

    @staticmethod
    def _generate_proposals(anchor_bboxes, objectnesses, transformers, image_width, image_height):
        proposal_score = objectnesses[:, 1]
        _, sorted_indices = torch.sort(proposal_score, dim=0, descending=True)
        #print(len(sorted_indices))
        #print(anchor_bboxes.shape)
        sorted_transformers = transformers[sorted_indices]
        sorted_anchor_bboxes = anchor_bboxes[sorted_indices]

        proposal_bboxes = BBox.apply_transformer(sorted_anchor_bboxes, sorted_transformers.detach())
        proposal_bboxes = BBox.clip(proposal_bboxes, 0, 0, image_width, image_height)

        area_threshold = 16
        non_small_area_indices = ((proposal_bboxes[:, 2] - proposal_bboxes[:, 0] >= area_threshold) &
                                  (proposal_bboxes[:, 3] - proposal_bboxes[:, 1] >= area_threshold)).nonzero().view(-1)
        proposal_bboxes = proposal_bboxes[non_small_area_indices]

        return proposal_bboxes
