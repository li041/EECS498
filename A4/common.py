"""
This module contains classes and functions that are common across both, one-stage
and two-stage detector implementations. You have to implement some parts here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction


def hello_common():
    print("Hello from common.py!")


class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(pretrained=True)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Initialize additional Conv layers for FPN.                   #
        #                                                                    #
        # Create THREE "lateral" 1x1 conv layers to transform (c3, c4, c5)   #
        # such that they all end up with the same `out_channels`.            #
        # Then create THREE "output" 3x3 conv layers to transform the merged #
        # FPN features to output (p3, p4, p5) features.                      #
        # All conv layers must have stride=1 and padding such that features  #
        # do not get downsampled due to 3x3 convs.                           #
        #                                                                    #
        # HINT: You have to use `dummy_out_shapes` defined above to decide   #
        # the input/output channels of these layers.                         #
        ######################################################################
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.
        # Add THREE lateral 1x1 conv and THREE output 3x3 conv layers.
        self.fpn_params = nn.ModuleDict()

        # Replace "pass" statement with your code
        # three lateral 1x1 conv layers
        self.fpn_params["lateral_c3"] = nn.Conv2d(
            in_channels=dummy_out_shapes[0][1][1],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.fpn_params["lateral_c4"] = nn.Conv2d(
            in_channels=dummy_out_shapes[1][1][1],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.fpn_params["lateral_c5"] = nn.Conv2d(
            in_channels=dummy_out_shapes[2][1][1],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        # finally, three output 3x3 conv layers to reduce aliasing effects of upsampling
        # same padding 
        self.fpn_params["output_p3"] = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.fpn_params["output_p4"] = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.fpn_params["output_p5"] = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Fill output FPN features (p3, p4, p5) using RegNet features  #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################
        # Replace "pass" statement with your code
        # 详见[UMich EECS/lab/A4 Figure 3], 
        # p4 = lateral_c4(c4) + upsample(p5), 最后再经过output_p4(reduce aliasing effects of upsampling)
        fpn_feats["p5"] = self.fpn_params["output_p5"](self.fpn_params["lateral_c5"](backbone_feats["c5"]))
        fpn_feats["p4"] = self.fpn_params["output_p4"](self.fpn_params["lateral_c4"](backbone_feats["c4"]) + F.interpolate(fpn_feats["p5"], scale_factor=2, mode='nearest'))
        fpn_feats["p3"] = self.fpn_params["output_p3"](self.fpn_params["lateral_c3"](backbone_feats["c3"]) + F.interpolate(fpn_feats["p4"], scale_factor=2, mode='nearest'))
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        ######################################################################
        # TODO: Implement logic to get location co-ordinates below.          #
        ######################################################################
        # Replace "pass" statement with your code
        # the output is (H*W, 2) tensor, where each row is the center of the receptive field of the corresponding location(i, j) in the FPN feature map
        # where i = index // W, j = index % W
        location_coords[level_name] = torch.zeros((feat_shape[2] * feat_shape[3], 2), dtype=dtype, device=device)
        for i in range(feat_shape[2]):
            for j in range(feat_shape[3]):
                location_coords[level_name][i * feat_shape[3] + j] = (torch.tensor([i, j], dtype=dtype, device=device) + 0.5) * level_stride  
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None
    #############################################################################
    # TODO: Implement non-maximum suppression which iterates the following:     #
    #       1. Select the highest-scoring box among the remaining ones,         #
    #          which has not been chosen in this step before                    #
    #       2. Eliminate boxes with IoU > threshold                             #
    #       3. If any boxes remain, GOTO 1                                      #
    #       Your implementation should not depend on a specific device type;    #
    #       you can use the device of the input if necessary.                   #
    # HINT: You can refer to the torchvision library code:                      #
    # github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
    #############################################################################
    # Replace "pass" statement with your code
    ''' 
    # naive implementation(already passed the test), but not efficient
    # 没有用到tensor, 在CPU的速度比GPU的快
    num_to_keep = 0
    N = scores.shape[0]
    # keep足够长, 最后根据num_to_keep截取
    keep = torch.zeros(N, dtype=torch.long, device=scores.device)
    suppressed = torch.zeros(N, dtype=torch.bool, device=scores.device)
    # 返回的是index
    order = scores.argsort(descending=True)
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    # areas.shape: (N, )
    areas = (x2 - x1) * (y2 - y1) 
    for _i in range(N):
        i = order[_i] 
        if suppressed[i] == True:
            continue
        keep[num_to_keep] = i
        num_to_keep += 1
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i+1, N):
            j = order[_j]
            if suppressed[j] == True:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])  
            yy2 = min(iy2, y2[j])  
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w * h
            iou = inter / (iarea + areas[j] - inter)
            if iou > iou_threshold:
                suppressed[j] = True
    keep = keep[:num_to_keep]
    '''
    # vectorized implementation(already passed the test)
    # run on GPU
    num_to_keep = 0
    N = scores.shape[0]
    # keep足够长, 最后根据num_to_keep截取
    keep = torch.zeros(N, dtype=torch.long, device=scores.device)
    sorted_scores, indices = scores.sort(descending=True)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    while indices.numel() > 0:
        i = indices[0]
        keep[num_to_keep] = i
        num_to_keep += 1
        if indices.numel() == 1:
            break
        xx1 = torch.max(x1[i], x1[indices[1:]])
        yy1 = torch.max(y1[i], y1[indices[1:]])
        xx2 = torch.min(x2[i], x2[indices[1:]])
        yy2 = torch.min(y2[i], y2[indices[1:]])
        ws = torch.max(torch.tensor(0, dtype=boxes.dtype, device=boxes.device), xx2 - xx1)
        hs = torch.max(torch.tensor(0, dtype=boxes.dtype, device=boxes.device), yy2 - yy1)
        inter = ws * hs
        ious = inter / (areas[i] + areas[indices[1:]] - inter)
        indices = indices[1:][ious <= iou_threshold]
    keep = keep[:num_to_keep]
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return keep


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
