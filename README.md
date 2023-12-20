# (Code in preparation)FDANet: A flow-deformation-aware point cloud completion network for industrial 3D plastic bent tubes.
A deeper understanding of the material plastic flow mechanism during CCSC manufacturing can help enhance the ability to learn from point clouds and achieve precise completion. A flow-deformation-aware point cloud completion network named FDANet is proposed for the industrial 3D plastic bent tube. Our insight of revealing geometrical details is to exploit the knowledge of cross-sectional distortions during the physical manufacturing process to fully learn and preserve the geometric information mapped by the material plastic flow for fine-grained completion of global shapes and locally deformed sections. FDANet introduces the attention mechanism in dynamic graph convolution to learn the interaction relationship of neighborhood nodes for feature extraction. An approximate prediction model of cross-section distortion for the spatial tube is proposed to inject the section-dimensional features into FDANet. We further design a global attention (GA) module in the transformer encoder to adaptively integrate the sectional points into the high-level features and summarize multi-hierarchical geometrical information.
![Graphic abstract](https://github.com/wangle0816/FDANet/assets/74782237/10fe5b6c-171e-4bd6-8302-1ef4969b1ee8)

## Usage
### Requirements
--PyTorch >= 1.7.0\
--python >= 3.7\
--CUDA >= 11.0\
--numpy\
--open3d\
--timm\
--torchvision
#### Pytorch Extensions
--PointNet++\
--KNN_CUDA\
--Chamfer Distance\
###Dataset
--Diverse bent tube dataset <https://drive.google.com/drive/folders/1gTOKAKfxA2WRqMTwwZcVaatI-N3Ce2el?usp=drive_link> \
--ShapeNet dataset <https://pan.baidu.com/s/1MavAO_GHa0a6BZh4Oaogug> <password:3hoe> \
--MVP dataset <https://drive.google.com/drive/folders/1XxZ4M_dOB3_OG1J6PnpNvrGTie5X9Vk_> \

(The code is being prepared for open source...)
