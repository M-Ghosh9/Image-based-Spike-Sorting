# Transfer-Learning-Spike-Sorting
Introduction to spike sorting using proposed image-based spike sorting to leverage the power of transfer learning of image processing techniques to signal processing problems in neuroscience. 

Proposed Architecture-

![image](https://github.com/user-attachments/assets/21f66b0d-756f-4898-ab43-f043ca392f61)

ResNet Architecture for Image-based Spike Sorting-
![image](https://github.com/user-attachments/assets/3978c2a5-c785-4243-af19-4d8860d4a0b5)

Advantages Over Other Models: - 
Better than K-means: ResNet learns complex spike features automatically, reducing the dependency on initial cluster choices. 
Better than 1D CNN: ResNet's deeper architecture captures intricate patterns, leading to higher accuracy. And deals with the vanishing gradient problem using residual connections.
Challenges:
Might lead to more use of memory when storing as images

Current Performance: The model is currently training with an accuracy of 77%. Manual inspection is still required to ensure correct labeling. 

