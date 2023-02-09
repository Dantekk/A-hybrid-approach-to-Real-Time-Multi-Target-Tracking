# A hybrid approach to Real-Time Multi-Target Tracking
The work included in this repo was to develop a hybrid strategy for real-time multi-target tracking that combines effectively a classical
optical flow algorithm with a deep learning architecture, targeted to a human-crowd tracking system exhibiting a desirable trade-off between
performance in tracking precision and computational costs. </br>
Tha main keypoints of this work are :

- FairMOT Methodology for tracking pipeline
- ConvNeXt Tiny and EfficientNetB3 as feature extractor in encoding step
- Deformable convolution and FPN in decoding step
- Byte Methodology for improving association step
- Optical Flow (Lucas-Kanade method) for improving execution time
