# Edge TPU Model Combinations (Coral USB Accelerator)

All combinations below exceed 8 MiB on-chip parameter cache available for co-compiled models, so without pruning at least one of the two models will stream parameters from off-chip memory. Sizes and Latencies are taken from https://www.coral.ai/models/.

## Vision Cascade: Detection First, Second Stage Processes ROI Crops

| Detection Model | Size | Latency | Second-Stage Model | Size | Latency | Total Size | Use Case |
|---|---|---|---|---|---|---|---|
| SSDLite MobileDet (320×320) | 5.1 MB | 9.1 ms | MoveNet Thunder (256×256) | 7.5 MB | 13.8 ms | 12.6 MB | Multi-person pose estimation (top-down): detect every person, then crop each one and feed to MoveNet for single-person pose. The most widely used pipeline for multi-person pose. |
| EfficientDet-Lite1 (384×384) | 7.6 MB | 56.3 ms | MobileNet V2 iNat plant (224×224) | 5.5 MB | 2.6 ms | 13.1 MB | Agriculture / botany identification: detect plants in the frame, then crop and classify across 2000+ species. Coral hosts the iNaturalist weights specifically for this pipeline. |
| SSD MobileNet V2 Face (320×320) | 6.7 MB | 5.2 ms | EfficientNet-EdgeTpu S (224×224) | 6.8 MB | 5.0 ms | 13.5 MB | Access control / attendance: face detector finds faces, then a transfer-learned EfficientNet identity classifier recognizes each person. |
| SSD MobileNet V1 (300×300) | 7.0 MB | 6.5 ms | MobileNet V2 iNat bird (224×224) | 4.1 MB | 2.6 ms | 11.1 MB | Smart bird-feeder camera: detect birds, then run fine-grained classification across 900+ species (birdwatching application). |

## Vision Parallel: Multiple Independent Perceptions on the Same Frame

| Model A | Size | Latency | Model B | Size | Latency | Total Size | Use Case |
|---|---|---|---|---|---|---|---|
| EfficientDet-Lite1 (384×384) | 7.6 MB | 56.3 ms | MobileNet v2 DeepLab v3, dm=1.0 (513×513) | 2.9 MB | 43.0 ms | 10.5 MB | Robotic navigation: detect obstacles (people, cars, furniture) plus semantic segmentation of ground / drivable area. |
| MoveNet Thunder (256×256) | 7.5 MB | 13.8 ms | MobileNet v2 DeepLab v3, dm=1.0 (513×513) | 2.9 MB | 43.0 ms | 10.4 MB | AR fitness coach: real-time single-person pose plus scene segmentation (walls, floor) used as anchors for AR overlays. |
| EfficientDet-Lite1 (384×384) | 7.6 MB | 56.3 ms | BodyPix MobileNet v1 (720×1280) | 2.3 MB | 38.8 ms | 9.9 MB | Video conferencing device: detect desk objects (cups, papers) for on-screen annotation plus person segmentation for background blur. |
| U-Net MobileNet v2 (256×256) | 7.3 MB | 29.0 ms | MoveNet Lightning (192×192) | 3.1 MB | 7.1 ms | 10.4 MB | Pet-interaction camera: pet semantic segmentation plus owner pose estimation, used to detect "owner playing with pet" scenes. |

---

Notes:
- Reported latency is per single inference of each model in isolation. When two models are co-compiled and at least one streams from off-chip memory, the per-inference latency of the streamed model increases (sometimes substantially) compared to the values above.
- The cascade pipelines have a total latency of roughly `T_detection + N × T_stage2`, where N is the number of detected ROIs. The parallel pipelines run the two models sequentially on the same TPU, so the per-frame latency is roughly `T_A + T_B`.
