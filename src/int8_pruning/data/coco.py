"""COCO split table shared by the detection families (effdet, ssdlite)."""

COCO_TRAIN_SUBSETS = {
    "train2017": ("annotations/instances_train2017.json", "train2017"),
    "minitrain": ("annotations/instances_minitrain2017.json", "train2017"),
    "val2017":   ("annotations/instances_val2017.json",   "val2017"),
}
