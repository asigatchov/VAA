from rfdetr import RFDETRMedium, RFDETRNano


DATASET_PATH = "/home/nssd/gled/vb/dataset-action4/"

DATASET_PATH = "datasets-coco-split/"

MYMODEL_DIR = "./model_action4_2026042_01"

model = RFDETRMedium()
#model = RFDETRNano()
model.train(
    dataset_dir=DATASET_PATH,
    epochs=70,
    batch_size=10,
    grad_accum_steps=10,
    lr=1e-4,
    output_dir=MYMODEL_DIR,
    aug_config={
        "HorizontalFlip": {"p": 0.5},
        "Rotate": {"limit": 7, "p": 0.3},
        "GaussianBlur": {"p": 0.2},
    },
#    resume='model_action4_20260408_01/last.ckpt'
)
