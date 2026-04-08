from rfdetr import RFDETRMedium


DATASET_PATH = "/home/nssd/gled/vb/dataset-action4/"

DATASET_PATH = "datasets-coco-split/"

MYMODEL_DIR = "./model_action4_20260408_02"

model = RFDETRMedium()
model.train(
    dataset_dir=DATASET_PATH,
    epochs=70,
    batch_size=6,
    grad_accum_steps=6,
    lr=1e-4,
    output_dir=MYMODEL_DIR,
    aug_config={
        "HorizontalFlip": {"p": 0.5},
        "Rotate": {"limit": 7, "p": 0.3},
        "GaussianBlur": {"p": 0.2},
    },
#    resume='model_action4_20260408_01/last.ckpt'
)
