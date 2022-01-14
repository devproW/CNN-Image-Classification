# Building an image classification model
This is a code to build a simple fruit classifier using CNN model. There are 3 classes of images in the dataset, which are apple, grapes, and lemon.

The model are written in pytorch.

---

## Environment
* Windows 10
* Python 3.7

```bash
# Install packages
pip install -r requirements.txt
```

---

## Tensorboard

Visualise the score using tensorboard.

```bash
# Visualise score
tensorboard --logdir logs/fruits_classifier --port 6060
```