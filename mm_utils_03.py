import torchvision.transforms as T
import urllib.request as urequest

from io import BytesIO
from PIL import Image as PImage, ImageDraw as PImageDraw, ImageFont as PImageFont

from torch import tensor


def show_object_predictions(img, predictions):
  font = PImageFont.load_default(12)
  dimg = img.copy()
  draw = PImageDraw.Draw(dimg)

  for obj in predictions:
    label = obj["label"]
    (x0, y0, x1, y1) = tuple(obj["box"].values())
    draw.rectangle((x0,y0,x1,y1),
                   outline=(10, 220, 10),
                   width=2)
    draw.rectangle((x0,y0-6,x0+6*len(label),y0+6), fill=(0,0,0,160))
    draw.text((x0,y0-6), label, font=font)

  return dimg


def image_from_url(url):
  with urequest.urlopen(url) as response:
    image_data = BytesIO(response.read())
    return PImage.open(image_data)

try:
  from ultralytics.data.dataset import ClassificationDataset
  from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator

except:
  print("no YOLO found")


class CustomizedDataset(ClassificationDataset):
  def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
    super().__init__(root, args, augment, prefix)

    # custom training transforms here
    train_transforms = T.Compose(
      [
        T.Resize((args.imgsz, args.imgsz)),
        T.RandomHorizontalFlip(p=args.fliplr),
        T.RandomVerticalFlip(p=args.flipud),
        T.RandAugment(interpolation=T.InterpolationMode.BILINEAR),
        T.ColorJitter(brightness=args.hsv_v, contrast=args.hsv_v, saturation=args.hsv_s, hue=args.hsv_h),
        T.ToTensor(),
        T.Normalize(mean=tensor(0), std=tensor(1)),
        T.RandomErasing(p=args.erasing, inplace=True),
      ]
    )

    # custom validation transforms here
    val_transforms = T.Compose(
      [
        T.Resize((args.imgsz, args.imgsz)),
        T.ToTensor(),
        T.Normalize(mean=tensor(0), std=tensor(1)),
      ]
    )
    self.torch_transforms = train_transforms if augment else val_transforms


class CustomizedTrainer(ClassificationTrainer):
  def build_dataset(self, img_path: str, mode: str = "train", batch=None):
    return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)


class CustomizedValidator(ClassificationValidator):
  def build_dataset(self, img_path: str, mode: str = "train"):
    return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=self.args.split)
