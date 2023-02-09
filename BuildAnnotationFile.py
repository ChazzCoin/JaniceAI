# Create Annotations of Each Class
import json
import os

import numpy as np
from PIL import Image
from pycocotools import coco
from pycocotools.coco import COCO


def run_class(class_images_anns, class_name):
    class_annotaion = []
    class_image = []

    class_color = CATEGORY_COLOR[class_name]  # Get Color Tuple for the class
    for img in tqdm(class_images_anns):  # For each annotation in the JSON

        # The Image annotations has .jpg whereas actual Image is png and vice-versa.
        # try and except to get correct image accordingly
        try:
            I = Image.open(dataDir / class_name / 'images' / img['file_name'])
        except FileNotFoundError:
            if img['file_name'].endswith('.jpg'):
                I = Image.open(dataDir / class_name / 'images' / img['file_name'].replace('jpg', 'png'))
            elif img['file_name'].endswith('.png'):
                I = Image.open(dataDir / class_name / 'images' / img['file_name'].replace('png', 'jpg'))

        # Convert any grayscale or RBGA to RGB
        I = I.convert('RGB')
        og_h, og_w = I.size

        # Get Annotation of the Image
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        # Create Polygon for custom Annotation.
        og_poly = gen_original_poly(anns)

        # Get DETR output on our Image w.r.t COCO classes
        trans_img = transform(I).unsqueeze(0)
        out = model(trans_img.to('cpu'))

        # Create Masks by stacking Attention Maps and Pasting our Annotation
        # Excluding the functions definition for brevity. Can be found from colab link.
        class_masks = generate_class_maps(out)
        pred_mask, color2class = generate_pred_masks(class_masks, out['pred_masks'].shape[2:])

        pred_mask = cv2.resize(pred_mask, (og_h, og_w), interpolation=cv2.INTER_NEAREST)
        # Pasting Our Class on Mask
        for op in og_poly:
            cv2.fillPoly(pred_mask, pts=np.int32([op.get_xy()]), color=class_color)

        # Convering Mask to ID using panopticapi.utils
        mask_id = rgb2id(pred_mask)

        # Final Segmentation Details
        segments_info = generate_gt(mask_id, color2class, class_name)

        # The ID image(1 Channel) converted to 3 Channel Mask to save.
        img_save = Image.fromarray(id2rgb(mask_id))
        mask_file_name = img['file_name'].split('.')[0] + '.png'
        img_save.save(dataDir / class_name / 'annotations' / mask_file_name)

        # Appending the Image Annotation to List
        class_annotaion.append(
            {
                "segments_info": segments_info,
                "file_name": mask_file_name,
                "image_id": int(img['id'])
            }
        )

    return class_annotaion, class_image


for class_name in list_class:  # Loop over all the classes names

    annFile = "< images_dir > / class_name / 'coco.json'"  # Path to the annotations file of each class
    coco = COCO(annFile)  # Convert JSON to coco object (pycocotools.COCO)
    cats = coco.loadCats(coco.getCatIds())

    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=[class_name])
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)

    try:
        os.mkdir( " images_dir  / class_name / annotations ")  # Create Annotations Folder for each Class
    except FileExistsError as e:
        print('WARNING!', e)

    CLASS_ANNOTATION = run_class(images, class_name)  # Generate Annotations for each class

    FINAL_JSON = {}

    FINAL_JSON['licenses'] = coco.dataset['licenses']
    FINAL_JSON['info'] = coco.dataset['info']
    FINAL_JSON['categories'] = CATEGORIES
    FINAL_JSON['images'] = coco.dataset['images']
    FINAL_JSON['annotations'] = CLASS_ANNOTATION

    out_file = open( "< images_dir > / class_name / 'annotations' / f'{class_name}.json'", "w")
    json.dump(FINAL_JSON, out_file, indent=4)
    out_file.close()