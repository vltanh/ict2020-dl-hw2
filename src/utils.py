def crop_object(image, box):
    """Crops an object in an image

    Inputs:
      image: PIL image
      box: one box from Detectron2 pred_boxes
    """

    x_top_left = box[0]
    y_top_left = box[1]
    x_bottom_right = box[2]
    y_bottom_right = box[3]

    crop_img = image.crop((
        int(x_top_left), int(y_top_left),
        int(x_bottom_right), int(y_bottom_right)
    ))
    return crop_img
