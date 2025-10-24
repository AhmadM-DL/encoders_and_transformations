import imgaug.augmenters as iaa
import numpy as np

def get_transformation(transformation_obj):
    transformation_name = transformation_obj["id"]
    # Create the base transformation
    if transformation_name == "blur":
        sigma_min = transformation_obj.get("sigma_min")
        sigma_max = transformation_obj.get("sigma_max")
        base_transform = iaa.GaussianBlur(sigma=(sigma_min, sigma_max))
    elif transformation_name == "noise":
        scale = transformation_obj.get("scale")
        per_channel = transformation_obj.get("per_channel")
        base_transform = iaa.AdditiveGaussianNoise(scale=scale, per_channel=per_channel)
    elif transformation_name == "jigsaw":
        nb_rows = transformation_obj.get("nb_rows")
        nb_cols = transformation_obj.get("nb_cols")
        base_transform = iaa.Jigsaw(nb_rows=(2, nb_rows), nb_cols=(2, nb_cols))
    elif transformation_name == "emboss":
        alpha_min = transformation_obj.get("alpha_min")
        alpha_max = transformation_obj.get("alpha_max")
        strength_min = transformation_obj.get("strength_min")
        strength_max = transformation_obj.get("strength_max")
        base_transform = iaa.Emboss(alpha=(alpha_min, alpha_max), strength=(strength_min, strength_max))
    elif transformation_name == "saltandpepper":
        p_min = transformation_obj.get("p_min")
        p_max = transformation_obj.get("p_max")
        per_channel = transformation_obj.get("per_channel")
        base_transform = iaa.SaltAndPepper(p=(p_min, p_max), per_channel=per_channel)
    elif transformation_name == "coarsedropout":
        p_min = transformation_obj.get("p_min")
        p_max = transformation_obj.get("p_max")
        size_percent_min = transformation_obj.get("size_percent_min")
        size_percent_max = transformation_obj.get("size_percent_max")
        per_channel = transformation_obj.get("per_channel")
        base_transform = iaa.CoarseDropout(p=(p_min, p_max), size_percent=(size_percent_min, size_percent_max), per_channel=per_channel)
    elif transformation_name == "multiplyhue":
        hue_min = transformation_obj.get("hue_min")
        hue_max = transformation_obj.get("hue_max")
        base_transform = iaa.MultiplyHue((hue_min, hue_max))
    elif transformation_name == "affine":
        scale_min = transformation_obj.get("scale_min")
        scale_max = transformation_obj.get("scale_max")
        rotate_min = transformation_obj.get("rotate_min")
        rotate_max = transformation_obj.get("rotate_max")
        shear_min = transformation_obj.get("shear_min")
        shear_max = transformation_obj.get("shear_max")
        base_transform = iaa.Affine(scale=(scale_min, scale_max), rotate=(rotate_min, rotate_max), shear=(shear_min, shear_max))
    else:
        raise Exception(f"Transformation {transformation_name} is not supported yet!")
    return _wrap_transformation(base_transform)

def _wrap_transformation(transformation):
    def _wrapped_transform(images):
        if type(images)!= 'list':
            raise Exception("The transformation functions expect a list of images as an input.")
        for img in images:
            if type(img)!= 'numpy.ndarray':
                raise Exception("The transformation functions expect images to be represented as numpy ndarrays.")
            if np.any(img < 0) or np.any(img > 1):
                raise ValueError("The transformation functions expect images be scaled between 0 and 1.")
        images_uint8 = [(img * 255).astype(np.uint8) for img in images]
        transformed_images = transformation(images=images_uint8)
        transformed_images_float = transformed_images.astype(np.float32) / 255.0
        return transformed_images_float
    return _wrapped_transform
