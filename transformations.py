import imgaug.augmenters as iaa

def get_transformation(transformation_obj):
    transformation_name = transformation_obj["id"]
    if transformation_name == "blur":
        sigma_min = transformation_obj.get("sigma_min")
        sigma_max = transformation_obj.get("sigma_max")
        return iaa.GaussianBlur(sigma=(sigma_min, sigma_max))
    elif transformation_name == "noise":
        scale = transformation_obj.get("scale")
        per_channel = transformation_obj.get("per_channel")
        return iaa.AdditiveGaussianNoise(scale=scale*255, per_channel=per_channel)
    elif transformation_name == "jigsaw":
        nb_rows = transformation_obj.get("nb_rows")
        nb_cols = transformation_obj.get("nb_cols")
        return iaa.Jigsaw(nb_rows=(2, nb_rows), nb_cols=(2, nb_cols))
    elif transformation_name == "emboss":
        alpha_min = transformation_obj.get("alpha_min")
        alpha_max = transformation_obj.get("alpha_max")
        strength_min = transformation_obj.get("strength_min")
        strength_max = transformation_obj.get("strength_max")
        return iaa.Emboss(alpha=(alpha_min, alpha_max), strength=(strength_min, strength_max))
    elif transformation_name == "saltandpepper":
        p_min = transformation_obj.get("p_min")
        p_max = transformation_obj.get("p_max")
        per_channel = transformation_obj.get("per_channel")
        return iaa.SaltAndPepper(p=(p_min, p_max), per_channel=per_channel)
    elif transformation_name == "coarsedropout":
        p_min = transformation_obj.get("p_min")
        p_max = transformation_obj.get("p_max")
        size_percent_min = transformation_obj.get("size_percent_min")
        size_percent_max = transformation_obj.get("size_percent_max")
        per_channel = transformation_obj.get("per_channel")
        return iaa.CoarseDropout(p=(p_min, p_max), size_percent=(size_percent_min, size_percent_max), per_channel=per_channel)
    elif transformation_name == "multiplyhue":
        hue_min = transformation_obj.get("hue_min")
        hue_max = transformation_obj.get("hue_max")
        return iaa.MultiplyHue((hue_min, hue_max))
    elif transformation_name == "affine":
        scale_min = transformation_obj.get("scale_min")
        scale_max = transformation_obj.get("scale_max")
        rotate_min = transformation_obj.get("rotate_min")
        rotate_max = transformation_obj.get("rotate_max")
        shear_min = transformation_obj.get("shear_min")
        shear_max = transformation_obj.get("shear_max")
        return iaa.Affine(scale=(scale_min, scale_max), rotate=(rotate_min, rotate_max), shear=(shear_min, shear_max))
    else:
        raise Exception(f"Transformation {transformation_name} is not supported yet!")