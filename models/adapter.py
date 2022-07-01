from .pvt import pvt_small as pvt_small_original
from .pvt_v2 import pvt_v2_b2 as pvt_v2_b2_original
from .swin_transformer import SwinTransformer


def adapt_kwargs(kwargs):
    kwargs['img_size'] = max(kwargs['image_size'])
    del kwargs['image_size']


def pvt_v2_b2(pretrained=False, **kwargs):
    adapt_kwargs(kwargs)
    return pvt_v2_b2_original(pretrained=False, **kwargs)

def swin_t(pretrained=False, **kwargs):
    adapt_kwargs(kwargs)
    return SwinTransformer(pretrained=False, **kwargs)