import math
import numpy as np
from ._convert_np import make_np
from ._utils import make_grid
from tensorboard.compat import tf
from tensorboard.plugins.projector.projector_config_pb2 import EmbeddingInfo

def make_tsv(metadata, save_path, metadata_header=None):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.utils.tensorboard._embedding.make_tsv', 'make_tsv(metadata, save_path, metadata_header=None)', {'tf': tf, 'metadata': metadata, 'save_path': save_path, 'metadata_header': metadata_header}, 0)

def make_sprite(label_img, save_path):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.utils.tensorboard._embedding.make_sprite', 'make_sprite(label_img, save_path)', {'math': math, 'make_grid': make_grid, 'make_np': make_np, 'np': np, 'tf': tf, 'label_img': label_img, 'save_path': save_path}, 0)

def get_embedding_info(metadata, label_img, filesys, subdir, global_step, tag):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._embedding.get_embedding_info', 'get_embedding_info(metadata, label_img, filesys, subdir, global_step, tag)', {'EmbeddingInfo': EmbeddingInfo, 'metadata': metadata, 'label_img': label_img, 'filesys': filesys, 'subdir': subdir, 'global_step': global_step, 'tag': tag}, 1)

def write_pbtxt(save_path, contents):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.utils.tensorboard._embedding.write_pbtxt', 'write_pbtxt(save_path, contents)', {'tf': tf, 'save_path': save_path, 'contents': contents}, 0)

def make_mat(matlist, save_path):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.utils.tensorboard._embedding.make_mat', 'make_mat(matlist, save_path)', {'tf': tf, 'matlist': matlist, 'save_path': save_path}, 0)

