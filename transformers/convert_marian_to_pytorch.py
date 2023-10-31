import argparse
import json
import os
import socket
import time
import warnings
from pathlib import Path
from typing import Dict, List, Union
from zipfile import ZipFile
import numpy as np
import torch
from tqdm import tqdm
from transformers import MarianConfig, MarianMTModel, MarianTokenizer
from transformers.hf_api import HfApi

def remove_suffix(text: str, suffix: str):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch.remove_suffix', 'remove_suffix(text, suffix)', {'text': text, 'suffix': suffix}, 1)

def remove_prefix(text: str, prefix: str):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch.remove_prefix', 'remove_prefix(text, prefix)', {'text': text, 'prefix': prefix}, 1)

def convert_encoder_layer(opus_dict, layer_prefix: str, converter: dict):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch.convert_encoder_layer', 'convert_encoder_layer(opus_dict, layer_prefix, converter)', {'remove_prefix': remove_prefix, 'torch': torch, 'opus_dict': opus_dict, 'layer_prefix': layer_prefix, 'converter': converter}, 1)

def load_layers_(layer_lst: torch.nn.ModuleList, opus_state: dict, converter, is_decoder=False):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch.load_layers_', 'load_layers_(layer_lst, opus_state, converter, is_decoder=False)', {'convert_encoder_layer': convert_encoder_layer, 'layer_lst': layer_lst, 'opus_state': opus_state, 'converter': converter, 'is_decoder': is_decoder, 'torch': torch}, 0)

def find_pretrained_model(src_lang: str, tgt_lang: str) -> List[str]:
    """Find models that can accept src_lang as input and return tgt_lang as output."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch.find_pretrained_model', 'find_pretrained_model(src_lang, tgt_lang)', {'HfApi': HfApi, 'remove_prefix': remove_prefix, 'src_lang': src_lang, 'tgt_lang': tgt_lang, 'List': List, 'str': str}, 1)

def add_emb_entries(wemb, final_bias, n_special_tokens=1):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch.add_emb_entries', 'add_emb_entries(wemb, final_bias, n_special_tokens=1)', {'np': np, 'wemb': wemb, 'final_bias': final_bias, 'n_special_tokens': n_special_tokens}, 2)

def _cast_yaml_str(v):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch._cast_yaml_str', '_cast_yaml_str(v)', {'v': v}, 1)

def cast_marian_config(raw_cfg: Dict[(str, str)]) -> Dict:
    return {k: _cast_yaml_str(v) for (k, v) in raw_cfg.items()}
CONFIG_KEY = 'special:model.yml'

def load_config_from_state_dict(opus_dict):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch.load_config_from_state_dict', 'load_config_from_state_dict(opus_dict)', {'CONFIG_KEY': CONFIG_KEY, 'cast_marian_config': cast_marian_config, 'opus_dict': opus_dict}, 1)

def find_model_file(dest_dir):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch.find_model_file', 'find_model_file(dest_dir)', {'Path': Path, 'dest_dir': dest_dir}, 1)
ROM_GROUP = 'fr+fr_BE+fr_CA+fr_FR+wa+frp+oc+ca+rm+lld+fur+lij+lmo+es+es_AR+es_CL+es_CO+es_CR+es_DO+es_EC+es_ES+es_GT+es_HN+es_MX+es_NI+es_PA+es_PE+es_PR+es_SV+es_UY+es_VE+pt+pt_br+pt_BR+pt_PT+gl+lad+an+mwl+it+it_IT+co+nap+scn+vec+sc+ro+la'
GROUPS = [('cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh', 'ZH'), (ROM_GROUP, 'ROMANCE'), ('de+nl+fy+af+da+fo+is+no+nb+nn+sv', 'NORTH_EU'), ('da+fo+is+no+nb+nn+sv', 'SCANDINAVIA'), ('se+sma+smj+smn+sms', 'SAMI'), ('nb_NO+nb+nn_NO+nn+nog+no_nb+no', 'NORWAY'), ('ga+cy+br+gd+kw+gv', 'CELTIC')]
GROUP_TO_OPUS_NAME = {'opus-mt-ZH-de': 'cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh-de', 'opus-mt-ZH-fi': 'cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh-fi', 'opus-mt-ZH-sv': 'cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh-sv', 'opus-mt-SCANDINAVIA-SCANDINAVIA': 'da+fo+is+no+nb+nn+sv-da+fo+is+no+nb+nn+sv', 'opus-mt-NORTH_EU-NORTH_EU': 'de+nl+fy+af+da+fo+is+no+nb+nn+sv-de+nl+fy+af+da+fo+is+no+nb+nn+sv', 'opus-mt-de-ZH': 'de-cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh', 'opus-mt-en_el_es_fi-en_el_es_fi': 'en+el+es+fi-en+el+es+fi', 'opus-mt-en-ROMANCE': 'en-fr+fr_BE+fr_CA+fr_FR+wa+frp+oc+ca+rm+lld+fur+lij+lmo+es+es_AR+es_CL+es_CO+es_CR+es_DO+es_EC+es_ES+es_GT+es_HN+es_MX+es_NI+es_PA+es_PE+es_PR+es_SV+es_UY+es_VE+pt+pt_br+pt_BR+pt_PT+gl+lad+an+mwl+it+it_IT+co+nap+scn+vec+sc+ro+la', 'opus-mt-en-CELTIC': 'en-ga+cy+br+gd+kw+gv', 'opus-mt-es-NORWAY': 'es-nb_NO+nb+nn_NO+nn+nog+no_nb+no', 'opus-mt-fi_nb_no_nn_ru_sv_en-SAMI': 'fi+nb+no+nn+ru+sv+en-se+sma+smj+smn+sms', 'opus-mt-fi-ZH': 'fi-cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh', 'opus-mt-fi-NORWAY': 'fi-nb_NO+nb+nn_NO+nn+nog+no_nb+no', 'opus-mt-ROMANCE-en': 'fr+fr_BE+fr_CA+fr_FR+wa+frp+oc+ca+rm+lld+fur+lij+lmo+es+es_AR+es_CL+es_CO+es_CR+es_DO+es_EC+es_ES+es_GT+es_HN+es_MX+es_NI+es_PA+es_PE+es_PR+es_SV+es_UY+es_VE+pt+pt_br+pt_BR+pt_PT+gl+lad+an+mwl+it+it_IT+co+nap+scn+vec+sc+ro+la-en', 'opus-mt-CELTIC-en': 'ga+cy+br+gd+kw+gv-en', 'opus-mt-sv-ZH': 'sv-cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh', 'opus-mt-sv-NORWAY': 'sv-nb_NO+nb+nn_NO+nn+nog+no_nb+no'}
OPUS_GITHUB_URL = 'https://github.com/Helsinki-NLP/OPUS-MT-train/blob/master/models/'
ORG_NAME = 'Helsinki-NLP/'

def convert_opus_name_to_hf_name(x):
    """For OPUS-MT-Train/ DEPRECATED"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch.convert_opus_name_to_hf_name', 'convert_opus_name_to_hf_name(x)', {'GROUPS': GROUPS, 'x': x}, 1)

def convert_hf_name_to_opus_name(hf_model_name):
    """Relies on the assumption that there are no language codes like pt_br in models that are not in
    GROUP_TO_OPUS_NAME."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch.convert_hf_name_to_opus_name', 'convert_hf_name_to_opus_name(hf_model_name)', {'remove_prefix': remove_prefix, 'ORG_NAME': ORG_NAME, 'GROUP_TO_OPUS_NAME': GROUP_TO_OPUS_NAME, 'hf_model_name': hf_model_name}, 1)

def get_system_metadata(repo_root):
    import git
    return dict(helsinki_git_sha=git.Repo(path=repo_root, search_parent_directories=True).head.object.hexsha, transformers_git_sha=git.Repo(path='.', search_parent_directories=True).head.object.hexsha, port_machine=socket.gethostname(), port_time=time.strftime('%Y-%m-%d-%H:%M'))
FRONT_MATTER_TEMPLATE = '---\nlanguage:\n{}\ntags:\n- translation\n\nlicense: apache-2.0\n---\n\n'
DEFAULT_REPO = 'Tatoeba-Challenge'
DEFAULT_MODEL_DIR = os.path.join(DEFAULT_REPO, 'models')

def write_model_card(hf_model_name: str, repo_root=DEFAULT_REPO, save_dir=Path('marian_converted'), dry_run=False, extra_metadata={}) -> str:
    """Copy the most recent model's readme section from opus, and add metadata.
    upload command: aws s3 sync model_card_dir s3://models.huggingface.co/bert/Helsinki-NLP/ --dryrun
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch.write_model_card', "write_model_card(hf_model_name, repo_root=DEFAULT_REPO, save_dir=Path('marian_converted'), dry_run=False, extra_metadata={})", {'remove_prefix': remove_prefix, 'ORG_NAME': ORG_NAME, 'convert_hf_name_to_opus_name': convert_hf_name_to_opus_name, 'Path': Path, 'get_system_metadata': get_system_metadata, 'FRONT_MATTER_TEMPLATE': FRONT_MATTER_TEMPLATE, 'hf_model_name': hf_model_name, 'repo_root': repo_root, 'save_dir': save_dir, 'dry_run': dry_run, 'extra_metadata': extra_metadata, 'DEFAULT_REPO': DEFAULT_REPO}, 2)

def make_registry(repo_path='Opus-MT-train/models'):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch.make_registry', "make_registry(repo_path='Opus-MT-train/models')", {'Path': Path, '_parse_readme': _parse_readme, 'repo_path': repo_path}, 1)

def convert_all_sentencepiece_models(model_list=None, repo_path=None, dest_dir=Path('marian_converted')):
    """Requires 300GB"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch.convert_all_sentencepiece_models', "convert_all_sentencepiece_models(model_list=None, repo_path=None, dest_dir=Path('marian_converted'))", {'Path': Path, 'make_registry': make_registry, 'tqdm': tqdm, 'os': os, 'download_and_unzip': download_and_unzip, 'convert_opus_name_to_hf_name': convert_opus_name_to_hf_name, 'convert': convert, 'model_list': model_list, 'repo_path': repo_path, 'dest_dir': dest_dir}, 1)

def lmap(f, x) -> List:
    return list(map(f, x))

def fetch_test_set(test_set_url):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch.fetch_test_set', 'fetch_test_set(test_set_url)', {'Path': Path, 'lmap': lmap, 'os': os, 'test_set_url': test_set_url}, 3)

def convert_whole_dir(path=Path('marian_ckpt/')):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch.convert_whole_dir', "convert_whole_dir(path=Path('marian_ckpt/'))", {'tqdm': tqdm, 'convert': convert, 'source_dir': source_dir, 'path': path}, 0)

def _parse_readme(lns):
    """Get link and metadata from opus model card equivalent."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch._parse_readme', '_parse_readme(lns)', {'lns': lns}, 1)

def save_tokenizer_config(dest_dir: Path):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch.save_tokenizer_config', 'save_tokenizer_config(dest_dir)', {'save_json': save_json, 'dest_dir': dest_dir}, 0)

def add_to_vocab_(vocab: Dict[(str, int)], special_tokens: List[str]):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch.add_to_vocab_', 'add_to_vocab_(vocab, special_tokens)', {'vocab': vocab, 'special_tokens': special_tokens, 'Dict': Dict, 'List': List, 'str': str}, 1)

def find_vocab_file(model_dir):
    return list(model_dir.glob('*vocab.yml'))[0]

def add_special_tokens_to_vocab(model_dir: Path) -> None:
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch.add_special_tokens_to_vocab', 'add_special_tokens_to_vocab(model_dir)', {'load_yaml': load_yaml, 'find_vocab_file': find_vocab_file, 'add_to_vocab_': add_to_vocab_, 'save_json': save_json, 'save_tokenizer_config': save_tokenizer_config, 'model_dir': model_dir}, 0)

def check_equal(marian_cfg, k1, k2):
    (v1, v2) = (marian_cfg[k1], marian_cfg[k2])
    assert v1 == v2, f'hparams {k1},{k2} differ: {v1} != {v2}'

def check_marian_cfg_assumptions(marian_cfg):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch.check_marian_cfg_assumptions', 'check_marian_cfg_assumptions(marian_cfg)', {'check_equal': check_equal, 'marian_cfg': marian_cfg}, 0)
BIAS_KEY = 'decoder_ff_logit_out_b'
BART_CONVERTER = {'self_Wq': 'self_attn.q_proj.weight', 'self_Wk': 'self_attn.k_proj.weight', 'self_Wv': 'self_attn.v_proj.weight', 'self_Wo': 'self_attn.out_proj.weight', 'self_bq': 'self_attn.q_proj.bias', 'self_bk': 'self_attn.k_proj.bias', 'self_bv': 'self_attn.v_proj.bias', 'self_bo': 'self_attn.out_proj.bias', 'self_Wo_ln_scale': 'self_attn_layer_norm.weight', 'self_Wo_ln_bias': 'self_attn_layer_norm.bias', 'ffn_W1': 'fc1.weight', 'ffn_b1': 'fc1.bias', 'ffn_W2': 'fc2.weight', 'ffn_b2': 'fc2.bias', 'ffn_ffn_ln_scale': 'final_layer_norm.weight', 'ffn_ffn_ln_bias': 'final_layer_norm.bias', 'context_Wk': 'encoder_attn.k_proj.weight', 'context_Wo': 'encoder_attn.out_proj.weight', 'context_Wq': 'encoder_attn.q_proj.weight', 'context_Wv': 'encoder_attn.v_proj.weight', 'context_bk': 'encoder_attn.k_proj.bias', 'context_bo': 'encoder_attn.out_proj.bias', 'context_bq': 'encoder_attn.q_proj.bias', 'context_bv': 'encoder_attn.v_proj.bias', 'context_Wo_ln_scale': 'encoder_attn_layer_norm.weight', 'context_Wo_ln_bias': 'encoder_attn_layer_norm.bias'}


class OpusState:
    
    def __init__(self, source_dir):
        npz_path = find_model_file(source_dir)
        self.state_dict = np.load(npz_path)
        cfg = load_config_from_state_dict(self.state_dict)
        assert cfg['dim-vocabs'][0] == cfg['dim-vocabs'][1]
        assert 'Wpos' not in self.state_dict, 'Wpos key in state dictionary'
        self.state_dict = dict(self.state_dict)
        (self.wemb, self.final_bias) = add_emb_entries(self.state_dict['Wemb'], self.state_dict[BIAS_KEY], 1)
        self.pad_token_id = self.wemb.shape[0] - 1
        cfg['vocab_size'] = self.pad_token_id + 1
        self.state_keys = list(self.state_dict.keys())
        assert 'Wtype' not in self.state_dict, 'Wtype key in state dictionary'
        self._check_layer_entries()
        self.source_dir = source_dir
        self.cfg = cfg
        (hidden_size, intermediate_shape) = self.state_dict['encoder_l1_ffn_W1'].shape
        assert hidden_size == cfg['dim-emb'] == 512, f"Hidden size {hidden_size} and configured size {cfg['dim_emb']} mismatched or not 512"
        decoder_yml = cast_marian_config(load_yaml(source_dir / 'decoder.yml'))
        check_marian_cfg_assumptions(cfg)
        self.hf_config = MarianConfig(vocab_size=cfg['vocab_size'], decoder_layers=cfg['dec-depth'], encoder_layers=cfg['enc-depth'], decoder_attention_heads=cfg['transformer-heads'], encoder_attention_heads=cfg['transformer-heads'], decoder_ffn_dim=cfg['transformer-dim-ffn'], encoder_ffn_dim=cfg['transformer-dim-ffn'], d_model=cfg['dim-emb'], activation_function=cfg['transformer-aan-activation'], pad_token_id=self.pad_token_id, eos_token_id=0, bos_token_id=0, max_position_embeddings=cfg['dim-emb'], scale_embedding=True, normalize_embedding='n' in cfg['transformer-preprocess'], static_position_embeddings=not cfg['transformer-train-position-embeddings'], dropout=0.1, num_beams=decoder_yml['beam-size'], decoder_start_token_id=self.pad_token_id, bad_words_ids=[[self.pad_token_id]], max_length=512)
    
    def _check_layer_entries(self):
        self.encoder_l1 = self.sub_keys('encoder_l1')
        self.decoder_l1 = self.sub_keys('decoder_l1')
        self.decoder_l2 = self.sub_keys('decoder_l2')
        if len(self.encoder_l1) != 16:
            warnings.warn(f'Expected 16 keys for each encoder layer, got {len(self.encoder_l1)}')
        if len(self.decoder_l1) != 26:
            warnings.warn(f'Expected 26 keys for each decoder layer, got {len(self.decoder_l1)}')
        if len(self.decoder_l2) != 26:
            warnings.warn(f'Expected 26 keys for each decoder layer, got {len(self.decoder_l1)}')
    
    @property
    def extra_keys(self):
        extra = []
        for k in self.state_keys:
            if (k.startswith('encoder_l') or k.startswith('decoder_l') or k in [CONFIG_KEY, 'Wemb', 'Wpos', 'decoder_ff_logit_out_b']):
                continue
            else:
                extra.append(k)
        return extra
    
    def sub_keys(self, layer_prefix):
        return [remove_prefix(k, layer_prefix) for k in self.state_dict if k.startswith(layer_prefix)]
    
    def load_marian_model(self) -> MarianMTModel:
        (state_dict, cfg) = (self.state_dict, self.hf_config)
        assert cfg.static_position_embeddings, 'config.static_position_embeddings should be True'
        model = MarianMTModel(cfg)
        assert 'hidden_size' not in cfg.to_dict()
        load_layers_(model.model.encoder.layers, state_dict, BART_CONVERTER)
        load_layers_(model.model.decoder.layers, state_dict, BART_CONVERTER, is_decoder=True)
        wemb_tensor = torch.nn.Parameter(torch.FloatTensor(self.wemb))
        bias_tensor = torch.nn.Parameter(torch.FloatTensor(self.final_bias))
        model.model.shared.weight = wemb_tensor
        model.model.encoder.embed_tokens = model.model.decoder.embed_tokens = model.model.shared
        model.final_logits_bias = bias_tensor
        if 'Wpos' in state_dict:
            print('Unexpected: got Wpos')
            wpos_tensor = torch.tensor(state_dict['Wpos'])
            model.model.encoder.embed_positions.weight = wpos_tensor
            model.model.decoder.embed_positions.weight = wpos_tensor
        if cfg.normalize_embedding:
            assert 'encoder_emb_ln_scale_pre' in state_dict
            raise NotImplementedError('Need to convert layernorm_embedding')
        assert not self.extra_keys, f'Failed to convert {self.extra_keys}'
        assert model.model.shared.padding_idx == self.pad_token_id, f'Padding tokens {model.model.shared.padding_idx} and {self.pad_token_id} mismatched'
        return model


def download_and_unzip(url, dest_dir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch.download_and_unzip', 'download_and_unzip(url, dest_dir)', {'unzip': unzip, 'os': os, 'url': url, 'dest_dir': dest_dir}, 0)

def convert(source_dir: Path, dest_dir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch.convert', 'convert(source_dir, dest_dir)', {'Path': Path, 'add_special_tokens_to_vocab': add_special_tokens_to_vocab, 'MarianTokenizer': MarianTokenizer, 'OpusState': OpusState, 'source_dir': source_dir, 'dest_dir': dest_dir}, 0)

def load_yaml(path):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_marian_to_pytorch.load_yaml', 'load_yaml(path)', {'path': path}, 1)

def save_json(content: Union[(Dict, List)], path: str) -> None:
    with open(path, 'w') as f:
        json.dump(content, f)

def unzip(zip_path: str, dest_dir: str) -> None:
    with ZipFile(zip_path, 'r') as zipObj:
        zipObj.extractall(dest_dir)
if __name__ == '__main__':
    '\n    Tatoeba conversion instructions in scripts/tatoeba/README.md\n    '
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='path to marian model sub dir', default='en-de')
    parser.add_argument('--dest', type=str, default=None, help='Path to the output PyTorch model.')
    args = parser.parse_args()
    source_dir = Path(args.src)
    assert source_dir.exists(), f'Source directory {source_dir} not found'
    dest_dir = (f'converted-{source_dir.name}' if args.dest is None else args.dest)
    convert(source_dir, dest_dir)

