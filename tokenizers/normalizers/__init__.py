from .. import normalizers
Normalizer = normalizers.Normalizer
BertNormalizer = normalizers.BertNormalizer
NFD = normalizers.NFD
NFKD = normalizers.NFKD
NFC = normalizers.NFC
NFKC = normalizers.NFKC
Sequence = normalizers.Sequence
Lowercase = normalizers.Lowercase
Strip = normalizers.Strip
StripAccents = normalizers.StripAccents
Nmt = normalizers.Nmt
Precompiled = normalizers.Precompiled
Replace = normalizers.Replace
NORMALIZERS = {'nfc': NFC, 'nfd': NFD, 'nfkc': NFKC, 'nfkd': NFKD}

def unicode_normalizer_from_str(normalizer: str) -> Normalizer:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('tokenizers.normalizers.__init__.unicode_normalizer_from_str', 'unicode_normalizer_from_str(normalizer)', {'NORMALIZERS': NORMALIZERS, 'normalizer': normalizer}, 1)

