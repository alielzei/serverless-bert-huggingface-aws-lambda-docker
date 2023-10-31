from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError('Python 2.7 or later required')
if (__package__ or '.' in __name__):
    from . import _sentencepiece
else:
    import _sentencepiece
try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('sentencepiece.__init__._swig_repr', '_swig_repr(self)', {'__builtin__': __builtin__, 'self': self}, 1)

def _swig_setattr_nondynamic_instance_variable(set):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('sentencepiece.__init__._swig_setattr_nondynamic_instance_variable', '_swig_setattr_nondynamic_instance_variable(set)', {'set': set}, 1)

def _swig_setattr_nondynamic_class_variable(set):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('sentencepiece.__init__._swig_setattr_nondynamic_class_variable', '_swig_setattr_nondynamic_class_variable(set)', {'set': set}, 1)

def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('sentencepiece.__init__._swig_add_metaclass', '_swig_add_metaclass(metaclass)', {'metaclass': metaclass}, 1)


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)



class ImmutableSentencePieceText_ImmutableSentencePiece(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    
    def __init__(self):
        _sentencepiece.ImmutableSentencePieceText_ImmutableSentencePiece_swiginit(self, _sentencepiece.new_ImmutableSentencePieceText_ImmutableSentencePiece())
    __swig_destroy__ = _sentencepiece.delete_ImmutableSentencePieceText_ImmutableSentencePiece
    
    def _piece(self):
        return _sentencepiece.ImmutableSentencePieceText_ImmutableSentencePiece__piece(self)
    
    def _surface(self):
        return _sentencepiece.ImmutableSentencePieceText_ImmutableSentencePiece__surface(self)
    
    def _id(self):
        return _sentencepiece.ImmutableSentencePieceText_ImmutableSentencePiece__id(self)
    
    def _begin(self):
        return _sentencepiece.ImmutableSentencePieceText_ImmutableSentencePiece__begin(self)
    
    def _end(self):
        return _sentencepiece.ImmutableSentencePieceText_ImmutableSentencePiece__end(self)
    piece = property(_piece)
    surface = property(_surface)
    id = property(_id)
    begin = property(_begin)
    end = property(_end)
    
    def __str__(self):
        return 'piece: "{}"\nid: {}\nsurface: "{}"\nbegin: {}\nend: {}\n'.format(self.piece, self.id, self.surface, self.begin, self.end)
    
    def __eq__(self, other):
        return (self.piece == other.piece and self.id == other.id and self.surface == other.surface and self.begin == other.begin and self.end == other.end)
    
    def __hash__(self):
        return hash(str(self))
    __repr__ = __str__

_sentencepiece.ImmutableSentencePieceText_ImmutableSentencePiece_swigregister(ImmutableSentencePieceText_ImmutableSentencePiece)


class ImmutableSentencePieceText(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    
    def __init__(self):
        _sentencepiece.ImmutableSentencePieceText_swiginit(self, _sentencepiece.new_ImmutableSentencePieceText())
    __swig_destroy__ = _sentencepiece.delete_ImmutableSentencePieceText
    
    def _pieces_size(self):
        return _sentencepiece.ImmutableSentencePieceText__pieces_size(self)
    
    def _pieces(self, index):
        return _sentencepiece.ImmutableSentencePieceText__pieces(self, index)
    
    def _text(self):
        return _sentencepiece.ImmutableSentencePieceText__text(self)
    
    def _score(self):
        return _sentencepiece.ImmutableSentencePieceText__score(self)
    
    def SerializeAsString(self):
        return _sentencepiece.ImmutableSentencePieceText_SerializeAsString(self)
    text = property(_text)
    score = property(_score)
    
    
    class ImmutableSentencePieceIterator:
        
        def __init__(self, proto):
            self.proto = proto
            self.len = self.proto._pieces_size()
        
        def __len__(self):
            return self.len
        
        def __getitem__(self, index):
            if isinstance(index, slice):
                return [self.proto._pieces(i) for i in range(self.len)][index.start:index.stop:index.step]
            if index < 0:
                index = index + self.len
            if (index < 0 or index >= self.len):
                raise IndexError('piece index is out of range')
            return self.proto._pieces(index)
        
        def __str__(self):
            return '\n'.join(['pieces {{\n{}}}'.format(str(x)) for x in self])
        __repr__ = __str__
    
    
    @property
    def pieces(self):
        return ImmutableSentencePieceText.ImmutableSentencePieceIterator(self)
    
    def __eq__(self, other):
        return self.SerializeAsString() == other.SerializeAsString()
    
    def __hash__(self):
        return hash(self.SerializeAsString())
    
    def __str__(self):
        return 'text: "{}"\nscore: {}\n{}'.format(self.text, self.score, '\n'.join(['pieces {{\n{}}}'.format(str(x)) for x in self.pieces]))
    __repr__ = __str__

_sentencepiece.ImmutableSentencePieceText_swigregister(ImmutableSentencePieceText)


class ImmutableNBestSentencePieceText(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    
    def __init__(self):
        _sentencepiece.ImmutableNBestSentencePieceText_swiginit(self, _sentencepiece.new_ImmutableNBestSentencePieceText())
    __swig_destroy__ = _sentencepiece.delete_ImmutableNBestSentencePieceText
    
    def _nbests_size(self):
        return _sentencepiece.ImmutableNBestSentencePieceText__nbests_size(self)
    
    def _nbests(self, index):
        return _sentencepiece.ImmutableNBestSentencePieceText__nbests(self, index)
    
    def SerializeAsString(self):
        return _sentencepiece.ImmutableNBestSentencePieceText_SerializeAsString(self)
    
    
    class ImmutableSentencePieceTextIterator:
        
        def __init__(self, proto):
            self.proto = proto
            self.len = self.proto._nbests_size()
        
        def __len__(self):
            return self.len
        
        def __getitem__(self, index):
            if isinstance(index, slice):
                return [self.proto._nbests(i) for i in range(self.len)][index.start:index.stop:index.step]
            if index < 0:
                index = index + self.len
            if (index < 0 or index >= self.len):
                raise IndexError('nbests index is out of range')
            return self.proto._nbests(index)
        
        def __str__(self):
            return '\n'.join(['nbests {{\n{}}}'.format(str(x)) for x in self])
        __repr__ = __str__
    
    
    @property
    def nbests(self):
        return ImmutableNBestSentencePieceText.ImmutableSentencePieceTextIterator(self)
    
    def __eq__(self, other):
        return self.SerializeAsString() == other.SerializeAsString()
    
    def __hash__(self):
        return hash(self.SerializeAsString())
    
    def __str__(self):
        return '\n'.join(['nbests {{\n{}}}'.format(str(x)) for x in self.nbests])
    __repr__ = __str__

_sentencepiece.ImmutableNBestSentencePieceText_swigregister(ImmutableNBestSentencePieceText)


class SentencePieceProcessor(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    
    def __init__(self):
        _sentencepiece.SentencePieceProcessor_swiginit(self, _sentencepiece.new_SentencePieceProcessor())
    __swig_destroy__ = _sentencepiece.delete_SentencePieceProcessor
    
    def LoadFromSerializedProto(self, serialized):
        return _sentencepiece.SentencePieceProcessor_LoadFromSerializedProto(self, serialized)
    
    def SetEncodeExtraOptions(self, extra_option):
        return _sentencepiece.SentencePieceProcessor_SetEncodeExtraOptions(self, extra_option)
    
    def SetDecodeExtraOptions(self, extra_option):
        return _sentencepiece.SentencePieceProcessor_SetDecodeExtraOptions(self, extra_option)
    
    def SetVocabulary(self, valid_vocab):
        return _sentencepiece.SentencePieceProcessor_SetVocabulary(self, valid_vocab)
    
    def ResetVocabulary(self):
        return _sentencepiece.SentencePieceProcessor_ResetVocabulary(self)
    
    def LoadVocabulary(self, filename, threshold):
        return _sentencepiece.SentencePieceProcessor_LoadVocabulary(self, filename, threshold)
    
    def CalculateEntropy(self, *args):
        return _sentencepiece.SentencePieceProcessor_CalculateEntropy(self, *args)
    
    def GetPieceSize(self):
        return _sentencepiece.SentencePieceProcessor_GetPieceSize(self)
    
    def PieceToId(self, piece):
        return _sentencepiece.SentencePieceProcessor_PieceToId(self, piece)
    
    def IdToPiece(self, id):
        return _sentencepiece.SentencePieceProcessor_IdToPiece(self, id)
    
    def GetScore(self, id):
        return _sentencepiece.SentencePieceProcessor_GetScore(self, id)
    
    def IsUnknown(self, id):
        return _sentencepiece.SentencePieceProcessor_IsUnknown(self, id)
    
    def IsControl(self, id):
        return _sentencepiece.SentencePieceProcessor_IsControl(self, id)
    
    def IsUnused(self, id):
        return _sentencepiece.SentencePieceProcessor_IsUnused(self, id)
    
    def IsByte(self, id):
        return _sentencepiece.SentencePieceProcessor_IsByte(self, id)
    
    def unk_id(self):
        return _sentencepiece.SentencePieceProcessor_unk_id(self)
    
    def bos_id(self):
        return _sentencepiece.SentencePieceProcessor_bos_id(self)
    
    def eos_id(self):
        return _sentencepiece.SentencePieceProcessor_eos_id(self)
    
    def pad_id(self):
        return _sentencepiece.SentencePieceProcessor_pad_id(self)
    
    def serialized_model_proto(self):
        return _sentencepiece.SentencePieceProcessor_serialized_model_proto(self)
    
    def LoadFromFile(self, arg):
        return _sentencepiece.SentencePieceProcessor_LoadFromFile(self, arg)
    
    def _EncodeAsIds(self, text, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece):
        return _sentencepiece.SentencePieceProcessor__EncodeAsIds(self, text, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece)
    
    def _EncodeAsPieces(self, text, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece):
        return _sentencepiece.SentencePieceProcessor__EncodeAsPieces(self, text, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece)
    
    def _EncodeAsSerializedProto(self, text, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece):
        return _sentencepiece.SentencePieceProcessor__EncodeAsSerializedProto(self, text, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece)
    
    def _EncodeAsImmutableProto(self, text, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece):
        return _sentencepiece.SentencePieceProcessor__EncodeAsImmutableProto(self, text, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece)
    
    def _EncodeAsIdsBatch(self, ins, num_threads, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece):
        return _sentencepiece.SentencePieceProcessor__EncodeAsIdsBatch(self, ins, num_threads, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece)
    
    def _EncodeAsPiecesBatch(self, ins, num_threads, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece):
        return _sentencepiece.SentencePieceProcessor__EncodeAsPiecesBatch(self, ins, num_threads, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece)
    
    def _EncodeAsSerializedProtoBatch(self, ins, num_threads, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece):
        return _sentencepiece.SentencePieceProcessor__EncodeAsSerializedProtoBatch(self, ins, num_threads, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece)
    
    def _EncodeAsImmutableProtoBatch(self, ins, num_threads, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece):
        return _sentencepiece.SentencePieceProcessor__EncodeAsImmutableProtoBatch(self, ins, num_threads, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece)
    
    def _DecodeIds(self, ids):
        return _sentencepiece.SentencePieceProcessor__DecodeIds(self, ids)
    
    def _DecodePieces(self, pieces):
        return _sentencepiece.SentencePieceProcessor__DecodePieces(self, pieces)
    
    def _DecodeIdsAsSerializedProto(self, ids):
        return _sentencepiece.SentencePieceProcessor__DecodeIdsAsSerializedProto(self, ids)
    
    def _DecodePiecesAsSerializedProto(self, pieces):
        return _sentencepiece.SentencePieceProcessor__DecodePiecesAsSerializedProto(self, pieces)
    
    def _DecodeIdsAsImmutableProto(self, ids):
        return _sentencepiece.SentencePieceProcessor__DecodeIdsAsImmutableProto(self, ids)
    
    def _DecodePiecesAsImmutableProto(self, pieces):
        return _sentencepiece.SentencePieceProcessor__DecodePiecesAsImmutableProto(self, pieces)
    
    def _DecodeIdsBatch(self, ins, num_threads):
        return _sentencepiece.SentencePieceProcessor__DecodeIdsBatch(self, ins, num_threads)
    
    def _DecodeIdsAsSerializedProtoBatch(self, ins, num_threads):
        return _sentencepiece.SentencePieceProcessor__DecodeIdsAsSerializedProtoBatch(self, ins, num_threads)
    
    def _DecodeIdsAsImmutableProtoBatch(self, ins, num_threads):
        return _sentencepiece.SentencePieceProcessor__DecodeIdsAsImmutableProtoBatch(self, ins, num_threads)
    
    def _DecodePiecesBatch(self, ins, num_threads):
        return _sentencepiece.SentencePieceProcessor__DecodePiecesBatch(self, ins, num_threads)
    
    def _DecodePiecesAsSerializedProtoBatch(self, ins, num_threads):
        return _sentencepiece.SentencePieceProcessor__DecodePiecesAsSerializedProtoBatch(self, ins, num_threads)
    
    def _DecodePiecesAsImmutableProtoBatch(self, ins, num_threads):
        return _sentencepiece.SentencePieceProcessor__DecodePiecesAsImmutableProtoBatch(self, ins, num_threads)
    
    def _NBestEncodeAsIds(self, text, nbest_size, add_bos, add_eos, reverse, emit_unk_piece):
        return _sentencepiece.SentencePieceProcessor__NBestEncodeAsIds(self, text, nbest_size, add_bos, add_eos, reverse, emit_unk_piece)
    
    def _NBestEncodeAsPieces(self, text, nbest_size, add_bos, add_eos, reverse, emit_unk_piece):
        return _sentencepiece.SentencePieceProcessor__NBestEncodeAsPieces(self, text, nbest_size, add_bos, add_eos, reverse, emit_unk_piece)
    
    def _NBestEncodeAsSerializedProto(self, text, nbest_size, add_bos, add_eos, reverse, emit_unk_piece):
        return _sentencepiece.SentencePieceProcessor__NBestEncodeAsSerializedProto(self, text, nbest_size, add_bos, add_eos, reverse, emit_unk_piece)
    
    def _NBestEncodeAsImmutableProto(self, text, nbest_size, add_bos, add_eos, reverse, emit_unk_piece):
        return _sentencepiece.SentencePieceProcessor__NBestEncodeAsImmutableProto(self, text, nbest_size, add_bos, add_eos, reverse, emit_unk_piece)
    
    def _SampleEncodeAndScoreAsIds(self, text, num_samples, alpha, wor, include_best, add_bos, add_eos, reverse, emit_unk_piece):
        return _sentencepiece.SentencePieceProcessor__SampleEncodeAndScoreAsIds(self, text, num_samples, alpha, wor, include_best, add_bos, add_eos, reverse, emit_unk_piece)
    
    def _SampleEncodeAndScoreAsPieces(self, text, num_samples, alpha, wor, include_best, add_bos, add_eos, reverse, emit_unk_piece):
        return _sentencepiece.SentencePieceProcessor__SampleEncodeAndScoreAsPieces(self, text, num_samples, alpha, wor, include_best, add_bos, add_eos, reverse, emit_unk_piece)
    
    def _SampleEncodeAndScoreAsSerializedProto(self, text, num_samples, alpha, wor, include_best, add_bos, add_eos, reverse, emit_unk_piece):
        return _sentencepiece.SentencePieceProcessor__SampleEncodeAndScoreAsSerializedProto(self, text, num_samples, alpha, wor, include_best, add_bos, add_eos, reverse, emit_unk_piece)
    
    def _SampleEncodeAndScoreAsImmutableProto(self, text, num_samples, alpha, wor, include_best, add_bos, add_eos, reverse, emit_unk_piece):
        return _sentencepiece.SentencePieceProcessor__SampleEncodeAndScoreAsImmutableProto(self, text, num_samples, alpha, wor, include_best, add_bos, add_eos, reverse, emit_unk_piece)
    
    def _CalculateEntropy(self, text, alpha):
        return _sentencepiece.SentencePieceProcessor__CalculateEntropy(self, text, alpha)
    
    def _CalculateEntropyBatch(self, ins, alpha, num_threads):
        return _sentencepiece.SentencePieceProcessor__CalculateEntropyBatch(self, ins, alpha, num_threads)
    
    def Init(self, model_file=None, model_proto=None, out_type=int, add_bos=False, add_eos=False, reverse=False, emit_unk_piece=False, enable_sampling=False, nbest_size=-1, alpha=0.1, num_threads=-1):
        """Initialzie sentencepieceProcessor.

      Args:
        model_file: The sentencepiece model file path.
        model_proto: The sentencepiece model serialized proto.
        out_type: output type. int or str.
        add_bos: Add <s> to the result (Default = false)
        add_eos: Add </s> to the result (Default = false) <s>/</s> is added after
          reversing (if enabled).
        reverse: Reverses the tokenized sequence (Default = false)
        emit_unk_piece: Emits the unk literal string (Default = false)
        nbest_size: sampling parameters for unigram. Invalid in BPE-Dropout.
                    nbest_size = {0,1}: No sampling is performed.
                    nbest_size > 1: samples from the nbest_size results.
                    nbest_size < 0: assuming that nbest_size is infinite and samples
                      from the all hypothesis (lattice) using
                      forward-filtering-and-backward-sampling algorithm.
        alpha: Soothing parameter for unigram sampling, and dropout probability of
               merge operations for BPE-dropout.
        num_threads: number of threads in batch processing (Default = -1, auto-detected)
      """
        _sentencepiece_processor_init_native(self)
        self._out_type = out_type
        self._add_bos = add_bos
        self._add_eos = add_eos
        self._reverse = reverse
        self._emit_unk_piece = emit_unk_piece
        self._enable_sampling = enable_sampling
        self._nbest_size = nbest_size
        self._alpha = alpha
        self._num_threads = num_threads
        if (model_file or model_proto):
            self.Load(model_file=model_file, model_proto=model_proto)
    
    def Encode(self, input, out_type=None, add_bos=None, add_eos=None, reverse=None, emit_unk_piece=None, enable_sampling=None, nbest_size=None, alpha=None, num_threads=None):
        """Encode text input to segmented ids or tokens.

        Args:
        input: input string. accepsts list of string.
        out_type: output type. int or str.
        add_bos: Add <s> to the result (Default = false)
        add_eos: Add </s> to the result (Default = false) <s>/</s> is added after
                 reversing (if enabled).
        reverse: Reverses the tokenized sequence (Default = false)
        emit_unk_piece: Emits the unk literal string (Default = false)
        nbest_size: sampling parameters for unigram. Invalid in BPE-Dropout.
                    nbest_size = {0,1}: No sampling is performed.
                    nbest_size > 1: samples from the nbest_size results.
                    nbest_size < 0: assuming that nbest_size is infinite and samples
                    from the all hypothesis (lattice) using
                    forward-filtering-and-backward-sampling algorithm.
        alpha: Soothing parameter for unigram sampling, and merge probability for
               BPE-dropout (probablity 'p' in BPE-dropout paper).
        num_threads: the number of threads used in the batch processing (Default = -1).
      """
        if out_type is None:
            out_type = self._out_type
        if add_bos is None:
            add_bos = self._add_bos
        if add_eos is None:
            add_eos = self._add_eos
        if reverse is None:
            reverse = self._reverse
        if emit_unk_piece is None:
            emit_unk_piece = self._emit_unk_piece
        if enable_sampling is None:
            enable_sampling = self._enable_sampling
        if nbest_size is None:
            nbest_size = self._nbest_size
        if alpha is None:
            alpha = self._alpha
        if num_threads is None:
            num_threads = self._num_threads
        if (enable_sampling == True and ((nbest_size is None or nbest_size == 0 or nbest_size == 1 or alpha is None))):
            raise RuntimeError('When enable_sampling is True, We must specify "nbest_size > 1" or "nbest_size = -1", and "alpha". "nbest_size" is enabled only on unigram mode ignored in BPE-dropout. when "nbest_size = -1" , this method samples from all candidates on the lattice instead of nbest segmentations.')
        if (num_threads is None or type(num_threads) is not int):
            raise RuntimeError('num_threads must be int')
        if type(input) is list:
            if out_type is int:
                return self._EncodeAsIdsBatch(input, num_threads, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece)
            if out_type is str:
                return self._EncodeAsPiecesBatch(input, num_threads, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece)
            if (out_type == 'serialized_proto' or out_type == 'proto'):
                return self._EncodeAsSerializedProtoBatch(input, num_threads, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece)
            if out_type == 'immutable_proto':
                return self._EncodeAsImmutableProtoBatch(input, num_threads, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece)
        if out_type is int:
            return self._EncodeAsIds(input, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece)
        if out_type is str:
            return self._EncodeAsPieces(input, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece)
        if (out_type == 'serialized_proto' or out_type == 'proto'):
            return self._EncodeAsSerializedProto(input, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece)
        if out_type == 'immutable_proto':
            return self._EncodeAsImmutableProto(input, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece)
        raise RuntimeError('unknown out_type={}'.format(out_type))
        return None
    
    def EncodeAsPieces(self, input, **kwargs):
        return self.Encode(input=input, out_type=str, **kwargs)
    
    def EncodeAsIds(self, input, **kwargs):
        return self.Encode(input=input, out_type=int, **kwargs)
    
    def EncodeAsSerializedProto(self, input, **kwargs):
        return self.Encode(input=input, out_type='serialized_proto', **kwargs)
    
    def EncodeAsImmutableProto(self, input, **kwargs):
        return self.Encode(input=input, out_type='immutable_proto', **kwargs)
    
    def SampleEncodeAsPieces(self, input, nbest_size=None, alpha=None, **kwargs):
        return self.Encode(input=input, nbest_size=nbest_size, alpha=alpha, out_type=str, enable_sampling=True, **kwargs)
    
    def SampleEncodeAsIds(self, input, nbest_size=None, alpha=None, **kwargs):
        return self.Encode(input=input, nbest_size=nbest_size, alpha=alpha, out_type=int, enable_sampling=True, **kwargs)
    
    def SampleEncodeAsSerializedProto(self, input, nbest_size=None, alpha=None, **kwargs):
        return self.Encode(input=input, nbest_size=nbest_size, alpha=alpha, out_type='serialized_proto', enable_sampling=True, **kwargs)
    
    def SampleEncodeAsImmutableProto(self, input, nbest_size=None, alpha=None, **kwargs):
        return self.Encode(input=input, nbest_size=nbest_size, alpha=alpha, out_type='immutable_proto', enable_sampling=True, **kwargs)
    
    def NBestEncode(self, input, out_type=None, add_bos=None, add_eos=None, reverse=None, emit_unk_piece=None, nbest_size=None):
        """NBestEncode text input to segmented ids or tokens.

        Args:
        input: input string. accepsts list of string.
        out_type: output type. int or str.
        add_bos: Add <s> to the result (Default = false)
        add_eos: Add </s> to the result (Default = false) <s>/</s> is added after reversing (if enabled).
        reverse: Reverses the tokenized sequence (Default = false)
        emit_unk_piece: Emits the unk literal string (Default = false)
        nbest_size: nbest size
      """
        if out_type is None:
            out_type = self._out_type
        if add_bos is None:
            add_bos = self._add_bos
        if add_eos is None:
            add_eos = self._add_eos
        if reverse is None:
            reverse = self._reverse
        if emit_unk_piece is None:
            emit_unk_piece = self._emit_unk_piece
        if nbest_size is None:
            nbest_size = self._nbest_size
        if nbest_size <= 0:
            nbest_size = 1
        
        def _encode(text):
            if out_type is int:
                return self._NBestEncodeAsIds(text, nbest_size, add_bos, add_eos, reverse, emit_unk_piece)
            if out_type is str:
                return self._NBestEncodeAsPieces(text, nbest_size, add_bos, add_eos, reverse, emit_unk_piece)
            if (out_type == 'serialized_proto' or out_type == 'proto'):
                return self._NBestEncodeAsSerializedProto(text, nbest_size, add_bos, add_eos, reverse, emit_unk_piece)
            if out_type == 'immutable_proto':
                return self._NBestEncodeAsImmutableProto(text, nbest_size, add_bos, add_eos, reverse, emit_unk_piece)
            raise RuntimeError('unknown out_type')
        if type(input) is list:
            return [_encode(n) for n in input]
        return _encode(input)
    
    def NBestEncodeAsPieces(self, input, nbest_size=None, **kwargs):
        return self.NBestEncode(input=input, nbest_size=nbest_size, out_type=str, **kwargs)
    
    def NBestEncodeAsIds(self, input, nbest_size=None, **kwargs):
        return self.NBestEncode(input=input, nbest_size=nbest_size, out_type=int, **kwargs)
    
    def NBestEncodeAsSerializedProto(self, input, nbest_size=None, **kwargs):
        return self.NBestEncode(input=input, nbest_size=nbest_size, out_type='serialized_proto', **kwargs)
    
    def NBestEncodeAsImmutableProto(self, input, nbest_size=None, **kwargs):
        return self.NBestEncode(input=input, nbest_size=nbest_size, out_type='immutable_proto', **kwargs)
    
    def SampleEncodeAndScore(self, input, out_type=None, add_bos=None, add_eos=None, reverse=None, emit_unk_piece=None, num_samples=None, alpha=None, wor=None, include_best=None):
        """SampleEncodeAndScore text input to segmented ids or tokens.

        Args:
        input: input string. accepsts list of string.
        out_type: output type. int or str or 'serialized_proto' or 'immutable_proto'
        add_bos: Add <s> to the result (Default = false)
        add_eos: Add </s> to the result (Default = false) <s>/</s> is added after reversing (if enabled).
        reverse: Reverses the tokenized sequence (Default = false)
        emit_unk_piece: Emits the unk literal string (Default = false)
        num_samples: How many samples to return (Default = 1)
        alpha: inverse temperature for sampling
        wor: whether to sample without replacement (Default = false)
        include_best: whether to include the best tokenization, requires wor=True (Default = false)
      """
        if out_type is None:
            out_type = self._out_type
        if add_bos is None:
            add_bos = self._add_bos
        if add_eos is None:
            add_eos = self._add_eos
        if reverse is None:
            reverse = self._reverse
        if emit_unk_piece is None:
            emit_unk_piece = self._emit_unk_piece
        if num_samples is None:
            num_samples = 1
        if alpha is None:
            alpha = 1.0
        if wor is None:
            wor = False
        if include_best is None:
            include_best = False
        if num_samples <= 0:
            raise RuntimeError('num_examples must be positive')
        if (include_best and not wor):
            raise RuntimeError('When include_best is True, We must specify "wor = True".')
        
        def _encode(text):
            if out_type is int:
                return self._SampleEncodeAndScoreAsIds(text, num_samples, alpha, wor, include_best, add_bos, add_eos, reverse, emit_unk_piece)
            if out_type is str:
                return self._SampleEncodeAndScoreAsPieces(text, num_samples, alpha, wor, include_best, add_bos, add_eos, reverse, emit_unk_piece)
            if (out_type == 'serialized_proto' or out_type == 'proto'):
                return self._SampleEncodeAndScoreAsSerializedProto(text, num_samples, alpha, wor, include_best, add_bos, add_eos, reverse, emit_unk_piece)
            if out_type == 'immutable_proto':
                return self._SampleEncodeAndScoreAsImmutableProto(text, num_samples, alpha, wor, include_best, add_bos, add_eos, reverse, emit_unk_piece)
            raise RuntimeError('unknown output type')
        if type(input) is list:
            return [_encode(n) for n in input]
        return _encode(input)
    
    def SampleEncodeAndScoreAsPieces(self, input, num_samples=None, alpha=None, **kwargs):
        return self.SampleEncodeAndScore(input=input, num_samples=num_samples, alpha=alpha, out_type=str, **kwargs)
    
    def SampleEncodeAndScoreAsIds(self, input, num_samples=None, alpha=None, **kwargs):
        return self.SampleEncodeAndScore(input=input, num_samples=num_samples, alpha=alpha, out_type=int, **kwargs)
    
    def SampleEncodeAndScoreAsSerializedProto(self, input, num_samples=None, alpha=None, **kwargs):
        return self.SampleEncodeAndScore(input=input, num_samples=num_samples, alpha=alpha, out_type='serialized_proto', **kwargs)
    
    def SampleEncodeAndScoreAsImmutableProto(self, input, num_samples=None, alpha=None, **kwargs):
        return self.SampleEncodeAndScore(input=input, num_samples=num_samples, alpha=alpha, out_type='immutable_proto', **kwargs)
    
    def Decode(self, input, out_type=str, num_threads=None):
        """Decode processed id or token sequences.

      Args:
        out_type: output type. str or 'serialized_proto' or 'immutable_proto' (Default = str)
        num_threads: the number of threads used in the batch processing (Default = -1).
      """
        if num_threads is None:
            num_threads = self._num_threads
        if (num_threads is None or type(num_threads) is not int):
            raise RuntimeError('num_threads must be int')
        if not input:
            return ''
        if out_type is str:
            if type(input) is int:
                return self._DecodeIds([input])
            if type(input) is str:
                return self._DecodePieces([input])
            if type(input) is list:
                if (len(input) == 0 or type(input[0]) is int):
                    return self._DecodeIds(input)
                if type(input[0]) is str:
                    return self._DecodePieces(input)
                if type(input[0]) is list:
                    if (len(input[0]) == 0 or type(input[0][0]) is int):
                        return self._DecodeIdsBatch(input, num_threads)
                    if type(input[0][0]) is str:
                        return self._DecodePiecesBatch(input, num_threads)
        if out_type == 'serialized_proto':
            if type(input) is int:
                return self._DecodeIdsAsSerializedProto([input])
            if type(input) is str:
                return self._DecodePiecesAsSerializedProto([input])
            if type(input) is list:
                if (len(input) == 0 or type(input[0]) is int):
                    return self._DecodeIdsAsSerializedProto(input)
                if type(input[0]) is str:
                    return self._DecodePiecesAsSerializedProto(input)
                if type(input[0]) is list:
                    if (len(input[0]) == 0 or type(input[0][0]) is int):
                        return self._DecodeIdsAsSerializedProtoBatch(input, num_threads)
                    if type(input[0][0]) is str:
                        return self._DecodePiecesAsSerializedProtoBatch(input, num_threads)
        if out_type == 'immutable_proto':
            if type(input) is int:
                return self._DecodeIdsAsImmutableProto([input])
            if type(input) is str:
                return self._DecodePiecesAsImmutableProto([input])
            if type(input) is list:
                if (len(input) == 0 or type(input[0]) is int):
                    return self._DecodeIdsAsImmutableProto(input)
                if type(input[0]) is str:
                    return self._DecodePiecesAsImmutableProto(input)
                if type(input[0]) is list:
                    if (len(input[0]) == 0 or type(input[0][0]) is int):
                        return self._DecodeIdsAsImmutableProtoBatch(input, num_threads)
                    if type(input[0][0]) is str:
                        return self._DecodePiecesAsImmutableProtoBatch(input, num_threads)
        raise RuntimeError('unknown output or input type')
        return None
    
    def DecodePieces(self, input, out_type=str, **kwargs):
        return self.Decode(input=input, out_type=out_type, **kwargs)
    
    def DecodeIds(self, input, out_type=str, **kwargs):
        return self.Decode(input=input, out_type=out_type, **kwargs)
    
    def DecodePiecesAsSerializedProto(self, input, out_type='serialized_proto', **kwargs):
        return self.Decode(input=input, out_type=out_type, **kwargs)
    
    def DecodeIdsAsSerializedProto(self, input, out_type='serialized_proto', **kwargs):
        return self.Decode(input=input, out_type=out_type, **kwargs)
    
    def DecodePiecesAsImmutableProto(self, input, out_type='immutable_proto', **kwargs):
        return self.Decode(input=input, out_type=out_type, **kwargs)
    
    def DecodeIdsAsImmutableProto(self, input, out_type='immutable_proto', **kwargs):
        return self.Decode(input=input, out_type=out_type, **kwargs)
    
    def CalculateEntropy(self, input, alpha, num_threads=None):
        """Calculate sentence entropy"""
        if type(input) is list:
            if num_threads is None:
                num_threads = self._num_threads
            if (num_threads is None or type(num_threads) is not int):
                raise RuntimeError('num_threads must be int')
            return self._CalculateEntropyBatch(input, alpha, num_threads)
        return self._CalculateEntropy(input, alpha)
    
    def piece_size(self):
        return self.GetPieceSize()
    
    def vocab_size(self):
        return self.GetPieceSize()
    
    def __getstate__(self):
        return self.serialized_model_proto()
    
    def __setstate__(self, serialized_model_proto):
        self.__init__()
        self.LoadFromSerializedProto(serialized_model_proto)
    
    def __len__(self):
        return self.GetPieceSize()
    
    def __getitem__(self, piece):
        return self.PieceToId(piece)
    
    def Load(self, model_file=None, model_proto=None):
        """Overwride SentencePieceProcessor.Load to support both model_file and model_proto.

      Args:
        model_file: The sentencepiece model file path.
        model_proto: The sentencepiece model serialized proto. Either `model_file`
          or `model_proto` must be set.
      """
        if (model_file and model_proto):
            raise RuntimeError('model_file and model_proto must be exclusive.')
        if model_proto:
            return self.LoadFromSerializedProto(model_proto)
        return self.LoadFromFile(model_file)

_sentencepiece.SentencePieceProcessor_swigregister(SentencePieceProcessor)

def SetRandomGeneratorSeed(seed):
    return _sentencepiece.SetRandomGeneratorSeed(seed)


class SentencePieceTrainer(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    
    def __init__(self, *args, **kwargs):
        raise AttributeError('No constructor defined')
    __repr__ = _swig_repr
    
    @staticmethod
    def _TrainFromString(arg):
        return _sentencepiece.SentencePieceTrainer__TrainFromString(arg)
    
    @staticmethod
    def _TrainFromMap(args):
        return _sentencepiece.SentencePieceTrainer__TrainFromMap(args)
    
    @staticmethod
    def _TrainFromMap2(args, iter):
        return _sentencepiece.SentencePieceTrainer__TrainFromMap2(args, iter)
    
    @staticmethod
    def _TrainFromMap3(args):
        return _sentencepiece.SentencePieceTrainer__TrainFromMap3(args)
    
    @staticmethod
    def _TrainFromMap4(args, iter):
        return _sentencepiece.SentencePieceTrainer__TrainFromMap4(args, iter)
    
    @staticmethod
    def _Train(arg=None, **kwargs):
        """Train Sentencepiece model. Accept both kwargs and legacy string arg."""
        if (arg is not None and type(arg) is str):
            return SentencePieceTrainer._TrainFromString(arg)
        
        def _encode(value):
            """Encode value to CSV.."""
            if type(value) is list:
                if sys.version_info[0] == 3:
                    f = StringIO()
                else:
                    f = BytesIO()
                writer = csv.writer(f, lineterminator='')
                writer.writerow([str(v) for v in value])
                return f.getvalue()
            else:
                return str(value)
        sentence_iterator = None
        model_writer = None
        new_kwargs = {}
        for (key, value) in kwargs.items():
            if key in ['sentence_iterator', 'sentence_reader']:
                sentence_iterator = value
            elif key in ['model_writer']:
                model_writer = value
            else:
                new_kwargs[key] = _encode(value)
        if model_writer:
            if sentence_iterator:
                model_proto = SentencePieceTrainer._TrainFromMap4(new_kwargs, sentence_iterator)
            else:
                model_proto = SentencePieceTrainer._TrainFromMap3(new_kwargs)
            model_writer.write(model_proto)
        elif sentence_iterator:
            return SentencePieceTrainer._TrainFromMap2(new_kwargs, sentence_iterator)
        else:
            return SentencePieceTrainer._TrainFromMap(new_kwargs)
        return None
    
    @staticmethod
    def Train(arg=None, logstream=None, **kwargs):
        with _LogStream(ostream=logstream):
            SentencePieceTrainer._Train(arg=arg, **kwargs)

_sentencepiece.SentencePieceTrainer_swigregister(SentencePieceTrainer)

def SentencePieceTrainer__TrainFromString(arg):
    return _sentencepiece.SentencePieceTrainer__TrainFromString(arg)

def SentencePieceTrainer__TrainFromMap(args):
    return _sentencepiece.SentencePieceTrainer__TrainFromMap(args)

def SentencePieceTrainer__TrainFromMap2(args, iter):
    return _sentencepiece.SentencePieceTrainer__TrainFromMap2(args, iter)

def SentencePieceTrainer__TrainFromMap3(args):
    return _sentencepiece.SentencePieceTrainer__TrainFromMap3(args)

def SentencePieceTrainer__TrainFromMap4(args, iter):
    return _sentencepiece.SentencePieceTrainer__TrainFromMap4(args, iter)
import re
import csv
import sys
import os
from io import StringIO
from io import BytesIO

def _add_snake_case(classname):
    """Added snake_cased method from CammelCased method."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('sentencepiece.__init__._add_snake_case', '_add_snake_case(classname)', {'re': re, 'classname': classname}, 0)

def _batchnize(classname, name):
    """Enables batch request for the method classname.name."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('sentencepiece.__init__._batchnize', '_batchnize(classname, name)', {'classname': classname, 'name': name}, 1)
_sentencepiece_processor_init_native = SentencePieceProcessor.__init__
setattr(SentencePieceProcessor, '__init__', SentencePieceProcessor.Init)
SentencePieceProcessor.Tokenize = SentencePieceProcessor.Encode
SentencePieceProcessor.Detokenize = SentencePieceProcessor.Decode
for m in ['PieceToId', 'IdToPiece', 'GetScore', 'IsUnknown', 'IsControl', 'IsUnused', 'IsByte']:
    _batchnize(SentencePieceProcessor, m)
_add_snake_case(SentencePieceProcessor)
_add_snake_case(SentencePieceTrainer)
set_random_generator_seed = SetRandomGeneratorSeed
from ._version import __version__


class _LogStream(object):
    
    def __init__(self, ostream=None):
        self.ostream = ostream
        if self.ostream is not None:
            self.orig_stream_fileno = sys.stderr.fileno()
    
    def __enter__(self):
        if self.ostream is not None:
            self.orig_stream_dup = os.dup(self.orig_stream_fileno)
            os.dup2(self.ostream.fileno(), self.orig_stream_fileno)
    
    def __exit__(self, type, value, traceback):
        if self.ostream is not None:
            os.close(self.orig_stream_fileno)
            os.dup2(self.orig_stream_dup, self.orig_stream_fileno)
            os.close(self.orig_stream_dup)
            self.ostream.close()


