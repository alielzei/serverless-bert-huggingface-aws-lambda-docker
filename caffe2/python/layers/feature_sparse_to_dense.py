from __future__ import absolute_import, division, print_function, unicode_literals
from collections import defaultdict
import numpy as np
from caffe2.python import schema
from caffe2.python.layers.layers import ModelLayer, AccessedFeatures


class FeatureSparseToDense(ModelLayer):
    
    def __init__(self, model, input_record, input_specs, name='feature_sparse_to_dense', **kwargs):
        """
        `input_specs` follows the format of FeatureSpec from schema. To be more
        precise it's a namedtuple that should have:
            'feature_type', 'feature_names', 'feature_ids'
        """
        super(FeatureSparseToDense, self).__init__(model, name, input_record, **kwargs)
        self.input_specs = input_specs
        outputs = []
        for (field, feature_specs) in self.input_specs:
            assert len(feature_specs.feature_names) == len(feature_specs.feature_ids)
            if feature_specs.feature_type == 'FLOAT':
                outputs.append((field, schema.Scalar((np.float32, (len(feature_specs.feature_ids), )), self.get_next_blob_reference(field + '_output'))))
            elif feature_specs.feature_type == 'ID_LIST':
                outputs.append((field, schema.Struct(('ranges', schema.Scalar((np.int32, (len(feature_specs.feature_ids), 2)), self.get_next_blob_reference(field + '_ranges'))), ('values', schema.Scalar(np.int64, self.get_next_blob_reference(field + '_values'))))))
            elif feature_specs.feature_type == 'ID_SCORE_LIST':
                outputs.append((field, schema.Struct(('ranges', schema.Scalar((np.int32, (len(feature_specs.feature_ids), 2)), self.get_next_blob_reference(field + '_ranges'))), ('ids', schema.Scalar(np.int64, self.get_next_blob_reference(field + '_ids'))), ('scores', schema.Scalar(np.float32, self.get_next_blob_reference(field + '_scores'))))))
            elif feature_specs.feature_type == 'EMBEDDING':
                outputs.append((field, schema.Struct(('ranges', schema.Scalar((np.int32, (len(feature_specs.feature_ids), 2)), self.get_next_blob_reference(field + '_ranges'))), ('values', schema.Scalar(np.float32, self.get_next_blob_reference(field + '_values'))))))
            elif feature_specs.feature_type == 'GENERIC_FEATURE':
                outputs.append((field, schema.Struct(('ranges', schema.Scalar((np.int32, (len(feature_specs.feature_ids), 2)), self.get_next_blob_reference(field + '_ranges'))), ('values', schema.Scalar(np.float32, self.get_next_blob_reference(field + '_values'))))))
            else:
                raise TypeError('Unsupported input type: {0}'.format(feature_specs.feature_type))
        self.output_schema = schema.Struct(*outputs)
        for (field, feature_specs) in input_specs:
            schema.attach_metadata_to_scalars(self.output_schema[field], schema.Metadata(feature_specs=feature_specs))
        self.zero = model.global_constants['ZERO']
        self.zero_range = model.global_constants['ZERO_RANGE']
    
    def add_ops(self, net):
        record = self.input_record
        for (field, feature_specs) in self.input_specs:
            if feature_specs.feature_type == 'FLOAT':
                net.SparseToDenseMask([record[field].keys(), record[field].values(), self.zero, record[field].lengths()], [self.output_schema[field]()], mask=feature_specs.feature_ids)
            elif feature_specs.feature_type == 'ID_LIST':
                id_list_ranges = net.LengthsToRanges(record[field].values.lengths(), net.NextScopedBlob('id_list_ranges'))
                net.SparseToDenseMask([record[field].keys(), id_list_ranges, self.zero_range, record[field].lengths()], self.output_schema[field].ranges(), mask=feature_specs.feature_ids)
                net.Alias(record[field].values.items(), self.output_schema[field].values())
            elif feature_specs.feature_type == 'ID_SCORE_LIST':
                id_list_ranges = net.LengthsToRanges(record[field].values.lengths(), net.NextScopedBlob('id_score_list_ranges'))
                net.SparseToDenseMask([record[field].keys(), id_list_ranges, self.zero_range, record[field].lengths()], self.output_schema[field].ranges(), mask=feature_specs.feature_ids)
                net.Alias(record[field].values.keys(), self.output_schema[field].ids())
                net.Alias(record[field].values.values(), self.output_schema[field].scores())
            elif feature_specs.feature_type == 'EMBEDDING':
                ranges = net.LengthsToRanges(record[field].values.lengths(), net.NextScopedBlob('embeddings_ranges'))
                net.SparseToDenseMask([record[field].keys(), ranges, self.zero_range, record[field].lengths()], self.output_schema[field].ranges(), mask=feature_specs.feature_ids)
                net.Alias(record[field].values.items(), self.output_schema[field].values())
            elif feature_specs.feature_type == 'GENERIC_FEATURE':
                (feature_lengths_blob, feature_ids_blob, value_lengths_blob, value_values_blob) = net.ParseGeneric([record[field]()], ['feature_lengths', 'feature_ids', 'value_lengths', 'value_values'], feature_type_enum=1)
                ranges = net.LengthsToRanges(value_lengths_blob, net.NextScopedBlob('generics_ranges'))
                net.SparseToDenseMask([feature_ids_blob, ranges, self.zero_range, feature_lengths_blob], self.output_schema[field].ranges(), mask=feature_specs.feature_ids)
                net.Alias(value_values_blob, self.output_schema[field].values())
    
    def get_metadata(self):
        metadata = []
        for (field, feature_specs) in self.input_specs:
            metadata.append(({'type': feature_specs.feature_type, 'names': feature_specs.feature_names, 'ids': feature_specs.feature_ids}, self.output_schema[field].field_blobs(), self.output_schema[field].field_types()))
            if feature_specs.feature_type == 'FLOAT':
                metadata[-1][0]['cardinality'] = 1
        return metadata
    
    def get_accessed_features(self):
        accessed_features = defaultdict(list)
        for (field, feature_specs) in self.input_specs:
            accessed_features[field].append(AccessedFeatures(feature_specs.feature_type, set(feature_specs.feature_ids)))
        return accessed_features


