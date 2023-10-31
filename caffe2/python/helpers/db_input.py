from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

def db_input(model, blobs_out, batch_size, db, db_type):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.db_input.db_input', 'db_input(model, blobs_out, batch_size, db, db_type)', {'model': model, 'blobs_out': blobs_out, 'batch_size': batch_size, 'db': db, 'db_type': db_type}, 1)

