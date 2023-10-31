import warnings

def get_enum(reduction):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn._reduction.get_enum', 'get_enum(reduction)', {'warnings': warnings, 'reduction': reduction}, 1)

def legacy_get_string(size_average, reduce, emit_warning=True):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn._reduction.legacy_get_string', 'legacy_get_string(size_average, reduce, emit_warning=True)', {'warnings': warnings, 'size_average': size_average, 'reduce': reduce, 'emit_warning': emit_warning}, 1)

def legacy_get_enum(size_average, reduce, emit_warning=True):
    return get_enum(legacy_get_string(size_average, reduce, emit_warning))

