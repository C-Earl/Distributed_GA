class Gene(dict):
  # Supported var_types: float, bool, array (np.ndarray)
  def __init__(self, var_type, min_val=0, max_val=1, shape=0, **fields):
    super().__init__()
    self['var_type'] = var_type
    self['min_val'] = min_val
    self['max_val'] = max_val
    self['shape'] = shape
    