class Subset_Pool(dict):
  def __init__(self, condition: callable):
    self.condition = condition          # Condition for subset
    super().__init__()

  def __setitem__(self, key, value):
    if self.condition(key, value):
      super().__setitem__(key, value)   # Set item only if subset condition is met


class Pool(dict):
  def __init__(self):
    self.subset_pools = []        # Pools to make proxy changes to
    super().__init__()

  def add_subset_pool(self, subset_pool: Subset_Pool):
    self.subset_pools.append(subset_pool)
    for key, value in self.items():
      subset_pool[key] = value    # Add all current items to subset pool (with condition)

  def __setitem__(self, key, value):
    for sub_pool in self.subset_pools:
      if key in sub_pool and not sub_pool.condition(key, value):
        del sub_pool[key]         # If condition no longer met, delete from subset
      elif sub_pool.condition(key, value):
        sub_pool[key] = value     # Adjust sub pools to match change
    super().__setitem__(key, value)

  def __delitem__(self, key):
    for sub_pool in self.subset_pools:
      if key in sub_pool:
        del sub_pool[key]         # Since subsets, only delete if in subset
    super().__delitem__(key)
