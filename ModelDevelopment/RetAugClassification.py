class RetAugClassification:
  def __init__(self, image_tensor):
    self.image_tensor = image_tensor
  
  def define_retrieval_module_universe(self):
    # Effectively a case study function.
    # This function will numerically determine if there is a class imbalance problem
    # And give important analytics on how imbalanced the data is
    pass
  
  def obtain_retrieval_module(self):
    # Depending on the image tensor, the right retrieval module is obtained
    # For example: determine if the image is one of our underrepresented classes
    # Then obtain the retrieval module for that class
    # Which will in turn help us counter the class imbalance problem
    # As we can learn more about this image by artificially expanding the available universe
    pass