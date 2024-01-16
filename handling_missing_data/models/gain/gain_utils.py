"""Utility functions for GAIN.
"""
import numpy as np

def rounding (cat_vars, data_imputed):
  """Use rounding for categorical variables.
  
  Args:
    - cat_vars: list of indicies of categorical variables
    - data_imputed: complete imputed data
    
  Returns:
    - data_imputed: imputed data after rounding
  """
  for i in range(data_imputed.shape[1]):
    # If the feature is categorical
    if i in cat_vars:
      # Rounding
      data_imputed[:, i] = np.round(data_imputed[:, i])
      
  return data_imputed


def rounding_automatic (data, data_imputed):
  """Use rounding for categorical variables.
  
  Args:
    - data: incomplete original data
    - data_imputed: complete imputed data
    
  Returns:
    - data_imputed: imputed data after rounding
  """
  for i in range(data.shape[1]):
    # If the feature is categorical (category < 20)
    if len(np.unique(data[:, i])) < 20:
      # If values are integer
      if sum(np.round(data[:, i]) == data[:, i]) == len(data[:, i]):
        # Rounding
        data_imputed[:, i] = np.round(data_imputed[:, i])
      
  return data_imputed


def normalization (data: np.ndarray, parameters=None):
  '''Normalize data in [0, 1] range.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  '''

  # Parameters
  _, dim = data.shape
  norm_data = data.copy()
  
  if parameters is None:
  
    # MixMax normalization
    min_val = np.zeros(dim)
    max_val = np.zeros(dim)
    
    # For each dimension
    for i in range(dim):
      min_val[i] = np.nanmin(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
      max_val[i] = np.nanmax(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
      
    # Return norm_parameters for renormalization
    norm_parameters = {'min_val': min_val,
                       'max_val': max_val}

  else:
    min_val = parameters['min_val']
    max_val = parameters['max_val']
    
    # For each dimension
    for i in range(dim):
      norm_data[:,i] = norm_data[:,i] - min_val[i]
      norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  
      
    norm_parameters = parameters    
      
  return norm_data, norm_parameters


def renormalization (norm_data: np.ndarray, norm_parameters) -> np.ndarray:
  '''Renormalize data from [0, 1] range to the original range.
  
  Args:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  
  Returns:
    - renorm_data: renormalized original data
  '''
  
  min_val = norm_parameters['min_val']
  max_val = norm_parameters['max_val']

  _, dim = norm_data.shape
  renorm_data = norm_data.copy()
    
  for i in range(dim):
    renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
    renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    
  return renorm_data
