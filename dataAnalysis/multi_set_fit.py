"""
Author: Fabian Oppliger
Email: fabianoppliger@bluewin.ch
Date: 2024-10-08
Description: This file contains the implementation of the MultiSetFit class for fitting a model to multiple datasets with shared and individual parameters.
"""


import numpy as np
from types import FunctionType, CodeType
import inspect

class MultiSetFit:
    """
    A class for fitting a model to multiple datasets with shared and individual parameters.
    The class generates a multi-set model function and combines the datasets and individual parameters into a single array.
    All arguments required by the model_function must be present in the independent_vars, shared_params, or individual_params.
    Args:
        model_function (function): The model function to fit to the datasets.
        dataset_idxs (list): A list of strings describing the dataset indices. The length must match the number of datasets.
        independent_vars (list): A list of independent variable names.
        shared_params (list, optional): A list of strings describing the  shared parameter names. Defaults to None.
        individual_params (list, optional): A list of strings describing the individual parameter names. The indices in dataset_idxs
                                            will be added to create an individual parameter for each dataset. Defaults to None.
    Attributes:
        n_sets (int): The number of datasets.
        dataset_idxs (list): A list of dataset indices.
        single_model (function): The model function.
        independent_vars (list): A list of independent variable names.
        shared_params (list): A list of shared parameter names.
        individual_params (list): A list of individual parameter names.
        individual_params_with_suffix (list): A list of individual parameter names with suffixes.
        arg_names (list): A list of argument names for the model function.
    Methods:
        generate_multi_model_function(print_string=False):
            Generates the multi-set model function. With multiple variants of each individual parameter, one for each dataset.
        combine_datasets(data, individual_params):
            Combines the datasets and individual parameters into a single flattened array.
    Raises:
        ValueError: If an argument is not found in independent_vars, shared_params, or individual_params.
        ValueError: If the number of datasets does not match the number of sets.
        ValueError: If the individual parameter arrays do not match the data shape.

    Example Usage:
        def linear_model(x, a, b):
            return a*x + b
        data = [np.array([1, 2, 3]), np.array([2, 3, 4])]
        x = np.array([1, 2, 3])

        multi_set_fit = MultiSetFit(linear_model, ['1', '2'], independent_vars=['x'], shared_params=['a'], individual_params=['b'])
        multi_model = multi_set_fit.generate_multi_model_function(print_string=True)
        dataset_idx, combined_data, combined_x = multi_set_fit.combine_datasets(data, [x, x])

        # define the lmfit parameters
        params = lmfit.Parameters()
        params.add('a', value=1, vary=True)
        params.add('b_1', value=1, vary=True)
        params.add('b_2', value=1, vary=True)

        # it is important to set the dataset_idx as an independent variable here
        model = lmfit.Model(multi_model, independent_vars=['dataset_idx', 'x'])
        result = model.fit(combined_data, params, dataset_idx=dataset_idx, x=combined_x)
        
    """

    def __init__(self, model_function, dataset_idxs: list, independent_vars: list, shared_params: list = None, individual_params: list = None):
        # Define the model functions
        self.n_sets = len(dataset_idxs)
        self.dataset_idxs = dataset_idxs
        self._suffixes = ['_' + dataset_idx for dataset_idx in dataset_idxs]
        self.single_model = model_function
        self.independent_vars = independent_vars
        self.shared_params = shared_params
        self.individual_params = individual_params
        if individual_params:
            self.individual_params_with_suffix = [param + suffix for param in individual_params for suffix in self._suffixes]

        # Check if all required arguments are found in the independent_vars, shared_params, or individual_params
        signature = inspect.signature(self.single_model)
        self.arg_names = [param.name for param in signature.parameters.values()]
        for arg in self.arg_names:
            if arg not in self.independent_vars + self.shared_params + self.individual_params:
                raise ValueError(f'Argument "{arg}" not found in independent_vars, shared_params or individual_params.')
        return

    def generate_multi_model_function(self, print_string: bool = False):
        """
        Generates a multi-set model function based on the given parameters. The function is constructed as a string
        and is then compiled and returned.
        Args:
            print_string (bool, optional): If True, prints the generated function string. Defaults to False.
        Returns:
            function: The compiled multi-set model function.
        Raises:
            None
        Example Usage:
            generate_multi_model_function(print_string=True)
        """
        # Generate the multi-set model

        ##### define the function string #####

        # define the function signature
        func_name = self.single_model.__name__
        func_txt = f'def {func_name}(dataset_idx, {", ".join(self.independent_vars)}'
        if self.shared_params:
            func_txt += ', ' + ', '.join([param for param in self.shared_params])
        if self.individual_params:
            for param in self.individual_params:
                func_txt += ',\n        ' + ', '.join([param + suffix for suffix in self._suffixes])
        func_txt += '):\n'
        
        # combine the individual parameters in a single array
        # np.unique sorts the uniques, so we have to make sure the order is preserved (hence uniques=={idx} below)
        if self.individual_params:
            func_txt += '    uniques, counts = np.unique(dataset_idx, return_counts=True)\n'
            for i, param in enumerate(self.individual_params):
                func_txt += '    {param} = np.array({indiv_params})\n'.format(
                    param = param,
                    indiv_params = ' + '.join([f'[{param}_{idx}]*counts[uniques=={idx}][0]' for idx in self.dataset_idxs]))

        # call and return the single model function for the comined datasets
        arg_str = ', '.join([f'{arg}={arg}' for arg in self.arg_names])
        call_single_model_func = f'{func_name}({arg_str})'
        func_txt += '\n    return ' + call_single_model_func


        # print the function string
        if print_string:
            print(func_txt)

        # compile and return the function
        func_code = compile(func_txt, "<string>", "exec")
        code = [entry for entry in func_code.co_consts if isinstance(entry, CodeType)][0]
        func = FunctionType(code, globals(), func_name, argdefs=func_code.co_consts[-1])
        return func


    def combine_datasets(self, datasets: list, *individual_params: list) -> tuple:
        """
        Combines multiple datasets into a single dataset along with individual parameters.
        Args:
            datasets (list): A list of datasets to be combined. Each dataset should be a numpy array.
            *individual_params (list): Variable number of individual parameters. Each parameter should contain a list of numpy arrays
                                        corresponding to the datasets.
        Returns:
            tuple: A tuple containing the combined dataset index, the combined data, and the combined individual parameters.
        Raises:
            ValueError: If the number of datasets does not match the number of sets.
            ValueError: If the individual parameter arrays do not match the shape of the data.
        Example:
            data = [dataset1, dataset2, dataset3]
            x = [x1, x2, x3]
            y = [y1, y2, y3]
            combine_datasets(data, x, y)
        """

        if len(datasets) != self.n_sets:
            raise ValueError('Number of datasets does not match number of sets')
        
        data_combo = np.array([])
        dataset_idx = np.array([])
        individual_params_combo = [np.array([])]*len(individual_params)
        # individual_params_combo = [[] for _ in range(len(individual_params))]
        for i, data in enumerate(datasets):
            data_combo = np.append(data_combo, data.flatten())
            # dataset_nr = np.append(dataset_nr,np.repeat(i, len(data_single_set.flatten())))
            dataset_idx = np.append(dataset_idx,np.repeat(self.dataset_idxs[i], len(data.flatten())))
        
        # for i, param_set in enumerate(individual_params):
            for j, param_set in enumerate(individual_params):
                # print(param)
                param = param_set[i]
                if param.shape != data.shape:
                    raise ValueError('Individual parameter arrays do not match the data shape.')
                individual_params_combo[j] = np.append(individual_params_combo[j], param.flatten())
        
        return (dataset_idx, data_combo, *individual_params_combo)
