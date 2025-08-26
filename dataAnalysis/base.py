
import numpy as np
import qcodes as qc
from qcodes.dataset.data_export import DSPlotData
from qcodes.dataset.plotting import _rescale_ticks_and_units, plot_dataset
import matplotlib.pyplot as plt
import copy

def val_to_index(array, value):
    start = array[0]
    step = array[1] - array[0]
    result =  int((value - start)/step)
    if result < 0:
        return 0
    if result > len(array):
        return [-1]
    return result

def filter_array(main_array:np.ndarray, lower_value=None, upper_value=None, other_arrays=None):
    lower_index = val_to_index(main_array, lower_value) if lower_value else 0
    upper_index = val_to_index(main_array, upper_value) if upper_value else -1
    main_array = main_array[lower_index:upper_index]
    for other_array in other_arrays:
        other_array = other_array[lower_index:upper_index]

    return main_array, other_arrays

class DataSet():
    """
    Represents a dataset and provides methods for extracting and manipulating the data.

    Args:
        exp (qcodes.dataset.experiment_container.Experiment, optional): The qcodes Experiment object.
        run_id (str, optional): The ID of the measurement run. If not provided, the last recorded run ID in exp will be used.
        station (qcodes.station.Station, optional): The qcodes Station object. Defaults to None.

    Attributes:
        exp (qcodes.dataset.experiment_container.Experiment): The Experiment object.
        run_id (str): The ID of the run.
        conn (Connection): The connection to the database.
        station (qcodes.station.Station): The qcodes Station object that contains metadata about the instruments. probably not even needed...
        dependent_parameters (dict): A dictionary containing the dependent parameters.
        independent_parameters (dict): A dictionary containing the independent parameters.
        dataset (DataSet): The dataset object.

    Methods:
        extract_data(run_id=None, exp=None): Extracts data from a dataset and organizes it into independent and dependent parameters.
        copy_dependent_parameter(parameter_to_copy, copy_name): Copies a dependent parameter with a new name.
        copy_independent_parameter(parameter_to_copy, copy_name): Copies an independent parameter with a new name.
        normalize_data(params_to_normalize, data_bg, x_bg, axis=0, operation='subtract', interpolate=True): Normalizes the data for the specified dependent parameters.
        normalize_data_raw(data_raw, x_raw, data_bg, x_bg, axis=0, operation='subtract', interpolate=True): Normalizes the given raw data by subtracting or dividing it with the background data.
        plot_2D(params_to_plot=None, cmap='viridis', **kwargs): Plots 2D data based on the specified parameters.
    
    """
    def __init__(self, exp, run_id=None, station=None):
        if run_id:
            self.run_id = run_id
        else:
            self.run_id = exp.last_data_set().run_id
        self.exp = exp
        self.conn = self.exp.conn
        self.station = station
        self.extract_data()

    def extract_data(self, exp=None, run_id=None):
        """
        Extracts data from a dataset and organizes it into independent and dependent parameters.
        The parameters are callable using self.dependent_parameters and self.independent_parameters.

        Args:
            run_id (str, optional): The ID of the run. If not provided, the last recorded run ID in exp will be used.
            exp (Experiment, optional): The Experiment object. If not provided, the last Experiment will be used.

        Returns:
            None
        """
        if not run_id: run_id = self.run_id
        if not exp: exp = self.exp
        dataset = qc.load_by_id(run_id, exp.conn)
        df = dataset.to_pandas_dataframe()
        interdeps = dataset.description.interdeps.dependencies
    
        paramspecs = dataset.paramspecs
        dependent_parameters = dataset.description.interdeps.non_dependencies
        # dependent_parameters = [paramspecs[param.name] for param in dataset.description.interdeps.non_dependencies]
    
        n_independent_parameters = 0
        for param in dependent_parameters:
            independent_parameters = interdeps[param]
            n_independent_parameters = max(n_independent_parameters, len(independent_parameters))
        if len(paramspecs) > n_independent_parameters + len(dependent_parameters):
            print('Warning: This function is not suitable for exctracting data from such a dataset. All dependent parameters have to depend on the same independent parameters.')
            return None
    
        if n_independent_parameters == 2:
            data_pivot = df.pivot_table(index=independent_parameters[1].name, columns=independent_parameters[0].name)
            independent_values = {
            'x': {'name': 'x', 'paramspec': paramspecs[independent_parameters[0].name], 'values': data_pivot[dependent_parameters[0].name].columns.values},
            'y': {'name': 'y', 'paramspec': paramspecs[independent_parameters[1].name], 'values': data_pivot[dependent_parameters[0].name].index.values}
            }
            dependent_values = {}
            for i, param in enumerate(dependent_parameters):
                dependent_values[f'param_{i}'] = {'name': f'param_{i}', 'paramspec': paramspecs[param.name], 'values': data_pivot[param.name].values}
        elif n_independent_parameters == 1:
            independent_values = {'x': {'name': 'x', 'paramspec': paramspecs[independent_parameters[0].name], 'values': df[dependent_parameters[0].name].index.values}}
            dependent_values = {}
            for i, param in enumerate(dependent_parameters):
                dependent_values[f'param_{i}'] = {'name': f'param_{i}', 'paramspec': paramspecs[param.name], 'values': df[param.name].values}
        self.dependent_parameters = dependent_values
        self.independent_parameters = independent_values
        self.dataset = dataset

    def copy_dependent_parameter(self, parameter_to_copy, copy_name):
        """
        Generates a copy of the dependent parameter with the name 'parameter_to_copy' and naming that copy 'copy_name'.
        """
        self.dependent_parameters[copy_name] = copy.deepcopy(self.dependent_parameters[parameter_to_copy])
        self.dependent_parameters[copy_name]['name'] = copy_name

    def copy_independent_parameter(self, parameter_to_copy, copy_name):
        """
        Generates a copy of the independent parameter with the name 'parameter_to_copy' and naming that copy 'copy_name'.
        """
        self.independent_parameters[copy_name] = copy.deepcopy(self.independent_parameters[parameter_to_copy])
        self.independent_parameters[copy_name]['name'] = copy_name

    def get_parameter_by_name(self, name):
        print('Deprecated function - Use get_independent_parameter_by_name or get_dependent_parameter_by_name instead.')

    def get_independent_parameter_by_name(self, name):
        param = None
        first_match = True
        for key in self.independent_parameters.keys():
            if name in self.independent_parameters[key]['paramspec'].name:
                if first_match:
                    param = self.independent_parameters[key]
                    first_match = False
                else:
                    print(f"Warning: {name} is the second match found in independent_parameters. Returning {param['paramspec'].name} instead.")
        return param

    def get_dependent_parameter_by_name(self, name):
        param = None
        first_match = True
        for key in self.dependent_parameters.keys():
            if name in self.dependent_parameters[key]['paramspec'].name:
                if first_match:
                    param = self.dependent_parameters[key]
                    first_match = False
                else:
                    print(f"Warning: {name} is the second match found in independent_parameters. Returning {param['paramspec'].name} instead.")
        return param

    def normalize_data_mag_phase(self, mag_param_name, phase_param_name, cData_bg, x_bg, axis=0, interpolate=True):
        """
        Normalize the magnitude and phase data with the given complex data.
    
        Args:
            mag_param_name (str): The name of the magnitude parameter(s) to normalize.
            phase_param_name (str): The name of the phase parameter(s) to normalize.
            cData_bg (numpy.ndarray): The complex background data.
            x_bg (numpy.ndarray): The x-values of the background data.
            axis (int, optional): The axis along which to normalize the data. Can be 0 and 1. Defaults to 0.
            interpolate (bool, optional): Whether to interpolate the data. Defaults to False.
    
        Returns:
            None

        """
    
        mag_vals = self.dependent_parameters[mag_param_name]['values']
        phase_vals = self.dependent_parameters[phase_param_name]['values']
        cData_to_normalize = 10**(mag_vals/20)*np.exp(1j*phase_vals)
        if (len(self.dependent_parameters[mag_param_name]['paramspec'].depends_on_) > 1) and (axis == 0):
            x_vals = self.independent_parameters['y']['values']
        else:
            x_vals = self.independent_parameters['x']['values']

        data_normalized = self.normalize_data_raw(cData_to_normalize, x_vals, cData_bg, x_bg, axis, 'divide', interpolate)
        self.copy_dependent_parameter(mag_param_name, mag_param_name+'_normalized')
        self.dependent_parameters[mag_param_name+'_normalized']['values'] = 20*np.log10(np.abs(data_normalized))
        self.copy_dependent_parameter(phase_param_name, phase_param_name+'_normalized')
        self.dependent_parameters[phase_param_name+'_normalized']['values'] = np.angle(data_normalized)


    def normalize_data(self, params_to_normalize, data_bg, x_bg, axis=0, operation='subtratct', interpolate=True):
        """
        Normalize the data for the specified dependent parameters.
    
        Args:
            params_to_normalize (str or list): The parameter(s) to normalize. It can be a single string or a list of strings corresponding to the parameter names.
            data_bg (numpy.ndarray or list): The background data.
            x_bg (numpy.ndarray or list): The x-values of the background data.
            axis (int or list, optional): The axis along which to normalize the data. Can be 0 and 1. Defaults to 0.
            operation (str or list, optional): The operation to perform for normalization. Can be 'subtract' or 'divide'. Defaults to 'subtract'.
            interpolate (bool or list, optional): Whether to interpolate the data. Defaults to False.
    
        Returns:
            None

        """
        # Convert single string to list
        if isinstance(params_to_normalize, str):
            params_to_normalize = [params_to_normalize]
    
        # Create a dictionary mapping variable names to their values
        variables = {'data_bg': data_bg, 'x_bg': x_bg, 'axis': axis, 'operation': operation, 'interpolate': interpolate}
        # Iterate over the dictionary
        for name, value in variables.items():
            if isinstance(value, list):
                if len(value) != len(params_to_normalize):
                    raise ValueError(f'The length of {name} must be equal to the length of params_to_normalize')
            else:
                variables[name] = [value] * len(params_to_normalize)
        # Extract the variables from the dictionary
        data_bg, x_bg, axis, operation, interpolate = variables.values()    
    
        
        nomalized_dict = {}
        for i, param_name in enumerate(params_to_normalize):
            if param_name not in self.dependent_parameters:
                raise ValueError(f'The parameter {param_name} does not exist in the data.')
            if (len(self.dependent_parameters[param_name]['paramspec'].depends_on_) > 1) and (axis[i] == 0):
                x_vals = self.independent_parameters['y']['values']
            else:
                x_vals = self.independent_parameters['x']['values']
            self.copy_dependent_parameter(param_name, param_name+'_normalized')
            y_vals = self.dependent_parameters[param_name]['values']
            data_normalized = self.normalize_data_raw(y_vals, x_vals, data_bg[i], x_bg[i], axis[i], operation[i], interpolate[i])
            self.dependent_parameters[param_name+'_normalized']['values'] = data_normalized
    
    def normalize_data_raw(self, data_raw, x_raw, data_bg, x_bg, axis=0, operation='subtratct', interpolate=True):
        """
        Normalize the given raw data by subtracting or dividing it with the background data.

        Parameters:
        - data_raw (numpy.ndarray): The raw data to be normalized.
        - x_raw (numpy.ndarray): The x-values corresponding to the raw data.
        - data_bg (numpy.ndarray): The background data used for normalization.
        - x_bg (numpy.ndarray): The x-values corresponding to the background data.
        - axis (int, optional): The axis along which the normalization is performed. Default is 0.
        - operation (str, optional): The operation to perform for normalization. Can be 'subtract' or 'divide'. Default is 'subtract'.
        - interpolate (bool, optional): Whether to interpolate the background data to match the resolution of the raw data. Default is False.

        Returns:
        - numpy.ndarray: The normalized data.

        Raises:
        - None

        Examples:
        >>> data_raw = np.array([1, 2, 3, 4, 5])
        >>> x_raw = np.array([0, 1, 2, 3, 4])
        >>> data_bg = np.array([0, 0, 0, 0, 0])
        >>> x_bg = np.array([0, 1, 2, 3, 4])
        >>> normalize_data_raw(data_raw, x_raw, data_bg, x_bg)
        array([1, 2, 3, 4, 5])
        """
        x_slice = self.find_slice(x_bg,[min(x_raw),max(x_raw)])
        if not interpolate and (len(x_raw) != len(x_bg[x_slice])):
            print('Could not normalize data. If interpolate=False, x_raw and x_bg must have the same resolution,')
            print(f'but they have len len(x_raw) = {len(x_raw)} and len(x_bg[x_slice]) = {len(x_bg[x_slice])}')
            return data_raw
    
        if interpolate:
            if (min(x_raw) < min(x_bg)) or (max(x_raw) > max(x_bg)):
                print('Some values in x_raw are outside the bounds of x_bg.')
            data_bg_converted = np.interp(x_raw, x_bg, data_bg)
        else:
            data_bg_converted = data_bg[x_slice]
        if (len(data_raw.shape) > 1) and (axis == 0):
            data_bg_converted = data_bg_converted[:,None]
        if operation=='subtratct':
            data_normalized = data_raw - data_bg_converted
        elif operation=='divide':
            data_normalized = data_raw / data_bg_converted
    
        return data_normalized

    def derive_data(self, parameters, axis=0):
        """
        Derive data from the specified parameters.

        Args:
            parameters (list): List of parameter names to derive.
            axis (int, optional): The axis along which to derive the data. Defaults to 0.

        Returns:
            None

        """
        # Convert single string to list
        if isinstance(parameters, str):
            parameters = [parameters]
        for param_name in parameters:
            if param_name not in self.dependent_parameters:
                raise ValueError(f'The parameter {param_name} does not exist in the data.')
            self.copy_dependent_parameter(param_name, param_name+'_derived')
            data = self.dependent_parameters[param_name]['values']

            if len(data.shape) == 1 or axis == 1:
                x_vals = self.independent_parameters['x']['values']
            elif axis == 0:
                x_vals = self.independent_parameters['y']['values']
            elif axis == (0,1):
                x_vals = (self.independent_parameters['y']['values'], self.independent_parameters['x']['values'])
            data_derived = np.gradient(data, x_vals, axis=axis)
            self.dependent_parameters[param_name+'_derived']['values'] = data_derived

    def plot_1D(self, params_to_plot=None, x_range=None, title=None, **kwargs):
        """
        Plots 1D data based on the specified parameters.

        Args:
            params_to_plot (list or str, optional): List of parameter names to plot. If not provided, all dependent parameters will be plotted. Defaults to None.
            x_range (list, optional): List of the plotting limits of the x-axis. Defaults to None (max range).
            title (str, optional): Plot title to add after run_id and experiment name. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the `plt.plot` function.

        Returns:
            tuple: A tuple containing the generated figures and axes.

        Raises:
            ValueError: If a parameter name in `params_to_plot` does not exist in the data.
        """
        # Convert single string to list
        if isinstance(params_to_plot, str):
            params_to_plot = [params_to_plot]
        if not params_to_plot:
            params_to_plot = self.dependent_parameters.keys()
        x_vals = self.independent_parameters['x']['values']
        x_plot = x_vals

        plot_title = (
        f"Run #{self.run_id}, "
        "Experiment " + self.exp.name
        )
        if title: plot_title += f", {title}"

        figs = []
        axes = []
        for param_name in params_to_plot:
            if param_name not in self.dependent_parameters:
                raise ValueError(f'The parameter {param_name} does not exist in the data.')
            x_param_name = self.dependent_parameters[param_name]['paramspec'].depends_on_[0]
            for key, param in self.independent_parameters.items():
                if param['paramspec'].name == x_param_name: x_key = key
            x_param = self.independent_parameters[x_key]
            y_param = self.dependent_parameters[param_name]
            x_vals = x_param['values']
            x_slice = slice(0,None)
            if x_range: x_slice = self.find_slice(x_vals, x_range)
            
            x_plot = x_vals[x_slice]
            y_plot = y_param['values'][x_slice]
            
            fig,ax = plt.subplots()
            figs.append(fig)
            axes.append(ax)
            ax.plot(x_plot, y_plot, **kwargs)
            ax.set_xlabel(x_param['paramspec'].label + f" ({x_param['paramspec'].unit})")
            ax.set_ylabel(y_param['paramspec'].label + f" ({y_param['paramspec'].unit})")
            ax.set_title(plot_title)
            ds_plot_data = self._generate_DSPlotData(x_param, y_param)
            _rescale_ticks_and_units(ax, ds_plot_data)
        return figs,axes

    def plot_1D_cut(self, params_to_plot=None, cut_value=None, cut_idx=None, x_range=None, axis=0, title=None, **kwargs):
        """
        Plots 1D data cut based on the specified parameters.

        Args:
            params_to_plot (list or str, optional): List of parameter names to plot. If not provided, all dependent parameters will be plotted. Defaults to None.
            cut_value (float, optional): The value at which to cut the data. Specify only one of cut_value and cut_idx. Defaults to None.
            cut_idx (int, optional): The index at which to cut the data. Specify only one of cut_value and cut_idx. Defaults to None.
            x_range (list, optional): List of the plotting limits of the x-axis. Defaults to None (max range).
            title (str, optional): Plot title to add after run_id and experiment name. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the `plt.plot` function.

        Returns:
            tuple: A tuple containing the generated figures and axes.

        Raises:
            ValueError: If a parameter name in `params_to_plot` does not exist in the data.
        """
        # Convert single string to list
        if isinstance(params_to_plot, str):
            params_to_plot = [params_to_plot]
        if not params_to_plot:
            params_to_plot = self.dependent_parameters.keys()

        plot_title = (
        f"Run #{self.run_id}, "
        "Experiment " + self.exp.name
        )
        if title: plot_title += f", {title}"
        if (cut_value is None and cut_idx is None) or (cut_value is not None and cut_idx is not None):
            # if neither or both of cut_value and cut_idx are provided
            raise ValueError(f'Specify one argument out of cut_value and cut_index.')
        
        figs = []
        axes = []
        for i, param_name in enumerate(params_to_plot):
            if param_name not in self.dependent_parameters:
                raise ValueError(f'The parameter {param_name} does not exist in the data.')
            dependencies = self.dependent_parameters[param_name]['paramspec'].depends_on_
            x_param_name = dependencies[0]
            y_param_name = dependencies[1]
            for key, param in self.independent_parameters.items():
                if param['paramspec'].name == x_param_name: x_key = key
                if param['paramspec'].name == y_param_name: y_key = key

            if axis==0:
                x_param = self.independent_parameters[y_key]
                cut_param = self.independent_parameters[x_key]
            elif axis==1:
                x_param = self.independent_parameters[x_key]
                cut_param = self.independent_parameters[y_key]
            else:
                raise ValueError(f'axis is {axis}, but must be 0 or 1.')
            if cut_value is not None:
                cut_idx = (np.abs(cut_param['values'] - cut_value)).argmin()
            z_param = self.dependent_parameters[param_name]
            
            x_plot = x_param['values']
            if axis==0:
                z_plot = z_param['values'][:,cut_idx]
            else:
                z_plot = z_param['values'][cut_idx]

            x_slice = slice(0,None)
            if x_range: x_slice = self.find_slice(x_plot, x_range)
            x_plot = x_plot[x_slice]
            z_plot = z_plot[x_slice]
            
            fig, ax = plt.subplots()
            figs.append(fig)
            axes.append(ax)
            ax.plot(x_plot, z_plot, **kwargs)
            ax.set_xlabel(x_param['paramspec'].label + f" ({x_param['paramspec'].unit})")
            ax.set_ylabel(z_param['paramspec'].label + f" ({z_param['paramspec'].unit})")
            ax.set_title(plot_title)
            ds_plot_data = self._generate_DSPlotData(x_param, z_param)
            _rescale_ticks_and_units(ax, ds_plot_data)
        return figs,axes
    
    def plot_2D(self, params_to_plot=None, cmap='viridis', x_range=None, y_range=None, transpose=False, title=None, **kwargs):
        """
        Plots 2D data based on the specified parameters.

        Args:
            params_to_plot (list or str, optional): List of parameter names to plot. If not provided, all dependent parameters will be plotted. Defaults to None.
            cmap (str, optional): The colormap to use for the plot. Defaults to 'viridis'.
            x_range (list, optional): List of the plotting limits of the x-axis. Defaults to None (max range).
            y_range (list, optional): List of the plotting limits of the y-axis. Defaults to None (max range).
            transpose (bool, optional): Whether to transpose the data. Defaults to False.
            title (str, optional): Plot title to add after run_id and experiment name. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the `qc.dataset.plotting.plot_on_a_plain_grid` function.

        Returns:
            tuple: A tuple containing the generated figures and axes.

        Raises:
            ValueError: If a parameter name in `params_to_plot` does not exist in the data.
        """
        # Convert single string to list
        if isinstance(params_to_plot, str):
            params_to_plot = [params_to_plot]
        if not params_to_plot:
            params_to_plot = self.dependent_parameters.keys()

        plot_title = (
        f"Run #{self.run_id}, "
        "Experiment " + self.exp.name
        )
        if title: plot_title += f", {title}"
        
        figs = []
        axes = []
        cbs = []
        for i, param_name in enumerate(params_to_plot):
            if param_name not in self.dependent_parameters:
                raise ValueError(f'The parameter {param_name} does not exist in the data.')
            dependencies = self.dependent_parameters[param_name]['paramspec'].depends_on_
            x_param_name = dependencies[0]
            y_param_name = dependencies[1]
            for key, param in self.independent_parameters.items():
                if param['paramspec'].name == x_param_name: x_key = key
                if param['paramspec'].name == y_param_name: y_key = key
            x_param = self.independent_parameters[x_key]
            y_param = self.independent_parameters[y_key]
            z_param = self.dependent_parameters[param_name]
            x_vals = x_param['values']
            y_vals = y_param['values']
            
            x_slice = slice(0,None)
            y_slice = slice(0,None)
            if x_range: x_slice = self.find_slice(x_vals, x_range)
            if y_range: y_slice = self.find_slice(y_vals, y_range)
            
            x_plot = np.tile(x_vals,(len(y_vals),1))[y_slice, x_slice]
            y_plot = np.tile(y_vals,(len(x_vals),1)).T[y_slice, x_slice]
            z_plot = z_param['values'][y_slice, x_slice]

            fig, ax = plt.subplots()
            ax, cb = qc.dataset.plotting.plot_on_a_plain_grid(x_plot, y_plot, z_plot, ax, cmap=cmap, **kwargs)
            ax.set_xlabel(x_param['paramspec'].label + f" ({x_param['paramspec'].unit})")
            ax.set_ylabel(y_param['paramspec'].label + f" ({y_param['paramspec'].unit})")
            cb.set_label(z_param['paramspec'].label + f" ({z_param['paramspec'].unit})")
            ax.set_title(plot_title)
            figs.append(fig)
            axes.append(ax)
            cbs.append(cb)
            ds_plot_data = self._generate_DSPlotData(x_param, y_param, z_param)
            _rescale_ticks_and_units(ax, ds_plot_data, cb)
        return figs, axes, cbs

    def _generate_DSPlotData(self, *param_dicts):
        """
        Generates a list of DSPlotData dictionaries based on the given parameter dictionaries.

        Args:
            *param_dicts: Variable number of parameter dictionaries of the form {'name': String, 'paramspec': ParamSpec, 'values': numpy.ndarray}

        Returns:
            list: A list of DSPlotData dictionaries, where each dictionary contains the following keys:
                - name (str): The name of the parameter.
                - unit (str): The unit of measurement for the parameter.
                - label (str): The label for the parameter.
                - data (list): The values associated with the parameter.
                - shape (None): The shape of the parameter data (currently set to None).

        """
        ds_plot = []
        for param in param_dicts:
            param_spec_base = param['paramspec']
            my_data_dict: DSPlotData = {
                "name": param_spec_base.name,
                "unit": param_spec_base.unit,
                "label": param_spec_base.label,
                "data": param['values'],
                "shape": None,
            }
            ds_plot.append(my_data_dict)
        return ds_plot

    
    def find_slice(self, array, values):
        idx_min = (np.abs(array - values[0])).argmin() if values[0] else 0
        idx_max = (np.abs(array - values[1])).argmin() + 1 if values[1] else -1
        return slice(idx_min,idx_max)

