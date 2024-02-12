# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:42:41 2022

@author: fopplige
"""

import Labber
import numpy as np


class LabberFile:
    """ Class to easily extract data from a Labber log file. """
    def __init__(self,file_name):
        """
        Get log file and get names of stepped and logged channels.

        Parameters
        ----------
        file_name : string
            file name of the Labber log file (incl. file path).
            
        """
        self.log_file = Labber.LogFile(file_name)
        self.channels_step = [self.log_file.getStepChannels()[ii]['name'] for ii in range(len(self.log_file.getStepChannels()))]
        self.channels_log = [self.log_file.getLogChannels()[ii]['name'] for ii in range(len(self.log_file.getLogChannels()))]
    
    def get_step_channel_data(self,channel_list):
        """
        get data of stepped channels using Labber.LogFile.getData

        Parameters
        ----------
        channel_list : list of integers or strings
            List of stepped channels to be extracted provided indexes
            or channel names.

        Returns
        -------
        tuple
            Data of stepped channels.

        """
        channels = []
        for channel in channel_list:
            if type(channel)==int:
                channels.append(self.log_file.getData(self.channels_step[channel]))
            else:
                channels.append(self.log_file.getData(channel))
        return tuple(channels)
    
    def get_step_channel_traceXY(self,channel_list):
        """
        get data of stepped channels using Labber.LogFile.getTraceXY

        Parameters
        ----------
        channel_list : list of integers or strings
            List of stepped channels to be extracted provided indexes
            or channel names.

        Returns
        -------
        tuple
            Data of stepped channels.

        """
        channels = []
        for channel in channel_list:
            if type(channel)==int:
                channels.append(self.log_file.getTraceXY(self.channels_step[channel])[0])
            else:
                channels.append(self.log_file.getTraceXY(channel)[0])
        return tuple(channels)
    
    def get_log_channel_data(self,channel_list):
        """
        get data of logged channels using Labber.LogFile.getData

        Parameters
        ----------
        channel_list : list of integers or strings
            List of stepped channels to be extracted provided indexes
            or channel names.

        Returns
        -------
        tuple
            Data of logged channels.

        """
        channels = []
        for channel in channel_list:
            if type(channel)==int:
                channels.append(self.log_file.getData(self.channels_log[channel]))
            else:
                channels.append(self.log_file.getData(channel))
        return tuple(channels)

    def get_log_channel_traceXY(self,channel_list):
        """
        get data of logged channels using Labber.LogFile.getTraceXY

        Parameters
        ----------
        channel_list : list of integers or strings
            List of stepped channels to be extracted provided indexes
            or channel names.

        Returns
        -------
        tuple
            Data of logged channels.

        """
        channels = []
        for channel in channel_list:
            if type(channel)==int:
                channels.append(self.log_file.getTraceXY(self.channels_log[channel])[0])
            else:
                channels.append(self.log_file.getTraceXY(channel)[0])
        return tuple(channels)
    
def slice_data(data, outer_dim, inner_dim):
    """
    

    Returns
    -------
    None.

    """
    data_sliced = np.zeros([outer_dim,inner_dim,data.shape[1]])
    for ii in range(outer_dim):
        print(data_sliced.shape)
        data_sliced[ii] = data[ii:ii+inner_dim]
    return data_sliced
        


