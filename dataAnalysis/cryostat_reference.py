"""Utilities to load RF line reference measurements from QCoDeS databases.

This module provides the Reference class which knows about predefined
calibration and background runs for different cryostats and returns a
FrequencyScanVNA instance normalized and phase-corrected for a requested
RF line.
"""

import qcodes as qc
from qcodes.dataset.sqlite.database import connect
from dataAnalysis.dataset import FrequencyScanVNA
import os
import numpy as np


class Reference:
    """Container for cryostat-specific RF line reference data.

    Args:
        cryostat: Name of the cryostat (e.g. 'ERiC' or 'Mistral').
        db_path: Optional base path to QCoDeS database files. Defaults to r'C:\Users\hqclabo\Documents\Data\qcodes_databases'.
    """
    def __init__(self, cryostat: str, db_path: str=r'C:\Users\hqclabo\Documents\Data\qcodes_databases'):
        self.cryostat = cryostat

        self.db_path = db_path

        if self.cryostat == 'ERiC':
            self.database_files = [
                os.path.join(self.db_path, '2025_07_15_Eric_fridge_update.db'),
                os.path.join(self.db_path, '2024_08_29_ERiC_RF_Calibration.db'),
            ]
            self.reference_runs = {
                'A1_high': '1_104',
                'A1_low': '1_94',
                'A2': '1_105',
                'A3': '1_106',
                'A4': '1_107',
                'B6': '2_77',
            }
            self.background_runs = {
                'A1_high': '1_28',
                'A1_low': '1_28',
                'A2': '1_28',
                'A3': '1_28',
                'A4': '1_28',
                'B6': '2_9',
            }
            self.phase_corrections = {
                'A1_high': dict(delay=73.7e-9, alpha=2),
                'A1_low': dict(delay=73.3e-9, alpha=0),
                'A2': dict(delay=71.85e-9, alpha=0.5),
                'A3': dict(delay=72.33e-9, alpha=0),
                'A4': dict(delay=72.29e-9, alpha=0.6),
                'B6': dict(delay=-0.55e-9, alpha=np.pi),
            }

        elif self.cryostat == 'Mistral':
            self.database_files = [
                os.path.join(self.db_path, '2025_07_16_Mistral_rf_line_calibration_after_rearrangements.db'),
            ]
            self.reference_runs = {
                'A1_high': '1_64',    
                'A1_low': '1_71',
                'A2': '1_65',
                'A3': '1_66',
                'A4': '1_67',
                'B6': '1_77',
            }
            self.background_runs = {
                'A1_high': '1_4',
                'A1_low': '1_4',
                'A2': '1_4',
                'A3': '1_4',
                'A4': '1_4',
                'B6': '1_4',
            }
            self.phase_corrections = {
                'A1_high': dict(delay=0e-9, alpha=0),
                'A1_low': dict(delay=0e-9, alpha=0),
                'A2': dict(delay=0e-9, alpha=0),
                'A3': dict(delay=0e-9, alpha=0),
                'A4': dict(delay=0e-9, alpha=0),
                'B6': dict(delay=0e-9, alpha=0),
            }

        connections = [connect(file) for file in self.database_files]
        self.exps = [qc.load_experiment(1, conn=conn) for conn in connections]


    def get_rf_line_reference(self, line: str):
        """Return a FrequencyScanVNA for the given RF line.

        The returned FrequencyScanVNA is normalized against the stored
        background run and has the configured delay/phase correction applied.

        Args:
            line: Key identifying the RF line (e.g. 'A1_high', 'B6').

        Returns:
            FrequencyScanVNA: normalized and phase-corrected dataset.
        """

        run = self.background_runs[line]
        exp_id, run_id = [int(id) for id in run.split('_')]
        ds_bg = FrequencyScanVNA(exp=self.exps[exp_id-1], run_id=run_id)

        run = self.reference_runs[line]
        exp_id, run_id = [int(id) for id in run.split('_')]
        ds_line = FrequencyScanVNA(exp=self.exps[exp_id-1], run_id=run_id)
        ds_line.normalize_data_vna(ds_bg=ds_bg)
        ds_line.delay_correction(**self.phase_corrections[line])

        for param in ['param_0', 'param_1']:
            ds_line.dependent_parameters[param]['values'] = ds_line.dependent_parameters[param+'_normalized']['values']
            del(ds_line.dependent_parameters[param+'_normalized'])
        ds_line.mag = ds_line.mag_norm
        ds_line.phase = ds_line.phase_norm
        ds_line.cData = ds_line.cData_norm
        del(ds_line.mag_norm)
        del(ds_line.phase_norm)
        del(ds_line.cData_norm)

        return ds_line

