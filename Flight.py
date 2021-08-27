import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Utils import Utils
from collections import defaultdict


class Flight:
    """
        A class that is used to represent the sensor data of an
        aircraft

        Attributes
        ----------
        aircraft_id         : str
            the aircraft identifier (PH-01, PH-33, etc.)
        filename            : str
           location of the flight on disk
        debug               : str
            the warning print flag
        error               : int
            the error print flag
        colors               : list
            list of strings indicating colors
        dataframe            : str
            the flight dataframe
        min_taxi_warmup_size : int
            minimum size of acceptable taxi warmup stage
    """
    def __init__(self, aircraft_id, flight_dir, debug, error):
        self.colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen', 'pink', 'gray', 'silver',
                       'chartreuse', 'darkturquoise', 'olive']
        self.debug = debug
        self.error = error
        self.aircraft_id = aircraft_id
        self.filename = flight_dir
        self.df = pd.read_parquet(flight_dir)
        self.df.sort_values(by="row_number", inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.min_taxi_warmup_size = 60
        self.min_taxi_high_turbulence_size = 150
        self.flight_stages = []

    def get_size(self):
        return len(self.df)

    def get_date(self):
        return pd.to_datetime(self.df['departure_datetime'].values[0], format='%Y-%m-%d %H:%M:%S')

    def plot_binary_signals(self):
        """Plots the binary signals of the flight
        """
        time = self.df['row_number'].values
        for i, on in zip(range(8), ['ON', 'ON', 'ON', 'ON', 'ON', 'ON', 'ON', 'OPEN']):
            col_binary = self.df.columns[i]
            param_binary = (self.df[col_binary].values == on) * (-0.1 * i)
            plt.plot(time, param_binary, color=self.colors[i], label=col_binary, lw=3)

    def plot(self):
        """Plots (generic) the signals of the flight
        """
        time = self.df['row_number'].values
        taxi_warmup = self.get_taxi_warmup_mask()
        taxi_warmup_turbulence = self.get_taxi_warmup_turbulence_mask()
        taxi_high_turbulence = self.get_taxi_high_turbulence_mask()
        for i in range(8, len(self.df.columns)-4):
            col_name = self.df.columns[i]
            param_orig = self.df[col_name].values
            print(col_name, param_orig)
            param = Utils.normalize(param_orig)
            plt.plot(time, param, color=self.colors[i-8], label=col_name)
            if col_name == 'pressure_altitude':
                flight_stages = np.unique(self.flight_stages)
                for flight_stage, j in zip( flight_stages, range(len(flight_stages))):
                    mask_stage = self.flight_stages == flight_stage
                    plt.scatter(time[mask_stage], param[mask_stage], color=self.colors[j], label=flight_stage, lw=5)
            if i == 8:
                if np.sum(taxi_warmup*1) > 0:
                    plt.scatter(time[taxi_warmup], [np.max(param[taxi_warmup])] * len(time[taxi_warmup]), color='yellow', lw=4)
                if np.sum(taxi_warmup_turbulence * 1) > 0:
                    plt.scatter(time[taxi_warmup_turbulence], [np.max(param[taxi_warmup_turbulence])*1.5] * len(time[taxi_warmup_turbulence]), color='blue', lw=4)
                if np.sum(taxi_high_turbulence * 1) > 0:
                    plt.scatter(time[taxi_high_turbulence], [np.max(param[taxi_high_turbulence])*2] * len(time[taxi_high_turbulence]), color='pink', lw=4)
        self.plot_binary_signals()
        plt.title("Aircraft=%s" % self.aircraft_id)
        plt.tight_layout()
        plt.grid(linestyle='-')
        plt.legend()
        plt.show()

    def get_taxi_high_turbulence_mask(self):
        """Gets a boolean mask to apply on the flight sensors to obtain the taxi warmup stage
            :return mask: boolean mask that filters the warmup taxi stage
            :rtype: list of bool
        """
        #mask1 = np.logical_or((self.df['par3_sys_1'].values == 'ON'), (self.df['par3_sys_2'].values == 'OFF'))
        mask1 = np.logical_and((self.df['par3_sys_1'].values == 'ON'), (self.df['par3_sys_2'].values == 'ON'))
        mask2 = np.logical_and((self.df['par5_sys_gen'].values == 'OPEN'), (self.df['par4_sys_2'].values == 'ON'))
        mask3 = np.logical_and((self.df['par4_sys_1'].values == 'ON'), (self.df['par2_sys_1'].values == 'OFF'))
        mask4 = np.logical_and((self.df['par2_sys_2'].values == 'OFF'), (self.df['par1_sys_gen'].values == 'OFF'))
        mask = mask1 & mask2 & mask3 & mask4
        mask[int(len(mask)*0.5):] = False
        mask_final = [False] * len(mask)
        found = False

        for elm1, elm2, i in zip(mask[:-1], mask[1:], range(len(mask)-1)):
            if elm1 and elm2:
                mask_final[i] = True
                found = True
            elif found:
                mask_final[i] = False
                break
        if np.sum(mask_final) > self.min_taxi_high_turbulence_size:
            return mask_final
        return [False] * len(mask)

    def get_taxi_warmup_mask(self):
        """Gets a boolean mask to apply on the flight sensors to obtain the taxi warmup stage
            :return mask: boolean mask that filters the warmup taxi stage
            :rtype: list of bool
        """
        #mask1 = np.logical_or((self.df['par3_sys_1'].values == 'ON'), (self.df['par3_sys_2'].values == 'OFF'))
        mask1 = (self.df['par3_sys_1'].values == 'OFF')
        mask2 = np.logical_and((self.df['par5_sys_gen'].values == 'OPEN'), (self.df['par4_sys_2'].values == 'ON'))
        mask3 = np.logical_and((self.df['par4_sys_1'].values == 'ON'), (self.df['par2_sys_1'].values == 'OFF'))
        mask4 = np.logical_and((self.df['par2_sys_2'].values == 'OFF'), (self.df['par1_sys_gen'].values == 'OFF'))
        mask = mask1 & mask2 & mask3 & mask4
        par6_sys_1 = self.df['par6_sys_1'].values
        par6_sys_2 = self.df['par6_sys_2'].values
        mask_final = [False] * len(mask)

        # ignores very short sequences
        if np.sum(mask*1) <= self.min_taxi_warmup_size:
            return mask_final

        # if founds more than one non-consecutive true sequence
        # in mask, ignores the non-first
        neg_indexes = list(np.where(~mask)[0])
        pos_indexes = list(np.where(mask)[0])
        if len(pos_indexes)== 0 or len(neg_indexes) == 0:
            return mask_final
        first_neg_index = neg_indexes[0]
        first_pos_index = pos_indexes[0]
        if first_neg_index < first_pos_index:
            second_neg_index = first_pos_index
            for elm in np.where(~mask)[0][first_pos_index:]:
                if elm is False: break
                else: second_neg_index += 1
            mask[second_neg_index:] = False
        else:
            mask[first_neg_index:] = False

        # ignores very short first sequences
        if np.sum(mask*1) <= self.min_taxi_warmup_size:
            return mask_final

        par6_sys_1_taxi = self.df.loc[mask == True, 'par6_sys_1'].values
        par6_sys_2_taxi = self.df.loc[mask == True, 'par6_sys_2'].values

        len_taxi = len(par6_sys_1_taxi)

        found_index = -1

        alpha = 0.20
        norm_par6_sys_1 = Utils.normalize(par6_sys_1)
        for center_percentage in range(50, 15, -5):
            center_percentage /= 100
            center_taxi_index = int(len(par6_sys_1_taxi) * center_percentage)
            low_index = max(0, int(center_taxi_index * (1 - alpha)))
            high_index = min(len_taxi, int(center_taxi_index * (1 + alpha)))
            derivative_par6_sys_1 = Utils.calculate_derivative_signal(norm_par6_sys_1, 20)[low_index:high_index]
            if np.max(derivative_par6_sys_1) > 0.02:
                #print(center_percentage, center_taxi_index, low_index, high_index)
                continue

        center_max_value = np.max(par6_sys_1_taxi[low_index:high_index])
        center_max_value2 = np.max(par6_sys_2_taxi[low_index:high_index])
        center_min_value = np.min(par6_sys_1_taxi[low_index:high_index])
        center_min_value2 = np.min(par6_sys_2_taxi[low_index:high_index])

        #if center_max_value > 3 or center_max_value2 > 3:
            #return mask_final

        if False:
            print('Len taxi=', len_taxi, 'center index=', center_taxi_index, 'center max value = ', center_max_value,
                  'high_index=', high_index, 'low_index=', low_index, 'center max value', center_max_value2)

        for mask_val, i in zip(mask, range(len(mask))):
            if mask_val and not (found_index > 0 and i > found_index + 1):
                mask_final[i] = True
                if par6_sys_1[i] > center_max_value + 0.5*abs(center_max_value) and par6_sys_2[i] > center_max_value2 + 0.5*abs(center_max_value2):
                    mask_final[i] = False
                if par6_sys_1[i] < center_min_value - 0.5*abs(center_min_value) and par6_sys_2[i] < center_min_value2 - 0.5*abs(center_min_value2):
                    mask_final[i] = False
                found_index = i
            elif not mask_val and found_index != -1 and i > found_index + 1:
                break

        if len(par6_sys_1[mask_final])< self.min_taxi_warmup_size or (np.std(par6_sys_1[mask_final]) > 10) or np.max(Utils.normalize(par6_sys_1)[mask_final]) > 0.10:
            return [False] * len(mask)

        return np.array(mask_final)

    def get_taxi_warmup_turbulence_mask(self):
        """Gets a boolean mask to apply on the flight sensors to obtain the taxi warmup more turbulent stage
            :return mask: boolean mask that filters the warmup taxi stage
            :rtype: list of bool
        """
        mask1 = (self.df['par3_sys_1'].values == 'OFF')
        mask2 = np.logical_and((self.df['par5_sys_gen'].values == 'OPEN'), (self.df['par4_sys_2'].values == 'ON'))
        mask3 = np.logical_and((self.df['par4_sys_1'].values == 'ON'), (self.df['par2_sys_1'].values == 'OFF'))
        mask4 = np.logical_and((self.df['par2_sys_2'].values == 'OFF'), (self.df['par1_sys_gen'].values == 'OFF'))
        mask = mask1 & mask2 & mask3 & mask4
        mask[int(len(mask) * 0.5):] = False

        taxi_warmup_mask = self.get_taxi_warmup_mask()
        mask_final = [False] * len(mask)
        i = 0
        found = False
        par6_sys_1 = self.df['par6_sys_1'].values
        norm_par6_sys_1 = Utils.normalize(par6_sys_1)
        for elm1, elm2 in zip(mask, taxi_warmup_mask):
            if elm1 and elm2:
                found = True
            if elm1 and found and elm1 != elm2 and norm_par6_sys_1[i] < 0.20:
                mask_final[i] = True
            i += 1

        if np.sum(mask_final*1) == 0:
            return mask_final

        mask_final2 = [False] * len(mask)
        found = False
        for elm1, elm2, i in zip(mask_final[:len(mask_final)-1], mask_final[1:], range(len(mask_final)-1)):
            if elm1 and elm2:
                mask_final2[i] = True
                found = True
            elif not (elm1 and elm2) and found:
                break
        if sum(mask_final2*1) > self.min_taxi_warmup_size:
            return mask_final2
        return np.array([False] * len(mask))

    def get_taxi_warmup_turbulence(self):
        """Gets a dictionary of signals with taxi more turbulent warmup stage
                :return signals: dictionary with the warmup taxi stage [more turbulence]
                :rtype: dictionary of floats
        """
        mask = self.get_taxi_warmup_turbulence_mask()
        return self.apply_mask(mask)

    def get_taxi_high_turbulence(self):
        """Gets a dictionary of signals with taxi more turbulent stage
                :return signals: dictionary with the warmup taxi stage
                :rtype: dictionary of floats
        """
        mask = self.get_taxi_high_turbulence_mask()
        return self.apply_mask(mask)

    def get_taxi_stage(self, dataframe):
        mask = np.logical_or((dataframe['par3_sys_1'].values == 'OFF'), (dataframe['par3_sys_2'].values == 'OFF'))
        mask2 = np.logical_and((dataframe['par5_sys_gen'].values == 'OPEN'), (dataframe['par4_sys_2'].values == 'ON'))
        mask3 = np.logical_and((dataframe['par4_sys_1'].values == 'ON'), (dataframe['par2_sys_1'].values == 'OFF'))
        mask4 = np.logical_and((dataframe['par2_sys_2'].values == 'OFF'), (dataframe['par1_sys_gen'].values == 'OFF'))
        mask5 = mask & mask2 & mask3 & mask4
        par6_sys_1 = dataframe['par6_sys_1'].values
        par6_sys_2 = dataframe['par6_sys_2'].values
        par6_sys_1_taxi = dataframe.loc[mask5 == True, 'par6_sys_1'].values
        par6_sys_2_taxi = dataframe.loc[mask5 == True, 'par6_sys_2'].values
        len_taxi = len(par6_sys_1_taxi)
        center_taxi_index = int(len(par6_sys_1_taxi) * 0.5)
        found_yellow, found_index = True, -1
        stage_colors = [0] * len(mask5)
        if len_taxi <= 5:
            return np.array(stage_colors)

        alpha = 0.15
        low_index = max(0, int(center_taxi_index * (1 - alpha)))
        high_index = min(len_taxi, int(center_taxi_index * (1 + alpha)))

        # print(low_index, high_index, par6_sys_1_taxi)

        center_max_value = np.max(par6_sys_1_taxi[low_index:high_index]) * 1.2
        center_max_value2 = np.max(par6_sys_2_taxi[low_index:high_index]) * 1.2
        if center_max_value > 5 or center_max_value2 > 5:
            return np.array(stage_colors)

        for val, i in zip(mask5, range(len(mask5))):
            if val and not (found_index != -1 and i > found_index + 1):
                stage_colors[i] = 1
                if par6_sys_1[i] > center_max_value and par6_sys_2[i] > center_max_value2:
                    if i < center_taxi_index:
                        stage_colors[i] = 2
                    else:
                        stage_colors[i] = 3
                found_index = i
            elif not val and found_index != -1 and i > found_index + 1:
                break
        return np.array(stage_colors)

    def get_taxi_warmup(self):
        """Gets a dictionary of signals with taxi warmup stage
                :return signals: dictionary with the warmup taxi stage
                :rtype: dictionary of floats
        """
        mask = self.get_taxi_warmup_mask()
        return self.apply_mask(mask)

    def get_stages(self):
        """Gets a dictionary of signals with taxi warmup stage
                :return signals: dictionary with the warmup taxi stage
                :rtype: dictionary of floats
        """
        from Flight_Stages import Fligh_Stages
        flight_stage_processor = Fligh_Stages()
        self.flight_stages = flight_stage_processor.process_flight_aircraft(self.df, self.aircraft_id)

    def get_climb(self):
        if len(self.flight_stages) == 0:
            self.get_stages()
        stage='CLIMB'
        mask = self.flight_stages == stage
        return self.apply_mask(mask)

    def get_cruise(self):
        if len(self.flight_stages) == 0:
            self.get_stages()
        stage='VALID CRUISE SUB'
        mask = self.flight_stages == stage
        return self.apply_mask(mask)

    def get_descent(self):
        if len(self.flight_stages) == 0:
            self.get_stages()
        stage='DESCENT'
        mask = self.flight_stages == stage
        return self.apply_mask(mask)

    def apply_mask(self, mask):
        """Get a masked stage
                :return signals: dictionary with a masked taxi stage
                :rtype: dictionary of floats
        """
        signals = defaultdict(float)
        self.dataframe = self.df
        for i in [6, 7, 8, 9]:
            param_name = 'par' + str(i) + '_sys_' + str(1)
            param_name2 = 'par' + str(i) + '_sys_' + str(2)
            if np.sum(mask*1) > 0:
                param = self.df.loc[mask, param_name].values
                param2 = self.df.loc[mask, param_name2].values
            if np.sum(mask*1) > 0 and len(param) > 1 and len(param2) > 1:
                mean_diff = np.sum(param - param2)/len(param)
                max_median = max(np.median(param), np.median(param2))
                if max_median > 40:
                    max_median = np.nan
                param_median = np.median(param)
                if param_median > 40:
                    param_median = np.nan
                param_median2 = np.median(param2)
                if param_median2 > 40:
                    param_median2 = np.nan

                #print('Mean diff', mean_diff)
                #self.plot()

                signals['Meandiff_' + param_name] = mean_diff
                signals['DiffMedian_' + param_name] = param_median2 - param_median
                signals['MaxMedian_' + param_name] = max_median
                signals['Paramedian_' + param_name] = param_median2
            else:
                signals['Meandiff_' + param_name] = np.nan
                signals['DiffMedian_' + param_name] = np.nan
                signals['MaxMedian_' + param_name] = np.nan
                signals['Paramedian_' + param_name] = np.nan
        return signals