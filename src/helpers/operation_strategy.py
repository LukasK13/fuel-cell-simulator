import numpy as np
import pandas as pd

from src.helpers.dewpoint_calculator import calculate_dewpoint


MOLAR_MASS_O2 = 31.9988
MOLAR_MASS_AIR = 28.97
MOLAR_MASS_WATER = 18.01528
DENSITY_N2 = 1.165
DENSITY_H2 = 0.08988
O2_FRACTION_AIR = 0.21


def f_t_in(x: np.array):
    return (x / 0.5 * 5 + 61) * (x <= 0.5) + ((x - 0.5) / 2 * 4 + 66) * (x > 0.5)


def f_delta_t(x: np.array):
    return x / 2.5 * 11 + 1


# Achtung: f_p_cat gibt den absoluten Druck an
def f_p_cat(x: np.array):
    return 1.4 / 2.5 ** 2 * x ** 2 + 1.1


# Achtung: f_p_an gibt den absoluten Druck an
def f_p_an(x: np.array):
    return 1.4 / 2.5 ** 2 * x ** 2 + 1.3


def f_pd_cat(x: np.array):
    return x / 2.5 * 0.35 + 0.05


def f_pd_an(x: np.array):
    return x / 2.5 * 0.2 + 0.05


# Achtung: Input hier min_gas_flow
def f_stoich_cat(x: np.array):
    exp = -3
    a = 1.6/(1-np.exp(exp * 0.5))
    b = 3.5-a
    return np.clip(a * np.exp(exp * x) + b, 1.9, None)


# Achtung: Input hier min_gas_flow
def f_stoich_an(x: np.array):
    exp = -3
    a = 1 / (1 - np.exp(exp * 0.5))
    b = 2.5 - a
    return np.clip(a * np.exp(exp * x) + b, 1.5, None)


def f_min_stoich_current(x: np.array):
    min_stoich_current = np.clip(x, 0.24, None)
    return min_stoich_current


def p_ws_magnus(t: np.array):
    return 0.61094 * np.exp(17.625 * t / (t + 243.04)) / 100


def specific_humidity(t: np.array, rh: np.array, p: np.array):
    p_ws = p_ws_magnus(t)
    return 0.622 * p_ws * rh / (p - rh * p_ws)


def specific_humidity_inv(t: np.array, y: np.array, p: np.array):
    p_ws = p_ws_magnus(t)
    return y / (0.622 + y) * p / p_ws


def f_t_dew_an_in(x: np.array, rh_out: np.array = 1.0, conc_n: np.array = 15.0):
    t_in = f_t_in(x)
    t_out = t_in + f_delta_t(x)
    p_in = f_p_an(x)
    p_out = np.clip(p_in - f_pd_an(x), 1, None)
    stoich = f_stoich_an(x)

    y_anode_out = specific_humidity(t_in, rh_out, p_out)
    y_anode_in = y_anode_out * (1 - 1 / (stoich * (1 + conc_n / 100 * DENSITY_N2 / DENSITY_H2)))
    rh_anode_in = specific_humidity_inv(t_out, y_anode_in, p_in)
    t_dew_anode_in = calculate_dewpoint(t_out, rh_anode_in * 100)
    return t_dew_anode_in


def f_t_dew_cat_in(x: np.array, rh_out: float = 1.0):
    t_in = f_t_in(x)
    t_out = t_in + f_delta_t(x)
    p_in = f_p_cat(x)
    p_out = np.clip(p_in - f_pd_cat(x), 1, None)
    stoich = f_stoich_cat(x)

    y_cathode_out = specific_humidity(t_out, rh_out, p_out)
    y_cathode_in = (y_cathode_out * (
            stoich / O2_FRACTION_AIR - MOLAR_MASS_O2 / MOLAR_MASS_AIR) - 2 * MOLAR_MASS_WATER / MOLAR_MASS_AIR) * O2_FRACTION_AIR / stoich
    rh_cathode_in = specific_humidity_inv(t_in, y_cathode_in, p_in)
    t_dew_cathode_in = calculate_dewpoint(t_in, rh_cathode_in * 100)
    return t_dew_cathode_in


def calculate_operating_conditions(current_density: np.array):
    rh_cathode_out = 1.0
    rh_anode_out = 0.9
    conc_n = 15

    load_points = pd.DataFrame({'current_density': current_density})
    load_points['min_stoich_current_density'] = f_min_stoich_current(current_density)
    load_points['temp_in'] = f_t_in(current_density)
    load_points['delta_t'] = f_delta_t(current_density)
    load_points['conc_n'] = conc_n
    load_points['t_dew_a'] = f_t_dew_an_in(current_density, rh_anode_out, conc_n)
    load_points['t_dew_c'] = f_t_dew_cat_in(current_density, rh_cathode_out)
    load_points['p_cat_in'] = f_p_cat(current_density)
    load_points['p_an_in'] = f_p_an(current_density)
    load_points['stoich_cat'] = f_stoich_cat(f_min_stoich_current(current_density))
    load_points['stoich_an'] = f_stoich_an(f_min_stoich_current(current_density))

    return load_points
