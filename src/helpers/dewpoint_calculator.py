from typing import Union

import numpy as np

a = 7.5  # Parameter for T > 0°C
b = 237.3  # Parameter for T > 0°C


def saturation_vapor_pressure(t: Union[np.array, float]) -> Union[np.array, float]:
    """
    Calculate the saturation vapor pressure of water.

    Parameters
    ----------
    t: Union[np.array, float]
        The temperature in °C to calculate the saturation vapor pressure for.

    Returns
    -------
    p: Union[np.array, float]
        The resulting saturation vapor pressure.
    """
    return 6.1078 * 10 ** ((a * t) / (b + t))


def vapor_pressure(rh: Union[np.array, float], t: Union[np.array, float]) -> Union[np.array, float]:
    """
    Calculate the vapor pressure of water.

    Parameters
    ----------
    rh: Union[np.array, float]
        Relative humidity in percent.
    t: Union[np.array, float]
        The temperature in °C to calculate the vapor pressure for.

    Returns
    -------
    p: Union[np.array, float]
        The resulting vapor pressure.
    """
    return rh / 100 * saturation_vapor_pressure(t)


def calculate_dewpoint(temperature: Union[np.array, float], relative_humidity: Union[np.array, float]) -> Union[np.array, float]:
    """
    Calculate the dew point temperature.

    Parameters
    ----------
    temperature: Union[np.array, float]
        The temperature in °C.
    relative_humidity: Union[np.array, float]
        The relative humidity in percent.

    Returns
    -------
    dewpoint: Union[np.array, float]
        The resulting dew point temperature in °C.
    """

    v = np.log10(vapor_pressure(relative_humidity, temperature) / 6.1078)
    dewpoint = b * v / (a - v)
    return dewpoint


def calculate_relative_humidity(temperature: Union[np.array, float], dewpoint: Union[np.array, float]) -> Union[np.array, float]:
    """
    Calculate the relative humidity.

    Parameters
    ----------
    temperature: Union[np.array, float]
        The temperature in °C.
    dewpoint: Union[np.array, float]
        The dew point temperature in °C.

    Returns
    -------
    relative_humidity: Union[np.array, float]
        The resulting relative humidity in percent.
    """

    relative_humidity = 100 * saturation_vapor_pressure(dewpoint) / saturation_vapor_pressure(temperature)
    return relative_humidity
