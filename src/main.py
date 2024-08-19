import pandas as pd
import streamlit as st
import onnxruntime
import plotly.express as px

from helpers.dewpoint_calculator import calculate_dewpoint
from helpers.operation_strategy import calculate_operating_conditions

st.set_page_config(layout="wide")
st.title('Fuel Cell Simulator')

calc_tab, dewpoint_tab = st.tabs(['⚡ Fuel Cell Simulator', '💦 Dew Point Calculator'])

with calc_tab:
    with st.expander("❓ Help"):
        st.markdown("""
        This page allows to predict the cell voltage of a fuel cell based on the operating conditions.
        The model is based on an ONNX model trained on simulated fuel cell data. You can find more information about the model in the [Reference Paper](link).
        
        #### How to use the simulator:
        To run the simulation, the operating conditions of the fuel cell need to be provided.
        You can either provide the operating conditions directly or decide to use the system operating conditions based on the current density.
        
        A demo load profile can be loaded by clicking on the button `📂 Open Load Profile`
        
        The following parameters are required:
        1. **Current Density**: The current density in A/cm².
        2. **Cathode Inlet Pressure**: The absolute cathode inlet pressure in bara.
        3. **Anode Inlet Pressure**: The absolute anode inlet pressure in bara.
        4. **Anode Stoichiometry**: The anode stoichiometry.
        5. **Cathode Stoichiometry**: The cathode stoichiometry.
        6. **Inlet Temperature**: The coolant inlet temperature in °C.
        7. **Coolant Temperature Difference**: The coolant temperature difference in °C.
        8. **Anode Nitrogen Concentration**: The anode nitrogen concentration in %.
        9. **Cathode Dew Point**: The cathode dew point in °C.
        10. **Anode Dew Point**: The anode dew point in °C.
        11. **Minimum Stoichiometric Current Density**: The minimum stoichiometric current density in A/cm². The maximum of minimum stoichiometric current density and the current density is used for the stoichiometric flow calculation. This is useful for low current densities where the stoichiometric flow would be too low.
        
        The table below allows to input the operating conditions and provides a button for downloading the load profile.
        The downloaded file can be used to upload the load profile in the future using the upload button.
        
        After providing the operating conditions, click on the calculate button to predict the cell voltage.
        The model predicts the cell voltage and the confidence interval based on the operating conditions.
        The confidence interval represents the model's prediction of the 1st sigma interval. Thereby the model output can be treated a gaussian distribution of the cell voltage.
        The results are displayed in a table and a scatter plot. The scatter plot shows the predicted cell voltage with the confidence interval.
        
        """)

    st.header('1. Input Parameters 🎚️')
    with st.popover('📂 Open Load Profile', help='Upload a csv file with the load profile or load the demo load profile.'):
        tab1, tab2 = st.tabs(['🔮 Demo', '⬆️ Upload Load Profile'])
        with tab1:
            demo_file = None
            if st.button('🔮 Load Demo Polarization Curve', type='primary',
                         help='Load the demo polarization curve to test the model.'):
                demo_file = 'model/demo.csv'
        with tab2:
            file = st.file_uploader("Upload Load Profile", type=["csv"],
                                    help="Upload a csv file with the load profile.")

    # Load points
    if file is not None or demo_file is not None:
        # Load from file
        if file is not None:
            points = pd.read_csv(file)
        else:
            points = pd.read_csv(demo_file)
        if points.shape[1] == 1:
            points = calculate_operating_conditions(points['current_density'].to_numpy())
    else:
        # Create empty dataframe
        points = pd.DataFrame(
            {'current_density': [], 'p_cat_in': [], 'p_an_in': [], 'stoich_an': [], 'stoich_cat': [], 'temp_in': [],
             'delta_t': [], 'conc_n': [], 't_dew_c': [], 't_dew_a': [], 'min_stoich_current': []})
    if 'points' not in st.session_state.keys() or len(st.session_state.points) == 0:
        st.session_state.points = points

    system_opcons = st.toggle('Use system operating conditions', True,
                              help='Automatically calculate the operating conditions based on the current density using the system operating strategy.')
    if not system_opcons:
        # Show data editor for all input parameters
        load_points = st.data_editor(
            st.session_state.points, hide_index=True, num_rows='dynamic', use_container_width=True, column_config={
                'current_density': st.column_config.NumberColumn("Current Density", required=True, default=0,
                                                                 min_value=0, max_value=3.5, format="%.3f A/cm²",
                                                                 help="Current Density in A/cm²"),
                'p_cat_in': st.column_config.NumberColumn("Cathode Inlet Pressure", required=True, default=1,
                                                          min_value=0, max_value=3.5, format="%.2f bara",
                                                          help="Absolute Cathode Inlet Pressure in bara"),
                'p_an_in': st.column_config.NumberColumn("Anode Inlet Pressure", required=True, default=1,
                                                         min_value=0, max_value=3.5, format="%.2f bara",
                                                         help="Absolute Anode Inlet Pressure in bara"),
                'stoich_an': st.column_config.NumberColumn("Anode Stoichiometry", required=True, default=1.5,
                                                           min_value=0, max_value=3, format="%.2f",
                                                           help="Anode Stoichiometry"),
                'stoich_cat': st.column_config.NumberColumn("Cathode Stoichiometry", required=True, default=2,
                                                            min_value=0, max_value=3, format="%.2f",
                                                            help="Cathode Stoichiometry"),
                'temp_in': st.column_config.NumberColumn("Inlet Temperature", required=True, default=70,
                                                         min_value=40, max_value=95, format="%.1f °C",
                                                         help="Coolant Inlet Temperature in °C"),
                'delta_t': st.column_config.NumberColumn("Coolant Temperature Difference", required=True,
                                                         default=10, min_value=0, max_value=20, format="%.1f °C",
                                                         help="Coolant Temperature Difference in °C"),
                'conc_n': st.column_config.NumberColumn("Nitrogen Concentration", required=True, default=15,
                                                        min_value=0, max_value=50, format="%.1f%%",
                                                        help="Nitrogen Concentration"),
                't_dew_c': st.column_config.NumberColumn("Cathode Dew Point", required=True, default=60,
                                                         min_value=40, max_value=80, format="%.1f °C",
                                                         help="Cathode Dew Point in °C"),
                't_dew_a': st.column_config.NumberColumn("Anode Dew Point", required=True, default=60,
                                                         min_value=40, max_value=80, format="%.1f °C",
                                                         help="Anode Dew Point in °C"),
                'min_stoich_current_density': st.column_config.NumberColumn(
                    "Minimum Stoichiometric Current Density", required=True,
                    default=0.1, min_value=0, max_value=3.5,
                    format="%.3f A/cm²",
                    help="Minimum Stoichiometric Current Density in A/cm². The maximum of minimum stoichiometric current density and the current density is used for the stoichiometric flow calculation. This is useful for low current densities where the stoichiometric flow would be too low."),
            },
            column_order=['current_density', 'p_cat_in', 'p_an_in', 'stoich_an', 'stoich_cat', 'temp_in', 'delta_t',
                          't_dew_c', 't_dew_a', 'conc_n', 'min_stoich_current_density'])
    else:
        # Show data editor only for current density and calculate operating conditions based on current density
        points = st.session_state.points['current_density']
        load_points = st.data_editor(points, hide_index=True, num_rows='dynamic', use_container_width=False,
                                     column_config={
                                         'current_density': st.column_config.NumberColumn(
                                             "Current Density", required=True, default=0, min_value=0,
                                             max_value=3.5, format="%.3f A/cm²", help="Current Density in A/cm²")})
        load_points = calculate_operating_conditions(load_points.to_numpy())

    if st.button('🚀 Calculate', type='primary', disabled=len(load_points) == 0,
                 help='Calculate the cell voltage based on the input parameters.'):
        with st.spinner('Calculating...'):
            # Preprocess data
            inputs = load_points.copy()

            # Calculate outlet temperature
            inputs['temp_out'] = inputs['temp_in'] + inputs['delta_t']
            inputs.drop(columns=['delta_t'], inplace=True)

            # Calculate relative pressure
            inputs['p_cat_in'] = inputs['p_cat_in'] - 1
            inputs['p_an_in'] = inputs['p_an_in'] - 1

            # change column order
            inputs = inputs[['p_cat_in', 'p_an_in', 'stoich_an', 'stoich_cat', 'temp_in', 'temp_out', 'conc_n',
                             't_dew_c', 't_dew_a', 'current_density', 'min_stoich_current_density']]

            # load onnx model
            sess = onnxruntime.InferenceSession("model/model.onnx")
            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name

            # Predict data
            predictions = sess.run([output_name], {input_name: inputs.to_numpy().astype('float32')})[0]

            # Rescale data
            load_points['cell_voltage'] = predictions[:, 1] * 1000
            load_points['confidence'] = predictions[:, 0] * 1000

            st.header('2. Results 📈')
            result_col1, result_col2 = st.columns(2)
            with result_col1:
                st.dataframe(load_points[['current_density', 'cell_voltage', 'confidence']], use_container_width=True,
                             column_config={
                                 'current_density': st.column_config.NumberColumn("Current Density",
                                                                                  format="%.2f A/cm²",
                                                                                  help="Current Density as given in the input parameters in A/cm²"),
                                 'cell_voltage': st.column_config.NumberColumn("Cell Voltage", format="%.1f mV",
                                                                               help="Predicted Cell Voltage in mV"),
                                 'confidence': st.column_config.NumberColumn("Confidence", format="%.2f mV",
                                                                             help="Prediction confidence (sigma interval) in mV")})
                st.markdown(f'**Average prediction confidence:** {load_points["confidence"].mean():.2f} mV')
            with result_col2:
                fig = px.scatter(load_points, x="current_density", y="cell_voltage", error_y="confidence",
                                 title='Polarization Plot',
                                 custom_data=["current_density", "cell_voltage", "confidence"])
                fig.update_traces(hovertemplate='j = %{x:.3f} A/cm² <br>U = %{y:.1f} mV ± %{customdata[2]:.2f} mV')
                fig.update_layout(xaxis_title='Current Density in A/cm²', yaxis_title='Cell Voltage in mV')
                st.plotly_chart(fig, use_container_width=True)

            st.download_button('⬇️ Download Results', load_points.to_csv(index=False), 'results.csv',
                               help='Download the results as a csv file.')

with dewpoint_tab:
    with st.expander("❓ Help"):
        st.markdown("""
        This page allows to compute the dewpoint of a gas based on the relative humidity. The calculations are based on https://www.wetterochs.de/wetter/feuchte.html.
        1. Enter the gas temperature  in °C.
        3. Enter the relative humidity in %.
        """)

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.number_input("🌡️ Temperature in °C", value=50.0, min_value=0.0, step=0.1,
                                      help="Enter the gas temperature in °C.")
    with col2:
        relative_humidity = st.number_input("Relative Humidity in %", value=30.0, min_value=0.0, max_value=100.0,
                                            step=0.1,
                                            help="Enter the relative humidity in %.")

    dewpoint = calculate_dewpoint(temperature, relative_humidity)
    st.metric(label="Dewpoint", value=f"{dewpoint:.1f} °C")
