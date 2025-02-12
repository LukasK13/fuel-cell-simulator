# Fuel Cell Simulator
The fuel cell simulator provides a graphical user interface to predict the cell voltage of a fuel cell based on its operating conditions.
The simulator uses a neural network based digital twin that was developed for fuel cell performance prediction.
You can find more information about the model and its validation in our peer reviewed article "Klass et al., *The Voltage Oracle: A Foundation Model for Probabilistic PEM Fuel Cell Voltage Prediction*, J. Electrochem. Soc. 2025,  [10.1149/1945-7111/adad43](https://iopscience.iop.org/article/10.1149/1945-7111/adad43)".

![Screenshot of the Fuel Cell Simulator](resources/main.png)

## Getting started
To run the fuel cell simulator, you need to have Python installed on your machine.
1. Clone the repository to your local machine.
    ```bash
    git clone
    ```
2. Install the required packages by running the following command in the terminal:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the simulator by executing the following command in the terminal:
    ```bash
    streamlit run src/main.py
    ```
