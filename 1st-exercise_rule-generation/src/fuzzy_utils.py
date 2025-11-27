from pathlib import Path
import pandas as pd
from skfuzzy import control as ctrl
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz


def charge_data(data: Path, column_to_target: str) -> pd.DataFrame:
    """Auxiliary function for loading and initial preparation of data.

        Reads the CSV file, filters the column of interest and renames it to the pattern
        expected by the rule generation algorithm.

        Args:
            data (Path): Full path to the input CSV file.
            column_to_target (str): Exact name of the column in the CSV that contains the time series
        Returns:
            pd.DataFrame: DataFrame containing a single column called 'target', ready
            for the creation of lags.
    """
    df = pd.read_csv(data)
    df = df[pd.to_numeric(df[column_to_target], errors='coerce').notna()]
    df[column_to_target] = pd.to_numeric(df[column_to_target])
    df = df.reset_index(drop=True)
    df = df[[column_to_target]]
    df.columns = ['target']
    return df


def lags_create(window_size: int, data: Path, column_to_target: str) -> pd.DataFrame:
    """Function to create lags with a variable window size.

    Args:
        window_size (int): Number of lags to create.
        data (Path): Full path to the input CSV file.
        column_to_target (str): Exact name of the column in the CSV that contains the time series

    Returns:
        pd.DataFrame: DataFrame containing the target column and the created lags.
    """
    df = charge_data(data, column_to_target)
    for i in range(1, window_size+1):
        df[f'lag_{i}'] = df['target'].shift(i)
    df.dropna(inplace=True)
    return df.iloc[:, ::-1]


def create_fuzzy_variables(df: pd.DataFrame, num_fuzzy_sets: int, defuzzyfication_method: str) -> list:
    """auxiliary function for creating fuzzy variables

    Args:
        df (pd.DataFrame): _input dataframe_
        num_fuzzy_sets (int): _number of fuzzy sets to create for each variable_
        defuzzyfication_method (str): _method of defuzzification to use for the output variable_

    Returns:
        list: _list of fuzzy variables created_
    """
    min_value = df.min().min()
    max_value = df.max().max()
    variable_list = [
        ctrl.Antecedent(np.arange(min_value - 1, max_value + 1, 1), name)
        if name != 'target' else ctrl.Consequent(np.arange(min_value - 1, max_value + 1, 1), name)
        for name in df.columns
    ]
    for variable in variable_list:
        variable.automf(num_fuzzy_sets)
    for var in variable_list:
        if isinstance(var, ctrl.Consequent):
            var.defuzzify_method = defuzzyfication_method
    return variable_list


def operacao_intersecao_min(valores):
    """Calculates the minimum of membership values.

    Args:
        valores (array-like): Membership values of the antecedents.
    Returns:
        float: Minimum value of memberships.
    """
    return np.min(valores)


def operacao_intersecao_prod(valores):
    """Calculates the product of membership values.

    Args:
        valores (array-like): Membership values of the antecedents.
    Returns:
        float: Product of memberships.
    """
    return np.prod(valores)


def operacao_intersecao_avg(valores):
    """Calculates the average of membership values.

    Args:
        valores (array-like): Membership values of the antecedents.
    Returns:
        float: Average of memberships.
    """
    return np.mean(valores)


def implicacao_min(pertinencia_antecedente, valor_consequente):
    """Calculates fuzzy implication using the minimum operator.

    Args:
        pertinencia_antecedente (float): Antecedent membership value.
        valor_consequente (float): Consequent value.
    Returns:
        float: Result of implication by minimum.
    """
    return min(pertinencia_antecedente, valor_consequente)


def implicacao_prod(pertinencia_antecedente, valor_consequente):
    """Calculates fuzzy implication using the product operator.

    Args:
        pertinencia_antecedente (float): Antecedent membership value.
        valor_consequente (float): Consequent value.
    Returns:
        float: Result of implication by product.
    """
    return pertinencia_antecedente * valor_consequente


def implicacao_maxmin(pertinencia_antecedente, valor_consequente):
    """Calculates fuzzy implication using the truncated sum operator.

    Args:
        pertinencia_antecedente (float): Antecedent membership value.
        valor_consequente (float): Consequent value.
    Returns:
        float: Result of implication by truncated sum.
    """
    return max(0, pertinencia_antecedente + valor_consequente - 1)


def defuzzificacao_centroid(pertinencias):
    """Defuzzification using the centroid method.

    Args:
        pertinencias (dict): Dictionary mapping fuzzy sets to their membership values.
    Returns:
        float: Defuzzified value by centroid method.
    """
    return fuzz.defuzz(np.array(list(pertinencias.keys())), np.array(list(pertinencias.values())), 'centroid')


def defuzzificacao_mean_of_maxima(pertinencias):
    """Defuzzification using the mean of maxima method.

    Args:
        pertinencias (dict): Dictionary mapping fuzzy sets to their membership values.
    Returns:
        float: Defuzzified value by mean of maxima.
    """
    max_pertinencia = max(pertinencias.values())
    max_conjuntos = [conjunto for conjunto, pertinencia in pertinencias.items(
    ) if pertinencia == max_pertinencia]
    return np.mean(max_conjuntos)


def defuzzificacao_weighted_average(pertinencias):
    """Defuzzification using the weighted average method.

    Args:
        pertinencias (dict): Dictionary mapping fuzzy sets to their membership values.
    Returns:
        float: Defuzzified value by weighted average.
    """
    numerador = sum(conjunto * pertinencia for conjunto,
                    pertinencia in pertinencias.items())
    denominador = sum(pertinencias.values())
    if denominador == 0:
        return 0
    return numerador / denominador


def predict(fuzzy_sim, df, metodo_defuzzificacao):
    """Performs predictions using the fuzzy system.

    Args:
        fuzzy_sim (ctrl.ControlSystemSimulation): Fuzzy inference system.
        df (pd.DataFrame): DataFrame containing input data.
        metodo_defuzzificacao (function): Defuzzification function to be used.

    Returns:
        pd.DataFrame: DataFrame with predictions added.
    """
    df_copy = df.copy()
    df_copy.loc[:, 'predict'] = np.nan
    for i in range(len(df_copy)):
        l = df_copy.iloc[i, :].shape[0] - 2
        for k in range(l):
            fuzzy_sim.input[f'lag_{l - k}'] = df_copy.iloc[i, k]
        try:
            fuzzy_sim.compute()
            pertinencias = {term: fuzzy_sim.output[term]
                            for term in fuzzy_sim.output}
            out = metodo_defuzzificacao(pertinencias)
        except:
            out = df_copy.iloc[i, k]
        df_copy.iloc[i, l + 1] = out
    return df_copy


def avaliar_modelo(fuzzy_sim, df, metodo_defuzzificacao):
    """Evaluates the fuzzy model performance using MSE.

    Args:
        fuzzy_sim (ctrl.ControlSystemSimulation): Fuzzy inference system.
        df (pd.DataFrame): DataFrame containing input data.
        metodo_defuzzificacao (function): Defuzzification function to be used.

    Returns:
        tuple: Tuple containing MSE and DataFrame with predictions.
    """
    predict_df = predict(fuzzy_sim, df, metodo_defuzzificacao)
    mse = mean_squared_error(predict_df['target'], predict_df['predict'])
    rmse = root_mean_squared_error(predict_df['target'], predict_df['predict'])
    return mse, rmse, predict_df
