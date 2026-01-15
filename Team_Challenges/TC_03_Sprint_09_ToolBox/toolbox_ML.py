#toolbox_ML.py

import pandas as pd

def describe_df(df:pd.DataFrame):
    """
    Esta función recibe como argumento un df y lo que hace es que crea una tabla
    con cada variable en una columna y como fila los tipos de las variables,
    y muestra en la tabla el % de valores nulos o missings, los valores unicos y
    porcentaje de catrdinalidad.

    Argumentos:
    :param df: un dataframe
    

    Retorno:
    Esta función retorna una tabla con el porcentaje de nulos, unicos y porcentaje de cardinalidad para todas las variables del dataframe original, 
    es como un describe pero específico para el machine learning.
    
    """
    #Cuerpo de la función
    resumen={}
    for col in df.columns:
        resumen[col]={
        "DATA_TYPE": df[col].dtype,
        "MISSINGS(%)": df[col].isnull().mean()*100,
        "UNIQUE_VALUES":df[col].nunique(),
        "CARDIN(%)":round((df[col].nunique()/len(df))*100,2)
        }
    
    return pd.DataFrame(resumen)

import pandas as pd

def tipifica_variables(df: pd.DataFrame, umbral_categoria: int, umbral_continua: float):
    """
    Clasifica las variables de un DataFrame según su cardinalidad y porcentaje de cardinalidad.

    :param df: DataFrame de entrada
    :type df: pd.DataFrame
    :param umbral_categoria: límite de cardinalidad para decidir si es categórica o numérica
    :type umbral_categoria: int
    :param umbral_continua: porcentaje mínimo para considerar una variable como numérica continua
    :type umbral_continua: float
    :return: DataFrame con columnas 'nombre_variable' y 'tipo_sugerido'
    :rtype: pd.DataFrame
    """
    resultados = []
    for col in df.columns:
        cardinalidad = df[col].nunique()
        porcentaje_card = (cardinalidad / len(df)) * 100

        if cardinalidad == 2:
            tipo = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo = "Categórica"
        else:
            if porcentaje_card >= umbral_continua:
                tipo = "Numérica Continua"
            else:
                tipo = "Numérica Discreta"

        resultados.append({
            "nombre_variable": col,
            "tipo_sugerido": tipo
        })

    return pd.DataFrame(resultados)


import pandas as pd
from scipy.stats import pearsonr

def selecciona_variables_corr(df: pd.DataFrame, target_col: str, umbral_corr: float, pvalue: float = None):
    # --- Comprobaciones ---
    if not isinstance(df, pd.DataFrame):
        print("El argumento df debe ser un DataFrame de pandas.")
        return None
    
    if target_col not in df.columns:
        print(f"La columna {target_col} no existe en el DataFrame.")
        return None
    
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"La columna {target_col} debe ser numérica.")
        return None
    
    if not (0 < umbral_corr < 1):
        print("umbral_corr debe ser un float entre 0 y 1.")
        return None
    
    if pvalue is not None and not (0 < pvalue < 1):
        print("pvalue debe ser None o un float entre 0 y 1.")
        return None
    
    # --- Selección de columnas numéricas ---
    num_cols = df.select_dtypes(include=['number']).columns.drop(target_col)
    seleccionadas = []
    
    for col in num_cols:
        corr = df[target_col].corr(df[col])
        if abs(corr) > umbral_corr:
            if pvalue is not None:
                # Test de hipótesis
                _, pval = pearsonr(df[target_col], df[col])
                if pval <= pvalue:
                    seleccionadas.append(col)
            else:
                seleccionadas.append(col)
    
    return seleccionadas