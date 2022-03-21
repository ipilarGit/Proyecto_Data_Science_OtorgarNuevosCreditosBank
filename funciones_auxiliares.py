import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.ticker as mtick
import seaborn as sns
import math
#import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, make_scorer
from sklearn.metrics import precision_score, recall_score, accuracy_score, cohen_kappa_score
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from scipy.stats import ks_2samp
from statistics import quantiles


def missing_values(df):
    """
    missing_values - return DataFrame with null values.

    @parameters:
        - df: pandas.core.frame.DataFrame.

    @returns:
        - DataFrame
    """
    total = df.isnull().sum().sort_values(ascending=False)
    porcentaje = (df.isnull().sum() / df.isnull().count()*100).sort_values(ascending = False)
    missing_df  = pd.concat([total, porcentaje], axis=1, keys=['Total Nulos', 'Porcentaje'])
    return missing_df
    
def grid_plot_batch(df, cols):
    # calcular un aproximado a la cantidad de filas
    rows = np.ceil(df.shape[1] / cols)

    # para cada columna
    plt.title('Distribución de atributo')
    for index, (colname, serie) in enumerate(df.iteritems()):
        plt.subplot(rows, cols, index + 1)
        sns.histplot(data = serie,  stat="count", kde_kws={"shade":True, 'linewidth':1}, alpha=1, bins=30)
        plt.title(colname, fontsize = 7, fontweight = "bold");
        plt.xlabel('');
    plt.tight_layout();
    plt.suptitle('Distribución variables numéricas', fontsize = 11, fontweight = "bold");
    plt.subplots_adjust(top =0.8, hspace=0.2, wspace=0.4)
    
def grafico_boxplot(datag):
    """
    grafico_boxplot - returns boxplot graphs of numeric variables. Adjust number of subplots based on column number.

    @parameters:
        - datag: pandas.core.frame.DataFrame.

    @returns:
        - boxplots graphs.
    """
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(11, 4))
    axes = axes.flat
    columnas_numeric = datag.select_dtypes(include=['float64', 'int']).columns
    tmp = datag[columnas_numeric]
    for i, colum in enumerate(columnas_numeric):
        sns.boxplot(
            data    = tmp[colum],
            orient  = 'v',
            color   = (list(plt.rcParams['axes.prop_cycle'])*2)[i]["color"],
            ax      = axes[i]
        )
        axes[i].set_title(colum, fontsize = 7, fontweight = "bold")
        axes[i].tick_params(labelsize = 6)
        axes[i].set_xlabel("")
    
    fig.tight_layout()
    plt.subplots_adjust(top =0.8, hspace=0.2, wspace=0.4)
    fig.suptitle('Boxplots de variables numéricas', fontsize = 10, fontweight = "bold");

def histograma(df, variable, etiqueta=''):
    """
    histograma - return histograma graph of numeric variable.

    @parameters:
        - datag: pandas.core.frame.DataFrame.
        - variable: categorical variable.
        - etiqueta: description variable.

    @returns:
        - histogram graph.
    """
    tmp = df[variable]
    plt.figure(figsize=(10, 2));

    plt.subplot(1,2,1)
    sns.histplot(abs(tmp), kde=True, bins=50, line_kws={'linewidth':1}, alpha=0.3);
    plt.title('Distribución original '+variable,  fontsize=10);
    plt.xlabel(etiqueta, fontsize = 8);

    flag_var=False
    if variable == 'DAYS_EMPLOYED' or variable == 'DAYS_REGISTRATION' or variable == 'DAYS_ID_PUBLISH' or variable== 'DAYS_LAST_PHONE_CHANGE':
        flag_var = True
    
    if flag_var:  
        plt.subplot(1,2,2)
        sns.distplot(np.log(tmp *-1 + 0.00001), hist=False, kde_kws={"shade":True, 'linewidth':1});
    else:
        plt.subplot(1,2,2)
        sns.distplot(np.log(abs(tmp)), hist=False, kde_kws={'shade':True, 'linewidth':1});
    
    plt.title('Transformación logarítmica '+variable, fontsize=10);  
    plt.xlabel(etiqueta, fontsize = 8);
    
def graficos_barra(datag):
    """
    graficos_barra - returns bar graphs of categorical variables. Adjust number of subplots based on column number

    @parameters:
        - datag: pandas.core.frame.DataFrame. 

    @returns:
        - bar graphics.
    """
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
    axes = axes.flat
    columnas_object = datag.select_dtypes(include=['object']).columns

    for i, colum in enumerate(columnas_object):
        datag[colum].value_counts().plot.barh(ax = axes[i])
        axes[i].set_title(colum, fontsize = 10, fontweight = "bold")
        axes[i].tick_params(labelsize = 10 ) #, rotation=r
        axes[i].set_xlabel("")

    fig.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.2)
    fig.suptitle('Distribución variables categóricas', fontsize = 11, fontweight = "bold");  


def countplot(df, variable, etiqueta='', flag=0):
    """
    countplot - returns countplot graph of categorical variable.

    @parameters:
        - df: pandas.core.frame.DataFrame.
        - variable: categorical variable.
        - etiqueta: description variable.
        - flag: porcentage distribution.

    @returns:
        - countplot graph.
    """
    tmp = df[variable]
    if flag == 1: 
        print(tmp.value_counts(normalize=True)*100)
    sns.countplot(y=tmp, order=tmp.value_counts().index);
    plt.title(etiqueta, fontsize=12);
    
def countplot2(df, variable, etiqueta=''):
    """
    countplot - returns countplot graph of categorical variable.

    @parameters:
        - df: pandas.core.frame.DataFrame.
        - variable: categorical variable.
        - etiqueta: description variable.

    @returns:
        - countplot graph.
    """
    tmp = df[variable]
    sns.countplot(x=tmp, order=tmp.value_counts().index);
    plt.title(etiqueta, fontsize=10);

def histplot_multiple(datag): # no
    """
    countplot - returns histplot graph of categorical variable. Adjust number of subplots based on column number

    @parameters:
        - datag: pandas.core.frame.DataFrame.

    @returns:
        - histplot graph.
    """
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    axes = axes.flat
    columnas_object = datag.select_dtypes(include=['object']).columns
    tmp = datag[columnas_object]
    for i, column in enumerate(columnas_object):
        sns.histplot(
                    binwidth=0.5,
                    x=tmp[column],
                    hue="TARGET", 
                    data=df_train, 
                    stat="count", 
                    multiple="stack",
                    ax      = axes[i]
        );
        axes[i].set_title(column, fontsize = 7, fontweight = "bold")
        axes[i].tick_params(labelsize = 6)
        axes[i].set_xlabel("")
    
    fig.tight_layout()
    plt.subplots_adjust(top =0.8, hspace=0.2, wspace=0.4)
    fig.suptitle('Distribución variables categoricas v/s Target', fontsize = 10, fontweight = "bold");
    
    
def variables_categoricas(df):
    columnas_categoricas = df.select_dtypes(include='object')
    for cat in columnas_categoricas:
        #print(cat)
        print(df[cat].value_counts())
        print("\n")

def preprocess_data(df):
    """
    Retorna dataframe con variables categoricas binarizadas
    Descarta dentro de las dummies la mayor categoria de cada variable
    """
    # temporal copy
    tmp_df = df
    # asegurar que var objetivo no se categorica
    tmp_df['TARGET'] = tmp_df['TARGET'].astype(int)
    
    # indentificar categoricas
    get_categoricals = df.dtypes[np.where(df.dtypes.values == 'object', True, False)]
    
    # Busca Categoria Mayor, guarda el nombre con el que se creará la dummie
    mayores_categorias = []
    for var_cat in get_categoricals.index:
        cat_value_mayor = tmp_df[var_cat].value_counts().index[0]
        categoria_mayor = var_cat+"_"+str(cat_value_mayor)
        mayores_categorias.append(categoria_mayor)    
    
    # Al hacer el get_dummies automaticamente se borran las columnas originales (no es necesario borrarlas)
    tmp_df = pd.get_dummies(tmp_df)  
    
    # Borrar las mayores categorias
    tmp_df = tmp_df.drop(columns = mayores_categorias)
    
    return tmp_df

def extrae_tipo(texto,subtexto=[":","Type"]):
    """
    Ajusta texto para generar texto2 modificado
    Criterios: Extrae lo que se encuentra antes subtexto (si hay varios itera por cada uno, en el orden especificado)
    """
    texto2 = texto
    for st in subtexto:
        n = texto2.count(st)
        if n > 0:
            indice = texto.index(st)
            texto2 = texto[0:indice]
            
    return(texto2.strip())


def resumen_cat_porc_malos(df, var ,var_objetivo = 'TARGET'):
    """
    Para una variable categorica, agrupa obteniendo la tasa de buenos y malos
    ordenando de mayor a menor tasa de malos.
    
    Entrega el n° de casos de cada categoria, el % de participacion y % de part. acumulado con respecto al total
    df: df con datos
    var = variable categorica 
    var_objetivo (debe tener valores 0 y 1, donde 1 indica que es malo, por defecto TARGET)
    """
    
    tmp = df.groupby(var).agg({var_objetivo:['count','sum']}).reset_index()
    tmp.columns = [var,'N_CASOS','N_MALOS']
    tmp['N_BUENOS'] = tmp['N_CASOS'] - tmp['N_MALOS']
    tmp['%_BUENOS'] = tmp['N_BUENOS']/tmp['N_CASOS']
    tmp['%_MALOS'] = tmp['N_MALOS']/tmp['N_CASOS']
    tmp = tmp.sort_values(by = '%_MALOS', ascending = False)
    tmp['N_ACUM'] = tmp['N_CASOS'].cumsum()
    tmp['%_Part'] = tmp['N_CASOS']/tmp['N_CASOS'].sum()
    tmp['%_Part_Acum'] = tmp['N_ACUM']/tmp['N_CASOS'].sum()
    tmp = tmp.drop(columns = ['N_MALOS','N_BUENOS','N_ACUM'])
    tmp = tmp.loc[:, [var,'N_CASOS','%_Part','%_Part_Acum','%_BUENOS','%_MALOS']]
    return tmp

def grafico_bivariado_cat(df, var_cat , var_obj, size=(8, 4), r=0):
    """
    Genera grafico bivariado para var_vat (variable categorica)
    Muestra Distribucion y Tasa de Malos por Categoria
    var_obj debe indicar 0 o 1 (1 mal pagador)
    """
    
    dftmp = df.groupby(var_cat).agg({var_obj:['mean','count']}).reset_index()
    dftmp.columns = [var_cat,'%_malos','n_casos']
    
    # grafico barras
    titulo =  'Distribución y % malos '+var_cat
    ax = plt.figure(figsize=size)
    ax = sns.barplot(x = var_cat, y = 'n_casos', data = dftmp, alpha = .8)
    ax2 = ax.twinx()
    ax2 = sns.lineplot(x=var_cat, y='%_malos', data=dftmp, color='navy', marker='o', linewidth=2, label='% Malos Pagadores', alpha=.8)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_xticklabels(dftmp[var_cat].values, rotation=r, ha='right')
    ax.set_xlabel('')
    ax.set_ylabel('Clientes')
    ax.set_title(titulo)
    ax2.set_ylabel('')
    
def barplot_multiple_porc(df, datag, r=0):
    """
    Genera grafico bivariado para var_vat (variable categorica)
    Muestra Distribucion y Tasa de Malos por Categoria
    var_obj debe indicar 0 o 1 (1 mal pagador)
    """
  
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4), sharey=True)
    axes = axes.flat
    columnas_object = datag.select_dtypes(include=['object']).columns
    tmp = datag[columnas_object]  
    
    for i, column in enumerate(columnas_object):
        dftmp = df.groupby(column).agg({"TARGET":['mean','count']}).reset_index()
        dftmp.columns = [ column ,'%_malos','n_casos']
        sns.barplot(x=column, y ='n_casos', data = dftmp, ax= axes[i],  alpha = .8);
        axes2 = axes[i].twinx()
        axes2 = sns.lineplot(x=column, y='%_malos', data=dftmp, color='navy', marker='o', linewidth=2, label='% Malos Pagadores', alpha=.8)
        axes2.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        dftmp[column] = dftmp[column].astype('object')
        axes[i].set_xticklabels(dftmp[column].values, rotation = r, ha='right')
        axes2.set_ylim(0, dftmp['%_malos'].max()*1.1)
        
        axes[i].set_title(column, fontsize = 8, fontweight = "bold")
        axes[i].tick_params(labelsize = 8)
        #axes[i].set_xlabel('')
        axes[i].set_ylabel('# clientes')
        axes2.set_ylabel('')
    
    fig.tight_layout()
    plt.subplots_adjust(top = 0.8, hspace=0.2, wspace=0.2)
    fig.suptitle('Distribución y % malos pagadores', fontsize = 12, fontweight = "bold");    
    
def grouped_boxplot(dataframe, variable, group_by):
    '''
    Definición: Genera grafico del tipo boxplot agrupando por una variable auxiliar.
    Parámetro de ingreso:   dataframe: Un dataframe.
                            variable: Variable a evaluar
                            group_by: Variable con la cual se segmentará el boxplot
    Retorno: Gráfico boxplot agrupado según categorías de la variable de agrupación.
    '''
    tmp = dataframe.dropna(subset=[variable])
    sns.boxplot(data=tmp, x=group_by, y = variable)
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
    titulo="Gráfico boxplot variable " + variable
    plt.title(titulo,fontdict=font)

    
def grouped_countplot(dataframe, variable, group_by):
    '''
    Definición: Genera grafico de barra con frecuencia de categorías de la variable a analizar, segmentando por variable de agrupación.
    Parámetro de ingreso:   dataframe: Un dataframe.
                            variable: Variable a evaluar
                            group_by: Variable con la cual se segmentará el gráfico
    Retorno: Gráfico de barra con frecuencia de categorías de 'variable' y agrupado según categorías de la variable de agrupación.
    '''
    tmp = dataframe.dropna(subset=[variable])
    sns.countplot(data=tmp, y = variable, hue=group_by)
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
    titulo="Gráfico frecuencias de variable " + variable
    plt.title(titulo,fontdict=font)
    
    
def plot_multiple_histogramas(df, columnas):
    
    num_plots = len(columnas)
    num_cols = math.ceil(np.sqrt(num_plots))
    num_rows = math.ceil(num_plots/num_cols)
        
    fig, axs = plt.subplots(num_rows, num_cols)
    
    for ind, col in enumerate(columnas):
        i = math.floor(ind/num_cols)
        j = ind - i * num_cols
        if num_rows == 1:
            if num_cols == 1:
                sns.distplot(df[col],kde_kws={"color": "k", "lw": 1}, ax=axs , kde= True, bins= 10)
            else:
                sns.distplot(df[col],kde_kws={"color": "g", "lw": 1}, ax=axs[j], kde= True, bins= 10)
        else:
            sns.distplot(df[col],kde_kws={"color": "b", "lw": 1}, ax=axs[i, j] , kde= True, bins= 10)

            
def plot_multiple_countplots(df, columnas):
    
    num_plots = len(columnas)
    num_cols = math.ceil(np.sqrt(num_plots))
    num_rows = math.ceil(num_plots/num_cols)
        
    fig, axs = plt.subplots(num_rows, num_cols)
    
    for ind, col in enumerate(columnas):
        i = math.floor(ind/num_cols)
        j = ind - i*num_cols
        
        if num_rows == 1:
            if num_cols == 1:
                sns.countplot(x=df[col], ax=axs, hue=df[col]);
            else:
                sns.countplot(x=df[col], ax=axs[j], hue=df[col]);
        else:
            sns.countplot(x=df[col], ax=axs[i, j], hue=df[col]);
            
    
def coefplot(model, varnames=True, intercept=False, fit_stats=True, figsize=(7, 12)):
    """
    coefplot - Visualize coefficient magnitude and approximate frequentist significance from a model.
    
    @parameters:
        - model: a `statsmodels.formula.api` class generated method, which must be already fitted.
        - varnames: if True, y axis will contain the name of all of the variables included in the model. Default: True
        - intercept: if True, coefplot will include the $\beta_{0}$ estimate. Default: False.
        - fit_stats: if True, coefplot will include goodness-of-fit statistics. Default: True.
        
    @returns:
        - A `matplotlib` object.
    """
    if intercept is True:
        coefs = model.params.values
        errors = model.bse
        if varnames is True:
            varnames = model.params.index
    else:
        coefs = model.params.values[1:]
        errors = model.bse[1:]
        if varnames is True:
            varnames = model.params.index[1:]
            
    tmp_coefs_df = pd.DataFrame({'varnames': varnames, 'coefs': coefs,'error_bars': errors})
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(y=tmp_coefs_df['varnames'], x=tmp_coefs_df['coefs'], 
                xerr=tmp_coefs_df['error_bars'], fmt='o', 
                color='slategray', label='Estimated point')
    ax.axvline(0, color='tomato', linestyle='--', label='Null Effect')
    ax.set_xlabel(r'$\hat{\beta}$')
    fig.tight_layout()
    plt.legend(loc='best')
    
    if fit_stats is True:
        if 'linear_model' in model.__module__.split('.'):
            plt.title(r'R$^{2}$' + "={0}, f-value={1}, n={2}".format(round(model.rsquared, 2),
                                                                     round(model.f_pvalue, 3),
                                                                     model.nobs))
        elif 'discrete_model' in model.__module__.split('.'):
            plt.title("Loglikelihood = {0}, p(ll-Rest)={1}, n={2}".format(round(model.llf, 2),
                                                                          round(model.llr_pvalue, 3),
                                                                          model.nobs))

            
def saturated_model(df, dependent, estimation=smf.ols, fit_model=True):
    """
    saturated_model - returns a saturated model

    @parameters:
        - df: pandas.core.frame.DataFrame.
        - dependent: String. Name of the variable that wants to be predicted.
        - estimation: Method. smf estimator
        - fit_model: Bool. If the model wants to be returned with fith done or not.

    @returns:
        - A `smf` model.

    """
    tmp_vars = "+".join(df.columns.drop(dependent))
    tmp_model = estimation(dependent+ '~ '+ tmp_vars, df)
    if fit_model is True:
        tmp_model = tmp_model.fit()
    
    return tmp_model


def identify_high_correlations(df, threshold=.7):
    """
    identify_high_correlations: Genera un reporte sobre las correlaciones existentes entre variables, condicional a un nivel arbitrario.

    Parámetros de ingreso:
        - df: un objeto pd.DataFrame, que es la base de datos a trabajar.
        - threshold: Nivel de correlaciones a considerar como altas. Por defecto es .7.

    Retorno:
        - Un pd.DataFrame con los nombres de las variables y sus correlaciones
    """
    # extraemos la matriz de correlación con una máscara booleana
    tmp = df.corr().mask(abs(df.corr()) <= .7, df)
    # convertimos a long format
    tmp = pd.melt(tmp)
    # agregamos una columna extra que nos facilitará los cruces entre variables
    tmp['var2'] = list(df.columns) * len(df.columns)
    # reordenamos
    tmp = tmp[['variable', 'var2', 'value']].dropna()
    # eliminamos valores duplicados
    tmp = tmp[tmp['value'].duplicated()]
    # eliminamos variables con valores de 1 
    return tmp[tmp['value'] < 1.00]


def plot_categoria(var,r):
    
    plt.figure(figsize = (20,5))
    
    target = df_train['TARGET'] 
    df_clase_1 = df_train[target == 1]
    df_clase_0 = df_train[target == 0]
    
    plt.subplot(1,2,1)
    sns.countplot(var, data = df_clase_0)
    plt.title('Distribución Buenos Pagadores ({})'.format(var), fontsize = 12)
    plt.xlabel(var, fontsize = 10)
    plt.xticks(rotation = r,fontsize = 10)
    plt.ylabel('# Buenos Pagadores', fontsize = 12)
    
    plt.subplot(1,2,2)
    sns.countplot(var, data = df_clase_1)
    plt.title('Distribución Malos Pagadores ({})'.format(var), fontsize = 12)
    plt.xlabel(var, fontsize = 10)
    plt.xticks(rotation = r,fontsize = 10)
    plt.ylabel('# Malos Pagadores', fontsize = 12)
    
    plt.show()

def resumen_vector_obj(X_train,y_train): # revisar 
    n = len(y_train)
    n_x = len(X_train)
    
    if str(type(X_train)) == "<class 'pandas.core.frame.DataFrame'>":
        n_var = X_train.shape[1]
    else:
        n_var = len(X_train[0])    
    
    print("n var:",n_var,"n reg X train: ",n_x,"n reg. Y train: ",n, "( % target = 1:", round(sum(y_train)/n*100,2), ")")
    
def tabla_comp_train(lista_metodo,lista_X_train,lista_y_train):
    """Recibe una lista de entrenamientos y objetivos (X_train,y_train) para armar resumen"""
    n_reg_lista = []
    n_target_1_lista = []

    for X_train in lista_X_train:
        n_reg = len(X_train)
        n_reg_lista.append(n_reg)

    for y_train in lista_y_train:
        y_train = list(y_train)
        n_target_1 = sum(y_train)
        n_target_1_lista.append(n_target_1)

    tabla = pd.DataFrame({'Base':lista_metodo, 'n_registros': n_reg_lista, 'n_target_1': n_target_1_lista})
    tabla['% target 1'] = round(tabla['n_target_1']/tabla['n_registros']*100,2)
    tabla = tabla.set_index('Base')
    return tabla

def feature_importance_reg_log(modelo,df,var_obj = 'TARGET'):
    importance = modelo.coef_[0]
    features = list(df.drop(columns = var_obj).columns)
    df_feat_import = pd.DataFrame({'Variable':features, 'Importancia': importance})
    df_feat_import['Importancia_abs'] = abs(df_feat_import['Importancia'])
    df_feat_import = df_feat_import.sort_values(by = 'Importancia_abs', ascending = False)
    return df_feat_import        
    
def feature_importance(modelo,df,var_obj = 'TARGET'):
    # Genera df con importancia de las carateristicas para un modelo que pueda ejcutar .predict()
    importance = modelo.feature_importances_
    features = list(df.drop(columns = var_obj).columns)
    df_feat_import = pd.DataFrame({'Variable':features, 'Importancia': importance})
    df_feat_import = df_feat_import.sort_values(by = 'Importancia', ascending = False)
    return df_feat_import  
    
def mostrar_resultados(y_test, y_pred):
    plt.figure(figsize=(4, 4))
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)    
    f1_ = f1_score(y_test, y_pred)
    print(f"Validation Accuracy:{accuracy}  Precision:{precision}  Recall:{recall}  F1 score:{f1_}") 
    conf_matrix = confusion_matrix(y_test, y_pred)
    LABELS = ['Negativa','Positiva']
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d"); 
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    print(classification_report(y_test, y_pred))
    #print("AUC:", roc_auc_score(y_test, y_probs))


def resumen_modelo(modelo,X_test,y_test,gam=False):
    y_predict = modelo.predict(X_test)
    
    if gam:
        y_predict_prob = modelo.predict_proba(X_test)
    else:
        y_predict_prob = modelo.predict_proba(X_test)[:,1]
    
    # indicadores
    auc = roc_auc_score(y_test, y_predict_prob)
    ks = ks_2samp(y_predict_prob[y_test==1],y_predict_prob[y_test==0]).statistic
    gini = (auc-0.5)*2
    
    print("Reporte Clasificación:")
    print(classification_report(y_test, y_predict))
    
    print("Indicadores:") 
    
    print('auc : ',round(auc*100,3),"%")
    print('ks  : ',round(ks*100,3),"%")
    print('gini: ',round(gini*100,3),"%")

def resumen_modelo_df(nombre_modelo,modelo,X_test,y_test):
    """Se genera un df con un resumen de los indicadores para un modelo específico"""
    
    # listado indicadores
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    auc_lis = []
    ks_list = []
    gini_list = []
    y_predict = modelo.predict(X_test)

    if nombre_modelo == 'GAM':
        y_predict_prob = modelo.predict_proba(X_test)
    else:
        y_predict_prob =  modelo.predict_proba(X_test)[:,1]
    
    accuracy = round(accuracy_score(y_test, y_predict),2)
    precision = round(precision_score(y_test,y_predict),2)
    recall = round(recall_score(y_test, y_predict),2)
    f1 = round(f1_score(y_test, y_predict),2)
    auc = round(roc_auc_score(y_test, y_predict_prob),4)
    ks = round(ks_2samp(y_predict_prob[y_test==1],y_predict_prob[y_test==0]).statistic,4)
    gini = round((roc_auc_score(y_test, y_predict_prob)-0.5)*2,4)
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_score_list.append(f1)
    auc_lis.append(auc)
    ks_list.append(ks)
    gini_list.append(gini)
    
    dic = { 'Modelo':nombre_modelo,
            'accuracy':accuracy_list,
            'precision':precision_list,
            'recall':recall_list,
            'f1-score':f1_score_list,
            'AUC':auc_lis,
            'KS': ks_list,
            'GINI': gini_list}

    df_kpi = pd.DataFrame(dic)
    return df_kpi
         
def resumen_compara_modelos_df(lista_nombres,lista_modelos,X_test_std,y_test):
    """A partir de un listado de modelos y sus nombres, se genera un resumen de los principales indicadores de los modelos
    Se llama a la función resumen_modelo_df()"""
    for index,nombre in enumerate(lista_nombres):
        if index == 0:
            df_kpi = resumen_modelo_df(nombre,lista_modelos[index],X_test_std,y_test)
        else:
            df_kpi_aux = resumen_modelo_df(nombre,lista_modelos[index],X_test_std,y_test)
            df_kpi = pd.concat([df_kpi,df_kpi_aux],axis = 0)
    
    df_kpi = df_kpi.set_index('Modelo')            
    return df_kpi


### APLICACIÓN MODELO BASE VALIDACIÓM #####################################
def procesa_data_validacion(df_in):
    """
    Procesa dataframe para que quede con la estructura de los datos del modelo.
    Realiza reemplazo de nulos, categorizaciones, creación de variables y binarizaciones
    """
    df = df_in.copy()
    df = df[df['NAME_CONTRACT_TYPE'] == 'Cash loans']
    # lista modelos incluir
    variables_modelo = ['CNT_CHILDREN','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_PHONE','CNT_FAM_MEMBERS',
                        'REGION_RATING_CLIENT','REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY',
                        'EXT_SOURCE_2','EXT_SOURCE_3','DEF_30_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE',
                        'AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_YEAR','DAYS_BIRTH','AMT_INCOME_TOTAL','AMT_CREDIT',
                        'DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE','CODE_GENDER',
                        'FLAG_OWN_CAR','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE',
                        'OCCUPATION_TYPE','ORGANIZATION_TYPE']
    
    variables_modelo_final = ['CNT_CHILDREN','FLAG_WORK_PHONE','FLAG_PHONE','CNT_FAM_MEMBERS',
                              'REGION_RATING_CLIENT','REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY',
                              'LIVE_CITY_NOT_WORK_CITY',
                              'EXT_SOURCE_2','EXT_SOURCE_3','DEF_30_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE',
                              'AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_YEAR',
                              'AGE','LOG_AMT_INCOME_TOTAL','LOG_AMT_CREDIT','WITH_DAYS_WORKED','LOG_DAYS_REGISTRATION',
                              'LOG_DAYS_ID_PUBLISH','PHONE_CHANGE','CODE_GENDER_M','FLAG_OWN_CAR_Y',
                              'NAME_INCOME_TYPE_commercial','NAME_INCOME_TYPE_government','NAME_INCOME_TYPE_others',
                              'NAME_EDUCATION_TYPE_school','NAME_EDUCATION_TYPE_university',
                              'NAME_FAMILY_STATUS_not_married','NAME_HOUSING_TYPE_others',
                              'NAME_HOUSING_TYPE_with_parents_or_rented','OCCUPATION_TYPE_OCCUPATION_TYPE_G1',
                              'OCCUPATION_TYPE_OCCUPATION_TYPE_G2','OCCUPATION_TYPE_OCCUPATION_TYPE_G3',
                              'OCCUPATION_TYPE_OCCUPATION_TYPE_G4','OCCUPATION_TYPE_OCCUPATION_TYPE_G5',
                              'ORGANIZATION_TYPE_ORGANIZATION_TYPE_G1','ORGANIZATION_TYPE_ORGANIZATION_TYPE_G2',
                              'ORGANIZATION_TYPE_ORGANIZATION_TYPE_G4','ORGANIZATION_TYPE_ORGANIZATION_TYPE_G5',
                              'ORGANIZATION_TYPE_ORGANIZATION_TYPE_G6','ORGANIZATION_TYPE_ORGANIZATION_TYPE_G7',
                              'ORGANIZATION_TYPE_ORGANIZATION_TYPE_G8']
    
    df = df.loc[:,variables_modelo]    
        
    ## ------ REEMPLAZO NULOS -----------------###################################################################
    
    # REEMPLAZOS POR LA MEDIANA    
    df['EXT_SOURCE_2'] = df['EXT_SOURCE_2'].replace(np.nan, 0.5653811867548313 )
    df['EXT_SOURCE_3'] = df['EXT_SOURCE_3'].replace(np.nan, 0.5370699579791587)
    
    ## REEMPLAZOS CON 0
    df['AMT_REQ_CREDIT_BUREAU_YEAR'] = df['AMT_REQ_CREDIT_BUREAU_YEAR'].replace(np.nan,0)
    df['AMT_REQ_CREDIT_BUREAU_MON'] = df['AMT_REQ_CREDIT_BUREAU_MON'].replace(np.nan,0)
    df['DEF_30_CNT_SOCIAL_CIRCLE'] = df['DEF_30_CNT_SOCIAL_CIRCLE'].replace(np.nan,0)
    df['DEF_60_CNT_SOCIAL_CIRCLE'] = df['DEF_60_CNT_SOCIAL_CIRCLE'].replace(np.nan,0)
    df['DAYS_LAST_PHONE_CHANGE'] = df['DAYS_LAST_PHONE_CHANGE'].replace(np.nan,0)
    
    ## REEMPLAZOS CATEGORICAS
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].replace(np.nan,'Unknown')
    
    
    ## ---RE CATEGORIZACIONES -----------------###################################################################
    
    
    #--- CATEGORIZACION OCCUPATION TYPE -------------------------------------------------------------------------------------------
    # Grupos
    lista_g1 = ['Core staff','High skill tech staff','HR staff','IT staff','Managers','Medicine staff',
                'Private service staff','Realty agents','Secretaries']
    lista_g2 = ['Cleaning staff','Sales staff']
    lista_g3 = ['Cooking staff','Laborers','Security staff']
    lista_g4 = ['Drivers','Low-skill Laborers','Waiters/barmen staff']
    lista_g5 = ['Accountants']
    # Categorizaciones
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].replace(lista_g1,'OCCUPATION_TYPE_G1')
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].replace(lista_g2,'OCCUPATION_TYPE_G2')
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].replace(lista_g3,'OCCUPATION_TYPE_G3')
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].replace(lista_g4,'OCCUPATION_TYPE_G4')
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].replace(lista_g5,'OCCUPATION_TYPE_G5')
    
    ## --- CATEGORIZACION NAME_INCOME_TYPE ---- # 
    # minusculas NAME_INCOME_TYPE, se categoriza en var aux NAME_INCOME_TYPE_2
    df['NAME_INCOME_TYPE'] = df['NAME_INCOME_TYPE'].str.lower()
    df['NAME_INCOME_TYPE2'] = 'others' # si aparece una categoria nueva, dejarla como 'others'
    df['NAME_INCOME_TYPE2'] = np.where(df['NAME_INCOME_TYPE'] == 'commercial associate', 'commercial', df['NAME_INCOME_TYPE2'] )
    df['NAME_INCOME_TYPE2'] = np.where(df['NAME_INCOME_TYPE'] == 'working', 'working', df['NAME_INCOME_TYPE2'])
    df['NAME_INCOME_TYPE2'] = np.where(df['NAME_INCOME_TYPE'] == 'state servant', 'government', df['NAME_INCOME_TYPE2'])
    # reemplazar y descartar variable auxiliar 
    df['NAME_INCOME_TYPE'] = df['NAME_INCOME_TYPE2']
    df = df.drop(columns = 'NAME_INCOME_TYPE2')  
    
    ## --- CATEGORIZACION NAME_HOUSING_TYPE ---- # 
    # minusculas, se categoriza en var auxiliar 'NAME_HOUSING_TYPE2'
    df['NAME_HOUSING_TYPE'] = df['NAME_HOUSING_TYPE'].str.lower()
    df['NAME_HOUSING_TYPE2'] = 'others'  # si aparece una categoria nueva, dejarla como 'others'
    df['NAME_HOUSING_TYPE2'] = np.where(df['NAME_HOUSING_TYPE'].isin(['with parents','rented apartment']),'with_parents_or_rented',df['NAME_HOUSING_TYPE2'])
    df['NAME_HOUSING_TYPE2'] = np.where(df['NAME_HOUSING_TYPE'] == 'house / apartment', 'house_apartment', df['NAME_HOUSING_TYPE2'])
    # reemplazar y descartar variable auxiliar 
    df['NAME_HOUSING_TYPE'] = df['NAME_HOUSING_TYPE2']
    df = df.drop(columns = 'NAME_HOUSING_TYPE2')  
    
    ## --- CATEGORIZACION NAME_EDUCATION_TYPE ---- # 
    # minusculas
    df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].str.lower()
    df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].replace(['secondary / secondary special','incomplete higher'], 'secondary')
    df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].replace(['higher education','academic degree'], 'university')
    df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].replace('lower secondary', 'school')
    
    ## --- CATEGORIZACION NAME_FAMILY_STATUS ---- # 
    df['NAME_FAMILY_STATUS'] = df['NAME_FAMILY_STATUS'].str.lower()
    df['NAME_FAMILY_STATUS'] = np.where(df['NAME_FAMILY_STATUS'].isin(['married','civil marriage']),'married','not_married')
    
    ## --- CATEGORIZACION ORGANIZATION_TYPE ---- 
    # Ajustes formato texto 
    df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].apply(lambda x: extrae_tipo(x))
    # Se agrupan las 35 categorias segun niveles de riesgo en 8 niveles
    lista_g1 = ["Construction","Realtor","Restaurant"]
    lista_g2 = ["Agriculture","Cleaning","Security","Self-employed","Transport"]
    lista_g3 = ["Business Entity","Trade"]
    lista_g4 = ["Advertising","Industry","Mobile","Postal"]
    lista_g5 = ["Housing","Legal Services","Other"]
    lista_g6 = ["Electricity","Emergency","Government","Kindergarten","Medicine","Services","Telecom"]
    lista_g7 = ["Hotel","Insurance","Religion","School"]
    lista_g8 = ["Bank","Culture","Military","Police","Security Ministries","University","XNA"]

    df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].replace(lista_g1,'ORGANIZATION_TYPE_G1')
    df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].replace(lista_g2,'ORGANIZATION_TYPE_G2')
    df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].replace(lista_g3,'ORGANIZATION_TYPE_G3')
    df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].replace(lista_g4,'ORGANIZATION_TYPE_G4')
    df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].replace(lista_g5,'ORGANIZATION_TYPE_G5')
    df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].replace(lista_g6,'ORGANIZATION_TYPE_G6')
    df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].replace(lista_g7,'ORGANIZATION_TYPE_G7')
    df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].replace(lista_g8,'ORGANIZATION_TYPE_G8')
    
       
    # ----- CREACION/TRANSFORMACIÓN VARIABLES  ----------#########################################################
    df['AGE'] = (df['DAYS_BIRTH'] * -1 ) /365
    df['LOG_AMT_INCOME_TOTAL'] = np.log(df['AMT_INCOME_TOTAL'])
    df['LOG_AMT_CREDIT'] = np.log(df['AMT_CREDIT'])
    df['LOG_DAYS_REGISTRATION'] = np.log(df['DAYS_REGISTRATION']*-1 + 0.00001)
    df['WITH_DAYS_WORKED'] = np.where(df['DAYS_EMPLOYED'] < 0, 1 , 0)
    df['LOG_DAYS_ID_PUBLISH'] = np.log(df['DAYS_ID_PUBLISH']*-1 + 0.00001)
    df['PHONE_CHANGE'] = np.where(df['DAYS_LAST_PHONE_CHANGE'] < 0, 1, 0)
        
      
    # Eliminar las originales de las que fueron transformadas
    var_transf_eliminar = ['DAYS_BIRTH','AMT_INCOME_TOTAL','AMT_CREDIT','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE']
    df = df.drop(columns = var_transf_eliminar)
    
    ## --- BINARIZAR  ----------  #########################################################
    df = pd.get_dummies(df)
    
    # Seleccionar las variables del modelo
    df = df.loc[:,variables_modelo_final]
    
    return df