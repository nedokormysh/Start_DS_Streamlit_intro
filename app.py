import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'


@st.cache_data
def open_data(path: str = TRAIN_PATH):
    """
    Функция загружает датасет
    :param path: path to datasets
    :type path: str
    :return: tuple[list[pandas.DataFame], dict[str, pandas.DataFrame], list[str]]
    """

    print('Датасет загружен.')

    df = pd.read_csv(path, index_col=0)
    df.columns = df.columns.str.replace(' ', '_')

    return df


def side_bar_params():
    """
    Функция формирования сайдбаров настройки размеров отображения графиков
    :return: кортеж с размерами
    """

    st.sidebar.markdown("*Настройки отображения матрицы корреляции*")

    width_corr = st.sidebar.slider("ширина таблицы корреляции", 3, 6, 3)
    height_corr = st.sidebar.slider("высота таблицы корреляции", 3, 6, 3)

    st.sidebar.markdown("---")

    st.sidebar.markdown("*Настройки отображения графиков*")

    check_verbose = st.sidebar.radio(
        "Сокрытие графиков",
        ["Режим отображения", "Скрыть"],
        captions=["Отображение графиков", "Скрытие графиков"])

    if check_verbose == 'Режим отображения':
        # st.write('Отображаем.')
        verbose = True
    else:
        # st.write("Скрываем.")
        verbose = False

    return height_corr, width_corr, verbose


# берём значения сайдбаров, которые будем использовать при отображении графиков
height_corr, width_corr, verbose = side_bar_params()


def corr_matrix(all_data, width=width_corr, height=height_corr):
    """
    Функция отображения матрицы корреляции
    :param all_data: датафрейм
    :param width: ширина
    :param height: высота
    :return: график
    """

    fig = plt.figure(figsize=(width, height))
    sns.set_style("whitegrid")

    mask = np.triu(np.ones_like(all_data.corr(
        numeric_only=True
    ), dtype=bool))

    heatmap = sns.heatmap(all_data.corr(
        numeric_only=True
    ).round(2),
                          annot=True,
                          square=True,
                          cmap="BrBG",
                          cbar_kws={"fraction": 0.01},
                          linewidth=2,
                          mask=mask,
                          )

    heatmap.set_title("Треугольная тепловая карта корреляции Пирсона", fontdict={"fontsize": 11}, pad=5)

    return fig


# Загрузка данных
with st.spinner('Пожалуйста, подождите. Идёт загрузка данных'):
    train = open_data(TRAIN_PATH)
st.success('Датасет загружен')

st.title('Датасет Abalone.')

st.markdown("**Сэмпл обучающего датасета**")

st.table(train.sample(3))

# определим категориальные и числовые признаки
cat_features = ['Sex']
targets = ['Rings']

num_features = [i for i in train.columns if i not in cat_features]

st.markdown("___")

st.markdown("# Краткая информация об исследуемом датасете")

st.text(('Есть пропуски!') if train.isna().any().any() else ('Пропусков нет'))
st.text(('Есть дубли!') if train.duplicated().any() else ('Дубликатов нет'))
st.text(f'Размер датасета {train.shape}')

st.markdown("### Описание числовых характеристик")
st.table(train[num_features].describe())

st.markdown("### Описание категориальных характеристик")
st.table(train.describe(include='object'))

if st.button(':rainbow[Раскрыть информацию о тестовом датасете]'):
    st.balloons()
    st.markdown("# Краткая информация об тестовом датасете")
    test = open_data(TEST_PATH)
    st.markdown("**Сэмпл тестового датасета**")
    st.table(test.sample(3))

    st.text(('Есть пропуски!') if test.isna().any().any() else ('Пропусков нет'))
    st.text(('Есть дубли!') if test.duplicated().any() else ('Дубликатов нет'))
    st.text(f'Размер тестового датасета {test.shape}')

    num_features = [i for i in test.columns if i not in cat_features]

    st.markdown("### Описание числовых характеристик тестового датасета")
    st.table(test[num_features].describe())

    st.markdown("### Описание категориальных характеристик тестового датасета")
    st.table(test.describe(include='object'))

st.markdown("___")
st.pyplot(corr_matrix(train), use_container_width=False)
st.info('По ширине графики ограничены: масштабируются под ширину контейнера')
st.markdown("___")


def lmplot_graph(df, feat: str, aspect=1.5, height=3.5, hue: bool = True, verbose: bool = True):
    """
    Функция отрисовывает график регрессии
    :param df: датасет
    :param feat: признак, который хотим рассмотреть
    :param aspect: соотношение сторон осей графика
    :param height: высота
    :param hue: разделения данных на подгруппы
    :param verbose: управление вывода графика
    :return: график
    """
    fig = plt.figure(figsize=(10, 10))

    if hue:
        di = {'data': df,
              'x': feat,
              'y': 'Rings',
              'hue': 'Sex',
              'scatter_kws': {'s': 6, 'alpha': 0.8},
              'line_kws': {"lw": 1, 'linestyle': '--'},
              'height': height,
              'aspect': aspect,
              'palette': 'Set1'}
    else:
        di = {'data': df,
              'x': feat,
              'y': 'Rings',
              'scatter_kws': {'s': 6, 'alpha': 0.8},
              'line_kws': {"lw": 1, 'linestyle': '--', 'color': 'r'},
              'height': height,
              'aspect': aspect}


    st.write(
            f'mean = {df[feat].describe()[1]}, min = {df[feat].describe()[3]}, max = {df[feat].describe()[7]}')

    fig = sns.lmplot(**di)

    return fig


def violin_graph(df, feat: str, verbose: bool = True):
    """
    Функция отрисовывает violin
    :param df: датасет
    :param feat: признак, который изучаем
    :param verbose: управление вывода графика
    :return: график
    """
    fig = plt.figure(figsize=(2, 2))

    if verbose:
        fig = sns.catplot(data=df,
                          y=feat,
                          height=4,
                          aspect=1,
                          kind="violin",).set_xticklabels(rotation=45, horizontalalignment="right")

    return fig

# раздел для формирования графики
st.markdown("# Графики")
st.markdown("### Выберем график требуемой числовой переменной")
cont_feat = st.selectbox('Выберете числовую переменную',
                        ('Length',
                         'Diameter',
                         'Height',
                         'Whole_weight',
                         'Whole_weight.1',
                         'Whole_weight.2',
                         'Shell_weight'))


def check_hue():
    """
    Фунцкия проверяет режим отображения графиков. Если есть, то отображаем radiobutton
    """

    check_hue = st.checkbox(':rainbow[Разделить данные на подгруппы в зависимости от пола]')

    if check_hue:
        hue = True
    else:
        hue = False

    return hue

if verbose:
    hue = check_hue()

    fig1 = lmplot_graph(train, cont_feat, hue=hue, verbose=verbose)
    st.pyplot(fig1, use_container_width=True)

    fig2 = violin_graph(train, cont_feat, verbose=verbose)
    st.pyplot(fig2, use_container_width=True)
else:
    hue = False
    st.table(train[cont_feat].describe())
    st.write(f'Коэффициент корреляции c целевой переменной = ', np.corrcoef(train[cont_feat], train['Rings'])[0][1])



def bar_graph(df: pd.DataFrame):
    """
    Функция отрисовывает столбчатый график
    :param df: датафрейм
    :return: график
    """
    fig = plt.figure(figsize=(10, 6))

    sns.barplot(x='Sex', y='Rings', data=df, palette='summer')
    plt.title('Sex - Rings')

    return fig


st.markdown("### Категориальная переменная")

if verbose:
    fig3 = bar_graph(train)
    st.pyplot(fig3, use_container_width=True)
else:
    st.table(train['Sex'].describe())

st.table(train.groupby('Sex').agg(Rings_mean=('Rings', 'mean')))
