import streamlit as st
import pandas as pd
from ds_undersampling import Undersampling

#st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)


sbA = st.sidebar.selectbox('Algorithmus:',['Decision Tree', 'Random Forest', 'K-nearest Neighbors'])
sbD = st.sidebar.selectbox('Data set:', ['Adult', 'Bank', 'Yeast1'])
sbRed = st.sidebar.radio('Reduced Set:', ['Yes', 'No'])
sbK = st.sidebar.number_input('k:', value=10)
sbParam = st.sidebar.number_input('Classifier parameter:', value=20)
sbKmM = st.sidebar.selectbox('k-means++ Method:', ['None', 'Centroids', 'Random Sampling', 'Top1', 'TopN', 'All'])

algos = ['dtree', 'rfc', 'knn']
data_sets = ['adult.data', 'bank.csv', 'yeast1.dat']
kmeans_methods = ['No undersampling', 'centroids', 'random_sampling', 'top1', 'topN']

red_sel = 0
algo_sel = 0
ds_sel = 0
kmM_sel = 0
all_barchar = 0

if sbA == 'Decision Tree':
    algo_sel = 0
elif sbA == 'Random Forest':
    algo_sel = 1
elif sbA == 'K-nearest Neighbors':
    algo_sel = 2

if sbD == 'Adult':
    ds_sel = 0
elif sbD == 'Bank':
    ds_sel = 1
elif sbD == 'Yeast1':
    ds_sel = 2

if sbRed == 'Yes':
    red_sel = 1
elif sbRed == 'No':
    red_sel = 0

if sbKmM == 'None':
    kmM_sel = 0
elif sbKmM == 'Centroids':
    kmM_sel = 1
elif sbKmM == 'Random Sampling':
    kmM_sel = 2
elif sbKmM == 'Top1':
    kmM_sel = 3
elif sbKmM == 'TopN':
    kmM_sel = 4
elif sbKmM == 'All':
    kmM_sel = 4
    all_barchar = 1


if st.sidebar.button('Calculate'):
    df = Undersampling(data_sets[ds_sel], algos[algo_sel], kmeans_methods[kmM_sel], sbK, red_sel, sbParam)
    st.write("A sample of the data set:")
    st.write(df.df_original.head())
    st.write("Minority-to-majority class ratio: ", round(len(df.maxClass.index)/len(df.minClass.index),2))
    st.markdown("---")
    st.write("PCAs of data set before and after clustering:")
    col1, col2 = st.beta_columns(2)
    col1.pyplot(df.performPCA(1))
    if sbKmM != 'None':
        col2.pyplot(df.performPCA(0))
    st.markdown("---")
    if all_barchar:
        st.write("Comparison of different undersampling methods:")
        result = df.allKmMethods()
        stat_ind = ['Accuracy', 'Precision', 'Recall', 'f-Measure', 'AUC']
        st.image(result[0])
        df_stats = pd.DataFrame(index=['No undersampling', 'centroids', 'random_sampling', 'top1', 'topN'])
        for i in range(5):
            df_stats[stat_ind[i]] = result[i+1]
        st.write("")
        st.write(df_stats)
    else:
        result = df.printBarChart()
        st.write("Scores:")
        st.image(result[0])