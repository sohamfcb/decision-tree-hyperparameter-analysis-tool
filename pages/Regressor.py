import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.metrics import r2_score
from sklearn.datasets import make_regression

st.set_page_config(page_title='Regression')

st.title('Decision Tree Regressor Hyperparameter Analysis')

plt.style.use('seaborn-v0_8-darkgrid')

def showData():

    fig=plt.figure(figsize=(16,7))

    sns.scatterplot(x=np.squeeze(X),y=y,s=100,c='#01B8AA',edgecolor='black')
    plt.xlabel('X',{'family': 'serif', 'weight': 'normal', 'size': 20})
    plt.ylabel('Y',{'family': 'serif', 'weight': 'normal', 'size': 20})

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    st.pyplot(fig)
    plt.close(fig)

def drawRegressionLine():

    dtree=DecisionTreeRegressor(
        max_depth=None if max_depth is None else int(max_depth),
        criterion=criterion,
        splitter=splitter,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_leaf_nodes=None if max_leaf_nodes is None else int(max_leaf_nodes),
        min_impurity_decrease=min_impurity_decrease
    )

    dtree.fit(X_train,y_train)

    y_pred=dtree.predict(X_test)

    st.subheader(f'Regression Line on the Training Set (R2 score: {round(r2_score(dtree.predict(X_train),y_train),2)})')
    
    fig1=plt.figure(figsize=(16,7))
    sns.scatterplot(x=np.squeeze(X_train),y=y_train,edgecolor='black',s=100,c='#01B8AA')
    sns.lineplot(x=np.squeeze(X_train),y=dtree.predict(X_train),color='black',linewidth=3)
    plt.xlabel('X',fontdict={'family': 'serif', 'weight': 'normal', 'size': 20})
    plt.ylabel('Y',fontdict={'family': 'serif', 'weight': 'normal', 'size': 20})
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    st.pyplot(fig=fig1)
    plt.close(fig1)

    st.subheader(f'Regression Line on the Test Set (R2 score: {round(r2_score(y_pred,y_test),2)})')
    fig2=plt.figure(figsize=(16,7))
    sns.scatterplot(x=np.squeeze(X_test),y=y_test,edgecolor='black',s=100,c='#01B8AA')
    sns.lineplot(x=np.squeeze(X_test),y=y_pred,color='black',linewidth=3)
    plt.xlabel('X',fontdict={'family': 'serif', 'weight': 'normal', 'size': 20})
    plt.ylabel('Y',fontdict={'family': 'serif', 'weight': 'normal', 'size': 20})
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    st.pyplot(fig=fig2)
    plt.close(fig2)

    st.subheader('Trained Tree Architecture')

    fig3=plt.figure(figsize=(16,7))
    plot_tree(decision_tree=dtree,
              filled=True)
    st.pyplot(fig3)
    plt.close(fig3)


X,y=make_regression(n_samples=500,n_features=1,n_targets=1,n_informative=1,noise=10,random_state=10)
y = y ** 2 
noise = np.random.normal(0, 15, size=y.shape)
y = y + noise

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=2)

st.sidebar.title('DecisionTreeRegressor() Hyperparameters')

showData()

criterion=st.sidebar.selectbox('Criterion',['squared_error','friedman_mse','absolute_error','poisson'])
splitter=st.sidebar.selectbox('Splitter',['best','random'])
max_depth=st.sidebar.number_input(label='Max Depth',step=1,min_value=1,value=None)
min_samples_split=st.sidebar.select_slider(label='Min Samples Split',options=list(range(2,len(X_test)+1)),value=2)
min_samples_leaf=st.sidebar.select_slider(label='Min Samples Leaf',options=list(range(1,len(X_test)+1)),value=1)
max_leaf_nodes=st.sidebar.number_input(label='Max Leaf Nodes',min_value=2,step=1,value=None)
min_impurity_decrease=st.sidebar.number_input(label='Min Impurity Decrease',min_value=0.0,value=0.0)

# if st.sidebar.button('Show Data'):
    # showData()

if st.sidebar.button('Run'):
    drawRegressionLine()

