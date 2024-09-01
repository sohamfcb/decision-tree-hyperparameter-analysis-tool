import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score

st.set_page_config(page_title='Classification')

plt.style.use('seaborn-v0_8-darkgrid')

st.title('Decision Tree Classifier Hyperparameter Analysis')

def showData():
    
    plt.figure(figsize=(10,5))

    plt.scatter(X.iloc[:,0][y==0],X.iloc[:,1][y==0],c='red',label='Didn\'t Purchase')
    plt.scatter(X.iloc[:,0][y==1],X.iloc[:,1][y==1],c='green',label='Purchased')
    plt.xlabel('Age')
    plt.ylabel('Salary')
    plt.legend()

    st.pyplot(plt)
    plt.close()

def plot_decision_boundary(max_leaf_nodes2,criterion2='gini',splitter2='best',min_samples_split2=2,min_samples_leaf2=1,max_features2=None,max_depth2=None,min_impurity_decrease2=None):

    
    dtree=DecisionTreeClassifier(max_depth=None if max_depth2 is None else int(max_depth2),
                                criterion=criterion2,
                                splitter=splitter2,
                                min_samples_split=int(min_samples_split2),
                                min_samples_leaf=min_samples_leaf2,
                                max_features=max_features2,
                                max_leaf_nodes=None if max_leaf_nodes2 is None else int(max_leaf_nodes2),
                                min_impurity_decrease=min_impurity_decrease2)
    
    dtree.fit(X_train_scaled,y_train)

    y_pred=dtree.predict(X_test_scaled)
    y_pred_training=dtree.predict(X_train_scaled)

    a=np.arange(start=combined_scaled_df[:,0].min()-1,stop=combined_scaled_df[:,0].max()+1,step=0.01)
    b=np.arange(start=combined_scaled_df[:,1].min()-1,stop=combined_scaled_df[:,1].max()+1,step=0.01)

    XX,YY=np.meshgrid(a,b)

    input_array=pd.DataFrame(np.array([XX.ravel(),YY.ravel()]).T)
    labels=dtree.predict(input_array)
    labels=labels.reshape(XX.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    plt.figure(figsize=(10,5))

    plt.contourf(XX,YY,labels,cmap=cmap_light)
    plt.scatter(combined_scaled_df[:, 0], combined_scaled_df[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=40)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')

    plt.grid()
    
    # plt.axis(False)

    st.pyplot(plt)

    plt.close()

    col1,col2=st.columns(2)

    with col1:
        st.write(f'Training Accuracy: {accuracy_score(y_train,y_pred_training)}')

    with col2:
        st.write(f'Test Accuracy: {accuracy_score(y_test,y_pred)}')

    plt.figure(figsize=(20, 10))
    plot_tree(dtree, filled=True,feature_names=X.columns)
    st.pyplot(plt)


df=pd.read_csv("Social_Network_Ads.csv")
scaler=StandardScaler()

X=df[['Age','EstimatedSalary']]
y=df['Purchased']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=2)

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

combined_scaled_df=np.vstack((X_train_scaled,X_test_scaled))

st.sidebar.title('DecisionTreeClassifier() Hyperparameters')

showData()

criterion=st.sidebar.selectbox('Criterion',['gini','entropy','log_loss'])
splitter=st.sidebar.selectbox('Splitter',['best','random'])
max_depth=st.sidebar.number_input(label='Max Depth',step=1,min_value=1,value=None)
min_samples_split=st.sidebar.select_slider(label='Min Samples Split',options=list(range(2,401)),value=2)
min_samples_leaf=st.sidebar.select_slider(label='Min Samples Leaf',options=list(range(1,401)),value=1)
max_features=st.sidebar.select_slider(label='Max Features',options=list(range(1,len(X.columns)+1)),value=None)
max_leaf_nodes=st.sidebar.number_input(label='Max Leaf Nodes',min_value=2,step=1,value=None)
min_impurity_decrease=st.sidebar.number_input(label='Min Impurity Decrease',min_value=0.0,value=0.0)


if st.sidebar.button('Run'):

    plot_decision_boundary(criterion2=criterion,max_depth2=max_depth,splitter2=splitter,min_samples_split2=min_samples_split,min_samples_leaf2=min_samples_leaf,max_features2=max_features,max_leaf_nodes2=max_leaf_nodes,min_impurity_decrease2=min_impurity_decrease)


