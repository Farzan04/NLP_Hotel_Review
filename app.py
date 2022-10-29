# Import the Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Libraries for Text
from pattern.en import lemma
from gensim.parsing.preprocessing import remove_stopwords
import spacy
from nltk.stem import WordNetLemmatizer
import re

# Libraries for Model Validation and Test
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB


st.set_page_config(page_title='Hotel Reviews', page_icon='logo.png')
                   #,layout = "wide" , initial_sidebar_state="collapsed")


if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame()
if 'data_label' not in st.session_state:
    st.session_state['data_label'] = pd.DataFrame()
    
if 'model_input_dataset' not in st.session_state:
    st.session_state['model_input_dataset'] = 0



def main_page():
    st.image('333.jpg')
    st.markdown("<h1 style='text-align: left;'>Hotel Reviews</h1>", unsafe_allow_html=True)
    st.write('---')
    st.session_state['start'] = 1

    def change():
        st.session_state['data'] = pd.DataFrame()
        st.session_state['data_label'] = pd.DataFrame()

    # Input DataSet Used to Train the Model---------------------------------------------------------------------------------------------------------------------------------------

    st.sidebar.header('User Input Parameters')
    st.sidebar.write('---')
    data_file = st.sidebar.selectbox(
        label ="Input DataSet",
        options=['Default','Upload'], on_change=change())

    # To use default file for training the model
    if data_file == 'Default':
        st.subheader('Input DataFrame')
        data = pd.read_csv('hotel_reviews.csv')
        st.session_state['data'] = data
        st.dataframe(data)
        use_cols = ['Review', 'Rating']
        data1 = data[use_cols]
        st.session_state['data_label'] = data1

    # Upload another file
    if data_file == 'Upload':
        file = st.sidebar.file_uploader('Upload DataSet In "csv" formate', type = 'csv', key='a')
        
        if file == None:
            st.error('Please Upload the file')
            st.stop()
            
        else:
            data = pd.read_csv(file)
            st.session_state['data'] = data
            st.subheader('Input DataSet')
            st.dataframe(data)
            
            # Columns
            use_cols = ['Review', 'Rating']
            
            # What to do if Columns is Present OR Not Present
            try:
                data1 = data[use_cols]
                st.session_state['data_label'] = data1

            except:
                st.error('Please Upload the correct file, your file must contain below columns')
                st.write(pd.DataFrame(use_cols, columns=['columns']))
                st.stop()

    # Input DataSet is Taken------------------------------------------------------------------------------------------------------------------------------------------------------



def hotel_attributes():
    try:
        st.image('download.jpg', 'Hotel Reviews')
        st.header('**Hotel Attributes**')
        st.write('---')
        
        # File to be Classify-----------------------------------------------------------------------------------------------------------------------------------------------------------
    
        # Upload that file you want to be Classify
        st.sidebar.header('User Input Parameters')
        st.sidebar.write('---')
        data_file = st.sidebar.selectbox(
            label ="Input File for Hotel Attributes",
            options=['Default','Upload'])
        
        # To use default file for training the model
        if data_file == 'Default':
            attributes_data = st.session_state['data_label']
            st.subheader('Input DataSet')
            st.dataframe(attributes_data)
            use_cols = ['Review']
            attributes_data = attributes_data[use_cols]
            
        # Upload another file
        if data_file == 'Upload':
            file = st.sidebar.file_uploader('Upload DataSet In "csv" formate', type = 'csv', key='a')
            
            if file == None:
                st.error('Please Upload the file')
                st.stop()
                
            else:
                data = pd.read_csv(file)
                st.subheader('Input DataSet')
                st.dataframe(data)
                
                # Columns
                use_cols = ['Review']
                
                # What to do if Columns is Present OR Not Present
                try:
                    attributes_data = data[use_cols]

                except:
                    st.error('Please Upload the correct file, your file must contain below columns')
                    st.write(pd.DataFrame(use_cols, columns=['columns']))
                    st.stop()
            
        # File to be Classify End-------------------------------------------------------------------------------------------------------------------------------------------------------
    
        
        def clean(data, col_name):
            from gensim.parsing.preprocessing import remove_stopwords
            data1 = data.copy()
            # Lowercase the reviews
            data1['cleaned']=data1[str(col_name)].apply(lambda x: x.lower())
            # Remove digits and punctuation marks
            data1['cleaned']=data1['cleaned'].apply(lambda x: re.sub('[^a-z]',' ', x))
            # Removing extra spaces if present
            data1['cleaned']=data1['cleaned'].apply(lambda x: re.sub(' +',' ',x))
            data1['cleaned']=data1['cleaned'].apply(lambda x: remove_stopwords(x))
        
            return data1
        
        def vector(data1, ngram_range=(2,2), max_features=10000):
            from sklearn.feature_extraction.text import CountVectorizer
        
            # Converet Text into Bi-grams Vectors
            vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=max_features)
            X = vectorizer.fit_transform(data1.cleaned)
    
            # Convert X into DataFrame
            count_vect_df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names_out())
        
            vocab = vectorizer.vocabulary_
            word_list = vocab.keys()
        
            return count_vect_df, word_list
        
        def pos_neg_comment(word_list):
            # Open the positive and negative words
            with open("positive-words.txt","r") as pos:
                pos_word = pos.read().split("\n")
        
            with open("negative-words.txt","r") as pos:
                neg_word = pos.read().split("\n")
        
            # Store positive comment in a list
            pos_comment = []
            for word in word_list:
                if word.split()[0] in pos_word:
                    pos_comment.append(word)
                elif word.split()[-1] in pos_word:
                    pos_comment.append(word)
    
    
            # Store negative comment in a list
            neg_comment = []
            for word in word_list:
                if word.split()[0] in neg_word:
                    neg_comment.append(word)
                elif word.split()[-1] in neg_word:
                    neg_comment.append(word)
        
            return pos_comment, neg_comment
        
        
        # Function for Lemmatization
        def lemma(word):
            #from nltk.stem import WordNetLemmatizer
        
            wordnet=WordNetLemmatizer()
            return wordnet.lemmatize(word)
        
        def pos_neg_attributes(pos_comment, neg_comment):
            #import spacy
            # Load the spacy module
            nlp = spacy.load('en_core_web_sm')
        
            # store positive attributes and the corresponding noun in List
            pos_attributes = []
            pos_attributes_noun = []
    
            for text in pos_comment:
                doc = nlp(text)
                if (doc[0].pos_=='ADJ' and doc[1].pos_=='NOUN') or (doc[0].pos_=='NOUN' and doc[1].pos_=='ADJ'):
                    pos_attributes.append(doc)
                    if doc[1].pos_=='NOUN':
                        pos_attributes_noun.append(lemma(str(doc[1])))
                    else:
                        pos_attributes_noun.append(lemma(str(doc[0])))
                
            #----------------------------------------------------------------------------------------------------------------------
        
            # store negative attributes and the corresponding noun in List
            neg_attributes = []
            neg_attributes_noun = []
    
            for text in neg_comment:
                doc = nlp(text)
                if (doc[0].pos_=='ADJ' and doc[1].pos_=='NOUN') or (doc[0].pos_=='NOUN' and doc[1].pos_=='ADJ'):
                    neg_attributes.append(doc)
                    if doc[1].pos_=='NOUN':
                        neg_attributes_noun.append(lemma(str(doc[1])))
                    else:
                        neg_attributes_noun.append(lemma(str(doc[0])))
                
            return pos_attributes, pos_attributes_noun, neg_attributes, neg_attributes_noun
        
        
        def pos_neg_attributes_df(count_vect_df, pos_attributes, pos_attributes_noun, neg_attributes, neg_attributes_noun):
            # store positive attributes and the corresponding noun into DataFrame and also add the count of that word
            pos_dt = {}
            no_of_review = []
            for word in pos_attributes:
                pos_dt[str(word)]=count_vect_df[str(word)].sum()
                unique = count_vect_df[count_vect_df[str(word)]>0].shape
                no_of_review.append(unique[0])
        
            pos_s = pd.Series(pos_dt)
    
            pos_df = pd.DataFrame(pos_s, columns=['counts_pos'])
            pos_df['pos_no_of_review'] = no_of_review
            pos_df['noun'] = pos_attributes_noun
    
            # Sort the dataframe into Descending Order
            pos_df = pos_df.sort_values('counts_pos', ascending=False)
        
            #-----------------------------------------------------------------------------------------------------------------------
        
            # store negative attributes and the corresponding noun into DataFrame and also add the count of that word
            neg_dt = {}
            no_of_review = []
            for word in neg_attributes:
                neg_dt[str(word)]=count_vect_df[str(word)].sum()
                unique = count_vect_df[count_vect_df[str(word)]>0].shape
                no_of_review.append(unique[0])
        
            neg_s = pd.Series(neg_dt)
    
            neg_df = pd.DataFrame(neg_s, columns=['counts_neg'])
            neg_df['neg_no_of_review'] = no_of_review
            neg_df['noun'] = neg_attributes_noun
    
            # Sort the dataframe into Descending Order
            neg_df = neg_df.sort_values('counts_neg', ascending=False)
        
            return pos_df, pos_dt, neg_df, neg_dt
        
        
        def pos_neg_attributes_noun(pos_df, neg_df):
            # group by based on 'noun' and add counts
            pos_attributes_noun_df = pos_df.groupby('noun', as_index=False).sum().sort_values('counts_pos', ascending=False)
            pos_attributes_noun_df = pos_attributes_noun_df[(pos_attributes_noun_df['noun']!='hotel') & 
                                                            (pos_attributes_noun_df['noun']!='bit')].reset_index(drop=True)
        
            # group by based on 'noun' and add counts
            neg_attributes_noun_df = neg_df.groupby('noun', as_index=False).sum().sort_values('counts_neg', ascending=False)
            neg_attributes_noun_df = neg_attributes_noun_df[(neg_attributes_noun_df['noun']!='hotel') & 
                                                            (neg_attributes_noun_df['noun']!='bit')].reset_index(drop=True)
        
            return pos_attributes_noun_df, neg_attributes_noun_df
        
        
        def pos_neg_compare_df(pos_attributes_noun_df, neg_attributes_noun_df):
            pos_neg_compare = pd.merge(pos_attributes_noun_df, neg_attributes_noun_df, how='inner', on='noun')
        
            # Find the % for a noun, i.e. how much % that attribute is used with + sentiment and with - sentiment
            pos_neg_compare['pos_percent'] = round(pos_neg_compare['pos_no_of_review']/(pos_neg_compare['pos_no_of_review']+
                                                                                        pos_neg_compare['neg_no_of_review']), 3)
    
            pos_neg_compare['neg_percent'] = round(pos_neg_compare['neg_no_of_review']/(pos_neg_compare['pos_no_of_review']+
                                                                                        pos_neg_compare['neg_no_of_review']), 3)
        
            return pos_neg_compare
        
    
        def wdcld():
            from wordcloud import WordCloud
            from PIL import Image
    
            image = np.array(Image.open("download.png"))
    
            wc = WordCloud(width = 3000, height = 2000, background_color='black', random_state=1, max_words=100, mask=image)
        
            return wc
        
        
        def pos_attributes_wordcloud(pos_dt, pos_attributes_noun_df):
        
            wc = wdcld()
            # generate positive attributes word cloud
            wc.generate_from_frequencies(pos_dt)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            # show
            plt.figure(figsize=(15,10))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title('Positive Attributes WordCloud', fontsize=20)
            plt.show()
            st.pyplot()
    
            # converting 'pos_attributes_noun' dataframe into dictionary
            pos_attributes_noun = pos_attributes_noun_df.set_index('noun').to_dict()['counts_pos']
    
            # generate positive-noun attributes word cloud
            wc.generate_from_frequencies(pos_attributes_noun)
    
            # show
            plt.figure(figsize=(10,8))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title('Positive Attributes WordCloud', fontsize=20)
            plt.show()
            st.pyplot()
        
        
        def neg_attributes_wordcloud(neg_dt, neg_attributes_noun_df):
        
            wc = wdcld()
            # generate negative attributes word cloud
            wc.generate_from_frequencies(neg_dt)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            # show
            plt.figure(figsize=(15,10))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title('Negative Attributes WordCloud', fontsize=20)
            plt.show()
            st.pyplot()
        
            # converting 'neg_attributes_noun' dataframe into dictionary
            neg_attributes_noun = neg_attributes_noun_df.set_index('noun').to_dict()['counts_neg']
        
            # generate negative-noun attributes word cloud
            wc.generate_from_frequencies(neg_attributes_noun)
    
            # show
            plt.figure(figsize=(10,8))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title('Negative Attributes WordCloud', fontsize=20)
            plt.show()
            st.pyplot()
        
        
        def plot_pos_neg_attributes_count(pos_neg_compare):
            no_att = 10
            fig, axes = plt.subplots(2, 1 ,figsize= (30,16))
            (ax1,ax2) = axes
        
            ind = np.array(range(len(pos_neg_compare[:no_att])))
    
            ax1.bar(x=ind-0.2, width=0.4, height='pos_no_of_review', data=pos_neg_compare[:no_att], label = 'pos_counts')
            ax1.bar(x=ind+0.2, width=0.4, height='neg_no_of_review', data=pos_neg_compare[:no_att], label = 'neg_counts')
    
            for rect in ax1.patches:
                height = rect.get_height()
                if np.isnan(height):
                    height = 0
                width = rect.get_width()
                if np.isnan(width):
                    width = 0
                x = rect.get_x()
                y = rect.get_y()
                label_x = x + width / 2
                label_y = y + height + 10
                ax1.text(label_x, label_y, round(height), ha='center' ,fontsize= 20)
        
            X = np.array(pos_neg_compare[:no_att].noun)
            ax1.set_xticklabels(X, fontsize=10, color='w')
    #       ax1.set_xlabel('attributes',fontsize=10)
            ax1.set_ylabel('counts',fontsize=20)
            ax1.set_title('No. of Reviews contain an attribute', fontsize=30)
            ax1.legend()
    
    
            ax2.bar(x='noun', height='pos_percent', data=pos_neg_compare[:no_att], label = 'pos_counts')
            ax2.bar(x='noun', height='neg_percent', data=pos_neg_compare[:no_att], bottom='pos_percent', label = 'neg_counts')
    
            for rect in ax2.patches:
                # Find where everything is located
                height = rect.get_height()
                width = rect.get_width()
            
                x = rect.get_x()
                y = rect.get_y()
    
                # The width of the bar is also not pixels, it's the
                label_text = '{:.2f}'.format(height)
    
                label_x = x + width / 2
                label_y = y + height / 2
                ax2.text(label_x, label_y, label_text, ha='center', va='center', fontsize= 20)
    
            ax2.set_xlabel('attributes',fontsize=20)
            ax2.set_ylabel('percentage',fontsize=20)
            ax2.set_xticklabels(X, fontsize=20)
            ax2.set_title('% of Reviews contain an attribute', fontsize=30)
            ax2.legend()
            
            st.pyplot(fig)
        
        def main(data, col_name):
            data1 = clean(data, col_name)
            #st.write(data1)
            count_vect_df, word_list = vector(data1, ngram_range=(2,2), max_features=10000)
            #st.write(count_vect_df)
            #st.write('word_list')
            #st.write(word_list)
            pos_comment, neg_comment = pos_neg_comment(word_list)
            #st.write('pos_comment')
            #st.write(pos_comment)
            #st.write('neg_comment')
            #st.write(neg_comment)
            pos_attributes, pos_attributes_noun, neg_attributes, neg_attributes_noun = pos_neg_attributes(pos_comment, neg_comment)
            #st.write('pos_attributes')
            #st.write(pos_attributes)
            #st.write('pos_attributes_noun')
            #st.write(pos_attributes_noun)
            #st.write('neg_attributes')
            #st.write(neg_attributes)
            #st.write('neg_attributes_noun')
            #st.write(neg_attributes_noun)
            pos_df, pos_dt, neg_df, neg_dt = pos_neg_attributes_df(count_vect_df, pos_attributes, pos_attributes_noun, neg_attributes, neg_attributes_noun)
            #st.write('pos_df')
            #st.write(pos_df)
            #st.write(pos_dt)
            #st.write('neg_df')
            #st.write(neg_df)
            #st.write('neg_dt')
            #st.write(neg_dt)
            pos_attributes_noun_df, neg_attributes_noun_df = pos_neg_attributes_noun(pos_df, neg_df)
            #st.write('pos_attributes_noun_df')
            #st.write(pos_attributes_noun_df)
            #st.write('neg_attributes_noun_df')
            #st.write(neg_attributes_noun_df)
            pos_neg_compare = pos_neg_compare_df(pos_attributes_noun_df, neg_attributes_noun_df)
            #st.write('pos_neg_compare')
            #st.write(pos_neg_compare)
            pos_attributes_wordcloud(pos_dt, pos_attributes_noun_df)
            neg_attributes_wordcloud(neg_dt, neg_attributes_noun_df)
            plot_pos_neg_attributes_count(pos_neg_compare)
        
        
        main(attributes_data, 'Review')

    except:
        st.error('Go to **Main Page** and Upload the DataSet')


def model_validation():
    try:
        
        st.image('download.jpg', 'Hotel Reviews')
        st.header('**Model Validation**')
        st.write('---')


        data1 = st.session_state['data_label']


        # Data Preprocessing --------------------------------------------------------------------------------------------------------------------------------------
        

        def lemmatized(text):
            try:
                return " ".join([lemma(wd) for wd in text.split()])
            except RuntimeError:
                return " ".join([lemma(wd) for wd in text.split()])

        pos = [5,4]
        neg = [1,2]
        neu=[3]

        def sentiment(rating):
            if rating in pos:
                return "positive"
            elif rating in neg:
                return "negative"
            elif rating in neu:
                return "neutral"

        def clean(data, review_col_name, rating_col_name):
            
            data1 = data.copy()
            #st.write('1', data1)
            # Word Length of reviews
            data1['word_len_review'] = data1[review_col_name].apply(lambda x: len(x.split()))
            #st.write('2', data1)
            # String Length of reviews
            data1['string_len_review'] = data1[review_col_name].apply(lambda x: len(x))
            #st.write('3', data1)
            
            # Lowercase the reviews
            data1['cleaned']=data1['Review'].apply(lambda x: x.lower())
            # Remove digits and punctuation marks
            data1['cleaned']=data1['cleaned'].apply(lambda x: re.sub('[^a-z]',' ', x))
            # Removing extra spaces if present
            data1['cleaned']=data1['cleaned'].apply(lambda x: re.sub(' +',' ',x))
            #st.write('4', data1)
            data1['cleaned']=data1['cleaned'].apply(lambda x: remove_stopwords(x))
            #st.write('5', data1)
            data1['lemmatization']=data1['cleaned'].apply(lambda x: lemmatized(x))
            
            #st.write('6', data1)
            data1['Sentiment'] = data1[rating_col_name].apply(sentiment)
            data1['label'] = data1['Sentiment'].map({'positive':1, 'negative':-1, 'neutral':0})
            #st.write('7', data1)
            
            return data1

        data1 = clean(data1, 'Review', 'Rating')
        st.subheader('Training DataSet')
        st.write(data1)

        st.session_state['model_input_dataset'] = data1


        # Convert Text Data Into Numerical Data
        tv = TfidfVectorizer(max_features=5000)
        X = tv.fit_transform(data1['lemmatization']).toarray()
        X=pd.DataFrame(X, columns=tv.get_feature_names())#.set_flags(allows_duplicate_labels=True)

        #final_data = pd.concat([data1[['label','word_len_review','string_len_review']],X], axis=1)
        #st.write('1.final_data')
        
        minmax = MinMaxScaler(feature_range = (0 , 1))

        minmax = minmax.fit_transform(data1[['word_len_review','string_len_review']])
        minmax = pd.DataFrame(minmax, columns=['word_len_review','string_len_review'])

        final_data = pd.concat([data1['label'],minmax,X], axis=1)
        #st.write('2.final_data')




        

        model = SelectKBest(score_func=chi2, k='all')
        fit = model.fit(final_data.iloc[:,1:], final_data.iloc[:,0])
        scores = np.around(fit.scores_, 3)

        idx_cols = list(np.where(scores>0.5)[0])
        idx_cols = [x+1 for x in idx_cols]

        final_data = pd.concat([final_data.iloc[:,0],final_data.iloc[:,idx_cols]], axis=1)
        #st.write('3.final_data')

                
                
        # Spliting into X, y
        X = final_data.iloc[:,1:]
        y = final_data.iloc[:,0]

        # split X and y into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

        # Model Training
        model = MultinomialNB()

        # Hyper parameter Value
        kfold = KFold()
        alpha = np.arange(0.1, 1.1, 0.1)
        param_grid = {'alpha':alpha}

        # Hyper parameter tunning using GridSearchCV
        model = MultinomialNB()
        grid = GridSearchCV(estimator=model, param_grid=param_grid , cv = kfold, n_jobs=2)
        grid.fit(X_train, y_train)
        para = grid.best_params_

        model = MultinomialNB(alpha=para['alpha'])
        model.fit(X_train, y_train)
        #st.write('Model Fitted')



        y_train_pred = model.predict(X_train)

        st.subheader('Training Accuracy')
        st.write('Train Accuracy Score : ' , round(accuracy_score(y_train, y_train_pred),3))
        st.write('Train F1 Score : ' , round(f1_score(y_train, y_train_pred, average='weighted'),3))
        st.write('Train Precision Score : ' , round(precision_score(y_train, y_train_pred, average='weighted'),3))
        st.write('Train Recall Score : ' , round(recall_score(y_train, y_train_pred, average='weighted'),3))

        # print classification report
        st.text('Model Report on Training DataSet:\n ' 
                +classification_report(y_train, y_train_pred, digits=4))

        cm = confusion_matrix(y_train, y_train_pred)
        dt = {'Negative':list(cm[0]), 'Neutral':list(cm[1]), 'Positive':list(cm[2])}
        # Confusion Matrix for Train Data
        cm = pd.DataFrame(dt, index=['Negative', 'Neutral', 'Positive'])

        sns.set_theme(style='dark')
        sns.set(rc={'axes.facecolor':'#282828', 'figure.facecolor':'#282828'})

        fig, ax = plt.subplots()
        sns.heatmap(cm,annot=True,fmt='.0f', ax=ax)
        #ax.tick_params(grid_color='r', labelcolor='r', color='r')

        plt.ylabel('Predictions', fontsize=18)
        plt.xlabel('Actuals', fontsize=18)
        plt.title('Training Confusion Matrix', fontsize=18)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.tick_params(colors='white')
        ax.figure.axes[-1].tick_params(colors='white')
        plt.show()
        st.write(fig)

        st.text("")
        st.text("")
        st.write('#')

        # Training Validation End___________________________________________________________________________________

                
        # Testing Validation________________________________________________________________________________________

        #Predict for X dataset       
        y_test_pred = model.predict(X_test)

        # Testing Accuracy and F1-Score
        st.subheader('Testing Accuracy')
        st.write('Test Accuracy Score : ' , round(accuracy_score(y_test, y_test_pred),3))
        st.write('Test F1 Score : ' , round(f1_score(y_test, y_test_pred, average='weighted'),3))
        st.write('Test Precision Score : ' , round(precision_score(y_test, y_test_pred, average='weighted'),3))
        st.write('Test Recall Score : ' , round(recall_score(y_test, y_test_pred, average='weighted'),3))

        # print classification report
        st.text('Model Report on Testing DataSet:\n ' 
                +classification_report(y_test, y_test_pred, digits=4))       
                

        # Confusion Matrix for Test Data      
        cm = confusion_matrix(y_test, y_test_pred)
        dt = {'Negative':list(cm[0]), 'Neutral':list(cm[1]), 'Positive':list(cm[2])}
        cm = pd.DataFrame(dt, index=['Negative', 'Neutral', 'Positive'])

        sns.set_theme(style='dark')
        sns.set(rc={'axes.facecolor':'#282828', 'figure.facecolor':'#282828'})

        fig, ax = plt.subplots()
        sns.heatmap(cm,annot=True,fmt='.0f', ax=ax)
        #ax.tick_params(grid_color='r', labelcolor='r', color='r')
            
        plt.ylabel('Predictions', fontsize=18)
        plt.xlabel('Actuals', fontsize=18)
        plt.title('Testing Confusion Matrix', fontsize=18)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.tick_params(colors='white')
        ax.figure.axes[-1].tick_params(colors='white')
        plt.show()
        st.write(fig)
    
        # Test Validation End_______________________________________________________________________________________
                
        # Model Validation End-------------------------------------------------------------------------------------------------------------------------------------------------------------
                
            
    except:
        st.error('Go to **Main Page** and Upload the DataSet')            
            
            
            
            

def model_test():
    try:
        st.image('download.jpg', 'Hotel Reviews')
        st.header('**Model Prediction**')
        st.write('---')
                
        data1 = st.session_state['model_input_dataset']
        
        # File to be Classify-----------------------------------------------------------------------------------------------------------------------------------------------------------
    
        # Upload that file you want to be Classify
        st.sidebar.write('**Upload the Data you want to Classify**')
        test_file = st.sidebar.file_uploader('Upload DataSet In "csv" formate', type = 'csv', key='b')
    
        # If File is not Uploaded
        if test_file == None:
            st.error('Please Upload the file')
            st.stop()
            
        # If File is Uploaded
        else:
            test_data = pd.read_csv(test_file)
            st.subheader('DataSet to be Classify')
            st.dataframe(test_data)
    
        # Column that are used for Test DataSet
        use_test_cols = ['Review']
    
        # What to do if Columns is Present OR Not Present
        try:
            test_data = test_data[use_test_cols]
    
        except:
            st.error('File to be Classify is not correct, Please upload the correct file and your file contain below columns')
            st.write(pd.DataFrame(use_test_cols, columns=['columns']))
            st.stop()
            
        # File to be Classify End-------------------------------------------------------------------------------------------------------------------------------------------------------
    
        # Data Preprocessing --------------------------------------------------------------------------------------------------------------------------------------


        def lemmatized(text):
            try:
                return " ".join([lemma(wd) for wd in text.split()])
            except RuntimeError:
                return " ".join([lemma(wd) for wd in text.split()])

        pos = [5,4]
        neg = [1,2]
        neu=[3]

        def sentiment(rating):
            if rating in pos:
                return "positive"
            elif rating in neg:
                return "negative"
            elif rating in neu:
                return "neutral"


        st.subheader('Training DataSet')
        st.write(data1)
        
        def clean(data, review_col_name):
            
            data1 = data.copy()
            #st.write('1', data1)
            # Word Length of reviews
            data1['word_len_review'] = data1[review_col_name].apply(lambda x: len(x.split()))
            #st.write('2', data1)
            # String Length of reviews
            data1['string_len_review'] = data1[review_col_name].apply(lambda x: len(x))
            #st.write('3', data1)
            
            # Lowercase the reviews
            data1['cleaned']=data1['Review'].apply(lambda x: x.lower())
            # Remove digits and punctuation marks
            data1['cleaned']=data1['cleaned'].apply(lambda x: re.sub('[^a-z]',' ', x))
            # Removing extra spaces if present
            data1['cleaned']=data1['cleaned'].apply(lambda x: re.sub(' +',' ',x))
            #st.write('4', data1)
            data1['cleaned']=data1['cleaned'].apply(lambda x: remove_stopwords(x))
            #st.write('5', data1)
            data1['lemmatization']=data1['cleaned'].apply(lambda x: lemmatized(x))
            
            return data1

        test_data1 = clean(test_data, 'Review')
        st.subheader('Testing DataSet')
        st.write(test_data1)


        # Convert Text Data Into Numerical Data
        #from sklearn.feature_extraction.text import TfidfVectorizer
        tv = TfidfVectorizer(max_features=5000)
        X = tv.fit_transform(data1['lemmatization']).toarray()
        X_test = tv.transform(test_data1['lemmatization']).toarray()
        X=pd.DataFrame(X, columns=tv.get_feature_names())#.set_flags(allows_duplicate_labels=True)
        #st.write('1.final_data', X.shape)
        X_test=pd.DataFrame(X_test, columns=tv.get_feature_names())
        #st.write('1.final_test_data', X_test.shape)
        #final_data = pd.concat([data1[['label','word_len_review','string_len_review']],X], axis=1)
        #st.write('1.final_data')

        #from sklearn.preprocessing import MinMaxScaler
        minmax_model = MinMaxScaler(feature_range = (0 , 1))

        minmax = minmax_model.fit_transform(data1[['word_len_review','string_len_review']])
        test_minmax = minmax_model.transform(test_data1[['word_len_review','string_len_review']])
        minmax = pd.DataFrame(minmax, columns=['word_len_review','string_len_review'])
        test_minmax = pd.DataFrame(test_minmax, columns=['word_len_review','string_len_review'])

        final_data = pd.concat([data1['label'],minmax,X], axis=1)
        #st.write('2.final_data', final_data.shape)
        final_test_data = pd.concat([test_minmax,X_test], axis=1)
        #st.write('2.final_test_data', final_test_data.shape)


        #from sklearn.feature_selection import SelectKBest
        #from sklearn.feature_selection import chi2

        model = SelectKBest(score_func=chi2, k='all')
        fit = model.fit(final_data.iloc[:,1:], final_data.iloc[:,0])
        scores = np.around(fit.scores_, 3)

        id_cols = list(np.where(scores>0.5)[0])
        idx_cols = [x+1 for x in id_cols]

        final_data = pd.concat([final_data.iloc[:,0],final_data.iloc[:,idx_cols]], axis=1)
        #st.write('3.final_data', final_data.shape)
        final_test_data = final_test_data.iloc[:,id_cols]
        #st.write('3.final_test_data', final_test_data.shape)

        # Spliting into X, y
        X = final_data.iloc[:,1:]
        y = final_data.iloc[:,0]

        X_test = final_test_data

        # Model Training
        #model = MultinomialNB()

        # Hyper parameter Value
        #kfold = KFold()
        #alpha = np.arange(0.1, 1.1, 0.1)
        #param_grid = {'alpha':alpha}

        # Hyper parameter tunning using GridSearchCV
        #model = MultinomialNB()
        #grid = GridSearchCV(estimator=model, param_grid=param_grid , cv = kfold, n_jobs=2)
        #grid.fit(X, y)
        #para = grid.best_params_
        #st.write(para)
        #model = MultinomialNB(alpha=para['alpha'])
        model = MultinomialNB(alpha=0.1)
        model.fit(X, y)
        #st.write('Model Fitted')


        y_train_pred = model.predict(X)

        st.subheader('Training Accuracy')
        st.write('Train Accuracy Score : ' , round(accuracy_score(y, y_train_pred),3))
        st.write('Train F1 Score : ' , round(f1_score(y, y_train_pred, average='weighted'),3))
        st.write('Train Precision Score : ' , round(precision_score(y, y_train_pred, average='weighted'),3))
        st.write('Train Recall Score : ' , round(recall_score(y, y_train_pred, average='weighted'),3))

        # print classification report
        st.text('Model Report on Training DataSet:\n ' 
                +classification_report(y, y_train_pred, digits=4))

        cm = confusion_matrix(y, y_train_pred)
        dt = {'Negative':list(cm[0]), 'Neutral':list(cm[1]), 'Positive':list(cm[2])}
        # Confusion Matrix for Train Data
        cm = pd.DataFrame(dt, index=['Negative', 'Neutral', 'Positive'])

        sns.set_theme(style='dark')
        sns.set(rc={'axes.facecolor':'#282828', 'figure.facecolor':'#282828'})

        fig, ax = plt.subplots()
        sns.heatmap(cm,annot=True,fmt='.0f', ax=ax)
        #ax.tick_params(grid_color='r', labelcolor='r', color='r')

        plt.ylabel('Predictions', fontsize=18)
        plt.xlabel('Actuals', fontsize=18)
        plt.title('Training Confusion Matrix', fontsize=18)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.tick_params(colors='white')
        ax.figure.axes[-1].tick_params(colors='white')
        plt.show()
        st.write(fig)

        st.text("")
        st.text("")
        st.write('#')

        # Training Validation End___________________________________________________________________________________

                
        # Testing Validation________________________________________________________________________________________

        #Predict for X dataset       
        y_test_pred = model.predict(X_test)

        otp_data = test_data.copy()
        otp_data['label'] = y_test_pred
        otp_data['Sentiment'] = otp_data['label'].map({-1:'Negative', 0:'Neutral', 1:'Positive'})

        # Fuction define to color the dataframe
        def color_df(clas):
            if clas == 'Negative':
                color = 'tomato'
            elif clas == 'Positive':
                color = 'green'
            else:
                color = 'dimgrey'
                
            return f'background-color: {color}'

        # Final DataFrame
        st.subheader('Classified Data or Output')
        st.dataframe(otp_data.style.applymap(color_df, subset=['Sentiment']))

        # Value Counts of Final Dataframe
        dt = {'Sentiment_Classification':otp_data['Sentiment'].value_counts().index.tolist(), 
              'Counts':otp_data['Sentiment'].value_counts().values.tolist()}
        value_counts = pd.DataFrame(dt)
        st.subheader('Value Counts of Classified Data')
        st.dataframe(value_counts.style.applymap(color_df, subset=['Sentiment_Classification']))


        # Test Validation End_______________________________________________________________________________________
                
        # Model Validation End-------------------------------------------------------------------------------------------------------------------------------------------------------------
                
            
    except:
        st.error('Go to **Main Page** and Upload the DataSet')



def made_by():
    st.header('**Made By**')
    st.write('---')
    col1, col2, col3= st.columns([3,6,4])


    with col1:
        st.subheader("**Name**")
        st.write('Ayush Patidar')
        st.write('Anup Vetal')
        st.write('Farzan Nawaz')
        st.write('Prashant Khandekar')
        st.write('Prasad Waje')
        st.write('M Vikram')
        st.write('Nilesh Suresh Patil')

    with col2:
        st.subheader("**Mail**")
        st.write('ayushpatidar1712@gmail.com')
        st.write('anupsv1997@gmail.com')
        st.write('farzannawaz4787@gmail.com')
        st.write('pkhandekar108@gmail.com')
        st.write('prasadwaje2029@gmail.com')
        st.write('vikramkrishna06@gmail.com')
        st.write('itsnilesh45@gmail.com')

    with col3:
        st.subheader("**Mob. No.**")
        st.write('9131985346')
        st.write('8668314822')
        st.write('7898480467')
        st.write('7030870449')
        st.write('8999714455')
        st.write('9553300586')
        st.write('7875581402')


page_names_to_funcs = {
    "Main Page": main_page,
    "Hotel Attributes": hotel_attributes,
    "Model Validation": model_validation,
    "Data Classification": model_test,
    "Made By": made_by
}

st.sidebar.header('Select a Task')
st.sidebar.write('---')
selected_page = st.sidebar.selectbox("Select", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

