import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn import preprocessing, model_selection
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from category_encoders import TargetEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import scale
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor



'''
This is a function to process the numerical columns. They are scaled using the 
simple scale function and NaN values are filled using backward fill.
'''
def process_num(col):
    col = col.fillna(method="bfill")
    col = scale(col)
    return col
'''
Age is processed in this function. I was having issus with including age in the above function so I opted
to give it its own function.
'''
def process(col):
    col = col.fillna(method="bfill")
    return col
'''
Processing Universit Degree was done here. 0 values were replaced with unknown and NaN values were forward
filled.
'''
def process_uni(col):
    col = col.fillna(method="ffill")
    col = col.str.replace('0', 'unknown')
    return col
'''
In work experience #NUM! values had to be replaced.
'''
def process_workexp(col):
    col = col.fillna(method="ffill")
    col = col.str.replace('#NUM!', '0')
    return col
'''
Looking at the unique value in the gender column, f was changed to female but this result in the 'f' in 'female' changing to female so
a concatenated word was created so that had to be replaced. 
'''
def process_Gender(col):
    col = col.fillna(method="ffill")
    col = col.str.replace('0', 'unknown')
    col = col.str.replace('f', 'female')
    col = col.str.replace('femaleemale', 'female')
    return col

'''
This function processes the additiona income column. Its main role is changing the values from string to float types so that
they are treated appropriately in the computation of the predictions. 
'''
def process_additional_income(col):
    col = col.fillna(method="ffill")
    col = col.str.replace('EUR', '')
    col = col.astype('float32')
    return col

def main():

    '''
    Reading in the train and test files.
    '''

    train_file = 'tcd-ml-1920-group-income-train.csv'
    trainingdata = pd.read_csv(train_file)

    test_file = "tcd-ml-1920-group-income-test.csv"
    testingdata = pd.read_csv(test_file)

    '''
    I found in the data that there between these rows there was alot of redundant information.
    Rows were duplicated, and some values were the same across multiple rows, so in an
    effort to combat this I deleted them.
    '''

    trainingdata = trainingdata.drop([trainingdata.index[45730], trainingdata.index[567421]])
    '''
    To speed up compuation time, I deleted duplicate rows from the training data.
    '''
    trainingdata = trainingdata.drop_duplicates()
    
    '''
    The following lines are calling the above preprocessing functions on the columns
    in the dataset.
    '''
    trainingdata['Year of Record'] = process_num(trainingdata['Year of Record'])
    testingdata['Year of Record'] = process_num(testingdata['Year of Record'])


    trainingdata['Size of City'] = process_num(trainingdata['Size of City'])
    testingdata['Size of City'] = process_num(testingdata['Size of City'])


    trainingdata['Age'] = process(trainingdata['Age'])
    testingdata['Age'] = process(testingdata['Age'])
    '''
    Note here that I made a new column specifically for smaller cities as I found there to be an above average correlation here.
    '''
    trainingdata['small city']=trainingdata['Size of City'] <= 3000
    testingdata['small city']=testingdata['Size of City'] <= 3000


    trainingdata['Crime Level in the City of Employement'] = process_num(trainingdata['Crime Level in the City of Employement'])
    testingdata['Crime Level in the City of Employement'] = process_num(testingdata['Crime Level in the City of Employement'])


    trainingdata['Work Experience in Current Job [years]'] = process_workexp(trainingdata['Work Experience in Current Job [years]'])
    testingdata['Work Experience in Current Job [years]'] = process_workexp(testingdata['Work Experience in Current Job [years]'])


    trainingdata['University Degree'] = process_uni(trainingdata['University Degree'])
    testingdata['University Degree'] = process_uni(testingdata['University Degree'])
    

    trainingdata['Country'] = process(trainingdata['Country'])
    testingdata['Country'] = process(testingdata['Country'])


    trainingdata['Gender'] = process_Gender(trainingdata['Gender'])
    testingdata['Gender'] = process_Gender(testingdata['Gender'])


    trainingdata['Profession'] = process(trainingdata['Profession'])
    testingdata['Profession'] = process(testingdata['Profession'])
 

    trainingdata['Yearly Income in addition to Salary (e.g. Rental Income)']=process_additional_income(trainingdata['Yearly Income in addition to Salary (e.g. Rental Income)'])
    testingdata['Yearly Income in addition to Salary (e.g. Rental Income)']=process_additional_income(testingdata['Yearly Income in addition to Salary (e.g. Rental Income)'])

    trainingdata['senior']=trainingdata['Profession'].str.contains("senior", na=False)
    testingdata['senior']=testingdata['Profession'].str.contains("senior", na=False)

    '''
    This is selecting which columns to use for the training. I had spent alot of time working out the correlation between each and the Income.
    The most important & relevant columns, which I found, can be seen in the smaller list X on line 128. We spent time manipulating
    the data to try only use relevant columns. Ultimatley in the end we went with the larger set of columns as generally this improved our score
    slightly. 
    '''

    X = trainingdata[['Year of Record', 'University Degree',
        'Gender', 'Profession','Crime Level in the City of Employement', 
        'Country', 'small city', 'Size of City', 'Body Height [cm]', 'Age',
        'Work Experience in Current Job [years]', 'Satisfation with employer', 'Yearly Income in addition to Salary (e.g. Rental Income)', 'senior']]
    

    #X = trainingdata[['Profession', 'Country', 'Yearly Income in addition to Salary (e.g. Rental Income)', 'Size of City', 'Year of Record', 'Age', 'University Degree', 
    #'senior', 'small city']]

    '''
    Scaling the data with the log function improved our score slightly. We made sure to get the exponent of the predictions so as
    to output the correct prediction.
    '''
    Y = trainingdata[['Total Yearly Income [EUR]']].apply(np.log)

    '''
    Splitting the data 70/30
    '''
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    '''
    We had first started using CatBoost and below is the code for that.
    '''
    #model = CatBoostRegressor()
    #parameters = {'depth'         : range(5,10),
                 #'iterations'    : [1000],
                  #'od_type' : ['IncToDec'],
                  #'od_wait' : [100, 150]
                 #}
    '''
    In the end we changed to the LGBM regressor because this improved our score further and gave us our best.
    The runtime was also alot quicker than catboost so it meant testing was alot easier.
    I played around with some paramaters but found that using just the n_estimators and the default values for
    the other parameters resulted in the best score. 
    '''
    model = LGBMRegressor()
    parameters = {  'n_estimators' : [1000,1500],
                    #'max_depth' : range(5,10),
                    #'boosting_type' : ['dart'],
                    #'max_bin' : [15],
                    #'num_leaves' : [50],
                    #'learning_rate' : [0.001, 0.01]
    }
    grid = GridSearchCV(estimator=model, param_grid = parameters,cv=5, scoring='neg_mean_absolute_error', verbose=5, n_jobs=-1)

    '''
    Here the code is encoded using the inbuilt TargetEncoder method. This is passed in the pipeline aswell as gridsearch cross 
    validation. 
    '''
    pi = Pipeline(steps=[
                        ('enc', TargetEncoder()),
                        ('grid', grid)])

    print("Fitting...")

    pi.fit(X_train, Y_train)


    print("Finished Fitting.")

    Y_pred1 = pi.predict(X_test)
    
    '''
    Computing the mean absolute error and printing this result and the best paramaters from running cross validation
    through gridsearch.
    '''

    mae = mean_absolute_error(np.exp(Y_test), np.exp(Y_pred1))

    print("mae=", mae)
    print(grid.best_params_)

    '''
    Choosing columns for the test data. See line 126. 
    '''
    X1 = testingdata[['Year of Record', 'University Degree',
        'Gender', 'Profession','Crime Level in the City of Employement', 
        'Country', 'small city', 'Size of City', 'Body Height [cm]', 'Age',
        'Work Experience in Current Job [years]', 'Satisfation with employer', 'Yearly Income in addition to Salary (e.g. Rental Income)', 'senior']]

    #X1 = testingdata[['Profession', 'Country', 'Yearly Income in addition to Salary (e.g. Rental Income)', 'Size of City', 
    #'Year of Record', 'Age', 'University Degree', 'senior', 'small city']]

    '''
    Calculating the prediction from the test data and writing it to a file alongside the appropriate instances.
    Uploaded file to Kaggle is called 'Submission10kl.csv' 
    - Best score on Kaggle with 11621 on the private leaderboard.
    '''
    Y_pred = pi.predict(X1)

    instance = testingdata['Instance']
    a=pd.DataFrame.from_dict({
    'Total Yearly Income [EUR]': np.exp(Y_pred.flatten()),
    'Instance' : instance
    })

    a.to_csv('submission.csv',index=False)
    
    

if __name__ == "__main__":
    main()