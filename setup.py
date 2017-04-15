import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def raw_training_set():
    return pd.read_csv('data/train.csv')

def raw_test_set():
    return pd.read_csv('data/test.csv')

def preprocessed_training_set():
    df = pd.read_csv('data/train.csv')
    df = process_data_set(df)
    return df

def preprocessed_test_set():
    df = pd.read_csv('data/test.csv')
    df = process_data_set(df)
    return df

def get_person(passenger):
    age, sex, name, parch = passenger
    if age < 16:
        return 'child'
    else:
        if sex == 'female' and parch > 0 and name == 'Mrs':
            return 'mother'
        else:
            return sex

def get_mother(passenger):
    age, name, person, parch = passenger
    if age > 16 and person == 'female' and parch > 0 and name == 'Mrs':
        return "yes"
    else:
        return "no"

def process_data_set(df):
    df = df.drop(['PassengerId', 'Ticket', 'Cabin'], 1)
    
    df['Embarked'] = df['Embarked'].fillna('S')
    
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Fare'] = df['Fare'].astype(int)

     
    #df['Fare'].fillna(8.05, inplace=True)

    #  df['Person'] = df[['Age', 'Sex', 'Name', 'Parch']].apply(get_person, axis=1)
    #  df = df.drop(['Sex'], 1)
   
    #  df['Cabin'].fillna('U', inplace=True)
    #  df['Cabin'] = df['Cabin'].apply(lambda x: str(x)[0])
    
    average_age = df["Age"].mean()
    std_age = df["Age"].std()
    count_nan_age = df["Age"].isnull().sum()
    
    rand = np.random.randint(average_age - std_age, average_age + std_age, size = count_nan_age)

    df["Age"][np.isnan(df["Age"])] = rand
    df['Age'] = df['Age'].astype(int)

    df = df.fillna(0)
    
    df['Name'].replace({'(.*, )|(\\..*)':'', 'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'}, regex=True, inplace=True)
    common_titles = { 'Mr', 'Mrs', 'Master', 'Miss' }
    df['Name'].loc[df['Name'].apply(lambda x: x not in common_titles)] = "Rare"


    df['Person'] = df[['Age', 'Sex', 'Name', 'Parch']].apply(get_person, axis=1)


    #  df['Mother'] = df[['Age', 'Name', 'Person', 'Parch']].apply(get_mother, axis=1)

    # Build family size feature and make discrete
    
    
    #  print df.groupby(['SibSp'])['Survived'].plot.bar()
    #  print df.groupby(['Parch'])['Survived'].mean()
    
    df['Family'] = df['SibSp'] + df['Parch'] + 1
    #  df['Family'].loc[df['Family'] > 1] = 1
    #  df['Family'].loc[df['Family'] == 1] = 0

    #  df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

    #  df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

    #  df1 = pd.DataFrame(df.groupby(['FamilySize'])['Survived'].count())
    #  df2 = pd.DataFrame(df.groupby(['FamilySize'])['Survived'].sum())
    #  df1['Died'] = df1['Survived'] - df2['Survived']
    #  df1.drop(['Survived'], 1, inplace=True)
    #  df1['Survived'] = df2['Survived']
#  
    #  df1.plot.bar()
    #  plt.show()
    #  print df.groupby(['Sex'])['Survived'].mean()

    #  print df.groupby(['Sex'])['Survived'].mean()
    #  print df.groupby(['Name'])['Survived'].mean()
    #  print df.groupby(['FamilySize'])['Survived'].mean()

    #df = df.drop(['SibSp', 'Parch'], 1)
    
    #  df['FamilySize'].loc[df['FamilySize'] > 4] = 5 
    #  df['FamilySize'].loc[(df['FamilySize'] < 5) & (df['FamilySize'] > 1)] = 3 


    #df['Age'].loc[(df['Age'] < 0) & ((df['Name'] == 'Miss') & (df['Parch'] > 0))] = 4
    #df['Age'].loc[(df['Age'] < 0) & (df['Name'] == 'Master')] = 4
    #df['Age'].loc[df['Age'] < 0] = 35

    #  df['Age'].loc[(df['Age'] < 15) & (df['Age'] > -1)] = "2"
    #  df['Age'].loc[df['Age'] > 64] = "1"
    #  df['Age'].loc[(df['Age'] < 65) & (df['Age'] > 14)] = "3"
    #  df['Age'].loc[df['Age'] < 0] = "-1"
    '''
    sMedian = df['Fare'].loc[df['Embarked'] == "S"].median()
    qMedian = df['Fare'].loc[df['Embarked'] == "Q"].median()
    cMedian = df['Fare'].loc[df['Embarked'] == "C"].median()

    df['Fare'].loc[(df['Embarked'] == "S") & (df['Fare'] <= sMedian)] = "low"
    df['Fare'].loc[(df['Embarked'] == "S") & (df['Fare'] != "low")] = "high"
    df['Fare'].loc[(df['Embarked'] == "C") & (df['Fare'] <= cMedian)] = "low"
    df['Fare'].loc[(df['Embarked'] == "C") & (df['Fare'] != "low")] = "high"
    df['Fare'].loc[(df['Embarked'] == "Q") & (df['Fare'] <= qMedian)] = "low"
    df['Fare'].loc[(df['Embarked'] == "Q") & (df['Fare'] != "low")] = "high"
    '''

    #  median = df['Fare'].median()
#  
    #  df['Fare'].loc[(df['Fare'] < median / 3.0)] = 1
    #  df['Fare'].loc[(df['Fare'] >= median / 3.0) & (df['Fare'] <= median * 2.0 / 3.0)] = 2
    #  df['Fare'].loc[(df['Fare'] > median * 2.0 / 3.0)] = 3
    
    #  df.drop(['Age'], axis=1, inplace=True)

    #  print df['Fare'].isnull().values.any() 
    
    gender_map = { "male": 1, "female": 2, "child": 3, "mother": 4} 
    port_map = { "S": 1, "C": 2, "Q": 3 }
    cabin_map = { "U": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8 }
    sex_map = {"male": 0, "female": 1 }
    #  age_map = {"-1": 0, "1": 1, "2": 2, "3": 3}
    title_map = { 'Mr':1, 'Master':2, 'Mrs':3, 'Miss':4, 'Rare':5 }
    mother_map = {"no": 0, "yes": 1 }
    #fare_map = {"low": 0, "high": 1}
    

    df.replace({ "Person": gender_map, "Embarked": port_map, "Name": title_map, "Sex": sex_map }, inplace=True)

    # "Cabin": cabin_map
    #  "Age": age_map

    #print(df)

    return df

def create_submission_csv(classifications):
    df = pd.DataFrame(pd.read_csv('data/test.csv')['PassengerId'])
    df['Survived'] = pd.DataFrame({ 'Survived': classifications })
    df.to_csv('data/result.csv', index=False)

if __name__ == "__main__":
    print preprocessed_training_set().head()
    print preprocessed_test_set().head()

