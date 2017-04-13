import pandas as pd

def raw_training_set():
    return pd.read_csv('data/train.csv')

def raw_test_set():
    return pd.read_csv('data/test.csv')

def preprocessed_training_set():
    df = pd.read_csv('data/train.csv')
    df = df.drop(['Ticket'], 1)
    df['PassengerId'].loc[df['PassengerId'] > 0] = "not mother"
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    #df['Fare'].fillna(8.05, inplace=True)
    df['Cabin'].fillna('U', inplace=True)
   
    df['Cabin'] = df['Cabin'].apply(lambda x: str(x)[0])
    df['Age'].fillna(-1, inplace=True)
    df = df.fillna(0)
    
    # Replace names with titles
    df['Name'].replace({'(.*, )|(\\..*)':'', 'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'}, regex=True, inplace=True)

    common_titles = { 'Mr', 'Mrs', 'Master', 'Miss' }
    df['Name'][df['Name'].apply(lambda x: x not in common_titles)] = "Rare"

    title_map = { 'Mr':1, 'Master':2, 'Mrs':3, 'Miss':4, 'Rare':5 }
    
    df['PassengerId'].loc[(df['Age'] > 16) & ((df['Sex'] == "female") & ((df['Parch'] > 0) & (df['Name'] == "Mrs")))] = "mother"
    # Build family size feature and make discrete
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    #  print df.groupby(['Sex'])['Survived'].mean()
    #  print df.groupby(['Name'])['Survived'].mean()
    #  print df.groupby(['FamilySize'])['Survived'].mean()

    #df = df.drop(['SibSp', 'Parch'], 1)
    
    df['FamilySize'].loc[df['FamilySize'] > 4] = 5 
    df['FamilySize'].loc[(df['FamilySize'] < 5) & (df['FamilySize'] > 1)] = 3 

    df['Sex'].loc[(df['Age'] < 15) & (df['Age'] > -1)] = "child"

    #df['Age'].loc[(df['Age'] < 0) & ((df['Name'] == 'Miss') & (df['Parch'] > 0))] = 4
    #df['Age'].loc[(df['Age'] < 0) & (df['Name'] == 'Master')] = 4
    #df['Age'].loc[df['Age'] < 0] = 35
    df['Age'].loc[(df['Age'] < 15) & (df['Age'] > -1)] = "2"
    df['Age'].loc[df['Age'] > 64] = "1"
    df['Age'].loc[(df['Age'] < 65) & (df['Age'] > 14)] = "3"
    df['Age'].loc[df['Age'] < 0] = "-1"
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
    gender_map = { "male": 1, "female": 2, "child": 3 } 
    port_map = { "S": 1, "C": 2, "Q": 3 }
    cabin_map = {"U": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}
    age_map = {"-1": 0, "1": 1, "2": 2, "3": 3}
    #fare_map = {"low": 0, "high": 1}
    mother_map = {"not mother": 0, "mother": 1}

    df = df.replace({"Sex": gender_map, "Embarked": port_map, "Name": title_map, "Cabin": cabin_map, "Age": age_map, "PassengerId": mother_map})

    #print(df)

    return df

def preprocessed_test_set():
    df = pd.read_csv('data/test.csv')
    df = df.drop(['Ticket'], 1)
    df['PassengerId'].loc[df['PassengerId'] > 0] = "not mother"
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    #df['Fare'].fillna(8.05, inplace=True)
    df['Cabin'].fillna('U', inplace=True)

    df['Cabin'] = df['Cabin'].apply(lambda x: str(x)[0])
    df['Age'].fillna(-1, inplace=True)
    df = df.fillna(0)

    # Replace names with titles
    df['Name'].replace({'(.*, )|(\\..*)':'', 'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'}, regex=True, inplace=True)

    common_titles = { 'Mr', 'Mrs', 'Master', 'Miss' }
    df['Name'][df['Name'].apply(lambda x: x not in common_titles)] = "Rare"
    
    title_map = { 'Mr':1, 'Mrs':2, 'Master':3, 'Miss':4, 'Rare':5 }

    df['PassengerId'].loc[(df['Age'] > 16) & ((df['Sex'] == "female") & ((df['Parch'] > 0) & (df['Name'] == "Mrs")))] = "mother"
    # Build family size feature and make discrete
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    #df = df.drop(['SibSp', 'Parch'], 1)

    df['FamilySize'].loc[df['FamilySize'] > 4] = 5 
    df['FamilySize'].loc[(df['FamilySize'] < 5) & (df['FamilySize'] > 1)] = 3 
   
    df['Sex'].loc[(df['Age'] < 15) & (df['Age'] > -1)] = "child"
    
    #df['Age'].loc[(df['Age'] < 0) & ((df['Name'] == 'Miss') & (df['Parch'] > 0))] = 4
    #df['Age'].loc[(df['Age'] < 0) & (df['Name'] == 'Master')] = 4
    #df['Age'].loc[df['Age'] < 0] = 35
    df['Age'].loc[(df['Age'] < 15) & (df['Age'] > -1)] = "2"
    df['Age'].loc[df['Age'] > 64] = "1"
    df['Age'].loc[(df['Age'] < 65) & (df['Age'] > 14)] = "3"
    df['Age'].loc[df['Age'] < 0] = "-1"
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
    gender_map = { "male": 1, "female": 2, "child": 3 } 
    port_map = { "S": 1, "C": 2, "Q": 3 }
    cabin_map = {"U": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}
    age_map = {"-1": 0, "1": 1, "2": 2, "3": 3}
    #fare_map = {"low": 0, "high": 1}
    mother_map = {"not mother": 0, "mother": 1}

    df = df.replace({"Sex": gender_map, "Embarked": port_map, "Name": title_map, "Cabin": cabin_map, "Age": age_map, "PassengerId": mother_map})

    return df

def create_submission_csv(classifications):
    df = pd.DataFrame(pd.read_csv('data/test.csv')['PassengerId'])
    df['Survived'] = pd.DataFrame({ 'Survived': classifications })
    df.to_csv('data/result.csv', index=False)

if __name__ == "__main__":
    print "Train: "
    print preprocessed_training_set()
    print "Test: "
    print preprocessed_test_set()

