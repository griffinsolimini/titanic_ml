import pandas as pd

def raw_training_set():
    return pd.read_csv('data/train.csv')

def raw_test_set():
    return pd.read_csv('data/test.csv')

def preprocessed_training_set():
    df = pd.read_csv('data/train.csv')
    df = df.drop(['PassengerId', 'Ticket', 'Cabin'], 1)
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df = df.fillna(0)
    
    # Replace names with titles
    df['Name'].replace({'(.*, )|(\\..*)':'', 'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'}, regex=True, inplace=True)

    common_titles = { 'Mr', 'Mrs', 'Master', 'Miss' }
    df['Name'][df['Name'].apply(lambda x: x not in common_titles)] = "Rare"

    title_map = { 'Mr':1, 'Master':2, 'Mrs':2, 'Miss':4, 'Rare':5 }
    
    # Build family size feature and make discrete
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    #  print df.groupby(['Sex'])['Survived'].mean()
    #  print df.groupby(['Name'])['Survived'].mean()
    #  print df.groupby(['FamilySize'])['Survived'].mean()

    df = df.drop(['SibSp', 'Parch'], 1)
    
    df['FamilySize'].loc[df['FamilySize'] > 4] = 5 
    df['FamilySize'].loc[(df['FamilySize'] < 5) & (df['FamilySize'] > 1)] = 3 

    gender_map = { "male": 1, "female": 2 } 
    port_map = { "S": 1, "C": 2, "Q": 3 }

    df = df.replace({"Sex": gender_map, "Embarked": port_map, "Name": title_map })

    return df

def preprocessed_test_set():
    df = pd.read_csv('data/test.csv')
    df = df.drop(['PassengerId', 'Ticket', 'Cabin'], 1)
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df = df.fillna(0)

    # Replace names with titles
    df['Name'].replace({'(.*, )|(\\..*)':'', 'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'}, regex=True, inplace=True)

    common_titles = { 'Mr', 'Mrs', 'Master', 'Miss' }
    df['Name'][df['Name'].apply(lambda x: x not in common_titles)] = "Rare"
    
    title_map = { 'Mr':1, 'Mrs':2, 'Master':3, 'Miss':4, 'Rare':5 }

    # Build family size feature and make discrete
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    df = df.drop(['SibSp', 'Parch'], 1)

    df['FamilySize'].loc[df['FamilySize'] > 4] = 5 
    df['FamilySize'].loc[(df['FamilySize'] < 5) & (df['FamilySize'] > 1)] = 3 

    gender_map = { "male": 1, "female": 2 } 
    port_map = { "S": 1, "C": 2, "Q": 3 }

    df = df.replace({"Sex": gender_map, "Embarked": port_map, "Name": title_map })

    return df

def create_submission_csv(classifications):
    df = pd.DataFrame(pd.read_csv('data/test.csv')['PassengerId'])
    df['Survived'] = pd.DataFrame({ 'Survived': classifications })
    df.to_csv('data/result.csv', index=False)

if __name__ == "__main__":
    print "Train: "
    print preprocessed_training_set().head()
    print "Test: "
    print preprocessed_test_set().head()

