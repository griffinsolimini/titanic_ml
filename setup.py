import pandas as pd

def raw_training_set():
    return pd.read_csv('data/train.csv')

def raw_test_set():
    return pd.read_csv('data/test.csv')

def preprocessed_training_set():
    df = pd.read_csv('data/train.csv')
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 1)
    df = df.fillna(0)

    gender_map = { "male": 0, "female": 1 } 
    port_map = { "S": 1, "C": 2, "Q": 3 }

    df = df.replace({"Sex": gender_map, "Embarked": port_map })

    return df

def preprocessed_test_set():
    df = pd.read_csv('data/test.csv')
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 1)
    df = df.fillna(0)

    gender_map = { "male": 0, "female": 1 } 
    port_map = { "S": 1, "C": 2, "Q": 3 }

    df = df.replace({"Sex": gender_map, "Embarked": port_map })

    return df

def create_submission_csv(classifications):
    df = pd.DataFrame(pd.read_csv('data/test.csv')['PassengerId'])
    df['Survived'] = pd.DataFrame({ 'Survived': classifications })

    df.to_csv('data/result.csv', index=False)

if __name__ == "__main__":
    print preprocessed_training_set().head()
    print preprocessed_test_set().head()
    print submission_ids().head()

