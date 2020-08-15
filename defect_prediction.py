import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import SelectFromModel
import time
from copy import deepcopy

#list of all project file names
file_names = []
#row for statistics file containing the results
statistics_row = []

def main():
    global statistics_row, file_names
    #list of all projects
    files = load_all_data()
    file_names = [x[0] for x in files]
    files = [x[1] for x in files]
    print('Data loaded')
    #iterate through all files in release level data
    for project_index in range(398):
        print('Project:',project_index, 'in process...')
        #csv file named statistics for saving results
        statistics = pd.read_csv('statistics2-sorted.csv')
        #current project name
        print(file_names[project_index])
        #target project for predictions
        target = pd.DataFrame(files[project_index])
        #adding target name to statistics
        statistics_row.append(file_names[project_index])
        #removing target project from training data
        current_target = files.pop(project_index)
        #concatenating list elements to data frame object
        con_data = pd.concat(files)
        #pd.concat([statistics['file'],pd.DataFrame(file_names)], axis=1).to_csv('vergleich.csv')

        #starting preprocessing data and predicting
        start(con_data,target)

        columns = statistics.columns
        #adding data from current project to statistics
        row = pd.DataFrame([statistics_row], columns=columns)
        statistics = statistics.append(row)
        #saving file
        statistics.to_csv('statistics2-sorted.csv', index=False)
        #clearing row for next iteration
        statistics_row.clear()
        #putting target back to file list
        files.insert(project_index, current_target)


def load_all_data():
    'Loading all files and names from release level data'
    global file_names
    directory = 'release-level-data'
    files = []
    child_files = []
    # iterate through all files in release level data
    for i, filename in enumerate(os.listdir(directory)):
        # appending only csv files to list
        if filename.endswith(".csv"):
            child_files.append(filename)
            #read csv file
            file = pd.read_csv(os.path.join(directory, filename), delimiter=';', index_col='file')
            # editing file content, cutting bug matrix
            file = file.iloc[:, :file.columns.get_loc('BUGFIX_count') + 1]
            child_files.append(file)
            files.append(deepcopy(child_files))
            child_files.clear()
        else:
            #continue if file doesn't end with .csv
            continue

    return sorted(files, key=lambda x: x[0])

#select 20 artifacts from whole data per artifact in target project using Nearest neighbor algorithm
def knn_data_selection(data, target):
    print('Start Selection')
    #declaring model
    model = NearestNeighbors(n_neighbors=20, n_jobs=-1)
    #fit model
    model.fit(data.iloc[:, :data.columns.get_loc('BUGFIX_count')])
    files = []
    #iterating through all target artifacts
    for x in range(len(target.index)):
        #getting list of indices of most similar artifacts from training data
        idx = model.kneighbors(target.iloc[[x]], return_distance=False)
        for i in idx[0]:
            #appending similar artifacts
            files.append(data.iloc[[i]])
    #concatenate files to data frame
    data = pd.concat(files)
    return data.iloc[:, :data.columns.get_loc('BUGFIX_count')], data['BUGFIX_count']>0

#select features using random forest algorithm
def select_features_rf(data, labels):
    sel = SelectFromModel(RandomForestClassifier()).fit(data, labels)
    indices_selected = sel.get_support(indices=True)
    cols = [data.columns[i] for i in indices_selected]
    return cols

#function for calculating upper bound
def upper_bound(predictions, loc, labels):
    sum_loc = sum([y for x,y in enumerate(loc) if predictions[x] == False])
    bugs_missed = sum([1 for x,y in enumerate(labels) if predictions[x] == False and y == True ])
    return sum_loc/bugs_missed

#function for calculating lower bound
def lower_bound(predictions, loc, labels):
    sum_loc = sum([y for x, y in enumerate(loc) if predictions[x]==True])
    bugs_predicted = sum([1 for x,y in enumerate(labels) if predictions[x] == True and y == True])
    return sum_loc/bugs_predicted

#RandomForestClassifier
def random_forest(data, labels, test_data):
    clf = RandomForestClassifier(random_state=7)
    clf.fit(data, labels)
    return clf.predict(test_data)

#K-Nearest Neighbor
def knn(data, labels, test_data):
    clf = KNeighborsClassifier(n_neighbors=4, weights='distance')
    clf.fit(data, labels)
    return clf.predict(test_data)

#preprocessing data
def preprocessing(data,target):
    global statistics_row
    print('Start preprocessing')
    data = data.drop('imports', axis='columns') #w
    target = target.drop('imports', axis='columns')
    labels = data['BUGFIX_count']>0
    test_labels = target['BUGFIX_count']>0
    test_loc = target['SM_file_lloc']

    print('Start Random Forest')
    predictions = random_forest(data.iloc[:, :data.columns.get_loc('BUGFIX_count')], labels,
                                target.iloc[:, :target.columns.get_loc('BUGFIX_count')])
    lb_infinite = False
    try:
        lb = lower_bound(predictions, test_loc, test_labels)
    except Exception as e:
        print(e)
        statistics_row.append(-1)
        lb_infinite = True
        print('RF1:', 'lb infinite')

    if not lb_infinite:
        try:
            ub = upper_bound(predictions, test_loc, test_labels)
            if(ub-lb)<0:
                statistics_row.append(-(abs(ub - lb) / (abs(ub - lb) + 1000)))
            else:
                statistics_row.append((ub - lb) / ((ub - lb) + 1000))
            print(lb, '< C <', ub)

        except Exception as e:
            print(e)
            statistics_row.append(1)
            print('RF1:','ub infinite')

    #selecting features with random forest
    features = select_features_rf(data.iloc[:, :data.columns.get_loc('BUGFIX_count')], labels)
    #applying features to data
    test_data = target[features]
    data = data[features]
    columns = data.columns
    try:
        #defining function (sigmoid) for scaling data
        scaler = FunctionTransformer(func=lambda x: (np.e ** x) /((np.e ** x) + 1), validate=True)
    except RuntimeWarning as e:
        print(e)
    #applying sigoid function on data
    data = pd.DataFrame(scaler.fit_transform(data), columns=columns).fillna(0)
    test_data = pd.DataFrame(scaler.fit_transform(test_data),columns=columns).fillna(0)

    labels = pd.DataFrame(labels).reset_index(drop=True)

    data = pd.concat([data, labels], axis=1)
    data, labels = knn_data_selection(data, test_data)

    return data, labels, test_data, test_labels, test_loc

def start(raw_data, target):
    global statistics_row
    start = time.time()

    data, labels, test_data, test_labels, test_loc = preprocessing(raw_data,target)

    station = time.time()
    print('Start fitting after',station-start,'secs')

    predictions1 = knn(data, labels, test_data)
    lb_infinite = False
    try:
        lb = lower_bound(predictions1, test_loc, test_labels)
    except Exception as e:
        print(e)
        statistics_row.append(-1)
        print('KNN:', 'lb infinite')
        lb_infinite = True

    if not lb_infinite:
        try:
            ub = upper_bound(predictions1, test_loc, test_labels)
            print('KNN:', lb, '< C <', ub)
            if (ub - lb) < 0:
                statistics_row.append(-(abs(ub - lb) / (abs(ub - lb) + 1000)))
            else:
                statistics_row.append((ub - lb) / ((ub - lb) + 1000))
        except Exception as e:
            print(e)
            statistics_row.append(1)
            print('KNN:',' ub infinite')

    predictions2 = random_forest(data, labels, test_data)
    lb_infinite = False
    try:
        lb = lower_bound(predictions2, test_loc, test_labels)
    except Exception as e:
        print(e)
        statistics_row.append(-1)
        print('RF:', 'lb infinite')
        lb_infinite = True

    if not lb_infinite:
        try:
            ub = upper_bound(predictions2, test_loc, test_labels)
            print('RF:', lb, '< C <', ub)
            if (ub - lb) < 0:
                statistics_row.append(-(abs(ub - lb) / (abs(ub - lb) + 1000)))
            else:
                statistics_row.append((ub - lb) / ((ub - lb) + 1000))
        except Exception as e :
            print(e)
            statistics_row.append(1)
            print('RF:','ub infinite')


    end = time.time()
    print('Ended after',end-start,'secs')

if __name__=='__main__':
    main()