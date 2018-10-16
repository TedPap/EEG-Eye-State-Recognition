Instructions on how to run the Matlab part of the project:

    1. Unzip the project folder
    2. Open Matlab and navigate to the Matlab/Code folder
    3. Open and run the featureHandling.m file
    4. Open Classification Learner from Matlab Apps tab
    5. On the Classification Learner window select New Session->From Workspace
    6. On the New Session window select Use rows as variables under the Step 1
    7. In the Step 1 area choose the variable you want to classify. We'll go on with the data_all variable.
    8. In the Step 2 area choose row_15 of data_all as a Response under Import as.
    9. In the Step 3 area 1o folds for cross validation and click Start Session.
    10. On the Classification Learner window select classifier (e.g. Fine KNN) and click Train.


Instructions on how to run the Neural Networks part of the project:

    1. Unzip the project folder
    2. Open a terminal in the Python/Code folder
    3. Run one of the networks by typing:
        3.1. python MLP.py
        3.2. python RNN.py (You should also comment line 64 of readData.py to disable data shuffling)
        3.3. python CFNN.py