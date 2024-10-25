import numpy as np
import math
from collections import Counter
import time


class DecisionNode:
    """Class to represent a nodes or leaves in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """
        Create a decision node with eval function to select between left and right node
        NOTE In this representation 'True' values for a decision take us to the left.
        This is arbitrary, but testing relies on this implementation.
        Args:
            left (DecisionNode): left child node
            right (DecisionNode): right child node
            decision_function (func): evaluation function to decide left or right
            class_label (value): label for leaf node
        """
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Determine recursively the class of an input array by testing a value
           against a feature's attributes values based on the decision function.

        Args:
            feature: (numpy array(value)): input vector for sample.

        Returns:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice index for data labels.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data contained in the ReadMe.
    It must be built fully starting from the root.
    
    Returns:
        The root node of the decision tree.
    """
    
    # Functions for determining which move to make at each node
    func0 = lambda feature: feature[0] <= 0.06
    func1 = lambda feature: feature[1] <= -1.7
    func2 = lambda feature: feature[2] <= -0.7

    # Define the nodes
    dt_node2 = DecisionNode(None, None, func2, None)
    dt_node1 = DecisionNode(None, dt_node2, func1, None)
    dt_node0 = DecisionNode(None, dt_node1, func0, None)

    # Define the decisions at each node
    dt_node0.left = DecisionNode(None, None, None, class_label=0)
    dt_node1.left = DecisionNode(None, None, None, class_label=0)
    dt_node2.left = DecisionNode(None, None, None, class_label=2)
    dt_node2.right = DecisionNode(None, None, None, class_label=1)

    # Return the root of the tree
    return dt_node0


def confusion_matrix(true_labels, classifier_output, n_classes=2):
    """Create a confusion matrix to measure classifier performance.
   
    Classifier output vs true labels, which is equal to:
    Predicted  vs  Actual Values.
    
    Output will sum multiclass performance in the example format:
    (Assume the labels are 0,1,2,...n)
                                     |Predicted|
                     
    |A|            0,            1,           2,       .....,      n
    |c|   0:  [[count(0,0),  count(0,1),  count(0,2),  .....,  count(0,n)],
    |t|   1:   [count(1,0),  count(1,1),  count(1,2),  .....,  count(1,n)],
    |u|   2:   [count(2,0),  count(2,1),  count(2,2),  .....,  count(2,n)],'
    |a|   .............,
    |l|   n:   [count(n,0),  count(n,1),  count(n,2),  .....,  count(n,n)]]
    
    'count' function is expressed as 'count(actual label, predicted label)'.
    
    For example, count (0,1) represents the total number of actual label 0 and the predicted label 1;
                 count (3,2) represents the total number of actual label 3 and the predicted label 2.           
    
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
    Returns:
        A two dimensional array representing the confusion matrix.
    """
    
    # Define the confusion matrix
    c_matrix = np.zeros((n_classes,n_classes))

    # Compare each input's actual label to its predicted label and input to appropriate location in confusion matrix
    for idx in range(len(true_labels)):
        c_matrix[true_labels[idx], classifier_output[idx]] += 1
    
    return c_matrix


def precision(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """
    Get the precision of a classifier compared to the correct values.
    In this assignment, precision for label n can be calculated by the formula:
        precision (n) = number of correctly classified label n / number of all predicted label n 
                      = count (n,n) / (count(0, n) + count(1,n) + .... + count (n,n))
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The list of precision of each classifier output. 
        So if the classifier is (0,1,2,...,n), the output should be in the below format: 
        [precision (0), precision(1), precision(2), ... precision(n)].
    """
    
    # Regular method:
    
    # Define matrices for counting correct classifications and total number of classifications, the two factors in precision
    correct_matrix = np.zeros((1,n_classes))
    count_matrix = np.zeros((1,n_classes))
    
    # Compare each classification with the true label and get precision
    for idx in range(len(true_labels)):
        true_label = true_labels[idx]
        classification = classifier_output[idx]
        
        # Increase count of correct amount for this label
        if true_label == classification:
            correct_matrix[0,true_label] += 1

        # Increase count of this label            
        count_matrix[0,classification] += 1
        
    precision_matrix = correct_matrix / count_matrix
    
    return precision_matrix[0]

    # Another method with numpy, not faster
    
    # true_labels = np.array(true_labels)
    # classifier_output = np.array(classifier_output)

    # count_matrix = np.bincount(classifier_output, minlength=n_classes)
    # correct_matrix = np.bincount(true_labels[classifier_output==true_labels], minlength=n_classes)

    # return correct_matrix / count_matrix


def recall(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """
    Get the recall of a classifier compared to the correct values.
    In this assignment, recall for label n can be calculated by the formula:
        recall (n) = number of correctly classified label n / number of all true label n 
                   = count (n,n) / (count(n, 0) + count(n,1) + .... + count (n,n))
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The list of recall of each classifier output..
        So if the classifier is (0,1,2,...,n), the output should be in the below format: 
        [recall (0), recall (1), recall (2), ... recall (n)].
    """
    
    # Define matrices for counting correct classifications and total number of classifications, the two factors in precision
    correct_matrix = np.zeros((1,n_classes))
    count_matrix = np.zeros((1,n_classes))
    
    # Compare each classification with the true label and get precision
    for idx in range(len(true_labels)):
        true_label = true_labels[idx]
        classification = classifier_output[idx]
        
        # Increase count of correct amount for this label
        if true_label == classification:
            correct_matrix[0,true_label] += 1
        
        # Increase count of this label
        count_matrix[0,true_label] += 1
        
    recall_matrix = correct_matrix / count_matrix
    
    return recall_matrix[0]

def accuracy(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """Get the accuracy of a classifier compared to the correct values.
    Balanced Accuracy Weighted:
    -Balanced Accuracy: Sum of the ratios (accurate divided by sum of its row) divided by number of classes.
    -Balanced Accuracy Weighted: Balanced Accuracy with weighting added in the numerator and denominator.

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The accuracy of the classifier output.
    """
    
    # Convert to numpy
    true_labels = np.array(true_labels)
    classifier_output = np.array(classifier_output)   

    # Count correct labels
    num_correct = sum(true_labels == classifier_output)
    
    return num_correct/(len(classifier_output))


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0, 1, 2, ...
    Returns:
        Floating point number representing the gini impurity.
    """
    
    gini = 0
    
    # Convert class vector to np array
    classes_list = np.unique(class_vector)
    
    # for each class, find the percentage it appears in the class_vector
    for clss in classes_list:
        
        # This is part of the gini impurity calculation
        gini += (sum(class_vector == clss)/len(class_vector))**2

    # The rest of the calculation
    return 1 - gini
    
def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0, 1, 2....
        current_classes (list(list(int): A list of lists where each list has
            0, 1, 2, ... values).
    Returns:
        Floating point number representing the gini gain.
    """
    
    gini_imp_parent = gini_impurity(previous_classes)
    avg_gini_imp_children = 0
    total_samples = len(previous_classes);
    
    # Iterate through children, adding their affect to the gini gain
    for child in current_classes:
        child_weight = len(child) / total_samples
        avg_gini_imp_children += child_weight * gini_impurity(child)
    
    # Return gain calc
    return gini_imp_parent-avg_gini_imp_children


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=22):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """
        
        # If there is one class left, we have reached a leaf
        if len(np.unique(classes)) == 1:
            return DecisionNode(None, None, None, class_label=classes[0])
            
        # If we have reached the max depth
        if depth >= self.depth_limit:
            chosen_class = np.argmax(np.bincount(classes.astype(int)))
            return DecisionNode(None, None, None, class_label=chosen_class)
            
        # Initialize the best gini gain so we can choose the best one later    
        gini_gain_best = -float('inf')
        
        # Calculate the Gini gain for each feature to choose which feature to split on
        for feature_idx in range(features.shape[1]):
            feature = features[:,feature_idx]
            
            # Test n cutoff values for this feature
            max_data = np.max(feature)
            min_data = np.min(feature)
            num_cutoff_tests = 3
            
            for cutoff in np.linspace(min_data, max_data, num_cutoff_tests):
                
                # Define test on decision boundary cutoff. True assigned as <= cutoff
                true_idx = feature <= cutoff
                false_idx = feature > cutoff
                true = classes[true_idx]
                false = classes[false_idx]

                # Get gini gain based former classes and new classes from decision boundary
                gini_gain_val = gini_gain(previous_classes=classes, current_classes=[true,false])
                
                # Store info if gini gain is better
                if gini_gain_val > gini_gain_best:
                    gini_gain_best = gini_gain_val
                    cutoff_best = cutoff
                    feature_best = feature_idx
                    true_idx_best = true_idx
                    false_idx_best = false_idx
    
        # We have now chosen a best feature and cutoff value. Assign children based on the chosen feature to split on with true going left and false going right
        left_child = self.__build_tree__(features[true_idx_best],classes[true_idx_best],depth+1)
        right_child = self.__build_tree__(features[false_idx_best],classes[false_idx_best],depth+1)

        # Assign children to parent node with their corresponding cuttoff
        func = lambda x: x[feature_best] <= cutoff_best
        return DecisionNode(left_child, right_child, func, class_label=None)


    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """
        class_labels = []
        
        # Put each input through the fitted tree to get its output
        for feature in features:
            class_labels.append(self.root.decide(feature))
        
        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """
    folds = []
    
    # Separate the features and the labels from the dataset tuple
    features = dataset[0]
    labels = dataset[1]
    
    # Suffle data
    rand_indices = np.random.permutation(features.shape[0]) # Generate random indices of the size of the data
    shuffled_features = features[rand_indices]
    shuffled_labels = labels[rand_indices]
    
    # Size of folds to get
    fold_size = int(len(shuffled_features)/k)
    
    # Create test set from each fold of the data. The rest of the data is the training set. Iterate until all combinations are accounted for
    for fold in range(k):
        
        # Create indices for masking test and training sets from the data
        test_set_idx = np.linspace(fold*fold_size,(fold+1)*fold_size-1,fold_size).astype(int)
        training_set_idx = np.setdiff1d(np.arange(len(shuffled_features)), test_set_idx)
        
        test_set_features = shuffled_features[test_set_idx]
        training_set_features = shuffled_features[training_set_idx]
        test_set_labels = shuffled_labels[test_set_idx]
        training_set_labels = shuffled_labels[training_set_idx]
        
        folds.append(((training_set_features, training_set_labels),(test_set_features, test_set_labels)))
            
    return folds


class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees=200, depth_limit=5, example_subsample_rate=.1,
                 attr_subsample_rate=.3):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """
        self.trees = []
        self.num_classes = 0
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        # Get number of classes
        self.num_classes = len(np.unique(classes))
        
        # Get the number of features and attributes to work with given the sampling rate (critical part of random forest = not using all the data in each tree)
        num_features = int(features.shape[0]*self.example_subsample_rate)
        num_attributes = int(features.shape[1]*self.attr_subsample_rate)
        
        # Create all the trees
        for _ in range(self.num_trees):
            start = time.time()
            # Select the random features and attributes to use for this tree
            rand_features_idx = np.random.choice(features.shape[0],num_features,replace=True)
            rand_attributes_idx = np.random.choice(features.shape[1],num_attributes,replace=False)
    
            rand_features = features[rand_features_idx]
            rand_features = rand_features[:,rand_attributes_idx]
            rand_classes = classes[rand_features_idx]
        
            # Build the tree and store the chosen attributes for later testing
            d_tree = DecisionTree(self.depth_limit)
            d_tree.fit(rand_features, rand_classes)
            d_tree.attr_indices = rand_attributes_idx
            
            self.trees.append(d_tree)     
            end = time.time()
            print(end-start)       

    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        Returns:
            votes (list(int)): m votes for each element
        """
        
        votes = []
        
        # Put each input through each tree to get its output, but only classify each on its randomly chosen attributes
        for tree in self.trees:
            tree_features = features[:,tree.attr_indices]
            votes.append(tree.classify(tree_features))
            
        # Take votes from each tree for final vote list
        votes = np.array(votes).astype(int)
        
        # find the max voted for class for each data point
        result = np.apply_along_axis(lambda x: np.bincount(x).argmax(),axis=0, arr=votes)
        
        return result # return a list of votes


class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self, n_clf=0, depth_limit=0, example_subsample_rt=0.0, \
                 attr_subsample_rt=0.0, max_boost_cycles=0):
        """Create a boosting class which uses decision trees.
        Initialize and/or add whatever parameters you may need here.
        Args:
             num_clf (int): fixed number of classifiers.
             depth_limit (int): max depth limit of tree.
             attr_subsample_rate (float): percentage of attribute samples.
             example_subsample_rate (float): percentage of example samples.
        """
        self.num_clf = n_clf
        self.depth_limit = depth_limit
        self.example_subsample_rt = example_subsample_rt
        self.attr_subsample_rt=attr_subsample_rt
        self.max_boost_cycles = max_boost_cycles
        # TODO: finish this.͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏︀͏󠄋͏︊͏󠄏
        raise NotImplemented()

    def fit(self, features, classes):
        """Build the boosting functions classifiers.
            Fit your model to the provided features.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        # TODO: finish this.͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏︀͏󠄋͏︊͏󠄏
        raise NotImplemented()

    def classify(self, features):
        """Classify a list of features.
        Predict the labels for each feature in features to its corresponding class
        Args:
            features (m x n): m examples with n features.
        Returns:
            A list of class labels.
        """
        # TODO: finish this.͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏︀͏󠄋͏︊͏󠄏
        raise NotImplemented()


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """
        
        mult = np.multiply(data, data)
        
        return mult + data

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return (max_sum, max_sum_index)

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        modified_data = np.sum(data[:100],axis=1)
        
        maximum = np.max(modified_data)
        
        idx = np.argmax(modified_data)
        
        return (maximum, idx)

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            Dictionary [(integer, number of occurrences), ...]
        """
        unique_dict = {}
        flattened = data.flatten()
        for item in flattened:
            if item > 0:
                if item in unique_dict:
                    unique_dict[item] += 1
                else:
                    unique_dict[item] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            Dictionary [(integer, number of occurrences), ...]
        """
        flattened = data.flatten()
        unique, counts = np.unique(flattened[flattened>0], return_counts=True)
        return dict(zip(unique, counts)).items()


    def non_vectorized_glue(self, data, vector, dimension='c'):
        """Element wise array arithmetic with loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
        """
        if dimension == 'c' and len(vector) == data.shape[0]:
            non_vectorized = np.ones((data.shape[0],data.shape[1]+1), dtype=float)
            non_vectorized[:, -1] *= vector
        elif dimension == 'r' and len(vector) == data.shape[1]:
            non_vectorized = np.ones((data.shape[0]+1,data.shape[1]), dtype=float)
            non_vectorized[-1, :] *= vector
        else:
            raise ValueError('This parameter must be either c for column or r for row')
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row, col] = data[row, col]
        return non_vectorized

    def vectorized_glue(self, data, vector, dimension='c'):
        """Array arithmetic without loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
        """
        if dimension == 'c':
            vector = vector.reshape(-1,1)
            axis = 1
        else:
            vector = vector.reshape(1,-1)
            axis = 0
        vectorized = np.concatenate((data,vector),axis=axis)
        return vectorized

    def non_vectorized_mask(self, data, threshold):
        """Element wise array evaluation with loops.
        This function takes a multi-dimensional array and then populates a new
        multi-dimensional array. If the value in data is below threshold it
        will be squared.
        Args:
            data: multi-dimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        non_vectorized = np.zeros_like(data, dtype=float)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                val = data[row, col]
                if val >= threshold:
                    non_vectorized[row, col] = val
                    continue
                non_vectorized[row, col] = val**2

        return non_vectorized

    def vectorized_mask(self, data, threshold):
        """Array evaluation without loops.
        This function takes a multi-dimensional array and then populates a new
        multi-dimensional array. If the value in data is below threshold it
        will be squared. You are required to use a binary mask for this problem
        Args:
            data: multi-dimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        vectorized = data.copy()
        vectorized[data < threshold] = data[data < threshold]**2
        return vectorized


def return_your_name():
    # return your name͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏︀͏󠄋͏︊͏󠄏
    return 'Jacob Blevins'
