class Tree():
    """An abstract parent class for Decision Tree"""
    def __init__(self, array):
        """Create a new tree instance.

        Parameters
        -------------
        array: array

        Returns
        -------------
        Tree object
        """
        if isinstance(array,list):
            n = len(array)
            if (n==0):
                print("TreeCreationWrong: No nodes!")
                return
            self.root = array[0]

            self.children = []
            for i in range(1,n):
                self.children.append(Tree(array[i]))
        else:
            self.root = array
            self.children = []

    def tostring(self):
        """Member function to print the tree.

        Parameters
        -------------

        Returns
        -------------
        string
        """
        string = '[' + str(self.root)
        if len(self.children)>0:
            for i in range(len(self.children)):
                string += self.children[i].tostring()
        string += ']'
        return string

    def __str__(self):
        return self.tostring()

class ClassifierAlgorithm:
    """An abstract parent class for different algorithms
     Its subclasses include:
     simplekNNClassifier class -- simple kNN Classifier algorithm
    """

    def __init__(self):
        """Create a new classifier algorithm instance.
        train_x  set a default value None to the train_x
        train_y set a default value None to the train_y

        Parameters
        -------------
        train_x: array
        train_y: array

        Returns
        -------------
        Classifier algorithm object
        """
        self.train_x = None
        self.train_y = None
        print("Classifieralgorithm object has been created.")

    def train(self,train_x,train_y):
        """Member function to train classifier.

        Parameters
        -------------
        train_x: array
        train_y: array

        Returns
        -------------
        None
        """
        if len(train_x) != len(train_y):
            raise ValueError("The length of features and labels must be the same.")

        self.train_x = train_x
        self.train_y = train_y
        print("train function has been invoked.")

    def test(self,test_x,test_y=None):
        """Member function to test classifier.
        test_y set a default value None to test_y
        prediction set a default value None to prediction

        Parameters
        -------------
        test_x: array
        test_y: array

        Returns
        -------------
        None
        """
        self.test_x = test_x
        self.test_y = test_y
        self.prediction = None
        print("test function has been invoked.")

class simplekNNClassifier(ClassifierAlgorithm):
    """Create a new simplekNNClassifier instance;
    inherits attributes and member functions from parent class ClassifierAlgorithm.

    Parameters
    -------------
    filename: string

    Returns
    -------------
    simplekNNClassifier object
    """
    def __init__(self):
        super().__init__()
        print("simplekNNClassifier object has been created.")

    def test(self,test_x, test_y,k=3):
        """Overriden from parent class; member function to train and test classifier, then make prediction.
        k set a default value int 3 to k

        Parameters
        -------------
        test_x: array
        k: int

        Returns
        -------------
        Predictions: array
        Accuracy Score: float
        """
        # train_vectors: train_x; train_labels = train_y
        # test_vectors: test_x; predictions = []
        # calculate distance between train_x and test_x, find k smallest distance, find the label.
        
        self.test_x = test_x #array
        self.test_y = test_y 
        self.predictions = [] #list
      

        for item in self.test_x: 
            dist = [] #list
            for i in range(len(self.train_x)):
                p1 = np.array(self.train_x)[i,:] #array
                p2 = item #array
                distance = np.sqrt(np.sum((p1-p2)**2))
                dist.append(distance) #list
            dist = np.array(dist) #array
            n_dist = np.argsort(dist)[:k]
            labels = []
            for j in range(len(n_dist)):
                labels.append(self.train_y[n_dist[j]])

            c = Counter(labels)
            mode = [k for k, v in c.items() if v == c.most_common(1)[0][1]]
            self.predictions.append(mode[0])

        correct = 0
        for i in range(len(self.predictions)):
            if self.predictions[i] == self.test_y[i]:
                correct += 1 
        self.score = float(correct/len(self.test_y))

        return self.predictions, self.score

class DecisionTree(ClassifierAlgorithm, Tree):
    """Create a new DecisionTree instance;
    inherits attributes and member functions from parent class ClassifierAlgorithm
    and class Tree.

    Parameters
    -------------
    feature_names: list

    Returns
    -------------
    DecisionTree object
    """
    def __init__(self, feature_names):
        super().__init__()
        self.feature_names = feature_names
        print("decisionTree object has been created.")


    def InfoEntropy(self,data_y):
        """Member function to calculate information entropy.

        Parameters
        -------------
        data_y: array

        Returns
        -------------
        float
        """
        n = len(data_y)
        # record the number of labels
        labels = Counter(data_y[:,-1])
        entropy = 0.0
        # calculate the information entropy
        for k, v in labels.items():
            prob = v / n
            entropy -= prob * math.log(prob, 2)
        return entropy

    def SplitDataset(self, data_x, data_y, idx):
        """Member function to split dataset.

        Parameters
        -------------
        data_x: array
        data_y: array
        idx: int

        Returns
        -------------
        array
        """

        # idx is the feature's idx
        splitData_x = defaultdict(list)
        splitData_y = defaultdict(list)
 
        for x, y in zip(data_x,data_y):
            splitData_x[x[idx]].append(np.delete(x, idx))
            splitData_y[x[idx]].append(y)

        for k,v in splitData_x.items():
            splitData_x[k] = np.array(splitData_x[k])
            splitData_y[k] = np.array(splitData_y[k])
        return splitData_x.keys(), splitData_x.values(), splitData_y.values()

    def ChooseFeature(self,data_x, data_y):
        """Member function to choose feature to split the dataset.

        Parameters
        -------------
        train_x: array
        train_y: array

        Returns
        -------------
        int
        """
        n = len(data_x[0])
        m = len(data_x)
        # entropy before splitting
        entropy = self.InfoEntropy(data_y)
        bestGain = 0.0
        feature = -1
        for i in range(n):
           # splitting based on i
            split_data_x = self.SplitDataset(data_x,data_y, i)[1]
            split_data_y = self.SplitDataset(data_x,data_y, i)[2]
            new_entropy = 0.0
            # entropy after aplitting
            for data in split_data_y:
                prob = len(data) / m
                new_entropy += prob * self.InfoEntropy(data)
            # calculate information gain
            gain = entropy - new_entropy
            if gain > bestGain:
                bestGain = gain
                feature = i
        return feature

    def train(self, train_x, train_y):
        """Member function to train classifier.

        Parameters
        -------------
        train_x: array
        train_y: array

        Returns
        -------------
        dictionary, array
        """
        train_x = np.array(train_x)
        train_y = np.array(train_y)

        counter = Counter(train_y[:,-1])
        # return if there is only one class in the dataset
        if len(counter) == 1:
            return train_y[0,-1],train_y[0,-1]
    
        # return if all the features are used
        if len(train_x[0]) == 0:
            most = counter.most_common(1)[0][0]
            return most, most
    
        # find the best feature to split
        fidx = self.ChooseFeature(train_x,train_y)
        fnames = self.feature_names[:]
        fname = self.feature_names[fidx]
        self.feature_names.remove(fname)

        node = {fname: {}}
        tree = [fname]
    
        # recursion
        vals, split_data_x, split_data_y = self.SplitDataset(train_x, train_y, fidx)
        for val, data_x, data_y in zip(vals, split_data_x, split_data_y):
            node[fname][val] = self.train(data_x,data_y)[0]
            tree.append([fname+"="+str(val),self.train(data_x,data_y)[1]])

        self.feature_names = fnames[:]
        self.node = node
        self.tree = tree
        return node,tree

    def classify(self, data):
        """Member function to classify the labels.

        Parameters
        -------------
        data: array

        Returns
        -------------
        string
        """
        node = self.node.copy()
        key = list(self.node.keys())[0]
        self.node = self.node[key]
        idx = self.feature_names.index(key)
    
        # recursion
        pred = None
        for key in self.node:
           # find the branch
            if data[idx] == key:
                # if there is still subtree left, continue the recurstion
                if isinstance(self.node[key], dict):
                    self.node = self.node[key]
                    pred = self.classify(data)
                else:
                    pred = self.node[key]
                
            # if there is no branch, return any branch
            if pred is None:
                for key in self.node:
                    if not isinstance(self.node[key], dict):
                        pred = self.node[key]
                        break
        self.node = node.copy()
        return pred

    def test(self, data_x, data_y):
        """Member function to test classifier.

        Parameters
        -------------
        train_x: array
        train_y: array

        Returns
        -------------
        list,float
        """
        predictions = []
        correct = 0.0
        for i in range(len(data_x)):
            pred = self.classify(data_x[i])
            predictions.append(pred)
            if pred == data_y[i][0]:
                correct += 1
        score = float(correct/len(data_y))
        return predictions,score
    
    def __str__(self):
        """Member function overriden from parent class Tree.

        Parameters
        -------------

        Returns
        -------------
        string
        """
        tree1 = Tree(self.tree)
        return tree1.tostring()

class mergesort:   
    """Helper class to kdTreeKNNClassifier class to build kdTree.

    Parameters
    -------------

    Returns
    -------------
    mergesort object
    """   
    def __init__(self, a, axis):
        self.a = a
        self.axis = axis
        self.mgsort(0, len(a) - 1)

    def merge(self, l, mid, r):
        aux = self.a[l:r + 1].copy()

        i, j = l, mid + 1
        for k in range(l, r + 1):
            if i > mid:
                self.a[k] = aux[j - l]
                j += 1
            elif j > r:
                self.a[k] = aux[i - l]
                i += 1
            elif aux[i - l, self.axis] < aux[j - l, self.axis]:
                self.a[k] = aux[i - l]
                i += 1
            else:
                self.a[k] = aux[j - l]
                j += 1

    def mgsort(self, l, r):
        if l >= r:
            return
        mid = int((r + l) / 2)
        self.mgsort(l, mid)
        self.mgsort(mid + 1, r)
        self.merge(l, mid, r)
        return

class node:
    """Helper class to kdTreeKNNClassifier class to build kdTree.

    Parameters
    -------------

    Returns
    -------------
    mergesort object
    """   
    def __init__(self, val, dep):
        self.val = val[:-1]
        self.kind = val[-1]
        self.dep = dep
        self.lchild = None
        self.rchild = None

class kdTreeKNNClassifier(ClassifierAlgorithm):
    """Create a new kdTreeKNNClassifier instance;
    inherits attributes and member functions from parent class ClassifierAlgorithm.

    Parameters
    -------------

    Returns
    -------------
    kdTreeKNNClassifier object
    """
    def __init__(self, k=3,p=2):
        super().__init__()
        self.k = k
        self._p = p
        self.KDtree = None
        print("kdTreeKNNClassifier object has been created.")

    def train(self,x_train, y_train):
        x_train = np.array(x_train)
        y_train = np.array(y_train).reshape(-1,1)
        self.n, self.dim = x_train.shape
        self.a = np.hstack((x_train,y_train))
        self.KDtree = self.build(0,self.n-1,0)
        print("Train function has been invoked.")

    def build(self,l,r,dep):    # build kdTree
        if l > r:
            return
        mid = (l+r)//2
        idx = dep % self.dim

        self.a[l:r+1] = mergesort(self.a[l:r+1],axis=idx).a

        newnode = node(self.a[mid],dep)
        newnode.lchild = self.build(l, mid-1, dep+1)    # recursion to build kdTree
        newnode.rchild = self.build(mid+1, r, dep+1)
        return newnode

    def search(self,t,k=None): # search k-nearest neighbors
        if k is not None:
            self.k = k
        nearest = np.array([[float('inf'),None] for _ in range(self.k)])
        node_lst = deque()
        node = self.KDtree
        while node:
            node_lst.appendleft(node)
            dim = node.dep % self.dim
            if t[dim] <= node.val[dim]:
                node = node.lchild
            else:
                node = node.rchild
        while len(node_lst) > 0:
            node = node_lst.popleft()
            dist = np.sum(np.abs(t-node.val)**self._p)**(1/self._p)
            idx_arr = np.where(dist<nearest[:,0])[0]
            if idx_arr.size > 0:
                nearest = np.insert(nearest,idx_arr[0],[dist,node],axis=0)[:self.k]

            r = nearest[:,0][self.k-1]
            dim_dist = t[node.dep%self.dim]-node.val[node.dep%self.dim]
            if r>abs(dim_dist):
                append_node = node.rchild if dim_dist < 0 else node.lchild
                if append_node is not None:
                    node_lst.append(append_node)
        return np.array([n[1].kind for n in nearest])

    def num_max(self,x):
        dic = Counter(x)
        return max(dic,key=dic.get)

    def test(self,x_test,y_test,k=None):
        x_test = np.array(x_test).reshape(-1,self.dim)
        predictions = []
        count = 0
        for i in range(len(x_test)):
            predic = self.num_max(self.search(x_test[i],k))
            predictions.append(predic)
        for j in range(len(predictions)):
            if predictions[j] == y_test[j]:
                count += 1
        score = count/len(y_test)

        return predictions,score
