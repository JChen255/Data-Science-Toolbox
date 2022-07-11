class Experiment:
    """An abstract class for evaluating algorithms and models."""

    def __init__(self,dataset,labels,classifiers):
        """Create a new experiment instance.

        Parameters
        -------------
        dataset: array
        labels: array
        classifiers: list

        Returns
        -------------
        Experiment object
        """
        self.dataset = dataset
        self.labels = labels
        self.classifiers = classifiers

        print("Experiment object has been created.")

    def runCrossVal(self, k=3):
        """Member function to run cross value on algorithms and models.

        Parameters
        -------------
        k:int
        predic:array

        Returns
        -------------
        Predictions: Array
        Cross value: float
        """
        self.predic = [] #list
        self.scores = []

        foldsize = int(len(self.dataset)/k) #int

        #split the dataset(features)
        split_features = [] #list
        split_features = [self.dataset[i:i + foldsize] for i in range(0, len(self.dataset), foldsize)] #list
        
        #split labels
        split_labels = [] #list
        for i in range(0,len(self.labels),foldsize):
            split_labels.append(self.labels[i:i+foldsize]) #list


        # run k_fold cross validation
        for c in self.classifiers:
            pred = []
            scor = []

            for i in range(len(split_features)):
                sf_copy = list(split_features) #list
                del sf_copy[i]
                train_x = sf_copy.pop(0) #numpy.ndarray
                for feature in sf_copy:
                    train_x = np.concatenate([train_x, feature])
                test_x = split_features[i]#numpy.ndarray

                tt_copy = list(split_labels) #list
                del tt_copy[i]
                train_y = tt_copy.pop(0)
                for label in tt_copy:
                    train_y = np.concatenate([train_y,label])
                test_y = split_labels[i]

                c.train(train_x,train_y)
                pre,sco = c.test(test_x,test_y)   
                pred.append(pre)
                scor.append(sco)
            self.predic.append(pred)
            self.scores.append(scor)

        return self.predic,self.scores
            

    def score(self, width = 61):
        """Member function to calculate accuracy on algorithms and models.

        Parameters
        -------------
        None

        Returns
        -------------
        Accuracy: float
        """
        scores = []
        
        for i in range(len(self.scores)):
            total = 0
            for j in range(len(self.scores[i])): 
                total += self.scores[i][j]
            score = total/len(self.scores[i])
            scores.append(score)

        width_single = (width-3)//2

        t1 = "^" + str(width_single) 
        t2 = "^" + str(width_single) + ".2f"
        print("+"+"-"*(width-2)+"+")
        print("|"+format("Classifiers", t1) + "|" + format("Scores", t1) + "|")
        for i in range(len(self.classifiers)):
            print("|"+"-"*(width-2)+"|")
            s= str(type(self.classifiers[i]))[19:-2]
            print("|"+format(s, t1) + "|" + format(scores[i], t2) + "|")
        print("+"+"-"*(width-2)+"+")
 

    def _confusionMatrix(self):
        """Private function to calculate confusion Matrix on algorithms and models.

        Parameters
        -------------
        None

        Returns
        -------------
        Confusion Matrix: float
        """
        for i in range(len(self.classifiers)):
            all_predictions=[]
            for j in self.predic[i]:
                for item in j:
                    all_predictions.append(item)
            p_copy = list(all_predictions)

            classes = np.unique(np.array(p_copy))
            n = len(classes)
            array = np.zeros((n,n))

            for j in range(len(self.labels)):
                c_idx = np.where(classes == self.labels[j])
                r_idx = np.where(classes == all_predictions[j])
                array[c_idx,r_idx] += 1

            
        t1 = "^" + str(20) 

        print("+"+"-"* ((n+1)*20+n) +"+")
        string = "|"+format("", t1)+"|" 
        for i in range(n):
            string += (format(classes[i], t1) + "|")

        print(string)
        for i in range(n):
            string = "|"+format(classes[i], t1) + "|"
            for j in range(n):
                string += format(array[i][j], t1) + "|"
            print(string)

        print("+"+"-"* ((n+1)*20+n) +"+")
