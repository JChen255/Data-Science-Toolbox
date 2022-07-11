import numpy as np
import math
from collections import Counter,defaultdict,deque
from numpy.typing import _128Bit
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('stopwords')
stopwords.words('english')

class DataSet:
    """An abstract parent class for reading, loading and editing different types of dataset.
     Its subclasses include:
     TimeSeriesDataSet class -- TimeSeries Data
     TextDataSet class -- Text Data
     QuantDataSet class -- Quantitative Data
     QualDataSet class -- Qualitative Data
    """

    def __init__(self, filename = None, header = True):
        """Create a new dataset instance.
        filename  set a default value None to the filename
        header    set a default value True to the header of the file
        conetent  set a default value None to the content which will store the content of file

        Parameters
        -------------
        filename: string
        header: Boolean
        content: array

        Returns
        -------------
        Dataset object
        """
        self.content = None

        # when filename is not None, calling function _readFromCSV
        if filename:    
            self.filename = filename
            self._readFromCSV(filename,header)

        # when filename is None, calling function _load
        else:   
            self._load(header)
        

    def _readFromCSV(self,filename,header=True):
        """Private function to open and read CSV files.

        Parameters
        -------------
        filename: string
        header: Boolean
        panda_content: dataframe
        content: array

        Returns
        -------------
        None
        """
        # check if the file is in csv format
        if filename[-4:] .lower() != ".csv":
            raise NameError("It is not a csv file.")

        # run if the file has a header
        if header == True: 
            try:
                 # store content in dataframe format
                self.panda_content = pd.read_csv(filename, index_col=0)
                # convert the df to numpy array
                self.content = self.panda_content.to_numpy()    
            except FileNotFoundError:
                print("The file does not exist.")

        # run if the file does not have a header
        else:   
            try:
                self.panda_content = pd.read_csv(filename, index_col=False,header=None)
                self.content = self.panda_content.to_numpy()
            except FileNotFoundError:
                print("The file does not exist.")
       
    def _load(self,header=True):
        """Private function to open and read csv file when no filename is input

        Parameters
        -------------
        header: Boolean

        Returns
        -------------
        None
        """
        # ask users to input the filename
        filename = input('Enter the file name: ')   

        # run if the file has a header
        if header == True:  
            try:
                # store content in dataframe format
                self.panda_content = pd.read_csv(filename, index_col=0)   
                # convert the df to numpy array
                self.content = self.panda_content.to_numpy()    
            except FileNotFoundError:
                print("The file does not exist.")
        # run if the file does not have a header
        else:   
            try:
                self.panda_content = pd.read_csv(filename, index_col=False,header=None)
                self.content = self.panda_content.to_numpy()
            except FileNotFoundError:
                print("The file does not exist.")

    def clean(self):
        """Member function to apply data cleaning,
           will be implemented specifically in each child class.

        Parameters
        -------------
        None

        Returns
        -------------
        None
        """
        print('clean function is invoked.')


    def explore(self):
        """Member function to visualize data,
           will be implemented specifically in each child class.

        Parameters
        -------------
        None

        Returns
        -------------
        None
        """
        print('explore function is invoked.')

class QualDataSet(DataSet):
    """Subclass of DataSet class; designed for qualitative dataset."""
    def __init__(self,filename,header=True):
        """Create a new qualitative dataset instance;
        inherits attributes and member functions from parent class DataSet.

        Parameters
        -------------
        filename: string
        header: Boolean

        Returns
        -------------
        QualDataset object
        """
        super().__init__(filename,header)

        # for test
        print("QualDataSet has been created.\n")
        print("panda.content(dataframe):\n")
        print(self.panda_content)
        print("content(array):\n")
        print(self.content)

    def clean(self):
        """Overriden from parent class; 
           member function to fill missing values with the mode.

        Parameters
        -------------
        None

        Returns
        -------------
        None
        """
        # find missing values
        total = self.panda_content.isnull().sum().sort_values(ascending=False)
        print('\nTotal missing value before cleaning: \n',total)

        # data processing; fill missing values with mode
        self.panda_content['comments'] = self.panda_content['comments'].fillna('None')
        self.panda_content['state'] = self.panda_content['state'].fillna(self.panda_content['state'].mode()[0])
        self.panda_content['work_interfere'] = self.panda_content['work_interfere'].fillna(self.panda_content['work_interfere'].mode()[0])
        self.panda_content['anonymity'] = self.panda_content['anonymity'].fillna(self.panda_content['anonymity'].mode()[0])
        self.panda_content['self_employed'] = self.panda_content['self_employed'].fillna(self.panda_content['self_employed'].mode()[0])
        self.content = self.panda_content.to_numpy()

        # doubel check
        total = self.panda_content.isnull().sum().sort_values(ascending=False)
        print('\nTotal missing value after cleaning:\n ',total)

    def explore(self):
        """Overriden from parent class; 
           member function to visualize qualitative data in the file.

        Parameters
        -------------
        None

        Returns
        -------------
        None
        """

        # plot a frequency histogram of a specific column
        for i in range(len(self.panda_content.columns)):
            if self.panda_content.columns[i] == 'Age':
                age = i
        x1 = self.content[:,age]

        total = 0
        count = 0
        for i in range(len(x1)):
            try:         
                total += int(x1[i])
                x1[i] = int(x1[i])
                count += 1
            except:
                continue
        mean = total/count

        # fill None with the mean of the column
        for i in range(len(x1)):    
            if x1[i] == 'None':
                x1[i] = mean 

        plt.hist(x1, bins=800)
        plt.gca().set(title='Frequency Histogram', xlabel='Age', ylabel='Frequency')
        plt.show()

        # plot a pie chart of a specific column
        for j in range(len(self.panda_content.columns)):
            if self.panda_content.columns[j] == 'Country':
                country = j
        x2 = self.content[:,country]

        # use hashtable to calculate the number of each item
        dic = {}    
        for i in range(x2.shape[0]):
            dic[x2[i]] = dic.get(x2[i],0) + 1
        key_iterable, value_iterable = dic.keys(), dic.values()
        key_list, value_list = list(key_iterable), list(value_iterable)
        key_list=np.array(key_list)
        value_list=np.array(value_list)
        value_list.astype(int)

        # modify parameters of the pie chart
        plt.axis('equal') 
        porcent = 100.*value_list/value_list.sum()
        patches, texts = plt.pie(value_list, startangle=90, radius=1.2)
        labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(key_list, porcent)]
        sort_legend = True
        if sort_legend:
            patches,labels,dummy=zip(*sorted(zip(patches, labels, value_list), key=lambda x: x[2],reverse=True))
        plt.legend(patches, labels, loc='upper left', bbox_to_anchor=(-0.1, 1.),fontsize=8)
        plt.show()

class QuantDataSet(DataSet):
    """Subclass of DataSet class; designed for quantitative dataset."""
    def __init__(self,filename,header=True):
        """Create a new quantitative dataset instance;
        inherits attributes and member functions from parent class DataSet.

        Parameters
        -------------
        filename: string
        header: Boolean

        Returns
        -------------
        QuantDataset object
        """
        super().__init__(filename, header)

        # for test
        print("QuantDataSet has been created.\n")
        print("panda.content(dataframe):\n")
        print(self.panda_content)
        print("content(array):\n")
        print(self.content)


    def clean(self):
        """Overriden from parent class;
           member function to fill missing value with the mean.

        Parameters
        -------------
        None

        Returns
        -------------
        None
        """
        # find missing values
        total = self.panda_content.isnull().sum().sort_values(ascending=False)
        print('\nTotal missing value before cleaning: \n',total)

        # calculate the mean of each column
        col_mean = np.nanmean(self.content, axis=0)  
        # find missing values
        inds = np.where(np.isnan(self.content))  
        # fill missing value with the mean
        self.content[inds] = np.take(col_mean, inds[1])
        
        # doubel check
        array_sum = np.sum(self.content)
        array_has_nan = np.isnan(array_sum)
        if array_has_nan == False:
            result = "No"
        print('\nIs there any missing value after cleaning: ',result)

    def explore(self):
        """Overriden from parent class; 
           member function to visualize quantative data in the file.

        Parameters
        -------------
        None

        Returns
        -------------
        None
        """

        # visualize the mean of columns
        x1 = range(1, self.content.shape[0]+1, 1)
        c1 = np.nanmean(self.content, axis=1)
        c1.astype(int)
        plt.plot(x1,c1,color='orange')
        plt.xlabel("Products")
        plt.ylabel("Mean")
        plt.show()

        # visualize the minimum and maximum values in each row
        for i in range(len(self.panda_content.columns)):
            if self.panda_content.columns[i] == 'MIN':
                min_index = i
                max_index = i+1

        c2 = self.content[:,min_index]
        c2.astype(int)
        c3 = self.content[:,max_index]
        c3.astype(int)

        # modify parameters of the pie chart
        plt.scatter(x1,c2,s=5,marker='s',color='y',label='Minimum')
        plt.scatter(x1,c3,s=5,marker='o',color='g',label='Maximum')
        plt.xlabel("Products")
        plt.ylabel("Value")
        plt.legend(loc='best')
        plt.show()

class TextDataSet(DataSet):
    """Subclass of DataSet class; designed for text dataset."""
    def __init__(self,filename,header=True):
        """Create a new text dataset instance;
        inherits attributes and member functions from parent class DataSet.

        Parameters
        -------------
        filename: string
        header: Boolean

        Returns
        -------------
        TextDataset object
        """
        super().__init__(filename, header)

        # for test
        print("TextDataSet has been created.\n")
        print("panda.content(dataframe):\n")
        print(self.panda_content)
        print("content(array):\n")
        print(self.content)


    def clean(self):
        """Overriden from parent class; 
           member function to remove stop words.

        Parameters
        -------------
        None

        Returns
        -------------
        None
        """

        # initialize stop words installed from nltk
        en_stops = set(stopwords.words('english'))  
        index = 0

        # find the column of text
        for i in range(len(self.panda_content.columns)):
            if self.panda_content.columns[i] == 'text': 
                index = i

        # print the first row of original text
        print("\nOriginal text:\n",self.content[0,index])

        # extract text
        for i in range(self.content.shape[0]):  
            words = word_tokenize(self.content[i,index]) 
            lst = []
            # remove stop words
            for word in words:
                if word.lower() not in en_stops:   
                    lst.append(word)
            # detokenize
            self.content[i,index] =" ".join(lst)    

        # check the text after cleaning
        print("\nText after cleaning:\n",self.content[0,index])

    def explore(self):
        """Overriden from parent class; 
           member function to display content in the file.

        Parameters
        -------------
        None

        Returns
        -------------
        None
        """

        # visualize frequency histograms of specifical columns of the file
        c1,c2,c3 = self.content[:,6],self.content[:,7],self.content[:,8]
        c1.astype(int)
        c2.astype(int)
        c3.astype(int)

        plt.hist(c1, bins=80)
        plt.gca().set(title='Frequency Histogram', xlabel='Cool Score', ylabel='Frequency')
        plt.show()

        plt.hist(c2, bins=80)
        plt.gca().set(title='Frequency Histogram', xlabel='Useful Score', ylabel='Frequency')
        plt.show()

        plt.hist(c3, bins=80)
        plt.gca().set(title='Frequency Histogram', xlabel='Funny Score', ylabel='Frequency')
        plt.show()

class TimeSeriesDataSet(DataSet):
    """Subclass of DataSet class; designed for timeseries dataset."""
    def __init__(self,filename,header=True):
        """Create a new timeseries dataset instance;
        inherits attributes and member functions from parent class DataSet.

        Parameters
        -------------
        filename: string
        header: Booolean

        Returns
        -------------
        TimeSeriesDataset object
        """
        super().__init__(filename,header)

        # for test
        print("TimeSeriesDataSet has been created.\n")
        print("panda.content(dataframe):\n")
        print(self.panda_content)
        print("\ncontent(array):\n")
        print(self.content)

    def clean(self,s=1):
        """Overriden from parent class; 
           member function to run a median filter with optional parameter s which determine the filter size.

        Parameters
        -------------
        s: interger

        Returns
        -------------
        None
        """

        # print orginal data
        print("\nOriginal data:\n",self.content[0])

        array = self.content.copy()
        # iterate columns; division for test due to large data size
        for i in range(array.shape[0]//10000):  
            # iterate each row
            for j in range(s,array.shape[1]-s):  
                nums = []
                # slice elements based on filter size
                for k in range(j-s,j+s+1):  
                    nums.append(array[i,k])
                arr = np.array(nums)
                # calculate median
                median = np.median(arr)
                # run median filter
                self.content[i,j] = median  

        # print data after cleaning
        print("\nData after cleaning:\n",self.content[0])

    def explore(self): 
        """Overriden from parent class; 
           member function to visualize time series data in the file.

        Parameters
        -------------
        None

        Returns
        -------------
        None
        """
        
        # visualize the time series data in the first row
        x1 = range(self.content.shape[1])  
        y1 = self.content[0]
        plt.plot(x1,y1,color='g')
        plt.xlabel("Time")
        plt.show()

        # visualize the minimum and maximum value of each row
        mn = np.max(self.content,axis=1)
        mx = np.min(self.content,axis=1)
        x1 = range(1, self.content.shape[0]+1, 1)  
        plt.scatter(x1,mn,s=5,marker='s',color='y',label='Minimum')
        plt.scatter(x1,mx,s=5,marker='o',color='g',label='Maximum')
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend(loc='best')
        plt.show()
