import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from scipy.cluster.vq import vq
from sklearn.decomposition import PCA


class Undersampling:
    ds = 'adult.data'
    df_original = None
    df = None
    X = None
    Y = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    kmMethod = 'No undersampling'
    k = 10
    kmeans = None
    n_clusters = 10
    classificator = None
    cfParam = 10
    ct = 'dtree'
    predictions = None
    minClass = None
    maxClass = None
    usD = None #undersampled dataframe
    reducedSet = 0


    def __init__(self, file='adult.data', classificatorString='dtree', kmMethod='No undersampling', k=10, reducedSet=0, cfParam=10):
        self.ds = file
        self.ct = classificatorString
        self.cfParam = cfParam
        self.kmMethod = kmMethod
        self.k = k
        self.reducedSet = reducedSet

        if file == 'adult.data':
            self.df = pd.read_csv('datasets/'+file, delimiter=', ', engine='python', 
                                  names = ['age', 'workclass', 'fnlwgtm', 'education', 'education-num', 
                                             'martial-status', 'occupation', 'relationship', 'race', 'sex', 
                                             'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'y'])
        elif file == 'bank.csv':
            self.df = pd.read_csv('datasets/'+file, delimiter=';')
        elif self.ds == 'yeast1.dat':
            self.df = pd.read_csv('datasets/'+file, delimiter=', ', engine='python',
                                  names = ['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc', 'y'])

        self.df_original = self.df.copy(deep=True)
        self.transformColumnsToNumeric()

        if self.reducedSet == 1:
            self.reduceDataSet()

        self.splitMinMaxClasses()

        self.initTrainTest()

        self.trainClassificator()
 

    def trainClassificator(self):
        if self.ct == 'dtree':
            self.classificator = DecisionTreeClassifier(max_depth = self.cfParam).fit(self.X_train, self.y_train)
        elif self.ct == 'rfc':
            self.classificator = RandomForestClassifier(n_estimators = self.cfParam, random_state = 0).fit(self.X_train, 
                                                                                                 self.y_train)
        elif self.ct == 'knn':
            self.classificator = KNeighborsClassifier(n_neighbors = self.cfParam).fit(self.X_train, self.y_train)
        
        self.prediction = self.classificator.predict(self.X_test)
        

    def transformColumnsToNumeric(self):
        le = preprocessing.LabelEncoder()

        if self.ds == 'adult.data':
            del self.df["education"]
            del self.df["fnlwgtm"]
            columns = ["workclass", "martial-status", "occupation", "relationship", 
                       "race", "sex", "native-country"]
            self.df['y'].replace({"<=50K" : 0, ">50K" : 1}, inplace=True)
        elif self.ds == 'bank.csv':
            columns = ["job", "marital", "education", "default", "housing", "loan", 
                       "contact", "month", "day_of_week", "poutcome"]
            self.df['y'].replace({"no" : 0, "yes" : 1}, inplace=True)
        elif self.ds == 'yeast1.dat':
            columns = []
            self.df['y'].replace({"negative" : 0, "positive" : 1}, inplace=True)
        else:
            columns = []

        for col in columns:
            wheather_encoded=le.fit_transform(self.df[col])
            self.df[col] = wheather_encoded


    def initTrainTest(self):

        #centroids of each cluster
        if self.kmMethod == 'centroids':
            self.n_clusters = len(self.minClass.index)
            self.kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++')
            label = self.kmeans.fit_predict(self.maxClass)
            centroids = self.kmeans.cluster_centers_

            self.usD = pd.DataFrame(centroids, columns = self.df.columns)
            self.usD = self.usD.append(self.minClass)

        #random samples for each cluster
        elif self.kmMethod == 'random_sampling':
            self.n_clusters = self.k
            self.kmeans = KMeans(n_clusters=self.k, init='k-means++')
            label = self.kmeans.fit_predict(self.maxClass)

            nM = round(len(self.minClass.index)/self.k)

            self.usD = self.minClass

            u_label = np.unique(label)

            oh = 0
            while True:
                for i in u_label:
                    if len(self.maxClass[label == i].index) >= nM+oh:
                        self.usD = self.usD.append(self.maxClass[label == i].sample(n=nM+oh, random_state = 0))
                        oh = 0
                    else:
                        self.usD = self.usD.append(self.maxClass[label == i])
                        oh = oh+nM-len(self.maxClass[label == i].index)
                        u_label = np.setdiff1d(u_label, np.array([i]))
                if oh == 0:
                    break

        #top nearest neighbor to centroids
        elif self.kmMethod == 'top1':
            self.n_clusters= len(self.minClass.index)
            self.kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++')
            label = self.kmeans.fit_predict(self.maxClass)
            centroids = self.kmeans.cluster_centers_
            
            closest, distances = vq(centroids, self.maxClass)
            self.usD = self.maxClass.iloc[closest]
            self.usD = self.usD.append(self.minClass)

        #top N nearest neighbor for each centroid
        elif self.kmMethod == 'topN':
            self.n_clusters = self.k
            self.kmeans = KMeans(n_clusters=self.k, init='k-means++')
            label = self.kmeans.fit_predict(self.maxClass)
            centroids = self.kmeans.cluster_centers_
            
            nM = round(len(self.minClass.index)/self.k)
            
            cTemp = self.maxClass
            self.usD = self.minClass
            
            for i in range(nM):
                closest, distances = vq(centroids, cTemp)
                self.usD = self.usD.append(cTemp.iloc[closest])
                cTemp = cTemp.drop(cTemp.index[closest.tolist()])

        #no undersampling
        else:
            self.usD = self.df


        self.X = self.usD.iloc[:,0:-1]
        self.Y = self.usD.iloc[:,-1]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.Y, 
                                                                                test_size=0.25, 
                                                                                random_state = 0)


    def splitMinMaxClasses(self):
        self.minClass = self.df.loc[self.df["y"] == 1]
        self.maxClass = self.df.loc[self.df["y"] == 0]


    def accuracy(self):
        return self.classificator.score(self.X_test, self.y_test)


    def precision(self):
        return precision_score(self.y_test, self.prediction,  average='binary', pos_label=1)


    def recall(self):
        return recall_score(self.y_test, self.prediction, average='binary', pos_label=1)


    def fMeasure(self):
        return f1_score(self.y_test, self.prediction, average='binary', pos_label=1)


    def auc(self):
        y_score = self.classificator.predict_proba(self.X_test)
        return roc_auc_score(self.y_test, y_score[:,1], multi_class='ovr')


    def printScores(self):
        print("-"*4, self.classificator, "on", self.ds, "-"*4)
        print("Accuracy: ", self.accuracy())
        print("Precision: ", self.precision())
        print("Recall: ", self.recall())
        print("F-measure: ", self.fMeasure())
        print("AUC: ", self.auc())
        print("-"*(14+len(str(self.classificator))+len(self.ds)))


    def printBarChart(self):
        bars = ['accuracy', 'precision', 'recall', 'f-measure', 'auc']
        stats = [self.accuracy()*100, self.precision()*100, self.recall()*100, self.fMeasure()*100, self.auc()*100]
        x_pos = [i for i, _ in enumerate(bars)]
        plt.bar(x_pos, stats, color=['tab:blue','tab:green','tab:red','tab:grey','tab:cyan'])
        plt.ylabel("Performance (%)")
        plt.title(str(self.classificator) + " on " + str(self.ds))
        
        for i, v in enumerate(stats):
            plt.text(i, v+2, " "+str(round(v,2)), ha='center')

        plt.xticks(x_pos, bars)
        plt.yticks([i for i in range(0,101,10)])
        
        if self.kmMethod == 0:
            plt.xlabel('No undersampling')
        else:
            plt.xlabel(self.kmMethod)

        plt.savefig('plots/'+self.ds+'_'+self.ct+'_'+self.kmMethod+'.png')
        
        #plt.show()
        return ['plots/'+str(self.ds)+'_'+str(self.ct)+'_'+str(self.kmMethod)+'.png']+stats

    #reduce size of data set to 20%
    def reduceDataSet(self):
        self.df = self.df.sample(n=round(len(self.df.index)*0.2), random_state = 0)

    #compute and plot all 4 kmeans++ methods at once
    def allKmMethods(self):
        kmeans_methods = ['No undersampling', 'centroids', 'random_sampling', 'top1', 'topN']
        accs = []
        precs = []
        recs = []
        fms = []
        aucs = []
        
        for km in kmeans_methods:
            self.kmMethod = km
            self.initTrainTest()
            self.trainClassificator()
            accs.append(round(self.accuracy()*100,2))
            precs.append(round(self.precision()*100,2))
            recs.append(round(self.recall()*100,2))
            fms.append(round(self.fMeasure()*100,2))
            aucs.append(round(self.auc()*100,2))

        x = np.arange(0,75,15)
        width = 2
        
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width*2-0.5, accs, width, label='Accuracy')
        rects2 = ax.bar(x - width-0.25, precs, width, label='Precision')
        rects3 = ax.bar(x, recs, width, label='Recall')
        rects4 = ax.bar(x + width+0.25, fms, width, label='f-Measure')
        rects5 = ax.bar(x + width*2+0.5, aucs, width, label='Auc')

        ax.set_ylabel('Performance (%)')
        ax.set_title(str(self.classificator) + " on " + str(self.ds))
        ax.set_xticks(x)
        kmeans_methods = ['no undersampling', 'centroids', 'random sampling', 'top1', 'topN']
        ax.set_xticklabels(kmeans_methods)
        ax.set_yticks([i for i in range(0,101,10)])
        ax.legend(loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.25))
        
        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        ax.bar_label(rects3, padding=3)
        ax.bar_label(rects4, padding=3)
        ax.bar_label(rects5, padding=3)
        
        fig.tight_layout()
        fig.set_size_inches(18.5, 5, forward=True)
        
        plt.savefig('plots/'+self.ds+'_'+self.ct+'_all.png')
        #plt.show()
        return ['plots/'+str(self.ds)+'_'+str(self.ct)+'_all.png',accs,precs,recs,fms,aucs]



    def performPCA(self, noU=1):
        features = list(self.df.columns)[:-2]
        X = self.df[features]
        if noU:
            y = self.df['y']
            colors = sns.color_palette(None, 2)
            iRange = [0, 1]
            
        else:
            y = self.kmeans.fit_predict(X)
            colors = sns.color_palette(None, self.n_clusters)
            iRange = range(self.n_clusters)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        plt.figure(figsize=(8, 8))
        for color, i in zip(colors, iRange):
            if noU:
                if i == 0:
                    label = 'Negative'
                else:
                    label = 'Positive'
            else:
                label = str(i)
            plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                    color=color, lw=2, label=label)
        plt.title('PCA of '+self.ds+' with '+self.kmMethod)
        plt.xlabel('Principal component 1')
        plt.ylabel('Principal component 2')
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.savefig('ds_visuals/'+self.ds+'_'+self.kmMethod+'_pca.png')
        plt.show()


def computeParams(alg, ds):
    if alg == 'knn':
        k = 51
    else:
        k = 101
    acc = []
    pre = []
    rec = []
    fme = []
    auc = []
    opt = 0
    opt_k = 0

    for i in range(1,k):
        df = Undersampling(ds, alg, 'No undersampling', 10, 0, i)
        acc.append(df.accuracy())
        pre.append(df.precision())
        rec.append(df.recall())
        fme.append(df.fMeasure())
        auc.append(df.auc())
        avg = ((df.accuracy()+df.precision()+df.recall()+df.fMeasure()+df.auc())/5)
        if avg > opt:
            opt = avg
            opt_k = i


    plt.plot(range(1,k), acc, label='acc')
    plt.plot(range(1,k), pre, label='pre')
    plt.plot(range(1,k), rec, label='rec')
    plt.plot(range(1,k), fme, label='fme')
    plt.plot(range(1,k), auc, label='auc')
    plt.title(df.ct + " on " + str(df.ds))
    plt.ylabel('Percent [%]')
    plt.xlabel('k [optimal = '+str(opt_k)+']')
    plt.legend()
    plt.savefig('params/'+df.ct+'_'+df.ds+'_param.png')
    plt.clf()



def computeAllStats():
    algos = ['dtree', 'knn', 'rfc']
    datasets = ['adult.data', 'bank.csv', 'yeast1.dat']
    ks = [20,20,21,9,21,21,87,73,57] #optimal algo parameters

    i = 0
    for alg in algos:
        for ds in datasets:
            df = Undersampling(ds, alg, 'No undersampling', 10, 1, ks[i])
            df.allKmMethods()
            print("one done")
            i += 1
    print("All done")



algos = ['dtree', 'knn', 'rfc']
datasets = ['adult.data', 'bank.csv', 'yeast1.dat']

