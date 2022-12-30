# Section 0.1: Imports, Globals, File Load
from IPython.display import display, HTML

from enum import Enum

import logging
import uuid

import pandas as pd

import numpy as np

import seaborn as sb

import matplotlib.pyplot as mp
import matplotlib.cm as cm

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

#### GLOBALS ####
pd.options.mode.chained_assignment = None  # default='warn'

#### LIBRARY ####
class OutcomeDefinition(Enum):
    TP = 1
    TN = 2
    FP = 3
    FN = 4
    TPR = 5
    TNR = 6

class FairAnalysis:
    ####### Constructor ##########
    def __init__(self, df, id_column, sensitive_column, label_column, cluster_count=10, logging_level=logging.INFO):
        self.orig_df = df;
        self.id_col = id_column;
        self.label_col = label_column;
        self.sens_col = sensitive_column;
        
        self.cohort_col = "cluster_" + str(uuid.uuid4());
        self.column_categories = dict();
        self.hot_encoded_cols = dict();
        self.label_space = [];

        self.cohort_ct = cluster_count;
        self.logger = logging.getLogger()
        self.logger.setLevel(logging_level)
        
        self.corr_matrix = None;
        self.corr_df = None;
        self.corr_preprocess(False);
        self.df = None;
        self.standard_preprocess(True);
        
        self.cohorts = None;
        self.set_cohorts();
        
    ##############################
    
    ####### MAIN METHODS #########

    def feature_analysis(self):
        #pre-process, no hot-encode for coor matrix, hot-encode sens attr
        df = self.corr_df.copy();
        
        # coorelation matrix for label (choose max for any hot-encoded columns? avg?)
        # coorelation matrix for sensitive attirbute
        logging.debug("Creating coor matrix");

        corrmatrix = df.corr();
        corr = pd.DataFrame({
                            self.label_col: corrmatrix.loc[self.label_col].values.tolist(), 
                            }, 
                           index=corrmatrix.columns.values);

        for col_name in self.hot_encoded_cols[self.sens_col]:
            corr[col_name] = corrmatrix.loc[col_name].values.tolist();

        corr.drop(index=self.hot_encoded_cols[self.sens_col], inplace=True);   
        self.corr_matrix = corr;
        
        # displaying heatmap
        display(HTML("<h4>Coorelation matrix of feature space:</h4>"));
        mp.figure(figsize = (6,9)) 
        dataplot = sb.heatmap(corr, annot=True)
        mp.show();
        
        #pre-process standard for histogram
        display(HTML("<h4>Histogram of sensitive group:</h4>"));

        #histogram graph label rate per sensitive group
        mp.figure(figsize = (10,8))
        dataplot = self.orig_df[self.sens_col].hist();
        mp.show()

        #pre-process standard for histogram
        display(HTML("<h4>Histogram of label incidence by sensitive group:</h4>"));

        #histogram graph label rate per sensitive group
        mp.figure(figsize = (10,8)) 
        dataplot = self.orig_df[self.label_col].hist(by=self.orig_df[self.sens_col])
        mp.show()

        display(HTML("<h4>Percent label incidence by sensitive group:</h4>"));
        breakdown = dict();
        breakdown["Group"] = list(["All"])
        breakdown["Group"].extend(self.orig_df[self.sens_col].unique())
        
        for labelval in self.orig_df[self.label_col].unique():
            ID = '% {0} is {1}'.format(self.label_col, labelval)
            breakdown[ID] = list()
            breakdown[ID].append(self.orig_df[self.label_col].value_counts()[labelval] / self.orig_df[self.label_col].values.size)

            for group in self.orig_df[self.sens_col].unique():
                filtered_group = self.orig_df.loc[self.orig_df[self.sens_col] == group]
                breakdown[ID].append(filtered_group[self.label_col].value_counts()[labelval] / filtered_group[self.label_col].values.size)

        tmp = pd.DataFrame.from_dict(breakdown);
        tmp.set_index("Group", inplace=True);
        display(tmp);
        return;
    
    def train_base_models(self):
        #examing ability of ML to guess senstive attributes
        self.ML_train_sens();
        
        #train on dataset without sensitive attribute, look at disparaties in outcome
        self.ML_train_label();
        
    def label_model_analysis(self, outcome_definition: OutcomeDefinition, positive_value, expanded=False):
        self.label_cohort_disparate_impact(outcome_definition, positive_value, expanded);
        
    def test_feature_space_corrections(self, favored_group, outcome_definition: OutcomeDefinition, positive_value):
        
        modifiers_a = list();
        modifiers_b = list();
        
        #Option 1. drop features most strongly coorelated to sensitive groups
        #call fit_transform
        for i in range(3):
            modifier = dict();
            
            def tmp_func(X, y, count=i): 
                return self.drop_top_corr_sens(X, y, count);
            if i == 0:
                modifier["name"]="Original".format(i);
            else:
                modifier["name"]="DropTop{0}Corr".format(i);
            modifier["fit_transform"] = tmp_func;
            modifiers_a.append(modifier);
        
        #Option 2. PCA - treat sensitive relation as noise, reduce dimensionality
        #call fit_transform
        modifier = PCA();
        modifiers_b.append(modifier)
        
        #Option 3. Change class weight on models
        model_names=list(("GaussianNB", "AdaBoostClassifier", "AdaBoostClassifierBalanced", "RandomForestClassifier", "RandomForestClassifierBalanced"))
        models = list((GaussianNB(), AdaBoostClassifier(), AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, class_weight="balanced")), RandomForestClassifier(), RandomForestClassifier(class_weight='balanced')));
        
        scaler = MinMaxScaler();
        df = self.df.copy();
        X_orig = df.loc[:, ~df.columns.isin([self.id_col, self.sens_col, self.label_col])]
        X_orig.iloc[:] = scaler.fit_transform(X_orig, [])
        y = df.loc[:, df.columns == self.label_col].values.ravel()
        
        #list of avg acc and avg cohort DI
        results = dict();
        
        results["Index"] = list()
        results["Index"].append("Model Accuracy");
        results["Index"].append("Average Cohort DI");
        for group in self.column_categories[self.sens_col].categories:
            results["Index"].append("Average {0} Cohort DI".format(group));
        
        written = list();
        
        for index, model in enumerate(models):
            mod_name = model_names[index];
            for m1 in modifiers_a:
                subname = m1["name"];
                mod1_name = mod_name + ":" + m1["name"];
                X1 = m1["fit_transform"](X_orig, y);
                
                if (subname not in written):
                    written.append(subname);
                    X1.to_csv("outputs/{0}{1}.csv".format(self.sens_col, subname));
                
                logging.info("Fitting {0}...".format(mod1_name));
                acc, avg_cohort_dis = self.test_data_transforms(model, X1.values, y, favored_group, outcome_definition, positive_value);
                results[mod1_name] = list();
                results[mod1_name].append(acc);
                results[mod1_name].append( sum(avg_cohort_dis) / len(avg_cohort_dis) );
                results[mod1_name].extend(avg_cohort_dis);
                
                for m2 in modifiers_b:
                    subname = m1["name"] + ":" + type(m2).__name__;
                    mod2_name = mod1_name + ":" + type(m2).__name__;
                    X2 = m2.fit_transform(X1, y);
                    
                    if (subname not in written):
                        written.append(subname);
                        X2df = pd.DataFrame(X2, columns=m2.get_feature_names_out(m2.feature_names_in_))
                        X2df.to_csv("outputs/{0}{1}.csv".format(self.sens_col, subname));
                    
                    logging.info("Fitting {0}...".format(mod2_name));
                    acc, avg_cohort_dis = self.test_data_transforms(model, X2, y, favored_group, outcome_definition, positive_value);
                    results[mod2_name] = list();
                    results[mod2_name].append(acc);
                    results[mod2_name].append( sum(avg_cohort_dis) / len(avg_cohort_dis) );
                    results[mod2_name].extend(avg_cohort_dis);
                    
        display(HTML("<h3>Disparate Impact for Minority/Disfavored Groups Compared to {0} Group</h3>".format(favored_group)));
        
        tmp = pd.DataFrame.from_dict(results);
        tmp.set_index("Index", inplace=True);
        display(tmp);
        return;
    
    ##############################
    
    ####### DATA PROCESSING ######

    # initial preprocessing for analysis
    def standard_preprocess(self, hot_encode):
        df = self.orig_df.copy()

        # drop null rows
        logging.info("Started pre-processing training data...")    

        logging.info("Dropping all rows with null values...")
        delta = len(df.index);
        df.dropna(inplace=True);
        delta = delta - len(df.index);
        logging.info("{0} rows removed.".format(delta))
        self.label_space = df[self.label_col].unique();

        # per non-numeric attr
        # date -> datetime -> int
        # label -> categorical if non-numeric
        # sensitive -> categorical if non-numeric
        # hot-encode or categorical mapping
        logging.info("Categorizing string column data with up to 10 unique values...")
        logging.info("One-Hot-Encoding remaining string columns...")
        logging.info("Normalizing numeric data...")

        starting_col = df.columns;
        to_normalize = list()        
        for index, column in enumerate(starting_col):
            logging.debug("Col Type {0}".format(self.orig_df.dtypes[column]));
            if self.orig_df.dtypes[column] == object:
                if ("date" in column or column == "dob"):
                    #datetime to int
                    df[column] = pd.to_datetime(df[column])
                    df[column] = df[column].apply(lambda x: x.value)
                    to_normalize.append(column);

                elif (df[column].unique().size > 10 and hot_encode is True):
                    #hot-encode string data
                    logging.debug("Hot-encoding {0}".format(column));
                    
                    ct = ColumnTransformer(transformers=[('onehot',
                                                          OneHotEncoder(dtype='int', min_frequency=15, sparse=False), 
                                                          [column])],
                                           remainder='passthrough', verbose_feature_names_out=False).fit(df);
                    
                    df = pd.DataFrame(ct.transform(df));
                    df.columns = ct.get_feature_names_out();
                    
                    feat_output = ct.transformers_[0][1].categories_[0];
                    rename = dict();
                    rename_list = list();
                    for feat in feat_output:
                        rename_list.append("{0}_{1}".format(column, feat));
                        rename[feat] = "{0}_{1}".format(column, feat);
                    df.rename(columns=rename, inplace=True);
                    self.hot_encoded_cols[column] = rename_list;
                    continue;

                else:
                    #categorical data
                    logging.debug("Categorizing {0}".format(column));
                    df[column] = pd.Categorical(df[column])
                    self.column_categories[column] = df[column].cat;
                    df[column] = df[column].cat.codes
            elif column not in [self.id_col, self.label_col, self.sens_col]:
                to_normalize.append(column);

        logging.debug("Normalizing columns {0}".format(to_normalize));
        scaler = MinMaxScaler();

        df[to_normalize] = scaler.fit_transform(df[to_normalize]);
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col]);

        logging.info("Pre-Processing Complete.")
        self.df = df;
        
    # Initial preprocessing for correlation matrix analysis
    def corr_preprocess(self, hot_encode):
        df = self.orig_df.copy()

        # drop null rows
        logging.info("Started pre-processing feature analysis data...")   

        logging.info("Dropping all rows with null values...")
        delta = len(df.index);
        df.dropna(inplace=True);
        delta = delta - len(df.index);
        logging.info("{0} rows removed.".format(delta))
        self.label_space = df[self.label_col].unique();

        # per non-numeric attr
        # date -> datetime -> int
        # label -> categorical if non-numeric
        # sensitive -> categorical if non-numeric
        # hot-encode or categorical mapping
        logging.info("Categorizing string column data with up to 10 unique values...")
        logging.info("One-Hot-Encoding remaining string columns...")
        logging.info("Normalizing numeric data...")

        to_normalize = list()
        for index, column in enumerate(df.columns):
            logging.debug("Col Type {0}".format(self.orig_df.dtypes[column]));
            if self.orig_df.dtypes[column] == object:
                if ("date" in column or column == "dob"):
                    #datetime to int
                    df[column] = pd.to_datetime(df[column])
                    df[column] = df[column].apply(lambda x: x.value)
                    to_normalize.append(column);
                elif (column == self.sens_col or (df[column].unique().size > 10 and hot_encode is True)):
                    #hot-encode string data
                    logging.debug("Hot-encoding {0}".format(column));
                    
                    ct = ColumnTransformer(transformers=[('onehot',
                                                          OneHotEncoder(dtype='int', min_frequency=1, sparse=False), 
                                                          [column])],
                                           remainder='passthrough', verbose_feature_names_out=False).fit(df);
                    
                    df = pd.DataFrame(ct.transform(df));
                    df.columns = ct.get_feature_names_out();
                    
                    feat_output = ct.transformers_[0][1].categories_[0];
                    rename = dict();
                    rename_list = list();
                    for feat in feat_output:
                        rename_list.append("{0}_{1}".format(column, feat));
                        rename[feat] = "{0}_{1}".format(column, feat);
                    df.rename(columns=rename, inplace=True);
                    self.hot_encoded_cols[column] = rename_list;
                    continue;

                else:
                    #categorical data
                    logging.debug("Categorizing {0}".format(column));
                    df[column] = pd.Categorical(df[column])
                    self.column_categories[column] = df[column].cat;
                    df[column] = df[column].cat.codes
            elif column not in [self.id_col, self.label_col, self.sens_col]:
                to_normalize.append(column);

        logging.debug("Normalizing columns {0}".format(to_normalize));
        scaler = MinMaxScaler();

        df[to_normalize] = scaler.fit_transform(df[to_normalize]);
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col]);

        logging.info("Pre-Processing Complete.")
        self.corr_df = df;
    
    #removes top <count> columns correlated to sens attribute, y is unchanged
    def drop_top_corr_sens(self, X, y, count):
        if count == 0:
            return X;
        
        to_remove = set();
        corr_matrix = self.corr_matrix.copy();
        corr_matrix.drop(index=[self.id_col, self.label_col], inplace=True);
        corr_matrix.drop(columns=[self.label_col], inplace=True);
        
        corr_matrix = corr_matrix.abs();
        
        for i in range(count):
            batch = corr_matrix.idxmax().tolist();
            corr_matrix.drop(index=batch, inplace=True);
            
            for col in batch:
                if col in self.hot_encoded_cols:
                    batch.remove(col);
                    
                    full_list = self.hot_encoded_cols[col];
                    freq_list_only = list(set(full_list) & set(X.columns.tolist()))
                    batch.extend(freq_list_only);
                    
            to_remove.update(batch);

        return X.drop(columns=list(to_remove));
    
    ##############################
    
    ###### DATA ANALYSIS ######
        
    def get_confusion_matrix_stats(self, cm):
        stats = list()
        for i in range(len(cm)):    
            #check if no class instances exist in this set for row
            if (cm[i][:].sum() == 0 and cm.transpose()[i][:].sum() == 0):
                stats.append({
                    "TP": None,
                    "FP": None,
                    "FN": None,
                    "TN": None,
                    "TPR": None,
                    "TNR": None
                });
                continue;

            #predictied in i and in i
            TPcount = float(cm[i][i])
            #predicted in j but in i
            FNcount = float(cm[i][:].sum() - TPcount); #correct

            #predicted in i but in j
            FPcount = float(cm.transpose()[i][:].sum() - TPcount);

            #predictied not in i and not in i
            TNcount = float(cm.sum() - cm[i][:].sum() - cm.transpose()[i][:].sum() + cm[i][i])

            TPR = TPcount;
            if (TPR > 0.0):
                TPR = TPcount / (TPcount + FNcount);

            TNR = TNcount;
            if (TNR > 0.0):
                TNR = TNcount / (TNcount + FPcount);

            stat = {
                "TP": TPcount,
                "FP": FPcount,
                "FN": FNcount,
                "TN": TNcount,
                "TPR": TPR,
                "TNR": TNR
            }
            stats.append(stat);
        return stats;

    def disparate_impact(self, dis, adv, outcome_def: OutcomeDefinition, pos_value, stats):
        net_dis = list((0.0, 0.0));
        net_adv = list((0.0, 0.0));

        avg_cohort_di = list((0.0, 0.0));
        for c in range(self.cohort_ct):
            if (stats[c][dis] == None or stats[c][adv] == None):
                continue;
            if (stats[c][dis][pos_value][outcome_def.name] == None or stats[c][adv][pos_value][outcome_def.name] == None):
                continue;
            #positive outcome ct and total instance ct in cohort
            c_dis_pos_count = stats[c][dis][pos_value][outcome_def.name];
            c_dis_count = (stats[c][dis][pos_value]["TP"] + stats[c][dis][pos_value]["FP"] + 
                        stats[c][dis][pos_value]["TN"] + stats[c][dis][pos_value]["FN"]);

            c_adv_pos_count = stats[c][adv][pos_value][outcome_def.name];
            c_adv_count = (stats[c][adv][pos_value]["TP"] + stats[c][adv][pos_value]["FP"] + 
                        stats[c][adv][pos_value]["TN"] + stats[c][adv][pos_value]["FN"]);

            #net DI and cohort DI calculated
            net_dis[0] += c_dis_pos_count
            net_dis[1] += c_dis_count

            net_adv[0] += c_adv_pos_count
            net_adv[1] += c_adv_count


            num = (c_dis_pos_count/c_dis_count);
            den = (c_adv_pos_count/c_adv_count);
            if (den == 0):
                continue;
            avg_cohort_di[0] += num/den;
            avg_cohort_di[1] += 1;

        num = "-";
        if (net_dis[1] != 0):
            num = net_dis[0]/net_dis[1];
        
        den = "-"
        if (net_adv[1] != 0):
            den = (net_adv[0]/net_adv[1]);

        net_di = "-";
        if (den != 0 and num != "-" and den != "-"):
            net_di = num/den;

        avg_c_di = "-"
        if (avg_cohort_di[1] != 0):
            avg_c_di = (avg_cohort_di[0]/avg_cohort_di[1]);

        return net_di, avg_c_di;
    
    ############################
    
    ###### MODEL ANALYSIS ######

    #stratefied 10 fold cross validation model 
    def ten_fold_cv(self, X, y, model, cohort_sensitive_analysis=False):
        kf = StratifiedKFold(n_splits=10,shuffle=True,random_state=None)
        acc = 0.0
        y_pred = None;
        cm = None;
        cohort_cms = np.full((self.cohort_ct, len(self.column_categories[self.sens_col].categories)), None);

        for train_index, test_index in kf.split(X, y):
            #split data
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]

            model = model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = acc + accuracy_score(y_test, y_pred)

            if (cm is None):
                cm = confusion_matrix(y_test, y_pred, labels=np.unique(y));
            else:
                cm = np.add(cm, confusion_matrix(y_test, y_pred, labels=np.unique(y)))

            #if we have cohorts and sens labels mapped to indexes
            if (cohort_sensitive_analysis is True != type(None) and not self.cohorts.empty):
                yt_cohort_sens_labels = self.cohorts.iloc[test_index];
                yt_cohort_sens_labels[self.label_col] = y_test;

                yp_cohort_sens_labels = self.cohorts.iloc[test_index];
                yp_cohort_sens_labels[self.label_col] = y_pred;

                #per cohort group
                for cohort_index in range(self.cohort_ct):
                    #per sens group
                    for group_index in yt_cohort_sens_labels[self.sens_col].unique():                    
                        yt_group_cohort = yt_cohort_sens_labels[(yt_cohort_sens_labels[self.sens_col] == group_index) & 
                                                 (yt_cohort_sens_labels[self.cohort_col] == cohort_index)];
                        yp_group_cohort = yp_cohort_sens_labels[(yp_cohort_sens_labels[self.sens_col] == group_index) & 
                                                 (yp_cohort_sens_labels[self.cohort_col] == cohort_index)];


                        yt_group_cohort = yt_group_cohort.loc[:, yt_group_cohort.columns == self.label_col].values.ravel();
                        yp_group_cohort = yp_group_cohort.loc[:, yp_group_cohort.columns == self.label_col].values.ravel();

                        if (len(yt_group_cohort) == 0):
                            continue;

                        group_cohort_cm = confusion_matrix(yt_group_cohort, yp_group_cohort, labels=np.unique(y));
                        stats = self.get_confusion_matrix_stats(group_cohort_cm);
                        cohort_cms[cohort_index][group_index] = stats;

        return acc / 10, cm, cohort_cms;
    
    def find_best_model(self, label, exclude_from_X, cohort_sensitive_analysis=False):
        best = -1.0
        best_mod = None
        best_cm = None
        best_cm_cohorts = None;

        scaler = MinMaxScaler();
        df = self.df.copy();
        X = df.loc[:, ~df.columns.isin(exclude_from_X)].values
        X = scaler.fit_transform(X, [])
        y = df.loc[:, df.columns == label].values.ravel()    
        
        models = list((GaussianNB(), AdaBoostClassifier(), RandomForestClassifier()));
        
        for model in models:
            logging.info("Testing {0} to detect '{1}'...".format(type(model).__name__, label))
            acc, cm, cohort_cms = self.ten_fold_cv(X, y, model, cohort_sensitive_analysis) 
            logging.info("{0} score: {1}".format(type(model).__name__, acc))
            if (acc > best):
                best = acc
                best_mod = model
                best_cm = cm
                best_cm_cohorts = cohort_cms;

        return best, best_mod, best_cm, best_cm_cohorts;
    
    #test each transformed feature space, return acc and avg DI per group against favored
    def test_data_transforms(self, model, X, y, favored_group, outcome_def: OutcomeDefinition, positive_value):
        acc, cm, cohort_cms = self.ten_fold_cv(X, y, model, cohort_sensitive_analysis=True);
        avg_cohort_DIs = self.get_avg_cohort_DIs(cohort_cms, favored_group, outcome_def, positive_value);
        return acc, avg_cohort_DIs;
    
    #cluster like points of data
    def set_cohorts(self):    
        #pre-process
        df = self.df.copy()

        #remove id, label, and sensitive attr from cluster data
        #create 10 clusters
        X = df.loc[:, ~df.columns.isin([self.id_col, self.sens_col, self.label_col])].values
        df[self.cohort_col] = KMeans(n_clusters=self.cohort_ct, random_state=0).fit(X).labels_

        logging.info("Cohorts of similar points in data identified.")
        self.cohorts = df[[self.cohort_col, self.sens_col]];
        return;
    
    
    # Ability to determine sensitive attribute
    def ML_train_sens(self):
        best, best_mod, cm, cohort_cms = self.find_best_model(label=self.sens_col, 
                                             exclude_from_X=[self.id_col, self.sens_col, self.label_col]);

        display(HTML("<h4>Top Model Accuracy: {0}</h4>".format(best)));
        display(HTML("<h4>Top Model: {0}</h4>".format(type(best_mod).__name__)))
        display(HTML("<h4>True Positive & True Negative Rates per group:</h4>"))

        breakdown = dict();
        breakdown["Group"] = list();
        breakdown["True Positive Rate"] = list();
        breakdown["True Negative Rate"] = list();

        logging.debug("Confusion Matrix:")
        logging.debug(cm)
        
        stats = self.get_confusion_matrix_stats(cm);
        for i in range(len(cm)):
            group = self.column_categories[self.sens_col].categories[i]
            breakdown["Group"].append(group)
            breakdown["True Positive Rate"].append(stats[i]["TPR"])
            breakdown["True Negative Rate"].append(stats[i]["TNR"])

        tmp = pd.DataFrame.from_dict(breakdown);
        tmp.set_index("Group", inplace=True);
        display(tmp);
        return;
    
    # Train model on label with sensitive attribute removed, record data per cohort and group
    def ML_train_label(self):
        best, best_mod, cm, cohort_sens_stats = self.find_best_model(label=self.label_col, 
                                             exclude_from_X=[self.id_col, self.sens_col, self.label_col],
                                             cohort_sensitive_analysis=True);

        display(HTML("<h4>Top Model Accuracy: {0}</h4>".format(best)));
        display(HTML("<h4>Top Model: {0}</h4>".format(type(best_mod).__name__)))

        logging.debug("Confusion Matrix:")
        logging.debug(cm)

        self.cohort_sens_stats = cohort_sens_stats;
        return;
    
    def get_avg_cohort_DIs(self, specific_cohort_sens_stats, favored_group, outcome_def, positive_value):
        #Table of Disparate Impact (compare disadvantaged row i to advantaged group j)
        average_c_di = list();

        for i in range(len(self.column_categories[self.sens_col].categories)):
            # j is majority/favored group
            j = self.column_categories[self.sens_col].categories.to_list().index(favored_group); 
            net_di = 0.0;
            avg_di = 0.0;

            if i == j:
                net_di = 1.0;
                avg_di = 1.0;
            else:
                net_di, avg_di = self.disparate_impact(i, j, outcome_def, positive_value, specific_cohort_sens_stats);

            average_c_di.append(avg_di);

        return average_c_di;

    def label_cohort_disparate_impact(self, outcome_def: OutcomeDefinition, positive_value, expanded=False):
        aggregate_rates = np.zeros((len(self.column_categories[self.sens_col].categories), 2));

        breakdown = dict();
        breakdown["Group"] = list();
        breakdown["Avg. Cohort {0}".format(outcome_def.name)] = list();
        if expanded is True:
            for c_index in range(self.cohort_ct):
                breakdown["Cohort #{0} {1}".format(c_index, outcome_def.name)] = list();

        for c_index in range(self.cohort_ct):
            for g_index in range(len(self.column_categories[self.sens_col].categories)):

                if (self.cohort_sens_stats[c_index][g_index] != None and 
                    self.cohort_sens_stats[c_index][g_index][positive_value][outcome_def.name] != None):
                    aggregate_rates[g_index][0] += self.cohort_sens_stats[c_index][g_index][positive_value][outcome_def.name];
                    aggregate_rates[g_index][1] += 1;
                    if expanded is True:
                        breakdown["Cohort #{0} {1}".format(c_index, outcome_def.name)].append(
                            self.cohort_sens_stats[c_index][g_index][positive_value][outcome_def.name]
                        );
                elif expanded is True:
                    breakdown["Cohort #{0} {1}".format(c_index, outcome_def.name)].append(
                        '-'
                    );

        #Table of desired stat as positive outcome                 
        for g_index in range(len(self.column_categories[self.sens_col].categories)):
            breakdown["Group"].append(self.column_categories[self.sens_col].categories[g_index])
            avg_rate = 0.0;
            if (aggregate_rates[g_index][0] == 0):
                avg_rate = 0.0;
            else:
                avg_rate = aggregate_rates[g_index][0] / aggregate_rates[g_index][1];
            breakdown["Avg. Cohort {0}".format(outcome_def.name)].append(avg_rate);
        tmp = pd.DataFrame.from_dict(breakdown);
        tmp.set_index("Group", inplace=True);
        display(tmp);

        #Table of Disparate Impact (compare disadvantaged row i to advantaged group j)
        overall_di = dict();
        average_c_di = dict(); 

        overall_di["Index"] = list();
        average_c_di["Index"] = list();
        
        for i in range(len(self.column_categories[self.sens_col].categories)):
            overall_di["Maj. " + self.column_categories[self.sens_col].categories[i]] = list();
            average_c_di["Maj. " + self.column_categories[self.sens_col].categories[i]] = list();

        for i in range(len(self.column_categories[self.sens_col].categories)):
            overall_di["Index"].append("Min. " + self.column_categories[self.sens_col].categories[i]);
            average_c_di["Index"].append("Min. " + self.column_categories[self.sens_col].categories[i]);
            for j in range(len(self.column_categories[self.sens_col].categories)):
                net_di = 0.0;
                avg_di = 0.0;

                if i == j:
                    net_di = 1.0;
                    avg_di = 1.0;
                else:
                    net_di, avg_di = self.disparate_impact(i, j, outcome_def, positive_value, self.cohort_sens_stats);

                overall_di["Maj. " + self.column_categories[self.sens_col].categories[j]].append(net_di);
                average_c_di["Maj. " + self.column_categories[self.sens_col].categories[j]].append(avg_di);

        #Cohort Avg Disparate Impact
        display(HTML("<h3>Overall Disparate Impact for Minority/Disfavored Group Compared to Majority/Favored Group</h3>"))
        tmp = pd.DataFrame.from_dict(overall_di)
        tmp.set_index("Index", inplace=True)
        display(tmp)

        display(HTML("<h3>Average Cohort Disparate Impact for Minority/Disfavored Group Compared to Majority/Favored Group</h3>"))
        tmp = pd.DataFrame.from_dict(average_c_di)
        tmp.set_index("Index", inplace=True)
        display(tmp)

        return;

    #####################


