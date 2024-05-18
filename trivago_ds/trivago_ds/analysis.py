
#%%

import pandas as pd
import matplotlib.pyplot as plt
from plotnine import (ggplot, ggtitle, aes, geom_col, theme_dark, 
                      theme_light, scale_x_discrete,
                      position_dodge, geom_text, element_text, theme,
                      geom_histogram
                    )
import scipy.stats as stats
import pingouin as pg
import numpy as np
import ast
from bioinfokit.analys import stat as infostat
import seaborn as sns

import numpy as np
from scipy.stats import iqr
import plotly.express as px

#%%

"""
1.Using the data provided, please answer the following questions:
a)Are all locales impacted by the loss of data equally?
b)Which 5 path_id have the highest average hits in each locale? How about globally?



"""



#%%

"""
# Problem statement

The completeness and accuracy of data that is to be analyzed is one of major checks required to be undertaken 
during data quality assurance and verification. This ensures that the precision of results is not reduced by 
altered data attributes and incase that is the case steps can be taken to correct it or different methodology with 
assumptions that cater for quality is used.

To ensure this, trivago has identified that the data has been corrupted for a week which leds us to ask whether this 
influenced the completeness, distribution and frequency of data points among locales. This check will be important 
in case the data will be used to conduct various A/B testing. The relevance of this quality check and its impact on 
such an analysis is rather high. Forinstance, this will enable us to determine whether to impute missing data and how as 
well as whether to use a parametric or nonparametric hypothesis testing method.


# Reasearch question

A viable and valid solution to the problem will require responding to the question below;

1. Are all locales impacted by the loss of data equally?

To conceptualize the question is interpreted as the number of missing data in each locales is equal.


Responding to this question requires hypothesis testing which is formulated as follows;

Null hypothesis (H0): There is no significant difference in number of missing hits among all locales

Alternative hypothesis (H1): There is significant difference in number of missing hits for at least one locale



# Research objective

1. To determine whether all locales are equally impacted by missing data



# Data analysis

This data analysis is undertaken using pandas among other frameworks. First, the data is import and some descriptive 
statistical analysis are performed to understand the data. A function is then produce to count the number of missing data 
for hits variable for each locale. The algorithm for this is such that first the data points with missing values for hits variable are 
identified, then they are grouped by locale variable and the number of index counted for each locale. Counting the number 
of index in this context produces the missing hits data becauses the index variable uniquely identifies each data point.
The implementation is as follows;


"""







#%% import
df = pd.read_csv('DataScientistCaseStudy.csv')


#%%
df.describe()

#%% This shows the number of data points for each variable and the data type of the variable
df.info()

# From the index variable had 581,489 data points which depeicts the total number 
# of data points in the dataset. Hits had 387,652 data points. While the data type of 
# variables such as agent_id, entry_page, and traffic_type are depicted as integers, they are 
# are actualy nominal and categorical variables for that matter.

#%%
# Descriptive statistics for numeric variables is computed 
# The numeric variables in the dataset are 'session_duration', 'hits' and 
# 'hour_of_day' which is time


#%% total number of missing data points for hits variable

total_missing_hits = df['hits'].isnull().sum()

print(f'total missing hits is: {total_missing_hits}')
# number of missing values is 193837

#%% percentage of data points missing in hits
# Total missing hits data as a percentage of all data points is estimated as follows

total_data = df['index'].count()

percent_missing = (total_missing_hits/total_data) * 100

print(f'Percentage of data missing in hits: {round(percent_missing, 2)}%')

# As much as 33.33% which correspond to  193837 data points is missing for hits. 
# The next important analysis is to determine how this missing data is distributed across
# all locales. This is undertaken as follows:

#%%
def count_missing_hits_per_locale(data: pd.DataFrame, loc_colname: str, hits_colname: str = 'hits'):
    # select only data where hits is missing
    df_missing_hits = data[data[hits_colname].isnull()]
    
    # group the data based on locale and count number of observations in each locale
    locale_missing_hits = df_missing_hits.groupby(by=loc_colname)[['index']].count()
    
    # rename columns to reflect the computation
    locale_missing_data_renamed = locale_missing_hits.reset_index().rename(columns={'index': 'num_missing_hits'})
    
    return locale_missing_data_renamed


#%%

missing_hits_per_locale = count_missing_hits_per_locale(data=df, loc_colname='locale', hits_colname='hits')

missing_hits_per_locale

"""  
The result of number of missing hits data per locale shows that L3 has the highest
missing data (59,721) while L1 has the lowest missing data (5,882). The mere fact that the same 
number of missing data is not recorded for each locale does not necessarily mean that number of 
missing data is not equal in statistical terms. For the differences in the number of missing data 
in each locale, the question that arises is whether such difference is statistically significant 
to argue that all local are not impacted impacted equally by loss of data. To answer this question,
the formulated hypothesis is tested. Before that, the data loss per locale is visualized as follows.
"""

#%%

(ggplot(data=missing_hits_per_locale, mapping=aes(x='locale', y='num_missing_hits', fill='locale'))
 + geom_col(stat='identity', position='dodge') + theme_dark() 
 + ggtitle('Data loss per locale')
 )


#%%

# The hypothesis is tested using kriskal wallis test which is a non-parametric method used when comapring 
# the means of more than two groups. Welch anova is also used to as another approach to verify the results.



kruskal_test = pg.kruskal(data=missing_hits_per_locale, dv='num_missing_hits', between='locale')
kruskal_test

# The result of the kruskal wallis test indicates p-value to be 0.4158. Given that p-value is greater than 0.05,
# the null hypothesis fail to be rejected at 5% significance level. This means that there is no statistical 
# significant in the number of missing data in the various locales. The differences visualized in the bar plot 
# are likely to be the result of random chance.
# 
#%%
#pg.anova(data=missing_hits_per_locale, dv='num_missing_hits', between='locale')


#%%

welch_test = pg.welch_anova(data=missing_hits_per_locale, dv='num_missing_hits', between='locale')
welch_test

# The result from welch anova test indicates an uncorrected p-value of 1.0 which is greater than 0.05
# hence the null hypothesis fail to be reject. By this, difference in number of missing data in the 
# locales is not statistically significantly different from each other.



#%%  path_id Highest average hits in each locale
# b)Which 5 path_id have the highest average hits in each locale? How about globally?

#%%

# avg_hits_per_path_id_per_locale = (df.groupby(by=['locale', 'path_id_set'])[['hits']]
#                                    .mean().reset_index()
#                                    .sort_values(by=['locale','hits'], ascending=False)
#                                    )


#%%
class AverageHitsCalculator(object):
    def __init__(self, data: pd.DataFrame, level: str = 'local', 
                 pathid_colname: str = 'path_id_set'
                 ):
        self.data = data
        self.level = level
        self.pathid_colname = pathid_colname
        
    def compute_average_hits_per_path_id(self,  locale_colname: str = 'locale',
                                         pathid_colname: str = 'path_id_set',
                                         hits_colname: str = 'hits',
                                         ):
        self._locale_colname = locale_colname
        self.hits_colname = hits_colname
        if self.level == 'local':
            self.avg_hits_per_pathid = (self.data.groupby(by=[self._locale_colname, 
                                                              pathid_colname]
                                                          )[[hits_colname]]
                                                        .mean().
                                                        reset_index()
                                    )
        elif self.level == 'global':
            self.avg_hits_per_pathid = (self.data.groupby(by=pathid_colname)[hits_colname]
                                                    .mean()
                                                    .reset_index()
                                   )
            
        return self.avg_hits_per_pathid
            
    
    def get_selected_pathid_with_average_hits(self, hits_data: pd.DataFrame = None, 
                                              number_of_pathid_select: int = 5, 
                                              locale_name: str = None,
                                               order: str = 'highest'):
        if hits_data is not None:
            self.avg_hits_per_pathid = hits_data
            
        self.locale_name = locale_name
        self._number_of_pathid_select = number_of_pathid_select
        
        if self.level == 'local':
            locale_avg_hits = (self.avg_hits_per_pathid[self.avg_hits_per_pathid
                                                [self._locale_colname]== self.locale_name
                                            ]
                            )
            if order == 'highest':
                self.selected_pathid_with_avg_hits = (locale_avg_hits
                                                        .sort_values(by=self.hits_colname, 
                                                                    ascending=False
                                                                    )
                                                        .head(self._number_of_pathid_select)
                                
                                                )
                return self.selected_pathid_with_avg_hits
            elif order == 'lowest':
                self.selected_pathid_with_avg_hits = (locale_avg_hits
                                                        .sort_values(by=self.hits_colname, 
                                                                    ascending=True
                                                                    )
                                                    .head(self._number_of_pathid_select)
                                
                                                )
                return self.selected_pathid_with_avg_hits
                
            
        elif self.level == 'global':
            if order == 'highest':
                self.selected_pathid_with_avg_hits = (self.avg_hits_per_pathid
                                                        .sort_values(by=self.hits_colname,
                                                                    ascending=False
                                                                    )
                                                        .head(self._number_of_pathid_select)
                                                    )
                return self.selected_pathid_with_avg_hits
            elif order == 'lowest':
                self.selected_pathid_with_avg_hits = (self.avg_hits_per_pathid
                                                      .sort_values(by=self.hits_colname,
                                                                   ascending=True
                                                                   )
                                                      .head(self._number_of_pathid_select)
                                                    )
                return self.selected_pathid_with_avg_hits
    
    
    def plot_average_hits_per_pathid(self, data_to_plot: pd.DataFrame = None):
        if data_to_plot is not None:
            self.data_to_plot= data_to_plot
        self.data_to_plot = self.selected_pathid_with_avg_hits.copy()
        if self.level == 'local':
            title = (f'Top {self._number_of_pathid_select} Path_id_set with Highest average hits in {self.locale_name}'
                     )
        elif self.level == 'global':
            title = f'Top {self._number_of_pathid_select} Path_id_set with Highest average hits globally'
        
        dodge_text = position_dodge(width=0.9)
        self.graph = (ggplot(data=self.data_to_plot, mapping=aes(x=self.pathid_colname, 
                                                         y=self.hits_colname, 
                                                         fill=self.pathid_colname
                                                         )
                            )
                            + geom_col() + theme_dark() 
                            + ggtitle(title) 
                            + geom_text(aes(label='hits'),                                   
                                        position=dodge_text,
                                        size=8, va='bottom'
                                        )
                            + theme(axis_text_x=element_text(rotation=45, hjust=1))
                    
                    )
        return self.graph
    
    
#%% implementation of avg hits

avg_hits = AverageHitsCalculator(data=df)

#%%

avg_hits_per_pathid = avg_hits.compute_average_hits_per_path_id()

#%%

l1_highest_avg_hits = avg_hits.get_selected_pathid_with_average_hits(locale_name='L1')        
    
#%%

avg_hits.plot_average_hits_per_pathid()


#%%

"""
Because different online users visits different number of locations, the length of path_id_set 
varies per user and and the name of some path_id_set is rather long due to the many locations vistited.
In order to improve legibility of the visualization and labels, a function is written to shorten the 
names of the path_id_set to 25 characters. This is implemented below
"""


def shorten_pathid(data: pd.DataFrame, pathid_colname: str = 'path_id_set', path_id_length: int = 25):
    for i in range(0, len(data)):
        pathid_len = len(data[pathid_colname].unique()[i])
        if pathid_len <= path_id_length:
            path_id = data[pathid_colname].unique()[i]#[:path_id_length]  
        else:
            path_id = data[pathid_colname].unique()[i][:path_id_length] + '...'
        row_index = data.index[i]
        data[pathid_colname][row_index] = path_id
    return data
  
  
#%% l2

locale2_avg_hits = avg_hits.get_selected_pathid_with_average_hits(locale_name='L2')

#%%

locale2_pathid_short = shorten_pathid(data=locale2_avg_hits)

locale2_pathid_short

#%%

avg_hits.plot_average_hits_per_pathid(data_to_plot=locale2_pathid_short)

#%%  Identify path_id with highest avg hits for all locale

for locale_name in df['locale'].unique():
    locale_avg_hits = avg_hits.get_selected_pathid_with_average_hits(locale_name=locale_name)
    locale_pathid_short = shorten_pathid(data=locale_avg_hits, path_id_length=25)
    graph = avg_hits.plot_average_hits_per_pathid(data_to_plot=locale_pathid_short)
    print(graph)

#%% global top 5 path_id average hits

global_avg_hits = AverageHitsCalculator(data=df, level='global')

#%%
global_avg_hits_per_pathid = global_avg_hits.compute_average_hits_per_path_id()


global_highest_avg_hits_5 = global_avg_hits.get_selected_pathid_with_average_hits()
global_pathid_short = shorten_pathid(data=global_highest_avg_hits_5, path_id_length=25)
global_avg_hits.plot_average_hits_per_pathid(data_to_plot=global_pathid_short)




#%%  ##################################### Hits prediction  ######################################

"""
2.Note that the column “hits” has missing values. 
Use this data to build a model that predicts the number of hits per session, 
depending on the given parameters.
a)What other metrics can your model predict 
that can be useful?
b)What other columns would you like to have to improve your model?
c)Can your model predict the hits for tomorrow?
"""

#%% Problem statement
"""
# Problem statement

As a metasearch engine for accommodation, trivago offers online tech product that requires a clear undertsnading 
of how and the extent to which users interact with the trivago page. Among others, a key metric to understand this 
user behaviour which has potential for increasing conversion is the number of hits during a session.
Higher hits may signal that users are more engrossed in the product we are providing and spending time to explore  
more of it. This directly feeds into further product development. To move in that direction, there is the need 
to understand how various factors descriptive of user behaviour influence and more importantly can 
be leverage to predict hits. Not only is it assummed that features of user behaviour and platform used during a session 
are predictive of hits but also they can be transformed and engineered into more useful predictors. 
The absence of a tool and mechanism that enables trivago leverage available data to predict hits remains a business 
problem that will be tackled in this task. 

Informed by this, this task aims to predict hits during sessions. Thus, the number of hits is the target 
variable and others featurs of user behaiour, platform and device used are the predictor variables. In order to 
translate this understanding into a technical solution that can be communicated to non-technical 
audience later, there is the need to capture the prevailing challenge as a snapshot with a problem design framework. 

The Situation Complication Question (SCQ) framework was employed for that purpose. 
A key element is identifying the stakeholder and end-user of the solution to be provided. 
This will require some level of insider information about the various stakeholders who 
have identified hits prediction as their pain point. 

Hypothetically, the stakeholder is defined to be the product team with support 
from the data science team. The prodcut team managing the trivago page 
may be interested in knowing the number of 
hits a user will make during a session. This could be deduce by having features that are related 
to the product such as entry page and locations that were visited. The advertiser team may be interested 
in understanding how the source of channel such as search engine marketing influences hits and the 
importance of this infleuncing hits hence the inclusion of traffic type as a variable in the dataset.



The SCQ is depicted below;


"""


"""
The main focus of a data science task defines it goals, category of algorithm chosen and which objective to 
optimize and questions to address hence identified here. This task focuses on achieving good accuracy and precision 
for predicting hits rather than intrepretability of model. This influences how the research question and objective 
is designed.



Research Question

(I) How to we leverage available data to develop a hit prediction model with an accuracy better than random guessing? 

Worthy of notice is that, not only does this question reflect the focus of model precision but also techniques employed 
within the context of limited resouces such as computational power and time alloted for the task.


Research objective

(I) To develop a model that predicts hits with an accuracy better than the benchmark baseline model of random guessing





### Expected Results

A machine learning model that predicts number of hits is expected to be discussed in this report.

In developing the machine learning model, the end goal which is prediction accuracy and precision influenced the 
end-product of the modeling process. This assertion informed the 
methodology and techniques employed. 

With the problem statement and user requirement identified, the whole modeling process 
can be defined to include data exploration, model development, hyperparameters tuning and evaluation, packaging 
and deployment into production environment. For this task, focus is mainly on detailing data exploration and modeling and 
evaluation. Given the time constrain for the exercise, suggestions on improving the results are provided rather than exhausting 
all possible techniques. 



"""

"""
### Identifying variables in the data
In developing an algorithm for prediction, identifying the variables to use and
how to categorize them is important. The following were deduced;

#### Target / Response / Outcome variable
hits: discrete quantitative variable

#### Predictor / feature variables

• locale: Categorical

• day of week: Categorical

• hour of day: Discrete quantitative 

• agent_id: Categorical

• entry page: Categorical 

• path_id_set: Categorical 

• traffic type: Categorical

• Session duration: Discrete quantitative variable


By identifying the type of variable, appropriate visualization can be undertaken for different 
variables during exploratory data analysis. The type of variable also influences the transformation techniques to 
employ hence organized to reflect that.

"""
 

#%%
""" 

## Exploratory analysis 

An important aspect of modeling is the data scientist's intuition about the problem domain 
to complement
the statistical analysis and machine learning algorithms to be applied. This underlined some of the 
logic and decisions in key aspects of the analysis such as feature selection, feature engineering and 
data transformation among others. A highlight that informed such exploratory analysis is provided as follows.


In selecting features to be included as predictors in the model, the relevance of the feature in terms of 
likelihood of providing actionable insights and predictive signals were considered. 
On this basis, index as a variable 
was not included in the model because from the metadata, it is more of an identifier for each session but 
does not provide an information about the session that allows hits made during the session to be predicted.


"Locale" which is the platform used for the session, is likely to have an impact on the number of hits as  
it is possible that cetrain platforms are more user friendly may render trivago web pages better to 
capture the attention of users more and engage them more and longer and probably make more hits. An equally 
important feature that follows this intuition is "agent_id" which is the device used hence also selected 
to be included in the model. Session duration is likely to be positively related to hits and offer some 
predictive signals for it as it is expected that longer session duration will lead to more hits. 

While this intuitions offer some guide as to the variables that can be outrightly excluded from the model, the 
selection of many others need to assess through exploratory analysis. This is undertaken as follows;
"""



#%% ############    descriptive statistics  ########################

# Most of the descriptive statistics were undertaken while tackling the first question of hypothesis testing 
# hence only summarized here to show that they are also important for the modelling process. The interpretation
# made are also valid here.

#%%
df.info()
# A key insight gained from this is the presence of missing data for "path_id_set", session_duration and hits.
# hence we may have to handle missing data as part of the data preparation process


# Most descriptive statistics such mean, minimum maximum among others, highlight the range and 
# distribution of variables that quantitative. Hence quantitative variables are selected for this 
# type of analysis as follows.

#%% describe numeric features

df[['hour_of_day', 'session_duration', 'hits']].describe()

"""
From the analysis, it is deduced that outliers are likely to be present in session duration and hits.
Session duration has a high difference between the 75% percentile (285) and the maximum (85889). The 
same can be said about hits with 75% percentile being 78 and maximum being 4602. Domain knowledge 
will be used in deciding how missing data is handled.
"""


"""  

While the data type of 
variables such as agent_id, entry_page, and traffic_type are depicted as integers, they are 
are actualy nominal and categorical variables for that matter.
As identified earlier, variables such as agent_id, entry_page, and traffic_type are categorical variables 
but stored as integers. For these variables, estimating the mean and various percentiles as undertaken will 
be wrong given that their actual data type does not allow such computations. 
The count of these variables are already capture.

"""

#% ######## Data visualization to ascertain certain assumptions required by some models
"""
Various plots are used to visualize certain characteristics of the data 
to ascertain if certain assumptions are met for some models to be used.
Generally, parametric models like linear regression requires data to be 
normally distributed and a linear relationship to exist between the predictor 
and target variable (s).


To investigate this, histogram was used to visualize the distribution of data for 
continuous variables. This is implemented below.



"""

#%%

from plotnine import scale_y_log10, scale_x_log10
def plot_histogram(data: pd.DataFrame, variable_to_plot: str, 
                   title: str = None, bins_method: str = 'freedman-Diaconis'
                   ):
    data = data.dropna(subset=variable_to_plot).copy()
    
    # by default Freedman–Diaconis rule is computed to determine optimal number of bins for histogram
    
    if bins_method == 'freedman-Diaconis':
        h = (2 * iqr(np.array(df['hits'].dropna().values))) / (len(df['hits'])**(1/3))

        nbins = (df['hits'].max() - df['hits'].min()) / h
        nbins = round(nbins, 1)
    else:
        nbins = 5
        
    if title is None:
        title = f"Distribution of {variable_to_plot}"
    histogram = (ggplot(data, aes(x=variable_to_plot))
                + geom_histogram(bins=nbins) 
                #+ theme_dark() 
                + ggtitle(title) 
            #    + scale_x_discrete()
                #+ scale_y_log10()
                # + geom_text(aes(label='hits'),                                   
                #             position=dodge_text,
                #             size=8, va='bottom'
                #             )
                # + theme(axis_text_x=element_text(rotation=45, hjust=1))
            )
    return print(histogram)

#%% Plot of target variable

plot_histogram(data=df, variable_to_plot='hits')

# The plot shows that the distribution of hits is right skewed. For a large range of values, 
# using logarithmic scale to visualize usually improve the legibility of the graph but in this 
# case it also lead to distortion of the actual distribution shape hence not used.



#%% plot of quantitative predictor variables

quant_predictor_var = ['hour_of_day', 'session_duration']

#%%
for predictor in quant_predictor_var:
    plot_histogram(data=df, variable_to_plot=predictor)

# From the histogram, it is clear that session duration is right skewed
#%%

#%%
from plotnine import scale_x_sqrt, geom_density
(ggplot(df, aes(x='hour_of_day'))
                + geom_histogram() 
                #+ theme_dark() 
                + ggtitle('hour') 
                #+ geom_density()
               # + scale_x_log10()
                #+ scale_y_log10()
                # + geom_text(aes(label='hits'),                                   
                #             position=dodge_text,
                #             size=8, va='bottom'
                #             )
                # + theme(axis_text_x=element_text(rotation=45, hjust=1))
            )



#%%

# Even though hours of the day is quantitative variable, it has few unique variables hence a bar plot will 
# be a better a visualization hence used.



#%%
from plotnine import geom_bar, xlab, ylab, geom_smooth, geom_point


#%%
def barplot(data_to_plot: pd.DataFrame, variable_to_plot: str, title: str,
            ylabel: str, y_colname: str = None):
    if y_colname is None:
        bar_graph = (ggplot(data=data_to_plot, mapping=aes(x=variable_to_plot, 
                                                fill=variable_to_plot
                                            )
                                )
                                + geom_bar() #+ theme_dark() 
                                + ggtitle(title) + xlab(variable_to_plot)
                                + ylab(ylabel)
                        )

        return print(bar_graph)
    else:
        bar_graph = (ggplot(data=data_to_plot, 
                            mapping=aes(x=variable_to_plot, 
                                        y=y_colname,
                                        fill=variable_to_plot
                                        )
                                )
                                + geom_col() #+ theme_dark() 
                                + ggtitle(title) + xlab(variable_to_plot)
                                + ylab(ylabel)
                        )

        return print(bar_graph)
        


#%%

barplot(data_to_plot=df, variable_to_plot='hour_of_day', title='Hour of Day when sessions are made',
        ylabel='Frequency of hour of day', )

## From the plot of hour of the day, one can argue that the variable is left-skewed with 
# users making more sessions during later hours of the day as number of sessions continue to increase 
# from the 5th hour to the 19th hour of the day. This may be related to the users' most active periods.


#%% ###### Visualizing the relationship between target variable and quantitative predictors #####
"""
A more important visualization is one that offers insight on the relationship 
between target variable and predictors. In pursuit of investigating whether or not 
a linear relationship between the 
target variable and predictor variables exists, a scatterplot is used to visualize 
them. Scatterplot is more appropriate for quantitative variables. 
Thus, all the quantitative predictor variables are plotted against hits as the target variable on 
the y-axis. This is implemented below;

"""

#%%
def plot_scatterplot(data: pd.DataFrame,
                 x_colname: str,
                 y_colname: str = 'hits'):
    """ Scatterplot to visualize relationship between two variables. 
    Args:
        data (pd.DataFrame): Data which contains variables to plot
        
        y_colname (str): column name (variable) to plot on y-axis
        x_colname (str): column name (variable) to plot on x-axis
    """
    print(
        (ggplot(data=data, 
                mapping=aes(y=y_colname, x=x_colname)
                ) 
                + geom_point() + geom_smooth(method='lm')
                + ggtitle(f'Scatter plot to visualize relationship between {y_colname} and {x_colname}'
                    )
        )
    )
#

#%% 
plot_scatterplot(data=df, x_colname='session_duration')

# The scatter plot shows that there is a possible positive relationship between hits and session duration.
# The strength of this relationship is not clear but can be determined using statistical methods later 
# after the visualization as part of the feature selection procedure.



 
#%%
barplot(data_to_plot=df, variable_to_plot='hour_of_day', title='Hits per hour of day',
        ylabel='Hits', y_colname='hits')

#%%
"""  

Because hour of day has limited values (0 to 23) using a scatter to visualize it with 
number of hits which has far higher values does not provide the best visual appear. 
A simple approach to visualizing the underlying trend is to aggregate the data by 
grouping based on hour of day and calculating the sum of hits to visualize how they relate.
This is implemented below;

"""

#%%  scatterplot of total hits and hour of the day.

hits_per_hour_of_day = df.groupby(by='hour_of_day')['hits'].sum().reset_index()
plot_scatterplot(data=hits_per_hour_of_day, x_colname='hour_of_day', y_colname='hits')

""" 
The above graph is based on the total hits at various hours of the day. With this tranformed data, some 
insights are discernible. Noticeable is the changing relation being hits and hour of the day with total 
hits decreasing as hour of the day increases in the early hours of 0 to 4 and then increasing for 5 to 19 
before decreasing thereafter. The insight gained from this gain to used to create a new feature which 
can be used for prediction of hits.

"""

#%%  ####### Visualization of categorical predictors  ###########

"""
Before visualizing the categorical variables, it is worth noting that such visualizations 
are usually more insightful
and appropriate when the categorical variable is of a low cardinality. By this, predictors with
high cardinality are first identified and visualization is done on low cardinal predictors. 
Preprocessing of predictors with High cardinality will be done separately after visualization.
Checking for high cardinality is implemented as follows.
"""

#%%

categorical_vars = ['locale', 'day_of_week', 'agent_id', 'entry_page', 'path_id_set', 'traffic_type']

def get_unique_values(data: pd.DataFrame, variable: str):
    num_values = data[variable].nunique()
    print(f'{variable} has {num_values} unique values')
    
    
#%%

for cat_var in categorical_vars:
    get_unique_values(data=df, variable=cat_var)   


#%%
"""
From the analysis "entry_page" and "path_id_set" have unique values more than 100 hence are considered 
to be of high cardinality. locale, day_of_week and agent_id have 6, 7, and 15 unique values respectively 
which can be considered to be relatively of a low cardinality.
"""



#%%
"""
In order to visualize the influence of categorical predictor variables on the target variable, 
bar charts were used to depict how total hits is distributed among the various categories. 

"""

#%% visualizing total hits among locale

total_hits_per_locale = df.groupby(by='locale')['hits'].sum().reset_index()

barplot(data_to_plot=total_hits_per_locale, variable_to_plot='locale', 
        y_colname='hits', ylabel='total hits', 
        title='Total hits on various platforms (locale)')


# The difference in total hits between varous locales if is not that much hence it is possible that 
# type of locale does not have a strong influence on hits. This needs statistical test to verify.

#%%   ###############  visualizing total hits among day_of_week    ###############

total_hits_per_dayofweek = df.groupby(by='day_of_week')['hits'].sum().reset_index().dropna()
total_hits_per_dayofweek

barplot(data_to_plot=df, variable_to_plot='day_of_week', y_colname='hits',
        title='Total hits per day of week', ylabel='Total hits')

""" 
From the visualization, there is not much of a difference in total hits between most days of 
the week. This suggests that day of week may not have a significant impact on hits hence 
not likely to offer adequate signal for prediction of hits. A statistical test will 
offer a more quantitative and objective measure to determine whether to select this 
feature for inclusion in the model.

"""


#%%  ############   visualizing total hits per traffic_type  ##############

total_hits_per_traffic_type = df.groupby(by='traffic_type')['hits'].sum().reset_index()

total_hits_per_traffic_type

barplot(data_to_plot=total_hits_per_traffic_type, variable_to_plot='traffic_type',
        y_colname='hits', ylabel='Total hits', title='Total hits per channel (traffic_type)'
        )


#%%
"""  

The bar graph of total hits by channel (traffic type) shows high difference in hits between channels
hence likely to be a relevant feature that impacts hits. To select this feature as an important predictor 
to be included in the model, statistical test will be conducted to verify the graphical interpretation.


"""

#%% visualizing total hits per agent_id

total_hits_per_agentid = df.groupby(by='agent_id')['hits'].sum().reset_index()

total_hits_per_agentid

barplot(data_to_plot=total_hits_per_agentid, variable_to_plot='agent_id',
        y_colname='hits', ylabel='Total hits', title='Total hits per device used (agent_id)'
        )


""" 
From the visualization, agent_id has total hits with less difference between certain devices 
and more difference between other devices. With 15 class, it is argued that this variable 
needs to treated as a high cardinality predictor in the feature transformation process. 
is likely to introduce sparsity and 

"""

#%% Feature selection
"""
A key element of eyploratory analysis is to gain insights that enable selection of only relevant predictors 
that actually contribute to and improve the model. Filter-based selection methods were used for the feature 
selection exercise based on statistical analysis that provides hints on the relationship between predictors 
and the target variable. Feature selection was done before transformation for some variables because feature 
transformation for them will be a more expensive operation. Regards, tranformed features were also 
subjected to the appropriate statistical methods for feature selection. The type of statistical analysis 
used for the feature selection as hinted earlier is determined by the data type of the predictor. 

Thus, feature selection is implemented as follows;
"""

#%% feature selection for categorical features with low cardinality

"""

To select relevant categoricla features, hypothesis is tested to determine whether the is a significant 
difference in hits between the classes of a categorical variable. A significant difference suggests that 
the variable is a significant predictor of hits hence have an influence that will improve the model. Univariate 
statistical methods are used to determine that.

"""

#%% ####### Feature selection: Is locale a relevant predictor of hits for the model  ###########

"""
A element that is considered in determining whether a categorical variable such as locale is a relevant 
predictor is the determination of variance of hits between the various categories of locale. By this, when 
hits significantly varies between the various categories of locale that is likely to be statistically 
significant predictor of hits. This notions applies other categorical predictors such as day_of_week, 
agent_id, 'locale', 'day_of_week' and 'traffic_type'. For high cardinality predictors such as 'agent_id',
'entry_page', 'path_id_set', it better to first 
treat them and reduce the classes to a manageable few before applying relevant statistical test.


In determining which statistical test to use, the assumptions required were tested to determine 
whether a parametric or non-parametric method of statistical test was appropriate. 
A parametric method such as Student t-test requires the data to be normally distributed and variance 
to be homogeneous for the various categories present in the predictor. 
When these assumptions are not captured by the data then a non-parametric method such as 
Welch's t-test can be used.

Both visualization and statistical methods are used to verify these assumptions. For a categorical 
variable, Boxplot is a good graphical technique to visulaize how hits varies with the various categories.
This is then supplemented by a levene test and bartlett test to statistically verify the varies 
depicted in the boxplot. levene method determines whether groups are homogeneous. 

The discussion is implemented in code for all low cardinal variables statring with locale as follows.
"""
#%% ### boxplot to visualize variance of hits among categories of predictors  #####
from plotnine import coord_flip, scale_x_discrete, geom_boxplot


#%% 
def boxplot(data_to_plot: pd.DataFrame, x_colname: str, y_colname: str,
            title: str = None):
    
    
        
    #if (x_colname is not None and y_colname is not None):
    if title is None:
        title = f'Distribution of {y_colname} among various {x_colname}'
        
    box_graph = (ggplot(data=data_to_plot, mapping=aes(x=x_colname, y=y_colname))
                    + geom_boxplot()
                    + coord_flip()
                    + ggtitle(title)
                    # + scale_x_discrete(
                    #     labels=months[::-1],
                    #     limits=flights.month[12::-1],
                    #     name='month',
                    # )
                )
    # return ggplot is printed to draw the graph which is not the case by default 
    # when not printed
    return print(box_graph)
    
#%%
boxplot(data_to_plot=df, x_colname='locale', y_colname='hits')

""" 
From the boxplot of hits among locales, there appears to be difference in how 
hits varies among the varous locales and some data points are arguably outliers. 
A statistical test is undertaken to determine if hits in homogenous among the 
varous groups. Such a statistical test is premised on a hypothesis which is framed as 
follows: 


Null Hypothesis (H0): There is no statistically significant difference in variance of hits 
            between categories of a predictor (locale)

Alternate Hypothesis (H1): There is statistically significant difference in variance of hits
            between categories of a predictor (locale)
            
            
For all hypothesis test of homogeneity, this framework is assummed for each categorical predictor.

Both Levene test and Bartlett test are to check homogeneity and implemented as follows:

"""
#%% test of homogeneity
def test_homogeneity(data: pd.DataFrame, target_var: str, predictor_var: str):
    infostat_test = infostat()
    sig_level = f'at 5% significance level'
    infostat_test.bartlett(df=data, res_var=target_var, xfac_var=predictor_var)
    bartlett_summary = infostat_test.bartlett_summary
    bartlett_pval = bartlett_summary[bartlett_summary['Parameter'] == 'p value']['Value'].item()
    
    if bartlett_pval <= 0.05:
        bart_res = 'reject Null hypothesis of equal variance'
    else:
        bart_res = 'fail to reject Null hypothesis of equal variance'
        
    bartlett_interprete = f'With a p-value of {bartlett_pval} the bartlett test suggests to: {bart_res} {sig_level}'
    
    infostat_test.levene(df=data, res_var=target_var, xfac_var=predictor_var)
    levene_summary = infostat_test.levene_summary
    levene_pval = levene_summary[levene_summary['Parameter'] == 'p value']['Value'].item()
    
    if levene_pval <= 0.05:
        levene_res = 'reject Null hypothesis of equal variance'
    else:
        levene_res = 'fail to reject Null hypothesis of equal variance'
        
    levene_interprete = f'With a p-value of {levene_pval}, the Levene test suggests to: {levene_res} {sig_level} '
    
    # results are printed and not return but in case of production environment they will be return
    print(f'Barlett test results of {predictor_var}')
    print(f'{bartlett_summary} \n')
    
    print(f'Levene test results of {predictor_var}')
    print(f'{levene_summary} \n')
    
    print(f'{bartlett_interprete} \n')
    print(f'{levene_interprete} \n')
    
    
    
#%%

test_homogeneity(data=df, target_var='hits', predictor_var='locale')

#%%
""" 
Given that the null hypothesis that variance is homogeneous is rejected, 
a non-parametric method such as kruskall wallis is used to 
determine whether there is statistically significance differnence in mean hits between various 
locale. 
Kruskal Wallis test is used to determine if the mean hits is equal among varous locales.
Generally, testing such a hypothesis is based on the following framework

Null hypothesis: There is no statistically significant difference in mean hits between 
     various categories of predictor (locale)
     
Alternative hypothesis: There is statistical significant difference in the mean hits 
    between various categoreis of the predictor (locale)
    
The significant difference between the mean hit of locales will suggest that locale 
is capable of discriminanting on the number of hits that a user can make ducring a session 
hence a good predictor of it.  This will want its selection. This logic applys to all 
categorical predictors being assessed with this method.

"""

#%%

pg.kruskal(data=df, dv='hits', between='locale')
### The uncorrected p-value of 0.0 suggests to reject the null hypothesis hence mean hits is 
# significantly different for at least one of the categories of locale. This suggest that 
# locales is likely to be a good predictor hence should be selected for the model building.


#%%  #### Determine relevance of 'day_of_week' to be selected for modelling   #########  

# Visualization for Homogeinety of variance of hits for various days of the week 
# is achieved using boxplot

#unkn_dayofweek = df['day_of_week'].unique()[-1]

#df_dayofweek = df[df['day_of_week']!=unkn_dayofweek].copy()

boxplot(data_to_plot=df, x_colname='day_of_week', y_colname='hits')

#%% test for homogeineity of variance in day_of_week

test_homogeneity(data=df, target_var='hits', predictor_var='day_of_week')

# The test results shows suggest the assumption of homogeneity of variance is rejected and a 
# non-parametric method is used to test equal mean hits among the various days of the week
#%% Testing that hits is equal among days of the week
pg.kruskal(data=df, dv='hits', between='day_of_week')

# Given that mean hit is different among the various days of the week, day of week is likely to be 
# a good predictor of hits hence selected to be included in the model.




#%%  #########  Determine relevance of traffic_type to be selected for modelling  ##########

df_traffic = df.copy()

df_traffic['traffic_type'] = df_traffic['traffic_type'].astype(str)

#%% visualizing how hits distribution various among various traffic type
boxplot(data_to_plot=df_traffic, x_colname='traffic_type', y_colname='hits')

#%%
test_homogeneity(data=df_traffic, target_var='hits', predictor_var='traffic_type')

#%%
pg.kruskal(data=df_traffic, dv='hits', between='traffic_type')

# From the kruskal test, the null hypothesis of no difference in hit between various traffic types
# reject which means that traffic type is a relevant predictor for hits


"""
After using statistical and visualizations method for filter-based feature selection of 
low cardinality categroical predictors, high cardinality predictors are transformed low cardinals.


## Handling predicotrs with high cardinality

While a number of techniques exists for handling categorical predictors with high cardinality,
the approach adopted for this task was to recategorize less representative values. By this, the number 
of hits and sessions recorded for a category is taken into consideration. The logic is that,
before a platform, channel or device is even considered for predicting hits, we have to make sure that 
users are even using it to interact with trivago page as often as it becomes a dominant medium for 
making hits. 

The techniques used are quite simple. First, the expected threshold for equal proportion or share of hits or 
session required by each category of a high cardinal predictor is computed. This is simply computed as
dividing 100% by the cardinality of the predictor (number of unique classes / categories). Thus, 
if all the categories of a predictor achieve this threshold of equal proportion, then there will equal 
variance among them and they are likely to have equal infleunec hence redundunct when the predictor is 
used for modelling. Base on this logic, it becomes intuitive that all categories that have not more than
this threshold estimate needed to be recategorized which is likely to boost their potential 
for offering predictive signal in addition to eliminating high cardinality.

This is implemented in code as follows:  

"""




#%%

def recategorize_predictor(data: pd.DataFrame, high_cardinality_predictor: str,
                          target_for_regroup: str, cat_value_to_assign: str = 'other'):
    total_hits_per_class = data.groupby(by=high_cardinality_predictor)[target_for_regroup].sum().reset_index()
    total_hits_per_class['percent_of_total_hits'] = ((total_hits_per_class[target_for_regroup]
                                                      / total_hits_per_class[target_for_regroup].sum()) * 100
                                                     )
    equal_prop_threshold = 100 / total_hits_per_class[high_cardinality_predictor].nunique()
    total_hits_per_class[f'{high_cardinality_predictor}_new_cat'] = (np.where(total_hits_per_class['percent_of_total_hits']
                                                                             <=equal_prop_threshold, cat_value_to_assign, 
                                                                             total_hits_per_class[high_cardinality_predictor]
                                                                             )
                                                                    )
    return total_hits_per_class


#%%
agent_id_recat = recategorize_predictor(data=df, high_cardinality_predictor='agent_id',
                       target_for_regroup='hits', cat_value_to_assign='other'
                       )


#%%

entry_page_recat = recategorize_predictor(data=df, high_cardinality_predictor='entry_page',
                                            target_for_regroup='hits'
                                            )

## The technique used reduce cardinality of entry_page from 115 to 8
## It is even possible to reduce the cardinality further by applying the function a second
# time for entry_page


#%%

def reduce_high_cardinality(data: pd.DataFrame, high_cardinality_predictor: str,
                            recategorize_loookup_data: pd.DataFrame, 
                            recategorized_colname_in_lookup: str,
                            cat_value_to_assign: str = 'other',
                            recategorized_value_in_lookup: str = 'other'
                            
                            ):
    cardinality_values_reclassed = (recategorize_loookup_data[recategorize_loookup_data[
                                                                recategorized_colname_in_lookup
                                                                ] 
                                                == recategorized_value_in_lookup]
                                                    [high_cardinality_predictor]
                                                    .values.tolist()
                                    )
    data[f'{high_cardinality_predictor}_new_cat'] = (np.where(data[high_cardinality_predictor]
                                                              .isin(cardinality_values_reclassed), 
                                                            cat_value_to_assign, 
                                                        data[high_cardinality_predictor]
                                                        )
                                                     )
    return data


#%%

df_recat = df.copy()

#%% reduce cardinality of agent_id

df_recat = reduce_high_cardinality(data=df_recat, recategorize_loookup_data=agent_id_recat,
                                    high_cardinality_predictor='agent_id',
                                    recategorized_colname_in_lookup='agent_id_new_cat',
                                    cat_value_to_assign='other',
                                    recategorized_value_in_lookup='other'
                                    )


#%% reduce cardinality of entry_page
df_recat = reduce_high_cardinality(data=df_recat, recategorize_loookup_data=entry_page_recat,
                                high_cardinality_predictor='entry_page',
                                recategorized_colname_in_lookup='entry_page_new_cat',
                                cat_value_to_assign='other',
                                recategorized_value_in_lookup='other'
                                )



#%% #######  Determinining relevance of recategorized categorical predictors for hits prediction ########
##########    recategorized  agent_id   #############
## visualize distribution of hits among agent_id

boxplot(data_to_plot=df_recat, x_colname='agent_id_new_cat', y_colname='hits')

#%%

test_homogeneity(data=df_recat, target_var='hits', predictor_var='agent_id_new_cat')

# The result indicate to reject the null hypothesis of homogeneity of variance and 
# non-parametric method is used to determine the relevance of recategorized agent_id

#%% agent_id_new_cat
pg.kruskal(data=df_recat, dv='hits', between='agent_id_new_cat')

# The result shows that the recategorized agent_id will be a relevant predictor for hits


#%%   ############ Determining the relevance of recategorized entry_page

boxplot(data_to_plot=df_recat, x_colname='entry_page_new_cat', y_colname='hits')

#%% test of homogeneity of variance
test_homogeneity(data=df_recat, target_var='hits', predictor_var='entry_page_new_cat')

# the result suggest the use of non-parametric method for further analysis given that
# hmogeneity of variance was reject.

#%% 
pg.kruskal(data=df_recat, dv='hits', between='entry_page_new_cat')
# the analysis shows that hits was significantly different among the various recategorized 
# entry-page hence chosen for the prediction  


"""

The feature path_id_set has a very high cardinality such that even after applying 
the techniques used here, it will still be of high cardinality.
Hence it be will subjected to separate feature engineering likely to produce 
a better transformed predictor.


"""


#%%  Feature engineering

"""
Adopting data centric approach to modelling requires strategies to develop new features that are relevant
for the modelling task and this was attended to. 

The path_id_set predictor offers valuable information that can be extracted for prediction of hits.
Applying domain knowledge in online advertisment products, page hits usually increase with increase in 
number of pages that an user visits during a session. In this case, as users visit more locations on the 
page one can expects hits to increase hence the number of locations visited is likely to be 
a good predictor. 

Thus, number of locations visited during a session is an additional indicator created by extracting 
insights from path_id_set. Given that each location visited was identified with an ID in path_id_set,
it was determine that counting the number of IDs in the path_id_set for each session is representative
of the number of locations visited durng a session

The algorithm for developing number of locations visited as a predictor is implemented as follows:

"""

#%%
def compute_number_of_locations_visited(data: pd.DataFrame, pathid_colname: str = 'path_id_set'):
    # create a column that will store the 
    data['num_locations_visited'] = np.nan

    for index, row in data.iterrows():
        pathid = row['path_id_set']
        
        # count number of path_id only where is not NaN because 
        # missing values need to be imputed
        if pathid is not np.nan:
            path_list = ast.literal_eval(pathid)
            data['num_locations_visited'][index] = len(path_list)
        
    return data


#%%  #######  implementation for creating number of locations visited predictor  #######

df_recat = compute_number_of_locations_visited(data=df_recat)


#%%  Numeric feature selection

"""
Techniques employed for numeric feature selection are different from that of categorical features.
Among others, multicollinearity is checked to prevent including redundant predictors.
Some algorithmns assume that
the variables are not strongly correlated to each other. Strong correlation between variables
implies the variables are supplying similar information to the algorithmn
hence dimension reduction technique could be used to reduce or select
only variables that enable the algorithmn to gain new insights 
from the data and improve predictive power.

Correlation analysis is undertaken on the numeric predictors to 
check for multicollinearity as follows;.



"""

#%% select numeric predictors for correlation

numeric_predictors = ['hour_of_day', 'session_duration', 'num_locations_visited']


#%%
df_numeric_var = df_recat[numeric_predictors]


"""
In order to determine the right method to determine the strength relationship between variables,
scatterplot is used to visualize whether or not the variables are linearly related. Parametric 
method such as Pearson's correlation requires linear relationship, homoscedascticity and 
continuous variable. In an instance where one of the variables is discrete ordinal or one of the 
identified assumption is violated then non-parametric methods such as spearman correlation is used.
Relationship between session_duration and num_locations_visited is visualized below.


"""
#%% 

plot_scatterplot(data=df_numeric_var, x_colname='session_duration', y_colname='num_locations_visited')

#%%
"""
Given that scatteplot does not appear to show a clear linear relationship, spearman correlation is used.
The implementation is provided below.
"""

#%%
corr_matrix = df_numeric_var.corr(method='spearman')

corr_matrix

#%%
"""
None of the predictors is strongly correlated which implies they will provide different signals for 
modeling and can be included upon further feature selection methods have been verified. 
The correlation result is visualize below.

"""

#%%
# Create a mask to hide the upper triangle
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True

# visualize correlation matrix
corr_plot = sns.heatmap(corr_matrix, mask=mask, cmap=sns.color_palette("GnBu_d"), 
                        vmax=.3, center=0, 
                        square=True, linewidths=.5, cbar_kws={"shrink": .5}
                        )

print(corr_plot)

#%%

"""
The correlation analysis shows a weak correlation 
between the predictor variables hence multicollinearity is absent.
"""

#%%

df_recat[['hits', 'session_duration']].corr()

#%%
df_recat[['hits', 'num_locations_visited']].corr(method='spearman')

#%%
df_recat[['hits', 'hour_of_day']].corr(method='spearman')

#%%
"""
Correlation analysis between the target variable and numberic predictors shows a moderate relationship
except for hour_of_day which weak. A decision is made to select session_duration and num_locations_visited
for modeling task while the very weak correlation of hour_of_day means that it is not selected to 
be included in the model.

"""


#%%  ###########   Visualizing outliers   ###########

"""
#### Boxplot to visualize outliers
Some algorithms 
are influence by the presence of outliers hence analyzed to make an inform 
decision on which class of algorithm to choose from.
"""

#%%

# function to create boxplot
def make_boxplot(data: pd.DataFrame, variable_name: str):
    """This function accepts a data and variable name and returns a boxplot

    Args:
        data (pd.DataFrame): Data to visualize
        variable_name (str): variable to visualize with boxplot
    """
    fig = px.box(data_frame=data, y = variable_name,
                 template='plotly_dark', 
                 title = f'Boxplot to visualize outliers in {variable_name}'
                 )
    fig.show()


#%%

make_boxplot(data=df_recat, variable_name='hits')


#%%
for var in numeric_predictors:
    make_boxplot(data=df_recat, variable_name=var)
    
#%%
    
"""
From the boxplot of hits, it can be argue that an oitlier exist in the data. This is 
also the case for number of locations visited predictor. A element to note is that, the 
outlier in the hits is also corresponds with the outlier in number of locations visited predictor.
This suggest that this is likely not a wrong measurement in which case the outlier can be imputted
with various methods such as mean or median imputation and regression methods. Rather, this 
could be the development of new trends that have not be adequately capture in the data hence appear to 
be an outlier in this sample.

This information informs the decision making process 
about which model to choose. From the visualization,
a decision is made in favour of choosing
a model that is fairly robust against outliers instead of removing
or imputing the outliers altogether.

"""  

#%% ##############  Handling missing data  ###################  

"""
From the point of the target variable where about 33.3% of the data is missing, a number of 
paths are possible some of which may not be appropriate for this task. These are identifed as 
follows:

1. Use semi-supervised method for the task
2. Impute the missing data
3. Use only labeled data points


From the above identfied approach, it can be argue that imputation of target variable is not 
adversable given that this directly influence the learning process of the data and such methods 
come with their margin of errors whereas we want to avoid errorneous labelling at all cost 
as it influence it does not reflect objective evaluation of the model. In any case, imputing as 
much as 33.3% means introducing a high amount of "artificial data" that can change the actual 
underlying distribution of the phenomenon being studied.


For this task and availaible time alotted, using only the labelled data based on the target variable 
is deem to a more reserved and appropriate approach.

"""


#%%

"""
### Using insights gained from exploratory analysis to inform modelling approach

The findings of non-linear relationship between 
the predictors and the outcome variable(s), presence of outliers,
and sizeable missing values suggest that a non-parametric
model that handles non-linear relationship,
outliers and missing values will be appropriate for the task.
 
Moreover, as identified in the objectives, the focus of the task is on 
a good precision rather than interpretability of the model and this informed 
a decision to choose an algorithm that satisfy those needs. 


On the basis of the findings from the exploratory analysis, a model 
that employs decision tree will be used. Before that, a critical aspect of 
the findings was that missing data is present and has to handled. 
For this, a model that natively handles missing data as part of the modelling
process is explored. Hence, HistGradientBoostingRegressor from the sklearn library
is implemented because it is a decison tree based model 
 with an inbuilt handling of missing data for predictors.

It is duely recognized that semi-supervised learning could be used given that 
both labelled and unlabelled target variable exist. Nonetheless, supervised learning
is used to demonstrate the modeling process and this means using only data points 
for which records are available for hits.

Given that the target variable is continuous, supervised regression is used for the 
task. 
"""

#%%

"""
## Preprocessing data for machine learning
The preprocessing pipeline for variables to be prepared for modeling is highlighted as follows

1. Multi-class categorical variables are encoded using One-hot encoding strategy 
2. Numercial variables are scaled using the standard scaler.

The preprocessing pipeline is implemented as follows: 

## Encoding categorical variables to prepare them for modelling

While several encoding strategies exist to transform categorical variable into 
forms that machine learning models can understand, one hot encoding was used 
in this task.

"""





#%%


one = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()

preprocess_pipeline =  make_column_transformer((scaler, args.selected_numeric_features),
                                                (one, args.categorical_features)
                                                )

logit_model_pipeline = make_pipeline(preprocess_pipeline,
                                    LogisticRegression(class_weight='balanced')
                                    )







#%%

df_recat[['hits']].plot(kind='box')
 


#%%

entry_page_recat.groupby(by='entry_page_new_cat')['percent_of_total_hits'].sum().reset_index()

entry_page_recat

#%%

entry_page_recat_2 = recategorize_predictor(data=entry_page_recat, 
                                            high_cardinality_predictor='entry_page_new_cat',
                       target_for_regroup='percent_of_total_hits'
                       )

#%%

path_id_recat = recategorize_predictor(data=df, high_cardinality_predictor='path_id_set',
                       target_for_regroup='hits'
                       )



#%%

"""
locale has 6 unique values
day_of_week has 7 unique values
agent_id has 15 unique values
entry_page has 115 unique values
path_id_set has 52345 unique values
traffic_type has 6 unique values
"""

#%%

#%
df.groupby(by='agent_id')['index'].count().reset_index()

#%%
df['agent_id'].value_counts(normalize=True) * 100

#%%




#%% ## bartlett's test for homogeneity



['locale', 'day_of_week', 'agent_id', 'traffic_type']







    

#%% Numeric predictor transformation



#%%

hits_per_sess_dur = df.groupby(by=['session_duration'])['hits'].sum().reset_index()

#%%

plot_scatterplot(data=hits_per_sess_dur, x_colname='session_duration', y_colname='hits')




#%%



"""
b)What other columns would you like to have to improve your model?

(I) Holidays variable which indicates whether or not a day is a holiday is likely to be a predictive 
feature of hits. This guess is based on the educated assumption that holidays offer users the freedom fo 
space and time to be online and engage more with trivago page which will possibly lead to higher hit compared
to non holidays.

(II) Number of days before a holiday or major celebration: There is likely to be a relationship between
the number of days before a holiday as users are likely to use the platform more often before a certain 
number of days priori to a holiday to make bookings and in the process make more hits on the page. How 
a "count down to a holiday" infleunces a page hits and the threshold beyond which page hits significanty 
increases is not  known for certain but can be researched and included in the model for predicting hits.



"""


#%%

# def compute_number_of_locations(data, pathid_colname: str = 'path_id_set'):
#     indexs = data.index
#     data['num_locations'] = np.nan
#     for i in range(0, len(indexs)):
#         rowindex = indexs[i]
#         pathid_list = json.loads(data[pathid_colname][rowindex])
#         data['num_locations'][rowindex] = len(pathid_list)
#     return data
        



#%%
small_testdata = locale2_avg_hits.copy()




#%%

#%%

stats.f_oneway(missing_hits_per_locale[missing_hits_per_locale['locale'] == 'L1']['num_missing_hits'],
               missing_hits_per_locale[missing_hits_per_locale['locale'] == 'L2']['num_missing_hits'],
               missing_hits_per_locale[missing_hits_per_locale['locale'] == 'L3']['num_missing_hits'],
               missing_hits_per_locale[missing_hits_per_locale['locale'] == 'L4']['num_missing_hits'],
               missing_hits_per_locale[missing_hits_per_locale['locale'] == 'L5']['num_missing_hits'],
               missing_hits_per_locale[missing_hits_per_locale['locale'] == 'L6']['num_missing_hits']

)





#%% #######  Are all locales impacted by the loss of data equally?  ############


#%%  ######### tesing for NAN replace with 1 and test  ###########

df_miss_test = df_missing_hits.copy()

df_miss_test['hits'] = 1
df_miss_test.groupby(by='locale')['hits'].mean()

#%%

pg.kruskal(data=df_miss_test, dv='hits', between='locale')

#%%
pg.anova(data=df_miss_test, dv='hits', between='locale')

#%%

pg.welch_anova(data=df_miss_test, dv='hits', between='locale')

#%%

stats.f_oneway(df_miss_test[df_miss_test['locale'] == 'L1']['hits'],
               df_miss_test[df_miss_test['locale'] == 'L2']['hits'],
               df_miss_test[df_miss_test['locale'] == 'L3']['hits']
               )

#df_miss_test[df_miss_test['locale'] == 'L1']['hits']

