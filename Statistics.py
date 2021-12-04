{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "75808f73",
   "metadata": {},
   "source": [
    "# Introduction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "dfca5603",
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import numpy; import numpy as np; import scipy; import matplotlib; import pandas; import quandl; import statsmodels\n",
    "import statistics; import math; import stats; import wooldridge as woo;\n",
    "import os; import pathlib"
   ]
  },
  {
   "cell_type": "raw",
   "id": "00857511",
   "metadata": {},
   "source": [
    "# IMPORTANT NUMPY FUNCTION AND METHODS:\n",
    "numpy.add(1,2); 1+2 #sums of elements in x and y\n",
    "numpy.subtract(1,2); 1-2\n",
    "numpy.divide(1,2); 1/2\n",
    "numpy.multiply(1,2); 1*2\n",
    "numpy.exp(x) # element -wise exponential of all element in x\n",
    "numpy.squrt(x) # square root of all element in x\n",
    "numpy.log(x) # natural algorithm of all elements in x\n",
    "numpy.linalg.inv (x) # inverse of x"
   ]
  },
  {
   "cell_type": "raw",
   "id": "b60b70b8",
   "metadata": {},
   "source": [
    "# IMPORTANT PANDAS FUNCTION AND METHODS:\n",
    "df.head() # first 5 observation in df\n",
    "df.tail() # last 5 observation in df\n",
    "df.describe() # print describe statistics\n",
    "df.set_index(x) # set the index of df as x\n",
    "df['x'] or df.x # access x in df\n",
    "df.iloc(i,j)# access variables and observations in df by integer position\n",
    "df.loc(names_i, names_j) # access variables and observations in df by names\n",
    "df['x'].shift(i) # create shifted variables of x by i rows\n",
    "df['x'].diff(i) # creates variables that contain the ith difference of x\n",
    "df.groubby('x').function() # apply a function to subgroup of df according to the x"
   ]
  },
  {
   "cell_type": "raw",
   "id": "226a2ca8",
   "metadata": {},
   "source": [
    "# Summary Econometrics in Python\n",
    "\n",
    "# Chapter One Introduction #\n",
    "1.1. First Python Script\n",
    "1.2 Python as a calculator \n",
    "1.3 Module math \n",
    "1.4 Object in Pythonx \n",
    "1.5 Lists - Copy \n",
    "1.6 Lists\n",
    "1.7 Dicts - Copy\n",
    "1.8 Dicts \n",
    "1.9 Numpy Arrays\n",
    "1.10 Numpy Special Cases \n",
    "1.11 Numpy Operations\n",
    "1.12 Pandas\n",
    "1.13 Pandas Operastions\n",
    "1.14 Wooldridge \n",
    "1.15 Import-Export\n",
    "1.16 Import Stock Data\n",
    "1.17 Graphs Basics\n",
    "1.18 Graphs Basics2\n",
    "1.19 Graphs Functions\n",
    "1.20 Graphs Builiding Blocks\n",
    "1.21 Graphs Export\n",
    "1.22 Describe Tables\n",
    "1.23 Describe Figures\n",
    "1.24 Histogram \n",
    "1.25 KDensity \n",
    "1.26 Descr-ECDF\n",
    "1.27 Descr-Stats (Data Summary)\n",
    "1.28 Descr-Boxplot \n",
    "1.29 PMF Binom\n",
    "1.30 PMF Example\n",
    "1.31 PDF Example \n",
    "1.32 CDF Example \n",
    "1.33 CDF Figure\n",
    "1.34 Quantile Example \n",
    "1.36 Sample Bernoulli \n",
    "1.37 Sample Norm \n",
    "1.38 Random Numbers\n",
    "1.39 Example C.2 (Woodridge's data)\n",
    "1.40 Example C.3 \n",
    "1.41 Critical Values of t\n",
    "1.42 Example C.5 \n",
    "1.43 Example C.6\n",
    "1.44 Example C.7\n",
    "1.45 Adv-Loops\n",
    "1.46 Adv-Loops2\n",
    "1.47 Adv-Functions\n",
    "1.48 Adv ObjOr\n",
    "1.49 Adv ObjOr2\n",
    "1.50 Adv ObjOr3\n",
    "1.51 Stimulate Estimate\n",
    "1.52 Simulation Repeated \n",
    "1.53 Simulation Repeated Results \n",
    "1.54 Simulaltion Inference Figure\n",
    "1.55 Simulation Inference\n",
    "\n",
    "# Chapter 2 The Simple Regression Model #\n",
    "2.1 Example 2.3\n",
    "...\n",
    "2.13 Example 2.12\n",
    "2.14 Simple Linear Regression(SLR).Sample\n",
    "2.15 SLR-Sim.Model \n",
    "2.16 SLR-Sim.Model.Condx\n",
    "2.17 SLR-Sim.Model.ViolSLR4\n",
    "2.18 SLR-Sim.Model.ViolSLR5\n",
    "\n",
    "# Chapter 3 Multiple Regression Analysis: Estimation #\n",
    "3.1 Example 3.1\n",
    "...\n",
    "3.6 Example 3.6 \n",
    "3.7 OLS Matrices\n",
    "3.8 Omitted Vars\n",
    "3.9 Multicollinearity - Standard Error (SE)\n",
    "3.10 Multicolinearity (MLR) - VIF\n",
    "\n",
    "# Chapter 4 Multiple Regression Analysis: Inference #\n",
    "4.1 Example 4.3 \n",
    "...\n",
    "4.5 Example 4.8 \n",
    "4.6 F-Test\n",
    "4.7 F-Test Automatic\n",
    "4.8 F-Test Automatic2\n",
    "\n",
    "# Chapter 5 Multiple Regression Analysis: OLS Asymptotics #\n",
    "5.1 Sim-Asymptotic.OLS.Norm\n",
    "5.2 Sim-Asymptotic.OLS.Chisq\n",
    "5.3 Sim-Asymptotic.OLS.unconditional \n",
    "5.4 Example 5.3 \n",
    "\n",
    "# Chapter 6 Multiple Regression Analysis: Further Issues #\n",
    "6.1 Data Scalling\n",
    "6.2 Example 6.1 \n",
    "6.3 Formula Logarithm\n",
    "6.4 Example 6.2 \n",
    "6.5 Example 6.2 F-Test\n",
    "6.6 Example 6.3 \n",
    "6.7 Predictions\n",
    "6.8 Example 6.5 \n",
    "6.9 Effects Manual \n",
    "\n",
    "# Chapter 7 Multiple Regression Analysis with Qualitative Regressors #\n",
    "7.1 Example 7.1 \n",
    "...\n",
    "7.3 Example 7.1 Boolean \n",
    "7.4 Regression Categorical \n",
    "7.5 Regression Categorical Anova \n",
    "7.6 Example 7.8\n",
    "7.7 Dummy Interact\n",
    "7.8 Dummy Interact Sep\n",
    "\n",
    "# Chapter 8 Heteroscedasticity #\n",
    "8.1 Example 8.2 \n",
    "...\n",
    "8.6 Weighted Least Square(WLS)-Robust \n",
    "8.7 Example 8.7 \n",
    "\n",
    "# Chapter 9 Specification and Data Issues #\n",
    "9.1 Example 9.2 manual \n",
    "9.2 Example 9.2 automatic\n",
    "9.3 Nonnested Test\n",
    "9.4 Sim-Measurement Error (ME) Dep\n",
    "9.5 Sim-ME.Explanation \n",
    "9.6 NA-NaN-Inf\n",
    "9.7 Missing\n",
    "9.8 Missing Analysis \n",
    "9.9 Outliers\n",
    "9.10 Least Absolute Deviations (LAD)\n",
    "\n",
    "# Chapter 10 Basic Regression Analysis with Time Series Data\n",
    "10.1 Example 10.2\n",
    "10.2 Example Barium\n",
    "10.3 Example Stock Data \n",
    "10.4 Example cont\n",
    "...\n",
    "10.7 Example 10.11\n",
    "\n",
    "# Chapter 11 Further Issues in Using OLS with Time Series Data #\n",
    "11.1 Example 11.4\n",
    "...\n",
    "11.3 Simulate Random Walk\n",
    "11.4 Simulate Random Walk Drift (trend)\n",
    "11.5 Simulate Random Walk Drift-Difference\n",
    "11.6 Example 11.6 \n",
    "\n",
    "# Chapter 12 Serial Correlation and Heteroscedasticity in Time Series Regression \n",
    "12.1 Example 12.2 Static\n",
    "...\n",
    "12.4 Example DW test \n",
    "...\n",
    "12.8 Example ARCH\n",
    "\n",
    "# ADVANCED TOPICS\n",
    "# Chapter 13 Pooling Cross-Sections Across Time:Simple Panel Data Methods #\n",
    "13.1 Example \n",
    "...\n",
    "13.5 Example 13.9\n",
    "\n",
    "# Chapter 14 Advanced Panel Data Methods #\n",
    "14.1 Example \n",
    "...\n",
    "14.4 Example Hausman Test \n",
    "14.5 Example Dummy Variable and Correlated Random Effects (CRE)\n",
    "14.6 Example CRE - Test Random Effects (RE)\n",
    "...\n",
    "14.8 Example 13.9 Clustered Standard Error (ClSE)\n",
    "\n",
    "# Chapter 15 \n",
    "15.1 Example \n",
    "...\n",
    "15.10 Example 15.10\n",
    "\n",
    "# Chapter 16 Simultaneous Equations Models\n",
    "16.1 Example 16.5-2SLS\n",
    "16.2 Example 16.5-3SLS\n",
    "\n",
    "# Chapter 17 Limited Dependent Variable Models and Sample Selection Corrections\n",
    "\n",
    "\n",
    "# Chapter 18 Advanced TI me Series Topics\n",
    "\n",
    "\n",
    "# Chapter 19 Carrying Out an Empirical Project\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b687a1b0",
   "metadata": {},
   "source": [
    "# Start to Code\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ab7a568b-d274-448a-b65f-38c81a690cc1",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# START TO CODE\n",
    "# NOTE: NUMPY IS LIMITED TO 32 DIMENSIONS UNLESS YOU BUILD IT MORE\n",
    "import numpy; import scipy; import matplotlib; import pandas; import quandl; import statsmodels\n",
    "import statistics; import math; import stats; import wooldridge as woo\n",
    "\n",
    "# GETTING START\n",
    "## IMPORTANT NUMPY FUNCTION AND METHODS\n",
    "numpy.add(1,2); 1+2 #sums of elements in x and y\n",
    "numpy.subtract(1,2); 1-2\n",
    "numpy.divide(1,2); 1/2\n",
    "numpy.multiply(1,2); 1*2\n",
    "numpy.exp(x) # element -wise exponential of all element in x\n",
    "numpy.squrt(x) # square root of all element in x\n",
    "numpy.log(x) # natural algorithm of all elements in x\n",
    "numpy.linalg.inv (x) # inverse of x\n",
    "\n",
    "## IMPORTANT PANDAS FUNCTION AND METHODS\n",
    "df.head() # first 5 observation in df\n",
    "df.tail() # last 5 observation in df\n",
    "df.describe() # print describe statistics\n",
    "df.set_index(x) # set the index of df as x\n",
    "df['x'] or df.x # access x in df\n",
    "df.iloc(i,j)# access variables and observations in df by integer position\n",
    "df.loc(names_i, names_j) # access variables and observations in df by names\n",
    "df['x'].shift(i) # create shifted variables of x by i rows\n",
    "df['x'].diff(i) # creates variables that contain the ith difference of x\n",
    "df.groubby('x').function() # apply a function to subgroup of df according to the x# GRAPHS EXPORT\n",
    "# Support same for normality density\n",
    "x = np.linspace(-4,4,num=100)\n",
    "# Get different density evaluations\n",
    "y1 = stats.norm.pdf(x,0,1)\n",
    "y2 = stats.norm.pdf(x,0,3)\n",
    "\n",
    "# Plot (a)\n",
    "plt.figure(figsize=(4,6))\n",
    "plt.plot(x,y1,linestyle= '-', color='black')\n",
    "plt.plot(x,y2,linestyle='--',color='0.3')\n",
    "# Export\n",
    "plt.savefig('PyGraphs/Graphs-Export-a.pdf')\n",
    "plt.close()\n",
    "\n",
    "# Plot(b)\n",
    "plt.figure(figsize=(6, 4))\n",
    "plt.plot(x, yl, linestyle='-', color='black')\n",
    "plt.plot(x, y2, linestyle='--', color='0.3')\n",
    "# plt.savefig('PyGraphs/Graphs-Export-b.png')\n"
   ]
  },
  {
   "cell_type": "raw",
   "id": "1dbcd59f-58c3-48c6-b562-b2cafdb6deba",
   "metadata": {},
   "source": [
    "# Difine to a data frame 1\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "ff586269",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "df:    icecream_sales  weather_code  customers  icecream_sales_lag2\n",
      "0              30             0       2000                  NaN\n",
      "1              40             1       2100                  NaN\n",
      "2              35             0       1500                 30.0\n",
      "3             130             1       8000                 40.0\n",
      "4             120             1       7200                 35.0\n",
      "5              60             0       2000                130.0\n"
     ]
    }
   ],
   "source": [
    "## DIFINE TO A DATA FRAME 1\n",
    "icecream_sales = numpy.array([30,40,35,130,120,60])\n",
    "weather_code = numpy.array([0,1,0,1,1,0])\n",
    "customers = numpy.array([2000,2100,1500,8000,7200,2000])\n",
    "df = pandas.DataFrame({'icecream_sales':icecream_sales, 'weather_code': weather_code, 'customers':customers })\n",
    "\n",
    "## INCLUDES SALES TWO MONTH AGO\n",
    "df['icecream_sales_lag2']= df['icecream_sales'].shift(2)\n",
    "print(f'df:',df)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "65acaaaf-8581-4739-9c56-7892df981360",
   "metadata": {},
   "source": [
    "## Define to a data frame 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "60f43b10",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "df:             icecream_sales  weather_code  customers  icecream_sales_lag2\n",
      "2010-04-30              30             0       2000                  NaN\n",
      "2010-05-31              40             1       2100                  NaN\n",
      "2010-06-30              35             0       1500                 30.0\n",
      "2010-07-31             130             1       8000                 40.0\n",
      "2010-08-31             120             1       7200                 35.0\n",
      "2010-09-30              60             0       2000                130.0\n"
     ]
    }
   ],
   "source": [
    "## DIFINE TO A DATA FRAME 2\n",
    "icecream_sales = numpy.array([30,40,35,130,120,60])\n",
    "weather_code = numpy.array([0,1,0,1,1,0])\n",
    "customers = numpy.array([2000,2100,1500,8000,7200,2000])\n",
    "df = pandas.DataFrame({'icecream_sales':icecream_sales, 'weather_code': weather_code, 'customers':customers })\n",
    "\n",
    "## DATA FRAME\n",
    "x = pandas.DataFrame({'var1':x1,'var2':x2})\n",
    "## DEFINE AND ASSIGN AN INDEX (six ends of month starting on april,2010)\n",
    "ourIndex = pandas.date_range(start='04/2010', freq='M', periods=6)\n",
    "df.set_index(ourIndex, inplace=True)\n",
    "\n",
    "## INCLUDES SALES TWO MONTH AGO\n",
    "df['icecream_sales_lag2']= df['icecream_sales'].shift(2)\n",
    "print(f'df:',df)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7f49e30f-5085-44d7-9743-fb2107fdccc2",
   "metadata": {},
   "source": [
    "## Difine to a data frame 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "77aa778e",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "df:             icecream_sales  weather_code  customers  icecream_sales_lag2  \\\n",
      "2010-04-30              30             0       2000                  NaN   \n",
      "2010-05-31              40             1       2100                  NaN   \n",
      "2010-06-30              35             0       1500                 30.0   \n",
      "2010-07-31             130             1       8000                 40.0   \n",
      "2010-08-31             120             1       7200                 35.0   \n",
      "2010-09-30              60             0       2000                130.0   \n",
      "\n",
      "           weather  \n",
      "2010-04-30     bad  \n",
      "2010-05-31    good  \n",
      "2010-06-30     bad  \n",
      "2010-07-31    good  \n",
      "2010-08-31    good  \n",
      "2010-09-30     bad  \n",
      "group_means          icecream_sales  weather_code    customers  icecream_sales_lag2\n",
      "weather                                                                \n",
      "bad           41.666667           0.0  1833.333333                 80.0\n",
      "good          96.666667           1.0  5766.666667                 37.5\n"
     ]
    }
   ],
   "source": [
    "## DIFINE TO A DATA FRAME 3\n",
    "icecream_sales = numpy.array([30,40,35,130,120,60])\n",
    "weather_code = numpy.array([0,1,0,1,1,0])\n",
    "customers = numpy.array([2000,2100,1500,8000,7200,2000])\n",
    "df = pandas.DataFrame({'icecream_sales':icecream_sales, 'weather_code': weather_code, 'customers':customers })\n",
    "\n",
    "## DEFINE AND ASSIGN AN INDEX (six ends of month starting on april,2010)\n",
    "ourIndex = pandas.date_range(start='04/2010', freq='M', periods=6)\n",
    "df.set_index(ourIndex, inplace=True)\n",
    "\n",
    "## INCLUDES SALES TWO MONTH AGO\n",
    "df['icecream_sales_lag2']= df['icecream_sales'].shift(2)\n",
    "\n",
    "## USE A PANDAS CATEGORICAL OBJECT TO ATTACH LABELS (0=bad, 1=good)\n",
    "df['weather']= pandas.Categorical.from_codes(codes=df['weather_code'], categories=['bad','good'])\n",
    "\n",
    "print(f'df:',df)\n",
    "\n",
    "## MEAN SALES FOR EACH WEATHER CATEGORY\n",
    "group_means= df.groupby('weather').mean()\n",
    "print(f'group_means', group_means)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a606c4da-227e-461f-b759-7cb7bfaf3517",
   "metadata": {},
   "source": [
    "# Data Frame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "877e3bd8",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "      x1  x2     y\n",
      "0   1839   1  1673\n",
      "1   1844   2  1688\n",
      "2   1831   3  1666\n",
      "3   1881   4  1735\n",
      "4   1883   5  1749\n",
      "5   1910   6  1756\n",
      "6   1969   7  1815\n",
      "7   2016   8  1867\n",
      "8   2126   9  1948\n",
      "9   2239  10  2048\n",
      "10  2336  11  2128\n",
      "11  2404  12  2165\n",
      "12  2487  13  2257\n",
      "13  2535  14  2316\n",
      "14  2595  15  2324\n",
      "Mean : x1    2126.33\n",
      "x2       8.00\n",
      "y     1942.33\n",
      "dtype: float64\n",
      "Median : 1831.0\n"
     ]
    }
   ],
   "source": [
    "# Examples of DATA Table \n",
    "x1 = numpy.array([1839, 1844, 1831, 1881, 1883, 1910, 1969, 2016, 2126, 2239, 2336, 2404, 2487, 2535, 2595])\n",
    "x2 = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])\n",
    "y = numpy.array([1673, 1688, 1666, 1735, 1749, 1756, 1815, 1867, 1948, 2048, 2128, 2165, 2257, 2316, 2324])\n",
    "x = numpy.array([[x1],[x2]])\n",
    "x = np.array([ [x1],[x2]])\n",
    "x.shape\n",
    "data = pandas.DataFrame({'x1':x1, 'x2':x2, 'y':y})\n",
    "print(data)\n",
    "\n",
    "## DATA SUMMARY\n",
    "dt_mean = np.mean(data) ; print (\"Mean :\",round(dt_mean,2)) # Calculate mean\n",
    "dt_median = np.median(data) ; print (\"Median :\",dt_median) # calculate median\n",
    "# dt_mode = stats.mode(data); print (\"Mode :\",dt_mode[0][0]) # calculate mode\n",
    "\n",
    "# dt_mode = stats.mode(data) ; print(dt_mode)\n",
    "# print (\"dt_mode :\",dt_mode[0][0]) # calculate mode\n",
    "# print(f'dt_mode',dt_mode)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0a456649-563b-42d3-9ea5-3bbfabb0fb77",
   "metadata": {},
   "source": [
    "## add line based on slope and intercept in Matplotlib?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "32bd1b80-3ac5-4be2-bc8a-a7ebebcf84e1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt \n",
    "import numpy as np    \n",
    "\n",
    "def abline(slope, intercept):\n",
    "    \"\"\"Plot a line from slope and intercept\"\"\"\n",
    "    axes = plt.gca()\n",
    "    x_vals = np.array(axes.get_xlim())\n",
    "    y_vals = intercept + slope * x_vals\n",
    "    plt.plot(x_vals, y_vals, '--')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "6df6477d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "# Some dummy data\n",
    "x = [1, 2, 3, 4, 5, 6, 7]\n",
    "y = [1, 3, 3, 2, 5, 7, 9]\n",
    "\n",
    "# Find the slope and intercept of the best fit line\n",
    "slope, intercept = np.polyfit(x, y, 1)\n",
    "\n",
    "# Create a list of values in the best fit line\n",
    "abline_values = [slope * i + intercept for i in x]\n",
    "\n",
    "# Plot the best fit line over the actual values\n",
    "plt.plot(x, y, '--')\n",
    "plt.plot(x, abline_values, 'b')\n",
    "plt.title(slope)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "0859e2af-daa4-4f4f-8bc4-28e344d0154b",
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    },
    "tags": []
   },
   "outputs": [
    {
     "ename": "AttributeError",
     "evalue": "module 'numpy' has no attribute 'scatter'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-74-a6aff3a01379>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0my\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0marray\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m1.1\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m1.9\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m3.0\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m4.1\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m5.2\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m5.8\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m7\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      6\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 7\u001b[1;33m \u001b[0mnumpy\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mscatter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0my\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      8\u001b[0m \u001b[0mslope\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mintercept\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpolyfit\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      9\u001b[0m \u001b[0mplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mx\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0mslope\u001b[0m \u001b[1;33m+\u001b[0m \u001b[0mintercept\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'r'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\numpy\\__init__.py\u001b[0m in \u001b[0;36m__getattr__\u001b[1;34m(attr)\u001b[0m\n\u001b[0;32m    301\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[0mTester\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    302\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 303\u001b[1;33m             raise AttributeError(\"module {!r} has no attribute \"\n\u001b[0m\u001b[0;32m    304\u001b[0m                                  \"{!r}\".format(__name__, attr))\n\u001b[0;32m    305\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mAttributeError\u001b[0m: module 'numpy' has no attribute 'scatter'"
     ]
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import numpy\n",
    "import numpy as np\n",
    "x = np.array([1, 2, 3, 4, 5, 6, 7])\n",
    "y = np.array([1.1,1.9,3.0,4.1,5.2,5.8,7])\n",
    "\n",
    "numpy.scatter(x,y)\n",
    "slope, intercept = np.polyfit(x, y, 1)\n",
    "plot(x, x*slope + intercept, 'r')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "1eaa5029-2494-4569-bc60-9a4fb7bdd695",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "i=3        # intercept\n",
    "s=2        # slope\n",
    "x=np.linspace(1,10,50)      # from 1 to 10, by 50\n",
    "plt.plot(x, s*x + i)        # abline\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "99dd1025-b94b-4fd9-aa66-e79d935dc879",
   "metadata": {},
   "source": [
    "# Calculate Statistics, Variance, Standard Deviation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "5c2590ec",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sample variance: 78793\n",
      "Sample std.dev: 280.7\n",
      "Range: 764\n"
     ]
    }
   ],
   "source": [
    "## CALCULATE STATISTICS, VARIANCE, STANDARD DEVIATION\n",
    "import statistics\n",
    "from statistics import variance, stdev\n",
    "dt_var = statistics.variance(x1); print (\"Sample variance:\", numpy.round(dt_var,2)) # Calculate variance\n",
    "dt_std = stdev(x1); print (\"Sample std.dev:\", numpy.round(dt_std,2))# calculate standard deviation\n",
    "dt_rng = np.max(x1,axis=0) - np.min(x1,axis=0) ; print (\"Range:\",dt_rng)# Calculate range"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "da1e8297",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean : x1    2126.33\n",
      "x2       8.00\n",
      "y     1942.33\n",
      "dtype: float64\n",
      "Median : 1831.0\n",
      "Median_x1: 2016.0\n",
      "Mean_x1: 2126\n",
      "Sample variance 78793\n",
      "Sample std.dev: 280.7\n",
      "Range: 764\n"
     ]
    }
   ],
   "source": [
    "## DATA SUMMARY\n",
    "dt_mean = np.mean(data) ; print (\"Mean :\",round(dt_mean,2)) # Calculate mean\n",
    "dt_median = np.median(data) ; print (\"Median :\",dt_median) # calculate median\n",
    "median_x1 = np.median(x1); print(f'Median_x1:',median_x1)\n",
    "mean_x1 = np.mean(x1); print(\"Mean_x1:\",round(mean_x1))\n",
    "\n",
    "## CALCULATE STATISTICS, VARIANCE, STANDARD DEVIATION\n",
    "import statistics\n",
    "from statistics import variance, stdev\n",
    "dt_var = statistics.variance(x1); print (\"Sample variance\", numpy.round(dt_var,2)) # Calculate variance\n",
    "dt_std = stdev(x1); print (\"Sample std.dev:\", numpy.round(dt_std,2))# calculate standard deviation\n",
    "dt_rng = np.max(x1,axis=0) - np.min(x1,axis=0) ; print (\"Range:\",dt_rng)# Calculate range"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "adf44139-b1d6-424c-87dc-95ad0e4e928a",
   "metadata": {},
   "source": [
    "# Access column by names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "e00bf68e",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "subset1:             icecream_sales  customers\n",
      "2010-04-30              30       2000\n",
      "2010-05-31              40       2100\n",
      "2010-06-30              35       1500\n",
      "2010-07-31             130       8000\n",
      "2010-08-31             120       7200\n",
      "2010-09-30              60       2000\n",
      "subset2             icecream_sales  weather_code  customers  icecream_sales_lag2  \\\n",
      "2010-05-31              40             1       2100                  NaN   \n",
      "2010-06-30              35             0       1500                 30.0   \n",
      "2010-07-31             130             1       8000                 40.0   \n",
      "\n",
      "           weather  \n",
      "2010-05-31    good  \n",
      "2010-06-30     bad  \n",
      "2010-07-31    good  \n",
      "subset3 2100\n",
      "subset4             icecream_sales  weather_code\n",
      "2010-05-31              40             1\n",
      "2010-06-30              35             0\n",
      "2010-07-31             130             1\n"
     ]
    }
   ],
   "source": [
    "## ACCESS COLUMN BY NAMES:\n",
    "subset1= df[['icecream_sales','customers']]\n",
    "\n",
    "## VARIABILITY OF PRINT:\n",
    "print('subset1:', subset1)\n",
    "\n",
    "## ACCESS COLUMN TO FOURTH ROW:\n",
    "subset2 = df[1:4] #\n",
    "df['2010-05-31':'2010-05-31']\n",
    "print(f'subset2', subset2)\n",
    "\n",
    "## ACCESS ROW AND COLUMNS BY INDEX AND VARIABLE NAMES:\n",
    "subset3 = df.loc['2010-05-31','customers']# as same as the result by following order\n",
    "df.iloc[1,2]\n",
    "print(f'subset3',subset3)\n",
    "\n",
    "## ACCESS OF ROW AND COLUMNS BY INDEX AND VARIABLE INTEGER POSITIONS\n",
    "subset4 = df.iloc[1:4,0:2]\n",
    "print(f'subset4',subset4)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "01c200c0-4bcf-4bb1-a384-8a437f35b57e",
   "metadata": {},
   "source": [
    "# Numpy functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "f93e52f1",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "type(testarraylD): <class 'numpy.ndarray'>\n",
      " dim:\n",
      " (3, 4)\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# 1.9. NUMPY ARRARYS\n",
    "import numpy as np \n",
    "# define arrays in numpy:\n",
    "testarray1D = np.array([1, 5, 41.3, 2.0])\n",
    "print(f'type(testarraylD): {type(testarray1D)}') \n",
    "testarray2D = np.array([[4, 9, 8, 3], [2, 6, 3, 2], [1, 1, 7, 4]]) \n",
    "# get dimensions of testarray2D:\n",
    "dim= testarray2D.shape\n",
    "print (f' dim:\\n {dim}\\n') "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "a0faaec5",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "testarray2D:\n",
      " [[4 9 8 3]\n",
      " [2 6 3 2]\n",
      " [1 1 7 4]]\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print (f'testarray2D:\\n {testarray2D}\\n')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "a65d2cff",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "sequence: \n",
      "[0.  0.2 0.4 0.6 0.8 1.  1.2 1.4 1.6 1.8 2. ]\n",
      "\n",
      "sequence_int: \n",
      "[0 1 2 3 4]\n",
      "\n",
      "zero_array: \n",
      "[[0. 0. 0.]\n",
      " [0. 0. 0.]\n",
      " [0. 0. 0.]\n",
      " [0. 0. 0.]]\n",
      "\n",
      "one_array: \n",
      "[[1. 1. 1. 1. 1.]\n",
      " [1. 1. 1. 1. 1.]]\n",
      "\n",
      "empty_array: \n",
      "[[0. 0. 0.]\n",
      " [0. 0. 0.]]\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# 1.10. NUMPY SPECIAL CASES\n",
    "import numpy as np\n",
    "# array of integers defined by the arguments start, end and sequence length:\n",
    "sequence= np.linspace(0, 2, num=11)\n",
    "print(f'sequence: \\n{sequence}\\n')\n",
    "# sequence of integers starting at 0, ending at 5-1:\n",
    "sequence_int = np.arange(5)\n",
    "print(f'sequence_int: \\n{sequence_int}\\n')\n",
    "# initialize array with each element set to zero:\n",
    "zero_array = np.zeros((4, 3))\n",
    "print(f'zero_array: \\n{zero_array}\\n')\n",
    "# initialize array with each element set to one:\n",
    "one_array = np.ones((2, 5))\n",
    "print(f'one_array: \\n{one_array}\\n')\n",
    "# uninitialized array (filled with arbitrary nonsense elements):\n",
    "empty_array = np.empty((2, 3))\n",
    "print(f'empty_array: \\n{empty_array}\\n')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "1ce60aa9",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "resultl: \n",
      "[[5.45981500e+01 8.10308393e+03 2.98095799e+03]\n",
      " [7.38905610e+00 4.03428793e+02 2.00855369e+01]]\n",
      "\n",
      "result2: \n",
      "[[ 5 14 10]\n",
      " [ 8 12  3]]\n",
      "\n",
      "matl_tr: \n",
      "[[4 2]\n",
      " [9 6]\n",
      " [8 3]]\n",
      "\n",
      "matprod: \n",
      "[[ 90 138  32]\n",
      " [ 50  70  13]]\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# 1.11. NUMPY OPERATIONS\n",
    "import numpy as np\n",
    "# define an arrays in numpy:\n",
    "mat1 = np.array([[4, 9, 8],[2, 6, 3]])\n",
    "mat2 = np.array([[1, 5, 2],[6, 6, 0], [4, 8, 3]])\n",
    "# use a numpy function:\n",
    "result1 = np.exp(mat1)\n",
    "print(f'resultl: \\n{result1}\\n')\n",
    "result2 = mat1 + mat2[[0, 1]] # same as np.add(mat1, mat2[[0, 1]])\n",
    "print(f'result2: \\n{result2}\\n')\n",
    "# use a method:\n",
    "mat1_tr = mat1.transpose()\n",
    "print(f'matl_tr: \\n{mat1_tr}\\n')\n",
    "# matrix algebra:\n",
    "matprod = mat1.dot(mat2) # same as mat1@ mat2\n",
    "print(f'matprod: \\n{matprod}\\n')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b403661f-c197-418b-944f-eedd55790fbc",
   "metadata": {},
   "source": [
    "# Pandas Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "13af0dcf",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "df: \n",
      "            icecream_sales  weather_coded  customers\n",
      "2010-04-30              30              0       2000\n",
      "2010-05-31              40              1       2100\n",
      "2010-06-30              35              0       1500\n",
      "2010-07-31             130              1       8000\n",
      "2010-08-31             120              1       7200\n",
      "2010-09-30              60              0       2000\n",
      "\n",
      "subsetl: \n",
      "            icecream_sales  customers\n",
      "2010-04-30              30       2000\n",
      "2010-05-31              40       2100\n",
      "2010-06-30              35       1500\n",
      "2010-07-31             130       8000\n",
      "2010-08-31             120       7200\n",
      "2010-09-30              60       2000\n",
      "\n",
      "subset2: \n",
      "            icecream_sales  weather_coded  customers\n",
      "2010-05-31              40              1       2100\n",
      "2010-06-30              35              0       1500\n",
      "2010-07-31             130              1       8000\n",
      "\n",
      "subset3: \n",
      "2100\n",
      "\n",
      "subset4: \n",
      "            icecream_sales  weather_coded\n",
      "2010-05-31              40              1\n",
      "2010-06-30              35              0\n",
      "2010-07-31             130              1\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# 1.12. PANDAS\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "# define a pandas DataFrame:\n",
    "icecream_sales = np.array([30, 40, 35, 130, 120, 60])\n",
    "weather_coded = np.array([0, 1, 0, 1, 1, 0])\n",
    "customers= np.array([2000, 2100, 1500, 8000, 7200, 2000])\n",
    "df = pd.DataFrame({'icecream_sales': icecream_sales,\n",
    "    'weather_coded': weather_coded,\n",
    "    'customers': customers})\n",
    "\n",
    "# define and assign an index (six ends of month starting in April, 2010)\n",
    "# (details on generating indices are given in Chapter 10):\n",
    "ourindex = pd.date_range(start='04/2010', freq='M', periods=6)\n",
    "df.set_index(ourindex, inplace=True)\n",
    "# print the DataFrame\n",
    "print(f'df: \\n{df}\\n')\n",
    "# access columns by variable names:\n",
    "subsetl = df[['icecream_sales', 'customers']]\n",
    "print(f'subsetl: \\n{subsetl}\\n')\n",
    "# access second to fourth row:\n",
    "subset2 = df[1:4] # same as df['2010-05-31' :'2010-07-31']\n",
    "print(f'subset2: \\n{subset2}\\n')\n",
    "# access rows and columns by index and variable names:\n",
    "subset3 = df.loc['2010-05-31', 'customers'] # same as df.iloc[l,2]\n",
    "print(f'subset3: \\n{subset3}\\n')\n",
    "# access rows and columns by index and variable integer positions:\n",
    "subset4 = df.iloc[1:4, 0:2]\n",
    "# same as df.loc['2010-05-31' :'2010-07-31', ['icecream_sales', 'weather']]\n",
    "print(f'subset4: \\n{subset4}\\n')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "301680cb",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "df: \n",
      "            icecream_sales  weather_coded  customers  icecream_sales_lag2  \\\n",
      "2010-04-30              30              0       2000                  NaN   \n",
      "2010-05-31              40              1       2100                  NaN   \n",
      "2010-06-30              35              0       1500                 30.0   \n",
      "2010-07-31             130              1       8000                 40.0   \n",
      "2010-08-31             120              1       7200                 35.0   \n",
      "2010-09-30              60              0       2000                130.0   \n",
      "\n",
      "           weather  \n",
      "2010-04-30     bad  \n",
      "2010-05-31    good  \n",
      "2010-06-30     bad  \n",
      "2010-07-31    good  \n",
      "2010-08-31    good  \n",
      "2010-09-30     bad  \n",
      "\n",
      "group_means: \n",
      "         icecream_sales  weather_coded    customers  icecream_sales_lag2\n",
      "weather                                                                 \n",
      "bad           41.666667            0.0  1833.333333                 80.0\n",
      "good          96.666667            1.0  5766.666667                 37.5\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# 1.13 PANDAS OPERASTIONS\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "# define a pandas DataFrame:\n",
    "icecream_sales = np.array([30, 40, 35, 130, 120, 60])\n",
    "weather_coded = np.array([0, 1, 0, 1, 1, 0])\n",
    "customers= np.array([2000, 2100, 1500, 8000, 7200, 2000])\n",
    "df = pd.DataFrame({'icecream_sales': icecream_sales,\n",
    "    'weather_coded': weather_coded,\n",
    "    'customers': customers})\n",
    "\n",
    "# define and assign an index (six ends of month starting in April, 2010) \n",
    "# (details on generating indices are given in Chapter 10):\n",
    "ourindex = pd.date_range(start='04/2010', freq='M', periods=6) \n",
    "df.set_index(ourindex, inplace=True) \n",
    "\n",
    "# include sales two months ago:\n",
    "df['icecream_sales_lag2'] = df['icecream_sales'] .shift(2) \n",
    "\n",
    "# use a pandas.Categorical object to attach labels (0 = bad; 1 =good):\n",
    "df['weather'] = pd.Categorical.from_codes(codes=df['weather_coded'], categories=['bad', 'good']) \n",
    "\n",
    "print(f'df: \\n{df}\\n') \n",
    "\n",
    "# mean sales for each weather category:\n",
    "group_means = df.groupby('weather') .mean() \n",
    "print(f'group_means: \\n{group_means}\\n') \n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6c16ef39-e5b1-423d-b15a-b54111bc09db",
   "metadata": {},
   "source": [
    "# Woolridge data loading,import,export"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "431e8314",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "type(wage1) <class 'pandas.core.frame.DataFrame'>\n",
      "wage1.head():    wage  educ  exper  tenure  nonwhite  female  married  numdep  smsa  \\\n",
      "0  3.10    11      2       0         0       1        0       2     1   \n",
      "1  3.24    12     22       2         0       1        1       3     1   \n",
      "2  3.00    11      2       0         0       0        0       2     0   \n",
      "3  6.00     8     44      28         0       0        1       0     1   \n",
      "4  5.30    12      7       2         0       0        1       1     0   \n",
      "\n",
      "   northcen  ...  trcommpu  trade  services  profserv  profocc  clerocc  \\\n",
      "0         0  ...         0      0         0         0        0        0   \n",
      "1         0  ...         0      0         1         0        0        0   \n",
      "2         0  ...         0      1         0         0        0        0   \n",
      "3         0  ...         0      0         0         0        0        1   \n",
      "4         0  ...         0      0         0         0        0        0   \n",
      "\n",
      "   servocc     lwage  expersq  tenursq  \n",
      "0        0  1.131402        4        0  \n",
      "1        1  1.175573      484        4  \n",
      "2        0  1.098612        4        0  \n",
      "3        0  1.791759     1936      784  \n",
      "4        0  1.667707       49        4  \n",
      "\n",
      "[5 rows x 24 columns]\n"
     ]
    }
   ],
   "source": [
    "# 1.14. WOOLRIDGE\n",
    "import wooldridge as woo\n",
    "\n",
    "## LOAD DATA\n",
    "wage1 = woo.dataWoo('wage1')\n",
    "\n",
    "## GET TYPE\n",
    "print(f'type(wage1)', type(wage1))\n",
    "## GET AN OVERVIEW\n",
    "print(f'wage1.head(): {wage1.head()}')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "00e53148",
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    },
    "tags": []
   },
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (<ipython-input-40-d98bb83ebfcf>, line 3)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  File \u001b[1;32m\"<ipython-input-40-d98bb83ebfcf>\"\u001b[1;36m, line \u001b[1;32m3\u001b[0m\n\u001b[1;33m    pip install wooldridge\u001b[0m\n\u001b[1;37m        ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "# 1.15. WOOLRIDGE IMPORT-EXPORT \n",
    "import wooldridge as woo\n",
    "pip install wooldridge\n",
    "\n",
    "## LOAD DATA\n",
    "wage1 = woo.dataWoo('wage1')\n",
    "\n",
    "## GET TYPE\n",
    "print(f'type(wage1)', type(wage1))\n",
    "## GET AN OVERVIEW\n",
    "print(f'wage1.head(): {wage1.head()}')\n",
    "\n",
    "## IMPORT- EXPORT CSV WITH PANDAS\n",
    "df1 = pandas.read_csv('data/sales.csv', delimiter= \",\", header=None, names=['year', 'product1', 'product2', 'product3'])\n",
    "print(f'{df1:df1}')\n",
    "\n",
    "## EXPORT TXT WITH PANDAS\n",
    "# df3.to_csv('data/sales2.csv')\n",
    "\n",
    "# add a row to dfl:\n",
    "df3 = dfl.append({'year': 2014, 'productl': 10, 'product2': 8, 'product3': 2},\n",
    "ignore_index=True)\n",
    "print(f'df3: \\n{df3}\\n')\n",
    "\n",
    "# export with pandas:\n",
    " df3.to_csv('data/sales2.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "aa9ad19f",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import pandas_datareader as pdr \n",
    "# download data for 'F' (= Ford Motor Company) and define start and end:\n",
    "tickers = ['F']\n",
    "start_date = '2014-01-01' \n",
    "end_date = '2015-12-31' "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fa21a443-0daa-4330-ae48-3be9dd5787f8",
   "metadata": {},
   "source": [
    "# Web reading and analyzing "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "df93610f-eccf-46a6-97b8-f4ba999896ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "import stats\n",
    "import pandas as pd\n",
    "import requests\n",
    "import json\n",
    "import requests\n",
    "import pandasdmx\n",
    "import cache"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "9e4a65bf",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import requests\n",
    "from bs4 import BeautifulSoup\n",
    "import textwrap"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "52e9f680",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import requests\n",
    "\n",
    "url = \"https://api.ons.gov.uk/timeseries/JP9Z/dataset/UNEM/data\"\n",
    "\n",
    "# Get the data from the ONS API:\n",
    "json_data = requests.get(url).json()\n",
    "\n",
    "# Prep the data for a quick plot\n",
    "title = json_data[\"description\"][\"title\"]\n",
    "df = (\n",
    "    pd.DataFrame(pd.json_normalize(json_data[\"months\"]))\n",
    "    .assign(\n",
    "        date=lambda x: pd.to_datetime(x[\"date\"]),\n",
    "        value=lambda x: pd.to_numeric(x[\"value\"]),\n",
    "    )\n",
    "    .set_index(\"date\")\n",
    ")\n",
    "\n",
    "df[\"value\"].plot(title=title, ylim=(0, df[\"value\"].max() * 1.2), lw=3.0);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dcf2d502",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import pandas_datareader.data as web\n",
    "\n",
    "df_u = web.DataReader(\"LRHUTTTTGBM156S\", \"fred\")\n",
    "\n",
    "df_u.plot(title=\"UK unemployment (percent)\", legend=False, ylim=(2, 6), lw=3.0);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "31ab5bb5",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# World Bank CO2 emissions (metric tons per capita)\n",
    "# https://data.worldbank.org/indicator/EN.ATM.CO2E.PC\n",
    "# World Bank pop\n",
    "# https://data.worldbank.org/indicator/SP.POP.TOTL\n",
    "# country and region codes at http://api.worldbank.org/v2/country\n",
    "from pandas_datareader import wb\n",
    "df = wb.download(\n",
    "    indicator=\"EN.ATM.CO2E.PC\",\n",
    "    country=[\"US\", \"CHN\", \"IND\", \"Z4\", \"Z7\"],\n",
    "    start=2017,\n",
    "    end=2017,\n",
    ")\n",
    "# remove country as index for ease of plotting with seaborn\n",
    "df = df.reset_index()\n",
    "# wrap long country names\n",
    "df[\"country\"] = df[\"country\"].apply(lambda x: textwrap.fill(x, 10))\n",
    "# order based on size\n",
    "df = df.sort_values(\"EN.ATM.CO2E.PC\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3ee8ddc0",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import seaborn as sns\n",
    "\n",
    "fig, ax = plt.subplots()\n",
    "sns.barplot(x=\"country\", y=\"EN.ATM.CO2E.PC\", data=df.reset_index(), ax=ax)\n",
    "ax.set_title(r\"CO$_2$ (metric tons per capita)\", loc=\"right\")\n",
    "plt.suptitle(\"The USA leads the world on per-capita emissions\", y=1.01)\n",
    "for key, spine in ax.spines.items():\n",
    "    spine.set_visible(False)\n",
    "ax.set_ylabel(\"\")\n",
    "ax.set_xlabel(\"\")\n",
    "ax.yaxis.tick_right()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6187b5f6",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import pandasdmx as pdmx\n",
    "# Tell pdmx we want OECD data\n",
    "oecd = pdmx.Request(\"OECD\")\n",
    "# Set out everything about the request in the format specified by the OECD API\n",
    "data = oecd.data(\n",
    "    resource_id=\"PDB_LV\",\n",
    "    key=\"GBR+FRA+CAN+ITA+DEU+JPN+USA.T_GDPEMP.CPC/all?startTime=2010\",\n",
    ").to_pandas()\n",
    "\n",
    "df = pd.DataFrame(data).reset_index()\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b8e25ad9-3511-46d9-8b55-8953505712aa",
   "metadata": {},
   "source": [
    "# Graphs basics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cd591e88",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1.17. GRAPHS-BASICS1\n",
    "import matplotlib\n",
    "import matplotlib.pyplot as plt\n",
    "# create data:\n",
    "x = [1, 3, 4, 7, 8, 9]\n",
    "y = [0, 3, 6, 9, 7, 8]\n",
    "# plot and save:\n",
    "plt.plot(x, y, color='black')\n",
    "# plt.savefig('PyGraphs/Graphs-Basics-a.pdf')\n",
    "plt . close ()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9507b14a",
   "metadata": {},
   "outputs": [],
   "source": [
    "x = [1, 3, 4, 7, 8, 9]\n",
    "y = [0, 3, 6, 9, 7, 8]\n",
    "# plot and save:\n",
    "plt.plot(x, y, color='black')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "76238157",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1.18 GRAPHS-BASICS2\n",
    "import matplotlib\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# DATA\n",
    "# x1; x2\n",
    "\n",
    "# BASIC 1\n",
    "x = [1,3,4,7,8,9]\n",
    "y = [0,3,6,9,7,8]\n",
    "plt.plot(x,y, color='blue')\n",
    "# plt.savefig('PyGraphs/Graphs-Basic-a.pdf')\n",
    "# plt.close()\n",
    "\n",
    "# BASIC 2\n",
    "plt.plot(x,y, color='black',linestyle= '--')\n",
    "# plt.savefig('PyGraphs/Graphs-Basic-b.pdf')\n",
    "# plt.close()\n",
    "\n",
    "plt.plot(x,y, color= 'black',linestyle= ':')\n",
    "# plt.savefig('PyGraphs/Graphs-Basics-c.pdf')\n",
    "# plt.close()\n",
    "\n",
    "plt.plot(x,y,color='black',linestyle= '-', linewidth=3)\n",
    "plt.close()\n",
    "\n",
    "plt.plot(x,y, color='black', marker='o')\n",
    "# plt.savefig('PyGraphs/Graphs-Basic-e.pdf')\n",
    "# plt.close()\n",
    "\n",
    "plt.plot(x,y, color='black', marker='v', linestyle='')\n",
    "# plt.savefig('PyGraphs/Graphs-Basics-f.pdf')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6b85d8f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1.20 GRAPHS BUILDINGS BLOCKS\n",
    "# Same for normality density\n",
    "import numpy as np\n",
    "import stats\n",
    "\n",
    "x = np.linspace(-4,4,num=100)\n",
    "# Get different density evaluations\n",
    "y1 = stats.norm.pdf(x,0,1)\n",
    "y2 = stats.norm.pdf(x,0,3)\n",
    "y3 = stats.norm.pdf(x,0,2)\n",
    "# Plot normal density\n",
    "plt.plot(x,y1,linestyle='-',color='black', label= 'standard normal')\n",
    "plt.plot(x,y2, linestyle='--', color='0.3', label='mu=1,sigma=0.5')\n",
    "plt.plot(x,y3,linestyle=':',color='0.6', label='mu=0, sigma=2')\n",
    "plt.xlim(-3,4)\n",
    "plt.title('Normal Densities')\n",
    "plt.ylabel('$\\phi(x)$')\n",
    "plt.xlabel('x')\n",
    "plt.legend()\n",
    "# plt.savefig('PyGraphs/Graphs-BuildingsBlocks.pdf')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2a49a425",
   "metadata": {
    "tags": [
     "Not",
     "allowed",
     "to",
     "conduct",
     "the",
     "alogritm"
    ]
   },
   "outputs": [],
   "source": [
    "# GRAPHS EXPORT\n",
    "# Support same for normality density\n",
    "x = np.linspace(-4,4,num=100)\n",
    "# Get different density evaluations\n",
    "y1 = stats.norm.pdf(x,0,1)\n",
    "y2 = stats.norm.pdf(x,0,3)\n",
    "\n",
    "# Plot (a)\n",
    "plt.figure(figsize=(4,6))\n",
    "plt.plot(x,y1,linestyle= '-', color='black')\n",
    "plt.plot(x,y2,linestyle='--',color='0.3')\n",
    "# Export\n",
    "plt.savefig('PyGraphs/Graphs-Export-a.pdf')\n",
    "plt.close()\n",
    "\n",
    "# Plot(b)\n",
    "plt.figure(figsize=(6, 4))\n",
    "plt.plot(x, yl, linestyle='-', color='black')\n",
    "plt.plot(x, y2, linestyle='--', color='0.3')\n",
    "plt.savefig('PyGraphs/Graphs-Export-b.png')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "33f8cbd6-2f8d-438a-b5c5-4b443d05ec47",
   "metadata": {},
   "source": [
    "# Describe statistics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2d7bd3c0",
   "metadata": {
    "tags": [
     "1.20"
    ]
   },
   "outputs": [],
   "source": [
    "# DESCRIPTIVE STATISTICS\n",
    "# 1.22. Describes Tables of py\n",
    "import wooldridge as woo\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "affairs= woo.dataWoo('affairs')\n",
    "# Adjust codings to [0-4] catergoricals require a start from 0\n",
    "affairs['haskids'] = affairs['ratemarr'] - 1\n",
    "\n",
    "# Use pandas categorical to attach labels for \"haskids\"\n",
    "affairs['ratemarr']= affairs['ratemarr'] - 1\n",
    "\n",
    "# Use pandas categorical object to attach labels for 'haskids'\n",
    "affairs['haskids']= pd.Categorical.from_codes(affairs['kids'],categories= ['no','yes'])\n",
    "\n",
    "# and \"marriage\" # for example: 0='very unhappy', 1='unhappy'\n",
    "mlab=['very unhappy','unhappy','average','happy','very happy']\n",
    "affairs['marriage']= pd.Categorical.from_codes(affairs['ratemarr'],categories=mlab)\n",
    "\n",
    "# frequencies table in numpy (alphabetical order of elements):\n",
    "ft_np = np.unique(affairs['marriage'], return_counts=True)\n",
    "unique_elem_np= ft_np[0]\n",
    "counts_np = ft_np[1]\n",
    "print(f'unique_elem_np: {unique_elem_np}')\n",
    "print(f'counts_np: {counts_np}')\n",
    "\n",
    "# frequency table in pandas\n",
    "ft_pd = affairs['marriage'].value_counts()\n",
    "print(f'ft_pd: {ft_pd}')\n",
    "\n",
    "# frequency table with groupby\n",
    "ft_pd2 = affairs['marriage'].groupby(affairs['haskids']).value_counts()\n",
    "print(f'ft_pd2: {ft_pd2}')\n",
    "\n",
    "# contingency table in pandas\n",
    "ct_all_abs = pd.crosstab(affairs['marriage'],affairs['haskids'], margins=3)\n",
    "print(f'ct_all_abs:{ct_all_abs}')\n",
    "ct_all_rel = pd.crosstab(affairs['marriage'], affairs['haskids'], normalize='all')\n",
    "print(f'ct_all_rel: {ct_all_rel}')\n",
    "\n",
    "# share within \"marriage\"\n",
    "ct_row = pd.crosstab(affairs['marriage'],affairs['haskids'], normalize='index')\n",
    "print(f'ct_row = {ct_row}')\n",
    "\n",
    "# share within \"haskids\", within a column:\n",
    "ct_col = pd.crosstab(affairs['marriage'], affairs['haskids'], normalize='index')\n",
    "print(f'ct_col: {ct_col}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8a74b65b",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# 1.24. HISTOGRAM\n",
    "import wooldridge as woo\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "ceosal1 = woo.dataWoo('ceosal1')\n",
    "# extract roe:\n",
    "roe= ceosal1['roe']\n",
    "# subfigure a (histogram with counts):\n",
    "plt.hist(roe, color='grey')\n",
    "plt.ylabel('Counts')\n",
    "plt.xlabel('roe')\n",
    "# plt.savefig('PyGraphs/Histogram1.pdf')\n",
    "# plt. close ()\n",
    "\n",
    "# subfigure b (histogram with density and explicit breaks):\n",
    "breaks= [0, 5, 10, 20, 30, 60]\n",
    "plt.hist(roe, color='grey', bins=breaks, density=True)\n",
    "plt.ylabel('density')\n",
    "plt.xlabel('roe')\n",
    "# plt.savefig('PyGraphs/Histogram2.pdf')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2bb20ea8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1.26: Descr-ECDF.\n",
    "import wooldridge as woo\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "ceosal1 = woo.dataWoo('ceosal1')\n",
    "# extract roe:\n",
    "roe= ceosal1['roe']\n",
    "# calculate ECDF:\n",
    "x np.sort (roe)\n",
    "n x.size\n",
    "y np.arange(l, n + 1) /n # generates cumulative shares of observations\n",
    "\n",
    "# plot a step function:\n",
    "plt.step(x, y, linestyle='-', color='black')\n",
    "plt.xlabel('roe')\n",
    "# plt.savefig('PyGraphs/ecdf.pdf')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "68856743",
   "metadata": {},
   "outputs": [],
   "source": [
    "import wooldridge as woo\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "ceosal1 = woo.dataWoo('ceosal1')\n",
    "# extract roe:\n",
    "roe= ceosal1['roe']\n",
    "# calculate ECDF:\n",
    "x np.sort (roe)\n",
    "n x.size\n",
    "y np.arange(l, n + 1) /n # generates cumulative shares of observations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c3902a7a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib\n",
    "import wooldridge as woo\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "\n",
    "ceosal1 = woo.dataWoo('ceosal1')\n",
    "# extract roe:\n",
    "roe= ceosal1['roe']\n",
    "# calculate ECDF:\n",
    "x = np.sort (roe)\n",
    "n = x.size\n",
    "y = np.arange(1, n + 1) /n # generates cumulative shares of observations\n",
    "\n",
    "ceosal1 = woo.dataWoo('ceosal1')\n",
    "# extract roe and salary:\n",
    "roe= ceosal1['roe']\n",
    "salary= ceosal1['salary']\n",
    "# sample average:\n",
    "roe_mean = np.mean(salary)\n",
    "print(f'roe_mean: {roe_mean}')\n",
    "# sample median:\n",
    "roe_median = np.median(salary)\n",
    "print(f'roe_median: {roe_median}')\n",
    "# standard deviation:\n",
    "roe_s = np.std(salary, ddof=1)\n",
    "print(f'roe_s: {roe_s}')\n",
    "# correlation with ROE:\n",
    "roe_corr = np.corrcoef(roe, salary)\n",
    "print(f'roe_corr: {roe_corr}')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f8a0187d-1319-48b9-9717-b198c40be5cf",
   "metadata": {},
   "source": [
    "# Describe boxplot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "8c35b894",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'roe')"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAX4AAAD4CAYAAADrRI2NAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAQjklEQVR4nO3dX2hk53nH8e8TRUbF+Wdh2Sxx4t0LE6YMxAElBLwUFHeD3ZbYBefPQsOmHeqbdkhooXU6F3GgY3y1NOjG3VahW5KdODgJNklo6u5OCWOMba3rpOsq7aZx3BgbrxK7OHGQo6hPL3a01v6ztV6dOat5vx8YzpxXc/Y8Aumnd99zzvtGZiJJKseb6i5AkjRaBr8kFcbgl6TCGPySVBiDX5IK8+a6C9iMK6+8Mnfu3Fl3GZK0rRw9evSnmTlzZvu2CP6dO3eyuLhYdxmStK1ExNPnaneoR5IKY/BLUmEMfkkqjMEvSYUx+CWpMAZ/QXq9Hs1mk4mJCZrNJr1er+6SJNVgW9zOqYvX6/XodDosLCywe/duBoMBrVYLgL1799ZcnaRRiu0wLfPs7Gx6H//FaTabzM/PMzc3d6qt3+/Tbrc5duxYjZVJqkpEHM3M2bPaDf4yTExMsLKywuTk5Km21dVVpqamWFtbq7EySVU5X/A7xl+IRqPBYDA4rW0wGNBoNGqqSFJdDP5CdDodWq0W/X6f1dVV+v0+rVaLTqdTd2mSRsyLu4VYv4DbbrdZWlqi0WjQ7Xa9sCsVyDF+SRpTjvFLkgCDX5KKY/BLUmEMfkkqjMEvSYUx+CWpMAa/JBXG4Jekwhj8klQYg1+SCmPwS1JhDH5JKozBL0mFMfglqTCVzscfET8Gfg6sAb/OzNmImAbuBXYCPwY+lpkvVlmHJOlVo+jxz2Xm9RvmhL4DOJyZ1wGHh/uSpBGpY6jnFuDg8P1B4NYaapCkYlUd/An8c0QcjYjbh21XZ+ZzAMPtVRXXIEnaoOo1d2/IzGcj4irgwYj4wWYPHP6huB3g3e9+d1X1SVJxKu3xZ+azw+0J4BvAB4DnI2IHwHB74jzHHsjM2cycnZmZqbJMSSpKZcEfEZdHxFvX3wMfBo4BDwD7hh/bB9xfVQ2SpLNV2eO/GhhExPeAR4FvZeY/AXcDeyLiOLBnuC+pYL1ej2azycTEBM1mk16vV3dJY62yMf7M/BHw3nO0/wy4sarzStpeer0enU6HhYUFdu/ezWAwoNVqAbB3796aqxtPkZl11/C6Zmdnc3Fxse4yJFWg2WwyPz/P3NzcqbZ+v0+73ebYsWM1Vrb9RcTRDc9Qvdpu8Euq08TEBCsrK0xOTp5qW11dZWpqirW1tRor2/7OF/zO1SOpVo1Gg8FgcFrbYDCg0WjUVNH4M/gl1arT6dBqtej3+6yurtLv92m1WnQ6nbpLG1tVP8AlSa9p/QJuu91maWmJRqNBt9v1wm6FHOOXpDHlGL8kCTD4Jak4Br8kFcbgl6TCGPySVBiDX1Lt2u02U1NTRARTU1O02+26SxprBr+kWrXbbe655x7uuusuXn75Ze666y7uuecew79C3scvqVZTU1PcdtttPPHEE6ce4Lr++uu57777WFlZqbu8bc37+CVdkl555RUeeugh5ufnWVlZYX5+noceeohXXnml7tLGlsEvqVYRwc0338zc3ByTk5PMzc1x8803ExF1lza2DH5JtTtw4AD79+/nl7/8Jfv37+fAgQN1lzTWHOOXNHJvpDe/HbLqUuMYv6RLRmaeeh06dIhdu3Zx5MgRAI4cOcKuXbs4dOjQaZ/T1nFaZkm12jgt8/rWaZmr5VCPpEtGRNi730IO9UiSAINfkopj8EtSYQx+SSqMwS9JhTH4JakwlQd/RExExL9FxDeH+9MR8WBEHB9ur6i6BknSq0bR4/80sLRh/w7gcGZeBxwe7kuSRqTS4I+Ia4DfBf5+Q/MtwMHh+4PArVXWIEk6XdU9/r8B/gL4vw1tV2fmcwDD7VXnOjAibo+IxYhYXF5errhMSSpHZcEfEb8HnMjMo2/k+Mw8kJmzmTk7MzOzxdVJUrmq7PHfAHwkIn4MfAX4UER8CXg+InYADLcnKqxBG/R6PZrNJhMTEzSbTXq9Xt0lSapBZcGfmZ/NzGsycyfwCeBIZv4B8ACwb/ixfcD9VdWgV/V6PTqdzmnL23U6HcNfKlAd9/HfDeyJiOPAnuG+KtbtdllYWDhtebuFhQW63W7dpUkaMadlLsTExAQrKytMTk6ealtdXWVqaoq1tbUaK5Ne5bTMW8tpmQvXaDQYDAantQ0GAxqNRk0VSaqLwV+ITqdDq9Wi3++zurpKv9+n1WrR6XTqLk3SiLn0YiE2Lm+3tLREo9FweTupUI7xS7pkOMa/tRzjlyQBBr8kFcfgl6TCGPySVBiDX5IKY/BLUmEMfkkqjMEvSYUx+CWpMAa/JBXG4Jekwhj8BXHpRUng7JzFWF96cWFhgd27dzMYDGi1WgDO0CkVxh5/IVx6UdI6p2UuhEsvajtwWuat5bTMhXPpRUnrDP5CuPSipHVe3C2ESy9KWucYv6RLhmP8W8sxfkkSYPBLUnEMfkkqTGXBHxFTEfFoRHwvIp6MiM8P26cj4sGIOD7cXlFVDZKks1XZ438F+FBmvhe4HrgpIj4I3AEczszrgMPDfUnSiFQW/HnSL4a7k8NXArcAB4ftB4Fbq6pBknS2Ssf4I2IiIp4ATgAPZuYjwNWZ+RzAcHvVeY69PSIWI2JxeXm5yjIlqSibDv6I2B0Rfzh8PxMRu17vmMxcy8zrgWuAD0REc7Pny8wDmTmbmbMzMzObPUyS9Do2FfwR8TngL4HPDpsmgS9t9iSZ+b/AvwI3Ac9HxI7hv7uDk/8bkCSNyGZ7/L8PfAR4GSAznwXe+loHDP9X8I7h+98Afhv4AfAAsG/4sX3A/RdctSTpDdts8P8qTz5HnQARcfkmjtkB9CPi+8BjnBzj/yZwN7AnIo4De4b7GgFX4JIEm5+k7asR8bfAOyLij4E/Av7utQ7IzO8D7ztH+8+AGy+0UF0cV+CStG7Tk7RFxB7gw0AA38nMB6ssbCMnabt4zWaT+fl55ubmTrX1+33a7TbHjh2rsTLpVU7StrXON0nbhQT/1cD7h7uPZubILsoa/BfPFbi0HRj8W+uiZueMiI8BjwIfBT4GPBIRt21tiaqSK3BJWrfZMf4O8P71Xn5EzAD/AtxXVWHaWp1Oh49//ONcfvnlPP3001x77bW8/PLLfOELX6i7NEkjttm7et50xtDOzy7gWF1iIqLuEiTV6HXDO06mxGMR8Z2I+FREfAr4FvDtqovT1ul2u9x777089dRTrK2t8dRTT3HvvffS7XbrLk1janp6moi4oBdwwcdMT0/X/J1uP5u6uBsRjwN/Dezm5F09383Mb1Rc2yle3L14XtzVqI3qQq0XhM/vfBd3NzvG/zDwk8z8s60tS1U513DOZZdd9rqf9RdIGn+bHaefAx6OiP+OiO+vv6osTBcnM097HTp0iF27dnHkyBEAjhw5wq5duzh06NBpn5M0/jbb47+50ipUufWnc9vt9qltt9v1qV2pQJt+gKtOjvFvLcdENQqO8dfvoh7gkiSND4Nfkgpj8EtSYQx+SSqMwS9JhTH4JakwBr8kFcbgl6TCGPySVBiDX5IKY/BLUmEMfkkqjMEvSYUx+CWpMJudj1+SLkh+7m1w59tHcx5dEINfUiXi8y+Nbj7+Oys/zVipbKgnIt4VEf2IWIqIJyPi08P26Yh4MCKOD7dXVFWDJOlsVY7x/xr488xsAB8E/iQifhO4AzicmdcBh4f7kqQRqSz4M/O5zHx8+P7nwBLwTuAW4ODwYweBW6uqQZJ0tpHc1RMRO4H3AY8AV2fmc3DyjwNw1XmOuT0iFiNicXl5eRRlSlIRKg/+iHgL8DXgM5n50maPy8wDmTmbmbMzMzPVFShJhak0+CNikpOh/+XM/Pqw+fmI2DH8+g7gRJU1SJJOV+VdPQEsAEuZuX/Dlx4A9g3f7wPur6oGSdLZqryP/wbgk8C/R8QTw7a/Au4GvhoRLeB/gI9WWIMk6QyVBX9mDoA4z5dvrOq8kqTX5lw9klQYg1+SCmPwS1JhDP5tbnp6moi4oBdwQZ+fnp6u+buUtJWcnXObe/HFFyufAXH9j4Wk8WCPX5IKY/BLUmEMfkkqjMEvSYXx4q6kyozixoArrnARvwtl8EuqxBu52ywiRrJOb+kc6pGkwhj8klQYg1+SCuMY/zaXn3sb3Pn26s8haWwY/NtcfP6lkUzZkHdWegpJI+RQjyQVxuCXpMIY/JJUGINfkgpj8EtSYQx+SSqMwS9JhTH4JakwBr8kFcYnd8dA1XOeO9+5NF4q6/FHxBcj4kREHNvQNh0RD0bE8eHWRLlImXnBrws97oUXXqj5u5S0laoc6vkH4KYz2u4ADmfmdcDh4b4kaYQqC/7M/C5wZlfxFuDg8P1B4Naqzi9JOrdRX9y9OjOfAxhurzrfByPi9ohYjIjF5eXlkRUoSePukr2rJzMPZOZsZs7OzMzUXY4kjY1RB//zEbEDYLg9MeLzS1LxRh38DwD7hu/3AfeP+PySVLwqb+fsAQ8D74mIZyKiBdwN7ImI48Ce4b4kaYQqe4ArM/ee50s3VnVOSdLru2Qv7kqSqmHwS1JhDH5JKozBL0mFMfglqTAGvyQVxuCXpMIY/JJUGINfkgpj8EtSYQx+SSqMwS9JhTH4JakwBr8kFcbgl6TCGPySVBiDX5IKY/BLUmEMfkkqjMEvSYUx+CWpMAa/JBXmzXUXIKk8EXHBX8vMqsopjsE/pl7rF+u1vu4vl0bBn7N6Gfxjyl8sSefjGH9Ber0ezWaTiYkJms0mvV6v7pIk1cAefyF6vR6dToeFhQV2797NYDCg1WoBsHfv3pqrkzRKtfT4I+KmiPjPiPhhRNxRRw2l6Xa7LCwsMDc3x+TkJHNzcywsLNDtdusuTdKIxajHgiNiAvgvYA/wDPAYsDcz/+N8x8zOzubi4uKIKhxPExMTrKysMDk5eaptdXWVqakp1tbWaqxMUlUi4mhmzp7ZXkeP/wPADzPzR5n5K+ArwC011FGURqPBYDA4rW0wGNBoNGqqSFJd6gj+dwI/2bD/zLDtNBFxe0QsRsTi8vLyyIobV51Oh1arRb/fZ3V1lX6/T6vVotPp1F2apBGr4+LuuW4gP2u8KTMPAAfg5FBP1UWNu/ULuO12m6WlJRqNBt1u1wu7UoHqCP5ngHdt2L8GeLaGOoqzd+9eg15SLUM9jwHXRcSuiLgM+ATwQA11SFKRRt7jz8xfR8SfAt8BJoAvZuaTo65DkkpVywNcmflt4Nt1nFuSSueUDZJUGINfkgoz8id334iIWAaerruOMXIl8NO6i5DOwZ/NrXVtZs6c2bgtgl9bKyIWz/UYt1Q3fzZHw6EeSSqMwS9JhTH4y3Sg7gKk8/BncwQc45ekwtjjl6TCGPySVBiDvyAR8cWIOBERx+quRdooIt4VEf2IWIqIJyPi03XXNM4c4y9IRPwW8AvgHzOzWXc90rqI2AHsyMzHI+KtwFHg1tdaklVvnD3+gmTmd4EX6q5DOlNmPpeZjw/f/xxY4hwr82lrGPySLikRsRN4H/BIzaWMLYNf0iUjIt4CfA34TGa+VHc948rgl3RJiIhJTob+lzPz63XXM84Mfkm1i4gAFoClzNxfdz3jzuAvSET0gIeB90TEMxHRqrsmaegG4JPAhyLiieHrd+oualx5O6ckFcYevyQVxuCXpMIY/JJUGINfkgpj8EtSYQx+SSqMwS9Jhfl/HUq2CEZUS4UAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# 1.28 Descrip-Boxplot\n",
    "import wooldridge as woo\n",
    "import matplotlib.pyplot as plt\n",
    "ceosal1 = woo.dataWoo('ceosal1')\n",
    "# extract roe and salary:\n",
    "roe= ceosal1['roe']\n",
    "consprod = ceosal1['consprod']\n",
    "# plotting descriptive statistics:\n",
    "plt.boxplot(roe, vert=False)\n",
    "plt.ylabel('roe')\n",
    "# plt.savefig('PyGraphs/Boxplot1.pdf')\n",
    "plt.close()\n",
    "# plotting descriptive\n",
    "roe_cp0 = roe[consprod == 0]\n",
    "roe_cp1 = roe[consprod == 1]\n",
    "plt.boxplot([roe_cp0, roe_cp1])\n",
    "plt.ylabel('roe')\n",
    "# plt.savefig('PyGraphs/Boxplot2.pdf')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "9b28f521",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "p1: 0.3019898880000002\n",
      "p2: 0.301989888\n"
     ]
    }
   ],
   "source": [
    "# 1.29 PMF-Binom\n",
    "import scipy\n",
    "import scipy.stats as stats\n",
    "import math\n",
    "# pedestrian approach:\n",
    "c = math.factorial(10) / (math.factorial(2) * math.factorial(10 - 2))\n",
    "p1 = c*(0.2**2)*(0.8**8)\n",
    "print(f'p1: {p1}')\n",
    "# scipy function:\n",
    "p2 = stats.binom.pmf(2, 10, 0.2)\n",
    "print(f'p2: {p2}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "6b000634",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "result: \n",
      "        x            fx\n",
      "0    0.0  1.073742e-01\n",
      "1    1.0  2.684355e-01\n",
      "2    2.0  3.019899e-01\n",
      "3    3.0  2.013266e-01\n",
      "4    4.0  8.808038e-02\n",
      "5    5.0  2.642412e-02\n",
      "6    6.0  5.505024e-03\n",
      "7    7.0  7.864320e-04\n",
      "8    8.0  7.372800e-05\n",
      "9    9.0  4.096000e-06\n",
      "10  10.0  1.024000e-07 \n",
      "\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'fx \\n ')"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# 1.30. PMF Example\n",
    "import scipy.stats as stats\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# values for x (all between O and 10):\n",
    "x = np.linspace(0, 10, num=11)\n",
    "# PMF for all these values:\n",
    "fx = stats.binom.pmf(x, 10, 0.2)\n",
    "# collect values in DataFrame:\n",
    "result= pd.DataFrame({'x':x, 'fx':fx})\n",
    "print(f'result: \\n {result} \\n')\n",
    "# plot:\n",
    "plt.bar(x, fx, color='0.6')\n",
    "plt.ylabel (' x \\n' )\n",
    "plt.ylabel ('fx \\n ')\n",
    "# plt.savefig('PyGraphs/PMF-example.pdf')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "03b5ddf9-8d26-4d4c-9617-b32cd0d2a3eb",
   "metadata": {},
   "source": [
    "# Distibution (PDF, CDF)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "e6c059f2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'dx')"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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wGoR2QJsABao6mJvZTWe7dy5k7i4OJo3b05SUhKhoaGuI5kgU7FiRebMmUO9evXo168f+/btcx0pqNiBcsajvXv3cuutt3Lu3DnWrFlDs2bNXEcyQSwtLY1bbrmFmjVrsnLlSurVq+c6UsCwczGZEjlw4ADdunXj1KlTLFu2zMrBONeqVSsWL17MwYMH6d69O7bHYvmwgjA/c/DgQbp160ZmZiaLFy+mTZs2riMZA0CnTp344IMP2LNnD3fffTdHjhxxHSngWUGY8/bu3cstt9zCvn37+OCDD+jY0XYcM77ltttuIzk5mbS0NLp06UJGRobrSAHNCsIAsGXLFm666SYOHz7Mhx9+yG233eY6kjEe9ezZk0WLFrF//35uvvlmdu7c6TpSwLKCMCxbtowuXboQFhbG6tWrufHGG11HMuai7rjjDj7++GPOnDnDzTffzOrVq11HCkhWEEEsNzeX8ePH06NHDxo1asTatWuJji56PkVjfFOHDh1Ys2YNNWrUoGvXrrz00ksE0l6ZvsAKIkgdPXqUAQMG8Oc//5n77ruP9evX07Bhw0sPNMaHNGvWjI0bN9K7d28eeeQRBg8ezIkTJ1zHChhWEEFo8eLFtG7dmgULFjBx4kSmT59ORESE61jGlEr16tVJTk7m+eefZ8aMGbRp04aPP/7YdayAYAURRI4ePcqIESPo2bMn1atXZ+3atTzyyCOIeDpnojH+IyQkhD/96U988sknhIaG0rVrV8aMGWNrE5fJCiIIZGdnnz8L69SpU3niiSfYtGkTN9xwg+toxpSpW265hS1btvDII4/wyiuv0KJFC9566y1ycnJcR/NLVhABLDc3l7lz59K2bVseeughWrVqRWpqKuPHj6dSpUqu4xnjFVWqVGHixImsW7eOq6++muHDhxMTE8PixYttI3YJWUEEoOzsbN577z3atm1L//79OXv2LMnJyXz00Ue0b9/edTxjykWnTp1Yu3Yt77//Pj/++CM9e/YkJiaGOXPm2EWIiskKIoDs37+fp59+miZNmjB48GBUlWnTprFjxw5iY2NtW4MJOiJCQkICu3bt4s033+Snn34iLi6Oa665hueee44DB4peosYUZmdz9XM//PAD//nPf5g1axbLly9HVbnrrrt46KGH6NOnDyEh9hnAmAI5OTmkpKTw6quvsmLFCkJDQ7n77ruJj4/n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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# 1.31. PDF Example\n",
    "import scipy; import numpy; import matplotlib\n",
    "import scipy.stats as stats\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# support of normal density:\n",
    "x_range = np.linspace(-4, 4, num=100)\n",
    "# PDF for all these values:\n",
    "pdf = stats.norm.pdf(x_range)\n",
    "# plot:\n",
    "plt.plot(x_range, pdf, linestyle='-', color='black')\n",
    "plt.xlabel ('x' )\n",
    "plt.ylabel('dx')\n",
    "# plt.savefig('PyGraphs/PDF-example.pdf')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "c6a2ff49",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "p1: 0.8791261184000001\n",
      "p2: 0.950004209703559\n"
     ]
    }
   ],
   "source": [
    "# 1.32. CDF Examples\n",
    "import scipy\n",
    "import scipy.stats as stats\n",
    "# binomial CDF:\n",
    "p1= stats.binom.cdf(3, 10, 0.2)\n",
    "print(f'p1: {p1}')\n",
    "# normal CDF:\n",
    "p2 = stats.norm.cdf(1.96) - stats.norm.cdf(-1.96)\n",
    "print(f'p2: {p2}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "a33b6223",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "p1_1: 0.4950149249061542 \n",
      "\n",
      "p1_2: 0.4950149249061542 \n",
      " \n",
      "p2  : 0.7702575944012563 \n",
      "\n"
     ]
    }
   ],
   "source": [
    "# 1.33. Example B-6\n",
    "import scipy\n",
    "import scipy.stats as stats\n",
    "# first example using the transformation:\n",
    "p1_1 = stats.norm.cdf(2/3) - stats.norm.cdf(-2/3)\n",
    "print(f'p1_1: {p1_1} \\n')\n",
    "# first example working directly with the distribution of X:\n",
    "p1_2 = stats.norm.cdf(6, 4, 3) - stats.norm.cdf(2, 4, 3)\n",
    "print(f'p1_2: {p1_2} \\n ')\n",
    "# second example:\n",
    "p2 = 1 - stats.norm.cdf(2, 4, 3) + stats.norm.cdf(-2, 4, 3)\n",
    "print(f'p2  : {p2} \\n')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "99cc5e48",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Fx')"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# 1.34. CDF-Figure\n",
    "import scipy; import numpy; import matplotlib; import stats\n",
    "import scipy.stats as stats\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# binomial:\n",
    "# support of binomial PMF:\n",
    "x_binom = np.linspace(-1, 10, num=1000)\n",
    "# PMF for all these values:\n",
    "cdf_binom = stats.binom.cdf(x_binom, 10, 0.2)\n",
    "\n",
    "# plot:\n",
    "plt.step(x_binom, cdf_binom, linestyle='-', color='black')\n",
    "plt. xlabel ('x')\n",
    "plt.ylabel('Fx')\n",
    "# plt.savefig('PyGraphs/CDF-figure-discrete.pdf')\n",
    "plt. close ()\n",
    "# normal:\n",
    "# support of normal density:\n",
    "x_norm = np.linspace(-4, 4, num=1000)\n",
    "# PDF for all these values:\n",
    "cdf_norm = stats.norm.cdf(x_norm)\n",
    "# plot:\n",
    "plt.plot(x_norm, cdf_norm, linestyle='-', color='black')\n",
    "plt. xlabel ('x')\n",
    "plt.ylabel('Fx')\n",
    "# plt.savefig('PyGraphs/CDF-figure-cont.pdf')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "f5742fa3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "q_975: 1.959963984540054\n",
      "sample: [1 0 0 1 1 0 0 1 1 1]\n",
      "sample: [-1.05007909 -0.93027577 -0.4695343   2.6199082  -0.8982373   0.15039921\n",
      " -0.82278071 -0.17243918 -2.73057491 -0.18747028]\n",
      "sample1: [-1.53304676  0.76580811  0.174952    0.18068228  2.53930675]\n",
      "sample2: [ 0.53472883  0.75378072  0.47318682 -1.33773586 -1.36449864]\n",
      "sample3: [ 1.18545933 -0.261977    0.30894761 -2.23354318  0.17612456]\n",
      "sample4: [-0.17500741 -1.30835159  0.5036692   0.14991385  0.99957472]\n",
      "sample5: [ 1.18545933 -0.261977    0.30894761 -2.23354318  0.17612456]\n",
      "sample6: [-0.17500741 -1.30835159  0.5036692   0.14991385  0.99957472]\n"
     ]
    }
   ],
   "source": [
    "# 1.35 Quantile - example\n",
    "import scipy; import stats; import numpy; import numpy as np; \n",
    "import scipy.stats as stats\n",
    "\n",
    "q_975 = stats.norm.ppf(0.975)\n",
    "print(f'q_975: {q_975}')\n",
    "\n",
    "# 1.36. Smpl-Bernoulli\n",
    "import scipy.stats as stats\n",
    "sample= stats.bernoulli.rvs(0.5, size=10)\n",
    "print(f'sample: {sample}')\n",
    "\n",
    "# 1.37. Smpl-Norm\n",
    "import scipy.stats as stats\n",
    "sample= stats.norm.rvs(size=10)\n",
    "print(f'sample: {sample}')\n",
    "\n",
    "# 1.38. Random-Numbers\n",
    "import numpy as np\n",
    "import scipy.stats as stats\n",
    "\n",
    "# sample from a standard normal RV with sample size n=S:\n",
    "sample1 = stats.norm.rvs(size=5)\n",
    "print(f'sample1: {sample1}')\n",
    "# a different sample from the same distribution:\n",
    "sample2 = stats.norm.rvs(size=5)\n",
    "print(f'sample2: {sample2}')\n",
    "# set the seed of the random number generator and take two samples:\n",
    "np.random.seed(6254137)\n",
    "sample3 = stats.norm.rvs(size=5)\n",
    "print(f'sample3: {sample3}')\n",
    "sample4 = stats.norm.rvs(size=5)\n",
    "print(f'sample4: {sample4}')\n",
    "# reset the seed to the same value to get the same samples again:\n",
    "np.random.seed(6254137)\n",
    "sample5= stats.norm.rvs(size=5)\n",
    "print(f'sample5: {sample5}')\n",
    "sample6 = stats.norm.rvs(size=5)\n",
    "print(f'sample6: {sample6}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "79a3e33d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "avgCh: -1.1544999999999999\n",
      "lowerCI95: -0.1936300609350276\n",
      "upperCI95: -0.07193010504007613\n",
      "lowerCI99: -0.21275050976771254\n",
      "upperCI99: -0.052809656207391184\n"
     ]
    }
   ],
   "source": [
    "# 1.39. Example-C.2\n",
    "import numpy; import scipy; import wooldridge; import stats\n",
    "import numpy as np\n",
    "import scipy.stats as stats\n",
    "\n",
    "# manually enter raw data from Wooldridge, Table C.3:\n",
    "SR87 = np.array([10, 1, 6, .45, 1.25, 1.3, 1.06, 3, 8.18, 1.67, .98, 1, .45, 5.03, 8, 9, 18, .28, 7, 3.97])\n",
    "SR88 = np.array([3,1,5,.5,1.54,1.5,.8,2,.67,1.17,.51,.5,.61,6.7,4,7,19,.2,5,3.83])\n",
    "# calculate change:\n",
    "Change = SR88 - SR87\n",
    "\n",
    "# ingredients to CI formula:\n",
    "avgCh = np.mean(Change)\n",
    "print(f'avgCh: {avgCh}')\n",
    "\n",
    "#1.40. Example C.3\n",
    "\n",
    "import wooldridge as woo\n",
    "import numpy as np\n",
    "import scipy.stats as stats\n",
    "audit= woo.dataWoo('audit')\n",
    "\n",
    "y = audit['y']\n",
    "# ingredients to CI formula:\n",
    "avgy = np.mean(y)\n",
    "n = len(y)\n",
    "sdy = np.std(y, ddof=1)\n",
    "se = sdy / np.sqrt(n)\n",
    "c95 = stats.norm.ppf(0.975)\n",
    "c99 = stats.norm.ppf(0.995)\n",
    "# 95% confidence interval:\n",
    "lowerCI95 = avgy - c95 * se\n",
    "print(f'lowerCI95: {lowerCI95}')\n",
    "upperCI95 = avgy + c95 * se\n",
    "print(f'upperCI95: {upperCI95}')\n",
    "# 99% confidence interval:\n",
    "lowerCI99 = avgy - c99 * se\n",
    "print(f'lowerCI99: {lowerCI99}')\n",
    "upperCI99 = avgy + c99 * se\n",
    "print(f'upperCI99: {upperCI99}')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "130c2763-6c09-40ba-aef6-7ce4f5c09ae4",
   "metadata": {},
   "source": [
    "# t value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "097c1d0f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "table: \n",
      "    alpha_one_tailed  alpha_two_tailed        CV\n",
      "0             0.100             0.200  1.327728\n",
      "1             0.050             0.100  1.729133\n",
      "2             0.025             0.050  2.093024\n",
      "3             0.010             0.020  2.539483\n",
      "4             0.005             0.010  2.860935\n",
      "5             0.001             0.002  3.579400 \n",
      "\n"
     ]
    }
   ],
   "source": [
    "# 1.41. Critical Value of t\n",
    "import numpy; import pandas; import scipy; import stats\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import scipy.stats as stats\n",
    "\n",
    "# degrees of freedom= n-1:\n",
    "df = 19\n",
    "# significance levels:\n",
    "alpha_one_tailed = np.array([0.1, 0.05, 0.025, 0.01, 0.005, .001])\n",
    "alpha_two_tailed = alpha_one_tailed * 2\n",
    "# critical values & table:\n",
    "CV= stats.t.ppf(1 - alpha_one_tailed, df)\n",
    "table= pd.DataFrame({'alpha_one_tailed': alpha_one_tailed,'alpha_two_tailed': alpha_two_tailed, 'CV':CV})\n",
    "print(f'table: \\n {table} \\n')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "28ebfcf2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "t_auto: -4.276816348963646\n",
      "p_auto/2: 1.369270781112999e-05\n",
      "t_manual: -4.27681634896364\n",
      "\n",
      "table:\n",
      "    alpha_one_tailed        CV\n",
      "0             0.100  1.285089\n",
      "1             0.050  1.651227\n",
      "2             0.025  1.969898\n",
      "3             0.010  2.341985\n",
      "4             0.005  2.596469\n",
      "5             0.001  3.124536 \n",
      "\n"
     ]
    }
   ],
   "source": [
    "# 1.42. Example C.5\n",
    "import wooldridge; import numpy; import pandas; import scipy; import stats\n",
    "import wooldridge as woo\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import scipy.stats as stats\n",
    "\n",
    "audit= woo.dataWoo('audit')\n",
    "y = audit['y']\n",
    "# automated calculation oft statistic for HO (mu=O):\n",
    "test_auto = stats.ttest_1samp(y, popmean=0)\n",
    "t_auto = test_auto.statistic # access test statistic\n",
    "p_auto = test_auto.pvalue # access two-sided p value\n",
    "print(f't_auto: {t_auto}')\n",
    "print(f'p_auto/2: {p_auto/2}')\n",
    "# manual calculation oft statistic for HO (mu=O):\n",
    "avgy = np.mean(y)\n",
    "n = len(y)\n",
    "sdy = np.std(y, ddof=1)\n",
    "se = sdy / np.sqrt(n)\n",
    "t_manual = avgy / se\n",
    "print(f't_manual: {t_manual}\\n')\n",
    "# critical values fort distribution with n-1=240 d.f.:\n",
    "alpha_one_tailed = np.array([0.1, 0.05, 0.025, 0.01, 0.005, .001])\n",
    "CV= stats.t.ppf(1 - alpha_one_tailed, 240)\n",
    "table= pd.DataFrame({'alpha_one_tailed': alpha_one_tailed, 'CV': CV})\n",
    "print(f'table:\\n {table} \\n')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "13d6ae60",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "t_auto: -2.150711003973493\n",
      "p_auto/2: 0.02229062646839212\n",
      "t_manual: -2.150711003973493\n",
      "p_manual: 0.02229062646839212\n"
     ]
    }
   ],
   "source": [
    "# 1.43. Example C.6\n",
    "import numpy; import scipy; import stats; import pandas; \n",
    "import numpy as np\n",
    "import scipy.stats as stats\n",
    "\n",
    "# manually enter raw data from Wooldridge, Table C.3:\n",
    "SR87 = np.array([10, 1, 6, .45, 1.25, 1.3, 1.06, 3, 8.18, 1.67, .98, 1, .45, 5.03, 8, 9, 18, .28, 7, 3.97])\n",
    "SR88 = np.array([3, 1, 5, .5, 1.54, 1.5, .8, 2, .67, 1.17, .51, .5, .61, 6.7, 4, 7, 19, .2, 5, 3.83])\n",
    "Change= SR88 - SR87\n",
    "# automated calculation oft statistic for HO (mu=O):\n",
    "test_auto = stats.ttest_1samp(Change, popmean=0)\n",
    "t_auto = test_auto.statistic\n",
    "p_auto = test_auto.pvalue\n",
    "print(f't_auto: {t_auto}')\n",
    "print(f'p_auto/2: {p_auto/2}')\n",
    "# manual calculation oft statistic for HO (mu=O):\n",
    "avg_Change = np.mean(Change)\n",
    "n = len(Change)\n",
    "sd_Change = np.std(Change, ddof=1)\n",
    "se = sd_Change/np.sqrt(n)\n",
    "t_manual = avg_Change/se\n",
    "print(f't_manual: {t_manual}')\n",
    "# manual calculation of p value for HO (mu=O):\n",
    "p_manual = stats.t.cdf(t_manual, n - 1)\n",
    "print(f'p_manual: {p_manual}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "6bf2a9c9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "t_auto: -4.276816348963646\n",
      "p_auto/2: 1.369270781112999e-05\n",
      "t_manual: -4.27681634896364\n",
      "p_manual: 1.3692707811130349e-05 \n",
      " \n"
     ]
    }
   ],
   "source": [
    "# 1.44. Example C.7\n",
    "import wooldridge; import numpy; import pandas; import scipy; import stats\n",
    "import wooldridge as woo\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import scipy.stats as stats\n",
    "\n",
    "audit= woo.dataWoo('audit')\n",
    "y = audit['y']\n",
    "# automated calculation oft statistic for HO (mu=O):\n",
    "test_auto = stats.ttest_1samp(y, popmean=0)\n",
    "t_auto = test_auto.statistic\n",
    "p_auto = test_auto.pvalue\n",
    "print(f't_auto: {t_auto}')\n",
    "print(f'p_auto/2: {p_auto/2}')\n",
    "# manual calculation oft statistic for HO (mu=O):\n",
    "avg_y = np.mean(y)\n",
    "n = len(y)\n",
    "sd_y = np.std(y, ddof=1)\n",
    "se = sd_y/np.sqrt(n)\n",
    "t_manual = avg_y/se\n",
    "print(f't_manual: {t_manual}')\n",
    "# manual calculation of p value for HO (mu=O):\n",
    "p_manual = stats.t.cdf(t_manual, n - 1)\n",
    "print(f'p_manual: {p_manual} \\n ')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c33d71bd-8f53-44e7-ac9c-403f2c75dd7a",
   "metadata": {},
   "source": [
    "# Python functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "f72ae7fe",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1\n",
      "8\n",
      "27\n",
      "16\n",
      "25\n",
      "36\n",
      "1\n",
      "8\n",
      "27\n",
      "16\n",
      "25\n",
      "36\n",
      "result1: 2.0\n",
      "result2: You fool!\n",
      "check: <class 'list'>\n",
      "count_six: 2\n",
      "a: [2, 3, 6, 6]\n"
     ]
    }
   ],
   "source": [
    "# 1.45. Adv-Loops\n",
    "seq = [1, 2, 3, 4, 5, 6]\n",
    "for i in seq:\n",
    "    if i < 4:\n",
    "        print(i ** 3)\n",
    "    else:\n",
    "        print(i ** 2)\n",
    "\n",
    "# 1.46. Adv-Loops2\n",
    "seq = [1, 2, 3, 4, 5, 6]\n",
    "for i in range(len(seq)):\n",
    "    if seq[i] < 4:\n",
    "        print(seq[i]**3)\n",
    "    else:\n",
    "        print(seq[i]**2)\n",
    "\n",
    "# 1.47. Adv-Functions\n",
    "# define function:\n",
    "def mysqrt (x) :\n",
    "    if x >= 0:\n",
    "        result = x ** 0.5\n",
    "    else:\n",
    "        result = 'You fool!'\n",
    "    return result\n",
    "# call function and save result:\n",
    "result1 = mysqrt(4)\n",
    "print(f'result1: {result1}')\n",
    "result2 = mysqrt(-1.5)\n",
    "print(f'result2: {result2}')\n",
    "\n",
    "# 1.48. Adv-ObjOr\n",
    "# use the predefined class 'list' to create an object:\n",
    "a = [2, 6, 3, 6]\n",
    "# access a local variable (to find out what kind of object we are dealing with):\n",
    "check= type(a)\n",
    "print(f'check: {check}')\n",
    "# make use of a method (how many 6 are in a?):\n",
    "count_six = a.count(6)\n",
    "print(f'count_six: {count_six}')\n",
    "# use another method (sort data in a):\n",
    "a.sort()\n",
    "print(f'a: {a}')\n"
   ]
  },
  {
   "cell_type": "raw",
   "id": "9d690019-f73e-431d-87fc-ccadb7e1ef40",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "8ab26e02-74e1-42da-8331-181c0d26f28c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "result_np: \n",
      "[[22 55 68]\n",
      " [27 55 76]]\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# 1.49 Adv-Obj0r2. \n",
    "\n",
    "import numpy as np \n",
    "# multiply these two matrices:\n",
    "a = np.array([[3, 6, 1], [2, 7, 4]])\n",
    "b = np.array([[1, 8, 6], [3, 5, 8], [1, 1, 2]]) \n",
    "# the numpy way:\n",
    "result_np = a.dot(b) \n",
    "print(f'result_np: \\n{result_np}\\n') "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "30ed5b75-2f9b-47fc-a6a1-3d949188218f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# next ... \n",
    "# or, do it again by yourself to define a class:\n",
    "\n",
    "class myMatrices:\n",
    "    def _init_ (self,A,B):\n",
    "        self()\n",
    "        self.A = A\n",
    "        self.B = B\n",
    "        \n",
    "    def _mult_(self):\n",
    "        N = self.A.shape[0] # number of rows in A\n",
    "        K = self.B.shape[1] # number of rows in B\n",
    "        out = np.empty(N,K) # initiale output\n",
    "        for i in range(N):\n",
    "            for j in range(K):\n",
    "                out[i,j] = sum(self.A[i,:]*self.B[:,j])\n",
    "        return out\n",
    "    \n",
    "# create an object:\n",
    "test = myMatrices(a, b)\n",
    "\n",
    "# access local variables: \n",
    "print(f'test.A: \\n{test.A}\\n')\n",
    "print(f'test.B: \\n{test.B}\\n') \n",
    "# use object method:\n",
    "result_own = test.mult() \n",
    "print(f'result_own: \\n{result_own}\\n') "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7c8108f7-04d5-420a-8787-3e03a43eb7b4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1.50: Adv-Obj0r3. \n",
    "\n",
    "import numpy as np \n",
    "# multiply these two matrices:\n",
    "a - np.array([[3, 6, 1], [2, 7, 4]])\n",
    "b = np.array([[1, 8, 6], [3, 5, 8], [1, 1, 2]]) \n",
    "# define your own class:\n",
    "class myMatrices: \n",
    "    def init (self, A, B):\n",
    "        self.A = A\n",
    "        self.B = B \n",
    "\n",
    "    def mult (self) :\n",
    "        N = self.A.shape[O] # number of rows in A\n",
    "        K = self.B.shape[l] # number of cols in B \n",
    "        out= np.empty((N, K)) # initialize output\n",
    "        for i in range(N): \n",
    "            for j in range(K):\n",
    "                out[i, j] = sum(self.A[i, :] * self.B[:, j]) \n",
    "        return out \n",
    "\n",
    "# define a subclass:\n",
    "class my_MatNew(myMatrices): \n",
    "    def getTotalElem(self):\n",
    "        N = self.A.shape[0] \n",
    "        K = self.B.shape[1]\n",
    "        return N * K \n",
    "\n",
    "# create an object of the subclass:\n",
    "test= myMatNew(a, b) \n",
    "\n",
    "# use a method of myMatrices:\n",
    "result_own = test.mult() \n",
    "print(f'result_own: \\n{result_own}\\n') \n",
    "\n",
    "# use a method of myMatNew:\n",
    "totalElem = test.getTotalElem() \n",
    "print(f'totalElem: {totalElem}\\n') "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e0cd1a06-7276-4c87-8fe2-930c654a069b",
   "metadata": {},
   "source": [
    "# Simulation Object"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "da1cdfc7-c0f6-4cbd-9dc1-0b9c234b8fea",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "estimate1: 9.573602656614304\n",
      "\n",
      "estimate2: 10.24798129790092\n",
      "\n",
      "estimate3: 9.96021755398913\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# 1.51 Simulate Estimate \n",
    "\n",
    "import numpy as np\n",
    "import scipy.stats as stats \n",
    "\n",
    "# set the random seed:\n",
    "np.random.seed(123456)\n",
    "# set sample size: \n",
    "n=100\n",
    "\n",
    "# draw a sample given the population parameters:\n",
    "sample1 = stats.norm.rvs(10, 2, size=n)\n",
    "\n",
    "# estimate the population mean with the sample average:\n",
    "estimate1 = np.mean(sample1) \n",
    "print(f'estimate1: {estimate1}\\n') \n",
    "\n",
    "# draw a different sample and estimate again:\n",
    "sample2 = stats.norm.rvs(10, 2, size=n)\n",
    "estimate2 = np.mean(sample2) \n",
    "print(f'estimate2: {estimate2}\\n') \n",
    "\n",
    "# draw a third sample and estimate again:\n",
    "sample3 = stats.norm.rvs(10, 2, size=n)\n",
    "estimate3 = np.mean(sample3)\n",
    "print(f'estimate3: {estimate3}\\n') \n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "0c79cd37-69f7-4de5-948d-37c2f4d5b8f8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1.52: Simulation-Repeated. \n",
    "\n",
    "import numpy as np\n",
    "import scipy.stats as stats \n",
    "\n",
    "# set the random seed:\n",
    "np.random.seed(123456)\n",
    "\n",
    "# set sample size: \n",
    "n = 100 \n",
    "# initialize ybar to an array of length r=lOOOO to later store results: \n",
    "r = 10000\n",
    "ybar = np.empty(r) \n",
    "# repeat r times:\n",
    "for j in range(r): \n",
    "    # draw a sample and store the sample mean in pos. j=O,l, ... of ybar:\n",
    "    sample= stats.norm.rvs(10, 2, size=n)\n",
    "    ybar[j] = np.mean(sample) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "e0dffed9-580a-4845-a2f9-8aefda0ce7fb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ybar[0:19]: \n",
      "[ 9.57360266 10.2479813   9.96021755  9.67635967  9.82261605  9.6270579\n",
      " 10.02979223 10.15400282 10.28812728  9.69935763 10.41950951 10.07993562\n",
      "  9.75764232 10.10504699  9.99813607  9.92113688  9.55713599 10.01404669\n",
      " 10.25550724]\n",
      "\n",
      "np.mean(ybar): 10.00082418067469\n",
      "\n",
      "np.var(ybar, ddof=1): 0.03989666893894718\n",
      "\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x258f024efd0>"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import numpy as np\n",
    "import statsmodels.api as sm \n",
    "import scipy.stats as stats\n",
    "import matplotlib.pyplot as plt \n",
    "# set the random seed:\n",
    "np.random.seed(123456) \n",
    "# set sample size: \n",
    "n = 100 \n",
    "\n",
    "# initialize ybar to an array of length r=lOOOO to later store results: \n",
    "r = 10000\n",
    "ybar = np.empty(r) \n",
    "# repeat r times:\n",
    "for j in range(r): \n",
    "    # draw a sample and store the sample mean in pos. j=0,1, ... of ybar: \n",
    "    sample= stats.norm.rvs(10, 2, size=n)\n",
    "    ybar[j] = np.mean(sample) \n",
    "    \n",
    "# the first 20 of 10000 estimates:\n",
    "print(f'ybar[0:19]: \\n{ybar[0:19]}\\n') \n",
    "# simulated mean:\n",
    "print(f'np.mean(ybar): {np.mean(ybar)}\\n') \n",
    "# simulated variance:\n",
    "print(f'np.var(ybar, ddof=1): {np.var(ybar, ddof=1)}\\n') \n",
    "# simulated density:\n",
    "kde = sm.nonparametric.KDEUnivariate(ybar) \n",
    "kde.fit() \n",
    "    \n",
    "# normal density:\n",
    "x_range = np.linspace(9, 11) \n",
    "y = stats.norm.pdf(x_range, 10, np.sqrt(0.04)) \n",
    "# create graph:\n",
    "plt.plot(kde.support, kde.density, color='black', label='ybar')\n",
    "plt.plot(x_range, y, linestyle='--', color='black', label='normal distribution') \n",
    "plt.ylabel('density')\n",
    "plt.xlabel('ybar') \n",
    "plt.legend ()\n",
    "# plt.savefig('PyGraphs/Simulation-Repeated-Results.pdf') "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c36f9dac-f78c-47e8-967a-4637c070c9fb",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1.54: Simulation-Inference-Figure.\n",
    "\n",
    "import numpy as np \n",
    "import scipy.stats as stats\n",
    "import matplotlib.pyplot as plt \n",
    "# set the random seed:\n",
    "np.random.seed(l23456) \n",
    "# set sample size and MC simulations:\n",
    "r = 10000 \n",
    "n = 100 \n",
    "# initialize arrays to later store results:\n",
    "Cilower - np.empty(r) \n",
    "Ciupper - np.empty(r)\n",
    "pvaluel - np.empty(r)\n",
    "pvalue2 - np.empty(r) \n",
    "\n",
    "# repeat r times:\n",
    "for j in range(r): \n",
    "    # draw a sample:\n",
    "    sample= stats.norm.rvs(lO, 2, size=n) \n",
    "    sample_mean = np.mean(sample)\n",
    "    sample_sd = np.std(sample, ddof=l) \n",
    "    # test the (correct) null hypothesis mu=lO:\n",
    "    testresl = stats.ttest_lsamp(sample, popmean=lO) \n",
    "    pvaluel[j] = testresl.pvalue\n",
    "    cv = stats.t.ppf(0.975, df=n - 1) \n",
    "    Cilower[j] = sample_mean - cv * sample_sd I np.sqrt(n)\n",
    "    Ciupper[j] = sample_mean + cv * sample_sd I np.sqrt(n) \n",
    "    # test the (incorrect) null hypothesis mu=9.5 & store the p value: \n",
    "    testres2 = stats.ttest_lsarnp(sample, popmean=9.5)\n",
    "    pvalue2[j] = testres2.pvalue \n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  },
  "toc-autonumbering": true,
  "toc-showcode": false,
  "toc-showmarkdowntxt": false
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
