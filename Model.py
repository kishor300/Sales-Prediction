import pickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from dateutil import relativedelta
from datetime import datetime, timedelta
import calendar

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

##############################################################################

with open('Model_Monthly_Sales.pkl', 'rb') as f:
    Model_Monthly_Sales = pickle.load(f)
with open('Model_ShipMode_Preference.pkl', 'rb') as f:
    Model_ShipMode_Preference = pickle.load(f)
with open('Model_Market_Sales.pkl', 'rb') as f:
    Model_Market_Sales = pickle.load(f)
with open('Model_Market_Profit.pkl', 'rb') as f:
    Model_Market_Profit = pickle.load(f)
with open('Model_Category_Sales.pkl', 'rb') as f:
    Model_Category_Sales = pickle.load(f)
with open('Model_Category_Profit.pkl', 'rb') as f:
    Model_Category_Profit = pickle.load(f)
with open('Model_Category_Quantity.pkl', 'rb') as f:
    Model_Category_Quantity = pickle.load(f)


def Historical_Data_Analysis():
    # loading dataset
    df = pd.read_csv('superstore_df.csv')
    
    # [1] Plotting Historical Sales 
    df['order_date'] = pd.to_datetime(df['order_date'],format='%d-%m-%Y')

    df['order_date'] = df['order_date'].dt.strftime('%Y-%m')  #comment 1

    df_trend = df.groupby('order_date').sum()['sales'].reset_index() #comment 2

    plt.figure(figsize=(12, 5))
    plt.plot(df_trend['order_date'], df_trend['sales'], marker='o')   #comment 2
    
    plt.xticks(rotation='vertical') # horizontal alignment of the tick labels. Align tick labels to right side of tick marks
    plt.ticklabel_format(style='plain', axis='y')   # to show entire number of amount on y-axis
    plt.title('Historical_Month_wise_Sales')
    plt.xlabel('Monthly Dates')
    plt.ylabel('Sales in $')
    plt.tight_layout()
    plt.savefig('static\Historical_Sales.png', bbox_inches='tight', dpi=400)

    # [2] Plotting  Top 10 products by sales
    # Grouping products by sales from original dataset
    prod_by_sales = pd.DataFrame(df.groupby('product_name').sum()['sales'])
    # Sorting the dataframe in descending order
    prod_by_sales = prod_by_sales.sort_values('sales',ascending=False)
    # Top 10 products by sales
    prod_by_sales =prod_by_sales[:10]
    prod_by_sales =prod_by_sales.reset_index()
    plt.figure(figsize=(8, 4))
    sns.barplot(x='sales', y='product_name', data=prod_by_sales)

    plt.xlabel('Sales')
    plt.ylabel('Product Name')
    plt.title('Top Products by Sales')
    plt.tight_layout()
    plt.savefig('static\Products_wise_Sales.png', bbox_inches='tight', dpi=400)

    # [3] Plotting Most Selling Products by Quantity
    # Grouping products by Quantity
    prod_by_quant = pd.DataFrame(df.groupby('product_name').sum()['quantity'])
    # Sorting the dataframe in descending order of quantity
    prod_by_quant = prod_by_quant.sort_values('quantity',ascending=False)
    # Top 10 products by quantity
    prod_by_quant = prod_by_quant[:10]
    prod_by_quant =prod_by_quant.reset_index()

    plt.figure( figsize=(12, 5) )
    sns.barplot(x='quantity', y='product_name', data=prod_by_quant)
    plt.xlabel('Quantity')
    plt.ylabel('Product Name')
    plt.title('Top Products by Quantity')
    plt.tight_layout()
    plt.savefig('static\Products_wise_Quantity.png', bbox_inches='tight', dpi=400)

    # [4] Plotting Most preferable ship mode
    count = df['ship_mode'].value_counts()

    # pie chart Most preferable ship mode
    plt.figure(figsize=(6, 4))
    plt.pie(count, labels=count.index, autopct='%1.1f%%')
    plt.title(' Most preferable ship mode')
    plt.tight_layout()
    plt.savefig('static\Preferable_Ship_mode.png', bbox_inches='tight', dpi=400)

    # [5] Plotting Most Profitable Category and Sub-Category
    # Grouping products by Category and Sub-Category with profit only
    cat = pd.DataFrame(df.groupby(['category']).sum()['profit'])
    subcat = pd.DataFrame(df.groupby(['sub_category']).sum()['profit'])
    # Sort in descending order of profit
    cat = cat.sort_values('profit', ascending=False)
    subcat = subcat.sort_values('profit', ascending=False)
    # Most Profitable Category and Sub-Category
    cat = cat.reset_index()
    subcat = subcat[:10]
    subcat = subcat.reset_index()

    # pie chart for category
    plt.figure(figsize=(6, 4))
    plt.pie(cat['profit'], labels=cat['category'], autopct='%1.1f%%')
    plt.title('Most Profitable Category')
    plt.tight_layout()
    plt.savefig('static\Category_wise_Profit.png', bbox_inches='tight', dpi=400)

    # barplot for sub_category
    plt.figure( figsize=(8, 5) )
    sns.barplot(x='sub_category', y='profit', data=subcat)
    plt.xlabel('sub_category')
    plt.ylabel('Profit')
    plt.title('Profitable Sub_Category')
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.savefig('static\Sub_Category_wise_Profit.png', bbox_inches='tight', dpi=400)

    # [6] Plotting Category and Sub-Category wise sales
    # Grouping products by Category and Sub-Category with sales only
    cat = pd.DataFrame(df.groupby(['category']).sum()['sales'])
    subcat = pd.DataFrame(df.groupby(['sub_category']).sum()['sales'])
    # Sort in descending order of sales
    cat = cat.sort_values('sales', ascending=False)
    subcat = subcat.sort_values('sales', ascending=False)
    # Category and Sub-Category wise sales
    cat = cat.reset_index()
    subcat = subcat[:10]
    subcat = subcat.reset_index()

    # pie chart for category
    plt.figure(figsize=(6, 4))
    plt.pie(cat['sales'], labels=cat['category'], autopct='%1.1f%%')
    plt.title('Category wise Sales')
    plt.tight_layout()
    plt.savefig('static\Category_wise_Sales.png', bbox_inches='tight', dpi=400)

    # barplot for sub_category
    plt.figure( figsize=(8, 5) )
    sns.barplot(x='sub_category', y='sales', data=subcat)
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel('sub_category')
    plt.ylabel('sales')
    plt.title('Sub-Category wise Sales')
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.savefig('static\Sub_Category_wise_Sales.png', bbox_inches='tight', dpi=400)

    # [7] Market wise Sales
    df['sales']=round(df['sales'],2)
    # Grouping market with sales
    market_sale = pd.DataFrame( df.groupby(['market']).sum()['sales']  )
    market_sale = market_sale.sort_values('sales', ascending=False)
    market_sale = market_sale.reset_index()

    plt.figure( figsize=(8, 6) )
    sns.barplot(x='market', y='sales', data=market_sale)
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel('Market')
    plt.ylabel('Sales')
    plt.title('Market wise Sales')
    plt.tight_layout()
    plt.savefig('static\Market_wise_Sales.png', bbox_inches='tight', dpi=400)    

Historical_Data_Analysis()


def Plot_Line(data):
    plt.figure(figsize=(12, 5))
    plt.plot(data.keys(), data.values(), marker='o', color='red')

    # gca() function is "get current axes. It returns the current Axes instance, which represents the plot or chart being displayed.
    # plt.gca().xaxis.set_major_formatter(plt.FixedFormatter(df_test.index.strftime('%Y-%m-%d')))
    
    plt.xticks(rotation=45, ha='right') # horizontal alignment of the tick labels. Align tick labels to right side of tick marks
    plt.ticklabel_format(style='plain', axis='y')   # to show entire number of amount on y-axis
    plt.title('Predicted_Sales')
    plt.xlabel('Monthly Dates')
    plt.ylabel('Sales in $')
    plt.tight_layout()
    plt.savefig('static/Predicted_Sales.png', bbox_inches='tight', dpi=400)

    # # Create a figure and axis object
    # x = data.keys()
    # y = data.values()
    # print("***********", x,'***********', y )

    # fig, ax = plt.subplots()

    # # Plot the data as a line plot
    # ax.plot(x, y)

    # # Set the title and axis labels
    # ax.set(title='Line Plot Example', xlabel='X', ylabel='Y')

    # # Save the figure as a PNG image in the root folder
    # fig.savefig('line_plot.png')

def Plot_Bar(col_exog,col_year,col_pred,X_Label,Y_Label,Title):
    print('\n\n Plot_Bar()\n\n')
    # Data
    col_exog = col_exog
    col_year = col_year
    col_pred = col_pred

    # Get unique ship_modes and years
    unique_exog = list(set(col_exog))
    unique_years = list(set(col_year))

    unique_years.sort() # Sort unique_years in ascending order

    # Set the positions of the bars on the x-axis
    x = np.arange(len(unique_years))

    # Set the total width of the bars
    total_width = 0.8

    # Calculate the width of each individual bar
    bar_width = total_width / len(unique_exog)

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate over each ship_mode and plot the corresponding predicted count
    for i, exog in enumerate(unique_exog):
        # Get the predicted count for the current ship_mode
        exog_metric = [col_pred[j] for j in range(len(col_exog)) if col_exog[j] == exog]

        # Calculate the x positions of bars for each ship_mode
        x_shifted = x + (i - len(unique_exog) / 2) * bar_width
        ax.bar(x_shifted, exog_metric, bar_width, label=exog)

    # Set the x-axis labels and tick positions
    ax.set_xticks(x)
    ax.set_xticklabels(unique_years)

    plt.ticklabel_format(style='plain', axis='y')   # to show entire number of amount on y-axis

    # Set the y-axis label
    plt.ylabel(f'{Y_Label}')
    plt.xlabel(f'{X_Label}')

    # Set the plot title
    ax.set_title(f'{Title}')

    # Add a legend
    ax.legend(title=f'{Title}',loc='upper right')

    # Adjust the layout
    plt.tight_layout()

    plt.savefig(f'static\{Title}_Prediction.png',bbox_inches='tight',dpi=400) # f' embed python expressions

def Plot_Pie_Chart(col_exog,col_year,col_pred,Title):
    print('\n\n Plot_Pie_Chart()\n\n')
    df_category = col_exog
    df_year = col_year
    df_sales = col_pred

    # Create a dictionary to store sales data for each year
    sales_data = {}
    for year in set(df_year):
        sales_data[year] = []   # set individual year as key in dictionary sales_data

    # Populate the sales data dictionary
    for category, year, sales in zip(df_category, df_year, df_sales):
        sales_data[year].append((category, sales))      # assigning values (cat,sales) to individual key (year) in dictionary

    # Plotting pie charts for each year
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']

    # Create subplots with a dynamic number of columns based on the number of years
    num_years = len(sales_data) # = 4
    num_cols = min(num_years, 3)  # = 3  #minimum 3 pie charts will shown in one row
    num_rows = (num_years + num_cols - 1) // num_cols   # (4+3-1)//3 = 2

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows)) # 8 axs will create for separate subplot of 4 years (if we have 2 years data)
    axs = np.array(axs).flatten()  # Flatten the 2D array to a 1D array for easier indexing ; # converts axs object (list of Axes) into a NumPy array.

    # getting index,year from sorted dictionary i,e sample_data={year='cat':sales}
    for index_of_year, year in enumerate( sorted(sales_data) ):
        ax = axs[index_of_year] # create separate axis for separate year
        
        categ_labels = []
        sales_values = []
        
        for category, sales in sales_data[year]:
            categ_labels.append(category)
            sales_values.append(abs(sales))
        ax.pie(sales_values, labels=categ_labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(f"Year {year}, {Title} Distribution ")
    
    # Hide the empty subplots num_years=4 , len(axs)=6  # Not displaying 5th & 6th subplot is empty bcause of less num_years
    for j in range(num_years, len(axs)):
        axs[j].axis('off')

    # Adjust the spacing between the subplots
    plt.subplots_adjust(wspace=1.5)

    plt.tight_layout()

    # Save the combined pie charts as a single image file
    plt.savefig(f'static\{Title}_Prediction.png',bbox_inches='tight',dpi=400)


def CATEGORY_QUANTITY(start_date, end_date):
    # print('\n start_year : ', start_date.year,'\n end_year : ', end_date.year )
    years = []
    category = []
    category_list = ['Furniture', 'Office Supplies', 'Technology']

    for year in range(start_date.year, end_date.year + 1):
        for c in category_list:
            years.append(year)
            category.append(c)

    year_category = {'year': years, 'category': category}
    df_pred = pd.DataFrame(year_category)
    # print('\n',df_pred)

    df_pred['category'] = LE.fit_transform(df_pred['category'])

    Predicted_Category_Quantity = Model_Category_Quantity.forecast(steps=len(df_pred['category']), exog= df_pred['category'] )

    df_pred['category'] = LE.inverse_transform(df_pred['category'])
    df_pred['Predicted_Category_Quantity'] = Predicted_Category_Quantity.values # storing total no. of predictions
    # print('\n\n',df_pred,'\n\n')

    col_exog = df_pred['category']
    col_year = df_pred['year']
    col_pred = df_pred['Predicted_Category_Quantity']
    X_Label = 'Year'
    Y_Label = 'Quantity'
    Title = 'Category_wise_Quantity'
    Plot_Pie_Chart(col_exog,col_year,col_pred,Title)
    # Plot_Bar(col_exog,col_year,col_pred,X_Label,Y_Label,Title)   # calling Plot_Bar function
 

def CATEGORY_PROFIT(start_date, end_date):
    # print('\n start_year : ', start_date.year,'\n end_year : ', end_date.year )
    years = []
    category = []
    category_list = ['Furniture', 'Office Supplies', 'Technology']

    for year in range(start_date.year, end_date.year + 1):
        for c in category_list:
            years.append(year)
            category.append(c)

    year_category = {'year': years, 'category': category}
    df_pred = pd.DataFrame(year_category)
    # print('\n',df_pred)

    df_pred['category'] = LE.fit_transform(df_pred['category'])

    Predicted_Category_Profit = Model_Category_Profit.forecast(steps=len(df_pred['category']), exog= df_pred['category'] )

    df_pred['category'] = LE.inverse_transform(df_pred['category'])
    df_pred['Predicted_Category_Profit'] = Predicted_Category_Profit.values # storing total no. of predictions
    # print('\n\n',df_pred,'\n\n')

    col_exog = df_pred['category']
    col_year = df_pred['year']
    col_pred = df_pred['Predicted_Category_Profit']
    X_Label = 'Year'
    Y_Label = 'Profit in $'
    Title = 'Category_wise_Profit'
    Plot_Pie_Chart(col_exog,col_year,col_pred,Title)
    # Plot_Bar(col_exog,col_year,col_pred,X_Label,Y_Label,Title)   # calling Plot_Bar function


def CATEGORY_SALES(start_date, end_date):
    # print('\n start_year : ', start_date.year,'\n end_year : ', end_date.year )
    years = []
    category = []
    category_list = ['Furniture', 'Office Supplies', 'Technology']

    for year in range(start_date.year, end_date.year + 1):
        for c in category_list:
            years.append(year)
            category.append(c)

    year_category = {'year': years, 'category': category}
    df_pred = pd.DataFrame(year_category)
    # print('\n',df_pred)

    df_pred['category'] = LE.fit_transform(df_pred['category'])

    Predicted_Category_Sales = Model_Category_Sales.forecast(steps=len(df_pred['category']), exog= df_pred['category'] )
    
    df_pred['category'] = LE.inverse_transform(df_pred['category'])
    df_pred['Predicted_Category_Sales'] = Predicted_Category_Sales.values # storing total no. of predictions
    # print('\n\n',df_pred,'\n\n')

    col_exog = df_pred['category']
    col_year = df_pred['year']
    col_pred = df_pred['Predicted_Category_Sales']
    X_Label = 'Year'
    Y_Label = 'Sales in $'
    Title = 'Category_wise_Sales'
    Plot_Pie_Chart(col_exog,col_year,col_pred,Title)
    # Plot_Bar(col_exog,col_year,col_pred,X_Label,Y_Label,Title)   # calling Plot_Bar function


def MARKET_PROFIT(start_date, end_date):
    # print('\n start_year : ', start_date.year,'\n end_year : ', end_date.year )
    years = []
    market = []
    market_list = ['Boston', 'Chicago', 'Denver', 'Los Angeles', 'New York', 'San Francisco', 'Washington D.C.']

    for year in range(start_date.year, end_date.year + 1):
        for m in market_list:
            years.append(year)
            market.append(m)

    year_market = {'year': years, 'market': market}
    df_pred = pd.DataFrame(year_market)
    # print('\n',df_pred)

    df_pred['market'] = LE.fit_transform(df_pred['market'])

    Predicted_Market_Profit = Model_Market_Profit.forecast(steps=len(df_pred['market']), exog=df_pred['market'] )
    
    df_pred['market'] = LE.inverse_transform(df_pred['market'])
    df_pred['Predicted_Market_Profit'] = Predicted_Market_Profit.values # storing total no. of predictions
    # print('\n\n',df_pred,'\n\n')

    col_exog = df_pred['market']
    col_year = df_pred['year']
    col_pred = df_pred['Predicted_Market_Profit']
    X_Label = 'Year'
    Y_Label = 'Profit in $'
    Title = 'Market_wise_Profit'
    Plot_Bar(col_exog,col_year,col_pred,X_Label,Y_Label,Title)   # calling Plot_Bar function


def MARKET_SALES(start_date, end_date):
    # print('\n start_year : ', start_date.year,'\n end_year : ', end_date.year )
    years = []
    market = []
    market_list = ['Boston', 'Chicago', 'Denver', 'Los Angeles', 'New York', 'San Francisco', 'Washington D.C.']

    for year in range(start_date.year, end_date.year + 1):
        for m in market_list:
            years.append(year)
            market.append(m)

    year_market = {'year': years, 'market': market}
    df_pred = pd.DataFrame(year_market)
    # print('\n',df_pred)

    df_pred['market'] = LE.fit_transform(df_pred['market'])

    Predicted_Market_Sales = Model_Market_Sales.forecast(steps= len(df_pred['market']), exog= df_pred['market'] )
    
    df_pred['market'] = LE.inverse_transform(df_pred['market'])
    df_pred['Predicted_Market_Sales'] = Predicted_Market_Sales.values # storing total no. of predictions
    # print('\n\n',df_pred,'\n\n')

    col_exog = df_pred['market']
    col_year = df_pred['year']
    col_pred = df_pred['Predicted_Market_Sales']
    X_Label = 'Year'
    Y_Label = 'Sales in $'
    Title = 'Market_wise_Sales'
    Plot_Bar(col_exog,col_year,col_pred,X_Label,Y_Label,Title)   # calling Plot_Bar function


def SHIP_MODE(start_date, end_date):
    # print('\n start_year : ', start_date.year,'\n end_year : ', end_date.year )
    years = []
    ship_mode = []
    modes = ['First Class', 'Same Day', 'Second Class', 'Standard Class']

    for year in range(start_date.year, end_date.year + 1):
        for m in modes:
            years.append(year)
            ship_mode.append(m)

    year_shipmode = {'year': years, 'ship_mode': ship_mode}
    df_pred = pd.DataFrame(year_shipmode)
    # print('\n',df_pred)

    df_pred['ship_mode'] = LE.fit_transform(df_pred['ship_mode'])

    Pref_Pred = Model_ShipMode_Preference.forecast(steps= len(df_pred['ship_mode']), exog= df_pred['ship_mode'] )
    
    df_pred['ship_mode'] = LE.inverse_transform(df_pred['ship_mode'])
    df_pred['Predicted_Count'] = Pref_Pred.values # storing total no. of predictions
    # print('\n\n',df_pred,'\n\n')

    col_exog = df_pred['ship_mode']
    col_year = df_pred['year']
    col_pred = df_pred['Predicted_Count']
    X_Label = 'Year'
    Y_Label = 'Preference Count'
    Title = 'Preferable_Ship_Mode'
    Plot_Bar(col_exog,col_year,col_pred,X_Label,Y_Label,Title)   # calling Plot_Bar function


def Prediction(start_date, end_date):

    print('\n***********************\n Start Date:', start_date,
            '\t End Date:', end_date,'\n***********************')

    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    first_day_of_month,total_days_of_month = calendar.monthrange(start_date.year,start_date.month) 
    last_day = total_days_of_month
    start_date = start_date.replace(day = last_day) # replace date day with last day (for showing sales at end of the month)

    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    first_day_of_month,total_days_of_month = calendar.monthrange(end_date.year,end_date.month)
    last_day = total_days_of_month
    end_date = end_date.replace(day = last_day) # replace date day with last day (for showing sales at end of the month)

    delta = relativedelta.relativedelta(end_date, start_date)   #+2
    months = delta.months + (delta.years * 12)

    ######################### ADDING FIRST DATE INDEX ####################################
    delta1 = relativedelta.relativedelta( datetime.strptime('2022-12-31', '%Y-%m-%d'),
                                          start_date)    #-1 # (last_date_of_dataset , user_entered_date)
    first_month_date = abs(delta1.months + (delta1.years * 12))     # calculate 'no._of_days' from last_date_of_dataset to user_entered_date
    # print('--------------------------------------')
    # print(abs(first_month_date))
    # print(months)

    ######################################################################################


    SHIP_MODE(start_date, end_date)

    MARKET_SALES(start_date, end_date)

    MARKET_PROFIT(start_date, end_date)

    CATEGORY_SALES(start_date, end_date)

    CATEGORY_PROFIT(start_date, end_date)

    CATEGORY_QUANTITY(start_date, end_date)

    Sales_Pred = Model_Monthly_Sales.predict(start=48+first_month_date-1,
                                    end=48+first_month_date + months-1, dynamic=True)    # add above 'no._of_days' to get prediction for exactly user entered dates

    # disct = {}
    dic = { str(i)[:10]: j for i, j in zip(Sales_Pred.index, np.round(Sales_Pred.values,2)) } # Rounding the value upto 2 decimal places
    # print(dic)
    Plot_Line(dic)

    data = { "dates": [i for i in dic.keys()],
            "values": [j for j in dic.values()] }
    
    return dic

# #Trial start_date & end_date for executing Model.py
# start_date, end_date = ('2023-01-01', '2024-12-01')     
# Prediction(start_date, end_date)
