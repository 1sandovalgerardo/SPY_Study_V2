import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader.data as web
from os import path
from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt
import logging
import argparse
from argparse import RawTextHelpFormatter as raw
import sys
import csv
import IPython


def create_log(debug):
    '''Will establish logging. debug(bool)'''
    if debug:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

        fileHandler = logging.FileHandler('logStockAnalysis.log', 'w')
        fileHandler.setLevel(logging.INFO)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        shellHandler = logging.StreamHandler()
        shellHandler.setLevel(logging.WARNING)
        logger.addHandler(shellHandler)
        logging.debug('logging established')
    else:
        pass

def Parser():
    '''Create parser. returns parser.parsearguments() object'''
    studyHelp = '[ExecSummary, WalkThrough]'
    studyDetails='''
ExecSummary: Calculates all studies but only shows the final result plot.
    The final result plot shows the number of high correlation periods found.
    Writing to csv is optional.
    If plot is passed in, all charts will be shown.

WalkThrough: Will plot all studies. Writing to csv is optional.
 '''
    parser = argparse.ArgumentParser(description='Analyze a stock',
                                     epilog=studyDetails, formatter_class=raw)
    parser.add_argument('TICKER',
                        metavar='<TickerSymbol>',
                        type=str,
                        help='Enter the ticker symbol you wish to analyze')
    parser.add_argument('StudyType',
                        metavar='<Study Type>',
                        choices=['ExecSummary', 'WalkThrough', 'Build' ],
                        help=studyHelp,
                        )
    parser.add_argument('RollingWindow',
                        metavar='<RollingWindow>',
                        type=int,
                        help='The number of days to use for rolling time periods.')
    parser.add_argument('Correlation_N',
                        metavar='<Correlation Threshold>',
                        type=float,
                        help='Correlation threshold. -1 < value < 1.')
    parser.add_argument('-d', '--debug',
                        dest='DEBUG',
                        action='store_true',
                        help='Turn on debugging',
                        default=False
                        )
    parser.add_argument('-p', '--plot',
                        dest='plot',
                        action='store_true',
                        help='If passed, plots will be shown.',
                        default=False
                        )
    parser.add_argument('-s', '--save',
                        dest='save',
                        action='store_true',
                        help='Save all data produced to csv files',
                        default=False)
    args = parser.parse_args()

    return args

def checkParser(parser):
    '''Check correct values were input into the command line.'''
    logging.info('checkParser() called')
    ticker = parser.TICKER
    window = parser.RollingWindow
    corrN = parser.Correlation_N
    # Test for letters in ticker argument
    try:
        test = int(ticker)
        raise TypeError
    except TypeError:
        print('Integers are invalid tickers')
        sys.exit()
    except:
        pass
    # test for number larger than 1 for window
    try:
        test = int(window)
        if test < 1:
            raise TypeError
    except TypeError:
        print('Window integer must be greater than 1')
        sys.exit()
    except:
        print('Window argument must be an integer')
        sys.exit()
    # test for correct range of float number for correlation threshold
    try:
        test = float(corrN)
        if test <= -1 or test >= 1:
            raise Exception
    except ValueError:
        print('Correlation Number must be a float type')
        sys.exit()
    except Exception:
        print('Correlation Threshold value must be: -1 < x < 1')
        sys.exit()



class StockAnalysis(object):
    '''
   A class built to search for yearly patterns in a stocks price movement.
   The search is done by using correlation over a specific time period.
   Final chart demonstrates how many occurences of high correlation exists
   in the data set.
   Contains functions that allow for plotting and saving to csv.
    '''

    def __init__(self, tickerSymbol, fromDate=None, toDate=None):
        '''
        self.ticker(str): ticker symbol for stock that is to be analyzed.
        self.data(tuple(list, list)): 2 element tuple. each element is a list.
            index 0: dates for data
            index 1: prices for data
        self.dates(list): contains datetime.date objects
        self.prices(list): contains the closing prices of the stock being analyzed.
        self.years(list): list of what years are contained in the data
        '''
        logging.info('StockAnalysis object created')
        self.ticker = str(tickerSymbol.upper())
        self.data = self._get_data()
        self.dates = self.data[0]
        self.prices = self.data[1]
        self.years = self._extract_years()
        #self.by_years = self.group_by_year()

    def __repr__(self):
        return self.data

    def __str__(self):
        return (f'Stock: {self.ticker}')

    def download_data(self, tickerSymbol=None, fromDate=None, toDate=None):
        '''
        Uses pandas.DataReader.Data API to download stock data from yahoo.
        Args:
            tickerSymbol(str): required argument. ticker symbol of stock you with to analyze.
            fromDate(datetime.date): For use when accessing just the function outside
                of the command line interface.
            toDate(datetime.date): for use when accesssing the fucntion outside of the
                command line interface.
        Returns:
            None
            Saves data to csv file titled: TICKER.csv
        '''

        logging.info('download_data called')
        startDate = dt.date(2000, 1, 1)
        endDate = dt.date(2020, 1, 1)
        if tickerSymbol:
            logging.info('custom ticker input into download_data')
            ticker = tickerSymbol
            if fromDate and toDate:
                startDate = fromDate
                endDate = toDate
                try:
                    stockData = web.DataReader(ticker, 'yahoo', startDate, endDate)['Close']
                    self._save_data(stockData)
                except Exception as error:
                    print(error)
                    print('Check the ticker, fromDate, or toDate')
                    sys.exit()

            else:
                try:
                    stockData = web.DataReader(ticker, 'yahoo', startDate, endDate)['Close']
                    self._save_data(stockData)
                except Exception as error:
                    print(error)
                    print('Check the ticker symbol')
                    sys.exit()
        else:
            try:
                stockData = web.DataReader(self.ticker, 'yahoo',
                                           startDate, endDate)['Close']
                self._save_data(stockData)
            except Exception as error:
                print(error)
                print('An error occurred.')
                print('You may have entered an incorrect ticker symbol')
                sys.exit()

    def _save_data(self, dataToSave):
        '''
        Used inside download_data(). Built for saving stock data to csv file.
        Args:
            dataToSave(pandas.DataFrame()): output from pandas yahoo API.
        Returns:
            None
            Saves dataToSave to file.
        '''
        logging.info('save_data() called')
        if type(dataToSave)==type(pd.Series()):
            logging.info('type of dataToSave type pd.Series')
            dataToSave.to_csv(f'{self.ticker}.csv', header=False)
        else:
            logging.info('type of dataToSave not pd.Series')

    def _save_group_by_year(self, byYearData):
        '''To be used inside group_by_year(). Will save the output of that func.'''
        keys = sorted(byYearData.keys())
        with open(f'By_Year_Data_{self.ticker}.csv', 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter='\t')
            writer.writerow(keys)
            writer.writerows(zip(*[byYearData[key] for key in keys]))

    def _save_high_corr_filter(self, filterData):
        '''Save correlation data.
        Args:
            filterData: output from high_corr_filter()
        Returns:
            None
            Saves filterData to file as csv.
        '''
        keys = sorted(filterData.keys())
        dataToSave = []
        for k, v in filterData.items():
            row = []
            row.append(str(k))
            for i in v:
                row.append(str(i[0]))
            dataToSave.append(row)
        with open(f'Corr_Filter_Results_{self.ticker}.csv', 'w') as csvFile:
            reader = csv.writer(csvFile, delimiter=',', quotechar='"')
            for row in dataToSave:
                reader.writerow(row)



    def _read_data_csv(self):
        '''Read stock data into memory for analysis.'''
        logging.info('read_data_csv called')
        localDates = []
        localPrices = []
        filePath = f'{self.ticker}.csv'
        with open(filePath, 'r') as dataFile:
            reader = csv.reader(dataFile, delimiter=',')
            for row in reader:
                # item 1 of row, split and turn into ints that will be used
                # to create a datetime.date object
                date = row[0].split('-')
                year, month, day = int(date[0]), int(date[1]), int(date[2])
                date = dt.date(year, month, day)
                localDates.append(date)
                # item 2 of row will be turned into rounded floats
                price = round(float(row[1]), 3)
                localPrices.append(price)
        return (localDates, localPrices)

    def _get_data(self):
        '''used in self.data to bring data into memory for analysis.'''
        logging.info('_get_data() called')
        filePath = f'{self.ticker}.csv'
        if path.exists(filePath):
            logging.info(f'{filePath} exists')
            return self._read_data_csv()
        else:
            logging.info('stock data file did NOT exists')
            self.download_data()
            return self._get_data()

    def _extract_years(self, dates=None):
        '''Creates a list of the years that the stock data spans.
        Args:
            dates(list): optional argument should the function be accessed
            outside of command line.
        Returns:
              yearsList(list): List containing the years spanning the data set.
        '''
        logging.info('_extract_years() called')
        if dates:
            dates = dates
            yearsList = []
            for date in dates:
                yearsList.append(date.year)
            yearsList = list(set(yearsList))
            return yearsList
        logging.info('_extract_years() was not passed a dates argument')
        yearsList = []
        for date in self.dates:
            yearsList.append(date.year)
        yearsList = list(set(yearsList))
        # yearsList type list
        return yearsList

    def _clean_length(self, aDict):
        '''Used to create data sets of same length. Allows for corr analysis.'''
        logging.info('_clean_length() called')
        toClean = aDict
        lenOfData = []
        for v in toClean.values():
            lenOfData.append(len(v))
        minLength = min(lenOfData)
        cleanDict = defaultdict(list)
        for k in toClean:
            cleanDict[k] = toClean[k][:minLength]
        return cleanDict

    def group_by_year(self, save=False, plot=False):
        '''Transforms data from a linear data set to a dictionary.
        The dictionary keys are the calendar years for the original data.
        The values of the dictionary is the trade history for the stock in that
        specific year.
        Args:
            save(bool): if True, saves return value to file as csv
            plot(bool): if True, will plot the data
        Returns:
            request(defaultdict):
                Keys: years.
                Values: trade history for given year.
        '''
        logging.info('group_by_year() called')
        request = defaultdict(list)
        for date, price in zip(self.dates, self.prices):
            request[str(date.year)].append(price)
        request = self._clean_length(request)
        if save:
            self._save_group_by_year(request)
        if plot:
            self.plot_group_by_year()
        # request type default dict
        return request

    def daily_pct_change(self, save=False):
        """Will take self.group_by_year() and convert to pandas.DataFrame
        It then creates a new DataFrame containing daily pct change.
        args: save(bool): True: save return value to disk as csv.
        returns: pandas.DataFrame
        """
        logging.info('daily_pct_change() called')
        byYear = pd.DataFrame(self.group_by_year())
        dailyPctChange = pd.DataFrame()
        for year in byYear:
            # Here I use numpy to calculate the pct change
            dailyPctChange[year] = round(np.log(byYear[year]).diff()*100, 3)
        dailyPctChange = dailyPctChange[1:]
        if save:
            dailyPctChange.to_csv(f'Daily_Pct_Change_{self.ticker}.csv')
        return dailyPctChange

    def roll_pct_change(self, period, save=False, plot=False):
        """Takes self.group_by_year() and converts it to a pandas DataFrame.
        returns a pandas dataframe where each element is the pct change for the
        n period."""
        logging.info('roll_pct_change() called')
        localData = pd.DataFrame(self.group_by_year())
        rollPctChange = round(localData.pct_change(periods=period)*100, 3)
        rollPctChange.dropna(inplace=True)
        if save:
            rollPctChange.to_csv(f'Roll_Pct_Change_{self.ticker}_{period}.csv')
        if plot:
            self.plot_roll_pct_change(data=rollPctChange, period=period)
        return rollPctChange

    def corr_matrix_daily(self, save=False, plot=False):
        """Takes self.daily_pct_change() and creates a correlation matrix
        using pd.DataFrame.corr()"""
        logging.info('corr_matrix_daily() called')
        localData = self.daily_pct_change()
        corrLocalData = localData.corr()
        if save:
            corrLocalData.to_csv(f'Corr_By_Year_{self.ticker}.csv')
        if plot:
            self.plot_corr_matrix(corrLocalData)

    def corr_matrix_roll(self, period, save=False):
        '''Creates rolling correlation matrix. The is the primary purpose of this code.
        Args:
            period(int): Rolling period to be used for analysis.
            save(bool): optional arg. If True, return data is saved to file as csv.
        Returns:
            corrLocalData(pandas.MultiIndex.DataFrame): outside index is the n trade date
            of the year. inside index values are the calendar years.
            columns are the calendar years as well.
            n=each trade date in the year
            There are n correlation matrices produced by this func.
        '''
        logging.info('corr_matrix_roll() called')
        localData = self.daily_pct_change()
        corrLocalData = localData.rolling(period).corr()
        corrLocalData.dropna(inplace=True)
        if save:
            corrLocalData.to_csv(f'Rolling_Correlation_{self.ticker}_{period}.csv')
        return corrLocalData

    def high_corr_filter(self, corrData, corrN, save=False, plot=False):
        """Filters a correlation matrix based on a minimum level of correlation.

        args:
            corrData(pandas.DataFrame), with a multi level index
            corrN(float): the correlation you want to filter by. Will apply the
            positive and negative value of your number.

        returns(dict):
            Keys: day(n trade day of the year)
            value(tuple): (year1, year2, correlation value)
              """
        logging.info('high_corr_filter() called')
        filter = (((corrData >= corrN) | (corrData <= -corrN)) & (corrData < 0.99))
        filteredCorrData = corrData[filter]
        request = defaultdict(list)
        # replace nan values with 0.0, allows if statement below.
        filteredCorrData.fillna(0.0, inplace=True)
        for index, row in filteredCorrData.iterrows():
            # index = tuple(day, year)
            for year, corr in row.items():
                #year = int(year)
                # corr=float(corr)
                if corr != 0.0:
                    value = []
                    value.append((index[1], year, round(corr, 3)))
                    request[str(index[0])].append(value)
        if save:
            self._save_high_corr_filter(request)
        if plot:
            self.plot_roll_corr_filter(request)
        return request

    def plot_price_history(self, x=None, y=None):
        '''Plots the stocks price history'''
        logging.info('plot_price_hisotry called')
        if x or y:
            xData = x
            yData = y
        else:
            xData = self.dates
            yData = self.prices
        fig, ax = plt.subplots()
        ax.plot(xData, yData)
        ax.set_title(f'{self.ticker} Price History')
        fig.autofmt_xdate()
        plt.show()

    def plot_roll_pct_change(self, period=None, data=False):
        '''Plots the rolling pct change via overlapped bar charts.
        This is one plot that needs some cleaning but I ran out of time.
        I think i needed to add some alpha modification so that bars became
        very dark if there were a lot of years where the given trade date had
        simliar pct returns.'''
        logging.info('plot_roll_pct_change() called')
        if type(data)==type(pd.DataFrame()):
            localData = data
            labels = list(range(len(localData.index)))
            x = np.arange(len(labels))
            width = 0.25
            localData.plot.bar(stacked=True,
                               title=f'Rolling Pct Change of {self.ticker}')
            plt.show()
        else:
            localData = self.roll_pct_change(period=period)
            labels = list(range(len(localData.index)))
            x = np.arange(len(labels))
            width = 0.25
            localData.plot.bar(stacked=True)
            plt.show()

    # This func is for my future reference.
    def alt_plot_roll_pct(self):
        for year in localData:
            year1 = str(int(year))
            year2 = str(int(year)+1)
            year3 = str(int(year)+2)

            y1 = localData[year1]
            y2 = localData[year2]
            y3 = localData[year3]
            fig, ax = plt.subplots()
            ax.bar(x-width/2, y1, width, label=year1)
            ax.bar(x-width/2, y2, width, label=year2)
            ax.bar(x-width/2, y3, width, label=year3)
            ax.legend()
            fig.autofmt_xdate()
            fig.tight_layout()
            plt.show()
        return None

    def plot_group_by_year(self):
        '''
        Cycling chart that plots 3 years at at time.
        purpose is to allow the observer to compare prices year by year.
        '''
        logging.info('plot_group_by_year() called')
        localData = self.group_by_year()
        try:
            count = 0
            for year in localData:
                count += 1
                year1 = str(int(year))
                year2 = str(int(year) + 1)
                year3 = str(int(year) + 2)
                x1 = list(range(len(localData[year1])))
                y1 = localData[year1]
                x2 = list(range(len(localData[year2])))
                y2 = localData[year2]
                x3 = list(range(len(localData[year3])))
                y3 = localData[year3]
                fig, axes = plt.subplots(3, 1)
                fig.suptitle(f'Compare {self.ticker} price year by year.'
                             f'\n{count}/{len(self.years)-1}')
                axes[0].set(ylabel=year3)
                axes[1].set(ylabel=year2)
                axes[2].set(ylabel=year1)
                axes[0].plot(x3, y3)
                axes[1].plot(x2, y2)
                axes[2].plot(x1, y1)
                plt.show()
        except:
            print('End of by year charts')

    def plot_corr_matrix(self, corrData):
        '''Plots correlation matrix'''
        logging.info('plot_corr_matrix() called')
        corrData = corrData
        corrArray = corrData.to_numpy()
        indexList = list(corrData.index)
        #print(indexList)
        #print(corrArray)
        fig, ax = plt.subplots()
        im = ax.imshow(corrArray)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('key', rotation=-90, va='bottom')
        ax.set_xticks(np.arange(len(indexList)))
        ax.set_yticks(np.arange(len(indexList)))
        ax.set_xticklabels(indexList)
        ax.set_yticklabels(indexList)

        for i in range(len(indexList)):
            for j in range(len(indexList)):
                text = ax.text(j, i, round(corrArray[i, j], 2),
                               ha='center', va='center', color='w')

        ax.set_title('Correlation Matrix')
        fig.tight_layout()
        plt.show()

    def plot_roll_corr_filter(self, corrResults):
        '''
        Plots the final and most import chart/plot of this program.
        Will show how many occurences of high correlation occurred in the data set.
        In my opinion, I would want to see about 75/100 occurrences in a given ticker.

        Args:
            corrResults(high_corr_filter()): This is to be the data produced by the
                high_corr_filter() func.
        Returns:
            None
            If called plots the results.
        '''
        logging.info('plot_roll_corr_filter() called')
        x = list(corrResults.keys())
        y = [int(len(v)/2) for v in corrResults.values()]
        outOf = ((len(self.group_by_year())**2)//2)-len(self.group_by_year())
        plt.bar(x, y)
        plt.title(f'Occurrences of minimum correlation by \n\
         trade day out of a possible {outOf}')
        plt.ylabel('Number of occurrences')
        plt.xlabel('Trade Day of the Year')
        plt.yticks(y)
        plt.show()

def main_exec(ticker, period=None, corrN=None, save=False, plot=False):
    '''
    Called with ExecSummary is passed through the command line.
    Program logic.
    '''
    logging.info('ExecSum() called')
    if period:
        rollWindow = period
        corrN = corrN
    else:
        rollWindow = 20
        corrN = 0.7
    logging.info('main_exec() called')
    data = StockAnalysis(ticker)
    if plot:
        data.plot_price_history()
    data.group_by_year(save=save, plot=plot)
    data.daily_pct_change(save=save)
    data.roll_pct_change(rollWindow, save=save, plot=plot)
    data.corr_matrix_daily(save=save, plot=plot)
    corrData = data.corr_matrix_roll(rollWindow, save=save)
    results = data.high_corr_filter(corrData, corrN, save=save)
    data.plot_roll_corr_filter(results)

def walkthrough(ticker,period=None, corrN=None, save=False, plot=True):
    '''Called when WalkThrough is passed through the command line. Program logic.'''
    logging.info('walkthrough() called')
    if period:
        rollWindow = period
        corrN = corrN
    else:
        rollWindow = 20
        corrN = 0.7
    logging.info('walkthrough() called')
    data = StockAnalysis(ticker)
    data.plot_price_history()
    data.group_by_year(save=save, plot=True)
    data.daily_pct_change(save=save)
    data.roll_pct_change(rollWindow, save=save, plot=True)
    data.corr_matrix_daily(save=save, plot=True)
    corrData = data.corr_matrix_roll(period=rollWindow, save=save)
    data.high_corr_filter(corrData, corrN, save=save, plot=True)


def main():
    create_log(debug=True)
    logging.info('main() called')
    # Parser creation and type checking
    parser = Parser()
    checkParser(parser)
    ticker = parser.TICKER
    study = parser.StudyType
    window = parser.RollingWindow
    corrN = parser.Correlation_N
    create_log(parser.DEBUG)
    plot = parser.plot
    save = parser.save
    if study=='Build':
        IPython.embed()
    else:
        # dictionary that holds the main funcstions
        mainDict = {'ExecSummary': main_exec, 'WalkThrough': walkthrough}
        # Start of logic
        #print(ticker, study, window, corrN, plot, save)
        mainDict[study](ticker, period=window, corrN=corrN, save=save, plot=plot)
    print('Thank you for using my study!')


if __name__=='__main__':
    main()


























