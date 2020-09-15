import unittest
import stock_analysis as sa
import logging

def setLog(debug=None):
    if debug:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        fileHandler = logging.FileHandler('Test_StockAnalysis.log', 'w')
        fileHandler.setLevel(logging.DEBUG)
        logger.addHandler(fileHandler)

        shellHandler = logging.StreamHandler()
        shellHandler.setLevel(logging.INFO)

class test_StockAnalysis(unittest.TestCase):
    def test_instantiation(self):
        data = sa.StockAnalysis('spy')
        self.assertEqual(data.ticker, 'SPY')
    def test_print(self):
        data = sa.StockAnalysis('spy')
        self.assertEqual(str(data), 'Stock: SPY')

    def test_extract_years(self):
        yearsList = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
                     2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
        data = sa.StockAnalysis('spy')
        testList = data.years
        self.assertEqual(testList, yearsList)


if __name__=="__main__":
    setLog(debug=True)
    unittest.main()
