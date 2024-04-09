import pandas as pd
import pandas_ta as ta
from tqdm import tqdm

class Strategy:
    def __init__(self, market_data, indicator_threshold, stoploss_pct, target_pct, capital=100000, leverage=5, entry_type="buy", max_trades_per_day=2, ):
        """
        Initialize the BacktestEngine object.

        Parameters:
        - market_data (DataFrame): DataFrame containing market data
        - stoploss_pct (float): Percentage for stop loss
        - target_pct (float): Percentage for target
        - capital (float): Initial capital for trading
        - leverage (int): Leverage multiplier
        - entry_type (str): Type of entry ("buy" or "sell")
        - max_trades_per_day (int): Maximum number of trades allowed per day
        """
        self.market_data = market_data
        self.capital = capital
        self.leverage = leverage
        self.entry_type = entry_type
        self.order_log = pd.DataFrame()
        self.max_trades_per_day = max_trades_per_day
        self.indicator_threshold = indicator_threshold

        # Calculate stop loss and target percentages based on entry type
        if self.entry_type == "buy":
            self.stoploss_pct = 1 - stoploss_pct
            self.target_pct = 1 + target_pct
        elif self.entry_type == "sell":
            self.stoploss_pct = 1 + stoploss_pct
            self.target_pct = 1 - target_pct
        else:
            raise ValueError("Invalid entry type")

    def preprocess_data(self):
        """Preprocess market data."""
        self.market_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        self.market_data['date'] = pd.to_datetime(self.market_data['date'])
        # Fill missing values
        self.market_data.ffill(inplace=True)
        # Remove timezone information
        self.market_data['date'] = self.market_data['date'].dt.tz_localize(None)
        #Convert to numeric format
        numeric_columns = ['open', 'high', 'low', 'close']
        self.market_data[numeric_columns] = self.market_data[numeric_columns].apply(pd.to_numeric, errors='coerce')
        # Sort data by date
        self.market_data = self.market_data.sort_values(by='date').reset_index(drop=True)

    def add_features(self):
        """Add technical indicators (e.g., RSI) to market data."""
        self.market_data['RSI'] = ta.rsi(self.market_data['close'], length=14)
        self.market_data['Prev RSI'] = self.market_data['RSI'].shift(1)

    def run_backtest(self):
        """Run backtest."""
        # Preprocess data and add features
        self.preprocess_data()
        self.add_features()

        trade_id = 1
        unique_dates = self.market_data['date'].dt.date.unique()
        trade_status = 'inactive'

        # Iterate over unique dates
        for date in tqdm(unique_dates, desc="Backtesting", ncols=150):
            filtered_data = self.market_data[self.market_data['date'].dt.date == date]
            daily_trade_count = 0

            # Iterate over filtered data for the current date
            for i, row in filtered_data.iterrows():
                # Check if the maximum trades per day limit is reached
                if daily_trade_count >= self.max_trades_per_day: continue
                current_datetime = row['date']
                
                # Check exit condition and place exit order if active
                if trade_status == 'active':
                    if self.check_exit_condition(row, trade_id) or i == filtered_data.index.max():
                        self.place_exit_order(current_datetime, trade_id)
                        trade_status = 'inactive'
                        trade_id += 1
                        daily_trade_count += 1

                # Check entry condition and place entry order if inactive
                if trade_status == 'inactive':
                    if self.check_entry_condition(row):
                        self.place_entry_order(current_datetime, trade_id)
                        trade_status = 'active'



        # Filter order log to remove incomplete trades
        self.order_log = self.order_log[self.order_log.groupby('tradeID')['tradeID'].transform('count') == 2]

        # Convert order log to trade log
        TradeLog = self.convert_to_trades()
        return TradeLog

    def check_entry_condition(self, row):
        """
        Check entry condition based on RSI.

        Parameters:
        - row (Series): Row of market data

        Returns:
        - bool: True if entry condition is met, False otherwise
        """
        rsi = row['RSI']
        prev_rsi = row['Prev RSI']
        rsi_cros_level = self.indicator_threshold
        if (rsi <= rsi_cros_level) and (prev_rsi > rsi_cros_level) :
            return True

    def place_entry_order(self, current_datetime, trade_id):
        """Place entry order."""
        entry_price = self.market_data.loc[self.market_data['date'] == current_datetime, 'close'].iloc[0]
        quantity = round(self.capital * self.leverage / entry_price)
        self.add_order(trade_id, current_datetime, entry_price, self.entry_type, quantity)

    def add_order(self, trade_id, orderTime, order_price, order_type, quantity):
        """Add order to order log."""
        new_order = {'tradeID': trade_id, 'orderTime': orderTime, 'orderType': order_type, 'Quantity': quantity,
                     'orderPrice': order_price}
        self.order_log = pd.concat([self.order_log, pd.DataFrame([new_order])], ignore_index=True)

    def check_exit_condition(self, row, trade_id):
        """
        Check exit condition based on stop loss and target.

        Parameters:
        - row (Series): Row of market data
        - trade_id (int): Trade ID

        Returns:
        - bool: True if exit condition is met, False otherwise
        """
        entry_price = self.order_log.loc[self.order_log['tradeID'] == trade_id, 'orderPrice'].iloc[0]
        stoploss_price = entry_price * self.stoploss_pct
        target_price = entry_price * self.target_pct
        current_price = row['close']
        if self.entry_type == "buy":
            return (current_price <= stoploss_price) or (current_price >= target_price)
        elif self.entry_type == "sell":
            return (current_price >= stoploss_price) or (current_price <= target_price)
        
    def place_exit_order(self, current_datetime, trade_id):
        """Place exit order."""
        order_type = "sell" if self.entry_type == "buy" else "buy"
        quantity = self.order_log.loc[self.order_log['tradeID'] == trade_id, 'Quantity'].iloc[0]
        current_price = self.market_data.loc[self.market_data['date'] == current_datetime, 'close'].iloc[0]
        self.add_order(trade_id, current_datetime, current_price, order_type, quantity)

    def convert_to_trades(self):
        """Convert order log to trade log."""
        exit_ordertype = "sell" if self.entry_type == "buy" else "buy"
        trade_log = []

        for trade_id in self.order_log['tradeID'].unique():
            trade_entry = self.order_log[(self.order_log['tradeID'] == trade_id) & (self.order_log['orderType'] == self.entry_type)]
            trade_exit = self.order_log[(self.order_log['tradeID'] == trade_id) & (self.order_log['orderType'] == exit_ordertype)]

            entry_price = trade_entry['orderPrice'].iloc[0]
            entry_time = trade_entry['orderTime'].iloc[0]
            quantity = trade_entry['Quantity'].iloc[0]
            exit_price = trade_exit['orderPrice'].iloc[0]
            exit_time = trade_exit['orderTime'].iloc[0]

            pnl = (exit_price - entry_price) * quantity if self.entry_type == "buy" else (entry_price - exit_price) * quantity
            trade_log.append((trade_id, entry_time, entry_price, exit_time, exit_price, quantity, pnl))

        trade_log_columns = ['TradeID', 'Entry Time', 'Entry Price', 'Exit Time', 'Exit Price', 'Quantity', 'Pnl']
        TradeLog = pd.DataFrame(trade_log, columns=trade_log_columns)
        return TradeLog
